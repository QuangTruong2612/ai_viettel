"""Inference loop chính.

Đọc data/input/{N}.txt (hoặc JSON list input khác), gọi LLM + RAG, ghi
output/{N}.json.

Hỗ trợ:
- concurrency thấp (4 parallel — LM Studio thường chỉ 1 worker thực sự).
- retry per-record.
- log ra file để debug.

Cách chạy:
    # Khuyến nghị (từ project root):
    python -m src.inference --input data/input --output output --target-ctx 8192

    # Hoặc trực tiếp (script tự thêm src/ vào sys.path):
    python src/inference.py --input data/input --output output --target-ctx 8192
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

# Đảm bảo có thể chạy trực tiếp `python src/inference.py` (không chỉ `python -m src.inference`)
# bằng cách thêm thư mục cha của src/ vào sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.llm_client import LLMClient
from src.icd_rag import ICDRetriever, ICD10VectorSearch, Translator
from src.postprocess import (
    assemble_record, validate_output, write_output,
    preprocess_input_for_llm,
    _preprocess_highlight_duplicates,
    align_and_expand_entities,
)
from src.prompts import (
    SYSTEM_PROMPT,
    build_user_prompt,
    format_few_shot_messages,
    load_few_shot,
)
from src.rxnorm_rag import RxNormRetriever

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------- #
# IO helpers
# ---------------------------------------------------------------------- #


def read_input_record(path: Path) -> str:
    """Đọc 1 record đầu vào.

    Hỗ trợ 2 format:
    - Plain text: input/1.txt → trả nguyên nội dung
    - JSON wrapper: input/1.json → lấy trường "text"/"input"/"content"
    """
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, str):
            return data
        if isinstance(data, dict):
            for key in ("text", "input", "content", "note_text"):
                if key in data and isinstance(data[key], str):
                    return data[key]
            # Fallback: stringify toàn bộ
            return json.dumps(data, ensure_ascii=False)
    else:
        return path.read_text(encoding="utf-8").strip()


def list_input_files(input_dir: Path) -> list[Path]:
    """Liệt kê file input theo thứ tự index."""
    files = sorted(
        input_dir.glob("*.txt"),
        key=lambda p: (
            int(re.findall(r"\d+", p.stem)[0]) if re.findall(r"\d+", p.stem) else 0
        ),
    )
    files += sorted(
        input_dir.glob("*.json"),
        key=lambda p: (
            int(re.findall(r"\d+", p.stem)[0]) if re.findall(r"\d+", p.stem) else 0
        ),
    )
    return files


# ---------------------------------------------------------------------- #
# LLM call wrapper with retry on JSON parse error
# ---------------------------------------------------------------------- #

_CURRENT_REC_ID: list[int] = [0]  # mutable closure for logging
_LAST_RAW_RESPONSE: str = ""  # stash raw LLM content for debugging


def _reset_per_record_state() -> None:
    """Reset module-level state TRƯỚC mỗi record để đảm bảo context cô lập.

    Mỗi input phải xử lý ĐỘC LẬP — không tích lũy state từ record trước:
      - Xoá last_raw_response (tránh leak từ record trước)
      - Reset rec_id
      - Translator cache GIỮ NGUYÊN (intentional cache hit, KHÔNG xoá)
      - rescan_cache, lookup context: fresh mỗi record (handled trong postprocess)
      - Ollama KV cache: GIỮ nguyên vì persistent connection — control
        bằng Modelfile `keep_alive` (đã note trong README).
    """
    global _LAST_RAW_RESPONSE
    _LAST_RAW_RESPONSE = ""
    _CURRENT_REC_ID[0] = 0


def _try_recover_ollama(llm: LLMClient, rec_id: int) -> None:
    """Cố gắng reset LLM client connection khi Ollama crash.

    Khi Ollama trả 500 hoặc connection refused, OpenAI client không tự
    recovery. Force reset HTTP transport bằng cách tạo client mới.
    """
    try:
        from openai import OpenAI  # type: ignore
        new_client = OpenAI(
            base_url=llm.config.base_url,
            api_key=llm.config.api_key,
            timeout=llm.config.timeout,
            max_retries=0,
        )
        llm._client = new_client
        logger.warning("[%d] Ollama client reset (HTTP transport recreated)", rec_id)
        # Give Ollama thời gian recover
        import time as _t
        _t.sleep(3)
        try:
            llm._client.models.list(timeout=5)
        except Exception as ping_exc:
            logger.warning("[%d] Ollama ping fail sau reset: %s", rec_id, ping_exc)
    except Exception as exc:
        logger.warning("[%d] Không reset được Ollama client: %s", rec_id, exc)


def _log_token_budget(
    rec_id: int,
    llm: Any,
    user_prompt: str,
    few_shot: list[dict[str, str]],
) -> None:
    """Log số tokens ước lượng trước mỗi LLM call.

    Giúp debug khi inference fail: biết ngay prompt có vượt num_ctx không.
    Không gửi tới LLM — chỉ log ở debug level.
    Dùng chars/2 cho VN (Qwen2.5 ratio thực tế) - chars/2.5 underestimate 25%.
    """
    # VN chars tokenize denser (~2 chars/token cho Qwen2.5, vs 4 chars/token heuristic)
    sys_tokens = len(SYSTEM_PROMPT) // 2
    few_shot_tokens = sum(len(m.get("content", "")) for m in few_shot) // 2
    user_prompt_tokens = len(user_prompt) // 2
    max_output = llm.config.max_tokens
    total_input = sys_tokens + few_shot_tokens + user_prompt_tokens
    total_with_output = total_input + max_output
    logger.debug(
        "[%d] Token budget (VN-ratio chars/2): sys=%d few_shot=%d user=%d "
        "total_in=%d total_io=%d (max_tokens=%d)",
        rec_id, sys_tokens, few_shot_tokens, user_prompt_tokens,
        total_input, total_with_output, max_output,
    )


def _call_with_retry(
    llm: LLMClient,
    system_prompt: str,
    user_prompt: str,
    *,
    history: list[dict[str, str]],
    max_retries: int = 1,
) -> Any:
    """Gọi LLM, parse JSON; retry 1 lần nếu parse fail (không gửi repair prompt).

    Bug history: trước kia repair prompt gây LLM trả [] thay vì extract entities.
    Fix: không sửa content sai — chỉ retry với cùng prompt. Nếu fail 2 lần → raise.

    Stash raw content vào module-level _LAST_RAW_RESPONSE để debug.
    """
    msgs: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    msgs.extend(history)
    msgs.append({"role": "user", "content": user_prompt})

    global _LAST_RAW_RESPONSE
    _LAST_RAW_RESPONSE = ""

    last_err: Optional[Exception] = None
    last_raw: str = ""
    for attempt in range(max_retries + 1):
        try:
            resp = llm._client.chat.completions.create(  # noqa: SLF001
                model=llm.config.model,
                messages=msgs,
                temperature=llm.config.temperature,
                top_p=llm.config.top_p,
                max_tokens=llm.config.max_tokens,
                timeout=llm.config.timeout,
                # Ollama-specific: truyền qua extra_body.
                # - keep_alive: "0" unload ngay → giải phóng VRAM (tốt cho 9b trên Kaggle).
                #   "5m" giữ model load 5 phút (nhanh hơn nhưng tốn VRAM).
                # - num_ctx: override Ollama context length PER-REQUEST. Default 8192 để
                #   tránh overflow khi user chưa bump Modelfile num_ctx.
                # - num_gpu: số layer GPU (default -1 = all). Giảm nếu OOM.
                extra_body={
                    "keep_alive": getattr(llm.config, "keep_alive", "0"),
                    "num_ctx": getattr(llm.config, "num_ctx", 8192),
                    "num_gpu": getattr(llm.config, "num_gpu", -1),
                },
            )
            content = resp.choices[0].message.content or ""
            last_raw = content
            _LAST_RAW_RESPONSE = content
            return llm._extract_json(content)  # noqa: SLF001
        except Exception as exc:
            last_err = exc
            logger.warning(
                "LLM call lỗi (rec-id=%d, attempt %d/%d) %s — content[:200]=%r",
                _CURRENT_REC_ID[0],
                attempt + 1,
                max_retries + 1,
                exc,
                last_raw[:200],
            )
            if attempt >= max_retries:
                break
            time.sleep(2)  # chờ 2s trước khi retry cùng prompt
    raise RuntimeError(f"LLM fail after {max_retries + 1} attempts: {last_err}")


# ---------------------------------------------------------------------- #
# Section-Based Chunking helper (Cách 1 - Exhaustive NER)
# ---------------------------------------------------------------------- #


def _split_into_sections(text: str, max_chunk_len: int = 1400) -> list[tuple[str, int]]:
    """Tách bài án dài thành các chunks theo đoạn (paragraphs) giữ nguyên absolute offset.

    Trả về danh sách tuple: (chunk_text, chunk_offset_in_original_text).
    Nếu bài án ngắn (<= max_chunk_len), trả về [(text, 0)].
    """
    if len(text) <= max_chunk_len:
        return [(text, 0)]

    lines = text.splitlines(keepends=True)
    chunks: list[tuple[str, int]] = []
    current_lines: list[str] = []
    current_len = 0
    current_offset = 0

    for line in lines:
        line_len = len(line)
        # Ưu tiên tách tại tiêu đề lớn nếu chunk hiện tại đã đủ lớn (> 400 chars)
        stripped = line.strip()
        is_header = False
        if current_len > 400 and (
            stripped.startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "I.", "II.", "III.", "IV.", "V."))
            or stripped.startswith(("Tiền sử", "Bệnh sử", "Khám", "Đánh giá", "Xét nghiệm", "Chẩn đoán", "Điều trị", "Lý do nhập viện"))
        ):
            is_header = True

        if current_lines and (current_len + line_len > max_chunk_len or is_header):
            chunk_str = "".join(current_lines)
            chunks.append((chunk_str, current_offset))
            current_offset += current_len
            current_lines = [line]
            current_len = line_len
        else:
            current_lines.append(line)
            current_len += line_len

    if current_lines:
        chunk_str = "".join(current_lines)
        if chunks and len(chunk_str) < 250:
            # Nếu chunk cuối quá ngắn (< 250 chars), gộp luôn vào chunk liền trước
            prev_str, prev_offset = chunks.pop()
            chunks.append((prev_str + chunk_str, prev_offset))
        else:
            chunks.append((chunk_str, current_offset))

    return chunks


# ---------------------------------------------------------------------- #
# Per-record handler
# ---------------------------------------------------------------------- #


def process_record(
    rec_id: int,
    input_text: str,
    llm: LLMClient,
    retriever: RxNormRetriever,
    icd_retriever: ICDRetriever,
    few_shot: list[dict[str, str]],
    output_dir: Path,
) -> None:
    """Xử lý 1 record: gọi LLM → assemble → ghi file.

    Đảm bảo context cô lập per-record:
      - Reset module-level state trước khi xử lý
      - LLM call rebuilds msgs từ đầu (stateless)
      - Translator cache giữ (intentional cache hit)
    """
    _reset_per_record_state()
    _CURRENT_REC_ID[0] = rec_id
    t0 = time.time()
    logger.info("[%d] Bắt đầu (len=%d)", rec_id, len(input_text))
    # Adaptive few-shot: cap ở 12-20 examples dựa trên test thực nghiệm.
    # Test data (user Kaggle 10/07/2026):
    #   12 few-shot → 37 entities (TỐT NHẤT - sweet spot)
    #   24 few-shot → 30-35 entities
    #   35 few-shot → <30 entities (overwhelm)
    # Quy tắc mới (KHÔNG cap dưới 12 vì sẽ giảm recall nghiêm trọng):
    #   user_prompt > 8000 chars → cap 12 (input rất dài)
    #   user_prompt > 5000 chars → cap 15 (input vừa)
    #   user_prompt <= 5000 chars → cap 20 (input ngắn, sweet spot)
    user_prompt_len = len(input_text)
    if user_prompt_len > 8000:
        adaptive_few_shot = few_shot[:12]
        logger.debug("[%d] Adaptive: keep 12 few-shot (input len=%d > 8000)",
                      rec_id, user_prompt_len)
    elif user_prompt_len > 5000:
        adaptive_few_shot = few_shot[:15]
        logger.debug("[%d] Adaptive: keep 15 few-shot (input len=%d > 5000)",
                      rec_id, user_prompt_len)
    else:
        adaptive_few_shot = few_shot[:20]
        logger.debug("[%d] Adaptive: keep 20 few-shot (input len=%d <= 5000, sweet spot)",
                      rec_id, user_prompt_len)

    # Fix #4: Log few-shot examples used (debug "which examples drove this output")
    if adaptive_few_shot:
        logger.debug(
            "[%d] Few-shot used: %d examples, first input: %s...",
            rec_id, len(adaptive_few_shot),
            adaptive_few_shot[0].get("content", "")[:80] if adaptive_few_shot else "none"
        )

    # Clean input TRƯỚC khi build_user_prompt: strip markdown, drop N/A,
    # truncate nếu quá dài. Áp dụng cho cả clean + adaptive để fit num_ctx.
    cleaned_input = preprocess_input_for_llm(input_text)
    if len(cleaned_input) != user_prompt_len:
        logger.info(
            "[%d] Input preprocessed: %d → %d chars (-%.0f%%)",
            rec_id, user_prompt_len, len(cleaned_input),
            100 * (1 - len(cleaned_input) / max(1, user_prompt_len)),
        )

    # R20.2: Highlight duplicate (đếm + mark) trước khi gửi LLM
    # LLM 7B không tự đếm được, cần pre-process để không miss duplicate
    highlighted_input = _preprocess_highlight_duplicates(cleaned_input)
    if len(highlighted_input) != len(cleaned_input):
        logger.info(
            "[%d] Input highlighted duplicate: %d → %d chars (+%d)",
            rec_id, len(cleaned_input), len(highlighted_input),
            len(highlighted_input) - len(cleaned_input),
        )

    # Section-Based Chunking (Cách 1): nếu bài án dài (> 1400 chars), tách thành các chunks theo đoạn
    # để LLM càn quét kiệt để từng đoạn nhỏ, triệt tiêu hiện tượng mỏi (fatigue) bỏ sót entities ở cuối.
    chunks = _split_into_sections(highlighted_input, max_chunk_len=1400)
    if len(chunks) > 1:
        logger.info(
            "[%d] Section-Based Chunking: Input %d chars → tách %d chunks để NER kiệt để",
            rec_id, len(highlighted_input), len(chunks),
        )

    raw: list[dict[str, Any]] = []
    for chunk_idx, (chunk_text, chunk_offset) in enumerate(chunks):
        if not chunk_text.strip():
            continue
        chunk_prompt = build_user_prompt(chunk_text)
        _log_token_budget(rec_id, llm, chunk_prompt, adaptive_few_shot)
        try:
            chunk_raw = _call_with_retry(
                llm,
                SYSTEM_PROMPT,
                chunk_prompt,
                history=adaptive_few_shot,
            )
            if not isinstance(chunk_raw, list):
                if isinstance(chunk_raw, dict):
                    for key in ("entities", "results", "data"):
                        if key in chunk_raw and isinstance(chunk_raw[key], list):
                            chunk_raw = chunk_raw[key]
                            break
                if not isinstance(chunk_raw, list):
                    chunk_raw = []

            # Điều chỉnh position [start, end] tương đối trong chunk về offset tuyệt đối trong highlighted_input
            for ent in chunk_raw:
                if isinstance(ent, dict) and "position" in ent and isinstance(ent["position"], list) and len(ent["position"]) == 2:
                    try:
                        s_rel, e_rel = int(ent["position"][0]), int(ent["position"][1])
                        ent["position"] = [s_rel + chunk_offset, e_rel + chunk_offset]
                    except (ValueError, TypeError):
                        pass
                raw.append(ent)

            if len(chunks) > 1:
                logger.debug(
                    "[%d] Chunk %d/%d (offset %d, len %d) → %d entities",
                    rec_id, chunk_idx + 1, len(chunks), chunk_offset, len(chunk_text), len(chunk_raw),
                )
        except Exception as exc:
            logger.error("[%d] LLM fail chunk %d/%d: %s", rec_id, chunk_idx + 1, len(chunks), exc)
            if len(chunks) == 1:
                # Nếu chỉ có 1 chunk và fail → ghi debug và return rỗng
                debug_path = output_dir / f"{rec_id}.debug.txt"
                try:
                    with debug_path.open("w", encoding="utf-8") as f:
                        f.write(f"RECORD {rec_id}\nINPUT:\n{input_text}\n\nRAW RESPONSE:\n{_LAST_RAW_RESPONSE}\n")
                except Exception:
                    pass
                write_output(output_dir / f"{rec_id}.json", [])
                return

    # Debug: nếu LLM trả [] thì log raw response để chẩn đoán
    if not raw:
        logger.warning(
            "[%d] LLM trả list rỗng. Raw response [:500]: %r",
            rec_id,
            _LAST_RAW_RESPONSE[:500],
        )

    # 2-STEP ARCHITECTURE:
    # Bước 2 — Python Universal Alignment & Duplicate Expansion
    # LLM đã trả text+type+assertions (không cần position chính xác).
    # align_and_expand_entities sẽ:
    #   - Tìm TẤT CẢ occurrences của mỗi text trên input_text GỐC (không phải chunk)
    #   - Tự động tính character offset [start, end] chính xác 100%
    #   - Không bao giờ miss duplicates
    # assemble_record nhận pre-aligned entities để chỉ cần RAG lookup + emit.
    pre_aligned = align_and_expand_entities(input_text, raw)
    logger.info(
        "[%d] 2-Step Alignment: %d raw entities → %d aligned entities (sau dedup)",
        rec_id, len(raw), len(pre_aligned),
    )
    final = assemble_record(
        input_text, pre_aligned, retriever, icd_retriever=icd_retriever, llm_client=llm
    )
    if not validate_output(final):
        logger.warning("[%d] Output fail schema validate", rec_id)

    out_path = output_dir / f"{rec_id}.json"
    write_output(out_path, final)
    n = len(final)
    elapsed = time.time() - t0
    logger.info("[%d] Xong: %d entities (%.1fs)", rec_id, n, elapsed)


# ---------------------------------------------------------------------- #
# Driver
# ---------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AI Race 2026 inference")
    parser.add_argument("--input", type=Path, default=Path("data/input"))
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--base-url",
        type=str,
        default="",
        help="Override LLM base URL (mặc định từ env OLLAMA_BASE_URL hoặc LMSTUDIO_BASE_URL)",
    )
    parser.add_argument("--model", type=str, default="", help="Override model name")
    parser.add_argument("--limit", type=int, default=0, help="0 = không giới hạn")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Xử lý lại tất cả records (mặc định: skip records đã có output). "
             "Dùng khi muốn re-run inference sau khi fix code.",
    )
    parser.add_argument(
        "--max-few-shot",
        type=int,
        default=35,
        help="Số few-shot examples TỐI ĐA (default 35 = tổng số examples hiện có). "
             "Cap runtime 12-20 examples (sweet spot 12). "
             "Test thực nghiệm: 12→37, 24→30-35, 35→<30 entities.",
    )
    
    parser.add_argument(
        "--target-ctx",
        type=int,
        default=65536,
        help="Context length Ollama (default 65536 = nhiều few-shot hơn). Nếu budget âm → ép 1 few-shot.",
    )
    parser.add_argument("--log-file", type=Path, default=Path("predictions.log"))
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(args.log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    if not args.input.exists():
        # Nếu path tương đối, thử resolve từ project root (parent của src/)
        # → user chạy từ bất kỳ đâu vẫn work
        if not args.input.is_absolute():
            try:
                project_root = Path(__file__).resolve().parents[1]
                candidate = project_root / args.input
                if candidate.exists():
                    args.input = candidate
                    logger.info("Resolved relative input path → %s", args.input)
                else:
                    logger.error(
                        "Input dir không tồn tại: %s "
                        "(cũng đã thử resolve từ project root: %s)",
                        args.input, candidate,
                    )
                    return 2
            except Exception:
                logger.error("Input dir không tồn tại: %s", args.input)
                return 2
        else:
            logger.error("Input dir không tồn tại: %s", args.input)
            return 2

    args.output.mkdir(parents=True, exist_ok=True)

    # Health check: đảm bảo server sống và model đang load
    # Hỗ trợ cả LM Studio (1234) và Ollama (11434) — auto-detect qua env.
    llm = LLMClient()
    # Allow CLI override of base_url/model
    if args.base_url:
        llm.config.base_url = args.base_url
    if args.model:
        llm.config.model = args.model
    try:
        models = llm._client.models.list(timeout=10)  # noqa: SLF001
        loaded = [m.id for m in models.data]
        logger.info("Models loaded từ %s: %s", llm.config.base_url, loaded)
        if llm.config.model not in loaded:
            logger.warning(
                "Model '%s' CHƯA load trong LM Studio. Có: %s. "
                "Có thể inference sẽ 500. Hãy load model trước.",
                llm.config.model,
                loaded,
            )
    except Exception as exc:
        logger.error(
            "Không kết nối được LM Studio ở %s: %s\n"
            "   → Mở LM Studio → Developer → Start Server (port 1234)",
            llm.config.base_url,
            exc,
        )
        return 3

    translator = Translator(llm_client=llm)
    # === ENHANCED ICD EXACT MATCH (mới 2026-07-09) ===
    # Thêm 30+ common VN terms vào exact match dict để L1 match ngay từ đầu.
    # Tránh vector search noise (BGE-M3 có thể match sai như "ung thư phổi" → "A30 Cholera").
    translator.preset({
        # === Oncology (C00-D48) ===
        "ung thư phổi": "lung cancer",
        "ung thư phổi không tế bào nhỏ": "non-small cell lung cancer",
        "ung thư phổi tế bào nhỏ": "small cell lung cancer",
        "u ác tính phổi": "lung cancer",  # BYT VN dùng "U ác tính" thay vì "Ung thư"
        "k phổi": "lung cancer",  # K = ung thư (viết tắt)
        "ung thư vú": "breast cancer",
        "ung thư gan": "liver cancer",
        "ung thư dạ dày": "stomach cancer",
        "ung thư đại tràng": "colon cancer",
        "ung thư trực tràng": "rectal cancer",
        "ung thư buồng trứng": "ovarian cancer",
        "ung thư cổ tử cung": "cervical cancer",
        "ung thư tuyến tiền liệt": "prostate cancer",
        "ung thư bàng quang": "bladder cancer",
        "ung thư não": "brain cancer",
        "u não": "brain tumor",
        "u ác tính não": "brain cancer",
        "di căn não": "secondary brain cancer",
        "u ác tính thứ phát ở não": "secondary brain cancer",
        "di căn xương": "secondary bone cancer",
        "di căn gan": "secondary liver cancer",
        "di căn phổi": "secondary lung cancer",
        "di căn": "secondary malignant neoplasm",
        # === Cardiovascular (I00-I99) ===
        "tăng huyết áp": "essential hypertension",
        "tăng huyết áp vô căn": "essential hypertension",
        "tăng huyết áp thứ phát": "secondary hypertension",
        "cao huyết áp": "essential hypertension",
        "nhồi máu cơ tim": "acute myocardial infarction",
        "nhồi máu cơ tim cấp": "acute myocardial infarction",
        "đau thắt ngực": "angina pectoris",
        "đau thắt ngực ổn định": "stable angina",
        "đau thắt ngực không ổn định": "unstable angina",
        "suy tim": "heart failure",
        "suy tim sung huyết": "congestive heart failure",
        "suy tim tâm thu": "systolic heart failure",
        "suy tim tâm trương": "diastolic heart failure",
        "rung nhĩ": "atrial fibrillation",
        "cuồng nhĩ": "atrial flutter",
        "ngoại tâm thu thất": "premature ventricular contraction",
        "ngoại tâm thu nhĩ": "premature atrial contraction",
        "nhịp xoang": "sinus rhythm",
        "rung thất": "ventricular fibrillation",
        "block nhĩ thất": "atrioventricular block",
        "tắc mạch": "thrombosis",
        "tắc mạch huyết khối": "thrombosis",
        "huyết khối": "thrombosis",
        "thuyên tắc phổi": "pulmonary embolism",
        "sa van hai lá": "mitral valve prolapse",
        "sa van 2 lá": "mitral valve prolapse",
        "sa van mitral": "mitral valve prolapse",
        "hở van hai lá": "mitral valve insufficiency",
        "hở van 2 lá": "mitral valve insufficiency",
        "hẹp van hai lá": "mitral valve stenosis",
        "sa van động mạch chủ": "aortic valve prolapse",
        # === Respiratory (J00-J99) ===
        "viêm phổi": "pneumonia",
        "hen phế quản": "asthma",
        "hen suyễn": "asthma",
        "bệnh phổi tắc nghẽn mạn tính": "copd",
        "copd": "copd",
        "viêm phế quản": "bronchitis",
        "viêm phế quản cấp": "acute bronchitis",
        "viêm phế quản mạn": "chronic bronchitis",
        "tràn khí màng phổi": "pneumothorax",
        "tràn dịch màng phổi": "pleural effusion",
        # === Digestive (K00-K95) ===
        "viêm dạ dày": "gastritis",
        "loét dạ dày": "gastric ulcer",
        "loét tá tràng": "duodenal ulcer",
        "trào ngược dạ dày thực quản": "gerd",
        "gerd": "gerd",
        "viêm đại tràng": "colitis",
        "viêm ruột thừa": "appendicitis",
        "xơ gan": "liver cirrhosis",
        "viêm gan b": "hepatitis b",
        "viêm gan c": "hepatitis c",
        "viêm gan": "hepatitis",
        "sỏi mật": "gallstones",
        "viêm tụy": "pancreatitis",
        "thoát vị": "hernia",
        "thoát vị đĩa đệm": "disc herniation",
        "trĩ": "hemorrhoids",
        # === Endocrine (E00-E90) ===
        "đái tháo đường": "diabetes mellitus",
        "đái tháo đường type 2": "type 2 diabetes mellitus",
        "đái tháo đường type 1": "type 1 diabetes mellitus",
        "đái tháo đường tuýp 2": "type 2 diabetes mellitus",
        "đái tháo đường tuýp 1": "type 1 diabetes mellitus",
        "suy giáp": "hypothyroidism",
        "cường giáp": "hyperthyroidism",
        "basedow": "graves disease",
        "suy thượng thận": "adrenal insufficiency",
        "hội chứng cushing": "cushing syndrome",
        # === Renal/Urinary (N00-N99) ===
        "suy thận": "renal failure",
        "suy thận cấp": "acute renal failure",
        "suy thận mạn": "chronic kidney disease",
        "sỏi thận": "kidney stones",
        "viêm bàng quang": "cystitis",
        "viêm đường tiết niệu": "urinary tract infection",
        "nhiễm trùng tiết niệu": "urinary tract infection",
        "viêm thận": "nephritis",
        "hội chứng thận hư": "nephrotic syndrome",
        # === Neurology (G00-G99) ===
        "đột quỵ": "stroke",
        "tai biến mạch máu não": "stroke",
        "nhồi máu não": "cerebral infarction",
        "xuất huyết não": "cerebral hemorrhage",
        "động kinh": "epilepsy",
        "parkinson": "parkinsons disease",
        "alzheimer": "alzheimers disease",
        "đau nửa đầu": "migraine",
        "đau đầu": "headache",
        "viêm màng não": "meningitis",
        # === Musculoskeletal (M00-M99) ===
        "thoái hóa khớp": "osteoarthritis",
        "thoái hóa cột sống": "spinal osteoarthritis",
        "viêm khớp": "arthritis",
        "viêm khớp dạng thấp": "rheumatoid arthritis",
        "gout": "gout",
        "loãng xương": "osteoporosis",
        "thoát vị đĩa đệm cổ": "cervical disc herniation",
        "thoát vị đĩa đệm thắt lưng": "lumbar disc herniation",
        # === Blood (D50-D89) ===
        "thiếu máu": "anemia",
        "thiếu máu thiếu sắt": "iron deficiency anemia",
        "xuất huyết giảm tiểu cầu": "thrombocytopenia",
        "leukemia": "leukemia",
        "lymphoma": "lymphoma",
        # === Skin (L00-L99) ===
        "viêm da": "dermatitis",
        "viêm da cơ địa": "atopic dermatitis",
        "vẩy nến": "psoriasis",
        "eczema": "eczema",
        "viêm tuyến mồ hôi mủ": "hidradenitis suppurativa",
        "nhọt ổ gà": "hidradenitis suppurativa",
    })
    retriever = RxNormRetriever()
    # VectorSearch: BGE-M3 vector search trên toàn bộ 71k mã ICD-10
    local_search = ICD10VectorSearch()
    icd_retriever = ICDRetriever(
        translator=translator, local_search=local_search
    )
    # Adaptive few-shot cap based on context budget.
    # Tính số few-shot vừa đủ dựa trên: target_ctx - sys_prompt - max_tokens - user_input.
    # Lưu ý: chars/4 heuristic underestimate ~60% cho VN. Dùng chars/2.5 (Qwen2.5 ratio)
    # để budget chính xác hơn.
    all_examples = load_few_shot()

    # Estimate budget (target_ctx = Ollama Context Length, do user truyền)
    # Dùng chars/2 cho VN text (chars/2.5 underestimate khiến budget âm → few_shot=0)
    sys_tokens_real = int(len(SYSTEM_PROMPT) / 2)
    max_output_tokens = llm.config.max_tokens  # 6144 sau khi tăng
    reserve_for_safety = 512  # buffer cho input + few-shot overhead
    budget_for_input = args.target_ctx - max_output_tokens - reserve_for_safety
    budget_for_sys_fewshot = budget_for_input
    remaining_for_fewshot = budget_for_sys_fewshot - sys_tokens_real
    # Mỗi few-shot (user + assistant) ≈ 600 chars input mỗi (~300 tokens với chars/2)
    if remaining_for_fewshot < 0:
        # Budget âm: SYSTEM_PROMPT quá dài so với target_ctx.
        # Ép dùng TỐI THIỂU 1 few-shot để có pattern (dù thiếu budget).
        auto_few_shot = 1
        logger.warning(
            "Context budget ÂM (sys=%d > budget_in=%d). Ép dùng 1 few-shot.",
            sys_tokens_real, budget_for_input,
        )
    else:
        # Each example ~ 300 real tokens (chars/2)
        auto_few_shot = max(
            0, min(remaining_for_fewshot // 300, len(all_examples), args.max_few_shot)
        )
        # Ít nhất 1 example để có diversity, nhiều nhất theo budget
        auto_few_shot = max(0, min(auto_few_shot, args.max_few_shot))

    few_shot = format_few_shot_messages(all_examples[:auto_few_shot])
    # Dùng real token estimate cho log
    sys_tokens_for_log = sys_tokens_real
    logger.info(
        "Context budget: target=%d sys=%d(real) max_out=%d budget_in=%d → few_shot=%d/%d",
        args.target_ctx,
        sys_tokens_for_log,
        max_output_tokens,
        budget_for_input,
        len(few_shot) // 2 if few_shot else 0,
        len(all_examples),
    )
    if few_shot:
        fs_chars = sum(len(m["content"]) for m in few_shot)
        logger.info(
            "Few-shot selected: %d msgs, ~%d input tokens from %d total tokens budget",
            len(few_shot),
            fs_chars // 4,
            budget_for_sys_fewshot // 4,
        )

    files = list_input_files(args.input)
    if args.limit:
        files = files[: args.limit]
    # Skip files đã được xử lý (resume sau khi Colab kill).
    # Default: skip nếu output file đã có. Dùng --no-resume để xử lý lại từ đầu.
    if not args.no_resume:
        before = len(files)
        files = [f for f in files
                 if not (args.output / f"{int(re.findall(r'\d+', f.stem)[0])}.json").exists()]
        skipped = before - len(files)
        if skipped:
            logger.info(
                "Resume: skipped %d records đã có output (chạy lại từ %d còn lại). "
                "Dùng --no-resume nếu muốn re-run toàn bộ.",
                skipped, len(files),
            )
    logger.info("Tìm thấy %d record", len(files))

    if args.workers <= 1:
        for f in files:
            rec_id = int(re.findall(r"\d+", f.stem)[0])
            try:
                text = read_input_record(f)
            except Exception as exc:
                logger.error("[%d] Không đọc được: %s → skip file này, continue", rec_id, exc)
                # rec_id calculation ở ngoài try nên có thể fail nếu filename không có số
                if 'rec_id' in dir() and rec_id is not None:
                    try:
                        write_output(args.output / f"{rec_id}.json", [])
                    except Exception:
                        pass
                continue

            # ===== WRAPPER ROBUST: catch MỌI exception trong process_record =====
            # Lý do: Ollama có thể crash mid-batch (memory, timeout, etc.).
            # Bất kỳ exception nào KHÔNG được catch trong process_record sẽ
            # làm dừng toàn bộ loop. Đảm bảo mỗi record luôn kết thúc sạch.
            try:
                process_record(
                    rec_id, text, llm, retriever, icd_retriever, few_shot, args.output
                )
            except SystemExit as se:
                # SystemExit thường từ Ollama client hoặc assertFail.
                logger.error("[%d] SystemExit (Ollama có thể đã crash): %s",
                             rec_id, se)
                _try_recover_ollama(llm, rec_id)
                write_output(args.output / f"{rec_id}.json", [])
            except KeyboardInterrupt:
                raise  # user Ctrl+C → propagate
            except BaseException as exc:  # bao gồm cả non-Exception (SystemExit)
                logger.exception(
                    "[%d] CRASH trong process_record (BÍ ẨN!): %s → ghi [] và continue",
                    rec_id, exc,
                )
                _try_recover_ollama(llm, rec_id)
                try:
                    write_output(args.output / f"{rec_id}.json", [])
                except Exception as write_exc:
                    logger.error("[%d] Không ghi được []: %s", rec_id, write_exc)
    else:
        with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = []
            for f in files:
                rec_id = int(re.findall(r"\d+", f.stem)[0])
                try:
                    text = read_input_record(f)
                except Exception as exc:
                    logger.error("[%d] Không đọc được: %s", rec_id, exc)
                    continue
                futures.append(
                    pool.submit(
                        process_record,
                        rec_id,
                        text,
                        llm,
                        retriever,
                        icd_retriever,
                        few_shot,
                        args.output,
                    )
                )
            for fut in cf.as_completed(futures):
                try:
                    fut.result()
                except SystemExit as se:
                    logger.error("Future SystemExit (Ollama có thể crashed): %s", se)
                    _try_recover_ollama(llm, -1)
                except BaseException as exc:  # noqa: BLE001
                    logger.exception("Future CRASH: %s → continue", exc)
                    _try_recover_ollama(llm, -1)
    # Save caches
    translator.save_cache()
    icd_retriever.save_index()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
