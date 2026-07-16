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
import os
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
from src.icd_rag import ICDRetriever, ICD10VectorSearch
from src.postprocess import (
    assemble_record, validate_output, write_output,
    preprocess_input_for_llm,
    _preprocess_highlight_duplicates,
    align_and_expand_entities,
    _validate_stage1_mentions,
    _refine_stage2_results,
    _stage2_fallback_classify,
    _validate_candidates_for_type,
)
from src.prompts import (
    SYSTEM_PROMPT,
    build_user_prompt,
    format_few_shot_messages,
    load_few_shot,
    select_dynamic_few_shot,
    STAGE1_PROMPT,
    STAGE2_PROMPT,
    build_stage1_user_prompt,
    build_stage2_user_prompt,
    format_few_shot_stage2_messages,
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
      - lookup context: fresh mỗi record (handled trong postprocess)
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
                # - think: TẮT Qwen3 thinking mode (chỉ apply cho Qwen3+; Qwen2.5 ignore tham số này).
                #   Thiếu dòng này khiến model sinh block suy luận dài trước JSON,
                #   ăn vào max_tokens budget và làm chậm inference đáng kể.
                extra_body={
                    "keep_alive": getattr(llm.config, "keep_alive", "0"),
                    "num_ctx": getattr(llm.config, "num_ctx", 8192),
                    "num_gpu": getattr(llm.config, "num_gpu", -1),
                    "think": False,
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
            time.sleep(1)
            # Upgrade D: Self-correction loop with entity-preserving error prompt
            if last_raw and len(last_raw.strip()) > 5:
                msgs.append({"role": "assistant", "content": last_raw[:1500]})
                msgs.append({
                    "role": "user",
                    "content": f"Lỗi cú pháp JSON ({exc}). Hãy BẢO TÒN ĐẦY ĐỦ tất cả các thực thể y khoa (bệnh, thuốc, triệu chứng, xét nghiệm) bạn vừa tìm thấy ở trên và CHỈ xuất lại dưới dạng mảng JSON hợp lệ theo đúng schema (không giải thích thêm, không trả về [] nếu đã tìm thấy thực thể):"
                })
    raise RuntimeError(f"LLM fail after {max_retries + 1} attempts: {last_err}")


# ---------------------------------------------------------------------- #
# Section-Based Chunking helper (Cách 1 - Exhaustive NER)
# ---------------------------------------------------------------------- #


def _split_into_sections(text: str, max_chunk_len: int = 750, overlap_len: int = 200) -> list[tuple[str, int]]:
    """Tách bài án dài thành các chunks theo sliding window có vùng gối đầu (overlap) và giữ nguyên absolute offset.

    Trả về danh sách tuple: (chunk_text, chunk_offset_in_original_text).
    Nếu bài án ngắn (<= max_chunk_len), trả về [(text, 0)].
    """
    if len(text) <= max_chunk_len:
        return [(text, 0)]

    raw_lines = text.splitlines(keepends=True)
    lines: list[str] = []
    for rl in raw_lines:
        if len(rl) <= max_chunk_len:
            lines.append(rl)
        else:
            # Nếu 1 dòng quá dài (> max_chunk_len), tách tiếp theo dấu câu (. ! ? ;) để tránh tràn chunk
            parts = re.split(r"(?<=[.!?;])\s+", rl)
            cur = ""
            for p in parts:
                if len(cur) + len(p) + 1 > max_chunk_len and cur:
                    lines.append(cur + " ")
                    cur = p
                else:
                    cur = (cur + " " + p) if cur else p
            if cur:
                if rl.endswith("\n") and not cur.endswith("\n"):
                    cur += "\n"
                lines.append(cur)

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

        if current_lines and current_len >= 120 and (current_len + line_len > max_chunk_len or is_header):
            chunk_str = "".join(current_lines)
            chunks.append((chunk_str, current_offset))

            # Tạo vùng gối đầu (overlap window) ~overlap_len chars từ cuối current_lines
            overlap_lines: list[str] = []
            ov_len = 0
            for prev_line in reversed(current_lines):
                if ov_len + len(prev_line) > overlap_len and overlap_lines:
                    break
                overlap_lines.insert(0, prev_line)
                ov_len += len(prev_line)

            # Cập nhật offset mới: offset của chunk vừa rồi + (current_len - ov_len)
            current_offset += (current_len - ov_len)
            current_lines = overlap_lines + [line]
            current_len = ov_len + line_len
        else:
            current_lines.append(line)
            current_len += line_len

    if current_lines:
        chunk_str = "".join(current_lines)
        if chunks and len(chunk_str) < 280:
            # Nếu chunk cuối quá ngắn (< 280 chars), gộp luôn vào chunk liền trước và giữ nguyên offset
            prev_str, prev_offset = chunks.pop()
            tail_len = max(0, (prev_offset + len(prev_str)) - current_offset)
            if tail_len < len(chunk_str):
                chunks.append((prev_str + chunk_str[tail_len:], prev_offset))
            else:
                chunks.append((prev_str, prev_offset))
        else:
            chunks.append((chunk_str, current_offset))

    return chunks


# ---------------------------------------------------------------------- #
# R37 (2026-07-16): STAGE 3 — LLM context analyzer cho ICD/RxNorm candidates
# ---------------------------------------------------------------------- #

def _stage3_refine_candidates(
    rec_id: int,
    input_text: str,
    entities: list[dict[str, Any]],
    llm: LLMClient,
    batch_size: int = 30,
) -> list[dict[str, Any]]:
    """R37 (2026-07-16): Stage 3 LLM pass to verify/refine ICD/RxNorm candidates.

    Args:
        rec_id: for logging only
        input_text: full clinical note (provides context for LLM)
        entities: list of dicts (each has text/type) — to be refined in-place
        llm: LLMClient for stage 3 call
        batch_size: max entities per LLM call (default 30)

    Returns:
        The same entities list with `candidates` field updated per entity (verdict-based).

    Behavior:
        - Skip entities with type != CHẨN_ĐOÁN and type != THUỐC
        - Skip entities already empty candidates
        - Batch entities (default 30 per call)
        - For each batch: build prompt → call LLM → parse JSON → validate codes → update
        - On LLM parse failure: keep RAG candidates (fallback)
    """
    from src.prompts import STAGE3_PROMPT, build_stage3_user_prompt

    # Filter: only CHẨN_ĐOÁN + THUỐC with non-empty candidates
    target_entities = [
        (i, e) for i, e in enumerate(entities)
        if e.get("type") in ("CHẨN_ĐOÁN", "THUỐC")
        and isinstance(e.get("candidates"), list)
        and len(e.get("candidates", [])) > 0
    ]
    if not target_entities:
        logger.debug("[%d] Stage 3: skip (no CHẨN_ĐOÁN/THUỐC with candidates)", rec_id)
        return entities

    # Build batches (preserve original indices for merging back)
    entity_payloads = [
        {
            "text": e.get("text", ""),
            "type": e.get("type", ""),
            "candidates": list(e.get("candidates", [])),
        }
        for _, e in target_entities
    ]
    user_prompts = build_stage3_user_prompt(input_text, entity_payloads, batch_size=batch_size)

    # Apply LLM per batch, default = unchanged RAG candidates
    for batch_idx, user_prompt in enumerate(user_prompts):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(entity_payloads))
        batch_payloads = entity_payloads[batch_start:batch_end]

        # R37 (2026-07-16): Add retry logic for transient LLM failures
        # (network blip, timeout, rate limit). 1 retry with 2s wait.
        content = ""
        last_exc = None
        for attempt in range(2):  # 0=first try, 1=retry
            try:
                resp = llm._client.chat.completions.create(
                    model=llm.config.model,
                    messages=[
                        {"role": "system", "content": STAGE3_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.05,
                    top_p=1.0,
                    max_tokens=llm.config.max_tokens,
                    timeout=min(60, llm.config.timeout),
                    extra_body={
                        "keep_alive": getattr(llm.config, "keep_alive", "0"),
                        "num_ctx": getattr(llm.config, "num_ctx", 32768),
                        "num_gpu": getattr(llm.config, "num_gpu", -1),
                        "think": False,
                    },
                )
                content = (resp.choices[0].message.content or "").strip()
                if content:
                    break  # Got valid response, exit retry loop
            except Exception as exc:
                last_exc = exc
                if attempt == 0:
                    import time as _t
                    logger.debug(
                        "[%d] Stage 3 batch %d attempt 1 failed (%s), retrying...",
                        rec_id, batch_idx, exc,
                    )
                    _t.sleep(1.5)

        if not content:
            logger.warning(
                "[%d] Stage 3 batch %d failed (last exc: %s) — keeping RAG candidates",
                rec_id, batch_idx, last_exc,
            )
            continue

        # Parse JSON response
        parsed: list[dict] = []
        try:
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1]) if content.endswith("```") else "\n".join(lines[1:])
            parsed = json.loads(content)
            if not isinstance(parsed, list):
                parsed = []
        except Exception as exc:
            logger.warning(
                "[%d] Stage 3 LLM parse fail (batch %d): %s — keeping RAG",
                rec_id, batch_idx, exc,
            )
            continue

        # Apply results
        for j, payload in enumerate(batch_payloads):
            if j >= len(parsed):
                break
            entry = parsed[j]
            if not isinstance(entry, dict):
                continue
            verdict = entry.get("verdict", "ok")
            new_cands = entry.get("candidates", [])
            etype = payload["type"]
            valid_cands = _validate_candidates_for_type(new_cands, etype)
            if verdict == "drop":
                valid_cands = []
            original_idx = batch_start + j
            entity_payloads[original_idx] = {
                "text": payload["text"],
                "type": etype,
                "candidates": valid_cands,
            }

    # Compute summary stats (R37): refined count + dropped count
    refined_count = 0
    dropped_count = 0
    for (orig_idx, _), tp in zip(target_entities, entity_payloads):
        old_cand = entities[orig_idx].get("candidates", [])
        new_cand = tp.get("candidates", [])
        if old_cand != new_cand:
            refined_count += 1
        if old_cand and not new_cand:
            dropped_count += 1
    logger.info(
        "[%d] Stage 3: %d CD/drug entities processed (refined=%d, dropped=%d)",
        rec_id, len(target_entities), refined_count, dropped_count,
    )

    # Merge back into entities list (in-place)
    for (orig_idx, _orig_ent), new_payload in zip(target_entities, entity_payloads):
        entities[orig_idx]["candidates"] = new_payload["candidates"]

    return entities


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
    few_shot_stage2: list[dict[str, str]] | None = None,
    use_two_stage: bool = True,
) -> None:
    """Xử lý 1 record: gọi LLM (Two-Stage hoặc Single-Pass) → assemble → ghi file.

    Đảm bảo context cô lập per-record:
      - Reset module-level state trước khi xử lý
      - LLM call rebuilds msgs từ đầu (stateless)
    """
    _reset_per_record_state()
    _CURRENT_REC_ID[0] = rec_id
    t0 = time.time()
    logger.info("[%d] Bắt đầu (len=%d, two_stage=%s)", rec_id, len(input_text), use_two_stage)
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
    # Upgrade H: Domain-Adaptive Few-Shot Selection based on Keyword/Domain Overlap
    if few_shot and len(few_shot) > 4:
        input_tokens = set(re.findall(r'[a-zà-ỹ0-9_/-]{3,}', input_text.lower()))
        def _score_few_shot_item(item: dict[str, str]) -> int:
            content = item.get("content", "").lower()
            item_tokens = set(re.findall(r'[a-zà-ỹ0-9_/-]{3,}', content))
            return len(input_tokens & item_tokens)
        
        paired_few_shot = [few_shot[i:i+2] for i in range(0, len(few_shot), 2) if i+1 < len(few_shot)]
        if paired_few_shot:
            paired_few_shot.sort(key=lambda pair: -_score_few_shot_item(pair[0]))
            sorted_few_shot = [msg for pair in paired_few_shot for msg in pair]
        else:
            sorted_few_shot = sorted(few_shot, key=lambda item: -_score_few_shot_item(item))
    else:
        sorted_few_shot = few_shot

    if few_shot_stage2 and len(few_shot_stage2) > 4:
        paired_s2 = [few_shot_stage2[i:i+2] for i in range(0, len(few_shot_stage2), 2) if i+1 < len(few_shot_stage2)]
        if paired_s2:
            paired_s2.sort(key=lambda pair: -_score_few_shot_item(pair[0]))
            sorted_few_shot_stage2 = [msg for pair in paired_s2 for msg in pair]
        else:
            sorted_few_shot_stage2 = sorted(few_shot_stage2, key=lambda item: -_score_few_shot_item(item))
    else:
        sorted_few_shot_stage2 = few_shot_stage2 or []

    if user_prompt_len > 8000:
        adaptive_few_shot = sorted_few_shot[:12]
        logger.debug("[%d] Adaptive: keep 6 pairs (12 msgs) domain-ranked few-shot (input len=%d > 8000)",
                      rec_id, user_prompt_len)
    elif user_prompt_len > 5000:
        adaptive_few_shot = sorted_few_shot[:14]
        logger.debug("[%d] Adaptive: keep 7 pairs (14 msgs) domain-ranked few-shot (input len=%d > 5000)",
                      rec_id, user_prompt_len)
    else:
        adaptive_few_shot = sorted_few_shot[:20]
        logger.debug("[%d] Adaptive: keep 10 pairs (20 msgs) domain-ranked few-shot (input len=%d <= 5000, sweet spot)",
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

    # Section-Based Chunking: nếu bài án dài (> 1500 chars), tách thành các chunks theo đoạn
    # để LLM càn quét kiệt để từng đoạn nhỏ, triệt tiêu hiện tượng mỏi (fatigue) bỏ sót entities ở cuối.
    # R37 (2026-07-16): Tăng từ 750 → 1500 chars để ít chunk boundaries hơn, giảm nguy cơ
    # qualifiers (vd "Shigella dysenteriae", "thùy dưới") bị cắt giữa 2 chunks → LLM miss.
    chunks = _split_into_sections(highlighted_input, max_chunk_len=1500, overlap_len=300)
    if len(chunks) > 1:
        logger.info(
            "[%d] Section-Based Chunking: Input %d chars → tách %d chunks để NER kiệt để",
            rec_id, len(highlighted_input), len(chunks),
        )

    seen_chunk_spans: set[tuple[str, str, int, int]] = set()
    raw: list[dict[str, Any]] = []

    if use_two_stage:
        # ======================================================================
        # TWO-STAGE PIPELINE (Stage 1: Mentions + Python Validate -> Stage 2: Classify)
        # ======================================================================
        raw_stage1: list[dict[str, Any]] = []
        for chunk_idx, (chunk_text, chunk_offset) in enumerate(chunks):
            if not chunk_text.strip():
                continue
            chunk_prompt = build_stage1_user_prompt(chunk_text)
            _log_token_budget(rec_id, llm, chunk_prompt, adaptive_few_shot)
            try:
                chunk_raw = _call_with_retry(
                    llm,
                    STAGE1_PROMPT,
                    chunk_prompt,
                    history=adaptive_few_shot,
                )
                if not isinstance(chunk_raw, list):
                    if isinstance(chunk_raw, dict):
                        for key in ("entities", "results", "data", "mentions"):
                            if key in chunk_raw and isinstance(chunk_raw[key], list):
                                chunk_raw = chunk_raw[key]
                                break
                    if not isinstance(chunk_raw, list):
                        chunk_raw = []

                for ent in chunk_raw:
                    if not isinstance(ent, dict):
                        continue
                    pos = ent.get("position", [0, 0])
                    if isinstance(pos, list) and len(pos) == 2:
                        try:
                            s_rel, e_rel = int(pos[0]), int(pos[1])
                            ent["position"] = [s_rel + chunk_offset, e_rel + chunk_offset]
                        except (ValueError, TypeError):
                            pass
                    raw_stage1.append(ent)
            except Exception as exc:
                logger.error("[%d] Stage 1 fail chunk %d/%d: %s", rec_id, chunk_idx + 1, len(chunks), exc)

        # Python Layer: validation & exact boundary recovery
        validated_mentions = _validate_stage1_mentions(input_text, raw_stage1)
        logger.info(
            "[%d] Stage 1 Mentions: %d raw spans → %d validated spans",
            rec_id, len(raw_stage1), len(validated_mentions)
        )

        if not validated_mentions:
            logger.warning("[%d] Stage 1 không tìm thấy mention nào hợp lệ.", rec_id)
        else:
            # Stage 2: Classification in batches (max 35 mentions per call)
            s2_history = sorted_few_shot_stage2[:16] if sorted_few_shot_stage2 else []
            batch_size = 35
            for i in range(0, len(validated_mentions), batch_size):
                batch = validated_mentions[i : i + batch_size]
                s2_prompt = build_stage2_user_prompt(input_text, batch)
                s2_raw = []
                try:
                    s2_raw = _call_with_retry(
                        llm,
                        STAGE2_PROMPT,
                        s2_prompt,
                        history=s2_history if i == 0 else [],
                    )
                    if not isinstance(s2_raw, list):
                        if isinstance(s2_raw, dict):
                            for key in ("entities", "results", "data"):
                                if key in s2_raw and isinstance(s2_raw[key], list):
                                    s2_raw = s2_raw[key]
                                    break
                        if not isinstance(s2_raw, list):
                            s2_raw = []
                except Exception as exc:
                    logger.error("[%d] Stage 2 fail batch %d-%d: %s → kích hoạt Smart Fallback", rec_id, i, i + len(batch), exc)

                if not s2_raw:
                    logger.warning("[%d] Stage 2 batch %d-%d rỗng hoặc lỗi → dùng Rule-Based Fallback Classifier", rec_id, i, i + len(batch))
                    s2_raw = _stage2_fallback_classify(batch)

                for ent in s2_raw:
                    if isinstance(ent, dict) and ent.get("text") and ent.get("type"):
                        raw.append(ent)

            # Python Refiner: Kiểm duyệt & tự động sửa type/assertions theo luật chuyên gia
            raw = _refine_stage2_results(input_text, raw)
            logger.info("[%d] Stage 2 Refinement hoàn tất: %d entities", rec_id, len(raw))

            # R37 (2026-07-16): STAGE 3 — LLM context analysis cho ICD/RxNorm candidates
            # Apply default ON; opt-out via env LLM_DISABLE_STAGE3=1 (tương đương --no-stage3).
            if not os.environ.get("LLM_DISABLE_STAGE3", "").strip() == "1":
                raw = _stage3_refine_candidates(
                    rec_id=rec_id, input_text=input_text,
                    entities=raw, llm=llm,
                )
                logger.info("[%d] Stage 3 LLM refine hoàn tất", rec_id)
    else:
        # ======================================================================
        # SINGLE-PASS PIPELINE (Legacy mode for benchmarking via --no-two-stage)
        # ======================================================================
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
                    if not isinstance(ent, dict):
                        continue
                    if "position" in ent and isinstance(ent["position"], list) and len(ent["position"]) == 2:
                        try:
                            s_rel, e_rel = int(ent["position"][0]), int(ent["position"][1])
                            abs_s, abs_e = s_rel + chunk_offset, e_rel + chunk_offset
                            ent["position"] = [abs_s, abs_e]
                            key = (str(ent.get("type", "")), str(ent.get("text", "")).lower().strip(), abs_s, abs_e)
                            if key in seen_chunk_spans:
                                continue
                            seen_chunk_spans.add(key)
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
    parser.add_argument(
        "--no-two-stage",
        action="store_true",
        help="Chạy ở chế độ Single-Pass (extract + classify cùng 1 call) thay vì Two-Stage.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_PROJECT_ROOT / "data",
        help="Thư mục chứa data (few-shot examples, icd index)",
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

    retriever = RxNormRetriever()
    # VectorSearch: BGE-M3 vector search trên toàn bộ 71k mã ICD-10
    local_search = ICD10VectorSearch()
    icd_retriever = ICDRetriever(
        local_search=local_search
    )
    # Adaptive few-shot cap based on context budget.
    # Tính số few-shot vừa đủ dựa trên: target_ctx - sys_prompt - max_tokens - user_input.
    # Lưu ý: chars/4 heuristic underestimate ~60% cho VN. Dùng chars/2.5 (Qwen2.5 ratio)
    # để budget chính xác hơn.
    use_two_stage = not args.no_two_stage
    data_dir = getattr(args, "data_dir", _PROJECT_ROOT / "data")
    s1_path = data_dir / "examples_stage1.jsonl"
    if not s1_path.exists():
        s1_path = _PROJECT_ROOT / "data" / "examples_stage1.jsonl"
    s2_path = data_dir / "examples_stage2.jsonl"
    if not s2_path.exists():
        s2_path = _PROJECT_ROOT / "data" / "examples_stage2.jsonl"

    if use_two_stage and s1_path.exists() and s2_path.exists():
        s1_ex = load_few_shot(s1_path)
        s2_ex = load_few_shot(s2_path)
        few_shot = format_few_shot_messages(s1_ex)
        few_shot_stage2 = format_few_shot_stage2_messages(s2_ex)
        logger.info("Two-Stage Pipeline mode: loaded %d S1 and %d S2 few-shot examples.", len(s1_ex), len(s2_ex))
    else:
        all_examples = load_few_shot()
        few_shot = format_few_shot_messages(all_examples)
        few_shot_stage2 = None
        if use_two_stage:
            logger.info("Two-Stage Pipeline mode (using universal few_shot fallback from %s).", _PROJECT_ROOT / "data" / "examples.jsonl")
        else:
            logger.info("Single-Pass Pipeline mode (--no-two-stage).")

    # Dùng real token estimate cho log
    sys_tokens_for_log = int(len(SYSTEM_PROMPT) / 3)
    logger.info(
        "Context budget target=%d → few_shot msgs=%d",
        args.target_ctx,
        len(few_shot),
    )
    if few_shot:
        fs_chars = sum(len(m["content"]) for m in few_shot)
        logger.info(
            "Few-shot Stage 1 selected: %d msgs, ~%d input tokens",
            len(few_shot),
            fs_chars // 4,
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
                    rec_id, text, llm, retriever, icd_retriever, few_shot, args.output,
                    few_shot_stage2=few_shot_stage2, use_two_stage=use_two_stage,
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
                        few_shot_stage2=few_shot_stage2,
                        use_two_stage=use_two_stage,
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
    icd_retriever.save_index()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
