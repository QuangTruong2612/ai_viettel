"""Inference loop chính.

Đọc data/input/{N}.txt (hoặc JSON list input khác), gọi LLM + RAG, ghi
output/{N}.json.

Hỗ trợ:
- concurrency thấp (4 parallel — LM Studio thường chỉ 1 worker thực sự).
- retry per-record.
- log ra file để debug.
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

from .llm_client import LLMClient
from .icd_rag import ICDRetriever, ICD10VectorSearch, Translator
from .postprocess import assemble_record, validate_output, write_output
from .prompts import (
    SYSTEM_PROMPT,
    build_user_prompt,
    format_few_shot_messages,
    load_few_shot,
)
from .rxnorm_rag import RxNormRetriever

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
    """Xử lý 1 record: gọi LLM → assemble → ghi file."""
    _CURRENT_REC_ID[0] = rec_id
    t0 = time.time()
    logger.info("[%d] Bắt đầu (len=%d)", rec_id, len(input_text))
    user_prompt = build_user_prompt(input_text)
    try:
        raw = _call_with_retry(
            llm,
            SYSTEM_PROMPT,
            user_prompt,
            history=few_shot,
        )
    except Exception as exc:
        # Save debug info khi LLM fail để debug sau
        logger.error("[%d] LLM fail hết retry: %s → ghi []", rec_id, exc)
        debug_path = output_dir / f"{rec_id}.debug.txt"
        try:
            with debug_path.open("w", encoding="utf-8") as f:
                f.write(f"RECORD {rec_id}\n")
                f.write(f"INPUT (len={len(input_text)}):\n{input_text}\n\n")
                f.write(f"RAW LLM RESPONSE ({len(_LAST_RAW_RESPONSE)} chars):\n")
                f.write(_LAST_RAW_RESPONSE if _LAST_RAW_RESPONSE
                        else "(empty)")
            logger.info("[%d] Saved debug → %s", rec_id, debug_path.name)
        except Exception as write_exc:
            logger.warning("[%d] Cannot write debug file: %s", rec_id, write_exc)
        write_output(output_dir / f"{rec_id}.json", [])
        return

    if not isinstance(raw, list):
        # LLM đôi khi wrap trong object {"entities": [...]}
        if isinstance(raw, dict):
            for key in ("entities", "results", "data"):
                if key in raw and isinstance(raw[key], list):
                    raw = raw[key]
                    break
        if not isinstance(raw, list):
            logger.warning(
                "[%d] Output không phải list: %r → ghi []", rec_id, type(raw)
            )
            raw = []

    # Debug: nếu LLM trả [] thì log raw response để chẩn đoán
    if not raw:
        logger.warning(
            "[%d] LLM trả list rỗng. Raw response [:500]: %r",
            rec_id,
            _LAST_RAW_RESPONSE[:500],
        )
    final = assemble_record(
        input_text, raw, retriever, icd_retriever=icd_retriever, llm_client=llm
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
        "--max-few-shot",
        type=int,
        default=10,
        help="Số few-shot examples TỐI ĐA (default 10)",
    )
    parser.add_argument(
        "--target-ctx",
        type=int,
        default=6144,
        help="Context length của Ollama/LM Studio (default 6144 cho qwen2.5:7b). Few-shot tự cap theo budget.",
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
    retriever = RxNormRetriever()
    # VectorSearch: BGE-M3 vector search trên toàn bộ 71k mã ICD-10
    local_search = ICD10VectorSearch()
    icd_retriever = ICDRetriever(
        translator=translator, local_search=local_search
    )
    # Adaptive few-shot cap based on context budget.
    # Tính số few-shot vừa đủ dựa trên: target_ctx - sys_prompt - max_tokens - user_input.
    all_examples = load_few_shot()

    # Estimate budget (target_ctx = LM Studio Context Length, do user truyền)
    sys_tokens = len(SYSTEM_PROMPT) // 4
    max_output_tokens = llm.config.max_tokens  # 1024
    reserve_for_safety = 256  # chừa buffer cho tokenizer over-estimate
    budget_for_input = args.target_ctx - max_output_tokens - reserve_for_safety
    budget_for_sys_fewshot = budget_for_input
    remaining_for_fewshot = budget_for_sys_fewshot - sys_tokens
    # Mỗi few-shot (user + assistant) ≈ 800 chars input mỗi
    if remaining_for_fewshot < 0:
        auto_few_shot = 0  # Không vừa, skip hết few-shot
    else:
        # Each example ~ 800 chars total (user+assistant ~400 each)
        auto_few_shot = max(
            0, min(remaining_for_fewshot // 200, len(all_examples), args.max_few_shot)
        )
        # Ít nhất 2 example để có diversity, nhiều nhất theo budget
        auto_few_shot = max(0, min(auto_few_shot, args.max_few_shot))

    few_shot = format_few_shot_messages(all_examples[:auto_few_shot])
    logger.info(
        "Context budget: target=%d sys=%d max_out=%d budget_in=%d → few_shot=%d/%d",
        args.target_ctx,
        sys_tokens,
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
    logger.info("Tìm thấy %d record", len(files))

    if args.workers <= 1:
        for f in files:
            rec_id = int(re.findall(r"\d+", f.stem)[0])
            try:
                text = read_input_record(f)
                process_record(
                    rec_id, text, llm, retriever, icd_retriever, few_shot, args.output
                )
            except Exception as exc:
                logger.exception("[%d] Lỗi: %s", rec_id, exc)
                write_output(args.output / f"{rec_id}.json", [])
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
                except Exception as exc:
                    logger.exception("Future lỗi: %s", exc)
    # Save caches
    translator.save_cache()
    icd_retriever.save_index()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
