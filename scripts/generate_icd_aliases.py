"""Sinh alias/cách viết thông thường cho từng mã ICD-10 bằng LLM (batch + resume).

MỤC ĐÍCH
--------
File `icd10.jsonl` gốc chỉ có 1 tên mô tả chính thức (desc_vi) / mã, dạng formal,
dài, khớp rất kém với cách bác sĩ Việt Nam viết tắt trong bệnh án thực tế
(vd: mã J18.9 desc_vi="Viêm phổi, không xác định" nhưng bác sĩ chỉ viết "viêm phổi").

Script này gọi LLM cho từng NHÓM mã (batch), yêu cầu liệt kê 3-5 cách viết
thông thường/ngắn gọn mà bác sĩ VN hay dùng, rồi ghi ra `icd_aliases.json`
để merge vào index tra cứu (xem `merge_icd_aliases.py`).

TẠI SAO BATCH THEO NHÓM (không gọi từng mã 1)
-----------------------------------------------
15,732 mã, nếu gọi LLM 1 lần/mã sẽ mất rất nhiều thời gian + rất tốn.
Script này gộp N mã (mặc định 15) vào 1 lần gọi LLM, giảm số lần gọi xuống
còn ~1,050 lần (15732/15) — nhanh hơn ~15 lần so với gọi từng mã.

RESUME / CHECKPOINT
--------------------
Sau mỗi `--checkpoint-every` batch, script ghi đè `--output` để không mất tiến độ
nếu bị dừng giữa chừng (mất mạng, hết pin, Ctrl+C...). Chạy lại script với cùng
tham số sẽ tự động BỎ QUA các mã đã có alias trong file output, chỉ xử lý mã
còn thiếu.

CÁCH DÙNG
---------
    # Chạy full (mặc định dùng Ollama qwen2.5:7b tại localhost:11434, giống pipeline chính)
    python generate_icd_aliases.py \
        --input data/icd10.jsonl \
        --output data/icd_aliases.json

    # Test nhanh trên 50 mã đầu tiên trước khi chạy full
    python generate_icd_aliases.py --input data/icd10.jsonl \
        --output data/icd_aliases_test.json --limit 50

    # Resume (tự động tiếp tục nếu bị dừng giữa chừng — chỉ cần chạy lại lệnh cũ)
    python generate_icd_aliases.py --input data/icd10.jsonl --output data/icd_aliases.json

    # Rebuild lại từ đầu (bỏ qua resume, ghi đè toàn bộ)
    python generate_icd_aliases.py --input data/icd10.jsonl --output data/icd_aliases.json --force

    # Đổi batch size (batch nhỏ hơn nếu model hay bị lỗi JSON/timeout)
    python generate_icd_aliases.py --input data/icd10.jsonl --output data/icd_aliases.json --batch-size 8

Sau khi chạy xong, dùng `merge_icd_aliases.py` để gộp kết quả vào `icd_index.json`
(xem hướng dẫn ở file đó / README đi kèm).
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("generate_icd_aliases")

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_DIR / "data" / "icd10.jsonl"
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "icd_aliases.json"

# Cho phép import LLMClient có sẵn của pipeline (Ollama/LM Studio, OpenAI-compatible).
# Nếu không import được (chạy độc lập ngoài project), tự fallback dùng `openai` trực tiếp.
try:
    sys.path.insert(0, str(PROJECT_DIR))
    from src.llm_client import LLMClient, LLMConfig  # type: ignore

    _HAS_PROJECT_CLIENT = True
except Exception:  # pragma: no cover
    _HAS_PROJECT_CLIENT = False


SYSTEM_PROMPT = """Bạn là bác sĩ Việt Nam có 20 năm kinh nghiệm viết bệnh án.

NHIỆM VỤ: Với mỗi mã ICD-10 và tên bệnh CHÍNH THỨC được cung cấp, hãy liệt kê 3-5
cách viết THÔNG THƯỜNG, NGẮN GỌN mà bác sĩ Việt Nam hay dùng trong bệnh án thực tế
cho ĐÚNG bệnh đó (không phải bệnh rộng hơn/hẹp hơn/liên quan).

QUY TẮC BẮT BUỘC:
1. Chỉ liệt kê cách viết có Ý NGHĨA Y KHOA GIỐNG HỆT tên chính thức — không được
   liệt kê bệnh rộng hơn, hẹp hơn, hoặc chỉ liên quan (vd: KHÔNG được coi "u" là
   đồng nghĩa với "ung thư" — u có thể lành tính; KHÔNG được coi "thận" đồng nghĩa
   với toàn bộ "hệ tiết niệu").
2. Ưu tiên cách viết TẮT/NGẮN mà bác sĩ thực tế hay dùng (bỏ bớt phần mô tả kỹ
   thuật/phân loại phụ nếu không ảnh hưởng nghĩa chính).
3. Bao gồm cả từ viết tắt phổ biến nếu có (vd: THA, ĐTĐ, NMCT, COPD, TBMMN...).
4. Nếu tên chính thức đã đủ ngắn gọn (< 4 từ) và không có cách viết tắt nào khác,
   trả về danh sách rỗng [] — KHÔNG bịa ra alias không cần thiết.
5. Nếu không chắc chắn về 1 mã nào đó, bỏ qua (không trả field đó) thay vì đoán bừa.

FORMAT ĐẦU RA: CHỈ trả về 1 JSON object, không kèm giải thích, không markdown code fence:
{
  "<mã 1>": ["cách viết 1", "cách viết 2", ...],
  "<mã 2>": ["cách viết 1", ...],
  ...
}

VÍ DỤ:
Input:
- J18.9: Viêm phổi, không xác định
- I10: Tăng huyết áp vô căn (nguyên phát)
- E11: Đái tháo đường không phụ thuộc insulin

Output:
{
  "J18.9": ["viêm phổi", "viêm phổi cấp", "VP"],
  "I10": ["tăng huyết áp", "THA", "cao huyết áp"],
  "E11": ["đái tháo đường type 2", "đái tháo đường tuýp 2", "ĐTĐ", "ĐTĐ type 2", "tiểu đường type 2"]
}
"""


def _build_user_prompt(batch: list[dict[str, str]]) -> str:
    lines = [f"- {row['code']}: {row['desc_vi']}" for row in batch]
    return "Danh sách mã cần sinh alias:\n" + "\n".join(lines) + "\n\nOUTPUT JSON:"


def _load_icd_rows(input_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            code = str(d.get("code", "")).strip()
            desc_vi = str(d.get("desc_vi", d.get("desc_en", ""))).strip()
            if code and desc_vi:
                rows.append({"code": code, "desc_vi": desc_vi})
    return rows


def _load_existing_output(output_path: Path) -> dict[str, list[str]]:
    if not output_path.exists():
        return {}
    try:
        return json.loads(output_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Không đọc được output cũ (%s), bắt đầu từ đầu: %s", output_path, exc)
        return {}


def _save_output(output_path: Path, data: dict[str, list[str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1, sort_keys=True)
    tmp_path.replace(output_path)  # atomic write — tránh corrupt file nếu bị ngắt giữa chừng


class _RawOpenAIFallbackClient:
    """Fallback client tối giản nếu không import được src.llm_client (chạy ngoài project).

    Dùng thư viện `openai`, trỏ tới Ollama OpenAI-compatible endpoint.
    """

    def __init__(self, base_url: str, model: str, api_key: str = "ollama") -> None:
        from openai import OpenAI  # type: ignore

        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=300, max_retries=0)
        self.model = model

    def call_sync(self, prompt: str, system_prompt: str, max_tokens: int = 2048, temperature: float = 0.3) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""


def _extract_json_object(text: str) -> dict[str, Any]:
    """Trích JSON object từ response LLM (robust, tự strip code fence)."""
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 3:
            inner = "```".join(parts[1:-1]).strip()
            if inner.lower().startswith("json"):
                inner = inner[4:].strip()
            text = inner
    if not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
    return json.loads(text)


def _call_llm_for_batch(
    client: Any,
    batch: list[dict[str, str]],
    max_retries: int = 2,
) -> dict[str, list[str]]:
    prompt = _build_user_prompt(batch)
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            if _HAS_PROJECT_CLIENT and hasattr(client, "call_sync") and not isinstance(
                client, _RawOpenAIFallbackClient
            ):
                # src.llm_client.LLMClient.call_sync thực tế nhận keyword arg `system_prompt=`
                # (không phải `system=`). Sửa tại đây cho khớp chữ ký thực tế.
                raw = client.call_sync(prompt, system_prompt=SYSTEM_PROMPT, max_tokens=2048, temperature=0.3)
            else:
                raw = client.call_sync(prompt, system_prompt=SYSTEM_PROMPT, max_tokens=2048, temperature=0.3)
            parsed = _extract_json_object(raw)
            if not isinstance(parsed, dict):
                raise ValueError(f"Kết quả không phải JSON object: {type(parsed)}")
            # Validate: value phải là list[str]
            cleaned: dict[str, list[str]] = {}
            valid_codes = {row["code"] for row in batch}
            for code, aliases in parsed.items():
                if code not in valid_codes:
                    logger.warning("LLM trả code lạ không có trong batch: %s (bỏ qua)", code)
                    continue
                if not isinstance(aliases, list):
                    continue
                cleaned[code] = [str(a).strip() for a in aliases if str(a).strip()]
            return cleaned
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning(
                "Batch fail (attempt %d/%d, %d codes bắt đầu từ %s): %s",
                attempt, max_retries, len(batch), batch[0]["code"], exc,
            )
            time.sleep(2 * attempt)
    logger.error("Batch fail hẳn sau %d lần thử, bỏ qua batch này (sẽ retry ở lần chạy sau): %s",
                 max_retries, [r["code"] for r in batch])
    return {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="icd10.jsonl gốc")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="File JSON output (code -> [aliases])")
    parser.add_argument("--batch-size", type=int, default=15, help="Số mã / lần gọi LLM (default 15)")
    parser.add_argument("--limit", type=int, default=None, help="Chỉ xử lý N mã đầu (để test nhanh)")
    parser.add_argument("--checkpoint-every", type=int, default=20, help="Ghi checkpoint sau mỗi N batch")
    parser.add_argument("--force", action="store_true", help="Bỏ qua resume, chạy lại từ đầu toàn bộ")
    parser.add_argument("--shuffle", action="store_true",
                         help="Xáo trộn thứ tự mã trước khi chạy (hữu ích khi --limit để test mẫu ngẫu nhiên thay vì luôn lấy N mã đầu)")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:11434/v1",
                         help="OpenAI-compatible endpoint (Ollama mặc định :11434, LM Studio :1234)")
    parser.add_argument("--model", type=str, default="qwen2.5:7b", help="Tên model (Ollama tag hoặc LM Studio model id)")
    args = parser.parse_args()

    if not args.input.exists():
        logger.error("Không tìm thấy input: %s", args.input)
        return 1

    rows = _load_icd_rows(args.input)
    logger.info("Đọc được %d mã ICD-10 từ %s", len(rows), args.input.name)

    if args.shuffle:
        random.seed(42)
        random.shuffle(rows)
    if args.limit:
        rows = rows[: args.limit]
        logger.info("Giới hạn --limit=%d → chỉ xử lý %d mã", args.limit, len(rows))

    existing = {} if args.force else _load_existing_output(args.output)
    if existing:
        logger.info("Resume: đã có %d mã trong %s, sẽ bỏ qua các mã này", len(existing), args.output.name)

    todo = [row for row in rows if row["code"] not in existing]
    logger.info("Còn %d/%d mã cần xử lý", len(todo), len(rows))

    if not todo:
        logger.info("Không còn gì để làm. Xong.")
        return 0

    # Khởi tạo LLM client
    if _HAS_PROJECT_CLIENT:
        try:
            cfg = LLMConfig(base_url=args.base_url, model=args.model, temperature=0.3, max_tokens=2048)
            client = LLMClient(cfg)
            logger.info("Dùng src.llm_client.LLMClient (project client) — base_url=%s model=%s",
                        args.base_url, args.model)
        except Exception as exc:
            logger.warning("Không khởi tạo được project LLMClient (%s), fallback raw OpenAI client", exc)
            client = _RawOpenAIFallbackClient(args.base_url, args.model)
    else:
        client = _RawOpenAIFallbackClient(args.base_url, args.model)
        logger.info("Dùng raw OpenAI-compatible client — base_url=%s model=%s", args.base_url, args.model)

    batches = [todo[i : i + args.batch_size] for i in range(0, len(todo), args.batch_size)]
    logger.info("Chia thành %d batch (batch-size=%d)", len(batches), args.batch_size)

    result = dict(existing)
    n_codes_done = 0
    n_batch_fail = 0
    t0 = time.time()

    for i, batch in enumerate(batches, start=1):
        aliases_map = _call_llm_for_batch(client, batch)
        if not aliases_map:
            n_batch_fail += 1
        # Ghi cả các mã fail thành list rỗng KHÔNG được — để trống để lần sau tự retry.
        # Chỉ ghi các mã LLM thực sự trả kết quả.
        for row in batch:
            code = row["code"]
            if code in aliases_map:
                result[code] = aliases_map[code]
                n_codes_done += 1

        if i % args.checkpoint_every == 0 or i == len(batches):
            _save_output(args.output, result)
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(batches) - i) / rate if rate > 0 else float("inf")
            logger.info(
                "Checkpoint: batch %d/%d | %d mã đã có alias | %d batch fail | "
                "%.1fs đã chạy | ETA ~%.0fs",
                i, len(batches), n_codes_done, n_batch_fail, elapsed, eta,
            )

    _save_output(args.output, result)
    logger.info(
        "HOÀN TẤT. Tổng %d/%d mã có alias trong %s. %d batch fail (chạy lại lệnh cũ để retry các mã còn thiếu).",
        len(result), len(rows), args.output, n_batch_fail,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())