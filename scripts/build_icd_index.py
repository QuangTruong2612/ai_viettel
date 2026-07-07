"""Build ICD-10 structured index từ data/DM_ICD10_19_8_BYT.json → data/icd_index.json.

Input JSON format (BYT chính thức, mỗi entry 1 dict):
    {
        "Mã": "Z95.8",
        "Tên bệnh": "Sự có mặt của dụng cụ cấy và mảnh ghép tim và mạch máu khác",
        "Nhóm bệnh": "Những người có nguy cơ sức khỏe tiềm ẩn ...",
        "Mô tả": "QĐ 4469/BYT ngày 28/10/2020",
        "Hiệu lực": "Có"
    }

Output structure (data/icd_index.json):
    {
        "exact": {"tăng huyết áp": ["I10"], ...},  # name.lower() -> [code]
        "names": ["Tăng huyết áp", ...],          # cho fuzzy
        "codes": ["I10", ...],                     # parallel với names
    }

Đặc biệt:
- Auto-detect format BYT (Mã/Tên bệnh) vs ICD10_Data cũ (Mã bệnh/Tên bệnh gốc).
- Parentheses stripping: tạo key sạch cho "Hen [suyễn]" → key "Hen", "Rối loạn... (modifier)" → key "Rối loạn...".
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("build_icd_index")

PROJECT_DIR = Path(__file__).resolve().parents[1]
# Default: icd10.jsonl mới nhất (WHO ICD-10 2019 VN+EN, 15,732 codes).
# Fallback: DM_ICD10_19_8_BYT.json (BYT chính thức, 36,689 codes, VN only).
DEFAULT_INPUT_CANDIDATES = [
    PROJECT_DIR / "data" / "icd10.jsonl",
    PROJECT_DIR / "data" / "DM_ICD10_19_8_BYT.json",
    PROJECT_DIR / "data" / "ICD10_Data.json",
]
DEFAULT_INPUT = next((c for c in DEFAULT_INPUT_CANDIDATES if c.exists()), DEFAULT_INPUT_CANDIDATES[0])
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "icd_index.json"


def _strip_parens(name: str) -> str:
    """Bỏ [...] và (...) ở cuối để có key sạch cho exact match.

    VD: 'Hen [suyễn]' → 'Hen', 'Rối loạn... (modifier)' → 'Rối loạn...'.
    """
    if not name:
        return name
    cleaned = re.sub(r"\s*\[.*?\]\s*$", "", name)
    cleaned = re.sub(r"\s*\(.*?\)\s*$", "", cleaned)
    return cleaned.strip()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build ICD-10 structured index từ JSON → JSON"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="JSON input (BYT format mới nhất)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Output JSON cho ICDIndex")
    parser.add_argument("--skip-if-exists", action="store_true",
                        help="Skip nếu output đã tồn tại")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild kể cả khi output đã tồn tại")
    args = parser.parse_args()

    if not args.input.exists():
        logger.error("Không tìm thấy input file: %s", args.input)
        return 1
    if args.output.exists() and args.skip_if_exists and not args.force:
        logger.info("%s đã tồn tại → skip. Dùng --force để rebuild.",
                    args.output.name)
        return 0

    logger.info("Bắt đầu đọc %s...", args.input.name)
    t0 = time.time()

    # Data structures
    exact: dict[str, list[str]] = {}
    names: list[str] = []
    codes: list[str] = []
    name_to_idx: dict[str, int] = {}

    n_total, n_kept, n_stripped = 0, 0, 0
    rows: list[dict] = []
    suffix = args.input.suffix.lower()

    if suffix == ".json":
        # BYT hoặc ICD10_Data cũ (JSON array)
        with args.input.open("r", encoding="utf-8") as f:
            data = json.load(f)
        rows = data
    else:
        # JSONL format (icd10.jsonl mới hoặc cũ)
        with args.input.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    # Auto-detect format
    sample = rows[0] if rows else {}
    if "Mã" in sample and "Tên bệnh" in sample:
        is_byt_format = True
        fmt_label = "BYT (DM_ICD10_19_8_BYT.json)"
    elif "Mã bệnh" in sample:
        is_byt_format = False
        fmt_label = "ICD10_Data.json (cũ)"
    elif "code" in sample and "desc_vi" in sample:
        # icd10.jsonl mới (WHO ICD-10 2019 VN+EN)
        is_who = True
        is_byt_format = False
        fmt_label = "WHO ICD-10 2019 (icd10.jsonl, VN+EN)"
    else:
        is_who = False
        is_byt_format = False
        fmt_label = "icd10.jsonl cũ (EN only)"
    logger.info("Detected format: %s", fmt_label)

    for row in rows:
        n_total += 1
        if is_byt_format:
            code = str(row.get("Mã", "")).strip()
            name = str(row.get("Tên bệnh", "")).strip()
        elif is_who:
            # icd10.jsonl mới: prefer desc_vi, fallback desc_en
            code = str(row.get("code", "")).strip()
            name = str(row.get("desc_vi", row.get("desc_en", ""))).strip()
        else:
            code = str(row.get("Mã bệnh", row.get("code", ""))).strip()
            name = str(row.get("Tên bệnh gốc", row.get("desc_en", ""))).strip()

        if not (code and name):
            continue
        n_kept += 1

        # Exact key (lowercase) — primary
        key = name.lower()
        exact.setdefault(key, []).append(code)

        # Cleaned key (no parens) — secondary cho "Hen [suyễn]" → "Hen"
        name_clean = _strip_parens(name)
        if name_clean and name_clean != name:
            clean_key = name_clean.lower()
            if clean_key not in exact:
                exact.setdefault(clean_key, []).append(code)
                n_stripped += 1

        # Names list (parallel với codes) — cho fuzzy
        if name not in name_to_idx:
            name_to_idx[name] = len(names)
            names.append(name)
            codes.append(code)

    data_out = {
        "exact": exact,
        "names": names,
        "codes": codes,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(data_out, f, ensure_ascii=False, indent=1)

    elapsed = time.time() - t0
    logger.info(
        "Done! %d/%d rows → %d unique names, %d exact keys (%d cleaned) "
        "(%.1fs) → %s",
        n_kept, n_total, len(names), len(exact), n_stripped,
        elapsed, args.output.name,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())