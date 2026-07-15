"""Gộp alias đã sinh (icd_aliases.json) vào icd_index.json để dùng ngay trong retrieval.

Script này KHÔNG gọi LLM — chỉ đọc 2 file JSON có sẵn và merge lại, chạy rất nhanh.

CƠ CHẾ MERGE
------------
`icd_index.json` (do build_icd_index.py tạo) có cấu trúc:
    {
        "exact": {"tăng huyết áp": ["I10"], ...},   # name.lower() -> [code, ...]
        "names": [...],
        "codes": [...]
    }

Script này thêm mỗi alias trong `icd_aliases.json` vào dict "exact" — tức là mỗi
cách viết thông thường (vd "viêm phổi", "THA", "ĐTĐ type 2"...) sẽ trở thành 1 key
tra cứu TRỰC TIẾP (Tier-1 exact match, tầng nhanh & tin cậy nhất trong icd_rag.py),
y hệt như tên chính thức.

Nếu 1 alias bị trùng với alias/tên chính thức khác (map sang nhiều code khác nhau),
script sẽ GIỮ CẢ 2 code (không ghi đè) — vì 1 cách viết ngắn gọn đôi khi đúng với
nhiều mã con khác nhau (vd "viêm phổi" có thể khớp J12, J18, J18.9...). Tầng rerank
phía sau (`_rerank_and_select` trong icd_rag.py) sẽ chọn candidate phù hợp nhất dựa
theo ngữ cảnh, nên việc giữ nhiều code ở đây là AN TOÀN và ĐÚNG THIẾT KẾ.

CÁCH DÙNG
---------
    python merge_icd_aliases.py \
        --index data/icd_index.json \
        --aliases data/icd_aliases.json \
        --output data/icd_index.json   # ghi đè trực tiếp (khuyến nghị backup trước)

Khuyến nghị BACKUP trước khi ghi đè:
    cp data/icd_index.json data/icd_index.json.bak
    python merge_icd_aliases.py --index data/icd_index.json --aliases data/icd_aliases.json --output data/icd_index.json

Sau khi merge, KHÔNG cần chạy lại build_icd_index.py — file đã sẵn sàng dùng ngay.
Nếu sau này bạn build lại icd_index.json từ đầu (vd icd10.jsonl gốc thay đổi),
nhớ chạy lại merge_icd_aliases.py để không mất phần alias đã sinh.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("merge_icd_aliases")

PROJECT_DIR = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=Path, default=PROJECT_DIR / "data" / "icd_index.json")
    parser.add_argument("--aliases", type=Path, default=PROJECT_DIR / "data" / "icd_aliases.json")
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "data" / "icd_index.json")
    args = parser.parse_args()

    if not args.index.exists():
        logger.error("Không tìm thấy %s — chạy build_icd_index.py trước.", args.index)
        return 1
    if not args.aliases.exists():
        logger.error("Không tìm thấy %s — chạy generate_icd_aliases.py trước.", args.aliases)
        return 1

    index = json.loads(args.index.read_text(encoding="utf-8"))
    aliases: dict[str, list[str]] = json.loads(args.aliases.read_text(encoding="utf-8"))

    exact: dict[str, list[str]] = index.get("exact", {})
    n_new_keys = 0
    n_merged_into_existing = 0
    n_skipped_empty = 0

    for code, alias_list in aliases.items():
        for alias in alias_list:
            alias_clean = alias.strip()
            if not alias_clean:
                n_skipped_empty += 1
                continue
            key = alias_clean.lower()
            if key not in exact:
                exact[key] = [code]
                n_new_keys += 1
            elif code not in exact[key]:
                exact[key].append(code)
                n_merged_into_existing += 1
            # nếu code đã có sẵn trong exact[key] → không làm gì (tránh trùng lặp)

    index["exact"] = exact

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=1)

    logger.info(
        "Merge xong: %d key alias MỚI, %d alias được thêm vào key đã tồn tại (multi-code), "
        "%d alias rỗng bị bỏ qua. Tổng exact keys hiện tại: %d. Ghi ra %s.",
        n_new_keys, n_merged_into_existing, n_skipped_empty, len(exact), args.output,
    )
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())