"""Build RxNorm structured index từ data/rxnorm.jsonl → data/rxnorm_index.json.

Input JSONL format (mỗi dòng 1 JSON object):
    {
        "rxcui": "1111439",
        "name": "metoprolol",
        "ingredient": "metoprolol",
        "strength": "25 MG",
        "doseform": "Oral Tablet",
        "source": "...",
        "version": "..."
    }

Output structure (data/rxnorm_index.json):
    {
        "by_ingredient_strength": {"(metoprolol, 25 MG)": ["1111439", ...]},
        "by_ingredient":          {"metoprolol": ["1111439", ...]},
        "names":                   [...],          # cho fuzzy fallback
        "rxcuis":                  [...],
        "name_to_idx":             {...},
    }

Cảnh báo đặc biệt: row nào có strength="" hoặc doseform="" vẫn được add — chỉ
không lưu vào by_ingredient_strength mà lưu vào by_ingredient (key chỉ tên).
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
logger = logging.getLogger("build_rxnorm_index")

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_DIR / "data" / "rxnorm.jsonl"
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "rxnorm_index.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build RxNorm structured index từ JSONL → JSON"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="JSONL input (mỗi dòng 1 drug)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="JSON output cho RxNormIndex")
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

    # Index structures
    by_ingredient_strength: dict[tuple[str, str], list[str]] = {}
    by_ingredient: dict[str, list[str]] = {}
    names: list[str] = []          # cho fuzzy fallback
    rxcuis: list[str] = []         # parallel với names
    name_to_idx: dict[str, int] = {}

    n_total, n_kept, n_with_strength = 0, 0, 0
    with args.input.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            rxcui = str(row.get("rxcui", "")).strip()
            ingredient = str(row.get("ingredient", "")).strip().lower()
            # Normalize strength: collapse space giữa số + đơn vị ("400 MG" → "400MG")
            # để khớp với normalize_strength() trong src/rxnorm_rag.py
            raw_strength = str(row.get("strength", "")).strip().upper()
            strength = re.sub(
                r"(\d+(?:\.\d+)?)\s+(MG|MCG|G|ML|IU|UNIT|%|MEQ)",
                r"\1\2", raw_strength,
            )
            name = str(row.get("name", "")).strip()

            if not rxcui:
                continue
            n_kept += 1

            # Exact tuple (ingredient, strength)
            if ingredient and strength:
                by_ingredient_strength.setdefault((ingredient, strength), []).append(rxcui)
                n_with_strength += 1

            # Ingredient-only (cho query thiếu strength)
            if ingredient:
                by_ingredient.setdefault(ingredient, []).append(rxcui)

            # Names list (parallel với rxcuis) — dùng cho fuzzy fallback
            if name and name not in name_to_idx:
                name_to_idx[name] = len(names)
                names.append(name)
                rxcuis.append(rxcui)

    data = {
        "by_ingredient_strength": {f"{k[0]}|{k[1]}": v
                                    for k, v in by_ingredient_strength.items()},
        "by_ingredient": by_ingredient,
        "names": names,
        "rxcuis": rxcuis,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)

    elapsed = time.time() - t0
    logger.info(
        "Done! %d/%d rows → %d unique (ingredient, strength) keys, %d ingredients, "
        "%d names (%.1fs) → %s",
        n_kept, n_total,
        len(by_ingredient_strength), len(by_ingredient), len(names),
        elapsed, args.output.name,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())