"""Build RxNorm index + dictionary tên thuốc VN↔EN.

Input : data/rxnorm_raw.json  (output của download_rxnorm.py)
Output:
- data/rxnorm_index.json: index tra nhanh
- data/vn_drug_names.csv : bảng EN,RXCLI,VN_NAMES
- data/rxterm_seed.csv   : seed từ BTC examples + vài thuốc phổ biến
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rxnorm_rag import RxNormIndex, save_index  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_vn_dict")


# ---------------------------------------------------------------------- #
# Seed mapping: tên EN ↔ rxcui (lấy từ ví dụ BTC + Top-100 VN phổ biến)
# Cập nhật thủ công khi thấy lỗi.
# ---------------------------------------------------------------------- #

SEED_ENTRIES: list[dict[str, Any]] = [
    # From the BTC example
    {"name": "amlodipine 10 mg oral tablet", "rxcui": "308135"},
    {"name": "aspirin 81 mg oral tablet", "rxcui": "243670"},
    {"name": "metoprolol succinate xl 50 mg oral tablet", "rxcui": "866436"},
    {"name": "guaifenesin oral solution", "rxcui": "392085"},
    {"name": "nystatin oral suspension", "rxcui": "7597"},
    {"name": "acetaminophen 325-650 mg oral tablet", "rxcui": "313782"},
    {"name": "pravastatin 40 mg oral tablet", "rxcui": "904475"},
    {"name": "docusate sodium 100 mg oral capsule", "rxcui": "1099279"},
    {"name": "senna 8.6 mg oral tablet", "rxcui": "312935"},
    {"name": "clonazepam 0.5 mg oral tablet", "rxcui": "197527"},
    {"name": "clonazepam 1.5 mg oral tablet", "rxcui": "197528"},
    # Common Vietnamese drugs
    {"name": "paracetamol 500 mg oral tablet", "rxcui": "313782"},
    {"name": "ibuprofen 400 mg oral tablet", "rxcui": "315677"},
    {"name": "diclofenac 50 mg oral tablet", "rxcui": "335548"},
    {"name": "losartan 50 mg oral tablet", "rxcui": "979480"},
    {"name": "metformin 500 mg oral tablet", "rxcui": "860975"},
    {"name": "atorvastatin 20 mg oral tablet", "rxcui": "617318"},
    {"name": "furosemide 40 mg oral tablet", "rxcui": "315677"},
    {"name": "hydrochlorothiazide 25 mg oral tablet", "rxcui": "315677"},
    {"name": "lisinopril 10 mg oral tablet", "rxcui": "314076"},
]


# Vietnamese transliterations (heuristic mapping tên VN phổ biến → EN)
VN_TRANSLITERATIONS: dict[str, str] = {
    "amlodipin": "amlodipine",
    "amlodipine": "amlodipine",
    "aspirin": "aspirin",
    "paracetamol": "acetaminophen",
    "acetaminophen": "acetaminophen",
    "metoprolol": "metoprolol",
    "guaifenesin": "guaifenesin",
    "nystatin": "nystatin",
    "pravastatin": "pravastatin",
    "atorvastatin": "atorvastatin",
    "simvastatin": "simvastatin",
    "clonazepam": "clonazepam",
    "ibuprofen": "ibuprofen",
    "losartan": "losartan",
    "valsartan": "valsartan",
    "metformin": "metformin",
    "diclofenac": "diclofenac",
    "senna": "senna",
    "docusate": "docusate",
    "lactulose": "lactulose",
}


def build_index_from_dump(dump_path: Path) -> RxNormIndex:
    idx = RxNormIndex()
    # Add seeds first
    for entry in SEED_ENTRIES:
        idx.add(entry["name"], entry["rxcui"], "SCD")
    # Then raw dump
    if dump_path.exists():
        with dump_path.open("r", encoding="utf-8") as f:
            rows = json.load(f)
        for row in rows:
            name = str(row.get("name", "")).strip()
            rxcui = str(row.get("rxcui", "")).strip()
            ttype = str(row.get("term_type", "")).strip()
            if name and rxcui:
                idx.add(name, rxcui, ttype)
            for rel in row.get("related", []) or []:
                if rel:
                    idx.add(rel, rxcui, ttype)
    return idx


def export_vn_csv(idx: RxNormIndex, path: Path) -> None:
    """Ghi CSV: name_en,rxcui,vn_aliases dùng cho tra cứu / debug."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name_en", "rxcui", "vn_aliases"])
        seen: set[str] = set()
        for name, rxcui in zip(idx.names, idx.rxcuis, strict=True):
            # Tìm alias VN nếu có
            token = name.split()[0].lower().strip(",.")
            vn_candidates = [k for k in VN_TRANSLITERATIONS if VN_TRANSLITERATIONS[k] == token]
            alias = "|".join(vn_candidates) if vn_candidates else ""
            key = f"{name}|{rxcui}"
            if key in seen:
                continue
            seen.add(key)
            w.writerow([name, rxcui, alias])
    logger.info("Ghi %s", path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build RxNorm index + VN dict")
    parser.add_argument(
        "--dump",
        type=Path,
        default=Path("data/rxnorm_raw.json"),
        help="JSON dump từ download_rxnorm.py",
    )
    parser.add_argument("--out-index", type=Path, default=Path("data/rxnorm_index.json"))
    parser.add_argument("--out-csv", type=Path, default=Path("data/vn_drug_names.csv"))
    args = parser.parse_args(argv)

    idx = build_index_from_dump(args.dump)
    save_index(idx, args.out_index)
    export_vn_csv(idx, args.out_csv)
    logger.info("Xong: %d names, %d exact keys", len(idx.names), len(idx.exact))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
