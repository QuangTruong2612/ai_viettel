"""Tải RxNorm qua endpoint mới /REST/drugs.json?name=<drug>.

Endpoint này trả về các concept liên quan đến một drug name, gồm cả
synonyms, brand names, dose forms. Input là một danh sách tên thuốc.

Ví dụ:
  https://rxnav.nlm.nih.gov/REST/drugs.json?name=aspirin

Trả:
  {"drugGroup": {"name": "...", "conceptGroup": [
      {"tty": "SCD", "conceptProperties": [{"rxcui": "...", "name": "..."}]},
      ...
  ]}}

Cách dùng:
  # Tải danh sách ~500 thuốc phổ biến
  python scripts/download_rxnorm.py --names data/drug_names.txt --out data/rxnorm_raw.json
  python scripts/build_vn_dict.py --dump data/rxnorm_raw.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("download_rxnorm")

RXNAV = "https://rxnav.nlm.nih.gov/REST"


# Danh sách thuốc phổ biến ở VN (top ~100) — nếu user không cung cấp file
DEFAULT_DRUGS = [
    # Từ ví dụ BTC
    "amlodipine", "aspirin", "metoprolol succinate", "guaifenesin", "nystatin",
    "acetaminophen", "pravastatin", "docusate sodium", "senna", "clonazepam",
    # Phổ biến ở VN
    "paracetamol", "ibuprofen", "diclofenac", "losartan", "metformin",
    "atorvastatin", "simvastatin", "hydrochlorothiazide", "furosemide",
    "lisinopril", "enalapril", "valsartan", "irbesartan", "candesartan",
    "carvedilol", "bisoprolol", "propranolol", "atenolol", "diltiazem",
    "nifedipine", "verapamil", "digoxin", "warfarin", "apixaban",
    "clopidogrel", "rivaroxaban", "amiodarone", "spironolactone",
    "potassium chloride", "insulin glargine", "insulin aspart", "glipizide",
    "glyburide", "sitagliptin", "pioglitazone", "rosuvastatin",
    "ciprofloxacin", "amoxicillin", "azithromycin", "ceftriaxone",
    "cefuroxime", "doxycycline", "metronidazole", "levofloxacin",
    "sulfamethoxazole trimethoprim", "valacyclovir", "oseltamivir",
    "fluconazole", "itraconazole", "albendazole",
    "loratadine", "cetirizine", "fexofenadine", "diphenhydramine",
    "salbutamol", "fluticasone", "budesonide", "montelukast",
    "omeprazole", "lansoprazole", "pantoprazole", "famotidine",
    "ondansetron", "metoclopramide", "loperamide",
    "tramadol", "morphine", "codeine", "gabapentin", "pregabalin",
    "sertraline", "fluoxetine", "escitalopram", "venlafaxine",
    "haloperidol", "risperidone", "olanzapine", "quetiapine",
    "trazodone", "mirtazapine", "amitriptyline",
    "levothyroxine", "methimazole", "prednisone", "methylprednisolone",
    "hydrocortisone", "dexamethasone",
    "albuterol", "tiotropium", "ipratropium",
]


def fetch_drug(name: str, timeout: int = 20) -> list[dict[str, Any]]:
    """Lấy tất cả concepts cho 1 drug name."""
    try:
        r = requests.get(f"{RXNAV}/drugs.json", params={"name": name}, timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        logger.warning("Lỗi fetch %r: %s", name, exc)
        return []
    group = data.get("drugGroup", {}) or {}
    out: list[dict[str, Any]] = []
    for g in group.get("conceptGroup", []) or []:
        ttype = g.get("tty", "")
        for c in g.get("conceptProperties", []) or []:
            rxcui = c.get("rxcui")
            nm = c.get("name")
            syn = c.get("synonym") or ""
            if rxcui and nm:
                entry = {"rxcui": rxcui, "name": nm, "term_type": ttype}
                if syn and syn != nm:
                    entry["synonym"] = syn
                out.append(entry)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Tải RxNorm qua /drugs.json")
    p.add_argument(
        "--names",
        type=Path,
        default=None,
        help="File .txt chứa danh sách drug name, 1 tên/dòng. Nếu không có → dùng DEFAULT_DRUGS",
    )
    p.add_argument("--out", type=Path, default=Path("data/rxnorm_raw.json"))
    p.add_argument("--delay", type=float, default=0.1, help="Giây giữa mỗi request")
    args = p.parse_args(argv)

    if args.names and args.names.exists():
        drug_names = [ln.strip() for ln in args.names.read_text(encoding="utf-8").splitlines() if ln.strip()]
    else:
        drug_names = DEFAULT_DRUGS
    logger.info("Tải %d drugs từ RxNav...", len(drug_names))

    all_concepts: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for i, name in enumerate(drug_names):
        concepts = fetch_drug(name)
        added = 0
        for c in concepts:
            key = (c["rxcui"], c["name"])
            if key in seen:
                continue
            seen.add(key)
            all_concepts.append(c)
            added += 1
        logger.info("[%d/%d] %r: +%d (tổng %d)", i + 1, len(drug_names), name, added, len(all_concepts))
        time.sleep(args.delay)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(all_concepts, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Đã ghi %s (%d entries)", args.out, len(all_concepts))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
