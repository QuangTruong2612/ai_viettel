"""Re-attach / refine ICD/RxNorm candidates across output files."""

from __future__ import annotations

import argparse
import json
import sys
import logging
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.WARNING)

from src.icd_rag import ICDRetriever, ICD10VectorSearch
from src.rxnorm_rag import RxNormRetriever
from src.postprocess import (
    sanitize_drug_text,
    _is_generic_drug_class,
    _strip_drug_class_prefix,
    _is_non_treatment_drug_context,
    _apply_deterministic_icd_rules,
)


def reattach_rag(output_dir: Path, input_dir: Path, force: bool = False, dry_run: bool = False) -> None:
    """Re-attach / refine RAG candidates in existing output files."""
    print("[INFO] Loading ICD + RxNorm retrievers...")
    try:
        local_search = ICD10VectorSearch()
        icd_retriever = ICDRetriever(local_search=local_search)
        rxnorm_retriever = RxNormRetriever()
        print("[INFO] Retrievers loaded successfully.")
    except Exception as exc:
        print(f"[ERROR] Cannot load retrievers: {exc}")
        return

    output_files = sorted(
        [f for f in output_dir.glob("*.json") if f.stem.isdigit()],
        key=lambda x: int(x.stem),
    )
    print(f"[INFO] Processing {len(output_files)} output files...")

    total_diag_updated = 0
    total_drug_updated = 0
    total_rules_applied = 0

    for fpath in output_files:
        rec_id = int(fpath.stem)
        input_text = ""
        for ext in (".txt", ".json"):
            inp = input_dir / f"{rec_id}{ext}"
            if inp.exists():
                try:
                    input_text = inp.read_text(encoding="utf-8").strip()
                except Exception:
                    pass
                break

        try:
            entities = json.loads(fpath.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[WARN] {fpath.name}: parse error {exc}")
            continue

        if not isinstance(entities, list):
            continue

        changed = False

        for ent in entities:
            if not isinstance(ent, dict):
                continue

            t = ent.get("type", "")
            text = ent.get("text", "").strip()
            if not text:
                continue

            cands = ent.get("candidates", [])

            if t == "CHẨN_ĐOÁN":
                # Apply deterministic ICD rules (e.g. nhịp xoang -> I49.8, MI location, organism)
                if cands:
                    new_cands = _apply_deterministic_icd_rules(text, cands)
                    # Fix specific bad vector hallucination: D83.1 on nhịp xoang
                    if "nhịp xoang" in text.lower():
                        new_cands = [c for c in new_cands if not c.startswith("D83")]
                        if not any(c.startswith("I49") for c in new_cands):
                            new_cands.append("I49.8")
                    if new_cands != cands:
                        ent["candidates"] = new_cands
                        changed = True
                        total_rules_applied += 1

                if not ent.get("candidates") or force:
                    try:
                        other_ents = [e for e in entities if e is not ent and e.get("text", "").strip()]
                        codes = icd_retriever.lookup(text, other_entities=other_ents, entity_type=t)
                        if codes:
                            new_codes = list(codes)
                            new_codes = _apply_deterministic_icd_rules(text, new_codes)
                            if "nhịp xoang" in text.lower():
                                new_codes = [c for c in new_codes if not c.startswith("D83")]
                                if not any(c.startswith("I49") for c in new_codes):
                                    new_codes.append("I49.8")
                            ent["candidates"] = new_codes
                            changed = True
                            total_diag_updated += 1
                    except Exception:
                        pass

            elif t == "THUỐC" and (not cands or force):
                try:
                    stripped = _strip_drug_class_prefix(text)
                    if stripped is None:
                        continue
                    if stripped != text:
                        text = stripped
                    elif _is_generic_drug_class(text):
                        continue
                    if _is_non_treatment_drug_context(text, input_text):
                        continue
                    drug_q = sanitize_drug_text(text)
                    if drug_q:
                        codes = rxnorm_retriever.lookup(drug_q)
                        if codes:
                            ent["candidates"] = list(codes)
                            changed = True
                            total_drug_updated += 1
                except Exception:
                    pass

        if changed and not dry_run:
            fpath.write_text(
                json.dumps(entities, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    print(f"[DONE] Updated CHẨN_ĐOÁN: {total_diag_updated}, THUỐC: {total_drug_updated}, Rules applied: {total_rules_applied}")
    if dry_run:
        print("[DRY RUN] No files written.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--input", type=Path, default=Path("input"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    reattach_rag(args.output, args.input, args.force, args.dry_run)


if __name__ == "__main__":
    main()
