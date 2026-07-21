"""Audit and fix ICD candidate hallucinations in output files."""

from __future__ import annotations

import json
import glob
from pathlib import Path
from collections import Counter

def audit_and_fix_icd_hallucinations(output_dir: Path) -> None:
    output_files = sorted(
        [f for f in output_dir.glob("*.json") if f.stem.isdigit()],
        key=lambda x: int(x.stem),
    )
    print(f"[INFO] Auditing ICD candidates on {len(output_files)} output files...")

    hallucinations_found = 0
    d83_count = 0

    for fpath in output_files:
        entities = json.loads(fpath.read_text(encoding="utf-8"))
        if not isinstance(entities, list):
            continue

        changed = False

        for ent in entities:
            if not isinstance(ent, dict):
                continue

            t = ent.get("type", "")
            txt = str(ent.get("text", "")).strip().lower()
            cands = ent.get("candidates", [])

            if t == "CHẨN_ĐOÁN" and cands:
                # Fix 1: 'nhịp xoang' -> D83.1 hallucination
                if "nhịp xoang" in txt:
                    if "D83.1" in cands or any(c.startswith("D83") for c in cands):
                        cands = [c for c in cands if not c.startswith("D83")]
                        if "I49.8" not in cands:
                            cands.append("I49.8")
                        ent["candidates"] = cands
                        changed = True
                        d83_count += 1

                # Fix 2: 'tăng huyết áp' -> remove generic 'I10' if specific 'I10.0' or 'I15' is present
                if "tăng huyết áp" in txt:
                    if not cands:
                        ent["candidates"] = ["I10"]
                        changed = True

                # Fix 3: 'đái tháo đường' -> E11
                if "đái tháo đường" in txt:
                    if not cands:
                        ent["candidates"] = ["E11", "E11.9"]
                        changed = True

                # Fix 4: 'suy tim' -> I50.9
                if "suy tim" in txt:
                    if not cands:
                        ent["candidates"] = ["I50.9"]
                        changed = True

                # Fix 5: 'rung nhĩ' -> I48
                if "rung nhĩ" in txt:
                    if not cands:
                        ent["candidates"] = ["I48"]
                        changed = True

        if changed:
            fpath.write_text(json.dumps(entities, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] Fixed {d83_count} D83.1 sinus rhythm hallucinations!")

if __name__ == "__main__":
    audit_and_fix_icd_hallucinations(Path("output"))
