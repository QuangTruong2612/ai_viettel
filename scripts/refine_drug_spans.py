"""Context-aware drug span refinement based on gold standard conventions in examples.jsonl."""

from __future__ import annotations

import json
import re
import glob
from pathlib import Path

# Suffixes used in treatment / discharge prescriptions that gold trims
TREATMENT_DRUG_SUFFIXES = re.compile(
    r"\s+(?:po|iv|im|sc|bid|tid|qid|daily|prn|hs|qd|q4h|q6h|q8h|q12h|x\s*\d+\s*(?:lần|viên)?.*)$",
    re.IGNORECASE | re.UNICODE,
)

def refine_drug_spans(output_dir: Path, input_dir: Path) -> None:
    output_files = sorted(
        [f for f in output_dir.glob("*.json") if f.stem.isdigit()],
        key=lambda x: int(x.stem),
    )
    print(f"[INFO] Checking drug span normalization on {len(output_files)} output files...")

    total_trimmed = 0

    for fpath in output_files:
        rec_id = int(fpath.stem)
        inp_path = input_dir / f"{rec_id}.txt"
        if not inp_path.exists():
            inp_path = input_dir / f"{rec_id}.json"
        if not inp_path.exists():
            continue

        input_text = inp_path.read_text(encoding="utf-8")
        entities = json.loads(fpath.read_text(encoding="utf-8"))
        if not isinstance(entities, list):
            continue

        changed = False

        for ent in entities:
            if not isinstance(ent, dict):
                continue
            if ent.get("type") != "THUỐC":
                continue

            pos = ent.get("position", [0, 0])
            txt = str(ent.get("text", "")).strip()

            if not txt or not isinstance(pos, list) or len(pos) != 2:
                continue

            s = int(pos[0])

            # Check preceding context (section header)
            pre_context = input_text[max(0, s - 200):s].lower()

            # If under "Điều trị:" or "Thuốc ra viện:" or "Kê đơn:" (new treatment)
            # Gold trims 'po daily', 'po bid' from prescription drugs!
            is_treatment_sec = False
            for h in ("điều trị", "kê đơn", "thuốc ra viện", "hướng xử trí"):
                if h in pre_context:
                    # Check if "tiền sử" or "trước nhập viện" was closer
                    ts_idx = pre_context.rfind("tiền sử")
                    tr_idx = pre_context.rfind("trước")
                    h_idx = pre_context.rfind(h)
                    if h_idx > ts_idx and h_idx > tr_idx:
                        is_treatment_sec = True
                        break

            if is_treatment_sec:
                # Trim po/bid/daily suffix
                m = TREATMENT_DRUG_SUFFIXES.search(txt)
                if m:
                    trimmed_txt = txt[:m.start()].strip()
                    # Verify trimmed_txt exists in input_text at offset s
                    if trimmed_txt and input_text[s:s + len(trimmed_txt)].lower() == trimmed_txt.lower():
                        ent["text"] = input_text[s:s + len(trimmed_txt)]
                        ent["position"] = [s, s + len(trimmed_txt)]
                        total_trimmed += 1
                        changed = True

        if changed:
            fpath.write_text(json.dumps(entities, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] Refined {total_trimmed} treatment drug spans!")

if __name__ == "__main__":
    inp_dir = Path("data/input") if Path("data/input").exists() else Path("input")
    refine_drug_spans(Path("output"), inp_dir)
