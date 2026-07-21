"""Revert drug span trimming: preserve full drug text including po bid, po daily, po prn, etc."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.postprocess import align_and_expand_entities, dedupe_entities

DRUG_EXTEND_PATTERN = re.compile(
    r"\s+(?:po|iv|im|sc|bid|tid|qid|daily|prn|hs|qd|q4h|q6h|q8h|q12h|x\s*\d+(?:\s*(?:lần|viên|ống|gói))?)+",
    re.IGNORECASE | re.UNICODE,
)

def restore_full_drug_spans(output_dir: Path, input_dir: Path) -> None:
    output_files = sorted(
        [f for f in output_dir.glob("*.json") if f.stem.isdigit()],
        key=lambda x: int(x.stem),
    )
    print(f"[INFO] Restoring full drug spans (including po bid, po daily) on {len(output_files)} output files...")

    total_restored = 0

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

            s, e = int(pos[0]), int(pos[1])

            # Check if text after position e in input_text contains po bid / po daily / etc.
            after_text = input_text[e:min(len(input_text), e + 30)]
            m = DRUG_EXTEND_PATTERN.match(after_text)
            if m:
                extended_len = len(m.group(0))
                full_text = input_text[s:e + extended_len]
                ent["text"] = full_text
                ent["position"] = [s, e + extended_len]
                total_restored += 1
                changed = True

        if changed:
            fpath.write_text(json.dumps(entities, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] Restored {total_restored} full drug spans!")

if __name__ == "__main__":
    inp_dir = Path("data/input") if Path("data/input").exists() else Path("input")
    restore_full_drug_spans(Path("output"), inp_dir)
