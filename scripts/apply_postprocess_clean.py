"""Apply updated _refine_stage2_results and assertion rules to all output files with deepcopy comparison."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.postprocess import _refine_stage2_results

def clean_all_outputs(output_dir: Path, input_dir: Path) -> None:
    output_files = sorted(
        [f for f in output_dir.glob("*.json") if f.stem.isdigit()],
        key=lambda x: int(x.stem),
    )
    print(f"[INFO] Running _refine_stage2_results on {len(output_files)} output files...")

    total_type_changes = 0
    total_assertion_changes = 0

    for fpath in output_files:
        rec_id = int(fpath.stem)
        input_path = input_dir / f"{rec_id}.txt"
        if not input_path.exists():
            input_path = input_dir / f"{rec_id}.json"
        if not input_path.exists():
            continue

        input_text = input_path.read_text(encoding="utf-8")
        entities = json.loads(fpath.read_text(encoding="utf-8"))
        original_copy = copy.deepcopy(entities)

        refined = _refine_stage2_results(input_text, entities)

        file_changed = False
        for old, new in zip(original_copy, refined):
            if old.get("type") != new.get("type"):
                total_type_changes += 1
                file_changed = True
            if old.get("assertions") != new.get("assertions"):
                total_assertion_changes += 1
                file_changed = True

        if file_changed:
            fpath.write_text(json.dumps(refined, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] Type changes: {total_type_changes}, Assertion changes: {total_assertion_changes}")

if __name__ == "__main__":
    inp_dir = Path("data/input") if Path("data/input").exists() else Path("input")
    clean_all_outputs(Path("output"), inp_dir)
