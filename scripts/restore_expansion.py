"""Re-expand entities to all occurrences in input text using align_and_expand_entities."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.postprocess import align_and_expand_entities, dedupe_entities

def restore_expanded(output_dir: Path, input_dir: Path) -> None:
    output_files = sorted(
        [f for f in output_dir.glob("*.json") if f.stem.isdigit()],
        key=lambda x: int(x.stem),
    )
    print(f"[INFO] Restoring expansion on {len(output_files)} output files...")
    
    total_before = 0
    total_after = 0

    for fpath in output_files:
        rec_id = int(fpath.stem)
        input_path = input_dir / f"{rec_id}.txt"
        if not input_path.exists():
            input_path = input_dir / f"{rec_id}.json"
        if not input_path.exists():
            print(f"[WARN] No input file for {rec_id}")
            continue

        input_text = input_path.read_text(encoding="utf-8")
        entities = json.loads(fpath.read_text(encoding="utf-8"))

        total_before += len(entities)

        # Re-expand all occurrences in input_text
        expanded = align_and_expand_entities(input_text, entities)
        # Deduplicate exact text + position + type duplicates
        expanded = dedupe_entities(expanded)

        total_after += len(expanded)
        fpath.write_text(json.dumps(expanded, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] Total entities before: {total_before} → after: {total_after}")

if __name__ == "__main__":
    restore_expanded(Path("output"), Path("data/input") if Path("data/input").exists() else Path("input"))
