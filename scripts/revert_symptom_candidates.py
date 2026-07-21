"""Revert: Xóa candidates khỏi TRIỆU_CHỨNG entities trong output files.

Lý do: Nếu gold data không có candidates cho TRIỆU_CHỨNG (candidates=[]),
mà pred có candidates (vd ["R05"]) → Jaccard = 0.0 (hallucination penalty).
Điều đó làm GIẢM điểm thay vì tăng.

Usage:
    uv run python scripts/revert_symptom_candidates.py --output output/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def revert(output_dir: Path, dry_run: bool = False) -> None:
    output_files = sorted(
        [f for f in output_dir.glob("*.json") if f.stem.isdigit()],
        key=lambda x: int(x.stem),
    )
    print(f"[INFO] Processing {len(output_files)} output files...")

    total_reverted = 0

    for fpath in output_files:
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
            if ent.get("type") == "TRIỆU_CHỨNG":
                if ent.get("candidates"):
                    ent["candidates"] = []
                    changed = True
                    total_reverted += 1

        if changed and not dry_run:
            fpath.write_text(
                json.dumps(entities, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    print(f"[DONE] Reverted {total_reverted} TRIỆU_CHỨNG candidates → []")
    if dry_run:
        print("[DRY RUN] No files written.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    revert(args.output, args.dry_run)


if __name__ == "__main__":
    main()
