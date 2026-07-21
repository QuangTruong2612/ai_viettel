"""Sửa span overlaps trong output files.

Khi có 2 entities cùng type, vị trí overlap → một entity là substring của entity kia.
Giữ entity dài hơn (entity ngắn thường là fragment), remove entity ngắn.

Usage:
    uv run python scripts/fix_span_overlaps.py --output output/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _spans_overlap(a: list[int], b: list[int]) -> bool:
    """Trả về True nếu 2 spans có phần giao."""
    if not a or not b or len(a) < 2 or len(b) < 2:
        return False
    s1, e1 = a[0], a[1]
    s2, e2 = b[0], b[1]
    if e1 <= s1 or e2 <= s2:
        return False
    return max(s1, s2) < min(e1, e2)


def fix_overlaps(output_dir: Path, dry_run: bool = False) -> None:
    """Fix span overlaps in output files - keep the longer/more specific entity."""
    output_files = sorted(
        [f for f in output_dir.glob("*.json") if f.stem.isdigit()],
        key=lambda x: int(x.stem),
    )
    print(f"[INFO] Processing {len(output_files)} output files...")

    total_removed = 0

    for fpath in output_files:
        try:
            entities = json.loads(fpath.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[WARN] {fpath.name}: parse error {exc}")
            continue

        if not isinstance(entities, list):
            continue

        # Sort by position start then by length (desc) so longer entities come first
        entities_with_pos = [
            (i, e) for i, e in enumerate(entities)
            if isinstance(e.get("position"), list) and len(e.get("position", [])) == 2
        ]

        to_remove: set[int] = set()

        for i, (idx_a, ent_a) in enumerate(entities_with_pos):
            if idx_a in to_remove:
                continue
            pos_a = ent_a["position"]
            type_a = ent_a.get("type", "")
            len_a = pos_a[1] - pos_a[0]

            for j, (idx_b, ent_b) in enumerate(entities_with_pos):
                if i == j or idx_b in to_remove:
                    continue
                pos_b = ent_b["position"]
                type_b = ent_b.get("type", "")
                len_b = pos_b[1] - pos_b[0]

                # Only check same-type overlaps
                if type_a != type_b:
                    continue

                if _spans_overlap(pos_a, pos_b):
                    # Remove the shorter one
                    if len_a >= len_b:
                        to_remove.add(idx_b)
                    else:
                        to_remove.add(idx_a)

        if to_remove:
            original_count = len(entities)
            entities = [e for i, e in enumerate(entities) if i not in to_remove]
            removed = original_count - len(entities)
            total_removed += removed
            if not dry_run:
                fpath.write_text(
                    json.dumps(entities, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            print(f"  [{fpath.stem}] Removed {removed} overlapping entities")

    print(f"[DONE] Total removed: {total_removed}")
    if dry_run:
        print("[DRY RUN] No files written.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    fix_overlaps(args.output, args.dry_run)


if __name__ == "__main__":
    main()
