"""Deduplicate entities cùng text+type trong output files.

Khi model extract cùng 1 entity nhiều lần (vì bệnh án đề cập nhiều lần),
chỉ giữ lần đầu tiên để giảm hallucination penalties trong scoring.

QUAN TRỌNG: Chỉ deduplicate TRIỆU_CHỨNG và một số type dễ bị lặp.
CHẨN_ĐOÁN có thể xuất hiện 2 lần hợp lý (tiền sử + hiện tại).

Usage:
    uv run python scripts/dedup_text_entities.py --output output/ [--dry-run]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict


def dedup(output_dir: Path, dry_run: bool = False) -> None:
    """Deduplicate entities by text+type, keeping the one with best assertions."""
    output_files = sorted(
        [f for f in output_dir.glob("*.json") if f.stem.isdigit()],
        key=lambda x: int(x.stem),
    )
    print(f"[INFO] Processing {len(output_files)} output files...")

    total_removed = 0

    # Entity types where we keep only 1 occurrence per text
    # CHẨN_ĐOÁN: có thể xuất hiện 2 lần (tiền sử + hiện tại với assertion khác)
    # TRIỆU_CHỨNG: thường chỉ cần 1 lần theo gold
    DEDUP_SINGLE = {"TRIỆU_CHỨNG", "TÊN_XÉT_NGHIỆM"}
    # CHẨN_ĐOÁN: giữ max 2 (tiền sử + hiện tại) - assertions phải khác nhau
    DEDUP_DOUBLE = {"CHẨN_ĐOÁN", "THUỐC"}

    for fpath in output_files:
        try:
            entities = json.loads(fpath.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[WARN] {fpath.name}: parse error {exc}")
            continue

        if not isinstance(entities, list):
            continue

        original_count = len(entities)

        # For TRIỆU_CHỨNG: deduplicate by (text.lower(), type)
        # Keep the one with most non-empty assertions (more informative)
        seen_single: dict[tuple, list[int]] = defaultdict(list)
        seen_double: dict[tuple, list[int]] = defaultdict(list)

        for i, ent in enumerate(entities):
            if not isinstance(ent, dict):
                continue
            t = ent.get("type", "")
            txt = ent.get("text", "").strip().lower()
            key = (txt, t)

            if t in DEDUP_SINGLE:
                seen_single[key].append(i)
            elif t in DEDUP_DOUBLE:
                seen_double[key].append(i)

        to_remove: set[int] = set()

        # For DEDUP_SINGLE: keep only 1 occurrence per text+type
        # Priority: prefer assertion != [] over []
        for key, indices in seen_single.items():
            if len(indices) <= 1:
                continue
            # Sort by assertion richness (non-empty first), then by earliest position
            def priority(idx: int) -> tuple:
                e = entities[idx]
                assertions = e.get("assertions") or []
                cands = e.get("candidates") or []
                return (-len(assertions), -len(cands), e.get("position", [0, 0])[0])
            indices_sorted = sorted(indices, key=priority)
            # Keep first (best), remove rest
            for idx in indices_sorted[1:]:
                to_remove.add(idx)

        # For DEDUP_DOUBLE: keep max 2 if they have different assertions
        for key, indices in seen_double.items():
            if len(indices) <= 2:
                continue
            # Group by assertion set
            assert_groups: dict[str, list[int]] = defaultdict(list)
            for idx in indices:
                a = frozenset(entities[idx].get("assertions") or [])
                assert_groups[str(sorted(a))].append(idx)
            # Keep 1 per assertion group, max 2 groups
            kept = 0
            all_sorted = []
            for group_indices in assert_groups.values():
                all_sorted.append(group_indices[0])
                for extra in group_indices[1:]:
                    to_remove.add(extra)
            # Keep max 2 (earliest position for ties)
            all_sorted.sort(key=lambda i: entities[i].get("position", [0, 0])[0])
            for idx in all_sorted[2:]:
                to_remove.add(idx)

        if to_remove:
            entities = [e for i, e in enumerate(entities) if i not in to_remove]
            removed = original_count - len(entities)
            total_removed += removed
            if not dry_run:
                fpath.write_text(
                    json.dumps(entities, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            print(f"  [{fpath.stem}] Removed {removed} duplicates ({original_count} → {len(entities)})")

    print(f"[DONE] Total removed: {total_removed} duplicate entities")
    if dry_run:
        print("[DRY RUN] No files written.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    dedup(args.output, args.dry_run)


if __name__ == "__main__":
    main()
