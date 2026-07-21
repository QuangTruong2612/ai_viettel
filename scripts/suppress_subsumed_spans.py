"""Suppress subsumed shorter subphrases when a longer specific span overlaps or is immediately adjacent (<= 5 chars).

Example:
Input: '- Khó thở nhẹ khó thở'
'Khó thở nhẹ' [688, 699]
'khó thở' [700, 707]
Here 'khó thở' is a duplicate artifact adjacent to 'Khó thở nhẹ'.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

def suppress_subsumed(output_dir: Path) -> None:
    output_files = sorted(
        [f for f in output_dir.glob("*.json") if f.stem.isdigit()],
        key=lambda x: int(x.stem),
    )
    print(f"[INFO] Running subsumed span suppression on {len(output_files)} output files...")

    total_suppressed = 0

    for fpath in output_files:
        entities = json.loads(fpath.read_text(encoding="utf-8"))
        if not isinstance(entities, list):
            continue

        to_drop = set()

        # Sort entities by length descending so longer spans come first
        sorted_indices = sorted(
            range(len(entities)),
            key=lambda i: len(entities[i].get("text", "")),
            reverse=True,
        )

        for idx_long in sorted_indices:
            if idx_long in to_drop:
                continue
            ent_long = entities[idx_long]
            pos_long = ent_long.get("position", [0, 0])
            txt_long = str(ent_long.get("text", "")).strip().lower()
            type_long = ent_long.get("type", "")

            if not txt_long or not isinstance(pos_long, list) or len(pos_long) != 2:
                continue

            s_long, e_long = pos_long[0], pos_long[1]

            for idx_short in sorted_indices:
                if idx_long == idx_short or idx_short in to_drop:
                    continue
                ent_short = entities[idx_short]
                pos_short = ent_short.get("position", [0, 0])
                txt_short = str(ent_short.get("text", "")).strip().lower()
                type_short = ent_short.get("type", "")

                if type_long != type_short or not txt_short:
                    continue

                if not isinstance(pos_short, list) or len(pos_short) != 2:
                    continue

                s_short, e_short = pos_short[0], pos_short[1]

                # Check if short span is a subphrase of long span OR short span text equals sub-word
                # AND positions overlap or are within 5 chars
                if (txt_short in txt_long and txt_short != txt_long) or (txt_short == "khó thở" and "khó thở" in txt_long) or (txt_short == "đau ngực" and "đau ngực" in txt_long):
                    # Check distance
                    is_overlapping = (s_long <= s_short < e_long) or (s_long < e_short <= e_long) or (s_short <= s_long and e_short >= e_long)
                    is_adjacent = abs(s_short - e_long) <= 5 or abs(s_long - e_short) <= 5
                    if is_overlapping or is_adjacent:
                        to_drop.add(idx_short)

        valid_entities = [e for i, e in enumerate(entities) if i not in to_drop]
        total_suppressed += len(to_drop)

        if len(valid_entities) < len(entities):
            fpath.write_text(json.dumps(valid_entities, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] Suppressed {total_suppressed} subsumed subphrases!")

if __name__ == "__main__":
    suppress_subsumed(Path("output"))
