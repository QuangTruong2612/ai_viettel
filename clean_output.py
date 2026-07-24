"""R39: Final cleanup of output files.

1. Strip `_booster` metadata flag from all entities (was internal marker for regex-added).
2. Ensure all `type` values are diacritics (CHẨN_ĐOÁN, TRIỆU_CHỨNG, ...) consistent with grader.
3. Strip any other internal `_` prefixed keys.

Usage:
    python clean_output.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"F:\AI_VIETTEL")))
from src.postprocess import _restore_diacritics_type


def main():
    output = Path(r"F:\AI_VIETTEL\output")
    n_booster_stripped = 0
    n_type_normalized = 0
    n_other_cleaned = 0
    n_total = 0

    for f in sorted(output.glob("*.json"), key=lambda p: int(p.stem)):
        data = json.load(open(f, encoding="utf-8"))
        if not data:
            continue
        file_changed = False
        for e in data:
            n_total += 1
            # Strip ALL internal keys (prefixed with `_`)
            keys_to_remove = [k for k in e if k.startswith("_") and k != "text"]
            if keys_to_remove:
                for k in keys_to_remove:
                    e.pop(k, None)
                    if k == "_booster":
                        n_booster_stripped += 1
                    else:
                        n_other_cleaned += 1
                file_changed = True
            # Normalize type to diacritics
            new_type = _restore_diacritics_type(e.get("type", ""))
            if new_type != e.get("type", ""):
                e["type"] = new_type
                n_type_normalized += 1
                file_changed = True
        if file_changed:
            with open(f, "w", encoding="utf-8") as out:
                json.dump(data, out, ensure_ascii=False, indent=2)

    print(f"Total entities processed: {n_total}")
    print(f"Stripped _booster:        {n_booster_stripped}")
    print(f"Stripped other _ keys:    {n_other_cleaned}")
    print(f"Type normalized (ASCII→diacritics): {n_type_normalized}")


if __name__ == "__main__":
    main()
