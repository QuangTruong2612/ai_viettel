"""R39: Restore diacritics types in all output files (revert ASCII normalization).

Lý do: grader hiện tại dùng diacritics enum (THUỐC, CHẨN_ĐOÁN, ...).
Trước đây postprocess strip diacritics sang ASCII → fail validation.

Usage:
    python restore_diacritics.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"F:\AI_VIETTEL")))
from src.postprocess import _restore_diacritics_type


def main():
    output = Path(r"F:\AI_VIETTEL\output")
    n_changed = 0
    n_total = 0

    for f in sorted(output.glob("*.json"), key=lambda p: int(p.stem)):
        data = json.load(open(f, encoding="utf-8"))
        for e in data:
            n_total += 1
            new_type = _restore_diacritics_type(e.get("type", ""))
            if new_type != e.get("type", ""):
                e["type"] = new_type
                n_changed += 1
        with open(f, "w", encoding="utf-8") as out:
            json.dump(data, out, ensure_ascii=False, indent=2)

    print(f"Total entities: {n_total}")
    print(f"Restored to diacritics: {n_changed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
