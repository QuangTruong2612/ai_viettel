"""Validate output đã generate: schema + spot check."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.postprocess import validate_output  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("output"))
    args = p.parse_args()

    files = sorted(args.input.glob("*.json"), key=lambda x: int(x.stem))
    if not files:
        print(f"No JSON files in {args.input}")
        return 1

    bad = 0
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"❌ {f.name}: parse fail {exc}")
            bad += 1
            continue
        if not validate_output(data):
            print(f"❌ {f.name}: schema invalid")
            bad += 1
        else:
            n_drug = sum(1 for e in data if e.get("type") == "THUỐC")
            n_symp = sum(1 for e in data if e.get("type") == "TRIỆU_CHỨNG")
            has_hist = sum(1 for e in data if "isHistorical" in (e.get("assertions") or []))
            print(f"✅ {f.name}: drugs={n_drug} symp={n_symp} historical={has_hist}")
    if bad:
        print(f"\n{bad} file không hợp lệ.")
    else:
        print(f"\nTất cả {len(files)} file hợp lệ.")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
