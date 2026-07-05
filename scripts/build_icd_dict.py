"""Build ICD-10 index từ seed JSONL → data/icd_index.json.

Cú pháp:
    python scripts/build_icd_dict.py [--seed data/icd_seed.jsonl] [--out data/icd_index.json]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.icd_rag import build_from_seed  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=Path, default=Path("data/icd_seed.jsonl"))
    p.add_argument("--out", type=Path, default=Path("data/icd_index.json"))
    args = p.parse_args(argv)

    if not args.seed.exists():
        print(f"Seed file không tồn tại: {args.seed}", file=sys.stderr)
        return 1

    idx = build_from_seed(args.seed, args.out)
    print(f"Built ICD index: {len(idx.names)} entries, {len(idx.exact)} keys")
    print(f"Saved → {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
