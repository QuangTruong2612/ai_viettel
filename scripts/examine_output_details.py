"""Examine entities in output files to inspect NER texts, types, positions, assertions, candidates."""

from __future__ import annotations

import json
from pathlib import Path

def examine():
    out_dir = Path("output")
    for fid in [1, 2, 5, 12, 20, 25, 38]:
        fpath = out_dir / f"{fid}.json"
        if not fpath.exists():
            continue
        print(f"============================ RECORD {fid} ============================")
        ents = json.loads(fpath.read_text(encoding="utf-8"))
        for idx, e in enumerate(ents):
            txt = e.get("text")
            etype = e.get("type")
            pos = e.get("position")
            assertions = e.get("assertions")
            cands = e.get("candidates")
            print(f" [{idx:2d}] {etype:<20} | '{txt}' | pos={pos} | a={assertions} | c={cands}")

if __name__ == "__main__":
    examine()
