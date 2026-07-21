"""Dry-run test of assemble_record directly from src.postprocess to prove pipeline correctness."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.postprocess import assemble_record

def test_pipeline():
    inp_dir = Path("data/input") if Path("data/input").exists() else Path("input")
    
    # Test cases: Record 1, 5, 38
    for fid in [1, 5, 38]:
        ipath = inp_dir / f"{fid}.txt"
        if not ipath.exists():
            ipath = inp_dir / f"{fid}.json"
        if not ipath.exists():
            continue
            
        text = ipath.read_text(encoding="utf-8")
        raw_ents = json.loads(Path(f"output/{fid}.json").read_text(encoding="utf-8"))
        
        print(f"\n==================== ASSEMBLE_RECORD DRY RUN (Record {fid}) ====================")
        final_ents = assemble_record(text, raw_ents, retriever=None)
        
        for idx, e in enumerate(final_ents[:15]):
            txt = e.get("text")
            etype = e.get("type")
            pos = e.get("position")
            assertions = e.get("assertions")
            cands = e.get("candidates")
            print(f" [{idx:2d}] {etype:<20} | '{txt}' | pos={pos} | a={assertions} | c={cands}")

if __name__ == "__main__":
    test_pipeline()
