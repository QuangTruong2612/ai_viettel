"""R39: Retroactive candidate fix-up — re-lookup ICD codes for entities
using the EXPANDED _icd_vn_to_codes map in src/icd_rag.py.

Files có candidates SAI (vd M08.1/Q55.0 cho G6PD) sẽ được thay bằng direct map.

Usage:
    python retroactive_candidate_fix.py
"""
import json
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(r"F:\AI_VIETTEL")))
from src.icd_rag import ICDRetriever  # noqa: E402


def main():
    output_dir = Path(r"F:\AI_VIETTEL\output")

    # Force load icd data
    print("[init] Loading ICD retriever...")
    try:
        retriever = ICDRetriever()
        # Try to load ICD data
        retriever._ensure_loaded()
        print(f"[init] ICDRetriever loaded, candidate map has {len(getattr(retriever, '_icd_vn_to_codes', {}))} entries")
    except Exception as e:
        print(f"[init] WARNING: ICDRetriever failed to load: {e}")
        print("[init] Will use a SIMPLE lookup against the in-memory map only.")
        retriever = None

    # Build a quick direct-map lookup (read directly from icd_rag.py source vars)
    direct_map = {}
    if retriever is not None and hasattr(retriever, "_icd_vn_to_codes"):
        direct_map = {k.lower(): v for k, v in retriever._icd_vn_to_codes.items()}
        print(f"[init] Direct map: {len(direct_map)} entries")

    total_fixed = 0
    total_no_cand = 0
    total_already_good = 0
    type_counts = Counter()

    for fout in sorted(output_dir.glob("*.json"), key=lambda p: int(p.stem)):
        fid = fout.stem
        data = json.load(open(fout, encoding="utf-8"))
        if not data:
            continue

        file_changed = False
        for ent in data:
            etype = ent.get("type", "")
            if etype in ("THUOC", "TRIEU_CHUNG"):
                continue  # skip non-CHAN_DOAN
            if etype != "CHAN_DOAN":
                continue
            text = str(ent.get("text", "")).strip()
            if not text:
                continue
            type_counts[etype] += 1
            cands = ent.get("candidates", [])
            if not cands:
                total_no_cand += 1

            # Try direct dict lookup
            text_lower = text.lower().strip()
            new_cands = direct_map.get(text_lower)

            if new_cands and set(new_cands) != set(cands):
                ent["candidates"] = new_cands
                total_fixed += 1
                file_changed = True
            elif cands:
                total_already_good += 1

        if file_changed:
            with open(fout, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 60)
    print("RETROACTIVE CANDIDATE FIX")
    print("=" * 60)
    print(f"  Diagnosis entities scanned:   {sum(type_counts.values())}")
    print(f"  Entities fixed (candidates):  {total_fixed}")
    print(f"  Entities already had good:    {total_already_good}")
    print(f"  Entities with no candidates:  {total_no_cand}")


if __name__ == "__main__":
    main()
