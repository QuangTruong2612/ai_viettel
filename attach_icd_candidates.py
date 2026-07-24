"""R39: Attach ICD candidates using ONLY direct dict lookup (no BGE-M3 loading).

Faster — uses _icd_vn_to_codes directly. No embedding inference.

Usage:
    python attach_icd_candidates.py
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"F:\AI_VIETTEL")))


def main():
    # Load ICD direct map via substring / L0 match (no model needed)
    # Read from src/icd_rag.py — load class but skip _ensure_loaded
    import src.icd_rag as icd_module

    # Get _icd_vn_to_codes (populated when ICDRetriever.__init__ runs)
    # Run __init__ with use_hybrid=False to skip model load
    from src.icd_rag import ICDRetriever

    print("[init] ICDRetriever(use_hybrid=False) — no BGE-M3 model load")
    icd = ICDRetriever(use_hybrid=False)
    direct_map = getattr(icd, "_icd_vn_to_codes", {})
    print(f"[init] Direct ICD map: {len(direct_map)} entries")

    output_dir = Path(r"F:\AI_VIETTEL\output")
    stats = {
        "files_processed": 0,
        "chan_doan_entities": 0,
        "candidates_attached": 0,
        "candidates_already_present": 0,
        "no_match": 0,
    }

    for fout in sorted(output_dir.glob("*.json"), key=lambda p: int(p.stem)):
        fid = fout.stem
        data = json.load(open(fout, encoding="utf-8"))
        if not data:
            continue

        changed = False
        for e in data:
            if e.get("type") != "CHẨN_ĐOÁN":
                continue
            stats["chan_doan_entities"] += 1
            text = str(e.get("text", "")).strip()
            existing = e.get("candidates", [])
            if existing:
                stats["candidates_already_present"] += 1
                continue

            # Direct map lookup
            norm_text = re.sub(r"\s+", " ", text.lower()).strip()
            codes = direct_map.get(norm_text)

            # Try ICD via substring (longer keys first, with word-boundary check)
            if not codes:
                # Sort keys by length DESC → prefer longest match
                # Use word-boundary `\b` to avoid matching inside other words
                for key in sorted(direct_map.keys(), key=len, reverse=True):
                    if len(key) >= 6 and re.search(r"\b" + re.escape(key) + r"\b", norm_text):
                        codes = direct_map[key]
                        break

            if codes:
                e["candidates"] = codes
                stats["candidates_attached"] += 1
                changed = True
            else:
                stats["no_match"] += 1

        if changed:
            with open(fout, "w", encoding="utf-8") as out:
                json.dump(data, out, ensure_ascii=False, indent=2)
        stats["files_processed"] += 1

    print()
    print("=" * 60)
    print("ICD CANDIDATE ATTACH (direct map)")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k:35s} = {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
