"""R39: Enrich assertions on existing output files using EXTENDED patterns.

Run:
    python enrich_assertions.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"F:\AI_VIETTEL")))
from src.postprocess import _enrich_assertions


def main():
    output_dir = Path(r"F:\AI_VIETTEL\output")
    input_dir = Path(r"F:\AI_VIETTEL\input")

    stats = {
        "files_processed": 0,
        "entities_with_assertions": 0,
        "isHistorical_added": 0,
        "isFamily_added": 0,
        "isNegated_added": 0,
    }

    for fout in sorted(output_dir.glob("*.json"), key=lambda p: int(p.stem)):
        fid = fout.stem
        data = json.load(open(fout, encoding="utf-8"))
        if not data:
            continue
        inp_p = input_dir / f"{fid}.txt"
        if not inp_p.exists():
            continue
        inp = inp_p.read_text(encoding="utf-8")

        # Snapshot before
        before_his = sum(1 for e in data if "isHistorical" in (e.get("assertions") or []))
        before_fam = sum(1 for e in data if "isFamily" in (e.get("assertions") or []))
        before_neg = sum(1 for e in data if "isNegated" in (e.get("assertions") or []))

        _enrich_assertions(inp, data)

        # Snapshot after
        after_his = sum(1 for e in data if "isHistorical" in (e.get("assertions") or []))
        after_fam = sum(1 for e in data if "isFamily" in (e.get("assertions") or []))
        after_neg = sum(1 for e in data if "isNegated" in (e.get("assertions") or []))

        stats["isHistorical_added"] += (after_his - before_his)
        stats["isFamily_added"] += (after_fam - before_fam)
        stats["isNegated_added"] += (after_neg - before_neg)

        with open(fout, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        stats["files_processed"] += 1

    print("=" * 60)
    print("R39 ASSERTION ENRICHMENT")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k:30s} = {v}")


if __name__ == "__main__":
    main()
