"""R39: Apply recall booster + final type fixes to all output files."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"F:\AI_VIETTEL")))
from src.postprocess import (
    _normalize_type_to_ascii,
    _enforce_position_strict,
    _is_chatbot_artifact,
    _is_overly_long_narrative,
    boost_recall_ner,
)


def main():
    output_dir = Path(r"F:\AI_VIETTEL\output")
    input_dir = Path(r"F:\AI_VIETTEL\input")

    stats = {
        "files_processed": 0,
        "files_skipped_empty": 0,
        "types_normalized": 0,
        "positions_fixed": 0,
        "positions_dropped": 0,
        "chatbot_dropped": 0,
        "narrative_dropped": 0,
        "booster_added": 0,
        "entities_before": 0,
        "entities_after": 0,
    }

    for fout in sorted(output_dir.glob("*.json"), key=lambda p: int(p.stem)):
        fid = fout.stem
        data = json.load(open(fout, encoding="utf-8"))
        if not data:
            stats["files_skipped_empty"] += 1
            continue
        inp_file = input_dir / f"{fid}.txt"
        if not inp_file.exists():
            continue
        inp = inp_file.read_text(encoding="utf-8")
        stats["entities_before"] += len(data)

        new_data = []
        for ent in data:
            new_type = _normalize_type_to_ascii(ent.get("type", ""))
            if new_type != ent.get("type", ""):
                stats["types_normalized"] += 1
            ent["type"] = new_type
            text = str(ent.get("text", "")).strip()

            if _is_overly_long_narrative(text, new_type):
                stats["narrative_dropped"] += 1
                continue

            if _is_chatbot_artifact(text):
                stats["chatbot_dropped"] += 1
                continue

            old_pos = ent.get("position", [])
            recovered = _enforce_position_strict(inp, ent)
            if recovered is None:
                stats["positions_dropped"] += 1
                continue
            if recovered.get("position") != old_pos:
                stats["positions_fixed"] += 1
            new_data.append(recovered)

        # Recall booster
        boosted = boost_recall_ner(inp, new_data)
        if boosted:
            new_data.extend(boosted)
            stats["booster_added"] += len(boosted)

        stats["entities_after"] += len(new_data)
        with open(fout, "w", encoding="utf-8") as f:
            json.dump(new_data, f, ensure_ascii=False, indent=2)
        stats["files_processed"] += 1

    print("=" * 60)
    print("R39 RECALL BOOSTER APPLIED")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k:25s} = {v}")
    print()
    print(f"  ENTITIES: {stats['entities_before']} → {stats['entities_after']}")
    print(f"  BOOST:    +{stats['booster_added']} new entities from recall patterns")
    return 0


if __name__ == "__main__":
    sys.exit(main())
