"""Apply R39 fixes to existing output/ files (idempotent: safe to re-run).

Run:
    python apply_r39_fixes.py

This rewrites all output/*.json with:
1. Type normalized: diacritics → ASCII (KẾT_QUẢ → KET_QUA, etc.)
2. Positions auto-recovered via _enforce_position_strict
3. Chatbot artifacts dropped
4. Overly long narratives dropped

Skip files where LLM returned [] (files 75, 96) — those need a re-run of LLM.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"F:\AI_VIETTEL")))
from src.postprocess import (
    _normalize_type_to_ascii,
    _enforce_position_strict,
    _is_chatbot_artifact,
    _is_overly_long_narrative,
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

        stats["entities_after"] += len(new_data)
        # Write back (always — to update types)
        with open(fout, "w", encoding="utf-8") as f:
            json.dump(new_data, f, ensure_ascii=False, indent=2)
        stats["files_processed"] += 1

    print("=" * 60)
    print("R39 FIXES APPLIED")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k:25s} = {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
