"""R39: Full pipeline — recall booster + assertion enrichment for all outputs."""
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
    _enrich_assertions,
)
import re

# Use SYMPTOM_NOT_DISEASE pattern from fix_ner_types.py
_SYMPTOM_NOT_DISEASE = re.compile(
    r"^(?:sốt\s*cao|vàng\s*da|vàng\s*mắt|khó\s*thở|đau\s*(?:đầu|bụng|ngực|lưng|họng|cổ|chân|tay)|"
    r"buồn\s*nôn|\bnôn\b(?!\s*ra)|mệt\s*mỏi|chóng\s*mặt|hoa\s*mắt|"
    r"\bho\b(?!\s*ra)|phù\s+\w+|\bngứa\b|phát\s*ban|tức\s*ngực|đánh\s*trống\s*ngực|"
    r"mất\s*ngủ|tê\s+\w+|yếu\s+\w+)$",
    re.IGNORECASE | re.UNICODE,
)

OUTPUT = Path(r"F:\AI_VIETTEL\output")
INPUT = Path(r"F:\AI_VIETTEL\input")


def main():
    stats = {
        "files_processed": 0,
        "types_normalized": 0,
        "positions_fixed": 0,
        "positions_dropped": 0,
        "chatbot_dropped": 0,
        "narrative_dropped": 0,
        "type_reclassified": 0,
        "booster_added": 0,
        "isHistorical_added": 0,
        "isFamily_added": 0,
        "isNegated_added": 0,
        "entities_before": 0,
        "entities_after": 0,
    }

    for inp_path in sorted(INPUT.glob("*.txt"), key=lambda p: int(p.stem)):
        fid = inp_path.stem
        out_p = OUTPUT / f"{fid}.json"
        if not out_p.exists():
            continue
        text = inp_path.read_text(encoding="utf-8")
        data = json.load(open(out_p, encoding="utf-8"))
        if not data and len(text) < 200:
            stats["files_processed"] += 1
            continue
        stats["entities_before"] += len(data)

        # Snapshot assertions BEFORE
        before_his = sum(1 for e in data if "isHistorical" in (e.get("assertions") or []))
        before_fam = sum(1 for e in data if "isFamily" in (e.get("assertions") or []))
        before_neg = sum(1 for e in data if "isNegated" in (e.get("assertions") or []))

        new_data = []
        for ent in data:
            new_type = _normalize_type_to_ascii(ent.get("type", ""))
            if new_type != ent.get("type", ""):
                stats["types_normalized"] += 1
            ent["type"] = new_type
            text_e = str(ent.get("text", "")).strip()

            if _is_overly_long_narrative(text_e, new_type):
                stats["narrative_dropped"] += 1
                continue

            if _is_chatbot_artifact(text_e):
                stats["chatbot_dropped"] += 1
                continue

            # Fix obvious type errors
            if new_type == "CHAN_DOAN" and _SYMPTOM_NOT_DISEASE.match(text_e):
                ent["type"] = "TRIEU_CHUNG"
                new_type = "TRIEU_CHUNG"
                stats["type_reclassified"] += 1

            old_pos = ent.get("position", [])
            recovered = _enforce_position_strict(text, ent)
            if recovered is None:
                stats["positions_dropped"] += 1
                continue
            if recovered.get("position") != old_pos:
                stats["positions_fixed"] += 1
            new_data.append(recovered)

        # Recall booster
        boosted = boost_recall_ner(text, new_data)
        if boosted:
            new_data.extend(boosted)
            stats["booster_added"] += len(boosted)

        # Assertion enrichment
        _enrich_assertions(text, new_data)

        # Snapshot assertions AFTER
        after_his = sum(1 for e in new_data if "isHistorical" in (e.get("assertions") or []))
        after_fam = sum(1 for e in new_data if "isFamily" in (e.get("assertions") or []))
        after_neg = sum(1 for e in new_data if "isNegated" in (e.get("assertions") or []))

        stats["isHistorical_added"] += (after_his - before_his)
        stats["isFamily_added"] += (after_fam - before_fam)
        stats["isNegated_added"] += (after_neg - before_neg)

        stats["entities_after"] += len(new_data)
        # Strip _booster flag if any
        for e in new_data:
            e.pop("_booster", None)
        new_data.sort(key=lambda e: e.get("position", [0, 0])[0])

        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(new_data, f, ensure_ascii=False, indent=2)
        stats["files_processed"] += 1

    print("=" * 60)
    print("R39 FULL PIPELINE (NER + ASSERTIONS)")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k:30s} = {v}")


if __name__ == "__main__":
    main()
