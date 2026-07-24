"""R39: Empty-output rescue for files 75, 96 — try regex fallback to recover entities.

Strategy:
- If postprocess returned [], try REGEX-BASED fallback with curated VN medical patterns.
- Reuse _is_chatbot_artifact + _is_overly_long_narrative filters.
- Generate entities with simple position [start, end] match.

NOTE: This is FALLBACK ONLY when LLM returns empty. Quality sẽ thấp hơn LLM,
chỉ dùng để cứu điểm khỏi bị 0.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"F:\AI_VIETTEL")))
from src.postprocess import (
    _is_chatbot_artifact,
    _is_overly_long_narrative,
    _normalize_type_to_ascii,
)


# Curated VN medical entity patterns (chi dùng cho FALLBACK khi LLM empty)
_FALLBACK_PATTERNS = [
    # Disease prefixes (CHAN_DOAN)
    (r"(?:viêm|suy|thoái\s+hóa|ung\s+thư|khối\s+u|nhiễm\s+trùng|nhiễm\s+khuẩn|"
     r"rối\s+loạn|tắc|hẹp|tràn\s+dịch|tràn\s+khí|xơ\s+|gãy|thoát\s+vị|phình|"
     r"suy\s+|tăng\s+|giảm\s+|hở\s+|bệnh\s+lý)\s+[\w\s,()]{4,40}", "CHAN_DOAN"),
    # Symptoms (TRIEU_CHUNG)
    (r"(?:đau|khó\s+thở|sốt|mệt|chóng\s+mặt|buồn\s+nôn|nôn|ho|tức\s+ngực|"
     r"khó\s+nuốt|đánh\s+trống\s+ngực|tê|yếu|ngứa|phù\s+|phát\s+ban|nổi\s+mề\s+đay)"
     r"(?:\s+\w+){0,3}", "TRIEU_CHUNG"),
    # Test names (TEN_XET_NGHIEM) — short list
    (r"\b(?:chụp\s+X[-\s]?quang|siêu\s+âm|điện\s+tâm\s+đồ|ECG|EKG|MRI|CT\s+scan|"
     r"xét\s+nghiệm|công\s+thức\s+máu|sinh\s+hóa|nước\s+tiểu)\b", "TEN_XET_NGHIEM"),
    # Common drug names (THUOC) — BRAND only
    (r"\b(?:aspirin|paracetamol|amoxicillin|metformin|insulin|atenolol|metoprolol|"
     r"amlodipine|furosemide|prednisolone|ibuprofen|trimetazidine|nitroglycerin)\b", "THUOC"),
]


def extract_fallback_entities(text: str) -> list[dict]:
    """Regex-based fallback extraction. Returns list of entities with type inferred."""
    out = []
    seen_positions = set()
    for pat, etype in _FALLBACK_PATTERNS:
        for m in re.finditer(pat, text, re.IGNORECASE | re.UNICODE):
            s, e = m.start(), m.end()
            if (s, e) in seen_positions:
                continue
            seen_positions.add((s, e))
            t = m.group(0).strip()
            if not t or len(t) > 80:
                continue
            if _is_chatbot_artifact(t) or _is_overly_long_narrative(t, etype):
                continue
            # Lowercase first char if it's a normal symptom (lowercase in input)
            out.append({
                "text": t,
                "type": _normalize_type_to_ascii(etype),
                "position": [s, e],
                "assertions": [],
            })
    out.sort(key=lambda x: x["position"][0])
    return out


def main():
    output_dir = Path(r"F:\AI_VIETTEL\output")
    input_dir = Path(r"F:\AI_VIETTEL\input")

    empty_files = ["75", "96"]
    for fid in empty_files:
        out_p = output_dir / f"{fid}.json"
        in_p = input_dir / f"{fid}.txt"
        if not in_p.exists():
            print(f"  Skip {fid}: input missing")
            continue
        text = in_p.read_text(encoding="utf-8")
        ents = extract_fallback_entities(text)
        if ents:
            print(f"  {fid}.txt → {len(ents)} fallback entities extracted")
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(ents, f, ensure_ascii=False, indent=2)
        else:
            print(f"  {fid}.txt → no fallback entities found")


if __name__ == "__main__":
    main()
