"""R39: Fix obvious type errors in booster entities.
Symptoms classified as CHAN_DOAN → reclassify to TRIEU_CHUNG.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"F:\AI_VIETTEL")))

OUTPUT = Path(r"F:\AI_VIETTEL\output")
INPUT = Path(r"F:\AI_VIETTEL\input")


# Patterns that should be TRIEU_CHUNG (symptoms), not CHAN_DOAN
_SYMPTOM_NOT_DISEASE = re.compile(
    r"^(?:sốt\s*cao|"
    r"vàng\s*da|"
    r"vàng\s*mắt|"
    r"khó\s*thở|"
    r"đau\s*(?:đầu|bụng|ngực|lưng|họng|cổ|chân|tay)|"
    r"buồn\s*nôn|"
    r"\bnôn\b(?!\s*ra)|"
    r"mệt\s*mỏi|"
    r"chóng\s*mặt|"
    r"hoa\s*mắt|"
    r"\bho\b(?!\s*ra)|"
    r"phù\s+\w+|"
    r"\bngứa\b|"
    r"phát\s*ban|"
    r"tức\s*ngực|"
    r"đánh\s*trống\s*ngực|"
    r"mất\s*ngủ|"
    r"tê\s+\w+|"
    r"yếu\s+\w+)$",
    re.IGNORECASE | re.UNICODE,
)


def main():
    fixed = 0
    for f in sorted(OUTPUT.glob("*.json"), key=lambda p: int(p.stem)):
        data = json.load(open(f, encoding="utf-8"))
        changed = False
        for ent in data:
            if ent.get("type") != "CHAN_DOAN":
                continue
            text = ent.get("text", "").strip()
            if _SYMPTOM_NOT_DISEASE.match(text):
                ent["type"] = "TRIEU_CHUNG"
                fixed += 1
                changed = True
        if changed:
            with open(f, "w", encoding="utf-8") as out:
                json.dump(data, out, ensure_ascii=False, indent=2)
    print(f"Fixed {fixed} misclassified SYMPTOM → TRIEU_CHUNG")


if __name__ == "__main__":
    main()
