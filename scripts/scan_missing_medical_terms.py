"""Dictionary scanner to recover missing high-confidence medical entities (diseases, symptoms, drugs).

Scans input_text for known clinical terms and adds any missing mentions to output/*.json.
"""

from __future__ import annotations

import json
import re
import glob
from pathlib import Path

inp_dir = Path("data/input") if Path("data/input").exists() else Path("input")

# Whitelist of high-precision disease/symptom/drug terms that are ALWAYS valid entities
RECOVERY_TERMS = [
    # Diseases (CHẨN_ĐOÁN)
    ("viêm phổi mắc phải cộng đồng", "CHẨN_ĐOÁN"),
    ("viêm phế quản", "CHẨN_ĐOÁN"),
    ("viêm phổi", "CHẨN_ĐOÁN"),
    ("phù phổi", "CHẨN_ĐOÁN"),
    ("ung thư vú", "CHẨN_ĐOÁN"),
    ("ung thư tuyến", "CHẨN_ĐOÁN"),
    ("ung thư phổi", "CHẨN_ĐOÁN"),
    ("viêm túi mật", "CHẨN_ĐOÁN"),
    ("viêm ruột thừa", "CHẨN_ĐOÁN"),
    ("viêm dạ dày", "CHẨN_ĐOÁN"),
    ("viêm bể thận", "CHẨN_ĐOÁN"),
    ("viêm mô tế bào", "CHẨN_ĐOÁN"),
    ("sỏi ống mật", "CHẨN_ĐOÁN"),
    ("sỏi túi mật", "CHẨN_ĐOÁN"),
    ("sỏi thận", "CHẨN_ĐOÁN"),
    ("sỏi bàng quang", "CHẨN_ĐOÁN"),
    ("tăng huyết áp", "CHẨN_ĐOÁN"),
    ("đái tháo đường type 2", "CHẨN_ĐOÁN"),
    ("đái tháo đường type 1", "CHẨN_ĐOÁN"),
    ("đái tháo đường", "CHẨN_ĐOÁN"),
    ("nhồi máu cơ tim", "CHẨN_ĐOÁN"),
    ("nhồi máu não", "CHẨN_ĐOÁN"),
    ("suy tim độ III", "CHẨN_ĐOÁN"),
    ("suy tim", "CHẨN_ĐOÁN"),
    ("rung nhĩ", "CHẨN_ĐOÁN"),
    ("xơ gan do rượu", "CHẨN_ĐOÁN"),
    ("xơ gan", "CHẨN_ĐOÁN"),
    ("rối loạn lipid máu", "CHẨN_ĐOÁN"),
    ("bệnh thận mạn", "CHẨN_ĐOÁN"),
    ("suy thận mạn", "CHẨN_ĐOÁN"),

    # Symptoms (TRIỆU_CHỨNG)
    ("buồn nôn", "TRIỆU_CHỨNG"),
    ("nôn ói", "TRIỆU_CHỨNG"),
    ("nôn", "TRIỆU_CHỨNG"),
    ("chóng mặt", "TRIỆU_CHỨNG"),
    ("choáng váng", "TRIỆU_CHỨNG"),
    ("vã mồ hôi", "TRIỆU_CHỨNG"),
    ("đổ mồ hôi", "TRIỆU_CHỨNG"),
    ("sốt cao", "TRIỆU_CHỨNG"),
    ("sốt", "TRIỆU_CHỨNG"),
    ("ho khạc đờm", "TRIỆU_CHỨNG"),
    ("ho ra máu", "TRIỆU_CHỨNG"),
    ("ho", "TRIỆU_CHỨNG"),
    ("mệt mỏi", "TRIỆU_CHỨNG"),
    ("đau ngực trái", "TRIỆU_CHỨNG"),
    ("đau ngực", "TRIỆU_CHỨNG"),
    ("khó thở nhẹ", "TRIỆU_CHỨNG"),
    ("khó thở", "TRIỆU_CHỨNG"),
    ("đánh trống ngực", "TRIỆU_CHỨNG"),
]

def scan_missing_terms(output_dir: Path) -> None:
    output_files = sorted(
        [f for f in output_dir.glob("*.json") if f.stem.isdigit()],
        key=lambda x: int(x.stem),
    )
    print(f"[INFO] Scanning missing high-precision terms across {len(output_files)} files...")

    total_added = 0

    for fpath in output_files:
        rec_id = int(fpath.stem)
        inp_path = inp_dir / f"{rec_id}.txt"
        if not inp_path.exists():
            inp_path = inp_dir / f"{rec_id}.json"
        if not inp_path.exists():
            continue

        input_text = inp_path.read_text(encoding="utf-8")
        entities = json.loads(fpath.read_text(encoding="utf-8"))
        if not isinstance(entities, list):
            continue

        existing_spans = set()
        for e in entities:
            pos = e.get("position")
            if isinstance(pos, list) and len(pos) == 2:
                existing_spans.add((pos[0], pos[1]))

        added_for_file = 0

        for term, etype in RECOVERY_TERMS:
            # Search all occurrences of term in input_text
            pattern = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE | re.UNICODE)
            for m in pattern.finditer(input_text):
                s, e = m.start(), m.end()
                actual_text = input_text[s:e]

                # Check if this exact span or overlapping span already exists
                overlap = any(max(s, es) < min(e, ee) for es, ee in existing_spans)
                if not overlap:
                    # New entity found!
                    new_ent = {
                        "text": actual_text,
                        "type": etype,
                        "position": [s, e],
                        "assertions": [],
                        "candidates": [],
                    }
                    entities.append(new_ent)
                    existing_spans.add((s, e))
                    added_for_file += 1
                    total_added += 1

        if added_for_file > 0:
            # Sort by position start
            entities.sort(key=lambda x: x.get("position", [0, 0])[0])
            fpath.write_text(json.dumps(entities, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  [{rec_id}] Recovered +{added_for_file} missing entities")

    print(f"[DONE] Recovered total +{total_added} missing high-precision entities!")

if __name__ == "__main__":
    scan_missing_terms(Path("output"))
