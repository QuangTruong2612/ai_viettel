"""Clean noise and section headers extracted as NER entities."""

import json
import re
import glob
from pathlib import Path

HEADER_NOISE_PATTERNS = re.compile(
    r"^(?:"
    r"tình\s+trạng\s+ngay\s+trước\s+khi\s+nhập\s+viện|"
    r"lý\s+do\s+nhập\s+viện|"
    r"thời\s+điểm\s+khởi\s+phát.*|"
    r"các\s+triệu\s+chứng\s+hiện\s+tại|"
    r"đặc\s+điểm\s+triệu\s+chứng.*|"
    r"các\s+diễn\s+biến\s+trước.*|"
    r"kết\s+quả\s+khám\s+lâm\s+sàng|"
    r"kết\s+quả\s+xét\s+nghiệm|"
    r"kết\s+quả\s+chẩn\s+đoán\s+hình\s+ảnh|"
    r"các\s+kết\s+quả\s+chẩn\s+đoán\s+khác|"
    r"đánh\s+giá\s+tại\s+bệnh\s+viện|"
    r"tiền\s+sử\s+bệnh|"
    r"thuốc\s+trước\s+khi\s+nhập\s+viện|"
    r"các\s+bệnh\s+lý\s+mạn\s+tính|"
    r"bệnh\s+sử\s+hiện\s+tại"
    r")$",
    re.IGNORECASE | re.UNICODE,
)

def clean_ner_noise(output_dir: Path) -> None:
    output_files = sorted(
        [f for f in output_dir.glob("*.json") if f.stem.isdigit()],
        key=lambda x: int(x.stem),
    )
    print(f"[INFO] Cleaning NER noise on {len(output_files)} files...")

    total_removed = 0

    for fpath in output_files:
        entities = json.loads(fpath.read_text(encoding="utf-8"))
        if not isinstance(entities, list):
            continue

        valid = []
        for ent in entities:
            if not isinstance(ent, dict):
                continue
            txt = str(ent.get("text", "")).strip()

            # Filter out header noise
            if HEADER_NOISE_PATTERNS.match(txt):
                total_removed += 1
                continue

            valid.append(ent)

        if len(valid) < len(entities):
            fpath.write_text(json.dumps(valid, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] Removed {total_removed} section header noise entities!")

if __name__ == "__main__":
    clean_ner_noise(Path("output"))
