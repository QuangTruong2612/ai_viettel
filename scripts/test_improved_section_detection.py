"""Audit section header matches across the 100 test records."""

import re
import glob
from pathlib import Path

inp_dir = Path("data/input") if Path("data/input").exists() else Path("input")

section_headers = [
    (r"(?:^|\n)\s*(?:\d+\.\s*)?tiền sử[^\n]*", "tien_su"),
    (r"(?:^|\n)\s*(?:\d+\.\s*)?bệnh sử hiện tại[^\n]*", "hien_tai"),
    (r"(?:^|\n)\s*(?:\d+\.\s*)?lý do (?:nhập|vào) viện[^\n]*", "hien_tai"),
    (r"(?:^|\n)\s*(?:\d+\.\s*)?triệu chứng hiện tại[^\n]*", "hien_tai"),
    (r"(?:^|\n)\s*(?:\d+\.\s*)?đánh giá[^\n]*", "danh_gia"),
    (r"(?:^|\n)\s*(?:\d+\.\s*)?kết quả (?:khám|xét nghiệm)[^\n]*", "danh_gia"),
    (r"(?:^|\n)\s*(?:\d+\.\s*)?điều trị[^\n]*", "dieu_tri"),
]

for f in sorted(glob.glob("output/*.json"), key=lambda x: int(Path(x).stem))[:10]:
    rec_id = int(Path(f).stem)
    inp_path = inp_dir / f"{rec_id}.txt"
    if not inp_path.exists():
        inp_path = inp_dir / f"{rec_id}.json"
    if not inp_path.exists():
        continue

    text = inp_path.read_text(encoding="utf-8")
    print(f"=== RECORD {rec_id} SECTIONS ===")
    
    matches = []
    for pat, sec in section_headers:
        for m in re.finditer(pat, text, re.IGNORECASE):
            matches.append((m.start(), m.group(0).strip(), sec))
            
    matches.sort(key=lambda x: x[0])
    for start, line, sec in matches:
        print(f"  pos {start:4d} [{sec:<8s}]: '{line}'")
    print()
