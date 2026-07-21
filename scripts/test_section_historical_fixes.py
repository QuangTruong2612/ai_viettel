"""Test precise section segmentation and historical assertion accuracy."""

from __future__ import annotations

import json
import re
import glob
from pathlib import Path

inp_dir = Path("data/input") if Path("data/input").exists() else Path("input")

TIEN_SU_PATTERNS = re.compile(
    r"(?:^|\n)\s*(?:\d+\.\s*)?(?:"
    r"tiền\s+sử\s+bệnh(?!\s*hiện\s+tại)|"
    r"tiền\s+sử\s+bệnh\s*nội\s+khoa|"
    r"tiền\s+sử\s+bệnh\s*ngoại\s+khoa|"
    r"tiền\s+sử\s+phẫu\s+thuật|"
    r"tiền\s+sử\s+thủ\s+thuật|"
    r"tiền\s+sử\s+dị\s+ứng|"
    r"tiền\s+sử\s+gia\s+đình|"
    r"tiền\s+sử\s+xã\s+hội|"
    r"tiền\s+sử(?!\s*bệnh\s*hiện\s+tại)|"
    r"bệnh\s+sử(?!\s*hiện\s+tại)|"
    r"tiền\s+căn|"
    r"thuốc\s+trước\s+khi\s+nhập\s+viện|"
    r"thuốc\s+đang\s+dùng|"
    r"đang\s+điều\s+trị\s+tại\s+nhà"
    r")",
    re.IGNORECASE | re.UNICODE,
)

HIEN_TAI_PATTERNS = re.compile(
    r"(?:^|\n)\s*(?:\d+\.\s*)?(?:"
    r"lý\s+do\s+(?:nhập|vào)\s+viện|"
    r"lý\s+do\s+khám|"
    r"tiền\s+sử\s+bệnh\s*hiện\s+tại|"
    r"bệnh\s+sử\s+hiện\s+tại|"
    r"triệu\s+chứng\s+hiện\s+tại|"
    r"triệu\s+chứng\s+cơ\s+năng|"
    r"diễn\s+biến\s+bệnh|"
    r"quá\s+trình\s+bệnh"
    r")",
    re.IGNORECASE | re.UNICODE,
)

DANH_GIA_PATTERNS = re.compile(
    r"(?:^|\n)\s*(?:\d+\.\s*)?(?:"
    r"đánh\s+giá\s+tại\s+bệnh\s+viện|"
    r"kết\s+quả\s+khám|"
    r"kết\s+quả\s+xét\s+nghiệm|"
    r"kết\s+quả\s+chẩn\s+đoán|"
    r"chẩn\s+đoán\s+hình\s+ảnh|"
    r"khám\s+lúc\s+vào\s+viện|"
    r"khám\s+lâm\s+sàng"
    r")",
    re.IGNORECASE | re.UNICODE,
)


def get_section(input_text: str, pos: int) -> str:
    text_before = input_text[:pos]
    
    # Find last match position for each section type
    ts_matches = list(TIEN_SU_PATTERNS.finditer(text_before))
    ht_matches = list(HIEN_TAI_PATTERNS.finditer(text_before))
    dg_matches = list(DANH_GIA_PATTERNS.finditer(text_before))
    
    ts_last = ts_matches[-1].start() if ts_matches else -1
    ht_last = ht_matches[-1].start() if ht_matches else -1
    dg_last = dg_matches[-1].start() if dg_matches else -1
    
    max_pos = max(ts_last, ht_last, dg_last)
    if max_pos == -1:
        return ""
    if max_pos == ts_last:
        return "tien_su"
    if max_pos == ht_last:
        return "hien_tai"
    return "danh_gia"


def audit_historical():
    output_files = sorted(
        [f for f in glob.glob("output/*.json") if Path(f).stem.isdigit()],
        key=lambda x: int(Path(x).stem),
    )
    
    hist_count = 0
    fixed_count = 0
    
    for f in output_files:
        rec_id = int(Path(f).stem)
        inp_path = inp_dir / f"{rec_id}.txt"
        if not inp_path.exists():
            inp_path = inp_dir / f"{rec_id}.json"
        if not inp_path.exists():
            continue
            
        text = inp_path.read_text(encoding="utf-8")
        ents = json.loads(Path(f).read_text(encoding="utf-8"))
        
        for e in ents:
            pos = e.get("position", [0, 0])
            txt = e.get("text", "")
            t = e.get("type", "")
            assertions = e.get("assertions", [])
            
            if not isinstance(pos, list) or len(pos) != 2:
                continue
                
            sec = get_section(text, pos[0])
            has_hist = "isHistorical" in assertions
            
            if sec == "tien_su" and not has_hist and t in ("CHẨN_ĐOÁN", "THUỐC", "TRIỆU_CHỨNG"):
                fixed_count += 1
            elif sec in ("hien_tai", "danh_gia") and has_hist:
                fixed_count += 1
                
            if has_hist:
                hist_count += 1

    print(f"Total isHistorical: {hist_count}")
    print(f"Total assertion fixes with precise section segmentation: {fixed_count}")

if __name__ == "__main__":
    audit_historical()
