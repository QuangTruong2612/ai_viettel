"""Rule-based refinement of NER types and Assertions across all output files."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.postprocess import (
    _DRUG_NAMES_UNIONED,
    _is_procedure,
    _find_current_section,
    _detect_assertions_from_context,
)

# Core dictionary regexes
DRUG_ENDINGS = re.compile(
    r"\d+\s*(?:mg|mcg|g|ml|iu|viên|ống|gói|po|bid|tid|daily|prn|x\s*\d+)$",
    re.IGNORECASE,
)
VITAL_PATTERNS = re.compile(
    r"^(?:\d{2,3}/\d{2,3}\s*(?:mmHg)?|SpO2.*|\d{2,3}\s*%|VS\d+\.\d+.*|\d{2,3}\s*(?:lần/phút|nhịp/phút|\s*°C))$",
    re.IGNORECASE,
)
NORMAL_PATTERNS = re.compile(
    r"^(?:bình\s+thường|không\s+ghi\s+nhận.*|nhịp\s+xoang.*|không\s+có\s+gì\s+đáng\s+chú\s+ý)$",
    re.IGNORECASE,
)

# Known disease prefixes
DISEASE_PREFIXES = (
    "viêm", "tăng huyết áp", "đái tháo đường", "nhồi máu", "suy tim",
    "rung nhĩ", "rối loạn", "thoái hóa", "sỏi", "hội chứng", "xuất huyết",
    "u ", "ung thư", "u bướu", "xơ gan", "trầm cảm", "lo âu", "tai biến",
    "đột quỵ", "liệt", "hoại tử", "nhiễm trùng", "nhiễm khuẩn", "áp xe",
    "phù phổi", "tràn dịch", "tràn khí", "bệnh ", "dị ứng", "hen ",
)

# Known test/procedure keywords
TEST_KEYWORDS = (
    "xét nghiệm", "chụp", "siêu âm", "x-quang", "ct ", "mri", "ecg",
    "điện tâm đồ", "công thức máu", "huyết học", "sinh hóa", "nước tiểu",
    "nội soi", "đo ", "kiểm tra", "kết quả", "bảng điểm", "thang điểm",
)

def refine_record(input_text: str, entities: list[dict]) -> list[dict]:
    refined = []

    for ent in entities:
        if not isinstance(ent, dict):
            continue
        text = str(ent.get("text", "")).strip()
        pos = ent.get("position", [0, 0])
        etype = str(ent.get("type", "")).strip()
        assertions = list(ent.get("assertions", [])) if isinstance(ent.get("assertions"), list) else []

        if not text or not isinstance(pos, list) or len(pos) != 2:
            refined.append(ent)
            continue

        tl = text.lower()
        s, e = int(pos[0]), int(pos[1])

        # 1. TYPE REFINEMENT
        if NORMAL_PATTERNS.match(text) or VITAL_PATTERNS.match(text):
            etype = "KẾT_QUẢ_XÉT_NGHIỆM"
        elif _is_procedure(text) or any(tl.startswith(tk) for tk in TEST_KEYWORDS):
            etype = "TÊN_XÉT_NGHIỆM"
        elif DRUG_ENDINGS.search(text) or any(tl.startswith(dn) for dn in _DRUG_NAMES_UNIONED if len(dn) >= 4):
            etype = "THUỐC"
        elif any(tl.startswith(dp) for dp in DISEASE_PREFIXES):
            etype = "CHẨN_ĐOÁN"

        # 2. ASSERTION REFINEMENT
        # Rule A: KẾT_QUẢ_XÉT_NGHIỆM and TÊN_XÉT_NGHIỆM have NO assertions
        if etype in ("KẾT_QUẢ_XÉT_NGHIỆM", "TÊN_XÉT_NGHIỆM"):
            assertions = []
        else:
            # Rule B: Context-based section & assertion detection
            sec = _find_current_section(input_text, s)
            
            # Historical check
            if sec == "tien_su":
                if "isHistorical" not in assertions:
                    assertions.append("isHistorical")
            elif sec in ("hien_tai", "danh_gia"):
                # Current section -> remove false isHistorical unless explicit past marker in immediate context
                past_context = input_text[max(0, s - 40):s].lower()
                if not re.search(r"\b(?:tiền\s+sử|cách\s+đây|đã\s+từng|trước\s+đây|năm\s+20\d\d)\b", past_context):
                    assertions = [a for a in assertions if a != "isHistorical"]

            # Negation & Family check from context
            rule_ass = _detect_assertions_from_context(text, input_text, etype, s)
            for ra in rule_ass:
                if ra in ("isNegated", "isFamily", "isHistorical") and ra not in assertions:
                    assertions.append(ra)

            # Rule C: Family negative check ("không có tiền sử gia đình bị X") -> isFamily + isNegated
            family_win = input_text[max(0, s - 80):s].lower()
            if re.search(r"gia\s+đình.*khô|khô.*gia\s+đình", family_win):
                if "isFamily" not in assertions:
                    assertions.append("isFamily")
                if "isNegated" not in assertions:
                    assertions.append("isNegated")

            # Clean duplicate & invalid assertions
            seen = set()
            clean_a = []
            for a in assertions:
                if a in ("isNegated", "isHistorical", "isFamily") and a not in seen:
                    seen.add(a)
                    clean_a.append(a)
            assertions = clean_a

        ent["type"] = etype
        ent["assertions"] = assertions
        refined.append(ent)

    return refined


def run_refinement(output_dir: Path, input_dir: Path) -> None:
    output_files = sorted(
        [f for f in output_dir.glob("*.json") if f.stem.isdigit()],
        key=lambda x: int(x.stem),
    )
    print(f"[INFO] Refining NER & Assertions on {len(output_files)} files...")

    type_changes = 0
    assertion_changes = 0

    for fpath in output_files:
        rec_id = int(fpath.stem)
        input_path = input_dir / f"{rec_id}.txt"
        if not input_path.exists():
            input_path = input_dir / f"{rec_id}.json"
        if not input_path.exists():
            continue

        input_text = input_path.read_text(encoding="utf-8")
        entities = json.loads(fpath.read_text(encoding="utf-8"))

        refined = refine_record(input_text, entities)

        for old_e, new_e in zip(entities, refined):
            if old_e.get("type") != new_e.get("type"):
                type_changes += 1
            if old_e.get("assertions") != new_e.get("assertions"):
                assertion_changes += 1

        fpath.write_text(json.dumps(refined, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] Type changes: {type_changes}, Assertion changes: {assertion_changes}")


if __name__ == "__main__":
    inp_dir = Path("data/input") if Path("data/input").exists() else Path("input")
    run_refinement(Path("output"), inp_dir)
