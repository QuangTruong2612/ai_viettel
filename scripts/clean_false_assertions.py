"""Clean up false positive assertions and add missing isHistorical assertions based on precise line-boundary section segmentation."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.postprocess import _find_current_section

FAMILY_PATTERNS = [
    r"\b(?:bố|cha|mẹ|anh|chị|em|con|ông|bà|cô|dì|chú(?!\s+ý)|bác(?!\s+sĩ))(?:\s+(?:trai|gái|nội|ngoại|ruột|chồng|vợ))?\s+(?:bị|mắc|có|từng|tiền\s+sử|mất\s+vì|chết\s+vì|đã\s+từng|được\s+chẩn\s+đoán)\b",
    r"gia\s+[đd][ìi]nh\s+kh[ôo]ng\s+(?:ai|b[ệe]n\s+n[âa]o)",
    r"\b(?:bố|cha|mẹ|anh|chị|em|con|ông|bà|cô|dì|chú(?!\s+ý)|bác(?!\s+sĩ))(?:\s+(?:trai|gái|nội|ngoại|ruột|chồng|vợ))?\s+b[ệe]nh\s+nh[âa]n",
    r"gia\s+đ[ìi]nh\s+(?:có|bị|từng|tiền\s+sử|ghi\s+nhận|ai|mắc)",
    r"ti[eề]n\s+s[ử]\s*gia\s+[đd][ìi]nh",
    r"\bhọ\s+hàng\b",
    r"\bngười\s+thân\b",
    r"\bdi\s+truyền\b",
]

PAST_INDICATORS = re.compile(
    r"\b(?:tiền\s+sử|cách\s+đây|đã\s+từng|trước\s+đây|năm\s+20\d\d|mạn\s+tính|mãn\s+tính|cũ|dùng\s+từ\s+trước)\b",
    re.IGNORECASE | re.UNICODE,
)

NEGATION_PATTERNS = re.compile(
    r"\b(?:không|chưa|âm\s+tính|không\s+thấy|chưa\s+thấy|loại\s+trừ|không\s+có|chưa\s+có|không\s+ghi\s+nhận|chưa\s+ghi\s+nhận)\b",
    re.IGNORECASE | re.UNICODE,
)

NON_NEGATION_CONTEXTS = re.compile(
    r"không\s+tuân\s+thủ|không\s+thể|không\s+có\s+khả\s+năng|chưa\s+rõ|không\s+được\s+(?:thực\s+hiện|làm|chụp|tiến\s+hành)",
    re.IGNORECASE | re.UNICODE,
)


def clean_assertions_for_record(input_text: str, entities: list[dict]) -> tuple[list[dict], int]:
    cleaned = []
    changes_count = 0

    for ent in entities:
        if not isinstance(ent, dict):
            continue
        text = str(ent.get("text", "")).strip()
        pos = ent.get("position", [0, 0])
        etype = str(ent.get("type", "")).strip()
        assertions = list(ent.get("assertions", [])) if isinstance(ent.get("assertions"), list) else []

        # Rule 1: TÊN_XÉT_NGHIỆM and KẾT_QUẢ_XÉT_NGHIỆM never have assertions
        if etype in ("TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"):
            if assertions:
                changes_count += len(assertions)
                assertions = []
                ent["assertions"] = []
            cleaned.append(ent)
            continue

        if not isinstance(pos, list) or len(pos) != 2:
            cleaned.append(ent)
            continue

        s, e = int(pos[0]), int(pos[1])
        valid_a = []

        # Rule 2: Section-based isHistorical auto-tagging / cleaning
        sec = _find_current_section(input_text, s)
        pre_span = input_text[max(0, s - 45):s]
        pre_clause = re.split(r"[\n.]", pre_span)[-1]

        should_be_historical = False
        if sec == "tien_su":
            should_be_historical = True
        elif PAST_INDICATORS.search(pre_clause) or PAST_INDICATORS.search(text):
            should_be_historical = True

        for a in assertions:
            if a == "isNegated":
                if NEGATION_PATTERNS.search(pre_clause) and not NON_NEGATION_CONTEXTS.search(pre_clause):
                    valid_a.append(a)
                else:
                    changes_count += 1

            elif a == "isFamily":
                win_span = input_text[max(0, s - 100):min(len(input_text), e + 50)]
                clause = re.split(r"[\n.]", win_span)
                rel_clause = [c for c in clause if text.lower() in c.lower()]
                clause_text = rel_clause[0] if rel_clause else win_span
                
                is_real_fam = False
                for fp in FAMILY_PATTERNS:
                    if re.search(fp, clause_text, re.IGNORECASE | re.UNICODE):
                        is_real_fam = True
                        break
                if is_real_fam:
                    valid_a.append(a)
                else:
                    changes_count += 1

            elif a == "isHistorical":
                if should_be_historical:
                    valid_a.append(a)
                else:
                    changes_count += 1

        if should_be_historical and "isHistorical" not in valid_a:
            valid_a.append("isHistorical")
            changes_count += 1

        ent["assertions"] = valid_a
        cleaned.append(ent)

    return cleaned, changes_count


def run_assertion_cleanup(output_dir: Path, input_dir: Path) -> None:
    output_files = sorted(
        [f for f in output_dir.glob("*.json") if f.stem.isdigit()],
        key=lambda x: int(x.stem),
    )
    print(f"[INFO] Synchronizing assertions using line-boundary section segmentation on {len(output_files)} output files...")

    total_changes = 0

    for fpath in output_files:
        rec_id = int(fpath.stem)
        input_path = input_dir / f"{rec_id}.txt"
        if not input_path.exists():
            input_path = input_dir / f"{rec_id}.json"
        if not input_path.exists():
            continue

        input_text = input_path.read_text(encoding="utf-8")
        entities = json.loads(fpath.read_text(encoding="utf-8"))

        cleaned, changes = clean_assertions_for_record(input_text, entities)
        total_changes += changes

        if changes > 0:
            fpath.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] Synchronized {total_changes} assertions across records!")


if __name__ == "__main__":
    inp_dir = Path("data/input") if Path("data/input").exists() else Path("input")
    run_assertion_cleanup(Path("output"), inp_dir)
