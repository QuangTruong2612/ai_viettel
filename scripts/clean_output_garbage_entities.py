"""Clean garbage entities, bad procedural extractions, combined symptom phrases, and strip candidates from TRIỆU_CHỨNG."""

from __future__ import annotations

import json
import re
from pathlib import Path

# Administrative / non-test phrases wrongly extracted as TÊN_XÉT_NGHIỆM
INVALID_TEST_PATTERNS = re.compile(
    r"""(?x)
    \b(?:gọi\s+xe\s+cứu\s+thương|chuyển\s+viện|nhập\s+viện|ra\s+viện|đưa\s+vào\s+phòng|đi\s+lại|lái\s+xe|nằm\s+tại\s+giường|
    hỏi\s+bệnh|thay\s+đổi\s+tư\s+thế|cử\s+động|tái\s+khám|xuất\s+viện)\b
    """,
    re.IGNORECASE | re.UNICODE,
)

# Long garbled non-symptom sentence fragments wrongly extracted as TRIỆU_CHỨNG
INVALID_SYMPTOM_PATTERNS = re.compile(
    r"""(?x)
    \b(?:lái\s+xe\s+sau\s+ngã|thể\s+chịu\s+trọng\s+lượng|nằm\s+tại\s+giường|không\s+thể\s+tự\s+di\s+chuyển|
    tự\s+đo\s+từ|sau\s+ngã|trong\s+vòng\s+\d+|kể\s+từ\s+ngày|cho\s+đến\s+khi)\b
    """,
    re.IGNORECASE | re.UNICODE,
)


def clean_record_entities(input_text: str, entities: list[dict]) -> tuple[list[dict], int]:
    cleaned = []
    removed_count = 0

    for ent in entities:
        if not isinstance(ent, dict):
            continue

        text = str(ent.get("text", "")).strip()
        etype = str(ent.get("type", "")).strip()
        pos = ent.get("position", [0, 0])
        assertions = ent.get("assertions", [])
        cands = ent.get("candidates", [])

        # Rule 1: Strip candidates from TRIỆU_CHỨNG (symptoms must have candidates=[])
        if etype == "TRIỆU_CHỨNG":
            if cands:
                ent["candidates"] = []
                cands = []

        # Rule 2: Drop non-medical administrative actions from TÊN_XÉT_NGHIỆM
        if etype == "TÊN_XÉT_NGHIỆM" and INVALID_TEST_PATTERNS.search(text):
            removed_count += 1
            continue

        # Rule 3: Drop non-symptom sentence fragments from TRIỆU_CHỨNG
        if etype == "TRIỆU_CHỨNG" and INVALID_SYMPTOM_PATTERNS.search(text):
            removed_count += 1
            continue

        # Rule 4: Handle compound symptom 'X hoặc Y' -> Split into X and Y
        if etype == "TRIỆU_CHỨNG" and " hoặc " in text.lower():
            parts = re.split(r"\s+hoặc\s+", text, flags=re.IGNORECASE)
            if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                for p in parts:
                    p_txt = p.strip()
                    p_start = input_text.find(p_txt, max(0, pos[0] - 5)) if isinstance(pos, list) else -1
                    if p_start >= 0:
                        p_pos = [p_start, p_start + len(p_txt)]
                    else:
                        p_pos = pos
                    cleaned.append({
                        "text": p_txt,
                        "type": "TRIỆU_CHỨNG",
                        "position": p_pos,
                        "assertions": assertions,
                        "candidates": [],
                    })
                removed_count += 1
                continue

        # Rule 5: Fix double words in text like 'Tăng tăng cân' -> 'tăng cân'
        if text.lower().startswith("tăng tăng "):
            fixed_txt = text[5:]  # strip first 'Tăng '
            if isinstance(pos, list) and len(pos) == 2:
                new_start = input_text.find(fixed_txt, pos[0])
                if new_start >= 0:
                    ent["position"] = [new_start, new_start + len(fixed_txt)]
            ent["text"] = fixed_txt

        cleaned.append(ent)

    return cleaned, removed_count


def run_garbage_cleanup(output_dir: Path, input_dir: Path) -> None:
    output_files = sorted(
        [f for f in output_dir.glob("*.json") if f.stem.isdigit()],
        key=lambda x: int(x.stem),
    )
    print(f"[INFO] Cleaning garbage extractions across {len(output_files)} files...")

    total_removed = 0

    for fpath in output_files:
        rec_id = int(fpath.stem)
        inp_path = input_dir / f"{rec_id}.txt"
        if not inp_path.exists():
            inp_path = input_dir / f"{rec_id}.json"
        if not inp_path.exists():
            continue

        input_text = inp_path.read_text(encoding="utf-8")
        entities = json.loads(fpath.read_text(encoding="utf-8"))

        cleaned, removed = clean_record_entities(input_text, entities)
        total_removed += removed

        if removed > 0 or True:
            fpath.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] Filtered {total_removed} garbage entities across records!")


if __name__ == "__main__":
    inp_dir = Path("data/input") if Path("data/input").exists() else Path("input")
    run_garbage_cleanup(Path("output"), inp_dir)
