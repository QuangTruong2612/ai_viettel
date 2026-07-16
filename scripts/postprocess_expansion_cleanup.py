"""Post-process expansion cleanup: thay thế text expanded về dạng viết tắt.

Trước R37, LLM hay extract text="aspartate aminotransferase" thay vì "AST"
(vì nó "hiểu" viết tắt). Scoring bị WER cao do text mismatch với gold.

Script này:
1. Load output JSON + input text tương ứng
2. Với mỗi entity, check text có phải dạng "expanded" của 1 viết tắt phổ biến
3. Nếu có → thay bằng viết tắt + tìm lại position trong input_text
4. Nếu không tìm thấy viết tắt trong input → giữ nguyên

Usage:
    python scripts/postprocess_expansion_cleanup.py --input input --output output
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


# R37 (2026-07-15): Mapping common abbreviation expansions → abbreviation form.
# Đây là những viết tắt lab/clinical phổ biến hay bị LLM "mở rộng" ra tên đầy đủ.
_ABBREV_TO_FULL: dict[str, list[str]] = {
    # Lab tests - biochemistry
    "AST": ["aspartate aminotransferase", "aspartate transaminase", "sgot"],
    "ALT": ["alanine aminotransferase", "alanine transaminase", "sgpt"],
    "GGT": ["gamma-glutamyl transferase", "gamma glutamyl transferase", "gammaglutamyl transferase", "gamma-glutamyl transpeptidase"],
    "LDH": ["lactate dehydrogenase", "lactic dehydrogenase"],
    "ALP": ["alkaline phosphatase"],
    # CBC
    "WBC": ["white blood cell", "white blood cell count", "white cell", "leukocyte"],
    "RBC": ["red blood cell", "red blood cell count", "erythrocyte"],
    "Hgb": ["hemoglobin", "haemoglobin"],
    "HbA1c": ["hemoglobin a1c", "glycated hemoglobin", "glycosylated hemoglobin"],
    # Cardiac / coagulation
    "PT": ["prothrombin time"],
    "PTT": ["partial thromboplastin time"],
    "aPTT": ["activated partial thromboplastin time"],
    "INR": ["international normalized ratio"],
    "BNP": ["brain natriuretic peptide", "b-type natriuretic peptide"],
    "CRP": ["c-reactive protein", "c reactive protein"],
    "ESR": ["erythrocyte sedimentation rate"],
    # Diseases (clinical abbreviations)
    "THA": ["tăng huyết áp", "tang huyet ap"],
    "ĐTĐ": ["đái tháo đường", "dai thao duong", "đái đường", "đường", "tiểu đường"],
    "ĐTĐ type 2": ["đái tháo đường type 2", "đái tháo đường tuýp 2"],
    "NMCT": ["nhồi máu cơ tim", "nhoi mau co tim"],
    "RLLL": ["rối loạn lipid máu", "roi loan lipid mau"],
    "COPD": ["bệnh phổi tắc nghẽn mạn tính", "copd"],
    "CKD": ["bệnh thận mạn", "benh than man"],
    "TBMMN": ["tai biến mạch máu não"],
    "BTMV": ["bệnh tim mạch vành", "benh tim mach vanh"],
    "HPH": ["hạ kali máu", "hạ natri máu", "hạ canxi máu"],
}


def _build_full_to_abbrev() -> dict[str, str]:
    """Inverted: full name (lowercase) → abbreviation."""
    inv: dict[str, str] = {}
    for ab, fulls in _ABBREV_TO_FULL.items():
        for full in fulls:
            inv[full.strip().lower()] = ab
    return inv


_FULL_TO_ABBREV = _build_full_to_abbrev()


def _find_span(text: str, snippet: str, start: int = 0) -> tuple[int, int] | None:
    """Find snippet trong text với word-boundary check."""
    if not snippet:
        return None
    n_text = len(text)
    n_snip = len(snippet)
    tl = text.lower()
    sl = snippet.lower()
    idx = start - 1
    while True:
        idx = tl.find(sl, idx + 1)
        if idx < 0:
            break
        end_idx = idx + n_snip
        # Word-boundary
        prev_ok = idx == 0 or not text[idx - 1].isalnum()
        next_ok = end_idx >= n_text or not text[end_idx].isalnum()
        if prev_ok and next_ok:
            return idx, end_idx
    return None


def _clean_text(text: str) -> str:
    """Strip diacritics + lowercase for matching."""
    import unicodedata
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text.lower().strip()


def _try_collapse_abbrev(text: str, input_text: str | None = None) -> tuple[str, tuple[int, int] | None] | None:
    """Nếu text có dạng expanded form của 1 viết tắt → return (abbrev, position).

    Returns None nếu không có match.
    """
    if not text:
        return None
    t_clean = _clean_text(text)

    # Direct lookup
    for full, ab in _FULL_TO_ABBREV.items():
        if t_clean == _clean_text(full):
            return (ab, None)  # Will re-find below if input_text given

    # Prefix match (vd "aspartate aminotransferase 45 U/L" → strip "45 U/L")
    for full, ab in _FULL_TO_ABBREV.items():
        full_clean = _clean_text(full)
        if t_clean.startswith(full_clean):
            # Nếu phần còn lại là value (digit + unit), split
            rest = text[len(full):].strip()
            if re.match(r"^[\d.,\s]*[a-z%/]*\s*$", rest, re.IGNORECASE):
                # Edge case — return both? For now return abbrev only.
                return (ab, None)
    return None


def _process_record(record: list[dict], input_text: str | None, rec_id: int) -> tuple[list[dict], int]:
    """Trả về (new_record, num_changed).
    R37: Thêm dedupe pass để xóa duplicate tạo ra bởi các lần chạy trước.
    Quy tắc dedupe:
      - Cùng (position, text.lower()) → giữ entity đầu tiên, bỏ các entity sau
      - Cùng position nhưng text khác (vd "ast" vs "AST") → giữ TÊN_XN / TRIỆU_CHỨNG / CHẨN_ĐOÁN ưu tiên hơn THUỐC (drug brand name thường có type sai)
    """

    # Dedupe pass: cùng position → giữ thực thể đầu tiên (Pydantic-style preserve order)
    seen_positions: set[tuple[int, int]] = set()
    deduped: list[dict] = []
    dropped_dup = 0
    # Priority order: nếu 2 ent trùng position nhưng khác text, giữ cái có type "quan trọng hơn"
    TYPE_PRIORITY = {"CHẨN_ĐOÁN": 4, "TÊN_XÉT_NGHIỆM": 3, "KẾT_QUẢ_XÉT_NGHIỆM": 2, "TRIỆU_CHỨNG": 1, "THUỐC": 0}
    pos_to_best: dict[tuple[int, int], dict] = {}

    for ent in record:
        pos = ent.get("position", [0, 0])
        if not (isinstance(pos, list) and len(pos) == 2):
            deduped.append(ent)
            continue
        pos_tuple = tuple(pos)
        text = ent.get("text", "").strip()

        # Cùng (position, text) → dedupe
        if pos_tuple in seen_positions and pos_to_best.get(pos_tuple, {}).get("text", "").strip().lower() == text.lower():
            dropped_dup += 1
            continue

        # Cùng position khác text → giữ best priority (nếu đã có 1 ent ở đây)
        if pos_tuple in pos_to_best:
            existing = pos_to_best[pos_tuple]
            ex_type = existing.get("type", "")
            new_type = ent.get("type", "")
            if TYPE_PRIORITY.get(new_type, 0) > TYPE_PRIORITY.get(ex_type, 0):
                # Replace existing with new
                pos_to_best[pos_tuple] = ent
                # Remove existing from deduped, add new
                deduped = [d for d in deduped if d is not existing]
                deduped.append(ent)
            dropped_dup += 1
            continue

        pos_to_best[pos_tuple] = ent
        seen_positions.add(pos_tuple)
        deduped.append(ent)

    # Pass 2: collapse expansion
    changed = 0
    new_ents = []
    existing_keys: set[tuple[tuple[int, int], str]] = set()
    for ent in deduped:
        text = ent.get("text", "")
        pos = ent.get("position", [0, 0])
        if isinstance(pos, list) and len(pos) == 2 and text:
            existing_keys.add((tuple(pos), text.lower()))

    for ent in deduped:
        text = ent.get("text", "")
        if not text:
            new_ents.append(ent)
            continue
        collapse = _try_collapse_abbrev(text, input_text)
        if collapse is None or not input_text:
            new_ents.append(ent)
            continue
        new_text, _ = collapse
        found = _find_span(input_text, new_text)
        if not found:
            new_ents.append(ent)
            continue
        # Check duplicate (position, text)
        if (found, new_text.lower()) in existing_keys:
            changed += 1
            continue  # REMOVE this entity instead
        # Check same position (any text)
        if tuple(found) in {p for p, _ in existing_keys}:
            changed += 1
            continue
        ent["text"] = new_text
        ent["position"] = list(found)
        existing_keys.add((tuple(found), new_text.lower()))
        changed += 1
        new_ents.append(ent)
    return new_ents, changed


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("input"))
    p.add_argument("--output", type=Path, default=Path("output"))
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    files = sorted(
        [f for f in args.output.glob("*.json") if f.stem.isdigit()],
        key=lambda x: int(x.stem),
    )
    if args.limit:
        files = files[: args.limit]

    total_changed = 0
    for fp in files:
        rec_id = int(fp.stem)
        # Load input text
        input_text = None
        input_path = args.input / f"{rec_id}.txt"
        if input_path.exists():
            input_text = input_path.read_text(encoding="utf-8").strip()
        # Process output
        data = json.loads(fp.read_text(encoding="utf-8"))
        new_data, changed = _process_record(data, input_text, rec_id)
        total_changed += changed
        fp.write_text(json.dumps(new_data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Processed {len(files)} files, {total_changed} entities collapsed")
    return 0


if __name__ == "__main__":
    sys.exit(main())