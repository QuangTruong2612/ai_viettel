"""Post-process: validate, sửa lỗi, deduplicate; gắn candidates từ RxNorm RAG.

Hàm chính:
- validate_positions(input_text, entities): sửa position sai bằng cách tìm lại.
- dedupe_entities(entities): bỏ trùng (cùng text + position).
- assemble_record(input_text, raw_entities, retriever): build list final có candidates.
- validate_output(record): kiểm tra cuối cùng.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Iterable

from .icd_rag import ICDRetriever
from .rxnorm_rag import RxNormRetriever

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------- #
# Position fixing
# ---------------------------------------------------------------------- #


def _find_span(text: str, snippet: str) -> tuple[int, int] | None:
    """Tìm vị trí đầu tiên của snippet trong text; trả [start, end) hoặc None."""
    if not snippet:
        return None
    idx = text.find(snippet)
    if idx >= 0:
        return idx, idx + len(snippet)
    # Fallback: lowercase
    idx = text.lower().find(snippet.lower())
    if idx >= 0:
        return idx, idx + len(snippet)
    # Fallback: bỏ khoảng trắng thừa ở hai đầu
    stripped = snippet.strip()
    if stripped != snippet:
        idx = text.find(stripped)
        if idx >= 0:
            return idx, idx + len(stripped)
    return None


def validate_positions(
    input_text: str,
    entities: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Sửa position cho từng entity.

    LLM có thể đoán sai index (off-by-one, hoặc skip token). Nếu text
    không khớp input[start:end] thì ta cố gắng tìm lại.
    """
    out: list[dict[str, Any]] = []
    for ent in entities:
        text = str(ent.get("text", "")).strip()
        pos = ent.get("position", [])
        if not text or not isinstance(pos, list) or len(pos) != 2:
            continue

        start, end = int(pos[0]), int(pos[1])
        # Sanity bounds
        if start < 0:
            start = 0
        if end > len(input_text):
            end = len(input_text)

        # Nếu substring không khớp → tìm lại
        if input_text[start:end] != text:
            found = _find_span(input_text, text)
            if found is None:
                logger.debug("Bỏ entity không tìm được: %r", text)
                continue
            start, end = found

        out.append({**ent, "text": text, "position": [start, end]})
    return out


# ---------------------------------------------------------------------- #
# Dedupe
# ---------------------------------------------------------------------- #


def dedupe_entities(entities: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Bỏ trùng (cùng text + start) — kết quả sort theo start."""
    seen: set[tuple[str, int]] = set()
    out: list[dict[str, Any]] = []
    for ent in entities:
        key = (str(ent.get("text", "")), int(ent.get("position", [0, 0])[0]))
        if key in seen:
            continue
        seen.add(key)
        out.append(ent)
    out.sort(key=lambda e: e["position"][0])
    return out


# ---------------------------------------------------------------------- #
# Drop drug names that look wrong (heuristic)
# ---------------------------------------------------------------------- #


_DRUG_NAME_BAD_PATTERNS = re.compile(
    r"^(thuốc|drug|medication|thuoc)\s*$", re.IGNORECASE
)


def sanitize_drug_text(text: str) -> str:
    """Bỏ một số chuỗi giả thuốc rõ ràng (placeholder)."""
    if _DRUG_NAME_BAD_PATTERNS.match(text.strip()):
        return ""
    return text


# VN/Vietnamese-English connectors between drug name and disease.
_DRUG_FOR_DISEASE_RE = re.compile(
    r"^(?P<drug>[A-Za-zÀ-ỹ][\w\s\-\.]*?)"
    r"\s+(?:cho|đ[eế]?u\s*tr[ịi]|treats?|for|to)\s+"
    r"(?P<disease>[A-Za-zÀ-ỹ][\w\s\-\.àáạảãăắằẳẵặâấầẩẫậèéẹẻẽêếềểễệìíịỉĩòóọỏõôốồổỗộơớờởỡợùúụủũưứừửữựỳýỷỹỵđ]+)$",
    re.IGNORECASE | re.UNICODE,
)


def _split_drug_cho_pattern(text: str) -> tuple[str, str | None]:
    """Tách cụm "drug A cho/treats disease B" thành 2 phần.

    Ví dụ:
        "doxycycline cho viêm tuyến mồ hôi"
            → ("doxycycline", "viêm tuyến mồ hôi")
        "methotrexate cho viêm khớp dạng thấp"
            → ("methotrexate", "viêm khớp dạng thấp")
        "aspirin 81 mg po daily"
            → ("aspirin 81 mg po daily", None)  # không khớp pattern

    Trả (text_gốc, None) nếu không match — caller xử lý bình thường.
    """
    s = text.strip()
    m = _DRUG_FOR_DISEASE_RE.match(s)
    if not m:
        return (s, None)
    drug = m.group("drug").strip()
    disease = m.group("disease").strip()
    if not drug or not disease or len(drug) < 3 or len(disease) < 3:
        return (s, None)
    return (drug, disease)


# ---------------------------------------------------------------------- #
# Main assembly
# ---------------------------------------------------------------------- #


def assemble_record(
    input_text: str,
    raw_entities: Iterable[dict[str, Any]],
    retriever: RxNormRetriever,
    icd_retriever: Optional[ICDRetriever] = None,
) -> list[dict[str, Any]]:
    """Build list thực thể cuối cùng cho một record.

    - Validate position.
    - Dedupe.
    - Gán candidates:
        + THUỐC → RxNorm (qua retriever)
        + CHẨN_ĐOÁN → ICD-10 (qua icd_retriever; cần VN→EN translation)
        + TRIỆU_CHỨNG → không gán candidates.
    - Chuẩn hoá assertions (unique, sorted).
    - Sắp xếp theo vị trí.
    """
    validated = validate_positions(input_text, raw_entities)
    validated = dedupe_entities(validated)

    final: list[dict[str, Any]] = []
    # Track text+type đã emit để dedupe (vd "doxycycline" trùng khi LLM trả 2 lần)
    seen_signatures: set[tuple[str, str]] = set()
    for ent in validated:
        etype = ent.get("type", "")
        text = str(ent.get("text", "")).strip()
        if not text:
            continue
        if etype not in (
            "THUỐC",
            "TRIỆU_CHỨNG",
            "TÊN_XÉT_NGHIỆM",
            "KẾT_QUẢ_XÉT_NGHIỆM",
            "CHẨN_ĐOÁN",
        ):
            continue

        # Skip trùng với entity đã emit (vd LLM trả "doxycycline" + "doxycycline cho X")
        sig = (text, etype)
        if sig in seen_signatures:
            continue
        seen_signatures.add(sig)

        # Assertions cleanup — chỉ giữ 3 giá trị hợp lệ
        assertions = ent.get("assertions", []) or []
        if not isinstance(assertions, list):
            assertions = []
        assertions = sorted(
            {a for a in assertions if a in {"isNegated", "isFamily", "isHistorical"}}
        )

        # Bug history: LLM trả ra "A cho B" (drug cho disease) gán nhầm type THUỐC
        # cho cả cụm. Fix: tách thành 2 entities nếu match. CHỈ emit mỗi phần nếu
        # chưa có trong seen_signatures (tránh duplicate khi LLM trả 2 lần drug).
        drug_part, diag_part = _split_drug_cho_pattern(text)
        if diag_part is not None and drug_part != text:
            # Find positions
            drug_pos_start = input_text.find(drug_part, int(ent["position"][0]))
            if drug_pos_start < 0:
                drug_pos_start = int(ent["position"][0])
            drug_pos_end = drug_pos_start + len(drug_part)
            diag_pos_start = input_text.find(diag_part, drug_pos_end)
            if diag_pos_start < 0:
                diag_pos_start = drug_pos_end + len(" cho ")
            diag_pos_end = diag_pos_start + len(diag_part)

            emitted_any = False
            # Emit drug part nếu chưa thấy
            if (drug_part, "THUỐC") not in seen_signatures:
                item: dict[str, Any] = {
                    "text": drug_part,
                    "type": "THUỐC",
                    "position": [drug_pos_start, drug_pos_end],
                    "assertions": list(assertions),
                }
                cleaned = sanitize_drug_text(drug_part)
                if cleaned:
                    cand = retriever.lookup(cleaned)
                    if cand:
                        item["candidates"] = cand
                final.append(item)
                seen_signatures.add((drug_part, "THUỐC"))
                emitted_any = True
            # Emit diagnosis part nếu chưa thấy
            if (diag_part, "CHẨN_ĐOÁN") not in seen_signatures:
                item2: dict[str, Any] = {
                    "text": diag_part,
                    "type": "CHẨN_ĐOÁN",
                    "position": [diag_pos_start, diag_pos_end],
                    "assertions": list(assertions),
                }
                if icd_retriever is not None:
                    cand2 = icd_retriever.lookup(diag_part)
                    if cand2:
                        item2["candidates"] = cand2
                final.append(item2)
                seen_signatures.add((diag_part, "CHẨN_ĐOÁN"))
                emitted_any = True
            # Nếu không emit gì thì skip (đã có từ entity trước)
            if not emitted_any:
                continue
            continue

        item: dict[str, Any] = {
            "text": text,
            "type": etype,
            "position": [int(ent["position"][0]), int(ent["position"][1])],
            "assertions": assertions,
        }
        # Candidates CHỈ cho THUỐC và CHẨN_ĐOÁN (per spec)
        if etype == "THUỐC":
            cleaned = sanitize_drug_text(text)
            if cleaned:
                cand = retriever.lookup(cleaned)
                if cand:
                    item["candidates"] = cand
        elif etype == "CHẨN_ĐOÁN" and icd_retriever is not None:
            cand = icd_retriever.lookup(text)
            if cand:
                item["candidates"] = cand
        final.append(item)
    return final


# ---------------------------------------------------------------------- #
# Output validation
# ---------------------------------------------------------------------- #


def validate_output(payload: list[dict[str, Any]]) -> bool:
    """Schema check cuối cùng."""
    try:
        from jsonschema import validate  # type: ignore

        from .prompts import OUTPUT_SCHEMA

        validate(instance=payload, schema=OUTPUT_SCHEMA)
        return True
    except Exception as exc:
        logger.warning("Validation lỗi: %s", exc)
        return False


def write_output(path: Path, payload: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------- #
# Self-test
# ---------------------------------------------------------------------- #

if __name__ == "__main__":  # pragma: no cover
    import sys

    logging.basicConfig(level=logging.DEBUG)
    sample_text = "Bệnh nhân dùng aspirin 81 mg po daily trước nhập viện điều trị nhức đầu."
    sample_ents = [
        {
            "text": "aspirin 81 mg po daily",
            "type": "THUỐC",
            "position": [13, 35],
            "assertions": ["isHistorical"],
        },
        {
            "text": "nhức đầu",
            "type": "TRIỆU_CHỨNG",
            "position": [56, 64],
            "assertions": [],
        },
    ]
    retriever = RxNormRetriever()
    out = assemble_record(sample_text, sample_ents, retriever)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print("Valid:", validate_output(out))
