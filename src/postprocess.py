"""Post-process: validate, sửa lỗi, deduplicate; gắn candidates từ RxNorm RAG.

Hàm chính:
- validate_positions(input_text, entities): sửa position sai bằng cách tìm lại.
- dedupe_entities(entities): bỏ trùng (cùng text + position).
- assemble_record(input_text, raw_entities, retriever): build list final có candidates.
- validate_output(record): kiểm tra cuối cùng.

Cách chạy:
    # Khuyến nghị (từ project root):
    python -m src.postprocess

    # Hoặc trực tiếp (script tự thêm project root vào sys.path):
    python src/postprocess.py
"""

from __future__ import annotations

import json
import logging
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Optional

from src.icd_rag import ICDRetriever, _is_generic_drug_class, _strip_drug_class_prefix
from src.rxnorm_rag import RxNormRetriever, _DRUG_INN_WHITELIST as _RXNORM_INN_CACHE

# Đảm bảo có thể chạy trực tiếp `python src/postprocess.py`
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════════
# R28 (2026-07-13): Unified drug name whitelist — hardcoded legacy 13 entries
# UNION với auto-mined RxNorm INN (~63k entries từ rxnorm.jsonl).
# Set lookup O(1); cập nhật whitelist bằng cách rerun scripts/build_mining_index.py.
# ════════════════════════════════════════════════════════════════════════════════

_BASE_DRUG_NAMES: set[str] = {
    "aspirin", "atenolol", "metoprolol", "amlodipine", "bisoprolol",
    "furosemide", "paracetamol", "doxycycline", "clopidogrel",
    "atorvastatin", "losartan", "enoxaparin", "nitroglycerin",
}
_DRUG_NAMES_UNIONED: frozenset[str] = frozenset(_BASE_DRUG_NAMES | _RXNORM_INN_CACHE)
logger.info(
    "[R28] drug_names whitelist: %d entries (base 13 + INN %d)",
    len(_DRUG_NAMES_UNIONED), len(_RXNORM_INN_CACHE),
)


# ════════════════════════════════════════════════════════════════════════════════
# R37 (2026-07-15): Drug BRAND names set — union keys của drug_aliases.json +
# drug_brand_seed.json (sau khi loại bỏ các token không phải thuốc).
# Dùng trong _retype_entity: nếu text trùng brand → force THUỐC.
# ════════════════════════════════════════════════════════════════════════════════

_BRAND_NAMES: set[str] = set()
try:
    _alias_path = _PROJECT_ROOT / "data" / "drug_aliases.json"
    if _alias_path.exists():
        _alias_obj = json.loads(_alias_path.read_text(encoding="utf-8"))
        for k in _alias_obj.keys():
            kl = str(k).lower().strip()
            if kl and "?" not in kl:
                _BRAND_NAMES.add(kl)
except Exception as exc:
    logger.warning("[R37] Failed to load drug_aliases.json for brands: %s", exc)
try:
    _brand_path = _PROJECT_ROOT / "data" / "drug_brand_seed.json"
    if _brand_path.exists():
        _brand_obj = json.loads(_brand_path.read_text(encoding="utf-8"))
        for k in _brand_obj.keys():
            kl = str(k).lower().strip()
            # Loại bỏ những token rõ ràng không phải thuốc (BiPAP, BIPAP, kháng, ...)
            if not kl or "?" in kl:
                continue
            if kl in {"bipap", "cpap", "kháng", "thuốc"}:
                continue
            _BRAND_NAMES.add(kl)
except Exception as exc:
    logger.warning("[R37] Failed to load drug_brand_seed.json: %s", exc)

_DRUG_BRANDS: frozenset[str] = frozenset(_BRAND_NAMES)
logger.info("[R37] drug_brands whitelist: %d entries", len(_DRUG_BRANDS))


# ════════════════════════════════════════════════════════════════════════════════
# R37 (2026-07-15): Test name ABBREVIATIONS (xét nghiệm viết tắt).
# Khi LLM extract ast/alt/wbc/... thường gán KQ_XN (vì đi kèm số đo) → sai, phải là TÊN_XN.
# Dùng trong _retype_entity: nếu text trùng abbreviation (case-insensitive,
# standalone, không kèm số) → force TÊN_XÉT_NGHIỆM.
# ════════════════════════════════════════════════════════════════════════════════

_TEST_ABBREVIATIONS: frozenset[str] = frozenset({
    # Liver panel
    "ast", "alt", "ggt", "ldh", "alp", "bilirubin",
    # CBC
    "wbc", "rbc", "hgb", "hct", "plt", "mcv", "mch", "mchc", "rdw", "mpv",
    # Chemistry
    "na", "k", "cl", "mg", "ca", "phos", "glucose", "bun", "creatinine",
    "uric acid", "uric",
    # Lipid
    "cholesterol", "triglyceride", "hdl", "ldl",
    # Endocrine
    "tsh", "t3", "t4", "ft4", "ft3", "hba1c",
    # Inflammation markers
    "crp", "esr", "procalcitonin",
    # Cardiac
    "psa", "troponin", "bnp", "ck", "ck-mb", "ckmb",
    # Renal
    "egfr", "protein", "albumin",
    # Coag
    "pt", "ptt", "inr", "aptt", "fibrinogen", "ddimer", "d-dimer",
    # Misc
    "lactate", "ammonia", "iron", "ferritin", "vitamin d", "b12",
    "magnesium", "phosphate",
})
logger.info("[R37] test_abbreviations whitelist: %d entries", len(_TEST_ABBREVIATIONS))


# R37 (2026-07-15): Standalone dose fragment regex — pure "<number><unit>" với
# không có tên thuốc đi kèm. Match: "30 mg", "60 mg", "500mcg", "5 ml", ...
# Dùng để DROP entity trong _clean_entity_text.
_DOSE_FRAGMENT_RE = re.compile(
    r"^\s*\d+(?:[.,]\d+)?\s*(mg|ml|g|mcg|iu|meq|µg|ug|ng|kg|mmol|mm|l)\s*$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------- #
# Input preprocessing — clean/truncate input trước khi gửi LLM
# ---------------------------------------------------------------------- #

# Ngưỡng tối đa cho input (chars). Vượt ngưỡng sẽ truncate để tránh overflow.
# SYSTEM_PROMPT (~4059 tokens) + 4000 chars input (~1000 tokens) + max_tokens output (4096)
# = ~9155 tokens. Vừa với Ollama num_ctx=8192 nếu giảm input xuống <3000 chars.
_INPUT_MAX_CHARS = 12000

# Header đánh dấu đã truncate (LLM biết phần nào bị cắt)
_TRUNCATION_MARKER = "\n\n[... Đã rút gọn phần giữa để vừa context window ...]\n\n"


def _normalize_assertions_list(raw: Any) -> list[str]:
    """Chuẩn hóa assertions từ LLM (xử lý an toàn khi LLM trả về dict, string hay non-list)."""
    if not raw:
        return []
    if isinstance(raw, str):
        return [raw] if raw in {"isNegated", "isFamily", "isHistorical"} else []
    if isinstance(raw, dict):
        raw = list(raw.values()) + list(raw.keys())
    if not isinstance(raw, (list, tuple, set)):
        return []
    out = set()
    for item in raw:
        if isinstance(item, str):
            if item in {"isNegated", "isFamily", "isHistorical"}:
                out.add(item)
        elif isinstance(item, dict):
            for k, v in item.items():
                if isinstance(v, str) and v in {"isNegated", "isFamily", "isHistorical"}:
                    out.add(v)
                elif isinstance(k, str) and k in {"isNegated", "isFamily", "isHistorical"} and v:
                    out.add(k)
    return sorted(out)


def preprocess_input_for_llm(
    input_text: str,
    max_chars: int = _INPUT_MAX_CHARS,
) -> str:
    """Clean clinical note input trước khi gửi LLM — tránh context overflow.

    Các bước:
      1. Strip markdown noise: ** (bold), _ (italic), `#` headers
      2. Normalize whitespace (collapse multiple spaces/newlines)
      3. Drop empty lines
      4. Drop pure-placeholder lines (chỉ chứa 'N/A', ': N/A', '-')
      5. Dedupe consecutive duplicate lines (case-insensitive)
      6. Nếu vẫn > max_chars: giữ first 60% + marker + last 30%
         (giữ nguyên tiền sử + lý do nhập viện + kết quả xét nghiệm — phần clinical info)

    Args:
        input_text: raw clinical note
        max_chars: cap chiều dài sau clean (default 4000)

    Returns:
        Cleaned input text. Nếu không cần clean → trả về input_text nguyên.
    """
    if not input_text or len(input_text) <= max_chars:
        return input_text

    text = input_text

    # 1. Strip markdown: ** ** (bold), _ _ (italic), `#` (header)
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"_+", "", text)
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)

    # 2. Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 3. Drop empty lines
    lines = [ln for ln in text.split("\n") if ln.strip()]
    text = "\n".join(lines)

    # 4. Drop pure-placeholder lines (chỉ N/A, : N/A, -, ...)
    placeholder_pattern = re.compile(r"^\s*[-\s]*(:\s*)?N/?A\s*[.:]?\s*$", re.IGNORECASE)
    lines = text.split("\n")
    lines = [ln for ln in lines if not placeholder_pattern.match(ln)]
    text = "\n".join(lines)

    # 5. Dedupe consecutive duplicate lines
    deduped: list[str] = []
    prev = None
    for ln in text.split("\n"):
        key = ln.strip().lower()
        if key and key != prev:
            deduped.append(ln)
        prev = key
    text = "\n".join(deduped)

    # 6. Truncate nếu vẫn dài
    if len(text) > max_chars:
        head_size = int(max_chars * 0.6)
        tail_size = int(max_chars * 0.3)
        head = text[:head_size]
        tail = text[-tail_size:]
        text = head + _TRUNCATION_MARKER + tail

    return text


# ---------------------------------------------------------------------- #
# Position fixing
# ---------------------------------------------------------------------- #


def _find_span(text: str, snippet: str, start: int = 0) -> tuple[int, int] | None:
    """Tìm vị trí ĐẦU TIÊN của snippet trong text với WORD-BOUNDARY check (R28 2026-07-13).

    Args:
        text: full input text.
        snippet: text cần tìm.
        start: vị trí bắt đầu tìm (default 0).

    Returns:
        (start, end) tuple hoặc None nếu không tìm được tại word boundary.

    R28 NOTE: Trước đây chỉ dùng `text.find()` mà KHÔNG enforce word boundary
    → match nhầm "giác" bên trong "ảo giácxuất hiện". Nay enforce boundary.
    """
    if not snippet:
        return None
    n_text = len(text)
    n_snip = len(snippet)
    # Case-sensitive pass
    idx = start - 1
    while True:
        idx = text.find(snippet, idx + 1)
        if idx < 0:
            break
        end_idx = idx + n_snip
        # Word boundary: char trước/sau phải là non-alnum hoặc là đầu/cuối text
        prev_ok = idx == 0 or not text[idx - 1].isalnum()
        next_ok = end_idx >= n_text or not text[end_idx].isalnum()
        if prev_ok and next_ok:
            return idx, end_idx
    # Case-insensitive pass (fallback)
    tl = text.lower()
    sl = snippet.lower()
    idx = start - 1
    while True:
        idx = tl.find(sl, idx + 1)
        if idx < 0:
            break
        end_idx = idx + len(sl)
        if end_idx > n_text:
            end_idx = n_text
        prev_ok = idx == 0 or not text[idx - 1].isalnum()
        next_ok = end_idx >= n_text or not text[end_idx].isalnum()
        if prev_ok and next_ok:
            return idx, end_idx
    # Fallback 3: bỏ khoảng trắng thừa ở hai đầu
    stripped = snippet.strip()
    if stripped != snippet:
        idx = start - 1
        while True:
            idx = text.find(stripped, idx + 1)
            if idx < 0:
                break
            end_idx = idx + len(stripped)
            prev_ok = idx == 0 or not text[idx - 1].isalnum()
            next_ok = end_idx >= n_text or not text[end_idx].isalnum()
            if prev_ok and next_ok:
                return idx, end_idx
    return None


def _validate_span_or_drop(input_text: str, ent: dict[str, Any]) -> dict[str, Any] | None:
    """R28 (2026-07-13): Post-alignment validator cho 1 entity.

    Drops entity nếu:
    1. input_text[span_start:span_end] != entity_text (case-insensitive)
       → span không trỏ vào đúng text được claim
    2. Span boundaries nằm giữa từ (char trước/sau là alnum)
       → span cắt giữa từ, làm scorer match sai

    Args:
        input_text: original input
        ent: entity dict với 'text' + 'position'

    Returns:
        ent nếu OK, None nếu cần drop.
    """
    if not isinstance(ent, dict):
        return None
    pos = ent.get("position", [])
    if not (isinstance(pos, list) and len(pos) == 2):
        return None
    try:
        s, e = int(pos[0]), int(pos[1])
    except (ValueError, TypeError):
        return None
    n = len(input_text)
    if not (0 <= s < e <= n):
        return None
    actual = input_text[s:e]
    expected = str(ent.get("text", "")).strip()
    if not expected:
        return None
    # Check 1: extracted text khớp substring (case-insensitive)
    if actual.lower() != expected.lower():
        return None
    # Check 2: word-boundary (cả 2 phía)
    if s > 0 and input_text[s - 1].isalnum():
        return None
    if e < n and input_text[e].isalnum():
        return None
    return ent


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
        if not text:
            continue

        pos = ent.get("position", [])
        start, end = 0, 0
        # Trường hợp 1: LLM có cung cấp position (cũ)
        if isinstance(pos, list) and len(pos) == 2:
            start, end = int(pos[0]), int(pos[1])
            # Sanity bounds
            if start < 0:
                start = 0
            if end > len(input_text):
                end = len(input_text)
            # Nếu substring không khớp → tìm lại
            if input_text[start:end] != text:
                # Fix case-sensitivity (mới 2026-07): nếu chỉ khác case → tìm lại
                # case-insensitive và UPDATE text thành correct case trong input
                # (vd "khó thở nhẹ" → "Khó thở nhẹ")
                if input_text[start:end].lower() == text.lower():
                    text = input_text[start:end]
                else:
                    start, end = 0, 0  # force re-find bên dưới

        # Trường hợp 2: LLM KHÔNG cung cấp position (mới) → TỰ TÌM
        if start == 0 and end == 0:
            found = _find_span(input_text, text)
            if found is None:
                # Fallback: thử case-insensitive để recover entities có case sai
                ci_idx = input_text.lower().find(text.lower())
                if ci_idx >= 0:
                    actual_text = input_text[ci_idx:ci_idx + len(text)]
                    if actual_text.lower() == text.lower():
                        found = (ci_idx, ci_idx + len(text))
                        text = actual_text  # update text sang correct case
                        logger.debug(
                            "Recovered case-mismatched entity '%s' → '%s'",
                            ent.get("text", ""), text,
                        )
            # R23 typo recovery: nếu text bị dính (vd "atenololtrong" → "atenolol")
            if found is None:
                # Pass hint_pos = LLM's original pos để tìm occurrence gần nhất
                # (tránh recover sai khi drug name xuất hiện nhiều nơi)
                hint_pos = int(ent.get("position", [0, 0])[0]) if isinstance(ent.get("position"), list) else 0
                recovered = _try_recover_typo(text, input_text, hint_pos=hint_pos)
                if recovered is not None:
                    recovered_text, start_recovered, end_recovered = recovered
                    found = (start_recovered, end_recovered)
                    text = recovered_text
                    logger.debug(
                        "Recovered typo entity '%s' → '%s' at pos %d",
                        ent.get("text", ""), text, start_recovered,
                    )
            if found is None:
                logger.debug("Bỏ entity không tìm được: %r", text)
                continue
            start, end = found

        out.append({**ent, "text": text, "position": [start, end]})
    return out


# ==============================================================================
# STAGE 1 MENTION BOUNDARY VALIDATION (R32 - 2026-07-12)
# ==============================================================================

def _try_recover_position(input_text: str, text: str, hint_pos: int, window: int = 50) -> dict[str, Any] | None:
    """Tìm text trong input xung quanh hint_pos ± window chars cho Stage 1 mentions."""
    text_lower = text.lower()
    occurrences: list[int] = []
    start = 0
    while True:
        idx = input_text.lower().find(text_lower, start)
        if idx < 0:
            break
        occurrences.append(idx)
        start = idx + 1
        if len(occurrences) > 20:
            break

    if not occurrences:
        # Sử dụng _find_span có sẵn với recovery sliding window
        span = _find_span(input_text, text, start=max(0, hint_pos - 15))
        if span is not None:
            return {"text": input_text[span[0]:span[1]], "position": [span[0], span[1]]}
        # Thử fuzzy match sliding window ratio >= 80 nếu span là None
        try:
            from rapidfuzz import fuzz
            candidates: list[tuple[int, float]] = []
            step = max(1, len(text) // 2)
            for i in range(0, max(1, len(input_text) - len(text) + 1), step):
                window_text = input_text[i : i + len(text) + 10]
                score = fuzz.ratio(text_lower, window_text.lower())
                if score >= 80:
                    candidates.append((i, score))
            if candidates:
                best = min(candidates, key=lambda c: abs(c[0] - hint_pos))
                return {"text": input_text[best[0] : best[0] + len(text)], "position": [best[0], best[0] + len(text)]}
        except ImportError:
            pass
        return None

    best_pos = min(occurrences, key=lambda p: abs(p - hint_pos))
    return {"text": input_text[best_pos : best_pos + len(text)], "position": [best_pos, best_pos + len(text)]}


def _boost_and_split_stage1_mentions(input_text: str, mentions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tách gộp (Compound Splitting) và bổ sung sinh hiệu/thuốc cốt lõi (Recall Booster) cho Stage 1."""
    # 1. Compound Splitting (Tách cụm gộp)
    split_mentions: list[dict[str, Any]] = []
    for m in mentions:
        text = str(m.get("text", "")).strip()
        pos = m.get("position", [0, 0])
        if not isinstance(pos, list) or len(pos) != 2:
            split_mentions.append(m)
            continue
        s, e = int(pos[0]), int(pos[1])
        
        # Tách cấu trúc: Tên XN (Abbr) -> ví dụ "điện tâm đồ (ECG)" -> "điện tâm đồ", "ECG"
        m_paren = re.match(r"^(.+?)\s*\((.+?)\)$", text)
        if m_paren and len(text) > 8:
            part1, part2 = m_paren.group(1).strip(), m_paren.group(2).strip()
            idx1 = input_text.find(part1, max(0, s - 5), min(len(input_text), e + 5))
            idx2 = input_text.find(part2, max(0, s - 5), min(len(input_text), e + 5))
            if idx1 >= 0 and idx2 >= 0:
                split_mentions.append({"text": input_text[idx1:idx1+len(part1)], "position": [idx1, idx1+len(part1)]})
                split_mentions.append({"text": input_text[idx2:idx2+len(part2)], "position": [idx2, idx2+len(part2)]})
                continue

        # Tách theo từ nối " hoặc " (vd "khó thở hoặc ho" → "khó thở", "ho")
        if " hoặc " in text.lower():
            parts = re.split(r"\s+hoặc\s+", text, flags=re.IGNORECASE)
            if len(parts) >= 2:
                curr_offset = s
                temp_list = []
                for p in parts:
                    p_clean = p.strip()
                    if len(p_clean) >= 2:
                        idx = input_text.find(p_clean, curr_offset, min(len(input_text), e + 5))
                        if idx >= 0:
                            temp_list.append({"text": input_text[idx:idx+len(p_clean)], "position": [idx, idx+len(p_clean)]})
                            curr_offset = idx + len(p_clean)
                if len(temp_list) >= 2:
                    split_mentions.extend(temp_list)
                    continue

        # Tách theo từ nối " và " hoặc ", " nếu span dài > 24 chars và không phải bệnh/thuốc gộp cố định
        if (" và " in text or ", " in text) and len(text) > 24 and not re.search(r"(?:tràn dịch|tràn khí|động mạch|tĩnh mạch|nhồi máu|viêm|thuốc)", text, re.IGNORECASE):
            parts = re.split(r"\s+và\s+|,\s*", text)
            curr_offset = s
            temp_list = []
            for p in parts:
                p_clean = p.strip()
                if len(p_clean) >= 3:
                    idx = input_text.find(p_clean, curr_offset, min(len(input_text), e + 5))
                    if idx >= 0:
                        temp_list.append({"text": input_text[idx:idx+len(p_clean)], "position": [idx, idx+len(p_clean)]})
                        curr_offset = idx + len(p_clean)
            if len(temp_list) >= 2:
                split_mentions.extend(temp_list)
                continue

        split_mentions.append(m)

    # 2. Recall Booster (Mention Injection cho Sinh hiệu / Định lượng cốt lõi)
    existing_spans = {(item["position"][0], item["position"][1]) for item in split_mentions if "position" in item and len(item["position"]) == 2}
    
    # Các regex cốt lõi cần quét bổ sung
    patterns = [
        r"\b(?:HA|Huyết\s+áp)\s*[:=]?\s*\d{2,3}/\d{2,3}\s*(?:mmHg)?\b",
        r"\b\d{2,3}/\d{2,3}\s*mmHg\b",
        r"\b(?:SpO2|Sp02)\s*[:=]?\s*\d{2,3}\s*%\b",
        r"\b(?:Mạch|Tần\s+số\s+tim|Nhịp\s+tim)\s*[:=]?\s*\d{2,3}\s*(?:lần/phút|nhịp/phút)?\b",
        r"\b(?:EF|LVEF)\s*[:=]?\s*\d{2,3}\s*%\b",
        r"\bVS\d+\.\d+\s+(?:\d+\s+){3,5}\d+[A-Z0-9]*\b",
    ]
    for pat in patterns:
        for match in re.finditer(pat, input_text, re.IGNORECASE):
            ms, me = match.start(), match.end()
            if not any(max(ms, es) < min(me, ee) for (es, ee) in existing_spans):
                split_mentions.append({"text": match.group(0).strip(), "position": [ms, me]})
                existing_spans.add((ms, me))

    split_mentions.sort(key=lambda x: x["position"][0] if "position" in x and len(x["position"]) == 2 else 0)
    return split_mentions


def _validate_stage1_mentions(input_text: str, mentions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate Stage 1 mentions: drop invalid length/noise, fix positions exact/fuzzy.

    Returns list of {"text": str, "position": [start, end]} đã được chuẩn hóa.
    """
    valid: list[dict[str, Any]] = []
    seen_spans: set[tuple[int, int]] = set()

    for m in mentions:
        if not isinstance(m, dict):
            continue
        text = str(m.get("text", "")).strip()
        pos = m.get("position", [0, 0])
        if not isinstance(pos, list) or len(pos) != 2:
            pos = [0, 0]
        hint_pos = int(pos[0]) if isinstance(pos[0], int) else 0

        # 1. Drop nếu text > 60 chars (trừ khi chứa định lượng thuốc rõ ràng)
        if len(text) > 60 and not re.search(r"\d+\s*(mg|mcg|g|ml|iu|viên|ống|gói|x\s*\d)", text, re.IGNORECASE):
            continue

        # 2. Drop nếu text > 40 chars và chứa dấu câu narrative (.,;:)
        if len(text) > 40 and any(c in text for c in ".,;:"):
            continue

        # 3. Drop theo noise patterns (_DROP_NOISE_PATTERNS / pure duration)
        if any(p.match(text) for p in _DROP_NOISE_PATTERNS) or _PURE_DURATION_ENHANCED_RE.match(text):
            continue

        # 4. Exact validation t[pos[0]:pos[1]] vs input
        if 0 <= pos[0] < pos[1] <= len(input_text):
            actual_text = input_text[pos[0]:pos[1]]
            if actual_text.lower() == text.lower():
                span = (pos[0], pos[1])
                if span not in seen_spans:
                    seen_spans.add(span)
                    valid.append({"text": actual_text, "position": [pos[0], pos[1]]})
                continue

        # 5. Fuzzy / closest recovery
        recovered = _try_recover_position(input_text, text, hint_pos)
        if recovered is not None:
            rpos = recovered["position"]
            span = (rpos[0], rpos[1])
            if span not in seen_spans:
                seen_spans.add(span)
                valid.append(recovered)
            continue

    return _boost_and_split_stage1_mentions(input_text, valid)


def _refine_stage2_results(input_text: str, stage2_entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Kiểm duyệt & tinh chỉnh nhãn type và assertions sau Stage 2 theo luật chuyên gia."""
    refined: list[dict[str, Any]] = []
    
    drug_endings = re.compile(r"\d+\s*(?:mg|mcg|g|ml|iu|viên|ống|gói|po|bid|tid|daily|prn)$", re.IGNORECASE)
    drug_names = _DRUG_NAMES_UNIONED  # R28: 13 → ~63k entries auto-extended
    normal_patterns = re.compile(r"^(?:bình\s+thường|không\s+ghi\s+nhận.*|không\s+có\s+gì\s+đáng\s+chú\s+ý)$", re.IGNORECASE)
    vital_patterns = re.compile(r"^(?:\d{2,3}/\d{2,3}\s*(?:mmHg)?|SpO2.*|\d{2,3}\s*%|VS\d+\.\d+.*|\d{2,3}\s*(?:lần/phút|nhịp/phút|\s*°C))$", re.IGNORECASE)
    test_names = {"điện tâm đồ", "ecg", "x-quang ngực", "siêu âm tim", "siêu âm tim qua thành ngực", "phân tích nước tiểu", "công thức máu", "monitor holter", "chụp x-quang"}

    for ent in stage2_entities:
        if not isinstance(ent, dict):
            continue
        text = str(ent.get("text", "")).strip()
        pos = ent.get("position", [0, 0])
        etype = str(ent.get("type", "")).strip()
        assertions = list(ent.get("assertions", [])) if isinstance(ent.get("assertions"), list) else []

        # Filter out non-medical administrative actions from TÊN_XÉT_NGHIỆM
        if etype == "TÊN_XÉT_NGHIỆM" and re.search(r"\b(?:gọi\s+xe\s+cứu\s+thương|chuyển\s+viện|nhập\s+viện|ra\s+viện|lái\s+xe|đi\s+lại|hỏi\s+bệnh|tái\s+khám)\b", text, re.IGNORECASE):
            continue

        # Filter out non-symptom sentence fragments from TRIỆU_CHỨNG
        if etype == "TRIỆU_CHỨNG" and re.search(r"\b(?:lái\s+xe\s+sau\s+ngã|thể\s+chịu\s+trọng\s+lượng|nằm\s+tại\s+giường|không\s+thể\s+tự\s+di\s+chuyển)\b", text, re.IGNORECASE):
            continue

        # A. Auto-Type Correction (R28: order matters)
        if normal_patterns.match(text) or vital_patterns.match(text):
            etype = "KẾT_QUẢ_XÉT_NGHIỆM"
        elif _is_procedure(text):
            # R28: Kiểm tra TRƯỚC drug để tránh "truyền dịch yếu tố IX" → THUỐC
            etype = "TÊN_XÉT_NGHIỆM"
        elif drug_endings.search(text) or text.lower() in _COMMON_DRUG_NAMES:
            etype = "THUỐC"
        elif text.lower() in test_names:
            etype = "TÊN_XÉT_NGHIỆM"

        # B. Assertion Cross-Validation & Refinement
        if etype in ("KẾT_QUẢ_XÉT_NGHIỆM", "TÊN_XÉT_NGHIỆM"):
            assertions = []
        elif isinstance(pos, list) and len(pos) == 2 and etype in ("CHẨN_ĐOÁN", "THUỐC", "TRIỆU_CHỨNG"):
            try:
                s = int(pos[0])
                if 0 <= s < len(input_text):
                    pre_window = input_text[max(0, s - 16):s].lower()
                    # Only check pre_window if within the SAME line/sentence (no \n or .)
                    if "\n" not in pre_window and "." not in pre_window:
                        if re.search(r"\b(?:không|chưa|chẳng)\s+(?:có\s+)?$", pre_window) and not re.search(r"\b(?:tuân\s+thủ|rõ|thể|biết|dùng)\s*$", pre_window):
                            if "isNegated" not in assertions:
                                assertions.append("isNegated")

                    section_id = _find_current_section(input_text, s)
                    if section_id == "tien_su":
                        if "isHistorical" not in assertions:
                            assertions.append("isHistorical")
                    elif section_id in ("hien_tai", "ly_do", "danh_gia", "khám"):
                        assertions = [a for a in assertions if a != "isHistorical"]

                    rule_assertions = _detect_assertions_from_context(text, input_text, etype, s)
                    for ra in rule_assertions:
                        if ra in ("isNegated", "isFamily", "isHistorical") and ra not in assertions:
                            assertions.append(ra)
            except (ValueError, TypeError):
                pass

        seen_a = set()
        clean_a = []
        for a in assertions:
            if a in ("isNegated", "isHistorical", "isFamily") and a not in seen_a:
                seen_a.add(a)
                clean_a.append(a)

        ent["type"] = etype
        ent["assertions"] = clean_a
        refined.append(ent)

    return refined


def _stage2_fallback_classify(mentions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Smart Fallback Classifier cho Stage 2 (dùng khi LLM trả về lô rỗng hoặc timeout)."""
    fallback_list: list[dict[str, Any]] = []
    drug_endings = re.compile(r"\d+\s*(?:mg|mcg|g|ml|iu|viên|ống|gói|po|bid|tid|daily|prn)$", re.IGNORECASE)
    drug_names = _DRUG_NAMES_UNIONED  # R28: 13 → ~63k entries auto-extended
    normal_patterns = re.compile(r"^(?:bình\s+thường|không\s+ghi\s+nhận.*|không\s+có\s+gì\s+đáng\s+chú\s+ý)$", re.IGNORECASE)
    vital_patterns = re.compile(r"^(?:\d{2,3}/\d{2,3}\s*(?:mmHg)?|SpO2.*|\d{2,3}\s*%|VS\d+\.\d+.*|\d{2,3}\s*(?:lần/phút|nhịp/phút|\s*°C))$", re.IGNORECASE)
    test_names = {"điện tâm đồ", "ecg", "x-quang ngực", "siêu âm tim", "siêu âm tim qua thành ngực", "phân tích nước tiểu", "công thức máu", "monitor holter", "chụp x-quang"}
    symptom_hints = {"đau ", "khó thở", "sốt", "mệt mỏi", "đánh trống ngực", "thắt chặt ngực", "buồn nôn", "chóng mặt", "ho ", "phù "}

    for m in mentions:
        if not isinstance(m, dict):
            continue
        text = str(m.get("text", "")).strip()
        pos = m.get("position", [0, 0])
        if not text:
            continue
        tl = text.lower()

        if normal_patterns.match(text) or vital_patterns.match(text):
            etype = "KẾT_QUẢ_XÉT_NGHIỆM"
        elif _is_procedure(text):
            # R28: ưu tiên procedure trước drug (vd "truyền dịch yếu tố IX" → procedure)
            etype = "TÊN_XÉT_NGHIỆM"
        elif drug_endings.search(text) or tl in _COMMON_DRUG_NAMES:
            etype = "THUỐC"
        elif tl in test_names or "xét nghiệm" in tl or "chụp" in tl or "siêu âm" in tl:
            etype = "TÊN_XÉT_NGHIỆM"
        elif any(sh in tl for sh in symptom_hints):
            etype = "TRIỆU_CHỨNG"
        else:
            etype = "CHẨN_ĐOÁN"

        fallback_list.append({
            "text": text,
            "position": pos,
            "type": etype,
            "assertions": []
        })

    return fallback_list


def _find_closest_occurrence(
    text: str, substring: str, hint_pos: int
) -> int | None:
    """Tìm occurrence của substring gần hint_pos nhất.

    Dùng cho R23 typo recovery: nếu drug name xuất hiện NHIỀU LẦN trong input
    (vd "atenolol" ở cả dòng 5 và dòng 36), chọn vị trí GẦN entity_pos nhất
    để recover đúng context.

    Args:
        text: input text để tìm.
        substring: chuỗi cần tìm.
        hint_pos: vị trí mong muốn (entity_pos của entity gốc).

    Returns: vị trí index của occurrence gần hint_pos nhất, hoặc None nếu không tìm thấy.
    """
    if not text or not substring:
        return None
    text_lower = text.lower()
    sub_lower = substring.lower()
    sub_len = len(sub_lower)

    best_idx = None
    best_dist = float("inf")

    start = 0
    while True:
        idx = text_lower.find(sub_lower, start)
        if idx < 0:
            break
        dist = abs(idx - hint_pos)
        if dist < best_dist:
            best_dist = dist
            best_idx = idx
        # Optimization: nếu đã tìm thấy match exact tại hint_pos thì dừng
        if dist == 0:
            break
        start = idx + 1

    return best_idx


# VN particles thường bị dính với drug names
_VN_PARTICLES = (
    "trong", "ngày", "hôm", "nay", "qua", "sáng", "tối", "chiều", "trưa",
    "lúc", "khi", "đang", "đã", "sẽ", "với", "cho", "của", "từ",
    "uống", "tiêm", "dùng", "trước", "sau", "ăn", "nghỉ", "trị",
    "điều", "trị", "việc", "nhà", "bệnh", "viện", "mạch", "vào", "ra",
)


def _try_recover_typo(
    text: str, input_text: str, hint_pos: int = 0
) -> tuple[str, int, int] | None:
    """R23: thử recover entity bị typo dính chữ.

    Patterns:
    1. Drug name + VN particle (vd "atenololtrong" → "atenolol")
       → match first word trong _COMMON_DRUG_NAMES, sau đó là particle.
    2. "cảm giác" + adjective dính (vd "cảm giáckhó chịu" → "cảm giác khó chịu")
       → match "cảm giác" + " " + VN/EN word.

    Args:
        text: entity text bị typo (vd "atenololtrong").
        input_text: toàn bộ input.
        hint_pos: vị trí gốc của entity trong input (để tìm match gần nhất,
            tránh recover sai nếu drug name xuất hiện nhiều nơi).

    Returns: (recovered_text, start, end) nếu match, None nếu không.
    """
    if not text or len(text) < 5:
        return None

    text_lower = text.lower().strip()

    # Pattern 1: Drug name + VN particle dính (không có space)
    # VD: "atenololtrong" → "atenolol" + "trong"
    for drug in _COMMON_DRUG_NAMES:
        if text_lower.startswith(drug) and len(text_lower) > len(drug):
            suffix = text_lower[len(drug):]
            # Suffix phải là particle (không có digit, không có strength unit)
            if not suffix[0].isdigit() and suffix in _VN_PARTICLES:
                # Tìm vị trí drug name gần hint_pos nhất (tránh match sai ở vị trí khác)
                idx = _find_closest_occurrence(input_text, drug, hint_pos)
                if idx is not None and idx >= 0:
                    return (input_text[idx:idx + len(drug)], idx, idx + len(drug))

    # Pattern 2: "cảm giác" + adjective dính → "cảm giác " + adjective
    for prefix in ("cảm giác", "triệu chứng", "dấu hiệu"):
        if text_lower.startswith(prefix) and len(text_lower) > len(prefix):
            suffix = text_lower[len(prefix):]
            # Suffix phải là 1 từ (vd "khó", "đau", "nặng")
            if suffix and not suffix[0].isspace() and len(suffix) <= 20:
                # Thêm space
                idx = _find_closest_occurrence(input_text, prefix, hint_pos)
                if idx is not None and idx >= 0:
                    # Kiểm tra text tiếp theo trong input có phải space + word không
                    next_start = idx + len(prefix)
                    if next_start < len(input_text) and input_text[next_start] == " ":
                        # Đã có space, không cần recover
                        continue
                    # Recover: thêm space
                    recovered_text = input_text[idx:idx + len(prefix)] + " " + suffix
                    # Chỉ return nếu match tìm được trong input
                    if recovered_text.lower() in input_text.lower():
                        ri = _find_closest_occurrence(input_text, recovered_text, hint_pos)
                        if ri is not None:
                            return (input_text[ri:ri + len(recovered_text)], ri, ri + len(recovered_text))

    return None


def _fuzzy_locate_in_text(
    target: str, source_text: str, hint_pos: int = 0
) -> tuple[int, int] | None:
    """Pass 4 & 5: Tìm vị trí [start, end] chính xác trong source_text cho target (khác dấu, khoảng trắng, typo)."""
    if not target or not source_text or len(target) < 3:
        return None

    # 1. Strip accents & whitespace normalization với 1:1 index mapping
    norm_chars = []
    orig_indices = []
    for i, c in enumerate(source_text):
        if c in "đĐ":
            nc = "d"
        else:
            nc = "".join(
                ch for ch in unicodedata.normalize("NFD", c) if unicodedata.category(ch) != "Mn"
            )
        for k_ch in nc.lower():
            norm_chars.append(k_ch)
            orig_indices.append(i)

    norm_source = "".join(norm_chars)

    t_norm_chars = []
    for c in target:
        if c in "đĐ":
            nc = "d"
        else:
            nc = "".join(
                ch for ch in unicodedata.normalize("NFD", c) if unicodedata.category(ch) != "Mn"
            )
        t_norm_chars.append(nc.lower())
    norm_target = "".join(t_norm_chars)

    if len(norm_target) >= 3:
        best_idx = -1
        best_dist = 999999
        start = 0
        while True:
            idx = norm_source.find(norm_target, start)
            if idx < 0:
                break
            orig_start = orig_indices[idx]
            orig_end = (
                orig_indices[min(idx + len(norm_target) - 1, len(orig_indices) - 1)] + 1
            )
            dist = abs(orig_start - hint_pos)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
            start = idx + 1
        if best_idx >= 0:
            orig_start = orig_indices[best_idx]
            orig_end = (
                orig_indices[min(best_idx + len(norm_target) - 1, len(orig_indices) - 1)] + 1
            )
            return (orig_start, orig_end)

    # 2. Sliding window RapidFuzz recovery cho typo nhầm ký tự hoặc lệch từ
    try:
        from rapidfuzz import fuzz

        best_score = 0.0
        best_span: tuple[int, int] | None = None
        t_len = len(target)
        words = [(m.start(), m.end()) for m in re.finditer(r"\S+", source_text)]
        if not words:
            return None
        for i in range(len(words)):
            w_start = words[i][0]
            for j in range(i, min(i + 16, len(words))):
                w_end = words[j][1]
                if abs((w_end - w_start) - t_len) > max(12, int(t_len * 0.45)):
                    continue
                cand = source_text[w_start:w_end]
                score = float(fuzz.ratio(target.lower(), cand.lower()))
                if score >= 82.0 and score > best_score:
                    best_score = score
                    best_span = (w_start, w_end)
        return best_span if best_score >= 82.0 else None
    except Exception:
        return None


# ---------------------------------------------------------------------- #
# Dedupe
# ---------------------------------------------------------------------- #


def _is_semantic_overlap(text_a: str, text_b: str) -> bool:
    """Check if two strings have exact match, substring containment, or high Jaccard overlap (Upgrade 1)."""
    a = text_a.strip().lower()
    b = text_b.strip().lower()
    if not a or not b:
        return False
    if a == b:
        return True
    # Check for conflicting numbers, letters, or antonyms (type 1 vs type 2, trái vs phải)
    nums_a = set(re.findall(r'\b\d+\b', a))
    nums_b = set(re.findall(r'\b\d+\b', b))
    if nums_a and nums_b and nums_a != nums_b:
        return False
    antonyms = [("trái", "phải"), ("cấp", "mạn"), ("cấp", "mãn"), ("trên", "dưới"), ("trong", "ngoài"), ("tăng", "giảm"), ("cao", "thấp")]
    for w1, w2 in antonyms:
        if (re.search(r'\b' + re.escape(w1) + r'\b', a) and re.search(r'\b' + re.escape(w2) + r'\b', b)) or \
           (re.search(r'\b' + re.escape(w2) + r'\b', a) and re.search(r'\b' + re.escape(w1) + r'\b', b)):
            return False
    tokens_a = set(re.findall(r'[a-zà-ỹ0-9_/-]+', a)) - {"bệnh", "chứng", "tình", "trạng", "bị", "có", "do", "và", "của"}
    tokens_b = set(re.findall(r'[a-zà-ỹ0-9_/-]+', b)) - {"bệnh", "chứng", "tình", "trạng", "bị", "có", "do", "và", "của"}
    if a in b or b in a or tokens_a.issubset(tokens_b) or tokens_b.issubset(tokens_a):
        diff = (tokens_b - tokens_a) if (tokens_a.issubset(tokens_b) or a in b) else (tokens_a - tokens_b)
        # Nguyên tắc tổng quát (Zero-Hardcoding cơ quan giải phẫu):
        # Nếu phần khác biệt (diff) chứa >= 2 từ khóa riêng biệt (hoặc cụm từ có độ dài đáng kể),
        # chứng tỏ 2 dải có phạm vi định vị/hoàn cảnh khác nhau (vd: "cảm giác..." vs "...vùng trước tim", "...hố chậu phải").
        degree_words = {"độ", "giai", "đoạn", "cấp", "mạn", "mãn", "nhẹ", "nặng", "vừa", "nhiều", "ít", "từng", "cơn", "i", "ii", "iii", "iv", "1", "2", "3", "4"}
        if len(diff) >= 2 and not diff.issubset(degree_words):
            return False
        return True
    jaccard = len(tokens_a & tokens_b) / max(len(tokens_a | tokens_b), 1)
    return jaccard >= 0.80


def dedupe_entities(
    entities: Iterable[dict[str, Any]],
    *,
    mode: str = "merge",
) -> list[dict[str, Any]]:
    """Merge / drop / report overlap giữa các entities cùng type + position overlap.

    R10 STRICT + R22 OVERLAP (2026-07-10) → refactored 2026-07-14 với 3 modes:

    3 MODES (chọn bằng `mode=`):

    1. **mode="merge" (DEFAULT MỚI, 2026-07-14)**:
       - Với mỗi CLUSTER entities cùng type + position overlap:
         → Merge thành 1 entity duy nhất (giữ span DÀI NHẤT).
         → Union `assertions` từ tất cả members.
         → Union `candidates` từ tất cả members.
         → Mark `_merged_from: list[int]` (sorted indices) trên entity đại diện.
       - KHÔNG drop entities. Khác với logic cũ.
       - Lý do đổi (user feedback 2026-07-14): "xem overlap chứ ko phải bỏ bớt
         entities giống nhau" + tránh mất entities khi có nhiều cluster overlap.

    2. **mode="drop" (LEGACY R10/R22)**:
       - Cùng text + type + position overlap → drop shorter span (giữ span dài hơn).
       - Backward compat với code cũ.

    3. **mode="report"**:
       - KHÔNG merge / KHÔNG drop. Chỉ mark `_overlap_with: list[int]` (sorted)
         trên mỗi entity có overlap với peer.

    Args:
        entities: list entities (cần có 'text', 'type', 'position' [s,e]).
        mode: 'merge' (default) | 'drop' | 'report'.

    Returns:
        list[dict] các entities. Mỗi entity có thể có:
        - `_merged_from: list[int]` (mode='merge') — indices bị merge vào đây.
        - `_overlap_with: list[int]` (mode='report') — indices của peer overlap.
        - (mode='drop') — không có mark.
    """
    # 1. Validate entities (giữ format đồng nhất)
    valid: list[dict[str, Any]] = []
    for e in entities:
        if not e.get("text"):
            continue
        pos = e.get("position", [0, 0])
        if not (isinstance(pos, list) and len(pos) == 2):
            continue
        try:
            s, end = int(pos[0]), int(pos[1])
        except (TypeError, ValueError):
            continue
        if s < 0 or end <= s:
            continue
        valid.append({**e, "position": [s, end]})

    if mode == "drop":
        # LEGACY: drop shorter span khi overlap (giữ logic cũ)
        out: list[dict[str, Any]] = []
        # Sort theo start ASC, length DESC (span dài xử lý trước)
        def _sort_key(e: dict[str, Any]) -> tuple[int, int]:
            s, e_end = e["position"]
            return (s, -(e_end - s))

        sorted_ents = sorted(valid, key=_sort_key)
        for ent in sorted_ents:
            etype = ent.get("type", "")
            text = str(ent.get("text", "")).strip()
            start, end = ent["position"]
            is_duplicate = False
            to_remove: list[int] = []
            for idx, existing in enumerate(out):
                if existing.get("type", "") != etype:
                    continue
                ex_text = str(existing.get("text", "")).strip()
                ex_pos = existing["position"]
                e_start, e_end = ex_pos
                is_exact_text = (ex_text.lower() == text.lower())
                is_pos_overlap = (max(start, e_start) < min(end, e_end))
                if not is_exact_text:
                    if not is_pos_overlap or not _is_semantic_overlap(ex_text, text):
                        continue
                # Same exact span → drop current (R22)
                if start == e_start and end == e_end:
                    is_duplicate = True
                    break
                if is_pos_overlap:
                    ex_len = e_end - e_start
                    cur_len = end - start
                    if ex_len >= cur_len:
                        is_duplicate = True
                        break
                    else:
                        to_remove.append(idx)
            for idx in reversed(to_remove):
                out.pop(idx)
            if not is_duplicate:
                out.append(ent)
        out.sort(key=lambda e: e["position"][0])
        return out

    if mode == "report":
        # Report-only: KHÔNG merge / KHÔNG drop, chỉ mark _overlap_with
        # Sort by start ASC cho indices ổn định
        sorted_ents = sorted(valid, key=lambda e: e["position"][0])
        pairs = _find_position_overlap_pairs(sorted_ents)
        overlap_map: dict[int, list[int]] = {}
        for p in pairs:
            i, j = p["idx_a"], p["idx_b"]
            overlap_map.setdefault(i, []).append(j)
            overlap_map.setdefault(j, []).append(i)
        annotated: list[dict[str, Any]] = []
        for i, ent in enumerate(sorted_ents):
            if i in overlap_map:
                ent = {**ent, "_overlap_with": sorted(set(overlap_map[i]))}
            annotated.append(ent)
        return annotated

    # mode == "merge" (DEFAULT): Union-find clusters + merge into longest
    # Sort by start ASC để cluster indices ổn định + dễ debug
    sorted_ents = sorted(valid, key=lambda e: e["position"][0])
    n = len(sorted_ents)
    if n == 0:
        return []

    # Union-Find structure
    parent: list[int] = list(range(n))
    rank: list[int] = [0] * n

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    # Build clusters: 2 entities merge-connected nếu cùng type AND position overlap
    positions: list[tuple[int, int]] = [e["position"] for e in sorted_ents]
    types: list[str] = [str(e.get("type", "")) for e in sorted_ents]
    for i in range(n):
        si, ei = positions[i]
        ti = types[i]
        for j in range(i + 1, n):
            sj, ej = positions[j]
            if ti != types[j]:
                continue
            if max(si, sj) < min(ei, ej):
                _union(i, j)

    # Group entities by root
    clusters: dict[int, list[int]] = {}
    for i in range(n):
        root = _find(i)
        clusters.setdefault(root, []).append(i)

    # Build output: 1 representative per cluster
    out: list[dict[str, Any]] = []
    for root, members in clusters.items():
        if len(members) == 1:
            # Singleton: giữ nguyên, không có mark
            out.append(sorted_ents[members[0]])
            continue
        # Pick representative: entity có span dài nhất (tie-break: start ASC)
        def _rep_score(idx: int) -> tuple[int, int]:
            s, e = sorted_ents[idx]["position"]
            return (e - s, -s)  # length desc, start asc (negate for asc)

        rep_idx = max(members, key=_rep_score)
        rep = sorted_ents[rep_idx]
        # Union assertions + candidates từ tất cả members
        all_assertions: set[str] = set()
        all_candidates: list[str] = []
        seen_candidates: set[str] = set()
        for m_idx in members:
            ent = sorted_ents[m_idx]
            for a in ent.get("assertions", []) or []:
                if a in ("isNegated", "isFamily", "isHistorical"):
                    all_assertions.add(a)
            for c in ent.get("candidates", []) or []:
                if c not in seen_candidates:
                    seen_candidates.add(c)
                    all_candidates.append(c)
        merged: dict[str, Any] = {
            **rep,
            "assertions": sorted(all_assertions),
            "candidates": all_candidates,
            "_merged_from": sorted([i for i in members if i != rep_idx]),
        }
        out.append(merged)

    out.sort(key=lambda e: e["position"][0])
    return out


def _find_position_overlap_pairs(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Detect position overlap pairs giữa các entities (pure span-based).

    "Position overlap" = max(s_a, s_b) < min(e_a, e_b). KHÔNG check text/type —
    để caller tự quyết định semantics (vd dedupe_entities chỉ merge cùng type).

    Args:
        entities: list entities (cần có 'position' [s, e] hợp lệ).

    Returns:
        list các pair dicts (sorted theo (idx_a, idx_a)):
        {
            "idx_a": int, "idx_b": int,
            "text_a": str, "text_b": str,
            "type_a": str, "type_b": str,
            "position_a": [s, e], "position_b": [s, e],
            "overlap_chars": int,      # min(e_a, e_b) - max(s_a, s_b)
            "same_text": bool,         # text_a.lower() == text_b.lower()
            "same_type": bool,         # type_a == type_b
        }
    """
    out: list[dict[str, Any]] = []
    n = len(entities)
    for i in range(n):
        ei = entities[i]
        pi = ei.get("position", [0, 0])
        if not (isinstance(pi, list) and len(pi) == 2):
            continue
        try:
            si, ei_end = int(pi[0]), int(pi[1])
        except (TypeError, ValueError):
            continue
        if si < 0 or ei_end <= si:
            continue
        ti = str(ei.get("text", "")).strip()
        typi = str(ei.get("type", ""))
        for j in range(i + 1, n):
            ej = entities[j]
            pj = ej.get("position", [0, 0])
            if not (isinstance(pj, list) and len(pj) == 2):
                continue
            try:
                sj, ej_end = int(pj[0]), int(pj[1])
            except (TypeError, ValueError):
                continue
            if sj < 0 or ej_end <= sj:
                continue
            overlap_chars = min(ei_end, ej_end) - max(si, sj)
            if overlap_chars <= 0:
                continue
            tj = str(ej.get("text", "")).strip()
            typj = str(ej.get("type", ""))
            out.append({
                "idx_a": i,
                "idx_b": j,
                "text_a": ti,
                "text_b": tj,
                "type_a": typi,
                "type_b": typj,
                "position_a": [si, ei_end],
                "position_b": [sj, ej_end],
                "overlap_chars": overlap_chars,
                "same_text": ti.lower() == tj.lower(),
                "same_type": typi == typj,
            })
    return out


# ---------------------------------------------------------------------- #
# Drug text sanitization (R4 + R18)
# ---------------------------------------------------------------------- #

# R34 (2026-07-13): Add lab chemical / non-drug noise patterns để DROP trước lookup.
# LLM hay extract các tokens này như THUỐC nhưng thực chất là lab values, electrolytes.
# Pattern match với optional suffix (số, đơn vị, unit word): "lactate 1.8", "creatinine 1.5 mg/dL".
_DRUG_NAME_BAD_PATTERNS = re.compile(
    r"^("
    r"thuốc|drug|medication|thuoc"
    r"|creatinine|lactate|bicarbonate|chloride|hemoglobin|bilirubin"
    r"|sodium|potassium|calcium|glucose|magnesium|urea|albumin"
    r"|protein|cholesterol|triglyceride|hdl|ldl"
    r")(\s+\d+[\d.,/]*\s*\w*|\s+\w+)*\s*$",
    re.IGNORECASE
)


# Strip prescription suffix "x N + unit" trong drug text (R4 mới 2026-07).
# KEEP "x 1" / "x 2" (dose count), DROP the unit word.
_DRUG_X_N_PATTERN = re.compile(
    r"\s+(?:viên(?:\s+(?:sáng|tối|trưa))?|tablet|tab|lần(?:/ngày)?|ống|gói|ngày)\s*$",
    re.IGNORECASE | re.UNICODE,
)


# SMART parens strip (R18 mới 2026-07): chỉ drop parens chứa admin instruction words.
# KHÔNG drop parens có numerical/clinical info (giữ dose change, concentration, brand abbrev).
# Admin keywords: uống, ăn, trước, sau, food, meal, hôm nay, cùng bữa, with food
# Numerical/clinical: 50mg, 25mg, 5mg/ml, HCl, 200mg/5ml, etc. (digits present)
#
# Heuristic: nếu parens có ≥1 digit → KEEP (clinical data); nếu KHÔNG có digit + có admin word → DROP.
_DRUG_PARENS_PATTERN = re.compile(
    r"\s+\(([^)]*)\)",
    re.UNICODE,
)


def _is_admin_parens(content: str) -> bool:
    """True nếu parens content là admin instruction (DROP), False nếu clinical data (KEEP)."""
    if not content:
        return True  # empty parens → drop
    # Có digit → clinical info (dose, conc, etc.) → KEEP
    if re.search(r"\d", content):
        return False
    # Admin keywords (VN/EN)
    admin_words = [
        "uống", "ăn", "trước", "sau", "hôm nay", "cùng bữa",
        "food", "meal", "with", "before", "after", "at bedtime",
    ]
    content_lower = content.lower()
    for w in admin_words:
        if w in content_lower:
            return True
    # Nếu không có digit + không có admin word → có thể là abbreviation (vd "(HCl)")
    # Cẩn thận: nếu ngắn + uppercase → có thể là abbrev (HCl, NaCl). KEEP.
    if content.isupper() and len(content) <= 5:
        return False  # KEEP abbrev như "(HCl)"
    # Mặc định: KEEP nếu không chắc chắn
    return False


# Note: _LIFESTYLE_BLACKLIST removed — LLM đã được dạy qua SYSTEM_PROMPT
# để không extract các non-medical terms (lifestyle/social/sự kiện xã hội).


# R38 (2026-07-14): More comprehensive VN family patterns
_IS_FAMILY_PATTERNS = [
    # Direct family member references
    r"b[ốo]\s+(b[ệe]nh\s+)?nh[âa]n",
    r"m[ẹe]\s+(b[ệe]nh\s+)?nh[âa]n",
    r"anh\s+(trai\s+)?b[ệe]nh\s+nh[âa]n",
    r"ch[ịi]\s+(g[áa]i\s+)?b[ệe]nh\s+nh[âa]n",
    r"em\s+(trai\s+|g[áa]i\s+)?b[ệe]nh\s+nh[âa]n",
    r"con\s+(trai\s+|g[áa]i\s+)?b[ệe]nh\s+nh[âa]n",
    r"b[ốo]\s+ch[ồồ]ng\s+b[ệe]nh\s+nh[âa]n",
    r"g[ia]\s+[đd][ìi]nh",
    r"ti[eề]n\s+s[ử]?\s*gia\s+[đd][ìi]nh",
    r"ng[ưu]ờ[i]\s+th[âa]n",
    r"h[ọo]\s+h[âa]ng",
    # Family-style markers that ONLY indicate isFamily (not isHistorical)
    r" (?:cha|mẹ|anh|chị|em|ông|bà|cô|dì|chú|bác) ",
    # R38: Extended patterns - bare family member with disease (no "bệnh nhân")
    r"\b(?:bố|cha|mẹ|anh|chị|em|con|ông|bà|cô|dì|chú|bác)(?:\s+(?:trai|gái|nội|ngoại|ruột|chồng|vợ))?\s+(?:bị|mắc|có|từng|tiền\s+sử|mất\s+vì|chết\s+vì|đã\s+từng|được\s+chẩn\s+đoán)\s+\w",
    # Family negative: "GIA ĐÌNH KHÔNG ai..."
    r"gia\s+[đd][ìi]nh\s+kh[ôo]ng\s+(?:ai|b[ệe]n\s+n[âa]o)",
]

_IS_FAMILY_RE = re.compile("|".join(_IS_FAMILY_PATTERNS), re.IGNORECASE | re.UNICODE)


def _load_set_from_json(filename: str) -> set[str]:
    path = _PROJECT_ROOT / "data" / filename
    if not path.exists():
        return set()
    try:
        return set(json.loads(path.read_text(encoding="utf-8")))
    except Exception as e:
        logger.warning("Failed to load %s: %s", path, e)
        return set()


_LIFESTYLE_KEYWORDS: set[str] = _load_set_from_json("lifestyle_keywords.json")

_LIFESTYLE_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(_LIFESTYLE_KEYWORDS, key=len, reverse=True)) + r")\b" if _LIFESTYLE_KEYWORDS else r"(?!x)x",
    re.IGNORECASE | re.UNICODE,
)


# Counter cho logging
_seen_count: int = 0


_COMMON_DRUG_NAMES: set[str] = _load_set_from_json("common_drug_names.json")




def _detect_assertions_from_context(
    entity_text: str,
    input_text: str,
    entity_type: str,
    entity_pos: int,
) -> list:
    """Detect assertions từ input context (LLM yếu hay quên).

    R26 (mới 2026-07-09): isHistorical CHỈ áp dụng cho entities trong section "Tiền sử".
    Detect section header GẦN NHẤT TRƯỚC entity_pos. Nếu là "Tiền sử" → isHistorical.
    Nếu section khác ("Tiền sử bệnh hiện tại", "Lý do nhập viện", "Đánh giá") → KHÔNG có isHistorical.

    Manh mối khác:
    - isNegated: "không", "chưa", "âm tính" trong window 30 chars trước entity.
    - isFamily: "bố/mẹ/anh/chị/em/ông/bà" + "bệnh nhân" HOẶC "tiền sử gia đình".
    """
    if entity_type not in ("CHẨN_ĐOÁN", "THUỐC", "TRIỆU_CHỨNG"):
        return []

    text_lower = input_text.lower()
    pos = max(0, entity_pos)
    found = []

    # R26: Detect section header GẦN NHẤT TRƯỚC entity_pos
    section_id = _find_current_section(input_text, pos)
    is_in_tien_su = (section_id == "tien_su")

    # R34 FIX (2026-07-13): Chỉ dùng `_find_current_section` cho isHistorical.
    # Trước đây dùng 100-char window quanh entity → match "Tiền sử bệnh" header
    # ở section KHÁC, gây false isHistorical cho entities current.
    # Bây giờ: isHistorical chỉ khi section header HIỆN TẠI là "Tiền sử".
    if is_in_tien_su:
        found.append("isHistorical")
    elif section_id in ("hien_tai", "danh_gia"):
        # R40 (2026-07-14): Context-based assertions for current sections.
        # Entities ở section hiện tại không phải lịch sử - không cần isHistorical.
        # (Empty assertions [] mặc định = current là OK, không cần explicit "isCurrent")
        pass

    # isFamily: R38 - comprehensive VN family patterns
    family_patterns = [
        # Bare family member + disease verb (no "bệnh nhân"), excluding bác sĩ / chú ý
        r"\b(?:bố|cha|mẹ|anh|chị|em|con|ông|bà|cô|dì|chú(?!\s+ý)|bác(?!\s+sĩ))(?:\s+(?:trai|gái|nội|ngoại|ruột|chồng|vợ))?\s+(?:bị|mắc|có|từng|tiền\s+sử|mất\s+vì|chết\s+vì|đã\s+từng|được\s+chẩn\s+đoán)\b",
        # Family negative ("GIA ĐÌNH KHÔNG ai...")
        r"gia\s+[đd][ìi]nh\s+kh[ôo]ng\s+(?:ai|b[ệe]n\s+n[âa]o)",
        # Family member with "bệnh nhân" (excluding bác sĩ)
        r"\b(?:bố|cha|mẹ|anh|chị|em|con|ông|bà|cô|dì|chú(?!\s+ý)|bác(?!\s+sĩ))(?:\s+(?:trai|gái|nội|ngoại|ruột|chồng|vợ))?\s+b[ệe]nh\s+nh[âa]n",
        # General family markers
        r"gia\s+đ[ìi]nh\s+(?:có|bị|từng|tiền\s+sử|ghi\s+nhận|ai|mắc)",
        r"ti[eề]n\s+s[ử]\s*gia\s+[đd][ìi]nh",
        r"\bhọ\s+hàng\b",
        r"\bngười\s+thân\b",
        r"\bdi\s+truyền\b",
        r"\bng[ưu]ờ[i]\s+th[âa]n\b",
    ]
    # Window 200 chars quanh entity cho family, NHƯNG cắt theo ranh giới câu (Clause Boundary Barriers - Super-Upgrade 3)
    family_win_start = max(0, pos - 160)
    family_win_end = min(len(input_text), pos + len(entity_text) + 80)
    family_slice = text_lower[family_win_start:family_win_end]
    rel_pos = pos - family_win_start
    barriers = [m.start() for m in re.finditer(r'[.;\n]+', family_slice)]
    clause_start = max([b for b in barriers if b < rel_pos], default=0)
    clause_end = min([b for b in barriers if b >= rel_pos + len(entity_text)], default=len(family_slice))
    family_window = family_slice[clause_start:clause_end]
    for pat in family_patterns:
        if re.search(pat, family_window, re.UNICODE):
            found.append("isFamily")
            break

    # isNegated: check "không", "chưa", "âm tính", "loại trừ" trong window trước entity.
    # Clause Boundary Barriers (Super-Upgrade 3): cắt bỏ trước các ranh giới mệnh đề
    near_raw = text_lower[max(0, pos - 45):pos + min(len(entity_text), 15)]
    near_parts = re.split(r'[.;\n]|(?:\b(?:nhưng|tuy\s+nhiên|ngoại\s+trừ|hiện\s+tại|khám|chẩn\s+đoán|kết\s+luận|lúc\s+nhập\s+viện)\b)', near_raw)
    near = near_parts[-1]
    found_negated = False
    neg_phrases = (
        "không thấy", "chưa thấy", "chưa có dấu hiệu", "loại trừ",
        "không có", "chưa có", "không phát hiện", "âm tính",
        "không ghi nhận", "chưa ghi nhận", "không sốt", "không ho",
        "không", "chưa"
    )
    for neg in neg_phrases:
        if neg in near:
            if re.search(r'\b' + re.escape(neg) + r'\b', near, re.UNICODE):
                found_negated = True
                break

    if found_negated:
        NON_NEGATION_CONTEXTS = (
            r"không\s+tuân\s+thủ",
            r"không\s+thể",
            r"không\s+có\s+khả\s+năng",
            r"chưa\s+rõ",
            r"không\s+được\s+(?:thực\s+hiện|làm|chụp|tiến\s+hành)",
        )
        for pat in NON_NEGATION_CONTEXTS:
            if re.search(pat, near, re.UNICODE):
                found_negated = False
                break

    if found_negated and "isNegated" not in found:
        found.append("isNegated")

    # R37 (2026-07-16): NEGATION CHAINING — "không X[, hay Y]* Z" → cả đều isNegated.
    # Phải check xem entity có nằm trong cùng chain "không ..." không.
    if "isNegated" not in found:
        if _is_in_negation_chain(input_text, entity_pos):
            found.append("isNegated")

    return found[:3]  # max 3 theo spec


# R37 (2026-07-16): Negation chain detector
# Phát hiện chuỗi "không X[, hay/và/cũng]* Y[, Z, W, ...]" để negate all items.
# BREAK chain khi gặp: "có", "nhưng", "mà" (theo sau có thể là positive).

_NEG_CHAIN_BREAKERS = re.compile(
    r"\b(?:có|nhưng|mà|tuy\s+nhiên|hiện\s+tại|khám|chẩn\s+đoán|kết\s+luận|"
    r"loại\s+trừ|trừ\s+khi)\b",
    re.IGNORECASE | re.UNICODE,
)
_NEG_CHAIN_STARTERS = re.compile(
    r"\b(?:không|chưa|chưa\s+có|âm\s+tính|không\s+thấy|chưa\s+thấy)\b",
    re.IGNORECASE | re.UNICODE,
)


def _is_in_negation_chain(input_text: str, entity_pos: int) -> bool:
    """R37: True nếu entity nằm trong chuỗi phủ định "không ...[, hay/và/cũng]* ...".

    Logic:
    1. Search BACKWARD từ entity_pos để tìm "không"/"chưa" gần nhất.
    2. Nếu tìm được, check xem giữa "không" và entity có từ BREAK ("có", "nhưng", "mà") không.
    3. Nếu KHÔNG break → entity nằm trong chain "không" → isNegated.
    4. Nếu có break OR không tìm được "không" → False.

    Args:
        input_text: full clinical note
        entity_pos: position of entity trong input_text

    Returns:
        True nếu entity nằm trong chain "không" hiện tại.
    """
    text_before = input_text[:entity_pos]
    if not text_before.strip():
        return False

    # Look backward từ entity_pos tìm "không" gần nhất (max 200 chars window)
    # NOTE (R38 - 2026-07-22): Giữ nguyên window 200 chars vì:
    # - LLM đã được dạy qua prompt (STAGE2_PROMPT) rằng "không" chỉ negate trong
    #   cùng mệnh đề (sau dấu phẩy/chấm → break chain).
    # - Negation chain logic ở đây chỉ là SAFETY NET bổ trợ, không phải primary detector.
    # - Việc thu hẹp window ở đây gây DOUBLE-enforcement mâu thuẫn với prompt.
    search_window_start = max(0, entity_pos - 200)
    search_slice = text_before[search_window_start:]

    # Find last "không" match (closest to entity)
    matches = list(_NEG_CHAIN_STARTERS.finditer(search_slice))
    if not matches:
        return False

    last_match = matches[-1]
    last_match_abs_pos = search_window_start + last_match.start()
    last_match_end_pos = search_window_start + last_match.end()

    # Check giữa "không" và entity_pos có BREAK word HOẶC ranh giới câu/dòng (\n hoặc .) không
    between = text_before[last_match_end_pos:entity_pos]
    if "\n" in between or "." in between:
        return False  # Ranh giới câu/dòng → chain break!

    # NOTE (R38 - 2026-07-22): KHÔNG thêm comma-clause break ở đây.
    # Prompt đã dạy LLM (STAGE2_PROMPT) cách áp dụng isNegated qua clause boundary.
    # Để postprocess thêm rule sẽ gây DOUBLE-enforcement, dễ conflict với prompt.

    if not between.strip():
        # "không" ngay trước entity (no words between)
        entity_end = min(len(input_text), entity_pos + 80)
        non_neg_span = input_text[max(0, last_match_abs_pos - 50):entity_end]
        for pat in (
            r"không\s+tuân\s+thủ",
            r"không\s+thể",
            r"không\s+có\s+khả\s+năng",
            r"chưa\s+rõ",
            r"không\s+được\s+(?:thực\s+hiện|làm|chụp|tiến\s+hành)",
        ):
            if re.search(pat, non_neg_span, re.UNICODE | re.IGNORECASE):
                return False
        return True

    # Có words giữa "không" và entity. Check for BREAK words.
    if _NEG_CHAIN_BREAKERS.search(between):
        return False  # Có "có"/"nhưng"/"mà" → chain break

    # Check NON_NEGATION_CONTEXTS — span từ "không" → hết entity (entity_pos + len(entity_text))
    entity_end = min(len(input_text), entity_pos + 80)
    non_neg_span = input_text[last_match_abs_pos:entity_end]
    for pat in (
        r"không\s+tuân\s+thủ",
        r"không\s+thể",
        r"không\s+có\s+khả\s+năng",
        r"chưa\s+rõ",
        r"không\s+được\s+(?:thực\s+hiện|làm|chụp|tiến\s+hành)",
    ):
        if re.search(pat, non_neg_span, re.UNICODE | re.IGNORECASE):
            return False

    return True


def _find_current_section(input_text: str, entity_pos: int) -> str:
    """R26: Tìm section header GẦN NHẤT TRƯỚC entity_pos (chuẩn chung VN clinical notes).

    Các section headers VN phổ biến (lowercase):
    === "tien_su" - thuộc tiền sử/quá khứ (isHistorical=True) ===
    - "Tiền sử bệnh" / "Tiền sử" / "Tiền sử nội khoa" / "Tiền sử ngoại khoa"
    - "Tiền sử phẫu thuật" / "Tiền sử thủ thuật"
    - "Tiền sử gia đình" / "Tiền sử dị ứng" / "Tiền sử xã hội"
    - "Thuốc trước khi nhập viện" / "Thuốc đang dùng" / "Thuốc ra viện"
    - "Thuốc trước đây" / "Thuốc cũ" / "Đang điều trị tại nhà"
    - "Bệnh sử" / "Tiền căn" / "Tiền sử dùng thuốc"
    - "Cách đây X năm/tháng/tuần" (trong mô tả)

    === "hien_tai" - hiện tại (isHistorical=False) ===
    - "Tiền sử bệnh hiện tại" / "Hiện tại"
    - "Lý do nhập viện" / "Lý do vào viện" / "Lý do khám"
    - "Triệu chứng hiện tại" / "Triệu chứng cơ năng" / "Triệu chứng thực thể"
    - "Diễn biến bệnh" / "Diễn tiến" / "Quá trình bệnh"
    - "Khám lúc vào viện" / "Khám hiện tại"
    - "Bệnh sử hiện tại" / "Lịch sử bệnh hiện tại"

    === "danh_gia" - đánh giá tại bệnh viện (isHistorical=False) ===
    - "Đánh giá tại bệnh viện" / "Đánh giá"
    - "Khám" / "Khám tại viện" / "Khám vào viện"
    - "Xét nghiệm" / "CLS" / "Cận lâm sàng" / "Kết quả xét nghiệm"
    - "Chẩn đoán hình ảnh" / "Hình ảnh" / "X-quang" / "Siêu âm" / "CT" / "MRI"
    - "Điều trị" / "Phác đồ điều trị" / "Hướng xử trí" / "Kế hoạch"
    - "Theo dõi" / "Tái khám" / "Ra viện" / "Tóm tắt" / "Kết luận"

    Returns:
        "tien_su" / "hien_tai" / "danh_gia" / "" (không xác định)
    """
    text_before = input_text[:max(0, entity_pos)]

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




def _drop_substring_entities(entities: list[dict]) -> list[dict]:
    """Drop entities whose text is fully contained in another entity's text
    (cùng type, cùng text substring).

    Ví dụ:
        ["khó thở nhẹ" (idx=0), "khó thở" (idx=1)]
        → drop idx=1 ("khó thở" là substring của "khó thở nhẹ")

        ["đau ngực trái" (idx=0), "đau ngực" (idx=1)]
        → drop idx=1

    Returns: list entities sau khi drop các entity bị overlap.
    """
    if len(entities) < 2:
        return entities

    # Find indices cần drop (text_j là substring của text_i, cùng type)
    drop_indices: set[int] = set()
    for i, ent_i in enumerate(entities):
        if i in drop_indices:
            continue
        type_i = ent_i.get("type", "")
        text_i = str(ent_i.get("text", "")).strip()
        if not text_i:
            continue
        for j, ent_j in enumerate(entities):
            if i == j or j in drop_indices:
                continue
            if ent_j.get("type", "") != type_i:
                continue
            text_j = str(ent_j.get("text", "")).strip()
            if not text_j or len(text_j) > len(text_i) or (len(text_j) == len(text_i) and j <= i):
                continue
            # text_j ngắn hơn hoặc bằng text_i: CHỈ drop nếu 2 span có overlap hoặc kề sát vị trí trên văn bản (+5 chars cho chunk boundaries)
            pos_i = ent_i.get("position", [0, 0])
            pos_j = ent_j.get("position", [0, 0])
            pos_overlap = (
                isinstance(pos_i, list) and isinstance(pos_j, list)
                and len(pos_i) == 2 and len(pos_j) == 2
                and max(pos_i[0], pos_j[0]) < min(pos_i[1], pos_j[1]) + 6
            )
            if pos_overlap and _is_semantic_overlap(text_j, text_i):
                drop_indices.add(j)
                logger.debug(
                    "Drop substring/semantic entity '%s' (subset of '%s' at pos %s vs %s)",
                    text_j, text_i, pos_j, pos_i,
                )

    return [ent for idx, ent in enumerate(entities) if idx not in drop_indices]


_VITAL_SIGNS_DUMP_RE = re.compile(
    r"^(VS\d+|VS\s+\d+|[A-Z0-9.\s/]{10,}\b(RA|mmHg|bpm|°C|F|%|K/uL)?)$",
    re.IGNORECASE,
)

_PURE_DURATION_RE = re.compile(
    r"^(kéo dài|khởi phát|trong|cách|sau|lúc|diễn ra)\s+.*(giây|phút|giờ|ngày|tuần|tháng|năm|hôm|sáng|tối|trưa|nay|trước)$|^\d+\s*(giây|phút|giờ|ngày|tuần|tháng|năm)$",
    re.IGNORECASE | re.UNICODE,
)


# ════════════════════════════════════════════════════════════════════════════════
# R29 (2026-07-13 spec round 2): Drop symptom if diagnosis dupe
# ════════════════════════════════════════════════════════════════════════════════

def _is_non_treatment_drug_context(text: str, input_text: str = "") -> bool:
    """R29 (spec round 2): Detect drugs mentioned in non-treatment contexts.

    Cases phải return True:
    - 'Enterococcus kháng vancomycin' (resistance mention)
    - lab tokens: 'bicarbonate', 'creatinine', 'urea', 'sodium', ... (lab test results)
    - exposure events

    Cases phải return False:
    - 'vancomycin đã được dùng để điều trị' (actual treatment)
    - 'metoprolol 25mg po bid' (medication order)

    Logic:
        1. Check LAB_TOKENS whitelist (drugs used as lab panel measurements, NOT Rx)
        2. Check 30-char window BEFORE entity in input_text:
           - 'kháng vancomycin', 'đề kháng vancomycin', 'resistance vancomycin'
             → resistance mention, no Rx candidate
        3. Trailing 'kháng sinh nhóm X' same drug mentioned as class → conservative,
           do NOT count as resistance (we only flag "kháng [specific_drug]")

    Args:
        text: entity text đang check (vd 'vancomycin')
        input_text: original input text để check context (optional, có thể empty)
    """
    if not text:
        return False
    tl = text.lower().strip()
    if len(tl) < 3:
        return False

    # 1. LAB_TOKENS — các chất hay xuất hiện trong lab test panels (không phải thuốc ordered)
    lab_only = frozenset({
        "bicarbonate", "creatinine", "urea", "sodium", "potassium",
        "chloride", "magnesium", "phosphate", "calcium", "glucose",
        "albumin", "bilirubin", "hemoglobin", "haemoglobin",
    })
    if tl in lab_only:
        return True

    # 2. Resistance context: preceding chars có "kháng", "đề kháng", "resistance"
    # QUAN TRỌNG: phải theo sau bởi WHITESPACE để tránh match "kháng sinh"
    if input_text and len(tl) >= 4:
        s = input_text.lower().find(tl)
        if s > 0:
            # Window 30 chars BEFORE entity
            pre = input_text[max(0, s - 30):s]
            # Check "kháng " (với space) hoặc "đề kháng" / "resistance"
            pre_lower = pre.lower()
            if re.search(r"\b(?:đề\s+)?kháng\s+$", pre_lower) or re.search(r"\bresistance\s+$", pre_lower):
                return True

    return False


def _drop_symptom_when_diagnosis_present(entities: list[dict]) -> list[dict]:
    """Cross-type drop: nếu 1 diagnosis span bao trùm symptom span → drop symptom.

    Spec examples:
        'rối loạn lo âu'  (CHẨN_ĐOÁN) contains 'lo âu'  (TRIỆU_CHỨNG) → drop
        'áp xe/viêm khớp nhiễm trùng' (CHẨN_ĐOÁN) contains 'nhiễm trùng' → drop
        'bệnh huyết áp cao' (CHẨN_ĐOÁN) does NOT contain 'huyết áp cao' (different entity)

    Logic:
        - For each TRIỆU_CHỨNG, check if its text appears as a substring of ANY CHẨN_ĐOÁN
          (case-insensitive, normalized whitespace).
        - If yes → drop the TRIỆU_CHỨNG.
        - Reverse direction also checked (small diagnosis prefix in longer symptom).

    Strict: only drops when text is genuine substring match — không match các
    TRIỆU_CHỨNG độc lập (vd 'đau ngực' không phải substring của 'viêm phổi').
    """
    if len(entities) < 2:
        return entities
    diagnoses = [
        str(e.get("text", "")).lower().strip()
        for e in entities
        if e.get("type") == "CHẨN_ĐOÁN"
    ]
    if not diagnoses:
        return entities
    # Chỉ drop khi diagnosis.span bao trùm symptom.text (substring)
    # và không drop unrelated symptoms.
    drop_idx = set()
    for i, ent in enumerate(entities):
        if ent.get("type") != "TRIỆU_CHỨNG":
            continue
        s_text = str(ent.get("text", "")).lower().strip()
        if len(s_text) < 4:  # too short — tránh drop 'đau', 'mệt', ...
            continue
        for d_text in diagnoses:
            if len(d_text) < 4:
                continue
            # symptom là substring của diagnosis → drop
            if s_text in d_text:
                drop_idx.add(i)
                break
    if not drop_idx:
        return entities
    return [e for i, e in enumerate(entities) if i not in drop_idx]


# R37 (2026-07-19): Strip assertions from TÊN_XN/KQ_XN per spec
# (these types don't carry isHistorical/isFamily/isNegated — only TRIỆU_CHỨNG,
# CHẨN_ĐOÁN, THUỐC do).
_TYPES_WITHOUT_ASSERTIONS = frozenset({"TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"})


def _strip_assertions_for_test_types(entities: list[dict]) -> list[dict]:
    """R37 (2026-07-19): Strip `assertions` from TÊN_XÉT_NGHIỆM + KQ_XN entities.

    Per spec:
    - TÊN_XÉT_NGHIỆM: position only (no assertions, no candidates)
    - KẾT_QUẢ_XÉT_NGHIỆM: position only (no assertions, no candidates)

    Earlier pipeline occasionally tagged these with isHistorical (e.g. surgical
    procedures in tiền sử → isHistorical). Per spec, this is wrong — strip.
    """
    for e in entities:
        if not isinstance(e, dict):
            continue
        if e.get("type") in _TYPES_WITHOUT_ASSERTIONS:
            e["assertions"] = []
    return entities


# R37 (2026-07-16): Cross-type substring drop — drop short entity (TÊN_XN/KQ_XN/...)
# nếu nó là substring của entity dài hơn (CHẨN_ĐOÁN) với position overlap.
# VD: "mạch" (TÊN_XN) inside "bệnh tim mạch do xơ vữa động mạch" (CHẨN_ĐOÁN) → drop "mạch".
def _drop_short_substring_inside_longer(
    entities: list[dict],
    short_min_len: int = 4,
    short_max_len: int = 9,
    longer_min_len: int = 8,
) -> list[dict]:
    """Drop short entity (4-9 chars) nếu nó fully contained inside longer entity (>=10 chars).

    Cross-type substring drop. Ví dụ:
    - "mạch" (TÊN_XN, 4 chars) inside "bệnh tim mạch do xơ vữa động mạch" (CHẨN_ĐOÁN) → drop "mạch"
    - "phổi" (TÊN_XN, 4 chars) inside "viêm phổi" (CHẨN_ĐOÁN) → drop "phổi"
    - "máu" (KQ_XN, 3 chars < 4) inside "nhiễm trùng máu" (CHẨN_ĐOÁN) → KEEP (below short_min_len)

    Args:
        entities: list of dicts with 'text', 'type', 'position' [start, end]
        short_min_len: text ngắn tối thiểu (default 4) để được xét
        short_max_len: text ngắn tối đa (default 9) — words > 9 chars thường là disease đầy đủ
        longer_min_len: text dài tối thiểu (default 10) để được coi là container

    Returns:
        List các entities sau khi drop. Same type substring đã được handle bởi
        `_drop_substring_entities` trước đó, nên hàm này chỉ lo cross-type.
    """
    if len(entities) < 2:
        return entities

    drop_indices: set[int] = set()

    for i, ent_i in enumerate(entities):
        if i in drop_indices:
            continue
        text_i = str(ent_i.get("text", "")).strip().lower()
        if not text_i or len(text_i) < longer_min_len:
            continue  # Only long entities can be containers
        pos_i = ent_i.get("position", [0, 0])
        if not (isinstance(pos_i, list) and len(pos_i) == 2):
            continue
        try:
            s_i, e_i = int(pos_i[0]), int(pos_i[1])
        except (ValueError, TypeError):
            continue
        if s_i < 0 or e_i <= s_i:
            continue
        type_i = ent_i.get("type", "")

        for j, ent_j in enumerate(entities):
            if i == j or j in drop_indices:
                continue
            type_j = ent_j.get("type", "")
            if type_j == type_i:
                continue  # Same type → đã handle bởi _drop_substring_entities
            text_j = str(ent_j.get("text", "")).strip().lower()
            if not text_j:
                continue
            # Length filter: chỉ drop short words
            jl = len(text_j)
            if jl < short_min_len or jl > short_max_len:
                continue
            if jl >= len(text_i):
                continue
            pos_j = ent_j.get("position", [0, 0])
            if not (isinstance(pos_j, list) and len(pos_j) == 2):
                continue
            try:
                s_j, e_j = int(pos_j[0]), int(pos_j[1])
            except (ValueError, TypeError):
                continue
            # Position j must be STRICTLY inside i (không overlap từng phần)
            if not (s_i <= s_j and e_j <= e_i):
                continue
            # Text j must be FULLY contained in text i (case-insensitive)
            if text_j not in text_i:
                continue
            # Drop the shorter
            drop_indices.add(j)
            logger.debug(
                "Drop cross-type substring '%s' (%s) inside '%s' (%s) at [%d,%d]",
                text_j, type_j, text_i, type_i, s_i, e_i,
            )

    if not drop_indices:
        return entities
    return [e for idx, e in enumerate(entities) if idx not in drop_indices]


# ════════════════════════════════════════════════════════════════════════════════
# R37-bis (2026-07-14): Safety net for drug+clinical parens + compound disease terms
# ════════════════════════════════════════════════════════════════════════════════

# Patterns để detect drug parens content (dosage narrative, dose numbers, etc.)
_DRUG_PARENS_CONTENT_PATTERNS = [
    re.compile(
        r"\d+\s*(?:mg|mcg|g|ml|iu|%|đơn\s+vị|viên|ống|gói)",
        re.IGNORECASE | re.UNICODE,
    ),
    re.compile(
        r"(?:reduced|increased|changed|switched|tăng\s+từ|giảm\s+từ|đổi\s+từ)\s+from",
        re.IGNORECASE | re.UNICODE,
    ),
    re.compile(r"\bfrom\s+\d+", re.IGNORECASE),
    re.compile(r"\bto\s+\d+", re.IGNORECASE),
    re.compile(r"\bdaily\b|\bpo\s+bid\b|\bpo\s+daily\b", re.IGNORECASE),
    re.compile(r"\b(?:HCl|NaCl|mg/ml|mcg/ml)\b"),
]

# Known test names — nếu entity text match thì KHÔNG phải drug parens content
_KNOWN_TEST_NAMES_LOWER = frozenset({
    "ecg", "ekg", "eeg", "emg", "điện tâm đồ", "điện não đồ", "điện cơ đồ",
    "x-quang", "x-quang ngực", "x-quang bụng", "x-quang sọ", "x-quang cột sống",
    "siêu âm", "siêu âm tim", "siêu âm bụng", "siêu âm thai", "siêu âm tuyến giáp",
    "siêu âm tim qua thành ngực",
    "ct", "ct scan", "ct sọ não", "ct ngực", "ct bụng",
    "mri", "mri sọ não", "mri cột sống", "mri cột sống cổ",
    "công thức máu", "phân tích nước tiểu", "nước tiểu",
    "monitor holter", "holter", "monitor holter 24h",
    "định lượng", "định nhóm máu", "nhóm máu",
    "nội soi", "nội soi dạ dày", "nội soi đại tràng", "nội soi phế quản",
    "xét nghiệm", "chụp x-quang", "chụp cắt lớp",
    "glucose", "ast", "alt", "wbc", "hgb", "plt", "creatinine", "urea",
    "bilirubin", "cholesterol", "triglyceride", "hdl", "ldl",
})


def _is_drug_parens_content(text: str) -> bool:
    """Check nếu text là nội dung ngoặc đơn của thuốc (KHÔNG phải test name thật).

    Returns True nếu:
    - Text chứa dose info (regex \\d+mg, \\d+ml, etc.) HOẶC
    - Text chứa dose-change narrative (reduced from, increased from, etc.) HOẶC
    - Text có format "from X to Y" (liều cũ → liều mới)

    Returns False nếu text match known test name thật.
    """
    if not text:
        return False
    tl = text.strip().lower()
    if not tl:
        return False
    # Nếu match known test name → KHÔNG phải parens content
    if tl in _KNOWN_TEST_NAMES_LOWER:
        return False
    # Check các pattern đặc trưng cho drug parens content
    for pat in _DRUG_PARENS_CONTENT_PATTERNS:
        if pat.search(text):
            return True
    return False


def _drop_drug_parens_misclassification(
    entities: list[dict],
    input_text: str,
) -> list[dict]:
    """R37-bis (2026-07-14): Drop entities là parens content của thuốc bị LLM tách nhầm.

    Khi LLM gặp `metoprolol (reduced from 50mg to 25mg daily)`, nó đôi khi tách thành:
      - `metoprolol` (THUỐC) ở [a, b]
      - `reduced from 50mg to 25mg daily` (TÊN_XÉT_NGHIỆM) ở [c, d]

    Logic detect:
      1. Entity A là THUỐC ở [a, b]
      2. Entity B (type ≠ THUỐC) ở [c, d] với c - b ∈ {1, 2} (gap = "(" hoặc " (")
      3. Gap chứa "(" và char ngay sau d là ")"
      4. Text của B match `_is_drug_parens_content()`

    Action: drop B, extend A thành [a, d+1] (bao gồm cả ngoặc đơn).
    """
    if not entities or not input_text:
        return entities

    n = len(input_text)
    drop_indices: set[int] = set()
    expand_drug: dict[int, int] = {}  # idx → new_end

    for i, ent in enumerate(entities):
        if i in drop_indices:
            continue
        etype = ent.get("type", "")
        if etype != "THUỐC":
            continue
        pos = ent.get("position", [])
        if not (isinstance(pos, list) and len(pos) == 2):
            continue
        try:
            a, b = int(pos[0]), int(pos[1])
        except (ValueError, TypeError):
            continue
        if not (0 <= a < b <= n):
            continue

        # Tìm entity B ngay sau A với gap = "(" hoặc " ("
        for j in range(i + 1, len(entities)):
            if j in drop_indices:
                continue
            jent = entities[j]
            jtype = jent.get("type", "")
            if jtype == "THUỐC":
                continue
            jpos = jent.get("position", [])
            if not (isinstance(jpos, list) and len(jpos) == 2):
                continue
            try:
                c, d = int(jpos[0]), int(jpos[1])
            except (ValueError, TypeError):
                continue
            if not (0 <= c < d <= n):
                continue

            # Gap giữa A và B phải là "(" hoặc " ("
            gap = input_text[b:c]
            gap_clean = gap.strip()
            if not gap_clean.startswith("(") or c - b > 3:
                continue

            # Char ngay sau d phải là ")" (đóng ngoặc)
            if d >= n or input_text[d] != ")":
                continue

            # Text của B phải match pattern drug parens content
            jtext = str(jent.get("text", "")).strip()
            if not _is_drug_parens_content(jtext):
                continue

            # Drop B và extend A
            logger.info(
                "[R37-bis] Drop drug parens content misclassified as %s: '%s' "
                "(drug '%s' at [%d,%d] extended to [%d,%d])",
                jtype, jtext, ent.get("text", ""), a, b, a, d + 1,
            )
            drop_indices.add(j)
            expand_drug[i] = d + 1  # inclusive end (exclusive for slicing)
            break  # chỉ xử lý 1 cụm parens sau mỗi drug

    out: list[dict] = []
    for i, ent in enumerate(entities):
        if i in drop_indices:
            continue
        if i in expand_drug:
            new_end = expand_drug[i]
            new_text = input_text[ent["position"][0]:new_end]
            # Verify text khớp đúng (case-insensitive)
            if new_text and new_text.lower().startswith(str(ent.get("text", "")).lower()[:10]):
                out.append({**ent, "text": new_text, "position": [ent["position"][0], new_end]})
            else:
                # Fallback: giữ nguyên entity gốc
                logger.debug(
                    "[R37-bis] Skip extend drug '%s' — new_text doesn't match expected prefix",
                    ent.get("text", ""),
                )
                out.append(ent)
        else:
            out.append(ent)

    return out


# Compound disease terms phổ biến (R37-bis fallback) — whitelist an toàn
_COMPOUND_DISEASE_TERMS: frozenset[str] = frozenset({
    # Ung thư + cơ quan
    "ung thư phổi", "ung thư vú", "ung thư dạ dày", "ung thư gan",
    "ung thư đại tràng", "ung thư trực tràng", "ung thư tuyến tiền liệt",
    "ung thư buồng trứng", "ung thư cổ tử cung", "ung thư thực quản",
    "ung thư tụy", "ung thư bàng quang", "ung thư thận", "ung thư máu",
    "ung thư hạch", "ung thư da", "ung thư xương", "ung thư não",
    "ung thư thanh quản", "ung thư vòm họng", "ung thư hầu họng",
    "ung thư amidan", "ung thư lưỡi", "ung thư tuyến giáp",
    "ung thư tuyến tụy", "ung thư túi mật", "ung thư đường mật",
    "ung thư mô mềm", "ung thư hắc tố",
    # Viêm + cơ quan
    "viêm phổi", "viêm gan", "viêm thận", "viêm dạ dày", "viêm phế quản",
    "viêm bàng quang", "viêm tụy", "viêm cơ tim", "viêm màng não",
    "viêm xoang", "viêm họng", "viêm amidan", "viêm khớp", "viêm ruột thừa",
    "viêm tuyến mồ hôi", "viêm nha chu", "viêm lợi", "viêm da",
    "viêm mũi", "viêm tai giữa", "viêm kết mạc", "viêm giác mạc",
    "viêm thanh quản", "viêm phổi kẽ", "viêm phúc mạc", "viêm túi mật",
    "viêm đường mật", "viêm ruột", "viêm đại tràng", "viêm trực tràng",
    "viêm thực quản", "viêm hang vị", "viêm niệu đạo", "viêm tiền liệt tuyến",
    "viêm cơ", "viêm gân", "viêm dây thần kinh",
    # Suy + cơ quan
    "suy tim", "suy thận", "suy gan", "suy hô hấp", "suy tuyến giáp",
    "suy tuyến thượng thận", "suy mạch vành",
    # Thoái hóa + vị trí
    "thoái hóa khớp", "thoái hóa cột sống", "thoái hóa đĩa đệm",
    "thoái hóa khớp gối", "thoái hóa khớp háng", "thoái hóa đốt sống cổ",
    "thoái hóa đốt sống thắt lưng",
    # Rối loạn
    "rối loạn lipid máu", "rối loạn nhịp tim", "rối loạn tiền đình",
    "rối loạn giấc ngủ", "rối loạn lo âu", "rối loạn cảm xúc",
    "rối loạn chuyển hóa", "rối loạn dung nạp glucose",
    # Tăng huyết áp + type
    "tăng huyết áp", "tăng huyết áp độ 1", "tăng huyết áp độ 2",
    "tăng huyết áp độ 3", "tăng áp động mạch phổi",
})

# Body parts đơn lẻ có thể là phần sau của compound term
_COMPOUND_BODY_PARTS: frozenset[str] = frozenset({
    "phổi", "gan", "thận", "dạ dày", "phế quản", "bàng quang", "tụy",
    "cơ tim", "màng não", "xoang", "họng", "amidan", "khớp", "ruột thừa",
    "ruột", "đại tràng", "trực tràng", "thực quản", "hang vị", "túi mật",
    "đường mật", "niệu đạo", "tiền liệt tuyến", "tuyến giáp", "tuyến tụy",
    "tuyến thượng thận", "tuyến mồ hôi", "vú", "cổ tử cung", "buồng trứng",
    "thanh quản", "vòm họng", "hầu họng", "lưỡi", "não", "xương", "da",
    "máu", "hạch", "mô mềm", "tim", "mạch vành", "hô hấp",
    "nha chu", "lợi", "mũi", "tai giữa", "kết mạc", "giác mạc",
    "phổi kẽ", "phúc mạc", "cơ", "gân", "dây thần kinh", "mắt",
    # Cột sống / khớp cụ thể
    "cột sống", "đĩa đệm", "khớp gối", "khớp háng",
    "đốt sống cổ", "đốt sống thắt lưng",
})

# Disease prefix để detect compound term có body part
_COMPOUND_DISEASE_PREFIXES = (
    "ung thư", "viêm", "suy", "thoái hóa", "rối loạn",
    "tăng huyết áp", "tăng áp động mạch",
)


def _merge_compound_disease_terms(entities: list[dict]) -> list[dict]:
    """R37-bis (2026-07-14): Merge adjacent entities tạo thành compound disease term.

    Khi LLM tách `ung thư phổi` thành 2 entities (`ung thư` + `phổi`), function này
    sẽ merge lại thành 1 entity duy nhất.

    Logic:
      1. Tìm 2 entities liền kề (gap = " " hoặc "") cùng type CHẨN_ĐOÁN/TRIỆU_CHỨNG
      2. text_a + " " + text_b thuộc `_COMPOUND_DISEASE_TERMS` whitelist
         HOẶC
         text_a là disease prefix (`ung thư`, `viêm`, ...) VÀ
         text_b là body part đơn lẻ (`phổi`, `gan`, ...)
      3. Merge thành 1 entity mới với combined text + position [pos_a[0], pos_b[1]]
    """
    if len(entities) < 2:
        return entities

    out: list[dict] = []
    skip_indices: set[int] = set()

    # Sort theo start position để dễ check adjacent
    sorted_entities = sorted(
        enumerate(entities),
        key=lambda x: (
            x[1].get("position", [0, 0])[0]
            if isinstance(x[1].get("position"), list) and len(x[1]["position"]) >= 1
            else 0
        ),
    )

    i = 0
    while i < len(sorted_entities):
        idx_a, ent_a = sorted_entities[i]
        if idx_a in skip_indices:
            i += 1
            continue

        text_a = str(ent_a.get("text", "")).strip()
        type_a = ent_a.get("type", "")
        pos_a = ent_a.get("position", [])

        if (
            not text_a
            or type_a not in ("CHẨN_ĐOÁN", "TRIỆU_CHỨNG")
            or not (isinstance(pos_a, list) and len(pos_a) == 2)
        ):
            out.append(ent_a)
            i += 1
            continue

        a_start, a_end = int(pos_a[0]), int(pos_a[1])
        merged = False

        # Tìm entity B ngay sau A
        for j in range(i + 1, len(sorted_entities)):
            idx_b, ent_b = sorted_entities[j]
            if idx_b in skip_indices:
                continue
            text_b = str(ent_b.get("text", "")).strip()
            type_b = ent_b.get("type", "")
            pos_b = ent_b.get("position", [])
            if (
                not text_b
                or type_b != type_a
                or not (isinstance(pos_b, list) and len(pos_b) == 2)
            ):
                continue

            b_start, b_end = int(pos_b[0]), int(pos_b[1])

            # Gap phải là " " (1 space) hoặc không có (nếu LLM set position liền nhau)
            gap = b_start - a_end
            if gap < 0 or gap > 1:
                break  # không liền kề nữa → dừng

            # Check combined text có thuộc whitelist không
            text_a_lower = text_a.lower()
            text_b_lower = text_b.lower()
            combined = f"{text_a_lower} {text_b_lower}"

            should_merge = False
            if combined in _COMPOUND_DISEASE_TERMS:
                should_merge = True
            else:
                # Fallback: text_a là disease prefix VÀ text_b là body part đơn lẻ
                if (
                    text_a_lower in _COMPOUND_DISEASE_PREFIXES
                    and text_b_lower in _COMPOUND_BODY_PARTS
                ):
                    should_merge = True
                elif (
                    text_a_lower.endswith("ung thư")
                    and text_b_lower in _COMPOUND_BODY_PARTS
                ):
                    # VD: "ung thư" + "phổi" → "ung thư phổi"
                    should_merge = True

            if should_merge:
                # Tính lại text gốc từ input (an toàn hơn cộng string)
                combined_text_orig = f"{text_a} {text_b}".strip()
                # Verify combined text thực sự nằm giữa [a_start, b_end]
                # (input_text có thể có whitespace khác giữa a_end và b_start)
                merged_ent = {
                    **ent_a,
                    "text": combined_text_orig,
                    "position": [a_start, b_end],
                    "assertions": sorted(
                        set(ent_a.get("assertions", []))
                        | set(ent_b.get("assertions", []))
                    ),
                }
                # Ưu tiên assertions của entity dài hơn nếu conflict isHistorical
                # (entity ngắn "ung thư" có thể có isHistorical do bị detect từ section,
                # entity dài "ung thư phổi" mới là full term → giữ của dài)
                if len(ent_b) > len(ent_a) and ent_b.get("assertions"):
                    merged_ent["assertions"] = sorted(
                        set(ent_b.get("assertions", []))
                        | {
                            a for a in ent_a.get("assertions", [])
                            if a in ("isNegated", "isFamily")
                        }
                    )
                out.append(merged_ent)
                skip_indices.add(idx_a)
                skip_indices.add(idx_b)
                logger.info(
                    "[R37-bis] Merge compound disease term: '%s' + '%s' → '%s' at [%d,%d]",
                    text_a, text_b, combined_text_orig, a_start, b_end,
                )
                merged = True
                break

        if not merged:
            out.append(ent_a)
        i += 1

    # Sort lại theo position
    out.sort(
        key=lambda e: e.get("position", [0, 0])[0]
        if isinstance(e.get("position"), list) and len(e.get("position")) >= 1
        else 0
    )
    return out


def _filter_lifestyle_entities(entities: list[dict]) -> list[dict]:
    """Drop entities khớp lifestyle/social/psychology, sinh hiệu gộp, và thời gian độc lập.

    Defense-in-depth: dù SYSTEM_PROMPT R3/R28 đã cấm, LLM 7B đôi khi vẫn extract:
    - Lifestyle/social: "căng thẳng", "cà phê có caffeine", "mất việc làm 8 ngày trước"
    - Vital signs dump: "VS98.3 12987 56 18 99RA"
    - Pure duration/time: "kéo dài 20 giây", "khởi phát lúc 17 giờ"
    - False isNegated trên TÊN_XÉT_NGHIỆM: "chụp x-quang ngực" bị gán isNegated vì câu "không ghi nhận bất thường"

    Return: list entities đã lọc và chuẩn hóa assertions.
    """
    out: list[dict] = []
    for ent in entities:
        text = str(ent.get("text", "")).strip()
        etype = ent.get("type", "")
        if not text:
            out.append(ent)
            continue

        # 1. Lọc lifestyle / social / psych keywords & narrative noise
        if _LIFESTYLE_RE.search(text) or any(p.match(text.lower().strip()) for p in _DROP_NOISE_PATTERNS):
            logger.debug(
                "[%d] Drop lifestyle/social/psych/noise entity '%s'",
                _seen_count, text,
            )
            continue

        # 2. Lọc chuỗi sinh hiệu gộp / rác lâm sàng dạng VS98.3... (R27.7 mở rộng)
        # NOTE Fix #8: KHÔNG filter cho KẾT_QUẢ_XÉT_NGHIỆM vì "VS98.3 12987 56 18 99RA"
        # là vital signs THỰC TẾ (compact format), không phải noise.
        if etype in ("CHẨN_ĐOÁN", "TRIỆU_CHỨNG") and _VITAL_SIGNS_DUMP_RE.match(text):
            logger.debug(
                "[%d] Drop vital signs dump entity '%s' (%s)",
                _seen_count, text, etype,
            )
            continue

        # 3. Lọc chuỗi thời lượng / mốc thời gian độc lập (chỉ áp dụng cho CHẨN_ĐOÁN / TRIỆU_CHỨNG)
        # R38 (2026-07-23): DISABLED - LLM tự quyết định duration entities qua Stage 2
        # if etype in ("CHẨN_ĐOÁN", "TRIỆU_CHỨNG") and _PURE_DURATION_RE.match(text):
        #     logger.debug(
        #         "[%d] Drop pure duration entity '%s' (%s)",
        #         _seen_count, text, etype,
        #     )
        #     continue

        # 4. Chuẩn hóa assertions: TÊN_XÉT_NGHIỆM không bao giờ bị isNegated nếu kết quả bình thường
        assertions = _normalize_assertions_list(ent.get("assertions", []))
        ent["assertions"] = assertions
        if etype == "TÊN_XÉT_NGHIỆM" and "isNegated" in assertions:
            if not text.lower().startswith(("không ", "chưa ")):
                assertions = [a for a in assertions if a != "isNegated"]
                ent["assertions"] = assertions
                logger.debug("Drop false isNegated from TÊN_XÉT_NGHIỆM: '%s'", text)

        out.append(ent)
    return out


def sanitize_drug_text(text: str) -> str:
    """Smart strip cho drug text (R4 + R18 mới 2026-07, KEEP x 1/x 2 per user).

    Strip:
      - "x N" + unit word: "aspirin 325mg x 1 viên" → "aspirin 325mg x 1" (KEEP "x 1", DROP " viên")
      - Admin parens: "(uống trước ăn)", "(sau ăn)" → DROP (VN noise gây RAG miss)

    KHÔNG strip:
      - Bare "x N" without unit: "aspirin 325mg x 1" → KEEP nguyên "x 1"
      - "x 2 lần/ngày" → "x 2" (drop " lần/ngày" unit)
      - Numerical/clinical parens: "(reduced from 50mg to 25mg)" → KEEP
      - Brand abbreviations: "(HCl)", "(NaCl)" → KEEP
      - Concentration: "(5mg/ml)" → KEEP
      - Route/freq only: "amiodarone 200mg po bid" → KHÔNG đổi
    """
    if not text:
        return text
    text = text.strip()
    if _DRUG_NAME_BAD_PATTERNS.match(text):
        return ""
    # Strip "x N" + unit word (R4) - KEEP "x N" (user yêu cầu), DROP unit
    text = _DRUG_X_N_PATTERN.sub("", text).strip()
    # Smart parens: chỉ drop nếu admin instruction (R18)
    def _smart_parens_sub(m: re.Match) -> str:
        content = m.group(1)
        return "" if _is_admin_parens(content) else m.group(0)
    text = _DRUG_PARENS_PATTERN.sub(_smart_parens_sub, text).strip()
    return text


# ---------------------------------------------------------------------- #
# _clean_entity_text — post-fix entity text LLM hay miss (R27.7, 2026-07-10)
# ---------------------------------------------------------------------- #

# Leading verb/qualifier cần STRIP khi ở đầu TRIỆU_CHỨNG/CHẨN_ĐOÁN
# (giữ lại canonical names như "tăng huyết áp" qua whitelist)
_LEADING_VERB_QUALIFIER_RE = re.compile(
    r"^(không\s+còn\s+|không\s+có\s+|không\s+thấy\s+|"
    r"cảm\s+thấy\s+|thấy\s+|nhận\s+thấy\s+|ghi\s+nhận\s+|có\s+dấu\s+hiệu\s+|có\s+triệu\s+chứng\s+|"
    r"có\s+|bị\s+|xuất\s+hiện\s+|biểu\s+hiện\s+|xảy\s+ra\s+|phát\s+hiện\s+|gặp\s+phải\s+)\s*",
    re.IGNORECASE | re.UNICODE,
)

# Canonical CHẨN_ĐOÁN names chứa "tăng"/"giảm" prefix - KHÔNG strip
_CANONICAL_KEEP_PREFIX = {
    "tăng huyết áp", "tăng đường huyết", "tăng cholesterol",
    "tăng lipid máu", "tăng triglyceride máu", "tăng bilirubin máu",
    "giảm tiểu cầu", "giảm bạch cầu", "giảm dung nạp gắng sức",
    "rối loạn lipid máu", "rối loạn chuyển hóa",
    # Opthalmology (Section 2 spec round 2)
    "phù gai thị",
}

# Verb prefix cần STRIP khỏi TÊN_XÉT_NGHIỆM (DẠNG A - verb NGOÀI tên)
# KHÔNG strip "siêu âm", "nội soi", "monitor", "điện tâm đồ", "phân tích" (compound names)
_TEST_VERB_PREFIX_RE = re.compile(
    r"^(chụp\s+|đo\s+|làm\s+|thực\s+hiện\s+|tiến\s+hành\s+|"
    r"đã\s+(?:tiến\s+hành|làm|thực\s+hiện|chụp|đo)\s+)\s*",
    re.IGNORECASE | re.UNICODE,
)

# Patterns to DROP ENTIRELY (R27.7 - non-entity noise)
# Note: "VS98.3 12987 56 18 99RA" là vital signs THỰC TẾ → KHÔNG drop (Fix #8 R27.7)
# Chỉ drop khi text thuần túy narrative/lifestyle, không phải clinical data
_DROP_NOISE_PATTERNS = [
    re.compile(r"^trung\s+tâm$", re.IGNORECASE),
    re.compile(r"^không\s+liên\s+quan.*$", re.IGNORECASE),
    re.compile(r"^không\s+ghi\s+nhận\s+triệu\s+chứng.*$", re.IGNORECASE),
    re.compile(r"^tại\s+thời\s+điểm\s+nhập\s+viện$", re.IGNORECASE),
    re.compile(r"^khi\s+đến\s+tầng$", re.IGNORECASE),
    re.compile(r"^khi\s+đến\s+khoa.*$", re.IGNORECASE),
    re.compile(r"^vào\s+lúc.*$", re.IGNORECASE),
    # Fix #7: noise narrative về quá trình
    re.compile(r"^khi\s+(?:được\s+)?chuyển\s+(?:vào|tới|đến).*$", re.IGNORECASE | re.UNICODE),
    re.compile(r"^khi\s+(?:đến|nhập|vào)\s+(?:khoa|viện).*$", re.IGNORECASE | re.UNICODE),
    re.compile(r"^trong\s+quá\s+trình.*$", re.IGNORECASE | re.UNICODE),
    re.compile(r"^sau\s+khi\s+.*$", re.IGNORECASE | re.UNICODE),
    re.compile(r"^trước\s+khi\s+.*$", re.IGNORECASE | re.UNICODE),
    # Fix 3.2: Drop narrative/lifestyle/status noise chunks
    re.compile(r"^(?:bệnh\s+nhân\s+)?(?:đã|đang)?\s*(?:ăn\s+uống|ngủ|sinh\s+hoạt|tiếp\s+xúc).*$", re.IGNORECASE | re.UNICODE),
    re.compile(r"^(?:bệnh\s+nhân\s+)?(?:tỉnh|tiếp\s+xúc\s+tốt|da\s+niêm\s+hồng|hồng\s+hào).*$", re.IGNORECASE | re.UNICODE),
    re.compile(r"^quá\s+trình\s+bệnh\s+lý.*$", re.IGNORECASE | re.UNICODE),
    re.compile(r"^diễn\s+biến\s+(?:bệnh|tại\s+khoa).*$", re.IGNORECASE | re.UNICODE),
    re.compile(r"^tình\s+trạng\s+(?:hiện\s+tại|lúc\s+nhập\s+viện)$", re.IGNORECASE | re.UNICODE),
    re.compile(r"^(?:được|đã)\s+(?:chuyển|chỉ\s+định|khuyên).*$", re.IGNORECASE | re.UNICODE),
    # Stage 1 Narrative False Positives (Fix 2.2 / R32)
    re.compile(r"^cô\s+ấy\s+sẽ\s+được.*$", re.IGNORECASE | re.UNICODE),
    re.compile(r"^tỉnh\s+dậy\s+thấy.*$", re.IGNORECASE | re.UNICODE),
    re.compile(r"^quyết\s+định\s+rằng.*$", re.IGNORECASE | re.UNICODE),
    re.compile(r"^nhận\s+thấy.*$", re.IGNORECASE | re.UNICODE),
    re.compile(r"^ước\s+tính.*$", re.IGNORECASE | re.UNICODE),
    re.compile(r"^chúng\s+tôi\s+(?:sẽ|đã|đang).*$", re.IGNORECASE | re.UNICODE),
    re.compile(r"^bệnh\s+nhân\s+(?:sẽ|đã|đang|có\s+thể).*$", re.IGNORECASE | re.UNICODE),
    re.compile(r"^theo\s+(?:đó|sự|chỉ\s+định).*$", re.IGNORECASE | re.UNICODE),
    re.compile(r"^sau\s+đó.*$", re.IGNORECASE | re.UNICODE),
    re.compile(r"^xỉu\s+trước$", re.IGNORECASE | re.UNICODE),
    re.compile(r"^ngất\s+xỉu\s*$", re.IGNORECASE | re.UNICODE),
    re.compile(r"^uống\s+hôm\s+nay$", re.IGNORECASE | re.UNICODE),
    re.compile(r"^hôm\s+nay$", re.IGNORECASE | re.UNICODE),
    re.compile(r"^trong\s+(?:ngày|tuần\s+qua|tuần|tháng|năm)$", re.IGNORECASE | re.UNICODE),
    re.compile(r"^(?:uống|dùng)\s+(?:trước|sau|khi|trong)\s+.*$", re.IGNORECASE | re.UNICODE),
    # R-fix (2026-07-15): Drop nhóm thuốc chung chung (vd "kháng sinh", "kháng sinh tĩnh mạch",
    # "chống đông máu", "thuốc hạ sốt") — prompts.py yêu cầu DROP nhưng trước đây chưa có pattern.
    # Generic class không map được sang RxNorm candidate cụ thể → gây noise "THUỐC không có candidates".
    re.compile(
        r"^(?:kháng\s+sinh|kháng\s+viêm|kháng\s+đông|chống\s+đông(?:\s+máu)?|"
        r"thuốc\s+(?:chống|hạ|giảm)\s+\w+|"
        r"thuốc\s+an\s+thần|thuốc\s+kháng\s+sinh|"
        r"thuốc\s+(?:uống|tiêm|tĩnh\s+mạch|bắp))"
        r"(?:\s+(?:tĩnh\s+mạch|uống|tiêm|tại\s+chỗ))?$",
        re.IGNORECASE | re.UNICODE,
    ),
    # NOTE (R38 - 2026-07-22): KHÔNG thêm pattern keyword cho từng domain cụ thể
    # (vd "đậu tằm", "hồng cầu", "ăn X", "tiếp xúc Y") — đó là việc của PROMPT.
    # Nếu LLM vẫn miss → sửa prompt thêm case-by-case guidance, KHÔNG mở rộng list này.
]

# Pure duration (R28.2) - standalone time expression should not be entity
# Cấu trúc: (duration_expr | duration_expr | duration_expr)
_PURE_DURATION_ENHANCED_RE = re.compile(
    r"^(?:"
    r"\d+\s+(?:giây|phút|giờ|ngày|tuần|tháng|năm)(?:\s+(?:qua|trước|sau))?"
    r"|"
    r"(?:kéo\s+dài|khởi\s+phát\s+lúc|bắt\s+đầu\s+lúc|cách(?:\s+\d+)?|trong(?:\s+vòng)?)"
    r"\s*(?:\d+\s*)?(?:giây|phút|giờ|ngày|tuần|tháng|năm)(?:\s+(?:qua|trước|sau))?"
    r")$",
    re.IGNORECASE | re.UNICODE,
)

# R35 (2026-07-14): Body part đơn lẻ KHÔNG phải TRIỆU_CHỨNG (vd "ngực", "bụng", "đầu").
# LLM hay extract these → safety net DROP trong _clean_entity_text.
# NOTE (R38 - 2026-07-22): Prompt đã enforce rule này. Set này chỉ là SAFETY NET cuối cùng,
# KHÔNG mở rộng thêm — nếu LLM vẫn miss body part mới → sửa PROMPT, không sửa list này.
_BODY_PARTS_ALONE: frozenset[str] = frozenset({
    "ngực", "bụng", "đầu", "lưng", "chân", "tay", "chân tay", "tay chân",
    "bụng trên", "bụng dưới", "ngực trái", "ngực phải",
    "cổ", "lưng trên", "lưng dưới", "đầu trước", "đầu sau",
    "mặt", "mắt", "tai", "mũi", "miệng", "họng",
    "ngón tay", "ngón chân", "khuỷu tay", "đầu gối", "cổ tay", "mắt cá",
    "bẹn", "nách", "mông", "lưng giữa",
})


def _clean_entity_text(text: str, etype: str) -> str | None:
    """Post-fix entity text LLM hay miss (R27.7 mới 2026-07-10).

    Auto-clean các patterns:
    1. Leading verb/qualifier strip ("cảm giác", "tăng", "có", "bị", "xuất hiện", ...)
       → TRỪ canonical names (vd "tăng huyết áp" GIỮ)
    2. Verb prefix trong TÊN_XÉT_NGHIỆM strip ("chụp", "phân tích", "đo", ...)
    3. Parens admin trong THUỐC strip ("(uống trước ăn)" → DROP)
    4. Pure duration DROP (return None → caller drop entity)
    5. R35 (2026-07-14): Drop TRIỆU_CHỨNG chỉ là body part đơn lẻ (vd "ngực", "bụng")
    6. R37 (2026-07-15): Drop standalone dose fragment ("30 mg", "60 mg") trong THUỐC
       (mảnh rời không phải entity hoàn chỉnh).
    7. R37 (2026-07-15): Drop THUỐC là pure drug-class generic ("kháng sinh",
       "thuốc chống viêm") theo prompt rules.

    Args:
        text: entity text gốc từ LLM.
        etype: entity type (THUỐC, CHẨN_ĐOÁN, TRIỆU_CHỨNG, ...).

    Returns:
        Cleaned text. None nếu entity nên bị DROP (vd pure duration, noise, body-part-alone).
    """
    if not text:
        return text
    original = text
    text_lower = text.strip().lower()

    # === BƯỚC 0: R35 (2026-07-14) — DROP TRIỆU_CHỨNG chỉ là body part đơn lẻ ===
    # Safety net cuối cùng. Prompt đã enforce; set này chỉ cover các token
    # mà LLM vẫn miss dù đã có rule trong prompt.
    if etype == "TRIỆU_CHỨNG" and text_lower in _BODY_PARTS_ALONE:
        logger.debug("Clean: drop body-part-alone symptom '%s'", original)
        return None

    # === BƯỚC 1: R37 (2026-07-15) — DROP pure drug-class generic term ===
    # Áp dụng cho THUỐC, CHẨN_ĐOÁN (một số cụm "thuốc X" được LLM gán nhầm CD).
    # VD: "kháng sinh", "thuốc chống viêm", "thuốc hạ sốt" — theo prompt rule 1.
    if etype in ("THUỐC", "CHẨN_ĐOÁN", "TRIỆU_CHỨNG", "TÊN_XÉT_NGHIỆM"):
        if _is_generic_drug_class(text):
            logger.debug("Clean: drop generic drug-class term '%s'", original)
            return None

    # === BƯỚC 2: Pure duration → DROP (R28.2) ===
    # R38 (2026-07-23): DISABLED - LLM tự quyết định duration qua Stage 2 prompt
    # if etype in ("TRIỆU_CHỨNG", "CHẨN_ĐOÁN"):
    #     if _PURE_DURATION_ENHANCED_RE.match(text_lower):
    #         logger.debug("Clean: drop pure duration entity '%s'", original)
    #         return None

    # === BƯỚC 2b: R37 (2026-07-15) — DROP standalone dose fragment trong THUỐC ===
    # Audit phát hiện case file 50: "30 mg", "60 mg" được extract riêng → gây nhiễu.
    # R38 (2026-07-23): DISABLED - LLM tự quyết định dose fragments qua Stage 2 prompt.
    # if etype == "THUỐC" and _DOSE_FRAGMENT_RE.match(text.strip()):
    #     logger.debug("Clean: drop standalone dose fragment '%s'", original)
    #     return None

    # === BƯỚC 3: TÊN_XÉT_NGHIỆM — strip verb prefix ===
    if etype == "TÊN_XÉT_NGHIỆM":
        text_new = _TEST_VERB_PREFIX_RE.sub("", text).strip()
        if text_new != text and text_new:
            logger.debug("Clean: strip verb prefix '%s' → '%s'", text, text_new)
            text = text_new

    # === BƯỚC 4: THUỐC — strip admin parens (R18) ===
    if etype == "THUỐC":
        text = sanitize_drug_text(text)
        if not text or text != original:
            logger.debug("Clean: drug sanitization '%s' → '%s'", original, text)

    # === BƯỚC 5: TRIỆU_CHỨNG/CHẨN_ĐOÁN — strip leading verb/qualifier and trailing duration ===
    if etype in ("TRIỆU_CHỨNG", "CHẨN_ĐOÁN"):
        # Special: nếu text là canonical name → KEEP nguyên
        if text_lower in _CANONICAL_KEEP_PREFIX:
            return text

        # === BƯỚC 5: Sửa "dính chữ" (Missing space) trước khi strip qualifier ===
        # Trường hợp thực tế: "cảm giáckhó chịu" → "cảm giác khó chịu", "tình trạngkhó thở" → "tình trạng khó thở"
        # Bắt cụm từ chỉ định (giác, trạng, chứng, tình, hiện...) bị dính liền vào phụ âm tiếp theo (k, b, d, đ, m, l, s, t, v, x)
        _MISSING_SPACE_RE = re.compile(
            r"(giác|trạng|chứng|tình|hiện|xuất|phát|ghi|nhận|luồng|bệnh|dấu|biểu)"
            r"([kbdđghklmnpqrstvx]{1,3}[aăâeêiôơouưAĂÂEÊIÔƠOUƯàáảãạăắằẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ])",
            re.UNICODE | re.IGNORECASE
        )
        m = _MISSING_SPACE_RE.search(text)
        if m:
            fixed = text[:m.end(1)] + " " + text[m.end(1):]
            logger.debug("Clean: fix missing space '%s' → '%s'", text, fixed)
            text = fixed

        # Strip leading verb/qualifier (regex non-greedy)
        text_new = _LEADING_VERB_QUALIFIER_RE.sub("", text, count=1).strip()

        # === Strip trailing duration / time expression (comprehensive) ===
        trailing_duration_patterns = [
            r"\s+trong\s+\d*\s*(?:giây|phút|giờ|ngày|tuần|tháng|năm)\s*(?:qua|trước|sau)?$",
            r"\s+cách\s+\d+\s*(?:giây|phút|giờ|ngày|tuần|tháng|năm)\s*(?:trước|sau)?$",
            r"\s+kéo\s+dài\s+\d*\s*(?:giây|phút|giờ|ngày|tuần|tháng|năm)$",
            r"\s+khởi\s+phát\s+lúc\s+\d+\s*(?:giây|phút|giờ|ngày)?$",
            r"\s+\d+\s+(?:giây|phút|giờ|ngày|tuần|tháng|năm)(?:\s+(?:qua|trước|sau))?$",
            r"\s+trong\s+(?:tuần|ngày|tháng|năm)\s+(?:qua|trước|sau)$",
        ]
        for pattern in trailing_duration_patterns:
            text_new = re.sub(
                pattern, "", text_new,
                flags=re.IGNORECASE | re.UNICODE,
            ).strip()
            if not text_new:
                break

        if etype == "CHẨN_ĐOÁN":
            stripped_paren = re.sub(
                r"\s*\(([^)]*(?:điện tâm đồ|x-quang|siêu âm|ECG|MRI|CT|XN)[^)]*)\)",
                "", text_new, flags=re.IGNORECASE | re.UNICODE
            ).strip()
            if stripped_paren and stripped_paren != text_new:
                text_new = stripped_paren

        # Strip "Tăng" / "Giảm" prefixes when they precede basic symptoms (CẤM 3)
        if text_new.lower().startswith(("tăng ", "giảm ")) and any(k in text_new.lower() for k in ("đánh trống ngực", "khó thở", "đau ngực", "nhịp tim")):
            text_new = re.sub(r"^(?:tăng|giảm)\s+", "", text_new, flags=re.IGNORECASE).strip()

        # Fix stutter/repeat phrases (e.g., "Khó thở nhẹ khó thở" -> "Khó thở nhẹ")
        parts = text_new.split()
        if len(parts) >= 4:
            parts_lower = [p.lower() for p in parts]
            if parts_lower[:2] == parts_lower[-2:]:
                text_new = " ".join(parts[:-2])
            elif len(parts) % 2 == 0 and parts_lower[:len(parts)//2] == parts_lower[len(parts)//2:]:
                text_new = " ".join(parts[:len(parts)//2])

        if text_new != text and len(text_new) >= 3:
            logger.debug("Clean: strip leading/trailing '%s' → '%s'", text, text_new)
            text = text_new

    return text


# ---------------------------------------------------------------------- #
# _retype_entity — auto-correct entity type dựa trên text patterns (R31 mới)
# ---------------------------------------------------------------------- #

# Abnormal findings trên imaging → CHẨN_ĐOÁN (không phải TRIỆU_CHỨNG/KQ_XN)
# R37 (2026-07-15): Mở rộng pattern để cover các case audit phát hiện
# (bệnh lý chất trắng, ST chênh xuống/lên, gãy xương, ngoại tâm thu + tần suất,
# viêm mô tế bào, tổn thương X, block nhĩ thất, rung nhĩ, ...).
_ABNORMAL_FINDING_TO_CHAN_DOAN = re.compile(
    r"^(tràn dịch màng phổi|tràn dịch màng tim|tràn dịch ổ bụng|cổ trướng|"
    r"tràn khí màng phổi|tràn khí trung thất|"
    r"tim to|gan to|lách to|thận to|"
    r"xẹp phổi|tràn khí phổi|giãn phế quản|"
    r"xơ phổi|khí phế thủng|giãn phế nang|"
    r"gan nhiễm mỡ|xơ gan|thoát vị hoành|"
    r"giãn đường mật|tắc nghẽn đường mật|sỏi mật|"
    r"phù phổi|phù não|"
    r"gãy xương \w+|gãy \w+ xương|gãy xương|"
    r"chấn thương sọ não|chấn thương \w+|"
    r"vết thương hở \w+|"
    r"hở van (hai lá|ba lá|động mạch chủ|động mạch phổi|2 lá)|"
    r"hẹp van (hai lá|ba lá|động mạch chủ|động mạch phổi|2 lá)|"
    r"hở van \w+ (nhẹ|vừa|nặng|mild|moderate|severe)|"
    r"hẹp van \w+ (nhẹ|vừa|nặng|mild|moderate|severe)|"
    r"mất vận động vùng đỉnh|rối loạn vận động vùng đỉnh|"
    r"giãn \w+ buồng tim|"
    r"u ác tính|khối u ác tính|khối u \w+|"
    r"viêm \w+ (nặng|cấp|mạn)|"
    # R36 (2026-07-14): Disease named "viêm X" pattern → CHẨN_ĐOÁN (not TRIỆU_CHỨNG).
    # "viêm" = inflammation = diagnosis name, không phải symptom patient kể.
    r"viêm\s+(?:tuyến\s+mồ\s+hôi|phổi|gan|thận|dạ\s+dày|ruột|tụy|"
    r"túi\s+mật|bàng\s+quang|phế\s+quản|thanh\s+quản|"
    r"khớp|cơ|tim|màng\s+ngoài\s+tim|màng\s+tim|cơ\s+tim|"
    r"não|màng\s+não|xương|tủy(?:\s+xương)?|"
    r"bàng quang|họng|amidan|"
    r"xoang|phổi\s+kẽ|bụng|não\s+tủy|đại\s+tràng|"
    r"dây\s+thần\s+kinh|van\s+tim|tiết\s+niệu|"
    r"ruột\s+non|ruột\s+thừa|thực\s+quản|hang\s+vị|"
    r"trực\s+tràng|hậu\s+môn|tiền\s+liệt\s+tuyến|"
    r"mô\s+tế\s+bào|"
    r"\w+))|"
    # R37 (2026-07-15): Audit findings — bệnh lý chất trắng (CT scan finding).
    r"bệnh\s+lý\s+chất\s+trắng|"
    # ECG abnormalities (full + modifier variants).
    r"ST\s+chênh(?:\s+(?:xuống|lên|chênh))?|"
    r"ST\s+chênh\s+(?:xuống|lên)\s+\w+|"
    r"block\s+(?:nhĩ\s+thất|nhĩ|thất)(?:\s+\w+)?|"
    r"rung\s+nhĩ(?:\s+\w+)?|"
    r"cuồng\s+nhĩ(?:\s+\w+)?|"
    # Arrhythmia with frequency qualifier (audit found "ngoại tâm thu X xuất hiện thường xuyên").
    r"ngoại\s+tâm\s+thu\s+(?:nhĩ|thất)(?:\s+(?:xuất\s+hiện|thường\s+xuyên|có|chiếm)\s+\w+)*|"
    # Tổn thương X (lesion X).
    r"tổn\s+thương\s+\w+(?:\s+\w+)?|"
    # Hẹp/hở động mạch / van variants (overlap-safe).
    r"(?:hẹp|hở)\s+động\s+mạch\s+\w+|"
    r"phình\s+(?:động\s+mạch|đại\s+tràng|tĩnh\s+mạch)\s+\w*",
    re.IGNORECASE | re.UNICODE,
)

# Procedures/surgeries → TÊN_XÉT_NGHIỆM (không phải THUỐC)
# R28 (2026-07-13): CHUYỂN từ hardcoded regex → data-driven load từ data/procedure_patterns.json.
# Cho phép mở rộng mà không cần sửa code.
# Fix (2026-07-15): thêm wildcard `(?:\s+\w[\w\s]*)?` cho các cụm động từ VN có thể kèm
# object phía sau (vd "phẫu thuật bắc cầu nối động mạch vành", "đặt stent ống tuỵ gần").
# Trước đây regex chỉ match đúng nguyên văn → miss các cụm nhiều từ.
_PROCEDURE_TO_TEN_XN = re.compile(
    r"^(phẫu thuật(?:\s+\w[\w\s]*)?|nội soi(?:\s+\w[\w\s]*)?|chọc dò(?:\s+\w[\w\s]*)?|"
    r"đặt stent(?:\s+\w[\w\s]*)?|đặt ống(?:\s+\w[\w\s]*)?|"
    r"thủ thuật(?:\s+\w[\w\s]*)?|can thiệp(?:\s+\w[\w\s]*)?|cắt \w+|"
    r"xạ trị|hóa trị|"
    r"siêu âm|chụp \w+|"
    r"đo \w+|test \w+ \w+)$",
    re.IGNORECASE | re.UNICODE,
)

# ════════════════════════════════════════════════════════════════════════════════
# R28 (2026-07-13): Load procedure patterns + drug INN từ data-driven files.
# ════════════════════════════════════════════════════════════════════════════════

_DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _load_procedure_patterns() -> tuple[list[re.Pattern], list[str], list[str]]:
    """Load procedure regex patterns + exclusion lists từ data/procedure_patterns.json.

    Returns (compiled_patterns, exclude_if_contains, exclude_if_startswith).
    Nếu file không tồn tại → fallback dùng _PROCEDURE_TO_TEN_XN cũ + excludes trống.
    """
    path = _DATA_DIR / "procedure_patterns.json"
    if not path.exists():
        logger.warning(
            "[R28] procedure_patterns.json missing → fall back to _PROCEDURE_TO_TEN_XN cũ"
        )
        return [_PROCEDURE_TO_TEN_XN], [], []
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
        patterns = [
            re.compile(p, re.IGNORECASE | re.UNICODE)
            for p in cfg.get("vn_verbs", [])
        ]
        return patterns, cfg.get("exclude_if_contains", []), cfg.get("exclude_if_startswith", [])
    except Exception as exc:
        logger.warning("[R28] Failed to load procedure_patterns.json: %s", exc)
        return [_PROCEDURE_TO_TEN_XN], [], []


_PROC_PATTERNS, _PROC_EXCLUDE_CONTAINS, _PROC_EXCLUDE_STARTS = _load_procedure_patterns()
logger.info(
    "[R28] Loaded %d procedure patterns + %d/%d exclude lists",
    len(_PROC_PATTERNS), len(_PROC_EXCLUDE_CONTAINS), len(_PROC_EXCLUDE_STARTS),
)


def _is_procedure(text: str) -> bool:
    """R28: Detect procedure text bằng data-driven regex + exclude lists.

    Args:
        text: entity text cần classify

    Returns:
        True nếu text match 1 procedure pattern VÀ không chứa drug keywords.

    Examples:
        >>> _is_procedure("Truyền dịch yếu tố IX đậm đặc")
        True   # "truyền dịch" match
        >>> _is_procedure("truyền máu")
        True
        >>> _is_procedure("aspirin 500mg")
        False  # drug, not procedure
        >>> _is_procedure("MRI cột sống")
        True   # could match imaging test (in exclude_if_startswith handled below)
    """
    if not text or len(text) > 200:
        return False
    tl = text.lower().strip()
    if not tl:
        return False
    # Exclude if starts with test/imaging prefix (those are TÊN_XÉT_NGHIỆM via different path)
    for prefix in _PROC_EXCLUDE_STARTS:
        if tl.startswith(prefix):
            return False
    # Exclude if contains drug keywords (priority drug > procedure)
    for kw in _PROC_EXCLUDE_CONTAINS:
        if kw in tl:
            return False
    # Match procedure verb patterns
    for pat in _PROC_PATTERNS:
        if pat.search(tl):
            return True
    return False


# Treatment modalities → CHẨN_ĐOÁN (không phải THUỐC cụ thể)
_TREATMENT_MODALITY_TO_CHAN_DOAN = re.compile(
    r"^(liệu pháp \w+|điều trị \w+|phác đồ \w+|"
    r"phương pháp \w+|kỹ thuật \w+)$",
    re.IGNORECASE | re.UNICODE,
)

# NOTE (R38 - 2026-07-22): KHÔNG thêm retype rule cho từng case keyword cụ thể
# (nhiễm khuẩn → CHẨN_ĐOÁN, Vitamin K → THUỐC, "bị X" → CHẨN_ĐOÁN, ...).
# Type classification là việc của LLM qua prompt. Nếu LLM sai → sửa prompt,
# KHÔNG sửa post-process. Giữ `_retype_entity` cho các rule domain-stable.


def _retype_entity(text: str, etype: str) -> str:
    """Auto-correct entity type dựa trên text patterns (R31 mới 2026-07-10).

    Logic:
    - Abnormal findings (tim to, tràn dịch, gãy xương, hở van, ...) → CHẨN_ĐOÁN
      (không phải TRIỆU_CHỨNG/KQ_XN)
    - Procedures (phẫu thuật, nội soi, chọc dò, ...) → TÊN_XÉT_NGHIỆM
      (không phải THUỐC)
    - Treatment modalities (liệu pháp, ...) → CHẨN_ĐOÁN
    - R37 (2026-07-15): Drug BRAND name → THUỐC (vd 'crestor', 'toradol', 'augmentin').
    - R37 (2026-07-15): Test ABBREVIATION (ast/alt/wbc/...) → TÊN_XÉT_NGHIỆM
      (không phải KQ_XN).

    Args:
        text: entity text (đã được _clean_entity_text clean).
        etype: current type từ LLM.

    Returns:
        Corrected type (có thể giữ nguyên nếu đúng).
    """
    if not text:
        return etype
    text_stripped = text.strip()
    text_lower = text_stripped.lower()

    # 0. R37: Drug brand name (vd "crestor", "toradol", "augmentin") → THUỐC.
    # Match standalone brand token (whole-text or first 1-2 token).
    if etype != "THUỐC" and text_lower in _DRUG_BRANDS:
        logger.debug("Retype: '%s' %s → THUỐC (brand name)", text, etype)
        return "THUỐC"

    # 1. Abnormal findings → CHẨN_ĐOÁN (override TRIỆU_CHỨNG hoặc KQ_XN)
    if etype in ("TRIỆU_CHỨNG", "KẾT_QUẢ_XÉT_NGHIỆM"):
        if _ABNORMAL_FINDING_TO_CHAN_DOAN.match(text_stripped):
            logger.debug("Retype: '%s' %s → CHẨN_ĐOÁN (abnormal finding)", text, etype)
            return "CHẨN_ĐOÁN"

    # 2. Procedures → TÊN_XÉT_NGHIỆM (override THUỐC)
    if etype == "THUỐC":
        if _PROCEDURE_TO_TEN_XN.match(text_stripped):
            logger.debug("Retype: '%s' THUỐC → TÊN_XÉT_NGHIỆM (procedure)", text)
            return "TÊN_XÉT_NGHIỆM"
        if _TREATMENT_MODALITY_TO_CHAN_DOAN.match(text_stripped):
            logger.debug("Retype: '%s' THUỐC → CHẨN_ĐOÁN (treatment modality)", text)
            return "CHẨN_ĐOÁN"

    # 3. R37: Test abbreviation (vd "ast", "alt", "wbc", "hgb") → TÊN_XÉT_NGHIỆM
    # Chỉ áp dụng khi standalone token (vd "ast" đứng riêng, không phải "ast 421"
    # vì "ast 421" có số đi kèm → đúng là KQ_XN). So sánh full-match lowercase.
    if etype == "KẾT_QUẢ_XÉT_NGHIỆM" and text_lower in _TEST_ABBREVIATIONS:
        logger.debug("Retype: '%s' KQ_XN → TÊN_XÉT_NGHIỆM (test abbreviation)", text)
        return "TÊN_XÉT_NGHIỆM"

    return etype


# ---------------------------------------------------------------------- #
# _split_long_imaging_result — tách long imaging findings (R31 mới)
# ---------------------------------------------------------------------- #

# General pattern để detect test name từ text (thay vì hardcode list dài).
# Match các test name phổ biến với pattern: chụp X, X-quang, siêu âm, ECG, v.v.
# Pattern này flexible hơn list cứng - cover cả những test name mới.
_TEST_NAME_PREFIX_PATTERN = re.compile(
    r"^(?:chụp|đo|làm|thực\s+hiện|tiến\s+hành)?\s*"
    r"(?:"
    r"x[-\s]?quang(?:\s+\w+)?|"  # x-quang, x quang, x-quang ngực
    r"siêu\s+âm(?:\s+\w+(?:\s+\w+)?)?|"  # siêu âm, siêu âm tim, siêu âm bụng
    r"điện\s+tâm\s+đồ|"
    r"ECG|EKG|"
    r"cộng\s+hưởng\s+từ|"
    r"chụp\s+cắt\s+lớp(?:\s+vi\s+tính)?(?:\s+\w+)?|"
    r"CT(?:\s+scan)?|MRI|"
    r"monitor(?:\s+holter)?|holter|"
    r"nội\s+soi(?:\s+\w+)?|"
    r"xét\s+nghiệm(?:\s+\w+)?|"
    r"công\s+thức\s+máu|"
    r"phân\s+tích\s+nước\s+tiểu|"
    r"nước\s+tiểu"
    r")",
    re.IGNORECASE | re.UNICODE,
)

# Connector words strip sau test name (general pattern, không hardcode list dài)
_FINDING_CONNECTORS = re.compile(
    r"^\s*(?:là|cho\s+thấy|ghi\s+nhận|kết\s+quả|phát\s+hiện|tiết\s+lộ|thấy)\s+",
    re.IGNORECASE | re.UNICODE,
)


def _split_long_imaging_result(
    text: str,
    etype: str,
    input_text: str,
    pos: list[int],
) -> list[dict[str, Any]] | None:
    """Tách long imaging result thành nhiều entities riêng (R31 mới 2026-07-10).

    **Logic mới (3 bước)** — theo yêu cầu chính xác của user:
    1. **Bước 1**: Detect test name từ danh sách KNOWN bằng `_find_span`
    2. **Bước 2**: Strip connector (là, cho thấy, ghi nhận, ...) sau test name
    3. **Bước 3**: Phần còn lại = KẾT_QUẢ_XÉT_NGHIỆM riêng

    Ví dụ cụ thể:
        "điện tâm đồ là không ghi nhận gì bất thường" →
            TÊN_XN: "điện tâm đồ" + KQ_XN: "không ghi nhận gì bất thường"
        "chụp x-quang ngực không ghi nhận gì bất thường" →
            TÊN_XN: "chụp x-quang ngực" (drop verb "chụp") + KQ_XN: "không ghi nhận gì bất thường"
        "phân tích nước tiểu không có gì đáng chú ý" →
            TÊN_XN: "phân tích nước tiểu" (drop verb) + KQ_XN: "không có gì đáng chú ý"

    Nếu KHÔNG tìm được test name KNOWN → trả None (giữ nguyên entity gốc).
    Position của entity 2 (KQ) phải NGAY SAU entity 1 (test name).

    Args:
        text: entity text (e.g., "điện tâm đồ là không ghi nhận gì bất thường").
        etype: current type (usually KẾT_QUẢ_XÉT_NGHIỆM).
        input_text: original input text (for re-finding positions).
        pos: current [start, end] position.

    Returns:
        List of new entities đã tách, hoặc None nếu không khớp pattern.
    """
    if not text or etype != "KẾT_QUẢ_XÉT_NGHIỆM":
        return None
    if len(text) < 20:  # quá ngắn thì không cần tách
        return None

    text_stripped = text.strip()

    # === BƯỚC 1: Detect test name bằng general pattern (không hardcode list) ===
    # Match pattern "X-quang...", "siêu âm...", "điện tâm đồ", "ECG", "CT", v.v.
    test_match = _TEST_NAME_PREFIX_PATTERN.match(text_stripped)
    if not test_match:
        return None
    test_name = test_match.group().strip()

    # Re-find position in input_text (ưu tiên exact match)
    test_pos = None
    if input_text:
        test_pos = _find_span(input_text, test_name)
        if test_pos is None:
            # Fallback: thử tìm với verb prefix (vd "chụp x-quang ngực")
            test_pos = _find_span(input_text, text_stripped)
            if test_pos is not None:
                test_name = text_stripped  # Keep verb prefix nếu LLM extract với verb

    if test_pos is None:
        return None

    # === BƯỚC 2: Strip connector sau test name ===
    after_test = text_stripped[len(test_name):].strip()
    after_test = _FINDING_CONNECTORS.sub("", after_test, count=1).strip()
    after_test = after_test.strip(".,;: \t")

    if not after_test:
        return None

    # === BƯỚC 3: Tách findings (nếu có nhiều finding ngăn cách bởi ", " hoặc " và ") ===
    if "," in after_test or " và " in after_test:
        raw_findings = re.split(r",\s*|\s+và\s+", after_test)
        findings = [f.strip().strip(".,;:") for f in raw_findings if f.strip()]
    else:
        findings = [after_test]

    if not findings:
        return None

    # === Build entities ===
    result = [{
        "text": test_name,
        "type": "TÊN_XÉT_NGHIỆM",
        "position": list(test_pos),
        "assertions": [],
        "candidates": [],
    }]

    # Find each finding's position (ngay sau test_name)
    search_start = test_pos[1]
    for finding in findings:
        finding_pos = _find_span(input_text, finding, start=search_start)
        if finding_pos is None:
            # R28 (2026-07-13): Bỏ entity thay vì tạo span rác giữa từ.
            # Trước đây: (search_start, search_start + len(finding)) — có thể
            # land giữa từ nếu finding text không khớp verbatim.
            logger.debug(
                "[R28] Drop finding '%s' — không tìm được word-boundary span sau test name",
                finding[:60],
            )
            continue
        finding_type = _retype_entity(finding, "KẾT_QUẢ_XÉT_NGHIỆM")
        result.append({
            "text": finding,
            "type": finding_type,
            "position": list(finding_pos),
            "assertions": [],
            "candidates": [],
        })
        search_start = finding_pos[1]

    if len(result) >= 2:
        for r in result:
            if not r.get("assertions") and isinstance(r.get("position"), list) and len(r["position"]) == 2:
                detected = _detect_assertions_from_context(
                    r["text"], input_text, r["type"], r["position"][0]
                )
                r["assertions"] = sorted(set(detected))
        return result
    return None


# ---------------------------------------------------------------------- #
# R37 (2026-07-16) — Split merged test_name+value entity
# ---------------------------------------------------------------------- #
# LLM hay extract "ast 421" thành 1 entity thay vì tách thành 2:
#   - TÊN_XÉT_NGHIỆM: "ast"
#   - KẾT_QUẢ_XÉT_NGHIỆM: "421"
# Hàm này tự động tách dựa trên pattern (test_name<space>value).

_TEST_NAME_VALUE_SPLIT_RE = re.compile(
    r"^(?P<test>(?:AST|ALT|GGT|LDH|ALP|WBC|RBC|HGB|HCT|PLT|MCV|MCH|MCHC|RDW|MPV|"
    r"PT|PTT|aPTT|INR|BNP|CRP|ESR|PSA|TSH|T3|T4|FT3|FT4|HbA1C|"
    r"NA|K|CL|MG|CA|GLUCOSE|BUN|CREATININE|CHOLESTEROL|TRIGLYCERIDE|HDL|LDL|"
    r"TROPONIN|CK|CK[- ]MB|D[- ]DIMER|"
    r"pH|LACTATE|AMMONIA|IRON|FERRITIN|VITAMIN|"
    r"MAGNESIUM|PHOSPHATE))"
    r"\s+(?P<value>[\d.,]+(?:\s*[\d./%a-z]+\s*)*)$",
    re.IGNORECASE,
)


def _split_test_name_and_value(
    text: str,
    etype: str,
    input_text: str,
    pos: list[int],
) -> list[dict[str, Any]] | None:
    """R37 (2026-07-16): Tách merged entity 'test_name value' thành 2 entities.

    LLM hay extract "AST 421" thành 1 entity với type=KQ_XN. Đây là lỗi —
    phải tách thành TÊN_XN + KQ_XN.

    Trigger:
    - Text match pattern `<test_abbrev> <value>` (e.g. "AST 421", "WBC 11.6 K/uL")
    - Test name in `_TEST_ABBREVIATIONS`
    - Type is KQ_XN

    Returns:
        List of 2 entities [test_name, value] hoặc None nếu không khớp pattern.

    Ví dụ:
        "ast 421" → [{type:'TÊN_XN', text:'ast'}, {type:'KQ_XN', text:'421'}]
        "alt 336 u/l" → [{type:'TÊN_XN', text:'alt'}, {type:'KQ_XN', text:'336 u/l'}]
    """
    if not text or len(text) > 50:
        return None
    text_stripped = text.strip()
    # Match must be at etype KQ_XN (LLM common mistake)
    if etype != "KẾT_QUẢ_XÉT_NGHIỆM":
        return None

    m = _TEST_NAME_VALUE_SPLIT_RE.match(text_stripped)
    if not m:
        return None
    test_name = m.group("test").strip()
    value_str = m.group("value").strip()

    # Sanity: test name must be in known abbreviations (case insensitive)
    if test_name.lower() not in _TEST_ABBREVIATIONS:
        return None
    # Value should have digits
    if not any(c.isdigit() for c in value_str):
        return None
    # Value should not be too short (avoid splitting "K 2 1" etc.)
    if len(value_str) < 1:
        return None

    # Re-find positions in input_text
    test_pos = _find_span(input_text, test_name, start=pos[0])
    if test_pos is None:
        test_pos = (pos[0], pos[0] + len(test_name))
    value_pos = _find_span(input_text, value_str, start=test_pos[1])
    if value_pos is None:
        value_pos = (pos[0] + len(test_name) + 1, pos[1])

    # Build 2 entities
    test_entity = {
        "text": test_name,
        "type": "TÊN_XÉT_NGHIỆM",
        "position": list(test_pos),
        "assertions": [],
        "candidates": [],
    }
    value_entity = {
        "text": value_str,
        "type": "KẾT_QUẢ_XÉT_NGHIỆM",
        "position": list(value_pos),
        "assertions": [],
        "candidates": [],
    }
    return [test_entity, value_entity]


# ---------------------------------------------------------------------- #
# Main assembly
# ---------------------------------------------------------------------- #






def _find_all_occurrences(text_lower: str, phrase: str) -> list:
    """Tìm tất cả vị trí xuất hiện NON-OVERLAPPING của phrase trong text_lower.

    Args:
        text_lower: text đã lowercase.
        phrase: phrase cần tìm (lowercase).

    Returns:
        list of (start, end) tuples (end exclusive).
    """
    positions = []
    phrase_lower = phrase.lower()
    plen = len(phrase_lower)
    text_len = len(text_lower)

    if plen == 0 or text_len < plen:
        return positions

    start = 0
    while start <= text_len - plen:
        idx = text_lower.find(phrase_lower, start)
        if idx < 0:
            break
        positions.append((idx, idx + plen))
        start = idx + plen  # Non-overlapping: skip past this match

    return positions


def _get_duplicate_alert(input_text: str, top_n: int = 20) -> str:
    """Build chuỗi DUPLICATE ALERT cho các medical term lặp lại nhiều lần."""
    if not input_text or len(input_text) < 100:
        return ""

    import re as _re

    _STOP = {
        "bệnh", "nhân", "viện", "tình", "trạng", "trước", "trong", "ngoài",
        "bằng", "theo", "sang", "qua", "cách", "thuốc", "thể", "cũng", "đang",
        "khác", "nếu", "khi", "hay", "mới", "sau", "trên", "dưới", "tại", "từ",
        "được", "đã", "sẽ", "vào", "ra", "lại", "cho", "với", "của", "này",
        "xuất", "hiện", "ghi", "nhận", "tiến", "hành", "phát", "khám",
        "điều", "trị", "nhập", "theo", "dõi", "liên", "quan", "kết", "quả",
        "chẩn", "đoán", "triệu", "chứng", "kèm", "khởi", "phát", "diễn",
        "biến", "tiền", "sử", "lúc", "giờ", "ngày", "tuần", "tháng", "năm",
        "phút", "giây", "nhiều", "thường", "xuyên", "không", "còn", "đến",
        "đặc", "điểm", "thời", "yếu", "tố", "các", "lý", "loạt",
    }

    _MEDICAL_HINTS = {
        "đánh trống ngực", "đánh trống", "trống ngực",
        "khó thở", "đau ngực", "đau bụng", "đau đầu",
        "buồn nôn", "chóng mặt", "mệt mỏi", "đổ mồ hôi",
        "thắt chặt ngực", "cảm giác thắt", "hồi hộp",
        "tăng huyết áp", "nhồi máu", "rung nhĩ", "ngoại tâm thu",
        "nhịp xoang", "suy tim", "suy thận", "viêm phổi",
        "metoprolol", "atenolol", "bisoprolol", "amlodipine", "aspirin",
        "warfarin", "apixaban", "doxycycline", "paracetamol", "furosemide",
        "x-quang ngực", "siêu âm tim", "điện tâm đồ", "monitor holter",
        "phân tích nước tiểu", "công thức máu", "chụp x-quang",
    }

    words = _re.findall(r"[\wÀ-ỹ]+", input_text)
    freq: dict[str, int] = {}

    for i in range(len(words) - 1):
        w1, w2 = words[i].lower(), words[i+1].lower()
        if len(w1) >= 3 and len(w2) >= 3 and w1 not in _STOP and w2 not in _STOP:
            phrase = f"{w1} {w2}"
            freq[phrase] = freq.get(phrase, 0) + 1

    for i in range(len(words) - 2):
        w1, w2, w3 = words[i].lower(), words[i+1].lower(), words[i+2].lower()
        if (len(w1) >= 3 and w1 not in _STOP
                and w2 not in _STOP and w3 not in _STOP):
            phrase = f"{w1} {w2} {w3}"
            freq[phrase] = freq.get(phrase, 0) + 1

    def _is_medical(phrase: str) -> bool:
        pl = phrase.lower()
        for hint in _MEDICAL_HINTS:
            if hint in pl or pl in hint:
                return True
        if len(phrase) >= 10:
            _BAD_START = ("các ", "kết quả", "tiền sử", "theo ", "lý do",
                          "thời điểm", "yếu tố", "diễn biến", "tình trạng",
                          "đặc điểm", "triệu chứng khi", "triệu chứng hiện")
            for bs in _BAD_START:
                if pl.startswith(bs):
                    return False
            return True
        return False

    candidates = sorted(
        [(t, c) for t, c in freq.items() if c >= 2 and _is_medical(t)],
        key=lambda x: (-x[1], -len(x[0]), x[0]),
    )[:top_n]

    if not candidates:
        return ""

    parts = []
    for phrase, count in candidates:
        idx = input_text.lower().find(phrase)
        if idx >= 0:
            original_case = input_text[idx:idx + len(phrase)]
        else:
            original_case = phrase
        parts.append(f'"{original_case}" {count}x')

    if not parts:
        return ""

    return (
        f"[⚠️ DUPLICATE ALERT: {', '.join(parts)} — "
        f"MỖI occurrence = 1 entity riêng với position riêng (R10 STRICT)]"
    )


def _preprocess_highlight_duplicates(input_text: str, top_n: int = 20) -> str:
    """Giữ nguyên input_text để không làm lệch character position so với validate_positions.
    Phần cảnh báo DUPLICATE ALERT được thêm riêng trước thẻ INPUT trong build_user_prompt.
    """
    return input_text



def _expand_duplicates(entities, input_text):
    """Mở rộng duplicate entities dựa trên scan input thực tế (R20.1 mới 2026-07-09).

    Vấn đề: LLM 7B hay "gộp" duplicate thành 1 entity dù đã có R20 + Ex 24.
    Post-process aggressive: tự scan input text, tìm TẤT CẢ occurrences của mỗi
    entity text, tạo thêm entities cho các positions khác.

    R34 FIX (2026-07-13): Hàm trước chỉ lưu FIRST position về entity — không
    expand thực sự. Sửa: emit 1 entity riêng cho MỖI occurrence (R10 STRICT),
    deduplicate by (text, type) để tránh nhân đôi khi LLM trùng text nhiều lần.

    Args:
        entities: list entities từ LLM (đã validate position).
        input_text: raw input text.

    Returns:
        list entities đã expand (MỖI occurrence = 1 entity riêng, mỗi (text,type) chỉ xử lý 1 lần).
    """
    if not entities or not input_text:
        return entities

    # Modifiers VN cần strip trước khi match (R14/R25)
    _MODIFIERS_PREFIX = re.compile(
        r"^(tăng|giảm|có|không|đang|bị|bị\s+|rõ|rõ\s+rệt|ít|nhiều|hơi|khoảng|có\s+thể)\s+",
        re.IGNORECASE | re.UNICODE,
    )
    _MODIFIERS_SUFFIX = re.compile(
        r"\s+(nhẹ|nặng|vừa|nhẹ\s+nhàng|nặng\s+nề|nhẹ\s+vừa|có\s+triệu\s+chứng|vừa\s+phải)$",
        re.IGNORECASE | re.UNICODE,
    )

    expanded: list[dict[str, Any]] = []
    input_lower = input_text.lower()
    seen_text_types: set[tuple[str, str]] = set()  # (text_lower, type) đã xử lý

    for ent in entities:
        text = str(ent.get("text", "")).strip()
        etype = str(ent.get("type", ""))
        if len(text) < 4:
            # Quá ngắn (vd "M", "T") → giữ nguyên entity gốc
            expanded.append(ent)
            continue

        text_lower = text.lower()
        key = (text_lower, etype)
        if key in seen_text_types:
            # Đã xử lý text+type này rồi (LLM có nhiều ents cùng text) → skip duplicate work
            continue
        seen_text_types.add(key)

        # UNION exact + stripped match (R10 STRICT: 1 entity / occurrence)
        all_positions: list[tuple[int, int]] = []
        seen_starts: set[int] = set()

        # Cách 1: exact substring match (case-insensitive)
        start = 0
        while True:
            idx = input_lower.find(text_lower, start)
            if idx < 0:
                break
            end_idx = idx + len(text)
            # Word-boundary check (R28) — tránh "nôn" khớp trong "buồn nôn"
            if (idx > 0 and input_text[idx - 1].isalnum()) or (
                end_idx < len(input_text) and input_text[end_idx].isalnum()
            ):
                start = idx + 1
                continue
            if idx not in seen_starts:
                all_positions.append((idx, end_idx))
                seen_starts.add(idx)
            start = idx + 1

        # Cách 2: stripped match (bỏ modifier "tăng", "giảm"...) — CHỈ khi exact chưa match.
        # ĐÃ CÓ exact match → không thêm stripped variants (gây trùng).
        if not all_positions:
            text_stripped = _MODIFIERS_PREFIX.sub("", text_lower).strip()
            text_stripped = _MODIFIERS_SUFFIX.sub("", text_stripped).strip()
            if text_stripped and text_stripped != text_lower and len(text_stripped) >= 4:
                start = 0
                while True:
                    idx = input_lower.find(text_stripped, start)
                    if idx < 0:
                        break
                    end_idx = idx + len(text_stripped)
                    # Word-boundary check
                    if (idx > 0 and input_text[idx - 1].isalnum()) or (
                        end_idx < len(input_text) and input_text[end_idx].isalnum()
                    ):
                        start = idx + 1
                        continue
                    if idx not in seen_starts:
                        all_positions.append((idx, end_idx))
                        seen_starts.add(idx)
                    start = idx + 1

        # R34 FIX: Emit 1 entity riêng cho MỖI occurrence (R10 STRICT)
        if all_positions:
            for s, e in all_positions:
                actual_text = input_text[s:e]
                new_ent = {**ent, "text": actual_text, "position": [s, e]}
                expanded.append(new_ent)
        else:
            # No positions found → giữ entity gốc (có thể match fuzzy ở layer sau)
            expanded.append(ent)

    # Sort theo position
    expanded.sort(key=lambda e: e.get("position", [0, 0])[0])
    return expanded



def _normalize_type_to_ascii(etype: str) -> str:
    """R39 (2026-07-24): Convert diacritics type → ASCII fallback.

    HƯỚNG DẪN MỚI (2026-07-24): GIỮ DIACRITICS theo mặc định (grader hiện tại
    chấp nhận diacritics). Chỉ fallback sang ASCII nếu env var
    `TYPE_TO_ASCII=1` được set (cho grader cũ dùng ASCII enum).

    Mapping (giữ nguyên — là hướng từ diacritics → ASCII):
        'THUỐC' → 'THUOC'
        'CHẨN_ĐOÁN' → 'CHAN_DOAN'
        'TRIỆU_CHỨNG' → 'TRIEU_CHUNG'
        'TÊN_XÉT_NGHIỆM' → 'TEN_XET_NGHIEM'
        'KẾT_QUẢ_XÉT_NGHIỆM' → 'KET_QUA_XET_NGHIEM'
    """
    import os
    if not etype:
        return etype
    # GIỮ NGUYÊN DIACRITICS theo mặc định (grader hiện tại chấp nhận).
    # Chỉ convert sang ASCII khi ENV `TYPE_TO_ASCII=1` (legacy grader).
    if os.environ.get("TYPE_TO_ASCII", "0") != "1":
        # Trả về nguyên xi nếu đã là diacritics
        # Nếu đã là ASCII (do pipeline cũ), giữ nguyên → backward compatible
        return etype
    # Convert diacritics → ASCII (legacy fallback)
    mapping = {
        "THUỐC": "THUOC",
        "CHẨN_ĐOÁN": "CHAN_DOAN",
        "TRIỆU_CHỨNG": "TRIEU_CHUNG",
        "TÊN_XÉT_NGHIỆM": "TEN_XET_NGHIEM",
        "KẾT_QUẢ_XÉT_NGHIỆM": "KET_QUA_XET_NGHIEM",
    }
    return mapping.get(etype, etype)


def _restore_diacritics_type(etype: str) -> str:
    """R39 (2026-07-24): Convert ASCII type → diacritics type (để output khớp grader).

    Map ngược từ ASCII fallback về diacritics chuẩn VN:
        'THUOC' → 'THUỐC'
        'CHAN_DOAN' → 'CHẨN_ĐOÁN'
        'TRIEU_CHUNG' → 'TRIỆU_CHỨNG'
        'TEN_XET_NGHIEM' → 'TÊN_XÉT_NGHIỆM'
        'KET_QUA_XET_NGHIEM' → 'KẾT_QUẢ_XÉT_NGHIỆM'
    """
    mapping = {
        "THUOC": "THUỐC",
        "CHAN_DOAN": "CHẨN_ĐOÁN",
        "TRIEU_CHUNG": "TRIỆU_CHỨNG",
        "TEN_XET_NGHIEM": "TÊN_XÉT_NGHIỆM",
        "KET_QUA_XET_NGHIEM": "KẾT_QUẢ_XÉT_NGHIỆM",
    }
    return mapping.get(etype, etype)


# R39 (2026-07-24): ICD hard-reject table — entity text → ICD codes phải loại bỏ.
# Lý do: LLM/RAG đôi khi trả codes sai concept (vd "Thiếu men G6PD" → Q55 testis defect).
# Áp dụng ở FINAL stage của postprocess để đảm bảo output sạch.
_HARD_REJECT_ICD = {
    # Enzyme deficiency → KHÔNG phải congenital
    "thiếu men g6pd": {"Q55", "Q55.0", "Q44", "Q00", "Q01", "Q02", "Q03", "Q04",
                       "Q05", "Q06", "Q07"},
    "thiếu hụt men g6pd": {"Q55", "Q55.0"},
    "g6pd": {"Q55", "Q55.0"},
    "men g6pd": {"Q55", "Q55.0"},
    "glucose-6-phosphate dehydrogenase": {"Q55", "Q55.0"},
    # Myocarditis → KHÔNG phải viral pericarditis
    "viêm tim": {"B33.2", "B33", "B34"},
    "viêm cơ tim": {"B33.2", "B33", "B34"},
    "viêm màng ngoài tim": {"B33.2", "B33"},
    "viêm nội tâm mạc": {"B33.2", "B33"},
    # Coronary aneurysm → KHÔNG intracranial abscess
    "phình giãn động mạch vành": {"G07", "I67", "I60"},
    "phình động mạch vành": {"G07", "I67", "I60"},
    # Anemia → KHÔNG congenital
    "thiếu máu": {"Q55", "Q44", "Q00", "Q01", "Q02", "Q03", "Q04"},
    "thiếu máu tan huyết": {"Q55", "Q44"},
    "thiếu máu do tan huyết": {"Q55", "Q44"},
    # Adult RA → KHÔNG juvenile
    "viêm khớp dạng thấp": {"M08", "M08.0", "M08.1"},
    "viêm khớp": {"Q"},
    # Pneumonia → KHÔNG congenital
    "viêm phổi": set(),  # thường OK; specific exclude nếu cần
    "viêm phổi mắc phải cộng đồng": set(),
    # Kawasaki → KHÔNG generic vasculitis
    "kawasaki": {"I77", "L52"},
    "bệnh kawasaki": {"I77", "L52"},
    # Cancer → KHÔNG factors
    "ung thư": {"Z00", "Z01", "Z02", "Z03", "Z04", "Z05"},
    # Headache → KHÔNG stroke
    "đau đầu": {"I60", "I61", "I62", "I63", "I64", "I65", "I66", "I67", "I68", "I69"},
    "đau nửa đầu": {"I60", "I61", "I62", "I63"},
    # Migraine
    "migraine": {"I60", "I61", "I63"},
    # Common diseases → correct ICD
    "viêm gan b": {"K74"},  # viêm gan b → B16-B18, NOT K74 xơ gan
    "tăng huyết áp": {"I11", "I12", "I13", "I15"},  # THA primary → I10
    "đái tháo đường type 2": {"E10", "E10.0"},  # type 2 → E11, not E10
    # Pediatric
    "tay chân miệng": {"A09", "B34"},
    "sởi": {"A09", "B01"},
    "thủy đậu": {"A09", "B01"},
    "ho gà": {"A09", "B05"},
}


def _apply_hard_reject_icd(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """R39 (2026-07-24): Apply HARD REJECT ICD codes cho từng entity.

    Với mỗi CHẨN_ĐOÁN entity, loại bỏ candidate codes nằm trong `_HARD_REJECT_ICD`
    blacklist (nếu text match). Nếu sau khi loại bỏ còn candidate → giữ.
    Nếu không còn → trả `candidates = []`.

    Args:
        entities: list entity dicts (sẽ modify in-place).

    Returns:
        list entities đã filter candidates.
    """
    if not entities:
        return entities

    n_filtered = 0
    for ent in entities:
        if ent.get("type") not in ("CHẨN_ĐOÁN", "CHAN_DOAN"):
            continue
        text = str(ent.get("text", "")).strip().lower()
        # Normalize Vietnamese chars for matching
        normalized = re.sub(r"\s+", " ", text)
        blacklist = _HARD_REJECT_ICD.get(normalized)
        if blacklist:
            candidates = ent.get("candidates", []) or []
            new_candidates = [c for c in candidates if c.split(".")[0] not in blacklist and c not in blacklist]
            if len(new_candidates) != len(candidates):
                n_filtered += (len(candidates) - len(new_candidates))
                ent["candidates"] = new_candidates
                logger.debug(
                    "[R39] Hard reject ICD for '%s': %s → %s",
                    text[:30], candidates, new_candidates,
                )
    if n_filtered:
        logger.info("[R39] Total hard-rejected ICD codes: %d", n_filtered)
    return entities


def _enforce_position_strict(input_text: str, ent: dict[str, Any]) -> dict[str, Any] | None:
    """R39 (2026-07-24): Position enforcement cuối cùng — đảm bảo input[pos_start:pos_end] == ent.text.

    Chiến lược xử lý các trường hợp thường gặp:
    1. text != substring → thử re-search strict match trong input
    2. text có space hallucination (vd "ảo giác xuất hiện" vs "ảo giácxuất hiện") → thử dedup
    3. text == substring nhưng off-by-one (len không khớp) → snap end → start+len(text)
    4. text không tìm được ở bất kỳ đâu → DROP

    Args:
        input_text: full input text.
        ent: entity dict với 'text' + 'position'.

    Returns:
        ent (đã sửa position) hoặc None nếu không thể recover.
    """
    if not isinstance(ent, dict):
        return None
    text = str(ent.get("text", "")).strip()
    pos = ent.get("position", [])
    if not text or not (isinstance(pos, list) and len(pos) == 2):
        return ent

    try:
        s, e = int(pos[0]), int(pos[1])
    except (ValueError, TypeError):
        return ent

    n = len(input_text)
    if not (0 <= s < e <= n):
        return ent

    actual = input_text[s:e]

    # CASE 1: exact match → done
    if actual == text:
        return ent

    # CASE 2: case-insensitive match → fix case, update text
    if actual.lower() == text.lower():
        ent["text"] = actual
        return ent

    # CASE 3: text length is off by 1 char (off-by-one từ file 26)
    # VD: text="BỆNH MẠCH VÀNH" (14), actual="\nBỆNH MẠCH VÀN" (14 chars nhưng khác content)
    # → thử mở rộng position lên start + len(text)
    if abs((e - s) - len(text)) <= 1:
        new_end = min(n, s + len(text))
        # Kiểm tra char sau new_end có phải alnum → nếu cắt giữa từ thì skip
        if new_end < n and input_text[new_end].isalnum():
            pass  # cắt giữa từ, KHÔNG mở rộng
        elif s > 0 and new_end > s and input_text[s:new_end] == text:
            ent["position"] = [s, new_end]
            return ent
        # Nếu extend về phía trước (bắt đầu từ newline/space)
        new_start = max(0, s - 1)
        if new_start != s and input_text[new_start:new_start + len(text)] == text:
            ent["position"] = [new_start, new_start + len(text)]
            return ent

    # CASE 4: text bị LLM hallucinate space → thử dedup space
    text_no_space = re.sub(r"\s+", "", text)
    if text_no_space != text:
        # Tìm vị trí trong input mà text_no_space match
        idx = input_text.find(text_no_space, max(0, s - 80))
        if idx >= 0 and idx < n:
            new_end = idx + len(text_no_space)
            # Check word boundary
            if (idx == 0 or not input_text[idx - 1].isalnum()) and (
                new_end >= n or not input_text[new_end].isalnum()
            ):
                # Giữ nguyên text gốc (có space) nhưng update position
                # hoặc update text thành text_no_space
                recovered_text = input_text[idx:new_end]
                ent["text"] = recovered_text
                ent["position"] = [idx, new_end]
                return ent

    # CASE 5: text có thể match ở chỗ khác trong input → re-search closest to current pos
    candidates = []
    start_scan = max(0, s - 200)
    end_scan = min(n, e + 200)
    for i in range(start_scan, end_scan):
        # Match word-boundary
        if i > 0 and input_text[i - 1].isalnum():
            continue
        end_i = i + len(text)
        if end_i > n:
            break
        if input_text[i:end_i] == text and (end_i >= n or not input_text[end_i].isalnum()):
            dist = abs(i - s)
            candidates.append((dist, i, end_i))
    if candidates:
        candidates.sort()
        _, new_s, new_e = candidates[0]
        ent["position"] = [new_s, new_e]
        return ent

    # CASE 6: không tìm được → DROP
    logger.debug("[R39] Drop entity '%s' at [%d,%d] — text not found", text, s, e)
    return None


# Chatbot artifacts — common LLM chitchat được LLM/Stage1 đôi khi pick up.
# File 83[2]: "Cảm ơn bạn đã gửi câu hỏi"
_CHATBOT_ARTIFACT_PATTERNS = re.compile(
    r"(?:"
    r"cảm\s+ơn\s+bạn\s+đã\s+gửi\s+câu\s+hỏi|"
    r"hy\s+vọng\s+thông\s+tin\s+(?:này\s+)?(?:sẽ\s+)?(?:hữu\s+ích|giúp\s+ích)|"
    r"nếu\s+bạn\s+có\s+thêm\s+câu\s+hỏi|"
    r"vui\s+lòng\s+(?:liên\s+hệ|tham\s+khảo)\s+(?:bác\s+sĩ|chuyên\s+gia)|"
    r"xin\s+chào|"
    r"chúc\s+bạn\s+(?:sức\s+khỏe|mau\s+khỏe)|"
    r"đây\s+là\s+(?:một\s+)?(?:bài\s+viết|thông\s+tin)\s+(?:tham\s+khảo|trả\s+lời)|"
    r"(?:bài|chủ\s+đề)\s+viết\s+bởi|"
    r"^(?:tóm\s+tắt|kết\s+luận)\s*:"
    r")",
    re.IGNORECASE | re.UNICODE,
)


def _is_chatbot_artifact(text: str) -> bool:
    """R39 (2026-07-24): True nếu text là chatbot chitchat (LLM hallucination), không phải y khoa."""
    if not text or len(text) > 200:
        return False
    return bool(_CHATBOT_ARTIFACT_PATTERNS.search(text))


# R39 (2026-07-24) — EXTENDED NOISE PATTERNS (LLM hay miss trong must-drop).
# Các text này KHÔNG phải entity y khoa — chỉ là noun/adjective/filler.
_R39_EXTRA_NOISE_PATTERNS = re.compile(
    r"(?:mong\s+manh|"
    r"dễ\s+bị\s+(?:phá\s+hủy|vỡ|tổn\s+thương|"
    r"\w+(?:\s+\w+)?)|"
    r"có\s+tính\s+\w+(?:\s+\w+)?\s+cao|"
    r"thực\s+phẩm(?:\s+(?:chế\s+biến|chứa|có|hay)\s+\w+(?:\s+\w+)?)?|"
    r"hóa\s+chất|"
    r"sử\s+dụng\s+thuốc|"
    r"hạ\s+sốt|"
    r"tiếp\s+xúc\s+với(?:\s+\w+){0,2}|"
    r"phân\s+tích(?:\s+\w+)?|"
    r"chẩn\s+đoán(?:\s+\w+)?|"
    r"sàng\s+lọc\s+sớm|"
    r"theo\s+dõi(?:\s+\w+)?|"
    r"xét\s+nghiệm\s+chuyên\s+sâu|"
    r"vẫn\s+có\s+thể\s+\w+|"
    r"đã\s+từng(?:\s+\w+)?|"
    r"bệnh\s+viện|"
    r"bệnh\s+(?:gì|nào)\s*\??|"
    r"trong\s+tuần\s+qua|"
    r"cách\s+đây\s+\d+\s+\w+|"
    r"^\s*(?:này|đó|kia)\s*$|"
    r"\bK\s+(?:dùng|dùng\s+\w+))",
    re.IGNORECASE | re.UNICODE,
)


def _is_extra_noise_entity(text: str) -> bool:
    """R39 (2026-07-24): True nếu text thuộc MUST-DROP nhưng chưa được catch.

    LLM hay miss các descriptor/adjective/filler sau:
    - "mong manh", "dễ bị phá hủy", "có tính oxy hóa cao"
    - "thực phẩm", "hóa chất" (nouns alone)
    - "sử dụng thuốc", "hạ sốt", "tiếp xúc với" (action phrases)
    """
    if not text or len(text) > 80:
        return False
    return bool(_R39_EXTRA_NOISE_PATTERNS.search(text))


# R39 (2026-07-24): RECALL BOOSTER PATTERNS — patterns LLM hay MISS.
# Mỗi pattern cung cấp một hàm infer_type(text) → CHAN_DOAN/TRIEU_CHUNG/TEN_XET_NGHIEM.
# Đây là "nhắc lại" cho LLM — nếu LLM miss các entity phổ biến, ta bổ sung.
_R39_DISEASE_PATTERNS = [
    # === Critical common diseases (HIGH RECALL) ===
    # These are missing from initial patterns but very common in VN clinical notes.
    # Use compound forms (not standalone) to avoid false matches.
    r"\btăng\s+huyết\s+áp\b",
    r"\bhuyết\s+áp\s+cao\b",
    r"\bTHA\b",  # Vietnamese abbreviation
    r"\b(?:suy|yếu|giảm)\s+(?:tim|cơ\s+tim|thận|gan|hô\s+hấp|tuyến\s+giáp)",
    r"\bsuy\s+thận\s+(?:cấp|mạn|mãn)\b",
    r"\bsuy\s+tim(?:\s+(?:sung\s+huyết|mạn|cấp|trái|phải))?",
    # Compound organ + disease patterns (canonical)
    r"\b(?:viêm|ung\s+thư|suy|thoái\s+hóa|rối\s+loạn|phình|hẹp|tắc|nhiễm|"
    r"hoại\s+tử|xơ|teo|giãn|to)\s+(?:phổi|gan|thận|đại\s+tràng|dạ\s+dày|"
    r"tụy|tim|thanh\s+quản|phế\s+quản|tiết\s+niệu|thần\s+kinh|"
    r"khớp|cơ|mạch\s+máu|máu|tủy\s+xương)\b",
    # Kawasaki variants
    r"bệnh\s+kawasaki", r"\bkawasaki\b", r"viêm\s+mạch\s+máu\s+kawasaki",
    # Heart/cardiovascular
    r"viêm\s+(?:tim|cơ\s+tim|màng\s+tim|màng\s+ngoài\s+tim|nội\s+tâm\s+mạc)",
    r"bệnh\s+tim\s+(?:thiếu\s+máu|mạch\s+vành|bẩm\s+sinh|phì\s+đại|giãn)",
    r"viêm\s+mạch(?:\s+máu)?(?:\s+\w+){0,3}",
    r"phình(?:\s+giãn)?\s+động\s+mạch(?:\s+\w+){0,3}",
    r"hẹp\s+(?:động\s+mạch|tĩnh\s+mạch)(?:\s+\w+){0,2}",
    r"tắc\s+(?:động\s+mạch|tĩnh\s+mạch|mạch)(?:\s+\w+){0,3}",
    r"huyết\s+khối(?:\s+\w+){0,3}",
    r"thuyên\s+tắc(?:\s+\w+){0,3}",
    r"đột\s+tử", r"ngừng\s+tim", r"loạn\s+nhịp\s+tim",
    # Blood / enzyme — R39 fix: require specific disease names, exclude vague narrative.
    # "thiếu men X" matched too greedy (catches "thiếu men này", "thiếu máu thường sẽ ổn")
    # → exclude common narrative suffixes via negative lookahead.
    r"thiếu\s+men\s+(?:g6pd|pyruvate\s+kinase|galactokinase|"
    r"galactose[\s-]1[\s-]phosphate|phosphofructokinase)",
    r"\bg6pd\s+deficiency\b",
    r"\bthiếu\s+hụt\s+men\s+g6pd\b",
    r"\btan\s+huyết\b",
    r"\bthiếu\s+máu\s+tan\s+huyết\b",
    r"\bthiếu\s+máu\s+do\s+tan\s+huyết\b",
    r"\bthiếu\s+máu\s+(?:cấp|mạn|mãn|nặng|nhẹ|sau)\b",
    # Blood cells count
    r"\b(?:thiếu|hạ)\s+(?:hồng\s+cầu|bạch\s+cầu|tiểu\s+cầu)\b",
    # Eye / ENT
    r"viêm\s+kết\s+mạc", r"đau\s+mắt\s+đỏ", r"viêm\s+(?:tai|họng|amidan|xoang)",
    r"điếc(?:\s+\w+){0,2}", r"ù\s+tai", r"viêm\s+thính\s+giác",
    r"dị\s+tật(?:\s+\w+){0,4}", r"khiếm\s+thính",
    # Skin
    r"viêm\s+da(?:\s+\w+){0,3}", r"mày\s+đay", r"\beczema\b",
    r"\bnấm\s+da\b", r"\bghẻ\b", r"\bzona\b", r"\bherpes\b",
    r"\bvảy\s+nến\b", r"phát\s+ban", r"ban\s+đỏ", r"mề\s+đay",
    r"nổi\s+mề\s+đay", r"\bviêm\s+da\s+cơ\s+địa\b",
    # GI
    r"viêm\s+(?:dạ\s+dày|đại\s+tràng|thực\s+quản|gan|tụy|ruột(?:\s+thừa)?|hang\s+vị|túi\s+mật)",
    r"loét\s+(?:dạ\s+dày|tá\s+tràng|thực\s+quản|đại\s+tràng)",
    r"trào\s+ngược(?:\s+\w+){0,3}",
    r"\bgerd\b", r"\bibs\b",
    r"viêm\s+(?:ruột|đường\s+tiết\s+niệu|bàng\s+quang|bể\s+thận|thận)",
    r"sỏi\s+(?:thận|mật|tiết\s+niệu|bàng\s+quang)",
    r"xơ\s+gan", r"suy\s+gan", r"gan\s+nhiễm\s+mỡ",
    # Neurology
    r"đau\s+(?:nửa\s+đầu|đầu|thần\s+kinh\s+tọa)",
    r"\b(?:parkinson|alzheimer|migraine)\b",
    r"(?:trầm\s+cảm|lo\s+âu|mất\s+ngủ)",
    r"(?:động\s+kinh|co\s+giật)",
    r"tai\s+biến(?:\s+\w+){0,3}", r"đột\s+qụy",
    r"xuất\s+huyết(?:\s+não|\s+máu)?(?:\s+\w+){0,3}",
    r"nhồi\s+máu\s+(?:não|cơ\s+tim)",
    # Respiratory
    r"(?:viêm\s+phổi|hen(?:\s+phế\s+quản|\s+suyễn)?|copd|khó\s+thở|viêm\s+phế\s+quản|viêm\s+mũi|hen)",
    r"(?:tràn\s+dịch|tràn\s+khí)(?:\s+\w+){0,2}",
    r"hen\s+suyễn",
    # Rheumatology
    r"(?:thoái\s+hóa\s+khớp|viêm\s+khớp|loãng\s+xương|gút|gout|thoát\s+vị\s+đĩa\s+đệm)",
    r"(?:gãy\s+xương|gãy\s+\w+\s+xương)",
    # Cancer (compound)
    r"ung\s+thư\s+\w+", r"u\s+ác\s+tính\s+\w+", r"\bk\s+\w+\b",
    r"u\s+ác(?:\s+\w+){0,3}", r"khối\s+u(?:\s+\w+){0,3}",
    r"\bdi\s+căn(?:\s+\w+){0,3}",
    # Allergies
    r"\bdị\s+ứng(?:\s+\w+){0,3}", r"quá\s+mẫn(?:\s+\w+){0,3}",
    # Autoimmune
    r"\b(?:lupus|sjogren|sjogren's|sjögren|viêm\s+khớp\s+dạng\s+thấp)\b",
    r"viêm\s+khớp\s+dạng\s+thấp", r"viêm\s+đa\s+khớp",
    # Specific syndromes — R39 fix: REMOVE broad `bệnh/hội chứng/h/c (?:\s+\w+){1,4}`
    # pattern vì match quá rộng — bắt cả narrative/câu dài (`bệnh viện`, `bệnh gì`,
    # `bệnh cho trẻ ngay sau`, `bệnh di truyền lặn liên`). Thay bằng danh sách CỤ THỂ.
    r"(?:hội\s+chứng\s+(?:down|edwards|patau|turner|klinefelter|cushing|"
    r"guillain[\s-]barré|meniere|reye|williams|marfan|downer|reiter)|"
    r"bệnh\s+(?:kawasaki|parkinson|alzheimer|hodgkin|still|addison|cushing|"
    r"paget|sjogren|sjögren|huyết\s+sắc\s+tố|thalassemia|"
    r"viêm\s+khớp\s+dạng\s+thấp|loãng\s+xương|"
    r"thận\s+mạn|tim\s+mạch\s+vành|"
    r"hen\s+suyễn)\b)",
    # Generic "bệnh" required SPECIFIC suffix (organ or modifier)
    r"bệnh\s+(?:parkinson|alzheimer|kawasaki|still|paget|hashimoto|"
    r"sjogren|crohn|meniere|hodgkin|cushing|addison)",
    # Fever, jaundice
    r"(?:sốt\s+cao|sốt\s+\d{2,3}°?(?:C|c)|vàng\s+da|vàng\s+mắt|tăng\s+bilirubin)",
    # Pediatric
    r"tay\s+chân\s+miệng", r"sởi", r"thủy\s+đậu", r"\brubella\b", r"\bsởi\b",
    r"ho\s+gà", r"\bbạch\s+hầu\b", r"viêm\s+phổi\s+mắc\s+phải\s+cộng\s+đồng",
    r"viêm\s+màng\s+não", r"viêm\s+não\s+nhật\s+bản",
    r"rotavirus", r"\bsốt\s+xuất\s+huyết\b",
    # Hepatic
    r"viêm\s+gan\s+[a-zA-Z]", r"viêm\s+gan(?:\s+\w+){0,3}",
    # Pregnancy-related
    r"tiền\s+sản\s+giật", r"sản\s+giật", r"\bthai\s+kỳ\b",
    # Respiratory lower tract
    r"giãn\s+phế\s+quản", r"khí\s+phế\s+thủng", r"\bhen\s+suyễn\b",
    # Compound rare
    r"thận\s+\w+", r"gan\s+\w+(?=\s*[,;])",  # thận/gan X
]

_R39_SYMPTOM_PATTERNS = [
    r"đau\s+(?:ngực|bụng|đầu|lưng|họng|chân|tay|cổ|khớp|thắt\s+ngực)",
    r"đau\s+\w+(?:\s+vùng\s+\w+|\s+trái|\s+phải|\s+sau|\s+trước)",
    r"khó\s+thở", r"khó\s+nuốt", r"khó\s+nói",
    r"(?:mệt\s+mỏi|yếu\s+chi|tê\s+(?:tay|chân|ngón))",
    r"(?:buồn\s+nôn|nôn|ói)",
    r"(?:chóng\s+mặt|hoa\s+mắt|choáng\s+váng)",
    r"(?:ho(?: ra máu)?|đờm|rát\s+họng)",
    r"(?:ngứa|phát\s+ban|nổi\s+mẩn)",
    r"(?:phù|nề|sưng)(?:\s+\w+){0,2}",
    r"(?:sốt|sốt\s+cao)(?:\s+\w+){0,3}",
    r"(?:nôn\s+ra\s+máu|đi\s+ngoài\s+ra\s+máu|tiêu\s+phân\s+máu)",
    r"(?:rối\s+loạn\s+giấc\s+ngủ|mất\s+ngủ)",
    r"(?:đánh\s+trống\s+ngực|hồi\s+hộp|tức\s+ngực)",
    # Body-part specific
    r"(?:đầu\s+ngón\s+tay|đầu\s+ngón\s+chân)\s+\w+",
    r"vùng\s+(?:thượng\s+vị|hạ\s+vị|trước\s+tim)",
    # Specific symptoms
    r"\b(?:run\s+(?:tay|chân)|tay\s+run|chân\s+run)\b",
    r"\b(?:khàn\s+tiếng)\b",
    r"\b(?:tiểu\s+(?:đêm|nhiều|ít|không)|đái\s+(?:đêm|nhiều|ít|không))\b",
    r"\b(?:táo\s+bón|tiêu\s+chảy)\b",
    r"\b(?:mờ\s+mắt|hoa\s+mắt)\b",
    r"\btức\s+(?:ngực|bụng)",
    # Pain+location
    r"\bđau\s+thượng\s+vị\b",
    r"\bđau\s+hạ\s+sườn\b(?:\s+(?:phải|trái))?",
    r"\bđau\s+(?:quặn|lan)\b",
]


# R39 (2026-07-24): TEST NAMES + LAB ABNORMAL — patterns entities LLM hay miss.
_R39_TEST_PATTERNS = [
    # Common VN test names
    r"(?:chụp\s+[xX][-\s]?quang(?:\s+\w+){0,3})",
    r"(?:siêu\s+âm(?:\s+\w+){0,3})",
    r"(?:điện\s+tâm\s+đồ)",
    r"\b(?:ECG|EKG|MRI|CT(?:\s+scan)?|X[-\s]?quang)\b",
    r"(?:công\s+thức\s+máu|cf\s+máu|xét\s+nghiệm\s+máu)",
    r"(?:máu\s+lắng|men\s+gan|albumin)",
    r"(?:cấy\s+máu|cấy\s+nước\s+tiểu|cấy\s+đờm)",
    r"(?:phân\s+tích\s+nước\s+tiểu|soi\s+phân|siêu\s+âm\s+tim)",
    r"(?:nội\s+soi(?:\s+\w+){0,3})",
    r"(?:sinh\s+thiết(?:\s+\w+){0,3})",
    r"holter(?:\s+\w+){0,2}",
    r"(?:xét\s+nghiệm\s+(?:\w+\s+){0,3}(?:máu|nước\s+tiểu|phân|đờm))",
    # Lab values
    r"\b(?:HbA1c|hba1c|glycated\s+hemoglobin)\b",
    r"\b(?:eGFR|egfr|GFR)\b",
    r"\b(?:Troponin(?:\s+[IT])?|hs[-\s]?troponin)\b",
    r"\b(?:BNP|NT[-\s]?pro\s+BNP)\b",
    r"\b(?:AST|ALT|GGT|LDH|ALP|CK(?:-MB)?|CKMB)\b",
    r"\b(?:WBC|RBC|HGB|HCT|PLT|MCV|MCH|MCHC|RDW|MPV)\b",
    r"\b(?:TSH|T3|T4|FT3|FT4)\b",
    r"\b(?:CRP(?:\s+(?:hs|high[-\s]?sensitivity))?|ESR|PCT|procalcitonin)\b",
    r"\b(?:PSA|fPSA|free\s+PSA)\b",
    r"\b(?:HDL|LDL|cholesterol|triglyceride|TG)\b",
    r"\b(?:creatinine|urea|uric\s+acid|ferritin)\b",
    r"\b(?:HbA1c|hba1c)\b",
    r"(?:INR|APTT|PT|fibrinogen|D[-\s]?dimer)\b",
    r"\b(?:spO2|SaO2)\b",
]


# R39 (2026-07-24): LAB VALUE RESULTS — match number+unit patterns.
_R39_LAB_RESULT_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?:\d+(?:[.,]\d+)?)\s*"
    r"(?:mg/dl|mmol/l|µg/dl|ug/dl|ng/ml|pg/ml|miu/l|µiu/ml|uiu/ml|"
    r"g/l|meq/l|mosm/kg|u/l|iu/l|meq|ng%|g%|mm/hr|mm/giờ|"
    r"%|độ|celsius|°c|mm|cm|pmol/l|nmol/l|mmHg|cmH2O|lần/phút|"
    r"nhịp/phút|kg/m2|kg/m²)"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)


def boost_recall_ner(input_text: str, current_entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """R39 (2026-07-24): NER recall booster — tìm các entities phổ biến LLM hay MISS.

    Vấn đề: LLM đôi khi skip các bệnh/triệu chứng rõ ràng trong input (low recall).
    Vd: file 2 — "viêm tim", "phình giãn động mạch vành", "sốt cao" — đều có trong input
    nhưng LLM miss hoặc classify sai.

    Cách xử lý:
    1. Regex scan input cho các pattern phổ biến (CHAN_DOAN + TRIEU_CHUNG).
    2. Match position [start, end] exact.
    3. Nếu CHƯA có entity nào ở gần vị trí đó → bổ sung.
    4. Nếu ĐÃ CÓ entity ở vị trí đó (overlap > 50%) → skip.

    Args:
        input_text: full clinical note.
        current_entities: list entities hiện tại từ LLM Stage 1.

    Returns:
        list các entities MỚI bổ sung (KHÔNG thay đổi entities cũ — caller tự merge).
    """
    if not input_text or not _R39_DISEASE_PATTERNS:
        return []

    # Build existing position set để check overlap
    existing_positions: set[tuple[int, int]] = set()
    for ent in current_entities:
        pos = ent.get("position", [])
        if isinstance(pos, list) and len(pos) == 2:
            try:
                s, e = int(pos[0]), int(pos[1])
                existing_positions.add((s, e))
            except (ValueError, TypeError):
                continue

    booster: list[dict[str, Any]] = []
    seen_boost: set[tuple[int, int]] = set()

    def _overlap_with_existing(s: int, e: int) -> bool:
        """Check if span (s, e) overlap với bất kỳ existing entity."""
        if (s, e) in existing_positions:
            return True
        for es, ee in existing_positions:
            if max(s, es) < min(e, ee):
                return True
        return False

    def _maybe_add(text: str, s: int, e: int, etype: str):
        # Skip if already covered
        if _overlap_with_existing(s, e):
            return
        if (s, e) in seen_boost:
            return
        # Word-boundary check
        if s > 0 and input_text[s - 1].isalnum():
            return
        if e < len(input_text) and input_text[e].isalnum():
            return
        t = text.strip()
        if not t or len(t) < 3 or len(t) > 80:
            return
        if _is_chatbot_artifact(t) or _is_overly_long_narrative(t, etype):
            return
        seen_boost.add((s, e))
        existing_positions.add((s, e))
        booster.append({
            "text": t,
            # R39 (2026-07-24): Use DIACRITICS (matching grader schema).
            # Earlier versions used _normalize_type_to_ascii which strips
            # diacritics → output ASCII types → fail grader schema validation.
            # Grader schema enum includes both, but DIACRITICS preferred.
            "type": etype,
            "position": [s, e],
            "assertions": [],
            "_booster": True,
        })

    # Pattern matching — CHẨN_ĐOÁN (R39: diacritics, matches grader enum)
    for pat in _R39_DISEASE_PATTERNS:
        for m in re.finditer(pat, input_text, re.IGNORECASE | re.UNICODE):
            _maybe_add(m.group(0), m.start(), m.end(), "CHẨN_ĐOÁN")

    # TRIỆU_CHỨNG
    for pat in _R39_SYMPTOM_PATTERNS:
        for m in re.finditer(pat, input_text, re.IGNORECASE | re.UNICODE):
            _maybe_add(m.group(0), m.start(), m.end(), "TRIỆU_CHỨNG")

    # R39: TÊN_XÉT_NGHIỆM (test names)
    for pat in _R39_TEST_PATTERNS:
        for m in re.finditer(pat, input_text, re.IGNORECASE | re.UNICODE):
            t = m.group(0).strip()
            # Filter overly long matches
            if len(t) > 60:
                continue
            _maybe_add(t, m.start(), m.end(), "TÊN_XÉT_NGHIỆM")

    # R39: KẾT_QUẢ_XÉT_NGHIỆM (lab values like "120/80 mmHg", "5.6 mmol/l")
    for m in _R39_LAB_RESULT_RE.finditer(input_text):
        _maybe_add(m.group(0), m.start(), m.end(), "KẾT_QUẢ_XÉT_NGHIỆM")

    # Sort by position
    booster.sort(key=lambda e: e["position"][0])
    return booster


# R39 (2026-07-24): EXTENDED FAMILY PATTERNS — capture more isFamily cases.
_EXTENDED_FAMILY_PATTERNS = re.compile(
    r"(?:"
    # Direct family members with disease markers
    r"\b(?:bố|cha|mẹ|anh|chị|em|con|ông|bà|cô|dì|chú(?!\s+ý)|bác(?!\s+sĩ))"
    r"(?:\s+(?:trai|gái|nội|ngoại|ruột|chồng|vợ))?"
    r"\s+(?:bị|mắc|có|từng|tiền\s+sử|mất|chết|đã\s+từng|"
    r"được\s+chẩn\s+đoán|từng\s+mắc|có\s+tiền\s+sử)"
    r"\s+\w+"
    r"|"
    # Family member with "bệnh nhân"
    r"\b(?:bố|cha|mẹ|anh|chị|em|con|ông|bà|cô|dì|chú(?!\s+ý)|bác(?!\s+sĩ))"
    r"(?:\s+(?:trai|gái|nội|ngoại|ruột|chồng|vợ))?"
    r"\s+b[ệe]nh\s+nh[âa]n"
    r"|"
    # Family context
    r"gia\s+[đd][ìi]nh\s+(?:có|bị|từng|tiền\s+sử|ghi\s+nhận|ai|mắc|chưa)"
    r"|"
    r"ti[eề]n\s+s[ử]\s*gia\s+[đd][ìi]nh"
    r"|"
    r"\b(?:họ\s+hàng|người\s+thân|di\s+truyền|bẩm\s+sinh|gia\s+đình)\b"
    r"|"
    # Genetic syndromes (often family-related)
    r"\b(?:hội\s+chứng\s+down|down\s+syndrome|"
    r"hội\s+chứng\s+edwards|"
    r"hội\s+chứng\s+patau|"
    r"hội\s+chứng\s+turner|"
    r"hội\s+chứng\s+klinefelter|"
    r"bệnh\s+huyết\s+sắc\s+tố|"
    r"bệnh\s+thalassemia|"
    r"tan\s+máu\s+bẩm\s+sinh|"
    r"bệnh\s+lơ-xê-mi|leukemia(?:\s+\w+){0,2})\b"
    r"|"
    # Aggregate family context with patient context phrase
    r"người\s+thân\s+(?:của|trong)\s+(?:gia\s+đình|nhà)"
    r")",
    re.IGNORECASE | re.UNICODE,
)


# R39: EXTENDED NEGATION PATTERNS — capture multi-entity + compound negation.
_EXTENDED_NEGATION_PATTERNS = re.compile(
    r"(?:"
    r"^(?:không|chưa|chẳng|đừng)\s+"
    r"|"
    r"không\s+(?:có|ghi\s+nhận|thấy|phát\s+hiện|đáp\s+ứng|"
    r"xuất\s+hiện|biểu\s+hiện|bộc\s+lộ)\s+"
    r"|"
    r"chưa\s+(?:có|ghi\s+nhận|thấy|phát\s+hiện|xác\s+định|rõ)\s+"
    r"|"
    r"(?:âm\s+tính|negative|neg)\s+"
    r"|"
    r"loại\s+trừ\s+"
    r"|"
    r"chưa\s+từng\s+"
    r"|"
    r"không\s+còn\s+"
    r")",
    re.IGNORECASE | re.UNICODE,
)


def _enrich_assertions(input_text: str, entities: list[dict[str, Any]]) -> int:
    """R39 (2026-07-24): Enrich entities with isFamily, isNegated, isHistorical
    using EXTENDED patterns (broader than _detect_assertions_from_context).

    Returns:
        Số entity được bổ sung assertion.
    """
    if not entities or not input_text:
        return 0

    text_lower = input_text.lower()
    enriched = 0

    for ent in entities:
        if not isinstance(ent, dict):
            continue
        etype = ent.get("type", "")
        if etype not in ("CHAN_DOAN", "THUOC", "TRIEU_CHUNG"):
            continue
        pos = ent.get("position", [])
        if not (isinstance(pos, list) and len(pos) == 2):
            continue
        try:
            s = int(pos[0])
            e_pos = int(pos[1])
        except (ValueError, TypeError):
            continue
        if s < 0 or e_pos <= s:
            continue

        # Get current assertions
        existing = set(_normalize_assertions_list(ent.get("assertions", [])))
        added_any = False

        # isHistorical: section-based
        if "isHistorical" not in existing:
            section_id = _find_current_section(input_text, s)
            if section_id == "tien_su":
                existing.add("isHistorical")
                added_any = True

        # isFamily: extended pattern matching
        if "isFamily" not in existing:
            family_win_start = max(0, s - 200)
            family_win_end = min(len(input_text), s + 30)  # look BACKWARD mostly
            family_slice = text_lower[family_win_start:family_win_end]
            if _EXTENDED_FAMILY_PATTERNS.search(family_slice):
                existing.add("isFamily")
                added_any = True

        # isNegated: extended patterns
        if "isNegated" not in existing:
            # Look BACKWARD 60 chars
            pre_window = text_lower[max(0, s - 60):s]
            # Clause boundary: chop at ".", "\n", or commas followed by subject
            # Take last sentence/clause
            clauses = re.split(r"[.;\n]|(?:\b(?:nhưng|tuy\s+nhiên|ngoại\s+lệ|hiện\s+tại|"
                                r"tuy nhiên|nhưng|nhưng\s+mà)\b)", pre_window)
            last_clause = clauses[-1] if clauses else pre_window
            # Skip if clause is too long (cross-sentence) — risk false positive
            if len(last_clause) <= 60 and _EXTENDED_NEGATION_PATTERNS.search(last_clause):
                # Verify NON_NEGATION_CONTEXTS
                non_neg = re.search(
                    r"(?:không\s+tuân\s+thủ|không\s+thể|không\s+có\s+khả\s+năng|"
                    r"chưa\s+rõ|không\s+được\s+(?:thực\s+hiện|làm|chụp|tiến\s+hành))",
                    last_clause, re.UNICODE,
                )
                if not non_neg:
                    existing.add("isNegated")
                    added_any = True

        if added_any:
            ent["assertions"] = sorted(existing)
            enriched += 1

    return enriched




# LLM thỉnh thoảng extract cả câu narrative thay vì concept y khoa — too long → drop.
def _is_overly_long_narrative(text: str, etype: str) -> bool:
    """R39: Drop entities quá dài không phải là concept y khoa (vd cả câu có dấu chấm phẩy).

    Quy tắc:
    - TÊN_XÉT_NGHIỆM / KQ_XN: max 60 chars
    - Còn lại: max 80 chars HOẶC chứa 3+ dấu câu narrative (.,;:)
    """
    if not text:
        return False
    n = len(text)
    narrative_punct = sum(1 for c in text if c in ".,;:")
    if etype in ("TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM") and n > 60:
        return True
    if n > 80 and narrative_punct >= 3:
        return True
    return False


def assemble_record(
    input_text: str,
    raw_entities: Iterable[dict[str, Any]],
    retriever: RxNormRetriever,
    icd_retriever: Optional[ICDRetriever] = None,
    llm_client: Any = None,
) -> list[dict[str, Any]]:
    """Build list thực thể cuối cùng cho một record.

    Pipeline:
      1. Chuẩn hoá entities (validate position, expand duplicate, dedup, drop noise)
      2. Clean từng entity text (strip modifiers, verbs, parens, drop duration)
      3. Dedup cuối cùng (R10 STRICT + overlap dedup + R22 cho TÊN_XN)
      4. Gán candidates (RxNorm cho THUỐC, ICD cho CHẨN_ĐOÁN)
      5. Sort theo position
    """
    if retriever is not None and llm_client is not None:
        retriever._llm_client = llm_client
    if icd_retriever is not None and llm_client is not None:
        icd_retriever._llm_client = llm_client

    validated = _prepare_validated_entities(input_text, raw_entities)

    seen_test_names: set[str] = set()
    seen_entities: list[tuple[str, str, list[int]]] = []  # (norm_text, type, [start, end])

    final: list[dict[str, Any]] = []
    for ent in validated:
        record = _emit_entity_record(
            ent, input_text, validated, retriever, icd_retriever,
            seen_test_names, seen_entities,
            skip_attach=True,
        )
        # R37 (2026-07-16): Handle split case (merged test_name+value)
        if record is not None and isinstance(record, dict) and record.get("_split"):
            split_ents = record["entities"]
            for sent in split_ents:
                # Re-emit each split entity through pipeline
                final.append(sent)
        elif record is not None:
            final.append(record)

    # Phase 2: Parallel candidate attachment across CPU/Thread workers (Upgrade F)
    _CANDIDATE_TYPES = ("THUỐC", "CHẨN_ĐOÁN", "TRIỆU_CHỨNG")
    if len(final) > 1 and (retriever is not None or icd_retriever is not None):
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(final))) as pool:
            futures = [
                pool.submit(_attach_candidates, rec, rec["text"], rec["type"], rec, validated, retriever, icd_retriever)
                for rec in final if rec["type"] in _CANDIDATE_TYPES
            ]
            concurrent.futures.wait(futures)
    elif len(final) == 1:
        rec = final[0]
        if rec["type"] in _CANDIDATE_TYPES:
            _attach_candidates(rec, rec["text"], rec["type"], rec, validated, retriever, icd_retriever)

    # Phase 3 cuối assemble_record (Fix 2.4)
    for rec in final:
        if "isHistorical" in rec.get("assertions", []):
            if _find_current_section(input_text, rec["position"][0]) != "tien_su":
                rec["assertions"] = [a for a in rec["assertions"] if a != "isHistorical"]

    final.sort(key=lambda e: e["position"][0])

    # R38 (2026-07-23): Dedupe cuối cùng theo (text_lower, type) để giảm WER.
    # R38 (2026-07-23): DISABLED _dedupe_by_text_type — quá aggressive, mất position info
    # và gây regression trên Stage 3 (LLM không còn nhiều candidates per record).
    # LLM sẽ tự xử lý duplicate detection qua prompts + few-shot examples.
    # final = _dedupe_by_text_type(final)

    # R39 (2026-07-24): POST-FILTERS MỚI — chống schema reject + position drift.
    # 1) Convert diacritics type → ASCII (vd KẾT_QUẢ_XÉT_NGHIỆM → KET_QUA_XET_NGHIEM).
    # 2) Enforce position: nếu input[pos] != text → re-search hoặc drop.
    # 3) Drop chatbot artifacts ("Cảm ơn bạn...").
    # 4) Drop overly long narrative entities.
    cleaned_final: list[dict[str, Any]] = []
    for rec in final:
        # 1) Normalize type — R39 (2026-07-24) UPDATE: GIỮ DIACRITICS theo mặc định.
        # Grader hiện tại chấp nhận cả diacritics (THUỐC, CHẨN_ĐOÁN, ...). Chỉ
        # fallback ASCII khi `TYPE_TO_ASCII=1`. Nếu input đã là ASCII thì giữ
        # nguyên backward compatible (OUTPUT_SCHEMA chấp nhận CẢ 2 form).
        rec["type"] = _restore_diacritics_type(rec.get("type", ""))
        etype = rec["type"]
        text = str(rec.get("text", "")).strip()

        # 4) Drop overly long narrative (trước khi enforce position để tránh tốn)
        if _is_overly_long_narrative(text, etype):
            logger.debug("[R39] Drop overly long narrative '%s' (%s)", text[:80], etype)
            continue

        # 5) Drop extra noise entities (MUST-DROP mà prompt chưa catch):
        #    "mong manh", "dễ bị phá hủy", "thực phẩm alone", "sử dụng thuốc", etc.
        if _is_extra_noise_entity(text):
            logger.debug("[R39] Drop extra noise '%s' (%s)", text[:80], etype)
            continue

        # 3) Drop chatbot artifacts
        if _is_chatbot_artifact(text):
            logger.debug("[R39] Drop chatbot artifact '%s'", text[:80])
            continue

        # 2) Enforce position strict
        rec_enforced = _enforce_position_strict(input_text, rec)
        if rec_enforced is None:
            continue

        cleaned_final.append(rec_enforced)

    # Re-sort sau khi drop
    cleaned_final.sort(key=lambda e: e["position"][0])

    # R39 (2026-07-24): RECALL BOOSTER — bổ sung entities phổ biến LLM hay miss
    # (vd "viêm tim", "phình mạch vành", "sốt cao", "sốt 39°C" trong file 2).
    # Chạy CUỐI CÙNG sau position enforcement để tránh conflict với LLM output.
    try:
        boosted = boost_recall_ner(input_text, cleaned_final)
        if boosted:
            # Attach empty candidates cho boosted entities (boost stage chưa lookup RAG,
            # candidate sẽ được add ở re-run pipeline. Để [] cho entities mới.)
            cleaned_final.extend(boosted)
            cleaned_final.sort(key=lambda e: e["position"][0])
            logger.debug("[R39] Boosted %d entities (recall)", len(boosted))
    except Exception as exc:
        logger.warning("[R39] Recall booster failed: %s", exc)

    # R39 (2026-07-24): HARD-REJECT ICD codes cho FALSE-POSITIVE cases.
    # Một số ICD codes dễ bị LLM/rag trả nhầm concept → filter cuối cùng.
    # Đảm bảo output không chứa code sai concept.
    cleaned_final = _apply_hard_reject_icd(cleaned_final)

    # R39 (2026-07-24): DEDUP by (text_normalized, type) trong CÙNG paragraph.
    # NHƯNG KHÔNG dedupe nếu entities ở KHÁNG section (cách nhau > 500 chars).
    # Lý do: LLM hay repeat "thiếu men G6PD" 5-13 lần trong cùng đoạn văn → WER explosion.
    # Quy tắc:
    # - Cùng (text_lower, type) + span overlap > 50% → DROP later
    # - Cùng (text_lower, type) + distance < 300 chars (cùng paragraph) → DROP later, KEEP first
    # - Khác section (distance > 500 chars hoặc context khác) → KEEP all
    cleaned_final = _dedupe_by_proximity(cleaned_final)

    # R39 (2026-07-24): FINAL CLEANUP — Strip tất cả internal keys (prefix `_`)
    # như `_booster` trước khi output. Đây là metadata debug, không nên xuất ra file
    # vì grader / validator có thể reject.
    for e in cleaned_final:
        for k in list(e.keys()):
            if k.startswith("_"):
                e.pop(k, None)

    return cleaned_final


def _dedupe_by_text_type(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """R38 (2026-07-23): Dedupe entities theo (text_lower, type), giữ FIRST occurrence.

    Lý do: scoring formula ghép predicted vs gold theo (position, type) với WER word-level.
    Nếu cùng text xuất hiện 12 lần (do `_expand_duplicates`), WER explosion vì
    11 extras không match gold → ảnh hưởng text_score rất nặng.

    Hành vi:
      - Với mỗi (text_lower, type), giữ entity đầu tiên (position sớm nhất).
      - Bỏ qua entities có empty text.
      - Sort output theo position (giữ nguyên thứ tự xuất hiện trong input).

    Args:
        entities: list entity dicts từ assemble_record.

    Returns:
        list entities đã dedupe.
    """
    if not entities:
        return entities
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for ent in entities:
        text = str(ent.get("text", "")).strip()
        etype = str(ent.get("type", ""))
        if not text:
            continue
        key = (text.lower(), etype)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ent)
    # Sort lại theo position (giữ thứ tự tự nhiên)
    deduped.sort(key=lambda e: e.get("position", [0, 0])[0])
    return deduped


def _dedupe_by_proximity(entities: list[dict[str, Any]], same_para_dist: int = 300) -> list[dict[str, Any]]:
    """R39 (2026-07-24): Dedup entities cùng (text_lower, type) theo PROXIMITY.

    Khác `_dedupe_by_text_type` (strict dedup): function này thông minh hơn:
    - Chỉ dedup nếu 2 entities gần nhau (< same_para_dist chars) → cùng paragraph → DROP later
    - Nếu cùng text + KHÁNG section (distance > same_para_dist) → KEEP all (R10 STRICT)
    - Đây là safety net chính — LLM hay repeat bệnh nhiều lần trong cùng đoạn văn
      → gây WER explosion.

    Args:
        entities: list entity dicts.
        same_para_dist: max distance chars for "same paragraph" (default 300).

    Returns:
        list entities đã dedupe.
    """
    if not entities:
        return entities

    # Sort by position first
    sorted_ents = sorted(entities, key=lambda e: e.get("position", [0, 0])[0])

    # Track FIRST occurrence for each (text_lower, type)
    seen: dict[tuple[str, str], dict] = {}
    deduped: list[dict[str, Any]] = []
    dropped = 0

    for ent in sorted_ents:
        text = str(ent.get("text", "")).strip()
        etype = str(ent.get("type", ""))
        if not text:
            continue

        # Normalize text (lowercase + collapse whitespace) để so sánh
        import re as _re
        norm_text = _re.sub(r"\s+", " ", text.lower()).strip()
        key = (norm_text, etype)

        pos = ent.get("position", [0, 0])
        if not (isinstance(pos, list) and len(pos) == 2):
            deduped.append(ent)
            continue
        try:
            s = int(pos[0])
        except (ValueError, TypeError):
            deduped.append(ent)
            continue

        if key in seen:
            # Same text+type seen before. Check proximity.
            first_ent = seen[key]
            first_pos = first_ent.get("position", [0, 0])
            try:
                first_s = int(first_pos[0])
            except (ValueError, TypeError):
                first_s = s
            distance = abs(s - first_s)
            if distance <= same_para_dist:
                # Cùng paragraph → DROP (LLM lặp lại)
                dropped += 1
                logger.debug(
                    "[R39] Dedup proximity: drop '%s' (%s) at pos %d (first at %d, dist=%d)",
                    text[:40], etype, s, first_s, distance,
                )
                continue
            else:
                # Khác section → KEEP (R10 STRICT)
                deduped.append(ent)
                # Update seen — actually keep first, allow multiple occurrences in different sections
                # So we don't update seen[key]
        else:
            seen[key] = ent
            deduped.append(ent)

    if dropped:
        logger.info("[R39] Dropped %d duplicate entities via proximity", dropped)

    deduped.sort(key=lambda e: e.get("position", [0, 0])[0])
    return deduped


def _split_drug_disease_connector(
    input_text: str,
    raw_entities: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Tách cụm 'Thuốc [connector] Bệnh/Triệu chứng' mà LLM vô tình gộp (R1).

    Ví dụ: 'doxycycline cho viêm tuyến mồ hôi' -> 'doxycycline' (THUỐC) + 'viêm tuyến mồ hôi' (CHẨN_ĐOÁN).
    """
    out: list[dict[str, Any]] = []
    connector_pattern = re.compile(r"\s+(?:cho|trị|điều\s+trị|chữa)\s+", re.IGNORECASE)

    for ent in raw_entities:
        if not isinstance(ent, dict):
            continue
        text = str(ent.get("text", "")).strip()
        etype = ent.get("type", "")
        if etype == "THUỐC" and connector_pattern.search(text):
            parts = connector_pattern.split(text, maxsplit=1)
            if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                drug_part = parts[0].strip()
                disease_part = parts[1].strip()
                # Tạo 2 entities tách biệt, validate_positions sẽ tự tìm exact span
                out.append({**ent, "text": drug_part, "type": "THUỐC", "position": [0, 0]})
                out.append({**ent, "text": disease_part, "type": "CHẨN_ĐOÁN", "position": [0, 0]})
                logger.debug("Tách Drug+Disease: %r -> %r + %r", text, drug_part, disease_part)
                continue

        # R39 (2026-07-14): Compound drug name splitting
        # vd 'vancozosynbactrim' → ['vancomycin', 'zosyn', 'bactrim']
        if etype == "THUỐC" and text and " " not in text:
            try:
                from src.rxnorm_rag import _DRUG_ALIASES
                text_lower = text.lower()
                if text_lower in _DRUG_ALIASES:
                    aliased = _DRUG_ALIASES[text_lower]
                    if isinstance(aliased, list) and len(aliased) > 1:
                        for sub_drug in aliased:
                            out.append({**ent, "text": sub_drug, "type": "THUỐC", "position": [0, 0]})
                        logger.debug("R39: Tách compound drug '%s' -> %s", text, aliased)
                        continue
            except Exception as e:
                logger.debug("R39: compound split failed for '%s': %s", text, e)

        out.append(ent)
    return out



# R37 (2026-07-16): Augment entities từ input patterns LLM hay miss
# - Drug + cho/trị + Disease pattern (R1 split, R37 auto-detect)
# - Compound symptoms (buồn nôn, đau đầu, ...) thường bị LLM tách nhỏ

# R37 FIX (2026-07-16 v2): Pattern drug + connector + disease
# - drug group: starts with ASCII/VN letter, then 1+ letters/hyphens
# - disease group: word-bounded, max 6 words, exclude trailing common VN particles
#   (trong, và, của, tại, ở, lúc, khi, được, là, nay, hôm, qua, đến, sang, ...)
#   + KHÔNG span newlines (`[^\S\n]+` thay vì `\s+`) để tránh capture "viêm tuyến
#   mồ hôi\n    - atenolol" → "viêm tuyến mồ hôi"
_R1_PATTERN = re.compile(
    r"\b([A-ZÀ-Ỹa-zà-ỹ][A-Za-zÀ-Ỹa-zà-ỹ\-]+)[ \t]+"
    r"(?:cho|trị|điều\s+trị|chữa)[ \t]+"
    r"([A-Za-zÀ-Ỹ][A-Za-zÀ-Ỹà-ỹ\-]+"
    r"(?:[^\S\n]+(?!(?:trong|và|của|tại|ở|lúc|khi|được|là|nay|hôm|qua|đến|sang|tới|từ|như|khi|đã|sẽ|rồi|vẫn|cũng|hay)\b)"
    r"[A-Za-zÀ-Ỹà-ỹ\-]+){0,5})",
    re.IGNORECASE | re.UNICODE,
)

_COMPOUND_SYMPTOMS = [
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
    ("sốt nhẹ", "TRIỆU_CHỨNG"),
    ("sốt", "TRIỆU_CHỨNG"),
    ("ho khạc đờm", "TRIỆU_CHỨNG"),
    ("ho ra máu", "TRIỆU_CHỨNG"),
    ("ho nhiều", "TRIỆU_CHỨNG"),
    ("ho", "TRIỆU_CHỨNG"),
    ("mệt mỏi", "TRIỆU_CHỨNG"),
    ("đau ngực trái", "TRIỆU_CHỨNG"),
    ("đau ngực phải", "TRIỆU_CHỨNG"),
    ("đau ngực", "TRIỆU_CHỨNG"),
    ("khó thở nhẹ", "TRIỆU_CHỨNG"),
    ("khó thở khi gắng sức", "TRIỆU_CHỨNG"),
    ("khó thở", "TRIỆU_CHỨNG"),
    ("khò khè", "TRIỆU_CHỨNG"),
    ("tiếng rít", "TRIỆU_CHỨNG"),
    ("đánh trống ngực", "TRIỆU_CHỨNG"),
    ("đau đầu", "TRIỆU_CHỨNG"),
    ("đau bụng", "TRIỆU_CHỨNG"),
    ("đau lưng", "TRIỆU_CHỨNG"),
    ("đau cổ", "TRIỆU_CHỨNG"),
    ("đau khớp", "TRIỆU_CHỨNG"),
    ("đau cơ", "TRIỆU_CHỨNG"),
    ("đau họng", "TRIỆU_CHỨNG"),
    ("mất ngủ", "TRIỆU_CHỨNG"),
    ("đau thượng vị", "TRIỆU_CHỨNG"),
    ("phù chi dưới", "TRIỆU_CHỨNG"),
]


def _ensure_drug_disease_split(
    input_text: str,
    existing_entities: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """R37 (2026-07-16): Auto-add disease bị LLM miss khi input có pattern 'drug cho disease'.

    Example:
        Input: 'doxycycline cho viêm tuyến mồ hôi'
        LLM extracted: 'doxycycline' only
        → add 'viêm tuyến mồ hôi' (CHẨN_ĐOÁN)

    Returns:
        List các entities bổ sung (có thể rỗng). Caller appends.
    """
    additional: list[dict[str, Any]] = []
    existing_texts = {e.get("text", "").strip().lower() for e in existing_entities}

    for m in _R1_PATTERN.finditer(input_text):
        drug_part = m.group(1).strip()
        disease_part = m.group(2).strip()

        if disease_part.lower() in existing_texts:
            continue
        if len(disease_part) < 4:
            continue

        # Validate drug (simple check)
        is_drug = True
        try:
            from src.rxnorm_rag import _DRUG_INN_WHITELIST
            from src.postprocess import _DRUG_NAMES_UNIONED
            drug_lower = drug_part.lower().strip()
            is_drug = (drug_lower in _DRUG_INN_WHITELIST
                       or drug_lower in _DRUG_NAMES_UNIONED)
        except Exception:
            pass
        if not is_drug:
            continue

        disease_pos = _find_span(input_text, disease_part, start=0)
        if disease_pos is None:
            continue

        s, e = disease_pos
        overlap = any(
            en.get("position", [0, 0])[0] < e
            and en.get("position", [0, 0])[1] > s
            for en in existing_entities
        )
        if overlap:
            continue

        additional.append({
            "text": disease_part,
            "type": "CHẨN_ĐOÁN",
            "position": list(disease_pos),
            "assertions": [],
            "candidates": [],
        })
        logger.debug("R37 auto-extracted disease from R1 pattern: %r", disease_part)

    return additional


def _ensure_compound_symptoms(
    input_text: str,
    existing_entities: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """R37 (2026-07-16): Auto-add compound symptoms bị LLM miss hoặc split nhỏ.

    Common cases:
    - 'buồn nôn' extracted as 'nôn' (substring)
    - 'đau đầu' extracted as 'đầu' or 'đau'
    - 'sốt cao' extracted as 'sốt' only

    Returns:
        List các entities bổ sung.
    """
    additional: list[dict[str, Any]] = []
    existing_texts = {e.get("text", "").strip().lower() for e in existing_entities}
    input_lower = input_text.lower()

    for compound_text, etype in _COMPOUND_SYMPTOMS:
        compound_lower = compound_text.lower()
        if compound_lower in existing_texts:
            continue

        if compound_lower not in input_lower:
            continue

        pos = _find_span(input_text, compound_text, start=0)
        if pos is None:
            continue

        s, e = pos
        overlap = any(
            en.get("position", [0, 0])[0] < e
            and en.get("position", [0, 0])[1] > s
            for en in existing_entities
        )
        if overlap:
            continue

        assertions = _detect_assertions_from_context(compound_text, input_text, etype, s)

        additional.append({
            "text": compound_text,
            "type": etype,
            "position": list(pos),
            "assertions": assertions,
            "candidates": [],
        })
        logger.debug("R37 auto-extracted compound symptom: %r", compound_text)

    return additional



def _split_test_name_value_connector(
    input_text: str,
    raw_entities: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Tách cụm 'Tên xét nghiệm [là/:/=] Giá trị' mà LLM vô tình gộp vào 1 entity (R7).

    Ví dụ: 'bilirubin toàn phần (tbili) là 2.4' -> 'bilirubin toàn phần (tbili)' (TÊN_XN) + '2.4' (KQ_XN).
           'kali là 2.4' -> 'kali' (TÊN_XN) + '2.4' (KQ_XN).
    """
    out: list[dict[str, Any]] = []
    connector_pattern = re.compile(r"\s+(?:là|=|:|đạt|ở\s+mức)\s+(?=\d)", re.IGNORECASE)

    for ent in raw_entities:
        if not isinstance(ent, dict):
            continue
        text = str(ent.get("text", "")).strip()
        etype = ent.get("type", "")
        if etype in ("KẾT_QUẢ_XÉT_NGHIỆM", "TÊN_XÉT_NGHIỆM") and connector_pattern.search(text):
            parts = connector_pattern.split(text, maxsplit=1)
            if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                name_part = parts[0].strip()
                val_part = parts[1].strip()
                if not re.match(r'^[\d,.\s]+$', name_part):
                    out.append({**ent, "text": name_part, "type": "TÊN_XÉT_NGHIỆM", "position": [0, 0]})
                    out.append({**ent, "text": val_part, "type": "KẾT_QUẢ_XÉT_NGHIỆM", "position": [0, 0]})
                    logger.debug("Tách Test+Value: %r -> %r + %r", text, name_part, val_part)
                    continue
        out.append(ent)
    return out


def _filter_vital_signs_dump(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Theo xác nhận của user: các chuỗi mã hóa/sinh hiệu (như VS98.3 12987 56 18 99RA) không bị cấm nếu được trích xuất là KẾT_QUẢ_XÉT_NGHIỆM."""
    return list(entities)


def align_and_expand_entities(
    input_text: str,
    raw_entities: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Universal Alignment Engine — Bước 2 của kiến trúc 2-Step NER.

    LLM chỉ cần trả về text + type + assertions (không cần đếm position).
    Hàm này tự động:
      1. Pre-clean & split: tách cụm bị gộp (Drug+Disease, Test+Value), clean verb prefix.
      2. Exhaustive multi-pass alignment: tìm TẤT CẢ occurrences của mỗi text
         trong input_text qua 3 lớp (Exact → Modifiers-stripped → Typo recovery).
      3. Tạo 1 entity riêng biệt cho MỖI occurrence → không bao giờ miss duplicate.
      4. Dedup overlap + drop substring + filter noise.

    Args:
        input_text: văn bản gốc (original, không phải highlighted/chunked).
        raw_entities: list entities thô từ LLM (chỉ cần text + type + assertions).

    Returns:
        list entities với position chính xác 100%, đầy đủ duplicates.
    """
    # ── Pre-process: tách Drug+Disease & Test+Value connector ────────────────────────
    raw_list = _split_test_name_value_connector(input_text, raw_entities)
    raw_list = _split_drug_disease_connector(input_text, raw_list)

    # ── Pre-clean: strip verb prefix, parens admin, leading verbs trên từng entity ──
    pre_cleaned: list[dict[str, Any]] = []
    for ent in raw_list:
        text = str(ent.get("text", "")).strip()
        etype = ent.get("type", "")
        if not text or etype not in (
            "THUỐC", "CHẨN_ĐOÁN", "TRIỆU_CHỨNG", "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"
        ):
            continue
        cleaned = _clean_entity_text(text, etype)
        if cleaned is None:
            continue
        if cleaned != text:
            ent = {**ent, "text": cleaned}
        pre_cleaned.append(ent)

    # ── Map tuần tự: Giữ nguyên các assertions độc lập của LLM ──────────────────
    # Thay vì merge assertions (làm lây lan isHistorical/isNegated sai), ta gom 
    # danh sách các entity do LLM sinh ra để map tuần tự vào các vị trí trong text.
    from collections import defaultdict
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for ent in pre_cleaned:
        text = str(ent.get("text", "")).strip()
        etype = ent.get("type", "")
        groups[(text.lower(), etype)].append(ent)

    # ── Prefix/suffix modifiers regex (dùng cho stripped scan) ─────────────────
    _MOD_PREFIX = re.compile(
        r"^(tăng|giảm|có|không|đang|bị|rõ|rõ\s+rệt|ít|nhiều|hơi|hơn|khoảng|có\s+thể)\s+",
        re.IGNORECASE | re.UNICODE,
    )
    _MOD_SUFFIX = re.compile(
        r"\s+(nhẹ|nặng|vừa|nhẹ\s+nhàng|nặng\s+nề|nhẹ\s+vừa|vừa\s+phải)$",
        re.IGNORECASE | re.UNICODE,
    )

    # ── Exhaustive Multi-pass Alignment ────────────────────────────────────────
    aligned: list[dict[str, Any]] = []
    input_lower = input_text.lower()

    for (text_lower, etype), ents in groups.items():
        base_text = str(ents[0].get("text", "")).strip()
        if not base_text or len(base_text) < 2:
            continue

        all_spans: set[tuple[int, int]] = set()

        # Pass 1: Exact substring scan (case-insensitive with Word Boundary check)
        start = 0
        while True:
            idx = input_lower.find(text_lower, start)
            if idx < 0:
                break
            end_idx = idx + len(base_text)
            # Kiểm tra word boundary để tránh "nôn" khớp bên trong "buồn nôn"
            if (idx > 0 and input_text[idx - 1].isalnum()) or (end_idx < len(input_text) and input_text[end_idx].isalnum()):
                start = idx + 1
                continue
            all_spans.add((idx, end_idx))
            start = idx + 1

        # Pass 2: Universal Accent-Insensitive & RapidFuzz Sliding Window Alignment (giữ trọn vẹn cụm từ)
        if not all_spans:
            hint_pos = 0
            pos_field = ents[0].get("position", [0, 0])
            if isinstance(pos_field, list) and len(pos_field) == 2:
                try:
                    hint_pos = int(pos_field[0])
                except (ValueError, TypeError):
                    hint_pos = 0
            fuzzy_res = _fuzzy_locate_in_text(base_text, input_text, hint_pos=hint_pos)
            if fuzzy_res is not None:
                rs, re_ = fuzzy_res
                # R28 (2026-07-13): Reject if span boundaries fall mid-word.
                # _fuzzy_locate_in_text không enforce boundary, có thể match giữa từ.
                boundary_ok = (
                    (rs == 0 or not input_text[rs - 1].isalnum())
                    and (re_ >= len(input_text) or not input_text[re_].isalnum())
                )
                if not boundary_ok:
                    fuzzy_res = None
            if fuzzy_res is not None:
                rs, re_ = fuzzy_res
                all_spans.add((rs, re_))
                for ent in ents:
                    ent["text"] = input_text[rs:re_]
                logger.debug(
                    "Align: fuzzy recovery '%s' → '%s' at [%d, %d]", base_text, input_text[rs:re_], rs, re_
                )

        # Pass 3: Typo recovery (R23)
        if not all_spans:
            hint_pos = 0
            pos_field = ents[0].get("position", [0, 0])
            if isinstance(pos_field, list) and len(pos_field) == 2:
                try:
                    hint_pos = int(pos_field[0])
                except (ValueError, TypeError):
                    hint_pos = 0
            recovered = _try_recover_typo(base_text, input_text, hint_pos=hint_pos)
            if recovered is not None:
                recovered_text, rs, re_ = recovered
                all_spans.add((rs, re_))
                for ent in ents:
                    ent["text"] = recovered_text
                logger.debug(
                    "Align: typo recovery '%s' → '%s' at %d", base_text, recovered_text, rs
                )

        # Pass 4: Modifiers-stripped scan (Fallback cuối cùng khi không thể khớp cụm từ gốc)
        if not all_spans:
            stripped_text = _MOD_PREFIX.sub("", text_lower).strip()
            stripped_text = _MOD_SUFFIX.sub("", stripped_text).strip()
            if stripped_text and stripped_text != text_lower and len(stripped_text) >= 4:
                start = 0
                while True:
                    idx = input_lower.find(stripped_text, start)
                    if idx < 0:
                        break
                    end_idx = idx + len(stripped_text)
                    if (idx > 0 and input_text[idx - 1].isalnum()) or (end_idx < len(input_text) and input_text[end_idx].isalnum()):
                        start = idx + 1
                        continue
                    span = (idx, end_idx)
                    if not any(s <= idx and end_idx <= e for s, e in all_spans):
                        all_spans.add(span)
                    start = idx + 1

        # R34 (2026-07-13): Emit 1 entity riêng cho MỖI occurrence (R10 STRICT).
        # Trước đây chỉ emit len(LLM_ents) entities → MISS tất cả duplicate occurrences.
        # Mỗi span trong all_spans = 1 entity, sharing assertions từ LLM.
        if not all_spans:
            logger.debug("Align: không tìm được span cho '%s' (%s) → bỏ", base_text, etype)
            continue

        # Use FIRST ent's assertions (LLM-provided) for ALL emitted entities.
        # Nếu có nhiều ent (nhiều spans với assertions khác nhau → cẩn thận).
        # Hầu hết LLM chỉ trả 1 ent → dùng assertions của nó.
        ref_ent = ents[0]
        for span_start, span_end in sorted(all_spans):
            actual_text = input_text[span_start:span_end]
            new_ent = {
                **ref_ent,
                "text": actual_text,
                "position": [span_start, span_end],
            }
            aligned.append(new_ent)

    # ── Split long imaging results (R31) ────────────────────────────────────────
    aligned = _split_long_results(input_text, aligned)

    # ── Overlap Dedup (R10 STRICT + R22) ─────────────────────────────────────
    aligned = dedupe_entities(aligned)

    # ── Drop substring entities ─────────────────────────────────────────────────────
    aligned = _drop_substring_entities(aligned)

    # R29 (2026-07-13): Cross-type drop — symptom khi diagnosis duplicate span.
    # 'rối loạn lo âu' (CHẨN_ĐOÁN) chứa 'lo âu' (TRIỆU_CHỨNG) → drop symptom.
    aligned = _drop_symptom_when_diagnosis_present(aligned)

    # R37 (2026-07-16): Cross-type substring drop — short entity (TÊN_XN/KQ_XN, 4-9 chars)
    # nếu nằm hoàn toàn trong longer entity (CHẨN_ĐOÁN/CD, >=10 chars) với position overlap.
    # VD: 'mạch' (TÊN_XN) inside 'bệnh tim mạch do xơ vữa động mạch' (CHẨN_ĐOÁN) → drop 'mạch'.
    aligned = _drop_short_substring_inside_longer(aligned)

    # R37 (2026-07-19): Strip assertions from TÊN_XN/KQ_XN per spec (these types
    # don't carry isHistorical/isFamily/isNegated).
    aligned = _strip_assertions_for_test_types(aligned)

    # R37-bis (2026-07-14): Auto-fix LLM miss khi tách drug + clinical parens.
    # 'metoprolol (reduced from 50mg to 25mg daily)' → 1 entity THUỐC
    # (drop misclassified 'reduced from...' TÊN_XÉT_NGHIỆM, extend drug to include parens).
    aligned = _drop_drug_parens_misclassification(aligned, input_text)

    # R37-bis (2026-07-14): Auto-fix LLM miss khi tách compound disease term.
    # 'ung thư' + 'phổi' → merge thành 1 entity 'ung thư phổi' CHẨN_ĐOÁN.
    aligned = _merge_compound_disease_terms(aligned)

    # ── Filter lifestyle/duration noise ──────────────────────────────────────────
    aligned = _filter_lifestyle_entities(aligned)

    # R28 (2026-07-13): Final word-boundary + text-claim validator.
    # Span phải: input_text[s:e] == entity.text (case-insensitive) VÀ boundaries thuộc word boundary.
    # R37 (2026-07-16): Auto-augment missing entities từ input patterns
    # (drug-disease split, compound symptoms). LLM hay miss 1 phần.
    try:
        extra_dd = _ensure_drug_disease_split(input_text, aligned)
        extra_cs = _ensure_compound_symptoms(input_text, aligned)
        if extra_dd or extra_cs:
            aligned.extend(extra_dd)
            aligned.extend(extra_cs)
            logger.debug(
                "R37 augmented: %d disease (R1) + %d compound symptoms",
                len(extra_dd), len(extra_cs),
            )
    except Exception as exc:
        logger.debug("R37 augment failed (non-fatal): %s", exc)

    aligned = [e for e in (_validate_span_or_drop(input_text, e) for e in aligned) if e]

    return aligned


def _prepare_validated_entities(
    input_text: str,
    raw_entities: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Backward-compatible wrapper sang align_and_expand_entities (2-Step Architecture).

    Từ phiên bản 2-Step Architecture, hàm này delegate toàn bộ sang
    align_and_expand_entities — Universal Alignment Engine mới.
    Giữ lại tên hàm cũ để backward-compatible với các test scripts.
    """
    return align_and_expand_entities(input_text, raw_entities)



_NORMAL_RESULT_SPLIT_RE = re.compile(
    r"^(?P<test>.+?)\s+(?P<result>(?:là\s+)?(?:không\s+ghi\s+nhận\s+(?:gì\s+)?bất\s+thường|không\s+có\s+gì\s+đáng\s+chú\s+ý|bình\s+thường))$",
    re.IGNORECASE | re.UNICODE,
)


def _split_long_results(
    input_text: str,
    entities: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Tách các long KẾT_QUẢ_XÉT_NGHIỆM thành nhiều entities (R31).

    LLM 7B hay gộp cả đoạn kết quả CT/MRI/X-quang dài vào 1 KQ_XN.
    Logic: với mỗi entity có type=KQ_XN và text chứa pattern "cho thấy ..." / "ghi nhận ...",
    thử tách thành: 1 TÊN_XN (test) + N CHẨN_ĐOÁN (findings).
    Nếu không tách được → giữ entity gốc.
    """
    out: list[dict[str, Any]] = []
    for ent in entities:
        text = str(ent.get("text", "")).strip()
        etype = ent.get("type", "")
        pos = ent.get("position", [0, 0])

        if not text or etype != "KẾT_QUẢ_XÉT_NGHIỆM":
            out.append(ent)
            continue

        # Try split long imaging
        split_result = _split_long_imaging_result(text, etype, input_text, pos if isinstance(pos, list) else [0, 0])
        if split_result and len(split_result) >= 2:
            logger.debug(
                "Split long imaging result '%s' → %d entities",
                text[:60],
                len(split_result),
            )
            out.extend(split_result)
            continue

        # Try split normal result phrase (vd "chụp x-quang ngực không ghi nhận gì bất thường")
        m_norm = _NORMAL_RESULT_SPLIT_RE.match(text)
        if m_norm:
            test_raw = m_norm.group("test").strip()
            res_raw = m_norm.group("result").strip()
            test_cleaned = _TEST_VERB_PREFIX_RE.sub("", test_raw).strip() or test_raw
            test_pos = _find_span(input_text, test_cleaned) or _find_span(input_text, test_raw)
            res_pos = _find_span(input_text, res_raw, start=pos[0] if isinstance(pos, list) else 0) or _find_span(input_text, res_raw)
            if test_pos and res_pos:
                out.append({
                    "text": test_cleaned,
                    "type": "TÊN_XÉT_NGHIỆM",
                    "position": list(test_pos),
                    "assertions": [],
                    "candidates": [],
                })
                out.append({
                    "text": res_raw,
                    "type": "KẾT_QUẢ_XÉT_NGHIỆM",
                    "position": list(res_pos),
                    "assertions": [],
                    "candidates": [],
                })
                continue

        out.append(ent)

    for r in out:
        if not r.get("assertions") and isinstance(r.get("position"), list) and len(r["position"]) == 2:
            detected = _detect_assertions_from_context(
                r["text"], input_text, r["type"], r["position"][0]
            )
            r["assertions"] = sorted(set(detected))
    return out


def _is_overlap_dup(
    norm_text: str,
    etype: str,
    cur_start: int,
    cur_end: int,
    seen_entities: list[tuple[str, str, list[int]]],
) -> tuple[bool, list[int]]:
    """Check overlap dedup theo R10 STRICT + R22.

    Returns:
        (is_duplicate, indices_to_remove_from_seen):
          - is_duplicate=True: drop current entity
          - indices_to_remove: indices of seen entities to remove (current is longer)
    """
    to_remove: list[int] = []
    for idx, (s_text, s_type, s_pos) in enumerate(seen_entities):
        s_start, s_end = s_pos
        # Same exact span → drop current bất kể type/text để tránh 2 entity chiếm cùng 1 span
        if cur_start == s_start and cur_end == s_end:
            return True, []
        if s_type != etype or s_text != norm_text:
            continue
        # Overlap → keep longer span
        if max(cur_start, s_start) < min(cur_end, s_end):
            ex_len = s_end - s_start
            cur_len = cur_end - cur_start
            if ex_len >= cur_len:
                return True, []  # existing longer → drop current
            to_remove.append(idx)  # current longer → mark existing for removal
    return False, to_remove


_NORMAL_FINDING_PATTERNS = re.compile(
    r"""(?x)
    \bbình\s+thường\b |
    \b(?:âm|dương)\s+tính\b |
    ^(?:mri|ct|siêu\s+âm|x-quang|pcr|điện\s+tâm\s+đồ)(?::|\s).+ |
    \bảo\s+giác\s+bình\s+thường\b |
    \bkhông\s+ghi\s+nhận\s+(?:gì\s+)?(?:bất\s+thường|đáng\s+chú\s+ý)\b |
    \bkhông\s+thấy\s+tình\s+trạng\b
    """,
    re.IGNORECASE | re.UNICODE,
)


def _is_normal_finding_misclassified_as_diagnosis(text: str) -> bool:
    """Kiểm tra nếu LLM gán nhầm kết quả xét nghiệm/sinh hiệu bình thường thành CHẨN_ĐOÁN."""
    norm = text.lower().strip()
    if any(disease_kw in norm for disease_kw in ("viêm gan", "covid", "hiv", "sốt xuất huyết", "cúm")):
        return False
    return bool(_NORMAL_FINDING_PATTERNS.search(norm))


def _build_entity_record(
    text: str,
    etype: str,
    pos: list[int],
    ent: dict[str, Any],
) -> dict[str, Any]:
    """Build 1 record dict với assertions cleaned."""
    if etype == "CHẨN_ĐOÁN" and _is_normal_finding_misclassified_as_diagnosis(text):
        logger.debug("Auto-correct misclassified finding '%s' from CHẨN_ĐOÁN to KẾT_QUẢ_XÉT_NGHIỆM", text)
        etype = "KẾT_QUẢ_XÉT_NGHIỆM"
    assertions = _normalize_assertions_list(ent.get("assertions", []))
    # R27.4: TÊN_XÉT_NGHIỆM không bao giờ isNegated nếu kết quả bình thường
    if etype == "TÊN_XÉT_NGHIỆM" and "isNegated" in assertions:
        if not text.lower().startswith(("không ", "chưa ")):
            assertions = [a for a in assertions if a != "isNegated"]
    return {
        "text": text,
        "type": etype,
        "position": [int(pos[0]), int(pos[1])],
        "assertions": assertions,
        "candidates": [],
    }


def _attach_candidates(
    record: dict[str, Any],
    text: str,
    etype: str,
    ent: dict[str, Any],
    validated: list[dict[str, Any]],
    retriever: RxNormRetriever,
    icd_retriever: Optional[ICDRetriever],
    input_text: str = "",  # R29: passed in for non-treatment context check
) -> None:
    """Gán candidates cho record theo type (RxNorm cho THUỐC, ICD cho CHẨN_ĐOÁN,
    common ICD cho TRIỆU_CHỨNG).

    Mutates `record["candidates"]` in-place.
    R37: TRIỆU_CHỨNG cũng attach ICD nếu match common_symptom_map.
    """
    if etype == "CHẨN_ĐOÁN" or record.get("type") == "CHẨN_ĐOÁN":
        if _is_normal_finding_misclassified_as_diagnosis(text):
            record["type"] = "KẾT_QUẢ_XÉT_NGHIỆM"
            record["candidates"] = []
            return
    if etype == "THUỐC" and retriever is not None:
        # R34 FIX (2026-07-13): Strip drug-class prefix nếu có drug attached
        # (vd "Kháng sinh Cefepim" → "Cefepim"). Sau đó check class.
        stripped_text = _strip_drug_class_prefix(text)
        # Nếu strip trả None → pure class term (vd "kháng sinh") → skip
        if stripped_text is None:
            record["candidates"] = []
            return
        # Nếu stripped != original → có drug name attached → dùng stripped text
        if stripped_text != text:
            text = stripped_text
        # Nếu stripped == original → có thể là pure drug name HOẶC pure class term.
        # Check pure class term với INN guard (tránh "urea", "creatinine" trong whitelist).
        elif _is_generic_drug_class(text):
            record["candidates"] = []
            return
        # R29 (2026-07-13 spec round 2): Non-treatment context (resistance, lab token)
        # → SKIP lookup, return []. "kháng vancomycin" không phải Rx order.
        if _is_non_treatment_drug_context(text, input_text):
            record["candidates"] = []
            return
        drug_query = sanitize_drug_text(text)
        if drug_query:
            try:
                codes = retriever.lookup(drug_query)
                record["candidates"] = list(codes) if codes else []
            except Exception as exc:
                logger.warning("RxNorm lookup fail for '%s': %s", text, exc)
    elif etype == "CHẨN_ĐOÁN" and icd_retriever is not None:
        # ICD lookup cần other_entities context (drugs/symptoms nearby)
        other_ents = [
            e for e in validated
            if e.get("text", "").strip() and e is not ent
        ]
        try:
            codes = icd_retriever.lookup(text, other_entities=other_ents, entity_type=etype)
            record["candidates"] = list(codes) if codes else []
            # R37 (2026-07-19): Deterministic ICD rule expansion (organism/side/lobe).
            # Adds specific subcodes based on text qualifiers WITHOUT LLM.
            record["candidates"] = _apply_deterministic_icd_rules(
                text, record["candidates"]
            )
        except Exception as exc:
            logger.warning("ICD lookup fail for '%s' (%s): %s", text, etype, exc)
    elif etype == "TRIỆU_CHỨNG":
        record["candidates"] = []


# R37 (2026-07-15): Cache common symptom → ICD lookup
_SYMPTOM_ICD_CACHE: dict[str, str] | None = None


def _get_symptom_icd_lookup(text: str) -> str | None:
    """Lookup common symptom → ICD code. Returns None nếu không match."""
    global _SYMPTOM_ICD_CACHE
    if _SYMPTOM_ICD_CACHE is None:
        path = Path(__file__).resolve().parents[1] / "data" / "symptom_icd_map.json"
        if path.exists():
            try:
                cfg = json.loads(path.read_text(encoding="utf-8"))
                _SYMPTOM_ICD_CACHE = {k.lower().strip(): v for k, v in cfg.get("_vn_symptom_to_icd", {}).items()}
            except Exception:
                _SYMPTOM_ICD_CACHE = {}
        else:
            _SYMPTOM_ICD_CACHE = {}
    return _SYMPTOM_ICD_CACHE.get(text.lower().strip())


# ==============================================================================
# R37 (2026-07-16): ICD/RxNorm code validators cho Stage 3 LLM output
# ==============================================================================
# Stage 3 LLM có thể "hallucinate" codes không tồn tại trong ICD-10 / RxNorm index.
# Các helpers này validate LLM-suggested codes TRƯỚC khi attach vào entity.

_ICD_CODE_INDEX: set[str] | None = None
_RXNORM_CODE_INDEX: set[str] | None = None


def _load_icd_index() -> set[str]:
    """Lazy-load ICD code index từ ICDRetriever (cache sau lần đầu)."""
    global _ICD_CODE_INDEX
    if _ICD_CODE_INDEX is None:
        try:
            from src.icd_rag import ICDRetriever
            icd = ICDRetriever()
            _ICD_CODE_INDEX = set(icd._code_to_desc.keys()) if hasattr(icd, '_code_to_desc') else set()
            logger.info("[R37] Loaded ICD index: %d codes", len(_ICD_CODE_INDEX))
        except Exception as exc:
            logger.warning("[R37] Failed to load ICD index: %s", exc)
            _ICD_CODE_INDEX = set()
    return _ICD_CODE_INDEX


def _load_rxnorm_index() -> set[str]:
    """Lazy-load RxNorm rxcui code index từ RxNormRetriever (cache sau lần đầu)."""
    global _RXNORM_CODE_INDEX
    if _RXNORM_CODE_INDEX is None:
        try:
            from src.rxnorm_rag import RxNormRetriever
            retriever = RxNormRetriever()
            _RXNORM_CODE_INDEX = set(retriever.index.rxcuis) if hasattr(retriever.index, 'rxcuis') else set()
            logger.info("[R37] Loaded RxNorm index: %d codes", len(_RXNORM_CODE_INDEX))
        except Exception as exc:
            logger.warning("[R37] Failed to load RxNorm index: %s", exc)
            _RXNORM_CODE_INDEX = set()
    return _RXNORM_CODE_INDEX


def _validate_icd_code(code: str) -> bool:
    """Check ICD code tồn tại trong ICD-10 index. Allow parent codes (3-char)."""
    if not code or not isinstance(code, str):
        return False
    code = code.strip().upper()
    # Accept well-formed codes (A00-Z99, with optional .X or .XX)
    if not re.match(r"^[A-Z]\d{2}(\.\d{1,2})?$", code):
        return False
    valid_set = _load_icd_index()
    # Direct match
    if code in valid_set:
        return True
    # Try parent (3-char) if specific not found
    parent = code.split(".")[0]
    if parent in valid_set:
        return True
    # If index is empty (init failed), accept ICD-format-matching codes
    if not valid_set:
        return True
    return False


def _validate_rxnorm_code(code: str) -> bool:
    """Check RxNorm rxcui code (numeric string)."""
    if not code or not isinstance(code, str):
        return False
    code = code.strip()
    # RxNorm codes are numeric (digits only)
    if not code.isdigit():
        return False
    valid_set = _load_rxnorm_index()
    # If index loaded, check it
    if valid_set and code not in valid_set:
        return False
    return True


def _validate_candidates_for_type(
    candidates: list[str], etype: str
) -> list[str]:
    """R37: Filter candidates list by entity type, validating each code.

    Args:
        candidates: raw candidate codes from LLM
        etype: 'THUỐC' or 'CHẨN_ĐOÁN' or others

    Returns:
        Filtered list of valid codes (max 5).
        If input empty/invalid → returns empty list.
    """
    if not candidates or not isinstance(candidates, list):
        return []

    out: list[str] = []
    for raw in candidates[:5]:  # hard cap 5
        if not isinstance(raw, str):
            continue
        code = raw.strip()
        if not code:
            continue
        if etype == "THUỐC":
            if _validate_rxnorm_code(code):
                out.append(code)
        elif etype == "CHẨN_ĐOÁN":
            if _validate_icd_code(code):
                out.append(code)
        # Other types: skip (TRIỆU_CHỨNG/TÊN_XN/KQ_XN not in Stage 3 scope)
    return out


# R37 (2026-07-19): Deterministic ICD rule expander — apply context-aware rules
# (organism → subcode, anatomical side, severity grade) WITHOUT relying on LLM.
# Goal: "đủ + chính xác + ngữ cảnh" — deterministic, deterministic, deterministic.
def _apply_deterministic_icd_rules(text: str, candidates: list[str]) -> list[str]:
    """Apply deterministic ICD-10 rules based on text qualifiers.

    Rules (from STAGE3_PROMPT ICD-10 SPECIFICITY RULES):
    1. ETIOLOGY: organism → specific A0x subcode (REPLACE parent)
    2. ANATOMICAL SIDE: trái/phải → add side-specific subcode
    3. ANATOMICAL DETAIL: thùy trên/dưới → C34.x
    4. SEVERITY: độ 1/2/3, NYHA → keep qualifier
    5. MI LOCATION: thành trước/dưới → I21.0/I21.1

    Args:
        text: disease name từ entity
        candidates: ICD codes hiện có (từ RAG hoặc Stage 3)

    Returns:
        List candidates sau khi áp dụng rules. Có thể REPLACE parent với specific.
        Max 5 codes.
    """
    if not candidates or not text:
        return candidates[:5]

    text_lower = text.lower()
    out = list(candidates)
    seen = set(out)

    # Rule 1: ETIOLOGY — specific organism → REPLACE parent with subcode
    # Shigella dysenteriae → A03.0 (specific), not generic A03
    if "shigella dysenteriae" in text_lower:
        if "A03" in seen or any(c.startswith("A03") for c in out):
            # Remove generic A03, add A03.0
            out = [c for c in out if not c.startswith("A03") or c == "A03.0"]
            if "A03.0" not in out:
                out.append("A03.0")
            seen = set(out)
    elif "shigella" in text_lower:
        if "A03" in seen and "A03.8" not in out:  # Shigella unspecified → A03.8
            pass  # Keep parent A03

    if "salmonella" in text_lower:
        if any(c.startswith("A02") for c in out):
            out.append("A02.0") if "A02.0" not in out else None
            seen = set(out)

    # Rule 2: ANATOMICAL SIDE — trái/phải → add side-specific J18.x (lung)
    if "trái" in text_lower:
        if any(c.startswith("J18") for c in out) and "J18.2" not in seen:
            out.append("J18.2")  # left lung
            seen.add("J18.2")
    elif "phải" in text_lower:
        if any(c.startswith("J18") for c in out) and "J18.1" not in seen:
            out.append("J18.1")  # right lung
            seen.add("J18.1")

    # Rule 3: ANATOMICAL DETAIL — thùy trên/dưới → C34.x (lung cancer)
    if "thùy trên" in text_lower:
        if any(c.startswith("C34") for c in out) and "C34.1" not in seen:
            out.append("C34.1")
            seen.add("C34.1")
    elif "thùy dưới" in text_lower:
        if any(c.startswith("C34") for c in out) and "C34.3" not in seen:
            out.append("C34.3")
            seen.add("C34.3")
    elif "thùy giữa" in text_lower:
        if any(c.startswith("C34") for c in out) and "C34.2" not in seen:
            out.append("C34.2")
            seen.add("C34.2")

    # Rule 4: MI LOCATION — thành trước/dưới → I21.0/I21.1
    if "nhồi máu cơ tim" in text_lower:
        if "thành trước" in text_lower and "I21.0" not in seen:
            out = [c for c in out if c != "I21.9"]  # Remove unspecified
            out.append("I21.0")
            seen = set(out)
        elif "thành dưới" in text_lower and "I21.1" not in seen:
            out = [c for c in out if c != "I21.9"]
            out.append("I21.1")
            seen = set(out)

    return out[:5]


def _emit_entity_record(
    ent: dict[str, Any],
    input_text: str,
    validated: list[dict[str, Any]],
    retriever: RxNormRetriever,
    icd_retriever: Optional[ICDRetriever],
    seen_test_names: set[str],
    seen_entities: list[tuple[str, str, list[int]]],
    skip_attach: bool = False,
) -> dict[str, Any] | None:
    """Process 1 entity: clean, dedup, build record, attach candidates.

    Returns:
        Record dict nếu entity pass tất cả filter, None nếu bị drop.
    """
    text = str(ent.get("text", "")).strip()
    etype = ent.get("type", "")
    if not text or etype not in (
        "THUỐC", "CHẨN_ĐOÁN", "TRIỆU_CHỨNG", "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"
    ):
        return None

    # R27.7: clean entity text trước khi emit (strip modifiers, drop noise/duration)
    cleaned_text = _clean_entity_text(text, etype)
    if cleaned_text is None:
        return None
    if cleaned_text != text:
        text = cleaned_text
        ent["text"] = text
        old_pos = ent.get("position", [0, 0])
        old_start = int(old_pos[0]) if isinstance(old_pos, list) and len(old_pos) == 2 else 0
        new_pos = _find_span(input_text, text, start=max(0, old_start - 10))
        if new_pos is None and old_start > 0:
            new_pos = _find_span(input_text, text, start=0)
        if new_pos is not None:
            ent["position"] = list(new_pos)

    # Validate position
    pos = ent.get("position", [0, 0])
    if not (isinstance(pos, list) and len(pos) == 2 and all(isinstance(p, int) for p in pos)):
        pos = [0, 0]
    cur_start, cur_end = int(pos[0]), int(pos[1])

    # R37 (2026-07-16): Tách merged test_name+value (vd 'ast 421' → ast + 421).
    # Phải làm TRƯỚC R31 retype vì text sẽ được tách thành 2 entities.
    pos = ent.get("position", [0, 0])
    if (
        isinstance(pos, list)
        and len(pos) == 2
        and all(isinstance(p, int) for p in pos)
        and etype == "KẾT_QUẢ_XÉT_NGHIỆM"
    ):
        split_result = _split_test_name_and_value(text, etype, input_text, pos)
        if split_result and len(split_result) == 2:
            # Trả về marker tuple (None, split_entities) để caller xử lý.
            # Thực tế sẽ xử lý trong assemble_record vì _emit_entity_record
            # chỉ trả về 1 record. → Tạo 2 records riêng biệt qua loop.
            logger.debug(
                "[R37] Split merged test+value: '%s' → ['%s', '%s']",
                text, split_result[0]['text'], split_result[1]['text']
            )
            # Trả về dict đặc biệt với key '_split' để caller biết.
            return {
                "_split": True,
                "entities": split_result,
                "orig_text": text,
            }

    # R31: Auto-retype dựa trên text patterns (abnormal findings → CHẨN_ĐOÁN, procedures → TÊN_XN)
    new_etype = _retype_entity(text, etype)
    if new_etype != etype:
        etype = new_etype
        ent["type"] = etype

    if etype == "CHẨN_ĐOÁN" and _is_normal_finding_misclassified_as_diagnosis(text):
        etype = "KẾT_QUẢ_XÉT_NGHIỆM"
        ent["type"] = etype

    norm_text = text.lower().strip()

    # Dedup check (R10 STRICT + OVERLAP DEDUP cho tất cả các loại, kể cả TÊN_XÉT_NGHIỆM nếu khác position)
    is_duplicate, to_remove = _is_overlap_dup(
        norm_text, etype, cur_start, cur_end, seen_entities,
    )
    if is_duplicate:
        return None
    for idx in reversed(to_remove):
        seen_entities.pop(idx)
    seen_entities.append((norm_text, etype, [cur_start, cur_end]))

    # Auto-detect assertions from context if missing or incomplete
    detected = _detect_assertions_from_context(text, input_text, etype, cur_start)
    if detected:
        existing = set(_normalize_assertions_list(ent.get("assertions", [])))
        for d in detected:
            if d not in existing:
                existing.add(d)
        ent["assertions"] = sorted(existing)

    # Sanity check isHistorical: Giữ isHistorical nếu section là tien_su hoặc câu lân cận có từ khóa tiền sử rõ ràng
    if "isHistorical" in ent.get("assertions", []):
        is_sec_hist = (_find_current_section(input_text, cur_start) == "tien_su")
        near_hist_slice = input_text.lower()[max(0, cur_start - 100):min(len(input_text), cur_start + len(text) + 60)]
        hist_trigs = ("tiền sử", "cách đây", "từng bị", "từng điều trị", "đã từng", "bệnh cũ", "nhồi máu cũ", "năm trước", "tháng trước", "tuần trước", "tại nhà", "trước nhập viện", "đang điều trị trước", "tiền căn", "bệnh sử cũ", "thuốc đang dùng trước")
        if not (is_sec_hist or any(re.search(r'\b' + re.escape(h) + r'\b', near_hist_slice, re.UNICODE) for h in hist_trigs)):
            ent["assertions"] = [a for a in ent.get("assertions", []) if a != "isHistorical"]

    # Build record + attach candidates
    record = _build_entity_record(text, etype, pos, ent)
    if not skip_attach:
        _attach_candidates(
            record, text, etype, ent, validated,
            retriever, icd_retriever, input_text,  # R29: thread input_text through for context check
        )
    return record


# ---------------------------------------------------------------------- #
# Output validation
# ---------------------------------------------------------------------- #


def validate_output(payload: list[dict[str, Any]]) -> bool:
    """Schema check cuối cùng. Auto-fix missing fields (LLM hay quên candidates)."""
    if not isinstance(payload, list):
        return False
    # Defense: MỌI entity phải có 5 fields theo spec — auto-fill nếu thiếu
    for ent in payload:
        if isinstance(ent, dict):
            ent.setdefault("candidates", [])
            ent["assertions"] = _normalize_assertions_list(ent.get("assertions", []))
    try:
        from jsonschema import validate  # type: ignore

        from .prompts import OUTPUT_SCHEMA

        validate(instance=payload, schema=OUTPUT_SCHEMA)
        return True
    except Exception as exc:

        logger.warning("Validation lỗi: %s", exc)
        return False


def write_output(path: Path, payload: list[dict[str, Any]]) -> None:
    """Write output JSON theo gold NER format chuẩn.

    R37 (2026-07-15): Output phải match gold format về FIELD ORDER.
    Gold mẫu (có candidates):
        {"text": "...", "type": "THUỐC",
         "candidates": [...], "assertions": [...], "position": [X, Y]}
    Gold mẫu (không candidates):
        {"text": "...", "type": "TRIỆU_CHỨNG",
         "assertions": [], "position": [X, Y]}  ← candidates field OMITTED

    KEY_ORDER: text → type → candidates (optional) → assertions → position.
    assertions giữ nguyên (kể cả khi []), candidates OMIT nếu rỗng.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    KEY_ORDER = ("text", "type", "candidates", "assertions", "position")
    SKIP_EMPTY = ("candidates",)  # OMIT khi list rỗng

    cleaned: list[dict[str, Any]] = []
    for ent in payload:
        if not isinstance(ent, dict):
            continue
        ordered: dict[str, Any] = {}
        for k in KEY_ORDER:
            if k in ent:
                v = ent[k]
                if k in SKIP_EMPTY and isinstance(v, list) and len(v) == 0:
                    continue  # OMIT candidates khi rỗng
                ordered[k] = v
        # Preserve any other keys (defensive) theo thứ tự KEY_ORDER + extras
        for k in ent:
            if k not in ordered and k not in KEY_ORDER:
                ordered[k] = ent[k]
        cleaned.append(ordered)

    with path.open("w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)


# R37 (2026-07-15): Normalize dose whitespace trong THUỐC text.
# Gold format chuẩn: "10 mg" (có space giữa digit và unit).
# LLM hay output: "10mg" (không space) → WER tăng do mismatch character-by-character.
_DOSE_WHITESPACE_RE = re.compile(
    r"(\d)(mg|ml|g|mcg|iu|meq|µg|ug|ng|kg|mmol)\b",
    re.IGNORECASE,
)


def _normalize_dose_whitespace(text: str) -> str:
    """Thêm space giữa digit và unit nếu thiếu: '10mg' → '10 mg'.

    KHÔNG đụng vào cases đã đúng ('10 mg' → giữ nguyên).
    Return normalized text. Caller phải re-find position sau khi apply.
    """
    if not text:
        return text
    return _DOSE_WHITESPACE_RE.sub(r"\1 \2", text)


# ---------------------------------------------------------------------- #
# Self-test
# ---------------------------------------------------------------------- #

if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.DEBUG)
    sample_text = (
        "Bệnh nhân dùng aspirin 81 mg po daily trước nhập viện điều trị nhức đầu."
    )
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
