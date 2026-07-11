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

from src.icd_rag import ICDRetriever
from src.rxnorm_rag import RxNormRetriever

# Đảm bảo có thể chạy trực tiếp `python src/postprocess.py`
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------- #
# Input preprocessing — clean/truncate input trước khi gửi LLM
# ---------------------------------------------------------------------- #

# Ngưỡng tối đa cho input (chars). Vượt ngưỡng sẽ truncate để tránh overflow.
# SYSTEM_PROMPT (~4059 tokens) + 4000 chars input (~1000 tokens) + max_tokens output (4096)
# = ~9155 tokens. Vừa với Ollama num_ctx=8192 nếu giảm input xuống <3000 chars.
_INPUT_MAX_CHARS = 4000

# Header đánh dấu đã truncate (LLM biết phần nào bị cắt)
_TRUNCATION_MARKER = "\n\n[... Đã rút gọn phần giữa để vừa context window ...]\n\n"


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
    """Tìm vị trí đầu tiên của snippet trong text (từ start); trả [start, end) hoặc None.

    Args:
        text: full input text.
        snippet: text cần tìm.
        start: vị trí bắt đầu tìm (default 0).
    """
    if not snippet:
        return None
    idx = text.find(snippet, start)
    if idx >= 0:
        return idx, idx + len(snippet)
    # Fallback: lowercase (từ start)
    idx = text.lower().find(snippet.lower(), start)
    if idx >= 0:
        return idx, idx + len(snippet)
    # Fallback: bỏ khoảng trắng thừa ở hai đầu
    stripped = snippet.strip()
    if stripped != snippet:
        idx = text.find(stripped, start)
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
    if a == b or a in b or b in a:
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
    if not tokens_a or not tokens_b:
        return False
    if tokens_a.issubset(tokens_b) or tokens_b.issubset(tokens_a):
        return True
    jaccard = len(tokens_a & tokens_b) / max(len(tokens_a | tokens_b), 1)
    return jaccard >= 0.80


def dedupe_entities(entities: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Bỏ trùng entities dựa trên (text, type, position overlap) - R10 STRICT + R22 OVERLAP (2026-07-10).

    R10 STRICT (đổi từ R10 LOOSE theo user feedback 2026-07-09):
    - Cùng text + type + cùng position → 1 entity (R22 dedup)
    - Cùng text + type + khác position → giữ cả N entities (R10 STRICT theo position)
    - **MỚI 2026-07-10 — OVERLAP DEDUP**: cùng text + type + positions OVERLAP (intersect)
      → giữ span DÀI HƠN, drop span ngắn hơn (vd [97,110] và [102,110] cùng "tăng huyết áp"
      → giữ [97,110], drop [102,110] vì span thứ 2 nằm trong span thứ 1).
    - Áp dụng cho TẤT CẢ loại (THUỐC, CHẨN_ĐOÁN, TRIỆU_CHỨNG, TÊN_XN, KQ_XN).

    Lý do R10 STRICT (đổi từ LOOSE):
    - LLM có position → extract đầy đủ duplicate ở các vị trí khác nhau
    - Postprocess giữ N entities để khớp với ground truth (48-51 entities/file)
    - Trade-off: tăng recall tuyệt đối, có thể tăng false positive nếu LLM extract duplicate giả

    Lý do OVERLAP DEDUP (mới 2026-07-10):
    - LLM 7B hay output duplicate VỚI POSITION LỆCH vài ký tự (vd "tăng huyết áp" [97,110] vs [102,110])
    - Hai span overlap nhưng khác start → start-only dedup miss cả hai
    - Ground truth KHÔNG có duplicate trùng text ở vị trí overlap
    - Fix: detect overlap, giữ span dài hơn (chứa span kia)
    """
    out: list[dict[str, Any]] = []
    # Track: (text_lower, type, [start, end]) đã thấy
    # Khi check mới: nếu cùng text+type VÀ (cùng start HOẶC overlap) → drop span ngắn hơn.

    # Sort theo start ASC, length DESC (span dài xử lý trước)
    def _sort_key(e: dict[str, Any]) -> tuple[int, int]:
        pos = e.get("position", [0, 0])
        if isinstance(pos, list) and len(pos) >= 2:
            try:
                start = int(pos[0])
                end = int(pos[1])
            except (TypeError, ValueError):
                return (0, 0)
            return (start, -(end - start))  # start asc, length desc
        return (0, 0)

    sorted_ents = sorted(
        [e for e in entities if e.get("text")],
        key=_sort_key,
    )

    for ent in sorted_ents:
        etype = ent.get("type", "")
        text = str(ent.get("text", "")).strip()
        pos = ent.get("position", [0, 0])
        if not (isinstance(pos, list) and len(pos) == 2 and all(isinstance(p, int) for p in pos)):
            continue
        start, end = int(pos[0]), int(pos[1])
        if start < 0 or end <= start:
            continue

        # Check overlap với existing entities cùng text+type
        is_duplicate = False
        to_remove: list[int] = []
        for idx, existing in enumerate(out):
            if existing.get("type", "") != etype:
                continue
            ex_text = str(existing.get("text", "")).strip()
            ex_pos = existing.get("position", [0, 0])
            if not (isinstance(ex_pos, list) and len(ex_pos) == 2):
                continue
            e_start, e_end = int(ex_pos[0]), int(ex_pos[1])

            is_exact_text = (ex_text.lower() == text.lower())
            is_pos_overlap = (max(start, e_start) < min(end, e_end))

            if not is_exact_text:
                if not is_pos_overlap or not _is_semantic_overlap(ex_text, text):
                    continue

            # Same exact span → drop current (R22)
            if start == e_start and end == e_end:
                is_duplicate = True
                logger.debug(
                    "R22 dedup: drop duplicate exact (text=%r, type=%r, pos=[%d,%d])",
                    text, etype, start, end,
                )
                break

            # OVERLAP check: max(start, e_start) < min(end, e_end) → intersect
            if is_pos_overlap:
                ex_len = e_end - e_start
                cur_len = end - start
                if ex_len >= cur_len:
                    # Existing span dài hơn hoặc bằng → drop current
                    is_duplicate = True
                    logger.debug(
                        "R10 overlap dedup: drop '%s' [%d,%d] (existing [%d,%d] longer/equal)",
                        text, start, end, e_start, e_end,
                    )
                    break
                else:
                    # Current span dài hơn → remove existing, add current
                    to_remove.append(idx)
                    logger.debug(
                        "R10 overlap dedup: replace '%s' [%d,%d] with longer [%d,%d]",
                        text, e_start, e_end, start, end,
                    )

        # Remove existing entities that current overlaps AND is longer
        for idx in reversed(to_remove):
            out.pop(idx)

        if not is_duplicate:
            out.append(ent)

    out.sort(key=lambda e: e["position"][0])
    return out


# ---------------------------------------------------------------------- #
# Drug text sanitization (R4 + R18)
# ---------------------------------------------------------------------- #

_DRUG_NAME_BAD_PATTERNS = re.compile(
    r"^(thuốc|drug|medication|thuoc)\s*$", re.IGNORECASE
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


# Patterns chỉ định isFamily (dùng để verify)
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

    # isHistorical: CHỉ áp dụng nếu entity trong section "Tiền sử"
    if is_in_tien_su:
        found.append("isHistorical")

    # isFamily: cần word boundary để tránh "ông " match "không "
    family_patterns = [
        r"\bbố\s+([bệe]nh\s+)?nh[âa]n",       # bố (bệnh) nhân
        r"\bm[ẹe]\s+([bệe]nh\s+)?nh[âa]n",      # mẹ/me (bệnh) nhân
        r"\bcha\s+([bệe]nh\s+)?nh[âa]n",      # cha (bệnh) nhân
        r"\banh\s+(trai\s+)?[bệe]nh\s+nh[âa]n",
        r"\bch[ịi]\s+(g[áa]i\s+)?[bệe]nh\s+nh[âa]n",
        r"\bem\s+(trai\s+|g[áa]i\s+)?[bệe]nh\s+nh[âa]n",
        r"\bcon\s+(trai\s+|g[áa]i\s+)?[bệe]nh\s+nh[âa]n",
        r"\bông\s+([bệe]nh\s+)?nh[âa]n",       # ông (bệnh) nhân — word boundary!
        r"\bbà\s+([bệe]nh\s+)?nh[âa]n",        # bà (bệnh) nhân
        r"\bti[eề]n\s+s[ử]?\s*gia\s+[đd][ìi]nh",  # tiền sử gia đình
    ]
    # Window 200 chars quanh entity cho family
    family_win_start = max(0, pos - 200)
    family_win_end = min(len(input_text), pos + len(entity_text) + 100)
    family_window = text_lower[family_win_start:family_win_end]
    for pat in family_patterns:
        if re.search(pat, family_window):
            found.append("isFamily")
            break

    # isNegated: check "không", "chưa", "âm tính", "loại trừ" trong window trước entity.
    # Lưu ý: "không" có thể nằm sát entity (vd "không sốt" → pre_window kết thúc bằng "khô").
    near = text_lower[max(0, pos - 35):pos + min(len(entity_text), 15)]  # rộng hơn để bắt đủ ngữ cảnh phủ định
    found_negated = False
    neg_phrases = (
        "không thấy", "chưa thấy", "chưa có dấu hiệu", "loại trừ",
        "không có", "chưa có", "không phát hiện", "âm tính",
        "không ghi nhận", "chưa ghi nhận", "không sốt", "không ho",
        "không", "chưa"
    )
    for neg in neg_phrases:
        if neg in near:
            if re.search(r'\b' + re.escape(neg) + r'\b', near):
                found_negated = True
                break
    if found_negated and "isNegated" not in found:
        found.append("isNegated")

    return found[:3]  # max 3 theo spec


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
    text_lower = input_text.lower()
    pos = max(0, entity_pos)

    # Section patterns (ưu tiên cao hơn cho cụm từ cụ thể)
    section_patterns = [
        # === TIỀN SỬ (isHistorical=True) ===
        # Generic "Tiền sử" (không cần số prefix) - PRIORITY CAO để detect section tiền sử ở mọi vị trí
        (r"\btiền sử bệnh\s*nội khoa", "tien_su", 95),
        (r"\btiền sử bệnh\s*ngoại khoa", "tien_su", 95),
        (r"\btiền sử phẫu thuật", "tien_su", 92),
        (r"\btiền sử thủ thuật", "tien_su", 92),
        (r"\btiền sử gia đình", "tien_su", 92),
        (r"\btiền sử dị ứng", "tien_su", 92),
        (r"\btiền sử xã hội", "tien_su", 92),
        (r"\btiền sử bệnh", "tien_su", 88),
        (r"\btiền sử", "tien_su", 85),
        (r"\bbệnh sử", "tien_su", 85),
        (r"\btiền căn", "tien_su", 85),
        (r"thuốc trước khi nhập viện", "tien_su", 95),
        (r"thuốc trước đây", "tien_su", 95),
        (r"thuốc đang dùng", "tien_su", 90),
        (r"thuốc ra viện", "hien_tai", 90),  # Thuốc ra viện = hiện tại
        (r"đang điều trị tại nhà", "tien_su", 90),
        (r"đang dùng thuốc", "tien_su", 85),
        # Số prefix (ưu tiên cao hơn cho "1. Tiền sử bệnh")
        (r"\d+\.\s*tiền sử bệnh\s*nội khoa", "tien_su", 100),
        (r"\d+\.\s*tiền sử bệnh\s*ngoại khoa", "tien_su", 100),
        (r"\d+\.\s*tiền sử bệnh", "tien_su", 95),
        (r"\d+\.\s*tiền sử phẫu thuật", "tien_su", 95),
        (r"\d+\.\s*tiền sử thủ thuật", "tien_su", 95),
        (r"\d+\.\s*tiền sử gia đình", "tien_su", 95),
        (r"\d+\.\s*tiền sử dị ứng", "tien_su", 95),
        (r"\d+\.\s*tiền sử xã hội", "tien_su", 95),
        (r"\d+\.\s*tiền sử", "tien_su", 80),
        (r"\d+\.\s*bệnh sử", "tien_su", 90),
        (r"\d+\.\s*tiền căn", "tien_su", 90),
        # === HIỆN TẠI (isHistorical=False) ===
        (r"\d+\.\s*tiền sử bệnh\s*hiện tại", "hien_tai", 100),
        (r"\d+\.\s*bệnh sử hiện tại", "hien_tai", 100),
        (r"\btiền sử bệnh\s*hiện tại", "hien_tai", 95),
        (r"\bbệnh sử hiện tại", "hien_tai", 95),
        (r"lý do nhập viện", "hien_tai", 95),
        (r"lý do vào viện", "hien_tai", 95),
        (r"lý do khám", "hien_tai", 95),
        (r"triệu chứng hiện tại", "hien_tai", 90),
        (r"triệu chứng cơ năng", "hien_tai", 90),
        (r"triệu chứng thực thể", "hien_tai", 90),
        (r"diễn biến bệnh", "hien_tai", 85),
        (r"diễn tiến", "hien_tai", 80),
        (r"quá trình bệnh", "hien_tai", 80),
        (r"\bhiện tại", "hien_tai", 70),
        # === ĐÁNH GIÁ (isHistorical=False) ===
        (r"\d+\.\s*đánh giá tại bệnh viện", "danh_gia", 100),
        (r"\d+\.\s*đánh giá", "danh_gia", 90),
        (r"đánh giá tại bệnh viện", "danh_gia", 95),
        (r"\bđánh giá", "danh_gia", 80),
        (r"khám lúc vào viện", "danh_gia", 95),
        (r"khám vào viện", "danh_gia", 95),
        (r"khám tại viện", "danh_gia", 95),
        (r"khám hiện tại", "danh_gia", 90),
        (r"\bkhám:", "danh_gia", 80),
        (r"\bkhám\b", "danh_gia", 70),
        (r"xét nghiệm", "danh_gia", 80),
        (r"\bcls\b", "danh_gia", 80),
        (r"cận lâm sàng", "danh_gia", 80),
        (r"kết quả xét nghiệm", "danh_gia", 85),
        (r"chẩn đoán hình ảnh", "danh_gia", 85),
        (r"hình ảnh", "danh_gia", 80),
        (r"\bđiều trị", "danh_gia", 75),
        (r"phác đồ điều trị", "danh_gia", 80),
        (r"hướng xử trí", "danh_gia", 80),
        (r"\btheo dõi", "danh_gia", 75),
        (r"tái khám", "danh_gia", 70),
        (r"ra viện", "danh_gia", 70),
        (r"tóm tắt", "danh_gia", 70),
        (r"kết luận", "danh_gia", 70),
    ]

    # Tìm tất cả matches TRƯỚC entity_pos
    matches = []  # (start_pos, section_id, priority)
    for pattern, section_id, priority in section_patterns:
        for m in re.finditer(pattern, text_lower):
            if m.start() < pos:
                matches.append((m.start(), section_id, priority))

    if not matches:
        return ""

    # Sort theo: (khoảng cách đến entity_pos giảm dần, priority giảm dần)
    # → Match gần nhất + priority cao nhất
    matches.sort(key=lambda x: (pos - x[0], -x[2]))

    return matches[0][1]




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
            if not text_j or len(text_j) >= len(text_i):
                continue
            # text_j ngắn hơn text_i: check substring OR semantic overlap when positions intersect
            pos_i = ent_i.get("position", [0, 0])
            pos_j = ent_j.get("position", [0, 0])
            pos_overlap = (isinstance(pos_i, list) and isinstance(pos_j, list) and len(pos_i) == 2 and len(pos_j) == 2 and max(pos_i[0], pos_j[0]) < min(pos_i[1], pos_j[1]))
            if text_j in text_i or (pos_overlap and _is_semantic_overlap(text_j, text_i)):
                drop_indices.add(j)
                logger.debug(
                    "Drop substring/semantic entity '%s' (subset of '%s')",
                    text_j, text_i,
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

        # 1. Lọc lifestyle / social / psych keywords
        if _LIFESTYLE_RE.search(text):
            logger.debug(
                "[%d] Drop lifestyle/social/psych entity '%s' (kw match)",
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
        if etype in ("CHẨN_ĐOÁN", "TRIỆU_CHỨNG") and _PURE_DURATION_RE.match(text):
            logger.debug(
                "[%d] Drop pure duration entity '%s' (%s)",
                _seen_count, text, etype,
            )
            continue

        # 4. Chuẩn hóa assertions: TÊN_XÉT_NGHIỆM không bao giờ bị isNegated nếu kết quả bình thường
        assertions = list(ent.get("assertions", []))
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
    r"^(cảm\s+giác|cảm\s+thấy|thấy|có\s+dấu\s+hiệu|có\s+triệu\s+chứng|"
    r"có\s+cảm\s+giác|nhận\s+thấy|ghi\s+nhận|"
    r"có\s+|bị\s+|xuất\s+hiện\s+|biểu\s+hiện\s+|xảy\s+ra\s+|phát\s+hiện\s+|gặp\s+phải\s+)\s*",
    re.IGNORECASE | re.UNICODE,
)

# Canonical CHẨN_ĐOÁN names chứa "tăng"/"giảm" prefix - KHÔNG strip
_CANONICAL_KEEP_PREFIX = {
    "tăng huyết áp", "tăng đường huyết", "tăng cholesterol",
    "tăng lipid máu", "tăng triglyceride máu",
    "giảm tiểu cầu", "giảm bạch cầu", "giảm dung nạp gắng sức",
    "rối loạn lipid máu", "rối loạn chuyển hóa",
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
    re.compile(r"^khi\s+(?:được\s+)?chuyển\s+(?:vào|tới)\s+\w+", re.IGNORECASE),
    re.compile(r"^khi\s+(?:đến|nhập|vào)\s+(?:khoa|viện)", re.IGNORECASE),
    re.compile(r"^trong\s+quá\s+trình\s+\w+", re.IGNORECASE),
    re.compile(r"^sau\s+khi\s+(?:được\s+)?\w+", re.IGNORECASE),
    re.compile(r"^trước\s+khi\s+(?:được\s+)?\w+", re.IGNORECASE),
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


def _clean_entity_text(text: str, etype: str) -> str | None:
    """Post-fix entity text LLM hay miss (R27.7 mới 2026-07-10).

    Auto-clean các patterns:
    1. Leading verb/qualifier strip ("cảm giác", "tăng", "có", "bị", "xuất hiện", ...)
       → TRỪ canonical names (vd "tăng huyết áp" GIỮ)
    2. Verb prefix trong TÊN_XÉT_NGHIỆM strip ("chụp", "phân tích", "đo", ...)
    3. Parens admin trong THUỐC strip ("(uống trước ăn)" → DROP)
    4. Pure duration DROP (return None → caller drop entity)

    Args:
        text: entity text gốc từ LLM.
        etype: entity type (THUỐC, CHẨN_ĐOÁN, TRIỆU_CHỨNG, ...).

    Returns:
        Cleaned text. None nếu entity nên bị DROP (vd pure duration, noise).
    """
    if not text:
        return text
    original = text
    text_lower = text.strip().lower()

    # === BƯỚC 1: DROP noise patterns (return None để caller drop) ===
    for pattern in _DROP_NOISE_PATTERNS:
        if pattern.match(text_lower):
            logger.debug("Clean: drop noise entity '%s'", original)
            return None

    # === BƯỚC 2: Pure duration → DROP (R28.2) ===
    if etype in ("TRIỆU_CHỨNG", "CHẨN_ĐOÁN"):
        if _PURE_DURATION_ENHANCED_RE.match(text_lower):
            logger.debug("Clean: drop pure duration entity '%s'", original)
            return None

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

        # Strip leading verb/qualifier (regex non-greedy)
        text_new = _LEADING_VERB_QUALIFIER_RE.sub("", text, count=1).strip()

        # === Strip trailing duration / time expression (comprehensive) ===
        # Covers: "trong tuần qua", "trong 3 ngày qua", "kéo dài 30 phút",
        #         "cách 10 ngày trước", "10 năm", "30 phút", etc.
        trailing_duration_patterns = [
            # "trong X (time-unit) (qua|trước|sau)" with optional number
            r"\s+trong\s+\d*\s*(?:giây|phút|giờ|ngày|tuần|tháng|năm)\s*(?:qua|trước|sau)?$",
            # "cách X (time-unit) (trước|sau)"
            r"\s+cách\s+\d+\s*(?:giây|phút|giờ|ngày|tuần|tháng|năm)\s*(?:trước|sau)?$",
            # "kéo dài X (time-unit)"
            r"\s+kéo\s+dài\s+\d*\s*(?:giây|phút|giờ|ngày|tuần|tháng|năm)$",
            # "khởi phát lúc X giờ"
            r"\s+khởi\s+phát\s+lúc\s+\d+\s*(?:giây|phút|giờ|ngày)?$",
            # Standalone "X (time-unit)" at end
            r"\s+\d+\s+(?:giây|phút|giờ|ngày|tuần|tháng|năm)(?:\s+(?:qua|trước|sau))?$",
            # "trong (tuần|ngày|tháng|năm) (qua|trước|sau)" without number
            r"\s+trong\s+(?:tuần|ngày|tháng|năm)\s+(?:qua|trước|sau)$",
        ]
        for pattern in trailing_duration_patterns:
            text_new = re.sub(
                pattern, "", text_new,
                flags=re.IGNORECASE | re.UNICODE,
            ).strip()
            if not text_new:
                break

        if text_new != text and len(text_new) >= 3:
            logger.debug("Clean: strip leading/trailing '%s' → '%s'", text, text_new)
            text = text_new

    return text


# ---------------------------------------------------------------------- #
# _retype_entity — auto-correct entity type dựa trên text patterns (R31 mới)
# ---------------------------------------------------------------------- #

# Abnormal findings trên imaging → CHẨN_ĐOÁN (không phải TRIỆU_CHỨNG/KQ_XN)
_ABNORMAL_FINDING_TO_CHAN_DOAN = re.compile(
    r"^(tràn dịch màng phổi|tràn dịch màng tim|tràn dịch ổ bụng|cổ trướng|"
    r"tràn khí màng phổi|tràn khí trung thất|"
    r"tim to|gan to|lách to|thận to|"
    r"xẹp phổi|tràn khí phổi|giãn phế quản|"
    r"xơ phổi|khí phế thủng|giãn phế nang|"
    r"gan nhiễm mỡ|xơ gan|thoát vị hoành|"
    r"giãn đường mật|tắc nghẽn đường mật|sỏi mật|"
    r"phù phổi|phù não|"
    r"gãy xương \w+|gãy \w+ xương|"
    r"chấn thương sọ não|chấn thương \w+|"
    r"vết thương hở \w+|"
    r"hở van (hai lá|ba lá|động mạch chủ|động mạch phổi|2 lá)|"
    r"hẹp van (hai lá|ba lá|động mạch chủ|động mạch phổi|2 lá)|"
    r"hở van \w+ (nhẹ|vừa|nặng|mild|moderate|severe)|"
    r"hẹp van \w+ (nhẹ|vừa|nặng|mild|moderate|severe)|"
    r"mất vận động vùng đỉnh|rối loạn vận động vùng đỉnh|"
    r"giãn \w+ buồng tim|"
    r"u ác tính|khối u ác tính|khối u \w+|"
    r"viêm \w+ (nặng|cấp|mạn))$",
    re.IGNORECASE | re.UNICODE,
)

# Procedures/surgeries → TÊN_XÉT_NGHIỆM (không phải THUỐC)
_PROCEDURE_TO_TEN_XN = re.compile(
    r"^(phẫu thuật|nội soi|chọc dò|đặt stent|đặt ống|"
    r"thủ thuật|nội soi|can thiệp|cắt \w+|"
    r"xạ trị|hóa trị|"
    r"siêu âm|chụp \w+|"
    r"đo \w+|test \w+ \w+)$",
    re.IGNORECASE | re.UNICODE,
)

# Treatment modalities → CHẨN_ĐOÁN (không phải THUỐC cụ thể)
_TREATMENT_MODALITY_TO_CHAN_DOAN = re.compile(
    r"^(liệu pháp \w+|điều trị \w+|phác đồ \w+|"
    r"phương pháp \w+|kỹ thuật \w+)$",
    re.IGNORECASE | re.UNICODE,
)


def _retype_entity(text: str, etype: str) -> str:
    """Auto-correct entity type dựa trên text patterns (R31 mới 2026-07-10).

    Logic:
    - Abnormal findings (tim to, tràn dịch, gãy xương, hở van, ...) → CHẨN_ĐOÁN
      (không phải TRIỆU_CHỨNG/KQ_XN)
    - Procedures (phẫu thuật, nội soi, chọc dò, ...) → TÊN_XÉT_NGHIỆM
      (không phải THUỐC)
    - Treatment modalities (liệu pháp, ...) → CHẨN_ĐOÁN

    Args:
        text: entity text (đã được _clean_entity_text clean).
        etype: current type từ LLM.

    Returns:
        Corrected type (có thể giữ nguyên nếu đúng).
    """
    if not text:
        return etype
    text_stripped = text.strip()

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
            finding_pos = (search_start, search_start + len(finding))
        finding_type = _retype_entity(finding, "KẾT_QUẢ_XÉT_NGHIỆM")
        result.append({
            "text": finding,
            "type": finding_type,
            "position": list(finding_pos),
            "assertions": [],
            "candidates": [],
        })
        search_start = finding_pos[1]

    return result if len(result) >= 2 else None





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

    Args:
        entities: list entities từ LLM (đã validate position).
        input_text: raw input text.

    Returns:
        list entities đã expand (có thể tăng số lượng nếu có duplicate).

    Rules:
    - Mỗi entity text xuất hiện N lần ở N vị trí khác nhau → giữ N entities
      (giữ entity gốc của LLM + tạo thêm N-1 entities cho các vị trí còn lại).
    - Exact match: text từ LLM khớp 100% với substring trong input.
    - Substring match: text LLM là CON trong input (vd "đánh trống ngực" trong "tăng đánh trống ngực").
    - Modifiers strip: bỏ "tăng", "giảm", "có", "không" trước khi match.
    - Mỗi entity text chỉ xuất hiện 1 lần → giữ nguyên.
    - Min text length = 4 chars để tránh false positive.
    """
    if not entities or not input_text:
        return entities

    # Modifiers VN cần strip trước khi match (R14/R25)
    # PREFIX modifiers: tăng, giảm, có, không, ...
    # SUFFIX qualifiers: nhẹ, nặng, vừa, ... (R6 cũng drop duration/intensity)
    _MODIFIERS_PREFIX = re.compile(
        r"^(tăng|giảm|có|không|đang|bị|bị\s+|rõ|rõ\s+rệt|ít|nhiều|hơi|khoảng|có\s+thể)\s+",
        re.IGNORECASE | re.UNICODE,
    )
    _MODIFIERS_SUFFIX = re.compile(
        r"\s+(nhẹ|nặng|vừa|nhẹ\s+nhàng|nặng\s+nề|nhẹ\s+vừa|có\s+triệu\s+chứng|vừa\s+phải)$",
        re.IGNORECASE | re.UNICODE,
    )

    expanded = list(entities)

    for ent in entities:
        text = str(ent.get("text", "")).strip()
        if len(text) < 4:
            continue

        # Tìm TẤT CẢ occurrences của text trong input (case-insensitive)
        text_lower = text.lower()
        input_lower = input_text.lower()

        # UNION exact + stripped match (lấy tất cả vị trí)
        # Set để tránh duplicate positions giữa exact và stripped
        all_positions_set = set()

        # Cách 1: exact substring match
        start = 0
        while True:
            idx = input_lower.find(text_lower, start)
            if idx < 0:
                break
            all_positions_set.add((idx, idx + len(text)))
            start = idx + 1

        # Cách 2: stripped match (bỏ modifier "tăng", "giảm"...)
        text_stripped = _MODIFIERS_PREFIX.sub("", text_lower).strip()
        # Also strip SUFFIX qualifiers (nhẹ, nặng, vừa)
        text_stripped = _MODIFIERS_SUFFIX.sub("", text_stripped).strip()
        if text_stripped and text_stripped != text_lower and len(text_stripped) >= 4:
            start = 0
            while True:
                idx = input_lower.find(text_stripped, start)
                if idx < 0:
                    break
                all_positions_set.add((idx, idx + len(text_stripped)))
                start = idx + 1

        all_positions = [list(p) for p in sorted(all_positions_set)]

        # Nếu chỉ có 1 occurrence → giữ nguyên
        if len(all_positions) <= 1:
            continue

        # Nếu có N occurrences > 1 entity hiện tại → tạo thêm entities
        # MỚI 2026-07-10: check overlap thay vì chỉ check start position
        # LLM 7B hay output duplicate với position LỆCH (vd [97,110] vs [102,110])
        # → tránh tạo thêm entity trùng overlap
        existing_positions = [
            (e.get("position", [0, 0])[0], e.get("position", [0, 0])[1])
            for e in expanded
            if e.get("text", "").lower() == text_lower
            or _MODIFIERS_PREFIX.sub("", e.get("text", "").lower()).strip() == text_stripped
        ]
        # Tìm các positions chưa có entity VÀ KHÔNG overlap với existing
        missing_positions = [
            p for p in all_positions
            if p[0] not in [ep[0] for ep in existing_positions]  # chưa có start y hệt
            and not any(
                max(p[0], ep[0]) < min(p[1], ep[1])  # overlap check
                for ep in existing_positions
            )
        ]

        # Tạo entities mới cho các positions còn thiếu
        for pos in missing_positions:
            new_ent = {
                "text": text,
                "type": ent.get("type", ""),
                "position": pos,
                "assertions": list(ent.get("assertions", [])),
                "candidates": [],
            }
            expanded.append(new_ent)
            logger.debug(
                "R20.1 expand duplicate: '%s' tại pos=%d (đã có tại %s)",
                text, pos[0], existing_positions,
            )

    # Sort theo position
    expanded.sort(key=lambda e: e.get("position", [0, 0])[0])
    return expanded



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
        if record is not None:
            final.append(record)

    # Phase 2: Parallel candidate attachment across CPU/Thread workers (Upgrade F)
    if len(final) > 1 and (retriever is not None or icd_retriever is not None):
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(final))) as pool:
            futures = [
                pool.submit(_attach_candidates, rec, rec["text"], rec["type"], rec, validated, retriever, icd_retriever)
                for rec in final if rec["type"] in ("THUỐC", "CHẨN_ĐOÁN", "TRIỆU_CHỨNG")
            ]
            concurrent.futures.wait(futures)
    elif len(final) == 1:
        rec = final[0]
        if rec["type"] in ("THUỐC", "CHẨN_ĐOÁN", "TRIỆU_CHỨNG"):
            _attach_candidates(rec, rec["text"], rec["type"], rec, validated, retriever, icd_retriever)

    final.sort(key=lambda e: e["position"][0])
    return final


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
        out.append(ent)
    return out


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

        # Pass 1: Exact substring scan (case-insensitive)
        start = 0
        while True:
            idx = input_lower.find(text_lower, start)
            if idx < 0:
                break
            all_spans.add((idx, idx + len(base_text)))
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
                    span = (idx, idx + len(stripped_text))
                    if not any(s <= idx and (idx + len(stripped_text)) <= e for s, e in all_spans):
                        all_spans.add(span)
                    start = idx + 1

        if not all_spans:
            logger.debug("Align: không tìm được span cho '%s' (%s) → bỏ", base_text, etype)
            continue

        # Gán tuần tự LLM entities cho các spans tìm được để GIỮ NGUYÊN ASSERTIONS
        sorted_spans = sorted(list(all_spans))
        for i, (span_start, span_end) in enumerate(sorted_spans):
            actual_text = input_text[span_start:span_end]
            if i < len(ents):
                ent_to_use = ents[i]
            else:
                # Nếu text xuất hiện nhiều hơn số LLM trả về -> auto-expand (bảo vệ recall)
                # XÓA assertions để tránh gán nhầm isHistorical/isNegated
                ent_to_use = ents[0].copy()
                ent_to_use["assertions"] = []
                
            new_ent = {
                **ent_to_use,
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

    # ── Filter lifestyle/duration noise ──────────────────────────────────────────
    aligned = _filter_lifestyle_entities(aligned)

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
        if s_type != etype or s_text != norm_text:
            continue
        s_start, s_end = s_pos
        # Same exact span → drop current (R22)
        if cur_start == s_start and cur_end == s_end:
            return True, []
        # Overlap → keep longer span
        if max(cur_start, s_start) < min(cur_end, s_end):
            ex_len = s_end - s_start
            cur_len = cur_end - cur_start
            if ex_len >= cur_len:
                return True, []  # existing longer → drop current
            to_remove.append(idx)  # current longer → mark existing for removal
    return False, to_remove


def _build_entity_record(
    text: str,
    etype: str,
    pos: list[int],
    ent: dict[str, Any],
) -> dict[str, Any]:
    """Build 1 record dict với assertions cleaned."""
    assertions = sorted({
        a for a in ent.get("assertions", [])
        if a in {"isNegated", "isFamily", "isHistorical"}
    })
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
) -> None:
    """Gán candidates cho record theo type (RxNorm cho THUỐC, ICD cho CHẨN_ĐOÁN).

    Mutates `record["candidates"]` in-place.
    """
    if etype == "THUỐC" and retriever is not None:
        drug_query = sanitize_drug_text(text)
        if drug_query:
            try:
                codes = retriever.lookup(drug_query)
                record["candidates"] = list(codes) if codes else []
            except Exception as exc:
                logger.warning("RxNorm lookup fail for '%s': %s", text, exc)
    elif etype in ("CHẨN_ĐOÁN", "TRIỆU_CHỨNG") and icd_retriever is not None:
        # ICD lookup cần other_entities context (drugs/symptoms nearby)
        other_ents = [
            e for e in validated
            if e.get("text", "").strip() and e is not ent
        ]
        try:
            codes = icd_retriever.lookup(text, other_entities=other_ents)
            record["candidates"] = list(codes) if codes else []
        except Exception as exc:
            logger.warning("ICD lookup fail for '%s' (%s): %s", text, etype, exc)


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
        new_pos = _find_span(input_text, text)
        if new_pos is not None:
            ent["position"] = list(new_pos)

    # Validate position
    pos = ent.get("position", [0, 0])
    if not (isinstance(pos, list) and len(pos) == 2 and all(isinstance(p, int) for p in pos)):
        pos = [0, 0]
    cur_start, cur_end = int(pos[0]), int(pos[1])

    # R31: Auto-retype dựa trên text patterns (abnormal findings → CHẨN_ĐOÁN, procedures → TÊN_XN)
    new_etype = _retype_entity(text, etype)
    if new_etype != etype:
        etype = new_etype
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
        existing = set(ent.get("assertions", []))
        for d in detected:
            if d not in existing:
                ent.setdefault("assertions", []).append(d)

    # Build record + attach candidates
    record = _build_entity_record(text, etype, pos, ent)
    if not skip_attach:
        _attach_candidates(
            record, text, etype, ent, validated,
            retriever, icd_retriever,
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
            ent.setdefault("assertions", [])
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
