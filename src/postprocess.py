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

import sys
from pathlib import Path

# Đảm bảo có thể chạy trực tiếp `python src/postprocess.py`
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import json
import logging
import re
from pathlib import Path
from typing import Any, Iterable, Optional

from src.icd_rag import ICDRetriever
from src.rxnorm_rag import RxNormRetriever

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


# ---------------------------------------------------------------------- #
# Dedupe
# ---------------------------------------------------------------------- #


def dedupe_entities(entities: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Bỏ trùng entities theo (text, type) - R10 LOOSE chuẩn (2026-07-09).

    R10 LOOSE (chuẩn, tối ưu cho model < 9B):
    - Cùng text + type → 1 entity (dedup theo text+type, bất kể position)
    - Áp dụng cho TẤT CẢ loại (THUỐC, CHẨN_ĐOÁN, TRIỆU_CHỨNG, TÊN_XN, KQ_XN)
    - Giữ entity ở vị trí sớm nhất (position[0] nhỏ nhất)

    R22: TÊN_XÉT_NGHIỆM duplicate cùng 1 admission → 1 entity (R22 dedup).

    Position field (R27.1, từ LLM):
    - Position KHÔNG quyết định dedup (vẫn dedup theo text+type)
    - Position chỉ giúp LLM extract ĐỦ entities (không miss duplicate)
    - Postprocess dùng position[0] để sort entities theo thứ tự xuất hiện

    Tại sao R10 LOOSE (không R10 STRICT theo position):
    - LLM có position giúp extract đủ duplicate (vd "ngất xỉu" x 4 → extract 4)
    - Postprocess dedup về 1 → đơn giản, ổn định
    - Trade-off: giảm recall tuyệt đối nhưng tăng precision (F1 ↑)
    - User preference (turn trước): "chỉ R10 loose, 1 pass"
    """
    out: list[dict[str, Any]] = []
    # Track: (text_lower, type) → đã thấy
    seen_keys: set[tuple[str, str]] = set()

    # Sort theo position để giữ entity sớm nhất
    sorted_ents = sorted(
        [e for e in entities if e.get("text")],
        key=lambda e: (
            int(e.get("position", [0, 0])[0])
            if isinstance(e.get("position"), list) and len(e.get("position")) >= 1
            else 0
        ),
    )

    for ent in sorted_ents:
        etype = ent.get("type", "")
        text = str(ent.get("text", "")).strip()

        # R10 LOOSE: cùng text + type → 1 entity (dedup bất kể position)
        dedup_key = (text.lower(), etype)
        if dedup_key in seen_keys:
            pos = ent.get("position", [0, 0])
            if isinstance(pos, list) and len(pos) >= 1:
                start = pos[0]
            else:
                start = 0
            logger.debug(
                "R10 LOOSE dedup: drop duplicate %s '%s' at pos %d",
                etype, text, start,
            )
            continue
        seen_keys.add(dedup_key)
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
    r"(?:cha|mẹ|anh|chị|em|ông|bà|cô|dì|chú|bác)",
]

_IS_FAMILY_RE = re.compile("|".join(_IS_FAMILY_PATTERNS), re.IGNORECASE | re.UNICODE)


# Bug history: LLM 7B extract nhầm lifestyle/social/psychology words thành
# TRIỆU_CHỨNG dù đã có R3 trong prompt (vd "căng thẳng", "cà phê có caffeine",
# "mất việc làm 8 ngày trước"). Fix: hard filter trong postprocess — drop entity
# có text khớp keyword lifestyle/social/psychology. Defense-in-depth để LLM
# vẫn gặp score cao khi extract thì vẫn bị filter.
_LIFESTYLE_KEYWORDS: set[str] = {
    # Lifestyle / risk factor
    "hút thuốc lá", "thuốc lá", "hút thuốc", "uống rượu", "rượu bia",
    "cà phê", "trà", "tập thể dục", "luyện tập", "căng thẳng", "stress",
    "chế độ ăn", "ăn kiêng", "ngủ ít", "ngủ nhiều", "ngủ đủ",
    # Social events
    "mất việc", "mất việc làm", "mới nghỉ việc", "nghỉ việc",
    "ly hôn", "ly thân", "chuyển nhà", "kết hôn", "sinh con",
    "thất nghiệp", "bị sa thải", "sa thải",
    # General psychology (non-clinical)
    "vui", "buồn", "lo lắng", "cô đơn", "giận", "sợ", "lo", "bực",
    "căng", "áp lực", "mệt mỏi tinh thần",
}

_LIFESTYLE_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(_LIFESTYLE_KEYWORDS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE | re.UNICODE,
)


# Counter cho logging
_seen_count: int = 0


# Bug history: LLM 7B hay gán nhầm tên thuốc thành TÊN_XÉT_NGHIỆM
# hoặc TRIỆU_CHỪNG (vd "guaifenesin ml" bị gán "TÊN_XÉT_NGHIỆM").
# Fix: rescue - nếu first_word của text khớp common drug name → ép về THUỐC.
_COMMON_DRUG_NAMES: set[str] = {
    # Cough / expectorant
    "guaifenesin",
    "dextromethorphan",
    "codeine",
    "ephedrine",
    "phenylephrine",
    "diphenhydramine",
    "chlorpheniramine",
    "loratadine",
    "cetirizine",
    "fexofenadine",
    "desloratadine",
    "levocetirizine",
    "promethazine",
    # Antibiotics
    "amoxicillin",
    "ampicillin",
    "azithromycin",
    "ciprofloxacin",
    "levofloxacin",
    "moxifloxacin",
    "cefixime",
    "ceftriaxone",
    "cefuroxime",
    "cefepime",
    "cefazolin",
    "cephalexin",
    "doxycycline",
    "minocycline",
    "tetracycline",
    "erythromycin",
    "metronidazole",
    "tinidazole",
    "nitrofurantoin",
    "trimethoprim",
    "vancomycin",
    "linezolid",
    "meropenem",
    "imipenem",
    # Cardiovascular
    "amlodipine",
    "nifedipine",
    "felodipine",
    "diltiazem",
    "verapamil",
    "atenolol",
    "metoprolol",
    "bisoprolol",
    "carvedilol",
    "propranolol",
    "lisinopril",
    "enalapril",
    "ramipril",
    "losartan",
    "valsartan",
    "irbesartan",
    "candesartan",
    "olmesartan",
    "telmisartan",
    "hydrochlorothiazide",
    "furosemide",
    "spironolactone",
    "eplerenone",
    "atorvastatin",
    "rosuvastatin",
    "simvastatin",
    "pravastatin",
    "clopidogrel",
    "aspirin",
    "warfarin",
    "apixaban",
    "rivaroxaban",
    "dabigatran",
    "digoxin",
    "amiodarone",
    "sotalol",
    # Diabetes
    "metformin",
    "glipizide",
    "gliclazide",
    "glyburide",
    "glimepiride",
    "sitagliptin",
    "linagliptin",
    "vildagliptin",
    "empagliflozin",
    "dapagliflozin",
    "liraglutide",
    "semaglutide",
    "insulin",
    "insulin-glargine",
    "insulin-aspart",
    "insulin-lispro",
    # GI
    "omeprazole",
    "pantoprazole",
    "lansoprazole",
    "esomeprazole",
    "rabeprazole",
    "ranitidine",
    "famotidine",
    "cimetidine",
    "ondansetron",
    "metoclopramide",
    "domperidone",
    "loperamide",
    "lactulose",
    "bisacodyl",
    "senna",
    "docusate",
    # Respiratory
    "salbutamol",
    "albuterol",
    "ipratropium",
    "tiotropium",
    "formoterol",
    "salmeterol",
    "budesonide",
    "fluticasone",
    "beclomethasone",
    "montelukast",
    "theophylline",
    # CNS
    "paracetamol",
    "acetaminophen",
    "ibuprofen",
    "naproxen",
    "diclofenac",
    "meloxicam",
    "celecoxib",
    "etoricoxib",
    "piroxicam",
    "tramadol",
    "morphine",
    "fentanyl",
    "oxycodone",
    "gabapentin",
    "pregabalin",
    "carbamazepine",
    "lamotrigine",
    "topiramate",
    "sertraline",
    "fluoxetine",
    "escitalopram",
    "venlafaxine",
    "duloxetine",
    "amitriptyline",
    "mirtazapine",
    "haloperidol",
    "risperidone",
    "olanzapine",
    "quetiapine",
    "aripiprazole",
    "diazepam",
    "lorazepam",
    "alprazolam",
    "clonazepam",
    "midazolam",
    "zopiclone",
    "zolpidem",
    # Other
    "methotrexate",
    "azathioprine",
    "cyclophosphamide",
    "hydroxychloroquine",
    "allopurinol",
    "febuxostat",
    "colchicine",
    "prednisone",
    "prednisolone",
    "methylprednisolone",
    "dexamethasone",
    "hydrocortisone",
    "thyroxine",
    "levothyroxine",
    "methimazole",
    "calcium",
    "iron",
    "folic-acid",
    "nystatin",
    "fluconazole",
    "itraconazole",
    "amphotericin",
    "acyclovir",
    "valacyclovir",
    "oseltamivir",
}




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

    # isNegated: check "không", "chưa", "âm tính" trong window 20 chars trước entity.
    # Lưu ý: "không" có thể nằm sát entity (vd "không sốt" → pre_window kết thúc bằng "khô").
    near = text_lower[max(0, pos - 15):pos + 5]  # rộng hơn để bắt "không "
    found_negated = False
    for neg in ("không", "chưa", "âm tính"):
        if neg in near:
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
            # text_j ngắn hơn text_i: check substring
            if text_j in text_i:
                drop_indices.add(j)
                logger.debug(
                    "Drop substring entity '%s' (subset of '%s')",
                    text_j, text_i,
                )

    return [ent for idx, ent in enumerate(entities) if idx not in drop_indices]


def _filter_lifestyle_entities(entities: list[dict]) -> list[dict]:
    """Drop entities khớp lifestyle / social / psychology keywords.

    Defense-in-depth: dù SYSTEM_PROMPT R3 đã cấm, LLM 7B đôi khi vẫn extract
    (vd "căng thẳng", "cà phê có caffeine", "mất việc làm 8 ngày trước") thành
    TRIỆU_CHỨNG. Filter này DROP chúng để khỏi tính F1.

    Return: list entities đã lọc.
    """
    out: list[dict] = []
    for ent in entities:
        text = str(ent.get("text", "")).strip()
        if not text:
            out.append(ent)
            continue
        if _LIFESTYLE_RE.search(text):
            logger.debug(
                "[%d] Drop lifestyle/social/psych entity '%s' (kw match)",
                _seen_count, text,
            )
            continue
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


# VN/Vietnamese-English connectors between drug name and disease.
_DRUG_FOR_DISEASE_RE = re.compile(
    # .+? (non-greedy any-char) thay vì [\w\s\-\.]*? để match được cả
    # dấu ':' trong 'q6h:prn', dấu ',' trong 'bid, prn', v.v.
    r"^(?P<drug>.+?)"
    r"\s+(?:cho|đi[eếề]?u\s*tr[ịiị]|treats?|for|to)\s+"
    r"(?P<disease>.+)$",
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
    # Min length 1 (cho "ho", "sốt", ...) thay vì 3
    if not drug or not disease or len(drug) < 2 or len(disease) < 1:
        return (s, None)
    return (drug, disease)


# ---------------------------------------------------------------------- #
# LLM Context Rescanning — rà soát ngữ cảnh để tối ưu câu truy vấn
# ---------------------------------------------------------------------- #


def rescan_entity_context(
    entity_text: str,
    entity_type: str,
    input_text: str,
    llm_client: Any,
    other_entities: list[dict] | None = None,  # deprecated, kept for signature compat
    cache: dict[str, str] | None = None,  # batch_rescan cache từ assemble_record
) -> str:
    """Dùng LLM dịch/chuẩn hóa entity text sang EN để tra mã.

    Với THUỐC: chuẩn hóa tên thuốc + liều + đường dùng + tần suất.
    Với CHẨN_ĐOÁN: dịch sang EN clinical phrase (giữ nguyên modifier có trong text).

    General principles (Fix 14 — applies to ALL entity types):
    1. Translate VERBATIM — không thêm modifier không có trong entity text.
    2. KHÔNG dùng nearby entities để infer context (đã gây bug ICD sai).
    3. Output phải ngắn gọn (< 100 chars), chỉ chứa thông tin y khoa.

    Args:
        other_entities: DEPRECATED — không còn dùng để thêm vào prompt.
            Giữ để tương thích signature.
        cache: optional dict {original_text: rescanned_text} từ batch_rescan.
            Nếu entity_text có trong cache → dùng luôn, KHÔNG gọi LLM.

    Returns: câu truy vấn tiếng Anh chuẩn hóa, hoặc entity_text gốc nếu LLM fail.
    """
    if llm_client is None:
        return entity_text

    # Check cache first (từ batch_rescan_entities)
    if cache and entity_text in cache:
        return cache[entity_text]

    # Validate entity_text trước khi gọi LLM
    if not entity_text or not entity_text.strip():
        return entity_text

    if entity_type == "THUỐC":
        prompt = (
            "You are a clinical pharmacology expert. Translate the following Vietnamese drug name "
            "into a standardized English RxNorm-searchable phrase.\n\n"
            "STRICT RULES:\n"
            "1. Translate the drug entity VERBATIM — keep generic name, strength, route, frequency.\n"
            "2. DO NOT add indications or modifiers not in the entity text.\n"
            "3. Strip VN parentheticals like '(uống hôm nay)', '(sau ăn)'.\n"
            "4. Convert Vietnamese abbreviations: 'thuốc kháng sinh' → 'antibiotic'.\n"
            "Output format: 'drug_name strength route frequency' (e.g., 'amlodipine 10 mg oral daily').\n"
            "Output ONLY the phrase. No explanation.\n\n"
            f'Drug entity: "{entity_text}"\n\n'
            "Standardized English drug query:"
        )
    elif entity_type == "CHẨN_ĐOÁN":
        prompt = (
            "You are a clinical coding expert. Translate the following Vietnamese diagnosis "
            "into a precise English clinical phrase that can be used to search ICD-10-CM codes.\n\n"
            "STRICT RULES:\n"
            "1. Translate the entity text VERBATIM — keep ALL modifiers (severity, location, cause) "
            "that are present in the entity text.\n"
            "2. DO NOT add modifiers that are NOT in the entity text. "
            "E.g., entity='hepatic encephalopathy' → output 'hepatic encephalopathy' "
            "(NOT 'alcohol-induced hepatic encephalopathy' even if note has alcohol history).\n"
            "3. If the entity text includes cause (e.g., 'xơ gan do rượu' = 'alcoholic cirrhosis'), "
            "keep the cause in the translation.\n"
            "4. DO NOT use nearby entities (drugs/symptoms) to ADD context to the diagnosis. "
            "Nearby entities are for context awareness only — keep diagnosis translation literal.\n\n"
            "Output ONLY the English clinical phrase. No explanation, no ICD code.\n\n"
            f'Diagnosis entity: "{entity_text}"\n\n'
            "Standardized English diagnosis query:"
        )
    else:
        return entity_text

    try:
        msg = [{"role": "user", "content": prompt}]
        resp = llm_client._client.chat.completions.create(  # noqa: SLF001
            model=llm_client.config.model,
            messages=msg,
            temperature=0.0,
            max_tokens=128,
        )
        result = (resp.choices[0].message.content or "").strip().strip('"').strip("'")
        # Fix 14: Validate rescan output — reject suspicious results
        if result and _validate_rescan_output(result, entity_text, entity_type):
            logger.debug("Rescan '%s' (%s) → '%s'", entity_text, entity_type, result)
            return result
        logger.debug("Rescan '%s' rejected (invalid output): %r", entity_text, result)
    except Exception as exc:
        logger.warning("Rescan lỗi (%r): %s", entity_text, exc)
    return entity_text


def batch_rescan_entities(
    entities: list[dict],
    llm_client: Any,
) -> dict[str, str]:
    """Rescan nhiều entities trong 1 LLM call duy nhất.

    Trước đây, mỗi entity được rescan riêng lẻ → note có 30 entities = 30 LLM calls
    liên tiếp → dễ trigger Ollama 500 crash ("model runner has unexpectedly stopped")
    do resource pressure tích lũy. Batch này gộp thành 1 call duy nhất.

    Args:
        entities: list các entity có type 'THUỐC' hoặc 'CHẨN_ĐOÁN'.
                  Mỗi dict cần 'text' và 'type'.
        llm_client: LLMClient instance (đã có _client + config).

    Returns:
        dict mapping {original_text: rescanned_text}.
        Entities không rescan được (fail, invalid) sẽ KHÔNG có trong dict
        — caller dùng original_text làm fallback.
    """
    if not entities or llm_client is None:
        return {}

    # Filter & dedup: chỉ THUỐC + CHẨN_ĐOÁN cần rescan
    to_rescan: list[tuple[int, str]] = []  # (idx, text)
    seen: set[str] = set()
    for i, e in enumerate(entities):
        etype = e.get("type", "")
        if etype not in ("THUỐC", "CHẨN_ĐOÁN"):
            continue
        text = str(e.get("text", "")).strip()
        if text and text not in seen:
            seen.add(text)
            to_rescan.append((len(to_rescan), text))

    if not to_rescan:
        return {}

    n = len(to_rescan)
    # Build compact prompt
    lines = []
    for idx, (i, t) in enumerate(to_rescan, 1):
        # Type tag để LLM biết phải xử lý thế nào
        etype = entities[next(j for j, e in enumerate(entities)
                              if e.get("text") == t)].get("type")
        tag = "DRUG" if etype == "THUỐC" else "DIAG"
        lines.append(f"{i}. [{tag}] {t}")

    entities_list = "\n".join(lines)

    prompt = (
        "You are a clinical entity refiner. For each entity below, output a "
        "standardized English ICD/RxNorm-searchable phrase.\n\n"
        "STRICT RULES:\n"
        "1. For DRUG: keep generic name + strength + route + frequency "
        "(e.g., 'metoprolol 25 mg oral bid'). Strip VN parentheticals like '(uống hôm nay)'.\n"
        "2. For DIAGNOSIS: translate VERBATIM to English clinical phrase "
        "(e.g., 'suy thận mãn' → 'chronic kidney disease'). Keep ALL modifiers.\n"
        "3. DO NOT add context that is NOT in the entity text.\n"
        "4. Keep each output ≤ 100 chars.\n\n"
        f"Entities (n={n}):\n{entities_list}\n\n"
        f'Output JSON object: {{"1": "refined_text", "2": "refined_text", ...}}\n'
        "Use the SAME NUMBERING as input. JSON only, no explanation."
    )

    # Call LLM once với retry
    msg = [{"role": "user", "content": prompt}]
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            resp = llm_client._client.chat.completions.create(  # noqa: SLF001
                model=llm_client.config.model,
                messages=msg,
                temperature=0.0,
                max_tokens=min(2048, 100 * n + 200),  # dynamic: ~100 chars/entity + overhead
            )
            content = (resp.choices[0].message.content or "").strip()

            # Extract JSON
            try:
                result_map = llm_client._extract_json(content)
            except Exception:
                # Fallback: try regex extract JSON object
                result_map = _regex_extract_json_object(content)
            if not isinstance(result_map, dict):
                raise ValueError(f"Expected JSON object, got {type(result_map).__name__}")

            # Map back: key "1" → entity[0].text
            out: dict[str, str] = {}
            for i, (_, text) in enumerate(to_rescan, 1):
                refined = result_map.get(str(i), result_map.get(i, ""))
                if isinstance(refined, str) and refined.strip():
                    if _validate_rescan_output(refined, text, entities[next(
                        j for j, e in enumerate(entities) if e.get("text") == text)].get("type", "")):
                        out[text] = refined.strip()
            logger.debug(
                "Batch rescan: %d/%d entities refined successfully",
                len(out), n,
            )
            return out

        except Exception as exc:
            last_exc = exc
            err_str = str(exc).lower()
            is_transient = (
                "500" in err_str
                or "model runner" in err_str
                or "unexpectedly stopped" in err_str
                or "connection" in err_str
                or "timeout" in err_str
            )
            if is_transient and attempt < 2:
                wait = 2 ** (attempt + 2)  # 4s, 8s — longer than per-entity (Ollama needs time to recover)
                logger.warning(
                    "Batch rescan lỗi transient (attempt %d/3, wait %ds): %r",
                    attempt + 1, wait, exc,
                )
                import time as _t
                _t.sleep(wait)
                continue
            break
    if last_exc is not None:
        logger.warning(
            "Batch rescan lỗi (%d entities): %s → fallback per-entity",
            n, last_exc,
        )
    return {}


def _regex_extract_json_object(content: str) -> dict | None:
    """Fallback: extract JSON object bằng regex khi _extract_json fail.
    """
    import re
    # Tìm {...} đầu tiên
    start = content.find("{")
    end = content.rfind("}")
    if start < 0 or end < 0 or end <= start:
        return None
    candidate = content[start: end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _validate_rescan_output(
    result: str,
    entity_text: str,
    entity_type: str,
) -> bool:
    """Validate rescan LLM output. Return False nếu output đáng ngờ.

    Reject nếu:
    - Output quá dài (>200 chars) — có thể chứa explanation
    - Output chứa nhiều câu (có dấu . nhiều) — LLM đang explain
    - Output chứa "I cannot", "I don't", "Note:", "The patient" — LLM nói nhảm
    - Output chứa "Note that", "Please note" — LLM giải thích
    - Cho THUỐC: output phải chứa first word (drug name) của entity
    """
    if not result:
        return False
    if len(result) > 200:
        return False
    if result.count(".") > 2:
        return False
    bad_starts = (
        "i cannot", "i don't", "i'm sorry", "note:", "the patient",
        "please note", "note that", "as an ai", "as a language",
    )
    lower = result.lower()
    for bad in bad_starts:
        if lower.startswith(bad):
            return False
    # Drug-specific check: first word of entity_text should appear in result
    if entity_type == "THUỐC" and entity_text:
        first_word = entity_text.strip().split()[0].lower() if entity_text.strip() else ""
        # Loại bỏ common prefix như "thuốc"
        if first_word in ("thuốc", "drug", "medicine"):
            first_word = (
                entity_text.strip().split()[1].lower() if len(entity_text.strip().split()) > 1 else ""
            )
        if first_word and len(first_word) > 2 and first_word not in lower:
            return False
    return True


# ---------------------------------------------------------------------- #
# Main assembly
# ---------------------------------------------------------------------- #


def assemble_record(
    input_text: str,
    raw_entities: Iterable[dict[str, Any]],
    retriever: RxNormRetriever,
    icd_retriever: Optional[ICDRetriever] = None,
    llm_client: Any = None,
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

    Rescan strategy: gọi LLM BATCH (1 call) cho tất cả entities có rescan;
    nếu batch fail → fallback per-entity (giữ backward compat).
    """
    validated = validate_positions(input_text, raw_entities)
    validated = dedupe_entities(validated)
    # Fix: _drop_substring_entities đã được định nghĩa nhưng KHÔNG BAO GIỜ được gọi.
    # Gọi ở đây để dedup duplicate substring (vd: "cảm giác thắt chặt ngực vùng trước tim"
    # và "cảm giác thắt chặt ngực" cùng type → chỉ giữ entity dài hơn).
    validated = _drop_substring_entities(validated)

    # Lifestyle/social/psychology hard filter (defense-in-depth: drop entity match keyword)
    # LLM 7B đôi khi extract "căng thẳng", "cà phê", "mất việc" thành TRIỆU_CHỨNG dù R3 đã cấm.
    validated = _filter_lifestyle_entities(validated)

    # Batch rescan: 1 LLM call cho N entities (giảm load Ollama 10-30x)
    # → giảm 500 crash do resource pressure.
    rescan_cache: dict[str, str] = batch_rescan_entities(validated, llm_client)

    final: list[dict[str, Any]] = []
    # Track text+type đã emit để dedupe (vd "doxycycline" trùng khi LLM trả 2 lần)
    seen_signatures: set[tuple[str, str]] = set()
    # Loại type được phép có candidates (per spec: CHỈ CHẨN_ĐOÁN + THUỐC)
    CANDIDATES_ALLOWED_TYPES = {"CHẨN_ĐOÁN", "THUỐC"}
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

        # Bug history: LLM 7B hay gán nhầm tên thuốc → TÊN_XÉT_NGHIỆM.
        # Fix: nếu text khớp common drug name → ép về THUỐC.
        first_word = text.strip().split()[0].lower() if text.strip() else ""
        if (
            etype in ("TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM", "TRIỆU_CHỨNG")
            and first_word in _COMMON_DRUG_NAMES
        ):
            logger.debug(
                "[%d] Rescue: '%s' %s → THUỐC (matches drug name)",
                _seen_count,
                text,
                etype,
            )
            etype = "THUỐC"

        # Rescue: Skip - dạy LLM qua prompt thay vì hardcode set.
        # (LLM đã được dạy để phân biệt triệu chứng phổ biến vs chẩn đoán qua examples.)

        # Bug history: LLM 7B gán isFamily cho mọi thứ trong "tiền sử",
        # kể cả bệnh của bệnh nhân. Fix: verify family-context GẦN entity (window 200 chars).
        if "isFamily" in assertions:
            start_pos = int(ent["position"][0])
            window_start = max(0, start_pos - 200)
            window_end = start_pos  # check trước entity
            nearby = input_text[window_start:window_end]
            if not _IS_FAMILY_RE.search(nearby):
                assertions = [a for a in assertions if a != "isFamily"]
                logger.debug(
                    "[%d] Drop isFamily: '%s' (no family-context nearby)",
                    _seen_count,
                    text,
                )

        # Lifestyle/behavior entities đã được dạy cho LLM qua SYSTEM_PROMPT
        # để không extract. Nếu LLM vẫn extract → tin tưởng LLM (general standard).

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
                    "candidates": [],
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
            diag_type = "CHẨN_ĐOÁN"
            diag_lower = diag_part.lower().strip()
            symptom_keywords = {
                "ho",
                "sốt",
                "đau",
                "nhức",
                "khó thở",
                "buồn nôn",
                "nôn",
                "táo bón",
                "tiêu chảy",
                "mất ngủ",
                "lú lẫn",
                "nói nhảm",
                "sụt cân",
                "chóng mặt",
                "mệt mỏi",
                "lo âu",
                "tê",
                "ngứa",
                "khó nuốt",
                "yếu nửa người",
            }
            if any(kw in diag_lower for kw in symptom_keywords):
                diag_type = "TRIỆU_CHỨNG"

            if (diag_part, diag_type) not in seen_signatures:
                item2: dict[str, Any] = {
                    "text": diag_part,
                    "type": diag_type,
                    "position": [diag_pos_start, diag_pos_end],
                    "assertions": list(assertions),
                    "candidates": [],
                }
                # Candidates chỉ cho CHẨN_ĐOÁN (Fix 3: apply rescan cho split diagnosis)
                if diag_type == "CHẨN_ĐOÁN" and icd_retriever is not None:
                    # Build other_entities list (đã loại bỏ chính diag_part)
                    other_for_diag = [
                        e for e in validated
                        if e.get("text") != diag_part and e.get("type") != diag_type
                    ]
                    rescanned_diag = rescan_entity_context(
                        diag_part, "CHẨN_ĐOÁN", input_text, llm_client,
                        other_entities=other_for_diag,
                        cache=rescan_cache,
                    )
                    cand2 = icd_retriever.lookup(
                        rescanned_diag, other_entities=other_for_diag,
                    )
                    # Fix 13: Fallback cho split ICD
                    if not cand2 and rescanned_diag != diag_part:
                        cand2 = icd_retriever.lookup(
                            diag_part, other_entities=other_for_diag,
                        )
                    if cand2:
                        item2["candidates"] = cand2
                final.append(item2)
                seen_signatures.add((diag_part, diag_type))
                emitted_any = True
            # Nếu không emit gì thì skip (đã có từ entity trước)
            if not emitted_any:
                continue
            continue

        # Sanitize text cho THUỐC (R4 mới 2026-07) — strip "x N" + parens VN instructions.
        # Update position nếu text bị thay đổi để tránh invalid position.
        if etype == "THUỐC":
            cleaned_text = sanitize_drug_text(text)
            if cleaned_text and cleaned_text != text:
                text = cleaned_text
                # Re-find new position trong input
                new_start = input_text.find(text, int(ent["position"][0]))
                if new_start >= 0:
                    ent["position"] = [new_start, new_start + len(text)]

        item: dict[str, Any] = {
            "text": text,
            "type": etype,
            "position": [int(ent["position"][0]), int(ent["position"][1])],
            "assertions": assertions,
            "candidates": [],   # SPEC: luôn có field, [] cho non-allowed types
        }
        # Postprocess smart-assert: NẾU LLM quên assertions, detect từ input context.
        if not assertions:
            assertions = _detect_assertions_from_context(
                text, input_text, etype, int(ent["position"][0]),
            )
            item["assertions"] = assertions
        # Candidates CHỈ được populate cho CHẨN_ĐOÁN và THUỐC (per spec).
        # 3 type còn lại (TRIỆU_CHỨNG, TÊN_XÉT_NGHIỆM, KẾT_QUẢ_XÉT_NGHIỆM)
        # sẽ có candidates=[] empty list.
        # Bước rescan: dùng LLM rà soát ngữ cảnh để sinh câu truy vấn tối ưu
        if etype == "THUỐC":
            # Rescan để lấy query chuẩn hóa (gom liều, đường dùng từ văn bản)
            # Pass other_entities để LLM có indication context (Fix 6)
            # Use cache từ batch_rescan để giảm LLM calls
            other_for_drug = [e for e in validated if e.get("text") != text]
            rescanned = rescan_entity_context(
                text, etype, input_text, llm_client,
                other_entities=other_for_drug,
                cache=rescan_cache,
            )
            cleaned = sanitize_drug_text(rescanned)
            cand: list[str] = []
            if cleaned:
                cand = retriever.lookup(cleaned)
            # Fix 13: Fallback strategy - nếu LLM rescan trả [] (do LLM confused hoặc
            # API fail), thử lại với text gốc.
            if not cand:
                cleaned_orig = sanitize_drug_text(text)
                if cleaned_orig and cleaned_orig != cleaned:
                    cand = retriever.lookup(cleaned_orig)
                    if cand:
                        logger.debug(
                            "[%d] Rescan fallback cho '%s': dùng text gốc → %d candidates",
                            _seen_count, text, len(cand),
                        )
            if cand:
                item["candidates"] = cand
        elif etype == "CHẨN_ĐOÁN" and icd_retriever is not None:
            # Rescan để lấy query EN chuẩn hóa (gom mức độ, biến chứng)
            # Pass other_entities để LLM có indication context (Fix 6)
            # KHÔNG truyền context_query cho BGE-M3 (Fix 7 contaminated embeddings)
            # Use cache từ batch_rescan
            other_for_diag = [e for e in validated if e.get("text") != text]
            rescanned = rescan_entity_context(
                text, etype, input_text, llm_client,
                other_entities=other_for_diag,
                cache=rescan_cache,
            )
            cand = icd_retriever.lookup(
                rescanned, other_entities=other_for_diag,
            )
            # Fix 13: Fallback cho ICD - nếu rescan fail, thử text gốc
            if not cand and rescanned != text:
                cand = icd_retriever.lookup(
                    text, other_entities=other_for_diag,
                )
                if cand:
                    logger.debug(
                        "[%d] ICD rescan fallback cho '%s': dùng text gốc → %d candidates",
                        _seen_count, text, len(cand),
                    )
            if cand:
                item["candidates"] = cand
        elif etype == "TÊN_XÉT_NGHIỆM":
            # SPEC: không có field `result`. Mối liên hệ TÊN_XÉT_NGHIỆM ↔ KẾT_QUẢ
            # được handle NGẦM bằng position proximity (cùng section trong input).
            # Logging ở debug level để debug, không ghi vào output.
            linked_results = _link_test_results(
                text, int(ent["position"][0]), int(ent["position"][1]), validated,
            )
            if linked_results:
                logger.debug(
                    "[%d] TÊN_XÉT_NGHIỆM '%s' ↔ %d KQ (link ngầm qua position)",
                    _seen_count, text, len(linked_results),
                )
        final.append(item)

    # Defense-in-depth: đảm bảo MỌI entity có field `candidates` (schema required).
    # Nếu LLM quên, auto-fill [] (theo spec: chỉ THUỐC/CHẨN_ĐOÁN mới có codes).
    for ent in final:
        if "candidates" not in ent:
            ent["candidates"] = []
    return final


def _link_test_results(
    test_name: str,
    test_start: int,
    test_end: int,
    validated: list[dict],
    window: int = 250,
) -> list[str]:
    """Tìm các KẾT_QUẢ_XÉT_NGHIỆM gần TÊN_XÉT_NGHIỆM và trả về list giá trị SỐ.

    Fix 17: result chỉ chứa SỐ thuần (vd "12.5", "180") — bỏ tên test + đơn vị.

    Logic:
    - Tìm KẾT_QUẢ_XÉT_NGHIỆM có position nằm trong window (test_end, test_end + window)
      (sau tên test, trong cùng dòng/đoạn).
    - Hoặc trong window (test_start - window, test_start) (trước tên test).
    - Extract số từ KQ text (regex).
    - Trả về list số (string), giữ nguyên format (string để không mất precision).
    """
    results: list[str] = []
    seen: set[str] = set()
    number_re = re.compile(r"-?\d+(?:[.,]\d+)?")
    for e in validated:
        if e.get("type") != "KẾT_QUẢ_XÉT_NGHIỆM":
            continue
        pos = e.get("position", [0, 0])
        if not isinstance(pos, list) or len(pos) != 2:
            continue
        e_start = int(pos[0])
        e_end = int(pos[1])
        # Check nếu KQ nằm trong window sau tên test (preferred)
        in_forward = e_start >= test_end and e_start - test_end <= window
        # Check nếu KQ nằm trong window trước tên test
        in_backward = e_end <= test_start and test_start - e_end <= window
        if not (in_forward or in_backward):
            continue
        text = e.get("text", "").strip()
        if not text:
            continue
        # Extract SỐ từ text (Fix 17: chỉ lấy số thuần)
        # Ví dụ: "WBC 12.5 K/uL" → "12.5"
        #         "glucose 180 mg/dL" → "180"
        #         "SpO2 96%" → "96"
        #         "AST 45 U/L" → "45"
        #         "Hgb 13.2 g/dL" → "13.2"
        numbers = number_re.findall(text)
        if not numbers:
            # Nếu không tìm được số, fallback lưu full text
            extracted = text
        else:
            # Lấy số đầu tiên (hoặc số lớn nhất cho range như "12-15")
            extracted = numbers[0]
        if extracted and extracted not in seen:
            results.append(extracted)
            seen.add(extracted)
    return results


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
    except Exception:
        return False
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
