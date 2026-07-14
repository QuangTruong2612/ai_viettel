"""ICD-10 RAG — tra cứu mã ICD-10 cho chẩn đoán.

Input/output là tiếng Việt (giữ nguyên text gốc trong output JSON).

Dữ liệu: DM_ICD10_19_8_BYT.json (BYT Việt Nam, 36,689 entries, QĐ 4469/BYT).
- Mã: ICD-10 code
- Tên bệnh: Tên bệnh tiếng Việt (có thể có parenthetical context như "Hen [suyễn]")
- Nhóm bệnh: Chương/nhóm bệnh (context phân biệt)
- Mô tả, Hiệu lực, ... (metadata)

Pipeline 5 lớp (offline, 100% local):
  L1: Exact match dict VN (prebuilt, ICDIndex.exact)
  L2: Skip Translate (vì data là VN, match trực tiếp)
  L3: Semantic extraction (BGE-M3 cosine ≥ 0.7) — trả TẤT CẢ codes match,
        không cap top-K. BM25 keyword dùng để mở rộng candidates.
  L4: Fuzzy match VN (rapidfuzz, partial_ratio)
  L5: BM25 fallback (nếu hybrid fail)
"""

from __future__ import annotations

import gzip
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import numpy as np

try:
    from rank_bm25 import BM25Okapi  # type: ignore

    _HAS_BM25 = True
except ImportError:  # pragma: no cover
    BM25Okapi = None  # type: ignore
    _HAS_BM25 = False

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# ════════════════════════════════════════════════════════════════════════════════
# R28 (2026-07-13): Loader cho auto-mined ICD aliases từ data/icd_aliases.json
# (chạy `python scripts/build_mining_index.py` để generate file này).
# Format file: {"code": [alias1, alias2, ...]} — sinh tự động từ icd10.jsonl
# với bracket/paren extraction. Không cần hand-curate.
# ════════════════════════════════════════════════════════════════════════════════

def _load_mined_icd_aliases(target_dict: dict[str, list[str]]) -> int:
    """Merge auto-mined aliases (alias_lower → [code]) vào target_dict in-place.

    Không làm gì nếu data/icd_aliases.json không tồn tại (chưa chạy mining script).
    Returns số aliases added. Idempotent — gọi nhiều lần OK.
    """
    path = DATA_DIR / "icd_aliases.json"
    if not path.exists():
        logger.debug(
            "[R28] %s chưa tồn tại — chạy `python scripts/build_mining_index.py` "
            "để auto-mine ICD aliases.", path.name,
        )
        return 0
    try:
        mined = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("[R28] Failed to load %s: %s", path, exc)
        return 0
    n_added = 0
    for code, alias_list in mined.items():
        for alias in alias_list:
            ak = alias.lower().strip()
            if not ak or len(ak) < 3:
                continue
            existing = target_dict.get(ak, [])
            if code not in existing:
                target_dict.setdefault(ak, []).append(code)
                n_added += 1
    logger.info(
        "[R28] Merged %d auto-mined ICD aliases (%d unique codes)",
        n_added, len(mined),
    )
    return n_added


# ════════════════════════════════════════════════════════════════════════════════
# R28 (2026-07-13): Generic class term detection + uninformative ICD precision guard.
# Patterns load từ data/generic_class_stoplist.json (dễ mở rộng, không hardcode).
# ════════════════════════════════════════════════════════════════════════════════

_UNINFORMATIVE_ICD_PATTERNS: tuple[re.Pattern, ...] = (
    # Catch-all chapters (R, Y, Z ở length 3)
    # Specific catch-alls ONLY (R17, R51, etc. are SPECIFIC — không flag)
    re.compile(r"^R69(\.\d+)?$"),
    re.compile(r"^Z00\.\d+$"),
    # T45.xx — broad poisoning by drugs class
    re.compile(r"^T45\.\d+$"),
    # Y50-Y57 — broad drug/medicament class
    re.compile(r"^Y5[0-7]\.\d+$"),
)


def _is_uninformative_icd_code(code: str) -> bool:
    """Return True nếu code là catch-all (low information for retrieval).

    R29 (2026-07-13): Fix #2 — 3-char R/Y/Z blanket rule was over-aggressive:
    R17 (hyperbilirubinemia), R51 (headache), R00-R09 chapters chứa codes CỤ THỂ.
    R69 là CỤ THỂ broad code — chỉ flag R69/Y95/Z00/T45/Y50-57 explicit patterns.
    """
    if not code:
        return True
    return any(p.match(code) for p in _UNINFORMATIVE_ICD_PATTERNS)


_GENERIC_DRUG_CLASS_PATTERNS: tuple[re.Pattern, ...] = (
    # R34 FIX: simpler patterns match WHOLE class term (e.g., "Thuốc chống đông", "kháng sinh").
    # Dùng `fullmatch` thay vì `match` để tránh partial match với drug attached.
    re.compile(r"^thuốc\s+\w+(\s+\w+)*$", re.IGNORECASE | re.UNICODE),
    re.compile(r"^(kháng|chống)\s+\w+(\s+\w+)*$", re.IGNORECASE | re.UNICODE),
    re.compile(r"^(nhóm|loại|họ)\s+\w+(\s+\w+)*$", re.IGNORECASE | re.UNICODE),
    re.compile(r"(toàn\s+thân)$", re.IGNORECASE | re.UNICODE),
    re.compile(r"^(thuốc|loại\s+thuốc|nhóm\s+thuốc)$", re.IGNORECASE | re.UNICODE),
)


def _is_generic_drug_class(text: str) -> bool:
    """Detect generic drug-class terms (vd 'thuốc chống đông', 'kháng sinh').

    Trả True nếu text là pure class term (vd "kháng sinh", "Thuốc chống đông"),
    KHÔNG phải drug name (vd "Kháng sinh Cefepim" → False, cần lookup Cefepim).

    Logic:
      - Strip drug-class prefix (kháng sinh, thuốc chống, ...).
      - Nếu stripped empty / same → check direct fullmatch.
      - Nếu stripped có drug name (token ∈ INN whitelist) → False (lookup drug).
      - Còn lại → True (class term).
    R34: Thêm `_DRUG_CLASS_ROUTE_PATTERNS` để match "kháng sinh tĩnh mạch" (class + route).
    """
    if not text:
        return False
    tl = text.lower().strip()
    if not tl or len(tl) > 60:
        return False

    # R34: Class + route (vd "kháng sinh tĩnh mạch") → class term, skip
    if _DRUG_CLASS_ROUTE_PATTERNS.fullmatch(tl):
        return True

    stripped = _strip_drug_class_prefix(text)
    if stripped is None or stripped == text:
        # Pure class term hoặc không có class prefix → check direct fullmatch
        return any(p.fullmatch(tl) for p in _GENERIC_DRUG_CLASS_PATTERNS)

    # Stripped result exists. Check if first token is a real drug (INN whitelist).
    stripped_lower = stripped.lower().strip()
    stripped_tokens = stripped_lower.split()
    if stripped_tokens:
        first_token = stripped_tokens[0]
        # Import lazily (avoid circular at module load)
        from src.rxnorm_rag import _DRUG_INN_WHITELIST
        if first_token in _DRUG_INN_WHITELIST:
            return False  # drug name attached → lookup drug
    return True  # no drug name → still class term


# R34 (2026-07-13): Strip drug-class prefix để lookup drug part
# (vd "Kháng sinh Cefepim" → "Cefepim", "Thuốc chống đông X" → "X")
_DRUG_CLASS_PREFIX_RE = re.compile(
    r"^(?:thuốc\s+\w+(\s+\w+)*\s+"
    r"|(?:kháng|chống)\s+\w+(\s+\w+)*\s+"
    r"|(?:nhóm|loại|họ)\s+\w+(\s+\w+)*\s+"
    r"|(?:thuốc|loại\s+thuốc|nhóm\s+thuốc)\s+)",
    re.IGNORECASE | re.UNICODE,
)

# R34 (2026-07-13): Additional patterns để detect drug-class + route concatenation
# (vd "kháng sinh tĩnh mạch", "thuốc an thần đường uống"). Skip luôn — không có drug cụ thể.
_DRUG_CLASS_ROUTE_PATTERNS = re.compile(
    r"^(?:kháng\s+sinh|chống\s+viêm|thuốc\s+(?:chống|kháng|giảm|hạ|tăng|lợi|cầm|an\s+thần|bổ|giúp)|"
    r"nhóm\s+thuốc)\s+(?:tĩnh\s+mạch|uống|tiêm|truyền|tiêm\s+tĩnh\s+mạch|"
    r"tiêm\s+bắp|tiêm\s+dưới\s+da|uống\s+sau\s+ăn|uống\s+trước\s+ăn|"
    r"đường\s+uống|đường\s+tiêm|đường\s+tĩnh\s+mạch|trước\s+ăn|sau\s+ăn|khi\s+đau)"
    r"(\s+\w+)*\s*$",
    re.IGNORECASE | re.UNICODE,
)


def _strip_drug_class_prefix(text: str) -> str | None:
    """R34 FIX: Strip drug-class prefix để lookup drug part.

    Returns:
        - None nếu text là pure class term (vd "kháng sinh", "thuốc chống đông")
        - Stripped drug name nếu có drug attached (vd "Kháng sinh Cefepim" → "Cefepim")
        - Original text nếu không match class pattern
    """
    if not text:
        return text
    stripped = _DRUG_CLASS_PREFIX_RE.sub("", text, count=1).strip()
    if not stripped:
        # Pure class term — caller SKIP lookup
        return None
    return stripped


def _load_generic_stoplist() -> tuple[tuple[re.Pattern, ...], ...]:
    """Load patterns từ data/generic_class_stoplist.json (overrides default)."""
    path = DATA_DIR / "generic_class_stoplist.json"
    if not path.exists():
        return _GENERIC_DRUG_CLASS_PATTERNS,
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
        patterns = cfg.get("vn_drug_class_patterns", [])
        return tuple(re.compile(p, re.IGNORECASE | re.UNICODE) for p in patterns),
    except Exception:
        return _GENERIC_DRUG_CLASS_PATTERNS,


# Re-init từ file nếu có
try:
    _GENERIC_DRUG_CLASS_PATTERNS, = _load_generic_stoplist()
    logger.info("[R28] Loaded %d generic drug class patterns", len(_GENERIC_DRUG_CLASS_PATTERNS))
except Exception:
    pass



# R34: ICD drug-class blacklist + resistance detection (R34 spec round 3)
_ICD_DRUG_CLASS_BLACKLIST: frozenset[str] = frozenset({
    "kháng sinh", "thuốc chống đông", "thuốc giảm đau", "thuốc hạ sốt",
    "nsaid", "corticoid", "thuốc lợi tiểu", "thuốc an thần",
})


def _has_resistance_context_icd(text: str) -> bool:
    """True nếu text là resistance mention (vd 'vi khuẩn kháng thuốc',
    'E. coli kháng vancomycin')."""
    t = text.lower()
    # Specific patterns (high precision)
    if re.search(r"\bkháng\s+thuốc\b", t):
        return True
    if re.search(r"\bkháng\s+sinh\b", t):
        return True
    # General: 'X kháng Y' where Y is 3+ chars (drug name)
    # Catches 'E. coli kháng vancomycin', 'Staph kháng methicillin', etc.
    if re.search(r"kháng\s+\w{4,}", t):
        return True
    return False


# R34 (2026-07-13): Resistance suffix stripping — strip "kháng thuốc" / "kháng sinh"
# khỏi diagnosis text để lookup core diagnosis (vd "Nhiễm trùng đường tiết niệu kháng thuốc" → N39.0).
_RESISTANCE_SUFFIX_RE = re.compile(
    r"\s*kháng\s+(?:thuốc|sinh|kháng\s+sinh)\s*$",
    re.IGNORECASE | re.UNICODE,
)


def _strip_resistance_suffix(text: str) -> str:
    """R34: Strip trailing resistance context → lookup core diagnosis."""
    if not text:
        return text
    return _RESISTANCE_SUFFIX_RE.sub("", text).strip()


def _looks_like_noise_tokens(text: str) -> bool:
    """Detect nonsense tokens (vd 'asdfgh', 'xyz test') — không phải clinical term.

    Heuristics:
      - Token toàn lowercase Latin letters, không có VN diacritic, AND
      - Không match VN→ICD dict, AND
      - Tất cả tokens đều random-like (chỉ chữ cái liền nhau không có nghĩa)
    """
    if not text or len(text) < 4:
        return False
    import re as _re

    # Has Vietnamese diacritic → likely real VN text
    if _re.search(r"[ăâđêôơưáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]", text.lower()):
        return False

    # R34 fix: ALL-UPPERCASE tokens → acronym (NMCT, TBMMN), KHÔNG phải noise.
    stripped = text.strip()
    if stripped.isupper() and stripped.replace(" ", "").replace(".", "").isalpha():
        return False

    # R34 fix: query is a KNOWN VN abbreviation (in acronym_map) → not noise
    # Catches lowercase forms như "rlll", "btmv", "bptnmt" → allow.
    if hasattr(_KNOWN_VN_ABBR_LOOKUP, "_initialized") is False:
        _init_known_abbrs()
    if stripped.lower() in _KNOWN_VN_ABBR_LOOKUP:
        return False

    t = text.lower().strip()
    tokens = _re.findall(r"[a-z0-9]+", t)
    if not tokens:
        return False
    # Reject nếu có ÍT NHẤT 1 token "garbled" (vd 'xyz', 'asdfgh')
    # Real medical terms có vowel pattern; noise là consonant cluster.
    for w in tokens:
        if _is_likely_garbled(w):
            return True
    return False


_KNOWN_VN_ABBR_LOOKUP: set[str] = set()


def _load_vn_medical_abbreviations(target_dict: dict[str, list[str]]) -> int:
    """R35 (2026-07-14): Load VN medical abbreviations → ICD-10 codes.

    File: data/icd_abbreviations.json
    Format: {"sp tlh": ["N85.8"], "btmv": ["I25"], ...}
    Empty list → admin/note (skip lookup), non-empty → lookup ICD codes.

    Returns:
        Number of abbreviations added.
    """
    path = DATA_DIR / "icd_abbreviations.json"
    if not path.exists():
        logger.debug("[R35] %s chưa tồn tại", path.name)
        return 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            logger.warning("[R35] %s phải là dict, got %s", path.name, type(data).__name__)
            return 0
        n_added = 0
        for key, codes in data.items():
            if not isinstance(key, str) or not isinstance(codes, list):
                continue
            ak = key.lower().strip()
            if not ak:
                continue
            existing = target_dict.get(ak, [])
            for c in codes:
                if c not in existing:
                    existing.append(c)
            if existing:
                target_dict[ak] = existing
                n_added += 1
            # R35: Also add to _KNOWN_VN_ABBR_LOOKUP so noise filter doesn't reject them
            _KNOWN_VN_ABBR_LOOKUP.add(ak)
        logger.info("[R35] Merged %d VN medical abbreviations (%d unique keys)",
                    n_added, len(data))
        return n_added
    except Exception as exc:
        logger.warning("[R35] Failed to load %s: %s", path, exc)
        return 0


def _init_known_abbrs() -> None:
    """Lazy-load known VN abbreviations from _VN_ABBREVIATIONS dict."""
    global _KNOWN_VN_ABBR_LOOKUP
    if _KNOWN_VN_ABBR_LOOKUP:
        return
    _KNOWN_VN_ABBR_LOOKUP = set(_VN_ABBREVIATIONS.keys())
    _KNOWN_VN_ABBR_LOOKUP |= {
        "btmv", "bptnmt", "vgb", "vgc", "rlntn", "nttn", "nttt",
        "đtrđ", "vpmpccđ", "st chênh",  # from acronym_map in lookup()
        "rlll",  # rối loạn lipid máu (from acronym_map)
        "nkh",  # nhiễm khuẩn huyết
    }
    _KNOWN_VN_ABBR_LOOKUP.add("_initialized")


def _is_likely_garbled(word: str) -> bool:
    """Heuristic: word có vẻ ngẫu nhiên.

    Real clinical terms (parkinson, copd) có vowel density ~0.3-0.5.
    Noise ('asdfgh', 'xyz') có density ~0.

    R34 (2026-07-13): Sửa để KHÔNG reject valid medical terms có consecutive
    consonants (vd 'parkinson' có 'rk', 'ns' → trước đây bị reject).
    """
    import re as _re
    if not word or len(word) < 3:
        return False
    word_low = word.lower()
    vowels = _re.findall(r"[aeiouăâđêôơư]+", word_low)
    density = len(vowels) / len(word_low)
    # Real VN/EN word has at least 20% vowel density
    if density < 0.20:
        return True
    return False


def _is_drug_class_term_icd(text: str) -> bool:
    """True nếu text là drug-class term thuần (không phải diagnosis cụ thể)."""
    return text.lower().strip() in _ICD_DRUG_CLASS_BLACKLIST


# R34 (2026-07-13): Generic/uninformative VN phrases nên return []
# "không xác định được" → chung chung, không phải diagnosis cụ thể.
# Random tokens → noise, không match ICD.
_UNINFORMATIVE_VN_TERMS = frozenset({
    "không xác định được", "không rõ", "chưa rõ", "cần xác định",
    "cần làm rõ", "không xác định", "chưa xác định",
})


def _is_uninformative_vn_term(text: str) -> bool:
    """True nếu text là generic phrase không phải diagnosis (vd 'không xác định được')."""
    t = text.lower().strip()
    return t in _UNINFORMATIVE_VN_TERMS


# ════════════════════════════════════════════════════════════════════════════════
# VN medical abbreviations / synonyms (R27.6 mới 2026-07-10)
# ════════════════════════════════════════════════════════════════════════════════

# Map abbreviation VN → full term trước khi vào lookup chain.
# LLM hay output viết tắt (THA, NMCT, ĐTĐ, COPD, OSA, ...) mà ICD desc_vi
# dùng full term (tăng huyết áp, nhồi máu cơ tim, đái tháo đường, ...).
_VN_ABBREVIATIONS = {
    # Tim mạch — BYT viết tắt chuẩn
    "tha": "tăng huyết áp",
    "tăng ha": "tăng huyết áp",
    "ha cao": "tăng huyết áp",
    "nmct": "nhồi máu cơ tim",
    "nmct cũ": "nhồi máu cơ tim cũ",
    "nmct mới": "nhồi máu cơ tim cấp",
    "nmct cấp": "nhồi máu cơ tim cấp",
    "nmct cấp stemi": "nhồi máu cơ tim cấp st chênh lên",
    "stemi": "nhồi máu cơ tim cấp st chênh lên",
    "nstemi": "nhồi máu cơ tim cấp không st chênh lên",
    "suy tim": "suy tim",
    "suy tim ứ huyết": "suy tim ứ huyết",
    "suy tim nyha": "suy tim",
    "bt": "block tim",
    "bav": "block nhĩ thất",
    "bbb": "block nhánh",
    "lbbb": "block nhánh trái",
    "rbbb": "block nhánh phải",
    "ntts": "ngoại tâm thu",
    "ntt nhĩ": "ngoại tâm thu nhĩ",
    "ntt thất": "ngoại tâm thu thất",
    "ntts nhĩ": "ngoại tâm thu nhĩ",
    "ntts thất": "ngoại tâm thu thất",
    "bptt": "block nhĩ thất",
    "pac": "ngoại tâm thu nhĩ",
    "pvc": "ngoại tâm thu thất",
    "af": "rung nhĩ",
    "svt": "nhịp nhanh trên thất",
    "vt": "nhịp nhanh thất",
    "vf": "rung thất",
    "tđm": "tai biến mạch máu não",
    "tbmmn": "tai biến mạch máu não",
    "tbmmnn": "tai biến mạch máu não",
    "đột quỵ": "tai biến mạch máu não",
    "cva": "tai biến mạch máu não",
    "tia": "cơn thiếu máu não thoáng qua",
    # Nội tiết
    "đtđ": "đái tháo đường",
    "dtd": "đái tháo đường",
    "đtđ type 2": "đái tháo đường type 2",
    "đtđ type 1": "đái tháo đường type 1",
    "đtđ2": "đái tháo đường type 2",
    "đtđ1": "đái tháo đường type 1",
    "dm": "đái tháo đường",
    "dm2": "đái tháo đường type 2",
    "dm type 2": "đái tháo đường type 2",
    "rlcd": "rối loạn chuyển hóa đường",
    "rlld": "rối loạn lipid máu",
    # Hô hấp
    "copd": "bệnh phổi tắc nghẽn mạn",
    "hpq": "hen phế quản",
    "hen pq": "hen phế quản",
    "vp": "viêm phổi",
    "vpmpcđ": "viêm phổi mắc phải cộng đồng",
    "vpq": "viêm phế quản",
    "osa": "ngưng thở khi ngủ",
    "suy hô hấp": "suy hô hấp",
    "tràn khí mp": "tràn khí màng phổi",
    "tràn dịch mp": "tràn dịch màng phổi",
    # Thận - Tiết niệu
    # IMPORTANT: 'st', 'stc', 'stm' bị xóa vì conflict với ECG term 'ST' (ST segment)
    # Sử dụng full form "suy thận", "suy thận cấp", "suy thận mạn" thay thế.
    "ckd": "suy thận mạn",  # CKD = Chronic Kidney Disease (sàng lọc TRUYỀN THUYẼN việt hóa)
    "akf": "suy thận cấp",
    "aki": "suy thận cấp",
    "vđtn": "viêm đường tiết niệu",
    "ntn": "nhiễm trùng đường tiết niệu",
    "uti": "viêm đường tiết niệu",
    # Tiêu hóa - Gan
    "vg": "viêm gan",
    "vgb": "viêm gan b",
    "vgc": "viêm gan c",
    "gerd": "trào ngược dạ dày thực quản",
    "ibs": "hội chứng ruột kích thích",
    "gastritis": "viêm dạ dày",
    # Ung thư / Khối u
    "hc": "hạch",
    "k": "ung thư",
    "ca": "ung thư",
    "ts": "tiền sử",
    "tm": "tiền sử",
    # Sản - Phụ khoa
    "có thai": "mang thai",
    "có thai tuần": "mang thai",
    # Mắt
    "dtvm": "đái tháo đường biến chứng võng mạc",
    # Chỉnh hình
    "oai": "thoái hóa khớp",
    "ra": "viêm khớp dạng thấp",
    "oa": "thoái hóa khớp",
}

# Map synonyms VN → canonical term (full term) trước khi lookup.
# LLM hay dùng "u ác" thay vì "u ác tính", "khối u" thay vì "u" → mismatch với ICD desc_vi.
_VN_SYNONYM_TO_CANONICAL = {
    "u ác tính": "u ác tính",  # canonical
    "u ác": "u ác tính",
    "k ác": "ung thư",
    "k": "ung thư",
    "ca": "ung thư",
    "khối u": "u",  # ICD desc_vi dùng "U" thay vì "Khối u"
    "khối u lành": "u lành tính",
    "u lành": "u lành tính",
    "viêm": "viêm",  # canonical
    "nhiễm trùng": "nhiễm khuẩn",
    "nhiễm khuẩn": "nhiễm khuẩn",  # canonical
}


def _get_alias_key_for_lookup(query: str) -> str | None:
    """R30 (2026-07-13): Anti-pollution guard cho L0 short-circuit.

    Trả về alias key nếu QUERY nên dùng key đó để exact-match L0.
    Returns None nếu query là COMPOUND text có chứa short alias bên trong.

    Logic:
    - key == query: returns query (exact match → always use)
    - len(query) >= 6 OR len(query) <= 3 (compound diagnosis phổ biến): returns query
    - SHORT alias (len 4-5) inside longer compound query (len >= 6):
        Returns None — let lower tiers (L1.5/L1.7) try specific compound.

    Examples:
        'sốt' → 'sốt'                          (key == query)
        'phù phổi' → None                       (len 8, contains short 'phù' 3 chars)
                                                  actually len(query) >= 6 → 'phù phổi' is acceptable
        'phù chân' → None → falls to L1+        (auto-mined 'phù' could pollute)
        'phù gai thị' → 'phù gai thị'           (NEW HARD-CODED entry exact match)
        'viêm gan b' → 'viêm gan b'             (6+ chars)
    """
    if not query:
        return None
    q = query.strip()
    if not q:
        return None
    # The bound match: query itself must equal the alias (no substring)
    # To allow "phù" alias to match "phù phổi" we'd need to require
    # word-boundary on LEFT and RIGHT of alias in query.
    # Simple heuristic: only allow short aliases if query IS short
    # OR query has at least 1 word boundary on both sides.
    # Conservative: short (<=5 char) aliases only match when query is the
    # whole short alias OR query is longer with whitespace/word boundaries.
    alias_like_keys = [
        "sốt", "ho", "phù", "hen", "nôn", "lao", "u", "ib", "ibs",
        "os", "osa", "sau", "hc", "pe",
        # R34 (2026-07-13): bổ sung short diagnoses (4-5 char) để không bị skip
        # trong L0 short-circuit. Trước đây các VN→code mappings cho 'ngất', 'btmv'
        # etc. bị miss do 4-5 char không có trong allowlist.
        "ngất", "ngừng", "hôn", "viêm", "gút", "bướu", "u ác", "ung",
        "btmv", "bptnmt", "vgb", "vgc", "rlntn", "nttn", "nttt", "đtrđ", "vpmpccđ", "st chênh",
    ]
    # If query is a known-short alias used standalone → return query
    if q in alias_like_keys:
        return q
    # Long-enough query (≥ 6 chars) → fine to return
    if len(q) >= 6:
        return q
    # 4-5 char short query that's NOT in our allowlist → suspect
    return None


def _normalize_vn_term(text: str) -> str:
    """Normalize VN medical text trước ICD lookup.

    Áp dụng các bước:
    1. Lowercase + strip
    2. Replace abbreviation VN → full term (vd "THA" → "tăng huyết áp")
    3. Replace synonym → canonical (vd "u ác" → "u ác tính")

    Args:
        text: VN medical text từ LLM (có thể có abbreviation/synonym).

    Returns:
        Normalized text để feed vào lookup chain.

    Bug history:
    - v1: 'st' abbreviation conflict với ECG term 'ST' segment.
      Fix: xóa 'st'/'stc'/'stm' khỏi _VN_ABBREVIATIONS để tránh normalize
      'ST chênh lên' thành 'suy thận chênh lên'.
    """
    if not text:
        return text
    s = text.strip()
    s_lower = s.lower()

    # 1. Abbreviation match (exact or substring)
    # Ưu tiên match exact trước, sau đó substring
    if s_lower in _VN_ABBREVIATIONS:
        return _VN_ABBREVIATIONS[s_lower]

    # 2. Try matching each abbreviation as substring (length-sorted to match longest first)
    # CHI MATCH substring nếu abbreviation dài ít nhất 3 ký tự để tránh false positive
    # ("st" 2 ký tự có thể match 'st' trong 'ST chênh lên' — sai!)
    sorted_abbrs = sorted(_VN_ABBREVIATIONS.items(), key=lambda x: -len(x[0]))
    for abbr, full in sorted_abbrs:
        if len(abbr) < 3:  # Skip very short abbreviations (2-char) to avoid false positives
            continue
        import re as _re
        pattern = _re.compile(r"\b" + _re.escape(abbr) + r"\b")
        if pattern.search(s_lower):
            s_lower = pattern.sub(full, s_lower)

    # 3. Synonym replacement (whole word)
    import re as _re
    for syn, canonical in _VN_SYNONYM_TO_CANONICAL.items():
        if syn == canonical:
            continue  # skip if same
        pattern = _re.compile(r"\b" + _re.escape(syn) + r"\b")
        s_lower = pattern.sub(canonical, s_lower)

    return s_lower


# ---------------------------------------------------------------------- #
# Data structures
# ---------------------------------------------------------------------- #




# ---------------------------------------------------------------------- #
# Data structures
# ---------------------------------------------------------------------- #


@dataclass
class ICDIndex:
    """Index ICD-10 local (gồm cả EN name và code)."""

    # name_en.lower() -> [code]
    exact: dict[str, list[str]] = field(default_factory=dict)
    # name gốc để fuzzy
    names: list[str] = field(default_factory=list)
    codes: list[str] = field(default_factory=list)
    name_to_idx: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"exact": self.exact, "names": self.names, "codes": self.codes}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ICDIndex":
        idx = cls(
            exact=data.get("exact", {}),
            names=data.get("names", []),
            codes=data.get("codes", []),
        )
        idx.name_to_idx = {n: i for i, n in enumerate(idx.names)}
        return idx

    def add(self, code: str, name: str) -> None:
        """Thêm 1 entry (code, name EN)."""
        key = name.lower().strip()
        self.exact.setdefault(key, []).append(code)
        if name not in self.name_to_idx:
            self.name_to_idx[name] = len(self.names)
            self.names.append(name)
            self.codes.append(code)


# ---------------------------------------------------------------------- #
# Translation
# ---------------------------------------------------------------------- #




# ---------------------------------------------------------------------- #
# ICD lookup (offline — không gọi NIH API)
# ---------------------------------------------------------------------- #


def _filter_irrelevant_codes(
    codes: list[str],
    entity_text: str,
    index=None,
) -> list[str]:
    """Filter ra các ICD codes rõ ràng không liên quan đến entity text.

    Áp dụng nguyên tắc chung: nếu entity không đề cập concept X mà code là về X,
    thì loại code đó.

    Hiện tại filter các patterns:
    - F10.x (alcohol-related): nếu entity không chứa "alcohol/rượu"
    - F11-F19 (drug-related): nếu entity không chứa "drug/chất"
    - T36-T50 (poisoning by drugs): nếu entity không phải poisoning/ngộ độc
    - V/W/X/Y (external causes): nếu entity không phải tai nạn/chấn thương
    - O00-O9A (pregnancy): nếu entity không phải pregnancy/mang thai

    Args:
        codes: list ICD codes (vd ["F10.159", "K72.9"])
        entity_text: VN hoặc EN text của diagnosis
        index: ICDIndex (optional) để lookup name từ code nếu cần

    Returns: filtered list codes.
    """
    if not codes:
        return codes

    entity_lower = entity_text.lower()
    out: list[str] = []

    for code in codes:
        # F10.x: alcohol-related
        if code.startswith("F10"):
            if any(kw in entity_lower for kw in (
                "alcohol", "rượu", "alcoholic", "ethanol", "liver",
            )):
                out.append(code)
            continue  # skip F10 if not alcohol-related

        # F11-F19: drug-related mental disorders
        if code.startswith(("F11", "F12", "F13", "F14", "F15", "F16", "F17", "F18", "F19")):
            if any(kw in entity_lower for kw in (
                "drug", "chất", "substance", "heroin", "cocaine", "amphetamine",
            )):
                out.append(code)
            continue

        # T36-T50: poisoning by drugs
        if code.startswith(("T36", "T37", "T38", "T39", "T40", "T41", "T42", "T43",
                            "T44", "T45", "T46", "T47", "T48", "T49", "T50")):
            if any(kw in entity_lower for kw in (
                "poisoning", "ngộ độc", "overdose", "quá liều", "toxic",
            )):
                out.append(code)
            continue

        # V/W/X/Y: external causes (accidents)
        if code[0] in ("V", "W", "X", "Y"):
            if any(kw in entity_lower for kw in (
                "accident", "tai nạn", "chấn thương", "injury", "trauma",
            )):
                out.append(code)
            continue

        # O00-O9A: pregnancy & obstetric conditions
        if code.startswith("O") and len(code) >= 2 and code[1].isdigit() and code[:2] < "O9":
            if any(kw in entity_lower for kw in (
                "pregnancy", "mang thai", "thai kỳ", "obstetric", "gestation",
                "chuyển dạ", "thai", "sản", "vỡ ối", "rỉ ối", "tiền sản giật", "sinh con",
            )):
                out.append(code)
            continue

        # P00-P96: Perinatal / Newborn conditions (chỉ dành cho sơ sinh / thai nhi)
        if code.startswith("P") and len(code) >= 2 and code[1].isdigit():
            if any(kw in entity_lower for kw in (
                "sơ sinh", "thai nhi", "newborn", "perinatal", "fetal", "fetus", "nhũ nhi",
            )):
                out.append(code)
            continue

        # Z00-Z99: Factors influencing health status (KHÔNG phải active diagnosis)
        # Drop theo mặc định; CHỈ giữ khi entity ngữ cảnh gợi ý family history / screening.
        if code.startswith("Z"):
            family_history_kws = (
                "tiền sử gia đình", "gia đình có", "tiền căn gia đình",
                "screening", "tầm soát", "vaccine", "tiêm chủng",
                "history of", "personal history", "family history",
            )
            if any(kw in entity_lower for kw in family_history_kws):
                out.append(code)
            continue  # default: drop Z

        # Default: keep
        out.append(code)

    return out


def build_context_query(
    entity_query: str,
    entity_type: str,
    other_entities: list[dict] | None = None,
) -> str:
    """Build BGE-M3 query với nearby context cho ICD retrieval.

    Kết hợp entity_text + nearby drugs/symptoms để disambiguate diagnosis
    (vd: amlodipine nearby → hypertension; polydipsia+polyuria → diabetes).

    Args:
        entity_query: câu truy vấn EN đã qua rescan (vd "essential hypertension").
        entity_type: CHẨN_ĐOÁN / THUỐC (chỉ áp dụng cho CHẨN_ĐOÁN).
        other_entities: list các entities khác trong note.

    Returns: enriched query string cho BGE-M3 embedding.
    """
    if not other_entities or entity_type != "CHẨN_ĐOÁN":
        return entity_query
    parts = [entity_query]
    drugs = [e.get("text", "") for e in other_entities if e.get("type") == "THUỐC"][:5]
    symps = [e.get("text", "") for e in other_entities if e.get("type") == "TRIỆU_CHỨNG"][:5]
    if drugs:
        parts.append("Patient on: " + ", ".join(drugs))
    if symps:
        parts.append("With symptoms: " + ", ".join(symps))
    return " | ".join(parts)


class ICDCrossEncoderReranker:
    """Re-rank top-K ICD candidates bằng BGE-M3 cosine giữa entity text và ICD description."""

    def __init__(self, encoder: Any) -> None:
        self.encoder = encoder

    def rerank(self, entity_text: str, candidates: list[str], icd_descriptions: dict[str, str], top_k: int = 1) -> list[str]:
        if not candidates or not hasattr(self, 'encoder') or not self.encoder:
            return candidates[:top_k] if candidates else []
        try:
            entity_emb = self.encoder.encode([entity_text.lower()], normalize_embeddings=True)[0]
            cand_texts = [f"{code}: {icd_descriptions.get(code, code)}" for code in candidates]
            cand_embs = self.encoder.encode(cand_texts, normalize_embeddings=True)
            import numpy as np
            scores = np.dot(cand_embs, entity_emb)
            ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
            return [c for c, _ in ranked[:top_k]]
        except Exception:
            return candidates[:top_k]


class ICDRetriever:
    """High-level wrapper: exact + semantic extraction + fuzzy (offline).

    Pipeline 4 lớp (chạy hoàn toàn local, KHÔNG gọi NIH API):
      L1: Exact match dict tiếng Việt (prebuilt, ICDIndex)
      L2: Semantic extraction (BGE-M3 cosine ≥ 0.7) — trả TẤT CẢ codes match,
            không cap top-K. BM25 keyword dùng để mở rộng candidates.
      L3: Fuzzy match EN (rapidfuzz, partial_ratio)
      L4: Fuzzy match VN (rapidfuzz, partial_ratio)

    Mặc định `use_hybrid=True`; truyền `use_hybrid=False` để fallback về vector-only.
    """

    def __init__(
        self,
        index_path: Optional[Path] = None,
        local_search: Optional["ICD10VectorSearch | ICD10HybridSearch"] = None,
        use_hybrid: bool = True,
        hybrid_alpha: float = 0.6,
        hybrid_beta: float = 0.4,
    ) -> None:
        self.idx = self._load_index(index_path)
        # local_search: nếu user truyền ICD10HybridSearch thì dùng trực tiếp;
        # nếu truyền ICD10VectorSearch mà use_hybrid=True thì auto-wrap;
        # nếu None và use_hybrid=True thì tạo hybrid mới (wrap vector mặc định).
        if local_search is None:
            if use_hybrid:
                vs = ICD10VectorSearch()
                self.local_search: Optional[ICD10HybridSearch] = ICD10HybridSearch(
                    vector_search=vs, alpha=hybrid_alpha, beta=hybrid_beta
                )
            else:
                self.local_search = ICD10VectorSearch()
        elif use_hybrid and isinstance(local_search, ICD10VectorSearch):
            self.local_search = ICD10HybridSearch(
                vector_search=local_search, alpha=hybrid_alpha, beta=hybrid_beta
            )
        elif use_hybrid and isinstance(local_search, ICD10HybridSearch):
            # user đã truyền hybrid rồi — giữ nguyên (kể cả alpha/beta user-config)
            self.local_search = local_search
        else:
            self.local_search = local_search  # type: ignore[assignment] -- caller opted out

        # Fix #6 (R27.7): Populate `_icd_vn_to_codes` dict NGAY trong __init__ để L0
        # short-circuit trong `lookup()` work ngay từ lần đầu (trước đó dict là dead code
        # nằm sau `return` của `_exact_match_vn_substring` nên không bao giờ được set).
        self._init_icd_vn_to_codes()

        self._code_to_desc = {}
        for c, n in zip(self.idx.codes, self.idx.names):
            if c not in self._code_to_desc:
                self._code_to_desc[c] = n.lower()
        if self.local_search is not None and hasattr(self.local_search, 'vector_search') and self.local_search.vector_search is not None:
            vs = self.local_search.vector_search
            if hasattr(vs, 'codes') and hasattr(vs, 'names'):
                for c, n in zip(vs.codes, vs.names):
                    if c not in self._code_to_desc:
                        self._code_to_desc[c] = n.lower()
            if hasattr(vs, 'model') and vs.model is not None:
                self._reranker = ICDCrossEncoderReranker(vs.model)
            else:
                self._reranker = None
        else:
            self._reranker = None
        self._llm_client = None

    # ------------------------------------------------------------------ #

    def _load_index(self, path: Optional[Path]) -> ICDIndex:
        path = path or (DATA_DIR / "icd_index.json")
        if not path.exists():
            # Fallback chain: icd10.jsonl mới → BYT → ICD10_Data cũ
            # NOTE: JSONL files LUÔN dùng local (Kaggle không có JSONL data source).
            seed_candidates = [
                DATA_DIR / "icd10.jsonl",                # WHO 2019 VN+EN (mới nhất)
                DATA_DIR / "DM_ICD10_19_8_BYT.json",    # BYT chính thức (lớn nhất)
                DATA_DIR / "ICD10_Data.json",           # cũ
            ]
            seed_path = next((c for c in seed_candidates if c.exists()), None)
            if seed_path:
                logger.info("icd_index.json không có, build từ %s", seed_path.name)
                return build_from_seed(seed_path)
            return ICDIndex()
        try:
            return ICDIndex.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            return ICDIndex()

    def save_index(self, path: Optional[Path] = None) -> None:
        path = path or (DATA_DIR / "icd_index.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.idx.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Saved ICD index → %s (%d entries)", path, len(self.idx.names))

    # ------------------------------------------------------------------ #

    def _filter_and_sort_codes(self, codes: list[str], text: str, other_entities: list[dict] | None = None, entity_type: str = "") -> list[str]:
        if not codes:
            return []
        filtered = _filter_irrelevant_codes(list(codes), text, self.idx)
        if not filtered:
            return []
        restricted = _restrict_chapter(filtered, text)
        result = restricted if restricted else filtered

        if not _text_matches_chapter_keyword(text):
            inferred = self._infer_chapter_from_other_entities(other_entities)
            if inferred:
                pref_filtered = [c for c in result if c.startswith(inferred)]
                if pref_filtered:
                    result = pref_filtered

        boosted_prefixes = _get_boosted_prefixes(other_entities) if entity_type != "TRIỆU_CHỨNG" else set()

        # Smart sorting: ưu tiên các chương bệnh phổ biến cho người lớn (I, J, K, E, N, M, S, T, C, D, G, A, B, R)
        # đẩy O (thai sản), P (sơ sinh), V/W/X/Y (tác nhân bên ngoài), Z (tiền căn) xuống dưới nếu không rõ
        def _chapter_priority(code: str) -> tuple[int, str]:
            if not code:
                return (99, code)
            ch = code[0].upper()
            # Với TRIỆU_CHỨNG → ưu tiên chapter R (Signs & Symptoms) trước tất cả
            if entity_type == "TRIỆU_CHỨNG" and ch == 'R':
                return (0, code)
            if ch in ('I', 'J', 'K', 'E', 'N', 'M', 'C', 'D', 'G', 'A', 'B', 'R', 'S', 'T'):
                return (1, code)
            if ch in ('L', 'H', 'F'):
                return (2, code)
            return (3, code)

        text_lower = text.lower()
        # Universal non-hardcoded lexical overlap & contradiction check
        text_tokens = set(re.findall(r'[a-zà-ỹ0-9_/-]{2,}', text_lower))
        stop_words = {"và", "của", "khi", "cho", "tại", "bị", "có", "do", "các", "những", "lần", "ngày", "bệnh", "chứng", "tình", "trạng", "không", "chưa", "hoặc", "hay", "là", "với", "trong"}
        core_text_tokens = _expand_tokens_with_synonyms(text_tokens - stop_words, text_lower)

        # Extract discriminative tokens: single letters (a, b, c...), digits (1, 2, 3...), Roman numerals (i, ii, iii, iv...)
        disc_tokens = set(re.findall(r'\b(?:[a-z]|\d+|ii+|iv|vi*)\b', text_lower)) - {"i", "a"} # ignore 'i'/'a' if used as grammar

        def _lex_score(code: str) -> tuple[float, int, int]:
            desc = self._code_to_desc.get(code, "").lower()
            desc_tokens = set(re.findall(r'[a-zà-ỹ0-9_/-]{2,}', desc))
            penalty = 0.0

            # 0. Drug <-> Disease Co-occurrence Cross-Scoring (Upgrade 3)
            if boosted_prefixes and any(code.startswith(bp) for bp in boosted_prefixes):
                penalty -= 2.5

            # 1. Universal Discriminative Token Match (Letters, Numbers, Roman numerals) - ZERO Disease Hardcoding
            if disc_tokens:
                desc_disc = set(re.findall(r'\b(?:[a-z]|\d+|ii+|iv|vi*)\b', desc)) - {"i", "a"}
                for dt in disc_tokens:
                    if dt in desc_disc:
                        penalty -= 1.5
                    elif desc_disc and not (disc_tokens & desc_disc):
                        # Candidate has a DIFFERENT digit/letter/Roman numeral than input text -> heavy penalty
                        penalty += 2.5

            # 2. General Linguistic Antonym Check (Zero Disease Names)
            antonym_pairs = [
                ("cấp", "mạn"), ("cấp", "mãn"),
                ("trái", "phải"), ("trái", "hai bên"), ("phải", "hai bên"),
                ("trên", "dưới"), ("trong", "ngoài"),
                ("lành tính", "ác tính"), ("lành", "ác"),
            ]
            for w1, w2 in antonym_pairs:
                if re.search(r'\b' + re.escape(w1) + r'\b', text_lower):
                    if re.search(r'\b' + re.escape(w1) + r'\b', desc): penalty -= 1.2
                    if re.search(r'\b' + re.escape(w2) + r'\b', desc): penalty += 2.5
                elif re.search(r'\b' + re.escape(w2) + r'\b', text_lower):
                    if re.search(r'\b' + re.escape(w2) + r'\b', desc): penalty -= 1.2
                    if re.search(r'\b' + re.escape(w1) + r'\b', desc): penalty += 2.5

            # 3. Token overlap ratio (Jaccard similarity across all 14,000+ codes)
            if core_text_tokens and desc_tokens:
                overlap = len(core_text_tokens & desc_tokens)
                union = len(core_text_tokens | desc_tokens)
                jaccard = overlap / max(union, 1)
                penalty -= jaccard * 1.5

            # 4. Universal Neoplasm check: nếu text nhắc đến u/khối u/bướu/ung thư → ưu tiên chương C và D00-D48
            if re.search(r'\b(?:u|khối\s+u|bướu|ung\s+thư|k)\b', text_lower):
                if code.startswith('C') or (code.startswith('D') and code[1:3].isdigit() and int(code[1:3]) <= 48):
                    penalty -= 0.5

            # Với TRIỆU_CHỨNG: ưu tiên R-chapter code hàng đầu, giảm penalty cho M/S/T
            if entity_type == "TRIỆU_CHỨNG":
                if code[0].upper() == 'R':
                    penalty -= 1.5  # bonus mạnh cho chapter R (triệu chứng)
                elif code[0].upper() in ('M', 'S', 'T', 'L'):
                    penalty += 1.0  # penalty nhẹ cho musculo-skeletal/injury khi lookup triệu chứng

            idx = result.index(code) if code in result else 999
            return (penalty, 0 if code[0].upper() in ('I', 'J', 'K', 'E', 'N', 'M', 'C', 'D', 'G', 'A', 'B', 'R', 'S', 'T') else 1, idx)

        return sorted(set(result), key=_lex_score)

    def _infer_chapter_from_other_entities(self, other_entities: list[dict] | None) -> str:
        if not other_entities:
            return ""
        drug_prefixes = _get_boosted_prefixes(other_entities)
        if not drug_prefixes:
            return ""
        from collections import Counter
        pref_count = Counter(p[:3] for p in drug_prefixes)
        most_common = pref_count.most_common(1)
        return most_common[0][0] if most_common else ""

    def _rerank_and_select(self, codes: list[str], text: str, max_k: int = 1) -> list[str]:
        if not codes:
            return []
        if hasattr(self, '_reranker') and self._reranker and len(codes) > 1:
            codes = self._reranker.rerank(text, codes[:10], getattr(self, '_code_to_desc', {}), top_k=max(3, max_k))
        return self._select_adaptive_top_k(codes, max_k=max_k, text=text)

    def _select_adaptive_top_k(self, codes: list[str], max_k: int = 1, text: str = "") -> list[str]:
        """Mặc định 1 code để tối ưu Jaccard. Chỉ giữ thêm nếu cùng 3-char prefix."""
        if not codes:
            return []

        # R28 (2026-07-13): PRECISION GUARD — drop uninformative ICD codes trước khi pick top.
        # Catch-all codes (R69, T45.xx, Y5x.xx, Z00.xx, 3-char chapters) làm giảm J_candidates
        # precision rất nặng. Pattern load từ data/generic_class_stoplist.json để dễ mở rộng.
        filtered = [c for c in codes if not _is_uninformative_icd_code(c)]
        if not filtered:
            return []  # tất cả catch-all → KHÔNG trả candidate gì (precision > recall)
        codes = filtered

        if len(codes) == 1:
            return codes
        out = [codes[0]]
        top_prefix_3 = codes[0][:3]
        for c in codes[1:max_k]:
            if c[:3] == top_prefix_3 and c not in out:
                out.append(c)
        if len(out) == 1 and max_k == 1 and text and len(codes) > 1:
            # Specificity-Aware Picker (Super-Upgrade 4)
            # Nếu text có từ khóa độ specific cao mà trong top-3 candidates có mã dài hơn cùng prefix (vd I21.1 thay vì I21) thì ưu tiên mã dài
            spec_keywords = ("vùng", "dưới", "trước", "sau", "bên", "độ 1", "độ i", "độ 2", "độ ii", "độ 3", "độ iii", "độ 4", "độ iv", "mạn tính", "cấp tính", "cấp", "mạn", "nhánh", "kịch phát", "giai đoạn cuối", "thùy")
            text_lower = text.lower()
            if any(k in text_lower for k in spec_keywords):
                for cand in codes[1:4]:
                    if cand[:3] == top_prefix_3 and len(cand) > len(out[0]):
                        out = [cand]
                        break
        return out

    def _cap_single_for_icd(self, codes: list[str]) -> list[str]:
        """R29 (spec round 2): CAP ≥1 cho single ICD lookup.

        User rule: "Ưu tiên candidate đơn. Không ném nhiều mã 'để chắc', vì Jaccard phạt candidate dư."

        Khi caller KHÔNG opt-in (max_k implicit = 1), return list ≤1 code.
        Caller có thể bypass bằng cách explicit max_k > 1 nếu thực sự cần multi-candidate
        (vd compound diagnosis).

        Args:
            codes: candidate list từ _select_adaptive_top_k

        Returns:
            codes capped to length 1 if non-empty.
        """
        if not codes:
            return codes
        return codes[:1]  # Section 8: cap=1 for single ICD call

    def _split_compound_diagnosis(self, text: str) -> list[str]:
        """Tách chẩn đoán kép (Multi-hop / Conjunction splitting) thành các chẩn đoán riêng lẻ.

        R34 (2026-07-13): bổ sung 'và' (AND conjunction) — trước đây chỉ có
        'hoặc'/'hay' (OR). Cho phép split 'đau ngực và khó thở' → ['đau ngực', 'khó thở'].
        """
        parts = re.split(r'\s+trên\s+nền\s+|\s+kèm\s+theo\s+|\s+kèm\s+|\s+biến\s+chứng\s+|\s+đồng\s+thời\s+|\s+hoặc\s+|\s+hay\s+|\s+và\s+', text, flags=re.IGNORECASE)
        if len(parts) == 1 and (' - ' in text or ' / ' in text):
            sub = re.split(r'\s+-\s+|\s+/\s+', text)
            if all(len(p.strip()) >= 4 for p in sub):
                parts = sub
        cleaned_parts = [p.strip().rstrip(',;. -') for p in parts if len(p.strip()) >= 3]
        return cleaned_parts if len(cleaned_parts) > 1 else [text]

    def lookup(
        self,
        vn_text: str,
        context_query: str | None = None,
        other_entities: list[dict] | None = None,
        entity_type: str = "",
    ) -> list[str]:
        """Tra ICD-10 cho 1 cụm chẩn đoán tiếng Việt (có tự động tách chẩn đoán kép)."""
        if not vn_text:
            return []
        # R34: Strip resistance suffix trước (vd "...kháng thuốc" → core diagnosis).
        # Chỉ strip khi suffix ở CUỐI text (R34: surgical precision).
        stripped_vn_text = _strip_resistance_suffix(vn_text)
        if stripped_vn_text != vn_text:
            vn_text = stripped_vn_text
        # R34: L_filter_context — reject resistance mentions + drug-class blacklisted terms
        if _has_resistance_context_icd(vn_text):
            logger.debug("R34: resistance context rejected: '%s'", vn_text)
            return []
        if _is_drug_class_term_icd(vn_text):
            logger.debug("R34: drug-class blacklist rejected: '%s'", vn_text)
            return []
        if _is_uninformative_vn_term(vn_text):
            logger.debug("R34: uninformative VN term rejected: '%s'", vn_text)
            return []
        if _looks_like_noise_tokens(vn_text):
            logger.debug("R34: noise tokens rejected: '%s'", vn_text)
            return []
        # Universal Vietnamese Clinical Acronym Resolution Map (Zero Hardcoding to specific files, covering standard Vietnamese clinical abbreviations):
        clean_norm = vn_text.strip().lower()
        acronym_map = {
            "rlll": "rối loạn lipid máu",
            "tha": "tăng huyết áp",
            "đtđ": "đái tháo đường",
            "dtd": "đái tháo đường",
            "đtđ tuýp 2": "đái tháo đường tuýp 2",
            "đtđ tuýp 1": "đái tháo đường tuýp 1",
            "nmct": "nhồi máu cơ tim",
            "btmv": "bệnh tim mạch vành",
            "tbmmn": "tai biến mạch máu não",
            "ckd": "bệnh thận mạn",
            "copd": "bệnh phổi tắc nghẽn mạn tính",
            "bptnmt": "bệnh phổi tắc nghẽn mạn tính",
            "vgb": "viêm gan b",
            "vgc": "viêm gan c",
            "rlntn": "rối loạn nhịp tim nhĩ",
            "nttn": "ngoại tâm thu nhĩ",
            "nttt": "ngoại tâm thu thất",
            "đtrđ": "đái tháo đường",
            "xơ gan rđ": "xơ gan rượu",
            "nkh": "nhiễm khuẩn huyết",
            "vtp": "viêm thận bể thận",
            "hcth": "hội chứng thận hư",
            "bkm": "bệnh cơ tim",
            "st chênh lên": "nhồi máu cơ tim có st chênh lên",
            "st không chênh lên": "nhồi máu cơ tim không có st chênh lên"
        }
        if clean_norm in acronym_map:
            vn_text = acronym_map[clean_norm]

        if not hasattr(self, '_cache'):
            self._cache = {}
        other_key = tuple(sorted((str(e.get("text", "")).strip().lower(), str(e.get("type", ""))) for e in (other_entities or []))) if other_entities else ()
        cache_key = (vn_text.strip().lower(), context_query, other_key, entity_type)
        if cache_key in self._cache:
            return list(self._cache[cache_key])

        # R34 (2026-07-13): check full text against VN dict FIRST (before split).
        # Compound forms như 'mất kiểm soát đại tiện hoặc tiểu tiện' có direct
        # mapping → ['R15', 'R32']. Nếu split trước, mỗi part resolve độc lập và
        # có thể miss codes (vd 'tiểu tiện' alone → R30.9 thay vì R32).
        if hasattr(self, '_icd_vn_to_codes'):
            full_key = vn_text.lower().strip()
            if full_key in self._icd_vn_to_codes:
                resolved = self._icd_vn_to_codes[full_key]
                filtered = self._filter_and_sort_codes(
                    list(resolved), vn_text,
                    other_entities=other_entities, entity_type=entity_type,
                )
                if not filtered:
                    filtered = list(resolved)
                # R34: BYPASS _select_adaptive_top_k (chỉ giữ same-prefix siblings)
                # cho full-compound → trả ALL resolved codes. User rule: "nếu có các từ
                # hoặc/và/kèm/trên nền phải in ra đủ ICD".
                logger.debug("R34: full-compound direct: '%s' → %s", vn_text, filtered)
                result = filtered
                if len(self._cache) > 4096:
                    self._cache.clear()
                self._cache[cache_key] = result
                return list(result)

        parts = self._split_compound_diagnosis(vn_text)
        if len(parts) > 1 and len(parts) <= 5:
            logger.debug("Multi-hop splitting '%s' → %s", vn_text, parts)
            out = []
            for p in parts:
                codes = self._lookup_single(p, context_query, other_entities, entity_type=entity_type)
                for c in codes:
                    if c not in out:
                        out.append(c)
            # R34 (2026-07-13): REMOVED cap `out[:2]`. Compound diagnoses với "hoặc" /
            # "và" / "kèm" / "trên nền" nên trả về TẤT CẢ codes (mỗi part một code).
            # User rule: "nếu có các từ đó phải in ra đủ ICD" — không cap 2 nữa.
            result = out
        else:
            result = self._lookup_single(vn_text, context_query, other_entities, entity_type=entity_type)

        if len(self._cache) > 4096:
            self._cache.clear()
        self._cache[cache_key] = result
        return list(result)

    def _lookup_single(
        self,
        vn_text: str,
        context_query: str | None = None,
        other_entities: list[dict] | None = None,
        entity_type: str = "",
    ) -> list[str]:
        if not vn_text:
            return []

        text = self._strip_clinical_prefix(vn_text)
        # R27.6 mới 2026-07-10: normalize abbreviation + synonym VN TRƯỚC lookup chain
        text = _normalize_vn_term(text)

        # R27.7 mới 2026-07-10: short-circuit khi có direct match trong _icd_vn_to_codes
        # R30 (2026-07-13): ANTI-POLLUTION GUARD — short aliases (< 6 chars) require
        # word-boundary match OR exact equality. Otherwise the L0 short-circuit
        # would over-fire on substrings (vd "phù" alias matching "phù chân" → R60
        # instead of leaving the L1.5/L1.7 chain find specific R60.0).
        if hasattr(self, '_icd_vn_to_codes'):
            key_lower = text.lower().strip()
            if key_lower in self._icd_vn_to_codes:
                # Word-boundary check for short aliases only
                alias_key = _get_alias_key_for_lookup(key_lower)
                if alias_key is None:
                    pass  # short alias but no word-boundary match → skip L0, let chain find right code
                else:
                    logger.debug("L0 short-circuit direct match: '%s' → %s", text, self._icd_vn_to_codes[alias_key])
                    resolved_codes = self._icd_vn_to_codes[alias_key]
                    filtered = self._filter_and_sort_codes(resolved_codes, text, other_entities=other_entities, entity_type=entity_type)
                    return self._rerank_and_select(filtered if filtered else resolved_codes, max_k=1, text=text)

            # Tier-1b: Prefix & Word-containment fallback trong _icd_vn_to_codes (Fix 1.3)
            for key, codes in self._icd_vn_to_codes.items():
                if len(key) >= 8 and key_lower.startswith(key):
                    filtered = self._filter_and_sort_codes(codes, text, other_entities=other_entities, entity_type=entity_type)
                    if filtered:
                        return self._rerank_and_select(filtered, max_k=1, text=text)
            text_words = f" {key_lower} "
            for key, codes in self._icd_vn_to_codes.items():
                if len(key) >= 6 and f" {key} " in text_words:
                    filtered = self._filter_and_sort_codes(codes, text, other_entities=other_entities, entity_type=entity_type)
                    if filtered:
                        return self._rerank_and_select(filtered, max_k=1, text=text)

        # L1: Exact (cao độ tin cậy nhất — cap 1)
        key = text.lower()
        if key in self.idx.exact:
            filtered = self._filter_and_sort_codes(self.idx.exact[key], text, other_entities=other_entities, entity_type=entity_type)
            return self._rerank_and_select(filtered if filtered else self.idx.exact[key], max_k=1, text=text)

        # L1.5: VN prefix exact match — nếu text là prefix của desc_vi
        prefix_codes = self._exact_match_vn_prefix(text)
        if prefix_codes:
            filtered = self._filter_and_sort_codes(prefix_codes, text, other_entities=other_entities, entity_type=entity_type)
            if filtered:
                return self._rerank_and_select(filtered, max_k=1, text=text)

        # L1.7 (NEW 2026-07-10): VN substring match (text chứa trong desc_vi)
        if len(text) >= 5:
            substring_codes = self._exact_match_vn_substring(text)
            if substring_codes:
                filtered = self._filter_and_sort_codes(substring_codes, text, other_entities=other_entities, entity_type=entity_type)
                if filtered:
                    logger.debug("L1.7 substring match '%s' → %s", text, filtered[:2])
                    return self._rerank_and_select(filtered, max_k=1, text=text)

        # L2: Build query có context cho BM25 (dùng nearby drugs/symptoms)
        bm25_query = build_context_query(text, "CHẨN_ĐOÁN", other_entities)
        # Vector vẫn dùng text gốc (không contaminate embedding - bug history #4)

        # L3: Hybrid RRF search (Vector BGE-M3 + BM25 Reciprocal Rank Fusion - Upgrade B):
        if self.local_search is not None:
            queries_to_try = [text]
            normalized = _normalize_vn_term(text)
            if normalized != text:
                queries_to_try.append(normalized)

            rrf_scores: dict[str, float] = {}
            for q in queries_to_try:
                matched_vec = self.local_search.search(q, threshold=0.50, top_k=15) or []
                for rank, code in enumerate(matched_vec, start=1):
                    rrf_scores[code] = rrf_scores.get(code, 0.0) + (1.0 / (60 + rank))

                if hasattr(self.local_search, 'bm25_index'):
                    bm25_codes, _ = self.local_search.bm25_index.search(bm25_query, top_k=15)
                    for rank, code in enumerate(bm25_codes or [], start=1):
                        rrf_scores[code] = rrf_scores.get(code, 0.0) + (1.0 / (60 + rank))

            if rrf_scores:
                sorted_codes = sorted(rrf_scores.keys(), key=lambda c: -rrf_scores[c])
                filtered = self._filter_and_sort_codes(sorted_codes[:8], text, other_entities=other_entities, entity_type=entity_type)
                if filtered:
                    return self._rerank_and_select(filtered, max_k=1, text=text)

        # L4: Local fuzzy match trên names VN (threshold 75, cap 1)
        fuzzy_vn = self._fuzzy_local(text, threshold=75)
        if fuzzy_vn:
            filtered = self._filter_and_sort_codes(fuzzy_vn, text, other_entities=other_entities, entity_type=entity_type)
            if filtered:
                return self._rerank_and_select(filtered, max_k=1, text=text)

        # L5: BM25 fallback (top-1 — adaptive)
        if self.local_search is not None and hasattr(self.local_search, 'bm25_index'):
            bm25_codes, _ = self.local_search.bm25_index.search(bm25_query, top_k=4)
            if bm25_codes:
                bm25_codes = self._filter_and_sort_codes(bm25_codes, text, other_entities=other_entities, entity_type=entity_type)
                if bm25_codes:
                    return self._rerank_and_select(bm25_codes, max_k=1, text=text)

        # L6 (NEW R27.7 2026-07-10): Aggressive final fallback - thử MULTIPLE strategies
        if len(text) >= 5:
            substring_codes = self._exact_match_vn_substring(text)
            if substring_codes:
                filtered = self._filter_and_sort_codes(substring_codes, text, other_entities=other_entities, entity_type=entity_type)
                if filtered:
                    logger.debug("L6 substring fallback '%s' → %s", text, filtered[:2])
                    return self._rerank_and_select(filtered, max_k=1, text=text)

        if _text_matches_chapter_keyword(text):
            prefix_codes = self._exact_match_vn_prefix(text)
            if prefix_codes:
                filtered = self._filter_and_sort_codes(prefix_codes, text, other_entities=other_entities, entity_type=entity_type)
                if filtered:
                    return self._rerank_and_select(filtered, max_k=1, text=text)
            chapter_codes = self._chapter_codes_lookup(text)
            if chapter_codes:
                filtered = self._filter_and_sort_codes(chapter_codes, text, other_entities=other_entities, entity_type=entity_type)
                if filtered:
                    logger.debug("L6 chapter lookup '%s' → %s", text, filtered[:2])
                    return self._rerank_and_select(filtered, max_k=1, text=text)

        if self.local_search is not None:
            low_threshold_codes = self.local_search.search(
                text, threshold=0.40, top_k=6
            ) or []
            if low_threshold_codes:
                filtered = self._filter_and_sort_codes(low_threshold_codes, text, other_entities=other_entities, entity_type=entity_type)
                if filtered:
                    logger.debug("L6 low-threshold vector '%s' → %s", text, filtered[:2])
                    return self._rerank_and_select(filtered, max_k=1, text=text)

        # L7: LLM Fallback (Strict Validated) khi tất cả tiers RAG đều empty
        if hasattr(self, '_llm_client') and self._llm_client:
            try:
                from src.prompts import ICD_LLM_FALLBACK_PROMPT
                context = ""
                if other_entities:
                    ctx_entities = [e.get("text", "") for e in other_entities[:3] if isinstance(e, dict)]
                    context = " | Nearby: " + ", ".join(ctx_entities)
                prompt = ICD_LLM_FALLBACK_PROMPT.format(
                    entity_text=vn_text,
                    context_window=context,
                )
                response = self._llm_client.call_sync(prompt, max_tokens=50, temperature=0.1)
                import re as _re
                codes = _re.findall(r'\b([A-TV-Z]\d{2}(?:\.\d{1,2})?)\b', response.upper())
                valid_codes = []
                for c in codes:
                    if not c.startswith(('U', 'V', 'W', 'X', 'Y')) and c in self.idx.codes:
                        valid_codes.append(c)
                if valid_codes:
                    logger.info("L7 LLM fallback ICD: '%s' → %s", vn_text, valid_codes[:1])
                    return self._rerank_and_select(valid_codes, max_k=1, text=text)
            except Exception as exc:
                logger.warning("L7 LLM fallback ICD failed: %s", exc)

        return []  # noqa: RET504

    # ------------------------------------------------------------------ #

    def _fuzzy_local(self, query: str, *, threshold: int) -> list[str]:
        if not query:
            return []
        try:
            from rapidfuzz import fuzz, process  # type: ignore
        except ImportError:
            return []
        # Trích các token y khoa (bỏ stop word)
        stop = {
            "the",
            "of",
            "and",
            "or",
            "with",
            "without",
            "in",
            "to",
            "a",
            "an",
            "unspecified",
            "type",
            "stage",
            "acute",
            "chronic",
            "primary",
            "secondary",
        }
        tokens = [t for t in re.split(r"[^a-z]+", query.lower()) if t and t not in stop]
        if not tokens:
            return []
        q = " ".join(tokens)

        matches = process.extract(q, self.idx.names, scorer=fuzz.WRatio, limit=5)
        matches += process.extract(
            q, self.idx.names, scorer=fuzz.partial_ratio, limit=5
        )

        seen_names: set[str] = set()
        out: list[str] = []
        for name, score, _ in matches:
            if name in seen_names:
                continue
            seen_names.add(name)
            if score < threshold:
                continue
            if name not in self.idx.name_to_idx:
                continue
            code = self.idx.codes[self.idx.name_to_idx[name]]
            if code not in out:
                out.append(code)
        return out

    # ------------------------------------------------------------------ #

    def _exact_match_vn_prefix(self, text: str) -> list[str]:
        """Match VN prefix: nếu text là prefix (case-insensitive) của bất kỳ
        desc_vi nào trong index → return codes tương ứng.

        Fix L73.2 case: entity "viêm tuyến mồ hôi" không match exact
        "viêm tuyến mồ hôi mủ [nhọt ổ gà]" vì thiếu "mủ [nhọt ổ gà]".
        Nhưng text là prefix của desc_vi → match L73.2.

        Min prefix length = 8 chars để tránh false positive ("viêm" match
        hàng trăm codes khác nhau).
        """
        if not text or len(text) < 8:
            return []
        text_lower = text.lower().strip()
        import re as _re
        out: list[str] = []
        for name, idx in self.idx.name_to_idx.items():
            name_lower = name.lower().strip()
            # Strip bracketed context "[...]" và "(...)" để so sánh prefix
            name_clean = _re.sub(r"\s*[\[\(].*?[\]\)]\s*", " ", name_lower).strip()
            if name_clean.startswith(text_lower):
                code = self.idx.codes[idx]
                if code not in out:
                    out.append(code)
        return out

    def _exact_match_vn_substring(self, text: str) -> list[str]:
        """Match VN substring: tìm codes có desc_vi chứa text (substring match).

        Fix ICD candidates SAI (R27.2, 2026-07-09):
        - Vector search trả codes không thuộc chapter đúng (vd "ung thư phổi" → B21.0)
        - Chapter restriction `_restrict_chapter(kept, text)` → restricted = []
        - Fallback: thử VN prefix exact match (nếu desc_vi bắt đầu bằng text)
        - Nếu prefix rỗng → RE-SEARCH bằng substring (desc_vi chứa text)

        Ví dụ:
        - "Ung thư phổi" → desc_vi có "Ung thư phổi không tế bào nhỏ"
          → substring match → code C34.x (KHÔNG phải B21.0)
        - "Tăng huyết áp" → desc_vi có "Tăng huyết áp vô căn"
          → substring match → code I10 (KHÔNG phải I15)
        - "Di căn não" → desc_vi có "Di căn não"
          → substring match → code C79.31 (KHÔNG phải D56.4)

        Min substring length = 5 chars để tránh false positive (vd "đau" match nhiều).
        """
        if not text or len(text) < 5:
            return []
        text_lower = text.lower().strip()
        import re as _re
        out: list[str] = []
        for name, idx in self.idx.name_to_idx.items():
            name_lower = name.lower().strip()
            name_clean = _re.sub(r"\s*[\[\(].*?[\]\)]\s*", " ", name_lower).strip()
            # Substring match (text là 1 phần của desc_vi)
            if text_lower in name_clean:
                code = self.idx.codes[idx]
                if code not in out:
                    out.append(code)
        return out

    def _init_icd_vn_to_codes(self) -> None:
        # ICD direct mapping (R27.5): VN → exact ICD codes.
        # desc_en BYT không exact-match bản dịch EN nên cần map trực tiếp.
        # R28 (2026-07-13): MERGE auto-mined aliases từ data/icd_aliases.json
        # (build_mining_index.py) để exact match VN synonyms (bệnh Hansen, EHEC, ...)
        # — KHÔNG phải hand-curate.
        self._icd_vn_to_codes = {
            "ung thư phổi": ["C34", "C34.9"],
            "ung thư phổi không tế bào nhỏ": ["C34", "C34.9"],
            "ung thư phổi tế bào nhỏ": ["C34", "C34.9"],
            "u ác tính phổi": ["C34", "C34.9"],
            "k phổi": ["C34", "C34.9"],
            "ung thư não": ["C71", "C71.9"],
            "u não": ["C71", "C71.9"],
            "u ác tính não": ["C71", "C71.9"],
            "di căn não": ["C79.3"],
            "u ác tính thứ phát ở não": ["C79.3"],
            "di căn xương": ["C79.5"],
            "di căn gan": ["C78.7"],
            "di căn phổi": ["C78.0"],
            "di căn": ["C79", "C79.9"],
            "ung thư vú": ["C50", "C50.9"],
            "ung thư gan": ["C22", "C22.9"],
            "ung thư dạ dày": ["C16", "C16.9"],
            "ung thư đại tràng": ["C18", "C18.9"],
            "ung thư trực tràng": ["C20"],
            "tăng huyết áp": ["I10"],
            "tăng huyết áp vô căn": ["I10"],
            "tăng huyết áp thứ phát": ["I15"],
            "cao huyết áp": ["I10"],
            "nhồi máu cơ tim": ["I21", "I21.9"],
            "đau thắt ngực": ["I20", "I20.9"],
            "suy tim": ["I50", "I50.9"],
            "rung nhĩ": ["I48", "I48.9"],
            "ngoại tâm thu thất": ["I49.3"],
            "ngoại tâm thu nhĩ": ["I49.1"],
            "sa van hai lá": ["I34.1"],
            "sa van 2 lá": ["I34.1"],
            "sa van mitral": ["I34.1"],
            "hở van hai lá": ["I34.0"],
            "hở van 2 lá": ["I34.0"],
            "hẹp van hai lá": ["I34.2"],
            "tắc mạch huyết khối": ["I82", "I82.9"],
            "tắc mạch": ["I82", "I82.9"],
            "huyết khối": ["I82"],
            "hen phế quản": ["J45", "J45.9"],
            "hen suyễn": ["J45", "J45.9"],
            "viêm phổi": ["J18", "J18.9"],
            "viêm tuyến mồ hôi": ["L73.2"],  # mới 2026-07-10: R27.7 — match L73.2 (Hidradenitis suppurativa)
            "viêm tuyến mồ hôi mủ": ["L73.2"],
            "nhọt ổ gà": ["L73.2"],
            "đái tháo đường": ["E11", "E11.9"],
            "đái tháo đường type 2": ["E11"],
            "suy thận": ["N19"],
            "suy thận cấp": ["N17"],
            "suy thận mạn": ["N18", "N18.9"],
            "viêm gan b": ["B16", "B18.1"],
            "viêm gan c": ["B17.1", "B18.2"],
            "xơ gan": ["K74", "K74.6"],
            "sỏi thận": ["N20", "N20.0"],
            "thoát vị đĩa đệm": ["M51", "M51.9"],
            "đột quỵ": ["I63", "I64"],
            "tai biến mạch máu não": ["I63", "I64"],
            "thiếu máu": ["D50", "D50.9"],
            "bệnh tim mạch vành": ["I25", "I25.1"],  # R34: add for btmv abbr expansion
            "bệnh mạch vành": ["I25", "I25.1"],
            # === R34: Fecal/Urinary incontinence compound forms ===
            "mất kiểm soát đại tiện": ["R15", "F98.1"],  # R15 (organic) + F98.1 (non-organic encopresis)
            "mất kiểm soát tiểu tiện": ["R32", "N39.3", "N39.4"],  # R32 (unspecified) + N39.3/4 (stress/urge incontinence)
            "đại tiện không tự chủ": ["R15", "F98.1"],
            "tiểu tiện không tự chủ": ["R32", "N39"],
            "đái không tự chủ": ["R32", "N39"],
            "tiểu són": ["N39.3", "N39.4"],  # stress/urge
            "đại tiện són": ["R15"],
            # Full compound forms (lookup trước split để có codes đúng)
            "mất kiểm soát đại tiện hoặc tiểu tiện": ["R15", "R32"],
            "mất kiểm soát tiểu tiện hoặc đại tiện": ["R32", "R15"],
            "tiểu tiện hoặc đại tiện không tự chủ": ["R32", "R15"],
            "đại tiện hoặc tiểu tiện không tự chủ": ["R15", "R32"],
            "mất kiểm soát đại tiện và tiểu tiện": ["R15", "R32"],
            "mất kiểm soát tiểu tiện và đại tiện": ["R32", "R15"],

            # === MỚI 2026-07-10 — abbreviations VN (R27.6) ===
            "tha": ["I10"],  # Tăng huyết áp
            "nmct": ["I21", "I21.9"],  # Nhồi máu cơ tim
            "đtđ": ["E11", "E11.9"],  # Đái tháo đường
            "đtđ type 2": ["E11"],
            "đtđ type 1": ["E10"],
            "tbmmn": ["I63", "I64"],
            "tbmmnn": ["I63", "I64"],
            "copd": ["J44", "J44.9"],
            "osa": ["G47.3"],
            "ngưng thở khi ngủ": ["G47.3"],
            "ngừng thở khi ngủ": ["G47.3"],
            "ngưng thở khi ngủ do tắc nghẽn": ["G47.3"],
            "ngừng thở khi ngủ do tắc nghẽn": ["G47.3"],
            "nhiễm khuẩn huyết do tụ cầu vàng": ["A41.0"],
            "nhiễm khuẩn huyết do tụ cầu vàng nhạy cảm methicillin": ["A41.0"],
            "nhiễm khuẩn đường tiết niệu": ["N39.0"],
            "nhiễm khuẩn đường tiết niệu tái phát": ["N39.0"],
            "viêm đường tiết niệu": ["N39.0"],
            "bất thường điện giải": ["E87.8"],
            "rối loạn điện giải": ["E87.8"],
            "tử vong": ["R96"],
            "hc": ["R59"],

            # === MỚI 2026-07-10 — synonyms cho u (R27.6) ===
            "u ác tính": ["C80"],
            "u ác": ["C80"],
            "u lành": ["D48"],
            "u lành tính": ["D48"],
            "khối u": ["D48"],
            "neoplasm": ["D48"],

            # === MỚI 2026-07-10 — Organ-based mappings (R27.6) ===
            "u ác tính trực tràng": ["C20"],
            "u ác trực tràng": ["C20"],
            "ung thư trực tràng": ["C20"],
            "khối u trực tràng": ["C20", "D12"],
            "u lành trực tràng": ["D12"],
            "u lành tính trực tràng": ["D12"],

            "u ác tính đại tràng": ["C18", "C18.9"],
            "u ác đại tràng": ["C18"],
            "ung thư đại tràng": ["C18"],
            "khối u đại tràng": ["C18", "D12"],

            "u ác tính dạ dày": ["C16", "C16.9"],
            "u ác dạ dày": ["C16"],
            "ung thư dạ dày": ["C16"],
            "khối u dạ dày": ["C16", "D13.1"],

            "u ác tính gan": ["C22", "C22.9"],
            "u ác gan": ["C22"],
            "ung thư gan": ["C22"],
            "khối u gan": ["C22", "D13.4"],

            "u ác tính vú": ["C50", "C50.9"],
            "u ác vú": ["C50"],
            "khối u vú": ["C50", "D24"],

            "u ác tính buồng trứng": ["C56"],
            "u ác buồng trứng": ["C56"],
            "khối u buồng trứng": ["C56", "D27"],

            "u ác tính cổ tử cung": ["C53", "C53.9"],
            "u ác cổ tử cung": ["C53"],
            "khối u cổ tử cung": ["C53", "D26.0"],

            "u ác tính tuyến giáp": ["C73"],
            "khối u tuyến giáp": ["C73", "D34"],

            "u ác tính tuyến tiền liệt": ["C61"],
            "khối u tuyến tiền liệt": ["C61", "D29.1"],

            "u ác tính bàng quang": ["C67", "C67.9"],
            "khối u bàng quang": ["C67", "D30.3"],

            "u ác tính thận": ["C64"],
            "khối u thận": ["C64", "D30.0"],

            "u ác tính tụy": ["C25", "C25.9"],
            "khối u tụy": ["C25", "D13.6"],

            "u ác tính thực quản": ["C15", "C15.9"],
            "khối u thực quản": ["C15", "D13.0"],

            # === TIM MẠCH (Cardiology) — chuẩn BYT/WHO ICD-10 2026 ===
            "nhồi máu cơ tim cấp st chênh lên": ["I21.0"],
            "nmct cấp st chênh lên": ["I21.0"],
            "nhồi máu cơ tim cấp không st chênh lên": ["I21.4"],
            "nmct cấp không st chênh lên": ["I21.4"],
            "nhồi máu cơ tim cũ": ["I25.2"],
            "nmct cũ": ["I25.2"],
            "bệnh tim thiếu máu cục bộ": ["I25"],
            "bệnh mạch vành": ["I25", "I25.1"],
            "hội chứng vành cấp": ["I24"],
            "đau thắt ngực không ổn định": ["I20.0"],
            "đau thắt ngực ổn định": ["I20.8"],
            "st chênh lên": ["I21.3"],
            "st chênh xuống": ["I21.4"],
            "st chênh lên v1-v4": ["I21.0"],
            "st chênh lên v1-v6": ["I21.0"],
            "sóng q bệnh lý": ["I25.2"],
            "sóng t đảo ngược": ["I24.8"],
            "cuồng nhĩ": ["I48.3"],
            "rung nhĩ kèm đáp ứng thất nhanh": ["I48"],
            "nhịp nhanh trên thất": ["I47.1"],
            "nhịp nhanh thất": ["I47.2"],
            "rung thất": ["I49.0"],
            "block nhánh trái": ["I44.7"],
            "block nhánh phải": ["I45.1"],
            "block nhánh": ["I44.7", "I45.1"],
            "hội chứng sick sinus": ["I49.5"],
            "nhịp nhanh xoang": ["R00.0"],
            "nhịp chậm xoang": ["R00.1"],
            "block nhĩ thất": ["I44"],
            "block nhĩ thất độ 1": ["I44.0"],
            "block nhĩ thất độ 2": ["I44.1"],
            "block nhĩ thất độ 3": ["I44.2"],
            "block nhĩ thất hoàn toàn": ["I44.2"],
            "block tim": ["I44"],
            "tách thành động mạch chủ": ["I71.0"],
            "phình động mạch chủ": ["I71"],
            "phình động mạch chủ bụng": ["I71.4"],
            "phình động mạch chủ ngực": ["I71.2"],
            "rò động tĩnh mạch": ["I77.0"],
            "rò động - tĩnh mạch": ["I77.0"],
            "rò động-tĩnh mạch": ["I77.0"],
            "rò động tĩnh mạch đùi": ["I77.0"],
            "rò động tĩnh mạch chủ": ["I77.0"],
            "hẹp động mạch": ["I77.1"],
            "tắc động mạch": ["I74"],
            "viêm tắc động mạch": ["I74"],
            "hở van động mạch chủ": ["I35.1"],
            "hẹp van động mạch chủ": ["I35.0"],
            "hở van ba lá": ["I36.1"],
            "hẹp van ba lá": ["I36.0"],
            "viêm nội tâm mạc": ["I33", "I33.9"],
            "viêm nội tâm mạc nhiễm khuẩn": ["I33.0"],
            "viêm cơ tim": ["I40", "I40.9"],
            "viêm màng ngoài tim": ["I30"],
            "tràn dịch màng tim": ["I31.3"],
            "bệnh cơ tim": ["I42"],
            "bệnh cơ tim phì đại": ["I42.1"],
            "bệnh cơ tim giãn": ["I42.0"],
            "suy tim độ i nyha": ["I50.9"],
            "suy tim độ ii nyha": ["I50.9"],
            "suy tim độ iii nyha": ["I50.0"],
            "suy tim độ iv nyha": ["I50.0"],
            "suy tim tâm thu": ["I50.1"],
            "suy tim tâm trương": ["I50.9"],
            "suy tim cấp": ["I50.9"],
            "suy tim mạn": ["I50.9"],
            "huyết khối tĩnh mạch sâu": ["I82.4"],
            "dvt": ["I82.4"],
            "thuyên tắc phổi": ["I26.9"],
            "pe": ["I26.9"],

            # === MỚI 2026-07-10 — TIÊU HÓA, GAN, MẬT ===
            "viêm gan virus": ["B19"],
            "viêm gan mạn": ["K73"],
            "viêm gan cấp": ["K72.0"],
            "gan nhiễm mỡ": ["K76.0"],
            "sỏi mật": ["K80"],
            "sỏi túi mật": ["K80.2"],
            "viêm túi mật": ["K81"],
            "viêm tụy": ["K85"],
            "viêm tụy cấp": ["K85"],
            "viêm tụy mạn": ["K86.1"],
            "loét dạ dày": ["K25"],
            "loét tá tràng": ["K26"],
            "trào ngược dạ dày thực quản": ["K21"],
            "gerd": ["K21", "K21.9"],
            "hội chứng ruột kích thích": ["K58"],
            "ibs": ["K58"],
            "viêm đại tràng": ["K51", "K52"],
            "viêm ruột thừa": ["K35"],
            "thoát vị bẹn": ["K40"],
            "thoát vị": ["K46"],

            # === MỚI 2026-07-10 — HÔ HẤP ===
            "viêm phế quản": ["J20", "J21"],
            "viêm phế quản cấp": ["J20"],
            "viêm phế quản mạn": ["J42"],
            "viêm tiểu phế quản": ["J21"],
            "viêm phổi mắc phải cộng đồng": ["J18", "J18.9"],
            "viêm phổi cộng đồng": ["J18", "J18.9"],
            "tràn khí màng phổi": ["J93"],
            "tràn dịch màng phổi": ["J90"],
            "xẹp phổi": ["J98.1"],
            "lao phổi": ["A15", "A15.0"],

            # === MỚI 2026-07-10 — THẦN KINH, TÂM THẦN ===
            "động kinh": ["G40", "G40.9"],
            "parkinson": ["G20", "G21"],
            "bệnh parkinson": ["G20"],
            "alzheimer": ["G30", "G30.9"],
            "sa sút trí tuệ": ["F03"],
            "trầm cảm": ["F32", "F32.9"],
            "rối loạn lo âu": ["F41", "F41.9"],
            "tâm thần phân liệt": ["F20"],
            "đau nửa đầu": ["G43", "G43.9"],
            "migraine": ["G43"],

            # === MỚI 2026-07-10 — THẬN, TIẾT NIỆU ===
            "sỏi niệu quản": ["N20.1"],
            "sỏi bàng quang": ["N21.0"],
            "viêm bàng quang": ["N30"],
            "viêm thận - bể thận": ["N12"],
            "viêm bể thận": ["N12"],
            "viêm cầu thận": ["N05"],
            "hội chứng thận hư": ["N04"],
            "suy thận giai đoạn cuối": ["N18.6"],

            # === MỚI 2026-07-10 — CƠ XƯƠNG KHỚP ===
            "thoái hóa khớp": ["M19"],
            "thoái hóa khớp gối": ["M17"],
            "thoái hóa khớp háng": ["M16"],
            "viêm khớp dạng thấp": ["M05", "M06"],
            "gout": ["M10", "M10.9"],
            "gút": ["M10"],
            "loãng xương": ["M81"],
            "viêm cơ": ["M60"],
            "đau cơ xơ hóa": ["M79.7"],

            # === MỚI 2026-07-10 — DA, MÔ LIÊN KẾT ===
            "vẩy nến": ["L40", "L40.9"],
            "eczema": ["L20", "L30"],
            "viêm da cơ địa": ["L20"],
            "viêm da tiếp xúc": ["L25"],
            "mày đay": ["L50"],
            "zona": ["B02"],
            "herpes": ["B00"],

            # === MỚI 2026-07-10 — NỘI TIẾT ===
            "basedow": ["E05.0"],
            "cường giáp": ["E05"],
            "suy giáp": ["E03", "E03.9"],
            "suy tuyến yên": ["E23.0"],
            "rối loạn lipid máu": ["E78"],
            "tăng cholesterol máu": ["E78.0"],
            "tăng triglyceride máu": ["E78.1"],
            "béo phì": ["E66"],
            "suy dinh dưỡng": ["E46"],

            # === MỚI 2026-07-10 — MÁU ===
            "thiếu máu thiếu sắt": ["D50"],
            "thiếu máu mạn": ["D50", "D63"],
            "thiếu máu cấp": ["D62"],
            "leukemia": ["C95"],
            "lymphoma": ["C85"],
            "xuất huyết giảm tiểu cầu": ["D69.3"],
            "đông máu nội quản lý tán": ["D65"],
            "dic": ["D65"],

            # === MỚI 2026-07-10 — TAI MŨI HỌNG, MẮT ===
            "viêm xoang": ["J32"],
            "viêm tai giữa": ["H66"],
            "viêm amidan": ["J03"],
            "đục thủy tinh thể": ["H25", "H26"],
            "glocom": ["H40", "H40.9"],
            "tăng nhãn áp": ["H40"],

            # === MỚI 2026-07-10 — MẠCH MÁU ===
            "viêm mạch": ["I77.6", "I80"],
            "viêm tĩnh mạch": ["I80"],
            "viêm tĩnh mạch chi": ["I80.0"],
            "suy giãn tĩnh mạch": ["I83"],
            "giãn tĩnh mạch": ["I83"],

            # === MỚI 2026-07-11 — CHẤN THƯƠNG, XUẤT HUYẾT, NGOẠI KHOA CHUNG ===
            "chấn thương": ["T14", "T14.9"],
            "tai nạn": ["V99", "X59"],
            "xuất huyết nội sọ": ["I61", "I61.9"],
            "xuất huyết não": ["I61", "I61.9"],
            "chảy máu nội sọ": ["I61", "I61.9"],
            "chảy máu não": ["I61", "I61.9"],
            "gãy xương": ["T14.2"],
            "gãy cổ xương đùi di lệch": ["S72.0"],
            "gãy cổ xương đùi": ["S72.0"],
            "bệnh bạch cầu dòng tủy mãn tính": ["C92.1"],
            "bạch cầu cấp": ["C95.0"],
            "xuất huyết tiêu hóa": ["K92.2"],
            "xuất huyết dạ dày": ["K92.2"],
            "suy hô hấp": ["J96", "J96.9"],
            "suy hô hấp cấp": ["J96.0"],
            "suy hô hấp mạn": ["J96.1"],
            "nhiễm trùng huyết": ["A41", "A41.9"],
            "nhiễm khuẩn huyết": ["A41", "A41.9"],
            "sốc nhiễm khuẩn": ["R57.2"],
            "đau đầu": ["R51"],
            "nhức đầu": ["R51"],
            "chóng mặt": ["R42"],
            "buồn nôn": ["R11"],
            "nôn": ["R11"],
            "sốt": ["R50", "R50.9"],
            "sốt cao": ["R50"],
            "ho": ["R05"],
            "khó thở": ["R06.0"],
            "đau ngực": ["R07.4"],
            "đau bụng": ["R10.4"],
            "đau bụng vùng thượng vị": ["R10.1"],
            "tiêu chảy": ["A09", "K52.9"],
            "táo bón": ["K59.0"],
            "phù": ["R60", "R60.9"],
            # Expanded common cardiac & liver & kidney & respiratory mappings (Priority 1.1)
            "hội chứng não gan": ["K72.9", "G94"],
            "xơ gan do rượu": ["K70.3", "K70"],
            "nghi ngờ xơ gan do rượu": ["K70.3"],
            "gan nhiễm mỡ do rượu": ["K70.0"],
            "viêm gan do rượu": ["K70.1"],
            "phình động mạch chủ": ["I71", "I71.9"],
            "phình động mạch chủ nhỏ": ["I71"],
            "phình động mạch chủ bụng": ["I71.4"],
            "phình động mạch chủ ngực": ["I71.2"],
            "nhồi máu cơ tim vùng dưới cũ": ["I25.2"],
            "nhồi máu cơ tim cũ": ["I25.2"],
            "nhồi máu cơ tim vùng dưới": ["I21.1"],
            "nmct vùng dưới": ["I21.1"],
            "nmct vùng dưới cũ": ["I25.2"],
            "cuồng nhĩ": ["I48.3", "I48"],
            "block nhánh trái": ["I44.7"],
            "block nhánh phải": ["I45.1"],
            "block nhĩ thất độ 1": ["I44.0"],
            "block nhĩ thất độ 2": ["I44.1"],
            "block nhĩ thất độ 3": ["I44.2"],
            "block av": ["I44"],
            "suy thận mạn giai đoạn cuối": ["N18.5", "N18.9"],
            "suy thận mạn tính": ["N18", "N18.9"],
            "viêm phổi mắc phải cộng đồng": ["J18", "J18.9"],
            "vpmpccđ": ["J18", "J18.9"],
            # === Plan Fix 1.1: Comprehensive Expansion ===
            # Cardiac & Circulatory
            "nhồi máu cơ tim không sóng q": ["I21.4"],
            "nhồi máu cơ tim không st chênh lên": ["I21.4"],
            "rung nhĩ kịch phát": ["I48.0"],
            "rung nhĩ mạn": ["I48.2"],
            "rung nhĩ dai dẳng": ["I48.1"],
            "cuồng nhĩ điển hình": ["I48.3"],
            "cuồng nhĩ không điển hình": ["I48.4"],
            "nhịp nhanh xoang": ["R00.0"],
            "nhịp chậm xoang": ["R00.1"],
            "nhịp nhanh kịch phát trên thất": ["I47.1"],
            "nhịp nhanh thất": ["I47.2"],
            "rung thất": ["I49.0"],
            "block av độ 1": ["I44.0"],
            "block av độ 2": ["I44.1"],
            "block av độ 3": ["I44.2"],
            "block av hoàn toàn": ["I44.2"],
            "block nhánh trái trước": ["I44.4"],
            "block nhánh trái sau": ["I44.5"],
            "block 2 nhánh": ["I45.2"],
            "block 3 nhánh": ["I45.3"],
            "suy tim độ i": ["I50.1"],
            "suy tim độ ii": ["I50.2"],
            "suy tim độ iii": ["I50.3"],
            "suy tim độ iv": ["I50.4"],
            "suy tim trái": ["I50.1"],
            "suy tim phải": ["I50.0"],
            "suy tim toàn bộ": ["I50.9"],
            "phình tách động mạch chủ": ["I71.0"],
            "hẹp van hai lá": ["I34.2"],
            "hở van động mạch chủ": ["I35.1"],
            "hẹp van động mạch chủ": ["I35.0"],
            "hẹp van 3 lá": ["I36.0"],
            "hở van 3 lá": ["I36.1"],
            "viêm cơ tim": ["I40"],
            "viêm màng ngoài tim": ["I30"],
            "viêm màng ngoài tim cấp": ["I30.0"],
            "viêm nội tâm mạc": ["I33"],
            "tràn dịch màng tim": ["I31.3"],
            "chèn ép tim": ["I31.1"],
            "bệnh cơ tim": ["I42"],
            "bệnh cơ tim phì đại": ["I42.1"],
            "bệnh cơ tim giãn": ["I42.0"],
            "bệnh cơ tim hạn chế": ["I42.5"],
            "thiếu máu cơ tim": ["I24.9", "I20"],
            "thiếu máu cơ tim cục bộ": ["I25"],
            "thiếu máu cơ tim thầm lặng": ["I25.6"],
            "đau thắt ngực ổn định": ["I20.8"],
            "đau thắt ngực không ổn định": ["I20.0"],
            "cơn đau thắt ngực": ["I20"],
            # Cerebrovascular & Neurological
            "nhồi máu não": ["I63"],
            "nhồi máu não cũ": ["I69.3"],
            "xuất huyết dưới màng nhện": ["I60"],
            "đột quỵ não": ["I64"],
            "liệt nửa người": ["G81"],
            "liệt mặt": ["G51.0"],
            "co giật": ["R56"],
            "động kinh": ["G40"],
            "động kinh cục bộ": ["G40.0"],
            "động kinh toàn thể": ["G40.3"],
            "hôn mê": ["R40.2"],
            "mất ý thức": ["R40"],
            "ngất": ["R55"],
            "ngất xỉu": ["R55"],
            "đau nửa đầu": ["G43"],
            "viêm màng não": ["G00", "G03"],
            "viêm màng não mủ": ["G00"],
            "viêm màng não virus": ["G02.0"],
            "viêm não": ["G04"],
            "viêm tủy": ["G04"],
            "parkinson": ["G20"],
            "alzheimer": ["G30"],
            "sa sút trí tuệ": ["F03"],
            "trầm cảm": ["F32"],
            "rối loạn lo âu": ["F41"],
            # Opthalmology / Eye (R29 user spec round 2 — WHO ICD-10 ONLY, KHÔNG dùng ICD-10-CM 5th-digit)
            "phù gai thị": ["H47.1"],
            # Liver / Lab-as-diagnosis (R29 user spec round 2 — capture full phrase, modifier "máu" preserved)
            "tăng bilirubin máu": ["R17"],
            # Respiratory
            "viêm phổi thùy": ["J18.1"],
            "viêm phổi không điển hình": ["J18.9"],
            "viêm phế quản": ["J42"],
            "viêm phế quản cấp": ["J20"],
            "viêm phế quản mạn": ["J42"],
            "hen": ["J45"],
            "hen bội nhiễm": ["J45"],
            "hen cấp": ["J46"],
            "hen mạn": ["J45"],
            "bệnh phổi tắc nghẽn mạn": ["J44"],
            "bệnh phổi tắc nghẽn mạn tính": ["J44"],
            "khí phế thũng": ["J43"],
            "tràn khí màng phổi": ["J93"],
            "xẹp phổi": ["J98.1"],
            "ung thư phế quản": ["C34"],
            "lao phổi": ["A15"],
            # Digestive & Hepatobiliary
            "viêm gan virus": ["B19"],
            "viêm gan a": ["B15"],
            "viêm gan mạn": ["K73"],
            "gan nhiễm mỡ": ["K76.0"],
            "xơ gan mật": ["K74.4"],
            "suy gan": ["K72"],
            "suy gan cấp": ["K72.0"],
            "suy gan mạn": ["K72.1"],
            "hôn mê gan": ["K72.9"],
            "não gan": ["K72.9"],
            "ung thư gan nguyên phát": ["C22.0"],
            "viêm túi mật": ["K81"],
            "viêm túi mật cấp": ["K81.0"],
            "sỏi mật": ["K80"],
            "sỏi túi mật": ["K80.2"],
            "sỏi ống mật chủ": ["K80.5"],
            "viêm đường mật": ["K83.0"],
            "viêm tụy": ["K86.1"],
            "viêm tụy cấp": ["K85"],
            "viêm tụy mạn": ["K86.1"],
            "ung thư tụy": ["C25"],
            "ung thư đường mật": ["C22.1", "C24"],
            "viêm dạ dày cấp": ["K29.0"],
            "viêm dạ dày mạn": ["K29.5"],
            "loét dạ dày": ["K25"],
            "loét tá tràng": ["K26"],
            "viêm ruột thừa": ["K35"],
            "viêm ruột thừa cấp": ["K35"],
            "thoát vị bẹn": ["K40"],
            "thoát vị đùi": ["K41"],
            "thoát vị rốn": ["K42"],
            "thoát vị hoành": ["K44"],
            "tắc ruột": ["K56"],
            "viêm phúc mạc": ["K65"],
            # Renal & Urinary
            "suy thận giai đoạn cuối": ["N18.6"],
            "viêm bể thận": ["N12"],
            "viêm bể thận cấp": ["N10"],
            "viêm thận": ["N05"],
            "viêm cầu thận": ["N05"],
            "viêm cầu thận cấp": ["N00"],
            "hội chứng thận hư": ["N04"],
            "sỏi niệu quản": ["N20.1"],
            "sỏi bàng quang": ["N21.0"],
            "sỏi đường tiết niệu": ["N20"],
            "viêm bàng quang": ["N30"],
            "ung thư thận": ["C64"],
            "ung thư bàng quang": ["C67"],
            "ung thư tuyến tiền liệt": ["C61"],
            # Endocrine & Metabolic
            "suy thượng thận": ["E27"],
            "hội chứng cushing": ["E24"],
            # Infectious Diseases
            "sốt xuất huyết": ["A91"],
            "sốt xuất huyết dengue": ["A91"],
            "sốt rét": ["B54"],
            "sốt rét ác tính": ["B50"],
            "lao": ["A15"],
            "lao ngoài phổi": ["A18"],
            "nhiễm hiv": ["B24"],
            "aids": ["B24"],
            "viêm gan siêu vi": ["B19"],
            "thương hàn": ["A01"],
            "viêm não nhật bản": ["A83.0"],
            "covid": ["U07"],
            "covid-19": ["U07"],
            # Oncology
            "ung thư vòm họng": ["C11"],
            "ung thư thực quản": ["C15"],
            "u lympho": ["C85"],
            # Musculoskeletal
            "thoái hóa cột sống": ["M47"],
            "thoái hóa khớp gối": ["M17"],
            "thoát vị đĩa đệm cột sống": ["M51"],
            "viêm khớp": ["M13"],
            "viêm đa khớp": ["M13"],
            "trượt đĩa đệm": ["M51"],
            "đau lưng": ["M54.5"],
            "đau cổ": ["M54.2"],
            "đau vai": ["M75"],
            # Hematology
            "thiếu máu ác tính": ["D51"],
            "giảm tiểu cầu": ["D69"],
            "tăng tiểu cầu": ["D75.8"],
            "đông máu nội mạch": ["D65"],
            # Symptoms / General
            "tiêu chảy": ["R19.7"],
            "phù chi dưới": ["R60.0"],
            "gan to": ["R16.0"],
            "lách to": ["R16.1"],
            "tim to": ["I51.7"],
            "suy kiệt": ["R64"],
            "mệt mỏi": ["R53"],
            "chán ăn": ["R63.0"],
            "sụt cân": ["R63.4"],
            "tăng cân": ["R63.5"],
        }
        # R34: MERGE auto-mined aliases từ data/icd_aliases.json (build_mining_index.py).
        _load_mined_icd_aliases(self._icd_vn_to_codes)

        # R35 (2026-07-14): MERGE VN medical abbreviations từ data/icd_abbreviations.json
        _load_vn_medical_abbreviations(self._icd_vn_to_codes)

        # R34 (2026-07-13): Fix duplicate keys in dict literal (Python dict literal
        # OVERWRITES earlier entries on duplicate keys → some codes get lost).
        # Source file has 66 duplicate keys; 15 of them have meaningful extras
        # (e.g., 'tiêu chảy' first defined as ['A09', 'K52.9'], then re-defined
        # as ['R19.7'] → A09/K52.9 are LOST). Re-merge lost codes here.
        self._merge_lost_duplicate_entries()

    def _merge_lost_duplicate_entries(self) -> None:
        """R34: Re-add codes bị mất do Python dict literal override duplicate keys.

        Source có 66 keys xuất hiện ≥2 lần. Mỗi key, dict giữ lần cuối — các
        lần trước bị mất. Helper này merge back các codes bị mất (chỉ thêm code
        mới, KHÔNG đè).
        """
        # Extras captured từ source analysis (R34 spec round 3 — 2026-07-13).
        # Format: {key: [codes_to_add_back]}
        _LOST_EXTRAS: dict[str, list[str]] = {
            "ung thư gan": ["C22.9"],
            "ung thư dạ dày": ["C16.9"],
            "ung thư đại tràng": ["C18.9"],
            "viêm nội tâm mạc": ["I33.9"],
            "viêm cơ tim": ["I40.9"],
            "viêm tụy": ["K85"],
            "viêm phế quản": ["J20", "J21"],
            "lao phổi": ["A15.0"],
            "động kinh": ["G40.9"],
            "parkinson": ["G21"],
            "alzheimer": ["G30.9"],
            "trầm cảm": ["F32.9"],
            "rối loạn lo âu": ["F41.9"],
            "đau nửa đầu": ["G43.9"],
            "tiêu chảy": ["A09", "K52.9"],
        }
        for key, codes in _LOST_EXTRAS.items():
            existing = self._icd_vn_to_codes.get(key, [])
            for c in codes:
                if c not in existing:
                    existing.append(c)
            if existing:
                self._icd_vn_to_codes[key] = existing
        logger.info("[R34] Merged %d lost entries from duplicate keys", len(_LOST_EXTRAS))

    def _chapter_codes_lookup(self, text: str) -> list[str]:
        """Lookup codes thuộc chapter cụ thể dựa trên keyword match.

        R27.5 (2026-07-09): 5-tier lookup để fix ICD candidates SAI:
        1. Direct ICD mapping (VN → exact codes) → trả codes đúng (vd "ung thư phổi" → C34.x)
        2. VN substring match (desc_vi chứa text)
        3. Drop modifiers → substring match lại
        4. Translator VN→EN → match desc_en
        5. Drop more modifiers → return [] (KHÔNG fallback về kept SAI)

        Returns:
            list codes thuộc chapter đúng, hoặc [] nếu không tìm được.
        """
        import re as _re

        # Tier 1: Direct ICD mapping (R27.5) - exact VN → ICD codes
        # (ưu tiên cao nhất vì chính xác, không phụ thuộc desc_vi)
        text_lower = text.lower().strip()
        if hasattr(self, '_icd_vn_to_codes'):
            if text_lower in self._icd_vn_to_codes:
                return list(self._icd_vn_to_codes[text_lower])
            # Tier 1b: Prefix match - dict key là prefix của text
            # (vd "tắc mạch huyết khối" là prefix của "tắc mạch huyết khối tĩnh mạch chủ dưới...")
            for key in self._icd_vn_to_codes:
                if text_lower.startswith(key) and len(key) >= 8:
                    return list(self._icd_vn_to_codes[key])
            # Tier 1c: Text là prefix của dict key (vd text = "di căn não" match dict key "di căn não vùng trán phải")
            for key in self._icd_vn_to_codes:
                if key.startswith(text_lower) and len(text_lower) >= 6:
                    return list(self._icd_vn_to_codes[key])

        # Tier 2: VN substring match trong ICD index (desc_vi)
        codes = self._exact_match_vn_substring(text)
        if codes:
            return codes

        # Tier 3: Drop modifiers từ text, thử substring match lại
        # Drop: "không tế bào nhỏ", "tế bào nhỏ", "có/không", ...
        modifiers_pattern = _re.compile(
            r"\s+(không tế bào nhỏ|tế bào nhỏ|không tế bào lớn|tế bào lớn|"
            r"vô căn|thứ phát|nguyên phát|có hoặc không|ở trẻ em|ở người lớn|"
            r"mạn tính|cấp tính|không đặc hiệu)$",
            _re.IGNORECASE
        )
        text_clean = modifiers_pattern.sub("", text.strip()).strip()
        if text_clean != text and text_clean:
            if text_clean in self._icd_vn_to_codes:
                return list(self._icd_vn_to_codes[text_clean])
            codes = self._exact_match_vn_substring(text_clean)
            if codes:
                return codes


        # Tier 5: Drop more modifiers (suffix)
        text_clean2 = _re.sub(
            r"\s+(do\s+\S+|vô căn|không đặc hiệu)$",
            "", text.strip(), flags=_re.IGNORECASE
        ).strip()
        if text_clean2 != text and text_clean2 != text_clean:
            if text_clean2 in self._icd_vn_to_codes:
                return list(self._icd_vn_to_codes[text_clean2])
            codes = self._exact_match_vn_substring(text_clean2)
            if codes:
                return codes

        return []

    # ------------------------------------------------------------------ #

    @staticmethod
    def _looks_vn(text: str) -> bool:
        """Heuristic: có ký tự có dấu tiếng Việt hay không."""
        for ch in text:
            if (
                ch
                in "ăâđêôơưĂÂĐÊÔƠƯáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵÁÀẢÃẠẮẰẲẴẶẤẦẨẪẬÉÈẺẼẸẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤỨỪỬỮỰÝỲỶỸỴ"
            ):
                return True
        return False

    @staticmethod
    def _strip_clinical_prefix(text: str) -> str:
        """Bỏ các prefix cậu lâm sàng VN phổ biến để lấy được core concept.

        Ví dụ:
            "bệnh nhân bị tăng huyết áp"          → "tăng huyết áp"
            "chẩn đoán: viêm phổi cộng đồng"    → "viêm phổi cộng đồng"
            "nghĩ nhiều đến: nhồi máu cơ tim cấp" → "nhồi máu cơ tim cấp"
            "theo dõi đái tháo đường type 2"      → "đái tháo đường type 2"
            "BN bị viêm phổi"                     → "viêm phổi"
        """
        import re as _re

        if not text:
            return text
        s = text.strip()

        # Strip leading verbosity (case-insensitive)
        prefixes = [
            r"^chẩn\s*đo[áa]n\s*[:\-]?\s*",
            r"^theo\s*d[õo]i\s*[:\-]?\s*",
            r"^ngh[ĩi]\s*(nhi[ềe]u\s*)?[đd][ếe]n\s*[:\-]?\s*",
            r"^(chẩn\s*đo[áa]n\s*)?ph[âa]n\s*bi[ệe]t\s*[:\-]?\s*",
            r"^b[ệe]nh\s*nh[âa]n\s*(b[ịi]\s*)?",
            r"^BN\s*(b[ịi]\s*)?",
            r"^bệnh\s*nhân\s*bị\s*",
            r"^(chẩn đoán|theo dõi|nghĩ đến)\s*(là|nhiều đến)?\s*",
        ]
        for pat in prefixes:
            s2 = _re.sub(pat, "", s, flags=_re.IGNORECASE)
            if s2 != s:
                s = s2.strip()
        return s


# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #
# Local Search — Vector RAG bằng BGE-M3 + ma trận nhúng NumPy (.npy)
# ---------------------------------------------------------------------- #


class ICD10VectorSearch:
    """Tra cứu ICD-10 bằng vector nhúng cosine similarity.

    Sử dụng mô hình BAAI/bge-m3 để nhúng câu truy vấn tiếng Việt/Anh
    và tìm các dòng có cosine similarity cao nhất trên 71k dòng ICD-10.
    """

    def __init__(
        self,
        jsonl_path: Optional[Path] = None,
        embeddings_path: Optional[Path] = None,
    ) -> None:
        # Default: icd10.jsonl mới nhất (WHO ICD-10 2019 VN+EN, 15,732 entries).
        # Có cả desc_vi (cho VN queries) và desc_en (cho hybrid EN lookup).
        # Fallback: DM_ICD10_19_8_BYT.json (BYT chính thức, 36,689 entries, VN only).
        # Fallback 2: ICD10_Data.json (cũ).
        # Fallback 3: icd10.jsonl cũ (EN only).
        self._jsonl_path = jsonl_path or self._pick_default_jsonl()
        self._embeddings_path = embeddings_path or (DATA_DIR / "icd10_embeddings.npy")

        self.codes: list[str] = []
        self.descs_raw: list[str] = []
        self.chapters: list[str] = []  # Nhóm bệnh / chapter (BYT data)

        self._embeddings: Optional[np.ndarray] = None
        self._model = None
        self._device: Optional[str] = None
        self._lock = threading.Lock()
        self._loaded = False

    @staticmethod
    def _pick_default_jsonl() -> Path:
        """Chọn file JSON/JSONL mới nhất có sẵn (ưu tiên WHO 2019 VN+EN).

        NOTE: JSONL files LUÔN dùng local (Kaggle không có JSONL data source).
        """
        candidates = [
            DATA_DIR / "icd10.jsonl",                # WHO 2019 VN+EN (mới nhất)
            DATA_DIR / "DM_ICD10_19_8_BYT.json",    # BYT chính thức (lớn nhất)
            DATA_DIR / "ICD10_Data.json",           # cũ
        ]
        for c in candidates:
            if c.exists():
                return c
        return candidates[0]  # fallback nếu không có file nào

    def _ensure_loaded(self) -> None:
        """Lazy load: đọc ICD data + load embeddings.

        Schema mặc định (icd10.jsonl mới — WHO ICD-10 2019 VN translation, 15,732 entries):
            "code": "A00",
            "desc_vi": "Bệnh tả",
            "desc_en": "Cholera",
            "source": "...",
            "version": "WHO ICD-10 2019"

        Có cả desc_vi (cho VN queries qua BGE-M3 multilingual) và desc_en
        (cho hybrid search chính xác theo chuẩn WHO).

        Hỗ trợ fallback (auto-detect):
        - DM_ICD10_19_8_BYT.json (BYT VN, 36,689 entries)
        - ICD10_Data.json (cũ, 11,384 entries)
        - icd10.jsonl EN-only (cũ)
        """
        if self._loaded:
            return

        if not self._jsonl_path.exists():
            logger.warning("ICD10VectorSearch: không thấy file %s", self._jsonl_path)
            self._loaded = True
            return

        # 1. Đọc file
        t0 = time.time()
        suffix = self._jsonl_path.suffix.lower()

        if suffix == ".json":
            # JSON format (BYT hoặc ICD10_Data cũ)
            try:
                with self._jsonl_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)

                sample = data[0] if data else {}
                is_byt = "Mã" in sample and "Tên bệnh" in sample

                for row in data:
                    if is_byt:
                        code = str(row.get("Mã", row.get("code", ""))).strip()
                        name_vi = str(row.get("Tên bệnh", row.get("desc_vi", row.get("name", "")))).strip()
                        nhom = str(row.get("Nhóm bệnh", "")).strip()
                        mo_ta = str(row.get("Mô tả", "")).strip()
                    else:
                        # ICD10_Data.json cũ
                        code = str(row.get("Mã bệnh", row.get("code", ""))).strip()
                        name_vi = str(row.get("Tên bệnh gốc", row.get("desc_vi", row.get("name", "")))).strip()
                        nhom = str(row.get("Tên nhóm", "")).strip()
                        mo_ta = ""

                    if not code:
                        continue
                    if not name_vi:
                        name_vi = mo_ta or nhom or code
                    # Concat context để embedding có thêm thông tin phân biệt
                    ctx_parts = [name_vi]
                    if nhom and nhom != name_vi:
                        ctx_parts.append(nhom)
                    if mo_ta and mo_ta != name_vi and "QĐ" not in mo_ta:
                        ctx_parts.append(mo_ta)
                    desc = " | ".join(ctx_parts)
                    self.codes.append(code)
                    self.descs_raw.append(desc)
                    self.chapters.append(nhom)
            except Exception as exc:
                logger.error("ICD10VectorSearch: lỗi đọc %s: %s", self._jsonl_path, exc)
                self._loaded = True
                return
        else:
            # JSONL format (icd10.jsonl mới hoặc cũ)
            with self._jsonl_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    code = str(row.get("code", "")).strip()
                    desc_vi = str(row.get("desc_vi", "")).strip()
                    desc_en = str(row.get("desc_en", "")).strip()
                    name_fallback = str(row.get("name", row.get("desc", row.get("Tên bệnh", "")))).strip()

                    if not code:
                        continue
                    # Prefer desc_vi cho embedding (VN queries); fallback desc_en / name
                    # Nếu có cả 2 thì embed cả 2 để hybrid VN-EN match
                    if desc_vi and desc_en and desc_vi != desc_en:
                        desc = f"{desc_vi} | {desc_en}"
                    elif desc_vi:
                        desc = desc_vi
                    elif desc_en:
                        desc = desc_en
                    elif name_fallback:
                        desc = name_fallback
                    else:
                        desc = code
                    self.codes.append(code)
                    self.descs_raw.append(desc)
                    self.chapters.append("")

        logger.info(
            "ICD10VectorSearch: Đã nạp %d mã ICD từ %s (%.2fs)",
            len(self.codes),
            self._jsonl_path.name,
            time.time() - t0,
        )

        # 2. Tải ma trận nhúng .npy nếu có sẵn
        if self._embeddings_path.exists():
            try:
                self._embeddings = np.load(self._embeddings_path)
                logger.info(
                    "ICD10VectorSearch: Đã tải ma trận embeddings %s (shape: %r)",
                    self._embeddings_path.name,
                    self._embeddings.shape,
                )
            except Exception as exc:
                logger.warning(
                    "Không thể load file embeddings %s: %s",
                    self._embeddings_path,
                    exc,
                )

        self._loaded = True

    def search(
        self, query: str, *, top_k: int = 10, threshold: float = 0.4
    ) -> list[str]:
        """Tìm các mã ICD-10 tương đồng nhất với câu truy vấn.

        Hỗ trợ cả truy vấn tiếng Việt và tiếng Anh nhờ mô hình nhúng đa ngôn ngữ BGE-M3.
        """
        self._ensure_loaded()
        if not query or not self.codes:
            return []

        # 1 & 2. Đảm bảo model và embeddings sẵn sàng
        if self._embeddings is None:
            logger.info(
                "ICD10VectorSearch: Không tìm thấy file embeddings.npy có sẵn. Đang sinh trực tiếp..."
            )
            try:
                t0 = time.time()
                self._embeddings = self._encode_safe(
                    self.descs_raw,
                    batch_size=128,
                    show_progress_bar=True,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )
                if self._embeddings is None or len(self._embeddings) == 0:
                    return []
                try:
                    np.save(self._embeddings_path, self._embeddings)
                    logger.info(
                        "Đã lưu ma trận nhúng ra %s (%.2fs)",
                        self._embeddings_path,
                        time.time() - t0,
                    )
                except Exception as exc_save:
                    logger.warning("Không thể lưu %s: %s", self._embeddings_path, exc_save)
            except Exception as exc:
                logger.error("Lỗi sinh embeddings: %s", exc)
                return []

        # 3. Mã hóa câu truy vấn qua bộ đệm thread-safe + auto CPU fallback
        query_vec = self._encode_safe(
            query, normalize_embeddings=True, convert_to_numpy=True
        )  # Shape: (1024,)
        if query_vec is None or len(query_vec) == 0:
            return []

        # Cast về cùng dtype với embeddings (float16 tiết kiệm RAM)
        if self._embeddings is not None and self._embeddings.dtype != query_vec.dtype:
            query_vec = query_vec.astype(self._embeddings.dtype)

        # 4. Tính Cosine Similarity bằng dot-product (vì vector đã được chuẩn hóa L2)
        scores = np.dot(self._embeddings, query_vec)  # Shape: (N,)

        # 5. Sắp xếp lấy Top K
        top_indices = np.argsort(-scores)[:top_k]

        out: list[str] = []
        for idx in top_indices:
            if idx < 0 or idx >= len(self.codes):
                continue
            score = float(scores[idx])
            # Có thể lọc theo threshold (Cosine Similarity thường từ 0.0 -> 1.0)
            if score >= threshold:
                code = self.codes[idx]
                if code not in out:
                    out.append(code)

        logger.debug(
            "Vector search '%s' -> Top 1 score: %.3f (%s)",
            query,
            scores[top_indices[0]] if len(top_indices) > 0 and 0 <= top_indices[0] < len(self.codes) else 0.0,
            out[0] if out else "None",
        )
        return out

    def _encode_safe(self, text_or_list: Any, **kwargs: Any) -> Any:
        """Thread-safe BGE-M3 encode với tự động bảo vệ VRAM và fallback CPU khi gặp CUDA OOM."""
        with self._lock:
            if self._model is None:
                try:
                    from sentence_transformers import SentenceTransformer  # type: ignore
                    import torch
                    target_device = "cuda" if torch.cuda.is_available() else "cpu"
                    if target_device == "cuda":
                        try:
                            free_mem, _ = torch.cuda.mem_get_info()
                            # Nếu VRAM trống dưới 1.5 GiB (do Ollama chiếm dụng), tự động nạp trên CPU
                            if free_mem < 1.5 * (1024**3):
                                logger.info(
                                    "ICD10VectorSearch: VRAM trống %.1f MiB (< 1.5 GiB), nạp BGE-M3 trên CPU để tránh OOM với Ollama.",
                                    free_mem / (1024**2),
                                )
                                target_device = "cpu"
                        except Exception:
                            pass
                    self._model = SentenceTransformer("BAAI/bge-m3", device=target_device)
                    self._device = target_device
                    logger.info("ICD10VectorSearch: Đã tải mô hình BGE-M3 trên device=%s.", target_device)
                except ImportError:
                    logger.error("Chưa cài sentence-transformers! Không thể chạy Vector Search.")
                    return np.array([])

            try:
                import torch
                with torch.inference_mode():
                    return self._model.encode(text_or_list, **kwargs)
            except RuntimeError as err:
                if "out of memory" in str(err).lower() or "cuda" in str(err).lower():
                    logger.warning(
                        "CUDA OOM trong BGE-M3 encode. Tự động chuyển model sang CPU và retry..."
                    )
                    try:
                        import torch
                        torch.cuda.empty_cache()
                        self._model = self._model.to("cpu")
                        self._device = "cpu"
                    except Exception:
                        pass
                    import torch
                    with torch.inference_mode():
                        return self._model.encode(text_or_list, **kwargs)
                raise

    def score_codes(self, query: str, codes: list[str]) -> dict[str, float]:
        """Tính cosine similarity cho 1 tập codes cụ thể.

        Hữu ích cho hybrid search: vector_search.search() chỉ trả codes (top-k),
        hybrid cần re-score các candidates từ BM25 để combine đúng tỷ trọng.

        Returns: dict {code: cosine_sim} (giá trị ∈ [0, 1] cho vectors đã chuẩn hoá L2).
        """
        self._ensure_loaded()
        if not query or not codes or self._embeddings is None:
            return {}

        q_vec = self._encode_safe(
            query, normalize_embeddings=True, convert_to_numpy=True
        )
        if q_vec is None or len(q_vec) == 0:
            return {}

        # Embeddings đã được L2-normalize lúc build → dot = cosine.
        all_scores = self._embeddings @ q_vec  # shape (N,)
        idx_by_code = {c: i for i, c in enumerate(self.codes)}
        out: dict[str, float] = {}
        for code in codes:
            idx = idx_by_code.get(code)
            if idx is not None:
                out[code] = float(all_scores[idx])
        return out


# ---------------------------------------------------------------------- #
# Local Search — BM25 keyword index cho ICD-10
# ---------------------------------------------------------------------- #


# Stop words y khoa thường gặp — tokenize sẽ bỏ qua để BM25 tập trung vào từ mang
# thông tin (disease name, modifier quan trọng: "stage", "grade", "acute" được GIỮ
# vì chúng phân biệt code; chỉ bỏ các từ chức năng tiếng Anh).
_BM25_STOP_WORDS = frozenset({
    "the", "of", "and", "or", "with", "without", "in", "to", "a", "an",
    "due", "by", "for", "from", "as", "at", "on", "is", "are", "be",
    "other", "specified", "not", "no", "nec", "nos",
})


def _bm25_tokenize(text: str) -> list[str]:
    """Tokenize cho BM25: giữ alnum runs + VN diacritics, bỏ stop words, giữ nguyên case-fold.

    Lưu ý: cố tình KHÔNG stemming (vì ICD codes dùng chính xác các từ "type 2",
    "stage III" — stemming sẽ làm hỏng match).

    Hỗ trợ cả VN và EN: regex `[\\w]+` match cả chữ cái VN có dấu.
    """
    if not text:
        return []
    tokens = re.findall(r"[\w]+", text.lower(), flags=re.UNICODE)
    return [t for t in tokens if len(t) > 1 and t not in _BM25_STOP_WORDS]


class ICD10BM25Index:
    """BM25 keyword index cho ICD-10 — bổ trợ cho vector search.

    Multi-field weighting (BM25F-style thủ công): mỗi document được build bằng
    cách concat các field với trọng số khác nhau:
        - desc_vi (Tên bệnh) × 2    (field chính — concept name)
        - chapter (Nhóm bệnh) × 2   (nếu có — context phân biệt)
        - code_lower × 1            (light — bắt partial code match như "I50")

    Vì rank-bm25 chỉ hỗ trợ đơn field, ta mô phỏng bằng cách nhân đôi token thay
    vì tính BM25F thực sự (vẫn giữ được tinh thần field-weighted ranking).

    Hỗ trợ cả BYT format và ICD10_Data.json cũ (xem _ensure_loaded).
    """

    def __init__(
        self,
        jsonl_path: Optional[Path] = None,
        tokens_cache_path: Optional[Path] = None,
    ) -> None:
        # Default: icd10.jsonl (WHO ICD-10 2019 VN+EN, 15,732 entries).
        # Fallback chain: BYT → ICD10_Data cũ.
        # NOTE: JSONL files LUÔN dùng local (Kaggle không có JSONL data source).
        candidates = [
            DATA_DIR / "icd10.jsonl",
            DATA_DIR / "DM_ICD10_19_8_BYT.json",
            DATA_DIR / "ICD10_Data.json",
        ]
        if jsonl_path is not None:
            self._jsonl_path = jsonl_path
        else:
            self._jsonl_path = next((c for c in candidates if c.exists()), candidates[0])
        self._tokens_cache_path = tokens_cache_path or (
            DATA_DIR / "icd10_bm25_tokens.jsonl.gz"
        )

        self.codes: list[str] = []
        self._bm25: Optional[Any] = None
        self._id_to_idx: dict[str, int] = {}
        self._loaded = False
        self._max_doc_score: float = 1.0  # cho normalize

    def _build_doc_text(
        self,
        desc_vi: str,
        code: str,
        ten_nhom: str = "",
        ten_chuong: str = "",
    ) -> str:
        """Ghép các field với trọng số nhân đôi để mô phỏng field-weighted BM25.

        Format BYT (DM_ICD10_19_8_BYT.json):
            desc_vi = "Tên bệnh" (có thể có (...) hoặc [...])
            ten_nhom = "Nhóm bệnh" (chapter)
            ten_chuong = "" (không có trong BYT format)

        Hoặc format ICD10_Data.json cũ:
            desc_vi = "Tên bệnh gốc"
            ten_nhom = "Tên nhóm"
            ten_chuong = "Tên chương"
        """
        parts: list[str] = []
        if desc_vi:
            parts.extend([desc_vi, desc_vi])  # nhân đôi để BM25 ưu tiên
        if ten_nhom and ten_nhom != desc_vi:
            parts.extend([ten_nhom, ten_nhom])
        if ten_chuong and ten_chuong != desc_vi:
            parts.append(ten_chuong)
        if code:
            parts.append(code.lower())
        return " ".join(parts)

    def _ensure_loaded(self) -> None:
        """Lazy load: thử cache tokens trước, không thì build từ jsonl."""
        if self._loaded:
            return
        if not _HAS_BM25:
            logger.warning(
                "rank-bm25 chưa cài → BM25 search disabled. "
                "Chạy: pip install rank-bm25"
            )
            self._loaded = True
            return

        tokenized: list[list[str]] = []

        # 1. Try load token cache
        if self._tokens_cache_path.exists():
            try:
                t0 = time.time()
                with gzip.open(self._tokens_cache_path, "rt", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        row = json.loads(line)
                        self.codes.append(row["code"])
                        tokenized.append(row["tokens"])
                logger.info(
                    "BM25: loaded %d tokenized docs từ cache (%.2fs)",
                    len(self.codes),
                    time.time() - t0,
                )
            except Exception as exc:
                logger.warning("BM25 cache load fail (%s) → rebuild", exc)
                self.codes.clear()
                tokenized.clear()

        # 2. Fallback: build from JSON (mới) hoặc JSONL (cũ)
        if not tokenized:
            if not self._jsonl_path.exists():
                logger.warning("BM25: thiếu file %s", self._jsonl_path)
                self._loaded = True
                return
            t0 = time.time()
            suffix = self._jsonl_path.suffix.lower()

            if suffix == ".json":
                # Format mới: DM_ICD10_19_8_BYT.json hoặc ICD10_Data.json
                try:
                    with self._jsonl_path.open("r", encoding="utf-8") as f:
                        data = json.load(f)

                    # Detect format
                    sample = data[0] if data else {}
                    is_byt_format = "Mã" in sample and "Tên bệnh" in sample

                    for row in data:
                        if is_byt_format:
                            code = str(row.get("Mã", "")).strip()
                            desc_vi = str(row.get("Tên bệnh", "")).strip()
                            ten_nhom = str(row.get("Nhóm bệnh", "")).strip()
                            ten_chuong = ""
                        else:
                            code = str(row.get("Mã bệnh", "")).strip()
                            desc_vi = str(row.get("Tên bệnh gốc", "")).strip()
                            ten_nhom = str(row.get("Tên nhóm", "")).strip()
                            ten_chuong = str(row.get("Tên chương", "")).strip()
                        if not code:
                            continue
                        if not desc_vi:
                            desc_vi = code
                        doc_text = self._build_doc_text(desc_vi, code, ten_nhom, ten_chuong)
                        self.codes.append(code)
                        tokenized.append(_bm25_tokenize(doc_text))
                except Exception as exc:
                    logger.error("BM25: lỗi đọc %s: %s", self._jsonl_path, exc)
                    self._loaded = True
                    return
            else:
                # Format cũ: icd10.jsonl
                with self._jsonl_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        code = str(row.get("code", "")).strip()
                        desc_en = str(row.get("desc_en", "")).strip()
                        desc_vi = str(row.get("desc_vi", "")).strip()
                        if not code:
                            continue
                        doc_text = self._build_doc_text(desc_vi or desc_en or code, code)
                        self.codes.append(code)
                        tokenized.append(_bm25_tokenize(doc_text))

            logger.info(
                "BM25: tokenized %d docs từ %s (%.2fs)",
                len(self.codes),
                self._jsonl_path.name,
                time.time() - t0,
            )

            # Save cache
            try:
                self._tokens_cache_path.parent.mkdir(parents=True, exist_ok=True)
                with gzip.open(
                    self._tokens_cache_path, "wt", encoding="utf-8"
                ) as f:
                    for code, toks in zip(self.codes, tokenized):
                        f.write(
                            json.dumps(
                                {"code": code, "tokens": toks}, ensure_ascii=False
                            )
                            + "\n"
                        )
                logger.info("BM25: cached → %s", self._tokens_cache_path.name)
            except Exception as exc:
                logger.warning("BM25: không cache được (%s)", exc)

        # 3. Build BM25Okapi
        t0 = time.time()
        self._bm25 = BM25Okapi(tokenized)
        self._id_to_idx = {c: i for i, c in enumerate(self.codes)}
        # Tính max_doc_score = max BM25 score trên tập dummy query "disease"
        # để normalize. Thực tế sẽ dùng dynamic max từ query thực.
        self._max_doc_score = 1.0
        logger.info("BM25: built Okapi index trong %.2fs", time.time() - t0)
        self._loaded = True

    def search(
        self, query: str, *, top_k: int = 20
    ) -> tuple[list[str], list[float]]:
        """Search BM25 → (codes, scores) sorted desc theo score.

        Args:
            query: câu truy vấn (EN hoặc VN — tokenizer đa ngôn ngữ vì regex alnum).
            top_k: số codes trả về tối đa.

        Returns:
            (codes, scores) — scores là BM25 raw (chưa normalize).
            Nếu query rỗng hoặc BM25 chưa load → trả về list rỗng.
        """
        self._ensure_loaded()
        if self._bm25 is None or not query:
            return [], []
        q_tokens = _bm25_tokenize(query)
        if not q_tokens:
            return [], []
        scores = self._bm25.get_scores(q_tokens)
        if scores.size == 0:
            return [], []
        top_idx = np.argsort(-scores)[:top_k]
        out_codes: list[str] = []
        out_scores: list[float] = []
        for idx in top_idx:
            s = float(scores[idx])
            if s > 0:
                out_codes.append(self.codes[int(idx)])
                out_scores.append(s)
        return out_codes, out_scores

    def score_codes(self, query: str, codes: list[str]) -> dict[str, float]:
        """Tính BM25 score cho 1 tập codes cụ thể, đã normalize về [0, 1].

        Normalize = score / max(scores_in_corpus) cho cùng query → cùng scale
        với cosine similarity ∈ [0, 1].
        """
        self._ensure_loaded()
        if self._bm25 is None or not query or not codes:
            return {}
        q_tokens = _bm25_tokenize(query)
        if not q_tokens:
            return {}
        scores = self._bm25.get_scores(q_tokens)
        if scores.size == 0:
            return {}
        max_s = float(scores.max()) or 1.0
        out: dict[str, float] = {}
        for code in codes:
            idx = self._id_to_idx.get(code)
            if idx is None:
                out[code] = 0.0
            else:
                out[code] = float(scores[idx]) / max_s
        return out


# ---------------------------------------------------------------------- #
# Hybrid Search — combine vector (cosine) + BM25 keyword
# ---------------------------------------------------------------------- #


class ICD10HybridSearch:
    """Semantic ICD extraction — embedding-based, threshold-based, NO top-K cap.

    Triết lý (không phải top-K ranking):
      1. Embed query (BGE-M3) → cosine với 71k code descriptions.
      2. Union candidates từ vector fanout + BM25 fanout (tăng recall).
      3. Re-score cosine cho từng candidate.
      4. Giữ TẤT CẢ codes có cosine ≥ threshold (mặc định 0.7).
      5. Trả về tất cả (KHÔNG cap top-K) — semantic extraction.

    Ví dụ:
      "suy thận" → match ['suy thận' (cos=0.85), 'suy thận nhiễm mỡ' (cos=0.75)] → cả 2.
      "viêm phổi" → match ['viêm phổi' (cos=0.82)] → 1 code.
      Threshold 0.7 loại bỏ C88.2 (Waldenström) khi search "pcr dương tính với virus bk".

    Args:
        vector_search: instance ICD10VectorSearch.
        bm25_index:    instance ICD10BM25Index (chỉ dùng để mở rộng candidates).
        alpha, beta:   legacy params (không dùng nữa, giữ cho backward compat).
        top_k:         legacy cap. None (mặc định) = no cap. Set số > 0 để cap.
        threshold:     cosine threshold tối thiểu (mặc định 0.7).
        fanout:        số candidates tối đa từ vector + BM25 mỗi method (union).
    """

    def __init__(
        self,
        vector_search: Optional[ICD10VectorSearch] = None,
        bm25_index: Optional[ICD10BM25Index] = None,
        alpha: float = 0.6,
        beta: float = 0.4,
        top_k: Optional[int] = None,    # None = no cap (semantic extraction)
        threshold: float = 0.5,         # Cosine threshold (giảm từ 0.7 → 0.5 để match VN→EN+VN)
        fanout: int = 50,
    ) -> None:
        # alpha/beta kept cho backward compat nhưng KHÔNG dùng trong search() nữa.
        if abs((alpha + beta) - 1.0) > 1e-6:
            raise ValueError(
                f"alpha + beta phải bằng 1.0 (got alpha={alpha}, beta={beta})."
            )
        self.vector_search = vector_search or ICD10VectorSearch()
        self.bm25_index = bm25_index or ICD10BM25Index()
        self.alpha = alpha
        self.beta = beta
        self.top_k = top_k
        self.threshold = threshold
        self.fanout = fanout

    def search(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> list[str]:
        """Semantic extraction: trả về TẤT CẢ codes có cosine ≥ threshold.

        KHÔNG cap top-K (mặc định top_k=None). BM25 chỉ dùng để mở rộng
        candidates, threshold cuối cùng vẫn là cosine similarity (semantic).
        """
        if not query:
            return []
        k = top_k if top_k is not None else self.top_k
        thr = threshold if threshold is not None else self.threshold
        fanout = max(self.fanout, 30)

        # 1. Lấy candidates rộng từ vector (chính) + BM25 (mở rộng recall)
        vec_codes = self.vector_search.search(query, top_k=fanout, threshold=0.0) or []
        bm25_codes, _ = self.bm25_index.search(query, top_k=fanout)

        # 2. Union — BM25 giúp catch exact keyword match mà vector miss
        candidates = list(dict.fromkeys(vec_codes + bm25_codes))
        if not candidates:
            return []

        # 3. Tính lại cosine cho tất cả candidates
        vec_scores = self.vector_search.score_codes(query, candidates)

        # 4. Filter theo cosine (semantic embedding) — alpha/beta không dùng
        matched: list[tuple[str, float]] = []
        for code in candidates:
            cos = vec_scores.get(code, 0.0)
            if cos >= thr:
                matched.append((code, cos))

        # 5. Sort desc theo cosine; tie-break bằng BM25
        bm25_scores = self.bm25_index.score_codes(query, [c for c, _ in matched])
        matched.sort(
            key=lambda x: (-x[1], -bm25_scores.get(x[0], 0.0))
        )

        if matched:
            logger.debug(
                "Semantic '%s' → %d codes (top1=%s cos=%.3f)",
                query, len(matched), matched[0][0], matched[0][1],
            )
        # 6. Trả tất cả matching (no cap nếu k=None)
        if k is None:
            return [c for c, _ in matched]
        return [c for c, _ in matched[:k]]


# ---------------------------------------------------------------------- #
# Chapter restriction — restrict RAG candidates to relevant ICD chapter
# based on detected clinical keywords (mới 2026-07 fix L73.2 case).
# Logic: nếu text chứa keyword → chỉ giữ codes thuộc chapter tương ứng.
# Dùng cosine score TRONG chapter đã restrict.
# ---------------------------------------------------------------------- #

def _load_chapter_restrictions() -> list[tuple[set[str], str]]:
    path = DATA_DIR / "chapter_restrictions.json"
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return [(set(kws), prefix) for kws, prefix in raw]
    except Exception as e:
        logger.warning("Failed to load %s: %s", path, e)
        return []

_CHAPTER_RESTRICTIONS: list[tuple[set[str], str]] = _load_chapter_restrictions()


class DynamicConfigManager:
    """Hot-Reloading Config & Pre-tokenized Trie/Cache (Upgrade E - ZERO Hardcode)."""
    def __init__(self) -> None:
        self.synonym_rings: dict[str, list[str]] = {}
        self.cooccurrence_map: dict[str, list[str]] = {}
        self._synonym_tokens_cache: dict[str, set[str]] = {}
        self._mtimes: dict[Path, float] = {}
        self._last_check: float = 0.0
        self.reload_if_needed(force=True)

    def reload_if_needed(self, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self._last_check) < 2.0:
            return
        self._last_check = now

        syn_path = DATA_DIR / "synonym_rings.json"
        coocc_path = DATA_DIR / "drug_disease_cooccurrence.json"

        # Check synonym rings
        if syn_path.exists():
            mtime = syn_path.stat().st_mtime
            if force or self._mtimes.get(syn_path) != mtime:
                self._mtimes[syn_path] = mtime
                try:
                    self.synonym_rings = json.loads(syn_path.read_text(encoding="utf-8"))
                    self._synonym_tokens_cache.clear()
                    for k, syn_list in self.synonym_rings.items():
                        tokens_set: set[str] = set()
                        for syn in syn_list:
                            tokens_set.update(re.findall(r'[a-zà-ỹ0-9_/-]{2,}', syn.lower()))
                        self._synonym_tokens_cache[k.lower()] = tokens_set
                except Exception as e:
                    logger.warning("Failed to hot-reload %s: %s", syn_path, e)

        # Check drug-disease cooccurrence
        if coocc_path.exists():
            mtime = coocc_path.stat().st_mtime
            if force or self._mtimes.get(coocc_path) != mtime:
                self._mtimes[coocc_path] = mtime
                try:
                    self.cooccurrence_map = json.loads(coocc_path.read_text(encoding="utf-8"))
                except Exception as e:
                    logger.warning("Failed to hot-reload %s: %s", coocc_path, e)

_CONFIG_MGR = DynamicConfigManager()
_DRUG_DISEASE_COOCCURRENCE = _CONFIG_MGR.cooccurrence_map
_SYNONYM_RINGS = _CONFIG_MGR.synonym_rings


def _expand_tokens_with_synonyms(tokens: set[str], text_lower: str) -> set[str]:
    _CONFIG_MGR.reload_if_needed()
    expanded = set(tokens)
    if not _CONFIG_MGR._synonym_tokens_cache:
        return expanded
    for k, cached_tokens in _CONFIG_MGR._synonym_tokens_cache.items():
        if k in text_lower or k in tokens:
            expanded.update(cached_tokens)
    return expanded


def _get_boosted_prefixes(other_entities: list[dict] | None) -> set[str]:
    _CONFIG_MGR.reload_if_needed()
    if not other_entities or not _CONFIG_MGR.cooccurrence_map:
        return set()
    boosted: set[str] = set()
    for ent in other_entities:
        ent_type = ent.get("type", "")
        if ent_type in ("THUỐC", "TRIỆU_CHỨNG", "CHẨN_ĐOÁN"):
            text_lower = str(ent.get("text", "")).lower()
            for kw, prefixes in _CONFIG_MGR.cooccurrence_map.items():
                if kw in text_lower:
                    boosted.update(prefixes)
    return boosted



def _restrict_chapter(codes, entity_text):
    """Restrict ICD codes to chapter dựa trên clinical keyword detection.

    Nếu text chứa keyword → chỉ giữ codes thuộc chapter_prefix.
    Nếu chapter-restricted rỗng → trả về codes gốc (fallback).
    """
    if not codes:
        return codes
    text_lower = entity_text.lower()
    for keywords, chapter_prefix in _CHAPTER_RESTRICTIONS:
        if any(kw in text_lower for kw in keywords):
            restricted = [c for c in codes if c.startswith(chapter_prefix)]
            if restricted:
                return restricted
    return codes


def _text_matches_chapter_keyword(text: str) -> bool:
    """True nếu text chứa keyword của bất kỳ chapter restriction rule nào.

    Dùng để trigger VN prefix exact match fallback khi vector search trả codes
    sai chapter (vd: "viêm tuyến mồ hôi" → vector trả A00.0/A04 bacterial,
    nhưng text match "L73" rule → fallback exact match sẽ tìm L73.2).
    """
    if not text:
        return False
    text_lower = text.lower()
    for keywords, _ in _CHAPTER_RESTRICTIONS:
        if any(kw in text_lower for kw in keywords):
            return True
    return False


# ---------------------------------------------------------------------- #
# Build helpers
# ---------------------------------------------------------------------- #


def build_from_seed(path: Path, out_index: Optional[Path] = None) -> ICDIndex:
    """Đọc ICD data và build ICDIndex.

    Hỗ trợ 4 format:
    - JSONL mới (icd10.jsonl — WHO 2019 VN+EN): [{"code", "desc_vi", "desc_en"}]
    - JSON BYT (DM_ICD10_19_8_BYT.json): [{"Mã", "Tên bệnh", "Nhóm bệnh"}]
    - JSON cũ (ICD10_Data.json): [{"Mã bệnh", "Tên bệnh gốc", "Tên nhóm"}]
    - JSONL cũ: [{"code", "name_en", "vn_aliases"}]

    Index chứa names (cho exact match + fuzzy). Parentheses như
    "Hen [suyễn]" hoặc "Rối loạn... (modifier)" được GIỮ NGUYÊN trong names
    (cho exact match), nhưng cũng tạo thêm key không có parenthetical.
    """
    import re as _re

    def _strip_parens(name: str) -> str:
        if not name:
            return name
        cleaned = _re.sub(r"\s*\[.*?\]\s*$", "", name)
        cleaned = _re.sub(r"\s*\(.*?\)\s*$", "", cleaned)
        return cleaned.strip()

    idx = ICDIndex()
    suffix = path.suffix.lower()

    if suffix == ".json":
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            sample = data[0] if data else {}
            is_byt_format = "Mã" in sample and "Tên bệnh" in sample

            for row in data:
                if is_byt_format:
                    code = str(row.get("Mã", row.get("code", ""))).strip()
                    name = str(row.get("Tên bệnh", row.get("desc_vi", row.get("name", "")))).strip()
                else:
                    code = str(row.get("Mã bệnh", row.get("code", ""))).strip()
                    name = str(row.get("Tên bệnh gốc", row.get("desc_vi", row.get("name", "")))).strip()
                if not code:
                    continue
                if not name:
                    name = code
                idx.add(code, name)
                name_clean = _strip_parens(name)
                if name_clean and name_clean != name and name_clean.lower() not in idx.exact:
                    idx.exact.setdefault(name_clean.lower(), []).append(code)
        except Exception as exc:
            logger.error("build_from_seed: lỗi đọc %s: %s", path, exc)
    else:
        # Format JSONL — hỗ trợ cả icd10.jsonl mới (desc_vi+desc_en) và cũ (desc_en)
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                code = str(row.get("code", "")).strip()
                name = str(row.get("desc_vi", row.get("desc_en", row.get("name", "")))).strip()
                if not code:
                    continue
                if not name:
                    name = code
                idx.add(code, name)
                name_clean = _strip_parens(name)
                if name_clean and name_clean != name and name_clean.lower() not in idx.exact:
                    idx.exact.setdefault(name_clean.lower(), []).append(code)

    if out_index:
        out_index.parent.mkdir(parents=True, exist_ok=True)
        out_index.write_text(
            json.dumps(idx.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return idx


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #


if __name__ == "__main__":  # pragma: no cover
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="ICD-10 builder + lookup")
    p.add_argument("--build-from", type=Path, default=Path("data/icd_seed.jsonl"))
    p.add_argument("--out", type=Path, default=Path("data/icd_index.json"))
    p.add_argument("--lookup", type=str, default="")
    args = p.parse_args()

    if args.build_from.exists():
        idx = build_from_seed(args.build_from, args.out)
        print(f"Built ICD index: {len(idx.names)} names, {len(idx.exact)} keys")
    else:
        print(f"Seed file missing: {args.build_from}", file=sys.stderr)

    if args.lookup:
        ret = ICDRetriever()
        codes = ret.lookup(args.lookup)
        print(f"Lookup {args.lookup!r} -> {codes}")
        ret.save_index()
