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


# ---------------------------------------------------------------------- #
# VN medical abbreviations / synonyms (R27.6 mới 2026-07-10)
# ---------------------------------------------------------------------- #

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

    def _filter_and_sort_codes(self, codes: list[str], text: str, other_entities: list[dict] | None = None) -> list[str]:
        if not codes:
            return []
        filtered = _filter_irrelevant_codes(list(codes), text, self.idx)
        if not filtered:
            return []
        restricted = _restrict_chapter(filtered, text)
        result = restricted if restricted else filtered
        
        boosted_prefixes = _get_boosted_prefixes(other_entities)

        # Smart sorting: ưu tiên các chương bệnh phổ biến cho người lớn (I, J, K, E, N, M, S, T, C, D, G, A, B, R)
        # đẩy O (thai sản), P (sơ sinh), V/W/X/Y (tác nhân bên ngoài), Z (tiền căn) xuống dưới nếu không rõ
        def _chapter_priority(code: str) -> tuple[int, str]:
            if not code:
                return (99, code)
            ch = code[0].upper()
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

            idx = result.index(code) if code in result else 999
            return (penalty, 0 if code[0].upper() in ('I', 'J', 'K', 'E', 'N', 'M', 'C', 'D', 'G', 'A', 'B', 'R', 'S', 'T') else 1, idx)

        return sorted(set(result), key=_lex_score)

    def _split_compound_diagnosis(self, text: str) -> list[str]:
        """Tách chẩn đoán kép (Multi-hop / Conjunction splitting) thành các chẩn đoán riêng lẻ."""
        parts = re.split(r'\s+trên\s+nền\s+|\s+kèm\s+theo\s+|\s+kèm\s+|\s+biến\s+chứng\s+|\s+đồng\s+thời\s+|\s+hoặc\s+|\s+hay\s+', text, flags=re.IGNORECASE)
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
    ) -> list[str]:
        """Tra ICD-10 cho 1 cụm chẩn đoán tiếng Việt (có tự động tách chẩn đoán kép)."""
        if not vn_text:
            return []
        if not hasattr(self, '_cache'):
            self._cache = {}
        other_key = tuple(sorted((str(e.get("text", "")).strip().lower(), str(e.get("type", ""))) for e in (other_entities or []))) if other_entities else ()
        cache_key = (vn_text.strip().lower(), context_query, other_key)
        if cache_key in self._cache:
            return list(self._cache[cache_key])

        parts = self._split_compound_diagnosis(vn_text)
        if len(parts) > 1 and len(parts) <= 5:
            logger.debug("Multi-hop splitting '%s' → %s", vn_text, parts)
            out = []
            for p in parts:
                codes = self._lookup_single(p, context_query, other_entities)
                for c in codes:
                    if c not in out:
                        out.append(c)
            result = out[:4]
        else:
            result = self._lookup_single(vn_text, context_query, other_entities)

        if len(self._cache) > 4096:
            self._cache.clear()
        self._cache[cache_key] = result
        return list(result)

    def _lookup_single(
        self,
        vn_text: str,
        context_query: str | None = None,
        other_entities: list[dict] | None = None,
    ) -> list[str]:
        if not vn_text:
            return []

        text = self._strip_clinical_prefix(vn_text)
        # R27.6 mới 2026-07-10: normalize abbreviation + synonym VN TRƯỚC lookup chain
        text = _normalize_vn_term(text)

        # R27.7 mới 2026-07-10: short-circuit khi có direct match trong _icd_vn_to_codes
        if hasattr(self, '_icd_vn_to_codes'):
            key_lower = text.lower().strip()
            if key_lower in self._icd_vn_to_codes:
                logger.debug("L0 short-circuit direct match: '%s' → %s", text, self._icd_vn_to_codes[key_lower])
                return self._filter_and_sort_codes(self._icd_vn_to_codes[key_lower], text, other_entities=other_entities)[:2]

        # L1: Exact (cao độ tin cậy nhất — cap 2)
        key = text.lower()
        if key in self.idx.exact:
            return self._filter_and_sort_codes(self.idx.exact[key], text, other_entities=other_entities)[:2]

        # L1.5: VN prefix exact match — nếu text là prefix của desc_vi
        prefix_codes = self._exact_match_vn_prefix(text)
        if prefix_codes:
            return self._filter_and_sort_codes(prefix_codes, text, other_entities=other_entities)[:2]

        # L1.7 (NEW 2026-07-10): VN substring match (text chứa trong desc_vi)
        if len(text) >= 5:
            substring_codes = self._exact_match_vn_substring(text)
            if substring_codes:
                logger.debug("L1.7 substring match '%s' → %s", text, substring_codes[:2])
                return self._filter_and_sort_codes(substring_codes, text, other_entities=other_entities)[:2]

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
                return self._filter_and_sort_codes(sorted_codes[:6], text, other_entities=other_entities)[:2]

        # L4: Local fuzzy match trên names VN (threshold 75, cap 2)
        fuzzy_vn = self._fuzzy_local(text, threshold=75)
        if fuzzy_vn:
            return self._filter_and_sort_codes(fuzzy_vn, text, other_entities=other_entities)[:2]

        # L5: BM25 fallback (top-2 — strict hơn cho Set-based F1)
        if self.local_search is not None and hasattr(self.local_search, 'bm25_index'):
            bm25_codes, _ = self.local_search.bm25_index.search(bm25_query, top_k=2)
            if bm25_codes:
                bm25_codes = self._filter_and_sort_codes(bm25_codes, text, other_entities=other_entities)
                if bm25_codes:
                    return bm25_codes[:2]

        # L6 (NEW R27.7 2026-07-10): Aggressive final fallback - thử MULTIPLE strategies
        if len(text) >= 5:
            substring_codes = self._exact_match_vn_substring(text)
            if substring_codes:
                logger.debug("L6 substring fallback '%s' → %s", text, substring_codes[:2])
                return self._filter_and_sort_codes(substring_codes, text, other_entities=other_entities)[:2]

        if _text_matches_chapter_keyword(text):
            prefix_codes = self._exact_match_vn_prefix(text)
            if prefix_codes:
                return self._filter_and_sort_codes(prefix_codes, text, other_entities=other_entities)[:2]
            chapter_codes = self._chapter_codes_lookup(text)
            if chapter_codes:
                logger.debug("L6 chapter lookup '%s' → %s", text, chapter_codes[:2])
                return self._filter_and_sort_codes(chapter_codes, text, other_entities=other_entities)[:2]

        if self.local_search is not None:
            low_threshold_codes = self.local_search.search(
                text, threshold=0.40, top_k=5
            ) or []
            if low_threshold_codes:
                logger.debug("L6 low-threshold vector '%s' → %s", text, low_threshold_codes[:2])
                return self._filter_and_sort_codes(low_threshold_codes, text, other_entities=other_entities)[:2]

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
        self._icd_vn_to_codes = {
            "ung thư phổi": ["C34", "C34.0", "C34.1", "C34.2", "C34.3", "C34.8", "C34.9"],
            "ung thư phổi không tế bào nhỏ": ["C34", "C34.0", "C34.1", "C34.2", "C34.3", "C34.8", "C34.9"],
            "ung thư phổi tế bào nhỏ": ["C34", "C34.0", "C34.1", "C34.2", "C34.3", "C34.8", "C34.9"],
            "u ác tính phổi": ["C34", "C34.0", "C34.1", "C34.2", "C34.3", "C34.8", "C34.9"],
            "k phổi": ["C34", "C34.0", "C34.1", "C34.2", "C34.3", "C34.8", "C34.9"],
            "ung thư não": ["C71", "C71.0", "C71.1", "C71.2", "C71.3", "C71.4", "C71.5", "C71.6", "C71.7", "C71.8", "C71.9"],
            "u não": ["C71", "C71.0", "C71.1", "C71.2", "C71.3", "C71.4", "C71.5", "C71.6", "C71.7", "C71.8", "C71.9"],
            "u ác tính não": ["C71", "C71.0", "C71.1", "C71.2", "C71.3", "C71.4", "C71.5", "C71.6", "C71.7", "C71.8", "C71.9"],
            "di căn não": ["C79.3", "C79.30", "C79.31", "C79.32"],
            "u ác tính thứ phát ở não": ["C79.3", "C79.30", "C79.31", "C79.32"],
            "di căn xương": ["C79.5"],
            "di căn gan": ["C78.7"],
            "di căn phổi": ["C78.0"],
            "di căn": ["C79", "C79.0", "C79.1", "C79.2", "C79.3", "C79.4", "C79.5", "C79.6", "C79.7", "C79.8", "C79.9"],
            "ung thư vú": ["C50", "C50.0", "C50.1", "C50.2", "C50.3", "C50.4", "C50.5", "C50.6", "C50.8", "C50.9"],
            "ung thư gan": ["C22", "C22.0", "C22.1", "C22.2", "C22.3", "C22.4", "C22.7", "C22.8", "C22.9"],
            "ung thư dạ dày": ["C16", "C16.0", "C16.1", "C16.2", "C16.3", "C16.4", "C16.5", "C16.6", "C16.8", "C16.9"],
            "ung thư đại tràng": ["C18", "C18.0", "C18.1", "C18.2", "C18.3", "C18.4", "C18.5", "C18.6", "C18.7", "C18.8", "C18.9"],
            "ung thư trực tràng": ["C20"],
            "tăng huyết áp": ["I10"],
            "tăng huyết áp vô căn": ["I10"],
            "tăng huyết áp thứ phát": ["I15"],
            "cao huyết áp": ["I10"],
            "nhồi máu cơ tim": ["I21", "I21.0", "I21.1", "I21.2", "I21.3", "I21.4", "I21.9"],
            "đau thắt ngực": ["I20", "I20.0", "I20.1", "I20.8", "I20.9"],
            "suy tim": ["I50", "I50.0", "I50.1", "I50.9"],
            "rung nhĩ": ["I48", "I48.0", "I48.1", "I48.2", "I48.3", "I48.4", "I48.9"],
            "ngoại tâm thu thất": ["I49.3"],
            "ngoại tâm thu nhĩ": ["I49.1"],
            "sa van hai lá": ["I34.1"],
            "sa van 2 lá": ["I34.1"],
            "sa van mitral": ["I34.1"],
            "hở van hai lá": ["I34.0"],
            "hở van 2 lá": ["I34.0"],
            "hẹp van hai lá": ["I34.2"],
            "tắc mạch huyết khối": ["I82", "I82.0", "I82.1", "I82.2", "I82.3", "I82.8", "I82.9"],
            "tắc mạch": ["I82", "I82.0", "I82.1", "I82.2", "I82.3", "I82.8", "I82.9"],
            "huyết khối": ["I82"],
            "hen phế quản": ["J45", "J45.0", "J45.1", "J45.8", "J45.9"],
            "hen suyễn": ["J45", "J45.0", "J45.1", "J45.8", "J45.9"],
            "viêm phổi": ["J12", "J13", "J14", "J15", "J16", "J17", "J18"],
            "viêm tuyến mồ hôi": ["L73.2"],  # mới 2026-07-10: R27.7 — match L73.2 (Hidradenitis suppurativa)
            "viêm tuyến mồ hôi mủ": ["L73.2"],
            "nhọt ổ gà": ["L73.2"],
            "đái tháo đường": ["E11", "E11.0", "E11.1", "E11.2", "E11.3", "E11.4", "E11.5", "E11.6", "E11.7", "E11.8", "E11.9"],
            "đái tháo đường type 2": ["E11"],
            "suy thận": ["N17", "N18", "N19"],
            "suy thận cấp": ["N17"],
            "suy thận mạn": ["N18", "N18.1", "N18.2", "N18.3", "N18.4", "N18.5", "N18.6", "N18.9"],
            "viêm gan b": ["B16", "B18.1"],
            "viêm gan c": ["B17.1", "B18.2"],
            "xơ gan": ["K74", "K74.0", "K74.1", "K74.2", "K74.3", "K74.4", "K74.5", "K74.6"],
            "sỏi thận": ["N20", "N20.0", "N20.1", "N20.2", "N20.9"],
            "thoát vị đĩa đệm": ["M51", "M51.0", "M51.1", "M51.2", "M51.3", "M51.4", "M51.5", "M51.6", "M51.7", "M51.8", "M51.9"],
            "đột quỵ": ["I63", "I64"],
            "tai biến mạch máu não": ["I63", "I64"],
            "thiếu máu": ["D50", "D50.0", "D50.1", "D50.2", "D50.3", "D50.4", "D50.5", "D50.6", "D50.7", "D50.8", "D50.9"],

            # === MỚI 2026-07-10 — abbreviations VN (R27.6) ===
            # VN doctors dùng viết tắt rất phổ biến. Map trước lookup chain.
            "tha": ["I10"],  # Tăng huyết áp
            "nmct": ["I21", "I21.0", "I21.1", "I21.2", "I21.3", "I21.4", "I21.9"],  # Nhồi máu cơ tim
            "đtđ": ["E11", "E11.0", "E11.1", "E11.2", "E11.3", "E11.4", "E11.5", "E11.6", "E11.7", "E11.8", "E11.9"],  # Đái tháo đường
            "đtđ type 2": ["E11"],  # ĐTĐ type 2
            "đtđ type 1": ["E10"],  # ĐTĐ type 1
            "tbmmn": ["I63", "I64"],  # Tai biến mạch máu não
            "tbmmnn": ["I63", "I64"],  # Tai biến mạch máu não
            "copd": ["J44", "J44.0", "J44.1", "J44.8", "J44.9"],  # Bệnh phổi tắc nghẽn mạn
            "osa": ["G47.3"],  # Ngưng thở khi ngủ
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
            "tử vong": ["R96", "R96.1", "R98"],
            "hc": ["R59"],  # Hạch (general lymphadenopathy)

            # === MỚI 2026-07-10 — synonyms cho u (R27.6) ===
            # ICD desc_vi dùng "U ác tính" nhưng LLM hay ghi "u ác" / "khối u"
            "u ác tính": ["C", "C00", "C80"],  # placeholder - cần organ context
            "u ác": ["C", "C00", "C80"],  # placeholder - cần organ context
            "u lành": ["D", "D00", "D48"],  # placeholder - cần organ context
            "u lành tính": ["D", "D00", "D48"],  # placeholder
            "khối u": ["D48"],  # unspecified neoplasm (placeholder)
            "neoplasm": ["D48"],

            # === MỚI 2026-07-10 — Organ-based mappings (R27.6) ===
            # Pattern: "u ác/khối u/u ác tính [organ]" → Cxx (malignant) hoặc Dxx (benign)
            # Direct dict cho common organs để match exact LLM text
            "u ác tính trực tràng": ["C20"],
            "u ác trực tràng": ["C20"],
            "ung thư trực tràng": ["C20"],
            "khối u trực tràng": ["C20", "D12"],  # ambiguous - default malignant first
            "u lành trực tràng": ["D12"],
            "u lành tính trực tràng": ["D12"],

            "u ác tính đại tràng": ["C18", "C18.0", "C18.1", "C18.2", "C18.3", "C18.4", "C18.5", "C18.6", "C18.7", "C18.8", "C18.9"],
            "u ác đại tràng": ["C18"],
            "ung thư đại tràng": ["C18"],
            "khối u đại tràng": ["C18", "D12"],

            "u ác tính dạ dày": ["C16", "C16.0", "C16.1", "C16.2", "C16.3", "C16.4", "C16.5", "C16.6", "C16.8", "C16.9"],
            "u ác dạ dày": ["C16"],
            "ung thư dạ dày": ["C16"],
            "khối u dạ dày": ["C16", "D13.1"],

            "u ác tính gan": ["C22", "C22.0", "C22.1", "C22.2", "C22.3", "C22.4", "C22.7", "C22.8", "C22.9"],
            "u ác gan": ["C22"],
            "ung thư gan": ["C22"],
            "khối u gan": ["C22", "D13.4"],

            "u ác tính phổi": ["C34", "C34.0", "C34.1", "C34.2", "C34.3", "C34.8", "C34.9"],
            "u ác phổi": ["C34"],
            "khối u phổi": ["C34", "D14.3"],

            "u ác tính vú": ["C50", "C50.0", "C50.1", "C50.2", "C50.3", "C50.4", "C50.5", "C50.6", "C50.8", "C50.9"],
            "u ác vú": ["C50"],
            "khối u vú": ["C50", "D24"],

            "u ác tính não": ["C71", "C71.0", "C71.1", "C71.2", "C71.3", "C71.4", "C71.5", "C71.6", "C71.7", "C71.8", "C71.9"],
            "u não ác tính": ["C71"],
            "khối u não": ["C71", "D33"],

            "u ác tính buồng trứng": ["C56"],
            "u ác buồng trứng": ["C56"],
            "khối u buồng trứng": ["C56", "D27"],

            "u ác tính cổ tử cung": ["C53", "C53.0", "C53.1", "C53.8", "C53.9"],
            "u ác cổ tử cung": ["C53"],
            "khối u cổ tử cung": ["C53", "D26.0"],

            "u ác tính tuyến giáp": ["C73"],
            "khối u tuyến giáp": ["C73", "D34"],

            "u ác tính tuyến tiền liệt": ["C61"],
            "khối u tuyến tiền liệt": ["C61", "D29.1"],

            "u ác tính bàng quang": ["C67", "C67.0", "C67.1", "C67.2", "C67.3", "C67.4", "C67.5", "C67.6", "C67.7", "C67.8", "C67.9"],
            "khối u bàng quang": ["C67", "D30.3"],

            "u ác tính thận": ["C64"],
            "khối u thận": ["C64", "D30.0"],

            "u ác tính tụy": ["C25", "C25.0", "C25.1", "C25.2", "C25.3", "C25.4", "C25.7", "C25.8", "C25.9"],
            "khối u tụy": ["C25", "D13.6"],

            "u ác tính thực quản": ["C15", "C15.0", "C15.1", "C15.2", "C15.3", "C15.4", "C15.5", "C15.8", "C15.9"],
            "khối u thực quản": ["C15", "D13.0"],

            # === TIM MẠCH (Cardiology) — chuẩn BYT/WHO ICD-10 2026 ===
            # Nhồi máu cơ tim và ECG findings
            "nhồi máu cơ tim cấp st chênh lên": ["I21.0", "I21.1", "I21.2", "I21.3"],
            "nmct cấp st chênh lên": ["I21.0", "I21.1", "I21.2", "I21.3"],
            "nhồi máu cơ tim cấp không st chênh lên": ["I21.4"],
            "nmct cấp không st chênh lên": ["I21.4"],
            "nhồi máu cơ tim cũ": ["I25.2"],
            "nmct cũ": ["I25.2"],
            "bệnh tim thiếu máu cục bộ": ["I25"],
            "bệnh mạch vành": ["I25", "I25.1"],
            "hội chứng vành cấp": ["I24"],
            "đau thắt ngực không ổn định": ["I20.0"],
            "đau thắt ngực ổn định": ["I20.8"],
            "st chênh lên": ["I21.3"],   # STEMI (acute)
            "st chênh xuống": ["I21.4"],  # NSTEMI pattern
            "st chênh lên v1-v4": ["I21.0"],
            "st chênh lên v1-v6": ["I21.0"],
            "sóng q bệnh lý": ["I25.2"],  # old MI
            "sóng t đảo ngược": ["I24.8"],
            # Rối loạn nhịp tim
            "cuồng nhĩ": ["I48.3"],
            "rung nhĩ kèm đáp ứng thất nhanh": ["I48"],
            "nhịp nhanh trên thất": ["I47.1"],
            "nhịp nhanh thất": ["I47.2"],
            "rung thất": ["I49.0"],
            "block nhánh trái": ["I44.4", "I44.5", "I44.6"],
            "block nhánh phải": ["I45.0", "I45.1", "I45.2"],
            "block nhánh": ["I44.4", "I45.0"],
            "hội chứng sick sinus": ["I49.5"],
            "nhịp nhanh xoang": ["R00.0"],
            "nhịp chậm xoang": ["R00.1"],
            # Block nhĩ thất
            "block nhĩ thất": ["I44", "I44.0", "I44.1", "I44.2", "I44.3"],
            "block nhĩ thất độ 1": ["I44.0"],
            "block nhĩ thất độ 2": ["I44.1"],
            "block nhĩ thất độ 3": ["I44.2"],
            "block nhĩ thất hoàn toàn": ["I44.2"],
            "block tim": ["I44"],
            # Van tim
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
            # Viêm cơ tim, màng ngoài tim
            "viêm nội tâm mạc": ["I33", "I33.0", "I33.9"],
            "viêm nội tâm mạc nhiễm khuẩn": ["I33.0"],
            "viêm cơ tim": ["I40", "I40.0", "I40.1", "I40.8", "I40.9"],
            "viêm màng ngoài tim": ["I30"],
            "tràn dịch màng tim": ["I31.3"],
            "bệnh cơ tim": ["I42"],
            "bệnh cơ tim phì đại": ["I42.1"],
            "bệnh cơ tim giãn": ["I42.0"],
            # Suy tim theo mức độ
            "suy tim độ i nyha": ["I50.9"],
            "suy tim độ ii nyha": ["I50.9"],
            "suy tim độ iii nyha": ["I50.0"],
            "suy tim độ iv nyha": ["I50.0"],
            "suy tim tâm thu": ["I50.1"],
            "suy tim tâm trương": ["I50.9"],
            "suy tim cấp": ["I50.9"],
            "suy tim mạn": ["I50.9"],
            # Mạch máu
            "huyết khối tĩnh mạch sâu": ["I82.4"],
            "dvt": ["I82.4"],
            "thuyên tắc phổi": ["I26", "I26.0", "I26.9"],
            "pe": ["I26.9"],

            # === MỚI 2026-07-10 — TIÊU HÓA, GAN, MẬT ===
            "viêm gan virus": ["B15", "B16", "B17", "B18", "B19"],
            "viêm gan b": ["B16", "B18.1"],
            "viêm gan c": ["B17.1", "B18.2"],
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
            "thoát vị": ["K40", "K41", "K42", "K43", "K44", "K45", "K46"],

            # === MỚI 2026-07-10 — HÔ HẤP ===
            "viêm phế quản": ["J20", "J21"],
            "viêm phế quản cấp": ["J20"],
            "viêm phế quản mạn": ["J42"],
            "viêm tiểu phế quản": ["J21"],
            "hen phế quản": ["J45", "J45.0", "J45.1", "J45.8", "J45.9"],
            "hen suyễn": ["J45"],
            "viêm phổi mắc phải cộng đồng": ["J18", "J18.9"],
            "viêm phổi cộng đồng": ["J18", "J18.9"],
            "tràn khí màng phổi": ["J93"],
            "tràn dịch màng phổi": ["J90"],
            "xẹp phổi": ["J98.1"],
            "lao phổi": ["A15", "A15.0"],

            # === MỚI 2026-07-10 — THẦN KINH, TÂM THẦN ===
            "động kinh": ["G40", "G40.0", "G40.1", "G40.2", "G40.3", "G40.4", "G40.5", "G40.6", "G40.7", "G40.8", "G40.9"],
            "parkinson": ["G20", "G21"],
            "bệnh parkinson": ["G20"],
            "alzheimer": ["G30", "G30.0", "G30.1", "G30.8", "G30.9"],
            "sa sút trí tuệ": ["F03"],
            "trầm cảm": ["F32", "F32.0", "F32.1", "F32.2", "F32.3", "F32.8", "F32.9"],
            "rối loạn lo âu": ["F41", "F41.0", "F41.1", "F41.2", "F41.3", "F41.8", "F41.9"],
            "tâm thần phân liệt": ["F20"],
            "đau nửa đầu": ["G43", "G43.0", "G43.1", "G43.2", "G43.3", "G43.8", "G43.9"],
            "migraine": ["G43"],

            # === MỚI 2026-07-10 — THẬN, TIẾT NIỆU ===
            "sỏi thận": ["N20"],
            "sỏi niệu quản": ["N20.1"],
            "sỏi bàng quang": ["N21.0"],
            "viêm đường tiết niệu": ["N39.0"],
            "viêm bàng quang": ["N30"],
            "viêm thận - bể thận": ["N10", "N11", "N12"],
            "viêm bể thận": ["N12"],
            "viêm cầu thận": ["N00", "N01", "N02", "N03", "N04", "N05", "N06", "N07", "N08"],
            "hội chứng thận hư": ["N04"],
            "suy thận cấp": ["N17"],
            "suy thận mạn": ["N18"],
            "suy thận giai đoạn cuối": ["N18.6"],

            # === MỚI 2026-07-10 — CƠ XƯƠNG KHỚP ===
            "thoái hóa khớp": ["M15", "M16", "M17", "M18", "M19"],
            "thoái hóa khớp gối": ["M17"],
            "thoái hóa khớp háng": ["M16"],
            "viêm khớp dạng thấp": ["M05", "M06"],
            "gout": ["M10", "M10.0", "M10.1", "M10.2", "M10.3", "M10.4", "M10.9"],
            "gút": ["M10"],
            "loãng xương": ["M80", "M81", "M82"],
            "viêm cơ": ["M60"],
            "đau cơ xơ hóa": ["M79.7"],

            # === MỚI 2026-07-10 — DA, MÔ LIÊN KẾT ===
            "vẩy nến": ["L40", "L40.0", "L40.1", "L40.2", "L40.3", "L40.4", "L40.5", "L40.8", "L40.9"],
            "eczema": ["L20", "L30"],
            "viêm da cơ địa": ["L20"],
            "viêm da tiếp xúc": ["L23", "L24", "L25"],
            "mày đay": ["L50"],
            "zona": ["B02"],
            "herpes": ["B00"],

            # === MỚI 2026-07-10 — NỘI TIẾT ===
            "basedow": ["E05.0"],
            "cường giáp": ["E05"],
            "suy giáp": ["E03", "E03.0", "E03.1", "E03.2", "E03.3", "E03.4", "E03.5", "E03.8", "E03.9"],
            "suy tuyến yên": ["E23.0"],
            "rối loạn lipid máu": ["E78"],
            "tăng cholesterol máu": ["E78.0"],
            "tăng triglyceride máu": ["E78.1"],
            "béo phì": ["E66"],
            "suy dinh dưỡng": ["E40", "E41", "E42", "E43", "E44", "E45", "E46"],
            "loãng xương": ["M80", "M81"],

            # === MỚI 2026-07-10 — MÁU ===
            "thiếu máu thiếu sắt": ["D50"],
            "thiếu máu mạn": ["D50", "D63"],
            "thiếu máu cấp": ["D62"],
            "leukemia": ["C91", "C92", "C93", "C94", "C95"],
            "lymphoma": ["C81", "C82", "C83", "C84", "C85", "C86", "C88", "C96"],
            "xuất huyết giảm tiểu cầu": ["D69.3"],
            "đông máu nội quản lý tán": ["D65"],
            "dic": ["D65"],

            # === MỚI 2026-07-10 — TAI MŨI HỌNG, MẮT ===
            "viêm xoang": ["J32"],
            "viêm tai giữa": ["H66"],
            "viêm amidan": ["J03"],
            "đục thủy tinh thể": ["H25", "H26"],
            "glocom": ["H40", "H40.0", "H40.1", "H40.2", "H40.3", "H40.4", "H40.5", "H40.6", "H40.8", "H40.9"],
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
        }

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
                        code = str(row.get("Mã", "")).strip()
                        name_vi = str(row.get("Tên bệnh", "")).strip()
                        nhom = str(row.get("Nhóm bệnh", "")).strip()
                        mo_ta = str(row.get("Mô tả", "")).strip()
                    else:
                        # ICD10_Data.json cũ
                        code = str(row.get("Mã bệnh", "")).strip()
                        name_vi = str(row.get("Tên bệnh gốc", "")).strip()
                        nhom = str(row.get("Tên nhóm", "")).strip()
                        mo_ta = ""

                    if not (code and name_vi):
                        continue
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

                    if not code:
                        continue
                    # Prefer desc_vi cho embedding (VN queries); fallback desc_en
                    # Nếu có cả 2 thì embed cả 2 để hybrid VN-EN match
                    if desc_vi and desc_en and desc_vi != desc_en:
                        desc = f"{desc_vi} | {desc_en}"
                    elif desc_vi:
                        desc = desc_vi
                    elif desc_en:
                        desc = desc_en
                    else:
                        continue
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

        # 1. Khởi tạo model SentenceTransformer
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                self._model = SentenceTransformer("BAAI/bge-m3")
                logger.info(
                    "ICD10VectorSearch: Đã tải mô hình BGE-M3 phục vụ tìm kiếm."
                )
            except ImportError:
                logger.error(
                    "Chưa cài sentence-transformers! Không thể chạy Vector Search."
                )
                return []

        # 2. Nếu file .npy chưa tồn tại, tự động sinh và lưu luôn tại đây
        if self._embeddings is None:
            logger.info(
                "ICD10VectorSearch: Không tìm thấy file embeddings.npy có sẵn. Đang sinh trực tiếp..."
            )
            try:
                t0 = time.time()
                self._embeddings = self._model.encode(
                    self.descs_raw,
                    batch_size=128,
                    show_progress_bar=True,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )
                np.save(self._embeddings_path, self._embeddings)
                logger.info(
                    "Đã lưu ma trận nhúng ra %s (%.2fs)",
                    self._embeddings_path,
                    time.time() - t0,
                )
            except Exception as exc:
                logger.error("Lỗi sinh embeddings: %s", exc)
                return []

        # 3. Mã hóa câu truy vấn
        query_vec = self._model.encode(
            query, normalize_embeddings=True, convert_to_numpy=True
        )  # Shape: (1024,)
        # Cast về cùng dtype với embeddings (float16 tiết kiệm RAM)
        if self._embeddings is not None and self._embeddings.dtype != query_vec.dtype:
            query_vec = query_vec.astype(self._embeddings.dtype)

        # 4. Tính Cosine Similarity bằng dot-product (vì vector đã được chuẩn hóa L2)
        scores = np.dot(self._embeddings, query_vec)  # Shape: (71705,)

        # 5. Sắp xếp lấy Top K
        top_indices = np.argsort(-scores)[:top_k]

        out: list[str] = []
        for idx in top_indices:
            score = float(scores[idx])
            # Có thể lọc theo threshold (Cosine Similarity thường từ 0.0 -> 1.0)
            if score >= threshold:
                code = self.codes[idx]
                if code not in out:
                    out.append(code)

        logger.debug(
            "Vector search '%s' -> Top 1 score: %.3f (%s)",
            query,
            scores[top_indices[0]] if len(top_indices) > 0 else 0.0,
            out[0] if out else "None",
        )
        return out

    def score_codes(self, query: str, codes: list[str]) -> dict[str, float]:
        """Tính cosine similarity cho 1 tập codes cụ thể.

        Hữu ích cho hybrid search: vector_search.search() chỉ trả codes (top-k),
        hybrid cần re-score các candidates từ BM25 để combine đúng tỷ trọng.

        Returns: dict {code: cosine_sim} (giá trị ∈ [0, 1] cho vectors đã chuẩn hoá L2).
        """
        self._ensure_loaded()
        if not query or not codes or self._embeddings is None:
            return {}

        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                self._model = SentenceTransformer("BAAI/bge-m3")
            except ImportError:
                logger.error("score_codes: thiếu sentence-transformers.")
                return {}

        q_vec = self._model.encode(
            query, normalize_embeddings=True, convert_to_numpy=True
        )
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
                        if not (code and desc_vi):
                            continue
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
                        if not (code and (desc_en or desc_vi)):
                            continue
                        doc_text = self._build_doc_text(desc_vi or desc_en, code)
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
                    code = str(row.get("Mã", "")).strip()
                    name = str(row.get("Tên bệnh", "")).strip()
                else:
                    code = str(row.get("Mã bệnh", "")).strip()
                    name = str(row.get("Tên bệnh gốc", "")).strip()
                if code and name:
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
                # Ưu tiên desc_vi cho exact match (VN queries); fallback desc_en
                name = str(row.get("desc_vi", row.get("desc_en", row.get("name", "")))).strip()
                if code and name:
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
