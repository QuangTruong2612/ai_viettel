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


class Translator:
    """Dịch cụm ngắn VN -> EN. 2 tầng:
    1. Dict cache (preset các bệnh phổ biến).
    2. LLM nếu cache miss (qwen3.5-4b đã đủ thông minh cho cụm 1-3 từ y khoa).
    """

    def __init__(
        self, llm_client: Any = None, cache_path: Optional[Path] = None
    ) -> None:
        self.llm_client = llm_client
        self._cache: dict[str, str] = {}
        self._cache_path = cache_path or (DATA_DIR / "translation_cache.json")
        if self._cache_path.exists():
            try:
                self._cache = json.loads(self._cache_path.read_text(encoding="utf-8"))
            except Exception:
                self._cache = {}

    # ------------------------------------------------------------------ #

    def preset(self, mapping: dict[str, str]) -> None:
        """Thêm mapping thủ công vào cache (verbatim + diacritic-stripped).

        Bug history:
        1. Trước kia cache chỉ key theo lowercase literal → query không dấu miss.
        2. NFKD không strip được ký tự `đ` (U+0111 LATIN SMALL LETTER D WITH STROKE)
           vì nó không có decomposition → fix bằng replace thủ công.

        Fix: lưu CẢ key literal và key bỏ dấu + xử lý đ/Đ đặc biệt.
        """
        import unicodedata as _ud

        def _strip(s: str) -> str:
            # Xử lý đ/Đ TRƯỚC khi NFKD (NFKD không strip được)
            s = s.lower().strip().replace("đ", "d").replace("Đ", "D")
            nfkd = _ud.normalize("NFKD", s)
            return "".join(c for c in nfkd if not _ud.combining(c))

        for vi, en in mapping.items():
            vi_l = vi.lower().strip()
            self._cache[vi_l] = en.strip()
            self._cache[_strip(vi)] = en.strip()

    def save_cache(self) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(
            json.dumps(self._cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------ #

    def translate(self, text: str) -> str:
        """Dịch 1 cụm tiếng Việt sang tiếng Anh (y khoa, ngắn gọn).

        Trả về tiếng Anh nếu thành công; ngược lại trả về text gốc để caller vẫn tra được.

        Lookup cache 2 lần: (1) key literal lowercase, (2) key sau khi bỏ dấu + đ.
        """
        if not text:
            return ""
        import unicodedata as _ud

        key_literal = text.lower().strip()
        if key_literal in self._cache:
            return self._cache[key_literal]
        # Xử lý đặc biệt cho 'đ' trước khi NFKD
        text_for_strip = key_literal.replace("đ", "d").replace("Đ", "D")
        nfkd = _ud.normalize("NFKD", text_for_strip)
        key_strip = "".join(c for c in nfkd if not _ud.combining(c))
        if key_strip in self._cache:
            return self._cache[key_strip]

        if self.llm_client is None:
            return text  # Không có LLM → trả gốc

        # Gọi LLM
        prompt = (
            "You are a professional medical translator. Translate the following Vietnamese clinical diagnosis "
            "term or abbreviation into standard English medical terminology used in clinical records.\n\n"
            "Guidelines:\n"
            "1. Output ONLY the translated English phrase. No explanations, no markdown, no quotes, no extra words.\n"
            "2. Expand and translate common Vietnamese medical abbreviations:\n"
            "   - THA -> essential hypertension\n"
            "   - ĐTĐ / ĐTĐ type 2 / ĐTĐ type II -> type 2 diabetes mellitus\n"
            "   - COPD -> chronic obstructive pulmonary disease\n"
            "   - NMCT -> myocardial infarction\n"
            "   - ST -> heart failure\n"
            "   - VP -> pneumonia\n"
            "   - VPQ -> bronchopneumonia\n"
            "   - HPQ / Hen phế quản -> asthma\n"
            "   - CVA -> cerebrovascular accident\n"
            "3. Translate clinical modifiers accurately:\n"
            "   - 'cấp' -> 'acute'\n"
            "   - 'mạn' / 'mạn tính' -> 'chronic'\n"
            "   - 'cộng đồng' -> 'community-acquired'\n"
            "   - 'độ I/II/III' -> 'grade I/II/III' (e.g., 'suy tim độ III' -> 'heart failure grade III')\n"
            "   - 'giai đoạn I/II/III/IV' -> 'stage I/II/III/IV'\n"
            "4. Keep it concise, clinical, and standard. Avoid layperson terms (e.g., use 'essential hypertension' instead of 'high blood pressure').\n\n"
            f"Vietnamese: {text}\n"
            "English:"
        )
        try:
            msg = [{"role": "user", "content": prompt}]
            resp = self.llm_client._client.chat.completions.create(  # noqa: SLF001
                model=self.llm_client.config.model,
                messages=msg,
                temperature=0.0,
                max_tokens=64,
            )
            en = (resp.choices[0].message.content or "").strip().strip('"').strip("'")
            en = re.sub(r"^English:\s*", "", en, flags=re.IGNORECASE)
            if en:
                # Lưu cả key literal và key strip để lần sau hit
                self._cache[key_literal] = en
                self._cache[key_strip] = en
                return en
        except Exception as exc:
            logger.warning("Translate lỗi (%r): %s", text, exc)
        return text


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

        # O00-O9A: pregnancy
        if code.startswith("O") and len(code) >= 2 and code[1].isdigit() and code[:2] < "O9":
            if any(kw in entity_lower for kw in (
                "pregnancy", "mang thai", "thai kỳ", "obstetric", "gestation",
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

    Pipeline 5 lớp (chạy hoàn toàn local, KHÔNG gọi NIH API):
      L1: Exact match dict tiếng Anh (prebuilt, ICDIndex)
      L2: VN -> EN translation (Translator, cached)
      L3: Semantic extraction (BGE-M3 cosine ≥ 0.7) — trả TẤT CẢ codes match,
            không cap top-K. BM25 keyword dùng để mở rộng candidates.
      L4: Fuzzy match EN (rapidfuzz, partial_ratio)
      L5: Fuzzy match VN (rapidfuzz, partial_ratio)

    Mặc định `use_hybrid=True`; truyền `use_hybrid=False` để fallback về vector-only.
    """

    def __init__(
        self,
        index_path: Optional[Path] = None,
        translator: Optional[Translator] = None,
        local_search: Optional["ICD10VectorSearch | ICD10HybridSearch"] = None,
        use_hybrid: bool = True,
        hybrid_alpha: float = 0.6,
        hybrid_beta: float = 0.4,
    ) -> None:
        self.idx = self._load_index(index_path)
        self.translator = translator or Translator()
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

    # ------------------------------------------------------------------ #

    def _load_index(self, path: Optional[Path]) -> ICDIndex:
        path = path or (DATA_DIR / "icd_index.json")
        if not path.exists():
            # Fallback chain: icd10.jsonl mới → BYT → ICD10_Data cũ
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

    def lookup(
        self,
        vn_text: str,
        context_query: str | None = None,
        other_entities: list[dict] | None = None,
    ) -> list[str]:
        """Tra ICD-10 cho 1 cụm chẩn đoán tiếng Việt.

        Format dữ liệu mới (ICD10_Data.json): Mã bệnh + Tên bệnh gốc (TIẾNG VIỆT).
        → Match trực tiếp VN-VN, KHÔNG cần Translate VN→EN.

        Args:
            vn_text: text gốc (VN). Có thể đã qua rescan (LLM đã chuẩn hóa).
            context_query: KHÔNG DÙNG — để giữ signature tương thích.
            other_entities: list các entities khác trong cùng record
                (drugs/symptoms). Dùng để build BM25 keyword-rich query.
                Mặc định: rỗng.

        Returns: list codes (string), unique + sorted.

        Bug history:
        1. Trước kia top_k=20 → quá nhiều noise. Fix: top_k=3 (strict).
        2. Trước kia BM25 keyword dùng raw en_query (không có context) → fail
           khi input chỉ nói "suy thận". Fix: enrich BM25 query với nearby
           drugs/symptoms via build_context_query.
        3. Format cũ dùng desc_en ICD NIH (English); data mới 2026-07 dùng
           ICD10_Data.json (VN). Match trực tiếp VN-VN, bỏ qua Translate step.
        """
        if not vn_text:
            return []

        text = self._strip_clinical_prefix(vn_text)

        # L1: Exact (nếu input đã chuẩn - vd "viêm phổi cộng đồng")
        key = text.lower()
        if key in self.idx.exact:
            return sorted(set(self.idx.exact[key]))

        # L2: Build query có context cho BM25 (dùng nearby drugs/symptoms)
        bm25_query = build_context_query(text, "CHẨN_ĐOÁN", other_entities)
        # Vector vẫn dùng text gốc (không contaminate embedding - bug history #4)

        # L3: Hybrid search (BGE-M3 cosine ≥ 0.5 + BM25 mở rộng candidates)
        # Threshold 0.5 (không phải 0.7) vì:
        # - VN query vs EN+VN concatenated desc: cosine ~0.55-0.70
        # - 0.7 quá strict → miss 70% medical terms
        # - 0.5 balance giữa precision và recall
        if self.local_search is not None:
            local_results = self.local_search.search(
                text, threshold=0.5  # giảm từ 0.7 để bắt VN→EN semantic
            )
            if local_results:
                local_results = _filter_irrelevant_codes(local_results, text, self.idx)
                if local_results:
                    return local_results

        # L4: Local fuzzy match trên names VN (low threshold để bắt substring)
        fuzzy_vn = self._fuzzy_local(text, threshold=70)
        if fuzzy_vn:
            fuzzy_vn = _filter_irrelevant_codes(fuzzy_vn, text, self.idx)
            if fuzzy_vn:
                return fuzzy_vn

        # L5: BM25 fallback (nếu hybrid fail)
        if self.local_search is not None and hasattr(self.local_search, 'bm25_index'):
            bm25_codes, _ = self.local_search.bm25_index.search(bm25_query, top_k=5)
            if bm25_codes:
                bm25_codes = _filter_irrelevant_codes(bm25_codes, text, self.idx)
                if bm25_codes:
                    return bm25_codes

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
        """Chọn file JSON/JSONL mới nhất có sẵn (ưu tiên WHO 2019 VN+EN)."""
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
