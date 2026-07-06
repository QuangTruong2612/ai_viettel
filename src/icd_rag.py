"""ICD-10 RAG — tra cứu mã ICD-10 cho chẩn đoán.

API: https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search?terms={query}
      -> trả về danh sách {code, name} cho tiếng Anh.

Vấn đề: input/output là tiếng Việt (giữ nguyên text gốc trong output JSON),
nhưng bảng ICD-10 chỉ có tiếng Anh. Cần:
1. Dịch VN -> EN cho diagnosis term trước khi query.
2. Tra ICD-10 bằng EN query.
3. Gắn mã code vào "candidates".

Pipeline 4 lớp:
  L1: Exact match dict tiếng Anh (prebuilt)
  L2: VN -> EN translation (LLM gọi 1 lần)
  L3: ICD-10 REST search với key EN
  L4: Fuzzy match bằng rapidfuzz (nếu exact miss)
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

import requests

try:
    from rank_bm25 import BM25Okapi  # type: ignore

    _HAS_BM25 = True
except ImportError:  # pragma: no cover
    BM25Okapi = None  # type: ignore
    _HAS_BM25 = False

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

ICD_API = "https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search"


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
# ICD lookup
# ---------------------------------------------------------------------- #


def _http_search(
    query: str, max_results: int = 8, timeout: int = 15
) -> list[tuple[str, str]]:
    """Gọi clinicaltables NIH; trả [(code, name_en), ...].

    Response format với sf=name (không có df):
        [count, [code, code, ...], null, [[code, name], [code, name], ...]]
      - data[0] = total count
      - data[1] = list codes (top maxList)
      - data[2] = null
      - data[3] = list các cặp [code, name] (FULL data, dùng cái này)
    """
    if not query:
        return []
    try:
        r = requests.get(
            ICD_API,
            params={
                "terms": query,
                "maxList": max_results,
                "sf": "name",  # search by NAME not code
            },
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        logger.warning("ICD search fail (%r): %s", query, exc)
        return []

    if not isinstance(data, list) or len(data) < 4:
        return []
    # data[3] chứa [[code, name], [code, name], ...] - format gốc từ API
    pairs_raw = data[3] if data[3] else []
    out: list[tuple[str, str]] = []
    for pair in pairs_raw:
        if isinstance(pair, list) and len(pair) >= 2:
            out.append((str(pair[0]).strip(), str(pair[1]).strip()))
    return out


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
    """High-level wrapper: exact + hybrid (vector+BM25) + fuzzy + NIH remote.

    Pipeline 6 lớp:
      L1: Exact match dict tiếng Anh (prebuilt, ICDIndex)
      L2: VN -> EN translation (Translator)
      L3: Hybrid search — vector (BGE-M3) + BM25 keyword (ICD10HybridSearch)
      L4: Fuzzy match bằng rapidfuzz trên EN query
      L5: Fuzzy match bằng rapidfuzz trên VN text
      L6: NIH clinicaltables API (remote_cache dedupe)

    Mặc định `use_hybrid=True`; truyền `use_hybrid=False` để fallback về vector-only.
    """

    def __init__(
        self,
        index_path: Optional[Path] = None,
        translator: Optional[Translator] = None,
        use_remote: bool = True,
        remote_cache_path: Optional[Path] = None,
        local_search: Optional["ICD10VectorSearch | ICD10HybridSearch"] = None,
        use_hybrid: bool = True,
        hybrid_alpha: float = 0.6,
        hybrid_beta: float = 0.4,
    ) -> None:
        self.idx = self._load_index(index_path)
        self.translator = translator or Translator()
        self.use_remote = use_remote
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
        self._remote_cache_path = remote_cache_path or (
            DATA_DIR / "icd_remote_cache.json"
        )
        self._remote_cache: dict[str, list[list[str]]] = {}
        if self._remote_cache_path.exists():
            try:
                self._remote_cache = json.loads(
                    self._remote_cache_path.read_text(encoding="utf-8")
                )
            except Exception:
                self._remote_cache = {}

    # ------------------------------------------------------------------ #

    def _load_index(self, path: Optional[Path]) -> ICDIndex:
        path = path or (DATA_DIR / "icd_index.json")
        if not path.exists():
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

    def save_remote_cache(self) -> None:
        self._remote_cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._remote_cache_path.write_text(
            json.dumps(self._remote_cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------ #

    def lookup(self, vn_text: str, context_query: str | None = None) -> list[str]:
        """Tra ICD-10 cho 1 cụm chẩn đoán tiếng Việt.

        Args:
            vn_text: text gốc (VN) hoặc EN đã qua rescan.
            context_query: KHÔNG DÙNG — để giữ signature tương thích.
                Trước đây context_query được truyền vào BGE-M3 vector search
                nhưng làm CONTAMINATE embedding → trả codes sai. Fix: bỏ.

        Returns: list codes (string), unique + sorted, tối đa top_k codes.

        Bug history:
        1. Trước kia `_looks_vn()` bỏ sót từ không dấu ("suy tim", "ho"). Fix: luôn gọi translate().
        2. Trước kia input "bệnh nhân bị tăng huyết áp" hoặc "chẩn đoán: viêm phổi"
           không strip prefix → fuzzy fail với seed EN. Fix: strip clinical prefixes.
        3. Trước kia `threshold=65` luôn > score [0,1] → filter chết. Fix: 0.5.
        4. Trước kia `context_query` (chứa nearby drugs/symptoms) contaminate
           BGE-M3 → trả codes về drugs/symptoms thay vì diagnosis. Fix: bỏ.
        5. Trước kia top_k=20 → quá nhiều noise. Fix: top_k=5.
        """
        if not vn_text:
            return []

        text = self._strip_clinical_prefix(vn_text)

        # L1: Exact (nếu input đã là EN — ví dụ LLM đã chuẩn hóa)
        key = text.lower()
        if key in self.idx.exact:
            return sorted(set(self.idx.exact[key]))

        # L2: Translate VN -> EN (luôn gọi; cache hit trả rất nhanh, miss thì LLM
        # hoặc fallback về text gốc). Cần thiết vì NIH chỉ có tiếng Anh.
        en_query = self.translator.translate(text)
        en_key = en_query.lower().strip()
        if en_key in self.idx.exact:
            return sorted(set(self.idx.exact[en_key]))

        # L3: Hybrid search (vector BGE-M3 + BM25 keyword) trên toàn bộ icd10.jsonl.
        # Mặc định ICD10HybridSearch được wrap; vector-only nếu use_hybrid=False.
        # threshold=0.5 giữ yêu cầu cao để tránh noise; hybrid kết hợp α=0.6·cosine
        # + β=0.4·bm25_normalized (BM25 keyword sẽ "cứu" code đúng khi vector "ảo"
        # giữa các code cùng concept khác grade/stage — vd heart failure II vs III).
        if self.local_search is not None:
            local_results = self.local_search.search(
                en_query, top_k=5, threshold=0.5
            )
            if local_results:
                # Fix 14: Post-filter obvious mismatches (F10.x without alcohol, etc.)
                local_results = _filter_irrelevant_codes(local_results, text, self.idx)
                if local_results:
                    return local_results

        # L4: Local fuzzy match trên names với EN query (low threshold để bắt
        # substring như "Insomnia" trong "Insomnia, unspecified")
        fuzzy_en = self._fuzzy_local(en_query, threshold=70)
        if fuzzy_en:
            fuzzy_en = _filter_irrelevant_codes(fuzzy_en, text, self.idx)
            if fuzzy_en:
                return fuzzy_en

        # L5: Fuzzy cả với VN text (cho term không dịch được / giống EN)
        fuzzy_vn = self._fuzzy_local(text, threshold=85)
        if fuzzy_vn:
            fuzzy_vn = _filter_irrelevant_codes(fuzzy_vn, text, self.idx)
            if fuzzy_vn:
                return fuzzy_vn

        # L6: Remote NIH (cache + dedupe) — fallback cuối cùng
        if self.use_remote:
            cache_key = en_key
            if cache_key in self._remote_cache:
                results = self._remote_cache[cache_key]
                return sorted(set(results))

            results = _http_search(en_query, max_results=5)
            if not results:
                return []
            codes_from_remote = [r[0] for r in results]
            # Fix 14: Post-filter
            codes_from_remote = _filter_irrelevant_codes(codes_from_remote, text, self.idx)
            if not codes_from_remote:
                return []
            self._remote_cache[cache_key] = codes_from_remote
            for code, name in results:
                if code in codes_from_remote:
                    self.idx.add(code, name)
            return codes_from_remote

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
        self._jsonl_path = jsonl_path or (DATA_DIR / "icd10.jsonl")
        self._embeddings_path = embeddings_path or (DATA_DIR / "icd10_embeddings.npy")

        self.codes: list[str] = []
        self.descs_raw: list[str] = []

        self._embeddings: Optional[np.ndarray] = None
        self._model = None
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Lazy load: đọc file jsonl, load ma trận embeddings và model."""
        if self._loaded:
            return

        if not self._jsonl_path.exists():
            logger.warning("ICD10VectorSearch: không thấy file %s", self._jsonl_path)
            self._loaded = True
            return

        # 1. Đọc file jsonl để lấy codes/descriptions
        t0 = time.time()
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
                desc = str(row.get("desc_en", "")).strip()
                if code and desc:
                    self.codes.append(code)
                    self.descs_raw.append(desc)

        logger.info(
            "ICD10VectorSearch: Đã nạp %d mã ICD từ jsonl (%.2fs)",
            len(self.codes),
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
    """Tokenize cho BM25: giữ alnum runs, bỏ stop words, giữ nguyên case-fold.

    Lưu ý: cố tình KHÔNG stemming (vì ICD codes dùng chính xác các từ "type 2",
    "stage III" — stemming sẽ làm hỏng match).
    """
    if not text:
        return []
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if len(t) > 1 and t not in _BM25_STOP_WORDS]


class ICD10BM25Index:
    """BM25 keyword index cho ICD-10 — bổ trợ cho vector search.

    Multi-field weighting (BM25F-style thủ công): mỗi document được build bằng
    cách concat các field với trọng số khác nhau:
        - desc_en × 2      (field chính — chứa concept name)
        - desc_vi × 2      (nếu có — VN alias quan trọng cho query VN)
        - code_lower × 1   (light — bắt partial code match như "I50" trong query)

    Vì rank-bm25 chỉ hỗ trợ đơn field, ta mô phỏng bằng cách nhân đôi token thay
    vì tính BM25F thực sự (vẫn giữ được tinh thần field-weighted ranking).
    """

    def __init__(
        self,
        jsonl_path: Optional[Path] = None,
        tokens_cache_path: Optional[Path] = None,
    ) -> None:
        self._jsonl_path = jsonl_path or (DATA_DIR / "icd10.jsonl")
        self._tokens_cache_path = tokens_cache_path or (
            DATA_DIR / "icd10_bm25_tokens.jsonl.gz"
        )

        self.codes: list[str] = []
        self._bm25: Optional[Any] = None
        self._id_to_idx: dict[str, int] = {}
        self._loaded = False
        self._max_doc_score: float = 1.0  # cho normalize

    def _build_doc_text(self, desc_en: str, desc_vi: str, code: str) -> str:
        """Ghép các field với trọng số nhân đôi để mô phỏng field-weighted BM25."""
        parts: list[str] = []
        if desc_en:
            parts.extend([desc_en, desc_en])
        if desc_vi:
            parts.extend([desc_vi, desc_vi])
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

        # 2. Fallback: build from jsonl
        if not tokenized:
            if not self._jsonl_path.exists():
                logger.warning("BM25: thiếu file %s", self._jsonl_path)
                self._loaded = True
                return
            t0 = time.time()
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
                    if not (code and desc_en):
                        continue
                    doc_text = self._build_doc_text(desc_en, desc_vi, code)
                    self.codes.append(code)
                    tokenized.append(_bm25_tokenize(doc_text))
            logger.info(
                "BM25: tokenized %d docs từ jsonl (%.2fs)",
                len(self.codes),
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
    """Kết hợp vector search (BGE-M3) + BM25 keyword.

    Công thức: combined = α · cosine + β · bm25_normalized

    Mặc định α = 0.6, β = 0.4 (ưu tiên semantic một chút nhưng vẫn dùng
    keyword làm tie-breaker, đặc biệt khi vector cosine "ảo" giữa các code
    cùng concept khác liều/grade — VD: "heart failure stage II" vs "stage III").

    Args:
        vector_search: instance ICD10VectorSearch (dùng .search() và .score_codes()).
        bm25_index:    instance ICD10BM25Index.
        alpha:         trọng số vector (∈ [0, 1]).
        beta:          trọng số BM25 (∈ [0, 1]).
        top_k:         số codes trả về.
        threshold:     ngưỡng combined score tối thiểu để giữ code.
        fanout:        số candidates tối đa lấy từ mỗi method (vector & BM25)
                       để build union (giúp BM25 "cứu" code mà vector bỏ sót).
    """

    def __init__(
        self,
        vector_search: Optional[ICD10VectorSearch] = None,
        bm25_index: Optional[ICD10BM25Index] = None,
        alpha: float = 0.6,
        beta: float = 0.4,
        top_k: int = 5,
        threshold: float = 0.35,
        fanout: int = 15,
    ) -> None:
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
        """Hybrid search. Trả về list codes đã rank desc theo combined score."""
        if not query:
            return []
        k = top_k if top_k is not None else self.top_k
        thr = threshold if threshold is not None else self.threshold
        fanout = max(self.fanout, k * 2)

        # 1. Lấy candidates rộng từ cả 2 phương pháp
        vec_codes = self.vector_search.search(query, top_k=fanout, threshold=0.0) or []
        bm25_codes, bm25_raw = self.bm25_index.search(query, top_k=fanout)

        # 2. Union — giữ thứ tự xuất hiện (vector trước, BM25 sau)
        candidates = list(dict.fromkeys(vec_codes + bm25_codes))
        if not candidates:
            return []

        # 3. Tính lại cosine cho tất cả candidates (re-score)
        vec_scores = self.vector_search.score_codes(query, candidates)
        # 4. Tính BM25 normalized cho tất cả candidates
        bm25_scores = self.bm25_index.score_codes(query, candidates)

        # 5. Combine
        scored: list[tuple[str, float]] = []
        for code in candidates:
            v = vec_scores.get(code, 0.0)
            b = bm25_scores.get(code, 0.0)
            combined = self.alpha * v + self.beta * b
            if combined >= thr:
                scored.append((code, combined))

        # 6. Sort desc theo combined score, lấy top-k
        scored.sort(key=lambda x: -x[1])
        if scored:
            logger.debug(
                "Hybrid '%s' → top1=%s (combined=%.3f, vec=%.3f, bm25=%.3f)",
                query,
                scored[0][0],
                scored[0][1],
                vec_scores.get(scored[0][0], 0.0),
                bm25_scores.get(scored[0][0], 0.0),
            )
        return [c for c, _ in scored[:k]]


# ---------------------------------------------------------------------- #
# Build helpers
# ---------------------------------------------------------------------- #


def build_from_seed(path: Path, out_index: Optional[Path] = None) -> ICDIndex:
    """Đọc JSONL [{code, name_en, vn_aliases (optional)}] rồi build index."""
    idx = ICDIndex()
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
            name = str(row.get("name_en", row.get("name", ""))).strip()
            if code and name:
                idx.add(code, name)
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
        ret.save_remote_cache()
        ret.save_index()
