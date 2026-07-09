"""RxNorm RAG — tra cứu mã RxNorm cho thực thể THUỐC.

Kiến trúc giống ICD RAG (semantic + hybrid offline):

Pipeline 5 lớp (offline, 100% local):
  L1: Exact tuple match (ingredient, strength) → top-1 rxcui
  L2: Hybrid search (BGE-M3 cosine ≥ 0.7 + BM25 keyword) — semantic
  L3: Fuzzy match (rapidfuzz, partial_ratio) trên name
  L4: Compound drug — split strength " / ", match 1 thành phần
  L5: Ingredient-only exact match

Dữ liệu: `data/rxnorm.jsonl` (RxNorm 2026 release, ~232k entries,
schema {rxcui, name, ingredient, strength, doseform, ...}).

Đặc điểt:
- Multilingual vector search: BGE-M3 có thể map VN drug text → EN RxNorm entries.
- Return 1 rxcui duy nhất (theo spec Jaccard).
- Strength normalization: "25mg" == "25 MG" == "25.0 MG".
"""

from __future__ import annotations

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

_LOCAL_DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Path Kaggle cố định (chỉ đọc - chỉ dùng nếu user đã upload sẵn).
# Bao gồm: embeddings (.npy) + index (.json) + translation cache (.json).
# KHÔNG có: JSONL data source (rxnorm.jsonl) - phải dùng local.
_KAGGLE_DATA_DIR = Path("/kaggle/input/datasets/quangtrg/data_match/data")


def _detect_data_dir() -> Path:
    """Auto-detect data directory: Kaggle cached nếu có, fallback local.

    Trên Kaggle, embeddings + index files đã được upload sẵn để tránh
    build lại (rất chậm). Đường dẫn: /kaggle/input/datasets/quangtrg/data_match/data/

    Returns:
        Path tới data directory ưu tiên cho EMBEDDINGS + INDEX.
        JSONL data source LUÔN dùng local (không có trên Kaggle).
    """
    if (
        _KAGGLE_DATA_DIR.exists()
        and _KAGGLE_DATA_DIR.is_dir()
        and any(_KAGGLE_DATA_DIR.iterdir())
    ):
        return _KAGGLE_DATA_DIR
    return _LOCAL_DATA_DIR


# DATA_DIR: cho EMBEDDINGS + INDEX (auto-detect Kaggle)
DATA_DIR = _detect_data_dir()

# JSONL_DATA_DIR: cho DATA SOURCE JSONL (luôn local vì Kaggle không có)
JSONL_DATA_DIR = _LOCAL_DATA_DIR


# ---------------------------------------------------------------------- #
# Normalization helpers
# ---------------------------------------------------------------------- #

_STRENGTH_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|iu|unit|%|meq)(?:/(mg|mcg|g|ml|iu|unit|%|meq|mEq))?",
    re.IGNORECASE,
)


def _normalize_strength(s: str) -> str:
    """Chuẩn hoá strength string → dạng "25MG", "25MG/ML"."""
    if not s:
        return ""
    s = s.strip().lower()
    s = re.sub(r"(\d+(?:\.\d+)?)\s+(mg|mcg|g|ml|iu|unit|%|meq)", r"\1\2", s, flags=re.IGNORECASE)
    return s.upper()


def _normalize_ingredient(s: str) -> str:
    """Lowercase + strip + collapse whitespace."""
    return re.sub(r"\s+", " ", s.strip().lower()) if s else ""


def _strip_route_freq(text: str) -> str:
    """Loại bỏ route/freq/doseform tokens khỏi VN/EN drug text.

    VD: 'metoprolol 25 mg (uống hôm nay) po bid' → 'metoprolol 25 mg'.
    """
    def _repl(m: "re.Match[str]") -> str:
        content = m.group(1).strip()
        if _STRENGTH_RE.search(content):
            return m.group(0)  # giữ parenthetical có strength
        return " "

    text = re.sub(r"\(([^)]*)\)", _repl, text)

    skip_tokens = {
        "po", "iv", "im", "sc", "sl", "pr", "topical", "inhale", "oral",
        "tablet", "tab", "capsule", "cap", "solution", "suspension",
        "injection", "inj", "cream", "ointment", "gel", "patch",
        "spray", "powder", "drop", "drops", "syrup",
        "extended", "release", "xl", "xr", "er", "sr", "la", "cr",
        "daily", "bid", "tid", "qid", "qhs", "qam", "qpm",
        "q6h", "q8h", "q12h", "prn", "qd", "qod", "hs", "ac", "pc",
        "uống", "tiêm", "tiêng", "viên", "ống", "gói", "lần", "ngày",
        "giờ", "tuần", "tháng", "sáng", "trưa", "chiều", "tối",
    }

    def _compact(m: "re.Match[str]") -> str:
        suffix = f"/{m.group(3).lower()}" if m.group(3) else ""
        return f"{m.group(1)}{m.group(2).lower()}{suffix}"

    text = _STRENGTH_RE.sub(_compact, text)
    tokens = [t for t in re.split(r"[^a-z0-9/]+", text.lower()) if t]
    return " ".join(t for t in tokens if t not in skip_tokens)


# ---------------------------------------------------------------------- #
# Drug aliases — brand → generic (Vietnamese common drug names)
# Mục đích: khi LLM extract "Panadol 500mg" → lookup RxNorm bằng "paracetamol".
# Vì RxNorm index chủ yếu theo generic name, brand cần map về generic.
# Cập nhật 2026-07.
# ---------------------------------------------------------------------- #

_DRUG_ALIASES: dict[str, str] = {
    # Analgesic / antipyretic
    "panadol": "acetaminophen", "panadol extra": "acetaminophen",  # EU/UK → US generic in index
    "efferalgan": "acetaminophen",  # FR
    "doliprane": "acetaminophen",  # FR
    "paracetamol": "acetaminophen",  # EU/UK → US generic in index
    "tylenol": "acetaminophen",  # US
    "acetaminophen": "acetaminophen",  # already generic (no-op)
    "aspirin": "aspirin",  # generic rồi
    "aspegic": "aspirin",  # aspirin lysine
    "bayer": "aspirin",
    # Anti-inflammatory / NSAID
    "voltaren": "diclofenac", "voltaren gel": "diclofenac",
    "ibuprofen": "ibuprofen", "advil": "ibuprofen", "motrin": "ibuprofen",
    "celebrex": "celecoxib",
    "meloxicam": "meloxicam", "mobic": "meloxicam",
    "naproxen": "naproxen", "aleve": "naproxen",
    # Cardiovascular - antihypertensive
    "norvasc": "amlodipine", "istin": "amlodipine",
    "lopressor": "metoprolol", "betaloc": "metoprolol",
    "tenormin": "atenolol",
    "coreg": "carvedilol", "dilatrend": "carvedilol",
    "capoten": "captopril", "lopril": "captopril",
    "zestril": "lisinopril",
    "cozaar": "losartan", "cozaar plus": "losartan",
    "diovan": "valsartan",
    # Statin
    "lipitor": "atorvastatin", "atorvastatin": "atorvastatin",
    "zocor": "simvastatin", "simvastatin": "simvastatin",
    "crestor": "rosuvastatin",
    # Antiplatelet / anticoag
    "plavix": "clopidogrel",
    "eliquis": "apixaban",
    "xarelto": "rivaroxaban",
    "coumadin": "warfarin", "marevan": "warfarin",
    # Diabetes
    "glucophage": "metformin", "glucophage xr": "metformin",
    "glucovance": "metformin",
    "januvia": "sitagliptin",
    "amaryl": "glimepiride",
    "diamicron": "gliclazide",
    # GI
    "nexium": "esomeprazole",
    "losec": "omeprazole", "prilosec": "omeprazole",
    "pantoloc": "pantoprazole",
    # Antibiotic
    "augmentin": "amoxicillin",
    "zithromax": "azithromycin",
    "cipro": "ciprofloxacin",
    "keflex": "cephalexin",
    "vibramycin": "doxycycline",
    # CNS
    "stilnox": "zolpidem",
    "xanax": "alprazolam",
    "valium": "diazepam",
    "lexapro": "escitalopram",
    "zoloft": "sertraline",
    "prozac": "fluoxetine",
    "lyrica": "pregabalin",
    "neurontin": "gabapentin",
    # Misc
    "ventolin": "albuterol", "salbutamol": "albuterol",
    "symbicort": "budesonide",
    "singulair": "montelukast",
    "trileptal": "oxcarbazepine",
    "depakote": "valproic acid",
    "lamictal": "lamotrigine",
}


def _alias_to_generic(drug_text: str) -> str:
    """Translate brand name in drug text → generic name.

    VD: "Panadol 500mg" → "paracetamol 500mg"
        "Voltaren gel 50g" → "diclofenac gel 50g" (giữ strength + form)
    Logic: extract first word (likely drug name) → lookup in _DRUG_ALIASES → replace.
    """
    if not drug_text or not _DRUG_ALIASES:
        return drug_text
    text_lower = drug_text.lower().strip()
    # Thử match từ đầu (longest first)
    for brand in sorted(_DRUG_ALIASES.keys(), key=len, reverse=True):
        if text_lower.startswith(brand + " ") or text_lower == brand:
            generic = _DRUG_ALIASES[brand]
            # Preserve rest of text (strength, route, etc.)
            rest = drug_text[len(brand):].lstrip()
            return f"{generic} {rest}".strip() if rest else generic
    return drug_text


def _parse_drug(text: str) -> tuple[str, str]:
    """Parse chuỗi thuốc → (ingredient_norm, strength_norm).

    Returns:
        (ingredient_norm, strength_norm)
    """
    text = _strip_route_freq(text)
    if not text:
        return ("", "")

    matches = list(_STRENGTH_RE.finditer(text))
    if not matches:
        ingredient = re.sub(r"\s+", " ", text.lower()).strip()
        return (ingredient, "")

    strengths = [m.group(0) for m in matches]
    remaining = text
    for s in strengths:
        remaining = re.sub(re.escape(s), " ", remaining, flags=re.IGNORECASE)

    ingredient_tokens = re.findall(r"[a-z][a-z0-9-]+", remaining.lower())
    ingredient = " ".join(ingredient_tokens).strip()
    if not ingredient:
        return ("", "")

    norm_strengths = [_normalize_strength(s) for s in strengths]
    norm_strengths = [s for s in norm_strengths if s]

    if len(norm_strengths) == 1:
        return (ingredient, norm_strengths[0])
    if len(norm_strengths) > 1:
        return (ingredient, " / ".join(norm_strengths))
    return (ingredient, "")


# ---------------------------------------------------------------------- #
# Index: exact match (ingredient, strength) — fast path
# ---------------------------------------------------------------------- #


@dataclass
class RxNormIndex:
    """Index exact match (ingredient, strength) → list[rxcui].

    Attributes:
        by_ingredient_strength: dict[(ing_norm, str_norm), list[rxcui]]
        by_ingredient: dict[ing_norm, list[rxcui]]
        names: list tên gốc — cho fuzzy
        rxcuis: parallel với names
        name_to_idx: name -> idx
    """

    by_ingredient_strength: dict[tuple[str, str], list[str]] = field(default_factory=dict)
    by_ingredient: dict[str, list[str]] = field(default_factory=dict)
    names: list[str] = field(default_factory=list)
    rxcuis: list[str] = field(default_factory=list)
    name_to_idx: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "RxNormIndex":
        bis_raw = data.get("by_ingredient_strength", {})
        bis: dict[tuple[str, str], list[str]] = {}
        if isinstance(bis_raw, dict):
            for k, v in bis_raw.items():
                if isinstance(k, (list, tuple)):
                    bis[(k[0], k[1])] = v
                elif isinstance(k, str) and "|" in k:
                    parts = k.split("|", 1)
                    bis[(parts[0], parts[1])] = v
        idx = cls(
            by_ingredient_strength=bis,
            by_ingredient=data.get("by_ingredient", {}),
            names=data.get("names", []),
            rxcuis=data.get("rxcuis", []),
        )
        idx.name_to_idx = {n: i for i, n in enumerate(idx.names)}
        return idx

    def to_dict(self) -> dict:
        return {
            "by_ingredient_strength": {f"{k[0]}|{k[1]}": v
                                        for k, v in self.by_ingredient_strength.items()},
            "by_ingredient": self.by_ingredient,
            "names": self.names,
            "rxcuis": self.rxcuis,
        }

    def add(self, rxcui: str, ingredient: str, strength: str, name: str = "") -> None:
        rxcui = str(rxcui).strip()
        ing_norm = _normalize_ingredient(ingredient)
        str_norm = _normalize_strength(strength)
        if not rxcui:
            return

        if ing_norm and str_norm:
            self.by_ingredient_strength.setdefault((ing_norm, str_norm), []).append(rxcui)
        if ing_norm:
            self.by_ingredient.setdefault(ing_norm, []).append(rxcui)

        if name and name not in self.name_to_idx:
            self.name_to_idx[name] = len(self.names)
            self.names.append(name)
            self.rxcuis.append(rxcui)

    # ------------------------------------------------------------------ #

    def lookup(self, drug_text: str) -> list[str]:
        """Pipeline L1 + L4 + L5 (fast exact path)."""
        ing, strength = _parse_drug(drug_text)
        if not ing:
            return []

        # L1: Exact (ingredient, strength) tuple
        if strength:
            cands = self.by_ingredient_strength.get((ing, strength), [])
            if cands:
                return [cands[0]]

            # L4: Compound split
            if " / " in strength:
                for sub in strength.split(" / "):
                    sub = sub.strip()
                    if not sub:
                        continue
                    cands = self.by_ingredient_strength.get((ing, sub), [])
                    if cands:
                        return [cands[0]]

        # L5: Ingredient-only
        cands = self.by_ingredient.get(ing, [])
        if cands:
            return [cands[0]]
        return []


# ---------------------------------------------------------------------- #
# Vector Search — BGE-M3 cosine trên name field
# ---------------------------------------------------------------------- #


class RxNormVectorSearch:
    """Tra cứu RxNorm bằng vector nhúng cosine.

    Embed field `name` (English: "metoprolol 25 MG Oral Tablet") bằng BGE-M3.
    BGE-M3 multilingual → có thể query bằng VN drug text (vd "metoprolol 25mg po bid")
    và vẫn match sang EN entries.
    """

    def __init__(
        self,
        jsonl_path: Optional[Path] = None,
        embeddings_path: Optional[Path] = None,
    ) -> None:
        # NOTE: JSONL files LUÔN dùng local (Kaggle không có JSONL data source).
        self._jsonl_path = jsonl_path or (JSONL_DATA_DIR / "rxnorm.jsonl")
        self._embeddings_path = embeddings_path or (DATA_DIR / "rxnorm_embeddings.npy")

        self.codes: list[str] = []
        self.names: list[str] = []
        self._embeddings: Optional[np.ndarray] = None
        self._model = None
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if not self._jsonl_path.exists():
            logger.warning("RxNormVectorSearch: không thấy %s", self._jsonl_path)
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
                rxcui = str(row.get("rxcui", "")).strip()
                name = str(row.get("name", "")).strip()
                if rxcui and name:
                    self.codes.append(rxcui)
                    self.names.append(name)

        logger.info(
            "RxNormVectorSearch: loaded %d entries (%.2fs)",
            len(self.codes), time.time() - t0,
        )

        # Load embeddings if exists
        if self._embeddings_path.exists():
            try:
                self._embeddings = np.load(self._embeddings_path)
                logger.info(
                    "RxNormVectorSearch: loaded embeddings %s (shape: %r)",
                    self._embeddings_path.name, self._embeddings.shape,
                )
            except Exception as exc:
                logger.warning("Embeddings load fail (%s)", exc)

        self._loaded = True

    def search(self, query: str, *, top_k: int = 10, threshold: float = 0.4) -> list[str]:
        """Cosine search trên top_k. Filter ≥ threshold."""
        self._ensure_loaded()
        if not query or not self.codes:
            return []

        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                self._model = SentenceTransformer("BAAI/bge-m3")
                logger.info("RxNormVectorSearch: loaded BGE-M3")
            except ImportError:
                logger.error("Sentence-transformers chưa cài!")
                return []

        if self._embeddings is None:
            logger.info("Auto-generating RxNorm embeddings (chưa có .npy)...")
            try:
                self._embeddings = self._model.encode(
                    self.names,
                    batch_size=128,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )
                np.save(self._embeddings_path, self._embeddings)
                logger.info("Saved RxNorm embeddings → %s", self._embeddings_path.name)
            except Exception as exc:
                logger.error("Lỗi sinh embeddings: %s", exc)
                return []

        query_vec = self._model.encode(query, normalize_embeddings=True, convert_to_numpy=True)
        # Cast về cùng dtype với embeddings (float16 tiết kiệm RAM cho 232k entries)
        if self._embeddings is not None and self._embeddings.dtype != query_vec.dtype:
            query_vec = query_vec.astype(self._embeddings.dtype)
        scores = np.dot(self._embeddings, query_vec)
        top_idx = np.argsort(-scores)[:top_k]

        out: list[str] = []
        for idx in top_idx:
            score = float(scores[idx])
            if score >= threshold:
                rxcui = self.codes[idx]
                if rxcui not in out:
                    out.append(rxcui)
        return out

    def score_codes(self, query: str, codes: list[str]) -> dict[str, float]:
        """Tính cosine similarity cho tập codes cụ thể (cho hybrid re-score)."""
        self._ensure_loaded()
        if not query or not codes or self._embeddings is None:
            return {}
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                self._model = SentenceTransformer("BAAI/bge-m3")
            except ImportError:
                return {}
        q_vec = self._model.encode(query, normalize_embeddings=True, convert_to_numpy=True)
        all_scores = self._embeddings @ q_vec
        idx_by_code = {c: i for i, c in enumerate(self.codes)}
        return {c: float(all_scores[idx_by_code[c]]) for c in codes if c in idx_by_code}


# ---------------------------------------------------------------------- #
# BM25 Index — keyword fallback
# ---------------------------------------------------------------------- #

_BM25_STOP_WORDS = frozenset({
    "the", "of", "and", "or", "with", "without", "in", "to", "a", "an",
    "due", "by", "for", "from", "as", "at", "on", "is", "are", "be",
})


def _bm25_tokenize(text: str) -> list[str]:
    """Tokenize cho BM25: alnum runs + VN diacritics, bỏ stop words."""
    if not text:
        return []
    tokens = re.findall(r"[\w]+", text.lower(), flags=re.UNICODE)
    return [t for t in tokens if len(t) > 1 and t not in _BM25_STOP_WORDS]


class RxNormBM25Index:
    """BM25 keyword index cho RxNorm — mở rộng candidates cho vector search.

    Multi-field weighting: name × 2 (field chính), ingredient × 2 (search target),
    strength × 1.
    """

    def __init__(
        self,
        jsonl_path: Optional[Path] = None,
        tokens_cache_path: Optional[Path] = None,
    ) -> None:
        # NOTE: JSONL files LUÔN dùng local (Kaggle không có JSONL data source).
        self._jsonl_path = jsonl_path or (JSONL_DATA_DIR / "rxnorm.jsonl")
        self._tokens_cache_path = tokens_cache_path or (DATA_DIR / "rxnorm_bm25_tokens.jsonl.gz")

        self.codes: list[str] = []
        self._bm25: Optional[Any] = None
        self._id_to_idx: dict[str, int] = {}
        self._loaded = False

    def _build_doc_text(self, name: str, ingredient: str, strength: str) -> str:
        parts: list[str] = []
        if name:
            parts.extend([name, name])
        if ingredient and ingredient != name:
            parts.extend([ingredient, ingredient])
        if strength:
            parts.append(strength.lower())
        return " ".join(parts)

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if not _HAS_BM25:
            logger.warning("rank-bm25 chưa cài → BM25 disabled")
            self._loaded = True
            return
        if not self._jsonl_path.exists():
            self._loaded = True
            return

        tokenized: list[list[str]] = []

        # 1. Try load cache
        if self._tokens_cache_path.exists():
            try:
                with open(self._tokens_cache_path, "rb") as f:
                    import gzip
                    with gzip.open(self._tokens_cache_path, "rt", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            row = json.loads(line)
                            self.codes.append(row["code"])
                            tokenized.append(row["tokens"])
                logger.info("BM25 loaded %d docs từ cache", len(self.codes))
            except Exception:
                self.codes.clear()
                tokenized.clear()

        # 2. Build from JSONL
        if not tokenized:
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
                    rxcui = str(row.get("rxcui", "")).strip()
                    name = str(row.get("name", "")).strip()
                    ing = str(row.get("ingredient", "")).strip()
                    strength = str(row.get("strength", "")).strip()
                    if not (rxcui and (name or ing)):
                        continue
                    doc_text = self._build_doc_text(name, ing, strength)
                    self.codes.append(rxcui)
                    tokenized.append(_bm25_tokenize(doc_text))
            logger.info("BM25 tokenized %d docs từ JSONL (%.2fs)", len(self.codes), time.time() - t0)

            # Save cache
            try:
                self._tokens_cache_path.parent.mkdir(parents=True, exist_ok=True)
                import gzip
                with gzip.open(self._tokens_cache_path, "wt", encoding="utf-8") as f:
                    for code, toks in zip(self.codes, tokenized):
                        f.write(json.dumps({"code": code, "tokens": toks}, ensure_ascii=False) + "\n")
            except Exception as exc:
                logger.warning("BM25 cache save fail: %s", exc)

        self._bm25 = BM25Okapi(tokenized)
        self._id_to_idx = {c: i for i, c in enumerate(self.codes)}
        self._loaded = True

    def search(self, query: str, *, top_k: int = 20) -> tuple[list[str], list[float]]:
        """Search BM25 → (codes, scores) sorted desc."""
        self._ensure_loaded()
        if self._bm25 is None or not query:
            return [], []
        q_tokens = _bm25_tokenize(query)
        if not q_tokens:
            return [], []
        scores = self._bm25.get_scores(q_tokens)
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
        """BM25 score cho 1 tập codes, normalize về [0,1]."""
        self._ensure_loaded()
        if self._bm25 is None or not query or not codes:
            return {}
        q_tokens = _bm25_tokenize(query)
        if not q_tokens:
            return {}
        scores = self._bm25.get_scores(q_tokens)
        max_s = float(scores.max()) or 1.0
        return {c: float(scores[self._id_to_idx[c]]) / max_s for c in codes if c in self._id_to_idx}


# ---------------------------------------------------------------------- #
# Hybrid Search — vector + BM25
# ---------------------------------------------------------------------- #


class RxNormHybridSearch:
    """Hybrid RxNorm search = vector (BGE-M3) + BM25 keyword.

    Threshold-based cosine (semantic), không cap top-K. Trả 1 rxcui (top match).

    Args:
        vector_search: instance RxNormVectorSearch.
        bm25_index: instance RxNormBM25Index.
        threshold: cosine minimum (mặc định 0.7).
        fanout: số candidates mỗi method (union).
    """

    def __init__(
        self,
        vector_search: Optional[RxNormVectorSearch] = None,
        bm25_index: Optional[RxNormBM25Index] = None,
        threshold: float = 0.5,  # giảm từ 0.7 → 0.5 để match VN drug text
        fanout: int = 50,
    ) -> None:
        self.vector_search = vector_search or RxNormVectorSearch()
        self.bm25_index = bm25_index or RxNormBM25Index()
        self.threshold = threshold
        self.fanout = fanout

    def search(self, query: str, *, threshold: Optional[float] = None, top_k: Optional[int] = 1) -> list[str]:
        """Semantic extraction: union candidates từ vector + BM25, re-score cosine."""
        if not query:
            return []
        thr = threshold if threshold is not None else self.threshold
        fanout = max(self.fanout, 30)

        # 1. Get candidates from vector + BM25
        vec_codes = self.vector_search.search(query, top_k=fanout, threshold=0.0) or []
        bm25_codes, _ = self.bm25_index.search(query, top_k=fanout)

        # 2. Union
        candidates = list(dict.fromkeys(vec_codes + bm25_codes))
        if not candidates:
            return []

        # 3. Re-score cosine
        vec_scores = self.vector_search.score_codes(query, candidates)

        # 4. Filter by cosine
        matched: list[tuple[str, float]] = []
        for code in candidates:
            cos = vec_scores.get(code, 0.0)
            if cos >= thr:
                matched.append((code, cos))

        if matched:
            logger.debug("Hybrid '%s' → %d codes (top1=%s cos=%.3f)",
                         query, len(matched), matched[0][0], matched[0][1])

        # 5. Sort desc + tie-break by BM25
        bm25_scores = self.bm25_index.score_codes(query, [c for c, _ in matched])
        matched.sort(key=lambda x: (-x[1], -bm25_scores.get(x[0], 0.0)))

        k = top_k if top_k is not None else 1
        return [c for c, _ in matched[:k]]


# ---------------------------------------------------------------------- #
# High-level Retriever — combines all 5 layers
# ---------------------------------------------------------------------- #


class RxNormRetriever:
    """High-level RxNorm retriever với 5 lớp offline.

    Pipeline:
      L1: Exact tuple (ingredient, strength) — fast path
      L2: Hybrid search (vector + BM25) — semantic fallback
      L3: Fuzzy match trên names — VN/EN variants
      L4: Compound drug split
      L5: Ingredient-only exact match
    """

    def __init__(
        self,
        index: Optional[RxNormIndex] = None,
        index_path: Optional[Path] = None,
        use_hybrid: bool = True,
        hybrid_threshold: float = 0.78,  # nâng từ 0.7 (2026-07 fix theo Kaggle eval):
                                        # - 0.7 quá thoáng → trả nhiều candidates noise
                                        # - 0.78 precision cao hơn, chỉ match rõ ràng
    ) -> None:
        if index is not None:
            self.index = index
        else:
            self.index = load_index(index_path)

        # Hybrid search (vector + BM25)
        self.use_hybrid = use_hybrid
        if use_hybrid:
            self.vector_search = RxNormVectorSearch()
            self.bm25_index = RxNormBM25Index()
            self.hybrid_search = RxNormHybridSearch(
                vector_search=self.vector_search,
                bm25_index=self.bm25_index,
                threshold=hybrid_threshold,
            )
        else:
            self.vector_search = None
            self.bm25_index = None
            self.hybrid_search = None

    # ------------------------------------------------------------------ #

    def lookup(self, drug_text: str) -> list[str]:
        """Tra RxNorm cho chuỗi thuốc VN/EN → list [rxcui] (0 hoặc 1 code, HIGH precision).

        2026-07 fix theo Kaggle eval:
        - Hybrid threshold nâng 0.7→0.78 (tránh candidates noise)
        - Top_k=1 đã đúng → đảm bảo "1 mã thuốc" semantic, KHÔNG list nhiều
        - Nếu confidence thấp → return [], KHÔNG đoán sai

        2026-07 fix: Thêm L4 fallback — nếu L1 ingredient-only exact match fail
        (vd "atenolol" có trong data nhưng L5 lookup miss) → thử fuzzy match trên
        name field với threshold thấp hơn (75) để bắt các trường hợp drug name
        ngắn gọn không match được qua L1.
        """
        if not drug_text:
            return []

        drug_text = drug_text.strip()

        # Pre-step: Translate brand → generic (mới 2026-07).
        # "Panadol 500mg" → "paracetamol 500mg" trước khi lookup RxNorm.
        drug_text = _alias_to_generic(drug_text)

        # L1: Exact (ing, strength) — highest confidence, cap 1
        result = self.index.lookup(drug_text)
        if result:
            return result[:1]

        # L2: Hybrid search — threshold 0.78 strict, top_k=1
        if self.use_hybrid and self.hybrid_search is not None:
            try:
                result = self.hybrid_search.search(
                    drug_text, top_k=1, threshold=0.78
                )
                if result:
                    return result[:1]
            except Exception as exc:
                logger.warning("Hybrid search fail (%s): %s", drug_text[:30], exc)

        # L3: Fuzzy match với threshold 80 (strict hơn 70 cũ)
        result = self._fuzzy_local(drug_text, threshold=80)
        if result:
            return result[:1]

        # L4: Fuzzy match lỏng hơn (75) — fallback cho drug name ngắn (vd "atenolol")
        # mà L1 miss (do index chưa build đúng hoặc ingredient normalization fail)
        result = self._fuzzy_local(drug_text, threshold=75)
        if result:
            logger.debug(
                "RxNorm L4 loose fuzzy fallback matched '%s' → %s",
                drug_text, result,
            )
            return result[:1]

        return []  # confidence thấp → empty

    def _fuzzy_local(self, query: str, threshold: int = 70) -> list[str]:
        """Fuzzy match trên name list (rapidfuzz)."""
        if not query or not self.index.names:
            return []
        try:
            from rapidfuzz import fuzz, process  # type: ignore
        except ImportError:
            return []

        # Strip route/freq trước khi fuzzy
        stripped = _strip_route_freq(query)
        if not stripped:
            return []
        q_tokens = [t for t in re.split(r"[^a-z0-9]+", stripped.lower()) if len(t) > 1]
        if not q_tokens:
            return []
        q = " ".join(q_tokens)

        matches = process.extract(q, self.index.names, scorer=fuzz.WRatio, limit=5)
        matches += process.extract(q, self.index.names, scorer=fuzz.partial_ratio, limit=5)
        for name, score, _ in matches:
            if score >= threshold and name in self.index.name_to_idx:
                return [self.index.rxcuis[self.index.name_to_idx[name]]]
        return []


# ---------------------------------------------------------------------- #
# Persistence
# ---------------------------------------------------------------------- #


def load_index(path: Optional[Path] = None) -> RxNormIndex:
    """Nạp RxNormIndex từ JSON. Trả RxNormIndex rỗng nếu không có file."""
    path = path or (DATA_DIR / "rxnorm_index.json")
    if not path.exists():
        return RxNormIndex()
    with path.open(encoding="utf-8") as f:
        return RxNormIndex.from_dict(json.load(f))


def save_index(idx: RxNormIndex, path: Optional[Path] = None) -> None:
    path = path or (DATA_DIR / "rxnorm_index.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(idx.to_dict(), f, ensure_ascii=False, indent=1)
    logger.info("Saved → %s (%d names, %d ing+strength, %d ingredients)",
                path.name, len(idx.names), len(idx.by_ingredient_strength),
                len(idx.by_ingredient))


# ---------------------------------------------------------------------- #
# Build from JSONL
# ---------------------------------------------------------------------- #


def build_from_rxnorm_dump(dump_path: Path, out_path: Optional[Path] = None) -> RxNormIndex:
    """Đọc JSONL [{rxcui, ingredient, strength, doseform, name, ...}] → build RxNormIndex."""
    idx = RxNormIndex()
    n = 0
    with dump_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rxcui = str(row.get("rxcui", "")).strip()
            ing = str(row.get("ingredient", "")).strip()
            strength = str(row.get("strength", "")).strip()
            name = str(row.get("name", "")).strip()
            if rxcui and ing:
                idx.add(rxcui, ing, strength, name)
                n += 1
    save_index(idx, out_path)
    logger.info("Built RxNormIndex from %d rows", n)
    return idx


# ---------------------------------------------------------------------- #
# CLI self-test
# ---------------------------------------------------------------------- #


if __name__ == "__main__":  # pragma: no cover
    import sys

    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) >= 2:
        idx = build_from_rxnorm_dump(Path(sys.argv[1]))
        print(f"Index: {len(idx.names)} names, "
              f"{len(idx.by_ingredient_strength)} ing+strength keys, "
              f"{len(idx.by_ingredient)} ingredients")
    else:
        idx = load_index()
        print(f"Loaded: {len(idx.names)} names, "
              f"{len(idx.by_ingredient_strength)} ing+strength keys, "
              f"{len(idx.by_ingredient)} ingredients")
