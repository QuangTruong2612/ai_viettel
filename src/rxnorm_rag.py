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

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _safe_json_load(path: Path, default=None):
    """R37 (2026-07-20): Safe JSON load — xem src/icd_rag.py để biết chi tiết.

    Trên Kaggle sau khi GIT_LFS_SKIP_SMUDGE=1, file JSON có thể bị empty.
    Hàm này trả default khi file missing/empty/invalid (no crash).
    """
    if default is None:
        default = {}
    try:
        if not path.exists():
            return default
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            logger.warning(
                "%s is EMPTY (likely LFS skip-smudge). Returning default.",
                path.name,
            )
            return default
        return json.loads(content)
    except json.JSONDecodeError as exc:
        logger.warning("%s invalid JSON: %s. Returning default.", path.name, exc)
        return default
    except Exception as exc:
        logger.warning("%s load fail: %s. Returning default.", path.name, exc)
        return default



# ---------------------------------------------------------------------- #
# Normalization helpers
# ---------------------------------------------------------------------- #

_STRENGTH_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|iu|unit|unt|%|meq)(?:/(mg|mcg|g|ml|iu|unit|unt|%|meq|mEq))?",
    re.IGNORECASE,
)
# R34: Range strength pattern (vd "325-650 mg" → use LOW value 325mg)
_RANGE_STRENGTH_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|iu|unit|unt|%|meq)",
    re.IGNORECASE,
)


def _normalize_strength(s: str) -> str:
    """Chuẩn hoá strength string → dạng "25MG", "25MG/ML".

    Logic:
        - Detect if norm is JUST bare ml (e.g. "5ML", "10ML") → return ""
        - Otherwise: collapse whitespace + uppercase (preserve /ml concentration)
    """
    if not s:
        return ""
    s = s.strip().lower()
    s = re.sub(r"(\d+(?:\.\d+)?)\s+(mg|mcg|g|ml|iu|unit|%|meq)", r"\1\2", s, flags=re.IGNORECASE)
    norm = s.upper()
    # R29: bare ML (volume) is not strength. /ML kept (concentration).
    if re.fullmatch(r"\d+(?:\.\d+)?ML", norm):
        return ""
    return norm


# R34 (2026-07-13): Parse strength thành số để so sánh closest match.
def _parse_strength_value(s: str) -> float | None:
    """Parse numerical từ strength. '25MG' → 25.0; '0.5MG' → 0.5; '5MG/ML' → 5.0.

    R42 (2026-07-14): Prefer numbers near strength units (MG, ML, ...) to avoid
    picking up "12" from "12 HR guaifenesin 1200 MG Extended Release Oral Tablet"
    → was wrongly returning 12.0 thay vì 1200.0, làm "12 HR" ER candidates rank
    cao hơn 800 MG OT trong secondary sort.
    """
    if not s:
        return None
    # First try: number directly followed by strength unit
    m = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:mg|mcg|g|ml|iu|unit|unt|%|meq)(?:/|\s|$)",
        s, re.IGNORECASE,
    )
    if m:
        return float(m.group(1))
    # Fallback: first number in string
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    if m:
        return float(m.group(1))
    return None


# R34 (2026-07-13): Salt preference — user pipeline thường pick salt forms
# (pravastatin sodium, metoprolol tartrate, etc.) over plain base.
_SALT_WORDS = (
    "sodium", "potassium", "calcium", "hydrochloride", "hcl",
    "sulfate", "tartrate", "maleate", "mesylate", "besylate",
)
_SALT_BONUS = 1.0  # mild preference


def _salt_preference_score(name_lower: str) -> float:
    """+1.0 nếu matched name có salt form."""
    if any(s in name_lower for s in _SALT_WORDS):
        return _SALT_BONUS
    return 0.0


def _normalize_ingredient(s: str) -> str:
    """Lowercase + strip + collapse whitespace + STRIP doseform tokens.

    R34 (2026-07-13): Một số rxcui có `ingredient="Drug Oral Tablet"` (vd
    373942 = "Spiramycin Oral Tablet"). Strip doseform tokens để index key
    khớp với plain ingredient (vd "spiramycin"). Fix quá trình lookup cho
    Spiramycin/Mifepristone variants.
    """
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s.strip().lower())
    # Doseform & descriptor words to strip from ingredient
    doseform_strip = {
        "oral", "tablet", "tablets", "capsule", "capsules", "solution",
        "suspension", "injection", "injectable", "cream", "ointment",
        "gel", "patch", "syrup", "powder", "spray", "drops", "granules",
        "suppository", "inhaler", "nebulizer",
        "extended", "release", "delayed", "disintegrating", "chewable",
        "long", "acting", "short", "hr", "formulation",
    }
    tokens = [t for t in s.split() if t not in doseform_strip]
    cleaned = " ".join(tokens).strip()
    # Use cleaned form nếu non-empty (preserves compound "metoprolol tartrate" etc.)
    return cleaned if cleaned else s


def _strip_route_freq(text: str) -> str:
    """Loại bỏ route/freq/doseform tokens khỏi VN/EN drug text.

    VD: 'metoprolol 25 mg (uống hôm nay) po bid' → 'metoprolol 25 mg'.

    R34: Cũng normalize range strength (vd '325-650 mg' → '325mg', dùng LOW value).
    R42 (2026-07-14): Handle dose-change parentheticals — extract LAST strength
    (current dose) thay vì strip cả. Vd:
      "(dose decreased from 5mg to 1mg)" → keep "1mg" (current)
      "(previously 5 mg, now 1 mg)" → keep "1 mg"
      "(tapered from 5mg bid to 1mg bid)" → keep "1mg"
    """
    # R42: Handle dose-change parentheticals in correct order.
    # Step 1: Extract "current" from "previously X, now Y" patterns FIRST
    # (before any strip that would remove the parenthetical entirely).
    # Allow Y to include strength unit (e.g., "now 1 mg" → keep "1 mg").
    _NOW_RE = re.compile(
        r"(?:previously|historically|was|had)\s+[^,]+,\s*"
        r"(?:now|currently|switched\s+to)\s+"
        r"(\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|iu|unit|unt|%|meq)?"
        r"(?:\s*(?:mg|mcg|g|ml|iu|unit|unt|%|meq))?)",
        re.IGNORECASE,
    )

    def _now_repl(m: "re.Match[str]") -> str:
        return f" {m.group(1).strip()} "

    text = _NOW_RE.sub(_now_repl, text)

    # Step 2: Strip parentheticals with ONLY history (no current dose).
    # Vd: "(previously on 5mg)", "(dose was 5mg)", "(history: 5mg)"
    _DOSE_HISTORY_ONLY_RE = re.compile(
        r"\([^)]*?(?:previously|historically|dose\s+was|was\s+\d|hx)[^)]*\)",
        re.IGNORECASE,
    )
    text = _DOSE_HISTORY_ONLY_RE.sub(" ", text)

    # Step 3: Extract "current" strength from "from <prev> to <curr>" patterns.
    # Vd: "(dose decreased from 5mg to 1mg)" → keep "1mg" (current).
    _FROM_TO_RE = re.compile(
        r"\bfrom\s+(\S+(?:\s+\S+)*?)\s+to\s+(\S+(?:\s+\S+)*?)(?=[\s,;)])",
        re.IGNORECASE,
    )

    def _from_to_repl(m: "re.Match[str]") -> str:
        return f" {m.group(2).strip()} "

    text = _FROM_TO_RE.sub(_from_to_repl, text)

    # Step 4: Strip remaining dose-context parentheticals WITHOUT strength.
    # (e.g., "(tapered from 5mg)" became "(tapered )" after step 3 → strip here).
    # IMPORTANT: only strip if no strength remains (otherwise we lose current dose).
    _DOSE_CONTEXT_RE = re.compile(
        r"\(([^)]*?)(?:tapered|decreased|increased|adjusted|reduced)([^)]*)\)",
        re.IGNORECASE,
    )

    def _ctx_repl(m: "re.Match[str]") -> str:
        content = m.group(1) + m.group(2)
        if _STRENGTH_RE.search(content):
            # Has strength — keep, drop only the noise words
            stripped = re.sub(
                r"(tapered|decreased|increased|adjusted|reduced|dose|was|previously)",
                "", content, flags=re.IGNORECASE,
            ).strip()
            return f" ({stripped}) "
        return " "

    text = _DOSE_CONTEXT_RE.sub(_ctx_repl, text)

    def _repl(m: "re.Match[str]") -> str:
        content = m.group(1).strip()
        if _STRENGTH_RE.search(content):
            return m.group(0)  # giữ parenthetical có strength
        return " "

    text = re.sub(r"\(([^)]*)\)", _repl, text)

    # R34: Range strength → LOW value (more conservative prescription)
    # "325-650 mg" → "325mg" trước khi main strength regex xử lý
    text = _RANGE_STRENGTH_RE.sub(
        lambda m: f"{m.group(1)}{m.group(3).lower()}", text
    )

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
        # R34: standalone "ml" is volume/noise (vd "guaifenesin ml" → drop).
        # Combined "5ml" (sau _STRENGTH_RE.sub) là token riêng, không bị ảnh hưởng.
        "ml",
    }

    def _compact(m: "re.Match[str]") -> str:
        suffix = f"/{m.group(3).lower()}" if m.group(3) else ""
        return f"{m.group(1)}{m.group(2).lower()}{suffix}"

    text = _STRENGTH_RE.sub(_compact, text)
    # R31 (2026-07-13): Fix Bug 9 — tokenize regex `[^a-z0-9/]+` was splitting
    # DECIMAL strengths like "0.5mg" into ["0", "5mg"] because "." is not in
    # the allowed char class. Added "." to keep-list so floats survive.
    tokens = [t for t in re.split(r"[^a-z0-9/.]+", text.lower()) if t]
    return " ".join(t for t in tokens if t not in skip_tokens)


# ---------------------------------------------------------------------- #
# Drug aliases — brand → generic (Vietnamese common drug names)
# Mục đích: khi LLM extract "Panadol 500mg" → lookup RxNorm bằng "paracetamol".
# Vì RxNorm index chủ yếu theo generic name, brand cần map về generic.
# Cập nhật 2026-07.
# ---------------------------------------------------------------------- #

def _load_drug_aliases() -> dict[str, str]:
    path = DATA_DIR / "drug_aliases.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Failed to load %s: %s", path, e)
        return {}

_DRUG_ALIASES: dict[str, str] = _load_drug_aliases()


# ════════════════════════════════════════════════════════════════════════════════
# R28 (2026-07-13): Auto-load drug INN whitelist từ data/drug_inn_cache.json
# (cache được sinh bởi scripts/build_mining_index.py từ rxnorm.jsonl).
# Trước đây whitelist cứng chỉ 13 entries → "truyền dịch yếu tố IX", "kháng sinh"
# không match → LLM classify sai. Nay whitelist ~63k unique INN → match hầu hết
# drug names. Cached set lookup O(1).
# ════════════════════════════════════════════════════════════════════════════════

def _load_drug_inn_whitelist() -> frozenset[str]:
    """Load unique INN/generic names từ cache. Trả frozenset (immutable, hashable)."""
    cache = DATA_DIR / "drug_inn_cache.json"
    if not cache.exists():
        logger.debug(
            "[R28] %s chưa tồn tại — chạy `python scripts/build_mining_index.py` "
            "để auto-mine drug INN.", cache.name,
        )
        return frozenset()
    try:
        return frozenset(json.loads(cache.read_text(encoding="utf-8")))
    except Exception as exc:
        logger.warning("[R28] Failed to load %s: %s", cache, exc)
        return frozenset()


_DRUG_INN_WHITELIST: frozenset[str] = _load_drug_inn_whitelist()
if _DRUG_INN_WHITELIST:
    logger.info("[R28] Loaded drug INN whitelist: %d entries", len(_DRUG_INN_WHITELIST))


# ════════════════════════════════════════════════════════════════════════════════
# R34 (2026-07-13): Auto-mined RxNorm signals — doseform re-rank, historical
# penalty, brand whitelist, resistance/class context filter.
#   - _HISTORICAL_RXCUI: rxcui có `historical=True` (159k rows) — soft penalty
#   - _BRAND_NAMES: brand names auto-mined từ "[Brand]" brackets (R34 5.4)
#   - _DOSEFORM_SCORE: doseform bonus table (R34 5.1)
#   - _NON_TREATMENT_TERMS: drug-class blacklist (nitrates, corticoid...)
# Cache: data/rxnorm_signals.json (~5-10s scan). Một lần duy nhất.
# ════════════════════════════════════════════════════════════════════════════════

_BRACKET_PATTERN = re.compile(r'\[([^\]]+)\]')

# 5.1 Doseform bonus table — R34 spec
_DOSEFORM_SCORE: dict[str, float] = {
    "extended release oral tablet": 15.0,
    "oral tablet": 10.0,
    "oral capsule": 5.0,
    "oral solution": 2.0,
    "oral suspension": 1.0,
    "injectable solution": 0.0,
    "injectable suspension": 0.0,
    "topical cream": 0.0,
    "topical ointment": 0.0,
    "disintegrating tablet": -5.0,
}

# Doseform keywords for L3 fuzzy post-filter (R34: avoid "urea → chemical" match)
_DOSEFORM_KEYWORDS = (
    "tablet", "capsule", "solution", "injection", "cream", "ointment",
    "gel", "patch", "syrup", "suspension", "powder", "spray", "drops",
    "inhaler", "suppository", "granules",
)

# R34: Drug-class blacklist (lookup() sẽ return [] nếu query thuần class term)
_NON_TREATMENT_TERMS: frozenset[str] = frozenset({
    "nitrates", "corticoid", "corticosteroid", "nsaid", "nsaids",
    "kháng sinh", "kháng viêm", "kháng đông",
    "thuốc chống đông", "thuốc giảm đau", "thuốc hạ sốt",
    "thuốc lợi tiểu", "thuốc an thần", "thuốc chống viêm",
    "thuốc kháng sinh",
})

# R42 (2026-07-14): Lab chemicals that have RxNorm SCDs (mostly topical/lab use)
# but should NOT be returned as drug lookups when queried bare (no strength,
# no drug context). Examples: 'urea' (mostly topical), 'creatinine' (lab test).
_LAB_CHEMICALS: frozenset[str] = frozenset({
    "urea", "creatinine", "hemoglobin", "albumin", "glucose", "sodium",
    "potassium", "chloride", "calcium", "magnesium", "phosphate",
    "lactate", "bicarbonate",
})


def _extract_brand_from_brackets(name: str) -> str | None:
    """Extract brand text từ 'Ingredient Strength [Brand Name]' format."""
    m = _BRACKET_PATTERN.search(name)
    return m.group(1).strip() if m else None


def _extract_doseform_score(name_lower: str, wants_extended: bool = False) -> float:
    """Score bonus dựa trên doseform name (5.1). Order: specific → generic.

    R34: ER Oral Tablet chỉ +15 khi query có ER signal (xl/xr/er/sr/24hr);
    nếu không, PENALIZE (R42 2026-07-14: was returning OT bonus = 10.0, làm ER
    candidates rank cao hơn OT plain khi query không có ER signal).
    """
    if "extended release oral tablet" in name_lower:
        return _DOSEFORM_SCORE["extended release oral tablet"] if wants_extended else -10.0
    if "disintegrating tablet" in name_lower:
        return _DOSEFORM_SCORE["disintegrating tablet"]
    if "oral tablet" in name_lower:
        return _DOSEFORM_SCORE["oral tablet"]
    if "oral capsule" in name_lower:
        return _DOSEFORM_SCORE["oral capsule"]
    if "oral solution" in name_lower:
        return _DOSEFORM_SCORE["oral solution"]
    if "oral suspension" in name_lower:
        return _DOSEFORM_SCORE["oral suspension"]
    if "injectable" in name_lower or "injection" in name_lower:
        return _DOSEFORM_SCORE["injectable solution"]
    if "topical cream" in name_lower:
        return _DOSEFORM_SCORE["topical cream"]
    if "topical ointment" in name_lower:
        return _DOSEFORM_SCORE["topical ointment"]
    return 0.0


def _load_rxnorm_signals() -> tuple[frozenset[str], frozenset[str]]:
    """One-shot scan rxnorm.jsonl → (historical_rxcui, brand_names). Cached to JSON."""
    cache = DATA_DIR / "rxnorm_signals.json"
    if cache.exists():
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            hist = frozenset(data.get("historical", []))
            brands = frozenset(data.get("brands", []))
            if hist or brands:
                logger.info("[R34] Loaded cached signals: %d historical, %d brands",
                            len(hist), len(brands))
                return hist, brands
        except Exception as exc:
            logger.debug("[R34] Cache read fail (%s) — rebuilding", exc)

    historical: set[str] = set()
    brands: set[str] = set()

    jsonl = DATA_DIR / "rxnorm.jsonl"
    if jsonl.exists():
        t0 = time.time()
        with jsonl.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rxcui = str(r.get("rxcui", "")).strip()
                if not rxcui:
                    continue
                # 5.2: collect historical
                if r.get("historical"):
                    historical.add(rxcui)
                # 5.4: mine brand names từ "[Brand]" brackets
                for m in _BRACKET_PATTERN.finditer(r.get("name", "")):
                    for part in m.group(1).split("/"):
                        b = part.strip()
                        if 3 <= len(b) <= 50 and any(c.isalpha() for c in b):
                            brands.add(b.lower())
        logger.info("[R34] Mined %d historical + %d brands (%.1fs)",
                    len(historical), len(brands), time.time() - t0)

    # Save cache
    try:
        cache.write_text(
            json.dumps(
                {"historical": sorted(historical), "brands": sorted(brands)},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("[R34] Cache save fail: %s", exc)

    return frozenset(historical), frozenset(brands)


_HISTORICAL_RXCUI, _BRAND_NAMES = _load_rxnorm_signals()


def _has_resistance_context(text: str) -> bool:
    """True nếu text là resistance mention (vd 'kháng vancomycin', 'kháng sinh').

    R34: Fix 0% pass trên `empty_resistance` — chuyển filter từ postprocess sang
    retriever để retriever standalone-correct.
    """
    t = text.lower()
    # "kháng sinh" = antibiotic class (generic) → reject
    if re.search(r"\bkháng\s+sinh\b", t):
        return True
    # "X kháng Y" pattern (resistance) → reject
    if re.search(r"\bkháng\s+\w{3,}", t):
        return True
    return False


def _has_drug_context(text: str) -> bool:
    """True nếu text có tín hiệu là drug đơn thuốc (R34: lọc nhiễu lab tokens).

    Cần ÍT NHẤT 1 trong:
      - Strength (digit + unit: mg, ml, %, ...)
      - Route token (po, iv, tiêm, uống, daily, bid, ...)
      - Doseform (tablet, capsule, cream, ...)
      - Compound separator (/, +, -, và)
      - Multi-word tổ hợp (≥ 2 alphanumeric tokens)
    """
    t = text.lower()
    if _STRENGTH_RE.search(t):
        return True
    if re.search(r"\b(po|iv|im|sc|pr|tiêm|tiêng|uống|siro|daily|bid|tid|qid|q\d+h|hs|prn|ac|pc|am|pm)\b", t):
        return True
    if re.search(r"\b(tablet|capsule|cream|injection|suspension|syrup|drop|oral|ointment|gel|patch|spray|viên|ống|gói)\b", t):
        return True
    if re.search(r"\s+[\/+\-]\s+|\s+và\s+", t):
        return True
    # Multi-word OK (>1 alphanumeric token) — likely drug name (vd "cipro flagyl")
    tokens = re.findall(r"[a-z0-9]{3,}", t)
    if len(tokens) >= 2:
        return True
    return False


def _alias_to_generic(drug_text: str) -> str | list[str]:
    """Translate brand name / misspellings / compound in drug text → generic name(s).

    R29 (2026-07-13 spec round 2): Hỗ trợ compound drug names — value trong
    data/drug_aliases.json có thể là LIST (compound, vd 'ciproflagyl' → ['cipro', 'flagyl']).
    Returns:
        - str: cho single brand → generic translation
        - list[str]: cho compound drug → nhiều generic names
        - str input: nếu không match (no-op)
    """
    if not drug_text or not _DRUG_ALIASES:
        return drug_text
    text_lower = drug_text.lower().strip()

    # Strip common Vietnamese prefix words before matching
    prefix_re = re.compile(r"^(?:viên\s+uống|viên\s+nén|viên\s+nang|thuốc\s+viên|thuốc\s+tiêm|thuốc\s+uống|thuốc|viên|viêm|tiêm|ống|gói|lọ|dung\s+dịch|hỗn\s+dịch|siro)\s+", re.IGNORECASE)
    stripped_prefix = ""
    m = prefix_re.match(drug_text)
    while m:
        stripped_prefix += m.group(0)
        drug_text = drug_text[len(m.group(0)):].lstrip()
        text_lower = drug_text.lower().strip()
        m = prefix_re.match(drug_text)

    # Thử match từ dài nhất trước
    for brand in sorted(_DRUG_ALIASES.keys(), key=len, reverse=True):
        value = _DRUG_ALIASES[brand]
        is_compound = isinstance(value, list)
        if text_lower == brand:
            # Exact match → return generic name(s)
            return list(value) if is_compound else value
        # Brand như PREFIX (vd "Augmentin 1g") → replace and append rest
        # R37 (2026-07-15): Yêu cầu word boundary ở CUỐI brand (tránh "omeprazol" match "omeprazole")
        if (
            text_lower.startswith(brand + " ")
            or text_lower == brand
            or (len(text_lower) > len(brand) and text_lower.startswith(brand) and not text_lower[len(brand)].isalnum())
        ):
            rest = drug_text[len(brand):].lstrip() if drug_text[len(brand):].strip() else ""
            if is_compound:
                # Compound: trả list các tên + strength suffix
                if rest:
                    return [f"{g} {rest}".strip() for g in value]
                return list(value)
            return f"{value} {rest}".strip() if rest else value
        # Brand nằm giữa hoặc có từ lót
        pattern = r"\b" + re.escape(brand) + r"\b"
        if re.search(pattern, text_lower):
            if is_compound:
                # Compound mid-text: replace 1 occurrence with FIRST generic, return single
                # (không safe để split giữa text lúc mid-match)
                replaced = re.sub(pattern, value[0], drug_text, count=1, flags=re.IGNORECASE)
                return replaced.strip()
            replaced = re.sub(pattern, value, drug_text, count=1, flags=re.IGNORECASE)
            return replaced.strip()

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
    rxcui_to_idx: dict[str, int] = field(default_factory=dict)  # R33

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
        # R33: also build rxcui_to_idx for O(1) name lookup by rxcui
        idx.rxcui_to_idx = {r: i for i, r in enumerate(idx.rxcuis)}
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
        # R34: 8.4k rows có ingredient="" (vd '24 HR metoprolol succinate 50 MG Extended Release Oral Tablet').
        # Fallback: parse ingredient từ name (skip "HR"/"Extended Release"/doseform).
        if not ing_norm and name:
            name_tokens = re.findall(r"[A-Za-z][a-z]+", name)
            skip_words = {
                "hr", "release", "extended", "oral", "tablet", "capsule",
                "solution", "injection", "cream", "ointment", "gel", "patch",
                "spray", "powder", "drops", "syrup", "suspension", "suppository",
                "granules", "delayed", "disintegrating", "chewable",
            }
            filtered = [t for t in name_tokens if t.lower() not in skip_words]
            if filtered:
                ing_norm = _normalize_ingredient(" ".join(filtered))
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
        """Pipeline L1 + L4 + L5 (fast exact path).

        Args:
            drug_text: text đã được strip route/freq. `_parse_drug` sẽ internally
                        strip lại (idempotent) để tách (ing, str).
        Returns:
            list[rxcui] (max 1).
        """
        return self._lookup_with_original(drug_text, original_text=None)

    def _lookup_with_original(
        self, drug_text: str, original_text: Optional[str] = None
    ) -> list[str]:
        """Lookup với original_text pass-through (R34: doseform hint detection).

        `original_text` là query GỐC (chưa strip) — dùng để detect doseform hint
        mà `_parse_drug` đã strip mất (vd 'Spiramycin Oral Tablet po prn' → sau
        strip là 'spiramycin' → mất thông tin doseform).
        """
        orig = original_text or drug_text
        ing, strength = _parse_drug(drug_text)
        if not ing:
            return []

        # L1: Exact (ingredient, strength) tuple
        # R42 (2026-07-14): Reject known lab chemicals (urea, creatinine, etc.)
        # even if they appear in INN whitelist (whitelist was auto-mined from
        # RxNorm, includes these as topical/lab SCDs).
        if ing.lower() in _LAB_CHEMICALS and not strength:
            return []
        if strength:
            cands = self.by_ingredient_strength.get((ing, strength), [])
            # R34: ALWAYS merge common salt variants (vd 'pravastatin' → 'pravastatin sodium').
            # User clinical practice thường dùng salt form, base name lookup phải expand.
            for salt in ("sodium", "hydrochloride", "tartrate", "sulfate",
                         "potassium", "calcium", "maleate", "mesylate", "besylate"):
                cands.extend(
                    self.by_ingredient_strength.get((f"{ing} {salt}", strength), [])
                )
            # Dedup preserve order
            if cands:
                cands = list(dict.fromkeys(cands))
                ranked = _rank_rxnorm_candidates(
                    cands, self.rxcui_to_idx, self.names, orig
                )
                return [ranked[0]]

            # L4: Compound split (also try salt variants for each part)
            if " / " in strength:
                for sub in strength.split(" / "):
                    sub = sub.strip()
                    if not sub:
                        continue
                    sub_cands = self.by_ingredient_strength.get((ing, sub), [])
                    for salt in ("sodium", "hydrochloride", "tartrate", "sulfate"):
                        sub_cands.extend(
                            self.by_ingredient_strength.get((f"{ing} {salt}", sub), [])
                        )
                    if sub_cands:
                        sub_cands = list(dict.fromkeys(sub_cands))
                        ranked = _rank_rxnorm_candidates(
                            sub_cands, self.rxcui_to_idx, self.names, orig
                        )
                        return [ranked[0]]

        # R41 (2026-07-14): L1B closest-strength fallback BEFORE L5.
        # Trước đây fall-through trực tiếp sang L5 (by_ingredient) → trả về cands[0]
        # (FIRST dict entry, thường là lowest rxcui = smallest strength).
        # Bug: "clonazepam 1.5 mg" → L1 miss → L5 returns cands[0] = 0.5MG
        # thay vì closest = 1MG (dist 0.5 thay vì 1.0).
        if strength:
            closest = self.closest_strength_lookup(drug_text, drug_text)
            if closest:
                return closest

        # L5: Ingredient-only (NO strength) — R34 conditional re-rank.
        # CHỈ fire khi query có drug context HOẶC ingredient nằm trong INN whitelist.
        # Bare single-word lab chemicals như 'urea', 'creatinine', 'hemoglobin' → reject.
        # R34 FIX: Bare drug names như 'doxycycline', 'atenolol' (real RxNorm drugs,
        # nhưng không có strength/route) phải lookup được qua L5.
        # R42 (2026-07-14): Extra guard for lab chemicals (urea, creatinine, ...)
        # even when they have RxNorm SCDs and are in INN whitelist.
        if ing.lower() in _LAB_CHEMICALS:
            return []
        cands = self.by_ingredient.get(ing, [])
        if cands:
            has_volume_in_orig = bool(re.search(r"\b\d+(?:\.\d+)?\s*ml\b", orig.lower()))
            ingredient_in_inn = ing.lower() in _DRUG_INN_WHITELIST
            # Guard: bare ingredient name → chỉ return nếu INN whitelist (real drug)
            if not _has_drug_context(orig) and not has_volume_in_orig and not ingredient_in_inn:
                return []  # likely lab chemical noise (urea, hemoglobin, etc.)
            if len(cands) > 1:
                ranked = _rank_rxnorm_candidates(
                    cands, self.rxcui_to_idx, self.names, orig
                )
                return [ranked[0]]
            return [cands[0]]
        return []

    def closest_strength_lookup(self, drug_text: str, original_text: str = "") -> list[str]:
        """R34 (2026-07-13): Closest strength fallback khi L1 miss.

        Khi query có strength (vd '1.5 mg') nhưng by_ingredient_strength lookup miss
        (vd 'clonazepam 1.5 MG' không có trong data), tìm SCD cùng ingredient với
        numerical strength GẦN NHẤT. Returns ranked rxcui list (max 1).

        Args:
            drug_text: stripped text (cho _parse_drug)
            original_text: full query (cho re-rank context)

        Returns:
            list[rxcui] of closest matches (max 1 re-ranked), or [] nếu không tìm được.
        """
        orig = original_text or drug_text
        ing, strength = _parse_drug(drug_text)
        if not ing or not strength:
            return []
        q_val = _parse_strength_value(strength)
        if q_val is None:
            return []

        # Find candidate keys for this ingredient với strength values
        scored_keys = []
        for k, cands in self.by_ingredient_strength.items():
            if k[0] != ing:
                continue
            k_val = _parse_strength_value(k[1])
            if k_val is None:
                continue
            dist = abs(k_val - q_val)
            scored_keys.append((dist, cands))

        if not scored_keys:
            return []
        scored_keys.sort(key=lambda x: x[0])
        # Lấy rxcui list từ closest
        closest_cands = scored_keys[0][1]
        if not closest_cands:
            return []
        # Re-rank top-1 với full signals (dùng original text cho context)
        ranked = _rank_rxnorm_candidates(
            closest_cands, self.rxcui_to_idx, self.names, orig
        )
        return [ranked[0]]
# R32 (2026-07-13): Name-based re-ranking for RxNorm candidates
# R34 (2026-07-13): +5.1 doseform, +5.2 historical, +5.4 brand tighten
# Gold uses: GENERIC (no [Brand] bracket), Oral Tablet form, shorter name.
# ════════════════════════════════════════════════════════════════════════════════

_BRACKET_BRAND_PENALTY = 50.0
_DISINTEGRATING_PENALTY = 5.0
_NON_ORAL_TABLET_PENALTY = 1.0
_EXTENDED_RELEASE_BONUS = 30.0  # R33: prefer XL/ER/SR when input has xl/er/sr/24hr token
# R34: thêm — known-branded penalty (no-bracket brand like "Tylenol Extra Strength")
_KNOWN_BRAND_PENALTY = 25.0
# R34: penalty cho `historical=True` entries (5.2)
_HISTORICAL_PENALTY = 2.0


def _rank_rxnorm_candidates(
    rxcui_list: list[str],
    rxcui_to_idx: dict,
    names: list[str],
    drug_text: str,
    vec_scores: Optional[dict[str, float]] = None,
    bm25_scores: Optional[dict[str, float]] = None,
) -> list[str]:
    """Re-rank RxNorm candidates dựa trên name + (optionally) vec/bm25 score.

    R34: Tích hợp 5.1 (doseform), 5.2 (historical), 5.4 (known brand ngoài bracket),
    5.3 (hybrid vec + bm25 + name weighted).

    Scoring components (higher = better):
      - Có [BrandName] bracket              : -50  (deprioritize branded)
      - Known brand không bracket (5.4)     : -25  (heuristic: title-case token)
      - Disintegrating form                  : -5
      - Form ≠ "Oral Tablet" AND "po" in text: -1  (prefer OT for oral route)
      - Extended release want + match       : +30
      - Extended release want + non-release  : -5
      - Doseform bonus (5.1)                 : +15 / +10 / +5 / +2 / 0 / -5
      - Historical entry (5.2)               : -2 (soft penalty)
      - Hybrid (5.3, chỉ khi truyền scores) : 50*vec + 30*bm25 + 0.5*name_signal

    Args:
        rxcui_list: candidates from by_ingredient_strength / by_ingredient
        rxcui_to_idx: rxcui → index map
        names: parallel name list
        drug_text: original query (for context like "po" → Oral Tablet preference)
        vec_scores: optional {rxcui: cosine} from BGE-M3 (5.3 hybrid path)
        bm25_scores: optional {rxcui: bm25_score} (5.3 hybrid path)

    Returns:
        Sorted rxcui list (best first). Nếu no names found, return input order.
    """
    if len(rxcui_list) <= 1:
        return rxcui_list

    drug_lower = drug_text.lower()
    wants_oral = "po" in drug_lower or "uống" in drug_lower
    wants_extended = bool(
        re.search(r"\b(?:xl|xr|er|sr|24\s*hr|24hr|extended\s+release)\b", drug_lower)
    )
    has_hybrid = vec_scores is not None or bm25_scores is not None
    # R42 (2026-07-14): Use PARSED strength (via _normalize_strength) to detect
    # "real strength" in query. Raw _STRENGTH_RE matches "5 ml" as strength but
    # _normalize_strength filters bare ML as volume → inconsistent. Use the
    # normalize result to determine if query has actual strength value.
    _qstr_m = _STRENGTH_RE.search(drug_text)
    query_has_str = bool(_qstr_m and _normalize_strength(_qstr_m.group(0)))
    # R42 (2026-07-14): Detect volume in query (digit + ml). ML→MG conversion
    # only meaningful if PLAIN SCDs (no compound, no /ML concentration) use
    # mass units (MG). E.g., nystatin's "MG" SCDs are all "MG/ML" concentrations
    # (not single-dose strengths), so ml→mg conversion yields no usable SCD.
    # If no plain MG SCD exists, conversion impossible → prefer parent.
    _vol_m = re.search(r"\b\d+(?:\.\d+)?\s*ml\b", drug_text.lower())
    query_has_volume = bool(_vol_m)
    _has_plain_mg_cands = any(
        # Plain MG (not MG/ML concentration, not compound with /)
        " mg" in f" {names[rxcui_to_idx[c]].lower()} " and
        " / ml" not in names[rxcui_to_idx[c]].lower() and
        " / " not in names[rxcui_to_idx[c]].lower() and
        "mg/ml" not in names[rxcui_to_idx[c]].lower()
        for c in rxcui_list if c in rxcui_to_idx
    ) if query_has_volume else True
    # R42 (2026-07-14): Detect if query uses BRAND name explicitly.
    # Vd: "prograf 1mg bid" → user wrote brand "prograf" → preserve brand info.
    # When True, REMOVE the [Brand] bracket penalty (was -50) for that brand.
    # This makes brand SCDs (vd 564557 "tacrolimus 1 MG [Prograf]") rank higher
    # when query explicitly uses the brand name.
    # IMPORTANT: Only treat as brand if KEY != VALUE (alias actually translates).
    # "aspirin" → "aspirin" is a no-op alias, NOT a brand translation.
    _query_brand: Optional[str] = None
    if _DRUG_ALIASES:
        _q_lower = drug_text.lower().strip()
        for _brand in sorted(_DRUG_ALIASES.keys(), key=len, reverse=True):
            _value = _DRUG_ALIASES[_brand]
            _value_str = _value if isinstance(_value, str) else _value[0]
            # Skip no-op aliases (key == value)
            if _brand.lower() == _value_str.lower():
                continue
            if _q_lower == _brand or _q_lower.startswith(_brand + " ") or _q_lower.startswith(_brand + ","):
                # Use canonical brand name (preserves original case)
                _query_brand = _brand
                break
    # R44 (2026-07-14): preserve input order, filter rxcui_to_idx later
    original_rxcui_list = list(rxcui_list)
    # R44b: ranked_cands = rxcui_list filtered to only valid rxcui_to_idx entries
    ranked_cands = [c for c in rxcui_list if c in rxcui_to_idx]

    def _name_signal(rxcui: str, name_lower: str, drug_text_lower: str) -> float:
        """Pure name-based signal (5.1, 5.2, 5.4 + R34 strength-presence)."""
        s = 0.0
        # Bracket brand
        if "[" in name_lower and "]" in name_lower:
            # R42 (2026-07-14): If query uses brand name explicitly, REMOVE
            # bracket penalty AND add brand-match bonus for matching brand
            # (so "prograf 1mg bid" → 564557 "tacrolimus 1 MG [Prograf]" wins
            # over 427808 plain generic).
            if _query_brand is not None:
                _brand_in_name = _extract_brand_from_brackets(
                    names[rxcui_to_idx[rxcui]]
                )
                if _brand_in_name and _brand_in_name.lower() == _query_brand.lower():
                    # Matching brand — bonus to overcome OT doseform bonus
                    s += 15.0
                else:
                    s -= _BRACKET_BRAND_PENALTY
            else:
                s -= _BRACKET_BRAND_PENALTY
        # Known brand không có bracket (5.4) — title-case suspect
        if not (("[" in name_lower and "]" in name_lower)):
            # Heuristic: bất kỳ title-case token nào ∈ _BRAND_NAMES thì penalize
            for tok in re.findall(r"\b[A-Z][a-zA-Z]+\b", names[rxcui_to_idx[rxcui]]):
                if tok.lower() in _BRAND_NAMES:
                    s -= _KNOWN_BRAND_PENALTY
                    break
        # Disintegrating
        if "disintegrating" in name_lower:
            s -= _DISINTEGRATING_PENALTY
        # For oral prescriptions, prefer OT
        if wants_oral and "oral tablet" not in name_lower:
            s -= _NON_ORAL_TABLET_PENALTY
        # Extended Release
        if wants_extended:
            if "extended release" in name_lower or " 24 hr " in f" {name_lower} ":
                s += _EXTENDED_RELEASE_BONUS
            elif "release" not in name_lower:
                s -= 5.0
        # 5.1: Doseform bonus (with R34 ER-conditional logic)
        df_score = _extract_doseform_score(name_lower, wants_extended=wants_extended)
        # R42 (2026-07-14): When query uses brand explicitly + no explicit
        # doseform in query → prefer SIMPLE brand SCDs (no doseform specified
        # in name) over brand+doseform combinations. Vd "prograf 1mg bid" →
        # 564557 "tacrolimus 1 MG [Prograf]" wins over 108513 "tacrolimus 1 MG
        # Oral Capsule [Prograf]".
        if _query_brand is not None and not wants_extended:
            has_doseform_in_name = any(kw in name_lower for kw in (
                "oral tablet", "oral capsule", "oral solution",
                "oral suspension", "tablet", "capsule",
                "solution", "suspension",
            ))
            has_doseform_in_query = any(kw in drug_text.lower() for kw in (
                "tablet", "capsule", "solution", "suspension",
            ))
            if not has_doseform_in_query and not has_doseform_in_name:
                # Simple brand SCD (no doseform) — BIG bonus to dominate doseform bonus
                s += 25.0
            elif not has_doseform_in_query and has_doseform_in_name:
                # Brand SCD with doseform but query has no doseform → penalize
                s -= 15.0
        s += df_score
        # 5.2: Historical penalty
        if rxcui in _HISTORICAL_RXCUI:
            s -= _HISTORICAL_PENALTY
        # R34: Salt preference (user pipeline thường pick salt over plain base)
        s += _salt_preference_score(name_lower)
        # R34: Compound penalty — prefer plain SCD over compound (' / ') khi
        # query không có ' / '. (vd 'guaifenesin ml' → plain, không phải
        # 'guaifenesin 400 MG / pseudoephedrine 40 MG').
        if " / " in name_lower and " / " not in drug_text_lower:
            s -= 3.0
        # R34: Strength-presence match (user-friendly cho L5 path)
        # - Query KHÔNG có strength + name CÓ strength: penalty (-2) — user muốn generic
        # - Query CÓ strength + name KHÔNG có strength: penalty (-5) — name quá generic
        query_has_str = bool(_STRENGTH_RE.search(drug_text_lower))
        name_has_str = bool(_STRENGTH_RE.search(name_lower))
        if not query_has_str and name_has_str:
            s -= 2.0
        elif query_has_str and not name_has_str:
            s -= 5.0
        return s

    def _score(rxcui: str) -> float:
        if rxcui not in rxcui_to_idx:
            return 0.0
        name_lower = names[rxcui_to_idx[rxcui]].lower()
        name_sig = _name_signal(rxcui, name_lower, drug_lower)

        # R42 (2026-07-14): Volume query (5 ml) + NO plain MG SCDs (e.g.,
        # nystatin uses UNT not MG for primary strength). ML→MG conversion
        # impossible, so apply HARD preference for parent SCD (no strength)
        # by overriding score with 1e6 (dominates over all name_signal).
        if query_has_volume and not _has_plain_mg_cands:
            name_has_str_local = bool(_STRENGTH_RE.search(name_lower))
            if not name_has_str_local:
                # Parent SCD (no strength) for volume query without MG unit
                return 1e6  # hard preference, dominates any name signal
            else:
                # Non-parent (has strength like UNT) for volume query → reject
                return -1e6

        if has_hybrid:
            # 5.3: vec + bm25 dominates, name signal = tiebreaker (R34 weighted)
            v = (vec_scores or {}).get(rxcui, 0.0) or 0.0
            b = (bm25_scores or {}).get(rxcui, 0.0) or 0.0
            # 50*vec (range 0-50) + 30*bm25 (range 0-30) + 0.5*name_sig (range ~-50/+30)
            return 50.0 * v + 30.0 * b + 0.5 * name_sig
        # L1 path: pure name signal
        return name_sig

    # Sort by score DESC, secondary tiebreak: when query has no strength,
    # prefer (1) names without strength (no-str penalty candidate), then
    # (2) HIGHEST numerical strength (clinical convention for OTC SCD).
    # Tertiary: input order (stable).
    def _sort_key(rxcui):
        if rxcui not in rxcui_to_idx:
            return (-1e9, 0.0, 9999999)
        name_l = names[rxcui_to_idx[rxcui]].lower()
        # R44: use ORIGINAL input order for tiebreaker (preserve dict insertion)
        try:
            order_idx = original_rxcui_list.index(rxcui)
        except ValueError:
            order_idx = 9999999
        # R42 (2026-07-14): Secondary sort - prefer SCD with lowest typical adult dose.
        # Bare ingredient name (e.g. "guaifenesin") thường là generic prescription →
        # chọn SCD với dose phổ biến (800 MG Oral Tablet) thay vì parent hay 1200 MG ER.
        # Fix: prefer "Oral Tablet"/"Oral Solution" doseform + LOWEST strength.
        # Higher dose forms (e.g. 1200 MG ER) ít phổ biến cho OTC generic prescription.
        sec = 0.0
        if not query_has_str:
            # Check if ORIGINAL query has explicit doseform keyword OR volume.
            # - Doseform in query + name has doseform + name has NO strength →
            #   parent SCD query (vd "Sulfadiazine Oral Tablet" → 373977).
            # - Volume in query (digit+ml) → parent SCD preferred
            #   (vd "nystatin oral suspension 5 ml" → 7597 "nystatin").
            query_has_doseform = any(kw in drug_text.lower() for kw in (
                "oral tablet", "oral capsule", "oral solution",
                "oral suspension", "tablet", "capsule",
                "solution", "suspension",
            ))
            query_has_volume = bool(re.search(
                r"\b\d+(?:\.\d+)?\s*ml\b", drug_text.lower()
            ))
            name_has_str = bool(_STRENGTH_RE.search(name_l))
            has_doseform = any(kw in name_l for kw in (
                "oral tablet", "oral capsule", "oral solution",
                "oral suspension", "tablet", "capsule",
                "solution", "suspension",
            ))
            if query_has_volume and not name_has_str and not has_doseform:
                # R42 (2026-07-14): Query has volume (5 ml) + name is bare
                # parent (no strength, no doseform). STRONGLY prefer parent SCD
                # (gold for "nystatin oral suspension 5 ml" → 7597 "nystatin").
                # Use 1e6 to dominate over score differences (max ~200).
                sec = 1e6
            elif query_has_volume:
                # R42 (2026-07-14): Volume query (5 ml) + name has strength.
                # ML→MG conversion impossible if data uses UNT/IU (substance-
                # specific, not volumetric). E.g., nystatin's strength is in
                # UNT not MG → no meaningful conversion possible.
                # In this case, prefer parent SCD over mass-based SCDs.
                # Heuristic: if ingredient has NO candidates with MG/G strength
                # (only UNT/IU), set high preference for parent.
                # We approximate by checking if the dominant strength unit in
                # this batch is non-mass (UNT/IU only).
                v = _parse_strength_value(name_l)
                if v is not None and v >= 1000:
                    # Large unit-only strengths (100000+ UNT) typical for
                    # anti-infectives like nystatin. Penalize heavily.
                    sec = -1e6
                elif v is not None and v >= 400:
                    # Mass-like strength (probably MG/G): could be valid
                    sec = -abs(v - 800.0)
                else:
                    # Small or unknown strength: weak preference
                    sec = -500.0 - (v or 0)
            elif query_has_doseform and not name_has_str and has_doseform:
                # R42 (2026-07-14): Query explicitly says doseform + name is
                # parent SCD (no strength). STRONGLY prefer (gold for
                # "Sulfadiazine Oral Tablet" → 373977).
                sec = 1000.0
            elif not name_has_str and has_doseform:
                # Bare drug name (e.g. "guaifenesin") + doseform "Oral Tablet"
                # but query has no explicit doseform → generic SCD parent
                # entry. De-prioritize vs SCDs.
                sec = -500.0
            elif not name_has_str:
                # Bare ingredient + no doseform → lowest
                sec = -1e9
            elif name_has_str:
                # R42 (refined 2026-07-14): Prefer strengths closest to TYPICAL
                # ADULT OTC DOSE (target = 800 MG). Heuristic rationale:
                #   - Strength < 400 MG → pediatric / low-dose (reject)
                #   - Strength in [400, 1200] MG → typical adult OTC range;
                #     among those, prefer MULTIPLES OF 200 closest to 800 MG.
                #   - Strength > 1200 MG → extended-release / high-dose (reject).
                # Bug history: "guaifenesin ml" was returning 1200 MG ER (310621)
                # because the original "lowest strength wins" picked 100 MG instead
                # of gold 800 MG. Fixed via target-based ranking + ER penalty.
                v = _parse_strength_value(name_l)
                if v is None or v <= 0:
                    sec = 0.0
                else:
                    if 400 <= v <= 1200 and v % 200 == 0:
                        # Round multiple of 200 in adult range: rank by closeness
                        # to 800 MG (target). Smaller distance = higher sec.
                        sec = -abs(v - 800.0)
                    else:
                        # Outside preferred pattern: heavy penalty proportional
                        # to distance from [400, 1200] range.
                        if v < 400:
                            sec = -1000.0 - v
                        else:
                            sec = -1000.0 - (v - 1200.0)
        # Primary score (must compute per-rxcui, not capture outer variable)
        score = _score(rxcui)
        # Tertiary: input order (negative for DESC sort)
        return (score, sec, -order_idx)

    # R44: sort ranked_cands (those in rxcui_to_idx), use original order
    scored = sorted(ranked_cands, key=_sort_key, reverse=True)
    return scored


# ════════════════════════════════════════════════════════════════════════════════
# Vector Search — BGE-M3 cosine
# ════════════════════════════════════════════════════════════════════════════════


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
        self._jsonl_path = jsonl_path or (DATA_DIR / "rxnorm.jsonl")
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
                # R37 (2026-07-20): Validate shape matches current RxNorm data.
                # Embeddings may be STALE if rxnorm.jsonl rebuilt without regenerating.
                # Mismatch: file tồn tại + load OK nhưng số vectors khác RxNorm codes → coi stale,
                # bỏ qua và regenerate để đảm bảo consistency.
                expected_count = len(self.names) if self.names else len(self.codes)
                if expected_count and self._embeddings.shape[0] != expected_count:
                    logger.warning(
                        "RxNormVectorSearch: Embeddings STALE — file %s có shape (%d, ...) "
                        "nhưng RxNorm data hiện có %d codes. Force regenerate.",
                        self._embeddings_path.name,
                        self._embeddings.shape[0],
                        expected_count,
                    )
                    self._embeddings = None  # Force regenerate
                else:
                    logger.info(
                        "RxNormVectorSearch: loaded embeddings %s (shape: %r)",
                        self._embeddings_path.name, self._embeddings.shape,
                    )
            except Exception as exc:
                logger.warning("Embeddings load fail (%s)", exc)
                self._embeddings = None

        self._loaded = True

    def search(self, query: str, *, top_k: int = 10, threshold: float = 0.4) -> list[str]:
        """Cosine search trên top_k. Filter ≥ threshold."""
        self._ensure_loaded()
        if not query or not self.codes:
            return []

        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                # R43 (2026-07-14): Support local model path via env var để tránh
                # download chậm từ HF khi chạy trên Kaggle (cache không persistent).
                import os
                model_path = os.environ.get("BGE_M3_PATH", "BAAI/bge-m3")
                if os.path.isdir(model_path):
                    logger.info("RxNormVectorSearch: Loading BGE-M3 từ LOCAL path: %s", model_path)
                else:
                    logger.info("RxNormVectorSearch: Loading BGE-M3 từ HuggingFace: %s (sẽ download ~2.3GB)", model_path)
                self._model = SentenceTransformer(model_path)
                logger.info("RxNormVectorSearch: loaded BGE-M3")
            except ImportError:
                logger.error("Sentence-transformers chưa cài!")
                return []

        if self._embeddings is None:
            logger.info(
                "Auto-generating RxNorm embeddings (file %s invalid/missing)...",
                self._embeddings_path.name,
            )
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
                # R43 (2026-07-14): Support local model path via env var (consistent với search()).
                import os
                model_path = os.environ.get("BGE_M3_PATH", "BAAI/bge-m3")
                self._model = SentenceTransformer(model_path)
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
        self._jsonl_path = jsonl_path or (DATA_DIR / "rxnorm.jsonl")
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

        # Drug Pre-cleaner: loại bỏ route, freq, doseform trước khi search (vd "po bid 1 viên")
        cleaned_query = _strip_route_freq(query)
        search_q = cleaned_query if cleaned_query.strip() else query

        # 1. Get candidates from vector + BM25
        vec_codes = self.vector_search.search(search_q, top_k=fanout, threshold=0.0) or []
        bm25_codes, _ = self.bm25_index.search(search_q, top_k=fanout)

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
        # R34 (5.3 + 5.1): Hybrid re-rank top candidates bằng weighted score
        # Vec + bm25 dominate, name_signal (incl. doseform bonus) tiebreaker.
        # Always fires (not just k>1) — also helps single-result case where
        # doseform preference should override cosine (vd 'guaifenesin ml' → prefer
        # 'guaifenesin 800 MG Oral Tablet' over 'Guaifenesin 6 MG/ML').
        if matched:
            top_cands = [c for c, _ in matched]
            vec_all = self.vector_search.score_codes(query, top_cands) or {}
            bm25_all = self.bm25_index.score_codes(query, top_cands) or {}
            ranked = _rank_rxnorm_candidates(
                top_cands, {c: i for i, c in enumerate(self.vector_search.codes)},
                self.vector_search.names, query,
                vec_scores=vec_all, bm25_scores=bm25_all,
            )
            return ranked[:k]
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
        if isinstance(index, (str, Path)):
            index_path = Path(index)
            index = None
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
        self._llm_client = None

    # ------------------------------------------------------------------ #

    def lookup(self, drug_text: str) -> list[str]:
        """External lookup entry point. Delegates to _lookup_uncached with re-ranking (R32)."""
        if not drug_text:
            return []
        if not hasattr(self, '_cache'):
            self._cache = {}
        cache_key = drug_text.strip().lower()
        if cache_key in self._cache:
            return list(self._cache[cache_key])
        # Pass self.index.name_to_idx + self.index.names for R32 re-rank
        result = self._lookup_uncached(drug_text)
        if len(self._cache) > 4096:
            self._cache.clear()
        self._cache[cache_key] = result
        return list(result)

    def _lookup_uncached(self, drug_text: str) -> list[str]:
        """Tra RxNorm cho chuỗi thuốc VN/EN → list [rxcui] (0 hoặc 1 code, HIGH precision).

        R29 (2026-07-13 spec round 2): Support COMPOUND drug aliasing.
        R34 (2026-07-13 spec round 3): + Context filter (resistance/class) + 5.1/5.2/5.3/5.4 re-rank.
        """
        drug_text = drug_text.strip()
        _raw_text = drug_text  # R42: preserve ORIGINAL text (pre-alias) for brand detection

        # R34: Context filter — reject resistance / non-treatment class
        if _has_resistance_context(drug_text):
            logger.debug("R34: resistance context rejected: '%s'", drug_text)
            return []
        if drug_text.lower().strip() in _NON_TREATMENT_TERMS:
            logger.debug("R34: drug-class blacklist rejected: '%s'", drug_text)
            return []

        # R34: Capture ORIGINAL text (chưa alias/strip) để doseform hint detection
        drug_text_orig = drug_text  # alias-translated hoặc original đều OK cho hint

        # Pre-step: Translate brand → generic
        alias_result = _alias_to_generic(drug_text)

        # R29: Compound detection
        if isinstance(alias_result, list) and len(alias_result) > 1:
            combined: list[str] = []
            for part in alias_result:
                try:
                    rxcui = self._lookup_single_part(part)
                    if rxcui:
                        for c in rxcui:
                            if c not in combined:
                                combined.append(c)
                except Exception as exc:
                    logger.warning("Compound lookup fail for '%s': %s", part, exc)
            if combined:
                logger.info("[R29] Compound drug '%s' → %s", drug_text, combined)
                return combined
            return []

        drug_text = alias_result if isinstance(alias_result, str) else drug_text
        # R42 (2026-07-14): Keep ORIGINAL (pre-alias) text in drug_text_orig_for_brand
        # so brand-aware re-rank works (vd "prograf 1mg bid" → 564557 [Prograf]).
        # Use post-alias drug_text_orig for doseform hint context.
        drug_text_orig = drug_text  # post-alias, used for doseform hint
        drug_text_orig_for_brand = _raw_text  # pre-alias, used for brand detection

        # L1: Exact (ing, strength) — with R32 name re-rank (R34: 5.1/5.2/5.3/5.4)
        # R42: pass _orig_with_brand_hint for brand-aware re-rank, but orig for doseform hint
        result = self.index._lookup_with_original(
            drug_text, original_text=drug_text_orig_for_brand
        )
        if result:
            ranked = _rank_rxnorm_candidates(
                result, self.index.rxcui_to_idx, self.index.names, drug_text_orig_for_brand
            )
            return [ranked[0]]

        # L1B: Closest strength fallback (R34) — khi L1 miss, tìm SCD cùng ing
        # với strength gần nhất (vd 'clonazepam 1.5 mg' → 1 MG ≈ 197528).
        result_closest = self.index.closest_strength_lookup(
            drug_text, original_text=drug_text_orig
        )
        if result_closest:
            logger.debug(
                "RxNorm L1B closest strength: '%s' → %s",
                drug_text_orig, result_closest,
            )
            return result_closest[:1]  # already ranked top-1

        # L2: Hybrid search — threshold 0.78 strict, top_k=1
        # R34: also require drug context to skip noise lab tokens
        if (
            self.use_hybrid
            and self.hybrid_search is not None
            and _has_drug_context(drug_text)
        ):
            try:
                result = self.hybrid_search.search(
                    drug_text, top_k=1, threshold=0.78
                )
                if result:
                    return result[:1]
            except Exception as exc:
                logger.warning("Hybrid search fail (%s): %s", drug_text[:30], exc)

        # L3: Fuzzy WRatio >= 80 — R34: require drug context + matched doseform
        if _has_drug_context(drug_text):
            result = self._fuzzy_local(drug_text, threshold=80)
            if result:
                return result[:1]

        # L4: Fuzzy partial >= 75 — fallback
        if _has_drug_context(drug_text):
            result = self._fuzzy_local(drug_text, threshold=75)
            if result:
                logger.debug(
                    "RxNorm L4 loose fuzzy fallback matched '%s' → %s",
                    drug_text, result,
                )
                return result[:1]

        # L6: Multi-Hop Compound Drug Splitting
        if re.search(r'\s+[\/+\-]\s+|\s+và\s+', drug_text, re.IGNORECASE):
            parts = [p.strip() for p in re.split(r'\s+[\/+\-]\s+|\s+và\s+', drug_text, flags=re.IGNORECASE) if len(p.strip()) >= 3]
            if len(parts) >= 2 and len(parts) <= 4:
                logger.debug("Multi-hop compound drug splitting '%s' -> %s", drug_text, parts)
                combined: list[str] = []
                for part in parts:
                    sub_res = self.lookup(part)
                    for code in sub_res:
                        if code not in combined:
                            combined.append(code)
                if combined:
                    return combined

        # L7: LLM Fallback (Strict Validated)
        if hasattr(self, '_llm_client') and self._llm_client:
            try:
                from src.prompts import RXNORM_LLM_FALLBACK_PROMPT
                prompt = RXNORM_LLM_FALLBACK_PROMPT.format(entity_text=drug_text)
                response = self._llm_client.call_sync(prompt, max_tokens=30, temperature=0.1)
                codes = re.findall(r'\b\d{4,8}\b', response)
                valid_set = set(self.index.rxcuis)
                valid_codes = [c for c in codes if c in valid_set]
                if valid_codes:
                    logger.info("L7 LLM fallback RxNorm: '%s' → %s", drug_text, valid_codes[:1])
                    return valid_codes[:1]
            except Exception as exc:
                logger.warning("L7 LLM fallback RxNorm failed: %s", exc)

        return []  # confidence thấp → empty

    def _lookup_single_part(self, drug_text: str) -> list[str]:
        """R29 (2026-07-13): Sub-lookup cho 1 phần của compound drug.

        Trả về max 1 rxcui (Section 8 candidate discipline). Gọi internal pipeline
        L1 → L7 mà không qua bước L6 (compound) để tránh infinite recursion.

        R34 fix: apply brand→generic alias translation (vd 'flagyl' → 'metronidazole').
        Trước đây bare brand name miss L1 và bị fuzzy block (no drug context).
        """
        drug_text = drug_text.strip()
        if not drug_text:
            return []
        # R34: alias translation cho sub-part (vd 'flagyl' → 'metronidazole')
        aliased = _alias_to_generic(drug_text)
        if isinstance(aliased, str) and aliased != drug_text:
            drug_text = aliased
        result = self.index.lookup(drug_text)
        if result:
            return result[:1]
        if self.use_hybrid and self.hybrid_search is not None:
            try:
                result = self.hybrid_search.search(
                    drug_text, top_k=1, threshold=0.78
                )
                if result:
                    return result[:1]
            except Exception:
                pass
        result = self._fuzzy_local(drug_text, threshold=80)
        if result:
            return result[:1]
        result = self._fuzzy_local(drug_text, threshold=75)
        if result:
            return result[:1]
        return []  # don't recurse, don't LLM fallback for sub-lookups

    def _fuzzy_local(self, query: str, threshold: int = 70) -> list[str]:
        """Fuzzy match trên name list (rapidfuzz). R34: tightened with drug context.

        Pre-filter (R34): Query PHẢI có drug context — strength, route, doseform,
        compound separator, hoặc ≥2 alphanumeric tokens. Ngăn match nhầm bare
        lab chemicals (urea, hemoglobin, sodium) mà L3 trước đây pick up.

        Post-filter (R34): Matched RxNorm name PHẢI chứa doseform keyword
        (tablet, capsule, ...) hoặc strength (digit + unit). Đây là drug
        formulation, không phải chemical concept.

        Args:
            query: input text
            threshold: min WRatio/partial_ratio score (default 70)
        Returns:
            list[rxcui] (max 1) hoặc [] nếu không match
        """
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

        # R34 pre-filter: yêu cầu drug context
        if not _has_drug_context(query):
            return []

        q = " ".join(q_tokens)
        matches = process.extract(q, self.index.names, scorer=fuzz.WRatio, limit=5)
        matches += process.extract(q, self.index.names, scorer=fuzz.partial_ratio, limit=5)
        for name, score, _ in matches:
            if score >= threshold and name in self.index.name_to_idx:
                # R34 post-filter: matched name phải có doseform hoặc strength
                nl = name.lower()
                has_doseform = any(kw in nl for kw in _DOSEFORM_KEYWORDS)
                has_strength = bool(_STRENGTH_RE.search(name))
                if has_doseform or has_strength:
                    return [self.index.rxcuis[self.index.name_to_idx[name]]]
        return []


# ---------------------------------------------------------------------- #
# Persistence
# ---------------------------------------------------------------------- #


def load_index(path: Optional[Path] = None) -> RxNormIndex:
    """Nạp RxNormIndex từ JSON. Trả RxNormIndex rỗng nếu không có file hoặc file invalid.

    R37 (2026-07-20): Robust handling cho empty file (vd LFS skip-smudge làm file rỗng).
    Log warning để user biết cần regenerate index.
    """
    path = path or (DATA_DIR / "rxnorm_index.json")
    if not path.exists():
        return RxNormIndex()
    try:
        with path.open(encoding="utf-8") as f:
            content = f.read().strip()
        if not content:
            # Empty file — likely LFS skip-smudge hoặc manual edit
            logger.warning(
                "RxNorm index file %s is EMPTY (likely LFS skip-smudge). "
                "Returning empty RxNormIndex. Run scripts/build_rxnorm_index.py to regenerate.",
                path.name,
            )
            return RxNormIndex()
        return RxNormIndex.from_dict(json.loads(content))
    except json.JSONDecodeError as exc:
        logger.warning(
            "RxNorm index file %s invalid JSON: %s. Returning empty RxNormIndex.",
            path.name, exc,
        )
        return RxNormIndex()
    except Exception as exc:
        logger.warning(
            "RxNorm index file %s load fail: %s. Returning empty RxNormIndex.",
            path.name, exc,
        )
        return RxNormIndex()


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
    """Đọc JSONL [{rxcui, ingredient, strength, doseform, name, ...}] → build RxNormIndex.

    R34: KHÔNG skip rows có empty ingredient (8.4k rows). `add()` sẽ fallback
    parse ingredient từ name.
    """
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
            if rxcui:
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
