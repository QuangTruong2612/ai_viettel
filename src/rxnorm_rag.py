"""RxNorm RAG — tra cứu mã RxNorm cho mỗi thực thể THUỐC.

Pipeline 3 lớp (sau khi drop NIH API live):
  L1: Exact tuple match (ingredient, strength) → top-1 rxcui.
  L2: Combination drug — split strength by " / ", match nếu 1 thành phần khớp.
  L3: Ingredient-only exact match (không có strength) → top-1 rxcui.

Dữ liệu dùng: `data/rxnorm.jsonl` (RxNorm Current Prescribable Content, 46k rows,
schema {rxcui, ingredient, strength, doseform, ...}). Build index 1 lần qua
`scripts/build_rxnorm_index.py` → `data/rxnorm_index.json`.

Đặc điểm:
- Return 1 rxcui duy nhất (Jaccard metric yêu cầu đúng 1).
- Không gọi NIH API → pipeline chạy hoàn toàn offline.
- Strength normalization: "25mg" == "25 MG" == "25.0 MG" trong index keys.
"""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


# ---------------------------------------------------------------------- #
# Normalization
# ---------------------------------------------------------------------- #

_STRENGTH_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|iu|unit|%|meq|mEq)(?:/(mg|mcg|g|ml|iu|unit|%|meq|mEq))?",
    re.IGNORECASE,
)


def _normalize_strength(s: str) -> str:
    """Chuẩn hoá strength string.

    Quy tắc:
    - Lowercase, strip khoảng trắng giữa số và đơn vị.
    - Uppercase đơn vị (MG, MG/ML, ...).
    - Compound: split by " / " trả về list các thành phần đã normalize.
    """
    if not s:
        return ""
    s = s.strip().lower()
    # Collapse whitespace giữa số và đơn vị
    s = re.sub(r"(\d+(?:\.\d+)?)\s+(mg|mcg|g|ml|iu|unit|%|meq)", r"\1\2", s,
               flags=re.IGNORECASE)
    return s.upper()


def _normalize_ingredient(s: str) -> str:
    """Lowercase + strip + collapse whitespace."""
    return re.sub(r"\s+", " ", s.strip().lower()) if s else ""


def _strip_route_freq(text: str) -> str:
    """Loại bỏ VN parentheticals + route + frequency tokens.

    VD: 'metoprolol 25 mg (uống hôm nay) po bid' → 'metoprolol 25 mg'.
    """
    # Drop VN parentheticals (...): không drop nếu trong ngoặc có số + đơn vị liều
    def _repl(m: "re.Match[str]") -> str:
        content = m.group(1).strip()
        if _STRENGTH_RE.search(content):
            return m.group(0)  # giữ nguyên
        return " "

    text = re.sub(r"\(([^)]*)\)", _repl, text)

    # Drop các token route/freq thường gặp
    skip_tokens = {
        "po", "iv", "im", "sc", "sl", "pr", "topical", "inhale", "oral",
        "tablet", "tab", "capsule", "cap", "solution", "suspension",
        "injection", "inj", "cream", "ointment", "gel", "patch",
        "spray", "powder", "drop", "drops", "syrup",
        "extended", "release", "xl", "xr", "er", "sr", "la", "cr",
        "daily", "bid", "tid", "qid", "qhs", "qam", "qpm",
        "q6h", "q8h", "q12h", "prn", "qd", "qod", "hs", "ac", "pc",
        "uống", "tiêm", "tiêng", "viên", "ống", "gói", "lần", "ngày",
        "giờ", "tuần", "tháng", "sáng", "trưa", "chiều", "tối", "tối",
    }
    # Tokenize (giữ cả số liều làm 1 token "25mg")
    # 1. Ghép số + đơn vị: "25 mg" → "25mg"; compound "100 mg/ml" → "100mg/ml"
    def _compact(m: "re.Match[str]") -> str:
        # group 3 (optional /unit) — preserve compound strengths
        suffix = f"/{m.group(3).lower()}" if m.group(3) else ""
        return f"{m.group(1)}{m.group(2).lower()}{suffix}"
    text = _STRENGTH_RE.sub(_compact, text)
    # 2. Split, filter, join. KHÔNG split trên "/" để giữ compound unit "100mg/ml".
    tokens = [t for t in re.split(r"[^a-z0-9/]+", text.lower()) if t]
    return " ".join(t for t in tokens if t not in skip_tokens)


def _parse_drug(text: str) -> tuple[str, str]:
    """Parse chuỗi thuốc → (ingredient, strength).

    Quy tắc:
    1. Strip route/freq → 'metoprolol 25 mg'.
    2. Tìm tất cả strength tokens (vd '25mg', '12.5mg/ml').
    3. Tất cả text còn lại trước strength → ingredient.
    4. Nếu có nhiều strength → join by ' / ' (compound).

    Returns:
        (ingredient_norm, strength_norm).
    """
    text = _strip_route_freq(text)
    if not text:
        return ("", "")

    # Ghép strength: lấy tất cả số+đơn vị theo thứ tự xuất hiện
    matches = list(_STRENGTH_RE.finditer(text))
    if not matches:
        # Không có strength → whole text = ingredient
        ingredient = re.sub(r"\s+", " ", text.lower()).strip()
        return (ingredient, "")

    # Lấy tất cả strength strings theo thứ tự, trùng lặp giữ
    strengths = [m.group(0) for m in matches]

    # Loại bỏ các token số+đơn vị ra khỏi text để lấy ingredient
    remaining = text
    for s in strengths:
        # Xoá cả match exact, kể cả viết thường/viết hoa
        remaining = re.sub(re.escape(s), " ", remaining, flags=re.IGNORECASE)

    # Ingredient = các từ còn lại (chữ cái + dấu nối)
    ingredient_tokens = re.findall(r"[a-z][a-z0-9-]+", remaining.lower())
    ingredient = " ".join(ingredient_tokens).strip()
    if not ingredient:
        return ("", "")

    # Normalize strengths (uppercase, no space)
    norm_strengths = [_normalize_strength(s) for s in strengths]
    norm_strengths = [s for s in norm_strengths if s]

    if len(norm_strengths) == 1:
        return (ingredient, norm_strengths[0])
    if len(norm_strengths) > 1:
        return (ingredient, " / ".join(norm_strengths))
    return (ingredient, "")


# ---------------------------------------------------------------------- #
# Index data structures
# ---------------------------------------------------------------------- #


@dataclass
class RxNormIndex:
    """Index tra cứu RxNorm từ data structured (ingredient, strength).

    Attributes:
        by_ingredient_strength: dict[(ing_norm, str_norm), list[rxcui]]
        by_ingredient: dict[ing_norm, list[rxcui]]  (chỉ ingredient)
        names: list tên gốc (parallel với rxcuis) — cho fuzzy
        rxcuis: list rxcui tương ứng
        name_to_idx: name -> idx
    """

    by_ingredient_strength: dict[tuple[str, str], list[str]] = field(default_factory=dict)
    by_ingredient: dict[str, list[str]] = field(default_factory=dict)
    names: list[str] = field(default_factory=list)
    rxcuis: list[str] = field(default_factory=list)
    name_to_idx: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "RxNormIndex":
        # Parse tuple keys từ "ingredient|strength" string format
        bis_raw = data.get("by_ingredient_strength", {})
        bis: dict[tuple[str, str], list[str]] = {}
        if isinstance(bis_raw, dict):
            # Old format: tuple keys (already parsed). New format: "ing|str" strings.
            for k, v in bis_raw.items():
                if isinstance(k, (list, tuple)):
                    bis[(k[0], k[1])] = v
                elif isinstance(k, str) and "|" in k:
                    parts = k.split("|", 1)
                    bis[(parts[0], parts[1])] = v
        by_ingredient = data.get("by_ingredient", {})
        if not isinstance(by_ingredient, dict):
            by_ingredient = {}
        idx = cls(
            by_ingredient_strength=bis,
            by_ingredient=by_ingredient,
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

    def add(self, rxcui: str, ingredient: str, strength: str,
            name: str = "") -> None:
        """Thêm 1 entry vào index."""
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
        """Tra cứu RxNorm cho một chuỗi thuốc. Trả top-1 rxcui (chuỗi rỗng nếu không match).

        Pipeline:
          L1: Exact (ingredient_norm, strength_norm) tuple match.
          L2: Compound — match 1 trong các strength con.
          L3: Ingredient-only exact match.
        """
        ing, strength = _parse_drug(drug_text)
        if not ing:
            return []

        # L1: Exact (ingredient, strength) tuple match
        if strength:
            cands = self.by_ingredient_strength.get((ing, strength), [])
            if cands:
                return [cands[0]]  # top-1

            # L2: Compound — strength có " / "
            if " / " in strength:
                for sub in strength.split(" / "):
                    sub = sub.strip()
                    if not sub:
                        continue
                    cands = self.by_ingredient_strength.get((ing, sub), [])
                    if cands:
                        return [cands[0]]

        # L3: Ingredient-only exact match
        cands = self.by_ingredient.get(ing, [])
        if cands:
            return [cands[0]]

        return []


# ---------------------------------------------------------------------- #
# Persistence
# ---------------------------------------------------------------------- #


def load_index(path: Optional[Path] = None) -> RxNormIndex:
    """Nạp index từ JSON. Trả RxNormIndex rỗng nếu không có file."""
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
    logger.info(
        "Saved → %s (%d names, %d ingredient+strength, %d ingredients)",
        path.name, len(idx.names), len(idx.by_ingredient_strength),
        len(idx.by_ingredient),
    )


# ---------------------------------------------------------------------- #
# High-level retriever
# ---------------------------------------------------------------------- #


class RxNormRetriever:
    """Wrapper đơn giản cho RxNormIndex.lookup(). Trả 1 rxcui duy nhất."""

    def __init__(self, index: Optional[RxNormIndex] = None,
                 index_path: Optional[Path] = None) -> None:
        if index is not None:
            self.index = index
        else:
            self.index = load_index(index_path)

    def lookup(self, drug_text: str) -> list[str]:
        """Tra RxNorm cho 1 chuỗi thuốc → list có 0 hoặc 1 rxcui."""
        return self.index.lookup(drug_text)


# ---------------------------------------------------------------------- #
# Build from JSONL dump
# ---------------------------------------------------------------------- #


def build_from_rxnorm_dump(dump_path: Path,
                           out_path: Optional[Path] = None) -> RxNormIndex:
    """Đọc JSONL [{rxcui, ingredient, strength, doseform, ...}] → build index."""
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