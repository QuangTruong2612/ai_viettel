"""RxNorm RAG — tra cứu mã RxNorm cho mỗi thực thể THUỐC.

Pipeline 4 lớp:
1. Exact match (case-insensitive) trong dict tra nhanh.
2. Normalized exact (bỏ dose form / route / freq).
3. Fuzzy match bằng rapidfuzz (WRatio + partial_ratio).
4. Live NIH RxNorm REST API (https://rxnav.nlm.nih.gov/REST/drugs.json?name=).

KHÔNG dùng embedding; chỉ exact + fuzzy + remote NIH API.

QUAN TRỌNG: Không bao giờ trả về code ngoài dict. Nếu không match thì [].
"""

from __future__ import annotations

import json
import logging
import re
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import requests  # type: ignore

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

RXNAV_BASE = "https://rxnav.nlm.nih.gov/REST"
# Cache các kết quả truy vấn RxNorm API theo tên (key đã normalize).
_RXNORM_API_CACHE_FILE = DATA_DIR / "rxnorm_api_cache.json"
_RXNORM_API_CACHE: dict[str, list[str]] = {}


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #


def _normalize(text: str) -> str:
    """Chuẩn hoá chuỗi: lowercase, bỏ diacritics, bỏ ký tự đặc biệt dư.

    Dùng để so khớp giữa tên thuốc tiếng Anh và phiên âm tiếng Việt.
    Ví dụ: "amlodipin" và "amlodipine" đều thành "amlodipin".

    Bug history:
    1. Trước kia split regex bao gồm `.` nên "0.5mg" bị split thành
       "0" + "5mg". Fix: dùng regex KHÔNG split dấu `.` để giữ "0.5mg"/"5.6mg".
    2. NFKD không strip được ký tự `đ` (U+0111 LATIN SMALL LETTER D WITH STROKE)
       vì không có decomposition → fix bằng replace thủ công trước NFKD.
    """
    text = text.lower().strip()
    # Xử lý đ/Đ trước (NFKD không strip được)
    text = text.replace("đ", "d").replace("Đ", "D")
    # Bỏ dấu tiếng Việt / Latin-1
    nfkd = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in nfkd if not unicodedata.combining(c))
    # Giữ liều lượng dạng "10mg", "0.5mg" làm 1 token để phân biệt strength
    text = re.sub(
        r"(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|iu|unit|%)",
        lambda m: f"{m.group(1)}{m.group(2)}",
        text,
    )
    # Bỏ pure-range số (vd "325-650" → "") để tránh nhiễu
    text = re.sub(r"\b\d+\s*[-–]\s*\d+\b", " ", text)
    # Bỏ các từ chỉ đường dùng / dạng bào chế / tần suất
    skip = {
        # dosage form / route
        "po", "iv", "im", "sc", "sl", "pr", "topical", "inhale",
        "oral", "tablet", "capsule", "solution", "suspension", "drop", "drops",
        "injection", "cream", "ointment", "gel", "patch", "spray", "powder",
        "liquid", "syrup", "granule", "lozenge", "film", "extended", "release",
        "xl", "xr", "er", "sr", "la", "cr", "mg", "mcg", "g", "ml", "iu",
        "unit", "dose", "strength", "tab", "cap", "inj", "amp", "vial",
        # frequency
        "daily", "bid", "tid", "qid", "qam", "qpm", "qhs", "q6h", "q8h", "q12h",
        "prn", "qd", "qod", "hs", "ac", "pc", "q4h",
    }
    # QUAN TRỌNG: regex split KHÔNG bao gồm `.` để giữ nguyên "0.5mg", "5.6mg"
    # còn sót lại sau khi regex mg replace (vd khi input là "0.5 mg/ml" thì giữ).
    tokens = [t for t in re.split(r"[^a-z0-9.]+", text) if t and t not in skip]
    return " ".join(tokens)


# ---------------------------------------------------------------------- #
# Index data structures
# ---------------------------------------------------------------------- #


@dataclass
class RxNormIndex:
    """Index tra cứu RxNorm.

    Attributes:
        exact: dict mapping normalized name → list[rxcui]
        names: list các tên gốc (cho fuzzy)
        rxcuis: list các rxcui tương ứng (parallel với names)
        name_to_idx: dict name string → idx trong names
    """

    exact: dict[str, list[str]] = field(default_factory=dict)
    names: list[str] = field(default_factory=list)
    rxcuis: list[str] = field(default_factory=list)
    name_to_idx: dict[str, int] = field(default_factory=dict)

    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        return {
            "exact": self.exact,
            "names": self.names,
            "rxcuis": self.rxcuis,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RxNormIndex":
        idx = cls(
            exact=data.get("exact", {}),
            names=data.get("names", []),
            rxcuis=data.get("rxcuis", []),
        )
        idx.name_to_idx = {n: i for i, n in enumerate(idx.names)}
        return idx

    # ------------------------------------------------------------------ #

    def add(self, name: str, rxcui: str, term_type: str = "") -> None:
        """Thêm 1 entry vào index.

        - name: tên thuốc (vd "Amlodipine 10 MG Oral Tablet")
        - rxcui: mã RxNorm
        - term_type: SCD/SBD/IN/BN/PIN/PSN — dùng để ưu tiên SCD > SBD > IN
        """
        key = name.lower().strip()
        self.exact.setdefault(key, []).append(rxcui)

        norm = _normalize(name)
        if norm:
            self.exact.setdefault(norm, []).append(rxcui)

        if name not in self.name_to_idx:
            self.name_to_idx[name] = len(self.names)
            self.names.append(name)
            self.rxcuis.append(rxcui)

    # ------------------------------------------------------------------ #

    def lookup(
        self,
        drug_text: str,
        *,
        fuzzy_threshold: int = 85,
    ) -> list[str]:
        """Tra cứu RxNorm cho một chuỗi thuốc. Trả list rxcui duy nhất, sorted."""
        text = drug_text.strip()
        if not text:
            return []

        # ----- 1. Exact (case-insensitive) -----
        key = text.lower()
        if key in self.exact:
            return sorted(set(self.exact[key]))

        # ----- 2. Normalized exact -----
        norm_text = _normalize(text)
        if norm_text in self.exact:
            return sorted(set(self.exact[norm_text]))

        # ----- 3. Fuzzy match trên toàn bộ names -----
        candidates = self._fuzzy_match(text, threshold=fuzzy_threshold)
        if candidates:
            return sorted(set(candidates))

        # ----- 4. Live NIH RxNorm REST API (drugs.json?name=) -----
        api_cands = _http_rxnorm_search(text, norm_text=norm_text)
        if api_cands:
            return sorted(set(api_cands))

        return []

    # ------------------------------------------------------------------ #

    def _fuzzy_match(self, text: str, *, threshold: int) -> list[str]:
        try:
            from rapidfuzz import fuzz, process  # type: ignore
        except ImportError:  # pragma: no cover
            logger.warning("rapidfuzz chưa cài — bỏ qua fuzzy match")
            return []

        # Tách các "từ thuốc" (loại bỏ số, đơn vị liều, route, freq)
        def _drug_tokens(s: str) -> list[str]:
            import re as _re

            # Bỏ đơn vị liều, đường dùng, tần suất
            pat = (
                r"\d+(\.\d+)?\s*(mg|mcg|g|ml|iu|unit|%)"
                r"|\bpo\b|\biv\b|\bim\b|\bsc\b|\bsl\b"
                r"|\bprn:?\b|\bdaily\b|\bbid\b|\btid\b|\bqid\b|\bqhs?\b"
                r"|\bq6h\b|\bq8h\b|\bq12h\b|\bqam\b|\bqpm\b"
                r"|\bqd\b|\bqod\b|\bhs\b|\bac\b|\bpc\b"
            )
            s2 = _re.sub(pat, " ", s.lower())
            tokens = [t for t in _re.split(r"[^a-z]+", s2) if t]
            return tokens

        # Dùng cả "từ thuốc" đầu tiên và toàn bộ tokens để thử fuzzy
        candidates_to_try = [text, _drug_tokens(text)]

        out: list[str] = []
        for cand in candidates_to_try:
            if not cand:
                continue
            query = cand if isinstance(cand, str) else " ".join(cand)
            if not query:
                continue
            # WRatio và partial_ratio kết hợp: WRatio cho cả chuỗi,
            # partial_ratio để bắt substring như "nystatin" trong "Nystatin Oral Suspension"
            matches_wr = process.extract(
                query, self.names, scorer=fuzz.WRatio, limit=5
            )
            matches_pr = process.extract(
                query, self.names, scorer=fuzz.partial_ratio, limit=5
            )
            seen_names: set[str] = set()
            merged = []
            for name, score, _ in matches_wr + matches_pr:
                if name in seen_names:
                    continue
                seen_names.add(name)
                merged.append((name, score))

            for name, score in merged:
                # partial_ratio thường cao; dùng max với WRatio
                if score < threshold:
                    continue
                # Lấy rxcui bằng tên
                if name not in self.name_to_idx:
                    continue
                rxcui = self.rxcuis[self.name_to_idx[name]]
                if rxcui not in out:
                    out.append(rxcui)
            if out:
                break  # đã có kết quả từ candidate đầu tiên khả thi
        return out




    # ------------------------------------------------------------------ #
    # (No-op: embedding cache removed — chỉ dùng NIH API + local fuzzy.)
    # ------------------------------------------------------------------ #

# ---------------------------------------------------------------------- #
# Live NIH RxNorm API (drugs.json?name=)
# ---------------------------------------------------------------------- #


def load_index(path: Optional[Path] = None) -> RxNormIndex:
    """Nạp index từ JSON; nếu chưa có thì trả về index rỗng."""
    path = path or (DATA_DIR / "rxnorm_index.json")
    if not path.exists():
        return RxNormIndex()
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return RxNormIndex.from_dict(data)


def save_index(idx: RxNormIndex, path: Optional[Path] = None) -> None:
    path = path or (DATA_DIR / "rxnorm_index.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(idx.to_dict(), f, ensure_ascii=False, indent=2)
    logger.info("Đã lưu index → %s (%d names)", path, len(idx.names))


def _drug_query_tokens(text: str) -> list[str]:
    """Tách phần 'tên thuốc' (bỏ dose, route, freq, intake notation) để query API.

    Bug history: trước kia filter thiếu "x N" intake pattern (vd "aspirin 325mg x 1"),
    làm NIH API không khớp trả về [] cho cùng drug ở format tiêu chuẩn.
    Fix: thêm `\\bx \\d+\\b` (x 1, x 2, x N), `\\bviên\\b`, `\\bống\\b`, `\\blần\\b` vào skip.
    """
    pat = (
        r"\d+(\.\d+)?\s*(mg|mcg|g|ml|iu|unit|%)"
        # Intake / quantity notation tiếng Việt: "x 1 viên", "uống 2 lần/ngày"
        r"|\bx\s*\d+\b"  # "x 1", "x 2", "x 10"
        r"|\b\d+\s*vi[eê]n\b"  # "1 viên", "2 viên"
        r"|\b\d+\s*(ống|lần|giọt|gói|viên)\b"
        r"|\bpo\b|\biv\b|\bim\b|\bsc\b|\bsl\b"
        r"|\bprn:?\b|\bdaily\b|\bbid\b|\btid\b|\bqid\b|\bqhs?\b"
        r"|q6h|q8h|q12h|qam|qpm"
        r"|\b(?:ngày|giờ|tuần|tháng)\b"  # đơn vị thời gian uống
        r"|\bu[oố]ng\b|\ban\b|\bu[oố]ng\b|\blần\b|\bvi[eê]n\b"
        r"|\bqd\b|\bqod\b|\bhs\b|\bac\b|\bpc\b"
        r"|\buống\b|\bti[eê]m\b|\bchích\b|\btruy[eề]n\b|\bxông\b|\bkh[íi] dung\b"
        r"|\boral tablet\b|\btablet\b|\boral capsule\b|\bcapsule\b"
        r"|\bsolution\b|\bsuspension\b|\binjection\b|\bpatch\b"
    )
    s2 = re.sub(pat, " ", text.lower())
    s2 = s2.replace("đ", "d").replace("Đ", "D")
    nfkd = unicodedata.normalize("NFKD", s2)
    s2 = "".join(c for c in nfkd if not unicodedata.combining(c))
    tokens = [t for t in re.split(r"[^a-z0-9]+", s2) if t]
    return tokens


def _http_rxnorm_search(
    drug_text: str,
    *,
    norm_text: str = "",
    timeout: int = 20,
    max_results: int = 10,
) -> list[str]:
    """Gọi NIH RxNorm REST API /REST/drugs.json?name=<drug>; trả list[rxcui] sorted."""
    if not drug_text:
        return []

    # Tách tên thuốc để gọi API
    tokens = _drug_query_tokens(drug_text)
    if not tokens:
        return []
    query_tokens = tokens[:3]
    cache_key = " ".join(query_tokens)

    # Cache check
    if cache_key in _RXNORM_API_CACHE:
        return list(_RXNORM_API_CACHE[cache_key])

    # Trích strength để filter
    strength_in_input = ""
    m = re.search(r"(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|iu|unit|%)", drug_text.lower())
    if m:
        strength_in_input = f"{m.group(1)}{m.group(2)}"

    api_query = " ".join(query_tokens)
    try:
        r = requests.get(
            f"{RXNAV_BASE}/drugs.json",
            params={"name": api_query},
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        logger.warning("RxNorm API fail (%r): %s", api_query, exc)
        return []
    time.sleep(0.05)

    group = data.get("drugGroup", {}) or {}
    concept_groups = group.get("conceptGroup", []) or []

    def _norm_for_match(s: str) -> str:
        return s.lower().replace(" ", "").replace("\t", "")

    priority_tty = ["SCD", "SBD", "BPCK", "GPCK", "SCDF", "SBDF", "IN", "PIN", "MIN", "SCDC"]
    by_tty: dict[str, list[tuple[str, str, str]]] = {}
    for cg in concept_groups:
        tty = cg.get("tty", "")
        for c in cg.get("conceptProperties", []) or []:
            rxcui = str(c.get("rxcui", "")).strip()
            name = str(c.get("name", "")).strip()
            syn = str(c.get("synonym", "")).strip()
            if rxcui and name:
                by_tty.setdefault(tty, []).append((rxcui, name, syn))

    out: list[str] = []
    for tty in priority_tty:
        concepts = by_tty.get(tty, [])
        if not concepts:
            continue
        if strength_in_input:
            standalone_match = [
                (rxcui, name)
                for rxcui, name, syn in concepts
                if "/" not in name
                and (strength_in_input in _norm_for_match(name)
                     or strength_in_input in _norm_for_match(syn))
            ]
            combo_match = [
                (rxcui, name)
                for rxcui, name, syn in concepts
                if "/" in name
                and (strength_in_input in _norm_for_match(name)
                     or strength_in_input in _norm_for_match(syn))
            ]
            any_match = [
                (rxcui, name)
                for rxcui, name, syn in concepts
                if strength_in_input in _norm_for_match(name)
                or strength_in_input in _norm_for_match(syn)
            ]
            chosen = standalone_match or combo_match or any_match
            for rxcui, _ in chosen:
                if rxcui not in out:
                    out.append(rxcui)
            if out:
                break
        standalone = [r for r, _, _ in concepts if "/" not in _]
        combos = [r for r, _, _ in concepts if "/" in _]
        for rxcui in (standalone + combos)[:max_results]:
            if rxcui not in out:
                out.append(rxcui)
        if out:
            break

    _RXNORM_API_CACHE[cache_key] = sorted(set(out))
    return sorted(set(out))


def save_rxnorm_cache() -> None:
    """Lưu cache NIH API ra đĩa."""
    _RXNORM_API_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _RXNORM_API_CACHE_FILE.write_text(
        json.dumps(_RXNORM_API_CACHE, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------- #
# High-level retriever
# ---------------------------------------------------------------------- #


class RxNormRetriever:
    """Wrapper đơn giản giữ RxNormIndex trong bộ nhớ.

    KHÔNG dùng embedding; chỉ gọi `index.lookup()` (đã có exact + fuzzy + NIH API).
    """

    def __init__(self, index_path: Optional[Path] = None) -> None:
        self.index = load_index(index_path)

    def lookup(self, drug_text: str) -> list[str]:
        """Tra RxNorm cho 1 chuỗi thuốc."""
        return self.index.lookup(drug_text)


# ---------------------------------------------------------------------- #
# CLI: build index từ RxNorm JSON dump
# ---------------------------------------------------------------------- #


def build_from_rxnorm_dump(dump_path: Path, out_path: Optional[Path] = None) -> RxNormIndex:
    """Đọc file JSON dạng [{rxcui, name, term_type}, ...] và dựng index."""
    with dump_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    idx = RxNormIndex()
    for row in rows:
        rxcui = str(row.get("rxcui", "")).strip()
        name = str(row.get("name", "")).strip()
        ttype = str(row.get("term_type", "")).strip()
        if rxcui and name:
            idx.add(name, rxcui, ttype)
    save_index(idx, out_path)
    return idx


if __name__ == "__main__":  # pragma: no cover
    import sys

    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) >= 2:
        idx = build_from_rxnorm_dump(Path(sys.argv[1]))
        print(f"Index có {len(idx.names)} names, {len(idx.exact)} exact keys")
    else:
        idx = load_index()
        print(f"Loaded index: {len(idx.names)} names, {len(idx.exact)} exact keys")
