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

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import requests

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

    def __init__(self, llm_client: Any = None, cache_path: Optional[Path] = None) -> None:
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
            "Translate the following Vietnamese medical term into a concise English "
            "expression used in clinical documentation. Keep it 1-4 words. "
            "Reply with ONLY the English phrase, no quotes, no explanation.\n\n"
            f"Vietnamese: {text}\nEnglish:"
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


def _http_search(query: str, max_results: int = 8, timeout: int = 15) -> list[tuple[str, str]]:
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


class ICDRetriever:
    """High-level wrapper: kết hợp local index + NIH API + translation.

    KHÔNG dùng embedding; chỉ exact + fuzzy + remote NIH API.
    """

    def __init__(
        self,
        index_path: Optional[Path] = None,
        translator: Optional[Translator] = None,
        use_remote: bool = True,
        remote_cache_path: Optional[Path] = None,
    ) -> None:
        self.idx = self._load_index(index_path)
        self.translator = translator or Translator()
        self.use_remote = use_remote
        self._remote_cache_path = remote_cache_path or (DATA_DIR / "icd_remote_cache.json")
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

    def lookup(self, vn_text: str) -> list[str]:
        """Tra ICD-10 cho 1 cụm chẩn đoán tiếng Việt.

        Returns: list codes (string), unique + sorted.

        Bug history:
        1. Trước kia `_looks_vn()` bỏ sót từ không dấu ("suy tim", "ho"). Fix: luôn gọi translate().
        2. Trước kia input "bệnh nhân bị tăng huyết áp" hoặc "chẩn đoán: viêm phổi"
           không strip prefix → fuzzy fail với seed EN. Fix: strip clinical prefixes.
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

        # L3: Local fuzzy match trên names với EN query (low threshold để bắt
        # substring như "Insomnia" trong "Insomnia, unspecified")
        fuzzy_en = self._fuzzy_local(en_query, threshold=70)
        if fuzzy_en:
            return fuzzy_en

        # L4: Fuzzy cả với VN text (cho term không dịch được / giống EN)
        fuzzy_vn = self._fuzzy_local(text, threshold=85)
        if fuzzy_vn:
            return fuzzy_vn

        # L5: Remote NIH (cache + dedupe)
        if self.use_remote:
            cache_key = en_key
            if cache_key in self._remote_cache:
                results = self._remote_cache[cache_key]
                return sorted(set(results))

            results = _http_search(en_query, max_results=8)
            self._remote_cache[cache_key] = [r[0] for r in results]
            for code, name in results:
                self.idx.add(code, name)
            return sorted({r[0] for r in results})

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
            "the", "of", "and", "or", "with", "without", "in", "to", "a", "an",
            "unspecified", "type", "stage", "acute", "chronic", "primary", "secondary",
        }
        tokens = [t for t in re.split(r"[^a-z]+", query.lower()) if t and t not in stop]
        if not tokens:
            return []
        q = " ".join(tokens)

        matches = process.extract(q, self.idx.names, scorer=fuzz.WRatio, limit=5)
        matches += process.extract(q, self.idx.names, scorer=fuzz.partial_ratio, limit=5)

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
            if ch in "ăâđêôơưĂÂĐÊÔƠƯáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵÁÀẢÃẠẮẰẲẴẶẤẦẨẪẬÉÈẺẼẸẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤỨỪỬỮỰÝỲỶỸỴ":
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
