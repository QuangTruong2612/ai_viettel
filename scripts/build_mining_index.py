"""Build mining index từ icd10.jsonl và rxnorm.jsonl.

Generates:
- data/icd_aliases.json: code → [desc_vi + bracket aliases + paren aliases]
- data/drug_inn_cache.json: sorted list of unique INN/generic names từ rxnorm.jsonl
- data/drug_brand_seed.json: suggested brand→generic mappings (gợi ý để user review)

Run:
    python scripts/build_mining_index.py

Idempotent — re-run anytime data thay đổi. Không modify runtime code, chỉ sinh
derived data files cho ICDRetriever + RxNormRetriever consume.
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

# Ưu tiên: src/ chứa helper import (nếu cần)
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

# ════════════════════════════════════════════════════════════════════════════════
# Alias-mining logic — mirror scripts/build_icd_index.py::_mine_vi_aliases
# Nếu logic thay đổi 1 chỗ, sync the other.
# ════════════════════════════════════════════════════════════════════════════════

_MIN_ALIAS_LEN = 3
_MAX_ALIAS_LEN = 80
_NON_NAME_PATTERN = re.compile(r"^[\d.,%/*\s\-]+$")
_BRACKET_STOPWORDS = frozenset({
    "xem", "xem thêm", "xem chú thích", "chú thích",
    "draft", "draft only", "deprecated", "xóa", "xóa bỏ",
    "mới", "cũ", "xác định", "tạm thời",
})


def _mine_vi_aliases(name: str) -> list[str]:
    if not name:
        return []
    aliases: list[str] = [name]
    for m in re.finditer(r"\[([^\]]+)\]", name):
        inner = m.group(1)
        pieces = re.split(r"[,;]|hoặc|và/hoặc", inner)
        for piece in pieces:
            piece = piece.strip()
            if not piece or piece.lower() in {"và"}:
                continue
            piece = re.sub(r"^và\s+", "", piece).strip()
            if (
                _MIN_ALIAS_LEN <= len(piece) <= _MAX_ALIAS_LEN
                and not _NON_NAME_PATTERN.match(piece)
                and piece.lower() not in _BRACKET_STOPWORDS
                and not piece.startswith(("http", "www", "xem"))
            ):
                aliases.append(piece)
    for m in re.finditer(r"\(([^)]+)\)", name):
        inner = m.group(1).strip()
        if (
            _MIN_ALIAS_LEN <= len(inner) <= _MAX_ALIAS_LEN
            and not _NON_NAME_PATTERN.match(inner)
            and not re.match(r"^[A-Z]\d{2}(\.\d+)?\*?$", inner)
            and not re.match(r"^(cấp|mạn|cấp tính|mạn tính|nặng|nhẹ)$", inner, re.IGNORECASE)
            and inner.lower() not in _BRACKET_STOPWORDS
        ):
            aliases.append(inner)
    return list(dict.fromkeys(aliases))


# ════════════════════════════════════════════════════════════════════════════════
# 1. Mine ICD aliases: {code: [aliases]}
# ════════════════════════════════════════════════════════════════════════════════

def mine_icd_aliases(icd_jsonl: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    if not icd_jsonl.exists():
        print(f"[WARN] {icd_jsonl} not found, skipping ICD mining")
        return out
    t0 = time.time()
    n_codes = 0
    n_aliases_raw = 0

    # Pass 1: mine thô (giữ nguyên logic cũ)
    raw: dict[str, list[str]] = {}
    with icd_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            code = str(row.get("code", "")).strip()
            desc_vi = str(row.get("desc_vi", "")).strip()
            if not code or not desc_vi:
                continue
            aliases = _mine_vi_aliases(desc_vi)
            raw[code] = aliases
            n_codes += 1
            n_aliases_raw += len(aliases)

    # Pass 2: TỰ LỌC COLLISION — loại bỏ alias bị dùng chung bởi >=2 mã KHÔNG liên quan.
    # Lý do: alias thật (đồng nghĩa đúng nghĩa) hầu như luôn gắn riêng với 1 mã (hoặc các
    # subcode CÙNG gốc, vd J18/J18.0/J18.9). Nếu 1 cụm từ (mined từ ngoặc) xuất hiện dùng
    # chung cho NHIỀU mã thuộc các chương/nhóm bệnh khác nhau (vd "mắc phải", "các", "lồng",
    # "nguyên phát" — đã phát hiện thực tế dùng chung tới 34-59 mã không liên quan), đó gần
    # như chắc chắn là qualifier/cross-reference chung chung, KHÔNG phải alias thật — cần bỏ
    # để tránh 1 alias trả về hàng chục candidate vô nghĩa (đã verify bằng data thật, không
    # phải giả thuyết).
    #
    # Ngoại lệ: nếu tất cả các mã dùng chung alias đó đều CÙNG "gốc" (3 ký tự đầu mã ICD, vd
    # J18 và J18.9 đều bắt đầu "J18") → được coi là hợp lệ (subcode của cùng 1 bệnh), không bị lọc.
    alias_to_codes: dict[str, set[str]] = defaultdict(set)
    for code, aliases in raw.items():
        for alias in aliases:
            ak = alias.strip().lower()
            if ak and ak != code.lower():  # bỏ qua chính desc_vi gốc (luôn giữ)
                alias_to_codes[ak].add(code)

    def _same_family(codes: set[str]) -> bool:
        roots = {c.split(".")[0][:3] for c in codes}
        return len(roots) <= 1

    noisy_aliases = {
        ak for ak, codes in alias_to_codes.items()
        if len(codes) >= 2 and not _same_family(codes)
    }
    if noisy_aliases:
        print(f"[ICD] Lọc bỏ {len(noisy_aliases)} alias bị dùng chung bởi nhiều mã "
              f"không liên quan (nghi vấn qualifier/cross-ref gây nhiễu), vd: "
              f"{sorted(noisy_aliases, key=lambda a: -len(alias_to_codes[a]))[:5]}")

    n_aliases_kept = 0
    for code, aliases in raw.items():
        kept: list[str] = []
        for i, a in enumerate(aliases):
            if i == 0:
                kept.append(a)  # desc_vi gốc — LUÔN giữ, không bao giờ bị lọc collision
                continue
            if a.strip().lower() not in noisy_aliases:
                kept.append(a)
        out[code] = kept
        n_aliases_kept += len(kept)

    elapsed = time.time() - t0
    print(f"[ICD] Mined {n_aliases_raw} aliases thô → giữ {n_aliases_kept} sau lọc collision, "
          f"từ {n_codes} codes ({elapsed:.1f}s)")
    return out



# ════════════════════════════════════════════════════════════════════════════════
# 2. Mine RxNorm INN whitelist: sorted unique ingredient names
# ════════════════════════════════════════════════════════════════════════════════

def mine_rxnorm_inn(rxnorm_jsonl: Path) -> set[str]:
    """Trả về set of unique INN (generic) names từ rxnorm.jsonl.

    Dùng để build drug_inn_cache.json. Cache file làm cho runtime load O(1).
    """
    if not rxnorm_jsonl.exists():
        print(f"[WARN] {rxnorm_jsonl} not found, skipping RxNorm mining")
        return set()
    t0 = time.time()
    out: set[str] = set()
    n_rows = 0
    with rxnorm_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_rows += 1
            ing = str(row.get("ingredient", "")).strip().lower()
            if ing and len(ing) >= 3:
                out.add(ing)
    elapsed = time.time() - t0
    print(f"[RxNorm] Mined {len(out)} unique INN từ {n_rows} entries ({elapsed:.1f}s)")
    return out


# ════════════════════════════════════════════════════════════════════════════════
# 3. Mine drug brand-name suggestions (heuristic — cần manual review)
# ════════════════════════════════════════════════════════════════════════════════

# Common VN drug-class / generic prefixes (KHÔNG phải brand — chỉ là gợi ý để tránh false positive)
_NON_BRAND_PREFIXES = frozenset({
    "thuốc", "viên", "thuốc viên", "thuốc uống", "thuốc tiêm",
    "viên nén", "viên nang", "viên sủi", "thuốc bột", "thuốc nước",
    "thuốc nhỏ", "dung dịch", "hỗn dịch", "siro", "gói", "ống", "lọ",
})


def _extract_drug_contexts(text: str) -> list[tuple[int, int]]:
    """Tìm các đoạn text có khả năng chứa drug name (drug contexts).

    Chiến lược: regex match các "anchor phrases" trước/sau drug listings trong VN records.
    Trả về list (start, end) byte spans của các contexts.

    Patterns thuộc 1 trong 3 classes:
    1. Section headers: "Thuốc trước khi nhập viện:", "Thuốc đang dùng:", "Điều trị:", "Kê đơn:"
    2. Inline mentions: "uống \\w+", "tiêm \\w+", "truyền \\w+", "khởi dùng \\w+", etc.
    3. Asset lists: bullet lines starting with "- " hoặc "• " containing drug-like tokens
    """
    contexts: list[tuple[int, int]] = []
    # Class 1: section headers — lấy ~500 chars sau mỗi header
    section_pat = re.compile(
        r"(?:thuốc\s+(?:trước\s+khi\s+)?(?:nhập\s+viện|đang\s+dùng|ra\s+viện|trước\s+đây|trước\s+nhập\s+viện)|"
        r"điều\s+trị|kê\s+đơn|chỉ\s+định(?:.*?điều\s+trị)?|"
        r"đang\s+sử\s+dụng|tiền\s+sử\s+dùng\s+thuốc)[:\.]?\s*",
        re.IGNORECASE | re.UNICODE,
    )
    for m in section_pat.finditer(text):
        contexts.append((m.end(), min(len(text), m.end() + 500)))
    # Class 2 + 3: bullet lines thường chứa drug names
    bullet_pat = re.compile(r"^\s*[-•*]\s*(.+?)$", re.MULTILINE | re.UNICODE)
    for m in bullet_pat.finditer(text):
        line = m.group(1).strip()
        # Bullet line phải có drug-like signal: suffix brand, dose unit, hoặc route verb
        line_lower = line.lower()
        has_drug_signal = (
            bool(re.search(r"\d+\s*(?:mg|mcg|g|ml|iu|viên|ống|gói)", line_lower))
            or bool(re.search(
                r"\b\w*(?:mab|lol|pin|zole|pril|sartan|statin|oxacin|mycin|vir|ximab|olol|ipine|oxime|navir|navir|navir)\b",
                line_lower,
            ))
            or bool(re.search(
                r"\b(?:uống|tiêm|truyền|hít|dùng|ngậm|bôi|tiêm\s+truyền|đặt|uống\s+trước|uống\s+sau)\b",
                line_lower,
            ))
        )
        if has_drug_signal:
            contexts.append((m.start(1), m.end(1)))
    return contexts


# Drug-name suffix/affix patterns (không phải hardcode list brand, mà là REGEX pattern)
# Đây là pharmacological convention: tất cả "-lol" = beta-blocker, "-pril" = ACEi, etc.
_DRUG_AFFIX_PATTERNS = re.compile(
    r"(?:"
    r"mab|lol|pin|zole|pril|sartan|statin|oxacin|mycin|"
    r"vir|ximab|navir|ciclovir|olol|ipine|oxime|ximab|"
    r"cillin|cefazolin|cefalexin|cefepim|ximab|sulfo|"
    r"caine|pam|pine|dopa|oxin|thasone"
    r")$",
    re.IGNORECASE,
)

# Common VN stopwords/adjectives/verbs KHÔNG bao giờ là brand name.
# Bổ sung từ kinh nghiệm — KHÔNG đầy đủ (chỉ high-frequency false positives).
_VN_STOPWORD_BLACKLIST = frozenset({
    "không", "triệu", "trong", "chiếu", "nhiều", "truyền", "chuyển",
    "khoảng", "trước", "nhiễm", "thuyên", "nhiệt", "huyết", "ngoại",
    "người", "thường", "ngừng", "thỉnh", "trung", "quyết", "nghẹt",
    "ngoài", "tiếng", "nghiêm", "nghiệm", "nghẽn", "thoát", "nguyên",
    "thông", "thiếu", "ngưng", "dùng", "theo", "trên", "dưới", "thuốc",
    "viên", "bệnh", "nhân", "tháng", "tuần", "ngày", "năm", "giờ",
    "bệnh", "phút", "giây", "lần", "đầu", "cuối", "cùng", "bên",
    "trái", "phải", "trước", "sau", "trong", "ngoài", "cách", "khoảng",
    "đến", "từ", "qua", "lên", "xuống", "vào", "ra", "lại",
})


def suggest_drug_brands(input_dir: Path, known_generics: set[str]) -> dict[str, str]:
    """R28 v2: Mine drug brand suggestions với CONTEXT-AWARE filtering.

    Yêu cầu TẤT CẢ conditions:
    1. Token phải nằm trong drug context (drug sections / bullet lines có drug signals)
    2. Token không trong known_generics (đã có rồi)
    3. Token không trong _NON_BRAND_PREFIXES (form, route, freq tokens)
    4. Token KHÔNG trong _VN_STOPWORD_BLACKLIST (common VN stopwords)
    5. Token match drug affix pattern HOẶC viết hoa đầu (TitleCase brand)
    6. Token length 5-22 chars
    7. Token phải appear ≥ 2 lần (single = noise)
    8. Token không phải số, không phải URL
    """
    if not input_dir.exists():
        return {}

    # Step 1: Extract drug contexts from all files
    candidate_freq: dict[str, int] = defaultdict(int)
    candidate_files: dict[str, set[str]] = defaultdict(set)

    for txt_path in sorted(input_dir.glob("*.txt")):
        try:
            text = txt_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        # Focus on drug contexts (sections + drug bullet lines)
        drug_spans = _extract_drug_contexts(text)
        for span_start, span_end in drug_spans:
            span_text = text[span_start:span_end]
            # Extract tokens trong drug context only
            for m in re.finditer(r"\b[A-Za-zÀ-ỹ][a-zà-ỹA-ZÀ-Ỹ\-]{4,21}\b", span_text):
                tok = m.group(0)
                tl = tok.lower()
                # Filters (1-4, 6, 8)
                if tl in known_generics:
                    continue
                if tl in _NON_BRAND_PREFIXES or tl in _VN_STOPWORD_BLACKLIST:
                    continue
                if len(tl) < 5 or len(tl) > 22:
                    continue
                # Filter (5): affix match OR TitleCase
                has_affix = bool(_DRUG_AFFIX_PATTERNS.search(tl))
                is_titlecase = tok[0].isupper() and not tok.isupper()  # First cap, not all caps
                if not (has_affix or is_titlecase):
                    continue
                candidate_freq[tl] += 1
                candidate_files[tl].add(txt_path.name)

    # Step 2: Filter (7) — must appear in ≥ 2 contexts (across multiple files or many times)
    qualified = {
        k: v for k, v in candidate_freq.items()
        if v >= 2 or len(candidate_files[k]) >= 2
    }

    # Step 3: Sort by frequency DESC, take top-100
    sorted_candidates = sorted(qualified.items(), key=lambda x: -x[1])[:100]

    # Step 4: Output with confidence hint
    out: dict[str, str] = {}
    for token, freq in sorted_candidates:
        # Confidence hint: files_with_token vs total freq
        n_files = len(candidate_files[token])
        if n_files >= 3 or freq >= 5:
            out[token] = "? ← likely brand"   # High confidence
        elif n_files >= 2 or freq >= 3:
            out[token] = "? ← maybe brand"
        else:
            out[token] = "?"                   # Borderline
    return out


# ════════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════════

def main() -> int:
    data_dir = PROJECT_DIR / "data"
    input_dir = PROJECT_DIR / "input"

    # 1. ICD aliases (code → [aliases])
    icd_path = data_dir / "icd10.jsonl"
    icd_aliases = mine_icd_aliases(icd_path)
    if icd_aliases:
        out_path = data_dir / "icd_aliases.json"
        out_path.write_text(
            json.dumps(icd_aliases, ensure_ascii=False, indent=0),
            encoding="utf-8",
        )
        print(f"[ICD] → {out_path} ({len(icd_aliases)} codes, "
              f"{sum(len(v) for v in icd_aliases.values())} alias entries)")

    # 2. RxNorm INN cache
    rxnorm_path = data_dir / "rxnorm.jsonl"
    inn_set = mine_rxnorm_inn(rxnorm_path)
    if inn_set:
        out_path = data_dir / "drug_inn_cache.json"
        out_path.write_text(
            json.dumps(sorted(inn_set), ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[RxNorm INN] → {out_path} ({len(inn_set)} unique INN)")

    # 3. Drug brand suggestions (review-only, không modify drug_aliases.json)
    if rxnorm_path.exists():
        brand_suggest = suggest_drug_brands(input_dir, inn_set)
        if brand_suggest:
            out_path = data_dir / "drug_brand_seed.json"
            out_path.write_text(
                json.dumps(brand_suggest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"[Drug brands] → {out_path} ({len(brand_suggest)} candidates — REVIEW cần thiết)")

    print("\nDone. Re-run anytime `data/icd10.jsonl` hoặc `data/rxnorm.jsonl` thay đổi.")
    return 0


if __name__ == "__main__":
    sys.exit(main())