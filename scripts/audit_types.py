"""Audit 100 output JSON cũ: tìm mọi case entity có type NGHI SAI so với bản chất y khoa.

Công thức chấm:
    final = 0.3·text + 0.3·assertions + 0.4·candidates
Sai TYPE (đúng span, đúng text, sai class) = trừ điểm ở CẢ 3 metric, NHÂN ĐÔI
(thừa 1 + thiếu 1) — tốn kém nhất trong toàn bộ scoring.

Script này quét không cần gold:
- Heuristic 1: text trùng whitelist thuốc → type phải là THUỐC (sai → nghi).
- Heuristic 2: text trùng pattern procedure/test → type phải là TÊN_XÉT_NGHIỆM.
- Heuristic 3: text là cụm bệnh danh phổ biến (viêm X, suy X, ung thư X, ...) nhưng
  type ≠ CHẨN_ĐOÁN.
- Heuristic 4: text giống kết quả định lượng (có số + đơn vị) mà type ≠ KQ_XN.
- Heuristic 5: cùng text xuất hiện trong cùng file nhưng gán type khác nhau.
- Heuristic 6: span overlap trong cùng file (dấu hiệu boundary sai).

Usage:
    python scripts/audit_types.py               # mặc định quét output/*.json
    python scripts/audit_types.py --limit 10    # chỉ quét 10 file đầu
    python scripts/audit_types.py --top 50      # top 50 suspect cases mỗi loại
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

# Đảm bảo có thể chạy trực tiếp mà không cần `python -m`
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.postprocess import (  # noqa: E402
    _PROC_PATTERNS,
    _PROC_EXCLUDE_CONTAINS,
    _PROC_EXCLUDE_STARTS,
    _DRUG_NAMES_UNIONED,
    _is_procedure,
)

# ════════════════════════════════════════════════════════════════════════════════
# Load data-driven whitelists
# ════════════════════════════════════════════════════════════════════════════════

DATA_DIR = _PROJECT_ROOT / "data"


def _load_jsonl_or_json(path: Path) -> Any:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return json.loads(path.read_text(encoding="utf-8"))


def _load_drug_set() -> set[str]:
    """Union: drug_inn_cache (63k) + common_drug_names (179) + drug_aliases keys + drug_brand_seed keys."""
    s: set[str] = set(_DRUG_NAMES_UNIONED)
    p = DATA_DIR / "drug_aliases.json"
    if p.exists():
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            s.update(str(k).lower().strip() for k in d.keys())
            for v in d.values():
                # value là 'aspirin / caffeine' → vẫn lấy token đầu làm canonical
                if v and "?" not in v and "/" not in v:
                    s.add(str(v).lower().strip())
        except Exception:
            pass
    p = DATA_DIR / "common_drug_names.json"
    if p.exists():
        try:
            arr = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(arr, list):
                s.update(str(x).lower().strip() for x in arr)
        except Exception:
            pass
    p = DATA_DIR / "drug_brand_seed.json"
    if p.exists():
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            for k in d.keys():
                if not k.startswith("?"):
                    s.add(str(k).lower().strip())
        except Exception:
            pass
    return s


_DRUG_NAMES: set[str] = _load_drug_set()
print(f"[audit] Loaded {len(_DRUG_NAMES)} drug names", file=sys.stderr)


# ════════════════════════════════════════════════════════════════════════════════
# Heuristic patterns
# ════════════════════════════════════════════════════════════════════════════════

# Procedure/test patterns (Vietnamese). Dùng lại _is_procedure() từ postprocess
# nhưng bổ sung vài pattern chỉ-tên (không có verb prefix) hay bị miss.
_TEST_NAME_KEYWORDS = [
    "X-quang", "xquang", "CT scan", "MRI", "siêu âm", "sieu am",
    "điện tâm đồ", "dien tam do", "ECG", "Holter", "monitor",
    "công thức máu", "nước tiểu", "phân tích", "xét nghiệm",
    "nội soi", "sinh thiết", "tế bào", "mô bệnh học",
    "PSA", "TSH", "WBC", "Hgb", "AST", "ALT", "GGT", "CRP",
]

# Disease name prefixes (CHẨN_ĐOÁN thường bắt đầu bằng các từ này)
_DISEASE_PREFIX_RE = re.compile(
    r"^(viêm|ung thư|suy|thoái hóa|rối loạn|hội chứng|bệnh|"
    r"tăng huyết áp|đái tháo đường|nhồi máu|xuất huyết|tràn dịch|"
    r"ngoại tâm thu|rung nhĩ|block|st chênh|suy hô hấp|suy tim|"
    r"suy thận|suy gan|nhiễm trùng|nhiễm khuẩn|tắc|"
    r"gãy|trật|thoát vị|phình|bướu|u nang|polyp|sỏi|"
    r"hen |COPD|CKD|THA\b|ĐTĐ|NMCT|RLLL|NMH)",
    re.IGNORECASE | re.UNICODE,
)

# Quantitative result pattern: số + đơn vị hoặc số/mmHg/°C/...
_KQ_QUANTITATIVE_RE = re.compile(
    r"^\d+([.,]\d+)?\s*(mmhg|cm|mm|kg|mg|g|ml|iu|u/l|g/dl|mg/dl|nmol/l|"
    r"pmol/l|µg|ug|ng|pg|k/ul|m/ul|fL|pg|%|độ|c|°c|celsius|lần/phút|"
    r"lần/phut|l/p|lần|tuần|tháng|năm)?\s*$",
    re.IGNORECASE,
)
_KQ_VITAL_SIGNS_RE = re.compile(
    r"^(VS|MAP|HR|SpO2|BP)\s*[:/]?\s*[\d.,/\s]+(mmHg|cm|kg|%)?$",
    re.IGNORECASE,
)
# Kết quả định tính
_KQ_QUALITATIVE = {"dương tính", "âm tính", "positive", "negative",
                   "bình thường", "không bình thường", "không ghi nhận gì bất thường"}


def _is_drug_text(text: str) -> bool:
    """True nếu text có chứa tên thuốc đã biết (theo multispan pattern)."""
    if not text:
        return False
    t = text.lower().strip()
    # Bỏ các ký tự đặc biệt hay nhiễm quanh tên thuốc
    t_norm = re.sub(r"[()\[\]/]", " ", t)
    t_norm = re.sub(r"\s+", " ", t_norm).strip()
    # Tách token
    tokens = t_norm.split()
    if not tokens:
        return False
    # Match 1 token đầu (VD: "aspirin", "metoprolol 25mg" → match "metoprolol")
    head = tokens[0]
    if head in _DRUG_NAMES:
        return True
    # Match multi-token (VD: "panadol extra")
    for n in (2, 3):
        joined = " ".join(tokens[:n])
        if joined in _DRUG_NAMES:
            return True
    return False


def _is_disease_text(text: str) -> bool:
    if not text:
        return False
    return bool(_DISEASE_PREFIX_RE.search(text.strip()))


def _is_test_text(text: str) -> bool:
    """Heuristic cho TÊN_XÉT_NGHIỆM: procedure pattern hoặc chứa test keyword."""
    if not text:
        return False
    tl = text.lower().strip()
    # keyword đơn giản
    for kw in _TEST_NAME_KEYWORDS:
        if kw.lower() in tl:
            return True
    # procedure pattern (đã exclude drug keywords)
    try:
        if _is_procedure(text):
            return True
    except Exception:
        pass
    return False


def _is_kq_text(text: str) -> bool:
    if not text:
        return False
    t = text.strip().lower()
    if t in _KQ_QUALITATIVE:
        return True
    if _KQ_QUANTITATIVE_RE.match(text.strip()):
        return True
    if _KQ_VITAL_SIGNS_RE.match(text.strip()):
        return True
    return False


# ════════════════════════════════════════════════════════════════════════════════
# Audit core
# ════════════════════════════════════════════════════════════════════════════════

VALID_TYPES = {"THUỐC", "CHẨN_ĐOÁN", "TRIỆU_CHỨNG", "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"}


def _audit_one_file(path: Path) -> dict[str, Any]:
    """Trả về dict chứa tất cả suspect + stats cho 1 file."""
    out: dict[str, Any] = {
        "file": path.name,
        "total_entities": 0,
        "drug_not_thuoc": [],          # text là thuốc nhưng type khác
        "test_not_tenxn": [],          # text là procedure/test nhưng type khác
        "disease_not_chandoan": [],    # text là bệnh danh nhưng type khác
        "kq_not_kqxn": [],             # text là kết quả nhưng type khác
        "type_inconsistent": [],       # cùng text nhưng khác type trong cùng file
        "span_overlap": [],            # span chồng lấn
        "duplicate_exact": 0,          # trùng exact (text, type, position)
        "bad_assertions": 0,           # assertions không nằm trong whitelist
        "invalid_type": 0,
    }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        out["parse_error"] = str(exc)
        return out

    if not isinstance(data, list):
        out["parse_error"] = "root not a list"
        return out

    out["total_entities"] = len(data)

    # Lookup table: text_normalized → list of (type, idx)
    text_index: dict[str, list[tuple[str, int]]] = defaultdict(list)
    seen_exact: set[tuple[str, str, tuple[int, int]]] = set()

    # Span overlap check (per type group)
    span_by_type: dict[str, list[tuple[int, int, int]]] = defaultdict(list)  # (start, end, idx)

    for i, ent in enumerate(data):
        if not isinstance(ent, dict):
            out["invalid_type"] += 1
            continue
        text = str(ent.get("text", "")).strip()
        etype = str(ent.get("type", "")).strip()
        pos = ent.get("position") or []
        text_norm = text.lower().strip()

        if etype not in VALID_TYPES:
            out["invalid_type"] += 1
            continue

        # ── Heuristic 1: drug text → không phải THUỐC ──
        if etype != "THUỐC" and _is_drug_text(text):
            out["drug_not_thuoc"].append({"idx": i, "text": text, "type": etype})

        # ── Heuristic 2: test/procedure text → không phải TÊN_XÉT_NGHIỆM ──
        # Lưu ý: bỏ qua nếu đã là CHẨN_ĐOÁN đúng (vd: 'block nhĩ thất' vừa là CHẨN_ĐOÁN vừa có 'block')
        if etype not in ("TÊN_XÉT_NGHIỆM", "CHẨN_ĐOÁN") and _is_test_text(text):
            out["test_not_tenxn"].append({"idx": i, "text": text, "type": etype})

        # ── Heuristic 3: disease text → không phải CHẨN_ĐOÁN ──
        if etype != "CHẨN_ĐOÁN" and etype != "TRIỆU_CHỨNG" and _is_disease_text(text):
            # Cẩn thận: vài triệu chứng cũng có thể chứa 'viêm' (vd: 'đau do viêm X' = TC)
            # nhưng ta chỉ flag nếu pattern khớp ở đầu cụm ngắn, không phải modifier
            out["disease_not_chandoan"].append({"idx": i, "text": text, "type": etype})

        # ── Heuristic 4: kq text → không phải KQ_XN ──
        if etype != "KẾT_QUẢ_XÉT_NGHIỆM" and _is_kq_text(text):
            out["kq_not_kqxn"].append({"idx": i, "text": text, "type": etype})

        # ── Heuristic 5: cùng text trong file nhưng khác type ──
        if text_norm:
            text_index[text_norm].append((etype, i))

        # ── Heuristic 6: span overlap (trong cùng type) ──
        if isinstance(pos, list) and len(pos) == 2 and all(isinstance(x, int) for x in pos):
            s, e = pos[0], pos[1]
            span_by_type[etype].append((s, e, i))
            sig = (text_norm, etype, (s, e))
            if sig in seen_exact:
                out["duplicate_exact"] += 1
            seen_exact.add(sig)

    # Build type-inconsistent report
    for text_norm, occurrences in text_index.items():
        types = {t for t, _ in occurrences}
        if len(types) > 1:
            for t, idx in occurrences:
                out["type_inconsistent"].append({
                    "idx": idx, "text": text_norm, "types": sorted(types),
                })

    # Span overlap detection (same type, partial overlap)
    for etype, spans in span_by_type.items():
        spans_sorted = sorted(spans, key=lambda x: x[0])
        for j in range(len(spans_sorted) - 1):
            s1, e1, i1 = spans_sorted[j]
            s2, e2, i2 = spans_sorted[j + 1]
            if s2 < e1:  # strict overlap
                out["span_overlap"].append({
                    "type": etype,
                    "span1": [s1, e1], "idx1": i1,
                    "span2": [s2, e2], "idx2": i2,
                })

    return out


def _print_report(per_file: list[dict[str, Any]], top: int) -> None:
    print(f"\n{'═' * 70}")
    print(f"AUDIT TYPE-MISMATCH — scanned {len(per_file)} file")
    print(f"{'═' * 70}")

    totals = {
        "drug_not_thuoc": 0,
        "test_not_tenxn": 0,
        "disease_not_chandoan": 0,
        "kq_not_kqxn": 0,
        "type_inconsistent": 0,
        "span_overlap": 0,
        "duplicate_exact": 0,
        "bad_assertions": 0,
        "invalid_type": 0,
        "total_entities": 0,
    }
    for r in per_file:
        if "parse_error" in r:
            continue
        for k in totals:
            v = r.get(k, 0)
            if isinstance(v, list):
                totals[k] += len(v)
            else:
                totals[k] += v

    print(f"\n── TỔNG QUAN ──")
    print(f"  Tổng entities:     {totals['total_entities']:>6}")
    print(f"  Type sai tên thuốc → không phải THUỐC:        {totals['drug_not_thuoc']:>6}  ⚠ nghiêm trọng nhất")
    print(f"  Type sai procedure/test → không phải TÊN_XN:   {totals['test_not_tenxn']:>6}")
    print(f"  Type sai bệnh danh → không phải CHẨN_ĐOÁN:    {totals['disease_not_chandoan']:>6}")
    print(f"  Type sai kết quả → không phải KQ_XN:           {totals['kq_not_kqxn']:>6}")
    print(f"  Cùng text khác type trong file:                 {totals['type_inconsistent']:>6}")
    print(f"  Span overlap:                                   {totals['span_overlap']:>6}")
    print(f"  Duplicate exact (text+type+pos):                {totals['duplicate_exact']:>6}")
    print(f"  Type không thuộc 5 loại hợp lệ:                {totals['invalid_type']:>6}")

    def _show(key: str, label: str, fmt_fields: Iterable[str]):
        examples: list[tuple[str, dict]] = []
        for r in per_file:
            if "parse_error" in r:
                continue
            for item in r.get(key, []):
                examples.append((r["file"], item))
        if not examples:
            print(f"\n  ✓ {label}: 0 case")
            return
        print(f"\n── {label}: {len(examples)} case (top {top}) ──")
        for fp, item in examples[:top]:
            row = " | ".join(f"{f}={item.get(f, '')}" for f in fmt_fields)
            print(f"  {fp}: {row}")

    _show("drug_not_thuoc", "TYPE NGHI SAI: text là THUỐC nhưng type khác", ["text", "type"])
    _show("test_not_tenxn", "TYPE NGHI SAI: text là TÊN_XN nhưng type khác", ["text", "type"])
    _show("disease_not_chandoan", "TYPE NGHI SAI: text là CHẨN_ĐOÁN nhưng type khác", ["text", "type"])
    _show("kq_not_kqxn", "TYPE NGHI SAI: text là KQ_XN nhưng type khác", ["text", "type"])
    _show("type_inconsistent", "CÙNG TEXT KHÁC TYPE trong cùng file", ["text", "types"])

    if totals["span_overlap"]:
        print(f"\n── SPAN OVERLAP (top {top}) ──")
        shown = 0
        for r in per_file:
            if shown >= top:
                break
            if "parse_error" in r:
                continue
            for item in r.get("span_overlap", []):
                if shown >= top:
                    break
                print(f"  {r['file']}: {item}")
                shown += 1

    # ── DISTRIBUTION BY TYPE in 'current actual output' ──
    print(f"\n── PHÂN BỐ TYPE trong tất cả file (đếm thực tế) ──")
    type_counter: Counter[str] = Counter()
    for fp in (_PROJECT_ROOT / "output").glob("*.json"):
        try:
            for ent in json.loads(fp.read_text(encoding="utf-8")):
                type_counter[ent.get("type", "?").strip()] += 1
        except Exception:
            continue
    for t, c in type_counter.most_common():
        bar = "█" * min(40, c // 10)
        print(f"  {t:<25} {c:>6}  {bar}")

    print()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("output"))
    p.add_argument("--limit", type=int, default=0, help="0 = all")
    p.add_argument("--top", type=int, default=30)
    p.add_argument("--save", type=Path, default=None,
                   help="Nếu set, ghi full report ra file JSON")
    args = p.parse_args()

    # Wrap stdout để không lỗi Unicode trên Windows cp1252
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    files = sorted(args.input.glob("*.json"), key=lambda x: int(x.stem))
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        print(f"Không có file JSON trong {args.input}", file=sys.stderr)
        return 1

    per_file = [_audit_one_file(f) for f in files]
    _print_report(per_file, args.top)

    if args.save:
        args.save.write_text(
            json.dumps(per_file, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Đã ghi full report → {args.save}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
