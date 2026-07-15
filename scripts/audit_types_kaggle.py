"""Audit type-mismatch cho output JSON — KAGGLE-READY, SELF-CONTAINED.

Không import từ src.* — tất cả regex/whitelist được embed sẵn.
Chỉ cần upload lên Kaggle cùng các file data:
    data/drug_inn_cache.json      (~63k INN names — optional nhưng rất nên có)
    data/drug_aliases.json        (brand → generic map)
    data/drug_brand_seed.json     (cleaned 2026-07-15)
    data/common_drug_names.json   (179 common names — optional)
    data/procedure_patterns.json  (Vietnamese procedure verbs)

Tự động detect:
    - Nếu chạy local (CWD có folder data/, output/): dùng luôn
    - Nếu chạy trên Kaggle (/kaggle/input/...): tự detect dataset path
    - Có thể override bằng --data-dir, --input, --save

Công thức chấm:
    final = 0.3·text + 0.3·assertions + 0.4·candidates
Sai TYPE (đúng span, đúng text, sai class) = trừ điểm CẢ 3 metric, NHÂN ĐÔI
(thừa 1 + thiếu 1) — tốn kém nhất trong toàn bộ scoring.

Usage (Kaggle notebook cell):
    !python audit_types_kaggle.py --input /kaggle/working/output
    !python audit_types_kaggle.py --input /kaggle/working/output --save /kaggle/working/audit.json
    !python audit_types_kaggle.py --input /kaggle/working/output --data-dir /kaggle/input/ai-viettel-data/data --top 50
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


# ════════════════════════════════════════════════════════════════════════════════
# Path resolution — Kaggle-aware
# ════════════════════════════════════════════════════════════════════════════════

def _resolve_data_dir(arg_data_dir: str | None) -> Path:
    """Tìm folder data/ theo thứ tự ưu tiên:
       1. CLI argument --data-dir
       2. ENV DATA_DIR
       3. CWD/data
       4. /kaggle/input/ai-viettel-data/data
       5. /kaggle/input/ai_viettel/data
       6. script_parent/data (chạy local trong repo)
    """
    if arg_data_dir:
        p = Path(arg_data_dir)
        if p.exists():
            return p
        print(f"[warn] --data-dir={arg_data_dir} không tồn tại, fallback...", file=sys.stderr)

    env = os.environ.get("DATA_DIR")
    if env and Path(env).exists():
        return Path(env)

    candidates = [
        Path.cwd() / "data",
        Path("/kaggle/input/ai-viettel-data/data"),
        Path("/kaggle/input/ai_viettel/data"),
        Path("/kaggle/input") / "ai-viettel-data" / "data",
        Path("/kaggle/input") / "ai_viettel" / "data",
        Path(__file__).resolve().parent.parent / "data" if "__file__" in globals() else None,
    ]
    for c in candidates:
        if c and c.exists():
            return c
    print("[warn] Không tìm thấy data/ — dùng drug whitelist rỗng", file=sys.stderr)
    return Path.cwd() / "data"  # fallback, có thể không tồn tại


def _resolve_input_dir(arg_input: str | None) -> Path:
    if arg_input:
        return Path(arg_input)
    candidates = [
        Path.cwd() / "output",
        Path("/kaggle/working/output"),
        Path("/kaggle/input/ai-viettel-output"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return Path.cwd() / "output"


def _resolve_save_path(arg_save: str | None) -> Path | None:
    if arg_save:
        return Path(arg_save)
    candidates = [
        Path("/kaggle/working/_audit_report.json"),
        Path.cwd() / "_audit_report.json",
    ]
    for c in candidates:
        try:
            c.parent.mkdir(parents=True, exist_ok=True)
            return c
        except Exception:
            continue
    return None


# Wrap stdout/stderr để tránh lỗi Unicode trên một số môi trường
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except AttributeError:
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


# ════════════════════════════════════════════════════════════════════════════════
# EMBEDDED REGEX — copy từ src/postprocess.py (không cần import)
# Cập nhật đồng bộ với R37 (2026-07-15).
# ════════════════════════════════════════════════════════════════════════════════

_ABNORMAL_FINDING_TO_CHAN_DOAN = re.compile(
    r"^(tràn dịch màng phổi|tràn dịch màng tim|tràn dịch ổ bụng|cổ trướng|"
    r"tràn khí màng phổi|tràn khí trung thất|"
    r"tim to|gan to|lách to|thận to|"
    r"xẹp phổi|tràn khí phổi|giãn phế quản|"
    r"xơ phổi|khí phế thủng|giãn phế nang|"
    r"gan nhiễm mỡ|xơ gan|thoát vị hoành|"
    r"giãn đường mật|tắc nghẽn đường mật|sỏi mật|"
    r"phù phổi|phù não|"
    r"gãy xương \w+|gãy \w+ xương|gãy xương|"
    r"chấn thương sọ não|chấn thương \w+|"
    r"vết thương hở \w+|"
    r"hở van (hai lá|ba lá|động mạch chủ|động mạch phổi|2 lá)|"
    r"hẹp van (hai lá|ba lá|động mạch chủ|động mạch phổi|2 lá)|"
    r"hở van \w+ (nhẹ|vừa|nặng|mild|moderate|severe)|"
    r"hẹp van \w+ (nhẹ|vừa|nặng|mild|moderate|severe)|"
    r"mất vận động vùng đỉnh|rối loạn vận động vùng đỉnh|"
    r"giãn \w+ buồng tim|"
    r"u ác tính|khối u ác tính|khối u \w+|"
    r"viêm \w+ (nặng|cấp|mạn)|"
    r"viêm\s+(?:tuyến\s+mồ\s+hôi|phổi|gan|thận|dạ\s+dày|ruột|tụy|"
    r"túi\s+mật|bàng\s+quang|phế\s+quản|thanh\s+quản|"
    r"khớp|cơ|tim|màng\s+ngoài\s+tim|màng\s+tim|cơ\s+tim|"
    r"não|màng\s+não|xương|tủy(?:\s+xương)?|"
    r"bàng quang|họng|amidan|"
    r"xoang|phổi\s+kẽ|bụng|não\s+tủy|đại\s+tràng|"
    r"dây\s+thần\s+kinh|van\s+tim|tiết\s+niệu|"
    r"ruột\s+non|ruột\s+thừa|thực\s+quản|hang\s+vị|"
    r"trực\s+tràng|hậu\s+môn|tiền\s+liệt\s+tuyến|"
    r"mô\s+tế\s+bào|"
    r"\w+))|"
    r"bệnh\s+lý\s+chất\s+trắng|"
    r"ST\s+chênh(?:\s+(?:xuống|lên|chênh))?|"
    r"ST\s+chênh\s+(?:xuống|lên)\s+\w+|"
    r"block\s+(?:nhĩ\s+thất|nhĩ|thất)(?:\s+\w+)?|"
    r"rung\s+nhĩ(?:\s+\w+)?|"
    r"cuồng\s+nhĩ(?:\s+\w+)?|"
    r"ngoại\s+tâm\s+thu\s+(?:nhĩ|thất)(?:\s+(?:xuất\s+hiện|thường\s+xuyên|có|chiếm)\s+\w+)*|"
    r"tổn\s+thương\s+\w+(?:\s+\w+)?|"
    r"(?:hẹp|hở)\s+động\s+mạch\s+\w+|"
    r"phình\s+(?:động\s+mạch|đại\s+tràng|tĩnh\s+mạch)\s+\w*",
    re.IGNORECASE | re.UNICODE,
)

_TEST_NAME_KEYWORDS = [
    "X-quang", "xquang", "CT scan", "MRI", "siêu âm", "sieu am",
    "điện tâm đồ", "dien tam do", "ECG", "Holter", "monitor",
    "công thức máu", "nước tiểu", "phân tích", "xét nghiệm",
    "nội soi", "sinh thiết", "tế bào", "mô bệnh học",
    "PSA", "TSH", "WBC", "Hgb", "AST", "ALT", "GGT", "CRP",
]

_DISEASE_PREFIX_RE = re.compile(
    r"^(viêm|ung thư|suy|thoái hóa|rối loạn|hội chứng|bệnh|"
    r"tăng huyết áp|đái tháo đường|nhồi máu|xuất huyết|tràn dịch|"
    r"ngoại tâm thu|rung nhĩ|block|st chênh|suy hô hấp|suy tim|"
    r"suy thận|suy gan|nhiễm trùng|nhiễm khuẩn|tắc|"
    r"gãy|trật|thoát vị|phình|bướu|u nang|polyp|sỏi|"
    r"hen |COPD|CKD|THA\b|ĐTĐ|NMCT|RLLL|NMH)",
    re.IGNORECASE | re.UNICODE,
)

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
_KQ_QUALITATIVE = {
    "dương tính", "âm tính", "positive", "negative",
    "bình thường", "không bình thường", "không ghi nhận gì bất thường",
}

# Procedure patterns (fallback nếu data/procedure_patterns.json không có)
_FALLBACK_PROC_PATTERNS = re.compile(
    r"^(phẫu thuật(?:\s+\w[\w\s]*)?|nội soi(?:\s+\w[\w\s]*)?|chọc dò(?:\s+\w[\w\s]*)?|"
    r"đặt stent(?:\s+\w[\w\s]*)?|đặt ống(?:\s+\w[\w\s]*)?|"
    r"thủ thuật(?:\s+\w[\w\s]*)?|can thiệp(?:\s+\w[\w\s]*)?|cắt \w+|"
    r"xạ trị|hóa trị|"
    r"siêu âm|chụp \w+|"
    r"đo \w+|test \w+ \w+)$",
    re.IGNORECASE | re.UNICODE,
)
_PROC_EXCLUDE_CONTAINS = [
    "thuốc", "viên uống", "viên nén", "viên nang",
    "uống", "tiêm truyền", "tiêm bắp", "tiêm tĩnh mạch",
    " mg ", " ml ", " mcg ", " iu ",
]
_PROC_EXCLUDE_STARTS = [
    "siêu âm", "chụp x-quang", "chụp ct", "chụp mri",
    "điện tâm đồ", "đo ", "xét nghiệm",
]

# ════════════════════════════════════════════════════════════════════════════════
# Load data files (drug whitelist + procedure patterns)
# ════════════════════════════════════════════════════════════════════════════════

def _safe_read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[warn] Không đọc được {path}: {exc}", file=sys.stderr)
        return None


def _load_drug_set(data_dir: Path) -> set[str]:
    """Union drug_inn_cache + common_drug_names + drug_aliases keys + drug_brand_seed keys."""
    s: set[str] = set()

    inn = data_dir / "drug_inn_cache.json"
    if inn.exists():
        d = _safe_read_json(inn)
        if isinstance(d, list):
            s.update(str(x).lower().strip() for x in d if isinstance(x, str))

    common = data_dir / "common_drug_names.json"
    if common.exists():
        d = _safe_read_json(common)
        if isinstance(d, list):
            s.update(str(x).lower().strip() for x in d if isinstance(x, str))

    aliases = data_dir / "drug_aliases.json"
    if aliases.exists():
        d = _safe_read_json(aliases)
        if isinstance(d, dict):
            for k in d.keys():
                kl = str(k).lower().strip()
                if kl and "?" not in kl:
                    s.add(kl)
            # Values là INN thật, vd "aspirin" → cũng add
            for v in d.values():
                if isinstance(v, str) and "?" not in v and "/" not in v:
                    s.add(v.lower().strip())

    brand_seed = data_dir / "drug_brand_seed.json"
    if brand_seed.exists():
        d = _safe_read_json(brand_seed)
        if isinstance(d, dict):
            for k in d.keys():
                if k.startswith("_"):
                    continue
                kl = str(k).lower().strip()
                if kl and "?" not in kl and kl not in {"bipap", "cpap", "kháng", "thuốc"}:
                    s.add(kl)

    return s


def _load_procedure_patterns(data_dir: Path) -> tuple[list[re.Pattern], list[str], list[str]]:
    """Đọc từ data/procedure_patterns.json nếu có, fallback regex cứng."""
    path = data_dir / "procedure_patterns.json"
    if not path.exists():
        return [_FALLBACK_PROC_PATTERNS], _PROC_EXCLUDE_CONTAINS, _PROC_EXCLUDE_STARTS
    d = _safe_read_json(path)
    if not isinstance(d, dict):
        return [_FALLBACK_PROC_PATTERNS], _PROC_EXCLUDE_CONTAINS, _PROC_EXCLUDE_STARTS
    patterns = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in d.get("vn_verbs", [])]
    if not patterns:
        patterns = [_FALLBACK_PROC_PATTERNS]
    return patterns, d.get("exclude_if_contains", _PROC_EXCLUDE_CONTAINS), d.get("exclude_if_startswith", _PROC_EXCLUDE_STARTS)


def _is_procedure(text: str, proc_patterns, exclude_contains, exclude_starts) -> bool:
    if not text or len(text) > 200:
        return False
    tl = text.lower().strip()
    if not tl:
        return False
    for prefix in exclude_starts:
        if tl.startswith(prefix):
            return False
    for kw in exclude_contains:
        if kw in tl:
            return False
    for pat in proc_patterns:
        if pat.search(tl):
            return True
    return False


# ════════════════════════════════════════════════════════════════════════════════
# Heuristic helpers
# ════════════════════════════════════════════════════════════════════════════════

def _is_drug_text(text: str, drug_set: set[str]) -> bool:
    if not text:
        return False
    t = text.lower().strip()
    t_norm = re.sub(r"[()\[\]/]", " ", t)
    t_norm = re.sub(r"\s+", " ", t_norm).strip()
    tokens = t_norm.split()
    if not tokens:
        return False
    if tokens[0] in drug_set:
        return True
    for n in (2, 3):
        joined = " ".join(tokens[:n])
        if joined in drug_set:
            return True
    return False


def _is_disease_text(text: str) -> bool:
    if not text:
        return False
    return bool(_DISEASE_PREFIX_RE.search(text.strip()))


def _is_test_text(text: str, proc_patterns, exclude_contains, exclude_starts) -> bool:
    if not text:
        return False
    tl = text.lower().strip()
    for kw in _TEST_NAME_KEYWORDS:
        if kw.lower() in tl:
            return True
    if _is_procedure(text, proc_patterns, exclude_contains, exclude_starts):
        return True
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


def _audit_one_file(
    path: Path,
    drug_set: set[str],
    proc_patterns,
    exclude_contains,
    exclude_starts,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "file": path.name,
        "total_entities": 0,
        "drug_not_thuoc": [],
        "test_not_tenxn": [],
        "disease_not_chandoan": [],
        "kq_not_kqxn": [],
        "type_inconsistent": [],
        "span_overlap": [],
        "duplicate_exact": 0,
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
    text_index: dict[str, list[tuple[str, int]]] = defaultdict(list)
    seen_exact: set[tuple[str, str, tuple[int, int]]] = set()
    span_by_type: dict[str, list[tuple[int, int, int]]] = defaultdict(list)

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

        if etype != "THUỐC" and _is_drug_text(text, drug_set):
            out["drug_not_thuoc"].append({"idx": i, "text": text, "type": etype})

        if etype not in ("TÊN_XÉT_NGHIỆM", "CHẨN_ĐOÁN") and _is_test_text(text, proc_patterns, exclude_contains, exclude_starts):
            out["test_not_tenxn"].append({"idx": i, "text": text, "type": etype})

        if etype != "CHẨN_ĐOÁN" and etype != "TRIỆU_CHỨNG" and _is_disease_text(text):
            out["disease_not_chandoan"].append({"idx": i, "text": text, "type": etype})

        if etype != "KẾT_QUẢ_XÉT_NGHIỆM" and _is_kq_text(text):
            out["kq_not_kqxn"].append({"idx": i, "text": text, "type": etype})

        if text_norm:
            text_index[text_norm].append((etype, i))

        if isinstance(pos, list) and len(pos) == 2 and all(isinstance(x, int) for x in pos):
            s, e = pos[0], pos[1]
            span_by_type[etype].append((s, e, i))
            sig = (text_norm, etype, (s, e))
            if sig in seen_exact:
                out["duplicate_exact"] += 1
            seen_exact.add(sig)

    for text_norm, occurrences in text_index.items():
        types = {t for t, _ in occurrences}
        if len(types) > 1:
            for t, idx in occurrences:
                out["type_inconsistent"].append({
                    "idx": idx, "text": text_norm, "types": sorted(types),
                })

    for etype, spans in span_by_type.items():
        spans_sorted = sorted(spans, key=lambda x: x[0])
        for j in range(len(spans_sorted) - 1):
            s1, e1, i1 = spans_sorted[j]
            s2, e2, i2 = spans_sorted[j + 1]
            if s2 < e1:
                out["span_overlap"].append({
                    "type": etype,
                    "span1": [s1, e1], "idx1": i1,
                    "span2": [s2, e2], "idx2": i2,
                })

    return out


def _print_report(per_file: list[dict[str, Any]], top: int, input_dir: Path) -> None:
    print(f"\n{'═' * 70}")
    print(f"AUDIT TYPE-MISMATCH — scanned {len(per_file)} file in {input_dir}")
    print(f"{'═' * 70}")

    totals = {
        "drug_not_thuoc": 0, "test_not_tenxn": 0, "disease_not_chandoan": 0,
        "kq_not_kqxn": 0, "type_inconsistent": 0, "span_overlap": 0,
        "duplicate_exact": 0, "invalid_type": 0, "total_entities": 0,
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
    print(f"  Drug text → type ≠ THUỐC:           {totals['drug_not_thuoc']:>6}  ⚠ nghiêm trọng nhất")
    print(f"  Test/proc text → type ≠ TÊN_XN:     {totals['test_not_tenxn']:>6}")
    print(f"  Disease text → type ≠ CHẨN_ĐOÁN:    {totals['disease_not_chandoan']:>6}")
    print(f"  KQ text → type ≠ KQ_XN:             {totals['kq_not_kqxn']:>6}")
    print(f"  Cùng text khác type:                {totals['type_inconsistent']:>6}")
    print(f"  Span overlap:                       {totals['span_overlap']:>6}")
    print(f"  Duplicate exact:                    {totals['duplicate_exact']:>6}")
    print(f"  Invalid type:                       {totals['invalid_type']:>6}")

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

    print(f"\n── PHÂN BỐ TYPE trong tất cả file (đếm thực tế) ──")
    type_counter: Counter[str] = Counter()
    for r in per_file:
        if "parse_error" in r:
            continue
        for ent in (r.get("_raw_data") or []):
            type_counter[ent.get("type", "?").strip()] += 1
    if not type_counter:
        # fallback: re-read from input dir
        for fp in sorted(input_dir.glob("*.json"), key=lambda x: int(x.stem) if x.stem.isdigit() else 0):
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
    p = argparse.ArgumentParser(
        description="Audit type-mismatch cho output JSON (Kaggle-ready, self-contained)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", type=str, default=None,
                   help="Folder chứa output *.json (default: auto-detect /kaggle/working/output hoặc ./output)")
    p.add_argument("--data-dir", type=str, default=None,
                   help="Folder chứa data/ (default: auto-detect Kaggle dataset hoặc ./data)")
    p.add_argument("--limit", type=int, default=0, help="0 = all")
    p.add_argument("--top", type=int, default=30)
    p.add_argument("--save", type=str, default=None,
                   help="Path JSON để ghi full report (default: auto-detect)")
    args = p.parse_args()

    data_dir = _resolve_data_dir(args.data_dir)
    input_dir = _resolve_input_dir(args.input)
    save_path = _resolve_save_path(args.save)

    print(f"[audit] data_dir = {data_dir}")
    print(f"[audit] input_dir = {input_dir}")

    drug_set = _load_drug_set(data_dir)
    proc_patterns, exclude_contains, exclude_starts = _load_procedure_patterns(data_dir)
    print(f"[audit] Loaded {len(drug_set)} drug names, "
          f"{len(proc_patterns)} procedure patterns", file=sys.stderr)

    files = sorted(
        [f for f in input_dir.glob("*.json") if f.stem.isdigit()],
        key=lambda x: int(x.stem),
    )
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        print(f"Không có file JSON (digit-stem) trong {input_dir}", file=sys.stderr)
        return 1

    per_file = [
        _audit_one_file(f, drug_set, proc_patterns, exclude_contains, exclude_starts)
        for f in files
    ]
    _print_report(per_file, args.top, input_dir)

    if save_path:
        # Drop _raw_data trước khi save (chỉ cần cho type-counter đã in ra rồi)
        slim = [{k: v for k, v in r.items() if not k.startswith("_")} for r in per_file]
        save_path.write_text(
            json.dumps(slim, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Đã ghi full report → {save_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())