"""Full audit of all output files - position + type accuracy."""
import json, os, sys, io
from pathlib import Path
from collections import Counter, defaultdict

INPUT_DIR = Path(r"F:\AI_VIETTEL\input")
OUTPUT_DIR = Path(r"F:\AI_VIETTEL\output")

# Configure stdout utf-8 on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

total_ents = 0
pos_ok = 0
pos_mismatch = 0
mismatch_files = defaultdict(int)
type_counter = Counter()
errors = []
empty_files = []

# categorize mismatches
off_by_one = 0   # exact text but len is off by 1
shift = 0         # text not in position at all
case_only = 0     # case mismatch only
ws_only = 0       # whitespace mismatch only

VALID_TYPES = {"THUỐC", "CHẨN_ĐOÁN", "TRIỆU_CHỨNG",
               "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"}

for fout in sorted(OUTPUT_DIR.glob("*.json"), key=lambda p: int(p.stem)):
    fid = fout.stem
    try:
        ents = json.load(open(fout, encoding="utf-8"))
    except Exception as e:
        errors.append(f"{fid}: {e}")
        continue
    if not ents:
        empty_files.append(fid)
        continue
    inp_path = INPUT_DIR / f"{fid}.txt"
    if not inp_path.exists():
        continue
    inp = inp_path.read_text(encoding="utf-8")
    for i, e in enumerate(ents):
        total_ents += 1
        t = e.get("type", "")
        type_counter[t] += 1
        text = e.get("text", "")
        pos = e.get("position", [])
        if len(pos) != 2:
            pos_mismatch += 1
            mismatch_files[fid] += 1
            continue
        actual = inp[pos[0]:pos[1]]
        if actual == text:
            pos_ok += 1
        else:
            pos_mismatch += 1
            mismatch_files[fid] += 1
            if actual.strip().lower() == text.strip().lower():
                if actual == actual.strip() and text == text.strip():
                    case_only += 1
                else:
                    ws_only += 1
            elif len(actual) == len(text) and len(text) > 1 and text[:-1] == actual[:-1] + actual[-1] or (
                len(actual) == len(text) and (text[:-1].lower() == actual[:-1].lower())
            ):
                off_by_one += 1
            else:
                shift += 1

print(f"Total entities: {total_ents}")
print(f"Positions OK: {pos_ok} ({pos_ok/max(1,total_ents)*100:.1f}%)")
print(f"Positions MISMATCH: {pos_mismatch} ({pos_mismatch/max(1,total_ents)*100:.1f}%)")
print()
print(f"Files with mismatches: {len(mismatch_files)}")
worst = sorted(mismatch_files.items(), key=lambda x: -x[1])[:10]
print(f"Top 10 worst files: {worst}")
print()
print(f"Empty files: {empty_files}")
print()
print("Type counter:")
for t, c in type_counter.most_common():
    invalid = "" if t in VALID_TYPES else " <- INVALID (not in schema)"
    print(f"  {t!r:30s} {c}{invalid}")
