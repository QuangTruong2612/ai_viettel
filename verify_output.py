"""Cross-check output entities against input text — verify positions + types."""
import json, os, sys
from pathlib import Path

INPUT_DIR = Path(r"F:\AI_VIETTEL\input")
OUTPUT_DIR = Path(r"F:\AI_VIETTEL\output")

# Load audit report
audit = json.load(open(r"F:\AI_VIETTEL\audit_report.json", encoding="utf-8"))
mismatches_in_audit = audit.get("position_mismatches_count", 0)
type_counts = audit.get("type_counts", {})

# Verify by sampling + scanning all files
samples_to_check = [1, 20, 26, 30, 43, 83]  # From audit + smoke test

def load_input(n):
    p = INPUT_DIR / f"{n}.txt"
    if p.exists():
        return p.read_text(encoding="utf-8")
    return None

def load_output(n):
    p = OUTPUT_DIR / f"{n}.json"
    if p.exists():
        return json.load(open(p, encoding="utf-8"))
    return None

print("=" * 70)
print("VERIFICATION SUMMARY (from audit_report.json)")
print("=" * 70)
print(f"  Empty outputs : {audit.get('empty_outputs')}")
print(f"  JSON errors   : {audit.get('json_errors')}")
print(f"  Pos mismatches: {mismatches_in_audit} (in files {audit.get('files_with_mismatches')})")
print(f"  Invalid schema: {len(audit.get('invalid_schema', []))} entries (all KET_QUA_XET_NGHIEM type missing)")
print()
print("Type counts:")
for t, c in type_counts.items():
    print(f"  {t:25s} {c}")
print()
print("Assertion counts:")
for a, c in audit.get("assertion_counts", {}).items():
    print(f"  {a:25s} {c}")

print()
print("=" * 70)
print("SAMPLING VERIFICATION — Re-read input + compare positions")
print("=" * 70)

# Detailed verification of sample files
for n in samples_to_check:
    inp = load_input(n)
    out = load_output(n)
    if inp is None or out is None:
        continue
    ents = out if isinstance(out, list) else out.get("entities", [])
    print(f"\n--- File {n}.txt ({len(inp)} chars, {len(ents)} entities) ---")
    pos_ok = 0
    pos_mismatch = 0
    mismatches_list = []
    for i, e in enumerate(ents):
        text = e.get("text", "")
        pos = e.get("position", [])
        etype = e.get("type", "")
        if len(pos) != 2:
            pos_mismatch += 1
            continue
        actual = inp[pos[0]:pos[1]]
        if actual == text:
            pos_ok += 1
        else:
            pos_mismatch += 1
            mismatches_list.append((i, text, actual, pos, etype))
    print(f"  Positions OK: {pos_ok}/{len(ents)}")
    print(f"  Mismatches:   {pos_mismatch}")
    if mismatches_list:
        print(f"  First 3:")
        for m in mismatches_list[:3]:
            i, t, a, p, ty = m
            print(f"    [{i}] {ty:15s} text={t!r} actual_at_{p}={a!r}")

print()
print("=" * 70)
print("KEY FINDINGS")
print("=" * 70)
print("""
1. POSITION MISMATCH (~85+ across 5+ files):
   - Common pattern: position ends 1 short of expected length
     e.g. expected 'Buồn nôn' [1105:1113], actual 'Buồn nô' [1105:1113]
     → looks like off-by-one error in postprocess position fix (len-1 vs len)
   - Files 26, 30, 43, 83 show systematic truncation: last char missing
   - This is HIGH IMPACT: WER drops drastically when text mismatches

2. INVALID SCHEMA: 189 entities with type 'KẾT_QUẢ_XÉT_NGHIỆM' (with diacritics)
   - Schema only accepts 'KET_QUA_XET_NGHIEM' (ASCII no-diacritics)
   - This means all 189 KQ entries are being counted as INVALID → score loss

3. EMPTY OUTPUTS: file 75 and 96 → 0 entities extracted

4. Type counts seem reasonable:
   - TRIỆU_CHỨNG: 1205 (largest — expected, many symptoms in clinical text)
   - CHẨN_ĐOÁN: 691
   - TÊN_XÉT_NGHIỆM: 361
   - THUỐC: 260
   - KẾT_QUẢ_XÉT_NGHIỆM: 189 (all INVALID due to diacritics bug)

5. ASSERTIONS (per correct schema):
   - isHistorical: 355 (good — looks like prescriptions/tiền sử)
   - isNegated: 244
   - isFamily: 107
""")
