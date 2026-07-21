"""Deep analysis: So sánh output với input để tìm nguyên nhân J_candidates thấp.

Phân tích:
1. Candidates phân bố theo type
2. Số entities có vs không có candidates
3. Ví dụ cụ thể về CHẨN_ĐOÁN/THUỐC có candidates
"""

import json
import glob
from pathlib import Path
from collections import Counter, defaultdict

diag_with_cands = []
drug_with_cands = []
diag_no_cands = []
drug_no_cands = []
sample_diag_cands = Counter()  # ICD code → count
sample_drug_cands = Counter()  # rxcui → count

for f in sorted(glob.glob('output/*.json'), key=lambda x: int(Path(x).stem)):
    try:
        ents = json.loads(Path(f).read_text('utf-8'))
        for e in ents:
            t = e.get('type','')
            txt = e.get('text','')
            cands = e.get('candidates', [])
            if t == 'CHẨN_ĐOÁN':
                if cands:
                    diag_with_cands.append({'text': txt, 'candidates': cands, 'file': Path(f).stem})
                    for c in cands:
                        sample_diag_cands[c] += 1
                else:
                    diag_no_cands.append({'text': txt, 'file': Path(f).stem})
            elif t == 'THUỐC':
                if cands:
                    drug_with_cands.append({'text': txt, 'candidates': cands, 'file': Path(f).stem})
                    for c in cands:
                        sample_drug_cands[c] += 1
                else:
                    drug_no_cands.append({'text': txt, 'file': Path(f).stem})
    except Exception as exc:
        print(f"Error {f}: {exc}")

print(f"=== CHẨN_ĐOÁN: {len(diag_with_cands)} có candidates, {len(diag_no_cands)} không có ===")
print(f"=== THUỐC: {len(drug_with_cands)} có candidates, {len(drug_no_cands)} không có ===")
print()
print("=== TOP 30 ICD CODES ĐƯỢC DÙNG ===")
for code, count in sample_diag_cands.most_common(30):
    print(f"  {count:3d}x {code}")
print()
print("=== TOP 30 RXNORM CODES ĐƯỢC DÙNG ===")
for code, count in sample_drug_cands.most_common(30):
    print(f"  {count:3d}x {code}")
print()
print("=== 15 VÍ DỤ CHẨN_ĐOÁN CÓ CANDIDATES ===")
for e in diag_with_cands[:15]:
    print(f"  [{e['file']}] {e['text'][:50]} → {e['candidates'][:3]}")
print()
print("=== 10 VÍ DỤ THUỐC CÓ CANDIDATES ===")
for e in drug_with_cands[:10]:
    print(f"  [{e['file']}] {e['text'][:50]} → {e['candidates'][:3]}")
print()
print("=== 10 CHẨN_ĐOÁN KHÔNG CÓ CANDIDATES ===")
for e in diag_no_cands[:10]:
    print(f"  [{e['file']}] {e['text'][:60]}")
