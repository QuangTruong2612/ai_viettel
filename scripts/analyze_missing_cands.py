import json, glob, collections
from pathlib import Path

diag_no_cand = collections.Counter()
drug_no_cand = collections.Counter()
sym_no_cand = collections.Counter()

for f in glob.glob('output/*.json'):
    try:
        ents = json.loads(Path(f).read_text('utf-8'))
        for e in ents:
            cands = e.get('candidates', [])
            t = e.get('type','')
            txt = e.get('text','').strip().lower()
            if not cands:
                if t == 'CHẨN_ĐOÁN':
                    diag_no_cand[txt] += 1
                elif t == 'THUỐC':
                    drug_no_cand[txt] += 1
                elif t == 'TRIỆU_CHỨNG':
                    sym_no_cand[txt] += 1
    except Exception as exc:
        print(f"Error {f}: {exc}")

print('=== TOP 30 CHẨN_ĐOÁN NO CANDIDATES ===')
for k,v in diag_no_cand.most_common(30):
    print(f'  {v:3d}x {k}')
print()
print('=== TOP 30 THUỐC NO CANDIDATES ===')
for k,v in drug_no_cand.most_common(30):
    print(f'  {v:3d}x {k}')
print()
print('=== TOP 30 TRIỆU_CHỨNG NO CANDIDATES (symptom map miss) ===')
for k,v in sym_no_cand.most_common(30):
    print(f'  {v:3d}x {k}')
