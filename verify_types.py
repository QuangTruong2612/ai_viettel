"""Check type string variants - both with/without diacritics."""
import json
from pathlib import Path
from collections import Counter

OUTPUT_DIR = Path(r"F:\AI_VIETTEL\output")
type_counter = Counter()
raw_types = Counter()

# Load audit
audit = json.load(open(r"F:\AI_VIETTEL\audit_report.json", encoding="utf-8"))
print("Audit invalid schema count:", len(audit.get("invalid_schema", [])))

# Sample some KET_QUA entities from each file
for fout in sorted(OUTPUT_DIR.glob("*.json"), key=lambda p: int(p.stem))[:30]:
    fid = fout.stem
    ents = json.load(open(fout, encoding="utf-8"))
    for e in ents:
        raw_types[e.get("type", "")] += 1

print("\nRaw type value counts across sampled files:")
for t, c in raw_types.most_common(15):
    print(f"  {t!r}: {c}")
