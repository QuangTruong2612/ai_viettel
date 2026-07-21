"""Phân tích assertion distribution trong output files."""
import json, glob, collections
from pathlib import Path

assert_dist = collections.Counter()
assert_by_type = collections.defaultdict(collections.Counter)
no_assert_by_type = collections.Counter()
total_by_type = collections.Counter()

for f in glob.glob('output/*.json'):
    try:
        ents = json.loads(Path(f).read_text('utf-8'))
        for e in ents:
            t = e.get('type','')
            total_by_type[t] += 1
            assertions = e.get('assertions', [])
            if assertions:
                for a in assertions:
                    assert_dist[a] += 1
                    assert_by_type[t][a] += 1
            else:
                no_assert_by_type[t] += 1
    except Exception as exc:
        print(f"Error {f}: {exc}")

print('=== ASSERTION DISTRIBUTION ===')
for k,v in assert_dist.most_common():
    print(f'  {v:4d}x {k}')
print()
print('=== ASSERTION BY TYPE ===')
for t, counter in sorted(assert_by_type.items()):
    print(f'  {t}:')
    for a, v in counter.most_common():
        print(f'    {v:4d}x {a}')
print()
print('=== NO ASSERTIONS BY TYPE ===')
for t, v in no_assert_by_type.most_common():
    total = total_by_type[t]
    print(f'  {t}: {v}/{total} ({100*v/max(1,total):.0f}% có empty assertions)')
