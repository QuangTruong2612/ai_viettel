"""Test family patterns properly."""
import re

family_patterns = [
    r'\bbố\s+([bệe]nh\s+)?nh[âa]n',
    r'\bm[ẹe]\s+([bệe]nh\s+)?nh[âa]n',
    r'\bcha\s+([bệe]nh\s+)?nh[âa]n',
]

tests = [
    'Bố bệnh nhân bị THA',
    'Mẹ bệnh nhân bị tiểu đường',
    'Cha bệnh nhân bị ung thư',
    'Anh trai bệnh nhân bị đau đầu',
    'tiền sử gia đình có THA',
    'Bệnh nhân bị THA',  # should NOT match
]

print('=== Family pattern tests ===')
for text in tests:
    matched = []
    for pat in family_patterns:
        if re.search(pat, text, re.IGNORECASE):
            matched.append(pat)
    print(f'  {text!r:50} -> matched: {len(matched) > 0} ({[p[:30] for p in matched]})')