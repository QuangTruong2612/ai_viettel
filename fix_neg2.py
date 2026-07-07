"""Fix isNegated: include 'không' (without trailing space) check near entity."""
from pathlib import Path

P = Path(r'f:\AI_VIETTEL\src\postprocess.py')
src = P.read_text(encoding='utf-8')

old = '''    # isNegated: cũng check pattern với "chưa", "âm tính", "không có"
    for neg_word in ("không có", "chưa", "âm tính"):
        if neg_word in pre_window or neg_word in text_lower[max(0, pos-10):pos + len(entity_text) + 5]:
            found.append("isNegated")
            break'''

new = '''    # isNegated: check "không", "chưa", "âm tính" trong window 20 chars trước entity.
    # Lưu ý: "không" có thể nằm sát entity (vd "không sốt" → pre_window kết thúc bằng "khô").
    near = text_lower[max(0, pos - 15):pos + 5]  # rộng hơn để bắt "không "
    found_negated = False
    for neg in ("không", "chưa", "âm tính"):
        if neg in near:
            found_negated = True
            break
    if found_negated and "isNegated" not in found:
        found.append("isNegated")'''

n = src.count(old)
print(f'Found {n} occurrences')
if n > 0:
    src = src.replace(old, new, 1)
    P.write_text(src, encoding='utf-8')
    print('Wrote fixed isNegated')