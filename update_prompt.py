"""Update KQ rule + add example with test/value separation."""
from pathlib import Path

P = Path(r'f:\AI_VIETTEL\src\prompts.py')
src = P.read_text(encoding='utf-8')

# Update KQ rule
old = '''    5. **KẾT_QUẢ_XÉT_NGHIỆM** — Số + đơn vị HOẶC kết luận ngắn. VD: "WBC 12.5 K/uL", "ecg bình thường" (kết luận ngắn), "SpO2 96%".'''
new = '''    5. **KẾT_QUẢ_XÉT_NGHIỆM** — CHỈ là SỐ (hoặc kết luận ngắn). KHÔNG kèm tên test.
       Phân tách test/value theo dấu `:` hoặc `=`:
         "WBC:14,43" → TÊN_XÉT_NGHIỆM="WBC", KQ="14,43"
         "SpO2 96%" → KQ="96%" (test name có thể ở TÊN_XÉT_NGHIỆM riêng)
         "ecg bình thường" → KQ="ecg bình thường" (kết luận ngắn)'''

n = src.count(old)
print(f'Found {n} occurrences of old KQ rule')
if n > 0:
    src = src.replace(old, new, 1)
    P.write_text(src, encoding='utf-8')
    print('Updated KQ rule')
else:
    print('Pattern not found - need different approach')