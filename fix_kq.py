"""Update KQ rule to clarify test/value separation."""
from pathlib import Path

P = Path(r'f:\AI_VIETTEL\src\prompts.py')
src = P.read_text(encoding='utf-8')

# Find and update the KQ rule
old_kq = '   - KẾT_QUẢ_XÉT_NGHIỆM: phải có số + đơn vị (vd "WBC 12.5 K/uL")'
new_kq = '   - KẾT_QUẢ_XÉT_NGHIỆM: CHỈ là số (vd "14,43", "76,4") hoặc kết luận ngắn. KHÔNG kèm tên test. Tên test tách riêng thành TÊN_XÉT_NGHIỆM. Phân tách theo dấu `:` hoặc `=` (vd "WBC:14,43" → TEN="WBC", KQ="14,43").'

n = src.count(old_kq)
print(f'Found {n} occurrences of old KQ rule')
if n > 0:
    src = src.replace(old_kq, new_kq, 1)
    P.write_text(src, encoding='utf-8')
    print('Updated KQ rule')
else:
    print('Pattern not found - showing lines around KQ:')
    for i, line in enumerate(src.split('\n')):
        if 'KẾT_QUẢ_XÉT' in line:
            print(f'  L{i}: {line}')