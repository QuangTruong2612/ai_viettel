"""Verify positions cho Ex 10 và Ex 12 trong prompts.py.

Chạy trên Kaggle để confirm các position tôi tính tay là đúng.
Nếu sai, in ra để tôi fix lại.
"""

# Ex 10 input (đã sửa)
ex10_input = "Bệnh nhân nam 60 tuổi nhập viện vì đánh trống ngực. Tiền sử đánh trống ngực 5 năm. Hiện tại đánh trống ngực nhiều hơn, kèm khó thở."

# Expected positions (đã tính tay)
ex10_expected = [
    (35, 50, "đánh trống ngực"),  # Occurrence 1
    (60, 75, "đánh trống ngực"),  # Occurrence 2
    (92, 107, "đánh trống ngực"),  # Occurrence 3
    (123, 130, "khó thở"),
]

print("=" * 60)
print("EX 10")
print("=" * 60)
print(f"Input length: {len(ex10_input)} chars")
print()
for start, end, expected_text in ex10_expected:
    actual = ex10_input[start:end]
    status = "✅" if actual == expected_text else "❌"
    print(f"{status} [{start}, {end}] = '{actual}' (expected: '{expected_text}')")

# Auto-find all occurrences để compare
print()
print("Auto-scan:")
import re
phrase = "đánh trống ngực"
positions = [m.start() for m in re.finditer(re.escape(phrase), ex10_input)]
print(f"  '{phrase}' appears {len(positions)} times at positions: {positions}")
for p in positions:
    end = p + len(phrase)
    print(f"    [{p}, {end}] = '{ex10_input[p:end]}'")

phrase = "khó thở"
positions = [m.start() for m in re.finditer(re.escape(phrase), ex10_input)]
print(f"  '{phrase}' appears {len(positions)} times at positions: {positions}")
for p in positions:
    end = p + len(phrase)
    print(f"    [{p}, {end}] = '{ex10_input[p:end]}'")

print()
print("=" * 60)
print("EX 12")
print("=" * 60)
ex12_input = "Bệnh nhân nam 58 tuổi vào viện vì đánh trống ngực, khó thở. Tiền sử đang dùng metoprolol 25mg. ECG: rung nhĩ, tần số thất 120 lần/phút. Chẩn đoán: rung nhĩ."

ex12_expected = [
    (34, 49, "đánh trống ngực"),
    (51, 58, "khó thở"),
    (78, 93, "metoprolol 25mg"),
    (100, 108, "rung nhĩ"),  # Occurrence 1 (ECG)
    (110, 134, "tần số thất 120 lần/phút"),
    (147, 155, "rung nhĩ"),  # Occurrence 2 (Chẩn đoán)
]

print(f"Input length: {len(ex12_input)} chars")
print()
for start, end, expected_text in ex12_expected:
    actual = ex12_input[start:end]
    status = "✅" if actual == expected_text else "❌"
    print(f"{status} [{start}, {end}] = '{actual}' (expected: '{expected_text}')")

# Auto-scan
print()
print("Auto-scan:")
phrase = "rung nhĩ"
positions = [m.start() for m in re.finditer(re.escape(phrase), ex12_input)]
print(f"  '{phrase}' appears {len(positions)} times at positions: {positions}")
for p in positions:
    end = p + len(phrase)
    print(f"    [{p}, {end}] = '{ex12_input[p:end]}'")

phrase = "đánh trống ngực"
positions = [m.start() for m in re.finditer(re.escape(phrase), ex12_input)]
print(f"  '{phrase}' appears {len(positions)} times at positions: {positions}")
for p in positions:
    end = p + len(phrase)
    print(f"    [{p}, {end}] = '{ex12_input[p:end]}'")

phrase = "khó thở"
positions = [m.start() for m in re.finditer(re.escape(phrase), ex12_input)]
print(f"  '{phrase}' appears {len(positions)} times at positions: {positions}")
for p in positions:
    end = p + len(phrase)
    print(f"    [{p}, {end}] = '{ex12_input[p:end]}'")

phrase = "metoprolol 25mg"
positions = [m.start() for m in re.finditer(re.escape(phrase), ex12_input)]
print(f"  '{phrase}' appears {len(positions)} times at positions: {positions}")
for p in positions:
    end = p + len(phrase)
    print(f"    [{p}, {end}] = '{ex12_input[p:end]}'")
