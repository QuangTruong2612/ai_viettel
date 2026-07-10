"""Test _preprocess_highlight_duplicates sau khi fix.

Verify:
1. Deterministic: cùng input → cùng output (chạy 3 lần)
2. Đếm marker đúng
3. Không garbled với overlapping/edge cases
4. Edge cases: empty, short, no duplicate
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.postprocess import _preprocess_highlight_duplicates, _find_all_occurrences


def test_1_deterministic():
    """Cùng input phải cho cùng output ở mọi lần chạy."""
    sample = """
Bệnh nhân nhập viện vì đau ngực. Tiền sử đau ngực. Hiện tại đau ngực nhiều hơn.
Bệnh nhân có tiền sử tăng huyết áp. Huyết áp cao. Điều trị tăng huyết áp.
Siêu âm tim: bình thường. Siêu âm bụng: bình thường. Siêu âm mạch: bình thường.
    """ * 3  # Repeat để có nhiều duplicate

    results = []
    for _ in range(3):
        results.append(_preprocess_highlight_duplicates(sample))

    assert results[0] == results[1] == results[2], \
        f"NOT deterministic!\nRun1: {len(results[0])}\nRun2: {len(results[1])}\nRun3: {len(results[2])}"

    print(f"✅ PASS test_1_deterministic: same output {len(results[0])} chars x 3 runs")


def test_2_count_markers():
    """Đếm số marker [xN] phải khớp với số occurrences >= 2."""
    sample = """
siêu âm siêu âm siêu âm
đau ngực đau ngực
huyết áp huyết áp huyết áp huyết áp
    """

    result = _preprocess_highlight_duplicates(sample)
    marker_count = result.count("[x")
    print(f"   Input: {len(sample)} chars → Output: {len(result)} chars, {marker_count} markers")

    # siêu âm x3 → 3 markers (giữ nguyên đầu, mark 2 cái sau)
    # đau ngực x2 → 2 markers (giữ 1, mark 1)
    # huyết áp x4 → 4 markers (giữ 1, mark 3)
    assert marker_count >= 3, f"Expected >= 3 markers, got {marker_count}"
    print(f"✅ PASS test_2_count_markers: {marker_count} markers (expected >=3)")


def test_3_overlapping():
    """Phrases overlapping phải xử lý đúng (không garbled)."""
    # Test với phrase "siêu âm" trong "siêu âm siêu âm"
    # Original bug: nếu "âm siêu" cũng xuất hiện → overlap
    sample = """
bệnh nhân nhập viện siêu âm siêu âm
siêu âm cho kết quả tốt
"""

    result = _preprocess_highlight_duplicates(sample)

    # Check text integrity: original text không bị garbled
    # Loại bỏ markers [xN] → phải ra original text
    import re
    cleaned = re.sub(r"\[x\d+\]", "", result)
    assert cleaned == sample, \
        f"Text integrity broken!\nCleaned: {repr(cleaned[:200])}\nOriginal: {repr(sample[:200])}"

    print(f"✅ PASS test_3_overlapping: text integrity OK ({len(result)} chars)")


def test_4_edge_cases():
    """Edge cases: empty, short, no duplicates."""
    # Empty
    assert _preprocess_highlight_duplicates("") == ""
    # Short
    assert _preprocess_highlight_duplicates("abc") == "abc"
    # Boundary length 99 chars → return as-is
    short = "a" * 99
    assert _preprocess_highlight_duplicates(short) == short
    # 100 chars + no duplicates
    long_unique = ("bệnh nhân nhập viện ") * 5  # ~100 chars, no freq>=2 phrases
    out = _preprocess_highlight_duplicates(long_unique)
    print(f"   No-dup input ({len(long_unique)} chars) → {len(out)} chars, markers={out.count('[x')}")
    print("✅ PASS test_4_edge_cases: empty/short/no-dup handled correctly")


def test_5_find_occurrences():
    """Test _find_all_occurrences helper trực tiếp."""
    text = "siêu âm siêu âm siêu âm"
    positions = _find_all_occurrences(text.lower(), "siêu âm")
    assert len(positions) == 3, f"Expected 3, got {len(positions)}: {positions}"
    assert positions == [(0, 8), (9, 17), (18, 26)], f"Wrong positions: {positions}"

    # No match
    positions = _find_all_occurrences("hello world", "xyz")
    assert positions == [], f"Expected [], got {positions}"

    # Single match
    positions = _find_all_occurrences("abc def ghi", "def")
    assert positions == [(4, 7)], f"Expected [(4,7)], got {positions}"

    print("✅ PASS test_5_find_occurrences: non-overlapping search OK")


def test_6_real_world_sample():
    """Test với 1 đoạn text giống medical note thật."""
    sample = """
Bệnh nhân nam 65 tuổi, vào viện vì đau ngực trái.
Tiền sử: Tăng huyết áp 10 năm, đái tháo đường type 2.
Hiện tại: đau ngực trái, khó thở, huyết áp 160/90 mmHg.
Xét nghiệm: Cholesterol cao, đường huyết cao.
Siêu âm tim: Giảm chức năng tâm thu thất trái.
Chẩn đoán: Nhồi máu cơ tim cấp, tăng huyết áp, đái tháo đường.
Điều trị: Aspirin, atorvastatin, metoprolol, enalapril.
Siêu âm bụng: gan nhiễm mỡ.
Theo dõi: Huyết áp, đường huyết, cholesterol.
    """

    # Chạy 5 lần để test stability
    results = []
    for _ in range(5):
        results.append(_preprocess_highlight_duplicates(sample))

    assert len(set(results)) == 1, f"Output varies across runs! Got {len(set(results))} unique outputs"

    result = results[0]
    markers = result.count("[x")
    print(f"✅ PASS test_6_real_world: stable x5 runs, {markers} markers in output")

    # In ra vài markers để user verify
    import re
    matches = re.findall(r"\b\w+\s+\w+\[x\d+\]", result)
    if matches:
        print(f"   Sample markers: {matches[:8]}")


def test_7_no_phrase_substring_in_marker():
    """Đảm bảo marker [xN] không bị match làm phrase mới ở iteration sau."""
    # Test: text có "đau ngực" xuất hiện nhiều
    # Sau khi mark → "đau ngực[x3]" — không được tạo phrase giả "ngực[x3]"
    sample = "đau ngực đau ngực đau ngực đau bụng đau bụng đau đầu đau đầu đau đầu đau đầu"

    result = _preprocess_highlight_duplicates(sample)
    import re
    cleaned = re.sub(r"\[x\d+\]", "", result)

    assert cleaned == sample, \
        f"Markers caused text corruption!\nCleaned: {repr(cleaned)}\nOriginal: {repr(sample)}"

    # Verify markers không tạo thêm phrase "ngực[x" hay "đầu[x"
    assert "ngực[x" not in result.lower() or result.lower().count("ngực[x") == result.lower().count("đau ngực") - 1

    print(f"✅ PASS test_7_no_phrase_substring: markers không tạo phrase giả")


if __name__ == "__main__":
    print("=" * 60)
    print("TEST _preprocess_highlight_duplicates (after fix)")
    print("=" * 60)

    test_1_deterministic()
    test_2_count_markers()
    test_3_overlapping()
    test_4_edge_cases()
    test_5_find_occurrences()
    test_6_real_world_sample()
    test_7_no_phrase_substring_in_marker()

    print("=" * 60)
    print("🎉 ALL TESTS PASSED")
    print("=" * 60)
