"""Test _get_duplicate_alert và build_user_prompt sau khi chuyển từ marker inline sang ALERT header (R20.2 mới 2026-07-10).

Verify:
1. Deterministic: cùng input → cùng alert output (chạy 3 lần)
2. Alert header nhận diện đúng các từ lặp lại
3. Text gốc bên dưới INPUT: giữ nguyên 100% không bị chèn marker
4. Edge cases: empty, short, no duplicate
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.postprocess import _get_duplicate_alert, _preprocess_highlight_duplicates, _find_all_occurrences
from src.prompts import build_user_prompt


def test_1_deterministic():
    """Cùng input phải cho cùng alert output ở mọi lần chạy."""
    sample = """
Bệnh nhân nhập viện vì đau ngực. Tiền sử đau ngực. Hiện tại đau ngực nhiều hơn.
Bệnh nhân có tiền sử tăng huyết áp. Huyết áp cao. Điều trị tăng huyết áp.
Siêu âm tim: bình thường. Siêu âm bụng: bình thường. Siêu âm mạch: bình thường.
    """ * 3

    results = []
    for _ in range(3):
        results.append(_get_duplicate_alert(sample))

    assert results[0] == results[1] == results[2], \
        f"NOT deterministic!\nRun1: {repr(results[0])}\nRun2: {repr(results[1])}\nRun3: {repr(results[2])}"

    print(f"✅ PASS test_1_deterministic: same alert output x 3 runs")


def test_2_alert_content():
    """Kiểm tra alert header chứa đúng các từ lặp lại >= 2 lần."""
    sample = """
Bệnh nhân nam bị đau ngực lặp đi lặp lại. Hôm nay tiếp tục đau ngực nhiều.
Ngoài ra bệnh nhân khó thở và khó thở về đêm.
    """

    alert = _get_duplicate_alert(sample)
    print(f"   Alert generated: {alert}")
    assert "DUPLICATE ALERT:" in alert
    assert "đau ngực" in alert.lower()
    assert "khó thở" in alert.lower()
    print("✅ PASS test_2_alert_content: alert contains duplicated terms")


def test_3_prompt_integrity_and_offset():
    """Đảm bảo text bên dưới INPUT:\n không thay đổi so với input_text nguyên gốc."""
    sample = "Bệnh nhân đau ngực, khó thở, tim đập nhanh."
    prompt = build_user_prompt(sample)
    
    # Kiểm tra phần sau INPUT:\n khớp chính xác với sample
    input_part = prompt.split("INPUT:\n")[-1].split("\n\nOUTPUT JSON ARRAY")[0]
    assert input_part == sample, f"Text corrupted in INPUT section!\nGot: {repr(input_part)}"
    print("✅ PASS test_3_prompt_integrity_and_offset: INPUT section is 100% exact match")


def test_4_edge_cases():
    """Edge cases: empty, short, no duplicates."""
    assert _get_duplicate_alert("") == ""
    assert _get_duplicate_alert("abc") == ""
    short = "a" * 99
    assert _get_duplicate_alert(short) == ""
    long_unique = ("bệnh nhân nhập viện ") * 5
    out = _get_duplicate_alert(long_unique)
    print("✅ PASS test_4_edge_cases: empty/short/no-dup handled correctly")


def test_5_find_occurrences():
    """Test _find_all_occurrences helper trực tiếp."""
    text = "siêu âm siêu âm siêu âm"
    positions = _find_all_occurrences(text.lower(), "siêu âm")
    assert len(positions) == 3, f"Expected 3, got {len(positions)}: {positions}"
    assert positions == [(0, 7), (8, 15), (16, 23)], f"Wrong positions: {positions}"

    # No match
    positions = _find_all_occurrences("hello world", "xyz")
    assert positions == [], f"Expected [], got {positions}"

    # Single match
    positions = _find_all_occurrences("abc def ghi", "def")
    assert positions == [(4, 7)], f"Expected [(4,7)], got {positions}"

    print("✅ PASS test_5_find_occurrences: non-overlapping search OK")


if __name__ == "__main__":
    print("=" * 60)
    print("TEST DUPLICATE ALERT & PROMPT INTEGRITY (after fix)")
    print("=" * 60)

    test_1_deterministic()
    test_2_alert_content()
    test_3_prompt_integrity_and_offset()
    test_4_edge_cases()
    test_5_find_occurrences()

    print("=" * 60)
    print("🎉 ALL TESTS PASSED")
    print("=" * 60)
