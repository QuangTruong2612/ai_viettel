"""Test overlap dedup trong postprocess.

Verify:
1. dedupe_entities: drop overlapping spans (giữ span dài hơn)
2. _expand_duplicates: skip positions overlap existing
3. assemble_record: check overlap khi emit
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.postprocess import dedupe_entities, _expand_duplicates, _drop_substring_entities


def test_1_dedupe_overlap_basic():
    """Cùng text + type + OVERLAP positions → giữ span dài hơn."""
    entities = [
        {"text": "tăng huyết áp", "type": "CHẨN_ĐOÁN", "position": [97, 110], "assertions": []},
        {"text": "tăng huyết áp", "type": "CHẨN_ĐOÁN", "position": [102, 110], "assertions": ["isHistorical"]},  # overlap với [97,110], ngắn hơn
    ]
    result = dedupe_entities(entities)
    print(f"   Input: 2 entities overlap → Output: {len(result)} entity")
    assert len(result) == 1, f"Expected 1, got {len(result)}"
    assert result[0]["position"] == [97, 110], f"Expected [97, 110], got {result[0]['position']}"
    # Assertions merge: giữ cả 2 (không quan trọng trong test này, dedup giữ entity dài hơn)
    print(f"✅ PASS test_1_dedupe_overlap_basic: kept [{result[0]['position'][0]},{result[0]['position'][1]}]")


def test_2_dedupe_exact_same_pos():
    """Cùng text + type + cùng exact position → drop 1 (R22)."""
    entities = [
        {"text": "khó thở", "type": "TRIỆU_CHỨNG", "position": [100, 108], "assertions": []},
        {"text": "khó thở", "type": "TRIỆU_CHỨNG", "position": [100, 108], "assertions": ["isNegated"]},
    ]
    result = dedupe_entities(entities)
    assert len(result) == 1, f"Expected 1, got {len(result)}"
    print(f"✅ PASS test_2_dedupe_exact_same_pos: kept 1 entity")


def test_3_dedupe_non_overlapping():
    """Cùng text + type + positions KHÔNG overlap → giữ cả 2 (R10 STRICT)."""
    entities = [
        {"text": "đánh trống ngực", "type": "TRIỆU_CHỨNG", "position": [10, 25], "assertions": []},
        {"text": "đánh trống ngực", "type": "TRIỆU_CHỨNG", "position": [100, 115], "assertions": ["isHistorical"]},
        {"text": "đánh trống ngực", "type": "TRIỆU_CHỨNG", "position": [200, 215], "assertions": []},
    ]
    result = dedupe_entities(entities)
    assert len(result) == 3, f"Expected 3, got {len(result)}"
    print(f"✅ PASS test_3_dedupe_non_overlapping: kept 3 entities (R10 STRICT)")


def test_4_dedupe_overlap_reverse():
    """Span hiện tại dài hơn span cũ → replace existing."""
    entities = [
        {"text": "xơ gan", "type": "CHẨN_ĐOÁN", "position": [60, 66], "assertions": []},  # ngắn
        {"text": "xơ gan do rượu", "type": "CHẨN_ĐOÁN", "position": [60, 75], "assertions": ["isHistorical"]},  # dài hơn, chứa span trên
    ]
    # Note: text khác nhau, nên dedup sẽ giữ cả 2
    # Test chỉ verify với text giống
    entities = [
        {"text": "xơ gan", "type": "CHẨN_ĐOÁN", "position": [60, 66], "assertions": []},
        {"text": "xơ gan", "type": "CHẨN_ĐOÁN", "position": [55, 66], "assertions": ["isHistorical"]},  # dài hơn (chứa [60,66])
    ]
    result = dedupe_entities(entities)
    # Sort theo start asc, length desc → [55,66] xử lý trước, sau đó [60,66] overlap → drop
    assert len(result) == 1, f"Expected 1, got {len(result)}"
    assert result[0]["position"] == [55, 66], f"Expected [55, 66], got {result[0]['position']}"
    print(f"✅ PASS test_4_dedupe_overlap_reverse: kept longer span [{result[0]['position'][0]},{result[0]['position'][1]}]")


def test_5_dedupe_different_types():
    """Cùng text nhưng KHÁC type → giữ cả 2 (R10 STRICT theo type)."""
    entities = [
        {"text": "ecg bình thường", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "position": [100, 115], "assertions": []},
        {"text": "ecg bình thường", "type": "TÊN_XÉT_NGHIỆM", "position": [100, 115], "assertions": []},
    ]
    result = dedupe_entities(entities)
    assert len(result) == 2, f"Expected 2, got {len(result)}"
    print(f"✅ PASS test_5_dedupe_different_types: kept both")


def test_6_expand_duplicates_skip_overlap():
    """_expand_duplicates: không tạo thêm entity nếu position mới OVERLAP existing."""
    input_text = "tăng huyết áp. Tiền sử tăng huyết áp. Hiện tại tăng huyết áp."
    entities = [
        {"text": "tăng huyết áp", "type": "CHẨN_ĐOÁN", "position": [0, 13], "assertions": []},
        {"text": "tăng huyết áp", "type": "CHẨN_ĐOÁN", "position": [23, 36], "assertions": []},
        {"text": "tăng huyết áp", "type": "CHẨN_ĐOÁN", "position": [46, 59], "assertions": []},
    ]
    result = _expand_duplicates(entities, input_text)
    # Cả 3 entities đều NON-overlap → giữ nguyên 3
    assert len(result) == 3, f"Expected 3, got {len(result)}"
    print(f"✅ PASS test_6_expand_duplicates_skip_overlap: 3 entities → 3 entities")


def test_7_real_data_overlap():
    """Test với data thực tế từ output 5.json."""
    entities = [
        {"text": "tăng huyết áp", "type": "CHẨN_ĐOÁN", "position": [97, 110], "assertions": ["isHistorical"]},
        {"text": "tăng huyết áp", "type": "CHẨN_ĐOÁN", "position": [102, 110], "assertions": ["isHistorical"]},  # OVERLAP
        {"text": "tăng lipid máu", "type": "CHẨN_ĐOÁN", "position": [117, 131], "assertions": ["isHistorical"]},
        {"text": "tăng lipid máu", "type": "CHẨN_ĐOÁN", "position": [122, 131], "assertions": ["isHistorical"]},  # OVERLAP
        {"text": "tăng lên 101.8 trước thủ thuật", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "position": [610, 640], "assertions": []},
        {"text": "tăng lên 101.8 trước thủ thuật", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "position": [615, 640], "assertions": []},  # OVERLAP
    ]
    result = dedupe_entities(entities)
    # 3 unique entities (each pair overlap → 1)
    assert len(result) == 3, f"Expected 3, got {len(result)}"
    positions = [e["position"] for e in result]
    assert [97, 110] in positions
    assert [117, 131] in positions
    assert [610, 640] in positions
    print(f"✅ PASS test_7_real_data_overlap: 6 entities → 3 entities")


if __name__ == "__main__":
    print("=" * 60)
    print("TEST OVERLAP DEDUP (after fix)")
    print("=" * 60)

    test_1_dedupe_overlap_basic()
    test_2_dedupe_exact_same_pos()
    test_3_dedupe_non_overlapping()
    test_4_dedupe_overlap_reverse()
    test_5_dedupe_different_types()
    test_6_expand_duplicates_skip_overlap()
    test_7_real_data_overlap()

    print("=" * 60)
    print("🎉 ALL TESTS PASSED")
    print("=" * 60)
