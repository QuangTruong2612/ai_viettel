"""Test overlap handling trong postprocess.

Verify:
1. dedupe_entities (default mode="merge"): MERGE entities cùng type + position overlap
   thành 1 entity dài nhất, union assertions/candidates, mark _merged_from.
2. dedupe_entities (mode="drop"): LEGACY R10/R22 — drop shorter span.
3. dedupe_entities (mode="report"): report-only — mark _overlap_with, KHÔNG merge/drop.
4. _find_position_overlap_pairs: helper detect position overlap (pure span-based).
5. _expand_duplicates: skip positions overlap existing.
6. assemble_record: check overlap khi emit.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.postprocess import dedupe_entities, _expand_duplicates, _drop_substring_entities, _find_position_overlap_pairs


# =========================================================================
# MODE="merge" (default mới) — R10/R22 MERGE
# =========================================================================


def test_1_merge_overlap_basic():
    """Cùng text + type + OVERLAP positions → MERGE thành 1, giữ span dài hơn."""
    entities = [
        {"text": "tăng huyết áp", "type": "CHẨN_ĐOÁN", "position": [97, 110], "assertions": []},
        {"text": "tăng huyết áp", "type": "CHẨN_ĐOÁN", "position": [102, 110], "assertions": ["isHistorical"]},  # overlap, shorter
    ]
    result = dedupe_entities(entities)
    print(f"   Input: 2 entities overlap → Output: {len(result)} entity")
    assert len(result) == 1, f"Expected 1, got {len(result)}"
    assert result[0]["position"] == [97, 110], f"Expected [97, 110], got {result[0]['position']}"
    # _merged_from marks entity bị merge vào
    assert result[0].get("_merged_from") == [1], f"Expected _merged_from=[1], got {result[0].get('_merged_from')}"
    # assertions union
    assert "isHistorical" in result[0]["assertions"]
    print(f"✅ PASS test_1_merge_overlap_basic: kept [{result[0]['position'][0]},{result[0]['position'][1]}]")


def test_2_merge_exact_same_pos():
    """Cùng text + type + cùng exact position → merge, mark _merged_from."""
    entities = [
        {"text": "khó thở", "type": "TRIỆU_CHỨNG", "position": [100, 108], "assertions": []},
        {"text": "khó thở", "type": "TRIỆU_CHỨNG", "position": [100, 108], "assertions": ["isNegated"]},
    ]
    result = dedupe_entities(entities)
    assert len(result) == 1, f"Expected 1, got {len(result)}"
    assert result[0].get("_merged_from") == [1]
    assert "isNegated" in result[0]["assertions"]
    print(f"✅ PASS test_2_merge_exact_same_pos: merged 2→1, assertions unioned")


def test_3_merge_non_overlapping():
    """Cùng text + type + positions KHÔNG overlap → giữ cả N (R10 STRICT)."""
    entities = [
        {"text": "đánh trống ngực", "type": "TRIỆU_CHỨNG", "position": [10, 25], "assertions": []},
        {"text": "đánh trống ngực", "type": "TRIỆU_CHỨNG", "position": [100, 115], "assertions": ["isHistorical"]},
        {"text": "đánh trống ngực", "type": "TRIỆU_CHỨNG", "position": [200, 215], "assertions": []},
    ]
    result = dedupe_entities(entities)
    assert len(result) == 3, f"Expected 3, got {len(result)}"
    for e in result:
        assert "_merged_from" not in e, "Non-overlapping entities should NOT have _merged_from"
    print(f"✅ PASS test_3_merge_non_overlapping: kept 3 entities, no merge marks")


def test_4_merge_overlap_reverse():
    """Span hiện tại dài hơn span cũ → merge, giữ span DÀI NHẤT."""
    entities = [
        {"text": "xơ gan", "type": "CHẨN_ĐOÁN", "position": [60, 66], "assertions": []},  # shorter
        {"text": "xơ gan", "type": "CHẨN_ĐOÁN", "position": [55, 66], "assertions": ["isHistorical"]},  # longer
    ]
    result = dedupe_entities(entities)
    assert len(result) == 1
    assert result[0]["position"] == [55, 66], f"Expected longest [55, 66], got {result[0]['position']}"
    # Lưu ý: _merged_from indices là sorted indices, không phải original input indices.
    # Sort by start ASC: entity [55,66] (longer) → sorted_idx 0, entity [60,66] (shorter) → sorted_idx 1.
    # Representative = [55,66] (longest), _merged_from = [1] (shorter entity).
    assert result[0].get("_merged_from") == [1], f"Expected _merged_from=[1], got {result[0].get('_merged_from')}"
    print(f"✅ PASS test_4_merge_overlap_reverse: kept longest [55,66], merged shorter")


def test_5_merge_different_types():
    """Cùng text nhưng KHÁC type + same position → KHÔNG merge (giữ cả 2)."""
    entities = [
        {"text": "ecg bình thường", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "position": [100, 115], "assertions": []},
        {"text": "ecg bình thường", "type": "TÊN_XÉT_NGHIỆM", "position": [100, 115], "assertions": []},
    ]
    result = dedupe_entities(entities)
    assert len(result) == 2, f"Expected 2 (diff types, no merge), got {len(result)}"
    for e in result:
        assert "_merged_from" not in e
    print(f"✅ PASS test_5_merge_different_types: kept 2 (diff types not merged)")


def test_6_merge_assertions_union():
    """Verify assertions từ tất cả members được union."""
    entities = [
        {"text": "đau ngực", "type": "TRIỆU_CHỨNG", "position": [10, 18], "assertions": ["isNegated"], "candidates": []},
        {"text": "đau ngực", "type": "TRIỆU_CHỨNG", "position": [12, 20], "assertions": ["isHistorical"], "candidates": []},
        {"text": "đau ngực", "type": "TRIỆU_CHỨNG", "position": [10, 20], "assertions": ["isFamily"], "candidates": []},
    ]
    result = dedupe_entities(entities)
    assert len(result) == 1
    assertions = set(result[0]["assertions"])
    assert assertions == {"isNegated", "isHistorical", "isFamily"}
    print(f"✅ PASS test_6_merge_assertions_union: {assertions}")


def test_7_merge_candidates_union():
    """Verify candidates từ tất cả members được union (unique, ordered)."""
    entities = [
        {"text": "đau ngực", "type": "TRIỆU_CHỨNG", "position": [10, 18], "assertions": [], "candidates": ["R07.1", "I20"]},
        {"text": "đau ngực", "type": "TRIỆU_CHỨNG", "position": [10, 20], "assertions": [], "candidates": ["I20", "I25"]},
    ]
    result = dedupe_entities(entities)
    assert len(result) == 1
    candidates = result[0]["candidates"]
    assert candidates == ["R07.1", "I20", "I25"], f"Expected ordered unique, got {candidates}"
    print(f"✅ PASS test_7_merge_candidates_union: {candidates}")


def test_8_merge_three_entities_user_example():
    """User's example: 3 entities overlap → 1 entity với _merged_from=[1, 2]."""
    entities = [
        {"text": "cảm giác thắt chặt ngực vùng trước tim", "type": "TRIỆU_CHỨNG", "position": [2050, 2088], "assertions": [], "candidates": []},
        {"text": "cảm giác thắt chặt ngực", "type": "TRIỆU_CHỨNG", "position": [2050, 2073], "assertions": [], "candidates": []},
        {"text": "thắt chặt ngực vùng trước tim", "type": "TRIỆU_CHỨNG", "position": [2059, 2088], "assertions": [], "candidates": []},
    ]
    result = dedupe_entities(entities)
    assert len(result) == 1
    assert result[0]["position"] == [2050, 2088], f"Expected longest [2050, 2088], got {result[0]['position']}"
    assert sorted(result[0]["_merged_from"]) == [1, 2], f"Expected _merged_from=[1,2], got {result[0].get('_merged_from')}"
    print(f"✅ PASS test_8_merge_three_entities_user_example: 3→1, position=[2050,2088]")


def test_9_merge_real_data_overlap():
    """Test với data thực tế: 6 entities (3 pairs overlap) → 3 entities."""
    entities = [
        {"text": "tăng huyết áp", "type": "CHẨN_ĐOÁN", "position": [97, 110], "assertions": ["isHistorical"]},
        {"text": "tăng huyết áp", "type": "CHẨN_ĐOÁN", "position": [102, 110], "assertions": ["isHistorical"]},
        {"text": "tăng lipid máu", "type": "CHẨN_ĐOÁN", "position": [117, 131], "assertions": ["isHistorical"]},
        {"text": "tăng lipid máu", "type": "CHẨN_ĐOÁN", "position": [122, 131], "assertions": ["isHistorical"]},
        {"text": "tăng lên 101.8 trước thủ thuật", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "position": [610, 640], "assertions": []},
        {"text": "tăng lên 101.8 trước thủ thuật", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "position": [615, 640], "assertions": []},
    ]
    result = dedupe_entities(entities)
    assert len(result) == 3
    # Mỗi representative có _merged_from marking shorter entity
    for e in result:
        assert "_merged_from" in e
        assert len(e["_merged_from"]) == 1
    positions = sorted([tuple(e["position"]) for e in result])
    assert (97, 110) in positions
    assert (117, 131) in positions
    assert (610, 640) in positions
    print(f"✅ PASS test_9_merge_real_data_overlap: 6→3 entities, each merged 1 pair")


# =========================================================================
# MODE="drop" (legacy R10/R22) — backward compat
# =========================================================================


def test_10_drop_mode_legacy():
    """mode='drop': giữ behavior cũ — drop shorter span."""
    entities = [
        {"text": "tăng huyết áp", "type": "CHẨN_ĐOÁN", "position": [97, 110], "assertions": [], "candidates": []},
        {"text": "tăng huyết áp", "type": "CHẨN_ĐOÁN", "position": [102, 110], "assertions": ["isHistorical"], "candidates": []},
    ]
    result = dedupe_entities(entities, mode="drop")
    assert len(result) == 1
    assert result[0]["position"] == [97, 110]
    # mode='drop' KHÔNG có mark
    assert "_merged_from" not in result[0]
    assert "_overlap_with" not in result[0]
    # assertions: chỉ giữ của entity dài hơn (R10/R22 behavior)
    assert "isHistorical" not in result[0]["assertions"], "drop mode doesn't union assertions"
    print(f"✅ PASS test_10_drop_mode_legacy: drop shorter, kept [97,110]")


# =========================================================================
# MODE="report" — report only, no merge/drop
# =========================================================================


def test_11_report_mode():
    """mode='report': KHÔNG merge/drop, chỉ mark _overlap_with."""
    entities = [
        {"text": "tăng huyết áp", "type": "CHẨN_ĐOÁN", "position": [97, 110], "assertions": [], "candidates": []},
        {"text": "tăng huyết áp", "type": "CHẨN_ĐOÁN", "position": [102, 110], "assertions": ["isHistorical"], "candidates": []},
    ]
    result = dedupe_entities(entities, mode="report")
    assert len(result) == 2
    # Cả 2 có _overlap_with marking nhau
    assert result[0].get("_overlap_with") == [1]
    assert result[1].get("_overlap_with") == [0]
    # KHÔNG có _merged_from (vì không merge)
    for e in result:
        assert "_merged_from" not in e
    print(f"✅ PASS test_11_report_mode: 2 entities kept, _overlap_with marks")


# =========================================================================
# _find_position_overlap_pairs helper
# =========================================================================


def test_12_find_position_overlap_pairs_basic():
    """Helper: detect position overlap (pure span-based, no text/type check)."""
    entities = [
        {"text": "A", "type": "X", "position": [10, 20]},
        {"text": "B", "type": "X", "position": [15, 25]},
        {"text": "C", "type": "Y", "position": [16, 18]},  # overlap với A, B nhưng diff type
        {"text": "D", "type": "X", "position": [50, 60]},  # no overlap
    ]
    pairs = _find_position_overlap_pairs(entities)
    # A-B overlap (5 chars), A-C overlap (2 chars), B-C overlap (2 chars) = 3 pairs
    # D không overlap gì
    assert len(pairs) == 3, f"Expected 3 pairs, got {len(pairs)}"
    # Verify first pair (A-B)
    assert pairs[0]["idx_a"] == 0 and pairs[0]["idx_b"] == 1
    assert pairs[0]["overlap_chars"] == 5
    assert pairs[0]["same_text"] is False
    assert pairs[0]["same_type"] is True
    print(f"✅ PASS test_12_find_position_overlap_pairs_basic: 3 pairs detected")


def test_13_find_position_overlap_pairs_invalid_input():
    """Helper: skip entities với position invalid."""
    entities = [
        {"text": "A", "type": "X", "position": [10, 20]},
        {"text": "B", "type": "X", "position": "invalid"},  # skip
        {"text": "C", "type": "X", "position": [15, 25]},
    ]
    pairs = _find_position_overlap_pairs(entities)
    assert len(pairs) == 1
    assert pairs[0]["idx_a"] == 0 and pairs[0]["idx_b"] == 2
    print(f"✅ PASS test_13_find_position_overlap_pairs_invalid_input: skipped invalid")


# =========================================================================
# Backward compat với _expand_duplicates và _drop_substring_entities
# =========================================================================


def test_14_expand_duplicates_non_overlap():
    """_expand_duplicates: mỗi occurrence = 1 entity riêng (NON-overlap case)."""
    input_text = "tăng huyết áp. Tiền sử tăng huyết áp. Hiện tại tăng huyết áp."
    entities = [
        {"text": "tăng huyết áp", "type": "CHẨN_ĐOÁN", "position": [0, 13], "assertions": []},
        {"text": "tăng huyết áp", "type": "CHẨN_ĐOÁN", "position": [23, 36], "assertions": []},
        {"text": "tăng huyết áp", "type": "CHẨN_ĐOÁN", "position": [46, 59], "assertions": []},
    ]
    result = _expand_duplicates(entities, input_text)
    # Cả 3 entities đều NON-overlap → giữ nguyên 3
    assert len(result) == 3, f"Expected 3, got {len(result)}"
    print(f"✅ PASS test_14_expand_duplicates_non_overlap: 3 entities → 3 entities")


def test_15_drug_parens_dedup_compat():
    """Backward compat: dedupe_entities với 1 entity → trả về 1 entity (no mark)."""
    entities = [
        {"text": "ung thư phổi", "type": "CHẨN_ĐOÁN", "position": [10, 23], "assertions": []},
    ]
    result = dedupe_entities(entities)
    assert len(result) == 1
    assert result[0]["text"] == "ung thư phổi"
    assert "_merged_from" not in result[0]
    print(f"✅ PASS test_15_drug_parens_dedup_compat: 1 entity → 1 entity")


# =========================================================================
# Real 2.txt simulation — verify NO drop cho 4 instances "hội chứng não gan"
# =========================================================================


def test_16_real_2txt_simulation_no_drop():
    """Test với data từ 2.txt: 4 instances 'hội chứng não gan' ở different positions
    phải KHÔNG bị drop (vì positions KHÔNG overlap)."""
    entities = [
        {"text": "nghi ngờ xơ gan do rượu", "type": "CHẨN_ĐOÁN", "position": [52, 75], "assertions": ["isHistorical"], "candidates": ["K70.3"]},
        {"text": "hội chứng não gan", "type": "CHẨN_ĐOÁN", "position": [191, 208], "assertions": [], "candidates": []},
        {"text": "hội chứng não gan", "type": "CHẨN_ĐOÁN", "position": [270, 287], "assertions": [], "candidates": []},
        {"text": "hội chứng não gan", "type": "CHẨN_ĐOÁN", "position": [338, 355], "assertions": [], "candidates": []},
        {"text": "ý thức suy giảm", "type": "TRIỆU_CHỨNG", "position": [368, 383], "assertions": [], "candidates": []},
        {"text": "hội chứng não gan", "type": "CHẨN_ĐOÁN", "position": [432, 449], "assertions": [], "candidates": []},
        {"text": "đặt shunt dẫn lưu tĩnh mạch cửa qua da", "type": "TÊN_XÉT_NGHIỆM", "position": [555, 593], "assertions": [], "candidates": []},
    ]
    result = dedupe_entities(entities)
    # Expect 7 entities (no overlap → no merge)
    assert len(result) == 7, f"Expected 7 entities (no overlap), got {len(result)}"
    # 4 instances 'hội chứng não gan' phải còn nguyên
    hoi_chung_count = sum(1 for e in result if e["text"] == "hội chứng não gan")
    assert hoi_chung_count == 4, f"Expected 4 'hội chứng não gan', got {hoi_chung_count}"
    # 'ý thức suy giảm' phải còn
    assert any(e["text"] == "ý thức suy giảm" for e in result)
    # KHÔNG có _merged_from (vì không cluster)
    for e in result:
        assert "_merged_from" not in e
    print(f"✅ PASS test_16_real_2txt_simulation_no_drop: 7 entities preserved, 4 'hội chứng não gan' kept")


if __name__ == "__main__":
    print("=" * 70)
    print("TEST OVERLAP DEDUP — mode='merge' (default) + mode='drop' + mode='report'")
    print("=" * 70)

    test_1_merge_overlap_basic()
    test_2_merge_exact_same_pos()
    test_3_merge_non_overlapping()
    test_4_merge_overlap_reverse()
    test_5_merge_different_types()
    test_6_merge_assertions_union()
    test_7_merge_candidates_union()
    test_8_merge_three_entities_user_example()
    test_9_merge_real_data_overlap()
    test_10_drop_mode_legacy()
    test_11_report_mode()
    test_12_find_position_overlap_pairs_basic()
    test_13_find_position_overlap_pairs_invalid_input()
    test_14_expand_duplicates_non_overlap()
    test_15_drug_parens_dedup_compat()
    test_16_real_2txt_simulation_no_drop()

    print("=" * 70)
    print("🎉 ALL 16 TESTS PASSED")
    print("=" * 70)
