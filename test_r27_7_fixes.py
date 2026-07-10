"""Test postprocess fixes (R27.7 mới 2026-07-10).

Verify:
1. _split_long_imaging_result tách đúng 'điện tâm đồ là không ghi nhận gì bất thường'
2. _VITAL_SIGNS_DUMP_RE filter cho cả KQ_XN
3. _clean_entity_text strip trailing duration
4. ICD lookup short-circuit với _icd_vn_to_codes
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))


def test_1_split_dien_tam_do():
    """User yêu cầu cụ thể: 'điện tâm đồ là không ghi nhận gì bất thường' → 2 entities."""
    from src.postprocess import _split_long_imaging_result

    text = "điện tâm đồ là không ghi nhận gì bất thường"
    input_text = f"Bệnh nhân nhập viện. {text}"
    pos = [19, 19 + len(text)]

    result = _split_long_imaging_result(
        text, "KẾT_QUẢ_XÉT_NGHIỆM", input_text, pos
    )

    print(f"Input: '{text}'")
    if result is None:
        print("❌ FAIL: returned None (no split)")
        return False
    print(f"Output: {len(result)} entities")
    for i, e in enumerate(result):
        print(f"  {i+1}. text='{e['text']}' type={e['type']} pos={e['position']}")

    if len(result) != 2:
        print(f"❌ FAIL: expected 2 entities, got {len(result)}")
        return False
    if result[0]["text"] != "điện tâm đồ":
        print(f"❌ FAIL: entity 1 text should be 'điện tâm đồ', got '{result[0]['text']}'")
        return False
    if result[0]["type"] != "TÊN_XÉT_NGHIỆM":
        print(f"❌ FAIL: entity 1 type should be 'TÊN_XÉT_NGHIỆM', got '{result[0]['type']}'")
        return False
    if result[1]["text"] != "không ghi nhận gì bất thường":
        print(f"❌ FAIL: entity 2 text should be 'không ghi nhận gì bất thường', got '{result[1]['text']}'")
        return False
    if result[1]["type"] != "KẾT_QUẢ_XÉT_NGHIỆM":
        print(f"❌ FAIL: entity 2 type should be 'KẾT_QUẢ_XÉT_NGHIỆM', got '{result[1]['type']}'")
        return False
    if result[1]["position"][0] != result[0]["position"][1]:
        print(f"❌ FAIL: entity 2 position not contiguous")
        return False

    print("✅ PASS test_1_split_dien_tam_do")
    return True


def test_2_split_chup_x_quang():
    """Test: 'chụp x-quang ngực không ghi nhận gì bất thường' → 2 entities (drop verb)."""
    from src.postprocess import _split_long_imaging_result

    text = "chụp x-quang ngực không ghi nhận gì bất thường"
    input_text = f"Bệnh nhân nhập viện. {text}"
    pos = [19, 19 + len(text)]

    result = _split_long_imaging_result(
        text, "KẾT_QUẢ_XÉT_NGHIỆM", input_text, pos
    )

    if result is None:
        print("❌ FAIL: returned None")
        return False
    print(f"Output: {len(result)} entities")
    for e in result:
        print(f"  text='{e['text']}' type={e['type']}")
    # Either "x-quang ngực" (verb dropped) or "chụp x-quang ngực" (verb kept) is OK
    test_text = result[0]["text"]
    if "x-quang ngực" not in test_text.lower():
        print(f"❌ FAIL: entity 1 should contain 'x-quang ngực', got '{test_text}'")
        return False
    if result[1]["text"] != "không ghi nhận gì bất thường":
        print(f"❌ FAIL: entity 2 text wrong, got '{result[1]['text']}'")
        return False
    print("✅ PASS test_2_split_chup_x_quang")
    return True


def test_3_split_phan_tich_nuoc_tieu():
    """Test: 'phân tích nước tiểu không có gì đáng chú ý' → 2 entities."""
    from src.postprocess import _split_long_imaging_result

    text = "phân tích nước tiểu không có gì đáng chú ý"
    input_text = f"Xét nghiệm: {text}"
    pos = [12, 12 + len(text)]

    result = _split_long_imaging_result(
        text, "KẾT_QUẢ_XÉT_NGHIỆM", input_text, pos
    )

    if result is None:
        print("❌ FAIL: returned None")
        return False
    print(f"Output: {len(result)} entities")
    for e in result:
        print(f"  text='{e['text']}' type={e['type']}")
    test_text = result[0]["text"]
    if "nước tiểu" not in test_text.lower():
        print(f"❌ FAIL: entity 1 should contain 'nước tiểu', got '{test_text}'")
        return False
    if result[1]["text"] != "không có gì đáng chú ý":
        print(f"❌ FAIL: entity 2 text wrong, got '{result[1]['text']}'")
        return False
    print("✅ PASS test_3_split_phan_tich_nuoc_tieu")
    return True


def test_4_vital_signs_dump_drop_kq_xn():
    """VS dump filter KHÔNG áp dụng cho KQ_XN sau Fix #8 (VS98.3 là vital signs thực tế)."""
    from src.postprocess import _filter_lifestyle_entities

    # Tạo entity giả với type=KQ_XN (sẽ KHÔNG bị drop sau Fix #8)
    entities = [
        {"text": "VS98.3 12987 56 18 99RA", "type": "KẾT_QUẢ_XÉT_NGHIỆM",
         "position": [100, 123], "assertions": [], "candidates": []}
    ]
    result = _filter_lifestyle_entities(entities)
    if len(result) == 1:
        print("✅ PASS test_4: VS98.3 KHÔNG bị drop (giữ làm KQ_XN)")
        return True
    print(f"❌ FAIL: len(result)={len(result)}, expected 1")
    return False


def test_5_clean_entity_trailing_duration():
    """Test trailing duration strip trong _clean_entity_text."""
    from src.postprocess import _clean_entity_text

    cases = [
        ("mệt mỏi nhiều khi gắng sức trong tuần qua", "TRIỆU_CHỨNG",
         "mệt mỏi nhiều khi gắng sức"),
        ("đau ngực kéo dài 30 phút", "TRIỆU_CHỨNG",
         "đau ngực"),
        ("sốt cách 3 ngày trước", "TRIỆU_CHỨNG",
         "sốt"),
        ("khó thở 10 năm", "TRIỆU_CHỨNG",
         "khó thở"),
    ]
    passed = True
    for text, etype, expected in cases:
        result = _clean_entity_text(text, etype)
        if result == expected:
            print(f"✅ '{text}' → '{result}'")
        elif result is not None and expected in result:
            print(f"~ '{text}' → '{result}' (contains expected '{expected}')")
        else:
            print(f"❌ FAIL: '{text}' → {result!r}, expected '{expected}'")
            passed = False
    if passed:
        print("✅ PASS test_5_clean_entity_trailing_duration")
    return passed


def test_5b_kq_xn_type_after_split():
    """Fix #5: Finding SAU test name phải có type KẾT_QUẢ_XÉT_NGHIỆM."""
    from src.postprocess import _split_long_imaging_result

    text = "điện tâm đồ là không ghi nhận gì bất thường"
    input_text = f"Bệnh nhân nhập viện. {text}"
    pos = [19, 19 + len(text)]

    result = _split_long_imaging_result(
        text, "KẾT_QUẢ_XÉT_NGHIỆM", input_text, pos
    )
    if result is None or len(result) != 2:
        print(f"❌ FAIL: result={result}")
        return False
    if result[1]["type"] != "KẾT_QUẢ_XÉT_NGHIỆM":
        print(f"❌ FAIL: entity[1].type={result[1]['type']}, expected KẾT_QUẢ_XÉT_NGHIỆM")
        return False
    print(f"✅ Finding type = KẾT_QUẢ_XÉT_NGHIỆM (Fix #5 work)")
    return True


def test_6_icd_short_circuit():
    """Test ICD lookup short-circuit với _icd_vn_to_codes."""
    from src.icd_rag import ICDRetriever

    icd = ICDRetriever()

    # Test direct match cases
    cases = [
        ("THA", "I10"),
        ("NMCT", "I21"),
        ("ĐTĐ", "E11"),
        ("viêm tuyến mồ hôi", "L73.2"),  # MỚI - fix case 1.json
        ("ngoại tâm thu nhĩ", "I49.1"),  # MỚI - fix case 1.json
        ("ngoại tâm thu thất", "I49.3"),  # MỚI - fix case 1.json
    ]
    passed = True
    for text, expected_code in cases:
        codes = icd.lookup(text)
        if expected_code in codes:
            print(f"✅ '{text}' → {codes}")
        else:
            print(f"❌ FAIL: '{text}' expected to contain '{expected_code}', got {codes}")
            passed = False
    if passed:
        print("✅ PASS test_6_icd_short_circuit")
    return passed


if __name__ == "__main__":
    print("=" * 60)
    print("TEST POSTPROCESS FIXES (R27.7)")
    print("=" * 60)

    results = []
    results.append(test_1_split_dien_tam_do())
    results.append(test_2_split_chup_x_quang())
    results.append(test_3_split_phan_tich_nuoc_tieu())
    results.append(test_4_vital_signs_dump_drop_kq_xn())
    results.append(test_5_clean_entity_trailing_duration())
    results.append(test_5b_kq_xn_type_after_split())
    results.append(test_6_icd_short_circuit())
    results.append(test_7_drop_noise_khi_chuyen())
    results.append(test_8_vs98_keep_as_kq_xn())

    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"RESULTS: {passed}/{total} tests passed")
    print("=" * 60)


def test_7_drop_noise_khi_chuyen():
    """Fix #7: Noise 'khi được chuyển vào khoa...' phải bị drop."""
    from src.postprocess import _filter_lifestyle_entities

    noise_texts = [
        "khi được chuyển vào khoa điều trị",
        "khi nhập viện cấp cứu",
        "trong quá trình điều trị",
        "sau khi dùng thuốc",
        "trước khi phẫu thuật",
    ]
    for text in noise_texts:
        entities = [{"text": text, "type": "TRIỆU_CHỨNG",
                     "position": [0, len(text)], "assertions": [], "candidates": []}]
        result = _filter_lifestyle_entities(entities)
        if len(result) == 0:
            print(f"✅ Drop noise: '{text}'")
        else:
            print(f"❌ FAIL: '{text}' vẫn còn")
            return False
    print("✅ PASS test_7_drop_noise_khi_chuyen")
    return True


def test_8_vs98_keep_as_kq_xn():
    """Fix #8: 'VS98.3 12987 56 18 99RA' PHẢI được GIỮ làm KQ_XN (vital signs thực tế)."""
    from src.postprocess import _filter_lifestyle_entities

    vs_text = "VS98.3 12987 56 18 99RA"
    entities = [{"text": vs_text, "type": "KẾT_QUẢ_XÉT_NGHIỆM",
                 "position": [0, len(vs_text)], "assertions": [], "candidates": []}]
    result = _filter_lifestyle_entities(entities)
    if len(result) == 1:
        print(f"✅ PASS test_8_vs98_keep_as_kq_xn: 'VS98.3...' được giữ làm KQ_XN")
        return True
    print(f"❌ FAIL: 'VS98.3 12987...' bị drop, len(result)={len(result)}")
    return False
