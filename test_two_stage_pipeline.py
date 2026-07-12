import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.postprocess import (
    _validate_stage1_mentions,
    _try_recover_position,
    _boost_and_split_stage1_mentions,
    _refine_stage2_results,
    _stage2_fallback_classify,
)
from src.prompts import (
    STAGE1_PROMPT,
    STAGE2_PROMPT,
    build_stage1_user_prompt,
    build_stage2_user_prompt,
    format_few_shot_stage2_messages,
)

def test_two_stage_prompts():
    assert len(STAGE1_PROMPT) > 500
    assert len(STAGE2_PROMPT) > 500
    s1_prompt = build_stage1_user_prompt("Bệnh nhân đau ngực")
    assert "INPUT:\nBệnh nhân đau ngực" in s1_prompt
    
    mentions = [{"text": "đau ngực", "position": [10, 18]}]
    s2_prompt = build_stage2_user_prompt("Bệnh nhân đau ngực và khó thở nhiều ngày qua", mentions)
    assert "- 1. text=\"đau ngực\" position=[10, 18]" in s2_prompt
    assert "ngữ cảnh:" in s2_prompt
    print("✅ PASS test_two_stage_prompts")

def test_stage1_validation_and_recovery():
    input_text = "Bệnh nhân có tăng huyết áp 10 năm. Tỉnh dậy thấy cháu gái hét lên vì cô ấy sẽ được phục vụ tốt hơn."
    raw_mentions = [
        {"text": "tăng huyết áp", "position": [13, 26]},
        {"text": "Tỉnh dậy thấy cháu gái hét lên", "position": [35, 65]},
        {"text": "cô ấy sẽ được phục vụ tốt hơn", "position": [69, 98]},
    ]
    validated = _validate_stage1_mentions(input_text, raw_mentions)
    valid_texts = [m["text"] for m in validated]
    assert "tăng huyết áp" in valid_texts
    assert not any("Tỉnh dậy" in t for t in valid_texts), f"Narrative noise not dropped: {valid_texts}"
    assert not any("cô ấy sẽ" in t for t in valid_texts), f"Narrative noise not dropped: {valid_texts}"
    print("✅ PASS test_stage1_validation_and_recovery")

def test_fuzzy_and_case_recovery():
    input_text = "Khám thấy ngoại tâm thu nhĩ và ST chênh lên V1-V4."
    raw_mentions = [
        {"text": "Ngoại Tâm Thu Nhĩ", "position": [0, 0]}, # wrong case and pos=0
        {"text": "ST chênh lên V1-V4", "position": [30, 48]},
    ]
    validated = _validate_stage1_mentions(input_text, raw_mentions)
    assert len(validated) == 2
    assert validated[0]["text"] == "ngoại tâm thu nhĩ"
    assert validated[0]["position"] == [10, 27]
    print("✅ PASS test_fuzzy_and_case_recovery")

def test_boost_split_and_refinement():
    input_text = "Bệnh nhân ghi nhận HA 160/90 mmHg, chỉ định điện tâm đồ (ECG) bình thường. Bệnh nhân không sốt."
    # 1. Test compound splitting and recall booster
    raw_mentions = [
        {"text": "điện tâm đồ (ECG)", "position": [44, 61]},
    ]
    boosted = _boost_and_split_stage1_mentions(input_text, raw_mentions)
    texts = [m["text"] for m in boosted]
    assert "HA 160/90 mmHg" in texts, f"Recall booster failed: {texts}"
    assert "điện tâm đồ" in texts and "ECG" in texts, f"Compound splitting failed: {texts}"

    # 2. Test refinement and assertion cross-validation
    stage2_entities = [
        {"text": "bình thường", "position": [63, 74], "type": "CHẨN_ĐOÁN", "assertions": ["isNegated"]},
        {"text": "sốt", "position": [91, 94], "type": "TRIỆU_CHỨNG", "assertions": []},
    ]
    refined = _refine_stage2_results(input_text, stage2_entities)
    assert refined[0]["type"] == "KẾT_QUẢ_XÉT_NGHIỆM" and "isNegated" not in refined[0]["assertions"]
    assert refined[1]["type"] == "TRIỆU_CHỨNG" and "isNegated" in refined[1]["assertions"]

    # 3. Test fallback classification
    fallback = _stage2_fallback_classify([{"text": "aspirin 100mg po daily", "position": [0, 22]}])
    assert fallback[0]["type"] == "THUỐC"
    print("✅ PASS test_boost_split_and_refinement")

if __name__ == "__main__":
    test_two_stage_prompts()
    test_stage1_validation_and_recovery()
    test_fuzzy_and_case_recovery()
    test_boost_split_and_refinement()
    print("🎉 ALL TWO-STAGE PIPELINE TESTS PASSED!")
