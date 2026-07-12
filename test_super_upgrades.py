import sys
from pathlib import Path
import re

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.icd_rag import ICDRetriever, ICDIndex
from src.postprocess import _detect_assertions_from_context, _refine_stage2_results


def test_super_upgrade_1_clinical_normalizer_and_mappings():
    """Verify expanded clinical terms directly resolve to high-precision ICD-10 codes."""
    idx = ICDIndex()
    idx.add("I10", "essential hypertension")
    retriever = ICDRetriever(local_search=None, use_hybrid=False)
    retriever.idx = idx

    assert retriever.lookup("nhồi máu cơ tim không st chênh lên") == ["I21.4"]
    assert retriever.lookup("rung nhĩ kịch phát") == ["I48.0"]
    assert "J18" in retriever.lookup("viêm phổi mắc phải cộng đồng") or "J18.9" in retriever.lookup("viêm phổi mắc phải cộng đồng")
    print("✅ PASS Super-Upgrade 1: Clinical Normalizer & Direct Mappings")


def test_super_upgrade_2_strict_llm_fallback():
    """Verify Tier 7 LLM fallback triggers when all RAG tiers return [] and strictly validates codes."""
    idx = ICDIndex()
    idx.add("I21.4", "non-st elevation myocardial infarction")
    idx.add("J18.0", "bronchopneumonia")
    retriever = ICDRetriever(local_search=None, use_hybrid=False)
    retriever.idx = idx

    class FakeFallbackLLM:
        def call_sync(self, prompt, max_tokens=50, temperature=0.1):
            return "DỰA VÀO NGỮ CẢNH, MÃ CHẨN ĐOÁN LÀ I21.4 VÀ FAKE_CODE Z999.9"

    retriever._llm_client = FakeFallbackLLM()
    # Query completely unknown to index & dict
    res = retriever.lookup("cụm từ chẩn đoán lạ hoàn toàn chưa có trong từ điển r27")
    assert res == ["I21.4"], f"Expected ['I21.4'] but got {res}"
    print("✅ PASS Super-Upgrade 2: Strict LLM Fallback (Tier 7)")


def test_super_upgrade_3_clause_boundary_barriers():
    """Verify assertion detection does not leak across sentence/clause boundaries."""
    # 1. Family barrier test
    text_family = "Bố bệnh nhân bị tiền sử THA. Bệnh nhân hiện tại bị đái tháo đường type 2."
    pos_dtd = text_family.find("đái tháo đường")
    assert pos_dtd != -1
    assertions = _detect_assertions_from_context("đái tháo đường", text_family, "CHẨN_ĐOÁN", pos_dtd)
    assert "isFamily" not in assertions, f"Leaked isFamily across sentence barrier! Assertions: {assertions}"

    # 2. Negation barrier test
    text_neg = "Bệnh nhân không bị ho, nhưng có sốt cao 39 độ."
    pos_sot = text_neg.find("sốt cao")
    assert pos_sot != -1
    assertions_neg = _detect_assertions_from_context("sốt cao", text_neg, "TRIỆU_CHỨNG", pos_sot)
    assert "isNegated" not in assertions_neg, f"Leaked isNegated across clause barrier ('nhưng')! Assertions: {assertions_neg}"
    print("✅ PASS Super-Upgrade 3: Clause Boundary Barriers")


def test_super_upgrade_4_specificity_aware_picker():
    """Verify _select_adaptive_top_k prioritizes longer 5-char specific codes when specificity modifiers present."""
    retriever = ICDRetriever(local_search=None, use_hybrid=False)
    codes = ["I21", "I21.1", "I21.9"]
    picked = retriever._select_adaptive_top_k(codes, max_k=1, text="nhồi máu cơ tim vùng dưới")
    assert picked == ["I21.1"], f"Expected ['I21.1'] but got {picked}"
    print("✅ PASS Super-Upgrade 4: Specificity-Aware Picker")


def test_super_upgrade_5_rule_dominance_merge():
    """Verify Rule-Dominance Merge corrects false positives/negatives in assertions during refinement."""
    # Case A: Entity in 'Lý do nhập viện' but LLM hallucinated 'isHistorical' -> must be stripped
    note_ly_do = "Lý do nhập viện: ho khan kéo dài 3 ngày. Khám lâm sàng bình thường."
    pos_ho = note_ly_do.find("ho khan")
    ent_wrong = {
        "text": "ho khan",
        "type": "TRIỆU_CHỨNG",
        "position": [pos_ho, pos_ho + 7],
        "assertions": ["isHistorical"]
    }
    refined = _refine_stage2_results(note_ly_do, [ent_wrong])
    assert "isHistorical" not in refined[0]["assertions"], f"Failed to strip isHistorical in Ly do nhap vien: {refined[0]['assertions']}"

    # Case B: Entity in 'Tiền sử:' section but LLM missed 'isHistorical' -> must be added
    note_ts = "Tiền sử bệnh nội khoa:\n- Tăng huyết áp 5 năm\nHiện tại ổn định."
    pos_tha = note_ts.find("Tăng huyết áp")
    ent_missed = {
        "text": "Tăng huyết áp",
        "type": "CHẨN_ĐOÁN",
        "position": [pos_tha, pos_tha + 13],
        "assertions": []
    }
    refined_ts = _refine_stage2_results(note_ts, [ent_missed])
    assert "isHistorical" in refined_ts[0]["assertions"], f"Failed to auto-add isHistorical in Tien su: {refined_ts[0]['assertions']}"
    print("✅ PASS Super-Upgrade 5: Rule-Dominance Merge")


if __name__ == "__main__":
    print("============================================================")
    print("RUNNING 5 SUPER-UPGRADES VERIFICATION TESTS")
    print("============================================================")
    test_super_upgrade_1_clinical_normalizer_and_mappings()
    test_super_upgrade_2_strict_llm_fallback()
    test_super_upgrade_3_clause_boundary_barriers()
    test_super_upgrade_4_specificity_aware_picker()
    test_super_upgrade_5_rule_dominance_merge()
    print("============================================================")
    print("🎉 ALL 5 SUPER-UPGRADES PASSED WITH FLIGHT-READY PRECISION!")
    print("============================================================")
