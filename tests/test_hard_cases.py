import json
from pathlib import Path
from src.postprocess import (
    _validate_stage1_mentions,
    _try_recover_position,
    _boost_and_split_stage1_mentions,
    _drop_substring_entities,
    _detect_assertions_from_context,
    _attach_candidates,
)
from src.icd_rag import ICDRetriever
from src.rxnorm_rag import RxNormRetriever

print("🔥 STARTING STRESS TEST: EXTREMELY HARD CLINICAL EDGE CASES...\n")

# Initialize RAG retrievers
icd_ret = ICDRetriever(Path("data/icd_index.json"))
rx_ret = RxNormRetriever(Path("data/rxnorm_index.json"))

# ---------------------------------------------------------
# CASE 1: Brutal Multi-Assertion + Acronym + Negation String
# ---------------------------------------------------------
text1 = (
    "Tiền sử gia đình: Bố bị THA, mẹ từng bị ĐTĐ tuýp 2 và RLLL. "
    "Tiền sử bản thân: Bệnh nhân có tiền sử NMCT cấp cách đây 3 năm, đang điều trị Aspirin 100mg x 1 tại nhà. "
    "Hiện tại vào viện vì đau thắt ngực sau xương ức. "
    "Bệnh nhân không ho, sốt cao, hay khó thở lúc nghỉ ngơi. "
    "Khám thấy ECG bình thường, loại trừ COPD và TBMMN."
)

print("=== CASE 1: COMPLEX ACRONYMS + FAMILY + HISTORICAL + NEGATION ===")
test_entities_c1 = [
    ("THA", "CHẨN_ĐOÁN"),
    ("ĐTĐ tuýp 2", "CHẨN_ĐOÁN"),
    ("RLLL", "CHẨN_ĐOÁN"),
    ("NMCT cấp", "CHẨN_ĐOÁN"),
    ("Aspirin 100mg x 1", "THUỐC"),
    ("ho", "TRIỆU_CHỨNG"),
    ("sốt cao", "TRIỆU_CHỨNG"),
    ("khó thở", "TRIỆU_CHỨNG"),
    ("ECG bình thường", "KẾT_QUẢ_XÉT_NGHIỆM"),
    ("COPD", "CHẨN_ĐOÁN"),
    ("TBMMN", "CHẨN_ĐOÁN"),
]

for ent_text, etype in test_entities_c1:
    pos = text1.find(ent_text)
    assert pos != -1, f"Missing in text: {ent_text}"
    assertions = _detect_assertions_from_context(ent_text, text1, etype, pos)
    cands = (
        icd_ret.lookup(ent_text)
        if etype == "CHẨN_ĐOÁN"
        else (rx_ret.lookup(ent_text) if etype == "THUỐC" else [])
    )
    print(
        f"[{etype:20s}] {ent_text:20s} -> Assertions: {str(assertions):32s} Candidates: {cands}"
    )

# Verify specific critical behaviors:
print("\nVerifying C1 Assertions & Candidates...")
assert "isFamily" in _detect_assertions_from_context(
    "THA", text1, "CHẨN_ĐOÁN", text1.find("THA")
), "THA must be isFamily!"
assert "isFamily" in _detect_assertions_from_context(
    "RLLL", text1, "CHẨN_ĐOÁN", text1.find("RLLL")
), "RLLL must be isFamily!"
assert "isHistorical" in _detect_assertions_from_context(
    "NMCT cấp", text1, "CHẨN_ĐOÁN", text1.find("NMCT cấp")
), "NMCT must be isHistorical!"
assert "isNegated" in _detect_assertions_from_context(
    "sốt cao", text1, "TRIỆU_CHỨNG", text1.find("sốt cao")
), "sốt cao must be isNegated inside comma list!"
assert (
    _detect_assertions_from_context(
        "ECG bình thường", text1, "KẾT_QUẢ_XÉT_NGHIỆM", text1.find("ECG bình thường")
    )
    == []
), "ECG bình thường MUST NOT be negated!"
assert icd_ret.lookup("COPD")[0].startswith("J44"), "COPD candidate must be J44!"
assert icd_ret.lookup("TBMMN")[0].startswith(("I63", "I64")), "TBMMN candidate must be I63 or I64!"

# ---------------------------------------------------------
# CASE 2: Tricky Substring & Degree Differentiation
# ---------------------------------------------------------
print("\n=== CASE 2: ZERO-HARDCODE SUBSTRING & DEGREE CHECK ===")
ents_c2 = [
    {"text": "suy tim độ II NYHA", "type": "CHẨN_ĐOÁN", "position": [10, 28]},
    {"text": "suy tim độ III NYHA", "type": "CHẨN_ĐOÁN", "position": [32, 51]},
]
dedup_c2 = _drop_substring_entities(ents_c2)
print(
    f'Before Dedup ({len(ents_c2)}): {[e["text"] for e in ents_c2]} -> After Dedup ({len(dedup_c2)}): {[e["text"] for e in dedup_c2]}'
)
assert (
    len(dedup_c2) == 2
), "Both degree II and degree III must be kept without being dropped!"

# ---------------------------------------------------------
# CASE 3: Hallucinated Offset Recovery
# ---------------------------------------------------------
print("\n=== CASE 3: HALLUCINATED OFFSET RECOVERY ===")
text3 = "Bệnh nhân thấy cảm giác đánh trống ngực dồn dập khi gắng sức."
fake_stage1 = [{"text": "đánh trống ngực", "position": [9999, 10000]}]
recovered = _validate_stage1_mentions(text3, fake_stage1)
print(
    f'Input with hallucinated [9999, 10000]: -> Recovered offset: {recovered[0]["position"]} (Exact text cut: "{text3[recovered[0]["position"][0]:recovered[0]["position"][1]]}")'
)
assert recovered[0]["position"] == [24, 39], "Must recover exact offset [24, 39]!"

print("\n🎉 ALL EXTREMELY HARD EDGE CASES PASSED PERFECTLY!")
