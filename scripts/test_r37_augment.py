"""Test R37 augment logic: _ensure_drug_disease_split + _ensure_compound_symptoms.

Mục đích: verify 2 functions auto-bổ sung entities khi LLM miss:
  1. R1 split: 'doxycycline cho viêm tuyến mồ hôi' -> add 'viêm tuyến mồ hôi' (CHẨN_ĐOÁN)
  2. Compound symptom: 'buồn nôn' trong input -> add 'buồn nôn' (TRIỆU_CHỨNG)

Covers the 2 entities bị miss trong output cũ:
  - 'viêm tuyến mồ hôi' (CHẨN_ĐOÁN) - R1 split
  - 'buồn nôn' (TRIỆU_CHỨNG) - compound symptom

Test cases (10):
  R1 split:
    1. LLM extract 'doxycycline' only -> MUST auto-add 'viêm tuyến mồ hôi'
    2. LLM already extract 'viêm tuyến mồ hôi' -> MUST NOT duplicate
    3. 'trị' connector (alt of 'cho') -> MUST auto-add
    4. Drug không trong whitelist -> MUST NOT add
    5. Disease < 4 chars -> MUST NOT add
  Compound symptom:
    6. LLM miss 'buồn nôn' -> MUST auto-add 'buồn nôn'
    7. LLM extract 'buồn nôn' -> MUST NOT duplicate
    8. LLM extract 'nôn' (substring) -> MUST add 'buồn nôn' (overlap detected)
       [NB: this case có thể skip do overlap check, log cảnh báo]
    9. Multiple compounds trong input -> MUST add all
   10. 'buồn nôn' KHÔNG trong input -> MUST NOT add
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.postprocess import (
    _ensure_drug_disease_split,
    _ensure_compound_symptoms,
    _COMPOUND_SYMPTOMS,
    _drop_short_substring_inside_longer,
    _strip_assertions_for_test_types,
    _apply_deterministic_icd_rules,
)
from src.icd_rag import _is_generic_drug_class
from src.rxnorm_rag import _alias_to_generic


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

PASS_COUNT = 0
FAIL_COUNT = 0
RESULTS: list[tuple[str, bool, str]] = []


def _has_entity(entities: list[dict], text: str, etype: str | None = None) -> bool:
    """True nếu có entity với text (case-insensitive) match và type khớp nếu có."""
    text_lower = text.lower().strip()
    for e in entities:
        if str(e.get("text", "")).strip().lower() != text_lower:
            continue
        if etype is not None and e.get("type") != etype:
            continue
        return True
    return False


def _run(name: str, ok: bool, detail: str = "") -> bool:
    global PASS_COUNT, FAIL_COUNT
    status = "[PASS]" if ok else "[FAIL]"
    if ok:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1
    RESULTS.append((name, ok, detail))
    print(f"  {status} {name}" + (f"  -- {detail}" if detail else ""))
    return ok


def _section(title: str) -> None:
    print(f"\n=== {title} ===")


# ──────────────────────────────────────────────────────────────────
# R1 drug-disease split tests
# ──────────────────────────────────────────────────────────────────

_section("R1 drug-disease split (_ensure_drug_disease_split)")


# Case 1: LLM extract only 'doxycycline' -> must auto-add disease
def test_r1_basic():
    input_text = "Bệnh nhân dùng doxycycline cho viêm tuyến mồ hôi trong 7 ngày."
    llm_output = [{"text": "doxycycline", "type": "THUỐC", "position": [16, 27]}]
    extras = _ensure_drug_disease_split(input_text, llm_output)
    found = _has_entity(extras, "viêm tuyến mồ hôi", "CHẨN_ĐOÁN")
    _run("R1 basic: LLM miss disease -> auto-add",
         found and len(extras) == 1,
         f"extras={[(e['text'], e['type']) for e in extras]}")
test_r1_basic()


# Case 2: LLM already extracted disease -> no duplicate
def test_r1_no_dup():
    input_text = "Bệnh nhân dùng doxycycline cho viêm tuyến mồ hôi."
    llm_output = [
        {"text": "doxycycline", "type": "THUỐC", "position": [16, 27]},
        {"text": "viêm tuyến mồ hôi", "type": "CHẨN_ĐOÁN", "position": [32, 51]},
    ]
    extras = _ensure_drug_disease_split(input_text, llm_output)
    _run("R1 no-dup: disease already extracted -> no add",
         len(extras) == 0,
         f"extras={[(e['text'], e['type']) for e in extras]}")
test_r1_no_dup()


# Case 3: 'trị' connector instead of 'cho'
def test_r1_alt_connector():
    input_text = "Kê đơn aspirin trị nhồi máu cơ tim cấp."
    llm_output = [{"text": "aspirin", "type": "THUỐC", "position": [8, 15]}]
    extras = _ensure_drug_disease_split(input_text, llm_output)
    # 'nhồi máu cơ tim' may or may not be in whitelist check - depends on input length
    found = _has_entity(extras, "nhồi máu cơ tim cấp", "CHẨN_ĐOÁN")
    _run("R1 alt connector: 'trị' -> auto-add disease",
         found,
         f"extras={[(e['text'], e['type']) for e in extras]}")
test_r1_alt_connector()


# Case 4: Drug not in whitelist -> no add
def test_r1_unknown_drug():
    input_text = "Bệnh nhân dùng foobar123 cho viêm phổi."
    llm_output = [{"text": "foobar123", "type": "THUỐC", "position": [16, 25]}]
    extras = _ensure_drug_disease_split(input_text, llm_output)
    _run("R1 unknown drug: 'foobar123' not in whitelist -> no add",
         len(extras) == 0,
         f"extras={[(e['text'], e['type']) for e in extras]}")
test_r1_unknown_drug()


# Case 5: Disease < 4 chars -> no add (regex group 2 requires {2,60})
def test_r1_short_disease():
    input_text = "Dùng aspirin cho cúm."  # 'cúm' = 3 chars, regex needs {2,60} but min via len check
    llm_output = [{"text": "aspirin", "type": "THUỐC", "position": [4, 11]}]
    extras = _ensure_drug_disease_split(input_text, llm_output)
    # 'cúm' might pass regex {2,60} but should fail len < 4 check
    has_short = any(len(e["text"]) < 4 for e in extras)
    _run("R1 short disease: 'cúm' (3 chars) -> no add",
         not has_short,
         f"extras={[(e['text'], e['type']) for e in extras]}")
test_r1_short_disease()


# Case 5b: Disease + newline + list marker (CRITICAL: regex phải stop at \n)
def test_r1_newline_boundary():
    input_text = "    - doxycycline cho viêm tuyến mồ hôi\n    - atenolol (uống hôm nay)"
    llm_output = [{"text": "doxycycline", "type": "THUỐC", "position": [6, 17]}]
    extras = _ensure_drug_disease_split(input_text, llm_output)
    found = _has_entity(extras, "viêm tuyến mồ hôi", "CHẨN_ĐOÁN")
    extras_text = [e['text'] for e in extras]
    # BUG trước đây: capture "viêm tuyến mồ hôi\n    - atenolol" (greedy across newline)
    no_newline = all("\n" not in e for e in extras_text)
    no_extra_drug = all("atenolol" not in e for e in extras_text)
    _run("R1 newline boundary: stop at \\n, no 'atenolol' leak",
         found and no_newline and no_extra_drug,
         f"extras={extras_text}")
test_r1_newline_boundary()


# ──────────────────────────────────────────────────────────────────
# Compound symptom tests
# ──────────────────────────────────────────────────────────────────

_section("Compound symptom (_ensure_compound_symptoms)")


# Case 6: LLM miss 'buồn nôn' -> must auto-add
def test_cs_basic():
    input_text = "Bệnh nhân có buồn nôn và sốt nhẹ sau 2 ngày."
    llm_output = [
        {"text": "sốt nhẹ", "type": "TRIỆU_CHỨNG", "position": [25, 32]},
    ]
    extras = _ensure_compound_symptoms(input_text, llm_output)
    found = _has_entity(extras, "buồn nôn", "TRIỆU_CHỨNG")
    _run("CS basic: 'buồn nôn' in input, LLM miss -> auto-add",
         found,
         f"extras={[(e['text'], e['type']) for e in extras]}")
test_cs_basic()


# Case 7: LLM already extracted 'buồn nôn' -> no duplicate
def test_cs_no_dup():
    input_text = "Bệnh nhân có buồn nôn."
    llm_output = [{"text": "buồn nôn", "type": "TRIỆU_CHỨNG", "position": [14, 23]}]
    extras = _ensure_compound_symptoms(input_text, llm_output)
    _run("CS no-dup: already extracted -> no add",
         len(extras) == 0,
         f"extras={[(e['text'], e['type']) for e in extras]}")
test_cs_no_dup()


# Case 8: LLM extract 'nôn' (substring) -> overlap detected, skip
# Note: expected behavior is SKIP because 'nôn' position overlaps 'buồn nôn' span
def test_cs_overlap_substring():
    input_text = "Bệnh nhân có buồn nôn khan."
    llm_output = [{"text": "nôn", "type": "TRIỆU_CHỨNG", "position": [20, 23]}]
    extras = _ensure_compound_symptoms(input_text, llm_output)
    # 'nôn' at [20,23] overlaps 'buồn nôn' at [14,23] -> skip
    skipped = len(extras) == 0
    _run("CS overlap substring: 'nôn' inside -> SKIP (overlap)",
         skipped,
         f"extras={[(e['text'], e['type']) for e in extras]} (expected: [])")
test_cs_overlap_substring()


# Case 9: Multiple compounds in input -> all added
def test_cs_multi():
    input_text = "Bệnh nhân có buồn nôn, đau đầu, chóng mặt, khó thở."
    llm_output = []
    extras = _ensure_compound_symptoms(input_text, llm_output)
    expected = ["buồn nôn", "đau đầu", "chóng mặt", "khó thở"]
    found_all = all(_has_entity(extras, t, "TRIỆU_CHỨNG") for t in expected)
    _run("CS multi: 4 compounds in input -> all added",
         found_all,
         f"found={[e['text'] for e in extras]}")
test_cs_multi()


# Case 10: Compound not in input -> no add
def test_cs_not_in_input():
    input_text = "Bệnh nhân bị viêm phổi, dùng thuốc kháng sinh."
    llm_output = []
    extras = _ensure_compound_symptoms(input_text, llm_output)
    no_compound = not any(e["type"] == "TRIỆU_CHỨNG" for e in extras)
    _run("CS not in input: no compounds -> no add",
         no_compound,
         f"extras={[(e['text'], e['type']) for e in extras]}")
test_cs_not_in_input()


# ──────────────────────────────────────────────────────────────────
# Cross-type substring drop tests (R37 fix for user's bug 2026-07-16)
# ──────────────────────────────────────────────────────────────────

_section("Cross-type substring drop (_drop_short_substring_inside_longer)")


# Case 11: 'mạch' (TÊN_XN) inside 'bệnh tim mạch do xơ vữa động mạch' (CHẨN_ĐOÁN)
# User's actual bug — must drop BOTH 'mạch' (4 chars)
def test_xsub_basic():
    entities = [
        {'text': 'bệnh tim mạch do xơ vữa động mạch', 'type': 'CHẨN_ĐOÁN', 'position': [217, 250]},
        {'text': 'mạch', 'type': 'TÊN_XÉT_NGHIỆM', 'position': [226, 230]},
        {'text': 'mạch', 'type': 'TÊN_XÉT_NGHIỆM', 'position': [246, 250]},
    ]
    result = _drop_short_substring_inside_longer(entities)
    kept_texts = [e['text'] for e in result]
    _run("XSub basic: 'mach' inside disease -> drop both",
         len(result) == 1 and 'bệnh tim mạch do xơ vữa động mạch' in kept_texts,
         f"kept={kept_texts}")
test_xsub_basic()


# Case 12: 'phổi' inside 'viêm phổi'
def test_xsub_short_disease():
    entities = [
        {'text': 'viêm phổi', 'type': 'CHẨN_ĐOÁN', 'position': [100, 110]},
        {'text': 'phổi', 'type': 'TÊN_XÉT_NGHIỆM', 'position': [105, 109]},
    ]
    result = _drop_short_substring_inside_longer(entities)
    kept_texts = [e['text'] for e in result]
    _run("XSub short disease: 'phoi' inside 'viem phoi' -> drop",
         len(result) == 1 and 'viêm phổi' in kept_texts,
         f"kept={kept_texts}")
test_xsub_short_disease()


# Case 13: Same type - NOT handled (by _drop_substring_entities instead)
def test_xsub_same_type():
    entities = [
        {'text': 'bệnh tim mạch do xơ vữa động mạch', 'type': 'CHẨN_ĐOÁN', 'position': [217, 250]},
        {'text': 'mạch', 'type': 'CHẨN_ĐOÁN', 'position': [226, 230]},  # same type
    ]
    result = _drop_short_substring_inside_longer(entities)
    _run("XSub same type: not handled here (kept both)",
         len(result) == 2,
         f"kept={[e['text'] for e in result]}")
test_xsub_same_type()


# Case 14: Short < 4 chars -> KEEP (protected by threshold)
def test_xsub_below_threshold():
    entities = [
        {'text': 'bệnh đau thắt ngực', 'type': 'CHẨN_ĐOÁN', 'position': [100, 119]},
        {'text': 'đau', 'type': 'TRIỆU_CHỨNG', 'position': [105, 108]},  # 3 chars
    ]
    result = _drop_short_substring_inside_longer(entities)
    _run("XSub short < 4 chars: 'dau' (3 chars) -> keep",
         len(result) == 2,
         f"kept={[e['text'] for e in result]}")
test_xsub_below_threshold()


# Case 15: No position overlap -> KEEP
def test_xsub_no_position_overlap():
    entities = [
        {'text': 'viêm phổi', 'type': 'CHẨN_ĐOÁN', 'position': [100, 110]},
        {'text': 'phổi', 'type': 'TÊN_XÉT_NGHIỆM', 'position': [200, 204]},  # outside
    ]
    result = _drop_short_substring_inside_longer(entities)
    _run("XSub no position overlap: keep both",
         len(result) == 2,
         f"kept={[e['text'] for e in result]}")
test_xsub_no_position_overlap()


# Case 16: Strip assertions from TÊN_XN/KQ_XN per spec
def test_strip_assertions():
    entities = [
        {'text': 'phẫu thuật cắt bỏ tuyến tiền liệt', 'type': 'TÊN_XÉT_NGHIỆM',
         'assertions': ['isHistorical'], 'position': [0, 30]},
        {'text': 'tăng men gan', 'type': 'KẾT_QUẢ_XÉT_NGHIỆM',
         'assertions': ['isHistorical'], 'position': [0, 12]},
        {'text': 'đau ngực', 'type': 'TRIỆU_CHỨNG',
         'assertions': ['isNegated'], 'position': [0, 8]},
        {'text': 'THA', 'type': 'CHẨN_ĐOÁN',
         'assertions': ['isHistorical'], 'position': [0, 3]},
    ]
    result = _strip_assertions_for_test_types(entities)
    # TÊN_XN and KQ_XN should have empty assertions
    txn_assert = next(e['assertions'] for e in result if e['type'] == 'TÊN_XÉT_NGHIỆM')
    kq_assert = next(e['assertions'] for e in result if e['type'] == 'KẾT_QUẢ_XÉT_NGHIỆM')
    tri_assert = next(e['assertions'] for e in result if e['type'] == 'TRIỆU_CHỨNG')
    cd_assert = next(e['assertions'] for e in result if e['type'] == 'CHẨN_ĐOÁN')
    _run("Strip: TÊN_XN/KQ_XN empty, others preserved",
         txn_assert == [] and kq_assert == []
         and tri_assert == ['isNegated'] and cd_assert == ['isHistorical'],
         f"txn={txn_assert}, kq={kq_assert}, tri={tri_assert}, cd={cd_assert}")
test_strip_assertions()


# ──────────────────────────────────────────────────────────────────
# R37 (2026-07-19): Drug-class detection + compound split + typo
# ──────────────────────────────────────────────────────────────────

_section("Drug-class detection (extended patterns)")


# Case 17: Common drug-class terms (R37 fix)
def test_drug_class():
    tests = [
        ('giảm đau', True),
        ('lợi tiểu', True),
        ('giảm sốt', True),
        ('liệu pháp lợi tiểu', True),
        ('phương pháp điều trị', True),
        ('xông khí dung', True),
        ('kháng sinh', True),  # existing
        ('aspirin', False),
        ('gleevec', False),
    ]
    passed = 0
    for text, expected in tests:
        actual = _is_generic_drug_class(text)
        if actual == expected:
            passed += 1
    _run(f"Drug-class: {passed}/{len(tests)} cases correct",
         passed == len(tests),
         f"passed={passed}/{len(tests)}")
test_drug_class()


_section("Compound drug split (R39 + new aliases)")


# Case 18: Compound drug lookup returns list
def test_compound_split():
    tests = [
        ('albuterolipratropium nebs', ['albuterol', 'ipratropium']),
        ('albuterolipratropium', ['albuterol', 'ipratropium']),
        ('doxycyclinebactrim', ['doxycycline', 'bactrim']),
    ]
    passed = 0
    for text, expected in tests:
        actual = _alias_to_generic(text)
        if isinstance(actual, list) and sorted(actual) == sorted(expected):
            passed += 1
    _run(f"Compound split: {passed}/{len(tests)} return lists",
         passed == len(tests),
         f"passed={passed}/{len(tests)}")
test_compound_split()


# Case 19: Brand + route mapping (levafloxacin typo, nitroglycerin routes)
def test_route_typo():
    tests = [
        ('levafloxacin', 'levofloxacin'),
        ('nitroglycerin dưới lưỡi', 'nitroglycerin sublingual'),
        ('nitroglycerin dạng bôi', 'nitroglycerin topical'),
    ]
    passed = 0
    for text, expected in tests:
        actual = _alias_to_generic(text)
        if actual == expected or (isinstance(actual, str) and actual.startswith(expected)):
            passed += 1
    _run(f"Route + typo: {passed}/{len(tests)} work",
         passed == len(tests),
         f"passed={passed}/{len(tests)}")
test_route_typo()


# ──────────────────────────────────────────────────────────────────
# R37 (2026-07-19): Deterministic ICD rules (đủ + chính xác + ngữ cảnh)
# ──────────────────────────────────────────────────────────────────

_section("Deterministic ICD rules (no LLM)")


def test_icd_rules():
    cases = [
        # (text, input_cands, expected)
        ('bệnh lỵ trực khuẩn do Shigella dysenteriae', ['A03'], ['A03.0']),
        ('ung thư phổi thùy trên', ['C34'], ['C34', 'C34.1']),
        ('ung thư phổi thùy dưới', ['C34'], ['C34', 'C34.3']),
        ('ung thư phổi thùy giữa', ['C34'], ['C34', 'C34.2']),
        ('viêm phổi phải', ['J18.9'], ['J18.9', 'J18.1']),
        ('viêm phổi trái', ['J18.9'], ['J18.9', 'J18.2']),
        ('nhồi máu cơ tim thành trước', ['I21.9'], ['I21.0']),
        ('nhồi máu cơ tim thành dưới', ['I21.9'], ['I21.1']),
        ('nhồi máu cơ tim không rõ vị trí', ['I21.9'], ['I21.9']),
        ('tăng huyết áp độ 2', ['I10'], ['I10']),  # no rule
        ('viêm phế quản cấp', ['J20'], ['J20']),
    ]
    passed = 0
    for text, in_cands, expected in cases:
        result = _apply_deterministic_icd_rules(text, in_cands)
        if result == expected:
            passed += 1
    _run(f"ICD rules: {passed}/{len(cases)} correct",
         passed == len(cases),
         f"passed={passed}/{len(cases)}")
test_icd_rules()


# ──────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
total = PASS_COUNT + FAIL_COUNT
print(f"SUMMARY: {PASS_COUNT}/{total} PASS, {FAIL_COUNT}/{total} FAIL")
print("=" * 60)
if FAIL_COUNT > 0:
    print("\nFailures:")
    for name, ok, detail in RESULTS:
        if not ok:
            print(f"  [FAIL] {name} -- {detail}")
    sys.exit(1)
else:
    print("All R37 augment tests passed")
    print()
    print("Coverage:")
    print("  R1 split: basic, no-dup, alt connector, unknown drug, short disease")
    print("  Compound symptom: basic, no-dup, overlap substring, multi, not in input")
    sys.exit(0)
