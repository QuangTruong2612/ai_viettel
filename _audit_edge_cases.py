"""Comprehensive audit of pipeline edge cases for Vietnamese medical NER."""
import sys
sys.path.insert(0, 'F:/AI_VIETTEL')
for k in list(sys.modules):
    if k.startswith('src'): del sys.modules[k]

from src.postprocess import (
    assemble_record, validate_output, _split_drug_cho_pattern, sanitize_drug_text,
    _find_span, validate_positions, dedupe_entities,
)
from src.rxnorm_rag import RxNormRetriever, _drug_query_tokens
from src.icd_rag import ICDRetriever, ICD10VectorSearch, Translator
from src.prompts import SYSTEM_PROMPT, OUTPUT_SCHEMA, load_few_shot

# Setup
import src.inference as inf
ret = RxNormRetriever()
from src.llm_client import LLMClient
llm = LLMClient()

# Check if LLM server is running; if not, disable LLM to run offline audit instantly
llm_active = False
try:
    import socket
    from urllib.parse import urlparse
    parsed = urlparse(llm.config.base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 1234
    with socket.create_connection((host, port), timeout=0.5):
        llm_active = True
except Exception:
    llm = None

trans = Translator(llm_client=llm if llm_active else None)
local_search = ICD10VectorSearch()
icd = ICDRetriever(translator=trans, use_remote=False, local_search=local_search)

# ============================================================================
# EDGE CASES for diagnose postprocess — find what fails
# Each case: (name, input_text, raw_entities_from_LLM, expected_action)
# ============================================================================

CASES = [
    # ============= CASE 1: Drug prefix issue =============
    ("Drug listed w/o dose + generic words",
     "Bệnh nhân đang uống thuốc kháng sinh cho viêm phổi.",
     [
         {'text': 'thuốc kháng sinh', 'type': 'THUỐC', 'position': [16, 32], 'assertions': []},
     ],
     "Should DROP (mô tả chung, không phải thuốc cụ thể)"),

    # ============= CASE 2: Diagnosis + symptom duality =============
    ("chẩn đoán X, sau đó triệu chứng Y",
     "Chẩn đoán viêm phổi. Bệnh nhân có ho và sốt cao.",
     [
         {'text': 'viêm phổi', 'type': 'CHẨN_ĐOÁN', 'position': [11, 20], 'assertions': []},
         {'text': 'ho', 'type': 'TRIỆU_CHỨNG', 'position': [44, 46], 'assertions': []},
         {'text': 'sốt cao', 'type': 'TRIỆU_CHỨNG', 'position': [51, 57], 'assertions': []},
     ],
     "Should keep all 3, no candidates for symptoms"),

    # ============= CASE 3: Drug with full info =============
    ("Drug with full RxFormat",
     "Ceftriaxone 1g iv daily x 7 ngày cho viêm màng não.",
     [
         {'text': 'Ceftriaxone 1g iv daily x 7 ngày', 'type': 'THUỐC',
          'position': [0, 33], 'assertions': []},
         {'text': 'viêm màng não', 'type': 'CHẨN_ĐOÁN',
          'position': [38, 50], 'assertions': []},
     ],
     "Drug + Disease should both attach candidates"),

    # ============= CASE 4: Quoted / parens =============
    ("Drug in parens",
     "Đang dùng thuốc (paracetamol 500 mg) cho nhức đầu.",
     [
         {'text': 'paracetamol 500 mg', 'type': 'THUỐC',
          'position': [19, 37], 'assertions': []},
         {'text': 'nhức đầu', 'type': 'TRIỆU_CHỨNG',
          'position': [44, 52], 'assertions': []},
     ],
     "OK parens trong text được position đúng"),

    # ============= CASE 5: Drug + drug (no conjunction) =============
    ("Two drugs, no comma/space",
     "Bệnh nhân đang dùng AspirinClopidogrel.",
     [
         {'text': 'AspirinClopidogrel', 'type': 'THUỐC',
          'position': [16, 35], 'assertions': []},
     ],
     "Should this be split into 2 drugs? (separate names)"),

    # ============= CASE 6: Disease abbreviation =============
    ("Abbreviation only",
     "Bệnh nhân có tiền sử THA và ĐTĐ.",
     [
         {'text': 'THA', 'type': 'CHẨN_ĐOÁN', 'position': [21, 24], 'assertions': ['isHistorical']},
         {'text': 'ĐTĐ', 'type': 'CHẨN_ĐOÁN', 'position': [29, 32], 'assertions': ['isHistorical']},
     ],
     "Preset có sẵn → ICD lookup hit"),

    # ============= CASE 7: Test results =============
    ("Test + result inline",
     "Xét nghiệm máu: WBC 12 K/uL, Hgb 14 g/dL.",
     [
         {'text': 'Xét nghiệm máu', 'type': 'TÊN_XÉT_NGHIỆM', 'position': [0, 13], 'assertions': []},
         {'text': 'WBC 12 K/uL', 'type': 'KẾT_QUẢ_XÉT_NGHIỆM', 'position': [15, 27], 'assertions': []},
         {'text': 'Hgb 14 g/dL', 'type': 'KẾT_QUẢ_XÉT_NGHIỆM', 'position': [29, 41], 'assertions': []},
     ],
     "No candidates for test/types"),

    # ============= CASE 8: Negation =============
    ("Negation 'không'",
     "Bệnh nhân không sốt, không ho.",
     [
         {'text': 'sốt', 'type': 'TRIỆU_CHỨNG', 'position': [13, 16], 'assertions': ['isNegated']},
         {'text': 'ho', 'type': 'TRIỆU_CHỨNG', 'position': [26, 28], 'assertions': ['isNegated']},
     ],
     "isNegated properly preserved"),

    # ============= CASE 9: Disease as part of medical history =============
    ("Family history",
     "Bố bệnh nhân có tiền sử tăng huyết áp, mẹ bị đái tháo đường.",
     [
         {'text': 'tăng huyết áp', 'type': 'CHẨN_ĐOÁN', 'position': [22, 34], 'assertions': ['isFamily', 'isHistorical']},
         {'text': 'đái tháo đường', 'type': 'CHẨN_ĐOÁN', 'position': [49, 63], 'assertions': ['isFamily']},
     ],
     "Both flagged as isFamily"),

    # ============= CASE 10: Drug with stage suffix =============
    ("Disease with stage",
     "Bệnh nhân bị ung thư phổi giai đoạn IV.",
     [
         {'text': 'ung thư phổi giai đoạn IV', 'type': 'CHẨN_ĐOÁN',
          'position': [15, 38], 'assertions': []},
     ],
     "Lookup should handle 'giai đoạn IV' (stage IV)"),

    # ============= CASE 11: Generic disease mention =============
    ("Generic term",
     "Bệnh nhân đến khám vì đau đầu thường xuyên.",
     [
         {'text': 'đau đầu', 'type': 'TRIỆU_CHỨNG', 'position': [18, 25], 'assertions': []},
         {'text': 'khám', 'type': 'TRIỆU_CHỨNG', 'position': [9, 12], 'assertions': []},  # likely wrong
     ],
     "'khám' should be DROPPED — not a symptom"),

    # ============= CASE 12: Drug dose in parens =============
    ("Drug dose alternative",
     "Bệnh nhân dùng Coversyl (Perindopril) 5mg uống sáng.",
     [
         {'text': 'Coversyl (Perindopril) 5mg uống sáng', 'type': 'THUỐC',
          'position': [16, 50], 'assertions': []},
     ],
     "Brand name in parens — should lookup by generic name? Or both?"),

    # ============= CASE 13: Symptom vs Context =============
    ("'triệu chứng X' explicit",
     "Triệu chứng: nhức đầu và chóng mặt. Tiền sử: tăng huyết áp.",
     [
         {'text': 'nhức đầu', 'type': 'TRIỆU_CHỨNG', 'position': [12, 19], 'assertions': []},
         {'text': 'chóng mặt', 'type': 'TRIỆU_CHỨNG', 'position': [24, 32], 'assertions': []},
         {'text': 'tăng huyết áp', 'type': 'CHẨN_ĐOÁN', 'position': [47, 59], 'assertions': ['isHistorical']},
     ],
     "All should be properly recognized"),

    # ============= CASE 14: Multiple symptoms same time =============
    ("Multiple symptoms",
     "Bệnh nhân nhập viện vì đau bụng, buồn nôn, nôn, sốt 39°C.",
     [
         {'text': 'đau bụng', 'type': 'TRIỆU_CHỨNG', 'position': [22, 29], 'assertions': []},
         {'text': 'buồn nôn', 'type': 'TRIỆU_CHỨNG', 'position': [31, 38], 'assertions': []},
         {'text': 'nôn', 'type': 'TRIỆU_CHỨNG', 'position': [40, 42], 'assertions': []},
         {'text': 'sốt 39°C', 'type': 'TRIỆU_CHỨNG', 'position': [44, 52], 'assertions': []},
     ],
     "Each symptom separate, no candidates"),

    # ============= CASE 15: Drug combo with " + " =============
    ("Drug combo",
     "Bệnh nhân dùng Aspirin 81mg + Clopidogrel 75mg.",
     [
         {'text': 'Aspirin 81mg', 'type': 'THUỐC', 'position': [16, 28], 'assertions': []},
         {'text': 'Clopidogrel 75mg', 'type': 'THUỐC', 'position': [31, 47], 'assertions': []},
     ],
     "LLM đã tách sẵn → ok"),
]

print('=' * 70)
print('EDGE CASE AUDIT')
print('=' * 70)

failures = []
for i, (name, input_text, raw_ents, expectation) in enumerate(CASES, 1):
    final = assemble_record(input_text, raw_ents, ret, icd_retriever=icd, llm_client=llm)
    valid = validate_output(final)
    n_ent = len(final)
    n_drugs = sum(1 for e in final if e['type'] == 'THUỐC')
    n_diag = sum(1 for e in final if e['type'] == 'CHẨN_ĐOÁN')
    n_symp = sum(1 for e in final if e['type'] == 'TRIỆU_CHỨNG')
    n_cand = sum(1 for e in final if e.get('candidates'))
    print(f'\nCase {i}: {name}')
    print(f'  Expectation: {expectation}')
    print(f'  Result: {n_ent} entities ({n_drugs} drugs, {n_diag} diags, {n_symp} symps, {n_cand} with candidates)')
    for e in final:
        text = e['text'][:40]
        cand = e.get('candidates', [])
        cand_str = f' [{len(cand)} codes]' if cand else ''
        print(f'    [{e["type"]:14s}] "{text}"{cand_str}')
    if not valid:
        print(f'  ⚠️ SCHEMA INVALID')
        failures.append((i, name, 'schema invalid'))

print()
print('=' * 70)
print(f'DONE: {len(CASES) - len(failures)}/{len(CASES)} cases OK, {len(failures)} failures')
for i, name, reason in failures:
    print(f'  ✗ Case {i}: {name} -- {reason}')
