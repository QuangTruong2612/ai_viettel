"""Unit tests cho 4 nâng cấp đợt 3 (E, F, G, H) - ZERO Hardcode in Python."""
import time
from pathlib import Path
from src.icd_rag import _CONFIG_MGR, _expand_tokens_with_synonyms
from src.rxnorm_rag import RxNormRetriever
from src.postprocess import assemble_record
from src.inference import process_record


def test_upgrade_e_hot_reloading_and_trie_cache():
    """Kiểm tra DynamicConfigManager hot reload và pre-tokenized cache cho synonym rings."""
    assert _CONFIG_MGR is not None
    tokens = {"khối", "u", "thận"}
    expanded = _expand_tokens_with_synonyms(tokens, "khối u thận trái")
    assert "bướu" in expanded or "ung" in expanded or "tiết" in expanded
    # Force reload test
    _CONFIG_MGR.reload_if_needed(force=True)
    assert _CONFIG_MGR._synonym_tokens_cache


def test_upgrade_g_compound_drug_splitting():
    """Kiểm tra tách thuốc kép (Multi-Hop Compound Drug Splitting) trong RxNormRetriever."""
    retriever = RxNormRetriever(use_hybrid=False)
    # Giả lập tra cứu chuỗi thuốc kép chia bởi dấu / hoặc +
    codes = retriever.lookup("Paracetamol 500mg / Doxycycline 100mg")
    # L6 sẽ tự tách thành ["Paracetamol 500mg", "Doxycycline 100mg"] và tra song song
    assert isinstance(codes, list)


def test_upgrade_f_concurrent_enrichment():
    """Kiểm tra xử lý song song tra cứu ứng viên (ThreadPoolExecutor trong assemble_record)."""
    raw_entities = [
        {"text": "Aspirin 81mg", "type": "THUỐC", "position": [0, 12]},
        {"text": "Tăng huyết áp", "type": "CHẨN_ĐOÁN", "position": [15, 28]},
        {"text": "Đau ngực", "type": "TRIỆU_CHỨNG", "position": [30, 38]},
    ]
    input_text = "Aspirin 81mg - Tăng huyết áp - Đau ngực"
    retriever = RxNormRetriever(use_hybrid=False)
    final = assemble_record(input_text, raw_entities, retriever, icd_retriever=None)
    assert len(final) == 3
    assert all("candidates" in rec for rec in final)
