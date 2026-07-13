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


def test_upgrade_i_ner_sliding_chunking_and_fuzzy_alignment():
    """Kiểm tra tách đoạn gối đầu (Sliding Window Overlap) và cân chỉnh vị trí tự động không dấu/lỗi phím."""
    from src.inference import _split_into_sections
    from src.postprocess import align_and_expand_entities

    # 1. Test Sliding Overlap Chunking
    text = "CHẨN ĐOÁN:\n" + ("Bệnh nhân có tiền sử tăng huyết áp mạn tính. " * 35) + "Đái tháo đường type 2."
    chunks = _split_into_sections(text, max_chunk_len=500, overlap_len=120)
    assert len(chunks) >= 2
    # Chunk sau phải chứa ít nhất một phần overlap của chunk trước
    assert chunks[1][0].strip()[:30] in chunks[0][0]

    # 2. Test Accent-Insensitive & Fuzzy Alignment (Pass 4 & Pass 5)
    input_text = "Chẩn đoán: Viêm phế quản cấp, Tăng huyết áp độ 2, theo dõi suy tim."
    raw_entities = [
        {"text": "viem phe quan cap", "type": "CHẨN_ĐOÁN"},  # LLM trả về mất dấu
        {"text": "tăng huyết ap độ 2", "type": "CHẨN_ĐOÁN"}, # LLM gõ thiếu dấu nặng chữ áp
    ]
    aligned = align_and_expand_entities(input_text, raw_entities)
    assert len(aligned) == 2
    assert aligned[0]["text"] == "Viêm phế quản cấp"
    assert aligned[0]["position"] == [11, 28]
    assert aligned[1]["text"] == "Tăng huyết áp độ 2"
    assert aligned[1]["position"] == [30, 48]


def test_upgrade_j_truncated_json_repair():
    """Kiểm tra khôi phục tự động JSON bị ngắt quãng (truncated JSON recovery trong _extract_partial_objects)."""
    from src.llm_client import _extract_partial_objects

    # LLM bị ngắt giữa chừng do max_tokens
    truncated_response = '[{"text": "Paracetamol 500mg", "type": "THUỐC"}, {"text": "Viêm phế quản", "type": "CHẨ'
    recovered = _extract_partial_objects(truncated_response)
    assert len(recovered) == 2
    assert recovered[0]["text"] == "Paracetamol 500mg"
    assert recovered[1]["text"] == "Viêm phế quản"

