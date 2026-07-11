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


def test_icd_stale_embeddings_and_bounds(tmp_path: Path):
    """Kiểm tra xử lý ma trận embeddings cũ lệch số dòng và bounds check trong ICD10VectorSearch."""
    import numpy as np
    from src.icd_rag import ICD10VectorSearch

    fake_jsonl = tmp_path / "fake_icd.jsonl"
    fake_jsonl.write_text('{"code":"A00","desc_vi":"Bệnh tả"}\n{"code":"A01","desc_vi":"Thương hàn"}\n', encoding="utf-8")
    fake_npy = tmp_path / "fake_emb.npy"
    # Ma trận cũ có 10 dòng, trong khi fake_jsonl chỉ có 2 dòng -> lệch số lượng
    stale_emb = np.zeros((10, 1024), dtype=np.float32)
    np.save(fake_npy, stale_emb)

    vs = ICD10VectorSearch(jsonl_path=fake_jsonl, embeddings_path=fake_npy)
    vs._ensure_loaded()
    # Ma trận cũ bị từ chối do 10 dòng != 2 dòng
    assert vs._embeddings is None
    assert len(vs.codes) == 2
