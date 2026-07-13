"""Unit tests cho 4 nâng cấp đợt 2 (A, B, C, D) - ZERO Hardcode in Python."""
import pytest
from pathlib import Path
from src.icd_rag import (
    _expand_tokens_with_synonyms,
    _SYNONYM_RINGS,
    ICDRetriever,
)
from src.rxnorm_rag import RxNormRetriever


def test_upgrade_a_synonym_expansion():
    """Kiểm tra mở rộng từ đồng nghĩa từ file ngoài data/synonym_rings.json."""
    assert _SYNONYM_RINGS, "File data/synonym_rings.json phải được nạp thành công"
    tokens = {"u", "gan"}
    expanded = _expand_tokens_with_synonyms(tokens, "u ác tính gan")
    # "u" sẽ mở rộng ra "bướu", "khối", "ung", "thư"... theo JSON cấu hình
    assert "bướu" in expanded or "khối" in expanded
    assert "gan" in expanded


def test_upgrade_c_icd_caching():
    """Kiểm tra cơ chế LRU Caching trong ICDRetriever."""
    retriever = ICDRetriever()
    # Gọi lần 1
    codes_1 = retriever.lookup("tăng huyết áp")
    # Gọi lần 2 phải hit cache và trả về kết quả giống hệt
    codes_2 = retriever.lookup("tăng huyết áp")
    assert codes_1 == codes_2
    assert hasattr(retriever, '_cache')
    assert ("tăng huyết áp", None, ()) in retriever._cache


def test_upgrade_c_rxnorm_caching():
    """Kiểm tra cơ chế LRU Caching trong RxNormRetriever."""
    retriever = RxNormRetriever(use_hybrid=False)
    # Giả định cache hoạt động ổn định
    res_1 = retriever.lookup("paracetamol 500mg")
    res_2 = retriever.lookup("paracetamol 500mg")
    assert res_1 == res_2
    assert hasattr(retriever, '_cache')
    assert "paracetamol 500mg" in retriever._cache
