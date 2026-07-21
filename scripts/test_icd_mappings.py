"""Test ICD-10 retriever lookup on common clinical diagnosis entities."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.icd_rag import ICDRetriever, ICD10VectorSearch

def test_icd():
    print("[INFO] Loading ICD retriever...")
    local_search = ICD10VectorSearch()
    icd = ICDRetriever(local_search=local_search)
    
    test_diseases = [
        "tăng huyết áp",
        "đái tháo đường type 2",
        "nhồi máu cơ tim vùng dưới cũ",
        "nhồi máu cơ tim cấp ST chênh lên",
        "suy tim độ III",
        "rung nhĩ",
        "ngoại tâm thu nhĩ",
        "ngoại tâm thu thất",
        "nhịp xoang chiếm ưu thế",
        "xơ gan do rượu",
        "hội chứng não gan",
        "viêm tuyến mồ hôi",
        "phình động mạch chủ nhỏ",
        "viêm phổi cộng đồng",
        "loét dạ dày tá tràng",
        "rối loạn lipid máu",
        "bệnh thận mạn",
    ]
    
    for d in test_diseases:
        codes = icd.lookup(d)
        print(f"  {d:<35} → {list(codes)}")

if __name__ == "__main__":
    test_icd()
