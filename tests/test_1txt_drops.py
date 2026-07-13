import json
import re
from src.postprocess import (
    align_and_expand_entities,
    dedupe_entities,
    _drop_substring_entities,
    _filter_lifestyle_entities,
    _split_test_name_value_connector,
    _split_drug_disease_connector,
    _clean_entity_text,
)

with open("input/1.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("Length of 1.txt:", len(text))

# Simulate all possible clinical phrases in 1.txt that LLM would extract
raw_candidates = [
    {"text": "metoprolol 25mg po bid", "type": "THUỐC"},
    {"text": "doxycycline", "type": "THUỐC"},
    {"text": "viêm tuyến mồ hôi", "type": "CHẨN_ĐOÁN"},
    {"text": "atenolol", "type": "THUỐC"},
    {"text": "Căng thẳng nhiều trong công việc", "type": "TRIỆU_CHỨNG"},
    {"text": "đánh trống ngực", "type": "TRIỆU_CHỨNG"},
    {"text": "cảm giác đánh trống ngực", "type": "TRIỆU_CHỨNG"},
    {"text": "Khó thở nhẹ khó thở", "type": "TRIỆU_CHỨNG"},
    {"text": "Khó thở", "type": "TRIỆU_CHỨNG"},
    {"text": "khó thở", "type": "TRIỆU_CHỨNG"},
    {"text": "cảm giác thắt chặt ngực vùng trước tim", "type": "TRIỆU_CHỨNG"},
    {"text": "thắt chặt ngực vùng trước tim", "type": "TRIỆU_CHỨNG"},
    {"text": "Tăng đánh trống ngực", "type": "TRIỆU_CHỨNG"},
    {"text": "Cảm thấy mệt mỏi nhiều khi gắng sức trong tuần qua", "type": "TRIỆU_CHỨNG"},
    {"text": "mệt mỏi nhiều khi gắng sức", "type": "TRIỆU_CHỨNG"},
    {"text": "Cảm thấy mệt mỏi nhiều hơn sau khi luyện tập thể dục so với mọi ngày", "type": "TRIỆU_CHỨNG"},
    {"text": "mệt mỏi", "type": "TRIỆU_CHỨNG"},
    {"text": "còn cảm giác đánh trống ngực khi nhập viện", "type": "TRIỆU_CHỨNG"},
    {"text": "cảm giác thắt chặt ngực", "type": "TRIỆU_CHỨNG"},
    {"text": "giảm dung nạp gắng sức", "type": "TRIỆU_CHỨNG"},
    {"text": "buồn nôn", "type": "TRIỆU_CHỨNG"},
    {"text": "nôn", "type": "TRIỆU_CHỨNG"},
    {"text": "đổ mồ hôi", "type": "TRIỆU_CHỨNG"},
    {"text": "Nhịp xoang chiếm ưu thế", "type": "CHẨN_ĐOÁN"},
    {"text": "ngoại tâm thu nhĩ", "type": "CHẨN_ĐOÁN"},
    {"text": "ngoại tâm thu thất", "type": "CHẨN_ĐOÁN"},
    {"text": "metoprolol 25mg po bid", "type": "THUỐC"},
    {"text": "atenolol", "type": "THUỐC"},
    {"text": "siêu âm tim qua thành ngực", "type": "TÊN_XÉT_NGHIỆM"},
    {"text": "cảm giác thắt chặt ngực vùng trước tim", "type": "TRIỆU_CHỨNG"},
    {"text": "tăng đánh trống ngực", "type": "TRIỆU_CHỨNG"},
    {"text": "aspirin 325mg", "type": "THUỐC"},
    {"text": "chụp x-quang ngực", "type": "TÊN_XÉT_NGHIỆM"},
    {"text": "phân tích nước tiểu", "type": "TÊN_XÉT_NGHIỆM"},
    {"text": "ecg", "type": "TÊN_XÉT_NGHIỆM"},
    {"text": "VS98.3 12987 56 18 99RA", "type": "KẾT_QUẢ_XÉT_NGHIỆM"},
    {"text": "khó chịu vùng ngực", "type": "TRIỆU_CHỨNG"},
    {"text": "phân tích nước tiểu", "type": "TÊN_XÉT_NGHIỆM"},
    {"text": "chụp x-quang ngực", "type": "TÊN_XÉT_NGHIỆM"},
    {"text": "điện tâm đồ", "type": "TÊN_XÉT_NGHIỆM"},
    {"text": "Nhịp xoang chiếm ưu thế", "type": "CHẨN_ĐOÁN"},
    {"text": "ngoại tâm thu nhĩ", "type": "CHẨN_ĐOÁN"},
    {"text": "ngoại tâm thu thất", "type": "CHẨN_ĐOÁN"},
]

aligned = align_and_expand_entities(text, raw_candidates)
print("Aligned count:", len(aligned))
