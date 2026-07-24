import sys
import re

with open('src/prompts.py', 'r', encoding='utf-8') as f:
    content = f.read()

new_stage3 = '''STAGE3_PROMPT = """Bạn là chuyên gia Clinical Coding (ICD-10 và RxNorm).
Nhiệm vụ: REVIEW lại các mã ICD-10 (cho CHẨN_ĐOÁN) và RxNorm (cho THUỐC) đã được hệ thống đề xuất.

# QUY TẮC ĐÁNH GIÁ (VERDICT)
Với mỗi cụm từ y khoa, bạn nhận được một danh sách các "candidates" (mã đề xuất).
1. Nếu candidate đầu tiên là chính xác nhất -> verdict: "ok", giữ nguyên danh sách.
2. Nếu có candidate khác trong danh sách chính xác hơn (hoặc cần loại mã sai) -> verdict: "refine", lọc lại các mã đúng.
3. Nếu TẤT CẢ candidate đều sai bản chất y khoa, HOẶC cụm từ là nhóm thuốc chung chung (ví dụ "kháng sinh", "NSAID") -> verdict: "drop", candidates: [].

# LƯU Ý QUAN TRỌNG (TRÁNH SAI LỆCH SEMANTIC):
- Bệnh của máu / di truyền (vd: Thiếu men G6PD, Tan huyết) -> Thuộc nhóm D (vd D55.0, D59). TUYỆT ĐỐI KHÔNG chọn nhóm Q (như Q55 là dị tật sinh dục) hay E (dinh dưỡng).
- Bệnh ở tim mạch (vd: viêm cơ tim) -> Thuộc nhóm I (vd I40). KHÔNG chọn nhóm B (nhiễm trùng chung).
- Thuốc: Chỉ giữ mã của thuốc cụ thể. Drop các từ chung chung.
- KHÔNG sáng tạo (invent) mã mới. Chỉ được dùng các mã có trong danh sách đề xuất.

# OUTPUT FORMAT (CHỈ TRẢ VỀ JSON ARRAY, KHÔNG GIẢI THÍCH):
[
  {
    "text": "<exact text from entity>",
    "type": "<exact type>",
    "verdict": "ok" | "refine" | "drop",
    "candidates": ["code1", "code2"],
    "reasoning": "Lý do ngắn gọn"
  }
]
"""
'''

content = re.sub(r'STAGE3_PROMPT = f"""Bạn là chuyên gia Clinical Coding.*?⚠️ CHỈ trả về JSON array, KHÔNG giải thích trước/sau\. Code phải tồn tại trong ICD-10 hoặc RxNorm — không invent\.\n"""', new_stage3, content, flags=re.DOTALL)

with open('src/prompts.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Done!')
