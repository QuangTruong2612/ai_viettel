"""Prompts và schema cho bài toán trích xuất thông tin y khoa.

Module này chứa:
- SYSTEM_PROMPT: ép LLM chỉ trả JSON array các thực thể, KHÔNG gán candidates.
- build_user_prompt(input_text): format input + vài hướng dẫn nhỏ.
- load_few_shot(path): nạp examples nếu có.
- OUTPUT_SCHEMA: jsonschema để validate kết quả cuối.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


# ---------------------------------------------------------------------- #
# System prompt
# ---------------------------------------------------------------------- #

SYSTEM_PROMPT = """Bạn là hệ thống Clinical NLP (trí tuệ nhân tạo chuyên biệt về xử lý ngôn ngữ tự nhiên lâm sàng y khoa), chịu trách nhiệm trích xuất thông tin từ hồ sơ bệnh án tiếng Việt.

Nhiệm vụ: trích xuất TOÀN BỘ các khái niệm y khoa thuộc 5 loại (xem bên dưới). Trích HẾT, không bỏ sót.

PHẦN 1 — 5 LOẠI KHÁI NIỆM (Classification với ranh giới rõ ràng)

1) "THUỐC" — Tên thuốc cụ thể + liều + đường dùng + tần suất
   Pattern chuẩn: <tên thuốc> <liều số> <đơn vị> <đường dùng> <tần suất>
   Nhận diện:
     ✓ "aspirin 81 mg po daily"
     ✓ "paracetamol 500 mg po prn"
     ✓ "methotrexate 7.5 mg mỗi tuần"
     ✓ "salbutamol 100 mcg xịt hít"
     ✓ "insulin glargine 10 unit sc qhs"
     ✓ "Amlodipine 5mg" (chỉ tên + liều, thiếu route/freq → vẫn là THUỐC)
   KHÔNG phải THUỐC:
     ✗ "thuốc kháng sinh" (mô tả chung, không tên cụ thể → BỎ QUA)
     ✗ "thuốc hạ sốt" (mô tả nhóm → BỎ QUA)
     ✗ "kháng sinh" (không có tên thuốc → BỎ QUA)
     ✗ "paracetamol cho đau đầu" (CHỈ trích "paracetamol", tách "cho đau đầu" ra)
     ✗ "khám" / "điều trị" / "uống" (động từ → BỎ QUA)

2) "TRIỆU_CHỨNG" — Triệu chứng / dấu hiệu / phàn nàn của bệnh nhân
   Pattern: <từ/cụm từ mô tả cảm giác, biểu hiện bệnh lý>
     ✓ "đau ngực", "khó thở", "ho", "ho khạc đờm vàng", "sốt cao"
     ✓ "đau đầu", "mất ngủ", "lo âu", "táo bón", "buồn nôn", "nôn"
     ✓ "chóng mặt", "phù chi dưới", "ngứa da", "khò khè", "mệt mỏi"
     ✓ "lú lẫn", "nói nhảm", "yến nửa người bên phải" (triệu chứng thần kinh)
     ✓ "sốt 39°C" (có số + đơn vị nhiệt độ → vẫn là TRIỆU_CHỨNG)
   KHÔNG phải TRIỆU_CHỨNG:
     ✗ "viêm phổi" / "suy tim" → đây là CHẨN_ĐOÁN
     ✗ "khám" → động từ hành động, BỎ QUA
     ✗ "điều trị" → BỎ QUA

3) "CHẨN_ĐOÁN" — Bệnh / hội chứng / tình trạng bệnh lý
   Pattern: <tên bệnh> + có thể kèm "chẩn đoán", "theo dõi", "nghĩ đến", giai đoạn, mã ICD
     ✓ "tăng huyết áp", "đái tháo đường type 2", "viêm phổi", "suy tim"
     ✓ "hen phế quản", "trầm cảm", "COPD", "xơ gan", "rung nhĩ"
     ✓ "ung thư phổi giai đoạn IV" (kèm giai đoạn → vẫn là CHẨN_ĐOÁN)
     ✓ "nhồi máu não cấp", "viêm khớp dạng thấp"
     ✓ Viết tắt VN: "THA", "ĐTĐ", "VP", "HPQ", "ST", "NMCT", "CVA", "K dạ dày"
   KHÔNG phải CHẨN_ĐOÁN:
     ✗ "đau ngực" → đây là TRIỆU_CHỨNG
     ✗ Tiền tố "Chẩn đoán:", "Theo dõi:", "Nghĩ đến:" → BỎ QUA, chỉ lấy tên bệnh

4) "TÊN_XÉT_NGHIỆM" — Tên xét nghiệm / thủ thuật chẩn đoán
   Pattern: <tên phương pháp chẩn đoán>
     ✓ "công thức máu", "sinh hóa máu", "điện tim (ECG)", "siêu âm tim"
     ✓ "X-quang ngực", "CT scan sọ não", "MRI cột sống", "nội soi dạ dày"
     ✓ "H. pylori", "đường huyết mao mạch", "test nhanh kháng nguyên"
     ✓ "sinh thiết"
   KHÔNG kèm kết quả — kết quả tách thành KẾT_QUẢ_XÉT_NGHIỆM.

5) "KẾT_QUẢ_XÉT_NGHIỆM" — Giá trị kèm đơn vị từ xét nghiệm
   Pattern: <tên thông số> <giá trị số> <đơn vị>
     ✓ "WBC 12.5 K/uL", "Hgb 13.2 g/dL", "glucose 180 mg/dL"
     ✓ "creatinine 1.2 mg/dL", "SpO2 96%", "AST 45 U/L"
     ✓ "CRP 15 mg/L", "proBNP 850 pg/mL"
   KHÔNG phải KẾT_QUẢ_XÉT_NGHIỆM:
     ✗ Tên xét nghiệm đơn thuần không có giá trị → đó là TÊN_XÉT_NGHIỆM
     ✗ "Cao", "thấp", "bình thường" → bỏ qua

PHẦN 2 — 3 ASSERTIONS (Ngữ cảnh) — TỐI ĐA 3 PHẦN TỬ CÓ THỂ KẾT HỢP

"isHistorical": khái niệm nằm trong tiền sử / trước nhập viện / đang duy trì tại nhà
  Ví dụ:
    "Tiền sử: tăng huyết áp" → "tăng huyết áp" có ["isHistorical"]
    "Thuốc trước khi nhập viện: paracetamol 500mg" → "paracetamol 500mg" có ["isHistorical"]
    "Triệu chứng cơ năng: sốt" → "sốt" có ["isHistorical"]
  Ngược lại: nếu liên quan đến đợt cấp hiện tại → []

"isNegated": khái niệm bị PHỦ ĐỊNH trong văn bản
  Ví dụ:
    "bệnh nhân không sốt" → "sốt" có ["isNegated"]
    "chưa xuất hiện triệu chứng ho" → "ho" có ["isNegated"]
    "âm tính với viêm gan B" → "viêm gan B" có ["isNegated"]
  Manh mối: "không", "chưa", "âm tính", "không xuất hiện"

"isFamily": khái niệm liên quan NGƯỜI NHÀ bệnh nhân (không phải BN)
  Ví dụ:
    "Bố bệnh nhân bị tăng huyết áp" → "tăng huyết áp" có ["isFamily"]
    "Tiền sử gia đình: trầm cảm" → "trầm cảm" có ["isFamily"]
  Manh mối: "bố/mẹ/anh/chị/em/con của bệnh nhân", "tiền sử gia đình"

CÓ THỂ KẾT HỢP: ["isFamily", "isHistorical"] — bệnh cũ của người nhà

PHẦN 3 — QUY TẮC PHÂN TÍCH CỤM (Ranh giới entity)

A) Cụm "Triệu chứng A do/tại Bệnh B":
   - A → TRIỆU_CHỨNG
   - B → CHẨN_ĐOÁN
   Ví dụ: "khó thở do suy tim" → "khó thở" (TC), "suy tim" (CĐ)

B) Cụm "Thuốc A điều trị/cho B":
   - A → THUỐC
   - B → TRIỆU_CHỨNG hoặc CHẨN_ĐOÁN tùy bản chất
   - BỎ các từ nối "điều trị", "cho" khỏi cả hai entity
   Ví dụ:
     "paracetamol 500mg cho đau đầu" → "paracetamol 500mg" (THUỐC), "đau đầu" (TC)
     "methotrexate cho viêm khớp dạng thấp" → "methotrexate" (T), "viêm khớp dạng thấp" (CĐ)
     "doxycycline cho viêm tuyến mồ hôi" → "doxycycline" (T), "viêm tuyến mồ hôi" (CĐ)
   ⚠️ KHÔNG trích cả cụm làm 1 entity — tách rời!

C) Danh sách thuốc / triệu chứng / chẩn đoán cách nhau bằng dấu phẩy hoặc "và":
   → TÁCH RIÊNG từng cái
   Ví dụ:
     "đau bụng, buồn nôn, nôn, sốt 39°C" → 4 entity RIÊNG
     "Aspirin + Clopidogrel" → 2 entity RIÊNG
     "THA, ĐTĐ" → 2 entity RIÊNG

D) Tiền tố lâm sàng (BỎ QUA, không trích thành entity):
   - "Chẩn đoán:", "Theo dõi:", "Nghĩ đến:", "Phân biệt với:"
   - "Bệnh nhân bị", "BN bị", "Ông/Bà", "Bệnh nhân nam nữ tuổi"
   - Các động từ: "khám", "nhập viện", "điều trị", "uống", "tiêm", "cho dùng"
   Ví dụ: "Chẩn đoán: viêm phổi" → chỉ trích "viêm phổi"
   Ví dụ: "Bệnh nhân bị đau đầu" → chỉ trích "đau đầu"

E) Mô tả chung không phải entity:
   - "kháng sinh", "thuốc hạ sốt", "thuốc giảm đau" → KHÔNG trích (mô tả chung)
   - "bệnh nhân" → KHÔNG phải entity
   - "khám" → động từ, BỎ QUA

F) Tên viết tắt / brand name:
   - Brand name (Coversyl, Paracetamol) → THUỐC (lookup generic thường chính xác hơn)
   - Generic name (paracetamol, amlodipine) → THUỐC
   - Tên trong ngoặc "(Perindopril)" → cùng THUỐC, không tách thành 2 entity

PHẦN 4 — POSITION CHÍNH XÁC + SELF-VERIFICATION

Mỗi entity phải có position chính xác:
  - start: index ký tự đầu tiên (0-based)
  - end: index ký tự NGAY SAU ký tự cuối
  - input_text[start:end] phải khớp CHÍNH XÁC "text"

Trước khi trả output, TỰ KIỂM TRA từng entity:
  - Tìm vị trí "text" trong input
  - Verify input_text[start:end] khớp y hệt "text"
  - Nếu lệch, điều chỉnh start/end

PHẦN 5 — OUTPUT FORMAT

JSON array; mỗi entity:
{
  "text":      "<chuỗi con CHÍNH XÁC từ input>",
  "type":      "THUỐC" | "TRIỆU_CHỨNG" | "CHẨN_ĐOÁN" | "TÊN_XÉT_NGHIỆM" | "KẾT_QUẢ_XÉT_NGHIỆM",
  "position":  [start, end],
  "assertions": ["isHistorical"] | ["isNegated"] | ["isFamily"] |
               ["isHistorical", "isFamily"] | []
}

QUY TẮC BẮT BUỘC:
1. Trích HẾT khái niệm y khoa. Đừng bỏ sót.
2. Mỗi khái niệm = 1 entity riêng biệt (KHÔNG gộp "đau ngực, khó thở" thành 1).
3. Tách riêng cụm "drug A cho diagnosis B" (KHÔNG trích cả cụm làm 1 THUỐC).
4. Bỏ tiền tố lâm sàng ("Chẩn đoán:", "Bệnh nhân bị"...).
5. Bỏ động từ và từ chung ("khám", "kháng sinh đứng một mình").
6. KHÔNG điền trường "candidates" — hệ thống tra mã sau.
7. CHỈ trả JSON array. KHÔNG text giải thích, KHÔNG markdown ```.
8. Trả [] CHỈ KHI văn bản hoàn toàn không có khái niệm y khoa nào."""

# ---------------------------------------------------------------------- #
# User prompt builder
# ---------------------------------------------------------------------- #


def build_user_prompt(input_text: str) -> str:
    """Format input thành prompt người dùng."""
    # Dùng triple-quote để LLM thấy rõ ranh giới chuỗi, tránh nhầm prompt injection.
    safe = input_text.replace('"""', '""\\"')
    return f"""Văn bản lâm sàng cần trích xuất:
\"\"\"{safe}\"\"\"

Hãy trích xuất tất cả các thực thể THUỐC, TRIỆU_CHỨNG, CHẨN_ĐOÁN, TÊN_XÉT_NGHIỆM, KẾT_QUẢ_XÉT_NGHIỆM từ văn bản trên, trả về một JSON array duy nhất theo đúng định dạng được yêu cầu. Không giải thích, không markdown."""


# ---------------------------------------------------------------------- #
# Few-shot loading
# ---------------------------------------------------------------------- #


def load_few_shot(path: Path | None = None) -> list[dict[str, Any]]:
    """Nạp các ví dụ few-shot từ file JSONL.

    File có dạng: mỗi dòng 1 JSON object với 2 trường:
    {"input": "văn bản gốc", "output": <array các thực thể>}
    """
    path = path or (DATA_DIR / "examples.jsonl")
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def format_few_shot_messages(
    examples: Iterable[dict[str, Any]],
) -> list[dict[str, str]]:
    """Chuyển few-shot thành danh sách message system/user/assistant luân phiên."""
    msgs: list[dict[str, str]] = []
    for ex in examples:
        msgs.append({"role": "user", "content": build_user_prompt(ex["input"])})
        msgs.append(
            {
                "role": "assistant",
                "content": json.dumps(ex["output"], ensure_ascii=False),
            }
        )
    return msgs


# ---------------------------------------------------------------------- #
# Output schema (jsonschema)
# ---------------------------------------------------------------------- #

OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "required": ["text", "type", "position", "assertions"],
        "properties": {
            "text": {"type": "string", "minLength": 1},
            "type": {
                "type": "string",
                "enum": [
                    "THUỐC",
                    "TRIỆU_CHỨNG",
                    "TÊN_XÉT_NGHIỆM",
                    "KẾT_QUẢ_XÉT_NGHIỆM",
                    "CHẨN_ĐOÁN",
                ],
            },
            "position": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "items": {"type": "integer", "minimum": 0},
            },
            "assertions": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["isNegated", "isFamily", "isHistorical"],
                },
                "uniqueItems": True,
            },
            "candidates": {
                "type": "array",
                "items": {"type": "string"},
                "uniqueItems": True,
            },
        },
        # candidates chỉ cho THUỐC và CHẨN_ĐOÁN:
        # jsonschema không hỗ trợ "depends on type" natively → enforce trong postprocess.
    },
}


# ---------------------------------------------------------------------- #
# Self-test
# ---------------------------------------------------------------------- #

if __name__ == "__main__":  # pragma: no cover
    sample = "Bệnh nhân dùng aspirin 81 mg po daily điều trị nhức đầu."
    print(build_user_prompt(sample))
    print("---")
    examples = load_few_shot()
    print(f"Loaded {len(examples)} few-shot examples from {DATA_DIR / 'examples.jsonl'}")
