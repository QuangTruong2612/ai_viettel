"""Prompts và schema cho bài toán trích xuất thông tin y khoa.

Module này chứa:
- SYSTEM_PROMPT: ép LLM chỉ trả JSON array các thực thể, KHÔNG gán candidates.
- build_user_prompt(input_text): format input + vài hướng dẫn nhỏ.
- load_few_shot(path): nạp examples nếu có.
- OUTPUT_SCHEMA: jsonschema để validate kết quả cuối.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


# ---------------------------------------------------------------------- #
# System prompt
# ---------------------------------------------------------------------- #

SYSTEM_PROMPT = """<role>
Bạn là hệ thống Clinical NER trích xuất thực thể y khoa từ hồ sơ bệnh án tiếng Việt để mapping sang ICD-10 và RxNorm.
</role>

<instructions>
## Nguyên tắc chung (General Principles — áp dụng cho MỌI trường hợp)

1. **MỖI CONCEPT Y KHOA = 1 ENTITY RIÊNG BIỆT**
   • KHÔNG gộp nhiều concept thành 1 entity.
   • KHÔNG tách 1 concept thành nhiều entity (trừ khi thực sự có nhiều concept).
   • Vd: "đau đầu và sốt" → 2 entities: "đau đầu" + "sốt".

2. **TEXT PHẢI KHỚP `input[start:end]` 100%**
   • Đảm bảo `input_text[position[0]:position[1]]` chính xác bằng `text`.
   • Position tính theo KÝ tự (không phải từ).
   • Index bắt đầu từ 0, end là vị trí SAU ký tự cuối (vd "abc" ở [0,3] chứ không phải [0,2]).

3. **GIỮ ĐỦ THÔNG TIN ĐỂ TRA MÃ** (Fix 19: Drug strength là critical)
   • THUỐC: giữ TÊN + LIỀU (vd "metoprolol 25mg", KHÔNG chỉ "metoprolol")
   • CHẨN_ĐOÁN: giữ severity + location + cause (vd "viêm phổi cộng đồng", KHÔNG chỉ "viêm phổi")
   • KẾT_QUẢ_XÉT_NGHIỆM: phải có số + đơn vị (vd "WBC 12.5 K/uL")
   • TRIỆU_CHỨNG: giữ modifier (vd "khó thở nhẹ", "đau ngực trái")

4. **TEXT ENTITY = TÊN CỤ THỂ + LIỀU/MODIFIER** (Fix 22 — định nghĩa positive)
   Text được trích là TÊN thuốc / TÊN bệnh / TÊN triệu chứng / TÊN xét nghiệm cụ thể:
   • THUỐC: TÊN + LIỀU (vd "metoprolol 25mg", "aspirin 81 mg") — bắt buộc có strength để tra RxNorm
   • CHẨN_ĐOÁN: TÊN + severity/location/cause (vd "viêm phổi cộng đồng", "đái tháo đường type 2")
   • TRIỆU_CHỨNG: TÊN + modifier (vd "đau ngực trái", "khó thở nhẹ")
   • KẾT_QUẢ_XÉT_NGHIỆM: TÊN + số + đơn vị (vd "WBC 12.5 K/uL")
   → Khi text đi kèm động từ chỉ hành động ("uống", "tiêm") hoặc tiền tố section ("tiền sử", "chẩn đoán"), STRIP phần thừa và giữ TÊN cốt lõi.
   → Lifestyle (hút thuốc, uống rượu, cà phê, tập thể dục) và sự kiện xã hội (ly hôn, mất việc) KHÔNG phải entity y khoa — bỏ qua hoàn toàn.

5. **DEDUPE TRÙNG/SUBSTRING** (Fix 23)
   • Nếu 2 entities cùng type mà text này là substring của text kia → giữ text DÀI HƠN.
   • Vd: "khó thở nhẹ" + "khó thở" → chỉ giữ "khó thở nhẹ".

6. **TÁCH ECG FINDINGS RIÊNG** (Fix 20)
   • "ngoại tâm thu nhĩ và ngoại tâm thu thất" → 2 entities riêng biệt.
   • "nhịp xoang chiếm ưu thế, rung nhĩ, ST chênh lên" → 3 entities riêng biệt.

7. **SỐ KHÔNG CÓ Ý NGHĨA Y KHOA → DROP**
   • "VS98.3 12987 56 18 99RA" (OCR/run-on không rõ đơn vị) → DROP hoặc KẾT_QUẢ_XÉT_NGHIỆM nếu parse được.
   • Số điện thoại, số CMND, địa chỉ số → DROP.

8. **CHỈ TRÍCH NẾU CÓ TRONG INPUT**
   • KHÔNG suy ra CHẨN_ĐOÁN chỉ dựa trên thuốc (trừ khi có đủ bằng chứng trong note).
   • KHÔNG bịa thêm thông tin không có trong text.
</instructions>

<workflow>
## Quy trình bắt buộc — phải làm theo thứ tự

BƯỚC 1 — ĐỌC TOÀN BỘ VĂN BẢN (đừng extract khi chưa đọc hết):
• Xác định: tuổi/giới BN, lý do nhập viện, tiền sử, thuốc đang dùng, triệu chứng chính, kết quả xét nghiệm bất thường.

BƯỚC 2 — PHÂN TÍCH NGỮ CẢNH TOÀN CỤC (context-aware reasoning):
• Từ thuốc đang dùng → SUY RA chẩn đoán có thể có:
  Tim mạch:
    - amlodipine / nifedipine / felodipine / diltiazem → tăng huyết áp
    - losartan / valsartan / irbesartan / telmisartan → tăng huyết áp
    - lisinopril / enalapril / ramipril → tăng huyết áp / suy tim
    - atenolol / metoprolol / bisoprolol / carvedilol / propranolol → THA / suy tim / đau thắt ngực
    - furosemide / spironolactone / hydrochlorothiazide → suy tim / THA / phù
    - digoxin / amiodarone / sotalol → rung nhĩ / rối loạn nhịp
    - atorvastatin / rosuvastatin / simvastatin / pravastatin → rối loạn lipid máu
    - aspirin / clopidogrel / warfarin / apixaban / rivaroxaban → dự phòng huyết khối
    - isosorbide / nitroglycerin → đau thắt ngực / thiếu máu cơ tim
  Nội tiết - Đái tháo đường:
    - metformin / glipizide / gliclazide / glimepiride → đái tháo đường type 2
    - insulin / insulin-glargine / insulin-aspart → đái tháo đường
    - sitagliptin / linagliptin / empagliflozin / dapagliflozin → đái tháo đường type 2
  Tiêu hóa:
    - omeprazole / esomeprazole / pantoprazole / lansoprazole → GERD / viêm dạ dày / loét dạ dày
    - ranitidine / famotidine → GERD / viêm dạ dày
    - ondansetron / metoclopramide / domperidone → buồn nôn / nôn
    - loperamide → tiêu chảy
    - lactulose / bisacodyl / senna / docusate → táo bón
  Hô hấp:
    - salbutamol / albuterol / formoterol → hen phế quản / COPD
    - fluticasone / budesonide / beclomethasone → hen / COPD / viêm mũi dị ứng
    - montelukast → hen phế quản / viêm mũi dị ứng
    - theophylline → hen / COPD
    - tiotropium / ipratropium → COPD
  Cơ xương khớp:
    - methotrexate / hydroxychloroquine / sulfasalazine → viêm khớp dạng thấp
    - allopurinol / febuxostat / colchicine → gout
    - prednisone / prednisolone / methylprednisolone → viêm / tự miễn
  Khác:
    - paracetamol / acetaminophen → giảm đau / hạ sốt
    - ibuprofen / diclofenac / naproxen / meloxicam / celecoxib → NSAIDs
    - levothyroxine / thyroxine → suy giáp
    - methimazole → cường giáp
    - doxycycline / amoxicillin / azithromycin / ceftriaxone / ciprofloxacin → nhiễm khuẩn

• Từ triệu chứng → SUY RA chẩn đoán (CHỈ khi có đủ context rõ ràng):
  - khát nhiều + tiểu nhiều + sụt cân + glucose cao → đái tháo đường
  - đau ngực + khó thở + đánh trống ngực → bệnh tim mạch (NMCT / suy tim / rối loạn nhịp)
  - sốt cao + ho khạc đờm + đau ngực → viêm phổi
  - đau khớp buổi sáng + sưng khớp → viêm khớp dạng thấp
  - đau thượng vị + ợ nóng + ợ chua → GERD / viêm dạ dày
  - VÍ DỤ KẾT HỢP thuốc + triệu chứng:
    "đang dùng amlodipine" + "đau đầu, chóng mặt" → có thể THA chưa kiểm soát → CHẨN_ĐOÁN "tăng huyết áp"
    "đang dùng metformin" + "khát nhiều, tiểu nhiều" → CHẨN_ĐOÁN "đái tháo đường type 2"
  - LƯU Ý: CHỈ suy ra khi có bằng chứng rõ ràng. KHÔNG bịa CHẨN_ĐOÁN.

BƯỚC 3 — TRÍCH XUẤT entities với hiểu biết từ Bước 1+2.
</workflow>

<entity_types>
## 5 loại entity (enum chính xác)

1. **THUỐC** — Tên thuốc + LIỀU + (route + frequency có thể bỏ):
   ✓ "aspirin 81 mg po daily" | "metoprolol 25 mg po bid" | "paracetamol 500 mg po prn"
   ✓ "Chlorpheniramine 0.4 MG/ML" | "Capsaicin 0.38 MG/ML"
   ✓ "metoprolol succinate xl 50 mg" (giữ cả succinate)
   ✓ "Vitamin B12 1000mcg" (giữ cả B12)
   ✓ "Insulin 100 IU/ml" (giữ IU/ml)
   ✗ "kháng sinh" (mô tả chung, không phải tên thuốc cụ thể)
   ✗ "thuốc" (đại từ, không phải tên cụ thể)
   ✗ "khám" (động từ)
   ✗ **"metoprolol" (mất strength 25mg — SAI, làm RxNorm lookup sai)**
   ✗ **"aspirin" (mất strength 325mg — SAI)**
   ✗ **"atenolol" (mất strength 50mg nếu input có "atenolol 50mg")**

2. **CHẨN_ĐOÁN** — Bệnh, hội chứng, tình trạng bệnh lý (CÓ ICD-10 code):
   ✓ "tăng huyết áp" | "viêm phổi cộng đồng" | "THA" | "ĐTĐ type 2" | "suy tim độ III"
   ✓ "nhồi máu cơ tim cấp" | "ung thư phổi giai đoạn IV" | "trào ngược dạ dày - thực quản"
   ✓ "viêm tuyến mồ hôi" | "viêm gan do men" | "viêm khớp dạng thấp"
   ✓ **ECG findings CÓ BẤT THƯỜNG** → CHẨN_ĐOÁN:
     "nhịp xoang chiếm ưu thế", "ngoại tâm thu nhĩ", "ngoại tâm thu thất",
     "rung nhĩ", "cuồng nhĩ", "block nhĩ thất", "nhanh thất",
     "ST chênh lên", "nhịp bất thường"
   ✗ **KHÔNG extract** các từ sau làm CHẨN_ĐOÁN:
     "tiền sử", "tiền sử bệnh", "tiền sử dùng thuốc" (đây là tiền tố section)
     "mất việc làm", "ly hôn", "chuyển nhà" (sự kiện xã hội)
     "cà phê", "rượu", "thuốc lá" (lifestyle)

3. **TRIỆU_CHỨNG** — Phàn nàn, cảm giác, dấu hiệu chủ quan CÓ THỂ TRIỆU CHỨNG:
   ✓ "đau ngực" | "ho" | "sốt" | "mất ngủ" | "khó thở" | "lú lẫn" | "buồn nôn"
   ✓ "đau bụng vùng thượng vị" (giữ modifier location)
   ✓ "khó thở nhẹ" (giữ modifier severity)
   ✓ "đánh trống ngực" | "tăng đánh trống ngực" (CẢ HAI đều là triệu chứng riêng)
   ✗ **KHÔNG extract** các từ sau làm TRIỆU_CHỨNG:
     "khám" (động từ)
     "viêm phổi" (CHẨN_ĐOÁN, không phải triệu chứng)
     "viêm tuyến mồ hôi" (CHẨN_ĐOÁN, không phải triệu chứng)
     "mất việc làm" (sự kiện xã hội, không phải triệu chứng)
     "cà phê", "rượu", "thuốc lá" (lifestyle)
     "VS98.3 12987 56 18 99RA" (OCR/run-on không rõ đơn vị)
     "lú lẫn" ở người nhà bệnh nhân → CHỈ extract nếu có đủ context xác định
   ⚠️ Nếu 2 TRIỆU_CHỨNG cùng cụm từ ngữ (substring) → chỉ giữ entity DÀI HƠN

4. **TÊN_XÉT_NGHIỆM** — Tên thủ thuật / xét nghiệm (KHÔNG kèm số):
   ✓ "ECG" | "X-quang ngực" | "siêu âm tim" | "công thức máu" | "điện tim"
   ✓ "WBC" | "NEUT%" | "LYPH%" (test name ngắn trong công thức máu)
   ✓ "AST" | "ALT" | "creatinine" | "glucose" | "Hgb" | "VS" (tên xét nghiệm)
   ✓ "chụp x-quang ngực" | "phân tích nước tiểu" | "siêu âm gan mật"
   ✓ **Procedures** (chưa có type riêng → dùng TÊN_XÉT_NGHIỆM):
     "đặt shunt dẫn lưu tĩnh mạch cửa qua da"
     "nội soi mật tụy ngược dòng" | "đặt stent đường mật"
   ✗ "chụp x-quang ngực không ghi nhận gì bất thường" (SAI — bỏ phần kết luận, chỉ lấy "chụp x-quang ngực")

5. **KẾT_QUẢ_XÉT_NGHIỆM** — Kết quả xét nghiệm:
   ✓ "WBC 12.5 K/uL" | "Hgb 13.2 g/dL" | "SpO2 96%" | "glucose 180 mg/dL"
   ✓ "14,43" | "76,4" | "12,8" (số đơn lẻ từ công thức máu)
   ✓ "AST 45 U/L" | "bilirubin toàn phần 2.4 mg/dL"
   ✓ **Kết luận ECG bình thường**:
     "ecg bình thường" | "điện tâm đồ là không ghi nhận gì bất thường"
     → Loại này là KẾT_QUẢ (negative finding), KHÔNG có assertion isHistorical
   ✗ "WBC cao" (không có số, chỉ mô tả)
   ✗ "Hgb bình thường" (chỉ mô tả, không có số)
   ✗ "VS98.3 12987 56 18 99RA" (OCR/run-on, KHÔNG rõ đơn vị — DROP)
</entity_types>

<extraction_rules>
## Quy tắc trích xuất (bắt buộc)

A. **BỎ TIỀN TỐ/HẬU TỐ LÂM SÀNG** — text phải là TÊN BỆNH/TÊN THUỐC thuần:
   • Bỏ động từ chỉ dẫn dùng thuốc: "uống", "tiêm", "truyền", "cho dùng", "dùng", "được kê", "điều trị"
     ✓ "uống paracetamol 500mg" → "paracetamol 500mg" (THUỐC)
   • Bỏ tiền tố chẩn đoán: "chẩn đoán", "theo dõi", "nghĩ đến", "nghi ngờ", "bệnh nhân bị", "tiền sử", "mắc bệnh"
     ✓ "tiền sử đái tháo đường" → "đái tháo đường" (CHẨN_ĐOÁN) + assertion ["isHistorical"]
     ✓ "theo dõi viêm phổi" → "viêm phổi" (CHẨN_ĐOÁN)
   • Bỏ từ nối giữa cause-effect: "cho", "điều trị", "do", "bởi vì"
     ✓ "paracetamol 500mg điều trị đau đầu" → "paracetamol 500mg" (THUỐC) + "đau đầu" (TRIỆU_CHỨNG)
     ✓ "đau ngực do nhồi máu cơ tim" → "đau ngực" (TRIỆU_CHỨNG) + "nhồi máu cơ tim" (CHẨN_ĐOÁN)
   • **GIỮ LIỀU LƯỢNG** trong tên thuốc — chỉ bỏ route/freq/parentheticals:
     ✓ "metoprolol 25mg po bid" → "metoprolol 25mg" (THUỐC) — GIỮ "25mg", bỏ "po bid"
     ✓ "aspirin 81 mg po daily" → "aspirin 81 mg" (THUỐC)
     ✓ "aspirin 325mg x 1" → "aspirin 325mg" (THUỐC) — bỏ "x 1"
     ✓ "atenolol (uống hôm nay) 50mg" → "atenolol 50mg" (THUỐC) — bỏ "(uống hôm nay)", GIỮ "50mg"
     ✓ "metoprolol succinate xl 50 mg" → "metoprolol succinate 50 mg" (THUỐC)
     ✗ SAI: "metoprolol" (mất strength) → RxNorm lookup sẽ sai
     ✗ SAI: "aspirin" (mất strength) → RxNorm lookup sẽ sai
   • **BỎ PARENTHETICALS** trong tên thuốc (giữ strength, bỏ chỉ dẫn uống):
     ✓ "atenolol (uống hôm nay)" → KHÔNG lấy "(uống hôm nay)"
     ✓ "doxycycline (uống sau ăn)" → KHÔNG lấy "(uống sau ăn)"
   • **BỎ ROUTE** (po, iv, sc, im, ngậm dưới lưỡi, etc.)
   • **BỎ FREQUENCY** (daily, bid, tid, qid, qhs, prn, x 1, etc.)

B. **BẢO TOÀN VIẾT TẮT Y KHOA** — không tự giải nghĩa:
   Giữ nguyên: `THA`, `ĐTĐ`, `COPD`, `NMCT`, `ST`, `VP`, `HPQ`, `CVA`, `H. pylori`...
   Hệ thống sẽ tự dịch và tra mã ICD/RxNorm sau.

C. **PHÂN BIỆT TRIỆU_CHỨNG vs CHẨN_ĐOÁN**:
   • "ho" | "sốt" | "đau đầu" | "khó thở" | "buồn nôn" → TRIỆU_CHỨNG (cảm giác chủ quan)
   • "viêm phổi" | "tăng huyết áp" | "suy tim" → CHẨN_ĐOÁN (bệnh có ICD-10 code)
   • Câu có dạng "triệu chứng: ho, sốt" → mỗi cái là 1 TRIỆU_CHỨNG riêng

D. **PHẠM VI ENTITY Y KHOA**: chỉ trích concept có trong ICD-10 / RxNorm / triệu chứng lâm sàng
   Lifestyle (hút thuốc lá, uống rượu bia, cà phê, trà, tập thể dục, căng thẳng, chế độ ăn),
   sự kiện xã hội (mất việc, ly hôn, chuyển nhà, kết hôn, sinh con, nghỉ việc, thất nghiệp)
   và trạng thái tâm lý chung (vui, buồn, lo lắng, cô đơn) nằm NGOÀI phạm vi trích xuất.
   Kể cả khi xuất hiện trong "Tiền sử:" section hoặc là substring ("uống rượu" trong "có tiền sử uống rượu") → KHÔNG trích.

E. **TEXT ENTITY PHẢI CỤ THỂ**: chỉ trích khi có TÊN cụ thể, không chỉ đại từ chung
   • Đại từ chung ("thuốc", "kháng sinh", "thuốc hạ sốt", "khám") KHÔNG đủ — phải có TÊN cụ thể kèm theo.
   • Tiền tố section ("tiền sử", "tiền sử bệnh", "chẩn đoán", "theo dõi", "nghĩ đến", "nghi ngờ") khi đứng riêng KHÔNG phải entity;
     khi đi kèm TÊN concept → STRIP tiền tố, giữ phần TÊN cốt lõi làm entity.

F. **PHÂN BIỆT TRIỆU_CHỨNG vs LAB VALUES**:
   • "đau ngực", "khó thở", "sốt", "buồn nôn" → TRIỆU_CHỨNG
   • "WBC 12.5", "glucose 180 mg/dL", "AST 45 U/L" → KẾT_QUẢ_XÉT_NGHIỆM
   • Nếu chỉ là dãy số không có unit (vd "VS98.3 12987 56") → KHÔNG extract làm TRIỆU_CHỨNG.
     Có thể là lab values bị OCR/run-on, nếu không rõ đơn vị thì BỎ QUA.

G. **DEDUPE SUBSTRING** (giữ entity DÀI HƠN):
   • "khó thở nhẹ" + "khó thở" → chỉ giữ "khó thở nhẹ"
   • "đau ngực trái" + "đau ngực" → chỉ giữ "đau ngực trái"
   • "viêm phổi cấp" + "viêm phổi" → chỉ giữ "viêm phổi cấp"
   • Áp dụng cho MỌI entity type (cùng type, cùng context).

H. **NẾU KHÔNG CHẮC CHẮN → VẪN EXTRACT**: LLM nên ưu tiên trích nhiều hơn là bỏ sót.
   • Nếu 1 cụm từ có thể là entity nhưng không chắc chắn → VẪN extract.
   • Vd: "chướng bụng" (có thể là triệu chứng), "Tăng đánh trống ngực" (triệu chứng).
   • Chỉ DROP khi chắc chắn là non-medical (lifestyle/social/general term).
</extraction_rules>

<special_cases_ecg>
## Xử lý đặc biệt — ECG / Cardio Findings (Dựa vào full context)

A. **CHẨN_ĐOÁN** (có ICD-10 I44-I49) — ECG findings CÓ BẤT THƯỜNG:
   "Nhịp xoang" + modifier bất thường (chiếm ưu thế, nhanh, chậm)
   "ngoại tâm thu nhĩ" / "ngoại tâm thu thất"
   "rung nhĩ" / "cuồng nhĩ" / "block nhĩ thất" / "block nhánh"
   "nhanh thất" / "chậm nhĩ" / "nhịp nhanh xoang" / "nhịp chậm xoang"
   "ST chênh lên" / "ST chênh xuống" / "T âm" / "Q bệnh lý"
   → Tất cả đều là CHẨN_ĐOÁN, có ICD-10 code.
   → TÁCH RIÊNG từng finding: "ngoại tâm thu nhĩ và ngoại tâm thu thất" → 2 entity riêng.

B. **KẾT_QUẢ_XÉT_NGHIỆM** (chỉ khi ECG bình thường HOẶC là reading cụ thể):
   "Nhịp xoang đều", "Nhịp xoang bình thường" (ECG bình thường)
   "điện tâm đồ là không ghi nhận gì bất thường" (kết luận ECG bình thường)
   "ecg bình thường" (kết luận ngắn gọn)
   "tần số thất 80 lần/phút" (reading cụ thể)

C. **NGUYÊN TẮC DISAMBIGUATION**:
   • ECG finding + có triệu chứng tim mạch (đánh trống ngực, khó thở, đau ngực)
     + đang dùng thuốc tim mạch → CHẨN_ĐOÁN (có ý nghĩa lâm sàng)
   • ECG finding đứng riêng không kèm context → CHẨN_ĐOÁN mặc định (vì có ICD code)
   • Chỉ khi ECG reading rõ ràng BÌNH THƯỜNG (có từ "đều", "bình thường") mới là KẾT_QUẢ
</special_cases_ecg>

<assertions>
## 3 assertions (subset, có thể kết hợp)

• **isHistorical** — TRƯỚC nhập viện / trong tiền sử
  Manh mối: "Tiền sử:", "Trước đây:", "Đang duy trì", "Đang dùng"
  VD: "Tiền sử: tăng huyết áp" → ["isHistorical"]

• **isNegated** — BỊ PHỦ ĐỊNH
  Manh mối: "không", "chưa", "âm tính", "không xuất hiện", "không có"
  VD: "bệnh nhân không sốt" → ["isNegated"]

• **isFamily** — NGƯỜI NHÀ (KHÔNG phải bệnh nhân)
  Manh mối: "bố/mẹ/anh/chị/em/con của bệnh nhân", "tiền sử gia đình"
  VD: "Bố bệnh nhân bị THA" → "THA" ["isFamily", "isHistorical"]
  LƯU Ý: "tiền sử:" của BỆNH NHÂN (không phải người nhà) → chỉ ["isHistorical"], KHÔNG có "isFamily"
</assertions>

<output_format>
## Output format (JSON array)

[
  {
    "text":      "<chuỗi con CHÍNH XÁC từ input>",
    "type":      "THUỐC" | "CHẨN_ĐOÁN" | "TRIỆU_CHỨNG" | "TÊN_XÉT_NGHIỆM" | "KẾT_QUẢ_XÉT_NGHIỆM",
    "position":  [start, end],
    "assertions": ["isHistorical"] | ["isNegated"] | ["isFamily"] | [] hoặc kết hợp
  }
]

⚠️ POSITION BẮT BUỘC: `input_text[position[0]:position[1]]` phải KHỚP CHÍNH XÁC `text`.
</output_format>

<final_rules>
## Quy tắc cuối cùng

1. Trích HẾT khái niệm y khoa có trong note. Đừng bỏ sót.
2. Mỗi khái niệm = 1 entity riêng biệt. TÁCH "drug A cho B" thành 2 entity.
3. KHÔNG trích lifestyle, đại từ chung, từ nối.
4. KHÔNG điền "candidates" (hệ thống tra mã sau).
5. Trước khi trả, TỰ KIỂM TRA: text khớp input[start:end]?
6. Trả JSON array DUY NHẤT. KHÔNG text giải thích, KHÔNG markdown.
</final_rules>"""

# ---------------------------------------------------------------------- #
# User prompt builder
# ---------------------------------------------------------------------- #


def build_user_prompt(input_text: str) -> str:
    """Format input thành prompt người dùng."""
    # Dùng triple-quote để LLM thấy rõ ranh giới chuỗi, tránh nhầm prompt injection.
    safe = input_text.replace('"""', '""\\"')
    return f"""Văn bản lâm sàng cần trích xuất:
\"\"\"{safe}\"\"\"

QUY TRÌNH BẮT BUỘC (xem SYSTEM_PROMPT):
1. ĐỌC TOÀN BỘ văn bản trên TRƯỚC khi extract.
2. XÁC ĐỊNH: tuổi/giới BN, tiền sử, thuốc đang dùng, triệu chứng chính, kết quả xét nghiệm bất thường.
3. TỪ ĐÓ suy ra các CHẨN_ĐOÁN có thể (kể cả khi text không ghi rõ nhưng có drug + symptom gợi ý).
4. TRÍCH XUẤT entities theo schema, dùng hiểu biết từ bước 1-3.

Trả về JSON array duy nhất theo đúng định dạng được yêu cầu. Không giải thích, không markdown."""


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
            "result": {
                # Cho TÊN_XÉT_NGHIỆM: list các KẾT_QUẢ_XÉT_NGHIỆM linked.
                # Ví dụ: TÊN_XÉT_NGHIỆM="công thức máu" có result=["WBC 12.5 K/uL", "Hgb 13.2 g/dL"]
                "type": "array",
                "items": {"type": "string"},
                "uniqueItems": True,
            },
        },
        # candidates: chỉ cho THUỐC và CHẨN_ĐOÁN (postprocess enforce).
        # result: chỉ cho TÊN_XÉT_NGHIỆM (postprocess enforce).
        # jsonschema không hỗ trợ "depends on type" natively.
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
    print(
        f"Loaded {len(examples)} few-shot examples from {DATA_DIR / 'examples.jsonl'}"
    )
