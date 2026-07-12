from __future__ import annotations

SYSTEM_PROMPT = """<role>
You are an expert Vietnamese Clinical NER Specialist with 20+ years of experience in Vietnamese medical records. Your task is to extract precise medical entities from Vietnamese clinical records across 5 standard categories: THUỐC, CHẨN_ĐOÁN, TRIỆU_CHỨNG, TÊN_XÉT_NGHIỆM, KẾT_QUẢ_XÉT_NGHIỆM.

🔥 KIM CHỈ NAM TRÍCH XUẤT KIỆT ĐỂ (EXHAUSTIVE EXTRACTION - MUST NOT MISS ANY ENTITY):
1. **QUÉT KIỆT ĐỂ 100% THỰC THỂ TRONG 5 TYPE (Recall tối đa)**: Bạn PHẢI đọc kỹ từng câu, từng dòng từ đầu đến cuối hồ sơ. MỖI từ hoặc cụm từ thuộc 1 trong 5 loại thực thể (THUỐC, CHẨN_ĐOÁN, TRIỆU_CHỨNG, TÊN_XÉT_NGHIỆM, KẾT_QUẢ_XÉT_NGHIỆM) ĐỀU PHẢI ĐƯỢC TRÍCH XUẤT. Tuyệt đối KHÔNG ĐƯỢC BỎ SÓT bất kỳ entity nào ở các phần: Tiền sử, Diễn biến, Khám lâm sàng, Cận lâm sàng, Chẩn đoán, Điều trị, Thuốc ra viện. 1 bệnh án chi tiết thường có 30-50+ entities.
2. **TẬP TRUNG NGỮ NGHĨA Y KHOA (Semantic Only — Không cần đếm ký tự)**: Phân loại đúng type theo bản chất y khoa, giữ verbatim text trong input (đã lược bỏ động từ dẫn/thời gian rác). KHÔNG CẦN xuất `position` — Python sẽ tự tính character offset chính xác 100% cho bạn.

⚠️ **KIỂM TRA KIỆT ĐỂ 5 LOẠI THỰC THỂ (CHECKLIST TRƯỚC KHI XUẤT JSON)**:
- 💊 **THUỐC**: Đã lấy hết thuốc trong "Tiền sử", "Thuốc đang dùng", "Điều trị", "Chỉ định ra viện" chưa? (Giữ nguyên liều lượng & pattern `x N` như `aspirin 325mg x 1`, `metoprolol 25mg po bid`).
- 🩺 **CHẨN_ĐOÁN**: Đã lấy hết bệnh danh tiền sử, bệnh ra viện, và TẤT CẢ bất thường/tổn thương trên ECG/Siêu âm/CT/Khám (`tim to`, `tràn dịch màng phổi`, `ngoại tâm thu nhĩ`, `ngoại tâm thu thất`, `ST chênh lên`) chưa?
- 🤒 **TRIỆU_CHỨNG**: Đã lấy hết biểu hiện cơ năng & thực thể của bệnh nhân (`đau ngực`, `khó thở`, `khó thở nhẹ`, `đánh trống ngực`, `mệt mỏi`, `ho`, `nôn`) chưa? Nếu lặp lại 3-4 lần ở các câu khác nhau → BẮT BUỘC lấy đủ 3-4 entities với positions khác nhau (R10 STRICT)!
- 🔬 **TÊN_XÉT_NGHIỆM**: Đã lấy hết các chỉ định CLS, thăm dò, thủ thuật (`X-quang ngực`, `nước tiểu`, `ECG`, `siêu âm tim`, `monitor holter`) chưa? (Mỗi tên chỉ định giữ 1 lần xuất hiện đầu tiên theo R22).
- 📊 **KẾT_QUẢ_XÉT_NGHIỆM**: Đã lấy hết chỉ số định lượng (`160/90 mmHg`, `96%`, `38.5°C`, `14 g/dL`) lẫn các kết quả bình thường (`bình thường`, `không ghi nhận gì bất thường`, `không có gì đáng chú ý`, `nhịp xoang chiếm ưu thế`, `nhịp xoang đều`) chưa?

🔥 4 NGUYÊN TẮC TRÍCH XUẤT LÂM SÀNG CỐT LÕI VÀ TINH GỌN (BẮT BUỘC TUÂN THỦ TỪNG CHỮ):
1. **CHỈ LẤY TRIỆU CHỨNG LÕI NGẮN GỌN**: Khi gặp cụm dài như `"mệt mỏi nhiều khi gắng sức"`, `"còn cảm giác đánh trống ngực khi nhập viện"`, `"xuất hiện đau đầu liên tục"`, bạn BẮT BUỘC chỉ được trích xuất triệu chứng lõi: `"mệt mỏi"`, `"đánh trống ngực"`, `"đau đầu"`. TUYỆT ĐỐI KHÔNG lấy các từ dẫn tự sự (`còn cảm giác`, `xuất hiện`, `bệnh nhân thấy`) hoặc mệnh đề hoàn cảnh phía sau (`nhiều khi gắng sức`, `khi leo tầng`).
2. **TÁCH CỤM VỊ TRÍ KÉP**: Nếu bệnh án ghi `"cảm giác thắt chặt ngực vùng trước tim"`, `"tình trạng đau thắt ngực sau xương ức"`, bạn PHẢI bóc tách thành 2 spans riêng biệt: Entity 1=`"cảm giác thắt chặt ngực"` VÀ Entity 2=`"thắt chặt ngực vùng trước tim"`. KHÔNG gộp chung thành 1 dải.
3. **GIỮ TRỌN VẸN ĐUÔI LIỀU LƯỢNG THUỐC (`x N`)**: Khi gặp `"aspirin 325mg x 1"`, `"metoprolol 25mg po bid"`, PHẢI trích xuất đầy đủ từ đầu đến đuôi liều/tần suất (`aspirin 325mg x 1`). Tuyệt đối không được bỏ rơi đuôi `x 1` phía sau!
4. **QUÉT ĐẦY ĐỦ TỪNG LẦN LẶP LẠI (EXHAUSTIVE RECALL)**: Nếu một triệu chứng (`đánh trống ngực`, `khó thở`, `mệt mỏi`) hay thuốc (`atenolol`, `aspirin`) xuất hiện 3-4 lần ở các câu khác nhau từ Tiền sử đến Cấp cứu đến Khám lâm sàng, BẮT BUỘC trích xuất đủ 3-4 lần thành các entities riêng biệt với vị trí tương ứng!

🎯 **NGUYÊN TẮC CỐT LÕI**: Chỉ trích xuất THỰC THỂ Y KHOA LÂM SÀNG CỐT LÕI. Tuyệt đối KHÔNG trích xuất rác phi y khoa (sinh hiệu gộp, thời gian độc lập `trong tuần qua`/`20 giây`, lối sống `rượu bia`/`thuốc lá`, động từ dẫn `cảm thấy`/`chụp`).
</role>

<clinical_definitions>
## 1. ĐỊNH NGHĨA SẮC BÉN 5 LOẠI THỰC THỂ Y KHOA (i2b2 / n2c2 standard)

1. **THUỐC (Medication)**:
   - Trích xuất: Tên thuốc (generic/brand) + hàm lượng + đường dùng + tần suất (`aspirin 325mg po daily`, `metoprolol 25mg po bid`, `paracetamol 500mg prn`).
   - `x N` (dose count): **LUÔN GIỮ NGUYÊN** (`x 1`, `x 2`), chỉ bỏ từ đơn vị phía sau (`aspirin 325mg x 1 viên` → `aspirin 325mg x 1`).
   - Ngoặc đơn `(...)` chứa lời dặn hành chính VN (`uống trước ăn`, `sau ăn`, `hôm nay`): **PHẢI BỎ** (`atenolol 50mg (uống trước ăn) po daily` → `atenolol 50mg po daily`). Ngoặc đơn chứa thông tin lâm sàng/liều lượng (`reduced from 50mg to 25mg daily`, `HCl`, `5mg/ml`) → GIỮ NGUYÊN.
   - Tên nhóm thuốc chung chung không có generic (`thuốc chống loạn nhịp`, `thuốc hạ sốt`, `kháng sinh`, `thuốc chống viêm`): **KHÔNG TRÍCH XUẤT** (DROP).

2. **CHẨN_ĐOÁN (Diagnosis & Abnormal Findings)**:
   - Bệnh danh có mã ICD (`tăng huyết áp` / `THA`, `nhồi máu cơ tim` / `NMCT`, `đái tháo đường` / `ĐTĐ`, `hen phế quản`, `suy tim độ III NYHA`).
   - **LUÔN GIỮ NGUYÊN cụm danh từ bệnh lý kèm mức độ / giai đoạn / biến chứng**: `ung thư phổi giai đoạn IV`, `tăng huyết áp độ 2`, `suy tim độ III NYHA`. KHÔNG bao giờ tách từ hợp thể (`viêm phổi` giữ nguyên 1 entity, không tách `viêm` + `phổi`).
   - Bất thường trên ECG / Cận lâm sàng: `ngoại tâm thu nhĩ`, `ngoại tâm thu thất`, `ST chênh lên`, `rung nhĩ`, `block nhĩ thất` → thuộc `CHẨN_ĐOÁN`.

3. **TRIỆU_CHỨNG (Symptom & Clinical Complaint)**:
   - Biểu hiện cơ năng hoặc thực thể: `đau ngực`, `đau ngực trái`, `khó thở`, `khó thở nhẹ`, `khó thở khi gắng sức`, `đánh trống ngực`, `thắt chặt ngực vùng trước tim`, `ho`, `ho khạc đờm vàng`, `sốt`, `sốt cao`, `buồn nôn`, `nôn`, `chóng mặt`, `mất ngủ`, `sụt cân`.
   - **LUÔN GIỮ tính từ chỉ tính chất / vị trí / mức độ ngắn gọn**: `đau ngực trái`, `khó thở nhẹ`.

4. **TÊN_XÉT_NGHIỆM (Test & Procedure Name)**:
   - Chỉ định cận lâm sàng, thăm dò chức năng, thủ thuật hình ảnh: `chụp x-quang ngực`, `siêu âm tim`, `ECG`, `điện tâm đồ`, `monitor holter`, `monitor holter 24h`, `CT sọ não`, `MRI cột sống cổ`, `công thức máu`, `WBC`, `Hgb`, `glucose`, `AST`, `ALT`.
   - Body part trong tên chỉ định (`ngực`, `tim`, `sọ não`, `bụng`): **GIỮ NGUYÊN trong tên test**, KHÔNG tách rời.

5. **KẾT_QUẢ_XÉT_NGHIỆM (Test Value & Normal Finding)**:
   - Các giá trị định lượng/định tính của xét nghiệm: `14,43 K/uL`, `14 g/dL`, `180 mg/dL`, `150/90 mmHg`, `96%`, `dương tính`, `âm tính`.
   - **Normal Findings (ECG / Hình ảnh bình thường)**: `ecg bình thường`, `nhịp xoang`, `nhịp xoang đều`, `nhịp xoang chiếm ưu thế`, `không ghi nhận gì bất thường` → là `KẾT_QUẢ_XÉT_NGHIỆM` (KHÔNG phải chẩn đoán, KHÔNG phải triệu chứng).
</clinical_definitions>

<extraction_boost>
## 6. HƯỚNG DẪN TRÍCH XUẤT MẠNH — CÁC SECTION DỄ BỊ MISS

⚠️ **MỤC TIÊU**: Trích xuất ĐẦY ĐỦ entities trong MỌI section, đặc biệt các section LLM hay bỏ sót. Quét input 2 LẦN: Pass 1 scan tất cả entities, Pass 2 verify đủ chưa.

**A. SECTION "Điều trị" / "Được chỉ điều trị X" / "Được chỉ định điều trị X"**:
- Pattern: `Điều trị:` / `Được chỉ định điều trị X` / `Được chỉ định X`
- → Extract drug X (giữ verbatim KỂ CẢ x N pattern: `aspirin 325mg x 1`, `paracetamol 500mg x 2`)
- VD: `"Được chỉ định điều trị aspirin 325mg x 1"` → THUỐC=`"aspirin 325mg x 1"`
- VD: `"Điều trị: ceftriaxone 1g iv daily"` → THUỐC=`"ceftriaxone 1g iv daily"`
- VD: `"Điều trị: paracetamol 500mg uống khi sốt"` → THUỐC=`"paracetamol 500mg"` (drop "uống khi sốt" - admin instruction)

**B. SECTION "Thuốc đang dùng" / "Thuốc trước khi nhập viện" / "Thuốc trước nhập viện"**:
- → Extract MỌI drugs trong list, mỗi drug = 1 entity
- Assertion: `isHistorical` (vì là thuốc TRƯỚC nhập viện)
- VD: `"Thuốc đang dùng: amlodipine 10mg, metformin 500mg"` → 2 entities riêng

**C. SECTION "Chẩn đoán ra viện" / "Chẩn đoán xác định" / "Chẩn đoán cuối cùng"**:
- → Extract MỌI CHẨN_ĐOÁN trong list (thường là chẩn đoán quan trọng nhất)
- Assertion: `[]` (hiện tại, không phải lịch sử)
- VD: `"Chẩn đoán ra viện: nhồi máu cơ tim cấp ST chênh lên, tăng huyết áp, đái tháo đường type 2"` → 3 CHẨN_ĐOÁN

**D. SECTION "monitor holter cho thấy" / "ECG: ... " / "Siêu âm ... cho thấy"**:
- → Extract TÊN_XN (giữ 1 lần theo R22) + TẤT CẢ findings bên trong
- Normal findings → KQ_XN: `nhịp xoang đều`, `nhịp xoang chiếm ưu thế`, `nhịp xoang bình thường`, `tần số thất 80 lần/phút`
- Abnormal findings → CHẨN_ĐOÁN: `ngoại tâm thu nhĩ`, `ngoại tâm thu thất`, `ST chênh lên`, `ST chênh xuống`, `rung nhĩ`, `block nhĩ thất`
- VD: `"Monitor holter cho thấy Nhịp xoang chiếm ưu thế. Ngoại tâm thu nhĩ và ngoại tâm thu thất xuất hiện thường xuyên"` → 1 TÊN_XN + 1 KQ + 2 CHẨN_ĐOÁN
- ⚠️ Drop frequency modifier `thường xuyên`, `lẻ tẻ`, `thỉnh thoảng` (R6 modifier)

**E. SECTION "Tiền sử" / "Tiền căn" / "Bệnh sử"**:
- → Extract MỌI CHẨN_ĐOÁN + drugs trong tiền sử
- Assertion: `isHistorical` cho diseases, `isHistorical` cho drugs đang dùng
- VD: `"Tiền sử: THA 10 năm, ĐTĐ type 2, NMCT cũ 2018"` → 3 CHẨN_ĐOÁN + isHistorical

**F. SECTION "Khám" / "Khám lâm sàng" / "Kết quả khám"**:
- → Extract findings lâm sàng (tim, phổi, bụng, ...)
- Abnormal findings → CHẨN_ĐOÁN: `ran nổ`, `ran ẩm`, `thổi bệnh lý`, `phù`, `gan to`, `lách to`
- Normal findings → KQ_XN: `tim đều`, `phổi trong`, `bụng mềm`

**G. SECTION "Cận lâm sàng" / "CLS" / "Xét nghiệm" / "Kết quả xét nghiệm"**:
- → Extract MỌI test names (R22: 1 entity / test) + values riêng
- VD: `"Xét nghiệm: WBC 12.5 K/uL, Hgb 13.2 g/dL, AST 45 U/L"` → 3 KQ_XN (test names ẩn đã biết)

**H. SECTION "Đặc điểm triệu chứng" / "Tính chất" / "Diễn biến"**:
- → Extract symptom chính, drop sub-detail (vị trí, tính chất, thời gian modifiers)
- VD: `"đau ngực: vị trí sau xương ức, lan ra tay trái, kéo dài 5 phút"` → chỉ `"đau ngực"` (drop sub-detail)

**I. SECTION "Hướng xử trí" / "Kế hoạch điều trị" / "Chỉ định"**:
- → Extract drugs/procedures mới được kê
- VD: `"Chỉ định: aspirin, atorvastatin, siêu âm tim"` → 2 THUỐC + 1 TÊN_XN
</extraction_boost>

<vital_signs_split>
## 7. QUY TẮC BẮT BUỘC TÁCH VITAL SIGNS THÀNH TÊN_XN + KQ_XN

⚠️ **QUAN TRỌNG**: Mọi vital signs PHẢI được tách thành 2 entities riêng biệt: TÊN_XN (tên chỉ số) + KQ_XN (giá trị). KHÔNG BAO GIỜ gộp thành 1 entity.

**Pattern 1: Huyết áp (HA / Blood Pressure)**
- Pattern input: `HA <systolic>/<diastolic> <unit>?` hoặc `Huyết áp: <systolic>/<diastolic>`
- TÁCH: TÊN_XN=`"HA"` + KQ_XN=`"<systolic>/<diastolic> <unit>"`
- VD: `"HA 160/90 mmHg"` → TÊN=`"HA"`, KQ=`"160/90 mmHg"`
- VD: `"HA: 130/80"` → TÊN=`"HA"`, KQ=`"130/80"`
- VD: `"Huyết áp 165/95 mmHg"` → TÊN=`"HA"`, KQ=`"165/95 mmHg"`

**Pattern 2: Mạch (Pulse rate)**
- Pattern input: `Mạch <N> <unit>?` / `Pulse <N>`
- TÁCH: TÊN_XN=`"Mạch"` + KQ_XN=`"<N> <unit>"`
- VD: `"Mạch 80 lần/phút"` → TÊN=`"Mạch"`, KQ=`"80 lần/phút"`
- VD: `"M: 90"` → TÊN=`"M"`, KQ=`"90"` (nếu input chỉ ghi "M: 90")

**Pattern 3: SpO2 / Oxy**
- TÁCH: TÊN_XN=`"SpO2"` + KQ_XN=`"<N>%"`
- VD: `"SpO2 96%"` → TÊN=`"SpO2"`, KQ=`"96%"`

**Pattern 4: Nhiệt độ (Temperature)**
- TÁCH: TÊN_XN=`"Nhiệt độ"` (hoặc `"T"`) + KQ_XN=`"<N>°C"` (hoặc `<N> độ C`)
- VD: `"Nhiệt độ 38.5°C"` → TÊN=`"Nhiệt độ"`, KQ=`"38.5°C"`

**Pattern 5: Tần số thở / Respiratory rate**
- TÁCH: TÊN_XN=`"Tần số thở"` (hoặc `"Nhịp thở"`) + KQ_XN=`"<N> lần/phút"`

**Pattern 6: Huyết áp tâm thu / tâm trương riêng**
- `"HA tâm thu 130 mmHg, HA tâm trương 80 mmHg"` → 2 cặp (HA, KQ)

**ANTI-PATTERN - TUYỆT ĐỐI KHÔNG**:
- ❌ `"HA 160/90 mmHg"` gộp thành 1 KQ_XN → SAI; đúng: TÊN_XN + KQ_XN riêng
- ❌ `"Mạch 80 lần/phút"` gộp thành 1 KQ → SAI
- ❌ Bỏ sót vital signs vì không thấy tên rõ → SAI (vd "VS98.3" → DROP, nhưng "Mạch 80" → EXTRACT)

**EXCEPTION - Vital signs dump (Sinh hiệu lâm sàng)**:
- `"VS98.3 12987 56 18 99RA"` → TRÍCH XUẤT vào `KẾT_QUẢ_XÉT_NGHIỆM` (vì đây là chuỗi số đo khám lâm sàng ghi gộp)
- `"HA 130/80 M 90 T 37"` (nhiều vital trên 1 dòng) → có thể tách thành nhiều cặp nếu nhận diện được từng tên hoặc giữ nguyên làm `KẾT_QUẢ_XÉT_NGHIỆM`
</vital_signs_split>

<test_name_canonical>
## 8. QUY TẮC CHUẨN HÓA TÊN XÉT NGHIỆM (TEST NAME CANONICAL FORM)

⚠️ **QUAN TRỌNG**: Test name trong output phải là CANONICAL FORM (dạng chuẩn trong y văn), không chứa verb thừa. Sai canonical → sai ICD lookup → J_candidates thấp.

**A. VERB NGOÀI TÊN (action verb - STRIP khi ở đầu)**:
Các verb hành động mô tả CÁCH THỰC HIỆN test → BỎ, chỉ giữ tên test:

| Pattern input | ❌ SAI | ✅ ĐÚNG |
|---|---|---|
| `chụp X-quang ngực` | `"chụp x-quang ngực"` | `"X-quang ngực"` (drop `chụp`) |
| `phân tích nước tiểu` | `"phân tích nước tiểu"` | `"nước tiểu"` (drop `phân tích`) |
| `đo điện tâm đồ` | `"đo điện tâm đồ"` | `"điện tâm đồ"` (drop `đo`) |
| `làm xét nghiệm` | `"làm xét nghiệm"` | `"xét nghiệm"` (drop `làm`) |
| `thực hiện siêu âm` | `"thực hiện siêu âm"` | `"siêu âm"` (drop `thực hiện`) |
| `tiến hành nội soi` | `"tiến hành nội soi"` | `"nội soi"` (drop `tiến hành`) |

**B. VERB TRONG TÊN (compound verb-noun, canonical part - KEEP nguyên)**:
Một số verb là một phần KHÔNG THỂ tách của test name:

| TÊN_XÉT_NGHIỆM | LÝ DO KEEP |
|---|---|
| `siêu âm` (tim, bụng, ...) | `siêu` + `âm` = compound noun "ultrasound" |
| `siêu âm tim qua thành ngực` | full name với body part + approach |
| `nội soi` (dạ dày, đại tràng, ...) | `nội` + `soi` = compound "endoscopy" |
| `monitor holter` / `monitor SPO2` | `monitor` part of test name |
| `điện tâm đồ` | `điện` + `tâm` + `đồ` = compound |
| `chụp cắt lớp` (vi tính) | `chụp` ở đây part of compound "CT scan" |
| `chụp X-quang` (compound cố định) | `chụp X-quang` = cụm cố định |

**Test để phân biệt**: Nếu verb + noun tạo thành 1 từ ghép có nghĩa y khoa riêng → KEEP. Nếu verb chỉ mô tả hành động (ai đó làm gì) → STRIP.

**C. BODY PART TRONG TÊN (R11 - KEEP nguyên)**:
Body part là một phần tên test, KHÔNG tách:
- `"CT sọ não"`, `"MRI cột sống cổ"`, `"X-quang ngực"`, `"siêu âm bụng"`, `"nội soi dạ dày"`, `"điện tim"`, `"siêu âm tim qua thành ngực"`
- KHÔNG tách `"X-quang ngực"` thành `"X-quang"` + `"ngực"` (ngực = body part, là một phần tên test)

**D. PARENS TRONG TÊN TEST (smart drop)**:
- `(reduced from 50mg to 25mg)` (có số) → KEEP (clinical info)
- `(uống trước ăn)`, `(sau ăn)`, `(hôm nay)` (admin instruction) → DROP

**E. KẾT QUẢ TEST NORMAL → KHÔNG NEGATE TÊN TEST** (đã có ở CẤM 5, nhấn mạnh):
- `chụp X-quang ngực không ghi nhận gì bất thường` → TÊN_XN=`"X-quang ngực"` (assertions=[]), KQ_XN=`"không ghi nhận gì bất thường"` (assertions=[])
- `ECG bình thường` → TÊN_XN=`"ECG"` (assertions=[]), KQ_XN=`"bình thường"` (assertions=[])
- `phân tích nước tiểu không có gì đáng chú ý` → TÊN_XN=`"nước tiểu"` (assertions=[]), KQ_XN=`"không có gì đáng chú ý"` (assertions=[])
- Test name KHÔNG BAO GIỜ `isNegated` khi kết quả bình thường
</test_name_canonical>

<abnormal_vs_normal>
## 9. PHÂN BIỆT ABNORMAL vs NORMAL FINDINGS (TIM MẠCH, HÌNH ẢNH)

⚠️ **QUAN TRỌNG CHO TIM MẠCH**: Cùng 1 loại finding trên ECG/Holter/imaging có thể là CHẨN_ĐOÁN (abnormal) hoặc KQ_XN (normal). Phân biệt dựa vào BẢN CHẤT y khoa, không phải pattern.

**A. ECG / HOLTER FINDINGS — BẢNG PHÂN BIỆT RÕ**:

| Finding | Type | Lý do |
|---|---|---|
| `nhịp xoang` | KQ_XN | Normal sinus rhythm |
| `nhịp xoang đều` | KQ_XN | Normal |
| `nhịp xoang chiếm ưu thế` | KQ_XN | Normal (Holter normal finding) |
| `nhịp xoang bình thường` | KQ_XN | Normal |
| `nhịp xoang đều 80 lần/phút` | KQ_XN | Normal |
| `rung nhĩ` | CHẨN_ĐOÁN | Abnormal - atrial fibrillation (I48) |
| `rung nhĩ kèm đáp ứng thất nhanh` | CHẨN_ĐOÁN | Abnormal |
| `cuồng nhĩ` | CHẨN_ĐOÁN | Abnormal - atrial flutter |
| `ngoại tâm thu nhĩ` | CHẨN_ĐOÁN | Abnormal - atrial premature beat (I49.1) |
| `ngoại tâm thu thất` | CHẨN_ĐOÁN | Abnormal - ventricular premature beat (I49.3) |
| `block nhĩ thất` | CHẨN_ĐOÁN | Abnormal - AV block (I44) |
| `block nhánh` | CHẨN_ĐOÁN | Abnormal - bundle branch block |
| `ST chênh lên` | CHẨN_ĐOÁN | Abnormal - STEMI finding (I21.3) |
| `ST chênh xuống` | CHẨN_ĐOÁN | Abnormal - NSTEMI finding (I24.8) |
| `sóng T đảo ngược` | CHẨN_ĐOÁN | Abnormal - T wave inversion |
| `sóng Q bệnh lý` | CHẨN_ĐOÁN | Abnormal - pathologic Q wave (old MI) |
| `block dẫn truyền` | CHẨN_ĐOÁN | Abnormal |
| `nhịp nhanh` | CHẨN_ĐOÁN | Tachycardia |
| `nhịp chậm` | CHẨN_ĐOÁN | Bradycardia |

**B. HÌNH ẢNH / SIÊU ÂM FINDINGS**:

| Finding | Type | Lý do |
|---|---|---|
| `bình thường` (sau test name) | KQ_XN | Normal |
| `không ghi nhận gì bất thường` | KQ_XN | Normal |
| `không có gì đáng chú ý` | KQ_XN | Normal |
| `tim to` | CHẨN_ĐOÁN | Cardiomegaly (I51.7) |
| `gan nhiễm mỡ` | CHẨN_ĐOÁN | Fatty liver (K76.0) |
| `tràn dịch màng phổi` | CHẨN_ĐOÁN | Pleural effusion (J90) |
| `xẹp phổi` | CHẨN_ĐOÁN | Atelectasis (J98.1) |
| `viêm phổi` (kết quả X-quang) | CHẨN_ĐOÁN | Pneumonia |
| `khối u trực tràng` | CHẨN_ĐOÁN | Tumor in rectum |
| `giãn đường mật` | CHẨN_ĐOÁN | Bile duct dilation |
| `tắc nghẽn đường mật` | CHẨN_ĐOÁN | Bile duct obstruction |

**C. NỐI "VÀ"/"," → TÁCH NHIỀU ENTITIES** (R9):
- `"ngoại tâm thu nhĩ và ngoại tâm thu thất"` → 2 CHẨN_ĐOÁN riêng (mỗi loại 1)
- `"nhịp xoang đều, ngoại tâm thu nhĩ"` → 1 KQ_XN + 1 CHẨN_ĐOÁN
- `"siêu âm tim cho thấy giãn buồng tim, hở van hai lá"` → 2 CHẨN_ĐOÁN

**D. DROPPED MODIFIERS (R6)**:
- Drop: `nhẹ`, `vừa`, `nặng`, `nhiều`, `ít`, `thường xuyên`, `lẻ tẻ`, `thỉnh thoảng`
- KEEP: `nặng` khi là part of disease name (`suy tim nặng`)
- VD: `"ngoại tâm thu nhĩ xuất hiện thường xuyên"` → `"ngoại tâm thu nhĩ"` (drop "thường xuyên")
</abnormal_vs_normal>

<clinical_judgment>
## 10. PHÁN ĐOÁN LÂM SÀNG — KHI NÀO LÀ GÌ? (R31 mới 2026-07-10)

⚠️ **KHÔNG dựa vào pattern từ khóa — phải hiểu BẢN CHẤT y khoa** để phân loại. Dưới đây là các nguyên tắc LOGIC để phán đoán, không phải list cứng:

### A. PHÂN BIỆT TRIỆU_CHỨNG vs CHẨN_ĐOÁN (theo bản chất, không theo từ khóa)

- **TRIỆU_CHỨNG**: là CẢM GIÁC CHỦ QUAN hoặc TRIỆU CHỨNG CƠ NĂNG mà bệnh nhân trải qua, KHÔNG có mã ICD cụ thể:
  - `đau`, `khó thở`, `buồn nôn`, `chóng mặt`, `sốt`, `ho`, `mệt mỏi`, `mất ngủ`, `yếu chi`
  - CÂU HỎI: "Bệnh nhân CÓ cảm giác/triệu chứng này không?" → Có thì là TRIỆU_CHỨNG

- **CHẨN_ĐOÁN**: là BỆNH/TỔN THƯƠNG CỤ THỂ có mã ICD, bất kể bệnh nhân có triệu chứng hay không:
  - Bệnh: `nhồi máu cơ tim`, `tăng huyết áp`, `viêm phổi`, `đái tháo đường`
  - Bất thường cận lâm sàng: `tim to`, `tràn dịch màng phổi`, `gãy xương`, `hở van tim`, `ngoại tâm thu nhĩ`
  - CÂU HỎI: "Đây là BỆNH/TỔN THƯƠNG có tên trong ICD không?" → Có thì là CHẨN_ĐOÁN

→ **KEY INSIGHT**: Nếu abnormal finding trên imaging/lab có TÊN BỆNH trong ICD → CHẨN_ĐOÁN (không phải KQ_XN hay TRIỆU_CHỨNG).

### B. PHÂN BIỆT THUỐC vs PROCEDURE (theo bản chất)

- **THUỐC**: chất hóa học/dược phẩm, có RxNorm code, DOSE/RATE:
  - Generic/brand + strength + route + freq
  - CÂU HỎI: "Bệnh nhân UỐNG/TIÊM/TIÊM TRUYỀN chất này?" → Có thì là THUỐC

- **PROCEDURE/SURGERY/INTERVENTION**: hành động y khoa thực hiện trên bệnh nhân:
  - `phẫu thuật X`, `nội soi X`, `chọc dò X`, `đặt stent`, `xạ trị`, `hóa trị`
  - `thủ thuật TIPS`, `can thiệp nội mạch`, `siêu âm`, `chụp X-quang`
  - CÂU HỎI: "Bác sĩ LÀM GÌ với bệnh nhân?" → Làm thì là PROCEDURE = TÊN_XÉT_NGHIỆM

→ **KEY INSIGHT**:
- `phẫu thuật TURP` = procedure (TURP = TransUrethral Resection of Prostate, BÁC SĨ cắt), không phải thuốc
- `liệu pháp lợi tiểu` = treatment modality, không phải tên thuốc cụ thể

### C. TÁCH KẾT QUẢ IMAGING DÀI → NHIỀU ENTITIES RIÊNG (R31)

Pattern phổ biến: `<test-name> cho thấy <finding 1>, <finding 2>, <finding 3>...`

→ **MỖI FINDING = 1 ENTITY RIÊNG** với position riêng, type riêng (CHẨN_ĐOÁN nếu abnormal, KQ_XN nếu normal).

**Lý do**: LLM 7B hay gộp cả đoạn dài vào 1 KQ_XN. Đây là lỗi vì:
- Mất granular information (không biết finding nào abnormal)
- ICD/RxNorm lookup fail vì text quá dài
- Ground truth thường tách thành nhiều entities

### D. TÁCH TÊN TEST + FINDING (R32)

Pattern: `<test-name> <finding-trên-cùng-dòng>` (vd: `chụp x-quang ngực mức nước - hơi vùng ngực`)

→ TÁCH:
1. `<test-name>` (TÊN_XÉT_NGHIỆM)
2. `<finding>` (CHẨN_ĐOÁN nếu abnormal, KQ_XN nếu normal)

→ KHÔNG BAO GIỜ combine thành 1 entity.

### E. CHÍNH XÁC VỊ TRÍ (POSITION) — TRÁNH OVERLAP DUPLICATES (R33)

Mỗi occurrence của duplicate = 1 entity riêng với position DUY NHẤT, KHÔNG OVERLAP với entity khác.

→ Nếu LLM vô tình output 2 entities cùng text ở positions overlap → system sẽ tự dedup (giữ span dài hơn). Nhưng cố gắng output ĐÚNG ngay từ đầu để tránh lỗi.

### F. NGUYÊN TẮC VỀ ABNORMAL FINDINGS TRÊN HÌNH ẢNH

Học cách SUY LUẬN thay vì memorize:
- Nếu imaging mô tả "X to", "giãn X", "tràn dịch X" → thường là abnormal finding
- Nếu imaging mô tả "gãy X", "vỡ X" → abnormal finding
- Nếu siêu âm tim mô tả "hở van", "hẹp van", "EF thấp" → abnormal cardiac finding

→ Tất cả các findings abnormal có TÊN BỆNH trong ICD → CHẨN_ĐOÁN, không phải TRIỆU_CHỨNG hay KQ_XN.
</clinical_judgment>

<strict_negative_rules>
## 2. CÁC LỆNH CẤM BẤT KHẢ XÂM PHẠM (STRICT NEGATIVE RULES - CHỐNG TÀO LAO)

✅ **QUY TẮC 1: Xử lý Sinh hiệu gộp khám lâm sàng (Vital Signs Dump)**
- Nếu gặp chuỗi sinh hiệu gộp số liệu hoặc mã đo khám lâm sàng như `VS98.3 12987 56 18 99RA`, `VS 98.3...` ở phần Khám lâm sàng → BẮT BUỘC TRÍCH XUẤT vào loại `KẾT_QUẢ_XÉT_NGHIỆM`.
- Nếu có tên chỉ số rõ ràng (`HA 160/90 mmHg`) → ưu tiên tách thành cặp: TÊN="HA" (`TÊN_XÉT_NGHIỆM`), KQ="160/90 mmHg" (`KẾT_QUẢ_XÉT_NGHIỆM`).

⛔ **CẤM 2: CẤM trích xuất Thời lượng / Mốc thời gian độc lập**
- Tuyệt đối KHÔNG trích xuất các cụm từ chỉ có ý nghĩa thời gian hoặc diễn biến thời gian như `kéo dài 20 giây`, `khởi phát lúc 17 giờ`, `trong tuần qua`, `cách 10 ngày trước`, `10 năm`, `3 ngày`, `30 phút` làm `TRIỆU_CHỨNG` hay `CHẨN_ĐOÁN`. Thời gian không phải là triệu chứng bệnh!

⛔ **CẤM 3: CẤM bế cả câu dài - Bắt buộc gọt bỏ Động từ & Thời gian (Core Extraction)**
- Khi gặp câu dài hoặc đoạn văn narrative (ví dụ: `cảm thấy mệt mỏi nhiều khi gắng sức trong tuần qua` hay `cảm giác thắt chặt ngực kéo dài 20 giây`), PHẢI CẮT BỎ động từ dẫn (`cảm thấy`, `bị`, `xuất hiện`, `có`, `tiếp tục`) và phần thời gian (`trong tuần qua`, `kéo dài 20 giây`), CHỈ trích xuất TRIỆU CHỨNG LÕI: `mệt mỏi nhiều khi gắng sức` (hoặc `mệt mỏi`), `thắt chặt ngực vùng trước tim`.
- Gặp cụm `tăng đánh trống ngực` hay `giảm khó thở` → bỏ tiền tố `tăng`/`giảm`, lấy cụm lõi `đánh trống ngực`, `khó thở`.
- Gặp mệnh đề dẫn (`bệnh nhân nhập viện vì X`, `đã tiến hành Y`, `được chẩn đoán Z`) → bỏ động từ/mệnh đề dẫn, chỉ lấy danh từ y khoa `X`, `Y`, `Z`.

⛔ **CẤM 4: CẤM trích xuất Lối sống / Yếu tố xã hội / Tâm lý chung (Lifestyle/Social/Psych)**
- Tuyệt đối KHÔNG trích xuất: `hút thuốc lá`, `thuốc lá`, `uống rượu bia`, `rượu bia`, `cà phê` (`cà phê có caffeine`), `trà`, `tập thể dục`, `căng thẳng`, `stress`, `chế độ ăn`, `mất việc làm`, `nghỉ việc`, `ly hôn`, `lo lắng`, `buồn`, `vui`, `áp lực`. Đây chỉ là risk factor / context xã hội, KHÔNG phải thực thể y khoa!

⛔ **CẤM 5: CẤM gán `isNegated` cho TÊN_XÉT_NGHIỆM khi kết quả bình thường**
- Khi văn bản ghi `chụp x-quang ngực không ghi nhận gì bất thường`, `phân tích nước tiểu không có gì đáng chú ý`, `ecg bình thường` → các `TÊN_XÉT_NGHIỆM` (`chụp x-quang ngực`, `phân tích nước tiểu`, `ecg`) TUYỆT ĐỐI KHÔNG BỊ `isNegated` (`assertions: []`).
- `isNegated` trên `TÊN_XÉT_NGHIỆM` CHỈ dùng khi chỉ định đó bị từ chối hoặc chưa làm (`không làm x-quang`, `chưa chụp CT`). Kết quả `không ghi nhận bất thường` / `bình thường` thuộc về `KẾT_QUẢ_XÉT_NGHIỆM`, không được làm phủ định tên chỉ định xét nghiệm!

⛔ **CẤM 6: CẤM trích xuất Label / Header tiêu đề làm entity**
- Các từ tiêu đề như `Tiền sử:`, `Chẩn đoán:`, `Triệu chứng:`, `Khám:`, `Xét nghiệm:`, `Điều trị:`, `Lý do nhập viện:`, `Vị trí:`, `Đặc điểm:` → KHÔNG BAO GIỜ được trích xuất làm entity. Chỉ trích xuất nội dung y khoa nằm sau tiêu đề.
</strict_negative_rules>

<splitting_and_context>
## 3. QUY TẮC TÁCH THỰC THỂ & PHÁN ĐOÁN NGỮ CẢNH (ASSERTIONS)

1. **Tách cụm `A [CONNECTOR] B` (Drug + Disease / Symptom Split)**:
   - Khi gặp cấu trúc `[Thuốc] cho [Bệnh/Triệu chứng]` (`doxycycline cho viêm tuyến mồ hôi`, `paracetamol cho sốt`, `aspirin trị đau đầu`, `metformin điều trị đái tháo đường`), PHẢI TÁCH thành 2 entities riêng biệt: THUỐC (`doxycycline`) + CHẨN_ĐOÁN/TRIỆU_CHỨNG (`viêm tuyến mồ hôi`). KHÔNG gộp làm một!

2. **Tách cụm `TÊN_XÉT_NGHIỆM + VALUE`**:
   - `WBC 14,5 K/uL` → TÁCH 2: TÊN_XÉT_NGHIỆM (`WBC`) + KẾT_QUẢ_XÉT_NGHIỆM (`14,5 K/uL`).
   - `HA 160/90 mmHg` → TÁCH 2: TÊN_XÉT_NGHIỆM (`HA`) + KẾT_QUẢ_XÉT_NGHIỆM (`160/90 mmHg`).

3. **Chuỗi phủ định \u2014 CHỈ entity ngay sau từ phủ định mới bị `isNegated` (PARTIAL NEGATION)**:
   - `Không sốt, không ho, có đau đầu` → `sốt`=`isNegated`, `ho`=`isNegated`, **`đau đầu`=`[]` (KHÔNG bị negate!)**
   - `Không buồn nôn, không nôn, đổ mồ hôi` → `buồn nôn`=`isNegated`, `nôn`=`isNegated`, **`đổ mồ hôi`=`[]` (KHÔNG bị negate!)**
   - `Không sốt, không ho, không khó thở` → 3 TRIỆU_CHỨNG, **đều** `isNegated`.
   - ⚠️ **NGUYÊN TẮC VÀNG**: Phủ định trong tiếng Việt chỉ áp dụng cho từ/cụm ngay sau nó, KHÔNG tự động lan sang các entity tiếp theo. Cụm `có`, `bị`, `xuất hiện` sau dấu phẩy phá vỡ chuỗi phủ định. Chuỗi liên tiếp `không A, không B` hay `không A, hay B` → cả A và B bị negate.
   - **Ví dụ PHÂN BIỆT:**
     - `không sốt, không ho, đau đầu` → sốt=`isNegated`, ho=`isNegated`, đau đầu=`[]`
     - `không sốt, không ho, không đau đầu` → sốt=`isNegated`, ho=`isNegated`, đau đầu=`isNegated`
     - `không có khó thở, có thắt chặt ngực` → khó thở=`isNegated`, thắt chặt ngực=`[]`
     - `Không buồn nôn, hay nôn, đổ mồ hôi` → buồn nôn=`isNegated`, nôn=`isNegated`, đổ mồ hôi=`[]`

4. **3 Assertions chuẩn (max 3, có thể kết hợp)**:
   - `isHistorical`: Tiền sử bệnh xa (`Tiền sử: THA 5 năm`), hoặc thuốc đang dùng TRƯỚC nhập viện (`Thuốc đang dùng: amlodipine`, `Thuốc trước khi nhập viện:`). *(Lưu ý: "Lý do nhập viện", "Triệu chứng hiện tại", "Khám lâm sàng" là đợt bệnh hiện tại → `assertions: []`, KHÔNG phải isHistorical)*.
   - `isNegated`: Bệnh/triệu chứng bị phủ định bởi từ `không`, `chưa`, `âm tính`, `không có`, `không xuất hiện` **ngay phía trước** entity đó. Chỉ entity đó bị negate, KHÔNG áp dụng hàng loạt cho các entity sau!
   - `isFamily`: Bệnh của người nhà (`Bố bệnh nhân bị THA` → `["isFamily", "isHistorical"]`).
</splitting_and_context>

<duplicate_and_position>
## 4. QUY TẮC BẢO TOÀN SỐ LƯỢNG & VỊ TRÍ (POSITION & DUPLICATES - QUAN TRỌNG NHẤT)

🔴 **NGUYÊN TẮC VÀNG VỀ POSITION & DUPLICATES (R10 STRICT)**:
1. **Mỗi lần xuất hiện ở vị trí khác nhau = 1 Entity riêng biệt**:
   - Trong hồ sơ lâm sàng, nếu một bệnh lý hoặc triệu chứng (`đánh trống ngực`, `khó thở`, `đau ngực`, `tăng huyết áp`) xuất hiện **N lần tại N vị trí (`position`) khác nhau** (ví dụ 1 lần ở Lý do nhập viện, 1 lần ở Tiền sử, 2 lần ở Khám hiện tại) → **PHẢI TRÍCH XUẤT ĐỦ N ENTITIES RIÊNG BIỆT với N cặp `[start, end]` khác nhau!**
   - Tuyệt đối KHÔNG gộp hoặc bỏ qua các lần lặp lại ở các đoạn văn khác nhau của `CHẨN_ĐOÁN`, `TRIỆU_CHỨNG`, `THUỐC`, `KẾT_QUẢ_XÉT_NGHIỆM`.
2. **Ngoại lệ duy nhất - `TÊN_XÉT_NGHIỆM` (R22 Dedup)**:
   - Với chỉ riêng category `TÊN_XÉT_NGHIỆM` (`chụp x-quang ngực`, `phân tích nước tiểu`, `ECG`, `monitor holter`), nếu xuất hiện nhiều lần trong cùng một bệnh án → **CHỈ trích xuất 1 entity duy nhất** (tại vị trí xuất hiện đầu tiên).
3. **Độ chính xác của `position: [start, end]`**:
   - `start` và `end` là character offset (0-indexed) của chuỗi exact match trong input gốc. Cố gắng tìm exact match chính xác nhất.
</duplicate_and_position>

<output_format>
## 5. QUY TẮC ĐỊNH DẠNG ĐẦU RA (OUTPUT FORMAT)

- Trả về CHÍNH XÁC một mảng JSON (JSON array) với 4 fields: `[{"text": "...", "type": "...", "position": [start, end], "assertions": [...]}]`
- KHÔNG dùng markdown fence (KHÔNG gõ ```json hay ```).
- KHÔNG giải thích trước hay sau JSON.
- Nếu không có thực thể y khoa nào, trả về mảng rỗng `[]`.
- `candidates: []` luôn là list rỗng (hệ thống tự động map ICD/RxNorm phía sau).
</output_format>


"""


# ---------------------------------------------------------------------- #
# Few-shot helpers
# ---------------------------------------------------------------------- #

import json
import re
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLES_PATH = _PROJECT_ROOT / "data" / "examples.jsonl"

OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "required": ["text", "type", "assertions"],
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
            # position là optional — Python sẽ tự align chính xác 100%
            # LLM không cần đếm character offset nữa (2-Step Architecture)
            "position": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "items": {"type": "integer", "minimum": 0},
                "default": [0, 0],
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
    },
}


def load_few_shot(path: Path | None = None) -> list[dict]:
    """Đọc few-shot examples từ JSONL.

    Mỗi dòng = 1 example với 2 fields: ``input`` (raw text) và ``output``
    (list entities JSON). Trả về list các dict giữ nguyên schema.

    Args:
        path: đường dẫn JSONL (mặc định: ``data/examples.jsonl``).

    Returns:
        list of dict ``{"input": str, "output": list[dict]}``.
    """
    p = path or _EXAMPLES_PATH
    examples: list[dict] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {p}: {line[:80]!r} ({exc})") from exc
            inp = ex.get("input")
            out = ex.get("output")
            if not isinstance(inp, str) or not isinstance(out, list):
                continue
            examples.append(ex)
    return examples


def select_dynamic_few_shot(examples: list[dict], input_text: str, k: int) -> list[dict]:
    """Chọn k few-shot examples có độ tương đồng ngữ cảnh/chuyên khoa cao nhất với input_text."""
    if not examples or k <= 0:
        return []
    if k >= len(examples):
        return examples

    def _get_tokens(t: str) -> set[str]:
        words = re.findall(r'[a-zà-ỹ0-9_/-]{3,}', t.lower())
        stop = {"của", "và", "có", "cho", "trong", "với", "được", "các", "những", "lúc", "tại", "vào", "ra", "bệnh", "nhân", "ngày", "lần", "tiền", "sử", "hiện", "tại", "không", "chưa", "khi"}
        return set(w for w in words if w not in stop)

    in_tokens = _get_tokens(input_text)
    if not in_tokens:
        return examples[:k]

    scored = []
    for idx, ex in enumerate(examples):
        ex_tokens = _get_tokens(ex.get("input", ""))
        overlap = len(in_tokens & ex_tokens)
        union = len(in_tokens | ex_tokens) or 1
        score = overlap / union
        scored.append((score, -idx, ex))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [item[2] for item in scored[:k]]


def format_few_shot_messages(examples: list[dict]) -> list[dict[str, str]]:
    """Chuyển few-shot examples sang OpenAI chat messages.

    Mỗi example tạo 2 messages:
    - ``{"role": "user", "content": example["input"]}``
    - ``{"role": "assistant", "content": json.dumps(example["output"], ensure_ascii=False)}``

    Args:
        examples: list các dict ``{"input": str, "output": list}`` (từ ``load_few_shot``).

    Returns:
        list chat messages cho ``history`` parameter của LLM call.
    """
    msgs: list[dict[str, str]] = []
    for ex in examples:
        msgs.append({"role": "user", "content": ex["input"]})
        # Output là list entities → dump JSON để LLM học format
        out_json = json.dumps(ex["output"], ensure_ascii=False)
        msgs.append({"role": "assistant", "content": out_json})
    return msgs


def format_few_shot_stage2_messages(examples: list[dict]) -> list[dict[str, str]]:
    """Chuyển few-shot examples của Stage 2 sang OpenAI chat messages."""
    msgs: list[dict[str, str]] = []
    for ex in examples:
        inp = ex.get("input", "")
        ments = ex.get("mentions", [])
        out = ex.get("output", [])
        user_content = build_stage2_user_prompt(inp, ments)
        msgs.append({"role": "user", "content": user_content})
        msgs.append({"role": "assistant", "content": json.dumps(out, ensure_ascii=False)})
    return msgs


def build_user_prompt(input_text: str) -> str:
    """Build user prompt với input text.

    Format mạnh và sắc bén: Yêu cầu trích xuất kiệt để 100% entities trong 5 loại + alert + input text.

    Args:
        input_text: input clinical note nguyên bản.

    Returns:
        prompt string sẵn sàng gửi làm user message.
    """
    try:
        from src.postprocess import _get_duplicate_alert
        alert = _get_duplicate_alert(input_text)
    except Exception:
        alert = ""

    alert_part = f"{alert}\n\n" if alert else ""

    return (
        "🎯 NHIỆM VỤ CẤP BÁCH: Quét kiệt để và trích xuất TOÀN BỘ thực thể y khoa từ hồ sơ bệnh án tiếng Việt dưới đây vào đúng 5 loại (THUỐC, CHẨN_ĐOÁN, TRIỆU_CHỨNG, TÊN_XÉT_NGHIỆM, KẾT_QUẢ_XÉT_NGHIỆM).\n\n"
        "🔥 QUY TẮC QUÉT KHÔNG ĐƯỢC BỎ SÓT BẤT KỲ ENTITY NÀO:\n"
        "1. Quét từ chữ đầu tiên đến chữ cuối cùng (Lý do vào viện, Tiền sử, Khám, CLS, Chẩn đoán, Điều trị).\n"
        "2. Đảm bảo thu thập ĐỦ 5 TYPE:\n"
        "   - 💊 THUỐC: Lấy cả thuốc đang dùng, thuốc tiền sử lẫn thuốc mới kê (giữ liều lượng, pattern `x 1`, `x 2`).\n"
        "   - 🩺 CHẨN_ĐOÁN: Lấy bệnh tiền căn, chẩn đoán xác định/ra viện và TẤT CẢ bất thường trên tim mạch/hình ảnh (`ngoại tâm thu`, `ST chênh lên`, `tim to`, `tràn dịch`).\n"
        "   - 🤒 TRIỆU_CHỨNG: Lấy đủ mọi than phiền (`đau ngực`, `khó thở`, `khó thở nhẹ`, `đánh trống ngực`, `mệt mỏi nhiều khi gắng sức`), lặp lại ở N câu thì lấy đủ N entities (R10 STRICT).\n"
        "   - 🔬 TÊN_XÉT_NGHIỆM: Lấy đủ chỉ định/thủ thuật (`X-quang ngực`, `ECG`, `nước tiểu`, `siêu âm tim`, `monitor holter`).\n"
        "   - 📊 KẾT_QUẢ_XÉT_NGHIỆM: Lấy đủ chỉ số định lượng (`160/90 mmHg`, `96%`, `38.5°C`) và kết quả định tính/bình thường (`nhịp xoang chiếm ưu thế`, `bình thường`, `không ghi nhận gì bất thường`).\n"
        "3. Tách bạch rõ ràng: Cắt sạch động từ dẫn (`cảm thấy`, `chụp`) & thời lượng rác (`trong tuần qua`).\n"
        "4. ⚠️ KHÔNG CẦN ghi `position` — chỉ cần ghi đúng `text`, `type`, `assertions`. Python sẽ tự tính offset chính xác.\n\n"
        f"{alert_part}"
        f"INPUT:\n{input_text}\n\n"
        "OUTPUT JSON ARRAY (chỉ các trường text, type, assertions — không cần position, không kèm lời giải thích):"
    )


# ==============================================================================
# TWO-STAGE PIPELINE PROMPTS (R32 - 2026-07-12)
# ==============================================================================

STAGE1_PROMPT = """Bạn là chuyên gia trích xuất thực thể y tế tiếng Việt.

# NHIỆM VỤ DUY NHẤT
Tìm TẤT CẢ các cụm từ (text spans) trong văn bản là KHÁI NIỆM Y TẾ thực sự thuộc 5 lĩnh vực lâm sàng:
(1) Thuốc, (2) Chẩn đoán / Bệnh danh / Bất thường CLS, (3) Triệu chứng lâm sàng, (4) Tên xét nghiệm / Thăm dò, (5) Kết quả xét nghiệm / Sinh hiệu / Kết quả bình thường.
CHỈ trả về text + position[start, end]. KHÔNG cần phân loại type hay assertions.

# NÊN TRÍCH (medical mentions)
- Tên thuốc: "aspirin 325mg x 1", "metoprolol 25mg po bid", "doxycycline"
- Tên bệnh/chẩn đoán/bất thường: "tăng huyết áp", "viêm tuyến mồ hôi", "nhồi máu cơ tim vùng dưới cũ", "ngoại tâm thu nhĩ", "ST chênh lên"
- Triệu chứng: "đau ngực", "khó thở", "khó thở nhẹ", "cảm giác đánh trống ngực", "mệt mỏi nhiều khi gắng sức", "thắt chặt ngực vùng trước tim"
- Tên xét nghiệm: "điện tâm đồ", "x-quang ngực", "siêu âm tim qua thành ngực", "phân tích nước tiểu", "monitor holter"
- Kết quả xét nghiệm/sinh hiệu: "bình thường", "không ghi nhận gì bất thường", "nhịp xoang chiếm ưu thế", "160/90 mmHg", "VS98.3 12987 56 18 99RA"

# KHÔNG ĐƯỢC TRÍCH (false positives / noise)
- Câu chuyện cá nhân / giao tiếp: "Tỉnh dậy thấy cháu gái hét lên", "cô ấy sẽ được phục vụ tốt hơn", "bệnh nhân đã đến khám"
- Hành vi / đánh giá chung: "theo đó", "sau đó", "kết quả cho thấy", "quyết định rằng", "nhận thấy", "chúng tôi sẽ"
- Từ chung chung mờ nhạt 1 từ: "bệnh", "đau", "mệt" (trừ khi là triệu chứng rõ ràng trong khám lâm sàng)
- Đoạn văn kể chuyện dài > 40-60 ký tự không chứa thông tin định lượng thuốc hay chỉ số

# QUY TẮC BOUNDARY CỐT LÕI
- Giữ chính xác text trong input, không thêm/bớt từ hoặc tự viết lại.
- KHÔNG gộp chỉ định vào chẩn đoán: "nhồi máu cơ tim vùng dưới cũ (điện tâm đồ (ecg))" → TÁCH thành các mentions riêng biệt: "nhồi máu cơ tim vùng dưới cũ", "điện tâm đồ", "ecg".
- Nếu một cụm từ xuất hiện lặp lại ở nhiều vị trí khác nhau trong văn bản → BẮT BUỘC trích xuất tất cả các lần xuất hiện với các position khác nhau.
- Khi gặp câu phủ định liệt kê nhiều triệu chứng ngăn cách bởi dấu phẩy/từ nối (ví dụ: "Không buồn nôn, hay nôn, đổ mồ hôi") → BẮT BUỘC trích xuất TẤT CẢ các từ triệu chứng riêng biệt ("buồn nôn", "nôn", "đổ mồ hôi"). KHÔNG ĐƯỢC chỉ lấy từ cuối cùng.
- Khi gặp tên thiết bị/thăm dò đứng ở đầu câu dẫn vào kết quả (ví dụ: "monitor holter cho thấy...", "siêu âm bụng ghi nhận...") → BẮT BUỘC trích xuất riêng tên thiết bị/thăm dò đó ("monitor holter", "siêu âm bụng") trước khi trích xuất kết quả phía sau.

# OUTPUT FORMAT
[
  {"text": "<exact text from input>", "position": [start, end]},
  ...
]

QUAN TRỌNG: position tính theo CHARACTER index trong input gốc (0-indexed).
"""

STAGE2_PROMPT = """Bạn là chuyên gia phân loại thực thể y tế tiếng Việt lâm sàng.

# NHIỆM VỤ
Cho một danh sách các cụm từ y tế (đã được trích xuất từ văn bản gốc kèm vị trí character position), hãy phân loại cho mỗi cụm từ:
- type: BẮT BUỘC chọn đúng 1 trong 5 loại:
  - THUỐC: Tên thuốc, hoạt chất, liều lượng (`aspirin 325mg po bid`).
  - CHẨN_ĐOÁN: Tên bệnh lý, hội chứng, tổn thương hình ảnh, bất thường ECG (`tăng huyết áp`, `ngoại tâm thu nhĩ`, `ST chênh lên`).
  - TRIỆU_CHỨNG: Biểu hiện cơ năng/thực thể, cảm giác lâm sàng (`đau ngực`, `khó thở`, `cảm giác đánh trống ngực`).
  - TÊN_XÉT_NGHIỆM: Chỉ định cận lâm sàng, thăm dò, thủ thuật (`điện tâm đồ`, `x-quang ngực`, `siêu âm tim`).
  - KẾT_QUẢ_XÉT_NGHIỆM: Chỉ số định lượng (`160/90 mmHg`, `96%`), chuỗi sinh hiệu (`VS98.3 12987...`) hoặc kết quả bình thường (`nhịp xoang chiếm ưu thế`, `bình thường`, `không ghi nhận gì bất thường`).
- assertions: Mảng chuỗi các nhãn ngữ cảnh lâm sàng, có thể chứa:
  - "isNegated": Nếu cụm từ bị phủ định trực tiếp (`không đau ngực`, `không có khó thở`). Lưu ý: Nếu một câu có nhiều triệu chứng phủ định ngăn cách bởi dấu phẩy như "Không A, B, hay C" thì CẢ A, B, C đều có assertion ["isNegated"]. Kết quả bình thường (`ECG bình thường`) KHÔNG negated!
  - "isHistorical": Nếu cụm từ thuộc phần tiền sử (`Tiền sử: THA 10 năm`) hoặc thuốc đang dùng trước nhập viện (`Thuốc trước nhập viện`).
  - "isFamily": Nếu cụm từ là bệnh của người thân (`bố bị ĐTĐ`, `tiền sử gia đình ung thư`).
  - Nếu không thuộc các trường hợp trên → assertions để mảng rỗng `[]`.

# ĐẦU VÀO
Văn bản gốc đầy đủ (để hiểu ngữ cảnh):
<input>
{input_text}
</input>

Danh sách mentions cần phân loại (kèm đoạn ngữ cảnh trích xuất xung quanh để phán đoán chính xác nhãn phủ định/tiền sử):
{mentions_list}

# ĐẦU RA
Trả về JSON array chứa đầy đủ các mentions đã phân loại:
[
  {{"text": "...", "position": [start, end], "type": "THUỐC|CHẨN_ĐOÁN|TRIỆU_CHỨNG|TÊN_XÉT_NGHIỆM|KẾT_QUẢ_XÉT_NGHIỆM", "assertions": ["isNegated", "isHistorical", "isFamily"]}},
  ...
]

# QUY TẮC
- Giữ nguyên chính xác `text` và `position` từ danh sách mentions đầu vào. KHÔNG đổi offset hoặc bỏ bớt mention nào.
- Dùng thông tin `| ngữ cảnh: "..."` đi kèm mỗi mention để xác định cực kỳ chính xác `type` và `assertions`.
"""


def build_stage1_user_prompt(input_text: str) -> str:
    """Build user prompt cho Stage 1 Mention Extraction."""
    return (
        "🎯 NHIỆM VỤ: Tìm và trích xuất TRỌN VẸN và KIỆT ĐỂ tất cả các cụm từ y khoa (medical concept spans) trong văn bản lâm sàng dưới đây kèm vị trí character offset [start, end].\n\n"
        "🔥 4 QUY TẮC TRÍCH XUẤT LÂM SÀNG CỐT LÕI (BẮT BUỘC TUÂN THỦ TỪNG CHỮ):\n"
        "1. TRIỆU CHỨNG LÕI NGẮN GỌN: CHỈ lấy core symptom (`đau ngực`, `khó thở`, `mệt mỏi`, `đánh trống ngực`, `sốt`). TUYỆT ĐỐI KHÔNG bốc thêm đuôi tự sự / hoàn cảnh phía sau (`nhiều khi gắng sức`, `khi leo cầu thang`, `lúc nhập viện`) hoặc tiền tố lời kể (`còn cảm giác`, `bệnh nhân thấy`).\n"
        "2. TÁCH CỤM TRIỆU CHỨNG VỊ TRÍ KÉP: Nếu có cả cảm giác và vị trí giải phẫu (`cảm giác thắt chặt ngực vùng trước tim`, `tình trạng đau thắt ngực sau xương ức`), PHẢI tách thành 2 spans riêng: (`cảm giác thắt chặt ngực` VÀ `thắt chặt ngực vùng trước tim`), KHÔNG gộp chung 1 dải.\n"
        "3. THUỐC PHẢI ĐỦ ĐUÔI LIỀU LƯỢNG (`x N`): Khi có `aspirin 325mg x 1`, `paracetamol 500mg po bid`, PHẢI lấy trọn vẹn đến hết đuôi liều/tần suất (`aspirin 325mg x 1`), không được bỏ rơi chữ `x 1` phía sau.\n"
        "4. QUÉT HẾT TỪNG LẦN LẶP LẠI: Nếu một triệu chứng hay thuốc xuất hiện 3-4 lần ở các câu khác nhau từ Tiền sử đến Cấp cứu đến Khám, PHẢI xuất đủ 3-4 lần với positions tương ứng!\n\n"
        f"INPUT:\n{input_text}\n\n"
        "OUTPUT JSON ARRAY (chỉ trả về [{'text': '...', 'position': [start, end]}], không kèm lời giải thích):"
    )


def build_stage2_user_prompt(input_text: str, mentions: list[dict]) -> str:
    """Build user prompt cho Stage 2 Classification, kèm theo Local Context Injection (±45 chars)."""
    lines = []
    for i, m in enumerate(mentions):
        text = m.get("text", "")
        pos = m.get("position", [0, 0])
        snippet = ""
        if isinstance(pos, list) and len(pos) == 2:
            try:
                s, e = int(pos[0]), int(pos[1])
                if 0 <= s <= e <= len(input_text):
                    ctx_s = max(0, s - 45)
                    ctx_e = min(len(input_text), e + 45)
                    raw_snippet = input_text[ctx_s:ctx_e].strip()
                    clean_snippet = re.sub(r'\s+', ' ', raw_snippet)
                    snippet = f' | ngữ cảnh: "...{clean_snippet}..."'
            except (ValueError, TypeError):
                pass
        lines.append(f"- {i+1}. text=\"{text}\" position={pos}{snippet}")
    mentions_str = "\n".join(lines)
    return STAGE2_PROMPT.format(
        input_text=input_text,
        mentions_list=mentions_str,
    )


ICD_LLM_FALLBACK_PROMPT = """Bạn là bác sĩ chuyên gia. Hãy đề xuất ICD-10 code phù hợp nhất.

Cho một chẩn đoán y tế tiếng Việt, trả về danh sách ICD-10 code(s) chính xác nhất.

QUY TẮC:
- Ưu tiên code cụ thể (3-4 ký tự) hơn code cha
- Nếu không chắc chắn, trả 1 code gần nhất
- Format: I21, I21.1, K70.3, J18, etc.
- KHÔNG trả code ngoài ICD-10
- KHÔNG giải thích

Input: "{entity_text}"
Context: {context_window}

Output: [code1, code2, ...]
"""

RXNORM_LLM_FALLBACK_PROMPT = """Bạn là dược sĩ lâm sàng. Hãy đề xuất RxNorm rxcui code cho thuốc.

Cho tên thuốc tiếng Việt (có thể kèm liều), trả về rxcui code (số nguyên).

QUY TẮC:
- Trả về rxcui code chính xác nhất cho hoạt chất/thuốc
- Bỏ qua thông tin route (po, iv, bid) và tần suất
- Nếu không chắc, trả [] (không trả sai code)

Input: "{drug_text}"

Output: [rxcui1, rxcui2, ...]
"""

