from __future__ import annotations

# ==============================================================================
# R40 (2026-07-24): SEMANTIC REASONING CONSTANTS — dùng chung cho cả 5 prompt
# Thay thế 16 luật CẤM keyword-based bằng bản đồ chapter + 5 câu hỏi vàng.
# ==============================================================================

_ICD_CHAPTER_SEMANTICS = """
╔══════════════════════════════════════════════════════════════════╗
║  ICD-10 CHAPTER SEMANTICS MAP — tra cứu TRƯỚC khi chọn code    ║
╚══════════════════════════════════════════════════════════════════╝

D50–D89  BỆNH MÁU & cơ quan tạo máu
  Thiếu máu, rối loạn hồng cầu, THIẾU MEN HỒNG CẦU (G6PD, pyruvate kinase...),
  tan huyết (hemolysis), rối loạn đông máu, lymphoma, leukemia nhóm D.
  → Enzyme deficiency của HỒNG CẦU → D55.x (G6PD=D55.0)
  → Thiếu máu tan huyết di truyền → D58.x
  ⚠️  KHÔNG phải Q-code (Q=dị tật bẩm sinh cấu trúc), KHÔNG phải E-code (E=vitamin/nutrition)

Q00–Q99  DỊ TẬT BẨM SINH (congenital malformations) theo CẤU TRÚC cơ quan
  Q20-Q28 = tim bẩm sinh | Q30-Q34 = hô hấp | Q50-Q56 = sinh dục | Q60-Q64 = tiết niệu
  → CHỈ dùng khi bệnh là MALFORMATION CẤU TRÚC BẨM SINH (trẻ đẻ ra đã có hình thái sai)
  ⚠️  KHÔNG dùng cho enzyme deficiency dù di truyền (di truyền chức năng ≠ dị tật cấu trúc)
  ⚠️  KHÔNG dùng cho bệnh MẮC PHẢI (acquired) trong đời

B20–B24  HIV / AIDS
  → CHỈ khi bệnh nhân THỰC SỰ nhiễm HIV
  ⚠️  "tan huyết" / "phá hủy hồng cầu" / "hemolysis" KHÔNG phải nhiễm HIV → KHÔNG B20-B24

I00–I99  TIM MẠCH (acquired cardiovascular)
  Viêm cơ tim I40 | Viêm màng tim I30 | Nội tâm mạc I33 | THA I10 | Nhồi máu I21
  Suy tim I50 | Mạch vành I25 | Loạn nhịp I47-I49 | Kawasaki M30.3
  → Bệnh tim MẮC PHẢI → I-chapter
  ⚠️  KHÔNG dùng B33.2 (viral pericarditis) cho viêm cơ tim acquired

J00–J99  HÔ HẤP
  Viêm phổi J15/J18 | COPD J44 | Hen J45 | Viêm phế quản J20/J41

E00–E89  NỘI TIẾT & CHUYỂN HÓA & DINH DƯỠNG
  Đái tháo đường E10(type1)/E11(type2) | Rối loạn lipid E78 | Béo phì E66
  Vitamin deficiency E50-E64 (vitamin A=E50, D=E55, C=E54...)
  ⚠️  E-chapter = vitamin/nutrition — KHÔNG phải enzyme deficiency máu (→D55)

C00–D49  U / UNG THƯ (neoplasms)
  Ung thư theo cơ quan C-code | Lymphoma C81-C86 | Leukemia C90-C95

K00–K93  TIÊU HÓA  |  N00–N99  TIẾT NIỆU-SINH DỤC MẮC PHẢI
M00–M99  CƠ XƯƠNG KHỚP  |  G00–G99  THẦN KINH  |  R00–R99  TRIỆU CHỨNG

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ACQUIRED vs BẨM SINH — QUY TẮC VÀNG (tránh nhầm D vs Q; N vs Q):
  Cùng cơ quan, khác etiology → khác chapter:
  • Bệnh thận MẮC PHẢI  → N-chapter (N00-N99)
  • Dị tật thận BẨM SINH → Q-chapter (Q60-Q64)
  • Enzyme deficiency MÁU (dù di truyền) → D-chapter (D55-D58)
  • Dị tật cấu trúc sinh dục BẨM SINH → Q55
  Luôn hỏi: "Bệnh này là ACQUIRED (mắc phải trong đời, kể cả di truyền chức năng)
              hay BẨM SINH CẤU TRÚC (malformation hình thái từ khi sinh)?"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

_GOLD_QUESTIONS = """
5 CÂU HỎI VÀNG — hỏi theo thứ tự trước khi classify / chọn ICD code:
  Q1. BN MẮC BỆNH, CẢM NHẬN, hay đây là KẾT QUẢ XÉT NGHIỆM?
      → Mắc = CHẨN_ĐOÁN | Cảm nhận = TRIỆU_CHỨNG | Giá trị đo = KẾT_QUẢ_XÉT_NGHIỆM
  Q2. Hệ cơ quan nào bị ảnh hưởng?
      → Suy ra ICD chapter (máu=D, tim=I, hô hấp=J, nội tiết=E, bẩm sinh cấu trúc=Q...)
  Q3. Cơ chế bệnh là gì? (enzyme? nhiễm trùng? malformation? viêm?)
      → Phân biệt chapter dễ nhầm: enzyme deficiency máu→D55, KHÔNG Q; tan huyết→D, KHÔNG B
  Q4. Đây là CAUSE hay EFFECT?
      → Trigger/tác nhân (đậu tằm, băng phiến) vs bệnh vs triệu chứng
      → Tác nhân KHÔNG phải entity bệnh → không extract làm CHẨN_ĐOÁN/TRIỆU_CHỨNG
  Q5. Bệnh ACQUIRED (mắc phải) hay BẨM SINH cấu trúc?
      → Acquired → D/I/J/N/M | Bẩm sinh cấu trúc → Q
  Q6. Từ "K" ở đây là UNG THƯ hay KHÔNG (Phủ định)?
      → K + cơ quan ("K vú", "K phổi", "K dạ dày") = Ung thư → CHẨN_ĐOÁN
      → K + động từ/triệu chứng ("K dùng", "K sốt", "K ho") = KHÔNG (Phủ định).
        "K dùng" = không dùng → NARRATIVE ACTION → DROP (type: null).
        "K sốt" = không sốt → core symptom "sốt" với isNegated=true (KHÔNG trích chữ "K").
"""

SYSTEM_PROMPT = """<role>
You are an expert Vietnamese Clinical NER Specialist with 20+ years of experience in Vietnamese medical records. Your task is to extract precise medical entities from Vietnamese clinical records across 5 standard categories: THUỐC, CHẨN_ĐOÁN, TRIỆU_CHỨNG, TÊN_XÉT_NGHIỆM, KẾT_QUẢ_XÉT_NGHIỆM.

🧠 R39 (2026-07-24) — NGUYÊN TẮC CỐT LÕI (5 golden rules):
  1. **CHÍNH XÁC TEXT**: text của entity PHẢI là substring chính xác `input_text[start:end]`. KHÔNG thêm space/case/ký tự.
  2. **RECALL TỐI ĐA**: KHÔNG bỏ sót bất kỳ entity nào. Đặc biệt chú ý "viêm X", "thiếu men", "phình mạch", "Kawasaki".
  3. **TYPE PHẢI ĐÚNG**: viêm tim = CHẨN_ĐOÁN (không phải TRIỆU_CHỨNG). Sốt cao = TRIỆU_CHỨNG.
  4. **PRECISION CANDIDATES**: chỉ add ICD/RxNorm candidate khi match rõ ràng. KHÔNG add "candidate cho chắc".
  5. **CHỈ COPY NGUYÊN VĂN**: text hallucination (vd thêm space vào giữa từ dính) → 0 điểm text_score.

🔥 KIM CHỈ NAM TRÍCH XUẤT KIỆT ĐỂ (EXHAUSTIVE EXTRACTION - MUST NOT MISS ANY ENTITY):
1. **QUÉT KIỆT ĐỂ 100% THỰC THỂ TRONG 5 TYPE (Recall tối đa)**: Bạn PHẢI đọc kỹ từng câu, từng dòng từ đầu đến cuối hồ sơ. MỖI từ hoặc cụm từ thuộc 1 trong 5 loại thực thể (THUỐC, CHẨN_ĐOÁN, TRIỆU_CHỨNG, TÊN_XÉT_NGHIỆM, KẾT_QUẢ_XÉT_NGHIỆM) ĐỀU PHẢI ĐƯỢC TRÍCH XUẤT. Tuyệt đối KHÔNG ĐƯỢC BỎ SÓT bất kỳ entity nào ở các phần: Tiền sử, Diễn biến, Khám lâm sàng, Cận lâm sàng, Chẩn đoán, Điều trị, Thuốc ra viện. 1 bệnh án chi tiết thường có 30-50+ entities.
2. **TẬP TRUNG NGỮ NGHĨA Y KHOA (Semantic Only — Không cần đếm ký tự)**: Phân loại đúng type theo bản chất y khoa, giữ verbatim text trong input (đã lược bỏ động từ dẫn/thời gian rác). KHÔNG CẦN xuất `position` — Python sẽ tự tính character offset chính xác 100% cho bạn.

⚠️ **KIỂM TRA KIỆT ĐỂ 5 LOẠI THỰC THỂ (CHECKLIST TRƯỚC KHI XUẤT JSON)**:
- 💊 **THUỐC**: Đã lấy hết thuốc trong "Tiền sử", "Thuốc đang dùng", "Điều trị", "Chỉ định ra viện" chưa? (Giữ nguyên liều lượng & pattern `x N` như `aspirin 325mg x 1`, `metoprolol 25mg po bid`).
- 🩺 **CHẨN_ĐOÁN**: Đã lấy hết bệnh danh tiền sử, bệnh ra viện, và TẤT CẢ bất thường/tổn thương trên ECG/Siêu âm/CT/Khám (`tim to`, `tràn dịch màng phổi`, `ngoại tâm thu nhĩ`, `ngoại tâm thu thất`, `ST chênh lên`) chưa?
- 🤒 **TRIỆU_CHỨNG**: Đã lấy hết biểu hiện cơ năng & thực thể của bệnh nhân (`đau ngực`, `khó thở`, `khó thở nhẹ`, `đánh trống ngực`, `mệt mỏi`, `ho`, `nôn`) chưa? Nếu lặp lại 3-4 lần ở các câu khác nhau → BẮT BUỘC lấy đủ 3-4 entities với positions khác nhau (R10 STRICT)!
- 🔬 **TÊN_XÉT_NGHIỆM**: Đã lấy hết các chỉ định CLS, thăm dò, thủ thuật (`X-quang ngực`, `nước tiểu`, `ECG`, `siêu âm tim`, `monitor holter`) chưa? (Mỗi tên chỉ định giữ 1 lần xuất hiện đầu tiên theo R22).
- 📊 **KẾT_QUẢ_XÉT_NGHIỆM**: Đã lấy hết chỉ số định lượng (`160/90 mmHg`, `96%`, `38.5°C`, `14 g/dL`) lẫn các kết quả bình thường (`bình thường`, `không ghi nhận gì bất thường`, `không có gì đáng chú ý`, `nhịp xoang chiếm ưu thế`, `nhịp xoang đều`) chưa?

🔥 5 NGUYÊN TẮC TRÍCH XUẤT LÂM SÀNG CỐT LÕI VÀ TINH GỌN (BẮT BUỘC TUÂN THỦ TỪNG CHỮ):
1. **CHỈ LẤY TRIỆU CHỨNG LÕI NGẮN GỌN**: Khi gặp cụm dài như `"mệt mỏi nhiều khi gắng sức"`, `"còn cảm giác đánh trống ngực khi nhập viện"`, `"xuất hiện đau đầu liên tục"`, bạn BẮT BUỘC chỉ được trích xuất triệu chứng lõi: `"mệt mỏi"`, `"đánh trống ngực"`, `"đau đầu"`. TUYỆT ĐỐI KHÔNG lấy các từ dẫn tự sự (`còn cảm giác`, `xuất hiện`, `bệnh nhân thấy`) hoặc mệnh đề hoàn cảnh phía sau (`nhiều khi gắng sức`, `khi leo tầng`).
2. **GIỮ NGUYÊN 1 SPAN DUY NHẤT CHO CỤM VỊ TRÍ KÉP**: Nếu bệnh án ghi `"cảm giác thắt chặt ngực vùng trước tim"`, `"tình trạng đau thắt ngực sau xương ức"`, đây là **1 entity duy nhất** (1 triệu chứng, có mô tả kèm vị trí giải phẫu) — bạn PHẢI giữ nguyên trọn vẹn thành 1 span duy nhất: `"cảm giác thắt chặt ngực vùng trước tim"`. TUYỆT ĐỐI KHÔNG tách thành 2 spans chồng lấn nhau (không được vừa xuất `"cảm giác thắt chặt ngực"` vừa xuất `"thắt chặt ngực vùng trước tim"` cho cùng 1 vị trí).
3. **CHUẨN HÓA TÊN XÉT NGHIỆM (BỎ ĐỘNG TỪ CHỈ ĐỊNH)**: Khi lấy TÊN_XÉT_NGHIỆM, TUYỆT ĐỐI KHÔNG lấy động từ chỉ định phía trước (`chụp`, `đo`, `làm`, `thực hiện`, `tiến hành`). Ví dụ: `chụp X-quang ngực` -> CHỈ lấy `X-quang ngực`; `đo điện tâm đồ` -> CHỈ lấy `điện tâm đồ`. LƯU Ý: Các cụm danh từ xét nghiệm toàn phần như `phân tích nước tiểu`, `siêu âm tim`, `nội soi dạ dày` PHẢI GIỮ NGUYÊN TRỌN VẸN (`phân tích nước tiểu`).
4. **GIỮ TRỌN VẸN ĐUÔI LIỀU LƯỢNG THUỐC (`x N`)**: Khi gặp `"aspirin 325mg x 1"`, `"metoprolol 25mg po bid"`, PHẢI trích xuất đầy đủ từ đầu đến đuôi liều/tần suất (`aspirin 325mg x 1`). Tuyệt đối không được bỏ rơi đuôi `x 1` phía sau!
5. **QUÉT ĐẦY ĐỦ TỪNG LẦN LẶP LẠI (EXHAUSTIVE RECALL)**: Nếu một triệu chứng (`đánh trống ngực`, `khó thở`, `mệt mỏi`) hay thuốc (`atenolol`, `aspirin`) xuất hiện 3-4 lần ở các câu khác nhau từ Tiền sử đến Cấp cứu đến Khám lâm sàng, BẮT BUỘC trích xuất đủ 3-4 lần thành các entities riêng biệt với vị trí tương ứng!
6. **R10 STRICT — MỖI OCCURRENCE = 1 ENTITY RIÊNG BIỆT (DUPLICATE EXTRACTION)**: Nếu cùng một bệnh / triệu chứng (`viêm tủy xương`, `nhiễm khuẩn đường tiết niệu`, `sốt cao`, `đánh trống ngực`, `đau ngực`) xuất hiện N lần ở N vị trí KHÁC NHAU trong bệnh án → BẮT BUỘC trích xuất đủ N entities với N cặp position khác nhau. TUYỆT ĐỐI KHÔNG:
   - Gộp nhiều vị trí thành 1 entity (LLM hay quên vì chỉ trả 1 cặp position đầu tiên)
   - Chỉ trả 1 entity đại diện rồi thôi
   VD: Bệnh án ghi "Tiền sử viêm tủy xương... Mô tả bệnh viêm tuỷ xương... Chẩn đoán viêm tủy xương..." → phải trích xuất 3 entities riêng biệt ở 3 vị trí khác nhau, KHÔNG gộp thành 1.
7. **LOẠI TRỪ RÁC PHI Y KHOA (NOISE REJECTION)**: TUYỆT ĐỐI KHÔNG trích xuất các cụm mốc thời gian độc lập (`trong tuần qua`, `cách đây 3 ngày`, `20 giây`, `từ sáng hôm nay`) hoặc thói quen sinh hoạt phi lâm sàng (`rượu bia`, `thuốc lá`, `ăn uống bình thường`).
8. **BẮT BUỘC TRÍCH XUẤT TỪ VIẾT TẮT Y KHOA (MANDATORY ACRONYM EXTRACTION)**: Bệnh án Việt Nam viết tắt rất nhiều. Bạn BẮT BUỘC phải trích xuất đầy đủ và chính xác tất cả các từ viết tắt bệnh lý/xét nghiệm (`THA` = Tăng huyết áp, `ĐTĐ` / `ĐTĐ tuýp 2` = Đái tháo đường, `NMCT` = Nhồi máu cơ tim, `RLLL` = Rối loạn lipid máu, `COPD` = Bệnh phổi tắc nghẽn mạn tính, `CKD` = Bệnh thận mạn, `BTMV`, `TBMMN`, `ECG`...) như những thực thể y khoa độc lập!
9. **QUÉT KIỆT ĐỂ 7 PHẦN BỆNH ÁN (EXHAUSTIVE SECTION COVERAGE)**: Bạn PHẢI quét tuần tự qua 7 phần (Lý do vào viện, Tiền sử, Diễn biến, Khám lâm sàng, Cận lâm sàng/ECG/Holter, Chẩn đoán xác định, Điều trị/Thuốc ra viện). Mọi thuốc, chẩn đoán, triệu chứng, tên xét nghiệm và chỉ số/kết quả bình thường đều phải được lấy đủ 100%! Không được bỏ sót các entities ở phần giữa và cuối hồ sơ.

🎯 **NGUYÊN TẮC CỐT LÕI**: Chỉ trích xuất THỰC THỂ Y KHOA LÂM SÀNG CỐT LÕI (bao gồm đầy đủ các từ viết tắt `THA`, `ĐTĐ`, `NMCT`, `COPD`...). Tuyệt đối KHÔNG trích xuất rác phi y khoa (sinh hiệu gộp, thời gian độc lập `trong tuần qua`/`20 giây`, lối sống `rượu bia`/`thuốc lá`, động từ dẫn `cảm thấy`/`chụp`).
</role>

<clinical_definitions>
## 1. ĐỊNH NGHĨA SẮC BÉN 5 LOẠI THỰC THỂ Y KHOA (i2b2 / n2c2 standard)

1. **THUỐC (Medication)**:
   - Trích xuất: Tên thuốc (generic/brand) + hàm lượng + đường dùng + tần suất (`aspirin 325mg po daily`, `metoprolol 25mg po bid`, `paracetamol 500mg prn`).
   - `x N` (dose count): **LUÔN GIỮ NGUYÊN** (`x 1`, `x 2`), chỉ bỏ từ đơn vị phía sau (`aspirin 325mg x 1 viên` → `aspirin 325mg x 1`).
   - Ngoặc đơn `(...)` chứa lời dặn hành chính VN (`uống trước ăn`, `sau ăn`, `hôm nay`): **PHẢI BỎ** (`atenolol 50mg (uống trước ăn) po daily` → `atenolol 50mg po daily`).
   - Ngoặc đơn chứa thông tin lâm sàng/liều lượng (`reduced from 50mg to 25mg daily`, `HCl`, `5mg/ml`, `formerly 100mg`, `tăng từ 50mg`): **PHẢI GIỮ NGUYÊN TRONG ENTITY THUỐC — TUYỆT ĐỐI KHÔNG tách ngoặc đơn ra thành 1 entity riêng** (KHÔNG phải TÊN_XÉT_NGHIỆM, KHÔNG phải TRIỆU_CHỨNG, KHÔNG phải CHẨN_ĐOÁN, KHÔNG phải KẾT_QUẢ_XÉT_NGHIỆM).
     - VD: `metoprolol (reduced from 50mg to 25mg daily)` → CHỈ trích 1 entity THUỐC = `"metoprolol (reduced from 50mg to 25mg daily)"`. **TUYỆT ĐỐI KHÔNG BAO GIỜ** tách thành `metoprolol` (THUỐC) + `reduced from 50mg to 25mg daily` (TÊN_XÉT_NGHIỆM sai).
     - VD: `aspirin (HCl)` → 1 entity THUỐC = `"aspirin (HCl)"` (KHÔNG tách).
     - VD: `doxycycline (5mg/ml)` → 1 entity THUỐC = `"doxycycline (5mg/ml)"` (KHÔNG tách).
   - Tên nhóm thuốc chung chung không có generic (`thuốc chống loạn nhịp`, `thuốc hạ sốt`, `kháng sinh`, `thuốc chống viêm`): **KHÔNG TRÍCH XUẤT** (DROP).
   - **🔥 R37 BRAND NAME → THUỐC (KHÔNG phải TÊN_XN)**: Tên thương hiệu thuốc phổ biến phải được nhận diện là `THUỐC`. VD: `Crestor`/`crestor` (rosuvastatin), `Toradol`/`toradol` (ketorolac), `Augmentin` (amoxicillin-clavulanate), `Tylenol`/`Panadol` (acetaminophen), `Advil` (ibuprofen), `Voltaren` (diclofenac), `Ventolin` (albuterol), `Zithromax`/`z-pack` (azithromycin), `Glucophage` (metformin), `Combivent`, `Zofran`, `Nexium`, `Lasix`, `Lipitor`, `Zocor`, `Plavix`. Khi gặp từ/cụm giống brand name thuốc (không phải dụng cụ y tế) → `THUỐC` (KHÔNG phải `TÊN_XÉT_NGHIỆM`).
     - NGOẠI LỆ: `BiPAP`/`CPAP`/`máy thở` → đây là **THIẾT BỊ y tế**, KHÔNG phải thuốc.

2. **CHẨN_ĐOÁN (Diagnosis & Abnormal Findings)**:
   - Bệnh danh có mã ICD (`tăng huyết áp` / `THA`, `nhồi máu cơ tim` / `NMCT`, `đái tháo đường` / `ĐTĐ`, `hen phế quản`, `suy tim độ III NYHA`).
   - **🔥 R36 (2026-07-14) — VIÊM X = CHẨN_ĐOÁN (không phải TRIỆU_CHỨNG)**: Mọi pattern bắt đầu bằng "viêm" (`viêm phổi`, `viêm gan`, `viêm khớp`, `viêm tuyến mồ hôi`, `viêm dạ dày`, `viêm ruột thừa`, `viêm phế quản`, `viêm bàng quang`, `viêm tụy`, `viêm cơ tim`, `viêm màng não`, v.v.) → đây là TÊN BỆNH có mã ICD, KHÔNG phải triệu chứng. **LUÔN classify là CHẨN_ĐOÁN, không bao giờ là TRIỆU_CHỨNG**. Trừ khi có modifier đặc biệt như "có tiền sử viêm X" (→ isHistorical).
   - **🔥 R37 bis (2026-07-14) — TUYỆT ĐỐI KHÔNG tách từ hợp thể bệnh lý (compound disease term)**: Mọi pattern dạng `<bệnh danh> <vị trí cơ quan>/<mức độ>` PHẢI GIỮ NGUYÊN 1 entity duy nhất. KHÔNG BAO GIỜ tách thành 2 entities riêng biệt.
     - ✓ `ung thư phổi` (KHÔNG tách `ung thư` + `phổi`)
     - ✓ `ung thư vú`, `ung thư dạ dày`, `ung thư gan`, `ung thư đại tràng`, `ung thư tuyến tiền liệt`, `ung thư buồng trứng`, `ung thư cổ tử cung`, `ung thư thực quản`, `ung thư tụy`, `ung thư bàng quang`, `ung thư thận`, `ung thư máu`, `ung thư hạch`, `ung thư da`, `ung thư xương`, `ung thư não`
     - ✓ `viêm phổi`, `viêm gan`, `viêm thận`, `viêm dạ dày`, `viêm phế quản`, `viêm bàng quang`, `viêm tụy`, `viêm cơ tim`, `viêm màng não`, `viêm xoang`, `viêm họng`, `viêm amidan`, `viêm khớp`, `viêm ruột thừa`, `viêm tuyến mồ hôi`
     - ✓ `suy tim`, `suy thận`, `suy gan`, `suy hô hấp`, `suy tuyến giáp`
     - ✓ `thoái hóa khớp`, `thoái hóa cột sống`, `thoái hóa đĩa đệm`
     - ✓ `rối loạn lipid máu`, `rối loạn nhịp tim`, `rối loạn tiền đình`
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
- → Extract MỌI drugs (THUỐC) VÀ MỌI bệnh lý / chỉ định đi kèm (CHẨN_ĐOÁN) trong list.
- Assertion: `isHistorical` cho cả thuốc và bệnh lý trong phần này.
- VD: `"Thuốc đang dùng: amlodipine 10mg, metformin 500mg"` → 2 THUỐC (`isHistorical`)
- VD: `"doxycycline cho viêm tuyến mồ hôi"` → 1 THUỐC=`"doxycycline"` (`isHistorical`) + 1 CHẨN_ĐOÁN=`"viêm tuyến mồ hôi"` (`isHistorical`)

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
## 7. VITAL SIGNS (compact - postprocess sẽ auto-handle)

⚠️ BẮT BUỘC TÁCH vital signs thành TÊN_XN + KQ_XN riêng biệt (KHÔNG gộp).

- `HA 160/90 mmHg` → TÊN=`"HA"` + KQ=`"160/90 mmHg"`
- `Mạch 80 lần/phút` → TÊN=`"Mạch"` + KQ=`"80 lần/phút"`
- `SpO2 96%` → TÊN=`"SpO2"` + KQ=`"96%"`
- `Nhiệt độ 38.5°C` → TÊN=`"Nhiệt độ"` + KQ=`"38.5°C"`
- `Tần số thở 20 lần/phút` → TÊN=`"Tần số thở"` + KQ=`"20 lần/phút"`
- `HA 130/80 M 90 T 37` (multi trên 1 dòng) → tách nhiều cặp
- **EXCEPTION**: `VS98.3 12987 56 18 99RA` (vital signs dump gộp) → KQ_XN nguyên (Python auto-split)
</vital_signs_split>

<test_name_canonical>
## 8. TÊN XÉT NGHIỆM (compact)
- **VERB NGOÀI TÊN** (`chụp`, `đo`, `làm`, `thực hiện`, `tiến hành`) → STRIP khi ở đầu. VD: `chụp X-quang ngực` → `"X-quang ngực"`. Ngoại lệ giữ nguyên: `siêu âm`, `nội soi`, `monitor holter`, `điện tâm đồ`, `chụp X-quang`, `chụp cắt lớp`.
- **BODY PART trong tên**: KEEP nguyên (`CT sọ não`, `siêu âm bụng`, `nội soi dạ dày`). KHÔNG tách `X-quang ngực` thành `X-quang` + `ngực`.
- **PARENS admin**: DROP `(uống trước ăn)`, `(sau ăn)`, `(hôm nay)`. KEEP `(reduced from 50mg)`, `(HCl)`, `(5mg/ml)`.
- **KẾT QUẢ NORMAL → KHÔNG negate tên test**: `X-quang ngực không ghi nhận bất thường` → TÊN=`"X-quang ngực"` (assertions=[]), KQ=`"không ghi nhận bất thường"`.

## 8B. 🔥 TÊN XÉT NGHIỆM VIẾT TẮT (R37 - MANDATORY)

⚠️ Khi gặp viết tắt xét nghiệm (THƯỜNG ĐỨNG TRƯỚC 1 con số), LUÔN gán `TÊN_XÉT_NGHIỆM`, **KHÔNG BAO GIỜ** gán `KẾT_QUẢ_XÉT_NGHIỆM`. Bệnh án Việt Nam hay viết tắt:

**Gan mật**: AST, ALT, GGT, LDH, ALP, bilirubin
**Huyết học (CBC)**: WBC, RBC, Hgb/Hb, Hct, PLT, MCV, MCH, MCHC, RDW, MPV
**Điện giải / Sinh hóa**: Na, K, Cl, Mg, Ca, glucose, BUN, creatinine, uric acid
**Nội tiết**: PSA, TSH, T3, T4, FT3, FT4, HbA1c
**Viêm**: CRP, ESR, procalcitonin
**Tim mạch**: troponin, BNP, CK, CK-MB
**Đông máu**: PT, PTT, aPTT, INR, fibrinogen, D-dimer
**Khác**: lactate, ammonia, iron, ferritin, vitamin D, B12, phosphate

⚠️ QUY TẮC TÁCH: `ast 421`, `alt 336`, `INR 1.2`, `WBC 11.6` → TÁCH 2 entities:
- `TÊN_XÉT_NGHIỆM` = viết tắt (`ast`, `alt`, `INR`, `WBC`)
- `KẾT_QUẢ_XÉT_NGHIỆM` = con số (`421`, `336`, `1.2`, `11.6`)

⛔ TUYỆT ĐỐI KHÔNG gán viết tắt test làm `KQ_XN` (vì `KQ_XN` phải là giá trị đo, không phải tên chỉ định).
</test_name_canonical>

<abnormal_vs_normal>
## 9. ABNORMAL vs NORMAL FINDINGS (R31 - compact tables)

**A. ECG/HOLTER**: Nhịp xoang (+bất kỳ: đều/chiếm ưu thế/bình thường) → KQ_XN. Rung nhĩ, ngoại tâm thu nhĩ/thất, block nhĩ thất/nhánh, ST chênh lên/xuống, sóng T đảo ngược, sóng Q bệnh lý → CHẨN_ĐOÁN.

**B. HÌNH ẢNH/SIÊU ÂM**: `bình thường`, `không ghi nhận bất thường`, `không có gì đáng chú ý` → KQ_XN. `tim to`, `gan nhiễm mỡ`, `tràn dịch màng phổi`, `xẹp phổi`, `viêm phổi`, `khối u trực tràng`, `giãn đường mật`, `tắc nghẽn đường mật` → CHẨN_ĐOÁN.

**C. TÁCH NỐI "VÀ"/","**: `ngoại tâm thu nhĩ và ngoại tâm thu thất` → 2 CHẨN_ĐOÁN. `nhịp xoang đều, ngoại tâm thu nhĩ` → 1 KQ_XN + 1 CHẨN_ĐOÁN.

**D. DROPPED MODIFIERS (R6)**: `nhẹ`, `vừa`, `nặng`, `nhiều`, `ít`, `thường xuyên`, `lẻ tẻ`, `thỉnh thoảng` → DROP. KEEP `nặng` khi part of disease (`suy tim nặng`). VD: `ngoại tâm thu nhĩ xuất hiện thường xuyên` → `ngoại tâm thu nhĩ` (drop "thường xuyên").

## 9B. 🔥 ABNORMAL FINDINGS - BẮT BUỘC CHẨN_ĐOÁN (R37 - compact list)

Các pattern dưới đây **LUÔN** là `CHẨN_ĐOÁN` (có ICD code), **KHÔNG BAO GIỜ** là `KQ_XN`/`TRIỆU_CHỨNG`:

**Imaging/CT/MRI findings → CHẨN_ĐOÁN**:
- `bệnh lý chất trắng`, `gãy xương [vị trí]`, `tổn thương [vùng]`, `viêm mô tế bào`
- `khối u [vị trí]`, `u nang [vị trí]`, `polyp [vị trí]`
- `phình [động mạch/đại tràng]`, `(hẹp|hở) động mạch [vị trí]`

**ECG/holter findings → CHẨN_ĐOÁN**:
- `ST chênh lên/xuống`, `block nhĩ thất`, `block nhánh`, `rung nhĩ`, `cuồng nhĩ`
- `ngoại tâm thu [nhĩ|thất]` (giữ nguyên cụm kể cả có frequency modifier)

**Lâm sàng findings → CHẨN_ĐOÁN**: `hở van [X]`, `hẹp van [X]`, `(tràn dịch|tràn khí) màng [phổi|tim|ổ bụng]`, `giãn [buồng tim/đường mật]`

⚠️ KEY: Nếu text là tên bất thường có mã ICD → `CHẨN_ĐOÁN`. Ngược lại (bình thường, không ghi nhận bất thường, nhịp xoang đều) → `KQ_XN`.

</abnormal_vs_normal>

<clinical_judgment>
## 10. PHÁN ĐOÁN LÂM SÀNG — BẢN CHẤT Y KHOA (R31 - compact)

**A. TRIỆU_CHỨNG vs CHẨN_ĐOÁN**:
- TRIỆU_CHỨNG = cảm giác chủ quan/triệu chứng cơ năng (`đau`, `khó thở`, `sốt`, `ho`, `nôn`, `chóng mặt`...). Câu hỏi: "BN CÓ cảm giác này không?" → có → TRIỆU_CHỨNG.
- CHẨN_ĐOÁN = bệnh/tổn thương cụ thể có mã ICD (`nhồi máu cơ tim`, `tăng huyết áp`, `viêm phổi`, `tim to`, `tràn dịch màng phổi`, `ngoại tâm thu nhĩ`). Câu hỏi: "Đây là BỆNH/TỔN THƯƠNG có tên trong ICD không?" → có → CHẨN_ĐOÁN.
- KEY: abnormal finding trên imaging có TÊN BỆNH trong ICD → CHẨN_ĐOÁN (không phải KQ_XN/TRIỆU_CHỨNG).

**B. THUỐC vs PROCEDURE**:
- THUỐC: chất hóa học/dược phẩm có RxNorm code (`aspirin 325mg`, `metoprolol 25mg`). Câu hỏi: "BN UỐNG/TIÊM chất này?" → có → THUỐC.
- PROCEDURE: hành động y khoa (`phẫu thuật X`, `nội soi`, `chọc dò`, `đặt stent`). Câu hỏi: "BS LÀM GÌ?" → Làm → TÊN_XÉT_NGHIỆM.
- VD: `phẫu thuật TURP` = procedure, `liệu pháp lợi tiểu` = treatment modality, không phải tên thuốc.

**C. TÁCH KẾT QUẢ IMAGING DÀI**: Pattern `<test-name> cho thấy <finding 1>, <finding 2>...` → MỖI finding = 1 entity riêng.

**D. TÁCH TÊN TEST + FINDING**: `<test-name> <finding-cùng-dòng>` → TÁCH.

**E. POSITION OVERLAP**: Mỗi occurrence của duplicate = 1 entity riêng với position duy nhất.

**F. ABNORMAL FINDINGS**: "X to", "giãn X", "tràn dịch X", "gãy X" → abnormal → CHẨN_ĐOÁN (không phải TRIỆU_CHỨNG/KQ_XN).
</clinical_judgment>

<noise_filter>
## 2. BỘ LỌC NOISE — DỰA TRÊN 5 CÂU HỎI VÀNG (SEMANTIC REASONING)

**DEFAULT = TRÍCH XUẤT.** Chỉ loại khi câu hỏi vàng cho kết quả rõ ràng là không phải entity y khoa.

🔥 **ĐỊNH DẠNG OUTPUT**: Chỉ trả về JSON array `[{...}, {...}]`. KHÔNG thêm text trước/sau JSON. Nếu không có entity → `[]`.

TRƯỚC KHI LOẠI bất kỳ cụm từ nào, tự hỏi theo thứ tự:

**[Q1] BN MẮC BỆNH, CẢM NHẬN, hay đây là KẾT QUẢ ĐO?**
→ Mắc bệnh = CHẨN_ĐOÁN | Cảm nhận = TRIỆU_CHỨNG | Giá trị đo = KẾT_QUẢ_XÉT_NGHIỆM
→ Không thuộc nhóm nào (thời gian, lối sống, chatbot) → DROP

Ví dụ áp dụng Q1:
- `"trong tuần qua"` → thời gian → DROP
- `"rượu bia"`, `"thuốc lá"` → lối sống → DROP
- `"Tiền sử:"`, `"Chẩn đoán:"` → header label → DROP
- `"mệt mỏi"` → BN cảm nhận → TRIỆU_CHỨNG ✓
- `"viêm phổi"` → BN mắc bệnh → CHẨN_ĐOÁN ✓

**[Q4] Đây là CAUSE hay EFFECT?**
→ Trigger/tác nhân gây bệnh (`đậu tằm`, `băng phiến`, `long não`) → DROP
→ Biểu hiện bệnh (`vàng da`, `sốt cao`) → extract

✅ **CẤM SPAN OVERLAP (R39)**: Mỗi cụm từ thuộc TỐI ĐA 1 entity. Span ngắn nằm trọn trong span dài → DROP span ngắn.
- SAI: `"phình giãn động mạch vành"` (CHẨN_ĐOÁN) + `"động mạch vành"` (TÊN_XN) → DROP span ngắn
- ĐÚNG: chỉ giữ `"phình giãn động mạch vành"` (CHẨN_ĐOÁN)

✅ **CẤM TÁCH COMPOUND DISEASE**: `"viêm X"`, `"ung thư X"`, `"suy X"` → 1 entity duy nhất, KHÔNG tách

✅ **CẤM ANATOMICAL-TERM-AS-TEST**: `"động mạch vành"`, `"tĩnh mạch cửa"` đứng một mình → KHÔNG phải TÊN_XÉT_NGHIỆM

✅ **DRUG-CLASS → DROP**: `"kháng sinh"`, `"NSAID"`, `"corticoid"`, `"thuốc hạ sốt"` (nhóm thuốc, không tên cụ thể) → DROP

✅ **STANDALONE DOSE → DROP**: `"30 mg"`, `"500 mg"` không có tên thuốc → DROP (fragment)

✅ **BODY PART ALONE → KHÔNG TRIỆU_CHỨNG**: `"ngực"`, `"bụng"`, `"đầu"` đứng một mình → DROP (trừ khi kèm tính từ: `"đau ngực"`, `"đau bụng"`)

✅ **CELL/PROCESS ALONE → DROP khi mô tả cơ chế**: `"hồng cầu"` / `"oxy hóa"` / `"tan huyết"` đứng riêng trong câu giải thích cơ chế → DROP. NGOẠI LỆ: `"thiếu máu tan huyết"` là CHẨN_ĐOÁN hợp lệ.

✅ **TRIGGER vs BIỂU HIỆN**: Dùng Q4 — `"đậu tằm"` (trigger gây tan huyết) → DROP. `"vàng da"` (biểu hiện tan huyết) → TRIỆU_CHỨNG ✓

NOTE: Bệnh án G6PD/Kawasaki/viêm tim → BẮT BUỘC extract đầy đủ. KHÔNG trả `[]` khi có thông tin y khoa.
</noise_filter>
<classification_principles>
## 3C. R38 (2026-07-22) - 6 NGUYÊN LÝ PHÂN LOẠI ĐÚNG (LUÔN ĐỌC TRƯỚC KHI GÁN TYPE)

⚠️ Khi gán type cho 1 entity, KHÔNG dựa vào keyword matching cứng. Hãy tự hỏi 6 câu hỏi sau theo thứ tự ưu tiên:

**NGUYÊN LÝ 1: "Đây là CÁI GÌ trong bệnh sử?"**
- **TÊN BỆNH có mã ICD** (vd `viêm phổi`, `THA`, `thiếu men G6PD`, `nhiễm khuẩn`, `nhiễm virus`) → **CHẨN_ĐOÁN**
- **Triệu chứng cơ năng/thực thể** (vd `đau ngực`, `khó thở`, `sốt cao`, `vàng da`, `Sốt cao`, `Tim đập nhanh`) → **TRIỆU_CHỨNG**
- **Thuốc cụ thể (generic/brand)** (vd `paracetamol`, `aspirin`, `Vitamin K`) → **THUỐC**
- **Tên chỉ định CLS** (vd `xét nghiệm G6PD`, `xét nghiệm máu`, `các xét nghiệm chuyên sâu`, `xét nghiệm sàng lọc`) → **TÊN_XÉT_NGHIỆM**

**NGUYÊN LÝ 2: "Câu trên đang nói GÌ về thứ này?"**
- Nếu câu nói về **nguyên nhân cơ chế** ("vì... nên...", "do... mà...", "Khi thiếu X, Y bị Z") → phần giải thích CƠ CHẾ không phải entity, chỉ phần KẾT QUẢ là triệu chứng/bệnh.
  - VD: `"Khi thiếu men này, hồng cầu trở nên mong manh và dễ bị phá hủy"` → KHÔNG extract `hồng cầu`, `oxy hóa`, `phá hủy`. Nếu có triệu chứng thật (`vàng da`, `mệt mỏi`) → extract.
- Nếu câu nói về **triệu chứng biểu hiện** ("BN có X", "xuất hiện X", "than phiền X") → X là TRIỆU_CHỨNG.
- Nếu câu nói về **chỉ định cận lâm sàng** ("làm X", "thực hiện X", "chỉ định X", "được làm X") → X là TÊN_XÉT_NGHIỆM (sau khi strip verb).

**NGUYÊN LÝ 3: "Cái này là NGUYÊN NHÂN hay HẬU QUẢ?"**
- NGUYÊN NHÂN / TRIGGER (tác nhân gây bệnh: `đậu tằm`, `băng phiến`, `long não`, `hóa chất`) → DROP, không phải entity.
- HẬU QUẢ / TRIỆU CHỨNG (`vàng da`, `khó thở`, `thiếu máu tan huyết`, `sốt cao`) → extract.

**NGUYÊN LÝ 4: "Bệnh nhân CẢM NHẬN được cái này không?"**
- BN CẢM NHẬN được (đau, sốt, khó thở, mệt, buồn nôn, vàng da, ...) → TRIỆU_CHỨNG.
- BN KHÔNG cảm nhận được (hồng cầu, oxy hóa, đông máu, chuyển hóa, ...) → DROP.
- Ngoại lệ: `đau X`, `nhức X` → TRIỆU_CHỨNG (BN cảm nhận được cơn đau ở vị trí X).

**NGUYÊN LÝ 5: "Cái này có TÊN RIÊNG trong ICD-10 / RxNorm không?"**
- Có tên trong ICD-10 (vd `viêm phổi`, `nhiễm khuẩn`, `thiếu máu tan huyết`) → CHẨN_ĐOÁN.
- Có tên trong RxNorm (vd `paracetamol`, `Vitamin K`, `aspirin`) → THUỐC.
- Là chỉ định CLS (vd `xét nghiệm`, `điện tâm đồ`) → TÊN_XÉT_NGHIỆM.
- KHÔNG có tên (vd `ăn uống`, `sinh hoạt`, `tiếp xúc`) → DROP.

**NGUYÊN LÝ 6: "Cái này BẢN THÂN NÓ là 1 test, hay chỉ là MÔ TẢ cách làm test?"**
- Là 1 test cụ thể (`xét nghiệm G6PD`, `công thức máu`, `điện tâm đồ`) → TÊN_XÉT_NGHIỆM.
- Chỉ là mô tả cách làm (`lấy máu ở gót chân`, `phân tích bằng máy`) → DROP.

🔥 **QUY TẮC VÀNG**: Khi nghi ngờ giữa TRIỆU_CHỨNG và CHẨN_ĐOÁN, hỏi: **"Đây là BỆNH/TỔN THƯƠNG (có ICD code) hay là BIỂU HIỆN BN kể?"**
- Bệnh/tổn thương → CHẨN_ĐOÁN.
- Biểu hiện BN kể → TRIỆU_CHỨNG.

🔥 **ÁP DỤNG NGUYÊN LÝ VÀO CÁC TRƯỜNG HỢP HAY SAI**:
- `"nhiễm khuẩn"`, `"nhiễm virus"`, `"nhiễm trùng"` → **CHẨN_ĐOÁN** (là bệnh có ICD code, dù đứng trong list "cần tránh").
- `"Vitamin K"`, `"Vitamin A"`, `"Vitamin B12"` → **THUỐC** (là supplement/drug, KHÔNG phải TÊN_XÉT_NGHIỆM).
- `"trẻ bị thiếu men G6PD"`, `"bệnh nhân bị viêm phổi"` → **CHẨN_ĐOÁN** (pattern `bị <disease>` → CHẨN_ĐOÁN, không phải TRIỆU_CHỨNG).
- `"ăn đậu tằm"`, `"tiếp xúc với băng phiến"` → **DROP** (hành động/trigger, không phải entity).
- `"hồng cầu"`, `"oxy hóa"`, `"phá hủy"`, `"chuyển hóa"` → **DROP** (tế bào/quá trình, BN không cảm nhận được).
- `"thực phẩm"`, `"hóa chất"`, `"thuốc"` (đứng riêng) → **DROP** (generic category).
</classification_principles>

<missing_entity_recovery>
## 3B. R37 (2026-07-14) - KHÔI PHỤC ENTITIES THƯỜNG GẶP (compact list)

⚠️ Một số entity phổ biến hay bị LLM miss do fatigue/context dài. BẮT BUỘC check & extract nếu có trong text:

**THUỐC TIM MẠCH** (hay miss): `aspirin`, `metoprolol`, `atenolol`, `bisoprolol`, `amlodipine`, `nifedipine`, `atorvastatin`, `simvastatin`, `rosuvastatin`, `warfarin`, `heparin`, `enoxaparin`, `rivaroxaban`, `apixaban`, `amiodarone`, `sotalol`, `digoxin`, `furosemide`, `spironolactone`, `insulin`, `metformin`, `glipizide`, `sitagliptin`

**KHÁNG SINH** (hay miss): `vancomycin`, `meropenem`, `ceftriaxone`, `cefepim` (Cefepime), `cefotaxime`, `amoxicillin`, `augmentin`, `biseptol`, `azithromycin`, `ciprofloxacin`, `levofloxacin`, `metronidazole`, `fluconazole`, `acyclovir`

**GIẢM ĐAU/HẠ SỐT**: `paracetamol`, `acetaminophen`, `morphine`, `fentanyl`, `tramadol`, `ibuprofen`

**CHẨN_ĐOÁN TIM MẠCH**: `THA` (tăng huyết áp), `ĐTĐ` (đái tháo đường), `NMCT` (nhồi máu cơ tim), `suy tim`, `rung nhĩ`, `block nhĩ thất`, `ngoại tâm thu nhĩ/thất`, `viêm cơ tim`, `viêm màng ngoài tim`, `viêm nội tâm mạc`, `viêm màng tim`, `RLLL` (rối loạn lipid máu), `tăng cholesterol`, `thoái hóa khớp`

**TRIỆU CHỨNG TIM MẠCH**: `đau ngực`, `đau ngực trái/phải`, `đau thắt ngực`, `khó thở`/`khó thở nhẹ`/`khó thở khi gắng sức`, `đánh trống ngực`, `hồi hộp`, `tim đập nhanh`, `ngất`, `choáng`, `hoa mắt`

**F. NGỮ CẢNH GIA ĐÌNH (FAMILY CONTEXT - R37 ASSERTION)**:
- `tiền sử gia đình X` → X là CHẨN_ĐOÁN/TRIỆU_CHỨNG + assertions=`["isFamily"]`
- `bố/mẹ/anh/chị/em ruột bị/đã từng/mắc X` → X + assertions=`["isFamily"]`
- `gia đình có người X` → X + assertions=`["isFamily"]`
- Pattern matching: `(bố|cha|mẹ|anh|chị|em|con|ông|bà|cô|dì|chú|bác)\\s+(?:đã\\s+)?(?:bị|mắc|có|từng|mất\\s+vì)\\s+([A-ZÀ-Ỹ][\\w\\s]+)`
- Ngược lại: `gia đình KHÔNG ai bị X` → X + assertions=`["isFamily", "isNegated"]`

**G. CHIA LIỀU THUỐC — BẮT BUỘC GIỮ NGUYÊN ĐUÔI `x N` (R37 KEEP)**:
- `aspirin 81mg x 1`, `metoprolol 25mg x 2`, `paracetamol 500mg x 3`
- KHÔNG BAO GIỜ được drop phần `x N` ở cuối — đây là dose count quan trọng cho prescription.
</missing_entity_recovery>

<splitting_and_context>
## 3. QUY TẮC TÁCH THỰC THỂ & PHÁN ĐOÁN NGỮ CẢNH (ASSERTIONS)

1. **Tách cụm `A [CONNECTOR] B` (Drug + Disease / Symptom Split)**:
   - Khi gặp cấu trúc `[Thuốc] cho [Bệnh/Triệu chứng]` (`doxycycline cho viêm tuyến mồ hôi`, `paracetamol cho sốt`, `aspirin trị đau đầu`, `metformin điều trị đái tháo đường`), PHẢI TÁCH thành 2 entities riêng biệt: THUỐC (`doxycycline`) + CHẨN_ĐOÁN/TRIỆU_CHỨNG (`viêm tuyến mồ hôi`). KHÔNG gộp làm một!

2. **Tách cụm `TÊN_XÉT_NGHIỆM + VALUE`**:
   - `WBC 14,5 K/uL` → TÁCH 2: TÊN_XÉT_NGHIỆM (`WBC`) + KẾT_QUẢ_XÉT_NGHIỆM (`14,5 K/uL`).
   - `HA 160/90 mmHg` → TÁCH 2: TÊN_XÉT_NGHIỆM (`HA`) + KẾT_QUẢ_XÉT_NGHIỆM (`160/90 mmHg`).

3. **Chuỗi phủ định — NEGATION CHAINING (R37 sửa 2026-07-16)**:
   - Bệnh án VN thường liệt kê nhiều triệu chứng VẮNG MẶT sau "không" đầu câu.
     Toàn bộ list trong cùng chuỗi "không ..." đều bị `isNegated` (CHAIN tiếp tục qua dấu `,` và `hay`/`và`/`cũng`).
   - `Không sốt, không ho, có đau đầu` → `sốt`=`isNegated`, `ho`=`isNegated`, **`đau đầu`=`[]`** (vì "có" BREAK chain trước "đau đầu")
   - `Không buồn nôn, hay nôn, đổ mồ hôi` → `buồn nôn`=`isNegated`, `nôn`=`isNegated`, **`đổ mồ hôi`=`isNegated`** (chain "không ... hay ... Z" → cả 3 đều negate, KHÔNG có "có" phá chain)
   - `Không buồn nôn, không nôn, không đổ mồ hôi` → 3 TRIỆU_CHỨNG, **đều** `isNegated` (chuỗi "không" liên tiếp)
   - `Không sốt, không ho` → cả 2 đều isNegated (chuỗi "không" liên tiếp)
   - ⚠️ **NGUYÊN TẮC VÀNG**:
     - "không X[, hay/và/cũng]* Y[, Z, W, ...]" → **TẤT CẢ** đều isNegated cho đến khi gặp `"có"`/`"nhưng"`/`"mà"` (BREAK chain)
     - "có"/"nhưng" + symptom mới = NEW assertion (=`[]`, không negate)
   - **Ví dụ PHÂN BIỆT:**
     - `không sốt, không ho, đau đầu` → sốt=`isNegated`, ho=`isNegated`, **đau đầu=`isNegated`** (chain tiếp qua `,`)
     - `không sốt, không ho, có đau đầu` → sốt=`isNegated`, ho=`isNegated`, đau đầu=`[]` (vì "có" break chain)
     - `không sốt, không ho, nhưng đau ngực` → sốt=`isNegated`, ho=`isNegated`, đau ngực=`[]` ("nhưng" break chain)
     - `không có khó thở, có thắt chặt ngực` → khó thở=`isNegated`, thắt chặt ngực=`[]`
     - `Không buồn nôn, hay nôn, đổ mồ hôi` → buồn nôn=`isNegated`, nôn=`isNegated`, **`đổ mồ hôi`=`isNegated`** (chain "không" qua "hay")
4. **3 Assertions chuẩn (max 3, có thể kết hợp)**:
   - `isHistorical`: Tiền sử bệnh xa (`Tiền sử: THA 5 năm`), hoặc thuốc đang dùng TRƯỚC nhập viện (`Thuốc đang dùng: amlodipine`, `Thuốc trước khi nhập viện:`). *(Lưu ý: "Lý do nhập viện", "Triệu chứng hiện tại", "Khám lâm sàng" là đợt bệnh hiện tại → `assertions: []`, KHÔNG phải isHistorical)*.
   - `isNegated`: Bệnh/triệu chứng bị phủ định bởi từ `không`/`chưa`/`không có` ở đầu HOẶC trong chuỗi chain "không X[, hay/và/cũng]* Y[, Z, ...]" (xem mục 3 ở trên). Chain tiếp tục qua dấu `,` cho đến khi gặp `"có"`/`"nhưng"`/`"mà"` (break).
   - `isFamily`: CHỈ khi người thân là CHỦ THỂ mắc bệnh (`Bố bệnh nhân bị THA` → `["isFamily", "isHistorical"]`). KHÔNG gán khi "bác sĩ" (nhân viên y tế) hoặc gia đình chỉ kể/quan sát hộ bệnh nhân (`"bác sĩ chăm sóc chính kê đơn..."`, `"Gia đình nhận thấy bệnh nhân..."` → đây là bệnh/thuốc của bệnh nhân, KHÔNG `isFamily`).

## 3A. ASSERTIONS — QUY TẮC CHI TIẾT & EDGE CASES (R37 cập nhật 2026-07-16)

**A. `isHistorical` — bệnh/thuốc của BỆNH NHÂN trong quá khứ**:

Gán `isHistorical` cho entity khi thuộc 1 trong các ngữ cảnh:
- **Section header tiền sử**: trong phần `Tiền sử`, `Tiền căn`, `Bệnh sử`, `Tiền sử bệnh nội/ngoại khoa`, `Tiền sử phẫu thuật`.
- **Thuốc trước nhập viện**: `Thuốc trước khi nhập viện:`, `Thuốc đang dùng:`, `Đang điều trị tại nhà:` → TẤT CẢ drugs trong đó → `isHistorical`.
- **Câu có temporal marker quá khứ**: `cách đây X năm`, `năm 2018`, `đã từng`, `trước đây`, `từng có`, `tiền sử X`, `cũ X`. VD: `"tiền sử THA 10 năm"`, `"đã từng NMCT 2018"`, `"phẫu thuật ruột thừa năm 2015"`.

KHÔNG gán `isHistorical`:
- Trong section **hiện tại**: `Lý do nhập viện:`, `Triệu chứng hiện tại:`, `Khám lâm sàng:`, `Cận lâm sàng:`, `Chẩn đoán xác định:`, `Điều trị:`, `Thuốc ra viện:` → `assertions: []`.
- Thuốc kê MỚI trong đợt điều trị hiện tại → KHÔNG `isHistorical`.

**B. `isFamily` — bệnh của NGƯỜI THÂN bệnh nhân**:

Gán `isFamily` (THƯỜNG KẾT HỢP với `isHistorical`) khi CHỦ THỂ mắc bệnh là người thân:
- **Section "Tiền sử gia đình"**: MỌI entity trong đó → `["isFamily"]` (không cần `isHistorical` vì bệnh không thuộc BN).
- **Pattern bố/mẹ/anh/chị/em mắc bệnh**: `(bố|cha|mẹ|anh|chị|em|con|ông|bà|cô|dì|chú|bác) (đã |từng |được chẩn đoán |bị |mắc |có )? [bệnh]` → bệnh = `["isFamily", "isHistorical"]`.
  - VD: `"Bố bệnh nhân bị THA"` → THA=`["isFamily", "isHistorical"]`
  - VD: `"Mẹ từng NMCT năm 2010"` → NMCT=`["isFamily", "isHistorical"]`
  - VD: `"Anh trai đái tháo đường type 2"` → ĐTĐ type 2=`["isFamily"]`
- **Pattern "gia đình có/có ai đó"**: `"gia đình có ai bị hen"` → hen=`["isFamily"]`.
- **Negative family**: `"gia đình KHÔNG ai bị X"` → X=`["isFamily", "isNegated"]`.

KHÔNG gán `isFamily`:
- Khi chủ thể là **bác sĩ/nhân viên y tế**: `"bác sĩ phụ trách chính kê đơn"`, `"điều dưỡng chăm sóc"` → đây là hành động chăm sóc, KHÔNG phải bệnh người thân.
- Khi **người nhà chỉ kể quan sát**: `"Gia đình nhận thấy bệnh nhân khó thở"`, `"Người nhà kể bệnh nhân đau ngực"` → đây là bệnh CỦA BỆNH NHÂN, người nhà chỉ là witness. Bệnh → `assertions: []` (KHÔNG `isFamily`).
- Khi bệnh được đề cập chung chung không có chủ thể rõ ràng.

**C. `isNegated` — bệnh/triệu chứng bị PHỦ ĐỊNH**:

Gán `isNegated` khi:
- **Từ phủ định trực tiếp trước entity** (window 5-15 từ): `không X`, `chưa X`, `chưa có X`, `không có X`, `không thấy X`, `không ghi nhận X`, `chưa phát hiện X`, `âm tính`, `loại trừ X`.
- **Negation chain** (xem section 3): `không X, hay/và/cũng Y, Z` → cả list trong chain đều `isNegated` cho đến khi gặp `có`/`nhưng`/`mà` (BREAK chain).
- **Câu phủ định mệnh đề**: `"bệnh nhân KHÔNG có tiền sử THA"` → THA=`isNegated` (KHÔNG `isHistorical` vì là PHỦ ĐỊNH tiền sử).

KHÔNG gán `isNegated`:
- **TÊN_XÉT_NGHIỆM** trong câu kết quả bình thường: `"x-quang ngực không ghi nhận bất thường"` → TÊN_XN (`x-quang ngực`) có `assertions: []`, KQ_XN (`không ghi nhận bất thường`) MỚI là `isNegated` (nếu có).
- **Bình thường / không có gì đáng chú ý**: đây là KẾT QUẢ bình thường (positive normal finding), KHÔNG `isNegated`. VD: `"X-quang bình thường"` → KQ_XN=`bình thường`, assertions=`[]`.
- **Dương tính**: có bệnh → `assertions: []`. **Âm tính**: không có bệnh → `isNegated`.

**D. KẾT HỢP ASSERTIONS** (max 3 assertions mỗi entity):
- `isHistorical` + `isFamily`: `"Bố bệnh nhân bị THA từ 2010"` → `["isFamily", "isHistorical"]`.
- `isFamily` + `isNegated`: `"Gia đình không ai bị hen"` → `["isFamily", "isNegated"]`.
- `isHistorical` + `isNegated`: `"Bệnh nhân không có tiền sử đái tháo đường"` → ĐTĐ=`["isHistorical", "isNegated"]`.

**E. BẢNG TRA NHANH ASSERTION**:

| Câu trong input | Entity type | Assertions |
|---|---|---|
| `Tiền sử: THA 10 năm` | THA (CHẨN_ĐOÁN) | `["isHistorical"]` |
| `Thuốc đang dùng: amlodipine 5mg` | amlodipine (THUỐC) | `["isHistorical"]` |
| `Bố bệnh nhân bị đái tháo đường type 2` | ĐTĐ type 2 (CHẨN_ĐOÁN) | `["isFamily", "isHistorical"]` |
| `Gia đình không ai bị hen` | hen (CHẨN_ĐOÁN) | `["isFamily", "isNegated"]` |
| `Lý do nhập viện: đau ngực` | đau ngực (TRIỆU_CHỨNG) | `[]` |
| `Chẩn đoán: nhồi máu cơ tim cấp` | NMCT cấp (CHẨN_ĐOÁN) | `[]` |
| `Không buồn nôn, hay nôn, đổ mồ hôi` | buồn nôn/nôn/đổ mồ hôi | `["isNegated"]` |
| `X-quang ngực không ghi nhận bất thường` | X-quang ngực (TÊN_XN) | `[]` |
| `X-quang ngực không ghi nhận bất thường` | không ghi nhận bất thường (KQ_XN) | `["isNegated"]` |
| `AST bình thường` | bình thường (KQ_XN) | `[]` (positive normal) |
| `Cấy máu âm tính` | cấy máu (TÊN_XN) | `[]` |
| `Cấy máu âm tính` | âm tính (KQ_XN) | `["isNegated"]` |

**F. POSITION-BASED HEURISTIC** (khi không có cue word rõ ràng):
- Nếu entity nằm trong section có header `"Tiền sử"`, `"Tiền căn"`, `"Tiền sử bệnh"` (không kèm "hiện tại") → `isHistorical`.
- Nếu entity nằm trong section `"Tiền sử gia đình"`, `"Tiền sử xã hội"` → `isFamily` (gộp `isHistorical` nếu bệnh có thời gian cụ thể).
- Nếu entity nằm trong section `"Lý do nhập viện"`, `"Triệu chứng cơ năng"`, `"Khám lâm sàng"`, `"Cận lâm sàng"`, `"Chẩn đoán xác định"`, `"Điều trị"`, `"Thuốc ra viện"` → `assertions: []`.
- Nếu không rõ section → default `[]` (KHÔNG default `isHistorical` để tránh over-assign).
</splitting_and_context>

<duplicate_and_position>
## 4. QUY TẮC BẢO TOÀN SỐ LƯỢNG & VỊ TRÍ (POSITION & DUPLICATES - QUAN TRỌNG NHẤT)

🔴 **NGUYÊN TẮC VÀNG VỀ POSITION & DUPLICATES (R10 STRICT)**:
1. **Mỗi lần xuất hiện ở vị trí khác nhau = 1 Entity riêng biệt**:
   - Trong hồ sơ lâm sàng, nếu một bệnh lý hoặc triệu chứng (`đánh trống ngực`, `khó thở`, `đau ngực`, `tăng huyết áp`) xuất hiện **N lần tại N vị trí (`position`) khác nhau** (ví dụ 1 lần ở Lý do nhập viện, 1 lần ở Tiền sử, 2 lần ở Khám hiện tại) → **PHẢI TRÍCH XUẤT ĐỦ N ENTITIES RIÊNG BIỆT với N cặp `[start, end)` khác nhau** (Python convention: end là EXCLUSIVE).
   - **Positions CÓ THỂ overlap** (vd "ung thư" [10,16] + "ung thư phổi" [10,22] đều hợp lệ — mỗi occurrence là 1 entity riêng biệt với position riêng).
   - Tuyệt đối KHÔNG gộp hoặc bỏ qua các lần lặp lại ở các đoạn văn khác nhau của `CHẨN_ĐOÁN`, `TRIỆU_CHỨNG`, `THUỐC`, `KẾT_QUẢ_XÉT_NGHIỆM`.
2. **Ngoại lệ duy nhất - `TÊN_XÉT_NGHIỆM` (R22 Dedup)**:
   - Với chỉ riêng category `TÊN_XÉT_NGHIỆM` (`chụp x-quang ngực`, `phân tích nước tiểu`, `ECG`, `monitor holter`), nếu xuất hiện nhiều lần trong cùng một bệnh án → **CHỈ trích xuất 1 entity duy nhất** (tại vị trí xuất hiện đầu tiên).
3. **Độ chính xác của `position: [start, end)`**:
   - `start` và `end` là character offset (0-indexed) của chuỗi exact match trong input gốc. **end là EXCLUSIVE** (Python slicing convention: `input_text[start:end]` = span text). Cố gắng tìm exact match chính xác nhất.
   - **Positions CÓ THỂ overlap** (vd "ung thư" ở [10,16] và "ung thư phổi" ở [10,22] đều hợp lệ — mỗi occurrence là 1 entity riêng biệt).
</duplicate_and_position>

<output_format>
## 5. QUY TẮC ĐỊNH DẠNG ĐẦU RA (OUTPUT FORMAT)

- Trả về CHÍNH XÁC một mảng JSON (JSON array) với 4 fields: `[{"text": "...", "type": "...", "position": [start, end), "assertions": [...]}]`
- `position` là `[start, end)` Python convention (end exclusive, vd `input_text[start:end]`). Positions CÓ THỂ overlap khi duplicate xuất hiện ở cùng vị trí.
- KHÔNG dùng markdown fence (KHÔNG gõ ```json hay ```).
- KHÔNG giải thích trước hay sau JSON.
- Nếu không có thực thể y khoa nào, trả về mảng rỗng `[]`.
- `candidates: []` luôn là list rỗng (hệ thống tự động map ICD/RxNorm phía sau).
</output_format>

<recall_and_precision>
## 11. ⚠️ CHECKLIST TUẦN TỰ THEO SECTION (RECALL TỐI ĐA)

Trước khi output JSON, scan TUẦN TỰ qua từng phần bệnh án. Với mỗi PHẦN, hỏi câu hỏi tương ứng:

| # | Phần bệnh án | Câu hỏi cần đặt | Loại extract |
|---|---|---|---|
| 1 | Lý do nhập viện | Triệu chứng chính / Chẩn đoán sơ bộ là gì? | TRIỆU_CHỨNG + CHẨN_ĐOÁN, `assertions=[]` |
| 2 | Tiền sử / Bệnh sử / Tiền căn | Bệnh nền nào? Thuốc đang dùng nào? | CHẨN_ĐOÁN + THUỐC, **isHistorical** |
| 3 | Diễn biến / Quá trình | Triệu chứng xuất hiện / leo thang / cải thiện? | TRIỆU_CHỨNG + assertions theo ngữ cảnh |
| 4 | Khám lâm sàng | Sinh hiệu (HA, Mạch, SpO2, Nhiệt độ)? Findings bất thường (tim to, ran, phù)? | TÊN_XN + KQ_XN (cho VS) + CHẨN_ĐOÁN (abnormal) |
| 5 | Cận lâm sàng / Xét nghiệm | Test nào? Kết quả định lượng? Bất thường nào? | TÊN_XN + KQ_XN + CHẨN_ĐOÁN |
| 6 | Chẩn đoán xác định / Chẩn đoán ra viện | Danh sách chẩn đoán chính? | CHẨN_ĐOÁN, `assertions=[]` |
| 7 | Điều trị / Thuốc ra viện / Chỉ định | Thuốc nào? Liều & tần suất? | THUỐC (verbatim `x N`) + TÊN_XN (nếu có chỉ định CLS) |
| 8 | Theo dõi / Tái khám / Dặn dò | Triệu chứng còn / tái phát / hết? | TRIỆU_CHỨNG + isNegated nếu đã hết |

⚠️ Với mỗi PHẦN, scan **2 LẦN**:
- **Pass 1 (Recall)**: tìm TẤT CẢ entities y khoa có vẻ khả thi. ĐỪNG sợ extract thừa — sẽ lọc ở Pass 2.
- **Pass 2 (Precision)**: verify mỗi entity (xem mục 14 bên dưới).

## 12. ❌ BẢNG FALSE POSITIVE PHẢI TRÁNH (PRECISION TỐI ĐA)

⛔ KHÔNG BAO GIỜ trích xuất các cụm dưới đây (đã được verify là noise qua 100 file audit thực tế):

| Input cụ thể | ❌ KHÔNG extract làm | Lý do |
|---|---|---|
| `HA 160/90 mmHg` | `"HA"` riêng (nếu đã tách TÊN+KQ) | Tránh duplicate "HA" |
| `bệnh nhân vào viện vì khó thở` | `"vào viện"`, `"vì khó thở"` | Drop verb dẫn narrative |
| `Tiền sử THA 10 năm` | `"10 năm"` (CHẨN_ĐOÁN) | Drop pure duration |
| `Tiền sử THA 10 năm` | `"Tiền sử"` (bất kỳ) | Drop section label |
| `Đã dùng aspirin 81mg` | `"Đã dùng"` | Drop narrative verb |
| `prednisone 40 mg/ngày, sau đó 30 mg` | `"30 mg"` (THUỐC) | Drop dose fragment không có tên thuốc |
| `Xuất huyết nội sọ không đặc hiệu` | `"không đặc hiệu"` (KQ_XN) | Drop standalone qualifier |
| `Bệnh nhân hút thuốc lá` | `"thuốc lá"` (THUỐC) | Drop lifestyle, không phải thuốc điều trị |
| `Được kê aspirin, atorvastatin và siêu âm tim` | `"Được kê"`, `"và siêu âm tim"` (TÊN_XN) | Tránh duplicate "siêu âm tim" |
| `Có thể là viêm phổi` | `"có thể là"` | Drop hedge verb |
| `Trong tuần qua`, `cách 3 ngày`, `kéo dài 20 giây` | (mọi type) | Drop pure time marker |
| `Tiếp tục cảm thấy đánh trống ngực` | `"Tiếp tục"`, `"cảm thấy"` | Drop narrative verbs |
| `Mất ngủ`, `mất việc`, `nghỉ việc` | `"mất việc"` | Drop social context |
| `Căng thẳng`, `stress`, `lo lắng` | (bất kỳ) | Drop psychological lifestyle |
| `Ăn uống bình thường`, `sinh hoạt điều độ` | (bất kỳ) | Drop lifestyle summary |
| `Lúc 17 giờ`, `vào lúc 8h sáng` | (bất kỳ) | Drop time-of-day standalone |
| `đã/đang/sẽ được chuyển/khuyên/chỉ định` | `"đã được chuyển"` | Drop administrative narrative |

## 13. 🌳 DECISION TREE — KHI PHÂN VÂN TYPE

Khi không chắc chắn 1 entity thuộc type nào, dùng cây quyết định sau (theo thứ tự ưu tiên):

```
[Text trong input]
    │
    ├─ Chỉ chứa digit + đơn vị (mg/ml/g/mcg/iu/mmHg/°C/%)?
    │   └─ YES → KẾT_QUẢ_XÉT_NGHIỆM
    │       └─ Trước đó có viết tắt test (AST, WBC, INR)? → tách thêm TÊN_XN riêng
    │
    ├─ Là viết tắt chỉ định test (AST, ALT, WBC, ECG, X-quang, MRI)?
    │   └─ YES → TÊN_XÉT_NGHIỆM
    │
    ├─ Có tên bệnh ICD (viêm X, suy X, ung thư X, NMCT, THA, ĐTĐ, hen)?
    │   └─ YES → CHẨN_ĐOÁN
    │
    ├─ Có tên bất thường trên CLS (gãy xương, ST chênh, block, ngoại tâm thu, tràn dịch)?
    │   └─ YES → CHẨN_ĐOÁN (KHÔNG phải KQ_XN)
    │
    ├─ Có tên thuốc + liều (aspirin 325mg, metoprolol 25mg)?
    │   └─ YES → THUỐC
    │
    ├─ Là brand name thuốc (Crestor, Toradol, Augmentin)?
    │   └─ YES → THUỐC (KHÔNG phải TÊN_XN)
    │
    ├─ Là tên máy/thiết bị y tế (BiPAP, CPAP, monitor, máy thở)?
    │   └─ YES → TÊN_XÉT_NGHIỆM
    │
    ├─ Là từ triệu chứng chủ quan (đau, khó thở, sốt, ho, nôn, chóng mặt)?
    │   └─ YES → TRIỆU_CHỨNG
    │
    ├─ Là "bình thường", "không ghi nhận bất thường", "nhịp xoang đều"?
    │   └─ YES → KẾT_QUẢ_XÉT_NGHIỆM (normal finding)
    │
    └─ CÒN LẠI (uncertain):
        ├─ Có trong ICD/từ điển y khoa? → drop là an toàn nhất
        └─ Nếu là narrative/lifestyle/social/time → drop
```

## 14. ⚠️ TWO-PASS VERIFICATION (BẮT BUỘC)

**Pass 1 — Recall (extract aggressively):**
- Scan TỪNG section bệnh án (mục 11) theo thứ tự.
- Extract MỌI entity có vẻ y khoa. ĐỪNG sợ extract thừa.
- Mỗi occurrence ở vị trí khác nhau = 1 entity riêng (R10 STRICT).

**Pass 2 — Precision (verify chặt):**
Với MỖI entity đã extract ở Pass 1, kiểm tra **TẤT CẢ** tiêu chí sau:

| # | Check | Nếu FAIL |
|---|---|---|
| 1 | Text khớp VERBATIM với input (case-sensitive, đầy đủ span)? | Fix text, hoặc DROP |
| 2 | Position `[start, end)` chính xác (word-boundary cả 2 phía)? | Re-find hoặc DROP |
| 3 | Type đúng theo Decision Tree (mục 13)? | RETYPE |
| 4 | Assertions đúng ngữ cảnh (isNegated/isHistorical/isFamily)? | Fix assertions |
| 5 | KHÔNG thuộc 10 CẤM rules (mục 2)? | DROP |
| 6 | KHÔNG nằm trong bảng False Positive (mục 12)? | DROP |
| 7 | KHÔNG phải fragment không có tên (drug name, body part đầy đủ)? | DROP |
| 8 | KHÔNG overlap trùng với entity đã giữ (vd "HA" riêng + "HA 160/90 mmHg" — giữ cặp TÊN+KQ, drop "HA" riêng)? | DROP duplicate |

⚠️ **Nếu entity FAIL ≥ 1 check** → DROP ngay tại Pass 2. KHÔNG để lọt sang output.

🎯 **MỤC TIÊU**: output cuối = MỌI entity y khoa cần thiết (RECALL tối đa) + KHÔNG noise (PRECISION tối đa).
</recall_and_precision>


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
_PRIORITY_EXAMPLES_PATH = _PROJECT_ROOT / "data" / "priority_examples.jsonl"

OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "required": ["text", "type", "assertions"],
        "properties": {
            "text": {"type": "string", "minLength": 1},
            "type": {
                "type": "string",
                # R39 (2026-07-24): Accept CẢ diacritics VN VÀ ASCII no-diacritics.
                # Local schema trước đây chỉ chấp nhận diacritics → reject các
                # entity normalized sang ASCII (THUOC, CHAN_DOAN, ...) gây invalid
                # schema 100% file. Nay chấp nhận cả 2 form để compatible với
                # grader dùng hoặc ASCII-only hoặc diacritics schema.
                "enum": [
                    # Diacritics (Vietnamese) — primary
                    "THUỐC",
                    "TRIỆU_CHỨNG",
                    "TÊN_XÉT_NGHIỆM",
                    "KẾT_QUẢ_XÉT_NGHIỆM",
                    "CHẨN_ĐOÁN",
                    # ASCII (no-diacritics) — fallback khi normalize
                    "THUOC",
                    "TRIEU_CHUNG",
                    "TEN_XET_NGHIEM",
                    "KET_QUA_XET_NGHIEM",
                    "CHAN_DOAN",
                ],
            },
            # R34 (2026-07-13): position BỎ khỏi schema. Python tự tính character offset
            # chính xác 100% trong align_and_expand_entities — không cần LLM đếm.
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

    R39 (2026-07-24): Nếu có priority_examples.jsonl, các priority examples
    (có field `_priority`) sẽ được load TRƯỚC, đảm bảo chúng xuất hiện đầu
    tiên trong few-shot context. Giúp LLM nhận các case khó trước case dễ.

    Args:
        path: đường dẫn JSONL (mặc định: ``data/examples.jsonl``).

    Returns:
        list of dict ``{"input": str, "output": list[dict]}``.
    """
    examples: list[dict] = []

    # R39: Load priority examples TRƯỚC (nếu tồn tại) để chúng xuất hiện đầu tiên.
    if path is None and _PRIORITY_EXAMPLES_PATH.exists():
        with _PRIORITY_EXAMPLES_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ex = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {_PRIORITY_EXAMPLES_PATH}: {line[:80]!r} ({exc})"
                    ) from exc
                inp = ex.get("input")
                out = ex.get("output")
                if not isinstance(inp, str) or not isinstance(out, list):
                    continue
                examples.append(ex)

    # Load main examples
    p = path or _EXAMPLES_PATH
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
    """Chọn k few-shot examples có độ tương đồng ngữ cảnh/chuyên khoa cao nhất với input_text.

    R39 (2026-07-24): Priority examples (có field `_priority` cao) LUÔN được
    chọn TRƯỚC. Các examples còn lại được xếp theo overlap tokens với input.
    """
    if not examples or k <= 0:
        return []
    if k >= len(examples):
        return examples

    def _get_tokens(t: str) -> set[str]:
        words = re.findall(r'[a-zà-ỹ0-9_/-]{3,}', t.lower())
        stop = {"của", "và", "có", "cho", "trong", "với", "được", "các", "những", "lúc", "tại", "vào", "ra", "bệnh", "nhân", "ngày", "lần", "tiền", "sử", "hiện", "tại", "không", "chưa", "khi"}
        return set(w for w in words if w not in stop)

    in_tokens = _get_tokens(input_text)

    scored = []
    for idx, ex in enumerate(examples):
        priority = ex.get("_priority", 0) or 0
        if in_tokens:
            ex_tokens = _get_tokens(ex.get("input", ""))
            overlap = len(in_tokens & ex_tokens)
            union = len(in_tokens | ex_tokens) or 1
            similarity = overlap / union
        else:
            similarity = 0
        score = priority * 1000 + similarity * 100
        scored.append((score, idx, ex, priority, similarity))

    scored.sort(key=lambda x: (-x[0], x[1]))
    selected = [ex for _, _, ex, _, _ in scored[:k]]
    return selected


def format_few_shot_messages(examples: list[dict]) -> list[dict[str, str]]:
    """Chuyển few-shot examples (Stage 1 / End-to-end) sang OpenAI chat messages."""
    msgs: list[dict[str, str]] = []
    for idx, ex in enumerate(examples):
        inp = ex.get("input", "")
        out = ex.get("output", [])
        user_content = build_stage1_user_prompt(inp)

        if ex.get("_priority", 0) > 0:
            priority_hint = (
                f"\n⚠️ DEMO HARD CASE (priority={ex.get('_priority', 0)}, "
                f"category={ex.get('_category', 'unknown')}): "
                "ĐÂY LÀ VÍ DỤ VỀ CASE KHÓ — Học kỹ để áp dụng cho input tương tự.\n\n"
            )
            user_content = priority_hint + user_content

        msgs.append({"role": "user", "content": user_content})
        msgs.append({"role": "assistant", "content": json.dumps(out, ensure_ascii=False)})
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
    """Build user prompt với input text."""
    try:
        from src.postprocess import _get_duplicate_alert
        alert = _get_duplicate_alert(input_text)
    except Exception:
        alert = ""

    alert_part = f"{alert}\n\n" if alert else ""

    return (
        "🎯 NHIỆM VỤ: Trích xuất TOÀN BỘ thực thể y khoa từ hồ sơ bệnh án tiếng Việt dưới đây vào 5 loại (THUỐC, CHẨN_ĐOÁN, TRIỆU_CHỨNG, TÊN_XÉT_NGHIỆM, KẾT_QUẢ_XÉT_NGHIỆM).\n\n"
        "🔥 QUY TẮC QUÉT (R10 STRICT — DUPLICATE HANDLING QUAN TRỌNG NHẤT):\n"
        "1. MỖI LẦN XUẤT HIỆN = 1 ENTITY RIÊNG: Nếu triệu chứng/thuốc (`đau ngực`, `aspirin`) xuất hiện 3-4 lần ở Tiền sử → Khám → CLS → Điều trị, BẮT BUỘC lấy đủ 3-4 entities.\n"
        "2. ĐỪNG gộp các vị trí khác nhau thành 1 entity. Mỗi position = 1 span riêng.\n"
        "3. Đảm bảo đủ 5 TYPE: THUỐC (giữ liều lượng `x 1`, `x 2`), CHẨN_ĐOÁN (kể cả viết tắt `THA`, `ĐTĐ`, `NMCT`, `COPD`, `RLLL`), TRIỆU_CHỨNG, TÊN_XN, KQ_XN (kể cả normal findings `nhịp xoang đều`).\n"
        "4. Cắt sạch động từ dẫn (`cảm thấy`, `chụp`) & thời lượng rác (`trong tuần qua`).\n"
        "5. KHÔNG CẦN ghi `position` — Python tự tính offset chính xác 100%.\n\n"
        "⚠️ CẤM: trích xuất thời gian độc lập (`3 ngày`, `20 giây`), lối sống (`rượu bia`), label header (`Tiền sử:`, `Chẩn đoán:`).\n\n"
        f"{alert_part}"
        f"INPUT:\n{input_text}\n\n"
        "OUTPUT JSON ARRAY (chỉ các trường text, type, assertions — không cần position, không kèm lời giải thích):"
    )


# ==============================================================================
# TWO-STAGE PIPELINE PROMPTS
# ==============================================================================

STAGE1_PROMPT = f"""Bạn là chuyên gia NER y khoa tiếng Việt với 20+ năm kinh nghiệm. Nhiệm vụ: trích xuất TẤT CẢ entities từ bệnh án vào 5 loại (THUỐC, CHẨN_ĐOÁN, TRIỆU_CHỨNG, TÊN_XÉT_NGHIỆM, KẾT_QUẢ_XÉT_NGHIỆM).

═══════════════════════════════════════════════════════════════
PHẦN 1 — TRIẾT LÝ CỐT LÕI (CORE PRINCIPLES)
═══════════════════════════════════════════════════════════════

🎯 4 NGUYÊN TẮC BẤT BIẾN:

(N1) **CHÍNH XÁC TEXT**: Mỗi entity text PHẢI là substring chính xác `input_text[start:end]`.
  - KHÔNG thêm space vào từ dính (vd "ảo giácxuất hiện" → giữ "ảo giácxuất hiện").
  - KHÔNG đổi case ("Buồn nôn" → giữ "Buồn nôn").
  - KHÔNG hallucinate entities không xuất hiện trong input.

(N2) **RECALL TỐI ĐA**: Mọi khái niệm y khoa trong input PHẢI được extract.
  - Bệnh án chi tiết > 2000 chars → CẦN ≥ 25 entities.
  - Bệnh án < 1000 chars → CẦN ≥ 8 entities.
  - KHÔNG bao giờ trả [] rỗng khi input có thông tin y khoa.

(N3) **TYPE CHÍNH XÁC**: dựa trên bản chất y khoa & 5 câu hỏi vàng (xem PHẦN 3).

(N4) **NO SPAN OVERLAP (R39)**: Mỗi cụm từ chỉ thuộc TỐI ĐA 1 entity.
  - Nếu đã extract "phình giãn động mạch vành" → KHÔNG extract "động mạch vành" riêng.
  - Khi 2 spans chồng lấn nhau → XOÁ span ngắn hơn.

═══════════════════════════════════════════════════════════════
PHẦN 2 — BẢNG CLASSIFICATION PATTERN
═══════════════════════════════════════════════════════════════

| Pattern của entity | Type | Khi nào |
|---|---|---|
| Compound `<verb> X` (X = organ) | CHẨN_ĐOÁN | "viêm X", "ung thư X", "suy X", "gãy X", "phình X", "hẹp X", "tắc mạch X", "huyết khối X" |
| Named disease / Acronym | CHẨN_ĐOÁN | THA, ĐTĐ, NMCT, COPD, RLLL, CKD, Kawasaki, Parkinson... |
| Cardiac / Vascular inflammatory | CHẨN_ĐOÁN | "viêm cơ tim", "viêm màng tim", "phình/hẹp mạch", "thuyên tắc" |
| Bệnh có verb "thiếu" (blood/enzyme) | CHẨN_ĐOÁN | "thiếu men G6PD", "thiếu máu tan huyết" |
| Đau X / Sốt / Ho / Khó thở / Vàng da | TRIỆU_CHỨNG | Bệnh nhân cảm nhận được |
| Substance alone (G6PD, men X) | TÊN_XÉT_NGHIỆM | Đứng riêng không có verb bệnh lý (tên chỉ định test) |
| Drug name cụ thể (Panadol, aspirin 325mg x 1) | THUỐC | Brand name hoặc INN + liều (nếu có) |
| Test name (CT, X-quang, siêu âm, ECG) | TÊN_XÉT_NGHIỆM | Chỉ định cận lâm sàng |
| Số + đơn vị (120/80 mmHg, 38.5°C, 96%) | KẾT_QUẢ_XÉT_NGHIỆM | Lab value / vital signs với unit |

═══════════════════════════════════════════════════════════════
PHẦN 3 — 5 CÂU HỎI VÀNG & BỘ LỌC SEMANTIC REASONING
═══════════════════════════════════════════════════════════════

{_GOLD_QUESTIONS}

{_ICD_CHAPTER_SEMANTICS}

## QUY TẮC LỌC NOISE (DỰA TRÊN 5 CÂU HỎI VÀNG):
1. **[Q1] Thời gian / Lối sống / Chatbot → DROP**: "trong tuần qua", "cách 3 ngày", "rượu bia", "thuốc lá", "Cảm ơn bạn đã hỏi..." → DROP.
2. **[Q2-Q3] Tế bào / Quá trình đứng riêng mô tả cơ chế → DROP**: "hồng cầu", "bạch cầu", "oxy hóa", "tan huyết", "phá hủy" đứng một mình trong câu giải thích cơ chế → DROP. (Bệnh nhân KHÔNG cảm nhận được). NGOẠI LỆ: "thiếu máu tan huyết", "giảm tiểu cầu" là CHẨN_ĐOÁN hợp lệ.
3. **[Q4] Tác nhân gây bệnh (Triggers) → DROP**: "đậu tằm", "băng phiến", "long não", "phấn hoa" (đứng riêng làm nguyên nhân) → DROP. Biểu hiện bệnh ("vàng da", "sốt") → TRIỆU_CHỨNG ✓.
4. **Drug-Class generic → DROP**: "kháng sinh", "corticoid", "NSAID", "thuốc hạ sốt" (nhóm thuốc, không tên cụ thể) → DROP. Cụ thể "aspirin", "metoprolol" → THUỐC ✓.
5. **Standalone dose / Qualifier → DROP**: "30 mg", "không đặc hiệu" đứng riêng → DROP.
6. **Core symptom extraction**: Cắt bỏ verb dẫn ("cảm thấy", "xuất hiện", "bị") và thời gian kéo dài. Lấy core: "mệt mỏi", "đau ngực", "khó thở".
7. **[Q6] "K dùng" / Phủ định "K" → DROP**: "K dùng" = "không dùng" (mệnh đề tự sự) → DROP. "K" + động từ/lối sống/mẹ cho bú ("K dùng thuốc nam", "Mẹ đang cho con bú") → DROP. NGOẠI LỆ: "K" + cơ quan ("K vú", "K phổi", "K dạ dày") = Ung thư → CHẨN_ĐOÁN ✓.

## QUY TẮC PHỤ:
- **Duplicates (R10 STRICT)**: Mỗi lần xuất hiện ở vị trí khác nhau trong input = 1 entity riêng biệt với position `[start, end)`.
- **Thuốc**: Giữ trọn vẹn đuôi liều `x N` (`aspirin 325mg x 1`). Bỏ lời dặn hành chính trong ngoặc.
- **Viết tắt y khoa**: BẮT BUỘC extract (`THA`, `ĐTĐ`, `NMCT`, `COPD`, `ECG`). KHÔNG mở rộng viết tắt trong text.

═══════════════════════════════════════════════════════════════
PHẦN 4 — CÁCH LÀM (BẮT BUỘC 2 BƯỚC CÓ COT)
═══════════════════════════════════════════════════════════════

## BƯỚC 1 — CHAIN-OF-THOUGHT (REASONING SCRATCHPAD):
Trước khi xuất JSON, quét qua 7 section bệnh án (Lý do vào viện, Tiền sử, Diễn biến, Khám, Cận lâm sàng, Chẩn đoán, Điều trị) và ghi chú ngắn:
```
SECTION 1 (Tiền sử): Found N entities [...]
SECTION 2 (Khám & CLS): Found M entities [...]
SECTION 3 (Chẩn đoán & Điều trị): Found K entities [...]
SELF-CHECK: Total = N+M+K entities. Spans exact match? Overlap checked?
```

## BƯỚC 2 — JSON OUTPUT:
```json
[
  {{"text": "exact_span", "position": [start, end)}},
  ...
]
```
- `position` là `[start, end)` Python convention (end EXCLUSIVE).
- CHỈ trả về JSON array sau scratchpad.
"""

STAGE2_PROMPT = f"""Bạn là chuyên gia phân loại thực thể y tế tiếng Việt lâm sàng.

═══════════════════════════════════════════════════════════════
PHẦN 1 — NGUYÊN TẮC BẤT BIẾN (CORE PRINCIPLES)
═══════════════════════════════════════════════════════════════

🎯 5 NGUYÊN TẮC:

(N1) **CHÍNH XÁC TEXT**: Giữ nguyên `text` từ input. KHÔNG sửa spelling, KHÔNG đổi case.
(N2) **BẢO TOÀN SỐ LƯỢNG**: MỖI mention ở input PHẢI có đúng 1 entry tương ứng ở output (thứ tự giữ nguyên). KHÔNG bỏ sót.
(N3) **5 TYPE DỰA TRÊN THỰC BẢN LÂM SÀNG**:
     - THUỐC | CHẨN_ĐOÁN | TRIỆU_CHỨNG | TÊN_XÉT_NGHIỆM | KẾT_QUẢ_XÉT_NGHIỆM
(N4) **ASSERTIONS CHÍNH XÁC**:
     - isNegated: từ phủ định (không, chưa, âm tính) trong cùng mệnh đề.
     - isHistorical: section Tiền sử hoặc marker quá khứ (từng bị, 5 năm trước).
     - isFamily: người thân trong gia đình (bố, mẹ, di truyền gia đình).
(N5) **TYPE=NULL (DROP) KHI LÀ NOISE**: nếu mention là rác/thời gian/lối sống/drug-class generic → set `type: null`.

═══════════════════════════════════════════════════════════════
PHẦN 2 — 5 CÂU HỎI VÀNG & CLASSIFICATION SEMANTICS
═══════════════════════════════════════════════════════════════

{_GOLD_QUESTIONS}

{_ICD_CHAPTER_SEMANTICS}

## QUY TẮC PHÂN LOẠI CHI TIẾT:
1. **[Q1] BN MẮC BỆNH hay CẢM NHẬN?**
   - Mắc bệnh / Tổn thương / Abnormal finding / Bệnh có ICD → **CHẨN_ĐOÁN**
     (vd: "viêm phổi", "THA", "ĐTĐ", "thiếu men G6PD", "nhiễm trùng", "tim to", "gãy xương", "ngoại tâm thu nhĩ")
   - BN cảm nhận cơ năng/thực thể → **TRIỆU_CHỨNG**
     (vd: "đau ngực", "khó thở", "sốt", "ho", "buồn nôn", "mệt mỏi", "vàng da")
   - Giá trị đo định lượng/định tính / Finding bình thường → **KẾT_QUẢ_XÉT_NGHIỆM**
     (vd: "120/80 mmHg", "38.5°C", "dương tính", "nhịp xoang đều", "ecg bình thường")

2. **[Q2-Q3] Enzyme / Substance / Test Name**:
   - Tên enzyme/chất đứng một mình không có disease verb ("G6PD", "men G6PD", "Glucose-6-Phosphate...") → **TÊN_XÉT_NGHIỆM**
   - Enzyme/chất kèm disease verb ("Thiếu men G6PD", "thiếu máu tan huyết") → **CHẨN_ĐOÁN**
   - Tên chỉ định CLS/hình ảnh ("ECG", "X-quang ngực", "siêu âm tim") → **TÊN_XÉT_NGHIỆM**

3. **[Q4] Thuốc cụ thể vs Class Generic**:
   - Tên thuốc cụ thể (brand hoặc INN, có hoặc không có liều) → **THUỐC**
     (vd: "aspirin 325mg", "metoprolol 25mg", "Panadol", "Vitamin K", "Crestor")
   - Drug-class generic ("kháng sinh", "corticoid", "NSAID", "thuốc hạ sốt") → **DROP** (`type: null`)

4. **Noise / Process / Trigger → DROP** (`type: null`):
   - Tế bào / Quá trình đứng riêng ("hồng cầu", "oxy hóa", "tan huyết", "phá hủy") → DROP
   - Trigger substances / Causes ("đậu tằm", "băng phiến", "ăn đậu tằm") → DROP
   - Time / Duration / Lifestyle / Family context ("trong tuần qua", "rượu bia", "thuốc lá", "Mẹ đang cho con bú") → DROP
   - Standalone body parts ("ngực", "đầu", "bụng" đứng một mình) → DROP

5. **[Q6] Từ viết tắt "K" (Ung thư vs Phủ định "Không")**:
   - `K` + cơ quan ("K vú", "K phổi", "K dạ dày", "K gan", "K giáp") → **CHẨN_ĐOÁN** (Ung thư)
   - `K` + động từ / tự sự ("K dùng", "K dùng thuốc nam", "K tiếp xúc") → `K` = `Không` (Không dùng) → **DROP** (`type: null`)

# ĐẦU VÀO
Văn bản gốc đầy đủ (để hiểu ngữ cảnh):
<input>
{{input_text}}
</input>

Danh sách mentions cần phân loại:
{{mentions_list}}

# ĐẦU RA
Trả về JSON array (KHÔNG kèm lời giải thích):
[
  {{"text": "...", "type": "THUỐC|CHẨN_ĐOÁN|TRIỆU_CHỨNG|TÊN_XÉT_NGHIỆM|KẾT_QUẢ_XÉT_NGHIỆM", "assertions": ["isNegated", "isHistorical", "isFamily"]}},
  ...
]
"""

def build_stage1_user_prompt(input_text: str) -> str:
    """Build user prompt cho Stage 1 Mention Extraction.

    R39 (2026-07-24): Enhanced với:
    - Anti-hallucination rules (chỉ copy nguyên văn, không thêm space/case)
    - Compound disease KHÔNG tách
    - Drug-class generic DROP
    - Standalone dose fragment DROP
    - Brand names → THUỐC
    - Acronym extraction mandatory
    - Span overlap check (R39)
    - Recall booster patterns (15 loại thường MISS)
    - RECALL COUNT CHECKLIST (đếm section theo type)
    """
    return (
        "🎯 NHIỆM VỤ: Tìm và trích xuất TRỌN VẸN và KIỆT ĐỂ tất cả các cụm từ y khoa (medical concept spans) trong văn bản lâm sàng dưới đây kèm vị trí character offset [start, end).\n\n"
        "🔥 9 QUY TẮC TRÍCH XUẤT LÂM SÀNG CỐT LÕI (BẮT BUỘC TUÂN THỦ TỪNG CHỮ):\n"
        "1. TRIỆU CHỨNG LÕI NGẮN GỌN: CHỈ lấy core symptom (`đau ngực`, `khó thở`, `mệt mỏi`, `đánh trống ngực`, `sốt`). TUYỆT ĐỐI KHÔNG bốc thêm đuôi tự sự / hoàn cảnh phía sau (`nhiều khi gắng sức`, `khi leo cầu thang`, `lúc nhập viện`) hoặc tiền tố lời kể/qualifier (`còn cảm giác`, `xuất hiện`, `bệnh nhân thấy`, `ghi nhận`, `có dấu hiệu`).\n"
        "2. TÁCH CỤM TRIỆU CHỨNG VỊ TRÍ KÉP: Nếu có cả cảm giác và vị trí giải phẫu (`cảm giác thắt chặt ngực vùng trước tim`, `tình trạng đau thắt ngực sau xương ức`), PHẢI tách thành 2 spans riêng: (`cảm giác thắt chặt ngực` VÀ `thắt chặt ngực vùng trước tim`), KHÔNG gộp chung 1 dải.\n"
        "3. CHUẨN HÓA TÊN XÉT NGHIỆM (BỎ ĐỘNG TỪ CHỈ ĐỊNH): Khi lấy TÊN_XÉT_NGHIỆM, TUYỆT ĐỐI KHÔNG lấy động từ chỉ định phía trước (`chụp`, `đo`, `làm`, `thực hiện`, `tiến hành`). Ví dụ: `chụp X-quang ngực` -> CHỈ lấy `X-quang ngực`; `đo điện tâm đồ` -> CHỈ lấy `điện tâm đồ`. LƯU Ý: Các cụm danh từ xét nghiệm toàn phần như `phân tích nước tiểu`, `siêu âm tim`, `nội soi dạ dày` PHẢI GIỮ NGUYÊN TRỌN VẸN (`phân tích nước tiểu`).\n"
        "4. THUỐC PHẢI ĐỦ ĐUÔI LIỀU LƯỢNG (`x N`): Khi có `aspirin 325mg x 1`, `paracetamol 500mg po bid`, PHẢI lấy trọn vẹn đến hết đuôi liều/tần suất (`aspirin 325mg x 1`), không được bỏ rơi chữ `x 1` phía sau.\n"
        "5. QUÉT HẾT TỪNG LẦN LẶP LẠI: Nếu một triệu chứng hay thuốc xuất hiện 3-4 lần ở các câu khác nhau từ Tiền sử đến Cấp cứu đến Khám, PHẢI xuất đủ 3-4 lần với positions tương ứng!\n"
        "6. LOẠI TRỪ RÁC PHI Y KHOA (NOISE REJECTION): TUYỆT ĐỐI KHÔNG trích xuất các cụm mốc thời gian độc lập (`trong tuần qua`, `cách đây 3 ngày`, `20 giây`, `từ sáng hôm nay`) hoặc thói quen sinh hoạt phi lâm sàng (`rượu bia`, `thuốc lá`, `ăn uống bình thường`).\n"
        "7. BẮT BUỘC TRÍCH XUẤT TỪ VIẾT TẮT Y KHOA (MANDATORY ACRONYM EXTRACTION): Hồ sơ bệnh án Việt Nam viết tắt rất nhiều. Bạn BẮT BUỘC phải trích xuất đầy đủ và chính xác tất cả các từ viết tắt bệnh lý/xét nghiệm (`THA` = Tăng huyết áp, `ĐTĐ` / `ĐTĐ tuýp 2` = Đái tháo đường, `NMCT` = Nhồi máu cơ tim, `RLLL` = Rối loạn lipid máu, `COPD` = Bệnh phổi tắc nghẽn mạn tính, `CKD` = Bệnh thận mạn, `BTMV`, `TBMMN`, `ECG`...) như những thực thể y khoa độc lập!\n"
        "8. QUÉT KIỆT ĐỂ 7 PHẦN BỆNH ÁN (EXHAUSTIVE SECTION COVERAGE): Bạn PHẢI quét tuần tự qua 7 phần (Lý do vào viện, Tiền sử, Diễn biến, Khám lâm sàng, Cận lâm sàng/ECG/Holter, Chẩn đoán xác định, Điều trị/Thuốc ra viện). Mọi thuốc, chẩn đoán, triệu chứng, tên xét nghiệm và chỉ số/kết quả bình thường đều phải được lấy đủ 100%! Không được lười biếng hay bỏ sót các entities ở phần giữa và cuối hồ sơ.\n"
        "9. 🚨 TUYỆT ĐỐI KHÔNG SPAN OVERLAP (R39 - 2026-07-22): Mỗi cụm từ chỉ thuộc về 1 entity duy nhất. KHÔNG BAO GIỜ extract 2 spans chồng lấn nhau. Vd: nếu đã extract `phình giãn động mạch vành` (25 chars) thì KHÔNG extract riêng `động mạch vành` (14 chars) làm entity khác — span ngắn nằm trọn trong span dài là redundant. Trước khi output, KIỂM TRA: có 2 spans nào overlap (1 span nằm trong span kia) không? Nếu có → XOÁ span ngắn hơn.\n\n"
        "🚨 10. CẤM 13 LOẠI TOKEN PHẢI DROP (R38 - 2026-07-22, BẮT BUỘC TUÂN THỦ):\n"
        "    - **Tế bào / thành phần máu alone**: `hồng cầu`, `bạch cầu`, `tiểu cầu` (BN không cảm nhận được → KHÔNG phải entity)\n"
        "    - **Quá trình hóa học / sinh học**: `oxy hóa`, `khử oxy`, `chuyển hóa`, `đông máu`, `tan huyết`, `phá hủy` (process, không phải entity)\n"
        "    - **Trigger substances**: `đậu tằm`, `băng phiến` (long não), `hóa chất`, `mủ cao su`, `phấn hoa` (causes, không phải diseases)\n"
        "    - **Generic categories**: `thuốc` (khi đứng riêng), `thực phẩm`, `Vitamin` (khi đứng riêng), `hormone` (khi đứng riêng), `chất điện giải` (loại chung, không phải specific drug)\n"
        "    - **Narrative actions**: `ăn đậu tằm`, `tiếp xúc với băng phiến`, `sử dụng thuốc`, `hạ sốt` (hành động trong câu, không phải entity)\n"
        "    - **Body parts alone**: `ngực`, `bụng`, `đầu`, `lưng`, `chân`, `tay`, `gót chân trẻ`, `máu khô` (chỉ là cơ quan, không có tính từ)\n"
        "    - **Descriptive fragments**: `thiếu men này`, `thiếu máu` (đứng riêng), `dễ bị phá hủy`, `có tính oxy hóa cao`, `thực phẩm chứa chất oxy hóa`, `mong manh` (mảnh câu, không phải entity)\n"
        "    - **Age descriptors**: `trẻ sơ sinh`, `trẻ em`, `người lớn` (chỉ là tuổi, không phải entity)\n"
        "    - **Generic procedures**: `phân tích`, `chẩn đoán`, `sàng lọc sớm`, `theo dõi`, `xét nghiệm chuyên sâu` (action verb, không phải tên test cụ thể)\n"
        "    - **Drug-class generic**: `kháng sinh`, `corticoid`, `NSAID`, `kháng viêm`, `kháng đông`, `thuốc hạ sốt`, `thuốc giảm đau` (class terms, không phải specific drug)\n"
        "    - **Mẹ đang cho con bú** (narrative, không phải entity)\n"
        "    - **lone descriptive words**: `này` (pronoun alone, không phải entity)\n"
        "    Test VÀNG: \"BN có CẢM NHẬN được cái này không?\" → KHÔNG → DROP. \"Cái này GÂY RA bệnh hay BIỂU HIỆN bệnh?\" → Gây ra → DROP. \"Đây là ANATOMY hay TEST?\" → Anatomy alone → DROP. \"Đây là DRUG CLASS hay SPECIFIC DRUG?\" → Class → DROP.\n\n"
        "11. 🔥 **ENZYME / ENZYME_NAME → TÊN_XÉT_NGHIỆM** (Stage 2 sẽ classify, Stage 1 PHẢI extract): Tên enzyme thuần (`Glucose-6-Phosphate Dehydrogenase`, `G6PD`, `AST`, `ALT`, `men G6PD` alone) → TÊN_XÉT_NGHIỆM (chỉ định test enzyme). NGOẠI LỆ: Khi enzyme name đi với text bệnh lý (vd \"Thiếu men G6PD\", \"G6PD deficiency\") → CHẨN_ĐOÁN (cả cụm là bệnh).\n\n"
        "12. 🔥 **BỆNH LÝ NẶNG / COMPLICATIONS → CHẨN_ĐOÁN** (không phải triệu chứng):\n"
        "    - `suy thận cấp`, `suy tim`, `suy gan`, `suy hô hấp` → CHẨN_ĐOÁN (bệnh lý nặng, không phải triệu chứng)\n"
        "    - `bại não`, `chậm phát triển trí tuệ`, `rối loạn vận động` → CHẨN_ĐOÁN (tổn thương thần kinh, abnormal findings)\n"
        "    - `thiếu máu tan huyết`, `thiếu máu do tan huyết` → CHẨN_ĐOÁN (bệnh lý về máu)\n"
        "    - `nhiễm khuẩn`, `nhiễm virus`, `nhiễm trùng` → CHẨN_ĐOÁN (bệnh nhiễm)\n"
        "    - `nhồi máu cơ tim`, `nhồi máu não`, `xuất huyết não` → CHẨN_ĐOÁN (bệnh lý nặng)\n"
        "    - Bất thường tim mạch (cardiac inflammatory) → CHẨN_ĐOÁN\n"
        "    - Bất thường / bệnh lý mạch máu (`phình X`, `hẹp X`, `tắc mạch X`, `huyết khối`, `thuyên tắc X`) → CHẨN_ĐOÁN\n"
        "    - Named disease (bất kỳ tên riêng nào của bệnh) → CHẨN_ĐOÁN\n"
        "    - Pediatric common diseases (bệnh truyền nhiễm thường gặp ở trẻ) → CHẨN_ĐOÁN\n"
        "    Test: Bệnh nhân MẮC BỆNH này hay CHỈ CẢM NHẬN? → Mắc bệnh = CHẨN_ĐOÁN.\n\n"
        "13. 🚀 **RECALL CHECKLIST** (đếm entity theo từng loại, đảm bảo không miss):\n"
        "    - CHẨN_ĐOÁN ≥ 5 entities (nếu bệnh án >2000 chars). Nếu không đủ → quay lại BƯỚC 1 scan thêm.\n"
        "    - TRIỆU_CHỨNG ≥ 8 entities. Đếm đủ số triệu chứng có trong input.\n"
        "    - TÊN_XÉT_NGHIỆM ≥ 2 entities. Nếu có CLS/chỉ định → extract.\n"
        "    - THUỐC đầy đủ name + strength + route. Nếu có điều trị mà chưa extract được thuốc nào → check lại.\n"
        "    - KẾT_QUẢ_XÉT_NGHIỆM: nếu input có số liệu có đơn vị (mmHg, °C, %, g/dL, etc.) → extract.\n\n"
        "14. 🚫 **KHÔNG EXTRACT** các thứ này (tránh hallucination penalty):\n"
        "    - Chatbot chitchat: \"Cảm ơn bạn đã gửi câu hỏi\", \"Hy vọng...\", \"Nếu bạn có thêm câu hỏi\", \"Xin chào\", \"Chúc bạn sức khỏe\", \"Xin lỗi bạn\".\n"
        "    - Câu narrative dài > 80 chars có dấu chấm phẩy (chắc chắn là câu, không phải span y khoa).\n"
        "    - Câu giải thích bệnh từ chatbot AI (vd \"Đây là một bệnh di truyền...\").\n"
        "    - Lời khuyên/phác đồ trừu tượng (vd \"Cách ly...\", \"Bổ sung...\").\n\n"
        "15. ✅ **CHUNK OVERLAP HANDLING** (R39): Nếu input có dấu hiệu overlap giữa 2 chunk (vd \"...viêm phổi...\" ở chunk 1 và chunk 2), KEEP tất cả các occurrence. Stage 3 sẽ dedupe.\n\n"
        f"INPUT:\n{input_text}\n\n"
        "🚨 ĐỊNH DẠNG OUTPUT: Trả về JSON array (vd `[{{'text': '...', 'position': [start, end]}}, ...]`). KHÔNG thêm text nào trước/sau JSON. KHÔNG dùng markdown code block. KHÔNG giải thích. Nếu bệnh án có thông tin y khoa (sốt, đau, phát ban, thuốc, xét nghiệm, chẩn đoán...) → BẮT BUỘC extract, KHÔNG trả `[]` rỗng trừ khi input thực sự không có entity nào.\n\n"
        "OUTPUT JSON ARRAY:"
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
    return STAGE2_PROMPT.replace("{input_text}", input_text).replace("{mentions_list}", mentions_str) + """

🚨 **ĐỊNH DẠNG OUTPUT BẮT BUỘC**: Trả về JSON array (vd `[{...}, {...}]`). KHÔNG thêm bất kỳ text nào TRƯỚC hoặc SAU JSON (không "Lý do:", không giải thích, không markdown code block). Nếu mention không rõ type → vẫn PHẢI trả entry với best guess type (KHÔNG được bỏ sót mention nào).
"""


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

Input: "{entity_text}"

Output: [rxcui1, rxcui2, ...]
"""


# ==============================================================================
# R37 (2026-07-16): STAGE 3 — LLM CONTEXT ANALYZER cho ICD/RxNorm candidates
# ==============================================================================
# Sau Stage 1 (extract) và Stage 2 (classify), Stage 3 cho LLM:
# - Review ICD codes cho CHẨN_ĐOÁN dựa trên full clinical context
# - Review RxNorm codes cho THUỐC dựa trên full clinical context
# - Verdict: ok (giữ), refine (đổi), drop (không nên có candidate)
#
# Quyết định:
# - Stage 3 default ON, opt-out bằng --no-stage3 flag (backward compat)
# - Scope: CHẨN_ĐOÁN + THUỐC only (skip các type khác để tránh hallucination)

STAGE3_PROMPT = f"""Bạn là chuyên gia Clinical Coding với 20+ năm kinh nghiệm ICD-10 và RxNorm. Nhiệm vụ: REVIEW lại các ICD-10 (cho CHẨN_ĐOÁN) và RxNorm (cho THUỐC) candidates đã được RAG đề xuất, dựa trên TOÀN BỘ clinical context phía trên.

═══════════════════════════════════════════════════════════════
PHẦN 1 — NGUYÊN TẮC BẤT BIẾN + ICD CHAPTER MAP
═══════════════════════════════════════════════════════════════

🎯 4 NGUYÊN TẮC:

(N1) **CODE PHẢI ĐÚNG CONCEPT**: code match y khoa (vd blood disease KHÔNG dùng Q-code testis).
(N2) **CHAPTER PHẢI ĐÚNG**: code thuộc đúng ICD chapter (vd viêm cơ tim → I40, KHÔNG B33.2 viral pericarditis).
(N3) **SPECIFICITY MATCH QUALIFIER**: có organism → specific subcode; có side/lobe → laterality subcode.
(N4) **NO INVENTED CODES**: chỉ trả codes có trong ICD-10/RxNorm standard. KHÔNG bịa code.

{_ICD_CHAPTER_SEMANTICS}

═══════════════════════════════════════════════════════════════
PHẦN 2 — CHAIN-OF-THOUGHT (CoT) BẮT BUỘC CHO MỖI ENTITY
═══════════════════════════════════════════════════════════════

Với MỖI entity, hãy suy nghĩ theo 4 bước:

1. **ĐỌC entity text + type + clinical context**:
   - "Entity này có text gì? Type gì (CHẨN_ĐOÁN/THUỐC)?"
   - "Trong context bệnh án, ý nghĩa của nó là gì?"

2. **ĐỐI CHIẾU với RAG candidates**:
   - "Mỗi candidate code match concept trong entity text không?"
   - "Code có cover đúng qualifier (organism, lobe, severity, side) không?"
   - "Code có thuộc đúng chapter/disease category không?"
   - "Code nào WRONG concept (vd Q55 testis defect cho G6PD anemia)?"

3. **CHẤM ĐIỂM relevance** (scale 1-10):
   - 9-10: PERFECT match (đúng concept, đúng subcode)
   - 7-8: GOOD match (đúng concept, generic subcode OK)
   - 5-6: PARTIAL match (đúng chapter, sai chi tiết)
   - 3-4: WEAK match (chỉ match keyword, khác concept)
   - 1-2: WRONG match (sai concept hoàn toàn)

4. **ĐƯA RA verdict**:
   - "ok": candidates hiện tại đúng, giữ nguyên.
   - "refine": có candidate tốt hơn, REPLACE.
   - "drop": candidates SAI hoặc entity không nên có candidate.

# OUTPUT FORMAT
TRẢ VỀ JSON array (MỖI element, KHÔNG thêm field thừa):
[
  {{
    "text": "<exact text from entity>",
    "type": "<exact type>",
    "verdict": "ok" | "refine" | "drop",
    "candidates": ["code1", "code2", ...],   // 0-5 codes, only ICD/RxNorm codes
    "reasoning": "<short — 1 sentence giải thích verdict + score>"
  }},
  ...
]

# QUY TẮC VERDICT
- "ok": current candidate chính xác (hoặc đủ tốt). Giữ nguyên `candidates`.
- "refine": có qualifier làm candidate hiện tại chưa chính xác. Replace `candidates` với codes tốt hơn.
- "drop": text là drug-class generic (vd "kháng sinh", "NSAID") hoặc không nên có candidate. Set `candidates = []`.

# ICD-10 SPECIFICITY RULES (chọn subcode cụ thể khi có qualifier trong text):
1. **Anatomical side** ("trái"/"phải"): prefer ICD có laterality khi text chỉ rõ. VD: "viêm phổi phải" → J18.1 nếu có; nếu không có, giữ J18.x generic.
2. **Etiology (organism)**: organism name → specific A0x subcode. VD: "Shigella dysenteriae" → A03.0 (KHÔNG A03 generic); "Salmonella" → A02.x; nếu organism không đặc hiệu → A04.x or generic.
3. **Acute vs chronic** ("cấp" vs "mạn"/"mãn"): different subcodes. VD: "viêm phế quản cấp" → J20; viêm phế quản mạn → J41-J42.
4. **Anatomical detail** ("thùy trên/dưới", "vách", "đoạn gần/xa"): check ICD có anatomical subcodes. VD: "ung thư phổi thùy trên" → C34.1 (KHÔNG chỉ C34 generic); "nhồi máu cơ tim thành trước" → I21.0; "vách liên thất" → specific I21 subcode.
5. **Severity grade** ("độ 1/2/3", "giai đoạn I-IV", "NYHA I/II/III/IV"): prefer subcode reflect severity. VD: tăng huyết áp độ 1/2/3 có thể map I10 vs I10 với qualifier thứ 5; nếu ICD không phân biệt severity, giữ I10.
6. **Organ + dysfunction pattern** ("rối loạn + organ"): typically single code, not multiple. VD: "rối loạn nhịp tim" → I47-I49 (arrhythmia block, KHÔNG cả I47 + I48 + I49); "rối loạn lipid máu" → E78 (single); "rối loạn giấc ngủ" → G47.
7. **Diabetes type** ("tiểu đường type 1/2"): "type 1" → E10, "type 2" → E11. Khi text có complication (neuropathy, retinopathy, nephropathy, ketoacidosis) → add 4th/5th char subcode (E10.4-, E11.6-, v.v.).
8. **Cancer + vị trí** ("ung thư X"): specific C-code với site. VD: "ung thư vú" → C50.x; ung thư phổi thùy trên → C34.1. Khi text có "di căn"/"metastasis" → add C77-C79 secondary code khi RAG chưa có.
9. **MI location** ("nhồi máu cơ tim vị trí"): "STEMI/NSTEMI + location" → I21.x; "thành trước" → I21.0; "thành dưới" → I21.1; "không rõ vị trí" → I21.3.
10. **Stroke type** ("đột quỵ"/"tai biến"): phân biệt nhồi máu não (ischemic) vs xuất huyết não (hemorrhagic). "nhồi máu não" → I63.x, "xuất huyết não" → I61, "chảy máu dưới màng nhện" → I60. Khi text KHÔNG nói rõ ischemic vs hemorrhagic → I64 (unspecified).
11. **Enzyme deficiency / genetic disease**: "Thiếu men X" → enzyme deficiency code. VD: "Thiếu men G6PD" → D55.0 (G6PD deficiency anemia). KHÔNG chọn Q-code (Q55 = testis defect) dù có từ "thiếu".

# 🚨 R39 (2026-07-24) — NGUYÊN TẮC LOẠI TRỪ CODE SAI CONCEPT (GENERIC):
KHÔNG hard-code từng case. Dùng NGUYÊN TẮC dưới đây áp dụng cho MỌI disease:

**A. Chapter/Block filter**: Chapter "Q" = Congenital malformations (BẨM SINH).
→ Nếu entity text là bệnh MẮC PHẢI (acquired) — KHÔNG phải bẩm sinh → loại bỏ TẤT CẢ Q-code.
→ Áp dụng cho: thiếu máu, viêm, nhiễm khuẩn, đột biến tự phát, v.v.

**B. Concept mismatch**: Code có cùng keyword nhưng khác concept.
→ VD: blood disease name match keywords Q55 (testis defect) → loại. Stroke code I60-I69 cho "đau đầu" → loại.

**C. Age filter**: Bệnh trẻ em không áp dụng cho người lớn (vd Kawasaki → M30.3 ở trẻ em, không phải generic vasculitis).
→ Bệnh người lớn không dùng M08 (juvenile RA).

**D. Modality filter**:
- Z00-Z99 = "factors influencing health" (screening, history) → KHÔNG dùng cho active disease.
- I77 (vasculitis NOS) → generic, prefer specific subcode.
- M70-M79 = "soft tissue disorders" → KHÔNG dùng cho organ-based disease.

**E. Blood/Kidney/Cardiac priority**:
- Bệnh về máu / enzyme → D50-D64 (D55 = enzyme deficiency)
- Bệnh về thận / tiết niệu → N00-N99
- Bệnh về tim mạch → I00-I99 (riêng I40 = myocarditis, I30 = pericarditis, I33 = endocarditis)
- KHÔNG dùng cross-chapter codes (vd G07 intracranial abscess cho coronary aneurysm)

**CÁCH ÁP DỤNG TRONG VERDICT**:
Với MỖI candidate code trong RAG list → ask 3 câu hỏi:
1. Code có thuộc đúng chapter không? (VASCULAR → I00-I99, BLOOD → D50-D64, etc.)
2. Code có match concept y khoa không? (Same body part + same pathology)
3. Code có bị blacklist không? (Q-code cho acquired disease, Z-code cho active disease, etc.)

→ Nếu fail bất kỳ câu nào → REMOVE khỏi candidates (verdict=refine).
→ Nếu TẤT CẢ candidates fail → verdict=drop, candidates=[].

# RxNorm SPECIFICITY RULES:
1. **Salt form**: "trimetazidine dihydrochloride" → rxcui 235779 (salt form). "trimetazidine" generic → rxcui 10826. Nếu text không specify salt → giữ generic (10826).
2. **Strength qualifier**: "aspirin 325mg" → SCD có strength 325 MG (vd 198467). "aspirin 81mg" → SCD 81 MG (vd 315677). KHÔNG chọn strength khác nếu text không match.
3. **Brand → INN**: Brand name có thể map sang INN ingredient. VD: "Panadol" → INN acetaminophen (rxcui 161). "Augmentin" → amoxicillin-clavulanate (rxcui 197884).
4. **Combination drugs**: Compound drug (vd "lisinopril/hydrochlorothiazide") → return MULTIPLE rxcui (mỗi component 1 rxcui). KHÔNG chỉ trả 1.
5. **No match / typo**: Nếu candidate từ RAG không match (vd "morphineoral" - typo ghép), verdict=drop, candidates=[].

# 🎯 CANDIDATE SCORING RUBRIC CHI TIẾT (BẮT BUỘC DÙNG)
Với MỖI candidate trong danh sách RAG, chấm điểm 1-10 theo 4 tiêu chí:

| Tiêu chí | Score cao (9-10) | Score trung bình (5-6) | Score thấp (1-2) |
|----------|-------------------|------------------------|-------------------|
| **Concept match** | Code mô tả ĐÚNG concept bệnh/thuốc | Code cùng chapter nhưng khác concept | Code hoàn toàn khác concept (vd Q55 testis defect cho G6PD anemia) |
| **Qualifier coverage** | Code cover đầy đủ organism + lobe + severity + side | Code cover partial (vd thiếu organism) | Code không cover qualifier nào |
| **Specificity** | Code có subcode cụ thể (vd J18.1 thay vì J18.9) | Code là parent generic | Code quá generic (chapter only) |
| **Clinical context** | Code match với bệnh cảnh lâm sàng (vd gout → M10, không phải M11) | Code có thể match nhưng không lý tưởng | Code match sai specialty |

# Ví dụ scoring:
- "viêm phổi do Streptococcus pneumoniae" + [J15.0, J15.9, J18.9]
  - J15.0 (pneumococcal pneumonia) → 10 (concept + organism)
  - J15.9 (bacterial pneumonia, unspecified) → 7 (correct concept, no organism)
  - J18.9 (pneumonia, unspecified) → 4 (correct chapter, too generic)
- "G6PD deficiency" + [D55.0, Q55.0, D55, E55.0]
  - D55.0 (G6PD deficiency anemia) → 10 (PERFECT)
  - D55 (anemia from enzyme disorders) → 7 (parent, correct concept)
  - Q55.0 (testis absence) → 1 (WRONG concept - testis defect, **CHỌC SAI CHAPTER**)
  - E55.0 (vitamin A deficiency) → 1 (WRONG concept - vitamin not blood, **CHỌC SAI CHAPTER**)

# CANDIDATE RANKING RULES (sau khi chấm điểm):
1. **Sort theo score giảm dần**.
2. **Giữ top-K candidates** có score ≥ 7 (high confidence).
3. **Bỏ candidates score ≤ 3** (likely wrong concept / noise).
4. **Nếu TẤT CẢ candidates score < 5**: trả `candidates: []` (entity không match code nào).

# EXAMPLES (THAM KHẢO, không bắt buộc giống):

# 🧠 R40 (2026-07-24) — SEMANTIC CHAPTER REASONING (thay thế HARD REJECTION TABLE):
# KHÔNG hard-code từng case. Dùng quy trình reasoning dưới đây cho MỌI candidate:

**QUY TRÌNH KIỂM TRA CHAPTER (bắt buộc với mỗi candidate):**
STEP 1: Xác định chapter của candidate code (tra bảng _ICD_CHAPTER_SEMANTICS phía trên)
STEP 2: Xác định chapter phù hợp với entity text (dựa trên cơ chế bệnh)
STEP 3: Chapter có khớp? → YES: giữ | NO: REMOVE với reasoning rõ ràng

**Ví dụ áp dụng:**

Case G6PD: text="Thiếu men G6PD" + candidates=[D55.0, Q55.0, D55, E55.0]
- D55.0: STEP1=D-chapter(máu) → STEP2=enzyme deficiency máu→D-chapter → MATCH ✓
- Q55.0: STEP1=Q-chapter(congenital testis) → STEP2=enzyme deficiency→D-chapter → MISMATCH ✗ REMOVE
- D55:   STEP1=D-chapter(máu) → STEP2=D-chapter → MATCH ✓ (parent ok)
- E55.0: STEP1=E-chapter(nutrition/vitamin) → STEP2=enzyme deficiency→D-chapter → MISMATCH ✗ REMOVE
→ verdict=refine, candidates=[D55.0, D55]

Case tan huyết: text="tan huyết do G6PD" + candidates=[D59, B21, D55.0]
- D59: STEP1=D-chapter(acquired hemolytic anemia) → STEP2=tan huyết→D-chapter → MATCH ✓
- B21: STEP1=B-chapter(HIV) → STEP2=tan huyết KHÔNG phải nhiễm HIV → MISMATCH ✗ REMOVE
- D55.0: STEP1=D-chapter(G6PD enzyme) → STEP2=tan huyết do G6PD → MATCH ✓
→ verdict=refine, candidates=[D59, D55.0]

Case viêm cơ tim: text="viêm cơ tim" + candidates=[I40, B33.2]
- I40: STEP1=I-chapter(acquired myocarditis) → STEP2=viêm cơ tim→I-chapter → MATCH ✓
- B33.2: STEP1=B-chapter(viral pericarditis) → STEP2=viêm CƠ TIM acquired→I-chapter, không phải pericarditis B-chapter → MISMATCH ✗ REMOVE
→ verdict=refine, candidates=[I40]

**NGUYÊN TẮC GENERAL:**
- Enzyme deficiency máu (G6PD, pyruvate kinase...) → D55-D58, KHÔNG Q, KHÔNG E
- Tan huyết (hemolysis) → D-chapter, KHÔNG B-chapter (B = nhiễm trùng)
- Bệnh tim acquired (viêm cơ tim, suy tim) → I-chapter, KHÔNG B33 viral
- Bệnh bẩm sinh cấu trúc → Q-chapter (CHỈ khi text nói rõ "bẩm sinh"/"congenital")
- Drug-class generic (corticoid, kháng sinh, Vitamin 3B) → DROP (không có specific rxcui)

# RxNorm SEMANTIC REJECTION:
- Tên nhóm thuốc (không phải INN/generic): kháng sinh, NSAID, corticoid, vitamin generic → DROP candidates=[]
- Vitamin supplement generic không có liều/form: Vitamin 3B, multivitamin → DROP candidates=[]
- Lab substance (bicarbonate, sodium) trong context measurement → DROP (lab value ≠ drug)
- Drug class resistance mention ("kháng methicillin") → DROP (resistance ≠ drug treatment)


- text="loét tá tràng" type="CHẨN_ĐOÁN" cand=[K26] → verdict=ok
- text="viêm phổi do covid" type="CHẨN_ĐOÁN" cand=[U07.1] → verdict=ok
- text="viêm phổi do vi khuẩn" type="CHẨN_ĐOÁN" cand=[J15.9] → verdict=refine, cand=[J15, J15.9]
- text="bệnh lỵ trực khuẩn do Shigella dysenteriae" type="CHẨN_ĐOÁN" cand=[A03] → verdict=refine, cand=[A03.0]
- text="ung thư phổi thùy trên" type="CHẨN_ĐOÁN" cand=[C34] → verdict=refine, cand=[C34.1]
- **(Disease có blacklist candidate → DROP code sai concept)**:
  - "X deficiency" cand=[D55.x, Q55, D55, vitamin E55] → verdict=refine, candidates=[D55.x only] (drop Q55 + E55 - sai concept)
  - "(Myocarditis có viral pericarditis trả) X viêm tim" cand=[I40, B33.2] → drop B33.2
- **text="corticoid liều cao kéo dài" type="THUỐC" cand=[???] → verdict=drop, cand=[]** (class generic, không specific drug)
- **text="Vitamin 3B" type="THUỐC" cand=[???] → verdict=drop, cand=[]** (vitamin generic, không specific)
- **text="trimetazidin" type="THUỐC" cand=[???] → verdict=ok, cand=[10826]** (typo nhỏ - fuzzy match "trimetazidine")
- text="kháng sinh" type="THUỐC" cand=[A07] → verdict=drop, cand=[]
- text="metoprolol 25mg" type="THUỐC" cand=[866924] → verdict=ok
- text="aspirin" type="THUỐC" cand=[198467] → verdict=ok

═══════════════════════════════════════════════════════════════
PHẦN 5 — R39-SELFVERIFY (TỰ KIỂM TRA CUỐI)
═══════════════════════════════════════════════════════════════

Trước khi trả JSON cuối cùng, LLM phải check 7 điều kiện cho MỖI entity:

1. ✅ Code tồn tại trong ICD-10/RxNorm (không invented)
2. ✅ Code match concept y khoa của entity text
3. ✅ Code thuộc đúng chapter/disease category
4. ✅ Code có SPECIFICITY (prefer subcode nếu qualifier có)
5. ✅ CANDIDATES ≤ 5 codes (top-K)
6. ✅ Loại bỏ codes score < 5 (không invent)
7. ✅ Verdict logic nhất quán: ok = keep, refine = replace, drop = []

→ Nếu fail bất kỳ check nào → SỬA TRƯỚC khi return. KHÔNG skip self-verify.

⚠️ CHỈ trả về JSON array, KHÔNG giải thích trước/sau. Code phải tồn tại trong ICD-10 hoặc RxNorm — không invent.
"""


# R37 (2026-07-16): Stage 3 few-shot examples — message-level (user → assistant pairs).
# Default 8 examples hardcoded từ prompt inline cũ, convert sang JSON response format.
# LLM học output format + verdict logic qua concrete input→output pairs.
_STAGE3_FEW_SHOT_POOL: list[dict] = [
    {
        "context": "Bệnh nhân nam 50 tuổi, đau thượng vị 2 tuần, nội soi thấy loét tá tràng kích thước 1.5cm.",
        "text": "loét tá tràng",
        "type": "CHẨN_ĐOÁN",
        "candidates": ["K26"],
        "verdict": "ok",
        "refined_candidates": ["K26"],
        "reasoning": "K26 đúng cho loét tá tràng không có biến chứng cụ thể.",
    },
    {
        "context": "Bệnh nhân nữ 65 tuổi, sốt cao, ho đờm vàng, PCR COVID dương tính, X-quang thấy thâm nhiễm hai phổi.",
        "text": "viêm phổi do covid",
        "type": "CHẨN_ĐOÁN",
        "candidates": ["U07.1"],
        "verdict": "ok",
        "refined_candidates": ["U07.1"],
        "reasoning": "U07.1 đúng cho COVID-19 có biểu hiện viêm phổi.",
    },
    {
        "context": "Bệnh nhân sốt cao, ho đờm mủ, X-quang thấy thâm nhiễm thùy dưới phổi phải, cấy đờm ra Streptococcus pneumoniae.",
        "text": "viêm phổi do vi khuẩn",
        "type": "CHẨN_ĐOÁN",
        "candidates": ["J15.9"],
        "verdict": "refine",
        "refined_candidates": ["J15", "J15.9"],
        "reasoning": "J15.9 quá generic — nên include parent J15 để giữ code-block.",
    },
    {
        "context": "Bệnh nhân tiêu chảy phân nhầy máu, cấy phân phân lập Shigella dysenteriae.",
        "text": "bệnh lỵ trực khuẩn do Shigella dysenteriae",
        "type": "CHẨN_ĐOÁN",
        "candidates": ["A03"],
        "verdict": "refine",
        "refined_candidates": ["A03.0"],
        "reasoning": "Shigella dysenteriae → A03.0 (cụ thể nhóm), A03 generic quá rộng.",
    },
    {
        "context": "Bệnh nhân nam 70 tuổi, ho máu, CT ngực thấy khối u thùy trên phổi trái kích thước 4cm, sinh thiết xác nhận ung thư biểu mô tuyến.",
        "text": "ung thư phổi thùy trên",
        "type": "CHẨN_ĐOÁN",
        "candidates": ["C34"],
        "verdict": "refine",
        "refined_candidates": ["C34.1"],
        "reasoning": "C34 generic — refine thành C34.1 (ung thư thùy trên phổi phải) hoặc C34.2 nếu là phổi trái.",
    },
    {
        "context": "Bệnh nhân được kê đơn kháng sinh amoxicillin cho viêm phổi.",
        "text": "kháng sinh",
        "type": "THUỐC",
        "candidates": ["A07"],
        "verdict": "drop",
        "refined_candidates": [],
        "reasoning": "'kháng sinh' là drug-class generic, không phải thuốc cụ thể — không nên có candidate.",
    },
    {
        "context": "Bệnh nhân tăng huyết áp, đang dùng metoprolol 25mg uống 2 lần/ngày.",
        "text": "metoprolol 25mg",
        "type": "THUỐC",
        "candidates": ["866924"],
        "verdict": "ok",
        "refined_candidates": ["866924"],
        "reasoning": "RxNorm 866924 đúng cho metoprolol tartrate 25mg.",
    },
    {
        "context": "Bệnh nhân đau ngực, được cho aspirin 500mg uống ngay.",
        "text": "aspirin",
        "type": "THUỐC",
        "candidates": ["198467"],
        "verdict": "ok",
        "refined_candidates": ["198467"],
        "reasoning": "RxNorm 198467 đúng cho aspirin (acetylsalicylic acid).",
    },
    # ===== R38 (2026-07-23) — New examples cho audit findings =====
    {
        "context": "Bệnh nhân nam 25 tuổi, xét nghiệm G6PD giảm, hồng cầu bình thường. Tiền sử gia đình có người thiếu men G6PD.",
        "text": "Thiếu men G6PD",
        "type": "CHẨN_ĐOÁN",
        "candidates": ["D55.0", "Q55.0", "D55", "E55.0"],
        "verdict": "refine",
        "refined_candidates": ["D55.0", "D55"],
        "reasoning": (
            "CHAPTER REASONING: 'Thiếu men G6PD' = enzyme deficiency của HỒNG CẦU → hệ MÁU → D-chapter. "
            "D55.0 = G6PD deficiency anemia (D-chapter blood) → CHAPTER MATCH ✓ GIỮ. "
            "D55 = parent code (D-chapter blood) → CHAPTER MATCH ✓ GIỮ. "
            "Q55.0 = testis absence bẩm sinh (Q-chapter congenital anatomy) → CHAPTER MISMATCH ✗ DROP "
            "(enzyme deficiency máu ≠ dị tật cấu trúc sinh dục). "
            "E55.0 = vitamin A deficiency (E-chapter nutrition) → CHAPTER MISMATCH ✗ DROP "
            "(enzyme deficiency ≠ vitamin deficiency). "
            "VERDICT: refine → giữ D55.0 + D55, drop Q55.0 + E55.0."
        ),
    },
    {
        "context": "Bệnh nhân nam 25 tuổi thiếu men G6PD. Sau khi ăn đậu tằm, xuất hiện vàng da, tiểu sẫm màu, hồng cầu bị phá hủy hàng loạt.",
        "text": "hồng cầu bị phá hủy hàng loạt",
        "type": "CHẨN_ĐOÁN",
        "candidates": ["D59.0", "B21", "D55.0", "D58.9"],
        "verdict": "refine",
        "refined_candidates": ["D59.0", "D55.0"],
        "reasoning": (
            "CHAPTER REASONING: 'hồng cầu bị phá hủy hàng loạt' = hemolysis (tan huyết) → hệ MÁU → D-chapter. "
            "D59.0 = autoimmune hemolytic anemia (D-chapter blood hemolysis) → CHAPTER MATCH ✓ GIỮ. "
            "D55.0 = G6PD deficiency (D-chapter blood, context confirms G6PD) → CHAPTER MATCH ✓ GIỮ. "
            "B21 = HIV disease (B-chapter infection) → CHAPTER MISMATCH ✗ DROP "
            "(hemolysis do G6PD KHÔNG phải nhiễm HIV; tan huyết ≠ B-chapter). "
            "D58.9 = hereditary hemolytic anemia (D-chapter) → CHAPTER MATCH nhưng context G6PD specific → lower priority. "
            "VERDICT: refine → giữ D59.0 + D55.0, drop B21."
        ),
    },

    {
        "context": "Bệnh nhân lupus ban đỏ hệ thống đang điều trị corticoid liều cao kéo dài 6 tháng qua.",
        "text": "corticoid liều cao kéo dài",
        "type": "THUỐC",
        "candidates": ["D07AA02", "H02AB06", "QD07AC"],
        "verdict": "drop",
        "refined_candidates": [],
        "reasoning": "'corticoid' là drug-class generic, không phải specific drug — không nên có candidate nào.",
    },
    {
        "context": "Bệnh nhân thiếu vitamin sau phẫu thuật, được bổ sung Vitamin 3B mỗi ngày.",
        "text": "Vitamin 3B",
        "type": "THUỐC",
        "candidates": ["D08AA02", "A11DA01", "B03BA05"],
        "verdict": "drop",
        "refined_candidates": [],
        "reasoning": "'Vitamin 3B' là vitamin supplement generic, không specific drug — không nên có candidate.",
    },
    {
        "context": "Bệnh nhân đau thắt ngực ổn định, đang dùng trimetazidin 35mg MR 2 lần/ngày.",
        "text": "trimetazidin",
        "type": "THUỐC",
        "candidates": ["10826"],
        "verdict": "ok",
        "refined_candidates": ["10826"],
        "reasoning": "Trimetazidine generic (typo 1 char missing 'e') → rxcui 10826.",
    },
    {
        "context": "Bệnh nhân nữ 30 tuổi, mệt mỏi, xét nghiệm cho thấy glucose máu tăng cao 280 mg/dL, HbA1c 9.2%.",
        "text": "đái tháo đường type 2",
        "type": "CHẨN_ĐOÁN",
        "candidates": ["E11", "E10", "E11.9", "E11.0"],
        "verdict": "refine",
        "refined_candidates": ["E11", "E11.9"],
        "reasoning": "Type 2 → E11 (correct), E11.9 (without complications, HbA1c cao chưa rõ biến chứng). E10 (type 1) SAI — drop.",
    },
    {
        "context": "Trẻ sơ sinh nam, sau sinh phát hiện bất thường bẩm sinh về đường tiết niệu.",
        "text": "bất thường bẩm sinh đường tiết niệu",
        "type": "CHẨN_ĐOÁN",
        "candidates": ["Q64.0", "Q60.0", "Q61.1"],
        "verdict": "ok",
        "refined_candidates": ["Q64.0", "Q60.0"],
        "reasoning": "Congenital urinary anomaly → Q-codes OK (entity có keyword 'bẩm sinh').",
    },
    {
        "context": "Bệnh nhân nam 60 tuổi, đau thắt lưng, MRI cột sống thắt lưng thoát vị đĩa đệm L4-L5.",
        "text": "thoát vị đĩa đệm",
        "type": "CHẨN_ĐOÁN",
        "candidates": ["M51.2", "M50.2", "M54.5"],
        "verdict": "refine",
        "refined_candidates": ["M51.2", "M54.5"],
        "reasoning": "Lumbar disc herniation → M51.2 (specific). M54.5 (low back pain) là TRIỆU_CHỨNG, không phải chẩn đoán.",
    },
    {
        "context": "Bệnh nhân nữ 45 tuổi, đau ngực trái dữ dội khi gắng sức, nghỉ ngơi giảm sau 5 phút.",
        "text": "đau ngực",
        "type": "TRIỆU_CHỨNG",
        "candidates": ["I20.9", "R07.4"],
        "verdict": "drop",
        "refined_candidates": [],
        "reasoning": "'đau ngực' là TRIỆU_CHỨNG, không phải CHẨN_ĐOÁN — drop candidates ICD.",
    },
]


def format_few_shot_stage3_messages(
    examples: list[dict] | None = None,
    max_examples: int = 16,
) -> list[dict[str, str]]:
    """R37 (2026-07-16): Convert Stage 3 few-shot examples → OpenAI chat message pairs.

    Mỗi example → 2 messages:
      - user: clinical context + entity payload (text, type, candidates)
      - assistant: JSON array với verdict + refined candidates

    Args:
        examples: list of dict (mặc định: _STAGE3_FEW_SHOT_POOL hardcoded).
        max_examples: cap số example (default 16, đủ cho LLM học format + verdict logic).
            R38 (2026-07-23): tăng từ 8 → 16 để có nhiều R38 patterns (Q-code, drug-class).

    Returns:
        List of message dicts (OpenAI chat format), xen giữa system và user prompt.
    """
    import json as _json

    pool = examples if examples is not None else _STAGE3_FEW_SHOT_POOL
    selected = pool[:max_examples]

    messages: list[dict[str, str]] = []
    for ex in selected:
        ctx = ex.get("context", "")[:600]
        candidates = ex.get("candidates", [])
        cand_str = ",".join(str(c) for c in candidates) if candidates else "(none)"
        user_msg = (
            f"# Clinical note\n{ctx}\n\n"
            f"# Entities cần verify (1 entity)\n"
            f"- 1. text=\"{ex['text']}\" type=\"{ex['type']}\" cand=[{cand_str}]\n\n"
            f"# Trả về JSON array (verdict + refined candidates cho MỖI entity theo thứ tự)."
        )
        assistant_payload = [{
            "text": ex["text"],
            "type": ex["type"],
            "verdict": ex["verdict"],
            "candidates": ex.get("refined_candidates", candidates),
            "reasoning": ex.get("reasoning", ""),
        }]
        assistant_msg = _json.dumps(assistant_payload, ensure_ascii=False)

        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})

    return messages


def build_stage3_user_prompt(
    input_text: str,
    entities_with_candidates: list[dict],
    batch_size: int = 30,
) -> list[str]:
    """R37 (2026-07-16): Build Stage 3 user prompts (batched).

    Args:
        input_text: full clinical note
        entities_with_candidates: list of {text, type, candidates} for CHẨN_ĐOÁN + THUỐC only
        batch_size: max entities per LLM call (default 30)

    Returns:
        List of user prompt strings (one per batch). Caller runs LLM on each, parses, merges results.
    """
    if not entities_with_candidates:
        return []

    batches: list[str] = []
    for i in range(0, len(entities_with_candidates), batch_size):
        batch = entities_with_candidates[i:i + batch_size]
        lines = []
        for j, e in enumerate(batch):
            cand = e.get("candidates", [])
            cand_str = ",".join(str(c) for c in cand) if cand else "(none)"
            lines.append(f"- {j+1}. text=\"{e.get('text','')}\" type=\"{e.get('type','')}\" cand=[{cand_str}]")

        entities_str = "\n".join(lines)
        # Compact context first 800 chars to avoid bloat (LLM has the entity text already)
        context_short = input_text[:800] + ("..." if len(input_text) > 800 else "")
        prompt = (
            f"# Clinical note\n{context_short}\n\n"
            f"# Entities cần verify ({len(batch)} entities)\n{entities_str}\n\n"
            f"# Trả về JSON array (verdict + refined candidates cho MỖI entity theo thứ tự)."
        )
        batches.append(prompt)
    return batches


# ════════════════════════════════════════════════════════════════════════════════
# R38 (2026-07-23): LLM ReRank prompt — Score-based candidate ranking
# ════════════════════════════════════════════════════════════════════════════════

RERANK_PROMPT = f"""Bạn là chuyên gia Clinical Coding với 20+ năm kinh nghiệm ICD-10 và RxNorm.

{_ICD_CHAPTER_SEMANTICS}

# NHIỆM VỤ
Với MỖI entity bệnh án dưới đây, bạn nhận được 1 danh sách các candidate codes (ICD-10 cho CHẨN_ĐOÁN, RxNorm rxcui cho THUỐC) đã được vector/BM25 search đề xuất.

# 🧠 CHAIN-OF-THOUGHT (CoT) - BẮT BUỘC
1. **ĐỌC TOÀN BỘ clinical note** để hiểu context đầy đủ (tiền sử, triệu chứng, kết quả xét nghiệm, v.v.).
2. **ĐỌC entity text** (có thể viết tắt, không chuẩn, hoặc đầy đủ).
3. **XÁC ĐỊNH chapter ICD phù hợp** với entity text (tra bảng Chapter Semantics Map phía trên):
   - Hỏi: "Bệnh này thuộc hệ cơ quan nào?" → Xác định chapter
   - Hỏi: "Bệnh này ACQUIRED hay BẨM SINH cấu trúc?" → Phân biệt D/I/J vs Q
4. **ĐỐI CHIẾU từng candidate với chapter đúng**:
   - "Code này thuộc chapter nào?"
   - "Chapter của code có khớp chapter cần không?"
   - "Code có sai concept không (vd Q55 testis defect cho G6PD anemia — sai cả chapter lẫn concept)?"
5. **CHẤM ĐIỂM** mỗi candidate từ 1-10 dựa trên độ phù hợp với entity + context:

# SCORING RUBRIC (CHAPTER-FIRST APPROACH):
| Score | Ý nghĩa | Ví dụ |
|-------|---------|--------|
| **9-10** | PERFECT match — đúng chapter + đúng concept + đầy đủ qualifier. | "G6PD deficiency" → D55.0 (D-chapter blood enzyme, correct); "viêm phổi thùy dưới phải do S.pneumoniae" → J15.1 |
| **7-8** | GOOD match — đúng chapter + đúng concept, thiếu 1-2 qualifier (organism, lobe, severity). | "viêm phổi do vi khuẩn" → J15.9 (no organism); "aspirin 500mg" → generic aspirin 161 |
| **5-6** | PARTIAL match — đúng chapter, sai chi tiết concept (wrong type, wrong subsite, wrong severity). | "đái tháo đường type 2" → E10 (type 1, SAI); "ung thư phổi" → C34 generic (thiếu lobe) |
| **3-4** | WEAK match — SAI CHAPTER. Code match keyword nhưng thuộc chapter không phù hợp với cơ chế bệnh. | "Thiếu men G6PD" → Q55.0 (Q-chapter congenital testis — SAI CHAPTER, G6PD là blood D-chapter) |
| **1-2** | WRONG match — SAI CHAPTER + SAI CONCEPT hoàn toàn. Noise từ vector search. | "Thiếu men G6PD" → E55.0 (E-chapter vitamin nutrition — không phải enzyme máu); "tan huyết" → B21 (B-chapter HIV — không phải hemolysis) |



4. **CHỌN TOP-K** candidates có score cao nhất (mặc định K=5, hoặc theo chỉ định trong input).
5. **BỎ QUA** candidates có score < 3 — đây là những match sai do vector search.
6. **GIỮ NGUYÊN THỨ TỰ** sort theo score giảm dần.

# 🎯 CRITERIA CHI TIẾT CHO MỖI ENTITY (4 tiêu chí đánh giá):
- **Concept match (40%)**: Code có mô tả ĐÚNG concept bệnh/thuốc trong entity không?
- **Qualifier coverage (30%)**: Code cover đầy đủ qualifier trong entity (organism, lobe, severity, side)?
- **Specificity (20%)**: Code có subcode cụ thể hay chỉ parent generic?
- **Clinical context (10%)**: Code match với bệnh cảnh lâm sàng và tuổi/giới?

# TOP-K SELECTION (sau khi chấm điểm):
- K=5 mặc định (hoặc theo top_k parameter)
- Chỉ giữ candidates có score ≥ 5 (medium confidence trở lên)
- Nếu TẤT CẢ candidates score < 5: trả `ranked_candidates: []`
- Sắp xếp theo score giảm dần, giữ context-aware order

# EDGE CASES (rất quan trọng - LLM cần handle đúng):
- **Tiền sử:** "Tiền sử: THA 10 năm" → code I10 vẫn valid, nhưng score giảm 1 điểm vì là isHistorical.
- **Family history**: "Bố mẹ bị X" → code đúng concept nhưng score = 1 (KHÔNG phải của bệnh nhân).
- **Compound diseases**: "viêm phổi thùy dưới do S. pneumoniae" → J15.1 (score 10) > J15.9 (score 5) > J18.9 (score 2).
- **Drug-class generic**: "kháng sinh", "NSAID" → KHÔNG có code RxNorm specific, trả `[]`.
- **Type mismatch**: text là TRIỆU_CHỨNG (vd "đau ngực") có candidate ICD (vd I20) → vẫn GIỮ I20 vì có thể useful cho diagnosis tracking, NHƯNG cảnh báo trong reason.

# OUTPUT FORMAT (JSON array, MỖI element cho 1 entity theo đúng thứ tự input)
- "Tiền sử:" trước entity → likely isHistorical, code vẫn valid nhưng confidence giảm.
- "Bố/Mẹ/Ông/Bà bị X" → isFamily, code đúng concept nhưng không phải của bệnh nhân.
- "di truyền" + congenital disease → Q-code có thể valid; nếu KHÔNG có keyword này → Q-code = 1.

# OUTPUT FORMAT (JSON array, MỖI element cho 1 entity theo đúng thứ tự input)
[
  {{
    "text": "<exact entity text>",
    "type": "<CHẨN_ĐOÁN hoặc THUỐC>",
    "ranked_candidates": [
      {{"code": "<code1>", "score": <int 1-10>, "reason": "<1 sentence giải thích score>"}},
      {{"code": "<code2>", "score": <int 1-10>, "reason": "<1 sentence>"}},
      ...
    ]
  }},
  ...
]

# QUY TẮC BẮT BUỘC
- MỖI entity trong input phải có ĐÚNG 1 object trong output (giữ thứ tự).
- `ranked_candidates` PHẢI sort theo score giảm dần (cao nhất trước).
- CHỈ giữ candidates có score >= 3. Nếu TẤT CẢ candidates có score < 3 → trả `ranked_candidates: []` (entity không match code nào).
- KHÔNG thêm candidates ngoài danh sách input (không hallucinate code mới).
- LÝ DO chấm điểm (`reason`) phải NGẮN GỌN (≤ 15 từ) và dựa trên context.
- ⚠️ Nếu entity text nói về bệnh lý của BẢN THÂN bệnh nhân (không phải tiền sử gia đình) → code vẫn là của bệnh nhân.
- ⚠️ CHỈ trả JSON array, KHÔNG giải thích trước/sau.

# EXAMPLES (tham khảo format, KHÔNG bắt buộc kết quả giống)

## Example 1 — G6PD deficiency (Q-code trap)
Input: text="Thiếu men G6PD" type="CHẨN_ĐOÁN"
Candidates: ["D55.0", "Q55.0", "D55", "E55.0"]
Context: bệnh nhân có "Thiếu men G6PD", "xét nghiệm G6PD giảm", "hồng cầu bình thường"
Output:
{{
  "text": "Thiếu men G6PD",
  "type": "CHẨN_ĐOÁN",
  "ranked_candidates": [
    {{"code": "D55.0", "score": 10, "reason": "G6PD deficiency chính xác"}},
    {{"code": "D55", "score": 7, "reason": "Generic anemia thiếu men, đúng category"}},
    {{"code": "E55.0", "score": 1, "reason": "Sai - E55 là vitamin A, không liên quan"}},
    {{"code": "Q55.0", "score": 1, "reason": "Sai - Q55 là testis defect, không phải G6PD"}}
  ]
}}

## Example 2 — Diabetes type 2 with complication
Input: text="đái tháo đường type 2 biến chứng thần kinh" type="CHẨN_ĐOÁN"
Candidates: ["E11", "E11.9", "E10", "E11.4", "E11.40", "G62.9"]
Output:
{{
  "text": "đái tháo đường type 2 biến chứng thần kinh",
  "type": "CHẨN_ĐOÁN",
  "ranked_candidates": [
    {{"code": "E11.4", "score": 10, "reason": "Type 2 + neurological complication"}},
    {{"code": "E11.40", "score": 9, "reason": "Specific subtype với neurological"}},
    {{"code": "E11", "score": 7, "reason": "Type 2 generic, cover đúng bệnh"}},
    {{"code": "E11.9", "score": 5, "reason": "Type 2 w/o complication, thiếu thần kinh"}},
    {{"code": "E10", "score": 1, "reason": "Sai - Type 1, không phải Type 2"}},
    {{"code": "G62.9", "score": 1, "reason": "Polyneuropathy generic, không phải complication của tiểu đường"}}
  ]
}}

## Example 3 — Drug with brand name
Input: text="Paracetamol 500mg" type="THUỐC"
Candidates: ["161", "198467", "44", "315677"]
Context: bệnh nhân sốt cao, dùng thuốc hạ sốt
Output:
{{
  "text": "Paracetamol 500mg",
  "type": "THUỐC",
  "ranked_candidates": [
    {{"code": "161", "score": 10, "reason": "Acetaminophen 500mg oral, chính xác"}},
    {{"code": "198467", "score": 8, "reason": "Acetaminophen generic, đúng thuốc"}},
    {{"code": "315677", "score": 3, "reason": "APAP 500mg, same drug different form"}},
    {{"code": "44", "score": 1, "reason": "Sai - Mesna, không phải paracetamol"}}
  ]
}}

## Example 4 — Drug-class generic (corticoid) → empty
Input: text="corticoid liều cao kéo dài" type="THUỐC"
Candidates: ["D07AA02", "H02AB06", "QD07AC"]
Output:
{{
  "text": "corticoid liều cao kéo dài",
  "type": "THUỐC",
  "ranked_candidates": []  // drug-class generic, không match specific
}}

## Example 5 — Disease with wrong concept match (polycystic kidney)
Input: text="bệnh di truyền lặn liên kết với nhiễm sắc thể X" type="CHẨN_ĐOÁN"
Candidates: ["E75.21", "Q61.1", "D55.0"]
Context: bệnh nhân nữ, family có bệnh di truyền
Output:
{{
  "text": "bệnh di truyền lặn liên kết với nhiễm sắc thể X",
  "type": "CHẨN_ĐOÁN",
  "ranked_candidates": [
    {{"code": "E75.21", "score": 5, "reason": "Fabry disease - X-linked, partial match"}},
    {{"code": "D55.0", "score": 4, "reason": "G6PD X-linked, cùng pattern di truyền"}},
    {{"code": "Q61.1", "score": 1, "reason": "Sai - polycystic kidney autosomal recessive"}}
  ]
}}

## Example 6 — Drug with typo (trimetazidin → trimetazidine)
Input: text="trimetazidin" type="THUỐC"
Candidates: ["10826"]
Output:
{{
  "text": "trimetazidin",
  "type": "THUỐC",
  "ranked_candidates": [
    {{"code": "10826", "score": 9, "reason": "Trimetazidine generic, typo 1 char"}}
  ]
}}
"""


def build_rerank_user_prompt(
    input_text: str,
    entities_with_candidates: list[dict],
    batch_size: int = 15,
    top_k: int = 5,
) -> list[str]:
    """R38 (2026-07-23): Build LLM ReRank user prompts (batched).

    Args:
        input_text: full clinical note (provides context for LLM scoring).
        entities_with_candidates: list of {{text, type, candidates}} for CHẨN_ĐOÁN + THUỐC.
            candidates: list of code strings (ICD-10 or RxNorm rxcui).
        batch_size: max entities per LLM call (default 15 — re-rank prompts are long).
        top_k: number of top candidates to keep per entity (default 5).

    Returns:
        List of user prompt strings (one per batch). Caller runs LLM on each, parses, merges.
    """
    if not entities_with_candidates:
        return []

    batches: list[str] = []
    for i in range(0, len(entities_with_candidates), batch_size):
        batch = entities_with_candidates[i:i + batch_size]

        lines = []
        for j, e in enumerate(batch):
            cand = e.get("candidates", [])
            if cand:
                cand_str = ", ".join(str(c) for c in cand)
            else:
                cand_str = "(no candidates — score all as 1)"
            lines.append(
                f"{{j+1}}. text=\"{{e.get('text','')}}\" "
                f"type=\"{{e.get('type','')}}\" "
                f"candidates=[{{cand_str}}]"
            )

        entities_str = "\n".join(lines)

        prompt = (
            f"# Clinical note (full context)\n"
            f"{{input_text}}\n\n"
            f"---\n\n"
            f"# Entities cần re-rank (top-{{top_k}} mỗi entity)\n"
            f"Batch {{i // batch_size + 1}}: {{len(batch)}} entities\n\n"
            f"{{entities_str}}\n\n"
            f"# Output\n"
            f"Trả JSON array (MỖI element cho 1 entity theo thứ tự trên). "
            f"Mỗi entity có `ranked_candidates` sort theo score giảm dần, "
            f"chỉ giữ score >= 3, tối đa top-{{top_k}}."
        )
        batches.append(prompt)
    return batches

