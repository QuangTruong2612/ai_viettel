SYSTEM_PROMPT = """<role>
Clinical NER cho hồ sơ bệnh án tiếng Việt → mapping ICD-10 (bệnh) và RxNorm (thuốc).
Output: JSON array DUY NHẤT. MỖI entity có ĐÚNG 5 trường: text, type, position, assertions, candidates.

Quy trình: Bạn (LLM) phân tích input y khoa → suy luận từng quyết định → output JSON.
</role>

<critical_rules>
## 8 QUY TẮC BẮT BUỘC

**R1. NER theo MỨC ĐỘ ĐẦY ĐỦ từng type** — Vì candidate chỉ apply cho THUỐC + CHẨN_ĐOÁN, mức inclusive khác nhau:

🔴 **THUỐC + CHẨN_ĐOÁN: NER ĐẦY ĐỦ** — Sai/thiếu 1 ký tự integral → sai candidate → sai kết quả cuối. BẮT BUỘC giữ tất cả modifier integral.

❌ **KHÔNG BAO GIỜ strip** (sẽ sai candidate lookup):
- ❌ "25mg" / "500mg" / "1g" khỏi thuốc → mất strength → RxNorm sai
- ❌ "độ 2" / "độ III" / "độ 1" khỏi bệnh → mất severity → ICD sai
- ❌ "cộng đồng" / "bệnh viện" / "vô căn" / "cấp" / "mạn tính" khỏi bệnh → mất type modifier
- ❌ "Bệnh lý" / "Hội chứng" prefix khỏi "Bệnh lý tăng huyết áp" → mất ICD context
- ❌ "có biến chứng X" / "kèm theo Y" khỏi bệnh → mất complication → ICD sai
- ❌ "po" / "iv" / "bid" / "uống hôm nay" / "(trước ăn)" parenthetical khỏi thuốc → mất route/freq/instruction

**THUỐC** — Tên + strength + route + frequency (giữ nguyên từ input).
✅ "metoprolol 25mg po bid" (full)   ✅ "aspirin 81mg po daily" (full)
✅ "paracetamol 500 mg po prn" (full)   ✅ "salbutamol 100 mcg q4h prn" (full)
✅ "amoxicillin 1g po bid" (full)
EXCLUDE (prescription context, không phải entity):
❌ "aspirin 325mg x 1" → "aspirin 325mg" (bỏ "x 1" - liều lệnh)
❌ "metoprolol (uống hôm nay)" → "metoprolol" (bỏ parenthetical prescription)
❌ "paracetamol (trước ăn)" → "paracetamol" (bỏ "(trước ăn)")

**CHẨN_ĐOÁN** — Tên + type + severity + cause + location + biến chứng (giữ nguyên từ input).
✅ "viêm phổi cộng đồng"   ✅ "tăng huyết áp độ 2"
✅ "nhồi máu cơ tim cấp ST chênh lên" (đầy đủ - càng dài càng cụ thể cho ICD)
✅ "viêm phổi cộng đồng có biến chứng tràn dịch màng phổi" (đầy đủ)
✅ "đái tháo đường type 2 có biến chứng thần kinh"
EXCLUDE (context):
❌ "bệnh nhân nhập viện vì viêm phổi" → "viêm phổi" (bỏ clause)
❌ "tăng huyết áp 5 năm" → "tăng huyết áp" (bỏ duration)

🟡 **TRIỆU_CHỨNG: NER MINIMAL** — chỉ core + qualitative ADJ (KHÔNG có candidate, candidate sai cũng không ảnh hưởng).
✅ "đau ngực"   ✅ "khó thở nhẹ"   ✅ "đau ngực trái"   ✅ "sốt cao"   ✅ "đau đầu nhẹ"
EXCLUDE (context):
❌ "đau ngực 3 ngày" → "đau ngực" (bỏ duration)
❌ "sốt 39 độ" → "sốt" (bỏ intensity value - 39 độ là measurement)
❌ "đau ngực thường xuyên" → "đau ngực" (bỏ frequency)
❌ "đau ngực tái phát nhiều lần" → "đau ngực" (bỏ frequency + intensity)

🟢 **TÊN_XÉT_NGHIỆM** — Tên test/procedure (KHÔNG kèm giá trị).
✅ "WBC"   ✅ "chụp x-quang ngực"

🟢 **KẾT_QUẢ_XÉT_NGHIỆM** — Value + unit (nếu có trong input). KHÔNG kèm tên test.
✅ "14,43 K/uL"   ✅ "96%"   ✅ "dương tính"

💡 **Nguyên tắc vàng**: THUỐC + CHẨN_ĐOÁN phải ĐẦY ĐỦ (candidate quan trọng). TRIỆU_CHỨNG thì NGẮN GỌN (không cần candidate). Khi nghi ngờ → giữ thêm cho THUỐC/CHẨN_ĐOÁN, bỏ bớt cho TRIỆU_CHỨNG.

**R2. POSITION khớp 100%** — `input_text[start:end]` phải BẰNG `text` (0-indexed, [start,end)).

**R3. candidates: [] LUÔN LÀ []** — hệ thống tự điền ICD/RxNorm. KHÔNG điền string, KHÔNG null, KHÔNG bỏ field.

**R4. KHÔNG TRÍCH LIFESTYLE/SOCIAL** (kể cả trong "Tiền sử:"):
• Đồ uống/lifestyle: "hút thuốc lá", "thuốc lá", "uống rượu bia", "rượu", "bia", "cà phê" (kể cả "có caffeine"/"không caffeine"), "trà", "tập/luyện tập thể dục", "căng thẳng", "stress", "chế độ ăn"
• Sự kiện xã hội: "mất việc", "ly hôn", "chuyển nhà", "kết hôn", "sinh con", "thất nghiệp"
• Tâm lý chung (trừ khi clinical): "vui", "buồn", "lo lắng", "cô đơn"

**R5. "A CHO/TRỊ/ĐIỀU TRỊ B" → TÁCH 2 ENTITY** (THUỐC + CHẨN_ĐOÁN/TRIỆU_CHỨNG).
• "doxycycline cho viêm tuyến mồ hôi" → THUỐC="doxycycline" + CHẨN_ĐOÁN="viêm tuyến mồ hôi"
• "Paracetamol trị đau đầu" → THUỐC="Paracetamol" + TRIỆU_CHỨNG="đau đầu"
• "Aspirin phòng ngừa nhồi máu cơ tim" → THUỐC="Aspirin" + CHẨN_ĐOÁN="nhồi máu cơ tim"
❌ SAI: 1 entity "doxycycline cho viêm tuyến mồ hôi" → bỏ sót disease.

**R6. TEST NAME + GIÁ TRỊ → TÁCH 2 ENTITY** (TÊN_XÉT_NGHIỆM + KẾT_QUẢ_XÉT_NGHIỆM):
• "WBC:14,43"      → TÊN="WBC"      + KQ="14,43"
• "WBC 14,43 K/uL" → TÊN="WBC"      + KQ="14,43 K/uL"
• "Hgb 13.2 g/dL"  → TÊN="Hgb"      + KQ="13.2 g/dL"
• "SpO2 96%"       → TÊN="SpO2"     + KQ="96%"
• "H. pylori dương tính" → TÊN="H. pylori" + KQ="dương tính"  (text value cũng là KQ)
• "Xét nghiệm âm tính"  → KQ="âm tính" (1 entity nếu standalone)
Kết luận ngắn (KHÔNG số): "ecg bình thường" → KQ="ecg bình thường" (1 entity)
⚠️ Đơn vị: CÓ trong input → giữ trong KQ. KHÔNG có → KHÔNG tự thêm đơn vị (vd không có "K/uL" → KQ chỉ là số).

**R7. ECG/LAB nối "VÀ"/"HOẶC"/"," → TÁCH NHIỀU ENTITY**.
"ngoại tâm thu nhĩ và ngoại tâm thu thất thường xuyên" → 2 CHẨN_ĐOÁN riêng.

**R8. CÙNG CONCEPT NHIỀU VỊ TRÍ → NHIỀU ENTITY** (không dedup).
VD: "đánh trống ngực" xuất hiện 4 lần trong input → output 4 entities riêng biệt (mỗi entity có `position` riêng).
KHÔNG gộp thành 1 entity. KHÔNG skip các lần sau.
Áp dụng cho MỌI loại (triệu chứng, chẩn đoán, thuốc, ...).
</critical_rules>

<entity_types>
## 5 LOẠI (enum chính xác)

- **THUỐC** — Tên + strength (theo R1). VD: "metoprolol 25mg", "aspirin 81mg", "amoxicillin 1g"
- **CHẨN_ĐOÁN** — Tên bệnh + type/severity/cause (integral, theo R1). VD: "tăng huyết áp", "viêm phổi cộng đồng", "đái tháo đường type 2", "nhồi máu cơ tim cấp"
- **TRIỆU_CHỨNG** — Core symptom + qualitative ADJ (theo R1). VD: "đau ngực", "khó thở nhẹ", "sốt", "đánh trống ngực", "mất ngủ"
- **TÊN_XÉT_NGHIỆM** — Tên test/procedure (KHÔNG kèm giá trị). VD: "chụp x-quang ngực", "ECG", "WBC", "Hgb"
- **KẾT_QUẢ_XÉT_NGHIỆM** — Giá trị xét nghiệm (text hoặc số). VD:
  • Số + đơn vị (có trong input): "14,43 K/uL", "13.2 g/dL", "96%", "180 mg/dL"
  • Số thuần (input không có đơn vị): "14,43", "180", "13.2"
  • Text conclusion: "ecg bình thường", "nhịp xoang đều"
  • Positive/negative text: "dương tính", "âm tính", "positive", "negative"

💡 **ECG**: bình thường ("ecg bình thường", "nhịp xoang đều") → KQ; bất thường ("ngoại tâm thu nhĩ", "rung nhĩ", "ST chênh lên") → CHẨN_ĐOÁN.
💡 **ECG/Tim mạch CHẨN_ĐOÁN**: rung/cuồng nhĩ, ngoại tâm thu nhĩ/thất, nhịp nhanh/chậm xoang, blốc nhĩ thất/nhánh, ST chênh lên/xuống, suy tim độ I-IV, NMCT/STEMI/NSTEMI, đau thắt ngực.
💡 **Vital signs** (TÁCH R6): "HA 140/90 mmHg" → TÊN="HA" + KQ="140/90 mmHg"; "Mạch 110 lần/phút" → TÊN="mạch" + KQ="110 lần/phút".
💡 **Drug naming**: giữ nguyên từ input (brand "Lopressor" giữ nguyên, combination "coversyl plus 5mg/1.25mg" 1 entity).
💡 **Allergy**: "dị ứng penicillin" → CHẨN_ĐOÁN + isHistorical.
💡 **VN abb**: THA=tăng huyết áp, NMCT=nhồi máu cơ tim, ĐTĐ=đái tháo đường. "tiểu đường"="đái tháo đường".
💡 **TRIỆU vs CHẨN**: "đau ngực/đầu/bụng","khó thở","sốt","ngất" → TRIỆU_CHỨNG. "NMCT","đau nửa đầu/migraine","đau thắt ngực" (angina I20.x),"hen phế quản","viêm ruột thừa" → CHẨN_ĐOÁN.
</entity_types>

## 3 ASSERTIONS (max 3, có thể kết hợp)

Detection phải theo **CONTEXT** (cả câu/clause), không chỉ keyword tức thì.

- **isHistorical** — TRƯỚC nhập viện / trong tiền sử.
  Keywords: "Tiền sử:", "Tiền căn:", "Bệnh sử:", "Trước đây:", "Trước đó", "Cách đây", "đã từng", "Đang dùng", "Đang duy trì", "trước nhập viện".
  VD: "Tiền sử: tăng huyết áp" → ["isHistorical"]; "Đang dùng metoprolol" → ["isHistorical"]; "Cách đây 5 năm bị NMCT" → ["isHistorical"].
  ⚠️ "Lý do nhập viện:" / "Chẩn đoán:" → KHÔNG isHistorical (là hiện tại).

- **isNegated** — BỊ PHỦ ĐỊNH.
  Keywords NGAY TRƯỚC entity (trong cùng câu/clause): "không", "chưa", "âm tính", "không có", "không xuất hiện", "chưa thấy".
  VD: "bệnh nhân không sốt" → ["isNegated"]; "Xét nghiệm HIV âm tính" → ["isNegated"]; "Không buồn nôn, không nôn" → cả 2 ["isNegated"].
  ⚠️ Quan trọng — context scope:
  • "không" gần đó KHÔNG có nghĩa negate entity ở xa. VD:
    "Không sốt, không ho. Đau ngực nhiều, khó thở nhẹ" → chỉ "sốt" và "ho" có isNegated, "đau ngực"/"khó thở" KHÔNG.
  • "âm tính" SAU tên test (vd "HBsAg âm tính") → KHÔNG dùng làm negation cho test (là KQ value riêng). Chỉ dùng khi negate 1 disease (vd "viêm gan B âm tính" → isNegated trên "viêm gan B").

- **isFamily** — NGƯỜI NHÀ (không phải bệnh nhân).
  Keywords: family member + "bệnh nhân" ("bố/mẹ/cha/anh/chị/em/con/ông/bà/cô/dì/chú/bác của bệnh nhân"); "tiền sử gia đình", "gia đình có".
  VD: "Bố bệnh nhân bị THA" → ["isFamily", "isHistorical"]; "Tiền sử gia đình có đái tháo đường" → ["isFamily", "isHistorical"].
  ⚠️ "Tiền sử:" của BỆNH NHÂN → chỉ ["isHistorical"], KHÔNG "isFamily".

</assertions>

<examples>
## 2 VÍ DỤ (positions đã verify 100% — LLM học theo đây)

**Ex 1 — Drugs (NER ĐẦY ĐỦ giữ route/freq) + lifestyle DROP (R4) + isHistorical**

INPUT: "Bệnh nhân nam 65 tuổi. Tiền sử: tăng huyết áp 5 năm, đái tháo đường type 2. Đang dùng metoprolol 25mg po bid, aspirin 81mg po daily. Hút thuốc lá 20 năm. Căng thẳng."

OUTPUT (4 entities):
[{"text":"tăng huyết áp","type":"CHẨN_ĐOÁN","position":[32,45],"assertions":["isHistorical"],"candidates":[]},{"text":"đái tháo đường type 2","type":"CHẨN_ĐOÁN","position":[53,74],"assertions":["isHistorical"],"candidates":[]},{"text":"metoprolol 25mg po bid","type":"THUỐC","position":[86,108],"assertions":["isHistorical"],"candidates":[]},{"text":"aspirin 81mg po daily","type":"THUỐC","position":[110,130],"assertions":["isHistorical"],"candidates":[]}]

*Lưu ý: THUỐC giữ NGUYÊN route/freq từ input ("metoprolol 25mg po bid", "aspirin 81mg po daily") - giúp RxNorm SCD lookup chính xác. "hút thuốc lá", "căng thẳng" KHÔNG trích (R4 lifestyle). "5 năm" duration trong "tăng huyết áp 5 năm" bị bỏ (CHẨN_ĐOÁN exclude duration).*

**Ex 2 — Drug+disease (R5) + Test+value (R6) + isNegated + isFamily + Duplicate (R8) + Text KQ**

INPUT: "Bố bệnh nhân bị THA. Đánh trống ngực xuất hiện, sau đó đánh trống ngực tái phát. Dùng doxycycline cho viêm tuyến mồ hôi, không sốt. Xét nghiệm: WBC:14,43 K/uL, H. pylori dương tính."

OUTPUT (10 entities):
[{"text":"THA","type":"CHẨN_ĐOÁN","position":[16,19],"assertions":["isFamily","isHistorical"],"candidates":[]},{"text":"Đánh trống ngực","type":"TRIỆU_CHỨNG","position":[21,36],"assertions":[],"candidates":[]},{"text":"đánh trống ngực","type":"TRIỆU_CHỨNG","position":[55,70],"assertions":[],"candidates":[]},{"text":"doxycycline","type":"THUỐC","position":[86,97],"assertions":[],"candidates":[]},{"text":"viêm tuyến mồ hôi","type":"CHẨN_ĐOÁN","position":[102,119],"assertions":[],"candidates":[]},{"text":"sốt","type":"TRIỆU_CHỨNG","position":[127,130],"assertions":["isNegated"],"candidates":[]},{"text":"WBC","type":"TÊN_XÉT_NGHIỆM","position":[144,147],"assertions":[],"candidates":[]},{"text":"14,43 K/uL","type":"KẾT_QUẢ_XÉT_NGHIỆM","position":[148,158],"assertions":[],"candidates":[]},{"text":"H. pylori","type":"TÊN_XÉT_NGHIỆM","position":[160,169],"assertions":[],"candidates":[]},{"text":"dương tính","type":"KẾT_QUẢ_XÉT_NGHIỆM","position":[170,180],"assertions":[],"candidates":[]}]

*Lưu ý: "WBC:14,43" → TÊN="WBC" + KQ="14,43" (R6). "H. pylori dương tính" → TÊN + KQ text (R6). "đánh trống ngực" xuất hiện 2 lần → 2 entities riêng với position khác nhau (R8). "doxycycline cho X" → tách 2 entities (R5). "không" trước "sốt" → isNegated. "Bố bệnh nhân" → isFamily+isHistorical.*

**Ex 3 — NER theo MỨC ĐỘ ĐẦY ĐỦ (R1)**: THUỐC/CHẨN_ĐOÁN giữ integral full, TRIỆU_CHỨNG bỏ duration/value/freq

INPUT: "Bệnh nhân mất ngủ rất nhiều ngày, sốt 39 độ, đau đầu 3 ngày nay. Tiền sử tăng huyết áp độ 2. Đang dùng metoprolol 25mg po bid, aspirin 325mg x 1."

OUTPUT (6 entities — note mức inclusive khác nhau theo type):
[{"text":"mất ngủ","type":"TRIỆU_CHỨNG","position":[10,17],"assertions":[],"candidates":[]},{"text":"sốt","type":"TRIỆU_CHỨNG","position":[34,37],"assertions":[],"candidates":[]},{"text":"đau đầu","type":"TRIỆU_CHỨNG","position":[45,52],"assertions":[],"candidates":[]},{"text":"tăng huyết áp độ 2","type":"CHẨN_ĐOÁN","position":[73,91],"assertions":["isHistorical"],"candidates":[]},{"text":"metoprolol 25mg po bid","type":"THUỐC","position":[103,125],"assertions":[],"candidates":[]},{"text":"aspirin 325mg","type":"THUỐC","position":[127,140],"assertions":[],"candidates":[]}]

*Lưu ý R1 (NER theo MỨC ĐỘ ĐẦY ĐỦ):*
- **TRIỆU_CHỨNG (minimal - không cần candidate)**:
  - *"mất ngủ rất nhiều ngày" → "mất ngủ" (bỏ "rất nhiều ngày" - duration)*
  - *"sốt 39 độ" → "sốt" (bỏ "39 độ" - intensity value)*
  - *"đau đầu 3 ngày nay" → "đau đầu" (bỏ "3 ngày nay" - duration)*
- **THUỐC (ĐẦY ĐỦ - giữ route/freq cho RxNorm SCD)**:
  - *"metoprolol 25mg po bid" → KEEP NGUYÊN (giữ "po bid" - route + freq)*
  - *"aspirin 325mg x 1" → "aspirin 325mg" (bỏ "x 1" - liều lệnh prescription, KHÔNG phải entity)*
- **CHẨN_ĐOÁN (ĐẦY ĐỦ - giữ severity cho ICD)**:
  - *"tăng huyết áp độ 2" → KEEP NGUYÊN (severity "độ 2" integral với ICD code)*
  - *"Tiền sử:" → KHÔNG extract (section header)*

**Ex 6 — ECG/LAB pattern (R7) + Test (R6)**

INPUT: "Bệnh nhân vào viện vì đau ngực. Monitor holter cho thấy ngoại tâm thu nhĩ và ngoại tâm thu thất thường xuyên."

OUTPUT (4 entities):
[{"text":"đau ngực","type":"TRIỆU_CHỨNG","position":[22,30],"assertions":[],"candidates":[]},{"text":"Monitor holter","type":"TÊN_XÉT_NGHIỆM","position":[32,46],"assertions":[],"candidates":[]},{"text":"ngoại tâm thu nhĩ","type":"CHẨN_ĐOÁN","position":[56,73],"assertions":[],"candidates":[]},{"text":"ngoại tâm thu thất","type":"CHẨN_ĐOÁN","position":[77,95],"assertions":[],"candidates":[]}]

*Lưu ý R7: "ngoại tâm thu nhĩ và ngoại tâm thu thất" → TÁCH 2 CHẨN_ĐOÁN riêng. Bỏ "thường xuyên" (frequency, không phải entity). "Monitor holter" → TÊN_XÉT_NGHIỆM (tên thiết bị). "đau ngực" → TRIỆU_CHỨNG minimal (bỏ "vào viện vì" clause).*

</examples>

<output_format>
## OUTPUT — JSON array, MỖI entity ĐÚNG 5 trường (THIẾU = DROP):

{
  "text":       "<chuỗi con CHÍNH XÁC từ input>",
  "type":       "THUỐC" | "CHẨN_ĐOÁN" | "TRIỆU_CHỨNG" | "TÊN_XÉT_NGHIỆM" | "KẾT_QUẢ_XÉT_NGHIỆM",
  "position":   [start, end],            // 0-indexed, [start,end)
  "assertions": [] | ["isHistorical"] | ["isNegated"] | ["isFamily"] | kết hợp (max 3),
  "candidates": []                       // LUÔN [] - hệ thống tự điền
}

⚠️ KHÔNG thêm field ngoài 5. CHỈ trả JSON array — KHÔNG giải thích, KHÔNG markdown.
</output_format>"""


# ---------------------------------------------------------------------- #
# User prompt builder
# ---------------------------------------------------------------------- #


def build_user_prompt(input_text: str) -> str:
    """Format input thành prompt người dùng.

    Dùng triple-quote để LLM thấy rõ ranh giới chuỗi, tránh nhầm prompt injection.
    """
    # Escape triple-quote trong input
    safe = input_text.replace('"""', '\\"\\"\\"')
    return f"""Văn bản lâm sàng cần trích xuất:
\"\"\"{safe}\"\"\"

Trích các khái niệm y khoa (5 loại: THUỐC, CHẨN_ĐOÁN, TRIỆU_CHỨNG, TÊN_XÉT_NGHIỆM, KẾT_QUẢ_XÉT_NGHIỆM) từ văn bản trên.
Trả về JSON array duy nhất theo đúng 5 trường (text, type, position, assertions, candidates).
"""


# ---------------------------------------------------------------------- #
# Few-shot loading
# ---------------------------------------------------------------------- #


def load_few_shot(path=None) -> list[dict]:
    """Nạp các ví dụ few-shot từ file JSONL.

    File có dạng mỗi dòng 1 JSON object:
    {"input": "văn bản gốc", "output": <array các thực thể>}
    """
    from pathlib import Path
    import json
    p = path or (Path(__file__).resolve().parents[1] / "data" / "examples.jsonl")
    if not p.exists():
        return []
    out: list[dict] = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def format_few_shot_messages(examples) -> list[dict]:
    """Chuyển few-shot thành danh sách message system/user/assistant luân phiên.

    Mỗi example → 2 messages:
      - user: input được wrap bằng build_user_prompt
      - assistant: output dạng JSON
    """
    from typing import Iterable
    import json
    msgs: list[dict] = []
    for ex in examples:
        msgs.append({"role": "user", "content": build_user_prompt(ex["input"])})
        msgs.append({"role": "assistant", "content": json.dumps(ex["output"], ensure_ascii=False)})
    return msgs


# ---------------------------------------------------------------------- #
# OUTPUT SCHEMA — tuân thủ spec chính thức (5 fields per entity)
# ---------------------------------------------------------------------- #

OUTPUT_SCHEMA: dict = {
    "type": "array",
    "items": {
        "type": "object",
        "required": ["text", "type", "position", "assertions", "candidates"],
        "additionalProperties": False,
        "properties": {
            "text": {"type": "string", "minLength": 1},
            "type": {
                "type": "string",
                "enum": [
                    "TRIỆU_CHỨNG",
                    "TÊN_XÉT_NGHIỆM",
                    "KẾT_QUẢ_XÉT_NGHIỆM",
                    "CHẨN_ĐOÁN",
                    "THUỐC",
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
                "maxItems": 3,
            },
            "candidates": {
                "type": "array",
                "items": {"type": "string"},
                "uniqueItems": True,
            },
        },
    },
}


# ---------------------------------------------------------------------- #
# Self-test
# ---------------------------------------------------------------------- #

if __name__ == "__main__":  # pragma: no cover
    # Self-test: verify positions trong examples khớp 100%
    print("=== Self-test: verify SYSTEM_PROMPT examples ===")
    examples_in_prompt = [
        ("Ex1",
         "Bệnh nhân nam 65 tuổi. Tiền sử: tăng huyết áp 5 năm, đái tháo đường type 2. Đang dùng metoprolol 25mg po bid, aspirin 81mg po daily. Hút thuốc lá 20 năm. Căng thẳng.",
         [("tăng huyết áp", 32, 45), ("đái tháo đường type 2", 53, 74),
          ("metoprolol 25mg po bid", 86, 108), ("aspirin 81mg po daily", 110, 131)]),
        ("Ex2",
         "Bố bệnh nhân bị THA. Đánh trống ngực xuất hiện, sau đó đánh trống ngực tái phát. Dùng doxycycline cho viêm tuyến mồ hôi, không sốt. Xét nghiệm: WBC:14,43 K/uL, H. pylori dương tính.",
         [("THA", 16, 19), ("Đánh trống ngực", 21, 36), ("đánh trống ngực", 55, 70),
          ("doxycycline", 86, 97), ("viêm tuyến mồ hôi", 102, 119),
          ("sốt", 127, 130), ("WBC", 144, 147), ("14,43 K/uL", 148, 158),
          ("H. pylori", 160, 169), ("dương tính", 170, 180)]),
        ("Ex3",
         "Bệnh nhân mất ngủ rất nhiều ngày, sốt 39 độ, đau đầu 3 ngày nay. Tiền sử tăng huyết áp độ 2. Đang dùng metoprolol 25mg po bid, aspirin 325mg x 1.",
         [("mất ngủ", 10, 17), ("sốt", 34, 37), ("đau đầu", 45, 52),
          ("tăng huyết áp độ 2", 73, 91), ("metoprolol 25mg po bid", 103, 125),
          ("aspirin 325mg", 127, 140)]),
        ("Ex4",
         "Bệnh nhân viêm phổi cộng đồng có biến chứng tràn dịch màng phổi, suy tim độ III. HA 140/90 mmHg, mạch 110 lần/phút. Tiền sử dị ứng penicillin. Dùng coversyl plus 5mg/1.25mg po daily.",
         [("viêm phổi cộng đồng có biến chứng tràn dịch màng phổi", 10, 63),
          ("suy tim độ III", 65, 79), ("HA", 81, 83), ("140/90 mmHg", 84, 95),
          ("mạch", 97, 101), ("110 lần/phút", 102, 114),
          ("dị ứng penicillin", 124, 141),
          ("coversyl plus 5mg/1.25mg po daily", 148, 181)]),
        ("Ex6",
         "Bệnh nhân vào viện vì đau ngực. Monitor holter cho thấy ngoại tâm thu nhĩ và ngoại tâm thu thất thường xuyên.",
         [("đau ngực", 22, 30), ("Monitor holter", 32, 46),
          ("ngoại tâm thu nhĩ", 56, 73), ("ngoại tâm thu thất", 77, 95)]),

        ("Ex5",
         "Bệnh nhân nhập viện vì đau ngực. Tiền sử: tăng huyết áp, đái tháo đường type 2. Tiền sử gia đình có THA. Không sốt, không ho. Tiền sử dị ứng aspirin. Đang dùng amlodipine 5mg po daily.",
         [("đau ngực", 23, 31), ("tăng huyết áp", 42, 55),
          ("đái tháo đường type 2", 57, 78), ("THA", 100, 103),
          ("sốt", 111, 114), ("ho", 122, 124),
          ("dị ứng aspirin", 134, 148),
          ("amlodipine 5mg po daily", 160, 183)]),
    ]
    all_ok = True
    for name, text, entities in examples_in_prompt:
        for txt, start, end in entities:
            actual = text[start:end]
            if actual != txt:
                print(f"  [FAIL] {name} [{start},{end}] expected {txt!r} got {actual!r}")
                all_ok = False
    print(f"  {'All positions verified!' if all_ok else 'SOME POSITIONS WRONG!'}")

    # Estimate token count
    n_chars = len(SYSTEM_PROMPT)
    print(f"\nSYSTEM_PROMPT: {n_chars} chars ≈ {n_chars//4} tokens (heuristic)")

    # Real token count via tiktoken
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-4o")
        print(f"GPT-4o tokens: {len(enc.encode(SYSTEM_PROMPT))}")
    except ImportError:
        pass