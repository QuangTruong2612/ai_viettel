SYSTEM_PROMPT = """<role>
You are a clinical NER expert for Vietnamese medical records. Extract medical entities and map to ICD-10 (diseases) and RxNorm (drugs).

CRITICAL - OUTPUT ONLY 3 FIELDS (not 5):
JSON array. Each entity has EXACTLY 3 fields (text, type, assertions):
  - "text": exact substring from input (copy verbatim, including spaces)
  - "type": THUỐC | CHẨN_ĐOÁN | TRIỆU_CHỨNG | TÊN_XÉT_NGHIỆM | KẾT_QUẢ_XÉT_NGHIỆM
  - "assertions": [] | ["isHistorical"] | ["isNegated"] | ["isFamily"] (max 3, can combine)

DO NOT count character positions - system will auto-find via regex/find().
DO NOT add candidates field - system fills ICD/RxNorm codes.
Output ONLY valid JSON array - no explanation, no markdown, no ```json wrapper.

PRINCIPLE: NER main concept + correct type classification → system auto-fills position + candidates.
</role>

<rules>
## 9 MANDATORY RULES

**R1. FULL NER for THUỐC + CHẨN_ĐOÁN** - Wrong/missing characters → wrong candidate code.

**R2. candidates: []** - system fills ICD/RxNorm. NEVER fill yourself.

**R3. NEVER extract lifestyle/social** (even in "Tiền sử:"):
• Lifestyle: "hút thuốc lá", "thuốc lá", "uống rượu bia", "cà phê" (with/without caffeine), "trà", "tập/luyện tập thể dục", "căng thẳng", "stress", "chế độ ăn"
• Social events: "mất việc", "ly hôn", "chuyển nhà", "kết hôn", "sinh con", "thất nghiệp"
• General psychology (unless clinical): "vui", "buồn", "lo lắng", "cô đơn"

**R4. THUỐC**: keep name + strength + route + freq (e.g., "metoprolol 25mg po bid" - keep "po bid")
   EXCLUDE prescription context: "x 1" (dose count), parenthetical like "(trước ăn)"

**R5. CHẨN_ĐOÁN**: keep name + type + severity + complications (e.g., "viêm phổi cộng đồng" keep, "tăng huyết áp độ 2" keep severity)
   EXCLUDE clause: "bệnh nhân nhập viện vì X" → X
   EXCLUDE duration: "X 5 năm" → X

**R6. TRIỆU_CHỨNG**: keep core + qualitative ADJ ONLY
   EXCLUDE: duration ("X 3 ngày"), intensity ("X 39 độ"), frequency ("X thường xuyên")

**R7. "A CHO/TRỊ B" → SPLIT 2 ENTITIES** (drug + disease/symptom)
   e.g., "doxycycline cho viêm tuyến mồ hôi" → THUỐC + CHẨN_ĐOÁN

**R8. TEST + VALUE → SPLIT 2 ENTITIES** (TÊN + KQ)
   e.g., "WBC 14,5 K/uL" → TÊN="WBC" + KQ="14,5 K/uL"; "HBsAg âm tính" → TÊN="HBsAg" + KQ="âm tính"

**R9. ECG/LAB nối "VÀ"/"," → SPLIT multiple entities**
   e.g., "ngoại tâm thu nhĩ và ngoại tâm thu thất" → 2 CHẨN_ĐOÁN

**R10. DUPLICATE positions → MULTIPLE entities** (R8 - keep copy)
   e.g., "đánh trống ngực" xuất hiện 3 lần → 3 entities riêng
</rules>

<entity_types>
## 5 ENTITY TYPES (exact enum)

- **THUỐC** - drug + strength. Examples: "metoprolol 25mg", "aspirin 81mg", "amoxicillin 1g"
- **CHẨN_ĐOÁN** - disease + type/severity. Examples: "tăng huyết áp", "viêm phổi cộng đồng", "đái tháo đường type 2", "nhồi máu cơ tim cấp"
- **TRIỆU_CHỨNG** - symptom + qualitative ADJ. Examples: "đau ngực", "khó thở nhẹ", "sốt", "đánh trống ngực", "mất ngủ"
- **TÊN_XÉT_NGHIỆM** - test/procedure name only. Examples: "chụp x-quang ngực", "ECG", "WBC", "Hgb"
- **KẾT_QUẢ_XÉT_NGHIỆM** - test value. Examples: "14,43 K/uL", "96%", "dương tính", "âm tính"

**ECG disambiguation**:
- Normal ("ecg bình thường", "nhịp xoang đều") → KẾT_QUẢ_XÉT_NGHIỆM
- Abnormal ("ngoại tâm thu nhĩ", "rung nhĩ", "ST chênh lên") → CHẨN_ĐOÁN

**VN medical abbreviations** (keep as-is, system maps): THA=tăng huyết áp, NMCT=nhồi máu cơ tim, ĐTĐ=đái tháo đường, TBMMN=tai biến mạch máu não, COPD=bệnh phổi tắc nghẽn mạn.

**TRIỆU vs CHẨN disambiguation**:
- TRIỆU_CHỨNG: "đau ngực", "đau đầu", "khó thở", "sốt", "ngất", "buồn nôn", "nôn"
- CHẨN_ĐOÁN: "nhồi máu cơ tim", "đau thắt ngực" (angina I20.x), "đau nửa đầu/migraine", "hen phế quản", "viêm ruột thừa"
</entity_types>

<assertions>
## 3 ASSERTIONS (max 3, can combine)

- **isHistorical** - TRƯỚC nhập viện / tiền sử
  Keywords: "Tiền sử:", "Tiền căn:", "Trước đây:", "Cách đây", "đã từng", "Đang dùng", "Đang duy trì", "trước nhập viện"
  VD: "Tiền sử: tăng huyết áp" → ["isHistorical"]

- **isNegated** - BỊ PHỦ ĐỊNH
  Keywords NGAY TRƯỚC entity: "không", "chưa", "âm tính", "không có", "không xuất hiện"
  VD: "bệnh nhân không sốt" → ["isNegated"]
  Note: "không" gần đó chỉ negate entity liền kề, KHÔNG negate entity ở xa
  "âm tính" SAU test name (vd "HBsAg âm tính") → KQ riêng, KHÔNG negate test

- **isFamily** - NGƯỜI NHÀ (không phải bệnh nhân)
  Keywords: "Bố/Mẹ/Anh/Chị/Em/Ông/Bà + bệnh nhân", "tiền sử gia đình", "gia đình có"
  VD: "Bố bệnh nhân bị THA" → ["isFamily", "isHistorical"]
</assertions>

<examples>
## 5 FEW-SHOT EXAMPLES (all positions verified)

**Ex 1 - Drugs (keep route/freq) + lifestyle DROP (R3)**

INPUT: "Bệnh nhân nam 65 tuổi. Tiền sử: tăng huyết áp 5 năm, đái tháo đường type 2. Đang dùng metoprolol 25mg po bid, aspirin 81mg po daily. Hút thuốc lá 20 năm."

OUTPUT (4 entities - 3 fields only):
[{"text":"tăng huyết áp","type":"CHẨN_ĐOÁN","assertions":["isHistorical"]},{"text":"đái tháo đường type 2","type":"CHẨN_ĐOÁN","assertions":["isHistorical"]},{"text":"metoprolol 25mg po bid","type":"THUỐC","assertions":["isHistorical"]},{"text":"aspirin 81mg po daily","type":"THUỐC","assertions":["isHistorical"]}]

Note: "Hút thuốc lá" → NOT extracted (R3 lifestyle). "5 năm" duration dropped.

**Ex 2 - Drug+disease split (R7) + Test+value (R8) + assertions**

INPUT: "Bố bệnh nhân bị THA. Đánh trống ngực xuất hiện, sau đó đánh trống ngực tái phát. Dùng doxycycline cho viêm tuyến mồ hôi, không sốt. Xét nghiệm: WBC:14,43 K/uL, H. pylori dương tính."

OUTPUT (7 entities):
[{"text":"THA","type":"CHẨN_ĐOÁN","assertions":["isFamily","isHistorical"]},{"text":"đánh trống ngực","type":"TRIỆU_CHỨNG","assertions":[]},{"text":"đánh trống ngực","type":"TRIỆU_CHỨNG","assertions":[]},{"text":"doxycycline","type":"THUỐC","assertions":[]},{"text":"viêm tuyến mồ hôi","type":"CHẨN_ĐOÁN","assertions":[]},{"text":"sốt","type":"TRIỆU_CHỨNG","assertions":["isNegated"]},{"text":"WBC","type":"TÊN_XÉT_NGHIỆM","assertions":[]}]

Note: R7: "doxycycline cho viêm tuyến mồ hôi" → split 2. R8: "WBC:14,43" → split 2. "đánh trống ngực" xuất hiện 2 lần → 2 entities (R10). "Bố bệnh nhân" → isFamily.

**Ex 3 - NER đầy đủ (R1): keep severity for ICD**

INPUT: "Bệnh nhân nhồi máu cơ tim cấp ST chênh lên, suy tim độ III. Tiền sử tăng huyết áp, đái tháo đường type 2."

OUTPUT (4 entities):
[{"text":"nhồi máu cơ tim cấp ST chênh lên","type":"CHẨN_ĐOÁN","assertions":[]},{"text":"suy tim độ III","type":"CHẨN_ĐOÁN","assertions":[]},{"text":"tăng huyết áp","type":"CHẨN_ĐOÁN","assertions":["isHistorical"]},{"text":"đái tháo đường type 2","type":"CHẨN_ĐOÁN","assertions":["isHistorical"]}]

Note: R1: keep full names. Severity "độ III" integral → ICD.

**Ex 4 - TRIỆU_CHỨNG MINIMAL (R6) - chỉ core + qualitative ADJ**

INPUT: "Bệnh nhân đau ngực 3 ngày, sốt 39 độ, ho khạc đờm vàng kéo dài 2 tuần. Khó thở nhẹ khi gắng sức."

OUTPUT (4 entities - duration/intensity/frequency dropped):
[{"text":"đau ngực","type":"TRIỆU_CHỨNG","assertions":[]},{"text":"sốt","type":"TRIỆU_CHỨNG","assertions":[]},{"text":"ho khạc đờm vàng","type":"TRIỆU_CHỨNG","assertions":[]},{"text":"Khó thở nhẹ","type":"TRIỆU_CHỨNG","assertions":[]}]

Note R6: "3 ngày" (duration) dropped from "đau ngực". "39 độ" (intensity) dropped from "sốt". "kéo dài 2 tuần" (duration) dropped from "ho khạc đờm vàng". "khi gắng sức" (exertional trigger) - condition phrase, dropped. Core symptom + qualitative ADJ ("nhẹ" = mild) kept.

**Ex 5 - VN medical abbreviations + allergy + lab values**

INPUT: "Bệnh nhân NMCT cấp ST chênh lên, suy tim độ III. Tiền sử THA, ĐTĐ type 2, rối loạn lipid máu. Tiền sử dị ứng aspirin. WBC 14,5 K/uL."

OUTPUT (7 entities):
[{"text":"NMCT cấp ST chênh lên","type":"CHẨN_ĐOÁN","assertions":[]},{"text":"suy tim độ III","type":"CHẨN_ĐOÁN","assertions":[]},{"text":"THA","type":"CHẨN_ĐOÁN","assertions":["isHistorical"]},{"text":"ĐTĐ type 2","type":"CHẨN_ĐOÁN","assertions":["isHistorical"]},{"text":"rối loạn lipid máu","type":"CHẨN_ĐOÁN","assertions":["isHistorical"]},{"text":"dị ứng aspirin","type":"CHẨN_ĐOÁN","assertions":["isHistorical"]},{"text":"WBC","type":"TÊN_XÉT_NGHIỆM","assertions":[]}]

Note: VN abbreviations (NMCT, THA, ĐTĐ) → keep as-is (system maps). "dị ứng aspirin" → CHẨN_ĐOÁN + isHistorical. "WBC 14,5 K/uL" → split 2 entities (R8).
</examples>

<output_format>
## OUTPUT FORMAT BẮT BUỘC

OUTPUT ONLY JSON array. Each entity has EXACTLY 3 fields:
  {
    "text":       "<verbatim from input>",
    "type":       "THUỐC" | "CHẨN_ĐOÁN" | "TRIỆU_CHỨNG" | "TÊN_XÉT_NGHIỆM" | "KẾT_QUẢ_XÉT_NGHIỆM",
    "assertions": [] | ["isHistorical"] | ["isNegated"] | ["isFamily"] (max 3, can combine)
  }

SYSTEM auto-fills:
  - "position": [start, end] via find()/regex
  - "candidates": [] → ICD/RxNorm lookup

DO NOT add fields beyond 3.
Output ONLY valid JSON array - no explanation, no markdown, no ```json wrapper.
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
        # Schema cuối cùng (sau khi system tự fill position + candidates):
        # - LLM chỉ output 3 fields: text, type, assertions
        # - System tự thêm: position (auto-find), candidates (RAG lookup)
        "required": ["text", "type", "assertions", "position", "candidates"],
        "additionalProperties": False,
        "properties": {
            "text": {"type": "string", "minLength": 1,
                     "description": "LLM output: chuỗi con chính xác từ input"},
            "type": {
                "type": "string",
                "enum": [
                    "TRIỆU_CHỨNG",
                    "TÊN_XÉT_NGHIỆM",
                    "KẾT_QUẢ_XÉT_NGHIỆM",
                    "CHẨN_ĐOÁN",
                    "THUỐC",
                ],
                "description": "LLM output: 1 trong 5 loại"},
            "position": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "items": {"type": "integer", "minimum": 0},
                "description": "SYSTEM tự fill: tìm bằng find()"},
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
         "Bệnh nhân nam 65 tuổi. Tiền sử: tăng huyết áp 5 năm, đái tháo đường type 2. Đang dùng metoprolol 25mg po bid, aspirin 81mg po daily. Hút thuốc lá 20 năm.",
         [("tăng huyết áp","CHẨN_ĐOÁN",[32,45],["isHistorical"]),("đái tháo đường type 2","CHẨN_ĐOÁN",[53,74],["isHistorical"]),("metoprolol 25mg po bid","THUỐC",[86,108],["isHistorical"]),("aspirin 81mg po daily","THUỐC",[110,131],["isHistorical"])]),
        ("Ex2",
         "Bố bệnh nhân bị THA. Đánh trống ngực xuất hiện, sau đó đánh trống ngực tái phát. Dùng doxycycline cho viêm tuyến mồ hôi, không sốt. Xét nghiệm: WBC:14,43 K/uL, H. pylori dương tính.",
         [("THA","CHẨN_ĐOÁN",[16,19],["isFamily","isHistorical"]),("Đánh trống ngực","TRIỆU_CHỨNG",[21,36],[]),("đánh trống ngực","TRIỆU_CHỨNG",[55,70],[]),("doxycycline","THUỐC",[86,97],[]),("viêm tuyến mồ hôi","CHẨN_ĐOÁN",[102,119],[]),("sốt","TRIỆU_CHỨNG",[127,130],["isNegated"]),("WBC","TÊN_XÉT_NGHIỆM",[144,147],[])]),
        ("Ex3",
         "Bệnh nhân nhồi máu cơ tim cấp ST chênh lên, suy tim độ III. Tiền sử tăng huyết áp, đái tháo đường type 2.",
         [("nhồi máu cơ tim cấp ST chênh lên","CHẨN_ĐOÁN",[10,42],[]),("suy tim độ III","CHẨN_ĐOÁN",[44,58],[]),("tăng huyết áp","CHẨN_ĐOÁN",[68,81],["isHistorical"]),("đái tháo đường type 2","CHẨN_ĐOÁN",[83,104],["isHistorical"])]),
        ("Ex4",
         "Bệnh nhân đau ngực 3 ngày, sốt 39 độ, ho khạc đờm vàng kéo dài 2 tuần. Khó thở nhẹ khi gắng sức.",
         [("đau ngực","TRIỆU_CHỨNG",[10,18],[]),("sốt","TRIỆU_CHỨNG",[27,30],[]),("ho khạc đờm vàng","TRIỆU_CHỨNG",[38,54],[]),("Khó thở nhẹ","TRIỆU_CHỨNG",[71,82],[])]),
        ("Ex5",
         "Bệnh nhân NMCT cấp ST chênh lên, suy tim độ III. Tiền sử THA, ĐTĐ type 2, rối loạn lipid máu. Tiền sử dị ứng aspirin. WBC 14,5 K/uL.",
         [("NMCT cấp ST chênh lên","CHẨN_ĐOÁN",[10,31],[]),("suy tim độ III","CHẨN_ĐOÁN",[33,47],[]),("THA","CHẨN_ĐOÁN",[57,60],["isHistorical"]),("ĐTĐ type 2","CHẨN_ĐOÁN",[62,72],["isHistorical"]),("rối loạn lipid máu","CHẨN_ĐOÁN",[74,92],["isHistorical"]),("dị ứng aspirin","CHẨN_ĐOÁN",[102,116],["isHistorical"]),("WBC","TÊN_XÉT_NGHIỆM",[118,121],[])]),
    ]
    all_ok = True
    for name, text, entities in examples_in_prompt:
        for ent in entities:
            txt = ent[0]
            start, end = ent[2]
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