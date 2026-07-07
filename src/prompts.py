SYSTEM_PROMPT = """<role>
Clinical NER cho hồ sơ bệnh án tiếng Việt → mapping ICD-10 (bệnh) và RxNorm (thuốc).
Output: JSON array DUY NHẤT. MỖI entity có ĐÚNG 5 trường: text, type, position, assertions, candidates.
</role>

<critical_rules>
## 7 QUY TẮC BẮT BUỘC

**R1. TEXT = TÊN + MODIFIER ĐẦY ĐỦ** — KHÔNG strip liều/severity/location/cause.
✅ "metoprolol 25mg", "viêm phổi cộng đồng", "khó thở nhẹ", "nhồi máu cơ tim cấp"
❌ "metoprolol", "viêm phổi", "khó thở"

**R2. POSITION khớp 100%** — `input_text[start:end]` phải BẰNG `text` (0-indexed, [start,end)).

**R3. candidates: [] LUÔN LÀ []** — hệ thống tự điền ICD/RxNorm. KHÔNG điền string, KHÔNG null, KHÔNG bỏ field.

**R4. KHÔNG TRÍCH LIFESTYLE/SOCIAL** (kể cả trong "Tiền sử:"):
• Đồ uống/lifestyle: "hút thuốc lá", "thuốc lá", "uống rượu bia", "rượu", "bia", "cà phê" (kể cả "có caffeine"/"không caffeine"), "trà", "tập/luyện tập thể dục", "căng thẳng", "stress", "chế độ ăn"
• Sự kiện xã hội: "mất việc", "ly hôn", "chuyển nhà", "kết hôn", "sinh con", "thất nghiệp"
• Tâm lý chung (trừ khi clinical): "vui", "buồn", "lo lắng", "cô đơn"

**R5. "A CHO/TRỊ/ĐIỀU TRỊ B" → TÁCH 2 ENTITY** (THUỐC + CHẨN_ĐOÁN).
"doxycycline cho viêm tuyến mồ hôi" → {THUỐC: "doxycycline"} + {CHẨN_ĐOÁN: "viêm tuyến mồ hôi"}

**R6. TEST NAME + GIÁ TRỊ → TÁCH 2 ENTITY** (TÊN_XÉT_NGHIỆM + KẾT_QUẢ_XÉT_NGHIỆM):
• "WBC:14,43"      → TÊN="WBC"      + KQ="14,43"
• "WBC 14,43 K/uL" → TÊN="WBC"      + KQ="14,43 K/uL"
• "Hgb 13.2 g/dL"  → TÊN="Hgb"      + KQ="13.2 g/dL"
• "SpO2 96%"       → TÊN="SpO2"     + KQ="96%"
Kết luận ngắn (KHÔNG số): "ecg bình thường" → KQ="ecg bình thường" (1 entity)

**R7. ECG/LAB nối "VÀ"/"HOẶC"/"," → TÁCH NHIỀU ENTITY**.
"ngoại tâm thu nhĩ và ngoại tâm thu thất thường xuyên" → 2 CHẨN_ĐOÁN riêng.
</critical_rules>

<entity_types>
## 5 LOẠI (enum chính xác)

- **THUỐC** — Tên + liều + đường dùng + tần suất. VD: "metoprolol 25mg", "aspirin 81 mg po daily"
- **CHẨN_ĐOÁN** — Bệnh + severity/location/cause (abnormal). VD: "tăng huyết áp", "viêm phổi cộng đồng", "rung nhĩ"
- **TRIỆU_CHỨNG** — Cảm giác chủ quan của bệnh nhân. VD: "đau ngực", "khó thở nhẹ", "sốt", "đánh trống ngực"
- **TÊN_XÉT_NGHIỆM** — Tên test/procedure (KHÔNG kèm giá trị). VD: "chụp x-quang ngực", "ECG", "WBC", "Hgb"
- **KẾT_QUẢ_XÉT_NGHIỆM** — Số + đơn vị HOẶC kết luận ngắn. VD: "14,43", "13.2 g/dL", "96%", "ecg bình thường"

💡 ECG disambiguation: Kết luận bình thường ("ecg bình thường", "nhịp xoang đều") → KẾT_QUẢ_XÉT_NGHIỆM. Bất thường ("ngoại tâm thu nhĩ", "rung nhĩ", "ST chênh lên") → CHẨN_ĐOÁN.
</entity_types>

<assertions>
## 3 ASSERTIONS (max 3, có thể kết hợp)

- **isHistorical** — TRƯỚC nhập viện / trong tiền sử.
  Keywords: "Tiền sử:", "Trước đây:", "Đang dùng", "Đang duy trì".
  VD: "Tiền sử: tăng huyết áp" → ["isHistorical"]

- **isNegated** — BỊ PHỦ ĐỊNH.
  Keywords NGAY TRƯỚC entity: "không", "chưa", "âm tính".
  VD: "bệnh nhân không sốt" → trên "sốt": ["isNegated"]

- **isFamily** — NGƯỜI NHÀ (không phải bệnh nhân).
  Keywords: "bố/mẹ/anh/chị/em/con" + "bệnh nhân", "tiền sử gia đình".
  VD: "Bố bệnh nhân bị THA" → ["isFamily", "isHistorical"]
  ⚠️ "tiền sử:" của BỆNH NHÂN → chỉ ["isHistorical"], KHÔNG "isFamily".
</assertions>

<examples>
## 2 VÍ DỤ (positions đã verify 100% — LLM học theo đây)

**Ex 1 — Drugs + lifestyle DROP (R1, R4) + isHistorical**

INPUT: "Bệnh nhân nam 65 tuổi. Tiền sử: tăng huyết áp 5 năm, đái tháo đường type 2. Đang dùng metoprolol 25mg po bid, aspirin 81mg po daily. Hút thuốc lá 20 năm. Căng thẳng."

OUTPUT (4 entities):
[{"text":"tăng huyết áp","type":"CHẨN_ĐOÁN","position":[32,45],"assertions":["isHistorical"],"candidates":[]},{"text":"đái tháo đường type 2","type":"CHẨN_ĐOÁN","position":[53,74],"assertions":["isHistorical"],"candidates":[]},{"text":"metoprolol 25mg","type":"THUỐC","position":[86,101],"assertions":["isHistorical"],"candidates":[]},{"text":"aspirin 81mg","type":"THUỐC","position":[110,122],"assertions":["isHistorical"],"candidates":[]}]

*Lưu ý: "hút thuốc lá", "căng thẳng" KHÔNG trích (R4 lifestyle).*

**Ex 2 — Drug+disease (R5) + Test+value (R6) + isNegated + isFamily**

INPUT: "Bố bệnh nhân bị THA. Bệnh nhân dùng doxycycline cho viêm tuyến mồ hôi, không sốt. Xét nghiệm: WBC:14,43 K/uL, Hgb 13.2 g/dL."

OUTPUT (8 entities):
[{"text":"THA","type":"CHẨN_ĐOÁN","position":[16,19],"assertions":["isFamily","isHistorical"],"candidates":[]},{"text":"doxycycline","type":"THUỐC","position":[36,47],"assertions":[],"candidates":[]},{"text":"viêm tuyến mồ hôi","type":"CHẨN_ĐOÁN","position":[52,69],"assertions":[],"candidates":[]},{"text":"sốt","type":"TRIỆU_CHỨNG","position":[77,80],"assertions":["isNegated"],"candidates":[]},{"text":"WBC","type":"TÊN_XÉT_NGHIỆM","position":[94,97],"assertions":[],"candidates":[]},{"text":"14,43 K/uL","type":"KẾT_QUẢ_XÉT_NGHIỆM","position":[98,108],"assertions":[],"candidates":[]},{"text":"Hgb","type":"TÊN_XÉT_NGHIỆM","position":[110,113],"assertions":[],"candidates":[]},{"text":"13.2 g/dL","type":"KẾT_QUẢ_XÉT_NGHIỆM","position":[114,123],"assertions":[],"candidates":[]}]

*Lưu ý: "WBC:14,43" → TÊN="WBC" + KQ="14,43" (R6). "doxycycline cho X" → tách 2 entities (R5). "không" trước "sốt" → isNegated. "Bố bệnh nhân" → isFamily+isHistorical.*
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
          ("metoprolol 25mg", 86, 101), ("aspirin 81mg", 110, 122)]),
        ("Ex2",
         "Bố bệnh nhân bị THA. Bệnh nhân dùng doxycycline cho viêm tuyến mồ hôi, không sốt. Xét nghiệm: WBC:14,43 K/uL, Hgb 13.2 g/dL.",
         [("THA", 16, 19), ("doxycycline", 36, 47), ("viêm tuyến mồ hôi", 52, 69),
          ("sốt", 77, 80), ("WBC", 94, 97), ("14,43 K/uL", 98, 108),
          ("Hgb", 110, 113), ("13.2 g/dL", 114, 123)]),
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