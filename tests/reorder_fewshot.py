"""Script: Reorder + add 5 new examples to data/examples.jsonl

Priorities (first 20 examples = 5 NEW + 15 most important existing).
Chosen to cover all 5 entity types, all 3 assertions (+ combos), and key edge cases:
 1. Vital signs split (HA + 140/85 mmHg)                     ← NEW
 2. Test name + finding split (X-quang ngực không ghi nhận)  ← NEW
 3. Long CT findings → CHẨN_ĐOÁN (CT ngực cho thấy A, B, C)   ← NEW
 4. Abnormal findings → CHẨN_ĐOÁN + procedure                ← NEW
 5. Procedure vs drug (phẫu thuật nội soi)                   ← NEW
 6. Sectioned note + drug list, isHistorical      (line 1)
 7. All 5 types + isHistorical & isNegated        (line 14)
 8. Negated chain (không sốt, không ho)           (line 11)
 9. isFamily (bố bệnh nhân)                        (line 12)
10. isFamily + isHistorical combo                 (line 16)
11. Drug vs disease split (doxycycline cho viêm)  (line 7)
12. Test name/result split (WBC 12.5 K/uL)        (line 10)
13. Complex full note n=17, all 5 types           (line 30)
14. isHistorical drug list                        (line 3)
15. R10 duplicate + isFamily + isHistorical       (line 24)
16. Decimal-comma number edge case (14,43)        (line 32)
17. Complex cardio note, isHistorical             (line 25)
18. CHẨN_ĐOÁN + TRIỆU_CHỨNG (ung thư giai đoạn)   (line 18)
19. Weekly drug dosing (methotrexate)             (line 8)
20. Negative case n=1: skip vague drug            (line 21)
"""

import json
from pathlib import Path

EXAMPLES_PATH = Path("f:/AI_VIETTEL/data/examples.jsonl")

# 5 NEW examples (priorities 1-5).
# Declared as (text, type[, assertions]) tuples; character positions are
# computed automatically from the input below so they can never drift.
NEW_EXAMPLES_RAW = [
    # Priority 1: Vital signs split
    {
        "input": "Khám: HA 140/85 mmHg, Mạch 80 lần/phút, SpO2 96%.",
        "ents": [
            ("HA", "TÊN_XÉT_NGHIỆM"),
            ("140/85 mmHg", "KẾT_QUẢ_XÉT_NGHIỆM"),
            ("Mạch", "TÊN_XÉT_NGHIỆM"),
            ("80 lần/phút", "KẾT_QUẢ_XÉT_NGHIỆM"),
            ("SpO2", "TÊN_XÉT_NGHIỆM"),
            ("96%", "KẾT_QUẢ_XÉT_NGHIỆM"),
        ],
    },
    # Priority 2: Test name + finding split
    {
        "input": "Chụp X-quang ngực không ghi nhận gì bất thường. Điện tâm đồ là không ghi nhận gì bất thường.",
        "ents": [
            ("X-quang ngực", "TÊN_XÉT_NGHIỆM"),
            ("không ghi nhận gì bất thường", "KẾT_QUẢ_XÉT_NGHIỆM"),
            ("Điện tâm đồ", "TÊN_XÉT_NGHIỆM"),
            ("không ghi nhận gì bất thường", "KẾT_QUẢ_XÉT_NGHIỆM"),
        ],
    },
    # Priority 3: Long CT findings split
    {
        "input": "Chụp CT ngực cho thấy tim to, tràn dịch màng phổi hai bên, xẹp phổi hai đáy.",
        "ents": [
            ("Chụp CT ngực", "TÊN_XÉT_NGHIỆM"),
            ("tim to", "CHẨN_ĐOÁN"),
            ("tràn dịch màng phổi hai bên", "CHẨN_ĐOÁN"),
            ("xẹp phổi hai đáy", "CHẨN_ĐOÁN"),
        ],
    },
    # Priority 4: Abnormal findings → CHẨN_ĐOÁN
    {
        "input": "Siêu âm tim: hở van hai lá vừa, EF 30%, tim to. Đặt stent động mạch vành.",
        "ents": [
            ("Siêu âm tim", "TÊN_XÉT_NGHIỆM"),
            ("hở van hai lá vừa", "CHẨN_ĐOÁN"),
            ("EF 30%", "KẾT_QUẢ_XÉT_NGHIỆM"),
            ("tim to", "CHẨN_ĐOÁN"),
            ("Đặt stent động mạch vành", "TÊN_XÉT_NGHIỆM"),
        ],
    },
    # Priority 5: Procedure vs drug
    {
        "input": "Bệnh nhân được phẫu thuật nội soi dạ dày, sau đó dùng esomeprazole 40mg.",
        "ents": [
            ("phẫu thuật nội soi dạ dày", "TÊN_XÉT_NGHIỆM"),
            ("esomeprazole 40mg", "THUỐC"),
        ],
    },
]


def _build_new_examples(raw: list[dict]) -> list[dict]:
    """Expand (text, type) tuples into full entity dicts with computed positions.

    Positions are found by a sequential, case-insensitive scan so repeated
    spans (e.g. the same finding twice) map to distinct offsets in order.
    """
    built = []
    for ex in raw:
        text = ex["input"]
        low = text.lower()
        cursor = 0
        out = []
        for ent in ex["ents"]:
            span, etype = ent[0], ent[1]
            assertions = list(ent[2]) if len(ent) > 2 else []
            idx = low.find(span.lower(), cursor)
            if idx < 0:
                raise ValueError(f"Span {span!r} not found in {text!r} after {cursor}")
            end = idx + len(span)
            out.append({
                "text": text[idx:end],
                "type": etype,
                "position": [idx, end],
                "assertions": assertions,
                "candidates": [],
            })
            cursor = end
        built.append({"input": text, "output": out})
    return built


NEW_EXAMPLES = _build_new_examples(NEW_EXAMPLES_RAW)

# 20 priority examples for the front of the file
# (5 NEW + 15 most important existing, covering all types/assertions/edge cases)
PRIORITY_ORDER = [
    "NEW:1",  # Vital signs split
    "NEW:2",  # Test name + finding split
    "NEW:3",  # Long CT findings → CHẨN_ĐOÁN
    "NEW:4",  # Abnormal findings → CHẨN_ĐOÁN + procedure
    "NEW:5",  # Procedure vs drug
    1,   # Sectioned note + drug list, isHistorical
    14,  # All 5 types + isHistorical & isNegated
    11,  # Negated chain
    12,  # isFamily
    16,  # isFamily + isHistorical combo
    7,   # Drug vs disease split
    10,  # Test name/result split
    30,  # Complex full note n=17, all 5 types
    3,   # isHistorical drug list
    24,  # R10 duplicate + isFamily + isHistorical
    32,  # Decimal-comma number edge case
    25,  # Complex cardio note, isHistorical
    18,  # CHẨN_ĐOÁN + TRIỆU_CHỨNG
    8,   # Weekly drug dosing
    21,  # Negative case n=1: skip vague drug
]

# Read existing
existing = []
with open(EXAMPLES_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            existing.append(json.loads(line))

print(f"Existing examples: {len(existing)}")

# Build new order
ordered = []
priority_indices_used = set()

# Add priority examples in order
for item in PRIORITY_ORDER:
    if isinstance(item, str) and item.startswith("NEW:"):
        idx = int(item.split(":")[1]) - 1
        ordered.append(NEW_EXAMPLES[idx])
    else:
        # Existing 1-indexed
        ordered.append(existing[item - 1])
        priority_indices_used.add(item)

# Add remaining existing (skip priority)
for i, ex in enumerate(existing, 1):
    if i not in priority_indices_used:
        ordered.append(ex)

print(f"Total examples after reorder+add: {len(ordered)} (expected: {len(existing) + 5})")

# Write back
with open(EXAMPLES_PATH, "w", encoding="utf-8") as f:
    for ex in ordered:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"✅ Done. File: {EXAMPLES_PATH}")
print(f"First 20 examples (priority):")
for i, ex in enumerate(ordered[:20], 1):
    types = sorted(set(e["type"] for e in ex["output"]))
    print(f"  {i}. types={types}: {ex['input'][:60]}...")
