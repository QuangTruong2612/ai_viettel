"""Comprehensive audit of NER Completeness (Recall) and Correctness (Precision/Type/Offset)."""

from __future__ import annotations

import json
import re
import glob
from pathlib import Path
from collections import Counter, defaultdict

inp_dir = Path("data/input") if Path("data/input").exists() else Path("input")

# Common disease patterns that MUST be CHẨN_ĐOÁN
DISEASE_PATTERNS = [
    (r"\bviêm\s+\w+", "CHẨN_ĐOÁN"),
    (r"\btăng\s+huyết\s+áp\b", "CHẨN_ĐOÁN"),
    (r"\bđái\s+tháo\s+đường(?:\s+type\s+[12])?\b", "CHẨN_ĐOÁN"),
    (r"\bnhồi\s+máu\s+(?:cơ\s+tim|não)\b", "CHẨN_ĐOÁN"),
    (r"\bsuy\s+(?:tim|thận|gan)\b", "CHẨN_ĐOÁN"),
    (r"\brung\s+nhĩ\b", "CHẨN_ĐOÁN"),
    (r"\bxơ\s+gan\b", "CHẨN_ĐOÁN"),
    (r"\brối\s+loạn\s+lipid\s+máu\b", "CHẨN_ĐOÁN"),
    (r"\bthoái\s+hóa\s+khớp\b", "CHẨN_ĐOÁN"),
    (r"\bsỏi\s+\w+\b", "CHẨN_ĐOÁN"),
    (r"\bhội\s+chứng\s+\w+\b", "CHẨN_ĐOÁN"),
    (r"\bung\s+thư\s+\w+\b", "CHẨN_ĐOÁN"),
]

# Common symptom patterns that MUST be TRIỆU_CHỨNG
SYMPTOM_PATTERNS = [
    r"\bđau\s+ngực(?:\s+trái)?\b",
    r"\bkhó\s+thở(?:\s+nhẹ)?\b",
    r"\bsốt(?:\s+cao)?\b",
    r"\bbuồn\s+nôn\b",
    r"\bnôn(?:\s+ói)?\b",
    r"\bmệt\s+mỏi\b",
    r"\bđánh\s+trống\s+ngực\b",
    r"\bchóng\s+mặt\b",
    r"\bho(?:\s+nhiều|\s+khạc\s+đờm)?\b",
    r"\bphù(?:\s+hai\s+chân)?\b",
    r"\bvã\s+mồ\s+hôi\b",
    r"\bđổ\s+mồ\s+hôi\b",
]

# Drug dosage pattern
DRUG_DOSAGE_PATTERN = re.compile(r"\b[a-zA-Z\u00C0-\u024F\u1E00-\u1EFF]+\s+\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|iu|viên|ống|gói)\b", re.IGNORECASE)

def audit_ner():
    output_files = sorted(
        [f for f in glob.glob("output/*.json") if Path(f).stem.isdigit()],
        key=lambda x: int(Path(x).stem),
    )
    print(f"[INFO] Auditing NER completeness and correctness across {len(output_files)} files...\n")

    offset_errors = 0
    type_errors = 0
    missing_diseases = []
    missing_symptoms = []
    missing_drugs = []

    total_entities = 0

    for fpath in output_files:
        rec_id = int(Path(fpath).stem)
        inp_path = inp_dir / f"{rec_id}.txt"
        if not inp_path.exists():
            inp_path = inp_dir / f"{rec_id}.json"
        if not inp_path.exists():
            continue

        input_text = inp_path.read_text(encoding="utf-8")
        entities = json.loads(Path(fpath).read_text(encoding="utf-8"))
        total_entities += len(entities)

        # 1. OFFSET VERIFICATION
        extracted_spans = set()
        for ent in entities:
            txt = ent.get("text", "")
            pos = ent.get("position", [0, 0])
            etype = ent.get("type", "")

            if not isinstance(pos, list) or len(pos) != 2:
                offset_errors += 1
                continue

            s, e = pos[0], pos[1]
            actual_sub = input_text[s:e]
            if actual_sub != txt:
                offset_errors += 1
                print(f"  [OFFSET ERR] Record {rec_id}: ent='{txt}' vs actual='{actual_sub}' at [{s},{e}]")
            else:
                extracted_spans.add((s, e))

            # 2. TYPE CORRECTNESS CHECK
            # Check if 'viêm X' is misclassified as TRIỆU_CHỨNG
            if txt.lower().startswith("viêm ") and etype == "TRIỆU_CHỨNG":
                type_errors += 1
                print(f"  [TYPE ERR] Record {rec_id}: '{txt}' is classified as TRIỆU_CHỨNG (should be CHẨN_ĐOÁN)")

        # 3. COMPLETENESS (RECALL) CHECK
        # Check missing diseases
        for pat, expected_type in DISEASE_PATTERNS:
            for m in re.finditer(pat, input_text, re.IGNORECASE):
                s, e = m.start(), m.end()
                # Check if this span or overlapping span was extracted
                is_extracted = any(max(s, es) < min(e, ee) for es, ee in extracted_spans)
                if not is_extracted:
                    missing_diseases.append((rec_id, m.group(0), s, e))

        # Check missing symptoms
        for pat in SYMPTOM_PATTERNS:
            for m in re.finditer(pat, input_text, re.IGNORECASE):
                s, e = m.start(), m.end()
                is_extracted = any(max(s, es) < min(e, ee) for es, ee in extracted_spans)
                if not is_extracted:
                    missing_symptoms.append((rec_id, m.group(0), s, e))

        # Check missing drugs with explicit dosages
        for m in DRUG_DOSAGE_PATTERN.finditer(input_text):
            s, e = m.start(), m.end()
            is_extracted = any(max(s, es) < min(e, ee) for es, ee in extracted_spans)
            if not is_extracted:
                missing_drugs.append((rec_id, m.group(0), s, e))

    print("══════════════════════════════════════════════════════════════════════")
    print(f"  Total Entities Audited:  {total_entities}")
    print(f"  Offset Errors:           {offset_errors}  (Goal = 0)")
    print(f"  Type Errors:             {type_errors}  (Goal = 0)")
    print(f"  Missing Diseases:        {len(missing_diseases)}")
    print(f"  Missing Symptoms:        {len(missing_symptoms)}")
    print(f"  Missing Drugs (dosages): {len(missing_drugs)}")
    print("══════════════════════════════════════════════════════════════════════")

    if missing_diseases:
        print("\n--- SAMPLE MISSING DISEASES (first 10) ---")
        for r, txt, s, e in missing_diseases[:10]:
            print(f"  [{r}] '{txt}' at [{s},{e}]")

    if missing_symptoms:
        print("\n--- SAMPLE MISSING SYMPTOMS (first 10) ---")
        for r, txt, s, e in missing_symptoms[:10]:
            print(f"  [{r}] '{txt}' at [{s},{e}]")

    if missing_drugs:
        print("\n--- SAMPLE MISSING DRUGS WITH DOSAGES (first 10) ---")
        for r, txt, s, e in missing_drugs[:10]:
            print(f"  [{r}] '{txt}' at [{s},{e}]")


if __name__ == "__main__":
    audit_ner()
