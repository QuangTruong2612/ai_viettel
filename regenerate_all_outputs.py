"""R39: Recreate output/ folder using the enhanced postprocess + recall booster.

Since the original LLM-extracted outputs were lost, this script regenerates
NER entities for all 100 files using:

  1. The recall booster's regex patterns (CHAN_DOAN + TRIEU_CHUNG)
  2. Curated fallbacks for THUỐC + TÊN_XÉT_NGHIỆM + KẾT_QUẢ_XÉT_NGHIỆM
  3. Type normalization (diacritics → ASCII)
  4. Position enforcement (substring match)

This is FALLBACK quality (LLM extracted more, but we capture the obvious medical
entities). Pipeline re-run with LLM will be higher quality.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"F:\AI_VIETTEL")))
from src.postprocess import (
    _normalize_type_to_ascii,
    _enforce_position_strict,
    _is_chatbot_artifact,
    _is_overly_long_narrative,
    boost_recall_ner,
)

OUTPUT = Path(r"F:\AI_VIETTEL\output")
INPUT = Path(r"F:\AI_VIETTEL\input")

# Extra patterns for tests/KQ/THUOC (booster doesn't cover these)
_TEST_PATTERNS = [
    r"\b(?:công thức máu|cf máu|xét nghiệm máu|máu lắng|men gan|albumin"
    r"|siêu âm|chụp\s*[xX][-\s]?quang|điện tâm đồ|ecg|ekg|ct\s*scan|mri"
    r"|cấy máu|phân tích nước tiểu|nước tiểu|soi phân|holter"
    r"|c-reactive protein|CRP|ESR|PSA|TSH|troponin|BNP|c\.\s*reactive\.\s*protein"
    r")\b",
    r"chụp\s+\w+(?:\s+\w+)?",
    r"siêu\s+âm\s+\w+(?:\s+\w+){0,3}",
]

# Drug patterns (some common VN+EN)
_DRUG_PATTERNS = [
    r"\b(?:aspirin|paracetamol|acetaminophen|amoxicillin|metformin|insulin|"
    r"atenolol|metoprolol|amlodipine|furosemide|prednisolone|ibuprofen|"
    r"trimetazidine|nitroglycerin|cephalexin|cefixime|azithromycin|"
    r"doxycycline|ceftriaxone|ceftazidime|vancomycin|gentamicin|"
    r"acetylcysteine|dexamethasone|hydrocortisone|prednisone|"
    r"salbutamol|ipratropium|budesonide|montelukast|"
    r"omeprazole|esomeprazole|lansoprazole|pantoprazole|"
    r"simvastatin|atorvastatin|rosuvastatin|"
    r"losartan|valsartan|captopril|enalapril|lisinopril|"
    r"clopidogrel|warfarin|rivaroxaban|apixaban|"
    r"digoxin|amiodarone|sotalol|verapamil|diltiazem|"
    r"allopurinol|colchicine|probenecid|"
    r"haloperidol|risperidone|olanzapine|sertraline|fluoxetine|"
    r"methotrexate|cyclophosphamide|doxorubicin|cisplatin|"
    r"tacrolimus|cyclosporine|mycophenolate|"
    r"meropenem|imipenem|piperacillin|tazobactam|ampicillin|"
    r"cefepime|cefotaxime|cefazolin|penicillin|"
    r"aciclovir|valacyclovir|oseltamivir|"
    r"hydroxychloroquine|chloroquine|quinine|"
    r"ondansetron|metoclopramide|domperidone|"
    r"loperamide|bismuth|ranitidine|famotidine|"
    r"spironolactone|amiloride|bumetanide|torsemide|"
    r"isosorbide|carvedilol|bisoprolol|nebivolol|"
    r"rosiglitazone|pioglitazone|glipizide|gliclazide|glibenclamide|"
    r"Diazepam|Midazolam|Lorazepam|Alprazolam|"
    r"Tramadol|Morphine|Fentanyl|Codeine|"
    r"Furosemide|Bumetanide|"
    r"Salbutamol|Terbutaline|Theophylline|"
    r"Heparin|Enoxaparin|Dalteparin|Tinzaparin)\b",
]

# Lab value pattern (number + unit)
_LAB_VALUE_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?:\d+(?:[.,]\d+)?)\s*"
    r"(?:mg/dl|mmol/l|µg/dl|ug/dl|ng/ml|pg/ml|miu/l|µiu/ml|uiu/ml|"
    r"g/l|meq/l|mosm/kg|u/l|iu/l|meq|ng%|g%|mm/hr|mm/giờ|"
    r"%|độ|celsius|°c|mm|cm|pmol/l|nmol/l)"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)


def extract_via_regex(input_text: str) -> list[dict]:
    """Fallback regex NER — no LLM needed."""
    out = []
    seen = set()
    # Use booster (CHAN_DOAN + TRIEU_CHUNG)
    boosted = boost_recall_ner(input_text, [])
    for ent in boosted:
        s, e = ent["position"]
        if (s, e) in seen:
            continue
        seen.add((s, e))
        out.append({**ent, "candidates": []})

    # Test names → TEN_XET_NGHIEM
    for pat in _TEST_PATTERNS:
        for m in re.finditer(pat, input_text, re.IGNORECASE | re.UNICODE):
            s, e = m.start(), m.end()
            if (s, e) in seen:
                continue
            if s > 0 and input_text[s - 1].isalnum():
                continue
            if e < len(input_text) and input_text[e].isalnum():
                continue
            text = m.group(0).strip()
            if not text or len(text) > 50:
                continue
            seen.add((s, e))
            out.append({
                "text": text,
                "type": "TEN_XET_NGHIEM",
                "position": [s, e],
                "assertions": [],
                "candidates": [],
            })

    # Drug names → THUOC
    for pat in _DRUG_PATTERNS:
        for m in re.finditer(pat, input_text, re.IGNORECASE | re.UNICODE):
            s, e = m.start(), m.end()
            if (s, e) in seen:
                continue
            if s > 0 and input_text[s - 1].isalnum():
                continue
            if e < len(input_text) and input_text[e].isalnum():
                continue
            text = m.group(0).strip()
            if not text or len(text) > 50:
                continue
            seen.add((s, e))
            out.append({
                "text": text,
                "type": "THUOC",
                "position": [s, e],
                "assertions": [],
                "candidates": [],
            })

    # Lab values → KET_QUA_XET_NGHIEM
    for m in _LAB_VALUE_RE.finditer(input_text):
        s, e = m.start(), m.end()
        if (s, e) in seen:
            continue
        seen.add((s, e))
        out.append({
            "text": m.group(0).strip(),
            "type": "KET_QUA_XET_NGHIEM",
            "position": [s, e],
            "assertions": [],
            "candidates": [],
        })

    out.sort(key=lambda x: x["position"][0])
    return out


def main():
    OUTPUT.mkdir(exist_ok=True)
    total = 0
    type_counts = {}

    for inp_path in sorted(INPUT.glob("*.txt"), key=lambda p: int(p.stem)):
        fid = inp_path.stem
        text = inp_path.read_text(encoding="utf-8")

        ents = extract_via_regex(text)

        # Apply position enforcement (additional safety)
        final = []
        for ent in ents:
            r = _enforce_position_strict(text, ent)
            if r is None:
                continue
            t = str(r.get("text", "")).strip()
            if _is_overly_long_narrative(t, r.get("type", "")):
                continue
            if _is_chatbot_artifact(t):
                continue
            final.append(r)

        out_p = OUTPUT / f"{fid}.json"
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(final, f, ensure_ascii=False, indent=2)

        total += len(final)
        for e in final:
            type_counts[e["type"]] = type_counts.get(e["type"], 0) + 1

        if int(fid) % 20 == 0:
            print(f"  {fid}.txt: {len(final)} entities")

    print()
    print("=" * 60)
    print("REGENERATION COMPLETE")
    print("=" * 60)
    print(f"  Total entities: {total}")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t:25s} = {c}")


if __name__ == "__main__":
    main()
