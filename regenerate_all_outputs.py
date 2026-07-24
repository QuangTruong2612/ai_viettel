"""R39: Regenerate output/ folder using postprocess + recall booster.

Output format:
- All types use DIACRITICS (CHẨN_ĐOÁN, TRIỆU_CHỨNG, ...)
- No `_booster` flag (internal marker, stripped before output)
- Each entity has: text, type, position, assertions, candidates
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"F:\AI_VIETTEL")))
from src.postprocess import (
    _restore_diacritics_type,
    _enforce_position_strict,
    _is_chatbot_artifact,
    _is_overly_long_narrative,
    boost_recall_ner,
)

OUTPUT = Path(r"F:\AI_VIETTEL\output")
INPUT = Path(r"F:\AI_VIETTEL\input")

# DIACRITICS — matches grader schema enum
T_CHAN_DOAN = "CHẨN_ĐOÁN"
T_TRIEU_CHUNG = "TRIỆU_CHỨNG"
T_TEN_XET_NGHIEM = "TÊN_XÉT_NGHIỆM"
T_KET_QUA = "KẾT_QUẢ_XÉT_NGHIỆM"
T_THUOC = "THUỐC"

_TEST_PATTERNS = [
    r"\b(?:công thức máu|cf máu|xét nghiệm máu|máu lắng|men gan|albumin"
    r"|siêu âm|chụp\s*[xX][-\s]?quang|điện tâm đồ|ecg|ekg|ct\s*scan|mri"
    r"|cấy máu|phân tích nước tiểu|nước tiểu|soi phân|siêu âm tim"
    r"|nội soi|sinh thiết|tế bào|mô bệnh học"
    r"|PSA|TSH|WBC|Hgb|AST|ALT|GGT|CRP)\b",
    r"chụp\s+\w+(?:\s+\w+)?",
    r"siêu\s+âm\s+\w+(?:\s+\w+){0,3}",
]

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

_LAB_VALUE_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?:\d+(?:[.,]\d+)?)\s*"
    r"(?:mg/dl|mmol/l|µg/dl|ug/dl|ng/ml|pg/ml|miu/l|µiu/ml|uiu/ml|"
    r"g/l|meq/l|mosm/kg|u/l|iu/l|meq|ng%|g%|mm/hr|mm/giờ|"
    r"%|độ|celsius|°c|mm|cm|pmol/l|nmol/l|mmHg|cmH2O|lần/phút|"
    r"nhịp/phút|kg/m2|kg/m²)"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)


def extract_via_regex(input_text: str) -> list[dict]:
    """Fallback regex NER — types use DIACRITICS."""
    out = []
    seen = set()

    # Use booster (CHẨN_ĐOÁN + TRIỆU_CHỨNG)
    boosted = boost_recall_ner(input_text, [])
    for ent in boosted:
        s, e = ent["position"]
        if (s, e) in seen:
            continue
        seen.add((s, e))
        # Skip the _booster flag — it's internal metadata
        clean_ent = {k: v for k, v in ent.items() if not k.startswith("_")}
        out.append({**clean_ent, "candidates": []})

    # Test names → TÊN_XÉT_NGHIỆM
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
                "type": T_TEN_XET_NGHIEM,
                "position": [s, e],
                "assertions": [],
                "candidates": [],
            })

    # Drug names → THUỐC
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
                "type": T_THUOC,
                "position": [s, e],
                "assertions": [],
                "candidates": [],
            })

    # Lab values → KẾT_QUẢ_XÉT_NGHIỆM
    for m in _LAB_VALUE_RE.finditer(input_text):
        s, e = m.start(), m.end()
        if (s, e) in seen:
            continue
        seen.add((s, e))
        out.append({
            "text": m.group(0).strip(),
            "type": T_KET_QUA,
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
    booster_count = 0

    for inp_path in sorted(INPUT.glob("*.txt"), key=lambda p: int(p.stem)):
        fid = inp_path.stem
        text = inp_path.read_text(encoding="utf-8")

        ents = extract_via_regex(text)

        # Apply position enforcement
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
            # Strip any _ prefixed internal keys
            for k in list(r.keys()):
                if k.startswith("_"):
                    r.pop(k, None)
            final.append(r)

        # Count booster-added entities (those without candidates but with assert=[])
        for e in final:
            if not e.get("candidates"):
                booster_count += 1

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
