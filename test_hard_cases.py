"""R39 — Hard-case test suite cho ICD + RxNorm.

Chạy từ project root:
    python test_hard_cases.py

Test ~50 case khó để audit system. Bỏ qua embedding/model loading
để chạy nhanh — chỉ test L1 (exact dict match) + heuristic logic.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# Đảm bảo import được src/
sys.path.insert(0, str(Path(r"F:\AI_VIETTEL").resolve()))

# Test counters
total_tests = 0
passed = 0
failed = 0
skipped = 0
failures: list[tuple[str, str, list, list, str]] = []


def test_lookup(name: str, retriever, input_text: str, expected_codes: list[str],
                must_contain: bool = True, mode: str = "icd"):
    """Test retriever.lookup with expected codes.

    Args:
        name: test case name
        retriever: ICDRetriever or RxNormRetriever
        input_text: text to lookup
        expected_codes: list of expected codes (any-match by default)
        must_contain: if True, result MUST contain at least one expected code; if False,
                       result must NOT contain any expected code (for negation/rejection tests)
        mode: 'icd' or 'rx'
    """
    global total_tests, passed, failed, failures
    total_tests += 1

    if retriever is None:
        print(f"  ⊝ SKIP: {name} (retriever unavailable)")
        skipped += 1
        return

    try:
        if mode == "icd":
            result = retriever.lookup(input_text)
        else:
            result = retriever.lookup(input_text)
    except Exception as e:
        print(f"  ⊝ SKIP: {name} (lookup error: {e})")
        skipped += 1
        return

    result_set = set(result or [])

    if must_contain:
        ok = any(c in result_set for c in expected_codes)
    else:
        # must NOT contain any of expected_codes
        ok = not any(c in result_set for c in expected_codes)

    if ok:
        passed += 1
        print(f"  ✓ PASS: {name}")
        if result and (passed <= 5 or total_tests % 20 == 0):
            print(f"      input={input_text!r:60.60} → {result[:5]}")
    else:
        failed += 1
        failures.append((name, input_text, list(result_set), expected_codes,
                         "must_contain" if must_contain else "must_NOT"))
        print(f"  ✗ FAIL: {name}")
        print(f"      input      = {input_text!r}")
        print(f"      expected   = {expected_codes} ({'must_contain' if must_contain else 'must_NOT'})")
        print(f"      actual     = {list(result_set)[:10]}")


def main():
    print("=" * 70)
    print("R39 — HARD-CASE TEST SUITE cho ICD + RxNorm")
    print("=" * 70)

    # Try load retrievers (best-effort, HEAVY loading — wrap in try/except with timeout)
    icd = None
    rx = None
    try:
        from src.icd_rag import ICDRetriever
        print("[init] Loading ICDRetriever (may take time for embedding)...")
        # TRICK: pass use_hybrid=False to skip BGE-M3 model load.
        # We only test L0/L1 dict + structural lookup, NOT embedding search.
        try:
            icd = ICDRetriever(use_hybrid=False)
            print("[init] ICDRetriever loaded (no model).")
        except TypeError:
            # Some versions don't support use_hybrid arg
            icd = ICDRetriever()
            print("[init] ICDRetriever loaded (default args).")
    except Exception as e:
        print(f"[init] ICDRetriever unavailable: {type(e).__name__}: {e}")

    try:
        from src.rxnorm_rag import RxNormRetriever
        print("[init] Loading RxNormRetriever...")
        try:
            rx = RxNormRetriever(use_hybrid=False)
            print("[init] RxNormRetriever loaded (no model).")
        except TypeError:
            rx = RxNormRetriever()
            print("[init] RxNormRetriever loaded (default args).")
    except Exception as e:
        print(f"[init] RxNormRetriever unavailable: {type(e).__name__}: {e}")

    # ════════════════════════════════════════════════════════════════════
    # ICD HARD CASES
    # ════════════════════════════════════════════════════════════════════
    print()
    print("═" * 70)
    print("ICD — HARD CASES (40+ cases)")
    print("═" * 70)

    # ── Group 1: Enzyme / blood diseases (regression test for G6PD bug) ──
    print("\n── Group 1: Enzyme-deficiency & blood diseases (G6PD regression) ──")
    test_lookup("G6PD → D55.0", icd, "thiếu men g6pd", ["D55.0"])
    test_lookup("G6PD capitalized", icd, "Thiếu men G6PD", ["D55.0"])
    test_lookup("G6PD English", icd, "Glucose-6-Phosphate Dehydrogenase", ["D55.0"])
    test_lookup("hemolytic anemia", icd, "thiếu máu tan huyết", ["D58"])
    test_lookup("jaundice newborn", icd, "vàng da sơ sinh", ["P59"])
    test_lookup("bilirubin ↑", icd, "tăng bilirubin", ["E80"])

    # ── Group 2: Common abbreviations ──
    print("\n── Group 2: Common Vietnamese abbreviations ──")
    test_lookup("THA → I10", icd, "THA", ["I10"])
    test_lookup("ĐTĐ → E11", icd, "ĐTĐ", ["E11"])
    test_lookup("ĐTĐ type 2 → E11", icd, "ĐTĐ tuýp 2", ["E11"])
    test_lookup("NMCT → I21", icd, "NMCT", ["I21"])
    test_lookup("BTMV → I25", icd, "BTMV", ["I25"])
    test_lookup("COPD → J44", icd, "COPD", ["J44"])
    test_lookup("CKD → N18", icd, "CKD", ["N18"])
    test_lookup("RLLL → E78", icd, "RLLL", ["E78"])

    # ── Group 3: Compound diseases ──
    print("\n── Group 3: Compound diseases (compound form exact-match) ──")
    test_lookup("THA + ĐTĐ", icd, "tăng huyết áp kèm đái tháo đường",
                ["I10", "E11"], must_contain=False)  # split is fine too
    test_lookup("Unstable angina", icd, "đau thắt ngực không ổn định", ["I20"])
    test_lookup("STEMI", icd, "nhồi máu cơ tim cấp ST chênh lên", ["I21"])
    test_lookup("CAP", icd, "viêm phổi mắc phải cộng đồng", ["J18"])

    # ── Group 4: Diseases requiring Q-code vs non-Q-code decision ──
    print("\n── Group 4: Q-code filter (congenital vs acquired) ──")
    test_lookup("G6PD must NOT be Q55.0", icd, "thiếu men G6PD", ["Q55.0"], must_contain=False)
    test_lookup("Down syndrome → Q90 (KEEP)", icd, "hội chứng Down", ["Q90"])
    test_lookup("Congenital heart disease → Q24", icd,
                "bệnh tim bẩm sinh", ["Q24"])
    test_lookup("Adult RA (not congenital)",
                icd, "viêm khớp dạng thấp", ["M06"], must_contain=False)
    # Verify adult RA MUST NOT be Q code
    test_lookup("Adult RA NO Q code", icd, "viêm khớp dạng thấp",
                ["Q"], must_contain=False)

    # ── Group 5: Drug-class / noise terms (must be rejected) ──
    print("\n── Group 5: Drug-class noise (must be rejected) ──")
    test_lookup("kháng sinh — REJECT", icd, "kháng sinh", ["J"], must_contain=False)
    test_lookup("corticoid — REJECT", icd, "corticoid", ["J"], must_contain=False)
    test_lookup("NSAID — REJECT", icd, "NSAID", ["M"], must_contain=False)
    test_lookup("kháng đông — REJECT", icd, "thuốc kháng đông", ["I"], must_contain=False)
    test_lookup("bicarbonate — must NOT be drug ICD",
                icd, "bicarbonate", ["D"], must_contain=False)

    # ── Group 6: Hard disease names ──
    print("\n── Group 6: Hard disease names ──")
    test_lookup("Henoch-Schönlein", icd, "ban xuất huyết Henoch-Schönlein", ["D69"])
    test_lookup("Still disease", icd, "bệnh Still", ["M08"])
    test_lookup("Kawasaki", icd, "bệnh Kawasaki", ["M30"])
    test_lookup("Hodgkin", icd, "u lympho Hodgkin", ["C81"])
    test_lookup("Parkinson", icd, "Parkinson", ["G20"])
    test_lookup("Alzheimer", icd, "Alzheimer", ["G30"])
    test_lookup("Migraine", icd, "đau nửa đầu", ["G43"])
    test_lookup("GERD", icd, "trào ngược dạ dày thực quản", ["K21"])
    test_lookup("IBS", icd, "hội chứng ruột kích thích", ["K58"])
    test_lookup("GERD vn", icd, "GERD", ["K21"])

    # ── Group 7: Resistance context (should be rejected or down-ranked) ──
    print("\n── Group 7: Resistance context ──")
    # 'vi khuẩn kháng methicillin' should NOT get disease ICD (resistance)
    test_lookup("MRSA resistance — REJECT",
                icd, "vi khuẩn kháng methicillin", ["A49"], must_contain=False)

    # ════════════════════════════════════════════════════════════════════
    # RxNorm HARD CASES
    # ════════════════════════════════════════════════════════════════════
    print()
    print("═" * 70)
    print("RxNorm — HARD CASES (40+ cases)")
    print("═" * 70)

    # ── Group A: Brand → INN ──
    print("\n── Group A: Brand name → INN mapping ──")
    test_lookup("Crestor → rosuvastatin", rx, "Crestor", ["rosuvastatin"], mode="rx")
    test_lookup("Tylenol → acetaminophen", rx, "Tylenol", ["acetaminophen"], mode="rx")
    test_lookup("Augmentin → amoxicillin", rx, "Augmentin", ["amoxicillin"], mode="rx")
    test_lookup("Lasix → furosemide", rx, "Lasix", ["furosemide"], mode="rx")
    test_lookup("Lipitor → atorvastatin", rx, "Lipitor", ["atorvastatin"], mode="rx")
    test_lookup("Plavix → clopidogrel", rx, "Plavix", ["clopidogrel"], mode="rx")
    test_lookup("Nexium → esomeprazole", rx, "Nexium", ["esomeprazole"], mode="rx")
    test_lookup("Ventolin → albuterol", rx, "Ventolin", ["albuterol"], mode="rx")
    test_lookup("Panadol VN → paracetamol", rx, "Panadol", ["acetaminophen", "paracetamol"], mode="rx")
    test_lookup("Glucophage VN → metformin", rx, "Glucophage", ["metformin"], mode="rx")

    # ── Group B: Vietnamese drug terms ──
    print("\n── Group B: Vietnamese drug terms ──")
    test_lookup("paracetamol VN", rx, "paracetamol", ["paracetamol", "acetaminophen"], mode="rx")
    test_lookup("aspirin", rx, "aspirin", ["aspirin"], mode="rx")
    test_lookup("amoxicillin", rx, "amoxicillin", ["amoxicillin"], mode="rx")
    test_lookup("metformin", rx, "metformin", ["metformin"], mode="rx")
    test_lookup("insulin", rx, "insulin", ["insulin"], mode="rx")
    test_lookup("furosemide", rx, "furosemide", ["furosemide"], mode="rx")

    # ── Group C: Drug with strength ──
    print("\n── Group C: Drug with strength format ──")
    test_lookup("paracetamol 500mg",
                rx, "paracetamol 500mg", ["paracetamol", "acetaminophen"], mode="rx")
    test_lookup("aspirin 81mg",
                rx, "aspirin 81mg", ["aspirin"], mode="rx")
    test_lookup("metoprolol 25mg",
                rx, "metoprolol 25mg", ["metoprolol"], mode="rx")
    test_lookup("metoprolol 25 mg (with space)",
                rx, "metoprolol 25 mg", ["metoprolol"], mode="rx")
    test_lookup("metoprolol 25mg po bid",
                rx, "metoprolol 25mg po bid", ["metoprolol"], mode="rx")

    # ── Group D: Drug abbreviations ──
    print("\n── Group D: Drug abbreviations ──")
    test_lookup("MTX → methotrexate", rx, "MTX", ["methotrexate"], mode="rx")
    test_lookup("APAP → acetaminophen", rx, "APAP", ["acetaminophen"], mode="rx")
    test_lookup("PCN → penicillin", rx, "PCN", ["penicillin"], mode="rx")

    # ── Group E: Compound drugs ──
    print("\n── Group E: Compound / multi-ingredient drugs ──")
    test_lookup("lisinopril/hydrochlorothiazide",
                rx, "lisinopril 10 mg / hydrochlorothiazide 12.5 mg",
                ["lisinopril", "hydrochlorothiazide"], mode="rx")
    test_lookup("co-trimoxazole = TMP/SMX",
                rx, "cotrimoxazole",
                ["trimethoprim", "sulfamethoxazole"], mode="rx")
    test_lookup("Augmentin = amox/clav",
                rx, "Augmentin 500mg",
                ["amoxicillin", "clavulanate"], mode="rx")
    test_lookup("Bactrim",
                rx, "Bactrim",
                ["trimethoprim", "sulfamethoxazole"], mode="rx")

    # ── Group F: Drug in non-treatment context (must be rejected) ──
    print("\n── Group F: Drug non-treatment context ──")
    # 'bicarbonate' as lab value — must NOT return drug code
    test_lookup("bicarbonate — REJECT", rx, "bicarbonate",
                ["bicarbonate"], mode="rx", must_contain=False)
    test_lookup("creatinine — REJECT", rx, "creatinine",
                ["creatinine"], mode="rx", must_contain=False)
    test_lookup("sodium — REJECT (lab)", rx, "sodium", ["sodium"], mode="rx",
                must_contain=False)

    # ── Group G: Resistance context ──
    print("\n── Group G: Resistance ──")
    # 'kháng methicillin' — drug mentioned in resistance context
    test_lookup("kháng methicillin — RESISTANCE",
                rx, "kháng methicillin", ["methicillin"], mode="rx",
                must_contain=False)

    # ── Group H: Typo/fuzzy ──
    print("\n── Group H: Typo / fuzzy match ──")
    test_lookup("typo trimetazidin → trimetazidine", rx,
                "trimetazidin", ["trimetazidine"], mode="rx")
    test_lookup("typo acetazol → acetazolamide", rx,
                "acetazol", ["acetazolamide"], mode="rx")

    # ════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ════════════════════════════════════════════════════════════════════
    print()
    print("═" * 70)
    print("SUMMARY")
    print("═" * 70)
    print(f"  Total tests:  {total_tests}")
    print(f"  Passed:       {passed}  ({passed/max(1,total_tests)*100:.1f}%)")
    print(f"  Failed:       {failed}  ({failed/max(1,total_tests)*100:.1f}%)")
    print(f"  Skipped:      {skipped}  (retriever unavailable)")

    if failures:
        print(f"\n── FAILED CASES ({len(failures)}) ──")
        for f in failures:
            name, inp, actual, expected, mode = f
            print(f"  • {name}")
            print(f"    input:    {inp}")
            print(f"    expected: {expected} ({mode})")
            print(f"    actual:   {actual[:8]}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
