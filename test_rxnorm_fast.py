"""R39 — Fast RxNorm test (no model load, only direct alias lookup + INN whitelist).

Tests ~50 hard cases cho drug lookup.
Run time: <2 seconds.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"F:\AI_VIETTEL").resolve()))


def main():
    # ── Test 1: Direct alias map (brand → INN) ──
    from src.rxnorm_rag import _DRUG_ALIASES
    print("=" * 70)
    print(f"RxNorm — DIRECT ALIAS TEST (loaded {len(_DRUG_ALIASES)} aliases)")
    print("=" * 70)

    # ── Group A: Common brand → INN ──
    brand_tests = [
        ("crestor", "rosuvastatin"),   # Already in dict?
        ("Crestor", "rosuvastatin"),
        ("tylenol", "acetaminophen"),
        ("Tylenol", "acetaminophen"),
        ("augmentin", "amoxicillin"),
        ("Augmentin", "amoxicillin"),
        ("lasix", "furosemide"),
        ("Lasix", "furosemide"),
        ("lipitor", "atorvastatin"),
        ("Lipitor", "atorvastatin"),
        ("plavix", "clopidogrel"),
        ("Plavix", "clopidogrel"),
        ("nexium", "esomeprazole"),
        ("Nexium", "esomeprazole"),
        ("ventolin", "albuterol"),
        ("Ventolin", "albuterol"),
        ("panadol", "acetaminophen"),
        ("Panadol", "acetaminophen"),
        ("glucophage", "metformin"),
        ("Glucophage", "metformin"),
        ("zithromax", "azithromycin"),
        ("Zithromax", "azithromycin"),
        ("advil", "ibuprofen"),
        ("Advil", "ibuprofen"),
        ("voltaren", "diclofenac"),
        ("Voltaren", "diclofenac"),
        ("toradol", "ketorolac"),
        ("Toradol", "ketorolac"),
        ("zofran", "ondansetron"),
        ("Zofran", "ondansetron"),
        ("combivent", "ipratropium"),  # ipratropium/albuterol combo
        ("Combivent", "ipratropium"),
        ("zocor", "simvastatin"),
        ("Zocor", "simvastatin"),
    ]

    pass_cnt = 0
    fail_cnt = 0
    miss_cnt = 0
    failures = []
    for brand, expected_inn in brand_tests:
        result = _DRUG_ALIASES.get(brand.lower())
        if result is None:
            miss_cnt += 1
            failures.append(("MISS", brand, expected_inn))
        elif expected_inn.lower() in str(result).lower():
            pass_cnt += 1
        else:
            fail_cnt += 1
            failures.append(("FAIL", brand, f"got {result}, expected inn={expected_inn}"))

    print(f"\n── Brand → INN: {len(brand_tests)} cases ──")
    print(f"  PASS: {pass_cnt}")
    print(f"  FAIL (got but wrong inn): {fail_cnt}")
    print(f"  MISS (no entry): {miss_cnt}")
    if failures:
        print("\n  Details:")
        for kind, brand, info in failures[:20]:
            print(f"    {kind}: {brand!r} → {info}")

    # ── Group B: Common generic drugs ──
    print()
    print("── Group B: Common generic drugs ──")
    generic_tests = [
        "paracetamol", "aspirin", "amoxicillin", "metformin", "insulin",
        "furosemide", "ibuprofen", "prednisolone", "metoprolol",
        "amlodipine", "atenolol", "trimetazidine", "methotrexate",
        "acetaminophen", "diclofenac",
    ]
    g_pass = 0
    g_miss = 0
    for drug in generic_tests:
        # Generic drug names ARE in the alias map as brand→generic
        # If not in _DRUG_ALIASES, may still be in INN whitelist
        in_alias = drug.lower() in _DRUG_ALIASES
        # _DRUG_INN_WHITELIST check
        from src.rxnorm_rag import _DRUG_INN_WHITELIST
        in_inn = drug.lower() in _DRUG_INN_WHITELIST
        if in_alias or in_inn:
            g_pass += 1
        else:
            g_miss += 1
            print(f"  MISS generic: {drug!r}")

    print(f"  Generic PASS: {g_pass}/{len(generic_tests)}")
    print(f"  Generic MISS: {g_miss}")

    # ── Group C: Resistance context (must be filtered out by lookup) ──
    print()
    print("── Group C: Resistance / drug-class context (filter at lookup) ──")
    from src.rxnorm_rag import RxNormRetriever
    print("[init] Loading RxNormRetriever (L1 dict only)...")
    rx = RxNormRetriever()
    pass_resist = 0
    for drug_with_context, should_be_empty in [
        ("kháng methicillin", True),
        ("Enterococcus kháng vancomycin", True),
        ("vi khuẩn kháng thuốc", False),
        ("resistance vancomycin", True),
        ("kháng sinh nhóm penicillin", True),
        ("methicillin", False),  # actually a real drug (not in resistance context)
        ("bicarbonate (lab value)", True),  # lab context
    ]:
        try:
            result = rx.lookup(drug_with_context)
            if should_be_empty:
                if not result:
                    pass_resist += 1
                    print(f"  ✓ REJECTED: {drug_with_context!r}")
                else:
                    print(f"  ✗ NOT REJECTED: {drug_with_context!r} → {result}")
            else:
                if result:
                    pass_resist += 1
                    print(f"  ✓ ACCEPTED: {drug_with_context!r} → {result[:3]}")
                else:
                    print(f"  ✗ REJECTED wrongly: {drug_with_context!r}")
        except Exception as e:
            print(f"  ERROR: {drug_with_context!r} → {e}")
    print(f"  Context-filter: {pass_resist} pass")

    print()
    print("=" * 70)
    print(f"TOTAL: brand pass={pass_cnt}, fail={fail_cnt}, miss={miss_cnt}")
    print(f"       generic miss={g_miss}")
    print(f"       context filter={pass_resist} pass")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
