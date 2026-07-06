"""Debug script: test RxNorm lookup chi tiết cho từng drug trong smoke test.

Chạy: uv run python scripts/debug_rxnorm.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rxnorm_rag import (
    RxNormIndex,
    _http_rxnorm_search,
    _drug_query_tokens,
    _normalize,
    _DRUG_NAME_VARIANTS,
    _strip_paren_keep_dose,
)

# Drugs from smoke test (entity text - matches input)
DRUGS = [
    "amlodipine 10 mg po daily",
    "aspirin 81 mg po daily",
    "metoprolol succinate xl 50 mg po daily",
    "guaifenesin ml po q6h:prn",
    "nystatin oral suspension 5 ml po qid:prn",
    "acetaminophen 325-650 mg po q6h:prn",
    "pravastatin 40 mg po daily",
    "docusate sodium 100 mg po bid",
    "senna 8.6 mg po bid:prn",
    "clonazepam 0.5 mg po qam:prn",
    "clonazepam 1.5 mg po qhs",
]

EXPECTED = {
    "amlodipine 10 mg po daily": ["308135"],
    "aspirin 81 mg po daily": ["243670"],
    "metoprolol succinate xl 50 mg po daily": ["866436"],
    "guaifenesin ml po q6h:prn": ["392085"],
    "nystatin oral suspension 5 ml po qid:prn": ["7597"],
    "acetaminophen 325-650 mg po q6h:prn": ["313782"],
    "pravastatin 40 mg po daily": ["904475"],
    "docusate sodium 100 mg po bid": ["1099279"],
    "senna 8.6 mg po bid:prn": ["312935"],
    "clonazepam 0.5 mg po qam:prn": ["197527"],
    "clonazepam 1.5 mg po qhs": ["197528"],
}


def test_drug(drug_text: str, expected: list[str]) -> dict:
    """Trace 1 drug through pipeline."""
    print(f"\n{'=' * 70}")
    print(f"TEST: {drug_text!r}")
    print(f"Expected: {expected}")
    print(f"{'=' * 70}")

    # Step 1: Strip parentheticals (Fix 4)
    stripped = __import__("re").sub(
        r"\(([^)]*)\)", _strip_paren_keep_dose, drug_text
    )
    print(f"\n1. After paren strip: {stripped!r}")

    # Step 2: _normalize
    norm = _normalize(drug_text)
    print(f"2. _normalize: {norm!r}")

    # Step 3: _drug_query_tokens
    tokens = _drug_query_tokens(drug_text)
    print(f"3. _drug_query_tokens: {tokens}")

    # Step 4: NIH API call
    print(f"\n4. NIH API call (this may take 5-10s)...")
    t0 = time.time()
    api_result = _http_rxnorm_search(drug_text)
    elapsed = time.time() - t0
    print(f"   API result: {api_result}")
    print(f"   Time: {elapsed:.2f}s")

    # Subset match
    expected_set = set(expected)
    api_set = set(api_result)
    if expected_set.issubset(api_set) or api_set.issubset(expected_set):
        status = "✅ PASS"
    else:
        status = "❌ FAIL"

    print(f"\n{status}")
    if expected_set.issubset(api_set):
        print(f"   ✓ Expected codes ({expected}) are subset of API result")
    elif api_set.issubset(expected_set):
        print(f"   ✓ API result is subset of expected")
    elif api_set:
        print(f"   Expected: {expected_set}")
        print(f"   Got: {api_set}")
        print(f"   Missing from API: {expected_set - api_set}")
        print(f"   Extra in API: {api_set - expected_set}")
    else:
        print(f"   API returned 0 candidates")

    return {
        "drug": drug_text,
        "expected": expected,
        "api_result": api_result,
        "pass": status.startswith("✅"),
        "time": elapsed,
    }


def main():
    print(f"Testing {len(DRUGS)} drugs against NIH RxNorm API...")
    print(f"Cache: rxnorm_api_cache.json (will use cached results if available)")

    results = []
    for drug in DRUGS:
        result = test_drug(drug, EXPECTED.get(drug, []))
        results.append(result)
        time.sleep(0.1)  # be nice to NIH API

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    passed = sum(1 for r in results if r["pass"])
    failed = len(results) - passed
    print(f"Total: {len(results)}")
    print(f"Passed (subset match): {passed}")
    print(f"Failed: {failed}")

    print(f"\nFailed drugs:")
    for r in results:
        if not r["pass"]:
            print(f"  - {r['drug']}: got {r['api_result']}")

    print(f"\nDetailed:")
    for r in results:
        marker = "✓" if r["pass"] else "✗"
        print(f"  {marker} {r['drug'][:50]:50} → {r['api_result']} ({r['time']:.1f}s)")


if __name__ == "__main__":
    main()