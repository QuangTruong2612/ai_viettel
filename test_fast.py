"""R39 — Fast hard-case test (NO embedding/model loading, only dict lookups).

Tests direct dictionary _icd_vn_to_codes match + structure lookups without BGE-M3.
Run time: <2 seconds.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"F:\AI_VIETTEL").resolve()))

# Avoid loading embedding model at all
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


total = passed = failed = skipped = 0
failures: list[tuple[str, str, list, list, str]] = []


def test(name: str, fn, input_text: str, expected: list[str], mode: str = "icd",
         must_contain: bool = True):
    """Run fn(text) → list; check expected."""
    global total, passed, failed
    total += 1
    try:
        result = fn(input_text)
    except Exception as e:
        failed += 1
        failures.append((name, f"EXCEPTION: {e}", [], expected, mode))
        print(f"  ✗ FAIL: {name} (exception)")
        return
    actual = set(result or [])
    if must_contain:
        ok = any(c in actual for c in expected)
    else:
        ok = not any(c in actual for c in expected)
    if ok:
        passed += 1
        print(f"  ✓ PASS: {name:55s} → {sorted(actual)[:3]}")
    else:
        failed += 1
        failures.append((name, input_text, sorted(actual), expected,
                         "contain" if must_contain else "NOT"))
        print(f"  ✗ FAIL: {name}")
        print(f"      input={input_text}")
        print(f"      expected={expected} ({'contain' if must_contain else 'NOT'})")
        print(f"      actual={sorted(actual)[:5]}")


def main():
    # Test 1: Direct dictionary lookup from icd_rag.py
    print("=" * 70)
    print("ICD — DIRECT DICT LOOKUP (no embedding)")
    print("=" * 70)

    import importlib
    import src.icd_rag as icd_module

    from src.icd_rag import ICDRetriever
    print("[init] ICDRetriever(use_hybrid=False) — skips BGE-M3 model load...")
    try:
        icd = ICDRetriever(use_hybrid=False)
    except Exception as e:
        print(f"[init] Failed: {e}")
        icd = None

    # Get the dict regardless of embedding init
    direct_map = {}
    if icd is not None and hasattr(icd, "_icd_vn_to_codes"):
        direct_map = {k.lower(): v for k, v in icd._icd_vn_to_codes.items()}
    print(f"[init] Direct ICD map: {len(direct_map)} entries")

    # ── Test the dict lookup directly (FAST, no model needed) ──
    print()
    print("── Group 1: Direct dictionary hits ──")
    direct_cases = [
        ("thiếu men g6pd", ["D55.0"]),  # The regression test for G6PD bug
        ("Thiếu men G6PD", ["D55.0"]),
        ("thiếu máu tan huyết", ["D58"]),
        ("vàng da sơ sinh", ["P59"]),
        ("tăng huyết áp", ["I10"]),
        ("THA", ["I10"]),
        ("đái tháo đường", ["E11"]),
        ("ĐTĐ", ["E11"]),
        ("đái tháo đường type 2", ["E11"]),
        ("nhồi máu cơ tim", ["I21"]),
        ("NMCT", ["I21"]),
        ("hen phế quản", ["J45"]),
        ("hen suyễn", ["J45"]),
        ("COPD", ["J44"]),
        ("CKD", ["N18"]),
        ("suy thận", ["N19"]),
        ("suy thận cấp", ["N17"]),
        ("suy thận mạn", ["N18"]),
        ("viêm phổi", ["J18"]),
        ("viêm dạ dày", ["K29"]),
        ("loét dạ dày", ["K25"]),
        ("GERD", ["K21"]),
        ("trào ngược dạ dày", ["K21"]),
        ("IBS", ["K58"]),
        ("hội chứng ruột kích thích", ["K58"]),
        ("migraine", ["G43"]),
        ("đau nửa đầu", ["G43"]),
        ("parkinson", ["G20"]),
        ("Parkinson", ["G20"]),
        ("alzheimer", ["G30"]),
        ("trầm cảm", ["F32"]),
        ("lo âu", ["F41"]),
        ("mất ngủ", ["G47.0"]),
        ("zona", ["B02"]),
        ("ghẻ", ["B86"]),
        ("nấm da", ["B35"]),
        ("mày đay", ["L50"]),
        ("eczema", ["L20"]),
        ("viêm da cơ địa", ["L20"]),
        ("viêm kết mạc", ["H10"]),
        ("đau mắt đỏ", ["H10"]),
        ("viêm tai giữa", ["H66"]),
        ("viêm họng", ["J02"]),
        ("viêm amidan", ["J03"]),
        ("viêm xoang", ["J32"]),
        ("thoái hóa khớp", ["M15"]),
        ("thoát vị đĩa đệm", ["M51"]),
        ("loãng xương", ["M80"]),
        ("gãy xương", ["S72"]),
        ("gãy cổ xương đùi", ["S72"]),
        ("sỏi thận", ["N20"]),
        ("nhiễm trùng tiết niệu", ["N39.0"]),
        ("viêm bàng quang", ["N30"]),
        ("viêm thận", ["N05"]),
        ("viêm bể thận", ["N10"]),
        ("viêm đường tiết niệu", ["N39.0"]),
        ("u ác tính phổi", ["C34"]),
        ("ung thư phổi", ["C34"]),
        ("ung thư vú", ["C50"]),
        ("ung thư gan", ["C22"]),
        ("ung thư dạ dày", ["C16"]),
        ("ung thư đại tràng", ["C18"]),
        ("ung thư trực tràng", ["C20"]),
        ("ung thư não", ["C71"]),
        ("u não", ["C71"]),
        ("di căn não", ["C79.3"]),
        ("di căn gan", ["C78.7"]),
        ("di căn xương", ["C79.5"]),
        ("nhồi máu cơ tim có st chênh lên", ["I21"]),
        ("đau thắt ngực", ["I20"]),
        ("suy tim", ["I50"]),
        ("rung nhĩ", ["I48"]),
        ("ngoại tâm thu thất", ["I49.3"]),
        ("ngoại tâm thu nhĩ", ["I49.1"]),
        ("hở van hai lá", ["I34.0"]),
        ("tắc mạch", ["I82"]),
        ("tai biến mạch máu não", ["I63"]),
        ("đột quỵ", ["I63"]),
        ("viêm gan b", ["B16"]),
        ("viêm gan c", ["B17.1"]),
        ("xơ gan", ["K74"]),
    ]
    for text, expected in direct_cases:
        result = direct_map.get(text.lower().strip(), [])
        ok = any(e in set(result) for e in expected) if result else False
        if ok:
            global passed, total
            passed += 1
            total += 1

    # Now use proper counter
    total_now = total
    passed_now = passed

    print()
    print("═" * 70)
    print(f"DIRECT MAP: {len(direct_map)} entries / {len(direct_cases)} cases")
    print("═" * 70)
    print()
    direct_pass = 0
    direct_fail = 0
    direct_misses = []
    for text, expected in direct_cases:
        result = direct_map.get(text.lower().strip(), [])
        if not result:
            direct_misses.append((text, expected))
            continue
        ok = any(e in set(result) for e in expected)
        if ok:
            direct_pass += 1
        else:
            direct_fail += 1
    print(f"  Direct map PASS: {direct_pass}/{len(direct_cases)}")
    print(f"  Direct map FAIL: {direct_fail}")
    print(f"  Direct map MISS (no entry): {len(direct_misses)}")
    if direct_fail or direct_misses:
        print()
        print("── FAILURES + MISSES ──")
        for text, exp in direct_cases:
            result = direct_map.get(text.lower().strip(), [])
            if not result:
                print(f"  MISS: {text!r} → expected {exp}")
            elif not any(e in set(result) for e in exp):
                print(f"  FAIL: {text!r} → got {result}, expected {exp}")
    if direct_misses:
        print("\n── MISSES (need to be added to _icd_vn_to_codes) ──")
        for text, exp in direct_misses[:20]:
            print(f"    {text!r} → expected {exp}")

    # Test 2: Filter rejection (Q-code for blood diseases)
    print()
    print("─" * 70)
    print("ICD — FILTER tests (must_NOT contain wrong code)")
    print("─" * 70)

    if icd is not None:
        # Use lookup() to test integration
        test("G6PD must NOT be Q55.0",
             lambda t: icd.lookup(t), "thiếu men g6pd", ["Q55.0", "M08.1"], must_contain=False)
        test("thiếu máu must NOT be Q55",
             lambda t: icd.lookup(t), "thiếu máu", ["Q55"], must_contain=False)
        test("Adult RA must NOT be Q",
             lambda t: icd.lookup(t), "viêm khớp dạng thấp", ["Q"], must_contain=False)
        test("THA must NOT have Q",
             lambda t: icd.lookup(t), "THA", ["Q"], must_contain=False)

    print()
    print("═" * 70)
    print("SUMMARY (DIRECT MAP)")
    print("═" * 70)
    print(f"  Tests:                {len(direct_cases)}")
    print(f"  Direct map HIT:       {direct_pass + direct_fail}")
    print(f"    └ PASS (correct):    {direct_pass}")
    print(f"    └ FAIL (wrong code): {direct_fail}")
    print(f"  Direct map MISS:      {len(direct_misses)} (entries to add)")

    return 0 if direct_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
