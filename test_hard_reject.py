"""R39: Verify hard_reject_icd works on simulated input."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"F:\AI_VIETTEL")))


def main():
    from src.postprocess import (
        _apply_hard_reject_icd,
        _HARD_REJECT_ICD,
    )
    print(f"Hard reject table size: {len(_HARD_REJECT_ICD)}")
    print()

    # Simulate bad entities from file 1
    test_entities = [
        # Enzyme deficiency should NOT have Q55
        {"text": "thiếu men G6PD", "type": "CHẨN_ĐOÁN", "candidates": ["D55.0", "Q55.0"]},
        {"text": "Thiếu men G6PD", "type": "CHẨN_ĐOÁN", "candidates": ["D55.0", "Q55.0"]},
        {"text": "G6PD", "type": "CHẨN_ĐOÁN", "candidates": ["D55.0", "Q55.0"]},
        # Myocarditis should NOT have B33.2
        {"text": "viêm tim", "type": "CHẨN_ĐOÁN", "candidates": ["I40.9", "B33.2"]},
        # Phình mạch vành should NOT have G07
        {"text": "phình giãn động mạch vành", "type": "CHẨN_ĐOÁN", "candidates": ["I25.4", "G07"]},
        # Thiếu máu should NOT have Q codes
        {"text": "thiếu máu tan huyết", "type": "CHẨN_ĐOÁN", "candidates": ["D58", "Q55"]},
        # Kawasaki should NOT have I77
        {"text": "bệnh Kawasaki", "type": "CHẨN_ĐOÁN", "candidates": ["M30.3", "I77"]},
        # Diabetes type 2 should NOT have E10
        {"text": "đái tháo đường type 2", "type": "CHẨN_ĐOÁN", "candidates": ["E11", "E10"]},
        # Headache should NOT have I63
        {"text": "đau đầu", "type": "CHẨN_ĐOÁN", "candidates": ["R51", "I63"]},
        # Should be unaffected (no entries in blacklist)
        {"text": "tăng huyết áp", "type": "CHẨN_ĐOÁN", "candidates": ["I10", "I11"]},
    ]

    # Test 1: Verify bad codes get filtered
    filtered = _apply_hard_reject_icd(test_entities)
    print("=== TEST: Hard reject ICD ===")
    for e, original in zip(filtered, test_entities):
        original_cands = original.get("candidates", [])
        new_cands = e.get("candidates", [])
        if original_cands != new_cands:
            print(f"  FILTERED: '{e['text']}': {original_cands} -> {new_cands}")
        else:
            print(f"  UNCHANGED: '{e['text']}': {new_cands}")

    # Test 2: Verify all Q55 removed from G6PD
    print()
    print("=== TEST: G6PD Q55 removal ===")
    count = 0
    for e in test_entities:
        if "thiếu men g6pd" in e["text"].lower() or e["text"].lower() == "g6pd":
            for c in e.get("candidates", []):
                if c.startswith("Q"):
                    print(f"  FAIL: '{e['text']}' still has Q code: {c}")
                    count += 1
    if count == 0:
        print("  ✓ ALL Q55 codes removed from G6PD entities")


if __name__ == "__main__":
    main()
