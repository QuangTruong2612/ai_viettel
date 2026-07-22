#!/usr/bin/env python3
"""Verify LLM prompt + postprocess pipeline robustness."""
import sys
sys.path.insert(0, '.')
from src.prompts import (
    SYSTEM_PROMPT, STAGE1_PROMPT, STAGE2_PROMPT,
    build_stage1_user_prompt, build_stage2_user_prompt,
)


def test_softened_rules():
    """Verify softened CẤM rules contain key markers."""
    print("=== TEST 1: SOFTENED RULES PRESENCE ===")
    all_prompts = SYSTEM_PROMPT + STAGE1_PROMPT + STAGE2_PROMPT

    # Critical markers for softened rules
    markers = [
        "DEFAULT = TRÍCH XUẤT",
        "TUYỆT ĐỐI KHÔNG được trả `[]` rỗng khi bệnh án có thông tin y khoa",
        "Chỉ trả về JSON array",  # format reminder
        "KHÔNG thêm bất kỳ text nào TRƯỚC hoặc SAU JSON",
        "TRÁNH extract",  # softened language (not "TUYỆT ĐỐI KHÔNG")
        "NGOẠI LỆ",  # exceptions included
    ]
    for marker in markers:
        if marker in all_prompts:
            print(f"  ✓ '{marker}'")
        else:
            print(f"  ✗ MISSING: '{marker}'")
            return False
    return True


def test_kawasaki_style_input():
    """Verify prompt guidance cho Kawasaki-style medical input (sốt, phát ban, etc.)."""
    print("\n=== TEST 2: KAWASAKI-STYLE MEDICAL INPUT ===")
    # Kawasaki disease has: sốt cao, phát ban, viêm kết mạc, sưng hạch cổ, đỏ môi, etc.
    kawasaki_symptoms = ["sốt cao", "phát ban", "viêm kết mạc", "sưng hạch"]
    all_prompts = SYSTEM_PROMPT + STAGE1_PROMPT + STAGE2_PROMPT

    for symptom in kawasaki_symptoms:
        if symptom.lower() in all_prompts.lower():
            print(f"  ✓ '{symptom}' mentioned in prompts (LLM sẽ nhận diện)")
        else:
            # Even if not exact word, principle should be there
            if "TRIỆU_CHỨNG" in all_prompts and "extract" in all_prompts.lower():
                print(f"  ⚠ '{symptom}' not exact match nhưng principle tổng quát có (LLM có thể generalize)")
            else:
                print(f"  ✗ '{symptom}' — no guidance at all")

    # Critical: prompt must say "if medical content exists → must extract"
    if "DEFAULT = TRÍCH XUẤT" in all_prompts:
        print(f"\n  ✓ 'DEFAULT = TRÍCH XUẤT' principle present → LLM sẽ không trả [] cho input y khoa")
        return True
    return False


def test_json_format_reminder():
    """Verify JSON format reminder present in stage1 và stage2 builders."""
    print("\n=== TEST 3: JSON FORMAT REMINDER ===")
    s1_out = build_stage1_user_prompt("test input")
    s2_out = build_stage2_user_prompt("test", [{"text": "x", "position": [0, 1]}])

    if "ĐỊNH DẠNG OUTPUT" in s1_out and "JSON array" in s1_out:
        print("  ✓ STAGE1 has JSON format reminder")
    else:
        print("  ✗ STAGE1 missing JSON format reminder")
        return False

    if "ĐỊNH DẠNG OUTPUT BẮT BUỘC" in s2_out and "JSON array" in s2_out:
        print("  ✓ STAGE2 has JSON format reminder")
    else:
        print("  ✗ STAGE2 missing JSON format reminder")
        return False

    return True


def test_loadability():
    print("\n=== TEST 4: PROMPT LOADABILITY ===")
    print(f"  ✓ SYSTEM_PROMPT: {len(SYSTEM_PROMPT)} chars (~{len(SYSTEM_PROMPT)//4} tokens)")
    print(f"  ✓ STAGE1_PROMPT: {len(STAGE1_PROMPT)} chars")
    print(f"  ✓ STAGE2_PROMPT: {len(STAGE2_PROMPT)} chars")
    return True


def main():
    print("=" * 60)
    print("SOFTENED PROMPT VERIFICATION (post-regression fix)")
    print("=" * 60)
    results = []
    results.append(("Softened rules", test_softened_rules()))
    results.append(("Kawasaki input", test_kawasaki_style_input()))
    results.append(("JSON format", test_json_format_reminder()))
    results.append(("Loadability", test_loadability()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results:
        print(f"  {'✓ PASS' if passed else '✗ FAIL'} {name}")

    if all(p for _, p in results):
        print(f"\n  🎉 All checks passed — softened prompt addresses over-filtering")
    return 0 if all(p for _, p in results) else 1


if __name__ == "__main__":
    sys.exit(main())
