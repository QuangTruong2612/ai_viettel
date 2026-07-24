"""Test new postprocess fixes by re-running them on existing outputs."""
import json
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(r"F:\AI_VIETTEL")))
from src.postprocess import (
    _normalize_type_to_ascii,
    _enforce_position_strict,
    _is_chatbot_artifact,
    _is_overly_long_narrative,
)


def main():
    output_dir = Path(r"F:\AI_VIETTEL\output")
    input_dir = Path(r"F:\AI_VIETTEL\input")

    # Test type normalization
    print("=" * 60)
    print("TEST 1: Type normalization (diacritics → ASCII)")
    print("=" * 60)
    test_types = [
        ("THUỐC", "THUOC"),
        ("CHẨN_ĐOÁN", "CHAN_DOAN"),
        ("TRIỆU_CHỨNG", "TRIEU_CHUNG"),
        ("TÊN_XÉT_NGHIỆM", "TEN_XET_NGHIEM"),
        ("KẾT_QUẢ_XÉT_NGHIỆM", "KET_QUA_XET_NGHIEM"),
    ]
    for vn, expected in test_types:
        result = _normalize_type_to_ascii(vn)
        status = "✓" if result == expected else "✗"
        print(f"  {status} {vn!r} → {result!r} (expected {expected!r})")

    # Test chatbot detection
    print()
    print("=" * 60)
    print("TEST 2: Chatbot artifact detection")
    print("=" * 60)
    chatbot_texts = [
        "Cảm ơn bạn đã gửi câu hỏi",
        "Hy vọng thông tin này sẽ giúp ích",
        "Nếu bạn có thêm câu hỏi, vui lòng liên hệ bác sĩ",
        "Viêm phổi",  # Should be False
        "Tăng huyết áp",  # Should be False
    ]
    for text in chatbot_texts:
        result = _is_chatbot_artifact(text)
        status = "✓ DROP" if result else "✓ KEEP"
        print(f"  {status}: {text!r}")

    # Test position enforcement
    print()
    print("=" * 60)
    print("TEST 3: Position enforcement (file 26 off-by-one)")
    print("=" * 60)
    inp = (input_dir / "26.txt").read_text(encoding="utf-8")
    cases = [
        # File 26 [0]: text="BỆNH MẠCH VÀNH", position [0, 14]
        {"text": "BỆNH MẠCH VÀNH", "position": [0, 14], "type": "CHAN_DOAN"},
        # File 30 [0]: text="methadone", position [132, 141]
    ]
    inp30 = (input_dir / "30.txt").read_text(encoding="utf-8")
    cases.append({"text": "methadone", "position": [132, 141], "type": "THUOC"})
    cases.append({"text": "chủ quan sốt", "position": [204, 216], "type": "TRIEU_CHUNG"})

    for case in cases:
        text = case["text"]
        pos = case["position"]
        type_ = case["type"]
        recovered = _enforce_position_strict(inp if case is cases[0] else inp30, dict(case))
        if recovered is None:
            print(f"  ✗ DROPPED: {text!r} {pos}")
        else:
            new_pos = recovered["position"]
            actual_at_pos = (inp if case is cases[0] else inp30)[new_pos[0]:new_pos[1]]
            match = "✓" if actual_at_pos == recovered["text"] else "?"
            print(f"  {match} {text!r} → text={recovered['text']!r}, pos={new_pos}, actual_at_pos={actual_at_pos!r}")

    # Re-process all output files (type normalization + position enforcement + drop)
    print()
    print("=" * 60)
    print("TEST 4: Re-process all 100 output files")
    print("=" * 60)
    fixed_files = 0
    types_normalized = 0
    positions_fixed = 0
    positions_dropped = 0
    chatbot_dropped = 0
    narrative_dropped = 0
    total_entities_before = 0
    total_entities_after = 0

    for fout in sorted(output_dir.glob("*.json"), key=lambda p: int(p.stem)):
        fid = fout.stem
        data = json.load(open(fout, encoding="utf-8"))
        if not data:
            continue
        inp_file = input_dir / f"{fid}.txt"
        if not inp_file.exists():
            continue
        inp = inp_file.read_text(encoding="utf-8")
        total_entities_before += len(data)

        new_data = []
        for ent in data:
            # 1) Normalize type
            new_type = _normalize_type_to_ascii(ent.get("type", ""))
            if new_type != ent.get("type", ""):
                types_normalized += 1
            ent["type"] = new_type
            text = str(ent.get("text", "")).strip()

            # 2) Drop overly long narrative
            if _is_overly_long_narrative(text, new_type):
                narrative_dropped += 1
                continue

            # 3) Drop chatbot artifacts
            if _is_chatbot_artifact(text):
                chatbot_dropped += 1
                continue

            # 4) Enforce position
            old_pos = ent.get("position", [])
            recovered = _enforce_position_strict(inp, ent)
            if recovered is None:
                positions_dropped += 1
                continue
            if recovered.get("position") != old_pos:
                positions_fixed += 1
            new_data.append(recovered)

        total_entities_after += len(new_data)
        # Only write if changed
        if len(new_data) != len(data) or any(e["type"] != d["type"] for e, d in zip(new_data, data)):
            fixed_files += 1

    print(f"  Files updated: {fixed_files}")
    print(f"  Types normalized: {types_normalized} (diacritics → ASCII)")
    print(f"  Positions fixed: {positions_fixed}")
    print(f"  Positions dropped (couldn't recover): {positions_dropped}")
    print(f"  Chatbot artifacts dropped: {chatbot_dropped}")
    print(f"  Narrative too long dropped: {narrative_dropped}")
    print(f"  Before: {total_entities_before} entities / After: {total_entities_after}")


if __name__ == "__main__":
    main()
