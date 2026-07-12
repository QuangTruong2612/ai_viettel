import json
from pathlib import Path

def build_examples():
    src = Path("data/examples.jsonl")
    s1_path = Path("data/examples_stage1.jsonl")
    s2_path = Path("data/examples_stage2.jsonl")

    lines = [line.strip() for line in src.read_text("utf-8").splitlines() if line.strip()]
    
    s1_entries = []
    s2_entries = []

    for line in lines:
        obj = json.loads(line)
        inp = obj["input"]
        out = obj["output"]

        # Stage 1: only text & position
        s1_out = []
        # Stage 2: mentions list + classification (text, position, type, assertions)
        s2_mentions = []
        s2_out = []

        for ent in out:
            text = ent.get("text", "").strip()
            pos = ent.get("position", [0, 0])
            etype = ent.get("type", "")
            assertions = ent.get("assertions", [])

            s1_out.append({"text": text, "position": pos})
            s2_mentions.append({"text": text, "position": pos})
            s2_out.append({
                "text": text,
                "position": pos,
                "type": etype,
                "assertions": assertions
            })

        s1_entries.append({"input": inp, "output": s1_out})
        s2_entries.append({"input": inp, "mentions": s2_mentions, "output": s2_out})

    # Add 4 negative / narrative-noise demonstrations to Stage 1
    neg_1 = {
        "input": "Bệnh nhân nam 50 tuổi nhập viện vì sốt cao, đau ngực. Tỉnh dậy thấy cháu gái hét lên sợ hãi. Theo đó cô ấy sẽ được phục vụ tốt hơn.",
        "output": [
            {"text": "sốt cao", "position": [33, 40]},
            {"text": "đau ngực", "position": [42, 50]}
        ]
    }
    neg_2 = {
        "input": "Bệnh nhân từng bị ngất xỉu trước khi chuyển tới khoa cấp cứu. Nhận thấy da niêm hồng hào, ăn uống sinh hoạt bình thường. Đang dùng aspirin 81mg.",
        "output": [
            {"text": "ngất xỉu", "position": [18, 26]},
            {"text": "aspirin 81mg", "position": [128, 140]}
        ]
    }
    neg_3 = {
        "input": "Chúng tôi quyết định rằng bệnh nhân cần theo dõi thêm. Chẩn đoán: nhồi máu cơ tim vùng dưới cũ. Chỉ định X-quang ngực và điện tâm đồ.",
        "output": [
            {"text": "nhồi máu cơ tim vùng dưới cũ", "position": [69, 97]},
            {"text": "X-quang ngực", "position": [108, 120]},
            {"text": "điện tâm đồ", "position": [124, 135]}
        ]
    }
    neg_4 = {
        "input": "Bệnh nhân có cảm giác đánh trống ngực và khó thở nhẹ khi gắng sức trong tuần qua. Không ghi nhận triệu chứng bất thường nào khác.",
        "output": [
            {"text": "cảm giác đánh trống ngực", "position": [13, 37]},
            {"text": "khó thở nhẹ khi gắng sức", "position": [41, 65]}
        ]
    }

    s1_entries.extend([neg_1, neg_2, neg_3, neg_4])

    # Also add matching Stage 2 for these 4 entries
    for neg in [neg_1, neg_2, neg_3, neg_4]:
        inp = neg["input"]
        ments = neg["output"]
        s2_out = []
        for m in ments:
            t = m["text"]
            p = m["position"]
            # Classify
            if "aspirin" in t.lower():
                s2_out.append({"text": t, "position": p, "type": "THUỐC", "assertions": []})
            elif "nhồi máu" in t.lower():
                s2_out.append({"text": t, "position": p, "type": "CHẨN_ĐOÁN", "assertions": []})
            elif "sốt" in t.lower() or "đau" in t.lower() or "ngất" in t.lower() or "trống ngực" in t.lower() or "khó thở" in t.lower():
                s2_out.append({"text": t, "position": p, "type": "TRIỆU_CHỨNG", "assertions": []})
            elif "x-quang" in t.lower() or "điện tâm đồ" in t.lower():
                s2_out.append({"text": t, "position": p, "type": "TÊN_XÉT_NGHIỆM", "assertions": []})
        s2_entries.append({"input": inp, "mentions": ments, "output": s2_out})

    s1_path.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in s1_entries) + "\n", "utf-8")
    s2_path.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in s2_entries) + "\n", "utf-8")
    print(f"Generated {len(s1_entries)} S1 examples and {len(s2_entries)} S2 examples.")

if __name__ == "__main__":
    build_examples()
