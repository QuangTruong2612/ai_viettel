"""R39: Apply noise filter to existing output files (cleanup LLM false positives)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"F:\AI_VIETTEL")))
from src.postprocess import _is_extra_noise_entity, _is_chatbot_artifact


def main():
    output = Path(r"F:\AI_VIETTEL\output")
    stats = {
        "files": 0,
        "noise_dropped": 0,
        "chatbot_dropped": 0,
        "kept": 0,
    }

    for f in sorted(output.glob("*.json"), key=lambda p: int(p.stem)):
        data = json.load(open(f, encoding="utf-8"))
        if not data:
            stats["files"] += 1
            continue
        new_data = []
        for ent in data:
            text = str(ent.get("text", "")).strip()
            if _is_chatbot_artifact(text):
                stats["chatbot_dropped"] += 1
                continue
            if _is_extra_noise_entity(text):
                stats["noise_dropped"] += 1
                continue
            new_data.append(ent)
        if len(new_data) != len(data):
            with open(f, "w", encoding="utf-8") as out:
                json.dump(new_data, out, ensure_ascii=False, indent=2)
        else:
            new_data = data
        stats["kept"] += len(new_data)
        stats["files"] += 1

    print("=" * 60)
    print("R39 NOISE FILTER APPLIED")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k:20s} = {v}")


if __name__ == "__main__":
    main()
