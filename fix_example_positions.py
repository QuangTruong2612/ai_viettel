"""Fix drifted character positions in data/examples.jsonl.

Only entities whose current ``input[position]`` slice does NOT match their
``text`` are recomputed. Each broken entity is reassigned to the earliest
occurrence of its text that does not overlap an already-used span, honouring
output order — this handles repeated spans (same text N times) and short spans
nested inside longer ones (e.g. "nôn" inside "buồn nôn"). Idempotent: once every
slice matches, a re-run changes nothing.
"""

import json
from pathlib import Path

EXAMPLES_PATH = Path("f:/AI_VIETTEL/data/examples.jsonl")


def _slice_ok(text: str, ent: dict) -> bool:
    a, b = ent["position"]
    return text[a:b].lower() == ent["text"].lower()


def _find_span(text_low: str, span_low: str, used: list[tuple[int, int]]) -> tuple[int, int] | None:
    start = 0
    while True:
        i = text_low.find(span_low, start)
        if i < 0:
            return None
        j = i + len(span_low)
        if not any(i < ub and a < j for a, ub in used):  # no overlap
            return i, j
        start = i + 1


def fix_example(ex: dict) -> int:
    text = ex["input"]
    low = text.lower()
    fixed = 0
    used: list[tuple[int, int]] = [tuple(e["position"]) for e in ex["output"] if _slice_ok(text, e)]
    for ent in ex["output"]:
        if _slice_ok(text, ent):
            continue
        found = _find_span(low, ent["text"].lower(), used)
        if found is None:
            raise ValueError(f"Cannot place {ent['text']!r} in {text!r}")
        a, b = found
        ent["position"] = [a, b]
        ent["text"] = text[a:b]  # keep original casing from input
        used.append((a, b))
        fixed += 1
    return fixed


def main() -> None:
    with EXAMPLES_PATH.open("r", encoding="utf-8") as f:
        exs = [json.loads(line) for line in f if line.strip()]

    total = 0
    for i, ex in enumerate(exs, 1):
        n = fix_example(ex)
        if n:
            total += n
            print(f"  ex{i}: fixed {n} span(s)")

    with EXAMPLES_PATH.open("w", encoding="utf-8") as f:
        for ex in exs:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Done. Fixed {total} span(s) across {len(exs)} examples.")


if __name__ == "__main__":
    main()
