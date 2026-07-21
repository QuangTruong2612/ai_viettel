"""Deep audit of NER and Assertions across the 100 output files."""

import json
import re
import glob
from pathlib import Path
from collections import Counter, defaultdict

inp_dir = Path("data/input") if Path("data/input").exists() else Path("input")

historical_ents = []
negated_ents = []
family_ents = []

overly_long_text = []
suspicious_types = []

for f in sorted(glob.glob("output/*.json"), key=lambda x: int(Path(x).stem)):
    rec_id = int(Path(f).stem)
    inp_path = inp_dir / f"{rec_id}.txt"
    if not inp_path.exists():
        inp_path = inp_dir / f"{rec_id}.json"
    if not inp_path.exists():
        continue
    
    text = inp_path.read_text(encoding="utf-8")
    ents = json.loads(Path(f).read_text(encoding="utf-8"))

    for e in ents:
        txt = e.get("text", "")
        t = e.get("type", "")
        pos = e.get("position", [0, 0])
        assertions = e.get("assertions", [])

        # Overly long text (> 6 words)
        if len(txt.split()) > 6:
            overly_long_text.append((rec_id, t, txt))

        # Check assertions
        if "isHistorical" in assertions:
            context = text[max(0, pos[0]-50):min(len(text), pos[1]+50)].replace("\n", " ")
            historical_ents.append((rec_id, t, txt, context))
        if "isNegated" in assertions:
            context = text[max(0, pos[0]-50):min(len(text), pos[1]+50)].replace("\n", " ")
            negated_ents.append((rec_id, t, txt, context))
        if "isFamily" in assertions:
            context = text[max(0, pos[0]-50):min(len(text), pos[1]+50)].replace("\n", " ")
            family_ents.append((rec_id, t, txt, context))

print(f"Total overly long entities (>6 words): {len(overly_long_text)}")
print(f"Total isHistorical entities: {len(historical_ents)}")
print(f"Total isNegated entities: {len(negated_ents)}")
print(f"Total isFamily entities: {len(family_ents)}")

print("\n--- SAMPLE OVERLY LONG ENTITIES (first 10) ---")
for r, t, txt in overly_long_text[:10]:
    print(f"  [{r}] ({t}) {txt}")

print("\n--- SAMPLE isHistorical ENTITIES (first 10) ---")
for r, t, txt, ctx in historical_ents[:10]:
    print(f"  [{r}] ({t}) '{txt}' | CTX: ...{ctx}...")

print("\n--- SAMPLE isNegated ENTITIES (first 10) ---")
for r, t, txt, ctx in negated_ents[:10]:
    print(f"  [{r}] ({t}) '{txt}' | CTX: ...{ctx}...")

print("\n--- SAMPLE isFamily ENTITIES ---")
for r, t, txt, ctx in family_ents[:10]:
    print(f"  [{r}] ({t}) '{txt}' | CTX: ...{ctx}...")
