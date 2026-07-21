"""Re-attach symptom ICD candidates vào existing output files mà không cần chạy lại inference.

Script này:
1. Đọc symptom_icd_map.json
2. Với mỗi output file, scan tất cả TRIỆU_CHỨNG entities
3. Nếu candidates == [] → tra cứu symptom map → gán ICD code
4. Ghi lại file output

Usage:
    uv run python scripts/reattach_symptom_icd.py --output output/
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def load_symptom_map(data_dir: Path) -> dict[str, str]:
    """Load symptom -> ICD mapping."""
    path = data_dir / "symptom_icd_map.json"
    if not path.exists():
        print(f"[ERROR] {path} không tồn tại")
        return {}
    cfg = json.loads(path.read_text(encoding="utf-8"))
    return {k.lower().strip(): v for k, v in cfg.get("_vn_symptom_to_icd", {}).items()}


def _normalize(text: str) -> str:
    """Normalize text for matching."""
    return re.sub(r"\s+", " ", text.lower().strip())


def reattach(output_dir: Path, data_dir: Path, dry_run: bool = False) -> None:
    """Re-attach symptom ICD codes to existing output files."""
    sym_map = load_symptom_map(data_dir)
    if not sym_map:
        print("[ERROR] Empty symptom map, abort.")
        return

    print(f"[INFO] Loaded {len(sym_map)} symptom mappings")

    output_files = sorted(
        [f for f in output_dir.glob("*.json") if f.stem.isdigit()],
        key=lambda x: int(x.stem),
    )
    print(f"[INFO] Processing {len(output_files)} output files...")

    total_updated = 0
    total_entities = 0

    for fpath in output_files:
        try:
            entities = json.loads(fpath.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[WARN] {fpath.name}: parse error {exc}")
            continue

        if not isinstance(entities, list):
            continue

        changed = False
        for ent in entities:
            if not isinstance(ent, dict):
                continue
            if ent.get("type") != "TRIỆU_CHỨNG":
                continue
            
            cands = ent.get("candidates", [])
            if cands:  # Already has candidates, skip
                continue

            text = ent.get("text", "")
            norm = _normalize(text)
            
            # Direct lookup
            icd = sym_map.get(norm)
            
            # If not found, try prefix matching (e.g. "khó thở khi nằm" → "khó thở")
            if not icd:
                for key, code in sym_map.items():
                    if norm.startswith(key) or key.startswith(norm):
                        icd = code
                        break
            
            if icd:
                ent["candidates"] = [icd]
                changed = True
                total_updated += 1

        total_entities += sum(1 for e in entities if isinstance(e, dict) and e.get("type") == "TRIỆU_CHỨNG")

        if changed and not dry_run:
            fpath.write_text(
                json.dumps(entities, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    print(f"[DONE] Updated {total_updated} TRIỆU_CHỨNG entities (from {total_entities} total)")
    if dry_run:
        print("[DRY RUN] No files written.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    reattach(args.output, args.data, args.dry_run)


if __name__ == "__main__":
    main()
