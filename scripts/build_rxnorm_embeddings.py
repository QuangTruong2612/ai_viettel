"""Build embeddings cho RxNorm data.

Input: data/rxnorm.jsonl (~232k entries, mỗi dòng 1 drug có field `name`).
Output: data/rxnorm_embeddings.npy (matrix N x 1024 với BGE-M3).

Sử dụng model BAAI/bge-m3 (multilingual — hỗ trợ cả EN và VN).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("build_rxnorm_embeddings")

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_DIR / "data" / "rxnorm.jsonl"
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "rxnorm_embeddings.npy"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build BGE-M3 embeddings cho RxNorm names"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="JSONL input (mỗi dòng có field 'name')")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Output .npy embeddings file")
    parser.add_argument("--model", default="BAAI/bge-m3",
                        help="SentenceTransformer model")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit số rows đọc (0 = tất cả)")
    parser.add_argument("--skip-if-exists", action="store_true",
                        help="Skip nếu output đã tồn tại")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild kể cả khi output đã tồn tại")
    args = parser.parse_args()

    if not args.input.exists():
        logger.error("Không tìm thấy input file: %s", args.input)
        return 1
    if args.output.exists() and args.skip_if_exists and not args.force:
        logger.info("%s đã tồn tại → skip. Dùng --force để rebuild.",
                    args.output.name)
        return 0

    # 1. Load model + data
    import json
    from sentence_transformers import SentenceTransformer  # type: ignore
    import numpy as np

    logger.info("Đang load model %s...", args.model)
    t0 = time.time()
    model = SentenceTransformer(args.model)
    logger.info("Model loaded (%.1fs)", time.time() - t0)

    logger.info("Đang đọc %s...", args.input.name)
    t0 = time.time()
    import json as _json
    rxcuis: list[str] = []
    names: list[str] = []
    with args.input.open("r", encoding="utf-8") as f:
        for line in f:
            if args.limit > 0 and len(rxcuis) >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                row = _json.loads(line)
            except _json.JSONDecodeError:
                continue
            rxcui = str(row.get("rxcui", "")).strip()
            name = str(row.get("name", "")).strip()
            if rxcui and name:
                rxcuis.append(rxcui)
                names.append(name)
    logger.info("Loaded %d names (%.1fs)", len(names), time.time() - t0)

    # 2. Generate embeddings
    logger.info("Encoding %d names (batch_size=%d)...", len(names), args.batch_size)
    t0 = time.time()
    embeddings = model.encode(
        names,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    logger.info("Encoding done: shape=%r, time=%.1fs",
                embeddings.shape, time.time() - t0)

    # 3. Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, embeddings)
    logger.info("Saved → %s (%.1f MB)",
                args.output.name,
                args.output.stat().st_size / (1024 * 1024))
    return 0


if __name__ == "__main__":
    sys.exit(main())
