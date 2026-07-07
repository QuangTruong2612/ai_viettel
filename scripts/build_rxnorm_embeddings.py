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
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size encoding (giảm nếu OOM, vd 16 cho 8GB VRAM)")
    parser.add_argument("--precision", choices=["float32", "float16"],
                        default="float16",
                        help="Float precision (float16 tiết kiệm 50% RAM, đủ cho cosine)")
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

    # Detect GPU
    device = "cpu"
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            total_mem_gb = torch.cuda.get_device_properties(0).total_mem / 1024**3
            logger.info("Device: cuda (%s, %.1f GB)", gpu_name, total_mem_gb)
            # Auto-suggest batch size cho 232k entries
            if total_mem_gb < 16 and args.batch_size > 32:
                logger.warning(
                    "GPU < 16GB với 232k entries → recommend --batch-size 16-32 để tránh OOM"
                )
    except ImportError:
        pass

    logger.info("Đang load model %s...", args.model)
    t0 = time.time()
    model = SentenceTransformer(args.model, device=device)
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
    logger.info("Encoding %d names (batch_size=%d, precision=%s)...",
                len(names), args.batch_size, args.precision)
    t0 = time.time()
    try:
        embeddings = model.encode(
            names,
            batch_size=args.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower() and device == "cuda":
            logger.error(
                "OOM! 232k entries cần nhiều RAM. Hãy thử:\n"
                "  --batch-size 16 (hoặc 8 cho 8GB VRAM)\n"
                "  --precision float16 (đã mặc định)"
            )
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass
        raise
    logger.info("Encoding done: shape=%r, time=%.1fs",
                embeddings.shape, time.time() - t0)

    # Convert to float16
    if args.precision == "float16" and embeddings.dtype != np.float16:
        embeddings = embeddings.astype(np.float16)
        logger.info("Converted sang float16")

    # 3. Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, embeddings)
    logger.info("Saved → %s (%.1f MB)",
                args.output.name,
                args.output.stat().st_size / (1024 * 1024))
    return 0


if __name__ == "__main__":
    sys.exit(main())
