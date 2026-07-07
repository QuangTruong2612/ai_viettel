"""Build ICD-10 embeddings bằng BGE-M3, lưu ra .npy.

Usage:
    uv run python scripts/build_icd_embeddings.py
    uv run python scripts/build_icd_embeddings.py --batch-size 64     # GPU yếu (8GB)
    uv run python scripts/build_icd_embeddings.py --skip-if-exists     # skip nếu file đã có
    uv run python scripts/build_icd_embeddings.py --force             # rebuild kể cả file đã có

Output: data/icd10_embeddings.npy (~280 MB)
- Shape: (71705, 1024) — float32
- L2-normalized (norm ≈ 1.0) → dot product tương đương cosine similarity.

Cải tiến so với bản gốc:
- CLI args (--batch-size, --skip-if-exists, --force) cho flexible GPU.
- Explicit device + log GPU name.
- OOM handling: gợi ý giảm batch size.
- L2 norm validation: cảnh báo nếu vector chưa normalized đúng.
- Reload verification: bắt file corrupt ngay sau khi save.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("build_icd_embeddings")

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Encode ICD-10 descriptions bằng BGE-M3 và lưu ra .npy"
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=DATA_DIR / "DM_ICD10_19_8_BYT.json",
        help="Đường dẫn file JSONL chứa ICD-10 codes + desc_en",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DATA_DIR / "icd10_embeddings.npy",
        help="Đường dẫn file .npy đầu ra",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size cho encoding (giảm nếu OOM trên GPU yếu, vd 64 cho 8GB VRAM)",
    )
    parser.add_argument(
        "--skip-if-exists",
        action="store_true",
        help="Skip nếu file .npy đã tồn tại (dùng --force để rebuild)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild kể cả khi file đã tồn tại (override --skip-if-exists)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip reload + L2 norm verification (chỉ dùng khi debug)",
    )
    args = parser.parse_args()

    jsonl_path: Path = args.jsonl
    npy_path: Path = args.out

    # ------------------------------------------------------------------ #
    # Validate inputs
    # ------------------------------------------------------------------ #
    if not jsonl_path.exists():
        logger.error("Không tìm thấy tệp dữ liệu %s", jsonl_path)
        return 1

    if npy_path.exists() and args.skip_if_exists and not args.force:
        logger.info(
            "File %s đã tồn tại → skip. Dùng --force để rebuild.", npy_path.name
        )
        return 0

    # ------------------------------------------------------------------ #
    # 1. Đọc JSONL
    # ------------------------------------------------------------------ #
    logger.info("Bắt đầu đọc dữ liệu từ %s...", jsonl_path.name)
    descriptions: list[str] = []
    codes: list[str] = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            code = str(row.get("code", "")).strip()
            desc = str(row.get("desc_en", "")).strip()
            if code and desc:
                codes.append(code)
                descriptions.append(desc)

    n = len(descriptions)
    logger.info("Đã nạp %d bản ghi ICD-10.", n)
    if n == 0:
        logger.error("File rỗng hoặc sai format (không tìm thấy code + desc_en).")
        return 3

    # ------------------------------------------------------------------ #
    # 2. Load model với explicit device
    # ------------------------------------------------------------------ #
    logger.info("Đang khởi tạo BAAI/bge-m3...")
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError:
        logger.error(
            "Chưa cài sentence-transformers! Chạy: uv pip install -r requirements.txt"
        )
        return 2

    device = "cpu"
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            logger.info("Device: cuda (%s)", gpu_name)
        else:
            logger.warning("CUDA không khả dụng → chạy CPU (chậm hơn 10-20x)")
    except ImportError:
        logger.warning("torch không khả dụng → mặc định device=cpu")

    t0 = time.time()
    model = SentenceTransformer("BAAI/bge-m3", device=device)
    logger.info("Load model xong trong %.2fs", time.time() - t0)

    # ------------------------------------------------------------------ #
    # 3. Encode (batch processing + OOM fallback hint)
    # ------------------------------------------------------------------ #
    logger.info("Encoding %d mô tả (batch_size=%d)...", n, args.batch_size)
    t0 = time.time()
    try:
        embeddings = model.encode(
            descriptions,
            batch_size=args.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
    except RuntimeError as exc:
        # RuntimeError("CUDA out of memory") thường gặp trên GPU 8GB
        if "out of memory" in str(exc).lower() and device == "cuda":
            logger.error(
                "OOM trên GPU! Hãy thử: --batch-size 64 (hoặc 32 cho 8GB VRAM)."
            )
            try:
                import torch  # type: ignore

                torch.cuda.empty_cache()
            except ImportError:
                pass
        raise
    elapsed = time.time() - t0
    logger.info(
        "Encode xong %.2fs (%.1f dòng/giây). Shape: %r, dtype: %s",
        elapsed,
        n / elapsed,
        embeddings.shape,
        embeddings.dtype,
    )

    # ------------------------------------------------------------------ #
    # 4. Verify L2 norms (sau normalize phải ≈ 1.0)
    # ------------------------------------------------------------------ #
    if not args.no_verify:
        norms = np.linalg.norm(embeddings, axis=1)
        logger.info(
            "L2 norm check: min=%.4f max=%.4f mean=%.4f (expect ~1.0)",
            norms.min(),
            norms.max(),
            norms.mean(),
        )
        if not (0.99 < norms.mean() < 1.01):
            logger.warning(
                "L2 norms KHÔNG khớp 1.0! normalize_embeddings có thể đã tắt. "
                "Pipeline downstream sẽ cho cosine similarity sai."
            )

    # ------------------------------------------------------------------ #
    # 5. Save
    # ------------------------------------------------------------------ #
    logger.info("Lưu ma trận → %s", npy_path)
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, embeddings)
    file_mb = npy_path.stat().st_size / (1024**2)
    logger.info(
        "Done! File: %.2f MB (%d codes × %d-dim × %d bytes)",
        file_mb,
        embeddings.shape[0],
        embeddings.shape[1],
        embeddings.itemsize,
    )

    # ------------------------------------------------------------------ #
    # 6. Reload verification (sanity check file không corrupt)
    # ------------------------------------------------------------------ #
    if not args.no_verify:
        reloaded = np.load(npy_path)
        if reloaded.shape != embeddings.shape:
            logger.error(
                "Shape mismatch sau reload: %r vs %r",
                reloaded.shape,
                embeddings.shape,
            )
            return 4
        if not np.allclose(reloaded, embeddings):
            logger.error("Data mismatch sau reload — file có thể corrupt!")
            return 5
        logger.info("✓ Reload + compare OK (file không corrupt).")

    return 0


if __name__ == "__main__":
    sys.exit(main())