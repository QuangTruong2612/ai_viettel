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
        default=DATA_DIR / "icd10.jsonl",
        help="Đường dẫn file JSONL (WHO ICD-10 2019 VN+EN, mới) hoặc JSON (BYT cũ)",
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
        default=32,
        help="Batch size cho encoding (giảm nếu OOM trên GPU yếu, vd 16 cho 8GB VRAM)",
    )
    parser.add_argument(
        "--precision",
        choices=["float32", "float16"],
        default="float16",
        help="Float precision (float16 tiết kiệm 50% RAM, OK cho cosine similarity)",
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
    # 1. Đọc data
    # ------------------------------------------------------------------ #
    logger.info("Bắt đầu đọc dữ liệu từ %s...", jsonl_path.name)
    descriptions: list[str] = []
    codes: list[str] = []

    def _extract_field(row: dict, fmt: str) -> tuple[str, str]:
        """Trích code + desc từ row theo format (BYT mới, BYT cũ, JSONL mới WHO, hoặc JSONL cũ)."""
        if fmt == "byt":
            return row.get("Mã", "").strip(), row.get("Tên bệnh", "").strip()
        elif fmt == "byt_old":
            return row.get("Mã bệnh", "").strip(), row.get("Tên bệnh gốc", "").strip()
        elif fmt == "jsonl_who":
            # icd10.jsonl mới (WHO ICD-10 2019 VN translation): code + desc_vi + desc_en
            return row.get("code", "").strip(), row.get("desc_vi", "").strip()
        else:  # jsonl cũ (EN only)
            return row.get("code", "").strip(), row.get("desc_en", "").strip()

    if jsonl_path.suffix.lower() == ".json":
        # BYT format
        with jsonl_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        sample = data[0] if data else {}
        is_byt = "Mã" in sample and "Tên bệnh" in sample
        is_byt_old = "Mã bệnh" in sample
        fmt = "byt" if is_byt else ("byt_old" if is_byt_old else "byt")
        logger.info("Format: %s", fmt)
        for row in data:
            code, desc = _extract_field(row, fmt)
            if code and desc:
                # Thêm " | Nhóm bệnh" vào desc để có context phân biệt
                nhom = row.get("Nhóm bệnh", "") or row.get("Tên nhóm", "")
                if nhom and nhom != desc:
                    desc = f"{desc} | {nhom.strip()}"
                codes.append(code)
                descriptions.append(desc)
    else:
        # JSONL format — detect WHO mới (desc_vi+desc_en) vs cũ (desc_en only)
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Auto-detect: nếu có desc_vi → format mới WHO
                is_who = "desc_vi" in row
                fmt = "jsonl_who" if is_who else "jsonl"
                code, desc = _extract_field(row, fmt)
                if code and desc:
                    # WHO mới: thêm desc_en để hybrid search tốt hơn
                    if is_who:
                        desc_en = row.get("desc_en", "").strip()
                        if desc_en and desc_en != desc:
                            desc = f"{desc} | {desc_en}"
                    codes.append(code)
                    descriptions.append(desc)

    n = len(descriptions)
    logger.info("Đã nạp %d bản ghi ICD-10.", n)
    if n == 0:
        logger.error("File rỗng hoặc sai format (không tìm thấy code + desc).")
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
            # PyTorch API: total_memory (was total_mem in older versions)
            total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info("Device: cuda (%s, %.1f GB)", gpu_name, total_mem_gb)
            # Auto-suggest batch size nếu GPU yếu
            if total_mem_gb < 12 and args.batch_size > 32:
                logger.warning(
                    "GPU < 12GB — recommend --batch-size 16-32 để tránh OOM"
                )
        else:
            logger.warning("CUDA không khả dụng → chạy CPU (chậm hơn 10-20x)")
    except ImportError:
        logger.warning("torch không khả dụng → mặc định device=cpu")

    t0 = time.time()
    # R43 (2026-07-14): Support local model path via env var để tránh
    # download chậm từ HF khi chạy trên Kaggle (cache không persistent).
    import os
    model_path = os.environ.get("BGE_M3_PATH", "BAAI/bge-m3")
    if os.path.isdir(model_path):
        logger.info("Loading BGE-M3 từ LOCAL path: %s", model_path)
    else:
        logger.info("Loading BGE-M3 từ HuggingFace: %s (sẽ download ~2.3GB)", model_path)
    model = SentenceTransformer(model_path, device=device)
    logger.info("Load model xong trong %.2fs", time.time() - t0)

    # ------------------------------------------------------------------ #
    # 3. Encode (batch processing + OOM fallback hint)
    # ------------------------------------------------------------------ #
    logger.info("Encoding %d mô tả (batch_size=%d, precision=%s)...",
                n, args.batch_size, args.precision)
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
                "OOM trên GPU! Hãy thử: --batch-size 16 (hoặc 8 cho 8GB VRAM) "
                "+ --precision float16"
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

    # Convert to float16 nếu user chọn (tiết kiệm 50% disk + RAM)
    if args.precision == "float16" and embeddings.dtype != np.float16:
        embeddings = embeddings.astype(np.float16)
        logger.info("Converted sang float16 (%.1f MB → %.1f MB)",
                    embeddings.nbytes / 1024**2, embeddings.nbytes / 1024**2)

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