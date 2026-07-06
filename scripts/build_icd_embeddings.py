import os
import json
import time
import logging
from pathlib import Path
import numpy as np

# Thêm logging cơ bản
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_icd_embeddings")

# Xác định các thư mục dự án
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"


def main():
    jsonl_path = DATA_DIR / "icd10.jsonl"
    npy_path = DATA_DIR / "icd10_embeddings.npy"

    if not jsonl_path.exists():
        logger.error("Không tìm thấy tệp dữ liệu %s", jsonl_path)
        return 1

    # 1. Đọc các mô tả từ file jsonl
    logger.info("Bắt đầu đọc dữ liệu từ %s...", jsonl_path.name)
    descriptions = []
    codes = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
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

    total_records = len(descriptions)
    logger.info("Đã nạp thành công %d bản ghi ICD-10.", total_records)

    # 2. Khởi tạo mô hình nhúng bge-m3
    logger.info("Đang khởi tạo mô hình BAAI/bge-m3 (Hugging Face / local cache)...")
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as exc:
        logger.error("Chưa cài đặt sentence-transformers! Vui lòng cài đặt trước.")
        return 2

    # Tự động chọn cuda nếu có, để tăng tốc tối đa
    t0 = time.time()
    model = SentenceTransformer("BAAI/bge-m3")
    logger.info(
        "Tải mô hình thành công trong %.2fs. Thiết bị sử dụng: %s",
        time.time() - t0,
        model.device,
    )

    # 3. Mã hóa hàng loạt (Encoding)
    logger.info("Bắt đầu sinh vector nhúng cho %d mô tả...", total_records)
    t0 = time.time()

    # model.encode hỗ trợ show_progress_bar hiển thị thanh tiến trình trực quan
    embeddings = model.encode(
        descriptions,
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=True,  # Chuẩn hóa L2 giúp tính dot-product tương đương cosine similarity
        convert_to_numpy=True,
    )

    elapsed = time.time() - t0
    logger.info(
        "Sinh vector nhúng hoàn tất trong %.2fs (Tốc độ: %.2f dòng/giây)",
        elapsed,
        total_records / elapsed,
    )
    logger.info("Ma trận embeddings có kích thước: %r", embeddings.shape)

    # 4. Lưu ra tệp NumPy (.npy)
    logger.info("Đang lưu ma trận ra %s...", npy_path)
    np.save(npy_path, embeddings)
    logger.info(
        "Hoàn tất! Kích thước file: %.2f MB", npy_path.stat().st_size / (1024 * 1024)
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
