"""Regenerate ICD aliases end-to-end: split baseline + LLM generate + merge into icd_index.

MỤC ĐÍCH
--------
Script "all-in-one" cho workflow Kaggle/local để bổ sung ICD alias tiếng Việt
bằng LLM, fix vấn đề mining baseline chỉ có desc_vi gốc (1 alias/mã, không
phải alias thông thường bác sĩ hay dùng).

PIPELINE
--------
1. Tách mining baseline ra file riêng (idempotent - skip nếu đã có).
2. Reset icd_aliases.json = {} để generate_icd_aliases.py chạy đầy đủ
   (không bị skip vì file đã có 15,732 keys từ baseline).
3. Gọi generate_icd_aliases.py qua subprocess (~1,050 batch LLM call,
   ~45-60 phút trên T4, có checkpoint + resume tự động).
4. Gộp baseline + LLM alias vào icd_aliases_merged.json (dedupe).
5. Backup icd_index.json (nếu chưa có .bak).
6. Gọi merge_icd_aliases.py để thêm alias vào icd_index["exact"] (Tier-1).
7. Verify chất lượng: in các alias phổ biến (THA, ĐTĐ, viêm phổi, NMCT...).

CÁCH DÙNG
---------
    # Full pipeline (default — chạy từ đầu đến merge)
    python scripts/regenerate_icd_aliases.py

    # Tùy chỉnh data dir (vd Kaggle mount)
    python scripts/regenerate_icd_aliases.py --data-dir /kaggle/working/ai-viettel/data

    # Đổi model / base URL (vd LM Studio :1234)
    python scripts/regenerate_icd_aliases.py --base-url http://127.0.0.1:1234/v1 --model qwen2.5:7b

    # Resume: nếu generate bị ngắt giữa chừng, chạy lại lệnh cũ sẽ tự động tiếp tục
    python scripts/regenerate_icd_aliases.py

    # Skip generate (dùng LLM alias đã có sẵn trong icd_aliases.json từ lần chạy trước)
    python scripts/regenerate_icd_aliases.py --skip-generate

    # Chỉ generate, không merge vào icd_index.json (để inspect trước)
    python scripts/regenerate_icd_aliases.py --skip-merge

    # Không tách baseline ra file riêng (giữ nguyên icd_aliases.json cũ)
    python scripts/regenerate_icd_aliases.py --keep-baseline

IDEMPOTENT / RESUME
-------------------
Mọi bước đều check điều kiện trước khi thực hiện (file đã tồn tại, content
đã đúng, v.v.) nên chạy lại nhiều lần an toàn. Generate có checkpoint ghi
mỗi 20 batch → ngắt phiên rồi chạy lại sẽ tiếp tục từ batch chưa xong.

RUNTIME (Kaggle T4)
-------------------
- Generate: 45-60 phút (1,050 batch × 15 mã/batch, qwen2.5:7b)
- Merge: <1 phút
- Tổng: ~1 giờ
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("regenerate_icd_aliases")

PROJECT_DIR = Path(__file__).resolve().parents[1]
GENERATE_SCRIPT = PROJECT_DIR / "scripts" / "generate_icd_aliases.py"
MERGE_SCRIPT = PROJECT_DIR / "scripts" / "merge_icd_aliases.py"


# ---------------------------------------------------------------------- #
# Pipeline steps
# ---------------------------------------------------------------------- #


def split_baseline(data_dir: Path) -> Path:
    """Tách mining baseline ra file riêng (idempotent).

    - Backup icd_aliases.json → icd_aliases.json.mining_backup (nếu chưa có).
    - Move icd_aliases.json → icd_aliases_baseline.json (nếu chưa có).
    """
    src = data_dir / "icd_aliases.json"
    backup = data_dir / "icd_aliases.json.mining_backup"
    baseline = data_dir / "icd_aliases_baseline.json"

    if not src.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {src} — cần có icd_aliases.json (do build_mining_index.py tạo) để bắt đầu."
        )

    if not backup.exists():
        shutil.copy2(src, backup)
        logger.info("✓ Backup baseline: %s", backup.name)
    else:
        logger.info("✓ Backup baseline đã có: %s", backup.name)

    if not baseline.exists():
        shutil.move(str(src), str(baseline))
        logger.info("✓ Tách baseline ra: %s", baseline.name)
    else:
        logger.info("✓ Baseline file đã có: %s", baseline.name)

    return baseline


def reset_aliases(data_dir: Path) -> None:
    """Tạo icd_aliases.json RỖNG để generate_icd_aliases.py chạy đầy đủ.

    Lý do: nếu file đã có 15,732 keys từ baseline, script sẽ skip hết (xem
    comment trong main() bên dưới).
    """
    target = data_dir / "icd_aliases.json"
    target.write_text("{}", encoding="utf-8")
    logger.info("✓ Reset %s = {} (rỗng, sẵn sàng generate đầy đủ)", target.name)


def run_generate(data_dir: Path, base_url: str, model: str, batch_size: int, shuffle: bool) -> None:
    """Chạy generate_icd_aliases.py qua subprocess (45-60 phút, 1,050 batch)."""
    if not GENERATE_SCRIPT.exists():
        raise FileNotFoundError(f"Không tìm thấy {GENERATE_SCRIPT}")

    cmd = [
        sys.executable, str(GENERATE_SCRIPT),
        "--input", str(data_dir / "icd10.jsonl"),
        "--output", str(data_dir / "icd_aliases.json"),
        "--batch-size", str(batch_size),
        "--base-url", base_url,
        "--model", model,
    ]
    if shuffle:
        cmd.append("--shuffle")

    logger.info("=" * 60)
    logger.info("Generate ICD aliases (≈ 1,050 batch × %d mã/batch)", batch_size)
    logger.info("Model: %s @ %s", model, base_url)
    logger.info("Estimated runtime: 45-60 phút trên Kaggle T4")
    logger.info("=" * 60)

    t0 = time.time()
    subprocess.run(cmd, check=True)
    elapsed = time.time() - t0
    logger.info("✓ Generate xong trong %.1f phút", elapsed / 60)


def merge_aliases(data_dir: Path) -> Path:
    """Gộp baseline + LLM alias → icd_aliases_merged.json (dedupe).

    Returns path tới file merged.
    """
    baseline_path = data_dir / "icd_aliases_baseline.json"
    llm_path = data_dir / "icd_aliases.json"
    out_path = data_dir / "icd_aliases_merged.json"

    baseline: dict[str, list[str]] = json.loads(baseline_path.read_text(encoding="utf-8"))
    llm: dict[str, list[str]] = json.loads(llm_path.read_text(encoding="utf-8"))

    final: dict[str, list[str]] = {}
    for code, aliases in baseline.items():
        final[code] = list(aliases)

    n_added = 0
    for code, aliases in llm.items():
        if code not in final:
            final[code] = list(aliases)
            n_added += len(aliases)
            continue
        existing = set(final[code])
        for a in aliases:
            if a and a not in existing:
                final[code].append(a)
                existing.add(a)
                n_added += 1

    out_path.write_text(
        json.dumps(final, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    logger.info(
        "✓ Merged: %s (%d mã, +%d alias LLM thêm vào baseline)",
        out_path.name, len(final), n_added,
    )
    return out_path


def run_merge(data_dir: Path, merged_path: Path) -> None:
    """Backup icd_index.json + gọi merge_icd_aliases.py qua subprocess."""
    index = data_dir / "icd_index.json"
    if not index.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {index} — chạy scripts/build_icd_index.py trước."
        )
    if not merged_path.exists():
        raise FileNotFoundError(f"Không tìm thấy {merged_path}")

    backup = data_dir / "icd_index.json.bak"
    if not backup.exists():
        shutil.copy2(index, backup)
        logger.info("✓ Backup: %s", backup.name)
    else:
        logger.info("✓ Backup icd_index.json.bak đã có")

    if not MERGE_SCRIPT.exists():
        raise FileNotFoundError(f"Không tìm thấy {MERGE_SCRIPT}")

    cmd = [
        sys.executable, str(MERGE_SCRIPT),
        "--index", str(index),
        "--aliases", str(merged_path),
        "--output", str(index),
    ]
    logger.info("=" * 60)
    logger.info("Merge aliases vào icd_index.json[\"exact\"]")
    logger.info("=" * 60)
    subprocess.run(cmd, check=True)
    logger.info("✓ Merge xong")


def verify(data_dir: Path) -> None:
    """Verify chất lượng alias sau merge — in các alias phổ biến để kiểm tra."""
    index = data_dir / "icd_index.json"
    idx = json.loads(index.read_text(encoding="utf-8"))
    exact = idx.get("exact", {})

    logger.info("=" * 60)
    logger.info("Verify chất lượng alias")
    logger.info("icd_index.json[\"exact\"] giờ có %d keys", len(exact))
    logger.info("=" * 60)

    test_aliases = [
        ("THA", "Tăng huyết áp"),
        ("ĐTĐ", "Đái tháo đường"),
        ("viêm phổi", "Viêm phổi"),
        ("VP", "Viêm phổi (viết tắt)"),
        ("NMCT", "Nhồi máu cơ tim"),
        ("COPD", "Bệnh phổi tắc nghẽn mạn"),
        ("CKD", "Bệnh thận mạn"),
        ("đột quỵ", "Tai biến mạch máu não"),
        ("tiểu đường", "Đái tháo đường (đồng nghĩa)"),
    ]
    n_found = 0
    for alias, desc in test_aliases:
        key = alias.lower()
        if key in exact:
            codes = exact[key]
            logger.info("  ✓ %-15s [%s] → %s", alias, desc, codes)
            n_found += 1
        else:
            logger.warning("  ✗ %-15s [%s] MISSING (LLM có thể không generate alias này)", alias, desc)
    logger.info("\nFound %d/%d alias phổ biến trong icd_index.json", n_found, len(test_aliases))


# ---------------------------------------------------------------------- #
# Driver
# ---------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT_DIR / "data",
        help="Thư mục chứa data (icd10.jsonl, icd_aliases.json, icd_index.json). "
             "Mặc định: <project>/data",
    )
    parser.add_argument(
        "--base-url", type=str, default="http://127.0.0.1:11434/v1",
        help="OpenAI-compatible endpoint (Ollama mặc định :11434, LM Studio :1234)",
    )
    parser.add_argument(
        "--model", type=str, default="qwen2.5:7b-instruct",
        help="Tên model Ollama/LM Studio (vd qwen2.5:7b, qwen3.5:9b)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=15,
        help="Số mã ICD / lần gọi LLM (mặc định 15). Giảm xuống 8 nếu hay timeout.",
    )
    parser.add_argument(
        "--shuffle", action="store_true",
        help="Xáo trộn thứ tự mã trước khi generate (đa dạng sample qua các lần chạy)",
    )
    parser.add_argument(
        "--skip-generate", action="store_true",
        help="Bỏ qua bước generate (dùng LLM alias đã có sẵn trong icd_aliases.json)",
    )
    parser.add_argument(
        "--skip-merge", action="store_true",
        help="Bỏ qua bước merge vào icd_index.json (chỉ generate + gộp file merged)",
    )
    parser.add_argument(
        "--keep-baseline", action="store_true",
        help="Không tách baseline ra file riêng (giữ nguyên icd_aliases.json hiện tại)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    if not data_dir.exists():
        logger.error("Không tìm thấy data dir: %s", data_dir)
        return 1

    logger.info("=" * 60)
    logger.info("Regenerate ICD Aliases")
    logger.info("Data dir: %s", data_dir)
    logger.info("Model:    %s @ %s", args.model, args.base_url)
    logger.info("=" * 60)

    # === Bước 1: Tách baseline ra file riêng (trừ khi --keep-baseline) ===
    if not args.keep_baseline and not args.skip_generate:
        split_baseline(data_dir)
        reset_aliases(data_dir)

    # === Bước 2: Generate LLM aliases (~45-60 phút, có resume) ===
    if args.skip_generate:
        logger.info("--skip-generate: dùng alias LLM đã có sẵn trong icd_aliases.json")
    else:
        run_generate(
            data_dir=data_dir,
            base_url=args.base_url,
            model=args.model,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
        )

    # === Bước 3: Gộp baseline + LLM ===
    merged_path = merge_aliases(data_dir)

    # === Bước 4: Merge vào icd_index.json ===
    if args.skip_merge:
        logger.info("--skip-merge: file merged ở %s (chưa merge vào index)", merged_path)
    else:
        run_merge(data_dir, merged_path)
        verify(data_dir)

    logger.info("=" * 60)
    logger.info("HOÀN TẤT. Bây giờ có thể chạy:")
    logger.info("  python -m src.inference --input data/input --output output --target-ctx 8192")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())