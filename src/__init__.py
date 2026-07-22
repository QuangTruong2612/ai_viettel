"""Package init cho src/ — Medical Information Extraction pipeline.

Modules:
- llm_client: OpenAI-compatible wrapper cho Ollama + JSON parser.
- prompts: SYSTEM_PROMPT (NER rules + few-shot loader).
- rxnorm_rag: RxNorm retrieval (vector + BM25 + exact match hybrid).
- icd_rag: ICD-10 retrieval (vector + BM25 hybrid trên BYT data VN).
- postprocess: Validate, dedupe, fix positions, populate candidates.
- inference: Main driver — orchestrate pipeline offline.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional


def resolve_data_path(filename: str, *, env_var: Optional[str] = None) -> Path:
    """Resolve path tới 1 file trong data/, có Kaggle fallback.

    Tìm file theo thứ tự ưu tiên:
      1. Explicit env var (nếu set + exists) — vd ``ICD_EMBEDDINGS_PATH=...``
      2. ``DATA_DIR / filename`` — local (project_root/data/)
      3. ``/kaggle/input/ai-viettel-embeddings/<filename>`` — Kaggle dataset mount
      4. ``/kaggle/input/ai-viettel-data/<filename>`` — alternative name
      5. ``/kaggle/input/ai_viettel/<filename>`` — alternative name

    Args:
        filename: tên file (vd "icd10_embeddings.npy").
        env_var: optional env var name override (vd "ICD_EMBEDDINGS_PATH").

    Returns:
        Path tới file đầu tiên tồn tại. Nếu không tìm thấy → trả về
        ``DATA_DIR / filename`` (file có thể không tồn tại — caller sẽ
        fallback về build-from-scratch).

    Note:
        Designed để vừa work local (Kaggle-mount paths không tồn tại →
        skip qua DATA_DIR) vừa work trên Kaggle (mount path tồn tại →
        dùng luôn, không cần copy sang /kaggle/working).
    """
    candidates: list[Path] = []

    # 1. Explicit env var override
    if env_var:
        env_path = os.environ.get(env_var)
        if env_path:
            candidates.append(Path(env_path))

    # 2. Project local data dir (relative to caller — caller passes DATA_DIR
    #    via closure or we use a best-effort guess from CWD).
    cwd_data = Path.cwd() / "data" / filename
    if cwd_data.exists():
        candidates.append(cwd_data)

    # 3. Kaggle dataset mounts (multiple naming conventions)
    kaggle_roots = [
        Path("/kaggle/input/ai-viettel-embeddings"),
        Path("/kaggle/input/ai-viettel-data"),
        Path("/kaggle/input/ai_viettel"),
        Path("/kaggle/input/ai-viettel-data/data"),
        Path("/kaggle/input/ai_viettel/data"),
        Path("/kaggle/input/datasets/nguynvnquangtrng/data-embedding/embedding")
    ]
    for root in kaggle_roots:
        candidates.append(root / filename)

    # Return first existing path
    for c in candidates:
        if c.exists():
            return c

    # Fallback: trả về local data path (có thể không tồn tại — caller xử lý)
    return cwd_data


def copy_to_local_data(src: Path, *, local_data_dir: Path) -> Path:
    """Copy file từ Kaggle mount sang local data dir nếu cần.

    Hữu ích khi code paths khác expect file ở local DATA_DIR. Để trống
    DATA_DIR/<filename> nếu đã tồn tại (skip).

    Args:
        src: source path (vd /kaggle/input/.../icd10_embeddings.npy).
        local_data_dir: destination (vd /kaggle/working/.../data).

    Returns:
        Path tới file ở local_data_dir (file vừa copy hoặc đã có sẵn).
    """
    dst = local_data_dir / src.name
    if dst.exists():
        return dst
    local_data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)
    return dst


def get_writable_cache_path(filename: str) -> Path:
    """Trả về path **writable** để cache file (vd rebuild embeddings).

    Trên Kaggle, ``/kaggle/input/`` là read-only → không thể ``np.save`` ở đó.
    Hàm này:
      1. Nếu đang ở Kaggle (``/kaggle/working`` tồn tại + writable) →
         trả về ``/kaggle/working/<filename>``
      2. Nếu không → trả về ``Path.cwd() / "data" / filename`` (local)

    Args:
        filename: tên file (vd "icd10_embeddings.npy").

    Returns:
        Path tới writable location để save file mới.
    """
    kaggle_writable = Path("/kaggle/working")
    if kaggle_writable.exists() and os.access(kaggle_writable, os.W_OK):
        return kaggle_writable / filename
    # Fallback local
    cwd_data = Path.cwd() / "data"
    cwd_data.mkdir(parents=True, exist_ok=True)
    return cwd_data / filename
