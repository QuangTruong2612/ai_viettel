# AI Race 2026 — Medical Information Extraction (VN)

Pipeline **trích xuất thực thể y khoa** từ hồ sơ bệnh án tiếng Việt và **liên kết mã chuẩn hoá** (`ICD-10`, `RxNorm`). Chạy **hoàn toàn offline** với Ollama + local embeddings (BGE-M3). Hỗ trợ cả **local (CPU/GPU)** và **Kaggle T4x2**.

---

## 🆕 Recent Updates (2026-07-22)

- **Kaggle Dataset support** — `resolve_data_path()` auto-detect embeddings từ `/kaggle/input/...` (xem [Kaggle Deployment](#-kaggle-deployment-t4-x2))
- **`get_writable_cache_path()`** — Save embeddings sang `/kaggle/working/` (tránh read-only `/kaggle/input/`)
- **R37 (2026-07-15)** — Enhanced prompt: BRAND name rule, test abbrev map, abnormal findings explicit, CẤM 8-10, two-pass verification
- **R37 postprocess** — Extended abnormal regex, brand retype, drop dose fragment + drug class
- **3-stage pipeline** — Stage 1 (mentions) → Stage 2 (classify) → Stage 3 (candidate refine) chạy **post-RAG** để tránh hallucination
- **R38 (2026-07-23) — LLM ReRank** — Score-based top-K candidate ranking với hard prompt + full clinical context. Sort candidates theo score 1-10, drop score < 3, configurable top-K via `RERANK_TOP_K=5`.
- **R38 — Q-code filter** — Filter Q00-Q99 (congenital) cho non-congenital entities. Trước đây match "Thiếu men G6PD" → Q55.0 (testis defect), correct là D55.0.
- **R38 — Drug-class generic** — Thêm corticoid/NSAID/vitamin/hormone/antibiotic patterns vào [data/generic_class_stoplist.json](data/generic_class_stoplist.json).
- **R38 — Fuzzy INN ingredient** — Bổ sung `_fuzzy_ingredient_lookup` để cover typo "trimetazidin" → "trimetazidine".
- **R38 — Text+type dedupe** — Dedupe entities cuối cùng theo `(text_lower, type)` để giảm WER explosion từ 14 duplicates của cùng concept.

---

## ✨ Highlights

- 🎯 **5 loại entity y khoa**: `THUỐC`, `CHẨN_ĐOÁN`, `TRIỆU_CHỨNG`, `TÊN_XÉT_NGHIỆM`, `KẾT_QUẢ_XÉT_NGHIỆM`
- 🔍 **NER thông minh theo MỨC ĐỘ ĐẦY ĐỦ**: THUỐC + CHẨN_ĐOÁN giữ nguyên (để lookup candidate chính xác); TRIỆU_CHỨNG minimal (bỏ duration/value)
- 🎯 **Hybrid RAG**: Vector (BGE-M3) + BM25 keyword + Fuzzy cho cả ICD và RxNorm
- 🇻🇳 **Vietnamese ICD data** (`icd10.jsonl` - WHO ICD-10 2019 VN translation, 15,732 codes có cả VN+EN)
- 💊 **RxNorm** lookup VN/EN với BGE-M3 multilingual
- 🏥 **3 assertions**: `isHistorical`, `isNegated`, `isFamily` (context-aware)
- 🚀 **Offline hoàn toàn** — không cần NIH/NLM API
- ☁️ **Kaggle-ready** — auto-detect embeddings từ Kaggle Dataset, handle read-only mounts
- 📊 **3-stage LLM pipeline** — Stage 1 (mentions) → Stage 2 (classify) → Stage 3 (post-RAG candidate refine)

---

## 🚀 Quick Start (5 phút — Local)

```bash
# 1. Setup
git clone https://github.com/QuangTruong2612/ai_viettel.git
cd ai_viettel
uv venv && source .venv/bin/activate   # hoặc .venv\Scripts\activate trên Windows
uv pip install -r requirements.txt

# 2. Cài Ollama + model
#    Tải từ https://ollama.com/download
ollama serve &                          # chạy nền
ollama pull qwen2.5:7b                  # ~4.7 GB, mặc định
# Hoặc:
ollama pull qwen3.5:9b                  # ~5.5 GB, mạnh hơn

# 3. Build indexes (1 lần, ~30 phút)
uv run python scripts/build_rxnorm_index.py        # ~1s, exact match
uv run python scripts/build_rxnorm_embeddings.py   # ~10 min (GPU), BGE-M3
uv run python scripts/build_icd_embeddings.py      # ~10 min (GPU), BGE-M3

# 4. Chạy thử
uv run python -m src.inference --input input --output output --target-ctx 16384 --limit 3
```

Kết quả: 3 files `output/1.json`, `output/2.json`, `output/3.json` chứa entities + ICD codes + RxNorm candidates.

---

## 📂 Cấu trúc thư mục

```
AI_VIETTEL/
├── src/                              # Production code
│   ├── __init__.py                   # Path resolver (Kaggle-aware)
│   ├── prompts.py                    # SYSTEM_PROMPT cho NER + few-shot loader
│   ├── llm_client.py                 # OpenAI-compatible wrapper cho Ollama
│   ├── inference.py                  # Main driver — orchestrate pipeline
│   ├── postprocess.py                # Validate, fix position, populate candidates
│   ├── icd_rag.py                    # ICD-10 RAG (BYT VN data, vector + BM25 hybrid)
│   └── rxnorm_rag.py                 # RxNorm RAG (vector + BM25 + exact match hybrid)
│
├── scripts/                          # Build & utility scripts
│   ├── build_icd_index.py            # Build icd_index.json (exact match)
│   ├── build_icd_embeddings.py       # Build icd10_embeddings.npy (BGE-M3)
│   ├── build_rxnorm_index.py         # Build rxnorm_index.json (exact match)
│   ├── build_rxnorm_embeddings.py    # Build rxnorm_embeddings.npy (BGE-M3)
│   ├── test_inference.py             # Smoke test 1 record + sanity check
│   ├── validate_outputs.py           # Validate output schema
│   └── audit_types_kaggle.py         # Kaggle-ready type audit
│
├── data/                             # Input data + indexes (LOCAL + Git LFS)
│   ├── icd10.jsonl                   # ICD-10 source (WHO 2019 VN+EN, 15,732 codes)
│   ├── DM_ICD10_19_8_BYT.json        # ICD-10 source (BYT VN official, 36,689 codes)
│   ├── rxnorm.jsonl                  # RxNorm source (232k entries EN)
│   ├── examples.jsonl                # Few-shot examples
│   ├── icd10_embeddings.npy          # BGE-M3 matrix ~15.7k x 1024 (~73 MB float16)
│   ├── rxnorm_embeddings.npy         # BGE-M3 matrix ~232k x 1024 (~464 MB float16)
│   └── ...                           # various indexes + caches
│
├── input/                            # Test input files (1.txt, 2.txt, ..., 100.txt)
├── output/                           # Output JSON files
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

---

## 🧠 Kiến trúc Pipeline (3-Stage)

```
┌──────────────────────────────────────────────────────────────┐
│ INPUT: hồ sơ bệnh án tiếng Việt (input/N.txt / input/N.json) │
└──────────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN 1 — Clinical NER Mentions (LLM via Ollama)         │
│ • Preprocess input (strip markdown, N/A, truncate)           │
│ • Section-based chunking (>1500 chars → chunks với overlap)  │
│ • Adaptive few-shot (auto-budget theo context length)        │
│ • Build STAGE1_PROMPT + few-shot messages                    │
│ • LLM call (qwen2.5:7b / qwen3.5:9b / etc.)                 │
│ • JSON parser với retry + recovery                           │
│ Output: Raw mentions (text + type, KHÔNG cần position)       │
└──────────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN 2 — Classify + Position (LLM Stage 2)             │
│ • Align raw mentions → original input text (auto position)   │
│ • Detect duplicate entities → expand multiple positions      │
│ • STAGE2_PROMPT: validate type, normalize entity text        │
│ • LLM refine type + assertions (isHistorical/Negated/Family) │
│ Output: Aligned entities (text, type, position, assertions)  │
└──────────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN 3 — Candidate Linking (Hybrid RAG)                 │
│ • 4-Tier Cascading Candidate Linking:                         │
│   - THUỐC → RxNormRetriever.lookup()                         │
│     * Tier 0: Drug Pre-cleaner (tách route/freq/doseform)    │
│     * Tier 1: INN ↔ USAN & Brand VN dictionary mapping       │
│     * Tier 2: Exact tuple match (ingredient, strength)       │
│     * Tier 3: Hybrid search (BGE-M3 + BM25) & Fuzzy match   │
│   - CHẨN_ĐOÁN → ICDRetriever.lookup()                        │
│     * Tier 0: Clinical prefix & abbreviation dictionary      │
│     * Tier 1: Direct VN-VN exact/substring match (O(1))      │
│     * Tier 2: Context-Enriched Hybrid Search (BGE-M3 + BM25) │
│     * Tier 3: Chapter Restriction & Fuzzy Match              │
└──────────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN 4 — Stage 3 LLM Candidate Refinement (Post-RAG)   │
│ • STAGE3_PROMPT: verify/refine ICD/RxNorm candidates         │
│ • Batch 30 entities / LLM call                               │
│ • Verdict per entity: "ok" / "refine" / "drop"               │
│ • On LLM parse fail: keep RAG candidates (no hallucination)   │
│ • Disabled via env: LLM_DISABLE_STAGE3=1                     │
└──────────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────────┐
│ OUTPUT: output/N.json — entities + ICD codes + RxNorm codes  │
└──────────────────────────────────────────────────────────────┘
```

### Hybrid RAG Architecture (ICD + RxNorm)

```
Query (VN diagnosis or drug text)
              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Tier 0 & 1: Pre-cleaner & Exact Match Dictionary (Fast Path)   │
│  - Strip liều dùng/cách dùng cho Thuốc (_strip_route_freq)      │
│  - Tra từ điển INN ↔ USAN, Brand VN, viết tắt Y khoa VN        │
│  → HIT? Return ngay mã (O(1), độ chính xác 100%, không tốn GPU) │
└─────────────────────────────────────────────────────────────────┘
              ↓ (miss)
┌─────────────────────────────────────────────────────────────────┐
│  Tier 2: Context-Enriched Hybrid Search                         │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Context Enrichment (thêm thuốc/triệu chứng lân cận)      │  │
│  │   ↓                                                       │  │
│  │ Vector (BGE-M3 cosine ≥ threshold)                        │  │
│  │   + BM25 keyword search                                   │  │
│  │   ↓                                                       │  │
│  │ Union & Re-score cosine                                   │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
              ↓ (miss)
┌─────────────────────────────────────────────────────────────────┐
│  Tier 3: Fuzzy Match (rapidfuzz) & Chapter Restriction          │
└─────────────────────────────────────────────────────────────────┘
              ↓
         Return top-K rxcui / ICD codes
```

---

## 🔧 Cài đặt chi tiết

### 1. Python dependencies

```bash
# Tạo venv
uv venv
source .venv/bin/activate   # Linux/macOS
# hoặc
.venv\Scripts\activate      # Windows PowerShell

# Cài packages
uv pip install -r requirements.txt
```

`requirements.txt`:

```
openai>=1.30.0              # OpenAI-compatible API cho Ollama
rapidfuzz>=3.9.0            # Fuzzy match (drug/disease names)
sentence-transformers>=3.0.0 # BGE-M3 embeddings
numpy>=1.24.0               # Matrix operations
jsonschema>=4.21.0           # Output validation
requests>=2.31.0            # HTTP
tqdm>=4.66.0                # Progress bars
rank-bm25>=0.2.2            # BM25 keyword index
```

### 2. Ollama + Model

```bash
# Linux
curl -fsSL https://ollama.com/install.sh | sh

# macOS
brew install ollama

# Windows: tải từ https://ollama.com/download

# Pull model
ollama pull qwen2.5:7b              # Mặc định (nhanh, ~4.7GB)
ollama pull qwen3.5:9b              # Mạnh hơn (~5.5GB)

# Khởi server (chạy nền)
ollama serve
```

Verify:

```bash
curl http://127.0.0.1:11434/v1/models
```

### 3. Build indexes (1 lần, ~30 phút với GPU)

```bash
# ICD exact match index (~1s) — optional, used for fast L1 exact lookup
uv run python scripts/build_icd_index.py
# → data/icd_index.json (~2.5 MB)

# ICD embeddings (BGE-M3, ~5-10 phút GPU) — REQUIRED cho hybrid search
uv run python scripts/build_icd_embeddings.py
# → data/icd10_embeddings.npy (~73 MB float16 / ~193 MB float32)

# RxNorm exact match index (~1s) — REQUIRED cho L1 exact lookup
uv run python scripts/build_rxnorm_index.py
# → data/rxnorm_index.json (~7 MB)

# RxNorm embeddings (BGE-M3, ~10-30 phút GPU) — REQUIRED cho hybrid search
uv run python scripts/build_rxnorm_embeddings.py
# → data/rxnorm_embeddings.npy (~464 MB float16 / ~668 MB float32)
```

> **Cả ICD + RxNorm đều có cùng kiến trúc**: (1) exact match index, (2) BGE-M3 embeddings, (3) hybrid pipeline kết hợp cả hai.

> **Tối ưu cho GPU yếu (Kaggle T4 14.56GB, Colab):**
>
> ```bash
> # Nếu OOM: giảm batch size + dùng float16 (mặc định)
> uv run python scripts/build_icd_embeddings.py --batch-size 16 --precision float16
> uv run python scripts/build_rxnorm_embeddings.py --batch-size 16 --precision float16
> ```
>
> `float16` tiết kiệm 50% RAM, đủ chính xác cho cosine similarity.

### 4. Chạy inference

```bash
# Smoke test (1 record)
uv run python scripts/test_inference.py --out output/smoke_test.json

# Full pipeline
uv run python -m src.inference \
    --input input \
    --output output \
    --workers 1 \
    --target-ctx 16384 \
    --max-few-shot 35
```

**Tham số quan trọng**:

| Tham số           | Mặc định    | Mô tả                                                                   |
| ------------------ | -------------- | ------------------------------------------------------------------------- |
| `--input`        | `data/input` | Thư mục chứa files input (`.txt` hoặc `.json`)                    |
| `--output`       | `output`     | Thư mục output JSON                                                     |
| `--workers`      | `1`          | Số parallel workers (Ollama serve 1 request/lần → nên giữ 1) |
| `--target-ctx`   | `65536`      | Context window của Ollama (giảm 16384 nếu OOM trên Kaggle T4)  |
| `--max-few-shot` | `35`         | Số few-shot example tối đa (auto-budget theo context)                  |
| `--limit`        | `0`          | Giới hạn số records (0 = all)                                          |
| `--no-resume`    | `False`      | Re-process tất cả (default: skip records đã có output)               |
| `--no-two-stage` | `False`      | Single-pass mode (Stage 1+2 gộp 1 call)                              |

### 5. Environment Variables

| Biến                       | Default                    | Mô tả                                                                  |
| ---------------------------- | ---------------------------- | ------------------------------------------------------------------------ |
| `OLLAMA_BASE_URL`          | `http://127.0.0.1:11434/v1` | URL Ollama server                                                    |
| `OLLAMA_MODEL`             | `qwen2.5-7b-instruct`     | Tên model                                                       |
| `OLLAMA_NUM_CTX`           | `65536`                   | Context window (giảm 16384 cho Kaggle T4)                  |
| `OLLAMA_KEEP_ALIVE`        | `0`                       | `0` unload model sau mỗi request (giải phóng VRAM)        |
| `LMSTUDIO_BASE_URL`        | -                          | Override cho LM Studio (port 1234)                                |
| `BGE_M3_PATH`              | `BAAI/bge-m3`             | Local path BGE-M3 (vd `/kaggle/input/datasets/...`)            |
| `ICD_EMBEDDINGS_PATH`      | auto-detect                | Override path tới `icd10_embeddings.npy`                          |
| `RXNORM_EMBEDDINGS_PATH`   | auto-detect                | Override path tới `rxnorm_embeddings.npy`                        |
| `LLM_DISABLE_STAGE3`       | `0`                       | Set `1` để skip Stage 3 LLM refine (tiết kiệm ~30% time)        |
| `LLM_DISABLE_RERANK`       | `0`                       | Set `1` để skip R38 LLM ReRank (score-based top-K)             |
| `RERANK_TOP_K`            | `5`                       | Số candidates giữ lại sau ReRank (1-10)                       |
| `HF_HUB_OFFLINE`           | `0`                       | Set `1` để không download từ HuggingFace (cần local model)  |
| `TRANSFORMERS_OFFLINE`     | `0`                       | Tương tự HF_HUB_OFFLINE                                              |

### 6. Validate output

```bash
uv run python scripts/validate_outputs.py --input output/
```

Output:

```
✅ 1.json (12 entities, 4 historical)
✅ 2.json (8 entities)
...
```

---

## ☁️ Kaggle Deployment (T4 x2)

### Setup từ đầu

**Bước 1**: Tạo Kaggle Dataset chứa embeddings (file lớn, không commit vào git):

1. Vào Kaggle → Datasets → New Dataset
2. Đặt tên: `ai-viettel-embeddings` (hoặc tên tuỳ ý)
3. Upload 2 files:
   - `icd10_embeddings.npy` (~73 MB float16 / 193 MB float32)
   - `rxnorm_embeddings.npy` (~464 MB float16 / 668 MB float32)
4. Create

**Bước 2**: Trong notebook, setup cells:

```python
# Cell 1: Environment
import os
os.environ['OLLAMA_BASE_URL'] = 'http://127.0.0.1:11434/v1'
os.environ['OLLAMA_MODEL'] = 'qwen2.5:7b-instruct'
os.environ['OLLAMA_KEEP_ALIVE'] = '0'           # Unload model sau mỗi request
os.environ['OLLAMA_NUM_CTX'] = '16384'           # Giảm từ 65536 → 16384 (T4 15GB)
os.environ['HF_HUB_OFFLINE'] = '1'               # Không download từ HF
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['BGE_M3_PATH'] = '/kaggle/input/datasets/YOUR_USER/bge-m3-model/bge-m3-model'

# Cell 2: Embeddings path (optional - nếu dataset tên khác)
os.environ['ICD_EMBEDDINGS_PATH'] = '/kaggle/input/datasets/YOUR_USER/data-embedding/embedding/icd10_embeddings.npy'
os.environ['RXNORM_EMBEDDINGS_PATH'] = '/kaggle/input/datasets/YOUR_USER/data-embedding/embedding/rxnorm_embeddings.npy'

# Cell 3: Install + clone repo
!pip install -r /kaggle/working/ai_viettel/requirements.txt

# Cell 4: Run inference
!python /kaggle/working/ai_viettel/src/inference.py \
    --input /kaggle/working/ai_viettel/input \
    --output /kaggle/working/ai_viettel/output \
    --workers 1 \
    --target-ctx 16384 \
    --max-few-shot 35
```

### Tại sao các config này quan trọng trên Kaggle

| Config | Lý do |
|--------|-------|
| `OLLAMA_NUM_CTX=16384` | T4 chỉ có 15GB VRAM. `num_ctx=65536` (default) + qwen2.5:7b FP16 + KV cache = ~13GB → OOM. 16384 tiết kiệm ~5GB KV cache |
| `OLLAMA_KEEP_ALIVE=0` | Unload model sau mỗi request → GPU 0 rảnh cho BGE-M3 encode ICD/RxNorm |
| `--workers 1` | Ollama thường chỉ handle 1 request/lần; parallel workers gây thêm overhead + duplicate STALE check |
| `BGE_M3_PATH=/kaggle/input/...` | Tránh download BGE-M3 (~2.3GB) mỗi lần restart notebook |
| `HF_HUB_OFFLINE=1` | Force dùng local BGE-M3, không cần internet |

### Auto-detection của Embeddings File

Code trong [src/__init__.py](src/__init__.py) tự động tìm embeddings theo thứ tự:

1. **Env var override**: `ICD_EMBEDDINGS_PATH` hoặc `RXNORM_EMBEDDINGS_PATH`
2. **cwd/data**: `Path.cwd() / "data" / filename` (local repo)
3. **Kaggle Dataset mounts**:
   - `/kaggle/input/ai-viettel-embeddings/<filename>`
   - `/kaggle/input/ai-viettel-data/<filename>`
   - `/kaggle/input/ai_viettel/<filename>`
   - `/kaggle/input/ai-viettel-data/data/<filename>`
   - `/kaggle/input/ai_viettel/data/<filename>`
   - `/kaggle/input/datasets/<user>/data-embedding/embedding/<filename>` (custom)

→ File đầu tiên tồn tại sẽ được dùng.

**Save path cho rebuild**: Nếu embeddings file **STALE** (count mismatch với data), code tự động:
- Tìm writable location (`/kaggle/working/` trên Kaggle, hoặc `cwd/data/` local)
- Save file mới ở đó
- Cập nhật internal path để lần sau load đúng

### GPU Memory Management

Trên Kaggle T4 x2 (2x 15GB VRAM), Ollama và BGE-M3 cạnh tranh GPU 0:

```python
# Ép BGE-M3 sang GPU 1 (đang idle) — sửa src/rxnorm_rag.py:1283
import torch
device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
self._model = SentenceTransformer(model_path, device=device)
```

Hoặc dùng env var:
```python
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # BGE-M3 dùng GPU 1
# Ollama mặc định bind về cuda:0 → không conflict
```

### Monitoring GPU Usage

Trong notebook cell riêng:
```python
!nvidia-smi
```

Output mẫu (Kaggle T4 x2):
```
GPU 0: 14.5/15GB used (Ollama running)  ← Nếu gần full → OOM
GPU 1: 6.3/15GB used (BGE-M3 if moved)   ← Move BGE-M3 sang đây
```

### Troubleshooting Kaggle

| Vấn đề | Nguyên nhân | Fix |
|--------|-------------|-----|
| `icd10_embeddings.npy MISSING` | Git LFS chưa pull | Upload qua Kaggle Dataset (xem Bước 1) |
| `Embeddings STALE — shape (47196) nhưng 62928 codes` | File cũ từ version data khác | Rebuild: `python -c "from src.icd_rag import ICD10VectorSearch; ICD10VectorSearch()._ensure_loaded()"` |
| `CUDA OOM — Tried to allocate 10.82 GB` | Ollama đầy GPU 0, BGE-M3 cố encode | Giảm `OLLAMA_NUM_CTX=16384` + move BGE-M3 sang GPU 1 |
| File save fail: `Read-only file system` | `/kaggle/input/` read-only | Code auto-redirect sang `/kaggle/working/` (xem `get_writable_cache_path`) |

---

## 🎯 ICD RAG (BYT data VN)

**Data**:
- `icd10.jsonl` — WHO ICD-10 2019 VN translation, 15,732 codes (default)
- `DM_ICD10_19_8_BYT.json` — BYT QĐ 4469/BYT, 36,689 entries (fallback)

Schema (`icd10.jsonl`):

```json
{
  "code": "A00",
  "desc_vi": "Bệnh tả",
  "desc_en": "Cholera",
  "source": "Phụ lục ICD-10 BYT",
  "version": "WHO ICD-10 2019"
}
```

Pipeline:

```
VN diagnosis text → Tier 0: Tra từ điển viết tắt/tiền tố chuyên khoa (THA → I10, ...)
                  → Tier 1: Direct VN-VN exact/substring match trong desc_vi (O(1))
                  → Tier 2: Context-Enriched Hybrid Search (BGE-M3 + BM25)
                  + _filter_irrelevant_codes (drop F10/T36/V/W/Y/Z nếu không hợp ngữ cảnh)
                  → Tier 3: Chapter Restriction & Fuzzy Match (rapidfuzz)
                  → top-K ICD codes
```

Filter loại bỏ (chapter restrictions):

- **F10.x**: alcohol-related (chỉ giữ nếu entity có "alcohol"/"rượu")
- **F11-F19**: drug-related mental disorders
- **T36-T50**: poisoning (chỉ giữ nếu có "ngộ độc"/"quá liều")
- **V/W/X/Y**: external causes (chỉ giữ nếu có "tai nạn"/"chấn thương")
- **O00-O9A**: pregnancy
- **Z00-Z99**: factors influencing health (chỉ giữ nếu có family history/screening)

---

## 💊 RxNorm RAG

**Data**: `rxnorm.jsonl` — RxNorm 2026 release, 232,111 entries.

Schema:

```json
{
  "rxcui": "44",
  "name": "mesna",
  "ingredient": "mesna",
  "strength": "",
  "doseform": "",
  "source": "RxNorm full 2026 release"
}
```

Pipeline:

```
VN/EN drug text → Tier 0: Drug Pre-cleaner (_strip_route_freq loại bỏ liều dùng, po, bid)
                → Tier 1: Tra từ điển INN ↔ USAN & Brand VN (_alias_to_generic: Panadol → Acetaminophen)
                → Tier 2: Exact tuple match (ingredient, strength) → top-1 rxcui
                → Tier 3: Hybrid search (BGE-M3 + BM25) với cleaned query
                → Tier 4: Fuzzy match trên names (rapidfuzz) & Compound drug split
```

**Strength normalization**: `"25 mg"`, `"25mg"`, `"25MG"`, `"25.0 MG"` → cùng key `"25MG"`.

**Compound drugs**: `"lisinopril 10 mg / hydrochlorothiazide 12.5 mg"` → split by `/` → match từng component.

---

## 📝 NER Rules (trong [src/prompts.py](src/prompts.py))

### 5 loại entity + mức inclusive

| Type                             | Strategy                                            | Lý do                                    |
| -------------------------------- | --------------------------------------------------- | ----------------------------------------- |
| **THUỐC**                 | Giữ name + strength + route + freq                 | RxNorm SCD lookup cần đầy đủ         |
| **CHẨN_ĐOÁN**           | Giữ tên + type + severity + cause + complications | ICD càng cụ thể càng tốt             |
| **TRIỆU_CHỨNG**          | Chỉ core + qualitative ADJ                         | Không cần candidate, bỏ duration/value |
| **TÊN_XÉT_NGHIỆM**      | Tên test (không kèm giá trị)                   | Tách với KQ                             |
| **KẾT_QUẢ_XÉT_NGHIỆM** | Value + unit (nếu có)                             | Tách với TÊN                           |

### 8 quy tắc bắt buộc

- **R1** NER theo MỨC ĐỘ ĐẦY ĐỦ (per-type)
- **R2** Position khớp 100% với input
- **R3** `candidates: []` LUÔN là `[]`
- **R4** KHÔNG trích lifestyle/social
- **R5** "A cho/trị B" → tách 2 entities
- **R6** Test name + value → tách 2 entities
- **R7** ECG/LAB nối "và"/"," → tách nhiều entities
- **R8** Cùng concept nhiều vị trí → nhiều entities (giữ duplicate)

### 3 assertions

- **isHistorical**: TRƯỚC nhập viện / trong tiền sử (keywords: "Tiền sử:", "Trước đây:", "Đang dùng", "Cách đây"...)
- **isNegated**: BỊ PHỦ ĐỊNH (keywords NGAY TRƯỚC: "không", "chưa", "âm tính")
- **isFamily**: NGƯỜI NHÀ (keywords: "Bố/Mẹ bệnh nhân", "Tiền sử gia đình")

### R37 Enhancements (2026-07-15)

Thêm vào `src/prompts.py`:

1. **BRAND NAME → THUỐC**: 16 brand names (Crestor, Toradol, Augmentin, Tylenol, Advil, Voltaren, Ventolin, Zithromax, Glucophage, Combivent, Zofran, Nexium, Lasix, Lipitor, Zocor, Plavix). NGOẠI LỆ: BiPAP/CPAP/máy thở là thiết bị.
2. **8B Test abbreviations**: 8 nhóm viết tắt (AST, ALT, WBC, INR, BNP, troponin, ...) — bắt buộc TÊN_XN.
3. **9B Abnormal findings → CHẨN_ĐOÁN**: bệnh lý chất trắng, gãy xương, tổn thương X, ST chênh, block nhĩ thất, rung nhĩ, ngoại tâm thu + frequency modifier.
4. **CẤM 8**: Drug-class generic (kháng sinh, NSAID, corticoid, chống đông/X) → DROP.
5. **CẤM 9**: Standalone dose fragment (30 mg, 60 mg) → DROP.
6. **CẤM 10**: Standalone qualifier (không đặc hiệu, không rõ, NOS) → DROP, merge vào diagnosis.
7. **11-14** `<recall_and_precision>`: Checklist theo 8 section bệnh án, bảng 17 false positive VD, decision tree 9 nhánh, two-pass verification (Pass 1 recall max + Pass 2 precision max).

### R37 Postprocess Fixes ([src/postprocess.py](src/postprocess.py))

1. **`_ABNORMAL_FINDING_TO_CHAN_DOAN`** (line ~2436): Extended với `bệnh lý chất trắng`, `ST chênh xuống/lên`, `gãy xương` (standalone), `block nhĩ thất`, `rung nhĩ`, `cuồng nhĩ`, `ngoại tâm thu + frequency modifier`, `tổn thương X`, `viêm mô tế bào`, `phình X`, `(hẹp|hở) động mạch X`.
2. **`_retype_entity`** (line ~2588): Brand name → THUỐC; Test abbreviation standalone → TÊN_XÉT_NGHIỆM.
3. **`_clean_entity_text`** (line ~2293): Drop drug-class generic; drop standalone dose fragment.
4. **`_DRUG_BRANDS`** (~176 entries), **`_TEST_ABBREVIATIONS`** (~64 entries).

---

## 🛠 Troubleshooting

### Connection & Model

| Vấn đề                                     | Nguyên nhân              | Giải pháp                                          |
| --------------------------------------------- | -------------------------- | ---------------------------------------------------- |
| `Connection refused` ở `127.0.0.1:11434` | Ollama chưa chạy         | `ollama serve` trong terminal khác                |
| `Model not found`                           | Chưa pull model           | `ollama pull qwen2.5:7b`                           |
| `404 Not Found` cho `/v1/models`            | Ollama version cũ        | Update Ollama lên 0.5.0+                          |
| Output `[]` rỗng                            | Ollama timeout / JSON fail | Xem `predictions.log`. Tăng `OLLAMA_TIMEOUT=900` |

### Performance & Quality

| Vấn đề | Nguyên nhân | Fix |
|--------|-------------|-----|
| LLM chậm (>2 phút/record) | Chưa dùng GPU / num_gpu thấp | Tăng `num_gpu` trong Modelfile; set `num_gpu=-1` cho full GPU |
| Entities bị miss (recall thấp) | num_ctx quá nhỏ → truncate few-shot | `OLLAMA_NUM_CTX=16384` (Kaggle) hoặc `65536` (≥24GB VRAM) |
| Hallucination type (đúng span, sai class) | Thiếu rules trong prompt | Update R37 prompt (`src/prompts.py`) |
| Stage 3 fail nhiều | num_ctx quá nhỏ cho Stage 3 batch | Set `LLM_DISABLE_STAGE3=1` để skip Stage 3 |

### Embeddings & RAG

| Vấn đề | Nguyên nhân | Fix |
|--------|-------------|-----|
| `FileNotFoundError: *embeddings.npy` | Chưa build embedding | `python scripts/build_*_embeddings.py` |
| `Embeddings STALE — shape (N) nhưng M codes` | Embeddings cũ, data mới | `python -c "from src.icd_rag import ICD10VectorSearch; ICD10VectorSearch()._ensure_loaded()"` (rebuild) |
| Vector search trả codes sai (ICD) | BGE-M3 model chưa load đúng | Check `BGE_M3_PATH=/kaggle/input/...` |
| RxNorm lookup miss nhiều | Drug name có route/freq chưa strip | Check `_strip_route_freq()` pre-cleaner |
| BM25 không có rank_bm25 | Chưa cài package | `uv pip install rank-bm25` |

### Kaggle-specific

| Vấn đề | Nguyên nhân | Fix |
|--------|-------------|-----|
| `CUDA OOM — Tried to allocate X GB` | Ollama + BGE-M3 cùng GPU | Giảm `OLLAMA_NUM_CTX=16384`; move BGE-M3 sang `cuda:1` |
| File save fail: Read-only filesystem | `/kaggle/input/` read-only | Code auto-redirect sang `/kaggle/working/` |
| Git LFS không pull được | GitHub LFS budget hết | Upload embeddings qua Kaggle Dataset thay vì git |
| Notebook restart mất embeddings | `/kaggle/working/` bị clear | Lưu embeddings vào Kaggle Dataset (persistent) |

### Tăng context length Ollama

```bash
# Modelfile cho qwen2.5:7b với ctx 16384
cat > Modelfile <<EOF
FROM qwen2.5:7b
PARAMETER num_ctx 16384
EOF
ollama create qwen2.5:7b-16k -f Modelfile
$env:OLLAMA_MODEL = "qwen2.5:7b-16k"

# Modelfile với GPU layers (cho máy GPU yếu)
cat > Modelfile <<EOF
FROM qwen2.5:7b
PARAMETER num_ctx 16384
PARAMETER num_gpu 20
EOF
ollama create qwen2.5:7b-cpu -f Modelfile
```

### Verify Ollama

```bash
# Health check
curl http://127.0.0.1:11434/

# List models
curl http://127.0.0.1:11434/v1/models

# Test inference
curl http://127.0.0.1:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5:7b","messages":[{"role":"user","content":"Xin chào"}]}'
```

### Font & Encoding

```bash
# Windows PowerShell — fix font VN
$env:PYTHONIOENCODING = "utf-8"

# Linux/macOS
export LANG=vi_VN.UTF-8
export LC_ALL=vi_VN.UTF-8
```

---

## 🌐 Kiến trúc chi tiết

### Token Budget (qwen2.5:7b)

| target_ctx | max_tokens | Sys prompt | Few-shot      | Input (max)    | Note                    |
| ---------- | ---------- | ---------- | ------------- | -------------- | ----------------------- |
| 4096       | 768        | ❌         | ❌            | ❌             | Quá nhỏ, không work    |
| 6144       | 768        | ~6200      | 0             | ❌             | Few-shot bị drop       |
| 8192       | 768        | ~6200      | 0 (auto-drop) | ✅ ~1200 chars | Tight                  |
| 16384      | 12288      | ~21000     | 13500         | ✅ ~4500 chars | **Kaggle T4 khuyến nghị** |
| 32768      | 12288      | ~21000     | 13500         | ✅ ~4500 chars | ≥24GB VRAM              |
| 65536      | 12288      | ~21000     | 13500         | ✅ ~4500 chars | Default, ≥24GB VRAM   |

→ **Khuyến nghị**: dùng `q4_k_m` quantized + `num_ctx=16384` cho input dài trên Kaggle.

### Self-test Prompts

```bash
uv run python src/prompts.py
# Output:
# === Self-test: verify SYSTEM_PROMPT examples ===
#   All positions verified!
```

### Estimate Score

```bash
# Sau khi có output, dùng estimate_score.py để check quality
python scripts/estimate_score.py --structural-only --pred output/
python scripts/estimate_score.py --gold data/gold/ --pred output/    # cần gold annotations
```

### Audit Type-Mismatch (Kaggle-ready)

```bash
python scripts/audit_types_kaggle.py --input /kaggle/working/output --top 50 --save /kaggle/working/audit.json
```

Detect 6 loại type-mismatch:
- drug → not-THUỐC
- test → not-TÊN_XN
- disease → not-CHẨN_ĐOÁN
- KQ → not-KQ_XN
- type-inconsistent
- span-overlap

---

## 📊 Scoring Formula (2026)

```python
final_score = 0.3 · text_score + 0.3 · assertions_score + 0.4 · candidates_score
```

**text_score**: average (1 - WER) trên text field, ghép predicted vs gold theo position+type
- WER word-level (Levenshtein trên tokens)
- Type mismatch = text khớp nhưng type khác → "khái niệm mới" → 0 điểm cả 3 metric, NHÂN ĐÔI (extra + missing)

**assertions_score, candidates_score**: Jaccard similarity per-sample
```python
J_X(i) = 1 nếu gold empty VÀ pred empty
J_X(i) = 0 nếu gold empty MÀ pred KHÔNG empty  ← hallucination penalty cực nặng
J_X(i) = |A∩B|/|A∪B| otherwise
```

**candidates_score có trọng số**:
```python
cand_score = Σ_i J_candidates(i) × w_i / Σ_i w_i
w_i = Σ_k (len(ground_truth_candidates(k)) + 1)
```

→ Sample có nhiều ground truth candidates → đóng góp nhiều hơn vào tổng → hallucination ở sample giàu candidate = penalty lớn.

**Strategy**: Precision-first cho candidates. KHÔNG thêm candidate "cho chắc" — chỉ thêm khi chắc chắn type deserves (THUỐC, CHẨN_ĐOÁN).

---

## 📜 License & References

- **ICD-10 data VN+EN**: WHO ICD-10 2019 (translation từ PDF BYT) — [scripts/extract_icd10_vi_pdf.py](scripts/extract_icd10_vi_pdf.py)
- **ICD-10 data BYT**: Bộ Y Tế Việt Nam, QĐ 4469/BYT ngày 28/10/2020 (`DM_ICD10_19_8_BYT.json`)
- **RxNorm data**: [NIH NLM RxNorm 2026 release](https://www.nlm.nih.gov/research/umls/rxnorm/) (offline JSONL dump)
- **BGE-M3 embedding**: [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) (MIT)
- **Ollama**: [ollama.com](https://ollama.com/) (MIT)
- **Models**: qwen2.5:7b, qwen3.5:9b (Alibaba, Apache 2.0)
- **rapidfuzz**: [maxbachmann/RapidFuzz](https://github.com/maxbachmann/RapidFuzz) (MIT)
- **rank-bm25**: [dorianbrown/rank_bm25](https://github.com/dorianbrown/rank_bm25) (Apache 2.0)

---

## 🤝 Contributing

Issues & PRs welcome! Trước khi submit:

1. Chạy `python scripts/test_r37_augment.py` — verify regex không bị corrupt
2. Chạy `python scripts/validate_outputs.py --input output/` — schema validate
3. Chạy `python scripts/estimate_score.py` — verify không regress

---

## 📝 Version History

| Date       | Version | Highlights                                                              |
|------------|---------|-------------------------------------------------------------------------|
| 2026-07-22 | latest  | Kaggle Dataset support, `get_writable_cache_path`, Stage 3 post-RAG  |
| 2026-07-15 | R37     | Enhanced prompt (BRAND, test abbrev, abnormal findings), CẤM 8-10   |
| 2026-07-14 | R44     | Temperature 0.1 → 0.05, max_tokens 8192 → 12288                       |
| 2026-07-13 | R28     | Auto-mined ICD aliases, custom generic drug class detection         |
| 2026-07-12 | R20.2   | Highlight duplicate (đếm + mark) trước khi gửi LLM                    |
| 2026-07-09 | R8      | Section-based chunking, keep duplicate entities                       |