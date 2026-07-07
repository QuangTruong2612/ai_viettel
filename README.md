# AI Race 2026 — Medical Information Extraction (VN)

Pipeline **trích xuất thực thể y khoa** từ hồ sơ bệnh án tiếng Việt và **liên kết mã chuẩn hoá** (`ICD-10`, `RxNorm`). Chạy **hoàn toàn offline** với Ollama + local embeddings (BGE-M3).

---

## ✨ Highlights

- 🎯 **5 loại entity y khoa**: `THUỐC`, `CHẨN_ĐOÁN`, `TRIỆU_CHỨNG`, `TÊN_XÉT_NGHIỆM`, `KẾT_QUẢ_XÉT_NGHIỆM`
- 🔍 **NER thông minh theo MỨC ĐỘ ĐẦY ĐỦ**: THUỐC + CHẨN_ĐOÁN giữ nguyên (để lookup candidate chính xác); TRIỆU_CHỨNG minimal (bỏ duration/value)
- 🎯 **Hybrid RAG**: Vector (BGE-M3) + BM25 keyword + Fuzzy cho cả ICD và RxNorm
- 🇻🇳 **100% Vietnamese ICD data** (`DM_ICD10_19_8_BYT.json` - 36,689 entries, QĐ 4469/BYT)
- 💊 **RxNorm** lookup VN/EN với BGE-M3 multilingual
- 🏥 **3 assertions**: `isHistorical`, `isNegated`, `isFamily` (context-aware)
- 🚀 **Offline hoàn toàn** — không cần NIH/NLM API

---

## 🚀 Quick Start (5 phút)

```bash
# 1. Setup
git clone <repo> && cd AI_VIETTEL
uv venv && source .venv/bin/activate   # hoặc .venv\Scripts\activate trên Windows
uv pip install -r requirements.txt

# 2. Cài Ollama + model
#    Tải từ https://ollama.com/download
ollama serve &                          # chạy nền
ollama pull qwen2.5:7b                  # ~4.7 GB, mặc định
# Hoặc:
ollama pull qwen3.5:9b                  # ~5.5 GB, mạnh hơn

# 3. Build indexes (1 lần, ~10-30 phút)
uv run python scripts/build_rxnorm_index.py        # ~1s, exact match
uv run python scripts/build_rxnorm_embeddings.py   # ~10 min (GPU), BGE-M3
uv run python scripts/build_icd_embeddings.py      # ~10 min (GPU), BGE-M3

# 4. Chạy thử
uv run python -m src.inference --input input --output output --target-ctx 8192 --limit 3
```

Kết quả: 3 files `output/1.json`, `output/2.json`, `output/3.json` chứa entities + ICD codes + RxNorm candidates.

---

## 📂 Cấu trúc thư mục

```
AI_VIETTEL/
├── src/                              # Production code
│   ├── __init__.py                   # Package init, export all modules
│   ├── prompts.py                    # SYSTEM_PROMPT cho NER + few-shot loader
│   ├── llm_client.py                 # OpenAI-compatible wrapper cho Ollama
│   ├── inference.py                  # Main driver — orchestrate pipeline
│   ├── postprocess.py                # Validate, fix position, populate candidates
│   ├── icd_rag.py                    # ICD-10 RAG (BYT VN data, vector + BM25 hybrid)
│   └── rxnorm_rag.py                 # RxNorm RAG (vector + BM25 + exact match hybrid)
│
├── scripts/                          # Build & utility scripts
│   ├── build_icd_index.py          # Build icd_index.json (exact match)
│   ├── build_icd_embeddings.py      # Build icd10_embeddings.npy (BGE-M3)
│   ├── build_rxnorm_index.py       # Build rxnorm_index.json (exact match)
│   ├── build_rxnorm_embeddings.py  # Build rxnorm_embeddings.npy (BGE-M3)
│   ├── test_inference.py           # Smoke test 1 record + sanity check
│   └── validate_outputs.py         # Validate output schema
│
├── data/                             # Input data + indexes
│   ├── DM_ICD10_19_8_BYT.json      # ICD-10 source (36,689 entries VN)
│   ├── rxnorm.jsonl                 # RxNorm source (232k entries EN)
│   ├── examples.jsonl               # Few-shot examples (33 entries, positions verified)
│   ├── icd_index.json               # [build] exact match index
│   ├── icd10_embeddings.npy         # [build] BGE-M3 matrix ~36k x 1024
│   ├── icd10_bm25_tokens.jsonl.gz   # [auto-build] BM25 token cache
│   ├── rxnorm_index.json            # [build] exact match index
│   ├── rxnorm_embeddings.npy        # [build] BGE-M3 matrix ~232k x 1024
│   ├── rxnorm_bm25_tokens.jsonl.gz  # [auto-build] BM25 token cache
│   └── translation_cache.json       # [auto-build] VN→EN translation cache
│
├── input/                            # Test input files (1.txt, 2.txt, ..., 100.txt)
├── output/                           # Output JSON files
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

---

## 🧠 Kiến trúc Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│ INPUT: hồ sơ bệnh án tiếng Việt (input/N.txt)              │
└──────────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN 1 — Clinical NER (LLM via Ollama)                  │
│ • Preprocess input (strip markdown, N/A, truncate)            │
│ • Adaptive few-shot (auto-budget theo context length)        │
│ • Build SYSTEM_PROMPT + few-shot messages                    │
│ • LLM call (qwen2.5:7b / qwen3.5:9b / etc.)                 │
│ • JSON parser với retry + recovery                           │
│ Output: JSON array entities (text, type, position, ...)     │
└──────────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN 2 — LLM Rescan (batch)                             │
│ • For each THUỐC + CHẨN_ĐOÁN:                                 │
│   - Translate VN→EN medical phrase (LLM call, cached)        │
│ • Context enrichment từ nearby entities                       │
└──────────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN 3 — Postprocess + RAG Population                   │
│ • Validate positions (auto-fix nếu LLM sai index)             │
│ • Dedupe entities (giữ duplicate ở vị trí khác nhau - R8)    │
│ • Detect substring entities (drop shorter)                   │
│ • Detect assertions (isHistorical/isNegated/isFamily)        │
│ • Populate candidates:                                         │
│   - THUỐC → RxNormRetriever.lookup()                         │
│   - CHẨN_ĐOÁN → ICDRetriever.lookup()                        │
│   - Smart assert detection (LLM bỏ sót)                      │
│ Output: Final JSON array với đầy đủ 5 trường               │
└──────────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────────┐
│ OUTPUT: output/N.json — entities + ICD codes + RxNorm codes  │
└──────────────────────────────────────────────────────────────┘
```

### Hybrid RAG Architecture (ICD + RxNorm)

```
Query (VN/EN drug or disease text)
              ↓
┌─────────────────────────────────┐
│  L1: Exact match (fast path)     │  ← (ingredient, strength) for RxNorm
│  - O(1) dictionary lookup         │  ← name → code for ICD
└─────────────────────────────────┘
              ↓ (miss)
┌─────────────────────────────────┐
│  L2: Hybrid search                │
│  ┌────────────────────────────┐  │
│  │ Vector (BGE-M3 cosine)      │  │  ← semantic similarity
│  │   ↓                          │  │  ← multilingual (VN/EN)
│  │ BM25 keyword                 │  │  ← exact token match
│  │   ↓                          │  │
│  │ Union candidates             │  │  ← expand recall
│  │ Re-score cosine ≥ 0.7       │  │  ← filter noise
│  └────────────────────────────┘  │
└─────────────────────────────────┘
              ↓ (miss)
┌─────────────────────────────────┐
│  L3: Fuzzy match (rapidfuzz)     │  ← VN/EN variants, partial_ratio
└─────────────────────────────────┘
              ↓
         Return top-1 rxcui / ICD code(s)
```

---

## 🔧 Cài đặt

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
openai>=1.30.0           # OpenAI-compatible API cho Ollama
rapidfuzz>=3.9.0         # Fuzzy match (drug/disease names)
sentence-transformers>=3.0.0   # BGE-M3 embeddings
numpy>=1.24.0            # Matrix operations
jsonschema>=4.21.0       # Output validation
requests>=2.31.0         # HTTP
tqdm>=4.66.0             # Progress bars
rank-bm25>=0.2.2         # BM25 keyword index
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
# → data/icd10_embeddings.npy (~140 MB)

# RxNorm exact match index (~1s) — REQUIRED cho L1 exact lookup
uv run python scripts/build_rxnorm_index.py
# → data/rxnorm_index.json (~7 MB)

# RxNorm embeddings (BGE-M3, ~10-30 phút GPU) — REQUIRED cho hybrid search
uv run python scripts/build_rxnorm_embeddings.py
# → data/rxnorm_embeddings.npy (~900 MB)
```

> **Không có GPU?** Có thể chạy trên Google Colab — xem script `build_icd_embeddings.py` (tự detect GPU/CPU).
>
> **Cả ICD + RxNorm đều có cùng kiến trúc 3 bước**: (1) exact match index, (2) BGE-M3 embeddings, (3) hybrid pipeline kết hợp cả hai.

### 4. Chạy inference

```bash
# Smoke test (1 record)
uv run python scripts/test_inference.py --out output/smoke_test.json

# Full pipeline
uv run python -m src.inference \
    --input input \
    --output output \
    --workers 1 \
    --target-ctx 8192 \
    --max-few-shot 3
```

**Tham số quan trọng**:

| Tham số           | Mặc định    | Mô tả                                                                   |
| ------------------ | -------------- | ------------------------------------------------------------------------- |
| `--input`        | `data/input` | Thư mục chứa files input (`.txt` hoặc `.json`)                    |
| `--output`       | `output`     | Thư mục output JSON                                                     |
| `--workers`      | `1`          | Số parallel workers (Ollama serve 1 request/lần, parallel không giúp) |
| `--target-ctx`   | `6144`       | Context window của Ollama (set 8192+ để fit few-shot)                  |
| `--max-few-shot` | `3`          | Số few-shot example tối đa (auto-budget theo context)                  |
| `--limit`        | `0`          | Giới hạn số records (0 = all)                                          |
| `--no-resume`    | `False`      | Re-process tất cả (default: skip records đã có output)               |

**Đổi model**:

```bash
# PowerShell
$env:OLLAMA_MODEL = "qwen3.5:9b"

# bash
export OLLAMA_MODEL=qwen3.5:9b

uv run python -m src.inference --input input --output output
```

### 5. Validate output

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

## 🎯 ICD RAG (BYT data VN)

**Data**: `DM_ICD10_19_8_BYT.json` — QĐ 4469/BYT, 36,689 entries, tiếng Việt.

Schema:

```json
{
  "Mã": "I10",
  "Tên bệnh": "Bệnh lý tăng huyết áp",
  "Nhóm bệnh": "Bệnh tăng huyết áp",
  "Mô tả": "QĐ 4469/BYT ngày 28/10/2020"
}
```

Pipeline:

```
VN diagnosis text → ICD10VectorSearch (BGE-M3 cosine ≥ 0.7)
                  + ICD10BM25Index (expand candidates)
                  → ICD10HybridSearch (union + re-score)
                  + _filter_irrelevant_codes (drop F10/T36/V/W/Y/Z)
                  → top-K ICD codes
```

Filter loại bỏ:

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
VN/EN drug text → _parse_drug (strip route/freq, parse ing+strength)
                 ↓
                 L1: Exact match (ing, strength) → top-1 rxcui
                 ↓ (miss)
                 L2: Hybrid search (BGE-M3 + BM25) — semantic match
                 ↓ (miss)
                 L3: Fuzzy match trên names
```

Strength normalization: `"25 mg"`, `"25mg"`, `"25MG"`, `"25.0 MG"` → cùng key `"25MG"`.

Compound drugs: `"lisinopril 10 mg / hydrochlorothiazide 12.5 mg"` → split by `/` → match từng component.

---

## 📝 NER Rules (trong `src/prompts.py`)

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

---

## 🛠 Troubleshooting

| Vấn đề                                     | Nguyên nhân              | Giải pháp                                          |
| --------------------------------------------- | -------------------------- | ---------------------------------------------------- |
| `Connection refused` ở `127.0.0.1:11434` | Ollama chưa chạy         | `ollama serve` trong terminal khác                |
| `Model not found`                           | Chưa pull model           | `ollama pull qwen2.5:7b`                           |
| Output`[]` rỗng                            | Ollama timeout / JSON fail | Xem`predictions.log`. Tăng `OLLAMA_TIMEOUT=300` |
| LLM chậm (>2 phút/record)                   | Chưa dùng GPU            | Tăng`num_gpu` trong Modelfile                     |
| `FileNotFoundError: *embeddings.npy`        | Chưa build embedding      | `python scripts/build_*_embeddings.py`             |
| Context overflow                              | target_ctx quá nhỏ       | Tăng`--target-ctx 8192` hoặc 16384               |
| Font VN lỗi trên Windows                    | CMD non-UTF8               | `$env:PYTHONIOENCODING = "utf-8"`                  |

### Tăng context length Ollama

```bash
# Modelfile cho qwen2.5:7b với ctx 8192
cat > Modelfile <<EOF
FROM qwen2.5:7b
PARAMETER num_ctx 8192
EOF
ollama create qwen2.5:7b-8k -f Modelfile
$env:OLLAMA_MODEL = "qwen2.5:7b-8k"

# Modelfile với GPU layers (cho máy GPU yếu)
cat > Modelfile <<EOF
FROM qwen2.5:7b
PARAMETER num_ctx 8192
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
```

---

## 🌐 Kiến trúc chi tiết

### Token Budget (qwen2.5:7b)

| target_ctx | max_tokens | Sys prompt | Few-shot      | Input (max)    |
| ---------- | ---------- | ---------- | ------------- | -------------- |
| 4096       | 768        | ~6200 ❌   | 0             | ❌             |
| 6144       | 768        | ~6200      | 0             | ❌             |
| 8192       | 768        | ~6200      | 0 (auto-drop) | ✅ ~1200 chars |
| 16384      | 768        | ~6200      | 720           | ✅ ~9000 chars |

→ **Khuyến nghị**: dùng `q4_k_m` quantized + `num_ctx=16384` cho input dài.

### Prompt Engineering

`src/prompts.py` chứa SYSTEM_PROMPT được tối ưu cho tiếng Việt với:

- 8 quy tắc bắt buộc (R1-R8)
- 5 loại entity với mức inclusive khác nhau
- 3 assertions với keywords + context awareness
- 5 examples in-context (positions 100% verified)
- ECG/tim mạch patterns, vital signs, drug naming, allergy, VN abbreviations
- "TRIỆU_CHỨNG vs CHẨN_ĐOÁN" disambiguation guide

Tự kiểm tra positions khi chạy:

```bash
uv run python src/prompts.py
# Output:
# === Self-test: verify SYSTEM_PROMPT examples ===
#   All positions verified!
```

### Build Indexes (chi tiết)

```bash
# 1. Build RxNorm exact match index (~1s)
#    → data/rxnorm_index.json (~7 MB)
uv run python scripts/build_rxnorm_index.py

# 2. Build RxNorm embeddings (~10-30 min GPU)
#    → data/rxnorm_embeddings.npy (~900 MB)
uv run python scripts/build_rxnorm_embeddings.py

# 3. Build ICD embeddings (~5-10 min GPU)
#    → data/icd10_embeddings.npy (~140 MB)
uv run python scripts/build_icd_embeddings.py
```

---

## 📜 License & References

- **ICD-10 data**: Bộ Y Tế Việt Nam, QĐ 4469/BYT ngày 28/10/2020 (`DM_ICD10_19_8_BYT.json`)
- **RxNorm data**: [NIH NLM RxNorm 2026 release](https://www.nlm.nih.gov/research/umls/rxnorm/) (offline JSONL dump)
- **BGE-M3 embedding**: [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) (MIT)
- **Ollama**: [ollama.com](https://ollama.com/) (MIT)
- **Models**: qwen2.5:7b, qwen3.5:9b (Alibaba, Apache 2.0)
