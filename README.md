# AI Race 2026 — Medical Information Extraction

Pipeline trích xuất thực thể y khoa và liên kết mã chuẩn hoá (`ICD-10`, `RxNorm`) từ hồ sơ bệnh án tiếng Việt, chạy **hoàn toàn offline** với **Ollama** + local embeddings.

## 🚀 Quick Start (5 phút)

```powershell
# 1. Clone + setup
git clone <repo>
cd AI_VIETTEL

# Tạo venv và cài dependencies
uv venv && uv pip install -r requirements.txt

# 2. Cài Ollama + pull model (chỉ làm 1 lần)
# Tải: https://ollama.com/download
ollama pull qwen2.5:7b          # ~4.7 GB
# Hoặc nếu muốn model mạnh hơn:
# ollama pull gemma2:9b          # ~5.5 GB (Recommend, hiểu JSON tốt hơn)

# 3. Khởi Ollama server (chạy nền)
ollama serve

# 4. Generate ICD-10 embeddings (~5 phút GPU, ~30 phút CPU)
uv run python scripts/build_icd_embeddings.py

# 5. Smoke test
uv run scripts/test_inference.py --out output/smoke_test.json
```

---

## 📋 Yêu cầu hệ thống

| Thành phần | Tối thiểu             | Khuyến nghị                   |
| ------------ | ----------------------- | ------------------------------- |
| Python       | 3.10+                   | 3.11                            |
| RAM          | 16 GB                   | 32 GB                           |
| GPU          | NVIDIA RTX 3060 (12GB)  | RTX 4090 (24GB) hoặc A100      |
| VRAM         | 8 GB (qwen2.5:7b)       | 16 GB (gemma2:9b / qwen2.5:14b) |
| Disk         | 10 GB                   | 20 GB                           |
| OS           | Windows / Linux / macOS | Linux + NVIDIA                  |

---

## 🆕 Cải tiến mới nhất

### Hybrid ICD Search (Vector + BM25)

- Kết hợp BGE-M3 cosine similarity với BM25 keyword ranking
- Công thức: `combined = α·cosine + β·bm25_normalized` (mặc định α=0.6, β=0.4)
- Khắc phục "vector ảo" giữa các code cùng concept khác grade/stage (heart failure II vs III)
- Files: [src/icd_rag.py](src/icd_rag.py) — `ICD10VectorSearch`, `ICD10BM25Index`, `ICD10HybridSearch`

### Prompt Engineering Improvements

- **XML structure**: SYSTEM_PROMPT wrap trong các `<role>`, `<instructions>`, `<workflow>`, `<entity_types>`, `<extraction_rules>`, `<special_cases_ecg>`, `<assertions>`, `<output_format>`, `<final_rules>` — giúp LLM phân biệt rõ từng phần chỉ thị.
- **Few-shot examples**: 32 ví dụ chất lượng cao trong [data/examples.jsonl](data/examples.jsonl) (đã fix 132/151 positions sai). Auto-budget theo `target_ctx`.
- **Positive framing**: các rule "BỎ lifestyle / BỎ đại từ chung" được refactor thành "TEXT ENTITY = TÊN CỤ THỂ", tránh paradoxical attention khi LLM đọc từ cấm.

### Context-Aware ICD/RxNorm Pipeline

- 6 lớp: Exact match → Translate VN→EN → Hybrid search → Fuzzy EN → Fuzzy VN → Remote NIH fallback
- LLM đọc TOÀN BỘ input trước khi extract (3-step reasoning)
- Context-aware query cho BGE-M3 (drugs + symptoms → disambiguate diagnosis)
- Drug→Diagnosis inference table (amlodipine → THA, etc.)

---

## 🧠 Kiến trúc tổng quan

Pipeline 3 giai đoạn:

### Giai đoạn 1 — Clinical NER (LLM)

Input: hồ sơ bệnh án tiếng Việt.
Output: JSON array các entity thô (`THUỐC`, `CHẨN_ĐOÁN`, `TRIỆU_CHỨNG`, `TÊN_XÉT_NGHIỆM`, `KẾT_QUẢ_XÉT_NGHIỆM`) với `position` và `assertions`.
Model: qwen2.5:7b / gemma2:9b (qua Ollama OpenAI-compatible API).

### Giai đoạn 2 — LLM Context Rescanning

- **THUỐC**: gom tên gốc + strength + route + frequency (vd `Amlodipine 10mg uống daily`).
- **CHẨN_ĐOÁN**: gom severity/location/complication (vd `Trào ngược dạ dày` + `không viêm thực quản` → `GERD without esophagitis`).

### Giai đoạn 3 — Medical RAG

- **RxNorm**: Local index + NIH REST API + cache.
- **ICD-10**: Hybrid search (BGE-M3 vector + BM25 keyword) trên 71,705 codes (`data/icd10.jsonl`).

---

## 📂 Cấu trúc thư mục

```
AI_VIETTEL/
├── src/
│   ├── llm_client.py       # OpenAI-compatible client (timeout, retry, JSON parser)
│   ├── prompts.py          # SYSTEM_PROMPT (XML-wrapped) + few-shot loader
│   ├── icd_rag.py          # Translator + ICD10HybridSearch (vector + BM25)
│   ├── rxnorm_rag.py       # RxNorm lookup + NIH API
│   ├── postprocess.py      # Position auto-fix, LLM rescan, dedupe
│   └── inference.py        # Main driver — orchestrate pipeline
│
├── scripts/
│   ├── build_icd_embeddings.py  # Generate icd10_embeddings.npy (BGE-M3)
│   ├── test_inference.py        # Smoke test 1 record
│   └── validate_outputs.py      # Schema validation
│
├── data/
│   ├── icd10.jsonl              # 71,705 ICD-10 codes (full)
│   ├── icd10_embeddings.npy     # BGE-M3 matrix ~280 MB (build 1 lần)
│   ├── icd10_bm25_tokens.jsonl.gz  # BM25 token cache ~1 MB (build 1 lần)
│   ├── examples.jsonl           # 32 few-shot examples (verified positions)
│   ├── rxnorm_index.json        # Local RxNorm exact-match dict
│   ├── translation_cache.json   # VN→EN translation cache
│   └── icd_remote_cache.json    # NIH ICD API cache
│
├── requirements.txt             # openai, rapidfuzz, sentence-transformers, rank-bm25, ...
└── README.md                    # File này
```

---

## 🛠️ Cài đặt chi tiết

### Bước 1: Python venv

```bash
# Linux / macOS / Git Bash
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Windows PowerShell
uv venv
uv pip install -r requirements.txt
```

### Bước 2: Ollama + Model

```bash
# Linux
curl -fsSL https://ollama.com/install.sh | sh

# macOS: brew install ollama
# Windows: tải từ https://ollama.com/download

# Pull model (chọn 1)
ollama pull qwen2.5:7b          # Default — nhanh, chất lượng OK
ollama pull gemma2:9b           # Tốt hơn cho JSON output (Recommend)
```

Kiểm tra:

```bash
ollama list                     # Models đã pull
ollama serve                    # Khởi server (default port 11434)
```

**Model test nhanh**:

```bash
curl http://127.0.0.1:11434/v1/models
# → {"data": [{"id": "qwen2.5:7b", ...}]}
```

### Bước 3: Generate ICD-10 Embeddings

Cần GPU để nhanh (~5 phút). Nếu chỉ có CPU, chạy qua đêm:

```bash
uv run python scripts/build_icd_embeddings.py
# Output: data/icd10_embeddings.npy (~280 MB)
```

**Trên Colab (khuyên dùng nếu GPU yếu)**:

```python
# Trong Colab cell
!pip install sentence-transformers numpy tqdm
from google.colab import files
uploaded = files.upload()  # Upload icd10.jsonl

import json, time, numpy as np
from sentence_transformers import SentenceTransformer

descs, codes = [], []
with open("icd10.jsonl", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        if row.get("code") and row.get("desc_en"):
            codes.append(row["code"])
            descs.append(row["desc_en"])

model = SentenceTransformer("BAAI/bge-m3", device="cuda")
emb = model.encode(descs, batch_size=256, show_progress_bar=True,
                   normalize_embeddings=True, convert_to_numpy=True)
np.save("icd10_embeddings.npy", emb)
files.download("icd10_embeddings.npy")  # Tải về
```

Sau đó copy file về `data/icd10_embeddings.npy`.

---

## 🚀 Chạy Pipeline

### 1. Smoke test (1 record)

```bash
$env:OLLAMA_MODEL = "qwen2.5:7b"   # PowerShell
export OLLAMA_MODEL=qwen2.5:7b     # bash

uv run scripts/test_inference.py --out output/smoke_test.json
```

Expected: 1 file `output/smoke_test.json` chứa entities + ICD codes + RxNorm candidates.

### 2. Inference toàn bộ data

Đặt file bệnh án vào `data/input/` (đặt tên `1.txt`, `2.txt`, ... hoặc `1.json` với field `"text"`):

```bash
uv run python -m src.inference `
    --input data/input `
    --output output/ `
    --workers 1 `
    --target-ctx 8192 `
    --max-few-shot 10
```

Tham số quan trọng:

- `--workers 1` — Ollama thường chỉ serve 1 request tại 1 thời điểm, parallel không giúp được.
- `--target-ctx 8192` — context window (qwen2.5:7b default 4096, cần set 8192 để fit few-shot).
- `--max-few-shot 10` — giới hạn số few-shot example (auto-budget theo context).

Đổi model qua env hoặc CLI:

```bash
$env:OLLAMA_MODEL = "gemma2:9b"   # switch model
uv run python -m src.inference --input data/input --output output/ --model gemma2:9b
```

### 3. Validate schema

```bash
uv run scripts/validate_outputs.py --input output/
```

---

## 🔧 Cấu hình Ollama nâng cao

### Tăng context length

Ollama default context = 2048 (qwen2.5:7b) hoặc 8192 (gemma2:9b). Để tăng:

```bash
# Tạo Modelfile riêng
cat > Modelfile <<EOF
FROM qwen2.5:7b
PARAMETER num_ctx 8192
EOF

ollama create qwen2.5:7b-8k -f Modelfile
$env:OLLAMA_MODEL = "qwen2.5:7b-8k"
```

### GPU layers (nếu GPU yếu)

```bash
# Chạy chủ yếu trên CPU, chỉ 20 layer trên GPU
cat > Modelfile <<EOF
FROM qwen2.5:7b
PARAMETER num_ctx 8192
PARAMETER num_gpu 20
EOF
ollama create qwen2.5:7b-cpu -f Modelfile
```

### Chạy Ollama trên Colab

```python
# Colab cell 1: install + serve
!curl -fsSL https://ollama.com/install.sh | sh
!ollama pull gemma2:9b
import subprocess, time
subprocess.Popen(["nohup", "ollama", "serve"], stdout=open("/tmp/ollama.log", "w"))
time.sleep(5)  # chờ server boot

# Colab cell 2: set env
import os
os.environ["OLLAMA_MODEL"] = "gemma2:9b"
os.environ["OLLAMA_BASE_URL"] = "http://127.0.0.1:11434/v1"

# Colab cell 3: run inference
!python -m src.inference --input data/input --output output/ --model gemma2:9b
```

---

## 🛠 Troubleshooting

| Lỗi                                                  | Nguyên nhân                         | Cách khắc phục                                                        |
| ----------------------------------------------------- | ------------------------------------- | ------------------------------------------------------------------------ |
| `Connection refused` ở `127.0.0.1:11434`         | Ollama chưa chạy                    | `ollama serve` trong terminal khác                                    |
| `Model 'qwen2.5:7b' not found`                      | Chưa pull model                      | `ollama pull qwen2.5:7b`                                               |
| Output`[]` rỗng                                    | Ollama trả JSON sai format / timeout | Xem`predictions.log`. Tăng timeout: `OLLAMA_TIMEOUT=300` env        |
| LLM chậm (>2 phút/record)                           | Chưa tận dụng GPU                  | Set`num_gpu` trong Modelfile                                           |
| `FileNotFoundError: icd10_embeddings.npy`           | Chưa build embedding                 | `uv run python scripts/build_icd_embeddings.py`                        |
| Lỗi font tiếng Việt trên Windows Terminal         | CMD/PowerShell mặc định non-UTF8   | `$env:PYTHONIOENCODING = "utf-8"` trước khi chạy                    |
| ICD lookup trả code irrelevant                       | Hybrid search chưa bật              | Verify trong code:`use_hybrid=True` (default)                          |
| Drug candidates có parenthetical`(uống)` bị mất | VN paren chưa strip                  | Đã fix:`_strip_paren_keep_dose()` trong `src/rxnorm_rag.py`        |
| ECG findings → 0 ICD candidates                      | NER phân loại sai                   | Đã fix: SYSTEM_PROMPT có rule + few-shot examples#5, #20, #26-28, #31 |
| Few-shot overflow context                             | target_ctx quá nhỏ                  | Tăng`--target-ctx 8192` hoặc giảm `--max-few-shot`                |

### Verify Ollama đang chạy

```bash
# Health check
curl http://127.0.0.1:11434/

# List models
curl http://127.0.0.1:11434/v1/models
```

### Debug inference

```bash
# Logs sẽ ghi ra predictions.log + stdout
$env:PYTHONIOENCODING = "utf-8"
uv run python -m src.inference --input data/input --output output/ --limit 3 2>&1 | Tee-Object output/run.log
```

---

## 📦 Data setup (nếu thiếu file)

```bash
# Khôi phục data files (translation_cache, ICD index, etc.)
git checkout HEAD -- data/

# Generate ICD-10 embeddings (1 lần, ~5 phút GPU)
uv run python scripts/build_icd_embeddings.py

# (Optional) Download RxNorm raw data
uv run python scripts/download_rxnorm.py   # → data/rxnorm_raw.json
uv run python scripts/build_vn_dict.py     # → data/vn_drug_names.csv

# Verify
ls data/
# Expect:
#   icd10.jsonl                          (71k codes)
#   icd10_embeddings.npy                 (~280 MB, BGE-M3 matrix)
#   icd10_bm25_tokens.jsonl.gz           (~1 MB, BM25 token cache)
#   examples.jsonl                       (32 few-shot examples)
#   translation_cache.json, icd_*.json   (auto-built on first inference)
```

---

## 🌐 Kiến trúc chi tiết

### Hybrid Search (Vector + BM25)

Để giải quyết vấn đề vector "ảo" giữa các code cùng concept (vd `Paracetamol 500mg` vs `Paracetamol 250mg` có cosine > 0.95 nhưng khác mã RxNorm), pipeline dùng công thức:

```
combined_score = α · cosine_similarity + β · bm25_normalized
                 (α = 0.6, β = 0.4 mặc định)
```

- **Vector**: BAAI/bge-m3 (đa ngôn ngữ VN/EN), 71k code × 1024-dim, normalized L2.
- **BM25**: rank_bm25 trên multi-field text (desc_en × 2 + desc_vi + code), cache gzip ~1 MB.

Xem [src/icd_rag.py](src/icd_rag.py) — class `ICD10HybridSearch`.

### Prompt Engineering (3 phases)

| Phase | Nội dung                                                                                                       |
| ----- | --------------------------------------------------------------------------------------------------------------- |
| 1     | XML structure: wrap SYSTEM_PROMPT thành`<role>`, `<instructions>`, `<workflow>`, `<entity_types>`, ... |
| 2     | Few-shot: 32 examples trong`data/examples.jsonl` (đã verify positions 100% đúng)                          |
| 3     | Positive framing: refactor rules "BỎ lifestyle" → "TEXT ENTITY = TÊN CỤ THỂ"                               |

### Token budget

| target_ctx | max_tokens | budget cho few-shot | estimated examples |
| ---------- | ---------- | ------------------- | ------------------ |
| 4096       | 2048       | -2108               | 0 ❌               |
| 8192       | 2048       | 1932                | ~9                 |
| 16384      | 2048       | 10132               | ~10 (capped)       |

→ Với gemma2:9b (context 8192 default), fit ~9 few-shot examples.

---

## ✅ Verify full setup

```bash
# 1. Smoke test
uv run scripts/test_inference.py --out output/smoke_test.json

# 2. Validate schema
uv run scripts/validate_outputs.py --input output/

# 3. Run trên 5 records
uv run python -m src.inference --input data/input --output output_test --limit 5

# 4. Diff vs baseline (nếu có)
```

Nếu tất cả pass → setup OK. Chạy full pipeline:

```bash
uv run python -m src.inference --input data/input --output output/ --target-ctx 8192
```

---

## 📜 License & References

- ICD-10 data: từ [kamillamagna/ICD-10-CSV](https://github.com/kamillamagna/ICD-10-CSV) (CC0 public domain).
- BGE-M3 embedding: [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) (MIT).
- Ollama: [ollama.com](https://ollama.com/) (MIT).
- RxNorm: NIH NLM [RxNav REST API](https://rxnav.nlm.nih.gov/RxNormAPIs.html).
