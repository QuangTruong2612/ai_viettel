# AI Race 2026 — Medical Information Extraction (Vòng 1)

Pipeline trích xuất thực thể y khoa có cấu trúc từ văn bản lâm sàng **tiếng Việt** — chạy hoàn toàn offline sau khi cache, không cần fine-tune.

| Loại | Mã chuẩn hoá | Nguồn |
|---|---|---|
| `THUỐC` | RxNorm (`rxcui`) | NIH RxNorm REST API + local cache |
| `CHẨN_ĐOÁN` | ICD-10 (`code`) | NIH Clinical Tables API + VN→EN translation |
| `TRIỆU_CHỨNG` | — | LLM trực tiếp |
| `TÊN_XÉT_NGHIỆM` | — | LLM trực tiếp |
| `KẾT_QUẢ_XÉT_NGHIỆM` | — | LLM trực tiếp |

Mỗi entity có `position` `[start, end]` (0-indexed) + `assertions` ∈ `{isHistorical, isNegated, isFamily}` (subset, có thể kết hợp nhiều flag). Output: `output/{N}.json` (1 record / file).

---

## Tech Stack

| Thành phần | Chi tiết |
|---|---|
| **LLM** | `qwen2.5-7b-instruct` (Q4_K_M ~4.7GB) trong LM Studio tại `http://127.0.0.1:1234/v1` |
| **RxNorm API** | `https://rxnav.nlm.nih.gov/REST/drugs.json?name=<drug>` |
| **ICD-10 API** | `https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search?terms=<q>&sf=name` |
| **Backend** | Python ≥ 3.10, `openai` client OpenAI-compatible |

**Ràng buộc**: LLM self-host ≤ 9B params · không fine-tune · chỉ gọi 2 API NIH ở trên.

---

## Cấu trúc dự án

```
F:\AI_VIETTEL\
├── src/
│   ├── llm_client.py        # OpenAI-compatible wrapper cho LM Studio (timeout, retry)
│   ├── prompts.py           # SYSTEM_PROMPT (5 loại + 3 assertions + rules A-F) + few-shot loader + schema
│   ├── rxnorm_rag.py        # RxNormRetriever: exact → normalize → fuzzy → NIH API live
│   ├── icd_rag.py           # ICDRetriever: Translator VN→EN + lookup exact/fuzzy/NIH API + prefix strip
│   ├── postprocess.py       # validate_positions + dedupe + assemble_record + drug-cho split
│   └── inference.py         # Main loop với health-check + adaptive few-shot budget
│
├── scripts/
│   ├── download_rxnorm.py   # Bulk tải RxNorm qua /drugs.json?name=
│   ├── build_vn_dict.py     # Build RxNorm index từ JSON seed
│   ├── build_icd_dict.py    # Build ICD-10 index từ JSONL seed
│   ├── test_inference.py    # Smoke test trên ví dụ BTC
│   └── validate_outputs.py  # Schema check cho output/
│
├── data/
│   ├── examples.jsonl       # 22 few-shot examples (đa dạng 5 loại + 3 assertions)
│   ├── rxnorm_seed.jsonl    # 11 thuốc từ đề bài (seed RxNorm)
│   ├── rxnorm_index.json    # Built (22 keys, 11 names)
│   ├── icd_seed.jsonl       # 27 ICD-10 phổ biến + VN alias
│   ├── icd_index.json       # Built (27 entries)
│   ├── translation_cache.json    # Auto-generated: VN→EN cache
│   ├── icd_remote_cache.json     # Auto-generated: NIH ICD remote cache
│   └── rxnorm_api_cache.json     # Auto-generated: NIH RxNorm remote cache
│
├── input/                   # 1.txt .. 100.txt (bắt buộc)
├── output/                  # 1.json .. 100.json (kết quả inference)
├── predictions.log          # Realtime log, ghi đè mỗi lần chạy
├── optimize_lm_studio.md    # Hướng dẫn tối ưu LM Studio
├── requirements.txt
└── README.md
```

### Cleanup đã làm

- ❌ Không còn `chat_json`, `_build_messages` (chỉ `_extract_json`)
- ❌ Không còn `build_repair_prompt` (retry cùng prompt)
- ❌ Không còn `embed_helper.py` (NIH API đủ dùng)

---

## Cài đặt

### Bước 1 — Python dependencies

```powershell
cd F:\AI_VIETTEL
pip install -r requirements.txt
```

`requirements.txt` gồm: `openai`, `rapidfuzz`, `sentence-transformers`, `numpy`, `jsonschema`, `requests`, `tqdm`.

### Bước 2 — LM Studio + Model

1. Mở LM Studio → tải **`qwen2.5-7b-instruct`** (Q4_K_M ~4.7GB) HOẶC **`qwen3.5-9b`** (Q4_K_M ~5.5GB)
2. **Settings → Inference**:
   - **Context Length**: `4096` (an toàn, mặc định) HOẶC `8192` (nếu GPU ≥ 8GB)
   - **GPU Offload**: Max
   - **Flash Attention**: ON (RTX 40 series)
3. **Developer → Start Server** (port `1234`)
4. Verify:
   ```powershell
   curl http://127.0.0.1:1234/v1/models
   ```
   Phải trả JSON có tên model đang load.

### Bước 3 — Chuẩn bị data

Indices đã sẵn sàng trong `data/`:
- `rxnorm_index.json` — 11 thuốc từ đề bài (22 exact keys)
- `icd_index.json` — 27 ICD-10 phổ biến + VN alias

Nếu muốn mở rộng dictionary (tùy chọn):
```powershell
python scripts/download_rxnorm.py --out data/rxnorm_raw.json
python scripts/build_vn_dict.py --dump data/rxnorm_raw.json
```

---

## Chạy inference

### Bước 1 — Đặt input vào `input/`

Mỗi file `input/N.txt` là 1 văn bản lâm sàng. Lên đến 100 file.

### Bước 2 — Smoke test

```powershell
cd F:\AI_VIETTEL
python scripts/test_inference.py --out output/smoke_test.json
```

→ Đảm bảo LM Studio server đang chạy với `qwen2.5-7b-instruct` trước.

### Bước 3 — Inference đầy đủ

```powershell
python -m src.inference `
    --input input `
    --output output `
    --workers 1 `
    --target-ctx 4096 `
    --max-few-shot 10
```

**Flags chi tiết:**

| Flag | Mặc định | Mô tả |
|---|---|---|
| `--input` | `data/input` | Thư mục chứa `N.txt` |
| `--output` | `output` | Thư mục ghi `N.json` |
| `--workers` | `1` | Số worker song song (LM Studio 1 request/lần) |
| `--target-ctx` | `4096` | Context Length LM Studio (adaptive cap few-shot) |
| `--max-few-shot` | `10` | Số few-shot examples TỐI ĐA (auto-cap theo budget) |
| `--limit` | `0` | Giới hạn số record (0 = hết) |
| `--log-file` | `predictions.log` | Log realtime |

### Bước 4 — Validate

```powershell
python scripts/validate_outputs.py --input output
```

Output mẫu:
```
✅ 1.json: drugs=11 symp=8 historical=11
✅ 2.json: drugs=0 symp=2 historical=0
...
Tất cả 100 file hợp lệ.
```

---

## Pipeline flow

```
Input: input/N.txt (Vietnamese clinical text)
        │
        ▼
┌────────────────────────────────────────────┐
│  LLM (qwen2.5-7b @ LM Studio)              │
│  → JSON spans: text/type/position/assertions│
│  (LLM được cấm điền candidates)             │
└────────────┬───────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────┐
│  Postprocess                                │
│  • validate_positions (self-heal)           │
│  • dedupe (text, type)                      │
│  • THUỐC            → RxNorm RAG (5 layers) │
│  • CHẨN_ĐOÁN        → VN→EN + ICD RAG      │
│  • TÊN_XÉT_NGHIỆM   → không candidates     │
│  • Split "drug A cho disease B" → 2 entity │
│  • Strip clinical prefix ("chẩn đoán:")    │
└────────────┬───────────────────────────────┘
             │
             ▼
     output/N.json ✅
```

### Adaptive few-shot budget

`inference.py` tự tính số few-shot vừa context:

```
budget = target_ctx - max_output_tokens - reserve
remaining = budget - sys_prompt_tokens   # còn cho few-shot + user input
n_few_shot = min(remaining // 200, max_examples)
```

Với `target_ctx=4096` + `max_tokens=1024`:
- 1.txt (3K chars) → **1 few-shot**
- 2.txt (770 chars) → **4 few-shot**
- 3.txt (4.4K chars) → **0 few-shot** (over budget)

→ Không bao giờ overflow context.

### RxNorm RAG — 5 lớp

1. **Exact match** local seed (sub-second)
2. **Normalized exact** (bỏ dose form / route / freq, giữ strength)
3. **Fuzzy match** (WRatio + partial_ratio)
4. **Live NIH API** `/drugs.json?name=` (filter standalone trước combo, đúng strength)

### ICD-10 RAG — 5 lớp

1. **Exact match** local seed (27 entries)
2. **Translator VN→EN** (preset 60+ cụm: viết tắt THA/ĐTĐ/CVA, multi-word)
3. **Fuzzy match** với partial_ratio
4. **Fuzzy VN text** (khi LLM fail dịch)
5. **Live NIH API** `/icd10cm/v3/search?sf=name`

---

## Cấu hình theo phần cứng

| Phần cứng | Model | `--target-ctx` | Workers | Tốc độ ước tính |
|---|---|---|---|---|
| **RTX 4050 8GB (laptop)** | qwen2.5-7b Q4_K_M | `8192` | 1 | 30-50s/record |
| **RTX 4090 24GB** | qwen3.5-9b Q4_K_M | `12288` | 1-2 | 8-15s/record |
| **CPU only (8GB RAM)** | qwen2.5-3b Q4_K_M | `4096` | 1 | 60-120s/record |
| **CPU (16GB+ RAM)** | qwen2.5-7b Q4_K_M | `6144` | 1 | 60-180s/record |

Bắt đầu với `--target-ctx 4096`. Nếu GPU ≥ 8GB → tăng lên `8192` để có nhiều few-shot diversity.

---

## Đóng gói submission

```powershell
Compress-Archive -Path output,src,scripts,data,requirements.txt,README.md `
    -DestinationPath output_bundle.zip -Force
```

**BTC yêu cầu:**
- `output/1.json .. 100.json`
- Source code (`src/`, `scripts/`)
- Data (giữ index files, bỏ `rxnorm_raw.json` nếu > 50MB)
- Model weights → `models/README.txt` với link tải `qwen2.5-7b-instruct` GGUF
- README này

---

## Khắc phục sự cố

| Triệu chứng | Cách xử lý |
|---|---|
| `Connection refused 1234` | Mở LM Studio → Developer → Start Server |
| `Request timed out` 6 phút mỗi record | Giảm `LLMConfig.timeout` còn `180` |
| `Unterminated string` khi parse JSON | Tăng `LLMConfig.max_tokens` lên `2048` |
| Output trắng `[]` cho mọi record | Kiểm tra model đã load đúng chưa, restart LM Studio |
| Output thiếu `candidates` | Clear `data/rxnorm_api_cache.json` rồi chạy lại |
| Inference chậm (>10 phút/record) | Giảm `max_tokens` hoặc đổi model nhỏ hơn |
| Context window overflow | Giảm `--max-few-shot` xuống `5` |
| LLM trả văn bản không phải JSON | `--max_tokens` đủ lớn để chứa entity list |

---

## Tài liệu kỹ thuật

- [`optimize_lm_studio.md`](optimize_lm_studio.md) — Tối ưu LM Studio theo GPU
- [`src/prompts.py`](src/prompts.py) — SYSTEM_PROMPT đầy đủ 5 phần + 3 assertions + rules A-F
- [`data/examples.jsonl`](data/examples.jsonl) — 22 few-shot examples

## Tham khảo

- LM Studio: <https://lmstudio.ai/>
- Qwen2.5: <https://huggingface.co/Qwen>
- NIH RxNorm REST: <https://rxnav.nlm.nih.gov/InteractionAPIs.html>
- NIH Clinical Tables: <https://clinicaltables.nlm.nih.gov/apidoc/icd10cm/v3/doc.html>
