# Tối ưu LM Studio cho qwen2.5-7b-instruct

Pipeline chạy chậm (~3 min cho input 3K chars) vì cấu hình LM Studio chưa tối ưu. Làm theo các bước sau:

## 1. Trong LM Studio → Settings

### Context Length
- **Hiện tại thường là 8192** (default)
- **Đổi thành 4096** → giảm RAM + tăng tốc ~2x
- Nếu 100 record có record dài hơn 4096 tokens (~3000 chars tiếng Việt + output) → KHÔNG cần 8192

### GPU Offload
- **Offload = Max** (kéo hết tất cả layers lên GPU)
- Kiểm tra tab "Resources" hiển thị GPU memory usage — phải chiếm >60% VRAM
- Nếu CPU-only (không có GPU rời): xem mục 3

### Cache Type K / Q
- Q4_K_M (mặc định Qwen3.5-9b) — tốt rồi
- Q5_K_M nếu máy có nhiều RAM (>16 GB) → chính xác hơn 5%

## 2. Restart server sau khi đổi settings

```
Developer → Stop Server → Start Server (lại)
```

Verify:
```powershell
curl -X POST http://127.0.0.1:1234/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"qwen2.5-7b-instruct","messages":[{"role":"user","content":"Reply with one word: OK"}],"max_tokens":5}'
```

Phải trả trong **<10s** nếu GPU offload tốt.

## 3. Nếu vẫn chậm

### Option A: Đổi model nhỏ hơn (nhanh gấp 3-4x)
- Tải `Qwen2.5-3B-Instruct-GGUF` (Q4_K_M ~2GB)
- Trong LM Studio → My Models → chọn → "Load"
- Set `--workers 1` vẫn OK
- Sửa `LLMConfig.model = "qwen2.5-3b-instruct"` trong [src/llm_client.py](src/llm_client.py)

### Option B: Skip record quá dài (>3000 chars)
Mình có thể thêm logic chunk ở [src/inference.py](src/inference.py): nếu input > 3000 chars, chỉ lấy phần "Tiền sử bệnh hiện tại + Triệu chứng" + skip phần còn lại.

### Option C: Lower max_tokens
Sửa trong [src/llm_client.py](src/llm_client.py):
```python
max_tokens: int = 512  # was 1024
```
→ LLM generate ngắn hơn → nhanh hơn. Risk: output bị cắt nếu văn bản có nhiều entity.

## 4. Test nhanh

```powershell
cd F:\AI_VIETTEL
python -m src.inference --input input --output output --workers 1 --limit 3
```

| Setup | Time/record (input 3K chars) |
|---|---|
| qwen2.5-7b-instruct, context 8192 | ~5-10 min |
| **qwen2.5-7b-instruct, context 4096, GPU offload Max** | ~1-3 min |
| qwen2.5-3b-instruct, context 4096 | ~30s |
