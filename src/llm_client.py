"""LLM client — wrapper quanh LM Studio OpenAI-compatible API.

LM Studio mặc định listen ở http://localhost:1234/v1 và trả response
theo chuẩn OpenAI chat completions. Module này đóng gói để:

- Timeout an toàn (một số record dài → inference lâu).
- Bắt buộc JSON-only output (cho nhiều local model không bật JSON mode
  được thì ép qua system prompt + parser bên dưới).
- Retry với backoff đơn giản cho lỗi tạm thời.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Cấu hình kết nối LM Studio / Ollama."""

    # LM Studio listen ở :1234/v1; Ollama listen ở :11434/v1 (OpenAI-compatible).
    # Default: Ollama (vì model người dùng "qwen2.5:7b" là Ollama naming convention).
    base_url: str = "http://127.0.0.1:11434/v1"
    api_key: str = "lm-studio"  # LM Studio chấp nhận bất kỳ; Ollama ignore
    model: str = "qwen2.5-7b-instruct"  # Ollama naming với colon (LM Studio: "qwen2.5-7b-instruct")
    temperature: float = 0.0
    top_p: float = 1.0

    # max_tokens=768 đủ chứa JSON ~25 entities (~30 chars mỗi).
    # 9b có thể output dài hơn 4b; restore từ 512 → 768 cho đủ entities.
    max_tokens: int = 768
    timeout: int = 300  # 5 phút/request (9b chậm hơn 7b, tránh timeout)
    max_retries: int = 1  # giảm retry để fail fast

    # Ollama-specific: keep_alive. Default "0" → UNLOAD model sau mỗi request
    # → giải phóng VRAM, tránh OOM khi 9b trên Kaggle T4x2.
    # Set "5m" nếu muốn giữ model load giữa các request (nhanh hơn nhưng tốn VRAM).
    keep_alive: str = "0"

    # Ollama-specific: num_ctx override PER-REQUEST (qua extra_body).
    # Default 8192 cho Kaggle T4x2 (16GB VRAM):
    # - qwen2.5:7b FP16 (~5GB) → 8192 safe, 16384 tight
    # - qwen3.5:9b FP16 (~5.5GB) → 8192 RECOMMENDED, 16384 có thể OOM
    # - Quantized Q4_K_M (~3GB) → 16384 comfortable
    # - 32768 chỉ work với quantized + GPU offload
    # Override qua env OLLAMA_NUM_CTX hoặc --target-ctx.
    num_ctx: int = 8192

    # Ollama-specific: num_gpu layers. -1 = all (default).
    # Giảm nếu OOM (vd num_gpu=20 → 20 layer trên GPU, phần còn trên CPU/RAM).
    num_gpu: int = -1

    @classmethod
    def from_env(cls) -> "LLMConfig":
        # Support cả LM Studio (1234) và Ollama (11434) — đều OpenAI-compatible.
        # Để dùng LM Studio: đặt LMSTUDIO_BASE_URL=http://127.0.0.1:1234/v1
        # Để dùng Ollama:       đặt OLLAMA_BASE_URL=http://127.0.0.1:11434/v1
        # Ưu tiên OLLAMA_BASE_URL nếu set, fallback LM Studio.
        ollama_url = os.environ.get("OLLAMA_BASE_URL")
        lm_url = os.environ.get("LMSTUDIO_BASE_URL", cls.base_url)
        chosen_url = ollama_url or lm_url
        return cls(
            base_url=chosen_url,
            api_key=os.environ.get("OLLAMA_API_KEY")
            or os.environ.get("LMSTUDIO_API_KEY", cls.api_key),
            model=os.environ.get("OLLAMA_MODEL")
            or os.environ.get("LMSTUDIO_MODEL", cls.model),
            temperature=float(os.environ.get("LMSTUDIO_TEMPERATURE", cls.temperature)),
            max_tokens=int(os.environ.get("LMSTUDIO_MAX_TOKENS", cls.max_tokens)),
            timeout=int(os.environ.get("LMSTUDIO_TIMEOUT", cls.timeout)),
            num_ctx=int(os.environ.get("OLLAMA_NUM_CTX", cls.num_ctx)),
        )


class LLMClient:
    """Client gọi LM Studio / OpenAI-compatible local server.

    Dùng thư viện `openai` chính thức để tận dụng retry/tương thích chuẩn.
    """

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        self.config = config or LLMConfig.from_env()
        # Import lazy để tránh crash nếu người dùng chưa cài openai.
        try:
            from openai import OpenAI  # type: ignore

            # max_retries=0: tắt retry mặc định của openai lib; ta tự retry ở _call_with_retry
            # để không bị hang 30+ phút khi LM Studio quá tải.
            self._client = OpenAI(
                base_url=self.config.base_url,
                api_key=self.config.api_key,
                timeout=self.config.timeout,
                max_retries=0,
            )
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Chưa cài thư viện openai. Chạy: pip install -r requirements.txt"
            ) from exc

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_json(content: str) -> Any:
        """Trích JSON ra khỏi content LLM.

        LLM có thể trả:
        - JSON thuần (lý tưởng)
        - JSON trong ```json ... ``` code fence
        - Markdown có text thừa quanh JSON
        - JSON lỗi nhẹ (trailing comma, missing comma, v.v.)

        Hàm này robust:
          1. Strip code fence (```json ... ``` hoặc ``` ... ```)
          2. Tìm [ đầu + ] cuối (nếu có text thừa)
          3. Thử parse strict
          4. Nếu fail, thử JSON repair (trailing commas, single quotes)
          5. Nếu vẫn fail, raise để caller retry/log
        """
        text = content.strip()

        # 1. Strip code fence (```json hoặc ```)
        if text.startswith("```"):
            parts = text.split("```")
            if len(parts) >= 3:
                inner = "```".join(parts[1:-1]).strip()
                if inner.lower().startswith("json"):
                    inner = inner[4:].strip()
                if inner.startswith(("[", "{")):
                    text = inner

        # 2. Nếu có text thừa quanh JSON, tìm [ đầu + ] cuối
        if not text.startswith(("[", "{")):
            first_bracket = text.find("[")
            last_bracket = text.rfind("]")
            if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
                text = text[first_bracket: last_bracket + 1]
            else:
                first_brace = text.find("{")
                last_brace = text.rfind("}")
                if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                    text = text[first_brace: last_brace + 1]

        # 3. Thử parse strict
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 4. Thử repair: trailing commas + close brackets nếu bị truncate
        repaired_text = _repair_json_text(text)
        if repaired_text is not None:
            try:
                return json.loads(repaired_text)
            except json.JSONDecodeError:
                pass

        # 5. Fallback cuối: extract từng {} riêng lẻ (khi array bị truncate giữa)
        objects = _extract_partial_objects(text)
        if objects:
            logger.warning(
                "JSON truncated; recovered %d entities (partial result)",
                len(objects),
            )
            return objects

        # 6. Fail — log full content rồi raise
        logger.error(
            "JSON extract fail. Full content (%d chars):\n%s",
            len(content),
            content if len(content) <= 2000 else content[:2000] + "\n...[truncated]",
        )
        raise json.JSONDecodeError(
            "Không parse được JSON từ LLM response",
            content[:200],
            0,
        )


def _repair_json_text(text: str) -> str | None:
    """Repair JSON lỗi nhẹ: trailing commas + close brackets nếu bị truncate.

    Returns text đã sửa, hoặc None nếu không sửa được.
    Caller tự parse lại sau.
    """
    import re

    repaired = text
    # 1. Strip trailing commas trước ] hoặc }
    repaired = re.sub(r",(\s*[\]}])", r"\1", repaired)

    # 2. Nếu text bị truncate (thiếu closing ] hoặc }), cân đối lại
    # Đếm số { và } — nếu lệch thì thêm } hoặc ]
    open_braces = repaired.count("{") - repaired.count("}")
    open_brackets = repaired.count("[") - repaired.count("]")
    if open_braces > 0:
        repaired += "}" * open_braces
    if open_brackets > 0:
        repaired += "]" * open_brackets

    return repaired if (repaired != text or open_braces or open_brackets) else None


def _extract_partial_objects(text: str) -> list[Any]:
    """Extract từng {} object parse được từ text bị truncate.

    Dùng regex để match các complete JSON object { ... } trong text.
    Thử parse từng cái — bỏ qua cái lỗi.
    """
    import re

    # Match các {...} objects (không nested quá sâu)
    objects: list[Any] = []
    # Greedy match balanced braces
    i = 0
    while i < len(text):
        if text[i] == "{":
            depth = 1
            j = i + 1
            in_str = False
            escape = False
            while j < len(text) and depth > 0:
                ch = text[j]
                if in_str:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_str = False
                else:
                    if ch == '"':
                        in_str = True
                    elif ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                j += 1
            if depth == 0:
                candidate = text[i:j]
                try:
                    obj = json.loads(candidate)
                    objects.append(obj)
                except json.JSONDecodeError:
                    pass
                i = j
            else:
                i += 1
        else:
            i += 1
    return objects
