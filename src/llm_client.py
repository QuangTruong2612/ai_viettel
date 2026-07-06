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

    # max_tokens=2048 đủ chứa JSON ~20 entities (~80 chars mỗi).
    # Giữ thấp để tránh context overflow khi input dài + few-shot nhiều.
    max_tokens: int = 2048
    timeout: int = 180
    max_retries: int = 1  # giảm retry để fail fast

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
        """Trích JSON ra khỏi content.

        LLM có thể trả:
        - JSON thuần (lý tưởng)
        - JSON trong ```json ... ```
        - Có text thừa quanh JSON
        Hàm này cố gắng robust với cả 3 trường hợp.
        """
        text = content.strip()
        # Bỏ code fence nếu có
        if text.startswith("```"):
            # cắt đến khối ``` đầu tiên đóng
            parts = text.split("```")
            if len(parts) >= 3:
                text = "```".join(parts[1:-1]).strip()
                if text.lower().startswith("json"):
                    text = text[4:].strip()
        # Nếu có text thừa, thử tìm [ đầu tiên và ] cuối cùng
        if not text.startswith("[") and not text.startswith("{"):
            first = text.find("[")
            last = text.rfind("]")
            if first != -1 and last != -1 and last > first:
                text = text[first : last + 1]
        return json.loads(text)
