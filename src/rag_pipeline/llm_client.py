"""统一聊天接口：显式密钥优先，其次为 DeepSeek、OpenRouter、OpenAI 环境配置。"""

from dotenv import load_dotenv
load_dotenv()

import os
from typing import Optional

OPENROUTER_KEY_PREFIX = "sk-or-v1-"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_DEFAULT_MODEL = "deepseek-flash"


def is_openrouter_key(key: str) -> bool:
    """Return True if the key is an OpenRouter API key (``sk-or-v1-...``)."""
    return key.startswith(OPENROUTER_KEY_PREFIX)


class ChatCompletionClient:
    """保留统一的聊天调用方式，并为 DeepSeek 预设官方地址和模型。"""

    def __init__(self, api_key: Optional[str] = None):
        self.provider: Optional[str] = None
        self._client = None
        self.default_model = "gpt-4o"

        key = api_key
        if key is not None:
            provider = "openrouter" if is_openrouter_key(key) else "openai"
        elif os.environ.get("DEEPSEEK_API_KEY", "").strip():
            key = os.environ["DEEPSEEK_API_KEY"].strip()
            provider = "deepseek"
        elif os.environ.get("OPENROUTER_API_KEY"):
            key = os.environ["OPENROUTER_API_KEY"]
            provider = "openrouter"
        elif os.environ.get("OPENAI_API_KEY"):
            key = os.environ["OPENAI_API_KEY"]
            provider = "openrouter" if is_openrouter_key(key) else "openai"
        else:
            return

        if provider == "openrouter":
            try:
                from openrouter import OpenRouter
            except ImportError:
                raise RuntimeError(
                    "未安装 openrouter，请执行：pip install openrouter"
                )
            self._client = OpenRouter(api_key=key)
        else:
            try:
                from openai import OpenAI
            except ImportError:
                raise RuntimeError("未安装 openai 依赖")
            if provider == "deepseek":
                # DeepSeek 密钥固定发往官方地址，不继承其他服务的 OPENAI_BASE_URL。
                self._client = OpenAI(api_key=key, base_url=DEEPSEEK_BASE_URL)
                self.default_model = DEEPSEEK_DEFAULT_MODEL
            else:
                self._client = OpenAI(api_key=key)
        self.provider = provider

    @property
    def is_configured(self) -> bool:
        return self._client is not None

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, model: str, messages=None, temperature: float = 0.0, **kwargs):
        client = self._client
        if client is None:
            raise RuntimeError(
                "尚未配置模型客户端。请设置 DEEPSEEK_API_KEY、OPENROUTER_API_KEY 或 "
                "OPENAI_API_KEY，或传入 api_key。"
            )
        if self.provider == "openrouter":
            if model and "/" not in model:
                model = f"openai/{model}"
            return client.chat.send(
                model=model,
                messages=messages,
                temperature=temperature,
                **kwargs,
            )
        if self.provider == "deepseek":
            # 普通回答和短格式判分关闭思考模式，避免思考过程占满输出预算。
            kwargs.setdefault("extra_body", {"thinking": {"type": "disabled"}})
        return client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            **kwargs,
        )
