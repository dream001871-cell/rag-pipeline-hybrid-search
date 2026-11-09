import logging
import os
import time
from typing import Dict, List, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OPENROUTER_KEY_PREFIX = "sk-or-"


class EmbeddingService:
    """Generates embeddings using OpenAI text-embedding-3-small.

    The encoder is idempotent per text and reused across the vector store and
    semantic chunking so that the same text always maps to the same vector.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        batch_size: int = 100,
        dimensions: Optional[int] = None,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.batch_size = batch_size
        self.dimensions = dimensions
        self._client = OpenAI(api_key=self.api_key) if self.api_key else None
        self._cache: Dict[str, List[float]] = {}

    @property
    def is_configured(self) -> bool:
        return self._client is not None

    def embed(self, text: str) -> List[float]:
        """Embed a single text string (cached)."""
        if text in self._cache:
            return self._cache[text]
        vector = self.embed_batch([text])[0]
        self._cache[text] = vector
        return vector

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts, honoring the configured batch size."""
        if not self.is_configured:
            raise RuntimeError(
                "尚未配置向量服务密钥。请设置 OPENAI_API_KEY 或传入 api_key。"
            )
        if not texts:
            return []

        vectors: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            kwargs = {"model": self.model, "input": batch}
            if self.dimensions:
                kwargs["dimensions"] = self.dimensions
            response = self._client.embeddings.create(**kwargs)
            # Responses are ordered by input index
            ordered = sorted(response.data, key=lambda d: d.index)
            vectors.extend([item.embedding for item in ordered])
            if i + self.batch_size < len(texts):
                time.sleep(0.05)  # gentle rate limiting
        return vectors

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string (cached)."""
        return self.embed(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        return self.embed_batch(texts)

    def clear_cache(self) -> None:
        self._cache.clear()


OpenAIEmbeddingService = EmbeddingService


class LocalEmbeddingService(EmbeddingService):
    """Local embeddings via sentence-transformers.

    Runs the full pipeline without an OpenAI key. The SentenceTransformer model
    is loaded once at construction and reused; ``embed_documents`` encodes the
    whole batch in a single ``.encode()`` call.
    """

    def __init__(
        self,
        model_name: str = LOCAL_EMBEDDING_MODEL,
        batch_size: int = 100,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "未安装 sentence-transformers。"
                "请执行：pip install sentence-transformers"
            ) from exc
        self.model = model_name
        self.model_name = model_name
        self.api_key = "local"
        self.batch_size = batch_size
        self._cache: Dict[str, List[float]] = {}
        self._encoder = SentenceTransformer(model_name)
        self.dimensions = int(self._encoder.get_sentence_embedding_dimension())
        logger.info(
            "本地向量模型：%s（%d 维）", model_name, self.dimensions
        )

    @property
    def is_configured(self) -> bool:
        return True

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vectors = self._encoder.encode(
            texts, batch_size=self.batch_size, convert_to_numpy=True
        )
        return [list(map(float, v)) for v in vectors]

    def embed_query(self, text: str) -> List[float]:
        return self.embed(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_batch(texts)


class MockEmbeddingService(EmbeddingService):
    """Deterministic, dependency-free embedding for offline development/testing.

    Produces a stable hash-based vector so the pipeline can run end-to-end
    without an OpenAI key, and can be swapped out for the real service.
    """

    def __init__(self, dimensions: int = 384):
        self.model = "mock"
        self.api_key = "mock"
        self.batch_size = 100
        self.dimensions = dimensions
        self._client = None
        self._cache = {}

    @property
    def is_configured(self) -> bool:
        return True

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self._hash_embed(t) for t in texts]

    def _hash_embed(self, text: str) -> List[float]:
        import hashlib
        vec = [0.0] * self.dimensions
        h = hashlib.sha256(text.encode("utf-8")).digest()
        nh = int.from_bytes(h, "big")
        for i in range(self.dimensions):
            nh_next, rem = divmod(nh, 7)
            nh = nh_next
            vec[i] = (rem / 7.0) * 2 - 1  # normalize to approx [-1, 1]
        norm = sum(x * x for x in vec) ** 0.5
        return [x / norm for x in vec] if norm > 0 else vec


def _local_embeddings_enabled() -> bool:
    value = os.environ.get("LOCAL_EMBEDDINGS", "true").strip().lower()
    return value not in ("0", "false", "no")


def _has_real_openai_key() -> bool:
    # DeepSeek 负责文本生成，向量保持本地计算，避免请求不存在的向量接口或切换索引维度。
    if os.environ.get("DEEPSEEK_API_KEY", "").strip():
        return False
    key = os.environ.get("OPENAI_API_KEY", "")
    return bool(key) and not key.startswith(OPENROUTER_KEY_PREFIX)


def create_embedding_service(
    model: str = "text-embedding-3-small", use_mock: Optional[bool] = None
) -> EmbeddingService:
    """显式模拟配置优先；DeepSeek 使用本地向量，其余服务保留原有选择规则。"""
    if use_mock is None:
        use_mock = not _local_embeddings_enabled()
    if use_mock:
        return MockEmbeddingService()

    if _has_real_openai_key():
        return OpenAIEmbeddingService(model=model)

    if os.environ.get("OPENROUTER_API_KEY"):
        logger.warning(
            "已配置 OpenRouter，但没有 OpenAI 向量服务密钥；"
            "使用本地向量模型（%s）。",
            LOCAL_EMBEDDING_MODEL,
        )
    return LocalEmbeddingService(model_name=LOCAL_EMBEDDING_MODEL)


def describe_embedding_service(
    model: str = "text-embedding-3-small",
) -> str:
    """Return a human-readable label for the active embedding backend without
    loading the model."""
    if not _local_embeddings_enabled():
        return "模拟向量（测试模式）"
    if _has_real_openai_key():
        return f"OpenAI {model}"
    return f"本地模型 {LOCAL_EMBEDDING_MODEL}"


class EmbeddingFactory:
    """Resolves the active embedding service via :func:`create_embedding_service`."""

    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model

    def get_service(self) -> EmbeddingService:
        return create_embedding_service(model=self.model)
