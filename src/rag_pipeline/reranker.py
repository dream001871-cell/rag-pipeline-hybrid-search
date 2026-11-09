import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .retrieval import RetrievalResult

logger = logging.getLogger(__name__)

DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


class Reranker(ABC):
    """Scores each candidate chunk's relevance to the query.

    The reranker is a precision-improving second pass over the fused candidates
    (e.g. top 20 -> keep top 5). Subclasses implement a concrete scoring
    backend: cross-encoder, LLM-as-judge, or a local lexical fallback.
    """

    @abstractmethod
    def score(self, query: str, candidates: List[RetrievalResult]) -> List[RetrievalResult]:
        """Return candidates with `rerank_score` set, sorted descending."""
        pass

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Re-rank chunk dicts, returning the same dicts sorted by score.

        Convenience dict interface on top of ``score``: each chunk dict must at
        minimum carry ``chunk_id`` and ``content`` (plus optional ``rrf_score``
        and ``dense_score`` / ``sparse_score``). ``rerank_score`` is written
        back onto the dict and the list is re-sorted descending.
        """
        candidates = [
            RetrievalResult(
                chunk_id=chunk.get("chunk_id", ""),
                content=chunk.get("content", ""),
                metadata=chunk.get("metadata") or {},
                dense_score=chunk.get("dense_score"),
                sparse_score=chunk.get("sparse_score"),
            )
            for chunk in chunks
        ]
        for cand, chunk in zip(candidates, chunks):
            cand.rrf_score = chunk.get("rrf_score")

        scored = self.score(query, candidates)
        by_id = {r.chunk_id: r for r in scored}
        for chunk in chunks:
            matched = by_id.get(chunk.get("chunk_id"))
            if matched is not None:
                chunk["rerank_score"] = matched.rerank_score

        reranked = sorted(chunks, key=lambda c: c.get("rerank_score", 0.0), reverse=True)
        if top_k:
            return reranked[:top_k]
        return reranked

    def _reorder(self, candidates: List[RetrievalResult], scores: List[float]) -> List[RetrievalResult]:
        for res, sc in zip(candidates, scores):
            res.rerank_score = float(sc)
        return sorted(candidates, key=lambda r: r.rerank_score, reverse=True)


class LexicalReranker(Reranker):
    """Local, dependency-free reranker using lexical/overlap relevance.

    Scores chunks by how well they cover the query's meaningful tokens
    (term overlap plus a small boost for phrase adjacency). Used as a
    deterministic fallback for offline development or when no model/API is
    configured. Not as strong as a learned cross-encoder, but makes the
    pipeline runnable end-to-end with zero external services.
    """

    def __init__(self, stopwords: Optional[set] = None):
        self.stopwords = stopwords or {
            "the", "a", "an", "and", "or", "of", "to", "in", "on", "for",
            "is", "are", "with", "how", "what", "whats", "do", "does", "i",
            "you", "it", "this", "that", "at", "by", "be", "can", "not",
        }

    def score(self, query: str, candidates: List[RetrievalResult]) -> List[RetrievalResult]:
        query_tokens = self._tokens(query)
        if not query_tokens:
            return self._reorder(candidates, [0.0] * len(candidates))

        scores = []
        for res in candidates:
            chunk_tokens = self._tokens(res.content)
            if not chunk_tokens:
                scores.append(0.0)
                continue
            overlap = sum(1 for t in query_tokens if t in chunk_tokens)
            # Precision: fraction of query tokens covered
            coverage = overlap / len(query_tokens)
            # Slight boost for raw term frequency in the chunk
            freq = sum(chunk_tokens.count(t) for t in query_tokens)
            scores.append(coverage + 0.1 * freq)
        return self._reorder(candidates, scores)

    def _tokens(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return [t for t in text.split() if t and t not in self.stopwords]


class CrossEncoderReranker(Reranker):
    """使用 sentence-transformers CrossEncoder，默认模型支持中文等多种语言。

    Requires `sentence_transformers` to be installed; the model is downloaded
    on first use. Query and document are passed together as a pair for cross
    attention — this is the highest-precision local option.

    Construction is graceful: if sentence-transformers / torch is unavailable
    or the model cannot be loaded, a warning is logged and the instance falls
    back to LexicalReranker so nothing else in the pipeline breaks.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self._encoder = None
        self._fallback: Optional[LexicalReranker] = None
        try:
            self._encoder = self._load_model()
        except Exception as exc:
            logger.warning(
                "CrossEncoder(%s) unavailable (%s); falling back to LexicalReranker.",
                model_name,
                exc,
            )
            self._fallback = LexicalReranker()

    def _load_model(self):
        from sentence_transformers import CrossEncoder
        return CrossEncoder(self.model_name)

    @property
    def available(self) -> bool:
        return self._encoder is not None

    def score(self, query: str, candidates: List[RetrievalResult]) -> List[RetrievalResult]:
        if self._encoder is None:
            return self._fallback.score(query, candidates)
        if not candidates:
            return candidates

        # Batch the whole candidate set into one forward pass: all (query,
        # chunk) pairs together, not one at a time.
        pairs = [(query, c.content) for c in candidates]
        raw = self._encoder.predict(
            pairs, convert_to_numpy=True, batch_size=self.batch_size
        )
        scores = [float(s) for s in raw]
        return self._reorder(candidates, scores)


class LLMJudgeReranker(Reranker):
    """LLM-as-judge reranker using an OpenRouter/OpenAI chat model.

    Asks the model to return a 0-10 relevance score for each (query, chunk)
    pair. Robust and flexible, but costs tokens and latency per query.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        max_chars: int = 1500,
    ):
        from .llm_client import ChatCompletionClient
        self.api_key = api_key
        self.max_chars = max_chars
        self._client = ChatCompletionClient(api_key=api_key)
        self.model = model or (
            self._client.default_model if self._client.provider == "deepseek" else "gpt-4o-mini"
        )
        if not self._client.is_configured:
            raise RuntimeError(
                "模型重排需要配置 DEEPSEEK_API_KEY、OPENROUTER_API_KEY 或 OPENAI_API_KEY"
            )

    def score(self, query: str, candidates: List[RetrievalResult]) -> List[RetrievalResult]:
        system = (
            "You are a retrieval relevance judge. Given a user question and a "
            "document chunk, output ONLY a single integer score from 0 to 10 "
            "indicating how relevant the chunk is to answering the question. "
            "10 = directly answers it, 5 = partially related, 0 = irrelevant."
        )
        scores = []
        for res in candidates:
            snippet = res.content[: self.max_chars]
            prompt = f"Question: {query}\n\nChunk:\n{snippet}"
            score = self._judge(system, prompt)
            scores.append(score)
        return self._reorder(candidates, scores)

    def _judge(self, system: str, prompt: str) -> float:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        text = (resp.choices[0].message.content or "").strip()
        numbers = re.findall(r"\d+", text)
        return float(numbers[0]) if numbers else 0.0


class RerankerFactory:
    """Builds the best available reranker.

    Returns a CrossEncoderReranker by default; falls back to LexicalReranker
    when sentence-transformers is not installed, torch is unavailable, or the
    model download fails. A warning is logged on every fallback.
    """

    def __init__(self, model_name: Optional[str] = None):
        # 显式参数优先，其次读取环境配置，未配置时使用支持中文的多语言模型。
        configured_model = os.environ.get("RAG_RERANKER_MODEL", "").strip()
        self.model_name = model_name or configured_model or DEFAULT_CROSS_ENCODER_MODEL

    def get_reranker(self, kind: str = "auto") -> Reranker:
        if kind == "lexical":
            return LexicalReranker()
        if kind == "llm":
            return LLMJudgeReranker()
        if kind == "cross_encoder":
            return self._cross_encoder_or_lexical()

        # auto (default): cross-encoder first, lexical as fallback.
        if not self._sentence_transformers_available():
            return LexicalReranker()
        return self._cross_encoder_or_lexical()

    def _cross_encoder_or_lexical(self) -> Reranker:
        try:
            return CrossEncoderReranker(model_name=self.model_name)
        except Exception as exc:
            logger.warning(
                "CrossEncoder(%s) unavailable (%s); falling back to LexicalReranker.",
                self.model_name,
                exc,
            )
            return LexicalReranker()

    def _sentence_transformers_available(self) -> bool:
        try:
            import sentence_transformers  # noqa: F401
            return True
        except Exception as exc:
            logger.warning(
                "sentence-transformers not available (%s); using LexicalReranker.",
                exc,
            )
            return False


def create_reranker(
    kind: str = "auto",
    model_name: Optional[str] = None,
    llm_model: Optional[str] = None,
) -> Reranker:
    """Backward-compatible wrapper for :class:`RerankerFactory`.

    Defaults to a CrossEncoderReranker with an automatic LexicalReranker
    fallback. `llm_model` is accepted for compatibility with callers of the old
    factory signature.
    """
    if kind == "llm":
        return LLMJudgeReranker(model=llm_model)
    return RerankerFactory(model_name=model_name).get_reranker(kind=kind)
