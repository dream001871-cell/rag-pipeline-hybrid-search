import logging
import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from rag_pipeline.retrieval import RetrievalResult, DenseRetriever, SparseRetriever
from rag_pipeline.fusion import HybridRetriever
from rag_pipeline.reranker import (
    Reranker,
    LexicalReranker,
    CrossEncoderReranker,
    RerankerFactory,
    create_reranker,
)


def result(chunk_id, content, dense=None, sparse=None):
    return RetrievalResult(
        chunk_id=chunk_id,
        content=content,
        metadata={"doc_id": "x", "chunk_index": 0},
        dense_score=dense,
        sparse_score=sparse,
    )


def fake_cross_encoder_model(predict_return):
    model = MagicMock()
    model.predict.return_value = predict_return
    return model


def test_lexical_reranker_orders_by_relevance():
    query = "how do I configure the API key for authentication"
    candidates = [
        result("c1", "The pricing model bills per seat each month."),
        result(
            "c2",
            "Set your authentication API key in the configuration file to enable access.",
        ),
        result("c3", "Recipes include chocolate cake, bread, and pastries."),
        result(
            "c4",
            "API key configuration: create an authentication key then load it at startup.",
        ),
    ]
    reranker = LexicalReranker()
    reranked = reranker.score(query, candidates)

    assert all(r.rerank_score is not None for r in reranked)
    # Sorted descending by rerank_score
    sc = [r.rerank_score for r in reranked]
    assert sc == sorted(sc, reverse=True)
    # c2 / c4 (both mention api key + authentication) should rank above c1/c3
    assert reranked[0].chunk_id in ("c2", "c4")
    assert reranked[-1].chunk_id == "c3"  # irrelevant recipe chunk last


def test_lexical_reranker_empty_query_keeps_order():
    candidates = [result("a", "first thing"), result("b", "second thing")]
    reranked = LexicalReranker().score("   ", candidates)
    assert all(r.rerank_score == 0.0 for r in reranked)


def test_hybrid_rerank_keeps_top_k():
    class StubDense(DenseRetriever):
        def retrieve(self, query, top_k=None):
            k = top_k or 20
            return [
                result("d1", "api key auth token setup"),
                result("d2", "pricing per seat"),
                result("d3", "rate limit 429 retry"),
            ][:k]

    class StubSparse(SparseRetriever):
        def retrieve(self, query, top_k=None):
            k = top_k or 20
            return [
                result("s1", "error handling retry after header"),
                result("d1", "api key auth token setup"),
            ][:k]

    hybrid = HybridRetriever(
        dense_retriever=StubDense(),
        sparse_retriever=StubSparse(),
        reranker=LexicalReranker(),
        candidate_k=20,
    )
    q = "how to set up an api key and handle rate limits"
    results = hybrid.retrieve(q, top_k=5)

    assert len(results) <= 5
    assert all(r.rerank_score is not None for r in results)
    # Sorted by rerank score desc
    sc = [r.rerank_score for r in results]
    assert sc == sorted(sc, reverse=True)


def test_rerank_can_be_disabled():
    class StubDense(DenseRetriever):
        def retrieve(self, query, top_k=None):
            return [result("a", "alpha"), result("b", "beta"), result("c", "gamma")]

    class StubSparse(SparseRetriever):
        def retrieve(self, query, top_k=None):
            return [result("b", "beta"), result("d", "delta")]

    hybrid = HybridRetriever(
        dense_retriever=StubDense(),
        sparse_retriever=StubSparse(),
        reranker=LexicalReranker(),
    )
    fused = hybrid.retrieve("beta", top_k=3, rerank=False)
    # No rerank_score populated when disabled
    assert all(r.rerank_score is None for r in fused)


def test_cross_encoder_batches_all_candidates_in_one_forward_pass(monkeypatch):
    # Mock the model loader so no real sentence-transformers download happens.
    model = MagicMock()
    model.predict.return_value = [0.1, 0.9, 0.5]
    monkeypatch.setattr(CrossEncoderReranker, "_load_model", lambda self: model)

    reranker = CrossEncoderReranker()
    candidates = [result("c1", "cc1"), result("c2", "cc2"), result("c3", "cc3")]
    reranked = reranker.score("q", candidates)

    # One forward pass over the full batch: predict called exactly once.
    model.predict.assert_called_once()
    args = model.predict.call_args[0][0]
    assert [pair[0] for pair in args] == ["q", "q", "q"]
    assert [pair[1] for pair in args] == ["cc1", "cc2", "cc3"]
    assert reranked[0].chunk_id == "c2"
    assert reranked[1].chunk_id == "c3"
    assert [r.rerank_score for r in reranked] == [0.9, 0.5, 0.1]


def test_cross_encoder_dict_interface_rerank(monkeypatch):
    model = fake_cross_encoder_model([0.9, 0.3, 0.7])
    monkeypatch.setattr(CrossEncoderReranker, "_load_model", lambda self: model)

    reranker = CrossEncoderReranker()
    chunks = [
        {"chunk_id": "a", "content": "aaa", "rrf_score": 0.02, "rerank_score": None},
        {"chunk_id": "b", "content": "bbb", "rrf_score": 0.02, "rerank_score": None},
        {"chunk_id": "c", "content": "ccc", "rrf_score": 0.02, "rerank_score": None},
    ]
    out = reranker.rerank("q", chunks, top_k=2)

    assert [c["chunk_id"] for c in out] == ["a", "c"]
    assert out[0]["rerank_score"] == 0.9
    assert out[1]["rerank_score"] == 0.7
    assert len(out) == 2


def test_cross_encoder_falls_back_when_dependency_missing(monkeypatch, caplog):
    # Simulate sentence-transformers / torch being unavailable.
    def boom(self):
        raise ImportError("simulate: no torch / no sentence_transformers")

    monkeypatch.setattr(CrossEncoderReranker, "_load_model", boom)

    with caplog.at_level(logging.WARNING):
        reranker = CrossEncoderReranker()

    assert reranker._encoder is None
    assert "falling back to LexicalReranker" in caplog.text

    # The instance still works: it delegates to the lexical fallback.
    candidates = [
        result("a", "The pricing model bills per seat each month."),
        result("b", "Set your authentication API key in the configuration file."),
    ]
    reranked = reranker.score("how do I configure the API key", candidates)
    assert reranked[0].chunk_id == "b"
    assert all(r.rerank_score is not None for r in reranked)


def test_cross_encoder_falls_back_when_model_download_fails(monkeypatch, caplog):
    def boom(self):
        raise RuntimeError("simulate: model download failed")

    monkeypatch.setattr(CrossEncoderReranker, "_load_model", boom)

    with caplog.at_level(logging.WARNING):
        reranker = CrossEncoderReranker()

    assert reranker.available is False
    assert reranker._fallback is not None
    assert "falling back to LexicalReranker" in caplog.text
    assert isinstance(reranker.score("q", [result("a", "content")])[0], RetrievalResult)


def test_factory_returns_cross_encoder_by_default(monkeypatch):
    # Pretend sentence-transformers IS installed (without a real install).
    model = fake_cross_encoder_model([0.5])
    fake_st = ModuleType("sentence_transformers")
    fake_st.CrossEncoder = MagicMock(return_value=model)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)

    reranker = RerankerFactory().get_reranker()
    assert isinstance(reranker, CrossEncoderReranker)
    assert reranker._encoder is model
    assert reranker.available


def test_factory_falls_back_when_sentence_transformers_missing(monkeypatch, caplog):
    real_import = __import__

    def blocked_import(name, *args, **kwargs):
        if name == "sentence_transformers" or name.startswith("sentence_transformers."):
            raise ImportError("simulate: not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", blocked_import)

    with caplog.at_level(logging.WARNING):
        reranker = RerankerFactory().get_reranker()

    assert isinstance(reranker, LexicalReranker)
    assert "LexicalReranker" in caplog.text


def test_factory_kind_lexical():
    assert isinstance(RerankerFactory().get_reranker("lexical"), LexicalReranker)


def test_create_reranker_remains_compatible():
    r = create_reranker("lexical")
    assert isinstance(r, LexicalReranker)
    assert isinstance(create_reranker(), Reranker)


def test_factory_uses_configured_model(monkeypatch):
    monkeypatch.setenv("RAG_RERANKER_MODEL", "local/chinese-reranker")
    assert RerankerFactory().model_name == "local/chinese-reranker"


def test_explicit_model_takes_precedence_over_environment(monkeypatch):
    monkeypatch.setenv("RAG_RERANKER_MODEL", "local/chinese-reranker")
    assert RerankerFactory(model_name="explicit/model").model_name == "explicit/model"


def test_blank_model_setting_uses_default_model(monkeypatch):
    from rag_pipeline.reranker import DEFAULT_CROSS_ENCODER_MODEL

    monkeypatch.setenv("RAG_RERANKER_MODEL", "  ")
    assert RerankerFactory().model_name == DEFAULT_CROSS_ENCODER_MODEL
