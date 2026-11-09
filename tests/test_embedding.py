import pytest

from rag_pipeline.embedding import (
    EmbeddingFactory,
    EmbeddingService,
    LocalEmbeddingService,
    MockEmbeddingService,
)


class FakeSentenceTransformer:
    """Mimics the SentenceTransformer surface; never downloads a model."""

    dim = 384

    def __init__(self, *args, **kwargs):
        self.encode_calls = []

    def get_sentence_embedding_dimension(self):
        return self.dim

    def encode(self, sentences, **kwargs):
        n = len(sentences) if isinstance(sentences, (list, tuple)) else 1
        self.encode_calls.append((sentences, kwargs))
        return [
            [round(0.5 + i * 1e-6 + j * 1e-9, 9) for j in range(self.dim)]
            for i in range(n)
        ]


@pytest.fixture
def fake_sentence_transformer(monkeypatch):
    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer", FakeSentenceTransformer
    )


@pytest.fixture
def no_keys(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("LOCAL_EMBEDDINGS", raising=False)


def test_local_embed_query_returns_correct_dimension(
    fake_sentence_transformer, no_keys
):
    svc = LocalEmbeddingService()
    v = svc.embed_query("hello world")
    assert len(v) == 384
    assert all(isinstance(x, float) for x in v)


def test_local_embed_documents_batch(fake_sentence_transformer, no_keys):
    svc = LocalEmbeddingService()
    texts = ["one two", "three four", "five six"]
    out = svc.embed_documents(texts)
    assert len(out) == 3
    assert [len(v) for v in out] == [384, 384, 384]
    assert len(svc._encoder.encode_calls) == 1
    encoded_sentences = svc._encoder.encode_calls[0][0]
    assert list(encoded_sentences) == texts


def test_local_embed_documents_empty_list(fake_sentence_transformer, no_keys):
    svc = LocalEmbeddingService()
    assert svc.embed_documents([]) == []


def test_factory_picks_local_when_no_openai_key(fake_sentence_transformer, no_keys):
    svc = EmbeddingFactory().get_service()
    assert isinstance(svc, LocalEmbeddingService)
    assert len(svc.embed_query("test")) == 384


def test_factory_picks_local_with_openrouter_key(
    monkeypatch, fake_sentence_transformer, no_keys
):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test-key")
    svc = EmbeddingFactory().get_service()
    assert isinstance(svc, LocalEmbeddingService)
    assert svc.is_configured


def test_factory_picks_openai_with_real_key(monkeypatch, no_keys):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-openrouter")
    svc = EmbeddingFactory().get_service()
    assert isinstance(svc, EmbeddingService)
    assert not isinstance(svc, (LocalEmbeddingService, MockEmbeddingService))


def test_factory_picks_mock_when_local_embeddings_false(monkeypatch, no_keys):
    monkeypatch.setenv("LOCAL_EMBEDDINGS", "false")
    svc = EmbeddingFactory().get_service()
    assert isinstance(svc, MockEmbeddingService)


def test_deepseek_keeps_local_embeddings_even_with_inherited_openai_key(
    monkeypatch, fake_sentence_transformer, no_keys
):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-inherited-openai-key")
    svc = EmbeddingFactory().get_service()
    assert isinstance(svc, LocalEmbeddingService)
    assert len(svc.embed_query("测试")) == 384
