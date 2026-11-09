import json
import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import httpx
from openai import OpenAI
import pytest

from rag_pipeline.generation import Generator
from rag_pipeline.citation import CitationVerifier
from rag_pipeline.reranker import LLMJudgeReranker


@pytest.fixture
def deepseek_requests(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://other-provider.invalid/v1")
    captured = []

    def respond(request):
        captured.append(request)
        return httpx.Response(200, json={
            "id": "test-completion", "object": "chat.completion", "created": 0,
            "model": "deepseek-flash",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "测试回答 [1]。"}, "finish_reason": "stop"}],
        })

    def client(*, api_key, base_url=None):
        return OpenAI(
            api_key=api_key, base_url=base_url,
            http_client=httpx.Client(transport=httpx.MockTransport(respond)),
        )

    monkeypatch.setattr("openai.OpenAI", client)
    return captured


def test_one_key_selects_deepseek_and_disables_thinking(deepseek_requests):
    generator = Generator()
    result = generator.generate("测试问题", [])
    request = deepseek_requests[0]
    payload = json.loads(request.content)
    assert str(request.url) == "https://api.deepseek.com/chat/completions"
    assert request.headers["authorization"] == "Bearer test-deepseek-key"
    assert payload["model"] == "deepseek-flash"
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["max_tokens"] == 1536
    assert result["answer"] == "测试回答 [1]。"


def test_explicit_model_is_preserved(deepseek_requests):
    Generator(model="deepseek-v4-pro").generate("测试", [])
    assert json.loads(deepseek_requests[0].content)["model"] == "deepseek-v4-pro"


def test_judges_select_deepseek_instead_of_openai_models(deepseek_requests):
    assert CitationVerifier().judge.model == "deepseek-flash"
    assert LLMJudgeReranker().model == "deepseek-flash"


def test_without_deepseek_key_existing_provider_default_is_preserved(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    generator = Generator()
    assert generator.is_configured
    assert generator.model == "gpt-4o"
    assert generator._client.provider == "openai"


def test_real_evaluation_uses_deepseek_and_local_embeddings(monkeypatch, deepseek_requests):
    from rag_evaluation.eval_metrics import resolve_eval_mode

    embedding = SimpleNamespace(model="local-test-embedding")
    monkeypatch.setattr("rag_pipeline.embedding.create_embedding_service", lambda: embedding)
    mode = resolve_eval_mode(force_real=True)
    assert mode["embedding_service"] is embedding
    assert mode["embedding_model"] == "local-test-embedding"
    assert mode["judge_model"] == "deepseek-flash"


@pytest.mark.parametrize("key, preference, expected", [
    ("", "auto", "true"),
    ("test-deepseek-key", "auto", "false"),
    ("test-deepseek-key", "true", "true"),
])
def test_launcher_automatically_selects_retrieval_or_answer_mode(monkeypatch, key, preference, expected):
    captured = {}
    monkeypatch.setenv("DEEPSEEK_API_KEY", key)
    monkeypatch.setenv("RAG_RETRIEVAL_ONLY", preference)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(sys, "argv", ["run_local.py", "api"])

    def run(*args, **kwargs):
        captured["retrieval_only"] = os.environ["RAG_RETRIEVAL_ONLY"]

    monkeypatch.setattr("uvicorn.run", run)
    runpy.run_path(str(Path(__file__).parent.parent / "scripts" / "run_local.py"), run_name="__main__")
    assert captured["retrieval_only"] == expected
