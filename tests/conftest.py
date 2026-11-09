import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "dashboards"))

import pytest

from rag_pipeline.embedding import MockEmbeddingService
from rag_evaluation.golden_dataset import GoldenDataset, GoldenQuestion


@pytest.fixture(autouse=True)
def _no_api_keys(monkeypatch):
    """Blank chat API keys for every test.

    load_dotenv() at import time (api/llm_client/seed) injects the real key
    from .env into the process, and some imports are lazy (e.g. llm_client is
    imported inside Generator.__init__). To keep tests hermetic regardless of
    import order, both keys are set to empty strings: load_dotenv() never
    overrides an already-set variable, and ChatCompletionClient treats an
    empty value as unconfigured.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.setenv("RAG_TRACE_ENDPOINT", "")
    # 单元测试不下载模型；真实模型检索由本机 HTTP 验证脚本单独检查。
    monkeypatch.setenv("LOCAL_EMBEDDINGS", "false")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")


@pytest.fixture
def mock_embedding() -> MockEmbeddingService:
    """A deterministic, offline embedding service (128 dims)."""
    return MockEmbeddingService(dimensions=128)


@pytest.fixture
def tmp_chroma_dir(tmp_path) -> Path:
    """A temporary, isolated directory for a ChromaDB persist_dir."""
    return tmp_path / "chroma"


@pytest.fixture
def sample_chunks() -> list[dict]:
    """Five synthetic chunks (one per request) to exercise retrieval/generation."""
    return [
        {
            "chunk_id": "c1",
            "content": "The API requires a bearer token issued by the admin dashboard.",
            "source_file": "guide.md",
            "metadata": {"doc_id": "guide", "chunk_index": 0, "section_heading": "Auth"},
        },
        {
            "chunk_id": "c2",
            "content": "Rate limits are enforced per account tier on the API.",
            "source_file": "guide.md",
            "metadata": {"doc_id": "guide", "chunk_index": 1, "section_heading": "Limits"},
        },
        {
            "chunk_id": "c3",
            "content": "Backups run nightly at 02:00 UTC.",
            "source_file": "operations-faq.md",
            "metadata": {"doc_id": "faq", "chunk_index": 0, "section_heading": "Backups"},
        },
        {
            "chunk_id": "c4",
            "content": "Alerts page the on-call engineer after 15 minutes.",
            "source_file": "operations-faq.md",
            "metadata": {"doc_id": "faq", "chunk_index": 1, "section_heading": "On-call"},
        },
        {
            "chunk_id": "c5",
            "content": "Widgets are billed at $0.05 per thousand renders.",
            "source_file": "pricing.md",
            "metadata": {"doc_id": "pricing", "chunk_index": 0, "section_heading": "Pricing"},
        },
    ]


@pytest.fixture
def golden_dataset() -> GoldenDataset:
    """A minimal golden dataset: one lookup, one no_answer, one ambiguous."""
    return GoldenDataset(questions=[
        GoldenQuestion(
            id="g1",
            category="lookup",
            question="What auth is required for the API?",
            answer="A bearer token is required.",
            source_documents=["guide.md"],
            source_sections=["Auth"],
        ),
        GoldenQuestion(
            id="g2",
            category="no_answer",
            question="What is the refund policy after cancellation?",
            answer="Not stated in the corpus.",
            source_documents=[],
            source_sections=[],
        ),
        GoldenQuestion(
            id="g3",
            category="ambiguous",
            question="How much does a widget cost?",
            answer="Pricing depends on the volume tier.",
            source_documents=["pricing.md"],
            source_sections=["Pricing"],
        ),
    ])
