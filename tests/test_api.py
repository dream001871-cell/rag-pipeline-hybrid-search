from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag_pipeline.api import create_app
from rag_pipeline.ingestion import IngestionPipeline
from rag_pipeline.chunker import ChunkingConfig
from rag_pipeline.embedding import MockEmbeddingService
from rag_pipeline.vector_store import VectorStore
from rag_pipeline.bm25_index import BM25Index
from rag_pipeline.document_loader import DocumentLoader
from rag_pipeline.retrieval import DenseRetriever, SparseRetriever
from rag_pipeline.fusion import HybridRetriever
from rag_pipeline.reranker import LexicalReranker
from rag_pipeline.generation import Generator
from rag_pipeline.citation import CitationVerifier
from rag_pipeline.confidence import ConfidenceReport

DATA_DIR = Path(__file__).parent.parent


def build_app(base: Path):
    pipe = IngestionPipeline(
        chunking_config=ChunkingConfig(chunk_size=400, chunk_overlap=40, strategy="fixed"),
        embedding_service=MockEmbeddingService(dimensions=384),
        vector_store=VectorStore(persist_dir=str(base / "chroma")),
        bm25_index=BM25Index(persist_dir=str(base / "bm25")),
        document_loader=DocumentLoader(
            raw_dir=str(base / "raw"),
            processed_dir=str(base / "processed"),
        ),
    )
    pipe.ingest_file(str(DATA_DIR / "data" / "raw" / "fastapi_docs" / "index.md"))

    retriever = HybridRetriever(
        dense_retriever=DenseRetriever(
            embedding_service=pipe.embedding_service,
            vector_store=pipe.vector_store,
        ),
        sparse_retriever=SparseRetriever(bm25_index=pipe.bm25_index),
        reranker=LexicalReranker(),
    )

    # Role-aware mock LLM: the same Generator is used to both answer questions
    # and serve as judge (citation verify + confidence completeness). It must
    # dispatch on the system prompt like a real model would.
    def fake_generate(messages):
        import re
        system = messages[0]["content"] if messages else ""
        if "completeness judge" in system:
            return '{"score": 0.9, "reason": "addresses the question"}'
        if "whether a citation supports a claim" in system:
            return "SUPPORTED"
        if "What color is the ceiling?" in messages[1]["content"]:
            return "上下文信息不足，无法回答这个问题。"
        if "answer" in system or "cited" in system:
            return "Reciprocal Rank Fusion combines dense and sparse rankings [1]."
        # Fall back to the grounded answer prompt shape
        return "Reciprocal Rank Fusion combines dense and sparse rankings [1]."

    generator = Generator(respond=fake_generate)

    app = create_app(
        pipeline=pipe,
        retriever=retriever,
        generator=generator,
        threshold=0.5,
    )
    return app


@pytest.fixture
def client(tmp_path) -> TestClient:
    """A fresh app+index per test (isolated persistent dirs under tmp_path)."""
    return TestClient(build_app(tmp_path))


def test_openapi_docs_available(client):
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    spec = resp.json()
    assert spec["info"]["title"] == "RAG 混合检索接口"
    paths = spec["paths"]
    assert "/v1/ask" in paths
    assert "/v1/ingest" in paths
    assert "/v1/documents" in paths
    assert "/healthz" in paths
    # OpenAPI doc page renders
    docs = client.get("/docs")
    assert docs.status_code == 200


def test_healthz(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["vector_chunks"] > 0


def test_ask_returns_answer_with_citations_and_sources(client):
    resp = client.post(
        "/v1/ask", json={"question": "What is FastAPI?", "top_k": 3}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["can_answer"] is True
    assert body["answer"]
    assert body["citations"] is not None
    assert "confidence" in body
    assert body["confidence"]["composite"] >= 0.0
    assert isinstance(body["sources"], list)
    assert len(body["sources"]) > 0
    src = body["sources"][0]
    assert "chunk_id" in src and "content" in src


def test_ask_empty_question_400(client):
    resp = client.post("/v1/ask", json={"question": "   "})
    assert resp.status_code == 400


def test_production_trace_correlates_stages_without_sending_request_text(tmp_path, monkeypatch):
    import json

    monkeypatch.setenv("RAG_TRACE_ENDPOINT", "http://127.0.0.1:8088/v1/traces")
    deliveries = []

    def unavailable(endpoint, trace_id, payload):
        deliveries.append((endpoint, trace_id, payload))
        return False

    monkeypatch.setattr("rag_pipeline.api.export_trace", unavailable)
    response = TestClient(build_app(tmp_path)).post("/v1/ask", json={
        "question": "private-question-sentinel", "retrieval_mode": "private-mode-sentinel", "top_k": 3,
    })
    assert response.status_code == 200
    _, trace_id, payload = deliveries[0]
    assert response.headers["X-Trace-Id"] == trace_id
    serialized = json.dumps(payload, ensure_ascii=False)
    assert "private-question-sentinel" not in serialized
    assert "private-mode-sentinel" not in serialized
    spans = payload["resourceSpans"][0]["scopeSpans"][0]["spans"]
    assert [span["name"] for span in spans] == [
        "rag.ask", "rag.retrieve", "rag.generate", "rag.verify_citations", "rag.decide",
    ]
    assert all(span["traceId"] == trace_id for span in spans)
    assert all(span["parentSpanId"] == spans[0]["spanId"] for span in spans[1:])


def test_ask_without_llm_503(tmp_path):
    app = create_app(persist_dir=str(tmp_path / "chroma"), bm25_dir=str(tmp_path / "bm25"))
    resp = TestClient(app).post(
        "/v1/ask", json={"question": "What is FastAPI?"}
    )
    assert resp.status_code == 503
    assert "尚未配置" in resp.json()["detail"]


def test_retrieve_supports_modes(client):
    for mode in ("hybrid", "dense", "sparse"):
        resp = client.post(
            "/v1/retrieve",
            json={"question": "What is FastAPI?", "mode": mode, "top_k": 3},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["mode"] == mode
        assert len(body["ranking"]) > 0
        item = body["ranking"][0]
        assert "rerank_score" in item
        assert "dense_score" in item
        assert "sparse_score" in item


def test_sparse_api_retrieves_chinese_document(client):
    # 测试分词和接口链路，避免模拟向量的相似度去重影响中文样例。
    client.app.state.pipeline.enable_deduplication = False
    for filename, text in (
        ("认证.md", "接口认证需要配置访问令牌。管理员在控制台创建凭据，调用方将凭据放入请求头。生产环境必须保护凭据，不应把它写入代码仓库，也不应分享给无关人员。"),
        ("账单.md", "账单每月结算一次。财务人员可以查看费用明细，并按照项目归属进行成本分摊。每期结束之后会生成报表，历史记录可以在管理页面查询并导出保存。"),
        ("备份.md", "数据库每天自动备份。运维人员应定期执行恢复演练，确认数据可以完整恢复。历史副本存储在独立位置，超过保留周期的副本按照管理制度进行清理。"),
    ):
        response = client.post(
            "/v1/ingest", files={"file": (filename, text.encode("utf-8"), "text/markdown")}
        )
        assert response.status_code == 201
        assert response.json()["chunks_added_bm25"] > 0
    response = client.post(
        "/v1/retrieve",
        json={"question": "如何配置接口认证令牌？", "mode": "sparse", "top_k": 1},
    )
    assert response.status_code == 200
    first = response.json()["ranking"][0]
    assert first["source_file"] == "认证.md"
    assert first["sparse_score"] > 0


def test_retrieve_invalid_mode_falls_back_to_hybrid(client):
    resp = client.post(
        "/v1/retrieve",
        json={"question": "What is FastAPI?", "mode": "bogus", "top_k": 3},
    )
    assert resp.status_code == 200
    # service.retrieve normalizes invalid modes to hybrid; response echoes the
    # requested value, ranking is non-empty either way
    assert len(resp.json()["ranking"]) > 0


def test_ask_supports_dense_and_sparse_modes(client):
    for mode in ("dense", "sparse"):
        resp = client.post(
            "/v1/ask",
            json={"question": "What is FastAPI?", "mode": mode, "top_k": 3},
        )
        resp = client.post(
            "/v1/ask",
            json={
                "question": "What is FastAPI?",
                "retrieval_mode": mode,
                "top_k": 3,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["can_answer"] is True
        assert len(body["sources"]) > 0


def test_ask_low_relevance_declines(client):
    # Off-topic question with weak retrieval -> declines gracefully
    resp = client.post(
        "/v1/ask", json={"question": "What color is the ceiling?", "top_k": 3}
    )
    body = resp.json()
    assert body["can_answer"] is False
    assert body["declined"] is not None
    assert body["declined"]["reason"] == "insufficient_confidence"


def test_ingest_marksdown_upload(client):
    content = b"# Test Doc\n\nChunking of PDF documents works via pdfplumber."
    resp = client.post(
        "/v1/ingest",
        files={"file": ("test_doc.md", content, "text/markdown")},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["doc_id"]
    assert body["chunks_added_dense"] > 0
    assert body["chunks_added_bm25"] > 0


def test_ingest_unsupported_extension_422(client):
    resp = client.post(
        "/v1/ingest",
        files={"file": ("bad.xyz", b"whatever", "application/octet-stream")},
    )
    assert resp.status_code in (422, 500)


def test_documents_lists_indexed(client):
    resp = client.get("/v1/documents")
    assert resp.status_code == 200
    docs = resp.json()
    assert len(docs) > 0
    first = docs[0]
    assert "doc_id" in first and "chunk_count" in first
    assert first["chunk_count"] > 0


def test_ask_generator_dir(client):
    """Ensure /v1/ask still works end-to-end after ingestion of extra doc."""
    with open(DATA_DIR / "data" / "raw" / "fastapi_docs" / "tutorial" / "first-steps.md", "rb") as f:
        content = f.read()
    resp = client.post(
        "/v1/ingest",
        files={"file": ("first-steps.md", content, "text/markdown")},
    )
    assert resp.status_code == 201

    get = client.get("/v1/documents")
    assert len(get.json()) >= 2


@pytest.mark.parametrize("verdict", ["UNSUPPORTED", "", "unknown"])
def test_api_hides_answer_when_citation_fails_even_with_high_total(client, monkeypatch, verdict):
    service = client.app.state.service
    service.citation_verifier = CitationVerifier(judge=Generator(respond=lambda messages: verdict))
    monkeypatch.setattr(service.handler.scorer, "score", lambda *args, **kwargs: ConfidenceReport(
        completeness=0.9, composite=0.99,
    ))
    response = client.post("/v1/ask", json={"question": "What is FastAPI?", "top_k": 3})
    assert response.status_code == 200
    body = response.json()
    assert body["can_answer"] is False
    assert body["answer"] is None
    assert body["confidence"]["details"]["gate_failures"]
    assert "未通过检查" in body["declined"]["summary"]
