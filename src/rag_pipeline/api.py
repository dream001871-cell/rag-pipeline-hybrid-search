"""FastAPI service exposing the RAG pipeline over HTTP.

Endpoints:
    POST /v1/ask       - answer a question with citations + confidence
    POST /v1/retrieve  - ranked chunks for a retrieval mode (no LLM needed)
    POST /v1/ingest    - ingest a new document (file upload)
    GET  /v1/documents - list indexed documents

OpenAPI docs are auto-generated at /docs and /openapi.json.

The service is fully runnable offline: if no chat API key (OPENROUTER_API_KEY
or OPENAI_API_KEY) is present it uses MockEmbeddingService and a pluggable
generator (see create_app).
"""

from dotenv import load_dotenv
load_dotenv()  # loads .env from project root silently

import io
import os
import uuid
from dataclasses import asdict
from contextlib import nullcontext
from typing import List, Optional
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Response, UploadFile
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel

from .ingestion import IngestionPipeline
from .document_loader import DocumentLoader
from .chunker import ChunkingConfig
from .embedding import MockEmbeddingService, create_embedding_service
from .vector_store import VectorStore
from .bm25_index import BM25Index
from .retrieval import DenseRetriever, SparseRetriever, RetrievalResult
from .fusion import HybridRetriever, DEFAULT_TOP_K
from .reranker import CrossEncoderReranker, LexicalReranker, create_reranker
from .generation import Generator, select_context
from .citation import CitationVerifier
from .confidence import ConfidenceScorer
from .unknown import UnknownHandler, AnswerOutcome
from .tracing import AnswerTrace, export_trace

DEFAULT_PERSIST_DIR = os.environ.get("RAG_PERSIST_DIR", "data/chroma")
DEFAULT_BM25_DIR = os.environ.get("RAG_BM25_DIR", "data/bm25")
DEFAULT_THRESHOLD = float(os.environ.get("RAG_CONFIDENCE_THRESHOLD", "0.5"))


def _chroma_endpoint_from_env() -> Optional[tuple]:
    """Return (host, port) for a standalone ChromaDB server, if configured.

    Reads CHROMA_HOST/CHROMA_PORT. When unset the API uses its embedded
    persistent store (data/chroma); when set it talks to the composed
    ChromaDB service over HTTP.
    """
    host = os.environ.get("CHROMA_HOST")
    if not host:
        return None
    try:
        port = int(os.environ.get("CHROMA_PORT", "8000"))
    except ValueError:
        port = 8000
    return host, port


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str
    top_k: int = DEFAULT_TOP_K
    retrieval_mode: str = "hybrid"  # hybrid | dense | sparse
    threshold: Optional[float] = None  # override service default per request


class AskResponse(BaseModel):
    can_answer: bool
    question: str
    answer: Optional[str] = None
    confidence: Optional[dict] = None
    citations: Optional[dict] = None
    sources: Optional[list] = None
    declined: Optional[dict] = None


class RetrieveRequest(BaseModel):
    question: str
    mode: str = "hybrid"  # hybrid | dense | sparse
    top_k: int = DEFAULT_TOP_K


class RetrieveResponse(BaseModel):
    question: str
    mode: str
    ranking: list = []


class IngestResponse(BaseModel):
    doc_id: str
    chunks_added_dense: int
    chunks_added_bm25: int
    duplicates: int
    source_file: str


class DocumentInfo(BaseModel):
    doc_id: str
    source_file: Optional[str] = None
    chunk_count: int
    strategies: List[str] = []


class AskService:
    """Wires the full retrieval -> generation -> verification -> gate flow."""

    RETRIEVAL_MODES = ("hybrid", "dense", "sparse")

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        generator: Generator,
        citation_verifier: Optional[CitationVerifier] = None,
        scorer: Optional[ConfidenceScorer] = None,
        handler: Optional[UnknownHandler] = None,
    ):
        self.hybrid_retriever = hybrid_retriever
        self.generator = generator
        self.citation_verifier = citation_verifier
        self.scorer = scorer or ConfidenceScorer()
        self.handler = handler or UnknownHandler(self.scorer)

    def retrieve(self, question: str, top_k: int = DEFAULT_TOP_K, retrieval_mode: str = "hybrid") -> List[RetrievalResult]:
        """Retrieve + rerank for a retrieval mode (hybrid/dense/sparse).

        Hybrid goes through the fused + reranked path. Dense/sparse rerank the
        single-list candidates with the same reranker so rankings are
        comparable side by side.
        """
        retriever = self.hybrid_retriever
        mode = retrieval_mode if retrieval_mode in self.RETRIEVAL_MODES else "hybrid"

        if mode == "dense":
            results = retriever.retrieve_dense(question, top_k=max(top_k, retriever.dense_k))
        elif mode == "sparse":
            results = retriever.retrieve_sparse(question, top_k=max(top_k, retriever.sparse_k))
        else:
            return retriever.retrieve(question, top_k=top_k)[:top_k]

        if retriever.reranker and results:
            results = retriever.reranker.score(question, results)
        return results[:top_k]

    def answer(self, question: str, top_k: int = DEFAULT_TOP_K, retrieval_mode: str = "hybrid",
               trace: Optional[AnswerTrace] = None) -> dict:
        # 先固定本次证据集合，保证生成、引用编号、校验与返回来源完全一致。
        with trace.step("rag.retrieve") if trace else nullcontext({}) as event:
            results = select_context(self.retrieve(question, top_k=top_k, retrieval_mode=retrieval_mode))
            mode = retrieval_mode if retrieval_mode in self.RETRIEVAL_MODES else "hybrid"
            event.update({"rag.mode": mode, "rag.top_k": top_k,
                          "rag.context_count": len(results)})

        with trace.step("rag.generate") if trace else nullcontext({}) as event:
            gen_out = self.generator.generate(question, results)
            answer_text = gen_out.get("answer", "")
            claims = gen_out.get("claims")
            generation_error = gen_out.get("generation_error")
            event.update({"gen_ai.request.model": gen_out.get("model", ""),
                          "rag.answer_chars": len(answer_text or "")})
            if claims is not None:
                event["rag.claim_count"] = len(claims)
            if generation_error:
                event["rag.generation_error"] = generation_error

        citation_report = None
        with trace.step("rag.verify_citations") if trace else nullcontext({}) as event:
            if self.citation_verifier and answer_text.strip():
                if claims is None:
                    citation_report = self.citation_verifier.verify(answer_text, results)
                else:
                    citation_report = self.citation_verifier.verify_claims(answer_text, claims, results)
                event.update({"rag.citations": citation_report.total,
                              "rag.citations_supported": citation_report.supported_count})
        with trace.step("rag.decide") if trace else nullcontext({}) as event:
            if claims is None:
                outcome = self.handler.decide(question, answer_text, results, citation_report=citation_report)
            else:
                outcome = self.handler.decide(
                    question, answer_text, results, citation_report=citation_report,
                    claims=claims, generation_error=generation_error,
                )
            event["rag.can_answer"] = outcome.can_answer
            if outcome.confidence is not None:
                event["rag.gate_failures"] = outcome.confidence.details.get("gate_failures", [])
                event["rag.confidence"] = outcome.confidence.composite

        reply = self._serialize(outcome, question, results)
        if outcome.can_answer and gen_out.get("partial"):
            reply["answer"] += "\n现有资料仅能支持以上内容。"
        return reply

    def _serialize(self, outcome: AnswerOutcome, question: str, results: List[RetrievalResult]) -> dict:
        sources = [self._source(r) for r in results]

        if outcome.can_answer:
            return {
                "can_answer": True,
                "question": question,
                "answer": outcome.answer,
                "confidence": outcome.confidence.to_dict() if outcome.confidence else None,
                "citations": outcome.citation_report.to_dict()
                if outcome.citation_report
                else None,
                "sources": sources,
                "declined": None,
            }
        return {
            "can_answer": False,
            "question": question,
            "answer": None,
            "confidence": outcome.confidence.to_dict() if outcome.confidence else None,
            "citations": None,
            "sources": sources,
            "declined": outcome.low_confidence.to_dict()
            if outcome.low_confidence
            else None,
        }

    def _source(self, r: RetrievalResult) -> dict:
        meta = r.metadata or {}
        return {
            "chunk_id": r.chunk_id,
            "source_file": meta.get("source_file"),
            "section": meta.get("section_heading"),
            "chunking_strategy": meta.get("chunking_strategy"),
            "rerank_score": r.rerank_score,
            "rrf_score": r.rrf_score,
            "dense_score": r.dense_score,
            "sparse_score": r.sparse_score,
            "content": r.content[:500],
        }


def create_app(
    pipeline: Optional[IngestionPipeline] = None,
    retriever: Optional[HybridRetriever] = None,
    generator: Optional[Generator] = None,
    citation_verifier: Optional[CitationVerifier] = None,
    handler: Optional[UnknownHandler] = None,
    threshold: float = DEFAULT_THRESHOLD,
    persist_dir: str = DEFAULT_PERSIST_DIR,
    bm25_dir: str = DEFAULT_BM25_DIR,
    chroma_host: Optional[str] = None,
    chroma_port: int = 8000,
) -> FastAPI:
    """Construct the FastAPI app.

    All components can be injected for tests. If none are given, a self-
    contained pipeline is built against persistent stores under data/ (or
    against a standalone ChromaDB server when CHROMA_HOST is set).
    """
    if pipeline is None:
        if chroma_host is None:
            chroma_host, chroma_port = _chroma_endpoint_from_env() or (None, 8000)
        vector_store = (
            VectorStore(host=chroma_host, port=chroma_port)
            if chroma_host
            else VectorStore(persist_dir=persist_dir)
        )
        pipeline = IngestionPipeline(
            chunking_config=ChunkingConfig(),
            embedding_service=create_embedding_service(),
            vector_store=vector_store,
            bm25_index=BM25Index(persist_dir=bm25_dir),
            document_loader=DocumentLoader(
                raw_dir=str(Path(persist_dir).parent / "raw"),
                processed_dir=str(Path(persist_dir).parent / "processed"),
            ),
        )

    if retriever is None:
        retriever = HybridRetriever(
            dense_retriever=DenseRetriever(
                embedding_service=pipeline.embedding_service,
                vector_store=pipeline.vector_store,
            ),
            sparse_retriever=SparseRetriever(bm25_index=pipeline.bm25_index),
            reranker=create_reranker(),
        )

    if generator is None:
        generator = Generator(structured_answers=True)

    if citation_verifier is None:
        citation_verifier = CitationVerifier(judge=generator)

    ranked_confidence = (
        os.environ.get("RAG_RANKED_RETRIEVAL_CONFIDENCE", "").lower() == "true"
        and isinstance(retriever.reranker, CrossEncoderReranker)
        and retriever.reranker.available
    )
    scorer = ConfidenceScorer(judge=generator, ranked_cross_encoder=ranked_confidence)
    if handler is None:
        handler = UnknownHandler(
            scorer, threshold=threshold,
            min_completeness=float(os.environ.get("RAG_MIN_COMPLETENESS", "0.5")),
        )

    service = AskService(
        hybrid_retriever=retriever,
        generator=generator,
        citation_verifier=citation_verifier,
        scorer=scorer,
        handler=handler,
    )

    app = FastAPI(
        title="RAG 混合检索接口",
        description=(
            "文档检索与问答服务，提供向量检索、BM25 关键词检索、RRF 融合、重排、"
            "带引用回答、引用校验及低置信度拒答。未配置模型密钥时可使用纯检索接口。"
        ),
        version="1.0.0",
    )

    app.state.pipeline = pipeline
    app.state.retriever = retriever
    app.state.service = service
    trace_endpoint = os.environ.get("RAG_TRACE_ENDPOINT", "").strip()

    @app.get("/healthz", tags=["系统"], summary="检查服务及双索引状态")
    def healthz():
        return {
            "status": "ok",
            "vector_chunks": pipeline.vector_store.count(),
            "bm25_chunks": pipeline.bm25_index.count(),
        }

    @app.get("/v1/documents", response_model=List[DocumentInfo], tags=["文档入库"], summary="查看已索引的文档")
    def list_documents():
        docs = pipeline.vector_store.list_documents()
        return [
            DocumentInfo(
                doc_id=d["doc_id"],
                source_file=d.get("source_file"),
                chunk_count=d["chunk_count"],
                strategies=d.get("strategies") or [],
            )
            for d in docs
        ]

    @app.post("/v1/reset", tags=["文档入库"], status_code=200, summary="清空索引和上传缓存")
    def reset_index():
        """删除全部向量与关键词索引片段，并清理上传缓存。"""
        pipeline.vector_store.reset()
        pipeline.bm25_index.reset()
        for d in (pipeline.document_loader.raw_dir, pipeline.document_loader.processed_dir):
            if d.is_dir():
                for f in d.glob("*"):
                    if f.is_file():
                        f.unlink()
        return {
            "status": "reset",
            "vector_chunks": pipeline.vector_store.count(),
            "bm25_chunks": pipeline.bm25_index.count(),
        }

    @app.post(
        "/v1/ingest",
        response_model=IngestResponse,
        tags=["文档入库"],
        summary="上传文档并建立双索引",
        status_code=201,
    )
    async def ingest_document(file: UploadFile = File(...)):
        if not file.filename:
            raise HTTPException(status_code=400, detail="请提供文件名")
        try:
            content = await file.read()
            result = pipeline.ingest_upload(file.filename, content)
        except ValueError as exc:  # unsupported extension
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"文档入库失败：{exc}")
        return IngestResponse(
            doc_id=result.get("doc_id", ""),
            chunks_added_dense=result.get("chunks_added_dense", 0),
            chunks_added_bm25=result.get("chunks_bm25", 0),
            duplicates=result.get("chunks_duplicated", 0),
            source_file=result.get("source_file", file.filename),
        )

    @app.post(
        "/v1/ask",
        response_model=AskResponse,
        tags=["检索与问答"],
        summary="生成带引用的回答（需要模型密钥）",
    )
    async def ask(req: AskRequest, response: Response, background_tasks: BackgroundTasks):
        if not req.question.strip():
            raise HTTPException(status_code=400, detail="问题不能为空")
        trace = AnswerTrace("rag-api-production", {"environment": "production"}) if trace_endpoint else None
        try:
            data = service.answer(
                req.question, top_k=req.top_k, retrieval_mode=req.retrieval_mode, trace=trace
            )
        except RuntimeError as exc:  # e.g. Generator not configured
            if trace is None:
                raise HTTPException(status_code=503, detail=str(exc))
            background_tasks.add_task(export_trace, trace_endpoint, trace.trace_id,
                                      trace.finish({"error.type": type(exc).__name__}))
            return JSONResponse(status_code=503, content={"detail": str(exc)}, background=background_tasks,
                                headers={"X-Trace-Id": trace.trace_id})
        except Exception as exc:
            if trace is not None:
                await run_in_threadpool(export_trace, trace_endpoint, trace.trace_id,
                                        trace.finish({"error.type": type(exc).__name__}))
            raise
        if trace is not None:
            response.headers["X-Trace-Id"] = trace.trace_id
            background_tasks.add_task(export_trace, trace_endpoint, trace.trace_id,
                                      trace.finish({"rag.can_answer": data["can_answer"]}))
        return AskResponse(**data)

    @app.post(
        "/v1/retrieve",
        response_model=RetrieveResponse,
        tags=["检索与问答"],
        summary="检索相关片段（无需模型密钥）",
    )
    async def retrieve(req: RetrieveRequest):
        if not req.question.strip():
            raise HTTPException(status_code=400, detail="问题不能为空")
        results = service.retrieve(req.question, top_k=req.top_k, retrieval_mode=req.mode)
        return RetrieveResponse(
            question=req.question,
            mode=req.mode,
            ranking=[service._source(r) for r in results],
        )

    return app


def run():
    """Entry point: uvicorn rag_pipeline.api:app"""
    import uvicorn

    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
