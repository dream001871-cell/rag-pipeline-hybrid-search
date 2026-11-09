"""为 CRUD-RAG 小样本建立独立知识库，并提供评测平台可调用的接口。"""

import json
import argparse
import hashlib
import os
from pathlib import Path
import re
import sys
import threading
import time

from dotenv import load_dotenv


PROJECT_DIR = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_DIR)
sys.path.insert(0, str(PROJECT_DIR / "src"))
load_dotenv(PROJECT_DIR / ".env", override=True)
os.environ.setdefault("HF_HOME", str(PROJECT_DIR / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from fastapi import HTTPException
from pydantic import BaseModel

from rag_pipeline.api import create_app
from rag_pipeline.generation import Generator
from rag_pipeline.reranker import CrossEncoderReranker
from rag_pipeline.tracing import AnswerTrace, export_trace


class AuditedGenerator(Generator):
    """沿用原有生成和检查逻辑，仅保存真实调用结果及实际上下文。"""

    def __init__(self):
        super().__init__()
        self.events = []

    def generate(self, question, results, **kwargs):
        started = time.perf_counter()
        result = super().generate(question, results, **kwargs)
        event = {
            "kind": "judge" if kwargs.get("system_prompt") else "generation",
            "model": result["model"],
            "duration_seconds": time.perf_counter() - started,
            "answer": result["answer"],
        }
        if kwargs.get("system_prompt"):
            event["judge_task"] = "completeness" if "completeness judge" in kwargs["system_prompt"] else "citation"
        if event["kind"] == "generation":
            event["context"] = []
            for chunk in result["source_chunks"]:
                event["context"].append({
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "metadata": chunk.metadata,
                    "rerank_score": chunk.rerank_score,
                })
        self.events.append(event)
        return result


class EvalRequest(BaseModel):
    input: str | dict
    task_id: str | None = None


def create_eval_app(variant="first_run", reranker_model=None, batch=None):
    benchmarks = PROJECT_DIR / "data" / "benchmarks"
    dataset_dir = benchmarks / "crud_rag_20"
    question_path = dataset_dir / "questions.json"
    corpus = dataset_dir / "corpus"
    document_count = 34
    trace_endpoint = ""
    index_dir = PROJECT_DIR / ".local" / "crud-rag-evaluation" / "data"
    runtime_dir = PROJECT_DIR / ".local" / "crud-rag-evaluation"
    if variant == "holdout":
        if not batch or not re.fullmatch(r"[a-z0-9-]{1,64}", batch):
            raise ValueError("独立评测需要有效的批次标识。")
        config = json.loads((benchmarks / "current_dataset.json").read_text(encoding="utf-8"))
        question_path = benchmarks / config["questions"]
        corpus = benchmarks / config["corpus"]
        document_count = config["document_count"]
        fingerprint = hashlib.sha256()
        for document in sorted(corpus.glob("*.md")):
            fingerprint.update(document.name.encode("utf-8"))
            fingerprint.update(document.read_bytes())
        index_dir = PROJECT_DIR / ".local" / "crud-rag-holdout-indexes" / fingerprint.hexdigest()
        runtime_dir = PROJECT_DIR / ".local" / "crud-rag-holdout-runs" / batch
        trace_endpoint = os.environ.get("RAG_TRACE_ENDPOINT", "http://127.0.0.1:8088/v1/traces").strip()
        if not trace_endpoint:
            raise RuntimeError("独立评测需要配置执行追踪接收地址。")
    elif variant != "first_run":
        runtime_dir = PROJECT_DIR / ".local" / "crud-rag-improvement" / variant
    output_dir = runtime_dir / "responses"
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = json.loads(question_path.read_text(encoding="utf-8"))
    question_ids = {}
    for question in dataset["questions"]:
        question_ids[question["question"]] = question["id"]
    if variant == "holdout" and len(question_ids) != config["question_count"]:
        raise RuntimeError("当前测试集题数不匹配或包含重复题目。")
    documents = sorted(corpus.glob("*.md"))
    if len(documents) != document_count:
        raise RuntimeError(f"测试语料数量与选题记录不一致，应为 {document_count} 篇。")

    generator = AuditedGenerator()
    if not generator.is_configured:
        raise RuntimeError("尚未配置生成模型，无法执行真实模型评测。")

    # 显式使用独立索引，防止测试语料混入现有 FastAPI 文档知识库。
    if os.environ.get("CHROMA_HOST", "").strip():
        raise RuntimeError("此测试服务要求本地独立 Chroma 索引，请先检查 CHROMA_HOST 配置。")
    app = create_app(
        generator=generator,
        persist_dir=str(index_dir / "chroma"),
        bm25_dir=str(index_dir / "bm25"),
    )
    pipeline = app.state.pipeline
    ingestion = []
    for document in documents:
        ingestion.append(pipeline.ingest_file(str(document)))
    sync = pipeline.sync_status()
    if not sync["in_sync"] or not sync["dense_count"]:
        raise RuntimeError("测试知识库索引为空或双索引数量不一致。")

    retriever = app.state.retriever
    if reranker_model:
        reranker = CrossEncoderReranker(model_name=reranker_model)
        if not reranker.available:
            raise RuntimeError("指定的中文重排模型未加载成功，停止该轮以免静默退回其他方法。")
        retriever.reranker = reranker
    if variant == "holdout" and not getattr(retriever.reranker, "available", False):
        raise RuntimeError("生产重排模型未加载，停止评测以避免使用降级排序。")
    settings = {
        "variant": variant,
        "model": generator.model,
        "provider": generator._client.provider,
        "embedding_model": getattr(pipeline.embedding_service, "model_name", None),
        "embedding_service": type(pipeline.embedding_service).__name__,
        "reranker": type(retriever.reranker).__name__,
        "reranker_available": getattr(retriever.reranker, "available", None),
        "reranker_model": getattr(retriever.reranker, "model_name", None),
        "chunk_size": pipeline.config.chunk_size,
        "chunk_overlap": pipeline.config.chunk_overlap,
        "dense_k": retriever.dense_k,
        "sparse_k": retriever.sparse_k,
        "candidate_k": retriever.candidate_k,
        "top_k": 20,
        "source_document_count": len(ingestion),
        "indexed_document_count": len(pipeline.vector_store.list_documents()),
        "index_status": sync,
        "ingestion": ingestion,
    }
    (runtime_dir / "settings.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    request_lock = threading.Lock()

    @app.get("/eval/status")
    def eval_status():
        return {"ready": True, "variant": variant, "model": generator.model, "questions": len(question_ids),
                "source_documents": len(ingestion), "indexes": sync, "batch": batch}

    @app.post("/eval")
    def evaluate(request: EvalRequest):
        question = request.input
        if isinstance(question, dict):
            question = question.get("text")
        if not isinstance(question, str) or question not in question_ids:
            raise HTTPException(status_code=422, detail="请求须包含本测试集中的原始题目。")
        record_id = question_ids[question]
        if request.task_id is not None and request.task_id != record_id:
            raise HTTPException(status_code=422, detail="平台题号与题目不匹配。")

        # 串行执行以防调用记录混入其他题目；不修改原系统的检索和拒答规则。
        with request_lock:
            # 平台重试时复用同一题的成功响应，避免重复调用模型或生成另一条调用链。
            cached_path = output_dir / f"{record_id}.json"
            if trace_endpoint and cached_path.is_file():
                cached = json.loads(cached_path.read_text(encoding="utf-8"))
                return cached["platform_response"]
            generator.events = []
            started = time.perf_counter()
            record = {"id": record_id, "question": question, "started_unix": time.time()}
            trace = None
            if trace_endpoint:
                trace = AnswerTrace("rag-benchmark", {"environment": "evaluation", "batch": batch,
                                                       "case_id": record_id, "dataset": "rag基线"})
            try:
                response = app.state.service.answer(question, top_k=20, retrieval_mode="hybrid", trace=trace)
                output = response.get("answer")
                if not output:
                    output = (response.get("declined") or {}).get("summary") or "证据不足，暂时无法回答。"
                record["response"] = response
                record["output"] = output
            except Exception as exc:
                record["error_type"] = type(exc).__name__
                raise HTTPException(status_code=502, detail="RAG 调用失败：" + type(exc).__name__) from exc
            finally:
                record["duration_seconds"] = time.perf_counter() - started
                record["model_calls"] = generator.events
                output_path = output_dir / f"{record_id}_{time.time_ns()}.json"
                if trace:
                    root_attributes = {"input.value": question, "output.value": record.get("output", ""),
                                       "evalsi.metadata": {"batch": batch, "case_id": record_id}}
                    if record.get("error_type"):
                        root_attributes["error.type"] = record["error_type"]
                    else:
                        root_attributes["rag.can_answer"] = response["can_answer"]
                    record["trace_id"] = trace.trace_id
                    record["trace_payload"] = trace.finish(root_attributes)
                    record["trace_exported"] = export_trace(trace_endpoint, trace.trace_id, record["trace_payload"])
                    if not record.get("error_type"):
                        record["platform_response"] = {"output": output, "trace_id": trace.trace_id,
                                                       "trajectory": trace.trajectory(),
                                                       "usage": {"latency_ms": record["duration_seconds"] * 1000}}
                        output_path = cached_path
                pending = output_path.with_suffix(".tmp")
                pending.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
                pending.replace(output_path)
                print(json.dumps({"id": record_id, "can_answer": record.get("response", {}).get("can_answer"),
                                  "seconds": round(record["duration_seconds"], 2),
                                  "model_calls": len(generator.events), "error": record.get("error_type")},
                                 ensure_ascii=False), flush=True)
            return record.get("platform_response", {"output": output, "usage": {"latency_ms": record["duration_seconds"] * 1000}})

    return app


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="运行隔离的 CRUD-RAG 对照评测服务。")
    parser.add_argument("--variant", choices=["first_run", "judge_fix", "zh_rerank", "holdout"], default="first_run")
    parser.add_argument("--batch", help="当前独立测试的批次标识")
    parser.add_argument("--reranker-model", help="本轮单独替换的重排模型；不改变向量和索引")
    args = parser.parse_args()
    uvicorn.run(create_eval_app(args.variant, args.reranker_model, args.batch), host="127.0.0.1", port=8001)
