from pathlib import Path
from types import SimpleNamespace

from rag_evaluation.eval_metrics import EvalRunner
from rag_pipeline.api import AskService, AskRequest, RetrieveRequest
from rag_pipeline.citation import CitationVerifier
from rag_pipeline.fusion import HybridRetriever
from rag_pipeline.generation import Generator
from rag_pipeline.retrieval import RetrievalResult
from rag_pipeline.unknown import AnswerOutcome


def test_default_retrieval_recovers_evidence_below_the_old_candidate_limit():
    candidates = [RetrievalResult(str(index), f"资料{index}", {}) for index in range(60)]
    seen = []

    def retrieve(question, top_k):
        seen.append(top_k)
        return candidates[:top_k]

    def rerank(question, results):
        # 关键证据在初次排名第31位；只有扩大候选后才有机会被重排到第一。
        return sorted(results, key=lambda result: result.chunk_id != "30")

    retriever = HybridRetriever(
        dense_retriever=SimpleNamespace(retrieve=retrieve),
        sparse_retriever=SimpleNamespace(retrieve=retrieve),
        reranker=SimpleNamespace(score=rerank),
    )
    results = retriever.retrieve("关键证据")
    assert seen == [60, 60]
    assert len(results) == 20
    assert results[0].chunk_id == "30"


def test_api_uses_the_same_budgeted_evidence_for_generation_verification_and_sources():
    chunks = [RetrievalResult("first", "甲" * 6000, {}),
              RetrievalResult("excluded", "乙" * 7000, {}),
              RetrievalResult("last", "丙" * 6000, {})]
    generation_inputs = []
    verification_inputs = []
    decision_inputs = []

    def generate(messages):
        generation_inputs.append(messages)
        return "第三段支持该结论 [2]。"

    def verify(messages):
        verification_inputs.append(messages)
        return "SUPPORTED"

    def decide(question, answer, results, citation_report):
        decision_inputs.append(results)
        return AnswerOutcome(can_answer=True, answer=answer, citation_report=citation_report)

    service = AskService(
        hybrid_retriever=SimpleNamespace(retrieve=lambda question, top_k: chunks),
        generator=Generator(respond=generate),
        citation_verifier=CitationVerifier(judge=Generator(respond=verify)),
        handler=SimpleNamespace(decide=decide),
    )
    assert AskRequest(question="问题").top_k == RetrieveRequest(question="问题").top_k == 20
    response = service.answer("问题")
    assert [source["chunk_id"] for source in response["sources"]] == ["first", "last"]
    assert [result.chunk_id for result in decision_inputs[0]] == ["first", "last"]
    assert "乙" not in generation_inputs[0][1]["content"]
    assert 'id="2"' in generation_inputs[0][1]["content"]
    assert "丙" in verification_inputs[0][1]["content"]
    assert "乙" not in verification_inputs[0][1]["content"]
    assert response["can_answer"]


def test_evaluation_builder_uses_production_retrieval_and_generation():
    chunks = [RetrievalResult("first", "甲" * 6000, {}),
              RetrievalResult("excluded", "乙" * 7000, {}),
              RetrievalResult("last", "丙" * 6000, {})]
    requested = []

    def retrieve(question, top_k):
        requested.append(top_k)
        return chunks

    generator = Generator(respond=lambda messages: "已找到资料 [2]。")
    runner = EvalRunner.build(
        dataset_path=str(Path(__file__).resolve().parents[1] / "data" / "golden_dataset.json"),
        ingest_fn=lambda: None,
        hybrid_retriever=SimpleNamespace(retrieve=retrieve),
        judge=generator,
    )
    selected = runner.retrieve_fn("问题")
    generated = runner.generate_fn("问题", selected)
    assert requested == [20]
    assert [result.chunk_id for result in selected] == ["first", "last"]
    assert generated["source_chunks"] == selected
    assert generated["answer"] == "已找到资料 [2]。"
