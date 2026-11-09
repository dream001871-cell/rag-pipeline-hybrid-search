import pytest

from rag_pipeline.retrieval import RetrievalResult
from rag_pipeline.confidence import ConfidenceScorer
from rag_pipeline.unknown import UnknownHandler, LowConfidenceAnswer
from rag_pipeline.citation import CitationReport, CitationParser


def verified_report(answer):
    citations = CitationParser().parse(answer)
    for citation in citations:
        citation.supported = True
    return CitationReport(answer=answer, citations=citations, verified=True)


def chunk(content, rerank=None, dense=None, meta=None):
    r = RetrievalResult(
        chunk_id="x", content=content,
        metadata=meta or {"doc_id": "d1", "source_file": "guide.md"},
    )
    r.rerank_score = rerank
    r.dense_score = dense
    return r


def test_high_confidence_returns_answer():
    scorer = ConfidenceScorer()
    handler = UnknownHandler(scorer, threshold=0.5)
    sources = [chunk("api key authentication is required for access", rerank=0.95)]
    outcome = handler.decide(
        "What authentication is required?",
        "An api key is required [1].",
        sources,
        citation_report=verified_report("An api key is required [1]."),
    )
    assert outcome.can_answer is True
    assert outcome.answer == "An api key is required [1]."
    assert outcome.low_confidence is None
    d = outcome.to_dict()
    assert d["can_answer"] is True and d["answer"]


def test_low_confidence_declines_gracefully():
    scorer = ConfidenceScorer()
    handler = UnknownHandler(scorer, threshold=0.8)  # high bar
    # Weak retrieval => low composite confidence
    sources = [chunk("unrelated content about pricing and seats", rerank=0.1)]
    outcome = handler.decide(
        "What is the exact SLA refund percentage?",
        "The refund is 50% [1].",
        sources,
    )
    assert outcome.can_answer is False
    assert isinstance(outcome.low_confidence, LowConfidenceAnswer)
    lc = outcome.low_confidence
    assert isinstance(lc.summary, str) and lc.summary
    d = outcome.to_dict()
    assert d["can_answer"] is False
    assert d["reason"] == "insufficient_confidence"
    assert d["threshold"] == 0.8


def test_low_confidence_reports_found_and_gaps():
    scorer = ConfidenceScorer()
    handler = UnknownHandler(scorer, threshold=0.9)
    sources = [
        chunk("authentication requires an api key with admin scope", rerank=0.4),
        chunk("billing is handled through the dashboard", rerank=0.2),
    ]
    outcome = handler.decide(
        "What is the refund policy after cancellation?",
        "The refund policy is generous [1].",
        sources,
    )
    lc = outcome.low_confidence
    # found reflects top retrieved chunks
    assert len(lc.found) > 0
    # 'cancellation' is a question term absent from answer and found -> a gap
    assert any("cancellation" in g for g in lc.gaps)
    # candidate documents surfaced
    assert len(lc.candidate_documents) > 0
    first = lc.candidate_documents[0]
    assert "doc_id" in first and "title" in first and "score" in first


def test_candidate_documents_deduped():
    scorer = ConfidenceScorer()
    handler = UnknownHandler(scorer, threshold=0.9, candidate_docs=2)
    sources = [
        chunk("content a", rerank=0.3, meta={"doc_id": "guide.md", "source_file": "Guide.pdf"}),
    ]
    outcome = handler.decide("q", "a [1].", sources)
    docs = outcome.low_confidence.candidate_documents
    # dedupe by doc_id
    ids = [d["doc_id"] for d in docs]
    assert len(ids) == len(set(ids))
    assert docs[0]["title"] == "Guide.pdf"


def test_threshold_validation():
    with pytest.raises(ValueError):
        UnknownHandler(ConfidenceScorer(), threshold=1.5)
    with pytest.raises(ValueError):
        UnknownHandler(ConfidenceScorer(), threshold=-0.1)


def test_to_dict_round_trip_answer_direct():
    handler = UnknownHandler(ConfidenceScorer(), threshold=0.0, min_completeness=0.0)
    sources = [chunk("content", rerank=0.99)]
    outcome = handler.decide("q", "answer [1].", sources, citation_report=verified_report("answer [1]."))
    d = outcome.to_dict()
    assert d["can_answer"] is True
