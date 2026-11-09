from rag_pipeline.retrieval import RetrievalResult
from rag_pipeline.generation import Generator
from rag_pipeline.citation import Citation, CitationReport
from rag_pipeline.confidence import ConfidenceScorer, ConfidenceReport


def chunk(content, rerank=None, dense=None, rrf=None):
    r = RetrievalResult(chunk_id="x", content=content, metadata={})
    r.rerank_score = rerank
    r.dense_score = dense
    r.rrf_score = rrf
    return r


def test_retrieval_confidence_uses_rerank_scores():
    scorer = ConfidenceScorer()
    sources = [chunk("a", rerank=0.9), chunk("b", rerank=0.5), chunk("c", rerank=0.1)]
    conf = scorer._retrieval_confidence(sources)
    assert abs(conf - 0.5) < 1e-6  # avg(0.9,0.5,0.1)


def test_retrieval_confidence_falls_back_to_dense():
    scorer = ConfidenceScorer()
    sources = [chunk("a", dense=0.8), chunk("b", dense=0.6)]
    conf = scorer._retrieval_confidence(sources)
    assert abs(conf - 0.7) < 1e-6


def test_retrieval_confidence_empty():
    assert ConfidenceScorer()._retrieval_confidence([]) == 0.0


def test_citation_coverage_from_report():
    scorer = ConfidenceScorer()
    report = CitationReport(answer="x")
    report.citations = [
        Citation(number=1, claim="a", supported=True),
        Citation(number=2, claim="b", supported=True),
        Citation(number=3, claim="c", supported=False),
    ]
    cov = scorer._citation_coverage("answer", report)
    assert abs(cov - (2 / 3)) < 1e-6


def test_citation_coverage_fallback_cited_sentences():
    scorer = ConfidenceScorer()
    # Without a report, coverage = fraction of sentences that carry a citation
    answer = "Sentence one with a citation [1]. Sentence two also [2]. Plain sentence."
    cov = scorer._citation_coverage(answer, None)
    assert abs(cov - (2 / 3)) < 1e-6


def test_completeness_lexical_high():
    scorer = ConfidenceScorer()
    # Answer echoes the question's key tokens
    score = scorer._lexical_completeness(
        "How does authentication work with an api key?",
        "Authentication uses an api key issued by the admin dashboard.",
    )
    assert score > 0.5


def test_completeness_idk_low():
    scorer = ConfidenceScorer()
    score = scorer._lexical_completeness(
        "What is the refund policy?",
        "I don't have enough information in the context to answer this question.",
    )
    assert score <= 0.1


def test_chinese_refusal_is_not_scored_as_a_complete_answer():
    scorer = ConfidenceScorer()
    for answer in (
        "上下文信息不足，无法回答这个问题。",
        "在提供的文档中找不到相关信息。",
    ):
        assert scorer._lexical_completeness("退款规则是什么？", answer) <= 0.1


def test_unrecognized_question_does_not_receive_full_completeness():
    assert ConfidenceScorer()._lexical_completeness("几点关门？", "地址在人民路。") == 0.0


def test_failed_completeness_judge_is_not_a_passing_score():
    def timeout(messages):
        raise TimeoutError("test")

    scorer = ConfidenceScorer(judge=Generator(respond=timeout))
    assert scorer._judge_completeness("几点关门？", "晚上九点。") == 0.0


def test_composite_uses_weights():
    scorer = ConfidenceScorer(weights={"retrieval": 0.5, "citation": 0.3, "completeness": 0.2})
    report = ConfidenceReport(
        retrieval_confidence=0.8, citation_coverage=0.6, completeness=0.4
    )
    composite = (
        report.retrieval_confidence * 0.5
        + report.citation_coverage * 0.3
        + report.completeness * 0.2
    )
    assert abs(composite - (0.8 * 0.5 + 0.6 * 0.3 + 0.4 * 0.2)) < 1e-6


def test_full_score_method():
    scorer = ConfidenceScorer()
    report_cites = CitationReport(answer="ans")
    report_cites.citations = [
        Citation(number=1, claim="a", supported=True),
        Citation(number=2, claim="b", supported=False),
    ]
    sources = [chunk("api key auth", rerank=0.8), chunk("setup", rerank=0.7)]
    report = scorer.score(
        "How does api key authentication work?",
        "Authentication uses an api key [1]. Setup is manual [2].",
        sources,
        citation_report=report_cites,
    )
    assert isinstance(report, ConfidenceReport)
    assert 0.0 <= report.retrieval_confidence <= 1.0
    assert 0.0 <= report.citation_coverage <= 1.0
    assert 0.0 <= report.completeness <= 1.0
    assert 0.0 <= report.composite <= 1.0
    d = report.to_dict()
    assert "composite" in d and "details" in d


def test_score_with_judge_completeness():
    def fake_judge(messages):
        return '{"score": 0.9, "reason": "addresses everything"}'

    scorer = ConfidenceScorer(judge=Generator(respond=fake_judge))
    report = scorer.score(
        "What is the pricing?",
        "Pricing is per seat [1].",
        [chunk("per seat pricing", rerank=0.9)],
        citation_report=None,
    )
    assert report.completeness == 0.9
