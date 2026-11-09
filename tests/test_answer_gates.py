import pytest

from rag_pipeline.citation import CitationParser, CitationReport
from rag_pipeline.confidence import ConfidenceScorer, ConfidenceReport
from rag_pipeline.retrieval import RetrievalResult
from rag_pipeline.unknown import UnknownHandler


class FixedScorer(ConfidenceScorer):
    """固定分数，独立验证门禁不会被高总分绕过。"""

    def __init__(self, completeness=0.9, composite=0.99):
        super().__init__()
        self.completeness = completeness
        self.composite = composite

    def score(self, *args, **kwargs):
        return ConfidenceReport(
            retrieval_confidence=0.8, citation_coverage=1.0,
            completeness=self.completeness, composite=self.composite,
        )


def report_for(answer, supported=True, verified=True):
    citations = CitationParser().parse(answer)
    for citation in citations:
        citation.supported = supported
    return CitationReport(answer=answer, citations=citations, verified=verified)


def test_unsupported_citation_blocks_the_original_059_case():
    answer = "默认返回一千条 [1]。"
    source = RetrievalResult("a", "默认返回十条。", {})
    handler = UnknownHandler(FixedScorer(composite=0.59))
    outcome = handler.decide("每页多少条？", answer, [source], report_for(answer, False))
    assert outcome.confidence.composite == 0.59
    assert not outcome.can_answer
    assert "存在原文不支持的引用" in outcome.confidence.details["gate_failures"]


@pytest.mark.parametrize("answer, report_kind, sources_present, reason", [
    ("默认返回十条 [1]。", "unknown", True, "存在无法验证的引用"),
    ("默认返回十条 [1]。", "missing", True, "引用尚未完成校验"),
    ("默认返回十条 [1]。", "unverified", True, "引用尚未完成校验"),
    ("默认返回十条 [9]。", "valid", True, "引用编号超出本次来源范围"),
    ("默认返回十条。", "valid", True, "答案没有引用"),
    ("默认返回十条 [1]。退款不限时。", "valid", True, "存在未标注引用的句子"),
    ("默认返回十条 [1]。", "mismatch", True, "引用报告与当前答案不匹配"),
    ("默认返回十条 [1]。", "valid", False, "没有可供核验的来源片段"),
    ("", "valid", True, "答案为空"),
])
def test_high_total_cannot_override_missing_evidence(answer, report_kind, sources_present, reason):
    report = report_for(answer)
    if report_kind == "unknown":
        report.citations[0].supported = None
    elif report_kind == "missing":
        report = None
    elif report_kind == "unverified":
        report.verified = False
    elif report_kind == "mismatch":
        report = report_for("另外一份答案 [1]。")
    sources = [RetrievalResult("a", "默认返回十条。", {})] if sources_present else []
    outcome = UnknownHandler(FixedScorer()).decide("每页多少条？", answer, sources, report)
    assert not outcome.can_answer
    assert reason in outcome.confidence.details["gate_failures"]
    assert reason in outcome.low_confidence.summary


@pytest.mark.parametrize("completeness, composite, accepted", [
    (0.1, 0.99, False),
    (0.9, 0.49, False),
    (0.5, 0.5, True),
    (0.9, 0.9, True),
    (float("nan"), 0.99, False),
    (0.9, float("inf"), False),
])
def test_completeness_floor_and_overall_threshold(completeness, composite, accepted):
    answer = "默认返回十条 [1]。"
    sources = [RetrievalResult("a", "默认返回十条。", {})]
    handler = UnknownHandler(FixedScorer(completeness, composite))
    outcome = handler.decide("每页多少条？", answer, sources, report_for(answer))
    assert outcome.can_answer is accepted


@pytest.mark.parametrize("floor", [-0.1, 1.1, float("nan")])
def test_invalid_completeness_floor(floor):
    with pytest.raises(ValueError):
        UnknownHandler(FixedScorer(), min_completeness=floor)
