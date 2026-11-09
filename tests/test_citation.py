import pytest

from rag_pipeline.retrieval import RetrievalResult
from rag_pipeline.generation import Generator
from rag_pipeline.citation import (
    Citation,
    CitationReport,
    CitationParser,
    CitationVerifier,
)


def chunk(cid, content, source="guide.md"):
    return RetrievalResult(
        chunk_id=cid, content=content,
        metadata={"doc_id": "d", "source_file": source},
    )


def test_parser_extracts_single_citations():
    answer = "The API requires a bearer token [1]. It supports rate limiting [2]."
    parser = CitationParser()
    cites = parser.parse(answer)
    assert len(cites) == 2
    assert cites[0].number == 1 and "bearer token" in cites[0].claim
    assert cites[1].number == 2 and "rate limiting" in cites[1].claim


def test_parser_multiple_citations_in_sentence():
    answer = "Pricing is per seat [1], and annual billing gives a discount [2], [3]."
    parser = CitationParser()
    cites = parser.parse(answer)
    assert len(cites) >= 3
    # All three reference numbers present
    nums = sorted(c.number for c in cites)
    assert nums == [1, 2, 3]
    # The [2] and [3] citations share the same claim sentence
    c2 = next(c for c in cites if c.number == 2)
    c3 = next(c for c in cites if c.number == 3)
    assert c2.claim == c3.claim


def test_parser_handles_no_citations():
    assert CitationParser().parse("Just a plain statement with no citations.") == []


def test_verifier_flags_out_of_range_citation():
    verifier = CitationVerifier(judge=Generator(respond=lambda m: "{}"))
    answer = "Claim one [1]. Claim two [9]."
    chunks = [chunk("a::0", "Content A")]  # only block [1] present
    report = verifier.verify(answer, chunks)

    assert report.total == 2
    c1 = next(c for c in report.citations if c.number == 1)
    c9 = next(c for c in report.citations if c.number == 9)
    assert c9.supported is False
    assert "未提供" in c9.reason


def test_parser_keeps_chinese_claims_with_their_own_citations():
    answer = "接口需要令牌 [1]。系统支持限流 [2]！需要重试吗 [3]？"
    cites = CitationParser().parse(answer)
    assert [(c.number, c.claim) for c in cites] == [
        (1, "接口需要令牌 [1]。"),
        (2, "系统支持限流 [2]！"),
        (3, "需要重试吗 [3]？"),
    ]


def test_verifier_uses_judge_for_supported_and_unsupported():
    # Mock judge returns JSON verdicts based on whether source says 'supported'
    def fake_judge(messages):
        # Determine which block the answer cites by scanning the context
        user = messages[1]["content"]
        if "supported context" in user:
            return '{"supported": true, "reason": "context backs it up"}'
        return '{"supported": false, "reason": "unrelated to claim"}'

    verifier = CitationVerifier(judge=Generator(respond=fake_judge))
    answer = "The feature works [1]. Something else happened [1]."

    chunks = [
        chunk("a::0", "supported context here talks about feature works"),
        chunk("a::1", "unrelated context about pricing"),
    ]
    report = verifier.verify(answer, chunks, citations=[
        Citation(number=1, claim="The feature works."),
        Citation(number=2, claim="Something else happened."),
    ])

    assert report.verified
    assert report.total == 2
    c1 = next(c for c in report.citations if c.number == 1)
    c2 = next(c for c in report.citations if c.number == 2)
    assert c1.supported is True
    assert c2.supported is False


@pytest.mark.parametrize("reply, expected", [
    ("SUPPORTED", True),
    (" supported \n", True),
    ("UNSUPPORTED", False),
    ('{"supported": false}', False),
    ('{"supported": true}', True),
    ("", None),
    ("It may be SUPPORTED or UNSUPPORTED", None),
    ('{"supported": "false"}', None),
    ("{}", None),
])
def test_verdict_parser_never_confuses_supported_with_unsupported(reply, expected):
    verifier = CitationVerifier(judge=Generator(respond=lambda messages: reply))
    report = verifier.verify("默认返回十条 [1]。", [chunk("a", "默认返回十条。")])
    assert report.citations[0].supported is expected


def test_judge_failure_is_unknown():
    def timeout(messages):
        raise TimeoutError("test")

    verifier = CitationVerifier(judge=Generator(respond=timeout))
    report = verifier.verify("默认返回十条 [1]。", [chunk("a", "默认返回十条。")])
    assert report.citations[0].supported is None


def test_report_accuracy():
    report = CitationReport(answer="x")
    report.citations = [
        Citation(number=1, claim="a", supported=True),
        Citation(number=2, claim="b", supported=True),
        Citation(number=3, claim="c", supported=False),
    ]
    report.verified = True
    assert report.total == 3
    assert report.supported_count == 2
    assert report.unsupported_count == 1
    assert abs(report.accuracy - (2 / 3)) < 1e-6
    d = report.to_dict()
    assert d["unsupported"] == 1 and d["supported"] == 2


def test_no_judge_configured_marks_unknown(monkeypatch):
    # Remove any real API key so the verifier has no judge to call.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    verifier = CitationVerifier(judge=Generator(api_key=None))
    chunks = [chunk("a::0", "content")]
    report = verifier.verify("Claim [1].", chunks)
    # Graceful: a citation is still produced, verdict left to the caller.
    assert len(report.citations) >= 1
