import json

import pytest

from rag_pipeline.citation import CitationVerifier
from rag_pipeline.confidence import ConfidenceScorer
from rag_pipeline.generation import Generator, parse_judge_json
from rag_pipeline.retrieval import RetrievalResult


def test_citation_judge_receives_only_claim_and_evidence():
    captured = []

    def respond(messages):
        captured.append(messages)
        return '{"supported": true, "reason": "日期和周年均有原文依据"}'

    source = RetrievalResult("a", "2023年7月26日举办建军96周年招待会。", {})
    answer = "招待会于2023年7月26日举办，庆祝建军96周年 [1]。"
    report = CitationVerifier(judge=Generator(respond=respond)).verify(answer, [source])
    assert report.citations[0].supported is True
    system, user = captured[0]
    payload = json.loads(user["content"])
    assert payload == {"claim": answer, "source": source.content}
    assert "回答格式" not in user["content"]
    assert "在回答中使用方括号" not in user["content"]
    assert source.content not in system["content"]


def test_completeness_judge_receives_original_question_and_existing_answer():
    captured = []

    def respond(messages):
        captured.append(messages)
        payload = json.loads(messages[1]["content"])
        assert payload["question"] == "GDP增长多少，连续几个季度增长？"
        assert payload["answer"] == "增长6.4%，连续第八个季度增长 [1]。"
        return '{"score": 1, "reason": "覆盖增长率及连续季度数"}'

    scorer = ConfidenceScorer(judge=Generator(respond=respond))
    report = scorer.score("GDP增长多少，连续几个季度增长？", "增长6.4%，连续第八个季度增长 [1]。", [])
    assert report.completeness == 1.0
    assert report.details["completeness_judge_error"] is None
    assert "回答格式" not in captured[0][1]["content"]


@pytest.mark.parametrize("reply", [
    '{"score": true}', '{"score": "1"}', '{"score": 2}', '{"score": -1}',
    '{"score": NaN}', '{"score": Infinity}', '{"score": 0, "score": 1}',
    '说明：{"score": 1}', '[{"score": 1}]', '{}',
])
def test_invalid_completeness_is_recorded_as_protocol_error(reply):
    scorer = ConfidenceScorer(judge=Generator(respond=lambda messages: reply))
    report = scorer.score("问题", "答案 [1]。", [])
    assert report.completeness == 0.0
    assert "协议错误" in report.details["completeness_judge_error"]


@pytest.mark.parametrize("reply, expected", [
    ('```json\n{"supported": false}\n```', False),
    ('{"supported": false, "supported": true}', None),
    ('{"supported": "true"}', None),
    ('SUPPORTED [1]', None),
    ('UNSUPPORTED', False),
    ('{"supported": true, "reason": {}}', None),
])
def test_citation_parser_does_not_turn_ambiguous_output_into_support(reply, expected):
    verifier = CitationVerifier(judge=Generator(respond=lambda messages: reply))
    source = RetrievalResult("a", "默认十条。", {})
    report = verifier.verify("默认十条 [1]。", [source])
    assert report.citations[0].supported is expected


def test_judge_errors_are_separate_from_negative_content_verdicts():
    source = RetrievalResult("a", "默认十条。", {})
    answer = "默认二十条 [1]。"
    verifier = CitationVerifier(judge=Generator(respond=lambda messages: '{"supported": false}'))
    citations = verifier.verify(answer, [source])
    scorer = ConfidenceScorer(judge=Generator(respond=lambda messages: '{"score": 0}'))
    report = scorer.score("多少条？", answer, [source], citations)
    assert report.details["citation_judge_errors"] == []
    assert report.details["completeness_judge_error"] is None
    assert report.citation_coverage == 0.0


def test_single_json_code_block_is_accepted_without_guessing_surrounding_prose():
    assert parse_judge_json('```json\n{"score": 0.5}\n```') == {"score": 0.5}
    with pytest.raises(ValueError):
        parse_judge_json('先给解释\n{"score": 1}')
