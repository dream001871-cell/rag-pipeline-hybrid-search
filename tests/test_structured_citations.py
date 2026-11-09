import json

import pytest

from rag_pipeline.api import AskService
from rag_pipeline.citation import CitationVerifier
from rag_pipeline.confidence import ConfidenceScorer
from rag_pipeline.generation import Generator, parse_grounded_claims
from rag_pipeline.retrieval import RetrievalResult


@pytest.mark.parametrize("claim", [
    {"text": "北线全长 10 公里", "evidence": []},
    {"text": "北线全长 10 公里", "evidence": [True]},
    {"text": "北线全长 10 公里", "evidence": [3]},
    {"text": "北线全长 10 公里", "evidence": [1, 1]},
    {"text": "北线全长 10 公里 [1]", "evidence": [1]},
    {"text": "北线全长 10 公里\n无依据的文字", "evidence": [1]},
])
def test_invalid_claims_are_rejected(claim):
    raw = json.dumps({"claims": [claim], "partial": False})
    with pytest.raises(ValueError):
        parse_grounded_claims(raw, 2)


@pytest.mark.parametrize("data", [
    {"partial": False},
    {"claims": [], "partial": None},
    {"claims": [], "partial": "false"},
    {"claims": [], "unexpected": True},
])
def test_invalid_answer_schema_is_rejected(data):
    with pytest.raises(ValueError):
        parse_grounded_claims(json.dumps(data), 2)


@pytest.mark.parametrize("supported", [True, False])
@pytest.mark.parametrize("partial_fields", [{}, {"partial": True}])
def test_joint_evidence_and_fixed_partial_note_through_ask_service(supported, partial_fields):
    chunks = [
        RetrievalResult("north", "北线全长 10 公里。", {}),
        RetrievalResult("south", "南线全长 5 公里。", {}),
    ]
    for chunk in chunks:
        chunk.rerank_score = 0.8

    class Retriever:
        def retrieve(self, question, top_k):
            return chunks

    verifications = []

    def respond(messages):
        if "completeness judge" in messages[0]["content"]:
            return '{"score": 0.8, "reason": "覆盖两条线路的比较"}'
        if "判断 source 列表" in messages[0]["content"]:
            verifications.append(json.loads(messages[1]["content"]))
            return json.dumps({"supported": supported, "reason": "共同证据判定"})
        return json.dumps({"claims": [{"text": "北线比南线长 5 公里", "evidence": [1, 2]}],
                           **partial_fields})

    generator = Generator(respond=respond, structured_answers=True)
    service = AskService(Retriever(), generator, CitationVerifier(generator), ConfidenceScorer(generator))
    reply = service.answer("哪条线路更长、相差多少？通车时间是什么？")
    assert verifications == [{"claim": "北线比南线长 5 公里", "source": [
        {"number": 1, "text": chunks[0].content},
        {"number": 2, "text": chunks[1].content},
    ]}]
    assert reply["can_answer"] is supported
    if supported:
        expected = "北线比南线长 5 公里 [1][2]。"
        if partial_fields:
            expected += "\n现有资料仅能支持以上内容。"
        assert reply["answer"] == expected
        assert reply["citations"]["supported"] == 2
    else:
        assert "存在原文不支持的引用" in reply["confidence"]["details"]["gate_failures"]


def test_rank_weighted_confidence_preserves_strong_head_scores():
    chunks = []
    for i, score in enumerate([7.13, 4.55] + [-6.0] * 18):
        chunk = RetrievalResult(str(i), "片段", {})
        chunk.rerank_score = score
        chunks.append(chunk)
    assert ConfidenceScorer()._retrieval_confidence(chunks) == 0.0
    scorer = ConfidenceScorer(ranked_cross_encoder=True)
    assert scorer._retrieval_confidence(chunks) > 0.8
    assert scorer._retrieval_confidence(chunks[2:]) < 0.01
