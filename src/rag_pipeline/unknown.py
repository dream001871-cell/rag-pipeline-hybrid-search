from typing import List, Optional, Dict
from dataclasses import dataclass, field
import re
import math

from .retrieval import RetrievalResult
from .generation import Generator, GroundedClaim, GroundedPrompt
from .confidence import ConfidenceScorer, ConfidenceReport
from .citation import CitationReport, CitationParser


@dataclass
class LowConfidenceAnswer:
    """Structured "I don't know" response produced when confidence is low.

    Instead of hallucinating, the system explains what it found, what it could
    not find, and which documents are worth checking manually.
    """

    question: str
    confidence: ConfidenceReport
    threshold: float
    found: List[str] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    candidate_documents: List[dict] = field(default_factory=list)
    summary: str = ""

    @property
    def should_answer(self) -> bool:
        return False

    def to_dict(self) -> dict:
        return {
            "can_answer": False,
            "reason": "insufficient_confidence",
            "threshold": self.threshold,
            "confidence": self.confidence.to_dict(),
            "found": self.found,
            "gaps": self.gaps,
            "candidate_documents": self.candidate_documents,
            "summary": self.summary,
        }


@dataclass
class AnswerOutcome:
    """Result of the confidence-gated answer flow.

    Either carries a full answer with citations + confidence (can_answer=True)
    or a LowConfidenceAnswer (can_answer=False) that declines gracefully.
    """

    can_answer: bool
    answer: Optional[str] = None
    confidence: ConfidenceReport = None
    low_confidence: Optional[LowConfidenceAnswer] = None
    citation_report: Optional[CitationReport] = None

    def to_dict(self) -> dict:
        if self.can_answer:
            return {
                "can_answer": True,
                "answer": self.answer,
                "confidence": self.confidence.to_dict(),
                "citations": (
                    self.citation_report.to_dict() if self.citation_report else None
                ),
            }
        return self.low_confidence.to_dict()


class UnknownHandler:
    """先检查引用依据与最低完整度，再按综合分数决定回答或拒答。"""

    def __init__(
        self,
        scorer: ConfidenceScorer,
        threshold: float = 0.5,
        found_highlights: int = 4,
        candidate_docs: int = 3,
        min_completeness: float = 0.5,
    ):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if not 0.0 <= min_completeness <= 1.0:
            raise ValueError("min_completeness 必须位于 [0, 1]")
        self.scorer = scorer
        self.threshold = threshold
        self.found_highlights = found_highlights
        self.candidate_docs = candidate_docs
        self.min_completeness = min_completeness

    def decide(
        self,
        question: str,
        answer: str,
        source_chunks: List[RetrievalResult],
        citation_report: Optional[CitationReport] = None,
        *,
        claims: Optional[list[GroundedClaim]] = None,
        generation_error: Optional[str] = None,
    ) -> AnswerOutcome:
        """必要条件不能被其他高分抵消；失败原因保留在评分报告中。"""
        confidence = self.scorer.score(
            question, answer, source_chunks, citation_report=citation_report
        )

        failures = []
        parser = CitationParser()
        parsed = parser.parse(answer, claims)
        if generation_error:
            failures.append("结构化答案格式无效")
            confidence.details["generation_error"] = generation_error
        if not answer.strip():
            failures.append("答案为空")
        if not source_chunks:
            failures.append("没有可供核验的来源片段")
        if not parsed:
            failures.append("答案没有引用")
        elif any(c.number < 1 or c.number > len(source_chunks) for c in parsed):
            failures.append("引用编号超出本次来源范围")

        if claims is None:
            # 兼容旧文本调用，仍要求每个文本句子都有引用。
            for sentence in parser._split_sentences(answer):
                if not parser.CITATION_RE.search(sentence):
                    failures.append("存在未标注引用的句子")
                    break
        elif claims and answer != "\n".join(claim.render() for claim in claims):
            failures.append("答案与已核验的事实不匹配")

        if citation_report is None or not citation_report.verified:
            failures.append("引用尚未完成校验")
        else:
            expected = [(c.number, c.claim) for c in parsed]
            received = [(c.number, c.claim) for c in citation_report.citations]
            if citation_report.answer != answer or expected != received:
                failures.append("引用报告与当前答案不匹配")
            if any(c.supported is False for c in citation_report.citations):
                failures.append("存在原文不支持的引用")
            if any(c.supported is not True and c.supported is not False for c in citation_report.citations):
                failures.append("存在无法验证的引用")

        if not math.isfinite(confidence.completeness) or confidence.completeness < self.min_completeness:
            failures.append("回答完整度低于最低要求")
        if not math.isfinite(confidence.composite) or confidence.composite < self.threshold:
            failures.append("综合置信度低于阈值")
        confidence.details["gate_failures"] = failures
        confidence.details["min_completeness"] = self.min_completeness

        if not failures:
            return AnswerOutcome(
                can_answer=True,
                answer=answer,
                confidence=confidence,
                citation_report=citation_report,
            )

        return AnswerOutcome(
            can_answer=False,
            confidence=confidence,
            low_confidence=self._build_low_confidence(
                question, answer, source_chunks, confidence
            ),
        )

    def _build_low_confidence(
        self,
        question: str,
        answer: str,
        source_chunks: List[RetrievalResult],
        confidence: ConfidenceReport,
    ) -> LowConfidenceAnswer:
        found = self._what_was_found(question, source_chunks)
        gaps = self._what_was_missing(question, answer, found)
        docs = self._candidate_documents(source_chunks)
        summary = self._summarize(question, found, gaps)
        failures = confidence.details.get("gate_failures", [])
        if failures:
            summary = "未通过检查：" + "；".join(failures) + "。" + summary

        return LowConfidenceAnswer(
            question=question,
            confidence=confidence,
            threshold=self.threshold,
            found=found,
            gaps=gaps,
            candidate_documents=docs,
            summary=summary,
        )

    # ---- What the system found -------------------------------------------
    def _what_was_found(
        self, question: str, source_chunks: List[RetrievalResult]
    ) -> List[str]:
        """Top topics / snippets from the retrieved chunks, so the user sees
        the system did find *something* relevant.
        """
        items = []
        for r in source_chunks[: self.found_highlights]:
            snippet = self._snippet(r.content, max_len=120)
            doc = (r.metadata or {}).get("source_file") or "unknown"
            item = snippet
            if doc != "unknown":
                item = f"{snippet}（来源：{doc}）"
            items.append(item)
        return items

    def _snippet(self, text: str, max_len: int = 120) -> str:
        text = re.sub(r"\s+", " ", text or "").strip()
        return text[:max_len] + ("..." if len(text) > max_len else "")

    # ---- What could not be found -----------------------------------------
    def _what_was_missing(
        self, question: str, answer: str, found: List[str]
    ) -> List[str]:
        """Question terms that did not surface in the answer or found chunks."""
        q_tokens = self.scorer._tokens(question)
        found_text = " ".join(found).lower()
        answer_bits = tokenize(answer)

        missing = []
        for t in q_tokens:
            if t not in answer_bits and t not in found_text:
                missing.append(t)
        return missing

    # ---- Documents worth checking ----------------------------------------
    def _candidate_documents(self, source_chunks: List[RetrievalResult]) -> List[dict]:
        """Deduplicated source documents, ranked by their top chunk score."""
        seen = {}
        for r in source_chunks:
            meta = r.metadata or {}
            doc_id = meta.get("doc_id") or meta.get("source_file") or "unknown"
            title = meta.get("source_file") or doc_id
            score = (
                r.rerank_score
                if r.rerank_score is not None
                else (r.rrf_score if r.rrf_score is not None else r.dense_score)
            )
            if doc_id not in seen:
                seen[doc_id] = {"doc_id": doc_id, "title": title, "score": score or 0.0}
            else:
                seen[doc_id]["score"] = max(seen[doc_id]["score"], score or 0.0)

        ranked = sorted(seen.values(), key=lambda d: d["score"], reverse=True)
        return ranked[: self.candidate_docs]

    def _summarize(
        self, question: str, found: List[str], gaps: List[str]
    ) -> str:
        parts = []
        if found:
            parts.append(
                f"找到了 {len(found)} 个与问题相关的片段，但证据不足，暂时无法给出确定回答。"
            )
        else:
            parts.append(
                "在已索引的文档中未找到与该问题相关的内容。"
            )
        if gaps:
            parts.append(
                "以下内容尚无法确认："
                + ", ".join(repr(g) for g in gaps)
                + "。"
            )
        parts.append(
            "请手动核对列出的原始文档后再作判断。"
        )
        return " ".join(parts)


def tokenize(text: str) -> set:
    out = set()
    for w in re.findall(r"[a-z0-9]+", (text or "").lower()):
        out.add(w)
    return out
