import os
import re
import json
import math
import statistics
from typing import List, Optional, Callable, Dict, Any
from dataclasses import dataclass, field

from rag_pipeline.retrieval import RetrievalResult
from rag_pipeline.embedding import EmbeddingService, MockEmbeddingService
from rag_pipeline.generation import Generator, GroundedPrompt, select_context
from rag_pipeline.fusion import DEFAULT_TOP_K
from rag_pipeline.citation import CitationReport, CitationParser, CitationVerifier
from rag_pipeline.confidence import ConfidenceScorer, ConfidenceReport
from rag_evaluation.golden_dataset import GoldenDataset, GoldenQuestion

REAL_EMBEDDING_MODEL = "text-embedding-3-small"
REAL_JUDGE_MODEL = "gpt-4o-mini"
MOCK_WARNING = "WARNING: Running on mock embeddings. Numbers are not meaningful for reporting."


@dataclass
class MetricResult:
    """A single metric's score and reasoning."""

    name: str
    score: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "score": round(self.score, 4), "reason": self.reason}


@dataclass
class EvalQuestionResult:
    """Full evaluation result for one golden question."""

    question_id: str
    category: str
    question: str
    golden_answer: str
    generated_answer: str
    source_chunks: List[RetrievalResult] = field(default_factory=list)
    citation_report: Optional[CitationReport] = None
    confidence: Optional[ConfidenceReport] = None
    correctness: Optional[MetricResult] = None
    faithfulness: Optional[MetricResult] = None
    retrieval_relevance: Optional[MetricResult] = None
    citation_accuracy: Optional[MetricResult] = None
    latency_ms: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        d = {
            "question_id": self.question_id,
            "category": self.category,
            "question": self.question,
            "golden_answer": self.golden_answer,
            "generated_answer": self.generated_answer,
            "latency_ms": round(self.latency_ms, 1),
            "error": self.error,
        }
        for name in ("correctness", "faithfulness", "retrieval_relevance", "citation_accuracy"):
            mr = getattr(self, name)
            d[name] = mr.to_dict() if mr else None
        return d


@dataclass
class EvalReport:
    """Aggregate evaluation report across all questions."""

    results: List[EvalQuestionResult] = field(default_factory=list)
    chunking_strategy: str = ""
    total_questions: int = 0
    errors: int = 0
    overall: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_category: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)

    def compute(self):
        self.total_questions = len(self.results)
        self.errors = sum(1 for r in self.results if r.error)
        self.overall = self._aggregate(self.results)
        by_cat: Dict[str, list] = {}
        for r in self.results:
            by_cat.setdefault(r.category, []).append(r)
        self.by_category = {cat: self._aggregate(res) for cat, res in by_cat.items()}

    def _aggregate(self, results: List[EvalQuestionResult]) -> Dict[str, Dict[str, float]]:
        metric_names = ["correctness", "faithfulness", "retrieval_relevance", "citation_accuracy"]
        agg = {}
        for name in metric_names:
            scores = [
                getattr(r, name).score
                for r in results
                if getattr(r, name) is not None
            ]
            if not scores:
                agg[name] = {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "n": 0}
                continue
            agg[name] = {
                "mean": round(sum(scores) / len(scores), 4),
                "median": round(statistics.median(scores), 4),
                "min": round(min(scores), 4),
                "max": round(max(scores), 4),
                "n": len(scores),
            }
        return agg

    def to_dict(self) -> dict:
        return {
            "chunking_strategy": self.chunking_strategy,
            "total_questions": self.total_questions,
            "errors": self.errors,
            "overall": self.overall,
            "by_category": self.by_category,
        }

    def summary(self) -> str:
        lines = [
            f"Eval Report (strategy={self.chunking_strategy!r}, "
            f"questions={self.total_questions}, errors={self.errors})",
            "",
            "Overall:",
        ]
        for metric, vals in self.overall.items():
            lines.append(
                f"  {metric:24s}  mean={vals['mean']:.3f}  "
                f"median={vals['median']:.3f}  n={vals['n']}"
            )
        if self.by_category:
            lines.append("")
            lines.append("By category:")
            for cat, metrics in self.by_category.items():
                parts = [f"{k}={v['mean']:.3f}" for k, v in metrics.items()]
                lines.append(f"  {cat:14s}  " + "  ".join(parts))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Metric scorers
# ---------------------------------------------------------------------------

class CorrectnessScorer:
    """LLM-as-judge comparing generated answer against the golden answer.

    Uses an LLM judge when configured; falls back to lexical overlap (token
    Jaccard) for offline/mock runs.
    """

    def __init__(self, judge: Optional[Generator] = None):
        self.judge = judge

    def score(self, question: GoldenQuestion, generated_answer: str) -> MetricResult:
        if self.judge and self.judge.is_configured:
            return self._judge_score(question, generated_answer)
        return self._lexical_score(question, generated_answer)

    def _judge_score(self, q: GoldenQuestion, answer: str) -> MetricResult:
        system = (
            "You are evaluating a RAG system answer. \n"
            f"Question: {q.question}\n"
            f"Golden answer: {q.answer}\n"
            f"System answer: {answer}\n"
            "Score the system answer from 0.0 to 1.0 where:\n"
            "1.0 = correct and complete, 0.7 = mostly correct minor gaps, \n"
            "0.4 = partially correct, 0.0 = wrong or hallucinated.\n"
            "Reply with ONLY a float between 0.0 and 1.0. No explanation."
        )
        raw = _ask_judge(self.judge, system)
        score = _parse_judge_float(raw, default=0.5)
        return MetricResult(name="correctness", score=score, reason="LLM judge")

    def _lexical_score(self, q: GoldenQuestion, answer: str) -> MetricResult:
        golden_tokens = _tokens(q.answer)
        if not golden_tokens:
            return MetricResult(name="correctness", score=1.0, reason="empty golden")
        answer_tokens = _tokens(answer)
        if not answer_tokens:
            return MetricResult(name="correctness", score=0.0, reason="empty answer")
        overlap = golden_tokens & answer_tokens
        jaccard = len(overlap) / len(golden_tokens | answer_tokens)
        # For no-answer questions, "I don't know" is the correct answer
        if q.category == "no_answer":
            decline = _is_decline(answer)
            score = 1.0 if decline else min(jaccard + 0.2, 0.5)
            reason = (
                "correct decline" if decline else
                "no_answer question but answer does not decline"
            )
        else:
            score = jaccard
            reason = f"token overlap {len(overlap)}/{len(golden_tokens | answer_tokens)}"
        return MetricResult(name="correctness", score=score, reason=reason)


class FaithfulnessScorer:
    """Checks what fraction of the generated answer's claims are grounded
    in the retrieved context.

    For LLM-as-judge mode: asks the judge whether each claim is supported.
    Lexical mode: extracts sentences from the answer and checks overlap with
    retrieved chunk content.
    """

    def __init__(self, judge: Optional[Generator] = None):
        self.judge = judge

    def score(
        self,
        answer: str,
        source_chunks: List[RetrievalResult],
    ) -> MetricResult:
        claims = CitationParser()._split_sentences(answer)
        claims = [c for c in claims if c.strip()]
        if not claims:
            return MetricResult(name="faithfulness", score=1.0, reason="no claims")

        if self.judge and self.judge.is_configured:
            return self._judge_score(claims, source_chunks)
        return self._lexical_score(claims, source_chunks)

    def _judge_score(
        self,
        claims: List[str],
        source_chunks: List[RetrievalResult],
    ) -> MetricResult:
        context = "\n\n".join(
            f"[{i+1}] {r.content}" for i, r in enumerate(source_chunks[:5])
        )
        answer_text = " ".join(claims)
        prompt = (
            "You are checking if an answer is grounded in the provided context.\n"
            f"Context chunks: {context}\n"
            f"Answer: {answer_text}\n"
            "Score 0.0 to 1.0 where 1.0 = every claim is in the context, \n"
            "0.0 = answer contains claims not in context.\n"
            "Reply with ONLY a float. No explanation."
        )
        raw = _ask_judge(self.judge, prompt)
        score = _parse_judge_float(raw, n_claims=len(claims), default=0.0)
        reason = f"LLM judge: {score:.2f}"
        return MetricResult(name="faithfulness", score=score, reason=reason)

    def _lexical_score(
        self,
        claims: List[str],
        source_chunks: List[RetrievalResult],
    ) -> MetricResult:
        context_text = " ".join(r.content for r in source_chunks).lower()
        grounded = 0
        for claim in claims:
            claim_tokens = _tokens(claim)
            if not claim_tokens:
                grounded += 1
                continue
            found = sum(1 for t in claim_tokens if t in context_text)
            if found / len(claim_tokens) >= 0.4:
                grounded += 1
        score = grounded / len(claims) if claims else 1.0
        return MetricResult(
            name="faithfulness",
            score=score,
            reason=f"{grounded}/{len(claims)} claims grounded (lexical)",
        )


class RetrievalRelevanceScorer:
    """Did the retrieved chunks actually relate to the question's expected
    source material?

    Two signals:
    1. Section coverage: does the golden answer's source_sections appear in the
       retrieved chunks' metadata or content?
    2. Content overlap: token overlap between retrieved content and golden answer.
    Both are combined with equal weight.
    """

    def __init__(self, judge: Optional[Generator] = None):
        self.judge = judge

    def score(
        self,
        question: GoldenQuestion,
        source_chunks: List[RetrievalResult],
    ) -> MetricResult:
        if not source_chunks:
            return MetricResult(name="retrieval_relevance", score=0.0, reason="no chunks")
        if self.judge and self.judge.is_configured:
            return self._judge_score(question, source_chunks)

        # Signal 1: source section coverage
        section_score = self._section_coverage(question, source_chunks)

        # Signal 2: content token overlap
        content_score = self._content_overlap(question, source_chunks)

        score = 0.5 * section_score + 0.5 * content_score
        reason = (
            f"section_coverage={section_score:.2f}, "
            f"content_overlap={content_score:.2f}"
        )
        return MetricResult(name="retrieval_relevance", score=score, reason=reason)

    def _judge_score(
        self,
        question: GoldenQuestion,
        source_chunks: List[RetrievalResult],
    ) -> MetricResult:
        context = "\n\n".join(
            f"[{i+1}] {r.content}" for i, r in enumerate(source_chunks[:5])
        )
        prompt = (
            "You are checking if retrieved chunks are relevant to a question.\n"
            f"Question: {question.question}\n"
            f"Retrieved chunks: {context}\n"
            "Score 0.0 to 1.0 where 1.0 = all chunks are directly relevant,\n"
            "0.0 = chunks are off-topic.\n"
            "Reply with ONLY a float. No explanation."
        )
        raw = _ask_judge(self.judge, prompt)
        score = _parse_judge_float(raw, default=0.5)
        return MetricResult(name="retrieval_relevance", score=score, reason="LLM judge")

    def _section_coverage(
        self,
        question: GoldenQuestion,
        chunks: List[RetrievalResult],
    ) -> float:
        if not question.source_sections:
            return 1.0
        chunk_text = " ".join(r.content.lower() for r in chunks)
        chunk_meta_text = " ".join(
            " ".join(str(v).lower() for v in (r.metadata or {}).values())
            for r in chunks
        )
        all_text = chunk_text + " " + chunk_meta_text
        hits = sum(1 for s in question.source_sections if s.lower() in all_text)
        return hits / len(question.source_sections)

    def _content_overlap(
        self,
        question: GoldenQuestion,
        chunks: List[RetrievalResult],
    ) -> float:
        golden_tokens = _tokens(question.answer)
        if not golden_tokens:
            return 1.0
        chunk_text = " ".join(r.content.lower() for r in chunks)
        hits = sum(1 for t in golden_tokens if t in chunk_text)
        return hits / len(golden_tokens)


class CitationAccuracyScorer:
    """Do the generated answer's citations actually support the claims?

    Reuses the CitationVerifier infrastructure from Phase 3.2. For no_answer
    questions, accuracy is 1.0 if the system produced no citations (correct) or
    penalized if citations are fabricated.
    """

    def __init__(self, verifier: Optional[CitationVerifier] = None):
        self.verifier = verifier

    def score(
        self,
        question: GoldenQuestion,
        generated_answer: str,
        source_chunks: List[RetrievalResult],
        citation_report: Optional[CitationReport] = None,
    ) -> MetricResult:
        citations = CitationParser().parse(generated_answer)

        # No-answer questions should not have citations
        if question.category == "no_answer":
            if not citations:
                return MetricResult(
                    name="citation_accuracy",
                    score=1.0,
                    reason="no-answer question correctly has no citations",
                )
            return MetricResult(
                name="citation_accuracy",
                score=0.0,
                reason=f"no-answer question incorrectly has {len(citations)} citations",
            )

        # For lookup / multi_hop / ambiguous, verify citations
        if not citations:
            return MetricResult(
                name="citation_accuracy",
                score=1.0 if question.category == "ambiguous" else 0.5,
                reason="no citations present",
            )

        if citation_report and citation_report.total > 0:
            score = citation_report.accuracy
            supported = citation_report.supported_count
            total = citation_report.total
            return MetricResult(
                name="citation_accuracy",
                score=score,
                reason=f"{supported}/{total} citations verified",
            )

        return MetricResult(
            name="citation_accuracy",
            score=0.5,
            reason="citation report unavailable; cannot verify",
        )


# ---------------------------------------------------------------------------
# EvalRunner
# ---------------------------------------------------------------------------

class EvalRunner:
    """Runs the full evaluation suite: retrieve → generate → verify → score.

    Accepts pluggable retrieve_fn and generate_fn for testing. The static
    build() method creates a runner wired to the real pipeline.
    """

    def __init__(
        self,
        dataset: GoldenDataset,
        retrieve_fn: Callable[[str], List[RetrievalResult]],
        generate_fn: Callable[[str, List[RetrievalResult]], dict],
        judge: Optional[Generator] = None,
        citation_verifier: Optional[CitationVerifier] = None,
        chunking_strategy: str = "fixed",
    ):
        self.dataset = dataset
        self.retrieve_fn = retrieve_fn
        self.generate_fn = generate_fn
        self.judge = judge
        self.citation_verifier = citation_verifier

        self.correctness = CorrectnessScorer(judge=judge)
        self.faithfulness = FaithfulnessScorer(judge=judge)
        self.retrieval_relevance = RetrievalRelevanceScorer(judge=judge)
        self.citation_accuracy = CitationAccuracyScorer(verifier=citation_verifier)

        self.chunking_strategy = chunking_strategy

    def run(self, filter_category: Optional[str] = None) -> EvalReport:
        questions = (
            self.dataset.filter(filter_category)
            if filter_category
            else list(self.dataset)
        )

        results = []
        for gq in questions:
            try:
                r = self._eval_question(gq)
                results.append(r)
            except Exception as exc:
                results.append(
                    EvalQuestionResult(
                        question_id=gq.id,
                        category=gq.category,
                        question=gq.question,
                        golden_answer=gq.answer,
                        generated_answer="",
                        error=str(exc),
                    )
                )

        report = EvalReport(results=results, chunking_strategy=self.chunking_strategy)
        report.compute()
        return report

    def _eval_question(self, gq: GoldenQuestion) -> EvalQuestionResult:
        import time

        t0 = time.time()
        chunks = self.retrieve_fn(gq.question)
        gen_out = self.generate_fn(gq.question, chunks)
        generated_answer = gen_out.get("answer", "")

        citation_report = None
        if self.citation_verifier and generated_answer.strip():
            citation_report = self.citation_verifier.verify(
                generated_answer, chunks
            )

        latency = (time.time() - t0) * 1000

        return EvalQuestionResult(
            question_id=gq.id,
            category=gq.category,
            question=gq.question,
            golden_answer=gq.answer,
            generated_answer=generated_answer,
            source_chunks=chunks,
            citation_report=citation_report,
            correctness=self.correctness.score(gq, generated_answer),
            faithfulness=self.faithfulness.score(generated_answer, chunks),
            retrieval_relevance=self.retrieval_relevance.score(gq, chunks),
            citation_accuracy=self.citation_accuracy.score(
                gq, generated_answer, chunks, citation_report
            ),
            latency_ms=latency,
        )

    @classmethod
    def build(
        cls,
        dataset_path: str,
        ingest_fn: Callable[[], Any],
        hybrid_retriever: Any,
        judge: Optional[Generator] = None,
        citation_judge: Optional[Generator] = None,
        chunking_strategy: str = "fixed",
    ) -> "EvalRunner":
        """Create a runner wired to the real pipeline components.

        ingest_fn: call once to ensure the corpus is indexed.
        hybrid_retriever: HybridRetriever instance with retrieve() method.
        """
        ingest_fn()
        generator = judge or Generator()

        def retrieve_fn(question: str) -> List[RetrievalResult]:
            return select_context(hybrid_retriever.retrieve(question, top_k=DEFAULT_TOP_K))

        def generate_fn(
            question: str, chunks: List[RetrievalResult]
        ) -> dict:
            return generator.generate(question, chunks)

        verifier = None
        if citation_judge:
            verifier = CitationVerifier(judge=citation_judge)

        ds = GoldenDataset.load(dataset_path)
        return cls(
            dataset=ds,
            retrieve_fn=retrieve_fn,
            generate_fn=generate_fn,
            judge=judge,
            citation_verifier=verifier,
            chunking_strategy=chunking_strategy,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_eval_mode(
    force_real: bool = False,
    force_mock: bool = False,
) -> Dict[str, Any]:
    """选择真实或模拟评估；DeepSeek 配合本地向量，其余服务保留原有配置。"""
    if force_real and force_mock:
        raise ValueError("force_real and force_mock are mutually exclusive")
    has_key = bool(
        os.environ.get("DEEPSEEK_API_KEY", "").strip()
        or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    )
    if force_real and not has_key:
        raise RuntimeError(
            "No LLM API key set; cannot run in real mode. "
            "Set DEEPSEEK_API_KEY, OPENROUTER_API_KEY or OPENAI_API_KEY, drop --real, "
            "or use --mock explicitly."
        )

    if has_key and not force_mock:
        if os.environ.get("DEEPSEEK_API_KEY", "").strip():
            from rag_pipeline.embedding import create_embedding_service

            embedding_service = create_embedding_service()
            judge = Generator()
            return {
                "mode": "real",
                "embedding_model": embedding_service.model,
                "judge_model": judge.model,
                "embedding_service": embedding_service,
                "judge": judge,
            }
        return {
            "mode": "real",
            "embedding_model": REAL_EMBEDDING_MODEL,
            "judge_model": REAL_JUDGE_MODEL,
            "embedding_service": EmbeddingService(model=REAL_EMBEDDING_MODEL),
            "judge": Generator(model=REAL_JUDGE_MODEL),
        }

    print(MOCK_WARNING)
    return {
        "mode": "mock",
        "embedding_model": "mock",
        "judge_model": "mock",
        "embedding_service": MockEmbeddingService(dimensions=384),
        "judge": None,
    }


def _ask_judge(judge: Generator, prompt: str) -> str:
    """Send a single unscaffolded prompt to the judge (real or mock respond)."""
    return judge._call([{"role": "user", "content": prompt}], {})


def _parse_judge_float(
    raw: str,
    n_claims: Optional[int] = None,
    default: float = 0.0,
) -> float:
    """Parse a judge reply into a 0..1 score.

    Accepts the calibrated "Reply with ONLY a float" response, plus backward
    compatible JSON: {"score": float} or {"grounded": [bool,...]}. Grounded
    arrays are averaged across `n_claims` claims.
    """
    text = (raw or "").strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
        except Exception:
            data = None
        if isinstance(data, dict):
            grounded = data.get("grounded")
            if isinstance(grounded, list):
                flags = [bool(x) for x in grounded]
                n = n_claims or len(flags)
                if n == 0:
                    return 1.0
                padded = [flags[i] if i < len(flags) else False for i in range(n)]
                return sum(1 for f in padded if f) / n
            if "score" in data:
                try:
                    return max(0.0, min(1.0, float(data["score"])))
                except (TypeError, ValueError):
                    pass
    m = re.search(r"[-+]?\d*\.?\d+", text)
    if m:
        try:
            return max(0.0, min(1.0, float(m.group(0))))
        except ValueError:
            pass
    return default


def _tokens(text: str) -> set:
    text = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
    stop = {"the", "a", "an", "and", "of", "to", "in", "on", "for", "is",
            "are", "was", "were", "been", "be", "have", "has", "had", "it",
            "its", "that", "this", "with", "you", "i", "do", "does", "did"}
    return {t for t in text.split() if t and t not in stop}


def _is_decline(answer: str) -> bool:
    low = (answer or "").lower()
    signals = [
        "i don't have enough information",
        "i don't have sufficient",
        "i do not have enough",
        "not enough information",
        "insufficient information",
        "cannot answer",
        "can't answer",
        "no information",
        "don't have information",
    ]
    return any(s in low for s in signals)
