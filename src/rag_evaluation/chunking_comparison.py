import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any

from rag_evaluation.golden_dataset import GoldenDataset
from rag_evaluation.eval_metrics import EvalRunner, EvalReport, EvalQuestionResult
from rag_pipeline.citation import CitationVerifier
from rag_pipeline.retrieval import RetrievalResult
from rag_pipeline.fusion import DEFAULT_TOP_K
from rag_pipeline.generation import select_context

METRICS = ["correctness", "faithfulness", "retrieval_relevance", "citation_accuracy"]


@dataclass
class StrategyRow:
    """Aggregate results for one chunking strategy across the eval suite."""

    strategy: str
    report: EvalReport
    means: Dict[str, float] = field(default_factory=dict)
    medians: Dict[str, float] = field(default_factory=dict)
    per_question: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def __post_init__(self):
        self.means = {m: self.report.overall[m]["mean"] for m in METRICS if m in self.report.overall}
        self.medians = {m: self.report.overall[m]["median"] for m in METRICS if m in self.report.overall}
        self.per_question = {
            r.question_id: {
                m: (getattr(r, m).score if getattr(r, m) else None)
                for m in METRICS
            }
            for r in self.report.results
        }

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "means": self.means,
            "medians": self.medians,
            "errors": self.report.errors,
            "questions": self.report.total_questions,
        }

    def report_dict(self) -> dict:
        """Per-strategy block for data/comparison_report.json (Part C).

        Flattened metric means, an overall avg, and a per-type breakdown
        (correctness + faithfulness per golden-dataset category).
        """
        means = self.means
        avg = sum(means.get(m, 0.0) for m in METRICS) / len(METRICS) if means else 0.0
        per_type: Dict[str, Dict[str, float]] = {}
        for cat in ("lookup", "multi_hop", "no_answer", "ambiguous"):
            bc = self.report.by_category.get(cat, {})
            per_type[cat] = {
                m: bc.get(m, {}).get("mean", 0.0)
                for m in ("correctness", "faithfulness")
            }
        return {
            "correctness": means.get("correctness", 0.0),
            "faithfulness": means.get("faithfulness", 0.0),
            "retrieval_relevance": means.get("retrieval_relevance", 0.0),
            "citation_accuracy": means.get("citation_accuracy", 0.0),
            "avg": avg,
            "per_type": per_type,
        }


@dataclass
class ComparisonResult:
    """Winner per metric, per-category winners, and all strategy rows."""

    rows: Dict[str, StrategyRow] = field(default_factory=dict)
    metric_winners: Dict[str, str] = field(default_factory=dict)
    margin: float = 0.01

    def decide_winners(self, margin: float = 0.01) -> None:
        """For each metric, the strategy with the highest mean wins.

        Strategies within `margin` of the winner are ties (reported as
        comma-joined names in rank order).
        """
        self.margin = margin
        self.metric_winners = {}
        for m in METRICS:
            ranked = sorted(
                ((s, row.means.get(m, 0.0)) for s, row in self.rows.items()),
                key=lambda kv: kv[1],
                reverse=True,
            )
            best_score = ranked[0][1] if ranked else 0.0
            tied = [s for s, sc in ranked if abs(sc - best_score) <= margin]
            self.metric_winners[m] = "+".join(tied) if len(tied) > 1 else ranked[0][0]

    def overall_ranking(self) -> List[tuple]:
        """Strategies sorted by overall avg (across all metrics), descending."""
        ranking = []
        for s, row in self.rows.items():
            means = row.means
            avg = sum(means.get(m, 0.0) for m in METRICS) / len(METRICS) if means else 0.0
            ranking.append((s, avg))
        return sorted(ranking, key=lambda kv: kv[1], reverse=True)

    def category_winners(self) -> Dict[str, Dict[str, str]]:
        """Per-category winners of each metric, if reports include categories."""
        out = {}
        for cat in ("lookup", "multi_hop", "no_answer", "ambiguous"):
            scores = {}
            for s, row in self.rows.items():
                bc = row.report.by_category.get(cat, {})
                for m in METRICS:
                    mean = bc.get(m, {}).get("mean", 0.0)
                    scores.setdefault(m, {})[s] = mean
            cat_winners = {}
            for m, strat_scores in scores.items():
                best = max(strat_scores.items(), key=lambda kv: kv[1])
                cat_winners[m] = best[0]
            out[cat] = cat_winners
        return out

    def summary(self) -> str:
        lines = [
            "Chunking Strategy Comparison",
            "============================",
            "",
            "Per-strategy mean scores:",
        ]
        headers = [f"{'strategy':12s}"] + [f"{m[:14]:>14s}" for m in METRICS]
        lines.append("  ".join(headers))
        for s, row in sorted(self.rows.items()):
            cells = [f"{s:12s}"] + [f"{row.means.get(m, 0.0):>14.3f}" for m in METRICS]
            lines.append("  ".join(cells))
        lines.append("")
        lines.append("Metric winners:")
        for m in METRICS:
            lines.append(f"  {m:22s}: {self.metric_winners.get(m, 'n/a')}")
        lines.append("")
        ca = self.category_winners()
        if any(ca.values()):
            lines.append("Category-level winners (by mean):")
            catnames = {"lookup", "multi_hop", "no_answer", "ambiguous"}
            for cat in catnames:
                parts = [f"{m}={vr}" for m, vr in ca.get(cat, {}).items()]
                lines.append(f"  {cat:12s}: " + ", ".join(parts))
        return "\n".join(lines)


class ChunkingComparison:
    """Runs the same eval suite for each chunking strategy and compares.

    Usage:
        from rag_evaluation.chunking_comparison import ChunkingComparison
        cmp = ChunkingComparison(
            dataset=dataset,
            strategies=["fixed", "recursive", "semantic"],
            build_pipeline_fn=pipeline_factory,   # returns a prepare/retrieve/generate trio
        )
        result = cmp.run()
        print(result.summary())
    """

    def __init__(
        self,
        dataset: GoldenDataset,
        strategies: List[str],
        pipeline_factory: Callable[[str], Any],
        generate_fn: Optional[Callable[[str, List[RetrievalResult]], dict]] = None,
        judge=None,
        margin: float = 0.01,
    ):
        """
        pipeline_factory(strategy) -> object with:
            - retrieve(question, top_k) -> List[RetrievalResult]
        Each factory call must build a FRESH index for that strategy so the
        comparison is apples-to-apples (chunk IDs differ across strategies).
        """
        self.dataset = dataset
        self.strategies = strategies
        self.pipeline_factory = pipeline_factory
        self.retrieve_fn = None  # replaced per-strategy
        self.generate_fn = generate_fn
        self.judge = judge
        self.margin = margin

    def run(self) -> ComparisonResult:
        # Structural fix: wire a real citation verifier (backed by the eval
        # judge) into the EvalRunner. Previously citation_accuracy defaulted to
        # a category-only constant, so it was identical across every chunking
        # strategy regardless of retrieval quality.
        verifier = (
            CitationVerifier(judge=self.judge)
            if self.judge is not None and self.judge.is_configured
            else None
        )
        rows: Dict[str, StrategyRow] = {}
        for strat in self.strategies:
            pipeline = self.pipeline_factory(strat)

            def make_retrieve(pipeline=pipeline):
                return lambda q: select_context(pipeline.retrieve(q, top_k=DEFAULT_TOP_K))

            runner = EvalRunner(
                dataset=self.dataset,
                retrieve_fn=make_retrieve(pipeline),
                generate_fn=self._default_generate,
                judge=self.judge,
                citation_verifier=verifier,
                chunking_strategy=strat,
            )
            report = runner.run()
            rows[strat] = StrategyRow(strategy=strat, report=report)

        result = ComparisonResult(rows=rows)
        result.decide_winners(margin=self.margin)
        self.result = result
        return result

    def _default_generate(self, question: str, chunks: List[RetrievalResult]) -> dict:
        """If no generate_fn provided, use the first chunk verbatim as a
        minimal deterministic stand-in so the comparison still exercises the
        retrieval path faithfully.
        """
        if self.generate_fn is not None:
            return self.generate_fn(question, chunks)
        if not chunks:
            return {"answer": "I don't have enough information in the context to answer this."}
        return {"answer": chunks[0].content[:200] + " [1]"}
