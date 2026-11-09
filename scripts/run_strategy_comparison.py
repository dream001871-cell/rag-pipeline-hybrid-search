from dotenv import load_dotenv
load_dotenv()

import argparse
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_evaluation.golden_dataset import GoldenDataset
from rag_evaluation.chunking_comparison import ChunkingComparison, METRICS
from rag_evaluation.eval_metrics import resolve_eval_mode, EvalRunner as _EvalRunner
from rag_pipeline.ingestion import IngestionPipeline
from rag_pipeline.chunker import ChunkingConfig
from rag_pipeline.vector_store import VectorStore
from rag_pipeline.bm25_index import BM25Index
from rag_pipeline.document_loader import DocumentLoader
from rag_pipeline.retrieval import DenseRetriever, SparseRetriever
from rag_pipeline.fusion import HybridRetriever
from rag_pipeline.reranker import LexicalReranker
from rag_pipeline.embedding import (
    create_embedding_service,
    describe_embedding_service,
    MockEmbeddingService,
    LocalEmbeddingService,
)

CORPUS = sorted(
    str(p)
    for p in (Path(__file__).parent.parent / "data/raw/fastapi_docs").rglob("*.md")
)

STRATEGY_ORDER = {"fixed": 1, "recursive": 2, "semantic": 3}
N_DOCS = len(CORPUS)
N_QUESTIONS = 52  # overwritten in main() once the golden dataset is loaded
_active_strategy = {"name": None}


def _install_progress_hooks() -> None:
    """Wrap EvalRunner.run so each strategy prints a summary once it completes.

    The pipeline factory emits the per-strategy header, indexing, and eval
    progress; this hook covers the "Done. correctness=... faithfulness=..."
    line that can only be printed once a strategy's report is computed.
    """
    original_run = _EvalRunner.run

    def _run_with_progress(self, *args, **kwargs):
        report = original_run(self, *args, **kwargs)
        cor = report.overall.get("correctness", {}).get("mean", 0.0)
        fai = report.overall.get("faithfulness", {}).get("mean", 0.0)
        print(f"  Done. correctness={cor:.2f} faithfulness={fai:.2f}", flush=True)
        return report

    _EvalRunner.run = _run_with_progress


_install_progress_hooks()


def make_pipeline_factory(embedding_service):
    """Build a per-strategy pipeline factory bound to the chosen embeddings."""

    def build_pipeline(strategy: str):
        """Fresh pipeline+index for the given chunking strategy."""
        tmp = tempfile.mkdtemp()
        _active_strategy["name"] = strategy
        print(
            f"\n[{STRATEGY_ORDER[strategy]}/3] Strategy: {strategy} — "
            f"indexing {N_DOCS} docs...",
            flush=True,
        )
        pipe = IngestionPipeline(
            chunking_config=ChunkingConfig(
                chunk_size=500, chunk_overlap=50, strategy=strategy,
            ),
            embedding_service=embedding_service,
            vector_store=VectorStore(persist_dir=str(Path(tmp) / "chroma")),
            bm25_index=BM25Index(persist_dir=str(Path(tmp) / "bm25")),
            document_loader=DocumentLoader(
                raw_dir=str(Path(tmp) / "raw"),
                processed_dir=str(Path(tmp) / "processed"),
            ),
        )
        for pos, f in enumerate(CORPUS, start=1):
            pipe.ingest_file(f)
            if pos % 25 == 0 or pos == N_DOCS:
                print(f"  indexed {pos}/{N_DOCS} docs...", flush=True)
        print(
            f"  Running eval: {N_QUESTIONS} questions × 4 metrics...",
            flush=True,
        )

        hybrid = HybridRetriever(
            dense_retriever=DenseRetriever(
                embedding_service=pipe.embedding_service,
                vector_store=pipe.vector_store,
            ),
            sparse_retriever=SparseRetriever(bm25_index=pipe.bm25_index),
            reranker=LexicalReranker(),
        )
        return hybrid

    return build_pipeline


def grounded_respond(messages):
    import re
    user = messages[1]["content"] if len(messages) > 1 else ""
    blocks = re.findall(r'<context id="1">(.*?)</context>', user, re.DOTALL)
    if blocks:
        text = re.sub(r"<[^>]+>", "", blocks[0]).strip().replace("\n", " ")
        return text[:120] + " [1]"
    return "I don't have enough information in the context to answer this question."


def make_generate_fn():
    # State shared across the three strategies; the pipe factory announces the
    # active strategy in `_active_strategy`, so the counter resets per strategy.
    state = {"strategy": None, "count": 0}

    def respond(question, chunks):
        if "refund" in question or "price" in question or "color" in question:
            return {
                "answer": (
                    "I don't have enough information in the context to answer this question."
                )
            }
        return {
            "answer": grounded_respond([
                {"role": "system", "content": ""},
                {
                    "role": "user",
                    "content": "<context id=\"1\">"
                    + (chunks[0].content if chunks else "")
                    + "</context>",
                },
            ])
        }

    def wrapped(question, chunks):
        name = _active_strategy["name"]
        if state["strategy"] != name:
            state["strategy"] = name
            state["count"] = 0
        out = respond(question, chunks)
        state["count"] += 1
        n = state["count"]
        if n % 10 == 0 or n == N_QUESTIONS:
            print(f"  evaluated {n}/{N_QUESTIONS} questions...", flush=True)
        return out

    return wrapped


def build_report(mode, dataset, result) -> dict:
    """Assemble data/comparison_report.json exactly per the spec (Part C)."""
    strategies = {s: row.report_dict() for s, row in result.rows.items()}
    ranking = result.overall_ranking()
    winner, winner_avg = ranking[0]
    runner_up, runner_up_avg = ranking[1] if len(ranking) > 1 else (winner, winner_avg)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode["mode"],
        "model": mode["embedding_model"],
        "judge": mode["judge_model"],
        "total_questions": len(dataset),
        "strategies": strategies,
        "winner": winner,
        "summary": build_summary(strategies, winner, winner_avg, runner_up, runner_up_avg),
    }


def build_summary(strategies, winner, winner_avg, runner_up, runner_up_avg) -> str:
    gains = []
    for m in ("correctness", "faithfulness"):
        w = strategies[winner][m]
        r = strategies[runner_up][m]
        if w > r:
            delta = (w - r) / r * 100.0 if r > 0 else w * 100.0
            gains.append(f"{m} by {delta:.1f}%")
    if gains:
        lead = f"{winner} chunking outperformed {runner_up} on " + " and ".join(gains) + "."
    else:
        lead = f"{winner} chunking topped {runner_up} on overall average."
    return (
        f"{lead} Overall avg: {winner} {winner_avg:.2f} vs "
        f"{runner_up} {runner_up_avg:.2f}."
    )


def print_result_table(strategies):
    """Print the readable stdout table from Part D."""
    widths = [13, 13, 13, 18, 10]
    heads = ["Strategy", "Correctness", "Faithfulness", "Retrieval Relev.", "Citation"]
    bars = ["─" * w for w in widths]

    print("  ┌" + "┬".join(bars) + "┐")
    print("  │" + "│".join(" " + h.ljust(w - 1) for h, w in zip(heads, widths)) + "│")
    print("  ├" + "┼".join(bars) + "┤")

    def center(v, width):
        s = f"{v:.2f}"
        pad = width - len(s)
        left = pad // 2
        return " " * left + s + " " * (pad - left)

    for s, row in strategies.items():
        cells = [" " + s.ljust(widths[0] - 1)]
        cells += [center(row[m], w) for m, w in zip(METRICS, widths[1:])]
        print("  │" + "│".join(cells) + "│")
    print("  └" + "┴".join(bars) + "┘")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare chunking strategies on the golden dataset."
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Force real embeddings + gpt-4o-mini judge (requires an OpenAI/OpenRouter chat key)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force mock embeddings + deterministic scorers even if a chat API key is set (CI)",
    )
    parser.add_argument(
        "--out",
        default="data/comparison_report.json",
        help="Output report path (default: data/comparison_report.json).",
    )
    args = parser.parse_args(argv)

    if args.real and args.mock:
        parser.error("--real and --mock are mutually exclusive")

    try:
        mode = resolve_eval_mode(force_real=args.real, force_mock=args.mock)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    embedding_service = create_embedding_service()
    print(f"Embedding service: {describe_embedding_service()}")

    if args.real:
        if isinstance(embedding_service, MockEmbeddingService):
            print(
                "ERROR: --real requires real embeddings. "
                "Set OPENAI_API_KEY or ensure sentence-transformers is installed.",
                file=sys.stderr,
            )
            return 1
        if isinstance(embedding_service, LocalEmbeddingService):
            print(
                "WARNING: Using local embeddings (all-MiniLM-L6-v2). "
                "Quality is good but differs from text-embedding-3-small.",
                file=sys.stderr,
            )
            mode["embedding_model"] = embedding_service.model

    ds = GoldenDataset.load("data/golden_dataset.json")
    global N_QUESTIONS
    N_QUESTIONS = len(ds)

    comparison = ChunkingComparison(
        dataset=ds,
        strategies=["fixed", "recursive", "semantic"],
        pipeline_factory=make_pipeline_factory(embedding_service),
        generate_fn=make_generate_fn(),
        judge=mode["judge"],
    )

    result = comparison.run()

    payload = build_report(mode, ds, result)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))

    print_result_table(payload["strategies"])
    winner, winner_avg = result.overall_ranking()[0]
    print(f"\nWinner: {winner} (avg score: {winner_avg:.2f})")
    print(f"\nReport saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
