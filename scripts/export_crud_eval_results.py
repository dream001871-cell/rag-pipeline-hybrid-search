"""导出平台首轮运行与本机真实调用记录，分别统计回答率和执行状态。"""

from collections import Counter
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics
import sys

import requests


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_dir / "src"))
    from rag_pipeline.generation import parse_judge_json

    parser = argparse.ArgumentParser(description="导出独立评测轮次，不覆盖其他轮次。")
    parser.add_argument("--variant", choices=["first_run", "judge_fix", "zh_rerank"], default="first_run")
    args = parser.parse_args()
    runtime = project_dir / ".local" / "crud-rag-evaluation"
    dataset_dir = project_dir / "data" / "benchmarks" / "crud_rag_20"
    output_dir = dataset_dir / "evaluation" / "first_run"
    if args.variant != "first_run":
        runtime = project_dir / ".local" / "crud-rag-improvement" / args.variant
        output_dir = dataset_dir / "evaluation" / args.variant
    admission = json.loads((runtime / "run_admission.json").read_text(encoding="utf-8"))
    run_id = admission["run"]["id"]
    session = requests.Session()
    base = "http://127.0.0.1:8788/v1alpha1/runs/" + run_id
    response = session.get(base, timeout=20)
    response.raise_for_status()
    run = response.json()["run"]
    if run["status"] in ("RUN_STATUS_PENDING", "RUN_STATUS_RUNNING"):
        raise SystemExit("运行尚未结束，请在全部题目完成后导出。")
    results = []
    records_by_trial = []
    page_token = ""
    while True:
        response = session.get(base + "/results", params={"page_size": 200, "page_token": page_token}, timeout=20)
        response.raise_for_status()
        page = response.json()
        results.extend(page.get("results", []))
        records_by_trial.extend(page.get("recordsByTrial", []))
        page_token = page.get("nextPageToken", "")
        if not page_token:
            break

    dataset = json.loads((dataset_dir / "questions.json").read_text(encoding="utf-8"))
    questions = {q["id"]: q for q in dataset["questions"]}
    started = datetime.fromisoformat(run["createdAt"].replace("Z", "+00:00")).timestamp()
    finished = datetime.fromisoformat(run["finishedAt"].replace("Z", "+00:00")).timestamp()
    calls = {}
    for path in sorted((runtime / "responses").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if not started - 2 <= record["started_unix"] <= finished + 2:
            continue
        if record["id"] in calls:
            raise RuntimeError("同一题存在多次调用，需要先明确所评估的尝试。")
        calls[record["id"]] = record
    if set(calls) != set(questions):
        raise RuntimeError("实际调用记录与 20 道题未一一对应，请检查平台错误。")

    rows = []
    gate_counts = Counter()
    for record_id, question in questions.items():
        call = calls[record_id]
        assert call["question"] == question["question"]
        answer = call.get("response", {})
        events = call["model_calls"]
        generations = [event for event in events if event["kind"] == "generation"]
        contexts = generations[0]["context"] if generations else []
        retrieved_documents = set()
        for chunk in contexts:
            filename = chunk["metadata"].get("source_file") or ""
            retrieved_documents.add(Path(filename).name)
        expected = set(question["source_documents"])
        matched = sorted(expected & retrieved_documents)
        gate_failures = (answer.get("confidence") or {}).get("details", {}).get("gate_failures", [])
        gate_counts.update(gate_failures)
        details = (answer.get("confidence") or {}).get("details", {})
        citation_calls = []
        judge_calls = [event for event in events if event["kind"] == "judge"]
        for event in judge_calls:
            if event.get("judge_task") == "citation":
                citation_calls.append(event)
        if args.variant == "first_run":
            citation_calls = judge_calls[:-1]
        verdicts = {"supported": 0, "unsupported": 0, "unknown": 0}
        for event in citation_calls:
            raw = (event.get("answer") or "").strip()
            supported = None
            if raw.upper() in ("SUPPORTED", "UNSUPPORTED"):
                supported = raw.upper() == "SUPPORTED"
            else:
                try:
                    supported = parse_judge_json(raw).get("supported")
                except (ValueError, TypeError):
                    pass
            if supported is True:
                verdicts["supported"] += 1
            elif supported is False:
                verdicts["unsupported"] += 1
            else:
                verdicts["unknown"] += 1
        rows.append({
            "id": record_id,
            "category": question["category"],
            "question": question["question"],
            "reference": question["answer"],
            "can_answer": answer.get("can_answer", False),
            "answer": answer.get("answer"),
            "returned_output": call.get("output"),
            "draft_answer": generations[0]["answer"] if generations else None,
            "gate_failures": gate_failures,
            "citation_verdicts": verdicts,
            "citation_judge_errors": details.get("citation_judge_errors", []),
            "completeness_judge_error": details.get("completeness_judge_error"),
            "expected_documents": sorted(expected),
            "matched_documents": matched,
            "gold_document_coverage": len(matched) / len(expected),
            "context_chunks": len(contexts),
            "duration_seconds": call["duration_seconds"],
            "successful_model_calls": len(events),
            "error_type": call.get("error_type"),
        })

    frozen = json.loads((runtime / "source_lock.json").read_text(encoding="utf-8"))
    unchanged = all(hashlib.sha256((project_dir / name).read_bytes()).hexdigest() == checksum
                    for name, checksum in frozen.items())
    answered = sum(bool(row["can_answer"] and row["answer"]) for row in rows)
    summary = {
        "run_id": run_id,
        "variant": args.variant,
        "url": "http://127.0.0.1:8788/ui/#/runs/" + run_id,
        "platform_status": run["status"],
        "total_questions": len(rows),
        "answered": answered,
        "declined": sum(not row["can_answer"] and not row["error_type"] for row in rows),
        "execution_errors": sum(bool(row["error_type"]) for row in rows),
        "answer_rate": answered / len(rows),
        "mean_seconds": statistics.mean(row["duration_seconds"] for row in rows),
        "successful_model_calls": sum(row["successful_model_calls"] for row in rows),
        "mean_gold_document_coverage": statistics.mean(row["gold_document_coverage"] for row in rows),
        "questions_with_any_gold_document": sum(bool(row["matched_documents"]) for row in rows),
        "questions_with_all_gold_documents": sum(set(row["matched_documents"]) == set(row["expected_documents"]) for row in rows),
        "gate_failures": dict(gate_counts),
        "citation_verdicts": {key: sum(row["citation_verdicts"][key] for row in rows)
                              for key in ("supported", "unsupported", "unknown")},
        "questions_with_judge_errors": sum(bool(row["citation_judge_errors"] or row["completeness_judge_error"])
                                          for row in rows),
        "source_unchanged": unchanged,
        "score_note": "平台成功表示调用和评分完成；chrF 为字符相似度，不是语义正确率。回答率表示系统最终放行比例。",
        "rows": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, value in [("run.json", run), ("platform_results.json", {"results": results, "recordsByTrial": records_by_trial}),
                            ("summary.json", summary), ("source_lock.json", frozen)]:
        (output_dir / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "run_request.json").write_bytes((runtime / "run_request.json").read_bytes())
    (output_dir / "settings.json").write_bytes((runtime / "settings.json").read_bytes())
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, ensure_ascii=False, indent=2))
    print("平台结果条数", len(results), "保存的输出记录条数", len(records_by_trial))
