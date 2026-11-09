"""通过真实 HTTP 请求验证已启动的本机检索服务。"""

import json
from pathlib import Path
import time

import requests


if __name__ == "__main__":
    base_url = "http://127.0.0.1:8000"
    report = {"question": "What is FastAPI?", "checks": []}

    response = requests.get(f"{base_url}/healthz", timeout=10)
    response.raise_for_status()
    health = response.json()
    assert health["vector_chunks"] > 0, "向量索引为空"
    assert health["vector_chunks"] == health["bm25_chunks"], "双索引数量不一致"
    report["health"] = health

    response = requests.get(f"{base_url}/v1/documents", timeout=10)
    response.raise_for_status()
    report["documents"] = len(response.json())
    assert report["documents"] > 0, "没有已导入的文档"

    for mode in ("hybrid", "dense", "sparse"):
        started = time.perf_counter()
        response = requests.post(
            f"{base_url}/v1/retrieve",
            json={"question": report["question"], "mode": mode},
            timeout=120,
        )
        response.raise_for_status()
        ranking = response.json()["ranking"]
        assert len(ranking) == 20, f"{mode} 默认应返回 B 方案的20个片段"
        for chunk in ranking:
            assert chunk["content"] and chunk["source_file"], "缺少原文或来源"
            assert chunk["rerank_score"] is not None, "未返回重排分数"
        report["checks"].append({
            "mode": mode,
            "status": response.status_code,
            "seconds": round(time.perf_counter() - started, 3),
            "ranking": ranking,
        })

    response = requests.post(
        f"{base_url}/v1/retrieve", json={"question": "   "}, timeout=10
    )
    assert response.status_code == 400, "空查询应被拒绝"
    report["empty_query_status"] = response.status_code

    response = requests.get(f"{base_url}/docs", timeout=10)
    assert response.status_code == 200, "API 文档不可用"
    response = requests.get("http://127.0.0.1:8501/_stcore/health", timeout=10)
    assert response.status_code == 200, "Streamlit 服务不可用"
    report["dashboard_status"] = response.status_code

    output_path = Path(__file__).resolve().parent.parent / ".local" / "smoke-test.json"
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": "passed",
        "documents": report["documents"],
        "chunks": health["vector_chunks"],
        "modes": [check["mode"] for check in report["checks"]],
        "report": str(output_path),
    }, ensure_ascii=False))
