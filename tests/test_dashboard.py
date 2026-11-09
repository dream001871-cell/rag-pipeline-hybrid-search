import py_compile
from pathlib import Path

import query_dashboard as qd

DASH = Path(__file__).parent.parent / "dashboards" / "query_dashboard.py"


def test_dashboard_compiles():
    py_compile.compile(str(DASH), doraise=True)


def test_modes_expose_backend_labels():
    assert qd.MODES["混合检索（向量 + 关键词）"] == "hybrid"
    assert qd.MODES["仅向量检索"] == "dense"
    assert qd.MODES["仅关键词检索（BM25）"] == "sparse"


def test_fmt_handles_none_and_floats():
    assert qd.fmt(None) == "—"
    assert qd.fmt(0.1234) == "0.1234"


def test_api_post_handles_unreachable_backend():
    body, code, detail = qd.api_post(
        "http://127.0.0.1:59999", "/v1/retrieve",
        {"question": "x", "mode": "hybrid", "top_k": 3}, timeout=1,
    )
    assert body is None
    assert detail in (None, "") or detail


def test_api_get_returns_none_when_backend_down():
    resp = qd.api_get("http://127.0.0.1:59999", "/healthz", timeout=1)
    assert resp is None
