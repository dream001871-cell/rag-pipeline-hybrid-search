"""将 RAG 各阶段的执行时间和状态发往本地测试中台。"""

from contextlib import contextmanager
import json
import logging
import time
from uuid import uuid4

import requests


logger = logging.getLogger(__name__)


class AnswerTrace:
    def __init__(self, service: str, labels: dict | None = None):
        self.trace_id = uuid4().hex
        self.root_span_id = uuid4().hex[:16]
        self.service = service
        self.labels = labels or {}
        self.started = time.time_ns()
        self.spans = []

    @contextmanager
    def step(self, name: str):
        started = time.time_ns()
        attributes = {}
        try:
            yield attributes
        except Exception as exc:
            attributes["error.type"] = type(exc).__name__
            raise
        finally:
            self.spans.append(self._span(name, uuid4().hex[:16], started, attributes, self.root_span_id))

    def finish(self, attributes: dict):
        root = self._span("rag.ask", self.root_span_id, self.started, attributes)
        resource = {"service.name": self.service}
        resource.update({f"evalsi.label.{key}": value for key, value in self.labels.items()})
        return {"resourceSpans": [{
            "resource": {"attributes": self._attributes(resource)},
            "scopeSpans": [{"scope": {"name": "rag_pipeline"}, "spans": [root, *self.spans]}],
        }]}

    def trajectory(self):
        steps = []
        for span in self.spans:
            steps.append({
                "name": span["name"], "type": "llm" if span["name"] == "rag.generate" else "tool",
                "span_id": span["spanId"],
                "duration_ms": (int(span["endTimeUnixNano"]) - int(span["startTimeUnixNano"])) / 1e6,
                "attributes": {"trace_id": self.trace_id},
            })
        return {"session_id": self.trace_id, "steps": steps}

    def _span(self, name, span_id, started, attributes, parent_id=None):
        span = {
            "traceId": self.trace_id, "spanId": span_id, "name": name, "kind": 1,
            "startTimeUnixNano": str(started), "endTimeUnixNano": str(time.time_ns()),
            "attributes": self._attributes(attributes),
        }
        if parent_id:
            span["parentSpanId"] = parent_id
        if "error.type" in attributes:
            span["status"] = {"code": 2}
        return span

    @staticmethod
    def _attributes(values):
        attributes = []
        for key, value in values.items():
            text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            attributes.append({"key": key, "value": {"stringValue": text}})
        return attributes


def export_trace(endpoint: str, trace_id: str, payload: dict) -> bool:
    """限制等待时间，让追踪故障独立于问答结果。"""
    try:
        with requests.Session() as session:
            session.trust_env = False
            response = session.post(endpoint, json=payload, timeout=(2, 3), allow_redirects=False,
                                    headers={"Idempotency-Key": f"rag-trace-{trace_id}"})
            if response.status_code != 200:
                raise ValueError(f"中台返回 HTTP {response.status_code}")
            result = response.json()
            if not isinstance(result, dict) or int(result.get("partialSuccess", {}).get("rejectedSpans", 0)):
                raise ValueError("中台拒收部分事件")
        return True
    except (requests.RequestException, ValueError, TypeError, AttributeError) as exc:
        logger.warning("RAG 追踪发送失败 trace_id=%s error=%s", trace_id, type(exc).__name__)
        return False
