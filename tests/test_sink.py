"""The OTLP sender, against a real HTTP server on this machine."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import boltztracer as bt
from boltztracer import _core


@pytest.fixture
def collector():
    """A server that keeps every request it is sent."""
    got = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            got.append((self.path, dict(self.headers), json.loads(body)))
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/v1/traces", got
    server.shutdown()
    server.server_close()


def attrs(span):
    out = {}
    for a in span["attributes"]:
        (value,) = a["value"].values()
        out[a["key"]] = value
    return out


def test_finished_steps_arrive_as_otlp(collector, tmp_path):
    url, got = collector
    bt.init(endpoint=url, api_key="k-123", project="pager", prices={"m1": {"input": 1.0, "output": 1.0}})
    with pytest.raises(KeyError):
        with bt.trace(task="fix-bug", model="m1", attempt=1) as root:
            with bt.llm(model="m1", input=[{"role": "user", "content": "hi"}]) as call:
                call.set(output="hello").usage(input=10, output=5, cached=90)
            with bt.step("lookup", kind="tool", input={"id": 7}):
                raise KeyError("missing id 7")
    assert bt.flush() is True

    path, headers, _ = got[0]
    assert path == "/v1/traces" and headers["Authorization"] == "Bearer k-123"
    assert headers["Content-Type"] == "application/json"

    spans = {}
    for _, _, body in got:
        (resource,) = body["resourceSpans"]
        assert {"key": "service.name", "value": {"stringValue": "pager"}} in resource["resource"]["attributes"]
        for s in resource["scopeSpans"][0]["spans"]:
            spans[s["name"]] = s
    assert set(spans) == {"fix-bug", "chat m1", "lookup"}  # only finished steps are sent

    llm, tool, top = spans["chat m1"], spans["lookup"], spans["fix-bug"]
    assert llm["traceId"] == root.trace_id and len(llm["traceId"]) == 32 and len(llm["spanId"]) == 16
    assert llm["parentSpanId"] == top["spanId"] and "parentSpanId" not in top
    assert int(llm["endTimeUnixNano"]) >= int(llm["startTimeUnixNano"])
    a = attrs(llm)
    assert a["boltz.span.kind"] == "llm" and a["gen_ai.operation.name"] == "chat"
    assert a["gen_ai.request.model"] == "m1"
    assert a["gen_ai.usage.input_tokens"] == "100" and a["gen_ai.usage.output_tokens"] == "5"
    assert a["gen_ai.usage.cache_read.input_tokens"] == "90"
    assert a["boltz.trace.task"] == "fix-bug" and a["boltz.trace.attempt"] == "1"
    assert json.loads(a["input.value"]) == [{"role": "user", "content": "hi"}] and a["output.value"] == "hello"
    assert a["boltz.cost.usd"] == pytest.approx(105 / 1e6)

    assert attrs(tool)["gen_ai.tool.name"] == "lookup"
    assert tool["status"]["code"] == 2 and "missing id 7" in tool["status"]["message"]
    exception = attrs(tool["events"][0])
    assert exception["exception.type"] == "KeyError" and "KeyError" in exception["exception.stacktrace"]
    assert attrs(top)["boltz.totals.llm_calls"] == "1"
    # An endpoint alone means no file is written.
    assert not (tmp_path / ".boltz").exists()


def test_an_unreachable_endpoint_costs_the_agent_nothing(monkeypatch):
    monkeypatch.setattr("boltztracer._sink.time.sleep", lambda s: None)
    bt.init(endpoint="http://127.0.0.1:9/v1/traces")  # nothing listens on the discard port
    with bt.trace(task="t"):
        with bt.step("s"):
            pass
    assert bt.flush(10) is True
    (sink,) = _core._cfg.sinks
    assert sink.dropped == 2


def test_the_boltz_key_only_goes_to_boltz(monkeypatch):
    monkeypatch.setenv("BOLTZLABS_API_KEY", "boltz-secret")

    bt.init(endpoint="https://collector.example.com/v1/traces")
    assert "Authorization" not in _core._cfg.sinks[0].headers

    bt.init(endpoint="https://boltzlabs.cloud/api/otel/v1/traces")
    assert _core._cfg.sinks[0].headers["Authorization"] == "Bearer boltz-secret"

    bt.init(endpoint="https://collector.example.com/v1/traces", headers={"x-api-key": "theirs"})
    assert _core._cfg.sinks[0].headers["x-api-key"] == "theirs"
