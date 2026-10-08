"""boltztracer and OpenTelemetry: out, back in, and spans from other tools."""

import base64
import io
import json

import pytest

import boltztracer as bt
from boltztracer import _core, otel
from boltztracer.__main__ import main as view


def record_a_run(out):
    """A run with every kind of step, a failure and a cached model call."""
    t = out(prices={"m1": {"input": 2.0, "output": 8.0, "cached": 0.2}})
    with pytest.raises(KeyError):
        with bt.trace(task="fix-bug", model="m1", attempt=2, tags=["nightly"], owner="ci"):
            with bt.llm(model="m1", input={"messages": [{"role": "user", "content": "hi"}]}, temperature=0) as call:
                call.set(output={"role": "assistant", "content": "hello"}).usage(input=10, output=5, cached=90, cache_write=4)
            with bt.step("reviewer", kind="agent", input="check it"):
                # Text that looks like something else must come back as text.
                with bt.step("plan", input="4") as plan:
                    plan.set(output='{"not": "an object, a string"}')
            with bt.step("lookup", kind="tool", input={"id": 7}):
                raise KeyError("missing id 7")
    return t.ended()


def test_a_trace_survives_the_trip_to_otlp_and_back(out):
    ours = record_a_run(out)
    (theirs,) = otel.from_otlp(otel.to_otlp(ours)).values()
    assert len(theirs) == len(ours) == 5
    back = {r["span_id"]: r for r in theirs}
    for want in ours:
        got = back[want["span_id"]]
        for key in ("trace_id", "parent_id", "name", "kind", "start_ns", "end_ns", "status", "trace"):
            assert got[key] == want[key], (want["name"], key)
        for key in ("input", "output", "model", "usage", "cost", "meta", "totals"):
            assert got.get(key) == want.get(key), (want["name"], key)
        if "error" in want:
            # A failure that only passed through a step keeps its kind and
            # message; where it came from is ours alone and does not travel.
            assert got["error"]["type"] == want["error"]["type"] and got["error"]["message"] == want["error"]["message"]
            assert got["error"].get("stack") == want["error"].get("stack")
    # In the order they happened, which is the order a reader wants.
    assert [r["start_ns"] for r in theirs] == sorted(r["start_ns"] for r in theirs)


def test_spans_are_named_as_the_conventions_name_them():
    assert otel.span_name({"kind": "tool", "name": "search"}) == "execute_tool search"
    assert otel.span_name({"kind": "agent", "name": "reviewer"}) == "invoke_agent reviewer"
    assert otel.span_name({"kind": "llm", "name": "chat m1"}) == "chat m1"
    assert otel.span_name({"kind": "step", "name": "plan"}) == "plan"
    # Already spelled that way: not spelled twice.
    assert otel.span_name({"kind": "tool", "name": "execute_tool search"}) == "execute_tool search"


def attr(key, value):
    kind = "intValue" if isinstance(value, int) else "stringValue"
    return {"key": key, "value": {kind: str(value) if isinstance(value, int) else value}}


def span(trace, sid, parent, name, start, end, attrs, **more):
    s = {"traceId": trace, "spanId": sid, "name": name, "startTimeUnixNano": str(start), "endTimeUnixNano": str(end),
         "attributes": [attr(k, v) for k, v in attrs.items()], "status": {"code": 1}}
    if parent:
        s["parentSpanId"] = parent
    s.update(more)
    return s


TRACE = "0af7651916cd43dd8448eb211c80319c"


def someone_elses_trace():
    """What an agent instrumented with an OpenTelemetry SDK sends: the GenAI
    conventions and nothing of ours."""
    spans = [
        span(TRACE, "00000000000000a1", None, "invoke_agent planner", 1000, 9000, {"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "planner"}),
        span(TRACE, "00000000000000a2", "00000000000000a1", "chat gpt-x", 1100, 3000, {
            "gen_ai.operation.name": "chat", "gen_ai.request.model": "gpt-x",
            "gen_ai.usage.input_tokens": 1000, "gen_ai.usage.output_tokens": 40, "gen_ai.usage.cache_read.input_tokens": 800,
            "gen_ai.input.messages": json.dumps([{"role": "user", "content": "plan it"}]),
            "gen_ai.output.messages": json.dumps([{"role": "assistant", "content": "ok"}]),
        }),
        span(TRACE, "00000000000000a3", "00000000000000a1", "execute_tool search", 3100, 4000,
             {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "search", "gen_ai.tool.call.arguments": '{"q":"cats"}', "error.type": "TimeoutError"},
             status={"code": 2, "message": "took too long"},
             events=[{"name": "exception", "attributes": [attr("exception.type", "TimeoutError"), attr("exception.message", "took too long"), attr("exception.stacktrace", "Traceback ...")]}]),
        # Nothing says what this one is.
        span(TRACE, "00000000000000a4", "00000000000000a1", "tidy up", 4100, 4200, {}),
    ]
    return {"resourceSpans": [{"resource": {"attributes": [attr("service.name", "planner")]}, "scopeSpans": [{"scope": {"name": "other-sdk"}, "spans": spans}]}]}


def test_spans_from_an_opentelemetry_sdk_become_a_trace():
    (records,) = otel.from_otlp(someone_elses_trace()).values()
    top, llm, tool, plain = records
    assert (top["kind"], top["name"], top["parent_id"]) == ("agent", "planner", None)
    assert (llm["kind"], llm["name"], llm["model"]) == ("llm", "chat gpt-x", "gpt-x")
    # 1000 read in all, 800 of them from cache.
    assert llm["usage"] == {"input": 200, "output": 40, "cached": 800}
    assert llm["input"] == [{"role": "user", "content": "plan it"}] and llm["output"] == [{"role": "assistant", "content": "ok"}]
    assert (tool["kind"], tool["name"], tool["status"]) == ("tool", "search", "error")
    assert tool["input"] == {"q": "cats"}
    assert tool["error"] == {"type": "TimeoutError", "message": "took too long", "stack": "Traceback ..."}
    assert (plain["kind"], plain["name"]) == ("step", "tidy up")
    # It named no task and no model, so it is known by what can be seen in it.
    assert all(r["trace"] == {"task": "planner", "model": "gpt-x"} for r in records)
    assert all(r["ev"] == "end" and r["trace_id"] == TRACE for r in records)


def test_other_dialects_and_encodings_are_read_too():
    b64 = lambda hexes: base64.b64encode(bytes.fromhex(hexes)).decode()
    payload = [
        # OpenInference names, ids as base64, two requests in a list.
        {"resourceSpans": [{"scopeSpans": [{"spans": [
            span(b64(TRACE), b64("00000000000000b1"), None, "agent", 10, 90, {"openinference.span.kind": "AGENT"}),
            span(b64(TRACE), b64("00000000000000b2"), b64("00000000000000b1"), "ChatModel", 20, 60, {
                "openinference.span.kind": "LLM", "llm.model_name": "claude-x", "llm.token_count.prompt": 50, "llm.token_count.completion": 7,
                "input.value": "what is 2+2", "output.value": "4"}),
        ]}]}]},
        {"resourceSpans": [{"scopeSpans": [{"spans": [
            span(b64(TRACE), b64("00000000000000b3"), b64("00000000000000b1"), "lookup", 60, 80, {"traceloop.span.kind": "tool", "traceloop.entity.input": '{"id":1}'}),
        ]}]}]},
    ]
    (records,) = otel.from_otlp(payload).values()
    assert [r["span_id"] for r in records] == ["00000000000000b1", "00000000000000b2", "00000000000000b3"]
    assert [r["kind"] for r in records] == ["agent", "llm", "tool"]
    assert records[1]["parent_id"] == "00000000000000b1" and records[1]["model"] == "claude-x"
    assert records[1]["usage"] == {"input": 50, "output": 7}
    assert records[1]["input"] == "what is 2+2" and records[1]["output"] == "4" and records[2]["input"] == {"id": 1}
    # Garbage is not a trace, and is not an error either.
    assert otel.from_otlp({"resourceSpans": [{"scopeSpans": [{"spans": [{"name": "no ids"}, "nope"]}]}]}) == {}
    assert otel.from_otlp(None) == {} and otel.from_otlp([]) == {}


def test_spans_in_a_file_become_trace_files_the_viewer_reads(tmp_path):
    one = tmp_path / "request.json"
    one.write_text(json.dumps(someone_elses_trace()))
    # A collector's file exporter writes a request a line.
    lines = tmp_path / "requests.jsonl"
    lines.write_text(json.dumps(someone_elses_trace()) + "\n\n" + json.dumps({"resourceSpans": []}) + "\n")
    for src in (one, lines):
        dest = tmp_path / ("out-" + src.stem)
        buf = io.StringIO()
        assert otel.main([str(src), "--out", str(dest)], out=buf) == 0
        assert buf.getvalue().splitlines()[0] == "TRACES=1"
        shown = io.StringIO()
        view([str(dest)], out=shown)
        text = shown.getvalue()
        assert "planner" in text and "gpt-x" in text and "error" in text and "search" in text
    assert otel.main([str(tmp_path / "missing.json"), "--out", str(tmp_path / "none")], out=io.StringIO()) == 0


def test_a_run_joins_the_trace_it_was_started_from(out, monkeypatch, tmp_path):
    parent_trace, parent_span = "4bf92f3577b34da6a3ce929d0e0e4736", "00f067aa0ba902b7"
    monkeypatch.setenv("TRACEPARENT", f"00-{parent_trace}-{parent_span}-01")
    t = out()
    with bt.trace(task="child") as root:
        assert root.trace_id == parent_trace and root.parent_id == parent_span
        with bt.step("work") as work:
            # And hands itself on in the same form.
            assert bt.traceparent() == f"00-{parent_trace}-{work.span_id}-01"
    top = t.one("child")
    assert top["parent_id"] == parent_span and t.one("work")["parent_id"] == top["span_id"]
    # It is still the top of its own run: it carries the totals and ends it.
    assert top["totals"]["spans"] == 2 and not _core._open
    # Only the first run adopts the parent; the next is a trace of its own.
    with bt.trace(task="next") as other:
        assert other.trace_id != parent_trace and other.parent_id is None
    # The viewer finds the top although it has a parent outside the file.
    shown = io.StringIO()
    view([t.path], out=shown)
    assert "child" in shown.getvalue() and "unfinished" not in shown.getvalue().splitlines()[1]
    assert bt.traceparent() is None


@pytest.mark.parametrize("header", [
    "", "garbage", "00-short-00f067aa0ba902b7-01",
    "00-00000000000000000000000000000000-00f067aa0ba902b7-01",  # no trace
    "00-4bf92f3577b34da6a3ce929d0e0e4736-0000000000000000-01",  # no parent
    "ff-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",  # a version not to be read
    "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01-extra",
])
def test_a_traceparent_that_is_not_one_is_ignored(out, monkeypatch, header):
    monkeypatch.setenv("TRACEPARENT", header)
    out()
    with bt.trace(task="t") as root:
        assert root.parent_id is None and len(root.trace_id) == 32 and root.trace_id != "4bf92f3577b34da6a3ce929d0e0e4736"


def test_an_id_given_outright_wins_over_a_traceparent(out, monkeypatch):
    monkeypatch.setenv("TRACEPARENT", "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01")
    monkeypatch.setenv("BOLTZ_TRACE_ID", "a" * 32)
    out()
    with bt.trace(task="t") as root:
        assert root.trace_id == "a" * 32 and root.parent_id is None
