"""boltztracer and OpenTelemetry, in both directions.

A trace here is already shaped the way OpenTelemetry shapes one: a 32-digit
trace id, a 16-digit id for each step and for its parent, start and end times
in nanoseconds, and a status. What differs is the wrapping. This module is the
two conversions, kept in one place so they stay inverses of each other:

    to_otlp(records)      our records  ->  an OTLP/HTTP JSON request
    from_otlp(payload)    an OTLP/HTTP JSON request  ->  our records

``to_otlp`` is what the ``endpoint`` setting sends. ``from_otlp`` is the way in
for an agent that is instrumented with OpenTelemetry instead of this package:
its spans become the same trace files, and everything that reads those files
works on them unchanged.

    python -m boltztracer.otel spans.json --out .boltz/traces

Names follow the OpenTelemetry GenAI semantic conventions:

    step kind   span name                 gen_ai.operation.name
    llm         chat <model>              chat
    tool        execute_tool <name>       execute_tool
    agent       invoke_agent <name>       invoke_agent
    step        <name>                    (none)

    model                gen_ai.request.model
    tokens read          gen_ai.usage.input_tokens (cached ones included)
    tokens written       gen_ai.usage.output_tokens
    read from cache      gen_ai.usage.cache_read.input_tokens
    written to cache     gen_ai.usage.cache_creation.input_tokens
    tool name            gen_ai.tool.name
    agent name           gen_ai.agent.name
    input, output        input.value, output.value (and their .mime_type)
    failure              status ERROR, error.type, and an "exception" event

What OpenTelemetry has no name for is kept under ``boltz.*``: the cost of a
call, and the task, model and attempt a run is compared by.

Reading is more forgiving than writing. Spans from OpenInference and from
OpenLLMetry are recognised by their own attribute names as well, and ids sent
as base64 (which some exporters do) are read as the hex they stand for.
"""

import argparse
import base64
import json
import sys

from ._sink import FileSink, span_name, to_otlp

__all__ = ["to_otlp", "from_otlp", "span_name", "write", "main"]

# gen_ai.operation.name, and the two other conventions in common use, to the
# four kinds of step.
_OPERATIONS = {
    "chat": "llm", "text_completion": "llm", "generate_content": "llm", "embeddings": "llm",
    "execute_tool": "tool",
    "invoke_agent": "agent", "create_agent": "agent",
}
_OPENINFERENCE = {
    "LLM": "llm", "EMBEDDING": "llm",
    "TOOL": "tool", "RETRIEVER": "tool", "RERANKER": "tool",
    "AGENT": "agent",
    "CHAIN": "step", "GUARDRAIL": "step", "EVALUATOR": "step",
}
_TRACELOOP = {"workflow": "agent", "agent": "agent", "task": "step", "tool": "tool"}


def _value(v):
    """An OTLP AnyValue as a plain Python value."""
    if not isinstance(v, dict):
        return v
    if "stringValue" in v:
        return v["stringValue"]
    if "intValue" in v:
        try:
            return int(v["intValue"])
        except (TypeError, ValueError):
            return None
    if "doubleValue" in v:
        return v["doubleValue"]
    if "boolValue" in v:
        return bool(v["boolValue"])
    if "arrayValue" in v:
        return [_value(x) for x in (v["arrayValue"] or {}).get("values", [])]
    if "kvlistValue" in v:
        return {kv.get("key"): _value(kv.get("value")) for kv in (v["kvlistValue"] or {}).get("values", [])}
    return None


def _attrs(items):
    return {a.get("key"): _value(a.get("value")) for a in items or [] if isinstance(a, dict)}


def _id(text, digits):
    """An id as lower-case hex, whether it was sent as hex or as base64."""
    s = str(text or "").strip()
    if len(s) == digits and all(c in "0123456789abcdefABCDEF" for c in s):
        return s.lower()
    try:
        raw = base64.b64decode(s, validate=True)
    except Exception:
        return s.lower()
    return raw.hex() if len(raw) * 2 == digits else s.lower()


def _parsed(text, mime=None):
    """A recorded value: JSON when it says it is, otherwise the text itself.

    With nothing said about it, text that is plainly an object or a list is
    read as one, and everything else is left as the text it is. "4" stays "4".
    """
    if not isinstance(text, str):
        return text
    t = text.strip()
    if mime == "application/json" or (mime is None and t[:1] in ("{", "[")):
        try:
            return json.loads(t)
        except ValueError:
            pass
    return text


def _side(a, side, *others):
    """What went in or came out: under the common name, or one of the others."""
    if a.get(side + ".value") is not None:
        return _parsed(a[side + ".value"], a.get(side + ".mime_type") or None)
    value = _first(a, *others)
    return None if value is None else _parsed(value)


def _int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _first(a, *keys):
    for k in keys:
        if a.get(k) not in (None, ""):
            return a[k]
    return None


def _kind(a, name, top):
    kind = a.get("boltz.span.kind")
    if kind in ("agent", "llm", "tool", "step"):
        return kind
    kind = (
        _OPERATIONS.get(str(a.get("gen_ai.operation.name") or ""))
        or _OPENINFERENCE.get(str(a.get("openinference.span.kind") or "").upper())
        or _TRACELOOP.get(str(a.get("traceloop.span.kind") or "").lower())
    )
    if kind:
        return kind
    head = name.split(" ", 1)[0]
    if head in _OPERATIONS:
        return _OPERATIONS[head]
    # Nothing says what it is. The top of a trace is the run; anything else
    # is a step, which is always a true thing to call it.
    return "agent" if top else "step"


def _record(span, top):
    a = _attrs(span.get("attributes"))
    name = str(span.get("name") or "")
    kind = _kind(a, name, top)

    # The name without the operation the conventions put in front of it.
    if kind == "tool":
        name = _first(a, "gen_ai.tool.name", "tool.name") or (name[13:] if name.startswith("execute_tool ") else name)
    elif kind == "agent":
        name = _first(a, "gen_ai.agent.name") or (name[13:] if name.startswith("invoke_agent ") else name)

    failed = (span.get("status") or {}).get("code") in (2, "STATUS_CODE_ERROR")
    rec = {
        "v": 1,
        "ev": "end",
        "trace_id": _id(span.get("traceId"), 32),
        "span_id": _id(span.get("spanId"), 16),
        "parent_id": _id(span.get("parentSpanId"), 16) if span.get("parentSpanId") else None,
        "name": str(name),
        "kind": kind,
        "start_ns": _int(span.get("startTimeUnixNano")),
        "end_ns": _int(span.get("endTimeUnixNano")),
        "status": "error" if failed else "ok",
    }

    labels, label_meta, meta, totals = {}, {}, {}, {}
    for key, value in a.items():
        if key.startswith("boltz.trace.meta."):
            label_meta[key[17:]] = value
        elif key.startswith("boltz.trace."):
            labels[key[12:]] = value
        elif key.startswith("boltz.meta."):
            meta[key[11:]] = value
        elif key.startswith("boltz.totals."):
            totals[key[13:]] = value
    if label_meta:
        labels["meta"] = label_meta
    if labels:
        rec["trace"] = labels

    value = _side(a, "input", "gen_ai.input.messages", "gen_ai.tool.call.arguments", "gen_ai.prompt", "traceloop.entity.input")
    if value is not None:
        rec["input"] = value
    value = _side(a, "output", "gen_ai.output.messages", "gen_ai.tool.call.result", "gen_ai.completion", "traceloop.entity.output")
    if value is not None:
        rec["output"] = value

    model = _first(a, "gen_ai.request.model", "gen_ai.response.model", "llm.model_name")
    if model:
        rec["model"] = str(model)

    read = _first(a, "gen_ai.usage.input_tokens", "gen_ai.usage.prompt_tokens", "llm.token_count.prompt")
    written = _first(a, "gen_ai.usage.output_tokens", "gen_ai.usage.completion_tokens", "llm.token_count.completion")
    if read is not None or written is not None:
        cached = _int(_first(a, "gen_ai.usage.cache_read.input_tokens", "gen_ai.usage.cache_read_input_tokens", "llm.token_count.prompt_details.cache_read"))
        kept = _int(_first(a, "gen_ai.usage.cache_creation.input_tokens", "gen_ai.usage.cache_creation_input_tokens", "llm.token_count.prompt_details.cache_write"))
        # The conventions count every token read in one number. Here the
        # ones that came from cache are kept apart, since they cost less.
        usage = {"input": max(_int(read) - cached - kept, 0), "output": _int(written)}
        if cached:
            usage["cached"] = cached
        if kept:
            usage["cache_write"] = kept
        rec["usage"] = usage
    if isinstance(a.get("boltz.cost.usd"), (int, float)):
        rec["cost"] = float(a["boltz.cost.usd"])
    if meta:
        rec["meta"] = meta
    if totals:
        rec["totals"] = totals

    if failed:
        err = {"type": str(a.get("error.type") or "Error"), "message": str((span.get("status") or {}).get("message") or "")}
        for event in span.get("events") or []:
            if event.get("name") == "exception":
                e = _attrs(event.get("attributes"))
                err["type"] = str(e.get("exception.type") or err["type"])
                err["message"] = str(e.get("exception.message") or err["message"])
                if e.get("exception.stacktrace"):
                    err["stack"] = str(e["exception.stacktrace"])
                break
        rec["error"] = err
    return rec


def _spans(payload):
    """Every span in a payload: one request, a list of them, or just spans."""
    if isinstance(payload, list):
        for item in payload:
            yield from _spans(item)
        return
    if not isinstance(payload, dict):
        return
    if "traceId" in payload and "spanId" in payload:
        yield payload
        return
    for resource in payload.get("resourceSpans") or payload.get("resource_spans") or []:
        for scope in resource.get("scopeSpans") or resource.get("scope_spans") or []:
            for span in scope.get("spans") or []:
                if isinstance(span, dict):
                    yield span


def from_otlp(payload):
    """Our records for the spans in an OTLP/HTTP JSON request.

    ``payload`` is an ``ExportTraceServiceRequest`` as a dict, or a list of
    them. Returns ``{trace_id: [records]}``, each list with parents before
    children, ready for :func:`write`.

    A trace that says nothing about what it was (spans from another tool carry
    no task or model labels) is given what can be seen in it: the name of its
    first step as the task, and the model its first model call used.
    """
    spans = [s for s in _spans(payload) if s.get("traceId") and s.get("spanId")]
    ids = {(_id(s["traceId"], 32), _id(s["spanId"], 16)) for s in spans}
    traces = {}
    for s in spans:
        trace_id = _id(s["traceId"], 32)
        parent = _id(s.get("parentSpanId"), 16) if s.get("parentSpanId") else None
        rec = _record(s, top=parent is None or (trace_id, parent) not in ids)
        traces.setdefault(trace_id, []).append(rec)

    for records in traces.values():
        records.sort(key=lambda r: (r["start_ns"], r["span_id"]))
        labels = next((r["trace"] for r in records if r.get("trace")), None)
        if labels is None:
            own = {r["span_id"] for r in records}
            top = next((r for r in records if r["parent_id"] not in own), records[0])
            labels = {"task": top["name"]}
            model = next((r["model"] for r in records if r.get("model")), None)
            if model:
                labels["model"] = model
        for r in records:
            r.setdefault("trace", labels)
    return traces


def write(traces, out_dir):
    """Write converted traces where the trace screens look for them."""
    sink = FileSink(str(out_dir))
    paths = []
    for trace_id, records in traces.items():
        for rec in records:
            sink.write(rec)
        paths.append(sink.file_for(trace_id))
    return paths


def _load(path):
    """A file of OTLP JSON: one request, or one request a line."""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    try:
        return [json.loads(text)]
    except ValueError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def main(argv=None, out=sys.stdout):
    ap = argparse.ArgumentParser(
        prog="python -m boltztracer.otel",
        description="Turn OpenTelemetry spans (OTLP/HTTP JSON) into boltztracer trace files.",
    )
    ap.add_argument("files", nargs="+", help="OTLP JSON files: one request each, or one request a line")
    ap.add_argument("--out", default=".boltz/traces", help="where to write the trace files")
    args = ap.parse_args(argv)
    payloads = []
    for path in args.files:
        try:
            payloads.extend(_load(path))
        except (OSError, ValueError) as exc:
            print("skipped %s: %s" % (path, exc), file=sys.stderr)
    paths = write(from_otlp(payloads), args.out)
    print("TRACES=%d" % len(paths), file=out)
    for p in paths:
        print(p, file=out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
