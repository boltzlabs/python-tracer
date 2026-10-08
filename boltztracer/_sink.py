"""Where finished steps go: a JSONL file, and optionally an OTLP endpoint.

The file is the primary sink. It needs no network, so it works inside a sandbox
with internet off, and every line is written the moment it exists, so a run
that is killed on a timeout still leaves behind everything up to the kill.

The HTTP sink speaks OTLP/HTTP JSON, the OpenTelemetry wire format, so the same
traces can be sent to any OTel-compatible tool. It runs on a background thread
and drops data rather than slow down or break the agent it is watching.
"""

import atexit
import json
import logging
import os
import queue
import threading
import time
import urllib.error
import urllib.request

__all__ = ["FileSink", "HttpSink", "to_otlp", "span_name"]

log = logging.getLogger("boltztracer")

VERSION = "0.2.0"


def _dumps(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=repr)


class FileSink:
    """Append one JSON line per record.

    ``path`` ending in ``.jsonl`` is one file for every trace; anything else is
    a directory holding ``<trace_id>.jsonl`` per trace.
    """

    def __init__(self, path):
        self.path = path
        self.per_trace = not path.endswith(".jsonl")
        self._lock = threading.Lock()
        self._made = set()

    def file_for(self, trace_id):
        return os.path.join(self.path, trace_id + ".jsonl") if self.per_trace else self.path

    def write(self, rec):
        path = self.file_for(rec["trace_id"])
        data = (_dumps(rec) + "\n").encode("utf-8")
        with self._lock:
            folder = os.path.dirname(path)
            if folder and folder not in self._made:
                os.makedirs(folder, exist_ok=True)
                self._made.add(folder)
            # One O_APPEND write per line: a sub-agent in another process can
            # append to the same file without the two lines interleaving.
            # 0600 because a trace holds prompts, code and tool output.
            fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            try:
                os.write(fd, data)
            finally:
                os.close(fd)

    def flush(self, timeout=None):
        return True


class HttpSink:
    """Batch finished spans to an OTLP/HTTP JSON endpoint in the background."""

    BATCH = 100
    TIMEOUT = 10.0

    def __init__(self, endpoint, headers=None, service="boltztracer"):
        self.endpoint = endpoint
        self.headers = {"Content-Type": "application/json", "User-Agent": "boltztracer-python/" + VERSION}
        self.headers.update(headers or {})
        self.service = service
        self.dropped = 0
        self._q = queue.Queue(maxsize=10000)
        self._thread = None
        self._pid = None
        self._warned = False
        self._lock = threading.Lock()
        atexit.register(self.flush, 5.0)

    def write(self, rec):
        # OTLP has no notion of a span that has started but not ended.
        if rec.get("ev") != "end":
            return
        try:
            self._q.put_nowait(rec)
        except queue.Full:
            self.dropped += 1
            return
        self._ensure_thread()

    def _ensure_thread(self):
        # A forked child inherits the queue but not the thread.
        if self._thread is not None and self._thread.is_alive() and self._pid == os.getpid():
            return
        with self._lock:
            if self._thread is None or not self._thread.is_alive() or self._pid != os.getpid():
                self._pid = os.getpid()
                self._thread = threading.Thread(target=self._run, name="boltztracer", daemon=True)
                self._thread.start()

    def _run(self):
        while True:
            batch = [self._q.get()]
            while len(batch) < self.BATCH:
                try:
                    batch.append(self._q.get_nowait())
                except queue.Empty:
                    break
            try:
                self._send(batch)
            except Exception as exc:  # never let the exporter thread die
                self._warn(exc)
            finally:
                for _ in batch:
                    self._q.task_done()

    def _send(self, batch):
        body = _dumps(to_otlp(batch, self.service)).encode("utf-8")
        req = urllib.request.Request(self.endpoint, data=body, headers=self.headers, method="POST")
        last = None
        for attempt in range(2):
            try:
                with urllib.request.urlopen(req, timeout=self.TIMEOUT) as resp:
                    resp.read()
                return
            except urllib.error.HTTPError as exc:
                last = exc
                # The request itself is wrong; sending it again cannot help.
                if exc.code < 500 and exc.code != 429:
                    break
            except Exception as exc:
                last = exc
            if attempt == 0:
                time.sleep(0.5)
        self.dropped += len(batch)
        self._warn(last)

    def _warn(self, exc):
        if not self._warned:
            self._warned = True
            log.warning("boltztracer: could not send traces to %s (%s); dropping", self.endpoint, exc)

    def flush(self, timeout=5.0):
        """Wait until everything queued has been sent. False if it timed out."""
        deadline = time.monotonic() + (timeout if timeout is not None else 5.0)
        done = self._q.all_tasks_done
        with done:
            while self._q.unfinished_tasks:
                left = deadline - time.monotonic()
                if left <= 0:
                    return False
                done.wait(left)
        return True


# -- OTLP encoding ------------------------------------------------------------

# What each kind of step is called in the OpenTelemetry GenAI conventions, so a
# tool that already understands those conventions draws our spans correctly.
_OPERATION = {"llm": "chat", "tool": "execute_tool", "agent": "invoke_agent"}


def _attr(key, value):
    if isinstance(value, bool):
        v = {"boolValue": value}
    elif isinstance(value, int):
        v = {"intValue": str(value)}
    elif isinstance(value, float):
        v = {"doubleValue": value}
    elif isinstance(value, str):
        v = {"stringValue": value}
    elif isinstance(value, (list, tuple)) and all(isinstance(x, str) for x in value):
        v = {"arrayValue": {"values": [{"stringValue": x} for x in value]}}
    else:
        v = {"stringValue": _dumps(value)}
    return {"key": key, "value": v}


def span_name(rec):
    """A step's name the way the GenAI conventions spell a span's.

    They name a span by its operation and then what it was done to:
    ``execute_tool search``, ``invoke_agent reviewer``, ``chat my-model``. A
    model call is already named that way here; a tool or an agent is known by
    its own name in the trace file, and gets the operation in front on export.
    """
    op = _OPERATION.get(rec["kind"])
    name = rec["name"]
    if op and rec["kind"] in ("tool", "agent") and not name.startswith(op + " "):
        return op + " " + name
    return name


def _span(rec):
    attrs = [_attr("boltz.span.kind", rec["kind"])]
    op = _OPERATION.get(rec["kind"])
    if op:
        attrs.append(_attr("gen_ai.operation.name", op))
    if rec["kind"] == "tool":
        attrs.append(_attr("gen_ai.tool.name", rec["name"]))
    if rec["kind"] == "agent":
        attrs.append(_attr("gen_ai.agent.name", rec["name"]))

    for key, value in (rec.get("trace") or {}).items():
        if key == "meta":
            for k, v in value.items():
                attrs.append(_attr("boltz.trace.meta." + k, v))
        else:
            attrs.append(_attr("boltz.trace." + key, value))

    if rec.get("model"):
        attrs.append(_attr("gen_ai.request.model", rec["model"]))
    usage = rec.get("usage") or {}
    if usage:
        cached = usage.get("cached", 0)
        written = usage.get("cache_write", 0)
        # The convention counts every input token here, cached or not.
        attrs.append(_attr("gen_ai.usage.input_tokens", usage.get("input", 0) + cached + written))
        attrs.append(_attr("gen_ai.usage.output_tokens", usage.get("output", 0)))
        if cached:
            attrs.append(_attr("gen_ai.usage.cache_read.input_tokens", cached))
        if written:
            attrs.append(_attr("gen_ai.usage.cache_creation.input_tokens", written))
    if rec.get("cost") is not None:
        attrs.append(_attr("boltz.cost.usd", float(rec["cost"])))

    # `input.value` / `output.value` are the names other tracing tools read,
    # with the kind of text beside each: without it, the text "4" and the
    # number 4 are the same thing on the wire and cannot be told apart again.
    for side in ("input", "output"):
        if side in rec:
            text = isinstance(rec[side], str)
            attrs.append(_attr(side + ".value", rec[side] if text else _dumps(rec[side])))
            attrs.append(_attr(side + ".mime_type", "text/plain" if text else "application/json"))
    for k, v in (rec.get("meta") or {}).items():
        attrs.append(_attr("boltz.meta." + k, v))
    for k, v in (rec.get("totals") or {}).items():
        if v is not None:
            attrs.append(_attr("boltz.totals." + k, v))

    span = {
        "traceId": rec["trace_id"],
        "spanId": rec["span_id"],
        "name": span_name(rec),
        "kind": 3 if rec["kind"] == "llm" else 1,  # CLIENT for a model call, else INTERNAL
        "startTimeUnixNano": str(rec["start_ns"]),
        "endTimeUnixNano": str(rec["end_ns"]),
        "attributes": attrs,
        "status": {"code": 2 if rec.get("status") == "error" else 1},
    }
    if rec.get("parent_id"):
        span["parentSpanId"] = rec["parent_id"]
    err = rec.get("error")
    if err:
        # The conventions put the class of failure on the span itself as well
        # as in the exception event, so it can be counted without the event.
        attrs.append(_attr("error.type", err.get("type") or "_OTHER"))
        span["status"]["message"] = err.get("message", "")
        span["events"] = [
            {
                "name": "exception",
                "timeUnixNano": str(rec["end_ns"]),
                "attributes": [
                    _attr("exception.type", err.get("type", "")),
                    _attr("exception.message", err.get("message", "")),
                    _attr("exception.stacktrace", err.get("stack", "")),
                ],
            }
        ]
    return span


def to_otlp(records, service="boltztracer"):
    """An OTLP/HTTP JSON ``ExportTraceServiceRequest`` for finished records."""
    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        _attr("service.name", service),
                        _attr("telemetry.sdk.name", "boltztracer"),
                        _attr("telemetry.sdk.language", "python"),
                        _attr("telemetry.sdk.version", VERSION),
                    ]
                },
                "scopeSpans": [
                    {
                        "scope": {"name": "boltztracer", "version": VERSION},
                        "spans": [_span(r) for r in records],
                    }
                ],
            }
        ]
    }
