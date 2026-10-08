"""The tracer itself: steps, the tree they form, and the ways to open one.

A *trace* is one run of one model on one task. It is a tree of *steps*, and
every step has a kind the viewer knows how to draw:

    agent   the run itself, or a sub-agent inside it
    llm     one call to a model: prompt, answer, tokens, cost
    tool    one tool the agent used
    step    anything else worth a box of its own

The tree is built from where the code is, not from anything the caller passes
around: a step opened while another is open becomes its child.

``trace()`` starts a run explicitly. A step with nothing open around it joins
the run that is open, and if none ever was, the whole process is one run. That
last rule is what makes an agent that only wraps its client and decorates its
tools produce one tree, instead of one single-step trace per call.

Nothing here is allowed to break the agent being traced. A failure to record is
logged at debug level and swallowed; an exception from the agent's own code is
recorded and then re-raised untouched.
"""

import atexit
import contextvars
import functools
import inspect
import logging
import os
import threading
import time
import traceback
from urllib.parse import urlsplit

from ._sink import FileSink, HttpSink
from ._values import clean

__all__ = [
    "Span",
    "init",
    "trace",
    "step",
    "llm",
    "observe",
    "tool",
    "agent",
    "price",
    "current",
    "flush",
    "traceparent",
]

log = logging.getLogger("boltztracer")

# Inside the workspace's `.boltz/`, which the eval recorder already leaves out
# of the code diff, so trace files never show up as changes the agent made.
DEFAULT_DIR = os.path.join(".boltz", "traces")

_current = contextvars.ContextVar("boltztracer_span", default=None)
# Re-entrant: opening the process-wide run happens while it is held.
_lock = threading.RLock()
_cfg = None
_totals = {}
# Runs started with trace() that have not ended yet.
_open = []

# Yielded items remembered as a generator step's output.
_KEEP = 200


def _flag(explicit, env_value, default):
    if explicit is not None:
        return bool(explicit)
    if env_value is not None:
        return env_value.strip().lower() not in ("", "0", "false", "no", "off")
    return default


def _labels(**kw):
    out = {k: v for k, v in kw.items() if v not in (None, "", [], {})}
    if "attempt" in out:
        try:
            out["attempt"] = int(out["attempt"])
        except (TypeError, ValueError):
            pass
    if "tags" in out:
        tags = out["tags"]
        out["tags"] = [tags] if isinstance(tags, str) else [str(t) for t in tags]
    return out


def _is_boltz(endpoint):
    """Is this endpoint ours? Decides whether it may be sent the Boltz key."""
    host = urlsplit(endpoint).hostname or ""
    own = urlsplit(os.environ.get("BOLTZLABS_API_URL") or "https://boltzlabs.cloud").hostname or ""
    return host == own or host == "boltzlabs.cloud" or host.endswith(".boltzlabs.cloud")


class _Config:
    def __init__(
        self,
        file=None,
        endpoint=None,
        headers=None,
        api_key=None,
        project=None,
        task=None,
        model=None,
        attempt=None,
        tags=None,
        prices=None,
        capture=None,
        mask=None,
        redact=None,
        max_chars=None,
        enabled=None,
    ):
        env = os.environ.get
        self.enabled = _flag(enabled, env("BOLTZ_TRACE"), True)
        self.capture = _flag(capture, env("BOLTZ_TRACE_CAPTURE"), True)
        self.redact = True if redact is None else bool(redact)
        self.mask = mask
        try:
            self.max_chars = int(max_chars or env("BOLTZ_TRACE_MAX_CHARS") or 16000)
        except (TypeError, ValueError):
            self.max_chars = 16000
        self.prices = dict(prices or {})
        self.project = project or env("BOLTZ_TRACE_PROJECT")
        self.labels = _labels(
            project=self.project,
            task=task or env("BOLTZ_TRACE_TASK"),
            model=model or env("BOLTZ_TRACE_MODEL"),
            attempt=attempt if attempt is not None else env("BOLTZ_TRACE_ATTEMPT"),
            tags=tags,
        )

        # Whoever launches the run may fix the id of its first trace, so the
        # file can be found afterwards without reading it.
        given = (env("BOLTZ_TRACE_ID") or "").lower()
        self.trace_id = given if len(given) == 32 and all(c in "0123456789abcdef" for c in given) else None
        # TRACEPARENT is how OpenTelemetry hands a trace to a child process
        # (the W3C trace context, in the environment). A run started under one
        # joins that trace, beneath the step that started it, so a tool that
        # reads OpenTelemetry sees one trace where there were two programs.
        self.parent_id = None
        if self.trace_id is None:
            self.trace_id, self.parent_id = _parse_traceparent(env("TRACEPARENT"))
        # The process-wide run, opened the first time a step needs one.
        self.implicit = None

        self.sinks = []
        if not self.enabled:
            return
        file = file or env("BOLTZ_TRACE_FILE")
        endpoint = endpoint or env("BOLTZ_TRACE_ENDPOINT")
        if file or not endpoint:
            self.sinks.append(FileSink(str(file or DEFAULT_DIR)))
        if endpoint:
            headers = dict(headers or {})
            # The account key in the environment goes to our own origin only.
            # Sending it to whatever endpoint was configured would hand a Boltz
            # credential to a third-party collector.
            key = api_key or (env("BOLTZLABS_API_KEY") if _is_boltz(endpoint) else None)
            if key and "Authorization" not in headers:
                headers["Authorization"] = "Bearer " + key
            self.sinks.append(HttpSink(endpoint, headers, self.project or "boltztracer"))

    def clean(self, value):
        return clean(value, self.max_chars, self.redact, self.mask)


def init(
    *,
    file=None,
    endpoint=None,
    headers=None,
    api_key=None,
    project=None,
    task=None,
    model=None,
    attempt=None,
    tags=None,
    prices=None,
    capture=None,
    mask=None,
    redact=None,
    max_chars=None,
    enabled=None,
):
    """Configure the tracer. Optional: the first step configures it from the
    environment if this was never called.

    ``file``       where traces are written. A path ending in ``.jsonl`` is one
                   file for everything; any other path is a directory with one
                   file per trace. Default ``.boltz/traces``.
    ``endpoint``   also send finished steps to this OTLP/HTTP traces URL.
    ``task``, ``model``, ``attempt``, ``tags``
                   labels for traces that do not set their own.
    ``prices``     ``{model: {"input": usd, "output": usd, "cached": usd}}`` per
                   million tokens, used to turn token counts into cost.
    ``capture``    ``False`` records the shape of a run but no inputs or outputs.
    ``mask``       a function applied to every input and output before it is kept.
    ``redact``     ``False`` turns off the built-in removal of API keys and tokens.
    ``max_chars``  longest string kept before it is clipped (default 16000).
    ``enabled``    ``False`` makes every call a no-op.
    """
    global _cfg
    new = _Config(
        file, endpoint, headers, api_key, project, task, model, attempt, tags,
        prices, capture, mask, redact, max_chars, enabled,
    )
    with _lock:
        old, _cfg = _cfg, new
    if old is not None:
        for sink in old.sinks:
            sink.flush(2.0)


def _config():
    global _cfg
    if _cfg is None:
        with _lock:
            if _cfg is None:
                _cfg = _Config()
    return _cfg


def price(model, input=0.0, output=0.0, cached=None, cache_write=None):
    """Set what ``model`` costs, in USD per million tokens."""
    _config().prices[model] = {"input": input, "output": output, "cached": cached, "cache_write": cache_write}


def _price_for(prices, model):
    if not model or not prices:
        return None
    if model in prices:
        return prices[model]
    # An endpoint often answers with a dated name ("gpt-x-2026-01-01") for the
    # model that was asked for; the longest matching prefix is that model.
    best = None
    for name in prices:
        if model.startswith(name) and (best is None or len(name) > len(best)):
            best = name
    return prices[best] if best is not None else None


def _cost(p, usage):
    if isinstance(p, (tuple, list)):
        p = {"input": p[0], "output": p[1]}
    p_in = float(p.get("input") or 0)
    p_out = float(p.get("output") or 0)
    p_cached = p_in if p.get("cached") is None else float(p["cached"])
    p_write = p_in if p.get("cache_write") is None else float(p["cache_write"])
    return (
        usage.get("input", 0) * p_in
        + usage.get("cached", 0) * p_cached
        + usage.get("cache_write", 0) * p_write
        + usage.get("output", 0) * p_out
    ) / 1e6


def _roll_up(span):
    """Add a finished step to its trace's running totals.

    Returns the totals when the step is the root, so one line of the file
    answers "what did this run cost" without reading the rest.
    """
    with _lock:
        t = _totals.get(span.trace_id)
        if t is None:
            t = _totals[span.trace_id] = {
                "spans": 0, "llm_calls": 0, "tool_calls": 0, "errors": 0,
                "input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "cost": None,
            }
        t["spans"] += 1
        if span.kind == "llm":
            t["llm_calls"] += 1
        elif span.kind == "tool":
            t["tool_calls"] += 1
        # A failure is counted where it happened, not again in every step it
        # passed through on the way up.
        if span.error is not None and "from" not in span.error:
            t["errors"] += 1
        usage = span._usage or {}
        t["input_tokens"] += usage.get("input", 0)
        t["output_tokens"] += usage.get("output", 0)
        t["cached_tokens"] += usage.get("cached", 0) + usage.get("cache_write", 0)
        if span.cost is not None:
            t["cost"] = round((t["cost"] or 0.0) + span.cost, 8)
        if span._top:
            return _totals.pop(span.trace_id)
    return None


class Span:
    """One step of a run. Use it in a ``with`` block, or call :meth:`end`."""

    def __init__(self, cfg, name, kind, trace_id, parent_id, labels, input, meta, model, adopted=None):
        self._cfg = cfg
        self.name = str(name)
        self.kind = kind
        self.trace_id = trace_id
        self.span_id = os.urandom(8).hex()
        # The top of a run is the step with nothing of this program above it.
        # It may still have a parent: the step, in another program, that the
        # run was started from.
        self._top = parent_id is None
        self.parent_id = parent_id if parent_id is not None else adopted
        self.labels = labels
        self.model = model
        self.status = "ok"
        self.error = None
        self.cost = None
        self._usage = None
        self._input = cfg.clean(input) if input is not None and cfg.capture else None
        self._output = None
        self._meta = {}
        if meta:
            self.set(**meta)
        self._token = None
        self._ended = False
        self.start_ns = time.time_ns()
        self._t0 = time.perf_counter_ns()
        self.end_ns = None
        self._emit("start")

    def __repr__(self):
        return f"<Span {self.kind} {self.name!r} {self.span_id}>"

    def set(self, output=None, **meta):
        """Record the step's output, and any extra facts about it."""
        if output is not None and self._cfg.capture:
            self._output = self._cfg.clean(output)
        if meta:
            cleaned = self._cfg.clean(meta)
            if isinstance(cleaned, dict):
                self._meta.update(cleaned)
        return self

    def usage(self, input=None, output=None, cached=None, cache_write=None, reasoning=None, cost=None, model=None):
        """Token counts for a model call.

        ``input`` is the input tokens that were *not* served from cache,
        ``cached`` the ones that were, ``cache_write`` the ones written to it.
        Pass ``cost`` in USD to override the price table.
        """
        counts = self._usage or {}
        for key, value in (
            ("input", input), ("output", output), ("cached", cached),
            ("cache_write", cache_write), ("reasoning", reasoning),
        ):
            if value:
                counts[key] = int(value)
        self._usage = counts or None
        if cost is not None:
            self.cost = float(cost)
        if model:
            self.model = model
        return self

    def fail(self, message, error_type="Error"):
        """Mark the step as failed without raising."""
        self.status = "error"
        self.error = {"type": error_type, "message": self._cfg.clean(str(message))}
        return self

    def _exception(self, exc):
        # A consumer that stops reading a generator early has not failed.
        if isinstance(exc, GeneratorExit):
            return
        self.status = "error"
        error = {"type": type(exc).__name__, "message": self._cfg.clean(str(exc))}
        # The same exception passes through every enclosing step on its way
        # out. Only the step it started in keeps the stack; the others point
        # at that step, which is what "where did it go wrong" needs.
        origin = getattr(exc, "__boltz_span__", None)
        if origin is not None and origin != self.span_id:
            error["from"] = origin
        else:
            stack = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            error["stack"] = self._cfg.clean(stack)
            try:
                exc.__boltz_span__ = self.span_id
            except Exception:
                pass
        self.error = error

    def end(self):
        """Finish the step. Safe to call more than once."""
        if self._ended:
            return
        self._ended = True
        self.end_ns = self.start_ns + (time.perf_counter_ns() - self._t0)
        if self.cost is None and self._usage:
            p = _price_for(self._cfg.prices, self.model)
            if p is not None:
                try:
                    self.cost = round(_cost(p, self._usage), 8)
                except Exception:
                    pass
        if self._top:
            with _lock:
                if self in _open:
                    _open.remove(self)
        self._emit("end", _roll_up(self))

    def _emit(self, ev, totals=None):
        rec = {
            "v": 1,
            "ev": ev,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "kind": self.kind,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "status": self.status if ev == "end" else "running",
        }
        if self.labels:
            rec["trace"] = self.labels
        if self._input is not None:
            rec["input"] = self._input
        if self.model:
            rec["model"] = self.model
        if self._meta:
            rec["meta"] = self._meta
        if ev == "end":
            if self._output is not None:
                rec["output"] = self._output
            if self.error is not None:
                rec["error"] = self.error
            if self._usage:
                rec["usage"] = self._usage
            if self.cost is not None:
                rec["cost"] = self.cost
            if totals is not None:
                rec["totals"] = totals
        for sink in self._cfg.sinks:
            try:
                sink.write(rec)
            except Exception as exc:
                log.debug("boltztracer: dropped a record: %s", exc)

    def __enter__(self):
        self._token = _current.set(self)
        return self

    def __exit__(self, exc_type, exc, tb):
        _reset(self._token)
        self._token = None
        if exc is not None:
            self._exception(exc)
        self.end()
        return False


class _Noop:
    """What every call returns when tracing is off: the same surface, no work."""

    trace_id = span_id = parent_id = model = None

    def set(self, *args, **kwargs):
        return self

    usage = fail = set

    def end(self):
        pass

    def _exception(self, exc):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


_NOOP = _Noop()


def _reset(token):
    if token is None:
        return
    try:
        _current.reset(token)
    except ValueError:
        # Closed from a different context than it was opened in (a generator
        # resumed by another task). The variable there was never ours to undo.
        pass


def _parse_traceparent(header):
    """The trace and parent ids out of a W3C ``traceparent``, or two Nones."""
    parts = (header or "").strip().lower().split("-")
    if len(parts) != 4:
        return None, None
    version, trace_id, parent_id, flags = parts
    hexes = all(c in "0123456789abcdef" for c in version + trace_id + parent_id + flags)
    if not hexes or len(version) != 2 or len(trace_id) != 32 or len(parent_id) != 16 or len(flags) != 2:
        return None, None
    # All zeros means "no trace" in that format, and ff is a version that
    # must not be read.
    if version == "ff" or not trace_id.strip("0") or not parent_id.strip("0"):
        return None, None
    return trace_id, parent_id


def traceparent(span=None):
    """The open step as a W3C ``traceparent``, for a program started from here.

        subprocess.run(cmd, env={**os.environ, "TRACEPARENT": bt.traceparent()})

    A child that uses this package, or any OpenTelemetry SDK, then records its
    steps into the same trace, under the step that started it. Returns None
    when nothing is open.
    """
    span = span or _current.get()
    if not isinstance(span, Span):
        return None
    return "00-%s-%s-01" % (span.trace_id, span.span_id)


def _root(cfg, name, labels, input=None):
    with _lock:
        trace_id, cfg.trace_id = cfg.trace_id or os.urandom(16).hex(), None
        parent_id, cfg.parent_id = cfg.parent_id, None
    span = Span(cfg, name, "agent", trace_id, None, labels, input, None, None, adopted=parent_id)
    return span


def _run(cfg):
    """The run a step belongs to when nothing is open around it.

    Usually that is a step opened in a worker thread, which cannot see what was
    open where it was started. With exactly one run open there is no doubt
    which it belongs to. Otherwise it joins the process-wide run, which is
    opened on first use and closed when the interpreter exits.
    """
    with _lock:
        if len(_open) == 1:
            return _open[0]
        if cfg.implicit is None:
            cfg.implicit = _root(cfg, cfg.labels.get("task") or "run", cfg.labels)
            atexit.register(cfg.implicit.end)
        return cfg.implicit


def _start(name, kind="step", input=None, parent=None, meta=None, model=None):
    cfg = _config()
    if not cfg.enabled:
        return _NOOP
    if parent is None:
        parent = _current.get()
    if not isinstance(parent, Span):
        parent = _run(cfg)
    return Span(cfg, name, kind, parent.trace_id, parent.span_id, parent.labels, input, meta, model)


# -- opening steps -------------------------------------------------------------


def trace(name=None, *, task=None, model=None, attempt=None, tags=None, input=None, **meta):
    """Start a new trace: one run of one model on one task.

        with bt.trace(task="fix-pagination", model="my-model", attempt=1):
            run_agent()

    ``task``, ``model`` and ``attempt`` are what line traces up against each
    other afterwards: the same task under several models is a comparison.
    Extra keyword arguments are kept as labels on the whole trace.
    """
    cfg = _config()
    if not cfg.enabled:
        return _NOOP
    base = cfg.labels
    labels = _labels(
        project=cfg.project,
        name=name,
        task=task or base.get("task"),
        model=model or base.get("model"),
        attempt=attempt if attempt is not None else base.get("attempt"),
        tags=tags or base.get("tags"),
        meta=cfg.clean(meta) if meta else None,
    )
    span = _root(cfg, name or labels.get("task") or "run", labels, input)
    with _lock:
        _open.append(span)
    return span


def step(name, kind="step", input=None, parent=None, **meta):
    """Open a step. It nests under whichever step is open around it.

        with bt.step("plan") as s:
            ...
            s.set(output=plan)

    ``parent`` attaches it to a specific step instead, which is how work
    handed to another thread keeps its exact place in the tree.
    """
    return _start(name, kind, input, parent=parent, meta=meta)


def llm(model=None, input=None, name=None, parent=None, **meta):
    """Open a model-call step by hand. :func:`wrap` does this for you.

        with bt.llm(model="my-model", input=messages) as s:
            reply = call_model(messages)
            s.set(output=reply.text)
            s.usage(input=reply.tokens_in, output=reply.tokens_out)
    """
    label = name or ("chat " + str(model) if model else "llm")
    return _start(label, "llm", input, parent=parent, meta=meta, model=model)


def current():
    """The step that is open right now, or ``None``."""
    return _current.get()


def flush(timeout=5.0):
    """Wait for anything still queued to be sent. Files need no flushing."""
    cfg = _cfg
    if cfg is None:
        return True
    return all([sink.flush(timeout) for sink in cfg.sinks])


# -- decorators ----------------------------------------------------------------


def _arguments(sig, args, kwargs):
    if not args and not kwargs:
        return None
    if sig is not None:
        try:
            bound = dict(sig.bind(*args, **kwargs).arguments)
            bound.pop("self", None)
            bound.pop("cls", None)
            return bound or None
        except TypeError:
            pass
    out = {}
    if args:
        out["args"] = list(args)
    if kwargs:
        out["kwargs"] = kwargs
    return out


def _wrap_fn(f, label, kind):
    try:
        sig = inspect.signature(f)
    except (TypeError, ValueError):
        sig = None

    def begin(args, kwargs):
        return _start(label, kind, _arguments(sig, args, kwargs))

    if inspect.isasyncgenfunction(f):

        @functools.wraps(f)
        async def wrapper(*args, **kwargs):
            span = begin(args, kwargs)
            items = []
            it = f(*args, **kwargs)
            try:
                while True:
                    # The step is "open" only while the generator's own code
                    # runs, so steps the consumer opens between items do not
                    # land inside it.
                    token = _current.set(span)
                    try:
                        item = await it.__anext__()
                    except StopAsyncIteration:
                        break
                    finally:
                        _reset(token)
                    if len(items) < _KEEP:
                        items.append(item)
                    yield item
            except BaseException as exc:
                span._exception(exc)
                raise
            finally:
                await it.aclose()
                span.set(output=items or None)
                span.end()

    elif inspect.iscoroutinefunction(f):

        @functools.wraps(f)
        async def wrapper(*args, **kwargs):
            with begin(args, kwargs) as span:
                out = await f(*args, **kwargs)
                span.set(output=out)
                return out

    elif inspect.isgeneratorfunction(f):

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            span = begin(args, kwargs)
            items = []
            it = f(*args, **kwargs)
            try:
                while True:
                    token = _current.set(span)
                    try:
                        item = next(it)
                    except StopIteration:
                        break
                    finally:
                        _reset(token)
                    if len(items) < _KEEP:
                        items.append(item)
                    yield item
            except BaseException as exc:
                span._exception(exc)
                raise
            finally:
                it.close()
                span.set(output=items or None)
                span.end()

    else:

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            with begin(args, kwargs) as span:
                out = f(*args, **kwargs)
                span.set(output=out)
                return out

    return wrapper


def observe(fn=None, *, name=None, kind="step"):
    """Turn every call of a function into a step.

        @bt.observe
        def plan(task): ...

        @bt.observe("load-context", kind="tool")
        async def load(path): ...

    The arguments become the step's input and the return value its output.
    Works on plain functions, coroutines, generators and async generators.
    """
    if isinstance(fn, str):
        name, fn = fn, None

    def decorate(f):
        return _wrap_fn(f, name or getattr(f, "__name__", kind), kind)

    return decorate(fn) if fn is not None else decorate


def tool(fn=None, *, name=None):
    """:func:`observe` for a tool the agent calls."""
    return observe(fn, name=name, kind="tool")


def agent(fn=None, *, name=None):
    """:func:`observe` for a sub-agent: its own box, with its steps inside it."""
    return observe(fn, name=name, kind="agent")
