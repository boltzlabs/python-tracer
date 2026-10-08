"""Recording model calls without the caller writing anything.

``wrap(client)`` replaces a client's ``create`` methods with ones that open an
``llm`` step around the call. It understands the three request shapes agents
actually use: OpenAI chat completions (which is also what most self-hosted and
third-party endpoints speak), OpenAI responses, and Anthropic messages.

Everything is read by attribute or key name rather than by importing a model
SDK, so this works with any client that has the same shape and adds no
dependency.
"""

import functools
import inspect
import logging

from . import _core

__all__ = ["wrap"]

log = logging.getLogger("boltztracer")


def _g(obj, name):
    """``obj.name`` or ``obj[name]``; ``None`` if either is missing."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _compact(d):
    return {k: v for k, v in d.items() if v not in (None, [], {})}


def _tool_names(tools):
    # The schemas are large and identical on every call of a run; the names are
    # what tells one call from another.
    names = []
    for t in tools or []:
        name = _g(_g(t, "function"), "name") or _g(t, "name")
        if name:
            names.append(name)
    return names


# -- what was asked ------------------------------------------------------------


def _ask_chat(kw):
    return _compact({"messages": kw.get("messages"), "tools": _tool_names(kw.get("tools"))})


def _ask_responses(kw):
    return _compact(
        {"instructions": kw.get("instructions"), "input": kw.get("input"), "tools": _tool_names(kw.get("tools"))}
    )


def _ask_messages(kw):
    return _compact(
        {"system": kw.get("system"), "messages": kw.get("messages"), "tools": _tool_names(kw.get("tools"))}
    )


# -- what came back ------------------------------------------------------------
# Each reader returns (output, usage, model, finish_reason).


def _answer(text, calls):
    out = {"role": "assistant", "content": text or None}
    if calls:
        out["tool_calls"] = calls
    return out


def _read_chat(resp):
    choice = (_g(resp, "choices") or [None])[0]
    msg = _g(choice, "message")
    calls = [
        {"name": _g(_g(c, "function"), "name"), "arguments": _g(_g(c, "function"), "arguments")}
        for c in _g(msg, "tool_calls") or []
    ]
    return _answer(_g(msg, "content"), calls), _g(resp, "usage"), _g(resp, "model"), _g(choice, "finish_reason")


def _read_responses(resp):
    texts, calls = [], []
    for item in _g(resp, "output") or []:
        kind = _g(item, "type")
        if kind == "function_call":
            calls.append({"name": _g(item, "name"), "arguments": _g(item, "arguments")})
        elif kind == "message":
            for part in _g(item, "content") or []:
                if _g(part, "text"):
                    texts.append(_g(part, "text"))
    return _answer("".join(texts), calls), _g(resp, "usage"), _g(resp, "model"), _g(resp, "status")


def _read_messages(resp):
    texts, calls = [], []
    for block in _g(resp, "content") or []:
        kind = _g(block, "type")
        if kind == "text":
            texts.append(_g(block, "text") or "")
        elif kind == "tool_use":
            calls.append({"name": _g(block, "name"), "arguments": _g(block, "input")})
    return _answer("".join(texts), calls), _g(resp, "usage"), _g(resp, "model"), _g(resp, "stop_reason")


def _usage(u):
    """Provider token counts as ``Span.usage`` arguments.

    The providers disagree on whether "input tokens" includes the ones served
    from cache. Ours never does, so a price table can charge the two rates
    without knowing which provider answered.
    """
    if u is None:
        return {}
    prompt, completion = _g(u, "prompt_tokens"), _g(u, "completion_tokens")
    if prompt is not None or completion is not None:  # chat completions: cached is inside prompt
        cached = _g(_g(u, "prompt_tokens_details"), "cached_tokens") or 0
        return {
            "input": max((prompt or 0) - cached, 0),
            "output": completion or 0,
            "cached": cached,
            "reasoning": _g(_g(u, "completion_tokens_details"), "reasoning_tokens") or 0,
        }
    inp, out = _g(u, "input_tokens") or 0, _g(u, "output_tokens") or 0
    details = _g(u, "input_tokens_details")
    if details is not None:  # responses: cached is inside input
        cached = _g(details, "cached_tokens") or 0
        return {
            "input": max(inp - cached, 0),
            "output": out,
            "cached": cached,
            "reasoning": _g(_g(u, "output_tokens_details"), "reasoning_tokens") or 0,
        }
    return {  # messages: cached is reported beside input
        "input": inp,
        "output": out,
        "cached": _g(u, "cache_read_input_tokens") or 0,
        "cache_write": _g(u, "cache_creation_input_tokens") or 0,
    }


def _finish(span, output, usage, model, finish):
    span.set(output=output)
    if finish:
        span.set(finish_reason=finish)
    # The step keeps the name that was asked for, because that is the name a
    # price was set for; the name the endpoint answered with is kept beside it.
    if model and span.model and model != span.model:
        span.set(response_model=model)
    span.usage(model=None if span.model else model, **_usage(usage))
    span.end()


# -- streaming -----------------------------------------------------------------


class _Acc:
    """Rebuild one answer from the pieces a streaming call delivers."""

    _ANTHROPIC_COUNTS = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")

    def __init__(self, api):
        self.api = api
        self.text = []
        self.calls = {}
        self.usage = None
        self.model = None
        self.finish = None
        self.final = None

    def add(self, ev):
        if self.api == "chat":
            self.model = _g(ev, "model") or self.model
            if _g(ev, "usage") is not None:
                self.usage = _g(ev, "usage")
            for choice in (_g(ev, "choices") or [])[:1]:
                delta = _g(choice, "delta")
                if _g(delta, "content"):
                    self.text.append(_g(delta, "content"))
                for call in _g(delta, "tool_calls") or []:
                    slot = self.calls.setdefault(_g(call, "index") or 0, {"name": "", "arguments": ""})
                    fn = _g(call, "function")
                    if _g(fn, "name"):
                        slot["name"] = _g(fn, "name")
                    if _g(fn, "arguments"):
                        slot["arguments"] += _g(fn, "arguments")
                if _g(choice, "finish_reason"):
                    self.finish = _g(choice, "finish_reason")
            return

        kind = _g(ev, "type")
        if self.api == "responses":
            # The last event carries the whole response, usage included.
            if kind in ("response.completed", "response.incomplete", "response.failed"):
                self.final = _g(ev, "response")
            elif kind == "response.output_text.delta" and _g(ev, "delta"):
                self.text.append(_g(ev, "delta"))
            return

        if kind == "message_start":
            self.model = _g(_g(ev, "message"), "model")
            self._count(_g(_g(ev, "message"), "usage"))
        elif kind == "content_block_start":
            block = _g(ev, "content_block")
            if _g(block, "type") == "tool_use":
                self.calls[_g(ev, "index") or 0] = {"name": _g(block, "name"), "arguments": ""}
        elif kind == "content_block_delta":
            delta = _g(ev, "delta")
            if _g(delta, "type") == "text_delta":
                self.text.append(_g(delta, "text") or "")
            elif _g(delta, "type") == "input_json_delta":
                slot = self.calls.setdefault(_g(ev, "index") or 0, {"name": "", "arguments": ""})
                slot["arguments"] += _g(delta, "partial_json") or ""
        elif kind == "message_delta":
            self._count(_g(ev, "usage"))
            self.finish = _g(_g(ev, "delta"), "stop_reason") or self.finish

    def _count(self, u):
        # Input counts arrive at the start and output counts at the end.
        if u is None:
            return
        self.usage = self.usage or {}
        for key in self._ANTHROPIC_COUNTS:
            if _g(u, key) is not None:
                self.usage[key] = _g(u, key)

    def result(self):
        if self.final is not None:
            return _read_responses(self.final)
        calls = [self.calls[i] for i in sorted(self.calls)]
        return _answer("".join(self.text), calls), self.usage, self.model, self.finish


class _Stream:
    """A streaming response that also finishes the step when it is done.

    It passes every chunk through unchanged and forwards anything else to the
    real stream, so code written against the SDK's stream keeps working.
    """

    def __init__(self, inner, span, acc):
        self._inner = inner
        self._span = span
        self._acc = acc
        self._it = None
        self._closed = False

    def _done(self, exc=None):
        if self._closed:
            return
        self._closed = True
        try:
            if exc is not None:
                self._span._exception(exc)
            _finish(self._span, *self._acc.result())
        except Exception as err:
            log.debug("boltztracer: could not finish a streamed step: %s", err)
            self._span.end()

    def _see(self, chunk):
        try:
            self._acc.add(chunk)
        except Exception as err:
            log.debug("boltztracer: could not read a stream chunk: %s", err)

    def __iter__(self):
        return self

    def __next__(self):
        if self._it is None:
            self._it = iter(self._inner)
        try:
            chunk = next(self._it)
        except StopIteration:
            self._done()
            raise
        except BaseException as exc:
            self._done(exc)
            raise
        self._see(chunk)
        return chunk

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._it is None:
            self._it = self._inner.__aiter__()
        try:
            chunk = await self._it.__anext__()
        except StopAsyncIteration:
            self._done()
            raise
        except BaseException as exc:
            self._done(exc)
            raise
        self._see(chunk)
        return chunk

    def __enter__(self):
        enter = getattr(self._inner, "__enter__", None)
        if enter is not None:
            enter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._done(exc)
        leave = getattr(self._inner, "__exit__", None)
        return leave(exc_type, exc, tb) if leave is not None else False

    async def __aenter__(self):
        enter = getattr(self._inner, "__aenter__", None)
        if enter is not None:
            await enter()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self._done(exc)
        leave = getattr(self._inner, "__aexit__", None)
        return await leave(exc_type, exc, tb) if leave is not None else False

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __del__(self):
        # A caller that stopped reading early still gets a finished step.
        try:
            self._done()
        except Exception:
            pass


# -- patching ------------------------------------------------------------------

_APIS = {
    "chat": ("chat.completions", _ask_chat, _read_chat),
    "responses": ("responses", _ask_responses, _read_responses),
    "messages": ("messages", _ask_messages, _read_messages),
}


def _traced(create, api):
    _, ask, read = _APIS[api]

    def after(span, kwargs, result):
        try:
            if kwargs.get("stream"):
                return _Stream(result, span, _Acc(api))
            _finish(span, *read(result))
        except Exception as err:
            log.debug("boltztracer: could not read a model response: %s", err)
            span.end()
        return result

    async def waited(span, kwargs, pending):
        try:
            result = await pending
        except BaseException as exc:
            span._exception(exc)
            span.end()
            raise
        return after(span, kwargs, result)

    @functools.wraps(create)
    def traced(*args, **kwargs):
        span = _core.llm(model=kwargs.get("model"), input=ask(kwargs))
        if not isinstance(span, _core.Span):  # tracing is off
            return create(*args, **kwargs)
        try:
            result = create(*args, **kwargs)
        except BaseException as exc:
            span._exception(exc)
            span.end()
            raise
        # An async client hands back something to await. Asking the result,
        # rather than the method, is what works when the SDK has wrapped its
        # coroutine in an ordinary function.
        if inspect.isawaitable(result):
            return waited(span, kwargs, result)
        return after(span, kwargs, result)

    traced._boltz = True
    return traced


def wrap(client):
    """Record every model call made through ``client``. Returns the same client.

        client = bt.wrap(OpenAI(base_url="https://my-endpoint/v1"))
        client.chat.completions.create(model="my-model", messages=[...])

    Each call becomes an ``llm`` step with the prompt, the answer, the tool
    calls the model asked for, token counts and (if a price is set) cost.
    Sync and async clients, streaming and not, are all covered.

    Only ``create`` is patched. Helper methods that bypass it (``.parse()``,
    ``.stream()``) are not recorded; use :func:`boltztracer.llm` around those.
    """
    found = 0
    for api, (path, _, _) in _APIS.items():
        target = client
        for part in path.split("."):
            target = getattr(target, part, None)
            if target is None:
                break
        create = getattr(target, "create", None) if target is not None else None
        if not callable(create):
            continue
        found += 1
        if getattr(create, "_boltz", False):
            continue
        try:
            setattr(target, "create", _traced(create, api))
        except Exception as err:
            found -= 1
            log.debug("boltztracer: could not wrap %s.create: %s", path, err)
    if not found:
        log.warning(
            "boltztracer.wrap: %s has no chat.completions, responses or messages API; nothing will be recorded",
            type(client).__name__,
        )
    return client
