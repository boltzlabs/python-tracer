"""wrap() against stand-ins with the same shape as the real model SDKs.

The stand-ins are plain objects and dicts on purpose: wrap() reads by name, so
whatever passes here is what a real client must look like for it to work.
"""

import asyncio
from types import SimpleNamespace as NS

import pytest

import boltztracer as bt

MESSAGES = [{"role": "user", "content": "hi"}]


def client_with(path, create):
    """A client whose `path.create` is the given function."""
    holder = NS(create=create)
    for part in reversed(path.split(".")):
        holder = NS(**{part: holder})
    return holder


def chat_response(text="hello", tool=None):
    calls = [NS(function=NS(name=tool, arguments='{"q": 1}'))] if tool else None
    return NS(
        model="m1-2026-01-01",
        choices=[NS(finish_reason="tool_calls" if tool else "stop", message=NS(content=text, tool_calls=calls))],
        usage=NS(
            prompt_tokens=100,
            completion_tokens=20,
            prompt_tokens_details=NS(cached_tokens=60),
            completion_tokens_details=NS(reasoning_tokens=5),
        ),
    )


def test_chat_completion(out):
    t = out(prices={"m1": {"input": 1.0, "output": 2.0, "cached": 0.1}})
    seen = {}

    def create(**kw):
        seen.update(kw)
        return chat_response(text=None, tool="search")

    client = bt.wrap(client_with("chat.completions", create))
    tools = [{"type": "function", "function": {"name": "search", "parameters": {}}}]
    with bt.trace(task="t") as root:
        reply = client.chat.completions.create(model="m1", messages=MESSAGES, tools=tools, temperature=0)

    assert reply.choices[0].message.tool_calls[0].function.name == "search"  # untouched
    assert seen["temperature"] == 0
    rec = t.one("chat m1")
    assert rec["kind"] == "llm" and rec["parent_id"] == root.span_id and rec["model"] == "m1"
    assert rec["input"] == {"messages": MESSAGES, "tools": ["search"]}
    assert rec["output"] == {
        "role": "assistant", "content": None, "tool_calls": [{"name": "search", "arguments": '{"q": 1}'}],
    }
    # prompt_tokens includes the cached ones; ours are split.
    assert rec["usage"] == {"input": 40, "output": 20, "cached": 60, "reasoning": 5}
    assert rec["meta"] == {"finish_reason": "tool_calls", "response_model": "m1-2026-01-01"}
    assert rec["cost"] == pytest.approx((40 * 1.0 + 60 * 0.1 + 20 * 2.0) / 1e6)


def chat_chunks():
    def chunk(content=None, call=None, finish=None, usage=None):
        delta = NS(content=content, tool_calls=[call] if call else None)
        return NS(model="m1", usage=usage, choices=[NS(delta=delta, finish_reason=finish)])

    return [
        chunk(content="Hel"),
        chunk(content="lo"),
        chunk(call=NS(index=0, function=NS(name="search", arguments='{"q"'))),
        chunk(call=NS(index=0, function=NS(name=None, arguments=": 1}"))),
        chunk(finish="tool_calls"),
        NS(model="m1", choices=[], usage=NS(prompt_tokens=10, completion_tokens=4, prompt_tokens_details=None)),
    ]


class FakeStream:
    """Iterable, closable and a context manager, like the SDK's stream."""

    def __init__(self, items):
        self.items, self.entered, self.exited, self.response = list(items), False, False, "raw"

    def __iter__(self):
        return iter(self.items)

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *exc):
        self.exited = True
        return False


def test_chat_stream_is_rebuilt_and_passed_through(out):
    t = out()
    inner = FakeStream(chat_chunks())
    client = bt.wrap(client_with("chat.completions", lambda **kw: inner))

    with client.chat.completions.create(model="m1", messages=MESSAGES, stream=True) as stream:
        assert stream.response == "raw"  # everything else reaches the real stream
        assert t.ended() == []  # still open while the caller reads
        got = list(stream)

    assert len(got) == 6 and inner.entered and inner.exited
    rec = t.one("chat m1")
    assert rec["output"] == {
        "role": "assistant", "content": "Hello", "tool_calls": [{"name": "search", "arguments": '{"q": 1}'}],
    }
    assert rec["usage"] == {"input": 10, "output": 4}
    assert rec["meta"] == {"finish_reason": "tool_calls"}


def test_a_stream_abandoned_early_still_finishes_its_step(out):
    t = out()
    client = bt.wrap(client_with("chat.completions", lambda **kw: FakeStream(chat_chunks())))
    stream = client.chat.completions.create(model="m1", messages=MESSAGES, stream=True)
    assert next(iter(stream)).choices[0].delta.content == "Hel"
    del stream
    assert t.one("chat m1")["output"]["content"] == "Hel"


def test_async_clients(out):
    t = out()

    async def create(**kw):
        return chat_response("from async")

    def create_returning_coroutine(**kw):
        # What a real SDK does: an ordinary function that hands back a coroutine.
        return create(**kw)

    class AsyncStream:
        def __aiter__(self):
            return self._gen()

        async def _gen(self):
            for c in chat_chunks():
                yield c

    async def create_stream(**kw):
        return AsyncStream()

    async def main():
        a = bt.wrap(client_with("chat.completions", create))
        b = bt.wrap(client_with("chat.completions", create_returning_coroutine))
        c = bt.wrap(client_with("chat.completions", create_stream))
        with bt.trace(task="t"):
            await a.chat.completions.create(model="a", messages=MESSAGES)
            await b.chat.completions.create(model="b", messages=MESSAGES)
            stream = await c.chat.completions.create(model="c", messages=MESSAGES, stream=True)
            return [chunk async for chunk in stream]

    assert len(asyncio.run(main())) == 6
    assert t.one("chat a")["output"]["content"] == "from async"
    assert t.one("chat b")["output"]["content"] == "from async"
    assert t.one("chat c")["output"]["content"] == "Hello"
    assert t.one("t")["totals"]["llm_calls"] == 3


def test_anthropic_messages(out):
    t = out()
    response = {
        "model": "model-x",
        "stop_reason": "tool_use",
        "content": [
            {"type": "text", "text": "Let me look."},
            {"type": "tool_use", "name": "read", "input": {"path": "a.py"}},
        ],
        "usage": {
            "input_tokens": 30, "output_tokens": 12,
            "cache_read_input_tokens": 500, "cache_creation_input_tokens": 70,
        },
    }
    events = [
        {"type": "message_start", "message": {"model": "model-x", "usage": {"input_tokens": 30, "cache_read_input_tokens": 500}}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Let me "}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "look."}},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "name": "read"}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"path": "a.py"}'}},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 12}},
    ]
    client = bt.wrap(client_with("messages", lambda **kw: iter(events) if kw.get("stream") else response))

    client.messages.create(model="model-x", system="be brief", messages=MESSAGES, max_tokens=100)
    list(client.messages.create(model="model-x", messages=MESSAGES, max_tokens=100, stream=True))

    whole, streamed = t.ended()
    assert whole["input"] == {"system": "be brief", "messages": MESSAGES}
    assert whole["output"] == {
        "role": "assistant", "content": "Let me look.", "tool_calls": [{"name": "read", "arguments": {"path": "a.py"}}],
    }
    # Here cached tokens are reported beside input, not inside it.
    assert whole["usage"] == {"input": 30, "output": 12, "cached": 500, "cache_write": 70}
    assert streamed["output"]["content"] == "Let me look."
    assert streamed["output"]["tool_calls"] == [{"name": "read", "arguments": '{"path": "a.py"}'}]
    assert streamed["usage"] == {"input": 30, "output": 12, "cached": 500}
    assert streamed["meta"] == {"finish_reason": "tool_use"}


def test_openai_responses(out):
    t = out()
    response = NS(
        model="m1",
        status="completed",
        output=[
            NS(type="message", content=[NS(type="output_text", text="42")]),
            NS(type="function_call", name="calc", arguments="{}"),
        ],
        usage=NS(
            input_tokens=50, output_tokens=8,
            input_tokens_details=NS(cached_tokens=10), output_tokens_details=NS(reasoning_tokens=3),
        ),
    )
    events = [NS(type="response.output_text.delta", delta="4"), NS(type="response.completed", response=response)]
    client = bt.wrap(client_with("responses", lambda **kw: iter(events) if kw.get("stream") else response))

    client.responses.create(model="m1", instructions="be exact", input="6*7?")
    list(client.responses.create(model="m1", input="6*7?", stream=True))

    whole, streamed = t.ended()
    assert whole["input"] == {"instructions": "be exact", "input": "6*7?"}
    assert whole["output"] == {"role": "assistant", "content": "42", "tool_calls": [{"name": "calc", "arguments": "{}"}]}
    assert whole["usage"] == {"input": 40, "output": 8, "cached": 10, "reasoning": 3}
    assert streamed["output"] == whole["output"] and streamed["usage"] == whole["usage"]


def test_a_failed_call_is_recorded_and_reraised(out):
    t = out()

    def create(**kw):
        raise TimeoutError("endpoint did not answer")

    client = bt.wrap(client_with("chat.completions", create))
    with pytest.raises(TimeoutError):
        client.chat.completions.create(model="m1", messages=MESSAGES)
    rec = t.one("chat m1")
    assert rec["status"] == "error" and rec["error"]["type"] == "TimeoutError"


def test_an_unreadable_response_never_breaks_the_call(out):
    t = out()
    client = bt.wrap(client_with("chat.completions", lambda **kw: 12345))
    assert client.chat.completions.create(model="m1", messages=MESSAGES) == 12345
    assert t.one("chat m1")["status"] == "ok"


def test_wrapping_twice_records_once_and_strangers_are_reported(out, caplog):
    t = out()
    client = client_with("chat.completions", lambda **kw: chat_response())
    assert bt.wrap(bt.wrap(client)) is client
    client.chat.completions.create(model="m1", messages=MESSAGES)
    assert len(t.ended()) == 1

    bt.wrap(NS(complete=lambda: None))
    assert "nothing will be recorded" in caplog.text


def test_wrap_steps_aside_when_tracing_is_off():
    bt.init(enabled=False)
    client = bt.wrap(client_with("chat.completions", lambda **kw: chat_response("plain")))
    assert client.chat.completions.create(model="m1", messages=MESSAGES).choices[0].message.content == "plain"
