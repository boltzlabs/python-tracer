"""wrap() against the real OpenAI and Anthropic packages.

The other wrap tests use stand-ins; these use the SDKs themselves, pointed at a
server on this machine that answers in each provider's wire format. No key and
no network are involved. They skip when the packages are not installed:

    pip install openai anthropic
"""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import boltztracer as bt

MESSAGES = [{"role": "user", "content": "hi"}]

CHAT = {
    "id": "c1", "object": "chat.completion", "created": 1, "model": "m1-2026-01-01",
    "choices": [{
        "index": 0, "finish_reason": "tool_calls",
        "message": {
            "role": "assistant", "content": "Hello",
            "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "search", "arguments": '{"q": 1}'}}],
        },
    }],
    "usage": {
        "prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
        "prompt_tokens_details": {"cached_tokens": 60},
    },
}


def chat_chunk(delta=None, finish=None, usage=None):
    choices = [] if delta is None and finish is None else [{"index": 0, "delta": delta or {}, "finish_reason": finish}]
    return {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "m1", "choices": choices, "usage": usage}


CHAT_STREAM = [
    chat_chunk({"role": "assistant", "content": "Hel"}),
    chat_chunk({"content": "lo"}),
    chat_chunk({"tool_calls": [{"index": 0, "id": "t1", "type": "function", "function": {"name": "search", "arguments": '{"q"'}}]}),
    chat_chunk({"tool_calls": [{"index": 0, "function": {"arguments": ": 1}"}}]}),
    chat_chunk(finish="tool_calls"),
    chat_chunk(usage={"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}),
]

RESPONSE = {
    "id": "r1", "object": "response", "created_at": 1, "status": "completed", "model": "m1",
    "output": [
        {"type": "message", "id": "msg1", "status": "completed", "role": "assistant",
         "content": [{"type": "output_text", "text": "42", "annotations": []}]},
        {"type": "function_call", "id": "fc1", "call_id": "call1", "name": "calc", "arguments": "{}", "status": "completed"},
    ],
    "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
    "usage": {
        "input_tokens": 50, "output_tokens": 8, "total_tokens": 58,
        "input_tokens_details": {"cached_tokens": 10}, "output_tokens_details": {"reasoning_tokens": 3},
    },
}

RESPONSE_STREAM = [
    {"type": "response.output_text.delta", "delta": "42", "item_id": "msg1", "output_index": 0,
     "content_index": 0, "sequence_number": 1, "logprobs": []},
    {"type": "response.completed", "response": RESPONSE, "sequence_number": 2},
]

MESSAGE = {
    "id": "msg_1", "type": "message", "role": "assistant", "model": "model-x",
    "content": [
        {"type": "text", "text": "Let me look."},
        {"type": "tool_use", "id": "tu1", "name": "read", "input": {"path": "a.py"}},
    ],
    "stop_reason": "tool_use", "stop_sequence": None,
    "usage": {"input_tokens": 30, "output_tokens": 12, "cache_read_input_tokens": 500, "cache_creation_input_tokens": 70},
}

MESSAGE_STREAM = [
    {"type": "message_start", "message": {
        "id": "msg_1", "type": "message", "role": "assistant", "model": "model-x", "content": [],
        "stop_reason": None, "stop_sequence": None,
        "usage": {"input_tokens": 30, "output_tokens": 1, "cache_read_input_tokens": 500, "cache_creation_input_tokens": 0},
    }},
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Let me "}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "look."}},
    {"type": "content_block_stop", "index": 0},
    {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "tu1", "name": "read", "input": {}}},
    {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"path": "a.py"}'}},
    {"type": "content_block_stop", "index": 1},
    {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None}, "usage": {"output_tokens": 12}},
    {"type": "message_stop"},
]

ROUTES = {
    "/v1/chat/completions": (CHAT, CHAT_STREAM, False),
    "/v1/responses": (RESPONSE, RESPONSE_STREAM, True),
    "/v1/messages": (MESSAGE, MESSAGE_STREAM, True),
}


@pytest.fixture(scope="module")
def endpoint():
    """A model endpoint on this machine, speaking all three wire formats."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            asked = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if asked.get("model") == "broken":
                return self.send(500, "application/json", json.dumps({"error": {"message": "boom", "type": "server_error"}}))
            whole, pieces, named = ROUTES[self.path]
            if not asked.get("stream"):
                return self.send(200, "application/json", json.dumps(whole))
            lines = []
            for piece in pieces:
                if named:  # these two name each event; chat completions does not
                    lines.append(f"event: {piece['type']}\n")
                lines.append(f"data: {json.dumps(piece)}\n\n")
            if not named:
                lines.append("data: [DONE]\n\n")
            self.send(200, "text/event-stream", "".join(lines))

        def send(self, status, kind, text):
            body = text.encode()
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    # Threaded: each client keeps its connection open, and a server that took
    # them one at a time would stall the second client behind the first.
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


CHAT_OUT = {"role": "assistant", "content": "Hello", "tool_calls": [{"name": "search", "arguments": '{"q": 1}'}]}
RESPONSE_OUT = {"role": "assistant", "content": "42", "tool_calls": [{"name": "calc", "arguments": "{}"}]}


def test_openai_client(out, endpoint):
    openai = pytest.importorskip("openai")
    t = out(prices={"m1": {"input": 1.0, "output": 2.0, "cached": 0.1}})
    client = bt.wrap(openai.OpenAI(base_url=endpoint + "/v1", api_key="test", max_retries=0))

    with bt.trace(task="t") as root:
        reply = client.chat.completions.create(model="m1", messages=MESSAGES)
        assert reply.choices[0].message.content == "Hello"  # the SDK's own object, untouched

        with client.chat.completions.create(
            model="m1", messages=MESSAGES, stream=True, stream_options={"include_usage": True}
        ) as stream:
            assert "".join(c.choices[0].delta.content or "" for c in stream if c.choices) == "Hello"

        assert client.responses.create(model="m1", input="6*7?").output_text == "42"
        events = list(client.responses.create(model="m1", input="6*7?", stream=True))
        assert events[-1].type == "response.completed"

        with pytest.raises(openai.InternalServerError):
            client.chat.completions.create(model="broken", messages=MESSAGES)

    whole, streamed, response, response_streamed, broken = [r for r in t.ended() if r["kind"] == "llm"]
    assert all(r["parent_id"] == root.span_id for r in (whole, streamed, response, response_streamed, broken))
    assert whole["input"] == {"messages": MESSAGES} and whole["output"] == CHAT_OUT
    assert whole["usage"] == {"input": 40, "output": 20, "cached": 60}
    assert whole["meta"] == {"finish_reason": "tool_calls", "response_model": "m1-2026-01-01"}
    assert whole["cost"] == pytest.approx((40 * 1.0 + 60 * 0.1 + 20 * 2.0) / 1e6)
    assert streamed["output"] == CHAT_OUT and streamed["usage"] == {"input": 10, "output": 4}
    assert response["output"] == RESPONSE_OUT
    assert response["usage"] == {"input": 40, "output": 8, "cached": 10, "reasoning": 3}
    assert response_streamed["output"] == RESPONSE_OUT and response_streamed["usage"] == response["usage"]
    assert broken["status"] == "error" and broken["error"]["type"] == "InternalServerError"
    assert t.one("t")["totals"]["llm_calls"] == 5


def test_async_openai_client(out, endpoint):
    openai = pytest.importorskip("openai")
    t = out()

    async def main():
        client = bt.wrap(openai.AsyncOpenAI(base_url=endpoint + "/v1", api_key="test", max_retries=0))
        with bt.trace(task="t"):
            reply = await client.chat.completions.create(model="m1", messages=MESSAGES)
            stream = await client.chat.completions.create(model="m1", messages=MESSAGES, stream=True)
            text = "".join([c.choices[0].delta.content or "" async for c in stream if c.choices])
            answer = await client.responses.create(model="m1", input="6*7?")
        await client.close()
        return reply.choices[0].message.content, text, answer.output_text

    assert asyncio.run(main()) == ("Hello", "Hello", "42")
    whole, streamed, response = [r for r in t.ended() if r["kind"] == "llm"]
    assert whole["output"] == CHAT_OUT and streamed["output"] == CHAT_OUT and response["output"] == RESPONSE_OUT


def test_anthropic_clients(out, endpoint):
    anthropic = pytest.importorskip("anthropic")
    t = out()
    client = bt.wrap(anthropic.Anthropic(base_url=endpoint, api_key="test", max_retries=0))

    reply = client.messages.create(model="model-x", system="be brief", messages=MESSAGES, max_tokens=100)
    assert reply.content[0].text == "Let me look."
    events = list(client.messages.create(model="model-x", messages=MESSAGES, max_tokens=100, stream=True))
    assert events[-1].type == "message_stop"

    async def main():
        aclient = bt.wrap(anthropic.AsyncAnthropic(base_url=endpoint, api_key="test", max_retries=0))
        again = await aclient.messages.create(model="model-x", messages=MESSAGES, max_tokens=100)
        await aclient.close()
        return again.content[0].text

    assert asyncio.run(main()) == "Let me look."

    whole, streamed, from_async = t.ended()
    assert whole["input"] == {"system": "be brief", "messages": MESSAGES}
    assert whole["output"] == {
        "role": "assistant", "content": "Let me look.", "tool_calls": [{"name": "read", "arguments": {"path": "a.py"}}],
    }
    assert whole["usage"] == {"input": 30, "output": 12, "cached": 500, "cache_write": 70}
    assert whole["meta"] == {"finish_reason": "tool_use"}
    assert streamed["output"]["content"] == "Let me look."
    assert streamed["output"]["tool_calls"] == [{"name": "read", "arguments": '{"path": "a.py"}'}]
    assert streamed["usage"] == {"input": 30, "output": 12, "cached": 500}
    assert from_async["output"] == whole["output"]
