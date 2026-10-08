"""Tracing a program that knows nothing about the tracer.

Each case is a real Python process, because that is what this feature is: a
line that runs before the program does. The programs use the real OpenAI and
Anthropic packages against a server on this machine.
"""

import json
import os
import subprocess
import sys

import pytest

from test_real_sdks import endpoint  # noqa: F401  (a fixture)

pytest.importorskip("openai")
pytest.importorskip("anthropic")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHAT = """
from openai import OpenAI
client = OpenAI(base_url=URL + "/v1", api_key="k")
r = client.chat.completions.create(model="m1", messages=[{"role": "user", "content": "hi"}])
print(r.choices[0].message.content)
"""


def run(tmp_path, code, endpoint, env=None, args=(), prelude="import boltztracer.auto\n"):
    """Run a program in its own process; return its output and its traces."""
    out = tmp_path / "traces"
    script = tmp_path / "agent.py"
    script.write_text(prelude + "URL = %r\n" % endpoint + code)
    full = {**os.environ, "PYTHONPATH": ROOT, "BOLTZ_TRACE_FILE": str(out)}
    full.pop("BOLTZ_TRACE_TASK", None)
    full.update(env or {})
    done = subprocess.run([sys.executable, *args, str(script)], capture_output=True, text=True, env=full, timeout=60, cwd=tmp_path)
    assert done.returncode == 0, done.stderr
    traces = []
    if out.is_dir():
        for name in sorted(os.listdir(out)):
            with open(out / name, encoding="utf-8") as fh:
                traces.append([json.loads(line) for line in fh if line.strip()])
    return done.stdout, traces


def ended(trace, kind):
    return [r for r in trace if r["ev"] == "end" and r["kind"] == kind]


def test_a_program_is_traced_without_knowing_it(tmp_path, endpoint):
    out, traces = run(tmp_path, CHAT, endpoint)
    assert out.strip() == "Hello"
    (trace,) = traces
    (call,) = ended(trace, "llm")
    assert call["model"] == "m1-2026-01-01" or call["model"] == "m1"
    assert call["input"]["messages"] == [{"role": "user", "content": "hi"}]
    assert call["output"]["content"] == "Hello" and call["usage"]["output"] == 20
    # The run is the process, named after the script, and it was closed.
    (root,) = ended(trace, "agent")
    assert root["name"] == "agent.py" and root["trace"]["task"] == "agent.py" and root["totals"]["llm_calls"] == 1
    assert call["parent_id"] == root["span_id"]


def test_it_does_not_matter_which_is_imported_first(tmp_path, endpoint):
    # The SDK first, then the line: clients made afterwards are still recorded.
    _, traces = run(tmp_path, "import boltztracer.auto\n" + CHAT, endpoint, prelude="import openai\n")
    assert len(ended(traces[0], "llm")) == 1


def test_anthropic_async_and_streams_are_traced_too(tmp_path, endpoint):
    code = """
import asyncio
from anthropic import Anthropic, AsyncAnthropic
from openai import AsyncOpenAI

sync = Anthropic(base_url=URL, api_key="k")
sync.messages.create(model="model-x", max_tokens=64, messages=[{"role": "user", "content": "hi"}])

async def go():
    a = AsyncAnthropic(base_url=URL, api_key="k")
    await a.messages.create(model="model-x", max_tokens=64, messages=[{"role": "user", "content": "hi"}])
    o = AsyncOpenAI(base_url=URL + "/v1", api_key="k")
    stream = await o.chat.completions.create(model="m1", stream=True, messages=[{"role": "user", "content": "hi"}])
    async for _ in stream:
        pass

asyncio.run(go())
"""
    _, traces = run(tmp_path, code, endpoint)
    (trace,) = traces  # one process, one run
    calls = ended(trace, "llm")
    assert len(calls) == 3
    assert sorted(c["model"] for c in calls) == ["m1", "model-x", "model-x"]
    assert all(c["output"]["content"] for c in calls)


def test_a_program_that_calls_no_model_leaves_nothing(tmp_path, endpoint):
    out, traces = run(tmp_path, "from openai import OpenAI\nOpenAI(base_url=URL, api_key='k')\nprint('done')\n", endpoint)
    assert out.strip() == "done" and traces == []
    # Nor one that never imports an SDK at all.
    out, traces = run(tmp_path, "print(1 + 1)\n", endpoint)
    assert out.strip() == "2" and traces == []


def test_it_can_be_switched_off_and_named(tmp_path, endpoint):
    _, traces = run(tmp_path, CHAT, endpoint, env={"BOLTZ_TRACE": "0"})
    assert traces == []
    _, traces = run(tmp_path, CHAT, endpoint, env={"BOLTZ_TRACE_TASK": "nightly", "BOLTZ_TRACE_MODEL": "m1"})
    assert ended(traces[0], "agent")[0]["trace"] == {"task": "nightly", "model": "m1"}


def test_a_program_that_wraps_its_own_client_is_not_recorded_twice(tmp_path, endpoint):
    code = """
import boltztracer as bt
from openai import OpenAI
client = bt.wrap(OpenAI(base_url=URL + "/v1", api_key="k"))
with bt.trace(task="mine", model="m1"):
    client.chat.completions.create(model="m1", messages=[{"role": "user", "content": "hi"}])
"""
    _, traces = run(tmp_path, code, endpoint)
    (trace,) = traces
    assert len(ended(trace, "llm")) == 1 and ended(trace, "agent")[0]["trace"]["task"] == "mine"


def test_one_line_in_site_packages_traces_every_program(tmp_path, endpoint):
    # What turning tracing on for a sandbox does: a .pth file, read at the
    # start of every Python process, with no line in the program at all.
    site = tmp_path / "site"
    site.mkdir()
    (site / "boltztracer.pth").write_text("import boltztracer.auto\n")
    starter = "import site, runpy, sys; site.addsitedir(%r); sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')" % str(site)
    out_dir = tmp_path / "traces"
    script = tmp_path / "plain.py"
    script.write_text("URL = %r\n" % endpoint + CHAT)
    env = {**os.environ, "PYTHONPATH": ROOT, "BOLTZ_TRACE_FILE": str(out_dir)}
    env.pop("BOLTZ_TRACE_TASK", None)
    done = subprocess.run([sys.executable, "-c", starter, str(script)], capture_output=True, text=True, env=env, timeout=60)
    assert done.returncode == 0 and done.stdout.strip() == "Hello", done.stderr
    (name,) = os.listdir(out_dir)
    with open(out_dir / name, encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    assert len(ended(records, "llm")) == 1


def test_a_broken_sdk_does_not_break_the_program(tmp_path, endpoint):
    # Something calling itself openai, with none of what is expected in it.
    fake = tmp_path / "fake"
    (fake / "openai").mkdir(parents=True)
    (fake / "openai" / "__init__.py").write_text("OpenAI = 7\nclass AsyncOpenAI:\n    __slots__ = ()\n")
    out, traces = run(tmp_path, "import openai\nprint(openai.OpenAI, openai.AsyncOpenAI())\n", endpoint, env={"PYTHONPATH": str(fake) + os.pathsep + ROOT})
    assert out.startswith("7 ") and traces == []
