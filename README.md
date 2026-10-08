# boltztracer

See what an agent did, step by step: model calls, tool calls, sub-agents, what
each cost, and where it went wrong. Run the same task under several models and
the traces line up for comparison.

No dependencies. Python 3.9+. Not on PyPI yet — install from git:

```bash
pip install git+https://github.com/boltzlabs/python-tracer.git
# or
uv add git+https://github.com/boltzlabs/python-tracer.git
```

```python
import boltztracer as bt
from openai import OpenAI

client = bt.wrap(OpenAI(base_url="https://my-endpoint/v1"))   # model calls

@bt.tool                                                      # tool calls
def run_tests(path): ...

@bt.agent                                                     # sub-agents
def reviewer(diff): ...

with bt.trace(task="fix-pagination", model="my-model", attempt=1):
    with bt.step("plan"):
        ...
    run_tests("tests/")
    reviewer(diff)
```

```bash
python -m boltztracer
```

```
task            model    attempt  status  time  llm  tools  errors  tokens in/out  cost
fix-pagination  model-a  1        ok      2ms   5    3      0       10.5k / 200    $0.0345
fix-pagination  model-b  1        error   2ms   2    2      2       3.0k / 80      $0.0009

trace 0091c0c2f46f99b88e958ba65fb95590  fix-pagination · model-b · attempt 1
error  2ms  2 llm calls  2 tool calls  2 errors  3.0k in / 80 out  $0.0009

agent fix-pagination  2ms  FAILED CheckFailed: tests failed: test_last_page: expected 3 items, got 4
  step turn           0ms
    llm chat model-b  0ms  1.3k in / 40 out  $0.0004
    tool edit_file    0ms
  step turn           1ms
    llm chat model-b  0ms  1.7k in / 40 out  $0.0005
    tool run_tests    1ms  FAILED AssertionError: test_last_page: expected 3 items, got 4
```

That is the output of `examples/compare_models.py`, which runs offline with a
stand-in client (hence the millisecond timings). The tree for `model-a` is
printed the same way.

## What a trace is

One trace is one run of one model on one task. It is a tree of steps, and each
step has a kind:

| Kind | What it is | How you get one |
| --- | --- | --- |
| `agent` | the run itself, or a sub-agent inside it | `bt.trace(...)`, `@bt.agent` |
| `llm` | one call to a model: prompt, answer, tokens, cost | `bt.wrap(client)`, `bt.llm(...)` |
| `tool` | one tool the agent used | `@bt.tool` |
| `step` | anything else worth a box of its own | `bt.step(...)`, `@bt.observe` |

A step opened while another is open becomes its child. Nothing is passed
around; the tree follows the code. Decorators work on plain functions,
coroutines, generators and async generators.

`bt.trace()` starts a run. `task`, `model` and `attempt` on it are what make
runs comparable: the same `task` under different `model`s is a comparison.

If you never call `bt.trace()`, the whole process is one run, labelled from
`bt.init()` or the environment. So an agent that only wraps its client and
decorates its tools still produces one tree.

## Model calls

`bt.wrap(client)` records every `create` call on an OpenAI-style client
(`chat.completions`, `responses`) or an Anthropic-style one (`messages`): sync
or async, streaming or not. Most self-hosted and third-party endpoints speak
the chat-completions shape, so they work too. It reads by attribute name and
imports no SDK.

Helpers that bypass `create` (`.parse()`, `.stream()`) are not recorded. Wrap
those, or any other client, by hand:

```python
with bt.llm(model="my-model", input=messages) as s:
    reply = call_model(messages)
    s.set(output=reply.text)
    s.usage(input=reply.tokens_in, output=reply.tokens_out)
```

## OpenCode

OpenCode is not a Python program, so it cannot import this package. It keeps
its own record of a session instead, and that converts into the same trace
files:

```bash
opencode export <session-id> > session.json
python -m boltztracer.opencode session.json --task fix-pagination
python -m boltztracer
```

The session is the run, each turn is a step holding its model call and the
tools it ran, and a sub-agent's session is drawn inside the call that started
it. Give it several files at once and sub-agent sessions find their parents.
`--model` and `--attempt` label the run for comparison; without `--model` the
session's own model is used. OpenCode records a cost only for models it has a
price list for; `--price 3,15,0.3` (input, output and cached input, in USD per
million tokens) prices the calls it left without one, and never replaces a
cost it did record. `boltztracer/opencode.py` imports nothing but the standard
library, so it also runs as a plain script where the package is not installed.

## Cost

Tokens are always recorded. Cost needs a price, in USD per million tokens:

```python
bt.init(prices={"my-model": {"input": 2.50, "output": 10.00, "cached": 0.25}})
```

A dated name from the endpoint (`my-model-2026-01-01`) uses the price of the
longest name it starts with. `s.usage(cost=0.012)` overrides the table.

## When it goes wrong

An exception is recorded on the step it was raised in, with its stack, and then
re-raised untouched. The steps it passes through on the way out are marked
failed and point at that step, so the tree shows one origin, not five copies.
`s.fail("2 tests failed")` marks a step failed without raising.

Every step is written when it starts and again when it ends. A run killed by a
timeout leaves its open steps in the file, marked `running`, with their input —
which is how you see what it was doing when it was killed.

## Threads

A step opened in another thread cannot see what was open where the thread was
started. It still lands in the run, directly under the run's root. To put it
exactly where it belongs, hand it its parent:

```python
parent = bt.current()
threading.Thread(target=lambda: work(parent)).start()

def work(parent):
    with bt.step("in-thread", parent=parent):
        ...
```

`asyncio` tasks need nothing; they inherit it. With several runs open at once
in one process, a thread's steps go to the process-wide run unless given a
parent.

## Where traces go

By default, `.boltz/traces/<trace-id>.jsonl` under the current directory, one
JSON line per event, written as it happens. No network is involved.

```python
bt.init(file="run.jsonl")          # one file for everything
bt.init(file="traces/")            # a directory, one file per trace
bt.init(endpoint="https://collector.example.com/v1/traces", headers={...})
```

`endpoint` sends finished steps as OTLP/HTTP JSON, the OpenTelemetry wire
format. Sending happens on a background thread and never blocks or breaks the
agent; call `bt.flush()` before exit if you need to be sure it left. A hosted
Boltz endpoint and dashboard for these traces do not exist yet.

## OpenTelemetry

A trace here is shaped the way OpenTelemetry shapes one: a 32-digit trace id, a
16-digit id for each step and its parent, nanosecond start and end times, and a
status. The file on disk is this package's own (one JSON line per event, which
is what makes it readable half-written); `boltztracer.otel` converts to and
from the OpenTelemetry form, and the two conversions are tested as inverses.

Out, steps follow the GenAI semantic conventions:

| Step | Span name | `gen_ai.operation.name` |
| --- | --- | --- |
| model call | `chat <model>` | `chat` |
| tool | `execute_tool <name>` | `execute_tool` |
| run or sub-agent | `invoke_agent <name>` | `invoke_agent` |
| step | `<name>` | none |

with `gen_ai.request.model`, `gen_ai.usage.input_tokens` and `output_tokens`
(and the cache counts), `gen_ai.tool.name`, `gen_ai.agent.name`, `input.value`
and `output.value`, and a failure as an error status, `error.type` and an
`exception` event. Cost, and the task, model and attempt a run is compared by,
have no OpenTelemetry name and go under `boltz.*`.

In, spans from an agent instrumented with OpenTelemetry become the same trace
files, so everything that reads those files works on them:

```bash
python -m boltztracer.otel spans.json --out .boltz/traces
```

```python
from boltztracer import otel
traces = otel.from_otlp(request)        # {trace_id: [records]}
otel.write(traces, ".boltz/traces")
```

Spans that use OpenInference or OpenLLMetry attribute names are recognised
too, and ids sent as base64 are read as the hex they stand for.

A run joins a trace that another program started. `TRACEPARENT` in the
environment (the W3C trace context, which is how OpenTelemetry hands a trace
to a child process) makes the first run of this process part of that trace,
under the step that started it. `bt.traceparent()` gives the open step in the
same form, to pass on:

```python
subprocess.run(cmd, env={**os.environ, "TRACEPARENT": bt.traceparent()})
```

This package does not use the OpenTelemetry SDK and does not need it.

## What is kept

Inputs and outputs are kept, with obvious credentials (API keys, bearer tokens,
private keys) replaced by `[redacted]` and long strings clipped at 16,000
characters. A trace still contains your prompts, code and tool output: treat
the files accordingly. The default directory is in this package's `.gitignore`;
add `.boltz/` to yours.

```python
bt.init(capture=False)             # shape, timing and tokens only
bt.init(mask=my_function)          # applied to every input and output
bt.init(max_chars=100_000)
```

## Environment

| Variable | Effect |
| --- | --- |
| `BOLTZ_TRACE=0` | turn tracing off; every call becomes a no-op |
| `BOLTZ_TRACE_FILE` | where to write |
| `BOLTZ_TRACE_ENDPOINT` | OTLP/HTTP traces URL to also send to |
| `BOLTZ_TRACE_TASK`, `BOLTZ_TRACE_MODEL`, `BOLTZ_TRACE_ATTEMPT` | labels for traces that set none |
| `BOLTZ_TRACE_ID` | 32 hex characters: the id of the first trace this process starts |
| `TRACEPARENT` | a W3C trace context: the first run joins that trace, under that step |
| `BOLTZ_TRACE_CAPTURE=0` | do not keep inputs and outputs |
| `BOLTZ_TRACE_MAX_CHARS` | clip length |

Arguments to `bt.init()` win over the environment. `BOLTZLABS_API_KEY` is sent
as a bearer token only to a Boltz endpoint, never to another collector.

## The file

Each line is one event for one step.

| Field | |
| --- | --- |
| `ev` | `start` or `end`; the later line for a `span_id` is the complete one |
| `trace_id`, `span_id`, `parent_id` | 32, 16 and 16 hex characters; `parent_id` is `null` on the root |
| `name`, `kind` | |
| `start_ns`, `end_ns` | Unix nanoseconds; `end_ns` is `null` until the step ends |
| `status` | `running`, `ok` or `error` |
| `trace` | the trace's labels, repeated on every line: `name`, `task`, `model`, `attempt`, `tags`, `project`, `meta` |
| `input`, `output` | what went in and came out |
| `model`, `usage`, `cost` | on model calls. `usage.input` excludes cached tokens; `usage.cached` counts those |
| `error` | `type`, `message`, and `stack` where it was raised or `from` (a `span_id`) where it passed through |
| `meta` | anything passed to `s.set(...)` |
| `totals` | on the root's `end` line only: steps, model and tool calls, errors, tokens, cost |

## Tests

```bash
pip install -e ".[dev]"
python -m pytest -q
```

`tests/test_real_sdks.py` drives the real `openai` and `anthropic` packages
against a server on your machine; it needs no key and skips when they are not
installed (`pip install openai anthropic`).
