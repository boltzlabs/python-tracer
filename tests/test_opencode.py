"""OpenCode sessions as traces.

tests/data/opencode_export.json is a real `opencode export`: OpenCode 1.18.34
fixing a one-line bug with a hosted model, with the machine's paths rewritten.
"""

import copy
import io
import json
import os
import stat

import pytest

from boltztracer import opencode
from boltztracer.__main__ import load as load_traces, main as view, summarize

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "data", "opencode_export.json")


@pytest.fixture
def session():
    return opencode.load(FIXTURE)


def by_name(records):
    out = {}
    for r in records:
        out.setdefault(r["name"], []).append(r)
    return out


def test_a_real_session_becomes_one_trace(session):
    traces = opencode.convert([session], task="fix-pagination", attempt=2)
    ((trace_id, records),) = traces.items()
    assert len(trace_id) == 32 and all(r["trace_id"] == trace_id for r in records)
    assert all(r["trace"] == {
        "name": "probe2", "task": "fix-pagination", "model": "deepseek-v4-flash", "attempt": 2, "tags": ["opencode"],
    } for r in records)

    names = by_name(records)
    (root,) = names["fix-pagination"]
    assert root["kind"] == "agent" and root["parent_id"] is None and root["status"] == "ok"
    assert root["output"].startswith("Fixed the off-by-one")
    assert root["totals"]["llm_calls"] == 4 and root["totals"]["tool_calls"] == 3 and root["totals"]["errors"] == 0

    turns = [r for r in records if r["kind"] == "step"]
    assert [t["name"] for t in turns] == ["turn 1", "turn 2", "turn 3", "turn 4"]
    assert all(t["parent_id"] == root["span_id"] for t in turns)

    calls = [r for r in records if r["kind"] == "llm"]
    first = calls[0]
    assert first["parent_id"] == turns[0]["span_id"] and first["model"] == "deepseek-v4-flash"
    # The user's request is the prompt of the first call only.
    assert "test_pager.py fails on the last page" in first["input"]["messages"][0]["content"]
    assert "input" not in calls[1]
    assert first["output"]["tool_calls"][0]["name"] == "bash"
    # 231 fresh + 7680 cached in; 58 out plus 7 of reasoning.
    assert first["usage"] == {"input": 231, "output": 65, "cached": 7680, "reasoning": 7}
    assert first["meta"]["finish_reason"] == "tool-calls" and first["meta"]["reasoning"]
    assert "cost" not in first  # OpenCode reported 0: unknown, not free

    tools = [r for r in records if r["kind"] == "tool"]
    assert [t["name"] for t in tools][1:] == ["edit pager.py", "bash python3 test_pager.py"]
    edit, test = tools[1], tools[2]
    assert edit["parent_id"] == turns[1]["span_id"]
    assert edit["input"]["newString"].strip().endswith("n * size + size]")
    assert test["output"] == "2 passed\n" and test["meta"] == {"tool": "bash", "exit": 0}
    # A model call ends when its first tool starts; the tool runs after it.
    assert calls[2]["end_ns"] <= test["start_ns"] <= test["end_ns"] <= turns[2]["end_ns"]
    assert root["start_ns"] <= turns[0]["start_ns"] and root["end_ns"] >= turns[3]["end_ns"]


def test_converting_twice_gives_the_same_trace(session):
    assert opencode.convert([session]) == opencode.convert([copy.deepcopy(session)])
    # Without a task, the session's own title names the run.
    (records,) = opencode.convert([session]).values()
    assert records[0]["trace"]["task"] == "probe2"


def make_child(parent, title="review the fix"):
    child = copy.deepcopy(parent)
    child["info"].update(id="ses_child", parentID=parent["info"]["id"], title=title, agent="general")
    child["messages"] = child["messages"][:2]
    for i, m in enumerate(child["messages"]):
        m["info"]["id"] = "msg_child_%d" % i
        for j, p in enumerate(m["parts"]):
            if p.get("callID"):
                p["callID"] = "call_child_%d_%d" % (i, j)
    return child


def test_a_sub_agent_is_drawn_inside_the_call_that_started_it(session):
    child = make_child(session)
    # The parent's second turn started the sub-agent with the task tool.
    task = next(p for p in session["messages"][2]["parts"] if p["type"] == "tool")
    task["tool"] = "task"
    task["state"]["metadata"]["sessionId"] = "ses_child"

    traces = opencode.convert([child, session], task="t")
    (records,) = traces.values()  # one run, not two
    names = by_name(records)
    (sub,) = names["review the fix"]
    (call,) = [r for r in records if r["kind"] == "tool" and r["name"].startswith("task")]
    assert sub["kind"] == "agent" and sub["parent_id"] == call["span_id"]
    inside = [r for r in records if r["parent_id"] == sub["span_id"]]
    assert [r["name"] for r in inside] == ["turn 1"]
    (root,) = names["t"]
    assert root["totals"]["llm_calls"] == 5  # the sub-agent's call is counted in the run

    # A sub-agent no call points at still lands in its parent's run.
    del task["state"]["metadata"]["sessionId"]
    (records,) = opencode.convert([session, make_child(session)], task="t").values()
    names = by_name(records)
    assert names["review the fix"][0]["parent_id"] == names["t"][0]["span_id"]


def test_a_failed_model_call_marks_the_run(session):
    last = session["messages"][-1]
    last["info"]["error"] = {"name": "APIError", "data": {"message": "upstream returned 529"}}
    (records,) = opencode.convert([session], task="t").values()
    names = by_name(records)
    call = [r for r in records if r["kind"] == "llm"][-1]
    assert call["status"] == "error" and call["error"] == {"type": "APIError", "message": "upstream returned 529"}
    (root,) = names["t"]
    assert root["status"] == "error" and root["error"]["from"] == call["span_id"]
    assert root["totals"]["errors"] == 1  # counted where it happened, once

    tool = next(p for p in session["messages"][1]["parts"] if p["type"] == "tool")
    tool["state"].update(status="error", error="command not found: pytest")
    (records,) = opencode.convert([session], task="t").values()
    failed = [r for r in records if r["kind"] == "tool" and r["status"] == "error"]
    assert failed[0]["error"] == {"type": "ToolError", "message": "command not found: pytest"}


def test_a_run_cut_off_mid_turn_stays_open(session, tmp_path):
    last = session["messages"][3]  # the turn that runs the tests
    session["messages"] = session["messages"][:4]
    del last["info"]["time"]["completed"]
    tool = next(p for p in last["parts"] if p["type"] == "tool")
    tool["state"]["status"] = "running"
    del tool["state"]["time"]["end"]

    paths = opencode.write(opencode.convert([session], task="t"), str(tmp_path))
    (spans,) = load_traces(paths).values()
    summary = summarize(spans)
    assert summary["status"] == "unfinished"
    open_steps = sorted(r["name"] for r in spans.values() if r["status"] == "running")
    assert open_steps == ["bash python3 test_pager.py", "t", "turn 3"]
    # The model had already answered when the tool started.
    assert [r["status"] for r in spans.values() if r["kind"] == "llm"] == ["ok", "ok", "ok"]


def test_secrets_and_long_output_do_not_reach_the_file(session):
    tool = next(p for p in session["messages"][1]["parts"] if p["type"] == "tool")
    tool["state"]["output"] = "OPENAI_API_KEY=sk-" + "a" * 40 + "\n" + "x" * 20000
    (records,) = opencode.convert([session]).values()
    out = next(r for r in records if r["kind"] == "tool")["output"]
    assert "sk-aaaa" not in out and "[redacted]" in out and out.endswith("chars]") and len(out) < 16100


def test_the_command_writes_files_the_viewer_reads(tmp_path, capsys):
    out = tmp_path / "traces"
    bad = tmp_path / "bad.json"
    bad.write_text("not an export")
    code = opencode.main([FIXTURE, str(bad), "--out", str(out), "--task", "fix-pagination", "--model", "my-label", "--attempt", "1"])
    printed = capsys.readouterr()
    assert code == 0 and "TRACES=1" in printed.out and "SKIPPED=1" in printed.out and "bad.json" in printed.err
    (path,) = out.iterdir()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert all(json.loads(line)["trace"]["model"] == "my-label" for line in path.read_text().splitlines())

    buf = io.StringIO()
    assert view([str(out)], out=buf) == 0
    text = buf.getvalue()
    assert "fix-pagination · my-label · attempt 1" in text
    assert "ok  " in text and "4 llm calls  3 tool calls" in text
    assert "    tool bash python3 test_pager.py" in text

    # Running it again replaces the file rather than doubling it.
    opencode.main([FIXTURE, "--out", str(out)])
    assert len(list(out.iterdir())) == 1



def test_a_price_fills_in_the_cost_opencode_did_not_know(session, tmp_path, capsys):
    # deepseek-v4-flash style prices: $0.14 in, $0.28 out, $0.028 cached per million.
    (records,) = opencode.convert([session], price=(0.14, 0.28, 0.028)).values()
    first = next(r for r in records if r["kind"] == "llm")
    # 231 fresh in, 7680 cached, 65 out (58 + 7 reasoning).
    assert first["cost"] == pytest.approx((231 * 0.14 + 7680 * 0.028 + 65 * 0.28) / 1e6)
    root = next(r for r in records if r["parent_id"] is None)
    assert root["totals"]["cost"] == pytest.approx(sum(r["cost"] for r in records if r["kind"] == "llm"))

    # A cost OpenCode did record is kept: it knew that model's price.
    session["messages"][1]["info"]["cost"] = 0.5
    (records,) = opencode.convert([session], price=(0.14, 0.28, 0.028)).values()
    assert next(r for r in records if r["kind"] == "llm")["cost"] == 0.5

    # From the command line, and the viewer shows it.
    out = tmp_path / "t"
    assert opencode.main([FIXTURE, "--out", str(out), "--price", "0.14,0.28,0.028"]) == 0
    buf = io.StringIO()
    view([str(out)], out=buf)
    assert "$0.00" in buf.getvalue() and "  -\n" not in buf.getvalue().splitlines()[1]
    with pytest.raises(SystemExit):
        opencode.main([FIXTURE, "--out", str(out), "--price", "cheap"])

