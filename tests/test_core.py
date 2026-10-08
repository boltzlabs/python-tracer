import asyncio
import threading

import pytest

import boltztracer as bt


def test_steps_nest_and_carry_the_trace_labels(out):
    t = out()
    with bt.trace(task="fix-bug", model="m1", attempt=2, tags=["nightly"], suite="pager") as root:
        with bt.step("plan", input={"goal": "fix"}) as plan:
            plan.set(output="read then edit", confidence=0.7)
            with bt.step("inner"):
                pass
        root.set(output="done")

    root_rec, plan_rec, inner_rec = t.one("fix-bug"), t.one("plan"), t.one("inner")
    assert root_rec["parent_id"] is None and root_rec["kind"] == "agent"
    assert plan_rec["parent_id"] == root_rec["span_id"]
    assert inner_rec["parent_id"] == plan_rec["span_id"]
    assert plan_rec["input"] == {"goal": "fix"} and plan_rec["output"] == "read then edit"
    assert plan_rec["meta"] == {"confidence": 0.7}
    assert plan_rec["end_ns"] >= plan_rec["start_ns"]

    labels = {"task": "fix-bug", "model": "m1", "attempt": 2, "tags": ["nightly"], "meta": {"suite": "pager"}}
    assert {r["trace_id"] for r in t.lines()} == {root_rec["trace_id"]}
    assert all(r["trace"] == labels for r in t.lines())
    assert root_rec["totals"]["spans"] == 3 and "totals" not in plan_rec


def test_without_a_run_the_whole_process_is_one(out):
    t = out(task="default-task")

    @bt.tool
    def search(q):
        return q

    with bt.step("first"):
        pass
    search("x")
    bt.flush()

    run = t.lines()[0]  # opened by the first step that needed it
    assert (run["name"], run["kind"], run["parent_id"]) == ("default-task", "agent", None)
    first, tool = t.one("first"), t.one("search")
    assert first["trace_id"] == tool["trace_id"] == run["trace_id"]
    assert first["parent_id"] == tool["parent_id"] == run["span_id"]
    assert first["trace"] == {"task": "default-task"}

    # An explicit run is still its own trace.
    with bt.trace(task="explicit") as explicit:
        pass
    assert explicit.trace_id != run["trace_id"]


def test_a_killed_run_still_leaves_its_open_steps(out):
    t = out()
    bt.step("hung", kind="tool", input="sleep 9999")  # started, never ended
    run, rec = t.lines()
    assert rec["name"] == "hung" and rec["parent_id"] == run["span_id"]
    assert rec["ev"] == "start" and rec["status"] == "running" and rec["end_ns"] is None
    assert rec["input"] == "sleep 9999"


def test_an_exception_is_recorded_where_it_happened_and_reraised(out):
    t = out()
    with pytest.raises(ValueError, match="bad page"):
        with bt.trace(task="t"):
            with bt.step("outer"):
                with bt.step("inner", kind="tool"):
                    raise ValueError("bad page")

    inner, outer, root = t.one("inner"), t.one("outer"), t.one("t")
    assert inner["status"] == outer["status"] == root["status"] == "error"
    assert inner["error"]["type"] == "ValueError" and "bad page" in inner["error"]["stack"]
    # The steps it passed through point at the origin instead of repeating it.
    assert outer["error"] == {"type": "ValueError", "message": "bad page", "from": inner["span_id"]}
    assert root["totals"]["errors"] == 1


def test_fail_marks_a_step_without_raising(out):
    t = out()
    with bt.step("check") as s:
        s.fail("2 tests failed", "CheckFailed")
    rec = t.one("check")
    assert rec["status"] == "error" and rec["error"] == {"type": "CheckFailed", "message": "2 tests failed"}


def test_decorators_cover_every_kind_of_function(out):
    t = out()

    @bt.tool
    def add(a, b=1):
        return a + b

    @bt.agent(name="helper")
    async def helper(x):
        with bt.step("inside-helper"):
            return x * 2

    @bt.observe("numbers")
    def numbers(n):
        for i in range(n):
            with bt.step(f"make-{i}"):
                yield i

    @bt.observe
    async def letters():
        yield "a"
        yield "b"

    async def collect():
        return [x async for x in letters()]

    with bt.trace(task="t"):
        assert add(2, b=3) == 5
        assert asyncio.run(helper(4)) == 8
        seen = []
        for i in numbers(2):
            with bt.step(f"use-{i}"):  # opened by the consumer, between items
                seen.append(i)
        assert seen == [0, 1]
        assert asyncio.run(collect()) == ["a", "b"]

    root = t.one("t")
    add_rec = t.one("add")
    assert add_rec["kind"] == "tool" and add_rec["input"] == {"a": 2, "b": 3} and add_rec["output"] == 5
    helper_rec = t.one("helper")
    assert helper_rec["kind"] == "agent" and helper_rec["output"] == 8
    assert t.one("inside-helper")["parent_id"] == helper_rec["span_id"]
    numbers_rec = t.one("numbers")
    assert numbers_rec["output"] == [0, 1]
    assert t.one("make-1")["parent_id"] == numbers_rec["span_id"]
    assert t.one("use-1")["parent_id"] == root["span_id"]
    assert t.one("letters")["output"] == ["a", "b"]


def test_an_error_inside_a_generator_is_recorded(out):
    t = out()

    @bt.observe
    def broken():
        yield 1
        raise RuntimeError("stream died")

    with pytest.raises(RuntimeError):
        list(broken())
    rec = t.one("broken")
    assert rec["status"] == "error" and rec["output"] == [1]


def test_tokens_become_cost_and_roll_up_to_the_run(out):
    t = out(prices={"m1": {"input": 2.0, "output": 10.0, "cached": 0.5}})
    bt.price("cheap", input=1.0, output=1.0)
    with bt.trace(task="t"):
        with bt.llm(model="m1", input="hi") as a:
            a.usage(input=1000, output=100, cached=2000)
        with bt.llm(model="cheap-2026-01-01") as b:  # dated name → longest prefix
            b.usage(input=1_000_000)
        with bt.llm(model="m1") as c:
            c.usage(input=5, cost=0.25)  # an explicit cost wins
        with bt.llm(model="unknown") as d:
            d.usage(input=7, output=3)
        with bt.step("search", kind="tool"):
            pass

    calls = [r for r in t.ended() if r["kind"] == "llm"]
    assert [r.get("cost") for r in calls] == [pytest.approx(0.004), pytest.approx(1.0), 0.25, None]
    assert calls[0]["usage"] == {"input": 1000, "output": 100, "cached": 2000}
    totals = t.one("t")["totals"]
    assert totals["llm_calls"] == 4 and totals["tool_calls"] == 1
    assert totals["input_tokens"] == 1_001_012 and totals["output_tokens"] == 103
    assert totals["cached_tokens"] == 2000
    assert totals["cost"] == pytest.approx(1.254)


def test_secrets_are_removed_and_long_values_clipped(out):
    t = out(max_chars=20)
    with bt.step("s", input={"key": "sk-" + "a" * 30, "auth": "Bearer " + "b" * 30}) as s:
        s.set(output="x" * 50, blob=b"\x00\x01", obj=object())
    rec = t.one("s")
    assert rec["input"] == {"key": "[redacted]", "auth": "[redacted]"}
    assert rec["output"] == "x" * 20 + "… [+30 chars]"
    assert rec["meta"]["blob"] == "<2 bytes>" and rec["meta"]["obj"].startswith("<object object")


def test_mask_and_capture_off(out):
    t = out(mask=lambda v: "***" if isinstance(v, str) else v)
    with bt.step("masked", input="secret plan") as s:
        s.set(output="secret answer")
    assert t.one("masked")["input"] == "***" and t.one("masked")["output"] == "***"

    t = out(capture=False)
    with bt.step("shape-only", input="secret") as s:
        s.set(output="secret", note="kept")
    rec = t.one("shape-only")
    assert "input" not in rec and "output" not in rec and rec["meta"] == {"note": "kept"}


def test_disabled_does_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    bt.init(enabled=False)

    @bt.tool
    def f(x):
        return x

    with bt.trace(task="t") as root:
        with bt.step("s") as s:
            s.set(output=1).usage(input=1).fail("no")
        assert f(3) == 3
    assert root.trace_id is None and bt.flush() is True
    assert list(tmp_path.iterdir()) == []


def test_another_threads_work_stays_in_the_tree(out):
    t = out()

    def work(name, parent=None):
        with bt.step(name, parent=parent):
            pass

    def in_thread(*args):
        thread = threading.Thread(target=work, args=args)
        thread.start()
        thread.join()

    with bt.trace(task="t") as root:
        with bt.step("turn") as turn:
            in_thread("exact", bt.current())  # handed its parent: exact place
            in_thread("loose")  # not handed one: still this run, under its root

    assert t.one("exact")["parent_id"] == turn.span_id
    assert t.one("loose")["parent_id"] == root.span_id
    assert {t.one("exact")["trace_id"], t.one("loose")["trace_id"]} == {root.trace_id}


def test_values_that_would_never_end_are_cut(out):
    t = out()
    loop = {"rows": list(range(50000))}
    loop["self"] = loop
    with bt.step("big", input=loop):
        pass
    rows = t.one("big")["input"]["rows"]
    assert len(rows) < 21000 and rows[-1].startswith("… [+")


def test_the_environment_configures_it(tmp_path, monkeypatch):
    path = tmp_path / "env.jsonl"
    monkeypatch.setenv("BOLTZ_TRACE_FILE", str(path))
    monkeypatch.setenv("BOLTZ_TRACE_TASK", "from-env")
    monkeypatch.setenv("BOLTZ_TRACE_MODEL", "m9")
    monkeypatch.setenv("BOLTZ_TRACE_ATTEMPT", "3")
    monkeypatch.setenv("BOLTZ_TRACE_ID", "ab" * 16)
    with bt.trace():
        pass
    with bt.trace():
        pass
    from conftest import Trace

    first, second = Trace(path).ended()
    assert first["trace"] == {"task": "from-env", "model": "m9", "attempt": 3}
    assert first["name"] == "from-env"
    # The given id names the first trace only; a second one must not collide.
    assert first["trace_id"] == "ab" * 16 and second["trace_id"] != "ab" * 16


def test_default_is_one_file_per_trace_under_dot_boltz(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with bt.trace(task="a") as a:
        pass
    with bt.trace(task="b") as b:
        pass
    names = sorted(p.name for p in (tmp_path / ".boltz" / "traces").iterdir())
    assert names == sorted([a.trace_id + ".jsonl", b.trace_id + ".jsonl"])
