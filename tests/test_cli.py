import io

import boltztracer as bt
from boltztracer.__main__ import main


def view(*paths):
    buf = io.StringIO()
    code = main([str(p) for p in paths], out=buf)
    return code, buf.getvalue()


def test_two_runs_are_compared_then_shown_as_trees(tmp_path):
    bt.init(file=str(tmp_path / "traces"), prices={"big": {"input": 10.0, "output": 10.0}})

    @bt.tool
    def run_tests(ok):
        if not ok:
            raise AssertionError("expected 3 items, got 4\nfull diff follows")
        return "passed"

    @bt.agent
    def reviewer():
        with bt.llm(model="big") as s:
            s.usage(input=2000, output=500)

    for model, ok in (("big", True), ("small", False)):
        with bt.trace(task="fix-pagination", model=model, attempt=1):
            try:
                run_tests(ok)
            except AssertionError:
                pass
            reviewer()
    bt.step("hung", kind="tool")  # a third run, killed before it finished

    code, text = view(tmp_path / "traces")
    assert code == 0
    lines = text.splitlines()
    assert lines[0].split() == ["task", "model", "attempt", "status", "time", "llm", "tools", "errors", "tokens", "in/out", "cost"]
    row = next(line for line in lines if line.startswith("fix-pagination  big"))
    assert "2.0k / 500" in row and "$0.0250" in row
    assert any(line.split()[:4] == ["-", "-", "-", "unfinished"] for line in lines)

    assert "tool run_tests" in text and "FAILED AssertionError: expected 3 items, got 4" in text
    assert "full diff follows" not in text  # only the first line of a message
    # The sub-agent's model call is drawn inside it.
    agent_at = next(i for i, line in enumerate(lines) if line.startswith("  agent reviewer"))
    assert lines[agent_at + 1].startswith("    llm chat big")
    assert any(line.split() == ["tool", "hung", "unfinished"] for line in lines)


def test_nothing_to_show(tmp_path):
    assert view(tmp_path / "missing") == (1, f"no such file or directory: {tmp_path / 'missing'}\n")
    (tmp_path / "empty").mkdir()
    assert view(tmp_path / "empty") == (1, "no traces found\n")
