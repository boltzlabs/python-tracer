"""Every test gets its own trace file and a tracer with no memory of the last."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import boltztracer as bt  # noqa: E402
from boltztracer import _core  # noqa: E402


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    for name in list(os.environ):
        if name.startswith("BOLTZ_TRACE") or name.startswith("BOLTZLABS_") or name == "TRACEPARENT":
            monkeypatch.delenv(name)
    _core._cfg = None
    _core._totals.clear()
    del _core._open[:]
    yield
    # Close the process-wide run now, while its file still exists, rather than
    # leaving it for interpreter exit.
    if _core._cfg is not None and _core._cfg.implicit is not None:
        _core._cfg.implicit.end()
    _core._cfg = None


class Trace:
    """Reads back what the tracer wrote."""

    def __init__(self, path):
        self.path = str(path)

    def lines(self):
        with open(self.path, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def ended(self):
        return [r for r in self.lines() if r["ev"] == "end"]

    def one(self, name):
        found = [r for r in self.ended() if r["name"] == name]
        assert len(found) == 1, f"{len(found)} finished steps named {name!r}"
        return found[0]


@pytest.fixture
def out(tmp_path):
    """A tracer writing to one file, and a reader for it."""
    path = tmp_path / "trace.jsonl"

    def start(**kwargs):
        bt.init(file=str(path), **kwargs)
        return Trace(path)

    return start
