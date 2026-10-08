"""boltztracer — see what an agent did, step by step.

    import boltztracer as bt

    client = bt.wrap(OpenAI(base_url="https://my-endpoint/v1"))   # model calls

    @bt.tool                                                      # tool calls
    def run_tests(path): ...

    @bt.agent                                                     # sub-agents
    def reviewer(diff): ...

    with bt.trace(task="fix-pagination", model="my-model"):       # one run
        with bt.step("plan"):
            ...
        run_tests("tests/")

Every step lands in ``.boltz/traces/<trace-id>.jsonl`` as it happens: what went
in, what came out, how long it took, what it cost, and where it failed. Run the
same task under several models and the traces line up for comparison.

    python -m boltztracer            # read them in a terminal

There is nothing to configure and nothing to install besides this package.
``bt.init()`` exists for when the defaults are not what you want.
"""

from ._core import (
    Span,
    agent,
    current,
    flush,
    init,
    llm,
    observe,
    price,
    step,
    tool,
    trace,
    traceparent,
)
from ._wrap import wrap

__version__ = "0.3.0"

__all__ = [
    "Span",
    "agent",
    "current",
    "flush",
    "init",
    "llm",
    "observe",
    "price",
    "step",
    "tool",
    "trace",
    "traceparent",
    "wrap",
    "__version__",
]
