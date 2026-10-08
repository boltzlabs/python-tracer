"""Two models on the same task, traced, with nothing to sign up for.

    python examples/compare_models.py
    python -m boltztracer .boltz/traces

The "models" here are a stand-in client that answers from a script, so this
runs offline and costs nothing. Swap `FakeClient` for `OpenAI(base_url=...)`
pointed at your own endpoint and the rest of the file stays the same: that is
the whole integration.
"""

import os
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import boltztracer as bt  # noqa: E402

TASK = "fix-pagination"

# What each stand-in model "decides" to do, turn by turn.
SCRIPTS = {
    "model-a": ["read_file", "edit_file", "run_tests", "done"],
    "model-b": ["edit_file", "run_tests", "done"],  # edits without reading first
}


class FakeClient:
    """The same shape as an OpenAI client: `client.chat.completions.create`."""

    def __init__(self):
        self.turn = 0
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, model, messages, **_):
        action = SCRIPTS[model][min(self.turn, len(SCRIPTS[model]) - 1)]
        self.turn += 1
        return NS(
            model=model,
            choices=[NS(finish_reason="stop", message=NS(content=action, tool_calls=None))],
            usage=NS(prompt_tokens=900 + 400 * self.turn, completion_tokens=40, prompt_tokens_details=None),
        )


@bt.tool
def read_file(path):
    return "def page(items, n, size):\n    return items[n * size : n * size + size + 1]\n"


@bt.tool
def edit_file(path, fixed):
    return "patched" if fixed else "patched (guessing)"


@bt.tool
def run_tests(fixed):
    if not fixed:
        raise AssertionError("test_last_page: expected 3 items, got 4")
    return "12 passed"


@bt.agent
def reviewer(client, model, diff):
    """A sub-agent: its model call shows up nested inside its own box."""
    reply = client.chat.completions.create(model=model, messages=[{"role": "user", "content": "review: " + diff}])
    return reply.choices[0].message.content


def run(model):
    client = bt.wrap(FakeClient())
    messages = [{"role": "user", "content": "The last page returns one item too many. Fix it."}]
    read = False
    with bt.trace(task=TASK, model=model, attempt=1) as t:
        while True:
            with bt.step("turn"):
                reply = client.chat.completions.create(model=model, messages=messages)
                action = reply.choices[0].message.content
                messages.append({"role": "assistant", "content": action})
                if action == "read_file":
                    read_file("pager.py")
                    read = True
                elif action == "edit_file":
                    edit_file("pager.py", fixed=read)
                elif action == "run_tests":
                    try:
                        run_tests(fixed=read)
                    except AssertionError as exc:
                        t.fail(f"tests failed: {exc}", "CheckFailed")
                        return
                else:
                    break
        reviewer(client, model, "items[n * size : n * size + size]")
        t.set(output="tests pass")


if __name__ == "__main__":
    bt.init(
        prices={
            "model-a": {"input": 3.00, "output": 15.00},
            "model-b": {"input": 0.25, "output": 1.25},
        }
    )
    for name in SCRIPTS:
        run(name)
    print("wrote traces to .boltz/traces — read them with:  python -m boltztracer")
