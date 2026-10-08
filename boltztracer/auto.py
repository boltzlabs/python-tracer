"""Tracing with nothing added to the program.

    import boltztracer.auto

After that line, every OpenAI and Anthropic client the program creates records
its model calls, exactly as if each had been passed through ``bt.wrap()``. The
line does not have to be in the program either. A ``.pth`` file holding it, in
the interpreter's site-packages, runs it at the start of every Python process:

    echo "import boltztracer.auto" > "$(python -c 'import site; print(site.getsitepackages()[0])')/boltztracer.pth"

which is how a sandbox is made to record whatever runs in it.

Nothing is written for a process that never calls a model. A process that does
is one run: its trace is named after the script, unless ``BOLTZ_TRACE_TASK``
says otherwise. Set ``BOLTZ_TRACE=0`` to switch it off for one command.

This module must never be the reason a program fails to start, so everything it
does is inside a guard, and it imports neither SDK itself: it waits for the
program to.
"""

import functools
import importlib.abc
import importlib.util
import os
import sys

from ._wrap import wrap

__all__ = ["install"]

# The classes whose instances make model calls. A name that an SDK version does
# not have is skipped.
_CLIENTS = {
    "openai": ("OpenAI", "AsyncOpenAI", "AzureOpenAI", "AsyncAzureOpenAI"),
    "anthropic": ("Anthropic", "AsyncAnthropic", "AnthropicBedrock", "AsyncAnthropicBedrock", "AnthropicVertex", "AsyncAnthropicVertex"),
}

_installed = False


def _patch(module, names):
    """Make every client the module's classes create a recorded one."""
    for name in names:
        cls = getattr(module, name, None)
        init = getattr(cls, "__init__", None)
        if cls is None or init is None or getattr(init, "_boltz", False):
            continue

        def make(init):
            @functools.wraps(init)
            def traced_init(self, *args, **kwargs):
                init(self, *args, **kwargs)
                try:
                    wrap(self)
                except Exception:
                    pass  # an unrecorded client still works

            traced_init._boltz = True
            return traced_init

        try:
            cls.__init__ = make(init)
        except Exception:
            pass


class _Loader(importlib.abc.Loader):
    """Runs a module as its own loader would, then patches it."""

    def __init__(self, inner, names):
        self._inner, self._names = inner, names

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        try:
            _patch(module, self._names)
        except Exception:
            pass

    def __getattr__(self, name):
        # Anything else asked of a loader (resources, source) is the real one's.
        return getattr(self._inner, name)


class _Finder(importlib.abc.MetaPathFinder):
    """Notices the SDKs being imported, however late the program does it."""

    _busy = False

    def find_spec(self, fullname, path=None, target=None):
        names = _CLIENTS.get(fullname)
        if names is None or _Finder._busy:
            return None
        _Finder._busy = True
        try:
            spec = importlib.util.find_spec(fullname)
        except Exception:
            spec = None
        finally:
            _Finder._busy = False
        if spec is None or spec.loader is None:
            return None
        spec.loader = _Loader(spec.loader, names)
        return spec


def install():
    """Start recording every client this process creates. Safe to call twice."""
    global _installed
    if _installed or os.environ.get("BOLTZ_TRACE", "").strip().lower() in ("0", "false", "off", "no"):
        return
    _installed = True
    # Where there is a sandbox workspace, every program writes to the one
    # place in it, whatever directory it was started from; and what it calls
    # itself, when nobody named the task, is the script it is.
    if "BOLTZ_TRACE_FILE" not in os.environ and "BOLTZ_TRACE_ENDPOINT" not in os.environ and os.path.isdir("/workspace"):
        os.environ["BOLTZ_TRACE_FILE"] = "/workspace/.boltz/traces"
    if "BOLTZ_TRACE_TASK" not in os.environ:
        script = os.path.basename((sys.argv[0:1] or [""])[0] or "")
        if script and script not in ("-c", "-m") and not script.startswith("-"):
            os.environ["BOLTZ_TRACE_TASK"] = script
    # An SDK the program imported before this line is patched now; one it
    # imports later is patched as it arrives.
    for name, names in _CLIENTS.items():
        module = sys.modules.get(name)
        if module is not None:
            _patch(module, names)
    sys.meta_path.insert(0, _Finder())


try:
    install()
except Exception:
    pass
