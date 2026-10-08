"""Turning whatever a step saw into JSON that is safe to keep.

A step's input and output are arbitrary Python: response objects, dataclasses,
bytes, a 4 MB file a tool just read. A trace line has to be plain JSON, small
enough to store, and free of the secrets that pass through an agent's hands.
This module does those three things in one walk, so no value reaches a sink
without all of them having happened.
"""

import dataclasses
import math
import re

__all__ = ["clean"]

# Shapes that are a credential on sight. Deliberately narrow: a pattern that
# also matched ordinary text would quietly corrupt the trace it protects.
_SECRETS = re.compile(
    r"boltzlabs_live_[A-Za-z0-9_\-]{8,}"
    r"|sk-[A-Za-z0-9_\-]{20,}"
    r"|gh[pousr]_[A-Za-z0-9]{30,}"
    r"|AKIA[0-9A-Z]{16}"
    r"|xox[baprs]-[A-Za-z0-9\-]{10,}"
    r"|(?i:bearer)\s+[A-Za-z0-9._\-]{20,}"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"
)

_MAX_DEPTH = 12
# Values visited per field. A structure that refers back to itself, or a list
# of a million rows, stops here instead of growing the trace without bound.
_MAX_NODES = 20000


def clean(value, max_chars=16000, redact=True, mask=None):
    """A JSON-safe copy of ``value``: redacted, clipped, then the caller's mask.

    Never raises. A value that cannot be converted becomes a marker string,
    because losing one field is better than losing the agent's run.
    """
    try:
        out = _walk(value, max_chars, redact, 0, [_MAX_NODES])
        return mask(out) if mask is not None else out
    except Exception:
        return "<unserializable>"


def _walk(v, cap, redact, depth, left):
    left[0] -= 1
    if v is None or isinstance(v, (bool, int)):
        return v
    if isinstance(v, float):
        return v if math.isfinite(v) else repr(v)
    if isinstance(v, str):
        # Redact before clipping, so a key that straddles the cut is not left
        # half-visible at the end of the string.
        if redact:
            v = _SECRETS.sub("[redacted]", v)
        if len(v) > cap:
            v = v[:cap] + f"… [+{len(v) - cap} chars]"
        return v
    if isinstance(v, (bytes, bytearray, memoryview)):
        return f"<{len(v)} bytes>"
    if depth >= _MAX_DEPTH:
        return _walk(repr(v), cap, redact, depth, left)
    if isinstance(v, dict):
        out = {}
        for i, (k, x) in enumerate(v.items()):
            if left[0] <= 0:
                out["…"] = f"[+{len(v) - i} more]"
                break
            out[str(k)] = _walk(x, cap, redact, depth + 1, left)
        return out
    if isinstance(v, (list, tuple, set, frozenset)):
        out = []
        for i, x in enumerate(v):
            if left[0] <= 0:
                out.append(f"… [+{len(v) - i} more]")
                break
            out.append(_walk(x, cap, redact, depth + 1, left))
        return out

    # Model SDK responses are pydantic objects; their dump is the readable form.
    dump = getattr(v, "model_dump", None)
    if callable(dump):
        try:
            return _walk(dump(), cap, redact, depth + 1, left)
        except Exception:
            pass
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        try:
            return _walk(dataclasses.asdict(v), cap, redact, depth + 1, left)
        except Exception:
            pass
    return _walk(repr(v), cap, redact, depth, left)
