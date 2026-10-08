"""OpenCode sessions as traces.

OpenCode keeps its own record of a session: every model call with its tokens,
every tool call with its input and output, and the sub-agents it started. This
turns `opencode export` output into the same trace files the rest of this
package writes, so a run done by OpenCode is read, compared and drawn exactly
like one done by an agent that imports the tracer.

    opencode export <session-id> > session.json
    python -m boltztracer.opencode session.json --task fix-pagination

Each top-level session becomes one trace:

    agent  the session
      step   turn 1
        llm    the model call that produced the turn
        tool   each tool the model asked for, in the order they ran
          agent  a sub-agent's own session, when the tool started one

This file imports nothing from the package and nothing outside the standard
library, on purpose: it is also copied into sandboxes and run there as a plain
script, where the package is not installed.
"""

import argparse
import hashlib
import json
import os
import re
import sys

__all__ = ["convert", "load", "write", "main"]

MAX_CHARS = 16000

# The same shapes boltztracer redacts everywhere else; kept in step by hand
# because this file has to stand alone.
_SECRETS = re.compile(
    r"boltzlabs_live_[A-Za-z0-9_\-]{8,}"
    r"|brt_[A-Za-z0-9_\-.]{16,}"
    r"|sk-[A-Za-z0-9_\-]{20,}"
    r"|gh[pousr]_[A-Za-z0-9]{30,}"
    r"|AKIA[0-9A-Z]{16}"
    r"|xox[baprs]-[A-Za-z0-9\-]{10,}"
    r"|(?i:bearer)\s+[A-Za-z0-9._\-]{20,}"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"
)


def _clean(v, depth=0):
    if isinstance(v, str):
        v = _SECRETS.sub("[redacted]", v)
        return v if len(v) <= MAX_CHARS else v[:MAX_CHARS] + "… [+%d chars]" % (len(v) - MAX_CHARS)
    if isinstance(v, dict):
        return {str(k): _clean(x, depth + 1) for k, x in v.items()} if depth < 12 else "<nested>"
    if isinstance(v, (list, tuple)):
        return [_clean(x, depth + 1) for x in v] if depth < 12 else "<nested>"
    return v


def _hex(text, length):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def _ns(ms):
    """OpenCode stamps milliseconds; traces are in nanoseconds."""
    return int(ms) * 1_000_000 if isinstance(ms, (int, float)) and ms > 0 else None


def _g(obj, *path):
    for key in path:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(key)
    return obj


def load(path):
    """One `opencode export` file. The command may print a line before the JSON."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    start = text.find("{")
    if start < 0:
        raise ValueError("%s holds no JSON" % path)
    doc = json.loads(text[start:])
    if not isinstance(_g(doc, "info"), dict) or not isinstance(doc.get("messages"), list):
        raise ValueError("%s is not an OpenCode session export" % path)
    return doc


class _Trace:
    """The records of one trace, and the running totals its root reports."""

    def __init__(self, trace_id, labels, price=None):
        self.trace_id = trace_id
        self.labels = labels
        self.price = price
        self.records = []
        self.totals = {
            "spans": 0, "llm_calls": 0, "tool_calls": 0, "errors": 0,
            "input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "cost": None,
        }

    def add(self, key, parent_id, name, kind, start_ns, end_ns, **fields):
        """Append one step. A step with no end is still open: the run was cut off."""
        if start_ns is None:
            start_ns = end_ns or 0
        if end_ns is not None and end_ns < start_ns:
            end_ns = start_ns
        error = fields.pop("error", None)
        rec = {
            "v": 1,
            "ev": "end" if end_ns is not None else "start",
            "trace_id": self.trace_id,
            "span_id": _hex(self.trace_id + ":" + key, 16),
            "parent_id": parent_id,
            "name": str(name)[:200],
            "kind": kind,
            "start_ns": start_ns,
            "end_ns": end_ns,
            "status": "running" if end_ns is None else ("error" if error else "ok"),
            "trace": self.labels,
        }
        for field, value in fields.items():
            if field == "meta":
                value = {k: v for k, v in value.items() if v not in (None, "", [], {})}
            if value not in (None, "", [], {}):
                rec[field] = _clean(value) if field in ("input", "output", "meta") else value
        if error:
            rec["error"] = _clean(error)
        self.records.append(rec)

        t = self.totals
        t["spans"] += 1
        t["llm_calls"] += kind == "llm"
        t["tool_calls"] += kind == "tool"
        if error and "from" not in error:
            t["errors"] += 1
        usage = fields.get("usage") or {}
        t["input_tokens"] += usage.get("input", 0)
        t["output_tokens"] += usage.get("output", 0)
        t["cached_tokens"] += usage.get("cached", 0) + usage.get("cache_write", 0)
        if fields.get("cost") is not None:
            t["cost"] = round((t["cost"] or 0.0) + fields["cost"], 8)
        return rec


def _text(parts, kind):
    return "".join(p.get("text") or "" for p in parts if p.get("type") == kind)


def _usage(tokens):
    """OpenCode's counts in the tracer's terms: input never includes cached."""
    if not isinstance(tokens, dict):
        return None
    reasoning = int(tokens.get("reasoning") or 0)
    usage = {
        "input": int(tokens.get("input") or 0),
        # Reasoning is output the model was billed for; OpenCode lists it apart.
        "output": int(tokens.get("output") or 0) + reasoning,
        "cached": int(_g(tokens, "cache", "read") or 0),
        "cache_write": int(_g(tokens, "cache", "write") or 0),
        "reasoning": reasoning,
    }
    usage = {k: v for k, v in usage.items() if v}
    return usage or None


def _failure(err, fallback="Error"):
    if not err:
        return None
    if isinstance(err, str):
        return {"type": fallback, "message": err}
    message = _g(err, "data", "message") or err.get("message") or json.dumps(err)[:500]
    return {"type": str(err.get("name") or fallback), "message": str(message)}


def _child_id(part):
    """The session a `task` tool call started, wherever this version put it."""
    meta = _g(part, "state", "metadata") or {}
    for key in ("sessionId", "sessionID", "session_id"):
        if isinstance(meta.get(key), str):
            return meta[key]
    return None


def _session(trace, sess, parent_id, by_id, children_of, used, name=None):
    """Add one session and everything under it. Returns its root record."""
    info = sess["info"]
    sid = info.get("id") or "session"
    used.add(sid)
    start = _ns(_g(info, "time", "created"))
    last = _ns(_g(info, "time", "updated"))
    root_key = "session:" + sid
    root_id = _hex(trace.trace_id + ":" + root_key, 16)

    asked = None  # what the user last said, the prompt of the next model call
    turn = 0
    open_run = False
    failure = None
    final_text = None

    for msg in sess.get("messages") or []:
        minfo, parts = msg.get("info") or {}, msg.get("parts") or []
        if minfo.get("role") == "user":
            asked = _text(parts, "text") or asked
            continue
        if minfo.get("role") != "assistant":
            continue
        turn += 1
        mid = minfo.get("id") or "turn-%d" % turn
        t0 = _ns(_g(minfo, "time", "created")) or start
        t1 = _ns(_g(minfo, "time", "completed"))
        tools = [p for p in parts if p.get("type") == "tool"]
        tools.sort(key=lambda p: _g(p, "state", "time", "start") or 0)
        error = _failure(minfo.get("error"), "ModelError")

        turn_key = "turn:" + mid
        turn_id = _hex(trace.trace_id + ":" + turn_key, 16)
        open_run = t1 is None and not error
        if error:
            failure = {"type": error["type"], "message": error["message"], "from": _hex(trace.trace_id + ":llm:" + mid, 16)}
        trace.add(
            turn_key, root_id, "turn %d" % turn, "step", t0, None if open_run else (t1 or last or t0),
            error=dict(failure) if error else None,
        )

        # The model has answered by the time its first tool starts; what is left
        # of the turn after that is the tools running.
        first_tool = _ns(_g(tools[0], "state", "time", "start")) if tools else None
        llm_end = first_tool or t1
        if llm_end is None and not open_run:
            llm_end = last or t0
        text = _text(parts, "text")
        if text:
            final_text = text
        model = minfo.get("modelID") or _g(info, "model", "id")
        usage = _usage(minfo.get("tokens"))
        cost = minfo.get("cost")
        cost = float(cost) if isinstance(cost, (int, float)) and cost > 0 else None
        if cost is None and trace.price and usage:
            # OpenCode did not know what this model costs; the caller does.
            p_in, p_out, p_cached = trace.price
            cost = round(
                (
                    (usage.get("input", 0) + usage.get("cache_write", 0)) * p_in
                    + usage.get("cached", 0) * p_cached
                    + usage.get("output", 0) * p_out
                )
                / 1e6,
                8,
            )
        trace.add(
            "llm:" + mid, turn_id, "chat %s" % model if model else "llm", "llm", t0, llm_end,
            model=model,
            input={"messages": [{"role": "user", "content": asked}]} if asked else None,
            output={
                "role": "assistant",
                "content": text or None,
                "tool_calls": [{"name": p.get("tool"), "arguments": _g(p, "state", "input")} for p in tools],
            },
            usage=usage,
            cost=cost,
            meta={"finish_reason": minfo.get("finish"), "reasoning": _text(parts, "reasoning"), "agent": minfo.get("agent")},
            error=error,
        )
        asked = None

        for part in tools:
            state = part.get("state") or {}
            status = state.get("status")
            tool_key = "tool:" + (part.get("callID") or part.get("id") or "%s-%d" % (mid, tools.index(part)))
            t_end = _ns(_g(state, "time", "end"))
            still_open = t_end is None and status in ("running", "pending") and open_run
            tool_error = _failure(state.get("error"), "ToolError") if status == "error" else None
            # "bash" says little in a tree of thirty; OpenCode's own title for
            # the call ("python3 test_pager.py", "pager.py") says what it did.
            label = " ".join(x for x in (part.get("tool") or "tool", str(state.get("title") or "")) if x)
            rec = trace.add(
                tool_key, turn_id, label, "tool",
                _ns(_g(state, "time", "start")) or t0,
                None if still_open else (t_end or t1 or last or t0),
                input=state.get("input"),
                output=state.get("output"),
                meta={"tool": part.get("tool"), "exit": _g(state, "metadata", "exit")},
                error=tool_error,
            )
            child = by_id.get(_child_id(part) or "")
            if child is not None and child["info"].get("id") not in used:
                _session(trace, child, rec["span_id"], by_id, children_of, used)

    # Sub-agent sessions no tool call pointed at still belong to this session.
    for child in children_of.get(sid, []):
        if child["info"].get("id") not in used:
            _session(trace, child, root_id, by_id, children_of, used)

    title = name or info.get("title") or info.get("agent") or "session"
    return trace.add(
        root_key, parent_id, title, "agent", start, None if open_run else (last or start),
        output=final_text,
        meta={"agent": info.get("agent"), "session": sid, "directory": info.get("directory")},
        error=failure,
    )


def convert(sessions, task=None, model=None, attempt=None, tags=None, price=None):
    """``{trace_id: [records]}`` for a list of `opencode export` documents.

    A session that another one started (a sub-agent) is drawn inside its parent
    rather than as a run of its own. ``task``, ``model`` and ``attempt`` label
    the traces; without them the session's title and model are used.

    ``price`` is ``(input, output, cached)`` in US dollars per million tokens.
    It prices the model calls OpenCode recorded no cost for, which is every
    call to a model it has no price list for.
    """
    by_id = {s["info"]["id"]: s for s in sessions if _g(s, "info", "id")}
    children_of = {}
    for s in by_id.values():
        parent = s["info"].get("parentID")
        if parent in by_id:
            children_of.setdefault(parent, []).append(s)
    for group in children_of.values():
        group.sort(key=lambda s: _g(s, "info", "time", "created") or 0)

    out, used = {}, set()
    tops = [s for s in by_id.values() if s["info"].get("parentID") not in by_id]
    tops.sort(key=lambda s: _g(s, "info", "time", "created") or 0)
    for sess in tops:
        info = sess["info"]
        labels = {
            "name": info.get("title") or None,
            "task": task or info.get("title") or None,
            "model": model or _g(info, "model", "id") or None,
            "attempt": attempt,
            "tags": ["opencode"] + list(tags or []),
        }
        labels = {k: v for k, v in labels.items() if v not in (None, "", [])}
        trace = _Trace(_hex("opencode:" + info["id"], 32), labels, price)
        root = _session(trace, sess, None, by_id, children_of, used, name=task)
        if root["ev"] == "end":
            root["totals"] = trace.totals
        # Parents before children, in the order things happened, like a file the
        # tracer wrote live.
        trace.records.sort(key=lambda r: (r["start_ns"], r["parent_id"] is not None))
        out[trace.trace_id] = trace.records
    return out


def write(traces, out_dir):
    """One ``<trace-id>.jsonl`` per trace, replacing an earlier conversion."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for trace_id, records in traces.items():
        path = os.path.join(out_dir, trace_id + ".jsonl")
        data = "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in records)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, data.encode("utf-8"))
        finally:
            os.close(fd)
        paths.append(path)
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(description="Turn `opencode export` files into trace files.")
    ap.add_argument("files", nargs="+", help="session files written by `opencode export`")
    ap.add_argument("--out", default=os.path.join(".boltz", "traces"), help="where to write (default .boltz/traces)")
    ap.add_argument("--task", help="what the run was asked to do; labels the trace")
    ap.add_argument("--model", help="the model that ran it; default is the session's own")
    ap.add_argument("--attempt", type=int, help="which attempt this was")
    ap.add_argument(
        "--price",
        help="INPUT,OUTPUT[,CACHED] in US dollars per million tokens, for calls the session has no cost for",
    )
    args = ap.parse_args(argv)

    price = None
    if args.price:
        try:
            parts = [float(x) for x in args.price.split(",")]
            if len(parts) not in (2, 3) or any(x < 0 for x in parts):
                raise ValueError
        except ValueError:
            ap.error("--price takes INPUT,OUTPUT or INPUT,OUTPUT,CACHED, e.g. 3,15,0.3")
        price = (parts[0], parts[1], parts[2] if len(parts) == 3 else parts[0])

    sessions, skipped = [], 0
    for path in args.files:
        try:
            sessions.append(load(path))
        except (OSError, ValueError) as exc:
            skipped += 1
            print("skipped %s: %s" % (path, exc), file=sys.stderr)
    paths = write(convert(sessions, task=args.task, model=args.model, attempt=args.attempt, price=price), args.out)
    print("TRACES=%d" % len(paths))
    print("SKIPPED=%d" % skipped)
    return 0 if paths or not sessions else 1


if __name__ == "__main__":
    sys.exit(main())
