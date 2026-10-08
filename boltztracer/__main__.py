"""Read trace files in a terminal.

    python -m boltztracer                 # everything under .boltz/traces
    python -m boltztracer run.jsonl dir/  # these files and directories

With more than one trace it prints a comparison table first, one row per run,
then each run as a tree. It is a way to look at traces before any dashboard
exists, and a statement of what the file format is meant to answer.
"""

import json
import os
import sys

from ._core import DEFAULT_DIR


def load(paths):
    """``{trace_id: {span_id: record}}`` from files and directories.

    A step is written when it starts and again when it ends; the later line
    wins. A step with only its first line belongs to a run that was killed.
    """
    files = []
    for path in paths:
        if os.path.isdir(path):
            files += sorted(os.path.join(path, f) for f in os.listdir(path) if f.endswith(".jsonl"))
        else:
            files.append(path)
    traces = {}
    for path in files:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue  # blank, or cut short by a kill
                if isinstance(rec, dict) and "span_id" in rec and "trace_id" in rec:
                    traces.setdefault(rec["trace_id"], {})[rec["span_id"]] = rec
    return traces


def summarize(spans):
    """The numbers one run is compared on, recomputed from its steps so a run
    that never finished still has them."""
    recs = list(spans.values())
    # The top of the run is the step with no parent in this file. It may name
    # one all the same: a run started from another program's trace does.
    tops = sorted((r for r in recs if r.get("parent_id") not in spans), key=lambda r: r["start_ns"])
    root = tops[0] if tops else None
    labels = (root or recs[0]).get("trace") or {}
    start = min(r["start_ns"] for r in recs)
    end = max(r.get("end_ns") or r["start_ns"] for r in recs)
    out = {
        "task": labels.get("task") or labels.get("name") or "-",
        "model": labels.get("model") or "-",
        "attempt": labels.get("attempt", "-"),
        "status": "unfinished" if root is None or root.get("status") == "running" else root["status"],
        "ns": end - start,
        "llm": 0, "tools": 0, "errors": 0, "in": 0, "out": 0, "cost": None,
    }
    for r in recs:
        out["llm"] += r["kind"] == "llm"
        out["tools"] += r["kind"] == "tool"
        if r.get("error") and "from" not in r["error"]:
            out["errors"] += 1
        usage = r.get("usage") or {}
        out["in"] += usage.get("input", 0) + usage.get("cached", 0) + usage.get("cache_write", 0)
        out["out"] += usage.get("output", 0)
        if r.get("cost") is not None:
            out["cost"] = (out["cost"] or 0.0) + r["cost"]
    return out


def _dur(ns):
    s = ns / 1e9
    if s < 1:
        return f"{s * 1000:.0f}ms"
    if s < 60:
        return f"{s:.1f}s"
    return f"{int(s // 60)}m{int(s % 60):02d}s"


def _tok(n):
    return str(n) if n < 1000 else f"{n / 1000:.1f}k"


def _cost(c):
    return "-" if c is None else f"${c:.4f}"


def _table(rows, out):
    head = ["task", "model", "attempt", "status", "time", "llm", "tools", "errors", "tokens in/out", "cost"]
    body = [
        [
            str(r["task"]), str(r["model"]), str(r["attempt"]), r["status"], _dur(r["ns"]),
            str(r["llm"]), str(r["tools"]), str(r["errors"]), f"{_tok(r['in'])} / {_tok(r['out'])}", _cost(r["cost"]),
        ]
        for r in rows
    ]
    widths = [max(len(row[i]) for row in [head] + body) for i in range(len(head))]
    for row in [head] + body:
        out.write("  ".join(cell.ljust(w) for cell, w in zip(row, widths)).rstrip() + "\n")


def _tree(spans, out):
    children = {}
    for r in spans.values():
        parent = r.get("parent_id")
        children.setdefault(parent if parent in spans else None, []).append(r)
    lines = []

    def walk(rec, depth):
        notes = []
        if rec.get("end_ns"):
            notes.append(_dur(rec["end_ns"] - rec["start_ns"]))
        else:
            notes.append("unfinished")
        usage = rec.get("usage") or {}
        if usage:
            total_in = usage.get("input", 0) + usage.get("cached", 0) + usage.get("cache_write", 0)
            notes.append(f"{_tok(total_in)} in / {_tok(usage.get('output', 0))} out")
        if rec.get("cost") is not None:
            notes.append(_cost(rec["cost"]))
        err = rec.get("error")
        if err and "from" not in err:
            first = str(err.get("message", "")).strip().splitlines()[:1]
            notes.append(f"FAILED {err.get('type', 'Error')}: {first[0][:100] if first else ''}".rstrip(": "))
        elif err:
            notes.append("failed")
        lines.append(("  " * depth + f"{rec['kind']} {rec['name']}", "  ".join(notes)))
        for child in sorted(children.get(rec["span_id"], []), key=lambda r: r["start_ns"]):
            walk(child, depth + 1)

    for top in sorted(children.get(None, []), key=lambda r: r["start_ns"]):
        walk(top, 0)
    width = min(max(len(left) for left, _ in lines), 60)
    for left, right in lines:
        out.write(f"{left.ljust(width)}  {right}".rstrip() + "\n")


def main(argv=None, out=None):
    argv = sys.argv[1:] if argv is None else argv
    out = out or sys.stdout
    paths = argv or [DEFAULT_DIR]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        out.write(f"no such file or directory: {', '.join(missing)}\n")
        return 1
    traces = load(paths)
    if not traces:
        out.write("no traces found\n")
        return 1

    ordered = sorted(
        ((tid, spans, summarize(spans)) for tid, spans in traces.items()),
        key=lambda t: (str(t[2]["task"]), str(t[2]["model"]), str(t[2]["attempt"])),
    )
    if len(ordered) > 1:
        _table([s for _, _, s in ordered], out)
    for tid, spans, s in ordered:
        out.write(f"\ntrace {tid}  {s['task']} · {s['model']} · attempt {s['attempt']}\n")
        out.write(
            f"{s['status']}  {_dur(s['ns'])}  {s['llm']} llm calls  {s['tools']} tool calls  "
            f"{s['errors']} errors  {_tok(s['in'])} in / {_tok(s['out'])} out  {_cost(s['cost'])}\n\n"
        )
        _tree(spans, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
