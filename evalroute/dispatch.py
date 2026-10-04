"""dispatch: one verb that routes, spawns `hermes chat` on that arm, and
prints the rate line — the plugin epilog's manual workflow, mechanized for a
subprocess worker. stdout stays machine-readable (the card goes to stderr).
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from . import flywheel as fw
from . import routing as tools

_ROUTE_ID_COMMENT = re.compile(r"<!--\s*evalroute:\s*route-id=([0-9a-fA-F]+)\s*-->")
_SESSION_ID = re.compile(r"session_id:\s*(\S+)")


def _default_task(brief_path: Path) -> str:
    """The brief's first non-empty paragraph, leading '#'s stripped."""
    text = _ROUTE_ID_COMMENT.sub("", brief_path.read_text(encoding="utf-8"))
    for para in text.replace("\r\n", "\n").split("\n\n"):
        stripped = para.strip()
        if stripped:
            return stripped.lstrip("#").strip()
    return brief_path.stem


def _route(brief: Path, task: str, lane_id: str | None,
           replace_id: str = "") -> tuple[str, str, str, str, str | None]:
    """Route via the same functions the `route --json` path calls.

    Returns (card, model, effort, provider, route_id).
    """
    raw = task
    if lane_id:
        raw = f"--lane {lane_id} {task}"
        if replace_id:
            raw = f"--lane {lane_id} --replace-route-id {replace_id} {task}"
    card, lane, conf, pinned, method, route_id = tools._route_for_args(raw)
    envelope = tools._tool_result(card, lane, conf, pinned, method=method, route_id=route_id)
    provider = json.loads(envelope)["provider"]
    return card, lane["model"], tools._effort_for_override(lane), provider, route_id


def _build_argv(model: str, effort: str, provider: str, brief: Path,
                indir: str | None) -> list[str]:
    bin_spec = os.environ.get("EVALROUTE_HERMES_BIN") or shutil.which("hermes") or "hermes"
    # posix=False on Windows so backslash paths in the bin spec survive
    argv = [t.strip('"') for t in shlex.split(bin_spec, posix=(os.name != "nt"))]
    argv += ["chat", "-Q", "--oneshot",
             "-m", model, "--provider", provider, "--reasoning", effort,
             "--query-file", str(brief)]
    if indir:
        argv += ["--in", indir]
    return argv


def _spawn(argv: list[str], out_path: Path, timeout: float | None) -> tuple[int, str]:
    """Run the child; stdout -> report, stderr -> log. Returns (code, stderr text)."""
    err_path = out_path.with_suffix(out_path.suffix + ".stderr.log")
    with out_path.open("w", encoding="utf-8") as out_f, \
            err_path.open("w", encoding="utf-8") as err_f:
        try:
            if os.name == "nt":
                proc = subprocess.Popen(
                    argv, stdout=out_f, stderr=err_f, stdin=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            else:
                proc = subprocess.Popen(
                    argv, stdout=out_f, stderr=err_f, stdin=subprocess.DEVNULL,
                    start_new_session=True)
        except FileNotFoundError as exc:
            err_f.write(str(exc))
            return 127, str(exc)
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Kill the whole tree: hermes chat spawns children of its own, and
            # killing only the parent orphans them.
            try:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                                   capture_output=True)
                else:
                    import signal
                    os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                pass
            proc.wait()
            err_f.write(f"evalroute dispatch: killed after {timeout}s timeout\n")
            return 124, ""
    stderr_text = err_path.read_text(encoding="utf-8", errors="replace")
    return code, stderr_text


def _fmt_dur(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


def run(args: Any) -> int:
    """Handler for `hermes evalroute dispatch`. Returns the process exit code."""
    brief = Path(args.brief).resolve()
    if not brief.exists():
        print(f"evalroute dispatch: brief not found: {brief}", file=sys.stderr)
        return 2
    task = (getattr(args, "task", None) or "").strip() or _default_task(brief)
    lane_id = getattr(args, "lane", None)
    replace_id = ""
    if lane_id:
        comment = _ROUTE_ID_COMMENT.search(brief.read_text(encoding="utf-8"))
        if comment:
            replace_id = comment.group(1)
    try:
        card, model, effort, provider, route_id = _route(brief, task, lane_id, replace_id)
    except Exception as exc:
        print(f"evalroute dispatch: {exc}", file=sys.stderr)
        return 1
    if route_id is None:
        # The non-replace _note_route path swallows ledger-append failures and
        # returns None; without a route id the outcome can never be attributed.
        print("evalroute: route could not be recorded; not spawning", file=sys.stderr)
        return 1
    print(card, file=sys.stderr)

    out_path = Path(args.out).resolve() if getattr(args, "out", None) else \
        brief.with_name(brief.stem + ".report.md")
    indir = getattr(args, "indir", None)
    argv = _build_argv(model, effort, provider, brief, indir)
    rate_line = (f"rate it:  hermes evalroute rate pass|fail --route-id {route_id} "
                 f"--model {model} --effort {effort} --note \"...\"")

    if getattr(args, "dry_run", False):
        print(f"would run: {shlex.join(argv)}")
        print(f"dry run: route {route_id} noted but never rated - it is a SKIP for the human "
              f"(/rate skip --route-id {route_id})")
        print(rate_line)
        return 0

    timeout = getattr(args, "timeout", None)
    started = time.time()
    code, stderr_text = _spawn(argv, out_path, timeout)
    elapsed = time.time() - started
    session = _SESSION_ID.search(stderr_text)
    session_id = session.group(1) if session else None
    print(f"dispatched route {route_id} -> {model} @ {effort} ({lane_id or 'auto'}), "
          f"exit {code}, {_fmt_dur(elapsed)}")
    print(f"report: {out_path}   session: {session_id or '-'}")
    print(rate_line)
    if code != 0 and getattr(args, "rate_on_exit", None) == "fail":
        confirmation = fw.handle_rate(
            f"fail --route-id {route_id} --model {model} --effort {effort} "
            f"--note exit {code}")
        print(confirmation)
    return code
