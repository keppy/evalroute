"""dispatch: one verb that routes, spawns `hermes chat` on that arm, and
prints the rate line — the plugin epilog's manual workflow, mechanized for a
subprocess worker. stdout stays machine-readable (the card goes to stderr).

`--follow` polls the session store read-only and streams the worker's new
messages to stderr; it never affects stdout, the exit code, or the report.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from . import flywheel as fw
from . import routing as tools
from .paths import hermes_home

_ROUTE_ID_COMMENT = re.compile(r"<!--\s*evalroute:\s*route-id=([0-9a-fA-F]+)\s*-->")
_SESSION_ID = re.compile(r"session_id:\s*(\S+)")

_FOLLOW_POLL_SECONDS = 2.0


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


# ------------------------------------------------------------------- follow

def _state_db_path() -> Path:
    return hermes_home() / "state.db"


def _connect_ro(db_path: Path) -> sqlite3.Connection:
    """Read-only, bounded-wait connection to the session store."""
    uri = "file:" + db_path.resolve().as_posix() + "?mode=ro"
    return sqlite3.connect(uri, uri=True, timeout=1)


def _find_session(db_path: Path, cwd: str, since: float) -> str | None:
    """Newest session in the store whose cwd matches and started after `since`."""
    try:
        con = _connect_ro(db_path)
    except sqlite3.Error:
        return None
    try:
        row = con.execute(
            "select id from sessions where cwd = ? and started_at >= ? "
            "order by started_at desc limit 1", (cwd, since)).fetchone()
        return row[0] if row else None
    except sqlite3.Error:
        return None
    finally:
        con.close()


def _fmt_line(role: str, tool_name: str | None, content: str | None) -> str | None:
    """One message row -> `role[/tool]  first 100 chars` (first line only)."""
    crlf, lf = chr(13) + chr(10), chr(10)
    text = (content or "").replace(crlf, lf).strip()
    head = text.split(lf, 1)[0]
    head = " ".join(head.split())  # collapse internal whitespace runs
    if not head:
        return None
    label = f"{role}/{tool_name}" if tool_name else role
    return f"{label}  {head[:100]}"


def _stream_rows(con: sqlite3.Connection, session_id: str, stream, last_id: int) -> int:
    """Emit unseen messages; returns the new high-water mark."""
    try:
        rows = con.execute(
            "select id, role, tool_name, content, timestamp from messages "
            "where session_id = ? and id > ? order by id", (session_id, last_id)).fetchall()
    except sqlite3.Error:
        return last_id
    for _mid, role, tool_name, content, _ts in rows:
        line = _fmt_line(role, tool_name, content)
        if line:
            stamp = time.strftime("%H:%M:%S", time.localtime(_ts or time.time()))
            print(f"{stamp}  {line}", file=stream)
    return rows[-1][0] if rows else last_id


def _kill_tree(proc) -> None:
    """Kill the whole child tree (hermes chat spawns children of its own)."""
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


def _follow(db_path: Path, cwd: str, since: float, proc, stream=None,
            timeout: float | None = None) -> int:
    """Poll the session store until the child exits, streaming new messages.

    Advisory only: a missing or locked store downgrades to one stderr line
    and the loop keeps waiting on the child; follow never affects stdout,
    the report file, or (bar a --timeout kill, which returns 124) the exit
    code. The exit branch runs the stream pass once more so rows written
    between the final poll and death are caught. Returns the child's code.
    """
    stream = stream if stream is not None else sys.stderr
    session_id: str | None = None
    last_id = 0
    warned_missing = False
    warned_locked = False
    code: int | None = None
    while True:
        exited = proc.poll() is not None
        if exited:
            code = proc.returncode
        if not exited and not db_path.exists():
            if not warned_missing:
                print("evalroute dispatch --follow: session store not found; "
                      "waiting on the child", file=stream)
                warned_missing = True
        else:
            if session_id is None:
                session_id = _find_session(db_path, cwd, since)
            if session_id is not None:
                try:
                    con = _connect_ro(db_path)
                    try:
                        last_id = _stream_rows(con, session_id, stream, last_id)
                    finally:
                        con.close()
                except sqlite3.Error:
                    if not warned_locked:
                        print("evalroute dispatch --follow: session store locked; "
                              "waiting on the child", file=stream)
                        warned_locked = True
        if exited:
            return code
        if timeout is not None and time.time() - since >= timeout:
            _kill_tree(proc)
            return 124
        time.sleep(_FOLLOW_POLL_SECONDS)


def _spawn_follow(argv: list[str], out_path: Path, timeout: float | None,
                  cwd: str, since: float) -> tuple[int, str]:
    """Run the child while following it in the session store.

    The child's stdout still goes to the report file and its stderr to the
    log; the follow loop streams from the session store to OUR stderr.
    """
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
        code = _follow(_state_db_path(), cwd, since, proc, timeout=timeout)
        if code == 124:
            err_f.write(f"evalroute dispatch: killed after {timeout}s timeout\n")
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
    if getattr(args, "follow", False):
        # Poll the session store while the child runs; stderr-only advisory.
        cwd = str(Path(indir).resolve()) if indir else str(brief.parent)
        code, stderr_text = _spawn_follow(argv, out_path, timeout, cwd, started)
    else:
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
