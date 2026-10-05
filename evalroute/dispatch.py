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


def _resolve_max_turns(cli_value: Any = None) -> int | None:
    """--max-turns CLI > `agent.max_turns` in <hermes_home>/config.yaml > None."""
    if cli_value is not None:
        return int(cli_value)
    cfg = hermes_home() / "config.yaml"
    if not cfg.exists():
        return None
    try:
        import yaml

        data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
    except Exception:
        return None
    raw = (data.get("agent") or {}).get("max_turns")
    if isinstance(raw, int) and not isinstance(raw, bool):
        return raw
    return None


def _build_argv(model: str, effort: str, provider: str, brief: Path,
                indir: str | None, max_turns: int | None = None) -> list[str]:
    bin_spec = os.environ.get("EVALROUTE_HERMES_BIN") or shutil.which("hermes") or "hermes"
    # posix=False on Windows so backslash paths in the bin spec survive
    argv = [t.strip('"') for t in shlex.split(bin_spec, posix=(os.name != "nt"))]
    argv += ["chat", "-Q", "--oneshot",
             "-m", model, "--provider", provider, "--reasoning", effort,
             "--query-file", str(brief)]
    if indir:
        argv += ["--in", indir]
    if max_turns is not None:
        argv += ["--max-turns", str(max_turns)]
    return argv


# ------------------------------------------------------------- runner templates

# Named runners. `hermes` is the built-in default (flags verified against this
# repo's own spawn path). Every other named runner below must have had its
# flags read from the CLI's real `--help` on the maintainer's machine before
# it lands here — unverified CLIs live in docs/runners.md as sketches only.
_NAMED_RUNNERS: dict[str, str] = {
    "hermes": "hermes chat -Q --oneshot -m {model} --provider {provider} "
              "--reasoning {effort} --query-file {brief}",
    # Verified 2026-10-05 against Claude Code 2.1.289 `-p` mode on the
    # maintainer's machine (~/.local/bin/claude.exe): exit 0, one JSON object
    # on stdout with result/is_error/num_turns/total_cost_usd/session_id/
    # terminal_reason. Efforts accepted: low|medium|high|xhigh|max.
    "claude-code": "claude -p --model {model} --effort {effort} "
                   "--output-format json --no-session-persistence "
                   "--max-turns {max_turns} --add-dir {indir} {brief_text}",
}

# Harness-specific effort renames: the routed effort is valid on the route
# table but not accepted (or not meaningful) on that harness's CLI. Only the
# spawned argv sees the mapped value; the card still shows the routed effort.
_HARNESS_EFFORT_MAP: dict[str, dict[str, str]] = {
    "claude-code": {"none": "low", "minimal": "low"},
}


def _harness_for(runner: str | None, template: str) -> str:
    """The harness name recorded on the arm (sidecar, rate line, outcomes).

    A named runner names its harness; a raw template names the binary it
    invokes (its argv[0] stem). EVALROUTE_HARNESS overrides both — the escape
    hatch tests use to point a `claude`-shaped fake at a template.
    """
    env = os.environ.get("EVALROUTE_HARNESS", "").strip()
    if env:
        return env
    spec = (runner or os.environ.get("EVALROUTE_RUNNER", "")).strip()
    if spec in _NAMED_RUNNERS:
        return spec
    return Path(shlex.split(template, posix=(os.name != "nt"))[0]).stem
 
_TEMPLATE_VARS = ("{model}", "{effort}", "{provider}", "{brief}", "{indir}",
                  "{brief_text}", "{max_turns}")


def _known_runners() -> list[str]:
    env = os.environ.get("EVALROUTE_RUNNER", "")
    return sorted(set(_NAMED_RUNNERS) | ({env} if env else set()))


def _resolve_runner(runner: str | None) -> str:
    """--runner value -> a template string (a known name or a raw template)."""
    spec = (runner or "").strip()
    if not spec:
        spec = os.environ.get("EVALROUTE_RUNNER", "").strip() or "hermes"
    if spec in _NAMED_RUNNERS:
        return _NAMED_RUNNERS[spec]
    if "{" in spec and "}" in spec:
        return spec
    raise ValueError(
        f"unknown runner {spec!r}; known runners: "
        + ", ".join(_known_runners())
        + " (or pass a template with {model} {effort} {provider} {brief} "
          "{indir} {brief_text} placeholders)")


def _build_runner_argv(template: str, model: str, effort: str, provider: str,
                       brief: Path, indir: str | None,
                       max_turns: int | None = None) -> list[str]:
    """Substitute placeholders into already-split tokens, never the raw string.

    shlex splits the template first; each token then gets its placeholder
    replaced wholesale, so spaces and quotes inside paths survive as one argv
    entry. A token of exactly '{indir}' with no indir is dropped; '{brief_text}'
    substitutes the brief's file contents (for CLIs that take the prompt
    inline). '{effort}' may be omitted by the template — the routed effort is
    still recorded on the card and the rate line.
    """
    argv: list[str] = []
    brief_text = brief.read_text(encoding="utf-8", errors="replace")
    for token in shlex.split(template, posix=(os.name != "nt")):
        if token == "{indir}" and not indir:
            # A dropped value token would leave its flag dangling (a bare
            # `--add-dir`); drop a preceding `--flag` token too.
            if argv and argv[-1].startswith("--"):
                argv.pop()
            continue
        if token == "{max_turns}":
            if max_turns is None:
                if argv and argv[-1].startswith("--"):
                    argv.pop()
                continue
            argv.append(str(max_turns))
            continue
        value = (token
                 .replace("{model}", model)
                 .replace("{effort}", effort)
                 .replace("{provider}", provider)
                 .replace("{brief}", str(brief))
                 .replace("{indir}", str(indir or ""))
                 .replace("{brief_text}", brief_text)
                 .replace("{max_turns}", str(max_turns or "")))
        # posix=False on Windows keeps literal quotes around quoted backslash
        # paths ("C:\Users\...\python.exe"); strip them per token AFTER the
        # substitution so a substituted path with spaces is never re-split.
        argv.append(value.strip('"'))
    return argv


# ------------------------------------------------------------------- sidecar

_SIDECAR_KEYS = ("route_id", "lane", "model", "effort", "provider", "runner",
                 "harness", "brief", "indir", "started", "ended", "duration_s",
                 "exit", "session_id", "report", "rate_line", "max_turns",
                 "harness_turns", "harness_cost_usd", "harness_terminal")


def _write_sidecar(brief: Path, run: dict[str, Any]) -> Path:
    """Append one run record to `<brief>.dispatch.json` (newest last).

    A file without a "runs" list (a bare single-run object from an older
    writer) is converted to {"runs": [<that object>]} before appending.
    """
    path = brief.with_name(brief.stem + ".dispatch.json")
    existing: dict[str, Any] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            existing = {}
    runs = existing.get("runs")
    if not isinstance(runs, list):
        runs = [existing] if existing else []
    runs.append({k: run.get(k) for k in _SIDECAR_KEYS})
    path.write_text(json.dumps({"runs": runs}, indent=2) + "\n", encoding="utf-8")
    return path


def _latest_run(sidecar: Path) -> dict[str, Any] | None:
    """Newest run record from a dispatch sidecar, or None."""
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    runs = data.get("runs") if isinstance(data, dict) else None
    if isinstance(runs, list) and runs and isinstance(runs[-1], dict):
        return runs[-1]
    if isinstance(data, dict) and data.get("route_id") is not None:
        return data
    return None


def _child_env() -> dict[str, str]:
    """Environment for the worker: the parent's, minus approval bypasses.

    Hermes carries ``--yolo`` in ``HERMES_YOLO_MODE``; a dispatched worker
    must not inherit it (catalog rule 12 — a child must never be an
    approval-free agent because of how its parent was launched). The worker's
    approvals follow its own ``approvals.single_query_mode``, default deny.
    """
    env = dict(os.environ)
    for key in ("HERMES_YOLO_MODE",):
        env.pop(key, None)
    return env


def _spawn(argv: list[str], out_path: Path, timeout: float | None) -> tuple[int, str]:
    """Run the child; stdout -> report, stderr -> log. Returns (code, stderr text)."""
    err_path = out_path.with_suffix(out_path.suffix + ".stderr.log")
    with out_path.open("w", encoding="utf-8") as out_f, \
            err_path.open("w", encoding="utf-8") as err_f:
        try:
            if os.name == "nt":
                proc = subprocess.Popen(
                    argv, stdout=out_f, stderr=err_f, stdin=subprocess.DEVNULL,
                    env=_child_env(),
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            else:
                proc = subprocess.Popen(
                    argv, stdout=out_f, stderr=err_f, stdin=subprocess.DEVNULL,
                    env=_child_env(),
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
                    env=_child_env(),
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            else:
                proc = subprocess.Popen(
                    argv, stdout=out_f, stderr=err_f, stdin=subprocess.DEVNULL,
                    env=_child_env(),
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


def _report_missing_argv0(argv: list[str]) -> None:
    """One diagnostic line when the runner binary does not exist (C2)."""
    print(f"evalroute dispatch: runner argv[0] not found: {argv[0]}\n"
          f"  resolved argv: {shlex.join(argv)}", file=sys.stderr)


def run(args: Any) -> int:
    """Handler for `hermes evalroute dispatch`. Returns the process exit code."""
    brief = Path(args.brief).resolve()
    if not brief.exists():
        print(f"evalroute dispatch: brief not found: {brief}", file=sys.stderr)
        return 2
    as_json = bool(getattr(args, "json", False))
    try:
        template = _resolve_runner(getattr(args, "runner", None))
    except ValueError as exc:
        print(f"evalroute dispatch: {exc}", file=sys.stderr)
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
    max_turns = _resolve_max_turns(getattr(args, "max_turns", None))
    is_hermes = template == _NAMED_RUNNERS["hermes"]
    harness = _harness_for(getattr(args, "runner", None), template)

    # --model / --effort overrides: the route table's arm for a lane may not be
    # runnable on this harness (e.g. z-ai/glm-5.3 on Claude Code). When given,
    # they replace the routed arm for the spawn and are written everywhere the
    # actual arm goes (sidecar, rate line) — through `rate --model/--effort`
    # they land in the outcome as arm_attribution: explicit_user. The card
    # still shows the routed arm.
    routed_model, routed_effort = model, effort
    override_model = getattr(args, "model_override", None)
    override_effort = getattr(args, "effort_override", None)
    if override_effort is not None:
        if override_effort not in tools._VALID_EFFORTS:
            print(f"evalroute dispatch: invalid --effort {override_effort!r}; "
                  f"valid: {', '.join(sorted(tools._VALID_EFFORTS))}",
                  file=sys.stderr)
            return 2
    if override_model is not None:
        model = override_model
    if override_effort is not None:
        effort = override_effort
    if override_model is not None or override_effort is not None:
        print(f"arm override: {routed_model}@{routed_effort} -> {model}@{effort}",
              file=sys.stderr)
    # The harness's CLI may not accept the routed effort verbatim; map it for
    # the spawn only — the sidecar keeps what the harness actually got.
    spawn_effort = _HARNESS_EFFORT_MAP.get(harness, {}).get(effort, effort)

    if is_hermes:
        argv = _build_argv(model, spawn_effort, provider, brief, indir, max_turns)
    else:
        argv = _build_runner_argv(template, model, spawn_effort, provider,
                                  brief, indir, max_turns)
    rate_model = model
    rate_effort = effort
    rate_harness = "" if harness == "hermes" else f" --harness {harness}"
    rate_line = (
        f"rate it:  {tools.cmd_rate(route_id, rate_model, rate_effort, note='...')}"
        f"{rate_harness}")
    started_ts = time.time()

    def _emit(payload: dict[str, Any], lines: list[str]) -> None:
        if as_json:
            print(json.dumps(payload))
        else:
            for line in lines:
                print(line)

    if getattr(args, "dry_run", False):
        sidecar: dict[str, Any] = {
            "route_id": route_id, "lane": lane_id or "auto", "model": model,
            "effort": effort, "provider": provider,
            "runner": getattr(args, "runner", None) or os.environ.get("EVALROUTE_RUNNER")
                      or "hermes",
            "harness": harness,
            "brief": str(brief), "indir": indir, "started": None, "ended": None,
            "duration_s": None, "exit": None, "session_id": None,
            "report": str(out_path), "rate_line": rate_line,
            "max_turns": max_turns,
        }
        sidecar["argv"] = shlex.join(argv)
        _write_sidecar(brief, sidecar)
        _emit(sidecar, [f"would run: {shlex.join(argv)}",
               f"dry run: route {route_id} noted but never rated - it is a SKIP for the human "
               f"({tools.cmd_rate(route_id=route_id, verdict='skip', note='')})",
               rate_line])
        return 0

    timeout = getattr(args, "timeout", None)
    if getattr(args, "follow", False):
        # Poll the session store while the child runs; stderr-only advisory.
        cwd = str(Path(indir).resolve()) if indir else str(brief.parent)
        code, stderr_text = _spawn_follow(argv, out_path, timeout, cwd, started_ts)
    else:
        code, stderr_text = _spawn(argv, out_path, timeout)
    elapsed = time.time() - started_ts
    session_id = None
    if is_hermes:
        # _SESSION_ID is Hermes-shaped (the child prints `session_id:` on
        # stderr); other runners have no contract for it, so session is "-".
        session = _SESSION_ID.search(stderr_text)
        session_id = session.group(1) if session else None
    harness_turns = harness_cost = harness_terminal = None
    is_error = False
    if harness == "claude-code":
        # Claude Code -p mode: stdout is one JSON object. The `result` string
        # is the report; the sidecar carries the harness's own bookkeeping.
        try:
            payload = json.loads(out_path.read_text(encoding="utf-8",
                                                    errors="replace"))
            if isinstance(payload, dict) and "result" in payload:
                out_path.write_text(str(payload.get("result") or ""),
                                    encoding="utf-8")
                session_id = payload.get("session_id") or session_id
                harness_turns = payload.get("num_turns")
                harness_cost = payload.get("total_cost_usd")
                harness_terminal = payload.get("terminal_reason")
                if payload.get("is_error"):
                    is_error = True
        except (OSError, ValueError):
            pass  # not JSON (or unreadable): the stdout stands as the report
    if is_error:
        code = code or 1  # is_error: true counts as a non-zero exit
    sidecar = {
        "route_id": route_id, "lane": lane_id or "auto", "model": model,
        "effort": effort, "provider": provider, "runner":
            getattr(args, "runner", None) or os.environ.get("EVALROUTE_RUNNER")
            or "hermes",
        "harness": harness,
        "brief": str(brief), "indir": indir,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started_ts)),
        "ended": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "duration_s": round(elapsed, 3), "exit": code, "session_id": session_id,
        "report": str(out_path), "rate_line": rate_line, "max_turns": max_turns,
        "harness_turns": harness_turns, "harness_cost_usd": harness_cost,
        "harness_terminal": harness_terminal,
    }
    _write_sidecar(brief, sidecar)
    rate_exit: dict[str, Any] | None = None
    if code != 0 and getattr(args, "rate_on_exit", None) == "fail":
        harness_flag = rate_harness
        confirmation = fw.handle_rate(
            f"fail --route-id {route_id} --model {rate_model} --effort {rate_effort}"
            f"{harness_flag} --max-turns {max_turns} --note exit {code}"
            if max_turns is not None else
            f"fail --route-id {route_id} --model {rate_model} --effort {rate_effort}"
            f"{harness_flag} --note exit {code}")
        if as_json:
            # C1: with --json, stdout must be exactly one JSON object — the
            # confirmation rides in the envelope instead of trailing it.
            rate_exit = {"logged": bool(fw._LAST_RATE.get("logged")),
                         "message": confirmation}
            sidecar["rate_on_exit"] = rate_exit
        else:
            print(confirmation)
    turns = f", {max_turns} turns" if max_turns is not None else ""
    harness_txt = f", {harness}" if harness != "hermes" else ""
    _emit(sidecar, [f"dispatched route {route_id} -> {model} @ {effort} "
           f"({lane_id or 'auto'}{turns}{harness_txt}), "
           f"exit {code}, {_fmt_dur(elapsed)}",
           f"report: {out_path}   session: {session_id or '-'}",
           rate_line])
    if rate_exit is None and code == 127:
        _report_missing_argv0(argv)
    return code
