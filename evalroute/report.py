"""evalroute report — one static, self-contained HTML page over the flywheel
ledger, the Hermes session store, the train directories, and the factory
drift findings.

Stdlib only (json, sqlite3, html, string.Template): no network, no external
assets, no JS. Read-only everywhere: the ledger via ``flywheel.read_labels()``,
the session store over a read-only sqlite URI, the trains dir and the
findings file from disk. Anything missing (store, trains, findings) downgrades
to a one-line note — the page always renders.
"""

from __future__ import annotations

import html
import json
import re
import sqlite3
import string
import sys
import time
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import flywheel, routing
from .dispatch import _latest_run
from .paths import hermes_home

_SESSION_LINE = re.compile(r"session:\s*(\S+)")
_BRIEF_FILE = re.compile(r"^\d\d-.*\.md$")
_VERDICTS = {"pass", "fail", "skip"}

_PAGE = string.Template("""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>evalroute report</title>
<style>
body { font-family: -apple-system, "Segoe UI", sans-serif; margin: 2rem auto;
       max-width: 72rem; padding: 0 1rem; color: #1a1a1a; background: #fafafa; }
h1 { font-size: 1.5rem; margin-bottom: 0.3rem; }
h2 { font-size: 1.15rem; margin-top: 2rem; border-bottom: 2px solid #ddd;
     padding-bottom: 0.2rem; }
h3 { font-size: 1rem; margin-bottom: 0.2rem; }
table { border-collapse: collapse; width: 100%; margin: 0.5rem 0 1rem;
        font-size: 0.85rem; }
th, td { border: 1px solid #ddd; padding: 0.3rem 0.5rem; text-align: left;
         vertical-align: top; }
th { background: #f0f0f0; }
td.pass { color: #0a7d32; font-weight: 600; }
td.fail { color: #b3261e; font-weight: 600; }
td.skip { color: #777; font-weight: 600; }
.note { color: #666; font-style: italic; }
.meta { color: #555; font-size: 0.85rem; }
.wrap { white-space: pre-wrap; word-break: break-word; }
.corr { margin: 0; padding-left: 0.6rem; border-left: 3px solid #d8a915;
        color: #7a5b00; font-size: 0.8rem; white-space: pre-wrap; }
footer { margin-top: 3rem; color: #777; font-size: 0.8rem;
         border-top: 1px solid #ddd; padding-top: 0.5rem; }
code { background: #f0f0f0; padding: 0 0.2rem; }
</style>
</head>
<body>
<h1>evalroute report</h1>
<p class="meta">${header}</p>
${body}
<footer>produced by: ${footer}</footer>
</body>
</html>
""")


def _esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def _age(ts: Any) -> str:
    try:
        seconds = max(0.0, time.time() - float(ts))
    except (TypeError, ValueError):
        return "?"
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h"
    return f"{int(seconds // 86400)}d"


def _dur(seconds: Any) -> str:
    try:
        seconds = max(0.0, float(seconds))
    except (TypeError, ValueError):
        return "-"
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m{secs:02d}s"


def _connect_ro(db_path: Path) -> sqlite3.Connection:
    uri = "file:" + db_path.resolve().as_posix() + "?mode=ro"
    return sqlite3.connect(uri, uri=True, timeout=1)


def _under(path_str: str, root: Path) -> bool:
    try:
        return Path(path_str).resolve().is_relative_to(root.resolve())
    except (OSError, ValueError):
        return False


# --------------------------------------------------------------- ledger bits

def _ledger_data() -> dict[str, Any]:
    """The flywheel half of the data model (JSON-able)."""
    records = flywheel.read_labels()
    routes = [r for r in records if r.get("kind") == "route"]
    route_by_id = {r["id"]: r for r in routes if r.get("id")}
    pending = list(flywheel._pending_routes(records))
    outcomes_all = [r for r in records if r.get("kind") == "outcome"]
    outcomes = [o for o in outcomes_all if o.get("arm_attribution") == "explicit_user"]
    corrections = [r for r in records if r.get("kind") == "rating_correction"]

    tally: dict[tuple[str, str], dict[str, int]] = {}
    for out in outcomes:
        lane = out.get("route_lane") or "?"
        arm = f"{out.get('actual_model') or '?'}@{out.get('actual_effort') or '?'}"
        cell = tally.setdefault((lane, arm), {"n": 0, "pass": 0, "fail": 0, "skip": 0})
        cell["n"] += 1
        verdict = out.get("rated")
        if verdict in _VERDICTS:
            cell[verdict] += 1
    provenance = {lane["id"]: (lane.get("provenance") or "")
                  for lane in routing._load_routes()}
    tally_rows = [
        {"lane": lane, "provenance": provenance.get(lane, ""), "arm": arm,
         **cell}
        for (lane, arm), cell in sorted(tally.items())]

    methods: dict[str, int] = {}
    for rec in routes:
        method = rec.get("method") or "?"
        methods[method] = methods.get(method, 0) + 1

    corr_by_route: dict[str, list[dict[str, Any]]] = {}
    for corr in corrections:
        corr_by_route.setdefault(corr.get("consumes_id") or "", []).append(corr)
    outcome_rows = []
    for out in reversed(outcomes):
        rid = out.get("consumes_id") or ""
        route = route_by_id.get(rid) or {}
        row = {"iso": out.get("iso"), "route_id": rid,
               "lane": out.get("route_lane") or "?",
               "arm": f"{out.get('actual_model') or '?'}@{out.get('actual_effort') or '?'}",
               "verdict": out.get("rated") or "?",
               "method": route.get("method") or "", "note": out.get("note") or "",
               "corrections": []}
        for corr in corr_by_route.get(rid, []):
            bits = ["corrected by dispatcher"]
            if corr.get("from") or corr.get("to"):
                bits.append(f"{corr.get('from', '?')} -> {corr.get('to', '?')}")
            if corr.get("note"):
                bits.append(str(corr.get("note")))
            row["corrections"].append(" — ".join(bits))
        outcome_rows.append(row)

    pending_rows = [{"iso": rec.get("iso"), "lane": rec.get("lane") or "?",
                     "arm": f"{rec.get('model') or '?'} @ {rec.get('effort') or '?'}",
                     "task": (rec.get("task") or "")[:80], "age": _age(rec.get("ts"))}
                    for rec in pending]
    return {"routes": len(routes), "outcomes": len(outcomes_all),
            "pending": pending_rows, "outcome_rows": outcome_rows,
            "tally": tally_rows,
            "methods": [{"method": m, "routes": c} for m, c in sorted(methods.items())]}


def _now_section(pending: list[dict[str, Any]]) -> str:
    rows = []
    for rec in pending:
        rows.append(
            "<tr>"
            f"<td>{_esc(rec.get('iso') or '')}</td>"
            f"<td>{_esc(rec.get('lane') or '?')}</td>"
            f"<td>{_esc(rec.get('model') or '?')} @ {_esc(rec.get('effort') or '?')}</td>"
            f"<td class='wrap'>{_esc(rec.get('task'))}</td>"
            f"<td>{_esc(_age(rec.get('ts')))}</td>"
            "</tr>")
    body = "".join(rows) if rows else \
        "<tr><td colspan='5' class='note'>no pending routes — nothing awaits a rating</td></tr>"
    return ("<h2 id='now'>Now — pending routes</h2>"
            "<table><tr><th>time</th><th>lane</th><th>arm</th><th>task</th><th>age</th></tr>"
            + body + "</table>")


def _outcomes_section(data: dict[str, Any]) -> str:
    rows = []
    for row in data["outcome_rows"]:
        verdict = row["verdict"]
        cls = verdict if verdict in _VERDICTS else "skip"
        rows.append(
            "<tr>"
            f"<td>{_esc(row['iso'] or '')}</td>"
            f"<td><code>{_esc(row['route_id'][:8])}</code></td>"
            f"<td>{_esc(row['lane'])}</td>"
            f"<td>{_esc(row['arm'])}</td>"
            f"<td class='{cls}'>{_esc(verdict)}</td>"
            f"<td>{_esc(row['method'])}</td>"
            f"<td class='wrap'>{_esc(row['note'])}</td>"
            "</tr>")
        for corr in row["corrections"]:
            rows.append("<tr><td colspan='7'><div class='corr'>"
                        f"{_esc(corr)}</div></td></tr>")
    if not rows:
        return ("<h2 id='outcomes'>Outcomes (user-confirmed arms)</h2>"
                "<p class='note'>no outcomes with arm_attribution 'explicit_user' on file</p>")
    return ("<h2 id='outcomes'>Outcomes (user-confirmed arms)</h2>"
            "<table><tr><th>time</th><th>route</th><th>lane</th><th>arm</th>"
            "<th>verdict</th><th>method</th><th>note</th></tr>" + "".join(rows) + "</table>")


def _tally_section(data: dict[str, Any]) -> str:
    rows = []
    for cell in data["tally"]:
        rows.append(
            f"<tr data-lane='{_esc(cell['lane'])}' data-arm='{_esc(cell['arm'])}'>"
            f"<td>{_esc(cell['lane'])}</td>"
            f"<td class='wrap'>{_esc(cell['provenance'])}</td>"
            f"<td>{_esc(cell['arm'])}</td>"
            f"<td>{cell['n']}</td>"
            f"<td class='pass'>{cell['pass']}</td>"
            f"<td class='fail'>{cell['fail']}</td>"
            f"<td class='skip'>{cell['skip']}</td>"
            "</tr>")
    if not rows:
        return ("<h2 id='tally'>Per-lane tally</h2>"
                "<p class='note'>no user-confirmed arms to tally yet</p>")
    return ("<h2 id='tally'>Per-lane tally</h2>"
            "<table><tr><th>lane</th><th>provenance (active table)</th><th>arm</th>"
            "<th>n</th><th>pass</th><th>fail</th><th>skip</th></tr>"
            + "".join(rows) + "</table>")


def _methods_section(data: dict[str, Any]) -> str:
    rows = "".join(
        f"<tr data-method='{_esc(m['method'])}'><td>{_esc(m['method'])}</td>"
        f"<td>{m['routes']}</td></tr>"
        for m in data["methods"])
    if not rows:
        rows = "<tr><td colspan='2' class='note'>no routes on file</td></tr>"
    return ("<h2 id='methods'>Classification methods</h2>"
            "<table><tr><th>method</th><th>routes</th></tr>" + rows + "</table>")


# ------------------------------------------------------------- session store

def _sessions_data(db_path: Path, per_train_ids: dict[str, set[str]],
                   trains: Path | None) -> dict[str, Any]:
    """JSON-able session-store half of the data model."""
    if not db_path.exists():
        return {"present": False,
                "note": f"session store not found at {db_path} — sessions section skipped",
                "trains": []}
    try:
        con = _connect_ro(db_path)
        try:
            rows = con.execute(
                "select id, title, cwd, model, started_at, ended_at, last_activity_at, "
                "message_count, tool_call_count, input_tokens, output_tokens, "
                "estimated_cost_usd from sessions").fetchall()
        finally:
            con.close()
    except sqlite3.Error as exc:
        return {"present": False,
                "note": f"session store unreadable ({exc}) — sessions section skipped",
                "trains": []}

    by_train: dict[str, list[Any]] = {name: [] for name in per_train_ids}
    for row in rows:
        (sid, title, cwd, model, started, ended, last_act, msgs, tools,
         tokens_in, tokens_out, cost) = row
        train = None
        if trains is not None:
            for name in per_train_ids:
                if cwd and _under(cwd, trains / name):
                    train = name
                    break
        if train is None:
            train = next((name for name, ids in per_train_ids.items()
                          if sid in ids), None)
        if train is None:
            continue
        by_train.setdefault(train, []).append(row)

    out = {"present": True, "trains": []}
    for train in sorted(by_train):
        sess = by_train[train]
        if not sess:
            continue
        sessions = []
        total = 0.0
        for (sid, title, cwd, model, started, ended, last_act, msgs, tools,
             tokens_in, tokens_out, cost) in sess:
            total += float(cost or 0)
            sessions.append({
                "id": sid, "title": title or "", "model": model or "",
                "started": datetime.fromtimestamp(started).strftime('%Y-%m-%d %H:%M')
                           if started else "",
                "duration": _dur((ended or last_act or started) - started)
                            if started else "-",
                "messages": msgs or 0, "tools": tools or 0,
                "tokens_in": tokens_in or 0, "tokens_out": tokens_out or 0,
                "cost_usd": round(float(cost or 0), 2)})
        out["trains"].append({"name": train, "sessions": sessions,
                              "total_cost_usd": round(total, 2),
                              "count": len(sessions)})
    return out


def _sessions_section(data: dict[str, Any]) -> str:
    if not data.get("present"):
        return (f"<h2 id='sessions'>Sessions</h2>"
                f"<p class='note'>{_esc(data.get('note'))}</p>")
    if not data["trains"]:
        return ("<h2 id='sessions'>Sessions</h2>"
                "<p class='note'>no sessions matched (no dispatch reports, no "
                "sessions running under the trains dir)</p>")
    parts = ["<h2 id='sessions'>Sessions</h2>"]
    for train in data["trains"]:
        rows_html = []
        for s in train["sessions"]:
            rows_html.append(
                "<tr>"
                f"<td><code>{_esc(s['id'])}</code></td>"
                f"<td class='wrap'>{_esc(s['title'])}</td>"
                f"<td>{_esc(s['model'])}</td>"
                f"<td>{_esc(s['started'])}</td>"
                f"<td>{_esc(s['duration'])}</td>"
                f"<td>{s['messages']}</td><td>{s['tools']}</td>"
                f"<td>{s['tokens_in']:,} / {s['tokens_out']:,}</td>"
                f"<td>${s['cost_usd']:.2f}</td>"
                "</tr>")
        parts.append(f"<h3>{_esc(train['name'])}</h3>")
        parts.append(
            "<table><tr><th>session</th><th>title</th><th>model</th><th>started</th>"
            "<th>duration</th><th>msgs</th><th>tools</th><th>tokens in/out</th>"
            "<th>cost</th></tr>" + "".join(rows_html) + "</table>")
        parts.append(f"<p class='meta'>Total {_esc(train['name'])}: "
                     f"${train['total_cost_usd']:.2f} ({train['count']} sessions)</p>")
    return "\n".join(parts)


# -------------------------------------------------------------------- trains

def _parse_dispatch_log(text: str) -> list[dict[str, str]]:
    """Rows of the '## Dispatch log' markdown table, as header->cell dicts."""
    rows: list[dict[str, str]] = []
    header: list[str] | None = None
    in_log = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("## ") and "dispatch log" in stripped.lower():
            in_log = True
            header = None
            continue
        if not in_log:
            continue
        if stripped.startswith("## "):
            break  # next section ends the log
        if not stripped.startswith("|"):
            continue
        cells = [c.strip().strip("`*").strip()
                  for c in stripped.strip("|").split("|")]
        if set("".join(cells)) <= set("-: "):
            continue  # separator row
        if header is None:
            header = cells
            continue
        if header:
            rows.append(dict(zip(header, cells)))
    return rows


def _sidecar_index(trains: Path) -> dict[str, dict[str, Any]]:
    """Map resolved brief path -> newest run, from **/*.dispatch.json under trains."""
    index: dict[str, dict[str, Any]] = {}
    for sidecar in trains.glob("**/*.dispatch.json"):
        run = _latest_run(sidecar)
        if run:
            brief = run.get("brief")
            if brief:
                try:
                    index[str(Path(brief).resolve())] = run
                except (OSError, ValueError):
                    continue
    return index


def _trains_data(trains: Path | None) -> dict[str, Any]:
    """JSON-able trains half of the data model.

    Brief -> route -> session comes from `<brief>.dispatch.json` sidecars
    (written by `dispatch`) as the primary source; the README dispatch-log
    table and the report's `session:` line are fallbacks for trains run
    before sidecars existed.
    """
    if trains is None or not trains.is_dir():
        return {"present": False,
                "note": "no trains directory (pass --trains DIR; "
                        "default ./docs/trains when it exists)",
                "trains": []}
    sidecars = _sidecar_index(trains)
    per_train_ids: dict[str, set[str]] = {}
    data: list[dict[str, Any]] = []
    for train_dir in sorted(p for p in trains.iterdir() if p.is_dir()):
        readme = train_dir / "README.md"
        if not readme.exists():
            continue
        log = _parse_dispatch_log(readme.read_text(encoding="utf-8", errors="replace"))
        briefs = sorted(p for p in train_dir.glob("*.md")
                        if _BRIEF_FILE.match(p.name)
                        and not p.name.endswith(".report.md"))
        session_ids: set[str] = set()
        rows = []
        for brief in briefs:
            report_path = brief.with_name(brief.stem + ".report.md")
            done = report_path.exists()
            num = brief.name[:2]
            log_row = next((r for r in log
                            if (r.get("step") or "").startswith(num)), None)
            route_id = (log_row or {}).get("route id", "") or ""
            session_id = ""
            run = sidecars.get(str(brief.resolve()))
            if run is None:
                direct = _latest_run(brief.with_name(brief.stem + ".dispatch.json"))
                run = direct
            if run:
                route_id = str(run.get("route_id") or route_id)
                session_id = str(run.get("session_id") or "")
                if run.get("session_id"):
                    session_ids.add(str(run["session_id"]))
            elif done:
                match = _SESSION_LINE.search(
                    report_path.read_text(encoding="utf-8", errors="replace"))
                if match and match.group(1) != "-":
                    session_id = match.group(1)
                    session_ids.add(session_id)
            rows.append({"brief": brief.name,
                         "status": "done" if done else "not run",
                         "route_id": route_id, "session_id": session_id})
        per_train_ids[train_dir.name] = session_ids
        data.append({"name": train_dir.name, "briefs": rows})
    return {"present": True, "trains": data}


def _trains_section(data: dict[str, Any]) -> str:
    if not data.get("present"):
        return ("<h2 id='trains'>Trains</h2>"
                f"<p class='note'>{_esc(data.get('note'))}</p>")
    per = data["trains"]
    if not per:
        return ("<h2 id='trains'>Trains</h2>"
                "<p class='note'>no train dirs with a README.md</p>")
    parts = ["<h2 id='trains'>Trains</h2>"]
    for train in per:
        parts.append(f"<h3>{_esc(train['name'])}</h3>")
        if train["briefs"]:
            rows = "".join(
                f"<tr data-brief='{_esc(r['brief'])}'>"
                f"<td>{_esc(r['brief'])}</td>"
                f"<td>{_esc(r['status'])}</td>"
                f"<td><code>{_esc(r['route_id'])}</code></td>"
                f"<td><code>{_esc(r['session_id'])}</code></td>"
                "</tr>"
                for r in train["briefs"])
            parts.append("<table><tr><th>brief</th><th>status</th><th>route id</th>"
                         "<th>session</th></tr>" + rows + "</table>")
        else:
            parts.append("<p class='note'>no brief files</p>")
    return "\n".join(parts)


# --------------------------------------------------------------------- drift

def _drift_data(factory_json: str | None) -> dict[str, Any]:
    if not factory_json:
        return {"present": False, "note": "", "findings": []}
    path = Path(factory_json)
    if not path.exists():
        return {"present": True,
                "note": f"findings file not found: {path}", "findings": []}
    try:
        findings = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"present": True,
                "note": f"findings file unreadable ({exc})", "findings": []}
    if not isinstance(findings, list):
        findings = [f for f in findings.get("findings", [])] \
            if isinstance(findings, dict) else []
    return {"present": True, "note": "",
            "findings": [f for f in findings if isinstance(f, dict)]}


def _drift_section(data: dict[str, Any]) -> str:
    if not data.get("present"):
        return ""
    if data.get("note"):
        return (f"<h2 id='drift'>Drift</h2>"
                f"<p class='note'>{_esc(data['note'])}</p>")
    rows = "".join(
        "<tr>"
        f"<td>{_esc(f.get('repo', ''))}</td>"
        f"<td>{_esc(f.get('severity', ''))}</td>"
        f"<td>{_esc(f.get('check', ''))}</td>"
        f"<td class='wrap'>{_esc(f.get('message', ''))}</td>"
        "</tr>"
        for f in data["findings"])
    if not rows:
        rows = "<tr><td colspan='4' class='note'>no findings</td></tr>"
    return ("<h2 id='drift'>Drift</h2>"
            "<table><tr><th>repo</th><th>level</th><th>check</th><th>message</th></tr>"
            + rows + "</table>")


# ---------------------------------------------------------------------- page

def _data_model(trains: Path | None, factory_json: str | None) -> dict[str, Any]:
    """The machine-readable report: everything the HTML page renders from."""
    ledger = _ledger_data()
    trains_data = _trains_data(trains)
    sessions = _sessions_data(hermes_home() / "state.db",
                              {t["name"]: {b["session_id"] for b in t["briefs"]
                                           if b["session_id"]}
                               for t in trains_data["trains"]},
                              trains)
    drift = _drift_data(factory_json)
    return {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evalroute_version": routing._lib_version(),
        "table": routing._table_line(),
        "ledger_path": str(flywheel.labels_path()),
        "routes": ledger["routes"],
        "outcomes": ledger["outcomes"],
        "pending": ledger["pending"],
        "outcome_rows": ledger["outcome_rows"],
        "tally": ledger["tally"],
        "methods": ledger["methods"],
        "sessions": sessions,
        "trains": trains_data,
        "drift": drift,
    }


def _render(trains: Path | None, factory_json: str | None, command: str) -> str:
    data = _data_model(trains, factory_json)
    ledger = {"pending": data["pending"], "outcome_rows": data["outcome_rows"],
              "tally": data["tally"], "methods": data["methods"]}

    header = (
        f"generated {data['generated']} | evalroute {data['evalroute_version']} | "
        f"{data['table']} | ledger: {data['ledger_path']} | "
        f"{data['routes']} routes, {data['outcomes']} outcomes, "
        f"{len(data['pending'])} pending")

    body = "\n".join([
        _now_section(ledger["pending"]),
        _outcomes_section(ledger),
        _tally_section(ledger),
        _methods_section(ledger),
        _sessions_section(data["sessions"]),
        _trains_section(data["trains"]),
        _drift_section(data["drift"]),
    ])
    return _PAGE.substitute(header=_esc(header), body=body, footer=_esc(command))


def _command_line(args: Any, out_path: Path, trains: Path | None,
                  factory_json: str | None) -> str:
    parts = ["evalroute report", "--out", str(out_path)]
    if trains is not None:
        parts += ["--trains", str(trains)]
    if factory_json:
        parts += ["--factory-json", str(factory_json)]
    if getattr(args, "open", False):
        parts.append("--open")
    if getattr(args, "watch", None):
        parts += ["--watch", str(args.watch)]
    return " ".join(parts)


def run(args: Any) -> int:
    """Handler for `evalroute report`. Returns 0 (advisory page; never fails)."""
    out_path = Path(getattr(args, "out", None)
                    or (hermes_home() / "evalroute" / "report.html"))
    trains_arg = getattr(args, "trains", None)
    trains = Path(trains_arg) if trains_arg else (
        Path("./docs/trains") if Path("./docs/trains").is_dir() else None)
    factory_json = getattr(args, "factory_json", None)
    watch = getattr(args, "watch", None)
    command = _command_line(args, out_path, trains, factory_json)

    def _write() -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(_render(trains, factory_json, command),
                            encoding="utf-8")

    if not watch:
        if getattr(args, "json", False):
            print(json.dumps(_data_model(trains, factory_json), indent=2))
            return 0
        _write()
        print(f"report: {out_path}")
        if getattr(args, "open", False):
            webbrowser.open(out_path.resolve().as_uri())
        return 0

    # Watch: initial render immediately, then regenerate every N seconds.
    # Ctrl-C (or any interrupt) stops cleanly; the page always exists.
    _write()
    if getattr(args, "open", False):
        webbrowser.open(out_path.resolve().as_uri())
    try:
        while True:
            time.sleep(watch)
            _write()
            print(f"evalroute report: regenerated {out_path}", file=sys.stderr)
    except KeyboardInterrupt:
        print("evalroute report: watch stopped", file=sys.stderr)
    return 0
