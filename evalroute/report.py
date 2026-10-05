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

from . import demo, flywheel, routing
from . import classify_encoder
from .dispatch import _latest_run
from .paths import hermes_home

_SESSION_LINE = re.compile(r"session:\s*(\S+)")
_BRIEF_FILE = re.compile(r"^\d\d-.*\.md$")
_VERDICTS = {"pass", "fail", "skip"}

_MONO = "ui-monospace, Consolas, monospace"

# One palette per theme; the <style> block is rendered from it, so the two
# themes cannot structurally drift. Light = the pre-theme values.
_THEMES: dict[str, dict[str, str]] = {
    "dark": {
        "bg": "#0e1116", "text": "#e6e6e6", "muted": "#9aa4b2",
        "border": "#2a2f3a", "thbg": "#161b24", "codebg": "#1a2029",
        "note": "#9aa4b2", "meta": "#9aa4b2", "skip": "#8a93a6",
        "pass": "#5fd38d", "fail": "#ff7b72",
        "corrb": "#e0b25a", "corrt": "#e0b25a",
        "base_fs": "18px", "table_fs": "1rem", "maxw": "110rem",
    },
    "light": {
        "bg": "#fafafa", "text": "#1a1a1a", "muted": "#666",
        "border": "#ddd", "thbg": "#f0f0f0", "codebg": "#f0f0f0",
        "note": "#666", "meta": "#555", "skip": "#777",
        "pass": "#0a7d32", "fail": "#b3261e",
        "corrb": "#d8a915", "corrt": "#7a5b00",
        "base_fs": "16px", "table_fs": "0.85rem", "maxw": "72rem",
    },
}

_CSS = string.Template("""body { font-family: -apple-system, "Segoe UI", sans-serif;
        margin: 2rem auto; max-width: $maxw; padding: 0 1rem;
        color: $text; background: $bg; font-size: $base_fs; }
h1 { font-size: 1.5rem; margin-bottom: 0.3rem; }
h2 { font-size: 1.15rem; margin-top: 2rem; border-bottom: 2px solid $border;
     padding-bottom: 0.2rem; }
h3 { font-size: 1rem; margin-bottom: 0.2rem; }
table { border-collapse: collapse; width: 100%; margin: 0.5rem 0 1rem;
        font-size: $table_fs; }
th, td { border: 1px solid $border; padding: 0.3rem 0.5rem; text-align: left;
         vertical-align: top; }
th { background: $thbg; }
td.pass, .pass { color: $pass; font-weight: 600; }
td.fail, .fail { color: $fail; font-weight: 600; }
td.skip, .skip { color: $skip; font-weight: 600; }
.note { color: $note; font-style: italic; }
.meta { color: $meta; font-size: 0.85rem; }
.wrap { white-space: pre-wrap; word-break: break-word; }
.corr { margin: 0; padding-left: 0.6rem; border-left: 3px solid $corrb;
        color: $corrt; font-size: 0.8rem; white-space: pre-wrap; }
footer { margin-top: 3rem; color: $skip; font-size: 0.8rem;
         border-top: 1px solid $border; padding-top: 0.5rem; }
code, .mono { font-family: $mono; }
code { background: $codebg; padding: 0 0.2rem; }
.nowstrip { display: flex; gap: 1.5rem; margin: 1rem 0 0.5rem; flex-wrap: wrap; }
.nowstrip .cell { border: 1px solid $border; padding: 0.5rem 1rem;
                  min-width: 14rem; }
.nowstrip .k { color: $muted; font-size: 0.75rem; text-transform: none; }
.nowstrip .v { font-size: 1.15rem; font-weight: 600; margin-top: 0.2rem;
               font-family: $mono; }
.prov { font-family: $mono; font-weight: 600; }
.prov-measured { color: #5fd38d; }
.prov-observed { color: #e0b25a; }
.prov-priors { color: #8a93a6; }
""")

_PAGE = string.Template("""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>evalroute report</title>
<style>
${style}
</style>
</head>
<body>
<h1>evalroute report</h1>
<p class="meta">${header}</p>
${nowstrip}
${body}
<footer>produced by: ${footer}</footer>
</body>
</html>
""")


def _prov_span(text: str) -> str:
    """Provenance text as a coloured span, keyed on its first word."""
    first = (text or "").split("-", 1)[0].split(" ", 1)[0].strip().lower()
    value = _esc(text)
    if first in ("measured", "observed", "priors"):
        return f"<span class='prov prov-{first}'>{value}</span>"
    return _esc(text)


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

def _ledger_data(records: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """The flywheel half of the data model (JSON-able)."""
    records = records if records is not None else flywheel.read_labels()
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
               "age": _age(out.get("consumes")), "corrections": []}
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

    # Real labeled rows per lane: pinned routes with usable task text (the
    # export_cases rule), deduped by task text, plus corrections per target lane.
    from .export_cases import MIN_TEXT_CHARS
    lane_ids = [lane["id"] for lane in routing._load_routes()]
    seen_texts: set[str] = set()
    real_rows: dict[str, int] = {lane: 0 for lane in lane_ids}
    for rec in routes:
        if rec.get("method") != "pinned":
            continue
        task = rec.get("task") or ""
        if len(task) < MIN_TEXT_CHARS or task in seen_texts:
            continue
        seen_texts.add(task)
        lane = rec.get("lane") or "?"
        if lane in real_rows:
            real_rows[lane] += 1
    corrections_to: dict[str, int] = {lane: 0 for lane in lane_ids}
    for rec in records:
        if rec.get("kind") == "lane_correction":
            lane = rec.get("to_lane") or "?"
            if lane in corrections_to:
                corrections_to[lane] += 1
    labels = [{"lane": lane, "real_rows": real_rows[lane],
               "corrections_to": corrections_to[lane]} for lane in lane_ids]

    return {"routes": len(routes), "_routes_raw": routes,
            "outcomes": len(outcomes_all),
            "pending": pending_rows, "outcome_rows": outcome_rows,
            "tally": tally_rows,
            "labels": labels,
            "methods": [{"method": m, "routes": c} for m, c in sorted(methods.items())]}


def _now_strip_data(data: dict[str, Any]) -> dict[str, str]:
    """The four Now strip values, also published in the JSON data model."""
    outcomes = data.get("outcome_rows") or []
    last = outcomes[0] if outcomes else None
    if last:
        last_v = f"{last['lane']} · {last['arm']} · {last['verdict']} · {last.get('age') or '?'}"
    else:
        last_v = "—"

    cheapest = "—"
    sessions = data.get("sessions") or {}

    by_arm: dict[str, dict[str, float]] = {}
    for train in (sessions.get("trains") or []):
        for s in (train.get("sessions") or []):
            cost = s.get("cost_usd") or 0
            model = s.get("model") or ""
            if not model or not cost:
                continue
            started = s.get("started") or ""
            try:
                # `started` is rendered in local time (see _sessions_data); compare
                # ISO weeks in local time too, or Sunday evening flips the week.
                st = datetime.strptime(started, "%Y-%m-%d %H:%M")
                if st.isocalendar()[:2] != datetime.now().isocalendar()[:2]:
                    continue
            except ValueError:
                pass  # no parseable timestamp: count it anyway
            key = f"{model}@{s.get('effort') or '?'}"
            cell = by_arm.setdefault(key, {"cost": 0.0, "passes": 0})
            cell["cost"] += float(cost)
            if cost:
                cell["passes"] += 1
    if by_arm:
        arm, cell = min(by_arm.items(), key=lambda kv: kv[1]["cost"] / max(1, kv[1]["passes"]))
        per = cell["cost"] / max(1, cell["passes"])
        cheapest = f"{arm} ${per:.2f}/pass"

    # Routes filed today (local calendar day), not the ledger's whole life.
    start_of_today = datetime.now().astimezone().replace(
        hour=0, minute=0, second=0, microsecond=0).timestamp()
    routes_today = sum(1 for r in data.get("_routes_raw", [])
                       if (r.get("ts") or 0) >= start_of_today)

    cells = {
        "routes_today": str(routes_today),
        "pending": str(len(data.get("pending") or [])),
        "last_outcome": last_v,
        "cheapest_arm_this_week": cheapest,
    }
    return cells


def _now_strip_html(cells: dict[str, str]) -> str:
    """The four-cell Now strip under the header, from the computed `now` block."""
    inner = "".join(
        f"<div class='cell'><div class='k'>{_esc(k.replace('_', ' '))}</div>"
        f"<div class='v'>{_esc(v)}</div></div>"
        for k, v in cells.items())
    return f"<div class='nowstrip' id='nowstrip'>{inner}</div>"


def _now_strip(data: dict[str, Any]) -> str:
    return _now_strip_html(_now_strip_data(data))


def _now_section(pending: list[dict[str, Any]]) -> str:
    # `pending` rows are the projection built in _ledger_data (iso/lane/arm/task/age),
    # not raw ledger records — read those keys, or every row renders "? @ ?".
    rows = []
    for rec in pending:
        rows.append(
            "<tr>"
            f"<td>{_esc(rec.get('iso') or '')}</td>"
            f"<td>{_esc(rec.get('lane') or '?')}</td>"
            f"<td>{_esc(rec.get('arm') or '? @ ?')}</td>"
            f"<td class='wrap'>{_esc(rec.get('task'))}</td>"
            f"<td>{_esc(rec.get('age') or '?')}</td>"
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
        prov_first = (cell.get("provenance") or "").split("-", 1)[0].split(" ", 1)[0].strip().lower()
        prov_cls = prov_first if prov_first in ("measured", "observed", "priors") else ""
        rows.append(
            f"<tr data-lane='{_esc(cell['lane'])}' data-arm='{_esc(cell['arm'])}'>"
            f"<td>{_esc(cell['lane'])}</td>"
            f"<td><span class='prov{(' prov-' + prov_cls) if prov_cls else ''}'>"
            f"{_esc(prov_first or cell.get('provenance') or '?')}</span></td>"
            f"<td class='mono'>{_esc(cell['arm'])}</td>"
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


def _labels_section(data: dict[str, Any]) -> str:
    rows = ""
    for lab in data["labels"]:
        note = ""
        if lab["real_rows"] < 8:
            note = (f"<td class='note'>pin 8 real tasks here: "
                    f"evalroute route --lane {lab['lane']} \"&lt;task&gt;\"</td>")
        else:
            note = "<td></td>"
        rows += (f"<tr><td>{_esc(lab['lane'])}</td><td>{lab['real_rows']}</td>"
                 f"<td>{lab['corrections_to']}</td>{note}</tr>")
    return ("<h2 id='labels'>Real labeled rows per lane</h2>"
            "<table><tr><th>lane</th><th>real labeled rows</th>"
            "<th>corrections → lane</th><th></th></tr>" + rows + "</table>")


# ------------------------------------------------------------- session store

def _sessions_data(db_path: Path, per_train_ids: dict[str, set[str]],
                   trains: Path | None,
                   sidecar_cost: dict[str, float] | None = None,
                   sidecar_meta: dict[str, dict[str, str]] | None = None
                   ) -> dict[str, Any]:
    """JSON-able session-store half of the data model.

    ``sidecar_cost`` (demo mode, no state.db): session_id -> cost from the
    dispatch sidecars, so the sessions section still shows real-looking
    per-run spend. ``sidecar_meta``: session_id -> model/effort for the strip.
    """
    if not db_path.exists() and sidecar_cost:
        out = {"present": True, "trains": [], "source": "dispatch sidecars"}
        for train, ids in sorted(per_train_ids.items()):
            sessions = []
            total = 0.0
            for sid in sorted(ids):
                cost = sidecar_cost.get(sid)
                if cost is None:
                    continue
                total += float(cost)
                meta = (sidecar_meta or {}).get(sid) or {}
                sessions.append({"id": sid, "title": "dispatched worker (demo)",
                                 "model": meta.get("model") or "",
                                 "effort": meta.get("effort") or "",
                                 "started": datetime.fromtimestamp(
                                     time.time() - 3600).strftime(
                                     "%Y-%m-%d %H:%M"),
                                 "duration": "-",
                                 "messages": 0, "tools": 0, "tokens_in": 0,
                                 "tokens_out": 0,
                                 "cost_usd": round(float(cost), 2)})
            if sessions:
                out["trains"].append({"name": train, "sessions": sessions,
                                      "total_cost_usd": round(total, 2),
                                      "count": len(sessions)})
        return out
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

def _data_model(trains: Path | None, factory_json: str | None,
                is_demo: bool = False) -> dict[str, Any]:
    """The machine-readable report: everything the HTML page renders from."""
    if is_demo:
        ledger_records = demo.load_demo_ledger()
        trains = demo.demo_trains_dir()
    else:
        ledger_records = None
    ledger = _ledger_data(ledger_records)
    trains_data = _trains_data(trains)
    sidecar_cost: dict[str, float] | None = None
    sidecar_meta: dict[str, dict[str, str]] | None = None
    if is_demo:
        sidecar_cost = {}
        sidecar_meta = {}
        for train in trains_data["trains"]:
            for brief in train["briefs"]:
                run = _latest_run(
                    (trains / train["name"] /
                     brief["brief"].replace(".md", ".dispatch.json")))
                if run and run.get("session_id") and run.get("estimated_cost_usd"):
                    sidecar_cost[str(run["session_id"])] = \
                        float(run["estimated_cost_usd"])
                if run and run.get("session_id"):
                    sidecar_meta[str(run["session_id"])] = {
                        "model": str(run.get("model") or ""),
                        "effort": str(run.get("effort") or "")}
    # Demo never reads the real session store: pass a path that cannot exist so
    # the sidecar branch is taken on machines that do have a state.db.
    sessions = _sessions_data(
        (Path(demo.demo_trains_dir()) / "no-state.db") if is_demo
        else hermes_home() / "state.db",
        {t["name"]: {b["session_id"] for b in t["briefs"]
                     if b["session_id"]}
         for t in trains_data["trains"]},
        trains, sidecar_cost=sidecar_cost, sidecar_meta=sidecar_meta)
    drift = _drift_data(factory_json if not is_demo else None)
    out = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evalroute_version": routing._lib_version(),
        "table": routing._table_line(),
        "classifier": ("rules+encoder@%.2f" % float(
            (classify_encoder.load() or {}).get("metrics", {}).get("calib_accuracy", 0.0))
            if classify_encoder.is_installed() else "rules+llm"),
        "ledger_path": ("demo fixture (evalroute/data/demo/labels.jsonl) "
                        if is_demo else str(flywheel.labels_path())),
        "demo": is_demo,
        "routes": ledger["routes"],
        "outcomes": ledger["outcomes"],
        "pending": ledger["pending"],
        "outcome_rows": ledger["outcome_rows"],
        "tally": ledger["tally"],
        "labels": ledger["labels"],
        "methods": ledger["methods"],
        "sessions": sessions,
        "trains": trains_data,
        "drift": drift,
        "now": _now_strip_data({"routes": ledger["routes"],
                                "_routes_raw": ledger["_routes_raw"],
                                "pending": ledger["pending"],
                                "outcome_rows": ledger["outcome_rows"],
                                "sessions": sessions}),
    }
    return out


def _render(trains: Path | None, factory_json: str | None, command: str,
            is_demo: bool = False, theme: str = "dark") -> str:
    data = _data_model(trains, factory_json, is_demo)
    theme_name = theme if theme in _THEMES else "dark"
    palette = _THEMES[theme_name]
    css = _CSS.substitute(**palette, mono=_MONO)
    ledger = {"pending": data["pending"], "outcome_rows": data["outcome_rows"],
              "tally": data["tally"], "methods": data["methods"],
              "labels": data["labels"]}

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
        _labels_section(ledger),
        _sessions_section(data["sessions"]),
        _trains_section(data["trains"]),
        _drift_section(data["drift"]),
    ])
    badge = ("<p style='display:inline-block;background:#b3261e;color:#fff;"
             "padding:0.15rem 0.5rem;border-radius:3px;font-size:0.8rem;"
             "font-weight:700'>DEMO DATA — synthetic fixture, never your "
             "ledger</p>") if is_demo else ""
    # One source of truth: the HTML strip renders the same `now` block --json
    # exports (recomputing from the top-level dict lost `_routes_raw` -> "0").
    return _PAGE.substitute(header=_esc(header) + badge,
                            style=css,
                            nowstrip=_now_strip_html(data["now"]),
                            body=body, footer=_esc(command))


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
    is_demo = bool(getattr(args, "demo", False))
    default_name = "report-demo.html" if is_demo else "report.html"
    out_path = Path(getattr(args, "out", None)
                    or (hermes_home() / "evalroute" / default_name))
    trains_arg = getattr(args, "trains", None)
    if is_demo and not trains_arg:
        trains = demo.demo_trains_dir()  # resolved after _data_model anyway
    else:
        trains = Path(trains_arg) if trains_arg else (
            Path("./docs/trains") if Path("./docs/trains").is_dir() else None)
    factory_json = getattr(args, "factory_json", None)
    watch = getattr(args, "watch", None)
    command = _command_line(args, out_path, trains, factory_json)

    def _write() -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            _render(trains, factory_json, command, is_demo,
                    getattr(args, "theme", None) or "dark"),
            encoding="utf-8")

    if not watch:
        if getattr(args, "json", False):
            print(json.dumps(_data_model(trains, factory_json, is_demo),
                             indent=2))
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
