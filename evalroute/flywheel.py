"""Flywheel: label routes and outcomes as part of daily workflow.

Everything lands in ONE append-only JSONL file inside the Hermes home
(``<home>/evalroute/labels.jsonl``), never in the plugin dir (wiped on
reinstall). Record kinds:

  route           a route card was issued (/route, /evalroute route, tool)
  model_switch    /model observed near a route - unverified candidate if different
  effort_switch   /reasoning used after a route
  lane_correction /rate --lane X: the route's lane was wrong
  outcome         /rate pass|fail on the last route

Privacy: no response bodies or conversation content are ever logged. The
route record carries the task text the user typed to /route (that IS the
label); turn records are not persisted at all. The last-seen model is kept
in memory for diagnostics only, not for /rate attribution.

Correlation caveat: command handlers have no reliable session id. Default
/rate targets the latest pending route in the profile-wide ledger, NOT
necessarily this session. Use --route-id for unambiguous attribution. A
post_llm_call turn cannot be joined to a route and is never asserted as its
actual model.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from . import routing
from .paths import hermes_home

# ------------------------------------------------------------------ storage


def labels_path() -> Path:
    """<hermes home>/evalroute/labels.jsonl (profile-safe; never inside the package)."""
    d = hermes_home() / "evalroute"
    d.mkdir(parents=True, exist_ok=True)
    return d / "labels.jsonl"


def _append(record: dict[str, Any]) -> None:
    record["ts"] = time.time()
    record["iso"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with labels_path().open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_labels() -> list[dict[str, Any]]:
    p = labels_path()
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


# ------------------------------------------------- in-process correlation

_MEMORY: dict[str, Any] = {"route": None, "turn": None}
_LAST_RATE: dict[str, Any] = {"logged": False}  # structured result of the last handle_rate

def _pending_routes(records: list[dict[str, Any]]):
    """UUID consumption for new rows; timestamp fallback for legacy rows."""
    consumed_ids = {r["consumes_id"] for r in records if r.get("consumes_id")}
    legacy_ts = {r["consumes"] for r in records if r.get("consumes") is not None
                 and not r.get("consumes_id")}
    return (r for r in reversed(records) if r.get("kind") == "route"
            and (r.get("id") not in consumed_ids if r.get("id") else r.get("ts") not in legacy_ts))


def note_route(task: str, lane: dict[str, Any], method: str, conf: float,
                facets: Optional[list[str]] = None, replace_route_id: str = "") -> str:
    """Persist a route; replace only an explicitly named pending route.

    Text equality and process-global recency cannot establish session identity.
    """
    if replace_route_id and method != "pinned":
        raise ValueError("replace_route_id requires a pinned lane")
    previous = _route_by_id(replace_route_id) if replace_route_id else None
    if replace_route_id and (previous is None or previous.get("task") != task[:500]):
        raise ValueError("replacement route ID is not pending with the same task text")
    if previous:
        _append({"kind": "outcome", "rated": "skip", "route_lane": previous.get("lane"),
                 "consumes": previous.get("ts"), "consumes_id": previous.get("id"),
                 "note": "replaced by pinned reroute"})
        if previous.get("lane") != lane["id"]:
            _append({"kind": "lane_correction", "from_lane": previous.get("lane"),
                     "to_lane": lane["id"], "task": task[:200]})
    record = {
        "kind": "route", "id": uuid.uuid4().hex, "task": task[:500], "lane": lane["id"], "lane_label": lane["label"],
        "model": lane["model"], "effort": lane["effort"], "method": method,
        "confidence": round(float(conf), 2),
    }
    if facets:
        record["facets"] = facets
    _append(record)
    _MEMORY["route"] = dict(record)
    return record["id"]


def note_turn(session_id: Optional[str], model: Optional[str]) -> None:
    """Diagnostic only: a turn from this process, not a route attribution."""
    if session_id and model:
        _MEMORY["turn"] = {"session_id": session_id, "model": model, "ts": time.time()}


def _last_route() -> Optional[dict[str, Any]]:
    """Most recent route not yet consumed by an outcome/skip.

    Always read the file (memory is process-global and may be stale). New
    records use a route UUID as ``consumes_id``; legacy records use ``ts``.
    """
    return next(_pending_routes(read_labels()), None)

def _route_by_id(route_id: str) -> Optional[dict[str, Any]]:
    return next((r for r in _pending_routes(read_labels()) if r.get("id") == route_id), None)


def pending_route_from_file() -> Optional[dict[str, Any]]:
    """The most recent unrated route per the labels file, ignoring memory.

    Used by first-turn sniff to flag a pending route whose session is unknown.
    """
    return next(_pending_routes(read_labels()), None)


def _pending_label(route: Optional[dict[str, Any]], confirmed_model: str = "") -> str:
    """One-line summary; distinguish user confirmation from unverified switches."""
    if not route:
        return "no unrated route on file - /rate has nothing to label"
    actual = _actual_model(route)
    bits = [f"{route.get('lane', '?')}"]
    if confirmed_model:
        bits.append(f"user confirmed model {confirmed_model} (self-report)")
    elif actual and route.get("model") and actual != route.get("model"):
        bits.append(f"card said {route.get('model')}, switch observed {actual} (session unverified)")
    elif not actual:
        bits.append(f"model unknown (card recommended {route.get('model', '?')})")
    else:
        bits.append(f"switch observed {actual} (session unverified)")
    task = (route.get("task") or "")[:48]
    return f"labeled: {' - '.join(bits)} - task: {task!r} - route id: {route.get('id', 'legacy')}"


def _actual_model(route: Optional[dict[str, Any]]) -> Optional[str]:
    """A route-bound /model choice, never the last turn from another session."""
    for rec in reversed(read_labels()):
        if (rec.get("kind") == "model_switch" and route and route.get("id")
                and rec.get("route_id") == route["id"]):
            return rec.get("new_model")
    return None


# ------------------------------------------------------------------- hooks

_MODEL_CMD = "model"
_REASONING_CMD = "reasoning"


def on_pre_command(command: str, args_raw: str = "", session_key: Optional[str] = None,
                   platform: Optional[str] = None, **_) -> None:
    """Observer for /model and /reasoning: implicit route feedback."""
    cmd = (command or "").strip().lower()
    if cmd not in (_MODEL_CMD, _REASONING_CMD):
        return
    route = _last_route()
    args = (args_raw or "").strip()
    if not args:
        return
    if cmd == _MODEL_CMD:
        tokens = args.split()
        new_model = next((t for t in tokens if not t.startswith("-")), tokens[0] if tokens else "")
        rec: dict[str, Any] = {"kind": "model_switch", "new_model": new_model,
                               "session_key": session_key or "",
                               "route_association": "process_global_unverified"}
        if route:
            rec["route_id"] = route.get("id")
            rec["prev_route_lane"] = route.get("lane")
            rec["prev_route_model"] = route.get("model")
            rec["candidate_rejection"] = (new_model != route.get("model"))
        try:
            _append(rec)
        except Exception:
            pass  # advisory logging cannot break /model
    else:  # /reasoning
        level = args.split()[0].lower()
        if level in ("show", "hide", "full", "clamp"):
            return  # display queries are not effort changes
        rec = {"kind": "effort_switch", "new_effort": level,
               "session_key": session_key or "",
               "route_association": "process_global_unverified"}
        if route:
            rec["route_id"] = route.get("id")
            rec["prev_route_lane"] = route.get("lane")
            rec["prev_route_effort"] = route.get("effort")
        try:
            _append(rec)
        except Exception:
            pass  # advisory logging cannot break /reasoning


def on_post_llm_call(session_id: Optional[str] = None, model: Optional[str] = None,
                     **_) -> None:
    """Keep the last-seen model for /rate correlation. Nothing persisted."""
    note_turn(session_id, model)


# ------------------------------------------------------------------ /rate

_RATINGS = {"pass", "fail", "p", "f", "skip"}


def handle_rate(raw_args: str) -> str:
    """Rate a selected route, or the profile's latest pending route.

    Also stashes the structured result in ``_LAST_RATE`` for `rate --json`
    (keys: logged, verdict, route_id, lane, model, effort, arm_attribution,
    message). Not thread-safe beyond the CLI's single-threaded use.
    """
    global _LAST_RATE
    _LAST_RATE = {"logged": False}
    args = (raw_args or "").split()
    if not args or args[0].lower() not in _RATINGS:
        return ("usage: /rate pass|fail|skip [--route-id <id>] [--lane <lane-id>] "
                "[--model <model-id> --effort <level>] [--note <text>]\n"
                "Without --route-id, labels the latest pending route in this profile, not necessarily this session. "
                "Model/effort switches are diagnostic only; use both --model and --effort to confirm the actual arm.")
    rating = args[0].lower()
    if rating in ("p", "f"):
        rating = "pass" if rating == "p" else "fail"
    lane_fix = route_id = note = confirmed_model = confirmed_effort = max_turns = ""
    i = 1
    while i < len(args):
        if args[i] in ("--lane", "--route-id", "--model", "--effort", "--max-turns") and i + 1 < len(args):
            flag, value = args[i], args[i + 1]
            if flag == "--lane": lane_fix = value
            elif flag == "--route-id": route_id = value
            elif flag == "--model": confirmed_model = value
            elif flag == "--max-turns": max_turns = value
            else: confirmed_effort = value.lower()
            i += 2
        elif args[i] == "--note" and i + 1 < len(args):
            note = " ".join(args[i + 1:])
            break
        else:
            return f"evalroute: invalid or incomplete /rate option {args[i]!r}; use /rate with no args for help"
    try:
        max_turns_val: Optional[int] = int(max_turns) if max_turns else None
    except ValueError:
        return "evalroute: invalid --max-turns (must be an integer)"
    if bool(confirmed_model) != bool(confirmed_effort):
        return "evalroute: use --model and --effort together to confirm the actual arm"
    if confirmed_model and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]*", confirmed_model):
        return "evalroute: invalid --model id"
    if confirmed_effort and confirmed_effort not in {"none", "minimal", "low", "medium", "high", "xhigh", "max"}:
        return "evalroute: invalid --effort level"

    route = _route_by_id(route_id) if route_id else _last_route()
    if route is None:
        message = ("evalroute: no route on record yet (or route ID already consumed/unknown) "
                   "- run /route <task> first; nothing rated.")
        _LAST_RATE = {"logged": False, "verdict": rating, "route_id": route_id,
                      "lane": "", "model": "", "effort": "",
                      "arm_attribution": "unknown", "message": message}
        return message
    if rating == "skip":
        _append({"kind": "outcome", "rated": "skip", "route_lane": route.get("lane"),
                 "consumes": route.get("ts"), "consumes_id": route.get("id"), "note": note[:300]})
        _MEMORY["route"] = None  # skip consumes the route so the next /rate doesn't re-file it
        _LAST_RATE = {"logged": True, "verdict": "skip", "route_id": route.get("id"),
                      "lane": route.get("lane"), "model": confirmed_model,
                      "effort": confirmed_effort, "arm_attribution": "unknown",
                      "message": "route skipped (no verdict); the pending route was consumed."}
        return _LAST_RATE["message"]

    # Process-global command observations are diagnostics, never an actual arm.
    # Only an explicit user-provided model *and* effort count as self-reported
    # observational arm evidence. This is still not a paired experiment.
    record = {
        "kind": "outcome", "rated": rating, "route_lane": route.get("lane"),
        "route_model": route.get("model"), "route_effort": route.get("effort"),
        "actual_model": confirmed_model or None, "actual_effort": confirmed_effort or None,
        "arm_attribution": "explicit_user" if confirmed_model else "unknown",
        "method": route.get("method"), "confidence": route.get("confidence"),
        "max_turns": max_turns_val,
        "consumes": route.get("ts"), "consumes_id": route.get("id"),
    }
    if route.get("facets"):
        record["facets"] = route["facets"]
    if lane_fix:
        if routing._lane_by_id(lane_fix) is None:
            known = ", ".join(l["id"] for l in routing._load_routes())
            return f"evalroute: unknown lane {lane_fix!r}; known lanes: {known}"
        record["lane_correction"] = lane_fix
        record["original_lane"] = route.get("lane")
        record["route_lane"] = lane_fix  # aggregate against the corrected label
    if note:
        record["note"] = note[:300]
    _append(record)
    _MEMORY["route"] = None  # outcome consumes the pending route

    confirm = _pending_label(route, confirmed_model)
    _LAST_RATE = {"logged": True, "verdict": rating, "route_id": route.get("id"),
                  "lane": record["route_lane"], "model": confirmed_model,
                  "effort": confirmed_effort,
                  "arm_attribution": record["arm_attribution"], "message": ""}
    if lane_fix:
        _append({"kind": "lane_correction", "from_lane": route.get("lane"),
                 "to_lane": lane_fix, "task": route.get("task", "")[:200]})
        _LAST_RATE["message"] = (f"logged: {rating} (lane corrected {route.get('lane')} -> {lane_fix}). "
                "The correction also feeds the classifier's keyword table.\n" + confirm)
        return _LAST_RATE["message"]
    arm = f"{confirmed_model or 'unknown'} @ {confirmed_effort or '?'}"
    attribution = "user-confirmed (observational)" if confirmed_model else "unknown; switches unverified"
    _LAST_RATE["message"] = (f"logged: {rating} for lane {route.get('lane')} (arm {arm}; {attribution}). "
            f"{confirm}. {_counts_summary()}")
    return _LAST_RATE["message"]


def _counts_summary() -> str:
    try:
        from .routes_from_labels import aggregate
        stats = aggregate(read_labels())
    except Exception:
        return ""
    n = stats.get("outcomes", 0)
    routes = stats.get("routes", 0)
    return f"({routes} routes, {n} rated outcomes on file)"
