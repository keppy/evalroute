from __future__ import annotations

import argparse
import json

from . import dataset, dispatch, flywheel, report, routing
from .routing import _route_for_args, _tool_result, install_routes, set_surface

_WORKFLOW_EPILOG = """\
workflow (route -> arm -> rate, in the session that runs the task):
  1. /route <task>            classify; prints the card (lane, model, effort)
  2. /model <model>          set the arm from the card's "run:" line
                             (/reasoning <effort> too, unless install-routes
                             already wrote it into agent.reasoning_overrides)
  3. do the task in that session
  4. /rate pass|fail [--lane <lane-id>] [--note ...]
                             label the outcome; --lane files a correction
                             when the route got the lane wrong
                             --route-id <id> selects a pending route when overlapping
same flow from the terminal: hermes evalroute route "<task>" (step 1) and
hermes evalroute rate pass --note ... (step 4); steps 2-3 are chat commands.
hermes evalroute dispatch <brief.md> runs steps 1-3 on a subprocess worker and
prints the rate line for step 4.
routing data improves only when routes are rated: unrouted tasks cost the
same as ever, unrated routes teach nothing."""


def setup_cli(subparser) -> None:
    """argparse wiring for `hermes evalroute` (register_cli_command setup_fn)."""
    subs = subparser.add_subparsers(dest="evalroute_action")
    route_p = subs.add_parser("route", help="Classify a task and print a route card",
                              epilog=_WORKFLOW_EPILOG,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
    route_p.add_argument("task", nargs="*", help="The task description")
    route_p.add_argument("--lane", help="Pin a lane id instead of classifying")
    route_p.add_argument("--replace-route-id", help="Replace a specific pending route (requires --lane)")
    route_p.add_argument("--json", action="store_true",
                         help="Print the tool-result JSON envelope instead of the card")
    rate_p = subs.add_parser("rate", help="Rate the last routed task: pass|fail",
                             epilog=_WORKFLOW_EPILOG,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
    rate_p.add_argument("verdict", nargs="?", choices=["pass", "fail", "skip"],
                        help="pass | fail | skip")
    rate_p.add_argument("--lane", help="File a lane correction (the lane it should have been)")
    rate_p.add_argument("--route-id", help="Select a pending route explicitly (profile-wide ledger)")
    rate_p.add_argument("--model", help="Confirm the actual arm's model id (diagnostic; with --effort)")
    rate_p.add_argument("--effort", help="Confirm the actual arm's effort (diagnostic; with --model)")
    rate_p.add_argument("--note", help="Why — the highest-value part of the label")
    rate_p.add_argument("--json", action="store_true",
                        help="Print {\"logged\": ...} JSON instead of the human line")
    install_p = subs.add_parser("install-routes", help="Write the route table's effort "
                                   "column into agent.reasoning_overrides")
    install_p.add_argument("--dry-run", action="store_true", help="Show the diff, write nothing")
    status_p = subs.add_parser("sync", help="Pin the published route table "
                               "(keppy/evalroute-flywheel) under the Hermes home")
    status_p.add_argument("--revision", help="Pin a specific dataset revision "
                          "(default: resolve 'main' to its commit sha)")
    status_p.add_argument("--status", action="store_true",
                          help="Show which route table is active; no network")
    status_p.add_argument("--clear", action="store_true",
                          help="Unpin the dataset table; route on the bundled table")
    status_p.add_argument("--json", action="store_true",
                          help="Print the active-table state as JSON")
    dispatch_p = subs.add_parser("dispatch",
                                 help="Route a brief, spawn hermes chat on that arm, "
                                      "print the rate line",
                                 epilog=_WORKFLOW_EPILOG,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    dispatch_p.add_argument("brief", help="Path to the brief markdown file")
    dispatch_p.add_argument("--lane", help="Pin a lane id instead of classifying")
    dispatch_p.add_argument("--in", dest="indir", help="Extra --in dir for the child session")
    dispatch_p.add_argument("--task", help="Task description (default: the brief's first paragraph)")
    dispatch_p.add_argument("--out", help="Report path (default: <brief stem>.report.md beside it)")
    dispatch_p.add_argument("--timeout", type=float, help="Kill the child after SECONDS (exit 124)")
    dispatch_p.add_argument("--rate-on-exit", choices=["fail"],
                            help="Auto-rate fail when the child exits non-zero (never auto-passes)")
    dispatch_p.add_argument("--follow", action="store_true",
                            help="Stream the worker session's new messages from the "
                                 "session store to stderr while it runs")
    dispatch_p.add_argument("--dry-run", action="store_true",
                            help="Route and print the argv; spawn nothing")
    dispatch_p.add_argument("--runner", help="Named runner (hermes) or a shell-style "
                            "template with {model} {effort} {provider} {brief} {indir} "
                            "{brief_text} placeholders (default: EVALROUTE_RUNNER or hermes)")
    dispatch_p.add_argument("--json", action="store_true",
                            help="Print the dispatch sidecar object as JSON on stdout "
                                 "(the card still goes to stderr)")
    report_p = subs.add_parser("report",
                               help="Render one static HTML page over the ledger, "
                                    "sessions, trains, and drift findings")
    report_p.add_argument("--out", help="Output path (default: <home>/evalroute/report.html)")
    report_p.add_argument("--json", action="store_true",
                          help="Print the report's data model as JSON instead of "
                               "writing HTML")
    report_p.add_argument("--open", action="store_true",
                          help="Open the rendered page in the default browser")
    report_p.add_argument("--watch", type=float, metavar="SECONDS",
                          help="Regenerate every SECONDS until Ctrl-C "
                               "(one stderr line per regen)")
    report_p.add_argument("--trains", help="Trains dir (default: ./docs/trains if it exists)")
    report_p.add_argument("--factory-json", help="factory check --json findings to embed")
    subparser.set_defaults(func=evalroute_cli)


def evalroute_cli(args) -> int:
    """Handler for `hermes evalroute ...` (register_cli_command handler_fn)."""
    # The plugin calls this handler directly (no process main()); claim the
    # hermes CLI surface here so card command strings render as
    # `hermes evalroute ...`. The console script (main) pins "cli" first,
    # which makes this a no-op there.
    if routing.SURFACE == "cli" and not routing._SURFACE_EXPLICIT:
        routing.set_surface("hermes-cli")
    action = getattr(args, "evalroute_action", None)
    if action == "install-routes":
        return install_routes(dry_run=bool(getattr(args, "dry_run", False)))
    if action == "dispatch":
        return dispatch.run(args)
    if action == "sync":
        return dataset.run(args)
    if action == "report":
        return report.run(args)
    if action == "rate":
        as_json = bool(getattr(args, "json", False))
        parts = [getattr(args, "verdict", None) or ""]
        if getattr(args, "lane", None):
            parts.append(f"--lane {args.lane}")
        if getattr(args, "route_id", None):
            parts.append(f"--route-id {args.route_id}")
        if getattr(args, "model", None):
            parts.append(f"--model {args.model}")
        if getattr(args, "effort", None):
            parts.append(f"--effort {args.effort}")
        if getattr(args, "note", None):
            parts.append(f"--note {args.note}")
        message = flywheel.handle_rate(" ".join(parts))
        if as_json:
            data = dict(getattr(flywheel, "_LAST_RATE", {}) or {})
            data.setdefault("logged", False)
            data["message"] = message
            for key in ("verdict", "route_id", "lane", "model", "effort",
                        "arm_attribution"):
                data.setdefault(key, None)
            print(json.dumps(data))
        else:
            print(message)
        # C5: a rating that was not logged (bogus or already-consumed route
        # id, usage error) must not masquerade as success on the exit code.
        # `dispatch --rate-on-exit` calls flywheel.handle_rate directly and
        # already returns the worker's code, so this does not propagate there.
        return 0 if getattr(flywheel, "_LAST_RATE", {}).get("logged") else 1
    if action == "route":
        task = " ".join(getattr(args, "task", []) or [])
        lane = getattr(args, "lane", None)
        replace_id = getattr(args, "replace_route_id", None)
        as_json = getattr(args, "json", False)
        if not task and not lane:
            print(_WORKFLOW_EPILOG)
            return 2
        if replace_id and not lane:
            print("evalroute: --replace-route-id requires --lane <lane-id>")
            return 2
        try:
            if lane:
                raw = f"--lane {lane} {f'--replace-route-id {replace_id}' if replace_id else ''} {task}".strip()
            else:
                raw = task
            card, lane_obj, conf, pinned, method, route_id = _route_for_args(raw)
        except Exception as exc:
            print(f"evalroute: {exc}")
            return 1
        if as_json:
            print(_tool_result(card, lane_obj, conf, pinned, method=method, route_id=route_id))
        else:
            print(card)
        return 0
    print(_WORKFLOW_EPILOG)
    return 2


def main() -> int:
    """Standalone entry point (`evalroute ...`): cli-surface command strings."""
    import sys

    set_surface("cli")  # pin the surface before the handler can claim another
    parser = argparse.ArgumentParser(
        prog="evalroute",
        description="Route tasks to the right (model, reasoning effort) arm",
    )
    setup_cli(parser)
    args = parser.parse_args()
    rc = args.func(args)
    sys.exit(rc)
