from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import dataset, dispatch, flywheel, report, routing
from .routing import _lib_version, _route_for_args, _tool_result, install_routes, set_surface

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
    subparser.add_argument("--version", action="version",
                           version=f"evalroute {_lib_version()}")
    subs = subparser.add_subparsers(dest="evalroute_action")
    route_p = subs.add_parser("route", help="Classify a task and print a route card",
                              epilog=_WORKFLOW_EPILOG,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
    route_p.add_argument("task", nargs="*", help="The task description")
    route_p.add_argument("--lane", help="Pin a lane id instead of classifying")
    route_p.add_argument("--replace-route-id", help="Replace a specific pending route (requires --lane)")
    route_p.add_argument("--wide", action="store_true",
                         help="Single-line card for log scrapers (default: wraps at 100 columns)")
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
    status_p.add_argument("--with-contributed", action="store_true",
                          help="Also download the contributed/ tree (redacted opt-in rows, read-only)")
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
    report_p.add_argument("--demo", action="store_true",
                          help="render from the bundled synthetic fixture "
                               "(never reads or writes your ledger)")
    report_p.add_argument("--theme", choices=("dark", "light"), default="dark",
                          help="Page theme (default: dark; light = the classic page)")
    contrib_p = subs.add_parser("contribute",
                                help="Upload redacted outcome rows to the flywheel dataset "
                                     "(opt-in; --dry-run first)")
    contrib_p.add_argument("--dry-run", action="store_true",
                           help="Print the exact redacted rows that would be uploaded; send nothing")
    contrib_p.add_argument("--rotate-salt", action="store_true",
                           help="Regenerate the per-install salt and reset the cursor")
    contrib_p.add_argument("--repo", default=None,
                           help="Dataset repo id (default: keppy/evalroute-flywheel)")
    contrib_p.add_argument("--json", action="store_true",
                           help="Wrap the dry-run summary + rows in one JSON object")
    export_p = subs.add_parser("export-cases",
                               help="Write human-asserted lane labels as thomas "
                                    "encoder cases (JSONL; local file, counts only "
                                    "on stdout)")
    export_p.add_argument("--out", required=True,
                          help="Output JSONL path (UTF-8, LF; one case per line)")
    export_p.add_argument("--tasksets",
                          help="Dir or glob of Tier-A tasksets "
                               "(e.g. examples/artifacts; rows -> taskset:<id>)")
    export_p.add_argument("--seed-text", action="store_true",
                          help="Add one hint row and one keywords row per lane "
                               "from the route table (id: seed:<lane>:<kind>)")
    export_p.add_argument("--min-per-lane", type=int, default=20,
                          help="Flag lanes with fewer cases (default: 20; a "
                               "finding, not an error)")
    export_p.add_argument("--strict", action="store_true",
                          help="Exit 3 when any lane is below --min-per-lane")
    export_p.add_argument("--json", action="store_true",
                          help="Print the summary as one JSON object")
    enc_p = subs.add_parser("install-encoder",
                            help="Install a CONTRACT §4 encoder artifact "
                                 "(HF save_pretrained dir) as the classification "
                                 "fallback; --remove to uninstall")
    enc_p.add_argument("dir", nargs="?", help="Artifact directory to copy from")
    enc_p.add_argument("--remove", action="store_true", help="Delete the installed artifact")
    enc_p.add_argument("--json", action="store_true", help="Print the result as one JSON object")
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
    if action == "contribute":
        from . import contribute
        return contribute.run(dry_run=bool(getattr(args, "dry_run", False)),
                              rotate=bool(getattr(args, "rotate_salt", False)),
                              repo_id=getattr(args, "repo", None) or contribute.REPO_ID,
                              as_json=bool(getattr(args, "json", False)))
    if action == "report":
        return report.run(args)
    if action == "export-cases":
        from . import export_cases
        return export_cases.run_export(
            out=__import__("pathlib").Path(args.out),
            tasksets=getattr(args, "tasksets", "") or "",
            seed_text=bool(getattr(args, "seed_text", False)),
            min_per_lane=int(getattr(args, "min_per_lane", 20)),
            strict=bool(getattr(args, "strict", False)),
            as_json=bool(getattr(args, "json", False)))
    if action == "install-encoder":
        return install_encoder(dir_arg=getattr(args, "dir", None),
                               remove=bool(getattr(args, "remove", False)),
                               as_json=bool(getattr(args, "json", False)))
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
            card, lane_obj, conf, pinned, method, route_id = _route_for_args(
                raw, wide=bool(getattr(args, "wide", False)))
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


def install_encoder(dir_arg: str | None, remove: bool = False,
                    as_json: bool = False) -> int:
    """Validate and copy a CONTRACT §4 encoder artifact into <home>/evalroute/encoder/."""
    import shutil

    from .classify_encoder import encoder_dir, is_installed

    d = encoder_dir()
    if remove:
        if d.exists():
            shutil.rmtree(d)
        msg = {"installed": False, "removed": True, "encoder_dir": str(d)}
        print(json.dumps(msg) if as_json else f"encoder removed ({d})")
        return 0
    src = Path(dir_arg) if dir_arg else None
    if src is None or not src.is_dir():
        print("evalroute: install-encoder needs an artifact directory")
        return 2
    required = ("config.json", "label2id.json", "temperature.json", "metrics.json")
    missing = [f for f in required if not (src / f).is_file()]
    if missing or not any(src.glob("*.safetensors")) and not any(src.glob("*.bin")):
        print(f"evalroute: artifact incomplete; missing: {', '.join(missing) or 'model weights'}")
        return 2
    if d.exists():
        shutil.rmtree(d)
    shutil.copytree(src, d)
    metrics = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    label2id = json.loads((d / "label2id.json").read_text(encoding="utf-8"))
    known = {lane["id"] for lane in routing._load_routes()}
    unknown = sorted(lab for lab in label2id if lab not in known)
    msg = {
        "installed": True,
        "encoder_dir": str(d),
        "contract_version": metrics.get("contract_version"),
        "calib_accuracy": metrics.get("calib_accuracy"),
        "num_labels": metrics.get("num_labels"),
        "unknown_labels": unknown,
    }
    if as_json:
        print(json.dumps(msg))
    else:
        print(f"encoder installed: {d}")
        print(f"contract_version: {metrics.get('contract_version')}  "
              f"calib_accuracy: {metrics.get('calib_accuracy')}  "
              f"num_labels: {metrics.get('num_labels')}")
        if unknown:
            print(f"warning: labels that are not lane ids: {', '.join(unknown)}")
    return 0


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
