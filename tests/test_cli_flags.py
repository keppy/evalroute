"""CLI flags: rate --model/--effort, route --replace-route-id, route --json."""

from __future__ import annotations

import json

from evalroute import flywheel
from evalroute import routing
from evalroute import cli

from tests.test_flywheel import home  # noqa: F401  re-exported fixture


def _cli_route(lane=None, task="fix a flaky test", replace_id=None, as_json=False, capsys=None):
    from types import SimpleNamespace
    ns = SimpleNamespace(evalroute_action="route", task=task.split(), lane=lane,
                         replace_route_id=replace_id, json=as_json, dry_run=False)
    rc = cli.evalroute_cli(ns)
    out = capsys.readouterr().out if capsys else None
    return rc, out


def _last_route_row():
    return next(r for r in reversed(flywheel.read_labels()) if r["kind"] == "route")


# ------------------------------------------------------------ rate --model/--effort

def test_cli_rate_explicit_arm(home, capsys):
    lane = routing._lane_by_id("dl-ml-research-engineering")
    route_id = flywheel.note_route("test the cli rate path", lane, "llm", 0.8)
    from types import SimpleNamespace
    ns = SimpleNamespace(evalroute_action="rate", verdict="pass", lane=None,
                         route_id=route_id, model="z-ai/glm-5.3-flash", effort="medium",
                         note="cli explicit arm", dry_run=False)
    rc = cli.evalroute_cli(ns)
    assert rc == 0
    out = capsys.readouterr().out
    assert "arm z-ai/glm-5.3-flash @ medium" in out
    outcome = next(r for r in flywheel.read_labels() if r["kind"] == "outcome")
    assert outcome["arm_attribution"] == "explicit_user"
    assert outcome["actual_model"] == "z-ai/glm-5.3-flash"
    assert outcome["actual_effort"] == "medium"


def test_cli_rate_model_without_effort_rejected(home, capsys):
    from types import SimpleNamespace
    ns = SimpleNamespace(evalroute_action="rate", verdict="pass", lane=None,
                         route_id=None, model="z-ai/glm-5.3-flash", effort=None,
                         note=None, dry_run=False)
    rc = cli.evalroute_cli(ns)
    assert rc == 1  # nothing was logged (C5): the usage string printed, exit 1
    assert "use --model and --effort together" in capsys.readouterr().out
    assert not [r for r in flywheel.read_labels() if r["kind"] == "outcome"]


# ------------------------------------------------------------ route --json

def test_cli_route_json_envelope(home, capsys):
    rc, out = _cli_route(task="fix a flaky test", as_json=True, capsys=capsys)
    assert rc == 0
    env = json.loads(out.strip())
    for key in ("route_id", "model", "effort", "provider", "lane"):
        assert key in env, key
    assert env["provider"] == "nous"
    assert env["route_id"]
    routes = [r for r in flywheel.read_labels() if r["kind"] == "route"]
    assert len(routes) == 1 and routes[0]["id"] == env["route_id"]


# ------------------------------------------- route --replace-route-id

def test_cli_route_replace_pending_route(home, capsys):
    _, first_out = _cli_route(task="fix a flaky test", as_json=True, capsys=capsys)
    old_id = json.loads(first_out.strip())["route_id"]
    rc, _ = _cli_route(lane="routine-coding", task="fix a flaky test",
                       replace_id=old_id, capsys=capsys)
    assert rc == 0
    recs = flywheel.read_labels()
    routes = [r for r in recs if r["kind"] == "route"]
    assert len(routes) == 2
    assert routes[-1]["lane"] == "routine-coding" and routes[-1]["method"] == "pinned"
    # note_route records the replacement as a skip outcome that consumes the old id
    replaced = next(r for r in recs if r["kind"] == "outcome" and r["rated"] == "skip")
    assert replaced["consumes_id"] == old_id
    assert replaced["note"] == "replaced by pinned reroute"


def test_cli_route_replace_requires_lane(home, capsys):
    rc, out = _cli_route(task="same task", replace_id="deadbeef", capsys=capsys)
    assert rc == 2
    assert "--replace-route-id requires --lane" in out
    assert not [r for r in flywheel.read_labels() if r["kind"] == "route"]


# ------------------------------------------- route --lane without --replace-route-id

def test_cli_route_lane_only(home, capsys):
    rc, out = _cli_route(lane="routine-coding", capsys=capsys)
    assert rc == 0
    assert out.strip()
    routes = [r for r in flywheel.read_labels() if r["kind"] == "route"]
    assert len(routes) == 1
    assert routes[0]["method"] == "pinned"
    assert routes[0]["lane"] == "routine-coding"


def test_cli_route_lane_only_json(home, capsys):
    rc, out = _cli_route(lane="routine-coding", as_json=True, capsys=capsys)
    assert rc == 0
    env = json.loads(out.strip())
    assert env["pinned"] is True
