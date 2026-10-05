"""Surface-aware command strings (CR1/CR2), the dispatch --json single-object
guarantee (C1), the runner argv[0] diagnostic (C2), and the rate exit code (C5).
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from evalroute import cli, flywheel, routing

from tests.test_flywheel import home  # noqa: F401  re-exported fixture
from tests.test_dispatch import STUB  # the same child stub test_dispatch uses


def _rate_ns(**kw):
    base = dict(evalroute_action="rate", verdict="pass", lane=None,
                route_id=None, model=None, effort=None, note=None,
                dry_run=False, json=False)
    base.update(kw)
    return SimpleNamespace(**base)


# ----------------------------------------------------------------- surfaces

def test_cmd_rate_per_surface():
    routing.set_surface("cli")
    assert routing.cmd_rate("abc", "m", "high", note="...") == (
        'evalroute rate pass|fail --route-id abc --model m --effort high --note "..."')
    assert routing.cmd_rate("abc", "m", "high") == (
        'evalroute rate pass|fail --route-id abc --model m --effort high --note "why"')
    routing.set_surface("hermes-cli")
    assert routing.cmd_rate("abc", "m", "high", note="...").startswith(
        "hermes evalroute rate pass|fail")
    routing.set_surface("hermes-chat")
    assert routing.cmd_rate("abc", "m", "high") == "/rate pass|fail --route-id abc --note why"
    assert routing.cmd_rate() == "/rate pass|fail --note why"
    assert routing.cmd_rate(route_id="abc", verdict="skip", note="") == "/rate skip --route-id abc"


def test_cmd_route_lane_and_switch_arm_per_surface(home):
    lane = routing._lane_by_id("routine-coding")
    model, effort = lane["model"], routing._effort_for_override(lane)
    routing.set_surface("cli")
    assert routing.cmd_route_lane() == 'evalroute route --lane <id> "<same task>"'
    assert routing.cmd_switch_arm(model, effort) == f"(run on {model} @ {effort})"
    routing.set_surface("hermes-cli")
    assert routing.cmd_route_lane().startswith("hermes evalroute route --lane")
    routing.set_surface("hermes-chat")
    assert routing.cmd_route_lane() == "/route --lane <id> <same task>"
    assert routing.cmd_switch_arm(model, effort) == f"/model {model} then /reasoning {effort}"


def test_set_surface_rejects_unknown():
    with pytest.raises(ValueError):
        routing.set_surface("telegram")


def test_cli_surface_card_has_no_slash_or_hermes(home):
    routing.set_surface("cli")
    lane = routing._lane_by_id("routine-coding")
    card = routing.route_card(lane, 0.9, ["test"], method="rules-strong",
                              route_id="abc123")
    assert "/model" not in card and "/rate" not in card and "/route" not in card
    assert "hermes" not in card
    assert "evalroute rate pass|fail --route-id abc123 --model" in card
    assert f"next:     run on {lane['model']} @ {routing._effort_for_override(lane)}" in card
    assert 'wrong lane? evalroute route --lane <id> "<same task>"' in card


def test_console_main_pins_cli_surface(monkeypatch):
    # evalroute_cli claims hermes-cli only when the surface is unclaimed;
    # main() pins "cli" first so the console script keeps plain strings.
    routing.reset_surface()
    ns = SimpleNamespace(evalroute_action="rate", verdict="pass", lane=None,
                         route_id="bogus", model=None, effort=None, note=None,
                         dry_run=False, json=False)
    assert cli.evalroute_cli(ns) == 1
    assert routing.SURFACE == "hermes-cli"  # unclaimed -> plugin handler claims it
    routing.set_surface("cli")             # what main() does
    assert cli.evalroute_cli(ns) == 1
    assert routing.SURFACE == "cli"


# ------------------------------------------------------------- C5: exit code

def test_rate_exit_1_on_bogus_route_id(home, capsys):
    rc = cli.evalroute_cli(_rate_ns(route_id="bogus"))
    assert rc == 1
    assert "nothing rated" in capsys.readouterr().out


def test_rate_exit_0_when_logged(home, capsys):
    lane = routing._lane_by_id("routine-coding")
    flywheel.note_route("rate exit code test", lane, "rules-strong", 0.9)
    rc = cli.evalroute_cli(_rate_ns(model="z-ai/glm-5.3-flash", effort="medium",
                                    note="tests green"))
    assert rc == 0
    assert capsys.readouterr().out.startswith("logged: pass")


def test_rate_json_exit_1_when_not_logged(home, capsys):
    rc = cli.evalroute_cli(_rate_ns(route_id="bogus", json=True))
    data = json.loads(capsys.readouterr().out)
    assert rc == 1 and data["logged"] is False


# ------------------------------------------- C1: dispatch --json one object

def _stub_env(tmp_path, monkeypatch, exit_code=0):
    stub = tmp_path / "hermes_stub.py"
    stub.write_text(STUB, encoding="utf-8")
    argv_file = tmp_path / "argv.log"
    monkeypatch.setenv("EVALROUTE_HERMES_BIN", f"{sys.executable} {stub}")
    monkeypatch.setenv("STUB_ARGV_FILE", str(argv_file))
    monkeypatch.setenv("STUB_EXIT", str(exit_code))
    return stub


def _dispatch_ns(brief, **kw):
    base = dict(evalroute_action="dispatch", brief=str(brief), lane="routine-coding",
                indir=None, task=None, out=None, timeout=None, rate_on_exit=None,
                dry_run=False, follow=False, runner=None, json=False)
    base.update(kw)
    return SimpleNamespace(**base)


def test_dispatch_json_rate_on_exit_is_one_object(home, tmp_path, monkeypatch, capsys):
    _stub_env(tmp_path, monkeypatch, exit_code=3)
    brief = tmp_path / "b.md"
    brief.write_text("# Do the thing\n\nbody\n", encoding="utf-8")
    rc = cli.evalroute_cli(_dispatch_ns(brief, rate_on_exit="fail", json=True))
    data = json.loads(capsys.readouterr().out)  # whole stdout is one JSON object
    assert rc == 3
    assert data["exit"] == 3
    assert data["rate_on_exit"]["logged"] is True
    assert "logged: fail" in data["rate_on_exit"]["message"]


def test_dispatch_json_without_rate_on_exit_stays_clean(home, tmp_path, monkeypatch, capsys):
    _stub_env(tmp_path, monkeypatch, exit_code=3)
    brief = tmp_path / "b.md"
    brief.write_text("# Do the thing\n\nbody\n", encoding="utf-8")
    rc = cli.evalroute_cli(_dispatch_ns(brief, json=True))
    data = json.loads(capsys.readouterr().out)
    assert rc == 3 and "rate_on_exit" not in data


def test_dispatch_human_rate_on_exit_still_four_lines(home, tmp_path, monkeypatch, capsys):
    _stub_env(tmp_path, monkeypatch, exit_code=3)
    brief = tmp_path / "b.md"
    brief.write_text("# Do the thing\n\nbody\n", encoding="utf-8")
    rc = cli.evalroute_cli(_dispatch_ns(brief, rate_on_exit="fail"))
    assert rc == 3
    assert len(capsys.readouterr().out.strip().splitlines()) == 4


# --------------------------------------------- C2: quoted Windows paths

def test_runner_template_quoted_backslash_path(tmp_path, monkeypatch, capsys):
    """A template quoting sys.executable (backslashes on Windows) must spawn."""
    stub = tmp_path / "runner_stub.py"
    stub.write_text("print('REPORT BODY')\n", encoding="utf-8")
    brief = tmp_path / "b.md"
    brief.write_text("# Do the thing\n\nbody\n", encoding="utf-8")
    template = f'"{sys.executable}" "{stub}" {{model}}'
    rc = cli.evalroute_cli(_dispatch_ns(brief, runner=template))
    err = capsys.readouterr().err
    assert rc == 0, err
    assert brief.with_name("b.report.md").read_text(encoding="utf-8") == "REPORT BODY\n"


def test_runner_missing_argv0_prints_diagnostic(tmp_path, monkeypatch, capsys):
    brief = tmp_path / "b.md"
    brief.write_text("# Do the thing\n\nbody\n", encoding="utf-8")
    template = '"C:/definitely/not/here/python.exe" {model}'
    rc = cli.evalroute_cli(_dispatch_ns(brief, runner=template))
    err = capsys.readouterr().err
    assert rc == 127
    assert "runner argv[0] not found: C:/definitely/not/here/python.exe" in err
    assert "resolved argv:" in err
