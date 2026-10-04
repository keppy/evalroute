"""dispatch --runner templates, the *.dispatch.json sidecar, and --json on
rate / sync / dispatch / report.

Everything stubbed or fixture-based: no network, no real Hermes, no live
profile (conftest isolates HERMES_HOME).
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from evalroute import cli, dispatch, flywheel, report, routing

from tests.test_flywheel import home  # noqa: F401  re-exported fixture
from tests.test_report import world  # noqa: F401  re-exported fixture

STUB = r'''
import os, sys
with open(os.environ["STUB_ARGV_FILE"], "a", encoding="utf-8") as f:
    f.write(repr(sys.argv) + "\n")
print("REPORT BODY")
print("session_id: stub123", file=sys.stderr)
sys.exit(0)
'''


def _make_stub(tmp_path, monkeypatch):
    stub = tmp_path / "hermes_stub.py"
    stub.write_text(STUB, encoding="utf-8")
    argv_file = tmp_path / "argv.log"
    argv_file.write_text("", encoding="utf-8")
    monkeypatch.setenv("STUB_ARGV_FILE", str(argv_file))
    return stub, argv_file


def _make_brief(tmp_path, text="# Do the thing\n\nSecond paragraph.\n"):
    brief = tmp_path / "brief.md"
    brief.write_text(text, encoding="utf-8")
    return brief


def _dispatch(args_dict):
    ns = SimpleNamespace(evalroute_action="dispatch", **args_dict)
    return cli.evalroute_cli(ns)


def _base_args(brief):
    return dict(brief=str(brief), lane="routine-coding", indir=None, task=None,
                out=None, timeout=None, rate_on_exit=None, dry_run=False,
                runner=None, json=False)


# --------------------------------------------------------- runner templates

def test_runner_template_substitution_spaces_and_quotes(tmp_path, monkeypatch):
    brief = tmp_path / "some dir" / "my brief.md"
    brief.parent.mkdir()
    brief.write_text("# Do the thing\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    template = ('python -X utf8 runner.py --model {model} --effort {effort} '
                '--prompt {brief_text} --out {brief} --dir {indir}')
    argv = dispatch._build_runner_argv(
        template, "z-ai/glm-5.3", "high", "nous",
        brief, "C:/a b/c 'd'")
    assert argv[0] == "python"
    assert argv[argv.index("--model") + 1] == "z-ai/glm-5.3"
    assert argv[argv.index("--effort") + 1] == "high"
    assert argv[argv.index("--out") + 1] == str(tmp_path / "some dir" / "my brief.md")
    assert argv[argv.index("--dir") + 1] == "C:/a b/c 'd'"
    # brief_text is the file's contents, one token even with newlines
    text = argv[argv.index("--prompt") + 1]
    assert "# Do the thing" in text


def test_runner_template_may_omit_effort_and_indir(tmp_path, monkeypatch):
    brief = _make_brief(tmp_path)
    argv = dispatch._build_runner_argv(
        "claude -p --model {model} {brief_text}", "m1", "high", "nous",
        brief, None)
    assert argv == ["claude", "-p", "--model", "m1", brief.read_text(encoding="utf-8")]


def test_named_runner_resolves_to_template():
    assert dispatch._resolve_runner("hermes") == dispatch._NAMED_RUNNERS["hermes"]
    assert dispatch._resolve_runner(None) == dispatch._NAMED_RUNNERS["hermes"]


def test_env_runner_template_used(tmp_path, monkeypatch, capsys):
    stub, argv_file = _make_stub(tmp_path, monkeypatch)
    monkeypatch.setenv("EVALROUTE_RUNNER",
                       f"{sys.executable} {stub} --model {{model}} --file {{brief}}")
    monkeypatch.delenv("EVALROUTE_HERMES_BIN", raising=False)
    brief = _make_brief(tmp_path)
    rc = _dispatch(_base_args(brief))
    assert rc == 0
    argv = eval(argv_file.read_text(encoding="utf-8").strip())
    lane = routing._lane_by_id("routine-coding")
    assert argv[argv.index("--model") + 1] == lane["model"]
    assert argv[argv.index("--file") + 1] == str(brief)


def test_unknown_runner_name_exit_2_with_list(tmp_path, monkeypatch, capsys):
    brief = _make_brief(tmp_path)
    args = _base_args(brief)
    args["runner"] = "aider"
    rc = _dispatch(args)
    assert rc == 2
    err = capsys.readouterr().err
    assert "unknown runner 'aider'" in err
    assert "hermes" in err


def test_hermes_env_bin_still_works_back_compat(tmp_path, monkeypatch, capsys):
    stub, argv_file = _make_stub(tmp_path, monkeypatch)
    monkeypatch.setenv("EVALROUTE_HERMES_BIN", f"{sys.executable} {stub}")
    brief = _make_brief(tmp_path)
    rc = _dispatch(_base_args(brief))
    assert rc == 0
    argv = eval(argv_file.read_text(encoding="utf-8").strip())
    assert "--query-file" in argv and str(brief) in argv


# ------------------------------------------------------------------ sidecar

def test_sidecar_written_on_real_dispatch_and_appends(tmp_path, monkeypatch):
    stub, argv_file = _make_stub(tmp_path, monkeypatch)
    monkeypatch.setenv("EVALROUTE_HERMES_BIN", f"{sys.executable} {stub}")
    brief = _make_brief(tmp_path)
    assert _dispatch(_base_args(brief)) == 0
    sidecar_path = brief.with_name("brief.dispatch.json")
    assert sidecar_path.exists()
    data = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert [set(r) == set(dispatch._SIDECAR_KEYS) for r in data["runs"]] == [True]
    run = data["runs"][-1]
    assert run["exit"] == 0 and run["session_id"] == "stub123"
    assert run["lane"] == "routine-coding" and run["provider"] == "nous"
    assert run["report"].endswith("brief.report.md") and run["rate_line"].startswith("rate it:")
    assert run["runner"] == "hermes"
    # re-run appends, newest last
    assert _dispatch(_base_args(brief)) == 0
    data = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert len(data["runs"]) == 2
    assert data["runs"][-1]["ended"] >= data["runs"][-2]["ended"]


def test_sidecar_written_on_dry_run(tmp_path, monkeypatch):
    brief = _make_brief(tmp_path)
    args = _base_args(brief)
    args["dry_run"] = True
    assert _dispatch(args) == 0
    data = json.loads(brief.with_name("brief.dispatch.json").read_text(encoding="utf-8"))
    run = data["runs"][-1]
    assert run["exit"] is None and run["session_id"] is None
    assert run["route_id"]


# ------------------------------------------------------------ report sidecar

def _state_db(home, sid, cwd, cost=0.40):
    db = home / "state.db"
    con = sqlite3.connect(db)
    con.execute("create table sessions (id TEXT PRIMARY KEY, title TEXT, cwd TEXT, "
                "model TEXT, started_at REAL, ended_at REAL, last_activity_at REAL, "
                "message_count INTEGER, tool_call_count INTEGER, input_tokens INTEGER, "
                "output_tokens INTEGER, estimated_cost_usd REAL, profile_name TEXT)")
    con.execute("create table messages (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "session_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT, "
                "tool_call_id TEXT, tool_calls TEXT, tool_name TEXT, timestamp REAL)")
    now = time.time()
    con.execute("insert into sessions values (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (sid, "sidecar session", cwd, "z-ai/glm-5.3",
                 now - 60, now, now, 10, 2, 1000, 100, cost, "default"))
    con.commit()
    con.close()


def test_report_sessions_row_from_sidecar(home, tmp_path):
    trains = home / "trains" / "2026-10-D"
    trains.mkdir(parents=True)
    (trains / "README.md").write_text("# Train 2026-10-D\n", encoding="utf-8")
    brief = trains / "01-worker.md"
    brief.write_text("# Worker brief\n", encoding="utf-8")
    report_path = brief.with_name("01-worker.report.md")
    report_path.write_text("REPORT BODY\n", encoding="utf-8")
    run = {"route_id": "abc12345", "lane": "routine-coding",
           "model": "z-ai/glm-5.3", "effort": "high", "provider": "nous",
           "runner": "hermes", "brief": str(brief), "indir": None,
           "started": "2026-10-03T01:00:00-0700",
           "ended": "2026-10-03T01:01:00-0700", "duration_s": 60.0,
           "exit": 0, "session_id": "20261003_010101_worker1",
           "report": str(report_path), "rate_line": "rate it: ..."}
    (brief.with_name("01-worker.dispatch.json")).write_text(
        json.dumps({"runs": [run]}), encoding="utf-8")
    _state_db(home, "20261003_010101_worker1", str(home / "elsewhere"), cost=0.40)

    model = report._data_model(home / "trains", None)
    train = model["trains"]["trains"][0]
    assert train["name"] == "2026-10-D"
    assert train["briefs"][0]["route_id"] == "abc12345"
    assert train["briefs"][0]["session_id"] == "20261003_010101_worker1"
    assert model["sessions"]["trains"][0]["sessions"][0]["cost_usd"] == 0.40

    # and the HTML page renders the same row
    out = home / "report.html"
    report.run(SimpleNamespace(out=str(out), trains=str(home / "trains"),
                               factory_json=None, open=False, watch=None, json=False))
    html = out.read_text(encoding="utf-8")
    assert "abc12345" in html and "20261003_010101_worker1" in html
    assert "$0.40" in html


def test_report_falls_back_to_session_line_when_no_sidecar(home, tmp_path):
    trains = home / "trains" / "2026-10-E"
    trains.mkdir(parents=True)
    (trains / "README.md").write_text("# Train 2026-10-E\n", encoding="utf-8")
    brief = trains / "01-legacy.md"
    brief.write_text("# Legacy\n", encoding="utf-8")
    brief.with_name("01-legacy.report.md").write_text(
        "report: 01-legacy.report.md   session: legacy_sess_1\n", encoding="utf-8")
    model = report._data_model(home / "trains", None)
    assert model["trains"]["trains"][0]["briefs"][0]["session_id"] == "legacy_sess_1"


# ------------------------------------------------------------- --json verbs

def test_rate_json_shape(home, capsys):
    lane = routing._lane_by_id("routine-coding")
    flywheel.note_route("fix the failing test", lane, "rules-strong", 0.9)
    ns = SimpleNamespace(evalroute_action="rate", verdict="pass", lane=None,
                         route_id=None, model="z-ai/glm-5.3-flash",
                         effort="medium", note="tests green", json=True)
    rc = cli.evalroute_cli(ns)
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["logged"] is True
    assert data["verdict"] == "pass"
    assert data["lane"] == "routine-coding"
    assert data["model"] == "z-ai/glm-5.3-flash"
    assert data["effort"] == "medium"
    assert data["arm_attribution"] == "explicit_user"
    assert data["route_id"]
    assert "message" in data


def test_rate_human_output_unchanged_without_json(home, capsys):
    lane = routing._lane_by_id("routine-coding")
    flywheel.note_route("fix the failing test", lane, "rules-strong", 0.9)
    ns = SimpleNamespace(evalroute_action="rate", verdict="pass", lane=None,
                         route_id=None, model=None, effort=None, note=None,
                         json=False)
    cli.evalroute_cli(ns)
    out = capsys.readouterr().out
    assert out.startswith("logged: pass for lane routine-coding")
    assert "{" != out[0]


def test_sync_status_json_shape(home, capsys):
    ns = SimpleNamespace(evalroute_action="sync", status=True, clear=False,
                         revision=None, json=True)
    assert cli.evalroute_cli(ns) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["active"] == "bundled"
    assert data["sha"] is None
    assert data["lanes"] >= 1 and data["measured"] >= 0
    assert Path(data["path"]).exists()


def test_sync_clear_json_shape(home, capsys):
    ns = SimpleNamespace(evalroute_action="sync", status=False, clear=True,
                         revision=None, json=True)
    assert cli.evalroute_cli(ns) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["active"] == "bundled"


def test_sync_human_status_unchanged(home, capsys):
    ns = SimpleNamespace(evalroute_action="sync", status=True, clear=False,
                         revision=None, json=False)
    cli.evalroute_cli(ns)
    out = capsys.readouterr().out
    assert out == f"active table: bundled\npath: {routing.ROUTES_FILE}\n"


def test_dispatch_json_dry_run_prints_sidecar_object(tmp_path, monkeypatch, capsys):
    brief = _make_brief(tmp_path)
    args = _base_args(brief)
    args["dry_run"] = True
    args["json"] = True
    assert _dispatch(args) == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["route_id"] and data["lane"] == "routine-coding"
    assert data["rate_line"].startswith("rate it:")
    # the card still goes to stderr
    assert "run:" in captured.err or "lane" in captured.err
    # human output (no --json) is still the three lines
    args2 = _base_args(brief)
    args2["dry_run"] = True
    args2["lane"] = "dl-ml-research-engineering"
    _dispatch(args2)
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0].startswith("would run: ")


def test_report_json_prints_data_model(world, capsys):
    ns = SimpleNamespace(evalroute_action="report", out=None, open=False,
                         watch=None, trains=None, factory_json=None, json=True)
    rc = cli.evalroute_cli(ns)
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    for key in ("generated", "table", "ledger_path", "pending", "outcome_rows",
                "tally", "methods", "sessions", "trains", "drift"):
        assert key in data, key
