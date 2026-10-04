"""dispatch: route, spawn a stubbed hermes chat on that arm, print the rate line."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace

from evalroute import flywheel
from evalroute import routing
from evalroute import cli

from tests.test_flywheel import home  # noqa: F401  re-exported fixture

STUB = r'''
import os, sys
with open(os.environ["STUB_ARGV_FILE"], "a", encoding="utf-8") as f:
    f.write(repr(sys.argv) + "\n")
print("REPORT BODY")
print("session_id: stub123", file=sys.stderr)
sys.exit(int(os.environ.get("STUB_EXIT", "0")))
'''


def _make_stub(tmp_path, monkeypatch):
    stub = tmp_path / "hermes_stub.py"
    stub.write_text(STUB, encoding="utf-8")
    argv_file = tmp_path / "argv.log"
    monkeypatch.setenv("EVALROUTE_HERMES_BIN", f"{sys.executable} {stub}")
    monkeypatch.setenv("STUB_ARGV_FILE", str(argv_file))
    return argv_file


def _make_brief(tmp_path, text="# Do the thing\n\nSecond paragraph.\n"):
    brief = tmp_path / "brief.md"
    brief.write_text(text, encoding="utf-8")
    return brief


def _dispatch(args_dict):
    ns = SimpleNamespace(evalroute_action="dispatch", **args_dict)
    return cli.evalroute_cli(ns)


def _outcomes():
    return [r for r in flywheel.read_labels() if r["kind"] == "outcome"]


def _routes():
    return [r for r in flywheel.read_labels() if r["kind"] == "route"]


def test_dispatch_spawns_stub_and_prints_three_lines(home, tmp_path, monkeypatch, capsys):
    argv_file = _make_stub(tmp_path, monkeypatch)
    brief = _make_brief(tmp_path)
    rc = _dispatch(dict(brief=str(brief), lane="routine-coding", indir=str(tmp_path),
                        task=None, out=None, timeout=None, rate_on_exit=None,
                        dry_run=False))
    out = capsys.readouterr().out
    assert rc == 0
    argv = eval(argv_file.read_text(encoding="utf-8").strip())
    lane = routing._lane_by_id("routine-coding")
    assert "-m" in argv and lane["model"] in argv
    assert argv[argv.index("--provider") + 1] == "nous"
    assert argv[argv.index("--reasoning") + 1] == routing._effort_for_override(lane)
    assert argv[argv.index("--query-file") + 1] == str(brief)
    assert argv[argv.index("--in") + 1] == str(tmp_path)
    report = brief.with_name("brief.report.md")
    assert report.read_text(encoding="utf-8") == "REPORT BODY\n"
    lines = [l for l in out.strip().splitlines()]
    assert len(lines) == 3
    route_id = _routes()[-1]["id"]
    assert lines[0].startswith(f"dispatched route {route_id} ->") and lines[0].endswith("exit 0, 0m00s")
    assert f"session: stub123" in lines[1] and str(report) in lines[1]
    assert lines[2].startswith("rate it:  hermes evalroute rate pass|fail --route-id ")
    assert f"--model {lane['model']}" in lines[2] and f"--effort {routing._effort_for_override(lane)}" in lines[2]


def test_dispatch_default_task_is_first_paragraph(home, tmp_path, monkeypatch, capsys):
    _make_stub(tmp_path, monkeypatch)
    brief = _make_brief(tmp_path)
    _dispatch(dict(brief=str(brief), lane=None, indir=None, task=None, out=None,
                   timeout=None, rate_on_exit=None, dry_run=False))
    row = _routes()[-1]
    assert row["task"] == "Do the thing"


def test_dispatch_rate_on_exit_fail(home, tmp_path, monkeypatch, capsys):
    _make_stub(tmp_path, monkeypatch)
    monkeypatch.setenv("STUB_EXIT", "3")
    brief = _make_brief(tmp_path)
    rc = _dispatch(dict(brief=str(brief), lane="routine-coding", indir=None, task=None,
                        out=None, timeout=None, rate_on_exit="fail", dry_run=False))
    out = capsys.readouterr().out
    assert rc == 3
    outcomes = _outcomes()
    assert len(outcomes) == 1
    oc = outcomes[0]
    assert oc["rated"] == "fail"
    assert oc["actual_model"] == routing._lane_by_id("routine-coding")["model"]
    assert oc["arm_attribution"] == "explicit_user"
    assert "exit 3" in oc.get("note", "")
    assert len(out.strip().splitlines()) == 4  # confirmation as a fourth line


def test_dispatch_no_flag_no_rate(home, tmp_path, monkeypatch, capsys):
    _make_stub(tmp_path, monkeypatch)
    monkeypatch.setenv("STUB_EXIT", "3")
    brief = _make_brief(tmp_path)
    rc = _dispatch(dict(brief=str(brief), lane="routine-coding", indir=None, task=None,
                        out=None, timeout=None, rate_on_exit=None, dry_run=False))
    assert rc == 3
    assert _outcomes() == []


def test_dispatch_dry_run(home, tmp_path, monkeypatch, capsys):
    argv_file = _make_stub(tmp_path, monkeypatch)
    brief = _make_brief(tmp_path)
    rc = _dispatch(dict(brief=str(brief), lane="routine-coding", indir=None, task=None,
                        out=None, timeout=None, rate_on_exit=None, dry_run=True))
    out = capsys.readouterr().out
    assert rc == 0
    assert not argv_file.exists()  # nothing spawned
    assert "would run:" in out
    assert "--query-file" in out
    assert len(_routes()) == 1
    assert "rate it:" in out
    assert _outcomes() == []


def test_dispatch_timeout(home, tmp_path, monkeypatch, capsys):
    stub = tmp_path / "hermes_stub.py"
    stub.write_text("import time; time.sleep(5)\n", encoding="utf-8")
    monkeypatch.setenv("EVALROUTE_HERMES_BIN", f"{sys.executable} {stub}")
    brief = _make_brief(tmp_path)
    rc = _dispatch(dict(brief=str(brief), lane="routine-coding", indir=None, task=None,
                        out=None, timeout=1, rate_on_exit=None, dry_run=False))
    assert rc == 124
    assert brief.with_name("brief.report.md").exists()


def test_dispatch_no_route_id_no_spawn(home, tmp_path, monkeypatch, capsys):
    argv_file = _make_stub(tmp_path, monkeypatch)
    brief = _make_brief(tmp_path)
    monkeypatch.setattr(routing, "_note_route", lambda *a, **k: None)
    rc = _dispatch(dict(brief=str(brief), lane="routine-coding", indir=None, task=None,
                        out=None, timeout=None, rate_on_exit=None, dry_run=False))
    err = capsys.readouterr()
    assert rc == 1
    assert not argv_file.exists()  # nothing spawned
    assert "rate it:" not in err.out
    assert "route could not be recorded" in err.err


def test_dispatch_timeout_kills_grandchild(home, tmp_path, monkeypatch, capsys):
    stub = tmp_path / "hermes_stub.py"
    stub.write_text(
        "import os, subprocess, sys, time\n"
        "gc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        "with open(os.environ['GC_PID_FILE'], 'w') as f:\n"
        "    f.write(str(gc.pid))\n"
        "time.sleep(30)\n",
        encoding="utf-8")
    gc_pid_file = tmp_path / "gc.pid"
    monkeypatch.setenv("EVALROUTE_HERMES_BIN", f"{sys.executable} {stub}")
    monkeypatch.setenv("GC_PID_FILE", str(gc_pid_file))
    brief = _make_brief(tmp_path)
    rc = _dispatch(dict(brief=str(brief), lane="routine-coding", indir=None, task=None,
                        out=None, timeout=1, rate_on_exit=None, dry_run=False))
    assert rc == 124
    gc_pid = int(gc_pid_file.read_text())
    if os.name == "nt":
        gone = False
        for _ in range(6):
            out = subprocess.run(["tasklist", "/FI", f"PID eq {gc_pid}"],
                                 capture_output=True, text=True).stdout
            if str(gc_pid) not in out:
                gone = True
                break
            time.sleep(0.5)
        assert gone, f"grandchild {gc_pid} still alive"
    else:
        with pytest.raises(ProcessLookupError):
            os.kill(gc_pid, 0)


def test_dispatch_route_id_comment_replaces(home, tmp_path, monkeypatch, capsys):
    _make_stub(tmp_path, monkeypatch)
    lane = routing._lane_by_id("routine-coding")
    old_id = flywheel.note_route("Do the thing", lane, "llm", 0.5)
    brief = _make_brief(tmp_path, text=f"<!-- evalroute: route-id={old_id} -->\n\n# Do the thing\n")
    rc = _dispatch(dict(brief=str(brief), lane="routine-coding", indir=None, task=None,
                        out=None, timeout=None, rate_on_exit=None, dry_run=False))
    assert rc == 0
    recs = flywheel.read_labels()
    routes = [r for r in recs if r["kind"] == "route"]
    assert len(routes) == 2
    assert routes[-1]["method"] == "pinned"
    replaced = [r for r in recs if r["kind"] == "outcome" and r["rated"] == "skip"]
    assert replaced and replaced[-1]["consumes_id"] == old_id


def test_dispatch_imported_as_package_module():
    """In a real package the path-load + sibling-injection hack is gone:
    dispatch is imported normally as evalroute.dispatch."""
    from evalroute import dispatch
    assert dispatch.__name__ == "evalroute.dispatch"
    assert dispatch.__package__ == "evalroute"
