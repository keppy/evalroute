"""report: one static HTML page over ledger, sessions, trains, and drift.

Fixtures only — no network, no real HERMES_HOME (conftest isolates it), no
real Hermes. The page must be self-contained: no <script src=, no <link href=.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from evalroute import flywheel, report, routing


# ---------------------------------------------------------------- fixtures

def _make_ledger(home):
    """2 lanes x 2 arms, 1 pending route, 1 dispatcher correction row."""
    lane_a = routing._lane_by_id("routine-coding")
    lane_b = routing._lane_by_id("dl-ml-research-engineering")
    # Route 1: routine-coding, arm flash@medium, pass
    flywheel.note_route("fix the failing test", lane_a, "rules-strong", 0.9)
    flywheel.handle_rate("pass --model z-ai/glm-5.3-flash --effort medium "
                         "--note clean run, tests green")
    # Route 2: routine-coding, arm glm@high, fail (corrected by dispatcher)
    rid2 = flywheel.note_route("port the shim to windows", lane_a, "llm", 0.7)
    flywheel.handle_rate("fail --model z-ai/glm-5.3 --effort high "
                         "--note regression on py3.12")
    flywheel._append({"kind": "rating_correction", "consumes_id": rid2,
                      "from": "fail", "to": "pass",
                      "note": "dispatcher re-verified: flaky fixture"})
    # Route 3: dl-ml, arm glm@high, pass
    flywheel.note_route("audit the rl training plan", lane_b, "pinned", 1.0)
    flywheel.handle_rate("pass --model z-ai/glm-5.3 --effort high "
                         "--note plan matches the receipts")
    # Route 4: dl-ml, arm flash@medium, skip — appended the way the
    # dl-ecosystem dispatcher does (handle_rate skip records no arm)
    rid4 = flywheel.note_route("skim the eval matrix", lane_b, "rules-weak", 0.4)
    flywheel._append({"kind": "outcome", "rated": "skip", "route_lane": lane_b["id"],
                      "actual_model": "z-ai/glm-5.3-flash", "actual_effort": "medium",
                      "arm_attribution": "explicit_user",
                      "consumes": time.time(), "consumes_id": rid4,
                      "note": "duplicate of route 3"})
    flywheel._MEMORY["route"] = None
    # Route 5: pending, never rated
    flywheel.note_route("a task nobody rated yet", lane_b, "llm", 0.6)


def _make_state_db(home):
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
    con.executemany(
        "insert into sessions values (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            ("20261003_010101_worker1", "Thin plugin adapter",
             str(home / "trains" / "2026-10-C"), "z-ai/glm-5.3",
             now - 7200, now - 3600, now - 3600, 120, 40, 50000, 8000, 0.25, "default"),
            ("20261003_020202_worker2", "Seed evalroute library",
             str(home / "trains" / "2026-10-C"), "z-ai/glm-5.3-flash",
             now - 3600, now - 1800, now - 1800, 200, 90, 90000, 12000, 0.75, "default"),
            ("20260101_000000_unrelated", "unrelated session",
             "C:\\elsewhere", "some/model",
             now - 999999, None, None, 1, 0, 100, 10, 99.0, "default"),
        ])
    con.commit()
    con.close()
    return db


def _make_trains(home):
    trains = home / "trains"
    train_c = trains / "2026-10-C"
    train_c.mkdir(parents=True)
    (train_c / "README.md").write_text(
        "# Train 2026-10-C\n\n## Dispatch log\n\n"
        "| step | route id | lane | arm | exit | verified | rated |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| 01 seed | `abc12345` | dl-ml | glm-5.3@high | 0 | lib seeded | pass |\n"
        "| 02 plugin | `fdec6789` | routine | flash@medium | 0 | thin adapter | pass |\n",
        encoding="utf-8")
    (train_c / "01-seed.md").write_text("# Seed the library\n", encoding="utf-8")
    (train_c / "01-seed.report.md").write_text(
        "dispatched route abc12345 -> z-ai/glm-5.3 @ high (dl-ml), exit 0, 0m10s\n"
        "report: 01-seed.report.md   session: 20261003_010101_worker1\n"
        "rate it:  hermes evalroute rate pass --route-id abc12345\n",
        encoding="utf-8")
    (train_c / "02-plugin.md").write_text("# Thin the plugin\n", encoding="utf-8")
    # 02-plugin has no report.md -> not run
    return trains


def _make_factory_json(home):
    path = home / "factory-check.json"
    path.write_text(json.dumps([
        {"repo": "plugin-evalroute", "check": "version_split", "severity": "warn",
         "message": "plugin version 0.5.1 vs catalog pin"},
        {"repo": "thomas", "check": "gitignore", "severity": "error",
         "message": ".venv is tracked"},
    ]), encoding="utf-8")
    return path


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME (the report reads the ledger + state.db from it)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    flywheel._MEMORY.clear()
    routing.reset_routes_cache()
    yield tmp_path
    flywheel._MEMORY.clear()


@pytest.fixture
def world(home, tmp_path):
    """Ledger + state.db + trains + factory json, all inside HERMES_HOME."""
    _make_ledger(home)
    _make_state_db(home)
    trains = _make_trains(home)
    factory = _make_factory_json(home)
    return SimpleNamespace(home=home, trains=trains, factory=factory)


def _run_report(world, **overrides):
    defaults = dict(out=str(world.home / "report.html"),
                    trains=str(world.trains),
                    factory_json=str(world.factory),
                    open=False, watch=None)
    defaults.update(overrides)
    args = SimpleNamespace(**defaults)
    rc = report.run(args)
    html_text = Path(defaults["out"]).read_text(encoding="utf-8")
    return rc, html_text


# ------------------------------------------------------------------- tests

def test_report_file_written(world):
    rc, html_text = _run_report(world)
    assert rc == 0
    assert html_text.startswith("<!doctype html>")
    assert html_text.rstrip().endswith("</html>")


def test_report_self_contained(world):
    _, html_text = _run_report(world)
    assert "<script src=" not in html_text
    assert "<link href=" not in html_text
    assert "http://" not in html_text and "https://" not in html_text


def test_report_all_sections_present(world):
    _, html_text = _run_report(world)
    for heading in ("Now — pending routes", "Outcomes (user-confirmed arms)",
                    "Per-lane tally", "Classification methods",
                    "Sessions", "Trains", "Drift"):
        assert heading in html_text, heading


def test_report_pending_route_in_now(world):
    _, html_text = _run_report(world)
    assert "a task nobody rated yet" in html_text
    # the Now table appears before the outcomes table
    assert html_text.index("Now — pending routes") < html_text.index("Outcomes")


def test_report_correction_renders_under_outcome(world):
    _, html_text = _run_report(world)
    assert "corrected by dispatcher" in html_text
    # it renders as a second line under the outcome it corrects
    outcome_pos = html_text.index("regression on py3.12")
    corr_pos = html_text.index("corrected by dispatcher")
    assert corr_pos > outcome_pos
    # only the explicit_user outcomes are tabulated: 4 rows, correction included
    assert html_text.count("<td class='pass'>pass</td>") == 2
    assert html_text.count("<td class='fail'>fail</td>") == 1
    assert html_text.count("<td class='skip'>skip</td>") == 1


def test_report_tally_numbers(world):
    _, html_text = _run_report(world)
    # routine-coding: flash@medium n=1 pass=1; glm@high n=1 fail=1
    # dl-ml: glm@high n=1 pass=1; flash@medium n=1 skip=1
    tally_rows = re.findall(
        r"<tr data-lane='([^']+)' data-arm='([^']+)'>"
        r"<td>.*?</td><td>.*?</td><td[^>]*>.*?</td>"
        r"<td>(\d+)</td><td[^>]*>(\d+)</td><td[^>]*>(\d+)</td><td[^>]*>(\d+)</td>",
        html_text)
    got = {(lane, arm): tuple(int(x) for x in rest)
           for lane, arm, *rest in tally_rows}
    assert got[("routine-coding", "z-ai/glm-5.3-flash@medium")] == (1, 1, 0, 0)
    assert got[("routine-coding", "z-ai/glm-5.3@high")] == (1, 0, 1, 0)
    assert got[("dl-ml-research-engineering", "z-ai/glm-5.3@high")] == (1, 1, 0, 0)
    assert got[("dl-ml-research-engineering", "z-ai/glm-5.3-flash@medium")] == (1, 0, 0, 1)
    # provenance from the active route table is visible next to the counts
    assert "measured" in html_text or "priors" in html_text


def test_report_cost_sum(world):
    _, html_text = _run_report(world)
    # both fixture sessions ran under trains/2026-10-C: 0.25 + 0.75 = 1.00
    assert "Total 2026-10-C: $1.00 (2 sessions)" in html_text
    # the unrelated expensive session is excluded
    assert "$99.00" not in html_text


def test_report_absent_state_db_notes_not_crash(world):
    (world.home / "state.db").unlink()
    rc, html_text = _run_report(world)
    assert rc == 0
    assert "session store not found" in html_text
    assert "Now — pending routes" in html_text


def test_report_watch_one_regen_then_ctrl_c(world, monkeypatch, capsys):
    _, html_text = _run_report(world)  # prime the page once
    calls = {"n": 0}

    def fake_sleep(_seconds):
        calls["n"] += 1
        if calls["n"] >= 2:
            raise KeyboardInterrupt()

    monkeypatch.setattr(time, "sleep", fake_sleep)
    rc, html_text = _run_report(world, watch=1)
    assert rc == 0
    err = capsys.readouterr().err
    assert err.count("regenerated") == 1
    assert "watch stopped" in err


def test_report_trains_table_links(world):
    _, html_text = _run_report(world)
    # 01 done with its session id; 02 not run
    assert "01-seed.md" in html_text and "done" in html_text
    assert "20261003_010101_worker1" in html_text
    assert "02-plugin.md" in html_text and "not run" in html_text
    # route id from the README dispatch log
    assert "abc12345" in html_text


def test_report_drift_table(world):
    _, html_text = _run_report(world)
    assert "plugin-evalroute" in html_text
    assert "version_split" in html_text
    assert ".venv is tracked" in html_text


def test_report_footer_carries_command(world):
    rc, html_text = _run_report(world)
    assert "produced by: evalroute report" in html_text
    assert "--factory-json" in html_text


def test_report_header_meta(world):
    _, html_text = _run_report(world)
    assert "ledger:" in html_text
    from evalroute.routing import _lib_version
    assert f"evalroute {_lib_version()}" in html_text
    assert "table:" in html_text
