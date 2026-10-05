"""tests/test_flywheel_schema2.py — lane corrections in contribute, max_turns on the arm, report labels."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from evalroute import contribute
from evalroute import dispatch
from evalroute import flywheel
from evalroute import report
from evalroute import routing

BASE_TS = time.mktime(time.strptime("2026-09-29", "%Y-%m-%d"))

pytestmark = pytest.mark.usefixtures("hermes_cli")


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    yield tmp_path


def _write_labels(home, records):
    f = home / "evalroute" / "labels.jsonl"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


# ------------------------------------------------------------- contribute


def test_correction_row_emitted_no_task(home):
    records = [
        {"kind": "route", "id": "r1", "ts": BASE_TS, "task": "fix the failing test",
         "lane": "a", "model": "m", "effort": "medium", "method": "rules-strong",
         "confidence": 0.9},
        {"kind": "lane_correction", "ts": BASE_TS + 40, "from_lane": "a",
         "to_lane": "b", "task": "fix the failing test"},
    ]
    rows, summary = contribute.redact_report(records, salt=b"\x01" * 32)
    corr = [r for r in rows if r["kind"] == "lane_correction"]
    assert len(corr) == 1
    assert set(corr[0]) == {"kind", "from_lane", "to_lane", "method", "week", "schema"}
    assert "task" not in corr[0]
    assert corr[0]["from_lane"] == "a" and corr[0]["to_lane"] == "b"
    assert corr[0]["method"] == "rules-strong"
    assert corr[0]["schema"] == contribute.SCHEMA_VERSION
    assert summary["n_corrections"] == 1
    assert summary["by_correction"] == {"a->b": 1}


def test_correction_respects_cursor(home):
    records = [
        {"kind": "route", "id": "r1", "ts": BASE_TS, "task": "t" * 20, "lane": "a",
         "model": "m", "effort": "medium", "method": "pinned", "confidence": 1.0},
        {"kind": "lane_correction", "ts": BASE_TS + 40, "from_lane": "a", "to_lane": "b"},
    ]
    rows, summary = contribute.redact_report(records, salt=b"\x01" * 32, cursor=BASE_TS + 40)
    assert summary["n_corrections"] == 0
    assert rows == []


def test_outcome_max_turns_copied(home):
    task = "fix the NaN loss in our GRPO run"
    base = [
        {"kind": "route", "id": "r1", "ts": BASE_TS, "task": task, "lane": "a",
         "model": "m", "effort": "medium", "method": "rules-strong", "confidence": 0.9},
        {"kind": "route", "id": "r2", "ts": BASE_TS + 100, "task": task + " again",
         "lane": "a", "model": "m", "effort": "medium", "method": "rules-strong",
         "confidence": 0.9},
    ]
    records = [dict(base[0]),
               base[1],
               {"kind": "outcome", "ts": BASE_TS + 50, "rated": "pass",
                "route_lane": "a", "route_model": "m", "route_effort": "medium",
                "actual_model": "x", "actual_effort": "medium",
                "arm_attribution": "explicit_user", "method": "rules-strong",
                "confidence": 0.9, "consumes_id": "r1", "max_turns": 150},
               {"kind": "outcome", "ts": BASE_TS + 150, "rated": "fail",
                "route_lane": "a", "route_model": "m", "route_effort": "medium",
                "actual_model": "x", "actual_effort": "medium",
                "arm_attribution": "explicit_user", "method": "rules-strong",
                "confidence": 0.9, "consumes_id": "r2"}]
    rows = contribute.redact_report(records, salt=b"\x01" * 32)[0]
    outcomes = [r for r in rows if r["kind"] == "outcome"]
    assert len(outcomes) == 2
    got = {r["max_turns"] for r in outcomes}
    assert got == {150, None}
    # the row consuming r1 (which carried max_turns on the outcome) has it
    task_hashes = {r["task_hash"]: r["max_turns"] for r in outcomes}
    assert len(task_hashes) == 2
    assert 150 in task_hashes.values() and None in task_hashes.values()


def test_schema_version_2_on_every_row(home):
    task = "write a blog post about the encoder"
    records = [
        {"kind": "route", "id": "r1", "ts": BASE_TS, "task": task, "lane": "a",
         "model": "m", "effort": "medium", "method": "pinned", "confidence": 1.0},
        {"kind": "lane_correction", "ts": BASE_TS + 10, "from_lane": "a", "to_lane": "b"},
        {"kind": "outcome", "ts": BASE_TS + 20, "rated": "pass", "route_lane": "a",
         "route_model": "m", "route_effort": "medium", "actual_model": "x",
         "actual_effort": "medium", "arm_attribution": "explicit_user",
         "method": "pinned", "confidence": 1.0, "consumes_id": "r1"},
    ]
    assert contribute.SCHEMA_VERSION == 2
    rows, summary = contribute.redact_report(records, salt=b"\x01" * 32)
    assert summary["schema"] == 2
    assert rows and all(r["schema"] == 2 for r in rows)


def test_kept_whitelist_still_leaks_nothing(home):
    records = [
        {"kind": "route", "id": "r1", "ts": BASE_TS,
         "task": "fix the NaN loss; contact alice@example.org", "lane": "a",
         "model": "m", "effort": "medium", "method": "rules-strong", "confidence": 0.9},
        {"kind": "lane_correction", "ts": BASE_TS + 10, "from_lane": "a",
         "to_lane": "b", "task": "alice@example.org"},
        {"kind": "outcome", "ts": BASE_TS + 20, "rated": "pass", "route_lane": "a",
         "route_model": "m", "route_effort": "medium", "actual_model": "x",
         "actual_effort": "medium", "arm_attribution": "explicit_user",
         "method": "rules-strong", "confidence": 0.9, "consumes_id": "r1"},
    ]
    rows = contribute.redact(records, salt=b"\x01" * 32)
    for row in rows:
        assert set(row) <= set(contribute.KEPT)
    blob = json.dumps(rows)
    assert "alice@example.org" not in blob


# ------------------------------------------------------------- dispatch


def test_build_argv_max_turns_appended(tmp_path):
    brief = tmp_path / "b.md"
    brief.write_text("x", encoding="utf-8")
    argv = dispatch._build_argv("m", "medium", "nous", brief, None, 150)
    assert argv[-2:] == ["--max-turns", "150"]
    argv2 = dispatch._build_argv("m", "medium", "nous", brief, None, None)
    assert "--max-turns" not in argv2


def test_max_turns_cli_beats_config(home, tmp_path, monkeypatch):
    (home / "config.yaml").write_text("agent:\n  max_turns: 150\n", encoding="utf-8")
    assert dispatch._resolve_max_turns(42) == 42
    assert dispatch._resolve_max_turns(None) == 150


def test_max_turns_config_alone(home, monkeypatch):
    (home / "config.yaml").write_text("agent:\n  max_turns: 150\n", encoding="utf-8")
    assert dispatch._resolve_max_turns() == 150


def test_max_turns_neither(home, tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "nope"))
    assert dispatch._resolve_max_turns() is None


def test_sidecar_has_max_turns(home, tmp_path, monkeypatch):
    brief = tmp_path / "b.md"
    brief.write_text("# Do the thing\n\nSecond paragraph.\n", encoding="utf-8")
    ns = dict(brief=str(brief), lane="routine-coding", indir=None, task=None,
              out=None, timeout=None, rate_on_exit=None, dry_run=True,
              max_turns=150, runner=None, json=False)
    from evalroute import cli
    ns["evalroute_action"] = "dispatch"
    rc = cli.evalroute_cli(SimpleNamespace(**ns))
    assert rc == 0
    side = json.loads(brief.with_name("b.dispatch.json").read_text(encoding="utf-8"))
    assert side["runs"][-1]["max_turns"] == 150


def test_runner_template_max_turns_placeholder(tmp_path):
    brief = tmp_path / "b.md"
    brief.write_text("x", encoding="utf-8")
    argv = dispatch._build_runner_argv(
        "run.sh -m {model} --turns {max_turns} {indir}",
        "m", "medium", "nous", brief, None, 42)
    assert argv == ["run.sh", "-m", "m", "--turns", "42"]
    argv2 = dispatch._build_runner_argv(
        "run.sh -m {model} --turns {max_turns} {indir}",
        "m", "medium", "nous", brief, None, None)
    # a dropped value token takes its flag with it — no dangling `--turns`
    assert argv2 == ["run.sh", "-m", "m"]


# ------------------------------------------------------------- flywheel


def test_correction_rows_carry_ts(home, monkeypatch):
    original = flywheel.note_route("same task", routing._lane_by_id("prose"),
                                   "rules-weak", 0.5)
    routing.handle_route_command(
        f"--lane routine-coding --replace-route-id {original} same task")
    labels = flywheel.read_labels()
    corr = [r for r in labels if r["kind"] == "lane_correction"]
    assert len(corr) == 1 and isinstance(corr[0].get("ts"), float)
    assert corr[0]["from_lane"] == "prose" and corr[0]["to_lane"] == "routine-coding"
    # the rate --lane path writes the same shape
    rid = flywheel.note_route("another task", routing._lane_by_id("prose"),
                              "rules-weak", 0.5)
    flywheel.handle_rate("fail --route-id " + rid + " --lane routine-coding")
    corr = [r for r in flywheel.read_labels() if r["kind"] == "lane_correction"]
    assert len(corr) == 2 and isinstance(corr[1].get("ts"), float)


def test_rate_records_max_turns(home):
    rid = flywheel.note_route("some task to rate", routing._lane_by_id("prose"),
                              "rules-weak", 0.5)
    flywheel.handle_rate("pass --route-id " + rid + " --max-turns 150")
    outcome = [r for r in flywheel.read_labels()
               if r["kind"] == "outcome" and r.get("consumes_id") == rid][0]
    assert outcome["max_turns"] == 150
    rid2 = flywheel.note_route("another task to rate", routing._lane_by_id("prose"),
                               "rules-weak", 0.5)
    flywheel.handle_rate("fail --route-id " + rid2)
    outcome2 = [r for r in flywheel.read_labels()
                if r["kind"] == "outcome" and r.get("consumes_id") == rid2][0]
    assert outcome2["max_turns"] is None


# ------------------------------------------------------------- report


def test_report_labels_json(home):
    lanes = routing._load_routes()
    records = [
        {"kind": "route", "id": "r1", "ts": BASE_TS, "task": "port the shim to windows",
         "lane": "prose", "model": "m", "effort": "medium", "method": "pinned",
         "confidence": 1.0},
        {"kind": "route", "id": "r2", "ts": BASE_TS + 1, "task": "rewrite the docs intro",
         "lane": "prose", "model": "m", "effort": "medium", "method": "pinned",
         "confidence": 1.0},
        {"kind": "route", "id": "r3", "ts": BASE_TS + 2, "task": "x", "lane": "prose",
         "model": "m", "effort": "medium", "method": "pinned", "confidence": 1.0},
        {"kind": "route", "id": "r4", "ts": BASE_TS + 3, "task": "port the shim to windows",
         "lane": "prose", "model": "m", "effort": "medium", "method": "pinned",
         "confidence": 1.0},  # duplicate text, not counted
        {"kind": "lane_correction", "ts": BASE_TS + 4, "from_lane": "a", "to_lane": "prose"},
    ]
    data = report._ledger_data(records)
    labels = {l["lane"]: l for l in data["labels"]}
    assert len(labels) == len(lanes) and len(lanes) == 9
    assert labels["prose"]["real_rows"] == 2
    assert labels["prose"]["corrections_to"] == 1


def test_report_labels_section_pin_nudge(home):
    records = []
    data = report._ledger_data(records)
    section = report._labels_section(data)
    assert "pin 8 real tasks here" in section
    assert "evalroute route --lane" in section
