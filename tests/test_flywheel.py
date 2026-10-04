"""Tests for the flywheel: route/outcome labeling and observed-provenance aggregation.

All tests point HERMES_HOME at a tmp dir so labels never touch the real
<home>/evalroute/labels.jsonl.
"""

from __future__ import annotations

import json

import pytest

from evalroute import flywheel
from evalroute import routes_from_labels as rfl
from evalroute import routing
from pathlib import Path

def test_public_tree_has_no_raw_labels_snapshot():
    root = Path(__file__).resolve().parents[1]
    assert not (root / "data/flywheel/labels.jsonl").exists()
    assert "data/flywheel/labels.jsonl" in (root / ".gitignore").read_text()


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    flywheel._MEMORY.clear()
    yield tmp_path
    flywheel._MEMORY.clear()


def _labels_file(home):
    return home / "evalroute" / "labels.jsonl"


def test_route_label_written(home):
    lane, conf, hits, method = routing.route_for(
        "fix the NaN loss in our GRPO run, loss curve spikes at 2k steps")
    flywheel.note_route("fix the NaN loss", lane, method, conf)
    rec = json.loads(_labels_file(home).read_text().strip())
    assert rec["kind"] == "route"
    assert rec["lane"] == "dl-ml-research-engineering"
    assert rec["model"] == "z-ai/glm-5.3"
    assert rec["method"] == "rules-strong"
    assert rec["ts"] > 0


def test_rate_requires_route(home):
    out = flywheel.handle_rate("pass")
    assert "no route on record" in out


def test_rate_does_not_attribute_unmatched_turn_to_route(home):
    lane, conf, _, method = routing.route_for("write a blog post about our launch")
    flywheel.note_route("blog post", lane, method, conf)
    flywheel.on_post_llm_call(session_id="s1", model="moonshotai/kimi-k3")
    out = flywheel.handle_rate("pass")
    assert out.startswith("logged: pass")
    rec = [json.loads(l) for l in _labels_file(home).read_text().splitlines()][-1]
    assert rec["kind"] == "outcome" and rec["rated"] == "pass"
    assert rec["actual_model"] is None
    assert rec["arm_attribution"] == "unknown"
    assert rec["route_lane"] == "prose"


def test_rate_consumes_route(home):
    lane, conf, _, method = routing.route_for("write a blog post")
    flywheel.note_route("blog", lane, method, conf)
    flywheel.handle_rate("pass")
    out = flywheel.handle_rate("pass")
    assert "no route on record" in out  # consumed; must route again first

def test_interleaved_turn_does_not_override_explicit_route_id(home):
    prose = routing._lane_by_id("prose")
    math = routing._lane_by_id("math-first-principles")
    first = flywheel.note_route("draft", prose, "pinned", 1)
    second = flywheel.note_route("proof", math, "pinned", 1)
    flywheel.on_post_llm_call(session_id="other-session", model="qwen/qwen3.8-max")
    assert first != second
    assert "route id: " + first in flywheel.handle_rate("pass --route-id " + first)
    outcomes = [r for r in flywheel.read_labels() if r["kind"] == "outcome"]
    assert outcomes[0]["consumes_id"] == first
    assert outcomes[0]["route_lane"] == "prose"
    assert outcomes[0]["actual_model"] is None
    assert flywheel.pending_route_from_file()["id"] == second
    assert "already consumed/unknown" in flywheel.handle_rate("fail --route-id " + first)


def test_unverified_switch_never_becomes_arm_outcome_but_explicit_confirmation_does(home):
    route_id = flywheel.note_route("draft", routing._lane_by_id("prose"), "pinned", 1)
    flywheel.on_pre_command(command="model", args_raw="wrong/session-model", session_key="other")
    out = flywheel.handle_rate(
        f"pass --route-id {route_id} --model moonshotai/kimi-k3 --effort medium")
    assert "user-confirmed" in out
    outcome = next(r for r in reversed(flywheel.read_labels()) if r["kind"] == "outcome")
    assert outcome["actual_model"] == "moonshotai/kimi-k3"
    assert outcome["arm_attribution"] == "explicit_user"
    stats = rfl.aggregate(flywheel.read_labels())
    assert "wrong/session-model@medium" not in stats["lanes"]["prose"]["arms"]
    assert stats["lanes"]["prose"]["arms"]["moonshotai/kimi-k3@medium"]["passes"] == 1


def test_uuids_disambiguate_routes_with_identical_timestamps(home, monkeypatch):
    monkeypatch.setattr(flywheel.time, "time", lambda: 123.0)
    first = flywheel.note_route("a", routing._lane_by_id("prose"), "pinned", 1)
    second = flywheel.note_route("b", routing._lane_by_id("routine-coding"), "pinned", 1)
    flywheel.handle_rate("pass --route-id " + first)
    assert flywheel.pending_route_from_file()["id"] == second
    flywheel.handle_rate("fail --route-id " + second)
    assert flywheel.pending_route_from_file() is None

def test_pinned_reroute_consumes_previous_and_records_correction(home):
    original = flywheel.note_route("same task", routing._lane_by_id("prose"), "rules-weak", .5)
    # Equal text alone must not skip another session's route.
    first_card = routing.handle_route_command("--lane routine-coding same task")
    assert "route id:" in first_card
    assert [r["kind"] for r in flywheel.read_labels()] == ["route", "route"]
    card = routing.handle_route_command(
        f"--lane routine-coding --replace-route-id {original} same task")
    assert "route id:" in card
    labels = flywheel.read_labels()
    assert [r["kind"] for r in labels] == ["route", "route", "outcome", "lane_correction", "route"]
    assert labels[2]["consumes_id"] == original
    assert labels[3]["to_lane"] == "routine-coding"
    assert flywheel.pending_route_from_file()["id"] == labels[-1]["id"]


def test_model_switch_after_route_logs_rejection(home):
    lane, conf, _, method = routing.route_for("prove the theorem, derivation, lemma")
    flywheel.note_route("theorem", lane, method, conf)
    flywheel.on_pre_command(command="model", args_raw="gpt-6-sol", session_key="s1")
    rec = json.loads(_labels_file(home).read_text().strip().splitlines()[-1])
    assert rec["kind"] == "model_switch"
    assert rec["new_model"] == "gpt-6-sol"
    assert rec["candidate_rejection"] is True  # route said qwen/qwen3.8-max
    assert rec["route_association"] == "process_global_unverified"


def test_model_switch_matching_route_is_not_rejection(home):
    lane, conf, _, method = routing.route_for("prove the theorem, derivation, lemma")
    flywheel.note_route("theorem", lane, method, conf)
    flywheel.on_pre_command(command="model", args_raw="qwen/qwen3.8-max", session_key="s1")
    rec = json.loads(_labels_file(home).read_text().strip().splitlines()[-1])
    assert rec["candidate_rejection"] is False


def test_reasoning_display_query_not_logged(home):
    flywheel.on_pre_command(command="reasoning", args_raw="show", session_key="s1")
    assert not _labels_file(home).exists()


def test_rate_lane_correction_feeds_classifier_data(home):
    lane, conf, _, method = routing.route_for("a proof of concept for the dashboard")
    flywheel.note_route("poc", lane, method, conf)
    out = flywheel.handle_rate("fail --lane routine-coding")
    assert "lane corrected" in out
    recs = [json.loads(l) for l in _labels_file(home).read_text().splitlines()]
    assert any(r["kind"] == "lane_correction" and r["to_lane"] == "routine-coding"
               for r in recs)
    outcome = next(r for r in recs if r["kind"] == "outcome")
    assert outcome["original_lane"] != outcome["route_lane"] == "routine-coding"


def test_rate_rejects_unknown_lane(home):
    lane, conf, _, method = routing.route_for("blog post")
    flywheel.note_route("blog", lane, method, conf)
    out = flywheel.handle_rate("fail --lane nonsense")
    assert "unknown lane" in out


def test_rate_skip_consumes(home):
    lane, conf, _, method = routing.route_for("blog post")
    flywheel.note_route("blog", lane, method, conf)
    out = flywheel.handle_rate("skip")
    assert "skipped" in out
    assert "no route" in flywheel.handle_rate("pass")


# ---------------------------------------------------------- aggregation

import time as _time

_NOW = _time.time()


def _seed(home, records):
    flywheel.labels_path()  # the plugin's own helper mkdirs <home>/evalroute/
    with _labels_file(home).open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_aggregate_stats(home):
    _seed(home, [
        {"kind": "route", "lane": "prose", "model": "moonshotai/kimi-k3", "effort": "medium",
         "method": "rules-strong", "confidence": 1.0, "ts": _NOW - 100},
        {"kind": "outcome", "rated": "pass", "route_lane": "prose",
         "actual_model": "moonshotai/kimi-k3", "actual_effort": "medium",
         "arm_attribution": "explicit_user", "ts": _NOW - 90},
        {"kind": "outcome", "rated": "fail", "route_lane": "prose",
         "actual_model": "z-ai/glm-5.3", "actual_effort": "high", "ts": _NOW - 80},
        {"kind": "lane_correction", "from_lane": "math-first-principles",
         "to_lane": "orchestration", "task": "x", "ts": _NOW - 70},
    ])
    stats = rfl.aggregate(flywheel.read_labels())
    assert stats["routes"] == 1 and stats["outcomes"] == 2
    assert stats["lane_corrections"] == 1
    prose = stats["lanes"]["prose"]
    assert prose["outcomes"] == 2
    assert prose["pass_rate"] == 0.5
    assert prose["arms"]["moonshotai/kimi-k3@medium"]["passes"] == 1
    assert prose["arms"]["?@?"]["fails"] == 1
    assert "z-ai/glm-5.3@high" not in prose["arms"]


def test_aggregate_stale_cutoff(home):
    _seed(home, [
        {"kind": "outcome", "rated": "pass", "route_lane": "prose",
         "actual_model": "m", "actual_effort": "low", "ts": 1.0},  # epoch 1970: stale
    ])
    stats = rfl.aggregate(flywheel.read_labels())
    assert stats["outcomes"] == 0


def test_merge_observed_does_not_touch_measured(home, tmp_path):
    routes = tmp_path / "routes.yaml"
    routes.write_text(
        "version: 1\n"
        "lanes:\n"
        "  - id: prose\n"
        "    label: Prose\n"
        "    model: moonshotai/kimi-k3\n"
        "    effort: medium\n"
        "    provenance: measured 10 tasks, cov 0.9, all-in $0.04/succ, 2026-09-26\n"
        "    keywords: [blog post]\n", encoding="utf-8")
    _seed(home, [
        {"kind": "outcome", "rated": "fail", "route_lane": "prose",
         "actual_model": "moonshotai/kimi-k3", "actual_effort": "medium", "ts": _NOW - 90},
    ])
    lanes, applied = rfl.merge_observed(routes, flywheel.read_labels())
    assert applied == 0  # measured row untouched
    prose = next(l for l in lanes if l["id"] == "prose")
    assert prose["provenance"].startswith("measured")


def test_merge_observed_flips_only_after_repeated_both_arm_outcomes(home, tmp_path):
    routes = tmp_path / "routes.yaml"
    routes.write_text(
        "version: 1\n"
        "lanes:\n"
        "  - id: prose\n"
        "    label: Prose\n"
        "    model: moonshotai/kimi-k3\n"
        "    effort: medium\n"
        "    provenance: priors 2026-09-26\n"
        "    keywords: [blog post]\n", encoding="utf-8")
    _seed(home, [
        {"kind": "outcome", "rated": "fail", "route_lane": "prose",
         "actual_model": "moonshotai/kimi-k3", "actual_effort": "medium", "arm_attribution": "explicit_user", "ts": _NOW - 90},
        {"kind": "outcome", "rated": "fail", "route_lane": "prose",
         "actual_model": "moonshotai/kimi-k3", "actual_effort": "medium", "arm_attribution": "explicit_user", "ts": _NOW - 80},
        {"kind": "outcome", "rated": "fail", "route_lane": "prose",
         "actual_model": "moonshotai/kimi-k3", "actual_effort": "medium", "arm_attribution": "explicit_user", "ts": _NOW - 75},
        {"kind": "outcome", "rated": "pass", "route_lane": "prose",
         "actual_model": "claude-opus-5-5", "actual_effort": "high", "arm_attribution": "explicit_user", "ts": _NOW - 70},
        {"kind": "outcome", "rated": "pass", "route_lane": "prose",
         "actual_model": "claude-opus-5-5", "actual_effort": "high", "arm_attribution": "explicit_user", "ts": _NOW - 65},
    ])
    lanes, applied = rfl.merge_observed(routes, flywheel.read_labels())
    assert applied == 1
    prose = next(l for l in lanes if l["id"] == "prose")
    assert prose["provenance"].startswith("observed")
    assert prose["model"] == "claude-opus-5-5" and prose["effort"] == "high"
    assert "observed flip" in prose["notes"]
    assert "non-randomized" in prose["notes"]

def test_two_fails_one_pass_does_not_flip(home, tmp_path):
    routes = tmp_path / "routes.yaml"
    routes.write_text("lanes:\n- id: prose\n  model: m\n  effort: medium\n  provenance: priors\n")
    _seed(home, [
        {"kind": "outcome", "rated": "fail", "route_lane": "prose", "actual_model": "m", "actual_effort": "medium", "ts": _NOW - 50},
        {"kind": "outcome", "rated": "fail", "route_lane": "prose", "actual_model": "m", "actual_effort": "medium", "ts": _NOW - 40},
        {"kind": "outcome", "rated": "pass", "route_lane": "prose", "actual_model": "n", "actual_effort": "high", "ts": _NOW - 30},
    ])
    rows, _ = rfl.merge_observed(routes, flywheel.read_labels())
    assert rows[0]["model"] == "m"
    assert "same-maintainer" in rows[0]["provenance"]


def test_merge_observed_no_flip_on_single_fail(home, tmp_path):
    # One failure is not flip evidence; the route stands, provenance becomes observed.
    routes = tmp_path / "routes.yaml"
    routes.write_text(
        "version: 1\n"
        "lanes:\n"
        "  - id: prose\n"
        "    label: Prose\n"
        "    model: moonshotai/kimi-k3\n"
        "    effort: medium\n"
        "    provenance: priors 2026-09-26\n", encoding="utf-8")
    _seed(home, [
        {"kind": "outcome", "rated": "fail", "route_lane": "prose",
         "actual_model": "moonshotai/kimi-k3", "actual_effort": "medium", "ts": _NOW - 90},
    ])
    lanes, applied = rfl.merge_observed(routes, flywheel.read_labels())
    prose = next(l for l in lanes if l["id"] == "prose")
    assert prose["model"] == "moonshotai/kimi-k3"  # unchanged
    assert prose["provenance"].startswith("observed")
