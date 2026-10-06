"""The opt-in explore fraction and runner-up arm pick."""

import json
import re

import pytest

import evalroute.routing as routing
from evalroute import explore
from evalroute import flywheel
from evalroute import report
from evalroute import routes_from_labels as rfl


from tests.test_contribute import home  # noqa: F401  (pytest fixture import)


@pytest.fixture(autouse=True)
def _no_explore_env(monkeypatch):
    monkeypatch.delenv("EVALROUTE_EXPLORE", raising=False)


def _make_cfg(hermes_home, yaml_fraction=None, json_fraction=None):
    if yaml_fraction is not None:
        (hermes_home / "config.yaml").write_text(
            f"evalroute:\n  explore:\n    fraction: {yaml_fraction}\n", encoding="utf-8")
    if json_fraction is not None:
        (hermes_home / "evalroute").mkdir(parents=True, exist_ok=True)
        (hermes_home / "evalroute" / "config.json").write_text(
            json.dumps({"explore": {"fraction": json_fraction}}), encoding="utf-8")


def test_env_beats_config_yaml(tmp_path, monkeypatch):
    _make_cfg(tmp_path, yaml_fraction=0.9)
    monkeypatch.setenv("EVALROUTE_EXPLORE", "0.2")
    assert explore.explore_fraction() == 0.2


def test_yaml_present_json_ignored(tmp_path):
    _make_cfg(tmp_path, yaml_fraction=0.9, json_fraction=0.1)
    assert explore.explore_fraction() == 0.9


def test_json_used_when_no_yaml(tmp_path):
    _make_cfg(tmp_path, json_fraction=0.3)
    assert explore.explore_fraction() == 0.3


def test_garbage_is_zero(tmp_path, monkeypatch):
    monkeypatch.setenv("EVALROUTE_EXPLORE", "garbage")
    assert explore.explore_fraction() == 0.0


def test_clamped_to_one(tmp_path, monkeypatch):
    monkeypatch.setenv("EVALROUTE_EXPLORE", "1.7")
    assert explore.explore_fraction() == 1.0


def test_zero_never_calls_rng(tmp_path, monkeypatch):
    monkeypatch.setenv("EVALROUTE_EXPLORE", "0")
    monkeypatch.setattr(explore, "_rng", lambda: (_ for _ in ()).throw(AssertionError("rng called")))
    lane = dict(next(l for l in routing._load_routes() if l["id"] == "routine-coding"))
    assert explore.pick(lane) is lane


def _routine_lane():
    return dict(next(l for l in routing._load_routes() if l["id"] == "routine-coding"))


def test_pick_explores(tmp_path, monkeypatch):
    monkeypatch.setenv("EVALROUTE_EXPLORE", "0.5")
    monkeypatch.setattr(explore, "_rng", lambda: 0.0)
    cached = _routine_lane()
    out = explore.pick(cached)
    assert out["model"] == "deepseek/deepseek-v4.1-flash"
    assert out["effort"] == "medium"
    assert out["explore_default"] == "z-ai/glm-5.3-flash@medium"
    assert out is not cached
    assert routing._load_routes()[[l["id"] for l in routing._load_routes()].index("routine-coding")] == cached


def test_pick_no_entry(tmp_path, monkeypatch):
    monkeypatch.setenv("EVALROUTE_EXPLORE", "1")
    lane = dict(next(l for l in routing._load_routes() if l["id"] == "prose"))
    assert explore.pick(lane) is lane


def test_pick_no_explore_flag(tmp_path, monkeypatch):
    monkeypatch.setenv("EVALROUTE_EXPLORE", "1")
    lane = _routine_lane()
    assert explore.pick(lane, no_explore=True) is lane


def test_end_to_end_explore(tmp_path, monkeypatch):
    monkeypatch.setenv("EVALROUTE_EXPLORE", "1")
    monkeypatch.setattr(explore, "_rng", lambda: 0.0)
    card, lane, conf, pinned, method, route_id = routing._route_for_args(
        "--lane routine-coding fix the parser test")
    assert lane["model"] == "deepseek/deepseek-v4.1-flash"
    assert "explore: runner-up arm (100% of routes)" in card
    assert "z-ai/glm-5.3-flash@medium," in card
    assert "rate it like any other" in card


def test_end_to_end_no_explore(tmp_path, monkeypatch):
    monkeypatch.setenv("EVALROUTE_EXPLORE", "1")
    monkeypatch.setattr(explore, "_rng", lambda: 0.0)
    card, lane, *_ = routing._route_for_args("--lane routine-coding fix the parser test",
                                             no_explore=True)
    assert lane["model"] == "z-ai/glm-5.3-flash"
    assert "explore:" not in card
    rows = [r for r in flywheel.read_labels() if r.get("kind") == "route"]
    assert "explore" not in rows[-1]


def test_non_explore_card_byte_identical(tmp_path, monkeypatch):
    monkeypatch.setattr(explore, "_rng", lambda: 0.0)
    card_off = routing._route_for_args("--lane routine-coding fix the parser test")[0]
    monkeypatch.setenv("EVALROUTE_EXPLORE", "1")
    monkeypatch.setattr(explore, "_rng", lambda: 1.0)  # >= fraction -> no explore
    card_on = routing._route_for_args("--lane routine-coding fix the parser test")[0]
    # Route IDs are random UUIDs; everything else must be byte-identical.
    assert re.sub(r"[0-9a-f]{32}", "<id>", card_off) == re.sub(r"[0-9a-f]{32}", "<id>", card_on)


# ------------------------------------------------------------------ brief 02


def test_dispatch_dry_run_override_forces_no_explore(home, tmp_path, monkeypatch, capsys):
    from tests.test_dispatch import _dry_run_dispatch
    monkeypatch.setenv("EVALROUTE_EXPLORE", "1")
    monkeypatch.setattr(explore, "_rng", lambda: 0.0)
    rc, out, sidecar = _dry_run_dispatch(home, tmp_path, monkeypatch, capsys,
                                                  lane="routine-coding",
                                                  model_override="z-ai/glm-5.3-flash",
                                                  effort_override="medium")
    assert rc == 0
    assert sidecar["model"] == "z-ai/glm-5.3-flash"
    rows = [r for r in flywheel.read_labels() if r.get("kind") == "route"]
    assert "explore" not in rows[-1]


def test_redact_outcome_rows_carry_explore_fields(home):
    import evalroute.contribute as contribute
    task = "fix the NaN loss in our GRPO run"
    records = [
        {"kind": "route", "id": "r1", "ts": 1000.0, "task": task,
         "lane": "routine-coding", "model": "deepseek/deepseek-v4.1-flash",
         "effort": "medium", "method": "rules-strong", "confidence": 0.9,
         "explore": True, "explore_default": "z-ai/glm-5.3-flash@medium"},
        {"kind": "route", "id": "r2", "ts": 1010.0, "task": "write a blog post",
         "lane": "routine-coding", "model": "z-ai/glm-5.3-flash", "effort": "medium",
         "method": "rules-weak", "confidence": 0.4},
        {"kind": "outcome", "ts": 1020.0, "rated": "pass",
         "route_lane": "routine-coding", "route_model": "deepseek/deepseek-v4.1-flash",
         "route_effort": "medium", "actual_model": None, "actual_effort": None,
         "arm_attribution": "unknown", "method": "rules-strong", "confidence": 0.9,
         "consumes": 1000.0, "consumes_id": "r1"},
        {"kind": "outcome", "ts": 1030.0, "rated": "fail",
         "route_lane": "routine-coding", "route_model": "z-ai/glm-5.3-flash",
         "route_effort": "medium", "actual_model": None, "actual_effort": None,
         "arm_attribution": "unknown", "method": "rules-weak", "confidence": 0.4,
         "consumes": 1010.0, "consumes_id": "r2"},
    ]
    rows = [r for r in contribute.redact(records, salt=b"\x01" * 32)
            if r["kind"] == "outcome"]
    assert len(rows) == 2
    by_rated = {r["rated"]: r for r in rows}
    assert by_rated["pass"]["explore"] is True
    assert by_rated["pass"]["explore_default"] == "z-ai/glm-5.3-flash@medium"
    assert by_rated["fail"]["explore"] is False
    assert by_rated["fail"]["explore_default"] is None
    for row in rows:
        assert set(row) <= set(contribute.KEPT), set(row) - set(contribute.KEPT)
        serialised = json.dumps(row)
        assert task not in serialised and "blog post" not in serialised


# ------------------------------------------------------------------ brief 03


def _outcome(model, effort, rated, who="a"):
    return {"kind": "outcome", "route_lane": "routine-coding",
            "route_model": model, "route_effort": effort,
            "actual_model": model, "actual_effort": effort,
            "arm_attribution": "explicit_user", "method": "rules-strong",
            "confidence": 1.0, "rated": rated, "week": "2026-W40",
            "task_hash": "0" * 64, "corrected": False, "schema": 1,
            "_contributor": who}


def test_merge_contributed_counts_explore_arm(home, tmp_path):
    """A contributed outcome on the routine-coding runner-up arm is not an
    unknown_arm: explore arms are maintainer-chosen, so they count as known."""
    routes = tmp_path / "routes.yaml"
    routes.write_text(
        "lanes:\n"
        "- id: routine-coding\n"
        "  model: z-ai/glm-5.3-flash\n"
        "  effort: medium\n"
        "  provenance: priors\n"
        "  keywords: [generic]\n", encoding="utf-8")
    runner_up = explore_arms_model("routine-coding")
    rows = [_outcome(runner_up, "medium", "pass")]
    # via the same known_models the merge builds: not an unknown_arm
    stats = rfl.aggregate_contributed(rows, known_models=_known_models(routes))
    assert stats["dropped"]["unknown_arm"] == 0
    assert stats["lanes"]["routine-coding"]["arms"][
        f"{runner_up}@medium"]["attempts"] == 1
    # a made-up model is still dropped
    made_up = [_outcome("evil/made-up-model", "max", "pass")]
    stats2 = rfl.aggregate_contributed(made_up, known_models=_known_models(routes))
    assert stats2["dropped"]["unknown_arm"] == 1
    lanes, _applied = rfl.merge_contributed(routes, rows)
    assert lanes  # sanity
    lanes2, _ = rfl.merge_contributed(routes, made_up)
    original = next(l for l in
                    __import__("yaml").safe_load(routes.read_text(encoding="utf-8"))["lanes"]
                    if l["id"] == "routine-coding")
    assert {l["id"]: l for l in lanes2}["routine-coding"] == original


def explore_arms_model(lane):
    from evalroute.explore import explore_arms
    return explore_arms()[lane]["model"]


def _known_models(routes_path):
    import yaml as _yaml
    from evalroute.explore import explore_arms
    raw = _yaml.safe_load(routes_path.read_text(encoding="utf-8")) or {}
    known = {l.get("model") for l in raw.get("lanes", []) if l.get("model")}
    known |= {a.get("model") for a in explore_arms().values() if a.get("model")}
    return known


def test_report_no_explore_in_output(home):
    records = [
        {"kind": "route", "id": "r1", "lane": "routine-coding",
         "model": "z-ai/glm-5.3-flash", "effort": "medium"},
    ]
    data = report._ledger_data(records)
    assert "explore" not in data
    out = report._data_model(None, None)
    assert "explore" not in out
    html = report._render(None, None, "evalroute report")
    assert "id='explore'" not in html
    assert '"explore"' not in html
