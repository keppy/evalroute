"""The opt-in explore fraction and runner-up arm pick."""

import json
import re

import pytest

import evalroute.routing as routing
from evalroute import explore
from evalroute import flywheel


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


def test_pick_provenance_answered(tmp_path, monkeypatch):
    monkeypatch.setenv("EVALROUTE_EXPLORE", "1")
    lane = _routine_lane()
    lane["provenance"] = "measured 2026-09-27 p=0.010"
    assert explore.pick(lane) is lane


def test_end_to_end_explore(tmp_path, monkeypatch):
    monkeypatch.setenv("EVALROUTE_EXPLORE", "1")
    monkeypatch.setattr(explore, "_rng", lambda: 0.0)
    card, lane, conf, pinned, method, route_id = routing._route_for_args(
        "--lane routine-coding fix the parser test")
    assert lane["model"] == "deepseek/deepseek-v4.1-flash"
    assert "explore: runner-up arm (100% of routes) instead of z-ai/glm-5.3-flash@medium" in card
    assert "explore basis:" in card
    rows = [r for r in flywheel.read_labels() if r.get("kind") == "route"]
    assert rows[-1]["explore"] is True
    assert rows[-1]["model"] == "deepseek/deepseek-v4.1-flash"
    assert rows[-1]["explore_default"] == "z-ai/glm-5.3-flash@medium"


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
