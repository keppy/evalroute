"""Behavior tests: classifier, route cards, tool handler, install-routes merge."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from evalroute import routing

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ table

def test_routes_yaml_loads_and_has_unique_lanes():
    lanes = routing._load_routes()
    ids = [l["id"] for l in lanes]
    assert len(ids) == len(set(ids))
    assert len(lanes) >= 8
    for lane in lanes:
        assert lane["model"] and lane["effort"] and lane["label"]


def test_provenance_not_truncated_by_yaml_comment():
    # An unquoted provenance containing '#' gets truncated at the hash (parsed
    # as a comment). Regressed live in the prose row; keep every row whole.
    raw = yaml.safe_load((ROOT / "evalroute" / "data" / "routes.yaml").read_text(encoding="utf-8"))
    for lane in raw["lanes"]:
        prov = lane.get("provenance", "")
        assert prov and not prov.rstrip().endswith("creative writing"), lane["id"]
        if "#" in prov:
            assert prov.count("#") >= 1 and not prov.endswith("writing"), lane["id"]



# -------------------------------------------------------------- classifier

@pytest.mark.parametrize("task,expected", [
    ("fix the NaN loss in our GRPO run, the loss curve spikes at step 2k", "dl-ml-research-engineering"),
    ("prove the number of labeled trees on 7 vertices; full derivation", "math-first-principles"),
    ("write a blog post about our launch", "prose"),
    ("refactor the auth middleware and add unit tests", "routine-coding"),
    ("migrate the monolith to the new schema, it's a multi-file repo-wide change", "hard-agentic-coding"),
    ("web research: who are the competitors in the eval tooling space, with citations", "web-research"),
    ("read this 80-page PDF spec and summarize the safety requirements", "long-doc-reading"),
    ("critique the paper's claim about reward hacking and Goodhart effects", "alignment-reasoning"),
    ("orchestrate three parallel agents to review the PR", "orchestration"),
])
def test_classify_lanes(task, expected):
    lane, conf, hits = routing.classify(task)
    assert lane["id"] == expected, f"got {lane['id']} (hits={hits})"
    assert conf > 0
    assert hits


def test_classify_no_hit_defaults_to_long_doc():
    lane, conf, hits = routing.classify("hello world")
    assert lane["id"] == "long-doc-reading"
    assert conf == 0.0 and hits == []


def test_negation_guard_blocks_negated_keyword():
    # The exact misroute from live use: "not usually hard math though" must
    # not count as a math hit.
    lane, conf, hits = routing.classify(
        "an agent that helps me manage life, writing, and researchy tasks. "
        "not usually hard math though.")
    assert "math" not in [h for h in hits]
    assert lane["id"] != "math-first-principles" or conf == 0.0


def test_negation_guard_keeps_positive_hits():
    lane, conf, hits = routing.classify("prove the theorem, give a derivation and a lemma")
    assert lane["id"] == "math-first-principles"

def test_negated_first_occurrence_does_not_hide_later_positive_hit():
    assert routing._hit("math", routing._norm("not math; now solve the math problem"))
    assert not routing._hit("math", routing._norm("not math, and definitely no math"))


# --------------------------------------------------- route_for + LLM fallback

class FakeLlm:
    """Stands in for ctx.llm; returns a fixed lane or raises."""

    def __init__(self, lane_id="orchestration", confidence=0.9):
        self.lane_id = lane_id
        self.confidence = confidence
        self.calls = 0

    def complete_structured(self, **kwargs):
        self.calls += 1
        if self.lane_id == "raise":
            raise RuntimeError("no trust")
        import types
        return types.SimpleNamespace(
            parsed={"lane": self.lane_id, "confidence": self.confidence})


def test_route_for_strong_rules_skip_llm():
    lane, conf, hits, method = routing.route_for(
        "fix the NaN loss in our GRPO run, loss curve spikes at 2k steps")
    assert method == "rules-strong"
    assert lane["id"] == "dl-ml-research-engineering"


def test_route_for_llm_fallback_on_paraphrase():
    # Zero keyword signal ("researchy" is not a keyword); LLM decides.
    fake = FakeLlm("orchestration", 0.9)
    monkey_llm = routing.set_llm_facade(fake)
    try:
        lane, conf, hits, method = routing.route_for(
            "an agent that helps me manage life, writing, and researchy tasks")
        assert method == "llm"
        assert lane["id"] == "orchestration"
        assert fake.calls == 1
    finally:
        routing.set_llm_facade(monkey_llm)


def test_route_for_llm_fallback_on_single_ambiguous_hit():
    # One hit ("proof" of concept) only -> weak -> LLM takes over.
    fake = FakeLlm("routine-coding", 0.8)
    monkey_llm = routing.set_llm_facade(fake)
    try:
        lane, conf, hits, method = routing.route_for("a proof of concept for the dashboard")
        assert method == "llm"
        assert lane["id"] == "routine-coding"
    finally:
        routing.set_llm_facade(monkey_llm)


def test_route_for_llm_failure_falls_back_to_weak_rules():
    # Single weak hit, LLM facade raises -> rules-weak result stands.
    fake = FakeLlm("raise")
    monkey_llm = routing.set_llm_facade(fake)
    try:
        lane, conf, hits, method = routing.route_for("a proof of concept for the dashboard")
        assert method == "rules-weak"
        assert lane["id"] == "math-first-principles"  # 'proof' hit, weak but standing
    finally:
        routing.set_llm_facade(monkey_llm)


def test_route_for_no_llm_configured_uses_weak_rules():
    assert routing._PLUGIN_LLM is None  # not stashed outside register()
    lane, conf, hits, method = routing.route_for("a proof of concept for the dashboard")
    assert method in ("rules-weak", "default")


def test_route_for_llm_unknown_lane_rejected():
    # LLM returns a lane id not in the table -> treat as failure -> rules stand.
    fake = FakeLlm("nonexistent-lane", 0.9)
    monkey_llm = routing.set_llm_facade(fake)
    try:
        lane, conf, hits, method = routing.route_for("a proof of concept for the dashboard")
        assert method == "rules-weak"
    finally:
        routing.set_llm_facade(monkey_llm)


def test_classify_math_beats_prose_on_dual_signal():
    # "proof" (math) co-occurs with "blog post" (prose); math wins on stronger signal.
    lane, _, _ = routing.classify("write a blog post with a full proof of the theorem")
    assert lane["id"] == "math-first-principles"


# --------------------------------------------------------------- tool api

def test_tool_returns_json_with_card():
    out = json.loads(routing.evalroute_route({"task": "prove the Riemann hypothesis"}))
    assert out["lane"] == "math-first-principles"
    assert out["model"] == "qwen/qwen3.8-max"
    assert "route:" in out["card"] and "/model" in out["card"]
    assert out["provenance"].startswith("priors")


def test_tool_pinned_lane_skips_classification():
    out = json.loads(routing.evalroute_route({"task": "whatever", "lane": "prose"}))
    assert out["lane"] == "prose" and out["pinned"] is True

def test_pinned_tool_route_is_logged_and_consumed(isolated_hermes_home):
    from evalroute import flywheel
    out = json.loads(routing.evalroute_route({"task": "reclassify", "lane": "prose"}))
    assert out["lane"] == "prose"
    assert flywheel.handle_rate("pass").startswith("logged: pass")
    route, outcome = flywheel.read_labels()[:2]
    assert route["method"] == "pinned" and outcome["route_lane"] == "prose"
    assert outcome["consumes_id"] == route["id"]


def test_tool_unknown_lane_is_actionable_error():
    out = json.loads(routing.evalroute_route({"task": "x", "lane": "nope"}))
    assert "known lanes" in out["error"]


def test_tool_empty_task_is_error():
    out = json.loads(routing.evalroute_route({"task": " "}))
    assert "error" in out


def test_tool_never_raises_on_missing_args():
    out = json.loads(routing.evalroute_route({}))
    assert "error" in out


# ------------------------------------------------------------ slash/CLI path

def test_route_command_happy_path():
    card = routing.handle_route_command("fix the NaN loss in our GRPO run")
    assert "lane: DL / ML research engineering" in card
    assert "/model z-ai/glm-5.3" in card


def test_route_command_lane_pinned():
    card = routing.handle_route_command("--lane prose draft something")
    assert "classification: lane pinned by caller" in card


def test_route_command_usage_on_empty():
    out = routing.handle_route_command("")
    assert out.startswith("usage: /route")


def test_route_command_error_is_string():
    out = routing.handle_route_command("--lane nope x")
    assert out.startswith("evalroute:")


# ---------------------------------------------------------- install-routes

def test_effort_extraction_prefers_higher_end_of_range():
    # "max / high" style entries and one model serving two lanes keep the higher.
    lane = {"model": "x", "effort": "Opus high, subagents low-medium"}
    assert routing._effort_for_override(lane) == "high"
    assert routing._effort_for_override({"model": "x", "effort": "low-medium"}) == "medium"
    assert routing._effort_for_override({"model": "x", "effort": "high-max"}) == "max"


def _fake_config_module(monkeypatch, fake_config, written=None, fail_on_write=False):
    """Point routing._config_module at a stub instead of importing hermes_cli."""
    class _Stub:
        @staticmethod
        def load_config():
            return fake_config

        @staticmethod
        def set_config_value(key, value):
            if fail_on_write:
                raise AssertionError("must not write")
            if written is not None:
                written[key] = value

    monkeypatch.setattr(routing, "_config_module", lambda: (_Stub.load_config, _Stub.set_config_value))


def test_install_routes_merge_logic(monkeypatch, capsys):
    """Merge keeps foreign keys, route keys win, higher effort wins per model."""
    fake_config = {"agent": {"reasoning_overrides": {
        "z-ai/glm-5.3": "medium",          # table wants max (serves hard-agentic max + dl-ml high)
        "someone/other-model": "low",      # not in route table -> kept
    }}}
    written = {}
    _fake_config_module(monkeypatch, fake_config, written)
    rc = routing.install_routes(dry_run=False)
    assert rc == 0
    assert "agent.reasoning_overrides" in written
    merged = json.loads(written["agent.reasoning_overrides"])
    assert merged["z-ai/glm-5.3"] == "max"            # higher of the two lanes it serves
    assert merged["someone/other-model"] == "low"      # foreign key kept
    assert merged["moonshotai/kimi-k3"] == "medium"
    assert merged["qwen/qwen3.8-max"] == "max"
    out = capsys.readouterr().out
    assert "~ z-ai/glm-5.3: medium -> max" in out
    assert "= someone/other-model: low (kept" in out


def test_install_routes_dry_run_writes_nothing(monkeypatch, capsys):
    fake_config = {"agent": {"reasoning_overrides": {}}}
    _fake_config_module(monkeypatch, fake_config, fail_on_write=True)
    rc = routing.install_routes(dry_run=True)
    assert rc == 0
    assert "dry run: nothing written" in capsys.readouterr().out


def test_install_routes_noop_when_matching(monkeypatch, capsys):
    desired = {}
    for lane in routing._load_routes():
        e = routing._effort_for_override(lane)
        cur = desired.get(lane["model"])
        if cur is None or routing._EFFORT_ORDER.index(e) > routing._EFFORT_ORDER.index(cur):
            desired[lane["model"]] = e
    fake_config = {"agent": {"reasoning_overrides": desired}}
    _fake_config_module(monkeypatch, fake_config, fail_on_write=True)
    rc = routing.install_routes(dry_run=False)
    assert rc == 0
    assert "nothing to do" in capsys.readouterr().out
