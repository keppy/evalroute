"""Tests for facet capture: multi-dimensional labels beyond the single lane."""

from __future__ import annotations

import json

import pytest

from evalroute import routing
from evalroute import flywheel
from evalroute import routes_from_labels as rfl

from tests.test_flywheel import home  # reuse the HERMES_HOME fixture


# ------------------------------------------------------------ facet helpers

def test_facets_for_hits_multi_lane():
    # "audit plan files about rl training" hits dl-ml (rl) and long-doc (plan/files)
    hits = ["dl-ml-research-engineering", "long-doc-reading"]
    fs = routing.facets_for_hits(hits)
    assert "domain-dlml" in fs and "long-doc" in fs


def test_facets_for_hits_empty():
    assert routing.facets_for_hits([]) == []


def test_normalize_facets_drops_unknown():
    out = routing.normalize_facets(["domain-dlml", "nonsense", "long-doc", "long-doc"])
    assert out == ["domain-dlml", "long-doc"]


def test_facet_dominance_rules():
    assert routing.facet_dominance([]) == ""
    assert routing.facet_dominance(["long-doc"]) == ""
    assert routing.facet_dominance(["long-doc", "domain-dlml"]) == "descriptive conjunction; lane chooses arm"
    assert routing.facet_dominance(["tier-hard", "domain-dlml"]) == "descriptive conjunction; lane chooses arm"


def test_route_full_returns_facets():
    lane, conf, hits, method, facets = routing.route_full(
        "read the PLAN.md and other docs, then audit my rl training setup and reasoning blocks")
    assert isinstance(facets, list)
    # the task claims both long-doc reading AND dl-ml domain
    assert "long-doc" in facets or "domain-dlml" in facets

def test_llm_lane_overrides_weak_rule_facets(monkeypatch):
    monkeypatch.setattr(routing, "_llm_fallback", lambda task, lanes:
                        (routing._lane_by_id("prose"), .9, []))
    lane, _, _, method, facets = routing.route_full("a proof of concept")
    assert method == "llm" and lane["id"] == "prose"
    assert facets == ["domain-prose"]


def test_route_card_shows_conjunction():
    lane = routing._lane_by_id("dl-ml-research-engineering")
    card = routing.route_card(lane, 0.8, ["rl"], method="llm",
                            facets=["long-doc", "domain-dlml"])
    assert "facets: long-doc + domain-dlml" in card
    assert "descriptive conjunction; lane chooses arm" in card


def test_route_card_single_facet():
    lane = routing._lane_by_id("prose")
    card = routing.route_card(lane, 0.9, ["blog post"], method="rules-strong",
                            facets=["domain-prose"])
    assert "facets: domain-prose" in card
    assert "conjunctive" not in card


# ------------------------------------------------------- flywheel capture

def test_note_route_logs_facets(home):
    lane = routing._lane_by_id("dl-ml-research-engineering")
    flywheel.note_route("audit plan files re rl training", lane, "llm", 0.8,
                        facets=["long-doc", "domain-dlml"])
    rec = json.loads((home / "evalroute" / "labels.jsonl").read_text().strip())
    assert rec["facets"] == ["long-doc", "domain-dlml"]


def test_note_route_without_facets(home):
    lane = routing._lane_by_id("prose")
    flywheel.note_route("blog post", lane, "rules-strong", 1.0)
    rec = json.loads((home / "evalrate.jsonl".replace("rate", "route") if False else
                      (home / "evalroute" / "labels.jsonl")).read_text().strip())
    assert "facets" not in rec


def test_rate_outcome_carries_facets(home):
    lane = routing._lane_by_id("dl-ml-research-engineering")
    flywheel.note_route("audit plan", lane, "llm", 0.8, facets=["long-doc", "domain-dlml"])
    flywheel.on_post_llm_call(session_id="s", model="z-ai/glm-5.3")
    flywheel.handle_rate("pass")
    recs = [json.loads(l) for l in (home / "evalroute" / "labels.jsonl").read_text().splitlines()]
    outcome = next(r for r in recs if r["kind"] == "outcome")
    assert outcome["facets"] == ["long-doc", "domain-dlml"]


# ------------------------------------------------------- aggregation

def test_facet_aggregation_pairs(home):
    import time as _time
    now = _time.time()
    p = home / "evalroute" / "labels.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        rows = [
            {"kind": "outcome", "rated": "pass", "route_lane": "dl-ml-research-engineering",
             "actual_model": "z-ai/glm-5.3", "actual_effort": "high",
             "facets": ["domain-dlml", "long-doc"], "ts": now - 100},
            {"kind": "outcome", "rated": "fail", "route_lane": "dl-ml-research-engineering",
             "actual_model": "z-ai/glm-5.3", "actual_effort": "high",
             "facets": ["domain-dlml", "long-doc"], "ts": now - 50},
        ]
        for r in rows:
            f.write(json.dumps(r) + "\n")
    stats = rfl.aggregate(flywheel.read_labels())
    assert stats["facets"]["domain-dlml"]["attempts"] == 2
    assert stats["facets"]["domain-dlml"]["passes"] == 1
    assert stats["facet_pairs"]["domain-dlml + long-doc"]["attempts"] == 2
    assert stats["facet_pairs"]["domain-dlml + long-doc"]["passes"] == 1
