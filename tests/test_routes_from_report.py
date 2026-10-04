"""Tests for routes_from_report (measured routes.yaml generation)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

from evalroute import routes_from_report as rfr

HEADER = ["lane", "model", "effort", "n", "pend", "trunc", "err", "complete", "expected", "pass_", "lo",
          "hi", "cov", "passk", "try_usd", "succ_usd", "vmin", "allin", "known",
          "med_out", "p50s"]


def _row(lane, model, effort="medium", n=30, cov="0.90", allin="0.05"):
    return dict(zip(HEADER, [lane, model, effort, n, "0", "0", "0", "True", str(n), "0.8",
                            "0.7", "0.9", cov, "0.7", "0.02", "0.04", "0.1",
                            allin, "True", "1000", "12.0"]))


def _write_csv(tmp_path, rows):
    p = tmp_path / "report.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader()
        w.writerows(rows)
    return p


def test_lane_slug_aliases(tmp_path):
    p = _write_csv(tmp_path, [
        _row("math", "qwen/qwen3.8-max", effort="max"),
        _row("hard agentic coding", "z-ai/glm-5.3", effort="max"),
    ])
    out = tmp_path / "routes.gen.yaml"
    assert rfr.generate(p, out, None) == 0
    doc = yaml.safe_load(out.read_text(encoding="utf-8"))
    ids = [l["id"] for l in doc["lanes"]]
    assert "math-first-principles" in ids
    assert "hard-agentic-coding" in ids


def test_routing_rule_lowest_allin_within_cov(tmp_path):
    # flash is cheaper but 10pts lower coverage -> GLM 5.3 must win the lane.
    p = _write_csv(tmp_path, [
        _row("math", "z-ai/glm-5.3-flash", cov="0.40", allin="0.01"),
        _row("math", "qwen/qwen3.8-max", cov="0.90", allin="0.05"),
    ])
    out = tmp_path / "routes.gen.yaml"
    assert rfr.generate(p, out, None) == 0
    doc = yaml.safe_load(out.read_text(encoding="utf-8"))
    math = next(l for l in doc["lanes"] if l["id"] == "math-first-principles")
    assert math["model"] == "qwen/qwen3.8-max"
    assert math["provenance"].startswith("measured 10 tasks")


def test_incomplete_arm_cannot_win_even_with_cheaper_sample(tmp_path):
    partial = _row("math", "cheap/model", n=1, cov="1.0", allin="0.001")
    partial["complete"] = "False"
    p = _write_csv(tmp_path, [partial, _row("math", "full/model", n=3, cov="1.0", allin="0.1")])
    out = tmp_path / "routes.yaml"
    assert rfr.generate(p, out, None, k=1) == 0
    assert yaml.safe_load(out.read_text())["lanes"][0]["model"] == "full/model"


def test_legacy_report_requires_runs_for_completeness_check(tmp_path):
    old_cols = [c for c in HEADER if c not in ("complete", "expected")]
    p = tmp_path / "old.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=old_cols)
        w.writeheader()
        w.writerow({k: v for k, v in _row("math", "cheap/model").items() if k in old_cols})
    with pytest.raises(ValueError, match="completeness evidence"):
        rfr.generate(p, tmp_path / "out.yaml", None)
    runs = tmp_path / "runs.jsonl"
    runs.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="completeness evidence"):
        rfr.generate(p, tmp_path / "out.yaml", None, runs_path=runs)


def test_existing_classifier_fields_preserved(tmp_path):
    p = _write_csv(tmp_path, [_row("prose", "moonshotai/kimi-k3", effort="medium")])
    existing = tmp_path / "routes.yaml"
    existing.write_text(yaml.safe_dump({
        "version": 1,
        "lanes": [{
            "id": "prose", "label": "Prose",
            "keywords": ["blog post", "essay"],
            "match_hint": "human is the checker",
            "model": "moonshotai/kimi-k3", "effort": "medium",
            "escalation": "claude-opus-5-5",
            "provenance": "priors 2026-09-26",
            "notes": "untested prior",
        }],
    }, sort_keys=False), encoding="utf-8")
    out = tmp_path / "routes.gen.yaml"
    assert rfr.generate(p, out, existing) == 0
    doc = yaml.safe_load(out.read_text(encoding="utf-8"))
    prose = next(l for l in doc["lanes"] if l["id"] == "prose")
    assert prose["keywords"] == ["blog post", "essay"]
    assert prose["escalation"] == "claude-opus-5-5"
    assert prose["notes"] == "untested prior"
    assert prose["label"] == "Prose"
    assert prose["provenance"].startswith("measured")


def test_unmeasured_lanes_carried_over_verbatim(tmp_path):
    p = _write_csv(tmp_path, [_row("prose", "moonshotai/kimi-k3")])
    existing = tmp_path / "routes.yaml"
    existing.write_text(yaml.safe_dump({
        "version": 1,
        "lanes": [{
            "id": "orchestration", "label": "Orchestrator / subagents",
            "keywords": ["orchestrate"], "match_hint": "plans",
            "model": "claude-opus-5-5", "effort": "high",
            "escalation": None, "provenance": "priors 2026-09-26",
            "notes": "image 2 only",
        }],
    }, sort_keys=False), encoding="utf-8")
    out = tmp_path / "routes.gen.yaml"
    assert rfr.generate(p, out, existing) == 0
    doc = yaml.safe_load(out.read_text(encoding="utf-8"))
    orch = next(l for l in doc["lanes"] if l["id"] == "orchestration")
    assert orch["provenance"] == "priors 2026-09-26"  # untouched, not "measured"


def test_empty_report_refuses(tmp_path):
    p = tmp_path / "report.csv"
    p.write_text(",".join(HEADER) + "\n", encoding="utf-8")
    out = tmp_path / "routes.gen.yaml"
    assert rfr.generate(p, out, None) == 1


def test_generated_file_loads_in_tool(tmp_path, monkeypatch):
    """The generated YAML must satisfy the plugin's own loader/validator.

    Uses the documented flow: regenerate FROM an existing routes.yaml so
    classifier fields are preserved, then load and classify with it.
    """
    p = _write_csv(tmp_path, [
        _row("math", "qwen/qwen3.8-max", effort="max"),
        _row("web research", "moonshotai/kimi-k3", effort="medium"),
    ])
    existing = tmp_path / "routes.yaml"
    existing.write_text(yaml.safe_dump({
        "version": 1,
        "lanes": [
            {"id": "web-research", "label": "Web research",
             "keywords": ["web research", "citations"], "match_hint": "search",
             "model": "moonshotai/kimi-k3", "effort": "medium",
             "provenance": "priors 2026-09-26"},
            {"id": "math-first-principles", "label": "Math", "keywords": ["proof"],
             "model": "qwen/qwen3.8-max", "effort": "max",
             "provenance": "priors 2026-09-26"},
        ],
    }, sort_keys=False), encoding="utf-8")
    out = tmp_path / "routes.gen.yaml"
    assert rfr.generate(p, out, existing) == 0
    routing_mod = __import__("evalroute.routing", fromlist=["routing"])
    monkeypatch.setattr(routing_mod, "ROUTES_FILE", out)
    monkeypatch.setattr(routing_mod, "_LLM_LANES", None)  # reset the lazy cache
    lanes = routing_mod._load_routes()
    assert {l["id"] for l in lanes} == {"math-first-principles", "web-research"}
    lane, conf, hits = routing_mod.classify("web research on competitors with citations")
    assert lane["id"] == "web-research"
    assert lane["provenance"].startswith("measured")

def test_generator_maps_arm_names_and_counts_three_samples_per_task(tmp_path):
    p = _write_csv(tmp_path, [_row("routine coding", "flash@medium", n=30, cov="1", allin="0.01")])
    mp = tmp_path / "models.json"
    mp.write_text(json.dumps([{"name": "flash@medium", "model": "z-ai/glm-5.3-flash"}]))
    out = tmp_path / "generated.yaml"
    assert rfr.generate(p, out, None, models_path=mp) == 0
    lane = yaml.safe_load(out.read_text())["lanes"][0]
    assert lane["model"] == "z-ai/glm-5.3-flash"
    assert "measured 10 tasks" in lane["provenance"]

@pytest.mark.parametrize("n,k", [(10, 3), (0, 3), (10, 0)])
def test_generator_rejects_inconsistent_sample_count(tmp_path, n, k):
    p = _write_csv(tmp_path, [_row("math", "qwen/qwen3.8-max", n=n)])
    if n == 0 and k > 0:
        assert rfr.generate(p, tmp_path / "out.yaml", None, k=k) == 1
    else:
        with pytest.raises(ValueError, match="samples|sample count|positive"):
            rfr.generate(p, tmp_path / "out.yaml", None, k=k)

def test_generator_rejects_unmapped_arm_and_invalid_model_id(tmp_path):
    p = _write_csv(tmp_path, [_row("prose", "flash@medium", n=30)])
    mp = tmp_path / "models.json"
    mp.write_text(json.dumps([{"name": "other@medium", "model": "provider/model"}]))
    with pytest.raises(ValueError, match="no model id"):
        rfr.generate(p, tmp_path / "out.yaml", None, models_path=mp)
    mp.write_text(json.dumps([{"name": "flash@medium", "model": "not a model id"}]))
    with pytest.raises(ValueError, match="invalid model id"):
        rfr.generate(p, tmp_path / "out.yaml", None, models_path=mp)

def test_generator_requires_exact_k_samples_per_task_when_runs_supplied(tmp_path):
    p = _write_csv(tmp_path, [_row("prose", "provider/model", n=3)])
    runs = tmp_path / "runs.jsonl"
    # Three graded rows, but spread across two tasks rather than 3 on one.
    runs.write_text("".join(json.dumps({"task": t, "lane": "prose", "model": "provider/model",
                                        "sample": s, "cfg": "c", "text": "x", "passed": True}) + "\n"
                            for t, s in [("a", 0), ("a", 1), ("b", 0)]))
    with pytest.raises(ValueError, match="exactly --k per task"):
        rfr.generate(p, tmp_path / "out.yaml", None, runs_path=runs)


def test_cfg_suffixed_winner_maps_base_arm_and_checks_recorded_model_id(tmp_path):
    p = _write_csv(tmp_path, [_row("prose", "flash@medium#abcdef", n=3)])
    mp = tmp_path / "models.json"
    mp.write_text(json.dumps([{"name": "flash@medium", "model": "z-ai/glm-5.3-flash"}]))
    runs = tmp_path / "runs.jsonl"
    def write_runs(model_id):
        runs.write_text("".join(json.dumps({
            "task": "t", "lane": "prose", "model": "flash@medium", "sample": i,
            "cfg": "abcdef1234567890", "model_id": model_id, "text": "ok", "passed": True,
        }) + "\n" for i in range(3)))
    write_runs("z-ai/glm-5.3-flash")
    out = tmp_path / "route.yaml"
    assert rfr.generate(p, out, None, models_path=mp, runs_path=runs, k=3) == 0
    assert yaml.safe_load(out.read_text())["lanes"][0]["model"] == "z-ai/glm-5.3-flash"
    write_runs("another/model")
    with pytest.raises(ValueError, match="disagrees"):
        rfr.generate(p, out, None, models_path=mp, runs_path=runs, k=3)
