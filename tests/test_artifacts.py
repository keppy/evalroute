"""Public claims derived from vendored artifacts; no provider or judge calls."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from evalroute import routes_from_report as rfr
from evalroute import adjudicate

ROOT = Path(__file__).resolve().parents[1]

from evalroute.harness import tier_a as h


def test_vendored_legacy_reports_and_cost_claim(capsys):
    artifacts = ROOT / "examples/artifacts"
    totals = []
    expected = {
        "tier-a-routine-coding": (120, 120),
        "tier-a-dl-ml": (150, 150),
        "tier-a-alignment": (149, 119),
    }
    for name, (n_rows, n_graded) in expected.items():
        runs = artifacts / name / "runs.jsonl"
        records = [r for r in h.jread(runs) if "text" in r]
        assert len(records) == n_rows
        assert sum(r.get("passed") is not None for r in records) == n_graded
        assert all("sig" not in r for r in records)  # old schema stays reportable
        totals.append(sum((r.get("cost") or 0) + (r.get("jcost") or 0) for r in records))
        h.report(SimpleNamespace(out=str(runs), tasks=str(artifacts / name / "tasks.jsonl"),
                                 k=3, usd_per_hour=100, tol=0, csv=None))
    assert sum(totals) == pytest.approx(3.0590051)
    assert "$3.0590051" in (ROOT / "README.md").read_text(encoding="utf-8")
    output = capsys.readouterr().out
    assert "route -> glm-5.3@medium" in output
    assert "qwen3.8-max@medium" in output
    assert "0   30" in output  # pending, not treated as graded


# the provenance-stamp comparison requires gonogo (McNemar); without it the
# generator degrades to "gap unverified" (same guard as tests/test_adjudicate.py)
@pytest.mark.skipif(adjudicate._gonogo() is None, reason="gonogo not installed")
def test_shipped_route_cards_match_fresh_vendored_adjudication(tmp_path):
    artifacts = ROOT / "examples/artifacts"
    stored = {r["id"]: r for r in yaml.safe_load((ROOT / "evalroute" / "data" / "routes.yaml").read_text())["lanes"]}
    for name, lane_id, models_file in (
        ("tier-a-routine-coding", "routine-coding", "tier-a-rc-models.json"),
        ("tier-a-dl-ml", "dl-ml-research-engineering", "tier-a-models.json"),
        ("tier-a-alignment", "alignment-reasoning", "tier-a-models.json"),
    ):
        d = artifacts / name
        csv_path, out_path = tmp_path / (name + ".csv"), tmp_path / (name + ".yaml")
        h.report(SimpleNamespace(out=str(d / "runs.jsonl"), tasks=str(d / "tasks.jsonl"),
                                 k=3, usd_per_hour=100, tol=0, csv=str(csv_path)))
        assert rfr.generate(csv_path, out_path, ROOT / "data/routes.yaml",
                            models_path=artifacts / models_file, k=3,
                            runs_path=d / "runs.jsonl") == 0
        generated = next(x for x in yaml.safe_load(out_path.read_text())["lanes"]
                         if x["id"] == lane_id)
        assert (stored[lane_id]["model"], stored[lane_id]["effort"]) == (
            generated["model"], generated["effort"])
        assert stored[lane_id]["provenance"].split("; gap vs runner-up: ")[1] == (
            generated["provenance"].split("; gap vs runner-up: ")[1])
        assert "[-33.4%, +33.4%]" in generated["provenance"]
