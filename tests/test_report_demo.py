"""report --demo: bundled synthetic fixture; never reads or writes real data.

Every case runs with HERMES_HOME pointed at a fake home that holds a
real-looking ledger and a real-looking report.html — the demo run must leave
both byte- and mtime-identical.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from evalroute import report

REPO = Path(__file__).resolve().parents[1]
DEMO = REPO / "evalroute" / "data" / "demo"


@pytest.fixture
def demo_home(tmp_path, monkeypatch):
    """HERMES_HOME with a real-looking ledger + report.html (demo must not touch)."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    ledger = home / "evalroute" / "labels.jsonl"
    ledger.parent.mkdir(parents=True)
    ledger.write_text(json.dumps({"kind": "route", "ts": time.time(),
                                  "task": "real private task"}) + "\n",
                      encoding="utf-8")
    page = home / "evalroute" / "report.html"
    page.write_text("<!doctype html><html><body>real report</body></html>\n",
                    encoding="utf-8")
    yield home


def _run(demo_home, **kw):
    defaults = dict(out=None, json=False, open=False, watch=None,
                    trains=None, factory_json=None, demo=True)
    defaults.update(kw)
    args = SimpleNamespace(**defaults)
    rc = report.run(args)
    return rc, (demo_home / "evalroute" / "report-demo.html")


def _isolation_snapshot(demo_home):
    ledger = demo_home / "evalroute" / "labels.jsonl"
    page = demo_home / "evalroute" / "report.html"
    stat = lambda p: (p.read_bytes(), p.stat().st_mtime_ns)
    return stat(ledger), stat(page)


def _sections(demo_home=None):
    data = report._data_model(None, None, is_demo=True)
    return data


def test_demo_json_all_sections(demo_home):
    rc, _ = _run(demo_home, json=True)
    assert rc == 0
    data = _sections()
    for key in ("pending", "outcome_rows", "tally", "methods",
                "sessions", "trains"):
        assert key in data and data[key], key
    assert data["sessions"]["present"] is True
    assert any(0.34 == s["cost_usd"]
               for t in data["sessions"]["trains"]
               for s in t["sessions"])


def test_demo_json_stdout(demo_home, capsys):
    rc, _ = _run(demo_home, json=True)
    data = json.loads(capsys.readouterr().out)
    for key in ("pending", "outcome_rows", "tally", "methods",
                "sessions", "trains"):
        assert key in data and data[key], key
    assert data["demo"] is True


def test_demo_html_badge_and_self_contained(demo_home):
    rc, page = _run(demo_home)
    assert rc == 0
    text = page.read_text(encoding="utf-8")
    assert "DEMO DATA" in text
    assert "<script src=" not in text
    assert "<link href=" not in text
    assert "http://" not in text and "https://" not in text


def test_demo_isolation(demo_home):
    before = _isolation_snapshot(demo_home)
    rc, page = _run(demo_home)
    assert rc == 0
    assert page.exists()
    after = _isolation_snapshot(demo_home)
    assert before == after
    assert page.name == "report-demo.html"
    assert not (demo_home / "evalroute" / "report.html").exists() or \
        (demo_home / "evalroute" / "report.html").read_text(
            encoding="utf-8").startswith("<!doctype html><html><body>real")


def test_demo_offsets_within_last_10_days():
    data = report._data_model(None, None, is_demo=True)
    now = time.time()
    isos = [row["iso"] for row in data["pending"]]
    assert isos, "pending rows must exist"
    # re-resolve offsets directly and bound them
    recs = [json.loads(l) for l in (DEMO / "labels.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()]
    import evalroute.demo as demo_mod
    resolved = demo_mod.resolve_offsets(recs)
    for r in resolved:
        assert now - 10.5 * 86400 < r["ts"] <= now, r["iso"]
    # consumes matches its route's offset exactly
    by_id = {r["id"]: r["ts"] for r in resolved if r.get("id")}
    for r in resolved:
        if r.get("consumes_id") and r.get("consumes") is not None:
            assert r["consumes"] == by_id[r["consumes_id"]]


def test_demo_ledger_shape_counts():
    recs = [json.loads(l) for l in (DEMO / "labels.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()]
    routes = [r for r in recs if r["kind"] == "route"]
    outcomes = [r for r in recs if r["kind"] == "outcome"]
    corrections = [r for r in recs if r["kind"] == "rating_correction"]
    assert len(routes) == 40
    assert 30 <= len(outcomes) <= 34
    assert len(corrections) == 2
    lanes = {r["lane"] for r in routes}
    assert len(lanes) == 9


def test_demo_fixture_ships_in_wheel(tmp_path):
    """`uv build --wheel` puts data/demo/** inside the wheel."""
    import shutil
    import zipfile
    uv = shutil.which("uv")
    if not uv:
        pytest.skip("uv not on PATH")
    whl_dir = tmp_path / "wheelhouse"
    subprocess.run(
        [uv, "build", "--wheel", "-o", str(whl_dir)],
        cwd=str(REPO), check=True,
        env={**os.environ, "PYTHONPATH": ""})
    (whl, ) = whl_dir.glob("evalroute-*.whl")
    names = zipfile.ZipFile(whl).namelist()
    assert "evalroute/data/demo/labels.jsonl" in names
    assert "evalroute/data/demo/trains/2026-10-X/README.md" in names


def test_demo_never_touches_flywheel_module_state(demo_home, monkeypatch):
    """--demo must not call flywheel.read_labels/labels_path at all."""
    from evalroute import flywheel
    calls = []
    monkeypatch.setattr(flywheel, "read_labels",
                        lambda: calls.append(1) or pytest.fail("read real ledger"))
    rc, _ = _run(demo_home)
    assert rc == 0
    assert calls == []
