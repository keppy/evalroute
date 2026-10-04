"""tests/test_dataset.py — sync, status, clear, table provenance. No network, ever.

huggingface_hub is faked in sys.modules: snapshot_download copies from a
fixture "remote" dir, dataset_info returns a stub with .sha.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import types
from pathlib import Path

import pytest
import yaml

from evalroute import dataset
from evalroute import routing
from evalroute import cli
from evalroute import __version__

ROOT = Path(__file__).resolve().parents[1]

# The card snapshots (MATH_041 / PROSE_041) were captured on the chat surface:
# the card's /-slash command strings are the hermes-chat rendering.
pytestmark = pytest.mark.usefixtures("hermes_chat")

FIXED_SHA = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
FAKE_HF_CALLS: dict[str, int] = {}

VALID_LANES = """
version: 1
lanes:
- id: fixture-lane
  label: Fixture lane
  model: z-ai/glm-5.3-flash
  effort: medium
  provenance: measured 10 tasks, cov 1.0, all-in $0.0001/succ, 2026-09-27
  keywords: [fixture]
  match_hint: only a test row
- id: fixture-lane-2
  label: Fixture lane 2
  model: z-ai/glm-5.3
  effort: high
  provenance: priors
  keywords: [fixturissimo]
  match_hint: only a test row
"""

DUP_LANES = """
version: 1
lanes:
- id: fixture-lane
  label: A
  model: m
  effort: medium
- id: fixture-lane
  label: B
  model: m
  effort: medium
"""


def _make_remote(path: Path, *, lanes_yaml: str = VALID_LANES,
                 sha256: str | None = None) -> Path:
    """A fake remote dataset revision (routes/ + README.md) under path."""
    routes = path / "routes"
    routes.mkdir(parents=True)
    (routes / "routes.yaml").write_text(lanes_yaml, encoding="utf-8")
    digest = sha256 if sha256 is not None else hashlib.sha256(
        (routes / "routes.yaml").read_bytes()).hexdigest()
    (routes / "MANIFEST.json").write_text(json.dumps({
        "plugin_version": "0.4.1",
        "plugin_commit": "abc1234567890abcdef1234567890abcdef12345",
        "routes_sha256": digest,
        "measured_sha256": {},
        "generated": "2026-10-03",
    }), encoding="utf-8")
    (path / "README.md").write_text("# evalroute flywheel\n", encoding="utf-8")
    return path


def _fake_hf(remote: Path) -> types.ModuleType:
    FAKE_HF_CALLS.clear()
    def snapshot_download(repo_id, repo_type=None, revision=None,
                          allow_patterns=None, local_dir=None):
        FAKE_HF_CALLS["snapshot_download"] = FAKE_HF_CALLS.get("snapshot_download", 0) + 1
        assert repo_type == "dataset" and revision == FIXED_SHA
        for p in remote.rglob("*"):
            if p.is_file():
                dst = Path(local_dir) / p.relative_to(remote)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, dst)
    class _Api:
        def dataset_info(self, repo_id, revision=None):
            FAKE_HF_CALLS["dataset_info"] = FAKE_HF_CALLS.get("dataset_info", 0) + 1
            return types.SimpleNamespace(sha=FIXED_SHA)
    return types.SimpleNamespace(HfApi=_Api, snapshot_download=snapshot_download)


@pytest.fixture
def synced_home(tmp_path, monkeypatch):
    """HERMES_HOME isolated (conftest does it) + a fake hf repo."""
    remote = _make_remote(tmp_path / "remote")
    monkeypatch.setitem(sys.modules, "huggingface_hub", _fake_hf(remote))
    return remote


# ------------------------------------------------------------------ sync

def test_sync_pins_dataset_and_routes_against_it(synced_home, capsys, monkeypatch):
    rc = dataset.sync()
    assert rc == 0
    out = capsys.readouterr().out
    lines = [l for l in out.splitlines() if l]
    assert lines[0] == f"synced {dataset.REPO_ID} @ {FIXED_SHA[:12]}"
    assert lines[1] == ("routes: 2 lanes, 1 measured (generated 2026-10-03, "
                        "plugin 0.4.1 @ abc1234)")
    assert lines[2] == "active table: dataset (was: bundled)"
    ds_dir = Path(routing._dataset_root()) / FIXED_SHA
    assert (ds_dir / "routes" / "routes.yaml").is_file()
    # measured/ is never downloaded by sync
    assert not (ds_dir / "measured").exists()
    assert (Path(routing._dataset_root()) / "current").read_text(encoding="utf-8").strip() == FIXED_SHA
    routing.reset_routes_cache()
    lanes = routing._load_routes()
    assert [l["id"] for l in lanes] == ["fixture-lane", "fixture-lane-2"]
    assert routing.ROUTES_SOURCE["kind"] == "dataset"
    assert routing.ROUTES_SOURCE["sha"] == FIXED_SHA
    lane = lanes[0]
    card = routing.route_card(lane, 1.0, ["fixture"])
    assert (f"table: {dataset.REPO_ID} @ {FIXED_SHA[:12]} (generated 2026-10-03, "
            "plugin 0.4.1)") in card


def test_sync_sha_mismatch_refuses(tmp_path, monkeypatch, capsys):
    remote = _make_remote(tmp_path / "remote", sha256="0" * 64)
    monkeypatch.setitem(sys.modules, "huggingface_hub", _fake_hf(remote))
    rc = dataset.sync()
    assert rc == 1
    assert "routes_sha256 mismatch" in capsys.readouterr().out
    assert not (Path(routing._dataset_root()) / "current").exists()
    assert not (Path(routing._dataset_root()) / FIXED_SHA).exists()
    assert routing.ROUTES_SOURCE["kind"] == "bundled"


def test_sync_bad_lanes_refuse(tmp_path, monkeypatch, capsys):
    remote = _make_remote(tmp_path / "remote", lanes_yaml=DUP_LANES)
    monkeypatch.setitem(sys.modules, "huggingface_hub", _fake_hf(remote))
    rc = dataset.sync()
    assert rc == 1
    assert "duplicate lane id" in capsys.readouterr().out
    assert not (Path(routing._dataset_root()) / "current").exists()
    assert not (Path(routing._dataset_root()) / FIXED_SHA).exists()


def test_sync_resolve_failure_refuses(tmp_path, monkeypatch, capsys):
    class _Api:
        def dataset_info(self, repo_id, revision=None):
            raise RuntimeError("offline")
    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        types.SimpleNamespace(HfApi=_Api))
    rc = dataset.sync()
    assert rc == 1
    assert "could not resolve" in capsys.readouterr().out


# --------------------------------------------------------- status / clear

def test_status_and_clear_bundled(tmp_path, capsys, monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", _fake_hf(tmp_path / "remote"))
    rc = dataset.status()
    out = capsys.readouterr().out
    assert rc == 0
    assert "active table: bundled" in out
    assert str(routing.ROUTES_FILE) in out
    rc = dataset.clear()
    assert rc == 0
    assert "active table: bundled" in capsys.readouterr().out
    # never touched the network or the fake module's functions
    assert FAKE_HF_CALLS == {}


def test_status_and_clear_dataset(tmp_path, capsys, monkeypatch):
    remote = _make_remote(tmp_path / "remote")
    monkeypatch.setitem(sys.modules, "huggingface_hub", _fake_hf(remote))
    assert dataset.sync() == 0
    capsys.readouterr()
    calls_after_sync = dict(FAKE_HF_CALLS)
    rc = dataset.status()
    out = capsys.readouterr().out
    assert rc == 0
    assert f"active table: dataset @ {FIXED_SHA[:12]}" in out
    assert str(routing._dataset_root() / FIXED_SHA / "routes" / "routes.yaml") in out
    assert FAKE_HF_CALLS == calls_after_sync  # status made no hub calls
    rc = dataset.clear()
    out = capsys.readouterr().out
    assert rc == 0 and "active table: bundled" in out
    assert not (Path(routing._dataset_root()) / "current").exists()
    # the pinned dir is left in place
    assert (Path(routing._dataset_root()) / FIXED_SHA / "routes" / "routes.yaml").is_file()
    routing.reset_routes_cache()
    lanes = routing._load_routes()
    assert all(l["id"] != "fixture-lane" for l in lanes)
    assert routing.ROUTES_SOURCE["kind"] == "bundled"


# ------------------------------------------------------- missing hub / clear

def test_sync_without_huggingface_hub(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    rc = dataset.sync()
    assert rc == 1
    err = capsys.readouterr().err
    assert "huggingface_hub" in err
    assert not (Path(routing._dataset_root()) / "current").exists()
    # routing still works without the package
    out = json.loads(routing.evalroute_route({"task": "prove the Riemann hypothesis"}))
    assert out["lane"] == "math-first-principles"


def test_corrupt_current_falls_back_to_bundled(tmp_path, capsys, monkeypatch):
    droot = Path(routing._dataset_root())
    droot.mkdir(parents=True)
    (droot / "current").write_text("f" * 40 + "\n", encoding="utf-8")
    routing.reset_routes_cache()
    lanes = routing._load_routes()
    assert lanes[0]["id"] != "fixture-lane"
    assert routing.ROUTES_SOURCE["kind"] == "bundled"
    err = capsys.readouterr().err
    assert "unusable" in err
    out = json.loads(routing.evalroute_route({"task": "write a blog post about our launch"}))
    assert out["lane"] == "prose"


# ------------------------------------------------------------- envelope

def test_route_json_envelope_has_table():
    out = json.loads(routing.evalroute_route({"task": "prove the Riemann hypothesis"}))
    assert out["table"] == f"table: bundled (evalroute {__version__})"
    assert out["evalroute_version"] == __version__


def test_route_cli_json_has_table(tmp_path, capsys, monkeypatch):
    import argparse
    args = argparse.Namespace(evalroute_action="route", task=["hello world"],
                              lane=None, replace_route_id=None, json=True)
    rc = cli.evalroute_cli(args)
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["table"].startswith("table: bundled")


# ------------------------------------------------- byte-identical regression
# Captured from 0.4.1 (main @ 31f7392) with the same inputs: the new card must
# equal this plus exactly one inserted `table:` line after `basis:`.

MATH_041 = (
    "lane: Math, first principles (math-first-principles)\n"
    "route: qwen/qwen3.8-max @ max   <- run: /model qwen/qwen3.8-max\n"
    "facets: symbolic\n"
    "escalation: gpt-6-sol (when coverage gaps or the task turns out harder)\n"
    "classification: rules match (0.67) on: derivation, proof\n"
    "basis: priors 2026-09-26 (Qwen3.8-Max MathArena 56% vs GPT-6 Sol 85%, Opus 5.5 83%)\n"
    "note: ~27-30 pts behind closed; skip open for one-shot math except Lean-verified pipelines\n"
    "why here: correctness comes from reasoning, not from tools\n"
    "route id: abc-123 (use /rate pass|fail --route-id abc-123 if routes overlap)\n"
    "next: /model qwen/qwen3.8-max then /reasoning max | wrong lane? /route --lane <id> <same task> "
    "| when done: /rate pass|fail --note why"
)

PROSE_041 = (
    "lane: Prose (prose)\n"
    "route: moonshotai/kimi-k3 @ medium   <- run: /model moonshotai/kimi-k3\n"
    "escalation: claude-opus-5-5 (when coverage gaps or the task turns out harder)\n"
    "classification: lane pinned by caller\n"
    "basis: priors 2026-09-26 (K3 EQ-Bench creative writing #2 behind Opus 5; judged by a Claude model)\n"
    "note: untested prior; blind-test K3 vs Opus 5.5 at low-medium\n"
    "why here: human is the checker\n"
    "route id: def-456 (use /rate pass|fail --route-id def-456 if routes overlap)\n"
    "next: /model moonshotai/kimi-k3 then /reasoning medium | wrong lane? /route --lane <id> <same task> "
    "| when done: /rate pass|fail --note why"
)


@pytest.mark.parametrize("lane_id,conf,hits,expected,pinned,method", [
    ("math-first-principles", 0.67, ["proof", "derivation"], MATH_041, False, "rules-weak"),
    ("prose", 1.0, [], PROSE_041, True, "pinned"),
])
def test_card_is_041_plus_one_table_line(lane_id, conf, hits, expected, pinned, method):
    lane = routing._lane_by_id(lane_id)
    route_id = "abc-123" if lane_id == "math-first-principles" else "def-456"
    card = routing.route_card(lane, conf, hits, pinned=pinned, method=method,
                            facets=["symbolic"] if not pinned else None,
                            route_id=route_id)
    assert card != expected  # the table line is new
    stripped = "\n".join(l for l in card.splitlines() if not l.startswith("table: "))
    assert stripped == expected
    lines = card.splitlines()
    basis_i = lines.index(next(l for l in lines if l.startswith("basis: ")))
    assert lines[basis_i + 1].startswith("table: ")
