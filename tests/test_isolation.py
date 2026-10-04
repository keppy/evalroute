"""Test isolation: the fallback hermes home must honor HERMES_HOME.

Without Hermes installed (any plain venv), labels_path() falls back to
Path.home()/".hermes" — which would silently append fixture rows to the
user's live ledger. HERMES_HOME must win in that fallback too.
"""

from __future__ import annotations

import sys
import json

from evalroute import routing
from evalroute import flywheel


def test_labels_path_honors_hermes_home_without_hermes_constants(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # hermes_constants unimportable for this test (None -> import raises ImportError)
    monkeypatch.setitem(sys.modules, "hermes_constants", None)
    p = flywheel.labels_path()
    assert p.is_relative_to(tmp_path), p
    assert p.name == "labels.jsonl"
    # and the append path actually lands there, not in ~/.hermes
    flywheel._append({"kind": "route", "task": "isolation probe", "lane": "routine-coding",
                      "lane_label": "Routine coding", "model": "m", "effort": "low",
                      "method": "rules", "confidence": 1.0})
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == 1 and rows[0]["task"] == "isolation probe"


def test_hermes_constants_missing_entirely(tmp_path, monkeypatch):
    # some venvs never had hermes_constants at all: import must raise
    monkeypatch.setitem(sys.modules, "hermes_constants", None)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    try:
        import hermes_constants  # noqa: F401
        monkeypatch.delitem(sys.modules, "hermes_constants")
    except ImportError:
        pass
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert flywheel.labels_path().is_relative_to(tmp_path)


def test_effort_auto_reads_hermes_home_config_without_hermes_constants(tmp_path, monkeypatch):
    # hermes_constants unimportable for this test (None -> import raises ImportError)
    monkeypatch.setitem(sys.modules, "hermes_constants", None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    lane = routing._lane_by_id("routine-coding")
    model, effort = lane["model"], routing._effort_for_override(lane)
    # no config file -> not auto
    assert routing._effort_auto(lane) is False
    # config.yaml with the lane's model -> effort override -> auto
    (tmp_path / "config.yaml").write_text(
        f"agent:\n  reasoning_overrides:\n    {model}: {effort}\n", encoding="utf-8")
    assert routing._effort_auto(lane) is True


# ------------------------------------------------- env precedence (C4 + H5)

def _fake_hermes_constants(tmp_path, monkeypatch, home_dir):
    """An importable hermes_constants naming a decoy home dir."""
    monkeypatch.delitem(sys.modules, "hermes_constants", raising=False)  # force a fresh import
    mod_dir = tmp_path / "fake_pkg"
    mod_dir.mkdir(exist_ok=True)
    (mod_dir / "hermes_constants.py").write_text(
        f"def get_hermes_home():\n    return {home_dir!r}\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(mod_dir))


def test_evalroute_home_beats_hermes_home_and_hermes_constants(tmp_path, monkeypatch):
    """Both env vars set, hermes_constants importable -> EVALROUTE_HOME wins."""
    er_home, h_home, const_home = (tmp_path / "er"), (tmp_path / "hh"), (tmp_path / "const")
    _fake_hermes_constants(tmp_path, monkeypatch, str(const_home))
    monkeypatch.setenv("EVALROUTE_HOME", str(er_home))
    monkeypatch.setenv("HERMES_HOME", str(h_home))
    assert flywheel.labels_path().is_relative_to(er_home)


def test_evalroute_home_beats_none_importable_hermes_constants(tmp_path, monkeypatch):
    """Same, with sys.modules['hermes_constants'] = None (import raises)."""
    er_home, h_home = (tmp_path / "er"), (tmp_path / "hh")
    monkeypatch.setitem(sys.modules, "hermes_constants", None)
    monkeypatch.setenv("EVALROUTE_HOME", str(er_home))
    monkeypatch.setenv("HERMES_HOME", str(h_home))
    assert flywheel.labels_path().is_relative_to(er_home)


def test_hermes_home_beats_importable_hermes_constants(tmp_path, monkeypatch):
    """Only HERMES_HOME set, fake importable hermes_constants -> HERMES_HOME wins."""
    h_home, const_home = (tmp_path / "hh"), (tmp_path / "const")
    const_home.mkdir()
    _fake_hermes_constants(tmp_path, monkeypatch, str(const_home))
    monkeypatch.setenv("HERMES_HOME", str(h_home))
    assert flywheel.labels_path().is_relative_to(h_home)


def test_importable_hermes_constants_used_when_no_env(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.delenv("EVALROUTE_HOME", raising=False)
    const_home = tmp_path / "const"
    _fake_hermes_constants(tmp_path, monkeypatch, str(const_home))
    assert flywheel.labels_path().is_relative_to(const_home)


def test_platform_default_home_matches_hermes(tmp_path, monkeypatch):
    """No env, no hermes_constants: Windows -> %LOCALAPPDATA%\hermes (if it exists),
    elsewhere ~/.hermes — the same place Hermes itself keeps the ledger."""
    from evalroute import paths
    monkeypatch.setitem(sys.modules, "hermes_constants", None)
    monkeypatch.delenv("EVALROUTE_HOME", raising=False)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setattr(paths.Path, "home", classmethod(lambda cls: tmp_path / "userhome"))
    monkeypatch.setattr(paths.os, "name", "nt")
    local = tmp_path / "LocalAppData"
    monkeypatch.setenv("LOCALAPPDATA", str(local))
    # neither exists yet -> Hermes's Windows default
    assert paths.hermes_home() == local / "hermes"
    # only a legacy ~/.hermes exists -> keep using it (no silent ledger split)
    (tmp_path / "userhome" / ".hermes").mkdir(parents=True)
    assert paths.hermes_home() == tmp_path / "userhome" / ".hermes"
    # the real Hermes home exists -> it wins
    (local / "hermes").mkdir(parents=True)
    assert paths.hermes_home() == local / "hermes"
    monkeypatch.setattr(paths.os, "name", "posix")
    assert paths.hermes_home() == tmp_path / "userhome" / ".hermes"
