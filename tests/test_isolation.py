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
