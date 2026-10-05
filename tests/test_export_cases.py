"""Tests for evalroute export-cases.

Every test points HERMES_HOME at a tmp dir so labels never touch the real
<home>/evalroute/labels.jsonl; the exported file lands in tmp too.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from evalroute import export_cases, flywheel

SENTINEL = "UNIQUE-TASK-TEXT-sentinel-zqxv"


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("EVALROUTE_HOME", str(tmp_path))
    flywheel._MEMORY.clear()
    yield tmp_path
    flywheel._MEMORY.clear()


def _lane():
    return {"id": "routine-coding", "label": "Routine coding",
            "model": "m", "effort": "e"}


def _write_ledger(home, records):
    p = home / "evalroute" / "labels.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for r in records:
        r.setdefault("ts", 1.0)
        r.setdefault("iso", "2026-10-04T00:00:00+00:00")
        lines.append(json.dumps(r))
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _pinned(route_id, task, lane="routine-coding", method="pinned"):
    return {"kind": "route", "id": route_id, "task": task, "lane": lane,
            "method": method, "confidence": 1.0}


def _taskset_dir(tmp_path):
    d = tmp_path / "tasksets" / "tier-a-routine-coding"
    d.mkdir(parents=True)
    rows = [{"id": "rc-1", "lane": "routine-coding", "prompt": "write slugify"},
            {"id": "rc-2", "lane": "routine-coding", "prompt": SENTINEL}]
    (d / "tasks.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return d


def _read(out):
    return [json.loads(l) for l in Path(out).read_text(encoding="utf-8").splitlines()
            if l.strip()]


def test_pinned_exported_nonpinned_unlabeled(home, tmp_path):
    _write_ledger(home, [
        _pinned("a" * 32, "task one"),
        _pinned("b" * 32, SENTINEL, method="llm"),
        _pinned("c" * 32, "task three", method="rules-strong"),
    ])
    out = tmp_path / "cases.jsonl"
    rc = export_cases.run_export(out, as_json=True)
    assert rc == 0
    rows = _read(out)
    assert {r["id"] for r in rows} == {"a" * 32}
    # summary via json stdout: re-run and capture
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = export_cases.run_export(out, as_json=True)
    stats = json.loads(buf.getvalue())
    assert stats["unlabeled"] == 2
    assert stats["sources"]["ledger"] == 1


def test_correction_written_twice(home, tmp_path):
    _write_ledger(home, [
        _pinned("old" * 16, SENTINEL, lane="dl-ml-research-engineering"),
        {"kind": "lane_correction", "from_lane": "dl-ml-research-engineering",
         "to_lane": "routine-coding", "task": SENTINEL[:200]},
        _pinned("new" * 16, SENTINEL, lane="routine-coding"),
    ])
    out = tmp_path / "cases.jsonl"
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = export_cases.run_export(out, as_json=True)
    assert rc == 0
    rows = _read(out)
    ids = sorted(r["id"] for r in rows)
    assert ids == ["new" * 16, "new" * 16 + "#2"]
    stats = json.loads(buf.getvalue())
    assert stats["sources"]["correction"] == 2


def test_tasksets_merge(home, tmp_path):
    d = _taskset_dir(tmp_path)
    out = tmp_path / "cases.jsonl"
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        export_cases.run_export(out, tasksets=str(d), as_json=True)
    rows = _read(out)
    assert {"taskset:rc-1", "taskset:rc-2"} <= {r["id"] for r in rows}
    assert all(set(r) == {"id", "text", "label"} for r in rows)


def test_seed_text_rows(home, tmp_path):
    out = tmp_path / "cases.jsonl"
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        export_cases.run_export(out, seed_text=True, as_json=True)
    rows = _read(out)
    seed_ids = [r["id"] for r in rows if r["id"].startswith("seed:")]
    # one hint + one keywords row per lane (9 lanes)
    assert len([i for i in seed_ids if i.endswith(":hint")]) == 9
    assert len([i for i in seed_ids if i.endswith(":keywords")]) == 9
    assert all(set(r) == {"id", "text", "label"} for r in rows)


def test_exact_text_dedupe(home, tmp_path):
    _write_ledger(home, [_pinned("a" * 32, "same text")])
    d = _taskset_dir(tmp_path)
    # make the taskset row duplicate the ledger text exactly
    p = d / "tasks.jsonl"
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    rows.append({"id": "rc-3", "lane": "routine-coding", "prompt": "same text"})
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    out = tmp_path / "cases.jsonl"
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        export_cases.run_export(out, tasksets=str(d), as_json=True)
    texts = [r["text"] for r in _read(out)]
    assert texts.count("same text") == 1


def test_stdout_never_contains_task_text(home, tmp_path, capsys):
    _write_ledger(home, [_pinned("a" * 32, SENTINEL)])
    d = _taskset_dir(tmp_path)
    out = tmp_path / "cases.jsonl"
    rc = export_cases.run_export(out, tasksets=str(d), as_json=False)
    captured = capsys.readouterr().out
    assert SENTINEL not in captured
    assert "write slugify" not in captured
    assert rc == 0
    # file does contain it (that's the point)
    assert SENTINEL in Path(out).read_text(encoding="utf-8")


def test_strict_exit_3_on_short_lane(home, tmp_path):
    _write_ledger(home, [_pinned("a" * 32, "task one")])
    out = tmp_path / "cases.jsonl"
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = export_cases.run_export(out, min_per_lane=20, strict=True,
                                     as_json=True)
    assert rc == 3
    stats = json.loads(buf.getvalue())
    assert stats["below_min"] and stats["below_min"][0]["count"] < 20
    # non-strict exits 0 on the same data
    with contextlib.redirect_stdout(buf):
        rc = export_cases.run_export(out, min_per_lane=20, strict=False,
                                     as_json=True)
    assert rc == 0


def test_summary_counts_match_file(home, tmp_path):
    _write_ledger(home, [
        _pinned("a" * 32, "task one"),
        _pinned("b" * 32, SENTINEL, method="llm"),
    ])
    d = _taskset_dir(tmp_path)
    out = tmp_path / "cases.jsonl"
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        export_cases.run_export(out, tasksets=str(d), as_json=True)
    stats = json.loads(buf.getvalue())
    rows = _read(out)
    assert stats["total"] == len(rows)
    assert stats["sources"]["ledger"] + stats["sources"]["correction"] \
        + stats["sources"]["taskset"] + stats["sources"]["seed"] == len(rows)
    assert stats["unlabeled"] == 1


def test_json_shape(home, tmp_path):
    _write_ledger(home, [_pinned("a" * 32, "task one")])
    out = tmp_path / "cases.jsonl"
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        export_cases.run_export(out, as_json=True)
    stats = json.loads(buf.getvalue())
    for key in ("sources", "unlabeled", "total", "per_lane", "below_min", "out"):
        assert key in stats
    assert set(stats["sources"]) == {"ledger", "correction", "taskset", "seed"}


def test_rows_validate_contract_shape(home, tmp_path):
    _write_ledger(home, [
        _pinned("a" * 32, "task one"),
        _pinned("b" * 32, SENTINEL, method="llm"),
    ])
    d = _taskset_dir(tmp_path)
    out = tmp_path / "cases.jsonl"
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        export_cases.run_export(out, tasksets=str(d), seed_text=True,
                                as_json=True)
    rows = _read(out)
    assert rows
    for r in rows:
        assert set(r) == {"id", "text", "label"}
        assert r["id"] and r["text"] and r["label"]
        assert isinstance(r["text"], str)


def test_cli_end_to_end(home, tmp_path):
    _write_ledger(home, [_pinned("a" * 32, "task one")])
    out = tmp_path / "cases.jsonl"
    env = dict()
    env.update(**{k: v for k, v in __import__("os").environ.items()})
    env["HERMES_HOME"] = str(home)
    env["EVALROUTE_HOME"] = str(home)
    env["PYTHONPATH"] = ""
    proc = subprocess.run(
        [str(Path(__file__).resolve().parents[1] / ".venv" / "Scripts" / "evalroute.exe"),
         "export-cases", "--out", str(out), "--json"],
        capture_output=True, text=True, env=env,
        cwd=str(Path(__file__).resolve().parents[1]))
    assert proc.returncode == 0, proc.stderr
    stats = json.loads(proc.stdout)
    assert stats["total"] >= 1
    rows = _read(out)
    assert rows and rows[0]["id"] == "a" * 32


def test_taskset_lane_spelling_normalised(tmp_path):
    """Harness tasksets spell lanes 'routine coding'; labels must be routes.yaml ids,
    or the encoder learns a tenth lane that no route can ever match."""
    from evalroute import export_cases
    d = tmp_path / "tier-a-x"; d.mkdir()
    (d / "tasks.jsonl").write_text(
        '{"id": "t1", "lane": "routine coding", "prompt": "add a flag"}\n', encoding="utf-8")
    rows = export_cases._load_tasksets(str(tmp_path))
    assert [r["label"] for r in rows] == ["routine-coding"]
