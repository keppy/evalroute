"""Offline contract tests: no provider call, no paid judge."""
import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]

from evalroute.harness import tier_a as h


def model(**changes):
    return {**{"name": "arm", "model": "example/model", "api": "openai", "effort": "medium",
               "in": 1.0, "out": 2.0, "max_tokens": 100}, **changes}


def task(**changes):
    return {**{"id": "t", "lane": "coding", "prompt": "ANSWER: yes",
               "check": {"type": "exact", "answer": "yes"}}, **changes}


def args(tmp_path, models, tasks, **changes):
    mp, tp, out = tmp_path / "models.json", tmp_path / "tasks.jsonl", tmp_path / "runs.jsonl"
    mp.write_text(json.dumps(models), encoding="utf-8")
    tp.write_text("".join(json.dumps(t) + "\n" for t in tasks), encoding="utf-8")
    return SimpleNamespace(models=str(mp), tasks=str(tp), out=str(out), only=None, judge=None,
                           conc=1, max_tokens=100, k=1, **changes)


def test_signature_changes_for_every_result_affecting_input():
    m, t = model(), task()
    base = h.run_signature(m, t, max_tokens=100, judge=None)
    assert h.run_signature({**m, "in": 1.1}, t, 100, None) != base
    assert h.run_signature({**m, "out": 2.1}, t, 100, None) != base
    assert h.run_signature({**m, "max_tokens": 101}, t, 101, None) != base
    no_cap = {k: v for k, v in m.items() if k != "max_tokens"}
    assert h.run_signature(no_cap, t, 100, None) != h.run_signature(no_cap, t, 101, None)
    assert h.run_signature(m, {**t, "prompt": "another"}, 100, None) != base
    assert h.run_signature(m, {**t, "check": {"type": "exact", "answer": "no"}}, 100, None) != base
    judged = task(check={"type": "judge", "rubric": "be correct"})
    assert h.run_signature(m, judged, 100, model(name="judge")) != h.run_signature(m, judged, 100, None)
    assert h.run_signature(m, judged, 100, {**model(name="judge"), "out": 3}) != h.run_signature(m, judged, 100, model(name="judge"))
    assert h.arm_contract(m, judged, 100, model(name="judge")) != h.arm_contract(m, judged, 100, model(name="judge", out=3))
    assert h.key({"task": "t", "model": "arm", "sample": 0, "cfg": "legacy"}) == "t|arm|0|legacy"


def test_resume_rejects_mixed_checker_then_reexecutes_in_separate_run(tmp_path, monkeypatch):
    calls = []
    async def fake_call(m, system, prompt, maximum):
        calls.append((prompt, maximum))
        return "ANSWER: yes", {"inp": 10, "out": 2, "cache_read": 0, "cache_write": 0}, 0.1, False
    monkeypatch.setattr(h, "call", fake_call)
    a = args(tmp_path, [model()], [task()])
    asyncio.run(h.run(a)); asyncio.run(h.run(a))
    assert len(calls) == 1
    Path(a.tasks).write_text(json.dumps(task(check={"type": "exact", "answer": "no"})) + "\n")
    with pytest.raises(SystemExit, match="mixed task contract"):
        asyncio.run(h.run(a))
    assert len(calls) == 1  # refuses before paid call
    a.out = str(tmp_path / "new-checker.jsonl")
    asyncio.run(h.run(a))
    assert len(calls) == 2
    Path(a.models).write_text(json.dumps([{**model(), "in": 1.1}]))
    asyncio.run(h.run(a))
    assert len(calls) == 3
    rows = h.jread(a.out)
    assert len({h.key(r) for r in rows}) == 2

def test_legacy_runs_reportable_but_not_resumable(tmp_path, monkeypatch):
    a = args(tmp_path, [model()], [task()])
    h.jadd(a.out, {"task": "t", "model": "arm", "sample": 0, "cfg": "old",
                   "ph": h.phash(task()), "text": "yes", "passed": True})
    async def no_call(*a):
        pytest.fail("legacy record must not trigger a provider call")
    monkeypatch.setattr(h, "call", no_call)
    with pytest.raises(SystemExit, match="mixed task contract"):
        asyncio.run(h.run(a))


def test_duplicate_ids_rejected_before_any_calls(tmp_path, monkeypatch):
    async def no_call(*a):
        pytest.fail("provider call not allowed")
    monkeypatch.setattr(h, "call", no_call)
    a = args(tmp_path, [model(), model()], [task()])
    with pytest.raises(SystemExit, match="duplicate model"):
        asyncio.run(h.run(a))
    Path(a.models).write_text(json.dumps([model()]))
    Path(a.tasks).write_text(json.dumps(task()) + "\n" + json.dumps(task()) + "\n")
    with pytest.raises(SystemExit, match="duplicate task"):
        asyncio.run(h.run(a))


def test_report_rejects_mixed_task_contract_but_vendored_legacy_reports(tmp_path, capsys):
    a = SimpleNamespace(out=str(tmp_path / "runs.jsonl"), usd_per_hour=100, tol=0, csv=None)
    old = {"task": "t", "lane": "coding", "model": "arm", "sample": 0, "cfg": "old",
           "ph": "old", "text": "yes", "check": "exact", "passed": True, "cost": 0.1,
           "jcost": 0, "usage": {"out": 10}, "latency": 0.1, "effort": "medium"}
    h.jadd(a.out, old)
    h.report(a)
    h.jadd(a.out, {**old, "sample": 1, "ph": "different"})
    with pytest.raises(SystemExit, match="mixed.*task"):
        h.report(a)
    a.out = str(ROOT / "examples" / "artifacts" / "tier-a-routine-coding" / "runs.jsonl")
    h.report(a)
    assert "routine coding" in capsys.readouterr().out

def test_report_keeps_distinct_price_schedules_in_separate_arms(tmp_path, capsys):
    a = SimpleNamespace(out=str(tmp_path / "runs.jsonl"), usd_per_hour=100, tol=0, csv=None)
    base = {"task": "t", "lane": "coding", "model": "arm", "ph": "same",
            "text": "yes", "check": "exact", "passed": True, "jcost": 0,
            "usage": {"out": 10}, "latency": .1, "effort": "medium"}
    h.jadd(a.out, {**base, "sample": 0, "cfg": "aaaaaaaaaaaaaaaa", "cost": .1})
    h.jadd(a.out, {**base, "sample": 1, "cfg": "bbbbbbbbbbbbbbbb", "cost": .2})
    h.report(a)
    output = capsys.readouterr().out
    assert "arm#aaaaaa" in output and "arm#bbbbbb" in output


def test_judge_pending_not_reported_as_checked(tmp_path, capsys):
    a = SimpleNamespace(out=str(tmp_path / "runs.jsonl"), usd_per_hour=100, tol=0, csv=None)
    h.jadd(a.out, {"task": "t", "lane": "alignment", "model": "arm", "sample": 0, "cfg": "old",
                   "ph": "p", "text": "text", "check": "judge", "passed": None, "cost": 0.2,
                   "jcost": 0, "usage": {"out": 10}, "latency": 0.1, "effort": "medium"})
    h.report(a)
    assert "route -> None" in capsys.readouterr().out

def test_pending_judge_requires_real_human_grade_not_audit(tmp_path, monkeypatch):
    t = task(check={"type": "judge", "rubric": "correct"})
    a = args(tmp_path, [model()], [t])
    a.audit = 0
    h.jadd(a.out, {"task": "t", "lane": "coding", "model": "arm", "sample": 0,
                   "cfg": "v2", "sig": "s", "contract_version": 2, "ph": h.task_contract(t),
                   "text": "ANSWER: yes", "check": "judge", "passed": None})
    monkeypatch.setattr("builtins.input", lambda _: "p")
    h.grade(a)
    grade = h.jread(a.out)[-1]
    assert grade["audit"] is False and grade["passed"] is True


def test_mixed_checker_tasks_keep_one_arm_but_distinct_cell_signatures():
    m, j = model(), model(name="judge", model="example/judge")
    automatic = task(id="auto", check={"type": "exact", "answer": "yes"})
    judged = task(id="judged", check={"type": "judge", "rubric": "yes"})
    assert h.arm_contract(m, automatic, 100, j) == h.arm_contract(m, judged, 100, j)
    assert h.run_signature(m, automatic, 100, j) != h.run_signature(m, judged, 100, j)


def test_pending_grade_cannot_win_on_partial_coverage_or_free_cost(tmp_path, capsys):
    a = SimpleNamespace(out=str(tmp_path / "runs.jsonl"), csv=str(tmp_path / "report.csv"),
                        tasks=str(tmp_path / "tasks.jsonl"), usd_per_hour=100, tol=0, k=1)
    task_specs = {id_: {"id": id_, "lane": "research", "prompt": id_,
                        "check": {"type": "judge" if id_ == "b" else "exact"}}
                  for id_ in ("a", "b")}
    for spec in task_specs.values():
        h.jadd(a.tasks, spec)
    for name, cost in (("cheap", 1.0), ("complete", 3.0)):
        for task_id in ("a", "b"):
            pending = name == "cheap" and task_id == "b"
            h.jadd(a.out, {"task": task_id, "lane": "research", "model": name,
                           "sample": 0, "cfg": name, "ph": h.phash(task_specs[task_id]), "text": "answer",
                           "check": "judge" if task_id == "b" else "exact", "passed": None if pending else True,
                           "cost": cost, "jcost": 0, "usage": {"out": 5}, "latency": 0.1,
                           "effort": "medium"})
    h.report(a)
    assert "route -> complete" in capsys.readouterr().out
    import csv
    rows = {r["model"]: r for r in csv.DictReader(Path(a.csv).open())}
    assert rows["cheap"]["complete"] == "False"
    assert rows["cheap"]["expected"] == "2"
    assert float(rows["cheap"]["cov"]) == 0.5
    assert float(rows["cheap"]["succ_usd"]) == 2.0  # includes paid pending attempt
    assert rows["complete"]["complete"] == "True"


def test_missing_declared_task_is_never_inferred_complete(tmp_path, capsys):
    a = SimpleNamespace(out=str(tmp_path / "runs.jsonl"), csv=str(tmp_path / "report.csv"),
                        tasks=str(tmp_path / "tasks.jsonl"), usd_per_hour=100, tol=0, k=1)
    for task_id in ("seen", "not-run"):
        h.jadd(a.tasks, {"id": task_id, "lane": "research", "prompt": task_id,
                         "check": {"type": "exact", "answer": "yes"}})
    spec = h.jread(a.tasks)[0]
    h.jadd(a.out, {"task": spec["id"], "lane": "research", "model": "cheap",
                   "sample": 0, "cfg": "a", "ph": h.phash(spec), "text": "yes",
                   "check": "exact", "passed": True, "cost": 1.0, "jcost": 0,
                   "usage": {"out": 5}, "latency": 0.1, "effort": "medium"})
    h.report(a)
    assert "route -> None" in capsys.readouterr().out
    import csv
    row = next(csv.DictReader(Path(a.csv).open()))
    assert row["complete"] == "False" and row["expected"] == "2"
    a.tasks = str(tmp_path / "absent.jsonl")
    h.report(a)
    assert "route -> None" in capsys.readouterr().out


def test_report_rejects_stale_declared_task_contract(tmp_path):
    a = SimpleNamespace(out=str(tmp_path / "runs.jsonl"), tasks=str(tmp_path / "tasks.jsonl"),
                        csv=None, usd_per_hour=100, tol=0, k=1)
    spec = {"id": "x", "lane": "research", "prompt": "original", "check": {"type": "exact"}}
    h.jadd(a.tasks, spec)
    h.jadd(a.out, {"task": "x", "lane": "research", "model": "m", "sample": 0,
                   "cfg": "a", "ph": h.phash(spec), "text": "yes", "check": "exact",
                   "passed": True, "cost": 1, "jcost": 0, "usage": {}, "latency": 0.1})
    Path(a.tasks).write_text(json.dumps({**spec, "prompt": "changed"}) + "\n")
    with pytest.raises(SystemExit, match="do not match declared"):
        h.report(a)
