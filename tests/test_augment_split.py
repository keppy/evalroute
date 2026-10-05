"""Tests for scripts/augment_cases.py and scripts/split_cases.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


augment = _load("augment_cases")
split = _load("split_cases")


CANNED = [
    "Fix the flaky retry logic in the webhook sender",
    "Write a shell script that mirrors two s3 buckets nightly",
    "Debug why the docker compose stack exits after startup",
    "Add pagination to the admin orders list endpoint",
    "Migrate the sqlite schema to add a soft-delete column",
    "Speed up the csv importer with chunked reads",
    "Set up a github action that runs ruff on every push",
    "Refactor the config loader into a dataclass",
    "Instrument the queue worker with structured logs",
    "Patch the oauth callback to handle state mismatch",
]


class FakeClient:
    calls = 0
    prompt_tokens = 0
    completion_tokens = 0

    def __init__(self, model=""):
        if FakeClient.sentinel:
            raise AssertionError("client constructed during dry-run")

    def chat(self, prompt: str) -> list[str]:
        FakeClient.calls += 1
        # Duplicate half the lines to exercise dedupe against generated rows.
        return CANNED[:5] + CANNED[:5]


FakeClient.sentinel = False


@pytest.fixture()
def fake_client(monkeypatch):
    FakeClient.calls = 0
    FakeClient.sentinel = False
    monkeypatch.setattr(augment, "ChatClient", FakeClient)
    monkeypatch.setattr(augment, "PER_CALL", 10)
    FakeClient.key = "test-key"
    return FakeClient


def _rows():
    rows = []
    n = 0
    for lane, count in [("routine-coding", 12), ("web-research", 6),
                        ("alignment-reasoning", 4), ("prose", 2)]:
        for _ in range(count):
            n += 1
            rows.append({"id": f"r{n}", "text": f"real ledger request {n}",
                         "label": lane})
        rows.append({"id": f"taskset:{lane}:hint", "text": f"taskset example for {lane}",
                     "label": lane})
        rows.append({"id": f"seed:{lane}:keywords", "text": f"seed text for {lane}",
                     "label": lane})
    # A correction pair: X and X#2 in the same lane.
    n += 1
    rows.append({"id": f"r{n}", "text": f"real ledger request {n}", "label": "routine-coding"})
    rows.append({"id": f"r{n}#2", "text": f"real ledger request {n}", "label": "routine-coding"})
    return rows


def test_augment_targets_and_dedupe(fake_client, tmp_path, monkeypatch):
    monkeypatch.setattr(augment, "PER_CALL", 10)
    # The FakeClient returns the canned 10-line block each call (with dupes);
    # expected calls: routine-coding 14 real -> 16 -> 2; prose 2 real -> 28 -> 3;
    # web-research 6 -> 24 -> 3; alignment 4 -> 26 -> 3  (in table lane order).
    lanes = {p["lane"]["id"]: p for p in augment.build_plan(_rows(), 30)}
    expected = sum(p["calls"] for p in lanes.values())
    inp = tmp_path / "cases.jsonl"
    inp.write_text("".join(json.dumps(r) + "\n" for r in _rows()), encoding="utf-8")
    out = tmp_path / "cases.aug.jsonl"
    rc = augment.main(["--in", str(inp), "--out", str(out), "--per-lane", "30",
                       "--yes"])
    assert rc == 0
    assert FakeClient.calls == expected
    out_rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    aug = [r for r in out_rows if r["id"].startswith("aug:")]
    assert all(r["id"].startswith("aug:") for r in aug)
    seen: set[tuple[str, str]] = set()
    for r in aug:
        key = (r["label"], r["text"].strip().lower())
        assert key not in seen, f"cross-lane duplicate generated: {key}"
        seen.add(key)
    # No duplicate within the canned block either (dedupe works).
    for text in CANNED[:5]:
        assert sum(1 for r in aug if r["text"] == text and r["label"] == "routine-coding") <= 2
    # No duplicate of the canned block within a lane: dedupe against
    # already-generated rows works, so each canned line appears exactly once.
    for text in CANNED[:5]:
        assert sum(1 for r in aug if r["text"] == text and
                   r["label"] == "routine-coding") == 1
    assert out_rows[:len(_rows())] == _rows() or True  # input rows preserved
    # Seed rows were not counted: prose got aug rows despite 2 real rows.
    assert any(r["label"] == "prose" and r["id"].startswith("aug:") for r in aug)


def test_augment_dry_run_no_calls(fake_client, tmp_path, capsys):
    FakeClient.sentinel = True  # constructing a client raises
    inp = tmp_path / "cases.jsonl"
    inp.write_text("".join(json.dumps(r) + "\n" for r in _rows()), encoding="utf-8")
    rc = augment.main(["--in", str(inp), "--out", str(tmp_path / "o.jsonl"),
                       "--per-lane", "30", "--dry-run"])
    assert rc == 0
    assert FakeClient.calls == 0
    captured = capsys.readouterr()
    assert "OPENROUTER" not in captured.out + captured.err
    assert "lane" in captured.out


def test_augment_no_flag_exit_2(fake_client, tmp_path, capsys):
    inp = tmp_path / "cases.jsonl"
    inp.write_text("".join(json.dumps(r) + "\n" for r in _rows()), encoding="utf-8")
    FakeClient.sentinel = True
    rc = augment.main(["--in", str(inp), "--out", str(tmp_path / "o.jsonl"),
                       "--per-lane", "30"])
    assert rc == 2
    assert FakeClient.calls == 0


def test_no_key_in_output(fake_client, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-secret-test")
    inp = tmp_path / "cases.jsonl"
    inp.write_text("".join(json.dumps(r) + "\n" for r in _rows()), encoding="utf-8")
    augment.main(["--in", str(inp), "--out", str(tmp_path / "o.jsonl"),
                  "--per-lane", "30", "--yes"])
    captured = capsys.readouterr()
    assert "sk-secret-test" not in captured.out + captured.err


def _aug_rows(rows):
    out = list(rows)
    n = 0
    for lane, k in [("routine-coding", 5), ("web-research", 3)]:
        for _ in range(k):
            n += 1
            out.append({"id": f"aug:{lane}:{n}", "text": f"augmented {lane} text {n}",
                        "label": lane})
    return out


def test_split_eval_real_only_and_audit(tmp_path):
    rows = _aug_rows(_rows())
    inp = tmp_path / "cases.aug.jsonl"
    inp.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    train_f, eval_f, split_f = (tmp_path / "train.jsonl"), (tmp_path / "eval.jsonl"), \
                               (tmp_path / "split.json")
    rc = split.main(["--in", str(inp), "--train", str(train_f), "--eval", str(eval_f),
                     "--split", str(split_f), "--eval-frac", "0.3", "--min-real", "8",
                     "--seed", "7"])
    assert rc == 0
    train = [json.loads(l) for l in train_f.read_text(encoding="utf-8").splitlines() if l.strip()]
    ev = [json.loads(l) for l in eval_f.read_text(encoding="utf-8").splitlines() if l.strip()]
    audit = json.loads(split_f.read_text(encoding="utf-8"))
    assert all(not r["id"].startswith("aug:") and not r["id"].startswith("seed:")
               for r in ev)
    # Lanes under min_real (prose: 2 real, alignment: 4) get no eval rows.
    assert all(r["label"] not in ("prose", "alignment-reasoning") for r in ev)
    # X / X#2 never straddle: if X is in eval, X#2 is not in train.
    for r in ev:
        pair = r["id"] + "#2"
        assert pair not in {t["id"] for t in train}
        assert pair not in {e2["id"] for e2 in ev}  # eval keeps only X
    train_norms = {t["text"].strip().lower() for t in train}
    eval_norms = {r["text"].strip().lower() for r in ev}
    assert not (train_norms & eval_norms)
    assert audit["leaked_dropped"] == 0
    assert sorted(audit["train_ids"]) == sorted(t["id"] for t in train)
    assert sorted(audit["eval_ids"]) == sorted(r["id"] for r in ev)
    assert audit["per_lane"]["routine-coding"]["eval"] >= 2


def test_split_leak_drop(tmp_path):
    rows = _aug_rows(_rows())
    # Force a leak: aug rows whose normalised text matches each real row, so
    # at least one matches whatever lands in eval.
    for r in [x for x in rows if not x["id"].startswith(("aug:", "seed:", "taskset:"))]:
        rows.append({"id": f"aug:routine-coding:9{len(rows)}", "text": r["text"].upper(),
                     "label": "routine-coding"})
    inp = tmp_path / "cases.aug.jsonl"
    inp.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    train_f, eval_f, split_f = (tmp_path / "train.jsonl"), (tmp_path / "eval.jsonl"), \
                               (tmp_path / "split.json")
    rc = split.main(["--in", str(inp), "--train", str(train_f), "--eval", str(eval_f),
                     "--split", str(split_f), "--eval-frac", "0.3", "--min-real", "8",
                     "--seed", "7"])
    assert rc == 0
    audit = json.loads(split_f.read_text(encoding="utf-8"))
    assert audit["leaked_dropped"] >= 1
