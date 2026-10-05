"""tests/test_contribute.py — whitelist redaction, gate, cursor, upload, merge invariant.

No network in this file: the HfApi is faked; `contribute` (non-dry-run) is
only exercised with the fake in place.
"""

from __future__ import annotations

import json
import re
import sys
import time
import types
from pathlib import Path

import pytest
import yaml

from evalroute import contribute
from evalroute import dataset
from evalroute import routes_from_labels as rfl

BASE_TS = time.mktime(time.strptime("2026-09-29", "%Y-%m-%d"))  # 2026-W40

# Poison that must NEVER appear in redacted output.
POISON = [
    "alice@example.org",
    "bob@stemuli.com",
    "C:\\Users\\keppy\\git\\founder_tycoon",
    "keppy-laptop",
    "sk-abc123sessionkey",
    "gpt-5 secret note about the pricing bug",
    "/home/keppy/thomas/configs",
]


def _ledger(home: Path) -> list[dict]:
    """Synthetic ledger with real-looking junk across all six kinds."""
    task = (f"{POISON[4]} fix the NaN loss in our GRPO run; contact {POISON[0]}; "
            f"paths {POISON[1]} and {POISON[2]}")
    return [
        {"kind": "route", "id": "r1", "ts": BASE_TS, "task": task,
         "lane": "dl-ml-research-engineering", "model": "z-ai/glm-5.3",
         "effort": "high", "method": "rules-strong", "confidence": 0.9,
         "facets": ["long-doc", "domain-dlml"]},
        {"kind": "route", "id": "r2", "ts": BASE_TS + 10, "task": f"{POISON[5]} write a blog post",
         "lane": "routine-coding", "model": "z-ai/glm-5.3-flash", "effort": "medium",
         "method": "rules-weak", "confidence": 0.4},
        {"kind": "model_switch", "ts": BASE_TS + 20, "session_key": POISON[4],
         "new_model": POISON[2], "route_id": "r1", "prev_route_lane": "dl-ml-research-engineering"},
        {"kind": "effort_switch", "ts": BASE_TS + 30, "session_key": POISON[4],
         "new_effort": "high", "route_id": "r1"},
        {"kind": "lane_correction", "ts": BASE_TS + 40, "from_lane": "a", "to_lane": "b",
         "task": POISON[2]},
        {"kind": "rating_correction", "ts": BASE_TS + 45, "consumes_id": "r1",
         "rated": "fail", "note": POISON[5]},
        {"kind": "outcome", "ts": BASE_TS + 50, "rated": "pass",
         "route_lane": "dl-ml-research-engineering", "route_model": "z-ai/glm-5.3",
         "route_effort": "high", "actual_model": "openai/gpt-5", "actual_effort": "high",
         "arm_attribution": "explicit_user", "method": "rules-strong", "confidence": 0.9,
         "consumes": BASE_TS, "consumes_id": "r1", "note": POISON[5],
         "session_key": POISON[4], "facets": ["long-doc", "domain-dlml"]},
        {"kind": "outcome", "ts": BASE_TS + 60, "rated": "fail",
         "route_lane": "routine-coding", "route_model": "z-ai/glm-5.3-flash",
         "route_effort": "medium", "actual_model": None, "actual_effort": None,
         "arm_attribution": "unknown", "method": "rules-weak", "confidence": 0.4,
         "consumes": BASE_TS + 10, "consumes_id": "r2"},
        {"kind": "outcome", "ts": BASE_TS + 70, "rated": "skip",
         "route_lane": "routine-coding", "consumes": 999.0,
         "consumes_id": "gone", "note": POISON[5]},  # no route -> dropped_no_route
    ]


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    yield tmp_path


def _labels_file(home):
    f = home / "evalroute" / "labels.jsonl"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("\n".join(json.dumps(r) for r in _ledger(home)) + "\n", encoding="utf-8")
    return f


# ---------------------------------------------------------------- redaction


def test_redact_whitelist_keeps_only_KEPT_and_leaks_nothing(home):
    records = _ledger(home)
    rows = [r for r in contribute.redact(records, salt=b"\x01" * 32)
            if r["kind"] == "outcome"]
    assert rows, "outcome rows must survive"
    for row in rows:
        # schema 2: KEPT is the union whitelist across row kinds (outcomes do
        # not carry from_lane/to_lane); the exact per-kind keys are asserted
        # in tests/test_flywheel_schema2.py.
        assert set(row) <= set(contribute.KEPT), set(row) - set(contribute.KEPT)
        assert re.fullmatch(r"[0-9a-f]{64}", row["task_hash"])
        assert isinstance(row["facets"], dict)
        assert all(isinstance(v, int) for v in row["facets"].values())
    blob = json.dumps(rows)
    for poison in POISON:
        assert poison not in blob, f"leaked: {poison}"
    # only the outcome kinds survived; route rows (task text) are gone
    assert {r["kind"] for r in rows} == {"outcome"}


def test_task_hash_differs_across_salts(home):
    records = _ledger(home)
    outcomes = lambda rows: [r for r in rows if r["kind"] == "outcome"]
    h1 = outcomes(contribute.redact(records, b"\x01" * 32))[0]["task_hash"]
    h2 = outcomes(contribute.redact(records, b"\x02" * 32))[0]["task_hash"]
    assert h1 != h2


def test_rating_correction_overrides_rated_before_redaction(home):
    rows = [r for r in contribute.redact(_ledger(home), b"\x01" * 32)
            if r["kind"] == "outcome"]
    by_hash = {r["task_hash"]: r for r in rows}
    fixed = [r for r in rows if r["corrected"]]
    assert len(fixed) == 1
    assert fixed[0]["rated"] == "fail"  # the correction's verdict, not the outcome's pass


def test_no_route_rows_are_dropped_and_counted(home):
    records = _ledger(home)
    rows, summary = contribute.redact_report(records, b"\x01" * 32)
    outcome_rows = [r for r in rows if r["kind"] == "outcome"]
    assert summary["dropped_no_route"] == 1
    assert summary["n_rows"] == len(outcome_rows)


def test_summary_counts(home):
    rows, summary = contribute.redact_report(_ledger(home), b"\x01" * 32)
    assert summary["n_rows"] == 2
    assert summary["by_rated"] == {"fail": 2}  # pass overridden by rating_correction
    assert summary["by_lane"] == {"dl-ml-research-engineering": 1, "routine-coding": 1}
    assert summary["cursor"] == 0.0


def test_week_is_iso_year_week(home):
    rows = [r for r in contribute.redact(_ledger(home), b"\x01" * 32)
            if r["kind"] == "outcome"]
    assert all(r["week"] == "2026-W40" for r in rows)


def test_arm_attribution_derived(home):
    rows = [r for r in contribute.redact(_ledger(home), b"\x01" * 32)
            if r["kind"] == "outcome"]
    attrs = {r["arm_attribution"] for r in rows}
    assert attrs == {"explicit_user", "unknown"}


# ---------------------------------------------------------------- salt/cursor


def test_salt_created_on_dry_run_never_printed(home, capsys):
    _labels_file(home)
    assert contribute.run(dry_run=True) == 0
    out = capsys.readouterr().out
    sp = home / "evalroute" / "contribute.salt"
    assert sp.exists() and len(sp.read_text(encoding="ascii").strip()) == 64
    assert sp.read_text(encoding="ascii").strip() not in out
    assert "created" in out


def test_cursor_skips_uploaded_rows(home):
    records = _ledger(home)
    rows, summary = contribute.redact_report(records, b"\x01" * 32, cursor=BASE_TS + 55)
    assert summary["n_rows"] == 1  # only the ts+60 fail; the +70 skip row has no route
    assert rows[0]["rated"] == "fail"


def test_rotate_salt_resets_cursor(home):
    cp = home / "evalroute" / "contribute.cursor"
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_text("123.0\n")
    contribute.rotate_salt()
    assert contribute.read_cursor() == 0.0
    salt1 = (home / "evalroute" / "contribute.salt").read_text()
    contribute.rotate_salt()
    assert (home / "evalroute" / "contribute.salt").read_text() != salt1


# ---------------------------------------------------------------- gate


def test_gate_off_exits_2_and_touches_salt_only(home, capsys):
    _labels_file(home)
    before = {p.name for p in (home / "evalroute").iterdir()}
    assert contribute.run(dry_run=False) == 2
    out = capsys.readouterr().out
    assert "contribute is off" in out
    touched = {p.name for p in (home / "evalroute").iterdir()} - before
    assert touched <= {"contribute.salt"}, touched
    assert (home / "evalroute" / "contribute.cursor").exists() is False


def test_gate_env(home, monkeypatch, capsys):
    monkeypatch.setenv("EVALROUTE_CONTRIBUTE", "1")
    # gate on, but no network: _get_hfapi faked below in upload tests; here
    # just verify contribute_enabled flips
    assert contribute.contribute_enabled() is True


def test_gate_standalone_config_json(home):
    d = home / "evalroute"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text('{"contribute": true}')
    assert contribute.contribute_enabled() is True
    (d / "config.json").write_text('{"contribute": false}')
    assert contribute.contribute_enabled() is False


def test_gate_hermes_config_yaml(home, monkeypatch):
    # The console script has no hermes_constants on its path; the gate must
    # still find config.yaml in the resolved home (the bug the first live run hit).
    monkeypatch.setitem(sys.modules, "hermes_constants", None)
    (home / "config.yaml").write_text("evalroute:\n  contribute: true\n", encoding="utf-8")
    assert contribute.contribute_enabled() is True
    (home / "config.yaml").write_text("evalroute:\n  contribute: false\n", encoding="utf-8")
    assert contribute.contribute_enabled() is False


def test_gate_hermes_config_yaml_via_hermes_constants(home, monkeypatch):
    fake = types.ModuleType("hermes_constants")
    fake.get_hermes_home = lambda: str(home)
    monkeypatch.setitem(sys.modules, "hermes_constants", fake)
    monkeypatch.delenv("EVALROUTE_HOME", raising=False)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    (home / "config.yaml").write_text("evalroute:\n  contribute: true\n", encoding="utf-8")
    assert contribute.contribute_enabled() is True


# ---------------------------------------------------------------- dry-run / upload


class FakeHfApi:
    calls: list = []

    def whoami(self):
        return {"name": "contributor-1"}

    def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id, repo_type, commit_message,
                    token=None, revision=None):
        # Keyword-only with the real HfApi.upload_file parameter names: a fake
        # that took **kwargs let a misspelt argument reach production unseen.
        type(self).calls.append(dict(path_or_fileobj=path_or_fileobj, path_in_repo=path_in_repo,
                                     repo_id=repo_id, repo_type=repo_type,
                                     commit_message=commit_message))
        return "cafe" * 10


def test_dry_run_prints_exact_rows_json_mode(home, capsys):
    _labels_file(home)
    assert contribute.run(dry_run=True, as_json=True) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["summary"]["n_rows"] == sum(r["kind"] == "outcome" for r in data["rows"])
    outcome_rows = [r for r in data["rows"] if r["kind"] == "outcome"]
    assert set(outcome_rows[0]) <= set(contribute.KEPT)
    for row in data["rows"]:
        assert set(row) <= set(contribute.KEPT)
    assert data["summary"]["salt"] in ("present", "created")


def test_upload_matches_dry_run_bytes_advances_cursor(home, monkeypatch, capsys):
    _labels_file(home)
    assert contribute.run(dry_run=True, as_json=True) == 0
    payload_expected = json.loads(capsys.readouterr().out)["rows"]
    monkeypatch.setenv("EVALROUTE_CONTRIBUTE", "1")
    monkeypatch.setattr(contribute, "_get_hfapi", lambda: FakeHfApi())
    FakeHfApi.calls = []
    assert contribute.run() == 0
    out = capsys.readouterr().out
    assert "uploaded 3 rows" in out
    assert "contributor-1" in out and "cafe" * 10 in out
    call = FakeHfApi.calls[0]
    assert call["path_in_repo"].startswith("contributed/contributor-1/")
    assert call["path_in_repo"].endswith(".jsonl")
    sent = [json.loads(l) for l in call["path_or_fileobj"].decode("utf-8").splitlines()]
    assert sent == payload_expected  # byte-for-byte the dry-run rows
    assert contribute.read_cursor() == BASE_TS + 70
    # second upload sends nothing new
    FakeHfApi.calls = []
    assert contribute.run() == 0
    assert not FakeHfApi.calls


def test_upload_without_hub_hint(home, monkeypatch, capsys):
    monkeypatch.setenv("EVALROUTE_CONTRIBUTE", "1")

    def no_hub():
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *a, **k):
            if name == "huggingface_hub":
                raise ImportError("no hub")
            return real_import(name, *a, **k)
        builtins.__import__ = fake_import
        try:
            return contribute._get_hfapi()
        finally:
            builtins.__import__ = real_import

    assert no_hub() is None
    assert "pip install" in capsys.readouterr().err


# ---------------------------------------------------------------- aggregation


def _redacted_rows(home, n_weeks=2):
    rows = []
    for w in range(n_weeks):
        ts = BASE_TS + 7 * 86400 * w
        records = _ledger(home)
        for r in records:
            if r.get("ts") is not None:
                r["ts"] = ts + (r["ts"] - BASE_TS)
        rows.extend(r for r in contribute.redact(records, b"\x01" * 32)
                    if r["kind"] == "outcome")
    return rows


def test_aggregate_contributed_shape(home):
    rows = _redacted_rows(home)
    for i, r in enumerate(rows):
        r["_contributor"] = "a" if i % 2 == 0 else "b"
    stats = rfl.aggregate_contributed(rows)
    assert stats["n_contributors"] == 2
    explicit = [r for r in rows if r["rated"] in ("pass", "fail")
                and r["arm_attribution"] == "explicit_user"]
    unattributed = [r for r in rows if r["rated"] in ("pass", "fail")
                    and r["arm_attribution"] != "explicit_user"]
    # Only rows with an explicit actual arm count; the rest are dropped and counted,
    # never re-attributed to the routed arm.
    assert stats["outcomes"] == len(explicit)
    assert stats["dropped"]["no_explicit_arm"] == len(unattributed) > 0
    assert "dl-ml-research-engineering" in stats["lanes"]
    assert set(stats["lanes"]["dl-ml-research-engineering"]["arms"]) == {"openai/gpt-5@high"}


def test_aggregate_contributed_unknown_model_dropped_not_reattributed(home):
    rows = [{"kind": "outcome", "route_lane": "routine-coding", "route_model": "z-ai/glm-5.3-flash",
             "route_effort": "medium", "actual_model": "evil/made-up-model", "actual_effort": "max",
             "arm_attribution": "explicit_user", "method": "rules-strong", "confidence": 1.0,
             "rated": "pass", "week": "2026-W40", "task_hash": "0" * 64, "corrected": False,
             "schema": 1, "_contributor": "evil"}]
    stats = rfl.aggregate_contributed(rows, known_models={"z-ai/glm-5.3-flash"})
    assert stats["outcomes"] == 0
    assert stats["dropped"]["unknown_arm"] == 1
    assert stats["lanes"] == {}


def test_aggregate_contributed_stale_week_dropped(home):
    old = _redacted_rows(home, 1)
    for r in old:
        r["week"] = "2020-W01"
        r["_contributor"] = "a"
    stats = rfl.aggregate_contributed(old)
    assert stats["outcomes"] == 0
    assert stats["dropped"]["stale"] == len(old)


ROUTES = """\
version: 1
lanes:
- id: routine-coding
  label: Routine coding
  model: z-ai/glm-5.3-flash
  effort: medium
  provenance: measured 10 tasks, cov 1.0, all-in $0.0001/succ, 2026-09-27
  keywords: [routine]
- id: dl-ml-research-engineering
  label: DL/ML research
  model: z-ai/glm-5.3
  effort: high
  provenance: priors
  keywords: [dlml]
"""


def test_merge_invariant(home, tmp_path):
    """Adversarial contributed rows change nothing on measured lanes and
    never introduce an unknown model. This is the invariant as a test."""
    routes = tmp_path / "routes.yaml"
    routes.write_text(ROUTES, encoding="utf-8")
    rows = _redacted_rows(home)
    # adversary: 1000 passes on a made-up model, aimed at both lanes; plus a
    # fake lane the table has never had
    for i in range(1000):
        rows.append({"kind": "outcome", "route_lane": "dl-ml-research-engineering",
                     "route_model": "z-ai/glm-5.3", "route_effort": "high",
                     "actual_model": "evil/made-up-model", "actual_effort": "max",
                     "arm_attribution": "explicit_user", "method": "rules-strong",
                     "confidence": 1.0, "rated": "pass", "week": "2026-W40",
                     "task_hash": "0" * 64, "corrected": False, "schema": 1,
                     "_contributor": "evil"})
        rows.append({"kind": "outcome", "route_lane": "measured-victim",
                     "route_model": "evil/made-up-model", "route_effort": "max",
                     "actual_model": "evil/made-up-model", "actual_effort": "max",
                     "arm_attribution": "explicit_user", "method": "rules-strong",
                     "confidence": 1.0, "rated": "pass", "week": "2026-W40",
                     "task_hash": "0" * 64, "corrected": False, "schema": 1,
                     "_contributor": "evil"})
        rows.append({"kind": "outcome", "route_lane": "routine-coding",
                     "route_model": "evil/made-up-model", "route_effort": "max",
                     "actual_model": "evil/made-up-model", "actual_effort": "max",
                     "arm_attribution": "explicit_user", "method": "rules-strong",
                     "confidence": 1.0, "rated": "pass", "week": "2026-W40",
                     "task_hash": "0" * 64, "corrected": False, "schema": 1,
                     "_contributor": "evil"})
    lanes, applied = rfl.merge_contributed(routes, rows)
    by_id = {l["id"]: l for l in lanes}
    # measured lane byte-identical
    original = yaml.safe_load(ROUTES)["lanes"][0]
    assert by_id["routine-coding"] == original
    # no unknown model anywhere, no minted lane
    assert set(by_id) == {"routine-coding", "dl-ml-research-engineering"}
    for lane in lanes:
        assert lane.get("model") != "evil/made-up-model"
    # The priors lane's only honest rows ran gpt-5, not the lane's own glm-5.3@high
    # arm, so nothing may speak for the lane: it stays priors, untouched.
    assert by_id["dl-ml-research-engineering"] == yaml.safe_load(ROUTES)["lanes"][1]
    assert applied == 0


def test_merge_contributed_own_arm_only(home, tmp_path):
    """Pass rate on a contested lane counts only outcomes on the lane's own arm."""
    routes = tmp_path / "routes.yaml"
    routes.write_text(ROUTES, encoding="utf-8")
    def row(model, effort, rated, who):
        return {"kind": "outcome", "route_lane": "dl-ml-research-engineering",
                "route_model": "z-ai/glm-5.3", "route_effort": "high",
                "actual_model": model, "actual_effort": effort, "arm_attribution": "explicit_user",
                "method": "rules-strong", "confidence": 1.0, "rated": rated, "week": "2026-W40",
                "task_hash": "0" * 64, "corrected": False, "schema": 1, "_contributor": who}
    rows = [row("z-ai/glm-5.3", "high", "pass", "a"), row("z-ai/glm-5.3", "high", "fail", "b"),
            row("z-ai/glm-5.3-flash", "medium", "pass", "a"), row("z-ai/glm-5.3-flash", "medium", "pass", "b"),
            row("z-ai/glm-5.3-flash", "medium", "pass", "b")]
    lanes, applied = rfl.merge_contributed(routes, rows)
    lane = {l["id"]: l for l in lanes}["dl-ml-research-engineering"]
    assert applied == 1
    # 2 own-arm outcomes, 1 pass -> 50%; the three flash passes do not inflate it
    assert lane["provenance"].startswith("observed 2 tasks across 2 contributors, single-arm, pass 50%")
    assert lane["model"] == "z-ai/glm-5.3" and lane["effort"] == "high"


def stats_k(rows):
    return len({r.get("_contributor") for r in rows})


def test_merge_invariant_no_contributors_tag(home, tmp_path):
    routes = tmp_path / "routes.yaml"
    routes.write_text(ROUTES, encoding="utf-8")
    rows = [{"kind": "outcome", "route_lane": "dl-ml-research-engineering",
             "actual_model": "z-ai/glm-5.3", "actual_effort": "high",
             "arm_attribution": "explicit_user", "rated": "pass", "week": "2026-W40",
             "route_model": "z-ai/glm-5.3", "route_effort": "high", "schema": 1}]
    lanes, applied = rfl.merge_contributed(routes, rows)
    assert applied == 1
    lane = next(l for l in lanes if l["id"] == "dl-ml-research-engineering")
    assert "across 1 contributors" in lane["provenance"]


# ---------------------------------------------------------------- dataset side


def test_sync_with_contributed_downloads_contributed_tree(home, tmp_path, monkeypatch, capsys):
    from tests.test_dataset import _make_remote, _fake_hf, FIXED_SHA
    from evalroute import routing
    remote = _make_remote(tmp_path / "remote")
    tree = remote / "contributed" / "someone"
    tree.mkdir(parents=True)
    (tree / "20260101T000000Z.jsonl").write_text('{"kind": "outcome"}\n', encoding="utf-8")
    monkeypatch.setitem(sys.modules, "huggingface_hub", _fake_hf(remote))
    assert dataset.sync(with_contributed=True) == 0
    ds_dir = Path(routing._dataset_root()) / FIXED_SHA
    assert (ds_dir / "contributed" / "someone" / "20260101T000000Z.jsonl").is_file()
    routing.reset_routes_cache()


def test_dataset_contributed_rows_tag_contributor(home):
    from evalroute import routing
    sha = "f" * 40
    d = routing._dataset_root() / sha / "contributed" / "someone"
    d.mkdir(parents=True)
    (d / "20260101T000000Z.jsonl").write_text(
        '{"kind": "outcome", "rated": "pass"}\n\n', encoding="utf-8")
    (routing._dataset_root() / "current").write_text(sha + "\n", encoding="utf-8")
    rows = dataset.contributed_rows()
    assert rows == [{"kind": "outcome", "rated": "pass", "_contributor": "someone"}]
    assert dataset.contributed_rows(sha="0" * 40) == []  # unknown revision: empty, not error


# ---------------------------------------------------------------- publish


@pytest.fixture
def pd(tmp_path, monkeypatch):
    """publish_dataset importable, with a tiny fake REPO_ROOT (no git, no artifacts)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import publish_dataset
    root = tmp_path / "repo"
    (root / "evalroute" / "data").mkdir(parents=True)
    (root / "evalroute" / "data" / "routes.yaml").write_text(ROUTES, encoding="utf-8")
    monkeypatch.setattr(publish_dataset, "REPO_ROOT", root)
    monkeypatch.setattr(publish_dataset, "_plugin_commit", lambda: "0" * 40)
    yield publish_dataset
    sys.path.remove(str(Path(__file__).resolve().parents[1] / "scripts"))
    sys.modules.pop("publish_dataset", None)


def test_publish_dataset_includes_contributed(pd, tmp_path):
    tree = tmp_path / "contributed" / "someone"
    tree.mkdir(parents=True)
    (tree / "20260101T000000Z.jsonl").write_text('{"kind": "outcome"}\n', encoding="utf-8")
    staging = tmp_path / "staging"
    manifest = pd.build_staging(staging, tmp_path / "contributed")
    assert manifest["contributed_files"] == 1
    assert (staging / "contributed" / "someone" / "20260101T000000Z.jsonl").is_file()
    card = (staging / "README.md").read_text(encoding="utf-8")
    assert "- config_name: contributed" in card
    assert "config_name: measured" in card  # measured untouched by contributed presence


def test_publish_dataset_without_contributed(pd, tmp_path):
    staging = tmp_path / "staging2"
    manifest = pd.build_staging(staging, tmp_path / "absent")
    assert "contributed_files" not in manifest
    assert not (staging / "contributed").exists()
    assert "- config_name: contributed" not in (staging / "README.md").read_text(encoding="utf-8")


def test_contributed_dir_option(home, tmp_path, capsys):
    routes = tmp_path / "routes.yaml"
    routes.write_text(ROUTES, encoding="utf-8")
    rows = _redacted_rows(home, 1)
    tree = tmp_path / "contributed"
    for contrib in ("alice", "bob"):
        d = tree / contrib
        d.mkdir(parents=True)
        for r in rows:
            r = dict(r)
            r["_contributor"] = contrib
        (d / "20260101T000000Z.jsonl").write_text(
            "\n".join(json.dumps({k: v for k, v in r.items() if k != "_contributor"})
                      for r in rows), encoding="utf-8")
    rc = rfl.main(["--contributed", str(tree), "--routes", str(routes)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "across 2 contributors" in out
    # fixture rows ran gpt-5 on the priors lane -> nothing speaks for its own arm
    assert "0 pooled observed rows applied" in out
