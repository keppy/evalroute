"""Encoder classification method, install-encoder, and the no-extra fallback."""

from __future__ import annotations

import json
import os
import sys

import pytest

pytestmark = pytest.mark.usefixtures("isolated_hermes_home") if False else []  # noqa


def _install(src, monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("EVALROUTE_HOME", str(home))
    from evalroute import routing
    routing.reset_routes_cache()
    from evalroute.cli import install_encoder
    rc = install_encoder(dir_arg=str(src))
    assert rc == 0
    return home


def test_fallback_unchanged_without_extra(tmp_path, monkeypatch):
    """No artifact: route_full matches the pre-encoder decision path."""
    monkeypatch.setenv("EVALROUTE_HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    from evalroute import classify_encoder, routing
    routing.reset_routes_cache()
    monkeypatch.setattr(classify_encoder, "is_installed", lambda: False)
    task = "manage life, writing, and researchy tasks"
    lane, conf, hits, method, facets = routing.route_full(task)
    assert method in ("llm", "rules-weak", "default")
    assert 0.0 <= conf <= 1.0
    # And importing routing/cli never pulled torch in:
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules


def _torch_available():
    import importlib.util
    return all(importlib.util.find_spec(m) is not None
               for m in ("torch", "transformers"))


@pytest.mark.skipif(not _torch_available(), reason="encoder extra not installed")
class TestWithArtifact:
    def test_route_full_uses_encoder(self, tiny_encoder, tmp_path, monkeypatch):
        _install(tiny_encoder, monkeypatch, tmp_path)
        from evalroute import routing
        lane, conf, hits, method, facets = routing.route_full(
            "write a weekly status update email")
        assert method == "encoder"
        assert lane is not None
        assert 0.0 < conf <= 1.0
        assert lane["id"] in ("routine-coding", "long-doc-reading", "math-first-principles")

    def test_card_encoder_line(self, tiny_encoder, tmp_path, monkeypatch):
        _install(tiny_encoder, monkeypatch, tmp_path)
        from evalroute import routing
        lane, conf, hits, method, facets = routing.route_full("write something")
        card = routing.route_card(lane, conf, hits, method="encoder", facets=facets)
        assert "classification: encoder (opt-in) 0.90 calib · conf" in card

    def test_threshold_defer_to_llm(self, tiny_encoder, tmp_path, monkeypatch):
        home = _install(tiny_encoder, monkeypatch, tmp_path)
        m = json.loads((tiny_encoder / "metrics.json").read_text(encoding="utf-8"))
        m["defer_below"] = 1.1
        (tiny_encoder / "metrics.json").write_text(json.dumps(m), encoding="utf-8")
        from evalroute.cli import install_encoder
        rc = install_encoder(dir_arg=str(tiny_encoder))
        assert rc == 0
        from evalroute import routing
        routing.reset_routes_cache()
        from evalroute import classify_encoder
        classify_encoder._state = None  # re-read metrics with the new threshold
        lane, conf, hits, method, facets = routing.route_full("write something")
        assert method != "encoder"
        from evalroute import routing as r
        card = r.route_card(lane, conf, hits, method=method, facets=facets)
        assert "(encoder abstained)" in card

    def test_unknown_label_falls_through(self, tiny_encoder, tmp_path, monkeypatch):
        _install(tiny_encoder, monkeypatch, tmp_path)
        import evalroute.classify_encoder as ce
        ce._state = None
        stats = ce.stats
        stats["unknown_lanes"] = 0
        monkeypatch.setattr(ce, "stats", stats)
        (ce.encoder_dir() / "label2id.json").write_text(
            json.dumps({"not-a-lane": 0, "routine-coding": 1,
                        "long-doc-reading": 2}), encoding="utf-8")
        ce.load()  # populate the real cache (tokenizer/model) once
        fake = dict(ce._state or {})
        fake["label2id"] = {"not-a-lane": 0, "unmapped-a": 1, "unmapped-b": 2}
        monkeypatch.setattr(ce, "load", lambda: fake)
        from evalroute import routing
        lane_id, conf = ce.classify("write something")
        assert (lane_id, conf) == (None, 0.0)
        assert stats["unknown_lanes"] >= 1
        _lane, _c, _h, method, _f = routing.route_full("write something")
        assert method != "encoder"

    def test_rules_strong_skips_encoder(self, tiny_encoder, tmp_path, monkeypatch):
        _install(tiny_encoder, monkeypatch, tmp_path)
        from evalroute import classify_encoder, routing
        called = {"n": 0}
        orig = classify_encoder.classify
        def spy(task):
            called["n"] += 1
            return orig(task)
        monkeypatch.setattr(classify_encoder, "classify", spy)
        routing.route_full("refactor the helper and add a unit test for the bugfix")
        assert called["n"] == 0

    def test_torch_not_imported_on_module_import(self):
        # Must run in a fresh interpreter: popping an already-loaded torch from
        # sys.modules and re-importing it breaks torch's C registrations
        # ("Only a single TORCH_LIBRARY can be used ...") for the rest of the run.
        import subprocess
        code = ("import sys, evalroute.routing, evalroute.cli; "
                "sys.exit(1 if ('torch' in sys.modules or 'transformers' in sys.modules) else 0)")
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                              env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"})
        assert proc.returncode == 0, proc.stderr[-500:]


@pytest.mark.skipif(not _torch_available(), reason="encoder extra not installed")
class TestInstallEncoder:
    def test_missing_temperature_fails_copies_nothing(self, tiny_encoder, tmp_path, monkeypatch):
        (tiny_encoder / "temperature.json").unlink()
        home = tmp_path / "home"
        monkeypatch.setenv("EVALROUTE_HOME", str(home))
        from evalroute.cli import install_encoder
        assert install_encoder(dir_arg=str(tiny_encoder)) == 2
        assert not (home / "evalroute" / "encoder").exists()

    def test_remove_and_json(self, tiny_encoder, tmp_path, monkeypatch):
        home = _install(tiny_encoder, monkeypatch, tmp_path)
        assert (home / "evalroute" / "encoder" / "label2id.json").is_file()
        from evalroute.cli import install_encoder
        rc = install_encoder(dir_arg=None, remove=True, as_json=True)
        assert rc == 0
        assert not (home / "evalroute" / "encoder").exists()

    def test_json_reports_metrics_and_unknown_labels(self, tiny_encoder, tmp_path,
                                                     monkeypatch, capsys):
        home = tmp_path / "home"
        monkeypatch.setenv("EVALROUTE_HOME", str(home))
        m = json.loads((tiny_encoder / "metrics.json").read_text(encoding="utf-8"))
        m["defer_below"] = 0.0
        (tiny_encoder / "metrics.json").write_text(json.dumps(m), encoding="utf-8")
        from evalroute.cli import install_encoder
        assert install_encoder(dir_arg=str(tiny_encoder), as_json=True) == 0
        out = json.loads(capsys.readouterr().out)
        assert out["installed"] is True
        assert out["calib_accuracy"] == 0.9
        assert out["num_labels"] == 3
        assert out["unknown_labels"] == []


def test_hf_id_detection():
    from evalroute.cli import _looks_like_hf_id
    assert _looks_like_hf_id("keppy/evalroute-lane-encoder")
    assert not _looks_like_hf_id("C:/Users/x/enc")
    assert not _looks_like_hf_id("./enc")
    assert not _looks_like_hf_id("a/b/c")
    assert not _looks_like_hf_id(None)
