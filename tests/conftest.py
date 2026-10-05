"""Test setup.

The library is installed (editable) so ``import evalroute`` resolves; the
repo-root insert keeps the cross-file ``from tests.test_flywheel import home``
fixture re-exports working.
"""

from __future__ import annotations

import sys
from pathlib import Path
import json
import pytest

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

@pytest.fixture(autouse=True)
def isolated_hermes_home(tmp_path, monkeypatch):
    """No test may read or write the user's live profile or labels."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("EVALROUTE_HOME", raising=False)  # first-priority; must not leak in
    from evalroute import flywheel
    flywheel._MEMORY.clear()
    from evalroute import routing
    routing.reset_routes_cache()  # no test inherits another test's table resolution
    routing.reset_surface()       # nor another test's command-string surface
    yield
    flywheel._MEMORY.clear()


@pytest.fixture
def tiny_encoder(tmp_path):
    """CONTRACT §4 artifact from a random-init tiny BERT (no network).

    Skips nothing itself: tests that need torch/transformers use
    pytest.importorskip first.
    """
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("torch")
    from transformers import BertConfig, BertTokenizerFast

    enc = tmp_path / "enc"
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join(f"tok{i}" for i in range(10)), encoding="utf-8")
    cfg = BertConfig(vocab_size=100, hidden_size=16, num_hidden_layers=1,
                     num_attention_heads=1, intermediate_size=32, num_labels=3)
    tok = BertTokenizerFast(vocab_file=str(vocab))
    model = transformers.AutoModelForSequenceClassification.from_config(cfg)
    model.save_pretrained(str(enc))
    tok.save_pretrained(str(enc))
    labels = ["routine-coding", "long-doc-reading", "math-first-principles"]
    (enc / "label2id.json").write_text(
        json.dumps({lab: i for i, lab in enumerate(labels)}), encoding="utf-8")
    (enc / "temperature.json").write_text(json.dumps({"temperature": 1.0}),
                                          encoding="utf-8")
    (enc / "metrics.json").write_text(json.dumps({
        "contract_version": "1.0", "temperature": 1.0, "num_labels": 3,
        "calib_accuracy": 0.9, "ece_after": 0.02, "defer_below": 0.0,
    }), encoding="utf-8")
    return enc


@pytest.fixture
def hermes_chat():
    """hermes-chat surface: card command strings keep the /-slash form."""
    from evalroute import routing
    routing.set_surface("hermes-chat")
    yield
    routing.reset_surface()


@pytest.fixture
def hermes_cli():
    """hermes-cli surface: card command strings as `hermes evalroute ...`."""
    from evalroute import routing
    routing.set_surface("hermes-cli")
    yield
    routing.reset_surface()
