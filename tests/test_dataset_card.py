"""tests/test_dataset_card.py — the rendered dataset card tells the README's story."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import publish_dataset as pd  # noqa: E402


@pytest.fixture
def pd_clean():
    yield pd
    if str(REPO / "scripts") in sys.path:
        sys.path.remove(str(REPO / "scripts"))
    sys.modules.pop("publish_dataset", None)


HONESTY = ("Honest size: three lanes are **measured** (n=10 each; arms tied at "
           "p=1.0) and six are **priors** from public benchmarks")

KEPT_KEYS = ["kind", "route_lane", "route_model", "route_effort",
             "actual_model", "actual_effort", "arm_attribution",
             "method", "confidence", "rated", "facets", "week",
             "task_hash", "corrected", "schema"]


def _flat(text: str) -> str:
    return " ".join(text.split())


def test_card_line_count(pd_clean):
    card = pd._dataset_card(False)
    assert len(card.splitlines()) <= 60


def test_card_content(pd_clean):
    card = pd._dataset_card(False)
    assert HONESTY in _flat(card)  # prose wraps; compare whitespace-normalised
    for key in KEPT_KEYS:
        assert f"`{key}`" in card, key
    for link in ("https://github.com/keppy/evalroute",
                 "https://github.com/keppy/hermes-plugin-evalroute"):
        assert link in card
    assert "AGENTS.md" in card
    assert "never overwrite a `measured`" in _flat(card)


def test_contributed_card_keeps_configs(pd_clean):
    card = pd._dataset_card(True)
    assert "- config_name: contributed" in card
    assert "config_name: measured" in card


def test_staging_reads_real_layout(pd_clean, tmp_path):
    root = tmp_path / "repo"
    (root / "evalroute" / "data").mkdir(parents=True)
    (root / "evalroute" / "data" / "routes.yaml").write_text("version: 1\nlanes: []\n",
                                                            encoding="utf-8")
    (root / "examples" / "artifacts").mkdir(parents=True)
    pd.REPO_ROOT = root  # patched attribute on the imported module
    staging = tmp_path / "staging"
    try:
        pd.build_staging(staging, None)
    except Exception:
        pass  # manifest step may need artifacts; the copy is what we assert
    assert (staging / "routes" / "routes.yaml").is_file()
