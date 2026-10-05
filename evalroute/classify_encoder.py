"""Encoder-based classification (CONTRACT §4 artifact).

Rules stay first; the encoder replaces the LLM fallback when an artifact is
installed under <home>/evalroute/encoder/. torch/transformers are imported
only inside load(), never at module import (the plugin's cold start must not
pay for them).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .paths import hermes_home

logger = logging.getLogger(__name__)

stats: dict[str, int] = {"calls": 0, "unknown_lanes": 0, "abstains": 0}

_state: dict[str, Any] | None = None  # lazy cache {"label2id", "temperature", "metrics", ...}


def encoder_dir() -> Path:
    """<home>/evalroute/encoder/ (profile-safe; never inside the package)."""
    return hermes_home() / "evalroute" / "encoder"


def is_installed() -> bool:
    """Both required manifest files exist."""
    d = encoder_dir()
    return (d / "label2id.json").is_file() and (d / "config.json").is_file()


def load() -> dict[str, Any] | None:
    """Lazy, cached loader. Returns None when torch/transformers are missing."""
    global _state
    if not is_installed():
        return None
    if _state is not None:
        return _state
    try:
        import torch  # noqa: F401  (used via transformers' output below)
        import transformers
    except Exception as exc:
        logger.warning("evalroute encoder unavailable (torch/transformers missing): %s", exc)
        return None
    d = encoder_dir()
    try:
        label2id = json.loads((d / "label2id.json").read_text(encoding="utf-8"))
        metrics = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        # CONTRACT §4: temperature.json is the authority; metrics.json mirrors it.
        tpath = d / "temperature.json"
        temp = float(json.loads(tpath.read_text(encoding="utf-8"))["temperature"]
                     if tpath.is_file() else metrics.get("temperature", 1.0))
        if temp <= 0:
            raise ValueError(f"temperature must be > 0, got {temp}")
        tok = transformers.AutoTokenizer.from_pretrained(str(d))
        model = transformers.AutoModelForSequenceClassification.from_pretrained(str(d))
        model.eval()
    except Exception as exc:
        logger.warning("evalroute encoder load failed: %s", exc)
        return None
    _state = {
        "label2id": label2id,
        "temperature": temp,
        "metrics": metrics,
        "tokenizer": tok,
        "model": model,
    }
    return _state


def threshold() -> float:
    """Defer threshold from metrics.json, else 0.5."""
    s = _state or {}
    m = s.get("metrics") or {}
    try:
        return float(m.get("defer_below", 0.5))
    except Exception:
        return 0.5


def classify(task: str) -> tuple[str | None, float]:
    """Top (lane_id, prob) via softmax(logits / T).

    (None, 0.0) when the top label is not a lane id in the route table
    (artifact/table drift) — counted in stats["unknown_lanes"], never
    returned as a lane.
    """
    st = load()
    if st is None:
        return None, 0.0
    import torch

    from .routing import _lane_by_id

    stats["calls"] += 1
    tok, model, temp = st["tokenizer"], st["model"], st["temperature"]
    with torch.no_grad():
        enc = tok(task, truncation=True, max_length=256, return_tensors="pt")
        probs = torch.softmax(model(**enc).logits / temp, dim=-1)[0]
        top = int(probs.argmax().item())
        conf = float(probs[top].item())
    for label, lid in st["label2id"].items():
        if int(lid) == top:
            if _lane_by_id(label) is None:
                stats["unknown_lanes"] += 1
                return None, 0.0
            return label, conf
    stats["unknown_lanes"] += 1
    return None, 0.0
