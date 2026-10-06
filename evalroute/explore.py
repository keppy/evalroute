"""Opt-in exploration: on a fraction of routes, run the lane's runner-up arm.

Off by default. The fraction comes from env ``EVALROUTE_EXPLORE`` > Hermes
``config.yaml`` (``evalroute: {explore: {fraction: x}}``) > ``config.json``
(``{"explore": {"fraction": x}}``) — the same precedence ``contribute_enabled``
uses. Runner-up arms live in the bundled ``data/explore.yaml`` (independent of
whether the active route table is bundled or a synced dataset); lanes without
an entry never explore.
"""

from __future__ import annotations

import importlib.resources
import json
import os
import random
from typing import Any, Optional

from .paths import hermes_home

_rng = random.random  # test seam


def explore_fraction() -> float:
    """The explore fraction, clamped to [0, 1]; missing/unparseable -> 0.0."""
    raw: Optional[str] = os.environ.get("EVALROUTE_EXPLORE")
    if raw is None:
        cfg = hermes_home() / "config.yaml"
        if cfg.exists():
            try:
                import yaml

                data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
                raw = str(((data.get("evalroute") or {}).get("explore") or {}).get("fraction", ""))
            except Exception:
                return 0.0
        else:
            local = hermes_home() / "evalroute" / "config.json"
            if local.exists():
                try:
                    data = json.loads(local.read_text(encoding="utf-8"))
                    raw = str(((data.get("explore") or {}).get("fraction", "")))
                except Exception:
                    return 0.0
    if raw is None or raw == "":
        return 0.0
    try:
        return max(0.0, min(1.0, float(raw)))
    except ValueError:
        return 0.0


def explore_arms() -> dict[str, dict[str, Any]]:
    """data/explore.yaml: lane id -> {model, effort, basis}; missing file -> {}."""
    ref = importlib.resources.files("evalroute").joinpath("data/explore.yaml")
    if not ref.is_file():
        return {}
    try:
        import yaml

        return ((yaml.safe_load(ref.read_text(encoding="utf-8")) or {}).get("explore_arms")) or {}
    except Exception:
        return {}


def pick(lane: dict[str, Any], no_explore: bool = False) -> dict[str, Any]:
    """Return the lane, or a copy swapped to the runner-up arm when exploring.

    Never mutates the cached lane from ``_load_routes()``. Conditions that
    suppress exploration: ``no_explore``, fraction 0 (the RNG is not even
    consulted), or no yaml entry for the lane id.
    """
    frac = explore_fraction()
    if no_explore or frac <= 0.0 or _rng() >= frac:
        return lane
    arm = explore_arms().get(lane["id"])
    if not arm:
        return lane
    out = dict(lane)
    out["explore_default"] = f"{lane['model']}@{lane['effort']}"
    out["explore_basis"] = str(arm.get("basis", ""))
    out["explore_fraction"] = frac
    out["model"] = arm["model"]
    out["effort"] = arm["effort"]
    return out
