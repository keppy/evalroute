"""evalroute — model routing with a verified-success flywheel.

The library behind the Hermes plugin of the same name: a keyword-first
classifier that routes a task description to a (model, reasoning effort)
arm, a ledger that labels routes with pass/fail outcomes, and the Tier-A
harness that measures arms (cost per verified success) and feeds the
results back into the route table.

The Hermes plugin is a thin adapter: keppy/hermes-plugin-evalroute.
"""

from __future__ import annotations

try:
    from importlib.metadata import version as _version
    __version__ = _version("evalroute")
except Exception:  # pragma: no cover - running from a checkout, not installed
    __version__ = "0.8.1"
