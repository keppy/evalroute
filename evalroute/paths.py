"""Hermes-home resolution.

Precedence: ``EVALROUTE_HOME`` > ``HERMES_HOME`` > ``hermes_constants`` (its
``get_hermes_home``, when importable) > ``~/.hermes``. The env names let the
library run with no Hermes installed (``EVALROUTE_HOME`` — the harness-neutral
name) and keep tests isolated (``HERMES_HOME``).
"""

from __future__ import annotations

import os
from pathlib import Path


def hermes_home() -> Path:
    env_home = os.environ.get("EVALROUTE_HOME") or os.environ.get("HERMES_HOME")
    if env_home:
        return Path(env_home)
    try:
        from hermes_constants import get_hermes_home
        return Path(get_hermes_home())
    except Exception:
        pass
    # Hermes's own platform default: %LOCALAPPDATA%\hermes on Windows, ~/.hermes
    # elsewhere. Without this the standalone console script on Windows kept a
    # second ledger in ~/.hermes and never saw the plugin's routes.
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        win = Path(os.environ["LOCALAPPDATA"]) / "hermes"
        if win.exists() or not (Path.home() / ".hermes").exists():
            return win
    return Path.home() / ".hermes"
