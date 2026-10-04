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
        return Path.home() / ".hermes"
