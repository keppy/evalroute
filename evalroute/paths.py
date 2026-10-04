"""Hermes-home resolution.

The library must run with no Hermes installed: when ``hermes_constants`` is
importable its ``get_hermes_home`` wins; otherwise ``EVALROUTE_HOME`` wins
(the harness-neutral name — an agent with no Hermes installed sets this),
then ``HERMES_HOME`` (tests set it), falling back to ``~/.hermes``.
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
        return Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes"))
