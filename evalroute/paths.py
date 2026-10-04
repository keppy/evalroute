"""Hermes-home resolution.

The library must run with no Hermes installed: when ``hermes_constants`` is
importable its ``get_hermes_home`` wins; otherwise ``HERMES_HOME`` must win
(tests set it), falling back to ``~/.hermes``.
"""

from __future__ import annotations

import os
from pathlib import Path


def hermes_home() -> Path:
    try:
        from hermes_constants import get_hermes_home
        return Path(get_hermes_home())
    except Exception:
        return Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes"))
