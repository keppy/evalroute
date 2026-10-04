"""Test setup.

The library is installed (editable) so ``import evalroute`` resolves; the
repo-root insert keeps the cross-file ``from tests.test_flywheel import home``
fixture re-exports working.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

@pytest.fixture(autouse=True)
def isolated_hermes_home(tmp_path, monkeypatch):
    """No test may read or write the user's live profile or labels."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from evalroute import flywheel
    flywheel._MEMORY.clear()
    from evalroute import routing
    routing.reset_routes_cache()  # no test inherits another test's table resolution
    yield
    flywheel._MEMORY.clear()
