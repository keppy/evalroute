"""Dataset sync: pin the published route table under the Hermes home.

`hermes evalroute sync` downloads ONLY the `routes/` config (and README.md)
of `keppy/evalroute-flywheel` — routing data, not the measured evidence —
into `<hermes home>/evalroute/dataset/<sha>/` and pins it via a `current`
file. The dataset is fetched only when the user runs `sync` (never on
install, never on route), it is pinned to a resolved commit sha, and the
plugin routes fully offline on the bundled table without it.

No network anywhere except inside `sync`; `--status`/`--clear` and all
routing paths never import huggingface_hub (it is imported lazily here).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from . import routing as tools
from .paths import hermes_home

REPO_ID = "keppy/evalroute-flywheel"
_INSTALL_HINT = ("evalroute sync needs the huggingface_hub package: "
                 "pip install huggingface_hub (or uv pip install into the Hermes venv)")


def _hermes_home() -> Path:
    return hermes_home()


def dataset_root() -> Path:
    return _hermes_home() / "evalroute" / "dataset"


def _current_sha() -> str | None:
    try:
        return tools._dataset_current_sha()
    except Exception:
        return None


def _active_label() -> str:
    """Bundled / dataset @ <sha12>, for the 'active table' output line."""
    if tools.ROUTES_SOURCE.get("kind") == "dataset":
        sha = str(tools.ROUTES_SOURCE.get("sha", ""))
        return f"dataset @ {sha[:12]}" if sha else "dataset"
    return "bundled"


def _load_hf():
    """Lazy huggingface_hub import; None (with a one-line hint) when absent."""
    try:
        import huggingface_hub  # noqa: F401
        return huggingface_hub
    except Exception:
        print(_INSTALL_HINT, file=sys.stderr)
        return None


def _validate_staged(dir_path: Path) -> dict[str, Any]:
    """Routes yaml parses + validates through the bundled loader's rules, and
    MANIFEST.json's routes_sha256 matches the file bytes. Returns the MANIFEST."""
    routes_path = dir_path / "routes" / "routes.yaml"
    import yaml
    raw = yaml.safe_load(routes_path.read_text(encoding="utf-8")) or {}
    lanes = tools._validate_lanes(raw)
    manifest = json.loads((dir_path / "routes" / "MANIFEST.json").read_text(encoding="utf-8"))
    import hashlib
    digest = hashlib.sha256(routes_path.read_bytes()).hexdigest()
    if manifest.get("routes_sha256") != digest:
        raise ValueError(
            f"MANIFEST routes_sha256 mismatch: expected {manifest.get('routes_sha256')}, "
            f"got {digest}")
    return {"manifest": manifest, "lanes": lanes}


def sync(revision: str | None = None) -> int:
    """Download and pin a route-table revision. Only network-touching path."""
    hf = _load_hf()
    if hf is None:
        return 1
    api = hf.HfApi()
    try:
        info = api.dataset_info(REPO_ID, revision=revision or "main")
        sha = info.sha
    except Exception as exc:
        print(f"evalroute: could not resolve {REPO_ID} @ {revision or 'main'}: {exc}")
        return 1
    target = dataset_root() / sha
    tools.reset_routes_cache()
    tools._load_routes()  # resolve the currently active table for the "was" line
    was = _active_label()
    try:
        hf.snapshot_download(repo_id=REPO_ID, repo_type="dataset", revision=sha,
                              allow_patterns=["routes/*", "README.md"],
                              local_dir=target)
        staged = _validate_staged(target)
    except Exception as exc:
        shutil.rmtree(target, ignore_errors=True)  # never leave a partial dir
        print(f"evalroute: refusing revision {sha[:12] if sha else revision}: {exc}")
        return 1
    manifest = staged["manifest"]
    (dataset_root() / "current").parent.mkdir(parents=True, exist_ok=True)
    (dataset_root() / "current").write_text(sha + "\n", encoding="utf-8")
    tools.reset_routes_cache()  # the table just changed; re-resolve on next route
    lanes = tools._load_routes()
    n_measured = sum(1 for l in lanes
                     if str(l.get("provenance", "")).startswith("measured"))
    print(f"synced {REPO_ID} @ {sha[:12]}")
    print(f"routes: {len(lanes)} lanes, {n_measured} measured "
          f"(generated {manifest.get('generated')}, "
          f"plugin {manifest.get('plugin_version')} @ "
          f"{str(manifest.get('plugin_commit', ''))[:7]})")
    print(f"active table: dataset (was: {was})")
    return 0


def status() -> int:
    """Which table is active; never touches the network or huggingface_hub."""
    sha = _current_sha()
    if sha:
        print(f"active table: dataset @ {sha[:12]}")
        print(f"path: {dataset_root() / sha / 'routes' / 'routes.yaml'}")
    else:
        print("active table: bundled")
        print(f"path: {tools.ROUTES_FILE}")
    return 0


def clear() -> int:
    """Unpin the dataset table; the bundled table routes again."""
    p = dataset_root() / "current"
    if p.exists():
        p.unlink()
    tools.reset_routes_cache()
    tools._load_routes()  # re-resolve against the bundled table
    print("active table: bundled")
    return 0


def run(args: Any) -> int:
    """argparse entry for `hermes evalroute sync`."""
    if getattr(args, "status", False):
        return status()
    if getattr(args, "clear", False):
        return clear()
    return sync(getattr(args, "revision", None))
