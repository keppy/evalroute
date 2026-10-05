"""Publish the evalroute flywheel as an HF dataset (run by hand, one-time/re-runnable).

Builds a staging dir from the repo and uploads it as `keppy/evalroute-flywheel`
(dataset, public):

  measured/<lane>/{runs.jsonl,report.csv,tasks.jsonl}   Tier-A evidence, as-is
  measured/models/{tier-a-models.json,tier-a-rc-models.json}
  routes/routes.yaml                                    byte copy of data/routes.yaml
  routes/MANIFEST.json                                  what produced this table
  README.md                                             dataset card

Stdlib + huggingface_hub only; never imported by the plugin.

Usage:
  python scripts/publish_dataset.py --dry-run   # build staging, print tree + MANIFEST
  python scripts/publish_dataset.py             # create repo if absent, upload, print sha
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO_ROOT / "examples" / "artifacts"
LANES = ["alignment", "dl-ml", "routine-coding"]
ROOT_MODEL_FILES = ["tier-a-models.json", "tier-a-rc-models.json"]
DEFAULT_REPO_ID = "keppy/evalroute-flywheel"


def _plugin_version() -> str:
    try:
        from importlib.metadata import version
        return version("evalroute")
    except Exception:
        return "?"


def _plugin_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True,
        text=True).stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def build_staging(staging: Path, contributed_dir: Path | None = None) -> dict:
    """Materialize the dataset layout; returns the MANIFEST dict."""
    for lane in LANES:
        src = ARTIFACTS / f"tier-a-{lane}"
        for name in ("runs.jsonl", "report.csv", "tasks.jsonl"):
            _copy(src / name, staging / "measured" / lane / name)
    for name in ROOT_MODEL_FILES:
        _copy(ARTIFACTS / name, staging / "measured" / "models" / name)
    (staging / "routes").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPO_ROOT / "evalroute" / "data" / "routes.yaml",
                    staging / "routes" / "routes.yaml")

    # contributed/: opt-in, redacted outcome rows, curated by hand into
    # contributed/<contributor>/<ts>.jsonl in the repo (or --contributed DIR).
    contributed_files: list[str] = []
    if contributed_dir and contributed_dir.exists():
        for src in sorted(contributed_dir.rglob("*.jsonl")):
            rel = src.relative_to(contributed_dir)
            _copy(src, staging / "contributed" / rel)
            contributed_files.append((staging / "contributed" / rel).as_posix())

    measured_files = sorted(
        p.relative_to(staging).as_posix().replace("\\", "/")
        for p in staging.glob("measured/**/*.*"))
    manifest = {
        "plugin_version": _plugin_version(),
        "plugin_commit": _plugin_commit(),
        "routes_sha256": _sha256(staging / "routes" / "routes.yaml"),
        "measured_sha256": {rel: _sha256(staging / rel) for rel in measured_files},
        "generated": date.today().isoformat(),
    }
    if contributed_files:
        manifest["contributed_files"] = len(contributed_files)
    (staging / "routes" / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (staging / "README.md").write_text(_dataset_card(bool(contributed_files)), encoding="utf-8")
    return manifest


def _dataset_card(contributed: bool = False) -> str:
    """The dataset card: tells the README's story, in the README's order."""
    front = f"""\
---
license: mit
pretty_name: evalroute flywheel
configs:
- config_name: measured
  data_files:
  - split: train
    path:
    - measured/*/*.jsonl
    - measured/*/*.csv
    - measured/models/*.json
- config_name: routes
  data_files:
  - split: train
    path: routes/routes.yaml
{f'''- config_name: contributed
  data_files:
  - split: train
    path: contributed/*/*.jsonl
''' if contributed else ''}---
"""
    body = f"""\
# evalroute flywheel

Honest size: three lanes are **measured** (n=10 each; arms tied at p=1.0) and
six are **priors** from public benchmarks — not our runs. The pipe is real,
the table is small, your outcomes grow it.

Configs: **measured** — Tier-A harness artifacts (`runs.jsonl`, `report.csv`,
`tasks.jsonl`, models manifests); written only by harness runs, never accepts
contributed rows · **routes** — the `routes.yaml` that `evalroute sync` pins
(`MANIFEST.json` says what produced it) · **contributed** — redacted outcome
rows from opted-in installs under `contributed/<contributor>/`{'' if contributed else ' (empty in this revision)'}.

## What a contributed row carries — only this, whitelist-redacted

| key | what it is |
| --- | --- |
| `kind` | always `outcome` |
| `route_lane`, `route_model`, `route_effort` | the routed arm |
| `actual_model`, `actual_effort`, `arm_attribution` | the arm you actually ran (only when you confirmed it) |
| `method`, `confidence` | how the route was classified |
| `rated` | pass / fail / skip (rating corrections applied) |
| `facets` | counts only, e.g. `{{"long-doc": 1}}` |
| `week` | ISO year-week (`2026-W40`) — no timestamps |
| `task_hash` | HMAC-SHA256 of the task text under a per-install salt |
| `corrected`, `schema` | correction flag; schema version |

Never leaves: task text, notes, paths, hostnames, session keys, the salt, the token.

```bash
uv tool install evalroute                    # or: pip install evalroute
evalroute route --json "<one-line task>"     # -> lane, model, effort, route_id
# ... run the task on that arm, your own way ...
evalroute rate pass --route-id <id> --model <model> --effort <effort> --note "why" --json
evalroute report --json                      # what the ledger says so far
evalroute contribute --dry-run               # the exact rows that would go; grep, then drop the flag
```

Pooled rows are **observational**: they can contest a `priors` lane, never
overwrite a `measured` one.

- code + contract: https://github.com/keppy/evalroute (see its `AGENTS.md`)
- Hermes plugin: https://github.com/keppy/hermes-plugin-evalroute
"""
    return front + body


def _print_tree(staging: Path) -> None:
    for p in sorted(staging.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(staging).as_posix()}  ({p.stat().st_size} bytes)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    ap.add_argument("--contributed", type=Path,
                    help="A contributed/ tree to include (default: REPO_ROOT/contributed "
                         "when it exists); adds the contributed config and a "
                         "contributed_files MANIFEST count")
    ap.add_argument("--dry-run", action="store_true",
                    help="Build the staging dir, print the tree + MANIFEST, upload nothing")
    args = ap.parse_args()

    staging = Path(tempfile.mkdtemp(prefix="evalroute-flywheel-"))
    try:
        contributed_dir = getattr(args, "contributed", None) or (
            REPO_ROOT / "contributed" if (REPO_ROOT / "contributed").exists() else None)
        manifest = build_staging(staging, contributed_dir)
        print(f"staging: {staging}")
        _print_tree(staging)
        print(json.dumps(manifest, indent=2))
        if args.dry_run:
            return 0

        import huggingface_hub  # never imported by the plugin
        api = huggingface_hub.HfApi()
        api.create_repo(args.repo_id, repo_type="dataset", exist_ok=True, private=False)
        commit = api.upload_folder(
            repo_id=args.repo_id, repo_type="dataset", folder_path=staging,
            commit_message=f"routes {manifest['plugin_version']} @ {manifest['plugin_commit'][:7]}")
        url = f"https://huggingface.co/datasets/{args.repo_id}"
        print(f"uploaded to {url}")
        print(f"revision (pin this): {commit}")
        return 0
    finally:
        if not args.dry_run:
            shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
