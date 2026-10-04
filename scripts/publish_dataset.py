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
    manifest = yaml.safe_load((REPO_ROOT / "plugin.yaml").read_text(encoding="utf-8")) or {}
    return str(manifest.get("version", "?"))


def _plugin_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True,
        text=True).stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def build_staging(staging: Path) -> dict:
    """Materialize the dataset layout; returns the MANIFEST dict."""
    for lane in LANES:
        src = ARTIFACTS / f"tier-a-{lane}"
        for name in ("runs.jsonl", "report.csv", "tasks.jsonl"):
            _copy(src / name, staging / "measured" / lane / name)
    for name in ROOT_MODEL_FILES:
        _copy(ARTIFACTS / name, staging / "measured" / "models" / name)
    (staging / "routes").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPO_ROOT / "data" / "routes.yaml", staging / "routes" / "routes.yaml")

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
    (staging / "routes" / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (staging / "README.md").write_text(_dataset_card(), encoding="utf-8")
    return manifest


def _dataset_card() -> str:
    return """\
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
---

# evalroute flywheel

Evidence behind the [hermes-plugin-evalroute](https://github.com/keppy/hermes-plugin-evalroute)
route table.

- **measured** — Tier-A harness artifacts exactly as measured: per-lane
  `runs.jsonl` (full model responses), `report.csv`, `tasks.jsonl`, plus the
  models manifests under `measured/models/`. Written only by harness runs;
  never accepts contributed rows.
- **routes** — the generated `routes.yaml`, produced by `routes_from_report.py`
  from `measured` at the plugin commit named in `routes/MANIFEST.json`
  (`plugin_commit`; `routes_sha256` pins the table bytes).

Provenance labels used by the route table: `measured` (harness run, coverage
and $/success), `observed` (single-arm ledger stats, weaker by construction),
`priors` (no data, human estimate). A `contributed` config (redacted,
opt-in outcome rows) may be added later; by construction it can never become
`measured` — observational data can contest a priors lane, never overwrite a
measured one.
"""


def _print_tree(staging: Path) -> None:
    for p in sorted(staging.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(staging).as_posix()}  ({p.stat().st_size} bytes)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    ap.add_argument("--dry-run", action="store_true",
                    help="Build the staging dir, print the tree + MANIFEST, upload nothing")
    args = ap.parse_args()

    staging = Path(tempfile.mkdtemp(prefix="evalroute-flywheel-"))
    try:
        manifest = build_staging(staging)
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
