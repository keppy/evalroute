"""routes_from_report.py — generate a measured routes.yaml from evalroute CSV.

The stable downstream interface is `evalroute.py report --csv` (see
evalroute/README.md "Integration notes"). This generator turns that CSV into
a routes.yaml whose rows carry `provenance: measured <n> tasks, <k> samples,
<date>` instead of the priors snapshot, so nothing measured masquerades as a
hypothesis or vice versa.

Mapping rules (documented in the plugin README):
- CSV rows are per (lane, model arm). Within each lane, the report's own
  routing rule already picked `route -> <model>`; we take that model and its
  effort column, with coverage+all-in stats as provenance.
- Keywords/match_hint/escalation are NOT in the CSV — they're classifier
  concerns, not measurement concerns. The generator PRESERVES them from the
  existing routes.yaml when the lane id matches, so classification quality
  survives regeneration.
- A lane with no measured data keeps its existing row untouched (and its
  provenance) — never silently delete a route the table already shipped.

Usage (inside the plugin repo):
  python routes_from_report.py --csv report.csv [--out data/routes.generated.yaml] \
      [--routes data/routes.yaml]
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

# Columns hermes evalroute report --csv writes (see evalroute.py report()):
# lane, model, effort, n, pend, trunc, err, complete, expected, pass_,
# lo, hi, cov, passk, try_usd, succ_usd, vmin, allin, known, med_out, p50s

FIELDNAMES = ["lane", "model", "effort", "n", "pend", "trunc", "err", "complete",
              "expected", "pass_", "lo", "hi", "cov", "passk", "try_usd",
              "succ_usd", "vmin", "allin", "known", "med_out", "p50s"]

# CSV lane names (evalroute task "lane" strings) -> routes.yaml lane ids.
# Keep this mapping the single spelling authority between harness and plugin.
LANE_ALIASES = {
    "routine coding": "routine-coding",
    "coding": "routine-coding",
    "hard agentic coding": "hard-agentic-coding",
    "dl/ml research engineering": "dl-ml-research-engineering",
    "dl / ml research engineering": "dl-ml-research-engineering",
    "long-doc reading": "long-doc-reading",
    "long-doc reading, lit review": "long-doc-reading",
    "web research": "web-research",
    "web research, long docs": "web-research",
    "math": "math-first-principles",
    "math, first principles": "math-first-principles",
    "math-abstraction": "math-first-principles",
    "alignment": "alignment-reasoning",
    "alignment reasoning, paper claims": "alignment-reasoning",
    "prose": "prose",
    "orchestration": "orchestration",
    "orchestrator / subagents": "orchestration",
}

KEEP_FROM_EXISTING = ("keywords", "match_hint", "escalation", "notes")


def _slug(lane: str) -> str:
    """CSV lane string -> routes.yaml lane id via aliases, else a slug."""
    low = (lane or "").strip().lower()
    if low in LANE_ALIASES:
        return LANE_ALIASES[low]
    return low.replace(" ", "-").replace("/", "-").replace(",", "").strip("-")


def _route_row(lane_id: str, stats: dict, date: str, model_id: str) -> dict:
    """One routes.yaml lane row from the winning CSV row for a lane."""
    return {
        "id": lane_id,
        "label": stats.get("label") or lane_id.replace("-", " ").title(),
        "model": model_id,
        "effort": stats["effort"],
        "provenance": (f"measured {stats['n_tasks']} tasks, cov {stats['cov']}, "
                       f"all-in ${float(stats['allin']):.4f}/succ, {date}"),
    }


def load_model_ids(models_path: Path | None) -> dict[str, str]:
    """Map harness arm names (name@effort or plain name) -> routable model ids.

    The models.json used by the run is the authority; without it, arm names
    pass through unchanged (flagged in provenance as unverified ids)."""
    if models_path is None:
        return {}
    if not models_path.exists():
        raise ValueError(f"models file not found: {models_path}")
    out: dict[str, str] = {}
    with models_path.open(encoding="utf-8") as f:
        models = json.load(f)
    if not isinstance(models, list):
        raise ValueError("models file must be a JSON list")
    for m in models:
        name, model = m.get("name"), m.get("model")
        if not name or not model or name in out:
            raise ValueError(f"missing/duplicate model name or id: {m}")
        _validate_model_id(model)
        out[name] = model
    return out


def _validate_model_id(model: str) -> None:
    if not isinstance(model, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]*", model):
        raise ValueError(f"invalid model id {model!r}; supply a routable /model id via --models")


def _routeable_row(row: dict) -> bool:
    """Exclude incomplete or unknown-cost arms from generated routes."""
    return (int(row.get("n", 0) or 0) > 0
            and str(row.get("complete", "True")).lower() == "true"
            and str(row.get("known", "True")).lower() == "true")


def load_report(csv_path: Path) -> dict[str, dict]:
    """Parse report CSV; per lane keep the row with the lowest all-in $/succ
    among models within `--tol` of the best coverage — the report's own
    routing rule, re-implemented here so the generated table matches what
    `report` prints as `route ->`."""
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    lanes: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        try:
            r["cov_f"] = float(r["cov"]) if r["cov"] not in ("", "nan") else -1.0
            r["allin_f"] = float(r["allin"]) if r["allin"] not in ("", "inf") else float("inf")
            r["n_i"] = int(r["n"] or 0)
        except (ValueError, KeyError):
            continue
        if _routeable_row(r):
            lanes[r["lane"]].append(r)

    winners: dict[str, dict] = {}
    for lane, lrows in lanes.items():
        best_cov = max((r["cov_f"] for r in lrows if r["cov_f"] >= 0), default=-1.0)
        elig = [r for r in lrows if r["cov_f"] >= best_cov - 1e-9 and r["allin_f"] < float("inf")]
        pick = min(elig, key=lambda r: r["allin_f"]) if elig else None
        if pick is not None:
            winners[lane] = pick
    return winners


def generate(csv_path: Path, out_path: Path, existing_path: Path | None,
             models_path: Path | None = None, k: int = 3,
             runs_path: Path | None = None) -> int:
    if k < 1:
        raise ValueError("samples per task (--k) must be positive")
    if runs_path is not None and not runs_path.exists():
        raise ValueError(f"runs file not found: {runs_path}")
    existing: dict[str, dict] = {}
    if existing_path and existing_path.exists():
        raw = yaml.safe_load(existing_path.read_text(encoding="utf-8")) or {}
        existing = {l["id"]: l for l in (raw.get("lanes") or []) if isinstance(l, dict) and l.get("id")}

    with csv_path.open(encoding="utf-8", newline="") as f:
        columns = csv.DictReader(f).fieldnames or []
    if not {"complete", "expected", "known"}.issubset(columns):
        raise ValueError("legacy report CSV lacks completeness evidence; regenerate it "
                         "with the current harness from the original runs")
    model_ids = load_model_ids(models_path)
    winners = load_report(csv_path)
    if not winners:
        print("no routeable rows in CSV (all lanes failed or no data); nothing generated")
        return 1

    # gonogo adjudication: winner vs runner-up per lane, from the raw runs.
    stamps: dict[str, str] = {}
    runs: list[dict] = []
    if runs_path:
        import json
        from . import adjudicate
        runs = [json.loads(l) for l in runs_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        all_rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
        by_lane: dict[str, list[dict]] = defaultdict(list)
        for r in all_rows:
            try:
                r["cov_f"] = float(r["cov"]) if r["cov"] not in ("", "nan") else -1.0
                r["allin_f"] = float(r["allin"]) if r["allin"] not in ("", "inf") else float("inf")
            except (ValueError, KeyError):
                continue
            if _routeable_row(r):
                by_lane[r["lane"]].append(r)
        for lane_name, lrows in by_lane.items():
            lane_id = _slug(lane_name)
            if lane_name not in winners:  # winners keyed by CSV lane name
                continue
            best_cov = max((r["cov_f"] for r in lrows if r["cov_f"] >= 0), default=-1.0)
            elig = [r for r in lrows if r["cov_f"] >= best_cov - 1e-9 and r["allin_f"] < float("inf")]
            if len(elig) < 2:
                continue
            runner = sorted(elig, key=lambda r: r["allin_f"])[1]
            if runner["model"] != winners[lane_name]["model"]:
                stamps[lane_id] = adjudicate.route_stamp(runs, winners[lane_name]["model"],
                                                        runner["model"])

    date = datetime.date.today().isoformat()
    lanes_out: list[dict] = []
    seen_ids: set[str] = set()

    # Measured lanes first, in stable order.
    for lane_name in sorted(winners):
        stats = winners[lane_name]
        stats["k"] = k
        if stats["n_i"] <= 0 or stats["n_i"] % k:
            raise ValueError(f"{lane_name}: sample count n={stats['n_i']} is not a positive multiple "
                             f"of --k {k}; cannot claim an exact task count")
        stats["n_tasks"] = stats["n_i"] // k
        if runs_path:
            # CSV n counts graded samples. Confirm the raw winner has exactly
            # k distinct graded sample IDs for every actual task, not merely
            # an aggregate count divisible by k.
            def _run_key(r):
                base = f'{r["task"]}|{r["model"]}|{r["sample"]}|{r.get("cfg", "")}'
                return base + (f'|{r["sig"]}' if r.get("sig") else "")
            grades = {r["grade_of"] for r in runs if "grade_of" in r}
            arm_name = stats["model"].split("#", 1)[0]
            cfg_prefix = stats["model"].split("#", 1)[1] if "#" in stats["model"] else None
            per_task: dict[str, set[int]] = defaultdict(set)
            matching_cfgs: set[str] = set()
            recorded_model_ids: set[str] = set()
            for r in runs:
                if ("text" not in r or r.get("lane") != lane_name or r.get("model") != arm_name
                        or (cfg_prefix is not None and not r.get("cfg", "").startswith(cfg_prefix))):
                    continue
                matching_cfgs.add(r.get("cfg", ""))
                if r.get("model_id"):
                    recorded_model_ids.add(r["model_id"])
                if r.get("passed") is not None or _run_key(r) in grades:
                    per_task[r["task"]].add(r["sample"])
            if cfg_prefix and len(matching_cfgs) != 1:
                raise ValueError(f"{lane_name}: cfg prefix {cfg_prefix!r} matches {len(matching_cfgs)} configurations")
            if (len(per_task) != stats["n_tasks"] or
                    any(samples != set(range(k)) for samples in per_task.values())):
                raise ValueError(f"{lane_name}: winner {stats['model']} has {stats['n_i']} graded "
                                 "samples in CSV but runs do not show exactly --k per task")
        lane_id = _slug(lane_name)
        if lane_id in seen_ids:
            continue
        seen_ids.add(lane_id)
        base_arm = stats["model"].split("#", 1)[0]
        if models_path and base_arm not in model_ids:
            raise ValueError(f"{lane_name}: no model id for winning arm {base_arm!r} in {models_path}")
        model_id = model_ids.get(base_arm, base_arm)
        _validate_model_id(model_id)
        if runs_path and recorded_model_ids and recorded_model_ids != {model_id}:
            raise ValueError(f"{lane_name}: --models id {model_id!r} disagrees with winner's recorded "
                             f"model id(s) {sorted(recorded_model_ids)}")
        row = _route_row(lane_id, stats, date, model_id)
        if models_path is None:
            row["provenance"] += "; model id unverified (no --models mapping)"
        elif runs_path and not recorded_model_ids:
            row["provenance"] += "; v1 runs omit model_id (mapped via models.json)"
        if lane_id in stamps:
            row["provenance"] = f'{row["provenance"]}; {stamps[lane_id]}'
        prev = existing.get(lane_id) or {}
        for key in KEEP_FROM_EXISTING:
            if prev.get(key):
                row[key] = prev[key]
        # A measured row must not carry stale priors-era caveats that
        # contradict it ("unmeasured", "verify with your own harness").
        if str(row.get("notes", "")).lower().lstrip().startswith(("unmeasured", "contested")):
            row.pop("notes", None)
        if prev.get("label"):
            row["label"] = prev["label"]
        lanes_out.append(row)

    # Unmeasured lanes keep their existing rows verbatim (with priors
    # provenance) so regeneration never deletes a route silently.
    for lane_id, prev in existing.items():
        if lane_id not in seen_ids:
            lanes_out.append(dict(prev))
            seen_ids.add(lane_id)

    doc = {
        "version": 1,
        "lanes": lanes_out,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    measured = sum(1 for l in lanes_out if str(l.get("provenance", "")).startswith("measured"))
    print(f"wrote {out_path}: {measured} measured lanes, {len(lanes_out) - measured} carried over")
    return 0


def main(argv=None) -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Generate measured routes.yaml from evalroute report CSV")
    ap.add_argument("--csv", required=True, help="path to `evalroute.py report --csv` output")
    ap.add_argument("--out", default=str(here / "data" / "routes.generated.yaml"))
    ap.add_argument("--routes", default=str(here / "data" / "routes.yaml"),
                    help="existing routes.yaml whose classifier fields/unmeasured rows are preserved")
    ap.add_argument("--models", default=None,
                    help="models.json used by the run; maps arm names -> routable model ids")
    ap.add_argument("--k", type=int, default=3,
                    help="samples per task in the run; the CSV's n column counts samples, not tasks")
    ap.add_argument("--runs", default=None,
                    help="the run's runs.jsonl; enables gonogo adjudication of the "
                         "winner vs runner-up (stamp appended to provenance)")
    a = ap.parse_args(argv)
    return generate(Path(a.csv), Path(a.out), Path(a.routes) if a.routes else None,
                     Path(a.models) if a.models else None, a.k,
                     Path(a.runs) if a.runs else None)


if __name__ == "__main__":
    sys.exit(main())
