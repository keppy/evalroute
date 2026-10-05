"""evalroute export-cases: human-asserted lane labels as thomas encoder cases.

Contract (thomas docs/CONTRACT.md §1): JSONL rows
``{"id": <str>, "text": <str>, "label": <lane id>}``, UTF-8, LF.

Label sources, in order of assert strength:
  correction  a lane_correction row AND the pinned route that replaced it
              (written twice: the original id and ``<id>#2`` — the ×2 weight)
  ledger      every other ``method == "pinned"`` route
  taskset     bundled Tier-A taskset rows (repo examples/, never the wheel;
              supplied explicitly via --tasksets, no default path)
  seed        per lane, the route table's match_hint and joined keywords

Non-pinned routes are never labels; they are counted as ``unlabeled``. Exact
duplicate text is deduped, first occurrence wins, a correction's duplicate is
exempt. Local file only; stdout carries counts, never task text.
"""
from __future__ import annotations

import glob as _glob
import json
from pathlib import Path
from typing import Any

from . import flywheel
from .routes_from_report import LANE_ALIASES
from .routing import _load_routes

SOURCES = ("ledger", "correction", "taskset", "seed")


def _load_tasksets(spec: str) -> list[dict[str, str]]:
    """Taskset rows from a dir (its */tasks.jsonl) or a glob of files."""
    p = Path(spec)
    if p.is_dir():
        # a dir of tier-a-* sets, or a single taskset dir — both accepted
        files = sorted(set(p.glob("tasks.jsonl")) | set(p.glob("*/tasks.jsonl")))
    else:
        files = sorted(Path(x) for x in _glob.glob(spec) if Path(x).is_file())
    rows: list[dict[str, str]] = []
    for tf in files:
        for line in tf.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("lane") and rec.get("prompt"):
                # Taskset lane strings are harness spellings ("routine coding");
                # the label must be the routes.yaml id. One spelling authority.
                lane = LANE_ALIASES.get(rec["lane"], rec["lane"])
                rows.append({"id": "taskset:" + rec["id"],
                             "text": rec["prompt"], "label": lane})
    return rows


def _correction_lane(task: str, corrections: list[dict[str, Any]]) -> str | None:
    """to_lane of the lane_correction whose task prefix matches, else None."""
    for c in corrections:
        if task.startswith(c["task"]) and c.get("to_lane"):
            return c["to_lane"]
    return None


def _seed_rows() -> list[dict[str, str]]:
    """Per lane: match_hint row + joined-keywords row from the route table."""
    rows: list[dict[str, str]] = []
    for lane in _load_routes():
        lid = lane["id"]
        hint = lane.get("match_hint") or ""
        if hint:
            rows.append({"id": f"seed:{lid}:hint", "text": hint, "label": lid})
        kws = lane.get("keywords") or []
        if kws:
            rows.append({"id": f"seed:{lid}:keywords",
                         "text": " ".join(kws), "label": lid})
    return rows


def collect_cases(tasksets: str = "", seed_text: bool = False,
                  labels: list[dict[str, Any]] | None = None) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Gather encoder-case rows from every qualifying source.

    Returns (rows, stats) where stats carries the per-source counts,
    ``unlabeled``, and the dedupe hits. Never prints.
    """
    if labels is None:
        labels = flywheel.read_labels()

    corrections = [r for r in labels if r.get("kind") == "lane_correction"
                   and r.get("to_lane") and r.get("task")]

    ledger_rows: list[dict[str, str]] = []
    correction_rows: list[dict[str, str]] = []
    unlabeled = 0
    for r in labels:
        if r.get("kind") != "route":
            continue
        if r.get("method") != "pinned":
            unlabeled += 1
            continue
        row = {"id": r["id"], "text": r["task"], "label": r["lane"]}
        if r["lane"] == _correction_lane(r["task"], corrections):
            correction_rows.append(row)
            continue
        ledger_rows.append(row)
    # ×2 weight: each correction row again under <id>#2 (copy the list —
    # the original code appended into the list it was iterating: runaway)
    for row in list(correction_rows):
        correction_rows.append({"id": row["id"] + "#2",
                                "text": row["text"], "label": row["label"],
                                "_source": "correction"})

    taskset_rows = _load_tasksets(tasksets) if tasksets else []
    seed = _seed_rows() if seed_text else []

    # dedupe on exact text, first occurrence wins, correction duplicates exempt
    seen: set[str] = set()
    deduped_sources = 0
    ordered: list[tuple[str, dict[str, str]]] = (
        [("correction", r) for r in correction_rows]
        + [("ledger", r) for r in ledger_rows]
        + [("taskset", r) for r in taskset_rows]
        + [("seed", r) for r in seed])
    out: list[dict[str, str]] = []
    counts = {s: 0 for s in SOURCES}
    for source, row in ordered:
        row = dict(row)
        row["_source"] = source
        if source != "correction" and row["text"] in seen:
            deduped_sources += 1
            continue
        seen.add(row["text"])
        out.append(row)
        counts[source] += 1

    stats: dict[str, Any] = {
        "sources": counts,
        "unlabeled": unlabeled,
        "dedupe_removed": deduped_sources,
        "total": len(out),
    }
    return out, stats


def _summary(rows: list[dict[str, str]], stats: dict[str, Any],
             min_per_lane: int) -> dict[str, Any]:
    """Per-lane counts by source + below_min findings. ``rows`` rows carry
    the private ``_source`` key set by collect_cases."""
    per_lane: dict[str, dict[str, int]] = {}
    for row in rows:
        entry = per_lane.setdefault(row["label"], {s: 0 for s in SOURCES})
        entry[row["_source"]] += 1
    lanes = sorted(set(r["label"] for r in rows)
                   | set(l["id"] for l in _load_routes()))
    below = [{"lane": lane, "count": sum(per_lane.get(lane, {}).values())}
             for lane in lanes if sum(per_lane.get(lane, {}).values()) < min_per_lane]
    stats["per_lane"] = per_lane
    stats["below_min"] = below
    return stats


def write_cases(rows: list[dict[str, str]], out: Path) -> Path:
    """Write the rows as JSONL (UTF-8, LF). Local only; no network, no print."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            pub = {k: v for k, v in row.items() if not k.startswith("_")}
            f.write(json.dumps(pub, ensure_ascii=False) + "\n")
    return out


def run_export(out: Path, tasksets: str = "", seed_text: bool = False,
               min_per_lane: int = 20, strict: bool = False,
               as_json: bool = False) -> int:
    rows, stats = collect_cases(tasksets=tasksets, seed_text=seed_text)
    stats = _summary(rows, stats, min_per_lane)
    write_cases(rows, out)
    stats["out"] = str(out)
    if as_json:
        print(json.dumps(stats))
    else:
        print(f"export-cases: {stats['total']} cases -> {out}")
        for lane in stats["per_lane"]:
            entry = stats["per_lane"][lane]
            parts = " ".join(f"{s}={entry[s]}" for s in SOURCES)
            print(f"  {lane}: {parts}")
        if stats["unlabeled"]:
            print(f"  unlabeled (non-pinned routes, not exported): {stats['unlabeled']}")
        for b in stats["below_min"]:
            print(f"  below min {min_per_lane}: {b['lane']} ({b['count']})")
    return 3 if (strict and stats["below_min"]) else 0
