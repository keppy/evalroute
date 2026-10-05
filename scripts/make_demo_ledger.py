"""Regenerate evalroute/data/demo/labels.jsonl (deterministic, seeded).

Routes spread over the last 10 days: ~6 in the last 24h, ~12 in days 1-3,
the rest older; each outcome keeps `consumes_offset` equal to its route's
offset and sits 10-90 minutes later; rating_correction rows come after
their outcome. Writes the file in place — run by hand, never in tests.
"""

from __future__ import annotations

import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

DEMO = Path(__file__).resolve().parents[1] / "evalroute" / "data" / "demo"
LABELS = DEMO / "labels.jsonl"


def main() -> int:
    rng = random.Random(20261004)
    recs = [json.loads(l) for l in LABELS.read_text(encoding="utf-8").splitlines() if l.strip()]

    routes = [r for r in recs if r.get("kind") == "route"]
    route_ids = {r["id"] for r in routes}

    # Route ages: ~6 within today (local calendar day), ~12 in days 1-3,
    # the rest in days 3-10. The recent ones are placed as a fraction of the
    # time elapsed since local midnight, so "routes today" stays 6 however
    # long after generation the fixture is loaded.
    ages: dict[str, float] = {}
    elapsed_today = time.time() - datetime.now().astimezone().replace(
        hour=0, minute=0, second=0, microsecond=0).timestamp()
    hour_offsets = [-elapsed_today * rng.uniform(0.1, 0.9) for _ in range(6)]
    early_offsets = sorted(rng.uniform(-72, -25) * 3600 for _ in range(12))
    late_offsets = sorted(rng.uniform(-240, -72) * 3600 for _ in range(len(routes) - 18))
    pool = hour_offsets + early_offsets + late_offsets
    rng.shuffle(pool)
    for rid, off in zip((r["id"] for r in routes), pool):
        ages[rid] = off

    # Every record referencing a route inherits that route's offset; free
    # records (none today) keep their own spread across the window.
    def spread(off: float) -> float:
        return rng.uniform(-240, -0.3) * 3600

    out = []
    for rec in recs:
        rec = dict(rec)
        rid = rec.get("consumes_id")
        if rid in route_ids:
            rec["consumes_offset"] = ages[rid]
            rec["ts_offset"] = ages[rid] + rng.uniform(10, 90) * 60
        elif rec.get("kind") == "route":
            rec["ts_offset"] = ages[rec["id"]]
        else:
            rec["ts_offset"] = spread(rec.get("ts_offset", 0.0))
        out.append(rec)

    # Rating corrections must come after their outcome: nudge them +2h on
    # top of whatever the outcome already has (they already were).
    for rec in out:
        if rec.get("kind") == "rating_correction":
            rec["ts_offset"] = max(rec["ts_offset"], rec["consumes_offset"] + 2 * 3600)

    # Sanity: routes never after their outcomes, corrections after both.
    by_id = {r["id"]: r["ts_offset"] for r in out if r.get("id")}
    for rec in out:
        if rec.get("consumes_id"):
            assert rec["consumes_offset"] == by_id[rec["consumes_id"]]
            assert rec["ts_offset"] >= by_id[rec["consumes_id"]]

    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    text = "\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n"
    LABELS.write_text(text, encoding="utf-8")
    print(f"wrote {len(out)} records to {LABELS}")
    recent24 = sum(1 for o in ages.values() if o > -86400)
    recent3 = sum(1 for o in ages.values() if -86400 * 3 < o <= -86400)
    print(f"routes: {len(ages)} (24h: {recent24}, days 1-3: {recent3})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
