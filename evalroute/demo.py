"""--demo mode: bundled synthetic fixture, never reads or writes real data.

Loads `evalroute/data/demo/labels.jsonl` (offsets resolved against now) and
the fixture `data/demo/trains/` tree. The state.db is always absent, so the
sessions half reads the cost from the demo sidecars instead.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

_DEMO_DIR = Path(__file__).resolve().parent / "data" / "demo"


def resolve_offsets(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Turn each record's `ts_offset` into absolute `ts`/`iso` (demo rows).

    `consumes_offset` likewise becomes `consumes`. Rows that already carry
    absolute `ts` pass through untouched.
    """
    now = time.time()
    out = []
    for rec in records:
        rec = dict(rec)
        if rec.get("ts_offset") is not None:
            ts = now + float(rec.pop("ts_offset"))
            rec["ts"] = ts
            rec["iso"] = datetime.fromtimestamp(
                ts, timezone.utc).isoformat(timespec="seconds")
        if rec.get("consumes_offset") is not None:
            rec["consumes"] = now + float(rec.pop("consumes_offset"))
        out.append(rec)
    return out


def load_demo_ledger() -> list[dict[str, Any]]:
    return resolve_offsets(
        [json.loads(l) for l in (_DEMO_DIR / "labels.jsonl")
         .read_text(encoding="utf-8").splitlines() if l.strip()])


def demo_trains_dir() -> Path:
    return _DEMO_DIR / "trains"
