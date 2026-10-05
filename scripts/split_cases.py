"""Split an augmented cases JSONL into train/eval files plus an audit file.

Real rows (no `aug:`/`seed:` id prefix) from lanes with >= --min-real real rows
are eval candidates, stratified per lane. A correction pair (`X` and `X#2`)
is one unit: both rows land in the same split, and only `X` is kept if it
lands in eval. Train never shares normalised text with eval (leaks are
dropped from train and counted).

Usage:
  python scripts/split_cases.py --in cases.aug.jsonl --train train.jsonl \
      --eval eval.jsonl --split split.json [--eval-frac 0.3] [--min-real 8] [--seed 7]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _is_real(rid: str) -> bool:
    return not (rid.startswith("aug:") or rid.startswith("seed:"))


def _base_id(rid: str) -> str:
    """The ledger id behind a row id: 'X#2' -> 'X'; taskset:/seed: passthrough."""
    if "#" in rid:
        return rid.split("#", 1)[0]
    return rid


def _real_count(rows: list[dict], lane: str) -> int:
    return sum(1 for r in rows if r.get("label") == lane and _is_real(r["id"]))


def split_rows(rows: list[dict], eval_frac: float, min_real: int, seed: int) -> dict:
    rng = random.Random(seed)
    lanes = sorted({r["label"] for r in rows})

    # Correction pairs become units so X and X#2 never straddle.
    by_id = {r["id"]: r for r in rows}
    units: dict[str, list[dict]] = {}
    for r in rows:
        if not _is_real(r["id"]):
            continue
        units.setdefault(_base_id(r["id"]), []).append(r)
    real_per_lane = {lane: _real_count(rows, lane) for lane in lanes}

    eval_ids: set[str] = set()
    per_lane = {}
    for lane in lanes:
        lane_units = sorted((u for bid, u in units.items()
                             if u[0]["label"] == lane), key=lambda u: u[0]["id"])
        n_real = real_per_lane[lane]
        n_eval = 0
        if n_real >= min_real:
            n_eval = max(2, round(eval_frac * n_real))
            # Units are 1 or 2 rows; walk picking whole units until n_eval.
            picks: list[list[dict]] = []
            picked = 0
            order = lane_units[:]
            rng.shuffle(order)
            for u in order:
                if picked >= n_eval:
                    break
                # Prefer single-row units so we do not overshoot badly.
                picks.append(u)
                picked += len(u)
            for u in picks:
                for r in u:
                    if r["id"] == u[0]["id"]:  # keep only X, drop X#2 from eval
                        eval_ids.add(r["id"])
        train_ids = [r["id"] for r in rows if r.get("label") == lane
                     and r["id"] not in eval_ids]
        eval_lane = [r["id"] for r in rows if r.get("label") == lane
                     and r["id"] in eval_ids]
        per_lane[lane] = {
            "train": len(train_ids),
            "eval": len(eval_lane),
            "real": n_real,
            "aug": sum(1 for r in rows if r.get("label") == lane
                       and r["id"].startswith("aug:")),
            "seed": sum(1 for r in rows if r.get("label") == lane
                       and r["id"].startswith("seed:")),
        }

    train_rows = [r for r in rows if r["id"] not in eval_ids]
    eval_rows = [r for r in rows if r["id"] in eval_ids]

    # Text-leak check: drop train rows whose normalised text matches eval.
    eval_norms = {_norm(r["text"]) for r in eval_rows}
    kept, leaked = [], 0
    for r in train_rows:
        if _norm(r["text"]) in eval_norms:
            leaked += 1
        else:
            kept.append(r)
    train_rows = kept

    # Disjointness asserts: by id and by normalised text.
    train_ids = {r["id"] for r in train_rows}
    eval_id_list = [r["id"] for r in eval_rows]
    assert not (train_ids & set(eval_id_list)), "train/eval id overlap"
    train_norms = {_norm(r["text"]) for r in train_rows}
    assert not (train_norms & eval_norms), "train/eval text overlap"

    return {
        "rows": (train_rows, eval_rows),
        "split": {
            "seed": seed,
            "eval_frac": eval_frac,
            "min_real": min_real,
            "train_ids": sorted(train_ids),
            "eval_ids": sorted(eval_id_list),
            "per_lane": per_lane,
            "leaked_dropped": leaked,
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="infile", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval", dest="evalfile", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--eval-frac", type=float, default=0.3)
    ap.add_argument("--min-real", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    rows = [json.loads(l) for l in Path(args.infile).read_text(encoding="utf-8").splitlines()
            if l.strip()]
    result = split_rows(rows, args.eval_frac, args.min_real, args.seed)
    train_rows, eval_rows = result["rows"]
    Path(args.train).write_text(
        "".join(json.dumps(r) + "\n" for r in train_rows), encoding="utf-8")
    Path(args.evalfile).write_text(
        "".join(json.dumps(r) + "\n" for r in eval_rows), encoding="utf-8")
    Path(args.split).write_text(
        json.dumps(result["split"], indent=2) + "\n", encoding="utf-8")

    print(f"{'lane':<28} {'train':>6} {'eval':>5} {'real':>5} {'aug':>5} {'seed':>5}")
    for lane, c in result["split"]["per_lane"].items():
        print(f"{lane:<28} {c['train']:>6} {c['eval']:>5} {c['real']:>5} "
              f"{c['aug']:>5} {c['seed']:>5}")
    print(f"leaked_dropped: {result['split']['leaked_dropped']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
