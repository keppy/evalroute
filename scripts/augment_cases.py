"""Augment exported lane cases with training-only paraphrases (run by hand).

Reads an `export-cases` JSONL file ({"id","text","label"} rows), and for each
lane writes synthetic task requests to bring the lane up to --per-lane rows.
Augmentation is training-only: eval must stay real rows (see split_cases).

The standalone library has no LLM client, so this script owns a minimal one:
OpenAI-compatible chat completions via urllib only. Endpoint and key come
from EVALROUTE_LLM_BASE_URL / EVALROUTE_LLM_API_KEY (any OpenAI-compatible
server), falling back to OpenRouter with OPENROUTER_API_KEY.
Ledger text is private — prompts carry lane metadata and taskset rows only,
never ledger route rows.

Usage:
  python scripts/augment_cases.py --in cases.jsonl --out cases.aug.jsonl \
      --per-lane 30 [--model z-ai/glm-5.3-flash] [--dry-run | --yes] [--seed 7]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import urllib.request
from pathlib import Path

from evalroute.routing import _load_routes

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


def _endpoint() -> tuple[str, str | None]:
    """(chat-completions URL, key) from env; generic names win over OpenRouter."""
    base = os.environ.get("EVALROUTE_LLM_BASE_URL") or DEFAULT_BASE_URL
    key = os.environ.get("EVALROUTE_LLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    return base.rstrip("/") + "/chat/completions", key
DEFAULT_MODEL = "z-ai/glm-5.3-flash"
PER_CALL = 10


class ChatClient:
    """Minimal OpenAI-compatible chat completions client (urllib only)."""

    def __init__(self, model: str, api_key: str | None = None):
        self.model = model
        self.url, env_key = _endpoint()
        self.key = api_key if api_key is not None else env_key
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def chat(self, prompt: str) -> list[str]:
        """One 10-line block; returns parsed lines, records usage tokens."""
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }).encode("utf-8")
        req = urllib.request.Request(
            self.url, data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.key}"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        usage = payload.get("usage") or {}
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)
        text = payload["choices"][0]["message"]["content"]
        return [ln.strip() for ln in text.splitlines()]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _clean_line(line: str) -> str | None:
    """Drop blanks, numbering and wrapping quotes; None if nothing survives."""
    t = line.strip()
    if not t:
        return None
    t = re.sub(r"^\s*\d+[.)]\s*", "", t)  # numbering
    t = t.strip('"') if t.startswith('"') and t.endswith('"') else t
    t = t.strip()
    if not t:
        return None
    return t


def _keyword_overlap(lane: dict, lanes: list[dict]) -> list[tuple[int, dict]]:
    kw = set(lane.get("keywords") or [])
    scored = []
    for other in lanes:
        if other["id"] == lane["id"]:
            continue
        ov = len(kw & set(other.get("keywords") or []))
        scored.append((ov, other))
    return sorted(scored, key=lambda x: (-x[0], x[1]["id"]))


def _prompt_for(lane: dict, lanes: list[dict], examples: list[str]) -> str:
    avoid = [f"- {o['label']} ({o['id']}): hint: {o.get('match_hint') or '(none)'}"
             for _, o in _keyword_overlap(lane, lanes)[:2]]
    ex = "\n".join(f"  - {e}" for e in examples) or "  (none)"
    return (
        f"Write 10 task requests a person would type to a coding agent, one per "
        f"line, 8-40 words, varied domains and vocabulary, no numbering, no quotes.\n"
        f"They must all fit the lane '{lane['label']}' (id {lane['id']}).\n"
        f"Match hint: {lane.get('match_hint') or '(none)'}\n"
        f"Keywords: {', '.join(lane.get('keywords') or [])}\n"
        f"Notes: {lane.get('notes') or '(none)'}\n"
        f"Example real requests in this lane (do not copy them):\n{ex}\n"
        f"Do not resemble these other lanes:\n" + "\n".join(avoid)
    )


def _count_real(rows: list[dict], lane_id: str) -> int:
    """Real rows for a lane: ledger, ledger#2 correction duplicates, taskset.

    seed:<lane>:... rows are seeds, not real counts.
    """
    return sum(1 for r in rows
               if r.get("label") == lane_id
               and not str(r.get("id", "")).startswith("seed:"))


def build_plan(rows: list[dict], per_lane: int) -> list[dict]:
    """Per-lane plan entries sorted by lane order in the table."""
    lanes = _load_routes()
    plan = []
    for lane in lanes:
        real = _count_real(rows, lane["id"])
        target = per_lane - real
        plan.append({"lane": lane, "real": real, "target": target,
                     "calls": max(0, math.ceil(target / PER_CALL)) if target > 0 else 0})
    return plan


def _print_table(plan: list[dict], short: dict[str, int] | None = None) -> None:
    header = f"{'lane':<28} {'real':>5} {'target':>7} {'calls':>6}"
    if short is not None:
        header += f" {'made':>6}"
    print(header)
    total = 0
    for p in plan:
        made = short.get(p["lane"]["id"], p["target"]) if short is not None else p["target"]
        total += made
        flag = "  SHORT" if short is not None and short.get(
            p["lane"]["id"], p["target"]) < p["target"] else ""
        print(f"{p['lane']['id']:<28} {p['real']:>5} {p['target']:>7} {p['calls']:>6}"
              + (f" {made:>6}{flag}" if short is not None else ""))
    if short is not None:
        print(f"total generated: {total}")


def run(rows: list[dict], out_path: Path, per_lane: int, model: str,
        seed: int, dry_run: bool, prompt_lane: str | None = None) -> int:
    plan = build_plan(rows, per_lane)
    lanes = _load_routes()
    rng = random.Random(seed)
    if dry_run:
        # Full prompt for one lane (--prompt-lane, else first with target > 0).
        chosen = None
        for p in plan:
            if prompt_lane and p["lane"]["id"] == prompt_lane:
                chosen = p
                break
            if not prompt_lane and p["calls"] > 0:
                chosen = p
                break
        if chosen and chosen["calls"] > 0:
            lane = chosen["lane"]
            ex = _lane_examples(rows, lane["id"], rng)
            print(_prompt_for(lane, lanes, ex))
        else:
            print(f"augment_cases: lane {prompt_lane!r} has no positive target; "
                  "nothing to prompt", file=sys.stderr)
        _print_table(plan)
        return 0
    client = ChatClient(model)
    if client.key is None:
        print("augment_cases: no API key — set EVALROUTE_LLM_API_KEY (+ EVALROUTE_LLM_BASE_URL) "
              "or OPENROUTER_API_KEY", file=sys.stderr)
        return 2
    generated: dict[str, list[str]] = {ln["id"]: [] for ln in lanes}
    real_norm = {_norm(r["text"]) for r in rows if not str(r.get("id", "")).startswith("seed:")}
    short: dict[str, int] = {}
    for p in plan:
        lane_id = p["lane"]["id"]
        made = 0
        if p["calls"] > 0:
            ex = _lane_examples(rows, lane_id, rng)
            prompt = _prompt_for(p["lane"], lanes, ex)
            for call in range(p["calls"]):
                lines: list[str] = []
                for attempt in (1, 2):
                    try:
                        lines = client.chat(prompt)
                        break
                    except Exception as exc:  # noqa: BLE001 — report, then retry once
                        # Never echo the prompt or key; the exception text is enough.
                        print(f"augment_cases: {lane_id} call {call + 1} attempt {attempt} "
                              f"failed: {type(exc).__name__}: {str(exc)[:200]}", file=sys.stderr)
                        if attempt == 2:
                            break
                for ln in lines:
                    text = _clean_line(ln)
                    if not text or _norm(text) in real_norm \
                            or _norm(text) in {_norm(x) for x in generated[lane_id]}:
                        continue
                    generated[lane_id].append(text)
                    made += 1
        short[lane_id] = min(made, max(p["target"], 0))
    out = list(rows)
    n = 0
    for p in plan:
        lane_id = p["lane"]["id"]
        for text in generated[lane_id][:p["target"]]:
            n += 1
            out.append({"id": f"aug:{lane_id}:{n}", "text": text, "label": lane_id})
    out_path.write_text("".join(json.dumps(r) + "\n" for r in out), encoding="utf-8")
    _print_table(plan, short)
    print(f"tokens: prompt={client.prompt_tokens} completion={client.completion_tokens}")
    print(f"wrote {out_path} ({len(rows)} -> {len(out)} rows)")
    return 0


def _lane_examples(rows: list[dict], lane_id: str, rng: random.Random) -> list[str]:
    """Up to 3 example texts from taskset rows only — never ledger rows."""
    texts = [r["text"] for r in rows
             if r.get("label") == lane_id and str(r.get("id", "")).startswith("taskset:")]
    rng.shuffle(texts)
    return texts[:3]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="infile", required=True)
    ap.add_argument("--out", dest="outfile", required=True)
    ap.add_argument("--per-lane", type=int, required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--seed", type=int, default=7)
    mode = ap.add_mutually_exclusive_group()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--prompt-lane", default=None,
                    help="dry-run only: which lane's prompt to print "
                         "(default: first lane with a positive target)")
    args = ap.parse_args(argv)
    rows = [json.loads(l) for l in Path(args.infile).read_text(encoding="utf-8").splitlines()
            if l.strip()]
    if not args.dry_run and not args.yes:
        plan = build_plan(rows, args.per_lane)
        _print_table(plan)
        return 2
    return run(rows, Path(args.outfile), args.per_lane, args.model, args.seed,
               args.dry_run, prompt_lane=args.prompt_lane)


if __name__ == "__main__":
    sys.exit(main())
