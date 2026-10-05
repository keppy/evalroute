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
- config_name: contributed
  data_files:
  - split: train
    path: contributed/*/*.jsonl
---
# evalroute flywheel

Honest size: three lanes are **measured** (n=10 each; arms tied at p=1.0) and
six are **priors** from public benchmarks — not our runs. The pipe is real,
the table is small, your outcomes grow it.

Configs: **measured** — Tier-A harness artifacts (`runs.jsonl`, `report.csv`,
`tasks.jsonl`, models manifests); written only by harness runs, never accepts
contributed rows · **routes** — the `routes.yaml` that `evalroute sync` pins
(`MANIFEST.json` says what produced it) · **contributed** — redacted outcome
rows from opted-in installs under `contributed/<contributor>/`.

## What a contributed row carries — only this, whitelist-redacted

| key | what it is |
| --- | --- |
| `kind` | always `outcome` |
| `route_lane`, `route_model`, `route_effort` | the routed arm |
| `actual_model`, `actual_effort`, `arm_attribution` | the arm you actually ran (only when you confirmed it) |
| `method`, `confidence` | how the route was classified |
| `rated` | pass / fail / skip (rating corrections applied) |
| `facets` | counts only, e.g. `{"long-doc": 1}` |
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
