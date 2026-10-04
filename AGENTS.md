# AGENTS.md — evalroute, for agent operators

evalroute routes a task to a **(model, reasoning effort) arm** from a table with
provenance on every row, and labels outcomes so the table improves from real work.
Honest size: three lanes are **measured** (n=10 each, arms tied at p=1.0); six are
**priors** from public benchmarks, not our runs. The pipe is real, the table is
small, your outcomes grow it.

## If you are an agent harness

You do not need Hermes. The whole loop is five commands:

```bash
pip install evalroute
evalroute route --json "<one-line task>"          # → lane, model, effort, route_id, table
# run the task on that arm, your own way
evalroute rate pass --route-id <id> --model <model> --effort <effort> --note "<why>" --json
evalroute report --json                           # what the ledger says so far
```

Read `model`, `effort`, `route_id`, `provenance`, `table` from the `route` envelope.
`provenance` begins with one of:

- `measured` — controlled runs on this lane. Prefer these.
- `observed` — same-maintainer workflow labels. A hint.
- `priors` — vendor benchmark numbers. A hint, not a result.

Always pass `--model` and `--effort` to `rate` — they are what you *actually ran*, and
they make the row `arm_attribution: explicit_user`. A `rate` without them is filed as
`arm unknown` and is worth little. A precise `fail --note "<cause>"` is worth more than
a soft pass. `skip` is for probes that were never meant to be judged — and for arms you
could not run: the table's arms name a provider (`"provider": "nous"` today); if you have
no access to that provider or model, run the task on what you have and `rate skip --note
"arm unavailable: ran <your model> instead"` rather than labelling an arm you did not use.
`rate` exits non-zero when nothing was logged (bogus or already-consumed route id); check
the exit code or `"logged"` in `--json`, never assume.

## Dispatching sub-tasks on a routed arm

```bash
evalroute dispatch brief.md [--lane <lane>] [--runner <name|template>] [--in <dir>] [--timeout S] [--json]
```

Routes the brief, spawns a worker on that arm, waits, writes `<brief>.report.md` and
`<brief>.dispatch.json`, and prints the exact `rate` line to run after you have checked
the work. Runners: `hermes` (default) or a template with `{model} {effort} {provider}
{brief} {brief_text} {indir}` — see `docs/runners.md` for sketches for other CLIs
(unverified; verify each CLI's flags against its own `--help` before trusting them).

**Never rate your own dispatch.** The worker is not the judge; if the same process
writes the code and the verdict, the ledger measures nothing. Verify the output
yourself (run the tests, run the oldest CLI path by hand), then rate.

## Sharing the table

```bash
evalroute sync              # pin the published table (keppy/evalroute-flywheel) at a revision
evalroute sync --status     # bundled or dataset @ <sha>
evalroute sync --clear      # back to bundled
```

`sync` downloads data, not code, on your explicit command, pinned to a dataset commit
sha, validated before activation; routing works offline on the bundled table without
it. Needs `pip install "evalroute[hub]"`. `contribute` (opt-in, redacted outcome rows)
is planned and not yet shipped; nothing leaves your machine today.

## Rules

- Do not hand-edit `evalroute/data/routes.yaml` or write `provenance` by hand. The table is
  regenerated from measurements (`routes_from_report.py`) and observed labels
  (`routes_from_labels.py`); observed rows never overwrite measured ones.
- The ledger (`<home>/evalroute/labels.jsonl`) may contain task text. It is local. Never
  commit it, never upload it raw.
- Home directory: `EVALROUTE_HOME`, else `HERMES_HOME`, else `~/.hermes`. Set
  `EVALROUTE_HOME` if you are not a Hermes user.
- `route`, `rate`, `sync`, `dispatch` and `report` speak `--json`; parse that, never the
  card. (`install-routes` is Hermes-only and has no JSON mode.)
- `<brief>.report.md` and `<brief>.dispatch.json` are train provenance, like the brief
  itself: commit them beside the brief. They hold ids, paths, exit codes and the worker's
  report — never the ledger's task text. Do not add them to `.gitignore`.
- Hermes touchpoints in this library are exactly three and all optional: the home path,
  an LLM facade for weak-signal classification (`None` disables it), and read-only
  `state.db` for `dispatch --follow` / `report` sessions.

## Where things are

| path | what |
| --- | --- |
| `evalroute/routing.py` | table load/validate, classifier, route card, `set_llm_facade` |
| `evalroute/cli.py` | argparse tree; `evalroute` console script; same tree the Hermes plugin registers |
| `evalroute/flywheel.py` | ledger: `note_route`, `handle_rate`, pending-route logic |
| `evalroute/dispatch.py` | route → spawn runner → report + sidecar → rate line |
| `evalroute/dataset.py` | `sync`: pinned dataset table under `<home>/evalroute/dataset/` |
| `evalroute/report.py` | one static HTML page / `--json` over ledger, sidecars, sessions, trains |
| `evalroute/contract.py` | `CONTRACT_VERSION` and the nine names the Hermes plugin may call |
| `evalroute/harness/tier_a.py` | the Tier-A measurement harness (paid calls when run by hand) |
| `evalroute/data/` | bundled `routes.yaml`, `facets.yaml` |
| `examples/artifacts/` | the measured runs behind the three measured lanes |
