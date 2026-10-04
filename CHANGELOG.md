# Changelog

## [Unreleased]

### Added

- `dispatch --follow`: after spawning the child, poll the session store
  (`<hermes home>/state.db`, read-only sqlite URI, 2 s) for the newest
  session whose `cwd` matches `--in` (or the brief's dir) and `started_at`
  is at/after spawn time; stream its new `messages` rows to stderr, one
  line each (`HH:MM:SS  role[/tool]  first 100 chars`). Stdout's three-line
  contract, the exit code, and the report file are untouched; a missing or
  locked store prints one stderr line and the loop keeps waiting on the
  child. `--timeout` still kills the whole tree (exit 124).
- `evalroute report [--out PATH] [--open] [--watch SECONDS] [--trains DIR]
  [--factory-json PATH]`: one self-contained static HTML page (stdlib only,
  no JS, no external assets) over the flywheel ledger (pending routes,
  user-confirmed outcomes with dispatcher corrections, per-lane per-arm
  tally with the active table's provenance, classification-method split),
  the Hermes session store (per-train session costs), the trains dir
  (brief → report → route id → session links), and `factory check --json`
  drift findings. Absent inputs downgrade to one-line notes. Registered in
  `cli.setup_cli`, so both `evalroute report` and `hermes evalroute report`
  work.
- `dispatch --runner <name|template>` (env `EVALROUTE_RUNNER`): spawn the
  worker through any agent CLI. A template carries `{model} {effort}
  {provider} {brief} {brief_text} {indir}` placeholders, substituted into
  already-split tokens so spaces/quotes in paths survive; `{effort}` and
  `{indir}` may be omitted (the card still records the routed effort and the
  rate line still carries `--effort`). Named runners: `hermes` (default,
  unchanged argv); other CLIs get unverified sketches in `docs/runners.md`
  until their real `--help` has been read. Unknown name exits 2 with the
  list. `EVALROUTE_HERMES_BIN` still works.
- Dispatch sidecar: every dispatch (real or `--dry-run`) appends a run
  record — route_id, lane, model, effort, provider, runner, brief, indir,
  started/ended, duration_s, exit, session_id, report, rate_line — to
  `<brief>.dispatch.json` (newest last under `"runs"`). `report` reads
  sidecars (`**/*.dispatch.json` under `--trains`) as the primary
  brief↔route↔session source, falling back to the README dispatch log and
  the report's `session:` line — so the Sessions section now populates on
  real dispatches.
- `--json` on rate, sync, dispatch, and report. `rate --json` →
  `{logged, verdict, route_id, lane, model, effort, arm_attribution, message}`;
  `sync --json` → `{active, sha, path, lanes, measured}` (status/sync/clear);
  `dispatch --json` → the sidecar object on stdout (card still on stderr);
  `report --json` → the report's computed data model instead of writing HTML.
  Human output is byte-identical when `--json` is absent.
- `EVALROUTE_HOME`: first-priority home env var in `paths.hermes_home()`, so
  non-Hermes users have a non-Hermes name; `HERMES_HOME` still honoured.
- `docs/runners.md` (verified vs sketch runner templates) and root
  `llms.txt` (llmstxt.org shape).

### Changed

- README reordered agent-first: the five-command loop and the `--json`
  output shape are the first screen; route table, classification, facets,
  flywheel, dataset/sync follow; "Where the line is" states the Hermes /
  evalroute UI boundary.

## [0.6.0] - 2026-10-03

The routing core moved out of `hermes-plugin-evalroute` v0.5.1 into this
library. A move, not a rewrite: no behaviour change. The Hermes plugin
remains at `keppy/hermes-plugin-evalroute` and becomes a thin adapter over
the nine contract names in `evalroute/contract.py`.

Moved to the library:

- `tools.py` → split mechanically along existing function boundaries into
  `evalroute/routing.py` (table load/validate, classifier, facets, route
  card, `_tool_result`, `_note_route`, `install_routes`) and `evalroute/cli.py`
  (`setup_cli`, `evalroute_cli`, the workflow epilog).
- `flywheel.py`, `dispatch.py`, `dataset.py`, `schemas.py`, `adjudicate.py`,
  `routes_from_report.py`, `routes_from_labels.py` (imports made relative;
  the path-load + sibling-injection hack in `evalroute_cli` and its
  `if "fw" not in globals()` guard deleted — `dispatch` is now
  `from . import dispatch`).
- `harness/evalroute.py` → `evalroute/harness/tier_a.py`;
  `runners/hermes-shim.py` → `evalroute/runners/hermes_shim.py`.
- `data/routes.yaml`, `data/facets.yaml` → `evalroute/data/` as package data
  (resolved via `importlib.resources`).
- `examples/artifacts/` (repo root, not package data) and
  `scripts/publish_dataset.py`.
- The tests, minus `test_registration.py` and `test_sniff.py`, which stay in
  the plugin (registration wiring and the Hermes sniff hook are plugin
  concepts). `sniff.py` and `__init__.py`'s `register(ctx)` stay in the
  plugin too.

Adapted:

- The `hermes_constants.get_hermes_home` try/except fallback at four sites
  is now one function, `evalroute.paths.hermes_home()`.
- `PLUGIN_DIR` anchoring is gone: package data comes from
  `importlib.resources`, and `_plugin_version()` (which read `plugin.yaml`)
  becomes the package version — `pyproject.toml` via
  `importlib.metadata.version("evalroute")`, literal fallback when not
  installed. The card's provenance line reads `table: bundled
  (evalroute 0.6.0)`.
- `evalroute.contract` is new: `CONTRACT_VERSION = 1` and the nine names the
  plugin relies on (`routing.set_llm_facade`, `routing.evalroute_route`,
  `routing.handle_route_command`, `cli.setup_cli`, `cli.evalroute_cli`,
  `flywheel.handle_rate`, `flywheel.on_pre_command`,
  `flywheel.on_post_llm_call`, `schemas.EVALROUTE_ROUTE`).
- A standalone `evalroute` console script wraps the same argparse tree, so
  `evalroute route|dispatch|sync|rate ...` matches `hermes evalroute ...`
  byte-for-byte.

Kept in the plugin: `__init__.py` (`register(ctx)`), `sniff.py`, `skills/`,
`plugin.yaml`, `catalog/`, `tests/test_registration.py`,
`tests/test_sniff.py`.
