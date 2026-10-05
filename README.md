# evalroute

Route a task to the right **(model, reasoning effort) arm** before you start.
evalroute is a Python package: it classifies the task, prints the arm, and
labels outcomes so the table improves from real work.

Honest size check: three lanes are **measured** (n=10 each; arms tied at
p=1.0) and six are **priors** from public benchmarks — not our runs. The pipe
is real, the table is small, your outcomes grow it.

## For agent harnesses

If you are a coding agent (Claude Code, Codex, Cursor, OpenHands) arriving
from a talk QR code, the whole loop is five commands — see `AGENTS.md` for the
full contract and rules:

```bash
uv tool install evalroute      # or: pip install evalroute
evalroute route --json "fix the failing test in tests/test_x.py"
```

```json
{"lane": "routine-coding", "model": "z-ai/glm-5.3-flash", "effort": "medium",
 "route_id": "a1b2c3d4", "provider": "nous", "confidence": 1.0,
 "classification_method": "rules-strong", "provenance": "measured ...",
 "table": "routes: bundled (9 lanes)", "card": "..."}
```

```bash
# ... run the task on that arm, your own way ...
evalroute rate pass --route-id a1b2c3d4 --model z-ai/glm-5.3-flash \
    --effort medium --note "tests green" --json
```

Every verb speaks `--json` (`route`, `rate`, `sync`, `dispatch`, `report`,
`contribute`);
human output is unchanged without it. `table:` provenance values:
`measured` (controlled runs — prefer these lanes), `observed`
(same-maintainer workflow labels — a hint), `priors` (vendor benchmarks — a
hint, not a result).

Sub-tasks can be dispatched to a worker on the routed arm:
`evalroute dispatch brief.md --runner <name-or-template>` — Hermes is the
default runner; any agent CLI works via a template (see
[docs/runners.md](docs/runners.md)).

## Install

```bash
uv tool install "evalroute[hub]"   # standalone CLI on PATH, + huggingface_hub for sync/contribute
pip install evalroute              # as a library: runtime dep is pyyaml only
evalroute --version
```

Hermes users: install the plugin from the catalog instead
(`hermes plugins install evalroute`); it pins this package and both surfaces
share one ledger.

## CLI

The same argparse tree the plugin registers, standalone. Cards render
command strings for the calling surface: `evalroute ...` here, `hermes
evalroute ...` under the plugin CLI, `/...` slash commands in chat:

```bash
evalroute route --lane routine-coding "fix the failing test"   # route card
evalroute route --json "read this 80-page spec and summarize"  # JSON envelope
evalroute rate pass --note "why"     # label the last routed task
evalroute sync --status              # which route table is active
evalroute install-routes --dry-run   # route table -> agent.reasoning_overrides
```

Works with no Hermes installed: the home resolves `$EVALROUTE_HOME`, then
`$HERMES_HOME`, then Hermes's platform default (`%LOCALAPPDATA%\hermes` on
Windows, `~/.hermes` elsewhere); the ledger, dataset pins, gate config and
effort-override reads all honor it, so a standalone install and the Hermes
plugin on the same machine see one ledger.

## Dispatch (with runners)

```bash
evalroute dispatch brief.md          # route a brief, spawn a worker on that arm,
                                     # print the rate line (--dry-run prints the argv)
evalroute dispatch brief.md --dry-run --json    # the dispatch sidecar object
```

`--runner` selects who executes the brief. Two named runners are built in and
verified against real flags: `hermes` (the default; its flags are this repo's
own spawn path) and `claude-code` (Claude Code 2.1.289 `-p` mode: `--effort`
maps 1:1, `--max-turns` is honoured, the JSON result's `session_id`,
`num_turns`, `total_cost_usd` and `terminal_reason` land in the sidecar).
Anything else is a template with placeholders `{model} {effort} {provider}
{brief} {brief_text} {indir} {max_turns}` — substituted into already-split
tokens, so spaces and quotes in paths survive:

```bash
evalroute dispatch brief.md --runner claude-code --model opus --effort high --max-turns 300
# or hand-roll the flags (budget, permission mode) — note the `--` before the prompt:
evalroute dispatch brief.md --runner "claude -p --model {model} --effort {effort} \
  --output-format json --permission-mode acceptEdits --max-budget-usd 20 \
  --max-turns {max_turns} --add-dir {indir} -- {brief_text}"
```

**The harness is part of the arm.** The same model at the same effort under
Hermes and under Claude Code is a different arm — different system prompt,
tools, approval model, and a different *unit* for `max_turns` (Hermes counts
tool-call iterations; Claude Code counts agentic turns). So every outcome row
carries `harness` and `max_turns` next to `model` and `effort`, the rate line
prints them, and `contribute` pools by all four. `--model`/`--effort` on
`dispatch` override the routed arm (the route table may name a model your
harness cannot run); the override is recorded as the actual arm.

Templates may omit `{effort}`: the card still records the routed effort and
the rate line still carries `--effort`. Templates for codex and aider live in
[docs/runners.md](docs/runners.md) as **unverified sketches**. `dispatch`
never rates its own work: it prints the rate line for a second agent or human
(the judge must not be the worker).

Every dispatch (including `--dry-run`) appends a run record to
`<brief>.dispatch.json` — route id, arm, runner, exit, session id, report
path, rate line — which `report` reads as the primary brief↔route↔session
source. `dispatch --json` prints exactly that record on stdout; the card goes
to stderr either way, and human stdout stays three lines without `--json`.

## Watch it run

```bash
evalroute dispatch brief.md --follow        # stream the worker session from
                                            # state.db to stderr while it runs
evalroute report --trains docs/trains --open   # one static HTML page: pending
                                            # routes, rated outcomes, per-lane
                                            # tally, sessions + train costs,
                                            # drift findings (add --watch 60 to
                                            # regenerate every minute)
```

`--follow` never touches stdout's three-line contract, the exit code, or the
report file; `evalroute report` is read-only over the ledger, `state.db`,
and the trains dir, and writes one self-contained HTML file.
`evalroute report --json` prints the computed data model instead of
writing HTML.

**Where the line is.** Hermes's UI shows Hermes things (transcripts,
sessions); evalroute's report shows evalroute things (routes, arms, outcomes,
cost). Neither embeds the other. The only Hermes touchpoints in the library
are `hermes_home()` (a path), the LLM facade seam, and read-only `state.db`
for `--follow` and the report.

## Route table

`evalroute/data/routes.yaml` — one row per lane: `id`, `keywords` (the rule
layer), `model`, `effort`, `escalation`, `provenance`, `notes`. Lane taxonomy
is the union of the two source tables (9 lanes); where they disagreed
(long-doc merged into web-research in one, orchestration only in the other)
both are kept as distinct lanes. Model ids must match `/model` spelling
exactly.

The table began with a **2026-09-26 priors snapshot** (public benchmarks,
many vendor-run). Three lanes now have small measured batches; the others
remain marked `priors`. Replace those rows only after your own controlled
data, and keep each row's `provenance` visible.

### What the harness measures

Three lanes measured with the evalroute harness via the Nous inference API —
10 tasks x 3 samples per arm (four routine-coding, five DL/ML, five
alignment arms, of which only four alignment arms have graded samples: 119
graded, 30 judge-pending, and one missing cell). Raw API and judge cost in
the vendored rows totals **$3.0590051**, excluding verification time across
routine coding, DL/ML research engineering, and alignment reasoning. Every
winner was statistically indistinguishable from its runner-up at n=10
(McNemar, via gonogo) — the empirical paired gap is zero, but the
conservative interval spans [-33.4%, +33.4%]; $p=1$ is not a population
equivalence test. All-coverage lanes have a ceiling on this taskset; the
routes choose cost among observed ties, not quality parity. The alignment
judge has no blind checker audit; its 30 pending Qwen outputs are excluded
from the route comparison. No quality claim spans that arm. The historical
v1 run records omit model IDs; the arm-name-to-ID mapping is the vendored
`models.json`, not an ID echoed by those records. Routine coding overturned
the priors' vendor pick: glm-5.3-flash at medium effort covered every task
at $0.00005/success, 2.4–3.9x cheaper than the V4.1 Flash arms at equal
coverage.

Every row is `priors`, `observed`, or `measured`. Nothing hypothesis-shaped
masquerades as a result — the card prints the row's provenance verbatim,
statistical stamp included.

### Regenerating from measured data

```bash
# in the harness venv (openai + anthropic; it stays out of the runtime venv)
python -m evalroute.harness.tier_a run -m models.json -t tasks.jsonl -k 3
python -m evalroute.harness.tier_a report -o runs.jsonl -t tasks.jsonl -k 3 --csv report.csv
python -m evalroute.routes_from_report --csv report.csv --runs runs.jsonl \
    --models models.json --k 3 --out routes.generated.yaml
```

`report` needs the matching `--tasks` file: it checks each run's prompt/checker
contract and treats missing declared tasks as incomplete. Without that file it
prints diagnostics but selects no route. The `report.csv` files under
`examples/artifacts/` were regenerated from their `runs.jsonl` and
`tasks.jsonl` with this harness and carry the `complete` column. A legacy CSV
(no `complete` column) is refused even with `--runs`, since the old winner
selection may have ignored pending cells; regenerate it the same way.

The harness is packaged so the loop is complete inside one repo: write
tasksets (deterministic `python` checkers where possible — validate every
checker against a reference solution before paid runs), run k samples per
arm, report, then flip the lane's row. `routes_from_report` applies the
report's own routing rule (coverage-gated lowest all-in $/success), stamps
`provenance: measured ...` with a gonogo McNemar stamp when the winner and
runner-up shared cases, preserves each lane's `keywords`/`match_hint`/
`escalation`/`notes`, and carries unmeasured lanes over verbatim —
regeneration never silently deletes a route. The tier-a tasksets and runs
that produced the current measured lanes are under `examples/artifacts/`
(sets: `tasks.jsonl`; raw run records: `runs.jsonl`).

Inspect the generated YAML before replacing the bundled table. `--models`
maps harness arm names such as `glm-5.3@high` to `/model` IDs; omission is
only safe if the CSV already contains routable IDs. `--k` defaults to 3;
graded sample counts must divide evenly by k, or the generator refuses to
invent a task count. The harness v2 resume key includes the full task and
checker spec, model price/config and effective max tokens, plus the judge
configuration when used. It keeps legacy JSONL reportable, but reporting
mixed legacy/v2 or multiple prompt/checker versions together fails explicitly.

### The Hermes shim (`api: "cmd"` arms)

To run Hermes itself as an evalroute arm (agentic cells):

```json
{"name": "hermes-glm@high", "api": "cmd", "effort": "high",
 "cmd": "python -m evalroute.runners.herbes_shim --model {model} --effort {effort} --prompt {prompt_file}",
 "model": "z-ai/glm-5.3", "in": 0.91, "out": 2.86, "timeout": 3600}
```

The shim wraps `hermes -z` (one-shot; tools, memory, AGENTS.md loaded as
normal; approvals auto-bypassed), reads the usage report (`--usage-file`),
and emits the contract evalroute expects: the answer on stdout, then one
JSON last line `{"text": ..., "usage": {"inp", "out", "cache_read"}}`. A
non-zero hermes exit becomes an `error` field in that line (the harness
records an error row and retries on the next `run`); the shim itself always
exits 0. `--system` is prepended to the prompt. `--hermes PATH` overrides
the executable (tests use this to point at a fake — no real runs).

## Classification: rules first, LLM when weak

The classifier's first layer is deterministic keyword rules over
`evalroute/data/routes.yaml` — free, no API calls — but rules alone misroute
paraphrase ("manage life, writing, and researchy tasks" has zero keyword
signal) and negation ("not usually hard math though" used to count as a
math hit; the rules now guard negated keywords). So routing is layered:

1. **Strong rules** (2+ distinct keyword hits on the winning lane) — trusted
   outright, no model call.
2. **Encoder** (opt-in) — if you have installed one, a small local classifier
   answers when the rules are weak, at $0 and offline, abstaining below its
   calibrated threshold. Needs `pip install "evalroute[encoder]"`. Two ways
   to get one, in this order:

   **Train your own on your own work** (the default path). Your ledger holds
   your real tasks with the lanes you pinned; nothing else does.
   ```bash
   evalroute export-cases --out cases.jsonl --seed-text     # your pinned routes as {id,text,label}; counts only on stdout
   python examples/evalroute_lane_encoder.py --train train.jsonl --eval eval.jsonl --out my-encoder   # thomas; 30 s on CPU
   evalroute install-encoder my-encoder
   ```
   Task text never leaves your machine. A dispatcher's ledger makes a coding
   router; a researcher's makes a different one — that is the point.

   **Or install the shared one** for day one:
   `evalroute install-encoder keppy/evalroute-lane-encoder`. v1 scored 63%
   on 19 real held-out tasks — a gonogo tie with rules+LLM, +32 points over
   rules alone, neither distinguishable at 0.95, hence opt-in. It was trained
   on public tasksets, lane seed text and LLM paraphrases of the lane
   descriptions; five of nine lanes have never seen a real task. The
   learning curve behind it says real rows are worth ~10× a paraphrase, so
   the shared model is a floor, not the target. Its data is moving to a
   public, human-reviewed corpus — [`keppy/evalroute-tasks`](https://github.com/keppy/evalroute-tasks)
   (real tasks with asserted lanes, no paraphrases); contribute a reviewed
   row there, especially for an empty lane, and the next shared encoder
   learns it.
3. **Weak signal** (0-1 hits, no encoder or it abstained) — one structured call
   via the facade set with `routing.set_llm_facade` (the host's own model and
   auth; the plugin sets it at register time, but **it consumes tokens and may
   incur provider charges**). `None` (the default, and what a bare `evalroute`
   CLI sees) disables the fallback. The LLM judges what the work IS — a
   description of an assistant's duties routes to `orchestration`, not to
   whatever nouns appear.

The card always prints which layer decided: `rules match`, `encoder (opt-in)`,
`LLM fallback`, or `no keyword hit - defaulted`, with `(encoder abstained)` when
an installed encoder deferred. If the LLM call fails (offline, no facade),
the weak rules result stands and the card says so. Pin manually with
`--lane <id>` when you know better.

## Facets: labels with dimensions

A lane is the routing decision; facets are the label. Every route also
captures the task's shape along three axes, defined in
`evalroute/data/facets.yaml`:

- **input-shape**: `long-doc` | `interactive`
- **domain**: `domain-dlml` | `domain-alignment` | `domain-math` | `domain-prose` | `domain-research`
- **demand-tier**: `tier-routine` | `tier-hard` | `tier-orchestration`

So "audit my RL training plan files" is recorded as `long-doc +
domain-dlml + tier-hard` — three facts about one task — instead of one
collapsed lane. The rules layer derives facets from keyword evidence
(conservative: only lanes that drew hits claim facets); the LLM fallback
names them semantically in the same structured call.

When a task claims facets on multiple axes, the card describes the
conjunction. **Facets do not alter the chosen arm**: the lane classifier
selects the arm; no domain/tier precedence is implemented:

```
facets: long-doc + domain-dlml (descriptive conjunction; lane chooses arm)
```

Facet conjunctions aggregate in `routes_from_labels`, so the
high-dimensional nodes — "how do long-doc x dl-ml tasks fare on arm X?" —
fill in from daily use without controlled-batch spend.

## Flywheel: labels from daily workflow

The controlled harness is not the only source of data. As you route in daily
sessions, the library quietly builds an observational dataset:

- **`route`** logs the assignment (lane, recommended arm, method,
  confidence, facets) to `<home>/evalroute/labels.jsonl` — the
  task text you typed is the label.
- **`/model` or `/reasoning` after a route** logs a process-global switch
  observation. Without a session join it is not a verified route rejection
  and cannot supply the actual arm for a flip.
- **`rate pass|fail [--route-id <id>] [--lane <id>] [--model <id> --effort <level>] [--note ...]`**
  labels the outcome when you finish. Use both arm flags to self-report what
  actually ran; otherwise the arm stays unknown. `--lane` corrects a lane;
  `skip` discards that route. Prefer `--route-id` in overlapping sessions.
- Nothing else is recorded: no response bodies or turn telemetry. Task text,
  switch arguments and optional notes are recorded. The last-seen model is
  process-global memory only and is **not** assigned to a route as fact.

The ledger therefore holds two row qualities: an **observed arm** (a
`/model` or `/reasoning` switch after a route — a process-global candidate,
not proof) and a **caller-stated arm** (`evalroute dispatch <brief.md>`
records the spawn arguments as `arm_attribution: explicit_user` on the
outcome row). `dispatch` is the one-line form of the flywheel: route the
brief, spawn a worker on exactly that arm, print the `rate it:` line —
it still never auto-rates `pass`.

The ledger is **profile-wide**, not session-scoped: command hooks do not
supply a reliable session ID for route and rate. The card prints a route ID;
when tasks overlap, select it with `--route-id`. Without it, `rate` consumes
the latest pending route in that profile, which may be another session's.
`/model` and `/reasoning` observations are process-global candidates, not
proof of which model served a given route; only an explicit `rate --model
... --effort ...` confirms an observational arm. To replace a pinned card,
use `route --lane <lane> --replace-route-id <old-id> <same task>`; identical
task text alone never consumes another pending route. Inspect the route ID,
task and lane before trusting a label.

**Continuing across sessions (turn caps).** A task that outlives its session —
the turn limit hits, the terminal closes mid-task — is a continuation, not a
new route. The row being labeled is (task, arm), not (task, session):

- Prefer staying in the session: `continue: <what remains>` gets a fresh
  iteration budget, and the arm is session-scoped and persists.
- Otherwise `hermes -c` continues the same conversation, or paste the capped
  session's final turn as the new session's opener.
- Never re-`route` the continuation. If the new session is a different task,
  `rate skip --route-id <id>` clears that specific pending row first if it
  belongs to you; don't consume another user's pending row.
- Steer as much as you like. The observed layer is defined as
  daily-workflow-with-a-human-in-the-loop; a directive continuation is normal
  operation, and it matches a detailed original prompt better than a bare
  "continue" — which quietly tests prompt-luck instead of the arm. Measured
  rows are untouched: they come from fixed-prompt, fresh-context harness cells.
- Put the methodology in the note: `--note "completed across two sessions
  (turn cap), directive continuation"`. The label records neither cost nor
  session boundaries, and a session-spanning pass re-reads the accumulated
  context at full input price — the note is where that lives.

**Turning labels into route data:** `python -m evalroute.routes_from_labels`
prints per-lane, per-arm pass rates, lane corrections, and facet conjunction
outcomes; `--apply` writes `routes.observed.yaml`. Observed rows carry
honest, weaker provenance:

```
observed 23 outcomes, same-maintainer observational single-arm, pass 78%, 2026-10-30;
not independent trials or a controlled comparison
```

and only where the lane has no `measured` row — observational data can contest
a priors row, never overwrite a measured one. When gonogo is installed, each
observed row also carries its decide() verdict, so a row with 3 outcomes reads
as INSUFFICIENT_EVIDENCE rather than a pass rate someone will trust. The one
provisional flip threshold is 3 user-confirmed arm failures on the recommended
arm and 2 user-confirmed wins on another observed arm. This is still
same-maintainer, non-randomized evidence — review it and verify with paired
controlled cases before treating it as a quality comparison. The two
same-arm outcomes in the old snapshot never justify a route flip.

**Publishing the labels.** The live ledger stays local and append-only; do
not check raw labels into a public repo. Task text and notes can expose
paths and private project details even without response bodies. Do not claim
retroactive erasure for any label data once shared.

## Dataset/sync

The table you route against is either the bundled one or a pinned dataset
revision; the card says which. The dataset is fetched only when you run
`sync` (never on install, never while routing), it is pinned to a resolved
revision, and the library routes fully offline without it:

```bash
evalroute sync --revision <sha>   # pin the published table (default: main)
evalroute sync --status           # bundled, or dataset @ <sha>
evalroute sync --clear            # back to the bundled table
evalroute sync --status --json    # {"active", "sha", "path", "lanes", "measured"}
```

`sync` downloads only the `routes/` config of
[keppy/evalroute-flywheel](https://huggingface.co/datasets/keppy/evalroute-flywheel)
into `<home>/evalroute/dataset/<sha>/` — never the measured evidence
(grows over time; leave it on the Hub). It needs `pip install
huggingface_hub` (the `hub` extra). `--with-contributed` also pulls the
`contributed/` tree, read-only, so you can pool the redacted rows yourself
(see below).

## Contributing outcomes

After a week of routing and rating, your ledger holds real outcomes on arms
nobody has measured. `contribute` shares them — opt-in, redacted, and
nothing on by default. The route table's honest size (top of this README)
only grows when people do this.

What leaves your machine — and only this, whitelist-redacted:

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
| `corrected`, `schema` | correction flag; schema version (2) |
| `max_turns` | the agent's per-run tool-turn cap the arm ran under (Hermes `agent.max_turns`), or null — part of the arm: the same model and effort at 150 turns and at 300 are different arms |

Lane corrections ship too, as lane pairs only: when you rerouted with `--lane` or
rated with `--lane`, a row `{"kind": "lane_correction", "from_lane", "to_lane",
"method", "week", "schema"}` says the classifier picked A and you said B. That is
a fact about the lane descriptions that pools across installs without any text.

What never leaves: task text, notes, paths, hostnames, session keys,
emails, the salt itself, the HF token (evalroute never reads it —
`huggingface_hub` uses its own store), and any ledger key not on the list.
A future whitelist change is a schema bump.

```bash
evalroute contribute --dry-run          # prints the exact rows that would go; grep first
evalroute contribute                    # uploads after you've read them
evalroute contribute --rotate-salt      # new salt + empty cursor
```

Turn the gate on for your surface (off by default): `EVALROUTE_CONTRIBUTE=1`
(standalone convenience; Hermes users use config), or `evalroute.contribute:
true` in the Hermes `config.yaml`, or `{"contribute": true}` in
`<home>/evalroute/config.json`. Uploads land as one JSONL file per run under
`contributed/<your-hf-username>/` in the dataset (`--repo` to target a
fork), and a cursor makes each upload send only new rows.

Pooled rows are **observational**: they can contest a `priors` lane, never
touch a `measured` one, and cannot introduce a model the table doesn't
already know. After the next `sync`, contributors' rows show up on route
cards as `observed N tasks across K contributors, single-arm, pass R%`.

## Known limitations

- **The rule layer is keywords.** Deterministic, free, and misses
  paraphrase — which is what the LLM fallback is for. The fallback's
  quality tracks whatever model the host is on; it costs one small
  structured call (temp 0, 256 tokens) only when rules are weak.
- **The library cannot switch the model for you.** `route` prints the card;
  you run the task on that arm. Run it before turn 1 — mid-session switches
  re-read the whole context at full input price.
- **One effort slot per model id** (`agent.reasoning_overrides`); when a
  model serves two lanes, `install-routes` keeps the higher effort. A
  lower-effort lane's card explicitly prints `/reasoning <lane-effort>`
  after `/model`, because the installed override alone would run the wrong
  arm.
- **Provenance is priors until you measure.** Several rows are explicitly
  untested/unmeasured/contested; the card prints the provenance verbatim so
  nobody mistakes a hypothesis for a result.
- **The harness stays out of the runtime venv.** Run it in its own venv with
  `openai`/`anthropic`; `EVALROUTE_PYTHON` points the runners at that
  interpreter. The package itself requires nothing beyond `pyyaml`
  (`huggingface_hub` only for `sync`).
- **The sniff hook is advisory only** (plugin side): it speaks when the
  classifier is confident and the session's model disagrees with the lane's
  route; it never rewrites, blocks, or switches.
- **`install-routes` needs the Hermes config module to write.** Standalone,
  writing `agent.reasoning_overrides` works inside the `hermes` process;
  `--dry-run` works anywhere.
- **Named runners are `hermes` only.** Other agent CLIs get templates in
  `docs/runners.md`; a sketch is unverified flags until someone reads the
  CLI's real `--help`.

## Hermes plugin

The Hermes plugin is a thin adapter over this package, at
`keppy/hermes-plugin-evalroute` (`/route`, `/rate`, the `evalroute_route`
tool, the first-turn sniff, and the bundled skill live there; the routing
core lives here). It relies on exactly the ten contract names in
`evalroute/contract.py` (`CONTRACT_VERSION = 1`): `routing.set_llm_facade`,
`routing.set_surface`, `routing.evalroute_route`,
`routing.handle_route_command`, `cli.setup_cli`, `cli.evalroute_cli`,
`flywheel.handle_rate`, `flywheel.on_pre_command`,
`flywheel.on_post_llm_call`, `schemas.EVALROUTE_ROUTE`.

## Roadmap

- More measured lanes. Six of nine rows are still priors; each one needs a
  taskset with deterministic checkers and a paid harness run (~$1 per lane at
  current prices). Contributed rows can contest a priors lane but never make
  it `measured`.
- Classifier: train your own encoder on your own ledger (free, local, 30 s).
  The shared one is a day-one floor trained on public rows only; it improves
  when people publish tasksets or contribute cases with text on purpose, not
  from daily use — 67% of one dispatcher's labels were `routine-coding`, and
  the five lanes it never routes to got zero rows.
