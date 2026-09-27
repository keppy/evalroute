# evalroute

Measures **cost per verified success** for LLMs, per task lane, and picks a route: the cheapest model that doesn't lose coverage on the hard tail. One file, two deps (`openai`, `anthropic`), append-only JSONL.

## Why this exists

Per-token price is the wrong variable for choosing a model. Three hidden terms dominate what a task actually costs:

1. **Verbosity.** Some cheap models emit 2x the median output tokens, so compare cost per task, not per token.
2. **The hard tail.** Easy benchmarks are saturated and hide real differences. On Terminal-Bench 2.1, frontier and open models cluster around 88–90. On Terminal-Bench 4.0 they spread from ~13% to ~50%. Retries can't buy a task a model never solves.
3. **Verification cost.** A cheap model is only cheap if its output is cheap to check.

The working rule: **use cheap models where the checker is cheap (tests, compilers, Lean), and expensive models where the checker is a human.** This harness puts numbers on that rule using your own tasks, instead of vendor tables that mix harnesses and grade each other.

## Quick start

```bash
pip install openai anthropic numpy          # numpy only for the example coding task
export NOUS_API_KEY=... NOUS_BASE_URL=...   # base URL from portal.nousresearch.com/api-docs
export OPENAI_API_KEY=... ANTHROPIC_API_KEY=...

python evalroute.py init                    # writes example models.json + tasks.jsonl
python evalroute.py run -k 3 --judge gpt-6-sol
python evalroute.py grade --audit 5         # blind human grading, timed
python evalroute.py report --usd-per-hour 100 --csv report.csv
```

`run` is resumable. Rerunning skips completed cells and retries errored ones. Use `--only a,b` to run a subset of models.

## Effort is part of the model

Every model entry must set `effort`, and `run` refuses to start if any entry is missing it. Provider defaults differ and change between releases:

- Kimi K3's native default is max.
- Hermes sends medium when nothing is configured.
- Opus 5.5 defaults to medium, where Opus 5 defaulted to high.

With effort unpinned, you don't know which configuration you measured. Effort also moves results about as much as the choice of model does. GPT-6 Luna at max beats GPT-6 Sol at high on DeepSWE, and V4.1 Flash drops from 39 to 25 on the Artificial Analysis index with reasoning off.

So the thing you route between is a **(model, effort) arm**, not a model. Give each effort level its own entry. `init` ships `@medium` and `@max` arms for every model:

- `@medium` matches what Hermes sends by default.
- `@max` matches the setting most published benchmarks used.

| `api` | Effort is sent as | Valid values |
|---|---|---|
| `openai` | `extra_body.reasoning_effort` | `none minimal low medium high xhigh max` (provider-dependent) |
| `anthropic` | `output_config.effort` | `low medium high xhigh max` |
| `cmd` | the `{effort}` template placeholder | whatever your harness accepts |

`"effort": "default"` sends nothing and uses the provider's default. It's allowed, but `run` prints a notice listing the unpinned arms. Validation also rejects:

- effort set inside `extra` instead of the `effort` field,
- `anthropic` entries with `none` or `minimal`,
- `cmd` templates without an `{effort}` placeholder, since the setting would be silently dropped.

**Config changes rerun instead of pooling.** Each record carries a hash of `api`, `model`, `base_url`, `effort`, `extra`, and `cmd`. If you edit an entry, `run` reruns it. If one model name has results from more than one config, the report shows them as separate `name#<hash>` rows with a warning, rather than mixing them.

**Token caps at high effort:** `max_tokens` caps thinking plus the answer combined, so raise it for `xhigh` and `max`. `init` sets 64000 on the max arms. Watch the report's `trunc=` flag.

### Matching what Hermes actually runs

These notes describe Hermes as of this writing:

- **Effort has three layers.** Session `/reasoning` > per-model `agent.reasoning_overrides` > global `agent.reasoning_effort`.
- **Per-model overrides have shipped.** `agent.reasoning_overrides` maps model id → effort and is re-resolved on `/model` switches, so effort follows the model automatically. `/model` is session-scoped by default.
- **Subagents have their own setting,** `delegation.reasoning_effort`. Mixture-of-Agents presets can set effort per slot.
- **The default is medium** when nothing is configured. That overrides model-native defaults, so K3 runs at medium, not max.

For your routing decisions to match your daily use, the arm you pick here should be the effort you actually run in Hermes. Writing the route table into `agent.reasoning_overrides` makes `/model` carry the effort; `/reasoning` remains a manual step only where two lanes share a model at different efforts.

## Metrics

Computed per (lane, model):

| Column | Meaning |
|---|---|
| `pass` | Mean over tasks of the per-task pass rate. The 95% CI is a cluster bootstrap over tasks, because samples within a task are correlated. |
| `cov` | Coverage: the fraction of tasks passed on at least one of k samples. This is the long-tail metric. |
| `pass^k` | The fraction of tasks passed on all k samples, i.e. reliability. |
| `$/try`, `$/succ` | API cost (plus judge cost) per attempt, and per success. |
| `vmin` | Human verification minutes per attempt, recorded by `grade`. Auto-checked tasks count as 0. |
| `all-in$/s` | `(API $ + judge $ + verify_hours × usd_per_hour) / passes`. This is the cost of retrying until a pass when every attempt has to be verified. |
| `med_out`, `p50s` | Median output tokens and median latency. |

**Routing rule.** Within each lane, keep the models whose coverage is at least the best coverage minus `--tol`, then pick the one with the lowest all-in $/success. Coverage acts as a gate, so a model with a great cost per success can't win a lane where it never solves the hard tasks. A `*` in the report marks the Pareto frontier on all-in $/success vs pass rate.

## Files

**`tasks.jsonl`**: one task per line.

```json
{"id": "code-lse", "lane": "coding", "system": "optional", "prompt": "...",
 "check": {"type": "python", "file": "sol.py", "cmd": "python test.py", "setup": {"test.py": "..."}, "timeout": 120}}
```

**`models.json`**: a list of model entries.

```json
{"name": "glm-5.3@max", "api": "openai", "base_url": "$NOUS_BASE_URL", "key_env": "NOUS_API_KEY",
 "model": "z-ai/glm-5.3", "effort": "max", "max_tokens": 64000, "in": 0.91, "out": 2.86,
 "cached": null, "cache_write": null, "max_param": "max_tokens", "extra": {}, "conc": 4}
```

- `effort` is required (see above).
- `in`, `out`, `cached`, and `cache_write` are USD per 1M tokens.
- Use `max_param: "max_completion_tokens"` for OpenAI reasoning models.
- `extra` is passed straight to the API (temperature, `extra_body`, and so on). Never put effort there.
- `max_tokens` overrides the `--max-tokens` default for that entry.

**`runs.jsonl`**: append-only, holding three record kinds.

- **run**: `task, lane, model, effort, cfg (config hash), sample, ph (prompt hash), check, text, usage, cost, latency, truncated, jcost, passed`
- **error**: the same keys plus `error`, with no `text`. These are retried on the next `run`.
- **grade**: `grade_of ("task|model|sample|cfg"), passed, verify_s, audit`

## Checkers

| `check.type` | Fields | Passes when |
|---|---|---|
| `exact` | `answer` | The last `ANSWER: x` line (or the last line, if none) equals `answer`. |
| `regex` | `pattern` | The pattern matches anywhere in the output. |
| `python` | `file, cmd, setup, timeout` | The last fenced code block (or the full text, if none) is written to `file` in a temp dir alongside `setup` files, and `cmd` exits 0. |
| `judge` | `rubric` | The `--judge` model ends its reply with `VERDICT: PASS`. If a model would be judging its own output, the item falls through to human grading. |
| `human` | `rubric` | You pass it in `grade`. Grading is blind: model names are hidden and order is shuffled. |

`grade --audit N` blind-regrades N outputs that were already auto-checked. The report then prints how often you agreed with each checker. A low agreement rate means the checker, not the model, is the problem.

## Agent harnesses (`api: "cmd"`)

For agentic runs (Hermes, or anything multi-turn or tool-using), point a model entry at a shell command:

```json
{"name": "hermes-glm@high", "api": "cmd", "effort": "high",
 "cmd": "hermes-shim --model {model} --effort {effort} --prompt {prompt_file}",
 "model": "z-ai/glm-5.3", "in": 0.91, "out": 2.86, "timeout": 3600}
```

The command template receives `{prompt_file}`, `{system_file}`, `{model}`, and `{effort}`. If a pinned effort isn't used in the template, validation fails. The Hermes batch runner already takes a `--reasoning_effort` flag, which is the natural thing for the shim to forward.

**Shim contract:** print the final answer, and make the last stdout line JSON: `{"text": "...", "usage": {"inp": N, "out": N, "cache_read": N}}`. Without that line, stdout is treated as the answer and cost shows as `?`. **A reference Hermes shim ships in [hermes-plugin-evalroute](https://github.com/keppy/hermes-plugin-evalroute) (`runners/hermes-shim.py`)**: it wraps `hermes -z`, reads `--usage-file` for real token counts, maps them to the contract keys, reports a non-zero hermes exit as an `error` row (retryable on the next `run`), and always exits 0 so the harness records rather than crashes. `--system` is prepended to the prompt; `--hermes PATH` overrides the executable for tests.

## Integration notes

- **Deterministic graders plug in as `python` checks.** Any grader that exits 0 on pass and non-zero on fail works, including gonogo-style matchers. The model output lands in `file`, and `cmd` runs with the temp dir as its working directory.
- **Use `report --csv` as the stable downstream interface.** `runs.jsonl` is the raw log, and its schema may grow. The [hermes-plugin-evalroute](https://github.com/keppy/hermes-plugin-evalroute) repo consumes this CSV directly: `routes_from_report.py` turns a report into a measured route table (coverage-gated cheapest-arm per lane, `provenance: measured ...` stamped per row), which the plugin then serves as route cards.
- **Prompt edits are detected.** Editing a task's prompt changes its hash, and the report warns when results mix prompt versions. Either give the edited task a new `id`, or filter old rows out of `runs.jsonl`.
- **The flywheel closes the loop.** The plugin's daily-use labels (`/route`, `/rate`, implicit verdicts from model/effort switches) accumulate observational per-lane, per-facet outcomes; snapshots live in the plugin repo (`data/flywheel/labels.jsonl`). Observational data prioritizes which lane to measure next and contests priors rows; only this harness's k-sample batches make a row measured.

## Priors (snapshot, 2026-09-26)

These are the starting routes this tool is meant to confirm or overturn. They come from public benchmarks, many of them vendor-run. **Replace them with your own report output as soon as you have data.**

**Most of these numbers were measured at max effort**:
- All of Kimi's published K3 results.
- The Artificial Analysis "(max)" entries for GLM 5.3 and V4.1 Flash.
- GPT-6 Sol at max or xhigh.

The one exception here is Opus 5.5 on MathArena, measured at high. At Hermes's default of medium, expect lower scores and fewer tokens than these figures suggest.

| Lane | Best open-weight tier | Gap to closed | Prior route (starting effort) |
|---|---|---|---|
| Routine coding | DeepSeek V4.1 Flash → GLM 5.3 Flash | small | V4.1 Flash @ low–medium: verbose, and tests catch misses cheaply |
| Hard agentic coding | GLM 5.3 (TB 4.0: 42% vs K3 13%, Artificial Analysis, at max) | ~10 pts | GLM 5.3 @ high–max, then GPT-6 Sol / Opus 5.5 |
| DL / ML research engineering | GLM 5.3 ≈ Kimi K3 (vendors disagree on PostTrainBench) | small | GLM 5.3 @ high; verify |
| Long-doc reading | V4.1 Flash (AA-LCR 84%, on par with GPT-5.6 Sol) | ~none | V4.1 Flash @ medium |
| Web research | Kimi K3 (BrowseComp 91.2, vendor-reported, at max) | ~none | K3 @ medium: input-heavy work, and max roughly triples output at $10.53/M |
| Math, first principles | Qwen3.8-Max (MathArena 56% vs GPT-6 Sol 85%, Opus 5.5 83%) | ~27–30 pts | GPT-6 Sol @ max / Opus 5.5 @ high; skip open (exception: Lean-verified pipelines) |
| Alignment reasoning, paper claims | GLM 5.3 (K3's hallucination rate rose to 51% on AA-Omniscience) | unmeasured | measure with this tool |
| Prose | Kimi K3 (EQ-Bench CW #2 behind Opus 5; judged by a Claude model) | small | blind-test K3 vs Opus 5.5 @ low–medium (untested prior) |
| Orchestration | — | — | Opus 5.5 @ high; subagents @ low–medium via `delegation.reasoning_effort` |

Two cost notes from the same snapshot:

- **Use the direct Anthropic key for Opus 5.5.** Nous charges the same list price as direct, but direct gets $0.20 cache reads and 50% off via the Batch API.
- **K3's Nous promo is mostly an input discount.** Input is 71% off but output only 30% off ($0.88 / $10.53). Since K3 is verbose, it's cheap on input-heavy tasks, not on long reasoning.

## Measured so far (2026-09-27)

First three lanes measured with this harness via the Nous inference API — 10 tasks, k=3, python checkers where the output is code (validated against a reference solution before any paid run) and a third-vendor judge (qwen3.8-max) otherwise. ~$5 total.

| Lane | Winner | cov | all-in $/succ | Settled |
|---|---|---|---|---|
| Routine coding | glm-5.3-flash @ medium | 1.00 | $0.00005 | Priors' vendor pick (V4.1 Flash) never won: 2.4–3.9x the winner's cost at equal coverage |
| DL / ML research engineering | glm-5.3 @ high | 1.00 | $0.0024 | "GLM 5.3 ≈ K3, vendors disagree" resolved: GLM 5.3 on cost (K3 slightly higher raw pass, ~3x the price; its @medium arm also dropped coverage to 0.90) |
| Alignment reasoning | glm-5.3 @ medium | 1.00 | $0.0119 | Ceiling effect — all four arms 100%, cost decided. A harder set is needed before quality gaps are detectable |

Two caveats the reports carry in their provenance stamps: every winner was statistically indistinguishable from its runner-up at n=10 (McNemar, via gonogo), so these are cost decisions on tied arms, not quality claims; and the alignment set is too easy to detect quality differences. Runs and reports live in `examples/artifacts/tier-a-*/`; the measured route table itself is in [hermes-plugin-evalroute](https://github.com/keppy/hermes-plugin-evalroute)'s `data/routes.yaml`.

## Known limitations

- **Only `cmd` models can run agentic tasks.** The `openai` and `anthropic` modes are single-turn with no tool use.
- **Some config values were guesses at first, now partly verified.** Verified against `/models` on 2026-09-27: the Nous base URL (`https://inference-api.nousresearch.com/v1`) and `z-ai/glm-5.3`, `z-ai/glm-5.3-flash` work as-is; the Qwen judge id is `qwen/qwen3.8-max-0902`, not `qwen/qwen3.8-max`. The portal's two GLM 5.3 Flash prices both work — use the one you're billed ($0.12/$0.40 matched the runs here).
- **Prices are hard-coded in `models.json`.** Portal promos change, so update prices before trusting any cost column.
- **Small samples mean wide CIs.** You need at least 2 tasks per lane for a CI at all. Use 10+ tasks per lane and k ≥ 3 for signal. Coverage is noisy at small k.
- **LLM judges carry bias.** Never let a model judge itself. Prefer a judge from a different vendor than the models being graded.
- **Long outputs get cut off.** The default `--max-tokens` is 32768. Verbose models at high effort can hit it, which is why `init` sets 64000 on the max arms. Truncated outputs are flagged in the report.
- **Effort values aren't validated against each provider.** The harness checks the global set of level names only. Whether a given OpenAI-compatible model accepts `xhigh` or `max` is up to that provider, and a rejection surfaces as an `error` record.
- **Anthropic caching and effort interact.** Changing effort between requests invalidates cached prefixes on Anthropic, so keep one effort level per arm.
- **⚠ `python` checks execute model-generated code on the host.** Run the harness inside a container or VM.
