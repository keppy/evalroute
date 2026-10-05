# Runner templates for other agent CLIs

`evalroute dispatch brief.md --runner <name-or-template>` spawns a worker on
the routed arm. A named runner is a template stored in
`dispatch._NAMED_RUNNERS`; a template is any string with placeholders.

Placeholders (substituted into already-split tokens, so spaces and quotes in
paths survive):

- `{model}` `{effort}` `{provider}` — the routed arm
- `{brief}` — path to the brief file
- `{brief_text}` — the brief's file contents, for CLIs that take the prompt inline
- `{indir}` — `--in` dir if given; a token that is exactly `{indir}` is
  dropped when no `--in` was passed

A template may omit `{effort}`: the card still records the routed effort and
the rate line still carries `--effort` (`arm_attribution` stays
caller-stated). Session capture is Hermes-shaped; other runners record
`session_id: null` in the sidecar.

## Verified

| name | template | verified how |
| --- | --- | --- |
| `hermes` (default) | `hermes chat -Q --oneshot -m {model} --provider {provider} --reasoning {effort} --query-file {brief}` | this repo's own spawn path; exercised by the test suite |
| `claude-code` | `claude -p --model {model} --effort {effort} --output-format json --no-session-persistence --max-turns {max_turns} --add-dir {indir} {brief_text}` | flags read from Claude Code 2.1.289 `-p` mode on the maintainer's machine (`~/.local/bin/claude.exe`), 2026-10-05: exit 0, one JSON object on stdout (`result`, `is_error`, `num_turns`, `total_cost_usd`, `session_id`, `terminal_reason`). Efforts accepted: `low medium high xhigh max` (`none`/`minimal` map to `low`). `--permission-mode` is deliberately not in the template — pass it via a `--runner` template override or Claude Code's own settings; never default to `bypassPermissions`. |

## Unverified sketches — flags NOT checked against the real CLIs

None of the CLIs below was present on the maintainer's machine at write time
(no `--help` was read). These are sketches; confirm the flags against the
CLI's own `--help` before relying on them, and move a runner to the verified
table only after that check.

```bash
# Codex CLI — non-interactive exec; effort mapping unclear (no --reasoning)
evalroute dispatch brief.md --runner "codex exec {brief_text}"

# Aider — one-shot message with the brief file as input
evalroute dispatch brief.md --runner "aider --model {model} --message {brief_text} {brief}"
```

Runners that omit `{indir}` never pass the extra input directory through.
