#!/usr/bin/env python3
"""hermes-shim — evalroute's `api: "cmd"` adapter for Hermes agent-harness arms.

The evalroute harness (evalroute.py) runs each model arm like:

    hermes-shim --model {model} --effort {effort} --prompt {prompt_file} \
                --system {system_file} --usage-file {usage_file}

and expects (README "Agent harnesses"):
  - stdout: the final answer text
  - last stdout line: {"text": "...", "usage": {"inp", "out", "cache_read"}}

The shim wraps `hermes -z` (one-shot mode), which prints only the final
response and supports --usage-file. Everything about the agent run — tools,
memory, AGENTS.md, auto-bypassed approvals — is Hermes' own verified code
path. This script only:
  1. reads the prompt/system files the harness materialized,
  2. invokes hermes -z with model + reasoning effort,
  3. reads the usage report (written even on failure),
  4. prints the answer, then ONE final JSON line with text + usage.

Design notes (see evalroute/README.md "Agent harnesses"):
  - The JSON line duplicates the text, but that is the contract: the harness
    parses the LAST line and uses j["text"], so a broken hermes -z output
    (no report) still grades on the raw stdout text before it.
  - Usage keys map: inp <- input_tokens, out <- output_tokens,
    cache_read <- cache_read_tokens (all present in the -z usage report).
  - reasoning_tokens are part of output_tokens in the report; the harness
    bills them at the output price, matching how evalroute bills cmd arms.
  - Non-zero hermes exit: the harness raises on non-zero cmd return codes, so
    we still print the JSON line with the error in "error" and exit 0, letting
    the harness record an error row rather than a crash (matches the
    "error record" path in evalroute.py `call`).
  - Deliberately stdlib-only: no openai/anthropic/yaml imports.

Usage standalone:
  python hermes-shim.py --model z-ai/glm-5.3 --effort max \
      --prompt prompt.txt [--system system.txt] [--usage-file u.json] [--timeout 3600]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="evalroute cmd-api shim around `hermes -z`")
    ap.add_argument("--model", required=True)
    ap.add_argument("--effort", required=True,
                    help="reasoning effort: none|minimal|low|medium|high|xhigh|max (or 'default' to omit)")
    ap.add_argument("--prompt", required=True, help="path to prompt file (from evalroute {prompt_file})")
    ap.add_argument("--system", default=None, help="path to system file (from evalroute {system_file})")
    ap.add_argument("--usage-file", default=None,
                    help="where hermes -z should write its JSON usage report (default: a temp file)")
    ap.add_argument("--hermes", default=None,
                    help="path to the hermes executable (default: PATH lookup)")
    ap.add_argument("--timeout", type=int, default=3600, help="seconds before the agent run is killed")
    a = ap.parse_args()

    prompt = Path(a.prompt).read_text(encoding="utf-8")
    system = Path(a.system).read_text(encoding="utf-8") if a.system else None

    # Compose the full user content: system (if any) prepended, then the prompt.
    # evalroute sends system + prompt as separate files; hermes -z takes one text.
    full_prompt = (f"{system}\n\n{prompt}" if system and system.strip() else prompt)

    usage_path = a.usage_file
    tmpdir = None
    if not usage_path:
        tmpdir = tempfile.mkdtemp(prefix="evalroute-shim-")
        usage_path = str(Path(tmpdir) / "usage.json")
    usage_file = Path(usage_path)
    usage_file.unlink(missing_ok=True)  # a stale report from an earlier run must not be billed

    hermes = a.hermes or shutil.which("hermes") or shutil.which("hermes.exe")
    if not hermes:
        print("hermes-shim: `hermes` not on PATH; install Hermes first (or pass --hermes)", file=sys.stderr)
        return 2
    hermes_argv = [hermes]
    if str(hermes).lower().endswith(".py"):
        # A python script (tests, or a user's custom wrapper): exec via this
        # interpreter (Windows can't exec a .py directly).
        hermes_argv = [sys.executable, hermes]

    # `-z` is the only transport that writes a usage report (the `chat`
    # --query-file path has no --usage-file wiring). Multi-line prompt text is
    # safe here: list-argv subprocess passes embedded newlines verbatim (only
    # a .cmd/.batch wrapper would truncate — see tests/test_shim.py).
    cmd = hermes_argv + ["-z", full_prompt, "-m", a.model, "--usage-file", str(usage_file)]
    if a.effort and a.effort != "default":
        cmd += ["--reasoning", a.effort]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=a.timeout,
        )
    except subprocess.TimeoutExpired:
        print(json.dumps({"text": "", "usage": {"inp": 0, "out": 0, "cache_read": 0},
                          "error": f"timeout after {a.timeout}s"}))
        return 0
    except OSError as exc:
        print(json.dumps({"text": "", "usage": {"inp": 0, "out": 0, "cache_read": 0},
                          "error": f"spawn failed: {exc}"}))
        return 0

    answer = (proc.stdout or "").strip()
    usage = {"inp": 0, "out": 0, "cache_read": 0}
    if usage_file.exists():
        try:
            rep = json.loads(usage_file.read_text(encoding="utf-8"))
            usage = {
                "inp": rep.get("input_tokens") or 0,
                "out": rep.get("output_tokens") or 0,
                "cache_read": rep.get("cache_read_tokens") or 0,
                # extra keys the harness's cost() will ignore harmlessly:
                "cache_write": rep.get("cache_write_tokens") or 0,
                "reasoning": rep.get("reasoning_tokens") or 0,
                "cost_usd": rep.get("estimated_cost_usd"),
            }
        except (json.JSONDecodeError, OSError):
            usage = {"inp": 0, "out": 0, "cache_read": 0}

    # Non-zero hermes exit: report as an error row (harness records error
    # records and retries on the next `run`), but still emit the JSON line so
    # nothing downstream needs to special-case a missing last line.
    err = None
    if proc.returncode != 0:
        err = f"hermes -z exit {proc.returncode}: {(proc.stderr or '').strip()[:300]}"

    if tmpdir:
        # best-effort cleanup of the shim-owned temp report
        try:
            usage_file.unlink(missing_ok=True)
            Path(tmpdir).rmdir()
        except OSError:
            pass

    # The contract: answer text on stdout, then ONE JSON line last.
    if answer and not err:
        print(answer)
    print(json.dumps({"text": answer, "usage": usage, **({"error": err} if err else {})}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
