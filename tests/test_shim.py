"""Tests for the hermes-shim (cmd-api adapter around `hermes -z`).

These test the shim's plumbing with a FAKE hermes executable — no real agent
run, no tokens, no credentials. The contract under test is the JSON last-line
+ stdout shape evalroute's `cmd` api expects.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

from evalroute.runners import hermes_shim as _shim_mod
SHIM = Path(_shim_mod.__file__)

# A fake `hermes` that prints a canned answer and writes a canned usage report.
# Arg order mirrors what the shim really sends: hermes -z PROMPT -m MODEL --usage-file F [--reasoning E]
FAKE_HERMES = textwrap.dedent("""
    import json, sys
    args = sys.argv[1:]
    z = args.index("-z")
    prompt = args[z + 1]
    model = args[args.index("-m") + 1]
    usage_file = args[args.index("--usage-file") + 1]
    effort = args[args.index("--reasoning") + 1] if "--reasoning" in args else None
    json.dump({"input_tokens": 111, "output_tokens": 222, "cache_read_tokens": 33,
               "cache_write_tokens": 4, "reasoning_tokens": 50,
               "estimated_cost_usd": 0.012}, open(usage_file, "w"))
    print(f"ANSWER for {model} @ {effort}: {len(prompt)} chars")
""")

# A fake that fails: no usage file, non-zero exit, stderr noise.
FAILING_HERMES = textwrap.dedent("""
    import sys
    print("boom", file=sys.stderr)
    sys.exit(3)
""")


@pytest.fixture
def fake_hermes(tmp_path):
    # --hermes points the shim straight at a python script: no PATH games, no
    # .cmd wrapper (batch is line-oriented and truncates multi-line argv).
    script = tmp_path / "fake_hermes.py"
    script.write_text(FAKE_HERMES, encoding="utf-8")
    return script


@pytest.fixture
def failing_hermes(tmp_path):
    script = tmp_path / "failing_hermes.py"
    script.write_text(FAILING_HERMES, encoding="utf-8")
    return script


def _run_shim(tmp_path, *extra, prompt="do the thing", system=None,
              usage_arg=True, hermes=None):
    import subprocess
    pf = tmp_path / "prompt.txt"
    pf.write_text(prompt, encoding="utf-8")
    cmd = [sys.executable, str(SHIM), "--model", "z-ai/glm-5.3", "--effort", "max",
           "--prompt", str(pf)]
    if hermes:
        cmd += ["--hermes", hermes]
    if system:
        sf = tmp_path / "system.txt"
        sf.write_text(system, encoding="utf-8")
        cmd += ["--system", str(sf)]
    if usage_arg:
        uf = tmp_path / "usage.json"
        cmd += ["--usage-file", str(uf)]
    cmd += list(extra)
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=60)


def test_shim_contract_answer_then_json_last_line(tmp_path, fake_hermes):
    proc = _run_shim(tmp_path, hermes=fake_hermes)
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.strip().splitlines()
    assert len(lines) >= 2
    payload = json.loads(lines[-1])
    assert payload["text"].startswith("ANSWER for z-ai/glm-5.3 @ max:")
    assert payload["usage"]["inp"] == 111
    assert payload["usage"]["out"] == 222
    assert payload["usage"]["cache_read"] == 33
    assert "error" not in payload


def test_shim_forwards_effort_and_model(tmp_path, fake_hermes):
    proc = _run_shim(tmp_path, hermes=fake_hermes)
    assert "@ max" in proc.stdout  # --reasoning max reached the fake hermes


def test_shim_system_prepended_to_prompt(tmp_path, fake_hermes):
    # system (14 chars) + 2 newlines + "hello" (5) = 21 chars total prompt.
    proc = _run_shim(tmp_path, system="You are terse.", prompt="hello",
                     hermes=fake_hermes)
    lines = proc.stdout.strip().splitlines()
    payload = json.loads(lines[-1])
    assert payload["text"].endswith("21 chars"), payload["text"]


def test_shim_missing_usage_report_zeros(tmp_path):
    # Fake that prints an answer but does NOT write the usage file.
    fake = tmp_path / "no_usage.py"
    fake.write_text("import sys\nprint('ANSWER no-usage')\n", encoding="utf-8")
    proc = _run_shim(tmp_path, hermes=fake)
    assert proc.returncode == 0
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["text"] == "ANSWER no-usage"
    assert payload["usage"]["inp"] == 0 and payload["usage"]["out"] == 0


def test_shim_reports_nonzero_hermes_exit_as_error_row(tmp_path, failing_hermes):
    proc = _run_shim(tmp_path, hermes=failing_hermes)
    assert proc.returncode == 0  # shim exits 0 so the harness records an error row
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert "error" in payload and "3" in payload["error"]
    assert "boom" in payload["error"]
