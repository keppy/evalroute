"""`evalroute contribute`: opt-in, redacted outcome rows to the flywheel dataset.

Privacy model (whitelist, not blacklist — the same failure philosophy as
``routes_from_report.KEEP_FROM_EXISTING``):

- Only outcome rows are considered; ``route`` rows carry task text and never
  leave the machine, nor do ``model_switch`` / ``effort_switch`` /
  ``rating_correction`` kinds. ``lane_correction`` rows ship as lane pairs
  only — ``from_lane`` / ``to_lane`` plus the classifier ``method`` and week;
  never any task text (a correction is a fact about the lane descriptions
  that pools across installs without text). A ``rating_correction``
  that consumed an outcome's id is folded in *before* redaction as a verdict
  override (the outcome's ``rated`` becomes the correction's and
  ``corrected: true`` is set); the correction row itself is not uploaded.
- Every kept key is on an explicit list. A new ledger key must never leak by
  default; adding one is a schema bump.
- Task text is replaced by an HMAC-SHA256 hex digest under a per-install salt
  (``<home>/evalroute/contribute.salt``, 32 random bytes, 0600 where the OS
  supports it) so the same task rateable across installs/weeks is
  countable without ever being readable. The salt is never printed and
  never uploaded. ``--rotate-salt`` regenerates it and resets the cursor.
- A cursor (``<home>/evalroute/contribute.cursor``) holds the ``ts`` of the
  last uploaded row; rows at or below it are skipped on the next upload.

The gate is explicit and off by default. Order:
``EVALROUTE_CONTRIBUTE=1`` env (a non-Hermes convenience; Hermes users set
the config key) > ``evalroute.contribute: true`` in the Hermes config.yaml >
``{"contribute": true}`` in ``<home>/evalroute/config.json``. Without the
gate ``contribute`` (not ``--dry-run``) exits 2 with the one-line fix;
``--dry-run`` always works — you may always see what *would* go.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .paths import hermes_home

SALT_NAME = "contribute.salt"
CURSOR_NAME = "contribute.cursor"
CONFIG_NAME = "config.json"

#: The exact keys an uploaded row carries — and only these.
KEPT = (
    "kind",
    "route_lane",
    "route_model",
    "route_effort",
    "actual_model",
    "actual_effort",
    "arm_attribution",
    "method",
    "confidence",
    "rated",
    "facets",
    "max_turns",
    "harness",
    "week",
    "task_hash",
    "corrected",
    "from_lane",
    "to_lane",
    "schema",
)

SCHEMA_VERSION = 2

REPO_ID = "keppy/evalroute-flywheel"


# ------------------------------------------------------------------ gate


def _hermes_config_path() -> Optional[Path]:
    """`config.yaml` in the resolved home — the same home the ledger lives in.

    Resolved through ``paths.hermes_home()`` (``EVALROUTE_HOME`` > ``HERMES_HOME``
    > ``hermes_constants`` > ``~/.hermes``) so the standalone console script,
    which has no ``hermes_constants`` on its path, still finds the Hermes
    config a user was told to edit. A standalone install simply has no such
    file and falls through to ``config.json``.
    """
    return hermes_home() / "config.yaml"


def contribute_enabled() -> bool:
    """The opt-in gate. Off by default; env > Hermes config.yaml > config.json."""
    if os.environ.get("EVALROUTE_CONTRIBUTE") == "1":
        return True
    cfg = _hermes_config_path()
    if cfg is not None and cfg.exists():
        try:
            import yaml

            data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
            return bool((data.get("evalroute") or {}).get("contribute"))
        except Exception:
            return False
    local = hermes_home() / "evalroute" / CONFIG_NAME
    if local.exists():
        try:
            data = json.loads(local.read_text(encoding="utf-8"))
            return bool(data.get("contribute"))
        except Exception:
            return False
    return False


def _gate_hint() -> str:
    """One-line fix, addressed to the surface that is actually active."""
    if _hermes_config_path() is not None:
        return ("evalroute: contribute is off (no outbound data without opt-in); "
                "set `evalroute.contribute: true` in config.yaml to enable")
    return ("evalroute: contribute is off (no outbound data without opt-in); "
            "set EVALROUTE_CONTRIBUTE=1 (non-Hermes convenience) or "
            f'{{"contribute": true}} in {hermes_home() / "evalroute" / CONFIG_NAME}')


# ------------------------------------------------------------------ salt / cursor


def salt_path() -> Path:
    return hermes_home() / "evalroute" / SALT_NAME


def cursor_path() -> Path:
    return hermes_home() / "evalroute" / CURSOR_NAME


def _ensure_salt(state: dict[str, Any]) -> bytes:
    """32 random bytes; created on first contribute (dry-run included)."""
    p = salt_path()
    if p.exists():
        state["salt"] = "present"
        return bytes.fromhex(p.read_text(encoding="ascii").strip())
    p.parent.mkdir(parents=True, exist_ok=True)
    salt = secrets.token_bytes(32)
    p.write_text(salt.hex() + "\n", encoding="ascii")
    try:  # 0600 where the OS supports it
        p.chmod(0o600)
    except Exception:
        pass
    state["salt"] = "created"
    return salt


def rotate_salt() -> None:
    """New salt, empty cursor: previously-uploaded hashes can't be extended."""
    p = salt_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    salt = secrets.token_bytes(32)
    p.write_text(salt.hex() + "\n", encoding="ascii")
    try:
        p.chmod(0o600)
    except Exception:
        pass
    c = cursor_path()
    if c.exists():
        c.unlink()


def read_cursor() -> float:
    try:
        return float(cursor_path().read_text(encoding="ascii").strip())
    except Exception:
        return 0.0


# ------------------------------------------------------------------ redaction


def _task_hash(task: str, salt: bytes) -> str:
    return hmac.new(salt, task.encode("utf-8"), hashlib.sha256).hexdigest()


def _week(ts: float) -> str:
    d = datetime.fromtimestamp(ts, timezone.utc)
    iso = d.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


def _facet_counts(value: Any) -> dict[str, int]:
    """Counts only: a list of ids, or a dict whose int values survive."""
    if isinstance(value, dict):
        return {str(k): v for k, v in value.items() if isinstance(v, int) and not isinstance(v, bool)}
    if isinstance(value, (list, tuple)):
        return dict(Counter(str(v) for v in value))
    return {}


def _consumed_key(record: dict[str, Any]) -> Any:
    return record.get("consumes_id") or record.get("consumes")


def redact_report(records: list[dict[str, Any]], salt: bytes,
                  cursor: Optional[float] = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """(redacted outcome rows, summary) from a ledger record list. Pure."""
    routes_by_id: dict[str, dict[str, Any]] = {}
    routes_by_ts: dict[float, dict[str, Any]] = {}
    for r in records:
        if r.get("kind") == "route":
            if r.get("id"):
                routes_by_id[r["id"]] = r
            if r.get("ts") is not None:
                routes_by_ts[r["ts"]] = r
    corrections: dict[str, dict[str, str]] = {"id": {}, "ts": {}}
    for c in records:
        if c.get("kind") == "rating_correction" and c.get("rated"):
            if c.get("consumes_id"):
                corrections["id"][c["consumes_id"]] = c["rated"]
            if c.get("consumes") is not None:
                corrections["ts"][c["consumes"]] = c["rated"]

    rows: list[dict[str, Any]] = []
    dropped_no_route = 0
    for out in records:
        if out.get("kind") != "outcome":
            continue
        if cursor is not None and float(out.get("ts", 0)) <= cursor:
            continue
        route = routes_by_id.get(out.get("consumes_id"))
        if route is None:
            route = routes_by_ts.get(out.get("consumes"))
        if route is None or not (route.get("task") or "").strip():
            dropped_no_route += 1
            continue
        rated = out.get("rated")
        corrected = False
        fix = (corrections["id"].get(out.get("consumes_id"))
               or corrections["ts"].get(out.get("consumes")))
        if fix is not None:
            rated, corrected = fix, True
        row: dict[str, Any] = {
            "kind": "outcome",
            "route_lane": out.get("route_lane") or route.get("lane"),
            "route_model": out.get("route_model") or route.get("model"),
            "route_effort": out.get("route_effort") or route.get("effort"),
            "actual_model": out.get("actual_model"),
            "actual_effort": out.get("actual_effort"),
            "arm_attribution": ("explicit_user" if (out.get("actual_model") and out.get("actual_effort"))
                                else "unknown"),
            "method": out.get("method"),
            "confidence": round(float(out.get("confidence", 0)), 2),
            "rated": rated,
            "facets": _facet_counts(out.get("facets") or route.get("facets")),
            "max_turns": out.get("max_turns"),
            # Old rows predate the key; hermes was the only harness then, so
            # the default is written in rather than left missing.
            "harness": out.get("harness") or "hermes",
            "week": _week(float(out.get("ts", 0))),
            "task_hash": _task_hash(route["task"], salt),
            "corrected": corrected,
            "schema": SCHEMA_VERSION,
        }
        rows.append((float(out.get("ts", 0)), row))
    rows.sort(key=lambda p: p[0])  # ts order: the cursor is a ts, so uploads stay monotonic
    rows = [row for _, row in rows]

    # Lane corrections ship as lane pairs only — never the task text.
    # A correction row is written right after the route it corrected, so the
    # classifier method is recovered from the last route seen before it.
    correction_rows: list[tuple[float, dict[str, Any]]] = []
    last_route: dict[str, Any] | None = None
    for corr in records:
        if corr.get("kind") == "route":
            last_route = corr
            continue
        if corr.get("kind") != "lane_correction":
            continue
        if cursor is not None and float(corr.get("ts", 0)) <= cursor:
            continue
        row = {
            "kind": "lane_correction",
            "from_lane": corr.get("from_lane"),
            "to_lane": corr.get("to_lane"),
            "method": (last_route or {}).get("method"),
            "week": _week(float(corr.get("ts", 0))),
            "schema": SCHEMA_VERSION,
        }
        correction_rows.append((float(corr.get("ts", 0)), row))
    correction_rows.sort(key=lambda p: p[0])
    all_rows = sorted(rows + [row for _, row in correction_rows],
                      key=lambda r: (r["kind"] != "lane_correction",))
    by_correction: dict[str, int] = {}
    for _, c in correction_rows:
        pair = f"{c['from_lane']}->{c['to_lane']}"
        by_correction[pair] = by_correction.get(pair, 0) + 1

    def _arm_key(r: dict[str, Any]) -> str:
        # The harness is part of the arm; hermes rows keep the bare
        # model@effort form so existing keys (and their consumers) are stable.
        if r.get("harness") and r["harness"] != "hermes":
            return f'{r["route_model"]}@{r["route_effort"]}@{r["harness"]}'
        return f'{r["route_model"]}@{r["route_effort"]}'

    summary = {
        "n_rows": len(rows),
        "dropped_no_route": dropped_no_route,
        "by_rated": dict(Counter(r["rated"] for r in rows)),
        "by_lane": dict(Counter(r["route_lane"] for r in rows)),
        "by_arm": dict(Counter(_arm_key(r) for r in rows)),
        "by_harness": dict(Counter(r.get("harness") or "hermes" for r in rows)),
        "n_corrections": len(correction_rows),
        "by_correction": by_correction,
        "cursor": cursor if cursor is not None else read_cursor(),
        "schema": SCHEMA_VERSION,
    }
    return all_rows, summary


def redact(rows: list[dict[str, Any]], salt: bytes) -> list[dict[str, Any]]:
    """Redacted outcome rows only (see the module docstring for the model)."""
    return redact_report(rows, salt)[0]


# ------------------------------------------------------------------ upload


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return "".join(json.dumps(r, ensure_ascii=False, sort_keys=False) + "\n" for r in rows).encode("utf-8")


def _get_hfapi():
    try:
        import huggingface_hub
    except Exception:
        print("evalroute contribute needs the huggingface_hub package: "
              'pip install "evalroute[hub]"', file=sys.stderr)
        return None
    return huggingface_hub.HfApi()


def _upload(api: Any, rows: list[dict[str, Any]], payload: bytes,
            repo_id: str) -> tuple[str, str]:
    """One JSONL file per upload under contributed/<contributor>/<utc-ts>.jsonl.

    The HF token is read by huggingface_hub from its own store; evalroute
    never sees it. Returns (path-in-repo, commit sha).
    """
    who = api.whoami().get("name") or "anonymous"
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path_in_repo = f"contributed/{who}/{ts}.jsonl"
    commit = api.upload_file(
        path_or_fileobj=payload,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"contribute {len(rows)} redacted outcome rows",
    )
    return path_in_repo, commit


def run(dry_run: bool = False, rotate: bool = False, repo_id: str = REPO_ID,
        as_json: bool = False) -> int:
    """argparse/handler entry for `evalroute contribute`."""
    from . import flywheel

    state: dict[str, Any] = {}
    salt = _ensure_salt(state)
    if rotate:
        rotate_salt()  # also removes the cursor
        salt, state["salt"] = bytes.fromhex(salt_path().read_text(encoding="ascii").strip()), "created"
    cursor = 0.0 if rotate else read_cursor()
    records = flywheel.read_labels()
    eligible = [o for o in records if o.get("kind") in ("outcome", "lane_correction")
                and float(o.get("ts", 0)) > cursor]
    last_ts = max((float(o["ts"]) for o in eligible), default=cursor)
    rows, summary = redact_report(records, salt, cursor=cursor)
    summary["salt"] = state["salt"]

    if dry_run:
        if as_json:
            print(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False))
        else:
            print(f"would upload {summary['n_rows']} redacted outcome rows and "
                  f"{summary['n_corrections']} lane-correction rows "
                  f"(dropped_no_route: {summary['dropped_no_route']}; cursor: {summary['cursor']}; "
                  f"salt: {summary['salt']})")
            for label, key in (("by_rated", "by_rated"), ("by_lane", "by_lane"),
                               ("by_arm", "by_arm"), ("by_harness", "by_harness"),
                               ("by_correction", "by_correction")):
                bits = ", ".join(f"{k}={v}" for k, v in summary[key].items())
                print(f"  {label}: {bits or '(none)'}")
            print("-- exact redacted rows below (grep before you send) --")
            sys.stdout.flush()
            sys.stdout.buffer.write(_jsonl_bytes(rows))
        return 0

    if not contribute_enabled():
        print(_gate_hint())
        return 2
    if not rows:
        print("nothing to contribute: every outcome is already uploaded (cursor) "
              "or had no route on file")
        return 0
    payload = _jsonl_bytes(rows)
    api = _get_hfapi()
    if api is None:
        return 1
    try:
        path_in_repo, commit = _upload(api, rows, payload, repo_id)
    except Exception as exc:
        print(f"evalroute: upload failed (nothing sent, cursor not advanced): {exc}")
        return 1
    cursor_path().parent.mkdir(parents=True, exist_ok=True)
    cursor_path().write_text(str(last_ts) + "\n", encoding="ascii")
    print(f"uploaded {len(rows)} rows → {repo_id}/{path_in_repo} @ {commit}")
    return 0
