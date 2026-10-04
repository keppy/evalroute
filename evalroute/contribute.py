"""`evalroute contribute`: opt-in, redacted outcome rows to the flywheel dataset.

Privacy model (whitelist, not blacklist — the same failure philosophy as
``routes_from_report.KEEP_FROM_EXISTING``):

- Only outcome rows are considered; ``route`` rows carry task text and never
  leave the machine, nor do ``model_switch`` / ``effort_switch`` /
  ``lane_correction`` / ``rating_correction`` kinds. A ``rating_correction``
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
    "week",
    "task_hash",
    "corrected",
    "schema",
)

SCHEMA_VERSION = 1

REPO_ID = "keppy/evalroute-flywheel"


# ------------------------------------------------------------------ gate


def _hermes_config_path() -> Optional[Path]:
    """The Hermes config.yaml, only when hermes_constants locates one."""
    try:
        import hermes_constants  # type: ignore

        home = getattr(hermes_constants, "get_hermes_home", None)
        if callable(home):
            return Path(home()) / "config.yaml"
    except Exception:
        return None
    return None


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
            "week": _week(float(out.get("ts", 0))),
            "task_hash": _task_hash(route["task"], salt),
            "corrected": corrected,
            "schema": SCHEMA_VERSION,
        }
        rows.append((float(out.get("ts", 0)), row))
    rows.sort(key=lambda p: p[0])  # ts order: the cursor is a ts, so uploads stay monotonic
    rows = [row for _, row in rows]
    summary = {
        "n_rows": len(rows),
        "dropped_no_route": dropped_no_route,
        "by_rated": dict(Counter(r["rated"] for r in rows)),
        "by_lane": dict(Counter(r["route_lane"] for r in rows)),
        "by_arm": dict(Counter(f'{r["route_model"]}@{r["route_effort"]}' for r in rows)),
        "cursor": cursor if cursor is not None else read_cursor(),
        "schema": SCHEMA_VERSION,
    }
    return rows, summary


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
        path_or_path_in_repo=path_in_repo,
        path_or_fileobj=payload,
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
    eligible = [o for o in records if o.get("kind") == "outcome"
                and float(o.get("ts", 0)) > cursor]
    last_ts = max((float(o["ts"]) for o in eligible), default=cursor)
    rows, summary = redact_report(records, salt, cursor=cursor)
    summary["salt"] = state["salt"]

    if dry_run:
        if as_json:
            print(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False))
        else:
            print(f"would upload {summary['n_rows']} redacted outcome rows "
                  f"(dropped_no_route: {summary['dropped_no_route']}; cursor: {summary['cursor']}; "
                  f"salt: {summary['salt']})")
            for label, key in (("by_rated", "by_rated"), ("by_lane", "by_lane"),
                               ("by_arm", "by_arm")):
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
