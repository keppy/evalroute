"""Gonogo adjudication for evalroute: is a route flip statistically real?

Reuses the merged gonogo plugin's stats (McNemar paired comparison, Wilson
intervals, INSUFFICIENT-EVIDENCE decisions) so evalroute never treats a
pilot-scale point-estimate gap as a measurement. Import is lazy and optional:
without gonogo installed, every stamp degrades to "unverified".

Three stamps:
  distinguishable(winner, runner, per_task_outcomes) -> str|None
      McNemar on paired per-task outcomes: "gap: +0.10 [95% CI -0.19,+0.39],
      p=0.48 - NOT distinguishable at n=10" or "... distinguishable (p=0.03)".
  observed_verdict(passes, trials) -> str|None
      gonogo_decide semantics for flywheel observed rows: verdict + what
      sample size would be needed.
  route_stamp(lane, winner, runner, ...) -> str|None
      The provenance line suffix for a measured lane.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _gonogo():
    """Import gonogo lazily; None when not installed (stamps degrade)."""
    try:
        import gonogo as mod
        return mod
    except Exception as exc:  # pragma: no cover - depends on environment
        logger.debug("gonogo unavailable: %s", exc)
        return None


def _paired_arm_outcomes(runs: list[dict[str, Any]], arm_a: str, arm_b: str
                         ) -> Optional[dict[tuple[str, str], tuple[bool, bool]]]:
    """Per-task paired outcomes for two arms: {(task): (a_passed, b_passed)}.

    Uses graded samples only; a task with k>1 samples is a task-pass iff any
    sample passed (coverage semantics - same rule the report's cov uses).
    Tasks not present in BOTH arms are dropped (McNemar needs pairs).
    """
    per: dict[str, dict[str, list[bool]]] = {}
    for r in runs:
        if "text" not in r or r.get("passed") is None:
            continue
        per.setdefault(r["task"], {}).setdefault(r["model"], []).append(bool(r["passed"]))
    pairs = {t: (any(arms[arm_a]), any(arms[arm_b]))
             for t, arms in per.items() if arm_a in arms and arm_b in arms}
    return pairs or None


def distinguishable(runs: list[dict[str, Any]], arm_a: str, arm_b: str,
                    level: float = 0.95) -> Optional[str]:
    """McNemar verdict on whether arm_a really beats arm_b, or None if
    gonogo is unavailable or the arms share no task pairs."""
    g = _gonogo()
    if g is None:
        return None
    pairs = _paired_arm_outcomes(runs, arm_a, arm_b)
    if pairs is None:
        return None
    try:
        rep_a = {"cases": [{"id": t, "passed": a} for t, (a, _) in pairs.items()]}
        rep_b = {"cases": [{"id": t, "passed": b} for t, (_, b) in pairs.items()]}
        cmp = g.compare(rep_a, rep_b, level=level)
        return str(cmp)  # "B vs A (+x% [lo,hi], p=..) — verdict on N shared cases"
    except Exception as exc:
        logger.warning("gonogo compare failed: %s", exc)
        return None


def observed_verdict(passes: int, trials: int, target: float = 0.9) -> Optional[str]:
    """gonogo decide() semantics for an observed lane's pass rate.

    decide() takes per-trial (confidence, passed) tuples; flywheel outcomes
    carry no confidence signal, so every trial uses the same value and gonogo
    itself will note no abstention threshold can be derived. Honest.
    """
    g = _gonogo()
    if g is None or trials <= 0:
        return None
    results = [(0.5, bool(i < passes)) for i in range(trials)]
    try:
        d = g.decide(results=results, target=target, unit="tasks")
        verdict = getattr(d, "verdict", None)
        name = getattr(verdict, "name", str(verdict))
        reason = getattr(d, "reason", "")
        return f"gonogo {name}: {reason}" if reason else f"gonogo {name}"
    except Exception as exc:
        logger.warning("gonogo decide failed: %s", exc)
        return None


def route_stamp(runs: list[dict[str, Any]], winner: str, runner_up: Optional[str],
                level: float = 0.95) -> str:
    """Provenance suffix for a measured lane: was the winner's edge real?

    Never raises; degrades to "unverified" so provenance stays honest about
    what was actually adjudicated rather than silently omitting the check.
    """
    if runner_up is None:
        return "no runner-up arm to compare"
    stamp = distinguishable(runs, winner, runner_up, level)
    if stamp is None:
        return "gap unverified (gonogo absent or arms share no tasks)"
    return f"gap vs runner-up: {stamp}"
