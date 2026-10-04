"""Tests for gonogo adjudication wiring (adjudicate.py).

gonogo itself may be absent in a test environment — the degrade-to-unverified
path is part of the contract, so tests exercise both when possible.
"""

from __future__ import annotations

import pytest

from evalroute import adjudicate


def _has_gonogo():
    return adjudicate._gonogo() is not None


def _mk_runs():
    """Two arms over 4 tasks: arm_a fails T4, arm_b fails T3 and T4."""
    runs = []
    for t in ("T1", "T2", "T3"):
        runs.append({"task": t, "model": "a", "sample": 0, "passed": True, "text": "x"})
        runs.append({"task": t, "model": "b", "sample": 0, "passed": t != "T3", "text": "x"})
    runs.append({"task": "T4", "model": "a", "sample": 0, "passed": False, "text": "x"})
    runs.append({"task": "T4", "model": "b", "sample": 0, "passed": False, "text": "x"})
    # ungraded + foreign-arm rows must be ignored
    runs.append({"task": "T1", "model": "a", "sample": 1, "passed": None, "text": "x"})
    runs.append({"task": "T1", "model": "zzz", "sample": 0, "passed": True, "text": "x"})
    return runs


def test_paired_outcomes_coverage_semantics():
    pairs = adjudicate._paired_arm_outcomes(_mk_runs(), "a", "b")
    assert pairs == {"T1": (True, True), "T2": (True, True),
                     "T3": (True, False), "T4": (False, False)}


def test_paired_outcomes_no_shared_tasks():
    runs = [{"task": "T1", "model": "a", "sample": 0, "passed": True, "text": "x"}]
    assert adjudicate._paired_arm_outcomes(runs, "a", "b") is None


@pytest.mark.skipif(not _has_gonogo(), reason="gonogo not installed")
def test_paired_verdict_distinguishable_direction():
    v = adjudicate.distinguishable(_mk_runs(), "a", "b")
    assert v is not None and "shared cases" in v


def test_route_stamp_degrades_without_pairs():
    runs = [{"task": "T1", "model": "a", "sample": 0, "passed": True, "text": "x"}]
    stamp = adjudicate.route_stamp(runs, "a", "b")
    assert "unverified" in stamp


def test_route_stamp_no_runner():
    assert adjudicate.route_stamp([], "a", None) == "no runner-up arm to compare"


@pytest.mark.skipif(not _has_gonogo(), reason="gonogo not installed")
def test_route_stamp_with_gonogo():
    stamp = adjudicate.route_stamp(_mk_runs(), "a", "b")
    assert stamp.startswith("gap vs runner-up:")
    assert "shared cases" in stamp


@pytest.mark.skipif(not _has_gonogo(), reason="gonogo not installed")
def test_observed_verdict_insufficient_evidence():
    v = adjudicate.observed_verdict(10, 10)  # 100% on 10 tasks, 90% target
    assert v is not None and "INSUFFICIENT_EVIDENCE" in v


@pytest.mark.skipif(not _has_gonogo(), reason="gonogo not installed")
def test_observed_verdict_assist_only():
    v = adjudicate.observed_verdict(4, 10)  # 40% on 10 tasks
    assert v is not None


def test_observed_verdict_no_gonogo_or_zero():
    assert adjudicate.observed_verdict(1, 0) is None
