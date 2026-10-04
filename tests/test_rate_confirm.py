"""The /rate confirmation line: row attribution made visible, mismatch flagged."""

from __future__ import annotations

from evalroute import flywheel
from evalroute import routing
from evalroute.flywheel import handle_rate

from tests.test_flywheel import home


def test_rate_confirmation_names_the_row(home):
    lane = routing._lane_by_id("dl-ml-research-engineering")
    flywheel.note_route("audit plan files re rl training", lane, "llm", 0.8,
                        facets=["long-doc", "domain-dlml"])
    flywheel.on_post_llm_call(session_id="s", model="z-ai/glm-5.3")
    out = handle_rate("pass --note good")
    assert "labeled:" in out
    assert "dl-ml-research-engineering" in out
    assert "audit plan files re rl training" in out
    assert "mismatch" not in out.lower()


def test_rate_flags_arm_mismatch(home):
    # /model in a process-global hook is a candidate, not a verified session join.
    lane = routing._lane_by_id("dl-ml-research-engineering")
    flywheel.note_route("fix the NaN loss in our GRPO run", lane, "llm", 0.7)
    flywheel.on_pre_command(command="model", args_raw="moonshotai/kimi-k3", session_key="s")
    out = handle_rate("pass")
    assert "session unverified" in out
    outcome = next(r for r in reversed(flywheel.read_labels()) if r["kind"] == "outcome")
    assert outcome["actual_model"] is None and outcome["arm_attribution"] == "unknown"
    assert "z-ai/glm-5.3" in out
    assert "moonshotai/kimi-k3" in out
    assert "switches unverified" in out


def test_rate_no_route_message(home):
    out = handle_rate("pass")
    assert "no route on record" in out
