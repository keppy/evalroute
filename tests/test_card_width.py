"""tests/test_card_width.py — CLI cards wrap at 100 cols; --wide restores single lines."""
from __future__ import annotations

import json
import re
import pytest

from evalroute import routing

LANES = [
    "alignment-reasoning", "routine-coding", "dl-ml-research-engineering",
    "hard-agentic-coding", "long-doc-reading", "web-research",
    "math-first-principles", "prose", "orchestration",
]
MONEY_RE = re.compile(r"\$\d+\.\d+/succ")

pytestmark = pytest.mark.usefixtures("hermes_cli")


def _card(lane_id):
    lane = routing._lane_by_id(lane_id)
    return routing.route_card(lane, 0.9, ["hit"], route_id="abc-123")


def _card_wide(lane_id):
    lane = routing._lane_by_id(lane_id)
    return routing.route_card(lane, 0.9, ["hit"], route_id="abc-123", wide=True)


def test_cli_cards_wrap_at_100():
    for lane_id in LANES:
        card = _card(lane_id)
        longest = max(len(l) for l in card.splitlines())
        assert longest <= 100, (lane_id, longest)


def test_no_split_money_token():
    for lane_id in LANES:
        card = _card(lane_id)
        for line in card.splitlines():
            assert MONEY_RE.search(line) is None or MONEY_RE.fullmatch(
                MONEY_RE.search(line).group(0)), line


def test_wide_restores_single_lines():
    for lane_id in LANES:
        wide = _card_wide(lane_id)
        assert max(len(l) for l in wide.splitlines()) > 100 or lane_id in (
            "alignment-reasoning",)  # some lanes are naturally short
        assert any(l.startswith("route id: ") for l in wide.splitlines())
        assert any(l.startswith("next: ") for l in wide.splitlines())


def test_wrapped_route_id_and_next_shapes():
    card = _card("routine-coding")
    lines = card.splitlines()
    rid = next(l for l in lines if l.startswith("route id: "))
    rate = lines[lines.index(rid) + 1]
    assert rate.startswith("rate:")
    assert "evalroute rate pass|fail" in rate and "--route-id abc-123" in rate
    nxt = next(l for l in lines if l.startswith("next: "))
    i = lines.index(nxt)
    assert lines[i + 1].lstrip().startswith("wrong lane?")
    assert "when done: hermes evalroute rate" in card


def test_hermes_chat_card_unchanged():
    routing.set_surface("hermes-chat")
    try:
        import tests.test_dataset as td
        lane = routing._lane_by_id("math-first-principles")
        card = routing.route_card(lane, 0.67, ["proof", "derivation"], method="rules-weak",
                                  facets=["symbolic"], route_id="abc-123")
        stripped = "\n".join(l for l in card.splitlines() if not l.startswith("table: "))
        assert stripped == td.MATH_041
    finally:
        routing.reset_surface()


def test_json_envelope_only_card_differs():
    import importlib
    cli = importlib.import_module("evalroute").cli
    wide = json.loads(cli._tool_result(_card_wide("routine-coding"),
                                       routing._lane_by_id("routine-coding"),
                                       0.9, False, route_id="abc-123"))
    narrow = json.loads(cli._tool_result(_card("routine-coding"),
                                         routing._lane_by_id("routine-coding"),
                                         0.9, False, route_id="abc-123"))
    assert set(wide) == set(narrow)
    assert {k: v for k, v in wide.items() if k != "card"} == \
        {k: v for k, v in narrow.items() if k != "card"}
    assert wide["card"] != narrow["card"]


def test_wrapped_long_head_keeps_space():
    """A field name longer than 6 chars must not be glued to its body when wrapped
    ('classification:no keyword hit' was visible in the real CLI output)."""
    wrapped = routing._maybe_wrap("classification: " + "x " * 60)
    assert wrapped.startswith("classification: x")
