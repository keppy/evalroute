"""report theme: dark projector default, --theme light, Now strip, provenance."""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from evalroute import flywheel, report, routing


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    flywheel._MEMORY.clear()
    routing.reset_routes_cache()
    yield tmp_path
    flywheel._MEMORY.clear()


@pytest.fixture
def demo_args(home):
    # probe only via --demo: never touches a real ledger
    return SimpleNamespace(out=str(home / "report.html"), json=False,
                           open=False, watch=None, trains=None,
                           factory_json=None, demo=True, theme=None)


@pytest.fixture
def theme_world(home):
    """One lane of each provenance kind in the tally."""
    lane_m = routing._lane_by_id("routine-coding")       # measured
    lane_p = routing._lane_by_id("math-first-principles")  # priors
    lane_o = routing._lane_by_id("prose")                # observed
    flywheel.note_route("t1", lane_m, "rules-strong", 0.9)
    flywheel.handle_rate("pass --model m/a --effort medium --note t1")
    rid = flywheel.note_route("t2", lane_p, "rules-weak", 0.5)
    flywheel.handle_rate("pass --model m/b --effort high --note t2")
    flywheel.note_route("t3", lane_o, "llm", 0.6)
    flywheel.handle_rate("fail --model m/c --effort high --note t3")
    return SimpleNamespace(home=home, id=rid)


def _render_html(world, **overrides):
    defaults = dict(out=str(world.home / "r.html"), json=False, open=False,
                    watch=None, factory_json=None, demo=False, theme=None)
    defaults.update(overrides)
    args = SimpleNamespace(**defaults)
    rc = report.run(args)
    return rc, Path(defaults["out"]).read_text(encoding="utf-8")


def _self_contained(html_text):
    assert "<script src=" not in html_text
    assert "<link href=" not in html_text
    assert "http://" not in html_text and "https://" not in html_text


@pytest.mark.parametrize("theme", [None, "dark", "light"])
def test_both_themes_render_and_sections(theme_world, theme):
    rc, html_text = _render_html(theme_world, theme=theme)
    assert rc == 0
    _self_contained(html_text)
    for sid in ("now", "outcomes", "tally", "methods", "sessions", "trains"):
        assert f"id='{sid}'" in html_text or f"id=\"{sid}\"" in html_text, sid


def test_dark_is_default_and_light_flag(theme_world):
    _, dark = _render_html(theme_world)
    assert "#0e1116" in dark
    _, light = _render_html(theme_world, theme="light")
    assert "#fafafa" in light
    assert "#0e1116" not in light


def test_theme_palettes_same_structure(theme_world):
    _, dark = _render_html(theme_world)
    _, light = _render_html(theme_world, theme="light")
    rules = lambda s: re.findall(r"^([a-z0-9 .,:#>'\"-]+?) \{", s, re.M)  # noqa: E731
    assert rules(dark) == rules(light)


def test_demo_json_now_block_populated(demo_args, home):
    report.run(demo_args)
    args = SimpleNamespace(**{**demo_args.__dict__, "json": True})
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report.run(args)
    import json as jsonlib
    data = jsonlib.loads(buf.getvalue())
    now = data["now"]
    for key in ("routes_today", "pending", "last_outcome",
                "cheapest_arm_this_week"):
        assert now.get(key), key


def test_demo_html_cheapest_cell(demo_args):
    rc = report.run(demo_args)
    assert rc == 0
    html_text = Path(demo_args.out).read_text(encoding="utf-8")
    assert "cheapest arm this week" in html_text
    cell = re.search(
        r"cheapest arm this week</div><div class='v'>([^<]+)</div>", html_text)
    assert cell and "$" in cell.group(1)


def test_tally_provenance_colours(theme_world):
    _, html_text = _render_html(theme_world)
    assert "prov-measured" in html_text and "prov-priors" in html_text \
        and "prov-observed" in html_text
    assert "prov{ color" not in html_text  # spans, not bare class names
    for colour in ("#5fd38d", "#e0b25a", "#8a93a6"):
        assert colour in html_text


def test_demo_badge_red_in_both_themes(home, demo_args):
    report.run(demo_args)
    demo_html = Path(demo_args.out).read_text(encoding="utf-8")
    assert "#b3261e" in demo_html
    args2 = SimpleNamespace(**{**demo_args.__dict__, "theme": "light",
                               "out": str(home / "r2.html")})
    report.run(args2)
    assert "#b3261e" in Path(str(home / "r2.html")).read_text(encoding="utf-8")
