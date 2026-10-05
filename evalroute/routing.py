"""Handlers for the evalroute plugin.

Strong-rule classification is local; weak-signal classification can call the
host LLM (token cost). The separately invoked harness and shim can make paid
provider calls; neither runs as a side effect of importing this module.
"""

from __future__ import annotations

import importlib.resources
import json
import logging
import re
import sys
import textwrap
from pathlib import Path
from typing import Any

import yaml

from . import classify_encoder  # noqa: E402
from .paths import hermes_home

logger = logging.getLogger(__name__)

ROUTES_FILE = importlib.resources.as_file(
    importlib.resources.files("evalroute").joinpath("data/routes.yaml")).__enter__()

# "low-medium" -> "medium" is a policy choice: the higher of a starting range,
# because reasoning_overrides has one slot per model and under-routing effort
# is the costlier miss (coverage, not verbosity).
_EFFORT_RE = re.compile(r"(none|minimal|low|medium|high|xhigh|max)")

_LLM_LANES = None  # lazy-loaded routes cache

# Which route table is in effect, set by _load_routes() whenever it resolves.
# {"kind": "bundled"} or {"kind": "dataset", "sha": ..., "generated": ...,
# "plugin_version": ...}.
ROUTES_SOURCE: dict[str, Any] = {"kind": "bundled"}

_REPO_ID = "keppy/evalroute-flywheel"


def _dataset_root() -> Path:
    """<hermes home>/evalroute/dataset (profile-safe; never inside the package)."""
    return hermes_home() / "evalroute" / "dataset"


def _validate_lanes(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Shared lanes validation for the bundled and dataset tables."""
    lanes = raw.get("lanes")
    if not isinstance(lanes, list) or not lanes:
        raise ValueError("routes table has no lanes list")
    seen: set[str] = set()
    for lane in lanes:
        for field in ("id", "label", "model", "effort"):
            if not lane.get(field):
                raise ValueError(f"lane entry missing required field {field!r}: {lane}")
        if lane["id"] in seen:
            raise ValueError(f"duplicate lane id {lane['id']!r}")
        seen.add(lane["id"])
    return lanes


def _dataset_current_sha() -> str | None:
    """The sha `current` names, or None (no pointer file)."""
    try:
        sha = (_dataset_root() / "current").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return sha or None


def reset_routes_cache() -> None:
    """Drop the lane cache so the next _load_routes() re-resolves the table."""
    global _LLM_LANES
    _LLM_LANES = None


def _load_routes() -> list[dict[str, Any]]:
    """Resolve the active route table: synced dataset first, bundled fallback.

    A synced table is only used when it parses and validates; anything else
    (missing, corrupt, bad yaml, duplicate ids) warns once on stderr and
    routes on the bundled table. Routing must always work.
    """
    global _LLM_LANES
    if _LLM_LANES is None:
        sha = _dataset_current_sha()
        if sha:
            ds_dir = _dataset_root() / sha
            try:
                raw = yaml.safe_load(
                    (ds_dir / "routes" / "routes.yaml").read_text(encoding="utf-8")) or {}
                _LLM_LANES = _validate_lanes(raw)
                manifest = json.loads((ds_dir / "routes" / "MANIFEST.json")
                                      .read_text(encoding="utf-8"))
                ROUTES_SOURCE.clear()
                ROUTES_SOURCE.update({
                    "kind": "dataset", "sha": sha,
                    "generated": manifest.get("generated"),
                    "plugin_version": manifest.get("plugin_version"),
                })
                return _LLM_LANES
            except Exception as exc:
                print(f"evalroute: synced route table at {ds_dir} unusable ({exc}); "
                      "using the bundled table", file=sys.stderr)
                reset_routes_cache()
        raw = yaml.safe_load(ROUTES_FILE.read_text(encoding="utf-8")) or {}
        ROUTES_SOURCE.clear()
        ROUTES_SOURCE.update({"kind": "bundled"})
        _LLM_LANES = _validate_lanes(raw)
    return _LLM_LANES


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


_PLUGIN_LLM = None  # stashed ctx.llm facade, set at register() time


def set_llm_facade(facade) -> None:
    """Stash the host-owned LLM facade from register(ctx). None = no fallback."""
    global _PLUGIN_LLM
    _PLUGIN_LLM = facade


# ------------------------------------------------------------------ surfaces
#
# One process-global surface, set by whoever invokes the handlers:
#   "cli"         the standalone console script (`evalroute ...`) — cli.main
#                 pins it before dispatching
#   "hermes-cli"  the plugin's register_cli_command handler (`hermes evalroute
#                 ...`) — cli.evalroute_cli claims it unless already claimed
#   "hermes-chat" the plugin's slash-command / tool handlers (/route, /rate)
# Every user-facing command string in the card and the dispatch rate line is
# rendered by cmd_rate / cmd_route_lane / cmd_switch_arm for that surface, so
# a reader with no Hermes gets `evalroute ...` commands it can actually run.
SURFACE: str = "cli"
_SURFACE_EXPLICIT = False  # main() pins "cli"; only then is it explicit
_SURFACES = ("cli", "hermes-cli", "hermes-chat")


def set_surface(name: str) -> None:
    """Declare the invoking surface (see contract.py — contract name #10)."""
    global SURFACE, _SURFACE_EXPLICIT
    if name not in _SURFACES:
        raise ValueError(f"unknown surface {name!r}; known surfaces: {', '.join(_SURFACES)}")
    SURFACE = name
    _SURFACE_EXPLICIT = True


def reset_surface() -> None:
    """Back to the standalone default, unclaimed (tests run in one process)."""
    global SURFACE, _SURFACE_EXPLICIT
    SURFACE = "cli"
    _SURFACE_EXPLICIT = False


def cmd_rate(route_id: str = "", model: str = "", effort: str = "",
             verdict: str = "pass|fail", note: str = "why") -> str:
    """The rate command, rendered for the active surface.

    chat keeps the slash form (model/effort are the session's /model and
    /reasoning there); cli and hermes-cli get the full terminal command with
    the arm confirmed inline. `note` is a placeholder or the given reason;
    empty note omits the flag (the card's route-id line has none).
    """
    if SURFACE == "hermes-chat":
        parts = f"/rate {verdict}"
        if route_id:
            parts += f" --route-id {route_id}"
        if note:
            parts += f" --note {note}"
        return parts
    prefix = "hermes evalroute" if SURFACE == "hermes-cli" else "evalroute"
    parts = f"{prefix} rate {verdict}"
    if route_id:
        parts += f" --route-id {route_id}"
    if model:
        parts += f" --model {model}"
    if effort:
        parts += f" --effort {effort}"
    if note:
        parts += f' --note "{note}"'
    return parts


def cmd_route_lane() -> str:
    """The wrong-lane reroute command for the active surface."""
    if SURFACE == "hermes-chat":
        return "/route --lane <id> <same task>"
    prefix = "hermes evalroute" if SURFACE == "hermes-cli" else "evalroute"
    return f'{prefix} route --lane <id> "<same task>"'


def cmd_switch_arm(model: str, effort: str) -> str:
    """How to get onto this arm, for the active surface.

    chat: /model (plus /reasoning unless the installed reasoning_overrides
    already carry this model's effort — see _effort_auto). cli/hermes-cli:
    the arm is stated inline; dispatch picks the model itself.
    """
    if SURFACE == "hermes-chat":
        if not _effort_auto({"model": model, "effort": effort}):
            return f"/model {model} then /reasoning {effort}"
        return f"/model {model}"
    return f"(run on {model} @ {effort})"


# Negation cues: a keyword hit preceded by one of these within a few words is
# not evidence for the lane. "not usually hard math though" must NOT count as
# a math hit — this exact phrasing shipped a life-assistant description to the
# math-first-principles route.
_NEGATION_CUES = {
    "not", "no", "never", "without", "except", "isn't", "isnt", "aren't", "arent",
    "don't", "dont", "doesn't", "doesnt", "avoid", "rarely", "seldom", "hardly",
    "unusual", "instead", "rather", "n't",
}
_NEG_WINDOW = 4  # words before the keyword to scan for a cue


def _hit(keyword: str, text: str) -> bool:
    """Word-boundary, negation-guarded match of a normalized keyword.

    Substring matching made 'rl' hit 'world'; boundaries kill that class of
    false positive. Negation guard kills 'not usually hard math' as a math
    signal. Multi-word phrases match as phrases ('unit test').
    """
    kw = _norm(keyword)
    if not kw:
        return False
    for m in re.finditer(rf"(?<!\S){re.escape(kw)}(?!\S)", text):
        before = text[:m.start()].split()
        if not any(w in _NEGATION_CUES for w in before[-_NEG_WINDOW:]):
            return True
    return False


def _lane_by_id(lane_id: str) -> dict[str, Any] | None:
    return next((l for l in _load_routes() if l["id"] == lane_id), None)


# Thresholds for escalating from rules to the host-owned LLM classifier.
# HIGH_CONF = rules are trusted outright. BELOW that, a lane with several
# distinct keyword hits is still trusted (evidence beats paraphrase); a single
# hit is not — that's where "researchy" descriptions with zero vocab signal
# lived, and where "not usually hard math" misrouted before the negation
# guard.
_LLM_HIGH_CONF = 0.75
_LLM_MIN_HITS = 2

_LLM_SCHEMA = {
    "type": "object",
    "properties": {
        "lane": {"type": "string"},
        "facets": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"},
    },
    "required": ["lane"],
}

_LLM_SYSTEM = (
    "You are a task router. Classify the user's task description into exactly "
    "one lane from the provided list. Judge by what the work IS, not by which "
    "words appear: a description of an assistant's duties is an ORCHESTRATION "
    "task (it plans and delegates); negated capabilities ('not usually hard "
    "math') are evidence AGAINST a lane, not for it. ALSO name the task's "
    "facets along the provided axes (a task may have several: e.g. reading "
    "many files AND judging RL training plans = long-doc + domain-dlml). "
    'Return JSON: {"lane": "<id>", "facets": ["<facet-id>", ...], '
    '"confidence": 0-1}.'
)


# ------------------------------------------------------------------ facets

def _load_facets() -> list[dict[str, Any]]:
    """data/facets.yaml; missing file -> no facet inference (not an error)."""
    ref = importlib.resources.files("evalroute").joinpath("data/facets.yaml")
    if not ref.is_file():
        return []
    try:
        return (yaml.safe_load(ref.read_text(encoding="utf-8")) or {}).get("facets") or []
    except Exception:
        return []


def facets_for_hits(hit_lanes: list[str]) -> list[str]:
    """Facets implied by the lanes that drew keyword hits (rules path)."""
    if not hit_lanes:
        return []
    return [f["id"] for f in _load_facets()
            if any(lid in hit_lanes for lid in f.get("lanes", []))]


def normalize_facets(raw: list[str]) -> list[str]:
    """Keep known facet ids only, deduped, stable order per facets.yaml."""
    known = {f["id"] for f in _load_facets()}
    return [fid for fid in dict.fromkeys(raw or []) if fid in known]


def facet_dominance(facet_ids: list[str]) -> str:
    """Describe coexistence, not an unimplemented arm precedence rule."""
    if len(facet_ids) <= 1:
        return ""
    return "descriptive conjunction; lane chooses arm"


def _llm_fallback(task: str, lanes: list[dict[str, Any]]
                  ) -> tuple[dict[str, Any] | None, float, list[str]]:
    """Classify via the host-owned LLM facade. Returns (lane, conf, facets).

    Failures (no facade, LLM error, unparseable/unknown lane) return
    (None, 0, []) and the caller falls through to the deterministic result.
    """
    if _PLUGIN_LLM is None:
        return None, 0.0, []
    try:
        lane_list = "\n".join(
            f'- {l["id"]}: {l["label"]} — {l.get("match_hint", "")}' for l in lanes)
        facet_list = "\n".join(
            f'- {f["id"]} ({f["axis"]}): {f["description"]}' for f in _load_facets())
        instructions = (
            "Classify this task description into exactly one lane.\n\n"
            f"Lanes:\n{lane_list}\n\nFacets (name all that apply):\n{facet_list}\n\nTask:\n{task}"
        )
        result = _PLUGIN_LLM.complete_structured(
            instructions=instructions,
            input=[{"type": "text", "text": task}],
            json_schema=_LLM_SCHEMA,
            schema_name="evalroute_lane",
            system_prompt=_LLM_SYSTEM,
            temperature=0.0,
            max_tokens=256,
            timeout=30.0,
        )
        parsed = getattr(result, "parsed", None) or {}
        lane_id = str(parsed.get("lane", "")).strip()
        conf = float(parsed.get("confidence", 0.0) or 0.0)
        facets = normalize_facets(parsed.get("facets") or [])
        lane = _lane_by_id(lane_id)
        if lane is None or conf <= 0:
            return None, 0.0, []
        return lane, min(1.0, conf), facets
    except Exception as exc:
        logger.warning("evalroute LLM fallback failed: %s", exc)
        return None, 0.0, []


def route_full(task: str) -> tuple[dict[str, Any], float, list[str], str, list[str]]:
    """route_for plus the task's facets (the label's extra dimensions).

    Facets come from the same evidence as the lane: rules path derives them
    from the lanes that drew hits (any lane with a hit is a facet claim);
    LLM path takes the classifier's facet list. Empty -> the winning LLM
    lane's facets, never facets inferred from the weak rule it overrode.
    """
    lanes = _load_routes()
    lane, conf, hits = classify(task)
    distinct = len(set(hits))
    hit_lane_ids = [l["id"] for l in lanes if any(_hit(kw, _norm(task)) for kw in l.get("keywords", []))]
    facets = facets_for_hits(hit_lane_ids)
    if distinct >= _LLM_MIN_HITS:
        return lane, conf, hits, "rules-strong", facets
    if classify_encoder.is_installed():
        enc_lane, enc_conf = classify_encoder.classify(task)
        if enc_lane is not None and enc_conf >= classify_encoder.threshold():
            return (_lane_by_id(enc_lane), enc_conf, hits, "encoder",
                    facets_for_hits([enc_lane]))
    llm_lane, llm_conf, llm_facets = _llm_fallback(task, lanes)
    if llm_lane is not None:
        return llm_lane, llm_conf, hits, "llm", (llm_facets or facets_for_hits([llm_lane["id"]]))
    if hits:
        return lane, conf, hits, "rules-weak", facets
    return lane, conf, hits, "default", facets


def route_for(task: str) -> tuple[dict[str, Any], float, list[str], str]:
    """Full routing decision: rules first, LLM fallback when rules are weak.

    Returns (lane, confidence, hits, method) where method is
    'rules-strong' | 'rules-weak' | 'llm' | 'default'. Rules are trusted only
    with >= _LLM_MIN_HITS DISTINCT keyword hits on the winning lane (a lone
    'proof' in 'proof of concept' is not evidence); confidence alone is not
    a strength signal (1 hit on 1 lane computes conf 1.0). Otherwise the
    host-owned LLM classifies (a paraphrase like 'manage life, writing, and
    researchy tasks' has zero keyword signal and needs it). LLM failure falls
    back to the weak rules result anyway.
    """
    lane, conf, hits, method, _facets = route_full(task)
    return lane, conf, hits, method


def classify(task: str) -> tuple[dict[str, Any], float, list[str]]:
    """Rules-first lane classification.

    Returns ``(lane, confidence, hits)``. Keyword hits score per lane; the
    highest total wins. Every lane is checked (a "proof ... blog post" goes to
    math, not prose, because math's keywords outrank on the stronger signal).
    No lane hits -> long-doc-reading with confidence 0.0 (the input-heavy
    default; the card says so). Confidence is hits[top]/hits[all], bounded to 1.
    """
    text = _norm(task)
    scores: dict[str, list[str]] = {}
    for lane in _load_routes():
        hits = [kw for kw in lane.get("keywords", []) if _hit(kw, text)]
        if hits:
            scores[lane["id"]] = hits
    if not scores:
        # Input-heavy default; a generated table may not contain that lane
        # (no keywords carried over), so fall back to the first lane rather
        # than returning None.
        fallback = _lane_by_id("long-doc-reading") or _load_routes()[0]
        return fallback, 0.0, []
    top_id = max(scores, key=lambda k: len(scores[k]))
    total = sum(len(h) for h in scores.values())
    conf = min(1.0, len(scores[top_id]) / total) if total else 0.0
    return _lane_by_id(top_id), conf, scores[top_id]


def _lib_version() -> str:
    """Version from pyproject.toml via the installed metadata (never breaks a card)."""
    try:
        from importlib.metadata import version
        return version("evalroute")
    except Exception:
        from . import __version__
        return __version__


def _table_line() -> str:
    """Card provenance for the table itself (bundled vs pinned dataset revision)."""
    _load_routes()  # ensure ROUTES_SOURCE reflects the table this card came from
    if ROUTES_SOURCE.get("kind") == "dataset":
        return (f"table: {_REPO_ID} @ {str(ROUTES_SOURCE.get('sha', ''))[:12]} "
                f"(generated {ROUTES_SOURCE.get('generated')}, "
                f"plugin {ROUTES_SOURCE.get('plugin_version')})")
    return f"table: bundled (evalroute {_lib_version()})"


def route_card(lane: dict[str, Any], conf: float, hits: list[str],
               pinned: bool = False, method: str = "rules",
               facets: list[str] | None = None, route_id: str | None = None,
               wide: bool = False) -> str:
    """Render the human-readable route card.

    On the CLI surfaces (not hermes-chat), long lines wrap at 100 columns
    unless `wide` asks for the one-line-per-field card for log scrapers.
    """
    lines = [
        f"lane: {lane['label']} ({lane['id']})",
        f"route: {lane['model']} @ {lane['effort']}"
        # chat only: the arm switch hint; on the CLI surfaces the arm is
        # already on this line and the next: footer carries the command.
        + (f"   <- run: /model {lane['model']}" if SURFACE == "hermes-chat" else ""),
    ]
    if facets and len(facets) > 1:
        lines.append(f"facets: {' + '.join(facets)} (descriptive conjunction; lane chooses arm)")
    elif facets:
        lines.append(f"facets: {facets[0]}")
    if lane.get("escalation"):
        lines.append(f"escalation: {lane['escalation']} (when coverage gaps or the task turns out harder)")
    if pinned:
        lines.append("classification: lane pinned by caller")
    elif method == "encoder":
        m = classify_encoder.load()
        calib = float((m or {}).get("metrics", {}).get("calib_accuracy", 0.0))
        lines.append(f"classification: encoder {calib:.2f} calib · conf {conf:.2f}")
    else:
        # An installed encoder that did not decide this route abstained (below
        # threshold or unknown label); say so on whichever fallback line follows.
        abstained = (" (encoder abstained)"
                     if classify_encoder.is_installed() and method != "rules-strong" else "")
        if method == "llm":
            lines.append(f"classification: LLM fallback ({conf:.2f}) - rules had weak signal "
                         f"({'no keyword hit' if not hits else 'single ambiguous hit'})"
                         f"{abstained}")
        elif hits:
            shown = ", ".join(sorted(set(hits))[:4])
            more = "" if len(set(hits)) <= 4 else f" (+{len(set(hits)) - 4} more)"
            lines.append(f"classification: rules match ({conf:.2f}) on: {shown}{more}{abstained}")
        else:
            lines.append("classification: no keyword hit - defaulted to long-doc-reading; "
                         f"pass --lane <id> to pin, or say the task in more words{abstained}")
    lines.append(f"basis: {lane.get('provenance', 'unknown')}")
    lines.append(_table_line())
    if lane.get("notes"):
        lines.append(f"note: {lane['notes']}")
    lines.append(f"why here: {lane.get('match_hint', '')}")
    if route_id:
        rate_cmd = cmd_rate(route_id=route_id, model=lane['model'],
                            effort=_effort_for_override(lane), note='')
        if _wrap_on() and not wide:
            lines.append(f"route id: {route_id}")
            # Break the rate command at the flag boundary ourselves so a flag never
            # lands on a different line from its value.
            head, sep, tail = rate_cmd.partition(" --model ")
            if sep and len(f"rate:     {rate_cmd}") > 100:
                lines.append(f"rate:     {head}")
                lines.append(f"          --model {tail}")
            else:
                lines.append(f"rate:     {rate_cmd}")
        else:
            lines.append(f"route id: {route_id} (use {rate_cmd} if routes overlap)")
    # Workflow footer: the card answers "what arm?", the footer answers
    # "what now?". Wrong lane -> fix it now (a --lane reroute re-logs the
    # assignment; /rate attributes to the LAST route on file).
    next_cmd = cmd_switch_arm(lane['model'], _effort_for_override(lane))
    wrong = f"wrong lane? {cmd_route_lane()}"
    done = f"when done: {cmd_rate(model=lane['model'], effort=_effort_for_override(lane), note=('' if _wrap_on() else 'why'))}"
    if _wrap_on() and not wide:
        lines.append(f"next:     {next_cmd.strip('()')}")
        indent = " " * len("next:     ")
        lines.append(f"{indent}{wrong}")
        lines.append(f"{indent}{done}")
    else:
        lines.append(f"next: {next_cmd} | {wrong} | {done}")
    return _maybe_wrap("\n".join(lines), wrap=not wide)


def _wrap_on() -> bool:
    """Wrapping applies only to the CLI surfaces; chat cards are pinned."""
    return SURFACE != "hermes-chat"


def _maybe_wrap(card: str, wrap: bool = True) -> str:
    """Wrap any line over 100 columns (CLI surfaces only).

    7-space hanging indent, no mid-token breaks: `$0.0001/succ` and model
    ids never split. `basis:`/`note:` are the fields that need it; the
    route-id/next lines are built wrapped above.
    """
    if not _wrap_on() or not wrap:
        return card
    out = []
    for line in card.splitlines():
        if len(line) <= 100:
            out.append(line)
            continue
        head, _, rest = line.partition(": ")
        pad = " " * max(1, 6 - len(head))  # never glue head to body
        out.extend(textwrap.wrap(
            f"{head}:{pad}{rest}", width=100,
            initial_indent="", subsequent_indent=" " * 7,
            break_long_words=False, break_on_hyphens=False) or [line])
    return "\n".join(out)


def _effort_auto(lane: dict[str, Any]) -> bool:
    """Only omit /reasoning when the active profile's model override matches.

    Another lane can share the model but need a lower effort. In that case
    /model applies the higher installed override, so an explicit command is
    required to reach this card's arm. Never write the config from a card.
    """
    try:
        config = yaml.safe_load((hermes_home() / "config.yaml").read_text(encoding="utf-8")) or {}
    except Exception:
        # no Hermes installed: HERMES_HOME must still win (tests set it);
        # no config file there -> not auto (footer keeps /reasoning)
        return False
    overrides = (config.get("agent") or {}).get("reasoning_overrides") or {}
    return isinstance(overrides, dict) and overrides.get(lane["model"]) == _effort_for_override(lane)


def _tool_result(card: str, lane: dict[str, Any], conf: float, pinned: bool,
                 method: str = "rules", route_id: str | None = None) -> str:
    """JSON envelope for the model-facing tool: card for the human, fields for the agent."""
    return json.dumps({
        "lane": lane["id"],
        "lane_label": lane["label"],
        "model": lane["model"],
        "effort": lane["effort"],
        "escalation": lane.get("escalation"),
        "confidence": round(conf, 2),
        "classification_method": method,
        "pinned": pinned,
        "route_id": route_id,
        # No provider field exists in routes.yaml; until the route table grows
        # one, every lane is served by the nous inference API
        # (examples/artifacts/tier-a-models.json base_url).
        "provider": lane.get("provider") or "nous",
        "provenance": lane.get("provenance", ""),
        "table": _table_line(),
        # The resolved library version, so an installer can assert what is
        # actually running from PATH (a verb probe cannot tell 0.7.0 from 0.7.1).
        "evalroute_version": _lib_version(),
        "card": card,
    }, ensure_ascii=False)


def _note_route(task: str, lane: dict[str, Any], method: str, conf: float,
                facets: list[str] | None = None, replace_route_id: str = "") -> str | None:
    """Best-effort logging; explicit replacements fail rather than disappearing."""
    try:
        from . import flywheel as fw
        return fw.note_route(task, lane, method, conf, facets=facets,
                             replace_route_id=replace_route_id)
    except Exception as exc:
        if replace_route_id:
            raise
        logger.warning("evalroute label append failed: %s", exc)
        return None


def evalroute_route(args: dict[str, Any], **_) -> str:
    """Handler for the evalroute_route tool. Never raises; errors are JSON."""
    try:
        task = (args.get("task") or "").strip()
        if not task:
            return json.dumps({"error": "task is required: a short description of the work"})
        lane_id = (args.get("lane") or "").strip()
        replace_id = (args.get("replace_route_id") or "").strip()
        if replace_id and not lane_id:
            return json.dumps({"error": "replace_route_id requires a pinned lane"})
        if lane_id:
            lane = _lane_by_id(lane_id)
            if lane is None:
                known = ", ".join(l["id"] for l in _load_routes())
                return json.dumps({"error": f"unknown lane {lane_id!r}; known lanes: {known}"})
            route_id = _note_route(task, lane, "pinned", 1.0, replace_route_id=replace_id)
            card = route_card(lane, 1.0, [], pinned=True, route_id=route_id)
            return _tool_result(card, lane, 1.0, pinned=True, method="pinned", route_id=route_id)
        lane, conf, hits, method, facets = route_full(task)
        route_id = _note_route(task, lane, method, conf, facets=facets)
        return _tool_result(route_card(lane, conf, hits, method=method, facets=facets, route_id=route_id),
                            lane, conf, pinned=False, method=method, route_id=route_id)
    except Exception as exc:  # route table broken -> actionable error, not a crash
        return json.dumps({"error": f"evalroute: {exc}"})


# ---------------------------------------------------------------- slash + CLI

def _route_for_args(raw_args: str, wide: bool = False) -> tuple[str, dict[str, Any], float, bool, str, str | None]:
    """Shared body for /route and `hermes evalroute route`.

    Returns (card, lane, confidence, pinned, method, route_id) so the CLI
    can also emit the _tool_result JSON envelope.
    """
    parts = (raw_args or "").split()
    lane_id = replace_id = ""
    task = (raw_args or "").strip()
    if parts and parts[0] == "--lane":
        if len(parts) < 3:
            raise ValueError("usage: route [--lane <lane-id> [--replace-route-id <id>]] <task description>")
        lane_id = parts[1]
        rest = parts[2:]
        if rest and rest[0] == "--replace-route-id":
            if len(rest) < 3:
                raise ValueError("--replace-route-id needs the prior route ID and task")
            replace_id, rest = rest[1], rest[2:]
        task = " ".join(rest)
    lane = _lane_by_id(lane_id) if lane_id else None
    if lane is not None:
        route_id = _note_route(task, lane, "pinned", 1.0, replace_route_id=replace_id)
        return route_card(lane, 1.0, [], pinned=True, route_id=route_id, wide=wide), \
            lane, 1.0, True, "pinned", route_id
    if lane_id:
        known = ", ".join(l["id"] for l in _load_routes())
        raise ValueError(f"unknown lane {lane_id!r}; known lanes: {known}")
    lane_obj, conf, hits, method, facets = route_full(task)
    route_id = _note_route(task, lane_obj, method, conf, facets=facets)
    return route_card(lane_obj, conf, hits, method=method, facets=facets,
                      route_id=route_id, wide=wide), \
        lane_obj, conf, False, method, route_id


def _card_for_args(raw_args: str) -> str:
    return _route_for_args(raw_args)[0]


def handle_route_command(raw_args: str) -> str:
    """Handler for /route (ctx.register_command)."""
    try:
        if not (raw_args or "").strip():
            lanes = _load_routes()
            return ("usage: /route [--lane <lane-id> [--replace-route-id <id>]] <task>\n"
                    "lanes: " + ", ".join(l["id"] for l in lanes))
        return _card_for_args(raw_args)
    except Exception as exc:
        return f"evalroute: {exc}"


# ------------------------------------------------------- install-routes (CLI)

_VALID_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
_EFFORT_ORDER = ["none", "minimal", "low", "medium", "high", "xhigh", "max"]


def _effort_for_override(lane: dict[str, Any]) -> str:
    """One effort value per model for agent.reasoning_overrides.

    A model serving two lanes keeps the HIGHER effort: reasoning_overrides has
    one slot per model id, and under-routing effort is the costlier miss
    (coverage, not verbosity). So take the highest effort mentioned in the
    lane's effort string, not the last one ("Opus high, subagents low-medium"
    -> high).
    """
    found = _EFFORT_RE.findall(str(lane.get("effort", "medium")))
    ranked = [e for e in _EFFORT_ORDER if e in found]
    return ranked[-1] if ranked else "medium"


def _config_module():
    """Import hermes_cli.config lazily (registration must not need it)."""
    from hermes_cli.config import load_config, set_config_value
    return load_config, set_config_value


def install_routes(dry_run: bool = False) -> int:
    """Write the route table's effort column into agent.reasoning_overrides.

    Runs inside the `hermes` process (plugin CLI command), so it calls
    set_config_value in-process rather than shelling out. Merges with existing
    overrides: keys the route table names win; keys it doesn't name are kept.
    """
    load_config, set_config_value = _config_module()

    lanes = _load_routes()
    desired: dict[str, str] = {}
    for lane in lanes:
        model, effort = lane["model"], _effort_for_override(lane)
        if effort not in _VALID_EFFORTS:
            print(f"  skipping {model}: bad effort {effort!r} in routes.yaml")
            continue
        # Keep the higher effort when a model serves two lanes.
        cur = desired.get(model)
        if cur is not None and _EFFORT_ORDER.index(effort) < _EFFORT_ORDER.index(cur):
            continue
        desired[model] = effort

    cfg = load_config() or {}
    agent_cfg = cfg.get("agent") if isinstance(cfg.get("agent"), dict) else {}
    existing = agent_cfg.get("reasoning_overrides") or {}
    if not isinstance(existing, dict):
        print("  agent.reasoning_overrides is not a mapping; refusing to touch it")
        return 1

    merged = {**existing, **desired}
    if merged == existing:
        print("agent.reasoning_overrides already matches the route table; nothing to do.")
        print(f"  current: {json.dumps(existing, ensure_ascii=False)}")
        return 0

    diff_added = {k: v for k, v in desired.items() if existing.get(k) != v}
    diff_kept = {k: v for k, v in existing.items() if k not in desired}
    print(f"route table -> agent.reasoning_overrides ({len(diff_added)} set, "
          f"{len(diff_kept)} pre-existing kept)")
    for model, effort in sorted(diff_added.items()):
        mark = "+" if model not in existing else "~"
        print(f"  {mark} {model}: {existing.get(model, '-')} -> {effort}")
    for model, effort in sorted(diff_kept.items()):
        print(f"  = {model}: {effort} (kept; not in route table)")

    if dry_run:
        print("dry run: nothing written.")
        return 0

    set_config_value("agent.reasoning_overrides", json.dumps(merged))
    print("written. /model now carries each model's lane effort; "
          "/reasoning still overrides per session.")
    return 0


