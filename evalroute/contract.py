"""The plugin-facing contract of the evalroute library.

Everything the Hermes plugin (keppy/hermes-plugin-evalroute) relies on is
listed here. Those ten names are the public API; everything else in the
package is private to the library and may change without notice.
"""

from __future__ import annotations

CONTRACT_VERSION = 1

#: The ten contract names, by module:
#:   evalroute.routing.set_llm_facade        (called at plugin register time)
#:   evalroute.routing.set_surface           (plugin sets "hermes-cli" /
#:                                            "hermes-chat" when it invokes
#:                                            the handlers; contract name #10,
#:                                            added without a version bump —
#:                                            the plugin guard compares
#:                                            CONTRACT_VERSION, not the list)
#:   evalroute.routing.evalroute_route       (tool handler for evalroute_route)
#:   evalroute.routing.handle_route_command  (/route slash command)
#:   evalroute.cli.setup_cli                 (register_cli_command setup_fn)
#:   evalroute.cli.evalroute_cli             (register_cli_command handler_fn)
#:   evalroute.flywheel.handle_rate          (/rate slash command + CLI rate)
#:   evalroute.flywheel.on_pre_command       (pre_command hook: /model /reasoning)
#:   evalroute.flywheel.on_post_llm_call     (post_llm_call hook)
#:   evalroute.schemas.EVALROUTE_ROUTE       (tool schema)
CONTRACT_NAMES = (
    "routing.set_llm_facade",
    "routing.set_surface",
    "routing.evalroute_route",
    "routing.handle_route_command",
    "cli.setup_cli",
    "cli.evalroute_cli",
    "flywheel.handle_rate",
    "flywheel.on_pre_command",
    "flywheel.on_post_llm_call",
    "schemas.EVALROUTE_ROUTE",
)
