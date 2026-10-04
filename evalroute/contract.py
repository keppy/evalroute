"""The plugin-facing contract of the evalroute library.

Everything the Hermes plugin (keppy/hermes-plugin-evalroute) relies on is
listed here. Those nine names are the public API; everything else in the
package is private to the library and may change without notice.
"""

from __future__ import annotations

CONTRACT_VERSION = 1

#: The nine contract names, by module:
#:   evalroute.routing.set_llm_facade        (called at plugin register time)
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
    "routing.evalroute_route",
    "routing.handle_route_command",
    "cli.setup_cli",
    "cli.evalroute_cli",
    "flywheel.handle_rate",
    "flywheel.on_pre_command",
    "flywheel.on_post_llm_call",
    "schemas.EVALROUTE_ROUTE",
)
