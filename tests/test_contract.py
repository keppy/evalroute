"""The ten contract names the Hermes plugin relies on must all resolve."""

from __future__ import annotations

import evalroute.contract as contract


def test_contract_version_is_one():
    assert contract.CONTRACT_VERSION == 1


def test_all_ten_contract_names_resolve():
    from evalroute import cli, flywheel, routing, schemas  # noqa: F401

    resolved = {
        "routing.set_llm_facade": routing.set_llm_facade,
        "routing.set_surface": routing.set_surface,
        "routing.evalroute_route": routing.evalroute_route,
        "routing.handle_route_command": routing.handle_route_command,
        "cli.setup_cli": cli.setup_cli,
        "cli.evalroute_cli": cli.evalroute_cli,
        "flywheel.handle_rate": flywheel.handle_rate,
        "flywheel.on_pre_command": flywheel.on_pre_command,
        "flywheel.on_post_llm_call": flywheel.on_post_llm_call,
        "schemas.EVALROUTE_ROUTE": schemas.EVALROUTE_ROUTE,
    }
    assert set(resolved) == set(contract.CONTRACT_NAMES)
    for name, obj in resolved.items():
        assert obj is not None, name
