"""``woof ensemble --sppt/--skeb``: the switches reach the ensemble request.

A switch the request never carries would be a member forecast that silently
runs without the scheme the user asked for, so the door must merge them into
``[ensemble.stochastic]`` and nothing else.
"""
import argparse

import pytest

from woof.ensemble import door


def _parse(*argv):
    parser = argparse.ArgumentParser()
    door.add_arguments(parser)
    door.add_stochastic_arguments(parser)
    return parser.parse_args(list(argv))


def test_switches_merge_into_the_request():
    args = _parse("--members", "4", "--sppt", "--skeb")
    request = door.request_for_payload(b"", members=4, stochastic=door.stochastic_switches(args))
    assert request.stochastic == {"sppt": True, "skebs": True}


def test_switches_keep_the_rest_of_a_configured_table():
    payload = b"[ensemble]\nmembers = 3\n[ensemble.stochastic]\nsppt = true\nskebs = true\n"
    args = _parse("--no-skeb")
    request = door.request_for_payload(payload, stochastic=door.stochastic_switches(args))
    assert request.stochastic["sppt"] is True and request.stochastic["skebs"] is False


def test_no_switch_leaves_the_request_unchanged():
    args = _parse("--members", "2")
    assert door.stochastic_switches(args) is None
    plain = door.request_for_payload(b"", members=2)
    assert door.request_for_payload(b"", members=2, stochastic=None) == plain


def test_a_switch_without_an_ensemble_is_refused():
    with pytest.raises(ValueError, match="need an ensemble"):
        door.request_for_payload(b"", stochastic={"sppt": True})


def test_the_go_parser_takes_the_switches():
    from woof.go_cli import _add_go_arguments
    parser = argparse.ArgumentParser()
    _add_go_arguments(parser)
    args = parser.parse_args(["x.toml", "--members", "2", "--sppt", "--no-skeb"])
    assert door.stochastic_switches(args) == {"sppt": True, "skebs": False}
