"""pgrad8: the strict check of every door (tangent, adjoint-test, sensitivity, param-grad) takes --seed N, the
runner's test direction."""
from __future__ import annotations

import argparse

import pytest

from woof.hex import ad_doors as d

SELF = ["--seeds", "1", "--direction", "field", "--stable-dir", "1", "--x-scale", "1e-12"]


def _parser():
    parser = argparse.ArgumentParser()
    d.add_ad_parsers(parser.add_subparsers(dest="cmd"))
    return parser


@pytest.mark.parametrize("door", ["param-grad", "tangent", "sensitivity", "adjoint-test"])
def test_strict_doors_take_seed(door):
    p = _parser()
    assert p.parse_args([door, "--out", "o"]).seed is None
    assert p.parse_args([door, "--out", "o", "--seed", "2"]).seed == 2


def test_seed_sets_the_runner_seeds_word_only():
    assert d.self_dot_args_seeded(SELF, None) == SELF
    out = d.self_dot_args_seeded(SELF, 3)
    assert out[:2] == ["--seeds", "3"] and out[2:] == SELF[2:]
    assert SELF[1] == "1"   # the table's list is not changed


def test_seed_refuses_zero_and_a_list_without_seeds():
    with pytest.raises(d.AdDoorError):
        d.self_dot_args_seeded(SELF, 0)
    with pytest.raises(d.AdDoorError):
        d.self_dot_args_seeded(["--direction", "field"], 2)


def test_table_self_dot_args_carry_seeds_word():
    args = d.load_table()["doors"]["adjoint-test"]["self_dot_args"]
    assert args[args.index("--seeds") + 1] == "1"
    assert d.self_dot_args_seeded(args, 2)[args.index("--seeds") + 1] == "2"
