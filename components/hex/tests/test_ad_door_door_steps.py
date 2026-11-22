# SPDX-License-Identifier: Apache-2.0
"""A door may run its own window on a row (rows.<row>.door_steps.<door>), so the 120 km example's adjoint-test runs
the window its directions resolve in while tangent, sensitivity and param-grad keep the row's (no card)."""
from __future__ import annotations

import argparse
import re

from woof.hex import ad_doors as d

ROWS120 = ("hrrrv5-x1.40962", "hrrrv5-full-x1.40962")


def _args(steps=None):
    return argparse.Namespace(steps=steps)


def test_a_given_steps_always_wins():
    row = {"default_steps": 3, "door_steps": {"adjoint-test": 1}}
    assert d.door_steps(row, "adjoint-test", _args(3)) == 3
    assert d.door_steps(row, "tangent", _args(2)) == 2


def test_a_door_without_its_own_window_runs_the_rows():
    row = {"default_steps": 3, "door_steps": {"adjoint-test": 1}}
    assert d.door_steps(row, "adjoint-test", _args()) == 1
    for door in ("tangent", "sensitivity", "param-grad", None):
        assert d.door_steps(row, door, _args()) == 3
    assert d.door_steps({"default_steps": 15}, "adjoint-test", _args()) == 15


def test_the_120km_rows_test_on_their_own_window_inside_the_recorded_plan():
    table = d.load_table()
    for name in ROWS120:
        row = table["rows"][name]
        own = d.door_steps(row, "adjoint-test", _args())
        assert own == int(row["door_steps"]["adjoint-test"]) and 1 <= own < int(row["default_steps"]), name
        assert own <= int(row["example_plan_steps"]), name
        assert row["door_steps_what"], name
        for door in ("tangent", "sensitivity", "param-grad"):
            assert d.door_steps(row, door, _args()) == int(row["default_steps"]), (name, door)


def test_the_120km_rows_rerun_the_real_model_on_their_own_ladder_and_others_on_the_tables():
    table = d.load_table()
    spec = table["doors"]["adjoint-test"]
    for name in ROWS120:
        hs = d.adjoint_test_taylor_h(spec, table["rows"][name])
        assert len(hs) >= 5 and hs == sorted(hs, reverse=True) and hs[0] >= 1.0, name
        assert all(0.4 <= b / a <= 0.6 for a, b in zip(hs, hs[1:])), name   # neighbours a factor of two apart
    assert d.adjoint_test_taylor_h(spec, {}) == [float(h) for h in spec["taylor_h"]]


ROW3 = "hrrrv5-q3.96.38278.n40.60w91.30"


def test_the_3km_row_reruns_the_real_model_above_the_rounding_floor_of_its_measure():
    table = d.load_table()
    spec = table["doors"]["adjoint-test"]
    row = table["rows"][ROW3]
    hs = d.adjoint_test_taylor_h(spec, row)
    assert hs == [0.3, 0.2, 0.1, 0.06, 0.03]
    assert min(hs) >= 0.03 and hs == sorted(hs, reverse=True)
    assert all(b / a >= 0.5 for a, b in zip(hs, hs[1:]))   # neighbours at most a factor of two apart
    assert "5.4 spacings" in row["adjoint_test_taylor_h_what"] and "RTX PRO 6000 Blackwell, 16 Nov 2026" in row["adjoint_test_taylor_h_what"]
    # no internal run name (a lettered machine, a numbered run leg) in a printed source
    assert not re.search(r"\bbox [A-Z]\d*\b|\b[a-z]\d{2}-[a-z0-9]", row["adjoint_test_taylor_h_what"])


def test_every_door_window_in_the_table_names_a_door_that_exists():
    table = d.load_table()
    for name, row in table["rows"].items():
        for door, steps in (row.get("door_steps") or {}).items():
            assert door in table["doors"] and int(steps) >= 1, (name, door)
