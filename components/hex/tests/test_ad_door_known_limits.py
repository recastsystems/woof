# SPDX-License-Identifier: Apache-2.0
"""adjoint-test names the measured limits of the linear model it tests beside its directions, and a known limit never
changes the verdict (no card: the table and the printing functions only)."""
from __future__ import annotations

from woof.hex import ad_doors as d

ROW12 = "hrrrv5-q12.96.2890.n40.60w91.30"
ROW3 = "hrrrv5-q3.96.38278.n40.60w91.30"


def _dirs(**status):
    return [{"name": n, "status": s, "seed": None, "rows": [], "value": None, "why": "w", "bar": 0.01}
            for n, s in status.items()]


def test_each_route_of_the_two_storm_rows_names_its_limits_with_a_source():
    table = d.load_table()
    for row, route in ((ROW12, None), (ROW12, "taped"), (ROW3, None), (ROW3, "taped")):
        limits = d.known_limits(table["rows"][row], route)
        assert limits, (row, route)
        for key, item in limits.items():
            assert key == "window" or key in table["doors"]["adjoint-test"]["directions"]["rows"], key
            assert item["text"] and item.get("source"), (row, route, key)


def test_the_default_12km_route_names_near_ground_moisture_and_its_cause():
    limits = d.known_limits(d.load_table()["rows"][ROW12], None)
    text = limits["near-ground-moisture"]["text"]
    assert "2 to 4 per cent short" in text and "speed and condensation" in text


def test_the_120km_example_names_its_two_unresolved_directions_and_why():
    table = d.load_table()
    limits = d.known_limits(table["rows"]["hrrrv5-x1.40962"], None)
    # on its own 1-step window and factor-two ladder near-ground moisture resolves inside the bar (one 96 GB card, 15 Nov 2026),
    # so only descent and the smooth draw are named
    assert set(limits) == {"descent", "smooth-draw"}
    assert set(limits) <= set(table["doors"]["adjoint-test"]["directions"]["rows"])
    assert "1-step window" in limits["descent"]["text"] and "0.01 bar" in limits["descent"]["text"]
    assert "nothing to test" in limits["smooth-draw"]["text"]
    assert all(item["source"] for item in limits.values())
    # the bump and near-ground moisture resolve there, so the example passes; a limit never changes the verdict
    dirs = _dirs(bump="inside", descent="unresolved", **{"near-ground-moisture": "inside",
                                                          "smooth-draw": "unresolved"})
    assert d.directions_verdict(dirs, True, {"min_resolved": 2})["verdict"] == "pass"
    dirs = _dirs(bump="inside", descent="unresolved", **{"near-ground-moisture": "unresolved",
                                                          "smooth-draw": "unresolved"})
    over = d.directions_verdict(dirs, True, {"min_resolved": 2})
    assert over["verdict"] == "unresolved"
    assert "smooth-draw" in d.limits_word(over, limits, False)


def test_the_120km_full_physics_row_names_its_descent_gap():
    row = d.load_table()["rows"]["hrrrv5-full-x1.40962"]
    assert d.known_limits(row, None) == {}   # the default (floored) boundary layer names none (F10-full)
    limits = d.known_limits(row, "taped")    # the row's own images keep the measured descent gap
    assert set(limits) == {"descent"} and "OUTSIDE" in limits["descent"]["text"] and limits["descent"]["source"]
    # a known limit never changes the verdict: the row's adjoint-test still FAILS along descent
    dirs = _dirs(bump="inside", descent="outside", **{"near-ground-moisture": "inside"})
    assert d.directions_verdict(dirs, True, {"min_resolved": 2})["verdict"] == "fail"


def test_a_row_without_any_limits_names_none():
    assert d.known_limits({"mesh": "x1.40962", "suite": "hrrrv5"}, None) == {}


def test_a_known_limit_never_turns_a_fail_into_a_pass():
    limits = {"near-ground-moisture": {"text": "t"}}
    dirs = _dirs(bump="inside", descent="inside", **{"near-ground-moisture": "outside"})
    over = d.directions_verdict(dirs, True, {"min_resolved": 2})
    assert over["verdict"] == "fail"
    assert d.limits_word(over, limits, False) == \
        "; near-ground-moisture: a known limit of this linear model, named above"


def test_a_pass_says_which_known_limits_it_does_not_cover(capsys):
    limits = {"near-ground-moisture": {"text": "about 2 to 4 per cent short"}, "window": {"text": "seeds 3, 4"}}
    dirs = _dirs(bump="inside", descent="inside", **{"near-ground-moisture": "unresolved", "smooth-draw": "unresolved"})
    over = d.directions_verdict(dirs, True, {"min_resolved": 2})
    assert over["verdict"] == "pass"
    assert d.limits_word(over, limits, True) == \
        "; not covered by this PASS: near-ground-moisture (a known limit, named above)"
    d.print_known_limits(dirs, limits)
    out = capsys.readouterr().out
    assert "known limits of this linear model" in out
    assert "[this run: UNRESOLVED] about 2 to 4 per cent short" in out
    assert "window" in out and "seeds 3, 4" in out


def test_no_limits_prints_nothing(capsys):
    d.print_known_limits(_dirs(bump="inside"), {})
    assert capsys.readouterr().out == ""
    assert d.limits_word({"unresolved": ["x"], "outside": []}, {}, True) == ""


def test_no_route_says_not_measured_where_a_run_exists():
    """THE BREAKAGE (the 0.6.0 candidate's checker, 2026-11-15): the 3 km default route had read FAIL on a 96 GB card
    (bump 1.10e-02 from 1) and the command's known limits and the README still called it not measured."""
    table = d.load_table()
    limits = d.known_limits(table["rows"][ROW3], None)
    assert set(limits) == {"bump", "descent", "window"}
    assert "INSIDE, 3.65e-04 from 1" in limits["bump"]["text"] and "3.653e-04" in limits["bump"]["text"]
    assert "INSIDE, 1.92e-03 from 1" in limits["descent"]["text"] and "0.3, 0.2, 0.1, 0.06 and 0.03" in limits["descent"]["text"]
    for row, route in ((ROW12, None), (ROW12, "taped"), (ROW3, None), (ROW3, "taped")):
        for key, item in d.known_limits(table["rows"][row], route).items():
            assert "not yet measured" not in item["text"] and "not measured" not in item["text"], (row, route, key)


def test_the_texts_state_the_3km_verdict_and_what_a_pass_means():
    from pathlib import Path
    root = (Path(d.__file__).resolve().parents[2] / "components" / "hex")
    readme = (root / "README.md").read_text(encoding="utf-8")
    limits = (root / "docs/differentiable/LIMITS.md").read_text(encoding="utf-8")
    guide = (root / "docs/differentiable/GUIDE.md").read_text(encoding="utf-8")
    section = readme[readme.index("## The differentiable engine"):]
    assert "the new default route has not been measured" not in section
    assert "ended **PASS** on the default route" in section and "INSIDE (3.65e-04 from 1)" in section
    assert "INSIDE (1.92e-03 from 1)" in section   # the descent at the row's own step sizes
    assert "**3 km: `adjoint-test` ends PASS on the default route and FAIL on the exact route.**" in limits
    for text in (section, guide):   # the verdict rule and its exit status, in each
        assert "PASS" in text and "NOT PASSED (UNRESOLVED)" in text and "exit status 0" in text and "exit status 1" in text


def test_a_fail_on_the_whole_state_dot_names_the_window_limit_not_an_unresolved_direction():
    # the checker's seed-3 run: bump and descent inside, near-ground moisture unresolved, the whole-state dot failed
    limits = {"near-ground-moisture": {"text": "t"}, "window": {"text": "seed 3"}}
    dirs = _dirs(bump="inside", descent="inside", **{"near-ground-moisture": "unresolved"})
    over = d.directions_verdict(dirs, True, {"min_resolved": 2})
    assert d.limits_word(over, limits, False, checks_passed=False) == \
        "; window: a known limit of this linear model, named above"
    assert d.limits_word(over, {"near-ground-moisture": {"text": "t"}}, False, checks_passed=False) == ""


def test_the_direction_line_names_the_seed_it_ran(capsys):
    check = {"value": 1e-07, "bar": 4.77e-07, "pass": True, "seed": 3, "how": "own",
             "direction": "the runner's own seeded whole-state direction (--direction field, test direction 1): x"}
    d.print_strict(check)
    out = capsys.readouterr().out
    assert "direction: the runner" in out and "test direction 3)" in out and "test direction 1)" not in out
