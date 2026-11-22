# SPDX-License-Identifier: Apache-2.0
"""What the gradient commands print and write, as a new user reads it (a review of 0.6.0):
the FAIL line names the check that failed, the direction line follows --seed, sensitivity.nc places its cells, the
map of a non-theta measure draws (or names) the measure's own field, the 12 km row's level count, the shipped Python
example's refusal. No card: the table, the printing functions and the NetCDF writer."""
from __future__ import annotations

import re
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

from woof.hex import ad_doors as d

ROW12 = "hrrrv5-q12.96.2890.n40.60w91.30"
TREE = Path(__file__).resolve().parents[1]


def _checks(**over):
    ok = {"dot": {"value": 1.1e-08, "bar": 2e-05, "pass": True}, "fd": {"value": 9.9e-04, "bar": 1e-02, "pass": True},
          "taylor": {"value": 9.9e-04, "bar": 1e-02, "pass": True}, "zero": {"value": 0, "bar": 0, "pass": True}}
    ok.update(over)
    return ok


def test_a_fail_on_the_whole_state_dot_alone_names_that_check_with_its_reading():
    # the checker's seed-3 run on batch9: every first-window check passed, the whole-state dot read over its bar
    c = _checks(self_dot={"value": 1.034e-07, "bar": 4.77e-07, "bar_roundings": 4, "roundings": 4.6, "pass": False,
                          "seed": 3})
    assert d.failed_checks(c) == ["the whole-state dot (1.034e-07, +4.60 roundings, against the bar of 4 roundings, "
                                  "4.77e-07; test direction 3)"]


def test_a_failed_first_window_check_is_named_and_is_not_the_window_limit():
    c = _checks(fd={"value": 1.098e-02, "bar": 1e-02, "pass": False},
                self_dot={"value": 2.4e-08, "bar": 4.77e-07, "pass": True, "seed": 1})
    assert d.failed_checks(c) == ["the first window's finite-difference check (1.098e-02 against 1e-02)"]
    limits = {"window": {"text": "w"}}
    over = {"verdict": "fail", "outside": [], "unresolved": [], "resolved": ["bump", "descent"]}
    assert d.limits_word(over, limits, False, False, dot_failed=False) == ""
    assert d.limits_word(over, limits, False, False, dot_failed=True) == \
        "; window: a known limit of this linear model, named above"


def test_the_verdict_line_no_longer_blames_valid_windows():
    # the windows were valid; the receipt's valid was false only because the whole-state dot failed
    src = (TREE / ".." / ".." / "woof" / "hex" / "ad_doors.py").read_text(encoding="utf-8")
    assert 'any(w.get("valid") is False for w in ctx.windows)' in src
    assert '[] if rec.get("valid") is not False else ["a window was not a valid derivative"]' not in src


def test_the_direction_line_names_a_range_of_seeds(capsys):
    check = {"value": 1e-08, "bar": 4.77e-07, "pass": True, "seed": "3..6", "how": "own",
             "direction": "the runner's own seeded whole-state direction (--direction field, test direction 1): x"}
    d.print_strict(check)
    out = capsys.readouterr().out
    assert "(--direction field, test directions 3..6)" in out and "test direction 1)" not in out


def test_sensitivity_nc_places_every_cell_and_edge_in_degrees(tmp_path):
    netCDF4 = pytest.importorskip("netCDF4")
    geo = {"latCell": np.array([40.0, 41.0, 42.0]), "lonCell": np.array([-91.0, -92.0, -93.5]),
           "latEdge": np.array([40.5, 41.5, 41.0, 40.2]), "lonEdge": np.array([-91.5, -92.5, -93.0, -90.9])}
    path = tmp_path / "sensitivity.nc"
    d.write_mesh_fields(path, {"dJ_dtheta": (np.ones((2, 3)), "K K-1", "t", "cell"),
                               "dJ_du_stored": (np.ones((2, 4)), "K per m s-1", "u", "edge"),
                               "dJ_dw_stored": (np.ones((3, 3)), "K per m s-1", "w", "interface")},
                        {"title": "x"}, coords=d.mesh_coordinates(geo))
    with netCDF4.Dataset(path) as ds:
        assert np.allclose(ds["latCell"][:], geo["latCell"]) and np.allclose(ds["lonCell"][:], geo["lonCell"])
        assert ds["latCell"].units == "degrees_north" and ds["lonCell"].standard_name == "longitude"
        assert ds["latCell"].dimensions == ("nCells",) and ds["latEdge"].dimensions == ("nEdges",)
        assert ds["dJ_dtheta"].coordinates == "lonCell latCell"
        assert ds["dJ_dw_stored"].coordinates == "lonCell latCell"
        assert ds["dJ_du_stored"].coordinates == "lonEdge latEdge"


def test_every_gradient_file_writer_passes_the_mesh_coordinates():
    src = (TREE / ".." / ".." / "woof" / "hex" / "ad_doors.py").read_text(encoding="utf-8")
    calls = re.findall(r"write_mesh_fields\(nc, .*?\n(?:.*\n)*?.*?\)\n", src)
    assert len(calls) == 3   # tangent.nc, sensitivity.nc, the carried sensitivity.nc
    assert all("coords=mesh_coordinates(" in c for c in calls)


def _grads():
    g_theta = np.zeros((10, 4))
    g_theta[8] = 5.0        # dJ/dtheta carries most at level 9
    g_qv = np.zeros((10, 4))
    g_qv[2] = 7.0           # dJ/dqv at level 3
    return g_theta, g_qv


def test_the_map_of_a_qv_measure_draws_dj_dqv_and_says_its_level_and_the_measures_levels():
    g_theta, g_qv = _grads()
    m = d.map_choice({"metric": "qv", "field": "qv", "levels_1based": [1, 2, 3, 4, 5]}, g_theta, g_qv, "kg kg-1", None)
    assert m["variable"] == "dJ_dqv" and m["level"] == 3 and np.array_equal(m["values"], g_qv)
    assert m["units"] == "kg kg-1/(kg/kg)" and m["start"] == "initial water vapour (qv)"
    assert m["said"] == ("map: dJ/dqv at model level 3 (the level that carries most of dJ/dqv); the measure is of qv "
                         "at model levels 1-5; --map-level draws another")


def test_the_map_of_a_theta_measure_is_drawn_and_said_as_before():
    g_theta, g_qv = _grads()
    m = d.map_choice({"metric": "theta_m", "field": "theta_m", "levels_1based": list(range(1, 11))}, g_theta, g_qv,
                     "K", None)
    assert m["variable"] == "dJ_dtheta" and m["level"] == 9 and m["units"] == "K/K" and m["start"] == "initial theta"
    assert m["said"] == "map: level 9, the level that carries most of the field; --map-level draws another"


def test_the_map_of_another_measure_says_plainly_which_field_it_draws():
    g_theta, g_qv = _grads()
    m = d.map_choice({"metric": "w", "field": "w", "levels_1based": [4, 6]}, g_theta, g_qv, "m s-1", 2)
    assert m["variable"] == "dJ_dtheta" and m["level"] == 2
    assert m["said"] == ("map: dJ/dtheta at model level 2 (--map-level); the measure is of w at model levels 4,6, "
                         "whose own start field's gradient is in sensitivity.nc, not drawn")


def test_the_12km_row_has_the_59_levels_its_plan_has_and_the_pages_say_so():
    table = d.load_table()
    for name, row in table["rows"].items():   # a row's level word and its own description agree
        said = re.findall(r"(\d+) levels\)", row.get("what", ""))
        if said and row.get("levels"):
            assert int(said[0]) == int(row["levels"]), name
    assert table["rows"][ROW12]["levels"] == 59   # sensitivity.nc of the shipped 12 km example: nVertLevels 59
    guide = (TREE / "docs" / "differentiable" / "GUIDE.md").read_text(encoding="utf-8")
    readme = (TREE / "README.md").read_text(encoding="utf-8")
    assert "12 km cut of the storm box, 2,890 cells, 59 levels" in guide
    assert "3 km, 38,278 cells, 59 levels" in guide
    assert "(2,890 cells of 12 km around 40.6N 91.3W, 59 levels)" in readme


def test_the_shipped_python_example_refuses_in_one_sentence_with_exit_status_2(monkeypatch, capsys, tmp_path):
    import woof.hex.ad_session as session

    class Refused:
        def __init__(self, *a, **k):
            raise d.AdDoorError("card 0 is an Ada card; this row's derivative images are built for compute "
                                "capability 12.0 only: run it on such a card")

    monkeypatch.setattr(session, "SensitivitySession", Refused)
    monkeypatch.setattr(sys, "argv", ["own_measure.py", str(tmp_path / "own")])
    with pytest.raises(SystemExit) as end:
        runpy.run_path(str(TREE / "examples" / "differentiable" / "own_measure.py"), run_name="__main__")
    assert end.value.code == 2
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and err.startswith("own_measure: card 0 is an Ada card") and "Traceback" not in err

