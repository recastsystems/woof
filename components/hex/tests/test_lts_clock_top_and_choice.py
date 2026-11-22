"""Full-step LTS at the forecast door: the class clock takes the plan's real top class, and the MPAS route runs
the fixed step where LTS is measured to lose (HEX-LAM-VS-ZOOM-2026-10-27 defects 1 and 2).

Defect 1: the door's clock took the config dataclass's ladder ceiling (16) while both executors plan at most 8, so a
bare LTS run on the 96 -> 3 km zoom globe stepped 12.5 s instead of 15 s (135.1 vs 105.9 s per forecast hour).
Defect 2: even at 15 s, active LTS on the MPAS route ran 24 % slower than the fixed step on that mesh (105.9 vs 85.7).
"""
from argparse import Namespace
import dataclasses

import numpy as np
import pytest

from woof.hex import forecast_door, mpas_defaults
from woof.hex.fullstep_lts_census import DOOR_DEFAULT_MAX_RATE, mesh_class_census, read_mesh_edges
from woof.hex.fullstep_lts_choice import FORCE_ENV, mpas_route_choice

# The 96 -> 3 km zoom globe of the measurement (149,322 cells), as mesh_class_census counts it at the default cap.
ZOOM_CENSUS = {1: 64832, 2: 955, 4: 7852, 8: 75683}
EVENTS_3H = dict(history_every_minutes=60, hours=3.0)


def _args(**values):
    base = dict(fullstep_local_timestep=True, fullstep_lts_mode="active", fullstep_lts_max_rate=None,
                fullstep_lts_buffer_rings=None, samehost_devices=None, local_timestep=False, grid=None, init=None,
                **EVENTS_3H)
    base.update(values)
    return Namespace(**base)


def _chain_mesh(path, spacings_m):
    """A closed 1-D chain of cells (cell i joins cell i+1) whose edge lengths are ``spacings_m``, as an MPAS file."""
    from netCDF4 import Dataset
    n_edges = len(spacings_m)
    with Dataset(path, "w") as dataset:
        dataset.createDimension("nCells", n_edges + 1)
        dataset.createDimension("nEdges", n_edges)
        dataset.createDimension("TWO", 2)
        dataset.createVariable("dcEdge", "f8", ("nEdges",))[:] = np.asarray(spacings_m, dtype=np.float64)
        coe = np.stack([np.arange(n_edges), np.arange(1, n_edges + 1)], axis=1) + 1   # MPAS files are 1-based
        dataset.createVariable("cellsOnEdge", "i4", ("nEdges", "TWO"))[:] = coe
    return path


def _graded(top_km):
    """3 km cells, then a ramp doubling to ``top_km``, then a long run of ``top_km`` cells."""
    ramp = [3000.0 * 2 ** k for k in range(int(np.log2(top_km / 3.0)) + 1)]
    return [3000.0] * 30 + [s for s in ramp for _ in range(6)] + [top_km * 1000.0] * 200


def test_the_door_default_top_rate_is_the_executors():
    """One default for both executors and the door's clock; the clock read the dataclass's 16 before the fix."""
    fields = {field.name: field.default for field in dataclasses.fields(forecast_door.ForecastRequest)}
    assert fields["fullstep_lts_max_rate"] == DOOR_DEFAULT_MAX_RATE == 8


def test_a_bare_lts_run_on_the_zoom_globe_steps_15_s_not_12_5_s():
    """The exact defect: the 18 s default with 1800 s radiation, hourly history, 3 h. 8 x 15 s = 120 s fits; the
    fix moves to 15 s. Before it, the door planned for 16 x the step and took 12.5 s (16 x 12.5 = 200 s)."""
    fixed = mpas_defaults.default_dt_seconds(3000.0)
    assert fixed["dt_seconds"] == 18.0
    moved = forecast_door.lts_step(_args(), fixed, 1800.0)
    assert moved["dt_seconds"] == 15.0 and moved["lts_max_rate"] == 8


def test_the_clock_takes_the_mesh_top_class(tmp_path):
    """A mesh that grades only to 4 keeps its 18 s default (4 x 18 = 72 s divides every event); one that grades
    to 8 or past the cap moves to 15 s; an explicit --fullstep-lts-max-rate 16 plans 16 where the mesh reaches it."""
    fixed = mpas_defaults.default_dt_seconds(3000.0)
    to12 = _chain_mesh(tmp_path / "to12.nc", _graded(12))
    to96 = _chain_mesh(tmp_path / "to96.nc", _graded(96))
    assert max(mesh_class_census(*read_mesh_edges(to12), max_rate=8)) == 4
    assert forecast_door.lts_step(_args(grid=to12), fixed, 1800.0)["dt_seconds"] == 18.0
    assert forecast_door.lts_step(_args(grid=to96), fixed, 1800.0)["dt_seconds"] == 15.0
    assert max(mesh_class_census(*read_mesh_edges(to96), max_rate=16)) == 16
    assert forecast_door.lts_step(_args(grid=to96, fullstep_lts_max_rate=16), fixed, 1800.0)["dt_seconds"] == 12.5


def test_the_census_holds_convective_cells_and_grades():
    """Cells finer than 8 km stay at rate 1 and no two neighbours differ by more than a factor 2."""
    spacing = np.asarray(_graded(96))
    n_cells = spacing.size + 1
    coe = np.stack([np.arange(spacing.size), np.arange(1, n_cells)], axis=1)
    census = mesh_class_census(spacing, coe, n_cells, max_rate=8)
    assert sorted(census) == [1, 2, 4, 8] and sum(census.values()) == n_cells
    assert census[1] >= 30 + 12   # the 3 and 6 km cells


def test_the_measured_zoom_globe_rows_and_picks():
    """The rows reproduce the fold tree's measured arms on the zoom globe (HEX-LTS-REGRESSION-2026-10-27, one RTX
    5090 per arm): MPAS fixed 76.2 vs measured 77.2/75.2, LTS top 8 at 15 s 75.2 vs 75.4, top 4 at 18 s 64.2 vs 64.0;
    the door keeps top class 4 at 18 s on both routes there."""
    from woof.hex.fullstep_lts_choice import plan_choice
    top4 = {1: 64832, 2: 955, 4: 83535}
    top2 = {1: 64832, 2: 84490}
    for route, fixed, lts8, lts4 in (("mpas-suite", 76.2, 75.2, 64.2), ("arwen", 80.5, 84.3, 71.4)):
        choice = plan_choice(149322, route=route, fixed_dt_seconds=18.0,
                             candidates=[(8, 15.0, ZOOM_CENSUS), (4, 18.0, top4), (2, 18.0, top2)])
        predicted = {c["top_rate"]: c["predicted_seconds_per_forecast_hour"] for c in choice["candidates"]}
        assert choice["predicted_seconds_per_forecast_hour"]["fixed"] == pytest.approx(fixed, abs=0.2)
        assert predicted[8] == pytest.approx(lts8, abs=0.3) and predicted[4] == pytest.approx(lts4, abs=0.3)
        assert choice["choice"] == "lts" and choice["lts_top_rate"] == 4 and choice["lts_dt_seconds"] == 18.0
    single = mpas_route_choice(ZOOM_CENSUS, fixed_dt_seconds=18.0, lts_dt_seconds=15.0)
    assert single["predicted_seconds_per_forecast_hour"]["lts"] == pytest.approx(75.2, abs=0.3)


def test_lts_wins_where_coarse_cells_dominate():
    """Not a blanket off switch: a census dominated by coarse cells predicts LTS cheaper."""
    choice = mpas_route_choice({1: 50_000, 8: 2_000_000}, fixed_dt_seconds=18.0, lts_dt_seconds=15.0)
    assert choice["choice"] == "lts"


def test_the_door_runs_the_plain_fixed_step_where_lts_loses(tmp_path):
    """LTS requested on a mesh where it loses: every LTS argument ends as a plain run's, so the run is that run."""
    grid = _chain_mesh(tmp_path / "to96.nc", _graded(96))
    args = _args(grid=grid)
    assert forecast_door.mpas_lts_choice(args, fixed_dt=18.0, lts_dt=15.0, environ={}) == "fixed"
    plain = _args(fullstep_local_timestep=False, fullstep_lts_mode=None)
    for name in ("fullstep_local_timestep", "fullstep_lts_mode", "fullstep_lts_max_rate", "fullstep_lts_buffer_rings"):
        assert getattr(args, name) == getattr(plain, name), name
    assert args.fullstep_lts_door_choice["door_choice"]["choice"] == "fixed"


def test_the_door_keeps_lts_when_forced_moving_or_on_several_cards(tmp_path):
    grid = _chain_mesh(tmp_path / "to96.nc", _graded(96))
    forced = _args(grid=grid)
    assert forecast_door.mpas_lts_choice(forced, fixed_dt=18.0, lts_dt=15.0, environ={FORCE_ENV: "1"}) == "lts"
    assert forced.fullstep_local_timestep and forced.fullstep_lts_mode == "active"
    moving = _args(grid=grid)
    assert forecast_door.mpas_lts_choice(moving, fixed_dt=18.0, lts_dt=15.0,
                                         environ={"WOOF_HEX_MOVING_CONTROL": "/x/control.json"}) is None
    assert moving.fullstep_local_timestep
    cards = _args(grid=grid, samehost_devices="0,1")
    assert forecast_door.mpas_lts_choice(cards, fixed_dt=18.0, lts_dt=15.0, environ={}) is None
    shadow = _args(grid=grid, fullstep_lts_mode=None)
    assert forecast_door.mpas_lts_choice(shadow, fixed_dt=18.0, lts_dt=15.0, environ={}) is None


def test_the_step_cost_table_has_a_row_per_route():
    """The choice is table work: one measured row per physics route, read from the shipped data file."""
    from woof.hex.fullstep_lts_choice import STEP_COST_TABLE, load_step_costs
    costs = load_step_costs()
    assert STEP_COST_TABLE.is_file() and {"mpas-suite", "arwen"} <= set(costs)
    for row in costs.values():
        assert row["seconds_per_step"] > 0 and row["seconds_per_cell_step"] > 0 and row["source"]


def test_the_arwen_route_prices_top_class_8_at_15_s_above_the_fixed_step():
    """On the zoom globe the WOOF-ArWen route's top class 8 at the clock's 15 s loses to the fixed step (84.3 vs 80.5,
    measured 83.0 vs 80.2): priced alone it picks fixed; a coarse-dominated census still picks LTS."""
    from woof.hex.fullstep_lts_choice import route_choice
    choice = route_choice(ZOOM_CENSUS, route="arwen", fixed_dt_seconds=18.0, lts_dt_seconds=15.0)
    predicted = choice["predicted_seconds_per_forecast_hour"]
    assert choice["choice"] == "fixed" and choice["route"] == "arwen"
    assert predicted["fixed"] == pytest.approx(80.5, abs=0.2) and predicted["lts"] == pytest.approx(84.3, abs=0.3)
    assert route_choice({1: 50_000, 8: 2_000_000}, route="arwen", fixed_dt_seconds=18.0,
                        lts_dt_seconds=15.0)["choice"] == "lts"


def test_a_route_without_a_measured_row_is_refused_by_name():
    from woof.hex.fullstep_lts_choice import route_choice
    with pytest.raises(ValueError, match="no measured step cost for the 'rrfs' route"):
        route_choice(ZOOM_CENSUS, route="rrfs", fixed_dt_seconds=18.0, lts_dt_seconds=15.0)


def test_the_arwen_door_runs_the_plain_fixed_step_where_lts_loses(tmp_path):
    """The WOOF-ArWen route's door hook: LTS requested on a mesh where it loses ends as the plain fixed run."""
    grid = _chain_mesh(tmp_path / "to96.nc", _graded(96))
    args = _args(grid=grid)
    assert forecast_door.lts_choice(args, route="arwen", fixed_dt=18.0, lts_dt=15.0, environ={}) == "fixed"
    assert args.fullstep_local_timestep is False and args.fullstep_lts_mode is None
    assert args.fullstep_lts_door_choice["route"] == "arwen"
    assert args.fullstep_lts_door_choice["door_choice"]["cost_model"]["seconds_per_step"] == pytest.approx(0.1063)


def test_a_lower_top_class_that_keeps_the_step_can_win():
    """The class clock alone keeps the top class and shrinks the step (8 x 15 s on the zoom globe); a top class of
    4 keeps 18 s (4 x 18 s = 72 s divides 1800 s).  plan_choice prices both against the fixed step and keeps the
    cheapest: with a per-step cost of 0.098 s and 1.95 us per cell-step (the zoom globe's WOOF-ArWen pair with the hoisted
    kernels, HEX-LTS-REGRESSION-2026-10-27) top class 4 at 18 s wins, top class 8 at 15 s does not."""
    from woof.hex.fullstep_lts_choice import plan_choice
    row = {"arwen": {"seconds_per_step": 0.0983, "seconds_per_cell_step": 1.9466e-6, "source": "test"}}
    top4 = {1: 64832, 2: 955, 4: 83535}
    choice = plan_choice(149322, route="arwen", fixed_dt_seconds=18.0,
                         candidates=[(8, 15.0, ZOOM_CENSUS), (4, 18.0, top4)], costs=row)
    predicted = {c["top_rate"]: c["predicted_seconds_per_forecast_hour"] for c in choice["candidates"]}
    assert predicted[8] == pytest.approx(80.1, abs=0.3) and choice["predicted_seconds_per_forecast_hour"]["fixed"] \
        == pytest.approx(77.8, abs=0.3)
    assert choice["choice"] == "lts" and choice["lts_top_rate"] == 4 and choice["lts_dt_seconds"] == 18.0
    assert predicted[4] < choice["predicted_seconds_per_forecast_hour"]["fixed"] < predicted[8]


def test_the_door_takes_the_cheaper_top_class_and_its_step(tmp_path, monkeypatch):
    """Through the door: the clock moves 18 s to 15 s for top class 8; where top class 4 at 18 s is predicted
    cheapest (a per-step part of 0.01 s against 1e-4 s per cell-step on this chain), the run plans at most 4 and
    steps 18 s (both executors read --fullstep-lts-max-rate)."""
    from woof.hex import fullstep_lts_choice
    grid = _chain_mesh(tmp_path / "to96.nc", _graded(96))
    monkeypatch.setattr(fullstep_lts_choice, "load_step_costs", lambda path=None: {
        "arwen": {"seconds_per_step": 0.01, "seconds_per_cell_step": 1.0e-4, "source": "test"}})
    args = _args(grid=grid)
    fixed = mpas_defaults.default_dt_seconds(3000.0)
    clocked = forecast_door.lts_step(args, fixed, 1800.0)
    assert clocked["dt_seconds"] == 15.0
    choice = forecast_door.lts_choice(args, route="arwen", fixed_dt=18.0, lts_dt=15.0, environ={})
    record = args.fullstep_lts_door_choice["door_choice"]
    assert {c["top_rate"] for c in record["candidates"]} == {8, 4, 2}
    assert choice == "lts" and record["lts_top_rate"] == 4 and args.fullstep_lts_max_rate == 4
    step = forecast_door.chosen_step(args, fixed, clocked, choice)
    assert step["dt_seconds"] == 18.0 and step["lts_max_rate"] == 4
    # Where the per-step part dominates, the plain fixed run is still the answer.
    monkeypatch.setattr(fullstep_lts_choice, "load_step_costs", lambda path=None: {
        "arwen": {"seconds_per_step": 0.05, "seconds_per_cell_step": 1.0e-4, "source": "test"}})
    args = _args(grid=grid)
    clocked = forecast_door.lts_step(args, fixed, 1800.0)
    choice = forecast_door.lts_choice(args, route="arwen", fixed_dt=18.0, lts_dt=15.0, environ={})
    assert choice == "fixed" and forecast_door.chosen_step(args, fixed, clocked, choice)["dt_seconds"] == 18.0
