"""The terrain clock read face by face: the default-off ``local_face`` candidate.

The shipped clock reads one triple per domain (steepest slope, highest
ground, strongest crest-band wind anywhere), which on NCAR's v4.4 CONUS
benchmarks put a Plains jet and a distant crest on a Sierra face and halved
both published steps.  ``terrain_clock = "local_face"``
(woof/terrain_clock_local.py) reads every face with the crest and wind in a
neighbourhood around it, a measured wind margin and the candidate's own
measured rows.  These tests pin: the neighbourhood measured on the map's own
ridges, the per-face slopes being the substep rule's, the candidate rows
and the cells they were not run at, the wind read per column from each
door's inputs, the margin and its cap at the domain-wide reading, the
derivation through the doors, a steep face under a local jet still capped,
and the NCAR benchmark readings reproduced from fixtures saved from the
WRF-input door itself.
"""
from __future__ import annotations

import json
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from woof import terrain_clock as tc
from woof import terrain_clock_local as tl
from woof.acoustic_adaptation import face_slopes, steepest_slope

from test_terrain_clock import (_cache_reader, _column_state,
                                _metem_inputs, _wizard_experiment)

DATA = Path(__file__).with_name("data") / "terrain_clock_local"


# ---------------------------------------------------------------------------
# What the candidate reads.
# ---------------------------------------------------------------------------


def test_it_is_the_default_from_2_8_8():
    """A bare configuration reads the local-face clock; "measured" and
    "pinned" stay selectable."""
    from woof.config import RunConfig, TERRAIN_CLOCK_MODES
    from woof.physics_source_defaults import PHYSICS_SELECTOR_VALUES

    assert RunConfig(nx=4, ny=4, nz=4, dx=3000.0, dy=3000.0, ztop=20000.0,
                     dt=15.0, run_seconds=60.0).terrain_clock == "local_face"
    assert _wizard_experiment([(60, 60)], (), 3000.0).domains[0] \
        .run.terrain_clock == "local_face"
    assert {"measured", "pinned"} <= set(TERRAIN_CLOCK_MODES)
    assert "local_face" in TERRAIN_CLOCK_MODES
    assert "local_face" in PHYSICS_SELECTOR_VALUES["terrain_clock"]


def test_the_selector_comment_names_it():
    from woof.physics_source_defaults import read_physics_selector_comment

    text = ('! gpuwm-physics-selectors-v1: {"terrain_clock": "local_face"}\n'
            "&time_control\n/\n")
    assert read_physics_selector_comment(text)["terrain_clock"] == \
        "local_face"


@pytest.mark.parametrize("dx, cells, metres", [
    # 2 km: the candidate's 4500 m crest, slope 0.05 ridge reaches 16 cells.
    (12000.0, 5, 60000.0), (2500.0, 13, 32000.0), (3000.0, 10, 30000.0),
    (2000.0, 16, 32000.0), (1000.0, 16, 16000.0), (500.0, 22, 11000.0)])
def test_the_neighbourhood_reaches_every_mapped_ridge_s_crest(dx, cells,
                                                              metres):
    """The window reaches, on every ridge the map reads at a spacing, from
    its steepest grid face to its crest; a spacing between two reads the
    longer of its neighbours'."""
    for arms in (("generated",), ("ncar",), ("generated", "ncar")):
        assert tl.neighbourhood_cells(dx, arms) == (cells, metres)


def test_the_reach_is_the_probe_ridge_s_own():
    """Measured on the probe's bell ridge itself (woof.core.terrain)."""
    import math

    from woof.core.terrain import bell_hill

    for dx, crest, slope in ((12000.0, 8850.0, 0.05), (3000.0, 4500.0, 0.4),
                             (1000.0, 1500.0, 0.6), (2000.0, 3000.0, 0.09)):
        a = (3.0 * math.sqrt(3.0) / 8.0) * crest / slope
        nx = max(96, math.ceil(10.0 * a / dx))
        nx += nx % 2
        h = bell_hill(SimpleNamespace(nx=nx, ny=1, dx=dx, hill_height=crest,
                                      hill_halfwidth=a))[0]
        face = int(np.argmax(np.abs(np.diff(h))))
        tops = np.flatnonzero(h >= h.max() - 1e-9)
        want = min(abs(p - c) for p in (face, face + 1) for c in tops)
        assert tl.ridge_reach_cells(dx, crest, slope) == want


def test_every_face_s_slope_is_the_substep_rule_s():
    rng = np.random.default_rng(11)
    h = np.cumsum(rng.normal(0.0, 60.0, (40, 50)), axis=1)
    msfu = 1.0 + 0.05 * rng.random((40, 51))
    msfv = 1.0 + 0.05 * rng.random((41, 50))
    sx, sy = face_slopes(h, 3000.0, 3000.0, msfu=msfu, msfv=msfv)
    reading = steepest_slope(h, 3000.0, 3000.0, msfu=msfu, msfv=msfv)
    assert max(sx.max(), sy.max()) == reading.slope
    axis, j, i = reading.face
    assert (sx[j, i - 1] if axis == "x" else sy[j - 1, i]) == reading.slope


def test_the_run_reads_the_arm_measured_at_its_own_dynamics():
    ncar = SimpleNamespace(epssm=0.1, smdiv=0.1, emdiv=0.01, damp_opt=3,
                           zdamp=5000.0, dampcoef=0.2, w_damping=1,
                           diff_6th_opt=0, diff_6th_factor=0.12,
                           diff_6th_slopeopt=0)
    generated = SimpleNamespace(**{**vars(ncar), "epssm": 0.5,
                                   "diff_6th_opt": 2, "diff_6th_slopeopt": 1})
    other = SimpleNamespace(**{**vars(ncar), "epssm": 0.3})
    assert tl.dynamics_arms(ncar) == ("ncar",)
    assert tl.dynamics_arms(generated) == ("generated",)
    # Between the measured dynamics: both arms, the least of them.
    assert tl.dynamics_arms(other) == ("generated", "ncar")
    # The filter's factor says nothing where the filter is off.
    assert tl.dynamics_arms(SimpleNamespace(**{**vars(ncar),
                                               "diff_6th_factor": 0.3})) \
        == ("ncar",)


def test_the_candidate_rows_merge_with_the_shipped_ones():
    """Each arm reads every shipped row, merged with its own three-hour row
    on the same ridge, and its rows on ridges the shipped map never ran."""
    shipped = tc.measured_map()
    document = tl.candidate_document()
    assert document["criterion"] == "blowup"
    for arm in ("generated", "ncar"):
        table = tl.candidate_map((arm,))
        assert table.winds == shipped.winds
        merged, added = tl._arm_rows(arm)
        assert len(table.rows) == len(shipped.rows) + len(added)
        mine = [row for row in document["rows"] if row["settings"] == arm]
        assert len([m for m in merged if m["arm"] == arm]) + len(added)             == len(mine)
        # A shipped row no candidate row re-ran keeps its fixed entries;
        # its adaptive entries are the 12-hour ones where the arm re-ran
        # that ridge on the adaptive clock, else the shipped ones.
        for k, (item, row) in enumerate(zip(merged, table.rows)):
            if item["arm"] is None:
                twelve = tl._adaptive_rows((arm,), shipped.rows[k],
                                           shipped.winds)
                want = (shipped.rows[k] if twelve is None else replace(
                    shipped.rows[k], adaptive=twelve[0],
                    adaptive_tried=twelve[1]))
                assert row == want
        # A candidate row was run at 20 to 60 m/s, and at 70 and 80 m/s
        # only where steps 5 and 6 ran it (wind_70_probes and
        # wind_80_probes, the ncar rows LA Santa Ana's parent grid reads
        # there): the stronger winds, and 70 and 80 m/s on every other
        # row, are cells it says nothing about.
        strong = [table.winds.index(w) for w in (90.0, 100.0)]
        assert all(not row.measured(k) for row in table.rows[len(merged):]
                   for k in strong)
        for probes, wind in (("wind_70_probes", 70.0),
                             ("wind_80_probes", 80.0)):
            ran = {(c["dx_m"], c["crest_m"], c["slope"], c["sound_steps"])
                   for c in document[probes]["cells_run"]
                   if c["settings"] == arm}
            column = table.winds.index(wind)
            for row in table.rows[len(merged):]:
                key = (row.dx_m, row.crest_m, row.slope, row.sound_steps)
                assert row.measured(column) == (key in ran), (key, wind)
    both = tl.candidate_map(("generated", "ncar"))
    # A shipped row both arms leave unchanged is read once.
    unchanged = sum(1 for a, b in zip(tl._arm_rows("generated")[0],
                                      tl._arm_rows("ncar")[0])
                    if a["arm"] is None and b["arm"] is None)
    assert len(both.rows) == (len(tl.candidate_map(("generated",)).rows)
                              + len(tl.candidate_map(("ncar",)).rows)
                              - unchanged)


@pytest.mark.parametrize("shipped, cand, merged", [
    # The candidate never ran the wind: the shipped cell.
    ((4.5, 5.0), (None, None), (4.5, 5.0)),
    # The shipped row tried nothing past the candidate: the candidate's,
    # even where it is shorter (lead decision (iii)) or longer.
    ((5.0, 5.0), (4.0, 6.5), (4.0, 6.5)),
    ((3.5, 5.0), (6.5, 6.5), (6.5, 6.5)),
    # The shipped 3 km rows were tried to 13.33 s/km: past the candidate's
    # top the shipped row still decides.
    ((4.5, 13.33), (6.5, 6.5), (6.5, 13.33)),
    ((8.0, 13.33), (6.5, 6.5), (8.0, 13.33)),
    ((None, 13.33), (6.5, 6.5), (6.5, 13.33)),
    # A longer rung ran away three hours in: a stop at the candidate's
    # entry, whatever the shipped half-hour row held.
    ((8.0, 13.33), (5.5, 6.5), (5.5, 13.33)),
    ((8.0, 13.33), (None, 6.5), (None, 13.33)),
])
def test_three_hour_evidence_wins_where_it_was_tried(shipped, cand, merged):
    assert tl.merge_cell(*shipped, *cand) == merged


def test_a_merged_row_reads_a_stop_past_the_candidate_s_top():
    """The breakage merge_cell keeps the shipped part for: a 3 km row the
    shipped map stopped at 8 s/km must not read as held at 10 s/km
    because the candidate only walked to 6.5 s/km."""
    winds = (20.0,)
    row = tc.MapRow(3000.0, 3000.0, 0.1, 4, (6.5,), (13.33,))
    table = tc.StableStepMap(winds, (5.0, 3.0), (row,), 10800.0)
    reading = tc.read_map(3000.0, 3000.0, 0.1, 20.0, 4, table)
    assert reading.stopped_per_km == 6.5


def test_a_cell_a_row_was_not_run_at_is_not_read():
    """A row with no run at a wind neither holds nor stops a step there."""
    winds = (20.0, 30.0)
    held = tc.MapRow(3000.0, 4500.0, 0.3, 4, (6.0, 6.0), (6.0, 6.0))
    unrun = tc.MapRow(3000.0, 4500.0, 0.2, 4, (6.0, None), (6.0, None))
    none = tc.MapRow(3000.0, 4500.0, 0.2, 4, (6.0, None), (6.0, 6.0))
    table = tc.StableStepMap(winds, (6.0, 3.0), (held, unrun), 10800.0)
    reading = tc.read_map(3000.0, 4400.0, 0.25, 30.0, 4, table)
    assert reading.per_km is None and not reading.held_everything_tried
    assert reading.beyond == ("wind",)
    assert reading.most_stable_per_km == 6.0
    assert reading.stopped_per_km is None
    # The same cell run, none of its steps holding, is no step held.
    table = tc.StableStepMap(winds, (6.0, 3.0), (held, none), 10800.0)
    assert tc.read_map(3000.0, 4400.0, 0.25, 30.0, 4, table).per_km is None


# ---------------------------------------------------------------------------
# The winds, per column, from each input.
# ---------------------------------------------------------------------------


def test_a_start_state_s_column_band_is_its_shipped_reading_at_one_top():
    state = _column_state(nz=6, ny=12, nx=14)
    state.u[2, 4, 6:8] = 70.0        # both faces of mass column (4, 6)
    state.u[4, 7, 3:5] = 95.0        # above a 3 km band, column (7, 3)
    start = tc.StartWinds.from_fields("d01", state.u, state.v, state.php,
                                      state.phb)
    for crest in (1000.0, 3000.0, 4500.0):
        column = start.column_band(np.full((12, 14), crest))
        speed, _height, _when = start.strongest(crest)
        assert np.nanmax(column) == pytest.approx(speed, rel=0, abs=0)
    # Per column, each to its own top: the 95 m/s level (5250 m centre,
    # above 3750) counts only under a column whose top reaches past 3750 m.
    tops = np.full((12, 14), 3000.0)
    tops[7, 3] = 6000.0
    column = start.column_band(tops)
    assert column[7, 3] == pytest.approx(np.hypot(95.0, 5.0))
    assert np.nanmax(np.delete(column.ravel(), 7 * 14 + 3)) == \
        pytest.approx(np.hypot(70.0, 5.0))


def test_a_boundary_s_column_band_is_its_shipped_reading_at_one_top():
    from woof.ingest.lateral_bc import build_lateral_boundaries
    from test_terrain_clock import _coupled

    first = _column_state()
    second = _column_state()
    second.u[1, :, :3] = 55.0
    second.v[1, -3:, :] = 40.0
    ny, nx = first.mub.shape
    msfu = np.full((ny, nx + 1), 1.02)
    msfv = np.full((ny + 1, nx), 1.02)
    boundaries = build_lateral_boundaries(
        [_coupled(first, msfu, msfv), _coupled(second, msfu, msfv)],
        [0.0, 10800.0], spec_bdy_width=5)
    geometry = tc.BoundaryGeometry(
        mub=first.mub, phb=first.phb, c1h=first.c1h, c2h=first.c2h,
        c1f=first.c1f, c2f=first.c2f, msfu=msfu, msfv=msfv)
    winds = tc.BoundaryWinds("d01", boundaries, geometry, 10800.0)
    column = winds.column_band(np.full((ny, nx), 3000.0))
    speed, _height, _when = winds.strongest(3000.0)
    assert np.nanmax(column) == pytest.approx(speed, rel=1e-12)
    # The slabs' own columns only: four cells in from each edge.
    assert np.isnan(column[5:-5, 5:-5]).all()
    assert np.isfinite(column[:, :4]).all() and np.isfinite(
        column[-4:, :]).all()
    # The west jet sits on the west columns, the north one on the north.
    assert column[6, 1] == pytest.approx(np.hypot(55.0, 5.0), rel=1e-6)
    assert column[-2, 6] == pytest.approx(np.hypot(10.0, 40.0), rel=1e-6)


# ---------------------------------------------------------------------------
# The faces, the margin and its cap.
# ---------------------------------------------------------------------------


def _fields(terrain, wind, *, radius, dx=3000.0):
    ny, nx = terrain.shape
    return tl.FaceFields(
        terrain=np.asarray(terrain, dtype=np.float64),
        msfu=np.ones((ny, nx + 1)), msfv=np.ones((ny + 1, nx)),
        wind=np.asarray(wind, dtype=np.float64),
        which=np.zeros((ny, nx), dtype=np.int16),
        sources=("d01 (start state)",), radius=radius, dx=dx, dy=dx)


def test_a_face_reads_its_own_window_and_the_margin_capped_at_the_domain():
    ny, nx = 20, 60
    terrain = np.zeros((ny, nx))
    terrain[:, 40:] = np.minimum(4500.0, (np.arange(20) + 1) * 1050.0)
    wind = np.full((ny, nx), 8.0)
    wind[:, :10] = 50.0                 # a jet 30 cells from the face
    fields = _fields(terrain, wind, radius=5)
    table = tl.candidate_map(("ncar",))
    local = tl.read_faces(fields, table=table, dx=3000.0, sound_steps=4,
                          domain_wind=50.0, arms=("ncar",), radius_m=15e3)
    steep = max(local.groups, key=lambda g: g.slope)
    assert steep.crest_m == 4500.0
    assert steep.input_wind_m_s == 8.0
    assert steep.wind_read_m_s == pytest.approx(8.0 * tl.WIND_MARGIN)
    assert not steep.wind_capped
    # Every face reads no more than the domain-wide wind or its own input
    # wind times the floor, whichever is more (tl.face_wind_read).
    assert max(g.wind_read_m_s for g in local.groups) <= max(
        50.0, tl.INPUT_WIND_FLOOR * 50.0)
    near = next(g for g in local.groups if g.input_wind_m_s == 50.0)
    assert near.wind_read_m_s == tl.INPUT_WIND_FLOOR * 50.0
    assert near.wind_capped


def test_faces_that_read_the_same_cells_read_once_and_the_least_governs():
    ny, nx = 12, 40
    terrain = np.zeros((ny, nx))
    terrain[:, 20:] = np.minimum(4500.0, (np.arange(20) + 1) * 1100.0)
    fields = _fields(terrain, np.full((ny, nx), 25.0), radius=4)
    table = tl.candidate_map(("generated",))
    local = tl.read_faces(fields, table=table, dx=3000.0, sound_steps=4,
                          domain_wind=60.0, arms=("generated",),
                          radius_m=12e3)
    assert sum(g.count for g in local.groups) == local.faces
    # Each group's reading is read_map's for its leading face.
    for g in local.groups:
        again = tc.read_map(3000.0, g.crest_m, g.slope, g.wind_read_m_s, 4,
                            table)
        assert again == g.reading
    limits = [tl._limit(g.reading) for g in local.groups]
    assert tl._limit(local.governing.reading) == min(limits)
    held = [g.reading.per_km for g in local.groups]
    assert local.combined.per_km == (None if None in held else min(held))


# ---------------------------------------------------------------------------
# Through the doors.
# ---------------------------------------------------------------------------


def _local(exp):
    return _clock(exp, "local_face")


def _measured(exp):
    """The shipped domain-wide clock, the default until 2.8.8."""
    return _clock(exp, "measured")


def _clock(exp, mode):
    return replace(exp, domains=tuple(
        replace(dc, run=replace(dc.run, terrain_clock=mode))
        for dc in exp.domains))


def _ridge_static(ny, nx, *, slope=0.35, dx=3000.0, crest=4500.0, at=40):
    """Flat ground, then a face of one slope up to a crest plateau."""
    rise = (np.arange(nx) - at) * slope * dx
    terrain = np.broadcast_to(np.clip(rise, 0.0, crest), (ny, nx)).copy()
    return MappingProxyType({"HGT_M": terrain,
                             "MAPFAC_U": np.ones((ny, nx + 1)),
                             "MAPFAC_V": np.ones((ny + 1, nx))})


def _state_with_jet(ny, nx, columns, jet=70.0):
    state = _column_state(nz=6, ny=ny, nx=nx)
    state.u[:3, :, columns] = jet
    return state


def test_a_steep_face_under_a_local_crest_level_jet_is_still_capped(capsys):
    from woof.prepared_domain_tree_forecast import _with_terrain_acoustics

    exp = _local(_wizard_experiment([(60, 90)], (), 3000.0))
    static = _ridge_static(60, 90)
    # A 50 m/s jet over the face itself (read at the 1.5 floor, 75 m/s,
    # inside the map's winds).
    state = _state_with_jet(60, 90, slice(30, 60), jet=50.0)
    derived = _with_terrain_acoustics(_metem_inputs(
        exp, {1: static}, {1: _cache_reader(state)}))
    row = derived.terrain_clock["domains"][0]
    assert row["clock"] == "local_face"
    assert row["status"] in {"ADAPTED", "BEYOND_MEASURED"}
    assert derived.experiment is not exp
    governing = row["local_faces"]["governing"]
    assert governing["slope"] == pytest.approx(0.35)
    assert governing["input_wind_m_s"] == pytest.approx(np.hypot(50.0, 5.0),
                                                        rel=1e-6)
    # The jet over the face is the domain's strongest, and the face is
    # read past it at the floor on its own wind (tl.INPUT_WIND_FLOOR).
    assert governing["wind_read_m_s"] == pytest.approx(
        tl.INPUT_WIND_FLOOR * governing["input_wind_m_s"])
    assert governing["wind_read_m_s"] > row["crest_level_wind_m_s"]
    # The run line names the floor, not the domain-wide reading, as what
    # the face was read at.
    line = capsys.readouterr().err
    assert "x its own input wind (the floor" in line, line
    assert ", the domain-wide reading;" not in line, line
    # Read face by face at the floor (75 m/s) the grid would halve its
    # 15 s step on four substeps; the shipped clock, on the 50 m/s
    # domain-wide reading, runs 15 s on six.  Never worse than shipped:
    # the grid keeps 15 s on six, and the receipt and line say so.
    never = row["never_worse"]
    assert never["applied"]
    # Gentle faces on candidate-only ridges have no 80 m/s run. The
    # full domain must say beyond measured, even when its steepest face
    # is measured, and keep the same shipped 15 s decision.
    assert never["reason"] == tl.NEVER_WORSE_BEYOND
    assert never["local_face"]["status"] == "BEYOND_MEASURED"
    assert never["shipped_measured"]["dt_s"]["seconds"] == 15.0
    assert row["status"] == "ADAPTED" == never["shipped_measured"]["status"]
    assert (row["dt_s"]["seconds"], row["time_step_sound"]) == (15.0, 6)
    assert derived.experiment.dt_exact(1) == exp.dt_exact(1)
    assert ("so d01 keeps the shipped measured clock's decision for this "
            "grid, 15 s steps on 6 acoustic substeps (ADAPTED)") in line


def test_a_jet_far_from_the_steep_face_no_longer_caps_it():
    """The defect: a jet over flat ground far from the face read on it."""
    from woof.prepared_domain_tree_forecast import _with_terrain_acoustics

    exp = _wizard_experiment([(60, 90)], (), 3000.0)
    static = _ridge_static(60, 90)
    # A 70 m/s jet over the flat ground 30 to 40 cells from the face.
    state = _state_with_jet(60, 90, slice(0, 10))
    shipped = _with_terrain_acoustics(_metem_inputs(
        _measured(exp), {1: static}, {1: _cache_reader(state)}))
    assert shipped.experiment.dt_exact(1) < exp.dt_exact(1)
    for local in (_local(exp), exp):
        # Named, and the bare default from 2.8.8: the same reading.
        derived = _with_terrain_acoustics(_metem_inputs(
            local, {1: static}, {1: _cache_reader(state)}))
        assert derived.experiment.dt_exact(1) == exp.dt_exact(1)
    row = derived.terrain_clock["domains"][0]
    assert row["status"] == "AS_CONFIGURED" and row["clock"] == "local_face"
    # Longer than the shipped clock's halved step: the local reading runs.
    assert not row["never_worse"]["applied"]
    assert row["never_worse"]["reason"] is None
    steep = max(row["local_faces"]["groups"], key=lambda g: g["slope"])
    assert steep["input_wind_m_s"] == pytest.approx(np.hypot(10.0, 5.0),
                                                    rel=1e-6)
    assert steep["wind_read_m_s"] == pytest.approx(
        tl.WIND_MARGIN * np.hypot(10.0, 5.0), rel=1e-6)


def test_a_pinned_run_keeps_the_shipped_advice():
    from woof.prepared_domain_tree_forecast import _with_terrain_acoustics

    exp = _wizard_experiment([(60, 90)], (), 3000.0)
    pinned = replace(exp, domains=tuple(
        replace(dc, run=replace(dc.run, terrain_clock="pinned"))
        for dc in exp.domains))
    static = _ridge_static(60, 90)
    state = _state_with_jet(60, 90, slice(0, 10))
    derived = _with_terrain_acoustics(_metem_inputs(
        pinned, {1: static}, {1: _cache_reader(state)}))
    row = derived.terrain_clock["domains"][0]
    assert row["clock"] == "pinned" and "local_faces" not in row
    assert derived.experiment.dt_exact(1) == exp.dt_exact(1)
    assert row["advice"]["step_division"] > 1


# ---------------------------------------------------------------------------
# NCAR's v4.4 CONUS benchmarks, from fixtures saved from the door itself.
# ---------------------------------------------------------------------------


def _load(name):
    import importlib.util
    import sys

    path = Path(__file__).resolve().parents[1] / "tools" / \
        "terrain_clock_local_check.py"
    spec = importlib.util.spec_from_file_location("tclc", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["tclc"] = module
    spec.loader.exec_module(module)
    return module.load_fixture(DATA / name)


def _ncar_run(meta):
    return SimpleNamespace(**{**meta["run"], "terrain_clock": "local_face"})


def _crest(meta):
    crest = meta["crest"]
    return tc.CrestWind(label="d01", crest_height_m=crest["crest_height_m"],
                        wind_m_s=crest["crest_level_wind_m_s"],
                        when=crest["when"], source=crest["source"],
                        height_m=crest["height_m"])


def _same(got, want):
    """Receipts equal, floats to 1e-12 relative.  Breakage the tolerance
    covers: the fixture was saved on Linux (numpy 2.5.3) and np.hypot's
    last bit differs on other platforms (the 12 km 0.0865 face's slope
    read 0.08652984551575497 on Windows numpy 2.2.6, ...496 on Linux)."""
    if isinstance(want, dict):
        assert isinstance(got, dict) and got.keys() == want.keys()
        for key in want:
            _same(got[key], want[key])
    elif isinstance(want, list):
        assert isinstance(got, list) and len(got) == len(want)
        for a, b in zip(got, want):
            _same(a, b)
    elif isinstance(want, float) and not isinstance(want, bool):
        assert got == pytest.approx(want, rel=1e-12)
    else:
        assert got == want


def test_ncar_12km_is_admitted_at_72_s_from_the_door_s_own_fields():
    """NCAR's 12 km grid at its published 72 s, read from the fields the
    WRF-input door itself read (test host, 2026-10-07, fix/step2/w1), on the
    rows with the 12-hour long-probe entries (fix/step4/LONG-PROBES.md)."""
    fields, meta = _load("ncar-conus12km-faces-d01.npz")
    door = json.loads((DATA / "ncar-conus12km-terrain-clock.json")
                      .read_text(encoding="utf-8"))["domains"][0]
    arms = tuple(meta["arms"])
    assert arms == ("ncar",)
    assert meta["margin"] == tl.WIND_MARGIN == 2.3
    faces = tl.DomainFaces(fields=fields, table=tl.candidate_map(arms),
                           arms=arms, radius_m=meta["radius_m"],
                           domain_wind=meta["domain_wind_m_s"],
                           dx=meta["run"]["dx"], margin=meta["margin"])
    adaptation = tl.derive_local(1, _ncar_run(meta), Fraction(72),
                                 meta["steepest_slope"], _crest(meta), faces)
    assert adaptation.dt == Fraction(72)
    assert adaptation.time_step_sound == 4
    assert adaptation.status == "AS_CONFIGURED"
    receipt = adaptation.receipt()
    # Never worse than shipped: the shipped clock halves it to 36 s, so
    # the face-by-face 72 s stands.
    assert not receipt["never_worse"]["applied"]
    assert receipt["never_worse"]["shipped_measured"]["dt_s"]["seconds"] \
        == 36.0
    # The fields read are the door's own.
    for key in ("neighbourhood_cells", "neighbourhood_m", "domain_wind_m_s",
                "faces_read", "wind_margin", "arms"):
        _same(receipt["local_faces"][key], door["local_faces"][key])
    assert door["dt_s"] == receipt["dt_s"]
    # On the 12-hour entries two neighbouring 3500 m rows (grid slopes
    # 0.0827 and 0.0881) dropped from 6.5 to 6.0 s/km at 50 m/s, so the
    # face that sets the limit is now the Owens Valley face at x (62,
    # 135), slope 0.080, under the same 3398 m local crest, read at the
    # domain's 48.4 m/s: 72 s held 12 hours and 78 s ran away, and 22
    # faces sit on that limit (one did on the three-hour rows).  No step
    # to spare.
    g = receipt["local_faces"]["governing"]
    assert g["face"] == {"axis": "x", "j": 135, "i": 62}
    assert g["slope"] == pytest.approx(0.08039, abs=1e-5)
    assert g["crest_m"] == pytest.approx(3398.39, abs=0.01)
    assert g["wind_capped_at_domain"]
    assert g["wind_read_m_s"] == pytest.approx(meta["domain_wind_m_s"])
    assert g["map_rows"] == {"crest_m": 3500.0, "slope": 0.0827,
                             "wind_m_s": 50.0, "beyond": []}
    assert g["held_s"] == 72.0 and g["longest_step_tried_s"] == 78.0
    for count in (4, 6):
        local = faces.reading(count)
        least = tl._limit(local.combined)
        assert least == 6.0, count
        assert sum(x.count for x in local.groups
                   if tl._limit(x.reading) == least) == 22, count
    # The shipped clock on the same inputs still halves it (the default
    # does not move on this lane).
    shipped = tc.derive_clock(1, SimpleNamespace(**{**meta["run"],
                                                    "terrain_clock":
                                                    "measured"}),
                              Fraction(72), meta["steepest_slope"],
                              _crest(meta))
    assert shipped.dt == Fraction(36)


def test_a_crest_rounded_to_4500_m_would_refuse_the_same_face():
    """What the 3500 m crest rows change: read on the 3000 and 4500 m rows
    alone, the same face (3398 m crest, slope 0.0915, 50 m/s) reads the
    4500 m ridge and refuses 72 s."""
    table = tl.candidate_map(("ncar",))
    without = replace(table, rows=tuple(r for r in table.rows
                                        if r.crest_m != 3500.0))
    reading = tc.read_map(12000.0, 3398.39, 0.09147, 48.39, 4, without)
    assert reading.crest_row == 4500.0
    assert reading.stopped_per_km is not None
    assert reading.stopped_per_km < 6.0
    with_rows = tc.read_map(12000.0, 3398.39, 0.09147, 48.39, 4, table)
    assert with_rows.crest_row == 3500.0
    assert with_rows.stopped_per_km == 6.0


def test_ncar_2p5km_keeps_15_s_on_six_substeps_from_the_door_s_face_groups():
    """The 2.5 km grid's own fields are 1901 x 1301 columns (an 18 MB
    fixture, kept with the evidence in CLOCK-CHECK-NCAR-2026-10-06/fix/
    step2/w1, where tools/terrain_clock_local_check.py reproduce read the
    door's receipt again from it); the suite keeps the face groups they
    read and the door's receipt, and reads each group's leading face again
    on the engine's own functions.

    On the 12-hour long-probe entries (fix/step4/LONG-PROBES.md) the 2 km,
    4500 m crest rows at 60 m/s stop at 6.0 s/km on four substeps and the
    ridge 0.05 and 0.06 cells hold only 5.5: NCAR's published 15 s on four
    substeps is refused (13.75 s), and 15 s holds on six, with no step to
    spare.  The full fixture reads ADAPTED, 15 s on 6 substeps
    (fix/step4/DECIDE.md)."""
    document = json.loads((DATA / "ncar-conus2p5km-groups.json")
                          .read_text(encoding="utf-8"))
    meta = document["meta"]
    door = document["door_receipt"]["domains"][0]
    arms = tuple(meta["arms"])
    assert arms == ("ncar",)
    assert meta["margin"] == tl.WIND_MARGIN == 2.3
    table = tl.candidate_map(arms)
    limits = {}
    for count, groups in document["counts"].items():
        readings = [tc.read_map(meta["run"]["dx"], g["crest_m"], g["slope"],
                                float(tl.face_wind_read(
                                    g["input_wind_m_s"],
                                    meta["domain_wind_m_s"])),
                                int(count), table)
                    for g in groups]
        combined, _governing = tl.combine(readings)
        limits[int(count)] = tl._limit(combined)
        for g in groups:
            # Each leading face is read as the door's reading reads it.
            # At the 1.5 floor one group's input (35.7 m/s) reads 53.6
            # m/s, past the 52.4 m/s domain-wide reading, in the same
            # 60 m/s column, so the limits do not move.
            assert g["wind_read_m_s"] == pytest.approx(float(
                tl.face_wind_read(g["input_wind_m_s"],
                                  meta["domain_wind_m_s"])), rel=1e-12)
    # 15 s at 2.5 km is 6.0 s/km; 13.75 s is 5.5.
    assert limits == {4: 5.5, 6: 6.0}
    assert door["dt_s"]["seconds"] == 15.0
    assert door["clock"] == "local_face"


@pytest.mark.parametrize("axis", ["x", "y"])
def test_the_run_route_reads_each_column_s_forcing_over_the_window(
        monkeypatch, axis):
    """The `woof run` door reads the decoded forcing on its own grid: under
    the candidate each domain column reads the source columns around it,
    over the window, with the grid's own map factors."""
    from datetime import timedelta

    from woof import runtime
    import woof.ingest.preflight as preflight
    from woof.static.projection import grids_from_projection_config
    from test_terrain_clock import _acoustic, _snapshot

    exp = _local(_wizard_experiment([(60, 60)], (), 3000.0, hours=3))
    start = exp.start_time
    snapshots = {start: _snapshot(start, 20.0),
                 start + timedelta(hours=3): _snapshot(
                     start + timedelta(hours=3), 72.0)}
    monkeypatch.setattr(runtime, "forcing_snapshots",
                        lambda data, catalog=None: snapshots)
    monkeypatch.setattr(preflight, "build_input_catalog", lambda data: None)
    grids = tuple(grids_from_projection_config(exp))
    # Each staggered map factor must govern a case. If x governed every
    # case, a missing V factor could hide under the larger x slope.
    jj, ii = np.indices((60, 60))
    terrain = {1: 4.0 * (ii if axis == "x" else jj)}
    sx, sy = face_slopes(terrain[1], 3000.0, 3000.0,
                         msfu=grids[0].mapfac_u(), msfv=grids[0].mapfac_v())
    acoustic = _acoustic(exp, {1: max(sx.max(), sy.max())})
    data = SimpleNamespace(forcing=[Path("gfs.grb2")])
    _adapted, adaptations = runtime._terrain_clock_for_case(
        exp, data, acoustic, terrain, grids, {})
    local = adaptations[0].faces_read
    assert adaptations[0].local_note is None and local is not None
    assert max(g.slope for g in local.groups) == max(sx.max(), sy.max())
    assert max(sx.max(), sy.max()) != pytest.approx(4.0 / 3000.0,
                                                  rel=1e-5)
    # Flat ground under a 3 km crest band reads the two lowest levels: the
    # 72 m/s level (5600 m) is above every column's band.
    assert max(g.input_wind_m_s for g in local.groups) == pytest.approx(10.0)
    assert {g.source for g in local.groups} == {"d01 (forcing over the "
                                                "window)"}
    # A 5 km crest takes the jet level into every column's band.
    terrain = {1: terrain[1] + 5000.0}
    _adapted, adaptations = runtime._terrain_clock_for_case(
        exp, data, acoustic, terrain, grids, {})
    local = adaptations[0].faces_read
    assert max(g.input_wind_m_s for g in local.groups) == pytest.approx(72.0)
    assert adaptations[0].local_note is None
    assert max(g.slope for g in local.groups) == max(sx.max(), sy.max())



# ---------------------------------------------------------------------------
# Step 4 (2026-10-07): the 12-hour entries, the blow-up-test adaptive rows,
# the input-wind floor, and the HRRR parent grids they decide
# (CLOCK-CHECK-NCAR-2026-10-06/fix/step4/DECIDE.md).
# ---------------------------------------------------------------------------


def test_the_12_hour_long_probe_entries_replace_the_three_hour_ones():
    """Breakage: 40 three-hour held verdicts ran away within 2.6 h on the
    wider domain a 12-hour probe uses; the map carries the 12-hour entries
    and lists every cell it changed."""
    document = tl.candidate_document()
    probes = document["long_probes"]
    assert probes["seconds"] == 43200.0
    assert probes["entries_shortened"] == 40
    winds = document["winds_m_s"]
    shortened = 0
    for cell in probes["patched"]:
        row = next(r for r in document["rows"]
                   if r["settings"] == cell["settings"]
                   and r["dx_m"] == cell["dx_m"]
                   and r["crest_m"] == cell["crest_m"]
                   and r["ridge_slope"] == cell["ridge_slope"]
                   and r["sound_steps"] == cell["sound_steps"])
        i = winds.index(cell["wind_m_s"])
        assert [row["stable_s_per_km"][i],
                row["top_s_per_km"][i]] == cell["twelve_hour"]
        # Only ever shorter: a 12-hour run never lengthened an entry.
        assert cell["twelve_hour"][0] <= cell["three_hour"][0]
        shortened += cell["twelve_hour"][0] < cell["three_hour"][0]
    assert shortened == 40
    # A NCAR 2.5 km deciding cell: 2 km, 4500 m, ridge 0.05, 60 m/s.
    row = next(r for r in document["rows"] if r["settings"] == "ncar"
               and r["dx_m"] == 2000.0 and r["crest_m"] == 4500.0
               and r["ridge_slope"] == 0.05 and r["sound_steps"] == 4)
    assert row["stable_s_per_km"][winds.index(60.0)] == 5.5


def test_the_adaptive_entries_are_the_12_hour_blow_up_rows():
    """Every one of the 224 rows (and step 5's eight 1500 m crest rows)
    lands on its arm's map row; a shipped row they do not cover keeps its
    shipped adaptive entries."""
    shipped = tc.measured_map()
    rows = tl.adaptive_document()["rows"]
    assert len(rows) == 224 + 8
    assert tl.adaptive_document()["step5_rows"]["rows"] == 8
    assert tl.adaptive_document()["adaptive"]["seconds"] == 43200.0
    for arm in ("generated", "ncar"):
        table = tl.candidate_map((arm,))
        for r in (r for r in rows if r["arm"] == arm):
            hits = [row for row in table.rows if row.sound_steps == 4
                    and row.dx_m == r["dx_m"]
                    and row.crest_m == r["crest_m"]
                    and abs(row.slope - r["slope"]) <= 5e-5]
            assert hits, (arm, r["dx_m"], r["crest_m"], r["ridge_slope"])
            for row in hits:
                assert row.adaptive == tuple(
                    None if v is None else float(v)
                    for v in r["adaptive_s_per_km"])
        # The 1500 m crest rows step 5 did not re-measure (every one at
        # the generated arm; at ncar all but the 2 and 3 km ridges 0.1 to
        # 0.25): shipped entries.
        for k, row in enumerate(shipped.rows):
            if row.crest_m == 1500.0 and row.adaptive is not None:
                if tl._adaptive_rows((arm,), row, shipped.winds) is None:
                    assert table.rows[k].adaptive == row.adaptive
                else:
                    assert arm == "ncar" and row.dx_m in (2000.0, 3000.0)


def _parent_grid(name, table=None):
    """A parent grid's face groups saved from the tree runner's preflight
    on test host (fix/step4, slim5.py), derived again on the engine's own
    functions: ``(document, adaptation)``."""
    doc = json.loads((DATA / name).read_text(encoding="utf-8"))
    run = SimpleNamespace(**doc["run"])
    arms = tuple(doc["arms"])
    table = tl.candidate_map(arms) if table is None else table

    def read(count):
        groups = doc["counts"][str(count)]
        return tl.combine([tc.read_map(doc["dx"], g["crest_m"], g["slope"],
                                       g["wind_read_m_s"], count, table)
                           for g in groups])[0]

    for groups in doc["counts"].values():
        for g in groups:
            assert g["wind_read_m_s"] == pytest.approx(float(
                tl.face_wind_read(g["input_wind_m_s"],
                                  doc["domain_wind_m_s"])), rel=1e-12)
    crest = tc.CrestWind(label="d01", **doc["crest"])
    adaptation = tc._derive_measured(1, run, Fraction(*doc["dt"]),
                                     doc["steepest_slope"], crest,
                                     table=table, read=read)
    return doc, adaptation


def test_sf_diablo_parent_grid_runs_uncapped_on_the_blow_up_adaptive_rows(
        monkeypatch):
    """SF Diablo 2021-01-18 12Z, d01 (2.25 km, max_time_step 13.5 s): every
    adaptive cell its faces read held the 15 s/km ladder top (33.75 s)
    for 12 h under the blow-up test, so no cap is written.  On the
    shipped adaptive entries (three hours, the retired old bound) the same
    faces read "none held", the verdict the step-3 review flagged."""
    doc, a = _parent_grid("sf-diablo-210118-48h-d01-groups.json")
    assert doc["arms"] == ["ncar"]
    assert a.adaptive and a.ceiling is None and a.division == 1
    assert a.status == "AS_CONFIGURED" == doc["decision"]["status"]
    assert a.cap_source == "adaptive held" and a.cap_per_km == 15.0
    r = a.reading
    assert r.adaptive_measured and r.adaptive_per_km == 15.0
    assert r.adaptive_stopped_per_km is None and not r.adaptive_none_held
    # Never worse than shipped: the shipped clock caps it at 11.25 s
    # (BEYOND_MEASURED), so the uncapped 13.5 s stands.
    run = SimpleNamespace(**doc["run"])
    on_shipped = tc._derive_measured(
        1, run, Fraction(*doc["dt"]), doc["steepest_slope"],
        tc.CrestWind(label="d01", **doc["crest"]))
    assert on_shipped.ceiling == Fraction(45, 4)
    assert on_shipped.status == "BEYOND_MEASURED"
    kept = tl.never_worse(a, on_shipped, run)
    assert not kept.never_worse.applied
    assert kept.ceiling is None and kept.status == "AS_CONFIGURED"
    assert tl.longest_step(kept, run) == Fraction(27, 2)
    # The 12-hour fixed rows alone limit it at its own 13.5 s maximum.
    assert tl._limit(r) * 2.25 == pytest.approx(13.5)
    # On the shipped adaptive entries instead: "none held".
    monkeypatch.setattr(tl, "_adaptive_rows", lambda *args: None)
    old = tl.candidate_map.__wrapped__(tuple(doc["arms"]))
    _doc, shipped = _parent_grid("sf-diablo-210118-48h-d01-groups.json",
                                 table=old)
    assert shipped.reading.adaptive_none_held
    assert shipped.status == "BEYOND_MEASURED"


def test_la_santa_ana_parent_grid_reads_beyond_the_map_at_the_1_5_floor():
    """LA Santa Ana 2025-01-07 12Z, d01 (2.25 km), step 6: at the 1.5
    input-wind floor (lead decision 2) the faces under the jet core are
    read at up to 73.9 m/s, the 80 m/s column.  Every cell they read there
    was run 12 h under the blow-up test with the 100 m/s peak rule (lead
    decision 3; the map's wind_80_probes), and three of them hold no step
    down to 3 s/km: 2 km, 4500 m crest, ridges 0.15 and 0.2 on four
    substeps and 0.2 on six, where the wave's peak |w| reaches 100 to
    104 m/s at every rung (fix/step6/APPLY.md).  So the map holds no step
    at the wind those faces are read at, and the derivation takes the
    most stable pair it measured at a weaker wind: 3.5 s/km on six
    substeps, 5 s steps capped at 7.87 s, BEYOND_MEASURED.

    The 70 m/s cell that set step 5's cap (3 km, 4500 m crest, ridge
    0.40, four substeps) held 4.5 s/km at 119 m/s; under the rule it holds
    4.0 (peak 50.5 m/s), and that is its 80 m/s entry too."""
    doc, a = _parent_grid("la-santaana-250107-48h-d01-groups.json")
    assert a.status == "BEYOND_MEASURED" == doc["decision"]["status"]
    assert a.ceiling == Fraction(787, 100)
    assert float(a.ceiling) == doc["decision"]["ceiling_s"]
    assert a.dt == Fraction(5) and a.division == 2
    assert a.time_step_sound == 6 == doc["decision"]["time_step_sound"]
    assert a.cap_source == "map" and a.limit_per_km == 3.5
    r = a.reading
    assert r.per_km is None and r.wind_row == 80.0 and not r.beyond
    assert r.most_stable_per_km == 3.5
    assert max(g["wind_read_m_s"] for g in doc["counts"]["4"]) ==         pytest.approx(1.5 * doc["domain_wind_m_s"])
    document = tl.candidate_document()
    winds = document["winds_m_s"]
    seventy, eighty = winds.index(70.0), winds.index(80.0)
    none_held = sorted((c["dx_m"], c["crest_m"], c["ridge_slope"],
                        c["sound_steps"])
                       for c in document["wind_80_probes"]["cells_run"]
                       if c["entry_s_per_km"] is None)
    assert none_held == [(2000.0, 4500.0, 0.15, 4), (2000.0, 4500.0, 0.2, 4),
                         (2000.0, 4500.0, 0.2, 6)]
    row = next(row for row in document["rows"]
               if row["settings"] == "ncar" and row["dx_m"] == 3000.0
               and row["crest_m"] == 4500.0 and row["ridge_slope"] == 0.4
               and row["sound_steps"] == 4)
    assert row["stable_s_per_km"][seventy] == 4.0
    assert row["top_s_per_km"][seventy] == 4.5
    assert row["stable_s_per_km"][eighty] == row["top_s_per_km"][eighty]         == 4.0
    # At the 1.25 floor on the same map (decision 3 alone) the grid would
    # read the 70 m/s column and hold 12.37 s on six substeps.
    table = tl.candidate_map(tuple(doc["arms"]))
    run = SimpleNamespace(**doc["run"])
    crest = tc.CrestWind(label="d01", **doc["crest"])

    def at_old_floor(count):
        return tl.combine([tc.read_map(
            doc["dx"], g["crest_m"], g["slope"],
            min(tl.WIND_MARGIN * g["input_wind_m_s"],
                max(doc["domain_wind_m_s"], 1.25 * g["input_wind_m_s"])),
            count, table) for g in doc["counts"][str(count)]])[0]

    old = tc._derive_measured(1, run, Fraction(*doc["dt"]),
                              doc["steepest_slope"], crest, table=table,
                              read=at_old_floor)
    assert old.status == "ADAPTED" and old.ceiling == Fraction(1237, 100)
    assert old.time_step_sound == 6


def test_a_face_is_never_read_under_1_5_x_its_own_input_wind():
    """The floor, raised from 1.25 to 1.5 (lead decision 2): at 1.25 it had
    no margin on the case it was built from, since two of LA's
    step-setting faces at f17 (input 43.21 m/s, actual 54.17 m/s) needed
    1.254 x their own input.  At 1.5 they read 64.8 m/s, a column above
    the one their actual wind reads, and LA's strongest step-setting
    input (49.28 m/s) reads 73.9 m/s."""
    assert tl.INPUT_WIND_FLOOR == 1.5
    domain = 49.28
    got = tl.face_wind_read(np.array([7.6, 30.0, 47.5, 49.28, 43.21]),
                            domain)
    np.testing.assert_allclose(got, [7.6 * 2.3, domain, 47.5 * 1.5,
                                     49.28 * 1.5, 43.21 * 1.5])
    assert 54.17 / 43.21 < tl.INPUT_WIND_FLOOR
    table = tl.candidate_map(("ncar",))
    face = dict(dx=2250.0, crest_m=3841.0, slope=0.339, sound_steps=4,
                table=table)
    f17 = tc.read_map(wind=float(got[4]), **face)
    actual = tc.read_map(wind=54.17, **face)
    capped = tc.read_map(wind=domain, **face)
    assert (capped.wind_row, actual.wind_row, f17.wind_row) == (
        50.0, 60.0, 70.0)
    assert tl._limit(f17) == 4.0 < tl._limit(actual) == 4.5 <         tl._limit(capped) == 5.0
    # The strongest input under the jet core reads the 80 m/s column.
    assert tc.read_map(wind=float(got[3]), **face).wind_row == 80.0


def test_ncar_2p5km_run_line_names_the_step_it_runs():
    """Breakage (fix/step4/REVIEW.md gap 3): NCAR's 2.5 km line said the
    map "holds steps up to 12.5 s there with 6 substeps" and then ran
    15 s.  12.5 s is the least held entry over every cell read,
    including cells never tried longer; the limit the clock applies is
    the shortest entry under a stop.  The line names that step first."""
    document = json.loads((DATA / "ncar-conus2p5km-groups.json")
                          .read_text(encoding="utf-8"))
    meta = document["meta"]
    door = document["door_receipt"]["domains"][0]
    table = tl.candidate_map(tuple(meta["arms"]))

    def read(count):
        return tl.combine([tc.read_map(
            meta["run"]["dx"], g["crest_m"], g["slope"],
            float(tl.face_wind_read(g["input_wind_m_s"],
                                    meta["domain_wind_m_s"])),
            int(count), table) for g in document["counts"][str(count)]])[0]

    dt = Fraction(door["configured_dt_s"]["numerator"],
                  door["configured_dt_s"]["denominator"])
    a = tc._derive_measured(1, _ncar_run(meta), dt, meta["steepest_slope"],
                            _crest(meta), table=table, read=read)
    assert (float(a.dt), a.time_step_sound) == (15.0, 6)
    assert (a.held_per_km, a.limit_per_km) == (5.0, 6.0)
    # Never worse than shipped: the shipped clock halves it to 7.5 s, so
    # 15 s on six stands.
    shipped = tc._derive_measured(1, _ncar_run(meta), dt,
                                  meta["steepest_slope"], _crest(meta))
    assert shipped.dt == Fraction(15, 2)
    kept = tl.never_worse(a, shipped, _ncar_run(meta))
    assert not kept.never_worse.applied
    assert (float(kept.dt), kept.time_step_sound) == (15.0, 6)
    line = a.sentence()
    assert ("the measured map lets d01 run steps up to 15 s there with 6 "
            "substeps (a step longer than 15 s stopped" in line)
    assert "tried no longer than 12.5 s" in line
    assert "holds steps up to 12.5 s" not in line


# ---------------------------------------------------------------------------
# Step 6 (2026-10-07): the 100 m/s peak rule and the 1.5 input-wind floor
# (lead decisions 3 and 2 after step 5; CLOCK-CHECK-NCAR-2026-10-06/fix/
# step6/APPLY.md).
# ---------------------------------------------------------------------------

EVIDENCE = Path(__file__).resolve().parents[1] / "docs" / "terrain-clock"


def _fixed_runs():
    """Every fixed-step probe run the lane keeps with the map: the 3 h
    sweep, step 5's cells and step 6's, as (cell, per_km, held, peak)."""
    sweep = json.loads((EVIDENCE / "terrain_clock_map_candidate-blowup-w1-"
                        "2026-10-07.json").read_text(encoding="utf-8"))
    for row in sweep["rows"]:
        for per_wind in row["per_wind"]:
            for run in per_wind["runs"]:
                yield ((row["settings"], row["dx_m"], row["crest_m"],
                        row["ridge_slope"], row["sound_steps"],
                        float(per_wind["wind"])), run["per_km"],
                       run["held"], run["peak_w"])
    for name in ("step5/terrain_clock_step5_cells-2026-10-07.json",
                 "step6/terrain_clock_step6_cells-2026-10-07.json"):
        doc = json.loads((EVIDENCE / name).read_text(encoding="utf-8"))
        for cell in doc["cells"]:
            if cell.get("kind", "fixed") != "fixed":
                continue
            for run in cell["runs"]:
                yield ((cell["settings"], cell["dx"], cell["crest"],
                        cell["ridge_slope"], cell["sound_steps"],
                        float(cell["wind"])), run["per_km"], run["held"],
                       run["peak_w"])


def test_no_fixed_entry_rests_on_a_run_past_100_m_s():
    """Breakage (lead decision 3): under ten times the old bound alone a
    run held at any peak w up to 200 m/s and more, and LA's cap rested on
    a 70 m/s cell held at 119 m/s.  Every recorded run that held at peak
    |w| of 100 m/s or more now sits above its cell's entry, and the entry
    rests on a run under 100 m/s."""
    document = tl.candidate_document()
    assert document["held_peak_w_m_s"] == 100.0
    winds = document["winds_m_s"]
    rows = {(r["settings"], r["dx_m"], r["crest_m"], r["ridge_slope"],
             r["sound_steps"]): r for r in document["rows"]}
    past, checked = [], 0
    held_at = {}
    for cell, per_km, held, peak in _fixed_runs():
        if held:
            held_at.setdefault(cell, {})[per_km] = max(
                peak, held_at.get(cell, {}).get(per_km, 0.0))
        if not held or peak is None or peak < 100.0:
            continue
        row = rows[cell[:5]]
        entry = row["stable_s_per_km"][winds.index(cell[5])]
        past.append(cell)
        assert entry is None or entry < per_km - 1e-9, (cell, per_km, entry)
    assert len(past) == 2
    for cell, by_rung in held_at.items():
        row = rows.get(cell[:5])
        if row is None or cell[5] not in winds:
            continue
        entry = row["stable_s_per_km"][winds.index(cell[5])]
        if entry is not None and entry in by_rung:
            checked += 1
            assert by_rung[entry] < 100.0, (cell, entry, by_rung[entry])
    assert checked > 1000
    moved = document["peak_rule"]["cells_moved"]
    assert len(moved) == 2
    for cell in moved:
        assert cell["entry_s_per_km"] < cell["was_s_per_km"]
        assert cell["rungs_run"][-1][1] and cell["rungs_run"][-1][2] < 100.0


def test_no_adaptive_entry_rests_on_a_run_past_100_m_s():
    """The adaptive rows: every row whose walk counted a run past 100 m/s
    was walked again under the rule, and no entry sits at or above a rung
    where such a run held."""
    adaptive = tl.adaptive_document()
    assert adaptive["held_peak_w_m_s"] == 100.0
    assert adaptive["step6_peak_rule"]["rows"] == 15
    winds = [float(w) for w in adaptive["winds_m_s"]]
    rows = {(r["arm"], r["dx_m"], r["crest_m"], r["ridge_slope"]): r
            for r in adaptive["rows"]}
    past = 0
    for name in ("step4/terrain_clock_adaptive_rows_evidence-2026-10-07.json",
                 "step5/terrain_clock_adaptive_rows_1500_evidence-"
                 "2026-10-07.json",
                 "step6/terrain_clock_adaptive_rows_step6_evidence-"
                 "2026-10-07.json"):
        doc = json.loads((EVIDENCE / name).read_text(encoding="utf-8"))
        for r in doc["rows"]:
            row = rows[(r["settings"], r.get("dx_m", r.get("dx")),
                        r.get("crest_m", r.get("crest")), r["ridge_slope"])]
            for run in r["runs"]:
                if not (run.get("held_as_recorded", run["held"])
                        and run["peak_w"] is not None
                        and run["peak_w"] >= 100.0):
                    continue
                assert not run["held"] or "held_as_recorded" not in run
                past += 1
                entry = row["adaptive_s_per_km"][winds.index(run["wind"])]
                assert entry is None or entry < run["per_km"] - 1e-9, (
                    r["settings"], r["ridge_slope"], run["key"], entry)
    assert past >= 46


# ---------------------------------------------------------------------------
# Never worse than the shipped measured clock (lead decision after step 6,
# CLOCK-CHECK-NCAR-2026-10-06/fix/step7/FALLBACK.md).
# ---------------------------------------------------------------------------


def _groups_faces(doc, table):
    """A parent grid's saved face groups as the derivation reads them
    (:class:`tl.DomainFaces`'s ``reading``), without the full fields."""

    def reading(count):
        combined, _governing = tl.combine([
            tc.read_map(doc["dx"], g["crest_m"], g["slope"],
                        g["wind_read_m_s"], count, table)
            for g in doc["counts"][str(count)]])
        return SimpleNamespace(
            combined=combined, governing=None,
            receipt=lambda dx, n=len(doc["counts"][str(count)]): {
                "groups_read": n})

    return SimpleNamespace(table=table, reading=reading)


def test_la_santa_ana_parent_grid_keeps_the_shipped_clock():
    """LA Santa Ana 2025-01-07 12Z, d01 (2.25 km).  Face by face at the 1.5
    floor the grid reads BEYOND_MEASURED (5 s steps capped at 7.87 s on
    six substeps, test above).  The shipped clock, on the domain-wide
    reading (slope 0.373, crest 3841 m, 49.3 m/s), runs 10 s steps capped
    at 10.12 s on six substeps: the step-3 48 h LA base run
    (fix/step3/PROVE.md, la-santaana-250107-48h-base) ran exactly that,
    17,280 root steps, finite at every hour with no recovery.  The grid
    keeps it, and the receipt says so."""
    doc = json.loads((DATA / "la-santaana-250107-48h-d01-groups.json")
                     .read_text(encoding="utf-8"))
    run = SimpleNamespace(**doc["run"])
    assert run.terrain_clock == "local_face"
    crest = tc.CrestWind(label="d01", **doc["crest"])
    faces = _groups_faces(doc, tl.candidate_map(tuple(doc["arms"])))
    a = tl.derive_local(1, run, Fraction(*doc["dt"]), doc["steepest_slope"],
                        crest, faces)
    assert a.status == "ADAPTED"
    assert (a.dt, a.division, a.time_step_sound) == (Fraction(10), 1, 6)
    assert a.ceiling == Fraction(253, 25)
    never = a.never_worse
    assert never.applied and never.reason == tl.NEVER_WORSE_BEYOND
    assert never.local.status == "BEYOND_MEASURED"
    assert (never.local.dt, never.local.ceiling) == (Fraction(5),
                                                     Fraction(787, 100))
    # The shipped decision is the shipped clock's own, on its own map.
    shipped = tc.derive_clock(1, SimpleNamespace(**{**doc["run"],
                                                    "terrain_clock":
                                                    "measured"}),
                              Fraction(*doc["dt"]), doc["steepest_slope"],
                              crest)
    assert (shipped.status, shipped.dt, shipped.ceiling,
            shipped.time_step_sound) == (a.status, a.dt, a.ceiling, 6)
    row = a.receipt()
    assert row["clock"] == "local_face"
    assert row["status"] == "ADAPTED"
    assert row["max_time_step_s"]["seconds"] == 10.12
    assert row["min_time_step_sound"] == 6
    # The faces recorded are the reading the rule replaced.
    assert row["local_faces"] == {"groups_read": len(doc["counts"]["6"])}
    receipt = row["never_worse"]
    assert receipt["applied"] and receipt["reason"] == "beyond_measured"
    assert receipt["local_face"]["status"] == "BEYOND_MEASURED"
    assert receipt["local_face"]["longest_step_s"]["seconds"] == 7.87
    assert receipt["shipped_measured"]["longest_step_s"]["seconds"] == 10.12
    line = never.sentence("d01")
    assert ("d01 would run 5 s steps on 6 acoustic substeps with an "
            "adaptive step capped at 7.87 s, which is past what the measured "
            "map holds (BEYOND_MEASURED), so d01 keeps the shipped measured "
            "clock's decision for this grid, 10 s steps on 6 acoustic "
            "substeps with an adaptive step capped at 10.12 s (ADAPTED)"
            ) in line, line


def _adaptation(*, division=1, sound=4, ceiling=None, adaptive=True,
                held=5.0):
    crest = tc.CrestWind(label="d01", crest_height_m=3000.0, wind_m_s=40.0,
                         when="start", source="start", height_m=3000.0)
    return tc.ClockAdaptation(
        grid_id=1, label="d01", slope=0.3, crest=crest, dx=2250.0,
        configured_dt=Fraction(10), configured_sound=4, division=division,
        time_step_sound=sound, held_per_km=held, reading=None,
        adaptive=adaptive,
        ceiling=None if ceiling is None else Fraction(ceiling))


@pytest.mark.parametrize("local, shipped, applied", [
    # A shorter cap than the shipped one.
    (dict(ceiling="9"), dict(ceiling="10.12"), True),
    # A divided first step, whatever the cap.
    (dict(division=2, ceiling="10.12"), dict(ceiling="10.12"), True),
    # Uncapped (13.5 s, the clock's own longest) against a cap: longer.
    (dict(), dict(ceiling="11.25"), False),
    # The same decision.
    (dict(ceiling="10.12", sound=6), dict(ceiling="10.12", sound=6), False),
    # A longer cap.
    (dict(ceiling="12.37", sound=6), dict(ceiling="10.12", sound=6), False),
])
def test_never_worse_takes_the_shipped_decision_only_where_it_is_longer(
        local, shipped, applied):
    run = SimpleNamespace(dx=2250.0, dy=2250.0, max_time_step=27,
                          max_time_step_den=2)
    a, b = _adaptation(**local), _adaptation(**shipped)
    got = tl.never_worse(a, b, run)
    assert got.never_worse.applied is applied
    want = b if applied else a
    assert (got.dt, got.ceiling, got.time_step_sound) == (
        want.dt, want.ceiling, want.time_step_sound)
    assert got.never_worse.local is a and got.never_worse.shipped is b
    if applied:
        assert got.never_worse.reason == tl.NEVER_WORSE_SHORTER


def test_never_worse_takes_the_shipped_decision_past_the_map():
    """A face-by-face reading past the map takes the shipped decision even
    where its own step is no shorter."""
    run = SimpleNamespace(dx=2250.0, dy=2250.0, max_time_step=27,
                          max_time_step_den=2)
    beyond = SimpleNamespace(per_km=None, beyond=(), adaptive_none_held=False,
                             most_stable_per_km=3.5)
    a = replace(_adaptation(division=1), reading=beyond)
    assert a.status == "BEYOND_MEASURED"
    b = _adaptation(ceiling="10.12", sound=6)
    got = tl.never_worse(a, b, run)
    assert got.never_worse.applied
    assert got.never_worse.reason == tl.NEVER_WORSE_BEYOND
    assert (got.ceiling, got.time_step_sound) == (Fraction(253, 25), 6)


# Startup grouping must preserve the selected face, including exact ties.
def _lexsort_face_leaders(keys, slope, wind, crest):
    order = np.lexsort((crest, wind, slope, keys))
    last = np.r_[keys[order][1:] != keys[order][:-1], True]
    return order[last], np.diff(np.r_[-1, np.flatnonzero(last)])


@pytest.mark.parametrize("size", [1, 4095, 4096, 20000])
def test_grouped_face_leaders_preserve_all_priorities_and_ties(size):
    rng = np.random.default_rng(20261018)
    keys = rng.integers(-15, 25, size) * 100000000
    values = [rng.integers(-3, 6, size).astype(float) for _ in range(3)]
    expected = _lexsort_face_leaders(keys, *values)
    actual = tl._grouped_face_leaders(keys, *values)
    for got, want in zip(actual, expected):
        np.testing.assert_array_equal(got, want)
    assert actual[1].sum() == size
    # Every face is tied: stable lexsort selects the last one.
    actual = tl._grouped_face_leaders(np.zeros(size, dtype=np.int64),
                                     *[np.zeros(size) for _ in range(3)])
    np.testing.assert_array_equal(actual[0], [size - 1])
    np.testing.assert_array_equal(actual[1], [size])


@pytest.mark.parametrize("special", [np.nan, np.inf, -np.inf, -0.0])
def test_grouped_face_leaders_preserve_nonfinite_and_signed_zero(special):
    size = 5000
    keys = np.arange(size) % 7
    values = [np.zeros(size) for _ in range(3)]
    for i, value in enumerate(values):
        value[i::11] = special
    expected = _lexsort_face_leaders(keys, *values)
    actual = tl._grouped_face_leaders(keys, *values)
    for got, want in zip(actual, expected):
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("arm", ["ncar", "generated"])
def test_grouped_face_reading_preserves_complete_receipt(monkeypatch, arm):
    from dataclasses import asdict
    rng = np.random.default_rng(20261018)
    terrain = rng.uniform(0, 4500, (70, 80))
    wind = rng.uniform(0, 65, terrain.shape)
    fields = _fields(terrain, wind, radius=4)
    kwargs = dict(table=tl.candidate_map((arm,)), dx=3000.0, sound_steps=4,
                  domain_wind=65.0, arms=(arm,), radius_m=12000.0)
    actual = tl.read_faces(fields, **kwargs)
    monkeypatch.setattr(tl, "_grouped_face_leaders", _lexsort_face_leaders)
    expected = tl.read_faces(fields, **kwargs)
    assert actual.faces > 4096
    assert asdict(actual) == asdict(expected)
