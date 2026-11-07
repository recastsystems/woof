"""A land cell HRRR's land mask has as sea takes the nearest HRRR land donor.

The breakage this guards: the hosted 750 m bay template (d01 94x92 at
2.25 km, d02 222x216 at 750 m, HRRR 2026-09-26 12Z) failed its d02
preparation on 2.7.7 with "no valid surface-matched HRRR donor within 8
cells for 2 target point(s)".  The two cells are a small offshore island
that the 750 m land-use table marks as land and HRRR's 3 km land mask
does not: every HRRR cell within 8 cells of them is sea, and the nearest
HRRR land cell is 10.99 cells (33 km) away on the mainland.  The search now keeps going past the configured radius and
takes that nearest land cell when the decoded window shows no nearer one
can exist, and the receipt names each such cell, its donor and the
distance.  The fixture below is the real HRRR land mask of that run's
decoded window and the real d02 geometry.
"""

from __future__ import annotations

import base64
from dataclasses import replace
from datetime import datetime
import zlib

import numpy as np
import pytest

import woof.ingest.hrrr as hrrr
from woof.ingest.hrrr import (HrrrNativeSnapshot, SurfaceDonorSearchError,
                               _build_masked_bilinear_stencil,
                               interpolate_hrrr_to_lambert)
from woof.static.lambert import LambertGrid


#: The decoded HRRR source window of the failing run (zero-based native
#: i=132..233, j=543..644), and its LANDSEA >= 0.5 as packed bits: the
#: land mask HRRR 2026-09-26 12Z f00 carries there, read back from the
#: 2.7.7 hierarchy stage's own snapshot.
_WINDOW_I_START = 132
_WINDOW_J_START = 543
_WINDOW_SHAPE = (102, 102)
_HRRR_LAND = (
    "eNqd0jFOxDAQhWFHLmY7XwDhI1DT4KtwBG5gJApuRUK15R6AYhdxAXcIYWXYtR3nFyINrmLJ"
    "n+eNJ8ac16Bl5cu3kbpJZePrZiqbWDfmn8SQWBK3SQ4g7eZK2rGJNxvc3IiQeJIIMpCcwxw7"
    "CarxzyoCMhy32lcQYfue7SuIJXEkEcSSOJII0mZUiZB4krBVZdDDSmQOK/EJRPcIlgOCJVZ5"
    "QrCMUdoTRulq5rmQK4cqDwHBXtGLPKPKdQC55SN/cC4CsiO5V33rwfZ8ZF8fo5zK8asTSS++"
    "Vwmnu9CJPuZeRWb8MD7f9GA7fR/1sxGveQ3WJlGJYpTLWAtZxqoca6ZP9BN+nk0ypvXiuRBH"
    "En4FG5cqcjmuvRf97kRBLHvxDKaoIiSeREEsiSOJIAOJkASQ9VT6AU/K8dg=")

#: The two island cells of d02 (row, column) and their recorded centres.
_ISLAND = ((120, 17), (120, 18))
_ISLAND_LATLON = ((37.70285139722228, -123.00954305541677),
                  (37.70291046559367, -123.0008852677941))

#: Their nearest HRRR land cell, zero-based native (i, j), and distances.
_DONOR_NATIVE = (164, 613)
_DONOR_CELLS = (10.98635956195316, 10.975995412961598)


def _hrrr_land() -> np.ndarray:
    bits = np.frombuffer(zlib.decompress(base64.b64decode(_HRRR_LAND)),
                         dtype=np.uint8)
    size = _WINDOW_SHAPE[0] * _WINDOW_SHAPE[1]
    return np.unpackbits(bits)[:size].reshape(_WINDOW_SHAPE).astype(bool)


def _nest_grid() -> LambertGrid:
    """d02 exactly as the template built it.

    d01 is 94x92 at 2.25 km with its reference point at its mass centre;
    d02 starts at parent cell (11, 11) with ratio 3, so the parent's
    reference point sits at d02 mass coordinate (111.5, 108.5).
    """
    return LambertGrid(
        37.620000000000005, -122.20000000000005, 27.62, 47.62,
        -122.20000000000005, 750.0, 750.0, 223, 217,
        known_x=111.5, known_y=108.5)


def _island_window_coordinates():
    lat, lon = _nest_grid().latlon_mass()
    source = hrrr.hrrr_source_grid()
    rows = np.array([row for row, _ in _ISLAND])
    cols = np.array([col for _, col in _ISLAND])
    sx, sy = source.latlon_to_ij(lat[rows, cols], lon[rows, cols])
    x = np.asarray(sx, dtype=np.float64) - 1.0 - _WINDOW_I_START
    y = np.asarray(sy, dtype=np.float64) - 1.0 - _WINDOW_J_START
    return x.reshape(1, 2), y.reshape(1, 2)


def test_the_fixture_is_the_run_it_names():
    lat, lon = _nest_grid().latlon_mass()
    assert lat.shape == (216, 222)
    for (row, col), (want_lat, want_lon) in zip(_ISLAND, _ISLAND_LATLON):
        assert lat[row, col] == pytest.approx(want_lat, abs=1e-10)
        assert lon[row, col] == pytest.approx(want_lon, abs=1e-10)
    land = _hrrr_land()
    donor_row = _DONOR_NATIVE[1] - _WINDOW_J_START
    donor_col = _DONOR_NATIVE[0] - _WINDOW_I_START
    assert land[donor_row, donor_col]
    # HRRR has only sea within 8 cells of either island cell.
    x, y = _island_window_coordinates()
    rows, cols = np.nonzero(land)
    for px, py in zip(x.ravel(), y.ravel()):
        nearest = np.sqrt(np.min((cols - px) ** 2 + (rows - py) ** 2))
        assert 8.0 < nearest < 11.0


@pytest.mark.requires_capability("masked_stencil_bridge")
def test_the_island_cells_take_the_nearest_hrrr_land_cell():
    """Fails on 2.7.7 with "no valid surface-matched HRRR donor within 8
    cells for 2 target point(s)"; the nearest land cell is found and
    shown to be the nearest, because the window reaches further from
    these cells than the donor is."""
    x, y = _island_window_coordinates()
    land = _hrrr_land()
    iy, ix, weights, report = _build_masked_bilinear_stencil(
        x, y, land, np.ones((1, 2), dtype=bool), fallback_radius=8)
    donor = (_DONOR_NATIVE[1] - _WINDOW_J_START,
             _DONOR_NATIVE[0] - _WINDOW_I_START)
    for col in range(2):
        assert (iy[0, 0, col], ix[0, 0, col]) == donor
        np.testing.assert_array_equal(weights[:, 0, col], [1, 0, 0, 0])
    assert report["fallback_target_count"] == 2
    assert report["unresolved_target_count"] == 0
    assert report["cross_surface_donor_count"] == 0
    assert report["fallback_max_distance_cells"] == pytest.approx(
        _DONOR_CELLS[0])
    assert report["fallback_distance_ceiling_histogram_cells"] == {"11": 2}
    assert report["distant_donor_count"] == 2
    listed = report["distant_donors"]
    assert [entry["target_index"] for entry in listed] == [[0, 0], [0, 1]]
    assert all(entry["source_index"] == list(donor) for entry in listed)
    assert [entry["distance_cells"] for entry in listed] == pytest.approx(
        list(_DONOR_CELLS))
    # Every one of them was shown to be the nearest, with room to spare.
    assert all(entry["distance_cells"] < entry["window_reach_cells"]
               for entry in listed)


@pytest.mark.requires_capability("masked_stencil_bridge")
def test_the_donor_does_not_depend_on_the_radius():
    x, y = _island_window_coordinates()
    land = _hrrr_land()
    apply = np.ones((1, 2), dtype=bool)
    beyond = _build_masked_bilinear_stencil(x, y, land, apply,
                                            fallback_radius=8)
    within = _build_masked_bilinear_stencil(x, y, land, apply,
                                            fallback_radius=11)
    for first, second in zip(beyond[:3], within[:3]):
        np.testing.assert_array_equal(first, second)
    # Named either way: 33 km is past the distance a donor is listed at.
    assert within[3]["distant_donor_count"] == 2


class _HostBackend:
    """A device-free preprocess backend (the shape tests/test_hrrr_target.py uses)."""

    name = "cpu-test"
    array_module = np

    @staticmethod
    def float32(value):
        return np.asarray(value, dtype=np.float32)

    @staticmethod
    def bool_array(value):
        return np.asarray(value, dtype=bool)

    @staticmethod
    def rotate_earth_to_grid(u, v, sina, cosa):
        u, v = np.asarray(u, np.float32), np.asarray(v, np.float32)
        sina, cosa = np.asarray(sina, np.float32), np.asarray(cosa, np.float32)
        return u * cosa + v * sina, v * cosa - u * sina

    @staticmethod
    def receipt():
        return {"backend": "cpu-test", "implementation": "numpy"}

    regular_plan = masked_nearest = era5_rh_to_water = staticmethod(
        lambda *_args, **_kwargs: None)
    prepare_wrf_vertical = staticmethod(lambda *_args, **_kwargs: None)


def _window_snapshot():
    """The run's window with its real land mask and a distinct soil
    column in every cell, so a donor can be told from its neighbours."""
    ny, nx = _WINDOW_SHAPE
    zeros_3d = np.zeros((50, ny, nx), dtype=np.float32)
    zeros_2d = np.zeros((ny, nx), dtype=np.float32)
    index = np.arange(ny * nx, dtype=np.float32).reshape(ny, nx)
    depth = np.arange(9, dtype=np.float32)[:, None, None]
    fields = {
        "PRES": np.full_like(zeros_3d, 80_000.0),
        "QC": zeros_3d.copy(), "QI": zeros_3d.copy(), "QR": zeros_3d.copy(),
        "QS": zeros_3d.copy(), "QG": zeros_3d.copy(),
        "HGT": np.full_like(zeros_3d, 1000.0),
        "TT": np.full_like(zeros_3d, 280.0),
        "SPFH": np.full_like(zeros_3d, 0.005),
        "PSFC": np.full_like(zeros_2d, 95_000.0),
        "SOILHGT": np.full_like(zeros_2d, 200.0),
        "SKINTEMP": np.full_like(zeros_2d, 288.0),
        "SNOW": zeros_2d.copy(), "SNOWH": zeros_2d.copy(),
        "T2": np.full_like(zeros_2d, 287.0),
        "Q2": np.full_like(zeros_2d, 0.006),
        "LANDSEA": _hrrr_land().astype(np.float32),
        "XICE": zeros_2d.copy(),
        "SOILT": (270.0 + 0.001 * index[None] + depth).astype(np.float32),
        "SOILW": (0.1 + 0.00004 * index[None] + 0.01 * depth).astype(
            np.float32),
        "U_MASS": zeros_3d.copy(), "V_MASS": zeros_3d.copy(),
        "U10_MASS": zeros_2d.copy(), "V10_MASS": zeros_2d.copy(),
    }
    return HrrrNativeSnapshot(
        valid_time=datetime(2026, 9, 26, 12), forecast_hour=0,
        i_start=_WINDOW_I_START, j_start=_WINDOW_J_START,
        ny=ny, nx=nx, fields=fields)


@pytest.mark.requires_capability("masked_stencil_bridge")
def test_the_nest_prepares_and_its_receipt_names_the_island(capsys, monkeypatch):
    """The whole mapping of d02's island, as the hierarchy stage runs it."""
    # The warning is said once per process for a (domain, cells) key, and
    # tests/test_masked_stencil_native.py maps this island too: run before
    # this file on the same worker it used the announcement up, and this
    # cell read no warning (test host 4's full CPU leg; reproduced by running
    # the two files in that order).  The cell asks for the first
    # announcement, so it starts from none.
    from woof.ingest import hrrr as hrrr_ingest
    monkeypatch.setattr(hrrr_ingest, "_DISTANT_DONORS_ANNOUNCED", set())
    snapshot = _window_snapshot()
    # Keep the actual mainland in this soil-donor fixture. Marking every
    # other target as water also asks the independent coastal temperature
    # mapper to find sea donors beneath inland mainland targets, beyond
    # this archived decoded window. That request is correctly refused.
    from woof.ingest.cpu_backend import CpuPreprocessBackend
    lat, lon = _nest_grid().latlon_mass()
    plan = hrrr._ProjectedCpuPlan(snapshot, lat, lon, CpuPreprocessBackend())
    landmask = plan.apply(snapshot.fields["LANDSEA"], method="nearest")
    for row, col in _ISLAND:
        landmask[row, col] = 1.0
    report: dict = {}
    mapped = interpolate_hrrr_to_lambert(
        snapshot, _nest_grid(), target_landmask=landmask,
        soil_mapping_report=report, surface_fallback_radius=8,
        backend=_HostBackend(), target_name="domain 2")

    donor_row = _DONOR_NATIVE[1] - _WINDOW_J_START
    donor_col = _DONOR_NATIVE[0] - _WINDOW_I_START
    for row, col in _ISLAND:
        np.testing.assert_array_equal(
            np.asarray(mapped.fields["SOILT"])[:, row, col],
            snapshot.fields["SOILT"][:, donor_row, donor_col])
        np.testing.assert_array_equal(
            np.asarray(mapped.fields["SOILW"])[:, row, col],
            snapshot.fields["SOILW"][:, donor_row, donor_col])

    stencil = report["land_stencil"]
    assert stencil["distant_donor_count"] == 2
    named = stencil["distant_donors"]
    assert [entry["target_index"] for entry in named] == [list(cell)
                                                         for cell in _ISLAND]
    for entry, (lat, lon), cells in zip(named, _ISLAND_LATLON, _DONOR_CELLS):
        assert entry["target_lat"] == pytest.approx(lat, abs=1e-9)
        assert entry["target_lon"] == pytest.approx(lon, abs=1e-9)
        assert entry["donor_hrrr_index"] == {"i": _DONOR_NATIVE[0],
                                             "j": _DONOR_NATIVE[1]}
        assert entry["donor_lat"] == pytest.approx(37.9989, abs=1e-3)
        assert entry["donor_lon"] == pytest.approx(-122.9931, abs=1e-3)
        assert entry["distance_cells"] == pytest.approx(cells)
        assert entry["distance_km"] == pytest.approx(cells * 3.0)
    warning = capsys.readouterr().err
    assert "warning:" in warning
    assert "domain 2" in warning
    assert "2 land cell(s)" in warning


@pytest.mark.requires_capability("masked_stencil_bridge")
def test_a_donor_the_window_cannot_show_is_nearest_is_still_refused():
    """The check keeps its purpose: nothing is taken on a guess.

    Land sits 12 cells south of the target cell; the window's north edge
    is 9 cells away, so a nearer land cell beyond it cannot be ruled out
    and the mapping refuses with the radius that would decide it.  When
    that edge is the source grid's own edge nothing lies beyond it, and
    the same donor is taken.
    """
    land = np.zeros((21, 31), dtype=bool)
    land[0, 15] = True
    x, y = np.array([[15.0]]), np.array([[12.0]])
    apply = np.ones((1, 1), dtype=bool)
    with pytest.raises(SurfaceDonorSearchError) as refusal:
        _build_masked_bilinear_stencil(x, y, land, apply, fallback_radius=8)
    assert refusal.value.unresolved_targets == ((0, 0),)
    assert refusal.value.required_radius_cells == 12
    assert "cannot be ruled out" in str(refusal.value)

    iy, ix, _weights, report = _build_masked_bilinear_stencil(
        x, y, land, apply, fallback_radius=8, closed_edges=("north",))
    assert (iy[0, 0, 0], ix[0, 0, 0]) == (0, 15)
    assert report["distant_donors"][0]["distance_cells"] == 12.0


@pytest.mark.requires_capability("masked_stencil_bridge")
def test_equidistant_far_donors_resolve_to_the_lowest_row_then_column():
    """The rule the radius scan has always used, kept past the radius."""
    x, y = np.array([[15.0]]), np.array([[11.0]])
    apply = np.ones((1, 1), dtype=bool)
    columns = np.zeros((23, 31), dtype=bool)
    columns[0, 12] = columns[0, 18] = True
    iy, ix, _w, _r = _build_masked_bilinear_stencil(
        x, y, columns, apply, fallback_radius=8)
    assert (iy[0, 0, 0], ix[0, 0, 0]) == (0, 12)
    rows = np.zeros((23, 31), dtype=bool)
    rows[0, 15] = rows[22, 15] = True
    iy, ix, _w, _r = _build_masked_bilinear_stencil(
        x, y, rows, apply, fallback_radius=8)
    assert (iy[0, 0, 0], ix[0, 0, 0]) == (0, 15)


@pytest.mark.requires_capability("masked_stencil_bridge")
def test_every_donor_is_the_nearest_land_cell_or_is_refused():
    """An independent witness over random masks: a cell with no land
    corner takes the nearest land cell (ties to the lowest row, then
    column) whenever that cell is nearer than the window's edge, and is
    refused otherwise; donors inside the radius are exactly the ones the
    radius scan always chose."""
    rng = np.random.default_rng(20260926)
    for trial in range(25):
        land = rng.random((40, 44)) < 0.02
        x = rng.uniform(6.0, 37.0, size=(3, 5))
        y = rng.uniform(6.0, 33.0, size=(3, 5))
        apply = np.ones(x.shape, dtype=bool)
        radius = int(rng.integers(2, 9))
        rows, cols = np.nonzero(land)
        expected, refused = {}, set()
        for r in range(x.shape[0]):
            for c in range(x.shape[1]):
                px, py = x[r, c], y[r, c]
                x0, y0 = int(np.floor(px)), int(np.floor(py))
                if land[y0:y0 + 2, x0:x0 + 2].any():
                    continue
                d2 = (cols - px) ** 2 + (rows - py) ** 2
                k = int(np.argmin(d2))
                reach = min(px + 1.0, 44 - px, py + 1.0, 40 - py)
                if d2[k] <= radius ** 2 or np.sqrt(d2[k]) < reach:
                    expected[(r, c)] = (rows[k], cols[k])
                else:
                    refused.add((r, c))
        try:
            iy, ix, _w, _report = _build_masked_bilinear_stencil(
                x, y, land, apply, fallback_radius=radius)
        except SurfaceDonorSearchError as error:
            assert set(error.unresolved_targets) == refused, trial
            for cell in refused:
                apply[cell] = False
            iy, ix, _w, _report = _build_masked_bilinear_stencil(
                x, y, land, apply, fallback_radius=radius)
        else:
            assert not refused, trial
        for (r, c), donor in expected.items():
            assert (iy[0, r, c], ix[0, r, c]) == donor, (trial, r, c)


def test_the_documented_radius_override_is_accepted(tmp_path):
    """The per-run override lives in the d01 target document
    (docs/native-wrf-direct-export.md).  2.7.7 refused a hand-set value
    there as drift from the configured experiment ("configured d01
    differs from the native target"); a different radius is not
    different geometry."""
    from tests.test_hrrr_configured_physics import _case
    from woof.hrrr_configuration import resolve_root_experiment

    exp, target, config, namelist, wps = _case(tmp_path)
    actual, _raw = resolve_root_experiment(
        target=replace(target, surface_fallback_radius_cells=11),
        vertical=exp.vertical, namelist_input=namelist,
        start_time=exp.start_time, run_seconds=exp.run_seconds,
        experiment_config=config, wps_namelist=wps)
    assert actual.root.run.nx == target.nx


@pytest.mark.requires_capability("masked_stencil_bridge")
@pytest.mark.parametrize("whole_grid", [True, False])
def test_a_distant_donor_receipt_on_the_whole_hrrr_grid_is_written(
        tmp_path, whole_grid):
    """A window that is all of HRRR's grid has nothing beyond any edge, so
    its reach is unlimited.  The receipt carried that as infinity and the
    prepared cache refused to write it ("Out of range float values are
    not JSON compliant: inf").  The receipt now says null, beside the four
    closed edges that explain it; a cropped window keeps its finite reach.
    """
    import json

    from woof.ingest.hrrr import _named_distant_donors, _native_edges_of
    from woof.ingest.hrrr_target import HRRR_SOURCE_NX, HRRR_SOURCE_NY
    from woof.ingest.prepared_cache import write_prepared_cache
    from test_prepared_cache import _fixture

    ny, nx = (HRRR_SOURCE_NY, HRRR_SOURCE_NX) if whole_grid else (41, 41)
    snapshot = HrrrNativeSnapshot(
        valid_time=datetime(2026, 9, 27), forecast_hour=0,
        i_start=0 if whole_grid else 700, j_start=0 if whole_grid else 400,
        ny=ny, nx=nx, fields={})
    land = np.zeros((ny, nx), dtype=bool)
    land[20, 30] = True
    x, y = np.array([[20.0]]), np.array([[20.0]])
    closed = _native_edges_of(snapshot)
    assert len(closed) == (4 if whole_grid else 0)
    iy, ix, weights, report = _build_masked_bilinear_stencil(
        x, y, land, np.ones((1, 1), dtype=bool), closed_edges=closed)
    assert (iy[0, 0, 0], ix[0, 0, 0]) == (20, 30)
    np.testing.assert_array_equal(weights[:, 0, 0], [1, 0, 0, 0])
    lat, lon = snapshot.source_cell_latlon(y, x)
    named = _named_distant_donors(report, snapshot, lat, lon, "domain 1")

    initial, met, boundaries = _fixture()
    write_prepared_cache(
        tmp_path / "cache", identity={"source": "hrrr-native"},
        initial_result=initial, met=met, boundaries=boundaries,
        metadata={"soil_mapping": {"land_stencil": named}})
    header = json.loads((tmp_path / "cache" / "header.json").read_text())
    saved = header["metadata"]["user"]["soil_mapping"]["land_stencil"]
    (entry,) = saved["distant_donors"]
    assert entry["distance_cells"] == 10.0
    assert entry["window_reach_cells"] == (None if whole_grid else 21.0)
    assert saved["closed_window_edges"] == sorted(closed)
