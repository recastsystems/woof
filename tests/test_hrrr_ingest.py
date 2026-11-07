"""CPU contracts for the native HRRR bridge loader and geometry."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from woof.ingest.hrrr import (
    HRRR_WPS_EQUIVALENT_DX_M,
    _build_masked_bilinear_stencil,
    _gate_forecast_hours,
    _wps_oned_cpu,
    hrrr_source_grid,
    load_hrrr_native_series,
    load_hrrr_native_window,
)
from woof.static.lambert import LambertGrid


_ATMOSPHERE_3D = (
    "PRES", "QC", "QI", "QR", "QS", "QG", "HGT", "TT", "SPFH",
    "U_MASS", "V_MASS",
)
_ATMOSPHERE_2D = (
    "PSFC", "SOILHGT", "SKINTEMP", "SNOW", "SNOWH", "T2", "Q2",
    "U10_MASS", "V10_MASS", "LANDSEA", "XICE",
)


def test_gate_accepts_absolute_contiguous_windows_through_cycle_horizon():
    for hours, cycle in (
            (tuple(range(49)), "2026-07-18 00:00:00"),
            (tuple(range(12, 19)), "2026-07-18 05:00:00"),
            (tuple(range(40, 47)), "2026-07-18 18:00:00")):
        gate = {
            "forecast_hours": ",".join(map(str, hours)),
            "series_count": str(len(hours)),
            "cycle": cycle,
        }
        assert _gate_forecast_hours(gate) == hours

    with pytest.raises(ValueError, match="contiguous, ordered"):
        _gate_forecast_hours({
            "forecast_hours": "12,13,15", "series_count": "3",
            "cycle": "2026-07-18 18:00:00"})
    with pytest.raises(ValueError, match="at least two"):
        _gate_forecast_hours({
            "forecast_hours": "12", "series_count": "1",
            "cycle": "2026-07-18 18:00:00"})
    with pytest.raises(ValueError, match="horizon f18"):
        _gate_forecast_hours({
            "forecast_hours": "18,19", "series_count": "2",
            "cycle": "2026-07-18 05:00:00",
        })


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


_CIMIXR_GATE = ("PASS discipline=0 category=1 parameter=82 "
                "level_type=105; finite/nonnegative/nonzero")


def _fake_bridge(root: Path, *, qice_mapping: str = _CIMIXR_GATE) -> str:
    root.mkdir()
    (root / "gate.txt").write_text(
        "status\tPASS\n"
        "cycle\t2026-07-18 00:00:00\n"
        "atmosphere_selected_per_time\t561\n"
        "hybrid_levels\t50\n"
        "soil_selected_per_time\t18\n"
        "window_zero_based_inclusive\ti=10..10 j=20..20\n"
        "window_shape\t1x1\n"
        f"qice_mapping\t{qice_mapping}\n"
        "cross_time_inventory\tPASS exact selected keys/levels/grid\n")
    for hour in (0, 1):
        atmosphere = root / f"atmosphere-f{hour:02d}"
        soil = root / f"soil-f{hour:02d}"
        atmosphere.mkdir()
        soil.mkdir()
        for index, name in enumerate(_ATMOSPHERE_3D):
            np.full((50, 1, 1), index + hour, dtype="<f4").tofile(
                atmosphere / f"{name}.f32le")
        for index, name in enumerate(_ATMOSPHERE_2D):
            np.full((1, 1), index + hour, dtype="<f4").tofile(
                atmosphere / f"{name}.f32le")
        for index, name in enumerate(("SOILT", "SOILW")):
            np.full((9, 1, 1), index + hour, dtype="<f4").tofile(
                soil / f"{name}.f32le")
    payloads = sorted(path for path in root.rglob("*") if path.is_file())
    lines = [f"{_hash(path)}  ./{path.relative_to(root).as_posix()}"
             for path in payloads]
    manifest = root / "SHA256SUMS"
    manifest.write_text("\n".join(lines) + "\n")
    return _hash(manifest)


def test_loader_requires_external_manifest_binding_and_exact_shapes(tmp_path):
    root = tmp_path / "bridge"
    manifest_hash = _fake_bridge(root)
    snapshot = load_hrrr_native_window(
        root, 1, expected_manifest_sha256=manifest_hash)
    assert snapshot.valid_time.isoformat() == "2026-07-18T01:00:00"
    assert (snapshot.i_start, snapshot.j_start, snapshot.ny, snapshot.nx) == (
        10, 20, 1, 1)
    assert snapshot.fields["PRES"].shape == (50, 1, 1)
    assert snapshot.fields["SOILT"].shape == (9, 1, 1)
    series = load_hrrr_native_series(
        root, (0, 1), expected_manifest_sha256=manifest_hash)
    assert [item.forecast_hour for item in series] == [0, 1]

    with pytest.raises(ValueError, match="SHA256SUMS hash mismatch"):
        load_hrrr_native_window(
            root, 0, expected_manifest_sha256="0" * 64)
    with (root / "gate.txt").open("a") as stream:
        stream.write("edited\tverdict\n")
    with pytest.raises(ValueError, match="payload hash mismatch"):
        load_hrrr_native_window(
            root, 0, expected_manifest_sha256=manifest_hash)


def test_loader_reads_a_bridge_that_bound_cloud_ice_to_cice(tmp_path):
    """A wrfnat file from before July 2018 publishes cloud ice as CICE.

    The bridge reads 0/6/0 where a file publishes no 0/1/82 and its gate
    names the code it bound.  The loader refused that gate, so a cycle
    the bridge had decoded still could not be loaded.
    """
    root = tmp_path / "bridge"
    manifest_hash = _fake_bridge(
        root, qice_mapping=("PASS discipline=0 category=6 parameter=0 "
                            "level_type=105; finite/nonnegative/nonzero"))
    snapshot = load_hrrr_native_window(
        root, 0, expected_manifest_sha256=manifest_hash)
    assert snapshot.fields["QI"].shape == (50, 1, 1)


@pytest.mark.parametrize("qice_mapping", [
    # Another cloud-category code, which QI is never read from.
    "PASS discipline=0 category=6 parameter=29 level_type=105; x",
    # The right code on the wrong level type.
    "PASS discipline=0 category=6 parameter=0 level_type=100; x",
    "FAIL discipline=0 category=1 parameter=82 level_type=105; x",
])
def test_loader_refuses_a_gate_binding_cloud_ice_to_any_other_code(
        tmp_path, qice_mapping):
    root = tmp_path / "bridge"
    manifest_hash = _fake_bridge(root, qice_mapping=qice_mapping)
    with pytest.raises(ValueError, match="none of the cloud-ice codes"):
        load_hrrr_native_window(
            root, 0, expected_manifest_sha256=manifest_hash)


def test_every_payload_is_still_verified_when_the_hashing_runs_in_parallel(
        tmp_path):
    """The concurrent hash sweep must not lose a late corruption.

    Payload verification runs on a thread pool, so a mismatch is discovered
    on whichever worker happens to reach it.  This corrupts the *last*
    payload the manifest lists -- the one a short-circuiting or
    first-result-only sweep would miss -- and demands the same refusal.
    """
    root = tmp_path / "bridge"
    manifest_hash = _fake_bridge(root)
    listed = [line.split(None, 1)[1].strip()
              for line in (root / "SHA256SUMS").read_text().splitlines()
              if line.strip()]
    last = [name for name in listed if not name.endswith("SHA256SUMS")][-1]
    victim = root / last.removeprefix("./")
    payload = bytearray(victim.read_bytes())
    payload[0] ^= 0xFF
    victim.write_bytes(bytes(payload))
    with pytest.raises(ValueError, match="payload hash mismatch"):
        load_hrrr_native_window(
            root, 0, expected_manifest_sha256=manifest_hash)


def test_radius_corrected_aligned_d01_maps_to_integer_hrrr_indices():
    source = hrrr_source_grid()
    target = LambertGrid(
        ref_lat=35.5028506728143,
        ref_lon=-98.002166928566,
        truelat1=38.5,
        truelat2=38.5,
        stand_lon=-97.5,
        dx=HRRR_WPS_EQUIVALENT_DX_M,
        dy=HRRR_WPS_EQUIVALENT_DX_M,
        e_we=200,
        e_sn=200,
    )
    lat, lon = target.latlon_mass()
    x, y = source.latlon_to_ij(lat, lon)
    expected_x, expected_y = np.meshgrid(
        np.arange(786.0, 985.0), np.arange(320.0, 519.0))
    np.testing.assert_allclose(x, expected_x, rtol=0.0, atol=3.0e-10)
    np.testing.assert_allclose(y, expected_y, rtol=0.0, atol=3.0e-10)


@pytest.mark.requires_capability("masked_stencil_bridge")
def test_masked_bilinear_is_convex_and_renormalizes_valid_land_only():
    source_land = np.array([[1, 0], [1, 0]], dtype=bool)
    target_land = np.ones((1, 1), dtype=bool)
    iy, ix, weights, report = _build_masked_bilinear_stencil(
        np.array([[0.25]]), np.array([[0.75]]), source_land, target_land)
    source = np.array([[0.2, -10.0], [0.6, -20.0]])
    mapped = sum(
        source[iy[corner], ix[corner]] * weights[corner]
        for corner in range(4))
    np.testing.assert_allclose(mapped, [[0.5]], rtol=0.0, atol=1.0e-7)
    np.testing.assert_allclose(np.sum(weights, axis=0), 1.0)
    assert np.all(weights >= 0.0)
    assert report["renormalized_target_count"] == 1
    assert report["fallback_target_count"] == 0


@pytest.mark.requires_capability("masked_stencil_bridge")
def test_masked_bilinear_uses_bounded_nearest_valid_fallback():
    source_land = np.zeros((5, 5), dtype=bool)
    source_land[0, 0] = True
    target_land = np.ones((1, 1), dtype=bool)
    iy, ix, weights, report = _build_masked_bilinear_stencil(
        np.array([[2.25]]), np.array([[2.25]]), source_land, target_land,
        fallback_radius=4)
    assert (iy[0, 0, 0], ix[0, 0, 0]) == (0, 0)
    np.testing.assert_array_equal(weights[:, 0, 0], [1.0, 0.0, 0.0, 0.0])
    assert report["fallback_target_count"] == 1
    assert report["fallback_max_distance_cells"] == pytest.approx(
        np.sqrt(2.0 * 2.25 ** 2))
    assert report["fallback_distance_ceiling_histogram_cells"] == {"4": 1}
    assert report["unresolved_target_count"] == 0
    assert report["cross_surface_donor_count"] == 0

    with pytest.raises(ValueError, match="no valid surface-matched"):
        _build_masked_bilinear_stencil(
            np.array([[2.25]]), np.array([[2.25]]), source_land,
            target_land, fallback_radius=1)


@pytest.mark.requires_capability("masked_stencil_bridge")
def test_unresolved_donor_search_reports_the_radius_that_works():
    """The refusal carries the facts remediation must be computed from.

    Field 2026-08: "no valid surface-matched HRRR donor within 8 cells
    for 2 target point(s)" was followed by advice to raise the radius,
    and the recommended raise was impossible.  Validating advice needs
    the failure to say WHAT radius would reach a donor and WHICH target
    cells failed -- both known here and nowhere else.
    """
    source_land = np.zeros((12, 12), dtype=bool)
    source_land[0, 0] = True
    target_land = np.ones((2, 2), dtype=bool)
    x = np.array([[0.0, 7.0], [0.0, 7.0]])
    y = np.array([[0.0, 0.0], [7.0, 7.0]])

    with pytest.raises(ValueError, match="no valid surface-matched") \
            as failure:
        _build_masked_bilinear_stencil(
            x, y, source_land, target_land, fallback_radius=8)
    error = failure.value
    # (0,0) resolves directly; (0,7) and (7,0) are exactly 7 away and
    # resolve by fallback; (7,7) is sqrt(98) = 9.90 away, so radius 10
    # is the smallest integer radius whose disk holds a valid donor.
    assert error.fallback_radius_cells == 8
    assert error.required_radius_cells == 10
    assert error.unresolved_targets == ((1, 1),)

    # Negative control: a window with no valid donor at ANY radius says
    # so, rather than inventing a radius that cannot work.
    with pytest.raises(ValueError, match="no valid surface-matched") \
            as hopeless:
        _build_masked_bilinear_stencil(
            np.array([[5.0]]), np.array([[5.0]]),
            np.zeros((12, 12), dtype=bool), np.ones((1, 1), dtype=bool),
            fallback_radius=8)
    assert hopeless.value.required_radius_cells is None
    assert hopeless.value.unresolved_targets == ((0, 0),)


@pytest.mark.requires_capability("masked_stencil_bridge")
def test_ohio_like_lake_edge_requires_explicit_radius_ten():
    source_land = np.zeros((12, 12), dtype=bool)
    source_land[0, 0] = True
    target_land = np.ones((1, 1), dtype=bool)
    x = np.array([[7.0]])
    y = np.array([[7.0]])

    with pytest.raises(ValueError, match="within 8 cells"):
        _build_masked_bilinear_stencil(
            x, y, source_land, target_land, fallback_radius=8)

    iy, ix, weights, report = _build_masked_bilinear_stencil(
        x, y, source_land, target_land, fallback_radius=10)
    assert (iy[0, 0, 0], ix[0, 0, 0]) == (0, 0)
    np.testing.assert_array_equal(
        weights[:, 0, 0], [1.0, 0.0, 0.0, 0.0])
    assert report["fallback_radius_cells"] == 10
    assert report["fallback_max_distance_cells"] == pytest.approx(np.sqrt(98.0))
    assert report["fallback_distance_ceiling_histogram_cells"] == {"10": 1}
    assert report["unresolved_target_count"] == 0
    assert report["cross_surface_donor_count"] == 0


#: The two HRRR 2 m specific-humidity stencils that refused a nested run.
#:
#: gpuwm 1.8.4, HRRR cycle 2026-08-08T09 f00, nested 12-3 km tree centred
#: 39.0,-103.0.  Both preparation stages passed and the tree forecast then
#: refused its own d02 input: "prepared near-surface surface_qv is outside
#: the physical range 0.0..0.2".  These are the source values behind it,
#: read out of that run's native bridge window -- HRRR's GRIB2 packing
#: quantises Q2 to 1e-5, so over the San Juan Mountains (source SOILHGT
#: 2557..3535 m) it decodes to EXACTLY zero beside neighbours three orders
#: of magnitude larger.  WPS routes SPECHUMD through the overshooting
#: sixteen_pt operator (METGRID.TBL
#: interp_option=sixteen_pt+four_pt+average_4pt), which undershoots such a
#: stencil below zero.  Real numbers, not a constructed contrast: nothing
#: synthetic reproduces how flat the dry side of these stencils is.
_SAN_JUAN_Q2_STENCILS = (
    # d02 (j=64, i=32); 37.75206 N, 106.58660 W; target terrain 3061.6 m
    {
        "stencil": np.array([
            [3.9e-04, 4.1e-04, 5.4e-04, 7.6e-04],
            [1.8e-04, 3.0e-05, 0.0e+00, 3.9e-04],
            [2.5e-04, 8.0e-05, 1.5e-04, 8.5e-04],
            [2.3e-04, 4.1e-04, 5.9e-04, 9.2e-04],
        ], dtype=np.float32),
        "fx": np.float32(0.2811026275),
        "fy": np.float32(0.513708055),
        "mapped": -1.785677523e-05,
    },
    # d02 (j=65, i=28); 37.77501 N, 106.72648 W; target terrain 3226.5 m
    {
        "stencil": np.array([
            [1.48e-03, 5.60e-04, 7.00e-05, 1.20e-04],
            [1.40e-04, 0.00e+00, 1.00e-05, 6.00e-05],
            [0.00e+00, 0.00e+00, 1.00e-05, 1.40e-04],
            [1.00e-04, 7.00e-05, 5.00e-05, 2.40e-04],
        ], dtype=np.float32),
        "fx": np.float32(0.2865612507),
        "fy": np.float32(0.767305851),
        "mapped": -1.478769718e-05,
    },
)


def _sixteen_point(stencil, fx, fy):
    """Run the shipped operator over one 4x4 stencil, as ``apply`` does.

    ``_ProjectedCpuPlan.apply`` substitutes 1e-20 for exact zeros before
    ``oned`` and maps an exact 1e-20 result back to zero; that is WPS's
    own REAL*4 quirk (interp_module.F:1255-1257,1299) and it is what
    makes a stencil containing zeros run the full overshooting parabolic
    instead of collapsing to WPS's ``b*c == 0`` zero.  Reproduced here so
    the pin exercises the same arithmetic the mapper does.
    """
    tiny = np.float32(1.0e-20)
    zero = np.float32(0.0)
    values = np.where(stencil == zero, tiny, stencil)
    rows = [_wps_oned_cpu(fx, *values[row]) for row in range(4)]
    result = _wps_oned_cpu(fy, *rows)
    return np.where(result == tiny, zero, result)


@pytest.mark.parametrize("case", _SAN_JUAN_Q2_STENCILS)
def test_wps_sixteen_point_undershoots_zero_valued_hrrr_surface_moisture(case):
    """The WPS operator really does go negative here -- pin it, do not fix it.

    METGRID.TBL puts SPECHUMD on ``sixteen_pt``, so this undershoot is
    what WPS produces and what ``real.exe`` then carries into ``grid%q2``
    unfloored (module_initialize_real.F:1157, :1257).  The engine's answer
    is a floor on the published surface value
    (:func:`woof.ingest.real._floor_flag_sh_surface_mixing_ratio`), NOT a
    quietly de-overshot operator: swapping this for a non-negative
    interpolator would change every HRRR field's numbers and diverge from
    WPS for no stated reason.  If this test starts failing because the
    result is no longer negative, the operator was changed.
    """
    mapped = _sixteen_point(case["stencil"], case["fx"], case["fy"])

    assert case["stencil"].min() == 0.0
    assert mapped < 0.0
    assert float(mapped) == pytest.approx(case["mapped"], rel=1e-6)


def test_zero_free_hrrr_surface_moisture_stencil_stays_positive():
    """The same operator on the same shape of stencil, minus the zeros.

    The control for the pin above: what makes those two cells negative is
    the exact zeros, not the terrain and not the operator on its own.  A
    stencil with the same span whose floor is a real HRRR value maps well
    clear of zero, which is why the flat-terrain Oklahoma tree at the same
    release completes.
    """
    case = _SAN_JUAN_Q2_STENCILS[0]
    lifted = np.maximum(case["stencil"], np.float32(8.0e-4))

    mapped = _sixteen_point(lifted, case["fx"], case["fy"])

    assert float(mapped) > 0.0


# ---------------------------------------------------------------------------
# The identity route: the target is the native grid, copied index for index.
# ---------------------------------------------------------------------------


def _native_target_grid(**updates) -> LambertGrid:
    """HRRR's own grid as its namelist.wps spells it (reference at the
    centre, 3 km on WPS's sphere)."""
    values = dict(ref_lat=38.5, ref_lon=-97.5, truelat1=38.5, truelat2=38.5,
                  stand_lon=-97.5, dx=3000.0, dy=3000.0, e_we=1800, e_sn=1060)
    values.update(updates)
    return LambertGrid(**values)


def _full_window_snapshot():
    from datetime import datetime

    from woof.ingest.hrrr import HrrrNativeSnapshot

    return HrrrNativeSnapshot(
        valid_time=datetime(2026, 10, 3, 12), forecast_hour=0,
        i_start=0, j_start=0, ny=1059, nx=1799, fields={})


def test_the_identity_route_copies_every_mass_point_exactly():
    from woof.ingest import hrrr
    from woof.ingest.hrrr import _ProjectedCpuPlan, _projected_index_geometry

    grid = _native_target_grid()
    snapshot = _full_window_snapshot()
    lat, lon = grid.latlon_mass()
    # Without the declaration the same target is refused: its outermost
    # row has no parabolic neighbour past the grid.
    with pytest.raises(ValueError, match="bilinear donor cell"):
        _projected_index_geometry(snapshot, lat, lon)
    plan = _ProjectedCpuPlan(snapshot, lat, lon, None, identity=True)
    assert plan.route == "identity"
    assert plan.operator == hrrr.PROJECTED_OPERATOR_NUMPY
    # Every point sits on a whole cell: fraction 0, or the cell before
    # with a unit fraction on the last row and column (what both
    # operators return exactly).
    rows, cols = np.indices((1059, 1799))
    expected_ix = np.where(cols == 1798, 1797, cols)
    expected_iy = np.where(rows == 1058, 1057, rows)
    assert np.array_equal(plan.ix, expected_ix)
    assert np.array_equal(plan.iy, expected_iy)
    assert np.array_equal(plan.fx, np.where(cols == 1798, 1.0, 0.0))
    assert np.array_equal(plan.fy, np.where(rows == 1058, 1.0, 0.0))
    field = (1.0 + cols + 3000.0 * rows).astype(np.float32)
    assert np.array_equal(plan.apply(field, method="parabolic"), field)
    assert np.array_equal(plan.apply(field, method="bilinear"), field)
    assert np.array_equal(plan.apply(field, method="nearest"), field)


def test_the_identity_route_puts_faces_between_cells_and_the_outer_face_on_the_edge():
    from woof.ingest.hrrr import _ProjectedCpuPlan

    grid = _native_target_grid()
    snapshot = _full_window_snapshot()
    u_lat, u_lon = grid.latlon_u()
    plan = _ProjectedCpuPlan(snapshot, u_lat, u_lon, None, identity=True)
    assert plan.fx.shape == (1059, 1800)
    # Interior faces: half way between two cells.
    assert np.all(plan.fx[:, 1:-1] == 0.5)
    assert np.array_equal(plan.ix[:, 1:-1],
                          np.broadcast_to(np.arange(1798), (1059, 1798)))
    # The outer faces, half a cell past the grid, take the edge cell.
    assert np.all(plan.ix[:, 0] == 0) and np.all(plan.fx[:, 0] == 0.0)
    assert np.all(plan.ix[:, -1] == 1797) and np.all(plan.fx[:, -1] == 1.0)
    rows, cols = np.indices((1059, 1799))
    field = (10.0 + 2.0 * cols).astype(np.float32)
    mapped = plan.apply(field, method="bilinear")
    assert np.array_equal(mapped[:, 0], field[:, 0])
    assert np.array_equal(mapped[:, -1], field[:, -1])
    assert np.allclose(mapped[:, 1:-1], 0.5 * (field[:, :-1] + field[:, 1:]))
    v_lat, v_lon = grid.latlon_v()
    plan = _ProjectedCpuPlan(snapshot, v_lat, v_lon, None, identity=True)
    assert plan.fy.shape == (1060, 1799)
    assert np.all(plan.fy[1:-1, :] == 0.5)
    assert np.all(plan.iy[0, :] == 0) and np.all(plan.fy[0, :] == 0.0)
    assert np.all(plan.iy[-1, :] == 1057) and np.all(plan.fy[-1, :] == 1.0)


def test_the_identity_route_refuses_a_point_off_the_lattice_and_a_cropped_window():
    from datetime import datetime

    from woof.ingest.hrrr import (IDENTITY_SNAP_LIMIT_CELLS,
                                   HrrrNativeSnapshot,
                                   _projected_index_geometry,
                                   _snap_to_native_lattice)

    # A 3 x 4 "native grid": mass points off their lattice positions by
    # a tenth of a cell are snapped and the drift is reported; one past
    # half a cell is refused.
    rows, cols = np.indices((3, 4), dtype=np.float64)
    snapped_x, snapped_y, distance = _snap_to_native_lattice(
        cols + 0.1, rows - 0.05, nx=4, ny=3)
    assert np.array_equal(snapped_x, cols) and np.array_equal(snapped_y, rows)
    assert distance == pytest.approx(0.1)
    off = cols.copy()
    off[-1, -1] += IDENTITY_SNAP_LIMIT_CELLS + 0.05
    with pytest.raises(ValueError, match="not that grid"):
        _snap_to_native_lattice(off, rows, nx=4, ny=3)
    # The u staggering: faces between cells, the outer faces on the edge.
    urows, ucols = np.indices((3, 5), dtype=np.float64)
    snapped_x, snapped_y, _ = _snap_to_native_lattice(
        ucols - 0.5, urows, nx=4, ny=3)
    assert snapped_x[0].tolist() == [0.0, 0.5, 1.5, 2.5, 3.0]
    with pytest.raises(ValueError, match="mass, u or v staggering"):
        _snap_to_native_lattice(np.zeros((2, 2)), np.zeros((2, 2)),
                                nx=4, ny=3)
    grid = _native_target_grid()
    lat, lon = grid.latlon_mass()
    cropped = HrrrNativeSnapshot(
        valid_time=datetime(2026, 10, 3, 12), forecast_hour=0,
        i_start=1, j_start=0, ny=1059, nx=1798, fields={})
    with pytest.raises(ValueError, match="whole native grid as its window"):
        _projected_index_geometry(cropped, lat, lon, identity=True)


def _identity_snapshot(nz: int = 3):
    """The whole native grid as a decoded window, every field distinct in
    every cell (so a copy can be told from a neighbour's value), with a
    land mask of west land / east water and a few lakes and islands."""
    from datetime import datetime

    from woof.ingest.hrrr import HrrrNativeSnapshot
    from woof.ingest.hrrr_target import HRRR_SOURCE_NX, HRRR_SOURCE_NY

    ny, nx = HRRR_SOURCE_NY, HRRR_SOURCE_NX
    rows, cols = np.indices((ny, nx), dtype=np.float64)
    unit = (cols + nx * rows) / float(nx * ny)  # 0 .. 1, distinct per cell
    level = np.arange(nz, dtype=np.float64)[:, None, None]
    depth = np.arange(9, dtype=np.float64)[:, None, None]
    land = cols < 0.6 * nx
    land[100:140, 200:260] = False      # a lake
    land[500:503, 1500:1502] = True     # an island
    land[:, -1] = True                  # the last column and row are land,
    land[-1, :] = True                  # where the stencil used to refuse

    def f32(value):
        return np.ascontiguousarray(value, dtype=np.float32)

    fields = {
        "PRES": f32(90_000.0 - 10_000.0 * level + 500.0 * unit[None]),
        "HGT": f32(500.0 + 1_000.0 * level + 300.0 * unit[None]),
        "TT": f32(280.0 - 5.0 * level + 10.0 * unit[None]),
        "SPFH": f32(0.002 + 0.004 * unit[None] + 0.0001 * level),
        "U_MASS": f32(3.0 + 2.0 * unit[None] + level),
        "V_MASS": f32(-1.0 + 4.0 * unit[None] - level),
        "PSFC": f32(95_000.0 + 1_000.0 * unit),
        "SOILHGT": f32(200.0 + 800.0 * unit),
        "SKINTEMP": f32(285.0 + 8.0 * unit),
        "SNOW": f32(np.where(rows > 0.8 * ny, 5.0 * unit, 0.0)),
        "SNOWH": f32(np.where(rows > 0.8 * ny, 0.02 * unit, 0.0)),
        "T2": f32(284.0 + 9.0 * unit),
        "Q2": f32(0.003 + 0.004 * unit),
        "U10_MASS": f32(1.0 + unit), "V10_MASS": f32(0.5 - unit),
        "LANDSEA": f32(land),
        "XICE": f32(rows > 0.95 * ny),
        "SOILT": f32(275.0 + 10.0 * unit[None] + 0.5 * depth),
        "SOILW": f32(0.1 + 0.3 * unit[None] + 0.01 * depth),
    }
    for index, name in enumerate(("QC", "QI", "QR", "QS", "QG")):
        fields[name] = f32(1.0e-5 * (index + 1) * unit[None] + 0.0 * level)
    return HrrrNativeSnapshot(
        valid_time=datetime(2026, 10, 3, 12), forecast_hour=0,
        i_start=0, j_start=0, ny=ny, nx=nx, fields=fields)


def _identity_target_landmask(source_land):
    """The model's land mask on the native grid: the source's, except a
    target land cell the source has as water (it takes the nearest source
    land cell) and a target water cell the source has as land (it takes
    the water fill), both on the grid's last column and row."""
    landmask = source_land.astype(np.float64)
    flipped_to_land = ((120, 230), (1000, 1700))   # in the lake, in the sea
    flipped_to_water = ((5, 1798), (1058, 7), (300, 300))
    for cell in flipped_to_land:
        landmask[cell] = 1.0
    for cell in flipped_to_water:
        landmask[cell] = 0.0
    return landmask, flipped_to_land, flipped_to_water


def check_identity_copy(mapped, snapshot, landmask, flipped_to_land,
                        flipped_to_water, host=np.asarray):
    """Every mass field, hydrometeor and soil column is a bit-for-bit copy
    of the source cell (target water: the documented fill; a target land
    cell over source water: the nearest source land cell).  Returns the
    count of compared arrays.  Shared with the GPU proof script."""
    source = snapshot.fields
    compared = 0
    for name in ("PRES", "HGT", "TT", "SPFH", "PSFC", "SOILHGT", "SKINTEMP",
                 "SNOW", "SNOWH", "T2", "Q2", "QC", "QI", "QR", "QS", "QG",
                 "XICE"):
        got = np.asarray(host(mapped.fields[name]), dtype=np.float32)
        assert got.tobytes() == source[name].tobytes(), name
        compared += 1
    land = landmask >= 0.5
    source_land = source["LANDSEA"] >= 0.5
    same = land & source_land
    skin = source["SKINTEMP"]
    for name, fill in (("SOILT", skin), ("SOILW", np.float32(1.0))):
        got = np.asarray(host(mapped.fields[name]), dtype=np.float32)
        want = source[name]
        assert got.shape == want.shape, name
        assert got[:, same].tobytes() == want[:, same].tobytes(), name
        water = ~land
        fill_values = np.broadcast_to(fill, water.shape)[water]
        assert np.array_equal(got[:, water],
                              np.broadcast_to(fill_values, got[:, water].shape)), name
        for row, col in flipped_to_land:
            # The nearest source land cell, ties to the lowest row then
            # column, exactly as the donor search orders them.
            land_rows, land_cols = np.nonzero(source_land)
            distance = (land_rows - row) ** 2 + (land_cols - col) ** 2
            best = np.flatnonzero(distance == distance.min())[0]
            donor = (land_rows[best], land_cols[best])
            assert np.array_equal(got[:, row, col],
                                  want[:, donor[0], donor[1]]), (name, row, col)
        compared += 1
    for name in ("UU", "VV", "U10", "V10"):
        assert np.isfinite(np.asarray(host(mapped.fields[name]))).all(), name
    return compared


@pytest.mark.requires_capability("masked_stencil_bridge")
def test_the_identity_route_maps_the_whole_native_grid_bit_for_bit():
    """The whole route on the whole native grid, as the HRRR preparation
    calls it: the land stencil is built (it refused the last column and
    row as leaving the window), and every mass field, hydrometeor and
    soil column comes out a copy of its own source cell."""
    from test_hrrr_island_donor import _HostBackend

    from woof.ingest.hrrr import interpolate_hrrr_to_lambert
    from woof.ingest.hrrr_target import native_grid_identity

    grid = _native_target_grid()
    assert native_grid_identity(grid)
    snapshot = _identity_snapshot()
    landmask, to_land, to_water = _identity_target_landmask(
        snapshot.fields["LANDSEA"] >= 0.5)
    report: dict = {}
    mapped = interpolate_hrrr_to_lambert(
        snapshot, grid, target_landmask=landmask, soil_mapping_report=report,
        surface_fallback_radius=24, backend=_HostBackend(),
        target_name="domain 1")
    assert check_identity_copy(mapped, snapshot, landmask, to_land,
                               to_water) == 19
    stencil = report["land_stencil"]
    assert stencil["fallback_target_count"] == len(to_land)


# The native grid's own specified-boundary strips (acceptance D-05, 2.8.8):
# the start state took the identity route, every boundary hour's strips
# were interpolated, and the full 1800 x 1060 grid died in prepare with
# "HRRR bridge window lacks the bilinear donor cell".

_STRIP_WIDTH = 10


def _native_strip_rectangles(ny=1059, nx=1799, width=_STRIP_WIDTH):
    return {
        "west": (0, ny, 0, width),
        "east": (0, ny, nx - width, nx),
        "south": (0, width, 0, nx),
        "north": (ny - width, ny, 0, nx),
    }


def _native_strip_grid(grid, y0, y1, x0, x1):
    """A strip exactly as tools/hrrr_single_domain_benchmark.py
    ``_boundary_mapping_targets`` cuts it."""
    return LambertGrid(
        grid.ref_lat, grid.ref_lon, grid.truelat1, grid.truelat2,
        grid.stand_lon, grid.dx, grid.dy, (x1 - x0) + 1, (y1 - y0) + 1,
        known_x=grid.known_x - x0, known_y=grid.known_y - y0,
        moad_cen_lat=grid.moad_cen_lat, moad_cen_lon=grid.moad_cen_lon)


def test_the_native_grids_boundary_strips_take_the_identity_route_cell_for_cell():
    from woof.ingest.hrrr import _ProjectedCpuPlan
    from woof.ingest.hrrr_target import native_lattice_placement

    grid = _native_target_grid()
    snapshot = _full_window_snapshot()
    full = {
        "mass": _ProjectedCpuPlan(snapshot, *grid.latlon_mass(), None,
                                  identity=True),
        "u": _ProjectedCpuPlan(snapshot, *grid.latlon_u(), None,
                               identity=True),
        "v": _ProjectedCpuPlan(snapshot, *grid.latlon_v(), None,
                               identity=True),
    }
    rows, cols = np.indices((1059, 1799))
    field = (1.0 + cols + 3000.0 * rows).astype(np.float32)
    for side, (y0, y1, x0, x1) in _native_strip_rectangles().items():
        strip = _native_strip_grid(grid, y0, y1, x0, x1)
        placement = native_lattice_placement(strip, parent=grid)
        assert (placement.i0, placement.j0, placement.nx, placement.ny) == (
            x0, y0, x1 - x0, y1 - y0), side
        crops = {
            "mass": (slice(y0, y1), slice(x0, x1)),
            "u": (slice(y0, y1), slice(x0, x1 + 1)),
            "v": (slice(y0, y1 + 1), slice(x0, x1)),
        }
        for stagger, (lat, lon) in (("mass", strip.latlon_mass()),
                                    ("u", strip.latlon_u()),
                                    ("v", strip.latlon_v())):
            # What every boundary hour did before: the interpolated route,
            # which the drift between the two spheres puts past the edge.
            with pytest.raises(ValueError, match="bilinear donor cell"):
                _ProjectedCpuPlan(snapshot, lat, lon, None)
            plan = _ProjectedCpuPlan(snapshot, lat, lon, None, identity=True,
                                     placement=placement)
            whole = full[stagger]
            crop = crops[stagger]
            for name in ("ix", "iy", "fx", "fy", "nearest_ix", "nearest_iy",
                         "x_host", "y_host"):
                assert np.array_equal(getattr(plan, name),
                                      getattr(whole, name)[crop]), (
                    side, stagger, name)
            for method in ("parabolic", "bilinear", "nearest"):
                assert np.array_equal(
                    plan.apply(field, method=method),
                    whole.apply(field, method=method)[crop]), (
                    side, stagger, method)


@pytest.mark.requires_capability("masked_stencil_bridge")
def test_the_native_grids_boundary_hour_maps_its_strips_as_crops_of_the_whole_map():
    """The boundary-hour route as the HRRR preparation runs it, on the
    native grid: each strip is a bit-for-bit crop of the start-state map
    (before, the first strip was refused and preparation died)."""
    from test_hrrr_island_donor import _HostBackend

    from woof.ingest.hrrr import interpolate_hrrr_to_lambert
    from tools.hrrr_single_domain_benchmark import (
        _boundary_mapping_targets, _map_boundary_snapshot)
    from types import SimpleNamespace

    grid = _native_target_grid()
    snapshot = _identity_snapshot()
    landmask, _to_land, _to_water = _identity_target_landmask(
        snapshot.fields["LANDSEA"] >= 0.5)
    whole = interpolate_hrrr_to_lambert(
        snapshot, grid, target_landmask=landmask, soil_mapping_report={},
        surface_fallback_radius=24, backend=_HostBackend(),
        target_name="domain 1")
    targets = _boundary_mapping_targets(
        grid, {"LANDMASK": landmask}, SimpleNamespace(nx=1799, ny=1059),
        width=_STRIP_WIDTH)
    report: dict = {}
    compact, _seconds = _map_boundary_snapshot(
        snapshot, targets, report, surface_fallback_radius=24,
        preprocess_backend=_HostBackend(), domain_grid=grid)
    compared = 0
    for side, (y0, y1, x0, x1) in _native_strip_rectangles().items():
        strip = compact[side].fields
        assert set(strip) == set(whole.fields), side
        for name, value in whole.fields.items():
            value = np.asarray(value)
            tail = value.shape[-2:]
            if tail == (1059, 1799):
                crop = value[..., y0:y1, x0:x1]
            elif tail == (1059, 1800):
                crop = value[..., y0:y1, x0:x1 + 1]
            elif tail == (1060, 1799):
                crop = value[..., y0:y1 + 1, x0:x1]
            else:
                raise AssertionError((side, name, value.shape))
            got = np.asarray(strip[name])
            assert got.shape == crop.shape, (side, name)
            assert got.tobytes() == np.ascontiguousarray(crop).tobytes(), (
                side, name)
            compared += 1
    assert compared == 4 * len(whole.fields)
