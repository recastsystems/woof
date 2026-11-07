"""The nested route pairs a PROJECTED source through its own projection.

The single-domain route always projected the target into a declared
source's plane (``woof.ingest.horiz.source_coordinate_transform``); the
hierarchy's donor-halo receipt did not, and compared the target's
geographic degrees against the source's projection axes.  Every
``max_dom = 2`` tree on a Lambert source was therefore refused as
uncovered -- an HRRR pressure-level case reported its 1799x1059 CONUS
grid as "lon 0..53.94 / lat 0..31.74" and placed an Oklahoma domain
thousands of columns west of it -- while the same case prepared fine on
one domain.

Nothing below names a source: the descriptor carries the projection and
the receipt reads it, so a geographic source takes the identity through
exactly the same code.
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from woof import mapped_source as ms
from woof.ingest.grib import Era5Snapshot
from woof.ingest.source_coverage import (SourceCoverageRefusal,
                                          SourceProjectionRefusal)
from woof.source_hierarchy import _spatial_coverage_receipt
from woof.static.lambert import LambertGrid


HRRR_PARAMETERS = {
    "latin1": 38.5, "latin2": 38.5, "lov": 262.5,
    "lat1": 21.138123, "lon1": 237.280472,
    "dx_m": 3000.0, "dy_m": 3000.0, "nx": 1799, "ny": 1059,
    "earth_radius_m": 6371229.0, "shape_of_earth": 6,
}


def _projected_snapshot(**parameter_overrides):
    parameters = dict(HRRR_PARAMETERS)
    parameters.update(parameter_overrides)
    latitude, longitude = ms._projected_axes(parameters)
    return Era5Snapshot(
        valid_time=datetime(2026, 8, 25, 18),
        levels_hpa=np.array([1000.0]),
        latitude=latitude,
        longitude=longitude,
        fields={},
        projection={
            "family": "lambert_conformal",
            "parameters": {
                **parameters, "axis_unit_m": ms.PROJECTED_AXIS_UNIT_M,
            },
        },
    )


def _target_grid(*, ref_lat, ref_lon, dx, e_we, e_sn):
    return LambertGrid(
        ref_lat=ref_lat, ref_lon=ref_lon, truelat1=30.0, truelat2=60.0,
        stand_lon=-97.5059, dx=dx, dy=dx, e_we=e_we, e_sn=e_sn)


def _experiment(count):
    return SimpleNamespace(
        domains=tuple(SimpleNamespace(grid_id=index + 1)
                      for index in range(count)))


def _expected_axis_index(parameters, lat, lon):
    """Where a lat/lon sits on the source's own axes, computed apart.

    The textbook spherical Lambert conformal conic forward map, written
    out from the projection's own definition and anchored on the declared
    grid's stated first point.  It shares no code with the tree: not the
    cone, not the transform, not the grid class.  The assertions below
    therefore pin the projection ARITHMETIC -- a wrong cone, a wrong axis
    origin, a unit or one-based slip -- and not merely that two callers
    of one function agree with each other.
    """

    degrees = np.pi / 180.0
    radius = float(parameters["earth_radius_m"])
    phi1 = float(parameters["latin1"]) * degrees
    phi2 = float(parameters["latin2"]) * degrees
    if abs(phi1 - phi2) <= 1.0e-12:
        cone = np.sin(phi1)
    else:
        cone = (np.log(np.cos(phi1) / np.cos(phi2))
                / np.log(np.tan(np.pi / 4.0 + phi2 / 2.0)
                         / np.tan(np.pi / 4.0 + phi1 / 2.0)))
    factor = np.cos(phi1) * np.tan(np.pi / 4.0 + phi1 / 2.0) ** cone / cone
    central = ((float(parameters["lov"]) + 180.0) % 360.0) - 180.0

    def plane(latitude, longitude):
        latitude = np.asarray(latitude, dtype=np.float64) * degrees
        delta = np.asarray(longitude, dtype=np.float64) - central
        delta = (delta + 180.0) % 360.0 - 180.0
        rho = radius * factor / np.tan(np.pi / 4.0 + latitude / 2.0) ** cone
        theta = cone * delta * degrees
        return rho * np.sin(theta), -rho * np.cos(theta)

    x0, y0 = plane(float(parameters["lat1"]),
                   ((float(parameters["lon1"]) + 180.0) % 360.0) - 180.0)
    x, y = plane(lat, lon)
    return ((y - y0) / float(parameters["dy_m"]),
            (x - x0) / float(parameters["dx_m"]))


def test_projected_source_maps_a_nested_tree_through_its_projection():
    snapshot = _projected_snapshot()
    parent = _target_grid(ref_lat=35.3115, ref_lon=-97.5059, dx=3000.0,
                          e_we=191, e_sn=153)
    # A real registration, not a second concentric grid: the child is
    # placed by the WPS parent_start/ratio arithmetic off the parent, so
    # what the receipt reads is what a max_dom = 2 tree hands it.
    child = parent.nest(36, 30, 3, 121, 109)
    receipt = _spatial_coverage_receipt(
        (snapshot, snapshot), (parent, child), _experiment(2), "declared")

    assert receipt["status"] == "PASS"
    assert receipt["source_shape"] == [1059, 1799]
    assert receipt["source_axis_plane"] == \
        "lambert_conformal plane in 100 km units"

    for grid, key in ((parent, "d01"), (child, "d02")):
        lat, lon = grid.latlon_mass()
        expected_y, expected_x = _expected_axis_index(
            HRRR_PARAMETERS, lat, lon)
        mass = receipt["domains"][key]["staggerings"]["mass"]
        np.testing.assert_allclose(
            mass["source_x_range"],
            [float(expected_x.min()), float(expected_x.max())],
            rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(
            mass["source_y_range"],
            [float(expected_y.min()), float(expected_y.max())],
            rtol=0.0, atol=1.0e-6)
        # The defect's signature: a CONUS domain landed thousands of
        # columns off the west edge because degrees were read as axis
        # units.  Both ends now sit inside the source with the donor
        # halo the receipt certifies.
        assert 0.0 < mass["source_x_range"][0]
        assert mass["source_x_range"][1] < 1798.0
        assert 0.0 < mass["source_y_range"][0]
        assert mass["source_y_range"][1] < 1058.0

    # The registration itself, read on the source's own axes: the child
    # sits strictly inside its parent's footprint there.
    outer = receipt["domains"]["d01"]["staggerings"]["mass"]
    inner = receipt["domains"]["d02"]["staggerings"]["mass"]
    assert outer["source_x_range"][0] < inner["source_x_range"][0]
    assert inner["source_x_range"][1] < outer["source_x_range"][1]
    assert outer["source_y_range"][0] < inner["source_y_range"][0]
    assert inner["source_y_range"][1] < outer["source_y_range"][1]


def test_a_geographic_source_still_takes_the_identity():
    """The same code, no projection: degrees against degrees, no new key."""

    latitude = np.arange(20.0, 60.0, 0.25)
    longitude = np.arange(-130.0, -60.0, 0.25)
    snapshot = Era5Snapshot(
        valid_time=datetime(2026, 8, 25, 18),
        levels_hpa=np.array([1000.0]),
        latitude=latitude, longitude=longitude, fields={})
    grid = _target_grid(ref_lat=35.3115, ref_lon=-97.5059, dx=12000.0,
                        e_we=101, e_sn=81)
    receipt = _spatial_coverage_receipt(
        (snapshot,), (grid,), _experiment(1), "declared")

    assert "source_axis_plane" not in receipt
    lat, lon = grid.latlon_mass()
    mass = receipt["domains"]["d01"]["staggerings"]["mass"]
    np.testing.assert_allclose(
        mass["source_x_range"],
        [float((lon.min() - longitude[0]) / 0.25),
         float((lon.max() - longitude[0]) / 0.25)], rtol=0.0, atol=1.0e-6)
    np.testing.assert_allclose(
        mass["source_y_range"],
        [float((lat.min() - latitude[0]) / 0.25),
         float((lat.max() - latitude[0]) / 0.25)], rtol=0.0, atol=1.0e-6)


def test_a_domain_off_a_projected_source_is_refused_in_the_source_plane():
    """A genuine miss still refuses -- naming the plane, not fake degrees."""

    snapshot = _projected_snapshot()
    grid = _target_grid(ref_lat=35.0, ref_lon=10.0, dx=3000.0,
                        e_we=101, e_sn=101)
    with pytest.raises(SourceCoverageRefusal) as raised:
        _spatial_coverage_receipt(
            (snapshot,), (grid,), _experiment(1), "declared")
    message = str(raised.value)
    assert "fall outside the source grid" in message
    assert "lambert_conformal plane in 100 km units" in message
    assert "at lat/lon" not in message
    # The plane's numbers are in no namelist; the domain's degrees are,
    # and the remedy is applied to those.
    corner_lat, corner_lon = (value[0, 0] for value in grid.latlon_mass())
    assert f"lat/lon ({corner_lat:.4f}, {corner_lon:.4f})" in message


def test_each_source_projection_has_independent_coverage():
    first = _projected_snapshot()
    second = _projected_snapshot(latin1=25.0)
    grid = _target_grid(ref_lat=35.3115, ref_lon=-97.5059, dx=3000.0,
                        e_we=191, e_sn=153)
    receipt = _spatial_coverage_receipt(
        (first, second, first), (grid,), _experiment(1), "declared")
    assert receipt["schema"].endswith("-v3")
    assert [r["forcing_indices"] for r in receipt["source_grids"]] == [[0, 2], [1]]
    for item, snapshot in zip(receipt["source_grids"], (first, second)):
        assert item["coverage"] == _spatial_coverage_receipt(
            (snapshot,), (grid,), _experiment(1), "declared")


def test_later_source_grid_still_requires_complete_target_coverage():
    from dataclasses import replace
    first = _projected_snapshot()
    second = replace(first, longitude=first.longitude + 1000)
    grid = _target_grid(ref_lat=35.3115, ref_lon=-97.5059, dx=3000.0,
                        e_we=191, e_sn=153)
    with pytest.raises(ValueError, match="outside|halo"):
        _spatial_coverage_receipt(
            (first, second), (grid,), _experiment(1), "declared")


def test_the_single_domain_route_refuses_in_the_plane_as_well(monkeypatch):
    """Fixed means default: the ONE-domain door names the plane too.

    The coverage refusal is raised inside whichever backend builds the
    plan, and a backend is handed bare axes by design, so it can name
    nothing.  The route's own pairing boundary -- the one place that
    holds the descriptor, the plane and the domain's degrees together --
    therefore raises it first.  The stub engine fails this test if a plan
    is ever built: reaching one means the user read the unlabelled
    message.
    """

    from woof.ingest import horiz
    from woof.ingest import preprocess_backend

    class _NoPlanEngine:
        array_module = np

        @staticmethod
        def regular_plan(*args, **kwargs):
            raise AssertionError(
                "the labelled refusal must fire at the pairing boundary, "
                "before any backend plan is built")

    monkeypatch.setattr(preprocess_backend, "resolve_preprocess_backend",
                        lambda *args, **kwargs: _NoPlanEngine())
    snapshot = _projected_snapshot()
    grid = _target_grid(ref_lat=35.0, ref_lon=10.0, dx=3000.0,
                        e_we=101, e_sn=101)
    with pytest.raises(SourceCoverageRefusal) as raised:
        horiz.interpolate_era5_to_lambert(snapshot, grid)
    message = str(raised.value)
    assert "lambert_conformal plane in 100 km units" in message
    corner_lat, corner_lon = (value[0, 0] for value in grid.latlon_mass())
    assert f"lat/lon ({corner_lat:.4f}, {corner_lon:.4f})" in message
    assert "at lat/lon" not in message


def test_a_geographic_source_reaches_the_backend_unchanged(monkeypatch):
    """The identity case takes no extra pass and raises nothing new."""

    from woof.ingest import horiz
    from woof.ingest import preprocess_backend

    built = []

    class _CountingEngine:
        array_module = np

        @staticmethod
        def regular_plan(*args, **kwargs):
            built.append(args[0])
            raise RuntimeError("stop at the first plan")

    monkeypatch.setattr(preprocess_backend, "resolve_preprocess_backend",
                        lambda *args, **kwargs: _CountingEngine())
    snapshot = Era5Snapshot(
        valid_time=datetime(2026, 8, 25, 18),
        levels_hpa=np.array([1000.0]),
        latitude=np.arange(20.0, 60.0, 0.25),
        longitude=np.arange(-130.0, -60.0, 0.25),
        fields={})
    grid = _target_grid(ref_lat=35.3115, ref_lon=-97.5059, dx=12000.0,
                        e_we=101, e_sn=81)
    with pytest.raises(RuntimeError, match="stop at the first plan"):
        horiz.interpolate_era5_to_lambert(snapshot, grid)
    assert len(built) == 1


def test_an_unevaluated_projection_family_is_refused_by_name():
    """A declared family with no transform has no reading at all."""

    from woof.ingest.horiz import (declared_source_projection,
                                    source_axis_space,
                                    source_coordinate_transform)

    snapshot = Era5Snapshot(
        valid_time=datetime(2026, 8, 25, 18),
        levels_hpa=np.array([1000.0]),
        latitude=np.arange(5.0), longitude=np.arange(5.0), fields={},
        # rotated_latitude_longitude was this test's example until the
        # HRDPS/RDPS rows taught the pairing that family (feat/hrdps-source);
        # polar stereographic is still unevaluated.
        projection={"family": "polar_stereographic",
                    "parameters": {}})
    for reader in (declared_source_projection, source_coordinate_transform,
                   source_axis_space):
        with pytest.raises(SourceProjectionRefusal) as raised:
            reader(snapshot)
        assert "polar_stereographic" in str(raised.value)
        assert "lambert_conformal" in str(raised.value)


def _small_projected_snapshot(**fields):
    snapshot = _projected_snapshot(nx=40, ny=30)
    return Era5Snapshot(
        valid_time=snapshot.valid_time, levels_hpa=snapshot.levels_hpa,
        latitude=snapshot.latitude, longitude=snapshot.longitude,
        fields=fields, projection=snapshot.projection)


def test_a_projected_source_samples_a_water_overlay_in_degrees():
    """The overlay is geographic; the source's cells must reach it as such."""

    from woof.ingest import water_overlay

    snapshot = _small_projected_snapshot()
    rows = np.array([0, 15, 29])
    cols = np.array([0, 20, 39])
    latitude, longitude = water_overlay._source_cell_coordinates(
        snapshot, rows, cols)
    source = ms.declared_lambert_source_grid(
        {**HRRR_PARAMETERS, "nx": 40, "ny": 30})
    expected_lat, expected_lon = source.ij_to_latlon(
        cols.astype(np.float64) + 1.0, rows.astype(np.float64) + 1.0)
    np.testing.assert_allclose(latitude, expected_lat, rtol=0.0, atol=1.0e-9)
    np.testing.assert_allclose(longitude, expected_lon, rtol=0.0, atol=1.0e-9)
    # Degrees over North America, not the plane's own 0..1.2 axis values.
    assert float(np.min(latitude)) > 20.0
    assert float(np.max(longitude)) < -60.0


def test_the_water_overlay_keeps_the_projection_descriptor():
    """Dropping it would re-read the axes as degrees one call later."""

    from pathlib import Path

    from woof.ingest.water_overlay import (WaterTemperatureOverlay,
                                            apply_water_temperature_overlay)

    snapshot = _small_projected_snapshot(
        LANDSEA=np.zeros((30, 40)), SKINTEMP=np.full((30, 40), 280.0))
    overlay = WaterTemperatureOverlay(
        path=Path("overlay.nc"), source_format="test", variable="sst",
        declared_units="K",
        latitude=np.linspace(20.0, 50.0, 31),
        longitude=np.linspace(-130.0, -60.0, 71),
        temperature_k=np.full((31, 71), 291.0),
        valid=np.ones((31, 71), dtype=bool))
    rebuilt, receipt = apply_water_temperature_overlay(snapshot, overlay)
    assert rebuilt.projection == snapshot.projection
    # The cells landed inside a CONUS overlay window, which only happens
    # if they were carried back to degrees before sampling.
    assert receipt["replaced_cells"] == receipt["water_cells"]
    np.testing.assert_allclose(rebuilt.fields["SKINTEMP"], 291.0,
                               rtol=0.0, atol=1.0e-9)


def test_preflight_coverage_pairs_a_projected_source_through_its_plane():
    """The catalog's bounds are the source's axes, so the target moves."""

    from woof.ingest.horiz import (source_axis_space,
                                    source_coordinate_transform)
    from woof.ingest.preflight import SpatialCoverage

    snapshot = _projected_snapshot()
    coverage = SpatialCoverage(
        shape=(snapshot.latitude.size, snapshot.longitude.size),
        latitude_min=float(snapshot.latitude.min()),
        latitude_max=float(snapshot.latitude.max()),
        longitude_min=float(snapshot.longitude.min()),
        longitude_max=float(snapshot.longitude.max()),
        latitude_order="ascending", longitude_order="ascending",
        projection=snapshot.projection)
    # The pairing helpers read a descriptor off whatever carries one,
    # which is why the coverage record carries it: the comparison site
    # holds the record and not the snapshot.
    transform, projected = source_coordinate_transform(coverage)
    assert projected
    assert source_axis_space(coverage) == \
        "lambert_conformal plane in 100 km units"
    grid = _target_grid(ref_lat=35.3115, ref_lon=-97.5059, dx=3000.0,
                        e_we=191, e_sn=153)
    target_lat, target_lon = grid.latlon_mass()
    y, x = transform(target_lat, target_lon)
    assert coverage.latitude_min <= float(np.min(y))
    assert float(np.max(y)) <= coverage.latitude_max
    assert coverage.longitude_min <= float(np.min(x))
    assert float(np.max(x)) <= coverage.longitude_max
    # Untransformed, the same domain reads far off the west edge: that
    # is the comparison this pairing removes.
    assert float(np.min(target_lon)) < coverage.longitude_min


def test_a_tree_rooted_on_hrrrs_own_grid_takes_the_identity_route():
    """The HRRR recipe's root IS the 1799 x 1059 source lattice (WPS's
    sphere against the GRIB header's, drifting to 0.35 cell at the far
    corner); with a child inside it, the tree used to be refused: "lacks the
    complete parabolic donor halo for d01 mass" (box D 2026-10-21), while the
    single-domain route prepared the same root.  The root now copies its own
    cells; the child still proves its interpolation halo."""
    snapshot = _projected_snapshot(same_grid_pairing="identity")
    root = LambertGrid(ref_lat=38.5, ref_lon=-97.5, truelat1=38.5,
                       truelat2=38.5, stand_lon=-97.5, dx=3000.0, dy=3000.0,
                       e_we=1800, e_sn=1060)
    child = root.nest(900, 500, 3, 121, 109)
    receipt = _spatial_coverage_receipt(
        (snapshot,), (root, child), _experiment(2), "declared")
    assert receipt["status"] == "PASS"
    assert receipt["domains"]["d01"]["pairing"] == "identity"
    assert "parabolic_donor_x" in receipt["domains"]["d02"]["staggerings"]["mass"]
    # Without the identity declaration the same root is still refused.
    with pytest.raises(ValueError, match="outside the source grid|parabolic donor halo"):
        _spatial_coverage_receipt((_projected_snapshot(),), (root, child),
                                  _experiment(2), "declared")
