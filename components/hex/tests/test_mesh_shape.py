"""Drawn regional cores (woof.hex.mesh_shape): the grown outline is the true offset set."""

import json
import math

import numpy as np
import pytest

from woof.hex import mesh_shape as ms
from woof.hex import mesh_point


def _arc_km(row, lat, lon):
    v = ms._xyz(*np.asarray(row["vertices_deg"]).T)
    return np.degrees(np.arccos(np.clip(v @ ms._xyz(lat, lon), -1, 1))) * math.pi / 180 * ms.EARTH_RADIUS_KM


def test_a_circle_grows_by_exactly_the_distance():
    circle = ms.ellipse_row(35.6, -97.5, 240.0, 240.0, 0.0)
    grown = ms.grow(circle, 67.5, tolerance_km=1.875)
    r = _arc_km(grown, 35.6, -97.5)
    assert abs(r.min() - 307.5) < 0.5 and abs(r.max() - 307.5) < 0.5


def test_a_concave_outline_grows_into_a_single_ring_with_its_notch_filled():
    # A U whose notch (60 km wide) closes when grown by 40 km.
    lat0, lon0 = 40.0, -100.0
    proj = ms._Azimuthal(lat0, lon0)
    xy = np.array([[-150, -100], [150, -100], [150, 100], [30, 100], [30, -20], [-30, -20],
                   [-30, 100], [-150, 100]], dtype=float)
    lat, lon = proj.inverse(xy[:, 0], xy[:, 1])
    u = {"kind": "polygon", "vertices_deg": [[float(a), float(b)] for a, b in zip(lat, lon)]}
    grown = ms.grow(u, 40.0, tolerance_km=1.0)
    gx, gy = proj.forward(*np.asarray(grown["vertices_deg"]).T)
    # Every grown vertex is 40 km from the U (to the tolerance and the grid).
    dense = ms._densify(ms._xyz(*np.asarray(u["vertices_deg"]).T), 0.5 / ms.EARTH_RADIUS_KM)
    d = np.arccos(np.clip(ms._xyz(*np.asarray(grown["vertices_deg"]).T) @ dense.T, -1, 1)).min(axis=1)
    assert np.all(np.abs(d * ms.EARTH_RADIUS_KM - 40.0) < 1.5)
    # The notch is gone: no grown vertex sits inside the U's bounding box.
    assert not np.any((np.abs(gx) < 140) & (np.abs(gy) < 90))


def test_shape_rows_and_geojson_read_alike(tmp_path):
    ring = [[36.5, -103.0], [37.0, -94.6], [33.6, -94.4], [34.0, -100.0]]
    row = tmp_path / "row.json"
    row.write_text(json.dumps({"kind": "polygon", "vertices_deg": ring}))
    geo = tmp_path / "shape.geojson"
    geo.write_text(json.dumps({"type": "Feature", "geometry": {
        "type": "Polygon", "coordinates": [[[b, a] for a, b in ring + ring[:1]]]}}))
    assert ms.load_shape(row) == ms.load_shape(geo)


def test_the_fine_region_is_the_drawn_outline_grown_by_one_ramp():
    drawn = ms.parse_ellipse("35.6,-97.5,339,170,60")
    rungs = mesh_point.ladder_rungs(120.0, 3750.0)
    spec = ms.shape_ladder_spec(drawn, rungs_km=rungs, background_km=120.0,
                                transition_factor=18.0, ring_factor=3.0, name="t")
    assert [r["spacing_km"] for r in spec["regions"]] == rungs
    fine = spec["regions"][-1]
    assert fine["transition_km"] == 18.0 * 3.75
    assert ms.area_km2(fine["shape"]) > ms.area_km2(drawn)
    # Each coarser rung contains the next finer one.
    areas = [ms.area_km2(r["shape"]) for r in spec["regions"]]
    assert areas == sorted(areas, reverse=True)


def test_a_request_with_no_core_is_refused_by_name():
    args = type("A", (), {"shape": None, "ellipse": None, "point": None})()
    with pytest.raises(mesh_point.PointPlanRefusal):
        mesh_point.request_from_arguments(args)


def test_a_continent_grown_past_its_raster_keeps_every_side():
    # The CONUS outline `woof hex country --polygon` outlined, grown 1,188 km (its 24 km rung at 96 -> 3 km). The band
    # reached past the old reach + 4 cells raster (the projection stretches it), marching squares traced open
    # chains, and the largest was closed by a chord: the rung lost its whole western side (west edge -132.2 deg,
    # 29 vertices) and rw_mpas_mesh refused the ladder at 18.4 % per cell. Every grown vertex is the growth
    # distance from the outline, and so is every point along its edges (a chord is not), to the raster and tolerance.
    conus = {"kind": "polygon", "vertices_deg": [[49.5, -125.0], [49.5, -95.0], [49.0, -83.0], [47.5, -67.0],
                                                 [43.5, -66.5], [35.0, -74.5], [30.5, -80.0], [24.5, -80.0],
                                                 [24.5, -82.5], [29.0, -89.0], [25.5, -97.0], [29.0, -104.5],
                                                 [31.3, -111.0], [32.5, -117.5], [40.0, -125.0]]}
    grown = ms.grow(conus, 1188.0, tolerance_km=12.0)
    dense = ms._densify(ms._xyz(*np.asarray(conus["vertices_deg"]).T), 2.0 / ms.EARTH_RADIUS_KM)
    edge = ms._densify(ms._xyz(*np.asarray(grown["vertices_deg"]).T), 20.0 / ms.EARTH_RADIUS_KM)
    d = np.arccos(np.clip(edge @ dense.T, -1, 1)).min(axis=1) * ms.EARTH_RADIUS_KM
    assert np.all(np.abs(d - 1188.0) < 30.0), (d.min(), d.max())
    # And the west coast is 1,188 km inside it: the old chord ran from (59.3, -132.2) to (14.0, -84.4).
    assert np.asarray(grown["vertices_deg"])[:, 1].min() < -139.0
