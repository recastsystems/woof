"""Outlines and zones a row must refuse, each in plain words, under both geometries.

Each test names the breakage its refusal prevents.  Found by mesh-check (2026-11-18): a polygon
whose sides cross was accepted by every door; and since then, a zone that names a holder it lies
wholly outside, and a zone finer than its holder by a ratio whose ramp is longer than the holder is
wide.  The rows that have built and stepped the model (the mesh doors' and the stepped one) still expand.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from woof.hex import mesh_ask, mesh_shape  # noqa: E402
from woof.hex.mesh_ask import MeshAskRefusal  # noqa: E402

BOWTIE = {"kind": "polygon", "vertices_deg": [[36.0, -100.0], [40.0, -94.0], [36.0, -94.0], [40.0, -100.0]]}
SQUARE = {"kind": "polygon", "vertices_deg": [[36.0, -100.0], [36.0, -94.0], [40.0, -94.0], [40.0, -100.0]]}


def row(zones, geometry=None, background_km=96):
    out = {"kind": mesh_ask.ROW_KIND, "background_km": background_km, "zones": zones}
    if geometry is not None:
        out["generator"] = {"geometry": geometry}
    return out


def cap(lat, lon, radius_km):
    return {"kind": "cap", "center_deg": [lat, lon], "radius_km": radius_km}


@pytest.mark.parametrize("geometry", [None, 2])
def test_a_polygon_that_crosses_itself_is_refused_in_plain_words(geometry) -> None:
    with pytest.raises(MeshAskRefusal) as refused:
        mesh_ask.expand(row([{"name": "fine", "shape": BOWTIE, "fine_dx_m": 3000}], geometry))
    words = str(refused.value)
    assert "crosses itself" in words and "no single inside" in words and "corner 1" in words, words
    # The same outline as a drawn region (--shape), the site's path.
    with pytest.raises(mesh_shape.ShapeRefusal, match="crosses itself"):
        mesh_shape.shape_from_document(BOWTIE)
    # And as GeoJSON ([lon, lat]).
    ring = [[lon, lat] for lat, lon in BOWTIE["vertices_deg"]]
    with pytest.raises(mesh_shape.ShapeRefusal, match="crosses itself"):
        mesh_shape.shape_from_document({"type": "Polygon", "coordinates": [ring + [ring[0]]]})


def test_the_same_corners_in_order_are_taken() -> None:
    assert mesh_shape.shape_from_document(SQUARE)["kind"] == "polygon"
    assert mesh_ask.expand(row([{"name": "fine", "shape": SQUARE, "fine_dx_m": 3000}], 2))["spec"]["regions"]


def test_a_side_that_runs_back_along_another_is_refused() -> None:
    folded = {"kind": "polygon", "vertices_deg": [[36.0, -100.0], [36.0, -94.0], [40.0, -94.0], [36.0, -94.0],
                                                  [36.0, -97.0], [40.0, -100.0]]}
    with pytest.raises(mesh_shape.ShapeRefusal, match="crosses itself"):
        mesh_shape.shape_from_document(folded)


@pytest.mark.parametrize("vertices, words", [
    ([[36.0, -100.0], [40.0, -94.0]], "encloses no area"),
    ([[36.0, -100.0], [36.0, -100.0], [40.0, -94.0]], "2 distinct corner"),
    ([[36.0, -100.0], [38.0, -97.0], [36.0, -100.0], [38.0, -97.0]], "crosses itself"),
])
def test_an_outline_of_fewer_than_three_corners_is_refused(vertices, words) -> None:
    with pytest.raises(MeshAskRefusal, match=words):
        mesh_ask.expand(row([{"name": "fine", "shape": {"kind": "polygon", "vertices_deg": vertices},
                              "fine_dx_m": 3000}], 2))


def test_corners_on_one_line_are_refused() -> None:
    # Along one meridian: one great circle.
    line = {"kind": "polygon", "vertices_deg": [[30.0, -100.0], [35.0, -100.0], [40.0, -100.0]]}
    with pytest.raises(mesh_shape.ShapeRefusal, match="lie on one line"):
        mesh_shape.shape_from_document(line)


@pytest.mark.parametrize("geometry", [None, 2])
def test_a_zone_wholly_outside_the_zone_it_names_is_refused(geometry) -> None:
    zones = [{"name": "zone", "shape": cap(38.0, -92.0, 150.0), "fine_dx_m": 12000},
             {"name": "core", "inside": "zone", "shape": cap(45.0, -80.0, 50.0), "fine_dx_m": 3000}]
    with pytest.raises(MeshAskRefusal) as refused:
        mesh_ask.expand(row(zones, geometry))
    words = str(refused.value)
    assert "lies wholly outside it" in words and "beyond 'zone'" in words, words


def test_a_zone_partly_inside_the_zone_it_names_is_taken() -> None:
    zones = [{"name": "zone", "shape": cap(38.0, -92.0, 300.0), "fine_dx_m": 12000},
             {"name": "core", "inside": "zone", "shape": cap(38.0, -89.0, 120.0), "fine_dx_m": 6000}]
    assert mesh_ask.expand(row(zones))["spec"]["regions"]


def test_a_ratio_the_ramp_cannot_reach_inside_the_holder_is_refused_under_geometry_2() -> None:
    # 1 km inside 3 km: the ramp needs 54 km beyond the core's line to reach 3 km; the holder reaches
    # no farther than 30 km from it, so no cell of the holder would be at 3 km.
    zones = [{"name": "zone", "shape": cap(38.0, -92.0, 40.0), "fine_dx_m": 3000},
             {"name": "core", "inside": "zone", "shape": cap(38.0, -92.0, 10.0), "fine_dx_m": 1000}]
    with pytest.raises(MeshAskRefusal) as refused:
        mesh_ask.expand(row(zones, 2, background_km=192))
    words = str(refused.value)
    assert "needs 54 km beyond 'core'" in words and "would be built nowhere" in words, words
    assert "at least 54 km" in words, words


def test_mesh_steps_nest_still_builds_and_says_where_the_ramp_crosses() -> None:
    # The row that was stepped (1 km circle of 20 km inside a 3 km circle of 80 km, off centre).
    zones = [{"name": "zone", "fine_dx_m": 3000, "shape": cap(38.0, -92.0, 80.0)},
             {"name": "core", "fine_dx_m": 1000, "inside": "zone", "shape": cap(38.1, -91.8, 20.0)}]
    asked = mesh_ask.expand(row(zones, 2, background_km=192))
    assert len(asked["spec"]["regions"]) == 2
    assert any("still finer than 3 km" in note for note in asked["notes"]), asked["notes"]


def test_a_4_km_zone_beside_a_3_km_zone_expands_under_geometry_1() -> None:
    # It stopped on a TypeError in the note that says what geometry 1 moved (4 km has no halving
    # of 3 km below it, so the other side is the core's own spacing).
    zones = [{"name": "missouri", "shape": cap(38.0, -92.0, 70.0), "fine_dx_m": 3000},
             {"name": "iowa", "shape": {"kind": "lat_lon_box", "lat_deg": [41.3, 42.7], "lon_deg": [-94.6, -92.6]},
              "fine_dx_m": 4000}]
    asked = mesh_ask.expand(row(zones, background_km=192), geometry=1)
    assert any("zone 'iowa': asked 4000 m, built 6000 m" in note for note in asked["notes"]), asked["notes"]
