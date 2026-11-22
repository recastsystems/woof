"""Geometry 2 through the rows and the doors: every row kind, every door word, every refusal.

Geometry 2 is the generator's second way of grading (``rw_mpas_mesh --geometry 2``): the fine spacing
holds to the zone's own line and grows by one rate per cell from there.  A row under it is built as
written, so what these tests hold is that nothing is moved onto a halving ladder, that each shape,
several zones, a nest and a region cut expand to one region a zone, that every door carries the
word to the generator and to its dry run, and that a request that names no geometry is the request
it always was.

Generator-free where the question is the row's; the tests at the end run the real generator's dry
run when one resolves (``WOOF_HEX_RW_MPAS_MESH``), because the count a plan prints is the
generator's own.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from woof.hex import mesh_ask, mesh_point  # noqa: E402
from woof.hex.cli import build_parser  # noqa: E402
from woof.hex.errors import MpasPortError  # noqa: E402
from woof.hex.mesh_ask import MeshAskRefusal  # noqa: E402


@pytest.fixture(autouse=True)
def _ladder_is_the_default(monkeypatch):
    """These tests are written for a request that names no geometry building the halving ladder (geometry 1).

    Since 0.6.2 such a request builds under geometry 2 (``mesh_ask.DEFAULT_GEOMETRY``); the ladder is
    ``--geometry 1``.  The ladder's own rules are unchanged, so these tests hold the old default and keep
    checking them; ``test_mesh_default_geometry.py`` checks the new default.
    """

    monkeypatch.setattr(mesh_ask, "DEFAULT_GEOMETRY", 1)

CORE = {"kind": "cap", "center_deg": [38.0, -92.0], "radius_km": 300.0}
G2 = {"geometry": 2}
WORDS = ["--geometry", "2", "--rings", "keep"]


def row(**words):
    base = {"kind": mesh_ask.ROW_KIND, "background_km": 96.0,
            "zones": [{"name": "core", "shape": dict(CORE), "fine_dx_m": 3000.0}]}
    base.update(words)
    return base


def g2(**words):
    return row(generator=dict(G2), **words)


# ---------------------------------------------------------------------------
# no word, no change
# ---------------------------------------------------------------------------
def test_a_row_that_names_no_geometry_is_the_row_it_always_was() -> None:
    plain = mesh_ask.expand(row())
    assert "generator" not in plain
    assert mesh_ask.expand(row(generator={"geometry": 1})) == plain
    assert mesh_ask.expand(row(), geometry=1) == plain
    # The registered ladder, to the digit, with and without geometry 1 said aloud.
    legacy = mesh_point.ladder_spec((38.0, -92.0), fine_dx_m=3000.0, radius_km=1000.0, background_km=96.0, name="x")
    as_row = mesh_ask.legacy_point_row((38.0, -92.0), fine_dx_m=3000.0, radius_km=1000.0, background_km=96.0, name="x")
    assert json.dumps(mesh_ask.expand(as_row, geometry=1)["spec"], sort_keys=True) == json.dumps(legacy, sort_keys=True)
    assert mesh_ask.generator_request(row())["words"] == []


def test_the_door_writes_the_legacy_spec_when_geometry_is_not_named_or_is_1() -> None:
    for extra in ((), ("--geometry", "1")):
        arguments = build_parser().parse_args(
            ["mesh-plan", "--point=38,-92", "--fine-dx-m", "3000", "--radius-km", "1000", "--background-km", "96",
             *extra])
        request = mesh_point.request_from_arguments(arguments)
        assert request.ask is None and request.generator_words == ()
        assert mesh_point.request_spec(request) == mesh_point.ladder_spec(
            (38.0, -92.0), fine_dx_m=3000.0, radius_km=1000.0, background_km=96.0)


# ---------------------------------------------------------------------------
# any spacing is the spacing built
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(("fine_m", "background", "ladder_would_build"), [
    (3000.0, 96.0, None), (3000.0, 24.0, None), (2000.0, 48.0, 64.0), (3000.0, 15.0, 12.0),
    (2500.0, 96.0, 80.0), (4000.0, 100.0, 128.0), (3000.0, 60.0, 48.0), (1000.0, 96.0, 128.0),
    (750.0, 100.0, 96.0), (5000.0, 37.0, 40.0),
])
def test_geometry_2_builds_the_spacing_asked_on_the_background_asked(fine_m, background, ladder_would_build) -> None:
    asked = mesh_ask.expand(g2(background_km=background,
                               zones=[{"name": "core", "shape": dict(CORE), "fine_dx_m": fine_m}]))
    spec = asked["spec"]
    assert spec["background_km"] == background
    assert [(r["spacing_km"], r["transition_km"]) for r in spec["regions"]] == [(fine_m / 1000.0, 18.0 * fine_m / 1000.0)]
    assert spec["regions"][0]["shape"] == CORE            # held to the line asked: no ring, no grown outline
    assert asked["background"] == {"asked_km": background, "built_km": background, "halvings": None, "moved": False,
                                   "finer_km": background, "coarser_km": background}
    assert asked["generator"] == {"geometry": 2, "ramp_onset": None, "words": WORDS}
    if ladder_would_build is None:
        assert asked["notes"] == []
    else:
        # The row's fit words: nothing to move, and the plan says what the ladder would have built.
        assert len(asked["notes"]) == 1
        assert f"built at the {background:g} km asked" in asked["notes"][0]
        assert f"geometry 1 would build {ladder_would_build:g} km" in asked["notes"][0]
        assert "fit has nothing to move" in asked["notes"][0]


@pytest.mark.parametrize("fit", ["nearest", "finer", "coarser", "exact"])
def test_fit_moves_nothing_under_geometry_2_and_exact_no_longer_refuses(fit) -> None:
    off_ladder = dict(background_km=48.0, fit=fit, zones=[{"name": "core", "shape": dict(CORE), "fine_dx_m": 2000.0}])
    assert mesh_ask.expand(g2(**off_ladder))["spec"]["background_km"] == 48.0
    if fit == "exact":
        with pytest.raises(MeshAskRefusal, match="does not divide 48 km by halving"):
            mesh_ask.expand(row(**off_ladder))


def test_the_doors_word_wins_over_the_rows() -> None:
    assert mesh_ask.expand(g2(), geometry=1) == mesh_ask.expand(row())
    assert mesh_ask.expand(row(), geometry=2) == mesh_ask.expand(g2())


# ---------------------------------------------------------------------------
# each shape
# ---------------------------------------------------------------------------
SHAPES = [
    {"kind": "cap", "center_deg": [38.0, -92.0], "radius_km": 400.0},
    {"kind": "ellipse", "center_deg": [38.0, -96.0], "semi_major_km": 700.0, "semi_minor_km": 400.0, "azimuth_deg": 90.0},
    {"kind": "rounded_box", "center_deg": [34.5, -98.0], "width_km": 800.0, "height_km": 600.0, "corner_km": 150.0,
     "azimuth_deg": 20.0},
    {"kind": "lat_lon_box", "lat_deg": [36.0, 44.0], "lon_deg": [-104.0, -94.0]},
    {"kind": "polygon", "vertices_deg": [[44.2, -9.8], [44.0, -1.5], [42.8, 3.6], [39.5, 1.0], [36.4, -1.6],
                                         [35.7, -5.6], [36.6, -9.6], [39.5, -10.2]]},
    {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[
        [-88.0, 31.2], [-80.8, 31.2], [-79.4, 27.0], [-79.8, 24.6], [-82.4, 24.2], [-84.2, 27.4], [-88.0, 29.6],
        [-88.0, 31.2]]]}},
]


@pytest.mark.parametrize("shape", SHAPES, ids=["circle", "ellipse", "rounded_box", "lat_lon_box", "polygon", "geojson"])
def test_each_shape_is_one_region_that_holds_its_outline(shape) -> None:
    import numpy as np

    asked = mesh_ask.expand(g2(background_km=60.0, zones=[{"name": "z", "shape": shape, "fine_dx_m": 3000.0}]))
    regions = asked["spec"]["regions"]
    assert len(regions) == 1 and regions[0]["spacing_km"] == 3.0          # no rings under geometry 2
    built = regions[0]["shape"]
    outline = asked["zones"][0]["shape"]
    if outline["kind"] == "cap":
        assert built == outline
    else:
        # At most the generator's corner budget, and the region covers every corner of the outline asked.
        assert len(built["vertices_deg"]) <= mesh_ask.RING_VERTICES
        corners = np.asarray(outline["vertices_deg"], dtype=float)
        assert float(np.max(mesh_ask.signed_distance_km(built, corners[:, 0], corners[:, 1]))) <= 0.5
        if len(outline["vertices_deg"]) <= mesh_ask.RING_VERTICES:
            assert built == outline                                      # a drawn outline goes in as drawn
    # The field: 3 km on the whole outline (the median inside it is 3 km to the metre, not a ramp's).
    counted = mesh_ask.census(asked, samples=300_000)
    zone = counted["zones"][0]
    assert zone["median_spacing_km"] == pytest.approx(3.0, abs=1e-9)
    assert zone["columns"] == pytest.approx(asked["zones"][0]["area_km2"] / (mesh_ask.HEXAGON_AREA_FACTOR * 9.0), rel=0.03)
    assert counted["spacing_km_by_column"]["max"] == pytest.approx(60.0)


def test_a_dense_outline_is_redrawn_within_the_corner_budget() -> None:
    ellipse = SHAPES[1]
    asked = mesh_ask.expand(g2(zones=[{"name": "z", "shape": ellipse, "fine_dx_m": 3000.0}]))
    assert len(asked["zones"][0]["shape"]["vertices_deg"]) > mesh_ask.RING_VERTICES
    assert len(asked["spec"]["regions"][0]["shape"]["vertices_deg"]) <= mesh_ask.RING_VERTICES


def test_a_continent_sized_box_keeps_its_outline_when_redrawn() -> None:
    """A 5,400 by 3,250 km rounded box (HRRR's footprint) grown half a 3 km cell.

    THE BREAKAGE THIS HOLDS: the outline raster's cell there (4.5 km) is wider than the 1.5 km the
    outline is grown by; the grown band enclosed nothing and the region came out as a polygon of TWO
    corners, which rw_mpas_mesh refused ("fewer than three enclose no area to refine").
    """

    import numpy as np

    from woof.hex import mesh_shape

    box = {"kind": "rounded_box", "center_deg": [38.5, -97.5], "width_km": 5400.0, "height_km": 3250.0,
           "corner_km": 300.0, "azimuth_deg": 0}
    asked = mesh_ask.expand(row(background_km=48, zones=[{"shape": box, "fine_dx_m": 3000.0}], generator=G2))
    (region,) = asked["spec"]["regions"]
    corners = np.asarray(region["shape"]["vertices_deg"], dtype=float)
    assert 8 <= len(corners) <= mesh_ask.RING_VERTICES
    x, y = mesh_shape._Azimuthal(38.5, -97.5).forward(corners[:, 0], corners[:, 1])
    assert 2690.0 < float(x.max()) < 2712.0 and -2712.0 < float(x.min()) < -2690.0
    assert 1615.0 < float(y.max()) < 1637.0 and -1637.0 < float(y.min()) < -1615.0
    area = 0.5 * abs(float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))
    drawn = 5400.0 * 3250.0 - (4.0 - math.pi) * 300.0 ** 2
    assert abs(area / drawn - 1.0) < 0.01


def test_a_continent_sized_box_keeps_its_long_sides_under_the_generators_great_circles() -> None:
    """rw_mpas_mesh joins a polygon's corners by great circles.

    THE BREAKAGE THIS HOLDS: the corners of the redrawn box were chosen on the azimuthal plane, where a
    4,800 km side is one straight line, so the side kept only its two ends; the great circle between
    them runs up to 145 km north of the box's north side, and the 3 km box was built with a dome
    nobody asked for.
    """

    import numpy as np

    from woof.hex import mesh_shape

    box = {"kind": "rounded_box", "center_deg": [38.5, -97.5], "width_km": 5400.0, "height_km": 3250.0,
           "corner_km": 300.0, "azimuth_deg": 0}
    asked = mesh_ask.expand(row(background_km=48, zones=[{"shape": box, "fine_dx_m": 3000.0}], generator=G2))
    corners = np.asarray(asked["spec"]["regions"][0]["shape"]["vertices_deg"], dtype=float)
    proj = mesh_shape._Azimuthal(38.5, -97.5)
    xyz = mesh_shape._xyz(corners[:, 0], corners[:, 1])
    mid = xyz + np.roll(xyz, -1, axis=0)
    mid /= np.linalg.norm(mid, axis=1)[:, None]
    mlat, mlon = mesh_shape._latlon(mid)
    x, y = proj.forward(np.asarray(mlat, dtype=float), np.asarray(mlon, dtype=float))
    # every mid-point of a built side lies within 8 km of the box drawn on the plane (half sides 2701.5 by 1626.5 km)
    assert float(np.abs(x).max()) < 2701.5 + 8.0 and float(np.abs(y).max()) < 1626.5 + 8.0
    top = (y > 1500.0) & (np.abs(x) < 2300.0)
    assert int(top.sum()) >= 6 and float(y[top].min()) > 1626.5 - 8.0


def test_the_inside_of_an_outline_is_counted_by_the_outline_itself() -> None:
    import numpy as np

    from woof.hex import mesh_shape

    square = {"kind": "polygon", "vertices_deg": [[37.0, -93.0], [37.0, -91.0], [39.0, -91.0], [39.0, -93.0]]}
    proj = mesh_shape._Azimuthal(*mesh_shape._check_size(square))
    xs = np.arange(-200.0, 200.1, 10.0)
    ys = np.arange(-200.0, 200.1, 10.0)
    inside = mesh_shape._inside_outline(square, proj, xs, ys, 10.0)
    gx, gy = np.meshgrid(xs, ys)
    lat, lon = proj.inverse(gx.ravel(), gy.ravel())
    lat, lon = np.asarray(lat).reshape(gx.shape), np.asarray(lon).reshape(gx.shape)
    clear_in = (lat > 37.1) & (lat < 38.9) & (lon > -92.9) & (lon < -91.1)
    clear_out = (lat < 36.9) | (lat > 39.1) | (lon < -93.1) | (lon > -90.9)
    assert inside[clear_in].all() and not inside[clear_out].any()


# ---------------------------------------------------------------------------
# several zones, a zone inside a zone
# ---------------------------------------------------------------------------
def test_several_zones_are_one_region_each_at_their_own_spacing() -> None:
    asked = mesh_ask.expand(g2(zones=[
        {"name": "missouri", "shape": dict(CORE), "fine_dx_m": 3000.0},
        {"name": "florida", "shape": {"kind": "ellipse", "center_deg": [28.0, -82.0], "semi_major_km": 350.0,
                                      "semi_minor_km": 200.0, "azimuth_deg": 160.0}, "fine_dx_m": 3000.0},
        {"name": "alps", "shape": {"kind": "rounded_box", "center_deg": [46.5, 10.5], "width_km": 900.0,
                                   "height_km": 500.0, "corner_km": 100.0}, "fine_dx_m": 5000.0}]))
    assert sorted(r["spacing_km"] for r in asked["spec"]["regions"]) == [3.0, 3.0, 5.0]
    assert [zone["inside"] for zone in asked["zones"]] == [None, None, None]
    # 5 km is not a halving of 3 km: built as asked, and said.
    assert any("zone 'alps': built at the 5000 m asked (geometry 1 would build 6000 m" in note for note in asked["notes"])


def test_a_true_three_to_one_nest_is_built_at_both_spacings_asked() -> None:
    nest = dict(zones=[
        {"name": "zone", "shape": {"kind": "cap", "center_deg": [38.0, -92.0], "radius_km": 600.0}, "fine_dx_m": 3000.0},
        {"name": "core", "shape": {"kind": "cap", "center_deg": [38.4, -91.2], "radius_km": 150.0}, "fine_dx_m": 1000.0,
         "inside": "zone"}])
    asked = mesh_ask.expand(g2(**nest))
    assert [(r["spacing_km"], r["shape"]["radius_km"]) for r in asked["spec"]["regions"]] == [(3.0, 600.0), (1.0, 150.0)]
    core = next(zone for zone in asked["zones"] if zone["name"] == "core")
    assert core["inside"] == "zone" and core["fine_m"] == 1000.0
    # The core's ramp climbs to its holder's 3 km, 1.5 x 18 km x (3 - 1) outside the core's line.
    assert core["ramp"]["to_km"] == 3.0 and core["ramp"]["reaches_it_km_beyond_outline"] == pytest.approx(54.0)
    # The halving generator cannot hold 1 and 3 km together: it builds the zone at 2 km.
    ladder = mesh_ask.expand(row(**nest))
    assert next(zone for zone in ladder["zones"] if zone["name"] == "zone")["fine_m"] == 2000.0
    with pytest.raises(MeshAskRefusal, match="a nested zone is finer than its holder"):
        mesh_ask.expand(g2(zones=[nest["zones"][0], {**nest["zones"][1], "fine_dx_m": 3000.0}]))


# ---------------------------------------------------------------------------
# the ramp
# ---------------------------------------------------------------------------
def test_the_ramp_is_one_growth_rate_per_cell() -> None:
    default = mesh_ask.ramp_numbers(None)
    assert default["ramp_scale_cells"] == 18.0 and default["growth_per_cell"] == pytest.approx(1.0 / 27.0)
    assert mesh_ask.ramp_numbers({"kind": "ladder", "transition_factor": 18.0, "ring_factor": 3.0}) == default
    asked = mesh_ask.ramp_numbers({"kind": "ramp", "growth_per_cell": 0.05})
    assert asked["ramp_scale_cells"] == pytest.approx(1.0 / (1.5 * 0.05)) and asked["growth_per_cell"] == 0.05
    doubling = mesh_ask.ramp_numbers({"kind": "ramp", "ramp_cells": 19.0})
    assert doubling["growth_per_cell"] == pytest.approx(2.0 ** (1.0 / 19.0) - 1.0)
    assert doubling["cells_per_doubling"] == pytest.approx(19.0)
    # 3 km reaches 96 km 1.5 x 54 km x 31 outside the line.
    zone = mesh_ask.expand(g2())["zones"][0]
    assert zone["ramp"] == {"growth_per_cell": pytest.approx(1.0 / 27.0), "to_km": 96.0,
                            "reaches_it_km_beyond_outline": pytest.approx(2511.0)}
    steeper = mesh_ask.expand(g2(grading={"kind": "ramp", "growth_per_cell": 0.074}))
    assert steeper["spec"]["regions"][0]["transition_km"] == pytest.approx(3.0 / (1.5 * 0.074))
    assert (mesh_ask.census(steeper, samples=200_000)["columns"]
            < mesh_ask.census(mesh_ask.expand(g2()), samples=200_000)["columns"])


@pytest.mark.parametrize(("grading", "words"), [
    ({"kind": "rings", "ring_cells": 10, "ramp_cells": 40}, "geometry 2 has no flat rings to lay"),
    ({"kind": "ramp", "ring_cells": 4}, "geometry 2 has no flat rings to lay"),
    ({"kind": "ramp", "ring_factor": 3.0}, "geometry 2 has no flat rings to lay"),
    ({"kind": "ramp", "growth_per_cell": 0.3}, "not between 0 and 0.25"),
    ({"kind": "ramp", "growth_per_cell": 0.0}, "a growth of nothing is no ramp"),
    ({"kind": "ramp", "ramp_cells": 2}, "the steepest ramp is 25 % a cell"),
    ({"kind": "ramp", "ramp_cells": 0}, "a ramp of no cells is a jump"),
    ({"kind": "ladder", "transition_factor": 2.0}, "the steepest is 25 %"),
    ({"kind": "ramp", "growth_per_cell": 0.05, "ramp_cells": 20}, "names the ramp twice"),
    ({"kind": "spiral"}, "is not ladder, rings or ramp"),
    ({"kind": "ramp", "speed": 3}, "words this door does not know: speed"),
])
def test_grading_refusals_under_geometry_2(grading, words) -> None:
    with pytest.raises(MeshAskRefusal, match=words):
        mesh_ask.expand(g2(grading=grading))


# ---------------------------------------------------------------------------
# --ramp-onset
# ---------------------------------------------------------------------------
def test_ramp_onset_goes_to_the_generator_and_costs_columns() -> None:
    bend = mesh_ask.expand(row(generator={"geometry": 2, "ramp_onset": 1}))
    assert bend["generator"]["words"] == ["--geometry", "2", "--ramp-onset", "1", "--rings", "keep"]
    assert bend["spec"] == mesh_ask.expand(g2())["spec"]                  # the spec is the same; the word is the ramp's
    # The bend starts the ramp half an onset later: 27 km at 3 km and one width.
    corner = mesh_ask.expand(g2())
    assert (bend["zones"][0]["ramp"]["reaches_it_km_beyond_outline"]
            == pytest.approx(corner["zones"][0]["ramp"]["reaches_it_km_beyond_outline"] + 27.0))
    assert mesh_ask.census(bend, samples=300_000)["columns"] > mesh_ask.census(corner, samples=300_000)["columns"]
    assert mesh_ask.expand(g2(), ramp_onset=1.0)["generator"]["words"] == bend["generator"]["words"]
    assert mesh_ask.expand(g2(), ramp_onset=0.0)["generator"]["words"] == WORDS


@pytest.mark.parametrize(("generator", "words"), [
    ({"geometry": 3}, "generator geometry 3 is not 1 or 2"),
    ({"geometry": "two"}, "is not 1 or 2"),
    ({"geometry": 2.5}, "is not 1 or 2"),
    ({"geometry": 1, "ramp_onset": 1}, "ramp_onset belongs to geometry 2"),
    ({"ramp_onset": 1}, "ramp_onset belongs to geometry 2"),
    ({"geometry": 2, "ramp_onset": 7}, "ramp_onset 7 is not between 0 and 6"),
    ({"geometry": 2, "ramp_onset": -1}, "is not between 0 and 6"),
    ({"geometry": 2, "ramp_onset": "wide"}, "ramp_onset is a number of transition widths"),
    ({"geometry": 2, "finish": "none"}, "generator has words this door does not know: finish"),
    ("2", "the row's generator is an object"),
])
def test_generator_refusals(generator, words) -> None:
    with pytest.raises(MeshAskRefusal, match=words):
        mesh_ask.expand(row(generator=generator))


# ---------------------------------------------------------------------------
# a region and its boundary rings
# ---------------------------------------------------------------------------
def test_a_region_holds_its_rim_spacing_from_the_ramp_to_the_cut() -> None:
    asked = mesh_ask.expand(g2(domain={"kind": "region", "rim_km": 300.0, "rim_spacing_km": 12.0}))
    cut = asked["cut"]
    assert cut["region"] == {"kind": "cap", "center_deg": [38.0, -92.0], "radius_km": 600.0}
    assert cut["rim_spacing_km"] == 12.0 and cut["boundary_rings"] == 7
    regions = asked["spec"]["regions"]
    assert [(r["spacing_km"], r["shape"]["radius_km"]) for r in regions] == [
        (12.0, 300.0 + 300.0 + (7 + mesh_ask.RIM_MARGIN_CELLS) * 12.0), (3.0, 300.0)]
    assert any("12 km cells are held from where the ramp reaches them (243 km outside the outline)" in note
               for note in asked["notes"])
    # Every region is a zone someone asked for: the generator is told to keep them all.
    assert asked["generator"]["words"][-2:] == ["--rings", "keep"]
    # The field at the cut and across the boundary rings is the rim's, not the ramp's 14 to 19 km.
    counted = mesh_ask.census(asked, samples=1_000_000)
    assert counted["spacing_at_cut_km"] == pytest.approx(12.0, rel=0.01)
    assert counted["columns_in_boundary_rings"] > 0


def test_a_region_without_a_rim_spacing_is_cut_on_the_ramp_and_adds_no_region() -> None:
    asked = mesh_ask.expand(g2(domain={"kind": "region", "rim_km": 200.0}))
    assert len(asked["spec"]["regions"]) == 1 and asked["cut"]["rim_spacing_km"] is None
    # 200 km out the ramp has reached 3 x (1 + 200 / 81) km.
    assert mesh_ask.census(asked, samples=1_000_000)["spacing_at_cut_km"] == pytest.approx(3.0 * (1 + 200 / 81), rel=0.06)


def test_a_drawn_region_is_cut_along_its_grown_outline_under_geometry_2() -> None:
    asked = mesh_ask.expand(g2(zones=[{"name": "iowa", "shape": {"kind": "ellipse", "center_deg": [42.0, -93.5],
                                                               "semi_major_km": 250.0, "semi_minor_km": 150.0,
                                                               "azimuth_deg": 70.0}, "fine_dx_m": 3000.0}],
                               domain={"kind": "region", "rim_km": 450.0, "rim_spacing_km": 15.0}))
    assert asked["cut"]["region"]["kind"] == "polygon" and asked["cut"]["around"] == "iowa"
    assert sorted(r["spacing_km"] for r in asked["spec"]["regions"]) == [3.0, 15.0]


@pytest.mark.parametrize(("domain", "words"), [
    ({"kind": "region", "rim_km": 200.0, "rim_spacing_km": 12.0},
     "the ramp needs 243 km beyond the outline to grow from 3 km to the 12 km rim at 3.7 % a cell, and rim_km is 200"),
    ({"kind": "region", "rim_km": 300.0, "rim_spacing_km": 2.0}, "is finer than the zone it rims"),
    ({"kind": "region", "rim_km": 300.0, "rim_spacing_km": 120.0}, "coarser than the 96 km background"),
    ({"kind": "region", "rim_km": 0.0}, "is not positive"),
    ({"kind": "region"}, "needs rim_km"),
    ({"kind": "region", "rim_km": 300.0, "boundary_rings": 5}, "boundary_rings 5 is not 7"),
    ({"kind": "region", "rim_km": 300.0, "around": "nowhere"}, "no zone has that name"),
    ({"kind": "region", "rim_km": 300.0, "halo": 2}, "words this door does not know: halo"),
    ({"kind": "globe", "rim_km": 300.0}, "a globe has no rim"),
    ({"kind": "torus"}, "is not globe or region"),
])
def test_domain_refusals_under_geometry_2(domain, words) -> None:
    with pytest.raises(MeshAskRefusal, match=words):
        mesh_ask.expand(g2(domain=domain))


def test_a_steeper_ramp_reaches_the_rim_sooner() -> None:
    domain = {"kind": "region", "rim_km": 200.0, "rim_spacing_km": 12.0}
    asked = mesh_ask.expand(g2(domain=domain, grading={"kind": "ramp", "growth_per_cell": 0.06}))
    assert any("(150 km outside the outline)" in note for note in asked["notes"])


# ---------------------------------------------------------------------------
# the plan: columns under geometry 2, the step that will run, memory for 1, 2, 4, 8 cards
# ---------------------------------------------------------------------------
def _fake_dry_run(monkeypatch, seen: list) -> None:
    from woof.hex.swath import sizing

    def dry_run(spec, *, engine=None, card=None, vram_gib=None, generator_words=()):
        seen.append(list(generator_words))
        asked = {"spec": spec, "zones": [], "cut": None,
                 "generator": {"geometry": 2 if "--geometry" in generator_words else 1}}
        return {"predicted_cells": mesh_ask.census(asked, samples=200_000)["columns"], "geometry": 2,
                "gates_applied_by_hexcore": None}

    monkeypatch.setattr(sizing, "dry_run", dry_run)


def test_the_plan_counts_under_geometry_2_and_says_the_step_that_will_run(monkeypatch) -> None:
    seen: list = []
    _fake_dry_run(monkeypatch, seen)
    planned = mesh_ask.plan(g2(), samples=200_000)
    assert seen == [WORDS]                                   # the generator is asked for the count WITH the word
    assert planned["generator"]["geometry"] == 2
    # More columns than the halving generator's field for the same row (the 3 km holds to the line asked).
    ladder_columns = mesh_ask.census(mesh_ask.expand(row(hold_fine_to_outline=False)), samples=200_000)["columns"]
    assert planned["columns"] > ladder_columns
    # The mesh's own step on the edge geometry 2 holds; and the step each suite's forecast runs.
    assert planned["step"]["dt_seconds"] == 18.0
    assert planned["step"]["smallest_edge_assumed_m"] == pytest.approx(0.79 * 3000.0)
    assert planned["forecast_step"]["hrrrv5"]["dt_seconds"] == 20.0          # NOAA's deck step near 3 km
    assert planned["forecast_step"]["hrrrv5"]["margin"] > 1.0
    assert planned["forecast_step"]["convection_permitting"]["dt_seconds"] == 18.0
    for key in ("96gb/hrrrv5", "96gb/convection_permitting", "32gb/hrrrv5", "32gb/convection_permitting"):
        assert sorted(planned["memory"][key]["cards"]) == ["1", "2", "4", "8"]
    text = "\n".join(mesh_ask.report(planned))
    for words in ("generator        geometry 2", "--geometry 2 --rings keep", "ramp of core",
                  "3 km held to the outline, then +3.7 % a cell, reaching 96 km 2,511 km outside it",
                  "time step        18 s admitted", "the floor geometry 2 holds",
                  "step hrrrv5", "the forecast runs 20 s", "memory 96gb/hrrrv5", "1 card", "8 cards"):
        assert words in text, words
    assert "rings of" not in text


def test_a_2_km_mesh_has_no_deck_row_and_runs_its_own_step(monkeypatch) -> None:
    _fake_dry_run(monkeypatch, [])
    planned = mesh_ask.plan(g2(background_km=48.0, zones=[{"name": "core", "shape": dict(CORE), "fine_dx_m": 2000.0}]),
                            samples=200_000, cards=["32gb"])
    assert planned["background"]["built_km"] == 48.0
    assert planned["step"]["dt_seconds"] == 12.0 and planned["forecast_step"]["hrrrv5"]["dt_seconds"] == 12.0
    assert list(planned["memory"]) == ["32gb/hrrrv5", "32gb/convection_permitting"]


def test_a_geometry_1_plan_asks_the_generator_with_no_word(monkeypatch) -> None:
    from woof.hex.swath import sizing

    calls = []

    def dry_run(spec, *, engine=None, card=None, vram_gib=None):          # the signature every older caller knows
        calls.append(spec["background_km"])
        return {"predicted_cells": 1000.0, "gates_applied_by_hexcore": {}}

    monkeypatch.setattr(sizing, "dry_run", dry_run)
    planned = mesh_ask.plan(row(), samples=50_000)
    assert calls == [96.0] and "generator" not in planned
    assert planned["forecast_step"]["hrrrv5"]["dt_seconds"] == 20.0


# ---------------------------------------------------------------------------
# the doors carry the word
# ---------------------------------------------------------------------------
def _parse(*words: str) -> argparse.Namespace:
    return build_parser().parse_args(["mesh-plan", *words])


def test_the_point_door_builds_any_ratio_under_geometry_2() -> None:
    words = ("--point=38,-92", "--fine-dx-m", "3000", "--radius-km", "400", "--background-km", "60")
    with pytest.raises(mesh_point.PointPlanRefusal, match="not a rung of the 60 km halving ladder"):
        mesh_point.request_spec(mesh_point.request_from_arguments(_parse(*words)))
    request = mesh_point.request_from_arguments(_parse(*words, "--geometry", "2"))
    assert request.ask is not None and request.generator_words == tuple(WORDS)
    spec = mesh_point.request_spec(request)
    assert spec["background_km"] == 60.0 and [r["spacing_km"] for r in spec["regions"]] == [3.0]
    # The cut is the pad's circle, as the door has always cut it.
    assert mesh_point.request_cull_region(request)["radius_km"] == pytest.approx(400.0 * 1.35)
    onset = mesh_point.request_from_arguments(_parse(*words, "--geometry", "2", "--ramp-onset", "1"))
    assert onset.generator_words == ("--geometry", "2", "--ramp-onset", "1", "--rings", "keep")


def test_the_point_doors_refusals_under_geometry_2_are_in_plain_words() -> None:
    words = ("--point=38,-92", "--fine-dx-m", "3000", "--radius-km", "400", "--background-km", "96")
    with pytest.raises(mesh_point.PointPlanRefusal, match="geometry 2 has no flat rings to lay"):
        mesh_point.request_from_arguments(_parse(*words, "--geometry", "2", "--ring-cells", "10"))
    with pytest.raises(mesh_point.PointPlanRefusal, match="generator geometry 3 is not 1 or 2"):
        mesh_point.request_from_arguments(_parse(*words, "--geometry", "3"))
    with pytest.raises(mesh_point.PointPlanRefusal, match="ramp_onset belongs to geometry 2"):
        mesh_point.request_from_arguments(_parse(*words, "--ramp-onset", "1"))
    with pytest.raises(mesh_point.PointPlanRefusal, match="is not between 0 and 6"):
        mesh_point.request_from_arguments(_parse(*words, "--geometry", "2", "--ramp-onset", "9"))
    with pytest.raises(mesh_point.PointPlanRefusal, match="the ramp needs 243 km"):
        mesh_point.request_from_arguments(_parse(*words, "--geometry", "2", "--rim-km", "100", "--rim-spacing-km", "12"))


def test_the_point_door_takes_a_drawn_core_and_more_zones_under_geometry_2() -> None:
    request = mesh_point.request_from_arguments(_parse(
        "--ellipse=42,-93.5,250,150,70", "--fine-dx-m", "3000", "--background-km", "96", "--geometry", "2",
        "--rim-km", "450", "--rim-spacing-km", "15",
        "--zone", json.dumps({"shape": {"kind": "cap", "center_deg": [42.0, -93.5], "radius_km": 40.0},
                              "fine_dx_m": 1000.0})))
    asked = request.ask
    assert sorted(r["spacing_km"] for r in asked["spec"]["regions"]) == [1.0, 3.0, 15.0]
    assert asked["zones"][1]["inside"] == "core" and request.fine_dx_m == 1000.0


def test_a_row_file_through_the_point_door(tmp_path) -> None:
    path = tmp_path / "row.json"
    path.write_text(json.dumps(g2(domain={"kind": "region", "rim_km": 300.0, "rim_spacing_km": 12.0})), encoding="utf-8")
    request = mesh_point.request_from_arguments(_parse("--spec", str(path)))
    assert request.generator_words == tuple(WORDS) and len(mesh_point.request_spec(request)["regions"]) == 2
    # The door's word wins over the row's: the same file under geometry 1 is the halving ladder with its rings.
    ladder = mesh_point.request_from_arguments(_parse("--spec", str(path), "--geometry", "1"))
    assert ladder.generator_words == () and len(mesh_point.request_spec(ladder)["regions"]) > 2
    path.write_text(json.dumps(row(domain={"kind": "region", "rim_km": 300.0})), encoding="utf-8")
    assert mesh_point.request_from_arguments(_parse("--spec", str(path), "--geometry", "2")).generator_words == tuple(WORDS)


def test_the_point_plan_prices_the_cut_geometry_2_builds(monkeypatch) -> None:
    seen: list = []
    _fake_dry_run(monkeypatch, seen)
    words = ("--point=35,-97", "--radius-km", "800", "--fine-dx-m", "30000", "--background-km", "120")
    plan = mesh_point.plan_point(mesh_point.request_from_arguments(_parse(*words, "--geometry", "2")))
    assert seen == [WORDS]
    assert plan["generator"]["geometry"] == 2 and plan["ladder_km"] == [30.0]
    # 30 km held on the whole 800 km: about 2,570 columns inside it, where the halving ladder's field has 1,241.
    assert plan["cull"]["core_cells"] == pytest.approx(2570.0, rel=0.03)
    assert plan["step"]["dt_seconds"] == 180.0 and sorted(plan["memory"]["96gb/hrrrv5"]["cards"]) == ["1", "2", "4", "8"]
    text = "\n".join(mesh_point._report(plan))
    for expected in ("generator        geometry 2", "30 km held to the outline, then +3.7 % a cell to the 120 km background",
                     "time step        180 s admitted", "the forecast runs 180 s", "memory 96gb/hrrrv5",
                     "geometry 2 gates itself at the build"):
        assert expected in text, expected
    assert "ladder " not in text and "transition band" not in text


def test_the_globe_door_expands_a_plain_point_as_a_row_under_geometry_2() -> None:
    arguments = _parse("--global-mesh", "--point=38,-96", "--fine-dx-m", "3000", "--radius-km", "400",
                       "--background-km", "96", "--geometry", "2")
    assert mesh_point.arguments_geometry(arguments) == 2
    asked = mesh_point.expand_arguments(arguments, globe=True)
    assert asked["cut"] is None and asked["generator"]["words"] == WORDS
    assert [(r["spacing_km"], r["shape"]["radius_km"]) for r in asked["spec"]["regions"]] == [(3.0, 400.0)]
    # A concentric cap family still: the door turns its premade template to the point.
    from woof.hex import mesh_plan_door

    assert mesh_plan_door.concentric_cap_center(asked["spec"]) == (38.0, -96.0)
    assert mesh_plan_door.door_generator_words(arguments) == ["--geometry", "2"]
    assert mesh_plan_door.door_generator_words(_parse("--global-mesh", "--point=38,-96")) == []


def test_the_regional_door_carries_the_word(tmp_path) -> None:
    from woof.hex import regional_door

    path = tmp_path / "row.json"
    path.write_text(json.dumps(row(domain={"kind": "region", "rim_km": 300.0, "rim_spacing_km": 12.0})), encoding="utf-8")
    arguments = argparse.Namespace(mesh_spec=path, point=None, shape=None, ellipse=None, fine_dx_m=None,
                                   radius_km=None, background_km=None, geometry=2, ramp_onset=None)
    asked = regional_door._mesh_row(arguments)
    assert asked["generator"]["words"] == WORDS and len(asked["spec"]["regions"]) == 2
    # The cut at 600 km and seven 12 km boundary rings, walked at their own spacing with the relaxed rows'
    # allowance (mesh_ask.RING_REACH_SCALE): never less than seven rings of 12 km.
    reach = regional_door._mesh_row_reach(asked, (38.0, -92.0))
    assert 600.0 + 7 * 12.0 <= reach <= 600.0 + mesh_ask.RING_REACH_SCALE * 7 * 12.5
    assert mesh_ask.door_argv(arguments) == ["--geometry", "2"]
    assert mesh_ask.door_argv(argparse.Namespace(geometry=2, ramp_onset=1.0)) == ["--geometry", "2", "--ramp-onset", "1.0"]
    assert mesh_ask.door_argv(argparse.Namespace(mesh_spec=path)) == []
    parsed = build_parser().parse_args(["regional", "--point=38,-92", "--cycle", "2024-05-21T18", "--work-dir", "x",
                                        "--geometry", "2", "--ramp-onset", "1"])
    assert parsed.geometry == 2 and parsed.ramp_onset == 1.0
    # The generator build is the door's to name too (it was refused as an unknown word).
    assert build_parser().parse_args(["regional", "--point=38,-92", "--cycle", "2024-05-21T18", "--work-dir", "x",
                                      "--mesh-exe", "/opt/rw_mpas_mesh"]).mesh_exe == Path("/opt/rw_mpas_mesh")
    wrong = argparse.Namespace(mesh_spec=path, point=None, shape=None, ellipse=None, fine_dx_m=None,
                               radius_km=None, background_km=None, geometry=3, ramp_onset=None)
    with pytest.raises(regional_door.RegionalRefusal, match="is not 1 or 2"):
        regional_door._mesh_row(wrong)


def test_the_country_door_builds_any_ratio_under_geometry_2(capsys) -> None:
    from woof.hex import country_door

    def region(*words):
        arguments = build_parser().parse_args(["country", "--point=38,-92", "--radius-km", "300", "--fine-dx-m",
                                               "2000", "--background-km", "48", "--cycle", "2024-05-21T18",
                                               "--work-dir", "x", *words])
        return country_door.region_request(arguments)[0]

    with pytest.raises(MpasPortError, match="not a rung of the 48 km halving ladder"):
        region()
    built = region("--geometry", "2")
    assert built["mesh"]["fine_dx_m"] == 2000.0 and built["mesh"]["background_km"] == 48.0     # not moved to 64
    assert built["mesh"]["generator"] == {"geometry": 2} and built["mesh"]["dt_seconds"] == 12.0
    spec = country_door.region_specification(built)
    assert spec["background_km"] == 48.0 and [r["spacing_km"] for r in spec["regions"]] == [2.0]
    assert country_door.region_generator_words(built) == WORDS
    # The geometry is part of the region's identity: its prepared mesh is never the halving ladder's.
    ladder = region("--background-fit", "nearest")
    assert ladder["id"] != built["id"] and country_door.region_generator_words(ladder) == []
    assert len(country_door.region_specification(ladder)["regions"]) > 1
    assert region("--geometry", "2", "--ramp-onset", "1")["id"] != built["id"]
    # On the table's own default spacing too.
    default = build_parser().parse_args(["country", "--point=35,-97", "--radius-km", "800", "--cycle", "2024-05-21T18",
                                         "--work-dir", "x", "--geometry", "2"])
    made = country_door.region_request(default)[0]
    assert made["mesh"]["generator"] == {"geometry": 2}
    assert [r["spacing_km"] for r in country_door.region_specification(made)["regions"]] == [made["mesh"]["fine_dx_m"] / 1000.0]


def test_the_country_door_refuses_geometry_2_on_a_registered_row() -> None:
    from woof.hex import country_door

    arguments = build_parser().parse_args(["country", "--row", "anything", "--cycle", "2024-05-21T18", "--work-dir", "x",
                                           "--geometry", "2"])
    with pytest.raises(MpasPortError, match="--geometry 2 builds a region drawn at run time"):
        country_door.region_request(arguments)


def test_the_moving_door_carries_the_word_in_the_case() -> None:
    from woof.hex import moving_author, moving_members
    from woof.hex.moving_case import MovingCaseError
    from woof.hex.moving_members import member_spec

    # No word: the block it always was, and no words for the generator.
    plain = moving_author.mesh_block(3.0, 48.0, 200.0)
    assert "generator" not in plain and [ring["spacing_km"] for ring in plain["rings"]] == [6.0, 12.0, 24.0]
    assert moving_author.mesh_block(3.0, 48.0, 200.0, geometry=1) == plain
    assert moving_members.mesh_words({"mesh": plain}) == []
    with pytest.raises(MovingCaseError, match="power of two"):
        moving_author.mesh_block(3.0, 60.0, 200.0)
    # Geometry 2: any ratio, the fine circle alone, and the words every member and the template are built with.
    block = moving_author.mesh_block(3.0, 60.0, 200.0, geometry=2)
    assert block["rings"] == [] and block["generator"] == {"geometry": 2} and block["background_km"] == 60.0
    case = {"mesh": block}
    assert moving_members.mesh_words(case) == WORDS
    spec = member_spec(case, "start", 25.0, -70.0)
    assert spec["background_km"] == 60.0
    assert [(r["spacing_km"], r["shape"]["radius_km"], r["transition_km"]) for r in spec["regions"]] == [(3.0, 200.0, 54.0)]
    bend = moving_author.mesh_block(3.0, 60.0, 200.0, geometry=2, ramp_onset=1.0)
    assert moving_members.mesh_words({"mesh": bend}) == ["--geometry", "2", "--ramp-onset", "1", "--rings", "keep"]
    with pytest.raises(MovingCaseError, match="is not 1 or 2"):
        moving_author.mesh_block(3.0, 60.0, 200.0, geometry=4)
    with pytest.raises(MovingCaseError, match="geometry 2 is not built for a corridor case"):
        moving_members.mesh_words({"mesh": {**block, "corridor": {"cross_track_km": 450.0}}})
    parsed = build_parser().parse_args(["moving", "template", "--fine-km", "3", "--background-km", "60",
                                        "--out-dir", "x", "--geometry", "2"])
    assert parsed.geometry == 2


def test_the_store_keys_a_geometry_2_mesh_apart(tmp_path) -> None:
    from woof.hex import prep_store

    engine = tmp_path / "rw_mpas_mesh"
    engine.write_bytes(b"#!/bin/sh\n")
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(mesh_ask.expand(g2())["spec"]), encoding="utf-8")
    base = [str(engine), "--spec", str(spec), "--out", str(tmp_path / "g.nc"), "--receipt", str(tmp_path / "r.json")]
    plain = prep_store.mesh_request(base, {})
    said = prep_store.mesh_request([*base, "--geometry", "1"], {})
    ramp = prep_store.mesh_request([*base, *WORDS], {})
    bend = prep_store.mesh_request([*base, "--geometry", "2", "--ramp-onset", "1", "--rings", "keep"], {})
    assert plain is not None and ramp is not None and bend is not None
    assert said[0] == plain[0]                         # geometry 1 said aloud is the request with no word
    assert len({plain[0], ramp[0], bend[0]}) == 3      # a geometry 2 mesh is never fetched for a geometry 1 request
    assert ramp[1]["options"] == {"--geometry": "2", "--rings": "keep"}


# ---------------------------------------------------------------------------
# the real generator's dry run (the count a plan prints is its own)
# ---------------------------------------------------------------------------
def _engine() -> Path | None:
    from woof.hex.engines import MESH, EngineRefusal, resolve

    try:
        engine = resolve(MESH, None)
    except EngineRefusal:
        return None
    # A generator that knows the word refuses a geometry that does not exist by naming the word; an older one
    # (the 0.6.1 wheel's) does not know the flag at all.
    done = subprocess.run([str(engine), "--background-km", "480", "--dry-run", "--geometry", "9"],
                          capture_output=True, text=True)
    return engine if done.returncode != 0 and "geometry" in (done.stdout + done.stderr).lower() else None


needs_engine = pytest.mark.skipif(_engine() is None, reason="no rw_mpas_mesh that knows --geometry resolves here")


@needs_engine
def test_the_dry_run_with_the_word_counts_the_ramp_and_the_onset() -> None:
    from woof.hex.swath.sizing import dry_run

    asked = mesh_ask.expand(g2(zones=[{"name": "core", "shape": {"kind": "cap", "center_deg": [38.0, -96.0],
                                                                 "radius_km": 400.0}, "fine_dx_m": 3000.0}]))
    ramp = dry_run(asked["spec"], generator_words=asked["generator"]["words"])
    assert ramp["geometry"] == 2 and ramp["dry_run"] is True and ramp["gates_applied_by_hexcore"] is None
    # MEASURED 18 Nov 2026: this ask builds 163,522 columns under geometry 2 (the site's default zoom globe).
    assert ramp["predicted_cells"] == pytest.approx(163_522, rel=0.005)
    # And the census this module prints per zone agrees with the generator's own count.
    assert mesh_ask.census(asked, samples=1_000_000)["columns"] == pytest.approx(ramp["predicted_cells"], rel=0.02)
    bend = dry_run(asked["spec"], generator_words=["--geometry", "2", "--ramp-onset", "1", "--rings", "keep"])
    assert bend["predicted_cells"] > ramp["predicted_cells"] * 1.01
    onset = mesh_ask.expand(g2(zones=[{"name": "core", "shape": {"kind": "cap", "center_deg": [38.0, -96.0],
                                                                 "radius_km": 400.0}, "fine_dx_m": 3000.0}]),
                            ramp_onset=1.0)
    assert mesh_ask.census(onset, samples=1_000_000)["columns"] == pytest.approx(bend["predicted_cells"], rel=0.02)
    # The halving ladder's dry run of the door's own spec is the smaller count the door used to plan with.
    ladder = dry_run(mesh_point.ladder_spec((38.0, -96.0), fine_dx_m=3000.0, radius_km=400.0, background_km=96.0))
    assert ladder["predicted_cells"] == pytest.approx(149_317, rel=0.005)


@needs_engine
def test_the_generator_refuses_bad_words_in_plain_words(tmp_path) -> None:
    engine = _engine()
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(mesh_ask.expand(g2())["spec"]), encoding="utf-8")
    for words, expected in ((["--geometry", "3"], "geometry"),
                            (["--geometry", "2", "--ramp-onset", "9"], "--ramp-onset is 9"),
                            (["--geometry", "2", "--rings", "maybe"], "--rings takes ramp or keep"),
                            (["--geometry", "2", "--ramp-widths", "0.1"], "--ramp-widths is 0.1")):
        done = subprocess.run([str(engine), "--spec", str(spec), "--dry-run", *words], capture_output=True, text=True)
        assert done.returncode != 0 and expected in (done.stderr + done.stdout), (words, done.stderr[:300])


@needs_engine
def test_plan_only_through_the_door_prints_the_geometry_2_price(tmp_path, capsys) -> None:
    from woof.hex.cli import main

    path = tmp_path / "row.json"
    path.write_text(json.dumps(g2(background_km=60.0)), encoding="utf-8")
    assert main(["mesh-plan", "--spec", str(path), "--plan-only"]) == 0
    text = capsys.readouterr().out
    for expected in ("generator        geometry 2", "background: built at the 60 km asked (geometry 1 would build 48 km",
                     "columns ", "the forecast runs 20 s", "memory 96gb/hrrrv5", "8 cards"):
        assert expected in text, expected
    # The same ask as door words, priced the same way.
    assert main(["mesh-plan", "--point=38,-92", "--fine-dx-m", "3000", "--radius-km", "300", "--background-km", "60",
                 "--global-mesh", "--plan-only", "--geometry", "2", "--json"]) == 0
    planned = json.loads(capsys.readouterr().out)
    assert planned["generator"]["words"] == WORDS and planned["background"]["built_km"] == 60.0
