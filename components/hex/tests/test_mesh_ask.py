"""Mesh rows (``woof.hex.mesh_ask``): every option a row can ask for, and every refusal.

Generator-free and card-free: a row is expanded to the generator's own spec and that spec is
read; where a plan needs the generator's count, the dry run is replaced by the census itself.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from woof.hex import mesh_ask, mesh_point  # noqa: E402
from woof.hex.cli import build_parser  # noqa: E402
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


def row(**words):
    base = {"kind": mesh_ask.ROW_KIND, "background_km": 96.0,
            "zones": [{"name": "core", "shape": dict(CORE), "fine_dx_m": 3000.0}]}
    base.update(words)
    return base


def spacings(spec):
    return [region["spacing_km"] for region in spec["regions"]]


# ---------------------------------------------------------------------------
# the registered ladder is a row
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(("point", "fine", "radius", "background"), [
    ((35.0, -97.0), 937.5, 100.0, 120.0),     # the registered v0.9.120.110533 recipe
    ((38.0, -92.0), 3000.0, 1000.0, 96.0),    # today's zoom globe
    ((0.0, 0.0), 3000.0, 1000.0, 48.0),
    ((38.0, -96.0), 1500.0, 400.0, 96.0),
])
def test_the_legacy_ladder_is_a_row_to_the_digit(point, fine, radius, background) -> None:
    legacy = mesh_point.ladder_spec(point, fine_dx_m=fine, radius_km=radius, background_km=background, name="x")
    asked = mesh_ask.expand(mesh_ask.legacy_point_row(point, fine_dx_m=fine, radius_km=radius,
                                                      background_km=background, name="x"))
    assert json.dumps(asked["spec"], sort_keys=True) == json.dumps(legacy, sort_keys=True)
    assert asked["notes"] == [] and asked["cut"] is None


def test_the_door_writes_the_legacy_spec_when_no_new_word_is_given() -> None:
    arguments = build_parser().parse_args(
        ["mesh-plan", "--point=38,-92", "--fine-dx-m", "3000", "--radius-km", "1000", "--background-km", "96"])
    assert not mesh_point.arguments_ask_more(arguments)
    request = mesh_point.request_from_arguments(arguments)
    assert request.ask is None
    assert mesh_point.request_spec(request) == mesh_point.ladder_spec(
        (38.0, -92.0), fine_dx_m=3000.0, radius_km=1000.0, background_km=96.0)


# ---------------------------------------------------------------------------
# any fine spacing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(("asked", "fine", "how", "built"), [
    (96.0, 1.0, "nearest", 128.0), (96.0, 1.0, "finer", 64.0), (96.0, 1.0, "coarser", 128.0),
    (48.0, 2.0, "nearest", 64.0), (48.0, 2.0, "finer", 32.0),
    (96.0, 2.5, "nearest", 80.0), (96.0, 4.0, "nearest", 128.0), (96.0, 4.0, "finer", 64.0),
    (15.0, 3.0, "nearest", 12.0), (15.0, 3.0, "coarser", 24.0),
    (96.0, 3.0, "exact", 96.0), (96.0, 1.5, "nearest", 96.0), (24.0, 3.0, "exact", 24.0), (12.0, 3.0, "exact", 12.0),
])
def test_fit_to_ladder(asked, fine, how, built) -> None:
    record = mesh_ask.fit_to_ladder(asked, fine, how)
    assert record["built_km"] == pytest.approx(built)
    assert record["moved"] == (abs(built - asked) > 1e-9)
    assert math.log2(record["built_km"] / fine) == pytest.approx(record["halvings"])


def test_fit_exact_refuses_naming_both_sides() -> None:
    with pytest.raises(MeshAskRefusal, match="64 km or 128 km") as caught:
        mesh_ask.fit_to_ladder(96.0, 1.0, "exact")
    assert "750 m or 1500 m" in str(caught.value)


def test_fit_never_builds_a_background_as_fine_as_the_core() -> None:
    # 1.4x the fine spacing: the finer side is the core itself, so the fit goes up.
    assert mesh_ask.fit_to_ladder(4.2, 3.0, "finer")["built_km"] == pytest.approx(6.0)
    assert mesh_ask.fit_to_ladder(4.2, 3.0, "nearest")["built_km"] == pytest.approx(6.0)
    with pytest.raises(MeshAskRefusal, match="not coarser"):
        mesh_ask.fit_to_ladder(3.0, 3.0)
    with pytest.raises(MeshAskRefusal, match="not one of"):
        mesh_ask.fit_to_ladder(96.0, 3.0, "closest")


@pytest.mark.parametrize("fine_m", [1000.0, 1500.0, 2000.0, 2500.0, 3000.0, 4000.0, 750.0, 5000.0])
@pytest.mark.parametrize("background", [96.0, 48.0, 24.0, 15.0, 12.0])
def test_any_fine_spacing_on_any_background_builds_a_ladder(fine_m, background) -> None:
    asked = mesh_ask.expand(row(background_km=background,
                                zones=[{"shape": dict(CORE), "fine_dx_m": fine_m}]))
    spec = asked["spec"]
    fine = fine_m / 1000.0
    assert min(spacings(spec)) == pytest.approx(fine)           # the fine spacing is what was bought
    ratio = math.log2(spec["background_km"] / fine)
    assert ratio == pytest.approx(round(ratio)) and round(ratio) >= 1
    assert abs(math.log2(spec["background_km"] / background)) <= 0.5 + 1e-9 or spec["background_km"] == 2 * fine
    assert sorted(spacings(spec), reverse=True) == spacings(spec)
    assert [a / b for a, b in zip(spacings(spec), spacings(spec)[1:])] == [2.0] * (len(spec["regions"]) - 1)
    moved = abs(spec["background_km"] - background) > 1e-9
    assert bool(asked["notes"]) == moved
    if moved:
        assert f"asked {background:g} km, built {spec['background_km']:g} km" in asked["notes"][0]


def test_row_fit_exact_refuses_an_off_ladder_background() -> None:
    with pytest.raises(MeshAskRefusal, match="does not divide 96 km by halving"):
        mesh_ask.expand(row(fit="exact", zones=[{"shape": dict(CORE), "fine_dx_m": 1000.0}]))


# ---------------------------------------------------------------------------
# holding the fine spacing to the outline
# ---------------------------------------------------------------------------
def test_hold_draws_the_fine_region_outside_the_outline() -> None:
    held = mesh_ask.expand(row())["spec"]["regions"][-1]
    assert held["shape"]["radius_km"] == pytest.approx(300.0 + 1.5 * 18.0 * 3.0)
    plain = mesh_ask.expand(row(hold_fine_to_outline=False))["spec"]["regions"][-1]
    assert plain["shape"]["radius_km"] == pytest.approx(300.0)


def test_the_field_is_fine_on_the_whole_outline_when_held() -> None:
    import numpy as np

    asked = mesh_ask.expand(row())
    counted = mesh_ask.census(asked, samples=2_000_000)
    zone = counted["zones"][0]
    area = mesh_ask.shape_area_km2(CORE)
    # Every column inside the outline is within 5 % of 3 km: the count is the area over a 3 km hexagon.
    assert zone["columns"] == pytest.approx(area / (mesh_ask.HEXAGON_AREA_FACTOR * 9.0), rel=0.05)
    assert zone["median_spacing_km"] == pytest.approx(3.0, rel=0.01)
    assert np.isfinite(counted["columns"]) and counted["columns"] > zone["columns"]
    assert counted["spacing_km_by_column"]["min"] == pytest.approx(3.0, rel=0.01)
    assert counted["spacing_km_by_column"]["max"] == pytest.approx(96.0, rel=0.01)


# ---------------------------------------------------------------------------
# shapes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shape", [
    {"kind": "cap", "center_deg": [38.0, -92.0], "radius_km": 250.0},
    {"kind": "ellipse", "center_deg": [38.0, -97.0], "semi_major_km": 500.0, "semi_minor_km": 250.0,
     "azimuth_deg": 90.0},
    {"kind": "rounded_box", "center_deg": [48.0, 12.0], "width_km": 600.0, "height_km": 400.0, "corner_km": 80.0,
     "azimuth_deg": 20.0},
    {"kind": "lat_lon_box", "lat_deg": [34.0, 38.0], "lon_deg": [-100.0, -94.0]},
    {"kind": "polygon", "vertices_deg": [[34.0, -100.0], [34.0, -94.0], [38.0, -94.0], [38.0, -100.0]]},
    {"type": "Polygon", "coordinates": [[[-100.0, 34.0], [-94.0, 34.0], [-94.0, 38.0], [-100.0, 38.0],
                                         [-100.0, 34.0]]]},
])
def test_every_shape_expands_to_rings_that_hold_it(shape) -> None:
    import numpy as np

    asked = mesh_ask.expand(row(zones=[{"shape": shape, "fine_dx_m": 3000.0}]))
    spec = asked["spec"]
    assert spacings(spec) == [48.0, 24.0, 12.0, 6.0, 3.0]
    outline = asked["zones"][0]["shape"]
    lat0, lon0 = asked["zones"][0]["centre_deg"]
    assert float(mesh_ask.signed_distance_km(outline, lat0, lon0)) < 0.0
    # Each ring contains the next finer one: the fine outline's own points are inside every ring.
    if outline["kind"] == "polygon":
        lat, lon = np.asarray(outline["vertices_deg"]).T
    else:
        lat, lon = np.asarray([lat0 + 1.0, lat0 - 1.0]), np.asarray([lon0, lon0])
    for region in spec["regions"]:
        assert (mesh_ask.signed_distance_km(region["shape"], lat, lon) < 0.0).all()
        if region["shape"]["kind"] == "polygon":
            assert len(region["shape"]["vertices_deg"]) <= mesh_ask.RING_VERTICES


def test_ellipse_axes_and_turn() -> None:
    shape = mesh_ask.shape_row({"kind": "ellipse", "center_deg": [0.0, 0.0], "semi_major_km": 600.0,
                                "semi_minor_km": 200.0, "azimuth_deg": 90.0})
    assert mesh_ask.shape_area_km2(shape) == pytest.approx(math.pi * 600.0 * 200.0, rel=0.01)
    east = float(mesh_ask.signed_distance_km(shape, 0.0, 5.0))      # 556 km east: inside the long axis
    north = float(mesh_ask.signed_distance_km(shape, 5.0, 0.0))     # 556 km north: outside the short one
    assert east < 0.0 < north


def test_rounded_box_area_and_refusals() -> None:
    shape = mesh_ask.shape_row({"kind": "rounded_box", "center_deg": [10.0, 20.0], "width_km": 600.0,
                                "height_km": 400.0, "corner_km": 100.0})
    exact = 600.0 * 400.0 - (4.0 - math.pi) * 100.0 ** 2
    assert mesh_ask.shape_area_km2(shape) == pytest.approx(exact, rel=0.01)
    with pytest.raises(MeshAskRefusal, match="largest is half the short side, 200 km"):
        mesh_ask.shape_row({"kind": "rounded_box", "center_deg": [0, 0], "width_km": 600.0, "height_km": 400.0,
                            "corner_km": 250.0})
    with pytest.raises(MeshAskRefusal, match="encloses nothing"):
        mesh_ask.shape_row({"kind": "rounded_box", "center_deg": [0, 0], "width_km": 0.0, "height_km": 400.0})
    with pytest.raises(MeshAskRefusal, match="missing 'height_km'"):
        mesh_ask.shape_row({"kind": "rounded_box", "center_deg": [0, 0], "width_km": 10.0})


@pytest.mark.parametrize(("shape", "words"), [
    ({"kind": "cap", "center_deg": [38.0, -92.0], "radius_km": 0.0}, "encloses nothing"),
    ({"kind": "cap", "center_deg": [38.0, -92.0], "radius_km": 12000.0}, "quarter of the way round"),
    ({"kind": "cap", "center_deg": [138.0, -92.0], "radius_km": 10.0}, "not a place"),
    ({"kind": "cap", "radius_km": 10.0}, "latitude, longitude"),
    ({"kind": "blob"}, "neither a Shape row"),
    ({"kind": "polygon", "vertices_deg": [[0, 0], [1, 1]]}, "encloses no area"),
    ({"kind": "ellipse", "center_deg": [0, 0], "semi_major_km": 9000.0, "semi_minor_km": 8000.0}, "from its centre"),
    ("a circle", "not an object"),
])
def test_shape_refusals(shape, words) -> None:
    with pytest.raises(MeshAskRefusal, match=words):
        mesh_ask.shape_row(shape)


# ---------------------------------------------------------------------------
# several zones, nested zones
# ---------------------------------------------------------------------------
def test_two_separate_zones_each_get_their_own_rings() -> None:
    asked = mesh_ask.expand(row(zones=[
        {"name": "plains", "shape": dict(CORE), "fine_dx_m": 3000.0},
        {"name": "gulf", "shape": {"kind": "cap", "center_deg": [28.0, -90.0], "radius_km": 200.0},
         "fine_dx_m": 3000.0}]))
    assert sorted(spacings(asked["spec"])) == sorted([3.0, 6.0, 12.0, 24.0, 48.0] * 2)
    assert [zone["inside"] for zone in asked["zones"]] == [None, None]


def test_a_nested_zone_climbs_only_to_its_holder() -> None:
    asked = mesh_ask.expand(row(zones=[
        {"name": "zone", "shape": {"kind": "cap", "center_deg": [38.0, -92.0], "radius_km": 600.0},
         "fine_dx_m": 3000.0},
        {"name": "core", "shape": {"kind": "cap", "center_deg": [38.4, -91.2], "radius_km": 100.0},
         "fine_dx_m": 750.0}]))
    zones = {zone["name"]: zone for zone in asked["zones"]}
    assert zones["core"]["inside"] == "zone"                       # found by its centre, not named
    assert [ring["spacing_km"] for ring in zones["core"]["rings"]] == [0.75, 1.5]
    assert [ring["spacing_km"] for ring in zones["zone"]["rings"]] == [3.0, 6.0, 12.0, 24.0, 48.0]
    assert asked["notes"] == []


def test_a_nested_ask_that_does_not_halve_is_moved_and_said() -> None:
    zones = [{"name": "zone", "shape": {"kind": "cap", "center_deg": [38.0, -92.0], "radius_km": 600.0},
              "fine_dx_m": 3000.0},
             {"name": "core", "shape": {"kind": "cap", "center_deg": [38.4, -91.2], "radius_km": 150.0},
              "fine_dx_m": 1000.0, "inside": "zone"}]
    asked = mesh_ask.expand(row(zones=zones))
    built = {zone["name"]: zone["fine_m"] for zone in asked["zones"]}
    assert built == {"zone": 2000.0, "core": 1000.0}               # finer, never coarser, for a zone
    assert any("asked 3000 m, built 2000 m" in note and "4000 m" in note for note in asked["notes"])
    assert asked["spec"]["background_km"] == 128.0
    zones[0]["fit"] = "coarser"
    assert mesh_ask.expand(row(zones=zones))["zones"][0]["fine_m"] == 4000.0
    zones[0]["fit"] = "exact"
    with pytest.raises(MeshAskRefusal, match="does not divide 3 km by halving"):
        mesh_ask.expand(row(zones=zones))


def test_nesting_refusals() -> None:
    with pytest.raises(MeshAskRefusal, match="no zone has that name"):
        mesh_ask.expand(row(zones=[{"shape": dict(CORE), "fine_dx_m": 3000.0, "inside": "nowhere"}]))
    with pytest.raises(MeshAskRefusal, match="which is no coarser"):
        mesh_ask.expand(row(zones=[
            {"name": "a", "shape": dict(CORE), "fine_dx_m": 3000.0},
            {"name": "b", "shape": {"kind": "cap", "center_deg": [38.0, -92.0], "radius_km": 50.0},
             "fine_dx_m": 3000.0, "inside": "a"}]))
    with pytest.raises(MeshAskRefusal, match="two zones are named"):
        mesh_ask.expand(row(zones=[{"name": "a", "shape": dict(CORE), "fine_dx_m": 3000.0},
                                   {"name": "a", "shape": dict(CORE), "fine_dx_m": 1500.0}]))


# ---------------------------------------------------------------------------
# grading
# ---------------------------------------------------------------------------
def test_grading_numbers() -> None:
    legacy = mesh_ask.grading_numbers(None)
    assert legacy["ramp_scale_cells"] == 18.0 and legacy["edge_step_cells"] == 27.0
    assert legacy["steepest_growth_per_cell"] == pytest.approx(2.0 / 18.0)
    ramp = mesh_ask.grading_numbers({"kind": "ramp", "growth_per_cell": 0.08})
    assert ramp["ramp_scale_cells"] == pytest.approx(25.0) and ramp["steepest_growth_per_cell"] == pytest.approx(0.08)
    rings = mesh_ask.grading_numbers({"kind": "rings", "ring_cells": 10, "ramp_cells": 40})
    assert rings["ramp_scale_cells"] == 20.0 and rings["edge_step_cells"] == 50.0 and rings["ring_cells"] == 10.0
    assert mesh_ask.grading_numbers({"transition_factor": 12, "ring_factor": 4})["edge_step_cells"] == 24.0


def test_ring_cells_set_the_distance_between_ring_edges() -> None:
    asked = mesh_ask.expand(row(grading={"kind": "rings", "ring_cells": 10, "ramp_cells": 40}))
    rings = asked["zones"][0]["rings"]
    assert rings[0]["edge_beyond_outline_km"] == pytest.approx(1.5 * 20.0 * 3.0)
    for inner, outer in zip(rings, rings[1:]):
        step = outer["edge_beyond_outline_km"] - inner["edge_beyond_outline_km"]
        assert step == pytest.approx((10.0 + 40.0) * outer["spacing_km"])
    assert all(region["transition_km"] == pytest.approx(20.0 * region["spacing_km"])
               for region in asked["spec"]["regions"])


def test_a_gentler_ramp_is_wider_and_costs_more_columns() -> None:
    steep = mesh_ask.census(mesh_ask.expand(row(grading={"kind": "ramp", "growth_per_cell": 0.15})), samples=200_000)
    gentle = mesh_ask.census(mesh_ask.expand(row(grading={"kind": "ramp", "growth_per_cell": 0.06})), samples=200_000)
    assert gentle["columns"] > steep["columns"]


@pytest.mark.parametrize(("grading", "words"), [
    ({"kind": "spiral"}, "not ladder, rings or ramp"),
    ({"kind": "ramp", "growth_per_cell": 0.0}, "not between 0 and 0.25"),
    ({"kind": "ramp", "growth_per_cell": 0.4}, "not between 0 and 0.25"),
    ({"kind": "ramp", "growth_per_cell": 0.1, "ramp_cells": 30}, "names the ramp twice"),
    ({"kind": "rings"}, "needs ring_cells"),
    ({"kind": "rings", "ring_cells": -1}, "negative"),
    ({"kind": "rings", "ring_cells": 4, "ring_factor": 3}, "ring width twice"),
    ({"kind": "ramp", "ring_cells": 4}, "has no flat rings"),
    ({"kind": "ladder", "ramp_cells": 0}, "must have a width"),
    ({"kind": "ladder", "ring_factor": 0}, "ring_factor must be positive"),
    ({"kind": "ladder", "wobble": 1}, "does not know: wobble"),
])
def test_grading_refusals(grading, words) -> None:
    with pytest.raises(MeshAskRefusal, match=words):
        mesh_ask.grading_numbers(grading)


# ---------------------------------------------------------------------------
# a region: rim, rim spacing, boundary rings
# ---------------------------------------------------------------------------
def test_a_region_holds_its_rim_spacing_to_the_cut() -> None:
    asked = mesh_ask.expand(row(
        zones=[{"name": "disc", "shape": {"kind": "cap", "center_deg": [38.0, -92.0], "radius_km": 1000.0},
                "fine_dx_m": 3000.0}],
        domain={"kind": "region", "rim_km": 400.0, "rim_spacing_km": 12.0}))
    cut = asked["cut"]
    assert cut["region"] == {"kind": "cap", "center_deg": [38.0, -92.0], "radius_km": 1400.0}
    assert cut["boundary_rings"] == 7 and cut["around"] == "disc" and cut["rim_spacing_km"] == 12.0
    twelve = [region for region in asked["spec"]["regions"] if region["spacing_km"] == 12.0]
    # The ladder's own 12 km ring, and the rim plateau past the cut by 7 rings and two ramp scales.
    assert max(region["shape"]["radius_km"] for region in twelve) == pytest.approx(
        1000.0 + 400.0 + (7 + 2 * 18.0) * 12.0)
    counted = mesh_ask.census(asked, samples=400_000)
    assert counted["spacing_at_cut_km"] == pytest.approx(12.0, rel=0.03)
    assert 0 < counted["columns_in_boundary_rings"] < 0.03 * counted["columns"]


def test_a_region_without_a_rim_spacing_adds_no_region() -> None:
    plain = mesh_ask.expand(row())
    region = mesh_ask.expand(row(domain={"kind": "region", "rim_km": 300.0}))
    assert region["spec"] == plain["spec"]
    assert region["cut"]["region"]["radius_km"] == pytest.approx(600.0) and region["cut"]["rim_spacing_km"] is None


def test_a_drawn_region_is_cut_along_its_grown_outline() -> None:
    shape = {"kind": "ellipse", "center_deg": [48.5, 13.0], "semi_major_km": 400.0, "semi_minor_km": 250.0}
    asked = mesh_ask.expand(row(zones=[{"shape": shape, "fine_dx_m": 3000.0}],
                                domain={"kind": "region", "rim_km": 300.0, "rim_spacing_km": 12.0}))
    cut = asked["cut"]["region"]
    assert cut["kind"] == "polygon"
    grown_area = math.pi * 700.0 * 550.0
    assert mesh_ask.shape_area_km2(cut) == pytest.approx(grown_area, rel=0.03)


@pytest.mark.parametrize(("domain", "words"), [
    ({"kind": "region"}, "needs rim_km"),
    ({"kind": "region", "rim_km": 0.0}, "not positive"),
    ({"kind": "region", "rim_km": 400.0, "boundary_rings": 5}, "exactly 7 rings"),
    ({"kind": "region", "rim_km": 100.0, "rim_spacing_km": 12.0}, "would fall on the ramp"),
    ({"kind": "region", "rim_km": 400.0, "rim_spacing_km": 1.0}, "finer than the zone it rims"),
    ({"kind": "region", "rim_km": 400.0, "around": "nowhere"}, "no zone has that name"),
    ({"kind": "region", "rim_km": 400.0, "walls": 2}, "does not know: walls"),
    ({"kind": "globe", "rim_km": 400.0}, "a globe has no rim"),
    ({"kind": "torus"}, "not globe or region"),
])
def test_domain_refusals(domain, words) -> None:
    with pytest.raises(MeshAskRefusal, match=words):
        mesh_ask.expand(row(domain=domain))


def test_a_region_with_two_outer_zones_needs_one_named() -> None:
    zones = [{"name": "plains", "shape": dict(CORE), "fine_dx_m": 3000.0},
             {"name": "gulf", "shape": {"kind": "cap", "center_deg": [28.0, -90.0], "radius_km": 200.0},
              "fine_dx_m": 3000.0}]
    with pytest.raises(MeshAskRefusal, match="name one with domain.around"):
        mesh_ask.expand(row(zones=zones, domain={"kind": "region", "rim_km": 400.0}))
    asked = mesh_ask.expand(row(zones=zones, domain={"kind": "region", "rim_km": 400.0, "around": "plains"}))
    assert asked["cut"]["around"] == "plains"


# ---------------------------------------------------------------------------
# the row itself
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(("change", "words"), [
    ({"zones": []}, "needs zones"),
    ({"zones": "core"}, "needs zones"),
    ({"zones": [{"shape": dict(CORE)}]}, "needs fine_dx_m"),
    ({"zones": [{"shape": dict(CORE), "fine_dx_m": 10.0}]}, "finest spacing this door writes is 50 m"),
    ({"zones": [{"shape": dict(CORE), "fine_dx_m": 3000.0, "colour": "red"}]}, "does not know: colour"),
    ({"background_km": 2.0}, "not coarser than the coarsest zone"),
    ({"background_km": 900.0}, "not between 0 and 480"),
    ({"background_km": "wide"}, "not a number"),
    ({"fit": "closest"}, "not one of"),
    ({"kind": "something/else"}, "this door reads"),
    ({"speed": "fast"}, "does not know: speed"),
])
def test_row_refusals(change, words) -> None:
    with pytest.raises(MeshAskRefusal, match=words):
        mesh_ask.expand(row(**change))


def test_is_row_tells_a_row_from_the_generators_spec() -> None:
    assert mesh_ask.is_row(row())
    assert mesh_ask.is_row({"zones": [], "background_km": 96})
    assert not mesh_ask.is_row({"background_km": 96, "regions": []})
    assert not mesh_ask.is_row(mesh_ask.expand(row())["spec"])


def test_load_row(tmp_path) -> None:
    path = tmp_path / "row.json"
    path.write_text(json.dumps(row()), encoding="utf-8")
    assert mesh_ask.load_row(path)["background_km"] == 96.0
    path.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(MeshAskRefusal, match="a row is a JSON object"):
        mesh_ask.load_row(path)
    with pytest.raises(MeshAskRefusal, match="cannot read a JSON row"):
        mesh_ask.load_row(tmp_path / "missing.json")


# ---------------------------------------------------------------------------
# the plan
# ---------------------------------------------------------------------------
def _fake_dry_run(monkeypatch) -> None:
    from woof.hex.swath import sizing

    def dry_run(spec, *, engine=None, card=None, vram_gib=None):
        asked = {"spec": spec, "zones": [], "cut": None}
        return {"predicted_cells": mesh_ask.census(asked, samples=200_000)["columns"],
                "steepest_requested_gradient_percent_per_cell": 10.6, "gates_applied_by_hexcore": {}}

    monkeypatch.setattr(sizing, "dry_run", dry_run)


def test_plan_prices_a_globe(monkeypatch) -> None:
    _fake_dry_run(monkeypatch)
    planned = mesh_ask.plan(row(background_km=48.0, zones=[{"name": "core", "shape": dict(CORE),
                                                             "fine_dx_m": 2000.0}]), samples=200_000)
    assert planned["domain"] == "globe" and planned["cut"] is None
    assert planned["background"]["built_km"] == 64.0 and planned["notes"]
    assert planned["columns"] == planned["columns_whole_sphere"]
    assert planned["step"]["dt_seconds"] == 12.0
    for key in ("96gb/hrrrv5", "96gb/convection_permitting", "32gb/hrrrv5", "32gb/convection_permitting"):
        assert sorted(planned["memory"][key]["cards"]) == ["1", "2", "4", "8"]
    one, eight = (planned["memory"]["96gb/hrrrv5"]["cards"][count]["peak_mib_per_card"] for count in ("1", "8"))
    assert one > eight > 0
    text = "\n".join(mesh_ask.report(planned))
    for words in ("asked 48 km, built 64 km", "zone core", "spacing by column", "time step        12 s",
                  "memory 96gb/hrrrv5", "1 card", "8 cards", "whole sphere, no rim"):
        assert words in text


def test_plan_prices_a_region_on_its_cut(monkeypatch) -> None:
    _fake_dry_run(monkeypatch)
    planned = mesh_ask.plan(row(domain={"kind": "region", "rim_km": 300.0, "rim_spacing_km": 12.0}),
                            samples=200_000, cards=["32gb"])
    assert planned["domain"] == "region" and planned["cull_region"]["radius_km"] == pytest.approx(600.0)
    assert planned["columns"] < planned["columns_whole_sphere"]
    assert list(planned["memory"]) == ["32gb/hrrrv5", "32gb/convection_permitting"]
    assert "7 boundary rings" in "\n".join(mesh_ask.report(planned))


# ---------------------------------------------------------------------------
# the doors
# ---------------------------------------------------------------------------
def _parse(*words: str) -> argparse.Namespace:
    return build_parser().parse_args(["mesh-plan", *words])


def test_point_words_become_a_row() -> None:
    arguments = _parse("--point=38,-92", "--fine-dx-m", "1000", "--radius-km", "150", "--background-km", "96",
                       "--background-fit", "nearest", "--hold-fine", "--growth-per-cell", "0.08", "--ring-cells", "6",
                       "--rim-km", "500", "--rim-spacing-km", "8",
                       "--zone", json.dumps({"shape": {"kind": "cap", "center_deg": [38.2, -91.8], "radius_km": 40.0},
                                             "fine_dx_m": 500.0}))
    assert mesh_point.arguments_ask_more(arguments)
    request = mesh_point.request_from_arguments(arguments)
    asked = request.ask
    assert asked is not None and request.fine_dx_m == 500.0 and request.background_km == 128.0
    assert [zone["name"] for zone in asked["zones"]] == ["core", "zone1"]
    assert asked["zones"][1]["inside"] == "core"
    assert asked["cut"]["rim_km"] == 500.0 and asked["cut"]["rim_spacing_km"] == 8.0
    assert asked["zones"][0]["grading"]["ramp_scale_cells"] == pytest.approx(25.0)
    assert mesh_point.request_spec(request) == asked["spec"]
    assert mesh_point.request_cull_region(request)["radius_km"] == pytest.approx(650.0)
    assert mesh_point.request_halo_region(request, 1600.0)["radius_km"] == pytest.approx(2250.0)


def test_an_off_ladder_point_is_still_refused_without_the_word() -> None:
    request = mesh_point.request_from_arguments(
        _parse("--point=38,-92", "--fine-dx-m", "1000", "--radius-km", "150", "--background-km", "96"))
    with pytest.raises(mesh_point.PointPlanRefusal, match="--background-fit nearest"):
        mesh_point.request_spec(request)


def test_a_region_row_file_is_the_limited_area_request(tmp_path) -> None:
    path = tmp_path / "row.json"
    path.write_text(json.dumps(row(domain={"kind": "region", "rim_km": 300.0})), encoding="utf-8")
    request = mesh_point.request_from_arguments(_parse("--spec", str(path)))
    assert request.ask is not None and request.point == (38.0, -92.0) and request.radius_km == 300.0
    path.write_text(json.dumps(row()), encoding="utf-8")
    with pytest.raises(mesh_point.PointPlanRefusal, match="whole globe"):
        mesh_point.request_from_arguments(_parse("--spec", str(path)))
    path.write_text(json.dumps({"background_km": 96, "regions": []}), encoding="utf-8")
    with pytest.raises(mesh_point.PointPlanRefusal, match="not a mesh row"):
        mesh_point.request_from_arguments(_parse("--spec", str(path)))


def test_a_drawn_core_on_a_globe_is_held_fine() -> None:
    arguments = _parse("--global-mesh", "--ellipse=38,-97,700,400,90", "--fine-dx-m", "3000", "--background-km", "96")
    asked = mesh_point.expand_arguments(arguments, globe=True)
    assert asked["cut"] is None and asked["hold_fine_to_outline"]
    assert spacings(asked["spec"]) == [48.0, 24.0, 12.0, 6.0, 3.0]
    with pytest.raises(mesh_point.PointPlanRefusal, match="a globe has no rim"):
        mesh_point.expand_arguments(_parse("--global-mesh", "--point=38,-92", "--rim-km", "300"), globe=True)


def test_zone_word_refusals(tmp_path) -> None:
    with pytest.raises(mesh_point.PointPlanRefusal, match="neither a JSON object nor a file"):
        mesh_point.arguments_row(_parse("--point=38,-92", "--zone", "{not json"))
    with pytest.raises(mesh_point.PointPlanRefusal, match="neither a JSON object nor a file"):
        mesh_point.arguments_row(_parse("--point=38,-92", "--zone", str(tmp_path / "missing.json")))


def test_the_regional_door_takes_a_region_row(tmp_path) -> None:
    from woof.hex import regional_door

    path = tmp_path / "row.json"
    path.write_text(json.dumps(row(domain={"kind": "region", "rim_km": 300.0, "rim_spacing_km": 12.0})),
                    encoding="utf-8")
    arguments = argparse.Namespace(mesh_spec=path, point=None, shape=None, ellipse=None, fine_dx_m=None,
                                   radius_km=None, background_km=None)
    asked = regional_door._mesh_row(arguments)
    assert arguments.point == "38.0,-92.0" and arguments.radius_km == 300.0
    assert arguments.fine_dx_m == 3000.0 and arguments.background_km == 96.0
    # The cut at 600 km and seven 12 km boundary rings, walked at their own spacing with the relaxed rows'
    # allowance (mesh_ask.RING_REACH_SCALE): never less than seven rings of 12 km.
    reach = regional_door._mesh_row_reach(asked, (38.0, -92.0))
    assert 600.0 + 7 * 12.0 <= reach <= 600.0 + mesh_ask.RING_REACH_SCALE * 7 * 12.5
    arguments = argparse.Namespace(mesh_spec=path, point="38,-92", shape=None, ellipse=None, fine_dx_m=3000.0,
                                   radius_km=None, background_km=None)
    with pytest.raises(regional_door.RegionalRefusal, match="--point, --fine-dx-m would say them twice"):
        regional_door._mesh_row(arguments)
    path.write_text(json.dumps(row()), encoding="utf-8")
    arguments = argparse.Namespace(mesh_spec=path, point=None, shape=None, ellipse=None, fine_dx_m=None,
                                   radius_km=None, background_km=None)
    with pytest.raises(regional_door.RegionalRefusal, match="is a whole globe"):
        regional_door._mesh_row(arguments)
    assert regional_door._mesh_row(argparse.Namespace(mesh_spec=None)) is None


def test_the_country_door_moves_the_background_only_when_asked(capsys) -> None:
    from woof.hex import country_door
    from woof.hex.errors import MpasPortError

    def region(*words):
        arguments = build_parser().parse_args(["country", "--point=38,-92", "--radius-km", "300", "--fine-dx-m",
                                               "2000", "--background-km", "48", "--cycle", "2024-05-21T18",
                                               "--work-dir", "x", *words])
        return country_door.region_request(arguments)[0]

    with pytest.raises(MpasPortError, match="not a rung of the 48 km halving ladder"):
        region()
    built = region("--background-fit", "nearest")
    assert built["mesh"]["fine_dx_m"] == 2000.0 and built["mesh"]["background_km"] == 64.0
    assert "background asked 48 km, built 64 km" in capsys.readouterr().out
    finer = region("--background-fit", "finer")
    assert finer["mesh"]["background_km"] == 32.0 and finer["id"] != built["id"]
    spec = country_door.region_specification(built)
    assert spec["background_km"] == 64.0 and min(r["spacing_km"] for r in spec["regions"]) == 2.0
