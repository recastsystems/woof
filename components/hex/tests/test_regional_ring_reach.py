# SPDX-License-Identifier: Apache-2.0
"""A regional run's boundary rings are walked at their own spacing, and the door reads a projected source's grid
before the mesh and the download.

THE BREAKAGE THIS PREVENTS (0.6.2, 20 Nov 2026): `woof hex regional --point 38.5,-93.0 --radius-km 800
--fine-dx-m 10000 --background-km 80 --source hrrr --cycle 2024-05-21T15` (geometry 2, the default) stopped at its
init stage after a 28-minute HRRR intermediate: one of its 35,043 cells lay at 27.2244 N, 0.0006 degrees south of a
window that ended at 27.2250 N. The plan gave the seven boundary rings 7 x 20.4 km (the spacing at the 1,080 km cut,
142.7 km), but past the cut geometry 2's spacing keeps growing; the rings reached 173.8 km. The HRRR row also
declares no coverage window, so nothing was read against the HRRR grid before the download began.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from woof.hex import hrrr_intermediate, mesh_ask, mesh_point  # noqa: E402
from woof.hex.cli import build_parser  # noqa: E402

CASE = ("--point=38.5,-93.0", "--radius-km", "800", "--fine-dx-m", "10000", "--background-km", "80")
#: The cell of the run that stopped: 27.2244 N on the -93.0 meridian, 1,253.8 km from the point.
FAR_CELL_LAT = 27.2244


@pytest.fixture
def no_generator(monkeypatch):
    """The plan's parent sizing without running the generator binary (its count does not enter the rings' reach)."""
    import woof.hex.swath.sizing as sizing

    monkeypatch.setattr(sizing, "dry_run", lambda spec, **_: {"predicted_cells": 70000, "region_attainment": []})


def _plan(*words: str) -> dict:
    arguments = build_parser().parse_args(["mesh-plan", *CASE, *words])
    return mesh_point.plan_point(mesh_point.request_from_arguments(arguments))


def test_the_rings_are_walked_at_their_own_growing_spacing(no_generator) -> None:
    cull = _plan()["cull"]
    assert cull["cut_radius_km"] == pytest.approx(1080.0)
    at_cut = cull["boundary_rings"] * cull["spacing_at_cut_km"]
    assert at_cut == pytest.approx(142.7, abs=0.5)  # the width the plan gave before
    assert cull["halo_km"] > at_cut + 30.0
    assert cull["halo_km"] > 173.8  # where the run's outermost cell lay


def test_the_from_plan_window_holds_the_cell_that_stopped_the_run(no_generator, tmp_path: Path) -> None:
    path = tmp_path / "point-generate.json"
    path.write_text(json.dumps({"plan": _plan()}))
    target, basis = hrrr_intermediate.target_from_plan(path, margin_km=30.0, spacing_deg=0.025)
    assert target.south < FAR_CELL_LAT - 0.1
    assert basis["halo_km"] > 173.8


def test_a_ladder_plan_keeps_its_rings_at_the_cut_spacing(no_generator) -> None:
    cull = _plan("--geometry", "1")["cull"]
    assert cull["halo_km"] == pytest.approx(cull["boundary_rings"] * cull["spacing_at_cut_km"])


def test_the_walk_never_gives_less_than_the_rings_at_the_cut_spacing(no_generator) -> None:
    storm = build_parser().parse_args(["mesh-plan", "--point=40.6,-91.3", "--fine-dx-m", "3000", "--radius-km",
                                       "300", "--background-km", "96"])
    cull = mesh_point.plan_point(mesh_point.request_from_arguments(storm))["cull"]
    assert cull["halo_km"] >= cull["boundary_rings"] * cull["spacing_at_cut_km"]
    assert cull["cut_radius_km"] + cull["halo_km"] > 471.3  # the 3 km storm box's outermost cell


def test_the_field_at_points_is_the_census_field() -> None:
    request = mesh_point.request_from_arguments(build_parser().parse_args(["mesh-plan", *CASE]))
    asked = request.ask
    (lat0, lon0), radius = mesh_ask._reach(asked["cut"]["region"])
    # On the cut, south of the point: the census reads its median spacing on a 25 km band about the cut line.
    lat = lat0 - radius / mesh_ask.EARTH_RADIUS_KM * 180.0 / math.pi
    spacing = float(mesh_ask.field_spacing_km(asked, [lat], [lon0])[0])
    assert spacing == pytest.approx(mesh_ask.census(asked, samples=400_000)["spacing_at_cut_km"], rel=0.05)
    # Inside the zone the field is the zone's own spacing; far away it is the background.
    assert float(mesh_ask.field_spacing_km(asked, [lat0], [lon0])[0]) == pytest.approx(10.0)
    assert float(mesh_ask.field_spacing_km(asked, [-lat0], [lon0 + 180.0])[0]) == pytest.approx(80.0)


def test_a_generated_plan_draws_the_box_over_its_cull_grid(tmp_path: Path, monkeypatch) -> None:
    grid = tmp_path / "cull.grid.nc"
    grid.write_bytes(b"")
    plan = tmp_path / "point-generate.json"
    plan.write_text(json.dumps({"plan": {"cull_region": {"kind": "cap", "center_deg": [38.5, -93.0],
                                                         "radius_km": 1080.0}, "cull": {"halo_km": 142.68}},
                                "cull": {"grid": str(grid)}}))
    lat = np.array([38.5, FAR_CELL_LAT, 49.0])
    lon = np.array([-93.0, -93.0, -93.0])
    monkeypatch.setattr(hrrr_intermediate, "mesh_cell_degrees", lambda path: (lat, lon))
    monkeypatch.setattr(hrrr_intermediate, "_mesh_point_degrees", lambda path, kind: (lat, lon))
    arguments = build_parser().parse_args(["intermediate", "--source", "hrrr", "--from-plan", str(plan),
                                           "--grib-dir", str(tmp_path / "grib"), "--cycle", "2024-05-21T15",
                                           "--hours", "0", "--out-dir", str(tmp_path / "met")])
    target, basis = hrrr_intermediate._target_from_arguments(arguments)
    assert basis["cull_grid"] == str(grid)
    assert target.south <= FAR_CELL_LAT


def _door(*words: str):
    from woof.hex import regional_door

    arguments = build_parser().parse_args(["regional", *words, "--source", "hrrr", "--cycle", "2024-05-21T15",
                                           "--hours", "1", "--work-dir", "/nonexistent/door-plan-only"])
    return regional_door, arguments


def test_the_door_reads_the_hrrr_grid_before_anything_is_built_or_fetched(monkeypatch) -> None:
    regional_door, arguments = _door("--point", "30.0,-75.0", "--radius-km", "800", "--fine-dx-m", "10000",
                                     "--background-km", "80")
    started = []
    monkeypatch.setattr(regional_door, "_start_source_prefetch", lambda *a, **k: started.append(a))
    with pytest.raises(regional_door.RegionalRefusal, match="grid on its east side"):
        regional_door.plan(arguments)
    assert not started


@pytest.mark.parametrize("words", [
    ("--point", "38.5,-93.0", "--radius-km", "800", "--fine-dx-m", "10000", "--background-km", "80"),
    ("--point", "41.8,-93.8", "--radius-km", "250", "--fine-dx-m", "3000"),
])
def test_regions_inside_hrrr_pass_the_door(words) -> None:
    regional_door, arguments = _door(*words)
    value = regional_door.plan(arguments)
    assert value["source"]["id"] == "hrrr"


def test_the_cap_check_reads_the_hrrr_grid() -> None:
    row = hrrr_intermediate.SOURCE_ROWS["hrrr"]
    assert hrrr_intermediate.cap_coverage_refusal(row, (38.5, -93.0), 1300.0) is None
    refusal = hrrr_intermediate.cap_coverage_refusal(row, (38.5, -93.0), 2500.0)
    assert refusal and "nothing was built or downloaded" in refusal
