"""A geometry-2 point plan carries the width of its boundary rings, so the HRRR window covers the whole cut.

THE BREAKAGE THIS PREVENTS (0.6.2, 20 Nov 2026): with geometry 2 the default, `woof hex regional --point
40.6,-91.3 --radius-km 300 --fine-dx-m 3000 --background-km 96 --source hrrr --cycle 2024-05-21T15` (the 3 km storm
box over eastern Iowa) stopped at its init stage: "362 of the 51671 cells ... lie outside the first guess". The
geometry-2 plan's cull block had no ``halo_km``, so ``woof hex intermediate --from-plan`` cut the HRRR window at
the cut radius plus its 30 km margin (435 km) while the seven boundary rings reach 471 km. The geometry-1 plan
carries ``halo_km`` (rings times the spacing at the cut) and its run went through.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from woof.hex import hrrr_intermediate, mesh_point  # noqa: E402
from woof.hex.cli import build_parser  # noqa: E402

POINT = ("--point=40.6,-91.3", "--fine-dx-m", "3000", "--radius-km", "300", "--background-km", "96")


@pytest.fixture
def no_generator(monkeypatch):
    """The plan's parent sizing without running the generator binary (its count does not enter the cull's reach)."""
    import woof.hex.swath.sizing as sizing

    monkeypatch.setattr(sizing, "dry_run", lambda spec, **_: {"predicted_cells": 70000, "region_attainment": []})


def _plan(*words: str) -> dict:
    arguments = build_parser().parse_args(["mesh-plan", *POINT, *words])
    return mesh_point.plan_point(mesh_point.request_from_arguments(arguments))


@pytest.mark.parametrize("words", [(), ("--geometry", "2"), ("--geometry", "1")])
def test_the_plan_names_its_boundary_ring_width(no_generator, words) -> None:
    cull = _plan(*words)["cull"]
    at_cut = cull["boundary_rings"] * cull["spacing_at_cut_km"]
    if words == ("--geometry", "1"):
        assert cull["halo_km"] == pytest.approx(at_cut)
    else:
        # Geometry 2's rings are walked at their own spacing, which keeps growing past the cut
        # (tests/test_regional_ring_reach.py); never less than the rings at the cut's spacing.
        assert cull["halo_km"] >= at_cut
    assert cull["halo_km"] > 0.0


def test_the_hrrr_window_from_a_geometry_2_plan_reaches_past_the_rings(no_generator, tmp_path: Path) -> None:
    plan = _plan()
    path = tmp_path / "point-generate.json"
    path.write_text(json.dumps({"plan": plan}))
    target, basis = hrrr_intermediate.target_from_plan(path, margin_km=30.0, spacing_deg=0.025)
    cull = plan["cull"]
    reach = cull["cut_radius_km"] + cull["halo_km"]
    assert basis["halo_km"] == pytest.approx(cull["halo_km"])
    assert cull["halo_km"] >= cull["boundary_rings"] * cull["spacing_at_cut_km"]
    # The window's half-height in km is at least the cut and its rings, plus the margin.
    half_lat_km = 0.5 * (target.ny - 1) * target.dlat * hrrr_intermediate.KM_PER_DEG
    assert half_lat_km >= reach + 30.0 - 1.0
    # The run that stopped: its cut cells reached 471.3 km from the point; the window must hold them.
    assert half_lat_km > 471.3
    half_lon_km = 0.5 * (target.nx - 1) * target.dlon * hrrr_intermediate.KM_PER_DEG * math.cos(math.radians(40.6))
    assert half_lon_km > 471.3
