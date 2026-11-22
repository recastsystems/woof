"""The regional door's HRRR intermediate covers every cell of the cull, boundary rings included.

THE BREAKAGE (0.6.2, 20 Nov 2026, a fresh install on an RTX PRO 6000): `woof hex regional --mesh-spec` with a
row of a 3 km zone of 120 km, a 300 km rim and 12 km rim cells cut at 420 km and laid its seven boundary rings of 12
km cells outside the cut, to about 505 km. The intermediate stage took its box from the plan (the cut radius plus a
30 km margin, 450 km), so 553 of the cull's 18,320 cells lay outside the first guess and the init stage refused.
The stage now draws the box over the cull grid itself (`intermediate --grid`).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex import regional_door as door
from woof.hex import hrrr_intermediate as hi
from woof.hex.cli import build_parser


def _request(tmp_path):
    args = build_parser().parse_args([
        "regional", "--point", "35.4,-97.5", "--radius-km", "120", "--source", "hrrr",
        "--cycle", "2024-05-06T21", "--hours", "1", "--work-dir", str(tmp_path)])
    return args, door.plan(args)


def test_the_intermediate_box_is_drawn_over_the_cull_grid(tmp_path, monkeypatch):
    args, value = _request(tmp_path)
    cull = tmp_path / "mesh" / "x-cull.grid.nc"
    monkeypatch.setattr(door, "_generated", lambda v: {"_path": "plan.json", "cull": {"grid": str(cull)}})
    calls = []
    monkeypatch.setattr(door, "_cli", lambda argv, status: calls.append(argv) or 0)
    assert door._stage_intermediate(args, value, SimpleNamespace(note=lambda s: None))["returncode"] == 0
    (argv,) = calls
    assert argv[argv.index("--grid") + 1] == str(cull)
    assert "--from-plan" not in argv


def test_a_plan_with_no_cull_grid_keeps_the_plan_box(tmp_path, monkeypatch):
    args, value = _request(tmp_path)
    monkeypatch.setattr(door, "_generated", lambda v: {"_path": "plan.json"})
    calls = []
    monkeypatch.setattr(door, "_cli", lambda argv, status: calls.append(argv) or 0)
    assert door._stage_intermediate(args, value, SimpleNamespace(note=lambda s: None))["returncode"] == 0
    assert calls[0][3:5] == ["--from-plan", "plan.json"]


def test_the_cull_box_covers_rings_the_plan_box_misses(tmp_path, monkeypatch):
    """The measured cull: cells to 30.85..40.00 N about 35.4 N (cut 420 km, rings to ~505 km)."""
    rng = np.random.default_rng(7)
    lat = np.concatenate([rng.uniform(30.91, 39.94, 4000), [30.91, 39.94]])
    lon = np.concatenate([rng.uniform(-103.16, -91.88, 4000), [-103.16, -91.88]])
    monkeypatch.setattr(hi, "mesh_cell_degrees", lambda grid: (lat, lon % 360.0))
    monkeypatch.setattr(hi, "_mesh_point_degrees", lambda grid, kind: (lat, lon % 360.0))
    plan_box = hi.target_for_cap((35.4, -97.5), 420.0)
    assert plan_box.south > lat.min()          # what the plan's box left out
    box, _ = hi.target_from_mesh(Path("cull.grid.nc"), None)
    assert box.south <= lat.min() and box.south + (box.ny - 1) * box.dlat >= lat.max()
