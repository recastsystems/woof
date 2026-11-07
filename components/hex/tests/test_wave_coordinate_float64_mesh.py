"""A float64 mesh (WOOF route) matches wave points prepared from its float32 native static; a wrong cell still fails.

THE BREAKAGE THIS CHECKS: every ArWen-route run with --waves on over the native global x1.40962 mesh refused at
service start, because the coordinate check gave a float64 mesh angle no float32 rounding allowance although the
preparation reads the float32 static.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from woof.hex.wave_prepared import coordinate_difference  # noqa: E402


def _prepared(radians64, *, longitude=False):
    static = (np.mod(radians64, 2 * np.pi) if longitude else radians64).astype(np.float32)  # the float32 static
    return np.rad2deg(static.astype(np.float64)).astype(np.float32)      # prepared/v1 float32 degrees


def test_float64_mesh_angles_match_points_prepared_from_the_float32_static():
    rng = np.random.default_rng(7)
    lat = rng.uniform(-np.pi / 2, np.pi / 2, 200_000)
    lon = rng.uniform(0, 2 * np.pi, 200_000)
    lon_signed = rng.uniform(-np.pi, np.pi, 200_000)   # a mesh in [-pi, pi), the static in [0, 2 pi)
    for mesh, longitude in ((lat, False), (lon, True), (lon_signed, True)):
        difference, tolerance = coordinate_difference(mesh, _prepared(mesh, longitude=longitude),
                                                      longitude=longitude, xp=np)
        assert np.all(difference <= tolerance)          # old code: tens of thousands outside


def test_a_neighbouring_cell_is_still_refused():
    lat = np.linspace(-1.2, 1.2, 1000)
    shifted = _prepared(lat + np.deg2rad(0.01))         # far less than any mesh spacing hex runs
    difference, tolerance = coordinate_difference(lat, shifted, xp=np)
    assert np.all(difference > tolerance)
