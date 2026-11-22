"""0.6.2: a mesh request that names no geometry is built under geometry 2; ``--geometry 1`` is the ladder.

THE BREAKAGE THIS PREVENTS (20 Nov 2026): under geometry 1, the generator in the public 0.6.1 wheel took 10 to
12 minutes a try on a 1.96 M-cell 3 km CONUS zoom globe on a 384-thread host and refused twice for one
four-edge cell; geometry 2 built the same ask in 96 s with no four-edge cell.  These tests hold the door's
default at 2 and the old meshes one word away.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from woof.hex import mesh_ask, mesh_plan_door, mesh_point  # noqa: E402
from woof.hex.cli import build_parser  # noqa: E402

POINT = ("--point=38,-92", "--fine-dx-m", "3000", "--radius-km", "1000", "--background-km", "96")


def _parse(*words: str) -> argparse.Namespace:
    return build_parser().parse_args(["mesh-plan", *words])


def test_the_default_geometry_is_2() -> None:
    assert mesh_ask.DEFAULT_GEOMETRY == 2
    asked = mesh_ask.generator_request({})
    assert asked["geometry"] == 2 and asked["words"] == ["--geometry", "2"]


def test_geometry_1_is_one_word_away_and_is_the_ladder() -> None:
    assert mesh_ask.generator_request({}, geometry=1)["words"] == []
    assert mesh_ask.generator_request({"generator": {"geometry": 1}})["words"] == []
    arguments = _parse(*POINT, "--geometry", "1")
    request = mesh_point.request_from_arguments(arguments)
    assert request.ask is None and request.generator_words == ()
    assert mesh_point.request_spec(request) == mesh_point.ladder_spec(
        (38.0, -92.0), fine_dx_m=3000.0, radius_km=1000.0, background_km=96.0)


def test_a_door_with_no_word_builds_what_geometry_2_named_builds() -> None:
    plain, named = _parse(*POINT), _parse(*POINT, "--geometry", "2")
    assert mesh_point.arguments_geometry(plain) == 2
    assert mesh_plan_door.door_generator_words(plain) == mesh_plan_door.door_generator_words(named) == ["--geometry", "2"]
    left = mesh_point.request_spec(mesh_point.request_from_arguments(plain))
    right = mesh_point.request_spec(mesh_point.request_from_arguments(named))
    assert json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def test_a_row_with_no_generator_is_built_under_geometry_2() -> None:
    row = {"name": "x", "background_km": 96.0,
           "kind": mesh_ask.ROW_KIND,
           "zones": [{"name": "core", "shape": {"kind": "cap", "center_deg": [38.0, -92.0], "radius_km": 400.0},
                      "fine_dx_m": 3000.0}]}
    plain = mesh_ask.expand(row)
    assert plain["generator"]["words"][:2] == ["--geometry", "2"]
    assert plain == mesh_ask.expand(row, geometry=2)
    assert "generator" not in mesh_ask.expand(row, geometry=1)
