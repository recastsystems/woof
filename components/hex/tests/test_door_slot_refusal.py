# SPDX-License-Identifier: Apache-2.0
"""A convection choice with no GPU route stops at the regional and global doors before anything is built or fetched.

THE BREAKAGE THIS PREVENTS (0.6.2, 21 Nov 2026): `woof hex regional --point 38.5,-93.0 --radius-km 800 --fine-dx-m 10000
--background-km 80 --source hrrr --cycle 2024-05-21T15 --hours 1 --forecast-arg=--physics-suite=hrrrv5
--forecast-arg=--physics=config_convection_scheme=cu_tiedtke`, from a fresh install, built the mesh, fetched HRRR and built
the start and the boundaries (17 GB, 282 s) before the forecast door refused cu_tiedtke, which has no GPU slot beside
hrrrv5's other schemes. The forecast door's own refusal is now read in the door's plan, before the mesh.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from woof.hex.cli import build_parser  # noqa: E402
from woof.hex.mpasphys.forecast_v842 import forecast_args_slot_refusal  # noqa: E402

HRRRV5 = ["--physics-suite", "hrrrv5"]


def _no_native_library(monkeypatch):
    """No native MPAS library is installed or named, as on every install of a release (none ships one)."""
    from woof.hex.mpasphys import resources_build

    def refuse(*_a, **_k):
        raise resources_build.PhysicsResourceError("not installed")

    monkeypatch.setattr(resources_build, "resolve_resource", refuse)
    monkeypatch.delenv("MPAS_NATIVE_LIBRARY", raising=False)


def test_a_cpu_slot_is_read_from_the_flags(monkeypatch) -> None:
    _no_native_library(monkeypatch)
    refusal = forecast_args_slot_refusal([*HRRRV5, "--physics", "config_convection_scheme=cu_tiedtke"])
    assert refusal is not None and "convection=cu_tiedtke" in refusal
    assert "--physics config_convection_scheme=cu_grell_freitas" in refusal


@pytest.mark.parametrize("flags", [
    [],
    HRRRV5,
    [*HRRRV5, "--physics", "config_convection_scheme=cu_grell_freitas"],
    ["--physics-suite", "convection_permitting"],
    [*HRRRV5, "--physics", "config_convection_scheme=cu_tiedtke", "--native-physics-library", "/site/libmpas.so"],
])
def test_choices_the_forecast_door_runs_pass(monkeypatch, flags) -> None:
    _no_native_library(monkeypatch)
    assert forecast_args_slot_refusal(flags) is None


def test_flags_the_door_cannot_read_are_left_to_the_forecast_door() -> None:
    assert forecast_args_slot_refusal(["--physics-suite", "no-such-suite"]) is None


def _regional(*words: str):
    from woof.hex import regional_door

    arguments = build_parser().parse_args(["regional", "--point", "38.5,-93.0", "--radius-km", "800", "--fine-dx-m",
                                           "10000", "--background-km", "80", "--source", "hrrr", "--cycle",
                                           "2024-05-21T15", "--hours", "1", "--work-dir", "/nonexistent/door-plan-only",
                                           *words])
    return regional_door, arguments


def test_the_regional_door_refuses_before_the_mesh(monkeypatch) -> None:
    _no_native_library(monkeypatch)
    regional_door, arguments = _regional("--forecast-arg=--physics-suite=hrrrv5",
                                         "--forecast-arg=--physics=config_convection_scheme=cu_tiedtke")
    started = []
    monkeypatch.setattr(regional_door, "_start_source_prefetch", lambda *a, **k: started.append(a))
    with pytest.raises(regional_door.RegionalRefusal, match="cu_tiedtke runs only on the CPU here"):
        regional_door.plan(arguments)
    assert not started


def test_the_regional_door_plans_grell_freitas_with_hrrrv5(monkeypatch) -> None:
    _no_native_library(monkeypatch)
    regional_door, arguments = _regional("--forecast-arg=--physics-suite=hrrrv5",
                                         "--forecast-arg=--physics=config_convection_scheme=cu_grell_freitas")
    assert regional_door.plan(arguments)["source"]["id"] == "hrrr"


def test_the_global_door_refuses_before_the_mesh(monkeypatch) -> None:
    _no_native_library(monkeypatch)
    from woof.hex import global_door

    arguments = build_parser().parse_args(["global", "--row", "global-120km", "--cycle", "2026-11-20T00",
                                           "--forecast-arg=--physics-suite=hrrrv5",
                                           "--forecast-arg=--physics=config_convection_scheme=cu_tiedtke",
                                           "--work-dir", "/nonexistent/door-plan-only"])
    with pytest.raises(global_door.GlobalDoorError, match="cu_tiedtke runs only on the CPU here"):
        global_door.plan(arguments)
