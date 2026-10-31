"""The both-routes gate leg (tools/both_routes_gate.py, tools/battery/both_routes.json), on the CPU.

Breakage these prevent: the GPU gate leg discovering a renamed or removed flag only after it has leased a card,
or a capability family quietly losing its run on one physics route (every hex capability works on every route).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

TREE = Path(__file__).resolve().parents[1]
SPEC = json.loads((TREE / "tools" / "battery" / "both_routes.json").read_text(encoding="utf-8"))


def _gate():
    spec = importlib.util.spec_from_file_location("both_routes_gate", TREE / "tools" / "both_routes_gate.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["both_routes_gate"] = module
    spec.loader.exec_module(module)
    return module


def _options(**overrides):
    values = dict(cycle=None, storm=None, storm_cycle=None, best_track=Path("/data/ibtracs.csv"),
                  staged=Path("/data/gfs"), storm_staged=Path("/data/gfs-storm"), geog=Path("/data/geog"))
    values.update(overrides)
    return SimpleNamespace(**values)


class _FakeContext:
    """The gate's Context with the base leg's plan replaced by a representative forecast argv."""

    def __new__(cls, gate, route, work_root):
        ctx = gate.Context(SPEC, _options(), route, work_root)
        forecast = ["forecast", "--mesh", "hex-x1.40962-native", "--grid", "/m/x1.40962.grid.nc", "--static",
                    "/m/x1.40962.native.static.nc", "--init", "/w/init/init.nc", "--start-time",
                    "2023-03-31_12:00:00", "--init-source", "gfs 2023-03-31T12", "--hours", "2",
                    "--history-every-minutes", "60", "--out", "/w/forecast", "--scratch", "/w/scratch",
                    "--case-label", "global-120km-2023033112", "--no-products", *SPEC["routes"][route]["forecast"]]
        plan = {"commands": {"forecast": forecast}, "start_time": "2023-03-31_12:00:00",
                "paths": {"init": "/w/init/init.nc", "grid": "/m/x1.40962.grid.nc",
                          "static": "/m/x1.40962.native.static.nc"}}
        ctx.base_plan = lambda: plan
        return ctx


def test_every_family_runs_on_both_routes():
    assert set(SPEC["routes"]) == {"arwen", "mpas"}
    ids = [leg["id"] for leg in SPEC["legs"]]
    assert len(ids) == len(set(ids))
    assert ids[0] == "global"
    for leg in SPEC["legs"]:
        assert leg["steps"] and leg["checks"], leg["id"]
        assert leg["family"], leg["id"]
        if any(step.get("gpu") for step in leg["steps"]):
            # Pool mode: every GPU leg declares the VRAM the pool admits it by, and never asks for a whole card.
            assert leg["vram_mib"] > 0 and leg["cards"] >= 1, leg["id"]


def test_the_mpas_route_is_named_at_every_door():
    mpas = SPEC["routes"]["mpas"]
    assert mpas["forecast"] == ["--physics-suite=convection_permitting"]
    assert mpas["door"] == ["--forecast-arg=--physics-suite=convection_permitting"]
    assert mpas["moving"] == ["--physics-arg=--physics-suite=convection_permitting"]
    # MPAS physics is the hex default (0.4.6), so the ArWen route is named too, at every door.
    assert SPEC["routes"]["arwen"] == {"door": ["--forecast-arg=--physics-suite=arwen_mpas_column"],
                                       "forecast": ["--physics-suite=arwen_mpas_column"],
                                       "moving": ["--physics-arg=--physics-suite=arwen_mpas_column"]}


@pytest.mark.parametrize("route", ["arwen", "mpas"])
@pytest.mark.parametrize("leg_id", [leg["id"] for leg in SPEC["legs"]])
def test_every_command_parses_with_the_real_parser(route, leg_id, tmp_path):
    from woof.hex.cli import build_parser

    gate = _gate()
    ctx = _FakeContext(gate, route, tmp_path)
    leg = next(leg for leg in SPEC["legs"] if leg["id"] == leg_id)
    work = ctx.leg_dir(leg_id)
    named = arwen_named = False
    for step in leg["steps"]:
        if "builtin" in step:
            assert step["builtin"] in gate.BUILTINS
            continue
        argv = [token if token != "{checkpoint:straight}" else "/w/straight/checkpoint.f001.pkl"
                for token in step["argv"]]
        expanded = ctx.expand(argv, work)
        assert not [t for t in expanded if t.startswith("{") and not t.startswith("{\"")], expanded
        build_parser().parse_args(expanded)  # SystemExit (a red) on any flag the doors no longer take
        named = named or any("convection_permitting" in t for t in expanded)
        # MPAS physics is the hex default (0.4.6): the ArWen leg names its suite, or it would run MPAS.
        arwen_named = arwen_named or any("arwen_mpas_column" in t for t in expanded)
        if route == "arwen":
            assert not any("convection_permitting" in t for t in expanded if t.startswith("--physics-suite")
                           or t.startswith("--forecast-arg=--physics-suite")), (leg_id, step["name"])
    if leg_id != "keep-as-init":  # exports the global leg's frames; the route was named when they were made
        assert named == (route == "mpas"), leg_id
        assert arwen_named == (route == "arwen"), leg_id


def test_the_base_forecast_drops_only_what_each_leg_sets(tmp_path):
    gate = _gate()
    argv = _FakeContext(gate, "mpas", tmp_path).base_forecast()
    for flag in ("--hours", "--out", "--scratch", "--case-label"):
        assert flag not in argv
    assert argv[0] == "forecast" and "--init" in argv and "--physics-suite=convection_permitting" in argv


def test_a_gpu_step_leases_by_vram_and_never_a_whole_card(monkeypatch):
    gate = _gate()
    monkeypatch.setenv(gate.LEASE_ENV, "gpu-lease")
    one = gate.lease_prefix({"vram_mib": 14000, "cards": 1}, "lane")
    two = gate.lease_prefix({"vram_mib": 8000, "cards": 2}, "lane")
    assert one == ["gpu-lease", "--vram-mib", "14000", "lane", "--"]
    assert two == ["gpu-lease", "--vram-mib", "8000", "--cards", "2", "lane", "--"]
    monkeypatch.delenv(gate.LEASE_ENV)
    monkeypatch.setenv("HEX_GPU_QUEUE", "queue")
    assert "--proof" not in gate.lease_prefix({"vram_mib": 1, "cards": 1, "minutes": 5}, "lane")


def test_same_words_counts_every_differing_value(tmp_path):
    netcdf = pytest.importorskip("netCDF4")
    import numpy as np

    gate = _gate()
    for name, bump in (("a.nc", 0.0), ("b.nc", 1.0)):
        with netcdf.Dataset(tmp_path / name, "w") as d:
            d.createDimension("n", 4)
            v = d.createVariable("theta", "f4", ("n",))
            v[:] = np.array([1, 2, 3, 4], "f4") + np.array([0, 0, bump, 0], "f4")
            d.setncattr("file_id", name)
    assert gate.compare_words(tmp_path / "a.nc", tmp_path / "a.nc")[0] == 0
    differing, total, names = gate.compare_words(tmp_path / "a.nc", tmp_path / "b.nc")
    assert (differing, total, names) == (1, 4, ["theta"])


def _frames(folder: Path, prefix: str, walls: list[float]) -> None:
    import os

    folder.mkdir(parents=True)
    for hour, wall in enumerate(walls):
        path = folder / f"{prefix}.2023-03-31_{12 + hour:02d}.00.00.nc"
        path.write_bytes(b"")
        os.utime(path, (1_000_000.0 + wall, 1_000_000.0 + wall))


@pytest.mark.parametrize("route", ["arwen", "mpas"])
def test_the_lts_speed_leg_fails_when_lts_is_not_faster_by_its_margin(route, tmp_path):
    """Breakage: LTS slower than the fixed step on the graded zoom globe (HEX-LTS-REGRESSION-2026-10-27) with no red."""
    gate = _gate()
    leg = next(leg for leg in SPEC["legs"] if leg["id"] == "lts-speed")
    check = next(c for c in leg["checks"] if c["kind"] == "lts_speed")
    assert check["min_speedup"][route] > 1.0 and route in check["recorded"]
    ctx = _FakeContext(gate, route, tmp_path)
    prefix = "history" if route == "mpas" else "cuda-history"
    # f000 at 0 s, then 80 s per forecast hour fixed; start-up before f001 is not counted
    _frames(tmp_path / "fixed" / "forecast", prefix, [0.0, 200.0, 280.0, 360.0])
    _frames(tmp_path / "fast" / "forecast", prefix, [0.0, 150.0, 215.0, 280.0])   # 65 s/h: 1.23x
    _frames(tmp_path / "slow" / "forecast", prefix, [0.0, 150.0, 250.0, 350.0])   # 100 s/h: 0.8x
    fast = gate.run_check(ctx, {**check, "lts": "fast/forecast"}, tmp_path, {"steps": []})
    slow = gate.run_check(ctx, {**check, "lts": "slow/forecast"}, tmp_path, {"steps": []})
    assert fast["fixed"]["seconds_per_forecast_hour"] == 80.0 and fast["lts"]["seconds_per_forecast_hour"] == 65.0
    assert fast["ok"] and fast["speedup"] == round(80 / 65, 3)
    assert not slow["ok"] and slow["speedup"] == 0.8
    # no warm interval (a run that died after f001) is a red, not a pass
    missing = gate.run_check(ctx, {**check, "lts": "none/forecast"}, tmp_path, {"steps": []})
    assert not missing["ok"] and missing["lts"] is None
