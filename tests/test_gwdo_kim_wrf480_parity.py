"""gwd_opt = 1 as WRF v4.8.0 runs it, word for word against the native referee.

The fixture (tests/data/oracles/gwdo_kim480/, PROVENANCE.md) is SUBROUTINE
gwdo of WRF v4.8.0's ``phys/module_bl_gwdo.F`` calling ``bl_gwdo_run`` of
``phys/physics_mmm/bl_gwdo.F90`` (MMM-physics 550b5b49), run as the object
files of the native WRF v4.8.0 strict referee build itself
(WRF-4.8.0-serial-pristine, commit 06d4240a; GNU Fortran 13.3.0, glibc
2.39, ``-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize``),
linked into ``tools/gwdo_kim_wrf480_oracle/run_gwdo_kim480.F90``.  Sets:
the v4.6.1 oracle's 149 regime columns on 50 and 90 levels and on a
60-level grid topped at 3 hPa, with ELVMAX added, each at six (dx, dt)
cases with WRF v4.8.0's defaults (gwd_dx_factor 2, gwd_if_nonhyd true)
and with (1, false); and real columns, every 97th column (plus up to 400 of the
columns WRF dragged) of four WOOF-ArWen HRRR-recipe history frames (3 km,
244 x 234 x 50) with the HRRR static's orographic statistics and ELVMAX
from the 30" terrain, at three cases.
``launch_gwdo(gwdo_version="wrf_480")`` must land on every output word.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu

from woof.verify import gwdo_kim480_oracle as go

SYNTHETIC = ("L50", "L60top3", "L90")
REAL = ("real-2023-12-10_18_00_00", "real-2023-12-11_06_00_00",
        "real-2024-04-25_18_00_00", "real-2024-04-26_00_00_00")
SETS = SYNTHETIC + REAL
KERNELS = Path(__file__).resolve().parents[1] / "woof" / "core" / "kernels"


def test_fixture_is_the_documented_oracle():
    for name in SETS:
        inputs, wrf = go.load_fixture(name)
        nz, ncol = inputs["u"].shape
        ncase = len(inputs["dx"])
        assert inputs["p_interface"].shape == (nz + 1, ncol)
        assert inputs["elvmax"].shape == (ncol,)
        for field in go.LEVEL_FIELDS:
            assert wrf[field].shape == (ncase, nz, ncol)
            assert wrf[field].dtype == np.float32
            assert np.all(np.isfinite(wrf[field]))
        for field in go.SURFACE_FIELDS:
            assert wrf[field].shape == (ncase, ncol)
        # both switches of each new key are exercised
        assert set(inputs["if_nonhyd"].tolist()) == {0, 1}
        assert {2.0, 1.0} <= set(inputs["dx_factor"].tolist())


def test_the_drag_is_not_vacuous():
    """Each set drags and leaves some columns undragged; the regime sets
    and the real sets together reach flow blocking."""
    blocking = {}
    for name in SETS:
        inputs, wrf = go.load_fixture(name)
        moved = (wrf["rublten"].view(np.uint32)
                 != inputs["rublten0"][None].view(np.uint32)).any(axis=1)
        active = moved.sum(axis=1)
        assert active.min() > 0, (name, active)
        assert (active < moved.shape[1]).all(), (name, active)
        blocking[name] = max(go.blocking_columns(inputs, wrf))
    assert all(blocking[name] > 0 for name in SYNTHETIC), blocking
    assert sum(blocking[name] for name in REAL) > 0, blocking
    # ELVMAX <= 50 m turns a dragging column off (bl_gwdo.F90:416)
    inputs, _ = go.load_fixture("L50")
    assert (inputs["elvmax"] <= 50.0).any() and (inputs["var"] > 0).any()
    # the 3 hPa top reaches the mesospheric limiter (p <= 7.5 hPa)
    inputs, _ = go.load_fixture("L60top3")
    assert (inputs["p"] <= 750.0).any()


def test_the_frozen_kernel_is_untouched():
    """gwdo_version = "wrf_471" compiles exactly the unit it was proven as."""
    from woof.core import kernel_frame_recordings as recordings
    from woof.core import terrain_drag

    frozen = terrain_drag.module_source()
    assert hashlib.sha256(frozen.encode()).hexdigest() == (
        recordings.TERRAIN_DRAG_COMPOSED_SOURCE_SHA256)
    new = terrain_drag.gwdo480_module_source()
    tail = (KERNELS / "terrain_drag_gwdo480.cu").read_text(encoding="utf-8")
    assert new == frozen + tail
    assert "gwdo480_column" not in frozen


def test_launcher_defaults_and_refusals():
    from woof.core import terrain_drag as td

    assert td.DEFAULT_GWDO_VERSION == "wrf_480"
    assert (td.DEFAULT_GWD_DX_FACTOR, td.DEFAULT_GWD_IF_NONHYD) == (2.0, True)
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="gwd_dx_factor"):
            td.gwdo_options(dx_factor=bad)
    with pytest.raises(ValueError, match="gwdo_version"):
        td.gwdo_options(gwdo_version="wrf_470")
    with pytest.raises(ValueError, match="ELVMAX"):
        td.launch_gwdo({}, np.zeros((4, 1, 1), np.float32), None, var=None,
                       con=None, oa=None, ol=None, sina=None, cosa=None,
                       dx=3000.0, dt=20.0, gwdo_version="wrf_480")


def test_the_static_set_names_elvmax_only_under_480():
    from woof.static.orographic import (GWDO_FIELDS, drag_static_fields,
                                        elvmax_missing_note,
                                        required_static_fields)

    assert drag_static_fields(0, 1) == GWDO_FIELDS + ("ELVMAX",)
    assert drag_static_fields(0, 1, "wrf_471") == GWDO_FIELDS
    # the GEOGRID-table request never asks for it
    assert "ELVMAX" not in required_static_fields(0, 1)
    assert "ELVMAX" not in drag_static_fields(0, 3)
    assert "wrf_471" in elvmax_missing_note(["ELVMAX"])
    assert elvmax_missing_note(["VAR"]) == ""


def test_run_config_carries_the_keys():
    from woof.config import ConfigurationRefused, RunConfig, validate_run_config

    base = dict(nx=10, ny=10, nz=10, dx=3000.0, dy=3000.0, ztop=20000.0,
                dt=12.0, run_seconds=3600.0, bl_pbl_physics=1, gwd_opt=1)
    cfg = RunConfig(**base)
    assert (cfg.gwdo_version, cfg.gwd_dx_factor, cfg.gwd_if_nonhyd) == (
        "wrf_480", 2.0, True)
    with pytest.raises(ConfigurationRefused, match="gwd_dx_factor"):
        validate_run_config(RunConfig(**base, gwd_dx_factor=0.0))
    with pytest.raises(ConfigurationRefused, match="gwdo_version"):
        validate_run_config(RunConfig(**base, gwdo_version="4.8"))


@requires_gpu
@pytest.mark.parametrize("name", SETS)
def test_launch_gwdo_480_is_bit_identical_to_the_referee(name):
    inputs, wrf = go.load_fixture(name)
    table = go.compare(go.run_gpu(inputs), wrf, inputs["regime"])
    missed = {k: (r["not_bit_identical"], r["max_ulp"])
              for k, r in table.items() if r["not_bit_identical"]}
    assert not missed, missed
