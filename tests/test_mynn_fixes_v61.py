"""``mynn_fixes_v61``: upstream MYNN v6.1 bug fixes on the WRF 4.6.1 MYNN route.

THE BREAKAGES THESE PREVENT
---------------------------
Each fix (woof/core/mynn_pbl_v61.py) is a marked edit of the frozen WRF
4.6.1 unit or of the SFCLAY_mynn first-step block.  One test per fix shows
the defect on the default route and its absence under the option:

* A1 dew/frost: the default deletes a downward surface vapour flux; the
  option limits it (NCAR/MYNN-EDMF f3c9d9e).
* A2 surface density: the default adds p608*qv to a temperature; the option
  multiplies by (1 + p608*qv) (eb5029c).
* A5 surface theta: the default divides the ground temperature by the lowest
  level's Exner function; the option by the ground's (27da0f0).
* A6 PBL-height search: the default starts at kts+1 and stops in a
  superadiabatic layer below the theta-v minimum; the option starts above
  it (27da0f0).
* S1 surface-layer cold start: the default resets UST/MOL/QSFC on a cycled
  start; the option keeps them (NCAR/MYNN-SFC f0c1839).

Plus the guards of the option itself: the variant source is the frozen
unit plus exactly the marked edits, and option-off kernels are the frozen
unit's.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu

from woof.core.mynn_pbl_v61 import (
    MYNN_V61_EDITS, MYNN_V61_FUNCS, MYNN_V61_NOTICE, MynnV61SourceError,
    mynn_sfclay_cold_start, mynn_v61_kernel_name, mynn_v61_source,
)

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "woof" / "core" / "kernels" / "mynn_pbl.cu"


def _frozen_text() -> str:
    return FROZEN.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# The variant is the frozen unit plus the marked edits, nothing else.
# --------------------------------------------------------------------------

def test_variant_is_the_frozen_unit_plus_exactly_the_marked_edits():
    frozen = _frozen_text()
    pins = json.loads((ROOT / "tests" / "data" / "kernel_source_pins.json")
                      .read_text(encoding="utf-8"))
    pinned = next(value for key, value in _walk(pins) if key == "mynn_pbl.cu")
    assert hashlib.sha256(FROZEN.read_bytes()).hexdigest() == pinned
    variant = mynn_v61_source(frozen)
    assert variant.startswith(MYNN_V61_NOTICE)
    body = variant[len(MYNN_V61_NOTICE):]
    for ident, old, new in MYNN_V61_EDITS:
        assert f"MYNN-V61 {ident[:2]}" in new, ident
        assert body.count(new) == 1, ident
    # Undo every edit: the frozen bytes come back exactly.
    for _, old, new in reversed(MYNN_V61_EDITS):
        body = body.replace(new, old, 1)
    assert body == frozen
    assert "Apache License, Version 2.0" in MYNN_V61_NOTICE
    assert "not bitwise to" in MYNN_V61_NOTICE


def _walk(node):
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, str):
                yield key, value
            else:
                yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


def test_variant_refuses_a_moved_anchor():
    frozen = _frozen_text()
    _, old, _ = MYNN_V61_EDITS[0]
    with pytest.raises(MynnV61SourceError, match="A6a"):
        mynn_v61_source(frozen.replace(old, old.replace("9.0e9f", "9.0e9f ")))


def test_option_off_loads_only_the_frozen_unit():
    from woof.core import mynn_pbl_gpu

    for func in MYNN_V61_FUNCS:
        assert mynn_v61_kernel_name(func, False) == "mynn_pbl"
        assert mynn_v61_kernel_name(func, True) == "mynn_pbl_v61"
    assert mynn_v61_kernel_name("mynn_dmp_mf_columns", True) == "mynn_pbl"
    with pytest.raises(ValueError, match="wrf_461"):
        mynn_pbl_gpu.mynn_pbl_kernel("mynn_tendencies_columns", "gsd_41",
                                     mynn_fixes_v61=True)


def test_variant_compiles_with_the_frozen_units_options():
    from woof.core import kernels

    assert kernels.module_options("mynn_pbl_v61") == \
        kernels.module_options("mynn_pbl")
    assert ("mynn_pbl_v61" in kernels._NO_FMAD_MODULES) == \
        ("mynn_pbl" in kernels._NO_FMAD_MODULES)
    assert ("mynn_pbl_v61" in kernels._NO_FTZ_MODULES) == \
        ("mynn_pbl" in kernels._NO_FTZ_MODULES)
    source = kernels.module_source("mynn_pbl_v61")
    assert source.count("MYNN-V61 A") >= len(MYNN_V61_EDITS)
    assert kernels.module_source("mynn_pbl").count("MYNN-V61") == 0


# --------------------------------------------------------------------------
# S1: surface-layer cold start under cycling.
# --------------------------------------------------------------------------

def test_s1_cycled_start_keeps_the_surface_layer_state():
    # WRF 4.6.1: every step-1 call resets UST/MOL/QSFC, cycled or not.
    assert mynn_sfclay_cold_start(1, cycling=False, fixes_v61=False)
    assert mynn_sfclay_cold_start(1, cycling=True, fixes_v61=False)
    # v6.1 fix: .not.(restart .or. cycling); a restart never sees step 1.
    assert mynn_sfclay_cold_start(1, cycling=False, fixes_v61=True)
    assert not mynn_sfclay_cold_start(1, cycling=True, fixes_v61=True)
    for cycling in (False, True):
        for fixes in (False, True):
            assert not mynn_sfclay_cold_start(2, cycling=cycling,
                                              fixes_v61=fixes)


# --------------------------------------------------------------------------
# Admission.
# --------------------------------------------------------------------------

def _mynn_cfg(**overrides):
    from woof.config import RunConfig

    values = dict(nx=24, ny=24, nz=8, dx=3000.0, dy=3000.0, ztop=10000.0,
                  dt=6.0, run_seconds=60.0, bl_pbl_physics=5,
                  sf_sfclay_physics=5)
    values.update(overrides)
    return RunConfig(**values)


def test_option_is_correct_mode_my1_and_inert_off_its_route():
    """Since 3.0.3 the key is correct-mode finding MY1: on by default (off
    in exact mode and in a strict process), read only by the wrf_461 MYNN
    PBL, and inert elsewhere, where it used to be refused (a default-on key
    must not refuse every non-MYNN run).  Breakage prevented: a default
    MYNN run that deletes dew and frost, and an option that refuses runs it
    never touches."""
    from woof import correct_mode
    from woof.config import ConfigurationRefused, RunConfig, validate_run_config
    from woof.core.mynn_pbl_v61 import pbl_fixes_active, sfclay_fixes_active

    field = RunConfig.__dataclass_fields__["mynn_fixes_v61"]
    assert field.kw_only and field.default is correct_mode.CORRECT_KEYS_DEFAULT
    assert correct_mode.FINDING_BY_KEY["mynn_fixes_v61"] == "MY1"
    validate_run_config(_mynn_cfg(mynn_fixes_v61=True))
    with pytest.raises((ConfigurationRefused, ValueError), match="true or false|boolean"):
        validate_run_config(_mynn_cfg(mynn_fixes_v61=1))
    for inert in (dict(bl_mynn_version="gsd_41"),
                  dict(bl_pbl_physics=1, sf_sfclay_physics=1)):
        cfg = _mynn_cfg(mynn_fixes_v61=True, **inert)
        validate_run_config(cfg)
        assert not pbl_fixes_active(cfg) and not sfclay_fixes_active(cfg)
    other_sfc = _mynn_cfg(mynn_fixes_v61=True, mynn_sfclay_variant="gsl_wrf39")
    validate_run_config(other_sfc)
    assert pbl_fixes_active(other_sfc) and not sfclay_fixes_active(other_sfc)
    assert not pbl_fixes_active(_mynn_cfg(mynn_fixes_v61=False))


# --------------------------------------------------------------------------
# Kernel fixes, each against the frozen unit on the same inputs.
# --------------------------------------------------------------------------

def _tendency_case(case: str):
    import cupy as cp
    from test_mynn_pbl import _tendencies_mf_inputs, _tendencies_mf_oracle

    _, fields = _tendencies_mf_oracle()
    inputs, edmf_mom = _tendencies_mf_inputs(fields, case)
    return {name: np.array(value, dtype=np.float32, copy=True)
            for name, value in inputs.items()}, edmf_mom


def _tendencies(inputs, edmf_mom, fixes):
    import cupy as cp
    from woof.core.mynn_pbl_gpu import mynn_tendencies_default_cuda

    out = mynn_tendencies_default_cuda(
        {name: cp.asarray(value) for name, value in inputs.items()},
        bl_mynn_edmf_mom=edmf_mom, mynn_fixes_v61=fixes)
    cp.cuda.get_current_stream().synchronize()
    return {name: cp.asnumpy(getattr(out, name)).copy()
            for name in ("du", "dv", "dth", "dqv", "dqc")}


@pytest.mark.gpu
@requires_gpu
def test_a1_dew_and_frost_are_limited_not_deleted():
    inputs, edmf_mom = _tendency_case("land_cumulus")
    dew = dict(inputs)
    dew["flqv"] = np.full_like(inputs["flqv"], -2.0e-5)    # a dew flux
    none = dict(inputs)
    none["flqv"] = np.zeros_like(inputs["flqv"])
    # Frozen unit: the downward flux is deleted -- dew and no flux agree.
    np.testing.assert_array_equal(_tendencies(dew, edmf_mom, False)["dqv"],
                                  _tendencies(none, edmf_mom, False)["dqv"])
    # Fixed: dew takes vapour from the lowest layer.
    fixed_dew = _tendencies(dew, edmf_mom, True)["dqv"]
    fixed_none = _tendencies(none, edmf_mom, True)["dqv"]
    assert fixed_dew[0, 0] < fixed_none[0, 0]
    # ... and the limiter still stops it short of emptying the layer.
    huge = dict(inputs)
    huge["flqv"] = np.full_like(inputs["flqv"], -1.0e3)
    sqv0 = np.float32(inputs["sqv"][0, 0])
    delt = np.float32(inputs["delt"][0])
    fixed_huge = _tendencies(huge, edmf_mom, True)["dqv"]
    assert np.isfinite(fixed_huge).all()
    assert sqv0 + fixed_huge[0, 0] * delt > -1.0e-6


@pytest.mark.gpu
@requires_gpu
def test_a2_surface_density_has_the_virtual_temperature_factor():
    inputs, edmf_mom = _tendency_case("land_cumulus")
    inputs["flqv"] = np.abs(inputs["flqv"])      # keep A1 out of the way
    dry = dict(inputs)
    dry["qv"] = np.zeros_like(inputs["qv"])
    # With qv = 0, r_d*(tk + 0) and (r_d*tk)*(1 + 0) are the same word.
    for name, value in _tendencies(dry, edmf_mom, False).items():
        np.testing.assert_array_equal(
            value, _tendencies(dry, edmf_mom, True)[name], err_msg=name)
    # With vapour the frozen density is the wrong one: the surface drag
    # on the lowest level moves.
    moist = dict(inputs)
    moist["qv"] = np.full_like(inputs["qv"], 0.012)
    frozen = _tendencies(moist, edmf_mom, False)
    fixed = _tendencies(moist, edmf_mom, True)
    assert frozen["du"][0, 0] != fixed["du"][0, 0]
    tk = np.float64(inputs["tk"][0, 0])
    p608 = 461.6 / 287.0 - 1.0
    ratio = (tk + p608 * 0.012) / (tk * (1.0 + p608 * 0.012))
    # rho_fixed/rho_frozen: the old density was ~0.6 % too large.
    assert 0.99 < ratio < 1.0


@pytest.mark.gpu
@requires_gpu
def test_a5_surface_theta_uses_the_grounds_exner_function():
    import cupy as cp
    from woof.core.mynn_pbl_gpu import _driver_surface_cuda, _scratch_for
    from woof.core.noahmp_libm import powf

    ncol, nz = 3, 4
    exner = np.full((ncol, nz), 0.99, dtype=np.float32)
    layers = {"rho": cp.asarray(np.full((ncol, nz), 1.1, np.float32)),
              "exner": cp.asarray(exner),
              "dz": cp.asarray(np.full((ncol, nz), 40.0, np.float32))}
    qv1 = cp.asarray(np.full((ncol, nz), 0.008, np.float32), order="F")
    ps = np.asarray([101325.0, 85000.0, 70000.0], dtype=np.float32)
    ts = np.asarray([300.0, 290.0, 280.0], dtype=np.float32)
    scalars = {"ust": cp.full(ncol, 0.3, dtype=np.float32),
               "hfx": cp.full(ncol, 100.0, dtype=np.float32),
               "qfx": cp.full(ncol, 5.0e-5, dtype=np.float32),
               "ts": cp.asarray(ts), "ps": cp.asarray(ps),
               "rmol": cp.zeros(ncol, dtype=np.float32)}
    work = _scratch_for(None, ncol, nz)
    frozen = cp.asnumpy(_driver_surface_cuda(
        layers, qv1, scalars, ncol, nz, work)["th_sfc"]).copy()
    fixed = cp.asnumpy(_driver_surface_cuda(
        layers, qv1, scalars, ncol, nz, work,
        mynn_fixes_v61=True)["th_sfc"]).copy()
    np.testing.assert_array_equal(frozen, ts / np.float32(0.99))
    rcp = np.float32(np.float32(287.0) / np.float32(1004.5))
    want = np.asarray([ts[i] / powf(ps[i] / np.float32(100000.0), rcp)
                       for i in range(ncol)], dtype=np.float32)
    np.testing.assert_array_equal(fixed, want)


@pytest.mark.gpu
@requires_gpu
def test_a6_pbl_height_search_starts_above_the_theta_v_minimum():
    import cupy as cp
    from woof.core.mynn_pbl_gpu import mynn_pblh_scale_columns_cuda

    nz = 12
    dz = np.full((1, nz), 50.0, dtype=np.float32)
    zw = np.concatenate(([0.0], np.cumsum(dz[0]))).astype(np.float32)[None]
    # Superadiabatic: theta-v at kts+1 is 2 K above the minimum at level 3
    # (and 1 K above the lowest level), then rises again above the minimum.
    thetav = np.asarray([[302.0, 303.0, 301.5, 301.0, 301.2, 301.6, 302.4,
                          303.0, 304.0, 305.0, 306.0, 307.0]], np.float32)
    qke = np.zeros((1, nz), dtype=np.float32)          # theta-v definition only
    args = (cp.asarray(thetav), cp.asarray(qke), cp.asarray(zw),
            cp.asarray(dz), cp.asarray([1.0], dtype=np.float32),
            cp.asarray([3000.0], dtype=np.float32))
    frozen = float(cp.asnumpy(mynn_pblh_scale_columns_cuda(*args).zi)[0])
    fixed = float(cp.asnumpy(
        mynn_pblh_scale_columns_cuda(*args, mynn_fixes_v61=True).zi)[0])
    # Frozen: thetav[1] = 303 >= 301 + 1.25 ends the search in the first
    # layer (12.5 m), below the theta-v minimum.
    assert 0.0 < frozen <= zw[0, 1]
    # Fixed: the search starts above kthv = 3 and finds the 302.25 K crossing
    # between levels 5 and 6.
    assert zw[0, 5] < fixed <= zw[0, 6]


# --------------------------------------------------------------------------
# The whole driver, option on and off.
# --------------------------------------------------------------------------

@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("step", (1, 2))
def test_driver_option_off_is_the_frozen_driver_and_on_runs(step):
    import cupy as cp
    from woof.core.mynn_pbl_gpu import mynn_bl_driver_cuda
    from test_mynn_pbl import _driver_step

    _, values, initflag, delt = _driver_step(step)
    values = {name: np.asarray(value)[:4].copy() for name, value in values.items()}

    def run(**kwargs):
        out = mynn_bl_driver_cuda(
            {name: cp.asarray(np.ascontiguousarray(v)) for name, v in values.items()},
            initflag=initflag, delt=delt, flag_qs=True, **kwargs)
        cp.cuda.get_current_stream().synchronize()
        return {name: cp.asnumpy(v).copy() for name, v in out.items()
                if hasattr(v, "dtype")}

    plain = run()
    off = run(mynn_fixes_v61=False)
    on = run(mynn_fixes_v61=True)
    for name, value in plain.items():
        np.testing.assert_array_equal(off[name], value, err_msg=name)
        assert np.isfinite(on[name].astype(np.float64)).all(), name
    assert any(not np.array_equal(on[name], plain[name]) for name in plain)
