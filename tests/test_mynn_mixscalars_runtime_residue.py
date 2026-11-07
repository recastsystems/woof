"""Keep scalar-mixing arithmetic and whole-driver residuals separate.

The qn tendency solver and DMP exports must match the CPU routines on their
actual consumed arrays. A whole-driver replay has upstream cloud and
PBL-height rounding differences, so it cannot establish a same-input DMP
claim. The historical whole-driver envelope remains unchanged and separate.
Ordinary mixing length now shares initialization's rounded helper; remaining
upstream residuals are not a scalar-specific error allowance.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from conftest import requires_gpu

try:
    import cupy as cp
except Exception:  # pragma: no cover - the marker skips
    cp = None

from woof.config import RunConfig, validate_run_config

QN_SPECIES = ("qnc", "qni", "qnwfa", "qnifa", "qnbca")
SAMPLE = (0, 7, 13, 19, 25, 31, 37, 43)
STEPS = 20

#: Measured 2026-08-26, RTX 3080 (sm_86): worst 21,846 ULP (rqnwfablten,
#: rel ~1.4e-3), 5,461 (rqnifablten), 247 (rqncblten, at 6.7e-31), 0
#: (rqniblten, rqnbcablten).  Bound one power-of-two bin above the
#: measurement; the mechanism carrying it is pinned exactly below.
QN_RESIDUE_ULP_ENVELOPE = 32768


def _run_capture():
    from woof.core.dycore import step
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.moist import init_moist_balanced
    from woof.core.physics import initialize_physics
    import woof.core.mynn_pbl_gpu as gpu_mod
    import woof.core.mynn_pbl_runtime as runtime_mod

    cfg = RunConfig(
        nx=8, ny=6, nz=50, dx=3000.0, dy=3000.0, ztop=16000.0,
        dt=12.0, run_seconds=0.0, time_step_sound=4, moist=True,
        mp_physics=28, sf_sfclay_physics=5, sf_surface_physics=2,
        bl_pbl_physics=5, bldt=0.0, bl_mynn_mixscalars=1)
    validate_run_config(cfg)

    def theta(z):
        z = np.asarray(z, np.float64)
        return np.where(z < 1500.0, 300.0,
                        np.where(z < 1700.0, 300.0 + 0.030 * (z - 1500.0),
                                 306.0 + 0.0045 * (z - 1700.0)))

    def qvapor(z):
        z = np.asarray(z, np.float64)
        return np.where(z < 1500.0, 0.0135,
                        np.maximum(0.0135 - 6.0e-6 * (z - 1500.0), 1.0e-5))

    coord = make_vertical_coord(cfg.nz, stretch=1.6)
    base = make_base_state(coord, theta, p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_moist_balanced(cfg, coord, base, qvapor)
    state.u[...] = cp.float32(7.0)
    state.v[...] = cp.float32(1.5)
    landmask = np.ones((cfg.ny, cfg.nx), np.float64)
    landmask[:, -2:] = 0.0
    tsk = np.full((cfg.ny, cfg.nx), 301.0)
    tsk[landmask == 0.0] = 297.0
    soil_t = np.stack([tsk - 0.5, tsk - 1.0, tsk - 1.5, tsk - 2.0])
    soil_m = np.full((4, cfg.ny, cfg.nx), 0.30)
    soil_m[:, landmask == 0.0] = 1.0
    driver = initialize_physics(
        state, cfg, landmask=landmask, tsk=tsk,
        soil_temperature=soil_t, soil_moisture=soil_m,
        liquid_moisture=soil_m,
        ivgtyp=np.where(landmask, 10, 17), isltyp=np.where(landmask, 6, 14),
        vegfra=55.0, tmn=287.0, swdown=600.0, glw=330.0, pblh=500.0)
    assert driver.scheme_dispatch["bl_pbl_physics"] == "_run_mynn_pbl"

    capture: dict[str, object] = {}
    orig_dmp = gpu_mod.mynn_dmp_mf_cuda
    orig_tend = gpu_mod.mynn_tendencies_default_cuda
    orig_drv = runtime_mod.mynn_bl_driver_cuda

    def dmp_wrap(values, **kw):
        capture["dmp_in"] = {k: cp.asnumpy(cp.asarray(v)).copy()
                             for k, v in values.items()}
        capture["dmp_kw"] = _cpu_replay_kw(kw)
        result = orig_dmp(values, **kw)
        capture["gpu_dmp"] = {
            f.name: cp.asnumpy(getattr(result, f.name))
            for f in dataclasses.fields(result)}
        return result

    def tend_wrap(values, **kw):
        capture["tend_in"] = {k: cp.asnumpy(cp.asarray(v))
                              for k, v in values.items()}
        capture["tend_kw"] = _cpu_replay_kw(kw)
        result = orig_tend(values, **kw)
        capture["tend_out"] = {
            f.name: cp.asnumpy(getattr(result, f.name))
            for f in dataclasses.fields(result)}
        return result

    def drv_wrap(values, **kw):
        capture["values"] = {k: cp.asnumpy(cp.asarray(v))
                             for k, v in values.items()}
        capture["kwargs"] = {k: v for k, v in kw.items() if k != "scratch"}
        out = orig_drv(values, **kw)
        capture["out"] = {k: cp.asnumpy(v) for k, v in out.items()}
        return out

    gpu_mod.mynn_dmp_mf_cuda = dmp_wrap
    gpu_mod.mynn_tendencies_default_cuda = tend_wrap
    runtime_mod.mynn_bl_driver_cuda = drv_wrap
    try:
        for _ in range(STEPS):
            step(state, cfg)
    finally:
        gpu_mod.mynn_dmp_mf_cuda = orig_dmp
        gpu_mod.mynn_tendencies_default_cuda = orig_tend
        runtime_mod.mynn_bl_driver_cuda = orig_drv
    return capture


#: Device-only generation selectors (ace27d973, f72b3f17f).  The CPU
#: reference in woof.core.mynn_pbl is the WRF 4.6.1 generation only, so a
#: replay is valid only when the captured run used that generation.
_DEVICE_GENERATION_SELECTORS = ("bl_mynn_version", "bl_mynn_cloud_tendency_form")


def _cpu_replay_kw(kw):
    """The device unit's keywords as the CPU reference takes them."""
    replay = {k: v for k, v in kw.items() if k != "scratch"}
    # export_sink is the device unit's GSD-4.1 diagnostic OUTPUT sink
    # (6d79b6e9c5), not an input; the WRF 4.6.1 DMP the CPU replays has
    # none, and the generation selector below holds the replay to it.
    assert replay.pop("export_sink", None) is None
    for selector in _DEVICE_GENERATION_SELECTORS:
        value = replay.pop(selector, "wrf_461")
        assert value == "wrf_461", (selector, value)
    return replay


@pytest.fixture(scope="module")
def runtime_capture():
    if cp is None:
        pytest.skip("no CUDA GPU / cupy")
    return _run_capture()


def _ulp(a, b):
    from woof.core.fp32_ulp import monotone_fp32_key
    return np.abs(monotone_fp32_key(np.asarray(a, np.float32))
                  - monotone_fp32_key(np.asarray(b, np.float32)))


@requires_gpu
def test_qn_solve_is_bitwise_on_the_gpu_solve_inputs(runtime_capture):
    """Fact (2): the CPU qn solves on the device unit's exact captured
    inputs reproduce every device tendency output at ULP 0 -- the
    mixscalars solve consumption itself is bitwise."""
    from woof.core.mynn_pbl import mynn_tendencies_default

    capture = runtime_capture
    rows = np.asarray(SAMPLE, dtype=np.intp)
    ncol = capture["tend_in"]["dz"].shape[0]
    tin = {k: (v[rows] if getattr(v, "ndim", 0) >= 1
               and v.shape[0] == ncol else v)
           for k, v in capture["tend_in"].items()}
    cpu_out = mynn_tendencies_default(tin, **capture["tend_kw"])
    for name in sorted(cpu_out):
        if name not in capture["tend_out"]:
            continue
        gpu = capture["tend_out"][name]
        gpu = gpu[rows] if (getattr(gpu, "ndim", 0) >= 1
                            and gpu.shape[0] == ncol) else gpu
        worst = int(_ulp(gpu, cpu_out[name]).max(initial=0))
        assert worst == 0, (
            f"{name}: {worst} ULP on identical inputs -- the tendencies "
            "unit itself diverged; the Stage-1 attribution is void")


@requires_gpu
def test_qn_flux_exports_are_bitwise_and_residue_is_bounded(runtime_capture):
    """Facts (1) and (3): the sibling DMP ``s_awqn*`` exports are
    bit-equal to the CPU replay, and the end-to-end qn residue those two
    exactness facts leave to the pre-discipline kernels stays inside the
    measured envelope."""
    from woof.core.mynn_pbl import mynn_bl_driver
    import woof.core.mynn_pbl as cpu_mod

    capture = runtime_capture
    rows = np.asarray(SAMPLE, dtype=np.intp)
    ncol = capture["gpu_dmp"]["ktop"].size
    values = {k: (v[rows] if getattr(v, "ndim", 0) >= 1
                  and v.shape[0] == ncol else v)
              for k, v in capture["values"].items()}
    kwargs = dict(capture["kwargs"])
    kwargs.pop("flag_qs", None)
    base_kw = dict(initflag=kwargs.pop("initflag"),
                   delt=kwargs.pop("delt"),
                   flag_qs=capture["kwargs"].get("flag_qs", False),
                   **kwargs)

    cpu0 = mynn_bl_driver(dict(values), **base_kw)
    # Whole-driver replay can change cloud/PBL-height inputs before DMP.
    # Isolate the DMP arithmetic on its actual consumed arrays instead.
    dmp_inputs = {k: (v[rows] if getattr(v, "ndim", 0) >= 1
                      and v.shape[0] == ncol else v)
                  for k, v in capture["dmp_in"].items()}
    cpu_plumes = cpu_mod.mynn_dmp_mf(dmp_inputs, **capture["dmp_kw"])

    # Fact (1): every s_awqn* interface bit-equal on the sampled columns.
    for name in QN_SPECIES:
        key = f"s_aw{name}"
        gpu = capture["gpu_dmp"][key][rows]
        worst = int(_ulp(gpu, cpu_plumes[key]).max(initial=0))
        assert worst == 0, (
            f"{key}: {worst} ULP -- the sibling DMP flux chain diverged; "
            "the Stage-1 attribution is void")

    # Fact (3): the bounded envelope, with its two structural zeros kept
    # exact -- a species whose column is zero must stay exactly zero.
    worst_all = {}
    for name in QN_SPECIES:
        key = f"r{name}blten"
        worst_all[key] = int(
            _ulp(capture["out"][key][rows], cpu0[key]).max(initial=0))
    assert worst_all["rqniblten"] == 0, worst_all
    assert worst_all["rqnbcablten"] == 0, worst_all
    worst = max(worst_all.values())
    assert worst <= QN_RESIDUE_ULP_ENVELOPE, (
        f"qn runtime residue {worst_all} exceeded the measured envelope "
        f"{QN_RESIDUE_ULP_ENVELOPE}; do not widen this bound without "
        "re-running the Stage-1 bisect and re-attributing the mechanism")
