"""Correct-mode ML1: the sign of MYNN's BouLac downward length in a layer of
constant theta (NCAR/MYNN-EDMF 399f50f, PR 94, issue 93; in v6.1.1 2b2da61).

A parcel from level iz that sinks a distance s into the layer between levels
izz and izz-1, where theta has gradient bbb, has used the energy
``zdo(s) = zdo_sup + beta*(theta(iz)-theta(izz))*s + beta*bbb*s**2/2``.  The
quadratic branch (bbb /= 0) solves ``zdo(s) = qtke`` correctly; its bbb -> 0
limit is ``(qtke - zdo_sup)/(beta*(theta(iz)-theta(izz)))``.  WRF v4.6.1's
linear branch divides by ``beta*(theta(izz)-theta(iz))``: the last step is
negative and the downward length short by twice it.

Breakage prevented: a default (correct-mode) run whose MYNN mixing length
stays short wherever a parcel stops in a layer of exactly constant theta (79
of 40,962 columns in one hour of the 120 km hrrrv5 forecast of 2023-03-31,
median +11 % in the free-atmosphere length there), and an exact run that no
longer reproduces WRF v4.6.1.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu
from woof import correct_mode
from woof.config import RunConfig
from woof.core import mynn_pbl

F = np.float32


def _issue_column():
    """The column of MYNN-EDMF issue 93 / WRF issue 2430: 20 layers of 100 m,
    stable at 3 K/km below level 8, with the layer between levels 3 and 4
    (zero-based) of constant theta, and nearly neutral above level 8 so the
    upward length is the longer one.  The parcel at level 8 gets the TKE that
    stops it 40 m into the constant layer, so the exact downward length is
    4*100 + 40 = 440 m; the WRF v4.6.1 sign gives 400 - 40 = 360 m."""
    nz = 20
    dz = np.full(nz, F(100.0), dtype=F)
    zw = np.arange(nz + 1, dtype=F) * F(100.0)
    theta = (F(300.0) + F(0.3) * np.arange(nz, dtype=F)).astype(F)
    theta[4:] = (theta[4:] - F(0.3)).astype(F)
    theta[4] = theta[3]                          # one layer of constant theta
    theta[9:] = (theta[8] + F(0.003) * np.arange(1, nz - 8, dtype=F)).astype(F)
    assert theta[3] == theta[4]
    beta = mynn_pbl.GTR
    iz, izz = 8, 4
    # Energy to cross levels 8 -> 4 (trapezoids, as the routine sums them),
    # then 40 m into the constant layer 4 -> 3.
    zdo_sup = F(0.0)
    for k in range(iz, izz, -1):
        zdo_sup = F(zdo_sup + F(beta * theta[iz] * dz[k - 1]))
        zdo_sup = F(zdo_sup - F(beta * F(theta[k - 1] + theta[k]) * dz[k - 1] * F(0.5)))
    qtke = np.full(nz, F(1.0e-6), dtype=F)
    qtke[iz] = F(zdo_sup + F(beta * F(theta[iz] - theta[izz]) * F(40.0)))
    return zw, dz, qtke, theta, iz


def test_issue_column_downward_length_is_exact_with_the_fix():
    zw, dz, qtke, theta, iz = _issue_column()
    nz = dz.size
    # lb1 = min(dlu, dld) blended with 2000 m; above level 8 the column is
    # nearly neutral, so the upward search runs to the top and lb1 = dld.
    old, _ = mynn_pbl._boulac_length(zw, dz, qtke, theta, boulac_down_sign=False)
    new, _ = mynn_pbl._boulac_length(zw, dz, qtke, theta, boulac_down_sign=True)
    blend = lambda d: F(d / F(F(1.0) + d / F(2000.0)))
    assert new[iz] == pytest.approx(blend(F(440.0)), rel=1e-5)
    assert old[iz] == pytest.approx(blend(F(360.0)), rel=1e-5)
    # Every other level is untouched by the sign.
    others = [k for k in range(nz) if k != iz]
    assert np.array_equal(old[others], new[others])


def test_linear_branch_is_continuous_with_the_quadratic_branch():
    """The fixed linear branch is the bbb -> 0 limit of the quadratic one."""
    zw, dz, qtke, theta, iz = _issue_column()
    _, exact = mynn_pbl._boulac_length(zw, dz, qtke, theta, boulac_down_sign=True)
    tilted = theta.copy()
    tilted[3] = F(tilted[3] - F(1.0e-3))           # bbb = +1e-5 K/m
    _, near = mynn_pbl._boulac_length(zw, dz, qtke, tilted, boulac_down_sign=True)
    _, near_old = mynn_pbl._boulac_length(zw, dz, qtke, tilted, boulac_down_sign=False)
    assert np.array_equal(near, near_old)           # quadratic branch: no change
    assert near[iz] == pytest.approx(exact[iz], rel=5e-3)
    _, wrong = mynn_pbl._boulac_length(zw, dz, qtke, theta, boulac_down_sign=False)
    assert abs(near[iz] - exact[iz]) < 0.1 * abs(near[iz] - wrong[iz])


def test_columns_without_equal_adjacent_theta_are_bit_identical():
    rng = np.random.default_rng(93)
    nz = 40
    for _ in range(200):
        dz = rng.uniform(20.0, 400.0, nz).astype(F)
        zw = np.concatenate([[F(0.0)], np.cumsum(dz, dtype=F)]).astype(F)
        theta = (F(290.0) + np.cumsum(rng.uniform(1e-4, 0.02, nz).astype(F) * dz,
                                      dtype=F)).astype(F)
        assert np.all(theta[1:] != theta[:-1])
        qtke = rng.uniform(0.005, 2.0, nz).astype(F)
        a = mynn_pbl._boulac_length(zw, dz, qtke, theta, boulac_down_sign=False)
        b = mynn_pbl._boulac_length(zw, dz, qtke, theta, boulac_down_sign=True)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_kernel_source_has_both_signs_behind_the_define():
    """Exact mode compiles WRF's line; ML1 compiles the upstream line."""
    from pathlib import Path
    text = (Path(mynn_pbl.__file__).parent / "kernels" / "mynn_pbl.cu").read_text()
    start = text.index("__device__ void mynn_boulac_elblavg(")
    body = text[start:text.index("__device__ void", start + 10)]
    fixed = "MYNN_SUB(thetaw[iz], thetaw[izz])"
    wrf = "MYNN_SUB(thetaw[izz], thetaw[iz])"
    at = body.index("#if defined(MYNN_BOULAC_DOWN_SIGN)")
    guarded = body[at:body.index("#endif", at)]
    assert fixed in guarded.split("#else")[0]
    assert wrf in guarded.split("#else")[1]


def test_ml1_is_a_default_correction_scoped_to_mynn():
    assert correct_mode.FINDING_BY_KEY["mynn_boulac_down_sign"] == "ML1"
    assert RunConfig.__dataclass_fields__["mynn_boulac_down_sign"].default \
        is correct_mode.CORRECT_KEYS_DEFAULT
    assert correct_mode.resolve("exact").settings["mynn_boulac_down_sign"] is False
    assert correct_mode.resolve("correct").settings["mynn_boulac_down_sign"] is True
    assert correct_mode.key_in_scope("mynn_boulac_down_sign", {"bl_pbl_physics": 5})
    assert correct_mode.key_in_scope("mynn_boulac_down_sign",
                                     {"bl_pbl_physics": 5, "bl_mynn_version": "gsd_41"})
    assert not correct_mode.key_in_scope("mynn_boulac_down_sign", {"bl_pbl_physics": 1})
    for version in ("wrf_471", "wrf_480"):
        assert not correct_mode.key_in_scope(
            "mynn_boulac_down_sign", {"bl_pbl_physics": 5, "bl_mynn_version": version})


def test_receipts_say_where_ml1_did_not_reach():
    """WRF 4.7.1/4.8.0 MYNN carries the same downward loop with no build of
    the fix: a correct-mode run on it lists ML1 as uncorrected, never as
    applied.  WRF's own BouLac (8) and k-epsilon (17) PBL carry it too and,
    from ML1's second key (pbl_boulac_down_sign), are corrected."""
    base = dict(nx=8, ny=6, nz=4, dx=1000.0, dy=1000.0, ztop=10000.0,
                dt=1.0, run_seconds=10.0, moist=True, mp_physics=0)
    correct = correct_mode.resolve("correct").settings
    for extra in ({"bl_pbl_physics": 5}, {"bl_pbl_physics": 8},
                  {"bl_pbl_physics": 17}):
        on = RunConfig(**{**base, **extra}, **correct)
        block = correct_mode.run_receipt({1: on})["domains"]["1"]
        assert "ML1" in block["applied_findings"] and block["uncorrected"] == [], extra
    cfg = RunConfig(**base, bl_pbl_physics=5, bl_mynn_version="wrf_480", **correct)
    block = correct_mode.run_receipt({1: cfg})["domains"]["1"]
    assert "ML1" not in block["applied_findings"]
    # WRF 4.8.0's MYNN also carries the dew limiter (MY1), which only the
    # wrf_461 generation corrects, so it is listed there beside ML1.
    assert block["uncorrected"] == ["MY1", "ML1"]


def test_driver_passes_ml1_only_where_the_length_kernels_read_it():
    from woof.core.mynn_pbl_gpu import (MYNN_BOULAC_DOWN_SIGN_FUNCS,
                                         MYNN_BOULAC_DOWN_SIGN_VERSIONS)
    from woof.ensemble.packed_production_physics import _mynn_pbl_selectors
    assert MYNN_BOULAC_DOWN_SIGN_FUNCS == {"mynn_mixlength_default_columns",
                                           "mynn_initialize_default_columns"}
    assert MYNN_BOULAC_DOWN_SIGN_VERSIONS == ("wrf_461", "gsd_41")

    class Cfg:
        bl_mynn_version = "wrf_461"
        bl_mynn_gsd41_unsquared_qtke = False
        bl_mynn_cloud_tendency_form = "wrf_461"
        mynn_boulac_down_sign = True
    assert _mynn_pbl_selectors(Cfg)["mynn_boulac_down_sign"] is True
    Cfg.mynn_boulac_down_sign = False
    assert _mynn_pbl_selectors(Cfg)["mynn_boulac_down_sign"] is False


def _mixlength_values(ncol=64, nz=30, seed=7):
    """Columns whose mass-level theta is exactly constant over three levels,
    so two adjacent w-level thetas are equal, with TKE that sinks parcels
    into that layer.  The first column is the issue column's shape."""
    rng = np.random.default_rng(seed)
    dz = rng.uniform(40.0, 200.0, (ncol, nz)).astype(F)
    lapse = rng.uniform(0.002, 0.008, (ncol, nz)).astype(F)
    theta = (F(295.0) + np.cumsum(lapse * dz, axis=1, dtype=F)).astype(F)
    k0 = rng.integers(3, nz - 12, ncol)
    for c, k in enumerate(k0):
        # Equal layer depths make afk = abk = 0.5 exactly, so the two w-level
        # thetas inside the constant layer are equal words, as in a model.
        dz[c, k - 1:k + 4] = dz[c, k]
        theta[c, k:k + 3] = theta[c, k]                       # constant layer
        theta[c, k + 3:] = (theta[c, k + 3:] - (theta[c, k + 3] - theta[c, k])
                            + F(0.3)).astype(F)
    zw = np.concatenate([np.zeros((ncol, 1), F), np.cumsum(dz, axis=1, dtype=F)], axis=1).astype(F)
    values = {
        "dz": dz, "zw": zw,
        "u": rng.uniform(-8, 8, (ncol, nz)).astype(F),
        "v": rng.uniform(-8, 8, (ncol, nz)).astype(F),
        "qke": rng.uniform(0.05, 3.0, (ncol, nz)).astype(F),
        "dtv": rng.uniform(1e-4, 5e-3, (ncol, nz)).astype(F),
        "theta": theta,
        "vt": np.zeros((ncol, nz), F), "vq": np.zeros((ncol, nz), F),
        "cldfra": np.zeros((ncol, nz), F),
        "edmf_w": np.zeros((ncol, nz), F), "edmf_a": np.zeros((ncol, nz), F),
        "xland": np.ones(ncol, F), "dx": np.full(ncol, F(3000.0)),
        "rmo": rng.uniform(-0.01, 0.01, ncol).astype(F),
        "flt": np.zeros(ncol, F), "fltv": rng.uniform(0.0, 0.05, ncol).astype(F),
        "flq": np.zeros(ncol, F), "zi": rng.uniform(300.0, 1500.0, ncol).astype(F),
        "psig_bl": np.ones(ncol, F),
    }
    return values


@requires_gpu
def test_gpu_length_kernel_moves_with_the_cpu_reference():
    """The built mixing-length kernel, with and without the define, against
    the CPU translation with and without the fix: the define changes the
    same columns the CPU fix changes, and each build stays within the ULP
    class of the existing MYNN length parity (powf/tanhf, at most 2 ULP)."""
    import cupy as cp
    from woof.core.fp32_ulp import fp32_ulp_distance
    from woof.core.mynn_pbl_gpu import mynn_mixlength_default_cuda
    values = _mixlength_values()
    device = {k: cp.asarray(v) for k, v in values.items()}
    cpu = {fixed: mynn_pbl.mynn_mixlength_default(values, bl_mynn_mixlength=1,
                                                  boulac_down_sign=fixed)["el"]
           for fixed in (False, True)}
    gpu = {fixed: cp.asnumpy(mynn_mixlength_default_cuda(
               device, bl_mynn_mixlength=1, mynn_boulac_down_sign=fixed).el).copy()
           for fixed in (False, True)}
    for fixed in (False, True):
        assert int(fp32_ulp_distance(gpu[fixed], cpu[fixed]).max()) <= 2, fixed
    cpu_changed = np.any(cpu[False] != cpu[True], axis=1)
    gpu_changed = np.any(gpu[False] != gpu[True], axis=1)
    assert cpu_changed.sum() >= 8                     # the branch is reached
    assert np.array_equal(cpu_changed, gpu_changed)


@requires_gpu
def test_gpu_length_option2_never_reads_ml1():
    import cupy as cp
    from woof.core.mynn_pbl_gpu import mynn_mixlength_default_cuda
    values = _mixlength_values()
    device = {k: cp.asarray(v) for k, v in values.items()}
    # Option 2 never runs BouLac: identical with and without the define.
    a = cp.asnumpy(mynn_mixlength_default_cuda(device, bl_mynn_mixlength=2).el).copy()
    b = cp.asnumpy(mynn_mixlength_default_cuda(device, bl_mynn_mixlength=2,
                                               mynn_boulac_down_sign=True).el).copy()
    assert np.array_equal(a, b)
