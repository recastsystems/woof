"""Resident WRF wrapper preparation for two-category and three-moment P3.

The separate CuPy expressions used by the older adapter each launched a
kernel, including a second evaluation of dry theta for moist_physics_prep.
One level-local kernel preserves those float32 roundings and saves the
pre-call theta/vapour before P3 changes them. No column arithmetic moves.
"""
from __future__ import annotations

import cupy as cp
import numpy as np

from woof.core import constants as c
from woof.core.device_cache import cuda_cache


@cuda_cache(maxsize=None)
def _prepare_kernel():
    return cp.ElementwiseKernel(
        "float32 base, float32 perturbation, float32 vapour, "
        "float32 phb_lo, float32 phb_hi, float32 php_lo, float32 php_hi, "
        "float32 gravity, float32 ratio, int32 moist",
        "float32 theta, float32 dz, float32 saved_theta, "
        "float32 saved_vapour, float32 ssat",
        r"""
        float full = __fadd_rn(base, perturbation);
        if (moist)
            full = __fdiv_rn(full, __fadd_rn(1.0f, __fmul_rn(ratio, vapour)));
        theta = full;
        saved_theta = full;
        saved_vapour = vapour;
        const float lower = __fdiv_rn(__fadd_rn(phb_lo, php_lo), gravity);
        const float upper = __fdiv_rn(__fadd_rn(phb_hi, php_hi), gravity);
        dz = __fsub_rn(upper, lower);
        ssat = 0.0f;
        """, "gpuwm_p3_3mom_prepare")


def prepare(state, theta, dz, ssat):
    base = state.thb if state.thb.ndim == 3 else state.thb[:, None, None]
    phb = state.phb if state.phb.ndim == 3 else state.phb[:, None, None]
    before = state.scratch(state.qv.shape, "mp_theta_m_qv_before")
    _prepare_kernel()(
        base, state.thp, state.qv, phb[:-1], phb[1:],
        state.php[:-1], state.php[1:], np.float32(c.G), np.float32(c.RVOVRD),
        np.int32(bool(getattr(state, "use_theta_m", 0))),
        theta, dz, state.h_diabatic, before, ssat)


@cuda_cache(maxsize=None)
def _radii_kernel():
    return cp.ElementwiseKernel(
        "float32 cloud_metres, float32 ice_metres", "float32 cloud, float32 ice",
        "cloud = __fmul_rn(cloud_metres, 1.0e6f); "
        "ice = __fmul_rn(ice_metres, 1.0e6f);", "gpuwm_p3_3mom_radii")


def radii(diag, state):
    _radii_kernel()(diag["effc"], diag["effi"], state.effc, state.effi)
