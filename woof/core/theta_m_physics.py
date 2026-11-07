"""WRF 4.6.1 dry-physics / moist-prognostic thermal conversions.

These kernels are selected only by ``use_theta_m=1``. The dry branch never
launches them. Explicit round-to-nearest operations retain WRF's expression
order, including its microphysics heating limiter and vapour increment.
"""
from __future__ import annotations

import numpy as np
import cupy as cp

from woof.core import constants as c
from woof.core.device_cache import cuda_cache


@cuda_cache(maxsize=None)
def _kernel(name):
    if name == "finish":
        return cp.ElementwiseKernel(
            "float32 after, float32 saved, float32 qbefore, float32 qafter, "
            "float32 base, float32 ratio, float32 limit, float32 dt, int32 disabled",
            "float32 perturbation, float32 heating",
            r"""
            if (disabled) {
                heating = 0.0f;
            } else {
                float increment = __fsub_rn(after, saved);
                increment = isnan(increment) || isnan(limit) ? nanf("") : fminf(limit, increment);
                increment = isnan(increment) || isnan(limit) ? nanf("") : fmaxf(-limit, increment);
                const float before_factor = __fadd_rn(1.0f, __fmul_rn(ratio, qbefore));
                const float after_factor = __fadd_rn(1.0f, __fmul_rn(ratio, qafter));
                const float moist_heat = __fmul_rn(increment, after_factor);
                const float vapour_heat = __fmul_rn(__fmul_rn(ratio, __fsub_rn(qafter, qbefore)), after);
                perturbation = __fsub_rn(__fadd_rn(__fadd_rn(__fmul_rn(saved, before_factor), moist_heat), vapour_heat), base);
                heating = __fdiv_rn(__fadd_rn(moist_heat, vapour_heat), dt);
            }
            """, "gpuwm_theta_m_mp_finish")
    if name == "physics_tendency":
        return cp.ElementwiseKernel(
            "float32 thermal, float32 qv, float32 vapour_tendency, float32 ratio",
            "float32 theta_tendency",
            r"""
            const float factor = __fadd_rn(1.0f, __fmul_rn(ratio, qv));
            theta_tendency = __fadd_rn(__fmul_rn(factor, theta_tendency),
                __fmul_rn(__fdiv_rn(__fmul_rn(ratio, thermal), factor), vapour_tendency));
            """, "gpuwm_theta_m_physics_tendency")
    if name == "cumulus_tendency":
        return cp.ElementwiseKernel(
            "float32 thermal, float32 qv, float32 vapour_tendency, float32 ratio",
            "float32 theta_tendency",
            r"""
            const float dry = __fdiv_rn(thermal, __fadd_rn(1.0f, __fmul_rn(ratio, qv)));
            theta_tendency = __fmul_rn(__fdiv_rn(dry, thermal),
                __fsub_rn(theta_tendency, __fmul_rn(__fmul_rn(ratio, dry), vapour_tendency)));
            """, "gpuwm_theta_m_cumulus_tendency")
    raise KeyError(name)


def save_vapour(state):
    """Snapshot qv before a microphysics call, on the current device."""
    saved = state.scratch(state.qv.shape, "mp_theta_m_qv_before")
    saved[...] = state.qv
    return saved


def finish_microphysics(state, cfg, theta_after, dt, *, theta_before=None):
    """``moist_physics_finish_em``:5493-5788, moist branch, in place.

    WRF zeros small species before this operation; at theta_m=1 the
    resulting vapour must enter both the heating and prognostic update.
    """
    if getattr(cfg, "mp_zero_out", 0):
        from woof.core.microphysics import microphysics_zero_out
        microphysics_zero_out(state, cfg)
    before = state.h_diabatic if theta_before is None else theta_before
    qbefore = state.scratch(state.qv.shape, "mp_theta_m_qv_before")
    base = (state.thermal_base() if hasattr(state, "thermal_base") else
            state.thb if state.thb.ndim == 3 else state.thb[:, None, None])
    _kernel("finish")(
        theta_after, before, qbefore, state.qv, base, np.float32(c.RVOVRD),
        np.float32(np.float32(cfg.mp_tend_lim) * np.float32(dt)), np.float32(dt),
        np.int32(cfg.no_mp_heating != 0), state.thp, state.h_diabatic)


def previous_physics_theta(state):
    """Read native t_1 before the time-t prognostics are overwritten."""
    base = (state.thermal_base() if hasattr(state, "thermal_base") else
            state.thb if state.thb.ndim == 3 else state.thb[:, None, None])
    return (base + state.thp0 if getattr(state, "thp0", None) is not None
            else state.prognostic_theta())


def convert_physics_theta_tendency(state, theta_tendency, qv_tendency, *, thermal=None):
    """``conv_t_tendf_to_moist``:6690-6714, after summed dry physics.

    Both tendencies must have the same mass and map-factor convention.
    Call once before diffusion is added. WRF uses the previous step's
    saved thermal value (t_1), but the current step's vapour (moist_old).
    Supply thermal before replacing the saved time-t prognostics.
    """
    if state.qv is None:
        return
    if thermal is None:
        thermal = previous_physics_theta(state)
    vapour = state.qv0 if getattr(state, "qv0", None) is not None else state.qv
    _kernel("physics_tendency")(
        thermal, vapour, qv_tendency,
        np.float32(c.RVOVRD), theta_tendency)


def dry_cumulus_theta_tendency(state, theta_tendency, qv_tendency):
    """``phy_prep``:5323-5334 inverse for uncoupled advective forcing."""
    if state.qv is None:
        return
    _kernel("cumulus_tendency")(
        state.prognostic_theta(), state.qv, qv_tendency,
        np.float32(c.RVOVRD), theta_tendency)


def finish_cumulus_forcing(state, cfg):
    """Convert GF's retained forcing before microphysics changes theta/qv.

    Native phy_prep_part2 runs after the last RK stage and before the
    microphysics adjustment. The following cumulus call consumes that dry
    forcing, even though its atmospheric inputs include the adjustment.
    """
    if not cfg.use_theta_m or cfg.cu_physics != 3 or state.rthften is None:
        return
    from woof.wrf_exact import ENABLED as WRF_EXACT
    thermal = state.rthften
    vapour = state.rqvften
    if WRF_EXACT:
        from woof.state_serialization_contract import WRF_MUTS_SLOT
        mut = state.existing_scratch(WRF_MUTS_SLOT)
        if mut is None:
            mut = state.total_mu()
        mass = state.c1h[:, None, None] * mut[None] + state.c2h[:, None, None]
        thermal = thermal / mass
        vapour = vapour / mass
    dry_cumulus_theta_tendency(state, thermal, vapour)
    state.rthften[...] = thermal
