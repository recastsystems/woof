"""MPAS-A's lagged microphysics heating for the WOOF column physics path (finding SG2).

What native MPAS-A v8.4.2 does (the authority, read at MPAS-Model f34984b2):

* ``mpas_atmphys_interface.F:909-919`` (microphysics_to_MPAS): before the
  microphysics results are written back, ``rt_diabatic_tend = theta_m``; after
  ``theta_m = th_p*(1 + rvord*qv_p)`` it becomes
  ``rt_diabatic_tend = (theta_m - rt_diabatic_tend)/dt_dyn`` -- the change of
  the MODIFIED potential temperature across the microphysics call, per second.
* ``mpas_atm_time_integration.F:6937``: every RK stage of the NEXT dynamics step
  adds ``rho_zz*rt_diabatic_tend`` to the theta tendency (after ``rthdynten``
  is formed), so the acoustic solver feels the latent heating.
* ``mpas_atm_time_integration.F:4436-4437``: the final RK recovery removes
  ``dt*rho_zz*rt_diabatic_tend`` again, because the microphysics call itself
  writes the theta change.  The endpoint does not double-count; the pressure
  and wind response inside the step is what the carry changes.

hex had the consumer half only on ``hex/gap-da-to-nest-les`` (commit e7be214b,
fed by the native mpasphys runtime) and no producer at all on the WOOF column
path that every state analysis forecast runs: the adapter declined the engine's WRF-style
``h_diabatic`` (a DRY-theta rate, which is not the MPAS quantity; the audit's
heating contract measured a 21 % error on a condensing fixture if it were
copied in as is).  This module is that producer.

The rate is formed from the same two values the native assignment uses:

* before: the MPAS ``theta_m`` the microphysics preparation read
  (``atmosphere.saved.theta_m``, which ``prep_mass_v841_f32`` divides by
  ``1 + rvord*max(qv, +0)`` to form ``th_p``);
* after: ``th_p*(1 + rvord*qv)`` from the scheme's returned dry theta and
  vapour, with the same sign-bit ``max(qv, +0)`` the post-microphysics
  recovery (``recover_wsm6_v841_f32``) couples with, so the carried change is
  exactly the change the commit writes into ``rho_theta``.  For ``qv >= 0`` this
  is native's line 918 bit for bit.

``dt`` is the step the microphysics ran at -- under full-step local time
stepping, the class step -- which is native's ``dt_dyn`` for that call.

There is no switch.  The carry is MPAS-A's own behaviour, so it moves the hex
WOOF path TOWARD the matched dynamics; with physics off nothing produces a
rate, the dycore launches nothing and the dry byte identity is untouched.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .cuda_fp32 import CUDA_FTZ_HELPERS

LAGGED_HEATING_FINDING = "SG2"
LAGGED_HEATING_SCHEMA = "hexcore.lagged-heating-v841/v1"

LAGGED_HEATING_SOURCE = (
    CUDA_FTZ_HELPERS
    + r"""
// SG2 -- mpas_atmphys_interface.F:909-919: the modified-theta change across
// the microphysics call, divided by the step the call ran at.
extern "C" __global__ void lagged_theta_m_rate_v842_f32(
    const int count, const float rvord, const float dt,
    const float *theta_m_before, const float *theta_after,
    const float *qv_after, float *rate, int *invalid)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= count) return;
    // The coupling recover_wsm6_v841_f32 commits: max(qv,+0) by sign bit.
    const unsigned int qv_bits = __float_as_uint(qv_after[i]);
    const float qv_coupled = (qv_bits & 0x80000000u) != 0u ? 0.0f : qv_after[i];
    const float theta_m_after = mpas_mul(
        theta_after[i], mpas_add(1.0f, mpas_mul(rvord, qv_coupled)));
    const float value = mpas_div(mpas_sub(theta_m_after, theta_m_before[i]), dt);
    if (!isfinite(value)) atomicExch(invalid, 1);
    rate[i] = value;
}
"""
)

_MODULE_KEY = "hexcore.lagged_heating_v841"
_KERNEL_NAME = "lagged_theta_m_rate_v842_f32"


def lagged_theta_m_rate_reference(
    theta_m_before: np.ndarray,
    theta_after: np.ndarray,
    qv_after: np.ndarray,
    dt: float,
    rvord: np.float32,
) -> np.ndarray:
    """Host twin of the kernel, binary32 operation for operation (no contraction)."""

    f32 = np.float32
    before = np.asarray(theta_m_before, dtype=f32)
    theta = np.asarray(theta_after, dtype=f32)
    qv = np.asarray(qv_after, dtype=f32)
    qv_coupled = np.where(np.signbit(qv), f32(0.0), qv).astype(f32)
    coeff = (f32(1.0) + (f32(rvord) * qv_coupled).astype(f32)).astype(f32)
    after = (theta * coeff).astype(f32)
    return ((after - before).astype(f32) / f32(dt)).astype(f32)


def lagged_theta_m_rate(
    *,
    theta_m_before: Any,
    theta_after: Any,
    qv_after: Any,
    dt: float,
    rvord: np.float32,
    kernel_cache: Any,
) -> Any:
    """Device rate ``(theta_m_after - theta_m_before)/dt`` for the next dynamics step."""

    import cupy as cp

    shape = tuple(theta_m_before.shape)
    for name, value in (("theta_after", theta_after), ("qv_after", qv_after)):
        if tuple(value.shape) != shape or value.dtype != cp.float32:
            raise ValueError(
                f"lagged heating {name} must be float32 {shape}; got "
                f"{value.dtype} {tuple(value.shape)} (a mismatched plane would carry "
                "another column's heating into this one)")
    if theta_m_before.dtype != cp.float32:
        raise ValueError("lagged heating theta_m_before must be float32")
    if not (float(dt) > 0.0 and np.isfinite(float(dt))):
        raise ValueError(f"lagged heating dt must be positive and finite, got {dt}")
    # The inputs are the CALLER's planes, and this launch must not write them.
    # On a limited-area run the shared KernelCache carries a post-launch
    # observer (cuda_regional_forecast_v841.RegionalGarbageDiscipline.scrub)
    # that rewrites the garbage column (index nCells) of every padded float32
    # argument of a launch whose module is not in
    # SELF_MANAGED_GARBAGE_MODULES -- and this module is not.  theta_after is
    # the post-RK update's own theta (prepared.th_p), whose garbage column the
    # physics preparation fills with its donor column, and qv_after aliases
    # the candidate's scalars.
    #
    # THE BREAKAGE THIS PREVENTS, MEASURED 2026-10-14 on the 1 km regional
    # cull q1.128.38027.n41.86w87.70 (dt 6 s, --convection off, the 10-13
    # transferred derecho analysis): the observer zeroed th_p's garbage
    # column after this launch, recover_wsm6_v841_f32 then read theta = 0 at
    # all 55 levels of column 38027 and refused, and every limited-area
    # physics run stopped in its first step with "post-RK WSM6 numeric
    # validation refused the candidate state".  On an observed cache the
    # launch therefore sees private copies only.  The output keeps the
    # observer's pool value, 0.0, in its garbage column, which is native's
    # value there: driver_microphysics runs over cellSolveThreadStart..End
    # (mpas_atm_time_integration.F:2811-2812, mpas_atmphys_driver_microphysics.F:489)
    # and microphysics_to_MPAS writes rt_diabatic_tend only for i = its,ite
    # (mpas_atmphys_interface.F:905), so the nCells+1 element stays at its
    # allocation value.  A cache without an observer (every global run)
    # launches on the caller's planes exactly as before.
    if getattr(kernel_cache, "post_launch", None) is not None:
        def private(value: Any) -> Any:
            return cp.array(value, dtype=cp.float32, copy=True, order="C")
    else:
        private = cp.ascontiguousarray
    before = private(theta_m_before)
    theta = private(theta_after)
    qv = private(qv_after)
    rate = cp.empty(shape, dtype=cp.float32)
    invalid = cp.zeros((1,), dtype=cp.int32)
    count = int(rate.size)
    kernel = kernel_cache.raw_kernel(_KERNEL_NAME, LAGGED_HEATING_SOURCE, module_key=_MODULE_KEY)
    threads = 128
    kernel(((count + threads - 1) // threads,), (threads,),
           (np.int32(count), np.float32(rvord), np.float32(dt), before, theta, qv, rate, invalid))
    if int(invalid.get()[0]) != 0:
        raise FloatingPointError(
            "lagged microphysics heating rate is not finite; carrying it would put a "
            "non-finite theta tendency into every RK stage of the next step")
    return rate


__all__ = [
    "LAGGED_HEATING_FINDING",
    "LAGGED_HEATING_SCHEMA",
    "LAGGED_HEATING_SOURCE",
    "lagged_theta_m_rate",
    "lagged_theta_m_rate_reference",
]
