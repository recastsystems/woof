"""Native output carriers; mpas_atm_core.F:925-966.

Scheme preparation owns its clipped carriers. Output diagnostics consume the
unclipped prognostic water vapour, as MPAS does before writing its streams.
"""
from __future__ import annotations

import numpy as np
from functools import lru_cache
from importlib.resources import files

from ..cuda_physics_prep_v841 import RV_OVER_RD_F32


CUDA_SOURCE = r'''
extern "C" __global__ void mpas_output_diagnostics_v842(
    const int count, const float rvord, const float *theta_m,
    const float *qv, const float *rho_zz, const float *zz,
    const float *pressure_base, const float *pressure_p,
    float *theta, float *rho, float *pressure)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    theta[i] = __fdiv_rn(theta_m[i],
        __fadd_rn(1.0f, __fmul_rn(rvord, qv[i])));
    rho[i] = __fmul_rn(rho_zz[i], zz[i]);
    pressure[i] = __fadd_rn(pressure_base[i], pressure_p[i]);
}

extern "C" __global__ void mpas_cold_thermodynamics_v842(
    const int count, const float rgas, const float p0, const float rcv,
    const float *zz, const float *rtheta_p, const float *rtheta_base,
    float *exner, float *exner_base, float *pressure_p, float *pressure_base)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float coefficient = __fdiv_rn(rgas, p0);
    const float factor = __fmul_rn(zz[i], coefficient);
    exner[i] = thompson_aa_powf(__fmul_rn(factor,
        __fadd_rn(rtheta_p[i], rtheta_base[i])), rcv);
    exner_base[i] = thompson_aa_powf(__fmul_rn(factor, rtheta_base[i]), rcv);
    pressure_p[i] = __fmul_rn(__fmul_rn(zz[i], rgas),
        __fadd_rn(__fmul_rn(exner[i], rtheta_p[i]),
            __fmul_rn(rtheta_base[i], __fsub_rn(exner[i], exner_base[i]))));
    pressure_base[i] = __fmul_rn(__fmul_rn(__fmul_rn(zz[i], rgas),
        exner_base[i]), rtheta_base[i]);
}
'''


@lru_cache(maxsize=1)
def native_diagnostic_source():
    """Bind shared device math; mpas_atm_time_integration.F:7628-7650.

    Built once per process: the three headers are package data.  THE BREAKAGE THIS PREVENTS (hex host-bound,
    reference system 2026-10-27): every pool synchronize (two or three a physics step) re-read the three files from disk
    and re-joined them on the thread that issues the card's work.
    """
    root = files('woof.core.kernels')
    return '\n'.join(root.joinpath(name).read_text() for name in (
        'glibc_flt32.cuh', 'glibc_flt64.cuh', 'thompson_aerosol_libm.cuh')) + CUDA_SOURCE


def output_diagnostics(atmosphere, names, cache, cp):
    """Launch complete native output conversion; mpas_atm_core.F:925-966."""
    state, saved = atmosphere.state, atmosphere.saved
    qv = state.scalars[tuple(names).index('qv')]
    theta, rho, pressure = (cp.empty_like(state.rho) for _ in range(3))
    count = state.rho.size
    kernel = cache.raw_kernel('mpas_output_diagnostics_v842', native_diagnostic_source(),
                             module_key='hexcore.mpasphys.output_diagnostics_v842')
    kernel(((count + 127) // 128,), (128,), (
        np.int32(count), RV_OVER_RD_F32, saved.theta_m, qv,
        state.rho, atmosphere.vertical.zz, atmosphere.reference.pressure_base,
        saved.pressure_perturbation, theta, rho, pressure))
    return theta, rho, pressure


def initialize_thermodynamics(atmosphere, cache, *, base_only=False):
    """Cold-start EOS with native powf; mpas_atm_time_integration.F:7628-7650.

    ``base_only`` (a restart) forms the base state alone (exner_base, pressure_base depend only on zz and
    rtheta_base) and leaves the carried perturbation state untouched: MPAS reads both from its restart
    stream, where the base words are the ones this cold start wrote.  THE BREAKAGE IT FIXES: a restart kept
    the init file's base words instead, which differ from the cold start's by one unit in the last place in
    about 900 cells, so every restart drifted away from the straight run from its first step (HEX-PHYSICS-
    PARITY run group 6 #21: 48 of 101 variables differed after 1 h).
    """
    saved, reference = atmosphere.saved, atmosphere.reference
    count = atmosphere.state.rho.size
    rgas, cp = np.float32(287.0), np.float32(1004.5)
    kernel = cache.raw_kernel('mpas_cold_thermodynamics_v842', native_diagnostic_source(),
                             module_key='hexcore.mpasphys.output_diagnostics_v842')
    kernel(((count + 127) // 128,), (128,), (
        np.int32(count), rgas, np.float32(1.e5), np.float32(rgas / np.float32(cp-rgas)),
        atmosphere.vertical.zz, saved.rho_theta_perturbation, reference.rho_theta_base,
        (saved.exner.copy() if base_only else saved.exner), reference.exner_base,
        (saved.pressure_perturbation.copy() if base_only else saved.pressure_perturbation),
        reference.pressure_base))
