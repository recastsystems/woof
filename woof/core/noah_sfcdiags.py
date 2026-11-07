"""Noah's SFCDIAGS (T2, Q2, TH2) as one launch.

``PhysicsDriver._refresh_surface_diagnostics`` runs after every Noah call.
Written as CuPy array expressions it was about 20 launches from Python for a
few loads and stores per column, about 0.11 ms of every default-suite step on
a 250 x 200 grid.  This kernel makes the same writes in one launch, in WRF
v4.6.1 SFCDIAGS's arithmetic (phys/module_sf_sfcdiags.F:45-72):

* every multiply, add, subtract and divide is spelled with its ``_rn``
  intrinsic, in WRF's association (``RHO*CP*CHS2`` is ``(RHO*CP)*CHS2``), so
  NVRTC cannot contract a multiply and an add or subtract into one FMA that
  gfortran -O0 never forms;
* TH2's ``(1.E5/PSFC)**ROVCP`` is a REAL**REAL power, which WRF's build
  hands to the C library's powf; it is WOOF's own float32 powf here
  (``gfk_pow``, kernels/glibc_flt32.cuh), not CUDA's.  CUDA's powf was this
  kernel's word until development branch: graded on every float32 surface
  pressure from 20 to 110 kPa (tools/noah_wrf461_oracle/sfcdiags_sweep.py)
  it gave a different TH2 on 1,234,353 of 20,617,217 pressures, up to 2 ULP;
  gfk_pow gives WRF's word on all of them;
* each WRF ``if`` is a select of values both of which are computed, so a
  column's branch never changes another column's bits.

TH2 reads the T2 this launch just wrote, as WRF's TH2 reads T2(I,J).  Q2
carries one documented bound WRF does not
(``PhysicsDriver._refresh_surface_diagnostics``).  Carriers that are not
C-contiguous float32 CuPy arrays of the grid's shape go through the same
kernel on contiguous copies (:func:`refresh_any`), so there is one
arithmetic, not two.
"""

from __future__ import annotations

import numpy as np

from woof.core.noah_forcing import device_arrays as device_arrays

_KERNEL_TEXT = r"""
extern "C" __global__
void noah_sfcdiags(const float* psfc, const float* tsk, const float* cqs2,
                   const float* chs2, const float* qsfc, const float* qfx,
                   const float* hfx, const float* qv1,
                   float* q2, float* t2, float* th2,
                   const float rd, const float cp_air, const float p0,
                   const float rcp, const float active_floor,
                   const long long ncol)
{
    const long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncol) return;
    const float ps = psfc[c];
    const float ts = tsk[c];
    // rho = psfc / (RD * tsk)
    const float rho = __fdiv_rn(ps, __fmul_rn(rd, ts));
    const float cq = cqs2[c];
    const float ch = chs2[c];
    const bool q_active = cq >= active_floor;
    const bool t_active = ch >= active_floor;
    const float safe_cqs2 = q_active ? cq : 1.0f;
    const float safe_chs2 = t_active ? ch : 1.0f;
    // diagnosed = where(q_active, qsfc - qfx / (rho * safe_cqs2), qsfc)
    const float qs = qsfc[c];
    const float inverted = __fsub_rn(
        qs, __fdiv_rn(qfx[c], __fmul_rn(rho, safe_cqs2)));
    const float diagnosed = q_active ? inverted : qs;
    // q2 = where(diagnosed > 0, diagnosed, qv[0])
    q2[c] = (diagnosed > 0.0f) ? diagnosed : qv1[c];
    // t2 = where(t_active, tsk - hfx / (rho * CP * safe_chs2), tsk)
    const float heated = __fsub_rn(
        ts, __fdiv_rn(hfx[c], __fmul_rn(__fmul_rn(rho, cp_air), safe_chs2)));
    const float temperature = t_active ? heated : ts;
    t2[c] = temperature;
    // th2 = t2 * power(P0 / psfc, RCP)
    th2[c] = __fmul_rn(temperature, gfk_pow(__fdiv_rn(p0, ps), rcp));
}
"""


def _source() -> str:
    """WOOF's float32 libm header, then the kernel: the string NVRTC gets."""
    from pathlib import Path

    header = Path(__file__).resolve().parent / "kernels" / "glibc_flt32.cuh"
    return header.read_text(encoding="utf-8") + _KERNEL_TEXT

_THREADS = 256
_KERNEL = None
_NATIVE_Q2_KERNEL = None


def _kernel(native_q2=False):
    global _KERNEL, _NATIVE_Q2_KERNEL
    cached = _NATIVE_Q2_KERNEL if native_q2 else _KERNEL
    if cached is None:
        import cupy as cp
        from woof.certify.kernel_manifest import record_module
        from woof.core.kernels import _compile_observed

        # A forecast translation unit, so the kernel manifest records it
        # (tests/test_kernel_manifest.py SITE_FILES): a RawModule, the
        # compile a RawKernel of the same source and options makes.
        source = _source()
        if native_q2:
            # Strict moist-theta proofs retain WRF's diagnostic word even
            # when its flux inversion gives a negative mixing ratio.
            source = source.replace(
                "q2[c] = (diagnosed > 0.0f) ? diagnosed : qv1[c];",
                "q2[c] = diagnosed;")
        module = cp.RawModule(code=source, options=())
        _compile_observed(module, "woof.core.noah_sfcdiags")
        record_module("woof.core.noah_sfcdiags", source=source,
                      options=(), module=module)
        cached = module.get_function("noah_sfcdiags")
        if native_q2:
            _NATIVE_Q2_KERNEL = cached
        else:
            _KERNEL = cached
    return cached


def refresh(fields, qv1, *, rd, cp_air, p0, rcp, active_floor,
            native_q2=False) -> None:
    """Write ``q2``, ``t2`` and ``th2`` in ``fields`` in one launch.

    The constants are the float32 scalars the expressions use, passed in so
    this module holds no second copy of them.
    """
    ncol = int(fields["tsk"].size)
    _kernel(native_q2)(
        ((ncol + _THREADS - 1) // _THREADS,), (_THREADS,),
        (fields["psfc"], fields["tsk"], fields["cqs2"], fields["chs2"],
         fields["qsfc"], fields["qfx"], fields["hfx"], qv1,
         fields["q2"], fields["t2"], fields["th2"],
         np.float32(rd), np.float32(cp_air), np.float32(p0), np.float32(rcp),
         np.float32(active_floor), np.int64(ncol)))


def refresh_any(fields, qv1, *, rd, cp_air, p0, rcp, active_floor,
                native_q2=False) -> None:
    """:func:`refresh` for any CuPy carrier of the grid's shape.

    Inputs are copied to C-contiguous float32 arrays, the one kernel runs on
    the copies, and the three results are written back into ``fields``.
    """
    import cupy as cp

    shape = fields["tsk"].shape
    names = ("psfc", "tsk", "cqs2", "chs2", "qsfc", "qfx", "hfx")
    work = {name: cp.ascontiguousarray(cp.broadcast_to(
        cp.asarray(fields[name], dtype=cp.float32), shape))
        for name in names}
    for name in ("q2", "t2", "th2"):
        work[name] = cp.empty(shape, dtype=cp.float32)
    level1 = cp.ascontiguousarray(cp.broadcast_to(
        cp.asarray(qv1, dtype=cp.float32), shape))
    refresh(work, level1, rd=rd, cp_air=cp_air, p0=p0, rcp=rcp,
            active_floor=active_floor, native_q2=native_q2)
    for name in ("q2", "t2", "th2"):
        fields[name][...] = work[name]
def refresh_ua(fields,qv1,t1,*,rcp):
    """WRF UA reads CHS and lowest-level temperature directly."""
    from woof.core.kernels import get_kernel
    n=int(fields["tsk"].size)
    get_kernel("noah_ua_sfcdiags","noah_ua_sfcdiags")(
        ((n+_THREADS-1)//_THREADS,),(_THREADS,),
        (fields["psfc"],fields["tsk"],fields["chs"],fields["chs2"],
         fields["cqs2"],fields["qsfc"],qv1,t1,
         fields["q2"],fields["t2"],fields["th2"],np.float32(rcp),np.int64(n)))
