"""Initial snow-layer temperatures for a Noah-MP cold start.

WRF's NOAHMP_INIT / SNOW_INIT (module_sf_noahmpdrv.F, SNOW_INIT lines
2411-2418 in v4.6.1) set every snow layer to TGXY, the skin temperature of the
analysis.  A cold start from GFS/HRRR at dusk over fresh deep snow therefore
starts the whole pack at the radiatively cooled skin (e.g. 248 K) while the
soil under it is 264-272 K.  A 15 cm pack at 200 kg/m3 conducts that error
away on a timescale of L^2/kappa ~ 20 h (k = 0.13 W/m/K, rho c = 0.42 MJ/m3/K),
so the cold reservoir drives the surface for the whole first night.

``noahmp_snow_tinit`` chooses the initial snow-layer temperatures:

1. WRF: every layer at TGXY (the v4.6.1/4.7.1 behaviour; byte-identical).
2. Conduction profile: each layer's midpoint takes the steady-state 1-D
   conduction temperature of a homogeneous slab between its two boundary
   temperatures (Fourier's law: a linear profile), TGXY at the snow surface and
   the top soil layer's temperature TSLB(1) at the snow/soil interface; capped
   at freezing.  The soil carries most of the column's heat capacity and has a
   conductivity an order above snow's, so the interface sits at the top soil
   temperature to within a fraction of a kelvin.

Only the cold start reads it; a warm start (restart, DA hand-off, hydrated
state) keeps the snow temperatures it carries.  Sea-ice columns (XICE at or
above the threshold) and columns with no snow layer are untouched.
"""
from __future__ import annotations

import numpy as np

SNOW_TINIT_CHOICES = (1, 2)
TFRZ = np.float32(273.16)   # module_sf_noahmplsm.F TFRZ


def validate(value) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) not in SNOW_TINIT_CHOICES:
        raise ValueError(f"noahmp_snow_tinit must be one of {SNOW_TINIT_CHOICES} "
                         f"(1 WRF skin-temperature pack, 2 conduction profile), got {value!r}")
    return int(value)


def conduction_profile_host(*, isnow, zsnso, tg, tslb1, xice, xice_threshold):
    """Host reference for one column: returns the three snow-layer temperatures
    (slab order: index 0 is snow layer -2, index 2 is layer 0), or None when the
    column is left as WRF set it."""
    nsnow = 3
    n = -int(isnow)
    if n <= 0 or float(xice) >= float(xice_threshold):
        return None
    h = np.float32(-zsnso[nsnow - 1])
    if not h > 0:
        return None
    out = [None] * nsnow
    zprev = np.float32(0.0)
    for k in range(nsnow - n, nsnow):
        zb = np.float32(-zsnso[k])
        zm = np.float32(np.float32(0.5) * np.float32(zprev + zb))
        t = np.float32(tg + np.float32(np.float32(tslb1 - tg) * np.float32(zm / h)))
        out[k] = min(t, TFRZ)
        zprev = zb
    return out


_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        import cupy as cp
        _KERNEL = cp.RawKernel(r'''
extern "C" __global__ void noahmp_snow_tinit_conduction(
    const int n, const int* __restrict__ isnow, const float* __restrict__ zsnso,
    const float* __restrict__ tg, const float* __restrict__ tslb1,
    const float* __restrict__ xice, const float xice_threshold,
    float* __restrict__ tsno)
{
    // zsnso and tsno are layer-major slabs: layer k of column c at k*n + c.
    const int c = blockDim.x * blockIdx.x + threadIdx.x;
    if (c >= n) return;
    const int nl = -isnow[c];
    if (nl <= 0 || xice[c] >= xice_threshold) return;
    const float h = -zsnso[2 * n + c];
    if (!(h > 0.0f)) return;
    const float t0 = tg[c], dt = tslb1[c] - t0;
    float zprev = 0.0f;
    for (int k = 3 - nl; k < 3; ++k) {
        const float zb = -zsnso[k * n + c];
        const float zm = __fmul_rn(0.5f, __fadd_rn(zprev, zb));
        const float t = __fadd_rn(t0, __fmul_rn(dt, __fdiv_rn(zm, h)));
        tsno[k * n + c] = fminf(t, 273.16f);
        zprev = zb;
    }
}
''', "noahmp_snow_tinit_conduction")
    return _KERNEL


def apply_conduction_profile(fields, *, xice_threshold: float) -> None:
    """Overwrite ``tsnoxy`` on the device with the conduction profile (choice 2).
    One thread per column; no host round trip."""
    import cupy as cp
    tsno = fields["tsnoxy"]
    nsnow, ny, nx = tsno.shape
    if nsnow != 3:
        raise ValueError(f"noahmp_snow_tinit=2 expects 3 snow layers, got {nsnow}")
    n = ny * nx
    isnow = cp.ascontiguousarray(fields["isnowxy"].reshape(n).astype(cp.int32, copy=False))
    zsnso = cp.ascontiguousarray(fields["zsnsoxy"][:nsnow].reshape(nsnow, n).astype(cp.float32, copy=False))
    tg = cp.ascontiguousarray(fields["tgxy"].reshape(n).astype(cp.float32, copy=False))
    tslb1 = cp.ascontiguousarray(fields["tslb"][0].reshape(n).astype(cp.float32, copy=False))
    xice = cp.ascontiguousarray(fields["xice"].reshape(n).astype(cp.float32, copy=False))
    work = cp.ascontiguousarray(tsno.reshape(nsnow, n).astype(cp.float32, copy=True))
    threads = 256
    _kernel()(((n + threads - 1) // threads,), (threads,),
              (np.int32(n), isnow, zsnso, tg, tslb1, xice, np.float32(xice_threshold), work))
    tsno[...] = work.reshape(nsnow, ny, nx).astype(tsno.dtype, copy=False)
