"""Radar reflectivity diagnostic: WRF ``do_radar_ref=1`` (REFL_10CM).

WRF computes ``refl_10cm`` inside the microphysics driver call on history
steps (``diagflag .and. do_radar_ref == 1``, phys/module_microphysics_driver.F
refl_10cm argument; Morrison wrapper phys/module_mp_morr_two_moment.F:911-918)
from the scheme's own post-call 1-D columns, so the diagnostic is consistent
with the ACTIVE scheme's intercepts, densities, and prognostic number
concentrations.  woof uses the same arrangement: an output-due flag reaches
the scheme adapter, which calls :func:`compute_refl_10cm` with the prepared
pressure/Exner and the scheme's post-call temperature before
``moist_physics_finish`` or the dycore EOS refresh.  The result is stashed
until the output frame consumes it once (see PROVENANCE.md D2).
Morrison's fixed-Nc cloud path (INUM=1, NC3D = 250 cm-3/rho) never enters:
``refl10cm_hm`` reads no cloud-water moment at all (F:4502-4675).

Transcription authority (local WRF v4.6.1):

- ``phys/module_mp_morr_two_moment.F``: the radar m(D) parameter block
  (:528-542: xam_r = PI*RHOW/6, xbm_r = 3, xmu_r = 0; xam_s = CS; xam_g = CG;
  ``morr_rimed_ice`` selects RHOG/CG, with WRF's Registry default 1 = hail),
  the wrapper call/clamp (:911-918), and ``refl10cm_hm`` (:4502-4675).
- ``phys/module_mp_radar.F``: ``radar_init`` (:74-190) products consumed by
  the column routine -- WRF builds them once at scheme init on the host, and
  :func:`radar_init` mirrors that split: the products are computed here in
  float64 and handed to both the CUDA kernel and the npref mirror (the same
  single-source pattern as ``CUDA_DEFINES``/``noah_frh2o``).  Per-column math
  (the Blahak melting-particle soak) is transcribed separately in
  ``kernels/refl.cu`` and ``woof.verify.npref.np_refl10cm_morrison_column``.

Kessler (mp_physics=1) fallback: WRF's Kessler carries NO reflectivity
diagnostic (module_mp_kessler.F has none), so the fallback is the classic
rain-only Rayleigh form for an exponential Marshall-Palmer PSD with fixed
intercept N0r = 8e6 m-4 and rho_w = 1000 kg m-3 (Smith et al. 1975, J. Appl.
Meteor. 14, 1156-1165, eq. for Ze of liquid rain; the same constants
Kessler's own terminal-velocity closure assumes and the form wrf-python/RIP
``dbzcalc`` uses for one-moment warm rain):

    lambda_r = (pi * rho_w * N0r / (rho * qr))**0.25
    Ze = Gamma(7) * N0r * lambda_r**-7 * 1e18   [mm6 m-3]

with the same air-density diagnosis, q-thresholds, and -35 dBZ floor
conventions as the Morrison routine so the two schemes' outputs are
interchangeable in wrfout/products.

Compatibility limits inherited from WRF are deliberate.  For an active
species (mass mixing ratio ``> 1e-9``), its number moment must be finite and
strictly positive.  ``refl10cm_hm`` gates only on mass and immediately forms
the fractional-power slope (module_mp_morr_two_moment.F:4544-4584); it does
not define a meteorological result for a zero, negative, or non-finite number
moment.  Normal Morrison outputs satisfy this contract because the scheme
clips number concentrations nonnegative and reconstructs them from bounded
slopes (:1528-1635).  woof explicitly leaves the affected output non-finite
instead of allowing CUDA's ``fmaxf`` to turn invalid slope arithmetic into
the meteorological -35 dBZ floor.  Likewise, the melting calculation is the
exact WRF 50-log-bin weighted sum (module_mp_radar.F:83-90/:124-146 and
module_mp_morr_two_moment.F:4626-4667), retained for compatibility; it is not
claimed to be a converged composite-Simpson integral.

WSM6 (mp_physics=6) uses its native fixed-intercept rain/snow/graupel-or-
hail PSDs from ``mp_wsm6.F90:2275-2444``, including the same Blahak melting
particle integration.  Float64 mirrors: ``np_refl10cm_morrison_column``,
``np_refl10cm_wsm6_column``, and ``np_refl10cm_kessler_column``
(``woof/verify/npref.py``).

Classic Thompson (mp_physics=8) uses ``calc_refl10cm`` from
``module_mp_thompson.F:5710-6028``: two-moment rain, the Field et al. snow
moment distribution and wet-snow Blahak integration, and its private
same-call classic-graupel number moment.  That graupel moment is supplied as
an output-due scratch shadow; it is not transported model state.

Aerosol-aware Thompson (mp_physics=28) uses the SAME routine with the same
inputs.  ``calc_refl10cm`` is not duplicated or specialized in WRF: one
subroutine serves both packages, it takes no droplet-number argument, and
its only cloud-water local is dead (see :func:`compute_refl_10cm`).  So
REFL_10CM under mp=28 is a function of qv/qr/nr/qs/qg/T/p exactly as it is
under mp=8, and the prognostic ``nc`` that defines mp=28 has no influence on
it whatsoever.  This is the one user-facing product field the aerosol port
adds no new numerics to, and stating that is more useful than a branch that
implies otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import lru_cache
from woof.core.device_cache import cuda_cache

import numpy as np

from woof.core.morrison_constants import rimed_ice_constants
from woof.core.wsm6_constants import rimed_ice_constants as wsm6_rimed

_COLUMN_TPB = 32
# The Kessler fallback is one independent cell per thread: no shared state,
# atomics, synchronization, or block-shape-dependent arithmetic.
_CELL_TPB = 256
_KMAX = 256                # unspecialized REFL_KMAX in kernels/refl.cu

#: Per-thread local frame each column kernel compiles to, per level of
#: ``REFL_KMAX``.  The column arrays are all that scale with the bound, so
#: the driver-reported frame is exactly ``bytes_per_level * REFL_KMAX``.
#: Measured on the RTX 5090 (driver 610.74, NVRTC 13.x, ``-std=c++17``) at
#: bounds 256 and 49; ``tests/test_kernel_local_bounds.py`` re-measures it.
KERNEL_LOCAL_FRAME_BYTES_PER_LEVEL = {
    "refl10cm_morrison_column": 72,     # 18,432 B at 256
    "refl10cm_thompson_column": 63,     # 16,128 B at 256
    "refl10cm_wsm6_column": 55,         # 14,080 B at 256
    "refl10cm_kessler_cell": 0,         # per-cell, no column arrays
}


def kernel_local_frame_bytes(nz: int) -> int:
    """Widest per-thread local frame in ``refl.cu`` at ``REFL_KMAX = nz``."""
    return max(KERNEL_LOCAL_FRAME_BYTES_PER_LEVEL.values()) * int(nz)


def kernel_capacity(nz: int) -> int:
    """The ``REFL_KMAX`` this ``nz`` compiles against.

    Exactly ``nz``.  Every loop in the three column kernels runs to the
    runtime ``nz`` and the highest index any of them forms is ``nz - 1``
    (the melting scan reads ``k + 1`` from ``k <= nz - 2``), so the bound is
    a pure allocation size.  At the unspecialized 256 on a 49-level case
    ``refl10cm_morrison_column`` costs 4,335 MiB of driver local-memory
    backing store the moment the first history frame comes due.
    """
    nz = int(nz)
    if nz < 1 or nz > _KMAX:
        raise ValueError(f"nz={nz} exceeds REFL_KMAX={_KMAX}")
    return nz


def _column_kernel(symbol: str, nz: int, module: str = "refl"):
    """One reflectivity column kernel, compiled at ``REFL_KMAX = nz``.

    ``module`` is the translation unit.  WDM6's kernel lives in its own
    (``wdm6_refl.cu``) rather than in ``refl.cu``, because ``refl.cu`` is
    byte-frozen -- see that file's header and
    ``tests/test_mp8_frozen.py::test_every_frozen_kernel_module_is_unchanged``
    -- so a new scheme may not be appended to it.
    """
    from woof.core.kernels import get_kernel, get_kernel_int_defines

    capacity = kernel_capacity(nz)
    if capacity == _KMAX:
        return get_kernel(module, symbol)
    return get_kernel_int_defines(module, symbol,
                                  (("REFL_KMAX", capacity),))

#: module_mp_radar.F:40/43/63-64: 50 size bins, 10 cm wavelength, and the
#: 90% external-meltwater assumption for snow and graupel.
NRBINS = 50
LAMDA_RADAR = 0.10
MELT_OUTSIDE_S = 0.9
MELT_OUTSIDE_G = 0.9

#: Morrison module PI (module_mp_morr_two_moment.F:99) -- also the radar
#: module's PIx (module_mp_radar.F:203/283).
_PI = 3.1415926535897932384626434
#: module_model_constants.F:19 r_d, Morrison's R (F:92).
_R = 287.0


def _m_complex_water_ray(lam: float, t: float) -> complex:
    """Complex refractive index of water after Ray (1972);
    module_mp_radar.F:193-222 verbatim (T in deg C, lam in m)."""
    epsinf = 5.27137 + 0.02164740 * t - 0.00131198 * t * t
    epss = 78.54 * (1.0 - 4.579e-3 * (t - 25.0)
                    + 1.190e-5 * (t - 25.0) * (t - 25.0)
                    - 2.800e-8 * (t - 25.0) * (t - 25.0) * (t - 25.0))
    alpha = -16.8129 / (t + 273.16) + 0.0609265
    lambdas = 0.00033836 * math.exp(2513.98 / (t + 273.16)) * 1e-2
    nenner = (1.0 + 2.0 * (lambdas / lam) ** (1.0 - alpha)
              * math.sin(alpha * _PI * 0.5)
              + (lambdas / lam) ** (2.0 - 2.0 * alpha))
    epsr = epsinf + ((epss - epsinf) * ((lambdas / lam) ** (1.0 - alpha)
                                        * math.sin(alpha * _PI * 0.5)
                                        + 1.0)) / nenner
    epsi = (((epss - epsinf) * ((lambdas / lam) ** (1.0 - alpha)
                                * math.cos(alpha * _PI * 0.5)
                                + 0.0)) / nenner
            + lam * 1.25664 / 1.88496)
    return complex(np.sqrt(complex(epsr, -epsi)))


def _m_complex_ice_maetzler(lam: float, t: float) -> complex:
    """Complex refractive index of ice after Maetzler (1998);
    module_mp_radar.F:226-262 verbatim (T in deg C, lam in m)."""
    c = 2.99e8
    tk = t + 273.16
    f = c / lam * 1e-9
    b1 = 0.0207
    b2 = 1.16e-11
    b = 335.0
    deltabeta = math.exp(-10.02 + 0.0364 * (tk - 273.16))
    betam = (b1 / tk) * (math.exp(b / tk)
                         / ((math.exp(b / tk) - 1.0) ** 2)) + b2 * f * f
    beta = betam + deltabeta
    theta = 300.0 / tk - 1.0
    alfa = (0.00504 + 0.0062 * theta) * math.exp(-22.1 * theta)
    m = complex(3.1884 + 9.1e-4 * (tk - 273.16), 0.0)
    m = m + complex(0.0, alfa / f + beta * f)
    return complex(np.sqrt(np.conj(m)))


@dataclass(frozen=True)
class RadarInit:
    """``radar_init`` products (module_mp_radar.F:74-190), float64.

    Gamma moments are exact ``math.gamma`` where WRF evaluates its
    REAL WGAMMA/GAMMLN pair (Numerical Recipes 2.02, :561-597) -- the same
    substitution the Morrison port already makes; the relative difference
    is O(1e-7), far below the FP32 state floor.
    """

    xam_r: float
    xam_s: float
    xam_g: float
    xcre: tuple      # (1+bm, 1+mu, 1+bm+mu, 1+2bm+mu) for rain, and
    xcse: tuple      # likewise snow/graupel (all bm=3, mu=0 for Morrison)
    xcge: tuple
    xcrg: tuple      # Gamma(xcre(n))
    xcsg: tuple
    xcgg: tuple
    xorg2: float     # 1/Gamma(1+mu) per species
    xosg2: float
    xogg2: float
    xobmr: float     # 1/bm and the (1/am)**(1/bm) soak geometry factors
    xoams: float
    xobms: float
    xocms: float
    xoamg: float
    xobmg: float
    xocmg: float
    pi5: float       # 3.14159**5 (truncated pi, :77 verbatim)
    lamda4: float
    m_w_0: complex
    m_i_0: complex
    k_w: float
    xxds: np.ndarray     # snow bin centers / widths (100 um .. 2 cm)
    xdts: np.ndarray
    xxdg: np.ndarray     # graupel bin centers / widths (100 um .. 5 cm)
    xdtg: np.ndarray
    simpson: np.ndarray  # composite-Simpson weights, first NRBINS entries


def _frozen(a: np.ndarray) -> np.ndarray:
    a.setflags(write=False)
    return a


def _log_bins(dmax: float) -> tuple[np.ndarray, np.ndarray]:
    """Log-spaced bin centers/widths from 100 um to ``dmax``
    (module_mp_radar.F:124-146)."""
    edges = np.empty(NRBINS + 1, dtype=np.float64)
    edges[0] = 100.0e-6
    edges[NRBINS] = dmax
    for n in range(1, NRBINS):
        edges[n] = math.exp(n / NRBINS * math.log(edges[NRBINS] / edges[0])
                            + math.log(edges[0]))
    return np.sqrt(edges[:-1] * edges[1:]), np.diff(edges)


@lru_cache(maxsize=2)
def radar_init(morr_rimed_ice: int = 1) -> RadarInit:
    """Mirror of ``radar_init`` for Morrison's m(D) parameters.

    Inputs are the values MORR_TWO_MOMENT_INIT sets before its
    ``call radar_init`` (module_mp_morr_two_moment.F:532-542): rain
    am = PI*RHOW/6 (RHOW = 997), bm = 3, mu = 0; snow am = CS = 100*PI/6;
    dense-ice am = CG = RHOG*PI/6, selected by ``morr_rimed_ice``.  WRF's
    Registry default 1 uses hail RHOG=900; explicit 0 uses graupel RHOG=400
    (Registry.EM_COMMON:2663-2666; F:337-411).
    """
    xam_r = _PI * 997.0 / 6.0
    xam_s = 100.0 * _PI / 6.0
    xam_g = rimed_ice_constants(morr_rimed_ice).rhog * _PI / 6.0
    xbm = 3.0
    xmu = 0.0

    pi5 = 3.14159 ** 5                        # :77, truncated pi verbatim
    lamda4 = LAMDA_RADAR ** 4                 # :78
    m_w_0 = _m_complex_water_ray(LAMDA_RADAR, 0.0)      # :79
    m_i_0 = _m_complex_ice_maetzler(LAMDA_RADAR, 0.0)   # :80
    k_w = abs((m_w_0 * m_w_0 - 1.0) / (m_w_0 * m_w_0 + 2.0)) ** 2  # :81

    simpson = np.zeros(NRBINS + 1, dtype=np.float64)    # :83-90
    basis = (1.0 / 3.0, 4.0 / 3.0, 1.0 / 3.0)
    for n in range(0, NRBINS - 1, 2):
        simpson[n] += basis[0]
        simpson[n + 1] += basis[1]
        simpson[n + 2] += basis[2]

    xxds, xdts = _log_bins(0.02)              # :123-134 snow to 2 cm
    xxdg, xdtg = _log_bins(0.05)              # :136-146 graupel to 5 cm

    # :152-181: per-species PSD-moment exponents and gamma values (bm and
    # mu identical across Morrison's three radar species).
    xce = (1.0 + xbm, 1.0 + xmu, 1.0 + xbm + xmu, 1.0 + 2.0 * xbm + xmu)
    xcg = tuple(math.gamma(v) for v in xce)
    o2 = 1.0 / xcg[1]

    return RadarInit(
        xam_r=xam_r, xam_s=xam_s, xam_g=xam_g,
        xcre=xce, xcse=xce, xcge=xce,
        xcrg=xcg, xcsg=xcg, xcgg=xcg,
        xorg2=o2, xosg2=o2, xogg2=o2,
        xobmr=1.0 / xbm,                       # :183-189
        xoams=1.0 / xam_s, xobms=1.0 / xbm,
        xocms=(1.0 / xam_s) ** (1.0 / xbm),
        xoamg=1.0 / xam_g, xobmg=1.0 / xbm,
        xocmg=(1.0 / xam_g) ** (1.0 / xbm),
        pi5=pi5, lamda4=lamda4, m_w_0=m_w_0, m_i_0=m_i_0, k_w=k_w,
        xxds=_frozen(xxds), xdts=_frozen(xdts),
        xxdg=_frozen(xxdg), xdtg=_frozen(xdtg),
        simpson=_frozen(simpson[:NRBINS]))


@lru_cache(maxsize=2)
def radar_init_wsm6(hail_opt: int = 0) -> RadarInit:
    """``radar_init`` products for WSM6's 1000/100/500-or-700 densities."""
    rimed = wsm6_rimed(hail_opt)
    xam_g = rimed.deng * _PI / 6.0
    base = radar_init(0)
    return replace(
        base, xam_r=1000.0 * _PI / 6.0, xam_g=xam_g,
        xoamg=1.0 / xam_g, xocmg=(1.0 / xam_g) ** (1.0 / 3.0))


@lru_cache(maxsize=2)
def radar_init_wdm6(hail_opt: int = 0) -> RadarInit:
    """``radar_init`` products for WDM6's m(D) inputs (wdm6init:2196-2208).

    Densities are WSM6's (rain 1000, snow 100, graupel 500 or 700 by
    ``hail_opt``), but WDM6 sets ``xmu_r = 1`` before ``call radar_init``
    while snow and graupel keep ``xmu = 0``.  So only the RAIN exponent and
    gamma tuples move: ``xcre = (1+bm, 1+mu, 1+bm+mu, 1+2bm+mu)`` becomes
    (4, 2, 5, 8) with ``xcrg = (6, 1, 24, 5040)`` and ``xorg2 = 1/G(2) = 1``.
    Snow and graupel keep the mu = 0 tuples ``radar_init`` already returns.
    """
    rimed = wsm6_rimed(hail_opt)
    xam_g = rimed.deng * _PI / 6.0
    base = radar_init(0)
    xbm, xmu_r = 3.0, 1.0
    xcre = (1.0 + xbm, 1.0 + xmu_r, 1.0 + xbm + xmu_r,
            1.0 + 2.0 * xbm + xmu_r)
    xcrg = tuple(math.gamma(v) for v in xcre)
    return replace(
        base, xam_r=1000.0 * _PI / 6.0, xam_g=xam_g,
        xcre=xcre, xcrg=xcrg, xorg2=1.0 / xcrg[1],
        xoamg=1.0 / xam_g, xocmg=(1.0 / xam_g) ** (1.0 / 3.0))


@cuda_cache(maxsize=None, ready=True)
def _device_tables():
    """Packed float64 device copy of the radar_init bin/weight tables:
    [xxDs | xdts | xxDg | xdtg | simpson], each NRBINS long."""
    import cupy as cp

    rc = radar_init()
    return cp.asarray(np.concatenate(
        [rc.xxds, rc.xdts, rc.xxdg, rc.xdtg, rc.simpson]))


def _check_fields(fields: dict, shape) -> None:
    from woof.core.state import DTYPE

    for name, value in fields.items():
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, "
                             f"got {value.shape}")
        if value.dtype != DTYPE:
            raise TypeError(f"{name} must be float32, got {value.dtype}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")


def launch_refl10cm_morrison(qv, qr, nr, qs, ns, qg, ng, t, p, refl, *,
                             morr_rimed_ice: int = 1) -> None:
    """Morrison ``refl10cm_hm`` into ``refl``: WRF's own routine, the
    transcription in ``kernels/morrison.cu`` (``mo_refl10cm``, graded
    against WRF's Fortran with the scheme by
    ``tools/morrison_wrf461_oracle``).

    All arrays are contiguous FP32 ``(nz, ny, nx)``; ``t`` is air
    temperature (K), ``p`` full pressure (Pa), mixing ratios kg/kg and
    number concentrations kg-1 (the scheme's own prognostic moments).
    ``refl`` receives dBZ with the wrapper's -35 floor
    (module_mp_morr_two_moment.F:913-917).  ``refl.cu``'s former column
    kernel ``refl10cm_morrison_column`` stays in that byte-frozen file and
    is no longer launched.
    """
    shape = refl.shape
    if len(shape) != 3:
        raise ValueError(f"refl fields must be 3-D, got {shape}")
    _check_fields({"qv": qv, "qr": qr, "nr": nr, "qs": qs, "ns": ns,
                   "qg": qg, "ng": ng, "t": t, "p": p, "refl": refl}, shape)
    from woof.core.morrison import launch_refl10cm_wrf
    launch_refl10cm_wrf(qv, qr, nr, qs, ns, qg, ng, t, p, refl,
                        morr_rimed_ice=morr_rimed_ice)


def launch_refl10cm_kessler(qv, qr, t, p, refl) -> None:
    """Per-cell Smith-1975 rain-only reflectivity into ``refl`` (dBZ).

    Same array contract as :func:`launch_refl10cm_morrison` minus the ice
    moments; the module docstring derives the fixed-intercept form.
    """
    shape = refl.shape
    if len(shape) != 3:
        raise ValueError(f"refl fields must be 3-D, got {shape}")
    _check_fields({"qv": qv, "qr": qr, "t": t, "p": p, "refl": refl}, shape)
    ncell = int(np.prod(shape))
    # Declares no column arrays, so its own frame is 0 B either way; it comes
    # from the specialized module so one compilation of refl.cu serves the
    # process instead of two.
    kernel = _column_kernel("refl10cm_kessler_cell", shape[0])
    blocks = (ncell + _CELL_TPB - 1) // _CELL_TPB
    kernel((blocks,), (_CELL_TPB,), (qv, qr, t, p, refl, np.int32(ncell)))


#: The module whose column state is the MPAS (hex) physics seam's.  Named,
#: not imported: the seam module is byte-pinned by hex's engine manifest,
#: so the MPAS-side default lives here and the seam file stays unchanged.
MPAS_SEAM_STATE_MODULE = "woof.core.mpas_column_batch"


def refl_melting_for(state) -> bool:
    """Whether ``state``'s REFL_10CM (WSM6 or Thompson) keeps the melting term.

    1. An explicit ``state.refl10cm_from_melting`` of True or False wins
       (A/B arms, and hex's analysis-frame view, which copies the seam's
       answer).  ``None`` or a missing attribute is not explicit: a
       permissive view whose ``__getattr__`` answers ``None`` for unknown
       names must not change a diagnostic silently.
    2. Otherwise the MPAS column seam's state (class defined in
       :data:`MPAS_SEAM_STATE_MODULE`) defaults to OFF, which is NOAA's own
       MPAS default (``config_tempo_refl10cm_from_melting`` false,
       physics/Registry_tempo.xml; the RRFS workflow never sets it).
       Measured on hex conus3km 2026-10-03 12Z f01: WSM6's term added a
       median +9 dB at the composite maximum and tripled 35 dBZ coverage
       against MRMS.
    3. Every other state (ARW ``DomainState``, tiles, DA views) keeps
       WRF's ``refl10cm_wsm6`` verbatim.
    """
    explicit = getattr(state, "refl10cm_from_melting", None)
    if explicit is True or explicit is False:
        return explicit
    return type(state).__module__ != MPAS_SEAM_STATE_MODULE


#: The original name, kept because hex imports it.
wsm6_refl_melting_for = refl_melting_for

def launch_refl10cm_wsm6(qv, qr, qs, qg, t, p, refl, *,
                          hail_opt: int = 0, melting: bool = True,
                          radar_initialization="column_oracle") -> None:
    """WRF ``refl10cm_wsm6`` into dBZ.

    ``melting=True`` (the WRF/MPAS-A routine verbatim, one CUDA thread per
    column) adds the Blahak melting-particle term below the melting level.
    ``melting=False`` omits that term: rain plus dry snow and dry
    graupel/hail only, which is what NOAA's own MPAS microphysics reports
    by default (TEMPO ``config_tempo_refl10cm_from_melting = false``,
    physics/Registry_tempo.xml).  Measured on hex conus3km 2026-10-03 12Z
    f01: with WSM6's fixed minimum snow intercept at 0 C the term adds a
    median +9 dB (p90 +12 dB) at the composite maximum, tripling 35 dBZ
    coverage against MRMS.

    One CUDA thread per column, ``kernels/wsm6.cu``'s
    ``wsm6_refl10cm_column``: WRF's float32/float64 mix line by line, with
    mp_radar's own radar_init words (``wsm6_constants.WRF_RADAR_*``), graded
    against WRF v4.6.1's refl10cm_wsm6 by
    tests/test_wsm6_wrf461_column_exact.py.  Both arms run it;
    ``melting=False`` skips only the melting-particle block.
    """
    shape = refl.shape
    if len(shape) != 3:
        raise ValueError(f"refl fields must be 3-D, got {shape}")
    nz, ny, nx = shape
    _check_fields({"qv": qv, "qr": qr, "qs": qs, "qg": qg,
                   "t": t, "p": p, "refl": refl}, shape)
    import cupy as cp
    from woof.core.wsm6 import (wsm6_kernel, wsm6_radar_words_on_device,
                                 wsm6_tile_columns)
    rf, rd = wsm6_radar_words_on_device(hail_opt, initialization=radar_initialization)
    kernel = wsm6_kernel("wsm6_refl10cm_column", nz)
    # Column arrays in a workspace for the columns in flight, as
    # launch_wsm6 (kernels/wsm6.cu, COLUMN WORKSPACE): five float, four
    # double and one byte row per level per column, tiles by column offset.
    ncol = ny * nx
    tile = wsm6_tile_columns(kernel, ncol)
    if tile < ncol and tile % _COLUMN_TPB:
        # Same breakage as launch_wsm6: a partial tile's last block would
        # also write the next tile's first columns.
        raise ValueError(f"WSM6 reflectivity tile {tile} must be a multiple "
                         f"of {_COLUMN_TPB} columns when it splits the mesh")
    lanes = ((tile + _COLUMN_TPB - 1) // _COLUMN_TPB) * _COLUMN_TPB
    wsf = cp.empty(lanes * 5 * nz, dtype=cp.float32)
    wsd = cp.empty(lanes * 4 * nz, dtype=cp.float64)
    wsc = cp.empty(lanes * nz, dtype=cp.uint8)
    try:
        for lo in range(0, ncol, tile):
            count = min(tile, ncol - lo)
            kernel(
                ((count + _COLUMN_TPB - 1) // _COLUMN_TPB,), (_COLUMN_TPB,),
                (qv, qr, qs, qg, t, p, rf, rd, refl,
                 np.int32(1 if melting else 0),
                 np.int32(nz), np.int32(ny), np.int32(nx),
                 wsf, wsd, wsc, np.int32(lo)))
    finally:
        del wsf, wsd, wsc


#: WRF v4.6.1's own radar words for WDM6, as ``wdm6init`` then
#: ``radar_init`` leave them (module_mp_wdm6.F:2198-2208, module_mp_radar.F:
#: 74-190), read back from the Fortran by tools/wdm6_wrf461_oracle
#: (gfortran 13.3 -O0, glibc 2.39).  :func:`radar_init_wdm6` derives the
#: same quantities in float64, and that is not WRF's word: radar_init's
#: xam/xc?g/xorg2/xobm?/xocm? are default REAL (xcrg(4) = WGAMMA(8.) is
#: 5040.002, not 5040), PI5 is a product of REAL literals, lamda_radar =
#: 0.10 is a REAL literal held in a DOUBLE PRECISION parameter, and m_w_0
#: is the square root of a default (single) COMPLEX.  The kernel takes
#: these words; tests/test_wdm6.py checks them against the float64
#: derivation to 1e-6 relative so a typo cannot hide here.
#:
#: Float32 words in WDM6_RW_* order (kernels/wdm6_refl.cu), as bit patterns:
#: pi, xam_r, xcrg(3), xcrg(4), xorg2, xobmr, xcre(2), xcre(4), xam_s,
#: xcsg(3), xcsg(4), xcse(1), xcse(4), xocms, xobms, xbm_s, xmu_s, xam_g,
#: xcgg(3), xcgg(4), xcge(1), xcge(4), xocmg, xobmg, xbm_g, xmu_g, n0g.
_WDM6_RADAR_FLOAT_BITS = {
    0: (0x40490FDB, 0x4402E653, 0x41C00000, 0x459D8004, 0x3F800000,
        0x3EAAAAAB, 0x40000000, 0x41000000, 0x42517084, 0x40C00000,
        0x44340001, 0x40800000, 0x40E00000, 0x3E88DBAA, 0x3EAAAAAB,
        0x40400000, 0x00000000, 0x4382E653, 0x40C00000, 0x44340001,
        0x40800000, 0x40E00000, 0x3E2011F6, 0x3EAAAAAB, 0x40400000,
        0x00000000, 0x4A742400),
    1: (0x40490FDB, 0x4402E653, 0x41C00000, 0x459D8004, 0x3F800000,
        0x3EAAAAAB, 0x40000000, 0x41000000, 0x42517084, 0x40C00000,
        0x44340001, 0x40800000, 0x40E00000, 0x3E88DBAA, 0x3EAAAAAB,
        0x40400000, 0x00000000, 0x43B74274, 0x40C00000, 0x44340001,
        0x40800000, 0x40E00000, 0x3E0F1658, 0x3EAAAAAB, 0x40400000,
        0x00000000, 0x471C4000),
}
#: Binary64 words in WDM6_RD_* order: K_w, PI5, lamda4, m_w_0 (re, im),
#: m_i_0 (re, im).  Independent of hail_opt.
_WDM6_RADAR_DOUBLE_BITS = (
    0x3FEDE4B5A805B813, 0x4073204BA0000000, 0x3F1A36E305532621,
    0x402213D1A0000000, 0xBFF634D6E0000000, 0x3FFC91DAE74D3B24,
    0xBF21ACB622AE704E)


def wdm6_radar_words(hail_opt: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """WRF's float32 and binary64 radar words for WDM6 (see above)."""
    if int(hail_opt) not in _WDM6_RADAR_FLOAT_BITS:
        raise ValueError(f"WDM6 hail_opt must be 0 or 1, got {hail_opt}")
    fw = np.asarray(_WDM6_RADAR_FLOAT_BITS[int(hail_opt)],
                    dtype=np.uint32).view(np.float32)
    dw = np.asarray(_WDM6_RADAR_DOUBLE_BITS, dtype=np.uint64).view(np.float64)
    return fw, dw


@cuda_cache(maxsize=None, ready=True)
def _wdm6_device_words(hail_opt: int):
    import cupy as cp

    fw, dw = wdm6_radar_words(hail_opt)
    return cp.asarray(fw), cp.asarray(dw)


def launch_refl10cm_wdm6(qv, qr, nr, qs, qg, t, p, refl, *,
                         hail_opt: int = 0, mp_physics: int = 16) -> None:
    """One CUDA thread per column: WRF ``refl10cm_wdm6`` into dBZ.

    ``nr`` is WDM6's transported rain number moment.  Unlike Thompson's
    diagnostic, it needs no private shadow: WDM6 predicts rain number as
    state, and the driver hands the routine the same ``nr`` array the
    scheme just updated (module_mp_wdm6.F:281-296).  The radar constants
    are WRF's own words (:func:`wdm6_radar_words`); the kernel evaluates
    the routine in the Fortran's declared REAL/DOUBLE PRECISION types and
    is graded against WRF by tools/wdm6_wrf461_oracle.
    """
    shape = refl.shape
    if len(shape) != 3:
        raise ValueError(f"refl fields must be 3-D, got {shape}")
    nz, ny, nx = shape
    if nz > _KMAX:
        raise ValueError(f"nz={nz} exceeds REFL_KMAX={_KMAX}")
    _check_fields({"qv": qv, "qr": qr, "nr": nr, "qs": qs, "qg": qg,
                   "t": t, "p": p, "refl": refl}, shape)
    if int(mp_physics) == 26:
        from woof.core.kernels import get_kernel_int_defines
        words_f, words_d = _wdm6_device_words(0)
        blocks = (ny * nx + _COLUMN_TPB - 1) // _COLUMN_TPB
        kernel = get_kernel_int_defines("wdm7_refl", "refl10cm_wdm6_column",
                                        (("REFL_KMAX", _KMAX), ("WDM_HAIL", 1)))
        kernel((blocks,), (_COLUMN_TPB,),
               (qv, qr, nr, qs, qg, t, p, _device_tables(), words_f, words_d,
                refl, np.int32(nz), np.int32(ny), np.int32(nx)))
        return
    words_f, words_d = _wdm6_device_words(int(hail_opt))
    blocks = (ny * nx + _COLUMN_TPB - 1) // _COLUMN_TPB
    _column_kernel("refl10cm_wdm6_column", nz, module="wdm6_refl")(
        (blocks,), (_COLUMN_TPB,),
        (qv, qr, nr, qs, qg, t, p, _device_tables(), words_f, words_d,
         refl, np.int32(nz), np.int32(ny), np.int32(nx)))


#: The binary64 words WRF v4.6.1's radar_init leaves for the melting-
#: particle backscatter WDM5 calls (module_mp_radar.F:76-81, dumped by
#: tools/wdm5_wrf461_oracle).  m_w_0 is evaluated by m_complex_water_ray in
#: default REAL and widened, so its words carry float32 precision; the
#: float64 radar_init mirror above carries more digits than WRF has, which is
#: why these are WRF's words and not the mirror's.
WDM5_RADAR_WORDS = {
    "K_w": 0x3FEDE4B5A805B813,
    "m_w_0_re": 0x402213D1A0000000,
    "m_w_0_im": 0xBFF634D6E0000000,
    "m_i_0_re": 0x3FFC91DAE74D3B24,
    "m_i_0_im": 0xBF21ACB622AE704E,
}


def _wdm5_radar_word(name: str) -> np.float64:
    return np.frombuffer(
        np.uint64(WDM5_RADAR_WORDS[name]).tobytes(), np.float64)[0]


def launch_refl10cm_wdm5(qv, qr, nr, qs, t, p, refl) -> None:
    """One CUDA thread per column: WRF ``refl10cm_wdm5`` into dBZ.

    Rain and snow only (WDM5 has no graupel); rain's intercept comes from
    the scheme's own transported ``nr`` (module_mp_wdm5.F:2202-2208).  The
    kernel lives in ``kernels/wdm5.cu`` beside the scheme, reusing
    the family melting-particle routines; it carries the radar module's
    REAL words for WDM5's m(D) inputs itself. The native -35 dBZ floor is
    retained and included in the strict replay.
    """
    from woof.core.kernels import get_kernel, get_kernel_int_defines
    from woof.core.wdm6_constants import WDM5_SHALLOW_KMAX, wdm5_level_tier

    shape = refl.shape
    if len(shape) != 3:
        raise ValueError(f"refl fields must be 3-D, got {shape}")
    nz, ny, nx = shape
    capacity = wdm5_level_tier(nz)
    _check_fields({"qv": qv, "qr": qr, "nr": nr, "qs": qs,
                   "t": t, "p": p, "refl": refl}, shape)
    kernel = (get_kernel("wdm5", "refl10cm_wdm5_column")
              if capacity == WDM5_SHALLOW_KMAX else
              get_kernel_int_defines("wdm5", "refl10cm_wdm5_column",
                                     (("WDM5_KMAX", capacity),)))
    blocks = (ny * nx + _COLUMN_TPB - 1) // _COLUMN_TPB
    kernel(
        (blocks,), (_COLUMN_TPB,),
        (qv, qr, nr, qs, t, p, _device_tables(),
         _wdm5_radar_word("K_w"), _wdm5_radar_word("m_w_0_re"),
         _wdm5_radar_word("m_w_0_im"), _wdm5_radar_word("m_i_0_re"),
         _wdm5_radar_word("m_i_0_im"), refl,
         np.int32(nz), np.int32(ny), np.int32(nx)))


#: Pointwise classic Thompson REFL_10CM WITHOUT the melting-snow term:
#: rain from its own number moment, Field et al. snow moments and the
#: classic 400 kg m-3 graupel from the call's number shadow, formed exactly
#: as ``refl10cm_thompson_column`` in the byte-frozen ``kernels/refl.cu``
#: forms them (float arithmetic where it is float, double where double).
#: The frozen kernel's melting block is the only thing left out.
_THOMPSON_DRY_SOURCE = r"""
    const float tpi = 3.1415926536f;
    const float am_r = tpi * 1000.0f / 6.0f;
    const float am_s = 0.069f;
    const float am_g = tpi * 400.0f / 6.0f;
    const float qvk = fmaxf(1.0e-10f, qv);
    const float rho = 0.622f * p / (287.04f * t * (qvk + 0.622f));
    float zer = 1.0e-22f, zes = 1.0e-22f, zeg = 1.0e-22f;
    if (qr > 1.0e-12f) {
        const float rr = qr * rho;
        const float nr_vol = fmaxf(1.0e-6f, nr * rho);
        const double lamr = pow((double)(am_r * 6.0f * nr_vol / rr), 1.0 / 3.0);
        const double n0r = (double)nr_vol * lamr;
        zer = (float)(n0r * 720.0 * pow(1.0 / lamr, 7.0));
    }
    if (qs > 1.0e-6f) {
        const float rs = qs * rho;
        const float tc = fminf(-0.1f, t - 273.15f);
        const float smo2 = __fdiv_rn(rs, am_s);
        const float mo = 4.0f, tc2 = tc * tc, m2 = mo * mo;
        const float loga = 5.065339f - 0.062659f * tc - 3.032362f * mo
            + 0.029469f * tc * mo - 0.000285f * tc2 + 0.31255f * m2
            + 0.000204f * tc2 * mo + 0.003199f * tc * m2 + 0.0f * tc2 * tc
            - 0.015952f * m2 * mo;
        const float b = 0.476221f - 0.015896f * tc + 0.165977f * mo
            + 0.007468f * tc * mo - 0.000141f * tc2 + 0.060366f * m2
            + 0.000079f * tc2 * mo + 0.000594f * tc * m2 + 0.0f * tc2 * tc
            - 0.003577f * m2 * mo;
        const float smoz = powf(10.0f, loga) * powf(smo2, b);
        zes = (0.176f / 0.93f) * (6.0f / tpi) * (6.0f / tpi)
            * (am_s / 900.0f) * (am_s / 900.0f) * smoz;
    }
    if (qg > 1.0e-6f) {
        const float rg = qg * rho;
        const float ng_vol = fmaxf(1.0e-6f, ng * rho);
        const double lamg = pow((double)(am_g * 6.0f * ng_vol / rg), 1.0 / 3.0);
        const double n0g = (double)ng_vol * lamg;
        zeg = (float)((0.176 / 0.93) * (6.0 / (double)tpi) * (6.0 / (double)tpi)
            * ((double)am_g / 900.0) * ((double)am_g / 900.0) * n0g * 720.0
            * pow(1.0 / lamg, 7.0));
    }
    const float zsum = zer + zes + zeg;
    dbz = fmaxf(-35.0f, (float)(10.0 * log10((double)zsum * 1.0e18)));
"""


@lru_cache(maxsize=1)
def _thompson_dry_kernel():
    import cupy as cp

    return cp.ElementwiseKernel(
        "float32 qv, float32 qr, float32 nr, float32 qs, float32 qg, "
        "float32 ng, float32 t, float32 p",
        "float32 dbz", _THOMPSON_DRY_SOURCE, "refl10cm_thompson_dry")


def launch_refl10cm_thompson(
        qv, qr, nr, qs, qg, graupel_number_shadow, t, p, refl, *,
        melting: bool = True) -> None:
    """Classic Thompson ``calc_refl10cm`` into dBZ.

    ``graupel_number_shadow`` is WRF classic mp=8's private, per-call ng1d
    moment after sources and number fallout.  It is consumed only here and is
    deliberately absent from the transported/restart state.

    ``melting=True`` is WRF's routine verbatim (one CUDA thread per column).
    ``melting=False`` omits the Blahak melting-snow term, which NOAA's MPAS
    TEMPO ships off by default (``config_tempo_refl10cm_from_melting``):
    rain, dry Field snow and dry graupel only, pointwise.
    """
    shape = refl.shape
    if len(shape) != 3:
        raise ValueError(f"refl fields must be 3-D, got {shape}")
    nz, ny, nx = shape
    if nz > _KMAX:
        raise ValueError(f"nz={nz} exceeds REFL_KMAX={_KMAX}")
    _check_fields({
        "qv": qv, "qr": qr, "nr": nr, "qs": qs, "qg": qg,
        "graupel_number_shadow": graupel_number_shadow,
        "t": t, "p": p, "refl": refl,
    }, shape)
    if not melting:
        _thompson_dry_kernel()(qv, qr, nr, qs, qg, graupel_number_shadow,
                               t, p, refl)
        return
    rc = radar_init()
    blocks = (ny * nx + _COLUMN_TPB - 1) // _COLUMN_TPB
    _column_kernel("refl10cm_thompson_column", nz)(
        (blocks,), (_COLUMN_TPB,),
        (qv, qr, nr, qs, qg, graupel_number_shadow, t, p,
         _device_tables(), np.float64(rc.k_w),
         np.float64(rc.m_w_0.real), np.float64(rc.m_w_0.imag),
         np.float64(rc.m_i_0.real), np.float64(rc.m_i_0.imag),
         refl, np.int32(nz), np.int32(ny), np.int32(nx)))


class NativeReflectivityScheme(ValueError):
    """The active scheme produces REFL_10CM, but not through this module.

    A ``ValueError`` subclass so every existing caller keeps catching it,
    and a NAMED type so a caller that can use the scheme's own field --
    a radar observation operator, say -- can tell "this scheme has no dBZ"
    apart from "this scheme's dBZ is on the state already".
    """


#: Schemes deliberately OUTSIDE :func:`compute_refl_10cm`'s dispatch even
#: though every one of them produces REFL_10CM, with the reason each is out.
#:
#: The PRODUCER set below (1, 6, 8, 10, 16, 28) and the CONSUMER set
#: ``REFL_10CM_MICROPHYSICS`` (1, 6, 8, 9, 10, 16, 18, 28, 50) differ on
#: purpose, and these keys are exactly the difference: WRF computes their
#: dBZ inside the scheme call and woof stashes the SCHEME's own field, so
#: there is no generic operator call to make.  The asymmetry was already
#: asserted arithmetically in
#: ``tests/test_mp28_runtime_reachability.py::
#: test_every_refl_10cm_admission_constant_admits_28``; it is recorded HERE,
#: at the gate, because a decision that lives only in a test reaches the
#: user as an omission -- which is what the mp=28 reachability audit found
#: and what this table exists to stop repeating.
#:
#: Adding any of these keys to the gate tuple is a DEFECT, not a widening:
#: none of them has a routine here to dispatch to, and the fall-through arm
#: is Kessler's rain-only Marshall-Palmer form.
# These schemes never write a forecast reflectivity diagnostic in WRF.
# Keep this separate from schemes that compute and stash their own field.
SCHEME_ABSENT_REFL_10CM: dict[int, str] = {
    96: "WRF MADWRF_MP is empty and never writes REFL_10CM",
    95: "WRF phys/module_mp_etanew.F writes no REFL_10CM diagnostic",
    0: "no microphysics, no reflectivity",
    11: "WRF CAMMGMP never writes REFL_10CM; its initialized zero field is retained",
}

SCHEME_NATIVE_REFL_10CM: dict[int, str] = {
    27: "UDM writes reflectivity in udm_step; no shared reflectivity kernel.",
    29: 'WRF 4.7.1 RCON computes reflectivity inside rcon_step and stashes its completed field; no shared refl kernel.',
    5: (
        "Ferrier-Aligo computes a reflectivity column inside EGCP01COLUMN "
        "(module_mp_fer_hires.F:2232-2251 snow/graupel Zsnow with its "
        "rime-factor number blend, :2339-2345 rain Zrain from the scheme's "
        "own N0r and mean diameter, :2494-2496 DBZ_col) from moments only "
        "the scheme holds at that point, and EGCP01DRV discards it, so WRF "
        "v4.6.1 never writes REFL_10CM under mp=5.  woof keeps that "
        "column: kernels/fer_hires.cu writes it on output steps and "
        "woof/core/fer_hires.py stashes it into this module's refl_10cm "
        "slot.  The scheme carries one total-ice mass and no qs/qg, so no "
        "branch here could read its state.  Note the floor: Ferrier's "
        "clear-air value is DBZmin = -20 dBZ (:1191), not -35."
    ),
    30: "FAST_SBM spectral polar-scattering diagnostics require the native bin operator; the generic bulk operator cannot reproduce it",
    24: (
        "WSM7 uses its own wsm7 translation unit via the WSM-family "
        "table runner. The runner fills the refl_10cm slot on output-due "
        "steps, including the declared hail Rayleigh term."
    ),
    55: "The adaptive-habit column produces REFL_10CM directly in WRF driver operation order",
    7: 'NU-WRF saticel_s computes hail-inclusive DBZ during its state update; '
       'module_mp_gsfcgce_4ice_nuwrf.F:4271-4318 and 4493-4503.',
    56: (
        "NTU multi-moment: WRF's NTU arm writes no REFL_10CM at all "
        "(module_microphysics_driver.F:1295-1319 binds no reflectivity "
        "argument), so the field is WOOF's own operator over the "
        "scheme's predicted gamma distributions -- each category's "
        "shape and slope from its own SOLVE_AFA* routine, Rayleigh sixth "
        "moments, ice dielectric factor and predicted density for the "
        "frozen categories (woof.core.ntu.reflectivity, kernel "
        "ntu_refl_cell).  It floors at -35 dBZ."
    ),
    9: (
        "Milbrandt-Yau two-moment carries Zet as an INOUT dummy that WRF's "
        "driver binds straight to refl_10cm (module_microphysics_driver.F:"
        "1878), so the scheme writes the slot on every call and "
        "woof/core/milbrandt2.py stashes that array rather than calling a "
        "generic operator.  Its Z block updates nothing, so it is also "
        "available as a standalone H(x) -- "
        "woof.core.milbrandt2.reflectivity over "
        "kernels/milbrandt2_zet.cu, which woof/da/obsop.py dispatches "
        "mp=9 to before reaching this module, exactly as it does mp=18.  "
        "Note the floor: Milbrandt-Yau's clear-air value is -99 dBZ "
        "(MY2_minZET, kernels/milbrandt2.cu:308), NOT the -35 dBZ this "
        "dispatcher's family floors at."
    ),
    18: (
        "NSSL two-moment has its own S-band diagnostic, radardd02 "
        "(module_mp_nssl_2mom.F), reading five ice categories, five number "
        "moments and two volume moments; woof ports it in "
        "woof.core.nssl2_diagnostics and dispatches to it directly "
        "(woof/da/obsop.py routes mp=18 there before reaching this "
        "module).  It also floors at 0 dBZ, not -35."
    ),
    50: (
        "P3 one-category has no standalone reflectivity routine anywhere "
        "to dispatch to: WRF computes dBZ INSIDE p3_main -- ze_rain from "
        "the scheme's own mu_r/lamr rain PSD (module_mp_p3.F:4755), ze_ice "
        "from the lookup-table normalised reflectivity moment f1pr13 read "
        "at the ice-PSD indices (:4789, accumulated :4868), and diag_ze = "
        "10*log10((ze_rain+ze_ice)*1e18) (:4887) -- and the driver binds "
        "diag_zdbz_3d straight to refl_10cm in CASE(P3_1CATEGORY), passing "
        "no do_radar_ref argument at all "
        "(module_microphysics_driver.F:1557-1602).  P3 predicts ONE ice "
        "category with a rime pair (qir rime mass, qib rime volume) and "
        "allocates NO qs and NO qg, so every branch this dispatcher owns "
        "except Kessler would ask the state for fields P3 never has, and "
        "the Kessler fall-through would report rain-only Rayleigh Z for an "
        "ice-bearing column -- the mp=28-on-Kessler class of defect, in "
        "the reflectivity field this time.  woof mirrors WRF: "
        "woof/core/p3.py stashes the scheme's own zdbz into this module's "
        "refl_10cm slot on every output-due step.  Note the floor: P3's "
        "clear-air value is -99 dBZ (module_mp_p3.F:2273, left standing by "
        "the hydrometeor-free skip at :2439/:3972; mirrored at "
        "woof/core/p3.py:513 and woof/core/kernels/p3.cu:626), NOT the "
        "-35 dBZ this dispatcher's family floors at, so a caller "
        "differencing H(x) against a clear-air observation must take the "
        "floor from the scheme and not from here."
    ),
    97: (
        "Goddard GCE (and every row of woof.core.mp_column_schemes) runs "
        "its own refl10cm_gsfc inside the microphysics call, as WRF's "
        "gsfcgce does (module_mp_gsfcgce.F:2866-2904, refl10cm_gsfc "
        ":2942-3115): fixed-intercept exponential rain, snow and "
        "graupel-or-hail PSDs with the scheme's own densities, launched "
        "from its own translation unit (kernels/gsfcgce_refl.cu) and "
        "stashed into this module's refl_10cm slot.  Its clear-air floor "
        "is this family's -35 dBZ.  Its H(x) between steps is "
        "woof.core.mp_column_schemes.reflectivity."
    ),
    53: (
        "P3 three-moment ice (CASE(P3_1CAT_3MOM)) is mp=50's p3_main with Z "
        "prognostic: dBZ is computed INSIDE its final-diagnostics loop "
        "(module_mp_p3.F:4795-4887, ze_ice from the 3momI table's "
        "reflectivity column at the solved mu_i) and the driver binds "
        "diag_zdbz_3d straight to refl_10cm "
        "(module_microphysics_driver.F:1739).  woof/core/p3.py stashes "
        "that field exactly as it does for mp=50, with the same -99 dBZ "
        "clear-air floor."
    ),
}
# mp=51 (CASE (P3_1CATEGORY_NC), module_microphysics_driver.F:1607-1647)
# binds diag_zdbz_3d to refl_10cm exactly as mp=50 does, and its Z reads
# rain and ice only (the predicted droplet number does not enter it), so the
# mp=50 sentence is its sentence.
SCHEME_NATIVE_REFL_10CM[51] = SCHEME_NATIVE_REFL_10CM[50]
SCHEME_NATIVE_REFL_10CM[52] = "P3 two-category computes reflectivity within its shared category diagnostics; use the P3 native operator to include both ice populations"

SCHEME_NATIVE_REFL_10CM[15] = SCHEME_NATIVE_REFL_10CM[5].replace("under mp=5", "under mp=5 or 15")


from woof.microphysics_schemes import NAMED_SCHEMES as _NAMED_MP_SCHEMES
# Named schemes whose capability row says they compute their own REFL_10CM
# (woof.microphysics_schemes): the adapter stashes the scheme's field.
SCHEME_NATIVE_REFL_10CM.update({
    _s.mp_id: (f"{_s.label} computes REFL_10CM inside its own column call; "
               "its adapter stashes that field (capability row "
               f"woof.microphysics_schemes.NAMED_SCHEMES[{_s.name!r}])")
    for _s in _NAMED_MP_SCHEMES.values() if _s.native_reflectivity})

#: The state fields :func:`compute_refl_10cm` READS for each scheme it
#: dispatches, keyed by ``mp_physics``.  ONE table, published, instead of
#: the four ``missing = [...]`` literals the dispatch used to carry inline:
#: a scheme's reflectivity inputs are a row here, the dispatch below
#: resolves its required-field check through this row, and every other
#: reader of "what does reflectivity need from the state" -- the streamed
#: composite (tilestream) that duplicated Morrison's six-moment list and
#: refused every other scheme on it, the physics registry's per-option
#: ``reflectivity_input_species`` row, the agreement test -- reads the same
#: row.  Water vapour is a genuine input (the Rayleigh sums take rho, and
#: rho takes qv); temperature and pressure are supplied by the caller and
#: are not state species, so they are not listed.
#:
#: The keys are exactly the PRODUCER set the gate below admits.  A scheme in
#: :data:`SCHEME_NATIVE_REFL_10CM` has no row here on purpose: it computes
#: Z itself and there is nothing to read for it.  ``tests/test_refl.py``
#: and ``tests/test_authority_agreement.py`` hold the three sets (this
#: table, the native table, the registry's implemented microphysics) to
#: their arithmetic.
REFL_10CM_INPUT_SPECIES: dict[int, tuple[str, ...]] = {
    # Kessler rain-only Marshall-Palmer form.
    1: ("qv", "qr"),
    2: ("qv", "qr", "qs", "qg"),
    # WSM3: WRF computes none.  woof's own diagnostic is refl10cm_wsm6 on
    # WSM3's phase split of qr (rain at or above 0 C, snow below; see
    # wsm3_precipitation_phases), whose intercepts and densities are
    # WSM3's own.
    3: ("qv", "qr"),
    # WSM6: rain, snow, graupel masses (``launch_refl10cm_wsm6``).
    6: ("qv", "qr", "qs", "qg"),
    # WSM6R: WRF's wsm6r computes no REFL_10CM; woof diagnoses it with
    # refl10cm_wsm6, whose PSD assumptions are WSM6R's own PARAMETERs.
    106: ("qv", "qr", "qs", "qg"),
    # Thompson: rain mass AND number, snow and graupel masses; the classic
    # graupel-number shadow is a same-call argument, not a state species.
    8: ("qv", "qr", "nr", "qs", "qg"),
    # Morrison: three two-moment species (``launch_refl10cm_morrison``).
    10: ("qv", "qr", "nr", "qs", "ns", "qg", "ng"),
    # WDM5: rain mass and number, snow mass (``launch_refl10cm_wdm5``).
    14: ("qv", "qr", "nr", "qs"),
    40: ("qv", "qr", "nr", "qs", "ns", "qg", "ng"),
    # WDM6: Thompson's list with WDM6's own kernel (``launch_refl10cm_wdm6``).
    16: ("qv", "qr", "nr", "qs", "qg"),
    # WDM7: refl10cm_wdm7 is refl10cm_wdm6 byte for byte
    # (module_mp_wdm7.F:3316-3490) and reads no hail; the same list and
    # kernel at WDM6's hail_opt = 0 radar constants, which are WDM7's.
    26: ("qv", "qr", "nr", "qs", "qg"),
    # Aerosol-aware Thompson: mp=8's list exactly -- see the docstring of
    # compute_refl_10cm for why nc/nwfa/nifa contribute nothing to Z.
    28: ("qv", "qr", "nr", "qs", "qg"),
    # SBU-YLin.  WRF computes no REFL_10CM for mp=13; the scheme's author
    # states how a reflectivity post-processor must read it: snow is
    # (1-Ri)*qs and graupel is Ri*qs (module_mp_sbu_ylin.F:12-13).  The
    # operator is the refl10cm_wsm6 routine over that partition, so rimi
    # is an input species.
    13: ("qv", "qr", "qs", "rimi"),
    38: ("qv", "qr", "nr", "qs", "qg", "ng", "qvolg"),
}

from woof.core.column_schemes import COLUMN_SCHEMES
REFL_10CM_INPUT_SPECIES.update({mp: row.refl_species for mp, row in COLUMN_SCHEMES.items()})


def refl_10cm_input_species(mp_physics: int) -> tuple[str, ...]:
    """The state species reflectivity reads for ``mp_physics``.

    Raises :class:`NativeReflectivityScheme` for a scheme that produces its
    own Z (so the caller learns where the field IS instead of what it
    lacks), and ``ValueError`` for a selector with no reflectivity
    formulation at all.
    """
    key = int(mp_physics)
    if key in REFL_10CM_INPUT_SPECIES:
        return REFL_10CM_INPUT_SPECIES[key]
    native = SCHEME_NATIVE_REFL_10CM.get(key)
    if native is not None:
        raise NativeReflectivityScheme(
            f"mp_physics={key} computes REFL_10CM itself and reads no "
            "operator inputs. " + native + "  Read the scheme's own dBZ out "
            "of the DomainState 'refl_10cm' scratch slot instead.")
    raise ValueError(
        f"mp_physics={key} has no reflectivity formulation; the operator "
        f"dispatches {sorted(REFL_10CM_INPUT_SPECIES)} and the native-Z "
        f"schemes are {sorted(SCHEME_NATIVE_REFL_10CM)}")


def _missing_refl_inputs(state, mp_physics: int) -> list[str]:
    """The species of :data:`REFL_10CM_INPUT_SPECIES` this state lacks.

    ``qv`` is checked by the caller's ``state.qv is None`` guard with its
    own moist-state message, so it is not repeated here.
    """
    return [name for name in refl_10cm_input_species(mp_physics)
            if name != "qv" and getattr(state, name, None) is None]


def compute_refl_10cm(
        state, cfg, *, temperature=None, pressure=None,
        thompson_graupel_number=None):
    """Compute REFL_10CM (dBZ, FP32 ``(nz, ny, nx)``).

    The microphysics adapters pass BOTH ``temperature`` and ``pressure``:
    respectively the scheme's post-call temperature and the unchanged
    prepared pressure that WRF passes as ``t1d``/``p1d`` at
    module_mp_morr_two_moment.F:913-914.  Omitting both retains the useful
    standalone diagnostic over the current state, but production history
    output uses the explicit microphysics-time pair (PROVENANCE.md D2).
    Dispatches like ``microphysics.apply``: 1 = Kessler fallback, 6 =
    WSM6 ``refl10cm_wsm6``, 8 and 28 = Thompson ``calc_refl10cm``,
    10 = Morrison ``refl10cm_hm``, 14 = WDM5 ``refl10cm_wdm5``, and 16 =
    WDM6 ``refl10cm_wdm6``, the WDM pair whose
    rain intercept is diagnosed from the scheme's own prognostic ``nr``
    rather than fixed.  Thompson requires its output-due private
    graupel-number scratch through ``thompson_graupel_number``.  Results
    land in the persistent
    ``refl_10cm`` scratch slot and are returned.

    The admitted set is NARROWER than the set of schemes that produce
    REFL_10CM, and the difference is recorded rather than implied: 9, 18
    and 50 compute their own dBZ inside the scheme call and refuse here BY
    NAME through :data:`SCHEME_NATIVE_REFL_10CM`, which also says where
    their field actually is.

    mp=28 shares the mp=8 branch's INPUTS, and that is WRF's own
    structure, not a convenience; on the v4.6.1 generation it evaluates them
    with mp=28's own WRF-exact transcription of the same routine
    (``thompson_aa_refl10cm``), because refl.cu's classic kernel is
    byte-frozen for mp=8 and its arithmetic is not WRF's (CUDA's math
    library, 720 for crg(4), binary64 slopes and radar constants).
    ``calc_refl10cm``
    (module_mp_thompson.F:5710-6028) is ONE routine with no
    ``is_aerosol_aware`` branch, reached from the single call site
    ``mp_gt_driver:1458`` -- which is itself gated only on
    ``diagflag .and. do_radar_ref == 1`` (:1449), never on the scheme.
    Its complete argument list
    (:5710-5711) is ``qv1d, qc1d, qr1d, nr1d, qs1d, qg1d, ng1d, qb1d, t1d,
    p1d, dBZ, kts, kte, ii, jj, ke_diag``: no ``nc1d``, no ``nwfa``, no
    ``nifa`` anywhere in the routine's 319 lines.  Cloud water enters only
    as ``rc(k) = MAX(R1, qc1d(k)*rho(k))`` at :5764, and ``rc`` is READ
    NOWHERE ELSE in the routine -- it is a dead local, so no cloud-water and
    no droplet-number information reaches any of the four Rayleigh sums.
    The prognostic droplet number that distinguishes mp=28 from mp=8
    therefore contributes EXACTLY ZERO to REFL_10CM in WRF, and a future
    "improvement" folding cloud water or nc into the sum would be a
    divergence from WRF, not a refinement.  Pinned by
    ``tests/test_mp28_runnable.py::
    test_refl_10cm_is_bit_identical_under_two_very_different_nc_fields``.
    """
    import cupy as cp

    from woof.core import constants as c
    from woof.core.state import DTYPE

    if cfg.mp_physics not in REFL_10CM_INPUT_SPECIES:
        native = SCHEME_NATIVE_REFL_10CM.get(int(cfg.mp_physics))
        if native is not None:
            # NOT "no reflectivity": the scheme already produced its own,
            # and saying otherwise is how a reachability gap reads as a
            # missing feature.  Name it, and name where the field is.
            raise NativeReflectivityScheme(
                f"mp_physics={int(cfg.mp_physics)} computes REFL_10CM "
                "itself and is deliberately not dispatched here. "
                + native + "  Read the scheme's own dBZ out of the "
                "DomainState 'refl_10cm' scratch slot, which its adapter "
                "fills on every output-due microphysics step, instead of "
                "recomputing it from another scheme's particle-size "
                "distributions.")
        admitted = ", ".join(str(selector) for selector in sorted(REFL_10CM_INPUT_SPECIES))
        raise ValueError("do_radar_ref needs an active microphysics scheme "
                         f"(mp_physics {admitted}), got {cfg.mp_physics}")
    if state.qv is None:
        raise ValueError("reflectivity requires a moist state")
    if (temperature is None) != (pressure is None):
        raise ValueError("temperature and pressure must be supplied together")
    nz, ny, nx = state.p.shape
    if temperature is None:
        thb = state.thb if state.thb.ndim == 3 else state.thb[:, None, None]
        t = state.scratch((nz, ny, nx), "refl_t")
        p = state.p
        # Standalone-current-state path.  Production output passes the
        # microphysics adapter's post-call T and prepared p explicitly.
        t[...] = (state.total_theta() if getattr(state, "use_theta_m", 0)
                  else thb + state.thp) * cp.power(
            p / DTYPE(c.P0), DTYPE(c.RCP))
    else:
        t = temperature
        p = pressure
    refl = state.scratch((nz, ny, nx), "refl_10cm")
    if cfg.mp_physics in COLUMN_SCHEMES:
        from woof.core.microphysics import launch_column_refl
        row = COLUMN_SCHEMES[cfg.mp_physics]
        missing = _missing_refl_inputs(state, cfg.mp_physics)
        if missing:
            raise ValueError(f"{row.label} reflectivity lacks fields: " + ", ".join(missing))
        launch_column_refl(row, [getattr(state, n) for n in row.refl_species], t, p, refl)
    elif cfg.mp_physics == 40:
        from woof.core.morrison_aero import launch_refl_morrison_aero
        launch_refl_morrison_aero(
            state.qv, state.qr, state.nr, state.qs, state.ns, state.qg, state.ng,
            t, p, refl, morr_rimed_ice=cfg.morr_rimed_ice)
    elif cfg.mp_physics == 10:
        missing = _missing_refl_inputs(state, 10)
        if missing:
            raise ValueError("mp_physics=10 reflectivity lacks Morrison "
                             "moments: " + ", ".join(missing))
        launch_refl10cm_morrison(state.qv, state.qr, state.nr,
                                 state.qs, state.ns, state.qg, state.ng,
                                 t, p, refl,
                                 morr_rimed_ice=cfg.morr_rimed_ice)
    elif cfg.mp_physics in (8, 28, 38):
        # ONE branch for both Thompson schemes.  See the docstring: WRF's
        # calc_refl10cm has no aerosol-aware arm and reads no droplet
        # number, so a separate mp=28 branch could only ever differ from
        # WRF.  The required-field list is identical too -- mp=28 allocates
        # a strict superset of mp=8's state.
        missing = _missing_refl_inputs(state, int(cfg.mp_physics))
        if missing:
            raise ValueError(
                f"mp_physics={cfg.mp_physics} reflectivity lacks Thompson "
                "fields: " + ", ".join(missing))
        if cfg.mp_physics == 38 and thompson_graupel_number is None:
            thompson_graupel_number = state.ng
        if thompson_graupel_number is None:
            raise ValueError(
                f"mp_physics={cfg.mp_physics} reflectivity requires the "
                "same-call classic graupel number shadow: classic "
                "Thompson carries no graupel number in the Registry, its "
                "wrapper diagnoses one at call entry and evolves it "
                "through the call's source and fallout tendencies, and "
                "WRF's calc_refl10cm reads THAT moment, so a caller "
                "inside the call has to hand it over rather than let a "
                "re-diagnosis stand in for it.  A caller BETWEEN steps "
                "has no evolved moment and needs none: "
                "woof.da.obsop.simulated_reflectivity derives the "
                "wrapper's own diagnosis from the state and passes it "
                "here.")
        if (cfg.mp_physics in THOMPSON_AEROSOL_FAMILY and getattr(
                cfg, "thompson_version", "wrf_461") == "wrf_461"):
            # The same WRF routine, through mp=28's own transcription
            # (thompson_aerosol_state.cu, thompson_aa_refl10cm): WOOF's
            # own math routines and WRF's REAL(4)/DOUBLE PRECISION
            # arithmetic, measured against WRF v4.6.1 by the 0 ULP column
            # oracle.  refl.cu's classic kernel is byte-frozen for mp=8.
            from woof.core.thompson_aerosol_state import launch_aa_refl10cm, stash_hail_size
            from woof.core.thompson_aerosol_launch import hail_aware_scope
            hail = THOMPSON_AEROSOL_FAMILY[cfg.mp_physics].hail_aware
            with hail_aware_scope(hail):
                launch_aa_refl10cm(
                    state.qv, state.qr, state.nr, state.qs, state.qg,
                    thompson_graupel_number, t, p, refl,
                    melting=refl_melting_for(state),
                    graupel_volume=getattr(state, "qvolg", None))
                if hail:
                    stash_hail_size(state)
        else:
            launch_refl10cm_thompson(
                state.qv, state.qr, state.nr, state.qs, state.qg,
                thompson_graupel_number, t, p, refl,
                melting=refl_melting_for(state))
    elif cfg.mp_physics == 14:
        missing = _missing_refl_inputs(state, 14)
        if missing:
            raise ValueError("mp_physics=14 reflectivity lacks WDM5 fields: "
                             + ", ".join(missing))
        launch_refl10cm_wdm5(state.qv, state.qr, state.nr, state.qs,
                             t, p, refl)
    elif cfg.mp_physics in (16, 26):
        mp = int(cfg.mp_physics)
        missing = _missing_refl_inputs(state, mp)
        if missing:
            raise ValueError(f"mp_physics={mp} reflectivity lacks "
                             f"{'WDM6' if mp == 16 else 'WDM7'} fields: "
                             + ", ".join(missing))
        # WDM7 has no hail_opt: its radar constants are WDM6's hail_opt = 0
        # set (wdm7init sets xam_g from deng = 500, module_mp_wdm7.F:2499).
        launch_refl10cm_wdm6(state.qv, state.qr, state.nr, state.qs,
                             state.qg, t, p, refl,
                             hail_opt=cfg.wdm6_hail_opt if mp == 16 else 0, mp_physics=mp)
    elif cfg.mp_physics == 13:
        missing = _missing_refl_inputs(state, 13)
        if missing:
            raise ValueError("mp_physics=13 reflectivity lacks SBU-YLin "
                             "fields: " + ", ".join(missing))
        # The author's partition (module_mp_sbu_ylin.F:12-13): without it
        # every rimed particle would radiate as low-density snow and the
        # echo would be underestimated, which is the note's own warning.
        snow = state.scratch((nz, ny, nx), "refl_sbu_snow")
        graupel = state.scratch((nz, ny, nx), "refl_sbu_graupel")
        cp.multiply(state.rimi, state.qs, out=graupel)
        cp.subtract(state.qs, graupel, out=snow)
        launch_refl10cm_wsm6(
            state.qv, state.qr, snow, graupel, t, p, refl,
            hail_opt=0, melting=refl_melting_for(state))
    elif cfg.mp_physics == 2:
        from woof.core.lin import launch_refl10cm_lin
        launch_refl10cm_lin(state.qv, state.qr, state.qs, state.qg, t, p, refl)
    elif cfg.mp_physics == 3:
        rain = state.scratch((nz, ny, nx), "refl_wsm3_rain")
        snow = state.scratch((nz, ny, nx), "refl_wsm3_snow")
        no_graupel = state.scratch((nz, ny, nx), "refl_wsm3_graupel")
        rain[...], snow[...] = wsm3_precipitation_phases(cp, state.qr, t)
        no_graupel.fill(DTYPE(0.0))
        launch_refl10cm_wsm6(
            state.qv, rain, snow, no_graupel, t, p, refl, hail_opt=0,
            melting=refl_melting_for(state))
    elif cfg.mp_physics in (6, 106):
        mp = int(cfg.mp_physics)
        missing = _missing_refl_inputs(state, mp)
        if missing:
            raise ValueError(f"mp_physics={mp} reflectivity lacks WSM6 "
                             "fields: " + ", ".join(missing))
        # The melting-particle term is a property of the state's owner,
        # not of RunConfig: an ARW DomainState keeps WRF's routine
        # verbatim, while the MPAS column seam's state takes the MPAS-side
        # default, off.  See wsm6_refl_melting_for.
        # WSM6R's graupel constants are WSM6's hail_opt = 0 set
        # (module_mp_wsm6r.F:7,16-18: n0g 4e6, avtg 330, bvtg 0.8, deng 500).
        launch_refl10cm_wsm6(
            state.qv, state.qr, state.qs, state.qg, t, p, refl,
            hail_opt=cfg.wsm6_hail_opt if mp == 6 else 0,
            melting=refl_melting_for(state),
            # A strict (GPUWM_WRF_EXACT=1) run replays the stock scalar
            # build's radar_init words; a default run keeps the column
            # oracle's, as 2.9.1 did.
            radar_initialization=("scalar_build" if _wrf_exact_enabled()
                                  else "column_oracle"))
    else:
        launch_refl10cm_kessler(state.qv, state.qr, t, p, refl)
    return refl



def _wrf_exact_enabled() -> bool:
    from woof.wrf_exact import ENABLED
    return bool(ENABLED)

def wsm3_precipitation_phases(xp, qr, t):
    """WSM3's own phase split of its one precipitation class: ``(rain, snow)``.

    WRF computes no REFL_10CM for mp_physics=3 (the driver binds no
    refl_10cm argument; module_diag_nwp.F:549 "nothing to do here"), so
    woof's REFL_10CM for WSM3 is its own diagnostic, declared in
    woof/core/wsm3.py: WRF's refl10cm_wsm6 evaluated with ``qr`` as rain
    where the temperature is at or above 0 C and as snow below it -- the
    split module_mp_wsm3.F makes everywhere it reads qr (slope_wsm3
    :1010, effectRad_wsm3's input :172-180) -- and no graupel.  WSM6's
    rain intercept (8e6), snow intercept with its temperature factor
    (2e6 * exp(0.12 * (T0 - T)) capped at 1e11) and snow density (100) are
    WSM3's own constants, so the operator reads WSM3's size distributions.
    """
    warm = t >= t.dtype.type(273.15)
    zero = qr.dtype.type(0.0)
    return xp.where(warm, qr, zero), xp.where(warm, zero, qr)


def stash_refl_10cm(state, refl) -> None:
    """Stash one output-due reflectivity field on the physics driver.

    A second unconsumed field is a cadence bug, so overwrites fail loudly.
    The array itself is state-owned ``refl_10cm`` scratch; the driver holds
    only the one-frame handoff reference.
    """
    driver = getattr(state, "physics", None)
    if driver is None:
        raise RuntimeError("cannot stash REFL_10CM without a physics driver")
    if driver.refl_10cm is not None:
        raise RuntimeError("REFL_10CM stash was not consumed before reuse")
    driver.refl_10cm = refl


def refl_10cm_is_stashed(state) -> bool:
    """Return the D2 handoff state without exposing its array to callers."""
    driver = getattr(state, "physics", None)
    return driver is not None and driver.refl_10cm is not None


def refl_10cm_stash_is_due(ticks: int, *, domain_start_ticks: int = 0
                           ) -> bool:
    """Does a history frame at ``ticks`` consume a microphysics stash?

    ONE predicate for every consume site, because the answer is a
    property of the handoff and not of the route asking.  A stash exists
    only after a microphysics step OF THAT DOMAIN has run with
    ``refl_10cm_due`` (:func:`stash_refl_10cm` is reachable from nowhere
    else), so the frame due at the domain's own start tick -- its
    analysis frame, before any of its steps -- consumes nothing.

    ``domain_start_ticks`` is that tick, on the domain's own clock
    (``DomainClock.spec.start_ticks``).  It is 0 for a domain that starts
    with the experiment and the ACTIVATION EPOCH for a nest that starts
    later.  Reading the absolute 0 there is defect #205: an activating
    nest's first frame consumed a stash no step had produced, and the
    run died at that frame with "REFL_10CM output is due but no
    microphysics-time field is stashed" -- deterministically, hours of
    integration in.
    """
    return int(ticks) != int(domain_start_ticks)


def domain_start_ticks_of(node) -> int:
    """The start tick of ``node``'s own clock, 0 when it carries none.

    The tree routes hand :class:`woof.core.model.DomainNode` objects to
    the history seam and every one of those carries a
    :class:`woof.core.clock.DomainClock`.  A node without one is a
    single-domain or hand-built caller, whose domain starts with the
    experiment by construction -- the same 0 the field defaults to.
    """
    clock = getattr(node, "clock", None)
    spec = getattr(clock, "spec", None)
    # DFL starts its ordinary forecast at the filtered midpoint. No
    # forecast microphysics stash exists at that new analysis instant.
    return (int(getattr(spec, "start_ticks", 0) or 0)
            + int(getattr(clock, "dfi_start_ticks", 0) or 0))


def analysis_refl_10cm(state, cfg=None, *, shape=None, array_module=None):
    """Diagnose analysis reflectivity from its hydrometeors, before stepping.

    History uses the same scheme diagnostic as later output. Host/store
    states are staged as bounded horizontal slabs on the GPU. The output
    retains the history owner's host/device placement and prognostic fields
    are never advanced.

    A streamed domain's fields are read from its store
    (:func:`woof.core.streaming.domain_field`), never from the state the
    model still holds: under a store-direct domain that state is a
    slab-height template, so reading it diagnosed one slab's rows and
    broadcast them over the domain (tests/test_verify_base4_review.py).

    Under ``GPUWM_WRF_EXACT=1`` the frame is WRF's own initial array, all
    zero: ``refl_10cm`` is history-only (Registry ``hdu``) and only the
    microphysics driver fills it; measured on the stock 4.6.1 combo
    recording (iowa-convective A088, WSM6, do_radar_ref = 1), frame 0 is an
    all-zero array. HRRR's GSD path instead calls its first-step Thompson
    helper when gsd_diagnostics and do_radar_ref are enabled (mp=8/28).
    """
    import numpy as np
    from woof.wrf_exact import ENABLED

    streamed = getattr(state, "_streamed_domain", None) is not None
    if shape is None:
        shape = state.qv.shape
    shape = tuple(shape)
    host = streamed or array_module is np or isinstance(state.qv, np.ndarray)
    cfg = cfg if cfg is not None else getattr(
        getattr(state, "physics", None), "refl_config", None)
    from woof.core.gsd_initial_reflectivity import enabled as gsd_initial_enabled
    initial_gsd = gsd_initial_enabled(cfg)
    # NU-WRF 4-ice computes DBZ inside its state update, so there is no
    # between-step diagnostic to run before that first update.  Its native
    # WRF analysis array is zero (Registry hdu; real MP7 capture at step 1).
    # Keep that initial value in both arithmetic modes rather than sending
    # MP7 to the standalone operator, which correctly refuses this scheme.
    # CAMMGMP never writes REFL_10CM, at analysis or on later steps.
    native_inactive_initial = getattr(cfg, "mp_physics", None) in (7, 11)
    # Native mp=38 leaves its history-only radar field at the initialized
    # zero until a diagnostic-due microphysics call. Its real initialization
    # also leaves number/volume moments zero, so evaluating a PSD against
    # those unanalyzed moments here produces spurious extreme echoes.
    # UDM (27) and RCON (29) also start with an initialized-zero history
    # radar field. Their retained first driver inputs witness those words;
    # module_mp_udm.F:374-395 fills it only after the microphysics call.
    # Keep these selectors' native initial-history contract in fast mode.
    native_mp38_initial = int(getattr(cfg, "mp_physics", 0)) in (27, 29, 38)
    if native_inactive_initial or ((ENABLED or native_mp38_initial) and not initial_gsd):
        if host:
            return np.zeros(shape, dtype=np.float32)
        import cupy as cp
        return cp.zeros(shape, dtype=cp.float32)
    import cupy as cp
    from types import SimpleNamespace
    from woof.core.state_reflectivity import diagnose

    if cfg is None:
        cfg = state.physics.refl_config
    if initial_gsd:
        from woof.core.gsd_initial_reflectivity import diagnose
    if not host:
        # The scheme diagnostics write the resident state's shared
        # ``refl_10cm`` scratch, which the forecast microphysics later fills
        # on the columns it computes.  The boundary ring is not one of them,
        # so leaving the analysis values there froze the edge reflectivity at
        # t0 for the whole run.  The frame is a copy; the scratch goes back to
        # WRF's initial zeros for every cell microphysics never writes.
        diagnosed = diagnose(state, cfg)
        frame = diagnosed.copy()
        diagnosed.fill(0.0)
        return frame
    if streamed:
        from woof.core.streaming import domain_field

        def field(name):
            return domain_field(state, name, setup=(name == "thb"))
    else:
        def field(name):
            return getattr(state, name, None)
    result = np.empty(shape, dtype=np.float32)
    # GPU diagnostic kernels own the data path. Python only stages slabs.
    ny = shape[1]
    for y0 in range(0, ny, 64):
        y1 = min(y0 + 64, ny)
        slab = SimpleNamespace()
        for name in ("p", "thb", "thp", "alt", "qv", "qc", "qr", "qi", "qs", "qg", "qh",
                     "nr", "ni", "ns", "ng", "nh", "nc", "qir", "qib", "qirv", "qhw", "qhl",
                     "volg", "f_rimef", "qrimef", "qzi", "qi2", "qi3", "ni2", "ni3",
                     "qvoli", "qvoli2", "qvoli3", "qaoli", "qaoli2", "qaoli3", "qnr", "qni", "qns", "qng", "qnh", "qvolg", "qvolh"):
            value = field(name)
            if value is not None:
                if value.ndim == 3:
                    value = value[:, y0:y1, :]
                setattr(slab, name, cp.ascontiguousarray(cp.asarray(value)))
            else:
                setattr(slab, name, None)
        buffers = {}
        def scratch(size, name):
            if name not in buffers:
                buffers[name] = cp.empty(size, dtype=cp.float32)
            return buffers[name]
        slab.scratch = scratch
        slab.use_theta_m = int(getattr(state, "use_theta_m", 0))
        if slab.use_theta_m:
            from woof.core import constants as c
            def dry_theta():
                base = slab.thb if slab.thb.ndim == 3 else slab.thb[:, None, None]
                return (base + slab.thp) / (cp.float32(1.0) + cp.float32(c.RVOVRD) * slab.qv)
            slab.total_theta = dry_theta
        slab.refl10cm_from_melting = refl_melting_for(state)
        result[:, y0:y1, :] = cp.asnumpy(diagnose(slab, cfg))
    return result


def consume_refl_10cm(state):
    """Consume and clear the one-frame REFL_10CM handoff exactly once."""
    driver = getattr(state, "physics", None)
    if driver is None or driver.refl_10cm is None:
        raise RuntimeError(
            "REFL_10CM output is due but no microphysics-time field is stashed")
    refl = driver.refl_10cm
    driver.refl_10cm = None
    return refl


def compute_and_stash_refl_10cm(
        state, cfg, temperature, pressure, *,
        thompson_graupel_number=None) -> None:
    """Compute from WRF's microphysics-time inputs and stage for output."""
    from woof.wrf_exact import ENABLED
    from woof.core.physics_inventory import WRF_INACTIVE_REFL_MICROPHYSICS
    if (cfg.mp_physics == 11 or
            ENABLED and cfg.mp_physics in WRF_INACTIVE_REFL_MICROPHYSICS):
        # Stock Kessler never writes the forecast radar slot. Retain its
        # real initialized value. CAMMGMP also never writes this slot and
        # has no standalone radar operator, under either arithmetic mode.
        refl = state.scratch(state.p.shape, "refl_10cm")
        refl.fill(0.0)
        stash_refl_10cm(state, refl)
        return
    stash_refl_10cm(
        state, compute_refl_10cm(
            state, cfg, temperature=temperature, pressure=pressure,
            thompson_graupel_number=thompson_graupel_number))


# ---------------------------------------------------------------------------
# AGREEMENT WITH THE REGISTRY, AT IMPORT.  ``REFL_10CM_INPUT_SPECIES`` and
# ``SCHEME_NATIVE_REFL_10CM`` are the source of every option's
# ``consumers.reflectivity_input_species`` / ``reflectivity_native_reason``
# rows; this holds them to the generated copy.  Every implemented
# microphysics selector must be in exactly one of the two tables (mp=0,
# which has no reflectivity at all, is the one cited absence), which is the
# arithmetic tests/test_refl.py and tests/test_mp28_runtime_reachability.py
# have always asserted -- now against the registry rather than a literal.
def _require_agreement_with_the_registry() -> None:
    from woof.physics_registry import require_consumer_rows_agreement

    observed = {mp: list(species)
                for mp, species in REFL_10CM_INPUT_SPECIES.items()}
    observed.update({mp: None for mp in SCHEME_NATIVE_REFL_10CM})
    require_consumer_rows_agreement(
        "woof.core.refl (REFL_10CM_INPUT_SPECIES, SCHEME_NATIVE_REFL_10CM)",
        "microphysics", "reflectivity_input_species", observed,
        project=lambda row: None if row is None else list(row),
        cited_absences=SCHEME_ABSENT_REFL_10CM)


_require_agreement_with_the_registry()

from woof.config import THOMPSON_AEROSOL_FAMILY
