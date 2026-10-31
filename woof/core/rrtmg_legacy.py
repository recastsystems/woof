# ======================================================================
# THIRD-PARTY NOTICE.  Parts of this file are hand transcriptions of
# third-party work.  ArWen distributes the file under the Apache License
# 2.0; the notices below belong to the transcribed parts and are kept here
# because their own licences require it.  Full texts are in the repository
# NOTICE and in the licenses/ directory.
#
#   RRTMG longwave/shortwave, transcribed from WRF v4.6.1
#   phys/module_ra_rrtmg_lw.F and phys/module_ra_rrtmg_sw.F, which carry
#   AER's own notice seven and nine times respectively:
#
#       Copyright 2002-2008, Atmospheric & Environmental Research, Inc. (AER).
#       This software may be used, copied, or redistributed as long as it is
#       not sold and this copyright notice is reproduced on each copy made.
#       This model is provided as is without any express or implied warranties.
#                             (http://www.rtweb.aer.com/)
#
#   ArWen takes this material under AER's own current grant instead: BSD
#   3-Clause, "Copyright (c) 2020, Atmospheric and Environmental
#   Research", published by AER at github.com/AER-RC/RRTMG_LW and
#   .../RRTMG_SW.  Text in licenses/LICENSE-AER-RRTMG-BSD-3-Clause.txt and
#   beside the packaged coefficients in woof/data/wrf_radiation/.
# ======================================================================
"""Forecast adapter for the exact port of WRF v4.6.1's bundled RRTMG.

``RRTMGLegacyRadiation`` serves ``RunConfig.ra_rrtmg_variant =
"rrtmg_legacy"`` on the resolved 4/4 radiation pair.  It is pure wiring:
every FP statement it performs is either (a) one of the bitwise-gated
building blocks (``rrtmg_legacy_prep`` batch twins, the device McICA
twin, the batched LW/SW CUDA engines, the WRF-level output mappings,
``woof.ingest.wrf_ozone``), (b) the documented driver-side seams that
have no bitwise oracle (``radconst``/``calc_coszen`` FP32 transcription,
phy_prep's ``t8w`` construction, ``cal_cldfra1`` imported from
``woof.core.rrtmgp``), or (c) the WRF radiation driver's own two output
lines (grid-wide GSW/RTHRATENSW zeroing and ``SWDOWN = GSW/(1-ALBEDO)``,
``module_radiation_driver.F:1721,1738,2877``).  The design authority is
``docs/rrtmg_legacy_integration.md``; the booby-trap boundary (section
10) is what tests/test_rrtmg_legacy_wiring.py proves: a legacy-selected
call must run the device McICA twin and the batched CUDA engines, never
the NumPy compute chain.

The implemented option envelope (dossier section 1) is pinned here and
fails closed on anything else: ``icloud=1 or 2``, ``cldovrlp=2`` (McICA
maximum-random), ``idcor=0``, ``o3input in {0, 2}``, ``ghg_input=0``,
``aer_opt=0``, ``swint_opt=0``.  ``o3input=0`` selects the wrapper's
O3DATA branch; ``o3input=2`` selects CAM climatology.

Night contract (dossier section 3): the driver-level zeroing is realized
by allocating RTHRATENSW/GSW as zeros and writing only day columns; the
wrapper-level night write (COSZR + the SW_NIGHT_ZEROED list) still runs
through ``swrad_night_outputs_batch`` so the wrapper contract stays
exercised even though none of those diagnostics reach
:class:`RadiationResult`.  WRF zeroes the SWUPT.. optional-diagnostic
bundle only IF PRESENT (module_ra_rrtmg_sw.F:11568-11589); woof's
contract is that the full bundle is ALWAYS materialized (the night
helper returns every member unconditionally).

Radii seam (dossier sections 9.3, 11): woof state ``effc/effi/effs``
carry MICRONS; the wrapper takes meters, so the adapter multiplies by
``F(1e-6)`` exactly once at this boundary.  Meter-scale state (a restart
written before the Thompson radii-units fix) is rejected with a
migration pointer, never silently rescaled.

Ozone routing: WRF v4.7.1's EM_CORE guards BOTH halves of the
``o3input = 2`` chain on ``id == 1`` -- ``oznini`` interpolates the
packaged climatology to a domain's own XLAT under
``o3input .EQ. 2 .AND. id .EQ. 1`` (module_physics_init.F:2203-2212), and
``ozn_time_int``/``ozn_p_int`` run under the same guard
(module_radiation_driver.F:1801-1823).  A NEST evaluates neither; it
receives the root's field through the parent-to-child forcing stream the
Registry declares on the variable itself (``rdf=(p2c)``,
Registry/Registry.EM_COMMON:1264).  The decision is "is this domain a
nest?", not "is this grid fine?".

The adapter mirrors that structure and takes its word for what it did
from :mod:`woof.core.cam_ozone`:

* A ROOT adapter (``ozone_parent=None``) runs the bit-ported
  ``woof.ingest.wrf_ozone`` chain on its own grid and RETAINS the
  resulting field: ``ozone_routing = 'root-climatology'``.
* A resident NEST takes an ``ozone_parent=`` provider
  (:class:`ParentOzoneProvider`, wired by ``runtime.prepare_child_case``)
  and obtains the parent's most recent retained field horizontally
  interpolated onto the child grid through woof's certified SINT
  mass-point operator (``woof.core.nest_interp.sint``); it never invokes
  the climatology chain.  ``ozone_routing = 'parent-interpolated'``.  The
  horizontal-interpolation ARITHMETIC is woof's own SINT
  transliteration, the same documented seam class as every other
  nest-interpolation arithmetic difference; the routing (root-compute +
  parent->child interpolation, the child's field updating when the
  parent's does) matches WRF.
* An OFFLINE child (``woof.offline_child_run``, the ndown-equivalent
  route) has no resident parent to interpolate from and is configured as
  a WRF root: ``specified = true``, ``nested = false``, ``parent_id = 0``,
  lateral boundaries read from a file.  WRF's own answer for such a
  domain is the climatology on its own grid, so the route builds a root
  adapter and declares ``ozone_routing = 'child-grid-climatology'`` -- the
  same arithmetic, named so a report says the ozone came from the
  climatology on the child grid and not from the parent.

SW aerosol: with ``aer_opt = 0`` the batched CUDA SW engine builds WRF's
neutral optics (tauaer 0 / ssaaer 1 / asmaer 0, module_ra_rrtmg_sw.F:
11333-11460) internally, exactly as before.  With ``aer_opt = 3`` (the
operational HRRR fork's Thompson aerosol branch) the adapter forms the
per-band optics on the device from the radiation inputs and the Thompson
numbers state.nwfa/state.nifa for each day-column chunk
(woof.core.rrtmg_aerosol_optics, held word for word to the fork's
gt_aod and calc_aerosol_rrtmg_sw) and hands them to the engine, which
feeds them to spcvmc unchanged (iaer = 10).  aer_opt=1 uses the WRF six-type
climatology and iaer=6 device optics on the root domain; aer_opt=2 takes the
user AOD/Angstrom/ssa/asymmetry inputs (woof.core.aer_opt2).  The longwave takes no aerosol in that fork
under any aer_opt without chemistry.

Sequencing/VRAM shape: the LW and SW pipelines run SEQUENTIALLY per
adapter chunk -- each chunk's prep outputs, device McICA slabs, and
engine transients are dropped before the next stage allocates -- so the
transient device peak is the MAX over the four phases (LW generate, LW
engine, SW generate, SW engine) plus the McICA slabs held as engine
inputs, never the LW+SW sum.  :func:`legacy_radiation_vram_bytes` prices
exactly that shape.
"""

from __future__ import annotations

from woof.core.device_cache import cuda_cache

from datetime import datetime, timedelta
from functools import lru_cache, partial
from pathlib import Path

import numpy as np

from woof.core.mynn_radiation import (
    MERGE_CLDFRA_BL_ABOVE,
    MERGE_QC_BELOW,
    MERGE_QI_BELOW,
    mynn_bl_cloud_active,
    wrf_itimestep,
)

from woof.core import rrtmg_legacy_prep as _prep
from woof.core import rrtmg_legacy_device as _prep_device
from woof.core import rrtmg_lw as _lw
from woof.core import rrtmg_mcica as _mcica
from woof.core import rrtmg_sw as _sw
from woof.core import rrtmg_aerosol_optics as _aer3
from woof.core import aer_opt2 as _aer2

__all__ = [
    "MAX_LONGWAVE_LAYERS", "MAX_SHORTWAVE_LAYERS",
    "ParentOzoneProvider", "RRTMGLegacyRadiation",
    "legacy_radiation_layer_counts", "legacy_radiation_vram_bytes",
    "radconst", "calc_coszen", "solar_geometry_device",
]

F = np.float32
MAX_LONGWAVE_LAYERS = _lw.MAX_RADIATION_LAYERS
MAX_SHORTWAVE_LAYERS = _sw.MAX_RADIATION_LAYERS


def legacy_radiation_layer_counts(
        nz: int, p_top: float) -> tuple[int, int]:
    """Return exact first-call legacy RRTMG ``(LW, SW)`` layer counts."""

    return (
        int(_prep.compute_lw_nlayers(int(nz) + 1, float(p_top))),
        int(nz) + 1,
    )

#: WRF share/module_model_constants.F: piconst (default REAL) and the
#: compile-time parameter folds DEGRAD = piconst/180., DPD = 360./365.
_PICONST = F(3.1415926535897932384626433)
_DEGRAD = F(_PICONST / F(180.0))
_DPD = F(F(360.0) / F(365.0))

#: The pinned WRF option combination this adapter implements (dossier
#: section 1).  A config carrying any other explicit value fails closed.
#: swint_opt is not here: it is a radiation-DRIVER option the physics
#: driver carries (woof.core.swint) and the adapter's call is the same
#: for 0 and 1.  aer_opt = 3 (Thompson aerosol shortwave optics) is
#: implemented by this adapter; validate_radiation_driver_options admits
#: 0, 1, 2 and 3; resident-nest aer_opt=1 refuses with its missing parent forcing path.
_PINNED_OPTIONS = {"cldovrlp": 2, "idcor": 0}
#: ghg_input 0 (fixed gases, CO2 a REAL(4) function of year) and 1 (the
#: CAMtr table re-interpolated at every call, woof.core.camtr_gases).
_ADMITTED_GHG_INPUT = (0, 1)
_ADMITTED_AER_OPT = (0, 1, 2, 3)


# ---------------------------------------------------------------------------
# Solar geometry: FP32 statement transcription of radconst / calc_coszen
# (module_radiation_driver.F:3469-3541).  numpy transcendentals stand in
# for the preserved wrf_461 host calculations (NumPy's float32 sin/cos/arcsin
# round differently from the libm gfortran calls: COSZEN up to 64 ULP off on
# the column oracle). The corrected wrf_480 consumers use the shared device
# sun and its word-for-word source oracle, and a strict (GPUWM_WRF_EXACT=1)
# wrf_461 run takes solar_geometry_device below (WOOF's own float32 trig,
# kernels/rrtmg_legacy_solar.cu), which matches WRF bit for bit.
# ---------------------------------------------------------------------------

def radconst(julian):
    """(declin, solcon) from WRF ``radconst`` at the FP32 julian."""
    j = F(julian)
    obecl = F(23.5) * _DEGRAD
    sinob = np.sin(obecl)
    if j >= F(80.0):
        sxlong = _DPD * (j - F(80.0))
    else:
        sxlong = _DPD * (j + F(285.0))
    sxlong = sxlong * _DEGRAD
    arg = sinob * np.sin(sxlong)
    declin = np.arcsin(arg)
    djul = (j * F(360.0)) / F(365.0)
    rjul = djul * _DEGRAD
    two = F(2.0) * rjul
    eccfac = F(1.000110) + F(0.034221) * np.cos(rjul)
    eccfac = eccfac + F(0.001280) * np.sin(rjul)
    eccfac = eccfac + F(0.000719) * np.cos(two)
    eccfac = eccfac + F(0.000077) * np.sin(two)
    solcon = F(1370.0) * eccfac
    return F(declin), F(solcon)


def calc_coszen(julian, xtime, gmt, xlat, xlon, declin, *, generation="wrf_480"):
    """WRF ``calc_coszen`` per point; returns float32 coszen (n,).

    ``xtime`` is the minutes value WRF passes (the driver hands
    ``xtime + radt*0.5`` for the interval-midpoint hour angle while
    ``julian``/``declin`` stay at call time; driver lines 1206-1208).
    """
    return calc_coszen_hrang(julian, xtime, gmt, xlat, xlon, declin, generation=generation)[0]


def calc_coszen_hrang(julian, xtime, gmt, xlat, xlon, declin, *, generation="wrf_480"):
    """WRF ``calc_coszen``'s two outputs, ``(coszen, hrang)``, float32.

    HRANG is the hour angle WRF stores beside COSZEN (grid%hrang,
    module_radiation_driver.F:3534); the slope-radiation adjustment reads
    both (module_surface_driver.F TOPO_RAD_ADJ).
    """
    j = F(julian)
    from woof.core.solar import solar_generation, solar_geometry_device
    solar_generation(generation)
    if generation == "wrf_480":
        import cupy as cp
        _ds, cz, ha = solar_geometry_device(julian, xtime, gmt, xlat, xlon,
            generation=generation, declination=declin)
        return cp.asnumpy(cz), cp.asnumpy(ha)
    xt = F(xtime)
    g = F(gmt)
    d = F(declin)
    xlat = np.asarray(xlat, np.float32)
    xlon = np.asarray(xlon, np.float32)
    da = (F(6.2831853071795862) * (j - F(1.0))) / F(365.0)
    two = F(2.0) * da
    eot = F(0.000075) + F(0.001868) * np.cos(da)
    eot = eot - F(0.032077) * np.sin(da)
    eot = eot - F(0.014615) * np.cos(two)
    eot = eot - F(0.04089) * np.sin(two)
    eot = eot * F(229.18)
    xt24 = np.fmod(xt, F(1440.0)) + eot          # Fortran MOD on reals
    tloctm = (g + xt24 / F(60.0)) + xlon / F(15.0)
    hrang = (F(15.0) * (tloctm - F(12.0))) * _DEGRAD
    xxlat = xlat * _DEGRAD
    coszen = (np.sin(xxlat) * np.sin(d)
              + (np.cos(xxlat) * np.cos(d)) * np.cos(hrang))
    return (np.minimum(np.maximum(coszen, F(-1.0)), F(1.0)).astype(np.float32),
            np.asarray(hrang, dtype=np.float32))


def solar_geometry_device(julian, xtime, gmt, xlat, xlon):
    """WRF v4.6.1 ``radconst`` + ``calc_coszen`` on the device.

    ``kernels/rrtmg_legacy_solar.cu`` holds the two routines statement for
    statement with WOOF's own float32 sinf, cosf and asinf and explicit
    single-rounded arithmetic, so the result is WRF's Fortran bit for bit
    (tests/test_rrtmg_legacy_column_oracle.py).  ``xtime`` is the minutes
    value WRF passes (``xtime + radt*0.5`` on a radiation call).  ``xlat``
    and ``xlon`` are float32 device or host arrays of one shape.

    Returns ``(coszen, hrang, declin, solcon)``: two flat float32 device
    arrays and two float32 scalars.  :func:`radconst` and
    :func:`calc_coszen_hrang` remain the NumPy statement twins.
    """
    import cupy as cp
    from woof.core.kernels import load_module

    mod = load_module("rrtmg_legacy_solar")
    lat = cp.ascontiguousarray(cp.asarray(xlat, dtype=cp.float32).reshape(-1))
    lon = cp.ascontiguousarray(cp.asarray(xlon, dtype=cp.float32).reshape(-1))
    if lat.shape != lon.shape:
        raise ValueError("solar geometry latitude/longitude shapes must match")
    pair = cp.empty(2, dtype=cp.float32)
    from woof.wrf_exact import ENABLED as exact
    if exact:
        # WRF's scalar intrinsic has its own binary32 rounding contract.
        # Share the already-qualified terrain boundary, using the computed
        # device argument. Default device math and its routines are unchanged.
        from woof.static.rust_bridge import scalar_reference_asin_f32
        argument = cp.empty(1, dtype=cp.float32)
        mod.get_function("rls_radconst_with_argument")(
            (1,), (1,), (F(julian), _DEGRAD, _DPD, pair, argument))
        native = scalar_reference_asin_f32(cp.asnumpy(argument))
        pair[0] = native[0]
    else:
        mod.get_function("rls_radconst")(
            (1,), (1,), (F(julian), _DEGRAD, _DPD, pair))
    n = int(lat.size)
    coszen = cp.empty(n, dtype=cp.float32)
    hrang = cp.empty(n, dtype=cp.float32)
    if n:
        mod.get_function("rls_calc_coszen")(
            ((n + 127) // 128,), (128,),
            (np.int64(n), lat, lon, F(julian), F(xtime), F(gmt), _DEGRAD,
             np.int32(0), pair, coszen, hrang))   # solar_480 = 0: WRF 4.6.1
    declin, solcon = (F(v) for v in cp.asnumpy(pair))
    return coszen, hrang, declin, solcon


# ---------------------------------------------------------------------------
# t8w: phy_prep transcription (module_big_step_utilities_em.F:4904-4936).
# Interior interfaces use the fnm/fnp half->full weights; surface/top are
# z-linear extrapolations.  Dossier section 9.4: matched semantics seam.
# ---------------------------------------------------------------------------

def _t8w_columns(t3d, z_at_w, fnm, fnp):
    """t8w (ncol, nz+1) from t3d (ncol, nz) and z_at_w (ncol, nz+1)."""
    t3d = np.asarray(t3d, np.float32)
    z_at_w = np.asarray(z_at_w, np.float32)
    fnm = np.asarray(fnm, np.float32)
    fnp = np.asarray(fnp, np.float32)
    ncol, nz = t3d.shape
    # z at half levels: z = 0.5*(z_at_w(k) + z_at_w(k+1))
    z = (F(0.5) * (z_at_w[:, :-1] + z_at_w[:, 1:])).astype(np.float32)
    t8w = np.empty((ncol, nz + 1), np.float32)
    # interior full levels k = 2..kde-1 (1-based): fzm(k)*t(k)+fzp(k)*t(k-1)
    t8w[:, 1:nz] = (fnm[None, 1:nz] * t3d[:, 1:nz]
                    + fnp[None, 1:nz] * t3d[:, 0:nz - 1]).astype(np.float32)
    # bottom: z-linear extrapolation from the two lowest half levels
    w1 = ((z_at_w[:, 0] - z[:, 1]) / (z[:, 0] - z[:, 1])).astype(np.float32)
    w2 = (F(1.0) - w1).astype(np.float32)
    t8w[:, 0] = (w1 * t3d[:, 0] + w2 * t3d[:, 1]).astype(np.float32)
    # top: z-linear extrapolation from the two highest half levels
    w1 = ((z_at_w[:, nz] - z[:, nz - 2])
          / (z[:, nz - 1] - z[:, nz - 2])).astype(np.float32)
    w2 = (F(1.0) - w1).astype(np.float32)
    t8w[:, nz] = (w1 * t3d[:, nz - 1] + w2 * t3d[:, nz - 2]).astype(
        np.float32)
    return t8w


# ---------------------------------------------------------------------------
# VRAM pricing (model preflight + the wiring accuracy gate).
# ---------------------------------------------------------------------------

#: WRF v4.6.1 use_mp_re scheme table, FIRST BLOCK: the
#: ``module_physics_init.F:988-1024`` disjunction, which sets
#: ``has_reqc = has_reqi = has_reqs = 1`` together for every scheme it
#: names -- which mp schemes hand their effective radii to RRTMG.
#: Keyed by mp_physics for every scheme woof can select; None-lookup
#: fails closed in __call__.
#:
#: THIS DICT IS HALF THE ANSWER.  WRF's SECOND block (:1027-1033) then
#: re-zeroes ``has_reqs`` for the P3 family and Jensen-Ishmael, so the
#: has_req triple is NOT uniform for every scheme and cannot be spelled
#: as one boolean.  That block is :data:`_MP_SNOW_RADII_SUPPRESSED`;
#: read the two together through :func:`legacy_scheme_has_req`, which is
#: what the adapter calls.  Morrison (10) is deliberately False --
#: WRF does NOT couple Morrison radii to RRTMG (its EFFI bound of
#: 525 um would trip cldprmc's [5,140] fatal); Kessler (1) and the
#: no-microphysics case (0) have no radii.  NSSL 2-moment (18) is True
#: for the campaign's nssl_2moment_on=1 configuration (the only NSSL
#: form woof runs).
_MP_DECLARES_RADII = {
    30: False,
    0: False,
    1: False,
    2: False,
    # WSM3.  WSM3SCHEME is in the use_mp_re disjunction
    # (module_physics_init.F:1008) and not in the P3 has_reqs=0 override,
    # so all three radii reach RRTMG; effectRad_wsm3 clamps them to
    # [2.51, 50], [10.01, 125] and [25, 999] um (module_mp_wsm3.F:1553-1571).
    3: True,
    4: True,     # WSM5, module_physics_init.F use_mp_re
    6: True,     # WSM6
    # WSM6R: NOT in WRF's use_mp_re list, so has_reqc = 0 and RRTMG
    # computes its own radii, as WRF does.
    106: False,
    8: True,     # Thompson
    # Milbrandt-Yau: NOT in WRF's use_mp_re list either
    # (module_physics_init.F:1004-1023 names THOMPSON, THOMPSONAERO,
    # NSSL_2MOM, the WSM/WDM family, nuwrf4ice, Jensen-Ishmael and the P3
    # family, and no MILBRANDT2MOM), and the scheme's own effective-radius
    # block is commented out (module_mp_milbrandt2mom.F:3362/:3364/
    # :3372/:3374), so there is nothing to declare and RRTMG computes its
    # own radii exactly as it does under any has_reqc=0 scheme.
    9: False,
    10: False,   # Morrison two-moment: NOT in WRF's use_mp_re list
    # WDM6.  WRF lists WDM6SCHEME explicitly in the use_mp_re disjunction
    # (module_physics_init.F:1013, beside WSM6SCHEME at :1010) and the
    # P3/Jensen-Ishmael has_reqs=0 override (:1027-1033) does not name it,
    # so has_reqc = has_reqi = has_reqs = 1.  The radii themselves come from
    # effectRad_wdm6 (module_mp_wdm6.F:3135-3234), which CLAMPS every one
    # of them: cloud to [2.51, 50] um (:3212), ice to [10.01, 125] um
    # (:3220), snow to [25, 999] um (:3229).  The ice bound is what matters
    # here -- cldprmc's [5, 140] um fatal is the reason Morrison (EFFI to
    # 525 um) is False above, and WDM6's 125 um cap sits inside it.
    16: True,    # WDM6 (WDM6SCHEME)
    18: True,    # NSSL 2-moment (nssl_2moment_on=1)
    # WSM7.  module_physics_init.F:1011 lists WSM7SCHEME in the use_mp_re
    # disjunction beside WSM6SCHEME and the P3 has_reqs=0 override does
    # not name it; effectRad_wsm7 (module_mp_wsm7.F:2964-3064) clamps
    # the radii exactly as WSM6's does (ice to [10.01, 125] um).
    24: True,    # WSM7 (WSM7SCHEME)
    # Thompson aerosol-aware.  WRF lists it EXPLICITLY and SEPARATELY from
    # THOMPSON in the same disjunction:
    #   module_physics_init.F:1005  config_flags%mp_physics .eq. THOMPSON
    #   module_physics_init.F:1006  config_flags%mp_physics .eq. THOMPSONAERO
    # (verified in wrf-stock-v461-gate-20260721; THOMPSONAERO is
    # mp_physics==28 per Registry/Registry.EM_COMMON:3036).  It is not
    # excluded by the P3/Jensen-Ishmael has_reqs=0 override at :1027-1033
    # either, so all three of has_reqc/has_reqi/has_reqs are 1.
    #
    # The radii themselves are already the mp=8 pair plus the prognostic-nc
    # dependence: woof's mp=28 adapter calls calc_effectRad and then applies
    # mp_gt_driver's OWN clamps (:1475-1477, MAX(RE_QC_BG, MIN(re, 50.E-6))
    # and the 125/999 um pair) -- the identical clamps mp=8 takes, because
    # mp_gt_driver is ONE driver serving both packages.  So the upper bounds
    # that keep Morrison out of this table (cldprmc's [5,140] um fatal) are
    # satisfied by construction for mp=28 exactly as they are for mp=8.
    28: True,    # Thompson aerosol-aware (THOMPSONAERO)
    # P3 one-category.  WRF names P3_1CATEGORY in the use_mp_re
    # disjunction (module_physics_init.F:1017, beside THOMPSON at :1005),
    # so has_reqc = has_reqi = 1 and P3's OWN predicted radii are what
    # radiation sees: module_microphysics_driver.F's P3_1CATEGORY arm
    # (:1557) binds diag_effc_3d -> re_cloud and diag_effi_3d -> re_ice
    # (:1597-1598) unconditionally, and woof's adapter writes both every
    # step in the micron convention (woof/core/p3.py:1821-1825 CPU,
    # :1979-1980 device), over the 10/25 um backgrounds
    # woof/core/state.py:494-495 seeds from module_mp_p3.F:2279-2281.
    #
    # has_reqs is a SEPARATE answer and is 0 -- see
    # _MP_SNOW_RADII_SUPPRESSED below.  P3 is the first scheme in this
    # table whose WRF triple is not uniform, which is why the value here
    # is only the first block's flag and the adapter reads both.
    #
    # The ice bound that keeps Morrison out is not in play.  Under the
    # resulting (1, 1, 0) the wrapper forces reice1d = 10 um and routes
    # P3's ice radius into the SNOW slot, where WRF clamps it to 130 um
    # with an area-conserving mass reduction before the optics
    # (module_ra_rrtmg_lw.F:12519-12522), so cldprmc's [5, 140] fatal is
    # unreachable however large P3's lookup-table effective radius grows
    # (the shipped p3_lookupTable_1.dat-v5.4_2momI reaches 2.1e4 um).
    50: True,    # P3 one-category (P3_1CATEGORY)
    # Ferrier-Aligo (FER_MP_HIRES) is absent from the use_mp_re
    # disjunction (module_physics_init.F:1004-1023): has_reqc/i/s stay 0
    # and the wrappers compute their own radii, as for Kessler.
    5: False,
}


from woof.microphysics_schemes import NAMED_SCHEMES as _NAMED_MP_SCHEMES
# Named schemes declare radii (their adapters write effc/effi/effs).
_MP_DECLARES_RADII.update({_s.mp_id: True for _s in _NAMED_MP_SCHEMES.values()})

def _require_agreement_with_the_registry() -> None:
    """``_MP_DECLARES_RADII`` is the registry's ``cloud_optics.legacy_declares_radii``.

    Pulled by tools/build_registry.py and held equal here at import, so
    a scheme with no row is a plan-review refusal rather than the
    NotImplementedError :func:`legacy_scheme_declares_radii` raises at
    the first radiation call.
    """

    from woof.physics_registry import require_consumer_rows_agreement

    require_consumer_rows_agreement(
        "woof.core.rrtmg_legacy._MP_DECLARES_RADII", "microphysics",
        "cloud_optics", dict(_MP_DECLARES_RADII),
        project=lambda row: row["legacy_declares_radii"])


_require_agreement_with_the_registry()


def legacy_scheme_declares_radii(mp_physics, use_mp_re):
    """WRF v4.6.1's ``use_mp_re`` gate around its scheme table.

    FIRST BLOCK ONLY (``module_physics_init.F:988-1024``): True means the
    scheme is named in the disjunction, i.e. ``has_reqc = has_reqi = 1``.
    It is NOT the has_reqs answer -- WRF's :1027-1033 override re-zeroes
    has_reqs for the P3 family, so a caller that needs the flags the
    wrapper actually takes must use :func:`legacy_scheme_has_req`.
    """
    table_declares = _MP_DECLARES_RADII.get(int(mp_physics))
    if table_declares is None:
        raise NotImplementedError(
            f"mp_physics={int(mp_physics)} has no has_req* entry in the "
            "WRF v4.6.1 use_mp_re scheme table carried by the legacy RRTMG "
            "adapter; extend _MP_DECLARES_RADII from "
            "module_physics_init.F:985-1024 rather than guessing")
    value = int(use_mp_re)
    if value not in (0, 1):
        raise ValueError(f"use_mp_re must be 0 or 1, got {use_mp_re!r}")
    return bool(value) and table_declares


#: WRF v4.6.1 use_mp_re scheme table, SECOND BLOCK: the P3 /
#: Jensen-Ishmael override at ``module_physics_init.F:1027-1033``, which
#: re-zeroes ``has_reqs`` AFTER the first block set all three flags to 1.
#: These are the schemes whose ice is ONE category with no separate snow
#: species, so there is no snow radius for them to declare: the mp=50
#: package is ``moist:qv,qc,qr,qi`` with ``state:re_cloud,re_ice`` and no
#: qs and no re_snow (``Registry/Registry.EM_COMMON:3038``).
#:
#: The resulting ``(has_reqc, has_reqi, has_reqs) = (1, 1, 0)`` is not a
#: degenerate corner: it is the SOLE trigger of the wrapper's "special
#: case for P3 microphysics" (``module_ra_rrtmg_lw.F:12250-12261``,
#: ``module_ra_rrtmg_sw.F:10853``), which moves the single ice category
#: into the SNOW optics slot -- resnow1d = MAX(10., re_ice*1e6),
#: QS1D = raw QI3D, QI1D = 0, reice1d = 10 -- and which this port already
#: carries at ``woof/core/rrtmg_legacy_prep.py:464-471`` (and its device
#: twin at :1322-1329).  Until mp=50 entered the table above, no adapter
#: state could produce that triple and the transcribed branch was
#: unreachable.
#:
#: Of WRF's P3/Jensen-Ishmael family only mp=50 is selectable here:
#: mp=51 and mp=52 are refused by name in
#: ``woof.config._P3_UNPORTED_VARIANTS`` and Jensen-Ishmael (mp=55) is
#: outside ``woof.config.MP_PHYSICS_ACCEPTED`` entirely.
_MP_SNOW_RADII_SUPPRESSED = frozenset((50,))


def legacy_scheme_has_req(mp_physics, use_mp_re):
    """WRF's ``(has_reqc, has_reqi, has_reqs)`` triple, both blocks.

    Read in WRF's own order (``module_physics_init.F:987-1033``): the
    flags start at 0, the :988-1024 disjunction sets all three to 1 for
    the schemes :data:`_MP_DECLARES_RADII` names, then the :1027-1033
    override re-zeroes has_reqs for the schemes in
    :data:`_MP_SNOW_RADII_SUPPRESSED`.  The triple is what the wrapper
    branches on, and it is not uniform for every scheme -- P3 (mp=50) is
    ``(1, 1, 0)`` -- so the adapter asks for the triple and never for one
    boolean.
    """
    declares = legacy_scheme_declares_radii(mp_physics, use_mp_re)
    flag = int(bool(declares))
    has_reqs = 0 if int(mp_physics) in _MP_SNOW_RADII_SUPPRESSED else flag
    return flag, flag, has_reqs

#: WRF Registry ``F_QI``/``F_QS`` membership: the schemes whose package
#: declaration carries ``qi`` and ``qs`` in ``moist``, which is what
#: ``cal_cldfra1`` keys on.  28 is a member --
#: ``Registry/Registry.EM_COMMON:3036`` declares
#: ``package thompsonaero mp_physics==28 - moist:qv,qc,qr,qi,qs,qg;...``,
#: character for character the same ``moist:qv,qc,qr,qi,qs,qg`` inventory
#: line 3024's ``thompson`` (mp=8) carries.  9 is a member for the same
#: reason: ``Registry/Registry.EM_COMMON:3025`` declares ``package
#: milbrandt2mom mp_physics==9 - moist:qv,qc,qr,qi,qs,qg,qh;scalar:qnc,
#: qnr,qni,qns,qng,qnh``.  Membership here is about F_QI/F_QS and is
#: INDEPENDENT of has_req*: mp=9 declares no radii and still gets ice
#: cloud fraction, which is exactly WRF's pairing.
#: 16 (WDM6) is a member on the same reading: ``package wdm6scheme`` at
#: ``Registry/Registry.EM_COMMON:3031`` gives it
#: ``moist:qv,qc,qr,qi,qs,qg``.
#:
#: This set answers ``F_QI AND F_QS``; it is NOT "does the scheme have
#: ice".  A scheme carrying ice with NO snow species belongs in
#: :data:`_LEGACY_ICE_ONLY_MICROPHYSICS` below instead -- adding it here
#: would assert a ``qs`` its Registry package never declares.  Resolve the
#: pair through :func:`legacy_cloud_fraction_flags`, never by reading one
#: boolean into both flags.
_LEGACY_ICE_ACTIVE_MICROPHYSICS = frozenset((2, 4, 6, 8, 9, 10, 16, 18, 24, 28, 30, 106))
# Named schemes carrying both qi and qs (capability row).
_LEGACY_ICE_ACTIVE_MICROPHYSICS = _LEGACY_ICE_ACTIVE_MICROPHYSICS | frozenset(
    _s.mp_id for _s in _NAMED_MP_SCHEMES.values()
    if {"qi", "qs"} <= set(_s.ice_mass_species))

#: WRF Registry ``F_QI and not F_QS``: schemes whose package carries ``qi``
#: in ``moist`` and no ``qs`` at all.  P3 one-category (mp=50) is the
#: member.  ``Registry/Registry.EM_COMMON:3038`` declares ``package
#: p3_1category mp_physics==50 - moist:qv,qc,qr,qi;scalar:qni,qnr,qir,qib;
#: state:re_cloud,re_ice,...`` -- there is a ``qi``, there is no ``qs`` and
#: no ``qg``, because P3 carries ONE ice category and predicts rime mass
#: and rime volume (``qir``/``qib``) instead of splitting snow from
#: graupel.  ``woof/core/state.py`` allocates exactly that inventory and
#: no ``qs``.
#:
#: mp=50's ABSENCE from :data:`_LEGACY_ICE_ACTIVE_MICROPHYSICS` is a
#: decision, not an omission: admitting it there would hand
#: ``cal_cldfra1`` ``F_QS = true`` for a scheme with no snow species.  WRF
#: does not fuse the two flags either -- ``phys/module_radiation_driver.F:
#: 3879-3887`` gives this case its own arm, commented "for P3, mp option
#: 50 or 51", with ``QCLD = QI + QC`` and ``weight = QI/QCLD``, distinct
#: from the ``F_QI .and. F_QC .and. F_QS`` arm at :3870-3877.
#:
#: mp=51 (``p3_1category_nc``, Registry.EM_COMMON:3039) is the same WRF
#: arm and is deliberately NOT listed: ``woof/config.py`` does not accept
#: 51, so it is refused by name at admission rather than half-answered
#: here.
#: Ferrier-Aligo (mp=5) is the second member on the same reading:
#: Registry.EM_COMMON:3019 declares ``package fer_mp_hires mp_physics==5 -
#: moist:qv,qc,qr,qi`` (qi is the total ice) with no qs, so cal_cldfra1
#: takes the P3 arm and then WRF's Ferrier selector arm (:3926-3937),
#: which ``cal_cldfra1(..., mp_physics=5)`` applies.
_LEGACY_ICE_ONLY_MICROPHYSICS = frozenset((5, 50))

#: WRF Registry ``not F_QI and not F_QS``: no frozen species in the package
#: at all -- ``passiveqv`` (mp=0, Registry.EM_COMMON:3014, ``moist:qv``)
#: ``kesslerscheme`` (mp=1, :3015, ``moist:qv,qc,qr``) and ``wsm3scheme``
#: (mp=3, :3017, ``moist:qv,qc,qr``; WSM3's cloud ice and snow live inside
#: qc and qr).  These take ``cal_cldfra1``'s qc-only arm with its 273.15 K
#: phase threshold (module_radiation_driver.F:3891-3899).
_LEGACY_NO_ICE_MICROPHYSICS = frozenset((0, 1, 3))


def legacy_ice_active(mp_physics: int) -> bool:
    """WRF Registry ``F_QI`` **and** ``F_QS`` membership for ``cal_cldfra1``.

    True only where the Registry package declares BOTH species.  It is not
    the answer to "is ice active": P3 (mp=50) has ice and no snow and is
    False here on purpose.  Anything wiring ``cal_cldfra1`` must call
    :func:`legacy_cloud_fraction_flags` instead.
    """

    return int(mp_physics) in _LEGACY_ICE_ACTIVE_MICROPHYSICS


def legacy_cloud_fraction_flags(mp_physics: int) -> tuple[bool, bool]:
    """``cal_cldfra1``'s ``(F_QI, F_QS)`` pair for one selector.

    WRF hands the radiation driver the Registry ``qi`` and ``qs`` package
    flags SEPARATELY: ``module_radiation_driver.F:3867`` tests all three of
    F_QI/F_QC/F_QS for presence, then :3870, :3880 and :3890 branch on
    their VALUES, and the three arms are different physics --
    ``QCLD = QI+QC+QS`` weighted ``(QI+QS)/QCLD``, ``QCLD = QI+QC``
    weighted ``QI/QCLD``, and ``QCLD = QC`` weighted by the freezing point.
    Reading ONE boolean into both flags can only reach the first and the
    third, and it silently gives the third -- a temperature-thresholded
    LIQUID cloud fraction -- to any scheme with ice but no snow.

    MEASURED on this tree for mp=50 before the split: a 258 K / 60 kPa
    column carrying 3e-5 kg/kg of P3 ice and no cloud water returned
    CLDFRA 0.0 (clear sky) where WRF's own P3 arm returns 1.0 (overcast).

    FAILS CLOSED.  Every selector ``woof/config.py`` accepts has a row in
    one of the three sets; an unmapped one raises instead of inheriting
    Kessler's ice-free arm.
    """

    selector = int(mp_physics)
    if selector in _LEGACY_ICE_ACTIVE_MICROPHYSICS:
        return (True, True)
    if selector in _LEGACY_ICE_ONLY_MICROPHYSICS:
        return (True, False)
    if selector in _LEGACY_NO_ICE_MICROPHYSICS:
        return (False, False)
    raise NotImplementedError(
        f"mp_physics={selector} has no F_QI/F_QS row in the legacy RRTMG "
        "adapter's Registry membership tables, so cal_cldfra1's arm is "
        "undecided; read the scheme's ``package`` line in "
        "Registry/Registry.EM_COMMON and add the selector to "
        "_LEGACY_ICE_ACTIVE_MICROPHYSICS (moist carries qi and qs), "
        "_LEGACY_ICE_ONLY_MICROPHYSICS (qi, no qs) or "
        "_LEGACY_NO_ICE_MICROPHYSICS (neither) rather than letting it "
        "default to the qc-only arm, which radiates an ice cloud as clear "
        "sky")


def legacy_radius_meters(effective_radius_microns):
    """Apply the RRTMG wrapper's micron-to-meter conversion exactly once.

    A float32 CuPy array is scaled by the glue unit's unflushed multiply
    (radii can be FP32 subnormals, which a CuPy ufunc would flush); the
    result is the NumPy expression's bits either way."""

    x = effective_radius_microns
    if type(x).__module__.split(".")[0] == "cupy" and x.dtype == np.float32:
        import cupy as cp
        adapter_gpu_preflight()
        x = cp.ascontiguousarray(x)
        out = cp.empty_like(x)
        _adapter_kernel("rla_scale")(
            *_grid_launch(x.size), (np.int64(x.size), x, F(1.0e-6), out))
        return out
    return (x * F(1.0e-6)).astype(np.float32)


def unsized_mynn_radii(re_cloud, re_ice, supplied_liquid, supplied_ice, *,
                       ice_rule):
    """Hand WRF's cloudy-layer rule the layers MYNN gave water as unsized.

    ``supplied_*`` come from
    :func:`woof.core.mynn_radiation.mynn_bl_cloud_supplied`.  There the
    scheme's radius is its no-cloud background or the size of a trace of its
    own condensate, not a size for MYNN's water, and WRF's radius tests (2.5
    um liquid, 5 um ice) miss NSSL's background (2.51 um, 10.01 um),
    Thompson's 2.51 um liquid floor and trace Thompson ice sized above 5 um.  A zero radius is what the
    rule sizes: ``max(2.5, 0)`` sits on its liquid bound and ``max(5, 0)``
    on its ice one, so these layers take 7.5 um over land, 10.5 um over
    water and the ice temperature table, exactly as RTE+RRTMGP gives them
    (woof.core.rrtmgp.cloudy_background_radii).  ``ice_rule`` is False for
    P3, whose ice WRF radiates as snow at P3's own radius.  Radii are metres
    or ``None`` (a scheme that declares none); new arrays are returned.
    """

    if supplied_liquid is None:
        return re_cloud, re_ice
    zero = F(0.0)
    if re_cloud is not None:
        re_cloud = np.where(supplied_liquid, zero, re_cloud).astype(
            np.float32)
    if re_ice is not None and ice_rule:
        re_ice = np.where(supplied_ice, zero, re_ice).astype(np.float32)
    return re_cloud, re_ice


_r512 = _lw._r512


@lru_cache(maxsize=1)
def legacy_shortwave_constant_bytes():
    """The three persistent CudaSW uploads, including pool rounding.

    The packed coefficient array, g-point band indices and three band
    metadata rows remain resident through both radiation spectra. This
    uses the runtime's table packer without constructing a CUDA engine.
    """
    tables = _sw_tables()
    packed, _ = _sw._pack_cuda_tables(tables)
    return (_r512(packed.nbytes) + _r512(np.asarray(tables.ngb).size * 4)
            + _r512(3 * _sw.NBNDSW * 4))


def legacy_radiation_vram_bytes(*, ncol, nz, p_top, column_chunk=None,
                                ncol_day=None, lw_coefficients=None,
                                longwave=True, shortwave=True,
                                resident_threads=None, o3input=2,
                                aer_opt=0, chem_optics_rows=0):
    """Peak transient device bytes of ONE adapter call.

    Composes the engines' own accurate pricing functions
    (``lw_batched_vram_bytes`` + ``lw_batched_const_bytes``,
    ``sw_batched_vram_bytes``) and the McICA device pricing
    (``mcica_device_vram_bytes``, fused column layout) with the
    adapter-held device arrays the engine functions do not price.

    The adapter runs each spectrum in SPANS (:data:`ADAPTER_SPAN_COLUMNS`,
    :func:`adapter_span`) and each span in engine-chunk WINDOWS.  A span
    holds its prep profiles (the engine inputs plus the cloud profiles
    the McICA windows read) for the whole span; a window adds its McICA
    slabs and radii pass-throughs, then either the generator's transient
    (generate phase) or the engine's (engine phase).  The LW and SW
    pipelines run sequentially with span storage freed in between, so the
    estimate is the max over the four allocation phases (LW generate, LW
    engine, SW generate, SW engine). Both engines' immutable coefficients
    remain resident across those phases and are added once outside the
    maximum. ``ncol_day`` bounds the SW day-column count (default:
    ``ncol``, the preflight upper bound).

    Whole-call result storage and the reusable ozone grid are priced
    separately. SWDOWN is allocated after the engines.

    ``resident_threads`` lets a host-side estimator use its already measured
    device profile without opening a CUDA context. Zero means an unknown
    device and prices the unchanged workspace ceilings; None retains the
    engines' live-device behavior for existing direct callers.

    ``aer_opt = 3`` adds what the SW span holds for the aerosol: the
    three (14 x nlay) optics slabs alive through the engine windows, the
    layer AOD and the two gathered Thompson number blocks (an upper bound:
    the engine's own zero-aerosol constants are priced by the engine
    function and not allocated on this path).
    """
    ncol = int(ncol)
    nz = int(nz)
    nlay_lw = _prep.compute_lw_nlayers(nz + 1, p_top)
    nlay_sw = nz + 1
    nday = ncol if ncol_day is None else int(ncol_day)
    chunk = None if column_chunk is None else int(column_chunk)
    chunk_lw = chunk or _lw.batch_column_chunk(
        _lw.NGPTLW, _lw.LW_BATCH_COLUMN_CHUNK_CEILING,
        resident_threads=resident_threads)
    nc_lw = min(chunk_lw, ncol)
    span_lw = adapter_span(chunk_lw, ncol) if ncol else 0
    # Host-side pricing passes no free-VRAM figure, so this is the
    # saturation width: an upper bound on the width a device run picks.
    chunk_sw = chunk or _sw.sw_batch_column_chunk(
        nlay_sw, resident_threads=resident_threads)
    nc_sw = min(chunk_sw, max(nday, 0))
    span_sw = adapter_span(chunk_sw, nday) if nday > 0 else 0
    f = 4
    estimate = 0
    constants = legacy_shortwave_constant_bytes() if shortwave else 0
    if longwave:
        C = lw_coefficients if lw_coefficients is not None else _lw_coeffs()
        constants += _lw.lw_batched_const_bytes(C)

        # Span-held LW prep profiles: play tlay h2o o3 and the eight
        # other gas profiles (12 x nlay), hgt and the seven cloud profiles
        # the McICA windows read (8 x nlay), plev tlev (nlay+1), emis
        # (16), tauaer (nlay x 16) and tsfc.
        sp_nl = _r512(span_lw * nlay_lw * f)
        span_held_lw = (20 * sp_nl + 2 * _r512(span_lw * (nlay_lw + 1) * f)
                        + _r512(span_lw * _lw.NBNDLW * f)
                        + _r512(span_lw * nlay_lw * _lw.NBNDLW * f)
                        + _r512(span_lw * f)
                        # the span's McICA stream states (16 B per subcolumn)
                        + _r512(span_lw * _lw.NGPTLW * 16))
        # One window's McICA slabs and its rei/rel/res pass-through copies.
        s_mcl = _r512(nc_lw * _lw.NGPTLW * nlay_lw * f)
        s_nl = _r512(nc_lw * nlay_lw * f)
        window_lw = 5 * s_mcl + 3 * s_nl
        lw_gen = (span_held_lw + window_lw
                  + _r512(_mcica.NBNDLW * nc_lw * nlay_lw * f)   # tauc
                  + _mcica.mcica_device_vram_bytes(
                      min(nc_lw, _mcica.MCICA_DEVICE_COLUMN_CHUNK),
                      nlay_lw, _lw.NGPTLW, layout="column"))
        lw_eng = span_held_lw + window_lw + _lw.lw_batched_vram_bytes(
            nc_lw, nlay_lw, mcica_layout="column")
        if chem_optics_rows:
            # Selected rows/met, Mie LW tau and padded LW tau coexist.
            extra = _r512(nc_lw * nz * (chem_optics_rows + 20) * f)
            extra += _r512(nc_lw * nlay_lw * 16 * f)
            extra += _r512(chem_optics_rows * 64 * f)
            lw_gen += extra
            lw_eng += extra
        estimate = max(lw_gen, lw_eng)

    if shortwave and nc_sw:
        # Span-held SW prep profiles: play tlay h2o o3 and four gas
        # profiles (8 x nlay), hgt and the seven cloud profiles (8 x
        # nlay), plev tlev (nlay+1), and the per-column tsfc, four
        # albedos, coszen, scon and the prep's solcon/obscur fills.
        sp_nl_s = _r512(span_sw * nlay_sw * f)
        span_held_sw = (16 * sp_nl_s + 2 * _r512(span_sw * (nlay_sw + 1) * f)
                        + 9 * _r512(span_sw * f)
                        + _r512(span_sw * _sw.NGPTSW * 16))   # McICA states
        if int(aer_opt) == 2:
            span_held_sw += (3 * _r512(span_sw * _sw.NBNDSW * nlay_sw * f)
                             + _r512(span_sw * nz * f)
                             + 2 * _r512(span_sw * 4 * f))
        if int(aer_opt) == 1:
            span_held_sw += (3 * _r512(span_sw * _sw.NBNDSW * nlay_sw * f)
                             + _r512(ncol * nz * 6 * f) + _r512(ncol * 72 * f))
        if int(aer_opt) == 3:
            span_held_sw += (3 * _r512(span_sw * _sw.NBNDSW * nlay_sw * f)
                             + 3 * _r512(span_sw * nz * f))
        s_mcl_s = _r512(nc_sw * _sw.NGPTSW * nlay_sw * f)
        s_nl_s = _r512(nc_sw * nlay_sw * f)
        window_sw = 8 * s_mcl_s + 3 * s_nl_s
        sw_gen = (span_held_sw + window_sw
                  + 2 * _r512(_mcica.NBNDSW * nc_sw * nlay_sw * f)  # 0 and 1 bands
                  + _mcica.mcica_device_vram_bytes(
                      min(nc_sw, _mcica.MCICA_DEVICE_COLUMN_CHUNK),
                      nlay_sw, _sw.NGPTSW, layout="column"))
        sw_eng = span_held_sw + window_sw + _sw.sw_batched_vram_bytes(
            nc_sw, nlay_sw, mcica_layout="column")
        if chem_optics_rows:
            # Four Mie wavelengths, converted bands and padded SW optics.
            extra = _r512(nc_sw * nz * (chem_optics_rows + 58) * f)
            extra += _r512(nc_sw * nlay_sw * 42 * f)
            extra += _r512(chem_optics_rows * 64 * f)
            sw_gen += extra
            sw_eng += extra
        estimate = max(estimate, sw_gen, sw_eng)
    # Whole-call storage is separate from the per-window McICA terms.
    # SWDOWN is allocated after the engines and does not raise this peak.
    result_bytes = 2 * _r512(nz * ncol * f) + 3 * _r512(ncol * f)
    ozone_bytes = _r512(nz * ncol * f) if int(o3input) == 2 else 0
    if int(aer_opt) == 2:
        # Held history plus the current radiation call's diagnostics.
        result_bytes += 2 * (_r512(nz*ncol*f) + 4*_r512(ncol*f))
    return estimate + result_bytes + ozone_bytes + constants


# ---------------------------------------------------------------------------
# LW module-DATA tables: WRF v4.6.1's module_ra_rrtmg_lw carries
# lwavplank (totplnk/totplk16), lwatmref (preflog/tref/chi_mls), lwcldpr
# (absice*/absliq*) and the rrlw_wvn delwave/ngb rosters as COMPILE-TIME
# DATA statements -- they are part of the algorithm, not of the
# RRTMG_LW_DATA coefficient file, so the packaged-file chain
# (load_rrtmg_lw_coefficients -> build_lw_coefficients) alone cannot
# feed the batched engine.  They are packaged as
# woof/data/wrf_radiation/rrtmg_lw_statics.npz, generated BIT-EXACT
# from the compiled UNMODIFIED Fortran's own module state (the same
# oracle authority every LW gate uses) by
# tools/rrtmg_wrf461_oracle/lw_statics_package.py, SHA-256-pinned here,
# and re-gated bitwise against the oracle dump by
# tests/test_rrtmg_legacy_wiring.py whenever the fixture deck is present
# (provenance chain: woof/data/wrf_radiation/PROVENANCE.md).
# ---------------------------------------------------------------------------

#: (key, shape, dtype) roster of the packaged LW module-DATA tables, in
#: packaged member order.  Shapes/dtypes are the Fortran module shapes.
_LW_STATIC_SPECS = (
    ("wvn/totplnk", (181, 16), "float32"),
    ("wvn/totplk16", (181,), "float32"),
    ("wvn/delwave", (16,), "float32"),
    ("wvn/ngb", (140,), "int32"),
    ("ref/preflog", (59,), "float32"),
    ("ref/tref", (59,), "float32"),
    ("ref/chi_mls", (7, 59), "float32"),
    ("cld/absice0", (2,), "float32"),
    ("cld/absice1", (2, 5), "float32"),
    ("cld/absice2", (43, 16), "float32"),
    ("cld/absice3", (46, 16), "float32"),
    ("cld/absliq0", (), "float32"),
    ("cld/absliq1", (58, 16), "float32"),
)

#: Packaged asset carrying the 13 tables (one npz member per roster key,
#: C-ordered little-endian, member order = the roster order).
_LW_STATICS_PATH = (Path(__file__).resolve().parents[1] / "data"
                    / "wrf_radiation" / "rrtmg_lw_statics.npz")

#: SHA-256 of rrtmg_lw_statics.npz (fail-closed on corruption).  Also
#: pinned into the adapter's restart identity; the restart asset
#: manifest carries the same file under the role "wrf_rrtmg_lw_statics".
RRTMG_LW_STATICS_SHA256 = \
    "edd2508db89180667b0f80b4cdd991f4aa1447e711bbfe3a6db8de5fdb778d62"

_LW_STATICS_CACHE = None


def _lw_static_tables():
    """Load the packaged module-DATA tables (fail-closed on drift).

    Returns the same immutable dict the base64-blob predecessor decoded:
    one read-only C-ordered array per ``_LW_STATIC_SPECS`` key (the
    ()-shaped ``cld/absliq0`` as a float32 scalar), file digest, member
    roster/order, shapes and dtypes all enforced before anything is
    handed to the coefficient merge.
    """
    global _LW_STATICS_CACHE
    if _LW_STATICS_CACHE is None:
        import io
        try:
            from woof.tables import table_path, _file_state
            from woof.file_digest import sha256_file
            path = table_path("rrtmg-lw", _LW_STATICS_PATH)
            before = _file_state(path)
            payload = path.read_bytes()
            digest = sha256_file(path)
            if _file_state(path) != before:
                raise RuntimeError(
                    f"RRTMG table {path} changed while being read; refusing mixed "
                    "lookup coefficients; run woof fetch-tables --schemes rrtmg-lw --from DIR --replace-damaged")
        except OSError as exc:
            raise RuntimeError(
                "packaged LW module-DATA asset is missing: "
                f"{_LW_STATICS_PATH} ({exc})") from exc
        if digest != RRTMG_LW_STATICS_SHA256:
            raise RuntimeError(
                "packaged LW module-DATA asset is corrupt: sha256 "
                f"{digest} != pinned {RRTMG_LW_STATICS_SHA256}")
        out = {}
        with np.load(io.BytesIO(payload)) as npz:
            expected = [key for key, _shape, _dtype in _LW_STATIC_SPECS]
            if list(npz.files) != expected:
                raise RuntimeError(
                    "packaged LW module-DATA asset carries the wrong "
                    f"member roster: {list(npz.files)} != {expected}")
            for key, shape, dtype in _LW_STATIC_SPECS:
                arr = npz[key]
                if tuple(arr.shape) != tuple(shape) or \
                        arr.dtype != np.dtype(dtype):
                    raise RuntimeError(
                        f"packaged LW module-DATA table {key} has shape "
                        f"{arr.shape} dtype {arr.dtype}; the contract "
                        f"requires {tuple(shape)} {dtype}")
                value = arr[()] if arr.ndim == 0 else \
                    np.ascontiguousarray(arr)
                if hasattr(value, "setflags"):
                    value.setflags(write=False)
                out[key] = value
        _LW_STATICS_CACHE = out
    return _LW_STATICS_CACHE


_LW_COEFFS_CACHE = None
_SW_TABLES_CACHE = None


def _lw_coeffs():
    """The COMPLETE batched-engine coefficient dict, once per process.

    Three bitwise-vouched sources, merged: (1) the init-chain builds of
    ``build_lw_coefficients`` on the packaged RRTMG_LW_DATA raw arrays
    (tables/constants/reductions); (2) the packaged loader's own reduced
    per-band arrays (``absa``/``absb``/... -- gated bitwise against the
    Fortran module dump by the ingest gates), which carry the
    ``absa``-equivalence views the CUDA band kernels index; (3) the
    packaged module-DATA tables above.  ``lw_batched_const_bytes`` runs
    as a completeness probe so a missing key fails at load, not at the
    first radiation call.
    """
    global _LW_COEFFS_CACHE
    if _LW_COEFFS_CACHE is None:
        from woof.ingest import rrtmg_coeffs as _coeffs
        loaded = _coeffs.load_rrtmg_lw_coefficients()
        C = _lw.build_lw_coefficients(loaded, np.float32(1004.5))
        for band in range(1, 17):
            module = loaded[f"rrlw_kg{band:02d}"]
            for name, value in module.items():
                C.setdefault(f"kg{band:02d}/{name}", value)
        C.update(_lw_static_tables())
        _lw.lw_batched_const_bytes(C)      # completeness probe
        _LW_COEFFS_CACHE = C
    return _LW_COEFFS_CACHE


def _sw_tables():
    """Load + assemble the SW tables once per process (immutable)."""
    global _SW_TABLES_CACHE
    if _SW_TABLES_CACHE is None:
        from woof.ingest import rrtmg_coeffs as _coeffs
        _SW_TABLES_CACHE = _sw.tables_from_coeffs(
            _coeffs.load_rrtmg_sw_coefficients())
    return _SW_TABLES_CACHE


_CUDA_SW_CACHE = None


def _cuda_sw_device() -> int:
    """The card a legacy SW engine is being asked for (the current one)."""
    import cupy as cp
    return int(cp.cuda.Device().id)


def _cuda_sw(tab):
    """The compiled CUDA SW engine for ``tab`` on this card, once per card.

    Construction is the expensive half of this adapter: ``CudaSW.__init__``
    runs a full NVRTC compile of ``kernels/rrtmg_sw.cu`` and uploads the
    packed tables.  Streaming builds one adapter per TILE BUFFER and the
    [devices] split one per SLAB, so an uncached engine would multiply both
    the compile and the resident tables.

    PER CARD, because a compiled module and its tables live on one card and
    a slab on another card must not launch them.  The engine's writable
    batched workspace is per STREAM inside the engine (``CudaSW._scratch``),
    so slabs stepping at once on one card share the read-only module and
    tables and never the workspace.  Keyed on the tables OBJECT, held in the
    entry so its id cannot be reused: a caller assembling its own tables
    gets its own engine rather than borrowing one built from different
    coefficients.
    """
    global _CUDA_SW_CACHE
    if _CUDA_SW_CACHE is None:
        _CUDA_SW_CACHE = {}
    key = (_cuda_sw_device(), id(tab))
    hit = _CUDA_SW_CACHE.get(key)
    if hit is None:
        hit = _CUDA_SW_CACHE.setdefault(key, (tab, _sw.CudaSW(tab)))
    return hit[1]


# ---------------------------------------------------------------------------
# Device-resident adapter glue (kernels/rrtmg_legacy_adapter.cu).
#
# The adapter used to download every field, pack host columns, run the
# wrapper glue in NumPy and upload the results.  The kernels below are the
# same float32 statements on the device, in the same order, so the call
# returns the same bits without the round trips.  The unit compiles through
# NVRTC directly with --ftz=false and writes its arithmetic as unflushed PTX
# (see the .cu header); adapter_gpu_preflight proves both on the live device.
# ---------------------------------------------------------------------------

_ADAPTER_MODULE = {}
_ADAPTER_KERNELS = {}
_ADAPTER_PREFLIGHTED = set()
_ADAPTER_BLOCK = 256


@cuda_cache(maxsize=None)
def _adapter_module():
    """Compile kernels/rrtmg_legacy_adapter.cu via NVRTC directly (the
    route and the missing ``-arch`` follow ``rrtmg_lw._gpu_module``)."""
    import cupy as cp
    device = int(cp.cuda.Device().id)
    if device not in _ADAPTER_MODULE:
        import cupy as cp
        from woof import nvrtc_ptx_cache as _cc
        path = (Path(__file__).resolve().parent / "kernels"
                / "rrtmg_legacy_adapter.cu")
        ptx, _mapping = _cc.compile_using_nvrtc(
            path.read_text(encoding="utf-8"),
            ("-std=c++17", "--ftz=false"),
            None, "rrtmg_legacy_adapter.cu")
        mod = cp.cuda.function.Module()
        mod.load(ptx.encode() if isinstance(ptx, str) else ptx)
        _ADAPTER_MODULE[device] = mod
    return _ADAPTER_MODULE[device]


@cuda_cache(maxsize=None)
def _adapter_kernel(name):
    import cupy as cp
    key = (int(cp.cuda.Device().id), name)
    if key not in _ADAPTER_KERNELS:
        _ADAPTER_KERNELS[key] = _adapter_module().get_function(name)
    return _ADAPTER_KERNELS[key]


def adapter_gpu_preflight(force=False):
    """Prove on the live device that the glue unit keeps FP32 subnormals:
    a subnormal product and quotient survive and a subnormal compares
    greater than zero, bitwise equal to NumPy."""
    import cupy as cp
    device = int(cp.cuda.Device().id)
    if device in _ADAPTER_PREFLIGHTED and not force:
        return
    import cupy as cp
    x = np.array([1.0e-30, 1.0e-10, 1.0e-40, 1.0e-38, 4.0], np.float32)
    o = cp.zeros(3, dtype=cp.float32)
    _adapter_kernel("rla_probe")((1,), (32,), (cp.asarray(x), o))
    got = cp.asnumpy(o)
    want = np.array([x[0] * x[1], 1.0 if x[2] > F(0.0) else 0.0,
                     x[3] / x[4]], np.float32)
    if got.view(np.uint32).tolist() != want.view(np.uint32).tolist():
        raise RuntimeError(
            "rrtmg_legacy adapter GPU preflight failed: got %r want %r "
            "(subnormal flush on this toolchain)" % (got, want))
    _ADAPTER_PREFLIGHTED.add(device)


#: Prep outputs that are WRF-level extras, not engine inputs; the device
#: adapter drops them before the engine call so they are not held through
#: the engine phase (see legacy_radiation_vram_bytes).
_LW_PREP_EXTRAS = ("o31d", "hgt", "cldfrac", "pdel")
_SW_PREP_EXTRAS = ("o31d", "pdel", "coszr", "mcica_inputs")

#: Columns one adapter SPAN covers.  The wrapper glue of a span -- the
#: column gather, t8w, the cloud fraction, the radii and the prep kernel
#: -- runs once over the whole span, so its one-thread-per-column kernels
#: fill the card and its launches are paid once per span; the McICA slabs
#: and the engines then run one engine chunk (WINDOW) at a time inside the
#: span, so the (column, layer, g-point) slabs never exist for more than
#: one window.  Every stage is per column (the prep kernel and the glue
#: by construction, the McICA stream by its per-column seed, the engines
#: by their own gated contracts), so a span is bitwise invisible.  The
#: span holds about 15 KB of prep profiles per column on top of one
#: window's engine transient (legacy_radiation_vram_bytes prices both).
ADAPTER_SPAN_COLUMNS = 16384


def adapter_span(chunk, ncol):
    """Columns per adapter span: the largest whole number of engine
    chunks within ADAPTER_SPAN_COLUMNS (at least one chunk), never more
    than the ``ncol`` columns there are."""
    chunk = int(chunk)
    span = chunk * max(1, ADAPTER_SPAN_COLUMNS // chunk)
    return max(1, min(span, int(ncol)))


def _grid_launch(n):
    return ((int(n) + _ADAPTER_BLOCK - 1) // _ADAPTER_BLOCK,), (_ADAPTER_BLOCK,)


def _device_grid(a, nk):
    """(nk, ny, nx) field -> device float32 (nk, ncol), C-contiguous.

    A float32 device field is reshaped in place (data movement only);
    anything else is cast on the host with NumPy semantics first, exactly
    as the host packing did."""
    import cupy as cp
    if isinstance(a, cp.ndarray) and a.dtype == cp.float32:
        return cp.ascontiguousarray(a.reshape(nk, -1))
    h = np.asarray(RRTMGLegacyRadiation._host(a), np.float32)
    return cp.asarray(np.ascontiguousarray(h.reshape(nk, -1)))


def _device_flat(a):
    """(ny, nx) field -> device float32 (ncol,)."""
    import cupy as cp
    if isinstance(a, cp.ndarray) and a.dtype == cp.float32:
        return cp.ascontiguousarray(a.reshape(-1))
    h = np.asarray(RRTMGLegacyRadiation._host(a), np.float32)
    return cp.asarray(np.ascontiguousarray(h.reshape(-1)))


class _ColumnGather:
    """Column blocks of several (nk, ncol) device grids, one launch a chunk.

    Built once per radiation call over the grids the chunks read; each call
    returns (nc, nk) C-contiguous blocks that are views of one packed buffer
    allocated for that chunk (so, like the separate gathers it replaces, it
    lives only while the chunk's prep inputs do).  Data movement only."""

    def __init__(self, grids, ncol):
        import cupy as cp
        self.cp = cp
        self.grids = list(grids)
        self.ncol = int(ncol)
        self.nk = [int(g.shape[0]) for g in self.grids]
        for g in self.grids:
            if g.dtype != cp.float32 or not g.flags.c_contiguous or                     g.shape[1] != self.ncol:
                raise ValueError("gathered grids must be C-contiguous float32 "
                                 "(nk, ncol) device arrays")
        self.src = cp.asarray(np.asarray([g.data.ptr for g in self.grids],
                                         dtype=np.uint64))
        self.nk_d = cp.asarray(np.asarray(self.nk, dtype=np.int32))

    def __call__(self, sel):
        cp = self.cp
        if isinstance(sel, slice):
            c0, nc, idx = int(sel.start), int(sel.stop) - int(sel.start), None
        else:
            c0, nc, idx = 0, int(sel.size), sel
        total = nc * sum(self.nk)
        buf = cp.empty(total, dtype=cp.float32)
        _adapter_kernel("rla_gather")(
            *_grid_launch(total),
            (np.int32(len(self.grids)), np.int32(nc), np.int64(self.ncol),
             np.int64(c0), np.uint64(0) if idx is None else idx, self.src,
             self.nk_d, buf))
        out, base = [], 0
        for nk in self.nk:
            out.append(buf[base:base + nc * nk].reshape(nc, nk))
            base += nc * nk
        return out


def _radius_meters_blocks(blocks):
    """Convert the present radius blocks in one unflushed launch."""
    import cupy as cp
    present = [(key, block) for key, block in blocks.items()
               if block is not None]
    result = dict.fromkeys(blocks)
    if not present:
        return result
    n = present[0][1].size
    buf = cp.empty(n * len(present), dtype=cp.float32)
    pointers = [block for key, block in present]
    pointers += [np.uint64(0)] * (3 - len(pointers))
    _adapter_kernel("rla_scale_radii")(
        *_grid_launch(buf.size),
        (np.int64(n), np.int32(len(present)), *pointers, F(1.0e-6), buf))
    for i, (key, block) in enumerate(present):
        result[key] = buf[i*n:(i+1)*n].reshape(block.shape)
    return result


def _f32_positive(a):
    """``a > 0`` for float32 ``a`` decided on the bits, so a subnormal
    counts as positive whatever the compile route of the comparison."""
    import cupy as cp
    bits = a.view(cp.int32)
    return (bits > 0) & (bits <= 0x7F800000)


# ---------------------------------------------------------------------------
# Ozone nest routing: the child side of WRF's root-only o3rad evaluation.
# ---------------------------------------------------------------------------

class ParentOzoneProvider:
    """Child-domain o33d: the parent's retained field, SINT-interpolated.

    Wired by ``runtime.prepare_child_case`` with the parent domain's
    :class:`RRTMGLegacyRadiation` and the mass-point ``wrapper="interp"``
    :class:`woof.core.nest_interp.NestRegistration` for the pair.  Each
    child radiation call reads the parent's most recent retained o33d
    grid (so the child's ozone updates exactly when the parent's does,
    like WRF's rdf-interpolated ``o3rad``) and interpolates it through
    the certified SINT operator.  Fails closed if the parent has not
    radiated yet: WRF's sequencing evaluates the root's o3rad before any
    nest radiation, so a None parent field is a sequencing violation,
    not a case for a silent climatology fallback.
    """

    def __init__(self, parent, registration):
        self.parent = parent
        self.registration = registration
        self.released = False

    def release(self) -> None:
        """Drop the parent adapter: the build this provider feeds is superseded.

        A nest waiting for a delayed start holds a startup build whose
        radiation reads its PARENT's adapter through this provider.  When
        the parent is itself a delayed nest and activates, its startup
        adapter is replaced, and this link was the one reference that kept
        the outgoing adapter (and what it holds) alive until the waiting
        nest activated in turn.  The waiting nest is rebuilt at its own
        start against the parent's new adapter, so the startup build never
        radiates through this provider again.
        """
        self.parent = None
        self.released = True

    def __call__(self):
        if self.released:
            raise RuntimeError(
                "child rrtmg_legacy radiation called through an ozone "
                "provider released with its superseded startup build; a "
                "nest waiting for its start is rebuilt against its "
                "parent's current radiation before it radiates")
        import cupy as cp

        from woof.core import nest_interp as _nest

        grid = getattr(self.parent, "_o33d_grid", None)
        if grid is None:
            raise RuntimeError(
                "child rrtmg_legacy radiation ran before its parent "
                "retained an o33d field: WRF computes the root domain's "
                "o3rad before any nest radiation (root-compute + "
                "parent->child interpolation routing); fix the call "
                "sequencing instead of falling back to a per-nest "
                "climatology WRF never evaluates")
        child = cp.asnumpy(_nest.sint(cp.asarray(grid), self.registration))
        nz = child.shape[0]
        return np.ascontiguousarray(
            child.transpose(1, 2, 0).reshape(-1, nz))


def release_waiting_ozone_links(outgoing, waiting) -> None:
    """Unlink nests waiting for their start from a parent's outgoing radiation.

    A nest's ozone reads its parent's radiation through a
    :class:`ParentOzoneProvider`, so a child of a delayed nest that has not
    started yet holds the parent's STARTUP radiation through its own
    startup build.  When that parent activates and its startup build is
    released, this link kept the outgoing radiation (and everything it
    holds) alive until the child activated as well.  ``outgoing`` is the
    parent's startup ``radiation_callable`` and ``waiting`` its child nodes
    that have not started.  A waiting child's startup build never radiates
    before it starts, and it is rebuilt against the parent's new radiation
    then, so each of its providers that reads ``outgoing`` is released.
    Every other provider, a started child's and the CAM carrier's
    :class:`woof.core.cam_ozone.DriverOzoneProvider` among them, is left
    alone.

    A waiting child whose state is a
    :class:`woof.core.streamed_state.CanonicalStoreState` (a nest the
    prepared domain-tree forecast streams through a host store) is
    skipped: that view refuses a read of its radiation with
    ``CanonicalStateRefused``, which stopped the run at the parent's
    start, and that route builds a nest's startup radiation from its
    prepared cache with no provider of this class, so there is no link to
    release.
    """
    from woof.core.radiation_composition import radiation_adapters
    from woof.core.streamed_state import CanonicalStoreState

    for child in waiting:
        state = getattr(child, "state", None)
        if isinstance(state, CanonicalStoreState):
            continue
        physics = getattr(state, "physics", None)
        radiation = getattr(physics, "radiation_callable", None)
        for adapter in radiation_adapters(radiation):
            provider = getattr(adapter, "_ozone_provider", None)
            if (isinstance(provider, ParentOzoneProvider)
                    and provider.parent is outgoing):
                provider.release()


# ---------------------------------------------------------------------------
# The adapter.
# ---------------------------------------------------------------------------

def _with_deferred_device_checks(method):
    """Run ``method`` inside one :func:`woof.core.deferred_device_checks.deferred` scope."""
    from functools import wraps

    @wraps(method)
    def wrapper(*args, **kwargs):
        from woof.core.deferred_device_checks import deferred
        with deferred():
            return method(*args, **kwargs)
    return wrapper


class RRTMGLegacyRadiation:
    """WRF v4.6.1 legacy RRTMG (option 4/4) forecast radiation adapter.

    Plugs into the same ``radiation_due`` slot as ``RRTMGPRadiation``
    (``__call__(atmosphere=, fields=, state=, cfg=)`` returning
    :class:`woof.core.physics.RadiationResult`).  Construction performs
    the full readiness proof -- packaged SHA-pinned assets, coefficient
    builds, CUDA kernel compilation and live-device preflights -- and
    fails closed with a clear receipt if anything is missing; this
    replaces the retired ``require_rrtmg_legacy_executable`` stub with
    executable reality.

    ``p_top`` may be None at construction (the value is then taken from
    ``state.p_top`` at call time; construction sites that know it pass it
    eagerly).  ``column_chunk=None`` uses each engine's own default chunk
    (LW 4096 / SW 2048); an explicit value bounds both pipelines and is
    part of the restart identity.  Chunking is bitwise invisible (prep
    batch entries, McICA twin, and both engines are each per-column
    bitwise at any width -- their own gated contracts).
    """

    #: RRTMG_LWRAD's ``OLR`` (TOA outgoing longwave, W m-2) is
    #: ``TOTUFLUX`` at the top level -- ``lwrad_outputs_batch`` already maps
    #: it out of ``uflx`` -- so this adapter fills the driver's OLR slot and
    #: the run's wrfout carries the field.  The declaration is what the
    #: driver reads to decide whether OLR exists at all.
    publishes_olr = True
    #: The chunk prep builds WRF's icloud=1 CLDFRA (plus the MYNN merge)
    #: on either spectrum, so the driver's held CLDFRA output buffer
    #: exists for every legacy RRTMG configuration.
    publishes_cldfra = True

    # No latitude-derived ozone is retained. The common tile gather and
    # moving-grid routes change latitude in place, so every radiation call
    # interpolates the CAM climatology from the live latitude_deg
    # (wrf_ozone.ozn_latitude_time_int, two months per call).

    def __init__(self, start_time, latitude_deg, longitude_deg, *,
                 p_top=None, column_chunk=None, ozone_parent=None,
                 o3input=2, longwave=True, shortwave=True, trace_gas_overrides=None,
                 ozone_routing=None, aer_opt=0, smoke_provider=None,
                 aer3_type_source="rural", transported_smoke=None,
                 ghg_input=0):
        if not isinstance(start_time, datetime):
            raise TypeError("radiation_start_time must be a datetime")
        self.start_time = start_time
        if int(ghg_input) not in _ADMITTED_GHG_INPUT:
            raise ValueError(f"ghg_input must be 0 or 1, got {ghg_input!r}")
        self.ghg_input = int(ghg_input)
        self.longwave = bool(longwave)
        self.shortwave = bool(shortwave)
        if not (self.longwave or self.shortwave):
            raise ValueError("radiation adapter needs at least one spectrum")
        #: WRF aer_opt for the shortwave this adapter runs: 0 (zero
        #: aerosol) or 3 (the Thompson aerosol optics,
        #: woof.core.rrtmg_aerosol_optics).  The longwave of the
        #: operational HRRR fork takes no aerosol under aer_opt = 3
        #: (module_ra_rrtmg_lw.F:12538-12560 fills tauaer from chemistry
        #: only), so a longwave-only adapter carries 0.
        self.aer_opt = int(aer_opt)
        if self.aer_opt not in _ADMITTED_AER_OPT:
            raise NotImplementedError(
                f"rrtmg_legacy implements aer_opt in {_ADMITTED_AER_OPT}; "
                f"got aer_opt={aer_opt!r}")
        if self.aer_opt and not self.shortwave:
            raise ValueError("aer_opt=3 acts on the shortwave; a "
                             "longwave-only adapter carries aer_opt=0")
        if smoke_provider is not None and not (self.shortwave and self.aer_opt == 3):
            raise ValueError("prescribed smoke requires legacy shortwave with aer_opt=3")
        #: aer_opt = 3 aerosol type per column: ``rural`` (the HRRR fork's
        #: type 1) or ``landmask`` (the NOAA MPAS fork: 1 over land, 3 over
        #: water, from the same xland the shortwave reads).
        if aer3_type_source not in _aer3.AER3_TYPE_SOURCES:
            raise ValueError(
                f"aer3_type_source must be one of {_aer3.AER3_TYPE_SOURCES}, "
                f"got {aer3_type_source!r}")
        if aer3_type_source != "rural" and self.aer_opt != 3:
            raise ValueError(
                "aer3_type_source='landmask' selects the aer_opt=3 shortwave "
                "aerosol type; this adapter runs aer_opt="
                f"{self.aer_opt}, where it would never be read")
        if aer3_type_source != "rural" and smoke_provider is not None:
            raise ValueError(
                "prescribed smoke follows the HRRR fork's rural optics; the "
                "NOAA MPAS land/water aerosol type has no smoke coupling")
        self.aer3_type_source = aer3_type_source
        self.publishes_aodtot = self.shortwave and self.aer_opt == 1
        self._smoke_provider = smoke_provider
        #: simple_dir_fdb = 1: the state attribute of the forecast's own
        #: transported smoke, whose layer AOD joins the aer_opt = 3 aerosol
        #: at every shortwave call (HRRR v4.1.21 module_radiation_driver.F
        #: :979-984, rural optics, MIN(3, layer AOD)).
        if transported_smoke is not None:
            if not (self.shortwave and self.aer_opt == 3):
                raise ValueError("simple_dir_fdb=1 (transported smoke) acts "
                                 "on the legacy shortwave with aer_opt=3")
            if smoke_provider is not None:
                raise ValueError("transported smoke and a prescribed smoke "
                                 "profile would count the plume twice")
            if aer3_type_source != "rural":
                raise ValueError(
                    "transported smoke follows the HRRR fork's rural optics; "
                    "the NOAA MPAS land/water aerosol type has no smoke coupling")
        self._transported_smoke = transported_smoke
        self.publishes_olr = self.longwave
        from woof.core.trace_gases import (
            LEGACY_LW_GASES, LEGACY_SW_GASES, validate_trace_gas_overrides)
        supported = ((LEGACY_LW_GASES if self.longwave else frozenset())
                     | (LEGACY_SW_GASES if self.shortwave else frozenset()))
        self.trace_gas_overrides = (None if trace_gas_overrides is None else
            validate_trace_gas_overrides(trace_gas_overrides, supported=supported,
                                         consumer="selected legacy RRTMG spectra"))
        lat = np.asarray(self._host(latitude_deg), np.float32)
        lon = np.asarray(self._host(longitude_deg), np.float32)
        if lat.shape != lon.shape:
            raise ValueError("radiation latitude/longitude shapes must match")
        self.latitude_deg = np.ascontiguousarray(lat)
        self.longitude_deg = np.ascontiguousarray(lon)
        self.p_top = None if p_top is None else float(p_top)
        self.o3input = int(o3input)
        if self.o3input not in (0, 2):
            raise ValueError(
                "legacy RRTMG implements o3input=0 (wrapper O3DATA) and "
                f"o3input=2 (CAM climatology), got {o3input!r}")
        if column_chunk is not None and int(column_chunk) < 1:
            raise ValueError("column_chunk must be positive")
        self.column_chunk = None if column_chunk is None \
            else int(column_chunk)
        if ozone_parent is not None and not callable(ozone_parent):
            raise TypeError(
                "ozone_parent must be a callable provider (e.g. "
                "ParentOzoneProvider) returning the child-grid o33d "
                "columns, or None on the root domain")
        if self.o3input == 0 and ozone_parent is not None:
            raise ValueError(
                "o3input=0 constructs ozone inside the legacy wrapper and "
                "must not receive a parent o3rad provider")
        if self.aer_opt == 1 and ozone_parent is not None:
            raise ValueError("aer_opt=1 resident nest requires domain-1 aerod parent forcing; child-grid climatology would change WRF routing")
        self._ozone_provider = ozone_parent
        #: How this adapter obtains ozone, in the one vocabulary
        #: woof.core.cam_ozone owns.  Derived from the construction unless
        #: the route declares a name the construction admits, which is how
        #: the offline child route says "the climatology, on the child's own
        #: grid" rather than being read as a root.
        from woof.core.cam_ozone import resolve_ozone_routing
        self.ozone_routing = resolve_ozone_routing(
            ozone_routing, o3input=self.o3input,
            has_parent=ozone_parent is not None)
        #: the most recent o33d field (nz, ny, nx) host float32, retained
        #: so child domains can interpolate it (WRF's root-compute +
        #: parent->child o3rad routing); None until the first call.
        self._o33d_grid = None
        self.update_count = 0
        #: test instrumentation forwarded to the engines' _stage_probe
        #: hooks (must not affect results); None on the forecast path.
        self._stage_probe = None

        # ---- fail-closed readiness: assets, tables, kernels ----------
        try:
            self._C = _lw_coeffs() if self.longwave else None
        except Exception as exc:
            raise RuntimeError(
                "ra_rrtmg_variant='rrtmg_legacy' is selected but the "
                "packaged RRTMG_LW_DATA coefficients cannot be "
                f"loaded/built: {exc}") from exc
        try:
            self._sw_tables = ((_sw.load_sw_tables(aer_opt=1) if self.aer_opt == 1 else _sw_tables()) if self.shortwave else None)
        except Exception as exc:
            raise RuntimeError(
                "ra_rrtmg_variant='rrtmg_legacy' is selected but the "
                "packaged RRTMG_SW_DATA coefficients cannot be "
                f"loaded/built: {exc}") from exc
        if self.o3input == 0:
            # O3DATA is evaluated independently inside lwrad/swrad prep and
            # does not read the CAM climatology or parent-routed o3rad field.
            self._ozone = None
            self._ozone_climo = None
        elif self._ozone_provider is None:
            # Root routing: the climatology chain runs here, on this
            # adapter's own grid.  WRF evaluates it for id == 1, which is
            # both a resident root and an offline child (that route's
            # domain is configured as a root and stamps parent_id = 0);
            # self.ozone_routing says which of the two this is.  The
            # latitude interpolation (WRF's oznini-time work) runs at each
            # call on the live latitude, for the two months that call uses.
            try:
                from woof.ingest import wrf_ozone as _ozone
                self._ozone = _ozone
                self._ozone_climo = _ozone.load_ozone_climatology()
            except Exception as exc:
                raise RuntimeError(
                    "ra_rrtmg_variant='rrtmg_legacy' is selected but the "
                    "packaged CAM ozone climatology (ozone*.formatted) "
                    f"cannot be loaded: {exc}") from exc
        else:
            # Child routing: o33d arrives from the parent; the child
            # never invokes the climatology chain (WRF nests receive the
            # parent-interpolated o3rad, never a fresh evaluation).
            self._ozone = None
            self._ozone_climo = None
        try:
            self._cuda_sw = (_cuda_sw(self._sw_tables)
                             if self.shortwave else None)
        except Exception as exc:
            raise RuntimeError(
                "ra_rrtmg_variant='rrtmg_legacy' is selected but the "
                "CUDA SW engine cannot be compiled/loaded on this "
                f"installation (cupy + a CUDA device are required): {exc}"
            ) from exc
        try:
            if self.longwave:
                _lw.gpu_preflight()
            _mcica.mcica_gpu_preflight()
        except Exception as exc:
            raise RuntimeError(
                "ra_rrtmg_variant='rrtmg_legacy' is selected but the "
                "CUDA LW/McICA kernels fail their live-device preflight: "
                f"{exc}") from exc

    # ------------------------------------------------------------------
    # The retained o33d field (nz, ny, nx).  The climatology route
    # computes it on the device and used to copy it to the host on every
    # radiation call (and the physics driver then copied it back into
    # o3rad): 180 MB down and up per call per rank at 900,000 columns, the
    # host-memory rise seen at every 1 km radiation step (A3).  It now
    # stays on the device; the host copy is made the first time a host
    # reader (a child provider, a restart, a test) asks, bit for bit the
    # same array.
    @property
    def _o33d_grid(self):
        host = self.__dict__.get("_o33d_host")
        device = self.__dict__.get("_o33d_device")
        if host is None and device is not None:
            import cupy as cp
            host = np.ascontiguousarray(cp.asnumpy(device))
            self.__dict__["_o33d_host"] = host
        return host

    @_o33d_grid.setter
    def _o33d_grid(self, value):
        self.__dict__["_o33d_host"] = value
        self.__dict__["_o33d_device"] = None

    def o33d_share(self, carrier) -> None:
        """Hold ``carrier`` (the driver's o3rad, just copied from this field)
        as the retained device field instead of a second copy, so keeping
        ozone on the card costs no memory over the carrier it fills."""
        if self.__dict__.get("_o33d_device") is not None:
            self.__dict__["_o33d_device"] = carrier

    def o33d_device(self):
        """The retained o33d field on the device, or ``None``."""
        device = self.__dict__.get("_o33d_device")
        if device is not None:
            return device
        host = self.__dict__.get("_o33d_host")
        if host is None:
            return None
        import cupy as cp
        return cp.asarray(host)

    @staticmethod
    def _host(a):
        """Bit-preserving host copy of a numpy or cupy array."""
        if type(a).__module__.split(".")[0] == "cupy":
            return a.get()
        return np.asarray(a)

    @classmethod
    def _vertical_weight_device(cls, value):
        """Keep canonical float32 weights on the device without conversion."""
        import cupy as cp
        if isinstance(value, cp.ndarray) and value.dtype == cp.float32:
            return cp.ascontiguousarray(value)
        # Preserve the host conversion semantics for noncanonical callers.
        return cp.asarray(np.ascontiguousarray(
            np.asarray(cls._host(value), np.float32)))

    # ------------------------------------------------------------------
    def restart_identity(self):
        """Strict-JSON identity: distinct from every RTE+RRTMGP identity
        so a restart written under one 4/4 implementation refuses to
        resume under the other (dossier section 1)."""
        from woof.io.restart import (
            RRTMG_LEGACY_ABOVE_ATMOSPHERE_POLICY,
            RRTMG_LEGACY_LW_ALGORITHM_IDENTITY,
            RRTMG_LEGACY_SW_ALGORITHM_IDENTITY)
        from woof.physics_compat import WRF_RRTMG_LEGACY
        identity = {
            "algorithm": (f"lw={RRTMG_LEGACY_LW_ALGORITHM_IDENTITY};"
                          f"sw={RRTMG_LEGACY_SW_ALGORITHM_IDENTITY}"),
            "algorithms": {
                "lw": RRTMG_LEGACY_LW_ALGORITHM_IDENTITY,
                "sw": RRTMG_LEGACY_SW_ALGORITHM_IDENTITY,
            },
            "above_atmosphere_policy":
                RRTMG_LEGACY_ABOVE_ATMOSPHERE_POLICY,
            "compatibility_token": WRF_RRTMG_LEGACY,
            "permuteseed_lw": int(_mcica.LW_PERMUTESEED),
            "permuteseed_sw": int(_mcica.SW_PERMUTESEED),
            "icld": 2,
            "idcor": 0,
            "o3input": self.o3input,
            "ghg_input": self.ghg_input,
            "aer_opt": self.aer_opt,
            "column_chunk": self.column_chunk,
            "p_top": self.p_top,
            "ozone_routing": self.ozone_routing,
            "statics_assets": {
                "rrtmg_lw_statics.npz": RRTMG_LW_STATICS_SHA256,
            },
        }
        if self.aer_opt == 2:
            identity["aer2_tables_sha256"] = _aer2.TABLES_SHA256
        if self.aer_opt == 1:
            from woof.ingest.rrtmg_coeffs import DATA_DIR
            import json
            identity["aer1_tables"] = json.loads((DATA_DIR / "rrtmg_aer1_manifest.json").read_text())
            identity["aer1_routing"] = "domain-1-climatology"
        if self.aer_opt == 3:
            from woof.core.rrtmg_aerosol_optics import AER3_TABLES_SHA256
            identity["aer3_tables_sha256"] = AER3_TABLES_SHA256
            if getattr(self, "aer3_type_source", "rural") != "rural":
                # Only a land/water run binds the maritime rows, so every
                # rural identity is the bytes it always was.
                from woof.core.rrtmg_aerosol_optics import (
                    AER3_MARITIME_TABLES_SHA256)
                identity["aer3_type_source"] = self.aer3_type_source
                identity["aer3_maritime_tables_sha256"] = (
                    AER3_MARITIME_TABLES_SHA256)
        if getattr(self, "_smoke_provider", None) is not None:
            identity["prescribed_smoke"] = self._smoke_provider.identity()
        if getattr(self, "_transported_smoke", None) is not None:
            identity["transported_smoke"] = {
                "state": self._transported_smoke,
                "optics": ("HRRR v4.1.21 simple_dir_fdb=1: 1e-6*4.5 m2/g*ug/kg*"
                           "(1/ALT)*dz8w, MIN(3, AOD), rural aer_opt=3")}
        if self.o3input == 2:
            # Constant pins only -- importing wrf_ozone performs no file I/O;
            # a child adapter still never CALLS its climatology chain.
            from woof.ingest.wrf_ozone import (OZONE_LAT_SHA256,
                                                OZONE_PLEV_SHA256,
                                                OZONE_SHA256)
            identity["ozone_assets"] = {
                "ozone.formatted": OZONE_SHA256,
                "ozone_lat.formatted": OZONE_LAT_SHA256,
                "ozone_plev.formatted": OZONE_PLEV_SHA256,
            }
        if not (self.longwave and self.shortwave):
            identity["spectra"] = {"lw": self.longwave, "sw": self.shortwave}
        if self.trace_gas_overrides:
            identity["trace_gas_overrides"] = dict(self.trace_gas_overrides)
        if self.ghg_input == 1:
            from woof.core.camtr_gases import camtr_identity
            identity["ghg_source"] = camtr_identity()
        return identity

    # ------------------------------------------------------------------
    def _check_pins(self, cfg):
        """Fail closed on any option outside the ported combination."""
        if int(getattr(cfg, "icloud", 1)) not in (1, 2):
            raise NotImplementedError(
                "icloud=0 lacks a recorded legacy clear-sky solve; "
                "icloud=3 requires unported cal_cldfra3 condensate adjustment")
        for name, pinned in _PINNED_OPTIONS.items():
            value = getattr(cfg, name, pinned)
            if int(value) != pinned:
                raise NotImplementedError(
                    f"rrtmg_legacy implements only {name}={pinned} "
                    f"(dossier section 1); got {name}={value!r} -- no "
                    "silent option substitution is applied")
        if int(getattr(cfg, "ghg_input", 0)) != self.ghg_input:
            raise ValueError(
                f"adapter was constructed for ghg_input={self.ghg_input} but "
                f"the run config requests ghg_input={cfg.ghg_input}; rebuild "
                "the adapter so its gases and restart identity match")
        aer_opt = int(getattr(cfg, "aer_opt", 0))
        if aer_opt not in _ADMITTED_AER_OPT:
            raise NotImplementedError(
                f"rrtmg_legacy implements aer_opt in {_ADMITTED_AER_OPT} "
                f"(0: no aerosol; 1: root-domain six-type climatology; 3: Thompson water- and ice-friendly "
                f"aerosol shortwave optics); got aer_opt={aer_opt!r} -- no "
                "silent option substitution is applied")
        if self.shortwave and aer_opt != self.aer_opt:
            raise ValueError(
                f"adapter was constructed for aer_opt={self.aer_opt} but the "
                f"run config requests aer_opt={aer_opt}; rebuild the adapter "
                "so its shortwave aerosol and restart identity match")
        requested_type = getattr(cfg, "rrtmg_aer3_type_source", "rural")
        if self.shortwave and requested_type != getattr(
                self, "aer3_type_source", "rural"):
            raise ValueError(
                "adapter was constructed for rrtmg_aer3_type_source="
                f"{getattr(self, 'aer3_type_source', 'rural')!r} but the run "
                f"config requests {requested_type!r}; rebuild the adapter so "
                "its shortwave aerosol type and restart identity match")
        if self.shortwave and bool(getattr(cfg, "rrtmg_smoke_manifest", "")) != (
                getattr(self, "_smoke_provider", None) is not None):
            raise ValueError("prescribed smoke configuration differs from the bound radiation adapter")
        if self.shortwave and bool(int(getattr(cfg, "simple_dir_fdb", 0) or 0)) != (
                getattr(self, "_transported_smoke", None) is not None):
            raise ValueError("simple_dir_fdb differs from the bound radiation "
                             "adapter's transported smoke; rebuild the adapter")
        requested_o3input = int(getattr(cfg, "o3input", 2))
        if requested_o3input != self.o3input:
            raise ValueError(
                f"adapter was constructed for o3input={self.o3input} but "
                f"the run config requests o3input={requested_o3input}; rebuild "
                "the adapter so its asset/routing identity matches")

    def _validate_radii_micron(self, name, eff_um, q):
        """Reject meter-scale radii state (dossier section 11 caveat)."""
        mask = (q > np.float32(0.0)) & (eff_um > np.float32(0.0))
        if bool(mask.any()) and float(eff_um[mask].max()) < 1.0e-3:
            raise ValueError(
                f"state.{name} looks meter-scale (every value on cloudy "
                f"points is < 1e-3, max {float(eff_um[mask].max()):.3e}) "
                "but the woof radii contract is MICRONS: this state was "
                "almost surely written before the Thompson radii-units "
                "fix.  Resuming it requires the cloud-radiation seam "
                "lane's restart migration; rrtmg_legacy will not silently "
                "rescale or radiate at clip floors")

    def _validate_radii_micron_device(self, name, eff_um, q):
        """:meth:`_validate_radii_micron` on (nz, ncol) device grids: the
        same mask (``q > 0`` and ``eff > 0``, decided on the float32 bits
        so a subnormal counts) and the same verdict, with one host read.
        An absent species (``q is None``) is the host path's zero column,
        whose mask is empty."""
        import cupy as cp
        if q is None:
            return
        mask = _f32_positive(q) & _f32_positive(eff_um)

        def verdict(top):
            top = float(top)
            if top != -np.inf and top < 1.0e-3:
                raise ValueError(
                    f"state.{name} looks meter-scale (every value on cloudy "
                    f"points is < 1e-3, max {top:.3e}) "
                    "but the woof radii contract is MICRONS: this state was "
                    "almost surely written before the Thompson radii-units "
                    "fix.  Resuming it requires the cloud-radiation seam "
                    "lane's restart migration; rrtmg_legacy will not silently "
                    "rescale or radiate at clip floors")

        # One host read per call, not per species (deferred_device_checks).
        from woof.core.deferred_device_checks import check
        check(cp.where(mask, eff_um, cp.float32(-np.inf)).max(), verdict)

    #: How many saturation widths one engine chunk may span when the card
    #: has the room (see :meth:`_wide_chunk`).
    WIDE_CHUNK_FACTORS = (4, 2)
    #: Free device memory kept back beyond the priced transient.
    WIDE_CHUNK_MARGIN_BYTES = 1 << 30

    def _wide_chunk(self, ncol, nz, p_top, nday):
        """Engine chunk width for this call: wider than one wave when it fits.

        THE BREAKAGE THIS PREVENTS (radiation-ranks, V61 8 x RTX 5090,
        nsys of a full-CONUS HRRR radiation step, 2026-10-21): the default
        width is one resident-thread wave (2,048 LW / 2,560 SW columns on a
        5090), so a 238,000-column rank slab runs 116 LW and 93 SW engine
        chunks, about 26,000 CUDA API calls (12,500 launches) per call.
        The card finished its 0.75 s of radiation kernels inside a 1.6-3.8 s
        span because the rank's host thread could not issue the chunks
        fast enough beside seven other rank threads.  Chunk width is
        workspace shape only (per-column results are bitwise identical at
        any width: the engines' own contract), so where the priced
        transient of a 4x or 2x width fits in the memory the card has free
        now, the call takes it and issues a quarter or half of the chunks.
        An explicit ``column_chunk`` is used as given; a card without the
        room keeps the one-wave width, exactly as before.
        """
        if self.column_chunk is not None:
            return self.column_chunk
        import cupy as cp
        base = int(_lw.LW_BATCH_COLUMN_CHUNK)
        if base >= ncol:
            return None
        free = _sw.sw_batch_free_device_bytes(cp)
        if free is None:
            return None
        for factor in self.WIDE_CHUNK_FACTORS:
            width = base * factor
            need = legacy_radiation_vram_bytes(
                ncol=ncol, nz=nz, p_top=p_top, column_chunk=width,
                ncol_day=nday, lw_coefficients=self._C,
                longwave=self.longwave, shortwave=self.shortwave,
                o3input=self.o3input, aer_opt=self.aer_opt)
            if need + self.WIDE_CHUNK_MARGIN_BYTES <= free:
                return width
        return None

    def _mcica_generator(self, gpu_entry):
        """Device McICA twin, resolved through the module attribute at
        call time so the wiring booby-traps can intercept it."""
        probe = self._stage_probe
        if probe is None:
            return gpu_entry
        return partial(gpu_entry,
                       _stage_probe=lambda c0, nc: probe("mcica"))

    # ------------------------------------------------------------------
    # Every device-side guard of the call (the LW/SW cloud aborts, the SW
    # day contract, the McICA seed guard, the radii unit check) is read
    # once when the call ends instead of once per chunk, so the host
    # queues the next chunk while the card runs this one
    # (woof.core.deferred_device_checks, A3).  A failed guard raises the
    # same error before any result leaves this call.
    @_with_deferred_device_checks
    def __call__(self, *, atmosphere, fields, state, cfg):
        import cupy as cp

        from woof.core.physics import RadiationResult
        from woof.core import rrtmgp as _rrtmgp

        self._check_pins(cfg)
        pressure = atmosphere["pressure"]
        nz, ny, nx = pressure.shape
        ncol = ny * nx
        from woof.core.chem_radiation import prepare
        chem_aerosols = prepare(state, cfg)
        if self.latitude_deg.shape != (ny, nx):
            raise ValueError(
                "radiation latitude/longitude must match state grid")
        p_top = self.p_top
        if p_top is None:
            declared = getattr(state, "p_top", None)
            if declared is None:
                raise ValueError(
                    "rrtmg_legacy needs p_top (constructor argument or "
                    "state.p_top) to build WRF's Cavallo buffer layers")
            p_top = float(declared)
        nlayers, sw_layers = legacy_radiation_layer_counts(nz, p_top)
        if self.longwave and nlayers > MAX_LONGWAVE_LAYERS:
            raise ValueError(
                f"LW nlayers={nlayers} exceeds the batched engine's "
                f"{MAX_LONGWAVE_LAYERS}-layer bound "
                "(rlw_rtrn_march RLW_MAXLAY)")

        mp_physics = int(getattr(cfg, "mp_physics", 0))
        warm_rain = mp_physics == 1
        icloud = int(getattr(cfg, "icloud", 1))
        # cal_cldfra1's two Registry package flags, resolved SEPARATELY.
        # One fused boolean cannot express P3's F_QI=true/F_QS=false, and
        # collapsing it sent mp=50 to the qc-only arm.
        f_qi, f_qs = legacy_cloud_fraction_flags(mp_physics)
        sf_surface_physics = int(getattr(cfg, "sf_surface_physics", 2))

        adapter_gpu_preflight()
        blk = _grid_launch

        # ---- device grids: every (nk, ny, nx) field as (nk, ncol) on the
        # device; column blocks are gathered per chunk below (data
        # movement only), so no field makes a host round trip. -----------
        g_p3d = _device_grid(pressure, nz)
        g_p8w = _device_grid(atmosphere["p_interface"], nz + 1)
        g_t3d = _device_grid(atmosphere["temperature"], nz)
        g_pi3d = _device_grid(atmosphere["exner"], nz)
        g_dz8w = _device_grid(atmosphere["dz"], nz)
        g_zw = _device_grid(atmosphere["z_interface"], nz + 1)
        smoke_gather = None
        smoke_provider = getattr(self, "_smoke_provider", None)
        if smoke_provider is not None:
            profile = smoke_provider.at(float(state.elapsed_seconds),
                latitude_deg=self.latitude_deg, longitude_deg=self.longitude_deg)
            quantity = profile["quantity"]
            if quantity == "layer_aod":
                smoke_aod = profile["value"]
            else:
                smoke_mass = profile["value"]
                if quantity == "posted_mass_concentration":
                    smoke_mass = _aer3.smoke_dry_mixing_ratio_from_posted_density_device(
                        smoke_mass, profile["donor_p"], profile["donor_t"])
                # ALT is dry specific volume. The shared atmosphere rho
                # includes vapor mass and is not the source dry density.
                dry_density = _aer3.smoke_dry_air_density_device(state.alt)
                smoke_aod = _aer3.smoke_aod_from_dry_mixing_ratio_device(
                    smoke_mass, dry_density, atmosphere["dz"])
                del smoke_mass, dry_density
            smoke_gather = _ColumnGather((_device_grid(smoke_aod, nz),), ncol)
            del profile, smoke_aod
        transported = getattr(self, "_transported_smoke", None)
        if transported is not None:
            smoke_mass = getattr(state, transported, None)
            if smoke_mass is None:
                raise ValueError(
                    f"simple_dir_fdb=1 reads transported smoke from "
                    f"state.{transported} and this state carries none; the "
                    "plume would not dim the sun")
            smoke_aod = _aer3.smoke_aod_from_transported_device(
                smoke_mass, state.alt, atmosphere["dz"])
            smoke_gather = _ColumnGather((_device_grid(smoke_aod, nz),), ncol)
            del smoke_mass, smoke_aod

        g_moist = {}
        f_flags = {}
        for name in ("qv", "qc", "qr", "qi", "qs", "qg"):
            value = getattr(state, name, None)
            if value is None:
                g_moist[name] = None
                f_flags[name] = False
            else:
                g_moist[name] = _device_grid(value, nz)
                f_flags[name] = True

        # ---- aer_opt = 3: the Thompson aerosol numbers the shortwave
        # optics read (woof.core.rrtmg_aerosol_optics), gathered per SW
        # chunk beside the wrapper inputs.
        aer_gather = None
        if self.shortwave and self.aer_opt == 3:
            g_aer = []
            for name in ("nwfa", "nifa"):
                value = getattr(state, name, None)
                if value is None:
                    raise ValueError(
                        f"aer_opt=3 reads the Thompson aerosol numbers "
                        f"(state.nwfa, state.nifa; mp_physics=28) and this "
                        f"state carries no {name}: the shortwave would run "
                        "zero aerosol under aer_opt=3")
                g_aer.append(_device_grid(value, nz))
            aer_gather = _ColumnGather(g_aer, ncol)

        # ---- radii: retain the MICRON state contract through prep -----
        # has_req* follows WRF v4.6.1's SCHEME TABLE, not field presence
        # (module_physics_init.F:987-1033, gated on use_mp_re=1, the
        # Registry default and the campaign setting): Thompson, NSSL
        # 2-moment, the WSM/WDM families, and P3 declare radii; MORRISON
        # DOES NOT -- WRF's Morrison+RRTMG runs has_req*=0 with the
        # wrapper's relcalc/reicalc temperature radii, and Morrison's
        # EFFI upper bound (3*(2*DCS+100um)/2 = 525 um) would otherwise
        # reach cldprmc's [5,140] wrf_error_fatal.  Presence-based
        # detection reproduced exactly that fatal on a real Morrison
        # forecast (integration finding, 2026-07-27).
        #
        # The three flags are read PER FIELD because WRF's answer is not
        # uniform: its :1027-1033 override re-zeroes has_reqs for P3, so
        # mp=50 is (1, 1, 0) -- cloud and ice radii taken from the scheme,
        # no snow radius asked for and none allocated on P3 state.  A
        # single boolean here would have to either invent a P3 snow radius
        # or throw away the two radii P3 does predict.
        has_reqc, has_reqi, has_reqs = legacy_scheme_has_req(
            mp_physics, getattr(cfg, "use_mp_re", 1))
        g_radii = {}
        has_req = {}
        for name, key, q, scheme_declares in (
                ("effc", "re_cloud", "qc", has_reqc),
                ("effi", "re_ice", "qi", has_reqi),
                ("effs", "re_snow", "qs", has_reqs)):
            value = getattr(state, name, None)
            if not scheme_declares:
                # WRF semantics: the wrapper never reads scheme radii
                # for this mp; relcalc/reicalc take over inside prep.
                g_radii[key] = None
                has_req[name] = 0
                continue
            if value is None:
                raise ValueError(
                    f"mp_physics={int(mp_physics)} declares {name} to "
                    "radiation (WRF use_mp_re scheme table) but "
                    f"state.{name} is missing -- the microphysics radii "
                    "contract is broken; refusing to radiate fallback "
                    "radii silently")
            eff_um = _device_grid(value, nz)
            self._validate_radii_micron_device(name, eff_um, g_moist[q])
            g_radii[key] = eff_um
            has_req[name] = 1

        surf = {name: _device_flat(fields[name])
                for name in ("tsk", "emiss", "albedo", "xland", "xice",
                             "snow")}
        lat_flat = self.latitude_deg.reshape(-1)
        lon_flat = self.longitude_deg.reshape(-1)
        lat_d = cp.asarray(lat_flat)
        fnm_d = self._vertical_weight_device(state.fnm)
        fnp_d = self._vertical_weight_device(state.fnp)

        active_bl = icloud == 1 and mynn_bl_cloud_active(
            getattr(cfg, "bl_pbl_physics", 0), getattr(cfg, "icloud_bl", 0))
        if active_bl:
            g_bl = {}
            for name in ("qc_bl", "qi_bl", "cldfra_bl"):
                if fields.get(name) is None:
                    raise ValueError(
                        "MYNN radiation coupling requires QC_BL, QI_BL, "
                        "and CLDFRA_BL")
                g_bl[name] = _device_grid(fields[name], nz)
            itimestep = wrf_itimestep(state.elapsed_seconds, cfg.dt)
            if type(itimestep) is not int or itimestep < 1:
                raise ValueError(
                    "MYNN radiation itimestep must be a positive int")
        else:
            g_bl = None
            itimestep = 1

        # ---- calendar / solar (dossier sections 4, 9.1) ---------------
        valid_time = (self.start_time
                      + timedelta(seconds=float(state.elapsed_seconds)))
        yr = int(valid_time.year)
        julday = int(valid_time.timetuple().tm_yday)
        hour = (valid_time.hour + valid_time.minute / 60.0
                + valid_time.second / 3600.0
                + valid_time.microsecond / 3.6e9)
        julian = F((julday - 1) + hour / 24.0)
        aer1_grid = None
        aodtot = None
        if self.shortwave and self.aer_opt == 1:
            monthly = getattr(state, "aerodm", None)
            if monthly is None:
                monthly = _aer3.aer1_climatology_device(
                    cp.asarray(self.latitude_deg.ravel()), cp.asarray(self.longitude_deg.ravel()),
                    self._sw_tables.aer1_climatology)
                state.aerodm = monthly
            aer1_grid, aodtot, _ = _aer3.aer1_time_pressure_device(
                monthly, julian, cp.ascontiguousarray(g_p3d.T), cp.ascontiguousarray(g_p8w.T),
                self._sw_tables.aer1_climatology)
        gmt = F(self.start_time.hour + self.start_time.minute / 60.0
                + self.start_time.second / 3600.0
                + self.start_time.microsecond / 3.6e9)
        xtime = F(float(state.elapsed_seconds) / 60.0)
        radt_minutes = F(cfg.radt if cfg.radt > 0.0 else cfg.radt_minutes)
        # WRF: hour angle at the interval midpoint, declination/solcon at
        # call time (module_radiation_driver.F:1206-1208).  Evaluated on the
        # device with WOOF's own float32 sinf/cosf/asinf: the NumPy twins
        # (radconst, calc_coszen) differ from WRF's Fortran by up to 64 ULP
        # in COSZEN (tools/rrtmg_legacy_column_oracle).
        xt_mid = xtime + radt_minutes * F(0.5)
        generation = getattr(cfg, "solar_generation", "wrf_480")
        from woof.wrf_exact import ENABLED
        if ENABLED and generation == "wrf_461":
            coszen_d, _hrang_d, declin, solcon = solar_geometry_device(
                julian, xt_mid, gmt, lat_d, cp.asarray(lon_flat))
            coszen = cp.asnumpy(coszen_d)
        elif generation == "wrf_480":
            from woof.core.solar import solar_geometry_device as _solar_device
            ds, coszen_d, _hrang = _solar_device(julian, xt_mid, gmt, lat_flat, lon_flat)
            declin, solcon = map(F, cp.asnumpy(ds))
            coszen = cp.asnumpy(coszen_d)
        else:
            declin, solcon = radconst(julian)
            coszen = calc_coszen(julian, xt_mid, gmt, lat_flat, lon_flat,
                                 declin, generation='wrf_461')
            coszen_d = cp.asarray(coszen)

        # ---- o33d: WRF's root-compute + parent->child routing ----------
        # Keep the interpolated grid for all chunk gathers. The retained
        # host field remains the provider and restart contract.
        grid_o3 = None
        if self.o3input == 0:
            # The wrapper ignores o33d in this mode and builds O3DATA from
            # pressure and latitude.  Shape-correct zeros keep one batch
            # contract for both modes.
            self._o33d_grid = None
        elif self._ozone_provider is not None:
            driver = getattr(state, "physics", None)
            owner = getattr(driver, "cam_ozone", None)
            if owner is not None and owner.mode == "parent-interpolated":
                if driver.call_counts.get("cam_ozone", 0) == 0:
                    raise RuntimeError("child radiation ran before parent CAM ozone FORCE")
                source = driver.o3rad
                if (isinstance(source, cp.ndarray)
                        and source.dtype == cp.float32
                        and source.shape == (nz, ny, nx)):
                    # Retain a snapshot, not an alias: a later parent FORCE
                    # can update the carrier before this child radiates again.
                    snapshot = source.copy(order="C")
                    self.__dict__["_o33d_host"] = None
                    self.__dict__["_o33d_device"] = snapshot
                    grid_o3 = snapshot.reshape(nz, ncol)
                else:
                    grid = cp.asnumpy(source)
                    o33d = np.ascontiguousarray(grid.transpose(1, 2, 0).reshape(-1, nz))
            else:
                o33d = np.asarray(self._ozone_provider(), np.float32)
            if grid_o3 is None:
                if o33d.shape != (ncol, nz):
                    raise ValueError(
                        f"ozone_parent provider returned shape {o33d.shape}; "
                        f"this child grid needs ({ncol}, {nz})")
                # Host providers retain their existing conversion contract.
                self._o33d_grid = np.ascontiguousarray(
                    o33d.reshape(ny, nx, nz).transpose(2, 0, 1))
                grid_o3 = cp.asarray(self._o33d_grid.reshape(nz, ncol))
        else:
            # One host buffer for every call (the native route fills it in
            # place): a fresh ~200 MB array per call per rank was faulted in
            # at every 1 km radiation step (A3).
            ozmixt = np.asarray(self._ozone.ozn_latitude_time_int(
                julday, julian, self.latitude_deg.reshape(-1), self._ozone_climo,
                out=self.__dict__.get("_ozmixt_buf")))
            self.__dict__["_ozmixt_buf"] = ozmixt
            pin = np.asarray(self._ozone_climo.plev)
            if pin.dtype != np.float32 or pin.ndim != 1:
                raise ValueError("pin must be float32 (levsiz,)")
            if not np.all(np.diff(pin) > 0):
                # WRF's runtime guard (ozn_p_int's kount > ncol check).
                raise ValueError("OZN_P_INT: Bad ozone data: "
                                 "non-monotonicity suspected")
            if (ozmixt.dtype != np.float32
                    or ozmixt.shape != (ncol, pin.size)):
                raise ValueError("ozmixt must be float32 (ncol, levsiz)")
            ozmixt = np.ascontiguousarray(ozmixt)
            pin_d = cp.asarray(pin)
            levsiz = int(pin.size)
            # ozn_p_int on the device over the (nz, ncol) grid, once, for
            # the field children and restarts retain and chunks reuse.
            grid_o3 = cp.empty((nz, ncol), dtype=cp.float32)
            ozmixt_d = cp.asarray(ozmixt)
            _adapter_kernel("rla_ozn_p_int")(
                ((ncol + 127) // 128,), (128,),
                (np.int32(ncol), np.int32(nz), np.int32(levsiz),
                 g_p3d, np.int64(ncol), np.int64(1), pin_d, ozmixt_d,
                 grid_o3, np.int64(ncol), np.int64(1)))
            del ozmixt_d
            # Retain for child domains (their providers read this field,
            # so a nest's ozone updates exactly when its parent's does).
            self.__dict__["_o33d_host"] = None
            self.__dict__["_o33d_device"] = grid_o3.reshape(nz, ny, nx)
            del pin_d

        shared = dict(
            icloud=icloud, warm_rain=warm_rain, cldovrlp=2, idcor=0,
            o3input=self.o3input,
            rrtmg_cloud_optics_form=getattr(cfg, "rrtmg_cloud_optics_form", "wrf_461"),
            has_reqc=has_req["effc"], has_reqi=has_req["effi"],
            has_reqs=has_req["effs"], f_qc=f_flags["qc"],
            f_qr=f_flags["qr"], f_qi=f_flags["qi"], f_qs=f_flags["qs"],
            f_qg=f_flags["qg"], yr=yr, julian=julian,
            mp_physics=mp_physics)
        ice_rule = bool(has_req["effs"])

        # Every grid a chunk reads, gathered in one launch per chunk.
        gather_grids = {"p3d": g_p3d, "t3d": g_t3d, "zw": g_zw,
                        "p8w": g_p8w, "dz8w": g_dz8w}
        if grid_o3 is not None:
            gather_grids["o33d"] = grid_o3
        gather_grids.update((name, g) for name, g in g_moist.items()
                            if g is not None)
        gather_grids.update((key, g) for key, g in g_radii.items()
                            if g is not None)
        if active_bl:
            gather_grids.update(g_bl)
        if icloud == 2:
            # One driver CLDFRA field is shared by LW and SW. Gather only
            # repacks it into each prep's column window; it never recomputes
            # the threshold on separate moisture buffers.
            zero_cloud = cp.zeros((nz, ncol), dtype=cp.float32)
            gather_grids["binary_cldfra"] = _rrtmgp.cal_cldfra2(
                zero_cloud if g_moist["qc"] is None else g_moist["qc"],
                zero_cloud if g_moist["qi"] is None else g_moist["qi"],
                f_qc=mp_physics != 0, f_qi=f_qi)
            del zero_cloud
        gather_names = list(gather_grids)
        gather = _ColumnGather(gather_grids.values(), ncol)

        def chunk_inputs(sel):
            """The prep keyword set for columns ``sel`` (a slice or a device
            index array), built
            on the device in the host adapter's order:
            columns, t8w, cal_cldfra1 on the raw moisture, radii in microns,
            then the MYNN merge (supplied masks first, then fraction, mass
            and radii).  Every step is per column, so a chunk equals the
            same columns of a whole-grid evaluation bitwise."""
            blocks = dict(zip(gather_names, gather(sel)))
            p3d = blocks["p3d"]
            nc = p3d.shape[0]
            t3d = blocks["t3d"]
            zw = blocks["zw"]
            t8w = cp.empty((nc, nz + 1), dtype=cp.float32)
            _adapter_kernel("rla_t8w")(
                *blk(nc * (nz + 1)),
                (np.int32(nc), np.int32(nz), t3d, zw, fnm_d, fnp_d, t8w))
            del zw
            moist = {}
            for name in ("qv", "qc", "qr", "qi", "qs", "qg"):
                g = g_moist[name]
                moist[name] = (cp.zeros((nc, nz), dtype=cp.float32)
                               if g is None else blocks[name])
            # cal_cldfra1 (icloud=1), the WRF transcription already gated
            # on the RTE+RRTMGP path -- imported, not copied; fed the raw
            # moisture before the MYNN merge writes qc/qi.
            if icloud == 2:
                cldfra = blocks["binary_cldfra"]
            else:
                cldfra = _rrtmgp.cal_cldfra1(
                    moist["qv"], moist["qc"], moist["qi"], moist["qs"], t3d,
                    p3d, f_qc=True, f_qi=f_qi, f_qs=f_qs, mp_physics=mp_physics)
            # Microphysics already applied the stock wrapper's metre to
            # micron multiply. Undoing and redoing it loses one ULP in
            # some radii on the next radiation call, so a strict run's prep
            # accepts these canonical microns directly. MYNN only resets
            # supplied cloud radii to zero, so its operation is unit
            # independent.
            radii = {
                key: None if g is None else blocks[key]
                for key, g in g_radii.items()}
            # Default retains its original metre conversion and rounding.
            if not ENABLED:
                radii = _radius_meters_blocks(radii)
            if active_bl:
                bl = {k: blocks[k] for k in g_bl}
                n = moist["qc"].size
                null = np.uint64(0)
                # The source cloud driver merges mass without rewriting
                # microphysics radii (v4.1.21 radiation_driver.F:1256-1304).
                source_bl = getattr(cfg, "bl_mynn_version", "wrf_461") == "gsd_41"
                # Strict mode follows the native wrapper's sizing policy:
                # adding MYNN water does not rewrite the supplied radii.
                # The ordinary arm retains its unsized-cloud policy.
                keep_source_radii = ENABLED or source_bl or getattr(
                    cfg, "rrtmg_cloud_optics_form", "wrf_461") == "noaa_wrf39"
                if source_bl:
                    # GSD MYNN v4.1 hands radiation in-cloud QC_BL; its
                    # merge multiplies by CLDFRA_BL and splits the phase
                    # at radiation time (rla_mynn_gsd41).
                    _adapter_kernel("rla_mynn_gsd41")(
                        *blk(n),
                        (np.int64(n), moist["qc"], moist["qi"], bl["qc_bl"],
                         bl["cldfra_bl"], t3d, cldfra,
                         np.int32(1 if itimestep == 1 else 0),
                         null if keep_source_radii or radii["re_cloud"] is None
                         else radii["re_cloud"],
                         null if keep_source_radii or radii["re_ice"] is None
                         else radii["re_ice"],
                         np.int32(1 if ice_rule else 0),
                         F(MERGE_QC_BELOW), F(MERGE_QI_BELOW),
                         F(MERGE_CLDFRA_BL_ABOVE)))
                else:
                    _adapter_kernel("rla_mynn")(
                        *blk(n),
                        (np.int64(n), moist["qc"], moist["qi"],
                         bl["qc_bl"], bl["qi_bl"], bl["cldfra_bl"], cldfra,
                         np.int32(1 if itimestep != 1 else 0),
                         null if keep_source_radii or radii["re_cloud"] is None
                         else radii["re_cloud"],
                         null if keep_source_radii or radii["re_ice"] is None
                         else radii["re_ice"],
                         np.int32(1 if ice_rule else 0),
                         F(MERGE_QC_BELOW), F(MERGE_QI_BELOW),
                         F(MERGE_CLDFRA_BL_ABOVE)))
                del bl
            o33d = (blocks["o33d"] if grid_o3 is not None else
                    cp.zeros((nc, nz), dtype=cp.float32))
            sel_kw = dict(
                radii_units="microns" if ENABLED else "meters",
                p3d=p3d, p8w=blocks["p8w"], t3d=t3d, t8w=t8w,
                dz8w=blocks["dz8w"], qv3d=moist["qv"],
                qc3d=moist["qc"], qr3d=moist["qr"], qi3d=moist["qi"],
                qs3d=moist["qs"], qg3d=moist["qg"], cldfra3d=cldfra,
                o33d=o33d, tsk=surf["tsk"][sel], xland=surf["xland"][sel],
                xice=surf["xice"][sel], snow=surf["snow"][sel],
                xlat=lat_d[sel])
            for key in ("re_cloud", "re_ice", "re_snow"):
                if radii[key] is not None:
                    sel_kw[key] = radii[key]
            return sel_kw

        from woof.core.trace_gases import LEGACY_LW_GASES, LEGACY_SW_GASES, trace_gas_subset
        lw_gases = trace_gas_subset(self.trace_gas_overrides, LEGACY_LW_GASES) or None
        sw_gases = trace_gas_subset(self.trace_gas_overrides, LEGACY_SW_GASES) or None
        file_gases = None
        if self.ghg_input == 1:
            # RRTMG_LWRAD/RRTMG_SWRAD: IF (GHG_INPUT .EQ. 1) CALL
            # read_CAMgases(yr, julian, ...) at every call
            # (module_ra_rrtmg_lw.F:11968, module_ra_rrtmg_sw.F:10553).
            # Explicit overrides still win, as they win over ghg_input=0.
            from woof.core.camtr_gases import camtr_trace_gases
            from woof.core.trace_gases import FileGases
            file_gases = camtr_trace_gases(yr, julian)
            lw_gases = FileGases({**trace_gas_subset(file_gases, LEGACY_LW_GASES),
                                  **(lw_gases or {})})
            sw_gases = FileGases({**trace_gas_subset(file_gases, LEGACY_SW_GASES),
                                  **(sw_gases or {})})
        day_seconds = F(86400.0)

        # ---- LW: all columns, adapter chunk == engine chunk -----------
        wide = self._wide_chunk(ncol, nz, p_top, int(np.count_nonzero(coszen > F(0.0))))
        chunk_lw = wide or _lw.LW_BATCH_COLUMN_CHUNK
        # Result grids live on the device for the whole call. Night SW
        # columns retain the driver's positive zero initialization.
        rthratenlw = cp.zeros((nz, ncol), dtype=cp.float32)
        glw = cp.zeros(ncol, dtype=cp.float32)
        olr = cp.zeros(ncol, dtype=cp.float32)
        from woof.core.physics_inventory import RADIATION_BOUNDARY_FLUXES
        boundary_fluxes = {
            name: (fields[name].reshape(-1) if name in fields else cp.zeros(ncol,dtype=cp.float32))
            for name in RADIATION_BOUNDARY_FLUXES
            if (self.longwave if name.startswith("lw") else self.shortwave)}
        for name,value in boundary_fluxes.items():
            if name.startswith("sw"):
                value.fill(0)
        rthratensw = cp.zeros((nz, ncol), dtype=cp.float32)
        gsw = cp.zeros(ncol, dtype=cp.float32)
        # SWDDIR/SWDDIF, rrtmg_swrad's swdkdir/swdkdif at the surface
        # (module_ra_rrtmg_sw.F, the driver's jararias 2013 arm;
        # swdkdif(1) = swdflx(1) - swdkdir(1), :9526, :11524), zero at night
        # as the radiation driver zeroes them (:1724-1726).  SWDDIR is built
        # only when BEP+BEM (sf_urban_physics = 3) allocated it in the
        # surface fields; SWDDIF also when slope_rad asks for it
        # (woof.core.topo_radiation).  Off, nothing is allocated.
        swddir = swddif = None
        if self.shortwave and "swddir" in fields:
            swddir = cp.zeros(ncol, dtype=cp.float32)
        if self.shortwave and (swddir is not None or getattr(
                self, "surface_diffuse_requested", False)):
            swddif = cp.zeros(ncol, dtype=cp.float32)
        # CLDFRA for the driver's held output buffer: the chunk prep's
        # cal_cldfra1 + MYNN-merge fraction (cldfra3d, what WRF's driver
        # leaves in grid%cldfra), copied out per chunk in column order.
        # The longwave loop covers every column; a shortwave-only adapter
        # covers only daylight ones, so it builds the cloud inputs for all
        # columns once more (cost only: chunk_inputs is per column and
        # touches nothing outside its own gathered blocks).
        cldfra_cols = cp.empty((ncol, nz), dtype=cp.float32)
        span_lw = adapter_span(chunk_lw, ncol)
        for c0 in range(0, 0 if self.longwave else ncol, span_lw):
            c1 = min(c0 + span_lw, ncol)
            kw = chunk_inputs(slice(c0, c1))
            cldfra_cols[c0:c1] = kw["cldfra3d"]
            del kw
        for c0 in range(0, ncol if self.longwave else 0, span_lw):
            c1 = min(c0 + span_lw, ncol)
            # ``sel`` is the span's column selector, and the ONE site
            # woof/ensemble/batch_physics_init.py's member adapter rewrites
            # (to the member-interleaved device indices of logical columns
            # [c0, c1)); every per-column read of the span goes through it,
            # and the windows below pass logical offsets c0 + w0 to the
            # output kernel, which the adapter's kernel source interleaves.
            sel = slice(c0, c1)
            kw = chunk_inputs(sel)
            cldfra_cols[c0:c1] = kw["cldfra3d"]
            # The wrapper glue on the device: rrtmg_legacy_device holds
            # lwrad_prep_batch's statements bitwise
            # (tests/test_rrtmg_legacy_prep_device.py).  The McICA slabs
            # are deferred to the engine windows below.
            pl = _prep_device.lwrad_prep_batch_device(
                **kw, emiss=surf["emiss"][sel],
                nlayers=nlayers, trace_gas_overrides=lw_gases,
                subcolumn_generator=self._mcica_generator(
                    _mcica.gpu_generate_lw_subcolumns),
                defer_subcolumns=True, **shared)
            del kw
            window = pl.pop("mcica_window")
            # The WRF-level extras are not engine inputs: drop them before
            # the engine runs (legacy_radiation_vram_bytes prices what the
            # engine phase holds).
            for extra in _LW_PREP_EXTRAS:
                pl.pop(extra, None)
            if self._stage_probe is not None:
                self._stage_probe("prep")
            for w0 in range(0, c1 - c0, chunk_lw):
                w1 = min(w0 + chunk_lw, c1 - c0)
                rows = slice(w0, w1)
                nc = w1 - w0
                # The window's columns through the span selector, so a
                # member adapter's interleaved indices reach the chemistry
                # optics and the boundary-flux rows as they did per chunk.
                wsel = (slice(c0 + w0, c0 + w1) if isinstance(sel, slice)
                        else sel[w0:w1])
                tauaer = (pl["tauaer"][rows] if chem_aerosols is None else
                          chem_aerosols.legacy(wsel, int(pl["nlay"]), "lw"))
                mc = window(w0, w1)
                res = _lw.gpu_rrtmg_lw_batched_device(
                    nc, pl["nlay"], pl["icld"], pl["play"][rows],
                    pl["plev"][rows], pl["tlay"][rows], pl["tlev"][rows],
                    pl["tsfc"][rows], pl["h2ovmr"][rows], pl["o3vmr"][rows],
                    pl["co2vmr"][rows], pl["ch4vmr"][rows],
                    pl["n2ovmr"][rows], pl["o2vmr"][rows],
                    pl["cfc11vmr"][rows], pl["cfc12vmr"][rows],
                    pl["cfc22vmr"][rows], pl["ccl4vmr"][rows],
                    pl["emis"][rows], pl["inflglw"], pl["iceflglw"],
                    pl["liqflglw"], mc["cldfmcl"], mc["taucmcl"],
                    mc["ciwpmcl"], mc["clwpmcl"], mc["cswpmcl"],
                    mc["reicmcl"], mc["relqmcl"], mc["resnmcl"],
                    tauaer, self._C, column_chunk=chunk_lw,
                    mcica_layout="column", _stage_probe=self._stage_probe)
                del mc, tauaer
                nl = int(res["hr"].shape[1])
                # Map directly into the global grids and read Exner there.
                _adapter_kernel("rla_lw_out_grid")(
                    *blk(nc * nz),
                    (np.int32(nc), np.int32(nz), np.int32(nl), res["hr"],
                     res["uflx"], res["dflx"], g_pi3d, day_seconds,
                     np.int64(ncol), np.int64(c0 + w0), rthratenlw, glw,
                     olr))
                for suffix, key, level in (
                        ("upt","uflx",-1),("uptc","uflxc",-1),
                        ("dnt","dflx",-1),("dntc","dflxc",-1),
                        ("upb","uflx",0),("upbc","uflxc",0),
                        ("dnb","dflx",0),("dnbc","dflxc",0)):
                    boundary_fluxes["lw"+suffix][wsel] = res[key][:,level]
                del res
            del pl, window

        # ---- SW: driver-level zeroing (dossier section 3) + day gather
        day_idx = np.nonzero(coszen > F(0.0))[0]
        night_idx = np.nonzero(coszen <= F(0.0))[0]
        if self.shortwave and night_idx.size:
            # Wrapper-level night contract: COSZR + the SW_NIGHT_ZEROED
            # list.  None of these reach RadiationResult, but the write
            # keeps the wrapper contract exercised (dossier section 3).
            self._night_outputs = _prep.swrad_night_outputs_batch(
                coszen[night_idx])
        else:
            self._night_outputs = None
        # The engine sizes the chunk to this device once per layer count
        # (swrad_prep_batch's nlay is kte + 1 = nz + 1, as the pricing
        # assumes); an explicit column_chunk is untouched.
        chunk_sw = (wide
                    or (_cuda_sw(self._sw_tables).batch_column_chunk(nz + 1)
                        if self.shortwave else _sw.SW_BATCH_COLUMN_CHUNK))
        aerosol_aod = cp.empty((nz,ncol),cp.float32) if self.aer_opt==2 else None
        aerosol_diag = cp.empty((ncol,4),cp.float32) if self.aer_opt==2 else None
        user_inputs = _aer2.auxiliary_inputs(cfg,self.start_time,state.elapsed_seconds,(ny,nx)) if self.aer_opt==2 else {}
        def user_optics(kw,sel):
            if self.aer_opt != 2:
                return None
            selected = {key:cp.asarray(value,dtype=cp.float32).reshape(-1)[sel]
                        for key,value in user_inputs.items()}
            produced = _aer2.optics_device(
                kw["p3d"],kw["t3d"],kw["qv3d"],kw["dz8w"],_device_flat(state.ht)[sel],
                aer_type=cfg.aer_type,options=tuple(getattr(cfg,k) for k in _aer2.OPTION_KEYS),
                values=tuple(getattr(cfg,k) for k in _aer2.VALUE_KEYS),
                inputs=selected,nlayers=nz+1)
            aerosol_aod[:,sel] = produced[3].T
            aerosol_diag[sel,:] = produced[4]
            return produced[0],produced[2],produced[1]
        if self.aer_opt == 2:
            for off in range(0,night_idx.size,chunk_sw):
                sel=cp.asarray(night_idx[off:off+chunk_sw].astype(np.int64))
                user_optics(chunk_inputs(sel),sel)
        day_idx_d = cp.asarray(day_idx.astype(np.int64))
        span_sw = adapter_span(chunk_sw, day_idx.size)
        for a0 in range(0, day_idx.size if self.shortwave else 0, span_sw):
            sel = day_idx_d[a0:a0 + span_sw]
            kw = chunk_inputs(sel)
            # Optional qualification hook. Off, no extra arrays or transfers.
            # It records all/clear flux at the same radiation-call sun; a
            # per-step interpolated SWDOWN must not be mixed with held clear
            # flux to infer cloud attenuation.
            diagnostic_sink = getattr(self, "_sw_diagnostic_sink", None)
            diagnostic_cloud = (kw["cldfra3d"] if diagnostic_sink else None)
            diagnostic_aerosol = None
            aerosol = user_optics(kw,sel)
            if aer_gather is not None:
                # The driver builds the optics from its own p, t, qv and
                # dz8w before RRTMG_SWRAD runs (module_radiation_driver.F
                # :975, :1890); the wrapper's layer above the model top
                # stays 0/1/0, so the slabs span the SW engine's nz + 1.
                nwfa_c, nifa_c = aer_gather(sel)
                smoke_options = {}
                if smoke_gather is not None:
                    smoke_options = dict(smoke_aod=smoke_gather(sel)[0], smoke_feedback=True)
                if self.aer3_type_source == "landmask":
                    # NOAA MPAS driver :583-586, from the xland this
                    # shortwave call reads.
                    smoke_options["aer_type"] = _aer3.aer_type_from_xland_device(
                        surf["xland"][sel])
                aerosol_result = _aer3.aer3_sw_optics_device(
                    kw["p3d"], kw["t3d"], kw["qv3d"], kw["dz8w"], nwfa_c,
                    nifa_c, nz + 1, **smoke_options)
                aerosol = aerosol_result[:3]
                if diagnostic_sink is not None:
                    diagnostic_aerosol = dict(nwfa=nwfa_c, nifa=nifa_c,
                                              taod550=aerosol_result[3])
                    if smoke_options:
                        diagnostic_aerosol["smoke_aod"] = smoke_options["smoke_aod"]
                del smoke_options
                del aerosol_result
                del nwfa_c, nifa_c
            if aer1_grid is not None:
                aerosol = _aer3.aer1_sw_optics_device(
                    cp.ascontiguousarray(aer1_grid[sel]), nz + 1, self._sw_tables)
                if diagnostic_sink is not None:
                    diagnostic_aerosol = dict(aerod=aer1_grid[sel], aodtot=aodtot[sel])
            ps = _prep_device.swrad_prep_batch_device(
                **kw, albedo=surf["albedo"][sel],
                xcoszen=coszen_d[sel], solcon=solcon,
                sf_surface_physics=sf_surface_physics, trace_gas_overrides=sw_gases,
                subcolumn_generator=self._mcica_generator(
                    _mcica.gpu_generate_sw_subcolumns),
                defer_subcolumns=True, **shared)
            del kw
            window = ps.pop("mcica_window")
            for extra in _SW_PREP_EXTRAS:
                ps.pop(extra, None)
            if self._stage_probe is not None:
                self._stage_probe("prep")
            nspan = int(ps["ncol"])
            for w0 in range(0, nspan, chunk_sw):
                w1 = min(w0 + chunk_sw, nspan)
                rows = slice(w0, w1)
                nc = w1 - w0
                selw = sel[rows]
                if chem_aerosols is not None:
                    aerosol_w = chem_aerosols.legacy(selw, int(ps["nlay"]), "sw")
                else:
                    aerosol_w = (None if aerosol is None
                                 else tuple(a[rows] for a in aerosol))
                mc = window(w0, w1)
                # Built on the card: a host array here was one pinned-staged
                # upload (and one CUDA event) per SW window (_window_tables).
                adjes = cp.full(nc, np.float32(ps["adjes"]), dtype=cp.float32)
                res = _cuda_sw(self._sw_tables).rrtmg_sw_batched_device(
                    nc, ps["nlay"], ps["icld"], ps["play"][rows],
                    ps["plev"][rows], ps["tlay"][rows], ps["tlev"][rows],
                    ps["tsfc"][rows], ps["h2ovmr"][rows], ps["o3vmr"][rows],
                    ps["co2vmr"][rows], ps["ch4vmr"][rows],
                    ps["n2ovmr"][rows], ps["o2vmr"][rows], ps["asdir"][rows],
                    ps["asdif"][rows], ps["aldir"][rows], ps["aldif"][rows],
                    ps["coszen"][rows], adjes, int(ps["dyofyr"]),
                    ps["scon"][rows], ps["inflgsw"], ps["iceflgsw"],
                    ps["liqflgsw"], mc["cldfmcl"], mc["taucmcl"],
                    mc["ssacmcl"], mc["asmcmcl"], mc["fsfcmcl"],
                    mc["ciwpmcl"], mc["clwpmcl"], mc["cswpmcl"],
                    mc["reicmcl"], mc["relqmcl"], mc["resnmcl"],
                    aer_opt=(1 if chem_aerosols is not None else
                             (0 if aerosol_w is None else self.aer_opt)),
                    aerosol=aerosol_w,
                    column_chunk=chunk_sw, _stage_probe=self._stage_probe,
                    mcica_layout="column")
                del mc, aerosol_w
                if diagnostic_sink is not None:
                    idx = day_idx[a0 + w0:a0 + w1]
                    diagnostic_sink(
                        elapsed_seconds=float(state.elapsed_seconds), shape=(ny, nx),
                        indices=idx, coszen=coszen[idx],
                        swdnb=res["swdflx"][:, 0], swdnbc=res["swdflxc"][:, 0],
                        cloud_fraction=diagnostic_cloud[rows],
                        aerosol=(None if diagnostic_aerosol is None else
                                 {key: value[rows] for key, value
                                  in diagnostic_aerosol.items()}),
                        adapter=self)
                if swddif is not None:
                    swddif[selw] = res["swdflx"][:, 0] - res["swdkdir"][:, 0]
                nlay_sw = int(res["swhr"].shape[1])
                if int(res["swdflx"].shape[1]) != nlay_sw + 1:
                    raise ValueError("SW engine flux/heating-rate layer counts "
                                     "disagree")
                _adapter_kernel("rla_sw_out_grid")(
                    *blk(nc * nz),
                    (np.int32(nc), np.int32(nz), np.int32(nlay_sw),
                     res["swdflx"], res["swuflx"], res["swhr"], g_pi3d,
                     day_seconds, np.int64(ncol), selw, rthratensw, gsw))
                if swddir is not None:
                    # SWDDIR at the surface for BEP+BEM: rrtmg_swrad's swdkdir.
                    swddir[selw] = res["swdkdir"][:, 0]
                for suffix, key, level in (
                        ("upt","swuflx",-1),("uptc","swuflxc",-1),
                        ("dnt","swdflx",-1),("dntc","swdflxc",-1),
                        ("upb","swuflx",0),("upbc","swuflxc",0),
                        ("dnb","swdflx",0),("dnbc","swdflxc",0)):
                    boundary_fluxes["sw"+suffix][selw] = res[key][:,level]
                del res
            del ps, window, aerosol, diagnostic_cloud, diagnostic_aerosol
        if self.shortwave:
            # The engine's per-chunk workspace served every chunk of this
            # event; handing it back here lets the model's step-boundary
            # pool trim return the bytes to the driver, so the run's
            # resident footprint is what it was before the workspace was
            # hoisted out of the chunk loop.
            _cuda_sw(self._sw_tables).release_scratch()
            diagnostic_sink = getattr(self, "_sw_diagnostic_sink", None)
            if diagnostic_sink is not None:
                smoke_receipt = {}
                if smoke_provider is not None:
                    smoke_receipt["smoke_binding"] = smoke_provider.binding_receipt()
                diagnostic_sink(elapsed_seconds=float(state.elapsed_seconds),
                    shape=(ny, nx), coszen=coszen, adapter=self, finish=True,
                    **smoke_receipt)

        # WRF v4.6.1 module_radiation_driver.F:2870 recovers SW from the
        # rounded total after the SW routine adds its heating to LW.
        # This diagnostic round trip is also part of the wrapper contract.
        if self.shortwave:
            _adapter_kernel("rla_driver_sw_tendency")(
                *blk(nz * ncol), (np.int64(nz * ncol), rthratenlw, rthratensw))

        # ---- driver-level SWDOWN = GSW/(1-ALBEDO) (driver line 2877) --
        swdown = cp.empty(ncol, dtype=cp.float32)
        _adapter_kernel("rla_swdown")(
            *blk(ncol), (np.int64(ncol), gsw, surf["albedo"], swdown))

        self.update_count += 1
        return RadiationResult(
            boundary_fluxes={name:value.reshape(ny,nx) for name,value in boundary_fluxes.items()},
            rthratenlw=rthratenlw.reshape(nz, ny, nx),
            rthratensw=rthratensw.reshape(nz, ny, nx),
            swdown=swdown.reshape(ny, nx),
            glw=(glw.reshape(ny, nx) if self.longwave
                 else fields["glw"]),
            gsw=gsw.reshape(ny, nx),
            coszen=coszen_d.reshape(ny, nx),
            olr=(olr.reshape(ny, nx) if self.longwave
                 else None),
            swddir=(None if swddir is None else swddir.reshape(ny, nx)),
            swddif=(None if swddif is None else swddif.reshape(ny, nx)),
            cldfra=cp.ascontiguousarray(cldfra_cols.T).reshape(nz, ny, nx),
            aodtot=None if aodtot is None else aodtot.reshape(ny, nx),
            aod5503d=None if aerosol_aod is None else aerosol_aod.reshape(nz,ny,nx),
            aerosol2d=None if aerosol_diag is None else {key:cp.ascontiguousarray(aerosol_diag[:,i]).reshape(ny,nx) for i,key in enumerate(_aer2.INPUT_FIELDS)})
