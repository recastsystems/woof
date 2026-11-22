# woof/verify/cases/cloud_topped_boundary_layer.py
"""Cloud-topped convective boundary layer under an LES closure (moist P1).

The moist sibling of :mod:`woof.verify.cases.convective_boundary_layer`.
Same doubly periodic, flat em_les topology, same prescribed-flux lower
boundary (isfflx=0: constant ``tke_heat_flux`` heat, constant
``tke_drag_coefficient`` drag, and therefore **no surface moisture flux
at all**), same seeded IC perturbation on the lowest four levels -- with
Kessler warm rain (``mp_physics=1``) on top of a capped, conditionally
saturated sounding whose LCL sits inside the mixed layer.  The deck is
capped from above by a real inversion, so the cloud layer is set by the
flow rather than by the model lid.

Two entry points, the same pair the dry case exposes:

``run(outdir)`` -- the VERIFY smoke: a small short integration whose
GATES are validity-only (finite fields, bounded w and CFL, dry-mass
closure on the periodic domain).  Cloud, buoyancy-flux and engagement
statistics are computed and FILED as receipts, never gated: LES bands
are measure-then-commit against the oracle, and the moist acceptance
criteria that exist today (AC-CAP.2/3/4) were cut from the *oracle's*
own draws and are not this engine's to inherit.

``main(argv)`` -- the sized LES driver: the matched 100 m configuration
(96 x 96 x 64 at dx = 100 m, ztop 2400 m, dt 0.5 s, two simulated hours,
one profile sample per minute), writing one receipts JSON plus the
same-instrument npz.

------------------------------------------------------------------------
THE CASE DEFINITION, AND WHERE IT COMES FROM
------------------------------------------------------------------------
The sounding is the ratified P1 reference: the capped moist matched
family, 8.0 K cap over 150 m with its base at 1500 m, qv 14.0 / 5.6
g/kg stepping at the same 1500 m.  Provenance, the decider that selected
it over the qv-12 fallback and over the 20 K cap probe, and the AC-CAP
acceptance record (v1 and the v2 registered beside it) are in
``docs/internal/receipts/les/MOIST-CASE-REFERENCE-SETTINGS.md`` §6 and
``docs/internal/receipts/les/wrf-moist-capped-decider-2026-08-04.md``.

The oracle initialises from a WRF ``input_sounding`` asset.  This module
reproduces that asset *exactly* -- see :func:`sounding_text`, pinned to
sha256 ``993fedb1...`` -- and interpolates from its own rounded columns,
so both engines read the same numbers rather than two spellings of the
same intent.  The rounding matters: the asset is written ``%10.2f``, and
that is what ``ideal.exe`` reads.

**AC-CAP.1 v2 applies to every windowed statistic this case reports.**
The capped family is accepted as NON-STATIONARY on geometric criteria
only; a receipt quoting a windowed moist statistic must say the window
is not stationary and give the LWP trend beside it.  ``lwp_trend_pct_per_h``
is on the receipt for exactly that reason.

------------------------------------------------------------------------
READ THIS BEFORE COMPARING z_i: ONE NAME, TWO HEIGHTS
------------------------------------------------------------------------
Quoted from the npz contract at the head of
``tools/wrf_em_les_oracle/same_instrument_moist.py``, because the
reduction it governs is implemented here in
:func:`reduce_moist_profiles`:

    ``zi_thetav_m`` here -- and ``zi_thetav_load_m`` in the WRF receipts
    -- is the height of the MINIMUM of the total buoyancy flux.  In a
    **clear** CBL that is the inversion.  In a **cloud-topped** CBL it is
    **cloud base**, because the buoyancy flux reverses there.  This is not
    a modelling choice and not a bug; it is what the definition does, and
    it was measured on two runs of the same capped family that differ
    only in their vapour column:

        capped-DRY   anchor : zi = 1526.3 m   (the inversion, base at 1500 m)
        capped-MOIST r1     : zi = 1274.4514 m
                              cloud_base_m = 1274.4514 m   <- the same number,
                                                              to every digit

    So a moist-vs-dry z_i difference is NOT a change in boundary-layer
    depth, and comparing a cloudy WOOF-ArWen z_i against a clear WRF z_i (or
    the reverse) compares two different heights while appearing to
    compare one quantity.  Both sides must reduce it identically, and any
    receipt quoting z_i for a cloudy case must say which of the two it
    means.  The inversion height in a cloudy case is not recoverable from
    this metric; use ``cloud_top_m`` or the theta profile.

``tests/test_les_moist_instrument.py`` holds :func:`reduce_moist_profiles`
bit-identical to the oracle's ``reduce_moist`` on the banked oracle
fixtures, so the two sides cannot drift into reducing different
quantities under one name.

------------------------------------------------------------------------
THE npz CONTRACT
------------------------------------------------------------------------
:func:`write_moist_npz` emits every array the oracle reducer requires,
under the oracle's names and shapes (nz mass levels, nz+1 w levels, nt
frames):

  z_mass (nz,)  t_seconds (nt,)  wthv_res (nt,nz)  wthv_sgs (nt,nz+1)
  wqv_res (nt,nz)  wqv_sgs (nt,nz+1)  qv qc qr theta thetav (nt,nz)
  cloud_frac sat_frac n2_moist_frac (nt,nz)  lwp (nt,)

plus the optional carriers ``e_sgs`` (km_opt=2 only; an all-zero array is
read as an ABSENT carrier), ``rwp``, ``rainnc``, the ``*_novload``
theta_v variants so a convention disagreement can be measured rather than
argued, and ``z_w``.

theta_v convention, matching the oracle: ``theta*(1 + EP_1*qv - qc - qr -
qi)``, condensate loading included, with ``EP_1 = RV/RD - 1`` evaluated in
float64.  Note that this is deliberately NOT
``woof.core.constants.RVOVRD - 1``: RVOVRD is WRF's float32 divide (the
engine's EOS spelling, ``constants.py:36-45``) and lands a ULP away.  The
instrument must match the *instrument's* spelling on both sides; the
engine's internal EOS convention is a separate, engine-internal matter.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from woof.config import RunConfig
from woof.core import constants as const
from woof.core import moist_n2_mutation
from woof.core.grid import make_base_state, make_vertical_coord
from woof.verify.cases.convective_boundary_layer import (
    DRAG_COEFFICIENT, ENVIRONMENTAL_FIELDS, HEAT_FLUX, C_S, _vram_fields,
    partition_receipt_fields)

__all__ = [
    "GATES", "ENVIRONMENTAL_FIELDS", "partition_receipt_fields",
    "sounding_text", "sounding_levels", "sounding_profiles",
    "make_config", "build", "reduce_moist_profiles", "write_moist_npz",
    "run", "main",
]

#: Validity gates only.  See the module docstring: no moist LES band is
#: cut here, and the oracle's AC-CAP criteria were cut from the oracle's
#: own draws.
GATES = {
    "w_max": (None, 50.0),
    "cfl_max": (None, 1.0),
    "mass_drift_rel": (None, 1.0e-4),
}

# --------------------------------------------------------------------------
# The sounding.  Reproduced from committed constants rather than read from a
# file: the derived assets are identified by hash, not committed, and a case
# that can rebuild its own initial condition can also prove it rebuilt the
# right one.  Every number below is quoted in MOIST-CASE-REFERENCE-SETTINGS.md.
# --------------------------------------------------------------------------

#: Base dry CBL sounding (the file form of the dry case's ``_theta_profile``):
#: 300 K mixed layer to 1000 m, then +3 K/km, sampled every 25 m to 3000 m,
#: over a 1000 hPa surface with no wind.
SOUNDING_SURFACE_PRESSURE_HPA = 1000.0
SOUNDING_SURFACE_THETA_K = 300.0
SOUNDING_MIXED_LAYER_TOP_M = 1000.0
SOUNDING_LAPSE_K_PER_M = 0.003
SOUNDING_DZ_M = 25.0
SOUNDING_TOP_M = 3000.0

#: The capping inversion (MOIST-CASE-REFERENCE-SETTINGS.md §6.3).  8 K over
#: 150 m is WRF v4.6.1's own shipped moist em_les construction read off
#: ``test/em_les/input_sounding``; the 1500 m base is this family's one
#: departure from "just above the mixed-layer top", taken because isfflx=0
#: gives no moisture source, so the LCL rises monotonically and a lower cap
#: would let the deck evaporate inside the scoring window.
INVERSION_BASE_M = 1500.0
INVERSION_DEPTH_M = 150.0
INVERSION_STRENGTH_K = 8.0

#: The ratified vapour column (§1, §6.4): 14.0 g/kg in the mixed layer,
#: 5.6 g/kg above, stepping at the thermal cap rather than at the sounding's
#: own first theta departure (which is still 1025 m in the capped profile).
QV_MIXED_LAYER_G_KG = 14.0
QV_FREE_G_KG = 5.6
QV_TRANSITION_M = 1500.0

#: sha256 of the three assets this module reproduces byte for byte.  The
#: capped pair is ``capped_family_assets.sha256`` in the oracle receipts;
#: the base asset is ``tools/wrf_em_les_oracle/input_sounding.arwen_cbl``.
SOUNDING_SHA256_BASE = (
    "7e81dadac9d900c13e1f4720c18f1b421166801b050d11005b5d850c6bc25b32")
SOUNDING_SHA256_CAPPED_DRY = (
    "9ffcbb87a427b37d48ebbd1948fa753097d5ccfed9db2773c6b4d55b8014e8c8")
SOUNDING_SHA256_CAPPED_MOIST = (
    "993fedb14ab5d57684535342c7d66507437a5e792186a79c8d565a825da0fd17")

#: ``dyn_em/module_diffusion_em.F:1544`` -- ``qc_cr = 0.00001``.  The same
#: constant is the cloud-presence threshold and the saturated-BN2 arm's qc
#: predicate, on both sides, deliberately: one predicate rather than two that
#: can disagree.  woof's kernel hardcodes the same value
#: (``core/kernels/smag2d.cu:1385``).
QC_CR = 1.0e-5

#: Cloud edges are read at this cloud-fraction threshold -- the oracle
#: instrument's ``cloud_base_m`` / ``cloud_top_m`` definition.
CLOUD_EDGE_FRACTION = 0.01

#: Virtual-temperature coefficient for the INSTRUMENT (see the module
#: docstring): float64 ``RV/RD - 1``, the spelling
#: ``tools/wrf_em_les_oracle/score_moist_les.py:47`` uses.
EP_1 = const.RV / const.RD - 1.0

#: em_les reference forcing, carried over unchanged from the dry case so the
#: moist arm is a superset of the certified dry one rather than a second
#: experiment.
C_K = 0.10


def _cap_offset(z: np.ndarray) -> np.ndarray:
    """The capping-inversion theta offset, ``add_capping_inversion.py``'s
    ramp: zero at or below the base, the full strength at or above the top,
    linear between."""
    z = np.asarray(z, dtype=np.float64)
    top = INVERSION_BASE_M + INVERSION_DEPTH_M
    ramp = INVERSION_STRENGTH_K * (z - INVERSION_BASE_M) / INVERSION_DEPTH_M
    return np.where(z <= INVERSION_BASE_M, 0.0,
                    np.where(z >= top, INVERSION_STRENGTH_K, ramp))


def _format_sounding(surface, levels) -> str:
    """The ``input_sounding`` text layout both oracle tools write."""
    lines = ["%10.2f %10.2f %10.2f" % tuple(surface)]
    for row in levels:
        lines.append("%10.2f %10.2f %10.2f %10.2f %10.2f" % tuple(row))
    return "\n".join(lines) + "\n"


def _parse_sounding(text: str):
    """``(surface, levels)`` of floats, ``add_capping_inversion.parse``."""
    rows = [ln.split() for ln in text.strip().splitlines()]
    return ([float(x) for x in rows[0]],
            [[float(x) for x in r] for r in rows[1:]])


def sounding_text(*, capped: bool = True, moist: bool = True) -> str:
    """The WRF ``input_sounding`` asset, byte for byte.

    Built as a CHAIN, not as one expression, because the oracle's chain is
    lossy and the loss is essential: ``add_capping_inversion.py`` and
    ``make_moist_sounding.py`` each write ``%10.2f`` columns and each
    re-parse the previous stage's rounded text.  Rounding once at the end
    instead of at every stage lands a different file -- theta at 1025 m is
    300.075 exactly, which single-rounds to 300.07 and double-rounds to
    300.08 -- and a different file is a different case.  Reproducing the
    chain is what makes this text hash to
    :data:`SOUNDING_SHA256_CAPPED_MOIST` and makes the profile this module
    interpolates the profile ``ideal.exe`` interpolates.
    """
    # Stage 0: the base dry CBL sounding (the file form of the dry case's
    # theta profile), sampled every 25 m to 3000 m over a calm 1000 hPa
    # surface.
    n = int(round(SOUNDING_TOP_M / SOUNDING_DZ_M))
    levels = []
    for i in range(1, n + 1):
        z = SOUNDING_DZ_M * i
        theta = SOUNDING_SURFACE_THETA_K
        if z > SOUNDING_MIXED_LAYER_TOP_M:
            theta += SOUNDING_LAPSE_K_PER_M * (z - SOUNDING_MIXED_LAYER_TOP_M)
        levels.append([z, theta, 0.0, 0.0, 0.0])
    surface = [SOUNDING_SURFACE_PRESSURE_HPA, SOUNDING_SURFACE_THETA_K, 0.0]
    text = _format_sounding(surface, levels)

    # Stage 1: add_capping_inversion.py --base-m 1500 --depth-m 150
    #          --strength-k 8.0
    if capped:
        surface, levels = _parse_sounding(text)
        for r in levels:
            r[1] += float(_cap_offset(r[0]))
        text = _format_sounding(surface, levels)

    # Stage 2: make_moist_sounding.py --qv-ml 14 --qv-free 5.6
    #          --transition-m 1500
    if moist:
        surface, levels = _parse_sounding(text)
        surface[2] = QV_MIXED_LAYER_G_KG
        for r in levels:
            r[2] = (QV_MIXED_LAYER_G_KG if r[0] <= QV_TRANSITION_M + 1e-9
                    else QV_FREE_G_KG)
        text = _format_sounding(surface, levels)
    return text


def sounding_levels(*, capped: bool = True, moist: bool = True):
    """The asset's level rows as ``[z, theta, qv_g_kg, u, v]`` lists."""
    return _parse_sounding(sounding_text(capped=capped, moist=moist))[1]


def sounding_profiles(*, capped: bool = True, moist: bool = True):
    """``(z_m, theta_K, qv_kg_kg)`` parsed back out of :func:`sounding_text`.

    Parsed rather than computed so the arrays carry the asset's own rounded
    values -- the numbers WRF reads -- and not a second, unrounded
    derivation of them.
    """
    lev = np.array(sounding_levels(capped=capped, moist=moist),
                   dtype=np.float64)
    return lev[:, 0], lev[:, 1], lev[:, 2] * 1.0e-3


def _profile_interpolators(*, capped: bool = True, moist: bool = True):
    """theta(z) and qv(z) callables over the asset's own levels.

    Linear in height between sounding levels and clamped outside them --
    the operation WRF's ideal-case initializer applies to the same table.
    The clamp is exact here rather than approximate: theta and qv are both
    constant from the surface to their first departure (1000 m and 1500 m
    respectively), and the sounding's 3000 m top is far above ztop.
    """
    z, theta, qv = sounding_profiles(capped=capped, moist=moist)

    def theta_of_z(zq):
        return np.interp(np.asarray(zq, dtype=np.float64), z, theta)

    def qv_of_z(zq):
        return np.interp(np.asarray(zq, dtype=np.float64), z, qv)

    return theta_of_z, qv_of_z


def make_config(*, nx=96, ny=96, nz=64, dx=100.0, ztop=2400.0, dt=0.5,
                minutes=120.0, heat_flux=HEAT_FLUX,
                drag_coefficient=DRAG_COEFFICIENT, c_s=C_S,
                km_opt=3, c_k=C_K, tke_budget=0,
                restart_interval_s=0.0) -> RunConfig:
    """The matched moist configuration.

    Every knob is the dry matched family's value; the moist delta is
    ``mp_physics 0 -> 1`` and ``moist False -> True`` and nothing else,
    which is the same two-line delta the oracle namelist carries
    (``diff namelist.match_km3_100m namelist.moist_match_km3_100m`` is
    ``mp_physics`` and the iofields filename).  ``use_theta_m`` is 0 on the
    oracle side because woof stores dry theta (``woof/core/moist.py:29``,
    ``woof/core/microphysics.py:23``); matching it is conformance, not a
    preference, and there is no woof knob to set -- dry theta is what the
    engine has.
    """
    if km_opt not in (2, 3, 4):
        raise ValueError(
            f"km_opt must be 3 (3-D Smagorinsky, the reference arm), 2 "
            f"(1.5-order prognostic TKE, the second campaign arm) or 4 "
            f"(the 2-D Smagorinsky negative control), got {km_opt!r}")
    return RunConfig(
        nx=nx, ny=ny, nz=nz, dx=dx, dy=dx, ztop=ztop, dt=dt,
        run_seconds=minutes * 60.0,
        km_opt=km_opt, mix_isotropic=1, c_s=c_s, c_k=c_k,
        bl_pbl_physics=0, sf_sfclay_physics=0,
        # Kessler over a state that carries qv/qc/qr.  isfflx=0 means the
        # prescribed heat flux and drag with NO moisture flux at all
        # (dycore.py:1024-1030) -- the deck is fed by the initial sounding
        # and nothing else, exactly as the oracle family runs it.
        moist=True, mp_physics=1, moist_adv_opt=1,
        isfflx=0, tke_heat_flux=heat_flux,
        tke_drag_coefficient=drag_coefficient,
        time_step_sound=4,
        tke_budget=tke_budget, restart_interval_s=restart_interval_s)


def build(cfg: RunConfig, seed: int = 0):
    """Balanced moist state at rest with the seeded theta perturbation on
    the lowest four levels (README.les: "A random perturbation is imposed
    initially on the mean temperature field at the lowest four grid
    levels") -- the dry case's perturbation, unchanged, so the two arms
    differ in their vapour column and in nothing else.

    ``init_moist_balanced`` re-integrates p' and phi' under the moist EOS
    from the qv column, which is what WRF's ideal initializer does for the
    same asset.  qc and qr start at zero; the sounding is marginally
    supersaturated in the 1007-1500 m layer at t = 0 (RH_max 1.182 on the
    oracle's pre-run screen), so condensation begins at t ~ 0 rather than
    after a spin-up.  That is a property of a cloud-topped CBL initialised
    from a well-mixed sounding, and it is why "wait for the cloud to form"
    is not a valid reading of any spin-up window in this case.
    """
    from woof.core.moist import init_moist_balanced

    theta_of_z, qv_of_z = _profile_interpolators()
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, theta_of_z, p_surf=cfg.p_surf,
                           ztop=cfg.ztop)

    def thp_func(x, z):
        rng = np.random.default_rng(seed)
        pert = np.zeros((cfg.nz, cfg.ny, cfg.nx))
        pert[:4] = 0.1 * rng.standard_normal((4, cfg.ny, cfg.nx))
        return pert

    return init_moist_balanced(cfg, coord, base, qv_of_z, thp_func=thp_func)


def _array_module(a):
    """``cupy`` or ``numpy``, whichever owns ``a``.

    Resolved from the array's own type rather than by importing cupy at
    module scope, so this module stays importable (and its contract
    testable) on a machine with no device -- the discipline
    ``tests/conftest.py`` relies on.
    """
    if type(a).__module__.split(".")[0] == "cupy":
        import cupy
        return cupy
    return np


def _qvs(t_k, p_pa):
    """Saturation mixing ratio, WRF's own formula.

    ``dyn_em/module_diffusion_em.F:1626-1631``, and woof's own kernel
    spelling of it (``core/kernels/smag2d.cu:1306-1314``)::

        tc  = t - SVPT0
        es  = 1000.0 * SVP1 * EXP( SVP2*tc / (t - SVP3) )
        qvs = EP_2 * es / (p - es)

    Evaluated here in float64 on the same constants the kernel compiles
    against (``woof.core.constants``), which are WRF's.  The kernel
    evaluates it in float32 on the same expression; this diagnostic is the
    float64 reading of the same predicate, exactly as the oracle's
    instrument is the float64 reading of WRF's float32 one.
    """
    xp = _array_module(t_k)
    tc = t_k - const.SVPT0
    es = 1000.0 * const.SVP1 * xp.exp(const.SVP2 * tc / (t_k - const.SVP3))
    return const.EP2 * es / (p_pa - es)


def _sgs_flux(field, khv, fnm, fnp, phi, nz):
    """SGS flux of a mass-point field on interior w levels.

    The dry case's construction, unchanged (``convective_boundary_layer.py``
    :316-320): WRF's fnm/fnp vertical interpolation of the live ``smag_khv``
    onto the w level, times the centred gradient, slab-averaged.  Applied
    here to theta_v and to qv.  Reusing the construction rather than
    re-deriving it is the point -- a moist-vs-dry difference must not be a
    difference in how the SGS term was assembled.

    Two spellings of one geometry, worth stating because the oracle's
    counterpart looks different: ``2*G/(phi[k+1]-phi[k-1])`` is identically
    ``1/(z_mass[k]-z_mass[k-1])``, which is what
    ``score_moist_les.sgs_flux_fnm`` writes.  This one is evaluated per
    column before the slab average; the oracle's uses the slab-mean height.

    Written against ``_array_module`` rather than against cupy directly so
    the index algebra can be pinned on the host: an off-by-one here is
    invisible in the output and would corrupt the SGS half of every moist
    metric.  ``tests/test_les_moist_instrument.py`` holds it equal to a
    literal transcription of the dry case's loop.
    """
    xp = _array_module(phi)
    out = xp.zeros(nz + 1, dtype=xp.float64)
    if khv is None:
        return out
    rdz = 2.0 * const.G / (phi[2:nz + 1] - phi[0:nz - 1])
    k_w = (fnm[1:nz, None, None] * khv[1:nz]
           + fnp[1:nz, None, None] * khv[0:nz - 1])
    flux = -k_w * (field[1:nz] - field[0:nz - 1]) * rdz
    out[1:nz] = flux.mean(axis=(1, 2))
    return out


def _column_mass(state) -> float:
    """Domain-mean total dry column mass (periodic: no lateral flux, so this
    is the exact-conservation observable).  The dry case's observable."""
    import cupy as cp
    return float(cp.asnumpy(
        (state.mub2d + state.mup).astype(cp.float64).mean()))


def _layer_dry_mass(state):
    """Layer dry mass per unit area, ``(c1h*mu + c2h)*(-dnw)/g`` kg/m2.

    The eta-measure column weight ``moist_bubble._water_mass`` uses.  With
    ``hybrid_opt=0`` (this case) ``c1h = 1, c2h = 0``, so it is WRF's
    ``-(MU+MUB)*DNW/g`` -- the weight ``score_moist_les.py:180-185`` builds
    the oracle's liquid-water path from.
    """
    import cupy as cp
    mu = (state.mub2d + state.mup)[None].astype(cp.float64)
    chm = (state.c1h[:, None, None].astype(cp.float64) * mu
           + state.c2h[:, None, None].astype(cp.float64))
    return chm * (-state.dnw[:, None, None].astype(cp.float64)) / const.G


def _moist_slab_profiles(state, cfg) -> dict:
    """Every moist profile for one frame, in the npz contract's units.

    Definitions are the oracle instrument's, applied to this engine's own
    fields: theta_v with condensate loading, the resolved fluxes from the
    slab-mean-removed w and field perturbations, the SGS halves from the
    live ``smag_khv``, and the engagement fractions under WRF's own
    predicate -- ``qv >= qvs .OR. qc >= 1e-5``, which is also the predicate
    this engine's own moist-N2 kernel evaluates
    (``core/kernels/smag2d.cu:1383-1386``), so the diagnostic reports the
    branch the run actually took.
    """
    import cupy as cp

    nz = cfg.nz
    th = state.total_theta().astype(cp.float64)
    qv = state.qv.astype(cp.float64)
    qc = state.qc.astype(cp.float64)
    qr = state.qr.astype(cp.float64)
    p_full = state.p.astype(cp.float64)
    t_k = th * (p_full / const.P0) ** const.RCP

    # theta_v, both conventions, so a convention disagreement between the
    # engines can be measured rather than argued.  qi is absent under
    # Kessler; the term is written out to keep the formula the oracle's.
    thv_load = th * (1.0 + EP_1 * qv - qc - qr)
    thv_nov = th * (1.0 + EP_1 * qv)

    def slab(a):
        return a.mean(axis=(1, 2))

    wm = 0.5 * (state.w[:-1] + state.w[1:]).astype(cp.float64)
    wp = wm - slab(wm)[:, None, None]

    out = {}
    for tag, fld in (("load", thv_load), ("novload", thv_nov)):
        fp = fld - slab(fld)[:, None, None]
        out["wthv_res_" + tag] = cp.asnumpy(slab(wp * fp))
    out["wqv_res"] = cp.asnumpy(slab(wp * (qv - slab(qv)[:, None, None])))

    khv = state.existing_scratch("smag_khv")
    if khv is not None:
        khv = khv.astype(cp.float64)
    phb = state.phb
    phi = ((phb if phb.ndim == 3 else phb[:, None, None])
           + state.php).astype(cp.float64)
    fnm = state.fnm.astype(cp.float64)
    fnp = state.fnp.astype(cp.float64)
    out["wthv_sgs_load"] = cp.asnumpy(
        _sgs_flux(thv_load, khv, fnm, fnp, phi, nz))
    out["wthv_sgs_novload"] = cp.asnumpy(
        _sgs_flux(thv_nov, khv, fnm, fnp, phi, nz))
    out["wqv_sgs"] = cp.asnumpy(_sgs_flux(qv, khv, fnm, fnp, phi, nz))

    out["qv"] = cp.asnumpy(slab(qv))
    out["qc"] = cp.asnumpy(slab(qc))
    out["qr"] = cp.asnumpy(slab(qr))
    out["theta"] = cp.asnumpy(slab(th))
    out["thetav"] = cp.asnumpy(slab(thv_load))

    qvs = _qvs(t_k, p_full)
    engaged = (qv >= qvs) | (qc >= QC_CR)
    out["n2_moist_frac"] = cp.asnumpy(engaged.mean(axis=(1, 2)))
    out["sat_frac"] = cp.asnumpy((qv >= qvs).mean(axis=(1, 2)))
    out["cloud_frac"] = cp.asnumpy((qc >= QC_CR).mean(axis=(1, 2)))

    dm = _layer_dry_mass(state)
    out["lwp"] = float((qc * dm).sum(axis=0).mean())
    out["rwp"] = float((qr * dm).sum(axis=0).mean())
    out["qc_max"] = float(qc.max())
    out["qr_max"] = float(qr.max())

    e_sgs = np.zeros(nz)
    if getattr(state, "tke", None) is not None:
        e_sgs = cp.asnumpy(state.tke.astype(cp.float64).mean(axis=(1, 2)))
    out["e_sgs"] = e_sgs

    rain = state.existing_scratch("mp_rainnc")
    out["rainnc"] = 0.0 if rain is None else float(
        rain.astype(cp.float64).mean())
    return out


def reduce_moist_profiles(z_mass, t_seconds, wthv_res_t, wthv_sgs_t,
                          wqv_res_t, wqv_sgs_t, qv_t, qc_t, qr_t, cloud_t,
                          sat_t, n2_t, lwp_t, window_s) -> dict:
    """One reduction, applied identically to either model's arrays.

    Transcribed from ``tools/wrf_em_les_oracle/same_instrument_moist.py``
    ``reduce_moist`` and held bit-identical to it by
    ``tests/test_les_moist_instrument.py``, which runs both on the banked
    oracle fixtures.  It is transcribed rather than imported because
    ``tools/`` is not an importable package and, more importantly, because
    the oracle side must stay able to disagree with the engine it scores --
    a woof -> oracle import would couple them.  The test is what makes the
    transcription safe.
    """
    t = np.asarray(t_seconds, dtype=np.float64)
    sel = t >= (t.max() - window_s - 1e-9)
    n = int(sel.sum())

    def w(a):
        return np.asarray(a, dtype=np.float64)[sel].mean(axis=0)

    wthv_res = w(wthv_res_t)
    wthv_sgs = w(wthv_sgs_t)
    wqv_res = w(wqv_res_t)
    wqv_sgs = w(wqv_sgs_t)
    qv = w(qv_t)
    qc = w(qc_t)
    qr = w(qr_t)
    cloud = w(cloud_t)
    sat = w(sat_t)
    n2 = w(n2_t)
    lwp = float(np.asarray(lwp_t, dtype=np.float64)[sel].mean())

    total_thv = wthv_res + 0.5 * (wthv_sgs[:-1] + wthv_sgs[1:])
    total_qv = wqv_res + 0.5 * (wqv_sgs[:-1] + wqv_sgs[1:])

    # ONE NAME, TWO HEIGHTS.  This is the height of the total-buoyancy-flux
    # MINIMUM.  In a clear CBL it is the inversion (the capped-dry anchor
    # reads 1526.3 m against a cap base at 1500 m); in a cloud-topped CBL it
    # is CLOUD BASE, because the buoyancy flux reverses there (the
    # capped-moist draw reads 1274.4514 m, which is cloud_base_m to every
    # digit).  Both engines must reduce it with this line and no other, or
    # the comparison silently compares two different heights under one name.
    # See the module docstring for the full quotation and the receipts it
    # was measured on.
    zi = float(z_mass[int(np.argmin(total_thv))])

    def edges(profile, thresh):
        hit = np.where(profile >= thresh)[0]
        if not len(hit):
            return None, None
        return float(z_mass[hit[0]]), float(z_mass[hit[-1]])

    cb, ct = edges(cloud, CLOUD_EDGE_FRACTION)

    # Resolved share of the buoyancy flux over the mixed layer, the moist
    # counterpart of the dry lane's resolved-fraction measure.  Index band
    # 0.1-0.7 z_i in HEIGHT, not index, because the two sides may not share
    # a level count.
    band = (z_mass >= 0.1 * zi) & (z_mass <= 0.7 * zi)
    sgs_m = 0.5 * (wthv_sgs[:-1] + wthv_sgs[1:])
    num = float(np.abs(wthv_res[band]).mean())
    den = num + float(np.abs(sgs_m[band]).mean())
    res_frac_thv = num / den if den > 0 else float("nan")

    sgs_q = 0.5 * (wqv_sgs[:-1] + wqv_sgs[1:])
    numq = float(np.abs(wqv_res[band]).mean())
    denq = numq + float(np.abs(sgs_q[band]).mean())
    res_frac_qv = numq / denq if denq > 0 else float("nan")

    return dict(
        n_samples=n,
        zi_thetav_m=zi,
        wthv_res_max=float(wthv_res.max()),
        wthv_total_min=float(total_thv.min()),
        wqv_res_max=float(wqv_res.max()),
        wqv_total_max=float(np.abs(total_qv).max()),
        resolved_fraction_wthv=res_frac_thv,
        resolved_fraction_wqv=res_frac_qv,
        qv_surface=float(qv[0]),
        qc_profile_max=float(qc.max()),
        qr_profile_max=float(qr.max()),
        cloud_fraction_max=float(cloud.max()),
        cloud_base_m=cb, cloud_top_m=ct,
        sat_fraction_max=float(sat.max()),
        n2_moist_fraction_max=float(n2.max()),
        n2_engaged_somewhere=bool(float(n2.max()) > 0.0),
        n2_engaged_everywhere=bool(float(n2.min()) >= 1.0),
        lwp_kg_m2=lwp,
        _profiles=dict(z=z_mass, thv_res=wthv_res, thv_sgs=wthv_sgs,
                       thv_total=total_thv, qv_res=wqv_res, qv_sgs=wqv_sgs,
                       qc=qc, cloud=cloud, n2=n2),
    )


#: The reducer's required array names, verbatim from the npz contract at the
#: head of ``same_instrument_moist.py``.  A writer that drops one of these
#: makes the comparison impossible, and filling it with zeros would be worse
#: than failing -- so :func:`write_moist_npz` checks.
NPZ_REQUIRED = ("z_mass", "t_seconds", "wthv_res", "wthv_sgs", "wqv_res",
                "wqv_sgs", "qv", "qc", "qr", "cloud_frac", "sat_frac",
                "n2_moist_frac", "lwp")


def moist_npz_arrays(samples, z_mass, z_w, *, km_opt: int) -> dict:
    """Assemble the same-instrument npz arrays from a list of frames.

    Split out from the writer so the contract is testable without a device.
    """
    def stack(key):
        return np.stack([np.asarray(s[key], dtype=np.float64)
                         for s in samples])

    def series(key):
        return np.array([float(s[key]) for s in samples], dtype=np.float64)

    arrays = {
        "z_mass": np.asarray(z_mass, dtype=np.float64),
        "z_w": np.asarray(z_w, dtype=np.float64),
        "t_seconds": series("t_seconds"),
        "wthv_res": stack("wthv_res_load"),
        "wthv_sgs": stack("wthv_sgs_load"),
        "wthv_res_novload": stack("wthv_res_novload"),
        "wthv_sgs_novload": stack("wthv_sgs_novload"),
        "wqv_res": stack("wqv_res"),
        "wqv_sgs": stack("wqv_sgs"),
        "qv": stack("qv"), "qc": stack("qc"), "qr": stack("qr"),
        "theta": stack("theta"), "thetav": stack("thetav"),
        "cloud_frac": stack("cloud_frac"), "sat_frac": stack("sat_frac"),
        "n2_moist_frac": stack("n2_moist_frac"),
        "lwp": series("lwp"), "rwp": series("rwp"),
        "rainnc": series("rainnc"),
    }
    # The prognostic SGS TKE carrier exists only under km_opt=2.  The
    # reducer reads an all-zero e_sgs as an ABSENT carrier rather than a
    # zero one; writing the zeros under km_opt=3 would be writing a
    # carrier that does not exist, so the key is simply omitted there.
    if km_opt == 2:
        arrays["e_sgs"] = stack("e_sgs")
    missing = [k for k in NPZ_REQUIRED if k not in arrays]
    if missing:
        raise RuntimeError(
            "the same-instrument npz contract requires %s and this writer "
            "did not produce them; the contract is documented at the head "
            "of tools/wrf_em_les_oracle/same_instrument_moist.py" % missing)
    return arrays


def write_moist_npz(path, samples, z_mass, z_w, *, km_opt: int) -> dict:
    """Write ``<prefix>_moist_profiles.npz`` in the oracle's own layout."""
    arrays = moist_npz_arrays(samples, z_mass, z_w, km_opt=km_opt)
    np.savez_compressed(path, **arrays)
    return arrays


def _lwp_trend_pct_per_h(t_seconds, lwp, window_s: float) -> float | None:
    """Least-squares LWP trend over the final ``window_s``, in % per hour.

    AC-CAP.1 v2 accepts this family as NON-STATIONARY, and requires every
    receipt quoting a windowed moist statistic to say so and give this
    number beside it (MOIST-CASE-REFERENCE-SETTINGS.md §6.11).  It is a
    reported quantity, not a gate, on both sides.
    """
    t = np.asarray(t_seconds, dtype=np.float64)
    y = np.asarray(lwp, dtype=np.float64)
    sel = t >= (t.max() - window_s - 1e-9)
    if int(sel.sum()) < 3:
        return None
    mean = float(y[sel].mean())
    if mean <= 0.0:
        return None
    slope = float(np.polyfit(t[sel], y[sel], 1)[0])
    return 100.0 * slope * 3600.0 / mean


def _integrate(cfg: RunConfig, seed: int, sample_every_s: float,
               window_min: float, progress=None) -> dict:
    import cupy as cp

    from woof.core.dycore import run_steps, stability_report

    state = build(cfg, seed=seed)
    z = state.height_half()
    z_mass = np.asarray(cp.asnumpy(z) if hasattr(z, "get") else z, float)
    if z_mass.ndim == 3:
        z_mass = z_mass.mean(axis=(1, 2))
    phb = state.phb
    phi0 = ((phb if phb.ndim == 3 else phb[:, None, None])
            + state.php).astype(cp.float64)
    z_w = cp.asnumpy(phi0.mean(axis=(1, 2)) / const.G).astype(np.float64)

    n_total = int(round(cfg.run_seconds / cfg.dt))
    n_chunk = max(int(round(sample_every_s / cfg.dt)), 1)
    mass0 = _column_mass(state)
    samples = []
    cfl_max = 0.0
    w_max = 0.0
    nan = False
    wall0 = time.monotonic()
    done = 0
    while done < n_total:
        n = min(n_chunk, n_total - done)
        run_steps(state, cfg, n)
        done += n
        rep = stability_report(state, cfg)
        nan = nan or bool(rep["nan"])
        if rep["cfl"] is not None:
            cfl_max = max(cfl_max, float(rep["cfl"]))
        w_max = max(w_max, float(rep["w_max"]))
        prof = _moist_slab_profiles(state, cfg)
        prof["t_seconds"] = done * cfg.dt
        prof["w_max"] = float(rep["w_max"])
        samples.append(prof)
        if progress is not None:
            progress(done, n_total, rep)
        if nan:
            break
    wall = time.monotonic() - wall0
    mass1 = _column_mass(state)

    t_seconds = np.array([s["t_seconds"] for s in samples], dtype=np.float64)
    window_s = window_min * 60.0
    R = reduce_moist_profiles(
        z_mass, t_seconds,
        np.stack([s["wthv_res_load"] for s in samples]),
        np.stack([s["wthv_sgs_load"] for s in samples]),
        np.stack([s["wqv_res"] for s in samples]),
        np.stack([s["wqv_sgs"] for s in samples]),
        np.stack([s["qv"] for s in samples]),
        np.stack([s["qc"] for s in samples]),
        np.stack([s["qr"] for s in samples]),
        np.stack([s["cloud_frac"] for s in samples]),
        np.stack([s["sat_frac"] for s in samples]),
        np.stack([s["n2_moist_frac"] for s in samples]),
        np.array([s["lwp"] for s in samples]),
        window_s)

    qc_run = max(s["qc_max"] for s in samples)
    first_cloud = None
    for s in samples:
        if s["qc_max"] >= QC_CR:
            first_cloud = float(s["t_seconds"])
            break

    vram = _vram_fields(
        pool_used_bytes=cp.get_default_memory_pool().used_bytes,
        device_free_total=cp.cuda.runtime.memGetInfo)

    metrics = {
        "nan": nan,
        "w_max": w_max,
        "cfl_max": cfl_max,
        "mass_drift_rel": abs(mass1 - mass0) / abs(mass0),
        # --- receipts (never gated here; see module docstring) ---
        "window_minutes": window_min,
        "window_frames": R["n_samples"],
        # AC-CAP.1 v2: this family is accepted as NON-STATIONARY.  Every
        # windowed statistic below is a mean over a window in which the
        # cloud deck is still evolving; the LWP trend is the size of that
        # evolution and travels with them.
        "window_is_stationary": False,
        "lwp_trend_pct_per_h": _lwp_trend_pct_per_h(
            t_seconds, [s["lwp"] for s in samples], 3600.0),
        # ONE NAME, TWO HEIGHTS -- this is cloud base whenever a deck
        # exists, and the inversion only when none does.  The companion
        # field says which reading applies to this run rather than leaving
        # a reader to infer it.
        "zi_thetav_load_m": R["zi_thetav_m"],
        "zi_meaning": ("cloud base (cloud-topped CBL: the buoyancy flux "
                       "reverses at cloud base)"
                       if R["cloud_base_m"] is not None
                       else "inversion height (clear CBL: no cloud layer)"),
        "cloud_base_m": R["cloud_base_m"],
        "cloud_top_m": R["cloud_top_m"],
        "cloud_fraction_max": R["cloud_fraction_max"],
        "sat_fraction_max": R["sat_fraction_max"],
        "n2_moist_fraction_max": R["n2_moist_fraction_max"],
        "n2_engaged_somewhere": R["n2_engaged_somewhere"],
        "n2_engaged_everywhere": R["n2_engaged_everywhere"],
        "n2_predicate": ("qv >= qvs .OR. qc >= 1e-5  "
                         "(dyn_em/module_diffusion_em.F:1636; "
                         "woof core/kernels/smag2d.cu:1383-1386)"),
        "wthv_res_max": R["wthv_res_max"],
        "wthv_total_min": R["wthv_total_min"],
        "wthv_res_max_over_qs": (R["wthv_res_max"] / cfg.tke_heat_flux
                                 if cfg.tke_heat_flux else None),
        "wqv_res_max": R["wqv_res_max"],
        "wqv_total_max": R["wqv_total_max"],
        "resolved_fraction_wthv": R["resolved_fraction_wthv"],
        "resolved_fraction_wqv": R["resolved_fraction_wqv"],
        "qv_surface": R["qv_surface"],
        "qc_profile_max": R["qc_profile_max"],
        "qr_profile_max": R["qr_profile_max"],
        "qc_max_pointwise_run": qc_run,
        "qr_max_pointwise_run": max(s["qr_max"] for s in samples),
        "first_cloud_seconds": first_cloud,
        "lwp_kg_m2": R["lwp_kg_m2"],
        "rwp_kg_m2": float(np.asarray(
            [s["rwp"] for s in samples])[t_seconds >= (
                t_seconds.max() - window_s - 1e-9)].mean()),
        "rainnc_mm_end": float(samples[-1]["rainnc"]),
        # Which BUILD produced this receipt.  A plain string on both arms,
        # never null on one of them: a null would read as "not recorded"
        # and would let a mutation-control receipt merge into a scored draw
        # set unnoticed.  ``draw_spread`` carries it in the configuration
        # identity, so mixing the two arms is REPORTED rather than averaged.
        "mutation_control": moist_n2_mutation.receipt_tag(),
        "wall_seconds": wall,
        "steps": done,
        "steps_per_second": done / wall if wall > 0 else None,
        **vram,
    }
    return {"metrics": metrics, "samples": samples, "z_mass": z_mass,
            "z_w": z_w, "state": state}


def _write_outputs(result, cfg, outdir: Path, tag: str) -> None:
    samples = result["samples"]
    write_moist_npz(outdir / f"{tag}_moist_profiles.npz", samples,
                    result["z_mass"], result["z_w"], km_opt=cfg.km_opt)
    receipt = dict(result["metrics"])
    receipt["sounding"] = {
        "asset": "input_sounding.arwen_cbl_capped_moist",
        "sha256": SOUNDING_SHA256_CAPPED_MOIST,
        "inversion_base_m": INVERSION_BASE_M,
        "inversion_depth_m": INVERSION_DEPTH_M,
        "inversion_strength_k": INVERSION_STRENGTH_K,
        "qv_mixed_layer_g_kg": QV_MIXED_LAYER_G_KG,
        "qv_free_g_kg": QV_FREE_G_KG,
        "qv_transition_m": QV_TRANSITION_M,
    }
    receipt["config"] = {
        "nx": cfg.nx, "ny": cfg.ny, "nz": cfg.nz, "dx": cfg.dx,
        "ztop": cfg.ztop, "dt": cfg.dt, "run_seconds": cfg.run_seconds,
        "km_opt": cfg.km_opt, "mix_isotropic": cfg.mix_isotropic,
        "sf_sfclay_physics": cfg.sf_sfclay_physics, "moist": cfg.moist,
        "mp_physics": cfg.mp_physics, "moist_adv_opt": cfg.moist_adv_opt,
        "c_s": cfg.c_s, "c_k": cfg.c_k, "isfflx": cfg.isfflx,
        "tke_heat_flux": cfg.tke_heat_flux,
        "tke_drag_coefficient": cfg.tke_drag_coefficient,
        "tke_upper_bound": cfg.tke_upper_bound,
        "bl_pbl_physics": cfg.bl_pbl_physics,
        "periodic": not (cfg.specified or cfg.open_x or cfg.open_y
                         or cfg.nested),
    }
    (outdir / f"{tag}_receipt.json").write_text(
        json.dumps(receipt, indent=1, default=float) + "\n",
        encoding="utf-8")


def run(outdir: Path | None = None) -> dict:
    """VERIFY smoke: small grid, 15 simulated minutes, validity gates.

    ztop and nz keep the inversion and the cloud layer inside the domain --
    a smoke that truncates the cap is not a smoke of this case.
    """
    cfg = make_config(nx=48, ny=48, nz=48, dx=100.0, ztop=2400.0,
                      dt=0.5, minutes=15.0)
    result = _integrate(cfg, seed=0, sample_every_s=60.0, window_min=5.0)
    metrics = dict(result["metrics"])
    metrics["nan"] = bool(metrics["nan"])
    if outdir is not None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        _write_outputs(result, cfg, outdir, tag="ctbl_smoke")
    metrics.pop("state", None)
    return metrics


def main(argv=None) -> int:
    """Sized moist LES driver (see module docstring)."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Cloud-topped convective boundary layer, moist LES")
    parser.add_argument("--nx", type=int, default=96)
    parser.add_argument("--ny", type=int, default=96)
    parser.add_argument("--nz", type=int, default=64)
    parser.add_argument("--dx", type=float, default=100.0)
    parser.add_argument("--ztop", type=float, default=2400.0)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--minutes", type=float, default=120.0)
    parser.add_argument("--heat-flux", type=float, default=HEAT_FLUX)
    parser.add_argument("--drag", type=float, default=DRAG_COEFFICIENT)
    parser.add_argument("--cs", type=float, default=C_S)
    parser.add_argument("--ck", type=float, default=C_K,
                        help="km_opt=2 c_k (em_les reference 0.10)")
    parser.add_argument("--km-opt", type=int, default=3, choices=(2, 3, 4),
                        help="LES closure: 3 = 3-D Smagorinsky (the "
                             "reference arm), 2 = 1.5-order prognostic TKE, "
                             "4 = the 2-D Smagorinsky negative control")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-every", type=float, default=60.0,
                        help="profile sampling interval, seconds -- 60 s "
                             "matches the oracle's history_interval_m = 1")
    parser.add_argument("--window-min", type=float, default=30.0,
                        help="trailing averaging window for the receipt's "
                             "reduction, minutes (the oracle's default)")
    parser.add_argument("--print-sounding", action="store_true",
                        help="write the input_sounding asset this case "
                             "reproduces to stdout and exit")
    parser.add_argument("--mutation-control", default=None,
                        choices=("moist-n2-forced-dry",),
                        help="INSTRUMENT QUALIFICATION, not a physics "
                             "option: run this case on a scratch build with "
                             "the saturated moist-N2 branch forced off. It "
                             "must MOVE the cloud-layer metrics; if it does "
                             "not, the instrument is rejected, not the "
                             "engine passed (LES completion spec 3.3). The "
                             "receipt is stamped so a control-arm run can "
                             "never be read as a scored one")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--tag", default="ctbl")
    args = parser.parse_args(argv)

    if args.print_sounding:
        import sys
        # Written as BYTES, not through the text layer: on Windows the text
        # layer translates "\n" to "\r\n" and the piped digest then does not
        # match the asset, which is the one thing this switch exists to let
        # a reader check.
        sys.stdout.buffer.write(sounding_text().encode("ascii"))
        sys.stdout.buffer.flush()
        return 0
    if args.out is None:
        parser.error("--out is required unless --print-sounding is given")

    cfg = make_config(nx=args.nx, ny=args.ny, nz=args.nz, dx=args.dx,
                      ztop=args.ztop, dt=args.dt, minutes=args.minutes,
                      heat_flux=args.heat_flux, drag_coefficient=args.drag,
                      c_s=args.cs, km_opt=args.km_opt, c_k=args.ck)

    def progress(done, total, rep):
        if done % max(total // 40, 1) < max(
                int(round(args.sample_every / cfg.dt)), 1):
            print(f"  t={done * cfg.dt:8.1f}s  w_max={rep['w_max']:6.2f}"
                  f"  cfl={rep['cfl'] if rep['cfl'] is None else round(rep['cfl'], 3)}",
                  flush=True)

    if args.mutation_control is None:
        result = _integrate(cfg, seed=args.seed,
                            sample_every_s=args.sample_every,
                            window_min=args.window_min, progress=progress)
    else:
        print("MUTATION CONTROL ARM: the saturated moist-N2 branch is "
              "FORCED OFF. This is instrument qualification. Its output is "
              "not a scored run and its receipt says so.", flush=True)
        with moist_n2_mutation.forced_dry_branch():
            result = _integrate(cfg, seed=args.seed,
                                sample_every_s=args.sample_every,
                                window_min=args.window_min,
                                progress=progress)
    args.out.mkdir(parents=True, exist_ok=True)
    _write_outputs(result, cfg, args.out, tag=args.tag)
    m = result["metrics"]
    print(json.dumps({k: v for k, v in m.items()}, indent=1, default=float))
    failed = bool(m["nan"])
    for key, (lo, hi) in GATES.items():
        value = m.get(key)
        if value is None:
            continue
        if (lo is not None and value < lo) or \
                (hi is not None and value > hi):
            print(f"GATE FAIL: {key} = {value} outside ({lo}, {hi})")
            failed = True
    print("VALIDITY", "FAIL" if failed else "PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
