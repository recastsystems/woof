# ======================================================================
# THIRD-PARTY NOTICE.  Parts of this file are hand transcriptions of
# third-party work.  ArWen distributes the file under the Apache License
# 2.0; the notices below belong to the transcribed parts and are kept here
# because their own licences require it.  Full texts are in the repository
# NOTICE and in the licenses/ directory.
#
#   RTE+RRTMGP, transcribed from earth-system-radiation/rte-rrtmgp at the
#   commit this file's own header cites.  BSD 3-Clause:
#
#       Copyright (c) 2015-2025, Atmospheric and Environmental Research,
#         Regents of the University of Colorado,
#         Trustees of Columbia University in the City of New York.
#
#   Clause 1 requires source redistributions to retain that notice, the
#   conditions and the disclaimer; the full text is in
#   licenses/LICENSE-RTE-RRTMGP-BSD-3-Clause.txt.
# ======================================================================
"""GPU RTE+RRTMGP longwave/shortwave radiation.

The coefficient loader in this first section mirrors the transformations in
RTE+RRTMGP ``mo_optics_utils_rrtmgp.F90`` and
``mo_gas_optics_rrtmgp.F90:init_abs_coeffs``.  NetCDF values remain float64 on
the host; :meth:`GasTables.to_device` and :meth:`CloudTables.to_device` create
and cache packed FP32 device copies.

Reference: earth-system-radiation/rte-rrtmgp commit
fa107a16120051c4124305c6b3d4c87059119f58; coefficient data are rrtmgp-data
v1.9.  See ``woof_data/data/rrtmgp/PROVENANCE.md`` in the ``recast-woof-data``
companion distribution, which carries these tables since 2.5.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from functools import lru_cache
from woof.core.device_cache import cuda_cache, cached_ready
from types import SimpleNamespace
import os
import threading
from pathlib import Path
from typing import Mapping, NamedTuple

import numpy as np

from woof.netcdf_bridge import open_dataset as _open_netcdf


def _open_table(path, *, prefetch: bool = True):
    """One RRTMGP coefficient or input file, read through the Rust route.

    THE BREAKAGE (D-10, 2.8.8 acceptance): the memory estimate packs the
    gas and cloud tables (:func:`woof.core.preflight.k_distribution_bytes`)
    while a run is being PREPARED, and these files were read with
    netCDF4.  A preparation hosted in the site's process -- which the
    command line's ``PYTHON_GIL=0`` re-exec cannot reach -- switched its
    free-threaded interpreter lock back on there, and NetCDF reads belong
    in Rust (the Python boundary).  ``rw_netcdf`` decodes every value as
    float64 (characters as bytes), which the packing below then casts
    exactly as it cast netCDF4's arrays; the packed tables are identical.

    ``prefetch`` decodes every variable in one ``rw_netcdf`` run instead
    of one run per variable read.  No in-process HDF5 is entered, so no
    :func:`woof.io.netcdf_serialization.netcdf4_session` is needed.
    """
    dataset = _open_netcdf(path)
    if prefetch:
        dataset.prefetch(list(dataset.variables))
    return dataset

from woof import data_assets
from woof.config import DEFAULT_COLUMN_CHUNK
from woof.wrf_exact import ENABLED as WRF_EXACT
from woof.physics_compat import (
    WRF_RRTMG_TO_RTE_RRTMGP,
    WRF_RRTMG_TO_RTE_RRTMGP_V1,
)
from woof.core.mynn_radiation import (
    merge_mynn_bl_clouds,
    mynn_bl_cloud_active,
    mynn_bl_cloud_supplied,
    wrf_itimestep,
)

#: The k-distribution and cloud-optics NetCDFs and the trace-gas
#: climatology derived from the RFMIP inputs.  Same files, same
#: relative layout, same bytes as when they sat at ``woof/data/rrtmgp``;
#: they ship in the ``recast-woof-data`` companion distribution since 2.5.0
#: because the wheel carrying them measured 103.62 MiB against PyPI's
#: 100 MiB cap.  ``woof.data_assets`` is the only thing that knows that,
#: and it refuses BY NAME (with the pip line) if the companion is absent
#: or a different version.
DATA_DIR = data_assets.rrtmgp_data_dir()
DTYPE = np.float32


#: Every NetCDF member this module opens, declared once.
#:
#: The registry publishes an asset requirement for this set
#: (``gpuwm-rte-rrtmgp-tables-v1``), and plan review RESOLVES asset
#: requirements -- but a requirement that names no file resolves
#: vacuously, which is what that row did: it declared a root and no
#: members, so plan review reported it satisfied without a single
#: ``stat()`` and an install short of a table still refused at load,
#: exactly the shape audit R-045 was written to close.  The files are
#: named here, in the module that opens them, and the builder imports
#: this tuple; :func:`_table` refuses a filename that is not in it, so a
#: member cannot be added to the load path without joining the set plan
#: review checks.
#:
#: WHICH DOOR ASKS, stated so the claim cannot grow.  ``woof run`` does
#: not call :func:`woof.physics_registry.validate_physics_plan` -- its
#: one caller is ``woof/source_cli.py``'s ``--validate-physics-plan`` --
#: so on a run the missing member is named by :func:`_table` here, through
#: ``data_assets.require_companion_member``, when radiation loads its
#: tables at model construction.  That is before step 0 and it names the
#: file and the pip line, which is what the gate law asks of it; it is not
#: plan review, and a report that says an install short of a table refuses
#: "at plan review" on a bare run is over-claiming.
RRTMGP_TABLE_FILES: tuple[str, ...] = (
    "rrtmgp-gas-lw-g256.nc",
    "rrtmgp-gas-sw-g224.nc",
    "rrtmgp-clouds-lw-bnd.nc",
    "rrtmgp-clouds-sw-bnd.nc",
    "rrtmgp-trace-gas-climatology.json",
)


def _table(filename: str):
    """One member, resolved through the refusal that can name it.

    ``Dataset(DATA_DIR / filename)`` on an absent member is a bare
    ``FileNotFoundError`` fifteen frames deep in whatever asked for
    radiation -- measured at the front door when the first public-tree
    companion wheel shipped without its ``rrtmgp/*.nc``.  The resolver
    turns the same absence into ``data_assets.CompanionDataMissing``,
    which ``woof.cli.main`` prints as the sentence it is.
    """

    if filename not in RRTMGP_TABLE_FILES:
        raise ValueError(
            f"rrtmgp/{filename} is not in RRTMGP_TABLE_FILES, which is the "
            "set plan review resolves before a run starts; add it there "
            "(and regenerate the registry) rather than opening a member "
            "no plan review checks")
    from woof.tables import table_path
    family = "rrtmgp-sw" if "-sw-" in filename else "rrtmgp-lw"
    return table_path(family, DATA_DIR / filename)

# ---------------------------------------------------------------------------
# Radiation-facing effective radii are contracted in MICRONS across every
# microphysics writer (state.py background fills, thompson_contract.py,
# the WSM6/Thompson/Morrison kernels).  These physical-plausibility bands
# gate that contract at the radiation boundary: clouds with re_liq of
# 0.0000025 um or 2,500,000 um do not exist, so any writer that emits a
# radius in the wrong metric unit must fail here instead of silently
# radiating at a clip floor.  Every writer background-fills clear cells
# with 2.49-25 um values, and every background times or divided by any
# metric-prefix factor (>= 1000) leaves its band, so a metre, millimetre,
# or nanometre emission trips the gate on the very first radiation call
# regardless of scheme or cloud state.  Bounds admit each scheme's clamp
# extremes: WSM6/Thompson clamps 2.49-50/4.99-125/9.99-999 um; Morrison
# lambda bounds give cloud (pgam+3)/(2(pgam+1)) = 0.59-0.83 um through
# 50 um, ice 1.5 through 525 um (1.5e6*(2*MDCS+100e-6)), snow 15 through
# 3000 um (1.5e6*2000e-6); backgrounds 2.49-25 um.
EFFECTIVE_RADIUS_PLAUSIBLE_UM = {
    "effc": (0.5, 100.0),
    "effi": (1.0, 600.0),
    "effs": (1.0, 5000.0),
}

#: Milbrandt-Yau's band row: the generic bands by value.  See
#: :func:`effective_radius_bands` for why it gates nothing on the
#: production path (the radii are derived in-adapter and clipped there).
MILBRANDT2_EFFECTIVE_RADIUS_BANDS_UM = dict(EFFECTIVE_RADIUS_PLAUSIBLE_UM)

# Milbrandt-Yau's own effective radii, the constants of WRF's commented
# block (module_mp_milbrandt2mom.F:3351-3378).  ``r_eff = M_D(3)/(2 M_D(2))``
# over MY2005a eqn (2), the scheme's generalised gamma
# ``N(D) = N0 D^((alpha+1) mu - 1) exp(-(lambda D)^mu)``:
#   cloud, alpha_c = 1, mu_c = 3:  r_eff = Gamma(3)/(2 Gamma(8/3)) / lambda
#                                        = 0.664639 / lambda   (:3362)
#   ice,   alpha_i = 0, mu_i = 1:  r_eff = Gamma(4)/(2 Gamma(3)) / lambda
#                                        = 1.5 / lambda        (:3372)
# and 1/lambda is the scheme's own ``iLAMDA_x(DE, Q, 1/N, icex, 1/dm)``
# with the per-species ``icex`` from woof/core/milbrandt2_constants.py
# (icexc9 = Gamma(2)/(Gamma(3) cm_r), icexi9 = Gamma(1)/(Gamma(4) cm_i),
# iGS20 = Gamma(1)/(Gamma(1 + dms) cm_s) with the Brandes cm_s, dms the
# pinned snowSpherical=.false. selects, :1290).  Snow is not in WRF's
# block; the same M3/M2 moment ratio over the scheme's alpha_s = 0
# exponential gives 1.5/lambda_s, and the adapter merges it into the one
# RRTMGP ice species by number exactly as Morrison's row does.  ``DE``
# cancels: the scheme's N is per m^3 inside the main routine (NC = NC*DE,
# :1228) and woof's state numbers are per kg, so (DE Q icex)/(N_kg DE)
# is (Q icex)/N_kg.
MILBRANDT2_CLOUD_RADIUS_PER_ILAMBDA = 0.664639
MILBRANDT2_ICE_RADIUS_PER_ILAMBDA = 1.5
MILBRANDT2_SNOW_RADIUS_PER_ILAMBDA = 1.5
#: The scheme's own floors on 1/lambda for ice and snow (``iLAMmin2``,
#: module_mp_milbrandt2mom.F:1686/:1706, repeated at :3371 in the radii
#: block) and its activity thresholds (``epsQ``, ``epsN``, the :3358 and
#: :3368 tests of the radii block).  epsN is stated per m^3 in the scheme and applied here to
#: a per-kg number; at the densities a radiation column sees the two
#: differ by less than an order of magnitude on a threshold that is nine
#: orders below any cloud.
MILBRANDT2_ILAMBDA_MIN_M = 1.0e-10
MILBRANDT2_EPS_Q = 1.0e-14
MILBRANDT2_EPS_N = 1.0e-3

#: ``itab`` column (1-based, as WRF spells it) that IS P3's ice effective
#: radius: ``f1pr06 = access_lookup_table(itab, ..., 6, ...)`` and
#: ``diag_effi = f1pr06`` (module_mp_p3.F:1595/:1610, woof/core/p3.py and
#: woof/core/kernels/p3.cu:1813/:1830), in METRES.
P3_ICE_RADIUS_TABLE_COLUMN = 6


@lru_cache(maxsize=None)
def p3_ice_radius_band_um(root: str | None = None) -> tuple[float, float]:
    """P3's ice-radius band, READ OFF the shipped lookup table.

    THE BREAKAGE THIS CLOSES, measured on a real forecast.  A 6 h GFS run
    on the shipped ``p3-mp50-...`` suite died at its FIRST radiation call
    (12 steps in, radt = 12 min) with "effi is outside the
    physical-plausibility band [1.0, 600.0] microns; the state contract is
    microns -- a radii writer probably emitted another unit".  The writer
    had emitted microns and the values were physical: over 5.59 M cells the
    field ran ``min 1.3832, max 21495`` microns, and those two numbers are
    EXACTLY the minimum and maximum of the shipped table's own column 6.
    P3 is a lookup-table scheme; ``diag_effi`` is that column and nothing
    else, so the attainable set is the column's range.

    Why P3 needs its own band while nothing else does: every other shipped
    scheme clamps its ice radius inside its own microphysics (Thompson's
    ``MIN(re, 125.E-6)``, WSM6's equivalent), so a value outside a generic
    band really is a unit mistake there.  WRF applies NO clamp to P3 --
    ``module_microphysics_driver.F:1597-1598`` passes ``re_ice`` straight
    out of the scheme -- and caps it in RADIATION instead, at
    ``resnow1d = 130`` with the mass discount ``MIN(0.99, (130/res)^2)``
    (module_ra_rrtmg_lw.F:12515-12532, _sw.F:11055-11067), which
    :func:`hydrometeor_paths`' p3 branch already transcribes.  A generic
    600 um band therefore refused the scheme for being itself, three cells
    in a thousand, and named a unit error that had not happened -- a
    refusal that misnames its breakage.

    NOT A WIDENED TOLERANCE.  The band is DERIVED from the table this
    install ships (sha256-pinned by
    :data:`woof.core.p3_tables.TABLE_1_2MOM_ASSET`), so it cannot drift
    from the scheme, and it keeps the whole point of the check: P3 returns
    metres and woof converts to microns, so an unconverted metres emission
    reads about 1.4e-6 and is still refused by the lower bound.
    """

    from woof.core.p3_tables import load_lookup_table_1, p3_table_root

    itab, _ = load_lookup_table_1(p3_table_root() if root is None else root)
    column = itab[:, :, :, P3_ICE_RADIUS_TABLE_COLUMN - 1]
    # The metres -> microns conversion is done HERE the way the writer does
    # it (woof/core/p3.py: ``diag["effi"] * cp.float32(1.0e6)``), in
    # float32.  A float64 multiply of the same table entry can land one ULP
    # away, and at the endpoint that difference is the whole answer: the
    # cell holding the table maximum would test one ULP ABOVE its own band
    # and the run would die on the value the band was derived from.
    scale = np.float32(1.0e6)
    return (float(np.float32(column.min()) * scale),
            float(np.float32(column.max()) * scale))


def effective_radius_bands(scheme: str) -> dict[str, tuple[float, float]]:
    """The micron bands to gate THIS scheme's radiation-facing radii with.

    Every scheme but P3 answers with :data:`EFFECTIVE_RADIUS_PLAUSIBLE_UM`
    unchanged.  P3's ice band is the shipped lookup table's own range, and
    its SNOW band is the same one for the reason the coupling exists: WRF
    radiates P3's single ice category as the snow species at P3's own ice
    radius (``resnow1D = MAX(10., re_ice*1.E6)``,
    module_ra_rrtmg_lw.F:12256, _sw.F:10857), so the value in the snow slot
    IS the ice radius and must be judged by the ice radius's range.

    Milbrandt-Yau's row is :data:`MILBRANDT2_EFFECTIVE_RADIUS_BANDS_UM`,
    the generic bands by value.  The scheme writes no ``effc/effi/effs``
    (its own block is commented out, module_mp_milbrandt2mom.F:3351-3378,
    so ``state.eff*`` for mp=9 hold their allocation-time background for
    the whole run and :func:`hydrometeor_paths` refuses them); the radii
    it radiates are derived inside the adapter from the transported
    number moments and clipped to the RRTMGP table domain there, so the
    band gates nothing on the production path and is published so a
    caller that asks for a scheme's band gets an answer, not a KeyError.
    """

    if scheme == "milbrandt2":
        return dict(MILBRANDT2_EFFECTIVE_RADIUS_BANDS_UM)
    if scheme != "p3":
        return dict(EFFECTIVE_RADIUS_PLAUSIBLE_UM)
    lower, upper = p3_ice_radius_band_um()
    bands = dict(EFFECTIVE_RADIUS_PLAUSIBLE_UM)
    generic_i = EFFECTIVE_RADIUS_PLAUSIBLE_UM["effi"]
    generic_s = EFFECTIVE_RADIUS_PLAUSIBLE_UM["effs"]
    bands["effi"] = (min(lower, generic_i[0]), max(upper, generic_i[1]))
    bands["effs"] = (min(lower, generic_s[0]), max(upper, generic_s[1]))
    return bands

# Snow treatment in the radiative ice path for schemes that provide an
# explicit snow effective radius (the WSM6/Thompson coupling surface).
#   full-snow-mass-into-ice: the adapter's original coupling -- snow joins
#     the ice optical path at full mass with its native radius.
#   wrf-rrtmg-130um-snow-discount: WRF v4.6.1's option-4 explicit-radius
#     coupling (inflg/iceflg=5) -- the ice path is cloud ice only, snow
#     mass is discounted by MIN(0.99, (130/re_s)^2) and re_s is capped at
#     130 um (module_ra_rrtmg_lw.F:12500-12532, module_ra_rrtmg_sw.F:
#     11040-11067; fixture: tests/data/wrf_rrtmg_snow_discount_fixture.csv).
# Selection is bound to the wrf_rrtmg_compatibility receipt token so no
# already-issued run is relabeled; unknown tokens fail closed.
SNOW_TREATMENT_FULL_MASS = "full-snow-mass-into-ice"
SNOW_TREATMENT_WRF_DISCOUNT = "wrf-rrtmg-130um-snow-discount"
SNOW_TREATMENTS = (SNOW_TREATMENT_FULL_MASS, SNOW_TREATMENT_WRF_DISCOUNT)

_SNOW_TREATMENT_BY_COMPATIBILITY = {
    # Native RTE+RRTMGP selection: not a WRF-mapped run, keeps the
    # adapter's original coupling unchanged.
    "none": SNOW_TREATMENT_FULL_MASS,
    # -v1 receipts predate the WRF-matching snow coupling; they keep the
    # behavior they were issued under.
    WRF_RRTMG_TO_RTE_RRTMGP_V1: SNOW_TREATMENT_FULL_MASS,
    # -v2 (current importer default): WRF-matching snow discount.
    WRF_RRTMG_TO_RTE_RRTMGP: SNOW_TREATMENT_WRF_DISCOUNT,
}


def snow_treatment_for_compatibility(token: str) -> str:
    """Map a ``wrf_rrtmg_compatibility`` receipt token to a snow treatment.

    Fails closed: a token this build does not recognize must never be
    silently coerced onto either behavior.
    """
    try:
        return _SNOW_TREATMENT_BY_COMPATIBILITY[token]
    except KeyError:
        raise ValueError(
            "unknown wrf_rrtmg_compatibility token for the radiation "
            f"snow treatment: {token!r}; known tokens: "
            f"{sorted(_SNOW_TREATMENT_BY_COMPATIBILITY)}") from None


# ---------------------------------------------------------------------------
# Which cloud-optics coupling each microphysics selector gets.
#
# This table decides THREE things at once inside
# :meth:`RRTMGPRadiation.__call__`, which is why it is stated here as data
# with its WRF citations rather than inlined as a ``.get(mp, "kessler")``
# default: the branch :func:`hydrometeor_paths` takes (scheme-native radii
# vs. Kessler's constant 10 um / 50 um pair), whether the scheme's
# effc/effi/effs columns are read out of state at all, and -- through
# :data:`_ICE_ACTIVE_SCHEMES` -- the ``f_qi``/``f_qs`` flags
# :func:`cal_cldfra1` is called with.  A scheme that falls through to
# "kessler" therefore does not merely lose its radii: its ice and snow stop
# producing cloud fraction, and an overcast ice cloud radiates as clear sky.
#
# THE ENTRY THAT WAS MISSING.  ``28`` (THOMPSONAERO) is the aerosol-aware
# Thompson package and takes the SAME radiative coupling as classic
# Thompson.  WRF's authority for that, all in the stock v4.6.1 tree:
#
#   * ``Registry/Registry.EM_COMMON:3036`` declares
#     ``package thompsonaero mp_physics==28 - moist:qv,qc,qr,qi,qs,qg;
#     scalar:...;state:re_cloud,re_ice,re_snow`` -- character for character
#     the same ``moist:`` inventory and the same three ``re_*`` state
#     fields as line 3024's ``thompson`` (mp==8).  ``F_QI``/``F_QS`` are
#     therefore both true for mp=28, which is exactly what
#     ``cal_cldfra1`` keys on.
#   * ``phys/module_physics_init.F:1005-1006`` lists ``THOMPSON`` and
#     ``THOMPSONAERO`` as two members of ONE disjunction that sets
#     ``has_reqc = has_reqi = has_reqs = 1`` (:1021-1023); the P3 /
#     Jensen-Ishmael ``has_reqs = 0`` override at :1027-1033 does not
#     name THOMPSONAERO.  So mp=28 hands radiation all three radii.
#     Those same three flags are the ONLY gate on the block that computes
#     them -- ``module_mp_thompson.F:1466`` opens
#     ``IF (has_reqc.ne.0 .and. has_reqi.ne.0 .and. has_reqs.ne.0)`` around
#     calc_effectRad and the :1475-1477 clamps -- so in WRF a scheme
#     computes re_cloud/re_ice/re_snow if and only if radiation consumes
#     them.  Computing them for mp=28 and then discarding them at the
#     radiation boundary is not a WRF configuration at all.
#   * Neither RRTMG wrapper branches on the selector for any of this:
#     ``phys/module_ra_rrtmg_lw.F`` tests ``mp_physics`` only at
#     :12131-12136 and ``_sw.F`` only at :10732-10737, both for
#     FER_MP_HIRES / FER_MP_HIRES_ADVECT / ETAMP_HWRF.  ``cal_cldfra1``'s
#     only ``mp_physics`` branch is the same Ferrier one
#     (``phys/module_radiation_driver.F:3926-3937``); mp=8 and mp=28 both
#     take the ``F_QI .and. F_QC .and. F_QS`` arm at :3870-3877.
#
# woof's own side of the contract is already in place: ``mp_physics == 28``
# allocates effc/effi/effs and background-fills them with the same
# RE_QC_BG/RE_QI_BG/RE_QS_BG values mp=8 uses
# (``woof/core/state.py``), and the mp=28 adapter writes them every step
# through ``launch_aerosol_effective_radius``
# (``woof/core/microphysics_aerosol.py``) under mp_gt_driver's OWN clamps
# (module_mp_thompson.F:1466-1479), which is the identical clamp pair mp=8
# takes because mp_gt_driver is one driver serving both packages.
# ``woof/core/rrtmg_legacy.py`` already carried this same judgement
# (``_MP_DECLARES_RADII[28] = True``, ``_LEGACY_ICE_ACTIVE_MICROPHYSICS``);
# this table is the RTE+RRTMGP half of it, and the two are pinned equal by
# ``tests/test_rrtmgp.py``.
#
# FAILS CLOSED.  Every selector ``woof/config.py`` accepts has a row.  A
# selector without one raises instead of silently resolving to Kessler --
# a silent default is precisely how mp=28 spent four waves radiating its
# ice clouds as clear sky.
_MP_CLOUD_OPTICS_SCHEME = {
    # Registry.EM_COMMON:3014, package passiveqv mp_physics==0 - moist:qv.
    # No condensate species exist at all, so the constant-radius branch is
    # inert: it multiplies zero paths.
    0: "kessler",
    # Registry.EM_COMMON:3015, package kesslerscheme mp_physics==1 -
    # moist:qv,qc,qr.  No qi, no qs, no re_* state.
    1: "kessler",
    # Ferrier-Aligo.  Registry.EM_COMMON:3019, package fer_mp_hires -
    # moist:qv,qc,qr,qi;scalar:qt;state:f_ice_phy,f_rain_phy,f_rimef_phy:
    # qi (the total ice) and no qs, so F_QI is true and F_QS false, and
    # cal_cldfra1 takes the P3 arm and then its own selector arm
    # (module_radiation_driver.F:3879-3887, :3926-3937).  FER_MP_HIRES is
    # absent from the use_mp_re disjunction (module_physics_init.F:
    # 1004-1023), so WRF hands radiation no radii; with no RRTMGP in WRF
    # to copy, the coupling takes Kessler's constant 10 um liquid radius
    # and 50 um ice diameter over qc and the total ice.
    5: "ferrier",
    # WSM3.  Its OWN coupling name, because neither neighbour is right:
    # Registry.EM_COMMON:3017 declares ``package wsm3scheme mp_physics==3 -
    # moist:qv,qc,qr;state:re_cloud,re_ice,re_snow`` -- no qi and no qs, so
    # cal_cldfra1's F_QI and F_QS are both FALSE and it takes the qc-only
    # arm with its 273.15 K phase threshold (module_radiation_driver.F:
    # 3891-3899), unlike WSM6's row -- while module_physics_init.F:1008
    # puts WSM3SCHEME in the use_mp_re disjunction, so the scheme's own
    # radii reach cloud optics, unlike Kessler's row.  WRF's wrappers move
    # cold qc into qi and cold qr into qs for a scheme without F_QI and
    # without warm_rain before reading them (module_ra_sw.F:193-203 and the
    # RRTMG wrappers); that conversion is the radiation side's, keyed on
    # these same Registry flags.
    3: "wsm3",       # Registry.EM_COMMON:3017, wsm3scheme
    4: "wsm6",       # Same cloud/ice/snow radii contract, no graupel.
    6: "wsm6",       # Registry.EM_COMMON:3021, wsm6scheme
    # WSM6R.  Registry.EM_COMMON:3022 declares wsm6rscheme as
    # moist:qv,qc,qr,qi,qs,qg (WSM6's inventory, so F_QI/F_QS are true).
    # WRF hands radiation no radii for it (WSM6RSCHEME is not in the
    # use_mp_re disjunction, module_physics_init.F:1004-1023); RTE+RRTMGP has
    # no WRF referent, and woof/core/wsm6.py writes effc/effi/effs every
    # step from WSM6's own mp_wsm6_effectRad_run on the WSM6R state, whose
    # size distributions are WSM6's.  So the coupling is WSM6's.
    106: "wsm6",     # Registry.EM_COMMON:3022, wsm6rscheme
    8: "thompson",   # Registry.EM_COMMON:3024, thompson
    # Milbrandt-Yau two-moment.  The VALUE is its own coupling name because
    # the radii are the scheme's own and no other row's: WRF hands
    # radiation nothing for mp=9 (MILBRANDT2MOM is absent from the
    # use_mp_re disjunction, module_physics_init.F:1004-1023, so
    # has_reqc/has_reqi/has_reqs stay 0 and RRTMG computes its own radii),
    # but the scheme's author DID write the radii -- as a commented-out
    # block at the end of mp_milbrandt2mom_main
    # (module_mp_milbrandt2mom.F:3351-3378): ``r_eff = M_D(3)/(2 M_D(2))``
    # over MY2005a eqn (2), hard-coded for alpha_c=1, mu_c=3 and
    # alpha_i=0, mu_i=1, giving ``reff_c = 0.664639*iLAMc`` and
    # ``reff_i = 1.5*iLAMi`` from the scheme's own iLAMDA_x.  woof's
    # RTE+RRTMGP adapter has no WRF referent for cloud optics (WRF has no
    # RRTMGP) and already derives Morrison's radii from Morrison's PSD
    # rather than borrowing WRF's has_reqc=0 fallback, so mp=9 takes the
    # same design with its own constants: :func:`hydrometeor_paths`
    # transcribes that block for cloud and ice and extends it to snow with
    # the scheme's own snow PSD (alpha_s=0 exponential over the Brandes
    # m(D) pair the pinned snowSpherical=.false. selects), fed by the six
    # transported number moments.  F_QI/F_QS: Registry.EM_COMMON:3025
    # declares ``package milbrandt2mom mp_physics==9 -
    # moist:qv,qc,qr,qi,qs,qg,qh; scalar:qnc,qnr,qni,qns,qng,qnh`` so
    # cal_cldfra1 takes its QCLD = QI + QC + QS arm
    # (module_radiation_driver.F:3870-3877), the same as Morrison's.
    # The row used to be a judged EXCLUSION (recorded below); it was
    # retired the way P3's was, because with the default variant being
    # RTE+RRTMGP the exclusion refused mp=9 outright on every bare run.
    9: "milbrandt2",  # Registry.EM_COMMON:3025, milbrandt2mom
    10: "morrison",  # Registry.EM_COMMON:3026, morr_two_moment
    # WDM6.  The VALUE names a COUPLING, not a scheme -- 28 resolves to
    # "thompson" on the same principle -- and WDM6's coupling is WSM6's,
    # for the two reasons that decide this table:
    #  * F_QI/F_QS.  Registry.EM_COMMON:3031 declares wdm6scheme as
    #    ``moist:qv,qc,qr,qi,qs,qg``, WSM6's inventory (:3021) character for
    #    character, so cal_cldfra1 takes the same QCLD = QI + QC + QS arm
    #    (module_radiation_driver.F:3870-3877).
    #  * Explicit radii.  module_physics_init.F:1013 lists WDM6SCHEME in the
    #    same has_reqc/has_reqi/has_reqs disjunction as WSM6SCHEME (:1010),
    #    and the P3 ``has_reqs = 0`` override (:1027-1033) does not name it,
    #    so all three are 1 and the scheme's own re_cloud/re_ice/re_snow
    #    (effectRad_wdm6, module_mp_wdm6.F:3135-3234) reach cloud optics.
    # What is DIFFERENT about WDM6 -- a droplet radius built from prognostic
    # nc rather than a fixed number -- is inside the radii the scheme
    # supplies, not in how radiation consumes them, so it changes the
    # values on this path and not the path.
    16: "wsm6",      # Registry.EM_COMMON:3031, wdm6scheme
    # WSM7: WSM6's coupling on both counts -- F_QI/F_QS true
    # (Registry.EM_COMMON:3034 moist:qv,qc,qr,qi,qs,qg,qh) and explicit
    # radii (WSM7SCHEME in the use_mp_re list, module_physics_init.F:1011)
    # -- and, as in WRF, hail is not a radiating species.
    24: "wsm6",      # Registry.EM_COMMON:3034, wsm7scheme
    18: "nssl",      # Registry.EM_COMMON:3033, nssl_2mom
    28: "thompson",  # Registry.EM_COMMON:3036, thompsonaero
    # P3 one-category.  The VALUE is its own coupling name and not a
    # borrowed one, because P3 is the only scheme in this table whose
    # has_req* triple is not all-ones -- and WRF's RRTMG wrappers carry a
    # named branch for exactly that case.  Read on a stock WRF v4.7.1 tree
    # (phys/module_mp_p3.F there hashes to the same
    # 716950a3081ec4e338c9a918d26ec80f7ee0e40b3e284283f070423237f6a3c6 the
    # Fortran oracle pins); v4.6.1 line numbers, the tree the rows above
    # cite, are given second where the text moved.
    #
    #  * ``Registry/Registry.EM_COMMON:3043`` (v4.6.1 :3038) declares
    #    ``package p3_1category mp_physics==50 -
    #    moist:qv,qc,qr,qi;scalar:qni,qnr,qir,qib;
    #    state:re_cloud,re_ice,vmi3d,rhopo3d,di3d,refl_10cm,th_old,qv_old``.
    #    ``moist`` carries qi and NO qs, so cal_cldfra1's ``F_QI`` is true
    #    and its ``F_QS`` is FALSE; ``state`` carries re_cloud and re_ice
    #    and NO re_snow.  Both absences are the same physical fact: P3 has
    #    ONE ice category spanning the snow-to-graupel continuum through
    #    rime mass and rime volume, so there is no snow species to have a
    #    mixing ratio or an effective radius.
    #  * ``phys/module_physics_init.F:1018`` (v4.6.1 :1017) names
    #    P3_1CATEGORY in the ``use_mp_re`` disjunction that sets
    #    ``has_reqc = has_reqi = has_reqs = 1`` at :1022-1024 (v4.6.1
    #    :1021-1023) -- and then :1027-1034 (v4.6.1 :1026-1033), under the
    #    comment "for P3, to ensure correct coupling with predicted
    #    effective radii", sets ``has_reqs = 0`` for P3_1CATEGORY,
    #    P3_1CATEGORY_NC, P3_1CAT_3MOM, P3_2CATEGORY and JENSEN_ISHMAEL.
    #    THE ROW IS has_reqc=1, has_reqi=1, has_reqs=0.  Cloud and ice
    #    effective radii come from the scheme -- WRF's diag_effc_3d and
    #    diag_effi_3d (module_mp_p3.F:757-758, written at :1557 and :1610)
    #    are woof's ``state.effc``/``state.effi`` (woof/core/p3.py) --
    #    and no snow radius exists to be supplied.
    #  * ``phys/module_radiation_driver.F:3879-3887`` (same lines in
    #    v4.6.1) is cal_cldfra1's OWN P3 arm, comment "for P3, mp option
    #    50 or 51": ``IF (F_QI .and. F_QC .and. .not. F_QS)`` gives
    #    QCLD = QI + QC and weight = QI/QCLD, distinct from the
    #    :3870-3877 arm every other row here takes.
    #  * ``phys/module_ra_rrtmg_lw.F:12250-12261`` and
    #    ``phys/module_ra_rrtmg_sw.F:10851-10863`` (same lines in v4.6.1)
    #    are the radiative half, and they are why this row is not merely
    #    "ice, without snow": under
    #    ``has_reqs == 0 .and. has_reqi /= 0 .and. has_reqc /= 0`` WRF
    #    sets inflg = iceflg = 5 and REMAPS the species --
    #    ``resnow1D = MAX(10., re_ice*1.E6)``, ``QS1D = QI3D``,
    #    ``QI1D = 0.``, ``reice1D = 10.`` -- so P3's ice mass is radiated
    #    through the SNOW (Fu) parameterisation at P3's own ice radius and
    #    the cloud-ice path is left empty.  :func:`hydrometeor_paths`
    #    transcribes that remap; no snow radius is invented, and none is
    #    accepted.
    50: "p3",
}

from woof.microphysics_schemes import NAMED_SCHEMES as _NAMED_MP_SCHEMES
# Named schemes: the capability row names the radii family.
_MP_CLOUD_OPTICS_SCHEME.update({
    _s.mp_id: _s.cloud_optics_family for _s in _NAMED_MP_SCHEMES.values()})

# WHY A SELECTOR THIS BUILD ACCEPTS HAS NO ROW ABOVE.  One entry per such
# selector, stating the reason a row would be WRONG -- not a note that one
# is missing.  A set that merely omits a scheme is an omission, and the
# next reader cannot tell an omission from an oversight; the tree's other
# deliberate exclusions (microphysics_transition's named-refusal tables)
# are recorded the same way.
#
# The table is EMPTY: every selector this build accepts has a row above.
# It stays as a mechanism, not a list -- an unjudged selector still fails
# closed with the request for a source row or an entry here, and
# tools/build_registry.py still derives ``constraints.refused_when`` from
# membership.  Its last entry, mp=9, was retired the day the
# ``"milbrandt2"`` row landed: the entry said Morrison's row "would derive
# the radii from a gamma distribution that is not this scheme's", which
# was true and was the argument for a row of the scheme's own, not for
# refusing the pairing.  P3's entry went the same way before it.
_NO_CLOUD_OPTICS_COUPLING: dict[int, str] = {2: "Lin supplies no effective radii to this adapter; borrowing another scheme would change its cloud distribution", 30: "the spectral effective-radius integration has not been qualified; reusing bulk radii would replace the size-distribution optics"}

#: The two adapters that DO serve these selectors, named by every refusal
#: so none of them is a dead end.  One copy: a user told different things
#: about the same door stops trusting either.
_CLOUD_OPTICS_REMEDY = (
    "Set ra_rrtmg_variant='rrtmg_legacy' (which computes its own radii "
    "the way WRF does), or select ra_lw_physics=0/ra_sw_physics=1 "
    "(Dudhia).")


def _require_agreement_with_the_registry() -> None:
    """The two tables above ARE the registry's ``cloud_optics`` row.

    tools/build_registry.py pulls ``rte_rrtmgp_coupling`` from them, so
    they are held equal to the generated copy at import: a scheme added
    to one and not the other fails here, and plan review
    (woof.physics_registry.consumer_row_gaps) refuses an implemented
    scheme with no row before step 0 instead of this module refusing it
    at the first radiation call.
    """

    from woof.physics_registry import require_consumer_rows_agreement

    observed = {mp: None for mp in _NO_CLOUD_OPTICS_COUPLING}
    observed.update(_MP_CLOUD_OPTICS_SCHEME)
    require_consumer_rows_agreement(
        "woof.core.rrtmgp._MP_CLOUD_OPTICS_SCHEME + _NO_CLOUD_OPTICS_COUPLING",
        "microphysics", "cloud_optics", observed,
        project=lambda row: row["rte_rrtmgp_coupling"])


_require_agreement_with_the_registry()

#: Schemes whose Registry package carries ``qi`` and ``qs`` in ``moist``,
#: i.e. the ones for which the radiation driver's ``F_QI``/``F_QS`` are
#: true and :func:`cal_cldfra1` takes its QCLD = QI + QC + QS arm
#: (module_radiation_driver.F:3870-3877).  Derived from the table above so
#: the two can never disagree; Kessler's package has neither species.
_ICE_ACTIVE_SCHEMES = ("wsm6", "thompson", "milbrandt2", "morrison", "nssl",
                       "p3", "ferrier")

#: The subset of the above whose Registry package ALSO carries ``qs`` in
#: ``moist``, i.e. the ones for which the driver's ``F_QS`` is true.  P3 is
#: the one member of this table that is ice-active WITHOUT a snow species:
#: ``Registry.EM_COMMON:3043`` gives mp=50 ``moist:qv,qc,qr,qi`` and no qs,
#: which is what selects cal_cldfra1's own P3 arm
#: (module_radiation_driver.F:3879-3887) instead of the :3870-3877 arm.
#: Kessler's package has neither species and is in neither tuple.
_SNOW_SPECIES_SCHEMES = ("wsm6", "thompson", "milbrandt2", "morrison", "nssl")


def cloud_optics_scheme(mp_physics) -> str:
    """Resolve an ``mp_physics`` selector to its cloud-optics coupling.

    See :data:`_MP_CLOUD_OPTICS_SCHEME`.  Raises rather than defaulting:
    an unmapped selector must not silently inherit Kessler's constant
    radii and ice-free cloud fraction.
    """
    selector = int(mp_physics)
    try:
        return _MP_CLOUD_OPTICS_SCHEME[selector]
    except KeyError:
        pass
    recorded = _NO_CLOUD_OPTICS_COUPLING.get(selector)
    if recorded is not None:
        # A JUDGED exclusion, not a gap.  Say what was decided and why,
        # and name the two adapters that do serve the scheme -- never
        # "add a row", which sends the reader to undo the decision.
        raise NotImplementedError(
            f"mp_physics={selector} has no RTE+RRTMGP cloud-optics "
            f"coupling, deliberately: {recorded} {_CLOUD_OPTICS_REMEDY}"
        ) from None
    raise NotImplementedError(
        f"mp_physics={selector} has no RTE+RRTMGP cloud-optics coupling; "
        "add a row to woof.core.rrtmgp._MP_CLOUD_OPTICS_SCHEME with its "
        "Registry package (which fixes F_QI/F_QS for cal_cldfra1) and "
        "its module_physics_init.F use_mp_re membership (which fixes "
        "whether its effective radii reach cloud optics) rather than "
        "letting it fall through to Kessler's constant 10 um / 50 um "
        "radii -- or, if the omission is deliberate, record the reason in "
        "woof.core.rrtmgp._NO_CLOUD_OPTICS_COUPLING so the next reader "
        "finds a decision instead of a gap") from None


def scheme_is_ice_active(scheme: str) -> bool:
    """``F_QI`` for a resolved cloud-optics scheme name.

    This used to answer ``F_QI`` and ``F_QS`` together, which was true of
    every scheme in the table until P3: mp=50 is ice-active and has no
    snow species at all, so the two flags separate.  Ask
    :func:`scheme_has_snow_species` for ``F_QS``.
    """

    return scheme in _ICE_ACTIVE_SCHEMES


def scheme_has_snow_species(scheme: str) -> bool:
    """``F_QS`` for a resolved cloud-optics scheme name."""

    return scheme in _SNOW_SPECIES_SCHEMES


# RRTMGP v1.9's lowest reference pressure.  WRF's wrappers use 0/1e-5 mb
# at TOA, but the pinned RRTMGP example raises that interface to the gas-table
# floor before computing dry-column amounts (the same adaptation formerly
# applied directly to woof's model top).
RRTMGP_TOA_PRESSURE_PA = 1.005183574463
WRF_LW_UPPER_DELTA_P_PA = 400.0
MAX_RADIATION_LAYERS = 128

# module_ra_rrtmg_lw.F:11904-11932.  Pressures are hPa.  The table is the
# weighted standard-atmosphere mean used by WRF to temperature its 4-hPa
# model-top buffer layers.
_WRF_LW_PPROF_HPA = np.array([
    1000.00, 855.47, 731.82, 626.05, 535.57, 458.16,
    391.94, 335.29, 286.83, 245.38, 209.91, 179.57,
    153.62, 131.41, 112.42, 96.17, 82.27, 70.38,
    60.21, 51.51, 44.06, 37.69, 32.25, 27.59,
    23.60, 20.19, 17.27, 14.77, 12.64, 10.81,
    9.25, 7.91, 6.77, 5.79, 4.95, 4.24,
    3.63, 3.10, 2.65, 2.27, 1.94, 1.66,
    1.42, 1.22, 1.04, 0.89, 0.76, 0.65,
    0.56, 0.48, 0.41, 0.35, 0.30, 0.26,
    0.22, 0.19, 0.16, 0.14, 0.12, 0.10,
], np.float64)
_WRF_LW_TPROF_K = np.array([
    286.96, 281.07, 275.16, 268.11, 260.56, 253.02,
    245.62, 238.41, 231.57, 225.91, 221.72, 217.79,
    215.06, 212.74, 210.25, 210.16, 210.69, 212.14,
    213.74, 215.37, 216.82, 217.94, 219.03, 220.18,
    221.37, 222.64, 224.16, 225.88, 227.63, 229.51,
    231.50, 233.73, 236.18, 238.78, 241.60, 244.44,
    247.35, 250.33, 253.32, 256.30, 259.22, 262.12,
    264.80, 266.50, 267.59, 268.44, 268.69, 267.76,
    266.13, 263.96, 261.54, 258.93, 256.15, 253.23,
    249.89, 246.67, 243.48, 240.25, 236.66, 233.86,
], np.float64)

#: ``(pressure, temperature)`` for WRF's LW upper-atmosphere climatology, in
#: ascending pressure order, one array pair per array module.  Uploaded once
#: instead of once per radiation call: the profile is a literal in this file
#: and the sort is over 30-odd elements, but every call used to pay two
#: pageable host-to-device copies for it -- and an H2D of any kind is illegal
#: inside a CUDA graph capture, so on a radiation-due step this was the line
#: that made the step uncapturable.  Same bytes, same interpolation, same
#: answer.
_LW_CLIMATOLOGY: dict = {}

#: The shortwave g-point solar source, uploaded once per gas-table set.
#: Same reasoning as :data:`_LW_CLIMATOLOGY`: a shipped table that was
#: re-copied from host memory on every radiation call.
#:
#: Held HERE and not on the radiation callable, which was the first attempt:
#: ``woof/io/restart.py`` audits every array attribute a callable owns and
#: refused it as unclassified, and it was right to -- an array on the driver
#: is state that a restart has to account for, and a cached constant is not
#: state.  The key holds a reference to the tables object so its ``id`` can
#: never be recycled under the cache.
_SOLAR_SOURCE: dict = {}


def _cache_device_key(xp=None):
    """Cache key that includes the CARD, because the value is a DEVICE array.

    A memoized device array keyed on anything but the device is handed to
    whichever card asks for it next.  With peer access that merely reads
    across the bus; WITHOUT it -- which is every GeForce box, including the
    4x RTX 5080 this was found on -- CuPy raises outright:

        ValueError: The device where the array resides (0) is different from
        the current device (1).  Peer access is unavailable...

    ``tilestream/mgstream.py`` documents five sites with exactly this defect
    (rrtmgp.GasTables/CloudTables.to_device, kf._device_table,
    mynn_pbl_runtime._VALIDITY_FLAGS, noahmp_vegeflux_gpu._module,
    noahmp_slab_libm._KERNEL_CACHE) and keys each on the current device.
    These two were NOT among them, and they are on the longwave and
    shortwave radiation paths, so any multi-GPU run with radiation on hits
    them at the first due step on the second card.
    """
    name = "cupy" if xp is None else getattr(xp, "__name__", str(xp))
    if name == "cupy":
        import cupy as cp
        return (name, int(cp.cuda.Device().id))
    return (name,)


def _solar_source_device(sw_tables):
    import cupy as cp

    key = (id(sw_tables), _cache_device_key())
    held = cached_ready(cp, _SOLAR_SOURCE, key,
                        lambda: (sw_tables, cp.asarray(sw_tables.solar_source, dtype=DTYPE)))
    return held[1]


#: Per-device trace-gas ``(slots, values)`` pairs, keyed the same way as
#: :data:`_LW_CLIMATOLOGY`.  Two arrays of one or two elements, otherwise
#: rebuilt per chunk -- two allocations and two host-to-device copies each,
#: 2364 times an hour, for a value that changes once per forecast.
_TRACE_VMR_DEVICE: dict = {}


def _trace_vmr_device(trace, *, xp, _uncached=False):
    """Device ``(slot_indices, values)`` for the well-mixed trace gases."""
    key = (_cache_device_key(xp), trace)
    if hasattr(xp, "cuda") and not _uncached:
        return cached_ready(xp, _TRACE_VMR_DEVICE, key,
                            lambda: _trace_vmr_device(trace, xp=xp, _uncached=True))
    cached = None if _uncached else _TRACE_VMR_DEVICE.get(key)
    if cached is None:
        cached = (
            xp.asarray(np.asarray([slot for slot, _ in trace],
                                  dtype=np.int32)),
            xp.asarray(np.asarray([value for _, value in trace],
                                  dtype=np.float32)),
        )
        if not _uncached:
            _TRACE_VMR_DEVICE[key] = cached
    return cached


def _lw_climatology(xp, _uncached=False):
    key = _cache_device_key(xp)
    if hasattr(xp, "cuda") and not _uncached:
        return cached_ready(xp, _LW_CLIMATOLOGY, key,
                            lambda: _lw_climatology(xp, _uncached=True))
    cached = None if _uncached else _LW_CLIMATOLOGY.get(key)
    if cached is None:
        order = np.argsort(_WRF_LW_PPROF_HPA)
        cached = (xp.asarray(_WRF_LW_PPROF_HPA[order], dtype=DTYPE),
                  xp.asarray(_WRF_LW_TPROF_K[order], dtype=DTYPE))
        if not _uncached:
            _LW_CLIMATOLOGY[key] = cached
    return cached


@dataclass(frozen=True)
class _RadiationColumnProfile:
    play: object
    plev: object
    tlay: object
    tlev: object
    qv: object
    model_nlay: int
    upper_nlay: int


def rrtmgp_above_model_layer_counts(
        p_top: float, *, pressure_floor: float = RRTMGP_TOA_PRESSURE_PA,
) -> tuple[int, int]:
    """Return WRF v4.6.1's ``(LW, SW)`` above-model layer counts.

    LW uses ``nint(p_top*.01/4)`` 4-hPa layers
    (module_ra_rrtmg_lw.F:11565,12998-13001).  SW uses one model-top-to-TOA
    layer (:10756-10760).  A column whose possible layer midpoint is already
    below the pinned coefficient floor needs no representable extra layer.
    """
    p_top = float(p_top)
    pressure_floor = float(pressure_floor)
    if (not np.isfinite(p_top) or not np.isfinite(pressure_floor)
            or p_top < 0.0 or pressure_floor <= 0.0):
        raise ValueError("radiation top pressures must be finite and nonnegative")
    if p_top <= pressure_floor:
        return 0, 0
    # Fortran NINT is nearest integer, with a positive half rounded upward.
    lw = int(np.floor(p_top / WRF_LW_UPPER_DELTA_P_PA + 0.5))
    sw = int(0.5 * p_top >= pressure_floor)
    return max(0, lw), sw


def _validate_model_top_interface(plev, p_top: float, *, xp=None) -> None:
    """Check the model-top interface against the declared ``p_top``.

    This reduces on device and reads the result back, so it SYNCHRONIZES.
    That is why the chunk loop hoists it: ``plev`` is the same array for
    every chunk of a firing, so checking each chunk over again bought
    nothing and cost a synchronization per chunk per domain per band.
    """
    if xp is None:
        import cupy as xp
    if not bool(xp.allclose(
            plev[:, -1], DTYPE(p_top), rtol=DTYPE(0.0),
            atol=DTYPE(max(1.0e-3, abs(float(p_top)) * 2.0e-7)))):
        raise ValueError(
            "radiation workspace top pressure does not match the model-top "
            f"interface ({p_top:g} Pa)")


#: Per-device above-model rows that depend on the model TOP alone, keyed
#: like :data:`_LW_CLIMATOLOGY`.  The synthetic cap above the model is built
#: from `plev[:, -1]`, and WRF's mass coordinate makes that one number for
#: the whole grid -- `physics.py` fills the top interface with a scalar
#: (`p_interface[nz] = state.p_top`).  So the pressure ladder and the
#: climatology lookups on it are the SAME for every column and every chunk,
#: and rebuilding them per chunk cost two `interp` calls and a dozen small
#: kernels 1182 times an hour.
_ABOVE_MODEL_ROWS: dict = {}


def _above_model_rows(kind, upper_nlay, p_top, pressure_floor, *, xp, _uncached=False):
    """``(plev_row, play_row, climo_top, climo_row)`` for a uniform cap.

    Shaped (1, upper_nlay) so the caller broadcasts against its own
    (ncol, 1) top temperature.  Every value is what the per-column
    construction produced when every column carried the same top pressure,
    element for element -- these are the same expressions on a single row.
    """
    key = (_cache_device_key(xp), kind, int(upper_nlay),
           float(p_top), float(pressure_floor))
    if hasattr(xp, "cuda") and not _uncached:
        return cached_ready(xp, _ABOVE_MODEL_ROWS, key,
                            lambda: _above_model_rows(kind, upper_nlay, p_top, pressure_floor, xp=xp, _uncached=True))
    cached = None if _uncached else _ABOVE_MODEL_ROWS.get(key)
    if cached is not None:
        return cached
    top_pressure = xp.full((1, 1), DTYPE(p_top), dtype=DTYPE)
    if kind == "sw":
        plev_row = xp.full((1, 1), DTYPE(pressure_floor), dtype=DTYPE)
        play_row = DTYPE(0.5) * top_pressure
        climo_top = climo_row = None
    else:
        offsets = (WRF_LW_UPPER_DELTA_P_PA
                   * xp.arange(1, upper_nlay + 1, dtype=DTYPE))[None, :]
        plev_row = top_pressure - offsets
        plev_row[:, -1] = DTYPE(pressure_floor)
        interfaces = xp.concatenate((top_pressure, plev_row), axis=1)
        play_row = DTYPE(0.5) * (interfaces[:, :-1] + interfaces[:, 1:])
        pprof, tprof = _lw_climatology(xp)
        climo_top = xp.interp(
            top_pressure.ravel() * DTYPE(0.01), pprof, tprof).reshape(1, 1)
        climo_row = xp.interp(
            plev_row.ravel() * DTYPE(0.01), pprof, tprof).reshape(
                1, upper_nlay)
    cached = (plev_row, play_row, climo_top, climo_row)
    if not _uncached:
        _ABOVE_MODEL_ROWS[key] = cached
    return cached


def _extend_above_model_profile(
        play, plev, tlay, tlev, qv, *, p_top: float, kind: str,
        pressure_floor: float = RRTMGP_TOA_PRESSURE_PA, xp=None,
        validate_top: bool = True,
        uniform_top: bool = False) -> _RadiationColumnProfile:
    """Append WRF's clear upper atmosphere in bottom-to-top layout.

    ``uniform_top`` asserts that every column's ``plev[:, -1]`` holds the
    same value, which lets the pressure ladder above the model and the two
    climatology lookups on it come from :func:`_above_model_rows` as single
    cached rows instead of being rebuilt per column per chunk.  It is
    OFF by default and the caller owns the claim: `RRTMGPRadiation.__call__`
    checks it on device once per firing.  The values are identical either
    way -- the fast path runs the same expressions on one row and lets
    broadcasting do what the wide arrays were doing by hand.

    The LW pressure/temperature construction transcribes
    ``module_ra_rrtmg_lw.F:12322-12393``: 400-Pa interfaces, a final TOA
    interface adapted to the RRTMGP coefficient floor, the 60-level WRF
    standard-atmosphere temperature interpolant shifted to meet the live
    model-top temperature, and layer temperatures averaged from interfaces.
    SW transcribes ``module_ra_rrtmg_sw.F:10910-10925``: one layer with
    ``play=.5*ptop`` and an isothermal top.  Both hold top-layer water vapor
    constant; well-mixed gases and pressure-interpolated ozone are filled by
    :meth:`RRTMGPRadiation._gas_vmr` after extension.
    """
    if kind not in ("lw", "sw"):
        raise ValueError("above-model profile kind must be 'lw' or 'sw'")
    if xp is None:
        import cupy as xp

    play = xp.ascontiguousarray(xp.asarray(play, dtype=DTYPE))
    plev = xp.ascontiguousarray(xp.asarray(plev, dtype=DTYPE))
    tlay = xp.ascontiguousarray(xp.asarray(tlay, dtype=DTYPE))
    tlev = xp.ascontiguousarray(xp.asarray(tlev, dtype=DTYPE))
    qv = xp.ascontiguousarray(xp.asarray(qv, dtype=DTYPE))
    if play.ndim != 2:
        raise ValueError("play must have shape (ncol,nlay)")
    ncol, model_nlay = play.shape
    expected = {
        "plev": (plev, (ncol, model_nlay + 1)),
        "tlay": (tlay, play.shape), "tlev": (tlev, plev.shape),
        "qv": (qv, play.shape),
    }
    for name, (value, shape) in expected.items():
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {value.shape}")

    lw_upper, sw_upper = rrtmgp_above_model_layer_counts(
        p_top, pressure_floor=pressure_floor)
    upper_nlay = lw_upper if kind == "lw" else sw_upper
    if model_nlay + upper_nlay > MAX_RADIATION_LAYERS:
        raise ValueError(
            "RRTMGP CUDA RTE supports at most "
            f"{MAX_RADIATION_LAYERS} layers including the above-model "
            f"column; got {model_nlay + upper_nlay}")
    if validate_top:
        _validate_model_top_interface(plev, p_top, xp=xp)
    if upper_nlay == 0:
        return _RadiationColumnProfile(
            play, plev, tlay, tlev, qv, model_nlay, 0)

    top_pressure = plev[:, -1:]
    top_temperature = tlev[:, -1:]
    top_qv = qv[:, -1:]
    rows = (_above_model_rows(kind, upper_nlay, p_top, pressure_floor, xp=xp)
            if uniform_top else None)
    if kind == "sw":
        if rows is not None:
            plev_row, play_row, _, _ = rows
            upper_plev = xp.broadcast_to(plev_row, (ncol, 1))
            upper_play = xp.broadcast_to(play_row, (ncol, 1))
        else:
            upper_plev = xp.full(
                (ncol, 1), DTYPE(pressure_floor), dtype=DTYPE)
            upper_play = DTYPE(0.5) * top_pressure
        upper_tlev = top_temperature
        upper_tlay = top_temperature
    else:
        if rows is not None:
            plev_row, play_row, climo_top, climo_upper = rows
            upper_plev = xp.broadcast_to(plev_row, (ncol, upper_nlay))
            upper_play = xp.broadcast_to(play_row, (ncol, upper_nlay))
        else:
            offsets = (WRF_LW_UPPER_DELTA_P_PA
                       * xp.arange(1, upper_nlay + 1, dtype=DTYPE))[None, :]
            upper_plev = top_pressure - offsets
            upper_plev[:, -1] = DTYPE(pressure_floor)
            if upper_nlay > 1:
                # THE BREAKAGE THIS PREVENTS: the layer count comes from the
                # one declared p_top, the ladder from each column's own top
                # interface.  WRF's mass coordinate makes those the same
                # number; a height-coordinate caller (the MPAS column seam)
                # hands a top pressure per column, and a column whose top
                # sits more than one 4-hPa step below the declared value got
                # an interface at or below the coefficient floor -- zero or
                # negative pressure, and a non-finite longwave heating rate
                # at the first call (global 30 km, 2026-10-11).  Such a
                # column takes the same number of layers evenly spaced from
                # its own top to the floor; every other column keeps WRF's
                # ladder to the bit.
                below = upper_plev[:, -2:-1] <= DTYPE(pressure_floor)
                fraction = (xp.arange(1, upper_nlay + 1, dtype=DTYPE)
                            / DTYPE(upper_nlay))[None, :]
                even = top_pressure + (
                    DTYPE(pressure_floor) - top_pressure) * fraction
                even[:, -1] = DTYPE(pressure_floor)
                upper_plev = xp.where(below, even, upper_plev)
            all_upper_interfaces = xp.concatenate(
                (top_pressure, upper_plev), axis=1)
            upper_play = DTYPE(0.5) * (
                all_upper_interfaces[:, :-1] + all_upper_interfaces[:, 1:])

            pprof, tprof = _lw_climatology(xp)
            climo_top = xp.interp(
                top_pressure.ravel() * DTYPE(0.01), pprof, tprof).reshape(
                    ncol, 1)
            climo_upper = xp.interp(
                upper_plev.ravel() * DTYPE(0.01), pprof, tprof).reshape(
                    ncol, upper_nlay)
        if hasattr(xp, "RawModule"):
            upper_qv = xp.broadcast_to(top_qv, (ncol, upper_nlay))
            outputs = _append_profile_fields(
                (play, plev, tlay, tlev, qv),
                (upper_play, upper_plev,
                 xp.broadcast_to(climo_top, (ncol, 1)),
                 xp.broadcast_to(climo_upper, (ncol, upper_nlay)), upper_qv),
                xp=xp, derive_lw_temperature=True)
            return _RadiationColumnProfile(*outputs, model_nlay, upper_nlay)
        # Broadcast, not tile: (1, upper) against the chunk's own (ncol, 1)
        # top temperature is the same subtraction and the same addition the
        # wide arrays performed, element for element.
        upper_tlev = climo_upper + (top_temperature - climo_top)
        all_upper_tlev = xp.concatenate(
            (top_temperature, upper_tlev), axis=1)
        upper_tlay = DTYPE(0.5) * (
            all_upper_tlev[:, :-1] + all_upper_tlev[:, 1:])
    upper_qv = xp.broadcast_to(top_qv, (ncol, upper_nlay))
    outputs = _append_profile_fields(
        (play, plev, tlay, tlev, qv),
        (upper_play, upper_plev, upper_tlay, upper_tlev, upper_qv), xp=xp)
    return _RadiationColumnProfile(*outputs, model_nlay, upper_nlay)


def _append_profile_fields(model, upper, *, xp, derive_lw_temperature=False):
    """Concatenate five fields without changing their promoted dtypes."""
    if not hasattr(xp, "RawModule"):
        return tuple(xp.ascontiguousarray(xp.concatenate((a, b), axis=1))
                     for a, b in zip(model, upper))
    from woof.core.kernels import get_kernel

    ncol, model_nlay = model[0].shape
    upper_nlay = upper[0].shape[1]
    temperature_dtype = xp.result_type(model[2].dtype, upper[2].dtype)
    if (temperature_dtype != xp.result_type(model[3].dtype, upper[3].dtype)
            or temperature_dtype not in (xp.float32, xp.float64)):
        raise TypeError("cap temperature fields must share float32 or float64")
    if derive_lw_temperature and not (upper[2].dtype == xp.float64
                                      == upper[3].dtype):
        # rrtmgp_append_lw_profile reads the climatology rows as doubles
        # (cupy.interp's float64 output); float32 rows would be read as
        # the wrong bytes.
        raise TypeError("LW cap climatology rows must be float64")
    shapes = tuple((ncol, a.shape[1] + upper_nlay) for a in model)
    outputs = tuple(xp.empty(shape, dtype=xp.result_type(a.dtype, b.dtype))
                    for shape, a, b in zip(shapes, model, upper))
    strides = tuple(np.int32(stride // v.itemsize)
                    for v in upper for stride in v.strides)
    threads = 256
    size = ncol * (model_nlay + upper_nlay + 1)
    suffix = "fp64" if temperature_dtype == xp.float64 else "fp32"
    entry = ("rrtmgp_append_lw_profile" if derive_lw_temperature else
             f"rrtmgp_append_profile_{suffix}")
    get_kernel("rrtmgp_cloud", entry)(
        ((size + threads - 1) // threads,), (threads,),
        (*model, *upper, *outputs, np.int32(ncol), np.int32(model_nlay),
         np.int32(upper_nlay), *strides))
    return outputs



def _model_flux_interfaces(flux, model_nlay: int, *, xp=None):
    """Discard upper-atmosphere heating levels while retaining model-top flux."""
    if xp is None:
        import cupy as xp
    flux = xp.asarray(flux, dtype=DTYPE)
    if flux.ndim != 2 or flux.shape[1] < int(model_nlay) + 1:
        raise ValueError("radiation flux column does not reach model top")
    return flux[:, :int(model_nlay) + 1]


def _store_model_flux_pair(up, dn, model_nlay, out_up, out_dn, *,
                           daylight=None, xp=None):
    """Copy both model fluxes straight to their final chunk views."""
    if xp is None:
        import cupy as xp
    if not hasattr(xp, "RawModule"):
        for source, target in ((up, out_up), (dn, out_dn)):
            values = _model_flux_interfaces(source, model_nlay, xp=xp)
            target[...] = values if daylight is None else xp.where(
                daylight[:, None], values, DTYPE(0.0))
        return
    from woof.core.kernels import get_kernel

    up, dn = (xp.ascontiguousarray(xp.asarray(v, dtype=DTYPE))
              for v in (up, dn))
    model_nlev = int(model_nlay) + 1
    if up.ndim != 2 or dn.shape != up.shape or up.shape[1] < model_nlev:
        raise ValueError("radiation flux column does not reach model top")
    ncol, source_nlev = up.shape
    outputs = tuple(_workspace_output(v, (ncol, model_nlev), "model flux")
                    for v in (out_up, out_dn))
    if daylight is None:
        mask = up
    else:
        mask = xp.ascontiguousarray(xp.asarray(daylight, dtype=xp.bool_))
        if mask.shape != (ncol,):
            raise ValueError("daylight must have one value per column")
    threads = 256
    get_kernel("rrtmgp_cloud", "rrtmgp_store_model_flux_pair")(
        ((ncol * model_nlev + threads - 1) // threads,), (threads,),
        (up, dn, mask, *outputs, np.int32(ncol), np.int32(source_nlev),
         np.int32(model_nlev), np.int32(daylight is not None)))


def _prepare_sw_inputs(albedo, solar, mu, nlay, *, out=None, xp=None):
    """Materialize all three exact solver-input broadcasts together."""
    if xp is None:
        import cupy as xp
    albedo, solar = (xp.ascontiguousarray(xp.asarray(v, dtype=DTYPE))
                     for v in (albedo, solar))
    mu = xp.ascontiguousarray(xp.asarray(mu))
    if mu.dtype not in (xp.float32, xp.float64) or not hasattr(xp, "RawModule"):
        mu = xp.asarray(mu, dtype=DTYPE)
    if albedo.ndim != 1 or solar.ndim != 1 or mu.shape != albedo.shape:
        raise ValueError("SW broadcast inputs must be one-dimensional")
    ncol, ngpt = albedo.size, solar.size
    shapes = ((ncol, ngpt), (ncol, ngpt), (ncol, int(nlay)))
    if not hasattr(xp, "RawModule"):
        values = (xp.broadcast_to(albedo[:, None], shapes[0]),
                  xp.broadcast_to(solar[None, :], shapes[1]),
                  xp.broadcast_to(mu[:, None], shapes[2]))
        if out is None:
            return tuple(xp.ascontiguousarray(v) for v in values)
        for a, b in zip(out, values): a[...] = b
        return out
    from woof.core.kernels import get_kernel

    if out is None:
        out = (None,) * 3
    if len(out) != 3:
        raise ValueError("SW broadcast outputs must contain three arrays")
    outputs = tuple(_workspace_output(v, shape, name)
                    for v, shape, name in zip(out, shapes, ("albedo", "inc", "mu0")))
    threads = 256
    size = ncol * max(ngpt, int(nlay))
    entry = ("rrtmgp_prepare_sw_inputs_fp64" if mu.dtype == xp.float64 else
             "rrtmgp_prepare_sw_inputs")
    get_kernel("rrtmgp_cloud", entry)(
        ((size + threads - 1) // threads,), (threads,),
        (albedo, solar, mu, *outputs, np.int32(ncol), np.int32(ngpt),
         np.int32(nlay)))
    return outputs


#: Named per-chunk scratch buffers.  `_prepare_above_model_chunk` runs 2364
#: times an hour and allocated ~15 fresh arrays each time; the profile puts
#: it at 2.78 s of HOST against 0.98 s of device, the worst ratio in
#: radiation. Each entry is keyed by device and stream, then NAME, shape and
#: dtype, which is what makes reuse safe: two temporaries that are live at
#: the same moment ask under different names and can never be handed the
#: same bytes.  Bounded by construction -- fifteen names times the handful
#: of chunk widths a domain tree produces (full width plus one ragged tail
#: per domain), a few MiB in total, and it REPLACES pool allocations rather
#: than adding to them.
_CHUNK_SCRATCH: dict = {}


def release_rrtmgp_stream_scratch(*, device_id, stream):
    """Retire the completed queue's chunk temporaries, retaining tables."""
    import cupy as cp
    device_id = int(device_id)
    if int(cp.cuda.runtime.getDevice()) != device_id:
        raise ValueError("RRTMGP scratch release requires the owning CUDA device")
    stream.synchronize()
    keys = [key for key in tuple(_CHUNK_SCRATCH)
            if key[0] == ("cupy", device_id) and key[1] == int(stream.ptr)]
    for key in keys:
        del _CHUNK_SCRATCH[key]
    return {"chunk_scratch": len(keys)}


def _chunk_scratch(name: str, shape, *, xp, dtype=DTYPE):
    """``(buffer, first_use)`` for one named per-chunk temporary.

    ``first_use`` lets a caller skip re-establishing bytes that its own
    previous call already wrote and nothing else can have touched -- see
    the clear-layer tail in :func:`_append_clear_upper_layers`.
    """
    shape = tuple(int(extent) for extent in shape)
    key = (_cache_device_key(xp),
           xp.cuda.get_current_stream().ptr if hasattr(xp, "cuda") else None,
           name, shape, np.dtype(dtype).str)
    buffer = _CHUNK_SCRATCH.get(key)
    if buffer is None:
        buffer = xp.empty(shape, dtype=dtype)
        _CHUNK_SCRATCH[key] = buffer
        return buffer, True
    return buffer, False


def _append_clear_upper_layers(value, upper_nlay: int, *, xp=None,
                               scratch: str | None = None):
    """Append zero-valued cloud/path layers to a model-layer field.

    ``scratch`` names a reused buffer (:func:`_chunk_scratch`).  The clear
    tail is then written ONCE: this call site is the only writer of that
    name, so on every later call the zeros it wrote the first time are
    still there.  Same bytes, one fewer fill kernel and one fewer
    allocation per call, five times per chunk per band.
    """
    if xp is None:
        import cupy as xp
    value = xp.ascontiguousarray(xp.asarray(value, dtype=DTYPE))
    if value.ndim != 2:
        raise ValueError("clear-layer input must have shape (ncol,nlay)")
    upper_nlay = int(upper_nlay)
    if upper_nlay < 0:
        raise ValueError("upper_nlay must be nonnegative")
    if upper_nlay == 0:
        return value
    # One allocation and one copy: `zeros` + `concatenate` allocated the
    # clear block only to copy it straight into a second array, and this
    # runs five times per chunk per band.
    model_nlay = value.shape[1]
    shape = (value.shape[0], model_nlay + upper_nlay)
    if scratch is None:
        out, clear_tail = xp.empty(shape, dtype=DTYPE), True
    else:
        out, clear_tail = _chunk_scratch(scratch, shape, xp=xp)
    out[:, :model_nlay] = value
    if clear_tail:
        out[:, model_nlay:] = DTYPE(0.0)
    return out


def _clear_upper_fields(values, upper_nlay, *, xp, names=None):
    """Copy all five fields and append clear tails with one launch."""
    if not hasattr(xp, "RawModule") or upper_nlay == 0:
        return tuple(_append_clear_upper_layers(
            value, upper_nlay, xp=xp,
            scratch=None if names is None else names[i])
            for i, value in enumerate(values))
    from woof.core.kernels import get_kernel

    values = tuple(xp.ascontiguousarray(xp.asarray(v, dtype=DTYPE))
                   for v in values)
    if len(values) != 5 or any(v.ndim != 2 or v.shape != values[0].shape
                               for v in values):
        raise ValueError("clear-layer fields must share a 2-D shape")
    if upper_nlay < 0:
        raise ValueError("upper_nlay must be nonnegative")
    ncol, model_nlay = values[0].shape
    shape = (ncol, model_nlay + upper_nlay)
    outputs = tuple(xp.empty(shape, dtype=DTYPE) if names is None else
                    _chunk_scratch(names[i], shape, xp=xp)[0]
                    for i in range(5))
    threads = 256
    size = ncol * shape[1]
    get_kernel("rrtmgp_cloud", "rrtmgp_clear_upper_layers")(
        ((size + threads - 1) // threads,), (threads,),
        (*values, *outputs, np.int32(ncol), np.int32(model_nlay),
         np.int32(shape[1])))
    return outputs


def _array(value, dtype):
    """Materialize a masked/NetCDF value as a C-contiguous plain array."""
    return np.ascontiguousarray(np.asarray(value, dtype=dtype))


def _packed_variable(variable, dimensions, dtype):
    """Pack a NetCDF variable in a kernel layout selected by dimension name."""
    source = tuple(variable.dimensions)
    target = tuple(dimensions)
    if len(source) != len(target) or set(source) != set(target):
        raise ValueError(
            f"{getattr(variable, 'name', 'variable')} dimensions {source} "
            f"do not match required dimensions {target}")
    permutation = tuple(source.index(name) for name in target)
    return _array(np.transpose(variable[:], permutation), dtype)


def _strings(variable) -> tuple[str, ...]:
    """A NetCDF character array's strings along its last dimension.

    What ``netCDF4.chartostring`` returned here: each row's bytes as text,
    trailing NULs dropped, then stripped and lowered.
    """
    values = np.asarray(variable[:])
    if values.dtype.kind != "S" or values.dtype.itemsize != 1 or values.ndim != 2:
        raise ValueError(
            f"{getattr(variable, 'name', 'variable')}: expected a 2-D NetCDF "
            f"character array, got {values.dtype} {values.shape}")
    return tuple(row.tobytes().decode("utf-8").rstrip("\x00").strip().lower()
                 for row in np.ascontiguousarray(values))


def _scalar(variable) -> float:
    """A scalar NetCDF variable's value (netCDF4's ``getValue()``)."""
    values = np.asarray(variable[...], dtype=np.float64)
    if values.size != 1:
        raise ValueError(
            f"{getattr(variable, 'name', 'variable')}: expected a scalar, "
            f"got shape {values.shape}")
    return float(values.reshape(()))


def _make_flavors(key_species: np.ndarray,
                  band_lims: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Transcribe ``create_flavor``/``create_gpoint_flavor`` (0-based).

    Gas slot zero is dry air.  Upstream rewrites the special ``(0,0)`` pair
    to ``(2,2)`` because those coefficients are identically zero.
    """
    pairs: list[tuple[int, int]] = []
    for iband in range(key_species.shape[0]):
        for iatm in range(2):
            pair = tuple(int(x) for x in key_species[iband, iatm])
            if pair == (0, 0):
                pair = (2, 2)
            if pair not in pairs:
                pairs.append(pair)
    flavors = _array(pairs, np.int32)
    gpoint_flavor = np.empty((2, int(band_lims[-1, 1]) + 1), np.int32)
    lookup = {pair: i for i, pair in enumerate(pairs)}
    for iband, (start, end) in enumerate(band_lims):
        for iatm in range(2):
            pair = tuple(int(x) for x in key_species[iband, iatm])
            if pair == (0, 0):
                pair = (2, 2)
            gpoint_flavor[iatm, start:end + 1] = lookup[pair]
    return flavors, np.ascontiguousarray(gpoint_flavor)


def _minor_gas_indices(gas_names: tuple[str, ...], gas_minor,
                       identifier_minor, minor_identifiers) -> np.ndarray:
    identifier_map = {name: i for i, name in enumerate(identifier_minor)}
    gas_map = {name: i + 1 for i, name in enumerate(gas_names)}
    return _array([gas_map[gas_minor[identifier_map[name]]]
                   for name in minor_identifiers], np.int32)


def _scaling_gas_indices(gas_names: tuple[str, ...], names) -> np.ndarray:
    gas_map = {name: i + 1 for i, name in enumerate(gas_names)}
    return _array([gas_map.get(name, -1) for name in names], np.int32)


#: One upload of a table set per card, ever.  ``GasTables.to_device`` and
#: ``CloudTables.to_device`` used to test ``dev not in self._device`` and
#: upload outside any lock, so slabs of one domain stepping on their own
#: streams at once each uploaded a copy at the first radiation call and the
#: last one to finish replaced the others in the cache.  A slab that had
#: picked up a replaced copy kept launching kernels that read it while the
#: copy's last reference died: its blocks went back to the memory pool of
#: the stream that allocated them, which handed them to that stream's next
#: allocation while the other slab's kernels were still reading.  MEASURED
#: (one RTX 4090 split into 2x2 slabs, the default suite): 4 of 5 runs
#: wrote frames differing from the unsplit
#: run, first in the shortwave heating of the first radiation call; with
#: every radiation call serialized across slabs, 3 of 3 were identical, and
#: with this lock alone 5 of 5 (6 of 6 beside the Thompson, legacy-LW and
#: RUC uploads made the same way).
_TABLE_UPLOAD_LOCK = threading.RLock()


def _upload_once(owner, upload):
    """``owner._device[dev]``, uploaded once under the lock, ordered on use.

    The upload's event orders a caller on another stream after the bytes
    land (the ``device_cache.cached_ready`` contract); the lock makes the
    first upload the only one, so no published copy is ever replaced or
    freed while a slab still reads it.
    """
    import cupy as cp

    dev = cp.cuda.runtime.getDevice()
    with _TABLE_UPLOAD_LOCK:
        cached = owner._device.get(dev)
        if cached is None:
            values = upload(cp)
            ready = cp.cuda.Event()
            ready.record()
            values["_upload_ready"] = ready
            values["_upload_stream"] = cp.cuda.get_current_stream().ptr
            cached = owner._device[dev] = DeviceTables(values)
    current = cp.cuda.get_current_stream()
    if current.ptr != cached._upload_stream and not cached._upload_ready.done:
        current.wait_event(cached._upload_ready)
    return cached


@dataclass
class DeviceTables:
    """Namespace holding cached device arrays plus scalar metadata."""

    _values: dict[str, object]

    def __getattr__(self, name):
        try:
            return self._values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


@dataclass
class GasOpticsResult:
    """Gas extinction optical depth and optional SW scattering properties."""

    tau: object
    ssa: object | None = None
    g: object | None = None
    col_dry: object | None = None


@dataclass
class FluxResult:
    flux_up: object
    flux_dn: object
    flux_dir: object | None = None


@dataclass
class PlanckSourceResult:
    lay_source: object
    lev_source: object
    sfc_source: object


@dataclass
class _InterpolationMetadata:
    """Driver-owned pressure/temperature indices and FP32 fractions.

    Instances are created for one ``RRTMGPRadiation.__call__`` and are never
    accepted by the exported gas/source entry points.  Keeping the object
    private makes its table/input provenance and one-call lifetime explicit.
    """

    iatm: object
    jt: object
    jp: object
    ftemp: object
    fpress: object

    def __getitem__(self, key):
        return _InterpolationMetadata(
            self.iatm[key], self.jt[key], self.jp[key],
            self.ftemp[key], self.fpress[key])


@dataclass
class CloudOpticsResult:
    """Band-resolved cloud extinction and scattering properties."""

    tau: object
    ssa: object
    g: object


@dataclass
class HydrometeorPaths:
    """Condensate paths (g m-2) and RRTMGP particle sizes (microns)."""

    clwp: object
    ciwp: object
    reliq: object
    dgice: object


@dataclass(frozen=True)
class _RadiationColumnChunk:
    """One solver chunk's synthetic upper atmosphere and interpolation."""

    profile: _RadiationColumnProfile
    paths: HydrometeorPaths
    cldfra: object
    metadata: _InterpolationMetadata
    solver_profile: _RadiationColumnProfile | None = None


def _solver_temperature_profile(profile, *, xp):
    """Keep the retained profile and share its two FP32 consumer casts."""
    if not hasattr(xp, "RawModule") or profile.tlay.dtype == xp.float32:
        return profile
    from woof.core.kernels import get_kernel

    if profile.tlay.dtype != xp.float64 or profile.tlev.dtype != xp.float64:
        raise TypeError("solver temperature profile must use float32 or float64")
    tlay = xp.empty(profile.tlay.shape, dtype=DTYPE)
    tlev = xp.empty(profile.tlev.shape, dtype=DTYPE)
    ncol, nlay = tlay.shape
    threads = 256
    get_kernel("rrtmgp_cloud", "rrtmgp_solver_temperatures")(
        ((ncol * (nlay + 1) + threads - 1) // threads,), (threads,),
        (profile.tlay, profile.tlev, tlay, tlev, np.int32(ncol), np.int32(nlay)))
    return _RadiationColumnProfile(
        profile.play, profile.plev, tlay, tlev, profile.qv,
        profile.model_nlay, profile.upper_nlay)


def _prepare_above_model_chunk(
        *, tables, play, plev, tlay, tlev, qv, paths, cldfra, columns,
        p_top: float, kind: str,
        pressure_floor: float = RRTMGP_TOA_PRESSURE_PA, xp=None,
        validate_top: bool = True, validate: bool = True,
        uniform_top: bool = False, scratch: bool = False,
) -> _RadiationColumnChunk:
    """Build every above-model temporary for exactly ``columns``.

    The model-layer columns remain full-domain inputs, but the synthetic
    thermodynamic cap, clear cloud/path layers, and gas-table interpolation
    coordinates are deliberately materialized only for the active solver
    chunk.  In particular, an irregular final slice retains its true column
    count rather than allocating or exposing a padded ``column_chunk`` tail.

    ``scratch`` RETURNS VIEWS OF REUSED BUFFERS, and that puts the caller
    under a contract: finish with the returned chunk before the next call
    of the same ``kind`` and shape, because that call overwrites it.  The
    driver's chunk loop satisfies this by construction -- it consumes each
    chunk and drops it inside one iteration -- and it is the only caller
    that opts in.  OFF BY DEFAULT, so a caller that collects chunks and
    compares them afterwards still gets independent arrays; the test that
    does exactly that is what established the contract needed stating.
    """
    if xp is None:
        import cupy as xp

    profile = _extend_above_model_profile(
        play[columns], plev[columns], tlay[columns], tlev[columns],
        qv[columns], p_top=p_top, kind=kind,
        pressure_floor=pressure_floor, xp=xp, validate_top=validate_top,
        uniform_top=uniform_top)
    # Distinct names: all five are live at once, so they must never share
    # a buffer.  `kind` keeps the LW and SW passes apart even where their
    # shapes coincide.
    clear_values = (paths.clwp[columns], paths.ciwp[columns],
                    paths.reliq[columns], paths.dgice[columns], cldfra[columns])
    clear_names = tuple(f"clear.{kind}.{name}" for name in
                        ("clwp", "ciwp", "reliq", "dgice", "cldfra"))
    clear_outputs = _clear_upper_fields(
        clear_values, profile.upper_nlay, xp=xp,
        names=clear_names if scratch else None)
    chunk_paths = HydrometeorPaths(*clear_outputs[:4])
    chunk_cldfra = clear_outputs[4]
    solver_profile = _solver_temperature_profile(profile, xp=xp)
    metadata = _interpolation_metadata(
        tables, solver_profile.play, solver_profile.tlay, validate=validate,
        scratch=(f"meta.{kind}" if scratch else None))
    return _RadiationColumnChunk(
        profile=profile, paths=chunk_paths, cldfra=chunk_cldfra,
        metadata=metadata, solver_profile=solver_profile)


# Shared-workspace counterpart to SCRATCH_SLOT_LIFETIME_AUDIT.  Each value
# names the operation that completely writes the slot after every reuse and
# before its first consumer.  RTE "carried" entries are identical-offset views
# of values fully produced in the same chunk's immediately preceding optics
# phase; no value is carried across chunks or domains.
RRTMGP_WORKSPACE_LIFETIME_AUDIT = {
    "lw_optics": {
        "gas_tau": "rrtmgp_gas_optics kernel",
        "vmr": "full zero fill plus complete active-gas assignment",
        "cld_tau": "rrtmgp_cloud_optics kernel",
        "cld_ssa": "rrtmgp_cloud_optics kernel",
        "cld_asy": "rrtmgp_cloud_optics kernel",
        "col_dry": "reserved unread slot; no producer required",
        "mcica_mask": "rrtmgp_mcica_maxran kernel",
    },
    "lw_rte": {
        # Only what the phase READS BACK.  With the finalize fused into the
        # solver, the carried set is what the solver combines in registers:
        # gas_tau, the two band cloud cubes it reads, and the McICA mask.
        # cld_asy and col_dry are dead here; the RTE outputs lie over them.
        "gas_tau": "same-chunk lw_optics producer at identical offset",
        "vmr": "same-chunk lw_optics producer at identical offset",
        "cld_tau": "same-chunk lw_optics producer at identical offset",
        "cld_ssa": "same-chunk lw_optics producer at identical offset",
        "mcica_mask": "same-chunk lw_optics producer at identical offset",
        # lay_source/lev_source/sfc_source are GONE: the solver derives
        # them from play/tlay/tlev/tsfc/vmr + metadata, which are the
        # carried slots and the caller's own arrays.
        "emiss_gpt": "complete band-expansion assignment",
        "incident": "full zero fill",
        "flux_up": "rrtmgp_lw_noscat kernel",
        "flux_dn": "rrtmgp_lw_noscat kernel",
    },
    "sw_optics": {
        "gas_tau": "rrtmgp_gas_optics kernel",
        "gas_ssa": "rrtmgp_gas_optics kernel",
        "vmr": "full zero fill plus complete active-gas assignment",
        "cld_tau": "rrtmgp_cloud_optics kernel",
        "cld_ssa": "rrtmgp_cloud_optics kernel",
        "cld_asy": "rrtmgp_cloud_optics kernel",
        "col_dry": "reserved unread slot; no producer required",
        "mcica_mask": "rrtmgp_mcica_maxran kernel",
    },
    "sw_rte": {
        # As in lw_rte, the carried set is what the fused solver reads:
        # both gas cubes, all three band cloud cubes, and the mask.  vmr
        # and col_dry are dead here -- SW builds no Planck source.
        "gas_tau": "same-chunk sw_optics producer at identical offset",
        "gas_ssa": "same-chunk sw_optics producer at identical offset",
        "cld_tau": "same-chunk sw_optics producer at identical offset",
        "cld_ssa": "same-chunk sw_optics producer at identical offset",
        "cld_asy": "same-chunk sw_optics producer at identical offset",
        "mcica_mask": "same-chunk sw_optics producer at identical offset",
        "albedo_gpt": "complete surface broadcast assignment",
        "inc_gpt": "complete solar broadcast assignment",
        "mu0": "complete cosine broadcast assignment",
        "flux_up": "rrtmgp_sw_2stream kernel",
        "flux_dn": "rrtmgp_sw_2stream kernel",
        "flux_dir": "rrtmgp_sw_2stream kernel",
    },
}


_VALIDATION_MESSAGES = (
    (1 << 0, "play is non-finite or outside the gas-table pressure range"),
    (1 << 1, "plev is non-finite or negative"),
    (1 << 2, "tlay is non-finite or outside the gas-table temperature range"),
    (1 << 3, "tlev is non-finite or outside the LW temperature range"),
    (1 << 4, "tsfc is non-finite or outside the LW temperature range"),
    (1 << 5, "qv is non-finite or negative"),
    (1 << 6, "qc is non-finite or negative"),
    (1 << 7, "qr is non-finite or negative"),
    (1 << 8, "qi is non-finite or negative"),
    (1 << 9, "qs is non-finite or negative"),
    (1 << 10, "cldfra is non-finite or outside [0, 1]"),
    (1 << 11, "nc is non-finite or negative"),
    (1 << 12, "nr is non-finite or negative"),
    (1 << 13, "ni is non-finite or negative"),
    (1 << 14, "ns is non-finite or negative"),
    (1 << 15, "effc is non-finite or negative"),
    (1 << 16, "effr is non-finite or negative"),
    (1 << 17, "effi is non-finite or negative"),
    (1 << 18, "effs is non-finite or negative"),
    (1 << 19, "surface emissivity is non-finite or outside [0, 1]"),
    (1 << 20, "the bottom four layer pressures are not bottom-to-top"),
    (1 << 21, "pressure thickness or Exner is non-positive"),
    (1 << 22, "effc is outside the physical-plausibility band "
              "(microns contract; radii writer unit defect?)"),
    (1 << 23, "effi is outside the physical-plausibility band "
              "(microns contract; radii writer unit defect?)"),
    (1 << 24, "effs is outside the physical-plausibility band "
              "(microns contract; radii writer unit defect?)"),
)


@dataclass
class RFMIPResult:
    lw_up: object
    lw_dn: object
    sw_up: object
    sw_dn: object


from woof.core.trace_gases import (
    RFMIP_GAS_NAMES as _RFMIP_GAS_NAMES, validate_trace_gas_overrides)



# NOAA Global Monitoring Laboratory, "Globally averaged marine surface
# annual mean CO2", dry-air mole fraction in ppm.  Pinned 2026-07-16 from
# https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_annmean_gl.txt
# (file creation 2026-07-05; DOI https://doi.org/10.15138/9N0H-ZH07).
# The source begins in 1979.  Dates outside the pinned range hold the nearest
# published annual mean unless the case declares an override.  Runtime
# performs no network access.
_NOAA_GML_CO2_ANNUAL_PPM = {
    1979: 336.85, 1980: 338.91, 1981: 340.11, 1982: 340.85,
    1983: 342.53, 1984: 344.07, 1985: 345.54, 1986: 346.97,
    1987: 348.68, 1988: 351.16, 1989: 352.79, 1990: 354.06,
    1991: 355.40, 1992: 356.09, 1993: 356.84, 1994: 358.33,
    1995: 360.18, 1996: 361.93, 1997: 363.04, 1998: 365.70,
    1999: 367.80, 2000: 368.96, 2001: 370.57, 2002: 372.58,
    2003: 375.14, 2004: 376.95, 2005: 378.98, 2006: 381.15,
    2007: 382.90, 2008: 385.02, 2009: 386.50, 2010: 388.75,
    2011: 390.62, 2012: 392.65, 2013: 395.40, 2014: 397.34,
    2015: 399.65, 2016: 403.07, 2017: 405.22, 2018: 407.61,
    2019: 410.07, 2020: 412.44, 2021: 414.70, 2022: 417.08,
    2023: 419.35, 2024: 422.79, 2025: 425.64,
}


def trace_gases(valid_date: date | datetime,
                override: Mapping[str, float] | None = None
                ) -> dict[str, float]:
    """Select date-indexed well-mixed gas VMRs, then apply case overrides.

    NOAA's annual global CO2 mean is selected by calendar year, holding the
    earliest/latest published value outside the table range.  Overrides are
    mole fractions and win over the dated selection.
    """
    if not isinstance(valid_date, (date, datetime)):
        raise TypeError("trace-gas selection date must be a date or datetime")
    years = tuple(_NOAA_GML_CO2_ANNUAL_PPM)
    selected_year = max((year for year in years if year <= valid_date.year),
                        default=min(years))
    selected = {
        "co2": _NOAA_GML_CO2_ANNUAL_PPM[selected_year] * 1.0e-6,
    }
    selected.update(validate_trace_gas_overrides(override))
    return selected


def _minor_gpoint_csr(limits: np.ndarray, ngpt: int
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Invert ``minor_limits_gpt`` into a per-g-point entry list.

    ``limits`` gives each minor entry an INCLUSIVE g-point range; the kernel
    wants the transpose -- given a g-point, which entries apply.  Returned as
    CSR: ``start[g]:start[g + 1]`` slices ``index``, and the entries inside a
    slice are ascending in ``m``, which is the order the scanning loop
    visited them in.  That ordering is the bit-exactness argument: the same
    additions accumulate into ``tau_abs`` in the same sequence.
    """
    limits = np.asarray(limits, dtype=np.int32)
    if limits.ndim != 2 or limits.shape[1] != 2:
        raise ValueError("minor g-point limits must have shape (nminor,2)")
    ngpt = int(ngpt)
    counts = np.zeros(ngpt + 1, dtype=np.int64)
    for lower, upper in limits:
        if not (0 <= lower <= upper < ngpt):
            raise ValueError(
                f"minor g-point range ({lower},{upper}) outside 0..{ngpt - 1}")
        counts[lower + 1:upper + 2] += 1
    start = np.cumsum(counts).astype(np.int32)
    index = np.empty(int(start[-1]), dtype=np.int32)
    cursor = start[:-1].copy()
    for m, (lower, upper) in enumerate(limits):
        for gpt in range(int(lower), int(upper) + 1):
            index[cursor[gpt]] = m
            cursor[gpt] += 1
    # An empty list would still need a valid pointer to launch with.
    if index.size == 0:
        index = np.zeros(1, dtype=np.int32)
    return np.ascontiguousarray(start), np.ascontiguousarray(index)


@dataclass
class GasTables:
    kind: str
    gas_names: tuple[str, ...]
    gas_index: Mapping[str, int]
    nband: int
    ngpt: int
    ntemp: int
    npres: int
    neta: int
    press_ref: np.ndarray
    temp_ref: np.ndarray
    press_ref_trop: float
    vmr_ref: np.ndarray
    band_lims_gpt: np.ndarray
    gpoint_bands: np.ndarray
    flavor: np.ndarray
    gpoint_flavor: np.ndarray
    kmajor: np.ndarray
    kminor_lower: np.ndarray
    kminor_upper: np.ndarray
    minor_limits_gpt_lower: np.ndarray
    minor_limits_gpt_upper: np.ndarray
    minor_scales_with_density_lower: np.ndarray
    minor_scales_with_density_upper: np.ndarray
    scale_by_complement_lower: np.ndarray
    scale_by_complement_upper: np.ndarray
    idx_minor_lower: np.ndarray
    idx_minor_upper: np.ndarray
    idx_minor_scaling_lower: np.ndarray
    idx_minor_scaling_upper: np.ndarray
    kminor_start_lower: np.ndarray
    kminor_start_upper: np.ndarray
    #: Derived in __post_init__, never read from the netCDF: for each
    #: atmosphere half, the minor entries that cover each g-point, as CSR
    #: (``start[ngpt + 1]`` offsets into ``list``).  See the kernel's minor
    #: loop -- these are the 6% of (g-point, entry) pairs that are not a
    #: skipped iteration.  Declared as fields so ``to_device`` uploads them
    #: and ``packed_arrays`` counts them like every other table.
    minor_gpt_start_lower: np.ndarray = None
    minor_gpt_list_lower: np.ndarray = None
    minor_gpt_start_upper: np.ndarray = None
    minor_gpt_list_upper: np.ndarray = None
    rayleigh: np.ndarray | None = None
    planck_fraction: np.ndarray | None = None
    temperature_planck: np.ndarray | None = None
    totplnk: np.ndarray | None = None
    solar_source: np.ndarray | None = None
    tsi_default: float | None = None
    optimal_angle_fit: np.ndarray | None = None
    #: PER CUDA DEVICE, not per process.  ``load_gas_tables`` is
    #: ``lru_cache``d on the kind alone, so one ``GasTables`` object serves
    #: the whole process; memoizing ONE ``DeviceTables`` on it handed every
    #: device the pointers of whichever device uploaded first.  MEASURED on a
    #: dual-4090 box: with device 0 touched first, ``gas_lw.kmajor`` lives on
    #: device 0 and a physics step on device 1 dies with
    #: CUDA_ERROR_ILLEGAL_ADDRESS inside RRTMGP; touch device 1 first and the
    #: failure moves to device 0.  Consumer Ada has no P2P to paper over it,
    #: and an illegal address destroys the CUDA context for the whole
    #: process, so one such step takes every later run down with it.
    #: The tables are duplicated per card -- see ``preflight
    #: .k_distribution_bytes``, which counts them once per PROCESS and is
    #: therefore a per-device figure that must be multiplied by the number of
    #: cards a run actually uses.
    _device: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        for half in ("lower", "upper"):
            if getattr(self, f"minor_gpt_start_{half}") is None:
                start, index = _minor_gpoint_csr(
                    getattr(self, f"minor_limits_gpt_{half}"), self.ngpt)
                object.__setattr__(self, f"minor_gpt_start_{half}", start)
                object.__setattr__(self, f"minor_gpt_list_{half}", index)

    @property
    def ngas(self) -> int:
        return len(self.gas_names)

    @property
    def nflav(self) -> int:
        return self.flavor.shape[0]

    def packed_arrays(self) -> dict[str, np.ndarray]:
        return {name: value for name, value in vars(self).items()
                if isinstance(value, np.ndarray)}

    def to_device(self) -> DeviceTables:
        """The k-distribution on the CURRENT device, uploaded once per card
        (:func:`_upload_once`)."""
        def upload(cp):
            values: dict[str, object] = {}
            for name, value in vars(self).items():
                if isinstance(value, np.ndarray):
                    dtype = (cp.float32 if np.issubdtype(value.dtype,
                                                         np.floating)
                             else cp.int32 if np.issubdtype(value.dtype,
                                                            np.integer)
                             else cp.bool_)
                    values[name] = cp.ascontiguousarray(cp.asarray(value,
                                                                   dtype=dtype))
                elif not name.startswith("_"):
                    values[name] = value
            return values
        return _upload_once(self, upload)


@dataclass
class CloudTables:
    kind: str
    nband: int
    nsize_liq: int
    nsize_ice: int
    nrghice: int
    radliq_lwr: float
    radliq_upr: float
    diamice_lwr: float
    diamice_upr: float
    extliq: np.ndarray
    ssaliq: np.ndarray
    asyliq: np.ndarray
    extice: np.ndarray
    ssaice: np.ndarray
    asyice: np.ndarray
    #: PER CUDA DEVICE -- see GasTables._device for the measurement.
    _device: dict = field(default_factory=dict, init=False, repr=False)

    @property
    def liq_step_size(self) -> float:
        return (self.radliq_upr - self.radliq_lwr) / (self.nsize_liq - 1)

    @property
    def ice_step_size(self) -> float:
        return (self.diamice_upr - self.diamice_lwr) / (self.nsize_ice - 1)

    def to_device(self) -> DeviceTables:
        """The cloud optics tables on the CURRENT device, once per card
        (:func:`_upload_once`)."""
        def upload(cp):
            return {name: (cp.ascontiguousarray(cp.asarray(value,
                                                             dtype=cp.float32))
                           if isinstance(value, np.ndarray) else value)
                    for name, value in vars(self).items()
                    if not name.startswith("_")}
        return _upload_once(self, upload)


@lru_cache(maxsize=1)
def packaged_table_facts() -> dict | None:
    """Sizes, shapes and gas names of the four tables, from the packaged receipt.

    Planning (the domain wizard, run-plan, check, tile memory, composition
    checks) needs these facts, not the tables' values.  Since 2.8.8 the
    tables decode through rw_netcdf, so decoding them to plan made every
    plan refuse on an install without the native decoder (measured: the
    native-free Windows cpu job failed the wizard with NetcdfBridgeMissing).
    The receipt (woof/data/rrtmgp_k_distribution_bytes.json) is bound to
    each table's sha256; any other bytes return None and callers decode.
    """
    import hashlib
    import json

    receipt_path = Path(__file__).resolve().parents[1] / "data" / "rrtmgp_k_distribution_bytes.json"
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        for name, digest in receipt["tables"].items():
            if hashlib.sha256(Path(_table(name)).read_bytes()).hexdigest() != digest:
                return None
        return receipt
    except (OSError, KeyError, ValueError):
        return None


def gas_table_shape(kind: str) -> SimpleNamespace:
    """``ngpt``, ``ngas`` and ``nband`` of one gas table, without decoding it
    when the packaged receipt matches (see :func:`packaged_table_facts`)."""
    if kind not in ("lw", "sw"):
        raise ValueError("kind must be 'lw' or 'sw'")
    facts = packaged_table_facts()
    if facts is not None and "meta" in facts:
        meta = facts["meta"]
        return SimpleNamespace(ngpt=int(meta[f"ngpt_{kind}"]),
                               ngas=int(meta[f"ngas_{kind}"]),
                               nband=int(meta[f"nband_{kind}"]))
    tables = load_gas_tables(kind)
    return SimpleNamespace(ngpt=tables.ngpt, ngas=tables.ngas, nband=tables.nband)


@lru_cache(maxsize=2)
def coefficient_gas_names(kind: str) -> tuple[str, ...]:
    """Read selected absorption operands without packing the full tables."""
    if kind not in ("lw", "sw"):
        raise ValueError("kind must be 'lw' or 'sw'")
    facts = packaged_table_facts()
    if facts is not None and kind in facts.get("gas_names", {}):
        return tuple(facts["gas_names"][kind])
    filename = ("rrtmgp-gas-lw-g256.nc" if kind == "lw"
                else "rrtmgp-gas-sw-g224.nc")
    with _open_table(_table(filename)) as nc:
        return _strings(nc.variables["gas_names"])


@lru_cache(maxsize=2)
def gas_table_temperature_range_k(kind: str) -> tuple[float, float]:
    """``(lowest, highest)`` of one gas table's ``temp_ref``, read alone.

    Every RRTMGP call refuses a layer temperature outside this span
    (``_validate_host_range`` and the fused validation kernel both read
    it from the loaded tables).  A caller that can see a temperature
    before any integration, such as the initial-state perturbation,
    reads the same two numbers here without packing the tables.
    """
    if kind not in ("lw", "sw"):
        raise ValueError("kind must be 'lw' or 'sw'")
    filename = ("rrtmgp-gas-lw-g256.nc" if kind == "lw"
                else "rrtmgp-gas-sw-g224.nc")
    with _open_table(_table(filename)) as nc:
        temp_ref = _array(nc.variables["temp_ref"][:], np.float64)
    return float(np.min(temp_ref)), float(np.max(temp_ref))


@lru_cache(maxsize=2)
def load_gas_tables(kind: str) -> GasTables:
    """Load and pack the v1.9 LW or SW gas k-distribution in float64."""
    kind = kind.lower()
    if kind not in ("lw", "sw"):
        raise ValueError("kind must be 'lw' or 'sw'")
    filename = ("rrtmgp-gas-lw-g256.nc" if kind == "lw"
                else "rrtmgp-gas-sw-g224.nc")
    with _open_table(_table(filename)) as nc:
        gas_names = _strings(nc.variables["gas_names"])
        gas_minor = _strings(nc.variables["gas_minor"])
        identifier_minor = _strings(nc.variables["identifier_minor"])
        band_lims = _array(nc.variables["bnd_limits_gpt"][:] - 1, np.int32)
        key_species = _array(nc.variables["key_species"][:], np.int32)
        flavor, gpoint_flavor = _make_flavors(key_species, band_lims)
        gpoint_bands = np.empty(len(nc.dimensions["gpt"]), np.int32)
        for iband, (start, end) in enumerate(band_lims):
            gpoint_bands[start:end + 1] = iband

        lower_names = _strings(nc.variables["minor_gases_lower"])
        upper_names = _strings(nc.variables["minor_gases_upper"])
        scaling_lower = _strings(nc.variables["scaling_gas_lower"])
        scaling_upper = _strings(nc.variables["scaling_gas_upper"])
        kwargs = dict(
            kind=kind,
            gas_names=gas_names,
            gas_index={name: i + 1 for i, name in enumerate(gas_names)},
            nband=len(nc.dimensions["bnd"]),
            ngpt=len(nc.dimensions["gpt"]),
            ntemp=len(nc.dimensions["temperature"]),
            npres=len(nc.dimensions["pressure"]),
            neta=len(nc.dimensions["mixing_fraction"]),
            press_ref=_array(nc.variables["press_ref"][:], np.float64),
            temp_ref=_array(nc.variables["temp_ref"][:], np.float64),
            press_ref_trop=_scalar(nc.variables["press_ref_trop"]),
            # NetCDF dimension order is (temperature, absorber, atmosphere).
            # The numerical kernels use (atmosphere, absorber, temperature).
            vmr_ref=_array(np.transpose(nc.variables["vmr_ref"][:], (2, 1, 0)),
                           np.float64),
            band_lims_gpt=band_lims,
            gpoint_bands=np.ascontiguousarray(gpoint_bands),
            flavor=flavor,
            gpoint_flavor=gpoint_flavor,
            # Kernel layouts are selected from declared NetCDF dimension
            # names, never from an assumed positional file order.
            kmajor=_packed_variable(
                nc.variables["kmajor"],
                ("temperature", "mixing_fraction", "pressure_interp", "gpt"),
                np.float64),
            kminor_lower=_packed_variable(
                nc.variables["kminor_lower"],
                ("temperature", "mixing_fraction", "contributors_lower"),
                np.float64),
            kminor_upper=_packed_variable(
                nc.variables["kminor_upper"],
                ("temperature", "mixing_fraction", "contributors_upper"),
                np.float64),
            minor_limits_gpt_lower=_array(
                nc.variables["minor_limits_gpt_lower"][:] - 1, np.int32),
            minor_limits_gpt_upper=_array(
                nc.variables["minor_limits_gpt_upper"][:] - 1, np.int32),
            minor_scales_with_density_lower=_array(
                nc.variables["minor_scales_with_density_lower"][:], bool),
            minor_scales_with_density_upper=_array(
                nc.variables["minor_scales_with_density_upper"][:], bool),
            scale_by_complement_lower=_array(
                nc.variables["scale_by_complement_lower"][:], bool),
            scale_by_complement_upper=_array(
                nc.variables["scale_by_complement_upper"][:], bool),
            idx_minor_lower=_minor_gas_indices(
                gas_names, gas_minor, identifier_minor, lower_names),
            idx_minor_upper=_minor_gas_indices(
                gas_names, gas_minor, identifier_minor, upper_names),
            idx_minor_scaling_lower=_scaling_gas_indices(
                gas_names, scaling_lower),
            idx_minor_scaling_upper=_scaling_gas_indices(
                gas_names, scaling_upper),
            kminor_start_lower=_array(
                nc.variables["kminor_start_lower"][:] - 1, np.int32),
            kminor_start_upper=_array(
                nc.variables["kminor_start_upper"][:] - 1, np.int32),
        )
        if kind == "lw":
            kwargs.update(
                planck_fraction=_packed_variable(
                    nc.variables["plank_fraction"],
                    ("temperature", "mixing_fraction", "pressure_interp",
                     "gpt"), np.float64),
                temperature_planck=_array(nc.variables["temperature_Planck"][:],
                                          np.float64),
                totplnk=_array(np.transpose(nc.variables["totplnk"][:]), np.float64),
                optimal_angle_fit=_array(nc.variables["optimal_angle_fit"][:],
                                         np.float64),
            )
        else:
            rayleigh = np.stack((nc.variables["rayl_lower"][:],
                                 nc.variables["rayl_upper"][:]), axis=0)
            from woof.wrf_exact import ENABLED
            if ENABLED:
                quiet = _array(nc.variables["solar_source_quiet"][:], np.float32)
                facular = _array(nc.variables["solar_source_facular"][:], np.float32)
                sunspot = _array(nc.variables["solar_source_sunspot"][:], np.float32)
                mg = np.float32(_scalar(nc.variables["mg_default"]))
                sb = np.float32(_scalar(nc.variables["sb_default"]))
                solar = np.asarray(quiet + np.float32(mg - np.float32(0.1495954)) * facular, np.float32)
                solar = np.asarray(solar + np.float32(sb - np.float32(0.00066696)) * sunspot, np.float32)
            else:
                quiet = _array(nc.variables["solar_source_quiet"][:], np.float64)
                facular = _array(nc.variables["solar_source_facular"][:], np.float64)
                sunspot = _array(nc.variables["solar_source_sunspot"][:], np.float64)
                mg = _scalar(nc.variables["mg_default"])
                sb = _scalar(nc.variables["sb_default"])
                solar = quiet + (mg - 0.1495954) * facular \
                    + (sb - 0.00066696) * sunspot
            kwargs.update(
                rayleigh=_array(rayleigh, np.float64),
                solar_source=_array(solar, np.float64),
                tsi_default=_scalar(nc.variables["tsi_default"]),
            )
    return GasTables(**kwargs)


from woof.core.rfmip_upstream import (  # noqa: E402
    TRACE_CLIMATOLOGY_SHA256, TRACE_CLIMATOLOGY_SOURCE)


@dataclass(frozen=True)
class TraceClimatology:
    trace_vmr: dict
    pressure_layer_pa: np.ndarray
    ozone_vmr: np.ndarray


@lru_cache(maxsize=1)
def load_trace_climatology() -> TraceClimatology:
    """The RFMIP-derived trace-gas and ozone climatology, pin-verified."""
    import hashlib
    import json

    raw = _table("rrtmgp-trace-gas-climatology.json").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != TRACE_CLIMATOLOGY_SHA256:
        raise ValueError(
            f"rrtmgp/rrtmgp-trace-gas-climatology.json has sha256 {digest}, "
            f"pinned {TRACE_CLIMATOLOGY_SHA256}: the radiation would run on "
            "a climatology nobody derived; reinstall recast-woof-data")
    table = json.loads(raw)
    if (table["source"]["sha256"] != TRACE_CLIMATOLOGY_SOURCE["sha256"]
            or table["source"]["bytes"] != TRACE_CLIMATOLOGY_SOURCE["bytes"]):
        raise ValueError("trace-gas climatology names a different source file")
    return TraceClimatology(
        trace_vmr={gas: float(value)
                   for gas, value in table["trace_vmr"].items()},
        pressure_layer_pa=np.asarray(table["pressure_layer_pa"], np.float64),
        ozone_vmr=np.asarray(table["ozone_vmr"], np.float64))


@lru_cache(maxsize=2)
def load_cloud_tables(kind: str) -> CloudTables:
    """Load band-resolved v1.9 liquid/ice cloud-optics tables."""
    kind = kind.lower()
    if kind not in ("lw", "sw"):
        raise ValueError("kind must be 'lw' or 'sw'")
    with _open_table(_table(f"rrtmgp-clouds-{kind}-bnd.nc")) as nc:
        return CloudTables(
            kind=kind,
            nband=len(nc.dimensions["nband"]),
            nsize_liq=len(nc.dimensions["nsize_liq"]),
            nsize_ice=len(nc.dimensions["nsize_ice"]),
            nrghice=len(nc.dimensions["nrghice"]),
            radliq_lwr=_scalar(nc.variables["radliq_lwr"]),
            radliq_upr=_scalar(nc.variables["radliq_upr"]),
            diamice_lwr=_scalar(nc.variables["diamice_lwr"]),
            diamice_upr=_scalar(nc.variables["diamice_upr"]),
            extliq=_array(np.transpose(nc.variables["extliq"][:]), np.float64),
            ssaliq=_array(np.transpose(nc.variables["ssaliq"][:]), np.float64),
            asyliq=_array(np.transpose(nc.variables["asyliq"][:]), np.float64),
            extice=_array(np.transpose(nc.variables["extice"][:], (2, 1, 0)),
                          np.float64),
            ssaice=_array(np.transpose(nc.variables["ssaice"][:], (2, 1, 0)),
                          np.float64),
            asyice=_array(np.transpose(nc.variables["asyice"][:], (2, 1, 0)),
                          np.float64),
        )


def hydrometeor_paths(plev, qc, qr=None, qi=None, qs=None, *,
                      microphysics="kessler", play=None, tlay=None,
                      nc=None, nr=None, ni=None, ns=None,
                      effc=None, effr=None, effi=None,
                      effs=None, cldfra=None,
                      snow_treatment=SNOW_TREATMENT_FULL_MASS,
                      radii_written=None,
                      validate=True) -> HydrometeorPaths:
    """Convert hydrometeor mixing ratios to cloud-optics inputs on device.

    Mixing ratios and Morrison number concentrations are per kg dry air.
    Per WRF the radiation liquid path is cloud water only and the ice path
    is cloud ice plus snow; rain never feeds the paths
    (module_ra_rrtmg_sw.F:11029-11034, module_ra_rrtmg_lw.F:12488-12493:
    ``gliqwp = qc1d(k) * pdel*100/gravmks*1000``).  When ``cldfra`` is
    given the grid-box paths become in-cloud paths through WRF's
    ``max(0.01, cldfrac)`` division (same lines).
    Kessler uses a 10 micron liquid radius and 50 micron ice diameter.
    WSM6, Thompson, and NSSL consume their scheme-native cloud/ice/snow
    effective radii (MICRONS, the state contract) and merge ice plus snow
    into the single RRTMGP ice species.  ``"thompson"`` serves BOTH Thompson
    packages: mp_physics=8 (Registry.EM_COMMON:3024) and the aerosol-aware
    mp_physics=28 (:3036) declare the same ``moist`` inventory and the same
    ``re_cloud/re_ice/re_snow`` state, and module_physics_init.F:1005-1006
    lists them together in one ``has_req*`` disjunction, so their radiative
    coupling is identical -- see :data:`_MP_CLOUD_OPTICS_SCHEME`.
    ``snow_treatment`` selects how snow joins that path:
    ``full-snow-mass-into-ice`` keeps the adapter's original
    full-mass merge; ``wrf-rrtmg-130um-snow-discount`` reproduces WRF
    v4.6.1's option-4 explicit-radius coupling -- ice path from cloud ice
    only (module_ra_rrtmg_lw.F:12500-12505, _sw.F:11040-11045), snow mass
    multiplied by ``MIN(0.99, (130/re_s)^2)`` with ``re_s`` floored at 10
    and capped at 130 microns (_lw.F:12242,12515-12532, _sw.F:10824,
    11055-11067; the FP32 discount expression is bitwise-pinned by
    tests/data/wrf_rrtmg_snow_discount_fixture.csv).  WRF's dead
    ``gicewp`` accumulation of the 1% remainder (_lw.F:12518, _sw.F:11058
    -- computed but never stored to ``cicewp``) is intentionally not
    reproduced.  The mass-weighted single-species diameter remains a
    documented adapter divergence from WRF's separate Fu snow species.
    ``snow_treatment`` is validated for every scheme but only alters the
    explicit-snow-radius (WSM6/Thompson/NSSL) coupling: WRF discounts snow
    only when the scheme supplies re_snow (iceflg=5); Morrison and Kessler
    keep WRF's merged path (_lw.F:12488-12493).
    ``"p3"`` is mp_physics=50 and takes neither of those: WRF remaps P3's
    single ice category ONTO the snow species and empties the ice path
    (_lw.F:12250-12261, _sw.F:10851-10863), so the branch consumes effc and
    effi, refuses an effs, and applies WRF's iceflg=5 discount at P3's own
    ice radius unconditionally -- see the branch for why the compatibility
    token does not gate it.
    Morrison liquid size is the cloud-droplet gamma radius alone -- rain
    carries no radiative mass, so it contributes no radius either
    (``effr`` is accepted for interface parity with Morrison's
    diagnostics and ignored); ice/snow combine by number before clipping
    to the shipped RRTMGP table domains.
    ``radii_written`` (Morrison only, one bool per column) marks the
    columns the microphysics ran on and so wrote ``effc/effi/effs`` and
    its fixed droplet number.  The specified-zone ring it never runs on
    (WRF's clipped tiles, woof.core.microphysics) keeps the allocation
    background there (2.5/5/10 um, zero droplets); those columns take the
    same radii from their own moments instead, with Morrison's fixed
    250 cm-3 droplet number.  Without it every column is taken as written.
    ``"milbrandt2"`` is mp_physics=9 and is shaped like Morrison's branch
    with the scheme's own PSD: the radii are WRF's commented-out
    Milbrandt-Yau block (module_mp_milbrandt2mom.F:3351-3378) evaluated
    from ``nc``/``ni``/``ns`` -- ``0.664639/lambda_c`` for the alpha_c=1,
    mu_c=3 cloud distribution and ``1.5/lambda`` for the exponential ice
    and snow, with the scheme's ``icex`` constants and its ``iLAMmin2``
    floors -- then the same number-weighted ice/snow merge and clips.  It
    takes no ``effc/effr/effi/effs``: the scheme never writes them, so
    ``state.eff*`` for mp=9 hold their allocation-time background and a
    caller passing them would radiate a constant.  ``play``/``tlay`` are
    accepted for interface parity with Morrison and not needed (air
    density cancels between a per-kg mass and a per-kg number).
    """
    import cupy as cp

    if snow_treatment not in SNOW_TREATMENTS:
        raise ValueError(
            f"snow_treatment must be one of {SNOW_TREATMENTS}, got "
            f"{snow_treatment!r}")

    plev = cp.ascontiguousarray(cp.asarray(plev, dtype=DTYPE))
    qc = cp.ascontiguousarray(cp.asarray(qc, dtype=DTYPE))
    if qc.ndim != 2 or plev.shape != (qc.shape[0], qc.shape[1] + 1):
        raise ValueError("plev/qc must have shapes (ncol,nlay+1)/(ncol,nlay)")

    def field(value, name):
        if value is None:
            return cp.zeros_like(qc)
        return _device_profile(value, qc.shape, name)

    qr, qi, qs = field(qr, "qr"), field(qi, "qi"), field(qs, "qs")
    if validate:
        if bool(cp.any(~cp.isfinite(plev))):
            raise ValueError("hydrometeor pressure inputs must be finite")
        _require_finite_nonnegative(qc=qc, qr=qr, qi=qi, qs=qs)
    mass_path = (cp.abs(cp.diff(plev, axis=1))
                 * DTYPE(1000.0 / 9.80665))
    clwp = qc * mass_path
    ciwp = (qi + qs) * mass_path
    if cldfra is not None:
        cldfra = _device_profile(cldfra, qc.shape, "cldfra")
        if validate and (bool(cp.any(~cp.isfinite(cldfra)))
                         or bool(cp.any(cldfra < 0.0))
                         or bool(cp.any(cldfra > 1.0))):
            raise ValueError("cldfra must be finite and within [0, 1]")
        incloud = cp.maximum(DTYPE(0.01), cldfra)
        clwp = clwp / incloud
        ciwp = ciwp / incloud
    clwp = cp.ascontiguousarray(clwp)
    ciwp = cp.ascontiguousarray(ciwp)
    scheme = str(microphysics).lower()
    if scheme in ("kessler", "ferrier"):
        return HydrometeorPaths(
            clwp, ciwp, cp.full_like(qc, DTYPE(10.0)),
            cp.full_like(qc, DTYPE(50.0)))
    if scheme in ("wsm3", "wsm6", "thompson", "nssl"):
        if any(x is None for x in (effc, effi, effs)):
            raise ValueError(
                f"{scheme} radii require effc, effi, and effs")
        re_c, re_i, re_s = (field(value, name) for value, name in
                            ((effc, "effc"), (effi, "effi"),
                             (effs, "effs")))
        if validate:
            _require_finite_nonnegative(effc=re_c, effi=re_i, effs=re_s)
            _require_plausible_radii_um(effc=re_c, effi=re_i, effs=re_s)
        tiny = DTYPE(1.0e-20)
        if snow_treatment == SNOW_TREATMENT_WRF_DISCOUNT:
            # WRF v4.6.1 option-4 explicit-snow-radius coupling, FP32 in
            # WRF's operation order (fixture-pinned, max_ulp 0):
            #   resnow = MAX(10., re_s)            (_lw.F:12242, _sw.F:10824;
            #                                       state is already microns)
            #   factor = 0.99; if resnow > 130:
            #       factor = MIN(0.99, (130/resnow)*(130/resnow))
            #       resnow = 130                   (_lw.F:12515-12528)
            #   snow path mass = qs * factor       (_lw.F:12529)
            # and the ice path is cloud ice only (_lw.F:12500-12505).
            re_s0 = cp.maximum(DTYPE(10.0), re_s)
            quotient = DTYPE(130.0) / re_s0
            factor = cp.where(
                re_s0 > DTYPE(130.0),
                cp.minimum(DTYPE(0.99), quotient * quotient),
                DTYPE(0.99))
            re_s_eff = cp.minimum(re_s0, DTYPE(130.0))
            qs_eff = qs * factor
            ciwp = (qi + qs_eff) * mass_path
            if cldfra is not None:
                ciwp = ciwp / incloud
            ciwp = cp.ascontiguousarray(ciwp)
        else:
            qs_eff = qs
            re_s_eff = re_s
        frozen = qi + qs_eff
        reice = cp.where(
            frozen > tiny,
            (qi * re_i + qs_eff * re_s_eff) / cp.maximum(frozen, tiny),
            DTYPE(25.0))
        reliq = cp.where(qc > tiny, re_c, DTYPE(10.0))
        return HydrometeorPaths(
            clwp, ciwp,
            cp.ascontiguousarray(cp.clip(reliq, DTYPE(2.5), DTYPE(21.5))),
            cp.ascontiguousarray(cp.clip(DTYPE(2.0) * reice,
                                         DTYPE(10.0), DTYPE(180.0))))
    if scheme == "p3":
        # WRF's own P3 remap, transcribed: module_ra_rrtmg_lw.F:12250-12261
        # and module_ra_rrtmg_sw.F:10851-10863 (same lines in v4.6.1 and
        # v4.7.1).  Under has_reqs == 0 with has_reqc and has_reqi set --
        # which module_physics_init.F:1022-1024 then :1033 makes exactly
        # the P3/Jensen-Ishmael case -- both wrappers do
        #     inflg = iceflg = 5
        #     resnow1D = MAX(10., re_ice*1.E6)
        #     QS1D = QI3D ;  QI1D = 0. ;  reice1D = 10.
        # so P3's single ice category is radiated as the SNOW species at
        # P3's own ice radius and the cloud-ice path is emptied.  The
        # iceflg == 5 snow discount then applies to it (_lw.F:12515-12532,
        # _sw.F:11055-11067) exactly as it does for WSM6/Thompson/NSSL.
        #
        # WHY THE DISCOUNT IS UNCONDITIONAL HERE, unlike the branch above.
        # There, ``snow_treatment`` exists to keep already-issued receipts
        # on the behaviour they were issued under.  mp=50 has no issued
        # receipts -- it had no cloud-optics row at all until this one --
        # and WRF reaches iceflg = 5 for P3 through the remap regardless of
        # any compatibility choice, so the WRF answer is the only answer
        # and is the default.  ``snow_treatment`` is still validated at the
        # top of this function; it simply does not select anything here.
        if any(x is None for x in (effc, effi)):
            raise ValueError("p3 radii require effc and effi")
        if effs is not None:
            raise ValueError(
                "p3 supplies no snow effective radius: mp_physics=50 "
                "declares state:re_cloud,re_ice and no re_snow "
                "(Registry.EM_COMMON:3043) because its one ice category "
                "spans the snow-to-graupel continuum through rime mass "
                "and rime volume. Passing effs here would radiate a "
                "radius the scheme never computed")
        re_c, re_i = (field(value, name) for value, name in
                      ((effc, "effc"), (effi, "effi")))
        if validate:
            _require_finite_nonnegative(effc=re_c, effi=re_i)
            _require_plausible_radii_um(
                bands=effective_radius_bands("p3"), effc=re_c, effi=re_i)
            if bool(cp.any(qs > DTYPE(0.0))):
                raise ValueError(
                    "p3 was handed a nonzero snow mixing ratio; "
                    "mp_physics=50 declares moist:qv,qc,qr,qi "
                    "(Registry.EM_COMMON:3043) and allocates no qs, so a "
                    "nonzero one means the caller mixed schemes")
        tiny = DTYPE(1.0e-20)
        # resnow1D = MAX(10., re_ice) -- the state contract is microns, so
        # WRF's *1.E6 has already happened.
        re_s = cp.maximum(DTYPE(10.0), re_i)
        quotient = DTYPE(130.0) / re_s
        factor = cp.where(
            re_s > DTYPE(130.0),
            cp.minimum(DTYPE(0.99), quotient * quotient),
            DTYPE(0.99))
        re_s_eff = cp.minimum(re_s, DTYPE(130.0))
        # QS1D = QI3D, QI1D = 0: the frozen path is the discounted P3 ice
        # mass alone.  WRF's dead ``gicewp`` accumulation of the 1%
        # remainder (_lw.F:12518, _sw.F:11058 -- computed, never stored to
        # ``cicewp``) is not reproduced, the same call the branch above
        # makes; and here WRF's cicewp is a literal zero, because
        # iceflg >= 4 rebuilds it from QI1D (_lw.F:12500-12505) which the
        # remap set to zero.
        qs_eff = qi * factor
        ciwp = qs_eff * mass_path
        if cldfra is not None:
            ciwp = ciwp / incloud
        ciwp = cp.ascontiguousarray(ciwp)
        # reice1D = 10. rides an empty ice path, so the merged
        # single-species radius is the snow radius wherever there is any
        # frozen mass; 25 um is this adapter's clear-sky placeholder, as in
        # the explicit-radius branch above.
        reice = cp.where(qs_eff > tiny, re_s_eff, DTYPE(25.0))
        reliq = cp.where(qc > tiny, re_c, DTYPE(10.0))
        return HydrometeorPaths(
            clwp, ciwp,
            cp.ascontiguousarray(cp.clip(reliq, DTYPE(2.5), DTYPE(21.5))),
            cp.ascontiguousarray(cp.clip(DTYPE(2.0) * reice,
                                         DTYPE(10.0), DTYPE(180.0))))
    if scheme == "milbrandt2":
        # Milbrandt-Yau's own radii, transcribed from the block WRF ships
        # commented out (module_mp_milbrandt2mom.F:3351-3378) and fed by
        # the six number moments woof transports.  See the constants'
        # comment block for the moment algebra; the branch is shaped like
        # Morrison's below because the RADIATIVE side is the same design
        # -- a two-moment scheme's PSD radii merged by number into the one
        # RRTMGP ice species -- and only the PSD differs.
        if any(x is None for x in (nc, ni, ns)):
            raise ValueError(
                "milbrandt2 radii require nc, ni and ns (the scheme's "
                "transported number moments)")
        if any(value is not None for value in (effc, effr, effi, effs)):
            raise ValueError(
                "milbrandt2 supplies no effective radii: mp_physics=9's "
                "own reff block is commented out "
                "(module_mp_milbrandt2mom.F:3351-3378) and nothing in "
                "woof writes state.effc/effi/effs for it, so a value "
                "passed here is the allocation-time background, not a "
                "radius the scheme computed; the radii are derived from "
                "nc/ni/ns instead")
        from woof.core.milbrandt2_constants import CONSTANTS as _MY2

        ncp, nip, nsp = field(nc, "nc"), field(ni, "ni"), field(ns, "ns")
        if validate:
            _require_finite_nonnegative(nc=ncp, ni=nip, ns=nsp)
            if play is not None or tlay is not None:
                for name, value in (("play", play), ("tlay", tlay)):
                    if value is not None and bool(cp.any(~cp.isfinite(
                            _device_profile(value, qc.shape, name)))):
                        raise ValueError(
                            "Milbrandt-Yau thermodynamic inputs must be "
                            "finite")
        tiny = DTYPE(1.0e-20)
        eps_q = DTYPE(MILBRANDT2_EPS_Q)
        eps_n = DTYPE(MILBRANDT2_EPS_N)
        ilam_min = DTYPE(MILBRANDT2_ILAMBDA_MIN_M)
        to_um = DTYPE(1.0e6)

        def inverse_lambda(qmass, number, icex, inverse_dm):
            # iLAMDA_x(DE, Q, 1/N, icex, 1/dm) = (DE Q icex / N)^(1/dm)
            # with N per m^3; per-kg numbers cancel DE.  Evaluated where
            # the species is active; elsewhere the value is masked out.
            ratio = qmass * DTYPE(icex) / cp.maximum(number, tiny)
            return cp.power(cp.maximum(ratio, tiny), DTYPE(inverse_dm))

        active_c = (qc > eps_q) & (ncp > eps_n)
        active_i = (qi > eps_q) & (nip > eps_n)
        active_s = (qs > eps_q) & (nsp > eps_n)
        ilam_c = inverse_lambda(qc, ncp, _MY2["icexc9"], 1.0 / 3.0)
        ilam_i = cp.maximum(
            ilam_min, inverse_lambda(qi, nip, _MY2["icexi9"], 1.0 / 3.0))
        ilam_s = cp.maximum(
            ilam_min, inverse_lambda(qs, nsp, _MY2["iGS20"],
                                     float(_MY2["idms"])))
        re_c = DTYPE(MILBRANDT2_CLOUD_RADIUS_PER_ILAMBDA) * ilam_c * to_um
        re_i = DTYPE(MILBRANDT2_ICE_RADIUS_PER_ILAMBDA) * ilam_i * to_um
        re_s = DTYPE(MILBRANDT2_SNOW_RADIUS_PER_ILAMBDA) * ilam_s * to_um
        wi = cp.where(active_i, nip, DTYPE(0.0))
        ws = cp.where(active_s, nsp, DTYPE(0.0))
        reliq = cp.where(active_c, re_c, DTYPE(10.0))
        reice = cp.where(wi + ws > 0, (wi * re_i + ws * re_s)
                         / cp.maximum(wi + ws, tiny), DTYPE(25.0))
        return HydrometeorPaths(
            clwp, ciwp,
            cp.ascontiguousarray(cp.clip(reliq, DTYPE(2.5), DTYPE(21.5))),
            cp.ascontiguousarray(cp.clip(DTYPE(2.0) * reice,
                                         DTYPE(10.0), DTYPE(180.0))))
    if scheme != "morrison":
        raise ValueError(
            "microphysics must be 'kessler', 'wsm6', 'thompson', 'nssl', "
            "'p3', 'milbrandt2', or 'morrison'")
    if play is None or tlay is None or any(x is None for x in (nc, nr, ni, ns)):
        raise ValueError("Morrison radii require play, tlay, nc, nr, ni, ns")
    play = _device_profile(play, qc.shape, "play")
    tlay = _device_profile(tlay, qc.shape, "tlay")
    ncp, nrp = field(nc, "nc"), field(nr, "nr")
    nip, nsp = field(ni, "ni"), field(ns, "ns")
    if validate and (bool(cp.any(~cp.isfinite(play)))
                     or bool(cp.any(~cp.isfinite(tlay)))):
        raise ValueError("Morrison thermodynamic inputs must be finite")
    if validate:
        _require_finite_nonnegative(nc=ncp, nr=nrp, ni=nip, ns=nsp)
    tiny = DTYPE(1.0e-20)

    rho_air = play / (DTYPE(287.15) * tlay)
    written = None
    if radii_written is not None and effc is not None:
        written = cp.asarray(radii_written, dtype=cp.bool_).reshape(-1, 1)
        if written.shape[0] != qc.shape[0]:
            raise ValueError("radii_written must have one value per column")
        # The ring holds a zero droplet number only because the scheme
        # never ran there; Morrison's INUM=1 number is this constant in
        # every column it does run on (kernels/morrison.cu, WRF NDCNST).
        ncp = cp.where(written, ncp,
                       DTYPE(MORRISON_FIXED_DROPLET_NUMBER_M3) / rho_air)
    shape = DTYPE(0.0005714) * (ncp / DTYPE(1.0e6) * rho_air) \
        + DTYPE(0.2714)
    pgam = cp.clip(DTYPE(1.0) / (shape * shape) - DTYPE(1.0),
                   DTYPE(2.0), DTYPE(10.0))
    gamma_ratio = (pgam + DTYPE(1.0)) * (pgam + DTYPE(2.0)) \
        * (pgam + DTYPE(3.0))
    lam_c = cp.power(DTYPE(np.pi * 997.0 / 6.0) * ncp * gamma_ratio
                     / cp.maximum(qc, tiny), DTYPE(1.0 / 3.0))
    re_c = (pgam + DTYPE(3.0)) / cp.maximum(DTYPE(2.0) * lam_c, tiny) \
        * DTYPE(1.0e6)

    def exponential_radius(qmass, number, density):
        lam = cp.power(DTYPE(np.pi * density) * number
                       / cp.maximum(qmass, tiny), DTYPE(1.0 / 3.0))
        return DTYPE(1.5e6) / cp.maximum(lam, tiny)

    gamma_c = re_c
    supplied_effective = (effc, effr, effi, effs)
    if any(value is not None for value in supplied_effective):
        if any(value is None for value in supplied_effective):
            raise ValueError("Morrison effective radii require effc/effr/"
                             "effi/effs together")
        re_c, re_r, re_i, re_s = (
            field(value, name) for value, name in zip(
                supplied_effective, ("effc", "effr", "effi", "effs")))
        if validate:
            _require_finite_nonnegative(
                effc=re_c, effr=re_r, effi=re_i, effs=re_s)
            # effr is interface parity only (ignored input, not gated).
            _require_plausible_radii_um(effc=re_c, effi=re_i, effs=re_s)
        if written is not None:
            re_c = cp.where(written, re_c, gamma_c)
            re_i = cp.where(written, re_i, exponential_radius(qi, nip, 500.0))
            re_s = cp.where(written, re_s, exponential_radius(qs, nsp, 100.0))
    else:
        re_i = exponential_radius(qi, nip, 500.0)
        re_s = exponential_radius(qs, nsp, 100.0)
    wc = cp.where((qc > 0) & (ncp > 0), ncp, DTYPE(0.0))
    wi = cp.where((qi > 0) & (nip > 0), nip, DTYPE(0.0))
    ws = cp.where((qs > 0) & (nsp > 0), nsp, DTYPE(0.0))
    reliq = cp.where(wc > 0, re_c, DTYPE(10.0))
    reice = cp.where(wi + ws > 0, (wi * re_i + ws * re_s)
                     / cp.maximum(wi + ws, tiny), DTYPE(25.0))
    return HydrometeorPaths(
        clwp, ciwp,
        cp.ascontiguousarray(cp.clip(reliq, DTYPE(2.5), DTYPE(21.5))),
        cp.ascontiguousarray(cp.clip(DTYPE(2.0) * reice,
                                     DTYPE(10.0), DTYPE(180.0))))


def cal_cldfra2(qc, qi, *, f_qc=True, f_qi=True, out=None):
    """WRF binary radiation CLDFRA through the shared no-FTZ loader.

    Raw FP32 columns feed the same kernel in default and strict mode.
    With neither Registry flag, the driver retains its initialized out.
    """
    import cupy as cp
    from woof.core import rrtmg_legacy_device as dev
    if (not isinstance(qc, cp.ndarray) or not isinstance(qi, cp.ndarray)
            or qc.dtype != cp.float32 or qi.dtype != cp.float32
            or qc.shape != qi.shape):
        raise TypeError("cal_cldfra2 requires matching CuPy float32 columns")
    if out is None:
        out = cp.zeros(qc.shape, dtype=cp.float32)
    if (not isinstance(out, cp.ndarray) or out.dtype != cp.float32
            or out.shape != qc.shape or not out.flags.c_contiguous):
        raise TypeError("cal_cldfra2 out must be contiguous CuPy float32 columns")
    if not f_qc and not f_qi:
        return out
    n = qc.size
    if n:
        dev.gpu_preflight()
        dev._gpu_module(cp.cuda.runtime.getDevice()).get_function("rp_cldfra2")(
            ((n + 127) // 128,), (128,),
            (np.int32(n), np.int32(bool(f_qc)), np.int32(bool(f_qi)),
             cp.ascontiguousarray(qc), cp.ascontiguousarray(qi), out))
    return out


def _host_expf(x):
    """WOOF's own float32 expf over a host array (noahmp_libm, bitwise the
    device slab_expf)."""
    from woof.core.noahmp_libm import expf
    x = np.asarray(x, np.float32)
    out = np.empty(x.shape, np.float32)
    flat = out.reshape(-1)
    for i, v in enumerate(x.reshape(-1)):
        flat[i] = expf(v)
    return out


def _host_powf(x, y):
    """WOOF's own float32 powf over host arrays (noahmp_libm.powf_array)."""
    from woof.core.noahmp_libm import powf_array
    return np.asarray(powf_array(np.asarray(x, np.float32),
                                 np.asarray(y, np.float32)), np.float32)


def cal_cldfra1(qv, qc, qi, qs, tlay, play, *, f_qc=True, f_qi=True,
                f_qs=True, mp_physics=None):
    """WRF icloud=1 Xu-Randall cloud fraction on device (FP32).

    Exact transcription of WRF v4.6.1 ``module_radiation_driver.F``
    ``cal_cldfra1`` (lines 3761-3986), the routine the radiation driver
    calls for the Registry default ``icloud=1`` (driver lines 1320-1332,
    Registry.EM_COMMON:2498).  Saturation follows Murray (1966) with the
    driver's constants (lines 3806-3816, 3861-3865); the liquid/ice
    saturation blend uses the condensate ice weight (line 3945) and the
    fraction is Xu and Randall (1996) with ALPHA0=100, GAMMA=0.49,
    QCLDMIN=1e-12, PEXP=0.25, RHGRID=1.0 plus the -6.9 ARG clamp and the
    0.01 truncation (lines 3950-3979).

    Moisture-set dispatch mirrors the driver flags: ``f_qc and f_qi and
    f_qs`` is the Morrison-class branch (lines 3870-3877, QCLD=QI+QC+QS,
    weight=(QI+QS)/QCLD); ``f_qc and f_qi and not f_qs`` is the branch WRF
    comments "for P3, mp option 50 or 51" (lines 3879-3887, QCLD=QI+QC,
    weight=QI/QCLD), for a package with one ice category and no snow
    species; ``f_qc`` alone is the Kessler-class branch (lines 3891-3899,
    QCLD=QC, 273.15 K phase threshold).  Rain never enters QCLD (lines
    3904-3916).

    ``mp_physics`` 5 or 15 (Ferrier-Aligo, fer_mp_hires and its advected
    sibling) adds WRF's own selector arm at lines 3926-3937, which runs
    AFTER the P3 arm above for the same qc+qi/no-qs package: QCLD is
    QC+QI, the weight QI/QCLD, and below -40 C the weight is 1 (all
    condensate ice at homogeneous freezing, the scheme's T_ICE).
    """
    import cupy as cp

    qv = cp.ascontiguousarray(cp.asarray(qv, dtype=DTYPE))
    if qv.ndim != 2:
        raise ValueError("qv must have shape (ncol,nlay)")
    qc = _device_profile(qc, qv.shape, "qc")
    qi = _device_profile(qi, qv.shape, "qi")
    qs = _device_profile(qs, qv.shape, "qs")
    tlay = _device_profile(tlay, qv.shape, "tlay")
    play = _device_profile(play, qv.shape, "play")
    if not f_qc or (f_qs and not f_qi):
        # Breakage this prevents: f_qs without f_qi is WRF's arm for the
        # OLD Eta Ferrier package (etampnew, mp_physics=95, whose qs holds
        # the total ice) at module_radiation_driver.F:3902-3922, weighted
        # by F_ICE_PHY.  woof admits no package with that species set
        # (95 is refused at admission), so a caller reaching it has built
        # flags no accepted scheme has; any other arm would weight the
        # saturation blend with the wrong ice fraction.  The current
        # Ferrier-Aligo (mp_physics=5) is f_qi without f_qs and is ported
        # below.
        raise NotImplementedError(
            "cal_cldfra1 port carries WRF's three F_QC arms: qc+qi+qs "
            "(module_radiation_driver.F:3870-3877), qc+qi with no snow "
            "species (the P3 arm, :3879-3887, and the Ferrier-Aligo arm "
            ":3926-3937 after it) and qc alone (:3891-3899). "
            f"f_qc={bool(f_qc)}/f_qi={bool(f_qi)}/f_qs={bool(f_qs)} is "
            "WRF's arm for the old Eta Ferrier package (mp_physics=95, "
            "qs carrying total ice) at :3902-3922, whose weight is the "
            "F_ICE_PHY ice fraction; no accepted scheme has that species "
            "set")
    # EXP and ** on REAL(4) are WOOF's own float32 expf and powf
    # (woof.core.noahmp_slab_libm), not CuPy's CUDA builtins: cp.exp and
    # cp.power round differently and moved CLDFRA by up to 2,113 ULP on the
    # legacy RRTMG column oracle (tools/rrtmg_legacy_column_oracle).  Host
    # arrays (the CPU-emulated adapter tests substitute NumPy for CuPy) take
    # the same functions' host twins, bit for bit.
    if type(qv).__module__.split(".")[0] == "cupy":
        from woof.core.noahmp_slab_libm import slab_expf, slab_powf
    else:
        slab_expf, slab_powf = _host_expf, _host_powf
    qcldmin = DTYPE(1.0e-12)
    svpt0 = DTYPE(273.15)
    tc = tlay - svpt0
    esw = DTYPE(1000.0) * DTYPE(0.61078) * slab_expf(
        DTYPE(17.2693882) * tc / (tlay - DTYPE(35.86)))
    esi = DTYPE(1000.0) * DTYPE(0.61078) * slab_expf(
        DTYPE(21.8745584) * tc / (tlay - DTYPE(7.66)))
    ep2 = DTYPE(287.0) / DTYPE(461.6)
    qvsw = ep2 * esw / (play - esw)
    qvsi = ep2 * esi / (play - esi)
    if f_qi and f_qs:
        qcld = qi + qc + qs
        weight = cp.where(qcld < qcldmin, DTYPE(0.0),
                          (qi + qs) / cp.maximum(qcld, qcldmin))
    elif f_qi:
        # module_radiation_driver.F:3879-3887, "for P3, mp option 50 or
        # 51": one ice category, no snow species, so QS never enters QCLD
        # and the ice weight is QI/QCLD.  qs is ignored here rather than
        # added as a zero -- a package without qs has no such array.
        qcld = qi + qc
        weight = cp.where(qcld < qcldmin, DTYPE(0.0),
                          qi / cp.maximum(qcld, qcldmin))
        if mp_physics is not None and int(mp_physics) in (5, 15):
            # module_radiation_driver.F:3926-3937, FER_MP_HIRES and
            # FER_MP_HIRES_ADVECT: QCLD = QC + QI (the same sum), weight
            # QI/QCLD, and weight = 1 where tc < -40.
            weight = cp.where(
                qcld < qcldmin, DTYPE(0.0),
                cp.where(tc < DTYPE(-40.0), DTYPE(1.0), weight))
    else:
        qcld = qc
        weight = cp.where(qcld < qcldmin, DTYPE(0.0),
                          cp.where(tlay > svpt0, DTYPE(0.0), DTYPE(1.0)))
    qvs_weight = (DTYPE(1.0) - weight) * qvsw + weight * qvsi
    rhum = qv / qvs_weight
    subsat = cp.maximum(DTYPE(1.0e-10), qvs_weight - qv)
    arg = cp.maximum(DTYPE(-6.9),
                     DTYPE(-100.0) * qcld / slab_powf(subsat, DTYPE(0.49)))
    fraction = slab_powf(cp.maximum(DTYPE(1.0e-10), rhum), DTYPE(0.25)) \
        * (DTYPE(1.0) - slab_expf(arg))
    fraction = cp.where(fraction < DTYPE(0.01), DTYPE(0.0), fraction)
    return cp.ascontiguousarray(
        cp.where(qcld < qcldmin, DTYPE(0.0),
                 cp.where(rhum >= DTYPE(1.0), DTYPE(1.0), fraction)))


#: The schemes that hand radiation their own radii (WRF's has_reqc/has_reqi
#: table, module_physics_init.F:1004-1033), keyed by cloud-optics coupling.
#: P3 is in it for the liquid radius only: WRF moves P3's ice onto the snow
#: species at P3's own radius (module_ra_rrtmg_sw.F:10851-10863), so the ice
#: rule below never reaches it.
SCHEME_RADII_COUPLINGS = ("wsm3", "wsm6", "thompson", "nssl", "p3")

#: Morrison's fixed cloud-droplet number (INUM=1), per m3: WRF's NDCNST =
#: 250 cm-3, which kernels/morrison.cu writes back as nc = 250e6/rho after
#: every call.  The radiation coupling uses it on the specified-zone ring,
#: whose nc carrier the scheme never writes (see hydrometeor_paths).
MORRISON_FIXED_DROPLET_NUMBER_M3 = 250.0e6

#: A cloudy layer whose liquid radius is at or below this (microns) carries
#: the microphysics' no-cloud background, not a droplet size: WRF's 2.5 um
#: (module_ra_rrtmg_sw.F:10781-10788, module_ra_rrtmg_lw.F:12184-12191), which
#: catches the Thompson/WSM6/WDM6 background of 2.49 um.  It is a radius
#: test and nothing more: NSSL's background (2.51 um) is its smallest
#: computed radius too, so a layer MYNN gave water is found by the merge
#: itself (woof.core.mynn_radiation.mynn_bl_cloud_supplied), never by a
#: radius, and resolved cloud keeps the scheme's own size.
_CLOUDY_BACKGROUND_LIQUID_MAX_UM = 2.5
#: WRF's liquid radius for such a layer: 10.5 um over water, 7.5 um over land
#: (the same two blocks), typical of marine and continental stratocumulus.
CLOUDY_LIQUID_RADIUS_OCEAN_UM = 10.5
CLOUDY_LIQUID_RADIUS_LAND_UM = 7.5
#: WRF's ice threshold: at or below 5 um a cloudy layer takes the
#: Kristjansson-Mitchell temperature table instead (_sw.F:10801-10812,
#: _lw.F:12200-12211), index clamped to 1..75 as there.
_CLOUDY_BACKGROUND_ICE_MAX_UM = 5.0


def _supplied_mask(mask, shape, name):
    import cupy as cp

    mask = cp.asarray(mask)
    if mask.dtype != cp.bool_ or mask.shape != shape:
        raise ValueError(
            f"{name} must be a boolean {shape} array, got {mask.dtype} "
            f"{mask.shape}")
    return mask


def cloudy_background_radii(effc, effi, cldfra, tlay, xland, *, scheme,
                            supplied_liquid=None, supplied_ice=None):
    """The radii WRF's RRTMG coupling hands its optics for a cloudy layer.

    A layer can be cloudy where the microphysics has no cloud of its own:
    MYNN's subgrid cloud (``icloud_bl = 1``) adds QC_BL/QI_BL where the
    resolved condensate is below threshold (:func:`merge_mynn_bl_clouds`),
    and there the scheme's radius is its no-cloud background, 2.49 um for
    liquid and 4.99 um for ice under Thompson.  WRF's RRTMG wrappers never
    radiate those: a cloudy layer at or below 2.5 um liquid takes 10.5 um
    over water and 7.5 um over land, and at or below 5 um ice takes the
    temperature table (module_ra_rrtmg_sw.F:10779-10812, the same block in
    module_ra_rrtmg_lw.F:12183-12211; woof.core.rrtmg_legacy_prep
    transcribes it for the legacy engine).  The RTE+RRTMGP adapter read the
    background radius straight into its cloud optics, which triples a
    liquid layer's optical depth at a given water path.

    ``supplied_liquid``/``supplied_ice`` (from
    :func:`woof.core.mynn_radiation.mynn_bl_cloud_supplied`) mark the
    layers the MYNN merge gave water.  They take the same sizes whatever
    radius the scheme wrote there, because that radius is not a size for
    MYNN's water: WRF's radius tests miss NSSL's background (2.51 um liquid,
    10.01 um ice), a trace of Thompson liquid at its 2.51 um floor and a
    trace of Thompson ice sized above 5 um, and would radiate MYNN's water
    at those.  On the product nest below, the ice case alone reached more
    than 80 percent of land columns from mid-morning.  The legacy engine does the same
    (woof.core.rrtmg_legacy).  A layer MYNN did not supply keeps WRF's
    radius tests alone, so resolved cloud at a scheme's smallest radius
    keeps that radius.

    THE BREAKAGE THIS CLOSES, measured on a 750 m coastal-basin product
    nest (MYNN ``icloud_bl = 1``, Thompson, a hot late-August day at solar
    noon): land-mean surface shortwave 549 W m-2 on RTE+RRTMGP against 701
    on legacy RRTMG (556 against 710 over land with no resolved cloud
    water), the brightest land column 932 against 937, and 2 m temperature
    colder at every one of 13 ASOS stations.  One clear column through both
    engines agrees within 5 W m-2; three overcast subgrid layers of 0.05
    g/kg put 170 W m-2 on the ground through RTE+RRTMGP at the 2.49 um
    background, 477 at the 7.5 um WRF uses, and 479 through legacy RRTMG
    (tests/test_rrtmgp_subgrid_cloud_radii.py).

    ``effc``/``effi``/``cldfra``/``tlay`` are ``(ncol, nlay)`` (microns, 0-1,
    K); ``xland`` is WRF's land/water flag per column (1 land, 2 water),
    any shape of ``ncol`` values; exactly 1.5 is neither, as in WRF.
    ``supplied_*`` are ``(ncol, nlay)`` booleans or ``None``.  Returns new
    ``(effc, effi)`` arrays; ``effi`` is returned as given for P3.
    """
    import cupy as cp
    from woof.core.rrtmg_legacy_prep import _RETAB

    if scheme not in SCHEME_RADII_COUPLINGS:
        raise ValueError(
            f"cloudy_background_radii is WRF's rule for the schemes that "
            f"hand radiation their own radii {SCHEME_RADII_COUPLINGS}; "
            f"{scheme!r} computes its radii in the adapter")
    effc = cp.ascontiguousarray(cp.asarray(effc, dtype=DTYPE))
    if effc.ndim != 2:
        raise ValueError("effc must have shape (ncol,nlay)")
    effi = _device_profile(effi, effc.shape, "effi")
    cldfra = _device_profile(cldfra, effc.shape, "cldfra")
    tlay = _device_profile(tlay, effc.shape, "tlay")
    xland = cp.asarray(xland, dtype=DTYPE).reshape(-1)
    if xland.shape != (effc.shape[0],):
        raise ValueError(
            f"xland must hold one value per column ({effc.shape[0]}), "
            f"got {xland.size}")
    cloudy = cldfra > DTYPE(0.0)
    background = cloudy & (effc <= DTYPE(_CLOUDY_BACKGROUND_LIQUID_MAX_UM))
    if supplied_liquid is not None:
        background |= cloudy & _supplied_mask(
            supplied_liquid, effc.shape, "supplied_liquid")
    water = (xland > DTYPE(1.5))[:, None]
    land = (xland < DTYPE(1.5))[:, None]
    effc = cp.where(background & water, DTYPE(CLOUDY_LIQUID_RADIUS_OCEAN_UM),
                    cp.where(background & land,
                             DTYPE(CLOUDY_LIQUID_RADIUS_LAND_UM), effc))
    if scheme == "p3":
        return cp.ascontiguousarray(effc), effi
    redo = cloudy & (effi <= DTYPE(_CLOUDY_BACKGROUND_ICE_MAX_UM))
    if supplied_ice is not None:
        redo |= cloudy & _supplied_mask(
            supplied_ice, effc.shape, "supplied_ice")
    whole = cp.trunc(tlay)
    index = cp.clip(cp.trunc(tlay - DTYPE(179.0)).astype(cp.int32), 1, 75)
    corr = tlay - whole
    table = cp.asarray(_RETAB, dtype=DTYPE)
    tabulated = (table[index - 1] * (DTYPE(1.0) - corr)
                 + table[index] * corr)
    effi = cp.where(redo, cp.maximum(tabulated, DTYPE(5.0)), effi)
    return cp.ascontiguousarray(effc), cp.ascontiguousarray(effi)


# WRF drives the LW/SW stochastic generators with distinct seed advances
# (module_ra_rrtmg_sw.F:11220-11222, module_ra_rrtmg_lw.F:12687-12689).
MCICA_PERMUTESEED_SW = 1
MCICA_PERMUTESEED_LW = 150


_MCICA_MWC = ((18000, 1179647999), (30903, 2025259007))


def _mcica_xorshift_step_matrix() -> list[int]:
    """The 32 basis images of one ``s2`` xorshift, as a GF(2) matrix."""
    cols = []
    for bit in range(32):
        v = 1 << bit
        v ^= (v << 13) & 0xFFFFFFFF
        v ^= v >> 17
        v ^= (v << 5) & 0xFFFFFFFF
        cols.append(v)
    return cols


def _mcica_gf2_apply(mat, v: int) -> int:
    out = 0
    while v:
        out ^= mat[(v & -v).bit_length() - 1]
        v &= v - 1
    return out


@cuda_cache(maxsize=8, ready=True)
def _mcica_jump_tables(nlay: int, ngpt: int):
    """One composed advance operator per subcolumn ``g``.

    Subcolumn ``g`` begins at stream position ``permuteseed + g*nlay``.  The
    level operators advance ``nlay * 2**j`` steps, and a thread used to walk
    the set bits of its own ``g`` applying up to ``bit_length(ngpt-1)`` of
    them -- with two 64-bit modulos and a 32-XOR matrix apply per level.
    They are all powers of one map, so they commute AND compose exactly, in
    each of the four algebras: affine mod 2**32, GF(2) matrix product, and
    modular multiplication for the two MWC residues.  Folding them here
    leaves the kernel one operator to apply, and lands it on the same state
    the level-by-level walk landed on.

    The fold costs one composition per ``g``, not one per set bit: ``g``'s
    operator is the operator of ``g`` with its lowest set bit cleared,
    composed with that bit's level.
    """
    import cupy as cp

    ngpt = int(ngpt)
    njump = max(1, int(ngpt - 1).bit_length())
    # s1: affine x -> A*x + C (mod 2**32); compose f then g = (Ag*Af, Ag*Cf+Cg).
    a, c = 69069, 1327217885
    A, C = 1, 0
    for _ in range(nlay):
        A, C = (a * A) & 0xFFFFFFFF, (a * C + c) & 0xFFFFFFFF
    lvl1 = []
    for _ in range(njump):
        lvl1.append((A, C))
        A, C = (A * A) & 0xFFFFFFFF, (A * C + C) & 0xFFFFFFFF
    # s2: GF(2) matrix power.
    step = _mcica_xorshift_step_matrix()
    M = [1 << i for i in range(32)]
    for _ in range(nlay):
        M = [_mcica_gf2_apply(step, col) for col in M]
    lvl2 = []
    for _ in range(njump):
        lvl2.append(M)
        M = [_mcica_gf2_apply(M, col) for col in M]
    # s3/s4: MWC state advances by multiplication by b**-1 modulo a*b-1.
    lvl_mwc = []
    for base, modulus in _MCICA_MWC:
        m = pow(pow(65536, -1, modulus), nlay, modulus)
        level = []
        for _ in range(njump):
            level.append(m)
            m = (m * m) % modulus
        lvl_mwc.append(level)

    # Fold, ascending-j exactly as the level walk applied them: g's operator
    # is `rest` (g without its lowest set bit, all of whose levels are
    # higher) applied AFTER that bit's level.
    s1 = np.zeros((ngpt, 2), dtype=np.uint32)
    s1[0] = (1, 0)
    s2 = np.zeros((ngpt, 32), dtype=np.uint32)
    s2[0] = [1 << i for i in range(32)]
    s3 = np.ones(ngpt, dtype=np.uint32)
    s4 = np.ones(ngpt, dtype=np.uint32)
    for g in range(1, ngpt):
        low = g & -g
        j = low.bit_length() - 1
        rest = g ^ low
        a_j, c_j = lvl1[j]
        a_r, c_r = int(s1[rest, 0]), int(s1[rest, 1])
        s1[g] = ((a_r * a_j) & 0xFFFFFFFF, (a_r * c_j + c_r) & 0xFFFFFFFF)
        rest_mat = [int(v) for v in s2[rest]]
        s2[g] = [_mcica_gf2_apply(rest_mat, col) for col in lvl2[j]]
        s3[g] = (int(s3[rest]) * lvl_mwc[0][j]) % _MCICA_MWC[0][1]
        s4[g] = (int(s4[rest]) * lvl_mwc[1][j]) % _MCICA_MWC[1][1]
    u32 = lambda a: cp.asarray(np.ascontiguousarray(a, dtype=np.uint32))
    # All subcolumns start with the same xorshift seed. At each selected
    # bit, neighboring g threads therefore read neighboring table words.
    return njump, u32(s1.ravel()), u32(s2.T.ravel()), u32(s3), u32(s4)


def mcica_cloud_masks(play, cldfra, ngpt, permuteseed, *, validate=True):
    return _mcica_cloud_masks(
        play, cldfra, ngpt, permuteseed, validate=validate, out=None)


def _mcica_cloud_masks(play, cldfra, ngpt, permuteseed, *, validate, out):
    """WRF RRTMG McICA maximum-random subcolumn cloud masks on device.

    Transcribes the kissvec generator, pmid-fraction seeding, cldmin
    floor, icld=2 maximum-random overlap walk (WRF Registry default
    ``cldovrlp=2``, Registry.EM_COMMON:2499), and the ``CDF >= 1-cldf``
    subcolumn decision of WRF v4.6.1 ``module_ra_rrtmg_sw.F``
    (lines 1692-1744, 1778-1813, 1941-1977, 2008-2040) with one
    subcolumn per g-point (line 1476).  Returns a boolean
    ``(ncol,nlay,ngpt)`` mask; ``play`` must be bottom-to-top in Pa,
    matching the generator's seed requirement.
    """
    import cupy as cp
    from woof.core.kernels import get_kernel

    play = cp.ascontiguousarray(cp.asarray(play, dtype=DTYPE))
    if play.ndim != 2 or play.shape[1] < 4:
        raise ValueError("play must have shape (ncol,nlay) with nlay >= 4")
    cldfra = _device_profile(cldfra, play.shape, "cldfra")
    if validate and bool(cp.any(play[:, 0] < play[:, 1])):
        # module_ra_rrtmg_sw.F:1734-1736 stops unless pmid is supplied
        # bottom-to-top.
        raise ValueError(
            "kissvec seeding requires pmid from the bottom four layers")
    ncol, nlay = play.shape
    mask = _workspace_output(
        out, (ncol, nlay, int(ngpt)), "mcica_mask", dtype=cp.bool_)
    _, j1, j2, j3, j4 = _mcica_jump_tables(int(nlay), int(ngpt))
    # One block per column: the block shares `1 - cldfra` across its ngpt
    # subcolumns instead of every thread rebuilding it in FP64.  ngpt is 256
    # (LW) or 224 (SW), both legal block sizes and both whole warps; the
    # kernel strides `g` anyway, so a clamped block stays correct.
    threads = min(int(ngpt), 1024)
    shared = int(nlay) * 8
    get_kernel("rrtmgp_mcica", "rrtmgp_mcica_maxran")(
        (int(ncol),), (threads,),
        (play, cldfra, mask, j1, j2, j3, j4,
         np.int32(ncol), np.int32(nlay),
         np.int32(ngpt), np.int32(permuteseed)),
        shared_mem=shared)
    return mask


def cloud_optics(tables: CloudTables, clwp, ciwp, reliq,
                 dgice) -> CloudOpticsResult:
    return _cloud_optics(tables, clwp, ciwp, reliq, dgice, out=None)


def _cloud_optics(tables: CloudTables, clwp, ciwp, reliq,
                  dgice, *, out) -> CloudOpticsResult:
    """Interpolate v1.9 cloud tables into band-resolved FP32 optics.

    Non-positive water paths are reference-defined clear-sky masks in
    ``mo_cloud_optics_rrtmgp.F90:332-340``; the reference kernel writes zero
    properties for them in ``mo_cloud_optics_rrtmgp_kernels.F90:45-60``.
    """
    import cupy as cp
    from woof.core.kernels import get_kernel

    clwp = cp.ascontiguousarray(cp.asarray(clwp, dtype=DTYPE))
    if clwp.ndim != 2:
        raise ValueError("clwp must have shape (ncol,nlay)")
    ciwp = _device_profile(ciwp, clwp.shape, "ciwp")
    reliq = _device_profile(reliq, clwp.shape, "reliq")
    dgice = _device_profile(dgice, clwp.shape, "dgice")
    d = tables.to_device()
    shape = (*clwp.shape, tables.nband)
    if out is None:
        tau, ssa, asym = (cp.empty(shape, dtype=DTYPE) for _ in range(3))
    else:
        tau, ssa, asym = (
            _workspace_output(value, shape, name)
            for value, name in zip(out, ("cld_tau", "cld_ssa", "cld_asy")))
    n = clwp.size * tables.nband
    threads = 256
    get_kernel("rrtmgp_cloud", "rrtmgp_cloud_optics")(
        ((n + threads - 1) // threads,), (threads,),
        (clwp, ciwp, reliq, dgice, d.extliq, d.ssaliq, d.asyliq,
         d.extice, d.ssaice, d.asyice, tau, ssa, asym,
         np.int32(clwp.size), np.int32(tables.nband),
         np.int32(tables.nsize_liq), np.int32(tables.nsize_ice),
         np.int32(tables.nrghice), DTYPE(tables.radliq_lwr),
         DTYPE(tables.liq_step_size), DTYPE(tables.diamice_lwr),
         DTYPE(tables.ice_step_size)))
    return CloudOpticsResult(tau, ssa, asym)


def add_cloud_optics(tables: GasTables, gas: GasOpticsResult,
                     cloud: CloudOpticsResult,
                     cloud_mask=None) -> GasOpticsResult:
    """Add band cloud properties to g-point gas properties on device.

    ``cloud_mask`` is an optional boolean ``(ncol,nlay,ngpt)`` McICA
    subcolumn mask: cloud properties are applied only to cloudy
    subcolumns, mirroring how WRF's RRTMG solvers consume the stochastic
    ``cldfmc``/``taucmc`` arrays per g-point (module_ra_rrtmg_sw.F:
    1951-1977, module_ra_rrtmg_lw.F:3288).  Without a mask every layer
    with condensate is treated as overcast.
    """
    import cupy as cp

    tau_gas = cp.ascontiguousarray(cp.asarray(gas.tau, dtype=DTYPE))
    if tau_gas.ndim != 3 or tau_gas.shape[2] != tables.ngpt:
        raise ValueError("gas optics do not match the gas table")
    band_shape = (*tau_gas.shape[:2], tables.nband)
    tau_cloud = _device_profile(cloud.tau, band_shape, "cloud.tau")
    ssa_cloud = _device_profile(cloud.ssa, band_shape, "cloud.ssa")
    g_cloud = _device_profile(cloud.g, band_shape, "cloud.g")
    bands = tables.to_device().gpoint_bands
    tc, wc, gc = (x[:, :, bands]
                  for x in (tau_cloud, ssa_cloud, g_cloud))
    if cloud_mask is not None:
        mask = cp.asarray(cloud_mask)
        if mask.shape != tau_gas.shape:
            raise ValueError(
                f"cloud_mask must have shape {tau_gas.shape}, "
                f"got {mask.shape}")
        tc = tc * mask
    if tables.kind == "lw":
        return GasOpticsResult(
            cp.ascontiguousarray(tau_gas + tc * (DTYPE(1.0) - wc)),
            col_dry=gas.col_dry)
    ssa_gas = _device_profile(gas.ssa, tau_gas.shape, "gas.ssa")
    g_gas = _device_profile(gas.g, tau_gas.shape, "gas.g")
    total_tau = tau_gas + tc
    scatter = tau_gas * ssa_gas + tc * wc
    floor = DTYPE(3.0 * np.finfo(np.float32).tiny)
    total_ssa = scatter / cp.maximum(floor, total_tau)
    total_g = (tau_gas * ssa_gas * g_gas + tc * wc * gc) \
        / cp.maximum(floor, scatter)
    return GasOpticsResult(cp.ascontiguousarray(total_tau),
                           cp.ascontiguousarray(total_ssa),
                           cp.ascontiguousarray(total_g), gas.col_dry)


def _finalize_cloud_optics(tables: GasTables, gas: GasOpticsResult,
                           cloud: CloudOpticsResult,
                           cloud_mask=None, *, out=None) -> GasOpticsResult:
    """Expand/add cloud optics and, for SW, delta-scale in one kernel."""
    import cupy as cp
    from woof.core.kernels import get_kernel

    tau_gas = cp.ascontiguousarray(cp.asarray(gas.tau, dtype=DTYPE))
    if tau_gas.ndim != 3 or tau_gas.shape[2] != tables.ngpt:
        raise ValueError("gas optics do not match the gas table")
    band_shape = (*tau_gas.shape[:2], tables.nband)
    tau_cloud = _device_profile(cloud.tau, band_shape, "cloud.tau")
    ssa_cloud = _device_profile(cloud.ssa, band_shape, "cloud.ssa")
    g_cloud = _device_profile(cloud.g, band_shape, "cloud.g")
    if cloud_mask is None:
        mask = tau_gas  # Valid dummy pointer; the kernels do not read it.
        have_mask = False
    else:
        mask = cp.ascontiguousarray(cp.asarray(cloud_mask, dtype=cp.bool_))
        if mask.shape != tau_gas.shape:
            raise ValueError(
                f"cloud_mask must have shape {tau_gas.shape}, "
                f"got {mask.shape}")
        have_mask = True

    n = tau_gas.size
    threads = 256
    launch = ((n + threads - 1) // threads,), (threads,)
    bands = tables.to_device().gpoint_bands
    if tables.kind == "lw":
        tau = (cp.empty_like(tau_gas) if out is None else
               _workspace_output(out[0], tau_gas.shape, "optics_tau"))
        get_kernel("rrtmgp_cloud", "rrtmgp_finalize_cloud_lw")(
            *launch, (tau_gas, tau_cloud, ssa_cloud, bands, mask, tau,
                      np.int32(n), np.int32(tables.ngpt),
                      np.int32(tables.nband), np.int32(have_mask)))
        return GasOpticsResult(tau=tau, col_dry=gas.col_dry)

    if gas.g is not None:
        raise ValueError("fused SW optics require the zero gas-g sentinel")
    ssa_gas = _device_profile(gas.ssa, tau_gas.shape, "gas.ssa")
    if out is None:
        tau, ssa, asym = (cp.empty_like(tau_gas) for _ in range(3))
    else:
        tau, ssa, asym = (
            _workspace_output(value, tau_gas.shape, name)
            for value, name in zip(
                out, ("optics_tau", "optics_ssa", "optics_g")))
    get_kernel("rrtmgp_cloud", "rrtmgp_finalize_cloud_sw")(
        *launch, (tau_gas, ssa_gas, tau_cloud, ssa_cloud, g_cloud,
                  bands, mask, tau, ssa, asym, np.int32(n),
                  np.int32(tables.ngpt), np.int32(tables.nband),
                  np.int32(have_mask)))
    return GasOpticsResult(tau, ssa, asym, gas.col_dry)


def _surface_emissivity_bands(value, tables: GasTables, ny: int, nx: int, *,
                              validate=True):
    """Normalize scalar or band-first surface emissivity to (ncol,nband)."""
    import cupy as cp

    emissivity = cp.asarray(value, dtype=DTYPE)
    ncol = ny * nx
    if emissivity.ndim == 0:
        bands = cp.broadcast_to(emissivity, (ncol, tables.nband))
    elif emissivity.shape == (ny, nx):
        bands = cp.broadcast_to(
            emissivity.reshape(ncol, 1), (ncol, tables.nband))
    elif emissivity.shape == (tables.nband,):
        bands = cp.broadcast_to(emissivity[None, :], (ncol, tables.nband))
    elif emissivity.shape == (tables.nband, ny, nx):
        bands = emissivity.transpose(1, 2, 0).reshape(ncol, tables.nband)
    else:
        raise ValueError(
            "emiss must be scalar, (ny,nx), (nband,), or (nband,ny,nx); "
            f"got {emissivity.shape}")
    if validate and (bool(cp.any(~cp.isfinite(bands)))
                     or bool(cp.any(bands < 0.0))
                     or bool(cp.any(bands > 1.0))):
        raise ValueError("surface emissivity must be finite and within [0, 1]")
    return cp.ascontiguousarray(bands)


def _expand_band_to_gpoint(values, tables: GasTables, name="band_values", *,
                           out=None):
    """Expand band values unchanged over each band's g-points.

    Transcribes ``mo_rte_lw.F90:188-191,266-268,476-496`` at fa107a1.
    Input is ``(ncol,nband)`` after adapting Fortran's ``(nband,ncol)``.
    """
    import cupy as cp

    values = cp.ascontiguousarray(cp.asarray(values, dtype=DTYPE))
    if values.ndim != 2 or values.shape[1] != tables.nband:
        raise ValueError(
            f"{name} must have shape (ncol,{tables.nband}), got {values.shape}")
    if out is None:
        return cp.ascontiguousarray(
            values[:, tables.to_device().gpoint_bands])
    target = _workspace_output(
        out, (values.shape[0], tables.ngpt), name)
    # cupy.take writes the full shared view directly; advanced indexing would
    # allocate a second call-local g-point array and defeat real sharing.
    cp.take(values, tables.to_device().gpoint_bands, axis=1, out=target)
    return target


def _validation_error_messages(flags: int) -> tuple[str, ...]:
    """Decode the production validation bitset without device dependencies."""
    return tuple(message for bit, message in _VALIDATION_MESSAGES if flags & bit)


def _validate_device_call_shapes(*, play, plev, tlay, tlev, tsfc, exner,
                                 qv, qc, qr, qi, qs, cldfra, emiss,
                                 numbers, effective):
    """Reject every extent mismatch before the fused kernel sees a pointer."""
    if len(play.shape) != 2:
        raise ValueError("play must have shape (ncol,nlay)")
    ncol, nlay = play.shape
    if ncol < 1 or nlay < 2:
        raise ValueError(
            "RRTMGP profiles must contain at least one column and two layers")
    cell_shape = (ncol, nlay)
    level_shape = (ncol, nlay + 1)
    expected = {
        "plev": (plev, level_shape), "tlay": (tlay, cell_shape),
        "tlev": (tlev, level_shape), "tsfc": (tsfc, (ncol,)),
        "exner": (exner, cell_shape), "qv": (qv, cell_shape),
        "qc": (qc, cell_shape), "qr": (qr, cell_shape),
        "qi": (qi, cell_shape), "qs": (qs, cell_shape),
        "cldfra": (cldfra, cell_shape),
    }
    expected.update((name, (value, cell_shape))
                    for name, value in numbers.items())
    expected.update((name, (value, cell_shape))
                    for name, value in effective.items())
    for name, (value, shape) in expected.items():
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if len(emiss.shape) != 2 or emiss.shape[0] != ncol:
        raise ValueError(
            f"surface emissivity must have {ncol} columns, got {emiss.shape}")


def _validation_flags_device(*, play, plev, tlay, tlev, tsfc, exner, qv,
                             qc, qr, qi, qs, cldfra, emiss,
                             numbers, effective, tables_lw, tables_sw,
                             radii_bands=None, describe=None):
    """Run the production predicate scan and return its host bitset.

    ``describe`` opts the readback into :mod:`woof.core.health_ledger`; see
    :func:`woof.core.microphysics.validate_surface_diagnostics`.
    """
    _validate_device_call_shapes(
        play=play, plev=plev, tlay=tlay, tlev=tlev, tsfc=tsfc,
        exner=exner, qv=qv, qc=qc, qr=qr, qi=qi, qs=qs,
        cldfra=cldfra, emiss=emiss, numbers=numbers,
        effective=effective)
    import cupy as cp
    from woof.core.kernels import get_kernel

    ncol, nlay = play.shape
    dummy = qc
    morrison = bool(numbers)
    have_effective = bool(effective)

    def optional(fields, name):
        return fields[name] if name in fields else dummy

    flags = cp.zeros((1,), dtype=cp.uint32)
    n = max(play.size, plev.size, emiss.size)
    threads = 256
    play_lower = max(float(np.min(tables_lw.press_ref)),
                     float(np.min(tables_sw.press_ref)))
    play_upper = min(float(np.max(tables_lw.press_ref)),
                     float(np.max(tables_sw.press_ref)))
    temp_lower = max(float(np.min(tables_lw.temp_ref)),
                     float(np.min(tables_sw.temp_ref)))
    temp_upper = min(float(np.max(tables_lw.temp_ref)),
                     float(np.max(tables_sw.temp_ref)))
    if radii_bands is None:
        radii_bands = EFFECTIVE_RADIUS_PLAUSIBLE_UM
    get_kernel("rrtmgp_validation", "rrtmgp_validate_call")(
        ((n + threads - 1) // threads,), (threads,), (
            play, plev, tlay, tlev, tsfc, exner, qv, qc, qr, qi, qs,
            cldfra,
            optional(numbers, "nc"), optional(numbers, "nr"),
            optional(numbers, "ni"), optional(numbers, "ns"),
            optional(effective, "effc"), optional(effective, "effr"),
            optional(effective, "effi"), optional(effective, "effs"),
            emiss, flags, np.int32(ncol), np.int32(nlay),
            np.int32(emiss.size), np.int32(morrison),
            np.int32(have_effective), np.float64(play_lower),
            np.float64(play_upper),
            DTYPE(temp_lower), DTYPE(temp_upper),
            DTYPE(radii_bands["effc"][0]), DTYPE(radii_bands["effc"][1]),
            DTYPE(radii_bands["effi"][0]), DTYPE(radii_bands["effi"][1]),
            DTYPE(radii_bands["effs"][0]), DTYPE(radii_bands["effs"][1])))
    # This is the production path's sole validation synchronization/D2H read
    # -- unless a health ledger is active, in which case it is not a read at
    # all and the drain reports the same bitset later.
    from woof.core import health_ledger

    return health_ledger.read_status(
        flags, site="rrtmgp", describe=describe)


def _validate_device_call(*, diagnose=None, **profiles):
    """Run fused guards, replaying legacy diagnostics only on failure.

    The deferred report drops ``diagnose()``: it replays the legacy
    validators over the LIVE arrays, and by the time a deferred drain runs
    those arrays hold a later step.  A message built from them would name
    the wrong values with complete confidence, which is worse than not
    having it.  The bitset itself is exact either way.
    """
    def _describe(observed: int) -> None:
        from woof.core import health_ledger

        raise ValueError(
            "RRTMGP input validation failed: "
            + "; ".join(_validation_error_messages(observed))
            + health_ledger.deferred_note())

    observed = _validation_flags_device(describe=_describe, **profiles)
    if observed:
        if diagnose is not None:
            diagnose()
        raise ValueError(
            "RRTMGP input validation failed: "
            + "; ".join(_validation_error_messages(observed)))


def _raise_full_call_validation_error(*, play, plev, tlay, tlev, tsfc,
                                      exner, qv, qc, qr, qi, qs, cldfra,
                                      emiss, numbers, effective, tables_lw,
                                      tables_sw, column_chunk,
                                      radii_bands=None):
    """Replay the former validators in their observable failure order."""
    import cupy as cp

    if bool(cp.any(~cp.isfinite(plev))):
        raise ValueError("hydrometeor pressure inputs must be finite")
    _require_finite_nonnegative(qc=qc, qr=qr, qi=qi, qs=qs)
    if (bool(cp.any(~cp.isfinite(cldfra)))
            or bool(cp.any(cldfra < 0.0))
            or bool(cp.any(cldfra > 1.0))):
        raise ValueError("cldfra must be finite and within [0, 1]")
    if numbers:
        if (bool(cp.any(~cp.isfinite(play)))
                or bool(cp.any(~cp.isfinite(tlay)))):
            raise ValueError("Morrison thermodynamic inputs must be finite")
        _require_finite_nonnegative(**numbers)
        if effective:
            _require_finite_nonnegative(**effective)
    if effective:
        bands = (EFFECTIVE_RADIUS_PLAUSIBLE_UM if radii_bands is None
                 else radii_bands)
        _require_plausible_radii_um(bands=bands, **{
            name: value for name, value in effective.items()
            if name in bands})
    if (bool(cp.any(~cp.isfinite(emiss)))
            or bool(cp.any(emiss < 0.0))
            or bool(cp.any(emiss > 1.0))):
        raise ValueError("surface emissivity must be finite and within [0, 1]")
    def replay_gas_chunk(tables, sl, *, planck):
        _require_finite_nonnegative(qv=qv[sl])
        _validate_host_range(
            "play", play[sl], float(np.min(tables.press_ref)),
            float(np.max(tables.press_ref)), "Pa")
        _validate_host_range("plev", plev[sl], 0.0, None, "Pa")
        _validate_host_range(
            "tlay", tlay[sl], float(np.min(tables.temp_ref)),
            float(np.max(tables.temp_ref)), "K")
        if bool(cp.any(play[sl, 0] < play[sl, 1])):
            raise ValueError(
                "kissvec seeding requires pmid from the bottom four layers")
        if planck:
            for name, value in (("tlev", tlev[sl]), ("tsfc", tsfc[sl])):
                _validate_host_range(
                    name, value, float(np.min(tables.temp_ref)),
                    float(np.max(tables.temp_ref)), "K")

    ncol = play.shape[0]
    for start in range(0, ncol, column_chunk):
        sl = slice(start, min(start + column_chunk, ncol))
        replay_gas_chunk(tables_lw, sl, planck=True)
    for start in range(0, ncol, column_chunk):
        sl = slice(start, min(start + column_chunk, ncol))
        replay_gas_chunk(tables_sw, sl, planck=False)
    dp = cp.abs(plev[:, 1:] - plev[:, :-1])
    if (bool(cp.any(dp <= DTYPE(0.0)))
            or bool(cp.any(exner <= DTYPE(0.0)))):
        raise ValueError("radiation pressure thickness and Exner must be positive")


@dataclass
class RRTMGPRadiation:
    """RTE+RRTMGP column driver for the frozen Phase-4 radiation slot.

    State arrays remain in woof's bottom-to-top ``(nz,ny,nx)`` layout;
    columns are packed only at the scheme boundary.  Trace gases use RFMIP
    experiment-zero climatology plus :func:`trace_gases` date selection and
    explicit case overrides.  Water vapor comes from the model and ozone is
    interpolated from the median RFMIP climatological profile.  Both come
    from :func:`load_trace_climatology`, the table derived from the RFMIP
    input file, which is not itself shipped.
    """

    #: RTE resolves the full level stack, so the top level's upward
    #: longwave flux IS WRF's OLR; the driver reads this declaration to
    #: decide whether the run's wrfout carries the field.  Unannotated on
    #: purpose: this is a class constant, not a dataclass field.
    publishes_olr = True
    #: The icloud=1 cloud fraction is built for every call whichever
    #: spectra run, so the driver's held CLDFRA output buffer exists for
    #: every RTE+RRTMGP configuration.  Class constant, as above.
    publishes_cldfra = True

    start_time: datetime
    latitude_deg: object
    longitude_deg: object
    # Controller benchmark on the 250x200x49 d01 grid (2026-07-15):
    # 256=21.9 s, 1024=5.54 s, 4096=1.71 s, 12500=1.12 s, 50000=1.36 s
    # per call, with memory flat at 0.65 GiB.  The multi-domain default is
    # capacity-led; keep it configurable for throughput tuning.
    column_chunk: int = DEFAULT_COLUMN_CHUNK
    validation_mode: str = "fused"
    # Trace-gas policy hook (Phase 5, Task 2): a mapping of well-mixed
    # RFMIP gas name -> mole fraction applied over the climatological
    # values.  None applies the pinned date-indexed policy; the frozen 1974
    # profile passes its declared 330 ppm choice explicitly.
    trace_gas_overrides: Mapping[str, float] | None = None
    longwave: bool = True
    shortwave: bool = True
    # The workspace-less chunk loop (the prepared single-domain route) runs
    # the paired solvers: one LW and one SW kernel per g-point tile give the
    # all-sky fluxes and the clear-sky top/surface rows together, with the
    # cloud finalize and (LW) the Planck sources derived in registers at the
    # staged kernels' rounding.  True selects the staged sequence instead
    # (finalize, Planck and two solves per spectrum): the reference the
    # paired kernels are held to word for word, kept selectable for that.
    staged_solvers: bool = False
    update_count: int = field(default=0, init=False)
    trace_vmr: dict[str, float] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        import cupy as cp

        if not (self.longwave or self.shortwave):
            raise ValueError("radiation adapter needs at least one spectrum")
        self.publishes_olr = bool(self.longwave)
        if not isinstance(self.start_time, datetime):
            raise TypeError("radiation_start_time must be a datetime")
        self.latitude_deg = cp.ascontiguousarray(
            cp.asarray(self.latitude_deg, dtype=DTYPE))
        self.longitude_deg = cp.ascontiguousarray(
            cp.asarray(self.longitude_deg, dtype=DTYPE))
        if self.latitude_deg.shape != self.longitude_deg.shape:
            raise ValueError("radiation latitude/longitude shapes must match")
        if self.column_chunk < 1:
            raise ValueError("column_chunk must be positive")
        if self.validation_mode not in ("fused", "full"):
            raise ValueError("validation_mode must be 'fused' or 'full'")
        self.lw_tables = load_gas_tables("lw")
        self.sw_tables = load_gas_tables("sw")
        if self.trace_gas_overrides is not None:
            supported = set()
            if self.longwave:
                supported.update(self.lw_tables.gas_names)
            if self.shortwave:
                supported.update(self.sw_tables.gas_names)
            self.trace_gas_overrides = validate_trace_gas_overrides(
                self.trace_gas_overrides, supported=supported,
                consumer="selected RRTMGP coefficient tables")
        if (not np.array_equal(self.lw_tables.press_ref,
                               self.sw_tables.press_ref)
                or not np.array_equal(self.lw_tables.temp_ref,
                                      self.sw_tables.temp_ref)
                or self.lw_tables.press_ref_trop
                != self.sw_tables.press_ref_trop):
            raise ValueError(
                "LW/SW gas tables must share an interpolation grid")
        if float(np.min(self.lw_tables.press_ref)) != RRTMGP_TOA_PRESSURE_PA:
            raise ValueError(
                "RRTMGP coefficient pressure floor drifted from the "
                "above-model column adapter")
        self.lw_cloud_tables = load_cloud_tables("lw")
        self.sw_cloud_tables = load_cloud_tables("sw")
        climatology = load_trace_climatology()
        for gas in _RFMIP_GAS_NAMES:
            self.trace_vmr[gas] = climatology.trace_vmr[gas]
        for gas, value in trace_gases(
                self.start_time, self.trace_gas_overrides).items():
            if gas not in self.trace_vmr:
                # Defensive parity with the pure policy validation: table
                # and packaged RFMIP names must never drift silently.
                raise ValueError(
                    f"unknown trace gas {gas!r}; known well-mixed gases: "
                    f"{sorted(self.trace_vmr)}")
            self.trace_vmr[gas] = value
        pressure = climatology.pressure_layer_pa
        ozone = climatology.ozone_vmr
        order = np.argsort(pressure)
        self._ozone_logp = cp.asarray(np.log(pressure[order]), dtype=DTYPE)
        self._ozone_vmr = cp.asarray(ozone[order], dtype=DTYPE)

    @staticmethod
    def _columns(array):
        """Pack ``(nz,ny,nx)`` into bottom-to-top ``(ncol,nlay)``."""
        import cupy as cp
        return cp.ascontiguousarray(array.transpose(1, 2, 0).reshape(
            array.shape[1] * array.shape[2], array.shape[0]))

    def _microphysics_columns(self, cfg, ny, nx):
        """``(ny*nx,)`` bool, True where the microphysics runs.

        False on the specified-zone ring of a specified or nested domain,
        the ring WRF's clipped tiles and woof.core.microphysics never
        touch (same predicate and slices).  None when there is no ring.
        Built on the device and cached, so a captured step adds no copy.
        """
        if not (getattr(cfg, "specified", False)
                or getattr(cfg, "nested", False)):
            return None
        sz = int(getattr(cfg, "spec_zone", 0) or 0)
        if sz <= 0:
            return None
        key = (int(ny), int(nx), sz)
        cached = getattr(self, "_microphysics_columns_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        import cupy as cp
        from woof.core.physics_inventory import spec_zone_ring_slices

        columns = cp.ones((ny, nx), dtype=cp.bool_)
        for ring in spec_zone_ring_slices(ny, nx, sz):
            columns[ring] = False
        columns = columns.reshape(-1)
        self._microphysics_columns_cache = (key, columns)
        return columns

    @staticmethod
    def _field_from_state(state, names, fallback):
        for name in names:
            value = getattr(state, name, None)
            if value is not None:
                return value
        return fallback

    def _gas_vmr(self, tables, play, qv, *, validate=True, out=None):
        """Build the (ncol, nlay, ngas + 1) volume-mixing-ratio cube.

        ONE kernel writes the whole cube.  It used to be five-plus CuPy
        expressions -- a zero fill, a scaled h2o write, an ozone write and
        one write per well-mixed trace gas -- each a strided store into the
        last axis, each its own launch.  This runs once per chunk per band,
        2364 times in a profiled hour, for 1.94 s of HOST time against
        0.94 s of device: the stores were never the cost, the submissions
        were.  Radiation is submission bound (HOWTO 13.5b) with 1.97 s of
        GPU idle in it, so that host time is wall time.

        ``cupy.interp`` is deliberately left alone and its output handed to
        the kernel.  The ozone lookup is the one piece here whose last bit
        depends on somebody else's implementation, and reproducing it in
        CUDA would be betting bit-exactness on matching it.
        """
        import cupy as cp
        from woof.core.kernels import get_kernel

        if not hasattr(cp, "RawModule"):
            # The CPU stand-in: tests substitute numpy for cupy to keep this
            # path runnable without a card.  No kernels there, so it takes
            # the expression sequence the fused kernel replaced -- which is
            # also the reference the fused path is gated against, bitwise,
            # over both tables and every trace-gas set (test_rrtmgp).
            return self._gas_vmr_expressions(
                cp, tables, play, qv, validate=validate, out=out)

        shape = (*play.shape, tables.ngas + 1)
        # Every slot is written by the kernel below -- the zero fill is part
        # of it -- so a fresh allocation need not be zeroed, and a reused
        # workspace slot is still completely overwritten before its first
        # read, exactly as RRTMGP_WORKSPACE_LIFETIME_AUDIT records.
        vmr = (cp.empty(shape, dtype=DTYPE) if out is None
               else _workspace_output(out, shape, "vmr"))
        # qv is kg water / kg dry air; RRTMGP consumes mole / mole dry air.
        if validate:
            _require_finite_nonnegative(qv=qv)
        qv = cp.ascontiguousarray(cp.asarray(qv, dtype=DTYPE))
        if qv.shape != play.shape:
            raise ValueError(
                f"gas VMR qv must have shape {play.shape}, got {qv.shape}")
        # float64 out of cupy.interp even from float32 tables; the kernel
        # narrows on the store, which is the store the CuPy path did.
        ozone = cp.interp(
            cp.log(play).ravel(), self._ozone_logp, self._ozone_vmr)
        if ozone.dtype != cp.float64:
            raise TypeError(
                "gas VMR ozone interpolation must be float64 for the fused "
                f"fill kernel, got {ozone.dtype}")
        # Later writes win, which is the precedence the sequence of CuPy
        # assignments had: h2o, then ozone, then trace gases in trace_vmr
        # order.
        trace = tuple(
            (int(tables.gas_index[gas]), float(value))
            for gas, value in self.trace_vmr.items()
            if tables.gas_index.get(gas) is not None)
        trace_index, trace_value = _trace_vmr_device(trace, xp=cp)
        ncells = int(play.shape[0]) * int(play.shape[1])
        threads = 256
        get_kernel("rrtmgp_gas", "rrtmgp_gas_vmr_fill")(
            ((ncells * (tables.ngas + 1) + threads - 1) // threads,), (threads,), (
                qv, ozone, trace_index, trace_value, vmr,
                DTYPE(0.028964 / 0.018016),
                np.int32(tables.gas_index["h2o"]),
                np.int32(tables.gas_index["o3"]),
                np.int32(len(trace)), np.int32(ncells),
                np.int32(tables.ngas + 1)))
        return vmr

    def _gas_vmr_expressions(self, xp, tables, play, qv, *, validate, out):
        """The pre-fusion VMR build, kept as the reference implementation.

        Reached on the NumPy stand-in, and it is what the fused kernel is
        gated against: the A/B runs both over both gas tables, three
        trace-gas sets, four shapes and a deliberately dirty workspace
        buffer, and requires bitwise equality.  An edit to one of these
        two belongs in the other.
        """
        shape = (*play.shape, tables.ngas + 1)
        if out is None:
            vmr = xp.zeros(shape, dtype=DTYPE)
        else:
            vmr = _workspace_output(out, shape, "vmr")
            # Slot zero and absent trace gases are real inputs to the gas
            # kernel.  The full fill is the write-before-read producer for
            # every reused byte, exactly matching zeros numerically.
            vmr.fill(DTYPE(0.0))
        # qv is kg water / kg dry air; RRTMGP consumes mole / mole dry air.
        if validate:
            _require_finite_nonnegative(qv=qv)
        vmr[:, :, tables.gas_index["h2o"]] = (
            qv * DTYPE(0.028964 / 0.018016))
        vmr[:, :, tables.gas_index["o3"]] = xp.interp(
            xp.log(play).ravel(), self._ozone_logp,
            self._ozone_vmr).reshape(play.shape)
        for gas, value in self.trace_vmr.items():
            index = tables.gas_index.get(gas)
            if index is not None:
                vmr[:, :, index] = DTYPE(value)
        return xp.ascontiguousarray(vmr)

    @staticmethod
    def _solar_constant_device(valid_time):
        """The wrf_480 SOLCON as a float32 0-d device array, capture-safe.

        The same rls_radconst launch :meth:`_solar_constant` reads back, with
        device-resident empty coordinates so nothing crosses the bus.
        """
        import cupy as cp
        from woof.core.solar import geometry_at
        empty = cp.empty((0,), cp.float32)
        return geometry_at(valid_time, empty, empty, generation="wrf_480")[0][1]

    @staticmethod
    def _solar_constant(valid_time, *, generation="wrf_480") -> float:
        """WRF SOLCON = 1370 * ECCFAC (Paltridge & Platt eccentricity).

        Transcribes ``radconst`` (module_radiation_driver.F:3504-3509) with
        WRF's 0-based fractional julian day (frame/module_domain.F:2165):
        1369.704 W/m2 at julian 92.75 (eccfac 0.999784).
        """
        from woof.core.solar import solar_generation
        solar_generation(generation)
        hour = (valid_time.hour + valid_time.minute / 60.0
                + valid_time.second / 3600.0
                + valid_time.microsecond / 3.6e9)
        julian = (valid_time.timetuple().tm_yday - 1) + hour / 24.0
        if generation == 'wrf_480':
            # WRF's REAL(4) radconst word, on the host: the same float32
            # operation order as the device kernel's rls_radconst (word for
            # word, tests/test_solar_generation_gpu.py), so a CPU-only
            # process (the npref references) gets the same SOLCON a card
            # does.  The capture-safe device twin is _solar_constant_device.
            from woof.core.rrtmg_legacy import radconst
            return float(radconst(julian)[1])
        da = 2.0 * np.pi * julian / 365.0
        eccfac = (1.000110 + 0.034221 * np.cos(da) + 0.001280 * np.sin(da)
                  + 0.000719 * np.cos(2.0 * da)
                  + 0.000077 * np.sin(2.0 * da))
        return 1370.0 * eccfac

    def _cosine_zenith(self, valid_time, *, hour_offset_seconds=0.0, generation="wrf_480"):
        """Selected WRF ``radconst``/``calc_coszen`` solar generation.

        This transcribes the standard reference case path in
        ``module_radiation_driver.F:3469-3541``.  WRF deliberately uses a
        fixed 365-day orbital phase even in leap years.  The absolute UTC
        ``valid_time`` supplies WRF's zero-based fractional ``julian`` and
        its ``gmt + mod(xtime, 1440)/60`` clock; ``hour_offset_seconds``
        shifts only the hour angle, matching the ``radt*0.5`` midpoint call.

        This is geometric COSZEN only.  It does not emulate optional WRF
        eclipse, slope-shadow, or shortwave-interpolation corrections, none
        of which are selected by the frozen reference case configuration.
        """
        import cupy as cp

        hour = (valid_time.hour + valid_time.minute / 60.0
                + valid_time.second / 3600.0
                + valid_time.microsecond / 3.6e9)
        from woof.core.solar import solar_generation, geometry_at
        solar_generation(generation)
        if generation == "wrf_480":
            start = getattr(self, 'start_time', valid_time)
            return geometry_at(valid_time, self.latitude_deg, self.longitude_deg,
                hour_offset_seconds=hour_offset_seconds, generation=generation,
                start_time=start, elapsed_seconds=(valid_time-start).total_seconds())[1]
        julian = valid_time.timetuple().tm_yday - 1.0 + hour / 24.0
        degrad = np.pi / 180.0
        dpd = 360.0 / 365.0
        if julian >= 80.0:
            solar_longitude = dpd * (julian - 80.0)
        else:
            solar_longitude = dpd * (julian + 285.0)
        declination = np.arcsin(
            np.sin(23.5 * degrad)
            * np.sin(solar_longitude * degrad))
        da = 2.0 * np.pi * (julian - 1.0) / 365.0
        equation = 229.18 * (
            0.000075 + 0.001868 * np.cos(da) - 0.032077 * np.sin(da)
            - 0.014615 * np.cos(2.0 * da) - 0.04089 * np.sin(2.0 * da))
        solar_minutes = (60.0 * (hour + hour_offset_seconds / 3600.0)
                         + equation + 4.0 * self.longitude_deg)
        hour_angle = cp.deg2rad(solar_minutes / 4.0 - 180.0)
        latitude = cp.deg2rad(self.latitude_deg)
        mu = (cp.sin(latitude) * DTYPE(np.sin(declination))
              + cp.cos(latitude) * DTYPE(np.cos(declination))
              * cp.cos(hour_angle))
        return cp.clip(mu, DTYPE(-1.0), DTYPE(1.0))

    def __call__(self, *, atmosphere, fields, state, cfg):
        pressure = atmosphere["pressure"]
        nz, ny, nx = pressure.shape
        from woof.core.chem_radiation import prepare, solve_lw, solve_sw
        chem_aerosols = prepare(state, cfg)
        full_validation = self.validation_mode == "full"
        if self.latitude_deg.shape != (ny, nx):
            raise ValueError("radiation latitude/longitude must match state grid")
        declared_p_top = getattr(state, "p_top", None)
        if declared_p_top is not None:
            lw_upper, sw_upper = rrtmgp_above_model_layer_counts(declared_p_top)
            active_upper = max(lw_upper if self.longwave else 0,
                               sw_upper if self.shortwave else 0)
            if nz + active_upper > 128:
                raise ValueError(
                    "RRTMGP radiation supports at most 128 layers including "
                    f"the above-model column, got {nz + active_upper}")
        if atmosphere["exner"].shape != pressure.shape:
            raise ValueError(
                f"exner must have shape {pressure.shape}, "
                f"got {atmosphere['exner'].shape}")
        if fields["tsk"].size != ny * nx:
            raise ValueError(
                f"tsfc must have {ny * nx} surface values, "
                f"got {fields['tsk'].size}")

        import cupy as cp

        play = self._columns(pressure)
        plev = self._columns(atmosphere["p_interface"])
        tlay = self._columns(atmosphere["temperature"])
        exner = _device_profile(
            self._columns(atmosphere["exner"]), play.shape, "exner")
        qv = _radiation_reads_nonnegative(self._columns(atmosphere["qv"]))
        if declared_p_top is None:
            # Legacy/synthetic direct callers do not carry BaseState.p_top.
            # Production always takes the scalar path above, so this fallback
            # does not add a synchronization to the forecast radiation loop.
            p_top = float(cp.asnumpy(plev[0, -1]))
        else:
            p_top = float(declared_p_top)
        # Both v1.9 gas tables share their lower pressure bound.  Clamp the
        # interface exactly as the upstream RFMIP example does.  For the reference case
        # this is now the appended TOA interface, not the 100-hPa model top.
        plev[:, -1] = cp.maximum(plev[:, -1],
                                 DTYPE(RRTMGP_TOA_PRESSURE_PA))
        profile_p_top = max(p_top, RRTMGP_TOA_PRESSURE_PA)
        tlev = _interface_temperatures(play, plev, tlay)
        tsfc = _device_profile(
            fields["tsk"].reshape(-1), (play.shape[0],), "tsfc")

        qc3 = self._field_from_state(state, ("qc",), atmosphere["qc"])
        qr3 = self._field_from_state(state, ("qr",), cp.zeros_like(qc3))
        qi3 = self._field_from_state(state, ("qi",), atmosphere["qi"])
        qs3 = self._field_from_state(state, ("qs",), cp.zeros_like(qi3))
        mp_physics = int(getattr(cfg, "mp_physics", 1))
        # Fail-closed table with its WRF citations; see
        # _MP_CLOUD_OPTICS_SCHEME.  mp=28 (THOMPSONAERO) resolves to
        # "thompson" -- the same coupling classic Thompson gets, which is
        # what Registry.EM_COMMON:3036 and module_physics_init.F:1005-1006
        # say.  Until 2026-08-01 this was a ``.get(mp_physics, "kessler")``
        # default and mp=28 silently landed on Kessler: no scheme radii,
        # and f_qi = f_qs = False into cal_cldfra1, so an overcast ice
        # cloud radiated as clear sky.
        scheme = cloud_optics_scheme(mp_physics)
        # The snow radiative treatment is bound to the compatibility
        # receipt token (fail closed on unknown values): -v2 selects the
        # WRF option-4 snow discount, -v1 and native 'none' keep the
        # original full-mass merge, so no already-issued run is relabeled.
        kwargs = {"snow_treatment": snow_treatment_for_compatibility(
            str(getattr(cfg, "wrf_rrtmg_compatibility", "none")))}
        numbers = {}
        effective_fields = {}
        if scheme == "morrison":
            for category, names in {
                    "nc": ("nc", "qnc"), "nr": ("nr", "qnr"),
                    "ni": ("ni", "qni"), "ns": ("ns", "qns")}.items():
                value = self._field_from_state(state, names, None)
                if value is None:
                    raise ValueError(
                        f"Morrison radiation coupling requires state.{names[0]} "
                        f"or state.{names[1]}")
                numbers[category] = self._columns(value)
            kwargs.update({
                "play": play, "tlay": tlay,
                **numbers,
            })
            # These diagnostics describe the just-completed Morrison update and
            # therefore become cloud optics on the *next* radiation call.  On
            # initialisation they are zero-filled state storage, not valid PSD
            # diagnostics, so retain the number-moment reconstruction until the
            # named post-RK contract has accepted at least one update.
            physics = getattr(state, "physics", None)
            have_effective = (getattr(physics, "microphysics_updates", 0) > 0)
            effective = {
                name: (self._field_from_state(state, (name,), None)
                       if have_effective else None)
                for name in ("effc", "effr", "effi", "effs")}
            if any(value is not None for value in effective.values()):
                if any(value is None for value in effective.values()):
                    raise ValueError(
                        "Morrison radiation coupling requires all of state."
                        "effc/effr/effi/effs")
                kwargs.update({name: self._columns(value)
                               for name, value in effective.items()})
                effective_fields = {
                    name: kwargs[name] for name in effective}
                # The microphysics never runs on the specified-zone ring,
                # so these carriers hold their allocation background there.
                # Without this the domain's edge columns radiated 2.5-5 um
                # cloud and ice (median 35 % of the neighbour's surface
                # shortwave under ice cloud, measured on a default run).
                kwargs["radii_written"] = self._microphysics_columns(
                    cfg, ny, nx)
        elif scheme == "milbrandt2":
            # Milbrandt-Yau derives its radii from the transported number
            # moments on EVERY call -- there is no post-update effective
            # radius to prefer, because the scheme never writes one
            # (module_mp_milbrandt2mom.F:3351-3378 is commented out and
            # woof/core/milbrandt2.py writes no eff*), so state.effc/effi/
            # effs for mp=9 are the allocation-time background for the whole
            # run and are deliberately NOT read here.  nr rides along for
            # the fused validator's number scan; hydrometeor_paths ignores
            # it, as Morrison's branch does (rain carries no radiative mass).
            for category in ("nc", "nr", "ni", "ns"):
                value = self._field_from_state(state, (category,), None)
                if value is None:
                    raise ValueError(
                        "Milbrandt-Yau radiation coupling requires "
                        f"state.{category} (woof/core/milbrandt2.py "
                        "NUMBER_SPECIES)")
                numbers[category] = self._columns(value)
            kwargs.update({"play": play, "tlay": tlay, **numbers})
        elif scheme == "p3":
            # state.effc/state.effi ARE WRF's diag_effc_3d/diag_effi_3d in
            # woof's micron convention (woof/core/p3.py writes them from
            # module_mp_p3.F:1557 and :1610).  There is no state.effs to
            # ask for: mp=50 allocates none (woof/core/state.py), which is
            # the same fact as Registry.EM_COMMON:3043's missing re_snow.
            effective = {
                name: self._field_from_state(state, (name,), None)
                for name in ("effc", "effi")}
            if any(value is None for value in effective.values()):
                raise ValueError(
                    "p3 radiation coupling requires state.effc/effi")
            kwargs.update({name: self._columns(value)
                           for name, value in effective.items()})
            # The VALIDATION view carries a third entry the PATH view does
            # not.  WRF hands its wrappers a resnow1D for P3 and its value
            # is MAX(10., re_ice*1.E6) (module_ra_rrtmg_lw.F:12256,
            # module_ra_rrtmg_sw.F:10857) -- the snow radius the coupling
            # uses IS the ice radius, so the plausibility band belongs on
            # it.  Leaving the slot empty would not skip the check: the
            # fused validator has one has-radii flag for all four slots and
            # substitutes qc for a missing one, which is a mixing ratio and
            # fails the micron band every time.
            effective_fields = {name: kwargs[name] for name in effective}
            effective_fields["effs"] = kwargs["effi"]
        elif scheme in ("wsm3", "wsm6", "thompson", "nssl"):
            effective = {
                name: self._field_from_state(state, (name,), None)
                for name in ("effc", "effi", "effs")}
            if any(value is None for value in effective.values()):
                raise ValueError(
                    f"{scheme} radiation coupling requires "
                    "state.effc/effi/effs")
            kwargs.update({name: self._columns(value)
                           for name, value in effective.items()})
            effective_fields = {name: kwargs[name] for name in effective}
        qc_cols = _radiation_reads_nonnegative(self._columns(qc3))
        qr_cols = _radiation_reads_nonnegative(self._columns(qr3))
        qi_cols = _radiation_reads_nonnegative(self._columns(qi3))
        qs_cols = _radiation_reads_nonnegative(self._columns(qs3))
        for name in list(numbers):
            clamped = _radiation_reads_nonnegative(numbers[name])
            numbers[name] = clamped
            if name in kwargs:
                kwargs[name] = clamped
        # WRF radiation driver, icloud=1 default: CLDFRA from cal_cldfra1
        # (module_radiation_driver.F:1320-1332), grid-box paths divided by
        # max(0.01, CLDFRA) into in-cloud paths, and McICA subcolumn masks
        # per g-point in the chunk loops below.
        # F_QI and F_QS are read separately because P3 separates them:
        # mp=50 is ice-active with no snow species at all
        # (Registry.EM_COMMON:3043), which is what selects cal_cldfra1's
        # own P3 arm (module_radiation_driver.F:3879-3887).  Every other
        # scheme in the table answers the same to both, so nothing else
        # moves.
        ice_active = scheme_is_ice_active(scheme)
        snow_species = scheme_has_snow_species(scheme)
        icloud = int(getattr(cfg, "icloud", 1))
        if icloud == 2:
            cldfra = cal_cldfra2(qc_cols, qi_cols,
                f_qc=int(getattr(cfg, "mp_physics", 0)) != 0, f_qi=ice_active)
        else:
            cldfra = cal_cldfra1(qv, qc_cols, qi_cols, qs_cols, tlay, play,
                                 f_qc=True, f_qi=ice_active, f_qs=snow_species,
                                 mp_physics=int(getattr(cfg, "mp_physics", 0)))
        if scheme == "wsm3":
            # WSM3 stores cloud ice in qc and snow in qr. WRF splits
            # these carriers below freezing in module_ra_sw.F:193-203
            # after cal_cldfra1 has consumed the original combined cloud.
            frozen = tlay < DTYPE(273.15)
            qi_cols = cp.ascontiguousarray(cp.where(frozen, qc_cols, DTYPE(0.0)))
            qs_cols = cp.ascontiguousarray(cp.where(frozen, qr_cols, DTYPE(0.0)))
            qc_cols = cp.ascontiguousarray(cp.where(frozen, DTYPE(0.0), qc_cols))
            qr_cols = cp.ascontiguousarray(cp.where(frozen, DTYPE(0.0), qr_cols))
        active_bl = icloud == 1 and mynn_bl_cloud_active(
            getattr(cfg, "bl_pbl_physics", 0), getattr(cfg, "icloud_bl", 0))
        qc_bl = self._columns(fields["qc_bl"]) if active_bl else None
        qi_bl = self._columns(fields["qi_bl"]) if active_bl else None
        cldfra_bl = (
            self._columns(fields["cldfra_bl"]) if active_bl else None)
        # Read before the merge, which writes qc/qi in place.
        supplied_liquid, supplied_ice = mynn_bl_cloud_supplied(
            qc_cols, qi_cols, qc_bl=qc_bl, qi_bl=qi_bl,
            cldfra_bl=cldfra_bl,
            bl_pbl_physics=getattr(cfg, "bl_pbl_physics", 0),
            icloud_bl=(getattr(cfg, "icloud_bl", 0) if icloud == 1 else 0),
            bl_mynn_version=getattr(cfg, "bl_mynn_version", "wrf_461"),
            temperature=tlay if active_bl else None)
        qc_cols, qi_cols, cldfra = merge_mynn_bl_clouds(
            qc_cols, qi_cols, cldfra, qc_bl=qc_bl, qi_bl=qi_bl,
            cldfra_bl=cldfra_bl,
            bl_pbl_physics=getattr(cfg, "bl_pbl_physics", 0),
            icloud_bl=(getattr(cfg, "icloud_bl", 0) if icloud == 1 else 0),
            itimestep=(wrf_itimestep(state.elapsed_seconds, cfg.dt)
                       if active_bl else 1),
            bl_mynn_version=getattr(cfg, "bl_mynn_version", "wrf_461"),
            temperature=tlay if active_bl else None,
        )
        if scheme in SCHEME_RADII_COUPLINGS:
            # After the MYNN merge, so its subgrid cloud and fraction are
            # the cloud being sized: see cloudy_background_radii.
            if fields.get("xland") is None:
                raise ValueError(
                    f"{scheme} radiation coupling requires fields['xland']: "
                    "a cloudy layer at the microphysics' no-cloud radius is "
                    "sized 7.5 um over land and 10.5 um over water, and "
                    "without the land/water flag it would be radiated at "
                    "the background radius, which put 150 W m-2 too little "
                    "shortwave on the ground under MYNN subgrid cloud")
            effc_cols, effi_cols = cloudy_background_radii(
                kwargs["effc"], kwargs["effi"], cldfra, tlay,
                fields["xland"], scheme=scheme,
                supplied_liquid=supplied_liquid, supplied_ice=supplied_ice)
            kwargs["effc"] = effc_cols
            kwargs["effi"] = effi_cols
            effective_fields["effc"] = effc_cols
            if scheme != "p3":
                effective_fields["effi"] = effi_cols
        ncol = play.shape[0]
        # ONE reduction per firing buys the whole above-model fast path.
        # WRF's mass coordinate fills the top interface from a scalar
        # (physics.py `p_interface[nz] = state.p_top`), so this is true by
        # construction -- but the chunk loop below is bit-exact only while
        # it holds, so it is checked rather than assumed.  Exact equality,
        # not the tolerance `_validate_model_top_interface` uses: a
        # one-ULP spread would be invisible to that and would still change
        # the synthetic cap.
        #
        # Inside a CUDA graph capture the reduction cannot be read back (a
        # device-to-host copy on a capturing stream is refused, which is
        # what took every radiation-due captured step down), so a captured
        # step takes the general per-column path instead.  That path is not
        # an assumption: it computes the same values
        # (`_extend_above_model_profile`), only without the cached rows.
        from woof.core.rrtm_lw import _stream_is_capturing

        if _stream_is_capturing(cp):
            uniform_top = False
        else:
            uniform_top = bool(cp.all(plev[:, -1] == plev[0, -1]))
        if declared_p_top is None:
            # Once per firing, not once per chunk: it synchronizes, and
            # `plev` does not vary across the chunks that follow.
            _validate_model_top_interface(plev, profile_p_top, xp=cp)
        if full_validation:
            # Validate the caller-supplied model column before constructing
            # WRF's synthetic above-model atmosphere.  Otherwise full mode
            # can report an appended cap pressure as the observed minimum,
            # while the fused production validator diagnoses the original
            # input.  Full mode exists to replay that legacy diagnostic
            # contract exactly, so run the common replay up front and let the
            # downstream cap solvers operate on already-valid profiles.
            emiss_bands = _surface_emissivity_bands(
                fields["emiss"], self.lw_tables, ny, nx, validate=False)
            _raise_full_call_validation_error(
                play=play, plev=plev, tlay=tlay, tlev=tlev,
                tsfc=tsfc, exner=exner, qv=qv, qc=qc_cols,
                qr=qr_cols, qi=qi_cols, qs=qs_cols, cldfra=cldfra,
                emiss=emiss_bands, numbers=numbers,
                effective=effective_fields, tables_lw=self.lw_tables,
                tables_sw=self.sw_tables,
                column_chunk=self.column_chunk,
                radii_bands=effective_radius_bands(scheme))
            paths = hydrometeor_paths(
                plev, qc_cols, qr_cols, qi_cols, qs_cols,
                microphysics=scheme, cldfra=cldfra, validate=False,
                **kwargs)
        else:
            emiss_bands = _surface_emissivity_bands(
                fields["emiss"], self.lw_tables, ny, nx, validate=False)
            _validate_device_call(
                play=play, plev=plev, tlay=tlay, tlev=tlev, tsfc=tsfc,
                exner=exner, qv=qv, qc=qc_cols, qr=qr_cols, qi=qi_cols,
                qs=qs_cols, cldfra=cldfra, emiss=emiss_bands,
                numbers=numbers, effective=effective_fields,
                tables_lw=self.lw_tables, tables_sw=self.sw_tables,
                radii_bands=effective_radius_bands(scheme),
                diagnose=lambda: _raise_full_call_validation_error(
                    play=play, plev=plev, tlay=tlay, tlev=tlev,
                    tsfc=tsfc, exner=exner, qv=qv, qc=qc_cols,
                    qr=qr_cols, qi=qi_cols, qs=qs_cols, cldfra=cldfra,
                    emiss=emiss_bands, numbers=numbers,
                    effective=effective_fields, tables_lw=self.lw_tables,
                    tables_sw=self.sw_tables,
                    column_chunk=self.column_chunk,
                    radii_bands=effective_radius_bands(scheme)))
            # Invalid hydrometeors are rejected above, before these path/radius
            # allocations and arithmetic.
            paths = hydrometeor_paths(
                plev, qc_cols, qr_cols, qi_cols, qs_cols,
                microphysics=scheme, cldfra=cldfra, validate=False, **kwargs)

        workspace = getattr(self, "chunk_workspace", None)
        if workspace is not None:
            if (int(workspace.nz) != nz
                    or int(workspace.column_chunk) != self.column_chunk
                    or float(workspace.p_top) != p_top):
                raise ValueError(
                    "RRTMGP adapter/workspace shape drift: "
                    f"adapter nz/chunk/p_top={(nz, self.column_chunk, p_top)}, "
                    "workspace "
                    f"{(workspace.nz, workspace.column_chunk, workspace.p_top)}")

        flux_shape = (ncol, nz + 1)
        lw_up = (cp.empty(flux_shape, dtype=DTYPE) if self.longwave
                 else cp.zeros(flux_shape, dtype=DTYPE))
        lw_dn = cp.empty_like(lw_up) if self.longwave else cp.zeros_like(lw_up)
        # OLR is WRF's upward longwave at the top of the WHOLE radiation
        # column, above-model layers included (module_ra_rrtmg_lw.F:
        # olr(i,j) = uflx(1,kte+2)), not at the model top: the model-flux
        # store below keeps levels 0..nz only, and the above-model column
        # emits and absorbs, so its top differs from lw_up[:, nz] by a few
        # W m-2.  One value per column, taken off the same solve.
        lw_up_toa = (cp.empty(ncol, dtype=DTYPE) if self.longwave
                     else None)
        # WRF's instantaneous top/bottom fluxes, all-sky and clear-sky, for
        # the spectra this adapter runs (SWUPT ... LWDNBC, the history rows
        # and the AC* accumulations of physics_inventory.
        # radiation_boundary_flux_names).  The clear-sky rows come from a
        # second RTE solve per chunk on the same gas optics with zero cloud;
        # it only reads its inputs, so the all-sky solve and every field
        # derived from it are unchanged.  Dark SW columns keep zero.
        from woof.core.physics_inventory import RADIATION_BOUNDARY_FLUXES
        boundary_names = tuple(
            name for name in RADIATION_BOUNDARY_FLUXES
            if (self.longwave if name.startswith("lw") else self.shortwave))
        boundary = cp.zeros((len(boundary_names), ncol), dtype=DTYPE)
        boundary_rows = {name: boundary[index]
                         for index, name in enumerate(boundary_names)}
        for start in range(0, ncol if self.longwave else 0, self.column_chunk):
            sl = slice(start, min(start + self.column_chunk, ncol))
            chunk_ncol = sl.stop - sl.start
            lw_chunk = _prepare_above_model_chunk(
                tables=self.lw_tables, play=play, plev=plev, tlay=tlay,
                tlev=tlev, qv=qv, paths=paths, cldfra=cldfra, columns=sl,
                p_top=profile_p_top, kind="lw",
                pressure_floor=RRTMGP_TOA_PRESSURE_PA, xp=cp,
                validate_top=False,
                validate=full_validation, uniform_top=uniform_top,
                scratch=workspace is not None)
            lw_profile = lw_chunk.solver_profile or lw_chunk.profile
            lw_paths = lw_chunk.paths
            lw_cldfra = lw_chunk.cldfra
            chunk_metadata = lw_chunk.metadata
            if workspace is None:
                vmr_lw = self._gas_vmr(
                    self.lw_tables, lw_profile.play, lw_profile.qv,
                    validate=full_validation)
                gas_lw = _gas_optics(
                    self.lw_tables, lw_profile.play, lw_profile.plev,
                    lw_profile.tlay, vmr_lw,
                    metadata=chunk_metadata, validate=full_validation,
                    zero_g_sentinel=True)
                cld_lw = cloud_optics(
                    self.lw_cloud_tables, lw_paths.clwp, lw_paths.ciwp,
                    lw_paths.reliq, lw_paths.dgice)
                mask_lw = mcica_cloud_masks(
                    lw_profile.play, lw_cldfra, self.lw_tables.ngpt,
                    MCICA_PERMUTESEED_LW, validate=full_validation)
                emiss = _expand_band_to_gpoint(
                    emiss_bands[sl], self.lw_tables, "surface emissivity")
                if chem_aerosols is not None:
                    planck_in = PlanckInputs(
                        tables=self.lw_tables, play=lw_profile.play,
                        plev=lw_profile.plev, tlay=lw_profile.tlay,
                        tlev=lw_profile.tlev, tsfc=tsfc[sl], vmr=vmr_lw,
                        metadata=chunk_metadata)
                    flux, clear = solve_lw(
                        chem_aerosols, sl, self.lw_tables, gas_lw,
                        cld_lw, mask_lw, emiss, planck_in)
                    del planck_in
                elif self.staged_solvers:
                    # The staged reference: finalize and Planck kernels
                    # write their cubes and the solver reads them back.
                    optics_lw = _finalize_cloud_optics(
                        self.lw_tables, gas_lw, cld_lw, cloud_mask=mask_lw)
                    sources = _planck_sources(
                        self.lw_tables, lw_profile.play, lw_profile.plev,
                        lw_profile.tlay, lw_profile.tlev, tsfc[sl],
                        vmr_lw, metadata=chunk_metadata,
                        validate=full_validation)
                    clear = lw_rte(
                        gas_lw.tau, sources.lay_source, sources.lev_source,
                        sources.sfc_source, emiss, top_at_1=False)
                    flux = lw_rte(
                        optics_lw.tau, sources.lay_source, sources.lev_source,
                        sources.sfc_source, emiss, top_at_1=False)
                    del optics_lw, sources
                else:
                    # One paired solve (rrtmgp_lw_noscat_pair): gas tau,
                    # band cloud optics and the McICA mask combine in
                    # registers at the finalize kernel's pinned rounding;
                    # the Planck sources are derived in the solver at the
                    # DEFAULT process's standalone-kernel rounding
                    # (planck_staged, rrtmgp_planck_staged_default.cuh);
                    # the clear-sky solve shares the reads and the clear
                    # layers' terms and folds only the top and surface.
                    # The finalized tau cube, the lay/lev/sfc source cubes
                    # and the second solve do not exist.  Same words as the
                    # staged branch above.
                    planck_in = PlanckInputs(
                        tables=self.lw_tables, play=lw_profile.play,
                        plev=lw_profile.plev,
                        tlay=lw_profile.tlay, tlev=lw_profile.tlev,
                        tsfc=tsfc[sl], vmr=vmr_lw, metadata=chunk_metadata)
                    flux, clear = _lw_rte(
                        gas_lw.tau, None, None, None, emiss, top_at_1=False,
                        out=None, incident_out=None,
                        fused_cloud=FusedCloudOptics(
                            tables=self.lw_tables, cloud=cld_lw,
                            mask=mask_lw),
                        planck=planck_in, planck_staged=True,
                        with_clear=True)
                    del planck_in
                del mask_lw
            else:
                # SCRATCH_SLOT_LIFETIME_AUDIT analogue: every optics slot is
                # a kernel output or full fill before its first read in this
                # chunk.  The RTE layout preserves common-slot offsets and
                # overwrites only the now-dead mask tail with its own outputs.
                work = workspace.phase("lw_optics", chunk_ncol)
                vmr_lw = self._gas_vmr(
                    self.lw_tables, lw_profile.play, lw_profile.qv,
                    validate=full_validation, out=work["vmr"])
                gas_lw = _gas_optics(
                    self.lw_tables, lw_profile.play, lw_profile.plev,
                    lw_profile.tlay, vmr_lw,
                    metadata=chunk_metadata, validate=full_validation,
                    zero_g_sentinel=True, out=(work["gas_tau"],),
                    compute_col_dry=False)
                cld_lw = _cloud_optics(
                    self.lw_cloud_tables, lw_paths.clwp, lw_paths.ciwp,
                    lw_paths.reliq, lw_paths.dgice,
                    out=(work["cld_tau"], work["cld_ssa"],
                         work["cld_asy"]))
                mask_lw = _mcica_cloud_masks(
                    lw_profile.play, lw_cldfra, self.lw_tables.ngpt,
                    MCICA_PERMUTESEED_LW, validate=full_validation,
                    out=work["mcica_mask"])
                # No finalize: the solver combines gas, cloud and mask in
                # registers (FusedCloudOptics).  gas_tau, both band cloud
                # cubes and the mask are CARRIED slots of the rte phase,
                # so the views below stay valid across the phase() call.
                work = workspace.phase("lw_rte", chunk_ncol)
                # No Planck kernel: the solver derives lay/lev/sfc itself.
                planck_in = PlanckInputs(
                    tables=self.lw_tables, play=lw_profile.play,
                    plev=lw_profile.plev,
                    tlay=lw_profile.tlay, tlev=lw_profile.tlev,
                    tsfc=tsfc[sl], vmr=vmr_lw, metadata=chunk_metadata)
                emiss = _expand_band_to_gpoint(
                    emiss_bands[sl], self.lw_tables, "surface emissivity",
                    out=work["emiss_gpt"])
                # Clear sky first, on fresh outputs: gas tau alone, the
                # same solver-derived Planck sources.
                if chem_aerosols is not None:
                    flux, clear = solve_lw(
                        chem_aerosols, sl, self.lw_tables, gas_lw,
                        cld_lw, mask_lw, emiss, planck_in,
                        out=(work["flux_up"], work["flux_dn"]),
                        incident_out=work["incident"])
                else:
                    clear = _lw_rte(
                        gas_lw.tau, None, None, None, emiss, top_at_1=False,
                        out=None, incident_out=None, planck=planck_in)
                    flux = _lw_rte(
                        gas_lw.tau, None, None, None, emiss, top_at_1=False,
                        out=(work["flux_up"], work["flux_dn"]),
                        incident_out=work["incident"],
                        fused_cloud=FusedCloudOptics(
                            tables=self.lw_tables, cloud=cld_lw, mask=mask_lw),
                        planck=planck_in)
                del mask_lw, planck_in
            _store_model_flux_pair(
                flux.flux_up, flux.flux_dn, nz, lw_up[sl], lw_dn[sl], xp=cp)
            lw_up_toa[sl] = flux.flux_up[:, -1]
            _store_boundary_fluxes(boundary_rows, "lw", sl, flux, clear,
                                   xp=cp)
            del vmr_lw, gas_lw, cld_lw, emiss, flux, clear
            del chunk_metadata, lw_profile, lw_paths, lw_cldfra, lw_chunk
            # ``work`` is the phase view mapping and every value in it is a
            # view of the workspace backing, so the name keeps the whole
            # allocation alive past the loop.  Rebind rather than ``del``:
            # the name does not exist on the workspace-less branch.
            work = None

        valid_time = (self.start_time
                      + timedelta(seconds=float(state.elapsed_seconds)))
        # WRF evaluates the HOUR ANGLE at the CENTER of the radiation
        # interval: calc_coszen receives xtime + radt*0.5 inside the
        # Solar_step block (module_radiation_driver.F:1206-1208, 'jararias
        # 2013/08/10') while declination/EOT stay at the call-time julian
        # (:3514-3541); that coszen is what RRTMG SW consumes (driver:2636).
        from woof.core.physics import _physics_period_seconds
        radt_minutes = cfg.radt if cfg.radt > 0.0 else cfg.radt_minutes
        radt_seconds = _physics_period_seconds(radt_minutes, cfg)
        from woof.core.solar import solar_period_seconds
        radt_seconds = solar_period_seconds(cfg, radt_seconds)
        mu_raw = self._cosine_zenith(
            valid_time, hour_offset_seconds=0.5 * radt_seconds,
            generation=getattr(cfg, "solar_generation", "wrf_480"))
        daylight = mu_raw > DTYPE(0.0)
        mu = cp.where(daylight, mu_raw, DTYPE(1.0)).reshape(-1)
        albedo_surface = cp.asarray(
            fields["albedo"], dtype=DTYPE).reshape(-1)
        solar = _solar_source_device(self.sw_tables)
        # WRF scales every SW band by scon/rrsw_scon so TOA irradiance
        # equals radconst's SOLCON (module_ra_rrtmg_sw.F:10872 'scon =
        # solcon*(1-obscur)', 9867-9871 'solvar(ib) = scon/rrsw_scon');
        # the RRTMGP-native equivalent normalizes the g-point source total
        # (the shipped table sums to tsi_default = 1360.8577 W/m2).
        source_total = float(np.sum(np.asarray(self.sw_tables.solar_source,
                                               dtype=np.float64)))
        generation = getattr(cfg, 'solar_generation', 'wrf_480')
        if generation == 'wrf_480':
            # The WRF 4.8 SOLCON is the device sun's REAL(4) radconst value.
            # It stays on the card: reading it back (or staging host inputs
            # for it) is refused inside a CUDA graph capture, which made
            # every radiation-due step uncapturable (tilestream graphcap,
            # 2.9.0 round 2).  float32 -> float64 division -> float32 is the
            # host expression below, evaluated on the device, so the scale
            # carries the same bits.
            solcon = self._solar_constant_device(valid_time)
            solar = solar * (solcon.astype(cp.float64)
                             / source_total).astype(DTYPE)
        else:
            solar = solar * DTYPE(
                self._solar_constant(valid_time, generation=generation)
                / source_total)
        # A dark column's shortwave result is replaced by zero below, so
        # it is never computed: every SW stage is a function of its own
        # column (per-cell optics, a McICA seed from the column's own
        # pressures, a per-column solve whose g-point sum order is fixed),
        # so a daylit column gives the same bits in any chunk.  Inside a
        # graph capture nothing can be read back, and the full-column
        # path runs as before.
        sw_ncol = ncol if self.shortwave else 0
        sw_columns = None
        if self.shortwave and not _stream_is_capturing(cp):
            sw_ncol = int(cp.count_nonzero(daylight))
            if 0 < sw_ncol < ncol:
                # Ascending column order, so chunks stay in model order.
                sw_columns = cp.flatnonzero(daylight.reshape(-1))
        sw_allocate = cp.empty_like if sw_ncol == ncol else cp.zeros_like
        sw_up = sw_allocate(lw_up)
        sw_dn = sw_allocate(lw_up)
        # The surface direct beam of the same two-stream solve (RRTMG's
        # SWDDIR analogue), computed only when a consumer asked for it:
        # BEP+BEM (sf_urban_physics = 3) allocated SWDDIR/SWDDIF in the
        # surface fields, or slope_rad asked for the surface DIFFUSE flux
        # (woof.core.topo_radiation), total minus direct beam at the
        # surface (RTE's flux_dn already carries the direct beam).  One
        # plane, night columns keep the radiation driver's zero, and
        # nothing it feeds changes any other output.
        want_swddir = bool(self.shortwave and "swddir" in fields)
        sw_dir_sfc = (cp.zeros(lw_up.shape[0], dtype=DTYPE)
                      if want_swddir or (self.shortwave and getattr(
                          self, "surface_diffuse_requested", False))
                      else None)
        for start in range(0, sw_ncol, self.column_chunk):
            stop = min(start + self.column_chunk, sw_ncol)
            sl = (slice(start, stop) if sw_columns is None
                  else sw_columns[start:stop])
            chunk_ncol = stop - start
            sw_chunk = _prepare_above_model_chunk(
                tables=self.sw_tables, play=play, plev=plev, tlay=tlay,
                tlev=tlev, qv=qv, paths=paths, cldfra=cldfra, columns=sl,
                p_top=profile_p_top, kind="sw",
                pressure_floor=RRTMGP_TOA_PRESSURE_PA, xp=cp,
                validate_top=False,
                validate=full_validation, uniform_top=uniform_top,
                # A moving daylight boundary produces many tail widths.
                # Reuse full chunks without retaining every tail in the cache.
                scratch=(workspace is not None and
                         (sw_columns is None or
                          chunk_ncol == self.column_chunk)))
            sw_profile = sw_chunk.solver_profile or sw_chunk.profile
            sw_paths = sw_chunk.paths
            sw_cldfra = sw_chunk.cldfra
            chunk_metadata = sw_chunk.metadata
            if workspace is None:
                vmr_sw = self._gas_vmr(
                    self.sw_tables, sw_profile.play, sw_profile.qv,
                    validate=full_validation)
                gas_sw = _gas_optics(
                    self.sw_tables, sw_profile.play, sw_profile.plev,
                    sw_profile.tlay, vmr_sw,
                    metadata=chunk_metadata, validate=full_validation,
                    zero_g_sentinel=True)
                cld_sw = cloud_optics(
                    self.sw_cloud_tables, sw_paths.clwp, sw_paths.ciwp,
                    sw_paths.reliq, sw_paths.dgice)
                mask_sw = mcica_cloud_masks(
                    sw_profile.play, sw_cldfra, self.sw_tables.ngpt,
                    MCICA_PERMUTESEED_SW, validate=full_validation)
                albedo, inc, mu_chunk = _prepare_sw_inputs(
                    albedo_surface[sl], solar, mu[sl], sw_profile.play.shape[1],
                    xp=cp)
                if chem_aerosols is not None:
                    flux, clear = solve_sw(
                        chem_aerosols, sl, self.sw_tables, gas_sw, cld_sw,
                        mask_sw, mu_chunk, albedo, inc)
                elif self.staged_solvers:
                    clear = _sw_rte(
                        gas_sw.tau, gas_sw.ssa, None, mu_chunk, albedo,
                        albedo, inc, top_at_1=False, out=None,
                        fused_cloud=_clear_sky_cloud(
                            self.sw_tables, *gas_sw.tau.shape[:2]))
                    optics_sw = _finalize_cloud_optics(
                        self.sw_tables, gas_sw, cld_sw, cloud_mask=mask_sw)
                    flux = sw_rte(
                        optics_sw.tau, optics_sw.ssa, optics_sw.g, mu_chunk,
                        albedo, albedo, inc, top_at_1=False)
                    del optics_sw
                else:
                    # As in the LW branch: one paired solve
                    # (rrtmgp_sw_2stream_pair) with the cloud finalize in
                    # registers and the clear-sky solve sharing the layer
                    # coefficients of every McICA-clear layer.
                    flux, clear = _sw_rte(
                        gas_sw.tau, gas_sw.ssa, None, mu_chunk,
                        albedo, albedo, inc, top_at_1=False, out=None,
                        fused_cloud=FusedCloudOptics(
                            tables=self.sw_tables, cloud=cld_sw,
                            mask=mask_sw),
                        with_clear=True)
                del mask_sw
            else:
                work = workspace.phase("sw_optics", chunk_ncol)
                vmr_sw = self._gas_vmr(
                    self.sw_tables, sw_profile.play, sw_profile.qv,
                    validate=full_validation, out=work["vmr"])
                gas_sw = _gas_optics(
                    self.sw_tables, sw_profile.play, sw_profile.plev,
                    sw_profile.tlay, vmr_sw,
                    metadata=chunk_metadata, validate=full_validation,
                    zero_g_sentinel=True,
                    out=(work["gas_tau"], work["gas_ssa"]),
                    compute_col_dry=False)
                cld_sw = _cloud_optics(
                    self.sw_cloud_tables, sw_paths.clwp, sw_paths.ciwp,
                    sw_paths.reliq, sw_paths.dgice,
                    out=(work["cld_tau"], work["cld_ssa"],
                         work["cld_asy"]))
                mask_sw = _mcica_cloud_masks(
                    sw_profile.play, sw_cldfra, self.sw_tables.ngpt,
                    MCICA_PERMUTESEED_SW, validate=full_validation,
                    out=work["mcica_mask"])
                # No finalize; see the LW branch.
                work = workspace.phase("sw_rte", chunk_ncol)
                albedo, inc, mu_chunk = _prepare_sw_inputs(
                    albedo_surface[sl], solar, mu[sl], sw_profile.play.shape[1],
                    out=(work["albedo_gpt"], work["inc_gpt"], work["mu0"]), xp=cp)
                # Clear sky first, on fresh outputs (see the LW branch).
                if chem_aerosols is not None:
                    flux, clear = solve_sw(
                        chem_aerosols, sl, self.sw_tables, gas_sw, cld_sw,
                        mask_sw, mu_chunk, albedo, inc,
                        out=(work["flux_up"], work["flux_dn"], work["flux_dir"]))
                else:
                    clear = _sw_rte(
                        gas_sw.tau, gas_sw.ssa, None, mu_chunk,
                        albedo, albedo, inc, top_at_1=False, out=None,
                        fused_cloud=_clear_sky_cloud(
                            self.sw_tables, *gas_sw.tau.shape[:2]))
                    flux = _sw_rte(
                        gas_sw.tau, gas_sw.ssa, None, mu_chunk,
                        albedo, albedo, inc, top_at_1=False,
                        out=(work["flux_up"], work["flux_dn"],
                             work["flux_dir"]),
                        fused_cloud=FusedCloudOptics(
                            tables=self.sw_tables, cloud=cld_sw, mask=mask_sw))
                del mask_sw
            _store_boundary_fluxes(
                boundary_rows, "sw", sl, flux, clear,
                daylight=(daylight.reshape(-1)[sl] if sw_columns is None
                          else None), xp=cp)
            del clear
            if sw_columns is None:
                _store_model_flux_pair(
                    flux.flux_up, flux.flux_dn, nz, sw_up[sl], sw_dn[sl],
                    daylight=daylight.reshape(-1)[sl], xp=cp)
            else:
                # Compacted daylit columns: an index array selects a COPY,
                # not a view, so the fused store fills chunk-shaped arrays
                # that are then scattered back.  Every column here is
                # daylit, so the store applies no mask.
                chunk_up = cp.empty((chunk_ncol, nz + 1), dtype=DTYPE)
                chunk_dn = cp.empty_like(chunk_up)
                _store_model_flux_pair(
                    flux.flux_up, flux.flux_dn, nz, chunk_up, chunk_dn,
                    xp=cp)
                sw_up[sl] = chunk_up
                sw_dn[sl] = chunk_dn
                del chunk_up, chunk_dn
            if sw_dir_sfc is not None:
                direct = _model_flux_interfaces(flux.flux_dir, nz, xp=cp)[:, 0]
                sw_dir_sfc[sl] = (direct if sw_columns is not None else
                                  cp.where(daylight.reshape(-1)[sl], direct,
                                           DTYPE(0.0)))
                del direct
            del vmr_sw, gas_sw, cld_sw
            del albedo, inc, flux, chunk_metadata
            del sw_profile, sw_paths, sw_cldfra, sw_chunk
            # As in the LW loop: ``work`` and the materialized mu0 broadcast
            # are workspace views and hold the backing alive past the loop.
            work = mu_chunk = None

        result = _fluxes_to_radiation(
            lw_up, lw_dn, sw_up, sw_dn, plev, exner, ny=ny, nx=nx,
            coszen=mu_raw, validate=full_validation, lw_up_toa=lw_up_toa)
        result.boundary_fluxes = {
            name: boundary[index].reshape(ny, nx)
            for index, name in enumerate(boundary_names)}
        if sw_dir_sfc is not None:
            if want_swddir:
                result.swddir = cp.ascontiguousarray(
                    sw_dir_sfc.reshape(ny, nx))
            result.swddif = cp.ascontiguousarray(
                (sw_dn[:, 0] - sw_dir_sfc).reshape(ny, nx))
        if not self.longwave:
            result.glw = fields["glw"]
            result.olr = None
        # CLDFRA: the icloud=1 fraction after the MYNN merge, the one both
        # spectra radiated through (read only by the path and mask
        # builders above), back on the model grid for the driver's held
        # output buffer.  Computed on every call whatever the spectra.
        result.cldfra = cp.ascontiguousarray(
            cldfra.reshape(ny, nx, nz).transpose(2, 0, 1))
        # Release hook.  The shipped SharedRRTMGPChunkWorkspace has no
        # ``on_call_end``, so it keeps its persistent behaviour with no
        # branch and no cost; a workspace that CAN hand its bytes back
        # between firings does so here, after the last chunk of both bands
        # has been consumed and _fluxes_to_radiation has read the model
        # fluxes out of the per-call arrays.  Deliberately not in a finally:
        # a call that raised leaves the backing intact for the traceback and
        # for whatever inspects the workspace afterwards.
        release = getattr(workspace, "on_call_end", None)
        if release is not None:
            release()
        self.update_count += 1
        return result


def _device_profile(value, shape, name):
    import cupy as cp
    out = cp.ascontiguousarray(cp.asarray(value, dtype=DTYPE))
    if out.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {out.shape}")
    return out


def _workspace_output(out, shape, name, *, dtype=DTYPE):
    """Return an exact contiguous output view, allocating only if absent."""
    import cupy as cp

    shape = tuple(int(extent) for extent in shape)
    if out is None:
        return cp.empty(shape, dtype=dtype)
    if tuple(out.shape) != shape:
        raise ValueError(
            f"RRTMGP workspace {name} shape {tuple(out.shape)} != {shape}")
    if np.dtype(out.dtype) != np.dtype(dtype):
        raise ValueError(
            f"RRTMGP workspace {name} dtype {out.dtype} != {np.dtype(dtype)}")
    if not bool(out.flags.c_contiguous):
        raise ValueError(f"RRTMGP workspace {name} must be C-contiguous")
    return out


def _interface_temperatures(play, plev, tlay):
    """Construct pressure-weighted level temperatures.

    Exact transcription of ``mo_gas_optics_rrtmgp.F90:890-909`` at the
    pinned fa107a1 commit, including pressure-linear boundary extrapolation
    and the reference's pressure-weighted interior interpolation.
    """
    import cupy as cp

    play = cp.ascontiguousarray(cp.asarray(play, dtype=DTYPE))
    if play.ndim != 2 or play.shape[1] < 2:
        raise ValueError("play must have shape (ncol,nlay), nlay >= 2")
    ncol, nlay = play.shape
    plev = _device_profile(plev, (ncol, nlay + 1), "plev")
    tlay = _device_profile(tlay, (ncol, nlay), "tlay")
    tlev = cp.empty((ncol, nlay + 1), dtype=DTYPE)
    tlev[:, 0] = tlay[:, 0] + (plev[:, 0] - play[:, 0]) * (
        tlay[:, 1] - tlay[:, 0]) / (play[:, 1] - play[:, 0])
    tlev[:, -1] = tlay[:, -1] + (plev[:, -1] - play[:, -1]) * (
        tlay[:, -1] - tlay[:, -2]) / (play[:, -1] - play[:, -2])
    tlev[:, 1:-1] = (
        play[:, :-1] * tlay[:, :-1] * (plev[:, 1:-1] - play[:, 1:])
        + play[:, 1:] * tlay[:, 1:] * (play[:, :-1] - plev[:, 1:-1])
    ) / (plev[:, 1:-1] * (play[:, :-1] - play[:, 1:]))
    return cp.ascontiguousarray(tlev)


def _validate_host_range(name, value, lower, upper, unit):
    """Validate a device profile on the host before launching RRTMGP."""
    import cupy as cp

    host = np.asarray(cp.asnumpy(value), dtype=np.float64)
    if host.size == 0 or not np.all(np.isfinite(host)):
        raise ValueError(f"{name} range contains non-finite values")
    observed = (float(np.min(host)), float(np.max(host)))
    if observed[0] < lower or (upper is not None and observed[1] > upper):
        bound = (f"[{lower:.9g}, {upper:.9g}]" if upper is not None
                 else f"[{lower:.9g}, infinity)")
        raise ValueError(
            f"{name} range [{observed[0]:.9g}, {observed[1]:.9g}] {unit} "
            f"is outside allowed range {bound} {unit}")


def _radiation_reads_nonnegative(columns):
    """Strict mode: the radiation reads max(0, q), as WRF's driver does.

    WRF's radiation column setup clamps every water species it reads
    (``QV1D(K)=max(0.,QV1D(K))``, ``QC1D(K)=max(0.,QC1D(K))`` and the
    rest, module_ra_rrtmg_lw.F:12009-12143).  Strict mode carries WRF's
    below-zero scalar residuals in the state (rk_update_scalar with no
    clamp), and handing them to this scheme unclamped refused 55 RTE+RRTMGP
    runs of combo sweep round 3 (2026-10-07) in the input validation.  The
    default path's state is already clamped, so it is left as it was.
    Returns a new array; the state is not touched.  ``None`` passes through.
    """
    if not WRF_EXACT or columns is None:
        return columns
    import cupy as cp

    xp = cp.get_array_module(columns)
    return xp.maximum(columns, xp.asarray(0, dtype=columns.dtype))


def _require_finite_nonnegative(**fields):
    """Reject upstream physics defects instead of silently clearing them."""
    import cupy as cp

    for name, value in fields.items():
        finite = cp.isfinite(value)
        negative = finite & (value < 0.0)
        invalid = ~finite | negative
        if bool(cp.any(invalid)):
            # This replay runs only after the fused production predicate has
            # already failed.  Spend the extra failure-path synchronizations
            # to preserve the actual upstream defect in the capsule instead
            # of reducing a multi-million-cell field to a generic label.
            flat_index = int(cp.asnumpy(cp.argmax(invalid.reshape(-1))))
            first_value = float(cp.asnumpy(value.reshape(-1)[flat_index]))
            negative_count = int(cp.asnumpy(cp.count_nonzero(negative)))
            nonfinite_count = int(cp.asnumpy(cp.count_nonzero(~finite)))
            index = tuple(int(part) for part in np.unravel_index(
                flat_index, tuple(int(extent) for extent in value.shape)))
            raise ValueError(
                f"{name} must be finite and non-negative: "
                f"first_index={index}, first_value={first_value:.9g}, "
                f"negative_count={negative_count}, "
                f"nonfinite_count={nonfinite_count}")


def _require_plausible_radii_um(*, bands=None, **fields):
    """Gate the micron contract of every radiation-facing radii writer.

    Keyword names select the band from ``bands``, which defaults to
    :data:`EFFECTIVE_RADIUS_PLAUSIBLE_UM`; a scheme whose declared range
    is wider than the generic one passes its own map from
    :func:`effective_radius_bands`.  A writer that emits metres (or any
    other metric prefix) lands outside its band on background-filled cells
    alone and fails here instead of silently radiating at a clip floor.
    """
    import cupy as cp

    if bands is None:
        bands = EFFECTIVE_RADIUS_PLAUSIBLE_UM
    for name, value in fields.items():
        lower, upper = bands[name]
        if bool(cp.any(value < DTYPE(lower))) \
                or bool(cp.any(value > DTYPE(upper))):
            raise ValueError(
                f"{name} is outside the physical-plausibility band "
                f"[{lower}, {upper}] microns; the state contract is "
                "microns -- a radii writer probably emitted another unit")


def _fluxes_to_radiation(lw_up, lw_dn, sw_up, sw_dn, plev, exner, *,
                         ny, nx, coszen=None, validate=True, lw_up_toa=None):
    """Map bottom-to-top broadband column fluxes into the radiation slot.

    ``lw_up_toa`` is the upward longwave at the top of the full radiation
    column (one value per column) and OLR is taken from it.  Without it the
    top of the given level stack is used, for callers whose stack already
    ends at the column top.
    """
    import cupy as cp
    from woof.core import constants
    from woof.core.physics import RadiationResult

    exner = cp.ascontiguousarray(cp.asarray(exner, dtype=DTYPE))
    if exner.ndim != 2 or exner.shape[0] != ny * nx:
        raise ValueError("exner columns must have shape (ny*nx,nlay)")
    ncol, nlay = exner.shape
    level_shape = (ncol, nlay + 1)
    lw_up = _device_profile(lw_up, level_shape, "lw_up")
    lw_dn = _device_profile(lw_dn, level_shape, "lw_dn")
    sw_up = _device_profile(sw_up, level_shape, "sw_up")
    sw_dn = _device_profile(sw_dn, level_shape, "sw_dn")
    plev = _device_profile(plev, level_shape, "plev")
    olr = (lw_up[:, -1] if lw_up_toa is None
           else _device_profile(lw_up_toa, (ncol,), "lw_up_toa"))
    dp = cp.abs(plev[:, 1:] - plev[:, :-1])
    if validate and (bool(cp.any(dp <= DTYPE(0.0)))
                     or bool(cp.any(exner <= DTYPE(0.0)))):
        raise ValueError("radiation pressure thickness and Exner must be positive")

    def theta_heating(flux_up, flux_dn):
        # RTE+RRTMGP ``rte/extensions/mo_heating_rates.F90:30-63`` diagnoses
        # temperature tendency from pressure-coordinate flux convergence.
        # The frozen woof slot consumes potential-temperature tendency.
        net_down = flux_dn - flux_up
        convergence = net_down[:, 1:] - net_down[:, :-1]
        packed = DTYPE(constants.G / constants.CP) * convergence / dp / exner
        return cp.ascontiguousarray(packed.reshape(ny, nx, nlay).transpose(
            2, 0, 1))

    return RadiationResult(
        rthratenlw=theta_heating(lw_up, lw_dn),
        rthratensw=theta_heating(sw_up, sw_dn),
        swdown=cp.ascontiguousarray(sw_dn[:, 0].reshape(ny, nx)),
        glw=cp.ascontiguousarray(lw_dn[:, 0].reshape(ny, nx)),
        # OLR: the upward longwave flux at the top of the radiation
        # column (the above-model layers included), WRF's uflx(1,kte+2).
        olr=cp.ascontiguousarray(olr.reshape(ny, nx)),
        gsw=cp.ascontiguousarray(
            (sw_dn[:, 0] - sw_up[:, 0]).reshape(ny, nx)),
        coszen=(None if coszen is None else cp.ascontiguousarray(
            cp.asarray(coszen, dtype=DTYPE).reshape(ny, nx))))


def _interpolation_metadata(tables: GasTables, play, tlay, *,
                            validate=True,
                            scratch: str | None = None,
                            ) -> _InterpolationMetadata:
    """Compute driver-owned reference interpolation coordinates once.

    The CUDA prepass transcribes the expressions formerly evaluated inside
    each gas-optics call and inside Planck's g-point loop.  It is intentionally
    recomputed for every radiation call; no cross-call cache or public reuse
    contract is kept.
    """
    import cupy as cp
    from woof.core.kernels import get_kernel

    play = cp.ascontiguousarray(cp.asarray(play, dtype=DTYPE))
    if play.ndim != 2:
        raise ValueError("play must be a 2-D (ncol,nlay) array")
    tlay = _device_profile(tlay, play.shape, "tlay")
    if validate:
        _validate_host_range("play", play, float(np.min(tables.press_ref)),
                             float(np.max(tables.press_ref)), "Pa")
        _validate_host_range("tlay", tlay, float(np.min(tables.temp_ref)),
                             float(np.max(tables.temp_ref)), "K")
    d = tables.to_device()
    if scratch is None:
        integer = tuple(cp.empty(play.shape, dtype=cp.int32) for _ in range(3))
        fraction = tuple(cp.empty(play.shape, dtype=DTYPE) for _ in range(2))
    else:
        # The prepass writes every element of all five, so a reused buffer
        # needs nothing established first.
        integer = tuple(
            _chunk_scratch(f"{scratch}.{name}", play.shape, xp=cp,
                           dtype=cp.int32)[0]
            for name in ("iatm", "jt", "jp"))
        fraction = tuple(
            _chunk_scratch(f"{scratch}.{name}", play.shape, xp=cp)[0]
            for name in ("ftemp", "fpress"))
    n = play.size
    threads = 256
    get_kernel("rrtmgp_gas", "rrtmgp_interpolation_prepass")(
        ((n + threads - 1) // threads,), (threads,), (
            play, tlay, d.press_ref, d.temp_ref,
            DTYPE(tables.press_ref_trop), *integer, *fraction,
            np.int32(n), np.int32(tables.ntemp), np.int32(tables.npres)))
    return _InterpolationMetadata(*integer, *fraction)


def _normalize_interpolation_metadata(metadata, shape):
    import cupy as cp

    def profile(value, dtype, name):
        out = cp.ascontiguousarray(cp.asarray(value, dtype=dtype))
        if out.shape != shape:
            raise ValueError(
                f"interpolation metadata {name} must have shape {shape}, "
                f"got {out.shape}")
        return out

    if not isinstance(metadata, _InterpolationMetadata):
        raise TypeError("metadata must be driver-owned interpolation metadata")
    return _InterpolationMetadata(
        profile(metadata.iatm, cp.int32, "iatm"),
        profile(metadata.jt, cp.int32, "jt"),
        profile(metadata.jp, cp.int32, "jp"),
        profile(metadata.ftemp, DTYPE, "ftemp"),
        profile(metadata.fpress, DTYPE, "fpress"))


def gas_optics(tables: GasTables, play, plev, tlay, vmr) -> GasOpticsResult:
    """Compute FP32 gas optical properties on device.

    ``vmr`` has shape ``(ncol,nlay,ngas+1)``; slot zero is reserved for dry
    air and ignored on input.  The CUDA transcription uses one thread per
    (column, layer) cell and retains the reference pressure/temperature/eta
    interpolation, all available minor gases, and SW Rayleigh scattering.
    """
    return _gas_optics(
        tables, play, plev, tlay, vmr, metadata=None, validate=True,
        zero_g_sentinel=False, out=None, col_dry_out=None)


def _gas_optics(tables: GasTables, play, plev, tlay, vmr, *, metadata,
                validate, zero_g_sentinel, out=None,
                col_dry_out=None, compute_col_dry=True) -> GasOpticsResult:
    """Internal gas optics path supporting one-call shared metadata."""
    import cupy as cp
    from woof.core.kernels import get_kernel

    play = cp.ascontiguousarray(cp.asarray(play, dtype=DTYPE))
    if play.ndim != 2:
        raise ValueError("play must be a 2-D (ncol,nlay) array")
    ncol, nlay = play.shape
    plev = _device_profile(plev, (ncol, nlay + 1), "plev")
    tlay = _device_profile(tlay, (ncol, nlay), "tlay")
    vmr = _device_profile(vmr, (ncol, nlay, tables.ngas + 1), "vmr")
    if validate:
        _validate_host_range("play", play, float(np.min(tables.press_ref)),
                             float(np.max(tables.press_ref)), "Pa")
        _validate_host_range("plev", plev, 0.0, None, "Pa")
        _validate_host_range("tlay", tlay, float(np.min(tables.temp_ref)),
                             float(np.max(tables.temp_ref)), "K")
    if metadata is None:
        metadata = _interpolation_metadata(
            tables, play, tlay, validate=False)
    else:
        metadata = _normalize_interpolation_metadata(metadata, play.shape)
    d = tables.to_device()
    shape = (ncol, nlay, tables.ngpt)
    if out is None:
        tau = cp.empty(shape, dtype=DTYPE)
        ssa = (cp.empty_like(tau) if tables.kind == "sw"
               else cp.empty((1,), dtype=DTYPE))
    else:
        tau = _workspace_output(out[0], shape, "gas_tau")
        ssa = (_workspace_output(out[1], shape, "gas_ssa")
               if tables.kind == "sw" else tau)
    # LW has no Rayleigh array; pass a valid dummy pointer which is never read.
    rayleigh = (d.rayleigh if tables.rayleigh is not None
                else (tau if out is not None
                      else cp.empty((1,), dtype=DTYPE)))
    # Four warp-local cells share a block without sharing their values.
    threads = 128
    blocks = (int(ncol) * int(nlay) + 3) // 4
    # Each flavor weight is one aligned 16-byte shared vector. Round the
    # minor-scaling prefix to four floats so every warp's vector rows stay
    # aligned, including bands whose minor-entry count is not divisible by 4.
    minor_slots = (max(int(tables.minor_limits_gpt_lower.shape[0]),
                       int(tables.minor_limits_gpt_upper.shape[0])) + 3) & ~3
    shared = 16 * (minor_slots + 8 * int(tables.nflav))
    kernel = get_kernel("rrtmgp_gas", "rrtmgp_gas_optics")
    kernel((blocks,), (threads,), (
        play, plev, tlay, vmr, metadata.iatm, metadata.jt, metadata.jp,
        metadata.ftemp, metadata.fpress, d.vmr_ref, d.flavor,
        d.gpoint_flavor, d.kmajor, d.kminor_lower, d.kminor_upper,
        d.minor_limits_gpt_lower, d.minor_limits_gpt_upper,
        d.minor_scales_with_density_lower,
        d.minor_scales_with_density_upper, d.scale_by_complement_lower,
        d.scale_by_complement_upper, d.idx_minor_lower, d.idx_minor_upper,
        d.idx_minor_scaling_lower, d.idx_minor_scaling_upper,
        d.kminor_start_lower, d.kminor_start_upper,
        d.minor_gpt_start_lower, d.minor_gpt_list_lower,
        d.minor_gpt_start_upper, d.minor_gpt_list_upper,
        rayleigh, tau, ssa,
        np.int32(ncol), np.int32(nlay), np.int32(tables.ngas),
        np.int32(tables.nflav), np.int32(tables.ngpt),
        np.int32(tables.ntemp), np.int32(tables.npres),
        np.int32(tables.neta),
        np.int32(tables.kminor_lower.shape[2]),
        np.int32(tables.kminor_upper.shape[2]),
        np.int32(tables.gas_index["h2o"]),
        np.int32(tables.kind == "sw"),
        np.int32(tables.minor_limits_gpt_lower.shape[0]),
        np.int32(tables.minor_limits_gpt_upper.shape[0])),
        shared_mem=shared)
    col_dry = None
    if compute_col_dry:
        # get_layer_number (mo_gas_optics_utils.F90), operator for
        # operator and in its association; each CuPy operator is one
        # rounded float32 operation, as the kernels' rrtmgp_col_dry is.
        h2o = vmr[:, :, tables.gas_index["h2o"]]
        fact = DTYPE(1.0) / (DTYPE(1.0) + h2o)
        m_air = (DTYPE(0.028964) + DTYPE(0.018016) * h2o) * fact
        if WRF_EXACT:
            col_dry_value = (DTYPE(10.0) * cp.abs(plev[:, :-1] - plev[:, 1:])
                             * DTYPE(6.02214076e23) * fact
                             / (DTYPE(1000.0) * m_air * DTYPE(100.0)
                                * DTYPE(9.80665)))
        else:
            col_dry_value = (cp.abs(plev[:, 1:] - plev[:, :-1])
                             * DTYPE(6.02214076e23) * fact
                             / (DTYPE(10000.0) * m_air * DTYPE(9.80665)))
        if col_dry_out is None:
            col_dry = col_dry_value
        else:
            col_dry = _workspace_output(
                col_dry_out, (ncol, nlay), "col_dry")
            col_dry[...] = col_dry_value
    if tables.kind == "lw":
        return GasOpticsResult(tau=tau, col_dry=col_dry)
    # The zero-field sentinel is private to the production fused finalizer.
    # Exported gas_optics retains its historical allocated zero array.
    asym = None if zero_g_sentinel else cp.zeros_like(tau)
    return GasOpticsResult(tau=tau, ssa=ssa, g=asym, col_dry=col_dry)


def delta_scale(tau, ssa, g):
    """Return reference-default delta-scaled FP32 two-stream properties."""
    import cupy as cp
    from woof.core.kernels import get_kernel

    tau = cp.ascontiguousarray(cp.asarray(tau, dtype=DTYPE))
    ssa = _device_profile(ssa, tau.shape, "ssa")
    g = _device_profile(g, tau.shape, "g")
    out = tuple(cp.empty_like(tau) for _ in range(3))
    n = tau.size
    threads = 256
    get_kernel("rrtmgp_rte", "rrtmgp_delta_scale")(
        ((n + threads - 1) // threads,), (threads,),
        (tau, ssa, g, *out, np.int32(n)))
    return out



#: WRF's top/bottom boundary fluxes off one chunk's RTE solves (levels run
#: bottom to top, top_at_1=False): index -1 is the top of the radiation
#: column, above-model layers included, as RRTMG_LWRAD/RRTMG_SWRAD publish
#: SWUPT/LWDNT (module_ra_rrtmg_lw.F and _sw.F, level kte+2), and index 0
#: the surface.  (suffix, from the clear-sky solve, flux, level).
_BOUNDARY_FLUX_TAKES = (
    ("upt", False, "flux_up", -1), ("uptc", True, "flux_up", -1),
    ("dnt", False, "flux_dn", -1), ("dntc", True, "flux_dn", -1),
    ("upb", False, "flux_up", 0), ("upbc", True, "flux_up", 0),
    ("dnb", False, "flux_dn", 0), ("dnbc", True, "flux_dn", 0))


def _store_boundary_fluxes(rows, side, columns, allsky, clear, *,
                           daylight=None, xp):
    """Write one chunk's eight boundary fluxes for ``side`` ("lw"/"sw").

    ``daylight`` masks dark shortwave columns to the radiation driver's
    zero on the full-column path; compacted daylit columns pass None.
    """
    for suffix, clear_sky, key, level in _BOUNDARY_FLUX_TAKES:
        value = getattr(clear if clear_sky else allsky, key)[:, level]
        if daylight is not None:
            value = xp.where(daylight, value, DTYPE(0.0))
        rows[side + suffix][columns] = value


def _clear_sky_cloud(tables, ncol, nlay):
    """Zero band cloud optics for the clear-sky RTE solve.

    Handed to the fused solvers, zero cloud extinction leaves the gas
    optics alone (the SW combine gives the gas tau and ssa and zero
    asymmetry), so the clear-sky solve needs no g-point asymmetry cube.
    One (ncol, nlay, nband) zero array serves tau, ssa and g.
    """
    import cupy as cp
    zero = cp.zeros((ncol, nlay, tables.nband), dtype=DTYPE)
    return FusedCloudOptics(tables=tables,
                            cloud=CloudOpticsResult(zero, zero, zero),
                            mask=None)


class FusedCloudOptics(NamedTuple):
    """Band cloud optics + McICA mask, combined INSIDE the RTE solvers.

    Two fusions failed here for named reasons: gas+finalize forced a
    pipeline reorder (13.5f), and Planck->solver dragged scattered gathers
    into a deliberately low-occupancy kernel (13.6f).  This one has
    neither: nothing reorders, and the solver's added reads are coalesced
    streams of the same element index it already walks -- band arrays 16x
    smaller than the cube, the mask a byte plane.  The finalize kernels
    and their g-point cube round trip then do not exist on this path.
    """

    tables: "GasTables"
    cloud: "CloudOpticsResult"
    mask: object  # bool (ncol,nlay,ngpt) McICA mask, or None


def _fused_cloud_args(fused, tau, ncol, nlay, ngpt):
    """Resolve FusedCloudOptics into the solver's trailing fz arguments.

    Validation mirrors _finalize_cloud_optics.  When ``fused`` is None the
    fz pointers are the tau array itself -- the same valid-dummy-pointer
    convention that launcher already uses for an absent mask -- and
    ``fz_on`` is 0, so the kernel never dereferences them.
    """
    import cupy as cp

    if fused is None:
        z = np.int32(0)
        return (tau, tau, tau, tau, tau, z, z, z)
    tables, cloud = fused.tables, fused.cloud
    if ngpt != tables.ngpt:
        raise ValueError("fused cloud optics do not match the gas table")
    band_shape = (ncol, nlay, tables.nband)
    cld_tau = _device_profile(cloud.tau, band_shape, "cloud.tau")
    cld_ssa = _device_profile(cloud.ssa, band_shape, "cloud.ssa")
    cld_asy = _device_profile(cloud.g, band_shape, "cloud.g")
    if fused.mask is None:
        mask, have_mask = tau, 0
    else:
        mask = cp.ascontiguousarray(cp.asarray(fused.mask, dtype=cp.bool_))
        if mask.shape != (ncol, nlay, ngpt):
            raise ValueError(
                f"fused cloud mask must have shape {(ncol, nlay, ngpt)}, "
                f"got {mask.shape}")
        have_mask = 1
    bands = tables.to_device().gpoint_bands
    return (cld_tau, cld_ssa, cld_asy, bands, mask,
            np.int32(tables.nband), np.int32(have_mask), np.int32(1))



class PlanckInputs(NamedTuple):
    """What the LW solver needs to derive its own Planck sources.

    Passing this instead of `lay_source`/`lev_source`/`sfc_source` deletes
    all three from the workspace -- 455 MiB of the binding phase at the
    default chunk -- and the round trip an ablation priced at 38% of the
    solver.  See `rrtmgp_planck_common.cuh` for why it costs no array:
    `pfrac` does not chain.
    """

    tables: "GasTables"
    play: object
    plev: object
    tlay: object
    tlev: object
    tsfc: object
    vmr: object
    metadata: object


def _planck_solver_args(planck, tau, ncol, nlay, ngpt):
    """Resolve PlanckInputs into the solver's trailing pk arguments.

    ``None`` yields the valid-dummy-pointer convention the fz arguments
    already use: the pointers are the tau array and ``pk_on`` is 0, so the
    kernel never dereferences them and reads lay/lev/sfc_source instead.
    """
    import cupy as cp

    if planck is None:
        z = np.int32(0)
        # Strict unit: 18 pointer parameters then 8 integers; the 2.9.1
        # default unit (rrtmgp_rte_default.cu, no plev and no H2O index):
        # 17 then 7.  The last is pk_on = 0.
        if not WRF_EXACT:
            return (tau,) * 17 + (z,) * 7
        return (tau,) * 18 + (z,) * 8
    t = planck.tables
    if t.kind != "lw":
        raise ValueError("in-solver Planck requires LW gas tables")
    if ngpt != t.ngpt:
        raise ValueError("Planck inputs do not match the gas table")
    d = t.to_device()
    play = _device_profile(planck.play, (ncol, nlay), "planck.play")
    plev = _device_profile(planck.plev, (ncol, nlay + 1), "planck.plev")
    tlay = _device_profile(planck.tlay, (ncol, nlay), "planck.tlay")
    tlev = _device_profile(planck.tlev, (ncol, nlay + 1), "planck.tlev")
    tsfc = _device_profile(planck.tsfc, (ncol,), "planck.tsfc")
    vmr = cp.ascontiguousarray(cp.asarray(planck.vmr, dtype=DTYPE))
    if vmr.shape != (ncol, nlay, t.ngas + 1):
        raise ValueError(
            f"planck.vmr must have shape {(ncol, nlay, t.ngas + 1)}, "
            f"got {vmr.shape}")
    m = planck.metadata
    if not WRF_EXACT:
        # The 2.9.1 default unit interpolates the Planck fractions in eta
        # built from play alone: no plev, no H2O index.
        return (play, tlay, tlev, tsfc, vmr, m.iatm, m.jt, m.jp, m.ftemp,
                m.fpress, d.temp_ref, d.vmr_ref, d.flavor, d.gpoint_flavor,
                d.gpoint_bands, d.planck_fraction, d.totplnk,
                np.int32(t.ngas), np.int32(t.ntemp), np.int32(t.npres),
                np.int32(t.neta), np.int32(t.nband),
                np.int32(t.totplnk.shape[0]), np.int32(1))
    return (play, plev, tlay, tlev, tsfc, vmr, m.iatm, m.jt, m.jp, m.ftemp,
            m.fpress, d.temp_ref, d.vmr_ref, d.flavor, d.gpoint_flavor,
            d.gpoint_bands, d.planck_fraction, d.totplnk,
            np.int32(t.ngas), np.int32(t.ntemp), np.int32(t.npres),
            np.int32(t.neta), np.int32(t.nband),
            np.int32(t.totplnk.shape[0]), np.int32(t.gas_index["h2o"]),
            np.int32(1))


def lw_rte(tau, lay_source, lev_source, sfc_source, sfc_emis,
           incident_flux=None, *, top_at_1: bool) -> FluxResult:
    """Run the FP32 one-angle LW no-scattering solver on device."""
    return _lw_rte(
        tau, lay_source, lev_source, sfc_source, sfc_emis,
        incident_flux=incident_flux, top_at_1=top_at_1,
        out=None, incident_out=None)


def _rte_kernel(func: str, nlay: int):
    """One RTE kernel, compiled for this run's layer count.

    The column arrays are declared ``[RRTMGP_MAX_LAYERS]``; leaving that at
    the 128-layer ceiling would spend 40% of the frame on layers a 74-layer
    column never touches, and the frame is exactly what
    :func:`_rte_gpt_tile` divides the L2 budget by.  So the waste costs
    g-point parallelism, not merely bytes.
    """
    from woof.core.kernels import get_kernel_int_defines
    return get_kernel_int_defines(
        "rrtmgp_rte", func, (("RRTMGP_MAX_LAYERS", int(nlay)),))


@cuda_cache(maxsize=None)
def _rte_sm_count() -> int:
    import cupy as cp
    return int(cp.cuda.Device().attributes["MultiProcessorCount"])


#: Threads per SM to aim for in the RTE stage kernels -- one 256-thread block.
_RTE_BLOCK = 256

#: A fold group is one block, so it cannot exceed the maximum block width.
_RTE_MAX_FOLD_BLOCK = 1024


def _rte_gpt_tile(kernel, ncol: int, ngpt: int, *,
                  fold: bool = False, longwave: bool = False) -> int:
    """Select buffered fold widths without querying the card SM count.

    Buffered folds use 128 LW g-points and 32 SW g-points (ngpt 224 admits
    no power of two above 32). Narrow bands use smaller full tiles. The
    helper mode fold=False retains coverage based on the frame.

    Measured on an RTX PRO 4500 Blackwell (82 SMs), 288 x 288 x 59, CUPTI
    ms per radiation call, every width byte-identical:

        LW width      16      32      64     128
        LW ms      135.9   114.4    87.0    76.6

    and SW at width 32 with 1, 2 or 4 columns per block: 123.2, 138.1,
    123.8 ms (width 16: 145.6).

    Measurement-only overrides are read once per solver call, before its
    tile loop. WOOF_RTE_TILE_WIDTH overrides either band; the optional
    WOOF_RTE_LW_TILE_WIDTH or WOOF_RTE_SW_TILE_WIDTH takes precedence.
    Widths must be integers from 1 through 1024 and are capped by the band
    length. WOOF_RTE_FOLD accepts auto, 0 or 1; 0 keeps that width and
    uses partials/reduce, while 1 requires a full tile of at least 16 and enough shared
    memory. WOOF_RTE_COLUMNS_PER_BLOCK overrides the requested packing;
    resource limits and complete wide groups still apply. These controls
    only partition an unchanged ascending sum. WOOF_RTE_TILE_COVERAGE
    remains a measurement-only override for the non-folding tile rule.
    """
    forced = os.environ.get("WOOF_RTE_TILE_WIDTH", "").strip()
    band_forced = os.environ.get(
        "WOOF_RTE_LW_TILE_WIDTH" if longwave else "WOOF_RTE_SW_TILE_WIDTH",
        "").strip()
    forced = band_forced or forced
    if forced:
        tile = int(forced)
        if not 1 <= tile <= _RTE_MAX_FOLD_BLOCK:
            raise ValueError("WOOF_RTE_TILE_WIDTH must be from 1 through 1024")
        return min(tile, int(ngpt))
    if fold:
        tile = 128 if longwave else 32
        while tile > int(ngpt) or int(ngpt) % tile:
            tile //= 2
        return max(1, tile)
    frame = 0
    if kernel is not None:
        try:
            frame = int(kernel.attributes["local_size_bytes"])
        except Exception:                                # pragma: no cover
            frame = 0
    coverage = 2 if 0 < frame <= 1024 else 1
    forced = os.environ.get("WOOF_RTE_TILE_COVERAGE", "").strip()
    if forced:
        coverage = max(1, int(forced))
    target = _rte_sm_count() * _RTE_BLOCK * coverage
    tile, cap = 1, min(max(1, target // max(1, int(ncol))), int(ngpt))
    while tile * 2 <= cap:
        tile *= 2
    return max(1, min(tile, int(ngpt)))


def _rte_launch(tile, ncol, ngpt, default_columns):
    """Shared LW/SW fold validation, packing and launch dimensions."""
    forced_fold = os.environ.get("WOOF_RTE_FOLD", "auto").strip().lower()
    forced_columns = os.environ.get("WOOF_RTE_COLUMNS_PER_BLOCK", "").strip()
    if forced_fold not in ("auto", "0", "1"):
        raise ValueError("WOOF_RTE_FOLD must be auto, 0 or 1")
    requested_columns = int(forced_columns) if forced_columns else default_columns
    if int(requested_columns) < 1:
        raise ValueError("WOOF_RTE_COLUMNS_PER_BLOCK must be positive")
    warp_fold = 1 if (tile >= 16 and ngpt % tile == 0
                      and tile <= _RTE_MAX_FOLD_BLOCK
                      and 16 * (tile + 1) * 4 + 16 * 8 <= 49152) else 0
    if forced_fold == "1" and not warp_fold:
        raise ValueError("WOOF_RTE_FOLD=1 requires a full tile of at least 16 "
                         "within the shared-memory limit")
    if forced_fold == "0":
        warp_fold = 0
    if warp_fold:
        columns = max(1, min(int(requested_columns), 1024 // tile,
                             49152 // (16 * (tile + 1) * 4 + 16 * 8)))
        # Wide groups use block barriers, so their last block must be full.
        if tile not in (16, 32) and ncol % columns:
            columns = 1
        threads = tile * columns
        stage_blocks = (ncol + columns - 1) // columns
        shared = columns * (16 * (tile + 1) * 4 + 16 * 8)
    else:
        threads = _RTE_BLOCK
        stage_blocks = (ncol * tile + threads - 1) // threads
        shared = 0
    return warp_fold, threads, stage_blocks, shared


def _lw_rte(tau, lay_source, lev_source, sfc_source, sfc_emis,
            incident_flux=None, *, top_at_1: bool, out,
            incident_out, fused_cloud=None,
            planck=None, _fold_columns=2,
            with_clear: bool = False, planck_staged: bool = False):
    import cupy as cp

    tau = cp.ascontiguousarray(cp.asarray(tau, dtype=DTYPE))
    if tau.ndim != 3:
        raise ValueError("tau must have shape (ncol,nlay,ngpt)")
    ncol, nlay, ngpt = tau.shape
    if nlay > 128:
        # The ceiling's owner is the Planck-source kernel's fixed
        # pfrac[128] (see _planck_sources), not the RTE: the solver
        # kernels compile RRTMGP_MAX_LAYERS to this run's nlay.  This
        # door-level copy stays because the LW chain feeds on Planck
        # sources either way (fused or precomputed), and a taller column
        # must refuse HERE, before any device allocation, rather than
        # inside whichever stage first touches the fixed array
        # (re-justified, stale-guard audit 2026-08-25).
        raise ValueError(
            "RRTMGP CUDA LW supports at most 128 layers (the Planck-"
            f"source kernel's fixed pfrac[128]); got {nlay}")
    if planck is None:
        lay_source = _device_profile(lay_source, tau.shape, "lay_source")
        lev_source = _device_profile(
            lev_source, (ncol, nlay + 1, ngpt), "lev_source")
        sfc_source = _device_profile(sfc_source, (ncol, ngpt), "sfc_source")
    else:
        # Derived in the kernel; these are valid dummy pointers it never
        # dereferences, the same convention the fz arguments use.
        lay_source = lev_source = sfc_source = tau
    sfc_emis = _device_profile(sfc_emis, (ncol, ngpt), "sfc_emis")
    if incident_flux is None:
        if incident_out is None:
            incident = cp.zeros((ncol, ngpt), dtype=DTYPE)
        else:
            incident = _workspace_output(
                incident_out, (ncol, ngpt), "incident")
            incident.fill(DTYPE(0.0))
    else:
        incident = _device_profile(
            incident_flux, (ncol, ngpt), "incident_flux")
    flux_shape = (ncol, nlay + 1)
    if out is None:
        up = cp.empty(flux_shape, dtype=DTYPE)
        down = cp.empty_like(up)
    else:
        up = _workspace_output(out[0], flux_shape, "flux_up")
        down = _workspace_output(out[1], flux_shape, "flux_dn")
    fz = _fused_cloud_args(fused_cloud, tau, ncol, nlay, ngpt)
    pk = _planck_solver_args(planck, tau, ncol, nlay, ngpt)
    stage = _rte_kernel("rrtmgp_lw_noscat", nlay)
    fold = _rte_kernel("rrtmgp_lw_flux_reduce", nlay)
    nlev = nlay + 1
    tile = _rte_gpt_tile(stage, ncol, ngpt, fold=True, longwave=True)
    # Full g-point tiles preserve the ascending fold. Each packed group
    # owns a separate shared buffer. A 16- or 32-thread group synchronizes
    # its lanes, so a final block may contain unused groups. Wider groups
    # use block barriers and require a full final block.
    warp_fold, threads, stage_blocks, shared = _rte_launch(
        tile, ncol, ngpt, _fold_columns)
    if warp_fold:
        part_up, part_dn = up, down
    else:
        part_up = cp.empty((tile, nlev, ncol), dtype=DTYPE)
        part_dn = cp.empty_like(part_up)
    fold_blocks = (ncol * nlev + _RTE_BLOCK - 1) // _RTE_BLOCK
    if with_clear:
        # All-sky and clear-sky in one pass (rrtmgp_lw_noscat_pair): the
        # clear solve on gas tau alone shares this solve's tau and Planck
        # reads, and its layer terms wherever the McICA cloud adds nothing,
        # and folds only the top and surface levels it is read at.  Default
        # unit, folding path, fused cloud only; anything else takes the two
        # separate solves, which give the same words.
        if fused_cloud is None:
            raise ValueError("with_clear needs fused cloud optics")
        if not warp_fold or WRF_EXACT:
            if planck is not None and planck_staged:
                # The staged words, from the standalone kernel itself.
                src = _planck_sources(
                    planck.tables, planck.play, planck.plev, planck.tlay,
                    planck.tlev, planck.tsfc, planck.vmr,
                    metadata=planck.metadata, validate=False)
                lay_source, lev_source, sfc_source = (
                    src.lay_source, src.lev_source, src.sfc_source)
                planck = None
            clear = _lw_rte(
                tau, lay_source, lev_source, sfc_source, sfc_emis,
                incident_flux, top_at_1=top_at_1, out=None,
                incident_out=None, planck=planck,
                _fold_columns=_fold_columns)
            flux = _lw_rte(
                tau, lay_source, lev_source, sfc_source, sfc_emis,
                incident_flux, top_at_1=top_at_1, out=out,
                incident_out=incident_out, fused_cloud=fused_cloud,
                planck=planck, _fold_columns=_fold_columns)
            return flux, clear
        pair = _rte_kernel("rrtmgp_lw_noscat_pair", nlay)
        if planck is not None and planck_staged:
            pk = pk[:-1] + (np.int32(2),)
        clr_up = cp.empty(flux_shape, dtype=DTYPE)
        clr_dn = cp.empty_like(clr_up)
        for gpt0 in range(0, ngpt, tile):
            pair((stage_blocks,), (threads,), (
                tau, lay_source, lev_source, sfc_source, sfc_emis, incident,
                up, down, clr_up, clr_dn, np.int32(ncol), np.int32(nlay),
                np.int32(ngpt), np.int32(gpt0), np.int32(tile),
                np.int32(top_at_1)) + fz + pk,
                shared_mem=shared)
        return FluxResult(up, down), FluxResult(clr_up, clr_dn)
    # Tiles are walked in ascending g-point order and each fold appends to the
    # running flux, so the FP32 addition sequence is the one the single-thread
    # column loop produced -- see rrtmgp_rte.cu.
    for gpt0 in range(0, ngpt, tile):
        stage((stage_blocks,), (threads,), (
            tau, lay_source, lev_source, sfc_source, sfc_emis, incident,
            part_up, part_dn, np.int32(ncol), np.int32(nlay),
            np.int32(ngpt), np.int32(gpt0), np.int32(tile),
            np.int32(top_at_1), np.int32(warp_fold)) + fz + pk,
            shared_mem=shared)
        if not warp_fold:
            fold((fold_blocks,), (_RTE_BLOCK,), (
                part_up, part_dn, up, down, np.int32(ncol), np.int32(nlev),
                np.int32(gpt0), np.int32(min(tile, ngpt - gpt0)))
                + ((np.int32(gpt0 + tile >= ngpt),) if WRF_EXACT else ()))
    return FluxResult(up, down)


def sw_rte(tau, ssa, g, mu0, sfc_alb_dir, sfc_alb_dif, inc_flux_dir,
           *, top_at_1: bool) -> FluxResult:
    """Run the FP32 delta-scaled PIFM two-stream SW solver on device."""
    return _sw_rte(
        tau, ssa, g, mu0, sfc_alb_dir, sfc_alb_dif, inc_flux_dir,
        top_at_1=top_at_1, out=None)


def _sw_rte(tau, ssa, g, mu0, sfc_alb_dir, sfc_alb_dif, inc_flux_dir,
            *, top_at_1: bool, out, fused_cloud=None,
            _fold_columns=1, with_clear: bool = False):
    import cupy as cp

    tau = cp.ascontiguousarray(cp.asarray(tau, dtype=DTYPE))
    if tau.ndim != 3:
        raise ValueError("tau must have shape (ncol,nlay,ngpt)")
    ncol, nlay, ngpt = tau.shape
    # No layer ceiling: the SW solver compiles RRTMGP_MAX_LAYERS to this
    # run's nlay and no kernel on the SW chain holds fixed per-layer
    # thread storage.  A 128-layer refusal used to sit here, inherited
    # from the days before per-run specialization; the ceiling's real
    # owner is the LW Planck-source kernel's pfrac[128], which has its
    # own raise (retired here 2026-08-25, stale-guard audit -- proven by
    # a 160-layer transparent-atmosphere run reproducing the analytic
    # direct beam on the measurement card).
    ssa = _device_profile(ssa, tau.shape, "ssa")
    # On the fused path the asym cube does not exist; the ssa array is the
    # valid-dummy pointer for the never-read parameter.
    g = (ssa if (fused_cloud is not None and g is None)
         else _device_profile(g, tau.shape, "g"))
    mu0 = cp.asarray(mu0, dtype=DTYPE)
    if mu0.shape == (ncol,):
        mu0 = cp.broadcast_to(mu0[:, None], (ncol, nlay))
    mu0 = _device_profile(mu0, (ncol, nlay), "mu0")
    alb_dir = _device_profile(sfc_alb_dir, (ncol, ngpt), "sfc_alb_dir")
    alb_dif = _device_profile(sfc_alb_dif, (ncol, ngpt), "sfc_alb_dif")
    inc = _device_profile(inc_flux_dir, (ncol, ngpt), "inc_flux_dir")
    flux_shape = (ncol, nlay + 1)
    if out is None:
        up = cp.empty(flux_shape, dtype=DTYPE)
        down = cp.empty_like(up)
        direct = cp.empty_like(up)
    else:
        up, down, direct = (
            _workspace_output(value, flux_shape, name)
            for value, name in zip(
                out, ("flux_up", "flux_dn", "flux_dir")))
    fz = _fused_cloud_args(fused_cloud, tau, ncol, nlay, ngpt)
    stage = _rte_kernel("rrtmgp_sw_2stream", nlay)
    fold = _rte_kernel("rrtmgp_sw_flux_reduce", nlay)
    nlev = nlay + 1
    tile = _rte_gpt_tile(stage, ncol, ngpt, fold=True)
    # Each column owns a padded buffer of 16 independent flux values.
    # The fold retains ascending g-point order for every value. Whole
    # unused groups in the final block can return before their lane barriers.
    warp_fold, threads, stage_blocks, shared = _rte_launch(
        tile, ncol, ngpt, _fold_columns)
    if warp_fold:
        part_up, part_dn, part_dir = up, down, direct
    else:
        part_up = cp.empty((tile, nlev, ncol), dtype=DTYPE)
        part_dn = cp.empty_like(part_up)
        part_dir = cp.empty_like(part_up)
    fold_blocks = (ncol * nlev + _RTE_BLOCK - 1) // _RTE_BLOCK
    if with_clear:
        # All-sky and clear-sky in one pass (rrtmgp_sw_2stream_pair): the
        # clear solve (zero band cloud) shares the gas optics reads and the
        # layer coefficients wherever the McICA cloud adds nothing, and
        # folds only the top and surface words it is read at.  Default
        # unit, folding path, bottom-to-top, fused cloud only; anything
        # else takes the two separate solves, which give the same words.
        if fused_cloud is None:
            raise ValueError("with_clear needs fused cloud optics")
        if not warp_fold or WRF_EXACT or top_at_1:
            clear = _sw_rte(
                tau, ssa, None, mu0, sfc_alb_dir, sfc_alb_dif, inc_flux_dir,
                top_at_1=top_at_1, out=None,
                fused_cloud=_clear_sky_cloud(fused_cloud.tables, ncol, nlay),
                _fold_columns=_fold_columns)
            flux = _sw_rte(
                tau, ssa, None, mu0, sfc_alb_dir, sfc_alb_dif, inc_flux_dir,
                top_at_1=top_at_1, out=out, fused_cloud=fused_cloud,
                _fold_columns=_fold_columns)
            return flux, clear
        pair = _rte_kernel("rrtmgp_sw_2stream_pair", nlay)
        clr_up = cp.empty(flux_shape, dtype=DTYPE)
        clr_dn = cp.empty_like(clr_up)
        # fz minus its trailing fz_on flag, then the runtime zero the clear
        # combine takes for tc, wc and gc (the zero band arrays' words).
        for gpt0 in range(0, ngpt, tile):
            pair((stage_blocks,), (threads,), (
                tau, ssa, mu0, alb_dir, alb_dif, inc,
                up, down, direct, clr_up, clr_dn,
                np.int32(ncol), np.int32(nlay), np.int32(ngpt),
                np.int32(gpt0), np.int32(tile)) + fz[:-1] + (DTYPE(0.0),),
                shared_mem=shared)
        return FluxResult(up, down, direct), FluxResult(clr_up, clr_dn)
    for gpt0 in range(0, ngpt, tile):
        stage((stage_blocks,), (threads,), (
            tau, ssa, g, mu0, alb_dir, alb_dif, inc,
            part_up, part_dn, part_dir,
            np.int32(ncol), np.int32(nlay), np.int32(ngpt),
            np.int32(gpt0), np.int32(tile), np.int32(top_at_1),
            np.int32(0), np.int32(warp_fold)) + fz, shared_mem=shared)
        if not warp_fold:
            fold((fold_blocks,), (_RTE_BLOCK,), (
                part_up, part_dn, part_dir, up, down, direct,
                np.int32(ncol), np.int32(nlev),
                np.int32(gpt0), np.int32(min(tile, ngpt - gpt0))))
    return FluxResult(up, down, direct)


def planck_sources(tables: GasTables, play, plev, tlay, tlev, tsfc,
                   vmr) -> PlanckSourceResult:
    """Compute FP32 RRTMGP LW Planck sources on device."""
    return _planck_sources(
        tables, play, plev, tlay, tlev, tsfc, vmr,
        metadata=None, validate=True, out=None)


def _planck_sources(tables: GasTables, play, plev, tlay, tlev, tsfc,
                    vmr, *, metadata, validate, out=None) -> PlanckSourceResult:
    """Internal Planck path supporting driver-owned shared metadata."""
    import cupy as cp
    from woof.core.kernels import get_kernel

    if tables.kind != "lw":
        raise ValueError("Planck sources require LW gas tables")
    play = cp.ascontiguousarray(cp.asarray(play, dtype=DTYPE))
    if play.ndim != 2:
        raise ValueError("play must have shape (ncol,nlay)")
    ncol, nlay = play.shape
    if nlay > 128:
        # The OWNER of the RRTMGP layer ceiling: this kernel holds a fixed
        # per-thread pfrac[128] and its device-side guard is a bare
        # `return`, so without this raise a taller column came back as
        # uninitialized source arrays that read as valid radiation
        # (stale-guard audit 2026-08-25).  The RTE solvers themselves
        # compile RRTMGP_MAX_LAYERS to this run's nlay and carry no such
        # ceiling.
        raise ValueError(
            "RRTMGP CUDA Planck sources support at most 128 layers: the "
            "rrtmgp_planck_sources kernel holds a fixed pfrac[128] per "
            f"thread and this profile has {nlay} layers")
    # plev gives each layer's dry-air column: the Planck fractions are
    # interpolated in eta built from gas COLUMNS (vmr * col_dry), as the
    # gas optics are (compute_Planck_source reads interpolation()'s fmajor).
    plev = _device_profile(plev, (ncol, nlay + 1), "plev")
    tlay = _device_profile(tlay, (ncol, nlay), "tlay")
    tlev = _device_profile(tlev, (ncol, nlay + 1), "tlev")
    tsfc = _device_profile(tsfc, (ncol,), "tsfc")
    vmr = _device_profile(vmr, (ncol, nlay, tables.ngas + 1), "vmr")
    if validate:
        _validate_host_range("play", play, float(np.min(tables.press_ref)),
                             float(np.max(tables.press_ref)), "Pa")
        _validate_host_range("plev", plev, 0.0, None, "Pa")
        for name, value in (("tlay", tlay), ("tlev", tlev), ("tsfc", tsfc)):
            _validate_host_range(name, value, float(np.min(tables.temp_ref)),
                                 float(np.max(tables.temp_ref)), "K")
    if metadata is None:
        metadata = _interpolation_metadata(
            tables, play, tlay, validate=False)
    else:
        metadata = _normalize_interpolation_metadata(metadata, play.shape)
    d = tables.to_device()
    shapes = ((ncol, nlay, tables.ngpt),
              (ncol, nlay + 1, tables.ngpt),
              (ncol, tables.ngpt))
    if out is None:
        lay = cp.empty(shapes[0], dtype=DTYPE)
        lev = cp.empty(shapes[1], dtype=DTYPE)
        sfc = cp.empty(shapes[2], dtype=DTYPE)
    else:
        lay, lev, sfc = (
            _workspace_output(value, shape, name)
            for value, shape, name in zip(
                out, shapes, ("lay_source", "lev_source", "sfc_source")))
    # One thread per (column, g-point); see the kernel for why the g-point
    # loop it replaces was independent.
    threads = 128
    cells = int(ncol) * int(tables.ngpt)
    if WRF_EXACT:
        head = (play, plev, tlay)
        tail = (np.int32(tables.totplnk.shape[0]),
                np.int32(tables.gas_index["h2o"]))
    else:
        # The 2.9.1 default unit's signature: no plev, no H2O index.
        head = (play, tlay)
        tail = (np.int32(tables.totplnk.shape[0]),)
    get_kernel("rrtmgp_gas", "rrtmgp_planck_sources")(
        ((cells + threads - 1) // threads,), (threads,),
        head + (tlev, tsfc, vmr, metadata.iatm, metadata.jt,
         metadata.jp, metadata.ftemp, metadata.fpress, d.temp_ref,
         d.vmr_ref, d.flavor,
         d.gpoint_flavor, d.gpoint_bands, d.planck_fraction, d.totplnk,
         lay, lev, sfc, np.int32(ncol), np.int32(nlay),
         np.int32(tables.ngas), np.int32(tables.ngpt),
         np.int32(tables.ntemp), np.int32(tables.npres),
         np.int32(tables.neta), np.int32(tables.nband)) + tail)
    return PlanckSourceResult(lay, lev, sfc)


def _rfmip_profiles(tables, sites, experiments, inputs=None):
    from woof.core.rfmip_upstream import fetch_rfmip

    sites = np.asarray(sites, dtype=np.intp)
    experiments = np.asarray(experiments, dtype=np.intp)
    source = fetch_rfmip("rfmip-clear-sky-inputs.nc", path=inputs)
    with _open_table(source, prefetch=False) as nc:
        nc.set_auto_mask(False)
        nc.prefetch(list(nc.variables))
        nsite, nexp = sites.size, experiments.size
        play_site = np.asarray(nc.variables["pres_layer"][sites], np.float64)
        plev_site = np.asarray(nc.variables["pres_level"][sites], np.float64)
        play = np.broadcast_to(play_site[None], (nexp, *play_site.shape))
        plev = np.broadcast_to(
            plev_site[None], (nexp, *plev_site.shape)).copy()
        # The RFMIP top boundary is 0.01 Pa, below the coefficient grid.
        # Match the upstream example's explicit input sanitization.
        top = 0 if play_site[0, 0] < play_site[0, -1] else -1
        plev[..., top] = tables.press_ref[-1] + np.finfo(np.float64).eps
        tlay = np.asarray(nc.variables["temp_layer"][experiments][:, sites], np.float64)
        tlev = np.asarray(nc.variables["temp_level"][experiments][:, sites], np.float64)
        tsfc = np.asarray(
            nc.variables["surface_temperature"][experiments][:, sites], np.float64)
        vmr = np.zeros((*tlay.shape, tables.ngas + 1), np.float64)
        vmr[..., tables.gas_index["h2o"]] = np.asarray(
            nc.variables["water_vapor"][experiments][:, sites], np.float64)
        vmr[..., tables.gas_index["o3"]] = np.asarray(
            nc.variables["ozone"][experiments][:, sites], np.float64)
        for gas, rfmip_name in _RFMIP_GAS_NAMES.items():
            variable = nc.variables[rfmip_name + "_GM"]
            scale = float(getattr(variable, "units", "1").replace(" ", ""))
            values = np.asarray(variable[experiments], np.float64) * scale
            vmr[..., tables.gas_index[gas]] = values[:, None, None]
        emiss = np.asarray(nc.variables["surface_emissivity"][sites], np.float64)
        albedo = np.asarray(nc.variables["surface_albedo"][sites], np.float64)
        sza = np.asarray(nc.variables["solar_zenith_angle"][sites], np.float64)
        tsi = np.asarray(nc.variables["total_solar_irradiance"][sites], np.float64)
    def flat(a):
        return np.ascontiguousarray(a.reshape(nexp * nsite, *a.shape[2:]))
    return (flat(play), flat(plev), flat(tlay), flat(tlev),
            np.ascontiguousarray(tsfc.reshape(-1)),
            flat(vmr), np.tile(emiss, nexp), np.tile(albedo, nexp),
            np.tile(sza, nexp), np.tile(tsi, nexp))


def rfmip_clear_sky(*, sites=None, experiments=None,
                    inputs=None) -> RFMIPResult:
    """Run the RFMIP clear-sky oracle profiles on the GPU.

    This reproduces the upstream physics-index-1/forcing-index-1 examples:
    one-angle LW, default solar spectrum normalized to each RFMIP TSI, and
    nighttime columns explicitly zeroed after the SW solve.

    The RFMIP input file is not shipped (see
    :mod:`woof.core.rfmip_upstream`): ``inputs`` names a local copy, and
    without it the pinned upstream file is fetched into the RFMIP cache.
    Either way its SHA-256 is verified before a byte is read.
    """
    import cupy as cp

    sites = np.arange(100) if sites is None else np.asarray(sites)
    experiments = (np.arange(18) if experiments is None
                   else np.asarray(experiments))
    lw = load_gas_tables("lw")
    (play, plev, tlay, tlev, tsfc, vmr, emiss, _albedo,
     _sza, _tsi) = _rfmip_profiles(lw, sites, experiments, inputs)
    dplay = cp.asarray(play, dtype=DTYPE)
    dplev = cp.asarray(plev, dtype=DTYPE)
    dtlay = cp.asarray(tlay, dtype=DTYPE)
    dvmr = cp.asarray(vmr, dtype=DTYPE)
    optics_lw = gas_optics(lw, dplay, dplev, dtlay, dvmr)
    sources = planck_sources(
        lw, dplay, dplev, dtlay, cp.asarray(tlev, dtype=DTYPE),
        cp.asarray(tsfc, dtype=DTYPE), dvmr)
    emis = cp.asarray(emiss, dtype=DTYPE)
    emis_band = cp.broadcast_to(emis[:, None], (play.shape[0], lw.nband))
    emis_gpt = _expand_band_to_gpoint(
        emis_band, lw, "RFMIP surface emissivity")
    lw_flux = lw_rte(optics_lw.tau, sources.lay_source,
                     sources.lev_source, sources.sfc_source, emis_gpt,
                     top_at_1=True)

    sw = load_gas_tables("sw")
    (play, plev, tlay, _tlev, _tsfc, vmr, _emiss, albedo,
     sza, tsi) = _rfmip_profiles(sw, sites, experiments, inputs)
    optics_sw = gas_optics(
        sw, cp.asarray(play, dtype=DTYPE), cp.asarray(plev, dtype=DTYPE),
        cp.asarray(tlay, dtype=DTYPE), cp.asarray(vmr, dtype=DTYPE))
    tau, ssa, asym = delta_scale(optics_sw.tau, optics_sw.ssa, optics_sw.g)
    mu_raw = cp.cos(cp.asarray(sza, dtype=DTYPE) * DTYPE(np.pi / 180.0))
    daylight = mu_raw > DTYPE(0.0)
    mu = cp.where(daylight, mu_raw, DTYPE(1.0))
    alb = cp.asarray(albedo, dtype=DTYPE)
    alb_gpt = cp.ascontiguousarray(cp.broadcast_to(
        alb[:, None], (play.shape[0], sw.ngpt)))
    inc = _normalized_solar_incident(sw.solar_source, tsi)
    sw_flux = sw_rte(tau, ssa, asym, mu, alb_gpt, alb_gpt, inc,
                     top_at_1=True)
    mask = daylight[:, None]
    sw_up = cp.where(mask, sw_flux.flux_up, DTYPE(0.0))
    sw_dn = cp.where(mask, sw_flux.flux_dn, DTYPE(0.0))
    return RFMIPResult(lw_flux.flux_up, lw_flux.flux_dn, sw_up, sw_dn)


def _normalized_solar_incident(solar_source, tsi):
    """Normalize a solar spectrum with a binding float64 host reduction."""
    import cupy as cp

    solar_host = np.asarray(solar_source, dtype=np.float64)
    norm = np.sum(solar_host, dtype=np.float64)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("solar-source normalization must be finite and positive")
    scale = np.asarray(tsi, dtype=np.float64) / norm
    solar = cp.asarray(solar_host, dtype=DTYPE)
    scale_device = cp.asarray(scale, dtype=DTYPE)
    return cp.ascontiguousarray(solar[None, :] * scale_device[:, None])


__all__ = ["CloudOpticsResult", "CloudTables", "DATA_DIR", "FluxResult",
           "GasOpticsResult", "GasTables", "HydrometeorPaths",
           "MCICA_PERMUTESEED_LW", "MCICA_PERMUTESEED_SW",
           "PlanckSourceResult", "RFMIPResult", "RRTMGPRadiation",
           "RRTMGP_TOA_PRESSURE_PA", "TRACE_CLIMATOLOGY_SHA256",
           "TRACE_CLIMATOLOGY_SOURCE", "TraceClimatology",
           "add_cloud_optics", "cal_cldfra1", "cal_cldfra2",
           "cloud_optics", "delta_scale", "gas_optics",
           "gas_table_temperature_range_k",
           "hydrometeor_paths", "load_cloud_tables", "load_gas_tables",
           "load_trace_climatology",
           "lw_rte", "mcica_cloud_masks", "planck_sources",
           "rfmip_clear_sky", "rrtmgp_above_model_layer_counts",
           "sw_rte", "trace_gases"]
