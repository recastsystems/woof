"""Warm start from a WRF-written restart file (``wrfrst_d0N_<time>``).

WRF 4.x writes its whole cross-step model state into a NetCDF restart file:
the ``_2`` time level of every prognostic, the moisture species and number
concentrations, the base state, the soil/snow/surface state, the held
physics tendencies, the precipitation accumulators and the clock.  This
module reads such a file, through the Rust NetCDF bridge like every other
WRF input, into the model's full state so a forecast continues from the
instant the WRF run stopped -- the warm-start half of the WRF-input door
(:mod:`woof.wrfinput_door`), which owns the cold start from real.exe.

THE CONTRACT, in both directions, closed-world:

* Every variable in the file is named by :data:`WRFRST_DISPOSITIONS`:
  read into the state, read into the physics, a time-level copy wrf.exe
  itself overwrites, a
  diagnostic rebuilt before any read, the state of a scheme this run does
  not select, or WRF bookkeeping.  A variable outside the table is read
  and must be identically zero (it then carries no state, and the receipt
  names it); a nonzero unclassified variable is refused by name.  A scheme
  whose state the table marks unmappable is refused when the run selects
  it.
* Every cross-step carrier the model holds -- the same manifest a woof
  checkpoint writes (:mod:`woof.io.restart`), walked after the physics is
  initialized -- is filled from a named WRF variable or carries a named
  reason it needs none (:func:`apply_warm_start_physics`).  A carrier with
  neither refuses the start, so state added to the model tomorrow cannot be
  silently cold-started from a WRF restart.

THE CLOCK.  The run's start is the simulation start the file records
(``SIMULATION_START_DATE``), which is also wrfbdy's first record, and the
model clock opens at the restart instant: ``ITIMESTEP`` steps of ``DT``.
That is how wrf.exe restarts -- it keeps the lateral-boundary clock and
every cadence (radiation every STEPRA steps, counted from step one) on the
original run's timeline -- and it is why the radiation the restart carries
is consumed, not recomputed, until the next due step.

MOIST THETA.  Under ``USE_THETA_M = 1`` the prognostic (``THM_2``) is
moist potential temperature.  woof integrates dry theta (the declared
``use_theta_m`` divergence of the WRF-input door), so the restored theta is
``(THM_2 + 300) / (1 + Rv/Rd qv) - 300`` in WRF's FP32 constants; the
file's own dry diagnostic ``T`` is compared and the difference recorded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping

import numpy as np

from woof import netcdf_bridge

WRF_TIME_FORMAT = "%Y-%m-%d_%H:%M:%S"
RESTART_PREFIX = "wrfrst_d"

#: The WRF Registry's two time levels: ``_2`` is the current state wrf.exe
#: integrates from, ``_1`` the RK time-t copy it overwrites from ``_2`` at
#: the next step's entry (woof's ``u0``/``v0``/... are rebuilt the same
#: way).  The ``_2`` names are restored under the names a wrfinput uses.
PROGNOSTIC_RENAMES = MappingProxyType({
    "U_2": "U", "V_2": "V", "W_2": "W", "PH_2": "PH", "MU_2": "MU",
})


@dataclass(frozen=True)
class Disposition:
    """What the importer does with one WRF restart variable, and why.

    ``kind`` is one of :data:`DISPOSITION_KINDS`.  ``active`` names the
    configuration under which a variable classified unmappable would carry
    state this importer cannot restore: when it returns True the start is
    refused with ``reason``.
    """

    kind: str
    reason: str
    active: Callable[[object], bool] | None = None


#: ``read``: decoded into the restored domain and consumed by the state
#: restore or the physics overlay.  ``time_level``/``rebuilt``/``setup``/
#: ``bookkeeping``: not read; the reason says why nothing reads it.
#: ``inactive``: the state of a scheme this run does not select; refused
#: (``active``) when the run does select it.
DISPOSITION_KINDS = frozenset({
    "read", "time_level", "rebuilt", "setup", "bookkeeping", "inactive",
})


def _cfg(name: str, default=0):
    def read(cfg):
        return getattr(cfg, name, default)
    return read


def _rows(kind, reason, names, active=None):
    return {name: Disposition(kind, reason, active) for name in names}


_READ = "read"

#: Every variable of a WRF 4.x restart, by name.  Built from the WRF v4.4
#: CONUS-suite restart NCAR publishes with its v4.4 benchmarks (577
#: variables, MEASURED 2026-10-09 from wrfrst_d01_2019-11-26_23:00:00) and
#: the WRF v4.6.1 Registry the names come from.
WRFRST_DISPOSITIONS: Mapping[str, Disposition] = MappingProxyType({
    # -- the dynamical core -------------------------------------------
    **_rows(_READ, "current prognostic (restored as the wrfinput name)",
            tuple(PROGNOSTIC_RENAMES)),
    **_rows(_READ, "prognostic theta (moist under USE_THETA_M = 1); restored "
            "as dry theta", ("THM_2",)),
    **_rows(_READ, "WRF's dry theta diagnostic th_phy_m_t0; compared with "
            "the restored theta", ("T",)),
    **_rows("time_level", "RK time-t copy: wrf.exe overwrites it from the "
            "_2 level at the next step's entry, and woof rebuilds its own "
            "u0/v0/w0/thp0/php0/mup0 there",
            ("U_1", "V_1", "W_1", "PH_1", "MU_1", "THM_1", "TKE_1", "PC_1")),
    **_rows(_READ, "base state and vertical coordinate (WRF's restart keeps "
            "the file's base state; start_em.F rebuilds it only off-restart)",
            ("PHB", "MUB", "PB", "ALB", "T_INIT", "P_TOP", "ZNU", "ZNW",
             "FNM", "FNP", "RDNW", "RDN", "DNW", "DN", "C1H", "C2H", "C1F",
             "C2F", "C3H", "C4H", "C3F", "C4F", "CF1", "CF2", "CF3",
             "P00", "T00", "TLP", "TISO", "TLP_STRAT", "P_STRAT")),
    **_rows(_READ, "pressure and inverse density, compared with the EOS "
            "woof rediagnoses from the restored prognostics",
            ("P", "AL", "ALT")),
    **_rows(_READ, "retained microphysical heating (WRF rdu), read by the "
            "next RK step", ("H_DIABATIC",)),
    **_rows(_READ, "the dycore's advective theta/qv forcing exported at RK "
            "stage 1 of the last step and read by the next cumulus call",
            ("RTHFTEN", "RQVFTEN")),
    **_rows(_READ, "km_opt = 2 prognostic SGS TKE", ("TKE_2",)),
    **_rows(_READ, "map factors, Coriolis and grid rotation",
            ("MAPFAC_M", "MAPFAC_U", "MAPFAC_V", "MAPFAC_MX", "MAPFAC_MY",
             "MAPFAC_UX", "MAPFAC_UY", "MAPFAC_VX", "MAPFAC_VY", "F", "E",
             "SINALPHA", "COSALPHA", "HGT")),
    **_rows(_READ, "grid coordinates", (
        "XLAT", "XLONG", "XLAT_U", "XLONG_U", "XLAT_V", "XLONG_V")),
    **_rows("rebuilt", "diagnosed from the prognostics at every RK stage "
            "before any read (omega, its acoustic average, density, the "
            "staggered hydrostatic pressure, the geopotential diagnostic)",
            ("WW", "WW_M", "RHO", "P_HYD_W", "PHP")),
    **_rows(_READ, "hydrostatic pressure of the last phy_prep, held for "
            "history until the next step rewrites it (Registry irh)",
            ("P_HYD",)),
    **_rows("rebuilt", "physics-grid copies phy_prep refills each step",
            ("T_PHY", "U_PHY", "V_PHY")),
    **_rows("rebuilt", "diffusion diagnostics (deformation, eddy "
            "viscosities, divergence, N^2) recomputed before use each step",
            ("DEFOR11", "DEFOR22", "DEFOR12", "DEFOR33", "DEFOR13",
             "DEFOR23", "XKMV", "XKMH", "XKHV", "XKHH", "DIV", "BN2")),
    **_rows("inactive", "WRF's ideal-case base profiles; zero on real data "
            "and not restored (the wrfinput door's zero-profile rule)",
            ("T_BASE", "U_BASE", "V_BASE", "QV_BASE", "Z_BASE")),
    **_rows("inactive", "initial geopotential kept only by adaptive "
            "vertical nesting", ("PH0",)),
    **_rows("inactive", "hydrostatic-mode pressure pair (non_hydrostatic = "
            ".false.), not implemented",
            ("PCB", "PC_2"), _cfg("hydrostatic", False)),
    **_rows("inactive", "WRF's moist-theta fold of microphysical moistening "
            "into the theta_m tendency; woof integrates dry theta, whose "
            "diabatic tendency is H_DIABATIC alone",
            ("QV_DIABATIC", "QC_DIABATIC")),
    # -- moisture ------------------------------------------------------
    **_rows(_READ, "moisture species and number concentrations (the "
            "active scheme's inventory is checked by the moisture map)",
            ("QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP", "QHAIL",
             "QNICE", "QNRAIN", "QNSNOW", "QNGRAUPEL", "QNHAIL", "QNCLOUD",
             "QNWFA", "QNIFA", "QNBCA", "QNCCN", "QNDROP", "QVGRAUPEL",
             "QVHAIL", "QIR", "QIB", "QNWFA2D", "QNIFA2D")),
    **_rows(_READ, "microphysics effective radii, read by the next "
            "radiation call", ("RE_CLOUD", "RE_ICE", "RE_SNOW")),
    **_rows(_READ, "WRF's effective-radius ownership flags, compared with "
            "the active scheme's", ("HAS_REQC", "HAS_REQI", "HAS_REQS")),
    **_rows("inactive", "previous-step vapour of the moisture-advection "
            "trigger (KF's kfeta_trigger = 2), which woof does not port",
            ("QVOLD",), lambda c: int(getattr(c, "kfeta_trigger", 1)) == 2),
    **_rows(_READ, "Kain-Fritsch's running vertical-velocity average",
            ("W0AVG",)),
    # -- precipitation and microphysics persistence ---------------------
    **_rows(_READ, "precipitation accumulators and last-step amounts",
            ("RAINNC", "RAINNCV", "SNOWNC", "SNOWNCV", "GRAUPELNC",
             "GRAUPELNCV", "HAILNC", "HAILNCV", "SR", "RAINC", "RAINCV",
             "PRATEC", "NCA", "RAINBL")),
    **_rows("inactive", "shallow-cumulus precipitation (shcu_physics), not "
            "selected", ("RAINSH", "PRATESH", "RAINSHV"),
            _cfg("shcu_physics")),
    **_rows("inactive", "Eta/Ferrier microphysics restart tables "
            "(mp_physics = 5), not implemented",
            ("MP_RESTART_STATE", "TBPVS_STATE", "TBPVS0_STATE"),
            lambda c: int(c.mp_physics) in (5, 85, 95)),
    # -- cumulus ---------------------------------------------------------
    **_rows(_READ, "cumulus tendencies of the last call", (
        "RTHCUTEN", "RQVCUTEN", "RQRCUTEN", "RQCCUTEN", "RQSCUTEN",
        "RQICUTEN", "RUCUTEN", "RVCUTEN")),
    **_rows("inactive", "aerosol-aware cumulus number tendencies, not "
            "selected", ("RQCNCUTEN", "RQINCUTEN")),
    **_rows("rebuilt", "cumulus cloud top/base and convective-cloud "
            "diagnostics, rewritten by the next call",
            ("HTOP", "HBOT", "HTOPR", "HBOTR", "CUTOP", "CUBOT")),
    **_rows("inactive", "convective precipitation accumulated for CAM "
            "radiation's cloud fraction; RRTMG does not read it", ("CUPPT",),
            lambda c: int(getattr(c, "ra_lw_physics", 0)) == 3),
    **_rows("inactive", "Betts-Miller-Janjic cloud efficiency (cu_physics "
            "= 2), not implemented", ("CLDEFI",),
            lambda c: int(c.cu_physics) == 2),
    **_rows("inactive", "Grell-family ensemble and closure state (cu 3/5/93 "
            "ensemble members, mass flux, cloud fractions); woof's GF "
            "recomputes its closures every call",
            ("MASS_FLUX", "CLDFRA_DP", "CLDFRA_SH", "APR_GR", "APR_W",
             "APR_MC", "APR_ST", "APR_AS", "APR_CAPMA", "APR_CAPME",
             "APR_CAPMI", "XF_ENS", "PR_ENS", "RAINCV_A", "RAINCV_B",
             "QC_CU", "QI_CU")),
    **_rows("inactive", "KF-CuP cumulus (cu_physics = 10), not implemented",
            ("CUPFLAG", "SLOPESFC", "SLOPEEZ", "SIGMASFC", "SIGMAEZ",
             "SHALL", "TAUCLOUD", "TACTIVE", "TCLOUD_CUP", "WCLOUDBASE",
             "ACTIVEFRAC", "WACT_CUP", "WULCL_CUP"),
            lambda c: int(c.cu_physics) == 10),
    **_rows("bookkeeping", "cudt averaging counter", ("STEPAVE_COUNT",)),
    # -- radiation ------------------------------------------------------
    **_rows(_READ, "held radiative heating (longwave, shortwave, total), "
            "consumed every step until the next due radiation call",
            ("RTHRATEN", "RTHRATLW", "RTHRATSW")),
    **_rows(_READ, "held surface radiation the land surface reads every "
            "step until the next due radiation call",
            ("SWDOWN", "GLW", "GSW")),
    **_rows(_READ, "radiation-cadence cloud fraction and TOA longwave, "
            "published between radiation calls", ("CLDFRA", "OLR")),
    **_rows("rebuilt", "clear-sky heating and flux diagnostics of the last "
            "radiation call, rewritten by the next one",
            ("RTHRATLWC", "RTHRATSWC",
             "SWCF", "LWCF", "HRANG", "DECLIN", "SOLCON",
             "DIFFUSE_FRAC", "SWDDIR", "SWDDIRC", "SWDDNI", "SWDDNIC",
             "SWDDIF", "O3RAD")),
    **_rows(_READ, "top and bottom radiative fluxes, the zenith cosine and "
            "the slope-normal shortwave of the last radiation call, held "
            "for history (and COSZEN for its consumers) until the next one",
            ("SWUPT", "SWUPTC", "SWDNT", "SWDNTC", "SWUPB", "SWUPBC",
             "SWDNB", "SWDNBC", "LWUPT", "LWUPTC", "LWDNT", "LWDNTC",
             "LWUPB", "LWUPBC", "LWDNB", "LWDNBC", "COSZEN", "SWNORM")),
    **_rows("inactive", "shortwave interpolation fit (swint_opt = 1), not "
            "selected", ("GX", "BX", "GG", "BB", "COSZEN_REF", "SWDOWN_REF",
                         "SWDDIR_REF"), _cfg("swint_opt")),
    **_rows("inactive", "spectral surface shortwave for Noah-MP/CLM, not "
            "selected", ("SWVISDIR", "SWVISDIF", "SWNIRDIR", "SWNIRDIF",
                         "ALSWVISDIR", "ALSWVISDIF", "ALSWNIRDIR",
                         "ALSWNIRDIF"),
            lambda c: int(c.sf_surface_physics) == 4),
    **_rows("inactive", "aerosol optical properties (aer_opt), not selected",
            ("AODTOT", "AOD5502D", "ANGEXP2D", "AERSSA2D", "AERASY2D"),
            _cfg("aer_opt")),
    **_rows(_READ, "radiation flux accumulations, continued from the "
            "restart as wrf.exe continues them",
            ("ACSWUPT", "ACSWUPTC", "ACSWDNT", "ACSWDNTC", "ACSWUPB",
             "ACSWUPBC", "ACSWDNB", "ACSWDNBC", "ACLWUPT", "ACLWUPTC",
             "ACLWDNT", "ACLWDNTC", "ACLWUPB", "ACLWUPBC", "ACLWDNB",
             "ACLWDNBC")),
    **_rows("setup", "slope-radiation and shading statics, rebuilt from "
            "terrain by the topographic radiation carrier",
            ("SLOPE", "SLP_AZI", "HGT_SHAD")),
    # -- PBL and surface layer -----------------------------------------
    **_rows(_READ, "PBL tendencies of the last call", (
        "RUBLTEN", "RVBLTEN", "RTHBLTEN", "RQVBLTEN", "RQCBLTEN",
        "RQIBLTEN")),
    **_rows("inactive", "MYNN ice-number PBL tendency, not selected",
            ("RQNIBLTEN",)),
    **_rows(_READ, "MYJ PBL carried TKE and mixing length, exchange "
            "coefficients", ("TKE_PBL", "EL_PBL", "EXCH_H", "EXCH_M")),
    **_rows(_READ, "surface-layer and land-surface carried state",
            ("UST", "USTM", "ZNT", "Z0", "MOL", "RMOL", "PBLH", "KPBL", "CT",
             "THZ0", "QZ0", "UZ0", "VZ0", "QSFC", "AKHS", "AKMS", "HFX",
             "QFX", "LH", "FLHC", "FLQC", "REGIME", "MAVAIL", "TSHLTR",
             "QSHLTR", "PSHLTR", "TH10", "Q10", "Q2", "T2", "TH2", "PSFC",
             "U10", "V10")),
    **_rows("rebuilt", "MM5 surface-layer drag/exchange diagnostics "
            "(isftcflx), rewritten by every call", ("CK", "CKA", "CD", "CDA")),
    **_rows("inactive", "ACM2 PBL index (bl_pbl_physics = 7), not selected",
            ("AKPBL",), lambda c: int(c.bl_pbl_physics) == 7),
    **_rows("inactive", "Noah-MP / urban surface-layer exchange carriers, "
            "not selected", ("CMR_SFCDIF", "CHR_SFCDIF", "CMC_SFCDIF",
                             "CHC_SFCDIF"),
            lambda c: int(c.sf_surface_physics) == 4),
    **_rows("inactive", "TEMF surface flux carrier, not selected",
            ("TKESFCF",)),
    # -- land surface ----------------------------------------------------
    **_rows(_READ, "soil, snow and surface state", (
        "TSK", "TSLB", "SMOIS", "SH2O", "SMCREL", "SNOW", "SNOWH", "SNOWC",
        "CANWAT", "ALBEDO", "ALBBCK", "EMISS", "EMBCK", "SNOTIME", "ACSNOM",
        "ACSNOW", "SFROFF", "UDROFF", "SMSTAV", "SMSTOT", "POTEVP", "SNOPCX",
        "GRDFLX", "TMN", "SST", "SEAICE", "XLAND", "LANDMASK", "LAKEMASK",
        "LU_INDEX", "IVGTYP", "ISLTYP", "VEGFRA", "SHDMAX", "SHDMIN",
        "SNOALB", "LAI", "FRC_URB2D")),
    **_rows(_READ, "surface energy accumulations (sensible, latent and "
            "ground heat), continued from the restart as wrf.exe continues "
            "them", ("ACHFX", "ACLHF", "ACGRDFLX")),
    **_rows("bookkeeping", "evaporation total woof does not carry, so a "
            "continued run's history publishes none of it", ("SFCEVP",)),
    **_rows("rebuilt", "Noah's exchange coefficient output, rewritten by "
            "every call", ("SFCEXC",)),
    **_rows(_READ, "previous-step sea-ice flag, held for the sea-ice "
            "refresh and history", ("XICEM",)),
    **_rows("inactive", "fractional sea-ice skin temperature save, not "
            "selected", ("TSK_SAVE",), _cfg("fractional_seaice")),
    **_rows(_READ, "skin SST, held at the restart's value while no SST "
            "skin producer runs (WRF holds it the same way)", ("SSTSK",)),
    **_rows("inactive", "SST warm-layer state (sst_skin = 1), not selected",
            ("DTW",), _cfg("sst_skin")),
    **_rows("inactive", "deep soil temperature update (tmn_update = 1), not "
            "selected", ("TYR", "TYRA", "TDLY", "TLAG", "NYEAR", "NDAY"),
            _cfg("tmn_update")),
    **_rows("inactive", "RUC land-surface state, not selected",
            ("QSG", "QVG", "QCG", "DEW", "TSNAV", "SOILTB"),
            lambda c: int(c.sf_surface_physics) == 3),
    **_rows("inactive", "thermal-diffusion slab land surface state, not "
            "selected", ("CAPG", "THC"),
            lambda c: int(c.sf_surface_physics) == 1),
    **_rows("inactive", "Noah-MP / CLM resistances, not selected",
            ("RA", "RS"), lambda c: int(c.sf_surface_physics) == 4),
    **_rows("inactive", "urban canopy state (sf_urban_physics), which this "
            "importer does not map", (
                "TR_URB", "TB_URB", "TG_URB", "TC_URB", "QC_URB", "UC_URB",
                "XXXR_URB", "XXXB_URB", "XXXG_URB", "XXXC_URB", "SH_URB",
                "LH_URB", "G_URB", "RN_URB", "TS_URB", "UTYPE_URB", "DZR",
                "DZB", "DZG"), _cfg("sf_urban_physics")),
    **_rows("setup", "land-use and soil tables WRF saves for its own "
            "restart; woof reads its own tables",
            ("LANDUSE_ISICE", "LANDUSE_LUCATS", "LANDUSE_LUSEAS",
             "LANDUSE_ISN", "LU_STATE")),
    **_rows("setup", "static fields the restored model does not consume "
            "(the wrfinput door's ignored set)",
            ("ZS", "DZS", "VAR_SSO", "LAP_HGT", "EROD", "BATHYMETRY_FLAG",
             "CFN", "CFN1", "RDX", "RDY", "DTS", "DTSEPS", "RESM", "ZETATOP",
             "FCX", "GCX", "TOPOSLPX", "TOPOSLPY", "LANDUSEF", "SOILCTOP",
             "SOILCBOT", "FNDALBSI", "FNDSNOWSI", "FNDICEDEPTH",
             "LAKE_DEPTH", "WATER_DEPTH", "UOCE", "VOCE", "VAR", "CLAT",
             "U_FRAME", "V_FRAME", "GOT_VAR_SSO", "MF_VX_INV", "CPLMASK",
             "SAVE_TOPO_FROM_REAL", "SST_INPUT", "AREA2D", "DX2D",
             "CTOPO", "CTOPO2")),
    # -- diagnostics and features this run does not run -------------------
    **_rows("rebuilt", "updraft helicity diagnostic", ("UH",)),
    **_rows("inactive", "observation and spectral nudging (FDDA), not "
            "implemented", ("URATX", "VRATX", "TRATX", "SN_NDG_NEW",
                            "SN_NDG_OLD", "STEPFG"),
            _cfg("grid_fdda")),
    **_rows("inactive", "stochastic perturbation patterns and seeds, not "
            "carried", ("SPFORCC5", "SPFORCS5", "SP_AMP5", "ISEEDARR_SPPT",
                        "ISEEDARR_SKEBS", "ISEEDARR_RAND_PERTURB",
                        "ISEEDARRAY_SPP_CONV", "ISEEDARRAY_SPP_PBL",
                        "ISEEDARRAY_SPP_LSM", "DID_STOCH", "PERT3D_STD",
                        "PERT3D_CUT", "PERT3D_LEN", "PERT3D_TIM",
                        "PERT3D_IVS", "PERT3D_ISD")),
    **_rows("inactive", "hailcast, lightning and averaged-flux diagnostics, "
            "not selected", ("HAILCAST_WUP_MASK", "HAILCAST_WDUR",
                             "HAILDTACTTIME", "ICCG_IN_NUM", "ICCG_IN_DEN",
                             "AVGFLX_COUNT", "LNINT", "TSTAR")),
    **_rows("inactive", "diurnal/monthly mean output diagnostics, not "
            "carried", ("NSTEP_M", "NSTEP_D", "PSFC_DTM", "TSK_DTM",
                        "T2_DTM", "TH2_DTM", "Q2_DTM", "U10_DTM", "V10_DTM",
                        "HFX_DTM", "LH_DTM", "SWDNB_DTM", "GLW_DTM",
                        "LWUPB_DTM", "SWUPB_DTM", "SWUPT_DTM", "SWDNT_DTM",
                        "LWUPT_DTM", "LWDNT_DTM", "NSTEP_DTM")),
    **_rows("inactive", "single-column-model surface forcing, not selected",
            ("HFX_FORCE", "LH_FORCE", "TSK_FORCE", "HFX_FORCE_TEND",
             "LH_FORCE_TEND", "TSK_FORCE_TEND")),
    **_rows("inactive", "moving-nest and vortex-following placement, which "
            "a static domain does not use",
            ("NEST_POS", "NEST_MASK", "HT_COARSE", "XI", "XJ", "VC_I",
             "VC_J")),
    # -- the clock and WRF bookkeeping -----------------------------------
    **_rows(_READ, "the clock: step count, elapsed minutes and time since "
            "the last boundary record", ("ITIMESTEP", "XTIME", "DTBC")),
    **_rows(_READ, "WRF's physics cadences in steps, compared with woof's",
            ("STEPRA", "STEPCU", "STEPBL")),
    **_rows("bookkeeping", "WRF clock, adaptive-step and I/O bookkeeping",
            ("STEP_NUMBER", "THIS_IS_AN_IDEAL_RUN", "STEPPING_TO_TIME",
             "LAST_STEP_UPDATED", "ADAPT_STEP_USING_CHILD", "LAST_DT_SEC",
             "LAST_DT_SEC_NUM", "LAST_DT_SEC_DEN", "LAST_DT_YR",
             "LAST_DT_MM", "RADTACTTIME", "BLDTACTTIME", "CUDTACTTIME",
             "LTNGACTTIME", "NSTEPS", "MIN_PTCHSZ", "WE_JUST_READ_SST",
             "WE_JUST_D01_LBC", "MF_FFT", "MAX_MSFTX", "MAX_MSFTY",
             "IS_CAMMGMP_USED")),
    **_rows("bookkeeping", "domain-corner coordinates", tuple(
        f"{axis}_{corner}_{point}" for axis in ("LAT", "LON")
        for corner in ("LL", "UL", "UR", "LR")
        for point in ("T", "U", "V", "D"))),
})


@dataclass(frozen=True)
class WrfRestartClock:
    """The clock a WRF restart carries, checked against itself."""

    path: Path
    grid_id: int
    valid_time: datetime
    simulation_start: datetime
    itimestep: int
    dt_seconds: float
    dtbc_seconds: float

    @property
    def elapsed_seconds(self) -> float:
        return float((self.valid_time - self.simulation_start).total_seconds())


def _time_text(chars) -> str:
    array = np.asarray(chars).reshape(-1)
    return b"".join(bytes(item) for item in array).decode("ascii").strip("\x00 ")


def _parse_time(text: str, *, path, what: str) -> datetime:
    try:
        return datetime.strptime(str(text).strip(), WRF_TIME_FORMAT)
    except ValueError:
        raise ValueError(f"{path}: {what} {text!r} is not a WRF timestamp") from None


def _scalar(dataset, name: str, path):
    if name not in dataset.variables:
        raise ValueError(
            f"{path} has no {name}; a WRF restart carries it and the warm "
            "start cannot place the clock without it")
    value = np.asarray(dataset.variables[name][...]).reshape(-1)
    if value.size != 1 or not np.isfinite(value.astype(np.float64)).all():
        raise ValueError(f"{path}: {name} must be one finite value")
    return value[0]


def read_wrfrst_clock(path) -> WrfRestartClock:
    """Read and cross-check the clock of one WRF restart, no fields.

    Refusals, each naming what it prevents: a file WRF did not stamp as a
    restart (a wrfinput or wrfout would be read as a mid-run state); a
    restart instant that disagrees with its own step count (an adaptive
    time step, whose varying steps woof's fixed clock cannot place); an
    XTIME that disagrees with both; a DTBC outside a boundary interval.
    """
    path = Path(path)
    with netcdf_bridge.open_dataset(path) as dataset:
        attrs = {name: dataset.getncattr(name) for name in dataset.ncattrs()}
        flag = attrs.get("FLAG_RESTART")
        if flag is None or int(float(flag)) != 1:
            raise ValueError(
                f"{path} is not a WRF restart file (FLAG_RESTART = {flag!r}; "
                "wrf.exe writes 1 into every wrfrst).  A wrfinput or wrfout "
                "is not a mid-run state: start a real.exe handoff with "
                "--wrfinput DIR, or name the wrfrst_d0N_<time> file")
        if "Times" not in dataset.variables:
            raise ValueError(f"{path} has no Times record")
        valid = _parse_time(_time_text(dataset.variables["Times"][...]),
                            path=path, what="Times")
        start_text = attrs.get("SIMULATION_START_DATE", attrs.get("START_DATE"))
        if start_text is None:
            raise ValueError(
                f"{path} records no SIMULATION_START_DATE; the restart cannot "
                "be placed on its boundary clock")
        start = _parse_time(start_text, path=path, what="SIMULATION_START_DATE")
        itimestep = int(_scalar(dataset, "ITIMESTEP", path))
        xtime = float(_scalar(dataset, "XTIME", path))
        dtbc = float(_scalar(dataset, "DTBC", path))
        grid_id = int(float(attrs.get("GRID_ID", 0)))
        dt = float(attrs.get("DT", 0.0))
    if valid < start:
        raise ValueError(
            f"{path}: restart instant {valid:%Y-%m-%d_%H:%M:%S} precedes its "
            f"simulation start {start:%Y-%m-%d_%H:%M:%S}")
    elapsed = (valid - start).total_seconds()
    if not dt > 0.0:
        raise ValueError(f"{path} records no positive DT")
    if not np.isclose(itimestep * dt, elapsed, rtol=0.0, atol=1e-3):
        raise ValueError(
            f"{path}: {itimestep} steps of DT = {dt:g} s is "
            f"{itimestep * dt:g} s but the restart lies {elapsed:g} s after "
            "the simulation start.  WRF ran it on an adaptive time step, "
            "whose varying steps woof's fixed restart clock cannot place; "
            "restart from a fixed-step run")
    if not np.isclose(xtime * 60.0, elapsed, rtol=0.0, atol=1e-2):
        raise ValueError(
            f"{path}: XTIME = {xtime:g} min disagrees with the restart "
            f"instant {elapsed / 60.0:g} min after the simulation start")
    if not 0.0 <= dtbc:
        raise ValueError(f"{path}: DTBC = {dtbc:g} s is negative")
    return WrfRestartClock(path, grid_id, valid, start, itimestep, dt, dtbc)


def restart_file_name(grid_id: int, valid_time: datetime) -> str:
    """The name wrf.exe reads a restart from."""
    return f"{RESTART_PREFIX}{int(grid_id):02d}_{valid_time:{WRF_TIME_FORMAT}}"


def discover_restart_files(directory, *, explicit=None,
                           restart_time: datetime | None = None
                           ) -> dict[int, Path]:
    """Every domain's restart file for one restart instant.

    ``explicit`` names one file (any name: NCAR ships its benchmark
    restarts as ``wrfrst_d01_<time>.gnu``); its siblings for the other
    domains are the same name with the grid number changed, as wrf.exe
    writes them.  Otherwise ``restart_time`` names the instant and the
    files are wrf.exe's own names in ``directory``.
    """
    directory = Path(directory)
    if explicit is not None:
        explicit = Path(explicit)
        if not explicit.is_file():
            raise ValueError(f"--restart-from {explicit} is not a file")
        clock = read_wrfrst_clock(explicit)
        found = {clock.grid_id: explicit}
        stem = explicit.name
        marker = f"{RESTART_PREFIX}{clock.grid_id:02d}_"
        if stem.startswith(marker):
            for sibling in sorted(explicit.parent.glob(
                    RESTART_PREFIX + "??_" + stem[len(marker):])):
                gid = sibling.name[len(RESTART_PREFIX):len(RESTART_PREFIX) + 2]
                if gid.isdigit() and int(gid) not in found:
                    found[int(gid)] = sibling
        return found
    if restart_time is None:
        raise ValueError("a restart needs an instant or a named file")
    found = {}
    for path in sorted(directory.glob(
            f"{RESTART_PREFIX}??_{restart_time:{WRF_TIME_FORMAT}}")):
        gid = path.name[len(RESTART_PREFIX):len(RESTART_PREFIX) + 2]
        if gid.isdigit():
            found[int(gid)] = path
    if 1 not in found:
        raise ValueError(
            f"namelist.input asks wrf.exe to restart at "
            f"{restart_time:{WRF_TIME_FORMAT}}, but {directory} has no "
            f"{restart_file_name(1, restart_time)}; name the file with "
            "--restart-from")
    return found


@dataclass(frozen=True)
class WarmStart:
    """What a restored domain carries about the WRF run it continues."""

    clock: WrfRestartClock
    use_theta_m: int
    file_physics: Mapping[str, int]
    dispositions: Mapping[str, str]
    checks: Mapping[str, object]


#: WRF's physics cadences in steps, as a restart writes them.
CADENCE_KEYS = ("STEPRA", "STEPCU", "STEPBL")


def wrf_interval_steps(minutes: float, dt: float) -> int:
    """WRF's ``MAX(NINT(minutes*60/dt), 1)``, importable without CuPy.

    The same float expression as ``woof.core.physics._physics_interval_steps``
    (a test pins the two equal); the restart reader runs before any card is
    taken, so it cannot import the physics driver.
    """
    if minutes <= 0.0:
        return 1
    return max(int(np.floor(minutes * 60.0 / dt + 0.5)), 1)


def compare_restart_cadences(written: Mapping[str, int], cfg, *,
                             radt_minutes: float | None = None,
                             radiation_active: bool | None = None
                             ) -> dict[str, list]:
    """Compare a restart's STEPRA/STEPCU/STEPBL with this run's cadences.

    Returns the per-key check rows; raises when a scheme this run calls
    would fire on a different phase than the producing run.  ``radt_minutes``
    and ``radiation_active`` default to what the physics driver derives from
    ``cfg`` (:func:`woof.config.effective_radt_minutes`, either radiation
    stream on), so the scalar-only check before the full read and the
    driver's check after it compare the same numbers.

    A cadence is compared only for a scheme this run calls.  Named breakage:
    NCAR's v4.4 CONUS 2.5 km benchmark (cu_physics = 0, cudt = 0) was refused
    after its whole restart had been read with "the restart's STEPCU = 1
    steps but this run's cadence is 20 steps": WRF writes STEPCU =
    MAX(NINT(cudt*60/dt), 1) = 1 whatever the scheme, and the importer leaves
    cudt_minutes to RunConfig's 5.0 when cumulus is off (namelist_import:
    "the same dead switch"), so 300 s / 15 s = 20 was compared with a
    cadence neither run fires.
    """
    from woof.config import radiation_scheme_ids

    if not written:
        return {}
    dt = float(cfg.dt)
    if radt_minutes is None:
        radt = float(getattr(cfg, "radt", 0.0) or 0.0)
        radt_minutes = radt if radt > 0.0 else float(
            getattr(cfg, "radt_minutes", 0.0) or 0.0)
    if radiation_active is None:
        radiation_active = any(radiation_scheme_ids(cfg))
    checks: dict[str, list] = {}
    for key, minutes, active in (
            ("STEPRA", float(radt_minutes), bool(radiation_active)),
            ("STEPCU", float(getattr(cfg, "cudt_minutes", 0.0) or 0.0),
             int(getattr(cfg, "cu_physics", 0) or 0) != 0),
            ("STEPBL", float(getattr(cfg, "bldt", 0.0) or 0.0),
             int(getattr(cfg, "bl_pbl_physics", 0) or 0) != 0)):
        if key not in written:
            continue
        value = int(written[key])
        mine = wrf_interval_steps(minutes, dt)
        if not active:
            checks[f"{key.lower()}_wrf_vs_gpuwm"] = [
                value, mine, "scheme off in this run: not compared"]
            continue
        checks[f"{key.lower()}_wrf_vs_gpuwm"] = [value, mine]
        if value != mine:
            raise ValueError(
                f"the restart's {key} = {value} steps but this run's "
                f"cadence is {mine} steps; continuing would fire that "
                "scheme on a different phase than the producing run")
    return checks


def _disposition(name: str) -> Disposition | None:
    return WRFRST_DISPOSITIONS.get(name)


def read_wrfrst(path, *, expected_dimensions: Mapping[str, int], cfg):
    """Read one WRF restart into a :class:`RestoredDomain`, no CuPy.

    The restored domain's ``raw`` holds every ``read`` variable under the
    name the shared wrfinput restore expects (``U`` for ``U_2``, dry ``T``
    for ``THM_2``) plus the restart's own carriers under their WRF names;
    its ``warm_start`` records the clock and the per-variable dispositions.
    """
    from woof.ingest.wrfinput import (
        RestoredDomain, WRFINPUT_DIMENSIONS, _explicit_wrfinput_dimensions,
        _read_numeric, active_moisture_inventory, check_supported_schemes,
        resolve_map_factors, _WRF_RVOVRD)

    path = Path(path)
    clock = read_wrfrst_clock(path)
    extents = dict(_explicit_wrfinput_dimensions(expected_dimensions))
    _, allowed_moisture = active_moisture_inventory(cfg)
    from woof.ingest.wrfinput import ALL_MOISTURE_WRFINPUT
    raw: dict[str, np.ndarray] = {}
    dispositions: dict[str, str] = {}
    refused: list[str] = []
    unclassified_nonzero: list[str] = []
    with netcdf_bridge.open_dataset(path) as dataset:
        dimensions = {name: len(dim) for name, dim in dataset.dimensions.items()}
        attrs = {name: dataset.getncattr(name) for name in dataset.ncattrs()}
        check_supported_schemes(attrs, source=str(path))
        # The cadences are scalars: compare them before the full read.
        # Named breakage: NCAR's 2.5 km restart was refused on STEPCU only
        # after 44 minutes of reading its 4D fields.
        cadence_checks = compare_restart_cadences(
            {key: int(_scalar(dataset, key, path))
             for key in CADENCE_KEYS if key in dataset.variables}, cfg)
        for axis, size in extents.items():
            if dimensions.get(axis) != size:
                raise ValueError(
                    f"{path}: dimension {axis} is {dimensions.get(axis)}, the "
                    f"configuration resolves {size}; this restart is from a "
                    "different domain or vertical grid")
        decode = netcdf_bridge.BatchedDecode(dataset, [
            name for name in dataset.variables
            if name != "Times" and (
                _disposition(name) is None or _disposition(name).kind == "read")])
        for name, variable in dataset.variables.items():
            if name == "Times":
                continue
            decode.ready(name)
            rule = _disposition(name)
            if rule is None:
                value = _read_numeric(variable)
                if np.any(value != 0):
                    unclassified_nonzero.append(name)
                    continue
                dispositions[name] = ("unclassified and identically zero: "
                                      "carries no state")
                continue
            if rule.kind == "inactive" and rule.active is not None \
                    and rule.active(cfg):
                refused.append(f"{name}: {rule.reason}")
                continue
            dispositions[name] = f"{rule.kind}: {rule.reason}"
            if rule.kind != "read":
                continue
            if name in ALL_MOISTURE_WRFINPUT and name not in allowed_moisture:
                value = _read_numeric(variable)
                if np.any(value != 0):
                    refused.append(
                        f"{name}: nonzero moisture outside mp_physics="
                        f"{int(cfg.mp_physics)}'s inventory")
                dispositions[name] = ("inactive: moisture outside the active "
                                      "scheme's inventory, identically zero")
                continue
            value = _read_numeric(variable)
            target = PROGNOSTIC_RENAMES.get(name, name)
            raw[target] = value
        if unclassified_nonzero:
            raise ValueError(
                f"{path} carries nonzero WRF restart variable(s) this "
                f"importer has no disposition for: {sorted(unclassified_nonzero)}. "
                "Each is model state of some scheme or feature; continuing "
                "without it would cold-start that state in the middle of a "
                "trajectory.  The table is woof/ingest/wrfrst.py "
                "WRFRST_DISPOSITIONS")
        if refused:
            raise ValueError(
                f"{path} carries state this warm start cannot map for the "
                "selected physics:\n" + "\n".join(f"  {row}" for row in refused))
    # Shapes: every read field that the wrfinput door also reads is checked
    # against its pinned geometry; restart-only carriers are checked where
    # they are consumed (against the live carrier's shape).
    for name, value in raw.items():
        layout = WRFINPUT_DIMENSIONS.get(name)
        if layout is None:
            continue
        expected = tuple(extents[dim] for dim in layout if dim in extents)
        if len(expected) == len(layout) and value.shape != expected:
            raise ValueError(
                f"{path}: {name} shape {value.shape} != {expected} on {layout}")
    use_theta_m = int(float(attrs.get("USE_THETA_M", 0)))
    if "THM_2" not in raw:
        raise ValueError(
            f"{path} has no THM_2, the prognostic theta wrf.exe restarts "
            "from; this is not a complete WRF restart")
    theta = np.asarray(raw.pop("THM_2"), np.float32)
    qv = np.asarray(raw.get("QVAPOR", np.zeros_like(theta)), np.float32)
    file_dry = raw.pop("T", None)
    if use_theta_m == 1:
        dry = np.asarray(
            np.asarray(theta + np.float32(300.0), np.float32)
            / np.asarray(np.float32(1.0) + _WRF_RVOVRD * qv, np.float32)
            - np.float32(300.0), np.float32)
    elif use_theta_m == 0:
        dry = theta
    else:
        raise ValueError(f"{path}: USE_THETA_M = {use_theta_m} is not 0 or 1")
    raw["T"] = dry
    from woof.ingest.wrfinput import missing_required_wrfinput
    missing = missing_required_wrfinput(raw, cfg)
    if missing:
        raise ValueError(
            f"{path} lacks state the warm start restores: {missing} (by the "
            "wrfinput names; a restart carries U_2/V_2/W_2/PH_2/MU_2/THM_2 "
            "for U/V/W/PH/MU/T).  A WRF restart written by wrf.exe carries "
            "every one of them, so this file was cut down by another tool")
    checks: dict[str, object] = dict(cadence_checks)
    # WRF's positive-definite transport leaves roundoff-sized negatives in
    # the hydrometeors and their numbers (NCAR's v4.4 CONUS restart: 322
    # cells, the largest -1.1e-14 /kg in QNICE and -4.6e-23 kg/kg in
    # QCLOUD), which woof's state health gate refuses at the first step.
    # A negative within 1e-9 of the field's own magnitude is that roundoff
    # and is set to zero, counted per field; a larger one is not roundoff.
    clamped = {}
    for name in sorted(set(raw) & allowed_moisture):
        value = np.asarray(raw[name], np.float32)
        negative = value < 0
        if not negative.any():
            continue
        scale = float(np.max(np.abs(value)))
        worst = float(value.min())
        if -worst > 1e-9 * scale:
            raise ValueError(
                f"{path}: {name} reaches {worst:.3g}, beyond roundoff of its "
                f"magnitude {scale:.3g}; a WRF restart's species are "
                "non-negative to within roundoff, so this file is damaged")
        raw[name] = np.where(negative, np.float32(0.0), value)
        clamped[name] = {"cells": int(negative.sum()), "most_negative": worst}
    if clamped:
        checks["roundoff_negative_moisture_set_to_zero"] = clamped
    if file_dry is not None:
        checks["dry_theta_vs_file_T_max_abs_K"] = float(
            np.max(np.abs(dry.astype(np.float64) - file_dry.astype(np.float64))))
    factor_resolution = resolve_map_factors(raw)
    if factor_resolution:
        import json
        attrs["ARWEN_MAP_FACTOR_RESOLUTION"] = json.dumps(
            factor_resolution, sort_keys=True)
    file_physics = file_physics_selectors(attrs)
    warm = WarmStart(
        clock=clock, use_theta_m=use_theta_m,
        file_physics=MappingProxyType(file_physics),
        dispositions=MappingProxyType(dict(sorted(dispositions.items()))),
        checks=MappingProxyType(checks))
    from woof.ingest.wrfinput import MAPPED_WRFINPUT
    return RestoredDomain(
        path=path, raw=MappingProxyType(raw),
        dimensions=MappingProxyType(dimensions),
        global_attributes=MappingProxyType(attrs),
        mapped_variables=tuple(sorted(set(raw) & MAPPED_WRFINPUT)),
        auxiliary_variables=tuple(sorted(
            {"XLAT", "XLONG", "XLAT_U", "XLONG_U", "XLAT_V", "XLONG_V"}
            & set(raw))),
        warm_start=warm)


#: Physics selectors whose scheme carries cross-step state in a restart.
#: The file's and the run's must agree: reading Morrison's state as
#: Thompson's, or MYJ's TKE as nothing, is a different model continuing
#: under the first one's name.  Cumulus is admitted to change between the
#: schemes that hold nothing across a step (they rebuild their tendencies
#: every call), and refused otherwise.
STATEFUL_SELECTORS = ("MP_PHYSICS", "BL_PBL_PHYSICS", "SF_SFCLAY_PHYSICS",
                      "SF_SURFACE_PHYSICS")
STATELESS_CUMULUS = frozenset({0, 3, 6, 16})


def file_physics_selectors(attributes: Mapping[str, object]) -> dict[str, int]:
    """The physics selectors a WRF file's global attributes record."""
    return {key: int(float(attributes[key])) for key in (
        "MP_PHYSICS", "CU_PHYSICS", "BL_PBL_PHYSICS", "SF_SFCLAY_PHYSICS",
        "SF_SURFACE_PHYSICS", "RA_LW_PHYSICS", "RA_SW_PHYSICS")
        if key in attributes}


def check_restart_physics(file_physics: Mapping[str, int], cfg, *,
                          path) -> list[str]:
    """Refuse a run whose stateful physics differs from the restart's.

    Returns the declared changes the run admits (cumulus between stateless
    schemes), for the receipt and the terminal.
    """
    run = {"MP_PHYSICS": int(cfg.mp_physics),
           "BL_PBL_PHYSICS": int(cfg.bl_pbl_physics),
           "SF_SFCLAY_PHYSICS": int(cfg.sf_sfclay_physics),
           "SF_SURFACE_PHYSICS": int(cfg.sf_surface_physics),
           "CU_PHYSICS": int(cfg.cu_physics)}
    problems = [f"{key}: the restart was written by {file_physics[key]}, "
                f"the run selects {run[key]}"
                for key in STATEFUL_SELECTORS
                if key in file_physics
                and file_physics[key] != run[key]]
    notes = []
    written = file_physics.get("CU_PHYSICS")
    if written is not None and written != run["CU_PHYSICS"]:
        if written in STATELESS_CUMULUS and run["CU_PHYSICS"] in STATELESS_CUMULUS:
            notes.append(
                f"cu_physics {written} -> {run['CU_PHYSICS']}: both schemes "
                "rebuild their tendencies every call, so the change takes "
                "effect at the first step; the restart's last cumulus "
                "tendencies and RAINCV are carried as written")
        else:
            problems.append(
                f"CU_PHYSICS: the restart was written by {written}, the run "
                f"selects {run['CU_PHYSICS']}, and one of them holds state "
                "across steps (Kain-Fritsch's NCA timer and W0AVG)")
    if problems:
        raise ValueError(
            f"{path} cannot be continued by this configuration; its "
            "scheme-held state belongs to different physics:\n"
            + "\n".join(f"  {row}" for row in problems)
            + "\nRestart with the namelist that wrote the file.")
    return notes


# ---------------------------------------------------------------------------
# The physics overlay: every cross-step carrier filled or named.
# ---------------------------------------------------------------------------

#: Driver surface fields whose WRF restart name is not the upper-case
#: field name.
FIELD_ALIASES = MappingProxyType({
    "tke_myj": "TKE_PBL", "el_myj": "EL_PBL",
    "sfcrunoff": "SFROFF", "udrunoff": "UDROFF",
    "xice": "SEAICE",
})

#: Driver surface fields a WRF restart does not carry because both models
#: rewrite them before any read in the next step, by name.
FIELDS_REBUILT = MappingProxyType({
    **{name: "a per-call Noah forcing/residual work plane, rewritten by "
             "the next Noah call before it is read"
       for name in ("dz8w1", "qv1", "sfcprs", "sfctmp", "ebal", "noahres",
                    "reslin")},
    "chklowq": "Noah's surface-saturation flag, rewritten by the land "
               "surface call before MYJ's PBL reads it in the same step",
    "rib": "the Eta layer's bulk Richardson number, written by the surface "
           "layer before Noah reads it in the same step",
    "mixht": "MYJ's mixed-layer height output, rewritten every PBL call",
    **{name: "a surface-layer diagnostic rewritten by every surface-layer "
             "call before any read"
       for name in ("br", "cd", "cda", "ch", "chs", "chs2", "ck", "cka",
                    "cpm", "cqs2", "fh", "fm", "gz1oz0", "psih", "psim",
                    "qgh", "wspd", "zol", "u10e", "v10e")},
})

#: Scratch slots and their WRF sources.
SCRATCH_SOURCES = MappingProxyType({
    "mp_rainnc": "RAINNC", "mp_rainncv": "RAINNCV", "mp_snownc": "SNOWNC",
    "mp_snowncv": "SNOWNCV", "mp_graupelnc": "GRAUPELNC",
    "mp_graupelncv": "GRAUPELNCV", "mp_hailnc": "HAILNC",
    "mp_hailncv": "HAILNCV", "mp_sr": "SR", "mp_kessler_sr": "SR",
    "cu_rainc": "RAINC", "cu_nca": "NCA", "cu_pratec": "PRATEC",
    "cu_raincv": "RAINCV", "cu_rthcuten": "RTHCUTEN",
    "cu_rqvcuten": "RQVCUTEN", "cu_rqccuten": "RQCCUTEN",
    "cu_rqicuten": "RQICUTEN", "cu_rqrcuten": "RQRCUTEN",
    "cu_rqscuten": "RQSCUTEN",
    # WRF diff_opt = 1 mixes t - t_init (module_big_step_utilities_em.F
    # horizontal_diffusion_3dmp, base_3d = t_init), and t_init is kept in
    # the restart; woof's coordinate reference is that field.
    "diff1_theta_initial": "T_INIT",
})

#: Kain-Fritsch's held rain (cu_physics = 1): the only woof cumulus that
#: holds a rate between calls (NCA) and accumulates RAINC from it every step
#: (woof.core.cumulus_clock.advance: RAINC += PRATEC * DT).  Every other
#: scheme adds each call's own rain and leaves these three at the values the
#: driver starts them with.  WRF writes PRATEC, RAINCV and NCA for those
#: schemes too, as the last call's diagnostics; restoring them armed the
#: clock with a rate nothing ever cleared, so a cu6 or cu16 warm start
#: accumulated its restart-instant PRATEC on top of each step's own rain:
#: twice WRF's convective rain over NCAR's CONUS 12 km restart hour (0.0313
#: mm against 0.0154 mm mean), and the same excess into RAINBL.
HELD_RAIN_CU_PHYSICS = 1
HELD_RAIN_COLD_VALUES = MappingProxyType({
    "cu_nca": -100.0, "cu_pratec": 0.0, "cu_raincv": 0.0})


def scratch_restore_plan(name: str, cu_physics: int):
    """How the warm start fills one ``scratch/`` carrier.

    ``("fill", WRF_NAME)`` copies the restart variable; ``("cold", value,
    reason)`` starts the carrier as the driver does; ``None`` names no rule.
    """
    if (name in HELD_RAIN_COLD_VALUES
            and int(cu_physics) != HELD_RAIN_CU_PHYSICS):
        return ("cold", HELD_RAIN_COLD_VALUES[name],
                f"held-rain carrier of Kain-Fritsch only; cu_physics = "
                f"{int(cu_physics)} adds each call's rain itself, so WRF's "
                f"{SCRATCH_SOURCES[name]} (a last-call diagnostic) is not "
                "a rate to keep accumulating")
    if name in SCRATCH_SOURCES:
        return ("fill", SCRATCH_SOURCES[name])
    if name in SCRATCH_REBUILT:
        return ("cold", 0.0, SCRATCH_REBUILT[name])
    return None


SCRATCH_REBUILT = MappingProxyType({
    "up_heli_max": "the UP_HELI_MAX running maximum restarts its window, as "
                   "it does after every history write",
    # Correct mode's settled-cloud booking (woof/core/thompson_cloud_booking.py,
    # the 2.9.0 default for Thompson).  Stock WRF keeps no such total, so a
    # wrf.exe restart cannot carry it; without this row every default-mode
    # Thompson warm start from a WRF restart was refused, NCAR's v4.4
    # benchmark restart leg among them.
    "mp_cloudnc": "the settled-cloud total CLOUDNC that correct mode books; "
                  "wrf.exe keeps none, so it counts from the restart instant",
    "mp_cloudncv": "the per-call settled-cloud amount CLOUDNCV; rewritten at "
                   "the first microphysics call after the restart",
})


def _fields_absent_rules() -> dict[str, tuple]:
    from woof.core.physics_inventory import (RADIATION_BOUNDARY_FLUXES,
                                              SURFACE_ENERGY_ACCUMULATORS)

    rules: dict[str, tuple] = {}
    for name in (*(f"ac{flux}" for flux in RADIATION_BOUNDARY_FLUXES),
                 *SURFACE_ENERGY_ACCUMULATORS):
        rules[name] = ("cold", 0.0, (
            f"{name.upper()} is an accumulation: wrf.exe continues it from a "
            "restart that carries it and starts it at zero from one that "
            "does not, so it counts from the restart instant"))
    for name in RADIATION_BOUNDARY_FLUXES:
        rules[name] = ("cold", 0.0, (
            f"{name.upper()} is written by the first radiation call after "
            "the restart; zero until then, as in a wrf.exe restart without it"))
    for name in ("swddir", "swddif", "swddni"):
        # Allocated for every shortwave scheme since 04e65faa5f (native solar
        # products in default history); NCAR's benchmark restart carries
        # none of them and refused here by name.
        rules[name] = ("cold", 0.0, (
            f"{name.upper()} is the shortwave direct/diffuse split written by "
            "the first radiation call after the restart; zero until then, as "
            "in a wrf.exe restart without it"))
    rules["coszen"] = ("cold", 0.0, (
        "COSZEN is recomputed at the first radiation call after the restart"))
    rules["swnorm"] = ("cold", 0.0, (
        "SWNORM is rewritten by the slope-radiation producer at the first "
        "radiation call; without slope_rad it holds zero, as in WRF"))
    rules["refd_max"] = ("cold", 0.0, (
        "REFD_MAX is never restart-written by WRF (Registry rh02); its "
        "running maximum restarts at zero, as after every history write"))
    rules["sstsk"] = ("cold", 0.0, (
        "SSTSK has no initializer in WRF (phy_init, start_em) and no "
        "producer without sst_skin, so a restart without it holds zero, as "
        "woof's own cold start does"))
    rules["p_hyd"] = ("copy", "state/p", (
        "P_HYD diagnosed from the restored state (total pressure, as start_em "
        "and the driver initialize it); the first step's phy_prep rewrites it"))
    rules["xicem"] = ("copy", "fields/xice", (
        "XICEM = XICE, as WRF's phy_init sets the previous-step sea-ice flag"))
    return rules


def fields_restore_plan(name: str, restart_names) -> tuple | None:
    """How the warm start fills one ``fields/`` carrier.

    ``("fill", WRF_NAME)`` copies the restart variable; ``("cold", value,
    reason)`` starts it as wrf.exe starts a field its restart lacks;
    ``("copy", key, reason)`` takes another restored carrier (``state/p``
    or ``fields/<name>``); ``None`` names no rule.

    Named breakage: the exact chain made WRF's history rows (the radiation
    flux rows and their AC* totals, ACHFX/ACLHF/ACGRDFLX, COSZEN, P_HYD,
    REFD_MAX, SSTSK, SWNORM, XICEM) cross-step carriers, and NCAR's
    documented benchmark warm start exited 2 here naming every one.
    """
    if name == "rainbl":
        return ("fill", "RAINBL")
    wrf_name = FIELD_ALIASES.get(name, name.upper())
    if wrf_name in restart_names:
        return ("fill", wrf_name)
    if name in FIELDS_REBUILT:
        return ("rebuilt", FIELDS_REBUILT[name])
    from woof.ocean import seaice
    if name in seaice.COLD_SEEDED:
        # WRF has no prognostic ice column, so its restart carries none of
        # this model's ice state: the column is seeded from the restart's
        # XICE, TSK, SST and SNOW exactly as a cold start seeds it.  Named
        # breakage: NCAR's benchmark warm start (default seaice_prognostic)
        # refused here naming ICE_ENERGY, ICE_GROWTH and the rest.
        return ("seaice", "the prognostic sea ice column, which WRF does not "
                "carry, cold-seeded from the restart's XICE, TSK, SST and "
                "SNOW as a cold start seeds it (woof/ocean/seaice)")
    return _fields_absent_rules().get(name)


STATE_SOURCES = MappingProxyType({
    "rthften": "RTHFTEN", "rqvften": "RQVFTEN", "tke": "TKE_2",
})


def _restored_state_reasons(cfg) -> dict[str, str]:
    """``state/`` carriers the shared restore already filled, and from what."""
    from woof.ingest.wrfinput import active_moisture_map

    reasons = {
        "u": "U_2", "v": "V_2", "w": "W_2", "php": "PH_2", "mup": "MU_2",
        "thp": "THM_2 (dry theta about the file's base state)",
        "h_diabatic": "H_DIABATIC",
        "p": "EOS of the restored prognostics (WRF's P compared)",
        "al": "EOS of the restored prognostics (WRF's AL compared)",
        "alt": "EOS of the restored prognostics (WRF's ALT compared)",
    }
    for wrf_name, state_name in active_moisture_map(cfg).items():
        reasons[state_name] = wrf_name
    return reasons


@dataclass
class OverlayReport:
    """Per-carrier record of the warm start's physics overlay."""

    filled: dict[str, str] = field(default_factory=dict)
    rebuilt: dict[str, str] = field(default_factory=dict)
    checks: dict[str, object] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def receipt(self) -> dict:
        return {"filled": dict(sorted(self.filled.items())),
                "rebuilt": dict(sorted(self.rebuilt.items())),
                "checks": dict(self.checks), "notes": list(self.notes)}


def _host_field(raw, name: str, target_shape, *, label: str) -> np.ndarray:
    value = np.asarray(raw[name])
    if value.shape == tuple(target_shape):
        return value
    if (value.ndim == 3 and len(target_shape) == 3
            and value.shape[1:] == tuple(target_shape[1:])
            and value.shape[0] == target_shape[0] + 1):
        # WRF allocates these PBL carriers on kms:kme and writes them on
        # bottom_top_stag, but the schemes define them on the nz mass
        # levels only (MYJ: K = KTS..KTE-1 of the staggered extent); the
        # top word is the allocation's unused level.
        return value[:target_shape[0]]
    raise ValueError(
        f"WRF restart {name} shape {value.shape} cannot fill {label} "
        f"shape {tuple(target_shape)}")


def _copy_into(target, value) -> None:
    import cupy as cp
    target[...] = cp.asarray(np.ascontiguousarray(value), dtype=target.dtype)


def apply_warm_start_physics(state, driver, restored, cfg) -> OverlayReport:
    """Fill every cross-step carrier of ``state``/``driver`` from the restart.

    Walks the checkpoint manifest (:func:`woof.io.restart.state_manifest`,
    the scratch manifest and the driver manifest) -- the exact set a woof
    checkpoint must carry for a resume to be exact -- and fills each member
    from its WRF variable, or records the reason none is needed.  A member
    with neither refuses the start by name.
    """
    from woof.io import restart as restart_io

    raw = restored.raw
    report = OverlayReport()
    restored_state = _restored_state_reasons(cfg)
    # Carriers a run creates lazily at its first step, which a restart
    # instant has long passed: created here so the manifest walk below
    # fills them, exactly as a checkpoint resume finds them in its file.
    if int(getattr(cfg, "diff_opt", 0)) == 1:
        # woof.core.dycore.initialize_coordinate_reference refuses to
        # rebuild this past t = 0, and rightly: it is WRF's t_init.
        state.scratch(state.thp.shape, "diff1_theta_initial")
    adapter = None if driver is None else getattr(driver, "cumulus_callable", None)
    if adapter is not None and hasattr(adapter, "w0avg") and "W0AVG" in raw:
        # Kain-Fritsch's trigger history, on the adapter as a resume puts
        # it (woof.io.restart._restore_driver), bound to THIS state so the
        # next update does not re-zero it.
        import cupy as cp
        value = np.asarray(raw["W0AVG"], np.float32)
        if value.shape[0] == state.thp.shape[0] + 1:
            value = value[:state.thp.shape[0]]
        adapter.w0avg = cp.asarray(np.ascontiguousarray(value))
        adapter._history_state = state
        adapter._history_time = float(restored.warm_start.clock.elapsed_seconds)
    manifest = {}
    manifest.update(restart_io.state_manifest(state))
    manifest.update(restart_io._scratch_manifest(state))
    driver_manifest = ({} if driver is None
                       else restart_io._driver_manifest(driver))
    manifest.update(driver_manifest)
    unmapped: list[str] = []
    copies: list[tuple] = []
    seaice_seeded: set[str] = set()

    def fill(key, target, wrf_name, *, transform=None):
        if wrf_name not in raw:
            unmapped.append(f"{key} <- {wrf_name} (absent from the restart)")
            return
        value = _host_field(raw, wrf_name, target.shape, label=key)
        if transform is not None:
            value = transform(value)
        _copy_into(target, value)
        report.filled[key] = wrf_name

    def micron(value):
        # WRF carries effective radii in metres; woof's radiation-facing
        # state is microns.  WRF's RRTMG makes the same FP32 product
        # (module_ra_rrtmg_lw.F: reliq = re_cloud*1.E6).
        return np.asarray(np.asarray(value, np.float32) * np.float32(1.0e6),
                          np.float32)

    for key in sorted(manifest):
        target = manifest[key]
        namespace, _, name = key.partition("/")
        if namespace == "state":
            if name in restored_state:
                report.filled[key] = restored_state[name]
            elif name in STATE_SOURCES:
                fill(key, target, STATE_SOURCES[name])
            elif name in ("effc", "effi", "effs"):
                fill(key, target, {"effc": "RE_CLOUD", "effi": "RE_ICE",
                                   "effs": "RE_SNOW"}[name], transform=micron)
            else:
                unmapped.append(key)
        elif namespace == "acoustic" and name == "ww_pp":
            target[...] = 0
            report.rebuilt[key] = (
                "the acoustic Omega'' a woof checkpoint carries for the "
                "forced outer column; WRF's restart has no counterpart (its "
                "small-step prep rederives omega), so it starts at zero on "
                "the specified row the lateral boundary rewrites")
        elif namespace == "scratch":
            plan = scratch_restore_plan(name, int(cfg.cu_physics))
            if plan is None:
                unmapped.append(key)
            elif plan[0] == "fill":
                fill(key, target, plan[1])
            else:
                target[...] = plan[1]
                report.rebuilt[key] = plan[2]
        elif namespace == "fields":
            plan = fields_restore_plan(name, raw)
            if plan is None:
                wrf_name = FIELD_ALIASES.get(name, name.upper())
                unmapped.append(f"{key} <- {wrf_name} (absent from the restart)")
            elif plan[0] == "fill":
                fill(key, target, plan[1])
            elif plan[0] == "rebuilt":
                report.rebuilt[key] = plan[1]
            elif plan[0] == "seaice":
                seaice_seeded.add(name)
                report.rebuilt[key] = plan[1]
            elif plan[0] == "cold":
                target[...] = plan[1]
                report.rebuilt[key] = plan[2]
            else:
                # After the walk: the source carrier is filled by then.
                copies.append((key, target, plan[1], plan[2]))
        elif key == "driver/rthratenlw":
            fill(key, target, "RTHRATLW")
        elif key == "driver/rthratensw":
            fill(key, target, "RTHRATSW")
        elif key == "driver/pending_rainbl":
            # WRF's surface driver adds the last step's RAINCV and RAINNCV
            # to RAINBL at the next call (module_surface_driver.F:1566-1568);
            # woof accumulates those same two amounts here as they are
            # produced, so the pending amount at the restart instant is
            # theirs.  RAINBL itself rides the fields above.
            parts = [name for name in ("RAINCV", "RAINNCV", "RAINSHV")
                     if name in raw]
            if not {"RAINCV", "RAINNCV"} <= set(parts):
                unmapped.append(f"{key} <- RAINCV + RAINNCV")
                continue
            total = np.zeros(target.shape, np.float32)
            for part in parts:
                total = np.asarray(total + np.maximum(
                    np.asarray(raw[part], np.float32), np.float32(0.0)),
                    np.float32)
            _copy_into(target, total)
            report.filled[key] = " + ".join(parts)
        elif key == "held/gf_rthblten":
            fill(key, target, "RTHBLTEN")
        elif key == "held/gf_rqvblten":
            fill(key, target, "RQVBLTEN")
        elif namespace == "pbl" and not name.startswith("diagnostics/"):
            source = {"du": "RUBLTEN", "dv": "RVBLTEN", "dtheta": "RTHBLTEN",
                      "dqv": "RQVBLTEN", "dqc": "RQCBLTEN",
                      "dqi": "RQIBLTEN"}.get(name)
            if source is None:
                unmapped.append(key)
            else:
                fill(key, target, source)
        elif key == "diag/rthraten":
            # The output-only radiative heating total, which the radiation
            # call writes as longwave + shortwave and history publishes
            # until the next call (the next call recomputes it).
            if not {"RTHRATLW", "RTHRATSW"} <= set(raw):
                unmapped.append(f"{key} <- RTHRATLW + RTHRATSW")
                continue
            total = np.asarray(
                np.asarray(_host_field(raw, "RTHRATLW", target.shape,
                                       label=key), np.float32)
                + np.asarray(_host_field(raw, "RTHRATSW", target.shape,
                                         label=key), np.float32),
                np.float32)
            _copy_into(target, total)
            report.filled[key] = "RTHRATLW + RTHRATSW"
        elif key == "diag/olr":
            fill(key, target, "OLR")
        elif key == "diag/cldfra":
            fill(key, target, "CLDFRA")
        elif key == "cumulus/w0avg":
            # Kain-Fritsch's running vertical-velocity average (WRF W0AVG,
            # restart-carried for exactly this reason).
            fill(key, target, "W0AVG")
        elif key == "radiation/o33d_grid":
            report.rebuilt[key] = (
                "the root domain's legacy-RRTMG ozone, rebuilt at every "
                "radiation call (WRF's O3RAD likewise)")
        elif namespace == "driver" and name.split("/")[0] in (
                "radiation_tendencies", "cumulus_tendencies",
                "pbl_tendencies", "fire_tendencies"):
            # Coupled compositions of the held rates above; recomposed
            # below from the restored rates and the restored dry mass.
            continue
        else:
            unmapped.append(key)
    for key, target, source, reason in copies:
        if source == "state/p":
            value = state.p
        elif source in manifest:
            value = manifest[source]
        else:
            unmapped.append(f"{key} <- {source} (not a carrier of this model)")
            continue
        target[...] = value.astype(target.dtype, copy=False)
        report.rebuilt[key] = reason
    if seaice_seeded and driver is not None:
        from woof.ocean import seaice
        seaice.cold_seed_in_place(driver.fields, cfg, seaice_seeded)
    if unmapped:
        raise ValueError(
            "the WRF restart cannot fill these cross-step carriers of the "
            "configured model, and no reason says they need no value:\n"
            + "\n".join(f"  {row}" for row in unmapped)
            + "\nThe map is woof/ingest/wrfrst.py (apply_warm_start_physics)")
    if driver is not None:
        _recompose_held_tendencies(state, driver, cfg, report)
        _warm_start_driver_counters(driver, restored, cfg, report)
    _record_eos_checks(state, raw, report)
    return report


def _recompose_held_tendencies(state, driver, cfg, report) -> None:
    """Couple the restored held rates with the restored dry mass.

    WRF holds its physics tendencies UNCOUPLED (RTHRATEN, RTHCUTEN, ...) and
    couples them with the current column mass every step
    (module_physics_addtendc.F); woof holds the coupled form.  At the
    restart instant the two agree exactly when coupled with the restored
    mass, which is what the relocation path does for a moved nest.
    """
    from woof.core.physics import couple_column_tendencies
    from woof.core.physics_inventory import physics_reuses_pbl_composition

    if driver.radiation_active:
        driver.radiation_tendencies = couple_column_tendencies(
            state, cfg, rtheta=driver.rthratenlw + driver.rthratensw)
        report.filled["driver/radiation_tendencies/*"] = (
            "RTHRATLW + RTHRATSW coupled with the restored dry mass")
    if driver.cu_physics:
        driver.recouple_cumulus_tendencies(state, cfg)
        report.filled["driver/cumulus_tendencies/*"] = (
            "R*CUTEN coupled with the restored dry mass")
    if physics_reuses_pbl_composition(cfg) or not getattr(
            driver, "pbl_raw_rates", {}):
        report.rebuilt["driver/pbl_tendencies/*"] = (
            "bldt = 0: the PBL call of the first step writes it before the "
            "composition reads it")
    else:
        from woof.core.physics import couple_ysu_tendencies
        driver.pbl_tendencies = couple_ysu_tendencies(
            state, cfg, driver.pbl_raw_rates)
        report.filled["driver/pbl_tendencies/*"] = (
            "R*BLTEN coupled with the restored dry mass")


def _last_radiation_step(itimestep: int, stepra: int) -> int:
    """The step of WRF's last radiation call at or before ``itimestep``.

    module_radiation_driver.F calls radiation on step 1 and whenever
    ``MOD(itimestep, STEPRA) == 1`` (ra_call_offset 0).
    """
    if stepra <= 1:
        return itimestep
    last = itimestep - ((itimestep - 1) % stepra)
    return max(last, 1)


def _warm_start_driver_counters(driver, restored, cfg, report) -> None:
    """Counters and provenance the producing run had at the restart instant."""
    from woof.core.physics import _physics_interval_steps
    from woof.core.radiation_carriers import (
        CARRIER_SOURCE_RADIATION_SCHEME, consumer_carriers)

    warm = restored.warm_start
    itimestep = int(warm.clock.itimestep)
    dt = float(cfg.dt)
    raw = restored.raw
    stepra = _physics_interval_steps(float(driver.radt_minutes), dt)
    written = {key: int(np.asarray(raw[key]).reshape(-1)[0])
               for key in CADENCE_KEYS if key in raw}
    report.checks.update(compare_restart_cadences(
        written, cfg, radt_minutes=float(driver.radt_minutes),
        radiation_active=bool(driver.radiation_active)))
    last = _last_radiation_step(itimestep, stepra)
    produced_at = (last - 1) * dt
    if driver.radiation_active:
        for name in consumer_carriers(int(cfg.sf_surface_physics)):
            if name in driver.fields:
                driver.carriers.declare(
                    name, source=CARRIER_SOURCE_RADIATION_SCHEME,
                    model_time=produced_at)
        report.checks["radiation_last_call_step"] = last
        report.checks["radiation_carriers_written_at_s"] = produced_at
    # The producing run's microphysics ran every one of its steps.
    driver.microphysics_updates = max(int(driver.microphysics_updates),
                                      itimestep)


def _record_eos_checks(state, raw, report) -> None:
    """How far woof's EOS of the restored prognostics sits from WRF's P/AL."""
    import cupy as cp

    for name, wrf_names, live in (("P", ("P", "PB"), state.p),
                                  ("ALT", ("ALT",), state.alt)):
        if not all(part in raw for part in wrf_names):
            continue
        expected = np.zeros(live.shape, np.float64)
        for part in wrf_names:
            expected += np.asarray(raw[part], np.float64)
        diff = np.abs(cp.asnumpy(live).astype(np.float64) - expected)
        report.checks[f"eos_{name.lower()}_max_abs"] = float(diff.max())
        report.checks[f"eos_{name.lower()}_rms"] = float(np.sqrt(np.mean(diff * diff)))


def apply_warm_start_clock(model, nodes, clocks_by_grid: Mapping[int, WrfRestartClock]):
    """Open every domain's clock at the restart instant, as a resume does.

    Mirrors :func:`woof.io.restart.restore_tree_restart`'s clock half: the
    tick count, the step count, WRF's boundary-interval clock and the
    state's elapsed seconds, after ``attach_lateral_boundaries`` reset them.
    Returns the restart info the tree runner reads.
    """
    from woof.core.model import PERIOD_BEGIN, ModelRuntimeStatus
    from woof.io.restart import TreeRestartInfo

    tick_den = model.schedule.clock.tick_den
    ticks_seen = set()
    for gid, node in nodes.items():
        clock_in = clocks_by_grid.get(int(gid))
        if clock_in is None:
            raise ValueError(
                f"d{int(gid):02d} has no WRF restart file; every domain of a "
                "restarted tree continues from its own wrfrst")
        exact = clock_in.elapsed_seconds * tick_den
        ticks = int(round(exact))
        if abs(ticks - exact) > 1e-6:
            raise ValueError(
                f"the restart instant {clock_in.elapsed_seconds:g} s is not a "
                "whole number of model ticks")
        ticks_seen.add(ticks)
        clock = node.clock
        if ticks % model.schedule.period_ticks != 0:
            raise ValueError(
                f"the restart instant {clock_in.elapsed_seconds:g} s is not "
                "on a root step of this configuration")
        if ticks > model.schedule.clock.run_ticks:
            raise ValueError(
                f"the restart instant {clock_in.elapsed_seconds:g} s lies "
                "after this run's end")
        clock.ticks = ticks
        clock.step_count = max(0, (ticks - clock.spec.start_ticks)
                               // clock.spec.step_ticks)
        if clock.step_count != clock_in.itimestep:
            raise ValueError(
                f"d{int(gid):02d}: WRF took {clock_in.itimestep} steps to the "
                f"restart instant, this configuration's step makes it "
                f"{clock.step_count}; the time step differs from the run that "
                "wrote the file")
        clock.dtbc_fp32 = np.float32(clock_in.dtbc_seconds)
        node.state.elapsed_seconds = ticks / tick_den
        node._started = True
        driver = getattr(node.state, "physics", None)
        from woof.core.physics_inventory import REFL_10CM_MICROPHYSICS
        from woof.core.refl import (analysis_refl_10cm, refl_10cm_is_stashed,
                                     stash_refl_10cm)
        if (driver is not None and not refl_10cm_is_stashed(node.state)
                and node.state.qv is not None
                and driver.mp_physics in REFL_10CM_MICROPHYSICS
                and clock.history_due()):
            # The restart instant's history frame: WRF's REFL_10CM is
            # history-only (Registry "hdu", no restart flag), so wrf.exe
            # writes its zero-filled array at a restart's first frame
            # (write_hist_at_0h_rst), and so does this one.  Only when that
            # frame is due: an unconsumed stash would refuse the next
            # microphysics call's.  Through the handoff's own API: the
            # stash is read and written by woof.core.refl alone
            # (tests/test_refl.py, output-only handoff).
            stash_refl_10cm(node.state, analysis_refl_10cm(node.state))
    if len(ticks_seen) != 1:
        raise ValueError(
            f"the domains' restart files are at different instants: "
            f"{sorted(t / tick_den for t in ticks_seen)} s")
    (ticks,) = ticks_seen
    model._runtime_status = ModelRuntimeStatus()
    model._resumed = True
    model._resume_committed_history_grid_ids = frozenset()
    return TreeRestartInfo(
        elapsed_ticks=ticks, tick_den=tick_den, phase=PERIOD_BEGIN,
        paths_by_grid_id={gid: clocks_by_grid[int(gid)].path for gid in nodes},
        headers_by_grid_id={},
        already_complete=ticks == model.schedule.clock.run_ticks)


__all__ = [
    "Disposition", "DISPOSITION_KINDS", "FIELD_ALIASES", "FIELDS_REBUILT",
    "HELD_RAIN_COLD_VALUES", "HELD_RAIN_CU_PHYSICS",
    "OverlayReport", "PROGNOSTIC_RENAMES", "SCRATCH_SOURCES",
    "STATEFUL_SELECTORS", "WRFRST_DISPOSITIONS", "WarmStart",
    "WrfRestartClock", "apply_warm_start_clock", "apply_warm_start_physics",
    "check_restart_physics", "discover_restart_files",
    "read_wrfrst", "read_wrfrst_clock", "restart_file_name",
]
