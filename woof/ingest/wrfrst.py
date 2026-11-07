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

MOIST THETA. Under ``USE_THETA_M = 1`` the prognostic ``THM_2`` is
retained as THM for a moist-thermal run. A separate dry T view is computed
with WRF's FP32 constants for physics/history and compared with file T.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping

import numpy as np

from woof import netcdf_bridge
from woof import wrf_exact

from woof.core.clwrf import HISTORY as CLWRF_HISTORY
from woof.core.rasm import NATIVE_SOURCES as RASM_SOURCES, WORK as RASM_WORK
from woof.core.solar_diagnostics import HISTORY as SOLAR_HISTORY, WORK as SOLAR_WORK

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

from woof.io.cam5_schema import CAM_OUTPUT_ROWS
_CAM_RESTART_NAMES = tuple(row[0].strip().upper() for row in CAM_OUTPUT_ROWS.values())

#: Every variable of a WRF 4.x restart, by name.  Built from the WRF v4.4
#: CONUS-suite restart NCAR publishes with its v4.4 benchmarks (577
#: variables, MEASURED 2026-10-09 from wrfrst_d01_2019-11-26_23:00:00) and
#: the WRF v4.6.1 Registry the names come from.
WRFRST_DISPOSITIONS: Mapping[str, Disposition] = MappingProxyType({
    **_rows(_READ, "CAM coupled cloud, precipitation and shallow state",
            _CAM_RESTART_NAMES),
    **_rows(_READ, "CAM UW PBL stresses, perturbations and interface diagnostics",
            ("TAURESX2D", "TAURESY2D", "TPERT2D", "QPERT2D", "WPERT2D",
             "TURBTYPE3D", "SMAW3D")),

    **_rows(_READ, "cloud feedback averages and FARMS/aerosol diagnostics held across steps", (
        "GD_CLOUD_A", "GD_CLOUD2_A", "GD_CLDFR", "QC_CU", "QI_CU",
        "CLDFRA_DP", "CLDFRA_SH", "RAINCV_A", "RAINCV_B", "SWDOWN2",
        "SWDDIR2", "SWDDNI2", "SWDDIF2", "SWDOWNC2", "SWDDNIC2",
        "TAOD5502D", "TAOD5503D", "ECOBSC", "ECMASK", "ELAT_TRACK", "ELON_TRACK")),
    **_rows("rebuilt", "instantaneous Grell cloud input rewritten before convtrans_prep at the next cumulus call", ("GD_CLOUD", "GD_CLOUD2")),
    **_rows("setup", "annual vegetation average; Noah consumes SHDFAC and the monthly extrema instead", ("SHDAVG",)),
    # -- the dynamical core -------------------------------------------
    **_rows(_READ, "current prognostic (restored as the wrfinput name)",
            tuple(PROGNOSTIC_RENAMES)),
    **_rows(_READ, "prognostic theta (moist under USE_THETA_M = 1); restored "
            "as selected prognostic THM, with a dry T view", ("THM_2",)),
    **_rows(_READ, "WRF's dry theta diagnostic th_phy_m_t0; compared with "
            "the restored theta", ("T",)),
    **_rows("time_level", "RK time-t copy: wrf.exe overwrites it from the "
            "_2 level at the next step's entry, and woof rebuilds its own "
            "u0/v0/w0/thp0/php0/mup0 there",
            ("U_1", "V_1", "W_1", "PH_1", "MU_1", "TKE_1", "PC_1")),
    **_rows(_READ, "previous thermal time level used by moist-theta physics "
            "conversion before small_step_prep refreshes it", ("THM_1",)),
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
    **_rows(_READ, "km_opt = 2/5 prognostic SGS TKE", ("TKE_2",)),
    **_rows(_READ, "SMS-3DTKE retained nonlocal momentum", ("GAMU", "GAMV")),
    **_rows(_READ, "map factors, Coriolis and grid rotation",
            ("MAPFAC_M", "MAPFAC_U", "MAPFAC_V", "MAPFAC_MX", "MAPFAC_MY",
             "MAPFAC_UX", "MAPFAC_UY", "MAPFAC_VX", "MAPFAC_VY", "F", "E",
             "SINALPHA", "COSALPHA", "HGT")),
    **_rows(_READ, "subgrid orography and annual vegetation statics, "
            "consumed by the ordinary WRF input/static restore",
            ("VAR", "CON", "OA1", "OA2", "OA3", "OA4",
             "OL1", "OL2", "OL3", "OL4", "SHDAVG")),
    **_rows("rebuilt", "gravity-wave stress diagnostics; module_bl_gwdo.F "
            "rewrites them each PBL call; held momentum is restored "
            "from RUBLTEN/RVBLTEN",
            ("DTAUX3D", "DTAUY3D", "DUSFCG", "DVSFCG")),
    **_rows(_READ, "grid coordinates", (
        "XLAT", "XLONG", "XLAT_U", "XLONG_U", "XLAT_V", "XLONG_V")),
    **_rows(_READ, "static orography consumed by the ported gwd_opt = 1 "
            "terrain drag (Registry.EM_COMMON orogwd fields)",
            ("VAR", "CON", "OA1", "OA2", "OA3", "OA4",
             "OL1", "OL2", "OL3", "OL4")),
    **_rows("rebuilt", "gravity-wave drag outputs: module_bl_gwdo.F "
            "declares these intent(out); the next call rebuilds them "
            "before use, including the integrated surface stresses",
            ("DTAUX3D", "DTAUY3D", "DUSFCG", "DVSFCG")),
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
    **_rows(_READ, "non-Chem EM dimensionless passive-tracer state",
            tuple(f"tr17_{index}" for index in range(1, 9))),
    **_rows(_READ, "native EM trajectory coordinates; inactive when traj_opt=0",
            ("TRAJ_I", "TRAJ_J", "TRAJ_K", "TRAJ_LONG", "TRAJ_LAT")),
    # -- moisture ------------------------------------------------------
    **_rows(_READ, "moisture species and number concentrations (the "
            "active scheme's inventory is checked by the moisture map)",
            ("QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP", "QHAIL",
             "QNICE", "QNRAIN", "QNSNOW", "QNGRAUPEL", "QNHAIL", "QNCLOUD",
             "QNWFA", "QNIFA", "QNBCA", "QNCCN", "QNDROP", "QVGRAUPEL",
             "QVHAIL", "QRIMEF", "QIR", "QIB", "QNWFA2D", "QNIFA2D", "CWM",
             "QICE2", "QNICE2", "QIR2", "QIB2")),
    **_rows(_READ, "P3's previous-call thermodynamic fields; the next "
            "supersaturation calculation consumes these saved values",
            ("TH_OLD", "QV_OLD")),
    **_rows("inactive", "annual mean vegetation fraction consumed only "
            "by Noah mosaic (module_surface_driver.F:2677)", ("SHDAVG",),
            lambda c: int(getattr(c, "sf_surface_mosaic", 0)) == 1),
    **_rows("rebuilt", "P3's output-only ice diagnostics, recomputed by "
            "the next microphysics call (Registry EM_COMMON P3 rows)",
            ("v_ice", "d_ice", "rho_ice", "v_ice2", "d_ice2", "rho_ice2")),
    **_rows(_READ, "ISHMAEL's additional prognostic ice habits and volume "
            "moments; restored through the active moisture map", (
                "QICE3", "QNICE3", "QVOLI", "QAOLI",
                "QVOLI2", "QAOLI2", "QVOLI3", "QAOLI3")),
    **_rows(_READ, "Ferrier-Aligo condensate partition and retained rime factor",
            ("F_ICE_PHY", "F_RAIN_PHY", "F_RIMEF_PHY")),
    **_rows(_READ, "microphysics effective radii, read by the next "
            "radiation call", ("RE_CLOUD", "RE_ICE", "RE_SNOW")),
    **_rows(_READ, "GSFC effective radii in microns: mp=7 supplies them "
            "directly; other radius providers rebuild them from RE_CLOUD/"
            "RE_ICE/RE_SNOW before Goddard radiation (WRF radiation_driver)",
            tuple(f"RE_{species}_GSFC" for species in
                  ("CLOUD", "ICE", "RAIN", "SNOW", "GRAUPEL", "HAIL"))),
    **_rows(_READ, "WRF's effective-radius ownership flags, compared with "
            "the active scheme's", ("HAS_REQC", "HAS_REQI", "HAS_REQS")),
    **_rows("inactive", "previous-step vapour of the moisture-advection "
            "trigger (KF's kfeta_trigger = 2), which woof does not port",
            ("QVOLD",), lambda c: int(getattr(c, "kfeta_trigger", 1)) == 2),
    **_rows(_READ, "Kain-Fritsch's running vertical-velocity average",
            ("W0AVG",)),
    **_rows(_READ, "old Kain-Fritsch's held kf_edrates diagnostics",
            ("UDR_KF", "DDR_KF", "UER_KF", "DER_KF", "TIMEC_KF")),
    # -- precipitation and microphysics persistence ---------------------
    **_rows(_READ, "precipitation accumulators and last-step amounts",
            ("RAINNC", "RAINNCV", "SNOWNC", "SNOWNCV", "GRAUPELNC",
             "GRAUPELNCV", "HAILNC", "HAILNCV", "SR", "RAINC", "RAINCV",
             "PRATEC", "NCA", "RAINBL")),
    **_rows(_READ, "HAILCAST held embryo sizes, updraft mask and duration "
            "(Registry restart flag r); diagnostics are retained if present",
            tuple(f"HAILCAST_DHAIL{i}" for i in range(1, 6))
            + ("HAILCAST_WUP_MASK", "HAILCAST_WDUR", "HAILCAST_DIAM_MAX",
               "HAILCAST_DIAM_MEAN", "HAILCAST_DIAM_STD")),
    **_rows("bookkeeping", "adaptive HAILCAST next-call clock; adaptive "
            "time stepping is refused by read_wrfrst_clock, and the fixed "
            "cadence is derived from the restored run clock",
            ("HAILDTACTTIME",)),
    **_rows(_READ, "CAM UW shallow precipitation and its held rate",
            ("RAINSH", "PRATESH", "RAINSHV")),
    **_rows("inactive", "Eta/Ferrier microphysics restart tables "
            "(mp_physics = 85/95), not implemented; Ferrier-Aligo 5 rebuilds "
            "its pinned lookup tables at initialization",
            ("MP_RESTART_STATE", "TBPVS_STATE", "TBPVS0_STATE"),
            lambda c: int(c.mp_physics) in (85, 95)),
    # -- cumulus ---------------------------------------------------------
    **_rows(_READ, "cumulus tendencies of the last call", (
        "RTHCUTEN", "RQVCUTEN", "RQRCUTEN", "RQCCUTEN", "RQSCUTEN",
        "RQICUTEN", "RUCUTEN", "RVCUTEN")),
    **_rows(_READ, "CAM deep convection held number tendencies",
            ("RQCNCUTEN", "RQINCUTEN")),
    **_rows(_READ, "last-call cumulus cloud bounds (module_first_rk_step_part1.F:1396 "
            "passes CUTOP/CUBOT to the cumulus driver's HTOP/HBOT arguments)",
            ("CUTOP", "CUBOT")),
    **_rows("rebuilt", "cumulus cloud top/base "
            "diagnostics, rewritten by the next call",
            ("HTOPR", "HBOTR")),
    **_rows("inactive", "convective precipitation accumulated for GFDL "
            "ETARA radiation; native CAM does not read CUPPT "
            "(module_radiation_driver.F:1944,2804)", ("CUPPT",),
            lambda c: 99 in (int(getattr(c, "ra_lw_physics", 0)),
                            int(getattr(c, "ra_sw_physics", 0)))),
    **_rows(_READ, "BMJ cloud efficiency and carried convective clouds; "
            "Registry.EM_COMMON:1246-1247,1577-1578,1700-1701,1852 "
            "restart-writes these between cumulus and radiation calls",
            ("CLDEFI", "QCCONV", "QICONV", "CONVCLD", "CCLDFRA",
             "HTOP", "HBOT")),
    **_rows("rebuilt", "GF shallow output diagnostics, zeroed before any "
            "read by WRF v4.6.1 GFDRV:278-284 and recomputed every call; "
            "the GPU driver retains no previous-call input for these outputs",
            ("K22_SHALLOW", "KBCON_SHALLOW", "KTOP_SHALLOW", "XMB_SHALLOW")),
    **_rows("inactive", "Grell-family ensemble and closure state (cu 3/5/93 "
            "ensemble members, mass flux, cloud fractions); woof's GF "
            "recomputes its closures every call",
            ("MASS_FLUX", "APR_GR", "APR_W",
             "APR_MC", "APR_ST", "APR_AS", "APR_CAPMA", "APR_CAPME",
             "APR_CAPMI", "XF_ENS", "PR_ENS",
             )),
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
    **_rows(_READ, "held direct/diffuse shortwave and scheme boundary "
            "flux diagnostics; restore when the native file carries them",
            ("SWDDNI", "SWDDIR", "SWDDIF", "TLWDN", "TLWUP", "SLWDN",
             "SLWUP", "TSWDN", "TSWUP", "SSWDN", "SSWUP")),
    **_rows("rebuilt", "clear-sky heating and flux diagnostics of the last "
            "radiation call, rewritten by the next one",
            ("HRANG", "DECLIN", "SOLCON",
             "DIFFUSE_FRAC", "SWDDIRC", "SWDDNIC")),
    **_rows(_READ, "ozone profile held between radiation calls", ("O3RAD",)),
    **_rows(_READ, "CAM radiation diagnostics held until the next call",
            ("SWCF", "LWCF", "RTHRATLWC", "RTHRATSWC",
             "TAUCLDC", "TAUCLDI")),
    **_rows(_READ, "top and bottom radiative fluxes, the zenith cosine and "
            "the slope-normal shortwave of the last radiation call, held "
            "for history (and COSZEN for its consumers) until the next one",
            ("SWUPT", "SWUPTC", "SWDNT", "SWDNTC", "SWUPB", "SWUPBC",
             "SWDNB", "SWDNBC", "LWUPT", "LWUPTC", "LWDNT", "LWDNTC",
             "LWUPB", "LWUPBC", "LWDNB", "LWDNBC", "COSZEN", "SWNORM")),
    **_rows("inactive", "shortwave interpolation fit (swint_opt = 1), not "
            "selected", ("GX", "BX", "GG", "BB", "COSZEN_REF", "SWDOWN_REF",
                         "SWDDIR_REF"), lambda c: int(c.swint_opt)==1),
    **_rows("inactive", "spectral surface shortwave for Noah-MP/CLM, not "
            "selected", ("SWVISDIR", "SWVISDIF", "SWNIRDIR", "SWNIRDIF",
                         "ALSWVISDIR", "ALSWVISDIF", "ALSWNIRDIR",
                         "ALSWNIRDIF"),
            lambda c: int(c.sf_surface_physics) == 4),
    **_rows(_READ, "held aerosol optical diagnostics", ("AODTOT", "AOD5502D", "ANGEXP2D", "AERSSA2D", "AERASY2D")),
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
    **_rows(_READ, "TEMF carried turbulence, mass-flux diagnostics and "
            "surface velocity scale", (
        "TE_TEMF", "KH_TEMF", "KM_TEMF", "SHF_TEMF", "QF_TEMF", "UW_TEMF",
        "VW_TEMF", "WUPD_TEMF", "MF_TEMF", "THUP_TEMF", "QTUP_TEMF",
        "QLUP_TEMF", "CF3D_TEMF", "HD_TEMF", "LCL_TEMF", "HCT_TEMF",
        "CFM_TEMF", "WM_TEMF")),
    **_rows(_READ, "PBL tendencies of the last call", (
        "RUBLTEN", "RVBLTEN", "RTHBLTEN", "RQVBLTEN", "RQCBLTEN",
        "RQIBLTEN")),
    **_rows("inactive", "MYNN ice-number PBL tendency, not selected",
            ("RQNIBLTEN",)),
    **_rows(_READ, "MYJ PBL carried TKE and mixing length, exchange "
            "coefficients", ("TKE_PBL", "EL_PBL", "EXCH_H", "EXCH_M")),
    **_rows(_READ, "GBM TKE exchange diagnostic, when explicitly included "
            "in a native restart", ("EXCH_TKE",),
            lambda c: int(c.bl_pbl_physics) == 12),

    **_rows(_READ, "K-epsilon carried closure state, transport copies and turbulent fluxes",
            ("DISS_PBL", "TPE_PBL", "PR_PBL", "TKE_ADV", "DISS_ADV", "TPE_ADV",
             "WU_TUR", "WV_TUR", "WT_TUR", "WQ_TUR")),
    **_rows(_READ, "MYNN carried turbulence, moments and boundary-layer cloud state",
            ("QKE", "TSQ", "QSQ", "COV", "SH3D", "SM3D",
             "QC_BL", "QI_BL", "CLDFRA_BL")),
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
    **_rows("setup", "annual vegetation climatology used by real.exe to "
            "initialize FRC_URB2D (module_initialize_real.F:3159), restored "
            "separately; SHDAVG is an unused input of Noah's mosaic "
            "wrapper (module_sf_noahdrv.F:2360,2624), not evolved state",
            ("SHDAVG",)),
    **_rows(_READ, "soil, snow and surface state", (
        "TSK", "TSLB", "SMOIS", "SH2O", "SMCREL", "SNOW", "SNOWH", "SNOWC",
        "CANWAT", "ALBEDO", "ALBBCK", "EMISS", "EMBCK", "SNOTIME", "ACSNOM",
        "ACSNOW", "SFROFF", "UDROFF", "SMSTAV", "SMSTOT", "POTEVP", "SNOPCX",
        "GRDFLX", "TMN", "SST", "SEAICE", "XLAND", "LANDMASK", "LAKEMASK",

        # Registry.EM_COMMON:770 declares SHDAVG as a restart/history static.
        # Keep the native annual vegetation average alongside its extrema.
        "LU_INDEX", "IVGTYP", "ISLTYP", "VEGFRA", "SHDMAX", "SHDMIN", "SHDAVG",
        "SNOALB", "LAI", "FRC_URB2D")),
    **_rows("setup", "annual vegetation climatology used by real.exe to "
            "initialize FRC_URB2D (module_initialize_real.F:3143); Noah's "
            "driver declares it but never reads it "
            "(module_sf_noahdrv.F:2360,2624), and the restart restores "
            "FRC_URB2D itself", ("SHDAVG",)),
    **_rows(_READ, "surface energy accumulations (sensible, latent and "
            "ground heat), continued from the restart as wrf.exe continues "
            "them", ("ACHFX", "ACLHF", "ACGRDFLX")),
    **_rows(_READ, "land-surface evaporation accumulation (RUC carries and continues it)",
            ("SFCEVP",)),
    **_rows("rebuilt", "Noah's exchange coefficient output, rewritten by "
            "every call", ("SFCEXC",)),
    **_rows(_READ, "previous-step sea-ice flag, held for the sea-ice "
            "refresh and history", ("XICEM",)),
    **_rows(_READ, "sea-ice refresh skin temperature save", ("TSK_SAVE",)),
    **_rows(_READ, "SST skin temperature, continued from native state", ("SSTSK",)),
    **_rows(_READ, "SST warm-layer state", ("DTW",)),
    **_rows(_READ, "deep soil daily and annual temperature history",
            ("TYR", "TYRA", "TDLY", "TLAG", "NYEAR", "NDAY")),
    **_rows(_READ, "RUC carried soil, snow, surface moisture and accumulations",
            ("QSG", "QVG", "QCG", "DEW", "TSNAV", "SOILTB", "SOILT1",
             "RHOSNF", "SNOWFALLAC", "PRECIPFR", "ACRUNOFF",
             "SMFR3D", "KEEPFR3DFLAG")),
    **_rows(_READ, "thermal-diffusion Slab heat capacity and conductivity",
            ("CAPG", "THC")),
    **_rows("rebuilt", "GF cloud diagnostics are outputs of each GF call; "
            "the GF adapter has no consumer of the previous call's clouds",
            ("GD_CLOUD", "GD_CLOUD2", "GD_CLDFR")),
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
             "LAKE_DEPTH", "WATER_DEPTH", "UOCE", "VOCE", "CLAT",
             "U_FRAME", "V_FRAME", "GOT_VAR_SSO", "MF_VX_INV", "CPLMASK",
             "SAVE_TOPO_FROM_REAL", "SST_INPUT", "AREA2D", "DX2D",
             "CTOPO", "CTOPO2", "SHDAVG")),
    # Stock CLWRF output names are the caller's Registry names, not its
    # CL-suffixed formals. Its INTEGER NSTEPS is a scalar in native WRF.
    **_rows(_READ, "stock CLWRF window state, restored when output_diagnostics "
            "uses auxhist3_interval", (*[row[0] for row in CLWRF_HISTORY.values()], "NSTEPS")),
    **_rows(_READ, "native RASM accumulator and counter state",
        tuple(RASM_SOURCES.values())),
    **_rows(_READ, "native lightning rates, counts, activation and monthly inputs",
        ("IC_FLASHRATE", "CG_FLASHRATE", "IC_FLASHCOUNT", "CG_FLASHCOUNT",
         "LPI", "LTNGACTTIME", "ICCG_IN_NUM", "ICCG_IN_DEN")),
    # -- diagnostics and features this run does not run -------------------
    **_rows(_READ, 'native WRF 4.7.1 precipitation interval accumulators',
            ('PREC_ACC_C','PREC_ACC_NC','SNOW_ACC_NC')),
    **_rows(_READ, "native AFWA severe cross-step carriers", ("AFWA_HAIL", "AFWA_LLWS", "AFWA_TORNADO", "TCOLI_MAX", "GRPL_FLX_MAX", "MIDRH_MIN_OLD", "TORNADO_MASK", "TORNADO_DUR")),
    **_rows(_READ, "native AFWA diagnostics continued from Registry r/rh02 carriers",
            ("AFWA_PRECIP", "AFWA_TOTPRECIP", "AFWA_PWAT", "AFWA_MSLP",
             "AFWA_HEATIDX", "AFWA_WCHILL", "AFWA_FITS", "AFWA_RAIN", "AFWA_SNOW",
             "AFWA_ICE", "AFWA_FZRA", "AFWA_SNOWFALL", "VIL", "RADARVIL",
             "REFD", "REFD_COM", "ECHOTOP", "AFWA_VIS", "AFWA_VIS_DUST", "AFWA_VIS_ALPHA", "AFWA_CAPE", "AFWA_CIN", "AFWA_ZLFC", "AFWA_PLFC", "AFWA_LIDX", "AFWA_CAPE_MU", "AFWA_CIN_MU",
             "FZLEV", "ICINGTOP", "ICINGBOT", "QICING_LG", "QICING_SM",
             "QICING_LG_MAX", "QICING_SM_MAX", "ICING_LG", "ICING_SM")),
    **_rows(_READ, "native NWP history windows, continued from the restart "
            "(WRF 4.7.1 Registry.EM_COMMON:2081-2093, rh/rh02)",
            ("WSPD10MAX", "W_UP_MAX", "W_DN_MAX", "UP_HELI_MAX", "W_MEAN",
             "GRPL_MAX", "REFD_MAX", "HAIL_MAXK1", "HAIL_MAX2D")),
    **_rows("rebuilt", "updraft helicity diagnostic", ("UH",)),
    **_rows("read", "observation FDDA surface ratios restored by the physics overlay",
            ("URATX", "VRATX", "TRATX")),
    **_rows("inactive", "spectral nudging restart history, not implemented",
            ("SN_NDG_NEW", "SN_NDG_OLD", "STEPFG"), _cfg("grid_fdda")),
    **_rows("inactive", "stochastic perturbation patterns and seeds, not "
            "carried", ("SPFORCC5", "SPFORCS5", "SP_AMP5", "ISEEDARR_SPPT",
                        "ISEEDARR_SKEBS", "ISEEDARR_RAND_PERTURB",
                        "ISEEDARRAY_SPP_CONV", "ISEEDARRAY_SPP_PBL",
                        "ISEEDARRAY_SPP_LSM", "DID_STOCH", "PERT3D_STD",
                        "PERT3D_CUT", "PERT3D_LEN", "PERT3D_TIM",
                        "PERT3D_IVS", "PERT3D_ISD")),
    **_rows("inactive", "lightning diagnostics, not selected",
            ("LNINT", "TSTAR")),
    **_rows("inactive", "native WRF averaged-flux restart needs its history "
            "alarm state, which this importer does not restore; use an "
            "WOOF checkpoint to preserve the mean, count and pending reset",
            ("AVGFLX_COUNT", "AVGFLX_RUM", "AVGFLX_RVM", "AVGFLX_WWM"),
            _cfg("do_avgflx_em", 0)),
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
             "MIN_PTCHSZ", "WE_JUST_READ_SST",
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
    # WRF's NOCOLONS changes the clock separators in restart filenames too.
    # Both spellings carry the same clock, read and checked from Times below.
    patterns = {f"{RESTART_PREFIX}??_{restart_time:{WRF_TIME_FORMAT}}"}
    patterns.update(pattern.replace(":", "_") for pattern in tuple(patterns))
    for path in sorted({path for pattern in patterns
                        for path in directory.glob(pattern)}):
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


def _ssib_restart_names():
    from woof.io.wrf_output_schema import SSIB_OUTPUT_FIELDS
    return {key: row.netcdf_name for key, row in SSIB_OUTPUT_FIELDS.items()} | {
        "isnow": "ISNOW", "tsk_save": "TSK_SAVE", "xice_save": "XICE_SAVE"}


SSIB_REBUILT_DIAGNOSTICS = frozenset({
    "ssib_lhf", "ssib_shf", "ssib_ghf", "ssib_egs", "ssib_eci",
    "ssib_ect", "ssib_egi", "ssib_egt", "ssib_sdn", "ssib_sup",
    "ssib_ldn", "ssib_lup", "ssib_wat", "ssib_shc", "ssib_shg",
    "ssib_lai", "ssib_vcf", "ssib_z00", "ssib_veg"})


def _disposition(name: str, cfg=None) -> Disposition | None:
    if int(getattr(cfg, "sf_surface_physics", 0)) == 8:
        if name in _ssib_restart_names().values() or name in (
                "SWVISDIR", "SWVISDIF", "SWNIRDIR", "SWNIRDIF",
                "ALSWVISDIR", "ALSWVISDIF", "ALSWNIRDIR", "ALSWNIRDIF",
                "SFCEVP"):
            return Disposition("read", "SSiB carried state and spectral radiation; "
                               "Registry/registry.ssib and module_surface_driver.F")
        if name == "SHDAVG":
            return Disposition("setup", "monthly greenness setup for Noah-MP; "
                               "the SSiB driver does not read SHDAVG")
    if cfg is not None and int(cfg.cu_physics) == 11:
        from woof.io.wrf_output_schema import MSKF_OUTPUT_FIELDS
        if name in {row.netcdf_name for key,row in MSKF_OUTPUT_FIELDS.items()
                    if key != "cu_act"}:
            return Disposition(_READ, "retained MSKF module state and cloud feedback")
    if cfg is not None and int(cfg.mp_physics) == 40:
        from woof.state_serialization_contract import MORRISON_AEROSOL_RESTART_FIELDS
        if name in {key.upper() for key in MORRISON_AEROSOL_RESTART_FIELDS}:
            return Disposition(_READ, "Morrison aerosol activation state")
        if int(getattr(cfg, "aercu_opt", 0)) > 0 and name in ("AEROCU", "AEROMCU", "AEROPCU"):
            return Disposition(_READ, "active native monthly aerosol state")
    if cfg is not None and name in ("HTOP", "HBOT") and int(cfg.cu_physics) != 2:
        return Disposition("setup", "BMJ cloud bounds are inactive; other cumulus schemes restore CUTOP/CUBOT")
    return WRFRST_DISPOSITIONS.get(name)


def read_wrfrst(path, *, expected_dimensions: Mapping[str, int], cfg,
                nudging_controls=None):
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

    # These Registry restart carriers are read only by PX. Keep the other
    # schemes' existing disposition contract unchanged.
    from woof.core.pxlsm_runtime import PX_FRACTIONS
    px_names = {n.upper() for n in (*PX_FRACTIONS, "ra", "rs", "capg", "vegf_px",
        "t2obs", "q2obs", "imperv", "canfra", "lai_px", "wwlt_px", "wfc_px",
        "wsat_px", "clay_px", "csand_px", "fmsand_px", "sfcevp",
        "t2_ndg_old", "t2_ndg_new", "q2_ndg_old", "q2_ndg_new",
        "sn_ndg_old", "sn_ndg_new", "tsk_save")}
    path = Path(path)
    clock = read_wrfrst_clock(path)
    extents = dict(_explicit_wrfinput_dimensions(expected_dimensions))
    _, allowed_moisture = active_moisture_inventory(cfg)
    from woof.ingest.wrfinput import ALL_MOISTURE_WRFINPUT
    raw: dict[str, np.ndarray] = {}
    dispositions: dict[str, str] = {}
    refused: list[str] = []
    unclassified_nonzero: list[str] = []
    surface = nudging_controls is not None and nudging_controls.grid_sfdda in (1,2)
    surface_read = frozenset(("RUNDGDTEN", "RVNDGDTEN", "RTHNDGDTEN", "RQVNDGDTEN", "HFX_FDDA", "STEPFG", *( ("SDA_HFX","SDA_QFX","QNORM","HFX_BOTH","QFX_BOTH") if nudging_controls is not None and nudging_controls.grid_sfdda==2 else () )))
    surface_rebuilt = frozenset(f"{field}_NDG_{end}" for end in ("OLD", "NEW")
        for field in ("U", "V", "T", "Q", "PH", "MU", "U10", "V10", "T2", "TH2", "Q2", "RH", "PSL", "PS", "TOB", "ODIS", "SN"))
    def disposition(name):
        if (int(cfg.sf_surface_physics) == 7 and int(cfg.bl_pbl_physics) == 7
                and name == "AKPBL"):
            return Disposition("rebuilt", "Real-valued PBL-top diagnostic copy; the native integer KPBL is restored into fields/kpbl")
        if int(cfg.sf_surface_physics) == 7 and name == "SHDAVG":
            return Disposition("inactive", "Annual mean vegetation is read by CLM, not PX; PX uses SHDMIN/SHDMAX and LANDUSEF")
        if int(cfg.sf_surface_physics) == 7 and name in px_names:
            return Disposition("read", "PX Registry restart-carried surface state")
        if surface and name in surface_read:
            return Disposition("read", "native surface/grid FDDA held tendency or cadence")
        if surface and name in surface_rebuilt:
            return Disposition("rebuilt", "native FDDA endpoints selected from the identity-checked auxinput9/10 files by the restored model clock")
        return _disposition(name, cfg)
    with netcdf_bridge.open_dataset(path) as dataset:
        dimensions = {name: len(dim) for name, dim in dataset.dimensions.items()}
        attrs = {name: dataset.getncattr(name) for name in dataset.ncattrs()}
        if int(cfg.sf_surface_physics) == 7:
            from woof.core.pxlsm import px_dataset
            ncat = int(attrs.get("NUM_LAND_CAT", 0))
            px_dataset(ncat)
            extents.update(land_cat_stag=ncat, soil_cat_stag=16)
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
                disposition(name) is None or disposition(name).kind == "read")])
        for name, variable in dataset.variables.items():
            if name == "Times":
                continue
            decode.ready(name)
            rule = disposition(name)
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
            if name == "TLAG":
                # Registry i&j writes (Time, south_north, lagday, west_east).
                # The CUDA bank is (lagday, south_north, west_east).
                axes = tuple(d for d in variable.dimensions if d != "Time")
                if axes.index("south_north") == 0:
                    value = np.ascontiguousarray(np.swapaxes(value, 0, 1))
            target = PROGNOSTIC_RENAMES.get(name, name)
            raw[target] = value
        if surface:
            expected_step = nudging_controls.cadence_steps(float(cfg.dt))
            if "STEPFG" not in raw or int(raw["STEPFG"]) != expected_step:
                raise ValueError(f"surface FDDA restart STEPFG differs from the requested {expected_step}-step cadence")
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
    if int(getattr(cfg, "use_theta_m", 0)) and "THM_1" not in raw:
        raise ValueError(
            f"{path} lacks THM_1: use_theta_m=1 needs the previous thermal "
            "time level to convert physics tendencies before small_step_prep; "
            "replacing it with THM_2 changes the resumed forecast")
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
    raw["T"] = (np.asarray(file_dry, np.float32)
                if use_theta_m and int(getattr(cfg, "use_theta_m", 0)) and file_dry is not None
                else dry)
    raw["THM"] = theta
    from woof.ingest.wrfinput import missing_required_wrfinput
    missing = missing_required_wrfinput(raw, cfg)
    if missing:
        raise ValueError(
            f"{path} lacks state the warm start restores: {missing} (by the "
            "wrfinput names; a restart carries U_2/V_2/W_2/PH_2/MU_2/THM_2 "
            "for U/V/W/PH/MU/T).  A WRF restart written by wrf.exe carries "
            "every one of them, so this file was cut down by another tool")
    checks: dict[str, object] = dict(cadence_checks)
    # Native completed forecasts carry signed transport residues. There is
    # no universal relative cutoff that can diagnose a corrupt restart.
    # Strict continuation, CAM5 moments and ordinary tracer/trajectory
    # transport preserve their native words. Other fast-state species use
    # their nonnegative representation; WSM6 condensate uses its entry MAX.
    # Numeric reading still checks finiteness and state health checks bounds.
    strict = bool(wrf_exact.ENABLED)
    transport = (int(getattr(cfg, "tracer_opt", 0)) == 2
                 or int(getattr(cfg, "traj_opt", 0)) == 1)
    cam = int(cfg.mp_physics) == 11
    clamped, preserved, signed, cam_signed, native_clamped, policy = ({}, {}, {}, {}, {}, {})
    for name in sorted(set(raw) & allowed_moisture):
        value = np.asarray(raw[name], np.float32)
        negative = value < 0
        if not negative.any():
            continue
        scale = float(np.max(np.abs(value)))
        worst = float(value.min())
        keep = strict or transport or cam
        row = {"cells": int(negative.sum()), "most_negative": worst,
               "field_scale": scale, "relative_magnitude": -worst / scale,
               "negative_sum": float(value[negative].sum(dtype=np.float64)),
               "action": "preserved" if keep else "set_to_zero"}
        policy[name] = row
        if keep:
            if strict:
                preserved[name] = row
            if transport:
                signed[name] = row
            if cam:
                cam_signed[name] = row
            continue
        raw[name] = np.maximum(value, np.float32(0.0))
        clamped[name] = row
        if int(cfg.mp_physics) == 6 and name in {"QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP"}:
            native_clamped[name] = row
    if policy:
        checks["native_negative_moisture_policy"] = policy
    if clamped:
        # Keep the old receipt key readable; the canonical policy above
        # names the action without classifying every residue as roundoff.
        checks["roundoff_negative_moisture_set_to_zero"] = clamped
        checks["native_signed_moisture_action"] = (
            "fast numerics normalize native signed species to zero for the "
            "nonnegative fast-state contract; original counts, minima and sums are recorded")
        if int(cfg.mp_physics) == 55:
            checks["native_habit_transport_negative_policy"] = (
                "mp55 native transport residues normalized to the default "
                "nonnegative scalar initialization; all changed cells counted")
    if preserved:
        checks["native_signed_moisture_preserved"] = preserved
        checks["strict_native_negative_moisture_retained"] = preserved
        checks["strict_native_signed_moisture_retained"] = preserved
    if signed:
        checks["signed_native_moisture_preserved"] = signed
    if cam_signed:
        checks["native_cam_signed_species_preserved"] = cam_signed
    if native_clamped:
        checks["wsm6_native_entry_condensate_set_to_zero"] = native_clamped
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
                "across steps (including Kain-Fritsch's NCA timer and "
                "W0AVG, or BMJ's CLDEFI and convective clouds)")
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
    "keps_tke": "TKE_PBL", "keps_diss": "DISS_PBL",
    "keps_tpe": "TPE_PBL", "keps_pr": "PR_PBL",
    "keps_tke_adv": "TKE_ADV", "keps_diss_adv": "DISS_ADV",
    "keps_tpe_adv": "TPE_ADV",
    "sfcrunoff": "SFROFF", "udrunoff": "UDROFF",
    "xice": "SEAICE",
    "rthratenlwc": "RTHRATLWC", "rthratenswc": "RTHRATSWC",
    # module_pbl_driver.F binds UW's full-interface diffusivities to
    # grid%exch_m/grid%exch_h. Preserve the top interface on restart.
    "uw_kvm": "EXCH_M", "uw_kvh": "EXCH_H", "uw_cldfra": "CLDFRA",
    "icwmrdp": "ICWMRDP3D",
    **{"cam_held_" + name: name.upper() for name in
       ("rucuten", "rvcuten", "rqcncuten", "rqincuten")},
    "surface_raincv": "RAINCV", "surface_rainncv": "RAINNCV",
    "surface_rainshv": "RAINSHV", "surface_snowncv": "SNOWNCV",
    "surface_graupelncv": "GRAUPELNCV", "surface_hailncv": "HAILNCV",
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
    **{name: "MYNN plume diagnostic, not native restart-written; "
             "rewritten by the next PBL call before any physics consumer"
       for name in ("maxwidth", "maxmf", "ztop_plume", "ktop_plume")},
    **{name: "a surface-layer diagnostic rewritten by every surface-layer "
             "call before any read"
       for name in ("br", "cd", "cda", "ch", "chs", "chs2", "ck", "cka",
                    "cpm", "cqs2", "fh", "fm", "gz1oz0", "psih", "psim",
                    "qgh", "wspd", "zol", "u10e", "v10e", "wstar", "qstar")},
})

#: Scratch slots and their WRF sources.
from woof.core.misc_buckets import COUNTERS as BUCKET_COUNTERS
SCRATCH_SOURCES = MappingProxyType({
    **{k:k.upper() for k in ("afwa_hail","afwa_llws","afwa_tornado","tcoli_max","grpl_flx_max")},
    **{f"afwa_{k}":k.upper() for k in ("midrh_min_old","tornado_mask","tornado_dur")},
    **{name:name.upper() for name in BUCKET_COUNTERS},
    **{name: name.upper() for name in ('prec_acc_c','prec_acc_nc','snow_acc_nc')},
    **{name: name.upper() for name in (
        'afwa_precip', 'afwa_totprecip', 'afwa_pwat', 'afwa_mslp',
        'afwa_heatidx', 'afwa_wchill', 'afwa_fits', 'afwa_rain', 'afwa_snow',
        'afwa_ice', 'afwa_fzra', 'afwa_snowfall', 'afwa_vis', 'afwa_vis_dust', 'afwa_vis_alpha','afwa_cape','afwa_cin','afwa_zlfc','afwa_plfc','afwa_lidx','afwa_cape_mu','afwa_cin_mu')},
    **{f'afwa_{k}': k.upper() for k in ('vil','radarvil','refd','refd_com','echotop',
        'fzlev','icingtop','icingbot','qicing_lg','qicing_sm','qicing_lg_max','qicing_sm_max','icing_lg','icing_sm')},

    **{slot: row[0] for slot, row in CLWRF_HISTORY.items()},
    "clwrf_nsteps": "NSTEPS",
    **RASM_SOURCES,
    **{"ltng_"+name: name.upper() for name in
       ("ic_flashrate","cg_flashrate","ic_flashcount","cg_flashcount",
        "lpi","ltngacttime","iccg_in_num","iccg_in_den")},
    "ltng_lat":"XLAT", "ltng_lon":"XLONG",
    "mp_rainnc": "RAINNC", "mp_rainncv": "RAINNCV", "mp_snownc": "SNOWNC",
    "mp_snowncv": "SNOWNCV", "mp_graupelnc": "GRAUPELNC",
    "mp_graupelncv": "GRAUPELNCV", "mp_hailnc": "HAILNC",
    "mp_hailncv": "HAILNCV", "mp_sr": "SR", "mp_kessler_sr": "SR",
    "cu_rainc": "RAINC", "cu_nca": "NCA", "cu_pratec": "PRATEC",
    "cu_raincv": "RAINCV", "cu_rthcuten": "RTHCUTEN",
    "cu_rqvcuten": "RQVCUTEN", "cu_rqccuten": "RQCCUTEN",
    "cu_rqicuten": "RQICUTEN", "cu_rqrcuten": "RQRCUTEN",
    "cu_rqscuten": "RQSCUTEN",
    "cu_rucuten": "RUCUTEN", "cu_rvcuten": "RVCUTEN",
    # WRF diff_opt = 1 mixes t - t_init (module_big_step_utilities_em.F
    # horizontal_diffusion_3dmp, base_3d = t_init), and t_init is kept in
    # the restart; woof's coordinate reference is that field.
    "diff1_theta_initial": "T_INIT",
})

#: Kain-Fritsch (cu_physics = 1/10/11/99) and BMJ (cu_physics = 2) hold PRATEC
#: between calls and accumulate RAINC from it every step. KF also uses a
#: positive NCA expiry timer; BMJ has no such timer. Nonpersistent schemes
#: add each call's own rain and leave these three at their cold values.
#: WRF writes PRATEC, RAINCV and NCA for those
#: schemes too, as the last call's diagnostics; restoring them armed the
#: clock with a rate nothing ever cleared, so a cu6 or cu16 warm start
#: accumulated its restart-instant PRATEC on top of each step's own rain:
#: twice WRF's convective rain over NCAR's CONUS 12 km restart hour (0.0313
#: mm against 0.0154 mm mean), and the same excess into RAINBL.
HELD_RAIN_CU_PHYSICS = 1
HELD_RAIN_CU_SCHEMES = frozenset({HELD_RAIN_CU_PHYSICS, 7, 11, 99})
HELD_RAIN_COLD_VALUES = MappingProxyType({
    "cu_nca": -100.0, "cu_pratec": 0.0, "cu_raincv": 0.0})


def scratch_restore_plan(name: str, cu_physics: int, restart_names=(), *, nwp_diagnostics: int = 0):
    """How the warm start fills one ``scratch/`` carrier.

    ``("fill", WRF_NAME)`` copies the restart variable; ``("cold", value,
    reason)`` starts the carrier as the driver does; ``None`` names no rule.
    """
    from woof.config import KF_FAMILY_CU_SCHEMES
    if name in {"temf_workspace", "temf_rublten", "temf_rvblten",
                "temf_rthblten", "temf_rqvblten", "temf_rqcblten"}:
        return ("cold", 0.0,
                "TEMF per-call solve scratch; the next PBL call rewrites "
                "it, while held PBL tendencies restore separately")
    if int(cu_physics) == 2 and name in HELD_RAIN_COLD_VALUES:
        if name == "cu_nca":
            return ("cold", -100.0,
                    "BMJ holds PRATEC between calls without Kain-Fritsch's "
                    "positive NCA expiry timer")
        return ("fill", SCRATCH_SOURCES[name])
    # OSAS holds PRATEC between due calls, as advance_ppt does. Its NCA
    # is unused and keeps the cold sentinel; the rain rate and raw native
    # interval amount must survive a native restart with positive cudt.
    if int(cu_physics) == 95 and name in ("cu_pratec", "cu_raincv"):
        return ("fill", SCRATCH_SOURCES[name])
    if nwp_diagnostics == 1:
        from woof.core.nwp_diagnostics import HISTORY
        if name in HISTORY or name == "up_heli_max":
            return ("fill", name.upper())
    from woof.core.hail import EMBRYOS, PERSISTENT
    if name in PERSISTENT:
        source = name.upper()
        if source in restart_names:
            return ("fill", source)
        if name in EMBRYOS and any(slot.upper() in restart_names
                                   for slot in EMBRYOS):
            # A partial active package is damaged state, not an inactive
            # native diagnostic. Let the manifest walk name the omission.
            return ("fill", source)
        if name in ("hailcast_wdur", "hailcast_wup_mask"):
            return ("fill", source)
        return ("cold", 0.0,
                "native WRF has no restart value for this hail diagnostic: "
                "PSD/run maxima count from the restart instant; absent "
                "HAILCAST embryo package means native hailcast_opt=0")
    from woof.core.physics_tendency_output import NAMES as TENDENCY_NAMES
    if name in TENDENCY_NAMES:
        return ('cold',0.,'WRF Registry.EM_COMMON:2142-2157 gives tendency amounts history only, no restart flag; native restart initializes them to zero')
    if (name in HELD_RAIN_COLD_VALUES
            and int(cu_physics) not in HELD_RAIN_CU_SCHEMES):
        return ("cold", HELD_RAIN_COLD_VALUES[name],
                f"held-rain carrier of the KF family (cu 1/11); cu_physics = "
                f"{int(cu_physics)} adds each call's rain itself, so WRF's "
                f"{SCRATCH_SOURCES[name]} (a last-call diagnostic) is not "
                "a rate to keep accumulating")
    if name in SCRATCH_SOURCES:
        return ("fill", SCRATCH_SOURCES[name])
    if name in SCRATCH_REBUILT:
        return ("cold", 0.0, SCRATCH_REBUILT[name])
    from woof.core.hail import HISTORY, PERSISTENT
    if name in PERSISTENT:
        native_name = HISTORY[name][0] if name in HISTORY else name.upper()
        if native_name in restart_names:
            return ("fill", native_name)
        # Automatic ArWen hail diagnostics also run when native WRF did not
        # select HAILCAST. Its absent diagnostic windows have no prior value.
        return ("cold", 0.0, "hail diagnostic absent from native WRF restart; "
                "its diagnostic window and duration start at the restart instant")
    return None


SCRATCH_REBUILT = MappingProxyType({
    **{k:"Native diagnostic-print fields have no restart flag; zero before first call" for k in ("misc_pk1m","misc_mu_2m","misc_dpsdt","misc_dmudt","misc_zero","misc_pressure_w","misc_rho","misc_phb","misc_zero3d","misc_surface_pressure")},
    "afwa_midrh_min": "Native MIDRH_MIN is initialized at warm start",
    **{k:"AFWA severe private work is rewritten before use" for k in ("afwa_severe_rh","afwa_severe_llws","afwa_severe_mask_prev","afwa_severe_dur_prev","afwa_severe_mu_cape","afwa_severe_mu_cin")},
    'tendency_qv_before': 'Private pre-microphysics vapor, overwritten before the first tendency accumulation after restart',
    **{name: "AFWA private atmosphere/work array, rewritten before its "
             "first diagnostic read after restart" for name in (
                 'afwa_rho', 'afwa_dz', 'afwa_t1', 'afwa_z1', 'afwa_zero',
                 'afwa_temperature', 'afwa_z', 'afwa_u', 'afwa_v', 'afwa_ze',
                 'afwa_theta', 'afwa_bad_status', 'afwa_buoy_w', 'afwa_buoy_dtv', 'afwa_buoy_b', 'afwa_buoy_meta')},
    "level_pl": "pressure-level outputs are not restart fields in WRF "
                "(registry.diags h{23}); start_em calls PLD again, and the "
                "level writer initializes them before any output or step",
    "level_zl": "height-level outputs are not restart fields in WRF "
                "(registry.diags h{22}); start_em calls ZLD again, and the "
                "level writer initializes them before any output or step",

    "ltng_ktop_deep": "native non-restart KTOP_DEEP, rewritten by G3/GF before the next lightning call",
    **{name: "RASM per-call work overwritten before every read" for name in RASM_WORK},
    **{name: "solar per-call work overwritten before every read" for name in
       (*SOLAR_WORK, "solar_geometry_pair", "solar_hrang", "solar_coszen")},
    **{name: "WRF Registry solar diagnostics are history-only, without a restart flag; initialized at zero on native warm start"
       for name in SOLAR_HISTORY},
    "clwrf_zero": "zero precipitation input when the corresponding scheme is off",
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
    rules["sfcevp"] = ("cold",0.0,"Native accumulated evaporation starts at zero when absent")
    from woof.core.physics_inventory import SCM_SURFACE_FORCING_FIELDS
    for name in SCM_SURFACE_FORCING_FIELDS:
        rules[name] = ("cold", 0.0, (
            "inactive single-column surface forcing; this engine admits "
            "WRF's absent prescribed surface forcing scalar starts at zero"))
    rules["exch_tke"] = ("cold", 0.0, (
        "GBM EXCH_TKE is history-only in WRF 4.7.1 Registry.EM_COMMON:1155 (h, "
        "without r); GBMPBL rewrites it at the next PBL call. A native "
        "restart without it starts this diagnostic at zero."))
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
    from woof.core.cam3_memory import CAM_FIELDS_2D, CAM_FIELDS_3D
    for name in (*CAM_FIELDS_2D, *CAM_FIELDS_3D):
        rules[name] = ("cold", 0.0, (
            f"CAM diagnostic {name.upper()} absent from this WRF restart; "
            "zero until the next radiation call, as native initialization"))
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
    rules["xice_save"] = ("copy", "fields/xicem", (
        "surface_driver copies XICEM to its local XICE_SAVE before any read"))
    return rules


def fields_restore_plan(name: str, restart_names, cfg=None, *, bl_pbl_physics=0) -> tuple | None:
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
    if int(getattr(cfg, "sf_surface_physics", 0)) == 8:
        alias = _ssib_restart_names().get(name)
        if alias in restart_names:
            return ("fill", alias)
        if name in SSIB_REBUILT_DIAGNOSTICS:
            return ("cold", 0., "SSiB per-call output, absent from WRF's "
                    "restart by Registry/registry.ssib:9-27; rewritten by "
                    "the next land column call")
        if name == "xice_save":
            return ("copy", "fields/xice", "previous sea-ice fraction: WRF "
                    "surface_driver seeds XICE_SAVE from SEAICE on restart")
        if name == "tsk_save":
            return ("copy", "fields/tsk", "component surface temperature: WRF "
                    "physics_init seeds TSK_SAVE from TSK on restart")
        if name == "tsk_sea" and "SST" in restart_names:
            return ("fill", "SST")
        if name in {f"{n}_sea" for n in (
                "znt", "ust", "mol", "zol", "flhc", "flqc", "cpm",
                "cqs2", "chs2", "chs", "qsfc", "qgh", "hfx", "qfx", "lh")}:
            return ("rebuilt", "WRF's temporary sea component in the MM5 "
                    "sea-ice wrapper; seeded by physics_init and rewritten "
                    "by the open-water surface-layer call")
        if name == "smcrel":
            return ("cold", 0., "Noah relative-soil-moisture diagnostic, "
                    "not read or written by SSiB")
        if name in {f"{n}{suffix}" for n in ("hfx", "lh", "tsk")
                    for suffix in ("_force", "_force_tend")}:
            return ("rebuilt", "single-column-model forcing plane; unused "
                    "by the admitted real-data SSiB surface coupling")
        if name == "_ssib_status":
            return ("cold", 0, "GPU fatal-status latch starts clear; native "
                    "WRF stops immediately on the corresponding fatal error")
        if name.startswith("surface_"):
            variable = name.removeprefix("surface_").upper()
            if variable in restart_names:
                return ("fill", variable)
            if variable == "RAINSHV":
                return ("cold", 0., "SSiB has no shallow-cumulus rain when "
                        "the native restart carries no RAINSHV")
    if name == "temf_tskold":
        return ("copy", "fields/tsk",
                "TEMF saves TSK before each surface call; WRF keeps this "
                "as wrapper-local TSKOLD, not restart state")
    if name == "exch_temf":
        return ("cold", 0.0,
                "TEMF surface call rewrites wrapper-local EXCH_TEMF "
                "before the next PBL call")
    if int(bl_pbl_physics) == 99 and name == "hol":
        # Registry.EM_COMMON:1957 declares HOL as scratch, without a
        # restart stream. MRF2D:847-855 writes it before its first read.
        return ("cold", 0.0, "MRF HOL scratch, absent from native WRF "
                "restarts and rewritten by the next MRF call before use")
    if name == "rainbl":
        return ("fill", "RAINBL")
    if name == "aod5502d" and "TAOD5502D" in restart_names:
        return ("fill", "TAOD5502D")
    if name == "cldfra_cu":
        return ("rebuilt", "driver-local diagnostic, recomputed from restored convective cloud profiles before the next radiation read")
    if name == "g3_diag_clock":
        return ("rebuilt", "convtrans counters reconstructed from the native step clock after the carrier overlay")
    if name.startswith("eclipse_clear_"):
        return ("cold", 0.0, "adapter-local clear-sky cache: native adapters dim incoming solar flux; output-scaling adapters rewrite this cache before every radiation read")
    if name == "eclipse_initialized":
        return ("cold", 1.0, "native restart already carries the last radiation call's fluxes, heating and eclipse diagnostics; retain the native radiation cadence")
    # Noah-MP's native output names omit the internal xy suffix. Reusing
    # the canonical schema keeps warm-start bytes aligned with history,
    # including accumulated fields whose Registry names contain spaces.
    if int(getattr(cfg, "sf_surface_physics", 0)) == 4:
        from woof.io.wrf_output_schema import NOAHMP_OUTPUT_FIELDS
        output = NOAHMP_OUTPUT_FIELDS.get(name)
        if output is not None and output.netcdf_name in restart_names:
            return ("fill", output.netcdf_name)
    wrf_name = FIELD_ALIASES.get(name, name.upper())
    if wrf_name in restart_names:
        return ("fill", wrf_name)
    if name == "smcrel" and int(getattr(cfg, "sf_surface_physics", 0)) == 1:
        return ("cold", 0.0, "SMCREL is an unused Noah diagnostic under Slab; "
                "native Slab does not allocate it, and the shared allocator "
                "starts its inactive carrier at zero")

    if name in ("cam_t_phy", "cam_p_hyd", "cam_p8w", "cam_kvh"):
        return ("rebuilt", "CAM geometry and diffusivity copied after PBL before microphysics")
    if name == "dlf":
        return ("cold", 0.0, "Native registry.cam:13 omits DLF from restart; allocated zero before resumed convection")
    if name in ("rushten", "rvshten", "rthshten", "rqvshten", "rqcshten",
                "rqishten", "rqcnshten", "rqinshten"):
        return ("rebuilt", "Shallow rates rewritten every step before composition")
    row = CAM_OUTPUT_ROWS.get(name)
    if row is not None and "r" not in row[7]:
        return ("cold", 0.0, "Native Registry omits this CAM diagnostic from restart; allocated zero until its next producer")
    if (name == "surface_rainshv" and "RAINSHV" not in restart_names
            and cfg is not None and not int(getattr(cfg, "shcu_physics", 0))):
        return ("cold", 0.0, "no shallow-cumulus scheme produces RAINSHV")
    if cfg is not None and int(cfg.sf_surface_physics) == 3:
        from woof.core.ruc_runtime import (
            RUC_MEASURED_INERT_CARRIERS, RUC_DIAGNOSTICS_2D)
        if name in (*RUC_MEASURED_INERT_CARRIERS, *RUC_DIAGNOSTICS_2D):
            return ("rebuilt", "RUC recomputes this per-call diagnostic before "
                    "reading it; native restart has no such output")
        if name == "smcrel":
            return ("cold", 0.0, "Noah relative soil moisture has no RUC producer or consumer")
        if name == "tsk_save":
            return ("copy", "fields/tsk", "WRF phy_init restores absent TSK_SAVE from TSK")
        if name == "tsk_sea":
            return ("copy", "fields/sst", "the open-water component starts from restored SST")
        if (name.endswith("_sea") and name[:-4] in (
                "chs2", "chs", "cpm", "cqs2", "flhc", "flqc", "hfx", "lh",
                "mol", "qfx", "qgh", "qsfc", "ust", "znt", "zol")
                and int(cfg.sf_sfclay_physics) in (1, 91)):
            return ("copy", "fields/" + name[:-4],
                    "WRF module_surface_driver.F:6150-6172,6585-6603: "
                    "MM5 open-water hold buffers start from incoming surface fields")
        from woof.core.physics_inventory import MYNN_SURFACE_OUTPUTS
        if (name.endswith("_sea") and name[:-4] in MYNN_SURFACE_OUTPUTS
                and int(cfg.sf_sfclay_physics) == 5):
            return ("rebuilt", "MYNN_SEAICE_WRAPPER copies each incoming surface "
                    "field into this hold buffer before either surface call")
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


from woof.state_serialization_contract import MORRISON_AEROSOL_RESTART_FIELDS
STATE_SOURCES = MappingProxyType({
    **{name:name.upper() for name in (*MORRISON_AEROSOL_RESTART_FIELDS,"aerocu")},
    "rthften": "RTHFTEN", "rqvften": "RQVFTEN", "tke": "TKE_2",
    "thp0": "THM_1",
    "p_perturbation": "P",
    "gamu": "GAMU", "gamv": "GAMV",
    "th_old": "TH_OLD", "qv_old": "QV_OLD",
    "f_ice": "F_ICE_PHY", "f_rain": "F_RAIN_PHY",
    "f_rimef": "F_RIMEF_PHY",

    "keps_tke_adv": "TKE_ADV", "keps_diss_adv": "DISS_ADV",
    "keps_tpe_adv": "TPE_ADV",
    **{f"re_{species}_gsfc": f"RE_{species.upper()}_GSFC"
       for species in ("cloud", "ice", "rain", "snow", "graupel", "hail")},

    **{name: name.upper() for name in (
        "cldfra_old_mp", "cldfra_mp", "cldfra_mp_all", "lradius",
        "iradius", "cldfrai", "cldfral", "cldfra_conv", "wsedl3d",
        "rh_old_mp", "lcd_old_mp")},

    "_wrf_th_phy_m_t0": "T",
    "surface_fdda_u": "RUNDGDTEN", "surface_fdda_v": "RVNDGDTEN",
    "surface_fdda_theta": "RTHNDGDTEN", "surface_fdda_qv": "RQVNDGDTEN",
    "hfx_fdda": "HFX_FDDA",
    "sda_hfx":"SDA_HFX", "sda_qfx":"SDA_QFX", "qnorm":"QNORM",
    "hfx_both":"HFX_BOTH", "qfx_both":"QFX_BOTH",
})


def _rebuilt_radius_reason(name, raw, cfg):
    """Morrison output diagnostics absent with native RRTM/Dudhia."""
    sources = {"effc": "RE_CLOUD", "effi": "RE_ICE", "effs": "RE_SNOW",
               "effr": "RE_RAIN"}
    if (name in sources and sources[name] not in raw
            and int(cfg.mp_physics) == 10
            and int(cfg.ra_lw_physics) == 1 and int(cfg.ra_sw_physics) == 1):
        return ("native RRTM/Dudhia does not allocate or consume microphysics "
                "effective radii (module_physics_init.F, has_reqc/has_reqi/"
                "has_reqs); Morrison rewrites these output diagnostics every "
                "call (kernels/morrison.cu), so retain the cold diagnostic "
                "until that first call")
    return None


def _restored_state_reasons(cfg) -> dict[str, str]:
    """``state/`` carriers the shared restore already filled, and from what."""
    from woof.ingest.wrfinput import active_moisture_map

    reasons = {
        "u": "U_2", "v": "V_2", "w": "W_2", "php": "PH_2", "mup": "MU_2",
        "thp": "THM_2 (selected thermal variable about the file's base state)",
        "h_diabatic": "H_DIABATIC",
        "p": "EOS of the restored prognostics (WRF's P compared)",
        "al": "EOS of the restored prognostics (WRF's AL compared)",
        "alt": "EOS of the restored prognostics (WRF's ALT compared)",
    }
    for wrf_name, state_name in active_moisture_map(cfg).items():
        reasons[state_name] = wrf_name
    if int(getattr(cfg, "use_theta_m", 0)):
        reasons.update(p="P + PB retained from native restart",
                       al="AL retained from native restart",
                       alt="ALT retained from native restart")

    if int(cfg.mp_physics)==28:
        reasons.update(nwfa2d="QNWFA2D",nifa2d="QNIFA2D")
    from woof.core.passive_tracer_contract import active_tracers
    for name in active_tracers(cfg):
        reasons[name] = name
    from woof.core.trajectory_contract import enabled as trajectories_enabled
    if trajectories_enabled(cfg):
        reasons["traj_coordinates"] = "TRAJ_I/J/K/LONG/LAT"
        reasons["traj_status"] = "immutable native point roster and new output buffer"
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
    if name in ("ICCG_IN_NUM", "ICCG_IN_DEN") and value.ndim == 2 and len(target_shape) == 3:
        # The admitted geographic/prescribed methods do not read this
        # monthly workspace; match initialize_lightning's device layout.
        return np.broadcast_to(value, target_shape)
    if name == "LTNGACTTIME" and value.size == 1:
        return value.reshape(target_shape)
    if (name == "NSTEPS" and label == "scratch/clwrf_nsteps" or name in ("NSTEP_M","NSTEP_DTM","NSTEP_D") and label.startswith("scratch/rasm_")) and value.size == 1:
        return np.broadcast_to(value.reshape(()), target_shape)
    if value.shape == tuple(target_shape):
        return value
    if name in ("NDAY", "NYEAR") and value.size == 1 and tuple(target_shape) == (1,):
        return value.reshape(1)

    if name in ("ELAT_TRACK", "ELON_TRACK") and value.size == 1:
        # Registry declares one Time scalar for the whole domain. Uniform
        # device planes let the generic nest/ensemble carrier path transport
        # it; history retains the original scalar dimension.
        return np.full(target_shape, value.reshape(-1)[0], dtype=value.dtype)
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


def unused_milbrandt_restart_radius(cfg, source, raw):
    """Native MP9/legacy RRTMG never reads its compatibility radius arrays.

    WRF Registry.EM_COMMON:3025 has no RE_CLOUD/ICE/SNOW in milbrandt2mom;
    module_physics_init's use_mp_re table excludes it. Other schemes and
    radiation adapters keep their existing required-source policy.
    """
    if int(cfg.mp_physics) != 9 or int(cfg.ra_physics) != 4 or source in raw:
        return False
    from woof.core.rrtmg_legacy import legacy_scheme_has_req
    return not any(legacy_scheme_has_req(9, int(cfg.use_mp_re)))


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
    if getattr(cfg, "obs_nudge_opt", 0):
        from woof.fddaobs.forecast import restore_native
        restored_state.update(restore_native(state, cfg, restored))
    # Carriers a run creates lazily at its first step, which a restart
    # instant has long passed: created here so the manifest walk below
    # fills them, exactly as a checkpoint resume finds them in its file.
    if int(getattr(cfg, "diff_opt", 0)) == 1:
        # woof.core.dycore.initialize_coordinate_reference refuses to
        # rebuild this past t = 0, and rightly: it is WRF's t_init.
        state.scratch(state.thp.shape, "diff1_theta_initial")
    if cfg.mean_diag:
        import cupy as cp
        state._wrf_th_phy_m_t0 = cp.zeros_like(state.thp)
    adapter = None if driver is None else getattr(driver, "cumulus_callable", None)
    if adapter is not None and ("W0AVG" in raw or int(cfg.cu_physics) == 11):
        ensure = getattr(adapter, "ensure_trigger_history", None)
        if ensure is not None:
            # Allocate all MSKF INOUT carriers before the manifest walk.
            # Binding W0AVG alone would suppress the adapter's cold allocation
            # on its first call and leave the retained clouds absent.
            ensure(state)
    if adapter is not None and int(cfg.cu_physics) == 99:
        # Materialise even absent fields so the manifest walk refuses an
        # incomplete native restart rather than silently cold-starting them.
        adapter.ensure_trigger_history(state)
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
    # Scheme-owned restart state may have a different native WRF policy
    # from woof checkpoints. CAM's Registry omits its absorption cache;
    # camrad detects the zero cache and refreshes it on the next LW call.
    native_rebuilt = {}
    if driver is not None:
        from woof.core.radiation_composition import radiation_adapters
        for index, component in enumerate(radiation_adapters(
                getattr(driver, "radiation_callable", None))):
            hook = getattr(component, "wrf_warm_start_rebuild", None)
            if callable(hook):
                for name, reason in hook().items():
                    native_rebuilt[f"radiation/owned/{index}/{name}"] = reason
    manifest = {}
    manifest.update(restart_io.state_manifest(state))
    manifest.update(restart_io._scratch_manifest(state))
    driver_manifest = ({} if driver is None
                       else restart_io._driver_manifest(driver))
    manifest.update(driver_manifest)
    # ArWen can add hail diagnostics automatically to a native run that
    # never selected them. These output-only accumulators then start a
    # new window at handoff, like CLOUDNC above. Do not erase an active
    # native HAILCAST trajectory: its embryos or activity require a
    # separate, generation-aware restoration rule.
    from woof.core.hail import PERSISTENT as hail_diagnostics
    native_hail_inactive = (
        not any(f"HAILCAST_DHAIL{i}" in raw for i in range(1, 6))
        and not any(np.any(np.asarray(raw[name]) != 0)
                    for name in ("HAILCAST_WUP_MASK", "HAILCAST_WDUR",
                                 "HAILDTACTTIME") if name in raw))
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
            elif name == "p_perturbation":
                fill(key, target, "P")
            elif name in STATE_SOURCES:
                fill(key, target, STATE_SOURCES[name])
            elif int(cfg.mp_physics) == 11 and name == "qndrop":
                # Registry.EM_COMMON:3031 does not enable the qndrop
                # scalar for CAMMGMP, so native restart carries no QNDROP.
                # Its radiation uses LRADIUS/IRADIUS instead. This
                # per-call diagnostic is rewritten by the first MP call.
                target[...] = 0
                report.rebuilt[key] = "CAM qndrop is absent from native restart and rewritten by microphysics; radiation reads LRADIUS/IRADIUS"
            elif int(cfg.mp_physics) == 11 and name in ("effc", "effi", "effs"):
                if name == "effs":
                    target[...] = 0
                    report.rebuilt[key] = "CAM radiation reads LRADIUS/IRADIUS; no snow radius"
                else:
                    fill(key, target, {"effc": "LRADIUS", "effi": "IRADIUS"}[name])
            elif reason := _rebuilt_radius_reason(name, raw, cfg):
                report.rebuilt[key] = reason
            elif name in ("effc", "effi", "effs"):
                source = {"effc": "RE_CLOUD", "effi": "RE_ICE", "effs": "RE_SNOW"}[name]
                if unused_milbrandt_restart_radius(cfg, source, raw):
                    report.rebuilt[key] = "Native MP9/legacy RRTMG does not read its compatibility radius arrays"
                else:
                    fill(key, target, source, transform=micron)
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
            if name == "ltng_calendar":
                report.rebuilt[key] = "simulation-start calendar from the native restart epoch; elapsed clock is restored separately"
                continue
            if name in ('prec_acc_epoch','bucket_epoch','tendency_epoch'):
                # Native curr_secs2 restarts at the namelist start instant;
                # woof's model clock continues the simulation's time.
                seconds = float(restored.warm_start.clock.elapsed_seconds) - float(
                    getattr(state, 'domain_start_offset', 0.) or 0.)
                high = np.float32(seconds)
                _copy_into(target, np.asarray(
                    (high, np.float32(seconds-float(high))), np.float32))
                report.rebuilt[key] = 'native run start clock, retained across woof checkpoints'
                continue
            if name == 'graup_acc_nc':
                target[...] = 0.
                report.rebuilt[key] = 'fork graupel interval has no native WRF 4.7.1 restart carrier'
                continue
            plan = scratch_restore_plan(name, int(cfg.cu_physics), raw,
                                        nwp_diagnostics=int(getattr(cfg, "nwp_diagnostics", 0)))
            if plan is None and int(cfg.sf_surface_physics) == 8:
                from woof.core.hail import PERSISTENT, HISTORY
                if name in PERSISTENT:
                    source = HISTORY.get(name, (name.upper(),))[0]
                    if source in raw:
                        plan = ("fill", source)
                    else:
                        plan = ("cold", 0., "woof hail diagnostic absent "
                                "from the native restart; starts counting "
                                "at this warm-start instant")
            if plan is None and name in hail_diagnostics and native_hail_inactive:
                if name.upper() in raw:
                    plan = ("fill", name.upper())
                else:
                    plan = ("cold", 0.0,
                            "additional WOOF hail output diagnostic; native "
                            "HAILCAST was inactive, so its diagnostic window "
                            "counts from the restart instant")
            if plan is None:
                unmapped.append(key)
            elif plan[0] == "fill":
                fill(key, target, plan[1])
            else:
                target[...] = plan[1]
                report.rebuilt[key] = plan[2]
        elif namespace == "fields":
            if int(cfg.sf_surface_physics) == 7 and name == "smcrel":
                # This common allocation is a Noah output, absent from the
                # native PX restart and never read by PX's two-layer solver.
                target[...] = 0
                report.rebuilt[key] = (
                    "Noah relative-soil-moisture diagnostic; PX neither "
                    "produces nor reads SMCREL")
                continue
            if name in ("hfx_force", "lh_force", "tsk_force", "hfx_force_tend",
                        "lh_force_tend", "tsk_force_tend") and int(getattr(cfg, "isfflx", 1)) != 2 and int(cfg.sf_sfclay_physics) != 89:
                target[...] = 0
                report.rebuilt[key] = "prescribed SCM surface fluxes are off (isfflx != 2)"
                continue
            plan = fields_restore_plan(
                name, raw, cfg, bl_pbl_physics=cfg.bl_pbl_physics)
            if plan is None:
                wrf_name = FIELD_ALIASES.get(name, name.upper())
                unmapped.append(f"{key} <- {wrf_name} (absent from the restart)")
            elif plan[0] == "fill":
                if int(cfg.bl_pbl_physics) == 99 and name == "kpbl":
                    # Native MRF alone exports a top-down level. Keep
                    # the inactive zero sentinel, as the kernel does.
                    fill(key, target, plan[1], transform=lambda value:
                         np.where(value > 0, cfg.nz + 1 - value, value))
                    if key in report.filled:
                        report.filled[key] += " (MRF top-down to bottom-up)"
                else:
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
        elif key == "diag/_wrf_rthraten":
            # WRF leaves RTHRATEN mass-coupled after first_rk_step_part2.
            # The next physics call uncouples this exact carried total.
            fill(key, target, "RTHRATEN")
            driver._wrf_rthraten_coupled = True
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
        elif int(cfg.cu_physics) == 95 and key in (
                "cumulus/last_htop", "cumulus/last_hbot"):
            fill(key, target, "CUTOP" if key.endswith("last_htop") else "CUBOT")
        elif key == "diag/radiation_flux_diagnostics":
            for index, source in enumerate(driver.radiation_callable.diagnostics):
                label = f"{key}/{source}"
                if source in raw:
                    fill(label, target[index], source)
                else:
                    # WRF's Goddard TL*/SL*/TS*/SS* rows are history-only
                    # (Registry h, no r). A native restart zero-initializes
                    # them until the next radiation call as well.
                    target[index][...] = 0
                    report.rebuilt[label] = (
                        f"{source} absent from native restart; held output "
                        "starts at zero until the next radiation call")

        elif key == "diag/aod5503d":
            fill(key, target, "TAOD5503D" if int(cfg.aer_opt)==3 else "AOD5503D")
        elif key == "cumulus/w0avg":
            # Kain-Fritsch's running vertical-velocity average (WRF W0AVG,
            # restart-carried for exactly this reason).
            fill(key, target, "W0AVG")
        elif key in native_rebuilt:
            report.rebuilt[key] = native_rebuilt[key]
        elif namespace == "cumulus" and int(cfg.cu_physics) == 11:
            from woof.io.wrf_output_schema import MSKF_OUTPUT_FIELDS
            from woof.core.mskf import AEROSOL_OUTPUTS, EDRATE_OUTPUTS
            schema = MSKF_OUTPUT_FIELDS.get(name)
            if name == "cu_act":
                # MSKF_CPS writes this local diagnostic on every call.
                # Native WRF has no Registry/restart carrier for it.
                target[...] = 0
                report.rebuilt[key] = "native MSKF local CU_ACT_FLAG; rewritten at next call"
            elif (schema is not None and schema.netcdf_name not in raw
                  and ((name in AEROSOL_OUTPUTS and int(getattr(cfg, "aercu_opt", 0)) == 0)
                       or (name in EDRATE_OUTPUTS and int(getattr(cfg, "kf_edrates", 0)) == 0))):
                # Native WRF does not allocate the inactive package's arrays.
                # They carry no module state and no native restart record.
                target[...] = 0
                report.rebuilt[key] = "inactive native MSKF package; zero history extension"
            elif schema is not None:
                fill(key, target, schema.netcdf_name)
            else:
                unmapped.append(key)

        elif (int(cfg.cu_physics) == 99 and namespace == "cumulus"
              and name in {"udr_kf", "ddr_kf", "uer_kf", "der_kf", "timec_kf"}):
            fill(key, target, name.upper())
        elif key == "radiation/o33d_grid":
            fill(key, target, "O3RAD")
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
    if cfg.lightning_option:
        from woof.core.lightning import invalidate_host_clock
        invalidate_host_clock(state)
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
        from woof.core.passive_tracer_contract import mixed_tracers
        tracer_rates = getattr(driver.pbl_tendencies, "extra_scalars", None) or {}
        driver.pbl_tendencies = couple_ysu_tendencies(
            state, cfg, driver.pbl_raw_rates)
        if mixed_tracers(cfg):
            # Native init_zero_tendency clears tracer_tend on the first
            # resumed step; no tracer R*BLTEN is stored in wrfrst.
            driver.pbl_tendencies.extra_scalars = {
                name: tracer_rates[name] for name in mixed_tracers(cfg)}
        report.filled["driver/pbl_tendencies/*"] = (
            "R*BLTEN coupled with the restored dry mass")


def _last_radiation_step(itimestep: int, stepra: int,
                         ra_call_offset: int = 0) -> int:
    """The step of WRF's last radiation call at or before ``itimestep``.

    module_radiation_driver.F calls radiation on step 1 and whenever
    ``MOD(itimestep, STEPRA) == 1 + ra_call_offset``.
    """
    if stepra <= 1:
        return itimestep
    last = itimestep - ((itimestep - 1 - ra_call_offset) % stepra)
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
    if "g3_diag_clock" in driver.fields:
        count=(itimestep+_physics_interval_steps(float(cfg.cudt_minutes),dt)-1)//_physics_interval_steps(float(cfg.cudt_minutes),dt)
        window=int(np.float32(cfg.convtrans_avglen_m)*np.float32(60)/np.float32(dt))
        _copy_into(driver.fields["g3_diag_clock"], np.asarray(
            (count,count%window,int(count<=window),int(count%window==0)),np.int32))
        report.rebuilt["fields/g3_diag_clock"]="native ITIMESTEP, cumulus cadence and convtrans averaging period"
    written = {key: int(np.asarray(raw[key]).reshape(-1)[0])
               for key in CADENCE_KEYS if key in raw}
    report.checks.update(compare_restart_cadences(
        written, cfg, radt_minutes=float(driver.radt_minutes),
        radiation_active=bool(driver.radiation_active)))
    last = _last_radiation_step(itimestep, stepra, cfg.ra_call_offset)
    produced_at = (last - 1) * dt
    if "radiation/o33d_grid" in report.filled and driver.o3rad is not None:
        # A warm root may FORCE its child before the next due radiation
        # call. Its retained native O3RAD is already a completed producer.
        driver.call_counts["cam_ozone"] = max(1, driver.call_counts.get("cam_ozone", 0))
        driver.carriers.declare("o3rad", source="cam_ozone", model_time=produced_at)
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
    "HELD_RAIN_COLD_VALUES", "HELD_RAIN_CU_PHYSICS", "HELD_RAIN_CU_SCHEMES",
    "OverlayReport", "PROGNOSTIC_RENAMES", "SCRATCH_SOURCES",
    "STATEFUL_SELECTORS", "WRFRST_DISPOSITIONS", "WarmStart",
    "WrfRestartClock", "apply_warm_start_clock", "apply_warm_start_physics",
    "check_restart_physics", "discover_restart_files",
    "read_wrfrst", "read_wrfrst_clock", "restart_file_name",
]
