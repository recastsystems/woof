"""Restore WRF real.exe fields and boundary tables through the Rust reader.

File geometry and physics inventory are validated before GPU allocation.
This module adapts state only; the shared forecast runner owns integration.
"""

from __future__ import annotations

from woof.config import ConfigurationRefused

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Mapping, Sequence


from woof import netcdf_bridge
from woof.wrf_exact import ENABLED as WRF_EXACT
import numpy as np
from woof.ingest.wrfinput_sfire import SFIRE_INPUT_DIMENSIONS, fire_input_extents

# Every science-field entry below has one explicit woof consumer.  The
# importer is intentionally closed-world: variables outside these inventories
# are errors, rather than being hidden in a catch-all auxiliary bucket.
REQUIRED_WRFINPUT = (
    "U", "V", "W", "T", "PH", "MU", "PHB", "MUB", "T_INIT",
    "P", "PB", "AL", "ALB", "QVAPOR", "QCLOUD", "QRAIN",
    "QICE", "QSNOW", "QGRAUP", "QNRAIN", "QNICE", "QNSNOW",
    "QNGRAUPEL", "HGT", "FNM", "FNP", "RDNW", "RDN", "DNW",
    "DN", "ZNU", "ZNW", "C1H", "C2H", "C1F", "C2F", "C3H",
    "C4H", "C3F", "C4F", "MAPFAC_M", "MAPFAC_U", "MAPFAC_V",
    "F", "E", "SINALPHA", "COSALPHA", "LANDMASK", "LU_INDEX",
    "ISLTYP", "TSK", "TSLB", "SMOIS", "SH2O", "TMN", "SNOW",
    "SNOWH", "VEGFRA", "SNOALB", "SHDMIN", "SHDMAX", "PSFC", "T2",
    "Q2", "TH2", "U10", "V10", "XLAND", "IVGTYP",
    "CF1", "CF2", "CF3",
)

ALIASES = {
    "XICE": ("XICE", "SEAICE"),
    "ALBBCK": ("ALBBCK", "ALBEDO"),
    "LAI": ("LAI", "LAI12M"),
    "P_TOP": ("P_TOP",),
}

MOISTURE_MAP = {
    "QVAPOR": "qv", "QCLOUD": "qc", "QRAIN": "qr",
    "QICE": "qi", "QSNOW": "qs", "QGRAUP": "qg",
    "QNRAIN": "nr", "QNICE": "ni", "QNSNOW": "ns",
    "QNGRAUPEL": "ng", "QNCLOUD": "nc",
    # mp_physics=28 (Thompson aerosol-aware).  QNCLOUD is already above --
    # Morrison declares the same Registry name -- but for mp=28 it stops
    # being an optional diagnosed field and becomes a REQUIRED prognostic
    # (see REQUIRED_QNCLOUD_MICROPHYSICS below).  QNWFA/QNIFA are new here; WRF
    # declares them as scalars with the input stream in their IO string
    # (Registry/registry.new3d_wif:87-90, ``i0rhusdf=(bdy_interp:dt)``), so
    # real.exe writes them into wrfinput exactly as it writes QNRAIN.
    "QNWFA": "nwfa", "QNIFA": "nifa",
    # mp_physics=16 (WDM6).  QNCLOUD and QNRAIN are already above and carry
    # WDM6's prognostic nc/nr unchanged; QNCCN is new here.  It maps to
    # ``nn``, WDM6's own state name for the CCN reservoir -- NOT to NSSL's
    # ``qnn``, which is the same WRF variable in a different scheme's state
    # and is why the two maps stay separate rather than merging.
    "QNCCN": "nn",
    # mp_physics=50 (P3 one-category).  QICE/QNICE/QNRAIN are already above
    # and carry P3's single ice mass and its two number moments unchanged;
    # QIR and QIB are new here.  They are the rime mass and rime volume P3
    # carries INSTEAD of a snow/graupel/hail split, so without a name for
    # them this closed-world importer cannot read a P3 wrfinput at all --
    # ``read_wrfinput`` rejects the file as carrying "unmapped WRF
    # variable(s)" before any inventory check is reached.
    "QIR": "qir", "QIB": "qib",
    # mp_physics=5 (Ferrier-Aligo).  QICE is already above and carries the
    # scheme's total ice.  CWM is the Registry's output name for the
    # scalar qt (Registry.EM_COMMON:529-530, IO ``i0rhusdf``), so real.exe
    # writes it; FER_HIRES rebuilds qt from qc+qr+qi on entry
    # (module_mp_fer_hires.F:388), so it is restored when present and not
    # required.  F_ICE_PHY/F_RAIN_PHY/F_RIMEF_PHY carry ``rhdu`` with no
    # ``i`` (:1333-1335) and are never in a wrfinput.
    "CWM": "qt",
    # mp_physics=9 (Milbrandt-Yau double-moment, seven class).  QNCLOUD,
    # QNRAIN, QNICE, QNSNOW and QNGRAUPEL are already above and carry
    # Milbrandt's five prognostic number moments unchanged; QHAIL and
    # QNHAIL are new here.  Both names ALSO appear in NSSL_MOISTURE_MAP
    # below, and they map to different state there -- NSSL's number
    # moments are ``qnh``/``qng``/... while Milbrandt's are the same
    # ``nh``/``ng``/... family Morrison and WDM6 use
    # (woof/core/state.py: the mp=9 arm allocates
    # ``qh, nc, nr, ni, ns, ng, nh``).  Same WRF Registry name, two
    # schemes, two state targets: that is exactly why the two maps stay
    # separate rather than merging.
    "QHAIL": "qh", "QNHAIL": "nh",
}

# NSSL reuses several WRF Registry names carried by Morrison, but the state
# names are scheme-native (for example QNRAIN -> qnr, not nr).  Keep this as
# a distinct map so restoration cannot choose a target by whichever attribute
# happens to exist on DomainState.
NSSL_MOISTURE_MAP = {
    "QVAPOR": "qv", "QCLOUD": "qc", "QRAIN": "qr",
    "QICE": "qi", "QSNOW": "qs", "QGRAUP": "qg", "QHAIL": "qh",
    "QNDROP": "qndrop", "QNRAIN": "qnr", "QNICE": "qni",
    "QNSNOW": "qns", "QNGRAUPEL": "qng", "QNHAIL": "qnh",
    "QNCCN": "qnn", "QVGRAUPEL": "qvolg", "QVHAIL": "qvolh",
}
from woof.core.fast_sbm_contract import BIN_SPECIES as _FSBM_BIN_SPECIES
MOISTURE_MAP.update({name:name for name in _FSBM_BIN_SPECIES})
MOISTURE_MAP.update({"EFF_RADIUS":"effc","ICE_EFF_RADIUS":"effi","TOT_EFF_RADIUS":"effs"})

ALL_MOISTURE_WRFINPUT = frozenset(MOISTURE_MAP) | frozenset(
    NSSL_MOISTURE_MAP)

# ``real.exe`` writes a physics-package-specific moisture inventory.
# The first three mass species are active when microphysics is enabled;
# WSM6 adds the three ice-category masses, while Morrison additionally owns
# four transported number moments.  Native option-18 NSSL defaults add hail,
# five two-moment number fields, predicted CCN, and graupel/hail volume.  Its
# Registry aliases overlap Morrison but map to distinct scheme-native state.
# QNCLOUD is a documented optional Morrison restart field (WRF's matched
# default diagnoses cloud number).
BASE_MOISTURE_WRFINPUT = ("QVAPOR", "QCLOUD", "QRAIN")
ICE_MASS_WRFINPUT = ("QICE", "QSNOW", "QGRAUP")
MORRISON_NUMBER_WRFINPUT = ("QNRAIN", "QNICE", "QNSNOW", "QNGRAUPEL")
MORRISON_OPTIONAL_MOISTURE_WRFINPUT = ("QNCLOUD",)
THOMPSON_NUMBER_WRFINPUT = ("QNRAIN", "QNICE")
#: mp_physics=28 adds the prognostic droplet number and the two aerosol
#: number tracers to classic Thompson's two moments.  All three are
#: transported scalars in WRF's own Registry (Registry.EM_COMMON:3036's
#: ``scalar:qni,qnr,qnc,qnwfa,qnifa,qnbca``; the Registry also writes qnbca,
#: but Thompson uses it only at wif_input_opt=2 and otherwise returns zero,
#: module_mp_thompson.F:3983-3988) and all
#: three carry the wrfinput stream in their IO strings
#: (Registry.EM_COMMON:542 for QNCLOUD, registry.new3d_wif:87-90 for
#: QNWFA/QNIFA).  Unlike Morrison's QNCLOUD -- which WRF diagnoses and woof
#: therefore treats as optional -- an mp=28 wrfinput that lacked QNCLOUD
#: would start every column at nc = 0, and WRF's terminal clamp
#: (module_mp_thompson.F:3976) would silently hold it at 2/rho rather than
#: raising.  Required, not optional.
THOMPSON_AEROSOL_NUMBER_WRFINPUT = ("QNRAIN", "QNICE", "QNCLOUD",
                                    "QNWFA", "QNIFA")
#: mp_physics=16 (WDM6).  ``Registry.EM_COMMON:3031`` declares the package
#: as ``moist:qv,qc,qr,qi,qs,qg;scalar:qnn,qnc,qnr``, so WDM6's wrfinput is
#: WSM6's six masses plus exactly three transported numbers -- the CCN
#: reservoir QNCCN (:539), the cloud droplet number QNCLOUD (:541) and the
#: rain number QNRAIN (:533).  All three are prognostic, none is diagnosed,
#: so all three are REQUIRED: a WDM6 wrfinput without QNCLOUD would start
#: every column at nc = 0 and WDM6's own slope floor would hold it there,
#: which is the mp=28 hazard in the note above, not Morrison's optional
#: field.
WDM6_NUMBER_WRFINPUT = ("QNCCN", "QNCLOUD", "QNRAIN")
#: The schemes for which QNCLOUD is a required prognostic rather than the
#: optional diagnosed Morrison field.  Consulted by
#: :func:`_restore_active_moisture`, whose "missing is tolerated" branch
#: keys on ``MORRISON_OPTIONAL_MOISTURE_WRFINPUT`` alone and would otherwise
#: extend Morrison's exemption to mp=28.  16 belongs for the same reason:
#: WDM6 PREDICTS the droplet number (Registry.EM_COMMON:3031's
#: ``scalar:qnn,qnc,qnr``), so QNCLOUD is the prognostic its double-moment
#: warm rain is built on, not Morrison's diagnosed convenience field.
#: Renamed off the mp28 spelling when 16 joined -- a set with two members
#: named for one of them is how the next scheme gets forgotten.
#: 9 belongs for the same reason 16 does: Milbrandt-Yau PREDICTS the
#: droplet number (its ``scalar:qnc,...`` package), so QNCLOUD is a
#: prognostic there, not Morrison's diagnosed convenience field.
REQUIRED_QNCLOUD_MICROPHYSICS = frozenset({9, 16, 28, 30})
#: mp_physics=50 (P3 one-category).  ``Registry.EM_COMMON:3038`` declares
#: the package as ``moist:qv,qc,qr,qi;scalar:qni,qnr,qir,qib`` -- ONE ice
#: mass, and no snow and no graupel anywhere in the package, because P3
#: predicts rime mass and rime volume in place of a snow/graupel/hail
#: split.  QICE is therefore added through this tuple rather than through
#: ``ICE_MASS_WRFINPUT``: putting 50 in the ice-mass branch would demand a
#: QSNOW and a QGRAUP that a P3 wrfinput never contains and that a P3
#: ``DomainState`` has no field to receive (woof/core/state.py:464-478).
#: All four scalars are transported prognostics, and every one carries the
#: wrfinput stream in its Registry IO string -- ``i0rhusdf=(bdy_interp:dt)``
#: on QNICE (Registry.EM_COMMON:523), QNRAIN (:533), QIR (:555) and QIB
#: (:557) -- so real.exe writes all four exactly as it writes Morrison's
#: QNRAIN.  None of them is diagnosed, so unlike Morrison there is no
#: optional member here, and a missing one does not raise inside P3, it
#: silently rewrites the analysis:
#:   * a zero QNICE beside a real QICE is clamped to ``nsmall`` = 1.e-16
#:     (module_mp_p3.F:231, :2573) and the mean-mass ice diameter built
#:     from it (:2578) then indexes the lookup table (:2582) in its
#:     largest-particle bin for the whole column;
#:   * a zero QIR/QIB sends ``calc_bulkRhoRime`` (:6784, called at :2580)
#:     down its ``bi_rim < 1.e-15`` branch, which hard-zeros both and
#:     returns rho_rime = 0 (:6812-6814), declaring every ice particle in
#:     the restored state unrimed -- P3's one distinguishing prognostic,
#:     silently set to nothing.
#: Both are finite, bounded and wrong: the mp=28 terminal-clamp hazard in
#: the note above, in P3's own numbers.
#: mp_physics=9 (Milbrandt-Yau double-moment, seven class).
#: ``Registry.EM_COMMON`` declares the package as
#: ``moist:qv,qc,qr,qi,qs,qg,qh;scalar:qnc,qnr,qni,qns,qng,qnh`` -- WSM6's
#: six masses plus HAIL, and a number moment for every one of the six
#: hydrometeors.  All six numbers are prognostic and none is diagnosed, so
#: all six are REQUIRED, QNCLOUD included: WRF's own driver binds
#: qnc/qnr/qni/qns/qng/qnh INOUT to ``mp_milbrandt2mom_driver``
#: (module_microphysics_driver.F:1857-1862), which is the same argument
#: that puts 16 and 28 in REQUIRED_QNCLOUD_MICROPHYSICS above.  QHAIL is a
#: MASS, so it is listed here rather than folded into ICE_MASS_WRFINPUT,
#: which is shared with the six-species schemes that have no hail at all.
MILBRANDT_MOISTURE_WRFINPUT = (
    "QHAIL", "QNCLOUD", "QNRAIN", "QNICE", "QNSNOW", "QNGRAUPEL",
    "QNHAIL",
)
P3_MOISTURE_WRFINPUT = ("QICE", "QNICE", "QNRAIN", "QIR", "QIB")
#: mp_physics=5 (Ferrier-Aligo, Registry.EM_COMMON:3019): one total-ice
#: mass, added through this tuple for P3's reason (no QSNOW/QGRAUP exists
#: in the package).  CWM, the scalar qt, is optional: see MOISTURE_MAP.
FERRIER_MOISTURE_WRFINPUT = ("QICE",)
FERRIER_OPTIONAL_MOISTURE_WRFINPUT = ("CWM",)
#: mp_physics=24 (WSM7).  ``Registry.EM_COMMON:3034`` declares the package
#: as ``moist:qv,qc,qr,qi,qs,qg,qh`` -- WSM6's six masses plus HAIL, no
#: scalar package.  QHAIL is a MASS, listed here rather than in
#: ICE_MASS_WRFINPUT for the reason MILBRANDT_MOISTURE_WRFINPUT gives.
WSM7_MOISTURE_WRFINPUT = ("QHAIL",)
NSSL_MOISTURE_WRFINPUT = (
    "QHAIL", "QNDROP", "QNRAIN", "QNICE", "QNSNOW", "QNGRAUPEL",
    "QNHAIL", "QNCCN", "QVGRAUPEL", "QVHAIL",
)

PHYSICS_FIELD_ALIASES = {
    "landmask": ("LANDMASK",), "xland": ("XLAND",),
    "tsk": ("TSK",), "pblh": ("PBLH",), "ivgtyp": ("IVGTYP", "LU_INDEX"),
    "isltyp": ("ISLTYP",), "vegfra": ("VEGFRA",), "tmn": ("TMN",),
    "xice": ("XICE", "SEAICE"), "swdown": ("SWDOWN",), "glw": ("GLW",),
    "snow": ("SNOW",), "snowh": ("SNOWH",), "smois": ("SMOIS",),
    "tslb": ("TSLB",), "sh2o": ("SH2O",), "psfc": ("PSFC",),
    "t2": ("T2",), "q2": ("Q2",), "th2": ("TH2",),
    "u10": ("U10",), "v10": ("V10",), "snoalb": ("SNOALB",),
    "albbck": ("ALBBCK", "ALBEDO"), "lai": ("LAI",),
    "shdmin": ("SHDMIN",), "shdmax": ("SHDMAX",),
    "ust": ("UST",), "znt": ("ZNT",), "hfx": ("HFX",),
    "qfx": ("QFX",), "lh": ("LH",), "grdflx": ("GRDFLX",),
    # RUC Registry input state -> the existing LSMRUC INOUT carriers.
    "acrunoff": ("ACRUNOFF",), "rhosnf": ("RHOSNF",),
    "snowfallac": ("SNOWFALLAC",), "soilt1": ("SOILT1",),
    # Stock v4.6.1 names this field qke; real.exe products also use QKE.
    "qke": ("qke", "QKE"),
    "qke_adv": ("qke_adv",),
}

RUC_INPUT_FIELDS = frozenset({"ACRUNOFF", "RHOSNF", "SNOWFALLAC", "SOILT1"})
from woof.static.orographic import OROGRAPHIC_ROWS, required_static_fields
OROGRAPHIC_WRFINPUT = frozenset(OROGRAPHIC_ROWS)
MYNN_QKE_INPUT_FIELDS = frozenset(PHYSICS_FIELD_ALIASES["qke"])
#: The MYNN subgrid cloud a cycled start (RunConfig.cycling) continues;
#: restored onto the physics fields of the same lower-case names.
MYNN_CYCLED_WRFINPUT = frozenset({"QC_BL", "CLDFRA_BL"})

# Optional restart-state fields have explicit consumers in
# ``restore_domain_state``.  They are permitted when present but are not
# synthesized when absent.
OPTIONAL_WRFINPUT = (
    "ICCG_IN_NUM", "ICCG_IN_DEN",
    "TML", "T0ML", "HML", "H0ML", "HUML", "HVML", "TMOML",
    "OM_TMP", "OM_S", "OM_U", "OM_V", "OM_DEPTH", "OM_TINI", "OM_SINI",
    "H_DIABATIC", "RAINNC", "RAINC", "QNCLOUD", "SST", "CANWAT", "LAKEMASK",
    "QNWFA2D", "QNIFA2D",
    "MAPFAC_MX", "MAPFAC_MY", "MAPFAC_UX", "MAPFAC_UY", "MAPFAC_VX", "MAPFAC_VY",
    "fire_smoke",
    # The reference-profile constants WRF's cold start rebuilds the base
    # state from (start_em.F:526-536); wrf_start_base_state reads them.
    "P00", "T00", "TLP", "TISO", "TLP_STRAT", "P_STRAT",
)

# These are the only non-science variable records allowed through the reader.
# ``Times`` is character metadata and is not copied into ``RestoredDomain.raw``.
EXPLICIT_AUXILIARY_WRFINPUT = (
    "Times", "XTIME", "ITIMESTEP", "FRC_URB2D",
    "XLAT", "XLONG", "XLAT_U", "XLONG_U", "XLAT_V", "XLONG_V",
)

_MASS_3D_DIMS = ("bottom_top", "south_north", "west_east")
_MASS_2D_DIMS = ("south_north", "west_east")
_U_3D_DIMS = ("bottom_top", "south_north", "west_east_stag")
_V_3D_DIMS = ("bottom_top", "south_north_stag", "west_east")
_W_3D_DIMS = ("bottom_top_stag", "south_north", "west_east")
_U_2D_DIMS = ("south_north", "west_east_stag")
_V_2D_DIMS = ("south_north_stag", "west_east")
_SOIL_DIMS = ("soil_layers_stag", "south_north", "west_east")
_WRFINPUT_GEOMETRY_DIMENSIONS = frozenset({
    "bottom_top", "bottom_top_stag", "south_north", "south_north_stag",
    "west_east", "west_east_stag", "soil_layers_stag",
})

# Field-specific staggering is checked while bytes are read, before any GPU
# state exists.  The dimension names are part of the contract as well as the
# resulting shape, so a truncated variable cannot borrow an unrelated
# dimension of the same length.
WRFINPUT_DIMENSIONS: dict[str, tuple[str, ...]] = {
    **{name: ("ocean_levels", "south_north", "west_east") for name in (
        "OM_TMP", "OM_S", "OM_U", "OM_V", "OM_DEPTH", "OM_TINI", "OM_SINI")},
    "ICCG_IN_NUM": _MASS_2D_DIMS, "ICCG_IN_DEN": _MASS_2D_DIMS,
    "RESM": (),
    "PC": _MASS_2D_DIMS,
    "O3_GFS_DU": _MASS_2D_DIMS,
    **{name: () for name in (
        "HFX_FORCE","LH_FORCE","TSK_FORCE","HFX_FORCE_TEND",
        "LH_FORCE_TEND","TSK_FORCE_TEND")},
    **{name: _MASS_2D_DIMS for name in OROGRAPHIC_WRFINPUT},
    **SFIRE_INPUT_DIMENSIONS,
    **{name: _MASS_3D_DIMS for name in (
        "T", "T_INIT", "P", "PB", "AL", "ALB", "QVAPOR", "QCLOUD",
        "QRAIN", "QICE", "QSNOW", "QGRAUP", "QNRAIN", "QNICE",
        "QNSNOW", "QNGRAUPEL", "QNCLOUD", "QHAIL", "QNDROP",
        "QNHAIL", "QNCCN", "QVGRAUPEL", "QVHAIL", "H_DIABATIC",
        "QIR", "QIB", "QNWFA", "QNIFA", "QNBCA", "qke", "QKE", "qke_adv",
        "CWM",
        "fire_smoke",
        "QC_BL", "CLDFRA_BL",
    )},
    "U": _U_3D_DIMS, "V": _V_3D_DIMS, "W": _W_3D_DIMS,
    "PH": _W_3D_DIMS, "PHB": _W_3D_DIMS,
    **{name: _MASS_2D_DIMS for name in (
        "MU", "MUB", "HGT", "MAPFAC_M", "MAPFAC_MX", "MAPFAC_MY", "F", "E", "SINALPHA",
        "COSALPHA", "LANDMASK", "LU_INDEX", "ISLTYP", "TSK", "TMN",
        "SNOW", "SNOWH", "VEGFRA", "SNOALB", "SHDMIN", "SHDMAX",
        "PSFC", "T2", "Q2", "TH2", "U10", "V10", "XLAND", "IVGTYP",
        "XICE", "SEAICE", "ALBBCK", "ALBEDO", "LAI", "LAI12M",
        "SWDOWN", "GLW", "QNWFA2D", "QNIFA2D", "FRC_URB2D",
        "PBLH", "UST", "ZNT", "HFX", "QFX", "LH", "GRDFLX", "RAINNC",
        "RAINC", "XLAT", "XLONG", "SST", "CANWAT", "LAKEMASK",
        "ACRUNOFF", "RHOSNF", "SNOWFALLAC", "SOILT1",
        "TML", "T0ML", "HML", "H0ML", "HUML", "HVML", "TMOML",
    )},
    "MAPFAC_U": _U_2D_DIMS, "MAPFAC_V": _V_2D_DIMS,
    "MAPFAC_UX": _U_2D_DIMS, "MAPFAC_UY": _U_2D_DIMS,
    "MAPFAC_VX": _V_2D_DIMS, "MAPFAC_VY": _V_2D_DIMS,
    "XLAT_U": _U_2D_DIMS, "XLONG_U": _U_2D_DIMS,
    "XLAT_V": _V_2D_DIMS, "XLONG_V": _V_2D_DIMS,
    **{name: _SOIL_DIMS for name in ("TSLB", "SMOIS", "SH2O")},
    **{name: ("bottom_top",) for name in (
        "FNM", "FNP", "RDNW", "RDN", "DNW", "DN", "ZNU",
        "C1H", "C2H", "C3H", "C4H",
    )},
    **{name: ("bottom_top_stag",) for name in (
        "ZNW", "C1F", "C2F", "C3F", "C4F",
    )},
    **{name: () for name in (
        "P_TOP", "CF1", "CF2", "CF3", "XTIME", "ITIMESTEP",
        "P00", "T00", "TLP", "TISO", "TLP_STRAT", "P_STRAT",
        "SAVE_TOPO_FROM_REAL",
    )},
}
from woof.io.wrf_input_history_schema import WRF_INPUT_HISTORY_ROWS
WRFINPUT_DIMENSIONS.update({name: tuple(row["input_dimensions"])
                           for name, row in WRF_INPUT_HISTORY_ROWS.items()})

# Registry urban static records are present in ordinary real.exe files even
# when their optional gridded morphology and external heat are all zero.
# They are not forecast prognostics. The current cold initializer uses the
# selected URBPARM morphology. Nonzero active records need the missing
# gridded-morphology/external-heat handover and must never be discarded.
URBAN_TABLE_ONLY_INPUTS = {
    **{name: ("Time", "south_north", "west_east") for name in (
        "BUILD_AREA_FRACTION", "BUILD_HEIGHT", "BUILD_SURF_RATIO",
        "LF_URB2D_S", "MH_URB2D", "STDH_URB2D", "Z0_URB2D")},
    "LF_URB2D": ("Time", "urban_wind_directions_stag", "south_north", "west_east"),
    "AHE": ("Time", "month_hour", "south_north", "west_east"),
}


def urban_static_input_dispositions(dataset, path, cfg, expected_dimensions):
    """Admit inactive static records, retaining a concrete active-data refusal."""
    active = int(getattr(cfg, "sf_urban_physics", 0)) > 0
    dispositions = {}
    for name, dimensions in URBAN_TABLE_ONLY_INPUTS.items():
        if name not in dataset.variables:
            continue
        variable = dataset.variables[name]
        value = np.asarray(variable[...])
        shape = tuple(1 if dim == 'Time' else expected_dimensions[dim]
                      if dim in expected_dimensions else len(dataset.dimensions[dim])
                      for dim in dimensions)
        if tuple(variable.dimensions) != dimensions or value.shape != shape or any(n <= 0 for n in shape):
            raise ValueError(f'{path}: urban static input {name} has incompatible dimensions or shape; expected {dimensions} {shape}')
        if active and (not np.isfinite(value).all() or np.any(value != 0)):
            raise ValueError(f'{path}: nonzero or nonfinite urban static input {name} requires the missing gridded urban morphology or external anthropogenic-heat handover; the table-only cold initializer cannot preserve it')
        dispositions[name] = ('zero optional urban morphology/external-heat record; the cold start uses the selected URBPARM table'
                              if active else 'urban static record is inactive with sf_urban_physics=0')
    return dispositions


def _mapped_wrfinput_names() -> set[str]:
    names = set(REQUIRED_WRFINPUT) | set(OPTIONAL_WRFINPUT)
    names.update(WRF_INPUT_HISTORY_ROWS)
    names.update(SFIRE_INPUT_DIMENSIONS)
    names.add("RESM")
    names.update(("PC", "O3_GFS_DU"))
    names.update(("HFX_FORCE","LH_FORCE","TSK_FORCE","HFX_FORCE_TEND",
                  "LH_FORCE_TEND","TSK_FORCE_TEND"))
    names.update(alias for aliases in ALIASES.values() for alias in aliases)
    names.update(ALL_MOISTURE_WRFINPUT)
    names.update(OROGRAPHIC_WRFINPUT)
    names.update(
        alias for aliases in PHYSICS_FIELD_ALIASES.values() for alias in aliases)
    return names


WRFINPUT_DIMENSIONS.update({name:WRFINPUT_DIMENSIONS["QVAPOR"] for name in _FSBM_BIN_SPECIES + ("EFF_RADIUS","ICE_EFF_RADIUS","TOT_EFF_RADIUS")})

MAPPED_WRFINPUT = frozenset(_mapped_wrfinput_names())
# Registry writes QNBCA for mp28 even when wif_input_opt != 2. In that
# selection Thompson sets its local and returned BC aerosol to zero
# (module_mp_thompson.F:1807-1811,3983-3988). Retain and validate the input
# without claiming a prognostic black-carbon consumer exists in ArWen.
INACTIVE_AEROSOL_WRFINPUT = frozenset({"QNBCA"})
# Registry.EM_COMMON:1118,3168 allocates the MYNN scalar QKE carrier.
# With advection selected it is restored independently from diagnostic QKE.
MYNN_SCALAR_WRFINPUT = frozenset({"qke_adv"})
ALLOWED_WRFINPUT = (MAPPED_WRFINPUT | frozenset(EXPLICIT_AUXILIARY_WRFINPUT)
                   | INACTIVE_AEROSOL_WRFINPUT | MYNN_SCALAR_WRFINPUT)

#: WRF's 1-D base-state profiles.  real.exe never assigns them
#: (Registry.EM_COMMON:372 and :1247-1249 name them "BASE STATE ... IN
#: IDEALIZED CASES"; only the module_initialize_<ideal>.F routines,
#: module_force_scm.F and nest_init_utils.F's parent-to-nest copy write
#: them) and WRF's allocation zero-fills them, so a real.exe wrfinput
#: carries zeros and an ideal.exe wrfinput carries its sounding.  They are
#: not restored (the engine's operators carry the zero profile), which is
#: why they also sit in IGNORED_WRFINPUT; they are read only by
#: :func:`require_zero_base_state_profiles`, and
#: :func:`base_state_profiles_read` says, path by path with its WRF line,
#: which selectors read which profile (diff_opt 2 and diff_opt 1 alike).
BASE_WIND_PROFILE_WRFINPUT = ("U_BASE", "V_BASE")
BASE_SCALAR_PROFILE_WRFINPUT = ("T_BASE", "QV_BASE")

#: Standard real.exe wrfinput variables the restored model does not consume
#: (F20 conformance defines exactly what is restored; everything else is
#: skipped).  Enumerated explicitly â€” first contact with the production
#: handoff inputs (registered identity e2c6fdf4...) surfaced these â€” so an
#: unanticipated variable still fails loudly instead of being silently
#: ignored.  THM/P_HYD are present-but-unconsumed under the registered
#: use_theta_m=0 restoration; SST/land-use climatology fields are superseded
#: by the mapped surface restoration set.
IGNORED_WRFINPUT = frozenset({
    "BATHYMETRY_FLAG", "CFN", "CFN1", "CLAT", "CLDFRA", "CPLMASK",
    "DTS", "DTSEPS", "DZS", "EROD", "FCX", "FNDALBSI", "FNDICEDEPTH",
    "FNDSNOWH", "FNDSNOWSI", "FNDSOILW", "GCX", "GOT_VAR_SSO",
    "LAKEFLAG", "LAKE_DEPTH", "LAKE_DEPTH_FLAG", "LANDUSEF",
    "LAT_LL_D", "LAT_LL_T", "LAT_LL_U", "LAT_LL_V", "LAT_LR_D", "LAT_LR_T",
    "LAT_LR_U", "LAT_LR_V", "LAT_UL_D", "LAT_UL_T", "LAT_UL_U", "LAT_UL_V",
    "LAT_UR_D", "LAT_UR_T", "LAT_UR_U", "LAT_UR_V", "LON_LL_D", "LON_LL_T",
    "LON_LL_U", "LON_LL_V", "LON_LR_D", "LON_LR_T", "LON_LR_U", "LON_LR_V",
    "LON_UL_D", "LON_UL_T", "LON_UL_U", "LON_UL_V", "LON_UR_D", "LON_UR_T",
    "LON_UR_U", "LON_UR_V", "MF_VX_INV",
    "PCB", "P_HYD", "QV_BASE", "RDX", "RDY",
    "SAVE_TOPO_FROM_REAL", "SHDAVG", "SMCREL", "SNOWC", "SOILCBOT",
    "SOILCTOP", "SR", "STEP_NUMBER", "THIS_IS_AN_IDEAL_RUN",
    "THM", "TOPOSLPX", "TOPOSLPY", "T_BASE",
    "UOCE", "U_BASE", "U_FRAME", "VAR", "VAR_SSO", "VOCE", "V_BASE",
    "V_FRAME", "WATER_DEPTH", "ZETATOP", "ZS", "Z_BASE",
}) - OROGRAPHIC_WRFINPUT

#: Variables a WRF input file carries that this model has no state for and
#: an in-place analysis may read or rewrite, each with the reason it is
#: passed through.  They are named, not waved through: the reader accepts
#: them, restores nothing from them, and
#: :func:`woof.io.analysis_exchange.write_back` hands them back with the
#: bytes they arrived with, so the next analysis reads the file it wrote.  A
#: name outside every inventory is still an error.  A row leaves this table
#: when the model gains a consumer for it.
#:
#: Sources.  The two analysis rows come from the analysis program's own WRF
#: mass-core netCDF interface, GSI as built into NOAA-EMC/HRRR tag v4.1.21
#: (sorc/hrrr_gsi.fd/src/gsi/cplr_wrf_netcdf_interface.f90): every other
#: variable that file reads (:179-:1179) or writes back (:2628-:3189) is
#: already mapped above.  The grouped rows are every remaining variable of
#: that tag's own real.exe output for its CONUS configuration (WRFV3.9 with
#: its Registry; 363 variables, MEASURED 2026-10-03 on the 2026-10-03 12 UTC
#: cold-start guess), sorted by what they are.
_ANALYSIS_PASSTHROUGH_ROWS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("the forecast model's own radar reflectivity diagnostic, which the "
     "analysis reads as its reflectivity background and writes back "
     "(cplr_wrf_netcdf_interface.f90:1203, :3217); this model diagnoses "
     "reflectivity from its hydrometeors and restores nothing from it",
     ("REFL_10CM",)),
    ("a radar-derived temperature tendency for the forecast's latent-heating "
     "forcing, which the cloud analysis writes "
     "(cplr_wrf_netcdf_interface.f90:1229, :3245) and a WRF input carries "
     "as four 15-minute slots and their times; it is not initial state.  "
     "Under mp_tend_radar = 1 the forecast applies it "
     "(woof/ingest/wrfinput_radar.py reads it from this file); otherwise "
     "it is carried and not applied, as wrf.exe does",
     ("RAD_TTEN_DFI", "RAD_TTEN_DFI_1", "RAD_TTEN_DFI_2", "RAD_TTEN_DFI_3",
      "RAD_TTEN_DFI_4", "TTEN_TIMES")),
    ("a forecast diagnostic (an accumulation, a maximum or minimum since the "
     "last output, a mean, a rate or a derived product); a forecast starts "
     "it from its own state and nothing reads it as initial state",
     ("WSPD10", "WSPD80", "SFROFF", "UDROFF", "ACSNOW", "ACGRAUP", "ACSNOM",
      "ACFRAIN", "BR", "ZOL", "FLHC", "CUTOP", "CUBOT", "CUPPT", "MAX_MSTFX",
      "MAX_MSTFY", "FRAIN", "PRATEC", "RAINCV", "RAINNCV", "SNOWNC",
      "GRAUPELNC", "SNOWNCV", "GRAUPELNCV", "COMPOSITE_REFL_10CM",
      "REFL_10CM_1KM", "REFL_10CM_4KM", "LWP", "IWP", "GSW", "SWDDNI",
      "SWDDIF", "SWDDNIC", "SWDDIFC", "TAOD5502D", "SWRADMEAN", "SWNORMMEAN",
      "U10MEAN", "V10MEAN", "SPDUV10MEAN", "OLR", "WSPD10MAX", "WSPD10UMAX",
      "WSPD10VMAX", "WSPD80MAX", "WSPD80UMAX", "WSPD80VMAX", "W_UP_MAX",
      "W_DN_MAX", "REFD_MAX", "REFDM10C_MAX", "UP_HELI_MAX", "UP_HELI_MAX16",
      "UP_HELI_MAX02", "UP_HELI_MAX03", "UP_HELI_MIN", "UP_HELI_MIN16",
      "UP_HELI_MIN02", "UP_HELI_MIN03", "REL_VORT", "REL_VORT_MAX",
      "REL_VORT_MAX01", "W_MEAN", "GRPL_MAX", "LTG1_MAX", "LTG2_MAX",
      "LTG3_MAX", "NCI_LTG", "NCA_LTG", "NCI_W", "NCA_W", "NCI_WQ",
      "NCA_WQ", "NCI_REFD", "NCA_REFD", "QR_MAX_CI", "QG_MAX_CI", "UH",
      "UH16", "UH02", "UH03", "HAIL_MAXK1", "HAIL_MAX2D", "MAXCLDFRA",
      "PREC_ACC_C", "PREC_ACC_C1", "PREC_ACC_NC", "PREC_ACC_NC1",
      "SNOW_ACC_NC", "SNOW_ACC_NC1", "GRAUP_ACC_NC", "GRAUP_ACC_NC1",
      "NUPDRAFT", "KTOP_PLUME", "MAXMF", "MF_AT_BASE", "KBOT_SHALLOW")),
    ("scheme state this reader does not restore (surface moisture, subgrid "
     "cloud, soil albedo, lake column, boundary clock); this model "
     "initializes it at the start of a run, so a value a cycled file "
     "carries is not continued",
     ("MU0", "DTBC", "QVG", "QCG", "MAVAIL", "QC_BL", "CLDFRA_BL", "ALBSOL",
      "ALBBCKSOL", "LAKEDEPTH2D", "SAVEDTKE12D", "SNOWDP2D", "H2OSNO2D",
      "SNL2D", "T_GRND2D", "T_LAKE3D", "LAKE_ICEFRAC3D", "Z_LAKE3D",
      "DZ_LAKE3D", "T_SOISNO3D", "H2OSOI_ICE3D", "H2OSOI_LIQ3D",
      "H2OSOI_VOL3D", "Z3D", "DZ3D", "ZI3D", "WATSAT3D", "CSOL3D", "TKMG3D",
      "TKDRY3D", "TKSATU3D")),
    # The sub-grid orography statistics of the gravity-wave drag schemes
    # (CON, OA1..OA4, OL1..OL4 and their SS/LS sets) left this table when
    # the verification lane (901a3fd0927) mapped OROGRAPHIC_WRFINPUT: the
    # reader restores them for woof.core.terrain_drag, so they are
    # ALLOWED_WRFINPUT, and a name cannot be both restored and passed
    # through (tests/test_analysis_exchange.py).
    ("a random-number seed of the stochastic perturbation schemes, which "
     "this model does not run",
     ("ISEEDARR_SPPT", "ISEEDARR_SKEBS", "ISEEDARR_RAND_PERTURB",
      "ISEEDARRAY_SPP_CONV", "ISEEDARRAY_SPP_PBL", "ISEEDARRAY_SPP_LSM",
      "ISEEDARRAY_SPP_MP")),
    ("a smoke tracer or fire emission input of the coupled smoke scheme; "
     "this reader does not restore it",
     ("smoke", "EBB_SMOKE", "MEAN_FRP", "STD_FRP", "MEAN_FSIZE",
      "STD_FSIZE")),
    ("a vertical-coordinate weight of the older hybrid coordinate; the C1 "
     "to C4 coefficients carry the same coordinate and are restored",
     ("BF", "BH")),
)


def _analysis_passthrough(rows) -> Mapping[str, str]:
    table: dict[str, str] = {}
    for reason, names in rows:
        for name in names:
            if name in table:
                raise ValueError(f"{name} appears twice in the pass-through rows")
            table[name] = reason
    return MappingProxyType(table)


ANALYSIS_PASSTHROUGH_WRFINPUT: Mapping[str, str] = _analysis_passthrough(
    _ANALYSIS_PASSTHROUGH_ROWS)

#: The radar temperature tendency ``mp_tend_radar = 1`` applies: four
#: 15-minute slots in NOAA's file convention and their end times in
#: minutes (the operational fork's Registry; RAD_TTEN_DFI, the single
#: static field of the older RAP arrangement, is not one of them).
RADAR_TENDENCY_VARIABLES = ("RAD_TTEN_DFI_1", "RAD_TTEN_DFI_2",
                            "RAD_TTEN_DFI_3", "RAD_TTEN_DFI_4", "TTEN_TIMES")


# ==========================================================================
# The scheme matrix: which real.exe products this door restores, by name.
# ==========================================================================
#
# A closed-world importer that refuses with "unmapped WRF variable(s):
# ['QAOLI', 'QICE2', ...]" has told the reader nothing they can act on.
# The names in that list are the hydrometeor inventory of ONE WRF
# microphysics package, and the file's own ``MP_PHYSICS`` global attribute
# says which -- ``real.exe`` writes it into every wrfinput it produces.
# So the door reads the attribute FIRST and refuses by scheme name with a
# remedy, and the variable-inventory check below it becomes the second
# line of defence rather than the message a user sees.

#: WRF ``mp_physics`` -> (scheme name, the wrfinput hydrometeor inventory
#: this door restores for it).  Membership here is the door's microphysics
#: contract: exactly the values :func:`_active_moisture_inventory` builds
#: an inventory for, in one table so the refusal and the reader cannot
#: disagree about what is supported.
SUPPORTED_MICROPHYSICS: Mapping[int, str] = MappingProxyType({
    0: "passive water vapor (microphysics off)",
    1: "Kessler warm rain",
    2: "Purdue Lin",
    3: "WSM3",
    4: "WSM5",
    6: "WSM6",
    8: "Thompson",
    106: "WSM6R (revised WSM6)",
    9: "Milbrandt-Yau double-moment (seven class)",
    10: "Morrison double-moment",
    16: "WDM6 double-moment warm rain",
    18: "NSSL double-moment",
    5: "Ferrier-Aligo",
    24: "WSM7 single-moment (seven class)",
    28: "Thompson aerosol-aware",
    30: "FAST_SBM warm and cold spectral bin",
    50: "P3 one-category",
})

#: ``mp_physics`` values that are real WRF v4.6.1 packages this port does
#: NOT restore, each with the reason and the remedy.  Every one of these
#: writes a wrfinput whose hydrometeor names ArWen has no state field for,
#: so restoring it would mean dropping prognostics on the floor.  The
#: three P3 siblings recite ``woof.config``'s own refusal text, which is
#: where their missing physics is argued in detail.
#:
#: The names are WRF v4.6.1's own (Registry/Registry.EM_COMMON:3014-3047,
#: the ``package ... mp_physics==N`` rows).  A WRF package missing from
#: this table was refused as "not a WRF microphysics package", which is
#: false for a real one, and 55/56 used to carry P3 names although WRF
#: assigns them to Jensen ISHMAEL and NTU.  tests/test_wrfinput_mp_names.py
#: pins every Registry row to this table or to SUPPORTED_MICROPHYSICS.
UNSUPPORTED_MICROPHYSICS: Mapping[int, str] = MappingProxyType({
    7: "Goddard 4-ice",
    11: "CAM 5.1",
    13: "SBU-YLin",
    14: "WDM5",
    15: "Ferrier-Aligo, advected species",
    17: "NSSL 2-moment 4-ice",
    19: "NSSL 1-moment",
    21: "NSSL 1-moment lfo",
    22: "NSSL 2-moment, no hail",
    26: "WDM7",
    32: "HUJI spectral bin (full)",
    38: "Thompson graupel-hail",
    40: "Morrison + CESM aerosol",
    51: "P3 with prognostic droplet number",
    52: "P3, two ice categories",
    53: "P3 one category, three-moment ice",
    55: "Jensen ISHMAEL",
    56: "NTU multi-moment",
    95: "Ferrier (Eta, old)",
    96: "MADWRF",
    97: "Goddard GCE",
    106: "WSM6 revised",
})

#: WRF ``sf_surface_physics`` -> (scheme name, the soil-layer count the
#: scheme's wrfinput carries).  This is the LAND-SURFACE half of the
#: contract and it is checked separately from microphysics, because the
#: two fail for different reasons and a reader fixing one wants to be told
#: about the other in the same breath.  The counts are WRF's own
#: (module_check_a_mundo.F's num_soil_layers table); the accepted set is
#: :data:`woof.config.LAND_SURFACE_SCHEMES`.
WRF_LAND_SURFACE_SCHEMES: Mapping[int, tuple[str, int]] = MappingProxyType({
    0: ("no land-surface model", 4),
    1: ("5-layer thermal diffusion (slab)", 5),
    2: ("Noah LSM", 4),
    3: ("RUC LSM", 9),
    4: ("Noah-MP LSM", 4),
    5: ("CLM4", 10),
    7: ("Pleim-Xiu LSM", 2),
    8: ("SSiB", 3),
})


def _integral_attribute(value) -> int | None:
    """A NetCDF global attribute as an int, or None if it is not one.

    WRF writes these as NC_INT, but woof decodes NetCDF through the Rust
    bridge, which promotes every numeric type to f64 -- so the same
    ``MP_PHYSICS = 6`` arrives as ``6`` from netCDF4 and ``6.0`` from the
    bridge.  Accepting an integral float here is what makes the scheme
    check read the same fact through either reader; accepting a
    NON-integral one would let ``6.5`` be scored as WSM6.
    """
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return int(value) if float(value).is_integer() else None
    if isinstance(value, np.ndarray) and value.size == 1:
        return _integral_attribute(value.reshape(-1)[0])
    return None


def _named_microphysics(mp_physics: int) -> str:
    """``6`` -> ``"mp_physics=6 (WSM6)"``; unknown values stay bare."""
    name = SUPPORTED_MICROPHYSICS.get(
        mp_physics, UNSUPPORTED_MICROPHYSICS.get(mp_physics))
    return (f"mp_physics={mp_physics}"
            if name is None else f"mp_physics={mp_physics} ({name})")


def supported_microphysics_sentence() -> str:
    """The accepted microphysics menu, as one sentence, for every refusal."""
    return "this door restores " + ", ".join(
        f"{value} ({SUPPORTED_MICROPHYSICS[value]})"
        for value in sorted(SUPPORTED_MICROPHYSICS))


def supported_land_surface_sentence() -> str:
    """The accepted land-surface menu, as one sentence, for every refusal."""
    from woof.config import LAND_SURFACE_SCHEMES
    return "this door restores " + ", ".join(
        f"{value} ({WRF_LAND_SURFACE_SCHEMES[value][0]})"
        for value in sorted(LAND_SURFACE_SCHEMES))


def unsupported_microphysics_refusal(mp_physics: int) -> str:
    """One sentence naming the scheme, why it is refused, and the remedy."""
    if mp_physics in UNSUPPORTED_MICROPHYSICS:
        head = (
            f"{_named_microphysics(mp_physics)} is a WRF package this port "
            "does not carry, so its wrfinput hydrometeor inventory has no "
            "WOOF state to be restored into")
    else:
        head = (
            f"mp_physics={mp_physics} is not a WRF microphysics package "
            "this door recognises")
    return (
        f"{head}. {supported_microphysics_sentence()}. "
        "Remedy: re-run real.exe with one of those mp_physics values in "
        "namelist.input -- real.exe writes a DIFFERENT set of moisture "
        "variables per scheme, so an existing wrfinput cannot be "
        "reinterpreted under another one.")


def unsupported_land_surface_refusal(sf_surface_physics: int) -> str:
    """One sentence naming the land-surface scheme and the remedy."""
    entry = WRF_LAND_SURFACE_SCHEMES.get(sf_surface_physics)
    if entry is None:
        head = (f"sf_surface_physics={sf_surface_physics} is not a WRF "
                "land-surface scheme this door recognises")
        soil = ""
    else:
        head = (f"sf_surface_physics={sf_surface_physics} ({entry[0]}) is a "
                "WRF land-surface scheme this port does not carry")
        soil = (f" Its wrfinput carries {entry[1]} soil layers and the soil "
                "state that scheme writes, which is not the state WOOF's "
                "land-surface drivers read.")
    return (
        f"{head}.{soil} {supported_land_surface_sentence()}. "
        "Remedy: re-run real.exe with one of those sf_surface_physics "
        "values in namelist.input -- the soil geometry and the soil "
        "fields real.exe writes are chosen by this setting, so an "
        "existing wrfinput cannot be reinterpreted under another one.")


def check_supported_schemes(attributes: Mapping[str, object], *,
                            source: str) -> dict[str, int]:
    """Refuse a wrfinput whose physics packages this door cannot restore.

    ``attributes`` is a wrfinput's global attribute mapping.  Returns the
    two resolved scheme ids on success.

    BOTH halves are reported together.  A reader who has just been told
    their microphysics is unported will change it, re-run ``real.exe`` for
    an hour, and be told about their land-surface scheme; one refusal
    listing both is one round trip instead of two.
    """
    from woof.config import LAND_SURFACE_SCHEMES

    problems = []
    resolved = {}
    for key, accepted, refusal in (
            ("MP_PHYSICS", set(SUPPORTED_MICROPHYSICS),
             unsupported_microphysics_refusal),
            ("SF_SURFACE_PHYSICS", set(LAND_SURFACE_SCHEMES),
             unsupported_land_surface_refusal)):
        if key not in attributes:
            # An absent attribute is not agreement.  real.exe writes both
            # into every wrfinput it produces; a file without them was not
            # written by real.exe, or was rewritten by something that
            # dropped them, and either way this door cannot establish
            # which physics the moisture inventory belongs to.
            problems.append(
                f"{source} has no {key} global attribute, so the physics "
                "package that produced it cannot be established.  Every "
                "real.exe wrfinput carries one; a file that does not was "
                "not written by real.exe or was rewritten by a tool that "
                "dropped it.")
            continue
        value = _integral_attribute(attributes[key])
        if value is None:
            problems.append(
                f"{source} global attribute {key}={attributes[key]!r} is "
                "not an integer scheme id")
            continue
        resolved[key] = value
        if value not in accepted:
            problems.append(f"{source}: {refusal(value)}")
    if problems:
        raise ConfigurationRefused("\n".join(problems))
    return resolved
@dataclass(frozen=True)
class RestoredDomain:
    path: Path
    raw: Mapping[str, np.ndarray]
    dimensions: Mapping[str, int]
    global_attributes: Mapping[str, object]
    mapped_variables: tuple[str, ...]
    auxiliary_variables: tuple[str, ...]
    surface_input_dispositions: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({}))
    soil_unit_conversions: Mapping[str, object] = field(
        default_factory=lambda: MappingProxyType({}))
    soil_recovery: Mapping[str, object] = field(
        default_factory=lambda: MappingProxyType({}))
    #: Set only for a domain read from a WRF-written restart file
    #: (:mod:`woof.ingest.wrfrst`): the restart instant, the clock and the
    #: per-variable dispositions.  ``None`` is a real.exe cold start.  The
    #: shared restore functions read it to take WRF's restart branch: the
    #: file's own base state and vertical velocity, no cold-start rebuild.
    warm_start: object | None = None

    def wrf_frame(self) -> dict[str, np.ndarray]:
        """CPU inverse of the restored atmospheric mapping."""
        raw = self.raw
        theta_prime = np.asarray(
            (raw["T"] + np.float32(300.0)) - raw["T_INIT"],
            dtype=np.float32)
        return {
            "U": raw["U"].copy(), "V": raw["V"].copy(),
            "W": raw["W"].copy(),
            "T": np.asarray(raw["T_INIT"] + theta_prime
                            - np.float32(300.0), dtype=np.float32),
            "PH": raw["PH"].copy(), "MU": raw["MU"].copy(),
            "PHB": raw["PHB"].copy(), "MUB": raw["MUB"].copy(),
            "QVAPOR": raw["QVAPOR"].copy(),
        }


def base_state_profiles_read(cfg) -> tuple[str, ...]:
    """The wrfinput base-state profiles WRF's arithmetic reads for ``cfg``.

    Read on WRFV3.9 as operational HRRR v4.1.21 builds it; v4.7.1 has the
    same call guards.  Every path below is one WRF takes:

    * ``diff_opt = 2``, ``mix_full_fields = false``: ``cal_deform_and_div``
      (called for diff_opt 1 OR 2, module_first_rk_step_part2.F:448 and
      :537) forms du/dz and dv/dz from ``u - u_base`` and ``v - v_base``
      (module_diffusion_em.F:842-860, :1017-1035), and under diff_opt 2
      D13/D23 reach horizontal_diffusion_2 whatever the PBL (part2.F:968).
      With no PBL scheme (part2.F:925-927) ``vertical_diffusion_2`` also
      mixes ``thp - t_base`` and ``qv - qv_base``
      (module_diffusion_em.F:3694-3712, :3778-3792) against xkhv, which
      smag2d_km (km_opt 4) sets to 0 (:2044) and isotropic_km (km_opt 1)
      sets from kvdif.
    * ``diff_opt = 1``, ``mix_full_fields = false``: the same D13/D23 are
      formed, but under diff_opt 1 only smag_km (km_opt 3) reads them
      (module_diffusion_em.F:1313, :1869-1880).  tke_rhs, whose tke_shear
      reads them under km_opt 2, runs only under diff_opt 2
      (part2.F:888), as do sfs_driver (:674) and the diffusion_2 operators
      (:925); smag2d_km reads only D11, D22 and D12.
    * ``diff_opt = 1`` with no PBL scheme, at EITHER value of
      ``mix_full_fields``: module_em.F:804 and :844-853 call
      vertical_diffusion_u and _v with ``u_base``/``v_base``, and :1368
      and :1381-1385 call
      vertical_diffusion_mp with ``qv_base`` for moisture
      (module_big_step_utilities_em.F:3554, :3752, :3853), each scaled by
      the constant kvdif, so they are read only when kvdif is not 0.
    """
    if cfg is None:
        return ()
    diff_opt = getattr(cfg, "diff_opt", 2)
    perturbation = not getattr(cfg, "mix_full_fields", True)
    no_pbl = getattr(cfg, "bl_pbl_physics", 0) == 0
    km_opt = getattr(cfg, "km_opt", 4)
    kvdif = getattr(cfg, "kvdif", 0.0)
    wind = theta = moisture = False
    if diff_opt == 2 and perturbation:
        wind = True
        if no_pbl and km_opt != 4 and not (km_opt == 1 and kvdif == 0):
            theta = moisture = True
    elif diff_opt == 1:
        if perturbation and km_opt == 3:
            wind = True
        if no_pbl and kvdif != 0:
            wind = True
            moisture = bool(getattr(cfg, "moist", True))
    return tuple(
        name for name, read in zip(
            BASE_WIND_PROFILE_WRFINPUT + BASE_SCALAR_PROFILE_WRFINPUT,
            (wind, wind, theta, moisture))
        if read)


def require_zero_base_state_profiles(dataset, path, cfg) -> None:
    """Refuse the one input WRF's base-state subtraction would treat differently.

    WRF subtracts the 1-D base-state profiles
    :func:`base_state_profiles_read` names before it mixes.  The engine's
    operators carry the zero profile real.exe leaves, so a file whose
    profile is not zero would be mixed as the full field where WRF mixes
    the departure from that profile.  Every real.exe wrfinput passes (its
    profiles are zero by construction); an ideal.exe file carrying its
    sounding is refused by name.  ``cfg=None`` and every selector set that
    reads nothing pass.
    """
    nonzero = []
    for name in base_state_profiles_read(cfg):
        if name not in dataset.variables:
            continue
        values = _read_numeric(dataset.variables[name])
        if np.any(values != 0):
            peak = float(np.max(np.abs(values)))
            nonzero.append(f"{name} (largest magnitude {peak:.6g})")
    if nonzero:
        diff_opt = getattr(cfg, "diff_opt", 2)
        if diff_opt == 2:
            where = ("under mix_full_fields = false with diff_opt = 2: WRF "
                     "subtracts that profile before it mixes (the wind pair "
                     "from the vertical shear in cal_deform_and_div, the "
                     "theta/qv pair in vertical_diffusion_2 when no PBL "
                     "scheme runs)")
            remedy = "set mix_full_fields = true"
        else:
            where = ("under diff_opt = 1: WRF subtracts that profile before "
                     "it mixes (the wind pair from the vertical shear that "
                     "smag_km reads under km_opt = 3 with mix_full_fields = "
                     "false, and the wind and qv profiles in the constant-"
                     "kvdif vertical diffusion when no PBL scheme runs, at "
                     "either mix_full_fields value)")
            remedy = ("set mix_full_fields = true and kvdif = 0, or run a "
                      "PBL scheme")
        raise ConfigurationRefused(
            f"{path} carries a nonzero base-state profile "
            f"({', '.join(nonzero)}) {where}, and the engine's operators "
            "carry only the zero profile real.exe writes, so this run would "
            "mix the full field where WRF mixes the departure from the "
            "profile. A real.exe wrfinput carries zeros and passes; for this "
            f"ideal.exe input {remedy}.")


def _read_numeric(variable, *, decoded=None) -> np.ndarray:
    value = np.ma.asarray(variable[...] if decoded is None else decoded)
    if np.ma.isMaskedArray(value) and np.any(np.ma.getmaskarray(value)):
        raise ConfigurationRefused(f"WRF input variable {variable.name} contains masked data")
    array = np.asarray(value)
    if variable.dimensions and variable.dimensions[0] == "Time":
        if array.shape[0] != 1:
            raise ConfigurationRefused(
                f"WRF input {variable.name} must carry exactly one Time record")
        array = array[0]
    if array.dtype.kind not in "iufb":
        raise TypeError(f"WRF input {variable.name} is not numeric")
    if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
        raise ConfigurationRefused(f"WRF input {variable.name} contains non-finite values")
    # ``np.ascontiguousarray`` promotes a zero-dimensional value to ``(1,)``.
    # WRF writes Registry scalars as ``(Time,)`` records, so preserve the
    # scalar produced by removing that singleton Time dimension.
    return np.array(array, copy=True, order="C")

def _explicit_wrfinput_dimensions(
        dimensions: Mapping[str, int]) -> Mapping[str, int]:
    """Validate a caller-pinned, non-N5S WRF domain geometry.

    Requiring the complete seven-dimension contract keeps an arbitrary file
    from defining its own expected geometry and thereby making a truncated
    but self-consistent handoff look valid.
    """
    if not isinstance(dimensions, Mapping):
        raise TypeError("expected_dimensions must be a mapping")
    names = set(dimensions)
    missing = sorted(_WRFINPUT_GEOMETRY_DIMENSIONS - names)
    extra = sorted(names - _WRFINPUT_GEOMETRY_DIMENSIONS)
    if missing or extra:
        raise ConfigurationRefused(
            "explicit WRF geometry dimension inventory mismatch: "
            f"missing={missing}, extra={extra}")
    normalized = {}
    for name in sorted(_WRFINPUT_GEOMETRY_DIMENSIONS):
        value = dimensions[name]
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or int(value) <= 0):
            raise ConfigurationRefused(
                f"explicit WRF geometry {name} must be a positive integer")
        normalized[name] = int(value)
    for mass, staggered in (
            ("bottom_top", "bottom_top_stag"),
            ("south_north", "south_north_stag"),
            ("west_east", "west_east_stag")):
        if normalized[staggered] != normalized[mass] + 1:
            raise ConfigurationRefused(
                f"explicit WRF geometry {staggered} must equal {mass} + 1")
    return MappingProxyType(normalized)


def active_moisture_inventory(cfg) -> tuple[frozenset[str], frozenset[str]]:
    """Return required/allowed wrfinput moisture names for ``cfg``.

    ``cfg=None`` is the registered N5S compatibility path and retains the
    historical Morrison contract verbatim.
    """
    if cfg is None:
        required = frozenset(
            BASE_MOISTURE_WRFINPUT + ICE_MASS_WRFINPUT
            + MORRISON_NUMBER_WRFINPUT)
        return required, required | frozenset(
            MORRISON_OPTIONAL_MOISTURE_WRFINPUT)
    if not hasattr(cfg, "moist") or not hasattr(cfg, "mp_physics"):
        raise TypeError("cfg must expose moist and mp_physics")
    if not isinstance(cfg.moist, (bool, np.bool_)):
        raise TypeError("cfg.moist must be boolean")
    if (isinstance(cfg.mp_physics, (bool, np.bool_))
            or not isinstance(cfg.mp_physics, (int, np.integer))):
        raise TypeError("cfg.mp_physics must be an integer")
    moist = bool(cfg.moist)
    mp_physics = int(cfg.mp_physics)
    if mp_physics not in SUPPORTED_MICROPHYSICS:
        # NAMED, with a remedy.  This used to read "unsupported active
        # wrfinput mp_physics=55", which tells a reader the number they
        # already typed and nothing else.
        raise ConfigurationRefused(unsupported_microphysics_refusal(mp_physics))
    if not moist:
        if mp_physics != 0:
            raise ConfigurationRefused(
                f"mp_physics={mp_physics} requires cfg.moist=True")
        return frozenset(), frozenset()
    if mp_physics == 0:
        # WRF's passiveqv package retains advected vapor with microphysics
        # disabled. Cloud and precipitation species remain inactive.
        vapor = frozenset(("QVAPOR",))
        return vapor, vapor
    required = BASE_MOISTURE_WRFINPUT
    # 16 belongs with the ice-carrying set: wdm6scheme's moist inventory
    # (Registry.EM_COMMON:3031) is qv,qc,qr,qi,qs,qg -- WSM6's, character
    # for character.  WDM6 is double-moment in the WARM half only.
    from woof.core.column_schemes import COLUMN_SCHEMES
    if mp_physics in COLUMN_SCHEMES:
        names = {"qi": "QICE", "qs": "QSNOW", "qg": "QGRAUP"}
        required += tuple(names[n] for n in COLUMN_SCHEMES[mp_physics].species if n in names)
    if mp_physics in (2, 6, 8, 9, 10, 16, 18, 24, 28, 30, 106):
        required += ICE_MASS_WRFINPUT
    if mp_physics == 30:
        required += _FSBM_BIN_SPECIES + ("QNCCN","QNCLOUD","QNRAIN","QNICE","QNSNOW","QNGRAUPEL","EFF_RADIUS","ICE_EFF_RADIUS","TOT_EFF_RADIUS")
    if mp_physics == 24:
        required += WSM7_MOISTURE_WRFINPUT
    if mp_physics == 9:
        # Milbrandt-Yau: WSM6's six masses (added just above) plus hail
        # and six number moments.  See MILBRANDT_MOISTURE_WRFINPUT.
        required += MILBRANDT_MOISTURE_WRFINPUT
    if mp_physics == 16:
        required += WDM6_NUMBER_WRFINPUT
    if mp_physics == 8:
        required += THOMPSON_NUMBER_WRFINPUT
    elif mp_physics == 28:
        # Thompson aerosol-aware: classic Thompson's two moments plus the
        # prognostic droplet number and the two aerosol tracers.
        required += THOMPSON_AEROSOL_NUMBER_WRFINPUT
    elif mp_physics == 10:
        required += MORRISON_NUMBER_WRFINPUT
    elif mp_physics == 18:
        required += nssl_moisture_wrfinput(cfg)
    elif mp_physics == 50:
        # P3 one-category.  Its whole inventory beyond the three base
        # masses arrives in one tuple because its ice mass does NOT come
        # from ``ICE_MASS_WRFINPUT`` -- see P3_MOISTURE_WRFINPUT above.
        # QSNOW and QGRAUP therefore stay out of ``allowed`` as well as out
        # of ``required``, so a six-species wrfinput handed to a P3 config
        # is refused as carrying inactive moisture rather than restored
        # with two frozen species dropped on the floor.
        required += P3_MOISTURE_WRFINPUT
    elif mp_physics == 5:
        required += FERRIER_MOISTURE_WRFINPUT
    allowed = frozenset(required)
    if mp_physics == 10:
        allowed |= frozenset(MORRISON_OPTIONAL_MOISTURE_WRFINPUT)
    if mp_physics == 5:
        allowed |= frozenset(FERRIER_OPTIONAL_MOISTURE_WRFINPUT)
    return frozenset(required), allowed


def nssl_moisture_wrfinput(cfg) -> tuple[str, ...]:
    """The NSSL names beyond the six masses that ``cfg``'s variant carries.

    WRF allocates, and real.exe writes, only the fields of the packages the
    resolved variant selects (Registry.EM_COMMON:3049-3056 after
    module_check_a_mundo.F:3377-3465).  The deprecated ``mp_physics = 17``
    turns predicted CCN off and ``22`` turns hail and CCN off, so their
    wrfinput carries no QNCCN, and 22's none of QHAIL, QNHAIL or QVHAIL
    either.  Requiring the full option-18 set refused 26 combo-sweep runs
    (2026-10-07, round 2) whose variants have a ported path
    (``PORTED_VARIANT_MODES``); the state pins the absent fields to zero
    (``pin_absent_nssl2_fields``) as WRF's absent packages are.  A ``cfg``
    without the selectors resolves WRF's option-18 defaults, the full set.
    """
    from woof.core.nssl2_contract import (NSSL2_SELECTOR_FIELDS,
                                           resolve_nssl2_mode)
    selectors = {name: getattr(cfg, name) for name in NSSL2_SELECTOR_FIELDS
                 if getattr(cfg, name, None) is not None}
    mode = resolve_nssl2_mode(**selectors)
    by_state = {state: wrf for wrf, state in NSSL_MOISTURE_MAP.items()}
    carried = {by_state[field] for field in mode.transported_fields
               if field in by_state}
    return tuple(name for name in NSSL_MOISTURE_WRFINPUT if name in carried)


def active_moisture_map(cfg) -> Mapping[str, str]:
    """Return the exact WRF-name -> DomainState-name map for ``cfg``."""
    _, allowed = active_moisture_inventory(cfg)
    if cfg is None or int(cfg.mp_physics) != 18:
        candidates = MOISTURE_MAP
    else:
        candidates = NSSL_MOISTURE_MAP
    return MappingProxyType({
        wrf_name: state_name
        for wrf_name, state_name in candidates.items()
        if wrf_name in allowed
    })


def _validate_wrfinput_geometry(name: str, variable,
                                expected_extents: Mapping[str, int],
                                value: np.ndarray) -> None:
    expected_dimensions = WRFINPUT_DIMENSIONS.get(name)
    if expected_dimensions is None:
        raise ConfigurationRefused(f"WRF input variable {name} has no mapped geometry")
    actual_dimensions = tuple(variable.dimensions)
    if actual_dimensions[:1] == ("Time",):
        actual_dimensions = actual_dimensions[1:]
    if name == "FMEP":
        from woof.ingest.wrfinput_sfire import FMEP_DIMENSION_ALIAS
        actual_dimensions = tuple("fuel_moisture_extended_parameters"
                                  if dim == FMEP_DIMENSION_ALIAS else dim
                                  for dim in actual_dimensions)
    if name == "FMC_GC":
        from woof.ingest.wrfinput_sfire import FMC_GC_DIMENSION_ALIAS
        actual_dimensions = tuple("fuel_moisture_classes"
                                  if dim == FMC_GC_DIMENSION_ALIAS else dim
                                  for dim in actual_dimensions)
    try:
        expected_shape = tuple(
            expected_extents[dim] for dim in expected_dimensions)
    except KeyError as exc:
        raise ConfigurationRefused(
            f"pinned WRF geometry has no dimension {exc.args[0]} for "
            f"WRF input {name}") from exc
    if actual_dimensions != expected_dimensions or value.shape != expected_shape:
        raise ConfigurationRefused(
            f"WRF input {name} shape mismatch: expected pinned {expected_shape} "
            f"on {expected_dimensions}, got {value.shape} on {actual_dimensions}")


def read_wrfinput(path: str | Path, *, require_complete: bool = True,
                  expected_dimensions: Mapping[str, int],
                  cfg=None, check_schemes: bool = True, soil_source=None,
                  ) -> RestoredDomain:
    """Read one wrfinput file without importing CuPy.

    ``expected_dimensions`` pins all seven WRF extents the caller expects,
    so an arbitrary file cannot define its own geometry and thereby make a
    truncated but self-consistent handoff look valid.  ``cfg`` selects the
    exact active hydrometeor inventory; ``cfg=None`` retains the historical
    Morrison contract the registered N5S case is scored against.

    ``check_schemes`` reads the file's own ``MP_PHYSICS`` and
    ``SF_SURFACE_PHYSICS`` global attributes and refuses an unported
    package BY NAME before any variable inventory is consulted (see
    :func:`check_supported_schemes`).  It is the front door's default and
    is off only for the registered verification fixtures, which are
    synthesised without global attributes.
    """
    path = Path(path)
    expected_extents = _explicit_wrfinput_dimensions(expected_dimensions)
    expected_extents = {**expected_extents, **fire_input_extents(cfg, expected_extents)}
    required_moisture, allowed_moisture = active_moisture_inventory(cfg)
    # Foreign input: WRF real.exe's own wrfinput, decoded field by
    # field through the Rust bridge. Times is validated by file identity.
    with netcdf_bridge.open_dataset(path) as dataset:
        dimensions = {name: len(dim) for name, dim in dataset.dimensions.items()}
        attrs = {name: dataset.getncattr(name) for name in dataset.ncattrs()}
        from woof.ingest.wrfinput_noahmp import input_dispositions
        from woof.ingest.wrfinput_surface_statics import (
            surface_static_dispositions)
        surface_dispositions = {**input_dispositions(cfg),
                                **surface_static_dispositions(cfg)}
        surface_dispositions.update(urban_static_input_dispositions(
            dataset, path, cfg, expected_dimensions))
        if surface_dispositions or _integral_attribute(attrs.get("SF_SURFACE_PHYSICS")) == 4:
            from woof.ingest.wrfinput_noahmp import require_cold_start
            require_cold_start(dataset)
        recorded_surface_dispositions = {
            name: reason for name, reason in surface_dispositions.items()
            if name in dataset.variables}
        if check_schemes:
            # BEFORE the inventory check below, on purpose.  An unported
            # package's hydrometeor names would otherwise surface as
            # "unmapped WRF variable(s): ['QAOLI', 'QICE2', ...]", which
            # names ten symbols and not the one fact -- the scheme -- that
            # a reader can act on.
            check_supported_schemes(attrs, source=str(path))
        unknown = sorted(
            set(dataset.variables) - ALLOWED_WRFINPUT - IGNORED_WRFINPUT
            - ANALYSIS_PASSTHROUGH_WRFINPUT.keys()
            - surface_dispositions.keys())
        if "fire_smoke" in dataset.variables and not bool(getattr(cfg, "fire_smoke", False)):
            raise ConfigurationRefused("native fire_smoke input requires the active endogenous bulk smoke profile; "
                             "enable fire_smoke to preserve its g_smoke/kg_air volume")
        if unknown:
            # Reached only when the scheme attributes said the package is
            # supported and the file still carries names this door has no
            # state for -- a genuinely unrecognised variable set.  Say
            # which scheme claimed them, so the disagreement is visible.
            claimed = _integral_attribute(attrs.get("MP_PHYSICS"))
            claim = ("" if claimed is None else
                     f" The file declares {_named_microphysics(claimed)},"
                     " whose inventory this door does map, so these names"
                     " are outside every scheme it knows.")
            raise ConfigurationRefused(
                f"{path} has unmapped WRF variable(s): {unknown}.{claim}")
        raw = {}
        soil_conversions = {}
        # A cycled start (cfg.cycling) continues the input's MYNN subgrid
        # cloud: the first radiation call merges QC_BL and CLDFRA_BL and the
        # gsd_41 MYNN first call keeps them (MYNN_CYCLED_WRFINPUT).  Any
        # other start passes them through unread, as before.
        carried = (MYNN_CYCLED_WRFINPUT
                   if cfg is not None and bool(getattr(cfg, "cycling", False))
                   else frozenset()) | OROGRAPHIC_WRFINPUT
        # Stock records the history schema writes straight from wrfinput
        # (grid constants, flags) are carried even where the model has no
        # state for them: IGNORED_WRFINPUT only says nothing is restored.
        carried |= frozenset(WRF_INPUT_HISTORY_ROWS)
        # A nest's start reads it: med_nest_initial blends the parent's
        # terrain into the nest's frame only when the file says 1
        # (woof/ingest/wrfinput_nest_initial.py).
        carried |= {"SAVE_TOPO_FROM_REAL"}
        def unread(name):
            return (name == "Times"
                    or (name in IGNORED_WRFINPUT and name not in carried)
                    or (name in ANALYSIS_PASSTHROUGH_WRFINPUT
                        and name not in carried)
                    or name in surface_dispositions)

        def layer_water(name, variable):
            units = str(getattr(variable, "units", "")).strip().lower()
            compact_units = "".join(character for character in units
                                    if not character.isspace() and character not in "^*()")
            return (name in ("SMOIS", "SH2O")
                    and compact_units in ("kgm-2", "kg/m2", "mm", "m"))

        decode = netcdf_bridge.BatchedDecode(dataset, [
            name for name, variable in dataset.variables.items()
            if not unread(name) and not layer_water(name, variable)])
        for name, variable in dataset.variables.items():
            if unread(name):
                continue
            decode.ready(name)
            if layer_water(name, variable):
                decoded, conversion = variable.read_layer_water("DZS")
                value = _read_numeric(variable, decoded=decoded)
                soil_conversions[name] = conversion
            else:
                value = _read_numeric(variable)
            _validate_wrfinput_geometry(
                name, variable, expected_extents, value)
            raw[name] = value
        require_zero_base_state_profiles(dataset, path, cfg)
    if int(getattr(cfg, "ifire", 0)) == 2:
        # LFN_TIME is declared as input but is never read by WRF's SFIRE
        # driver/model. Keep its native bytes and disclose that disposition.
        if "LFN_TIME" in raw:
            recorded_surface_dispositions["LFN_TIME"] = (
                "Registry input retained; WRF v4.7.1 SFIRE does not read it; "
                "ignition timing comes from &fire ignition_start_time")
    factor_resolution = resolve_map_factors(raw)
    if factor_resolution:
        import json
        attrs["ARWEN_MAP_FACTOR_RESOLUTION"] = json.dumps(factor_resolution, sort_keys=True)
    soil_recovery = {}
    if "SMOIS" not in soil_conversions:
        from woof.ingest.wrf_soil_recovery import recover_supplied_soil
        recovered, recovery = recover_supplied_soil(
            path, raw, attrs, source_directory=soil_source)
        raw.update(recovered)
        soil_recovery = recovery or {}
    _validate_supplied_physics_fields(raw, cfg, attrs)
    if require_complete:
        wrfinput_terrain_drag_inputs(SimpleNamespace(raw=raw), cfg)
    if "QNBCA" in raw and int(getattr(cfg, "wif_input_opt", 0)) == 2:
        # The fact and the way out come from the table that owns this
        # selector, so this door and the namelist importer cannot describe
        # one configuration differently.  Imported on call: this module
        # keeps its module-scope imports to numpy and the netCDF bridge.
        from woof.config import MP28_AEROSOL_SOURCE_OPTIONS

        _, _, why = MP28_AEROSOL_SOURCE_OPTIONS["wif_input_opt"]
        raise NotImplementedError(
            "QNBCA is supplied with wif_input_opt=2, but the black-carbon "
            f"state/physics consumer is not implemented: {why}. Set "
            "wif_input_opt=1 with aer_init_opt=1 for the ported monthly "
            "climatology, or remove QNBCA from the supplied wrfinput.")
    present_moisture = set(raw) & ALL_MOISTURE_WRFINPUT
    extra_moisture = sorted(present_moisture - allowed_moisture)
    if extra_moisture:
        raise ConfigurationRefused(
            f"{path} has inactive WRF moisture variable(s) for the active "
            f"physics: {extra_moisture}")
    if require_complete:
        missing = missing_required_wrfinput(raw, cfg)
        if missing:
            raise ConfigurationRefused(f"{path} is missing mapped WRF variable(s): {missing}")
    mapped = MAPPED_WRFINPUT & set(raw)
    auxiliary = (set(EXPLICIT_AUXILIARY_WRFINPUT) - {"Times"}) & set(raw)
    return RestoredDomain(
        path=path, raw=MappingProxyType(raw),
        dimensions=MappingProxyType(dimensions),
        global_attributes=MappingProxyType(attrs),
        mapped_variables=tuple(sorted(mapped)),
        auxiliary_variables=tuple(sorted(auxiliary)),
        surface_input_dispositions=MappingProxyType(recorded_surface_dispositions),
        soil_unit_conversions=MappingProxyType(soil_conversions),
        soil_recovery=MappingProxyType(soil_recovery))


def missing_required_wrfinput(raw: Mapping[str, np.ndarray], cfg) -> list[str]:
    """The names a forecast start needs that ``raw`` does not carry.

    The completeness half of :func:`read_wrfinput`, on its own so a caller
    that reads a file without starting from it (an analysis exchange, an
    inventory) can still say what a start would lack.  ``cfg`` selects the
    moisture inventory as it does for the reader.
    """
    required_moisture, _ = active_moisture_inventory(cfg)
    present_moisture = set(raw) & ALL_MOISTURE_WRFINPUT
    non_moisture_required = set(REQUIRED_WRFINPUT) - ALL_MOISTURE_WRFINPUT
    missing = sorted(name for name in non_moisture_required if name not in raw)
    missing.extend(sorted(required_moisture - present_moisture))
    if int(getattr(cfg, "mp_physics", 0)) == 28:
        missing.extend(name for name in ("QNWFA2D", "QNIFA2D")
                       if name not in raw)
    missing.extend(sorted(
        name for name, alternatives in ALIASES.items()
        if not any(alias in raw for alias in alternatives)))
    return missing
def resolve_map_factors(raw):
    """Select positive native directional aliases when legacy factors are zero.

    The compact ARW state has one isotropic factor per grid location. Equal
    X/Y factors are exact aliases; unequal factors need separate operators.
    No map-factor field arithmetic or inferred geometry is performed here.
    """
    import hashlib
    receipt = {}
    for point in ("M", "U", "V"):
        name = f"MAPFAC_{point}"
        legacy = raw.get(name)
        if legacy is None or bool(np.all(np.isfinite(legacy) & (legacy > 0))):
            continue
        x, y = raw.get(name + "X"), raw.get(name + "Y")
        if x is None or y is None:
            raise ConfigurationRefused(f"{name} is nonpositive and has no supplied directional factors; grid metric divisions would be invalid")
        if x.shape != legacy.shape or y.shape != legacy.shape or not bool(np.all(np.isfinite(x) & (x > 0))):
            raise ConfigurationRefused(f"{name} directional factor shape or values cannot define positive native grid metrics")
        if not np.array_equal(x, y):
            raise ConfigurationRefused(f"{name} directional X/Y factors differ; collapsing anisotropic metrics would change flux divergence")
        receipt[name] = dict(source=name + "X", equivalent=name + "Y",
            original_sha256=hashlib.sha256(legacy.tobytes()).hexdigest(),
            selected_sha256=hashlib.sha256(x.tobytes()).hexdigest())
        raw[name] = x
    return receipt


def _validate_supplied_physics_fields(raw, cfg, attributes):
    """Every newly admitted physics field has a selected, named consumer."""
    surface = (getattr(cfg, "sf_surface_physics", None) if cfg is not None
               else _integral_attribute(attributes.get("SF_SURFACE_PHYSICS")))
    pbl = (getattr(cfg, "bl_pbl_physics", None) if cfg is not None
           else _integral_attribute(attributes.get("BL_PBL_PHYSICS")))
    ruc = RUC_INPUT_FIELDS & raw.keys()
    if ruc and surface != 3:
        raise ConfigurationRefused(
            f"WRF physics input {sorted(ruc)} requires the RUC "
            "sf_surface_physics=3 consumer")
    mynn = (MYNN_QKE_INPUT_FIELDS | MYNN_SCALAR_WRFINPUT
            | MYNN_CYCLED_WRFINPUT) & raw.keys()
    if mynn and pbl != 5:
        raise ConfigurationRefused(
            f"WRF physics input {sorted(mynn)} requires the MYNN "
            "bl_pbl_physics=5 consumer")
    if "qke" in raw and "QKE" in raw and not np.array_equal(raw["qke"], raw["QKE"]):
        raise ConfigurationRefused("WRF physics input qke and QKE contain conflicting aliases")

def _expect_shape(name: str, value: np.ndarray, expected: tuple[int, ...]) -> None:
    if value.shape != expected:
        raise ValueError(f"WRF {name} shape {value.shape} != expected {expected}")


def _first(raw: Mapping[str, np.ndarray], names: Sequence[str], *,
           required: bool = True):
    for name in names:
        if name in raw:
            return raw[name]
    if required:
        raise ValueError(f"WRF input is missing every alias in {tuple(names)}")
    return None


def _restore_active_moisture(state, raw: Mapping[str, np.ndarray], cfg,
                             array_module, *, warm: bool = False) -> None:
    """Restore the scheme-native moisture fields and their RK ``*0`` copies.

    ``warm`` is a WRF restart: the species are the producing run's own
    evolved state, so no cold-start replacement (WDM6's CCN fill) applies.
    """
    # The QNCLOUD exemption is Morrison's alone (WRF diagnoses cloud number
    # there).  For mp=28 QNCLOUD is the prognostic droplet number the whole
    # scheme turns on, and a wrfinput without it must fail here rather than
    # start every column at WRF's 2/rho terminal clamp floor
    # (module_mp_thompson.F:3976), which is finite, bounded and wrong.
    optional_moisture = frozenset(MORRISON_OPTIONAL_MOISTURE_WRFINPUT)
    if (cfg is not None
            and int(cfg.mp_physics) in REQUIRED_QNCLOUD_MICROPHYSICS):
        optional_moisture = frozenset()
    # Surface emission tendencies are inputs too, and must be installed
    # before microphysics_init tests whether to fill a synthetic profile.
    for wrf_name, state_name in (("QNWFA2D", "nwfa2d"), ("QNIFA2D", "nifa2d")):
        target = getattr(state, state_name, None)
        if target is not None and wrf_name in raw:
            _expect_shape(wrf_name, raw[wrf_name], target.shape)
            target[...] = array_module.asarray(
                raw[wrf_name], dtype=array_module.float32)
    state_names = []
    for wrf_name, state_name in active_moisture_map(cfg).items():
        target = getattr(state, state_name, None)
        if target is None:
            raise ValueError(
                f"DomainState lacks active mp_physics={cfg.mp_physics} "
                f"field {state_name} for WRF {wrf_name}")
        state_names.append(state_name)
        if wrf_name in raw:
            _expect_shape(wrf_name, raw[wrf_name], target.shape)
            target[...] = array_module.asarray(
                raw[wrf_name], dtype=array_module.float32)
        elif wrf_name not in optional_moisture:
            raise ValueError(
                f"WRF input lacks active mp_physics={cfg.mp_physics} "
                f"field {wrf_name}")

    if cfg is not None and int(cfg.mp_physics) == 16 and not warm:
        # WRF module_mp_wdm6.F:220-227 replaces the complete cold-start
        # CCN reservoir with ccn_conc on its first step, even when wrfinput
        # carries a nonzero QNCCN. Preserve that initialization after the
        # file restore, before saving the RK-beginning copies. Checkpoint
        # restoration has its separate transport and does not enter here.
        state.nn[...] = array_module.float32(cfg.wdm6_ccn_conc)

    # DomainState owns RK-beginning copies only for prognostic fields.  Sync
    # every active one that exists, including all ten NSSL-only fields.
    for state_name in dict.fromkeys(state_names):
        source = getattr(state, state_name)
        initial = getattr(state, f"{state_name}0", None)
        if initial is not None:
            initial[...] = source


def wrf_coordinate_and_base(restored, hypsometric_opt=None, *, cold_start=None):
    """The file's vertical coordinate and WRF's cold-start base state.

    The coordinate is the file's.  The base state is the one WRF's
    ``start_domain_em`` rebuilds before the first step of a run that is
    not a restart (:func:`wrf_start_base_state`), not the file's words.
    """
    from woof.core.grid import VerticalCoord, BaseState

    raw = restored.raw
    names = ('znw', 'znu', 'dnw', 'rdnw', 'dn', 'rdn', 'fnp', 'fnm',
             'c1f', 'c2f', 'c3f', 'c4f', 'c1h', 'c2h', 'c3h', 'c4h')
    top = float(np.asarray(raw['P_TOP']).item())
    coord = VerticalCoord(**{name: raw[name.upper()] for name in names},
                          hybrid_opt=int(restored.global_attributes['HYBRID_OPT']),
                          etac=float(restored.global_attributes['ETAC']), p_top=top)
    if hypsometric_opt is None:
        hypsometric_opt = int(restored.global_attributes.get(
            'HYPSOMETRIC_OPT', 2))
    if getattr(restored, 'warm_start', None) is not None:
        # WRF's restart branch: start_em.F rebuilds the base state only for
        # a run that is not a restart (the :554-636 block); a restart keeps
        # the words the file carries, which the producing run integrated on.
        start = {name: np.asarray(raw[name], np.float32)
                 for name in ('MUB', 'PB', 'T_INIT', 'ALB', 'PHB')}
    else:
        start = (wrf_start_base_state(raw, int(hypsometric_opt))
                 if cold_start is None else cold_start)
    # WRF writes T_INIT minus t0.  DomainState.thb is absolute base
    # potential temperature.
    theta_base = start['T_INIT'] + np.float32(300.0)
    base = BaseState(start['MUB'], top, start['PB'], start['ALB'],
                     theta_base, start['PHB'], raw['HGT'])
    return coord, base


def _glibc_expf_array(x):
    """glibc 2.39 ``expf`` over a float32 array, the same bits.

    glibc's expf is not correctly rounded, but its error stays within
    0.502 ULP, so it can differ from the once-rounded binary64 exponential
    only where the exact value lies within 0.002 ULP of a rounding
    midpoint.  Those few elements (0.06 % of a terrain field) take the
    scalar transcription; every other element is the binary64 result
    rounded once, which is then glibc's.
    """
    from woof.core.noahmp_libm import expf

    x = np.asarray(x, np.float32)
    exact = np.exp(x.astype(np.float64))
    out = exact.astype(np.float32)
    wide = out.astype(np.float64)
    ulp = np.abs(np.nextafter(out, np.float32(np.inf)).astype(np.float64)
                 - wide)
    near = np.abs(np.abs(exact - wide) - 0.5 * ulp) < 0.01 * ulp
    for index in zip(*np.nonzero(near)):
        out[index] = expf(x[index])
    return out


def wrf_start_base_state(raw, hypsometric_opt: int):
    """MUB, PB, T_INIT, ALB and PHB as WRF v4.6.1's cold start rebuilds them.

    ``start_em.F:554-636`` (input_from_file, not a restart) discards the
    file's base state and recomputes it from the terrain and the file's
    reference-profile constants P00/T00/TLP/TISO/TLP_STRAT/P_STRAT::

        p_surf = p00*EXP(-t00/a + ((t00/a)**2 - 2.*g*ht/a/r_d)**0.5)
        pb     = c3h*(p_surf - p_top) + c4h + p_top
        temp   = MAX(tiso, t00 + a*LOG(pb/p00))
                 (tiso + a_strat*LOG(pb/p_strat) where pb < p_strat)
        t_init = temp*(p00/pb)**(r_d/cp) - t0
        alb    = (r_d/p1000mb)*(t_init+t0)*(pb/p1000mb)**cvpm
        mub    = p_surf - p_top
        phb    = ht*g, then the hypsometric_opt integration upward

    and ``start_em.F:670-680`` (max_dom = 1) re-forms pb and alb from that
    mub and t_init, which is the same arithmetic.  Every pressure-gradient
    and EOS evaluation of the run reads these words.  An exported pair can
    carry its producer's own base state: on the stock-WRF export of the
    g400 crop the file's ALB sat up to 4.5 m3/kg from WRF's formula at the
    model top and its MUB/PHB 7 and 20 float32 ULPs off; the door used
    the file's words.

    FP32 in Fortran's operation order (the reference gfortran build does
    not contract), with glibc 2.39's logf, powf and expf for LOG, ``**``
    and EXP.
    """
    from woof.core.noahmp_libm import logf_array, powf_array

    f32 = np.float32

    def scalar(name):
        if name not in raw:
            raise ValueError(
                f"WRF input lacks the base-state constant {name}; WRF's "
                "start_em stops on the same file (no base state "
                "parameters in wrfinput)")
        return f32(np.asarray(raw[name], np.float32).reshape(-1)[0])

    p00, t00, a = scalar('P00'), scalar('T00'), scalar('TLP')
    tiso, a_strat, p_strat = (scalar('TISO'), scalar('TLP_STRAT'),
                              scalar('P_STRAT'))
    if t00 < f32(100.0) or p00 < f32(10000.0):
        raise ValueError(
            "WRF input base-state constants T00/P00 are unset; WRF's "
            "start_em stops on the same file")
    g, r_d, t0, p1000mb = f32(9.81), f32(287.0), f32(300.0), f32(100000.0)
    cp_ = f32(f32(f32(7.0) * r_d) / f32(2.0))
    cvpm = f32(f32(-f32(cp_ - r_d)) / cp_)
    rcp = f32(r_d / cp_)
    p_top = scalar('P_TOP')

    def col(name):
        return np.asarray(raw[name], np.float32).reshape(-1)[:, None, None]

    c3h, c4h, c3f, c4f = col('C3H'), col('C4H'), col('C3F'), col('C4F')
    ht = np.asarray(raw['HGT'], np.float32)
    ratio = f32(t00 / a)
    term = np.asarray(f32(f32(2.0) * g) * ht, np.float32)
    term = np.asarray(np.asarray(term / a, np.float32) / r_d, np.float32)
    radicand = np.asarray(f32(ratio * ratio) - term, np.float32)
    exponent = np.asarray(f32(-ratio) + powf_array(radicand, f32(0.5)),
                          np.float32)
    p_surf = np.asarray(p00 * _glibc_expf_array(exponent), np.float32)
    mub = np.asarray(p_surf - p_top, np.float32)

    def level(c3, c4):
        return np.asarray(np.asarray(np.asarray(c3 * mub, np.float32) + c4,
                                     np.float32) + p_top, np.float32)

    pb = level(c3h, c4h)
    temp = np.maximum(tiso, np.asarray(
        t00 + np.asarray(a * logf_array(np.asarray(pb / p00, np.float32)),
                         np.float32), np.float32))
    strat = pb < p_strat
    if np.any(strat):
        upper = np.asarray(tiso + np.asarray(a_strat * logf_array(np.asarray(
            np.where(strat, pb / p_strat, f32(1.0)), np.float32)),
            np.float32), np.float32)
        temp = np.where(strat, upper, temp)
    t_init = np.asarray(
        np.asarray(temp * powf_array(np.asarray(p00 / pb, np.float32), rcp),
                   np.float32) - t0, np.float32)
    lead = np.asarray(f32(r_d / p1000mb) * np.asarray(t_init + t0, np.float32),
                      np.float32)
    alb = np.asarray(
        lead * powf_array(np.asarray(pb / p1000mb, np.float32), cvpm),
        np.float32)
    nz = pb.shape[0]
    phb = np.empty((nz + 1,) + mub.shape, np.float32)
    phb[0] = np.asarray(ht * g, np.float32)
    if hypsometric_opt == 2:
        for k in range(1, nz + 1):
            pfu = level(c3f[k], c4f[k])
            pfd = level(c3f[k - 1], c4f[k - 1])
            phm = level(c3h[k - 1], c4h[k - 1])
            thick = np.asarray(
                np.asarray(alb[k - 1] * phm, np.float32)
                * logf_array(np.asarray(pfd / pfu, np.float32)), np.float32)
            phb[k] = np.asarray(phb[k - 1] + thick, np.float32)
    elif hypsometric_opt == 1:
        dnw = np.asarray(raw['DNW'], np.float32).reshape(-1)
        c1h = np.asarray(raw['C1H'], np.float32).reshape(-1)
        c2h = np.asarray(raw['C2H'], np.float32).reshape(-1)
        for k in range(1, nz + 1):
            mass = np.asarray(np.asarray(c1h[k - 1] * mub, np.float32)
                              + c2h[k - 1], np.float32)
            phb[k] = np.asarray(phb[k - 1] - np.asarray(
                np.asarray(dnw[k - 1] * mass, np.float32) * alb[k - 1],
                np.float32), np.float32)
    else:
        raise ValueError(
            f"hypsometric_opt must be 1 or 2, got {hypsometric_opt}")
    return {'MUB': mub, 'PB': pb, 'T_INIT': t_init, 'ALB': alb, 'PHB': phb}


def restore_domain_state(restored: RestoredDomain, cfg, *, scratch_arena=None,
                         dycore_state_workspace=None, radiation=None,
                         radiation_start_time=None, radiation_latitude=None,
                         radiation_longitude=None):
    """Upload one CPU-restored domain into a production ``DomainState``."""
    import cupy as cp
    from woof.core.state import DomainState

    raw = restored.raw
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    expected = {
        "U": (nz, ny, nx + 1), "V": (nz, ny + 1, nx),
        "W": (nz + 1, ny, nx), "T": (nz, ny, nx),
        "PH": (nz + 1, ny, nx), "MU": (ny, nx),
        "PHB": (nz + 1, ny, nx), "MUB": (ny, nx),
        "T_INIT": (nz, ny, nx), "P": (nz, ny, nx),
        "PB": (nz, ny, nx), "AL": (nz, ny, nx),
        "ALB": (nz, ny, nx),
    }
    for name, shape in expected.items():
        _expect_shape(name, raw[name], shape)
    state_kwargs = {}
    if scratch_arena is not None:
        state_kwargs["scratch_arena"] = scratch_arena
    if dycore_state_workspace is not None:
        state_kwargs["dycore_state_workspace"] = dycore_state_workspace
    # Admitted before the constructor: a wrfinput too big for the card
    # stopped in a CUDA out-of-memory inside DomainState instead of being
    # refused by name.  A shared dycore workspace hands the constructor its
    # rebuilt arrays as views, so those are not charged twice.
    from woof.core.resident_admission import admit_construction

    admit_construction(
        "restoring this wrfinput onto the card", cfg,
        shared_symbols=getattr(dycore_state_workspace, "_symbol_shapes", ()))
    state = DomainState(cfg, **state_kwargs)
    cold_start = wrf_start_base_state(
        raw, int(getattr(cfg, 'hypsometric_opt', 2)))
    coord, base = wrf_coordinate_and_base(
        restored, int(getattr(cfg, 'hypsometric_opt', 2)), cold_start=cold_start)
    state.load_base(coord, base)

    for wrf_name, state_name in (("U", "u"), ("V", "v"), ("W", "w"),
                                 ("PH", "php"), ("MU", "mup")):
        getattr(state, state_name)[...] = cp.asarray(raw[wrf_name], dtype=cp.float32)
    from woof.wrf_exact import ENABLED as wrf_exact
    if wrf_exact:
        # Keep the native perturbation word throughout integration. A round
        # trip through full theta cannot recover its low bits.
        state.thb[...] = cp.float32(300.0)
        state.thp[...] = cp.asarray(raw["T"], dtype=cp.float32)
        if cfg.diff_opt == 1:
            # horizontal_diffusion_3dmp uses start_em's base thermal
            # reference, not the full initial atmospheric theta. Keep the
            # original T_INIT words before thb becomes the 300 K RK basis.
            state.scratch(state.thp.shape, "diff1_theta_initial")[...] = (
                cp.asarray(cold_start["T_INIT"], dtype=cp.float32))
    else:
        state.thp[...] = cp.asarray(
            (raw["T"].astype(np.float32) + np.float32(300.0))
            - base.thb, dtype=cp.float32)
    state.p[...] = cp.asarray(raw["P"] + raw["PB"], dtype=cp.float32)
    from woof.wrf_exact import DIAGNOSTICS_ENABLED
    if DIAGNOSTICS_ENABLED:
        state.p_perturbation = cp.asarray(raw["P"], dtype=cp.float32)
    state.al[...] = cp.asarray(raw["AL"], dtype=cp.float32)
    state.alt[...] = cp.asarray(raw["AL"] + raw["ALB"], dtype=cp.float32)
    for name in ("CF1", "CF2", "CF3"):
        source = np.asarray(raw[name]).reshape(-1)
        _expect_shape(name, source, (1,))
        setattr(state, name.lower(), cp.float32(source[0]))
    # wrf.exe integrates on the file's top extrapolation pair, not on one
    # rebuilt from FNP (start_em never recomputes it).  A file whose pair is
    # not real.exe's formula of its own DN/DNW (the direct exporter wrote a
    # template's pair before 2026-10-29) moved every top-level ph tendency:
    # the UAE 9 km d01 differed from wrf.exe 4.6.1 in 495,278 U words at
    # step 1 with every physics scheme off.  read_wrfinput carries the pair
    # as input-history records; a file without it keeps the FNP-derived one
    # load_base installed.
    for name in ("CFN", "CFN1"):
        if name in raw:
            source = np.asarray(raw[name]).reshape(-1)
            _expect_shape(name, source, (1,))
            setattr(state, name.lower(), cp.float32(source[0]))
    state.set_map_coriolis(
        raw["MAPFAC_M"], raw["MAPFAC_U"], raw["MAPFAC_V"], raw["F"],
        raw["E"], sina=raw["SINALPHA"], cosa=raw["COSALPHA"],
        msfv_inv=raw.get("MF_VX_INV"))
    warm = getattr(restored, "warm_start", None) is not None
    _restore_active_moisture(state, raw, cfg, cp, warm=warm)
    if "H_DIABATIC" in raw:
        state.h_diabatic[...] = cp.asarray(raw["H_DIABATIC"], dtype=cp.float32)
    # WRF's cold start does not keep the file's P, AL or ALT either:
    # start_em.F:774-842 re-derives AL from the file's PH, MU and the
    # rebuilt base (calc_p_rho_phi's hypsometric form) and P from the EOS
    # on T and QV, and the first step's pressure force and every physics
    # scheme read that P.  The door kept the file's P; on the stock-WRF
    # export (g400 crop, 2024-05-21 18Z) WRF's start P sat up to 324 Pa
    # from it (9.6 Pa RMS at the lowest level), and the first surface-layer
    # call followed that difference (HFX 0.22 W/m2 RMS, 77 times WRF's
    # ULP twin).  The runtime's own EOS diagnostic, so the start P is the
    # one every later stage diagnoses from the same prognostics.
    from woof.core.diagnostics import update_diagnostics
    update_diagnostics(state, int(cfg.hypsometric_opt))

    # WRF start_domain_em diagnoses a cold-start W column after reading
    # real.exe's zero W, before its first history frame. The native door
    # admits the WRF default use_input_w=False; an explicit use_input_w
    # namelist key has no translation in this door.
    # A WRF restart carries the producing run's own W (W_2), which wrf.exe
    # reads and integrates on; only a real.exe cold start diagnoses it.
    if not warm:
        from woof.ingest.wrfinput_startup import initialize_wrfinput_vertical_velocity
        initialize_wrfinput_vertical_velocity(
            state, restored, cfg,
            periodic_x=not (cfg.open_x or cfg.specified or cfg.nested),
            periodic_y=not (cfg.open_y or cfg.specified or cfg.nested))

    for current, initial in (
            ("u", "u0"), ("v", "v0"), ("w", "w0"),
            ("thp", "thp0"), ("php", "php0"), ("mup", "mup0")):
        source = getattr(state, current, None)
        target = getattr(state, initial, None)
        if source is not None and target is not None:
            target[...] = source

    if "fire_smoke" in raw:
        from woof.core.chem_sfire import import_native
        import_native(state, raw["fire_smoke"])
    return state


def wrfinput_terrain_drag_inputs(restored, cfg):
    """Use the native file's active orography, without rebuilding its values."""
    names = required_static_fields(getattr(cfg, "topo_wind", 0),
                                   getattr(cfg, "gwd_opt", 0))
    if not names:
        return {}
    missing = sorted(set((*names, "HGT")) - restored.raw.keys())
    if missing:
        raise ValueError(f"WRF input lacks terrain-drag fields {missing}; "
                         "the active drag would otherwise read absent statistics. "
                         "Run geogrid and real.exe with the selected orographic datasets.")
    return {"terrain_drag_static": {"HGT_M": restored.raw["HGT"],
                                    **{name: restored.raw[name] for name in names}}}


def initialize_wrfinput_physics(state, restored, cfg, *, radiation=None,
                               radiation_start_time=None, radiation_latitude=None,
                               radiation_longitude=None, landuse=None,
                               constant_glw_wm2=None, cam_ozone=None,
                               fractional_seaice=False, hydro_parent_geometry=None):
    """Initialize physics from the file's actual surface and land categories.

    ``fractional_seaice`` is the run namelist's (&physics, WRF default 0,
    not a wrfinput header); only the Noah mosaic tile door reads it.
    ``hydro_parent_geometry`` is the run's ProjectedGrid, reconstituted from
    the wrfinput's own projection attributes (woof.wrfinput_door); the
    coupled hydro exchange binds to it.
    """
    _validate_supplied_physics_fields(restored.raw, cfg, restored.global_attributes)
    from woof.core.physics import physics_driver_required
    if not physics_driver_required(cfg) and cam_ozone is None:
        # Every scheme off is a run stock WRF makes (a dynamics-only
        # replay); there is no driver to attach, as on the idealised doors
        # (woof/verify/cases/nest_ideal_common.py).  initialize_physics
        # refuses a driverless call, which used to stop this door.
        return None
    import cupy as cp
    from woof.core.physics import initialize_physics
    from woof.ingest.ruc_mosaic import wrfinput_ruc_mosaic_inputs
    from woof.ingest.lake_physics import wrfinput_lake_physics_inputs
    from woof.surface_ocean import ocean_input_fields

    raw = restored.raw
    xice = _first(raw, ALIASES["XICE"])
    albbck = _first(raw, ALIASES["ALBBCK"])
    lai = _first(raw, ALIASES["LAI"])
    swdown = _first(raw, ("SWDOWN",), required=False)
    glw = (constant_glw_wm2 if constant_glw_wm2 is not None
           else _first(raw, ("GLW",), required=False))
    pblh = _first(raw, ("PBLH",), required=False)
    # real.exe leaves soil columns 0.0 at water points (WRF never reads
    # them there; it uses SST/TSK).  woof's health gate bounds TSLB
    # globally, so fill water columns with TSK â€” the same values WRF's
    # own surface init uses over water â€” before restoration.
    landmask = np.asarray(raw["LANDMASK"])
    tslb = np.array(raw["TSLB"], copy=True)
    water = landmask < 0.5
    if np.any(water):
        tslb[:, water] = np.broadcast_to(
            np.asarray(raw["TSK"])[water], tslb[:, water].shape)
    raw = {**raw, "TSLB": tslb}  # alias loop below re-reads raw (immutable)
    from woof.ingest.wrfinput_sfire import fire_static_fields
    driver = initialize_physics(
        state, cfg, landmask=raw["LANDMASK"], tsk=raw["TSK"],
        landuse=landuse, xland=raw.get("XLAND"),
        landuse_dataset=str(restored.global_attributes["MMINLU"]),
        sst=raw.get("SST", raw["TSK"]),
        soil_temperature=tslb, soil_moisture=raw["SMOIS"],
        liquid_moisture=raw["SH2O"], ivgtyp=raw["LU_INDEX"],
        isltyp=raw["ISLTYP"], vegfra=raw["VEGFRA"], tmn=raw["TMN"],
        xice=xice, snow=raw["SNOW"], snow_depth=raw["SNOWH"],
        swdown=(0.0 if swdown is None else swdown),
        # These diagnostics are not wrfinput fields for this Registry.  WRF
        # v4.6.1 phy_init initializes all three to zero before the first
        # radiation/PBL calls, so preserve that start-of-run convention.
        glw=glw,
        pblh=(0.0 if pblh is None else pblh),
        lightning_inputs={key.lower(): raw[key] for key in ("ICCG_IN_NUM", "ICCG_IN_DEN") if key in raw},
        radiation=radiation,
        radiation_start_time=radiation_start_time,
        radiation_latitude=radiation_latitude,
        radiation_longitude=radiation_longitude,
        **wrfinput_terrain_drag_inputs(restored, cfg),
        **wrfinput_ruc_mosaic_inputs(restored, cfg),
        **wrfinput_lake_physics_inputs(restored, cfg),
        **ocean_input_fields(cfg, raw),
        **({"fire_static_data": fire_static_fields(restored, cfg)}
           if int(getattr(cfg, "ifire", 0)) == 2 else {}),
        # The file's own urban fraction reaches urban_var_init, which keeps
        # a value in (0, 1] and takes the table's otherwise
        # (module_sf_urban.F:2767-2777).  Passed only to an urban run.
        **({"frc_urb2d": raw["FRC_URB2D"]}
           if int(getattr(cfg, "sf_urban_physics", 0)) > 0
           and raw.get("FRC_URB2D") is not None else {}),
        **({"cam_ozone": cam_ozone} if cam_ozone is not None else {}),
        # Breakage it prevents: on any domain the global hydro row covers,
        # `woof run --wrfinput` exited 2 with "parent runoff exchange needs
        # the actual forecast ProjectedGrid".
        hydro_parent_geometry=hydro_parent_geometry)
    if (int(getattr(cfg, "sf_urban_physics", 0)) > 0
            and "utype_urb2d" in driver.fields):
        from woof.ingest.wrfinput_surface_statics import (
            require_table_urban_morphology)
        require_table_urban_morphology(
            restored, cp.asnumpy(driver.fields["utype_urb2d"]))
    # landuse_init resets Noah's background albedo to the seasonal table
    # when USEMONALB is false (WRF module_physics_init.F:1958). The native
    # file can also contain a distinct monthly ALBBCK. Restoring that over
    # the initialized carrier undoes the cold-start authority.
    retain_landuse_albbck = (int(cfg.sf_surface_physics) in (2, 3)
                            and landuse is not None and not cfg.usemonalb)
    from woof.ingest.wrfinput_noahmp import NOAHMP_INITIALIZED_SURFACE_FIELDS
    # A WRF restart's carriers are the producing run's own and are applied
    # after this initialization, every one with its shape and name rules
    # (woof.ingest.wrfrst.apply_warm_start_physics); the cold-start alias
    # pass would read a restart's staggered PBL carriers as surface inputs.
    warm = getattr(restored, "warm_start", None) is not None
    for field in (() if warm else driver.fields):
        if (int(cfg.sf_surface_physics) == 3
                and field in ("sh2o", "smfr3d", "mavail", "znt")):
            # RUCLSMINIT owns these cold-start outputs. real.exe's file
            # carriers precede physics initialization and must not replace
            # its liquid/frozen partition or water/ice values in history.
            continue
        if field == "albbck" and retain_landuse_albbck:
            continue
        if (int(cfg.sf_surface_physics) == 4
                and field in NOAHMP_INITIALIZED_SURFACE_FIELDS):
            continue
        if field == "glw" and constant_glw_wm2 is not None:
            continue
        aliases = PHYSICS_FIELD_ALIASES.get(field, (field.upper(),))
        value = _first(raw, aliases, required=False)
        if value is not None:
            if value.shape != driver.fields[field].shape:
                raise ValueError(
                    f"WRF physics field {aliases[0]} shape {value.shape} != "
                    f"woof {field} shape {driver.fields[field].shape}")
            driver.fields[field][...] = cp.asarray(
                value, dtype=driver.fields[field].dtype)
    if cfg.bl_pbl_physics == 5 and cfg.bl_mynn_tkeadvect and "qke_adv" not in raw:
        state.qke_adv[...] = driver.fields["qke"]
    if not retain_landuse_albbck:
        driver.fields["albbck"][...] = cp.asarray(albbck, dtype=cp.float32)
    if int(cfg.sf_surface_physics) != 4:
        driver.fields["lai"][...] = cp.asarray(lai, dtype=cp.float32)
    if "RAINNC" in raw:
        driver.microphysics.rainnc[...] = cp.asarray(raw["RAINNC"], dtype=cp.float32)
    if driver.rainc is not None and "RAINC" in raw:
        driver.rainc[...] = cp.asarray(raw["RAINC"], dtype=cp.float32)
    if (int(cfg.sf_surface_physics) == 2
            and getattr(restored, "warm_start", None) is None):
        # LSMINIT follows native cold-start field restoration. A later
        # checkpoint load replaces this initial state, including SH2O.
        # A WRF restart is that later load: lsminit skips the partition
        # under restart (module_sf_noahdrv.F's .NOT.restart branch), and
        # the file's SH2O is the producing run's own.
        # Native inputs already contain fractional SNOALB. WRF retains it
        # when RDMAXALB is true, but otherwise replaces it from MAXALB.
        if not cfg.rdmaxalb:
            from woof.core.noah import VEG_COLS
            from woof.static.rust_bridge import wrf_noah_snow_albedo
            driver.fields["snoalb"][...] = cp.asarray(wrf_noah_snow_albedo(
                driver.noah_params.veg[:, VEG_COLS.index("maxalb")],
                raw.get("IVGTYP", raw["LU_INDEX"])), dtype=cp.float32)
        from woof.core.noah import initialize_noah_liquid_water
        from woof.core.physics import NOAH_LAYER_THICKNESS_M
        initialize_noah_liquid_water(driver.fields, driver.noah_params,
                                    NOAH_LAYER_THICKNESS_M)
    elif int(cfg.sf_surface_physics) == 3:
        # RUCLSMINIT owns the initial liquid/frozen partition, availability
        # and roughness. Native wrfinput placeholders restored above must
        # precede this cold start, just as LSMINIT does for Noah. A later
        # checkpoint restore replaces the resulting prognostic state.
        from woof.core.ruc_runtime import ruc_cold_start
        ruc_cold_start(driver.fields, params=driver.ruc_params)
    if getattr(cfg, "sf_surface_mosaic", 0) == 1:
        from woof.core.noah_mosaic_door import attach_wrfinput_noah_mosaic
        attach_wrfinput_noah_mosaic(driver, cfg, restored,
                                    fractional_seaice=fractional_seaice)
    return driver

def _decode_times(variable) -> tuple[datetime, ...]:
    data = np.ma.asarray(variable[...])
    if np.ma.isMaskedArray(data) and np.any(np.ma.getmaskarray(data)):
        raise ConfigurationRefused(f"WRF time variable {variable.name} contains masked data")
    data = np.asarray(data)
    if data.ndim != 2 or data.shape[1] != 19:
        raise ConfigurationRefused(
            f"WRF time variable {variable.name} must have shape (records, 19), "
            f"got {data.shape}")
    values = []
    for row in data:
        try:
            text = b"".join(np.asarray(row, dtype="S1").tolist()).decode(
                "ascii")
        except UnicodeDecodeError as exc:
            raise ConfigurationRefused(
                f"WRF time variable {variable.name} is not ASCII") from exc
        try:
            value = datetime.strptime(text, "%Y-%m-%d_%H:%M:%S")
        except ValueError as exc:
            raise ConfigurationRefused(
                f"WRF time variable {variable.name} has invalid timestamp "
                f"{text!r}") from exc
        if value.strftime("%Y-%m-%d_%H:%M:%S") != text:
            raise ConfigurationRefused(
                f"WRF time variable {variable.name} timestamp is not canonical: "
                f"{text!r}")
        values.append(value)
    return tuple(values)


_WRFBDY_FIELDS = {
    "u": ("U", "bottom_top", "south_north", "west_east_stag"),
    "v": ("V", "bottom_top", "south_north_stag", "west_east"),
    "theta": ("T", "bottom_top", "south_north", "west_east"),
    "phi": ("PH", "bottom_top_stag", "south_north", "west_east"),
    "mu": ("MU", None, "south_north", "west_east"),
    "qv": ("QVAPOR", "bottom_top", "south_north", "west_east"),
}


def _wrfbdy_side_table(variable, index: int, side_name: str) -> np.ndarray:
    """Transpose WRF ``(width,z,side)`` into woof's side convention."""
    value = np.ma.asarray(variable[index])
    if np.ma.isMaskedArray(value) and np.any(np.ma.getmaskarray(value)):
        raise ConfigurationRefused(f"wrfbdy_d01 {variable.name} contains masked data")
    value = np.asarray(value, dtype=np.float32)
    if value.ndim == 3:
        axes = ((1, 2, 0) if side_name in ("west", "east")
                else (1, 0, 2))
        value = np.transpose(value, axes)
    elif value.ndim == 2:
        value = (np.transpose(value, (1, 0))[None]
                 if side_name in ("west", "east") else value[None])
    else:
        raise ConfigurationRefused(
            f"wrfbdy_d01 {variable.name} must be a 2-D or 3-D side table")
    if not np.all(np.isfinite(value)):
        raise ConfigurationRefused(f"wrfbdy_d01 {variable.name} contains non-finite data")
    return np.ascontiguousarray(value)


def _boundary_strip(field: np.ndarray, side_name: str,
                    width: int) -> np.ndarray:
    if side_name == "west":
        return field[..., :width]
    if side_name == "east":
        return field[..., -width:][..., ::-1]
    if side_name == "south":
        return field[..., :width, :]
    return field[..., -width:, :][..., ::-1, :]


def _mass_from_boundary_sides(initial_mu: np.ndarray,
                              sides: Mapping[str, np.ndarray]) -> np.ndarray:
    """Restore a boundary-frame MU field from gpuwm-oriented side tables."""
    mu = np.asarray(initial_mu, dtype=np.float32).copy()
    width = sides["west"].shape[-1]
    for distance in range(width):
        mu[:, distance] = sides["west"][0, :, distance]
        mu[:, -1 - distance] = sides["east"][0, :, distance]
        mu[distance, :] = sides["south"][0, distance, :]
        mu[-1 - distance, :] = sides["north"][0, distance, :]
    return mu


def _wrf_and_gpuwm_mass_weights(restored: RestoredDomain,
                                mu: np.ndarray
                                ) -> tuple[dict[str, np.ndarray],
                                           dict[str, np.ndarray]]:
    """Return source-WRF and target-gpuwm FP32 dry-mass weights.

    ``real_em.F`` calls ``couple`` before packing wrfbdy.  At mass points
    that routine retains separate perturbation/base products, whereas
    woof's producer first forms total column mass.  U/V additionally use
    WRF's one-sided physical faces and staggered mass averages.  Keeping the
    two expression trees distinct makes read-time normalization deterministic
    instead of relying on algebraic equivalence across FP32 roundoff.
    """
    return _mass_weights_from_raw(restored.raw, mu)


def _mass_weights_from_raw(raw: Mapping[str, np.ndarray], mu: np.ndarray
                           ) -> tuple[dict[str, np.ndarray],
                                      dict[str, np.ndarray]]:
    """:func:`_wrf_and_gpuwm_mass_weights` on the initial file's variables
    themselves, for a writer that holds them before any file exists."""
    required = {
        "MUB", "C1H", "C2H", "C1F", "C2F",
        "MAPFAC_M", "MAPFAC_U", "MAPFAC_V",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(
            f"wrfinput lacks wrfbdy coupling variable(s): {missing}")
    mub = np.asarray(raw["MUB"], dtype=np.float32)
    if mu.shape != mub.shape:
        raise ValueError(
            f"wrfbdy MU frame {mu.shape} != wrfinput MUB {mub.shape}")
    c1h = np.asarray(raw["C1H"], dtype=np.float32)[:, None, None]
    c2h = np.asarray(raw["C2H"], dtype=np.float32)[:, None, None]
    c1f = np.asarray(raw["C1F"], dtype=np.float32)[:, None, None]
    c2f = np.asarray(raw["C2F"], dtype=np.float32)[:, None, None]

    # WRF v4.6.1 module_big_step_utilities_em.F:423-506,576-606.
    wrf_muu = np.empty((mu.shape[0], mu.shape[1] + 1), dtype=np.float32)
    wrf_muv = np.empty((mu.shape[0] + 1, mu.shape[1]), dtype=np.float32)
    wrf_muu[:, 1:-1] = np.asarray(
        np.float32(0.5) * (mu[:, 1:] + mu[:, :-1]
                           + mub[:, 1:] + mub[:, :-1]), dtype=np.float32)
    wrf_muv[1:-1, :] = np.asarray(
        np.float32(0.5) * (mu[1:, :] + mu[:-1, :]
                           + mub[1:, :] + mub[:-1, :]), dtype=np.float32)
    wrf_muu[:, 0] = np.asarray(mu[:, 0] + mub[:, 0], dtype=np.float32)
    wrf_muu[:, -1] = np.asarray(mu[:, -1] + mub[:, -1], dtype=np.float32)
    wrf_muv[0, :] = np.asarray(mu[0, :] + mub[0, :], dtype=np.float32)
    wrf_muv[-1, :] = np.asarray(mu[-1, :] + mub[-1, :], dtype=np.float32)
    wrf_half = np.asarray(
        c1h * mu[None] + (c1h * mub[None] + c2h), dtype=np.float32)
    wrf_full = np.asarray(
        c1f * mu[None] + (c1f * mub[None] + c2f), dtype=np.float32)

    # woof lateral_bc._coupled_device_fields: total MU is formed first.
    total = np.asarray(mub + mu, dtype=np.float32)
    gpu_muu = np.empty_like(wrf_muu)
    gpu_muv = np.empty_like(wrf_muv)
    gpu_muu[:, 1:-1] = np.asarray(
        np.float32(0.5) * (total[:, 1:] + total[:, :-1]),
        dtype=np.float32)
    gpu_muv[1:-1, :] = np.asarray(
        np.float32(0.5) * (total[1:, :] + total[:-1, :]),
        dtype=np.float32)
    gpu_muu[:, 0], gpu_muu[:, -1] = total[:, 0], total[:, -1]
    gpu_muv[0, :], gpu_muv[-1, :] = total[0, :], total[-1, :]
    gpu_half = np.asarray(c1h * total[None] + c2h, dtype=np.float32)
    gpu_full = np.asarray(c1f * total[None] + c2f, dtype=np.float32)

    wrf = {
        "u": np.asarray(c1h * wrf_muu[None] + c2h, dtype=np.float32),
        "v": np.asarray(c1h * wrf_muv[None] + c2h, dtype=np.float32),
        "theta": wrf_half, "phi": wrf_full, "qv": wrf_half,
    }
    woof = {
        "u": np.asarray(c1h * gpu_muu[None] + c2h, dtype=np.float32),
        "v": np.asarray(c1h * gpu_muv[None] + c2h, dtype=np.float32),
        "theta": gpu_half, "phi": gpu_full, "qv": gpu_half,
    }
    if any(not np.all(np.isfinite(weight)) or np.any(weight <= 0.0)
           for weight in (*wrf.values(), *woof.values())):
        raise ValueError("wrfinput produces invalid wrfbdy dry-mass weights")
    return wrf, woof


def _normalize_wrfbdy_endpoint(
        restored: RestoredDomain, gpu_name: str, side_name: str,
        coupled: np.ndarray, width: int,
        wrf_weights: Mapping[str, np.ndarray],
        gpuwm_weights: Mapping[str, np.ndarray]) -> np.ndarray:
    """Decouple one WRF endpoint and recouple it in woof producer order."""
    if gpu_name == "mu":
        return np.asarray(coupled, dtype=np.float32)
    wrf_weight = _boundary_strip(
        _boundary_mass_weight(wrf_weights, gpu_name), side_name, width)
    gpuwm_weight = _boundary_strip(
        _boundary_mass_weight(gpuwm_weights, gpu_name), side_name, width)
    if np.array_equal(wrf_weight, gpuwm_weight):
        # Avoid a lossy divide/multiply round trip when the expression trees
        # already agree bitwise (normally U/V).
        return np.asarray(coupled, dtype=np.float32)
    primitive = np.asarray(coupled / wrf_weight, dtype=np.float32)
    if gpu_name in ("u", "v"):
        map_name = "MAPFAC_U" if gpu_name == "u" else "MAPFAC_V"
        map_factor = _boundary_strip(
            np.asarray(restored.raw[map_name], dtype=np.float32)[None],
            side_name, width)[0]
        primitive = np.asarray(primitive * map_factor[None], dtype=np.float32)
        return np.asarray(
            np.asarray(primitive * gpuwm_weight, dtype=np.float32)
            / map_factor[None], dtype=np.float32)
    return np.asarray(primitive * gpuwm_weight, dtype=np.float32)


# module_model_constants.F declares R_v and R_d as default REAL. The
# real.exe initial/boundary writer divides those FP32 constants; rounding
# the FP64 ratio instead changes one ULP and breaks its THM pairing proof.
_WRF_RVOVRD = np.float32(461.6)/np.float32(287.0)


def _moist_theta_time_law(restored, side_name, tables, width,
                          wrf_weights, gpuwm_weights):
    """Convert source THM/QV/MU interpolation into a dry-theta time law.

    A and Q are WRF's coupled perturbation THM and QV. M is WRF's mass
    weight, G is the shared dycore's weight. In exact arithmetic the target
    is G*(A-300*a*Q)/(M+a*Q), with a=Rv/Rd. All five source quantities are
    linear in time, making this a quadratic/linear rational function.
    Its derivative is supplied to spec_bdytend, not an endpoint secant.
    """
    from woof.ingest.lateral_bc import RationalTimeLaw, SideBoundary
    a = float(_WRF_RVOVRD)
    A, Ad = (np.asarray(x, np.float64) for x in tables["theta"][side_name])
    Q, Qd = (np.asarray(x, np.float64) for x in tables["qv"][side_name])
    _, mud = tables["mu"][side_name]
    M = np.asarray(_boundary_strip(wrf_weights["theta"], side_name, width), np.float64)
    G = np.asarray(_boundary_strip(gpuwm_weights["theta"], side_name, width), np.float64)
    Md = np.asarray(restored.raw["C1H"], np.float64)[:, None, None] * np.asarray(mud, np.float64)
    Gd = Md
    H, Hd = A-300.0*a*Q, Ad-300.0*a*Qd
    denominator = M+a*Q
    if np.any(denominator <= 0.0) or not np.isfinite(denominator).all():
        raise ValueError("wrfbdy moist-theta conversion requires positive finite mass and moisture denominator")
    rate = (Md+a*Qd)/denominator
    value = G*H/denominator
    tendency = (Gd*H+G*Hd)/denominator-value*rate
    return SideBoundary(value, tendency, RationalTimeLaw(Gd*Hd/denominator, rate))


def _boundary_mass_weight(weights, name):
    """Only declared mass-grid scalar fields share water-vapour coupling."""
    from woof.ingest.lateral_bc import COUPLED_SCALAR_STATE_FIELDS
    return weights["qv" if name in COUPLED_SCALAR_STATE_FIELDS else name]


def wrf_initial_boundary_head(raw, layouts=None, *, moist_theta: bool,
                              weights=None) -> dict[str, np.ndarray]:
    """The first wrfbdy record real.exe writes for one initial state.

    Every boundary field of ``raw`` (wrfinput variables, by WRF name)
    coupled as ``couple()`` couples it before ``stuff_bdy`` packs the
    first record (WRF v4.6.1 main/real_em.F:866-899): U by the staggered
    mass over MAPFAC_U, V likewise, T, QVAPOR and the declared scalars by
    the half-level mass, PH by the full-level mass, MU copied.  FP32, in
    WRF's own expression order, on the whole domain; the caller takes the
    side strips.  Under ``moist_theta`` T is first rebuilt as real.exe's
    moist theta from the file's dry T and QVAPOR.

    The door's consistency check reads its expectation from here, and the
    stock-WRF exporter writes its first record from here, so the pair the
    exporter publishes and the pair the door accepts are one definition.
    """
    layouts = _WRFBDY_FIELDS if layouts is None else layouts
    if weights is None:
        weights, _ = _mass_weights_from_raw(
            raw, np.asarray(raw["MU"], dtype=np.float32))
    head = {}
    for key, (name, *_dimensions) in layouts.items():
        input_name = "qke_adv" if key == "qke_adv" else name
        field = np.asarray(raw[input_name], dtype=np.float32)
        if key == "theta" and moist_theta:
            field = _moist_theta_from_dry(field, raw["QVAPOR"])
        if key == "mu":
            head[key] = field[None]
            continue
        field = np.asarray(field * _boundary_mass_weight(weights, key),
                           dtype=np.float32)
        if key in ("u", "v"):
            field = np.asarray(
                field / np.asarray(raw["MAPFAC_" + name],
                                   dtype=np.float32)[None],
                dtype=np.float32)
        head[key] = field
    return head


def _check_initial_boundary_pair(restored, tables, width, *, layouts=None):
    """Compare the first boundary values with the file that initializes them.

    Reconstruct WRF's own FP32 couple operation, before normalization into
    WOOF's coupling order. Never replace MU and then use the replacement as
    proof that an unrelated boundary file belongs to this initial condition.
    The breakage this prevents: a wrfbdy from another real.exe run (another
    start, analysis or namelist) silently forcing this initial state, its
    first record a different atmosphere than the one the run starts from.
    """
    layouts = _WRFBDY_FIELDS if layouts is None else layouts
    raw = restored.raw
    wrf, _ = _wrf_and_gpuwm_mass_weights(restored, np.asarray(raw["MU"], dtype=np.float32))
    moist_theta = int(restored.global_attributes["USE_THETA_M"]) == 1
    head = wrf_initial_boundary_head(raw, layouts, moist_theta=moist_theta,
                                     weights=wrf)
    for key, (name, *_dimensions) in layouts.items():
        field = head[key]
        # How far two builds' reconstructions of a field can differ before
        # coupling, per element of one side's strip (a function of the
        # side, so it is computed on the width-``width`` strips the check
        # compares, never on the whole domain): none for a field copied
        # from the file.
        field_bound = other = other_bound = None
        if key == "theta":
            # WRF 4.0 through 4.7.1 real.exe writes wrfinput T dry under
            # both settings (Registry.EM_COMMON:209 maps th_phy_m_t0 to
            # "T", saved at module_initialize_real.F:4913 before the
            # conversion) and couples its runtime theta into T_B*
            # (real_em.F:872): moist theta when use_theta_m=1
            # (module_initialize_real.F:4923-4932, this file's own
            # QVAPOR).  The writer's representation is the head above;
            # keep the other one to say what the boundary holds instead
            # when the two files disagree.
            dry = np.asarray(raw[name], dtype=np.float32)
            moist = _moist_theta_from_dry(dry, raw["QVAPOR"])
            def moist_bound(side, dry=dry, qv=raw["QVAPOR"]):
                return _moist_theta_build_bound(
                    _boundary_strip(dry, side, width),
                    _boundary_strip(np.asarray(qv, np.float32), side, width))
            if moist_theta:
                field_bound, other = moist_bound, dry
            else:
                other, other_bound = moist, moist_bound
        if key != "mu":
            weight = _boundary_mass_weight(wrf, key)
            if other is not None:
                other = np.asarray(other * weight, dtype=np.float32)
            # couple() multiplies the reconstruction by the dry column mass,
            # and with it the reconstruction's build-to-build difference.
            if field_bound is not None:
                field_bound = _coupled_bound(field_bound, weight, width)
            if other_bound is not None:
                other_bound = _coupled_bound(other_bound, weight, width)
        for side, (actual, _tendency) in tables[key].items():
            expected = _boundary_strip(field, side, width)
            actual = np.asarray(actual, dtype=np.float32)
            tolerance = _coupled_build_tolerance(expected, field_bound, side, width)
            if actual.shape != expected.shape or np.any(np.abs(actual - expected) > tolerance):
                other_strip = None if other is None else _boundary_strip(other, side, width)
                raise ConfigurationRefused(_initial_boundary_refusal(
                    restored, name, side, actual, expected, tolerance,
                    other_strip, moist_theta,
                    None if other is None else _coupled_build_tolerance(
                        other_strip, other_bound, side, width)))


def _coupled_build_tolerance(expected, reconstruction_bound, side, width):
    """How far an independent WRF build's value of one coupled boundary
    element may sit from ours, without treating a changed boundary field
    as a new initial state.

    Four FP32 spacings of the coupled value cover couple()'s own rounding
    (its mass weight and the multiply, and the map-factor divide of U/V).
    That rounding scales with the coupled value itself: the weight sums
    terms of one size and sign (MU is a few percent of MUB), so no operand
    outweighs the result; a GCC real.exe that fuses those multiply-adds
    sat at most 3 spacings from ours in every field copied from the file
    (U, V, PH, MU, QVAPOR and the aerosol numbers).  A field real.exe
    derived before coupling also carries its reconstruction's
    build-to-build difference, which scales with the reconstruction's
    operands and was carried through the coupling by the caller.
    """
    tolerance = 4.0 * np.abs(np.spacing(expected)) + 1e-6
    if reconstruction_bound is None:
        return tolerance
    return tolerance + reconstruction_bound(side)


def _coupled_bound(bound, weight, width):
    """``bound`` carried through couple()'s multiply by the mass weight,
    side by side on the strips alone."""

    def coupled(side):
        strip = np.abs(_boundary_strip(np.asarray(weight), side, width)
                       .astype(np.float64))
        return bound(side) * strip

    return coupled


def _moist_theta_from_dry(theta, qv):
    """WRF's theta_m - T0 from theta - T0, in real.exe's FP32 operation order.

    module_initialize_real.F:4930 (WRF 4.7.1; v4.6.1 :4909):
    ``t_2 = (t_2 + T0) * (1. + (R_v/R_d) * qv) - T0``.
    """
    factor = np.float32(1.0) + _WRF_RVOVRD * np.asarray(qv, np.float32)
    return np.asarray(factor * (np.asarray(theta, np.float32) + np.float32(300.0))
                      - np.float32(300.0), dtype=np.float32)


def _moist_theta_build_bound(theta, qv):
    """Largest |difference|, in K, between two FP32 builds of
    ``t_2 = (t_2 + T0) * (1. + (R_v/R_d) * qv) - T0``, per element.

    Operation count.  s = t_2 + T0 is a lone add of two stored values, so
    every build rounds it identically.  Five operations remain: the
    R_v/R_d quotient (folded at compile time, at the compiler's choice of
    precision), a*qv, f = 1 + a*qv, p = s*f and r = p - T0.  A build rounds
    each to nearest or fuses a multiply into the add after it (a GCC build
    for a CPU with FMA forms 1 + a*qv and s*f - T0 with one rounding each),
    and a rounding moves its result by at most u = 2**-24 of the result's
    magnitude.  Two builds therefore differ at one operation by at most 2u
    of it, carried to r by the partial derivative of r (s*qv*a for the
    quotient and the product, s for f, 1 for p and r).  Summed over the
    five, to first order in u (the remainder is u times smaller):

        |r_A - r_B| <= 2u * (2*|s*a*qv| + |s*f| + |p| + |r|)

    That is two to six FP32 spacings of the full moist theta p, a few
    hundred K, while r = p - 300 is often a few K and its own spacing 64
    or more times finer: the spacing of the result cannot bound it.
    """
    theta = np.asarray(theta, np.float32)
    s = np.asarray(theta + np.float32(300.0), np.float32)
    aq = np.asarray(_WRF_RVOVRD * np.asarray(qv, np.float32), np.float32)
    f = np.asarray(np.float32(1.0) + aq, np.float32)
    p = np.asarray(f * s, np.float32)
    r = np.asarray(p - np.float32(300.0), np.float32)
    s, aq, f, p, r = (np.abs(x.astype(np.float64)) for x in (s, aq, f, p, r))
    return 2.0 * 2.0**-24 * (2.0 * s * aq + s * f + p + r)


def _initial_boundary_refusal(restored, name, side, actual, expected, tolerance,
                              other, moist_theta, other_tolerance):
    """Name what was compared, how far apart the files are, and what the
    boundary holds instead, so a mixed pair and a representation mismatch
    read differently at the terminal."""
    head = f"wrfbdy {name} {side} does not match initial {restored.path.name}"
    tail = "use the pair produced by the same real.exe run"
    if actual.shape != expected.shape:
        return (f"{head}: boundary table shape {actual.shape} != initial "
                f"strip {expected.shape}; {tail}")
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    # Scaled by the strip's largest |value|, not point by point: theta-300
    # crosses zero inside a column, so a per-point ratio would be dominated
    # by whichever point sits nearest zero.
    scale = max(float(np.abs(expected.astype(np.float64)).max()), 1e-30)
    size = (f"{int(np.count_nonzero(difference > tolerance))} of {difference.size} "
            f"points differ, max |difference| {difference.max():.6g} "
            f"({difference.max() / scale:.3g} of the largest value)")
    if other is None:
        return f"{head}: {size}; {tail}"
    if moist_theta:
        compared = ("USE_THETA_M=1 on both files: wrfinput T (dry theta-300, as "
                    "WRF 4.x real.exe writes it) was converted to moist theta with "
                    "wrfinput QVAPOR, (T+300)*(1+Rv/Rd*QVAPOR)-300 as real.exe does "
                    "before coupling T_B*, coupled with dry column mass and compared "
                    "with wrfbdy T_B*")
        other_name = "DRY"
        writers = ("WRF 4.0 to 4.7.1 real.exe never writes that under use_theta_m=1 "
                   "(its wrfinput carries THM and its T_B* holds moist theta); WRF "
                   "3.7 to 3.9.1.1 real.exe always does (use_theta_m entered the "
                   "namelist in 3.7; no THM variable; the solver converted to moist "
                   "theta at run time)")
    else:
        compared = ("USE_THETA_M=0 on both files: wrfinput T (dry theta-300) was "
                    "coupled with dry column mass and compared with wrfbdy T_B* "
                    "as dry theta")
        other_name = "MOIST"
        writers = "no stock real.exe writes that under use_theta_m=0"
    other_difference = np.abs(actual.astype(np.float64) - other.astype(np.float64))
    if not np.any(other_difference > other_tolerance):
        return (f"{head}: {compared}; {size}. The boundary equals the {other_name} "
                f"coupling of this wrfinput T at every point instead: {writers}. "
                f"Re-run real.exe so both files come from one run whose boundary "
                f"representation matches its USE_THETA_M")
    return (f"{head}: {compared}; {size}; the boundary is not the {other_name} "
            f"coupling of this T either (max |difference| {other_difference.max():.6g}), "
            f"so T or QVAPOR differ between the two files; {tail}")


def _check_boundary_identity(dataset, restored):
    for name in ("GRID_ID", "DX", "DY", "MAP_PROJ", "TRUELAT1", "TRUELAT2",
                 "STAND_LON", "CEN_LAT", "CEN_LON", "HYBRID_OPT", "ETAC", "USE_THETA_M"):
        if name not in dataset.ncattrs() or name not in restored.global_attributes:
            raise ConfigurationRefused(f"wrfbdy/wrfinput pair lacks {name} identity")
        found = dataset.getncattr(name)
        expected = restored.global_attributes[name]
        if not np.isfinite([found, expected]).all() or float(found) != float(expected):
            raise ConfigurationRefused(f"wrfbdy {name}={found} differs from wrfinput {expected}")
    if int(dataset.getncattr("USE_THETA_M")) not in (0, 1):
        raise ConfigurationRefused("wrfbdy USE_THETA_M must identify dry (0) or moist (1) theta")


def read_wrfbdy(path: str | Path, *, run_seconds: float,
                restored: RestoredDomain,
                forcing_interval_seconds: float,
                spec_bdy_width: int = 5,
                first_interval_only: bool = False, cfg=None,
                pair_restored: RestoredDomain | None = None):
    """Build coupled woof d01 boundary intervals from real.exe tables.

    ``restored`` normally initializes the run at wrfbdy's first record, and
    the first record is checked against it.  A domain restored from a WRF
    restart (``restored.warm_start``) is a later instant of the same
    simulation: its base state and map factors still couple the tables,
    but its prognostics are not the first record's, so that check gives way
    to the restart's own clock check (wrf.exe likewise only seeks the
    bounding interval on restart).

    ``cfg`` carries the actual selected scalar boundary operation. The
    public WRF door always supplies it; its absence retains the original
    dry/vapour inventory for compatibility with verification callers.

    ``spec_bdy_width`` is the caller's ``&bdy_control spec_bdy_width``; the
    file's own ``bdy_width`` dimension must equal it.  It was pinned at the
    registered N5S value of 5 while this was a verification importer, which
    would have refused any user who set it otherwise.

    ``pair_restored`` is the state the first record must be the coupling of when that is not
    the state the run starts from: an incremental analysis update starts from the background
    and forces the analysis in over its window, and its boundary was rebased onto the
    ANALYSIS (woof.da.analysis_door, WRFDA's da_update_bc), so the rim and the interior
    head for one atmosphere.  The check is the same check, against that state; it is
    ``None`` (the start itself) on every other route.

    ``first_interval_only`` keeps a run inside the FIRST forcing interval
    -- the registered N5S contract, which is scored only there.  A front
    door leaves it False and consumes every validated record, which is what
    lets ``run_seconds`` reach the full ``wrfbdy`` coverage.  It used to be
    inferred from the forcing interval being six hours, which meant a user
    with ordinary six-hourly GFS forcing silently inherited the
    verification case's ceiling.
    """
    from woof.ingest.lateral_bc import (
        BoundaryInterval, FieldBoundary, LateralBoundaries, SideBoundary,
    )

    if not isinstance(restored, RestoredDomain):
        raise TypeError("read_wrfbdy requires the restored d01 wrfinput")
    if (isinstance(forcing_interval_seconds, (bool, np.bool_))
            or not isinstance(
                forcing_interval_seconds,
                (int, float, np.integer, np.floating))
            or not np.isfinite(float(forcing_interval_seconds))
            or float(forcing_interval_seconds) <= 0.0
            or not float(forcing_interval_seconds).is_integer()):
        raise ValueError(
            "forcing_interval_seconds must be a positive whole number")
    forcing_interval = int(forcing_interval_seconds)
    from woof.boundary_fields import external_scalar_fields
    layouts = dict(_WRFBDY_FIELDS)
    if cfg is not None:
        by_state = {state: wrf for wrf, state in active_moisture_map(cfg).items()}
        for name in external_scalar_fields(cfg):
            layouts[name] = (by_state[name], "bottom_top", "south_north", "west_east")
    field_names = {name: layout[0] for name, layout in layouts.items()}
    with netcdf_bridge.open_dataset(path) as dataset:
        _check_boundary_identity(dataset, restored)
        if (cfg is not None and cfg.bl_pbl_physics == 5 and cfg.bl_mynn_tkeadvect
                and any(name.upper().startswith("QKE_ADV_B") for name in dataset.variables)):
            prefix = ("qke_adv" if any(name.startswith("qke_adv_B")
                                      for name in dataset.variables) else "QKE_ADV")
            layouts["qke_adv"] = (prefix, "bottom_top", "south_north", "west_east")
            field_names["qke_adv"] = prefix
        if "Times" not in dataset.variables:
            raise ValueError("wrfbdy_d01 has no Times variable")
        if "bdy_width" not in dataset.dimensions:
            raise ValueError("wrfbdy_d01 has no bdy_width dimension")
        width = len(dataset.dimensions["bdy_width"])
        if width != int(spec_bdy_width):
            raise ValueError(
                f"{path} bdy_width {width} != the requested "
                f"spec_bdy_width {int(spec_bdy_width)}; real.exe writes "
                "the boundary tables at the width &bdy_control asked for, "
                "so this pair came from two different namelists")
        times = _decode_times(dataset.variables["Times"])
        if not times:
            raise ValueError("wrfbdy_d01 has no records")
        if any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("wrfbdy_d01 records must increase")
        next_name = ("md___nextbdytimee_x_t_d_o_m_a_i_n_m_e_t_a_data_")
        if len(times) == 1:
            # real.exe's standard single-interval product: one record of
            # boundary values at thisbdytime plus _BT tendencies valid to
            # nextbdytime.  The interval loop below already extrapolates the
            # last record by the registered cadence; here we only verify the
            # file's own metadata pins that same interval.
            if next_name not in dataset.variables:
                raise ValueError(
                    "single-record wrfbdy_d01 needs nextbdytime metadata")
            next_times = _decode_times(dataset.variables[next_name])
            expected_next = times[0] + timedelta(
                seconds=forcing_interval)
            if len(next_times) != 1 or next_times[0] != expected_next:
                raise ValueError(
                    "single-record wrfbdy_d01 nextbdytime does not pin the "
                    f"requested {forcing_interval}-second forcing interval")
        lead = None
        if len(times) > 1:
            # A first record covering less than the cadence: a cold start
            # inside a boundary interval (woof.da.analysis_boundary).
            cadence, lead = lead_interval_seconds(
                times, _decode_times(dataset.variables[next_name])
                if next_name in dataset.variables else ())
            if lead is not None and cadence != forcing_interval:
                raise ValueError(
                    f"{path} steps by {cadence} seconds after its first "
                    f"record, not the requested {forcing_interval}-second "
                    "forcing cadence")
        if len(times) > 1 and next_name in dataset.variables:
            # WRF products encountered in the wild use either one metadata
            # record for the final endpoint or one next-time record per Times
            # record.  Accept only those two explicit encodings.
            next_times = _decode_times(dataset.variables[next_name])
            expected_all = tuple(
                value + timedelta(seconds=forcing_interval)
                for value in times)
            if lead is not None:
                expected_all = (times[1],) + expected_all[1:]
            expected_final = (expected_all[-1],)
            if next_times not in (expected_all, expected_final):
                raise ValueError(
                    "multi-record wrfbdy_d01 nextbdytime does not match "
                    f"Times plus {forcing_interval} seconds")
        required_records = []
        for gpu_name, wrf_name in field_names.items():
            _, zdim, ydim, xdim = layouts[gpu_name]
            for side_name, suffix in (
                    ("west", "XS"), ("east", "XE"),
                    ("south", "YS"), ("north", "YE")):
                value_name = f"{wrf_name}_B{suffix}"
                tendency_name = f"{wrf_name}_BT{suffix}"
                for record_name in (value_name, tendency_name):
                    if record_name not in dataset.variables:
                        raise ValueError(f"wrfbdy_d01 is missing {record_name}")
                    records = dataset.variables[record_name].shape[0]
                    if records != len(times):
                        raise ValueError(
                            f"wrfbdy_d01 {record_name} record count mismatch: "
                            f"expected {len(times)}, got {records}")
                if (dataset.variables[value_name].shape
                        != dataset.variables[tendency_name].shape):
                    raise ValueError(
                        f"wrfbdy_d01 {value_name}/{tendency_name} shape "
                        "mismatch")
                side_dim = ydim if side_name in ("west", "east") else xdim
                expected_dimensions = (["Time", "bdy_width"]
                                       + ([] if zdim is None else [zdim])
                                       + [side_dim])
                for record_name in (value_name, tendency_name):
                    actual_dimensions = list(dataset.variables[record_name].dimensions)
                    if actual_dimensions != expected_dimensions:
                        raise ValueError(
                            f"wrfbdy_d01 {record_name} dimensions "
                            f"{tuple(actual_dimensions)} != expected real.exe "
                            f"{tuple(expected_dimensions)}")
                required_records.append(
                    (gpu_name, wrf_name, side_name, value_name, tendency_name))
        origin = times[0]
        if origin.strftime("%Y-%m-%d_%H:%M:%S") != str(restored.global_attributes.get("START_DATE")):
            raise ValueError("wrfbdy first Times record differs from wrfinput START_DATE")
        starts = [(time - origin).total_seconds() for time in times]
        forcing_intervals = [
            later - earlier for earlier, later in zip(starts, starts[1:])]
        if lead is not None:
            forcing_intervals = forcing_intervals[1:]
        if any(interval != forcing_interval
               for interval in forcing_intervals):
            observed = sorted({int(value) for value in forcing_intervals})
            raise ValueError(
                f"{path} must use the requested {forcing_interval}-second "
                f"forcing cadence; its Times records step by {observed} "
                "seconds")
        first_interval_end = (starts[1] if len(starts) > 1
                              else starts[0] + forcing_interval)
        coverage_end = (first_interval_end if first_interval_only
                        else starts[-1] + forcing_interval)
        if not 0.0 < float(run_seconds) <= coverage_end:
            # BOTH numbers, before any integration.  A run that outlives
            # its lateral forcing does not fail -- it keeps extrapolating
            # the last tendency and produces a finite, bounded, wrong
            # forecast, which is the exact failure mode this door exists
            # to make impossible.
            scope = ("the FIRST boundary interval" if first_interval_only
                     else f"{len(times)} boundary record(s)")
            raise ValueError(
                f"requested run of {float(run_seconds):.0f} s exceeds the "
                f"{coverage_end:.0f} s of lateral forcing in {path} "
                f"({scope} at a {forcing_interval}-second cadence, the "
                "last one extrapolated by its own _BT tendency for one "
                "further interval).  Shorten the run to at most "
                f"{coverage_end:.0f} s, or re-run real.exe over a longer "
                "window so wrfbdy_d01 carries more records.")
        # Boundary variables share the same bounded Rust decode path as
        # initial-state fields. Keep first-use order and each variable's
        # mask/scale policy while avoiding a process per side and tendency.
        decode = netcdf_bridge.BatchedDecode(dataset, [
            name for record in required_records for name in record[3:]])
        intervals = []
        for index, start in enumerate(starts):
            if index + 1 < len(starts):
                end = starts[index + 1]
            else:
                end = start + forcing_interval
            raw_tables = {gpu_name: {} for gpu_name in field_names}
            for (gpu_name, wrf_name, side_name, value_name,
                 tendency_name) in required_records:
                decode.ready(value_name)
                value = _wrfbdy_side_table(
                    dataset.variables[value_name], index, side_name)
                decode.ready(tendency_name)
                tendency = _wrfbdy_side_table(
                    dataset.variables[tendency_name], index, side_name)
                raw_tables[gpu_name][side_name] = (value, tendency)
            if index == 0 and getattr(restored, "warm_start", None) is None:
                _check_initial_boundary_pair(
                    restored if pair_restored is None else pair_restored,
                    raw_tables, width, layouts=layouts)
            duration = float(end - start)
            mu_start_sides = {
                side_name: pair[0]
                for side_name, pair in raw_tables["mu"].items()}
            mu_end_sides = {
                side_name: np.asarray(
                    pair[0] + np.float32(duration) * pair[1],
                    dtype=np.float32)
                for side_name, pair in raw_tables["mu"].items()}
            mu_start = _mass_from_boundary_sides(
                restored.raw["MU"], mu_start_sides)
            mu_end = _mass_from_boundary_sides(
                restored.raw["MU"], mu_end_sides)
            wrf_start, gpuwm_start = _wrf_and_gpuwm_mass_weights(
                restored, mu_start)
            wrf_end, gpuwm_end = _wrf_and_gpuwm_mass_weights(
                restored, mu_end)
            side_tables = {gpu_name: {} for gpu_name in field_names}
            for gpu_name, sides in raw_tables.items():
                for side_name, (value, tendency) in sides.items():
                    if (gpu_name == "theta" and
                            int(restored.global_attributes["USE_THETA_M"]) == 1):
                        side_tables[gpu_name][side_name] = _moist_theta_time_law(
                            restored, side_name, raw_tables, width,
                            wrf_start, gpuwm_start)
                        continue
                    if WRF_EXACT:
                        # Strict mode relaxes and specifies toward real.exe's
                        # own words, as WRF does: relax_bdytend and
                        # spec_bdytend read field_bdy + dtbc*field_bdy_tend
                        # straight from wrfbdy.  The default path recouples
                        # each endpoint in woof's mass order and re-derives
                        # the tendency in float64, which moves boundary
                        # words by an ULP and the relaxation zone with them
                        # (combo-sweep round 2, LOCALIZE.md item 5).
                        side_tables[gpu_name][side_name] = SideBoundary(
                            np.asarray(value, np.float32).astype(np.float64),
                            np.asarray(tendency, np.float32).astype(
                                np.float64))
                        continue
                    future = np.asarray(
                        value + np.float32(duration) * tendency,
                        dtype=np.float32)
                    normalized = _normalize_wrfbdy_endpoint(
                        restored, gpu_name, side_name, value, width,
                        wrf_start, gpuwm_start)
                    normalized_future = _normalize_wrfbdy_endpoint(
                        restored, gpu_name, side_name, future, width,
                        wrf_end, gpuwm_end)
                    target_tendency = (
                        normalized_future.astype(np.float64)
                        - normalized.astype(np.float64)) / duration
                    side_tables[gpu_name][side_name] = SideBoundary(
                        normalized.astype(np.float64), target_tendency)
            fields = {
                gpu_name: FieldBoundary(**side_tables[gpu_name])
                for gpu_name in field_names
            }
            intervals.append(BoundaryInterval(start, end, fields))
    return LateralBoundaries(tuple(intervals), width, 1, width - 1)


# ==========================================================================
# The front door's own instruments: cheap, CPU-only, and refusing FIRST.
# ==========================================================================


@dataclass(frozen=True)
class WrfinputMetadata:
    """Everything a door needs from a wrfinput without decoding a field.

    Read from dimensions, global attributes and variable headers, so it is
    cheap enough to run on every domain before anything is restored.  The
    checks that use it -- scheme support, geometry agreement with
    ``namelist.input`` -- must all fire before the first byte of U is
    decoded, or a mismatch costs the user a full read of an 87 MB file to
    be told the grid is the wrong shape.
    """

    path: Path
    grid_id: int
    dimensions: Mapping[str, int]
    mp_physics: int
    sf_surface_physics: int
    start_date: str
    global_attributes: Mapping[str, object]
    has_qv: bool = False
    #: The file carries the cloud analysis's radar temperature tendency,
    #: all of :data:`RADAR_TENDENCY_VARIABLES` (the four 15-minute slots
    #: and their times), which ``mp_tend_radar = 1`` applies.
    has_radar_tendency: bool = False

    @property
    def nx(self) -> int:
        return self.dimensions["west_east"]

    @property
    def ny(self) -> int:
        return self.dimensions["south_north"]

    @property
    def nz(self) -> int:
        return self.dimensions["bottom_top"]

    @property
    def soil_layers(self) -> int:
        return self.dimensions["soil_layers_stag"]

    def pinned_dimensions(self) -> Mapping[str, int]:
        """The seven-extent geometry contract, as the file declares it.

        Handed to :func:`read_wrfinput` as ``expected_dimensions`` ONLY
        after :func:`check_grid_agreement` has confirmed the file's own
        claim against the namelist.  Skipping that step and pinning the
        geometry to the file would make the pin vacuous -- the file would
        be checked against itself.
        """
        return MappingProxyType({
            name: int(self.dimensions[name])
            for name in sorted(_WRFINPUT_GEOMETRY_DIMENSIONS)})


def read_wrfinput_metadata(path: str | Path) -> WrfinputMetadata:
    """Read one wrfinput's geometry and physics identity, no fields."""
    path = Path(path)
    with netcdf_bridge.open_dataset(path) as dataset:
        dimensions = {name: len(dim)
                      for name, dim in dataset.dimensions.items()}
        attrs = {name: dataset.getncattr(name)
                 for name in dataset.ncattrs()}
        has_qv = "QVAPOR" in dataset.variables
        has_radar_tendency = all(name in dataset.variables
                                 for name in RADAR_TENDENCY_VARIABLES)
    missing = sorted(_WRFINPUT_GEOMETRY_DIMENSIONS - set(dimensions))
    if missing:
        raise ConfigurationRefused(
            f"{path} is missing WRF grid dimension(s) {missing}; a "
            "real.exe wrfinput declares all seven "
            f"({sorted(_WRFINPUT_GEOMETRY_DIMENSIONS)})")
    resolved = check_supported_schemes(attrs, source=str(path))
    grid_id = _integral_attribute(attrs.get("GRID_ID"))
    if grid_id is None:
        raise ConfigurationRefused(
            f"{path} has no integer GRID_ID global attribute, so which "
            "domain of the nest hierarchy it initializes cannot be "
            "established")
    return WrfinputMetadata(
        path=path, grid_id=grid_id,
        dimensions=MappingProxyType(
            {name: int(value) for name, value in dimensions.items()}),
        mp_physics=resolved["MP_PHYSICS"],
        sf_surface_physics=resolved["SF_SURFACE_PHYSICS"],
        start_date=str(attrs.get("START_DATE", "")),
        global_attributes=MappingProxyType(attrs), has_qv=has_qv,
        has_radar_tendency=has_radar_tendency)


def check_grid_agreement(metadata: WrfinputMetadata, cfg, *,
                         source: str) -> None:
    """Refuse a wrfinput/namelist pair that disagrees about the grid.

    ``real.exe`` and WOOF are being handed the SAME ``namelist.input``,
    so nx/ny/nz and the soil geometry must match.  When they do not, one
    of the two files is from a different experiment, and every field that
    follows would either raise deep inside a shape check or -- worse, for
    the 1-D vertical coefficient arrays -- silently broadcast.  Both
    numbers are named for every axis that disagrees, in ONE refusal, so a
    reader is not walked through four axes one re-run at a time.
    """
    from woof.config import soil_layer_count

    rows = [
        ("west_east (nx)", metadata.nx, int(cfg.nx)),
        ("south_north (ny)", metadata.ny, int(cfg.ny)),
        ("bottom_top (nz)", metadata.nz, int(cfg.nz)),
        ("soil_layers_stag", metadata.soil_layers, soil_layer_count(cfg)),
    ]
    bad = [(axis, found, wanted) for axis, found, wanted in rows
           if found != wanted]
    if not bad:
        return
    lines = "\n".join(
        f"  {axis}: {metadata.path.name} has {found}, "
        f"{source} resolves to {wanted}"
        for axis, found, wanted in bad)
    raise ConfigurationRefused(
        f"{metadata.path} and {source} disagree about the grid:\n{lines}\n"
        "Entering at wrfinput means real.exe already baked the domain "
        "layout and the vertical grid, so the namelist cannot change "
        "them: use the namelist.input that produced these files, or "
        "re-run real.exe with this namelist.")


@dataclass(frozen=True)
class BoundaryCoverage:
    """How far in time one ``wrfbdy`` file can force a run."""

    path: Path
    times: tuple[datetime, ...]
    forcing_interval_seconds: int
    spec_bdy_width: int
    coverage_seconds: float
    #: The first record's own interval when it is shorter than the cadence
    #: (:func:`lead_interval_seconds`), else ``None``.
    lead_interval_seconds: int | None = None

    @property
    def start(self) -> datetime:
        return self.times[0]

    @property
    def end(self) -> datetime:
        return self.times[0] + timedelta(seconds=self.coverage_seconds)


def lead_interval_seconds(times, next_times) -> tuple[int, int | None]:
    """``(cadence, lead)`` of a multi-record ``wrfbdy``'s ``Times``.

    Every interval is the cadence, except that the FIRST record may cover
    a shorter lead interval, ``lead`` (``None`` when it does not): the
    file a cold start inside a boundary interval runs from
    (:func:`woof.da.analysis_boundary.rebase_first_record`), whose first
    record was restamped to the start instant and ends where the
    original interval ended, at the second record.  The cadence is the
    step of the later records, or, with two records, the last record's
    own ``nextbdytime`` interval; a lead is accepted only where that
    metadata says so (record 0's ``nextbdytime`` is record 1, or only a
    final one is written), because it is the file's own statement of
    where each interval ends.  The breakage the uniform-cadence refusal
    prevents is still refused: a file whose later intervals differ,
    a first interval LONGER than the cadence, or a short one no
    metadata declares.
    """
    steps = [int((later - earlier).total_seconds())
             for earlier, later in zip(times, times[1:])]
    later = sorted(set(steps[1:]))
    if len(later) > 1:
        raise ValueError(
            "Times records do not step by one uniform forcing interval: "
            f"{sorted(set(steps))} seconds")
    if later:
        cadence = later[0]
    elif next_times:
        cadence = int((next_times[-1] - times[-1]).total_seconds())
    else:
        cadence = steps[0]
    if steps[0] == cadence:
        return cadence, None
    declared = bool(next_times) and (
        (len(next_times) == len(times) and next_times[0] == times[1])
        or len(next_times) == 1)
    if not (0 < steps[0] < cadence) or not declared:
        raise ValueError(
            "Times records do not step by one uniform forcing interval: "
            f"{sorted(set(steps))} seconds (a first record covering less "
            "than the cadence is read as a start inside an interval only "
            "when its nextbdytime declares it)")
    return cadence, steps[0]


def wrfbdy_coverage(path: str | Path) -> BoundaryCoverage:
    """Read a ``wrfbdy``'s time coverage without decoding a boundary table.

    This is the instrument behind "run length is capped by wrfbdy".  It
    opens the file for its ``Times`` records, its ``bdy_width`` and its
    ``nextbdytime`` metadata only -- no side table is touched -- so a run
    whose length outstrips its forcing is refused at the front door, in
    milliseconds, rather than after every boundary array has been
    decoded and uploaded.
    """
    path = Path(path)
    with netcdf_bridge.open_dataset(path) as dataset:
        if "Times" not in dataset.variables:
            raise ConfigurationRefused(f"{path} has no Times variable")
        if "bdy_width" not in dataset.dimensions:
            raise ConfigurationRefused(f"{path} has no bdy_width dimension")
        width = len(dataset.dimensions["bdy_width"])
        times = _decode_times(dataset.variables["Times"])
        if not times:
            raise ConfigurationRefused(f"{path} has no records")
        if any(later <= earlier for earlier, later in zip(times, times[1:])):
            raise ConfigurationRefused(f"{path} records must increase")
        next_name = "md___nextbdytimee_x_t_d_o_m_a_i_n_m_e_t_a_data_"
        next_times = (_decode_times(dataset.variables[next_name])
                      if next_name in dataset.variables else ())
    lead = None
    if len(times) > 1:
        try:
            interval, lead = lead_interval_seconds(times, next_times)
        except ValueError as error:
            raise ConfigurationRefused(f"{path} {error}") from None
    elif next_times:
        # real.exe's standard single-interval product declares its own
        # cadence in the nextbdytime metadata, and that is the only place
        # a one-record file states it.
        interval = int((next_times[-1] - times[0]).total_seconds())
    else:
        raise ConfigurationRefused(
            f"{path} carries a single Times record and no nextbdytime "
            "metadata, so the forcing interval it covers cannot be "
            "established; a real.exe wrfbdy always writes one")
    if interval <= 0:
        raise ConfigurationRefused(
            f"{path} declares a non-positive forcing interval of "
            f"{interval} seconds")
    return BoundaryCoverage(
        path=path, times=times, forcing_interval_seconds=interval,
        spec_bdy_width=width,
        coverage_seconds=float(
            (times[-1] - times[0]).total_seconds() + interval),
        lead_interval_seconds=lead)


def check_boundary_coverage(coverage: BoundaryCoverage,
                            run_seconds: float, *, source: str) -> None:
    """Refuse, BEFORE any integration, a run longer than its forcing."""
    if 0.0 < float(run_seconds) <= coverage.coverage_seconds:
        return
    raise ConfigurationRefused(
        f"{source} requests a run of {float(run_seconds):.0f} s but "
        f"{coverage.path} carries only {coverage.coverage_seconds:.0f} s "
        f"of lateral forcing ({len(coverage.times)} record(s) from "
        f"{coverage.start:%Y-%m-%d_%H:%M:%S} at a "
        f"{coverage.forcing_interval_seconds}-second cadence, valid "
        f"through {coverage.end:%Y-%m-%d_%H:%M:%S}).  Entering at "
        "wrfinput caps run length at wrfbdy coverage: shorten the run to "
        f"at most {coverage.coverage_seconds:.0f} s, or re-run real.exe "
        "over a longer window.")


#: What nest relocation needs that a ``wrfinput`` handoff does not carry.
#:
#: VERIFIED 2026-09-04 against woof/ingest/relocation_init.py,
#: woof/ingest/nest_init.py and woof/core/nest_relocation.py: none of
#: the three so much as mentions ``wrfinput`` or ``wrfbdy``.  A relocated
#: child's statics are rebuilt for the new footprint from the nest's own
#: static source at nest resolution and its atmosphere is filled from the
#: LIVE PARENT, so nothing on that path reopens an initial-condition file
#: after t=0 and the capability survives this entry point intact.
#:
#: It is not free, though, and the one condition is structural:
#: :func:`woof.ingest.relocation_init.real_relocation_initializer`
#: requires EXACTLY ONE statics source -- a geography ``catalog`` or a
#: sealed-corridor ``statics_builder`` -- and refuses with neither.  A
#: wrfinput handoff supplies neither, because real.exe's statics are baked
#: into the wrfinput at the ORIGINAL footprint and say nothing about
#: ground the nest has not stood on yet.  So a wrfinput-entry run that
#: wants a moving nest must additionally declare a geography root.
RELOCATION_REQUIREMENT = (
    "nest relocation needs additional static coverage: the relocation path "
    "rebuilds a moved nest's statics from the geography source at nest "
    "resolution and takes its atmosphere from the live parent, and never "
    "reopens wrfinput/wrfbdy after t=0.  It does need a statics source of "
    "its own -- real_relocation_initializer takes exactly one of a "
    "geography catalog or a sealed statics_builder and refuses with "
    "neither -- and a wrfinput carries statics only for the footprint "
    "real.exe wrote, so a moving-nest run entering here must also declare "
    "a geography root. This adapter does not yet bind that future coverage."
)


def format_scheme_matrix() -> str:
    """The accepted/refused scheme matrix, for the door's own --help."""
    from woof.config import LAND_SURFACE_SCHEMES

    lines = ["  microphysics (MP_PHYSICS) restored:"]
    for value in sorted(SUPPORTED_MICROPHYSICS):
        lines.append(f"    {value:>3d}  {SUPPORTED_MICROPHYSICS[value]}")
    lines.append("  microphysics refused by name (with a remedy):")
    lines.append("    " + ", ".join(
        str(value) for value in sorted(UNSUPPORTED_MICROPHYSICS)))
    lines.append("  land surface (SF_SURFACE_PHYSICS) restored:")
    for value in sorted(LAND_SURFACE_SCHEMES):
        name, soil = WRF_LAND_SURFACE_SCHEMES[value]
        lines.append(f"    {value:>3d}  {name} ({soil} soil layers)")
    lines.append("  land surface refused by name (with a remedy):")
    lines.append("    " + ", ".join(
        f"{value} ({WRF_LAND_SURFACE_SCHEMES[value][0]})"
        for value in sorted(set(WRF_LAND_SURFACE_SCHEMES)
                            - set(LAND_SURFACE_SCHEMES))))
    return "\n".join(lines)


__all__ = [
    "BoundaryCoverage", "RELOCATION_REQUIREMENT", "RestoredDomain",
    "SUPPORTED_MICROPHYSICS", "UNSUPPORTED_MICROPHYSICS",
    # PUBLIC because the N5S verification battery reads this door's
    # moisture contract instead of keeping a fork of it: the fork's copy
    # stopped at mp=50 and refused Milbrandt-Yau by hard-coded literal
    # while the door had read the scheme in (audit R-019).
    "active_moisture_inventory", "active_moisture_map",
    "WRF_LAND_SURFACE_SCHEMES", "WrfinputMetadata",
    "check_boundary_coverage", "check_grid_agreement",
    "check_supported_schemes", "format_scheme_matrix", "read_wrfbdy",
    "read_wrfinput", "read_wrfinput_metadata", "restore_domain_state",
    "supported_land_surface_sentence", "supported_microphysics_sentence",
    "unsupported_land_surface_refusal", "unsupported_microphysics_refusal",
    "wrfbdy_coverage",
]
