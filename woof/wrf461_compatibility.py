"""Declarative WRF v4.6.1 physics-combination authority.

The tables in this module are a transcription of WRF v4.6.1 commit
``d66e442fccc04111067e29274c9f9eaccc3cef28``.  They deliberately separate
WRF's verdict from WOOF implementation readiness: a WRF-legal tuple may
still have a named WOOF structural blocker, but WOOF must never describe
that blocker as a WRF incompatibility.

Only the schemes ported by this release are represented.  Keeping the
cross-product dimensions and citations here makes admission a table lookup,
not a collection of subtly different front-door conditionals.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import product
from types import MappingProxyType
from typing import Iterator, Mapping


WRF_VERSION = "v4.6.1"
WRF_COMMIT = "d66e442fccc04111067e29274c9f9eaccc3cef28"


class WRFVerdict(str, Enum):
    """WRF's disposition for one represented physics tuple."""

    LEGAL = "legal"
    FATAL = "fatal-in-physics-init"
    LEGAL_RECONFIGURED = "legal-with-silent-reconfiguration"
    NOT_EXPRESSIBLE = "not-expressible-in-wrf-v4.6.1"


@dataclass(frozen=True)
class WRFCitation:
    """One source anchor carried by an authority-table cell."""

    path: str
    lines: str
    law: str

    @property
    def anchor(self) -> str:
        return f"{self.path}:{self.lines}"


@dataclass(frozen=True)
class WRFCompatibilityCell:
    """One fully cited cell of the represented cross-product."""

    mp_physics: int
    bl_pbl_physics: int
    sf_sfclay_physics: int
    sf_surface_physics: int
    radiation: str
    cu_physics: int
    verdict: WRFVerdict
    citations: tuple[WRFCitation, ...]
    silent_reconfiguration: str | None = None


# The microphysics axis carries every selector the registry publishes as
# implemented, because this matrix's verdict does not depend on it: a cell's
# verdict comes from the PBL/surface-layer pair, the radiation label and
# sf_surface_physics (compatibility_cell below), and mp_physics contributes
# only its Registry citation.  Carrying six of the ten therefore refused
# mp_physics 0, 9, 16 and 50 with "tuple axis is outside the represented WRF
# v4.6.1 matrix" -- a message that reads as an incompatibility for four
# schemes WRF v4.6.1 itself declares and this tree runs (audit R-012).  Each
# added row is an enumerated admission with WRF's own package line, exactly
# as the Shin-Hong and Grell-Freitas rows below are; adding them scales every
# verdict count by 10/6 and the pinned counts in
# tests/test_wrf461_compatibility.py were re-measured off the enlarged
# matrix rather than projected.
MP_OPTIONS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 18, 24, 26, 28, 30, 38, 40, 50, 51, 52, 53, 55, 56, 95, 97, 106)
# 11 (Shin-Hong) joined the PBL axis with the Shin-Hong port: an implemented
# scheme with a certified oracle (max ULP 0 CPU authority against the
# byte-frozen WRF v4.6.1 module_bl_shinhong.F,
# tests/test_shinhong_wrf461_parity.py) joins the matrix, and the matrix
# claims exactly what WRF's own init enforces -- the SHINHONGSCHEME case at
# phys/module_physics_init.F:3702-3704.  This is an enumerated admission
# row, not a relaxation: every (11, sfclay) cell below carries WRF's own
# verdict and citation.
#
# 16 (EEPS) joined with the EEPS port (woof/core/kernels/eeps.cu, max ULP
# 0 against WRF v4.6.1 module_bl_eepsilon.F, tests/test_eeps_wrf461_parity
# .py) on the same terms: its EEPSSCHEME init arm at
# phys/module_physics_init.F:3871-3873 fatals unless isfc is 1, 2, 5 or 91,
# and each (16, sfclay) cell below carries that verdict.
PBL_OPTIONS = (0, 1, 5, 7, 8, 11, 16, 99)
# Runtime breakage beside the WRF law, selected as scheme metadata.
PBL_RUNTIME_BREAKAGE = MappingProxyType({
    16: ("EEPS reads UST, RMOL, WSPD, HFX, QFX and QSFC from its "
         "surface layer and divides by WSPD. With the surface layer off "
         "those fields remain the allocated zeros, so its turbulent lower "
         "boundary would divide by zero (module_bl_eepsilon.F:575,589)."),
})
SURFACE_LAYER_OPTIONS = (0, 1, 3, 5, 7, 91)
# 1 (five-layer thermal diffusion) joined with the slab port: an
# implemented scheme graded word for word against the byte-unmodified
# phys/module_sf_slab.F (tests/test_slab_wrf461_oracle.py).  The land axis
# is verdict-independent except for sf_surface_physics=0's soil-layer
# reconfiguration, so the no-LSM reconfiguration slice stays fixed while the other axes grow.
LAND_SURFACE_OPTIONS = (0, 1, 2, 3, 4, 7, 8)
RADIATION_OPTIONS = (
    "off",
    "dudhia-shortwave",
    "rrtmg-rte-rrtmgp",
    "rrtmg-legacy",
    "analytic",
)
# 3 (Grell-Freitas) joined the cumulus axis with the GF port: an
# implemented scheme with a certified oracle (woof/core/kernels/gf.cu,
# max ULP 0 at the GFDRV boundary on the 208 driver-exact columns).  The
# cumulus axis is verdict-independent in WRF v4.6.1 -- no SELECT CASE in
# phys/module_physics_init.F conditions another component's legality on
# the cumulus choice -- so admitting 3 scales every verdict count by 3/2,
# and the pinned counts in tests/test_wrf461_compatibility.py were
# re-measured off the enlarged matrix rather than projected.
#
# 16 (New Tiedtke) joined on the same terms with the New Tiedtke port.
# It was the axis's own drift: this module's header says only the ported
# schemes are represented, woof.config.CU_SCHEMES has admitted 16 since
# the port landed, and the registry this module generates already
# published cu_physics=16 as an implemented option while the matrix
# dimension it also publishes said (0, 1, 3) -- one artifact
# contradicting itself.  Nothing on a run path reads the cumulus axis, so
# the drift was latent rather than a live refusal; it is closed here, and
# test_authority_agreement now holds every axis of this transcription
# equal to the selectable set so the next one cannot be latent either.
#
# 5 (Grell 3D) joined with the G3 port (woof/core/kernels/g3.cu, 0 ULP
# under the strict arm against tools/g3_wrf461_oracle), on the same
# verdict-independent terms; 96 (New SAS with NSCV) joined with its port,
# on the same terms.
# CAM ZM (7) keeps its direct admission law in AXIS_EXCLUSIONS below.
CUMULUS_OPTIONS = (0, 1, 2, 3, 5, 6, 10, 11, 14, 16, 93, 94, 95, 96, 99)

#: Values ``woof.config`` lets a user select that this transcription does
#: NOT represent, each with the reason it is absent.  The table exists
#: because the cumulus axis drifted silently: cu_physics=16 was ported,
#: selectable and published by the registry this module generates while the
#: matrix dimension said (0, 1, 3), and nothing compared the two.  An
#: axis's represented set plus its rows here must equal the selectable set
#: exactly (``tests/test_authority_agreement.py``), so a newly ported
#: scheme has to do one of two things -- join the axis with its citation,
#: or say here why it cannot -- and neither can be forgotten.
#:
#: A blank reason is not a row.  These are not refusals (nothing on a run
#: path consults this module for any of these values); they state what the
#: transcription's silence means, which is what stops the next reader from
#: mistaking absence for illegality.
AXIS_EXCLUSIONS = MappingProxyType({
    ("cu_physics", 7): "CAM ZM requires MYJ/Eta or CAM UW at get_tpert; those PBL choices are outside this physics_init matrix.",
    ("mp_physics", 900): (
        "TEMPO is a WOOF named scheme (woof.microphysics_schemes, id 900, "
        "outside WRF's mp_physics range): WRF v4.6.1 has no TEMPO option, so "
        "there is no WRF surface-layer or PBL pairing law to transcribe."),
    ("bl_pbl_physics", 10): (
        "The TEMF bidirectional pairing is enforced from "
        "woof.config.PHYSICS_SELECTOR_COUPLINGS, using the WRF v4.6.1 "
        "share/module_check_a_mundo.F:2294-2299 law."),
    ("sf_sfclay_physics", 10): (
        "The other side of the same TEMF bidirectional coupling row."),

    ("bl_pbl_physics", 2): (
        "MYJ's WRF law is stronger than a matrix cell and is implemented "
        "directly: phys/module_physics_init.F:3770-3772 fatals the MYJ "
        "PBL with any surface layer but the Eta one, and "
        "woof.config.validate_myj_pairing raises exactly that, in both "
        "directions, before this module is consulted. A row here would be "
        "a second authority over a question that already has one."),
    ("bl_pbl_physics", 3): (
        "GFS is admitted directly by woof.config.validate_gfs_pbl_config: "
        "WRF module_physics_init.F:3742-3743 requires isfc=2, and the "
        "implemented SF_GFS layer supplies its complete stability boundary. "
        "The Eta layer has no GZ1OZ0/PSIM/PSIH and is refused."),
    ("bl_pbl_physics", 4): (
        "QNSE's WRF law is stronger than a matrix cell and is implemented "
        "directly: phys/module_physics_init.F:3790-3791 fatals the QNSE "
        "PBL with any surface layer but the QNSE one, and "
        "woof.config.validate_qnse_config raises exactly that before this "
        "module is consulted. A row here would be a second authority over "
        "a question that already has one."),
    ("bl_pbl_physics", 9): (
        "the UW moist-turbulence PBL is ported from WRF v4.7.1 "
        "(module_bl_camuwpbl_driver.F), not the v4.6.1 this matrix "
        "transcribes, and WRF has no surface-layer law for it to "
        "transcribe: its init case (module_physics_init.F:3825-3832) "
        "carries only the BEP/BEM urban fatal, which "
        "woof.config.validate_uwpbl_config implements, and the Eta-layer "
        "cell is refused by woof.config.validate_myj_pairing for its own "
        "reason (the TKE_MYJ scan). A row here would be a second "
        "authority over questions that already have one."),
    ("bl_pbl_physics", 901): ("CCPP TKE-EDMF has no WRF selector counterpart; its admission follows its own registry metadata."),
    ("bl_pbl_physics", 12): (
        "GBM's WRF law is one surface-layer fatal, stronger than a matrix "
        "cell and implemented directly: phys/module_physics_init.F:3892-3893 "
        "refuses GBMPBLSCHEME unless the surface layer set isfc = 1 "
        "(sf_sfclay_physics 1 or 91), and woof.config.validate_gbmpbl_config "
        "raises exactly that before this module is consulted. A row here "
        "would be a second authority over a question that already has one."),

    ("bl_pbl_physics", 17): (
        "WRF v4.6.1's bl_init has no KEPSSCHEME case at all "
        "(module_physics_init.F pbl_select falls through to CASE DEFAULT), "
        "so WRF places no surface-layer or land-surface law on the "
        "K-epsilon PBL for this matrix to transcribe. WOOF's own pairing "
        "requirements (a surface layer that publishes PSIM/PSIH, no "
        "gwd_opt) are woof.config.validate_keps_config, each naming what "
        "it prevents. A row here would be a second authority over "
        "questions that already have one."),
    ("bl_pbl_physics", 900): (
        "SASE is a WOOF closure with no WRF counterpart: WRF v4.6.1 "
        "registers no package at bl_pbl_physics=900, so there is no "
        "verdict to transcribe and a row would be an invention. Its "
        "admission is stated on its own registry option and in "
        "woof.config.validate_sase_config."),
    ("sf_sfclay_physics", 4): (
        "The QNSE surface layer is the other half of bl_pbl_physics=4. "
        "Its admission is handled by woof.config.validate_qnse_config, "
        "which binds the carried TKE column and AKHS/AKMS boundary state. "
        "See phys/module_physics_init.F:3171,3790-3791 and the PBL exclusion row."),
    ("sf_sfclay_physics", 2): (
        "the Eta surface layer is the other half of the MYJ pairing above "
        "and is refused or admitted by the same implemented law; see the "
        "bl_pbl_physics=2 row."),
})


MATRIX_CELL_COUNT = (
    len(MP_OPTIONS)
    * len(PBL_OPTIONS)
    * len(SURFACE_LAYER_OPTIONS)
    * len(LAND_SURFACE_OPTIONS)
    * len(RADIATION_OPTIONS)
    * len(CUMULUS_OPTIONS)
)


_MP_CITATION = MappingProxyType({
    96: WRFCitation("Registry/Registry.EM_COMMON", "3047", "the madwrf_mp package binds mp_physics=96"),
    11: WRFCitation("Registry/Registry.EM_COMMON", "3032", "cammgmpscheme binds mp_physics=11 and its four number moments"),
    2: WRFCitation("Registry/Registry.EM_COMMON", "3016",
        "the linscheme package binds mp_physics=2 and carries six masses"),
    30: WRFCitation("Registry/registry.sbm", "318",
        "the fast spectral bin package binds mp_physics=30"),
    95: WRFCitation("Registry/Registry.EM_COMMON", "3045",
        "WRF declares the mp_physics=95 package and its own carried species"),
    0: WRFCitation(
        "Registry/Registry.EM_COMMON", "3014",
        "the passiveqv package binds mp_physics=0 and allocates qv alone"),
    1: WRFCitation(
        "Registry/Registry.EM_COMMON", "3015",
        "the Kessler package binds mp_physics=1 and allocates qv/qc/qr"),
    5: WRFCitation(
        "Registry/Registry.EM_COMMON", "3019",
        "the fer_mp_hires package binds mp_physics=5"),
    3: WRFCitation(
        "Registry/Registry.EM_COMMON", "3017",
        "the wsm3scheme package binds mp_physics=3 and allocates qv/qc/qr"),
    4: WRFCitation("Registry/Registry.EM_COMMON", "3018", "the WSM5 package binds mp_physics=4 and qv/qc/qr/qi/qs"),
    7: WRFCitation(
        "Registry/Registry.EM_COMMON", "3023",
        "the nuwrf4icescheme package binds mp_physics=7"),
    15: WRFCitation(
        "Registry/Registry.EM_COMMON", "3020",
        "the fer_mp_hires_advect package binds mp_physics=15"),
    6: WRFCitation(
        "Registry/Registry.EM_COMMON", "3021",
        "the WSM6 package binds mp_physics=6"),
    8: WRFCitation(
        "Registry/Registry.EM_COMMON", "3024",
        "the Thompson package binds mp_physics=8"),
    9: WRFCitation(
        "Registry/Registry.EM_COMMON", "3025",
        "the milbrandt2mom package binds mp_physics=9"),
    10: WRFCitation(
        "Registry/Registry.EM_COMMON", "3026",
        "the Morrison two-moment package binds mp_physics=10"),
    13: WRFCitation(
        "Registry/Registry.EM_COMMON", "3029",
        "the sbu_ylinscheme package binds mp_physics=13"),
    14: WRFCitation(
        "Registry/Registry.EM_COMMON", "3030",
        "the wdm5scheme package binds mp_physics=14"),
    16: WRFCitation(
        "Registry/Registry.EM_COMMON", "3031",
        "the wdm6scheme package binds mp_physics=16"),
    106: WRFCitation(
        "Registry/Registry.EM_COMMON", "3022",
        "the wsm6rscheme package binds mp_physics=106 and allocates "
        "qv/qc/qr/qi/qs/qg"),
    18: WRFCitation(
        "Registry/Registry.EM_COMMON", "3033",
        "the NSSL two-moment package binds mp_physics=18"),
    24: WRFCitation(
        "Registry/Registry.EM_COMMON", "3034",
        "the wsm7scheme package binds mp_physics=24"),
    26: WRFCitation(
        "Registry/Registry.EM_COMMON", "3035",
        "the wdm7scheme package binds mp_physics=26"),
    28: WRFCitation(
        "Registry/Registry.EM_COMMON", "3036",
        "the aerosol-aware Thompson package binds mp_physics=28"),
    52: WRFCitation("Registry/Registry.EM_COMMON", "3040",
                    "the P3_2CATEGORY package binds mp_physics=52"),
    38: WRFCitation(
        "Registry/Registry.EM_COMMON", "3037",
        "the thompsongh package binds mp_physics=38 and carries qng/qvolg"),
    40: WRFCitation(
        "Registry/Registry.EM_COMMON", "3042",
        "the morr_tm_aero package binds mp_physics=40"),
    50: WRFCitation(
        "Registry/Registry.EM_COMMON", "3038",
        "the P3_1CATEGORY package binds mp_physics=50"),
    97: WRFCitation(
        "Registry/Registry.EM_COMMON", "3046",
        "the gsfcgcescheme package binds mp_physics=97"),
    51: WRFCitation(
        "Registry/Registry.EM_COMMON", "3039",
        "the P3_1CATEGORY_NC package binds mp_physics=51"),
    53: WRFCitation(
        "Registry/Registry.EM_COMMON", "3041",
        "the P3_1CAT_3MOM package binds mp_physics=53"),
    55: WRFCitation(
        "Registry/Registry.EM_COMMON", "3043",
        "the jensen_ishmael package binds mp_physics=55 and carries "
        "three ice habits with their number, volume and aspect moments"),
    56: WRFCitation(
        "Registry/Registry.EM_COMMON", "3044",
        "the ntu package binds mp_physics=56"),
})

_LAND_SURFACE_CITATION = MappingProxyType({
    0: WRFCitation(
        "Registry/Registry.EM_COMMON", "3144",
        "the no-LSM package binds sf_surface_physics=0"),
    1: WRFCitation(
        "Registry/Registry.EM_COMMON", "3145",
        "the slabscheme package binds sf_surface_physics=1"),
    2: WRFCitation(
        "Registry/Registry.EM_COMMON", "3146",
        "the Noah LSM package binds sf_surface_physics=2"),
    3: WRFCitation(
        "Registry/Registry.EM_COMMON", "3147",
        "the RUC LSM package binds sf_surface_physics=3"),
    4: WRFCitation(
        "Registry/Registry.EM_COMMON", "3149",
        "the Noah-MP package binds sf_surface_physics=4"),
    7: WRFCitation(
        "Registry/Registry.EM_COMMON", "3161",
        "the Pleim-Xiu package binds sf_surface_physics=7"),
    8: WRFCitation("Registry/Registry.EM_COMMON", "3152",
                   "the ssibscheme package binds sf_surface_physics=8"),
})

_CUMULUS_CITATION = MappingProxyType({
    94: WRFCitation("Registry/Registry.EM_COMMON", "3203", "sasscheme binds cu_physics=94"),
    0: WRFCitation(
        "Registry/Registry.EM_COMMON", "3189",
        "the no-cumulus package binds cu_physics=0"),
    1: WRFCitation(
        "Registry/Registry.EM_COMMON", "3190",
        "the Kain-Fritsch package binds cu_physics=1"),
    2: WRFCitation(
        "Registry/Registry.EM_COMMON", "3191",
        "the bmjscheme package binds cu_physics=2"),
    3: WRFCitation(
        "Registry/Registry.EM_COMMON", "3192",
        "the gfscheme package binds cu_physics=3"),
    14: WRFCitation(
        "Registry/Registry.EM_COMMON", "3199",
        "the ksasscheme package binds cu_physics=14"),
    5: WRFCitation(
        "Registry/Registry.EM_COMMON", "3194",
        "the g3scheme package binds cu_physics=5"),
    6: WRFCitation(
        "Registry/Registry.EM_COMMON", "3195",
        "the tiedtkescheme package binds cu_physics=6"),
    11: WRFCitation(
        "Registry/Registry.EM_COMMON", "3198",
        "the mskfscheme package binds cu_physics=11"),
    16: WRFCitation(
        "Registry/Registry.EM_COMMON", "3201",
        "the ntiedtkescheme package binds cu_physics=16"),
    10: WRFCitation(
        "Registry/Registry.EM_COMMON", "3197",
        "the kfcupscheme package binds cu_physics=10"),
    95: WRFCitation(
        "Registry/Registry.EM_COMMON", "3204",
        "the osasscheme package binds cu_physics=95"),
    96: WRFCitation(
        "Registry/Registry.EM_COMMON", "3200",
        "the nsasscheme package binds cu_physics=96"),
    93: WRFCitation(
        "Registry/Registry.EM_COMMON", "3202",
        "the gdscheme package binds cu_physics=93"),
    99: WRFCitation(
        "Registry/Registry.EM_COMMON", "3205",
        "the kfscheme package binds cu_physics=99 (the previous "
        "Kain-Fritsch, phys/module_cu_kf.F) and allocates w0avg"),
})

_RADIATION_CITATION = MappingProxyType({
    "off": WRFCitation(
        "phys/module_physics_init.F", "2230-2345,2348-2445",
        "the LW and SW initialization SELECT CASE blocks accept zero through "
        "their no-action default paths"),
    "dudhia-shortwave": WRFCitation(
        "Registry/Registry.EM_COMMON", "3117",
        "the Dudhia shortwave package binds ra_sw_physics=1; longwave zero "
        "uses the no-action initialization path"),
    "rrtmg-rte-rrtmgp": WRFCitation(
        "Registry/Registry.EM_COMMON", "3109,3120",
        "WRF's bundled RRTMG packages bind the 4/4 LW/SW selectors; WOOF's "
        "RTE+RRTMGP adapter is an explicitly receipted implementation of "
        "those same WRF selectors"),
    "rrtmg-legacy": WRFCitation(
        "Registry/Registry.EM_COMMON", "3109,3120",
        "WRF's bundled RRTMG packages bind the 4/4 LW/SW selectors; WOOF's "
        "legacy variant ports that implementation directly"),
    "analytic": WRFCitation(
        "Registry/Registry.EM_COMMON", "3107-3125",
        "WRF v4.6.1 registers no ra_lw_physics=90 or ra_sw_physics=90 "
        "package; analytic radiation is WOOF-specific"),
})

_SOIL_LAYER_CITATION = WRFCitation(
    "share/module_check_a_mundo.F", "3548-3563",
    "set_physics_rconfigs silently sets the WRF soil-layer count: no-LSM=5, "
    "Noah=4, Noah-MP=4, and an invalid RUC count=6 (RUC 6 or 9 is retained)")


def _pair(
    verdict: WRFVerdict, path: str, lines: str, law: str
) -> tuple[WRFVerdict, WRFCitation]:
    return verdict, WRFCitation(path, lines, law)


# This is the complete PBL/surface-layer compatibility law intersecting the
# ported selectors.  PBL-off executes no PBL initializer and therefore accepts
# every represented surface layer.  YSU requires isfc=1, which revised/classic
# MM5 set.  MYNN PBL accepts isfc in {5,1,2}; the represented values 5,1,91
# initialize isfc to 5,1,1 respectively.  Shin-Hong requires isfc=1 exactly
# as YSU does, through its own SELECT CASE arm: the SHINHONGSCHEME case at
# phys/module_physics_init.F:3702-3704 fatals with 'Use sf_sfclay_physics= 1
# or 91 for this pbl option' unless the initialized surface-layer class is
# isfc=1.  ACM2 accepts isfc 1 or 7 through its own ACMPBLSCHEME arm
# (phys/module_physics_init.F:3728-3730), so among the represented surface
# layers it takes the revised and classic MM5 ones.
PBL_SURFACE_LAYER_AUTHORITY: Mapping[
    tuple[int, int], tuple[WRFVerdict, WRFCitation]
] = MappingProxyType({
    (0, 89): _pair(
        WRFVerdict.LEGAL, "phys/module_physics_init.F", "3140-3237,3697-3915",
        "the ideal SCM surface leaves isfc unassigned; PBL-off selects no "
        "PBL initializer and never reads it"),
    (0, 0): _pair(
        WRFVerdict.LEGAL, "phys/module_physics_init.F", "3697-3915",
        "bl_pbl_physics=0 selects no PBL initialization branch"),
    (0, 1): _pair(
        WRFVerdict.LEGAL, "phys/module_physics_init.F", "3143-3152,3697-3915",
        "revised MM5 sets isfc=1; PBL-off selects no PBL initializer"),
    (0, 3): _pair(
        WRFVerdict.LEGAL, "phys/module_physics_init.F", "3186-3199,3697-3915",
        "the GFS surface layer sets isfc=2; PBL-off selects no PBL "
        "initializer"),
    (0, 5): _pair(
        WRFVerdict.LEGAL, "phys/module_physics_init.F", "3213-3219,3697-3915",
        "MYNN surface sets isfc=5; PBL-off selects no PBL initializer"),
    (0, 91): _pair(
        WRFVerdict.LEGAL, "phys/module_physics_init.F", "3140-3142,3697-3915",
        "classic MM5 sets isfc=1; PBL-off selects no PBL initializer"),
    (1, 0): _pair(
        WRFVerdict.FATAL, "phys/module_physics_init.F", "3699-3701",
        "YSU fatals unless the initialized surface-layer class is isfc=1"),
    (1, 1): _pair(
        WRFVerdict.LEGAL, "phys/module_physics_init.F", "3143-3152,3699-3701",
        "revised MM5 sets isfc=1, satisfying YSU"),
    (1, 3): _pair(
        WRFVerdict.FATAL, "phys/module_physics_init.F", "3186-3199,3699-3701",
        "the GFS surface layer sets isfc=2, and YSU fatals unless isfc=1"),
    (1, 5): _pair(
        WRFVerdict.FATAL, "phys/module_physics_init.F", "3213-3219,3699-3701",
        "MYNN surface sets isfc=5, and YSU fatals unless isfc=1"),
    (1, 91): _pair(
        WRFVerdict.LEGAL, "phys/module_physics_init.F", "3140-3142,3699-3701",
        "classic MM5 sets isfc=1, satisfying YSU"),
    (5, 0): _pair(
        WRFVerdict.FATAL, "phys/module_physics_init.F", "3837-3839",
        "MYNN PBL fatals unless the initialized surface class is 5, 1, or 2"),
    (5, 1): _pair(
        WRFVerdict.LEGAL, "phys/module_physics_init.F", "3143-3152,3837-3839",
        "revised MM5 sets isfc=1, which MYNN PBL explicitly accepts"),
    (5, 3): _pair(
        WRFVerdict.LEGAL, "phys/module_physics_init.F", "3186-3199,3837-3839",
        "the GFS surface layer sets isfc=2, which MYNN PBL explicitly "
        "accepts"),
    (5, 5): _pair(
        WRFVerdict.LEGAL, "phys/module_physics_init.F", "3213-3219,3837-3839",
        "MYNN surface sets isfc=5, which MYNN PBL explicitly accepts"),
    (5, 91): _pair(
        WRFVerdict.LEGAL, "phys/module_physics_init.F", "3140-3142,3837-3839",
        "classic MM5 sets isfc=1, which MYNN PBL explicitly accepts"),
    # Shin-Hong (bl_pbl_physics=11): the SHINHONGSCHEME case, NOT a copy of
    # YSU's cells -- WRF's own arm at phys/module_physics_init.F:3702-3704
    # fatals with 'Use sf_sfclay_physics= 1 or 91 for this pbl option'
    # unless isfc=1.
    # BouLac (bl_pbl_physics=8): the BOULACSCHEME case at
    # phys/module_physics_init.F:3814-3816 fatals with 'use sfclay or myjsfc
    # scheme for this pbl option' unless isfc is 1 or 2.  The MM5 layers set
    # isfc=1; the MYNN layer sets isfc=5; with the slot off isfc is unset.
    (8, 0): _pair(
        WRFVerdict.FATAL, "phys/module_physics_init.F", "3814-3816",
        "BouLac fatals unless the initialized surface-layer class is isfc=1 "
        "or 2: 'use sfclay or myjsfc scheme for this pbl option'"),
    (8, 1): _pair(
        WRFVerdict.LEGAL, "phys/module_physics_init.F", "3143-3152,3814-3816",
        "revised MM5 sets isfc=1, satisfying BouLac"),
    (8, 3): _pair(
        WRFVerdict.LEGAL, "phys/module_physics_init.F", "3186-3199,3814-3816",
        "the GFS surface layer sets isfc=2, satisfying BouLac"),
    (8, 5): _pair(
        WRFVerdict.FATAL, "phys/module_physics_init.F", "3213-3219,3814-3816",
        "MYNN surface sets isfc=5, and BouLac fatals unless isfc is 1 or 2"),
    (8, 7): _pair(
        WRFVerdict.FATAL, "phys/module_physics_init.F", "3153-3155,3814-3816",
        "Pleim-Xiu sets isfc=7, and BouLac fatals unless isfc is 1 or 2"),
    (8, 91): _pair(
        WRFVerdict.LEGAL, "phys/module_physics_init.F", "3140-3142,3814-3816",
        "classic MM5 sets isfc=1, satisfying BouLac"),
    (11, 0): _pair(
        WRFVerdict.FATAL, "phys/module_physics_init.F", "3702-3704",
        "Shin-Hong fatals unless the initialized surface-layer class is "
        "isfc=1: 'Use sf_sfclay_physics= 1 or 91 for this pbl option'"),
    (11, 1): _pair(
        WRFVerdict.LEGAL, "phys/module_physics_init.F", "3143-3152,3702-3704",
        "revised MM5 sets isfc=1, satisfying Shin-Hong"),
    (11, 3): _pair(
        WRFVerdict.FATAL, "phys/module_physics_init.F", "3186-3199,3702-3704",
        "the GFS surface layer sets isfc=2, and Shin-Hong fatals unless "
        "isfc=1"),
    (11, 5): _pair(
        WRFVerdict.FATAL, "phys/module_physics_init.F", "3213-3219,3702-3704",
        "MYNN surface sets isfc=5, and Shin-Hong fatals unless isfc=1"),
    (11, 91): _pair(
        WRFVerdict.LEGAL, "phys/module_physics_init.F", "3140-3142,3702-3704",
        "classic MM5 sets isfc=1, satisfying Shin-Hong"),
    # EEPS (bl_pbl_physics=16): the EEPSSCHEME case at
    # phys/module_physics_init.F:3871-3873 fatals with 'use mynnsfc or
    # sfclay or myjsfc scheme for this pbl option' unless isfc is 1, 2, 5
    # or 91.
    (16, 0): _pair(
        WRFVerdict.FATAL, "phys/module_physics_init.F", "3871-3873",
        "EEPS fatals unless the initialized surface-layer class is isfc "
        "1, 2, 5 or 91: 'use mynnsfc or sfclay or myjsfc scheme for this "
        "pbl option'"),
    (16, 1): _pair(
        WRFVerdict.LEGAL, "phys/module_physics_init.F", "3143-3152,3871-3873",
        "revised MM5 sets isfc=1, which EEPS accepts"),
    (16, 3): _pair(
        WRFVerdict.LEGAL, "phys/module_physics_init.F", "3186-3199,3871-3873",
        "the GFS surface layer sets isfc=2, which EEPS accepts"),
    (16, 5): _pair(
        WRFVerdict.LEGAL, "phys/module_physics_init.F", "3213-3219,3871-3873",
        "MYNN surface sets isfc=5, which EEPS accepts"),
    (16, 7): _pair(
        WRFVerdict.FATAL, "phys/module_physics_init.F", "3153-3155,3871-3873",
        "Pleim-Xiu sets isfc=7, and EEPS fatals unless isfc is 1, 2, 5 or 91"),
    (16, 91): _pair(
        WRFVerdict.LEGAL, "phys/module_physics_init.F", "3140-3142,3871-3873",
        "classic MM5 sets isfc=1, which EEPS accepts"),
    # MRF (bl_pbl_physics=99): the MRFSCHEME case, WRF's own arm at
    # phys/module_physics_init.F:3715-3717, 'Use sf_sfclay_physics= 1 or 91
    # for this pbl option' unless isfc=1.
    (99, 0): _pair(
        WRFVerdict.FATAL, "phys/module_physics_init.F", "3715-3717",
        "MRF fatals unless the initialized surface-layer class is isfc=1: "
        "'Use sf_sfclay_physics= 1 or 91 for this pbl option'"),
    (99, 1): _pair(
        WRFVerdict.LEGAL, "phys/module_physics_init.F", "3143-3152,3715-3717",
        "revised MM5 sets isfc=1, satisfying MRF"),
    (99, 3): _pair(
        WRFVerdict.FATAL, "phys/module_physics_init.F", "3186-3199,3715-3717",
        "the GFS surface layer sets isfc=2, and MRF fatals unless isfc=1"),
    (99, 5): _pair(
        WRFVerdict.FATAL, "phys/module_physics_init.F", "3213-3219,3715-3717",
        "MYNN surface sets isfc=5, and MRF fatals unless isfc=1"),
    (99, 7): _pair(
        WRFVerdict.FATAL, "phys/module_physics_init.F", "3153-3155,3715-3717",
        "Pleim-Xiu sets isfc=7, and MRF fatals unless isfc=1"),
    (99, 91): _pair(
        WRFVerdict.LEGAL, "phys/module_physics_init.F", "3140-3142,3715-3717",
        "classic MM5 sets isfc=1, satisfying MRF"),

    # ACM2 (bl_pbl_physics=7): the ACMPBLSCHEME case at
    # phys/module_physics_init.F:3728-3730 fatals with 'use sfclay or pxsfc
    # scheme for this pbl option' unless isfc is 1 (the MM5 layers) or 7
    # (Pleim-Xiu, :3153-3155).
    (7, 0): _pair(
        WRFVerdict.FATAL, "phys/module_physics_init.F", "3728-3730",
        "ACM2 fatals unless the initialized surface-layer class is isfc=1 "
        "or 7: 'use sfclay or pxsfc scheme for this pbl option'"),
    (7, 1): _pair(
        WRFVerdict.LEGAL, "phys/module_physics_init.F", "3143-3152,3728-3730",
        "revised MM5 sets isfc=1, satisfying ACM2"),
    (7, 5): _pair(
        WRFVerdict.FATAL, "phys/module_physics_init.F", "3213-3219,3728-3730",
        "MYNN surface sets isfc=5, and ACM2 fatals unless isfc is 1 or 7"),
    (7, 3): _pair(
        WRFVerdict.FATAL, "phys/module_physics_init.F", "3186-3199,3728-3730",
        "GFS sets isfc=2, and ACM2 fatals unless isfc is 1 or 7"),
    (7, 7): _pair(
        WRFVerdict.LEGAL, "phys/module_physics_init.F", "3153-3155,3728-3730",
        "Pleim-Xiu sets isfc=7, satisfying ACM2"),
    (7, 91): _pair(
        WRFVerdict.LEGAL, "phys/module_physics_init.F", "3140-3142,3728-3730",
        "classic MM5 sets isfc=1, satisfying ACM2"),
    # Pleim-Xiu (sf_sfclay_physics=7): PXSFCSCHEME sets isfc=7
    # (phys/module_physics_init.F:3153-3155). PBL-off and CAMUW accept it;
    # CAMUW carries no isfc law (:3823-3828). WRF also admits ACM2
    # (:3728-3730), whose separate port does not belong to this lane.
    (0, 7): _pair(
        WRFVerdict.LEGAL, "phys/module_physics_init.F", "3153-3155,3697-3915",
        "Pleim-Xiu sets isfc=7; PBL-off selects no PBL initializer"),
    (1, 7): _pair(
        WRFVerdict.FATAL, "phys/module_physics_init.F", "3153-3155,3699-3701",
        "Pleim-Xiu sets isfc=7, and YSU fatals unless isfc=1"),
    (5, 7): _pair(
        WRFVerdict.FATAL, "phys/module_physics_init.F", "3153-3155,3837-3839",
        "Pleim-Xiu sets isfc=7, and MYNN PBL fatals unless the initialized "
        "surface class is 5, 1, or 2"),
    (11, 7): _pair(
        WRFVerdict.FATAL, "phys/module_physics_init.F", "3153-3155,3702-3704",
        "Pleim-Xiu sets isfc=7, and Shin-Hong fatals unless isfc=1: 'Use "
        "sf_sfclay_physics= 1 or 91 for this pbl option'"),
})


def pbl_surface_layer_verdict(
    bl_pbl_physics: int, sf_sfclay_physics: int
) -> tuple[WRFVerdict, WRFCitation]:
    """Return the exact WRF table entry for one represented pairing."""

    try:
        return PBL_SURFACE_LAYER_AUTHORITY[
            (int(bl_pbl_physics), int(sf_sfclay_physics))]
    except KeyError as exc:
        raise ValueError(
            "pair is outside the WRF v4.6.1 ported-set authority table: "
            f"bl_pbl_physics={bl_pbl_physics}, "
            f"sf_sfclay_physics={sf_sfclay_physics}") from exc


def compatibility_cell(
    *,
    mp_physics: int,
    bl_pbl_physics: int,
    sf_sfclay_physics: int,
    sf_surface_physics: int,
    radiation: str,
    cu_physics: int,
) -> WRFCompatibilityCell:
    """Build one cited cell without applying a WOOF structural gate."""

    pair_verdict, pair_citation = pbl_surface_layer_verdict(
        bl_pbl_physics, sf_sfclay_physics)
    try:
        citations = (
            _MP_CITATION[int(mp_physics)],
            pair_citation,
            _LAND_SURFACE_CITATION[int(sf_surface_physics)],
            _RADIATION_CITATION[str(radiation)],
            _CUMULUS_CITATION[int(cu_physics)],
            _SOIL_LAYER_CITATION,
        )
    except KeyError as exc:
        raise ValueError(
            f"tuple axis {exc.args[0]!r} is outside the represented "
            "WRF v4.6.1 matrix") from exc

    silent_reconfiguration = None
    if int(sf_surface_physics) == 8 and (
            int(bl_pbl_physics) != 1 or int(sf_sfclay_physics) not in (1, 91)
            or radiation in ("analytic", "dudhia-shortwave")):
        verdict = WRFVerdict.FATAL
        citations += (WRFCitation("phys/module_physics_init.F", "3542-3550",
                       "SSiB requires isfc=1, YSU and LW/SW in 1,3,4"),)
    elif pair_verdict is WRFVerdict.FATAL:
        verdict = WRFVerdict.FATAL
    elif radiation == "analytic":
        verdict = WRFVerdict.NOT_EXPRESSIBLE
    elif int(sf_surface_physics) == 0:
        # ArWen's no-LSM state deliberately carries an empty four-level soil
        # dimension for byte identity.  WRF overwrites the namelist value to
        # five even though its no-LSM package allocates no soil prognostics.
        verdict = WRFVerdict.LEGAL_RECONFIGURED
        silent_reconfiguration = "WRF sets num_soil_layers=5 for no-LSM"
    else:
        verdict = WRFVerdict.LEGAL

    return WRFCompatibilityCell(
        mp_physics=int(mp_physics),
        bl_pbl_physics=int(bl_pbl_physics),
        sf_sfclay_physics=int(sf_sfclay_physics),
        sf_surface_physics=int(sf_surface_physics),
        radiation=str(radiation),
        cu_physics=int(cu_physics),
        verdict=verdict,
        citations=citations,
        silent_reconfiguration=silent_reconfiguration,
    )


def iter_compatibility_matrix() -> Iterator[WRFCompatibilityCell]:
    """Yield the complete represented cross-product: :data:`MATRIX_CELL_COUNT`
    cells, which is the product of the six transcribed axes and is COMPUTED
    from them, never typed here.  Widening an axis (mp_physics=28, then
    bl_pbl_physics=11, then cu_physics=3 and 16) moves the count on its own;
    a hand-written total in this docstring went stale on two of those
    widenings before it was deleted."""

    for mp, pbl, sfclay, lsm, radiation, cumulus in product(
        MP_OPTIONS,
        PBL_OPTIONS,
        SURFACE_LAYER_OPTIONS,
        LAND_SURFACE_OPTIONS,
        RADIATION_OPTIONS,
        CUMULUS_OPTIONS,
    ):
        yield compatibility_cell(
            mp_physics=mp,
            bl_pbl_physics=pbl,
            sf_sfclay_physics=sfclay,
            sf_surface_physics=lsm,
            radiation=radiation,
            cu_physics=cumulus,
        )


__all__ = [
    "CUMULUS_OPTIONS",
    "LAND_SURFACE_OPTIONS",
    "MATRIX_CELL_COUNT",
    "MP_OPTIONS",
    "PBL_OPTIONS",
    "PBL_SURFACE_LAYER_AUTHORITY",
    "RADIATION_OPTIONS",
    "SURFACE_LAYER_OPTIONS",
    "WRF_COMMIT",
    "WRF_VERSION",
    "WRFCitation",
    "WRFCompatibilityCell",
    "WRFVerdict",
    "compatibility_cell",
    "iter_compatibility_matrix",
    "pbl_surface_layer_verdict",
]
