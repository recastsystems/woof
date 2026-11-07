"""Measure the physics composition space by walking it through the loader.

The question this answers is "is woof's physics hardcoded" -- can a user
compose microphysics, PBL, surface layer, land surface, radiation, cumulus
and turbulence freely, or does some engine lock pin them to the shipped
named suites?  It is answered by MEASUREMENT, not by reading code: every
combination below is written into a real experiment TOML and pushed
through :func:`woof.experiment.build_experiment` -- the one front door
``woof run``, ``woof go``, ``woof check``, both prepared runners and the
DA drivers reach the per-domain ``RunConfig`` through -- and the verdict
recorded is whatever that call did.

ACCEPTED means the loader returned an experiment AND the resolved
per-domain RunConfig carries every selector the file asked for, value for
value (an "accepted" that quietly rewrote a switch would be an engine lock
wearing a disguise, so preservation is checked on every accepted row and
any mismatch is a hard error here).  REFUSED records the refusal's own
message, verbatim.

The receipt asserts nothing.  ``tests/test_physics_composition_walk.py``
regenerates it, compares byte for byte, and pins the properties.

Usage
-----
    python tools/report_physics_composition_walk.py --out <path>
    python tools/report_physics_composition_walk.py --table   # human view
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import itertools
import pathlib
import re
import sys
import tomllib
import warnings

MODEL = pathlib.Path(__file__).resolve().parents[1]
if str(MODEL) not in sys.path:
    sys.path.insert(0, str(MODEL))

from woof.config import (  # noqa: E402
    CU_SCHEMES,
    KM_OPT_ZERO_ACK,
    LAND_SURFACE_SCHEMES,
    MP_PHYSICS_ACCEPTED,
    PBL_SCHEMES,
    RA_LW_PHYSICS_ACCEPTED,
    RA_SW_PHYSICS_ACCEPTED,
    SASE_PBL_SCHEME,
    SURFACE_LAYER_SCHEMES,
    validated_soil_layer_count,
)
from woof.experiment import build_experiment  # noqa: E402
from woof.physics_compat import (  # noqa: E402
    UnsupportedPhysicsSuiteError,
    first_local_night_time,
)
from woof.physics_registry import canonical_json  # noqa: E402

RECEIPT_PATH = (
    MODEL / "docs" / "public" / "receipts" / "physics-composition-walk.json"
)

SCHEMA = "gpuwm-physics-composition-walk-v1"

# ---------------------------------------------------------------------------
# The harness.  Everything not an axis is held fixed, and every held value is
# either a RunConfig default or is listed in HELD_COMPANIONS with its reason.
# ---------------------------------------------------------------------------

#: A one-hour local-afternoon window at a plains reference point.  The window
#: is DELIBERATELY daylight-only: ``build_experiment`` carries the 1.7.1
#: nocturnal-radiation guard, which refuses an undeclared longwave-off /
#: shortwave-on suite for any window containing local night, and that guard
#: would otherwise mix a time-of-day verdict into a composition measurement.
#: ``contains_local_night`` below is measured with the shipped scanner, not
#: assumed, so this control cannot rot silently.
START_TIME = "2021-06-01T18:00:00"
RUN_SECONDS = 3600.0
REF_LAT = 38.0
REF_LON = -97.0

#: 41x41x40 at 3 km.  Small enough that the walk is seconds, wide and deep
#: enough that nothing is refused for being a toy.
#:
#: nz=40 is not a round number, it is the INTERSECTION of every vertical
#: bound in :mod:`woof.physics_vertical_contract`: at or above the
#: Grell-Freitas floor of 12 and the Kain-Fritsch floor of 8, at or below
#: the WSM6 ceiling of 80 -- the tightest ceiling of any admitted scheme --
#: and inside :data:`woof.config.SASE_MAX_NZ`.  A column outside that
#: intersection would make a VERTICAL refusal look like a composition
#: refusal; the first draft of this walk ran nz=8 and reported cu_physics=3
#: as never admitted, which was the harness talking, not the loader.  The
#: test asserts the intersection rather than the number, so a bound that
#: moves fails loudly instead of quietly re-introducing that artefact.
#: 1681 columns is under Noah-MP's projected-cost warning width.
GRID = {"e_we": 41, "e_sn": 41, "dx": 3000.0, "dy": 3000.0, "time_step": 15,
        "nz": 40, "ztop": 20000.0}

BASE_TOML = """
[experiment]
name = "physics-composition-walk"
start_time = {start_time}
run_seconds = {run_seconds}
restart_interval_s = 0.0
{acknowledgements}
[projection]
map_proj = "lambert"
ref_lat = {ref_lat}
ref_lon = {ref_lon}
truelat1 = 30.0
truelat2 = 45.0
stand_lon = {ref_lon}

[shared]
nz = {nz}
ztop = {ztop}
map_proj = 1
moist = true
cudt_minutes = 0.0
{shared}

[[domain]]
grid_id = 1
parent_id = 0
i_parent_start = 1
j_parent_start = 1
parent_grid_ratio = 1
parent_time_step_ratio = 1
history_interval_s = 3600.0
e_we = {e_we}
e_sn = {e_sn}
dx = {dx}
dy = {dy}
time_step = {time_step}
specified = true
"""

#: Non-axis keys the walk pins, each with the reason it is not left to drift.
HELD_COMPANIONS = {
    "moist": (
        "true -- mp_physics and cu_physics both require it "
        "(woof.config.validate_run_config), and a dry state would refuse "
        "every moist scheme for a reason that is not about composition"),
    "cudt_minutes": (
        "0.0 -- the cumulus CADENCE knob, held at every-step for the whole "
        "walk because cu_physics=3 (Grell-Freitas) runs on the model step "
        "and refuses a nonzero cudt; holding it makes cu_physics the only "
        "thing the cumulus axis moves"),
    "num_soil_layers": (
        "derived per land-surface scheme from the shipped resolver "
        "woof.config.validated_soil_layer_count, so a soil-geometry "
        "refusal cannot masquerade as a land-surface refusal"),
    "km_opt_zero_acknowledgement": (
        "supplied at km_opt=0 for every PBL except SASE, because "
        "woof.config.km_opt_zero_producer names SASE as the one closure "
        "that supplies the missing operator itself and the acknowledgement "
        "is refused when a producer exists; both directions are measured in "
        "tier D"),
}


@dataclasses.dataclass(frozen=True)
class Axis:
    """A per-domain physics selector and the values woof's schema admits."""

    name: str
    values: tuple[int, ...]
    authority: str


#: The physics selectors a user sets per domain, with the admitted values
#: taken from the exported validation tuples in ``woof/config.py`` and
#: from ``validate_km_opt`` for turbulence. Tier E
#: MEASURES each of these sets against the loader rather than trusting the
#: transcription.
AXES = (
    # Re-derived from the production validator at the 1.9 assembly, not
    # from a test's assertion error: 9 (Milbrandt-Yau), 16 (WDM6) and 50
    # (P3 one-category) are admitted by the same inline check that admits
    # the rest, so they belong in the swept set.  Leaving them out shrank
    # the measured space while the schema advertised them.  900 (TEMPO, the
    # one named scheme 2.8.8 admits) joins for the same reason, and 30
    # (FAST_SBM) joined the same check with the 2.8.9 port merge, 2 (Purdue
    # Lin), 3 (WSM3), 4 (WSM5), 5 (Ferrier-Aligo), 24 (WSM7) and 106
    # (WSM6R) with their ports.  ra_lw/ra_sw 5 (New Goddard, golden/ports-rad)
    # joined the radiation checks below the same way in 2.9.0.
    Axis("mp_physics", tuple(MP_PHYSICS_ACCEPTED),
         "woof.config.MP_PHYSICS_ACCEPTED"),
    Axis("bl_pbl_physics", tuple(PBL_SCHEMES),
         "woof.config.PBL_SCHEMES"),
    Axis("sf_sfclay_physics", tuple(SURFACE_LAYER_SCHEMES),
         "woof.config.SURFACE_LAYER_SCHEMES"),
    Axis("sf_surface_physics", tuple(LAND_SURFACE_SCHEMES),
         "woof.config.LAND_SURFACE_SCHEMES"),
    Axis("ra_lw_physics", tuple(RA_LW_PHYSICS_ACCEPTED),
         "woof.config.RA_LW_PHYSICS_ACCEPTED"),
    Axis("ra_sw_physics", tuple(RA_SW_PHYSICS_ACCEPTED),
         "woof.config.RA_SW_PHYSICS_ACCEPTED"),
    Axis("cu_physics", tuple(CU_SCHEMES),
         "woof.config.CU_SCHEMES"),
    Axis("km_opt", (0, 1, 2, 3, 4, 5),
         "woof.config.validate_km_opt"),
)

AXIS_NAMES = tuple(axis.name for axis in AXES)
AXIS_VALUES = {axis.name: axis.values for axis in AXES}

#: The legacy aggregate.  ``ra_physics`` is not a ninth composition axis: it
#: is the pre-split spelling of the radiation pair, and a config that sets it
#: must leave ``ra_lw_physics``/``ra_sw_physics`` unset.  Walked in tier C.
RA_PHYSICS_AGGREGATE = (0, 4, 90)

#: Short axis tags for the compact combination key.
_TAG = {"mp_physics": "mp", "bl_pbl_physics": "pbl",
        "sf_sfclay_physics": "sl", "sf_surface_physics": "lsm",
        "ra_lw_physics": "lw", "ra_sw_physics": "sw",
        "cu_physics": "cu", "km_opt": "km"}

#: Suites drawn from tier A's ACCEPTED set, one per PBL family, used as the
#: held background when a non-nexus axis is swept.  Every one of them is
#: asserted to be in the accepted set, so an anchor cannot silently become a
#: refused configuration and hide a whole tier's worth of coverage.
ANCHORS = {
    "gbm-mm5-noah-rrtmgp": {
        "mp_physics": 8, "bl_pbl_physics": 12, "sf_sfclay_physics": 1,
        "sf_surface_physics": 2, "ra_lw_physics": 4, "ra_sw_physics": 4,
        "cu_physics": 1, "km_opt": 4},
    "mynn-rrtmgp-noah": {
        "bl_pbl_physics": 5, "sf_sfclay_physics": 5,
        "sf_surface_physics": 2, "ra_lw_physics": 4, "ra_sw_physics": 4,
        "km_opt": 1},
    "ysu-mm5-noah-rrtmgp": {
        "bl_pbl_physics": 1, "sf_sfclay_physics": 1,
        "sf_surface_physics": 2, "ra_lw_physics": 4, "ra_sw_physics": 4,
        "km_opt": 1},
    # CAM ZM (cu=7) owns a MYJ/CAM UW PBL contract. Include its native
    # MYJ/Janjic background instead of measuring it only on refused pairs.
    "myj-janjic-noah-rrtmgp": {
        "bl_pbl_physics": 2, "sf_sfclay_physics": 2,
        "sf_surface_physics": 2, "ra_lw_physics": 4, "ra_sw_physics": 4,
        "km_opt": 1},
    "shinhong-mm5-noahmp-analytic": {
        "bl_pbl_physics": 11, "sf_sfclay_physics": 91,
        "sf_surface_physics": 4, "ra_lw_physics": 90, "ra_sw_physics": 90,
        "km_opt": 4},
    "pbl-off-les-no-radiation": {
        "bl_pbl_physics": 0, "sf_sfclay_physics": 0,
        "sf_surface_physics": 0, "ra_lw_physics": 0, "ra_sw_physics": 0,
        "km_opt": 3},
    "sase-mm5-ruc-rrtmgp": {
        "bl_pbl_physics": SASE_PBL_SCHEME, "sf_sfclay_physics": 1,
        "sf_surface_physics": 3, "ra_lw_physics": 4, "ra_sw_physics": 4,
        "km_opt": 0},
}

#: Held selectors for tier A, the nexus walk.  Thompson + Kain-Fritsch is the
#: shipped default template's microphysics/cumulus pair.
TIER_A_HELD = {"mp_physics": 8, "cu_physics": 1}

#: The out-of-schema probe range for tier E: one negative, every integer a
#: WRF namelist plausibly carries, and the two values around woof's
#: ArWen-only SASE number.  ``-1`` is deliberately absent -- it is the UNSET
#: sentinel of the radiation pair (``woof.config.radiation_scheme_ids``),
#: not a scheme, and tier C is where the unset spelling is walked.
_SCAN = (-2,) + tuple(range(0, 107)) + (SASE_PBL_SCHEME, 901)

COVERAGE_RULE = (
    "TIER A -- FULL cartesian over the six coupled axes: bl_pbl_physics x "
    "sf_sfclay_physics x sf_surface_physics x ra_lw_physics x ra_sw_physics "
    "x km_opt.  These six are walked exhaustively because every documented "
    "coupling in woof's admission code lies between them (the WRF v4.6.1 "
    "PBL/surface-layer matrix, the land-surface exchange-coefficient seam, "
    "radiation/land-surface requirements, the SASE and LES turbulence rules).  "
    "mp_physics and cu_physics are held at the default template's pair "
    "(8 / 1).",
    "TIER B -- FULL cartesian over the two axes tier A holds, mp_physics x "
    "cu_physics, crossed with every anchor suite.  The anchors are one "
    "accepted tier-A suite per represented PBL family (none, YSU, MYJ, "
    "MYNN, Shin-Hong, GBM, SASE). Each microphysics and each cumulus "
    "scheme is measured on every background, including its named native "
    "pairing when a different family is refused.",
    "TIER C -- the legacy ra_physics aggregate, every admitted value, "
    "crossed with every anchor, with the explicit LW/SW pair left unset.",
    "TIER D -- the km_opt=0 acknowledgement, both directions, on every "
    "anchor: withheld (expected refusal) and supplied (expected admission), "
    "plus the SASE case where supplying it is itself refused.",
    "TIER F -- remedy follow-through.  Every refusal rule tier A-D "
    "produced gets an explicit before/after pair: the before must be "
    "refused by that rule, and the after -- built by DOING WHAT THE "
    "MESSAGE SAYS -- must be accepted.  A refusal whose own remedy is "
    "itself refused is a defect, and this tier is what catches it "
    "(it caught one: see the km_opt=2 entry).  Every refusal also names "
    "a selector a user can change.  The former coupled-LW/SW and "
    "classic-RRTM/Dudhia-only rules are retired: independently selected "
    "spectra now compose, and accepted before/after pairs are not remedies.",
    "TIER E -- schema completeness.  For each axis, every integer in "
    "[-1, 106] plus 900 and 901 is offered to the loader and classified by "
    "whether the refusal is the axis's own schema message.  The measured "
    "in-schema set is compared with the declared one, so the axis tables "
    "above are verified against the artifact instead of transcribed from "
    "it.",
    "COVERAGE GUARANTEE -- every admitted value of every axis appears in at "
    "least one tried combination, and (tier A and B being full cartesians "
    "over disjoint axis groups) every PAIR of values drawn from within a "
    "group appears together.",
)

#: What an ACCEPTED row here does and does not entitle a reader to claim.
#: Stated because the accurate boundary of a measurement is part of it.
SCOPE = {
    "what_is_measured": (
        "CONFIG ADMISSION: woof.experiment.build_experiment resolving an "
        "experiment TOML into per-domain RunConfigs, which is the gate a "
        "user writing a config meets, and the gate the 'is the physics "
        "hardcoded' question is about"),
    "what_is_not_measured": [
        "EXECUTION.  An accepted config is not a completed forecast; "
        "nothing here launches a step, and no GPU is touched.  Runtime "
        "correctness per scheme is the job of the oracle and parity "
        "suites, not of this walk.",
        "THE PREPARED RUNNERS' ROUTE DECLARATIONS.  "
        "woof.physics_compat.land_surface_route_blocker refuses a "
        "land-surface component a given prepared SOURCE's registry route "
        "does not offer (for example RUC on the GFS route).  That is a "
        "per-source data-availability refusal layered above this gate, "
        "not a composition rule, and it names the registry pointer it "
        "enforces.",
        "MULTI-DOMAIN TREES.  Every row is a single specified d01.",
    ],
    "what_the_runner_adds": (
        "woof.prepared_single_domain_forecast._validate_physics applies "
        "the registry's tuple governance to named, matched and unnamed "
        "suites alike as a WARNING, not a refusal (warn-not-block ruling, "
        "owner 2026-07-31): an unblessed tuple runs, prints one line "
        "naming the acknowledgement spellings, and is recorded in the run "
        "receipt with acknowledged=false.  So a combination this walk "
        "accepts is not subsequently blocked for being unnamed -- it is "
        "labelled."),
}

SKIPPED = (
    {
        "what": "the full eight-axis cartesian",
        "why": ("7 x 5 x 4 x 4 x 4 x 4 x 3 x 5 = 134400 combinations at "
                "~0.4 ms each is ~1 minute of loader calls for coverage the "
                "tiered rule already provides; tiers A and B together are "
                "full cartesians over disjoint axis groups, so no coupling "
                "inside either group is missed and only cross-group "
                "triples are traded away"),
    },
    {
        "what": "sf_surface_physics=3 (RUC) at num_soil_layers=6",
        "why": ("WRF's share/module_soil_pre.F:init_soil_depth_3 tabulates "
                "a six-level RUC grid as well as the nine-level one, but "
                "woof.config.validated_soil_layer_count admits nine only "
                "(every RUC oracle fixture in the tree is nine-level and "
                "the CUDA leaves index a nine-element constant array), so "
                "six is a soil-geometry question and not a composition "
                "one"),
    },
    {
        "what": "per-scheme OPTION knobs (bl_mynn_*, icloud_bl, iz0tlnd, "
                "the Noah-MP and RUC option identities, isftcflx, "
                "aer_init_opt/wif_input_opt, morr_rimed_ice, wsm6_hail_opt)",
        "why": ("these select a BRANCH INSIDE a scheme, not a scheme.  Each "
                "is pinned to its single implemented value by an identity "
                "refusal that names the knob, which is a different property "
                "from composition and is pinned by its own tests "
                "(tests/test_mynn_radiation_profiles.py, "
                "tests/test_physics_compat.py)"),
    },
    {
        "what": "sf_urban_physics, sf_ocean_physics, shcu_physics",
        "why": ("not RunConfig fields: woof has no such selectors, so "
                "there is no axis to walk.  They appear in wrfout selector "
                "attributes as constant zeros"),
    },
    {
        "what": "multi-domain trees",
        "why": ("the walk is single-domain.  Whether a per-domain override "
                "is honoured on a nest is a nesting property, pinned by "
                "tests/test_physics_registry_declarations.py and "
                "tests/test_experiment_config.py"),
    },
)


def _suite(**overrides: object) -> dict[str, object]:
    """A complete eight-axis combination, spelled out."""

    combination: dict[str, object] = {
        "mp_physics": 8, "bl_pbl_physics": 1, "sf_sfclay_physics": 1,
        "sf_surface_physics": 2, "ra_lw_physics": 4, "ra_sw_physics": 4,
        "cu_physics": 1, "km_opt": 4}
    combination.update(overrides)
    return combination


_SASE = {"bl_pbl_physics": SASE_PBL_SCHEME, "sf_sfclay_physics": 1,
         "sf_surface_physics": 3, "km_opt": 0}

#: Tier F.  One entry per refusal rule the walk produces: a configuration
#: the rule refuses, and the configuration you get by DOING WHAT THE MESSAGE
#: SAYS.  ``remedy`` quotes the message's own instruction, so the pair can be
#: audited against the text a user actually sees.  The test asserts that
#: these entries cover exactly the rules the walk produced -- a new refusal
#: cannot land without someone demonstrating that its advice works.
REMEDIES = (
    {"id": "sms-3dtke-needs-surface-fluxes",
     "remedy": "Select revised MM5 to supply the SMS lower boundary fluxes.",
     "before": _suite(km_opt=5, bl_pbl_physics=0, sf_sfclay_physics=3, sf_surface_physics=0),
     "after": _suite(km_opt=5, bl_pbl_physics=0, sf_sfclay_physics=1, sf_surface_physics=0)},
    {"id": "keps-needs-surface-state",
     "remedy": "Select revised MM5 to supply UST HFX QFX BR ZNT PSIM PSIH.",
     "before": _suite(bl_pbl_physics=17, sf_sfclay_physics=0),
     "after": _suite(bl_pbl_physics=17, sf_sfclay_physics=1)},
    {"id": "myj-needs-eta-instead-of-gfs",
     "remedy": "Select Eta similarity to supply AKHS AKMS THZ0 QZ0 UZ0 VZ0.",
     "before": _suite(bl_pbl_physics=2, sf_sfclay_physics=3),
     "after": _suite(bl_pbl_physics=2, sf_sfclay_physics=2)},
    {"id": "gfs-surface-needs-gfs-pbl",
     "remedy": "Select GFS PBL, which consumes the GFS surface contract.",
     "before": _suite(bl_pbl_physics=5, sf_sfclay_physics=3),
     "after": _suite(bl_pbl_physics=3, sf_sfclay_physics=3)},
    {"id": "gfs-pbl-needs-paired-surface",
     "remedy": "Select the GFS surface layer to supply GFS PBL exchange fields.",
     "before": _suite(bl_pbl_physics=3, sf_sfclay_physics=0, sf_surface_physics=0),
     "after": _suite(bl_pbl_physics=3, sf_sfclay_physics=3, sf_surface_physics=0)},
    {"id": "flg-longwave-needs-active-shortwave",
     "remedy": "select an active ra_sw_physics",
     "before": _suite(ra_lw_physics=7, ra_sw_physics=0),
     "after": _suite(ra_lw_physics=7, ra_sw_physics=1)},
    {"id": "flg-shortwave-needs-active-longwave",
     "remedy": "select an implemented ra_lw_physics",
     "before": _suite(ra_lw_physics=0, ra_sw_physics=7),
     "after": _suite(ra_lw_physics=1, ra_sw_physics=7)},
    {"id": "sms-3dtke-supplies-the-pbl-closure",
     "remedy": "select bl_pbl_physics=0 when km_opt=5 supplies turbulence",
     "before": _suite(km_opt=5),
     "after": _suite(km_opt=5, bl_pbl_physics=0)},
    {"id": "cloud-optics-without-effective-radii",
     "remedy": "Set ra_rrtmg_variant='rrtmg_legacy' for the backend that diagnoses its own radii.",
     "before": _suite(mp_physics=2, ra_lw_physics=4, ra_sw_physics=4),
     "after": _suite(mp_physics=2, ra_lw_physics=4, ra_sw_physics=4,
                     ra_rrtmg_variant="rrtmg_legacy",
                     wrf_rrtmg_compatibility="wrf-rrtmg-4-4-legacy-v1")},
    {"id": "ccpp-tke-edmf-needs-surface-stability-functions",
     "remedy": "Select revised MM5 (sf_sfclay_physics=1), which supplies "
               "the FM/FH stability functions the CCPP closure divides by",
     "before": _suite(bl_pbl_physics=901, sf_sfclay_physics=0,
                      sf_surface_physics=0, ra_lw_physics=0,
                      ra_sw_physics=0),
     "after": _suite(bl_pbl_physics=901, sf_sfclay_physics=1,
                     sf_surface_physics=0, ra_lw_physics=0,
                     ra_sw_physics=0)},
    {"id": "temf-pbl-needs-its-surface-writer",
     "remedy": "select sf_sfclay_physics=10 to write shared TEMF energy and height",
     "before": _suite(bl_pbl_physics=10, sf_sfclay_physics=1),
     "after": _suite(bl_pbl_physics=10, sf_sfclay_physics=10)},
    {"id": "temf-surface-needs-its-pbl-writer",
     "remedy": "select bl_pbl_physics=10 to advance shared TEMF energy and height",
     "before": _suite(bl_pbl_physics=1, sf_sfclay_physics=10),
     "after": _suite(bl_pbl_physics=10, sf_sfclay_physics=10)},
    {"id": "gbm-requires-mm5-surface-inputs",
     "remedy": "select sf_sfclay_physics=1 or 91 to supply GZ1OZ0 and PSIM",
     "before": _suite(bl_pbl_physics=12, sf_sfclay_physics=5),
     "after": _suite(bl_pbl_physics=12, sf_sfclay_physics=1)},
    {"id": "gbm-requires-condensate-species",
     "remedy": "select a microphysics with cloud-water and cloud-ice state",
     "before": _suite(bl_pbl_physics=12, mp_physics=1),
     "after": _suite(bl_pbl_physics=12, mp_physics=6)},
    {"id": "qnse-pbl-requires-qnse-surface-layer",
     "remedy": "Select sf_sfclay_physics=4 to produce the QNSE lower boundary state.",
     "before": _suite(bl_pbl_physics=4, sf_sfclay_physics=1),
     "after": _suite(bl_pbl_physics=4, sf_sfclay_physics=4)},
    {"id": "qnse-surface-layer-requires-qnse-pbl",
     "remedy": "Select bl_pbl_physics=4 to carry the TKE column the surface layer scans.",
     "before": _suite(bl_pbl_physics=1, sf_sfclay_physics=4),
     "after": _suite(bl_pbl_physics=4, sf_sfclay_physics=4)},
    # Independent spectrum composition retired the coupled-LW/SW and
    # classic-RRTM/Dudhia-only refusals. Their former before cases now
    # accept unchanged and are covered by the positive composition gate.
    {"id": "wrf-461-pbl-surface-layer-matrix",
     "remedy": "the citation names the surface-layer class the PBL needs; "
               "YSU fatals unless isfc=1, which is sf_sfclay_physics 1 or 91",
     "before": _suite(bl_pbl_physics=1, sf_sfclay_physics=5),
     "after": _suite(bl_pbl_physics=1, sf_sfclay_physics=1)},
    # The two MYJ rules.  The PBL and its surface layer refuse in both
    # directions, so both directions carry a pair: neither message is
    # allowed to be the one that sends a user nowhere.
    {"id": "myj-pbl-requires-the-eta-surface-layer",
     "remedy": "'bl_pbl_physics=2 (MYJ) requires sf_sfclay_physics=2 (Eta "
               "similarity)' -- so move the surface layer, which is the "
               "only producer of the AKHS/AKMS/THZ0/QZ0/UZ0/VZ0 lower "
               "boundary MYJ solves against",
     "before": _suite(bl_pbl_physics=2, sf_sfclay_physics=1),
     "after": _suite(bl_pbl_physics=2, sf_sfclay_physics=2)},
    # The PBL-OFF cell of that same reverse rule, which refuses for its
    # own reason: nothing reads fm/fh with the slot off, but the Eta
    # layer's own PBLH scan reads the carried TKE_MYJ column, and that
    # column is allocated for the MYJ PBL selector alone.
    {"id": "eta-surface-layer-with-the-pbl-slot-off",
     "remedy": "'Select bl_pbl_physics=2 to run the Eta layer, or "
               "sf_sfclay_physics=1 (revised MM5) or 91 (classic MM5), "
               "which carry no TKE column' -- followed the first way, "
               "which keeps the Eta surface layer the config asked for",
     "before": _suite(bl_pbl_physics=0, sf_sfclay_physics=2),
     "after": _suite(bl_pbl_physics=2, sf_sfclay_physics=2)},
    {"id": "eta-surface-layer-is-admitted-with-myj-only",
     "remedy": "'Select sf_sfclay_physics=1 (revised MM5) or 91 (classic "
               "MM5) for those schemes, or bl_pbl_physics=2 for this one' "
               "-- followed the second way, which keeps the Eta surface "
               "layer the config asked for",
     "before": _suite(bl_pbl_physics=1, sf_sfclay_physics=2),
     "after": _suite(bl_pbl_physics=2, sf_sfclay_physics=2)},
    # The UW moist-turbulence PBL's two surface-layer rules.  The Eta cell
    # refuses for its own reason (the TKE_MYJ scan), and the off cell
    # because nothing would write UST/HFX/QFX; both messages name the
    # layers the scheme takes, and both are followed to revised MM5, which
    # keeps the UW PBL the config asked for.
    {"id": "uw-pbl-with-the-eta-surface-layer",
     "remedy": "'Select sf_sfclay_physics=1 (revised MM5), 91 (classic "
               "MM5) or 5 (MYNN) with the UW PBL' -- followed the first way",
     "before": _suite(bl_pbl_physics=9, sf_sfclay_physics=2),
     "after": _suite(bl_pbl_physics=9, sf_sfclay_physics=1)},
    {"id": "uw-pbl-needs-a-surface-layer",
     "remedy": "'Select sf_sfclay_physics=1 (revised MM5), 91 (classic "
               "MM5) or 5 (MYNN)' -- followed the first way, with the "
               "land surface the config already had",
     "before": _suite(bl_pbl_physics=9, sf_sfclay_physics=0,
                      sf_surface_physics=0, ra_lw_physics=0,
                      ra_sw_physics=0),
     "after": _suite(bl_pbl_physics=9, sf_sfclay_physics=1,
                     sf_surface_physics=0, ra_lw_physics=0,
                     ra_sw_physics=0)},
    # milbrandt-yau-has-no-rrtmgp-cloud-optics is deliberately ABSENT, and
    # its absence is the record of a fix.  The pair walked the refusal's
    # Dudhia remedy while woof.core.rrtmgp had no row for mp=9; the row
    # now exists (``9: "milbrandt2"``, the scheme's own radii from the
    # block WRF ships commented out at module_mp_milbrandt2mom.F:3351-3378,
    # evaluated over the transported number moments), so the pairing is
    # ACCEPTED and a pair for it would fail this file's own "the before arm
    # must be REFUSED" check -- exactly as P3's did before it.
    # p3-has-no-rrtmgp-cloud-optics is deliberately ABSENT, and its absence
    # is the record of a fix.  mp=50 was ADMITTED against RTE+RRTMGP at 1.9
    # and died at the first radiation call; validate_p3_radiation then
    # refused the pairing up front, and this catalogue carried the pair.
    # The row it was waiting for now exists -- woof.core.rrtmgp's
    # ``50: "p3"``, WRF's own P3 coupling (has_reqc=1, has_reqi=1,
    # has_reqs=0 at module_physics_init.F:1022-1024 then :1033, and the
    # wrappers' ice-into-snow remap at module_ra_rrtmg_lw.F:12250-12261) --
    # so the refusal is retired with the defect and there is no before/after
    # pair left to walk.  A refusal pair for a refusal that no longer fires
    # would fail this file's own "the before arm must be REFUSED" check.
    {"id": "sase-supplies-its-own-horizontal-mixing",
     "remedy": "'Set km_opt=0.'",
     "before": _suite(**_SASE, ra_lw_physics=4, ra_sw_physics=4) | {
         "km_opt": 1},
     "after": _suite(**_SASE)},
    # rrtm-longwave-not-executable is deliberately ABSENT.  It was a
    # remedy row for a refusal that no longer exists: WRF RRTM longwave
    # with Dudhia shortwave is ported, so ra_lw_physics=1 with
    # ra_sw_physics=1 is ACCEPTED and there is nothing to follow through
    # on.  A remedy row whose "before" case is accepted measures nothing,
    # and tier F reads that as a broken remedy rather than a retired one.
    {"id": "land-surface-needs-a-surface-layer",
     "remedy": "'requires a surface layer (sf_sfclay_physics != 0)'",
     "before": _suite(bl_pbl_physics=0, sf_sfclay_physics=0,
                      sf_surface_physics=2, ra_lw_physics=0,
                      ra_sw_physics=0),
     "after": _suite(bl_pbl_physics=0, sf_sfclay_physics=1,
                     sf_surface_physics=2, ra_lw_physics=0,
                     ra_sw_physics=0)},
    {"id": "smagorinsky-3d-is-pbl-off-only",
     "remedy": "'or km_opt=4 (2-D Smagorinsky), which is the "
               "horizontal-only closure every PBL-on template pins'",
     "before": _suite(km_opt=3),
     "after": _suite(km_opt=4)},
    {"id": "prognostic-tke-is-pbl-off-only",
     "remedy": "'Select bl_pbl_physics=0 for an LES domain ..., or "
               "km_opt=4 (2-D Smagorinsky)' -- and this walk is why the "
               "message reads that way.  It used to end 'select the LES "
               "topology or km_opt 3/4', and following it to km_opt=3 with "
               "a PBL scheme on lands on the rule directly above, which is "
               "PBL-off-gated for the same reason",
     "before": _suite(km_opt=2),
     "after": _suite(km_opt=4)},
    # RETIRED with the rule (audit R-036): SASE with the MYNN surface
    # layer is admitted.  The exclusion was an intersection of two tables
    # -- MYNN's row transcribes WRF's isfc matrix, which has no cell for
    # bl_pbl_physics=900 at all -- while the field contract says the
    # opposite: MYNN publishes every one of ust/hfx/qfx/wspd that the
    # closure reads.  A remedy row whose "before" case is accepted
    # measures nothing, so it is deleted rather than left to read as a
    # broken remedy.
    {"id": "sase-needs-a-surface-layer",
     "remedy": "'requires a surface-layer scheme (sf_sfclay_physics != 0)'",
     "before": _suite(**_SASE) | {"sf_sfclay_physics": 0},
     "after": _suite(**_SASE)},
    {"id": "grell-freitas-needs-a-pbl",
     "remedy": "'requires a PBL scheme' -- and YSU in turn requires its own "
               "surface-layer class, so the remedy moves both",
     "before": _suite(cu_physics=3, bl_pbl_physics=0, sf_sfclay_physics=0,
                      sf_surface_physics=0, ra_lw_physics=0,
                      ra_sw_physics=0, km_opt=3),
     "after": _suite(cu_physics=3)},
    # RETIRED with the rule (audit R-006): New Tiedtke runs with the PBL
    # slot off.  Its requirement was cloned from Grell-Freitas above,
    # whose reason is GF's own one-based read of fields["kpbl"]; New
    # Tiedtke reads no kpbl, takes hfx/qfx from the surface stack, and
    # folds zero advective-forcing lanes the way WRF's cumulus driver
    # does.  The cudt_minutes=0 half of that message is a separate rule
    # and keeps its own control in tests/test_ntiedtke_phase2_gates.py.
    {"id": "km-opt-zero-needs-the-acknowledgement",
     "remedy": "'write the acknowledgement out in full: "
               "km_opt_zero_acknowledgement = ...'",
     "before": _suite(km_opt=0) | {"km_opt_zero_acknowledgement": "",
                                   "_ack_withheld": True},
     "after": _suite(km_opt=0) | {
         "km_opt_zero_acknowledgement": KM_OPT_ZERO_ACK}},
    {"id": "km-opt-zero-acknowledgement-is-refused-where-a-producer-exists",
     "remedy": "'km_opt = 0 is admitted here without any acknowledgement' -- "
               "so withdraw it",
     "before": _suite(**_SASE) | {
         "km_opt_zero_acknowledgement": KM_OPT_ZERO_ACK},
     "after": _suite(**_SASE)},
    # The three 1.8.8 radiation-absence refusals.  Each `after` is the same
    # suite with the declaration the message names, which is what
    # render_config writes once the withhold marker is gone -- so the
    # remedy is followed literally rather than paraphrased into a
    # different suite.
    {"id": "constant-downward-longwave-consumed-by-a-land-surface",
     "remedy": "'declare it by adding acknowledgements = "
               "[\"constant-downward-longwave-v1\"] to [experiment]'",
     "before": _suite(ra_lw_physics=0, ra_sw_physics=1) | {
         "_radiation_ack_withheld": True},
     "after": _suite(ra_lw_physics=0, ra_sw_physics=1)},
    {"id": "constant-downward-longwave-published-to-wrfout",
     "remedy": "'declare it by adding acknowledgements = "
               "[\"constant-downward-longwave-v1\"] to [experiment]'",
     "before": _suite(ra_lw_physics=0, ra_sw_physics=1,
                      sf_surface_physics=0, sf_sfclay_physics=0,
                      bl_pbl_physics=0) | {"_radiation_ack_withheld": True},
     "after": _suite(ra_lw_physics=0, ra_sw_physics=1,
                     sf_surface_physics=0, sf_sfclay_physics=0,
                     bl_pbl_physics=0)},
    {"id": "radiation-off-under-a-land-surface",
     "remedy": "'declare the experiment by adding acknowledgements = "
               "[\"radiation-off-land-surface-v1\"] to [experiment]'",
     "before": _suite(ra_lw_physics=0, ra_sw_physics=0) | {
         "_radiation_ack_withheld": True},
     "after": _suite(ra_lw_physics=0, ra_sw_physics=0)},
    # SBU-YLin supplies no native effective radii. Exercise its named
    # native legacy-RRTMG remedy as well as the refused modern pairing.
    {"id": "sbu-ylin-has-no-rte-rrtmgp-cloud-optics",
     "remedy": "'Set ra_rrtmg_variant='rrtmg_legacy' (which computes its "
               "own radii the way WRF does)'",
     "before": _suite(mp_physics=13),
     "after": _suite(mp_physics=13) | {"ra_rrtmg_variant": "rrtmg_legacy"}},
    # FAST_SBM (mp_physics=30) on the default RTE+RRTMGP arm: a JUDGED
    # cloud-optics exclusion (woof.core.rrtmgp._NO_CLOUD_OPTICS_COUPLING),
    # refused at the run door.  Followed the first way it names.
    {"id": "fast-sbm-has-no-rte-rrtmgp-cloud-optics",
     "remedy": "'Set ra_rrtmg_variant='rrtmg_legacy' (which computes its "
               "own radii the way WRF does)'",
     "before": _suite(mp_physics=30),
     "after": _suite(mp_physics=30) | {"ra_rrtmg_variant": "rrtmg_legacy"}},
    {"id": "ntu-has-no-rte-rrtmgp-cloud-optics",
     "remedy": "'Set ra_rrtmg_variant='rrtmg_legacy' (which computes its "
               "own radii the way WRF does)'",
     "before": _suite(mp_physics=56),
     "after": _suite(mp_physics=56) | {"ra_rrtmg_variant": "rrtmg_legacy"}},
    {"id": "cam-zm-needs-native-pbl-temperature-excess",
     "remedy": "'bl_pbl_physics=2 with sf_sfclay_physics=2'",
     "before": _suite(cu_physics=7),
     "after": _suite(cu_physics=7, bl_pbl_physics=2,
                     sf_sfclay_physics=2)},
    # The WRF 4.7.1 port's IDEALSCM prescribed-flux surface layer (89):
    # WRF gives it no PBL isfc and no surface producer, so it runs alone.
    {"id": "idealscm-surface-needs-pbl-off",
     "remedy": "'sf_sfclay_physics=89 requires bl_pbl_physics=0'",
     "before": _suite(sf_sfclay_physics=89, bl_pbl_physics=1,
                      sf_surface_physics=0),
     "after": _suite(sf_sfclay_physics=89, bl_pbl_physics=0,
                     sf_surface_physics=0)},
    {"id": "idealscm-surface-needs-land-surface-off",
     "remedy": "'sf_sfclay_physics=89 requires sf_surface_physics=0'",
     "before": _suite(sf_sfclay_physics=89, bl_pbl_physics=0,
                      sf_surface_physics=2),
     "after": _suite(sf_sfclay_physics=89, bl_pbl_physics=0,
                     sf_surface_physics=0)},
    # RCON-SPEED-01 (woof/config.py): RCON is held at the forecast door
    # until its speed proof clears; the message names the scheme to run.
    {"id": "rcon-held-for-its-speed-proof",
     "remedy": "'select another microphysics scheme, such as mp_physics=28'",
     "before": _suite(mp_physics=29),
     "after": _suite(mp_physics=28)},
    # The port's MAD-WRF, UDM and Eta Ferrier schemes: judged RTE+RRTMGP
    # cloud-optics exclusions, followed the legacy-RRTMG way they name.
    {"id": "madwrf-has-no-rte-rrtmgp-cloud-optics",
     "remedy": "'use --rrtmg-variant rrtmg_legacy'",
     "before": _suite(mp_physics=96),
     "after": _suite(mp_physics=96) | {"ra_rrtmg_variant": "rrtmg_legacy"}},
    {"id": "udm-has-no-rte-rrtmgp-cloud-optics",
     "remedy": "'Set ra_rrtmg_variant='rrtmg_legacy' (which computes its "
               "own radii the way WRF does)'",
     "before": _suite(mp_physics=27),
     "after": _suite(mp_physics=27) | {"ra_rrtmg_variant": "rrtmg_legacy"}},
    {"id": "eta-ferrier-has-no-rte-rrtmgp-cloud-optics",
     "remedy": "'Set ra_rrtmg_variant='rrtmg_legacy' (which computes its "
               "own radii the way WRF does)'",
     "before": _suite(mp_physics=95),
     "after": _suite(mp_physics=95) | {"ra_rrtmg_variant": "rrtmg_legacy"}},
)


# ---------------------------------------------------------------------------
# Running one combination through the real loader.
# ---------------------------------------------------------------------------

def _toml_scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f'"{value}"'
    return repr(value)


def _soil_layers(scheme: object) -> int:
    """The shipped resolver, tolerant of the out-of-schema tier-E probes."""

    try:
        return validated_soil_layer_count(int(scheme))
    except Exception:  # noqa: BLE001 -- an unknown scheme has no geometry
        return 4


def radiation_acknowledgements(shared: dict[str, object]) -> tuple[str, ...]:
    """The governance tokens THIS suite's radiation selectors require.

    The walk measures which physics a user can REACH.  Two 1.8.8 guards
    refuse a longwave-free suite at config load unless the experiment
    declares it, so a walk that offered every combination undeclared
    would report a whole family of admitted selector values as
    unreachable -- which is the harness talking, not the loader, exactly
    as an nz that was too small once reported cu_physics=3 as never
    admitted.  Supplying the declaration is the same idiom this tool
    already uses for ``km_opt_zero_acknowledgement``.

    Derived from the production classifier and the production nocturnal
    predicate, never re-implemented here, so the walk cannot drift from
    what the door does.  Tier G withholds these deliberately, which is
    how the guards themselves stay measured.
    """

    from woof.physics_compat import (ASYMMETRIC_RADIATION_NOCTURNAL_ACK,
                                      CONSTANT_DOWNWARD_LONGWAVE_ACK,
                                      RADIATION_OFF_LAND_SURFACE_ACK,
                                      downward_longwave_disposition,
                                      first_local_night_time)
    from datetime import datetime

    aggregate = int(shared.get("ra_physics", 0) or 0)
    lw = int(shared.get("ra_lw_physics", aggregate))
    sw = int(shared.get("ra_sw_physics", aggregate))
    lw = aggregate if lw < 0 else lw
    sw = aggregate if sw < 0 else sw
    surface = int(shared.get("sf_surface_physics", 0) or 0)
    required: list[str] = []
    if lw == 0 and sw == 0 and surface != 0:
        required.append(RADIATION_OFF_LAND_SURFACE_ACK)
    kind, _consumer = downward_longwave_disposition(
        ra_lw_physics=lw, ra_sw_physics=sw, sf_surface_physics=surface)
    if kind in ("consumed", "published"):
        required.append(CONSTANT_DOWNWARD_LONGWAVE_ACK)
    if sw > 0 and lw == 0 and first_local_night_time(
            datetime.fromisoformat(START_TIME), RUN_SECONDS,
            ref_lat=REF_LAT, ref_lon=REF_LON) is not None:
        required.append(ASYMMETRIC_RADIATION_NOCTURNAL_ACK)
    return tuple(required)


def render_config(combination: dict[str, object]) -> str:
    """The experiment TOML for one combination, companions resolved."""

    shared = dict(combination)
    from woof.physics_registry import column_surface_contract
    contract = column_surface_contract(shared.get("sf_surface_physics", 0))
    if contract:
        for key, row in contract["admission"].items():
            if key not in AXIS_VALUES and key not in shared and len(row["values"]) == 1:
                shared[key] = row["values"][0]
    shared["num_soil_layers"] = _soil_layers(
        shared.get("sf_surface_physics", 0))
    if (shared.get("km_opt") == 0
            and shared.get("bl_pbl_physics") != SASE_PBL_SCHEME
            and "km_opt_zero_acknowledgement" not in shared):
        shared["km_opt_zero_acknowledgement"] = KM_OPT_ZERO_ACK
    shared.pop("_ack_withheld", None)
    # The declarations this suite's radiation selectors require, written
    # into [experiment] exactly as a real config carries them.  Tier G
    # sets the withhold marker to measure the guards themselves.
    withheld = bool(shared.pop("_radiation_ack_withheld", False))
    acknowledgements = () if withheld else radiation_acknowledgements(shared)
    declared = (
        "acknowledgements = ["
        + ", ".join(f'"{value}"' for value in acknowledgements) + "]\n"
        if acknowledgements else "")
    body = "\n".join(f"{key} = {_toml_scalar(value)}"
                     for key, value in sorted(shared.items()))
    return BASE_TOML.format(
        start_time=START_TIME, run_seconds=RUN_SECONDS, ref_lat=REF_LAT,
        ref_lon=REF_LON, shared=body, acknowledgements=declared, **GRID)


@dataclasses.dataclass(frozen=True)
class Outcome:
    verdict: str                    # "ACCEPTED" | "REFUSED"
    error_type: str = ""
    message: str = ""
    rewritten: tuple[str, ...] = ()  # accepted rows whose switches moved


def attempt(combination: dict[str, object]) -> Outcome:
    """Push one combination through the production front door."""

    text = render_config(combination)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            experiment = build_experiment(
                tomllib.loads(text), source="physics-composition-walk.toml")
    except Exception as error:  # noqa: BLE001 -- the verdict IS the exception
        return Outcome("REFUSED", type(error).__name__, str(error))
    run = experiment.root.run
    rewritten = tuple(
        f"{key}: asked {value!r}, resolved {getattr(run, key)!r}"
        for key, value in sorted(combination.items())
        if key in AXIS_VALUES or key == "ra_physics"
        if getattr(run, key) != value
    )
    return Outcome("ACCEPTED", rewritten=rewritten)


# ---------------------------------------------------------------------------
# Grouping refusals.
# ---------------------------------------------------------------------------

#: Any integer not glued to an identifier character.  Normalising these is
#: what turns "sf_surface_physics=2 requires a surface layer" and its 3 and 4
#: spellings into ONE rule with three messages.
_INTEGER = re.compile(r"(?<![A-Za-z_])-?\d+(?:\.\d+)?")


def refusal_rule(error_type: str, message: str,
                 blockers: tuple[str, ...] = ()) -> str:
    """A stable key for "the same rule fired", derived mechanically."""

    if blockers:
        return f"{error_type}: " + "; ".join(sorted(blockers))
    head = " ".join(message.split("[[explain]]")[0].split())
    return f"{error_type}: {_INTEGER.sub('<n>', head)}"


def rule_of_outcome(outcome: "Outcome") -> str:
    """The refusal rule of an outcome, or ``""`` when it was accepted."""

    if outcome.verdict != "REFUSED":
        return ""
    blockers = _blocker_keys(outcome.message) \
        if outcome.error_type == UnsupportedPhysicsSuiteError.__name__ else ()
    return refusal_rule(outcome.error_type, outcome.message, blockers)


def _blocker_keys(message: str) -> tuple[str, ...]:
    """Blocker signatures for a layered suite receipt, from its text."""

    keys = []
    for line in message.split("[[explain]]")[0].splitlines():
        line = line.strip()
        if line.startswith("- "):
            keys.append(_INTEGER.sub("<n>", line[2:]))
    return tuple(keys)


def key_of(combination: dict[str, object]) -> str:
    """The compact, sortable identity of one combination."""

    parts = [f"{_TAG[name]}{combination[name]}"
             for name in AXIS_NAMES if name in combination]
    if "ra_physics" in combination:
        parts.append(f"raagg{combination['ra_physics']}")
    for name in ("ra_rrtmg_variant", "wrf_rrtmg_compatibility"):
        if name in combination:
            parts.append(f"{name}={combination[name]}")
    if combination.get("_radiation_ack_withheld"):
        parts.append("noradack")
    if combination.get("_ack_withheld"):
        parts.append("noack")
    elif combination.get("km_opt_zero_acknowledgement"):
        parts.append("ack")
    return ".".join(parts)


# ---------------------------------------------------------------------------
# The walk.
# ---------------------------------------------------------------------------

def _attempt_batch(rows):
    """Evaluate independent loader calls; preserve their declared order."""
    return [(tier, label, dict(combination), attempt(combination))
            for tier, label, combination in rows]


def _batches(rows, size=256):
    rows = iter(rows)
    while batch := list(itertools.islice(rows, size)):
        yield batch


class Walk:
    """Accumulates every attempt so the receipt can be built from it."""

    def __init__(self, jobs: int | None = None) -> None:
        import os
        available = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
        requested = int(os.environ.get('WOOF_COMPOSITION_WORKERS', '8')) if jobs is None else jobs
        if requested < 1:
            raise ValueError('composition workers must be positive')
        self.jobs = min(requested, available)
        self.rows: list[tuple[str, str, dict, Outcome]] = []

    def run_many(self, rows) -> None:
        # Bounded batches avoid one Future per combination in the full
        # Cartesian. Ordered map keeps receipt bytes independent of scheduling.
        if self.jobs == 1:
            for batch in _batches(rows):
                self.rows.extend(_attempt_batch(batch))
            return
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=self.jobs) as pool:
            for outcomes in pool.map(_attempt_batch, _batches(rows)):
                self.rows.extend(outcomes)

    def run(self, tier: str, combination: dict[str, object],
            label: str = "") -> Outcome:
        outcome = attempt(combination)
        self.rows.append((tier, label, dict(combination), outcome))
        return outcome

    # -- tiers ----------------------------------------------------------
    def tier_a(self) -> None:
        axes = ("bl_pbl_physics", "sf_sfclay_physics", "sf_surface_physics",
                "ra_lw_physics", "ra_sw_physics", "km_opt")
        self.run_many(("A-nexus-cartesian", "", dict(TIER_A_HELD, **dict(zip(axes, values))))
                      for values in itertools.product(*(AXIS_VALUES[a] for a in axes)))

    def tier_b(self) -> None:
        def rows():
            for name, anchor in ANCHORS.items():
                for mp, cu in itertools.product(
                        AXIS_VALUES["mp_physics"], AXIS_VALUES["cu_physics"]):
                    combination = dict(anchor, mp_physics=mp, cu_physics=cu)
                    yield "B-scheme-cartesian-on-anchors", name, combination
                    if mp in (2, 13, 56) and combination.get("ra_lw_physics") == 4 and combination.get("ra_sw_physics") == 4:
                        legacy = dict(combination, ra_rrtmg_variant="rrtmg_legacy",
                                      wrf_rrtmg_compatibility="wrf-rrtmg-4-4-legacy-v1")
                        yield "B-scheme-cartesian-on-anchors", name + ".legacy-cloud-optics", legacy
        self.run_many(rows())

    def tier_c(self) -> None:
        for name, anchor in ANCHORS.items():
            for aggregate in RA_PHYSICS_AGGREGATE:
                combination = dict(anchor, **TIER_A_HELD)
                combination.pop("ra_lw_physics")
                combination.pop("ra_sw_physics")
                combination["ra_physics"] = aggregate
                self.run("C-legacy-radiation-aggregate", combination, name)

    def tier_d(self) -> None:
        for name, anchor in ANCHORS.items():
            sase = anchor["bl_pbl_physics"] == SASE_PBL_SCHEME
            withheld = dict(anchor, **TIER_A_HELD)
            withheld["km_opt"] = 0
            withheld["_ack_withheld"] = True
            withheld["km_opt_zero_acknowledgement"] = ""
            self.run("D-km-opt-zero-acknowledgement", withheld,
                     f"{name}/withheld")
            supplied = dict(anchor, **TIER_A_HELD)
            supplied["km_opt"] = 0
            supplied["km_opt_zero_acknowledgement"] = KM_OPT_ZERO_ACK
            self.run("D-km-opt-zero-acknowledgement", supplied,
                     f"{name}/supplied" + ("/sase" if sase else ""))

    def tier_g(self) -> None:
        """The radiation declarations, withheld and then supplied.

        Tiers A-D declare what each suite requires, which is what makes
        "every admitted value reaches an accepted run" answerable at all.
        That would leave the two 1.8.8 radiation guards unmeasured, so
        this tier offers the same suites with the declaration held back:
        the refusal is the observation, and tier F pairs each one with
        the remedy that clears it.
        """

        for label, suite in (
                ("consumed/lw0-sw1-noah", _suite(ra_lw_physics=0,
                                                 ra_sw_physics=1)),
                ("published/lw0-sw1-no-lsm",
                 _suite(ra_lw_physics=0, ra_sw_physics=1,
                        sf_surface_physics=0, sf_sfclay_physics=0,
                        bl_pbl_physics=0, km_opt=4)),
                ("both-off/noah", _suite(ra_lw_physics=0, ra_sw_physics=0)),
        ):
            self.run("G-radiation-declaration",
                     suite | {"_radiation_ack_withheld": True},
                     f"{label}/withheld")
            self.run("G-radiation-declaration", dict(suite),
                     f"{label}/supplied")

    def column_remedies(self):
        """Follow each registered column coupling's selector remedy."""
        from woof.physics_registry import column_surface_contract
        for selector in LAND_SURFACE_SCHEMES:
            contract = column_surface_contract(selector)
            if not contract:
                continue
            admitted = contract["admission"]
            after = _suite(sf_surface_physics=selector,
                           ra_rrtmg_variant="rrtmg_legacy")
            for key, row in admitted.items():
                if len(row["values"]) == 1:
                    after[key] = row["values"][0]
            rejected = {"sf_sfclay_physics": 91, "bl_pbl_physics": 0,
                        "ra_lw_physics": 90, "ra_sw_physics": 90}
            for key, value in rejected.items():
                if value in admitted[key]["values"]:
                    continue
                yield {"id": f"column-{selector}-{key}",
                       "remedy": admitted[key]["reason"],
                       "before": dict(after, **{key: value}), "after": after}
            yield {"id": f"column-{selector}-spectral-carriers",
                   "remedy": "Select ra_rrtmg_variant='rrtmg_legacy'.",
                   "before": dict(after, ra_rrtmg_variant="rte-rrtmgp"),
                   "after": after}

    def tier_f(self) -> list[dict[str, object]]:
        """Do what each refusal says to do, and see whether it works."""

        rows = []
        for entry in (*REMEDIES, *self.column_remedies()):
            before = self.run("F-remedy-follow-through", entry["before"],
                              f"{entry['id']}/before")
            after = self.run("F-remedy-follow-through", entry["after"],
                             f"{entry['id']}/after")
            rows.append({
                "id": entry["id"],
                "remedy_the_message_gives": entry["remedy"],
                "before": {
                    "combination": key_of(entry["before"]),
                    "verdict": before.verdict,
                    "rule": rule_of_outcome(before),
                    "message": before.message,
                },
                "after": {
                    "combination": key_of(entry["after"]),
                    "verdict": after.verdict,
                    "message": after.message,
                },
                "remedy_works": (before.verdict == "REFUSED"
                                 and after.verdict == "ACCEPTED"),
            })
        return rows

    def tier_e(self) -> dict[str, list[int]]:
        """Measure each axis's admitted set instead of trusting the table."""

        measured: dict[str, list[int]] = {}
        anchor = dict(ANCHORS["ysu-mm5-noah-rrtmgp"], **TIER_A_HELD)
        for axis in AXES:
            in_schema = []
            for candidate in _SCAN:
                combination = dict(anchor)
                combination[axis.name] = candidate
                outcome = self.run(
                    "E-schema-scan", combination, axis.name)
                # The predicate asks "did the loader recite this axis's
                # whole menu", which is the shape every VALUE refusal
                # takes and no COMBINATION refusal does.
                #
                # mp_physics has two value gates -- the generic unknown
                # branch and the P3 siblings 51/52/53, which state their
                # own missing physics -- and BOTH recite the menu, via
                # woof.config._MP_PHYSICS_SCHEMA_MENU.  That is a
                # producer-side invariant, deliberately: a widening here
                # to match "<axis>=<value>" plus "is not ported" was
                # tried and rejected, because it also matches the km_opt=2
                # COMPOSITION refusal, whose message legitimately contains
                # both, and it dropped a genuinely in-schema value.  A
                # value refusal is not separable from a combination
                # refusal by classifying free text, so the two are kept
                # distinguishable where they are written.
                schema_refusal = (
                    outcome.verdict == "REFUSED"
                    and f"{axis.name} must be" in outcome.message)
                if not schema_refusal:
                    in_schema.append(candidate)
            measured[axis.name] = in_schema
        return measured


def evaluate(*, jobs: int | None = None) -> dict[str, object]:
    """Walk the space and return the receipt."""

    from datetime import datetime

    night = first_local_night_time(
        datetime.fromisoformat(START_TIME), RUN_SECONDS,
        ref_lat=REF_LAT, ref_lon=REF_LON)

    walk = Walk(jobs=jobs)
    walk.tier_a()
    walk.tier_b()
    walk.tier_c()
    walk.tier_d()
    walk.tier_g()
    remedies = walk.tier_f()
    measured = walk.tier_e()

    #: Tiers A-D are the composition measurement.  E is a schema probe (it
    #: deliberately offers values no schema admits) and F re-runs the same
    #: space to test advice, so neither belongs in the headline counts.
    composition = [row for row in walk.rows
                   if row[0] not in ("E-schema-scan",
                                     "F-remedy-follow-through")]

    rewritten = [
        {"combination": key_of(combination), "rewritten": list(o.rewritten)}
        for _tier, _label, combination, o in walk.rows if o.rewritten
    ]

    tiers: dict[str, dict[str, object]] = {}
    for tier, _label, _combination, outcome in walk.rows:
        entry = tiers.setdefault(
            tier, {"tried": 0, "accepted": 0, "refused": 0})
        entry["tried"] += 1
        entry["accepted" if outcome.verdict == "ACCEPTED"
              else "refused"] += 1

    rules: dict[str, dict[str, object]] = {}
    for tier, _label, combination, outcome in composition:
        if outcome.verdict != "REFUSED":
            continue
        blockers = _blocker_keys(outcome.message) \
            if outcome.error_type == UnsupportedPhysicsSuiteError.__name__ \
            else ()
        rule = refusal_rule(outcome.error_type, outcome.message, blockers)
        entry = rules.setdefault(rule, {
            "rule": rule, "error_type": outcome.error_type, "count": 0,
            "messages": collections.Counter(), "first": {}})
        entry["count"] += 1
        entry["messages"][outcome.message] += 1
        entry["first"].setdefault(outcome.message, key_of(combination))

    refusal_rows = []
    for entry in sorted(rules.values(), key=lambda e: (-e["count"],
                                                       e["rule"])):
        messages: collections.Counter = entry["messages"]
        refusal_rows.append({
            "rule": entry["rule"],
            "error_type": entry["error_type"],
            "count": entry["count"],
            "distinct_messages": len(messages),
            "messages": [
                {"count": count,
                 "example_combination": entry["first"][message],
                 "message": message}
                for message, count in sorted(
                    messages.items(), key=lambda kv: (-kv[1], kv[0]))
            ],
        })

    per_axis: dict[str, dict[str, dict[str, int]]] = {}
    for axis in AXES:
        per_value: dict[str, dict[str, int]] = {}
        for value in axis.values:
            tried = accepted = 0
            for _tier, _label, combination, outcome in composition:
                if combination.get(axis.name) != value:
                    continue
                tried += 1
                accepted += outcome.verdict == "ACCEPTED"
            per_value[str(value)] = {
                "tried": tried, "accepted": accepted,
                "refused": tried - accepted}
        per_axis[axis.name] = per_value

    #: DISTINCT suites, not accepted rows: the anchors are re-tried in
    #: several tiers, and a reader counting this list should be counting
    #: configurations, not attempts.
    accepted_keys = sorted({
        key_of(combination)
        for _tier, _label, combination, outcome in composition
        if outcome.verdict == "ACCEPTED"})

    mynn = [
        (combination, outcome)
        for _tier, _label, combination, outcome in composition
        if combination.get("bl_pbl_physics") == 5
        and combination.get("sf_sfclay_physics") == 5
    ]
    mynn_slice = {
        "definition": ("bl_pbl_physics=5 with sf_sfclay_physics=5, the "
                       "pairing WRF v4.6.1 admits for the MYNN PBL"),
        "tried": len(mynn),
        "accepted": sum(o.verdict == "ACCEPTED" for _c, o in mynn),
        "radiation_options_accepted": sorted({
            f"{c['ra_lw_physics']}/{c['ra_sw_physics']}"
            for c, o in mynn
            if o.verdict == "ACCEPTED" and "ra_lw_physics" in c}),
        "land_surface_options_accepted": sorted({
            int(c["sf_surface_physics"]) for c, o in mynn
            if o.verdict == "ACCEPTED"}),
        "microphysics_options_accepted": sorted({
            int(c["mp_physics"]) for c, o in mynn
            if o.verdict == "ACCEPTED"}),
        "cumulus_options_accepted": sorted({
            int(c["cu_physics"]) for c, o in mynn
            if o.verdict == "ACCEPTED"}),
        "turbulence_options_accepted": sorted({
            int(c["km_opt"]) for c, o in mynn if o.verdict == "ACCEPTED"}),
    }

    #: The MYNN question, spelled out cell by cell rather than summarised:
    #: MYNN PBL + MYNN surface layer against every land surface and every
    #: radiation pair, microphysics and cumulus held at the default
    #: template's.  64 cells, each carrying its own verdict.
    mynn_matrix = {}
    for lsm, lw, sw in itertools.product(
            AXIS_VALUES["sf_surface_physics"], AXIS_VALUES["ra_lw_physics"],
            AXIS_VALUES["ra_sw_physics"]):
        combination = dict(
            TIER_A_HELD, bl_pbl_physics=5, sf_sfclay_physics=5,
            sf_surface_physics=lsm, ra_lw_physics=lw, ra_sw_physics=sw,
            km_opt=1)
        outcome = attempt(combination)
        mynn_matrix[f"lsm{lsm}.lw{lw}.sw{sw}"] = (
            outcome.verdict if outcome.verdict == "ACCEPTED"
            else f"REFUSED: {rule_of_outcome(outcome).split(': ', 1)[1][:90]}")

    anchors_accepted = {}
    for name, anchor in ANCHORS.items():
        combination = dict(anchor, **TIER_A_HELD)
        outcome = attempt(combination)
        anchors_accepted[name] = {
            "combination": key_of(combination),
            "verdict": outcome.verdict,
            "message": outcome.message,
        }

    full_cartesian = 1
    for axis in AXES:
        full_cartesian *= len(axis.values)

    tried = len(composition)
    accepted = sum(o.verdict == "ACCEPTED" for _t, _l, _c, o in composition)

    return {
        "schema": SCHEMA,
        "question": (
            "can a user compose woof's physics freely, or is some engine "
            "lock pinning runs to the shipped named suites?"),
        "loader": "woof.experiment.build_experiment",
        "verdict_definition": {
            "ACCEPTED": ("the loader returned an experiment AND every "
                         "selector the file set survived onto the resolved "
                         "per-domain RunConfig unchanged"),
            "REFUSED": ("the loader raised; the refusal's own message is "
                        "recorded verbatim"),
        },
        "harness": {
            "start_time": START_TIME,
            "run_seconds": RUN_SECONDS,
            "reference_point": {"lat": REF_LAT, "lon": REF_LON},
            "window_contains_local_night": night is not None,
            "window_night_control": (
                "measured with woof.physics_compat.first_local_night_time; "
                "a daylight-only window keeps the 1.7.1 nocturnal-radiation "
                "guard out of a composition measurement"),
            "grid": dict(GRID),
            "held_companions": dict(HELD_COMPANIONS),
        },
        "axes": [
            {"name": axis.name, "admitted_values": list(axis.values),
             "authority": axis.authority,
             "measured_admitted_values": measured[axis.name],
             "measured_matches_declared":
                 measured[axis.name] == list(axis.values)}
            for axis in AXES
        ],
        "legacy_aggregate": {
            "name": "ra_physics",
            "admitted_values": list(RA_PHYSICS_AGGREGATE),
            "note": ("the pre-split spelling of the radiation pair, not a "
                     "ninth axis; a config that sets it must leave "
                     "ra_lw_physics/ra_sw_physics unset"),
        },
        "coverage_rule": list(COVERAGE_RULE),
        "scope": dict(SCOPE),
        "skipped": [dict(entry) for entry in SKIPPED],
        "anchors": anchors_accepted,
        "totals": {
            "full_eight_axis_cartesian": full_cartesian,
            "tried": tried,
            "accepted": accepted,
            "distinct_accepted_suites": len(accepted_keys),
            "refused": tried - accepted,
            "schema_scan_probes": sum(
                1 for row in walk.rows if row[0] == "E-schema-scan"),
            "remedy_follow_through_probes": sum(
                1 for row in walk.rows if row[0] == "F-remedy-follow-through"),
            "accepted_with_a_rewritten_switch": len(rewritten),
        },
        "tiers": {
            tier: dict(counts) for tier, counts in sorted(tiers.items())},
        "refusal_rules": refusal_rows,
        "distinct_refusal_rules": len(refusal_rows),
        "distinct_refusal_messages": sum(
            row["distinct_messages"] for row in refusal_rows),
        "per_axis": per_axis,
        "mynn_slice": mynn_slice,
        "mynn_matrix": mynn_matrix,
        "remedy_follow_through": remedies,
        "switch_rewrites": rewritten,
        "accepted_combinations": accepted_keys,
    }


def render(report: dict[str, object]) -> bytes:
    return (canonical_json(report) + "\n").encode("utf-8")


# ---------------------------------------------------------------------------
# Human view.
# ---------------------------------------------------------------------------

def table(report: dict[str, object]) -> str:
    out: list[str] = []
    totals = report["totals"]
    out.append("PHYSICS COMPOSITION WALK -- "
               f"loader {report['loader']}")
    out.append("")
    out.append(f"  tried     {totals['tried']:6d}")
    out.append(f"  accepted  {totals['accepted']:6d}")
    out.append(f"  refused   {totals['refused']:6d}")
    out.append(f"  rewritten {totals['accepted_with_a_rewritten_switch']:6d}"
               "   (accepted rows whose switches did not survive)")
    out.append("")
    out.append("TIERS")
    for tier, counts in report["tiers"].items():
        out.append(f"  {tier:34s} tried {counts['tried']:6d}  "
                   f"accepted {counts['accepted']:5d}  "
                   f"refused {counts['refused']:6d}")
    out.append("")
    out.append("PER AXIS (composition tiers only)")
    for axis in report["axes"]:
        out.append(f"  {axis['name']}  "
                   f"(measured == declared: "
                   f"{axis['measured_matches_declared']})")
        rows = report["per_axis"][axis["name"]]
        for value, counts in rows.items():
            out.append(f"      {value:>4s}  tried {counts['tried']:6d}  "
                       f"accepted {counts['accepted']:5d}  "
                       f"refused {counts['refused']:6d}")
    out.append("")
    out.append(f"REFUSAL RULES ({report['distinct_refusal_rules']} distinct "
               f"rules, {report['distinct_refusal_messages']} distinct "
               "messages)")
    for row in report["refusal_rules"]:
        out.append(f"  {row['count']:6d}  [{row['error_type']}] "
                   f"{row['rule'].split(': ', 1)[1][:140]}")
    out.append("")
    out.append("REMEDY FOLLOW-THROUGH (does the refusal's own advice work?)")
    for row in report["remedy_follow_through"]:
        mark = "ok  " if row["remedy_works"] else "FAIL"
        out.append(f"  {mark}  {row['id']}")
        out.append(f"          before {row['before']['combination']} -> "
                   f"{row['before']['verdict']}")
        out.append(f"          after  {row['after']['combination']} -> "
                   f"{row['after']['verdict']}")
    out.append("")
    out.append("MYNN SLICE (bl_pbl_physics=5, sf_sfclay_physics=5)")
    for key, value in report["mynn_slice"].items():
        out.append(f"  {key}: {value}")
    out.append("")
    out.append("MYNN MATRIX -- mp_physics=8, cu_physics=1, km_opt=1, "
               "MYNN PBL + MYNN surface layer")
    out.append("            " + "".join(
        f"sw{sw:<8d}" for sw in AXIS_VALUES["ra_sw_physics"]))
    for lsm in AXIS_VALUES["sf_surface_physics"]:
        for lw in AXIS_VALUES["ra_lw_physics"]:
            cells = []
            for sw in AXIS_VALUES["ra_sw_physics"]:
                verdict = report["mynn_matrix"][f"lsm{lsm}.lw{lw}.sw{sw}"]
                cells.append("ACCEPT  " if verdict == "ACCEPTED"
                             else "refused ")
            out.append(f"  lsm{lsm:<3d} lw{lw:<4d} " + "".join(cells))
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=pathlib.Path, default=RECEIPT_PATH)
    parser.add_argument("--table", action="store_true",
                        help="print the human-readable table to stdout")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--jobs", type=int, default=None, help="bounded loader workers (default 8)")
    args = parser.parse_args(argv)

    report = evaluate(jobs=args.jobs)
    if not args.no_write:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(render(report))
    if args.table:
        print(table(report))
    totals = report["totals"]
    print(f"tried={totals['tried']} accepted={totals['accepted']} "
          f"refused={totals['refused']} "
          f"rules={report['distinct_refusal_rules']} | "
          f"{'(not written)' if args.no_write else args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
