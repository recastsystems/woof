"""``woof domain``: turn "my location + my GPU" into an experiment TOML.

The wizard emits a complete ``[experiment]``/``[projection]``/``[shared]``/
``[[domain]]`` TOML centered on ``--point`` or fitted around a local
``--polygon`` GeoJSON footprint, with grid dimensions chosen so
the itemized VRAM estimate (:func:`woof.core.preflight.estimate_experiment`)
plus the machine-peak envelope over it
(:func:`woof.core.preflight.machine_peak_envelope_bytes`) fits the
requested card's budget with headroom to spare.  Nothing here is a new
model of anything: the
physics/dynamics block is the product's default suite (the four-domain
reference configuration's selections with the microphysics slot on
Thompson mp8, the wrf-matched-run scheme), the projection
and nest-registration math is the existing :mod:`woof.static.projection`
(lambert/mercator/polar, auto-selected from the point latitude), and the
memory arithmetic is the existing preflight estimator called in-process.

Accuracy contract (terrain/static story): woof builds static fields from a
locally staged NCAR WPS_GEOG tree (nine fixed dataset directories -- see
``GEOG_DATASETS``); ``woof fetch-geog`` downloads and stages it.
Forcing data for the
config-driven ``woof check``/``run`` front door is decoded by the native
GRIB1 route, i.e. ERA5 today.  GFS/HRRR downloads (``woof fetch``) feed
the ``rw-wps``/``woof-wrf-init`` native initialization front door, which
consumes the same ``[experiment]``/``[[domain]]`` tables but not
``[case_data]``.  The wizard therefore emits ``[case_data]`` only for
``--source era5``. An explicit geography root is saved in that table for
ERA5 and in ``[static].geog_root`` for every other source, so the printed
next command uses the same tree without another flag.

Sizing conventions (all documented, none silent):

* Automatic sizing uses the local probe's available VRAM. Explicit card
  declarations use the free VRAM a card of that capacity usually presents
  (:func:`card_assumed_free_gib` -- never the nameplate; see
  :data:`CARD_UNAVAILABLE_VRAM_GIB`) minus THIS configuration's own
  reserve (:func:`sizing_budget_bytes`, the same
  ``ReservePolicy.n0_alloc`` call ``woof check`` makes).  The reserve is
  not flat: it carries the local-memory backing store of the selected
  kernel set, which is 1.93 GiB for WSM6+MYNN and 3.94 for NSSL2
  double-moment, and a fit loop assuming a flat figure emitted configs
  that failed their own check.
* Fit criterion: ``peak envelope <= budget - fit_headroom_bytes`` --
  the estimator's own machine-peak envelope, which is AFFINE (the
  itemized estimate, plus the non-pool residency that scales with the
  device rather than the grid, plus a measured constant and a per-nest
  fraction), on every driver model and radiation lane alike since A163
  retired the pool-slack term -- the same model `woof check` and
  `woof go` price, so
  a wizard PASS cannot become a check refusal on the same machine
  state.  The loop stops SHORT of the budget on purpose: a config that
  exactly touches its budget has nothing left for the machine to be
  slightly less generous than the model, which is how every v1.4.0
  ladder came to sit 0.01-0.19 GiB from the wall.
* Root time step: the real-data recommendation, 5 s per km of grid
  spacing (60 s at 12 km), halved inside the tropics. An omitted clock
  may be shortened to land the requested event schedule exactly; children divide
  down the ratio chain exactly, and a half-second root clock is carried
  exactly through WRF's rational clock keys.
* Ladders: the presets in ``LADDER_RATIOS``, or ``--root-dx`` +
  ``--chain`` for an arbitrary root spacing and integer refinement
  chain.  Both go through the same fit loop, loader, and ``woof
  check``; a chain reaching below 1 km with a 1-D PBL scheme active
  earns a gray-zone advisory (never a refusal).
"""

from __future__ import annotations

from woof.physics_registry import canonical_template_id

import argparse
import json
import hashlib
import math
import os
import re
import shlex
import shutil
import tomllib
from dataclasses import dataclass
from datetime import datetime, timedelta
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType

import numpy as np

from woof.core.preflight import (CUDA_CONTEXT_BYTES,
                                  non_pool_basis,
                                  ENVELOPE_UNMODELLED_BYTES,
                                  EXTERNAL_MARGIN_BYTES, GIB,
                                  INGEST_PEAK_ENVELOPE_BASIS,
                                  PROBE_REASON_NO_RUNTIME,
                                  ReservePolicy,
                                  card_local_memory_profile,
                                  device_memory_probe_reason,
                                  device_memory_probe_subprocess,
                                  envelope_platform, estimate_experiment,
                                  estimate_phases,
                                  profile_from_device_probe,
                                  unknown_platform_note)
from woof.cli_numbers import positive_float, positive_float_list, positive_int
from woof.experiment import ExperimentConfig, build_experiment
from woof.explain import explain_enabled, muted_warnings, warn
from woof.fetch import parse_cycle
from woof.physics_compat import (ASYMMETRIC_RADIATION_NOCTURNAL_ACK,
                                  CONSTANT_DOWNWARD_LONGWAVE_ACK,
                                  RRTMG_VARIANT_LEGACY,
                                  MORRISON_PROFILE_ID, MYNN_PROFILE_ID,
                                  MYNN_RTE_RRTMGP_PROFILE_ID,
                                  MYNN_RUC_PROFILE_ID,
                                  MYNN_RUC_RTE_RRTMGP_PROFILE_ID,
                                  NSSL2_LEGACY_RRTMG_PROFILE_ID,
                                  NSSL2_PROFILE_ID,
                                  RUC_PROFILE_ID,
                                  THOMPSON_LEGACY_RRTMG_PROFILE_ID,
                                  THOMPSON_MYNN_RUC_DUDHIA_PROFILE_ID,
                                  THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID,
                                  THOMPSON_PROFILE_ID,
                                  THOMPSON_SHINHONG_LEGACY_RRTMG_PROFILE_ID,
                                  WSM6_PROFILE_ID,
                                  first_local_night_time,
                                  single_domain_runtime_switches)
from woof.hrrr_route_inputs import (ROUTE_DEFAULT_PHYSICS_PROFILE,
                                     HrrrRouteInputError, coverage_advisory,
                                     coverage_refusal, route_input_paths,
                                     route_physics_blocker,
                                     write_hrrr_route_inputs)
from woof.physics_menu import (WIZARD_PHYSICS_PROFILES,
                                profile_route_blocker)
from woof.grid_requirements import FIFTH_ORDER_STENCIL_AXIS, boundary_axis
from woof.source_adapters import (get_source_adapter, source_adapters,
                                   source_forcing_interval_seconds,
                                   wizard_cadence_named_source_ids,
                                   wizard_planable_source_ids)
# config_source_coverage_refusal and its geometry live in source_coverage so
# the RW-WPS wheel, which excludes this module, can review a config too.
from woof.source_coverage import (_root_coverage_gap, _root_grid,
                                   config_source_coverage_refusal)
from woof.static.projection import (EARTH_RADIUS_M, POLE_CLEARANCE_CELLS,
                                     WRF_MAP_PROJ_CODES,
                                     footprint_contains_pole,
                                     footprint_longitude_span, _wrap180)

#: Card tiers -> total VRAM (GiB).  ``--vram-gib`` accepts anything else.
CARD_VRAM_GIB = {"12gb": 12.0, "16gb": 16.0, "24gb": 24.0, "32gb": 32.0}

#: Board memory of the NVIDIA models a ``--card`` spelling is matched
#: against, in GiB, keyed by the model token left after the vendor words,
#: spaces and hyphens are stripped ("RTX 3080" -> "3080", "5070 Ti" ->
#: "5070ti").  Where a model shipped in two sizes the smaller is listed,
#: because a budget that is too large fails on the card and one that is
#: too small only refuses a fit; put the size in the name ("4060 Ti 16GB")
#: to say otherwise.  Longer tokens are tried first so "3080ti" is not
#: read as "3080".  A model with no row is not refused for being
#: unrecorded: the size in the name or ``--vram-gib`` prices it.
KNOWN_CARD_VRAM_GIB = {
    "3060ti": 8.0, "3060": 12.0, "3070ti": 8.0, "3070": 8.0,
    "3080ti": 12.0, "3080": 10.0, "3090ti": 24.0, "3090": 24.0,
    "4060ti": 8.0, "4060": 8.0, "4070tisuper": 16.0, "4070ti": 12.0,
    "4070super": 12.0, "4070": 12.0, "4080super": 16.0, "4080": 16.0,
    "4090": 24.0,
    "5060ti": 8.0, "5060": 8.0, "5070ti": 16.0, "5070": 12.0,
    "5080": 16.0, "5090": 32.0,
    "a6000": 48.0, "a5000": 24.0, "a4000": 16.0, "a40": 48.0,
    "a30": 24.0, "a10": 24.0, "a100": 40.0, "h100": 80.0, "h200": 141.0,
    "l40s": 48.0, "l40": 48.0, "l4": 24.0, "t4": 16.0, "v100": 16.0,
}

_CARD_SIZE_IN_NAME = re.compile(r"(\d+(?:\.\d+)?)\s*gi?b\b")
_CARD_VENDOR_WORDS = ("nvidia", "geforce", "quadro", "tesla", "rtx", "gtx")


def card_capacity_gib(card: str) -> float | None:
    """The VRAM capacity a ``--card`` spelling names, or ``None``.

    Three spellings are read, in order: a tier (``12gb``/``16gb``/``24gb``/
    ``32gb``), a size written into the name (``10gb``, ``"RTX 3080 10GB"``,
    a bare ``10``), and a model with a row in :data:`KNOWN_CARD_VRAM_GIB`
    (``"RTX 3080"``, ``rtx3080``, ``"5070 Ti"``).  ``None`` means the
    spelling carries no capacity, which is the only thing the sizing door
    refuses it for.
    """
    text = str(card).strip().lower()
    if text in CARD_VRAM_GIB:
        return CARD_VRAM_GIB[text]
    sized = _CARD_SIZE_IN_NAME.search(text)
    if sized:
        return float(sized.group(1))
    token = text
    for word in _CARD_VENDOR_WORDS:
        token = token.replace(word, " ")
    token = re.sub(r"[\s_\-]+", "", token)
    for model in sorted(KNOWN_CARD_VRAM_GIB, key=len, reverse=True):
        if model in token:
            return KNOWN_CARD_VRAM_GIB[model]
    # A bare number is a GiB figure only where it can be one; "3080" with
    # no vendor word is a model number with no row, not 3 TiB.
    try:
        bare = float(text)
    except ValueError:
        return None
    return bare if 0 < bare <= 512 else None


def declared_card_gib(card: str) -> float:
    """``card_capacity_gib`` that refuses a spelling with no capacity in it."""
    capacity = card_capacity_gib(card)
    if capacity is None:
        raise ValueError(
            f"--card {card!r} names no capacity this tree knows: it is not a "
            f"tier ({'/'.join(sorted(CARD_VRAM_GIB))}), carries no size, and "
            "has no model row. Put the size in the name (--card 'ada 9000 "
            "24gb') or pass --vram-gib N beside it.")
    return capacity

#: Nest ladders: dx chain in km (root fixed at 12 km) -> parent grid/time
#: ratios.  Ratios follow the certified 12->3->1 chain (4, 3); the 500 m
#: extension refines the 1 km nest by 2.
LADDER_RATIOS = {
    "12": (),
    "12-3": (4,),
    "12-3-1": (4, 3),
    "12-3-1-0.5": (4, 3, 2),
}
_LADDERS_DEEPEST_FIRST = ("12-3-1-0.5", "12-3-1", "12-3")

#: What a bare invocation emits: one 12 km domain, the shape ``woof
#: go`` runs end to end and the shape FIRST-LIGHT 3a's worked first run
#: uses.  ``auto`` -- the deepest preset that fits the card -- was the
#: default until the first user-zero run of the published wheel piped
#: the default emission (a 4-domain tree) into the default runner and
#: was refused; the interactive door had already ruled the same way
#: (:data:`woof.domain_interactive.DEFAULT_LADDER`, its own constant
#: because that door imports nothing heavy).  Nest trees are explicit
#: opt-in: a deeper preset, ``auto``, or --root-dx/--chain.
#:
#: The argparse default is ``None``, not this value, and that is
#: essential: --root-dx/--chain are refused WITH --ladder, and a
#: reader who typed only the custom form must not be refused for
#: "combining" it with a flag they never passed.  ``domain_main``
#: resolves ``None`` to this constant (bare) or to the custom form's
#: pass-through value (--root-dx/--chain present).
DEFAULT_LADDER = "12"

ROOT_DX_M = 12000.0
#: Certified real-data clock convention: 60 s at 12 km = 5 s per km.
ROOT_TIME_STEP_S = 60

#: Halved root clock (2.5 s per km) for domains inside the tropics.
#:
#: A tropical Mercator domain at Manila (14.6 N) on the wizard's own
#: 60 s clock reached w_max 6.62 m/s and destabilized at +1 h; the same
#: domain at 15 s kept w_max at 1.5-3.0 m/s and completed 6 h.  The old
#: monitor exaggerated that event by pairing the global w maximum with
#: the unrelated thinnest layer; v1.1 now uses co-located |w|/dz.  The
#: trajectory evidence still supports a conservative tropical clock:
#: convection is deeper and more continuous at low latitude, so the
#: 5 s/km rule of thumb (WRF's own 6*dx_km agrees with it) is not
#: conservative enough there.
#: Receipt: ARWEN-NODE2-4090-CUDA128-WORLDWIDE-20260730.md, PP-9.
#:
#: RE-MEASURED 2026-08-26 under the corrected v1.1 monitor, which is
#: what the stale-guard audit of 2026-08-25 asked for: this guard had
#: been retained on a substituted rationale after its motivating
#: instrument was found wrong, so it was re-run on the instrument that
#: replaced it.  The motivating case (14.6 N, Mercator, 386x308 at
#: 12 km, morrison rte-rrtmgp, 6 h GFS) on an RTX 3080, one arm per
#: clock, identical in every other byte:
#:
#:   ========  ==========  ==================  ============  =========
#:   clock     completes   peak vertical CFL   peak w_max    forecast
#:   ========  ==========  ==================  ============  =========
#:   30 s      yes         0.976               5.11 m/s      261 s
#:   60 s      yes         1.997               8.52 m/s      151 s
#:   ========  ==========  ==================  ============  =========
#:
#: The un-halved clock runs the tropical domain at TWICE the vertical
#: Courant limit -- the co-located |w|/dz the corrected monitor
#: measures, not the old global-max artefact -- while the halved clock
#: holds it just under 1.  Neither arm produced a NaN in 6 h, so the
#: breakage this prevents is stated as what it is: sustained integration
#: past the vertical Courant limit, where the scheme is outside its
#: stability region and any given case surviving is luck, not margin.
#: Note also that 0.976 leaves this guard almost NO headroom, so it is
#: not over-conservative at 30 s.
#:
#: The cost is larger than the +22% this note used to quote from node 2:
#: on this card the halved clock cost +73% of forecast wall (261 s vs
#: 151 s).  Recorded as measured rather than left at the friendlier
#: figure -- the guard is justified by the CFL evidence, not by being
#: cheap.
TROPICAL_ROOT_TIME_STEP_S = 30

#: Parent rows a child boundary must clear: spec_bdy_width + blend_width
#: (both 5, the emitted [experiment] values; woof/experiment.py enforces).
_SPEC_BDY_WIDTH = 5
_BLEND_WIDTH = 5
_CLEARANCE_ROWS = _SPEC_BDY_WIDTH + _BLEND_WIDTH

#: Child linear extent as a fraction of the parent's, by nest depth --
#: the certified four-domain layout's proportions, then quantized.
_CHILD_SPAN_FRACTION = (0.5, 0.36, 0.4)

#: diff_6th_factor by nest depth (certified ladder).
_DIFF6_FACTORS = (0.12, 0.10, 0.08, 0.06)


def _at_depth(table: tuple, depth: int) -> float:
    """``table[depth]``, clamped to its ends -- never an IndexError.

    Both per-depth tables above were tabulated at the CERTIFIED
    four-domain layout's depth and then indexed by nest depth with no
    bound, so ``--chain`` with four or more nests died on a bare
    ``IndexError: tuple index out of range`` -- inside the sizing fit
    loop, where it read as a crash rather than as a refusal.  That is
    precisely the 12 -> 3 -> 1 -> 0.5 -> 0.25 km ladder this program
    exists to run.

    Clamping is the DEFINED behaviour for a depth past the table, not a
    guess: both tables are monotone toward their inner end (spans get
    proportionally tighter, sixth-order damping gets weaker), so the
    last entry is the innermost value anybody certified, and a deeper
    nest inheriting it is the conservative continuation.  It is also the
    convention already in force elsewhere in this codebase --
    ``woof.da.nested_forecast._diff6_factor`` clamps the same table the
    same way for the same reason.

    This is a bound on a LOOKUP, not on a limit.  Nothing here softens a
    refusal: a nest that cannot be hosted inside its parent with the
    boundary clearance still raises :class:`DomainFitError` naming the
    extent, and the fit loop still prices every candidate against the
    card.
    """
    if not table:
        raise ValueError("per-depth table is empty")
    return table[min(max(int(depth), 0), len(table) - 1)]


def _child_span_fraction(depth: int) -> float:
    """Child linear extent as a fraction of its parent's, at ``depth``."""

    return float(_at_depth(_CHILD_SPAN_FRACTION, depth))


def _diff6_factor(depth: int) -> float:
    """``diff_6th_factor`` for the domain at ``depth`` (0 = root)."""

    return float(_at_depth(_DIFF6_FACTORS, depth))


def nest_diff6_factors(root_factor: float, domains: int) -> list[float]:
    """``diff_6th_factor`` for each domain of a ladder, root first.

    The root carries its suite's own value; each nest takes the certified
    depth ladder's value (:func:`_diff6_factor`) or its parent's,
    whichever is smaller, so sixth-order damping never grows inward.  The
    breakage this prevents: a suite that pins a weaker root damping than
    the ladder's second rung (the registry's MYNN, PBL-off and
    no-radiation suites pin 0.08, the sub-km default among them) was
    written with 0.08 on the parent and 0.10 on its child, a child damped
    harder than the parent that drives it.  A suite on the certified 0.12 root keeps exactly the
    certified ladder, because that ladder already falls with depth.
    """

    factors = [float(root_factor)]
    for depth in range(1, int(domains)):
        factors.append(min(factors[-1], _diff6_factor(depth)))
    return factors


#: What a card of nominal capacity NEVER hands to a process: the driver's
#: own reservation, the display/compositor allocations on a card that has
#: one, and the gap between a marketing "16 GB" and the 16,376 MiB the
#: silicon carries.
#:
#: MEASURED: an idle headless RTX 4080 (16,376 MiB physical = 15.99 GiB)
#: presents 15.33 GiB free to a fresh CUDA context -- a 0.66 GiB gap.  The
#: 16 GiB tier assumed the card would hand over its whole nominal size, so
#: every ladder it emitted was sized against 0.33 GiB of VRAM that does
#: not exist, landed 0.13-0.32 GiB over the real budget, and failed the
#: product's own ``woof check`` minutes after the wizard printed PASS.
#: 0.75 rounds the measured gap UP: a tier must be conservative against
#: real cards of its class, not equal to the best one.
CARD_UNAVAILABLE_VRAM_GIB = 0.75

#: ...and the same gap as a FRACTION, because it is not the same number
#: of gibibytes on every card.  The one 32 GiB free figure this codebase
#: has measured is 30.27 of 31.84 -- a 1.57 GiB gap, because that machine
#: also had a desktop on the card.  A tier has to be conservative against
#: the cards of its class that exist, not against the best one, so the
#: larger of the two forms binds.
CARD_UNAVAILABLE_VRAM_FRACTION = 0.06

#: How much of the budget the fit loop refuses to spend.  The loop grows
#: the grid until the envelope TOUCHES the budget, which is how every
#: emitted ladder landed 0.01-0.19 GiB from the wall -- a rounding error
#: away from a refusal, and with nothing left for the card to be one
#: driver revision less generous than it was when the config was written.
FIT_HEADROOM_FRACTION = 0.05
FIT_HEADROOM_MIN_BYTES = GIB // 4


def card_assumed_free_gib(vram_gib: float) -> float:
    """Free VRAM a card of ``vram_gib`` nominal capacity really presents.

    Never the nominal size: see :data:`CARD_UNAVAILABLE_VRAM_GIB`.
    """

    unavailable = max(CARD_UNAVAILABLE_VRAM_GIB,
                      CARD_UNAVAILABLE_VRAM_FRACTION * float(vram_gib))
    return max(0.0, float(vram_gib) - unavailable)


def fit_headroom_bytes(budget_bytes: int) -> int:
    """Budget the fit loop leaves unspent, so nothing lands on the wall."""

    return max(FIT_HEADROOM_MIN_BYTES,
               int(FIT_HEADROOM_FRACTION * max(0, int(budget_bytes))))


def vram_reserve_gib(vram_gib: float) -> float:
    """Flat VRAM reserve (GiB) by card capacity: WDDM/driver/CUDA context
    plus the near-capacity stability ceiling of consumer cards.

    RETIRED as the sizing path's reserve on 2026-08-01, and kept for the
    callers that have no experiment to price (``woof downscale``'s
    standalone child fit) and for the "your card is too small before we
    even start" refusal.  A FLAT figure was the whole of defect 4: the
    reserve's overhead term tracks the local-memory backing store of the
    SELECTED KERNEL SET, which measured 1.93 GiB for WSM6+MYNN and 3.94
    for NSSL2 double-moment -- so a fit loop assuming a flat 4.0 sized
    both NSSL2 profiles against one budget and then verified them against
    a smaller one, at every card size.  The sizing path now prices the
    reserve from the candidate experiment itself
    (:func:`woof.core.preflight.ReservePolicy.n0_alloc`), which is the
    same call ``woof check`` makes, so the two cannot disagree.

    The small-card figure was 3.0 GiB and it was a promise the preflight
    would not keep: ``ReservePolicy.n0_alloc`` charges the CUDA context
    plus the widest launched kernel's local-memory backing store plus a
    retention residual, which lands at 3.5-3.6 GiB on exactly those 12
    and 16 GiB cards.  Sizing a layout against a 3.0 GiB reserve and then
    handing it to a preflight applying 3.55 is how a test host G pilot got a
    wizard-certified config whose own check said the envelope did not
    fit.  4.0 -- the figure the 24 GiB tier already uses -- clears the
    measured reserve on both small tiers with about 0.45 GiB to spare.
    """
    if vram_gib <= 24.0:
        return 4.0
    return 6.0


#: Projection auto-selection bands (absolute latitude of --point):
#: below MERCATOR_MAX_LAT the Lambert cone is ill-conditioned and the
#: wizard selects Mercator; above LAMBERT_MAX_LAT it selects polar
#: stereographic; between them, hemisphere-correct Lambert conformal.
#: --projection overrides the choice explicitly.  All three projections
#: are oracle-gated against the pinned WRF v4.6.1 module_llxy.F
#: (tests/test_projection_oracle.py).
MERCATOR_MAX_LAT, LAMBERT_MAX_LAT = 25.0, 60.0
#: Clearance from the Lambert projection pole, shared with plan review.
#: Polar stereographic supports a cap containing its hemisphere's pole;
#: its forcing and static windows retain periodic longitude donors.
_POLE_CLEARANCE_CELLS = POLE_CLEARANCE_CELLS

#: Cells of pole clearance the point FIT sizes to, as distinct from the
#: clearance the refusal above enforces.  The fit works on a discretised
#: ladder -- every candidate is rounded to even mass points, and a nest
#: chain rounds again -- so a search that stopped exactly on the
#: refusal's own margin would emit layouts that sometimes land on the
#: wrong side of it.  Sizing to four times the margin puts the emitted
#: layout clear of the refusal instead of on its edge.
_FIT_POLE_CLEARANCE_CELLS = 4.0 * _POLE_CLEARANCE_CELLS

#: Largest ROOT extent, per axis, that a POINT request is sized to (km).
#:
#: A point carries no extent, so the fit has to choose one, and until
#: 2.7.3 the only thing that chose was the card: the search grew the
#: root until memory bound.  With streaming on (``--tiles auto``, what
#: the desktop asks for) memory stops binding at all, and a 7 GiB card
#: sized a 12 km root of 2326 x 1860 mass points -- 27,912 x 22,320 km,
#: wider than the Earth's circumference at that latitude, wrapped around
#: the projection pole, and therefore refused at plan review by the
#: pole guard for a request that named a point in the mid-latitudes.
#: Large resident cards reached the same place more slowly (180 GiB
#: sized 26,952 km).
#:
#: 6,000 km per axis is the product's own documented continental
#: examples, measured as GROUND: they run 60-80 degrees of longitude and
#: 30-50 of latitude (the range :data:`_WIDE_FOOTPRINT_DEGREES` and
#: :data:`_TALL_FOOTPRINT_DEGREES` record), which between 35 and 45 N is
#: about 4,700-7,300 km of longitude and 3,300-5,600 km of latitude.  A
#: 6,000 x 4,800 km root sits inside both spans, so a point is sized to
#: the largest domain this product actually shows anyone running, and
#: not to whatever the card holds.
#:
#: What it is NOT is a promise about the FETCH BOX, which this comment's
#: first version claimed it was.  Those two degree thresholds are
#: applied to the margined lat/lon box (:func:`oversized_footprint_advisory`
#: reads the ``--area`` string), not to the root, and a conformal root
#: of this size fans out well past them: measured on the desktop's own
#: argument shape, the box is 108 x 76 degrees at 30 N, 129 x 77 at
#: 41.5 N, and the source's full 360-degree band at 60 N, where a
#: 4,800 km tall Lambert domain reaches into the high Arctic.  So the
#: oversized-footprint advisory still fires on a capped point fit, by
#: design: the download IS large and the reader should hear it once.  It
#: fires saying which bound chose the size and naming
#: ``--point-extent-km`` and ``--polygon``, because on this bound the
#: card is no longer the lever (:func:`point_request_bound`).  A fetch
#: that spans the whole band keeps its own separate warning, which the
#: cap did not silence.
#:
#: It caps every fit that starts from a point -- :func:`fit_ladder` is
#: also the sizer behind ``woof domain-fit --point``, the starter
#: template's own door, at that template's own root dx -- and nothing
#: else.  A drawn area is sized to the drawing by
#: :func:`fit_polygon_ladder`, which never consults this number, so
#: asking for more ground than the cap is done by drawing it, with
#: ``woof domain --polygon``.
#:
#: The number is the DEFAULT of ``--point-extent-km`` on ``woof domain``
#: and ``woof domain-fit``, not a limit of the engine.  What the flag
#: cannot move is what keeps a point fit a valid domain: the projection's
#: polar envelope, one trip around the globe in longitude (a Mercator
#: root has no pole to stop it, so without this a large enough extent
#: wrapped the grid onto its own ground), the source's coverage and
#: one-crop bound, and the card (or, streamed, the tiling fit).  Nor can
#: it shrink a root below the smallest one its ladder hosts: an extent
#: under that gets the smallest root, and the plan summary says so
#: instead of reporting a cap that did not bind.
POINT_FIT_MAX_EXTENT_KM = 6000.0


def point_extent_argument(value: str) -> float:
    """``--point-extent-km`` as a finite positive number of kilometres."""

    import argparse as _argparse

    try:
        extent = float(value)
    except ValueError:
        raise _argparse.ArgumentTypeError(
            f"--point-extent-km must be a number of kilometres, got "
            f"{value!r}") from None
    if not math.isfinite(extent) or extent <= 0.0:
        raise _argparse.ArgumentTypeError(
            f"--point-extent-km must be a finite positive number of "
            f"kilometres, got {value!r}")
    return extent


def refuse_point_extent_on_polygon(point_extent_km, polygon) -> None:
    """A non-default ``--point-extent-km`` beside ``--polygon`` refuses.

    A drawn area is sized to the drawing and never reads the point
    extent, so accepting the value there would drop it silently and run
    a domain the reader did not ask for.
    """

    if polygon is not None and point_extent_km is not None \
            and float(point_extent_km) != POINT_FIT_MAX_EXTENT_KM:
        raise ValueError(
            f"--point-extent-km {float(point_extent_km):g} sizes a --point "
            "request; a --polygon is sized to the drawing and would drop "
            "it.  Remove --point-extent-km, or draw the ground you want")


#: Names for what decided a POINT request's size when the card did not,
#: as :func:`point_request_bound` and :func:`fit_ladder` report them and
#: :func:`point_fit_cap_note` speaks them.  They are constants because
#: three call sites compare against them and a fourth prints them; a
#: literal that drifted in one of the four would silently stop the fit
#: and the plan summary agreeing.
#:
#: Three are bounds that shrink a fit: the requested extent, the
#: projection pole, and one trip around the globe in longitude.  The
#: fourth is the floor: a requested extent below the smallest root the
#: ladder hosts gets that smallest root, which is not a cap binding and
#: must not be reported as one.
POINT_FIT_EXTENT_SCOPE = "REQUESTED EXTENT"
POINT_FIT_PROJECTION_SCOPE = "PROJECTION"
POINT_FIT_BAND_SCOPE = "LONGITUDE BAND"
POINT_FIT_FLOOR_SCOPE = "SMALLEST LAYOUT"
POINT_FIT_SCOPES = (POINT_FIT_EXTENT_SCOPE, POINT_FIT_PROJECTION_SCOPE,
                    POINT_FIT_BAND_SCOPE, POINT_FIT_FLOOR_SCOPE)

#: Degrees of latitude the SUGGESTED FORCING BOX keeps clear of a pole.
#:
#: The refusal above guards the domain footprint, but the fetch hint is
#: the footprint plus a margin, and clamping that at exactly +-90 made
#: the wizard print `--area 42.93,-45.56,90.00,83.48` for Tromso: a top
#: edge sitting on the very singularity the README says is refused, with
#: no comment.  `woof fetch` accepted it and downloaded 89 MB.  The
#: same 2-cell clearance the domain refusal enforces, expressed as
#: degrees of meridian at the root dx, keeps the suggestion accurate.
def pole_clearance_deg(root_dx_m: float = ROOT_DX_M) -> float:
    """The forcing box's pole clearance in degrees, at this root dx."""

    return _POLE_CLEARANCE_CELLS * float(root_dx_m) / 111_195.0


def max_fetch_abs_lat(root_dx_m: float = ROOT_DX_M) -> float:
    """The most poleward latitude a suggested forcing box will name."""

    return 90.0 - pole_clearance_deg(root_dx_m)


POLE_CLEARANCE_DEG = pole_clearance_deg()
#: The most poleward latitude the suggested forcing box will name.
MAX_FETCH_ABS_LAT = max_fetch_abs_lat()

#: Degrees of forcing margin beyond the root domain for the fetch hint.
#: ERA5 needs only interpolation-halo coverage; HRRR files are CONUS-wide
#: (``--area`` is a coverage check, not a crop, and the hint is clamped
#: into the grid's own envelope -- see ``fetch_area_hint``), so both
#: keep the small margin.  GFS uses
#: :func:`woof.fetch.gfs_suggested_fetch_margin_deg`: the GFS front
#: door takes every model lake's nearest source-water donor from the
#: crop, so the wizard's suggested area carries that documented margin
#: and each lake's donor is its nearest GFS water.
_FETCH_MARGIN_DEG = 2.0


def _fetch_margin_deg(source: str) -> float:
    if source == "gfs":
        from woof.fetch import gfs_suggested_fetch_margin_deg
        return gfs_suggested_fetch_margin_deg()
    return _FETCH_MARGIN_DEG

#: Forcing cadence per source: estimator LBC-interval sizing + the
#: ``&share/interval_seconds`` the emitted namelist.wps carries.
#:
#: REGISTRY-DERIVED since 2.5.0.  It used to be this three-entry literal:
#:
#:     {"era5": 21600.0, "gfs": 10800.0, "hrrr": 3600.0}
#:
#: and that dict was the whole reason `woof domain --source` offered three
#: sources while the product shipped sixteen runnable ones.  The
#: 2026-08-17 model battery hand-assembled a TOML and a namelist.wps for
#: every other model, typing this one number in from the packaged mapping.
#: It is a source's own published fact, so it lives in the source's own
#: registry row (:func:`woof.source_adapters.source_forcing_interval_seconds`)
#: and a new model reaches this door without touching this file.
SOURCE_FORCING_INTERVAL_S = MappingProxyType({
    source_id: source_forcing_interval_seconds(source_id)
    for source_id in wizard_planable_source_ids()
})


def _fetch_ladder_cadence_h() -> dict[str, int]:
    """The ``cadence = N`` an emitted ``[fetch]`` table carries, per source.

    The download ladder's spacing is the source's own native cadence -- a
    fetch that skipped valid times would hand the preparation a series
    with holes in it -- so this comes from the registry rather than from a
    second hand-written table.  It used to be ``{"era5": 6, "gfs": 3}``,
    and `--source gdas` was unreachable from this door precisely because
    nothing had ever added the third entry: opening the door without
    deriving this priced every source at the GFS spacing, so the GFS
    window planner refused a ``cadence`` key nobody had filled in.

    A source whose fetch takes no cadence at all carries no entry, and
    which sources those are is asked of the fetch module
    (:func:`woof.fetch.fetch_accepts_cadence`) rather than held here as
    a second spelling: the two used to be written separately, and the
    emission wrote a ``cadence`` key into a table whose own fetch refuses
    the flag.
    """

    from woof.fetch import fetch_accepts_cadence, fetch_front_door_sources

    return {
        source: int(source_forcing_interval_seconds(source) // 3600)
        for source in fetch_front_door_sources()
        if fetch_accepts_cadence(source)
        and not get_source_adapter(source).fetch_entire_window
        and source in SOURCE_FORCING_INTERVAL_S
    }


_SOURCE_CADENCE_H = _fetch_ladder_cadence_h()


def planable_sources() -> tuple[str, ...]:
    """Every source id this door can emit a runnable configuration for."""

    return wizard_planable_source_ids()


def resolve_source(raw: str, *, cadence_h: int | None = None) -> str:
    """RAW (an id or a registry alias) as the canonical source id.

    Fail-closed in three named ways, because "invalid choice: 'rap'" is a
    refusal that tells a reader nothing about why the model they can see in
    `woof prep --list-sources` cannot be planned for:

    * an unknown name lists the registry, so a misspelling is one glance
      from correct;
    * a registered row with no runnable route says what the row says about
      itself, so the reader learns the state of that model rather than the
      state of this parser;
    * a registered runnable row with no declared forcing cadence plans
      only with ``cadence_h`` (the caller's ``--cadence HOURS``): its
      spacing is whatever the history files it is handed carry, so the
      caller names it and that number becomes the ``interval_seconds``
      written.  Without one the refusal names the flag, because emitting
      a guessed ``interval_seconds`` would ask the preparation for
      boundary times the inputs may not carry.  Breakage: ``woof domain
      --source hex-global`` exited 2 even with ``--cadence 3``, so the
      documented ``woof prep --source hex-global`` line had no config to
      point at (2.9.2 hex-as-source door test).
    """

    try:
        adapter = get_source_adapter(raw)
    except ValueError:
        raise ValueError(
            f"--source {raw!r} is not a registered source; "
            f"woof domain plans for {', '.join(planable_sources())}.  "
            "`woof prep --list-sources` lists the whole registry, "
            "including the rows "
            "that are registered but have no runnable route yet") from None
    if not adapter.runnable:
        raise ValueError(
            f"--source {adapter.source_id}: this source is registered but "
            f"has no runnable initialization route ({adapter.status.value})"
            + (f" -- {adapter.composition_requirement}"
               if adapter.composition_requirement else "")
            + f".  A configuration emitted for it could not be prepared by "
              f"anything; plan with one of {', '.join(planable_sources())}")
    if adapter.forcing_interval_seconds is None and cadence_h is None:
        raise ValueError(
            f"--source {adapter.source_id}: this source declares no native "
            "boundary cadence (its spacing is whatever the history files "
            "you hand `woof prep` carry), so this door has no "
            "interval_seconds to write until you name it: pass --cadence "
            "HOURS, the spacing of those files (a woof hex export written "
            "every 3 h is --cadence 3).  The sources that declare their own "
            f"cadence ({', '.join(planable_sources())}) need no --cadence")
    from woof.source_adapters import canonical_source_token
    return canonical_source_token(raw)


def source_has_fetch_front_door(source: str) -> bool:
    """Can ``woof fetch`` go and get this source's bytes today?

    The wizard emits an advisory ``[fetch]`` table only when the answer is
    yes.  A table naming a source the fetch door does not serve is refused
    at every later config load, and a table that quietly loads would
    advertise a download nothing can make -- so the emission asks here and
    prints the manual acquisition route otherwise.
    """

    from woof.fetch import fetch_front_door_sources

    return get_source_adapter(source).source_id in fetch_front_door_sources()


def source_credential_notes(source: str) -> list[str]:
    """Pointer lines for what SOURCE needs configured and does not have.

    The registry row's CREDENTIAL column, wrapped for a terminal.  This
    used to be an ``if source == <one id>`` arm in two places, so a
    second source needing an account key would have needed two more
    arms; now a declared credential reaches every door that asks here,
    and a row that declares none produces no line at all.

    Never raises.  An unresolvable source id is the caller's business to
    refuse -- printing a traceback in place of a pointer would replace a
    helpful line with a broken command.
    """

    from woof.source_adapters import get_source_adapter
    from woof.source_credentials import absent_credential_notes

    try:
        adapter = get_source_adapter(source)
    except Exception:  # noqa: BLE001 - an advisory line, not a gate
        return []
    return absent_credential_notes(adapter.credentials)


def _candidate_fetch_hints(source: str) -> dict | None:
    """The ``[fetch]`` stub a sizing candidate carries, or ``None``.

    A candidate is rendered and reloaded through the real experiment
    loader, so it must be a file that LOADS: a ``[fetch]`` table naming a
    source the fetch door does not serve is refused there, which would
    have turned every new source's first fit iteration into a load error
    rather than a size.
    """

    return {"source": source} if source_has_fetch_front_door(source) else None


def source_fetch_takes_a_crop_box(source: str) -> bool:
    """Does this source's fetch accept an ``area``/``point`` crop?

    Asked of the fetch module rather than decided here, for the same
    reason the door question is: the hand-written transports subset at
    the publisher and the table routes take whole objects, and a
    ``[fetch]`` table carrying ``area`` for one of the latter prints a
    step 1 that exits 2 and is refused at every later config load.
    """

    from woof.fetch import fetch_accepts_area

    return fetch_accepts_area(source)


def source_reaches_forecast_leads(source: str) -> bool:
    """Does SOURCE publish forecast leads, or only analyses at valid times?

    The registry's ``max_forecast_hour`` is the whole answer: a reanalysis
    and an every-member analysis archive both declare 0, and the front door
    that used to spell this ``{"gfs", "gdas", "hrrr"}`` refused ``rap`` a
    lead RAP publishes 51 hours of.
    """

    return get_source_adapter(source).max_forecast_hour > 0


def forecast_start_hour_refusal(source: str, start_hour: int,
                                hours: float) -> str | None:
    """Why a run of ``hours`` cannot begin at lead f{start_hour}, or None.

    One spelling for every door that takes ``--forecast-start-hour`` (the
    domain wizard and ``--recipe``), asked from the source's registry row
    before any cycle is resolved or byte fetched.
    """

    if start_hour < 0:
        return "--forecast-start-hour must be a nonnegative forecast lead"
    if start_hour and not source_reaches_forecast_leads(source):
        return (
            f"--forecast-start-hour: {source} publishes no forecast "
            "leads (its registry row declares max_forecast_hour = 0), so "
            "every time in it is an analysis at its own valid time and "
            "there is no lead to begin at; name the analysis time you want "
            "with --cycle")
    horizon = get_source_adapter(source).max_forecast_hour
    if horizon and start_hour + hours > horizon:
        # Named here, from the source's own declared horizon, rather than
        # after the acquisition.  gdas stops at f009 and rap at f051; a
        # window that walks past either is a window the product never
        # published, and it used to be discovered by whichever stage first
        # went looking for the missing lead.
        return (
            f"--hours {hours} beginning at f{start_hour:03d} reaches "
            f"f{start_hour + hours:03d}, past {source}'s declared "
            f"f{horizon:03d} horizon; shorten the window, start earlier, or "
            "choose a source with a longer forecast")
    return None


def _fetch_cadence_h(source: str, start_hour: int,
                     hours: float | None = None, *,
                     cycle: datetime | None = None) -> int | None:
    """The fetch cadence this window can actually be taken on.

    The default is the source's usual spacing, and for a window starting
    at f000 that is what every prior release emitted.  A forecast LEAD
    changes the question: a fetch window must CONTAIN the lead it begins
    at, and ``woof fetch`` refuses one that does not.  So a config
    written with ``cadence = 3`` and ``forecast_start_hour = 4`` named a
    download that could never be made -- step 1 of the wizard's own
    printed recipe exited 2 with "f004 is not on the 3 h cadence", and
    for two leads in every three the one-command `woof go` path could
    not be made to work at all.

    The cadence is therefore chosen to divide the lead.  GFS publishes
    hourly through f120, so 1 h is available wherever it is needed; a
    window that would cross f120 hourly is refused by the fetch planner
    below with the structural reason, before the file is written.

    A window LENGTH changes it again (``hours``, when the caller knows
    it): a source whose ladder coarsens past some lead -- IFS every 6 h
    past f144, GEFS past f240, ICON-EU every 3 h past f078 -- cannot be
    fetched at its usual spacing across that lead, and the boundary
    series is one spacing, so the window takes the coarsest spacing it
    runs into (:func:`woof.fetch_routes.window_cadence`, which reads the
    route table's ladder rows).  A 240 h IFS run used to need ``--cadence
    6`` typed by hand and was refused at the default.  ``cycle`` asks that
    one cycle's ladder, for a door whose cycle is already named; without
    it any cycle hour whose ladder serves the window answers.
    """

    cadence = _SOURCE_CADENCE_H.get(source)
    if cadence is None:
        return cadence
    if start_hour and start_hour % cadence:
        # Hourly divides every integer lead, and is the only other cadence
        # these sources publish.
        return 1
    if hours is not None:
        from woof import fetch_routes
        route = fetch_routes.table_route(source)
        if route is not None:
            by_lead = fetch_routes.window_cadence(
                route, int(start_hour), math.ceil(hours), cycle=cycle,
                floor=cadence, round_up=True)
            if by_lead is not None:
                return by_lead
    return cadence


def fetch_window(source: str, hours: float, start_hour: int = 0,
                 cadence: int | None = None) -> tuple[int | None, int]:
    """(spacing, length) of the download a forecast of ``hours`` asks for.

    What this door writes into ``[fetch]``: the source's own file spacing
    (``cadence``, when the caller names none) and the length rounded up to
    a whole number of those steps, so a 3-hour forecast from files that
    come every 6 hours downloads hours 0 and 6.  The date guidance
    (:mod:`woof.source_availability`) asks its questions about this same
    download; asking the fetch about the 3-hour window no run requests
    raised inside a page request and failed the whole source list.  A
    source whose fetch takes no spacing downloads the length as asked.
    """

    if cadence is None:
        cadence = _fetch_cadence_h(source, start_hour, hours)
    if cadence is None:
        return None, math.ceil(hours)
    return cadence, max(cadence, math.ceil(hours / cadence) * cadence)

#: Default output cadences, root and nest, in seconds.
#:
#: They were bare literals inside :func:`_domain_tables` with no knob,
#: which made "how often does this write" a thing a reader could see in
#: the emitted TOML and not change.  Named here because they are now a
#: DEFAULT rather than a fixture: ``--history-interval`` overrides them.
#:
#: The nest writes four times as often as the root on purpose -- a nest
#: exists to resolve what the root cannot, over a window shorter than
#: the root's whole forecast, so its output is the point of running it.
DEFAULT_ROOT_HISTORY_INTERVAL_S = 3600.0
DEFAULT_NEST_HISTORY_INTERVAL_S = 900.0
DEFAULT_RESTART_INTERVAL_S = 3600.0

#: Grid-scale search bounds.  _MIN_SCALE puts the root at 60 x 48 mass
#: points, the smallest layout that still hosts the deepest ladder with
#: full Davies/blend clearance.
#:
#: _MAX_SCALE is deliberately NOT a physical limit -- it exists only so the
#: bisection has a finite upper bracket, and it must stay far enough above
#: what any real budget wants that MEMORY is what decides the answer.  It
#: was 8.0, which is 880 x 704 at the root, and that bound bound: every
#: single-domain budget at or above 64 GiB returned exactly 880 x 704 and
#: reported a comfortable fit, so 64, 96 and 180 GiB cards were all sized
#: like a 64 GiB one.  Raising it to 16.0 changes nothing for budgets that
#: were not already saturated and lets 96 GiB reach 1178 x 944 and 180 GiB
#: reach 1674 x 1340; raising it further to 32.0 changes nothing again,
#: which is the check that the bracket, not the bound, now decides.
#: 64.0 is chosen with that headroom on purpose.  If it ever binds the
#: wizard says so out loud rather than silently under-sizing -- see
#: fit_ladder.
_MIN_SCALE, _MAX_SCALE = 0.55, 64.0

#: The nine WPS_GEOG dataset directories the static builder opens (the
#: ``default`` geog_data_res selector; woof/static/build.py), relative to
#: the geog root -- one list, kept by woof.geog_assets.
from woof.geog_assets import WRF_DEFAULT_GEOG_DIRS as GEOG_DATASETS  # noqa: E402

#: Certified 49-mass-level vertical coordinate (50 full eta levels) --
#: the reference configuration's ladder.  Eta is normalized, so one
#: ladder serves any emitted model top: certified at 100 hPa, and run
#: A/B at the 50 hPa default on the 2026-08-24 plains receipt (same
#: levels, deeper column, byte-verified P_TOP in both arms' output).
_ETA_LEVELS = (
    1.0, 0.9978, 0.99519, 0.99212, 0.98849,
    0.98422, 0.97918, 0.97325, 0.96627, 0.95808,
    0.94846, 0.93719, 0.92402, 0.90866, 0.89079,
    0.87006, 0.84612, 0.81857, 0.78706, 0.75124,
    0.7108, 0.66556, 0.61547, 0.56067, 0.50519,
    0.45474, 0.40886, 0.36713, 0.32918, 0.29466,
    0.26328, 0.23473, 0.20877, 0.18516, 0.16369,
    0.14417, 0.12641, 0.11026, 0.09557, 0.08222,
    0.07007, 0.05902, 0.04898, 0.03984, 0.03153,
    0.02398, 0.0171, 0.01085, 0.00517, 0.0,
)

_PACKAGED_VTABLE = Path(__file__).parent / "data" / "vtables" / \
    "Vtable.ERA5_CDO"

#: The product's default physics/dynamics selections -- the reference
#: four-domain configuration's [shared] block with the microphysics slot
#: on Thompson (mp_physics 8: the wrf-matched-run scheme,
#: WRF's own tables packaged and hash-pinned): the MM5 surface layer
#: (91), Noah LSM (2), YSU PBL (1), RTE+RRTMGP radiation (4, the
#: ratified WRF-RRTMG 4/4 substitution), and the certified
#: diffusion/damping/acoustic settings.  Morrison (10) stays fully
#: selectable at its registry maturity label; its morr_rimed_ice knob is
#: Morrison-only and is deliberately absent here.
#: The default model top (Pa).  50 hPa (~20.6 km) -- WRF v4.6.1's own
#: p_top_requested Registry default (Registry.EM_COMMON:2275) and the
#: convection-allowing community standard.  The prior 10000 Pa (100 hPa,
#: ~16 km) default put the damp_opt=3 zdamp=5000 sponge base near
#: 10.9 km AGL, inside deep convection's anvil layer: measured on the
#: 2026-08-24 plains A/B, the 100 hPa arm's anvil-layer max updrafts
#: plateaued at ~17 m/s and collapsed at the sponge base while the
#: 50 hPa arm peaked at 22 m/s with natural decay near 13 km, carried
#: more >=40 dBZ core cells and higher cloud tops, scored equal-or-
#: better MRMS FSS, and ran 12.9% faster at the same VRAM (tests/
#: test_ptop_default.py carries the receipt).  Emissions bound this
#: default per source via :func:`emitted_model_top_pa`.
DEFAULT_MODEL_TOP_PA = 5000.0


def emitted_model_top_pa(source: str | None) -> float:
    """The model top (Pa) a config emitted for SOURCE carries.

    The default, unless the source's registry row declares a certified
    inventory top the default sits above (``certified_source_top_pa``)
    that its fetch cannot extend to the default
    (``extendable_source_top_pa``).  Without the bound, a bare
    ``woof domain --source X`` emission would ask for a model top its
    own source cannot cover and refuse at preparation, after the user
    already paid for the acquisition.  A source whose fetch reaches the
    default when asked (GFS: its certified ladder stops at 100 hPa, and
    every door that downloads for a config asks for the config's own
    top) gets the default like any other.
    """

    if source is None:
        return DEFAULT_MODEL_TOP_PA
    adapter = get_source_adapter(source)
    ceiling = adapter.certified_source_top_pa
    reach = adapter.extendable_source_top_pa
    if ceiling is None or (reach is not None
                           and float(reach) <= DEFAULT_MODEL_TOP_PA):
        return DEFAULT_MODEL_TOP_PA
    return max(DEFAULT_MODEL_TOP_PA, float(ceiling))


_SHARED_GRID_AND_DYNAMICS = {
    "nz": 49, "ztop": 20000.0, "p_top": DEFAULT_MODEL_TOP_PA,
    "eta_levels": _ETA_LEVELS,
    "hybrid_opt": 2, "etac": 0.2, "base_temp": 290.0,
    "time_step_sound": 4, "emdiv": 0.01,
    "hypsometric_opt": 2, "h_sca_adv_order": 5, "smdiv": 0.1,
    "moist_adv_opt": 1,
    "w_damping": 1, "damp_opt": 3, "zdamp": 5000.0, "dampcoef": 0.2,
    "khdif": 0.0, "kvdif": 0.0, "spec_zone": 1, "relax_zone": 4,
    "bldt": 0.0,
    # WRF nwp_diagnostics: wizard runs are convective forecasts whose
    # audience reads UH products, so the UP_HELI_MAX running-max
    # diagnostic ships ON (trajectory-inert; tests/test_uh_lifecycle.py).
    "nwp_diagnostics": 1,
}

#: Physics switches the ROOT domain carries rather than ``[shared]``.
_PER_DOMAIN_PHYSICS = ("radt", "cu_physics", "cudt_minutes",
                       "diff_6th_factor")

#: The wizard's default physics for real cases (gfs/era5; hrrr has its
#: own route-constrained default, HRRR_DEFAULT_PROFILE below).
#:
#: Owner directive 2026-08-06, after a shipped 48 h case: the default a
#: real case gets must be a CERTIFIED, NOCTURNALLY VALID suite -- both
#: radiation streams on, registry maturity read off the registry rather
#: than asserted here.  This profile is the registry's only user-ready
#: ``wrf-matched-run`` template with full lw+sw radiation
#: (RTE+RRTMGP 4/4 + Kain-Fritsch), it is declared by every route for
#: every source, it is FIRST-LIGHT section 3a's worked example, and it
#: is already the interactive door's default
#: (:data:`woof.domain_interactive.DEFAULT_PHYSICS_PROFILE_BY_SOURCE`)
#: -- the two doors now agree.  It replaced ``None`` (the unshipped
#: "product default suite", DEFAULT_SUITE_PHYSICS below, supported but
#: not WRF-verified), which remains reachable programmatically and is
#: NOT emitted by any door.  Validation profiles with asymmetric
#: radiation (shortwave on, longwave off) are never a default on any
#: door; they stay selectable explicitly and are emitted with the
#: nocturnal declaration (see render_config).
#:
DEFAULT_PHYSICS_PROFILE = MORRISON_PROFILE_ID

#: The default suite's physics switches, in the registry's OWN radiation
#: representation.
#:
#: v1.0.0 wrote ``ra_physics = 4`` (the legacy combined selector) while
#: every shipped profile -- and the physics registry -- writes the split
#: pair ``ra_physics = 0`` + ``ra_lw_physics = 4`` + ``ra_sw_physics =
#: 4``.  The two are semantically identical, and
#: ``radiation_scheme_ids`` resolves both to (4, 4), but the runner's
#: guard compares the raw switch dicts, so no wizard-emitted config
#: could ever pass it.  Emitting the split form is the fix at the
#: source, and it also removes a real misreading hazard: a pilot report
#: read the profiles' ``ra_physics: 0`` as "radiation off" when three of
#: those profiles run RTE+RRTMGP on both streams.
DEFAULT_SUITE_PHYSICS = {
    "moist": True, "moist_cq": True, "mp_physics": 8, "top_lid": False,
    "epssm": 0.5, "wrf_rrtmg_compatibility":
        "wrf-rrtmg-4-4-to-rte-rrtmgp-v1",
    "ra_physics": 0, "ra_lw_physics": 4, "ra_sw_physics": 4,
    "sf_sfclay_physics": 91, "sf_surface_physics": 2,
    "bl_pbl_physics": 1, "terrain_opt": 1,
    "km_opt": 4, "diff_6th_opt": 2, "diff_6th_slopeopt": 1,
    # Per-domain (root values; nests override in _domain_tables).
    "radt": 12.0, "cu_physics": 1, "cudt_minutes": 5.0,
    "diff_6th_factor": _DIFF6_FACTORS[0],
}

#: The profiles the prepared single-domain forecast runner accepts, in
#: the order the help lists them.
#:
#: DEFINED in :mod:`woof.physics_menu` and imported above, with the
#: ordering rationale (nocturnally valid suites first, the two
#: legacy-RRTMG Thompson entries as the only full-radiation suites the
#: nested HRRR route admits) written out there.  Re-exported under this
#: name because every reader in the tree spells it
#: ``domain_wizard.WIZARD_PHYSICS_PROFILES``.


def _radiation_words(switches: dict) -> str:
    """Plain language for what a profile's radiation switches DO.

    ``ra_physics = 0`` beside ``ra_lw_physics = 4`` means RTE+RRTMGP
    longwave, not "radiation off" -- a reading that has already been got
    wrong once in a pilot report, because the split and legacy
    representations look alike and only one of them is the truth.  The
    wizard therefore never prints the raw switches without the words.
    """

    # Selector 4 is "RRTMG" in WRF's spelling and woof implements it two
    # ways -- the exact legacy-RRTMG transcription and the RTE+RRTMGP
    # substitution -- so the words follow ``ra_rrtmg_variant`` rather than
    # calling every 4 RTE+RRTMGP.  A header that names the wrong solver is
    # the same misreading hazard this function exists to remove.
    legacy = switches.get("ra_rrtmg_variant") == RRTMG_VARIANT_LEGACY
    names = {0: "OFF", 1: "Dudhia",
             4: "legacy RRTMG" if legacy else "RTE+RRTMGP"}
    lw = int(switches.get("ra_lw_physics", -1))
    sw = int(switches.get("ra_sw_physics", -1))
    if (lw, sw) == (-1, -1):
        lw = sw = int(switches.get("ra_physics", 0))
    return (f"longwave {names.get(lw, lw)}, "
            f"shortwave {names.get(sw, sw)}")


def prepared_route_physics_notice(profile: str | None,
                                  source: str) -> list[str]:
    """The emitted suite's standing on the prepared single-domain route.

    Owner ruling 2026-07-31: the prepared single-domain forecast runner
    executes any suite the engine implements, exactly as this file
    writes it -- the profile whitelist and its exact-equality refusal
    are gone, and verification status is reported, never gating.  These
    lines are the --explain detail; the always-printed ``physics:``
    summary line already carries the one-sentence status.  Picking a
    shipped profile still quietly changes the science -- several run no
    cumulus and shortwave Dudhia with longwave OFF -- so each candidate
    is still named with what it actually runs.
    """

    if source not in ("gfs", "hrrr"):
        return []
    if profile is not None:
        return [
            "note: this config is bound to a shipped profile, so every "
            "runner enforces it switch for switch, exactly as emitted."
        ]
    lines = [
        "note: the suite above runs as written on the prepared "
        "single-domain route and the multi-domain (domain-tree) route "
        "alike; no --physics-profile is required, and its "
        "WRF-verification status is reported in the run receipts."
        if source == "gfs" else
        "note: the HRRR route's cold-start evidence contract is keyed "
        "by shipped profile, so it prepares "
        f"{HRRR_DEFAULT_PROFILE} when none is named -- Thompson "
        "microphysics, MYNN PBL and surface layer, RUC, "
        "RTE+RRTMGP longwave AND shortwave and no "
        "cumulus at 3 km; pass --physics-profile <id> to choose "
        "another, the same composition on the legacy RRTMG engines "
        "included.",
        "  --physics-profile <id> binds the config to a shipped suite "
        "and every runner then enforces it switch for switch.  What "
        "each one ACTUALLY runs:",
    ]
    for candidate in WIZARD_PHYSICS_PROFILES:
        lines.append(f"    {physics_summary(candidate)}")
    return lines


#: Longitude span, in degrees, past which the sized fetch box gets a word
#: said about it.
#:
#: Not a limit and not a refusal: the sizer's job is to use the card it
#: was given, and on a 32 GiB card one 12 km domain legally fills it --
#: an owner's first emission covered 91 degrees of latitude and 152 of
#: longitude with 0.00 GiB of headroom, which is correct arithmetic and
#: an absurd first run.  The number below is a "that is bigger than you
#: probably meant" threshold, chosen so a continental domain (the
#: documented examples run 60-80 degrees of longitude at 24 GiB) passes
#: without comment.
_WIDE_FOOTPRINT_DEGREES = 90.0

#: The same "bigger than you probably meant" bar for latitude.  The
#: longitude test alone let a Linux ``--card 24gb --ladder 12`` print a
#: 144 x 88 degree domain behind a 174-degree fetch box without the
#: latitude -- pole to equator and then some -- being mentioned at all.
#: Documented continental examples run 30-50 degrees of latitude.
_TALL_FOOTPRINT_DEGREES = 60.0


def _area_span_degrees(area: str) -> tuple[float, float] | None:
    """``(lat_span, lon_span)`` for a ``S,W,N,E`` box, or None."""

    parts = str(area).split(",")
    if len(parts) != 4:
        return None
    try:
        south, west, north, east = (float(value) for value in parts)
    except ValueError:
        return None
    lon = east - west
    if lon < 0:
        lon += 360.0
    return north - south, lon


def oversized_footprint_advisory(area: str, *,
                                 request_bound: str | None = None
                                 ) -> list[str]:
    """Say when the sized domain fills the card rather than the map.

    An advisory, never a refusal, and it changes no sizing: the layout
    is already chosen by the time this runs.  What it changes is whether
    a first-time reader learns that the hemisphere-wide fetch box they
    are about to download is a single flag away from something smaller.

    It used to open "sized to fill your card, not your map".  That
    asserted a CAUSE this function cannot see, and since 2.5.0 gave the
    fit a servable-crop bound the cause is sometimes false: on a large
    card the wizard now stops on the source and says so on stderr, and
    an advisory blaming the card two lines later contradicts it.  The
    sentence states what the thresholds above actually measure -- the
    box is much bigger than the documented examples -- and keeps the
    remedy, which is true either way.

    ``request_bound``, added in 2.7.3, is the other half of that same
    correction and the reason the sentence is not one string.  The point
    fit now stops on bounds the REQUEST carries rather than on the card
    (:func:`point_request_bound`), and on those the card-shaped remedy
    is not merely unhelpful, it is inert: with streaming on, the mode
    the desktop asks for, memory never binds at all, so every
    ``--vram-gib`` above the refusal floor emits the identical grid.
    Measured on the shipped 2.7.2 door's own argument shape, that was
    every mid-latitude point request -- the advisory fired at 30, 41.5,
    48.5 and 60 N and named a flag that changed nothing.  When a request
    bound is in force the sentence names ``--point-extent-km``, which is
    the direct way to ask a point for less ground, and ``--polygon``,
    which is how it asks for different ground; those are the levers
    that still move the answer.  When the fit sat on the smallest root
    its ladder hosts, neither of them can make it smaller, and the
    sentence names ``--root-dx`` instead.  It makes no claim about
    ``--vram-gib`` in that branch: a resident fit (``--tiles off``) on a
    small enough
    card is still bounded by memory below the extent cap, and saying
    otherwise would install the mirror image of the defect being fixed.

    Two things the first version got wrong, both found by a wheel user
    on Linux at ``--card 24gb --ladder 12``.  It measured longitude
    only, so a box 88 degrees tall passed unremarked; and its remedy
    named ``--card 12gb``, a tier not every platform runs, while saying
    nothing about the ``--area`` on the very fetch command printed two
    lines below -- the number a reader looks at when they wonder what
    this will cost them.  The sentence now names the box as the fetch
    box, and says why narrowing that flag alone is not the fix: the
    download exists to cover the domain, so the domain is what has to
    get smaller.
    """

    spans = _area_span_degrees(area)
    if spans is None:
        return []
    lat_span, lon_span = spans
    if (lon_span < _WIDE_FOOTPRINT_DEGREES
            and lat_span < _TALL_FOOTPRINT_DEGREES):
        return []
    box = (f"this domain is much wider than the documented examples: the "
           f"fetch command below downloads a {lon_span:.0f} x "
           f"{lat_span:.0f} degree --area box")
    if request_bound == POINT_FIT_FLOOR_SCOPE:
        return [
            f"{box}.  This is the smallest root the ladder hosts, so no "
            f"card and no --point-extent-km makes it smaller -- a finer "
            f"--root-dx KM does, for a smaller first run; narrowing "
            f"--area on its own would starve the domain it feeds"]
    if request_bound is not None:
        return [
            f"{box}.  A point carries no extent, so the fit chose this "
            f"one and stopped on the {request_bound}, not on the card "
            f"-- lower --point-extent-km, or draw the ground you want "
            f"with --polygon, for a smaller first run; narrowing --area "
            f"on its own would starve the domain it feeds"]
    return [
        f"{box}, so pass --vram-gib N (or a "
        f"finer --root-dx KM) for a smaller first run -- narrowing "
        f"--area on its own would starve the domain it feeds"]


def _guard_exports_block(profile: str | None) -> str:
    """Environment the printed chain needs, printed with the chain.

    The mp8 runners under ``tools/`` keep a launch contract the library
    itself retired: both variables must be set in the SHELL that starts
    the chain.  Preparation launches the forecast runner as a subprocess
    and inherits the environment, so one export pair covers both stages
    -- but only if the reader knows to type it, and nothing printed it.
    A field run of the shipped 1.5.0 wheel discovered the requirement by
    failing twice and then went looking for the table root by hand.

    Empty for every other profile, so the emitted block stays exactly as
    short as it was for the suites that need nothing.
    """

    if profile != THOMPSON_PROFILE_ID:
        return ""
    from woof.physics_compat import thompson_guard_exports

    lines = "".join(f"  {line}\n" for line in thompson_guard_exports())
    return (
        "# this suite's runners are gated on two environment variables "
        "-- export them in\n"
        "#   THIS shell before the chain: preparation launches the "
        "forecast runner as a\n"
        "#   subprocess, so one pair covers both stages.  The root below "
        "is the one this\n"
        "#   install resolves; every asset in it is byte-checked before "
        "GPU setup.\n"
        + lines)


def hrrr_route_commands(out: "Path", exp: ExperimentConfig, *,
                        profile: str | None, data_dir: str,
                        cycle: "datetime | None" = None,
                        forecast_start_hour: int = 0) -> str:
    """The HRRR chain, with every file this emission just wrote bound.

    HRRR reaches the GPU through the native front door: neither ``woof
    run`` (the ERA5 ``[case_data]`` route) nor ``woof go`` (whose five
    commands do not compose this route's stage vocabulary) drives it,
    and it is NOT prepared with ``rw-wps``, which is the GFS door.
    Until 2026-08-01 a multi-domain HRRR emission was told to use
    exactly that, because the multi-domain branch of
    :func:`final_step_command` returned before the HRRR one was ever
    reached.

    **The commands printed here are the SHIPPED ones**, rendered from
    :func:`woof.stage_cli.staged_route_commands` -- the same helper
    ``woof go``'s hrrr refusal and ``woof sim``'s unbindable-tree
    refusal render, so a reader cannot be handed three spellings of one
    route.  What this block printed until 2026-08-18 was the route's
    INTERNALS: ``python -m tools.prepare_hrrr_wrf`` and ``python
    tools/hrrr_single_domain_benchmark.py`` -- two ``tools/`` paths a
    pip wheel does not contain at all -- plus ``python -m
    woof.hrrr_hierarchy_direct`` and ``python -m
    woof.prepared_domain_tree_forecast``.  Every one of those is a
    program ``woof prep``/``woof sim`` spawn (MEASURED: ``woof prep
    --source hrrr ... --dry-run`` prints each line verbatim), so the
    old block was machinery where a door exists.

    Every value this emission knows is bound -- the four input files,
    the cycle, the lead, the run length, the cadence, the profile, and
    ``--statics-corridor`` on the hierarchy stage when the config
    declares a ``[relocation]`` follow source.  What is left as a
    placeholder is what cannot exist yet: the WPS_GEOG root, which is
    the reader's install, and the source manifest's own digest, which
    ``woof fetch`` prints.  The FORECAST stage now asks for no
    placeholder at all -- ``woof sim`` reads the preparation's digests
    off the bundle it is pointed at, so the two ``<printed by the
    hierarchy>`` / ``<sha256 of that file>`` values a reader used to
    have to produce by hand are gone.

    **Every time printed here is the CYCLE, and the lead is printed
    beside it.**  Model time zero (cycle + K) is derived by each stage,
    never typed.  Both stages used to be handed one ``--valid-time``
    string, and the two stages read that flag differently -- the
    preparer as the cycle (it opens ``hrrr.tHHz.wrfnatfNN.grib2``), the
    hierarchy as the model start (it is compared to the namelist's
    start_time).  At lead 0 those are the same instant, which is why one
    string served both for four releases; at lead K one of them is
    wrong by K hours.  Printing the cycle and the lead separately means
    the same two values appear on every line of the chain and neither
    stage has to be told a time the other stage computed.
    """
    paths = route_input_paths(Path(out))
    printed = {key: _printed_path(value) for key, value in paths.items()}
    source_root = _printed_path(data_dir)
    root, tree = "hrrr-root-prep", "hrrr-hierarchy"
    if cycle is None:
        # No lead was resolved by the caller: the model start IS the
        # cycle, which is what every pre-lead emission printed.
        cycle = exp.start_time - timedelta(hours=forecast_start_hour)
    cycle_text = cycle.strftime("%Y-%m-%d_%H:%M:%S")
    lead = ((f"--forecast-start-hour {forecast_start_hour}",)
            if forecast_start_hour else ())
    run_seconds = int(exp.run_seconds)
    cadence = int(exp.domains[0].history_interval_s)
    profile_flag = ((f"--physics-profile {profile}",)
                    if profile is not None else ())
    # The hierarchy stage's corridor flag, on exactly the configs that
    # need it.  Derived from the corridor module's own follow predicate
    # -- the same function `woof go`'s plan and run-plan's decision
    # read -- so a config whose printed chain omits this flag is a
    # config no door would have added it for.  A pasted chain that
    # forgot it would prepare a bundle the last line of the same chain
    # refuses.
    from woof.stage_cli import staged_route_commands
    from woof.static.corridor import config_declares_follow_source

    corridor = (("--statics-corridor",)
                if config_declares_follow_source(exp) else ())
    manifest = f"{source_root}/SHA256SUMS"
    manifest_digest = "<printed by woof fetch>"
    prepare_arguments = (
        f"--source-root {source_root}",
        f"--source-sha256s {manifest}",
        f"--source-sha256s-sha256 {manifest_digest}",
        f"--experiment-config {_printed_path(out)}",
        f"--domain-spec {printed['target_domain']}",
        f"--namelist-input {printed['namelist_input']}",
        # Handed to the PREPARATION, not just to the hierarchy: the
        # forecast stage's HRRR manifest inventory requires a
        # wps_namelist role, and the preparer records the role only if
        # it is given the file.  A chain that omitted it prepared a
        # bundle `woof sim` could not read at all -- which is how HRRR
        # came to be sent to a benchmark script instead of a forecast.
        f"--wps-namelist {printed['wps_namelist']}",
        "--geog-root <your WPS_GEOG>",
        *profile_flag,
        f"--valid-time {cycle_text}",
        *lead,
        f"--run-seconds {run_seconds}",
        f"--history-interval-seconds {cadence}",
    )
    prepare_command, root_forecast = staged_route_commands(
        "hrrr", prep_arguments=prepare_arguments, prepared_root=root,
        outdir="hrrr-forecast", indent="  ", wrap=True)
    prepare = _guard_exports_block(profile) + prepare_command + "\n"
    header = (
        "# HRRR runs the native route -- not `woof run` and not `woof "
        "go`, whose five\n"
        "#   commands do not compose this route's stages.  Two commands "
        "do, and they are\n"
        "#   the shipped ones.  Every input below was written beside this "
        "config; the\n"
        "#   placeholders are your WPS_GEOG and the digest `woof fetch` "
        "printed.\n")
    if len(exp.domains) == 1:
        # The forecast stage is `woof sim`, and that is not a rewording
        # of what stood here.  This block used to print
        # `tools/hrrr_single_domain_benchmark.py` under a comment saying
        # HRRR "does not reach woof.prepared_single_domain_forecast --
        # that runner's --source takes gfs/era5/20crv3".  It does reach
        # it: hrrr is in that runner's SUPPORTED_SOURCES, the
        # preparation publishes proof.json plus the authorities the
        # runner binds on every run, and `woof sim` derives the three
        # digests from them.  A reader was being sent to a benchmark
        # script -- one that lives under tools/ and is therefore absent
        # from every wheel -- for a run the shipped door does.
        return (header + prepare + root_forecast + "\n"
                + "# the second command reads the preparation's own "
                "digests off the bundle;\n"
                "#   nothing is copied by hand.  A ladder with a nest "
                "(--ladder 12-3, or\n"
                "#   --root-dx/--chain) takes one more `woof prep` "
                "between them: the\n"
                "#   hierarchy stage, printed for those configs.")
    hierarchy_arguments = (
        f"--root-preparation {root}",
        f"--domain-spec {printed['target_domain']}",
        f"--wps-namelist {printed['wps_namelist']}",
        f"--namelist-input {printed['namelist_input']}",
        f"--stock-wrf-namelist-input {printed['stock_namelist_input']}",
        "--geog-root <your WPS_GEOG>",
        f"--source-sha256s {manifest}",
        f"--source-sha256s-sha256 {manifest_digest}",
        f"--valid-time {cycle_text}",
        *lead,
        *corridor,
        # The config's acknowledgements: this stage imports the
        # namelists, which cannot spell them.
        *(f"--ack {acknowledgement}"
          for acknowledgement in exp.acknowledgements),
    )
    # The tree runner binds its preparation receipt rather than a
    # namelist digest, so --wps-namelist is left off the forecast line:
    # printing a flag the runner does not read is how a reader learns a
    # printed chain is approximate.
    hierarchy, tree_forecast = staged_route_commands(
        "hrrr", prep_arguments=hierarchy_arguments, prepared_root=tree,
        outdir="hrrr-forecast", experiment_config=_printed_path(out),
        wps_namelist=False, indent="  ", wrap=True)
    return (header + prepare + hierarchy + "\n" + tree_forecast + "\n"
            + "# the last command reads the hierarchy's own preparation "
            "receipt off the\n"
            "#   tree it is pointed at; no digest is copied by hand.")


def local_staging_lines(source: str, *, cycle: str, hours, cadence,
                        start_hour: int, area: str, out: str) -> list[str]:
    """How to fill a hand-staged source's folder, read from its row.

    Two kinds of line, both from the source's refusal row in the
    acquisition-route table: the provider request for the bytes no fetch
    door serves (dates, hours and area of THIS config filled in, one
    request per date so no request asks for an hour outside the window),
    and a ``woof fetch`` line for each supplement a fetch door does
    serve, written into the same folder.  Empty for a source whose row
    declares no folder layout, and nothing here names a model.
    """

    from woof import fetch_routes
    from woof.fetch import parse_area

    layout = fetch_routes.source_root_layout(source)
    if layout is None:
        return []
    lines: list[str] = []
    request = layout["request"]
    try:
        first = datetime.strptime(str(cycle), "%Y-%m-%dT%H") + timedelta(
            hours=int(start_hour or 0))
        step = int(cadence) if cadence else int(
            source_forcing_interval_seconds(source)) // 3600
        box = parse_area(str(area)).as_cds()
    except (TypeError, ValueError):
        request = None
    if request is not None:
        days: dict[str, list[str]] = {}
        for index in range(math.ceil(float(hours) / step) + 1):
            when = first + timedelta(hours=index * step)
            days.setdefault(f"{when:%Y-%m-%d}", []).append(f"{when:%H:%M:%S}")
        lines.append(f"#   {request['dataset']} request"
                     + (", one per date:" if len(days) > 1 else ":"))
        lattice = request.get("lattice_deg")
        if lattice:
            # Inward onto the provider's lattice: the points the surface
            # analysis fetched for this same --area delivers, so the two
            # files share exactly one grid whichever way the provider
            # anchors an interpolated request's area.
            north, west, south, east = box
            box = [math.floor(north / lattice + 1e-9) * lattice,
                   math.ceil(west / lattice - 1e-9) * lattice,
                   math.ceil(south / lattice - 1e-9) * lattice,
                   math.floor(east / lattice + 1e-9) * lattice]
        area_text = "/".join(f"{value + 0.0:g}" for value in box)
        for day, times in days.items():
            lines.append(f"#     {request['keywords']} date={day} "
                         f"time={'/'.join(times)} area={area_text}")
    for row in layout["supplements"]:
        if row["fetch"] is None:
            continue
        donor = str(row["fetch"]["source"])
        parts = [f"woof fetch --source {donor}", f"--cycle {cycle}",
                 f"--hours {hours}"]
        if source_fetch_takes_a_crop_box(donor):
            parts.append(f"--area={area}" if str(area).startswith("-")
                         else f"--area {area}")
        if cadence is not None and cadence != _SOURCE_CADENCE_H.get(donor):
            parts.append(f"--cadence {cadence}")
        if fetch_routes.supplement_fetch_retrieves(row["fetch"]):
            parts.append("--retrieve")
        if start_hour:
            parts.append(f"--forecast-start-hour {start_hour}")
        parts.append(f"--out {out}")
        lines.append(f"#   and beside it the {row['role']} file "
                     f"({row['match'][0]}), from this fetch:")
        lines.append(" ".join(parts))
    return lines


def final_step_command(out: "Path", *, source: str, profile: str | None,
                       domain_count: int, data_dir: str | None,
                       case_data: dict | None,
                       exp: ExperimentConfig | None = None,
                       cycle: "datetime | None" = None,
                       forecast_start_hour: int = 0) -> str:
    """Name the executable route declared by the emitted source's capabilities."""
    printed = _printed_path(out)
    if case_data is not None:
        return f"woof run {printed}"
    from woof.runplan import PlanError, prepared_chain_for_source

    try:
        prepared_chain_for_source(source, source_root=data_dir)
    except PlanError as error:
        from woof.explain import split
        return "# " + split(str(error))[0]
    command = f"woof go {printed}"
    if data_dir is not None:
        command += f" --data-dir {_printed_path(data_dir)}"
    return command


#: Existing source recommendation. Valid user settings do not depend on it.
HRRR_DEFAULT_PROFILE = ROUTE_DEFAULT_PHYSICS_PROFILE


def resolved_physics_profile(source: str, requested: str | None, *,
                             finest_dx_m: float | None = None,
                             domains: int = 1) -> str | None:
    """The profile this emission actually binds.

    ``finest_dx_m`` is the finest grid spacing of the emission, when the
    caller knows it: the default by grid spacing
    (:data:`woof.physics_menu.SPACING_DEFAULTS`) binds ahead of the
    source's own default, so a sub-km domain with no --physics-profile
    gets the suite that row names wherever the source's route admits it
    for ``domains`` domains.

    An explicit ``--physics-profile`` always wins -- including one the
    HRRR routes will refuse, which is refused at emission with the
    switch named rather than silently replaced.

    The DEFAULT is derived, not tabled.  This function used to read
    ``if source == "hrrr": return HRRR_DEFAULT_PROFILE`` and hand every
    other source the gfs/era5 default -- which is a branch on a model
    name, and it only answered correctly because exactly one source had
    ever had a route gate written for it.  A second gated source would
    have been handed a default its own route refuses, and a bare run on
    it could not start.

    :func:`woof.physics_menu.default_profile_for` computes it instead:
    the first suite in this module's listed order that the source's
    route admits and that runs both radiation streams.  That reproduces
    both defaults this product shipped -- ``tests/test_physics_menu.py``
    binds the native HRRR route's answer to
    :data:`woof.hrrr_route_inputs.ROUTE_DEFAULT_PHYSICS_PROFILE` and
    the rest to :data:`DEFAULT_PHYSICS_PROFILE` -- and it is what lets a
    source registered tomorrow get a working default as table work.
    """
    if requested is not None:
        return canonical_template_id(requested)
    from woof.physics_menu import default_profile_for

    return default_profile_for(source, finest_dx_m, domains)


def finest_spacing_m(root_dx_m: float, ratios) -> float:
    """The finest grid spacing a ladder reaches, in metres."""

    return float(root_dx_m) / math.prod(int(ratio) for ratio in ratios)


def profile_switches(profile: str | None) -> dict:
    """Every physics switch for PROFILE, or for the default suite."""

    if profile is None:
        return dict(DEFAULT_SUITE_PHYSICS)
    return single_domain_runtime_switches(profile)


def physics_summary(profile: str | None, *,
                    cu_physics: int | None = None,
                    switches: dict | None = None,
                    label: str | None = None) -> str:
    """One line naming what the emitted suite actually runs.

    ``cu_physics`` overrides the suite's cumulus switch with the one the
    EMISSION carries.  The wizard retires the scheme on a
    convection-permitting root (:func:`_domain_tables`), and a summary
    line that reads "Kain-Fritsch cumulus" over a file whose root says
    ``cu_physics = 0`` is the same class of misreading hazard that
    :func:`_radiation_words` exists to remove -- worse here, because the
    line and the table it describes sit in the same file.  Callers
    describing a SUITE rather than an emission leave it None.

    ``switches`` and ``label`` describe a switch table that is no suite:
    a physics mix's root as the file runs it (:func:`with_physics_mix`).
    """

    if switches is None:
        switches = profile_switches(profile)
    selected = (int(switches["cu_physics"]) if cu_physics is None
                else int(cu_physics))
    cumulus = ({1: "Kain-Fritsch cumulus",
                3: "Grell-Freitas cumulus"}.get(selected,
                                                "parameterized cumulus")
               if selected
               else "NO cumulus parameterization")
    if label is None:
        label = profile if profile is not None else (
            "product default suite (supported, not yet WRF-verified; every "
            "runner executes it as written)")
    return (f"{label}: mp_physics {switches['mp_physics']}, "
            f"{_radiation_words(switches)} (radt "
            f"{float(switches['radt']):g} min), {cumulus}, "
            f"bl_pbl_physics {switches['bl_pbl_physics']}, "
            f"sf_surface_physics {switches['sf_surface_physics']}")


def shared_physics(profile: str | None) -> dict:
    """The ``[shared]`` block for one physics suite.

    A named profile is taken from :mod:`woof.physics_compat`, never
    restated here: the prepared single-domain forecast runner compares
    an experiment's switches to that same registry for exact equality,
    so an emitted config passes its guard by construction.
    """

    switches = profile_switches(profile)
    for key in _PER_DOMAIN_PHYSICS:
        switches.pop(key, None)
    return {**_SHARED_GRID_AND_DYNAMICS, **switches}


#: The default suite's [shared] block.
_SHARED_CERTIFIED = shared_physics(DEFAULT_PHYSICS_PROFILE)


class DomainFitError(ValueError):
    """The requested ladder cannot fit; only typed memory failures are retryable.

    ``budget_bytes`` is the budget a layout's ``phases`` were held to when
    the refusal is that the layout does not fit the card.  With both,
    :meth:`memory_record` gives the figures the sentence prints, so a
    front end shows how much a too-big draft needs without reading them
    back out of the words, which are worded differently for a source
    whose preprocessing is priced and for one whose preprocessing is not.
    """

    def __init__(self, message, *, resource=None, phases=None,
                 budget_bytes=None):
        super().__init__(message)
        self.resource = resource
        self.phases = phases
        self.budget_bytes = budget_bytes

    def memory_record(self) -> dict | None:
        """The priced layout's peak envelope against its budget, in bytes.

        The fields :func:`fit_memory` gives for a layout that fits, as far
        as a refusal knows them.  ``None`` when this refusal is not the
        card's, or its budget was not positive.
        """

        if self.phases is None or self.budget_bytes is None \
                or self.budget_bytes <= 0:
            return None
        return {
            "peak_envelope_bytes": int(self.phases.peak_envelope_bytes),
            "budget_bytes": int(self.budget_bytes),
            "binding_phase": self.phases.binding_phase,
        }


class DomainFitCancelled(RuntimeError):
    """An author cancelled a fit before accepting or publishing it."""


def check_fit_cancelled(cancelled=None):
    if cancelled is not None and cancelled():
        raise DomainFitCancelled("Domain fitting cancelled")


#: Appended to every --point parse refusal: the one form that cannot be
#: mis-parsed no matter how the shell or argparse feels about a leading
#: minus sign.
_POINT_FORM_HINT = (
    " -- southern and western points are ordinary here; both "
    "'--point -33.87,151.21' and '--point=-33.87,151.21' are accepted")


def _parse_point(raw: str) -> tuple[float, float]:
    parts = raw.split(",")
    if len(parts) != 2:
        raise ValueError("--point must be lat,lon in decimal degrees"
                         + _POINT_FORM_HINT)
    try:
        lat, lon = (float(part) for part in parts)
    except ValueError as error:
        raise ValueError(
            "--point must be lat,lon in decimal degrees"
            + _POINT_FORM_HINT) from error
    if not (math.isfinite(lat) and math.isfinite(lon)):
        raise ValueError("--point coordinates must be finite")
    if not -90.0 <= lat <= 90.0:
        raise ValueError(
            f"--point latitude {lat:g} must lie within [-90, 90]")
    if not -180.0 <= lon <= 180.0:
        if -360.0 <= lon <= 360.0:
            # float(): _wrap180 is array code (np.where) and returns a 0-d
            # ndarray even for a scalar.  Without the cast the wrapped
            # longitude stays an ndarray all the way into the emitted
            # artifacts, where _toml_value quotes it as a STRING and the
            # namelist's {...!r} renders it `array(-160.)` -- neither of
            # which is the number the user asked for.  The --polygon
            # sibling below already casts; this path never did.
            wrapped = float(_wrap180(lon))
            warn(f"--point longitude {lon:g} wrapped to {wrapped:g} "
                 "(the [-180, 180] convention this project uses)")
            lon = wrapped
        else:
            # Name the range actually enforced.  One wrap is accepted, so
            # claiming [-180, 180] here described a refusal that does not
            # happen: --point 40,270 is taken, with a warning.
            raise ValueError(
                f"--point longitude {lon:g} must lie within [-180, 180] "
                "(or one wrap within [-360, 360])")
    return lat, lon


@dataclass(frozen=True)
class PolygonFootprint:
    """Validated local GeoJSON rings on their minimum longitude branch.

    GeoJSON positions are ``longitude, latitude``.  ``west`` and ``east``
    are deliberately unwrapped around ``center_lon``; their difference is
    therefore the small circular span even when the footprint crosses the
    antimeridian.
    """

    path: Path
    rings: tuple[tuple[tuple[float, float], ...], ...]
    south: float
    west: float
    north: float
    east: float
    center_lat: float
    center_lon: float

    @property
    def longitude_span(self) -> float:
        return self.east - self.west


_POLYGON_TYPES = ("Polygon", "MultiPolygon")
_POLYGON_SAMPLE_STEP_DEG = 0.25
_POLYGON_MAX_SAMPLES = 2_000_000
_POLYGON_FIT_SLACK_CELLS = 1.0


def _geojson_position(value, label: str,
                      wrapped: list[tuple[float, float]]) \
        -> tuple[float, float]:
    if not isinstance(value, list) or len(value) < 2:
        raise ValueError(
            f"--polygon {label} must be a GeoJSON position [lon, lat]")
    if isinstance(value[0], bool) or isinstance(value[1], bool):
        raise ValueError(
            f"--polygon {label} must contain numeric longitude/latitude")
    try:
        lon, lat = float(value[0]), float(value[1])
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"--polygon {label} must contain numeric longitude/latitude"
        ) from error
    if not (math.isfinite(lon) and math.isfinite(lat)):
        raise ValueError(f"--polygon {label} coordinates must be finite")
    if not -90.0 <= lat <= 90.0:
        raise ValueError(
            f"--polygon {label} latitude {lat:g} must lie within [-90, 90]")
    if not -180.0 <= lon <= 180.0:
        if not -360.0 <= lon <= 360.0:
            raise ValueError(
                f"--polygon {label} longitude {lon:g} must lie within "
                "[-180, 180] (or one wrap within [-360, 360])")
        normalized = float(_wrap180(lon))
        wrapped.append((lon, normalized))
        lon = normalized
    return lon, lat


def _geojson_ring(value, label: str,
                  wrapped: list[tuple[float, float]]) \
        -> tuple[tuple[float, float], ...]:
    if not isinstance(value, list) or len(value) < 4:
        raise ValueError(
            f"--polygon {label} must be a linear ring with at least four "
            "positions")
    ring = tuple(_geojson_position(position, f"{label}[{index}]", wrapped)
                 for index, position in enumerate(value))
    first_lon, first_lat = ring[0]
    last_lon, last_lat = ring[-1]
    if first_lat != last_lat or abs(float(
            _wrap180(last_lon - first_lon))) > 1e-12:
        raise ValueError(
            f"--polygon {label} is not closed (its first and last positions "
            "must match)")
    distinct = {(float(lon % 360.0), lat) for lon, lat in ring[:-1]}
    if len(distinct) < 3:
        raise ValueError(
            f"--polygon {label} needs at least three distinct positions")
    branch = first_lon + np.asarray(_wrap180(
        np.asarray([position[0] for position in ring], dtype=float)
        - first_lon))
    latitudes = np.asarray([position[1] for position in ring], dtype=float)
    twice_area = float(np.sum(
        branch[:-1] * latitudes[1:] - branch[1:] * latitudes[:-1]))
    if abs(twice_area) <= 1e-14:
        raise ValueError(f"--polygon {label} encloses zero area")
    return ring


def _geojson_polygon(value, label: str,
                     wrapped: list[tuple[float, float]]) \
        -> list[tuple[tuple[float, float], ...]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"--polygon {label} has no linear rings")
    return [_geojson_ring(ring, f"{label}[{index}]", wrapped)
            for index, ring in enumerate(value)]


def _geojson_geometry(value, label: str,
                      wrapped: list[tuple[float, float]]) \
        -> list[tuple[tuple[float, float], ...]]:
    if not isinstance(value, dict):
        raise ValueError(f"--polygon {label} must be a GeoJSON geometry")
    kind = value.get("type")
    if kind == "Polygon":
        return _geojson_polygon(value.get("coordinates"),
                                f"{label}.coordinates", wrapped)
    if kind == "MultiPolygon":
        polygons = value.get("coordinates")
        if not isinstance(polygons, list) or not polygons:
            raise ValueError(f"--polygon {label} has no polygons")
        rings = []
        for index, polygon in enumerate(polygons):
            rings.extend(_geojson_polygon(
                polygon, f"{label}.coordinates[{index}]", wrapped))
        return rings
    raise ValueError(
        f"--polygon {label} type {kind!r} is unsupported; accepted geometry "
        f"types are {', '.join(_POLYGON_TYPES)}")


def _geojson_rings(document: object,
                   wrapped: list[tuple[float, float]]) \
        -> tuple[tuple[tuple[float, float], ...], ...]:
    if not isinstance(document, dict):
        raise ValueError("--polygon must contain one GeoJSON object")
    crs = document.get("crs")
    if crs not in (None, {}):
        properties = crs.get("properties", {}) if isinstance(crs, dict) \
            else {}
        name = str(properties.get("name", "")).upper()
        code = properties.get("code")
        longitude_latitude = (
            name in {"EPSG:4326", "OGC:CRS84", "CRS84"}
            or name.endswith(":CRS84")
            or name.endswith(":EPSG::4326")
            or code in {4326, "4326"}
        )
        if not longitude_latitude:
            raise ValueError(
                "--polygon declares a custom coordinate reference system; "
                "GeoJSON longitude/latitude coordinates are required")
    kind = document.get("type")
    if kind in _POLYGON_TYPES:
        rings = _geojson_geometry(document, "geometry", wrapped)
    elif kind == "Feature":
        rings = _geojson_geometry(document.get("geometry"),
                                  "feature.geometry", wrapped)
    elif kind == "FeatureCollection":
        features = document.get("features")
        if not isinstance(features, list) or not features:
            raise ValueError("--polygon FeatureCollection has no features")
        rings = []
        for index, feature in enumerate(features):
            if not isinstance(feature, dict) \
                    or feature.get("type") != "Feature":
                raise ValueError(
                    f"--polygon features[{index}] must be a GeoJSON Feature")
            rings.extend(_geojson_geometry(
                feature.get("geometry"), f"features[{index}].geometry",
                wrapped))
    else:
        raise ValueError(
            f"--polygon top-level type {kind!r} is unsupported; use Polygon, "
            "MultiPolygon, Feature, or FeatureCollection")
    if not rings:
        raise ValueError("--polygon contains no polygon coordinates")
    return tuple(rings)


def _minimum_longitude_arc(longitudes) -> tuple[float, float, float]:
    """Return ``(center, west, east)`` on the minimum circular arc."""

    values = np.unique(np.mod(np.asarray(longitudes, dtype=np.float64),
                              360.0))
    if not values.size:
        raise ValueError("--polygon contains no longitude coordinates")
    extended = np.concatenate((values, values[:1] + 360.0))
    gap_index = int(np.argmax(np.diff(extended)))
    start = float(extended[gap_index + 1])
    end = float(extended[gap_index] + 360.0)
    span = end - start
    if span > 180.0 + 1e-10:
        raise ValueError(
            f"--polygon minimum longitude footprint spans {span:.1f} "
            "degrees; footprints wider than 180 degrees cannot be served "
            "as one source crop")
    center = float(_wrap180((0.5 * (start + end)) % 360.0))
    on_branch = center + np.asarray(
        _wrap180(np.asarray(longitudes, dtype=np.float64) - center))
    west, east = float(on_branch.min()), float(on_branch.max())
    # At an exact 180-degree tie either semicircle is legal.  Pin the
    # numerically reconstructed span to the law above rather than allowing
    # a roundoff-scale false refusal.
    if east - west > 180.0 + 1e-10:
        raise ValueError(
            f"--polygon minimum longitude footprint spans {east - west:.1f} "
            "degrees; footprints wider than 180 degrees cannot be served "
            "as one source crop")
    return center, west, east


def load_polygon_footprint(path: str | Path) -> PolygonFootprint:
    """Read and validate a local Polygon-family GeoJSON document."""

    raw_path = str(path)
    if "://" in raw_path or raw_path.lower().startswith("file:"):
        raise ValueError(
            "--polygon accepts a local GeoJSON file path, not a URL")
    polygon_path = Path(path)
    if not polygon_path.is_file():
        # The sentence names the path THIS PROCESS looked at, absolute,
        # plus the directory a relative one was resolved against.
        #
        # It used to echo the argument as typed.  A caller that passes a
        # relative path and runs the wizard in a different working
        # directory than its own -- which is what every stage-runner
        # subprocess does -- then produced "local GeoJSON file does not
        # exist: danow\case\domain-box.geojson" for a file that was on
        # disk, and sent the user hunting for it.  A refusal that states
        # something the user can check and find false is worse than no
        # refusal: absolute here means the claim is always true, and
        # `cd`-shaped bugs identify themselves on the first read.
        resolved = polygon_path.expanduser()
        try:
            resolved = resolved.resolve()
        except OSError:  # pragma: no cover - unresolvable path shapes
            resolved = polygon_path
        where = ""
        if not polygon_path.is_absolute():
            where = (f" (relative to the working directory "
                     f"{Path.cwd()})")
        kind = ("is not a regular file"
                if polygon_path.exists() else "does not exist")
        raise ValueError(
            f"--polygon local GeoJSON file {kind}: {resolved}{where}")
    try:
        document = json.loads(polygon_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(
            f"--polygon local GeoJSON file could not be read: "
            f"{polygon_path} ({error})") from error
    except UnicodeDecodeError as error:
        raise ValueError(
            f"--polygon {polygon_path} is not UTF-8 text") from error
    except json.JSONDecodeError as error:
        raise ValueError(
            f"--polygon {polygon_path} is not valid JSON: "
            f"line {error.lineno}, column {error.colno}") from error

    wrapped: list[tuple[float, float]] = []
    rings = _geojson_rings(document, wrapped)
    if wrapped:
        example, normalized = wrapped[0]
        warn(f"--polygon wrapped {len(wrapped)} longitude coordinate(s) "
             f"into [-180, 180] (for example {example:g} to "
             f"{normalized:g})")
    positions = [position for ring in rings for position in ring]
    lons = np.asarray([position[0] for position in positions], dtype=float)
    lats = np.asarray([position[1] for position in positions], dtype=float)
    center_lon, west, east = _minimum_longitude_arc(lons)
    south, north = float(lats.min()), float(lats.max())
    return PolygonFootprint(
        path=polygon_path, rings=rings, south=south, west=west,
        north=north, east=east, center_lat=0.5 * (south + north),
        center_lon=center_lon)


def _polygon_samples(footprint: PolygonFootprint, *,
                     max_step_deg: float = _POLYGON_SAMPLE_STEP_DEG) \
        -> tuple[np.ndarray, np.ndarray]:
    """Densify GeoJSON segments on the footprint's longitude branch."""

    if not math.isfinite(max_step_deg) or max_step_deg <= 0.0:
        raise ValueError("polygon sampling step must be finite and positive")
    sample_lons: list[float] = []
    sample_lats: list[float] = []
    for ring in footprint.rings:
        lons = footprint.center_lon + np.asarray(_wrap180(
            np.asarray([point[0] for point in ring], dtype=float)
            - footprint.center_lon))
        lats = np.asarray([point[1] for point in ring], dtype=float)
        for index in range(len(ring) - 1):
            lon0, lon1 = float(lons[index]), float(lons[index + 1])
            lat0, lat1 = float(lats[index]), float(lats[index + 1])
            steps = max(1, int(math.ceil(max(abs(lon1 - lon0),
                                               abs(lat1 - lat0))
                                         / max_step_deg)))
            if len(sample_lons) + steps + 1 > _POLYGON_MAX_SAMPLES:
                raise ValueError(
                    "the polygon and requested grid spacing require more "
                    f"than {_POLYGON_MAX_SAMPLES:,} containment samples; "
                    "choose a coarser --root-dx, fewer refinement levels, "
                    "or a simpler polygon")
            for fraction in np.arange(steps, dtype=float) / steps:
                sample_lons.append(lon0 + fraction * (lon1 - lon0))
                sample_lats.append(lat0 + fraction * (lat1 - lat0))
        sample_lons.append(float(lons[-1]))
        sample_lats.append(float(lats[-1]))
    return (np.asarray(sample_lats, dtype=float),
            np.asarray(sample_lons, dtype=float))


def parse_level_buffers(raw: str | None) -> tuple[float, ...] | None:
    """Parse comma-separated outer-to-inner buffer distances in km."""

    if raw is None:
        return None
    fields = str(raw).split(",")
    if not fields or any(not field.strip() for field in fields):
        raise ValueError(
            "--buffer-km must be one distance or a comma-separated distance "
            "per domain level")
    values = []
    for field in fields:
        try:
            value = float(field)
        except ValueError as error:
            raise ValueError(
                f"--buffer-km entry {field!r} is not a number") from error
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"--buffer-km entry {field!r} must be finite and "
                "nonnegative")
        values.append(value)
    return tuple(values)


def _buffers_for_levels(values: tuple[float, ...] | None,
                        count: int) -> tuple[float, ...]:
    if values is None:
        return (0.0,) * count
    if len(values) == 1:
        return values * count
    if len(values) != count:
        raise ValueError(
            f"--buffer-km supplies {len(values)} distances but the selected "
            f"ladder has {count} domain levels; supply one distance to use "
            "at every level or exactly one outer-to-inner distance per level")
    return values


def _resolve_cycle(raw: str, *, source: str, hours: int,
                   start_hour: int = 0, **selection) -> datetime:
    """Parse ``--cycle``, resolving ``latest`` the way ``fetch`` does.

    v1.0.0 refused ``--cycle latest`` here with a message that said
    ``latest`` was allowed, and since the documented order is
    wizard-then-fetch there was nothing to tell a user which cycle was
    current -- they had to run a throwaway fetch first.  The resolver
    already existed; the wizard now calls it, and says what it picked.

    ``start_hour`` is the forecast lead the run will START at, so
    ``latest`` resolves to a cycle complete through the END of the
    window (lead + length) rather than through its length alone.
    """

    if raw.strip().lower() != "latest":
        return parse_cycle(raw, source)
    # NO PER-SOURCE BRANCH HERE.  This door used to refuse era5 by name
    # ("a reanalysis with weeks of latency") and everything without a
    # fetch front door by another, so `latest` was a capability three
    # models had.  It is now one question asked of the source's declared
    # initialization grid, and the refusal for a source that declares
    # none is that resolver's own -- which names the missing declaration
    # rather than a list this door would have to keep in step.
    from woof.fetch import (DEFAULT_AS_POSTED, _startable_rule_applies,
                             resolve_latest_cycle)
    # As posted (the default [fetch] as_posted the config carries), latest
    # is the newest cycle whose first leads are out, as `woof fetch`
    # resolves it (DESIGN A136 2.2).
    as_posted = DEFAULT_AS_POSTED and _startable_rule_applies(source)
    try:
        cycle = resolve_latest_cycle(source, start_hour + hours,
            **(dict(selection, start_hour=start_hour) if start_hour else selection),
            **({"as_posted": True} if as_posted else {}))
    except (RuntimeError, OSError) as error:
        raise ValueError(
            f"--cycle latest could not be resolved for {source}: {error}"
            " -- the resolver probes the public mirrors, so this needs "
            "network access; pass an explicit YYYY-MM-DDTHH (UTC) cycle "
            "instead") from error
    # "complete" is a claim a PROBE earns.  A source with no object to
    # probe resolves from its declared publication delay, and saying
    # "complete" there would attest to a check nothing ran.
    from woof.fetch import cycle_is_probeable

    standing = ("newest startable" if as_posted
                else "newest complete" if cycle_is_probeable(source)
                else "newest published")
    print(f"woof domain: --cycle latest resolved to "
          f"{cycle:%Y-%m-%dT%H}Z ({standing} {source} cycle "
          f"covering f{start_hour + hours:03d})")
    return cycle


def _even(value: float) -> int:
    return max(2, 2 * round(value / 2.0))


def _dims_for_scale(scale: float, ratios: tuple[int, ...], *,
                    clearance_rows: int = _CLEARANCE_ROWS
                    ) -> list[tuple[int, int]]:
    """Mass dimensions per domain at ``scale`` (root 110 x 88 at 1.0).

    Everything even so centered children register exactly; child mass
    dimensions are span * ratio, satisfying the loader's divisibility and
    clearance rules by construction (still re-validated by the loader).
    """
    dims = [(_even(110.0 * scale), _even(88.0 * scale))]
    for depth, ratio in enumerate(ratios):
        pnx, pny = dims[-1]
        spans = []
        for parent_extent in (pnx, pny):
            span = _even(_child_span_fraction(depth) * parent_extent)
            span = min(span, parent_extent - 2 * clearance_rows)
            if span < 12:
                raise DomainFitError(
                    f"parent extent {parent_extent} cannot host a nest "
                    f"with {clearance_rows}-row clearance at scale "
                    f"{scale:g}")
            spans.append(span)
        dims.append((spans[0] * ratio, spans[1] * ratio))
    return dims


def radt_ladder_minutes(root_radt_minutes: float,
                        domains: int) -> list[float]:
    """``radt`` per emitted domain, outer to inner: the root's, inherited.

    A nest INHERITS its parent's radiation cadence.  Radiative transfer
    varies on cloud timescales, not on grid scales, so nothing about
    halving dx makes a shorter radiation interval more correct -- and
    WRF's own namelist guidance says so outright: set ``radt`` once for
    the coarsest domain and use the same value for every nest.

    Until 2.5.0 this was ``max(1.0, dx_km)`` per nest, which was wrong in
    both directions and expensive in one:

    * Under the 12-minute suites (every RRTMGP/RRTMG profile the wizard
      offers) the 12-3-1-0.5 ladder emitted 12/3/1/1 -- radiation once a
      simulated MINUTE on both sub-km rungs, measured at 79% of a real
      1 km run's wall clock.  The floor also flattened the bottom of the
      ladder: 1 km and 500 m were handed the same 1.0, so the refinement
      it was charging for had already stopped.  At the 250 m LES target
      this program exists for -- 12-3-1-0.5-0.25 -- it taxed three rungs
      out of five.
    * Under the ``radt = 1.0`` suites (the mp8 validation profile and the
      four no-radiation profiles) it ran the other way and emitted 3.0 on
      the 3 km nest: a CHILD calling radiation three times less often
      than the parent feeding its boundaries.

    Inheritance ships default-ON and takes no flag: an opt-in remedy for
    a correctness defect is a workaround, not a fix.  Per-domain ``radt``
    stays overridable in the emitted TOML for anyone who wants a nest to
    depart deliberately.
    """

    return [float(root_radt_minutes)] * int(domains)


def radiation_cadence_advisory(profile: str | None,
                               domains: int) -> list[str]:
    """The spoken half of :func:`radt_ladder_minutes`: one line, or none.

    "Fixed means default" ships a remedy default-on WITH an advisory, and
    the inheritance rule earns its one line only where the AUTO
    derivation actually decides something a reader could mistake: a
    NESTED emission.  Layered like the gray-zone advisories: the default
    screen carries a compact clause on the physics line domain_main
    already prints (the one-screen cap is a measured gate), and this
    full line prints under --explain.  It names the single cadence
    every domain runs, says the nests inherit
    it -- the pre-2.5.0 wizard refined it with dx instead, flooring
    sub-km nests at 1-minute radiation, measured at 79% of a real run's
    wall clock -- and points at the per-domain ``radt`` key in the
    emitted config, which is the ONE user override the wizard honours
    (it takes no radt flag, and an explicit value in the file always
    wins at load).

    Silent for a single domain: nothing inherits, and the physics
    summary already speaks the root's radt.  Radiation-off compositions are silent because their cadence schedules no work.
    """

    if int(domains) < 2:
        return []
    switches = profile_switches(profile)
    from woof.physics_menu import radiation_scheme_ids
    if not any(radiation_scheme_ids(switches)):
        return []
    return [
        f"radt {float(switches['radt']):g} min on all {int(domains)} "
        "domains: nests inherit the root's radiation cadence rather than "
        "refining it with dx (pre-2.5.0 the wizard floored sub-km nests "
        "at 1-minute radiation -- 79% of a measured run's wall clock); "
        "set a domain's radt in the emitted config to depart "
        "deliberately"]


#: Step the hosting-scale scan walks upward by.  Small enough that the
#: layout it returns is within 5% of the smallest one that hosts the
#: ladder, coarse enough that the whole scan is under a hundred
#: iterations of integer arithmetic.
_HOSTING_SCALE_STEP = 1.05


def _min_hosting_scale(ratios: tuple[int, ...], *,
                       clearance_rows: int = _CLEARANCE_ROWS,
                       minimum_axis: int = 1,
                       dimensions_builder=None) -> float:
    """Smallest scale in the bracket whose layout can host ``ratios``.

    ``_MIN_SCALE``'s comment claims it "still hosts the deepest ladder"
    -- true of the deepest PRESET ladder, which is three nests.  A
    custom ``--chain`` can ask for more, and each level spends both a
    span fraction and a fixed :data:`_CLEARANCE_ROWS` boundary margin,
    so a four-nest chain of ratio-2 refinements runs out of interior at
    the 60x48 root ``_MIN_SCALE`` bottoms out at.  The fit loop probed
    exactly that scale first, so those ladders were refused outright
    even though a larger root hosts them comfortably -- the search never
    looked.

    Hosting is monotone in scale (a larger parent has more interior), so
    the first scale that works is the floor of the whole feasible range,
    and returning it lets the existing bisection do the rest.  For every
    preset -- and for any chain of three nests or fewer -- this returns
    ``_MIN_SCALE`` unchanged, so nothing that fitted before moves.

    Refuses, with the depth and the remedy named, when the ladder cannot
    be hosted anywhere in the bracket.
    """
    scale = _MIN_SCALE
    while scale <= _MAX_SCALE:
        try:
            dims = (_dims_for_scale(scale, ratios, clearance_rows=clearance_rows)
                    if dimensions_builder is None else dimensions_builder(scale))
            if min(min(pair) for pair in dims) < minimum_axis:
                raise DomainFitError("template stencil/boundary needs larger axes")
        except DomainFitError:
            scale *= _HOSTING_SCALE_STEP
            continue
        return scale
    raise DomainFitError(
        f"a ladder of {len(ratios)} nests "
        f"({'-'.join(f'{v:g}' for v in _ladder_dx_km(ratios))} km at a "
        "12 km root) cannot be hosted at any layout this wizard will "
        f"consider: every level spends {_CLEARANCE_ROWS} parent rows of "
        "Davies/blend clearance on each side plus its share of the "
        "parent's interior, and by the innermost level there are fewer "
        f"than 12 cells left even at the {_MAX_SCALE:g}x scale ceiling. "
        "Ask for fewer nests, or reach the same spacing in fewer steps "
        "with larger --chain ratios")


def seconds_per_km(ref_lat: float) -> Fraction:
    """The clock convention that applies at REF_LAT, in s per km of dx.

    5 s/km outside the tropics, halved to 2.5 s/km inside them -- see
    :data:`TROPICAL_ROOT_TIME_STEP_S` for the measurement behind it.
    """

    tropical = abs(float(ref_lat)) < MERCATOR_MAX_LAT
    return Fraction(5, 2) if tropical else Fraction(5)


def root_time_step_s(ref_lat: float,
                     root_dx_m: float = ROOT_DX_M) -> Fraction:
    """The exact root clock for a domain centred at REF_LAT.

    Returns a :class:`~fractions.Fraction`: an arbitrary ``--root-dx``
    in the tropics can land on a half second (3 km -> 7.5 s), which the
    WRF rational clock keys represent exactly.
    """

    return seconds_per_km(ref_lat) * Fraction(float(root_dx_m)) / 1000


def nest_history_interval_at(nest_history_interval_s, nest_index: int,
                             nests: int | None = None) -> float:
    """The history interval of the ``nest_index``-th nest (0 is d02).

    ``nest_history_interval_s`` is what ``--nest-history-interval``
    parsed to: ``None`` (the default for every nest), one number (every
    nest), or one number per nest, outer to inner.  A list whose length
    is not the number of nests is refused by name: a 3-domain ladder
    given three intervals would otherwise drop or invent one.
    """
    value = nest_history_interval_s
    if value is None:
        return float(DEFAULT_NEST_HISTORY_INTERVAL_S)
    if isinstance(value, (list, tuple)):
        if nests is not None and len(value) != nests:
            raise ValueError(
                f"--nest-history-interval names {len(value)} intervals but "
                f"the ladder has {nests} nest(s); give one per nest, outer "
                "to inner, or one value for every nest")
        if nest_index >= len(value):
            raise ValueError(
                f"--nest-history-interval names {len(value)} intervals but "
                f"the ladder has a nest number {nest_index + 1}; give one "
                "per nest, outer to inner, or one value for every nest")
        return float(value[nest_index])
    return float(value)


def derived_time_step_s(ref_lat: float, root_dx_m: float, *,
                        run_seconds: float, ratios: tuple[int, ...],
                        history_interval_s: float | None = None,
                        nest_history_interval_s: float | None = None,
                        restart_interval_s: float = 0.0,
                        physics_periods_s: tuple[Fraction, ...] = ()) -> Fraction:
    """An omitted wizard clock compatible with its actual event schedule.

    Keep an already compatible spacing-derived recommendation exactly.
    Otherwise choose the largest compatible exact rational clock no larger
    than that recommendation. Binding the original physics periods keeps
    their decimal-minute representation unchanged, even for a nonterminating
    rational timestep carried by WRF's integer clock keys.
    Explicit authored clocks never enter this authoring helper.
    """
    recommended = root_time_step_s(ref_lat, root_dx_m)
    if recommended <= 0:
        raise ValueError("the derived root time step must be positive")
    root_history = (DEFAULT_ROOT_HISTORY_INTERVAL_S if history_interval_s is None
                    else float(history_interval_s))
    child_histories = [nest_history_interval_at(nest_history_interval_s, i,
                                                len(ratios))
                       for i in range(len(ratios))]
    # Fraction() raises OverflowError on an infinity and ValueError on a
    # NaN, neither naming the interval; the parser refuses both by option
    # name, and a programmatic caller gets a named sentence here.  The
    # nest interval is read only when there is a nest.
    intervals = [("history interval", root_history)]
    intervals.extend(("nest history interval", child) for child in child_histories)
    for name, value in intervals:
        if not (math.isfinite(value) and value > 0):
            raise ValueError(f"the {name} must be a finite number of seconds "
                             f"above zero, not {value:g}")
    # Match the loader's exact event representations; no epsilon or rounding.
    periods = [Fraction(str(run_seconds)), Fraction(root_history)]
    ratio_product = 1
    for ratio, child_history in zip(ratios, child_histories):
        ratio_product *= ratio
        periods.append(Fraction(child_history) * ratio_product)
    if restart_interval_s:
        periods.append(Fraction(float(restart_interval_s)))
    periods.extend(physics_periods_s)
    if any(period <= 0 for period in periods):
        raise ValueError("run and output/checkpoint intervals must be positive")
    if all((period / recommended).denominator == 1 for period in periods):
        return recommended
    denominator = math.lcm(*(period.denominator for period in periods))
    ticks = math.gcd(*(period.numerator * (denominator // period.denominator)
                       for period in periods))
    interval = Fraction(ticks, denominator)
    return interval / max(1, math.ceil(interval / recommended))


def _root_physics_periods(physics: dict, shared: dict) -> tuple[Fraction, ...]:
    """Active periods checked by the experiment loader, in exact seconds."""
    periods = []
    lw = int(shared.get("ra_lw_physics", shared.get("ra_physics", 0)))
    sw = int(shared.get("ra_sw_physics", shared.get("ra_physics", 0)))
    if lw or sw:
        from woof.config import RunConfig
        radt = physics.get("radt", 0.0)
        periods.append(radt if radt > 0 else physics.get("radt_minutes",
            shared.get("radt_minutes", RunConfig.radt_minutes)))
    if int(physics["cu_physics"]) == 1:
        periods.append(physics["cudt_minutes"])
    if any(int(shared.get(name, 0)) for name in
           ("bl_pbl_physics", "sf_sfclay_physics", "sf_surface_physics")):
        periods.append(shared.get("bldt", 0.0))
    return tuple(Fraction(str(value)) * 60 for value in periods if value > 0)


def _derived_clock_note(recommended: Fraction, selected: Fraction,
                        physics_periods_s: tuple[Fraction, ...]) -> str:
    periods = ", ".join(f"{float(value):g}" for value in sorted(set(physics_periods_s)))
    return (f"derived root time step adjusted {float(recommended):g} -> "
            f"{float(selected):g} s to land run/output/checkpoint and physics events "
            "on exact steps; requested spacing and event intervals are unchanged"
            + (f"; active physics periods: {periods} s" if periods else ""))


def _clock_keys(dt: Fraction) -> dict[str, int]:
    """WRF's rational clock keys for an exact root time step."""

    whole = dt.numerator // dt.denominator
    remainder = dt - whole
    keys = {"time_step": int(whole)}
    if remainder:
        keys["time_step_fract_num"] = remainder.numerator
        keys["time_step_fract_den"] = remainder.denominator
    return keys


#: ``woof domain --clock``: how the emitted run steps.  ``adaptive``
#: writes ``use_adaptive_time_step = true`` into ``[shared]`` and nothing
#: else, so every domain keeps WRF's own bounds for its spacing;
#: ``fixed`` writes nothing and the run keeps one step; ``auto`` (the
#: door's default) is adaptive where :func:`clock_decision` finds the
#: grid inside what the adaptive clock and the terrain clock both cover.
CLOCK_CHOICES = ("auto", "adaptive", "fixed")
DEFAULT_CLOCK = "auto"


def _clock_band_misses(time_step: Fraction, root_dx_m: float,
                       ratios: tuple[int, ...]) -> list[str]:
    """Domains whose first step lies outside the adaptive clock's bounds.

    Left at -1, a domain's bounds are WRF's fill-ins for its spacing,
    ``NINT(3*dx km)`` and ``NINT(8*dx km)`` seconds
    (:func:`woof.core.adaptive_clock.wrf_default_clamps`), and its first
    step is the configured one (the adaptive driver's named divergence
    from WRF's ``4*dx``).  A floor above that step is applied every step
    after the first (max first, min second, as WRF clamps), so the run
    would never again take the step this door chose for the grid; a floor
    of 0 s lets the step shrink below one tick of the clock lattice.
    """

    from woof.core.adaptive_clock import wrf_default_clamps

    misses = []
    dt = Fraction(time_step)
    dx = float(root_dx_m)
    for index in range(len(ratios) + 1):
        if index:
            dt /= int(ratios[index - 1])
            dx /= int(ratios[index - 1])
        _, top, floor = wrf_default_clamps(dx, dx)
        if floor <= 0 or not floor <= dt <= top:
            misses.append(
                f"d{index + 1:02d} at {dx:g} m starts at {float(dt):g} s, "
                f"outside the {floor}..{top} s the adaptive clock allows "
                "a grid of that spacing")
    return misses


def clock_decision(choice: str, *, time_step: Fraction, root_dx_m: float,
                   ratios: tuple[int, ...]) -> tuple[bool, str]:
    """``(adaptive, why)`` for a ``--clock`` choice on this grid.

    ``auto`` is adaptive when two tables agree the grid is covered: every
    domain's first step sits inside the adaptive clock's bounds for its
    spacing (:func:`_clock_band_misses`), and every spacing lies within
    the spacings the terrain clock's stability map measured
    (:func:`woof.terrain_clock.measured_map`), which is what caps the
    adaptive step over steep ground at launch.  The tropical clock
    (2.5 s per km) always starts below the 3 s per km floor, so ``auto``
    keeps it fixed.  ``adaptive`` on a grid outside those bounds is
    REFUSED, naming the domain: the controller would clamp the step
    above the one the grid was given for the whole run.
    """

    if choice not in CLOCK_CHOICES:
        raise ValueError(f"--clock must be one of {', '.join(CLOCK_CHOICES)}, "
                         f"got {choice!r}")
    if choice == "fixed":
        return False, "--clock fixed"
    misses = _clock_band_misses(time_step, root_dx_m, ratios)
    if choice == "adaptive":
        if misses:
            raise ValueError(
                "--clock adaptive cannot run this grid: "
                + "; ".join(misses) + ".  The adaptive clock clamps every "
                "step after the first to those bounds, so the run would "
                "step outside the one this grid was given for its whole "
                "length (the tropical clock's 2.5 s per km sits under the "
                "3 s per km floor).  Use --clock fixed.")
        return True, "--clock adaptive"
    if misses:
        return False, "auto kept the fixed step: " + "; ".join(misses)
    from woof.terrain_clock import measured_map

    mapped = sorted({row.dx_m for row in measured_map().rows})
    spacings = [float(root_dx_m)]
    for ratio in ratios:
        spacings.append(spacings[-1] / int(ratio))
    outside = [f"{value:g} m" for value in spacings
               if not mapped[0] * (1 - 1e-9) <= value <= mapped[-1] * (1 + 1e-9)]
    if outside:
        return False, (
            "auto kept the fixed step: " + ", ".join(outside) + " lies "
            f"outside the {mapped[0]:g}..{mapped[-1]:g} m spacings the "
            "terrain clock's stability map measured, so nothing would cap "
            "the adaptive step over steep ground there")
    return True, (
        "auto chose adaptive: every grid starts inside the adaptive "
        "clock's bounds and within the terrain clock's measured spacings "
        f"({mapped[0]:g}..{mapped[-1]:g} m)")


def snap_cadences_to_clock(time_step: Fraction | int | float,
                           physics: dict
                           ) -> tuple[dict, tuple[str, ...]]:
    """Whole-step minute cadences for a derived root clock: (physics, notes).

    The wizard derives BOTH sides of the cadence check: ``--root-dx``
    fixes dt through the s-per-km convention, and the profile fixes
    ``radt``/``cudt_minutes``.  At ``--root-dx 9`` those meet as dt =
    45 s against cudt = 300 s, 300/45 = 20/3 steps, and the loader
    rightly refuses fractional-step cadences -- so the wizard exited 2
    over arithmetic the user supplied no part of (UX finding N14).  The
    author reconciles its own derivation instead: each profile cadence
    that is not a whole number of root steps moves to the NEAREST
    whole-step cadence, and the move is spoken.

    Hand-written configs are untouched -- the loader's refusal in
    :mod:`woof.experiment` still stands wherever a USER pinned an
    incompatible pair, because there both numbers are the user's.

    The snapped value must survive the round trip the loader takes:
    ``Fraction(str(minutes)) * 60 / dt`` has to land on a whole
    number, and not every whole-step cadence has minutes a float can
    carry exactly (17 steps of a tropical 17.5 s clock is 297.5 s =
    4.9583... min).  So the nearest step count whose minutes round-trip
    exactly is taken -- one exists within a few steps for every clock
    the s-per-km convention can produce, for the decimal minute values emitted by the wizard.
    """

    dt = Fraction(time_step)
    adjusted = dict(physics)
    notes: list[str] = []
    for key in ("radt", "cudt_minutes"):
        minutes = adjusted.get(key)
        if minutes is None or float(minutes) <= 0.0:
            continue  # 0 = every step (WRF convention); nothing to snap
        if key == "cudt_minutes" and int(adjusted.get("cu_physics", 0)) != 1:
            continue  # the loader paces cudt only under Kain-Fritsch
        seconds = Fraction(str(minutes)) * 60
        steps = seconds / dt
        if steps.denominator == 1:
            continue
        target = max(1, round(float(steps)))
        chosen = None
        for offset in range(0, 64):
            for count in ((target,) if offset == 0
                          else (target - offset, target + offset)):
                if count < 1:
                    continue
                snapped_minutes = float(count * dt / 60)
                if Fraction(str(snapped_minutes)) * 60 == count * dt:
                    chosen = (count, snapped_minutes)
                    break
            if chosen is not None:
                break
        if chosen is None:  # pragma: no cover - no wizard clock reaches this
            continue  # leave the pair for the loader's refusal
        count, snapped_minutes = chosen
        adjusted[key] = snapped_minutes
        notes.append(
            f"{key} adjusted {float(minutes):g} -> {snapped_minutes:g} "
            f"min: the profile's {float(seconds):g} s is {steps} steps "
            f"of the derived {float(dt):g} s root clock, not a whole "
            f"number, and the loader refuses fractional-step cadences; "
            f"{float(count * dt):g} s = {count} steps is the nearest "
            f"cadence that is")
    return adjusted, tuple(notes)


def _domain_tables(dims: list[tuple[int, int]],
                   ratios: tuple[int, ...],
                   *, time_step: Fraction | int = ROOT_TIME_STEP_S,
                   root_dx_m: float = ROOT_DX_M,
                   profile: str | None = DEFAULT_PHYSICS_PROFILE,
                   cumulus_requested: bool = False,
                   history_interval_s: float | None = None,
                   nest_history_interval_s: float | None = None
                   ) -> list[dict]:
    """[[domain]] table dicts (centered children, certified cadences).

    The ROOT's radiation/cumulus/diffusion cadences come from the shipped
    physics profile, so the emitted d01 satisfies the prepared-forecast
    runner's exact-equality guard at any --root-dx.  Nests keep the
    certified ladder's depth-varying ``diff_6th_factor``, never above
    their parent's (:func:`nest_diff6_factors`), and their pinned
    ``cu_physics = 0``: those two really are grid-scale decisions, and
    the multi-domain runner has no profile whitelist to stop them.

    ROOT ``cu_physics`` IS ONE OF THOSE GRID-SCALE DECISIONS TOO, below
    the convection-permitting bound.  Until 2.5.0 the root took the
    profile's cumulus switch at every spacing, so `--root-dx 3` emitted
    Kain-Fritsch on a grid that resolves its own deep convection, printed
    the sentence naming the heating and rainfall it therefore counts
    twice, and wrote the file anyway -- a defect the product diagnosed
    correctly and then shipped.  The default now follows the diagnosis:
    below :data:`CUMULUS_CONVECTION_PERMITTING_DX_KM` the root's cumulus
    scheme is OFF, and ``cudt_minutes`` goes to the registry's own
    spelling for a domain with no scheme to pace (0.0, as every
    ``cu_physics = 0`` template in :mod:`woof.physics_compat` carries).
    That bound is not a new number: it is the one
    :func:`cumulus_gray_zone_advisory` already judged the emission
    against, so the switch and its advisory read the same declaration.

    ``cumulus_requested`` is the user having NAMED the suite.  Naming
    ``--physics-profile`` asserts the config IS that shipped suite --
    :func:`woof.gfs_direct.front_door_physics_selection` enforces it
    switch for switch on both routes -- so a named suite is emitted
    verbatim at any spacing and keeps the advisory, which is the whole
    point of the advisory being advisory.  Per-domain ``cu_physics``
    stays overridable in the emitted TOML either way.

    ``radt`` is NOT one of them.  It is the root's, inherited by every
    nest (:func:`radt_ladder_minutes`) -- radiation varies on cloud
    timescales, not grid scales.  Refining it with dx is what the
    ``max(1.0, dx_km)`` rule did until 2.5.0, and it cost 79% of a
    measured 1 km run's wall clock for no science.

    EVERY domain, root and nest alike, gets the profile's ``epssm``.
    Until 2026-08-01 the nests were written ``epssm = 0.1`` while the
    root took 0.5 from the profile, and that one line was the whole of
    a reported nested-forecast blow-up: epssm is the vertical-acoustic
    off-centering coefficient, 0.1 is nearly centred, and the nest is
    exactly where the terrain is steepest -- a wizard 3 km -> 750 m
    ladder over the Cascades grew w from -15 to -289 to -976 m/s to
    non-finite in seven acoustic substeps at the steepest cell in the
    child (34.6 degrees; the 3 km parent smooths the same peak to 15.7
    and survives).  Setting the nest to the profile's 0.5 and changing
    nothing else ran both geometries clean for the full two hours.

    The 0.1 was not invented here -- it is WRF's Registry default, and
    it is what an IMPORTED namelist legitimately produces: ``epssm =
    0.5`` in a namelist.input assigns d01 only, so the shipped
    the reference case ladders carry 0.5/0.1/0.1/0.1 and are right to
    (:mod:`woof.namelist_import`, which reads the Registry column).
    The wizard is an author, not an importer, and copying an
    importer's per-domain tail is how the value got here.  Terrain-
    scaled epssm -- more off-centering where slopes are steepest -- is
    a real future refinement; it is deliberately NOT attempted here.
    Per-domain ``epssm`` stays overridable in the emitted TOML.
    """
    root_physics = {key: profile_switches(profile)[key]
                    for key in _PER_DOMAIN_PHYSICS}
    root_physics = root_cumulus(root_physics, float(root_dx_m) / 1000.0,
                                cumulus_requested=cumulus_requested)
    # The author reconciles its own two derivations (dt from --root-dx,
    # cadences from the profile) rather than emitting a file the loader
    # refuses: see snap_cadences_to_clock (UX finding N14).  It runs
    # AFTER the cumulus decision because cudt is paced only under an
    # active Kain-Fritsch, and the decision above may have retired it.
    root_physics, _ = snap_cadences_to_clock(
        Fraction(time_step), root_physics)
    epssm = profile_switches(profile)["epssm"]
    radt = radt_ladder_minutes(root_physics["radt"], len(dims))
    diff6 = nest_diff6_factors(root_physics["diff_6th_factor"], len(dims))
    tables = []
    for index, (nx, ny) in enumerate(dims):
        if index == 0:
            table = {
                "grid_id": 1, "parent_id": 0, "i_parent_start": 1,
                "j_parent_start": 1, "parent_grid_ratio": 1,
                "parent_time_step_ratio": 1, "nx": nx, "ny": ny,
                **_clock_keys(Fraction(time_step)),
                "dx": float(root_dx_m),
                "specified": True, "nested": False,
                "history_interval_s": (
                    DEFAULT_ROOT_HISTORY_INTERVAL_S
                    if history_interval_s is None
                    else float(history_interval_s)),
                **root_physics,
            }
        else:
            ratio = ratios[index - 1]
            pnx, pny = dims[index - 1]
            table = {
                "grid_id": index + 1, "parent_id": index,
                "i_parent_start": (pnx - nx // ratio) // 2 + 1,
                "j_parent_start": (pny - ny // ratio) // 2 + 1,
                "parent_grid_ratio": ratio,
                "parent_time_step_ratio": ratio, "nx": nx, "ny": ny,
                "specified": False, "nested": True,
                "history_interval_s": nest_history_interval_at(
                    nest_history_interval_s, index - 1, len(ratios)),
                "epssm": epssm,
                "radt": radt[index], "cu_physics": 0,
                "diff_6th_factor": diff6[index],
            }
        tables.append(table)
    return tables


def _toml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%dT%H:%M:%S")
    if isinstance(value, (list, tuple)):
        if len(value) > 5:  # long numeric arrays: 5 per line
            lines = ["["]
            for start in range(0, len(value), 5):
                chunk = ", ".join(
                    _toml_value(v) for v in value[start:start + 5])
                lines.append(f"    {chunk},")
            lines.append("]")
            return "\n".join(lines)
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    if isinstance(value, (str, Path)):
        return '"' + str(value).replace("\\", "/") + '"'
    scalar = _builtin_scalar(value)
    if scalar is not None:
        return _toml_value(scalar)
    # Everything used to land in the quoted branch, so a numeric value of
    # any type this function did not enumerate was emitted as a STRING
    # into a typed config file, silently.  That is how a wrapped --point
    # longitude shipped as ref_lon = "-160.0".  An unrenderable value is
    # now a loud failure at emission rather than a quiet mistyping.
    raise TypeError(
        f"cannot render {value!r} ({type(value).__name__}) into TOML: it "
        "is neither a scalar, a string/path, nor a list of those.  "
        "Quoting it would emit a value of the wrong TYPE under the right "
        "key, which reads as valid TOML and is not what was computed.")


def _builtin_scalar(value):
    """Python bool/int/float for a numpy scalar or 0-d array, else None.

    numpy's scalar types are not all builtins -- np.float64 subclasses
    float but np.float32 does not, and a 0-d ndarray subclasses nothing
    -- so array code returning "a number" can hand this module something
    no isinstance branch above matches.
    """
    if isinstance(value, np.generic) or (
            isinstance(value, np.ndarray) and value.ndim == 0):
        item = value.item()
        if isinstance(item, (bool, int, float)):
            return item
    return None


def _render_table(name: str, entries: dict, array_of_tables: bool = False,
                  comment: str | None = None) -> str:
    header = f"[[{name}]]" if array_of_tables else f"[{name}]"
    lines = ([f"# {comment}"] if comment else []) + [header]
    for key, value in entries.items():
        lines.append(f"{key} = {_toml_value(value)}")
    return "\n".join(lines) + "\n"


def _ladder_dx_km(ratios: tuple[int, ...],
                  root_dx_m: float = ROOT_DX_M) -> list[float]:
    chain = [float(root_dx_m) / 1000.0]
    for ratio in ratios:
        chain.append(chain[-1] / ratio)
    return chain


def auto_projection(lat: float) -> str:
    """The auto-selected WPS projection for a point latitude."""
    if abs(lat) < MERCATOR_MAX_LAT:
        return "mercator"
    if abs(lat) <= LAMBERT_MAX_LAT:
        return "lambert"
    return "polar"


def _projection_entries(lat: float, lon: float,
                        choice: str = "auto") -> dict:
    """[projection] table for the point.

    ``auto`` selects by |lat| (:func:`auto_projection`).  Lambert:
    hemisphere-correct secant cone bracketing the point, truelats at
    |lat| +/- 10 deg clamped to [15, 70] and signed with the point's
    hemisphere.  Mercator: true at the point's latitude (stand_lon is
    recorded but does not enter Mercator math, module_llxy semantics).
    Polar stereographic: the point's hemisphere pole, scale true at the
    point's latitude, stand_lon at the point.
    """
    map_proj = auto_projection(lat) if choice == "auto" else choice
    if map_proj == "lambert":
        sign = -1.0 if lat < 0.0 else 1.0
        alat = abs(lat)
        return {
            "map_proj": "lambert", "ref_lat": lat, "ref_lon": lon,
            "truelat1": sign * round(max(15.0, alat - 10.0), 2),
            "truelat2": sign * round(min(70.0, alat + 10.0), 2),
            "stand_lon": lon,
        }
    if map_proj in ("mercator", "polar"):
        return {
            "map_proj": map_proj, "ref_lat": lat, "ref_lon": lon,
            "truelat1": round(lat, 2), "truelat2": round(lat, 2),
            "stand_lon": lon,
        }
    raise ValueError(
        f"--projection {map_proj!r} is not implemented (choices: auto, "
        "lambert, mercator, polar). Regular/rotated latitude-longitude "
        "needs angular dx/dy rather than metre spacing and WRF's "
        "global/pole polar filter; rotated grids also need "
        "pole_lat/pole_lon state and the map_proj == 6 curvature branch.")


#: Bounds on a custom root dx (km).  Wide, because the point of
#: --root-dx is that the presets are not the whole product; narrow
#: enough that a typo (metres for kilometres, say) is caught.
MIN_ROOT_DX_KM, MAX_ROOT_DX_KM = 0.05, 200.0
#: Bounds on one custom nest ratio.  WRF's own guidance is odd ratios of
#: 3 or 5; 2 and 4 are routine here, and beyond 8 the interpolation
#: stencil and the boundary blend stop being defensible in one step.
MIN_CHAIN_RATIO, MAX_CHAIN_RATIO = 2, 8
#: Most nests a custom chain may declare (the presets go to 4 domains).
MAX_CHAIN_DEPTH = 8


def parse_chain(raw: str) -> tuple[int, ...]:
    """``"4,3,3"`` -> ``(4, 3, 3)``; each entry an integer nest ratio."""

    text = str(raw).strip()
    if not text:
        return ()
    ratios = []
    for field in text.split(","):
        field = field.strip()
        try:
            ratio = int(field)
        except ValueError:
            raise ValueError(
                f"--chain entry {field!r} is not an integer; --chain is a "
                "comma-separated list of whole nest ratios, e.g. "
                "--chain 4,3,3") from None
        if ratio < MIN_CHAIN_RATIO:
            raise ValueError(
                f"--chain ratio {ratio} is below {MIN_CHAIN_RATIO}; a "
                "ratio-1 child is not a refinement")
        if ratio > MAX_CHAIN_RATIO:
            warn(f"--chain ratio {ratio} exceeds the blessed maximum of "
                 f"{MAX_CHAIN_RATIO}; continuing with it as written",
                 why="WRF's own guidance is odd ratios of 3 or 5; beyond "
                     f"{MAX_CHAIN_RATIO} the interpolation stencil and "
                     "the boundary blend are undemonstrated in one step "
                     "-- refining in more steps is the proven route.")
        ratios.append(ratio)
    if len(ratios) > MAX_CHAIN_DEPTH:
        warn(f"--chain declares {len(ratios)} nests, more than the "
             f"{MAX_CHAIN_DEPTH} any configuration has demonstrated; "
             "continuing")
    return tuple(ratios)


def parse_custom_ladder(*, root_dx_km, chain, ladder: str):
    """``(root_dx_m, ratios)`` for a custom ladder, or None for a preset.

    ``--root-dx``/``--chain`` are the general form of ``--ladder``: an
    arbitrary root spacing and an arbitrary chain of integer refinement
    ratios.  Everything downstream -- the estimator fit loop, the
    clearance and cadence rules in the real experiment loader, the
    projection math, ``woof check`` -- is the same code the presets go
    through, so a custom ladder is validated exactly as strictly.
    """

    if root_dx_km is None and chain is None:
        return None
    if ladder != "auto":
        raise ValueError(
            "--ladder is a preset chain and cannot be combined with "
            "--root-dx / --chain; drop --ladder to use the custom form")
    root_km = (ROOT_DX_M / 1000.0 if root_dx_km is None
               else float(root_dx_km))
    if not math.isfinite(root_km) or root_km <= 0.0:
        raise ValueError(
            f"--root-dx {root_km:g} km must be a positive spacing")
    if not MIN_ROOT_DX_KM <= root_km <= MAX_ROOT_DX_KM:
        warn(f"--root-dx {root_km:g} km is outside the expected "
             f"[{MIN_ROOT_DX_KM:g}, {MAX_ROOT_DX_KM:g}] km window "
             "(check the unit -- this flag takes kilometres); continuing")
    ratios = parse_chain("" if chain is None else chain)
    return root_km * 1000.0, ratios


#: Grid spacing (km) below which a 1-D PBL parameterization and resolved
#: convection overlap -- the "terra incognita" / gray zone.
GRAY_ZONE_DX_KM = 1.0


def gray_zone_advisory(chain_km, shared: dict, les_below_m: float | None = None,
                       les_closure: str | None = None) -> list[str]:
    """One accurate sentence when a domain lands in the PBL gray zone.

    Advisory, never a refusal: sub-kilometre nests are exactly what this
    product is for, and people will run them.  But a 1-D column PBL
    scheme assumes the whole boundary-layer eddy spectrum is
    subgrid-scale, and below about 1 km the largest eddies are partly
    resolved, so the scheme and the dynamics do the same transport
    twice.  Saying so once, in the file and on stdout, is the accurate
    thing; refusing would be wrong, and silence would be worse.

    THE RECIPE CARRIES ``mix_isotropic = 1``, and that is a correctness
    repair rather than a wording preference.  This sentence is the only
    place the product tells anybody how to configure turbulence below a
    kilometre, so it is where the recommended configuration is actually
    chosen.  It used to name ``km_opt`` and ``bl_pbl_physics`` and stop
    there -- but ``mix_isotropic`` defaults to 0 (WRF's Registry value,
    :class:`woof.config.RunConfig`), and with ``km_opt`` 2 or 3 that is
    the per-axis path where the vertical exchange coefficient is built
    and capped on the LAYER DEPTH and then handed to the horizontal
    diffusion of ``w``.  A reader who followed the old recipe at 250 m
    landed on ``mix_upper_bound*(dz_max/dx)^2 = 0.702`` against a limit
    of 0.25 -- the tier where the operator amplifies a 2-grid-interval
    mode instead of damping it -- and found out only if they later read
    stderr at config load.  Recommending a configuration and separately
    warning about it is not a fix; the recipe now is one that holds.

    SINCE THE AUTO-SWITCH (project ruling, 2026-08-16) the recipe's key is also
    the running default: a config that leaves ``mix_isotropic`` unset
    and violates the criterion runs isotropic anyway
    (``woof.experiment.resolve_auto_mix_isotropic``, announced at load
    and in ``woof check``).  The recipe keeps NAMING the key so the
    file a reader authors says what it runs;
    ``woof.config.warn_anisotropic_w_mixing`` still only advises when a
    config WRITES ``mix_isotropic = 0`` -- an explicit setting is kept,
    and refusing it would make the frozen crash records unloadable.
    """

    from woof.config import EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT

    if not shared.get("bl_pbl_physics"):
        return []
    # The domains the door writes the LES closure on run PBL off, so they
    # are not in the gray zone and are not counted as running the 1-D
    # scheme.  The advisory used to be computed before the closure was
    # written and told a 3 -> 1 -> 0.333 -> 0.111 km tree that BOTH
    # sub-kilometre nests ran bl_pbl_physics = 5 while d04 was PBL off
    # with km_opt 3.
    les = set(les_domain_indices(chain_km, les_below_m))
    les_line = []
    if les:
        name = DEFAULT_LES_CLOSURE if les_closure is None else str(les_closure)
        listed = ", ".join(f"d{i + 1:02d} {float(chain_km[i]) * 1000:.0f} m"
                           for i in sorted(les))
        les_line = [
            f"LES: {len(les)} domain(s) at or below {float(les_below_m):g} m "
            f"({listed}) run PBL off with the {LES_CLOSURE_WORDS[name]} "
            "closure, mix_isotropic = 1 and surface fluxes from the surface "
            "layer, so their boundary-layer eddies are resolved, not "
            "parameterized"
            + (f"; the coarser domains keep bl_pbl_physics = "
               f"{shared['bl_pbl_physics']}." if len(les) < len(chain_km)
               else ".")]
    below = [dx for i, dx in enumerate(chain_km)
             if dx < GRAY_ZONE_DX_KM and i not in les]
    if not below:
        return les_line
    finest = min(below)
    listed = ", ".join(f"d{i + 1:02d} {float(dx) * 1000:.0f} m"
                       for i, dx in enumerate(chain_km)
                       if dx < GRAY_ZONE_DX_KM and i not in les)
    return [
        f"GRAY ZONE: {len(below)} domain(s) refine below "
        f"{GRAY_ZONE_DX_KM:g} km (finest {finest * 1000:.0f} m"
        + (f"; {listed}" if les else "") + ") with the "
        f"1-D PBL scheme bl_pbl_physics = {shared['bl_pbl_physics']} "
        "active, so boundary-layer eddies are partly resolved by the "
        "dynamics and simultaneously parameterized as if they were not. "
        "The proper tool at these scales is a 3-D turbulence closure with "
        "the PBL off: set km_opt = 3 (3-D Smagorinsky) or km_opt = 2 "
        "(1.5-order prognostic TKE) with bl_pbl_physics = 0 AND "
        "mix_isotropic = 1 on the "
        "domain(s) below the gray zone -- both are per-domain, so a PBL "
        "parent can carry a PBL-off child (see docs/public/LES.md) -- or "
        "the SASE closure (bl_pbl_physics = 900), which is implemented "
        "and selectable but EXPERIMENTAL and not WRF-verified. "
        "mix_isotropic = 1 is part of that recipe and not an optional "
        "extra: km_opt 2 and 3 with mix_isotropic = 0 (WRF's default; "
        "left unset, WOOF auto-selects 1 where the criterion below "
        "fails, and says so) build "
        "the vertical exchange coefficient on the LAYER DEPTH and then "
        "hand it to the horizontal diffusion of w, and the reachable "
        "mix_upper_bound*(dz_max/dx)^2 rises as the grid narrows while "
        "the layers do not -- it reads 0.702 on this project's own 250 m "
        "LES child against a limit of "
        f"{EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT}, which is the tier where "
        "the horizontal mixing of w amplifies a 2-grid-interval mode "
        "instead of damping it. Choose it at t = 0: mix_isotropic is "
        "inside the RunConfig restart fingerprint, so it cannot be "
        "changed part-way through a campaign you intend to resume. Until "
        "you do, treat sub-kilometre PBL structure as indicative rather "
        "than quantitative.",
    ] + les_line


#: Grid spacing (km) below which operational convection-permitting
#: practice turns the cumulus parameterization OFF: the dynamics resolve
#: deep convection at these spacings, and an active scheme convects the
#: same air a second time.
CUMULUS_CONVECTION_PERMITTING_DX_KM = 4.0

#: Upper edge (km) of the convective gray zone.  Between
#: :data:`CUMULUS_CONVECTION_PERMITTING_DX_KM` and this spacing deep
#: convection is neither fully subgrid (the closure assumption every
#: cumulus scheme makes) nor well resolved (the convection-permitting
#: assumption) -- the genuine gray zone, where running the scheme is
#: common operational practice and still worth one accurate sentence.
CUMULUS_GRAY_ZONE_TOP_DX_KM = 10.0


def convection_permitting(dx_km: float) -> bool:
    """True where the dynamics resolve deep convection themselves.

    ONE predicate off :data:`CUMULUS_CONVECTION_PERMITTING_DX_KM`, read
    by both the switch the wizard EMITS (:func:`_domain_tables`) and the
    sentence that judges it (:func:`cumulus_gray_zone_advisory`), so the
    emission and its own advisory can never disagree about where the
    bound is -- which is how the wizard came to print "counted twice"
    about a file it had just written.
    """

    return float(dx_km) < CUMULUS_CONVECTION_PERMITTING_DX_KM


def root_cumulus(switches, root_dx_km: float, *,
                 cumulus_requested: bool) -> dict:
    """``switches`` with the root's cumulus retired where the grid resolves it.

    The switch half of the rule :func:`_domain_tables` emits and
    :func:`cumulus_retired_note` reports: an active scheme on a root
    below the convection-permitting bound goes to ``cu_physics = 0``
    (``cudt_minutes = 0.0``) unless the user asked for cumulus.  Every
    caller that describes what a root will run reads this function, so
    a description cannot disagree with the emission.
    """

    switches = dict(switches)
    if (int(switches.get("cu_physics", 0) or 0)
            and not cumulus_requested
            and convection_permitting(root_dx_km)):
        switches["cu_physics"] = 0
        switches["cudt_minutes"] = 0.0
    return switches


def cumulus_requested_by(args) -> bool:
    """Whether this invocation asked for the root's cumulus.

    ``--cumulus suite`` or ``--cumulus grid`` when given; otherwise
    naming ``--physics-profile`` is the request.
    """

    stated = getattr(args, "cumulus", None)
    if stated is not None:
        return stated == "suite"
    return getattr(args, "physics_profile", None) is not None


def cumulus_retired_note(profile: str | None, root_dx_km: float, *,
                         cumulus_requested: bool) -> list[str]:
    """One sentence when the wizard turned the suite's cumulus OFF.

    The emission changed a switch the named suite carries, so it says
    so, in the file and on the screen, with the bound and the way back.
    Silence here would be the mirror of the defect this replaced: a file
    that does not run what its own PHYSICS line claims.
    """

    scheme = int(profile_switches(profile).get("cu_physics", 0))
    if (not scheme or cumulus_requested
            or not convection_permitting(root_dx_km)):
        return []
    return [
        f"CUMULUS OFF AT {float(root_dx_km):g} KM: the derived suite "
        f"carries cu_physics = {scheme} and this root sits below the "
        f"{CUMULUS_CONVECTION_PERMITTING_DX_KM:g} km "
        "convection-permitting bound, so the emitted root runs "
        "cu_physics = 0 (cudt_minutes = 0.0) instead of convecting the "
        "same air twice -- the dynamics resolve deep convection at this "
        "spacing.  Every other switch is the suite's; name the suite "
        "with --physics-profile to emit it verbatim, cumulus included, "
        "or set cu_physics yourself in the emitted [[domain]] table."]


def cumulus_retired_headline(profile: str | None, root_dx_km: float, *,
                             cumulus_requested: bool) -> list[str]:
    """The retirement's first clause -- same contract as
    :func:`cumulus_gray_zone_headline`: derived from the same call, so a
    headline that could disagree with the sentence cannot exist."""

    return [line.split(", so ", 1)[0] + "."
            for line in cumulus_retired_note(
                profile, root_dx_km, cumulus_requested=cumulus_requested)]


def cumulus_gray_zone_advisory(chain_km, cu_physics_by_domain
                               ) -> list[str]:
    """Accurate sentences when an active cumulus scheme meets fine grids.

    Advisory, never a refusal -- the same contract, channel and tone as
    :func:`gray_zone_advisory`: the full sentence lives in the emitted
    file's header comment, stdout carries the finding (headline) by
    default and the whole sentence under ``--explain``.  At most one
    sentence per finding, strong one first:

    * below ~4 km the dynamics resolve deep convection, so an active
      scheme double-counts it; operational convection-permitting
      practice runs ``cu_physics = 0`` there.
    * in the 4-10 km band convection is neither fully subgrid nor well
      resolved -- the genuine gray zone.  Running the scheme there is
      common operational practice, so the note is softer.

    ``cu_physics_by_domain`` is per-domain (outer to inner, same order
    as ``chain_km``) because cumulus is a per-domain switch here, not a
    ``[shared]`` one -- the wizard activates it on the root only.

    Scale awareness is honoured per scheme, as this docstring always
    promised it would be.  Grell-Freitas (``cu_physics = 3``) carries
    its own ``sig = (1-frh)**2`` taper -- the parameterized contribution
    withdraws as dx approaches cloud-resolving spacing, which is the
    scheme's whole design point -- so a GF domain in the 4-10 km gray
    zone earns no advisory, and below 4 km it earns the SOFT sentence
    (the taper is the scheme's own answer there, but no ArWen-vs-WRF
    trajectory receipt backs it yet, so the wording says measured
    restraint rather than silence).  Classic KF (``cu_physics = 1``) is
    not scale-aware and keeps both sentences.
    """

    pairs = [(float(dx), int(cu)) for dx, cu
             in zip(chain_km, cu_physics_by_domain, strict=True)]
    active = [(dx, cu) for dx, cu in pairs if cu]
    lines: list[str] = []
    scale_aware_below = [(dx, cu) for dx, cu in active
                         if cu == 3 and convection_permitting(dx)]
    if scale_aware_below:
        finest = min(dx for dx, _ in scale_aware_below)
        lines.append(
            f"CUMULUS: {len(scale_aware_below)} domain(s) run "
            "Grell-Freitas (cu_physics = 3) at convection-permitting "
            f"spacing below {CUMULUS_CONVECTION_PERMITTING_DX_KM:g} km "
            f"(finest {finest:g} km); the scheme's own sig = (1-frh)^2 "
            "taper withdraws the parameterized contribution as the grid "
            "resolves convection, so this is the scheme answering the "
            "gray zone rather than double-counting it -- but no "
            "WOOF-versus-WRF trajectory receipt backs GF yet "
            "(implemented-unverified), so verify convective placement "
            "against observations before trusting it there.")
    below = [(dx, cu) for dx, cu in active
             if cu != 3 and convection_permitting(dx)]
    if below:
        finest = min(dx for dx, _ in below)
        switch = "/".join(str(cu) for cu
                          in sorted({cu for _, cu in below}))
        lines.append(
            f"CUMULUS: {len(below)} domain(s) run at "
            "convection-permitting spacing below "
            f"{CUMULUS_CONVECTION_PERMITTING_DX_KM:g} km (finest "
            f"{finest:g} km) with the cumulus parameterization "
            f"cu_physics = {switch} active, so deep convection is "
            "resolved by the dynamics and parameterized by the scheme "
            "at the same time -- its heating and rainfall counted "
            "twice; operational convection-permitting practice runs "
            "cu_physics = 0 below about "
            f"{CUMULUS_CONVECTION_PERMITTING_DX_KM:g} km and lets the "
            "resolved dynamics convect (per-domain override in the "
            "emitted [[domain]] tables).")
    band = [(dx, cu) for dx, cu in active
            if cu != 3
            and not convection_permitting(dx)
            and dx <= CUMULUS_GRAY_ZONE_TOP_DX_KM]
    if band:
        finest = min(dx for dx, _ in band)
        switch = "/".join(str(cu) for cu
                          in sorted({cu for _, cu in band}))
        lines.append(
            f"CUMULUS GRAY ZONE: {len(band)} domain(s) sit in the "
            f"{CUMULUS_CONVECTION_PERMITTING_DX_KM:g}-"
            f"{CUMULUS_GRAY_ZONE_TOP_DX_KM:g} km convective gray zone "
            f"(finest {finest:g} km) with cu_physics = {switch} "
            "active, so deep convection is neither fully subgrid nor "
            "well resolved and the scheme's closure assumptions only "
            "partly hold; this pairing is common operational practice "
            "-- keep it if it is deliberate, and read convective "
            "placement and intensity on those domains as indicative "
            "rather than quantitative.")
    return lines


def cumulus_by_domain(dims, ratios, *,
                      profile: str | None,
                      root_dx_m: float = ROOT_DX_M,
                      cumulus_requested: bool = False) -> list[int]:
    """``cu_physics`` per emitted domain, outer to inner.

    Read from the same ``[[domain]]`` tables the emitted file carries
    (:func:`_domain_tables`: root from the profile unless the spacing
    resolves its own convection, nests pinned to 0), never re-derived,
    so the advisory cannot disagree with the emission.

    ``root_dx_m`` used to be left at its default here, on the reasoning
    that it "cannot change which cumulus switch a domain carries".  That
    reasoning ended when the root's switch became convection-permitting
    aware: pass the emission's own spacing, or this reports the 12 km
    answer for a 3 km file.  ``time_step`` genuinely cannot change a
    cumulus switch and is still left alone.
    """

    return [int(table["cu_physics"])
            for table in _domain_tables(
                dims, ratios, profile=profile, root_dx_m=root_dx_m,
                cumulus_requested=cumulus_requested)]


def _footprint_contains_pole(projection: dict, nx: int, ny: int,
                             root_dx_m: float = ROOT_DX_M,
                             margin_cells: float = _POLE_CLEARANCE_CELLS
                             ) -> bool:
    """Measure the projection pole in the root plane.

    The same geometry bounds Lambert fitting and identifies a polar cap
    whose forcing box must include every meridian and the pole latitude.
    """

    return footprint_contains_pole(projection, nx, ny, root_dx_m,
                                   margin_cells)


def point_request_bound(projection: dict, nx: int, ny: int,
                        root_dx_m: float = ROOT_DX_M,
                        max_extent_km: float = POINT_FIT_MAX_EXTENT_KM
                        ) -> tuple[str, str] | None:
    """Which bound says this layout is larger than a POINT may ask for,
    and why -- or ``None`` when neither does.

    The card is not the only thing that decides how big a domain grown
    from a single point should be, and until 2.7.3 it was the only thing
    that did.  Two bounds, both properties of the REQUEST:

    * the projection's own usable envelope.  A Lambert or polar root
      grown far enough poleward swallows the projection pole, where
      lat-lon source interpolation and static-tile windowing do not work
      -- the limit :func:`_pole_clearance_refusal` states.  It is a
      SIZING bound here, not a refusal: the point is a legal request and
      shrinking the domain honours it, so the fit shrinks.  The refusal
      stays where a smaller domain cannot help -- a drawn area that
      itself reaches the pole, and a point so close to one that even the
      minimum layout contains it.

    * a maximum extent, ``max_extent_km`` (``--point-extent-km``,
      default :data:`POINT_FIT_MAX_EXTENT_KM`), checked on the longer
      root axis: the root keeps the 5:4 shape of :func:`_dims_for_scale`,
      so the shorter (y) axis comes out at 0.8 of it.  A point carries no
      extent at all, so "as much ground as this card can hold" is an
      answer to a question nobody asked; with streaming on it is not
      even bounded by the card.

    * one trip around the globe in longitude
      (:func:`woof.static.projection.footprint_longitude_span`).  The
      pole bound never fires on Mercator, and on the Lambert cones this
      wizard opens the band is crossed before the pole margin is reached.
      While the extent was fixed at 6,000 km neither could happen; once
      the extent became an argument, ``--point-extent-km 60000`` at the
      equator sized a 5000 x 4000 Mercator root running 540 degrees of
      longitude and the door printed PASS, and at 26 N a 2646 x 2116
      Lambert root running 407.  A root past this bound holds
      the same ground twice and integrates the two copies apart, so the
      fit shrinks to stay inside it.  A drawn area never reaches it: a
      polygon cannot describe more than one turn.

    Monotone in scale, which is what :func:`fit_ladder`'s bisection
    requires: growing a centered root moves its poleward edge further
    poleward and every axis further out, so a layout rejected here stays
    rejected at every larger scale.

    Module level rather than a closure inside the fit so the bounds can
    be asked about and tested on their own.  Only the fit calls it: what
    stopped the search is carried forward by ``fit_ladder``'s
    ``stop_out``, not re-derived, because the two consumers of that
    answer -- the plain fact the plan summary states and the flag the
    oversized-footprint advisory is allowed to name -- must agree with
    the search and with each other exactly, and a re-derivation off the
    emitted root misattributes a fit that stopped one discretisation
    step under a cap.
    """

    dx_km = float(root_dx_m) / 1000.0
    extent_km = max(nx, ny) * dx_km
    if extent_km > float(max_extent_km):
        return (POINT_FIT_EXTENT_SCOPE,
                f"a {nx} x {ny} root at {dx_km:g} km spans "
                f"{extent_km:.0f} km, past the "
                f"{float(max_extent_km):.0f} km a point request "
                "is sized to (raise --point-extent-km, or draw the area "
                "to ask for more ground)")
    if projection["map_proj"] != "polar" and _footprint_contains_pole(
            projection, nx, ny, root_dx_m, _FIT_POLE_CLEARANCE_CELLS):
        pole = "north" if projection["truelat1"] >= 0.0 else "south"
        return (POINT_FIT_PROJECTION_SCOPE,
                f"a {nx} x {ny} root at {dx_km:g} km reaches the "
                f"{pole} pole, where Lambert longitude is singular; "
                "use --projection polar for a pole-enclosing domain")
    # Asked after the pole: a footprint around the pole also winds a
    # full turn, and the pole is the accurate name for that one.
    span_deg = footprint_longitude_span(projection, nx, ny, root_dx_m)
    if projection["map_proj"] != "polar" and span_deg >= 360.0:
        return (POINT_FIT_BAND_SCOPE,
                f"a {nx} x {ny} root at {dx_km:g} km runs "
                f"{span_deg:.0f} degrees of longitude, more than once "
                "around the globe, so its grid would hold the same "
                "ground twice and integrate the two copies apart")
    return None


def _shorter_axis_note(nx: int, ny: int, dx_km: float) -> str:
    """The extent of the shorter root axis, for a non-square root.

    The extent a point note quotes is the LONGER axis (the one
    ``--point-extent-km`` bounds); a 5:4 root's other axis is 0.8 of it,
    and a note quoting one number for a non-square root read as if both
    axes had it."""

    if nx == ny:
        return ""
    return (f" on the longer axis, {min(nx, ny) * dx_km:.0f} km on the "
            "shorter")


def point_fit_cap_note(scope: str, dims, root_dx_m: float = ROOT_DX_M,
                       point_extent_km: float | None = None) -> str:
    """One plain sentence of plan-summary fact for a capped point fit.

    ``point_extent_km`` is the extent the request asked for; the floor
    sentence quotes it, because there the root is LARGER than that and
    the reader has to see both numbers to see why.

    Not a warning, and deliberately not on stderr.  The cap is the
    DEFAULT sizing of a request that carries no extent, so it fires on
    the ordinary mid-latitude point on any card from about 16 GiB up --
    every such run, on the door the desktop drives.  A stderr
    ``warning:`` there would say a normal request is abnormal, and it
    said it loudly enough to turn the release suite red
    (``tests/test_go_chain.py`` holds the door to an empty stderr on its
    default emission, which is the same promise stated as a test).  The
    reader still has to be able to see why the grid stopped where it did
    -- an invisible saturation is the defect
    ``tests/test_domain_wizard_budget_monotonic.py`` exists for -- so
    the fact is stated once, on stdout, beside the sizing line that
    prints the envelope it is under.

    A stderr warning is left for a request that is itself unusual: a
    drawn area reaching the pole still refuses, and ``|lat| 90`` and a
    centre no layout clears still refuse.
    """

    nx, ny = dims[0]
    dx_km = float(root_dx_m) / 1000.0
    extent_km = max(nx, ny) * dx_km
    axes = _shorter_axis_note(nx, ny, dx_km)
    if scope == POINT_FIT_EXTENT_SCOPE:
        return (f"point request: extent capped at {extent_km:.0f} km "
                f"({nx}x{ny} at {dx_km:g} km){axes}, memory allows more; raise "
                f"--point-extent-km or draw the ground you want with "
                f"--polygon for a larger domain")
    if scope == POINT_FIT_PROJECTION_SCOPE:
        return (f"point request: extent capped at {extent_km:.0f} km "
                f"({nx}x{ny} at {dx_km:g} km){axes} to stay clear of the "
                f"projection pole, memory allows more; move --point "
                f"equatorward for a larger domain")
    if scope == POINT_FIT_BAND_SCOPE:
        return (f"point request: extent capped at {extent_km:.0f} km "
                f"({nx}x{ny} at {dx_km:g} km){axes} so the root runs less than "
                f"once around the globe, memory allows more; this is the "
                f"widest root a point can take")
    if scope == POINT_FIT_FLOOR_SCOPE:
        asked = ("the requested extent" if point_extent_km is None
                 else f"--point-extent-km {float(point_extent_km):g}")
        return (f"point request: root extent {extent_km:.0f} km "
                f"({nx}x{ny} at {dx_km:g} km){axes} is the smallest root this "
                f"ladder hosts, larger than {asked}; a finer root "
                f"spacing gives a smaller root")
    raise ValueError(f"not a point-request bound: {scope!r}")


def point_extent_note(dims, root_dx_m: float = ROOT_DX_M,
                      point_extent_km: float = POINT_FIT_MAX_EXTENT_KM) -> str:
    """The plan-summary fact for a point fit that no request bound stopped:
    the extent it used and the ``--point-extent-km`` it stayed inside."""

    nx, ny = dims[0]
    dx_km = float(root_dx_m) / 1000.0
    return (f"point request: extent {max(nx, ny) * dx_km:.0f} km "
            f"({nx}x{ny} at {dx_km:g} km)"
            f"{_shorter_axis_note(nx, ny, dx_km)}, inside --point-extent-km "
            f"{float(point_extent_km):.0f}")


def _pole_clearance_refusal(projection: dict, nx: int, ny: int,
                            root_dx_m: float = ROOT_DX_M,
                            target_option: str = "--point") -> None:
    """Keep Lambert roots clear of their singular projection pole.

    A polar stereographic root is regular through its hemisphere's pole
    and uses periodic source and static donors. Mercator cannot reach it.
    """
    if projection["map_proj"] == "polar":
        return
    if _footprint_contains_pole(projection, nx, ny, root_dx_m):
        pole_lat = 90.0 if projection["truelat1"] >= 0.0 else -90.0
        drawn = target_option == "--polygon"
        remedy = ("the domain has to contain what you drew, so no card "
                  "and no ladder clears the pole: the drawing is what "
                  "moves" if drawn else
                  "this request was already sized to the smallest "
                  "layout its ladder has, so no card and no shallower "
                  "ladder clears the pole: the centre is what moves")
        raise ValueError(
            f"the fitted root domain ({nx} x {ny} mass points at "
            f"{float(root_dx_m) / 1000:g} km) contains or touches the "
            f"{'north' if pole_lat > 0 else 'south'} pole on Lambert; "
            "its longitude is singular there. Use --projection polar "
            f"or move {target_option} away from the pole; "
            f"{remedy}")


def _margined_longitude_span(lon_c, center: float,
                             margin_deg: float) -> tuple["np.ndarray", float]:
    """Root longitudes on the branch nearest ``center``, and their span.

    ONE expression of the arithmetic, because two readers depend on it
    agreeing with itself: :func:`_fetch_area`, which refuses to emit a
    box wider than one source crop, and :func:`fetch_crop_refusal`,
    which is what stops the fit loop from choosing that layout in the
    first place.  A sizing bound computed one way and an emission gate
    computed another is how a wizard sizes a domain for twelve seconds
    and then refuses the file it just sized.
    """

    lon_u = center + np.asarray(
        _wrap180(np.asarray(lon_c, dtype=float) - center))
    span = ((float(lon_u.max()) + margin_deg)
            - (float(lon_u.min()) - margin_deg))
    return lon_u, span


def fetch_crop_refusal(projection: dict, nx: int, ny: int, *, source: str,
                       root_dx_m: float = ROOT_DX_M,
                       target_option: str = "--point") -> str | None:
    """Why SOURCE cannot force this root as ONE crop, or ``None``.

    The same bound :func:`_fetch_area` enforces at emission, asked early
    enough that :func:`fit_ladder` can shrink against it -- a sizing
    constraint that is not the card, exactly like
    :func:`source_coverage_refusal`.

    The breakage it names is specific and was measured, not theorized: a
    forcing box whose MARGINED longitude span exceeds 180 degrees is
    read back by :func:`woof.fetch.parse_area` as the complementary
    antimeridian-crossing box -- the wrong crop, silently -- so
    ``_fetch_area`` refuses to write one.  Until this bound joined the
    fit, the fit was free to choose a layout the emission would then
    refuse: on Linux, where the peak envelope carries no WDDM floor and
    the same card therefore buys a much larger grid, ``woof domain
    --point=35.3,-97.5 --source gfs --ladder 12 --vram-gib 32`` sized a
    181.2-degree footprint and died with rc 2 and no layout to fall back
    to.  The identical command on Windows emitted a 102 x 67 degree box.
    Shrinking is the answer the refusal's own remedy asks for ("shrink
    the configuration"), and the wizard is the thing that knows by how
    much.
    """

    margin_deg = _fetch_margin_deg(source)
    _, lon_c = _root_grid(projection, nx, ny, root_dx_m).latlon_c()
    _, span = _margined_longitude_span(
        lon_c, float(projection["ref_lon"]), margin_deg)
    if span <= 180.0 or str(source).strip().lower() == "gfs":
        # GFS already supports an explicit -180..180 longitude band. The
        # emitted hint widens forcing coverage instead of shrinking the grid
        # or letting parse_area select the complementary narrow box.
        return None
    return (
        f"the {nx}x{ny} root's forcing box spans {span:.1f} degrees of "
        f"longitude once {source}'s {margin_deg:g}-degree fetch margin is "
        "added, and boxes wider than 180 degrees cannot be served as a "
        f"single source crop.  Move {target_option} equatorward, or size a "
        "narrower domain (--vram-gib N, or a finer --root-dx)")


def _fetch_area(projection: dict, nx: int, ny: int,
                margin_deg: float = _FETCH_MARGIN_DEG,
                *, notes: list[str] | None = None,
                root_dx_m: float = ROOT_DX_M,
                target_option: str = "--point",
                coverage: tuple[float, float, float, float] | None = None,
                coverage_label: str = "the source grid",
                coverage_notes: list[str] | None = None,
                allow_full_longitude: bool = False,
                ) -> tuple[float, float, float, float]:
    """Forcing bbox (S, W, N, E) = root corners + margin, worldwide.

    Corner longitudes are unwrapped onto the branch nearest the
    reference longitude, so a footprint straddling the antimeridian
    yields a continuous span; the emitted box then wraps back into the
    signed convention, producing W > E for a crossing box (the
    ``woof fetch`` contract; NOMADS and CDS both consume it).  The
    latitude edges clamp to [-90, 90].  A footprint wider than 180
    degrees of longitude is refused unless the source explicitly supports
    a full-longitude band. That opt-in widens only the forcing coverage to
    -180..180. The refusal is checked on the MARGINED span
    -- the margin is part of the emitted box, and a box whose margined
    width exceeds 180 degrees would be read back by
    :func:`woof.fetch.parse_area` as the complementary
    antimeridian-crossing box (the wrong crop, silently).

    ``coverage`` is a source's own ``(S, W, N, E)`` lat/lon envelope --
    :func:`woof.fetch.source_coverage_envelope`'s data, never a source
    name -- and the box is clamped INTO it, quantized inward to the
    hint's printed precision so the formatted string cannot round back
    out of coverage.  Clamp rather than shrink the domain, and clamp
    rather than refuse: by emission time the DOMAIN is already bounded
    by source coverage (the fit loop keeps every HRRR candidate's
    interpolation window inside the native grid), so what overruns here
    is only the margined lat/lon bbox -- a projection artifact of a
    Lambert footprint's curved edges plus the fetch margin, not data
    the preparation needs; for a coverage-boxed source ``--area`` is a
    coverage check, not a crop.  The field exhibit: a 1234 x 986 root
    at 3 km (39, -98) fits the HRRR grid with rows to spare, yet its
    margined bbox named 54.39 N -- north of anything HRRR carries --
    and the wizard's own printed fetch refused to run.  Every clamp is
    disclosed through ``coverage_notes``.
    """
    lat_c, lon_c = _root_grid(projection, nx, ny, root_dx_m).latlon_c()
    center = float(projection["ref_lon"])
    lon_u, span = _margined_longitude_span(lon_c, center, margin_deg)
    contains_pole = (projection["map_proj"] == "polar"
                     and _footprint_contains_pole(
                         projection, nx, ny, root_dx_m, 0.0))
    full_longitude = (contains_pole or span > 180.0) and allow_full_longitude
    if span > 180.0 and not full_longitude:
        raise ValueError(
            f"the root domain's forcing footprint spans {span:.1f} "
            "degrees of longitude; boxes wider than 180 degrees cannot "
            "be served as a single source crop -- shrink the "
            f"configuration or move {target_option} equatorward")
    # A polar cap includes the pole row. Lambert keeps its established
    # clearance from the singular longitude at the cone tip.
    pole_clear = (90.0 if projection["map_proj"] == "polar"
                  else max_fetch_abs_lat(root_dx_m))
    lat_s = max(-pole_clear, float(lat_c.min()) - margin_deg)
    lat_n = min(pole_clear, float(lat_c.max()) + margin_deg)
    if notes is not None:
        for edge, raw, clamped in (
                ("south", float(lat_c.min()) - margin_deg, lat_s),
                ("north", float(lat_c.max()) + margin_deg, lat_n)):
            if raw != clamped:
                notes.append(
                    f"the suggested forcing box's {edge} edge was clamped "
                    f"from {raw:.2f} to {clamped:.2f} to stay "
                    f"the {pole_clear:g}-degree latitude limit")
    if contains_pole:
        if projection["truelat1"] >= 0.0:
            lat_n = 90.0
        else:
            lat_s = -90.0
    lon_w = float(_wrap180(float(lon_u.min()) - margin_deg))
    lon_e = float(_wrap180(float(lon_u.max()) + margin_deg))
    if lon_e == -180.0:
        lon_e = 180.0
    if full_longitude:
        lon_w, lon_e = -180.0, 180.0
        if notes is not None:
            notes.append(
                f"the {span:.1f}-degree forcing footprint uses the source's "
                "full longitude band (-180..180); only forcing coverage "
                "is expanded, with the forecast grid and latitude bounds preserved")
    if coverage is not None and lon_w <= lon_e:
        # A crossing box (W > E) cannot lie inside a non-crossing
        # envelope; it is left for the emission-time proof to refuse
        # loudly rather than silently reshaped here.
        from woof.fetch import area_bounds_inward

        cov_s, cov_w, cov_n, cov_e = area_bounds_inward(coverage)
        clamped_box = {
            "south": (lat_s, max(lat_s, cov_s)),
            "west": (lon_w, max(lon_w, cov_w)),
            "north": (lat_n, min(lat_n, cov_n)),
            "east": (lon_e, min(lon_e, cov_e)),
        }
        for edge, (raw, clamped) in clamped_box.items():
            if raw != clamped and coverage_notes is not None:
                coverage_notes.append(
                    f"the suggested forcing box's {edge} edge was "
                    f"clamped from {raw:.2f} to {clamped:.2f}: "
                    f"{coverage_label} coverage ends there (grid "
                    f"envelope lat {cov_s:.2f}..{cov_n:.2f}, "
                    f"lon {cov_w:.2f}..{cov_e:.2f})")
        lat_s, lat_n = clamped_box["south"][1], clamped_box["north"][1]
        lon_w, lon_e = clamped_box["west"][1], clamped_box["east"][1]
        if lat_s >= lat_n or lon_w >= lon_e:
            raise ValueError(
                f"the root domain's forcing footprint lies outside "
                f"{coverage_label} coverage (grid envelope lat "
                f"{cov_s:.2f}..{cov_n:.2f}, lon {cov_w:.2f}..{cov_e:.2f})"
                " entirely; choose a source whose coverage includes "
                f"{target_option}")
    return lat_s, lon_w, lat_n, lon_e


def fetch_area_hint(projection: dict, nx: int, ny: int, *, source: str,
                    root_dx_m: float = ROOT_DX_M,
                    target_option: str = "--point",
                    notes: list[str] | None = None,
                    coverage_notes: list[str] | None = None) -> str:
    """The exact ``--area`` string the wizard prints and writes, for SOURCE.

    One seam for the emission and its tests: the fitted root's forcing
    box (root corners + the source's own margin), bounded by the
    source's coverage envelope
    (:func:`woof.fetch.source_coverage_envelope` -- itself derived
    from the native grid definition, the same data the fetch guard
    enforces), formatted at the fixed precision the fetch parser reads
    back.  Every emission is then round-tripped through
    :func:`woof.fetch.validate_fetch_hints` before the file is
    written, so a hint this function produces and a command ``woof
    fetch`` refuses cannot coexist -- the field defect this closes.
    """

    from woof.fetch import AREA_HINT_DECIMALS, source_coverage_envelope

    coverage = source_coverage_envelope(source)
    area = _fetch_area(
        projection, nx, ny, margin_deg=_fetch_margin_deg(source),
        notes=notes, root_dx_m=root_dx_m, target_option=target_option,
        coverage=coverage,
        coverage_label=source.upper(), coverage_notes=coverage_notes,
        allow_full_longitude=(str(source).strip().lower() == "gfs"
                              or (projection["map_proj"] == "polar"
                                  and coverage is None)))
    return ",".join(f"{value:.{AREA_HINT_DECIMALS}f}" for value in area)


def source_coverage_refusal(projection: dict, nx: int, ny: int, *,
                            source: str,
                            root_dx_m: float = ROOT_DX_M,
                            target_option: str = "--point") -> str | None:
    """Why SOURCE's native grid cannot force this root, or ``None``.

    THE plan-time answer to the question the 2026-08-17 model battery could
    only get out of a preparation traceback.  ICON-EU over a central-US
    domain is a refusal by construction -- a European grid cannot reach
    Kansas -- and the run learned it after decoding 1,752 objects, 73
    seconds into a preparation, as ten lines of internal call stack.  The
    facts needed to say it first are all in the registry row: the source's
    declared window, and this root's own mass points.

    The message is built to be the same three sentences the preparation
    stage gets right: WHICH point is outside, WHERE it lands in the
    source's own index space, and WHAT the source covers.  That is what
    separates "this source does not reach the target" from "the crop is
    too small", and it is the difference between moving the domain and
    filing a bug.

    A global source (no declared window) returns ``None``: it reaches
    everything, and there is no bound to state.
    """

    gap = _root_coverage_gap(projection, nx, ny, source=source,
                             root_dx_m=root_dx_m)
    if gap is None:
        return None
    return (
        f"{gap} -- move {target_option} inside that grid, shrink the "
        f"ladder, or choose a source whose coverage includes this domain")


def _posix(path) -> str:
    return str(path).replace("\\", "/")


def _printed_path(path) -> str:
    """A path inside a printed command, quoted if a shell would split it.

    ``--out C:/my domains/case`` is a valid destination and was printed
    bare, so the "next:" command -- whose entire value is that it can be
    pasted -- became two arguments the moment it was.  Ordinary paths
    come back unquoted.
    """

    return shlex.quote(_posix(path))


def _relative_or_absolute(path: Path, base: Path) -> str:
    try:
        return _posix(os.path.relpath(path, base))
    except ValueError:  # different drive on Windows
        return _posix(Path(path).resolve())


def declared_nocturnal_night(profile: str | None, *, start_time: datetime,
                             hours: int, projection: dict):
    """First local night of this window IF the suite forces a declaration.

    ``None`` when nothing is declared: either the profile runs both
    radiation streams, or the window is all daylight.  A ``datetime``
    means :func:`render_config` will write
    ``acknowledgements = [ASYMMETRIC_RADIATION_NOCTURNAL_ACK]`` into the
    emitted ``[experiment]`` -- which disarms the load-time guard
    (:func:`woof.physics_compat.nocturnal_radiation_refusal`) at every
    other front door for this file.

    One function, two readers: the emission rule in
    :func:`render_config` and the spoken advisory in
    :func:`domain_main`.  They were allowed to be two expressions of the
    same predicate exactly once, and the result was a wizard that wrote
    the declaration and said nothing about it -- so the only statement of
    a nocturnally invalid run was a comment inside a file the reader had
    no reason to open.
    """

    shared = shared_physics(profile)
    lw = int(shared.get("ra_lw_physics", shared.get("ra_physics", 0)))
    sw = int(shared.get("ra_sw_physics", shared.get("ra_physics", 0)))
    if not (sw > 0 and lw == 0):
        return None
    return first_local_night_time(
        start_time, float(hours * 3600),
        ref_lat=projection["ref_lat"], ref_lon=projection["ref_lon"])


def render_config(*, name: str, start_time: datetime, hours: int,
                  projection: dict, dims: list[tuple[int, int]],
                  ratios: tuple[int, ...],
                  fetch_hints: dict, case_data: dict | None,
                  root_dx_m: float = ROOT_DX_M,
                  profile: str | None = DEFAULT_PHYSICS_PROFILE,
                  cumulus_requested: bool = False,
                  interactive: bool = False,
                  nz: int | None = None, tiles: str | None = None,
                  level_buffers_km: tuple[float, ...] | None = None,
                  history_interval_s: float | None = None,
                  nest_history_interval_s: float | None = None,
                  acknowledgements: tuple[str, ...] = (),
                  physics_mix: dict | None = None,
                  clock: str = "fixed",
                  noah_mosaic_options=None,
                  les_below_m: float | None = None,
                  les_closure: str | None = None) -> str:
    """The emitted TOML text (the exact bytes the wizard validates).

    ``les_below_m``/``les_closure`` only shape the header's turbulence
    advisory, so it describes the closure :func:`with_les_closure` writes
    afterwards; the closure itself is written by that function.

    ``clock`` is ``woof domain --clock`` (:func:`clock_decision`).  An
    adaptive emission writes ``use_adaptive_time_step = true`` into
    ``[shared]`` and states the clock in the header; a fixed one writes
    the same bytes it always did.  The library default is ``fixed`` so a
    caller that does not ask keeps its file.

    ``physics_mix`` is a ``woof physics-catalog --check`` request whose
    ``choices`` replace the suite's own schemes (``--physics-choices``):
    see :func:`with_physics_mix`.

    ``acknowledgements`` is written into ``[experiment]`` verbatim and is
    the ONLY source of that field.  Until 2026-08-09 this function wrote
    ``[ASYMMETRIC_RADIATION_NOCTURNAL_ACK]`` into every emitted file
    whose selected suite met a night window -- a declaration the user
    never made, in a file that outlives the terminal session, which
    disarmed the load-time guard at ``woof check``, ``woof run``,
    ``woof go``, run-plan and both prepared runners forever after.  An
    acknowledgement is a person stating that they know what they are
    running; a program cannot make it on their behalf and have it mean
    anything.  ``woof domain --ack <id>`` is how it is made now, and
    :func:`domain_main` REFUSES rather than emit a file that needs one it
    was not given.

    ``interactive`` records WHICH front door authored the file -- the
    prompt session or the flags -- in the header the file already
    carries.  Both doors produce the same config for the same answers,
    so this is provenance rather than a difference: it tells whoever
    reads the file later how the numbers in it were arrived at, which
    is the question asked of an emitted file nobody remembers writing.
    """
    experiment = {
        "name": name, "start_time": start_time,
        "run_seconds": float(hours * 3600), "feedback": 0,
        "smooth_option": 0, "blend_width": _BLEND_WIDTH,
        "spec_bdy_width": _SPEC_BDY_WIDTH,
        # Both prepared runners use the canonical checkpoint transport.
        # New configurations checkpoint hourly, or at the end of a shorter
        # run. The existing event-clock author below binds this interval;
        # explicitly authored configurations, including zero/off, bypass it.
        "restart_interval_s": min(DEFAULT_RESTART_INTERVAL_S, float(hours * 3600)),
    }
    shared = shared_physics(profile)
    if nz is not None:
        from woof.core.grid import resample_eta_levels

        levels = resample_eta_levels(_ETA_LEVELS, nz)
        if nz < 4:
            raise ValueError("--nz must be at least 4 (the vertical stencil width)")
        shared["nz"] = nz
        shared["eta_levels"] = tuple(float(level) for level in levels)
    from woof.physics_source_defaults import (authored_ruc_defaults,
                                               land_scoped_defaults,
                                               recipe_physics_defaults)
    shared.update(land_scoped_defaults(
        recipe_physics_defaults((fetch_hints or {}).get("source")),
        shared.get("sf_surface_physics"), shared.get("mp_physics")))
    # A RUC suite from any source starts the ground vapour from the air
    # (physics_source_defaults.AUTHORED_RUC_DEFAULTS); a source recipe that
    # states it keeps its value.
    for key, value in authored_ruc_defaults(shared.get("sf_surface_physics")).items():
        shared.setdefault(key, value)
    if tiles is not None and tiles not in {"off", "auto", "on"}:
        raise ValueError("--tiles must be off, auto, or on")
    # The vertical default is bounded by the source's certified column:
    # the ladder is eta-normalized, so only p_top moves (see
    # DEFAULT_MODEL_TOP_PA / emitted_model_top_pa).
    shared["p_top"] = emitted_model_top_pa(
        (fetch_hints or {}).get("source"))
    shared["map_proj"] = WRF_MAP_PROJ_CODES[projection["map_proj"]]
    # Nocturnal validity of the emitted radiation pairing, stated in the
    # header of EVERY emitted file and, where the pairing is asymmetric
    # (shortwave on, longwave off) across a window that includes local
    # night, DECLARED in [experiment].acknowledgements -- the same
    # declaration the load-time guard in woof.experiment demands, so an
    # explicitly selected validation profile emits a file that still
    # loads.  Asymmetric suites are never a default on the gfs/era5
    # doors; explicit selection is the declaration, and it is made in
    # ink here rather than by silence.
    emitted_lw = int(shared.get("ra_lw_physics", shared.get("ra_physics", 0)))
    emitted_sw = int(shared.get("ra_sw_physics", shared.get("ra_physics", 0)))
    asymmetric_radiation = emitted_sw > 0 and emitted_lw == 0
    # One predicate, read by the emission rule here and by the spoken
    # advisory in domain_main: see declared_nocturnal_night.  It asks
    # exactly what the inline expression here used to ask -- shortwave on,
    # longwave off, and a local night inside the window.
    first_night = declared_nocturnal_night(
        profile, start_time=start_time, hours=hours, projection=projection)
    # The SECOND declaration, and a separate question from the first:
    # with no longwave scheme under a land-surface scheme, downward
    # longwave is a declared constant rather than a computed flux, at
    # noon as much as at midnight.  The nocturnal token used to stand in
    # for this by accident -- it is checked before any physics is
    # inspected, so a config carrying it never had its GLW source looked
    # at -- and an all-daylight asymmetric emission declared nothing at
    # all while still running Noah on a fixed 300 W m-2.  Both are now
    # stated, separately, in ink.  The condition is the load guard's own
    # classification, not a re-derivation, so an emission can never pass
    # here and refuse there.
    from woof.physics_compat import downward_longwave_disposition
    _glw_kind, _ = downward_longwave_disposition(
        ra_lw_physics=emitted_lw, ra_sw_physics=emitted_sw,
        sf_surface_physics=int(shared.get("sf_surface_physics", 0)))
    constant_longwave = _glw_kind in ("consumed", "published")
    # WHO DECLARES WHICH TOKEN, and why the two are not treated alike.
    #
    # The NOCTURNAL token is a claim about the WINDOW the user chose --
    # "I know this run contains night and I want it anyway" -- and the
    # wizard cannot make it for them.  It wrote that line by itself
    # through 1.8.7, into a file that outlives the terminal session, and
    # the line disarmed the load guard at every other front door
    # forever after.  It now comes from ``woof domain --ack`` or not at
    # all, and domain_main refuses rather than emit a file needing one
    # it was not given.
    #
    # The CONSTANT-LONGWAVE token is not a judgment: it is a mechanical
    # consequence of the SUITE the user named on the command line, true
    # of every window and every place that suite is run in.  Naming the
    # profile IS the declaration, so it is written here in ink -- with
    # the JUSTIFY line the shipped-config convention requires -- rather
    # than left to silence.  Emitting it is what keeps the wizard's own
    # daylight output loadable, which is the property that made this
    # worth separating: an all-daylight asymmetric run has no nocturnal
    # claim to make and still integrates a fabricated flux.
    acknowledgements = tuple(acknowledgements)
    emitted_acks = list(acknowledgements)
    if constant_longwave and CONSTANT_DOWNWARD_LONGWAVE_ACK not in emitted_acks:
        emitted_acks.append(CONSTANT_DOWNWARD_LONGWAVE_ACK)
    if emitted_acks:
        experiment["acknowledgements"] = emitted_acks
    declared = ASYMMETRIC_RADIATION_NOCTURNAL_ACK in acknowledgements
    if not asymmetric_radiation:
        nocturnal_note = (
            "# NOCTURNALLY VALID: longwave and shortwave both run, so the "
            "surface longwave\n"
            "# budget stays closed through the night.\n")
    elif first_night is None:
        nocturnal_note = (
            "# NOT NOCTURNALLY VALID (shortwave on, longwave OFF) -- "
            "acceptable for this\n"
            "# all-daylight window; re-emit with a full lw+sw profile "
            "before running any\n"
            "# window that includes local night.\n")
    elif declared:
        nocturnal_note = (
            "# NOT NOCTURNALLY VALID: shortwave heats by day, longwave is "
            "OFF, and this\n"
            f"# window includes local night (first at "
            f"{first_night:%Y-%m-%dT%H:%M}Z), so the surface\n"
            "# radiates with no downward longwave after sunset and skin "
            "temperature and\n"
            "# 2 m moisture collapse.  Emitted because YOU declared it: "
            "the acknowledgement\n"
            "# in [experiment] below came from `woof domain --ack "
            f"{ASYMMETRIC_RADIATION_NOCTURNAL_ACK}`,\n"
            "# and it disarms the load guard at every other front door "
            "for this file --\n"
            "# check, run, go, run-plan and both prepared runners.\n")
    else:
        # `woof domain` refuses this combination before it renders (see
        # domain_main), so this is the LIBRARY caller's copy: a file that
        # will not load must not also look fine.
        nocturnal_note = (
            "# NOT NOCTURNALLY VALID, AND THIS FILE WILL NOT LOAD: "
            "shortwave heats by day,\n"
            "# longwave is OFF, and this window includes local night "
            "(first at\n"
            f"# {first_night:%Y-%m-%dT%H:%M}Z).  Every front door refuses "
            "it at config load.\n"
            "# Re-emit with a full lw+sw profile, or -- if you mean the "
            "daytime-only\n"
            "# suite and accept the night -- re-emit with `woof domain "
            "--ack\n"
            f"# {ASYMMETRIC_RADIATION_NOCTURNAL_ACK}`.\n")
    if constant_longwave:
        # The justification convention shipped configs are held to
        # (tests/test_shipped_acknowledgement_justifications.py), written
        # into every emitted file that needs the token so an emitted
        # config meets the same bar as a committed one.  The middle
        # sentence states the actual exposure -- integrated by the land
        # surface, or merely published to wrfout -- from the same
        # disposition the guard read.
        if _glw_kind == "consumed":
            exposure = (
                "# the surface integrates a declared constant 300 W m-2 "
                "for the whole\n"
                "# forecast.")
        else:
            exposure = (
                "# the declared constant 300 W m-2 is published as the "
                "GLW row of every\n"
                "# wrfout frame.")
        nocturnal_note += (
            f"# JUSTIFY {CONSTANT_DOWNWARD_LONGWAVE_ACK}: this suite "
            "sets\n"
            "# ra_lw_physics = 0, so NOTHING computes downward longwave "
            "and\n"
            + exposure +
            "  Emitted only because this profile was selected "
            "explicitly.\n"
            "# Re-emit with a full lw+sw profile for any run whose "
            "surface fields you\n"
            "# intend to believe.\n")
    recommended_time_step = root_time_step_s(projection["ref_lat"], root_dx_m)
    clock_kwargs = dict(run_seconds=experiment["run_seconds"], ratios=ratios,
        history_interval_s=history_interval_s,
        nest_history_interval_s=nest_history_interval_s,
        restart_interval_s=experiment["restart_interval_s"])
    event_clock = derived_time_step_s(projection["ref_lat"], root_dx_m, **clock_kwargs)
    chain_km = _ladder_dx_km(ratios, root_dx_m)
    cu_by_domain = cumulus_by_domain(
        dims, ratios, profile=profile, root_dx_m=root_dx_m,
        cumulus_requested=cumulus_requested)
    root_cu = cu_by_domain[0]
    clock_physics = {key: profile_switches(profile)[key] for key in _PER_DOMAIN_PHYSICS}
    clock_physics["cu_physics"] = root_cu
    if root_cu == 0:
        clock_physics["cudt_minutes"] = 0.0
    cadence_notes = ()
    if not cumulus_requested and event_clock == recommended_time_step:
        # Keep already loadable omitted-profile output, including its old
        # spoken default-cadence reconciliation. A named suite is authority;
        # its cadence never moves to accommodate our omitted timestep.
        adjusted, notes = snap_cadences_to_clock(recommended_time_step, clock_physics)
        if all((period / recommended_time_step).denominator == 1
               for period in _root_physics_periods(adjusted, shared)):
            clock_physics, cadence_notes = adjusted, notes
    time_step = derived_time_step_s(projection["ref_lat"], root_dx_m,
        physics_periods_s=_root_physics_periods(clock_physics, shared), **clock_kwargs)
    retired = cumulus_retired_note(
        profile, root_dx_m / 1000.0, cumulus_requested=cumulus_requested)
    # THE VERBATIM CLAIM IS TRUE OR IT IS NOT MADE.  Where the emission
    # retired the suite's cumulus it names the switch it moved, instead
    # of asserting a switch-for-switch identity this file does not have.
    verbatim_claim = (
        "Taken verbatim from woof.physics_compat EXCEPT the root's "
        "cumulus\n# switch (see CUMULUS OFF below), so this file passes "
        "the prepared-"
        if retired else
        "Taken verbatim from woof.physics_compat, so this file passes "
        "the prepared-")
    mix_line = ""
    if physics_mix:
        # The file runs the picked schemes, so it says so, and it does not
        # claim to be the suite verbatim.
        mix_line = (f"# PHYSICS MIX: {physics_mix_words(physics_mix.get('choices') or {})} "
                    "replace the suite's own schemes.\n")
        verbatim_claim = (
            "The switches below are that mix, not the suite verbatim, and "
            "no suite is asserted, so this file passes the prepared-")
    if level_buffers_km is None:
        # This is the original point header byte-for-byte.  Polygon support
        # must not perturb existing point-authored artifacts.
        header = (
            "# Emitted by `woof domain`"
            + (" (interactive session)" if interactive else "")
            + " -- point "
            f"{projection['ref_lat']:g},{projection['ref_lon']:g}, ladder "
            f"{'-'.join(f'{v:g}' for v in chain_km)} km.\n"
            f"# PHYSICS: {physics_summary(profile, cu_physics=root_cu)}.\n"
            + mix_line +
            f"# {verbatim_claim}\n"
            "# forecast runner's profile guard as emitted.  Child dx/dt "
            "derive exactly from\n"
            "# the parent chain and are never hand-typed "
            "(woof/experiment.py).\n")
    else:
        buffers = ", ".join(f"{value:g}" for value in level_buffers_km)
        header = (
            "# Emitted by `woof domain` -- polygon center "
            f"{projection['ref_lat']:g},{projection['ref_lon']:g}, ladder "
            f"{'-'.join(f'{v:g}' for v in chain_km)} km.\n"
            f"# Polygon buffers by domain level (outer to inner): "
            f"{buffers} km.\n"
            f"# PHYSICS: {physics_summary(profile, cu_physics=root_cu)}.\n"
            + mix_line +
            f"# {verbatim_claim}\n"
            "# forecast runner's profile guard as emitted.  Child dx/dt "
            "derive exactly from\n"
            "# the parent chain and are never hand-typed "
            "(woof/experiment.py).\n")
    header += nocturnal_note
    for note in cadence_notes:
        header += "# " + note + "\n"
    if time_step != recommended_time_step:
        header += "# " + _derived_clock_note(
            recommended_time_step, time_step, _root_physics_periods(clock_physics, shared)) + "\n"
    per_km = seconds_per_km(projection["ref_lat"])
    if per_km != 5:
        clock_label = ("time_step" if time_step == recommended_time_step
                       else "spacing-derived time_step")
        header += (
            f"# TROPICAL CLOCK: |lat| < {MERCATOR_MAX_LAT:g}, so "
            f"{clock_label} is {float(recommended_time_step):g} s "
            f"({float(per_km):g} s per km), half the "
            f"{float(recommended_time_step) * 2:g} s\n"
            "# the 5 s/km convention would give at this dx.  The "
            "stability gate uses the\n"
            "# maximum co-located vertical |w|/layer-thickness; "
            "tropical convection\n"
            "# destabilised a measured 12 km Mercator domain at 60 s and "
            "was comfortably\n"
            "# stable at a shorter step, for +22% wall time (radiation "
            "and cumulus are\n"
            "# called on wall-clock intervals, so extra dynamics steps "
            "are cheap).\n")
    for line in retired:
        header += f"# {line}\n"
    for line in gray_zone_advisory(chain_km, shared, les_below_m, les_closure):
        header += f"# {line}\n"
    for line in cumulus_gray_zone_advisory(chain_km, cu_by_domain):
        header += f"# {line}\n"
    adaptive, _ = clock_decision(clock, time_step=time_step,
                                 root_dx_m=root_dx_m, ratios=ratios)
    if adaptive:
        # The one key.  The bounds stay WRF's per-spacing fill-ins (-1),
        # because a clamp written once into [shared] reaches every nest
        # in seconds (docs/ADAPTIVE-TIMESTEP.md, "A shared clamp is a
        # per-domain trap"), and the first step stays the time_step
        # below.  The terrain clock and the steep-ground substep rule run
        # at launch and write their own ceiling and substep floor onto
        # the domains that need them.
        shared["use_adaptive_time_step"] = True
        if ratios:
            # Declare the existing 5% child-growth default in generated nests.
            # Shared inheritance survives catalog geometry reconstruction;
            # targets and per-resolution bounds retain their existing values.
            shared["max_step_increase_pct"] = 5
        header += (
            "# CLOCK: adaptive.  Each grid starts at the time_step below "
            "and then follows\n"
            "# its own Courant number between 3 and 8 s per km of its "
            "spacing; over steep\n"
            "# ground the terrain clock caps it at launch.  Re-emit with "
            "`--clock fixed`\n"
            "# for one step throughout.\n")
    parts = [
        header,
        _render_table("experiment", experiment),
        _render_table("projection", projection),
        _render_table("shared", shared),
    ]
    if tiles is not None:
        parts.append(_render_table("tiles", {"mode": tiles}))
    from woof.physics_source_defaults import (
        recipe_root_defaults, with_recipe_root_defaults)
    domain_tables = _domain_tables(
            dims, ratios, time_step=time_step, root_dx_m=root_dx_m,
            profile=profile, cumulus_requested=cumulus_requested,
            history_interval_s=history_interval_s,
            nest_history_interval_s=nest_history_interval_s)
    with_recipe_root_defaults(
        shared, domain_tables, recipe_root_defaults((fetch_hints or {}).get("source")))
    for table in domain_tables:
        parts.append(_render_table("domain", table, array_of_tables=True))
    if fetch_hints:
        parts.append(_render_table(
            "fetch", fetch_hints,
            comment="Advisory data-acquisition hints (validated, not "
                    "executed); keys mirror `woof fetch` flags."))
    else:
        # NO [fetch] TABLE, DELIBERATELY.  The table is validated at every
        # config load against the sources `woof fetch` can actually
        # download, so writing one for a source with no fetch route would
        # emit a file that refuses to load -- and writing one that loads
        # would advertise a download this ArWen cannot make.  The
        # acquisition route is stated as a comment instead, which is the
        # accurate shape until the fetch door grows the route.
        parts.append(
            "# NO [fetch] TABLE: `woof fetch` has no download route for\n"
            "# this source yet, so its bytes are staged by hand (see\n"
            "# docs/public/SOURCES.md, 'Sources with no fetch door').  The\n"
            "# geometry, levels, physics and boundary cadence in this file\n"
            "# are complete -- only the acquisition step is manual.\n")
    if case_data is not None:
        parts.append(_render_table(
            "case_data", case_data,
            comment="Declared inputs for the config-driven "
                    "check/static/ingest/run front door (ERA5 native-GRIB1 "
                    "route; era5_z_invariant source orography)."))
    text = with_physics_mix("\n".join(parts), physics_mix)
    return (text if noah_mosaic_options is None else
            with_noah_mosaic_options(text, *noah_mosaic_options))


def physics_mix_words(choices) -> str:
    """The picked schemes in a line: ``microphysics thompson-mp8, pbl myj``."""

    words = []
    for family, choice in sorted(dict(choices).items()):
        if isinstance(choice, dict):
            choice = ", ".join(f"{key} {value}" for key, value in sorted(choice.items()))
        words.append(f"{str(family).replace('_', ' ')} {choice}")
    return ", ".join(words)


def physics_mix_request(text: str | None, *, source: str,
                        profile: str | None) -> dict | None:
    """``--physics-choices`` as the check request the mix is written with.

    ``text`` is the JSON object the flag carries (family to scheme, or
    for radiation a ``{"longwave": n, "shortwave": n}`` pair).  The
    suite it changes is the named one, else the default at the written
    file's finest grid, which is what the check takes when it names no
    suite (:func:`woof.physics_catalog.apply_to_experiment` reads that
    grid from the file).
    """

    if text is None:
        return None
    try:
        choices = json.loads(text)
    except ValueError as error:
        raise ValueError(f"--physics-choices must be a JSON object of family to scheme: {error}") from None
    if not isinstance(choices, dict) or not choices or not all(
            isinstance(key, str) and isinstance(value, (str, dict)) for key, value in choices.items()):
        raise ValueError("--physics-choices must be a JSON object of family to scheme, for example "
                         '{"microphysics": "thompson-mp8"}')
    mix = {"choices": choices, "source": source}
    if profile is not None:
        mix["suite"] = profile
    return mix


def with_physics_mix(text: str, physics_mix: dict | None) -> str:
    """``text`` running the mix's schemes in place of its suite's own.

    The write is the one ``woof physics-catalog --check JSON --into
    EXPERIMENT.toml`` makes (:func:`woof.physics_catalog.apply_to_experiment`):
    the engine's check answers at this file's own root spacing first, and
    a refused mix is refused in its words.  Every size the fit tries is
    rendered through here, so the card is priced for the schemes that
    run, not the suite they replace; each is loaded by the caller, as
    every rendering is (:func:`experiment_from_text`).  ``None`` leaves
    the text as is.
    """

    if not physics_mix:
        return text
    from woof.physics_catalog import CatalogError, apply_to_experiment

    try:
        mixed = apply_to_experiment(text, physics_mix, load=False)
    except CatalogError as error:
        raise ValueError(f"--physics-choices: {error}") from None
    # The header's PHYSICS line was written for the suite the mix
    # replaced; it names what the root now runs instead, since the line
    # and the switches it describes sit in the same file.
    return re.sub(r"^# PHYSICS: .*$", lambda _: f"# PHYSICS: {mix_physics_summary(mixed, physics_mix)}.",
                  mixed, count=1, flags=re.M)


def mix_physics_summary(text: str, physics_mix: dict) -> str:
    """:func:`physics_summary` of a mixed file's root, as the file runs it."""

    from woof.physics_catalog import experiment_grid, request_default_suite

    document = tomllib.loads(text)
    root = {**(document.get("shared") or {}), **((document.get("domain") or [{}])[0])}
    # With no suite named, the default at this file's finest grid: the
    # base the check changed when it wrote the mix.
    base = physics_mix.get("suite") or request_default_suite(
        {**experiment_grid(text), **physics_mix})
    return physics_summary(None, switches=root, label=f"schemes picked over {base}")


#: What each ``isftcflx`` value does, in the words the emitted file carries.
#: The kernel implements all three (woof/core/kernels/sfclay.cu); only the
#: MM5 surface layer reads it, and the loader refuses it on any other.
ISFTCFLX_WORDS = {
    0: "standard MM5 roughness over water",
    1: "Donelan drag, which levels off in strong wind, with a constant "
       "heat and moisture roughness (WRF's tropical cyclone option)",
    2: "Donelan drag with Garratt heat and moisture roughness",
}


def with_surface_flux_option(text: str, isftcflx: int | None) -> str:
    """``text`` with ``isftcflx`` set tree-wide in ``[shared]``.

    The one physics switch a caller may set beside a named suite: it does
    not select the suite (it is not a profile selector key) and it moves
    no memory, so the fit is unchanged.  ``None`` leaves the text as is.
    """
    if isftcflx is None:
        return text
    value = int(isftcflx)
    if value not in ISFTCFLX_WORDS:
        raise ValueError(f"--isftcflx must be 0, 1 or 2, got {isftcflx!r}")
    lines = text.splitlines(keepends=True)
    try:
        start = next(i for i, line in enumerate(lines)
                     if line.strip() in ("[shared]", '["shared"]'))
    except StopIteration:
        raise ValueError("the emitted config has no [shared] table to "
                         "carry --isftcflx") from None
    end = next((i for i in range(start + 1, len(lines))
                if lines[i].lstrip().startswith("[")), len(lines))
    lines = lines[:start + 1] + [line for line in lines[start + 1:end]
                                 if not line.lstrip().startswith("isftcflx")] + lines[end:]
    note = f"# Surface flux over water: isftcflx = {value}, {ISFTCFLX_WORDS[value]}.\n"
    lines[start + 1:start + 1] = [note, f"isftcflx = {value}\n"]
    return "".join(lines)


#: The resolved-turbulence closure a domain at or below the LES spacing
#: takes (docs/public/LES.md; configs/les_nest_250m_km3.toml's child, the
#: recipe the gray-zone advisory names): 3-D Smagorinsky with the 1-D PBL
#: off, one isotropic mixing length, and surface fluxes from the surface
#: layer.  Per domain, so a PBL parent carries a PBL-off child.
LES_CLOSURE = (("km_opt", 3), ("bl_pbl_physics", 0), ("c_s", 0.25),
               ("mix_isotropic", 1), ("mix_upper_bound", 0.1), ("isfflx", 1))
#: The same recipe on WRF's 1.5-order prognostic TKE closure (km_opt 2,
#: c_k at its Registry 0.15): the second closure docs/public/LES.md
#: measures, kernel-checked against FP64 and within 0.1-0.3 % of WRF em_les
#: at 100 m.
LES_CLOSURE_TKE = (("km_opt", 2), ("bl_pbl_physics", 0),
                   ("mix_isotropic", 1), ("mix_upper_bound", 0.1),
                   ("isfflx", 1))
#: ``--les-closure`` spelling -> the keys an LES domain takes.
LES_CLOSURES = {"smagorinsky": LES_CLOSURE, "tke": LES_CLOSURE_TKE}
DEFAULT_LES_CLOSURE = "smagorinsky"
#: The spacing at and below which ``woof domain`` writes LES_CLOSURE on a
#: domain unless ``--les-below-m 0`` (the advisory's "coarse LES" bound).
DEFAULT_LES_BELOW_M = 250.0
#: Words for the closure, for the file header and stdout.
LES_CLOSURE_WORDS = {"smagorinsky": "3-D Smagorinsky (km_opt 3)",
                     "tke": "1.5-order prognostic TKE (km_opt 2)"}


def les_closure_keys(closure: str | None) -> tuple:
    """The (key, value) pairs ``closure`` writes, or a refusal naming the choices."""
    name = DEFAULT_LES_CLOSURE if closure is None else str(closure)
    try:
        return LES_CLOSURES[name]
    except KeyError:
        raise ValueError(f"--les-closure must be one of "
                         f"{', '.join(sorted(LES_CLOSURES))}, got {closure!r}") from None


def les_domain_indices(chain_km, below_m: float | None) -> list[int]:
    """Indices of the domains ``woof domain`` writes the LES closure on.

    Every real-data domain at or below ``below_m``, the root included: a
    root this fine is the top grid of a run fed by specified boundaries
    (the hex add-nest's single WOOF grid, an API single grid), which used
    to keep the suite's 1-D PBL at LES spacing because the rule skipped
    index 0.
    """
    if not below_m:
        return []
    return [i for i, dx_km in enumerate(chain_km)
            if float(dx_km) * 1000.0 <= float(below_m)]


def _domain_blocks(lines: list[str]) -> list[tuple[int, int]]:
    """(start, end) line spans of every ``[[domain]]`` block, in order."""
    starts = [i for i, line in enumerate(lines)
              if line.strip() in ("[[domain]]", '[["domain"]]')]
    spans = []
    for start in starts:
        end = next((i for i in range(start + 1, len(lines))
                    if lines[i].lstrip().startswith("[")), len(lines))
        spans.append((start, end))
    return spans


def _domain_spacings_m(text: str) -> list[float]:
    """Each [[domain]]'s grid spacing in metres, in file order."""
    raw = tomllib.loads(text)
    domains = raw.get("domain", [])
    by_id = {int(d["grid_id"]): d for d in domains}
    out = []
    for d in domains:
        dx, node = None, d
        ratio = 1
        while node is not None:
            if "dx" in node:
                dx = float(node["dx"]) / ratio
                break
            ratio *= int(node.get("parent_grid_ratio", 1))
            node = by_id.get(int(node.get("parent_id", 0)))
        out.append(float("nan") if dx is None else dx)
    return out


def _block_value(lines: list[str], start: int, end: int, key: str):
    """The raw right-hand side of ``key`` inside one [[domain]] block, or None."""
    for line in lines[start + 1:end]:
        if ("=" in line and not line.lstrip().startswith("#")
                and line.split("=", 1)[0].strip() == key):
            return line.split("=", 1)[1].split("#", 1)[0].strip()
    return None


def with_les_closure(text: str, below_m: float | None,
                     closure: str | None = None,
                     inflow_seeding: bool = True) -> str:
    """``text`` with the LES closure on every domain at or below ``below_m``.

    The fix for the door only ADVISING the closure: a 100 m nest written
    by this door ran the parent's 1-D PBL inside an LES grid unless the
    user edited the file (WOOF run-controls feasibility, defect 6).  A key
    the block already writes is kept, and ``below_m`` ``None`` or 0 leaves
    the text as is.

    The root takes it too when it is a real-data grid (``specified =
    true``): the hex add-nest and the API's single grid are the top domain
    of their run, and skipping index 0 left them on the suite's 1-D PBL at
    any spacing.  An idealised root (periodic or open sides) is still
    never changed.

    ``inflow_seeding`` writes ``inflow_perturbation = true`` on an LES
    nest whose parent still runs the 1-D PBL: the generator's own
    applicability (woof.core.inflow_perturbation.parent_pbl_refusal), and
    the handoff it was built for.  Measured: on the shipped 250 m child
    the resolved-turbulence profile reached its plateau inside the domain
    in 4 of 4 afternoon frames with it on and 1 of 4 with it off
    (docs/superpowers/receipts/les/inflow-generator/PILOT-ON-2026-08-04.md,
    G4 REDUCED), for wall time inside run-to-run noise (468.0 s against
    466.9 s, PROOFS-2026-08-04.md).  An LES nest under an LES parent is
    left off: the parent's resolved eddies are its inflow turbulence.
    """
    if not below_m:
        return text
    keys = les_closure_keys(closure)
    name = DEFAULT_LES_CLOSURE if closure is None else str(closure)
    spacings = _domain_spacings_m(text)
    lines = text.splitlines(keepends=True)
    spans = _domain_blocks(lines)
    raw = tomllib.loads(text)
    domains = raw.get("domain", [])
    shared_pbl = int(raw.get("shared", {}).get("bl_pbl_physics", 0) or 0)
    # Seeding is measured on resident state only (the pilot and its
    # proofs); a streamed tree ([tiles] on/auto) keeps it off until a
    # streamed LES nest has run with it.
    streamed = str(raw.get("tiles", {}).get("mode", "off")) not in ("off", "")
    inflow_seeding = bool(inflow_seeding) and not streamed
    les = {i for i, dx in enumerate(spacings) if dx <= float(below_m)}
    if 0 in les:
        start, end = spans[0]
        if str(_block_value(lines, start, end, "specified")).lower() != "true":
            les.discard(0)
    index_of = {int(d.get("grid_id", i + 1)): i for i, d in enumerate(domains)}
    for index in reversed(range(len(spans))):
        if index not in les:
            continue
        start, end = spans[index]
        present = {line.split("=", 1)[0].strip() for line in lines[start + 1:end]
                   if "=" in line and not line.lstrip().startswith("#")}
        keep = [f"{key} = {_toml_value(value)}\n" for key, value in keys
                if key not in present]
        parent = (index_of.get(int(domains[index].get("parent_id", 0) or 0))
                  if index < len(domains) else None)
        parent_pbl = None
        if parent is not None and parent not in les:
            pstart, pend = spans[parent]
            own = _block_value(lines, pstart, pend, "bl_pbl_physics")
            parent_pbl = shared_pbl if own is None else int(float(own))
        seed = (inflow_seeding and index != 0 and parent_pbl
                and "inflow_perturbation" not in present)
        if seed:
            keep.append("inflow_perturbation = true\n")
        if not keep:
            continue
        what = "nest" if index else "grid"
        note = (f"# LES closure: {spacings[index]:g} m is at or below "
                f"{float(below_m):g} m, so this {what} resolves its boundary "
                f"layer (PBL off, {LES_CLOSURE_WORDS[name]}).\n")
        if seed:
            note += ("# Inflow seeding on: the parent runs a 1-D PBL, so "
                     "cell perturbations on the inflow faces start the "
                     "turbulence the smooth parent inflow lacks.\n")
        insert = end
        while insert > start + 1 and not lines[insert - 1].strip():
            insert -= 1
        lines[insert:insert] = [note, *keep]
    return "".join(lines)


def with_quicklook(text: str, every_s: float | None,
                   history_vars: tuple[str, ...] | None = None) -> str:
    """``text`` with the quick-look stream on its FINEST domain.

    ``quicklook = { every_s, history_vars }`` (woof.experiment
    DomainConfig.full_history_interval_s): the finest domain keeps its
    full frames at its history interval and writes the subset every
    ``every_s`` between them.  ``None`` leaves the text as is.
    """
    if every_s is None:
        return text
    lines = text.splitlines(keepends=True)
    spans = _domain_blocks(lines)
    if not spans:
        raise ValueError("the emitted config has no [[domain]] to carry "
                         "--quicklook-interval")
    spacings = _domain_spacings_m(text)
    finest = min(range(len(spans)), key=lambda i: spacings[i])
    start, end = spans[finest]
    if any(line.split("=", 1)[0].strip() == "quicklook"
           for line in lines[start + 1:end]):
        return text
    table = {"every_s": float(every_s)}
    if history_vars:
        table["history_vars"] = list(history_vars)
    value = "{ " + ", ".join(f"{k} = {_toml_value(v)}" for k, v in table.items()) + " }"
    note = (f"# Quick-look stream: the subset every {float(every_s):g} s "
            "between this domain's full frames (quicklook/wrfout_dNN_*).\n")
    insert = end
    while insert > start + 1 and not lines[insert - 1].strip():
        insert -= 1
    lines[insert:insert] = [note, f"quicklook = {value}\n"]
    return "".join(lines)


def with_noah_mosaic_options(text: str, option=None, count=None,
                             canopy=None) -> str:
    """Write explicit run-wide options; omission preserves the emitted bytes.

    ``canopy`` is ``mosaic_urban_canopy``, written to [shared] so every
    domain takes it."""
    if option is None and count is None and canopy is None:
        return text
    lines = text.splitlines(keepends=True)
    start = next(i for i, line in enumerate(lines) if line.strip() == "[shared]")
    end = next((i for i in range(start + 1, len(lines))
                if lines[i].lstrip().startswith("[")), len(lines))
    supplied = {key: value for key, value in (
        ("sf_surface_mosaic", option), ("mosaic_cat", count),
        ("mosaic_urban_canopy", None if canopy is None else f'"{canopy}"'))
        if value is not None}
    lines[start + 1:end] = [line for line in lines[start + 1:end]
                             if line.split("=", 1)[0].strip() not in supplied]
    lines[start + 1:start + 1] = [f"{key} = {value}\n" for key, value in supplied.items()]
    result = "".join(lines)
    experiment_from_text(result, source="domain mosaic options")
    return result


def experiment_from_text(text: str, *, source: str) -> ExperimentConfig:
    """Round-trip emitted text through the canonical owner-validating builder.

    Companion [fetch], [case_data], [static] and [ingest] tables share
    the same validation boundary as the CLI file loader.

    [static] is validated here and consumed where the statics are built:
    each preparation route reads it back from the config file
    (:func:`woof.static.highres_production.load_static_highres`).  Left
    in the bare experiment-table builder, it stopped ``woof go`` on any
    config that turned the
    high-resolution overlay on, before a byte was fetched.
    """
    raw = tomllib.loads(text)
    from woof.experiment import build_experiment_from_config_tables

    # The wizard authors a one-file case with source-selected static metadata.
    # Use the same validating owner boundary as the file loader, including
    # case_data/static/ingest. Bare build_experiment remains strict.
    return build_experiment_from_config_tables(
        raw, source=source, base_dir=Path(source).parent)


def sizing_budget_bytes(exp: ExperimentConfig, *, free_bytes: int,
                        vram_gib: float | None,
                        forcing_interval_seconds: float,
                        profile=None) -> int:
    """The budget the machine-peak ENVELOPE is compared against.

    ``free_bytes`` minus what this process's own envelope does not model,
    and nothing else.  :func:`~woof.core.preflight.
    machine_peak_envelope_bytes` is a model of the WHOLE device residency
    a run of this configuration reaches -- the itemized pool, the CUDA
    context, the local-memory backing store of its kernel set, the
    measured pool margin and the measured residue -- so the only thing
    left outside it is OTHER processes, which is exactly
    :data:`~woof.core.preflight.EXTERNAL_MARGIN_BYTES`.

    IT USED TO SUBTRACT THE ALLOCATION RESERVE, and the allocation
    reserve carries the non-pool residency too.  The envelope was then
    compared against a budget from which its own CUDA context and its own
    backing store had ALREADY been removed: one process charged twice for
    the same bytes.  On a 10 GiB RTX 3080 that was 2.91 GiB of a 10 GiB
    card spent twice, and the hrrr ladder's SMALLEST layout -- 60x48 --
    was refused on a card that fits it (task 206; the acceptance walk is
    in the lane report).  The two doors are still coherent, and more
    strictly than before: this comparison is provably tighter than
    ``woof check``'s allocation gate at every grid size, so a wizard
    PASS remains a check PASS (``test_vram_measured_reserve.py::
    test_the_fit_gate_is_never_looser_than_the_allocation_gate``).

    ``profile`` is the device the non-pool terms were priced against and
    is accepted so callers can pass the card they MEASURED; it does not
    enter this arithmetic, and is kept in the signature because every
    caller of this function has to have decided it.
    """

    del exp, vram_gib, forcing_interval_seconds, profile  # see above
    return int(free_bytes) - EXTERNAL_MARGIN_BYTES


@dataclass(frozen=True)
class LighterProfiles:
    """What a memory refusal may say about the physics suite.

    ``fitting`` is the advice: suites that price below the refused one
    AND fit the refused budget, in the order a reader should try them.
    ``lightest`` is the cheapest admissible suite that prices below the
    refused one, as ``(name, bytes)``, whether or not it fits; it lets a
    refusal with nothing to advise say how close the lightest came.
    ``compared`` says the shipped suites were priced against the refused
    one at all; when they were and none is lighter, the refusal says
    that too, rather than falling silent on the suite.
    ``preferred`` is the source's own default when the refused suite is
    the door's default by grid spacing and that source default fits: it
    is named as its own way out, ahead of the rest, because it is the
    suite a reader got at every spacing before the spacing row bound.
    """

    fitting: tuple[str, ...] = ()
    lightest: tuple[str, int] | None = None
    compared: bool = False
    preferred: str | None = None


def _lighter_profiles_than(profile: str | None, source: str,
                           price_bytes, *,
                           budget_bytes: int, domains: int = 1,
                           preferred: str | None = None
                           ) -> LighterProfiles:
    """Shipped suites this source can run that PRICE less than ``profile``
    and fit ``budget_bytes``.

    A suite that is cheaper but still over the budget is not advice.
    Naming one sends the reader straight back into the refusal that
    named it: a small declared card refused a source's default suite at
    4.28 GiB against a 3.75 GiB budget, advised first a suite that
    prices 4.00 GiB at the same layout, and refused that one too.
    ``budget_bytes`` is the budget the refusal compared against, and a
    candidate is kept only when its envelope at the refused layout is
    within it.  That is the comparison that refused, so a suite named
    here is admitted at that layout.

    Ranked by the estimator's own peak envelope at the refused layout --
    ``price_bytes(profile_name) -> int | None`` -- never by a species
    heuristic.  The heuristic ranked by microphysics species and
    radiation presence, and on the 3080 walk it advised three
    legacy-RRTMG suites as "lighter" than the rte-rrtmgp default; the
    calibration runs then MEASURED the advised suite at 5.53 GiB against
    the default's 2.60 at the same 110x88 grid, because the legacy
    call-peak workspace dominates the envelope and the heuristic never
    priced it.  A candidate the pricer cannot price is dropped, not
    guessed at.

    Returned in the order a reader should try them -- cheapest last, so
    the first name is the smallest step down rather than the biggest
    sacrifice.

    Every candidate passes :func:`profile_route_blocker` -- the SAME
    pairing predicate the emission refuses by -- before it is priced,
    asked for the refused layout's own domain count, as the emission
    asks it.  Admissibility used to be checked for one hard-coded source only,
    so the 2.5.0 walk's gfs refusal ranked a RUC-LSM suite FIRST while
    the very same wizard refuses that pairing outright: following the
    printed advice was refused by the door that printed it.

    ``preferred`` is a suite to name first when it passes the same
    tests (the source's own default, when the refused suite is the
    door's default by grid spacing).  It rides in ``fitting`` like any
    other and is also returned alone, so the sentence can say what it is.

    Named in a refusal, never applied: a suite is the operator's choice
    and a wizard that silently downgraded physics to make a number fit
    would be lying about what it emitted.
    """

    if profile is None:
        return LighterProfiles()
    cost = price_bytes(profile)
    if cost is None:
        return LighterProfiles()
    lighter = []
    for candidate in WIZARD_PHYSICS_PROFILES:
        if candidate == profile:
            continue
        try:
            single_domain_runtime_switches(candidate)
        except ValueError:
            continue
        if profile_route_blocker(candidate, source,
                                 domains=domains) is not None:
            continue
        candidate_cost = price_bytes(candidate)
        if candidate_cost is None or candidate_cost >= cost:
            continue
        lighter.append((candidate_cost, candidate))
    if not lighter:
        return LighterProfiles(compared=True)
    lightest_cost, lightest_name = min(lighter)
    fitting = [(candidate_cost, candidate)
               for candidate_cost, candidate in lighter
               if candidate_cost <= budget_bytes]
    names = [name for _cost, name in sorted(fitting, reverse=True)]
    if preferred not in names:
        preferred = None
    if preferred is not None:
        names = [preferred] + [name for name in names if name != preferred]
    return LighterProfiles(
        fitting=tuple(names[:3]),
        lightest=(lightest_name, lightest_cost), compared=True,
        preferred=preferred)


def _minimum_layout_memory_remedy(*, lighter: LighterProfiles,
                                  envelope_bytes: int, free_bytes: int,
                                  budget_bytes: int, source: str,
                                  shallower: str | None) -> str:
    """The ways out of a memory refusal at a ladder's minimum layout.

    Only levers that move this refusal are named.  A lighter suite is
    named only when it fits (:func:`_lighter_profiles_than`); when a
    lighter suite exists and none fits, the sentence says so and how far
    the lightest is over, rather than leaving a reader to find out by
    being refused again.  A shallower ladder is named only when one was
    priced and fits: ``shallower`` is the flags that request it
    (:func:`_ladder_request_flags`), for the deepest ladder that drops
    nests from this one and fits the same budget at its own minimum
    layout.  A single-domain ladder has no nest to drop, and on a nested
    ladder whose root alone is over the budget, dropping nests is
    refused again the same way.  The card is always a way out, and it
    is named with the free memory this suite needs at this layout: the
    budget is free memory less the external margin, so that is the
    envelope plus the margin, measured against what this card presents.
    """

    need_free = envelope_bytes + EXTERNAL_MARGIN_BYTES
    levers = []
    if shallower is not None:
        levers.append(f"a shallower ladder ({shallower} fits at its "
                      f"minimum layout)")
    others = tuple(name for name in lighter.fitting
                   if name != lighter.preferred)
    if lighter.preferred is not None:
        levers.append(f"--source {source}'s own default suite "
                      f"(--physics-profile {lighter.preferred}, which fits)")
    if others:
        levers.append(f"a lighter --physics-profile "
                      f"({', '.join(others)})")
    levers.append(f"a larger card (this suite needs about "
                  f"{need_free / GIB:.2f} GiB free at this layout, and this "
                  f"card presents about {free_bytes / GIB:.2f} GiB)")
    if len(levers) == 1:
        remedy = f"choose {levers[0]}"
    else:
        remedy = f"choose {', '.join(levers[:-1])}, or {levers[-1]}"
    if not lighter.fitting and lighter.lightest is not None:
        name, cost = lighter.lightest
        over = cost - budget_bytes
        remedy = (f"no lighter shipped --physics-profile fits this budget "
                  f"either: the lightest {source} can run, {name}, needs "
                  f"{cost / GIB:.2f} GiB here, {over / GIB:.2f} GiB over; "
                  + remedy)
    elif not lighter.fitting and lighter.compared:
        remedy = (f"no lighter shipped --physics-profile fits this budget "
                  f"either: none that {source} can run prices below this "
                  f"one here; " + remedy)
    return remedy


def _ladder_request_flags(ratios: tuple[int, ...], root_dx_m: float) -> str:
    """The ``woof domain`` flags that request this ladder.

    A preset is named by ``--ladder``; any other root spacing or chain by
    ``--root-dx`` and, when it has nests, ``--chain``.  A refusal that
    names a ladder as its way out names it in the form the reader types.
    """

    ratios = tuple(int(ratio) for ratio in ratios)
    if float(root_dx_m) == ROOT_DX_M:
        for preset, preset_ratios in LADDER_RATIOS.items():
            if tuple(preset_ratios) == ratios:
                return f"--ladder {preset}"
    flags = f"--root-dx {float(root_dx_m) / 1000.0:g}"
    if ratios:
        flags += " --chain " + ",".join(str(ratio) for ratio in ratios)
    return flags


def _sizing_phases(exp, *, free_bytes: int, machine=None, **kwargs):
    """Price the emitted route against the declared card, including refusals."""
    from woof.core.mynn_pbl_scratch import (
        mynn_pricing_memory, mynn_pricing_total_bytes)

    capacity = kwargs.get("vram_gib")
    if capacity is None:
        return _sizing_phases_for_card(
            exp, free_bytes=free_bytes, machine=machine, **kwargs)
    with mynn_pricing_memory(total_bytes=mynn_pricing_total_bytes(
            capacity, measured=kwargs.get("profile") is not None),
                             free_bytes=free_bytes):
        return _sizing_phases_for_card(
            exp, free_bytes=free_bytes, machine=machine, **kwargs)


def _sizing_phases_for_card(exp, *, free_bytes: int, machine=None, **kwargs):
    if kwargs.get("forcing_interval_seconds") is not None:
        kwargs["ingest_forcing_interval_seconds"] = kwargs["forcing_interval_seconds"]
    from woof.core import streaming
    from tilestream.autoplan import CannotPlan

    # THE TABLES THAT GOVERN THIS TREE'S DOMAINS, not the tree-wide one
    # read raw.  A domain carrying its own ``tiles = {...}`` under a
    # tree-wide ``mode = "off"`` returned from here without being decided
    # at all, while every run door decided it on its own table -- one
    # configuration, two answers, the review's arriving first and the
    # door's arriving after the download.  The mode NAMED in a refusal is
    # the one that put this configuration on the tiled road: the
    # tree-wide table where that is enabled, and otherwise the first
    # domain table that is.
    tree_options = getattr(exp, "tiles", None) or streaming.OFF
    governing = [streaming.options_for_domain(dc, tree_options)
                 for dc in exp.domains]
    if not any(entry.enabled for entry in governing):
        return estimate_phases(exp, machine=machine, **kwargs)
    options = (tree_options if tree_options.enabled
               else next(entry for entry in governing if entry.enabled))
    profile = kwargs.get("profile")
    if machine is None:
        machine = streaming.planner_machine(
            vram_bytes=free_bytes, name="woof domain budget",
            device_profile=profile)
    if machine is None:
        raise DomainFitError(
            "--tiles needs host RAM available to the shared planner; "
            "run the wizard on the forecast host or use --tiles off")
    if profile is not None and getattr(machine, "device_profile", None) is None:
        # THE CALLER'S MEASURED CARD, ON THE SHARED ADMISSION.  The
        # admission (:func:`woof.core.preflight.admission_estimate`, for
        # one domain and for a tree alike) takes its device term from the
        # MACHINE and from nowhere else, on purpose: an optional second
        # way in is what let the review and the run door price the same
        # configuration against two different cards.  A caller that
        # measured the card and handed it here beside a machine built
        # without it therefore has to fold it in before asking, or the
        # shared question is answered against the 170-SM reference.
        # MEASURED on the shipped RTX 3080 profile at 6.54 GiB free,
        # budget 6,485,400,616 bytes: 7,091,619,592 bytes with the bare
        # machine against 5,602,673,416 with the profile carried -- the
        # first is above the budget and the second is below it, so the
        # wizard refused a domain that fits.
        from dataclasses import replace as _replace

        machine = _replace(machine, device_profile=profile)
    # Build the same configured, device-profiled estimate used by the phase
    # gate before asking whether auto can stay resident. Falling back inside
    # decide() discards this caller's measured card and prices a 68-SM 3080
    # against the 170-SM reference before the correct phase estimate can run.
    phases = estimate_phases(exp, machine=machine, **kwargs)
    decision = None
    if len(exp.domains) == 1:
        try:
            # The SHARED admission, not this report's own forecast term:
            # the run door asks the same function of the same
            # configuration, and a review that admitted a domain the door
            # then refused is the defect
            # woof.core.streaming.cold_single_domain_decision documents.
            decision = streaming.cold_single_domain_decision(
                exp, machine=machine, source=kwargs.get("source"))
        except (streaming.StreamingRefused, CannotPlan) as error:
            raise DomainFitError(f"--tiles {options.mode}: {error}",
                                 resource=getattr(error, "resource", None),
                                 phases=phases) from error
    if len(exp.domains) > 1:
        road = phases.tree_road
        if road is None or not road.priced or road.refusal:
            reason = ((road.refusal or getattr(road, "report_error", None))
                      if road is not None else None)
            raise DomainFitError(
                f"--tiles {options.mode}: "
                f"{reason or 'the shared planner could not price this tree'}",
                resource=getattr(road, "refusal_resource", None), phases=phases)
    elif decision.stream and phases.streamed is None:
        raise DomainFitError(
            f"--tiles {options.mode}: the shared planner could not price this domain")
    # THE STREAMED FORECAST'S HOST RAM, on the admission ``woof go``
    # refuses with.  The planner above weighs the pinned store and arena
    # alone; the run also holds the lateral-boundary series beside them, so
    # a layout whose store just fit was emitted here and refused by go
    # before the download.  Typed as a host failure, so the fit shrinks
    # the layout rather than stopping.
    refusal = phases.streamed_host_refusal()
    if refusal is not None:
        raise DomainFitError(f"--tiles {options.mode}: {refusal}",
                             resource="host", phases=phases)
    # THE INGEST TERM ON THE CPU ROAD.  A [tiles] declaration prepares a
    # CPU-prepared source on the host, so ``ingest_envelope_bytes`` is zero
    # and the device comparison above never sees the preparation at all.
    # Its working set is host RAM instead, and it is weighed here against
    # the same machine the planner was handed.  Unweighed, a card larger
    # than the host's RAM sizes a domain whose preparation cannot be held,
    # and the preparation dies after the download.  The wall is the
    # preparation's floor (what it certainly holds at once); the search
    # below steers on its estimated peak.  Typed as a host failure, so the
    # fit shrinks the layout rather than stopping.
    refusal = phases.host_preparation_refusal()
    if refusal is not None:
        raise DomainFitError(f"--tiles {options.mode}: {refusal}",
                             resource="host", phases=phases)
    return phases


def _host_preparation_over_fit_target(phases) -> str | None:
    """Why a candidate's CPU preparation leaves the host no fit headroom.

    The search keeps the same headroom off the host's RAM that it keeps
    off the card's budget (:func:`fit_headroom_bytes`), weighed with the
    preparation's estimated peak
    (:attr:`PhaseMemoryEstimate.host_preparation_bytes`), so a fitted
    layout does not land on the host wall either.  Steering only: the wall
    itself is refused on the preparation's floor by
    :meth:`PhaseMemoryEstimate.host_preparation_refusal`, and a minimum
    layout that fits the wall is still offered.
    """

    host = getattr(phases, "host_ram_bytes", None)
    need = getattr(phases, "host_preparation_bytes", 0)
    if host is None or not need:
        return None
    target = host - fit_headroom_bytes(host)
    if need <= target:
        return None
    return (f"CPU preparation holds {need} bytes of host RAM, over the "
            f"{target} byte host fit target (including headroom)")


def _exhausted_point_bound_remedy(scope: str) -> str:
    """What moves when EVERY rung of a bounded ladder is past a bound the
    REQUEST itself carries.

    The bisecting road cannot end here: it shrinks to its minimum layout
    and the post-fit guard (:func:`_pole_clearance_refusal`) names the
    remedy there.  The bounded road runs out of rungs instead, and said
    only which bound rejected the last one -- a refusal naming no way
    out, on the one road a ``--point`` door now takes.
    """

    if scope == POINT_FIT_PROJECTION_SCOPE:
        return ("every rung of this ladder reaches it, so no card and no "
                "smaller layout clears the pole: the centre is what "
                "moves -- request a point further from it")
    if scope == POINT_FIT_BAND_SCOPE:
        return ("every rung of this ladder runs more than once around the "
                "globe, so no card makes one fit: a finer root spacing is "
                "what moves")
    # Only the cyclone door takes this road, and it sizes to the default
    # extent: it has no --point-extent-km to raise.
    return ("every rung of this ladder is past it, so no card buys more "
            "ground here: draw the ground you want with --polygon")


def _warn_source_stop(root, reason) -> None:
    """A fit stopped by the SOURCE rather than by the card, said once.

    Both roads through :func:`fit_ladder` can end this way and both owe
    the same warning: a source-shaped ceiling (a coverage window, or a
    forcing box too wide to be fetched as one crop) stops the search
    below what the card affords, and from the outside that is
    indistinguishable from a comfortable fit -- the sizing line prints an
    envelope well under budget and says nothing about why the grid is not
    larger.  It lived inside the bisection, so the bounded largest-first
    road filled the same ``stop_out`` while saying nothing on stderr, and
    a bounded-road caller stopped by a coverage window would have been
    told nothing at all.

    The REQUEST-shaped bounds (:func:`point_request_bound`) do not come
    through here on either road; they are the ordinary sizing of a
    request that carries no extent and are reported as plan summary
    (:func:`point_fit_cap_note`), not as a warning.
    """

    warn(f"domain search stopped at {root[0]}x{root[1]} on the SOURCE, "
         "not the card: a larger card buys no more grid here",
         why="The fit is bounded by every constraint, not only memory."
             f"  The next larger layout was rejected because {reason}")


def fit_ladder(*, ladder: str | None = None, free_bytes: int, hours: int,
               start_time: datetime, projection: dict, source: str,
               name: str, ratios: tuple[int, ...] | None = None,
               root_dx_m: float = ROOT_DX_M,
               profile: str | None = DEFAULT_PHYSICS_PROFILE,
               cumulus_requested: bool = False,
               vram_gib: float | None = None,
               device_profile=None, target_machine=None,
               nz: int | None = None, tiles: str | None = None,
               forcing_interval_seconds: float | None = None,
               forcing_intervals: int | None = None,
               history_interval_s: float | None = None,
               nest_history_interval_s: float | None = None,
               acknowledgements: tuple[str, ...] = (),
               candidate_builder=None,
               clearance_rows: int = _CLEARANCE_ROWS,
               minimum_axis: int = 1,
               dimensions_builder=None,
               layout_label: str | None = None,
               stop_out: dict | None = None,
               candidate_scales: tuple[float, ...] | None = None,
               cancelled=None,
               physics_mix: dict | None = None,
               clock: str = "fixed",
               point_extent_km: float = POINT_FIT_MAX_EXTENT_KM,
               profile_at=None,
               noah_mosaic_options=None,
               ) -> tuple[list[tuple[int, int]], ExperimentConfig]:
    """Largest centered layout whose peak envelope fits the budget, with
    headroom left over.

    ``profile_at``, when given, maps a ladder's ratios to the suite that
    ladder binds.  The door passes it when the suite is its default, which
    is keyed on grid spacing (:data:`woof.physics_menu.SPACING_DEFAULTS`),
    so a shallower ladder offered as a way out of a memory refusal is
    priced with the suite it would run rather than this ladder's.

    ``point_extent_km`` is the largest root extent the point request is
    sized to (``--point-extent-km``), on the longer root axis (x); the
    root keeps the 5:4 shape of :func:`_dims_for_scale`, so y is 0.8 of
    it.  See :func:`point_request_bound`.

    ``device_profile`` is the CARD the non-pool terms are priced against
    -- the one this machine MEASURED when no ``--card``/``--vram-gib``
    was declared, and ``None`` (the conservative reference) when the
    caller is sizing for a machine that is somewhere else.  It used to be
    ``None`` unconditionally, so a wizard that had just measured a 68-SM
    RTX 3080 priced that card's local-memory backing store on the 170-SM
    reference profile: 1.49 GiB of another card's shader count, on the
    one term shrinking the grid cannot move, while ``woof check`` on the
    same box used the live profile and disagreed (task 206, open task
    #162's mechanism).

    Bisects a continuous scale factor; every candidate is validated by the
    real experiment loader (clearance, cadence, ratio rules) and priced by
    the real estimator -- the wizard owns no memory arithmetic of its own.
    A custom ``--root-dx``/``--chain`` goes through this same loop, so a
    hand-specified ladder is validated and sized exactly like a preset.

    A template may supply dimensions_builder for its existing parent tree.
    The same hosting search, complete candidate validation, source bounds,
    and phase budget apply; the callback changes only proposed grid sizes.

    ``candidate_scales`` opts into a bounded, largest-first search instead
    of bisection. At most 64 strictly decreasing positive scales are allowed;
    the first fully admitted candidate wins. This avoids assuming monotonic
    tiling/host admission. It uses the SAME candidate, coverage, REQUEST
    and headroom checks, and retries only explicitly typed memory refusals.
    ``cancelled`` is a nonblocking predicate checked around expensive
    candidate operations.

    The request bounds are not optional on that road.  Both searches serve
    ``--point``, which carries no extent, so both have to be bounded by
    what a point may ask for (:func:`point_request_bound`) and both report
    what stopped them the same way, through ``stop_out``: one plain line of
    plan summary on stdout (:func:`point_fit_cap_note`), never a stderr
    warning, because a cap that fires on the ordinary request is not an
    abnormality.  A bounded search that skipped them would have grown a
    high-latitude cyclone request into the projection pole on a large card
    while the bisecting search shrank away from it -- the same door
    answering the same question two ways.

    ``candidate_builder(dims)`` optionally supplies a complete, already
    validated experiment from an editable template. It replaces canned
    rendering BEFORE every price call; no profile substitution is attempted.
    Template-owned boundary/stencil requirements travel with the geometry.

    Takes FREE VRAM, not a budget: the reserve is a property of the
    candidate experiment (see :func:`sizing_budget_bytes`), so it cannot
    be computed before the candidate exists.  And it stops short of the
    budget by :func:`fit_headroom_bytes` -- a config that exactly touches
    its budget is a config with nothing left for the machine to be
    slightly less generous than the model.

    ``stop_out``, when given, is filled with ``{"scope", "reason"}`` for
    the bound that actually stopped the search -- the one the tightest
    rejected candidate crossed -- and left EMPTY when memory stopped it.
    It exists so a caller can attribute the stall exactly rather than
    guessing at it: the first version of the point cap re-priced the
    emitted root one cell larger per axis and read the bound off that
    neighbour, which misattributes a fit that stopped one discretisation
    step under a cap.  The search already knows the answer; this hands
    it over instead of reconstructing it.
    """
    if stop_out is not None:
        stop_out.clear()
    if (ladder is None) == (ratios is None):
        raise ValueError("fit_ladder takes exactly one of ladder / ratios")
    if (isinstance(point_extent_km, bool)
            or not isinstance(point_extent_km, (int, float))
            or not math.isfinite(point_extent_km) or point_extent_km <= 0):
        raise ValueError("point_extent_km must be a finite positive number "
                         f"of kilometres, got {point_extent_km!r}")
    if ratios is None:
        ratios = LADDER_RATIOS[ladder]
    label = layout_label or (ladder if ladder is not None else "-".join(
        f"{v:g}" for v in _ladder_dx_km(ratios, root_dx_m)))
    interval = (source_forcing_interval_seconds(source)
                if forcing_interval_seconds is None else forcing_interval_seconds)

    if candidate_scales is not None:
        if (not isinstance(candidate_scales, tuple)
                or not 1 <= len(candidate_scales) <= 64
                or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                       for v in candidate_scales)
                or any(a <= b for a, b in zip(candidate_scales, candidate_scales[1:]))):
            raise ValueError("candidate_scales must be a tuple of 1..64 decreasing positive finite scales")

    def candidate(scale: float):
        check_fit_cancelled(cancelled)
        dims = (_dims_for_scale(scale, ratios, clearance_rows=clearance_rows)
                if dimensions_builder is None else dimensions_builder(scale))
        if candidate_builder is not None:
            exp = candidate_builder(dims)
        else:
            text = render_config(
                noah_mosaic_options=noah_mosaic_options,
                name=name, start_time=start_time, hours=hours,
                projection=projection, dims=dims, ratios=ratios,
                fetch_hints=_candidate_fetch_hints(source), case_data=None,
                root_dx_m=root_dx_m, profile=profile,
                # The candidate is the same file the user will get, so it
                # carries the same declaration -- otherwise the sizing loop
                # would refuse a layout the emission is allowed to write.
                # The cumulus decision rides along for the same reason: a
                # retired scheme is a kernel set the envelope no longer
                # prices, and sizing against one the file will not carry
                # would fit a smaller domain than the card can hold.
                cumulus_requested=cumulus_requested,
                acknowledgements=acknowledgements, nz=nz, tiles=tiles,
                history_interval_s=history_interval_s,
                nest_history_interval_s=nest_history_interval_s,
                physics_mix=physics_mix, clock=clock)
            exp = experiment_from_text(text, source=f"<candidate {label}>")
        check_fit_cancelled(cancelled)
        # Every PHASE, not just the forecast.  Sizing a domain against the
        # forecast alone is what let this wizard hand a user a config that
        # fit their card, take a multi-gigabyte download, and then OOM in
        # preprocessing -- the phase it had never priced.
        phases = _sizing_phases(
            exp, machine=target_machine, forcing_intervals=forcing_intervals,
            free_bytes=free_bytes, source=source,
            forcing_interval_seconds=interval,
            vram_gib=vram_gib, profile=device_profile)
        budget = sizing_budget_bytes(
            exp, free_bytes=free_bytes, vram_gib=vram_gib,
            forcing_interval_seconds=interval, profile=device_profile)
        check_fit_cancelled(cancelled)
        return dims, exp, phases.peak_envelope_bytes, budget, phases

    def uncovered(exp, dims) -> str | None:
        """Why the SOURCE cannot force this layout, if it cannot.

        The budget is not the only constraint on how large a domain may
        be.  A regional source's native grid is finite, so a card-filling
        ladder near its edge is a legal, well-sized experiment that no
        fetch of that source can force.  Sizing against VRAM alone
        produced exactly that on a 24 GiB card: a 3 km root whose halo
        ran nine rows off the top of the HRRR grid, discovered by the
        root preparation after the download.

        THREE checks, and the order matters.  HRRR's certified route
        knows more than the grid rectangle -- the interpolation stencil
        needs real source cells outside the target on every side -- so
        its own refusal runs first and is the stricter one.
        Every other regional source is bounded by its declared window
        (:func:`source_coverage_refusal`), which is what turned ICON-EU
        over a central-US domain from a preparation traceback into a
        sizing bound this loop can shrink against.  Last, and for every
        source including the global ones, the fitted root has to be
        servable as ONE crop (:func:`fetch_crop_refusal`) -- the bound
        that stops a card-filling Linux layout from being sized and then
        refused by its own emission.
        """
        if source == "hrrr":
            try:
                refusal = coverage_refusal(exp)
            except (HrrrRouteInputError, ValueError) as error:
                # Not a coverage answer: the spec itself cannot be built.
                # Shrinking the ladder cannot fix that, so it refuses with
                # its OWN cause and remedy instead of masquerading as an
                # off-grid polygon.
                raise DomainFitError(str(error)) from None
            if refusal is not None:
                return (f"{refusal}.  Move --point away from the edge of "
                        f"the {source.upper()} grid, or choose a source "
                        "whose coverage includes it")
        else:
            refusal = source_coverage_refusal(
                projection, dims[0][0], dims[0][1], source=source,
                root_dx_m=root_dx_m)
            if refusal is not None:
                return refusal
        # Last, and only when the source reaches the domain at all: can
        # ONE crop of it be fetched?  Asked last because it is the only
        # one of the three that costs a fresh grid evaluation on the
        # global sources, which have no window to answer from.
        #
        # A pole-containing footprint is not this bound's business:
        # such a footprint spans every longitude, so the crop bound is
        # true of it and useless, and its remedy ("size a narrower
        # domain") would send the reader after the wrong flag.  The
        # search never offers one -- `over_extent` shrinks away from the
        # pole first -- and one that survives to here came from the
        # minimum layout, where the post-fit refusal
        # (:func:`_pole_clearance_refusal`) is the right answer.
        if _footprint_contains_pole(
                projection, dims[0][0], dims[0][1], root_dx_m):
            return None
        return fetch_crop_refusal(
            projection, dims[0][0], dims[0][1], source=source,
            root_dx_m=root_dx_m)

    def over_extent(dims) -> tuple[str, str] | None:
        """This layout against the bounds a POINT request carries.

        The bounds themselves live at module level
        (:func:`point_request_bound`) because the advisory printed after
        the fit has to know which of them stopped the search before it
        can name a flag that still moves the answer.
        """

        return point_request_bound(projection, dims[0][0], dims[0][1],
                                   root_dx_m, point_extent_km)

    if candidate_scales is not None:
        # A bounded, largest-first search over an authored ladder of
        # scales, bounded by EXACTLY what the bisection below is bounded
        # by.  The REQUEST's own bounds are asked before the source's for
        # the same reason they are asked first there: a layout past them
        # is not a question about the source at all, and a pole-wrapped
        # footprint would otherwise reach `uncovered`, which hands that
        # case straight back with no bound.
        #
        # What stopped the search leaves through `stop_out`, not
        # re-derived downstream, so a point request that this door sizes
        # is cap-bound and fit-bound in one answer and the plan summary
        # states it ONCE, on stdout (:func:`point_fit_cap_note`).  A
        # memory rejection clears the carried bound because memory is
        # visible from the outside -- the sizing line prints the envelope
        # against the budget -- and claiming a cap that did not bind is
        # the misattribution `point_request_bound` documents.
        last_error = None
        last_bound: str | None = None
        binding: tuple[str, str] | None = None
        for scale in candidate_scales:
            try:
                dims, exp, envelope, budget, phases = candidate(scale)
            except DomainFitError as error:
                check_fit_cancelled(cancelled)
                if error.resource not in {"vram", "host", "memory"}:
                    raise
                last_error = error
                binding = last_bound = None
                continue
            target = budget - fit_headroom_bytes(budget)
            if envelope > target:
                last_error = DomainFitError(
                    f"{dims}: peak {envelope} bytes exceeds the {target} byte "
                    "fit target (including headroom)", resource="vram")
                binding = last_bound = None
                continue
            host_short = _host_preparation_over_fit_target(phases)
            if host_short is not None:
                last_error = DomainFitError(f"{dims}: {host_short}",
                                            resource="host")
                binding = last_bound = None
                continue
            bounded = over_extent(dims)
            scope, reason = bounded if bounded else ("SOURCE",
                                                     uncovered(exp, dims))
            check_fit_cancelled(cancelled)
            if reason is None:
                if binding is not None:
                    if stop_out is not None:
                        stop_out["scope"], stop_out["reason"] = binding
                    # The same warning the bisection owes, on the road
                    # that was silently exempt from it.
                    if binding[0] == "SOURCE":
                        _warn_source_stop(dims[0], binding[1])
                return dims, exp
            binding = (scope, reason)
            last_bound = scope if bounded else None
            last_error = DomainFitError(
                reason, resource="extent" if bounded else "coverage")
        check_fit_cancelled(cancelled)
        # A ladder exhausted against a REQUEST bound has a way out, and it
        # is not the card: every rung is bounded the same way, so the
        # sentence names what actually moves.  A coverage stop already
        # carries its own remedy (:func:`source_coverage_refusal`), and a
        # memory stop is the card, which the envelope numbers state.
        raise DomainFitError(
            f"No candidate in the bounded {label} search fits: {last_error}"
            + ("" if last_bound is None
               else "; " + _exhausted_point_bound_remedy(last_bound)),
            resource=last_error.resource, phases=last_error.phases,
            budget_bytes=last_error.budget_bytes) from last_error

    # The MINIMUM layout is a property of the ladder, not a constant: a
    # chain deeper than any preset needs a larger root before its
    # innermost nest has any interior at all (:func:`_min_hosting_scale`).
    min_scale = _min_hosting_scale(ratios, clearance_rows=clearance_rows,
                                   minimum_axis=minimum_axis,
                                   dimensions_builder=dimensions_builder)
    dims, exp, envelope, budget, _phases = candidate(min_scale)
    #: The part of the envelope no grid can move: this suite's CUDA
    #: context, the local-memory backing store of its kernel set, and the
    #: measured residue.  When THAT alone is the whole card there is
    #: nothing to size -- every layout on every ladder is refused for the
    #: same reason, and the fit loop's per-layout refusal would name a
    #: grid the reader cannot usefully shrink.
    #:
    #: This used to be spelled ``budget <= 0``, which worked only while
    #: the budget subtracted the whole allocation reserve.  It no longer
    #: does (that reserve carries the same non-pool bytes the envelope
    #: carries, and charging both is the double count task 206 removed),
    #: so the floor is asked directly instead of inferred from a
    #: subtraction that has stopped containing it.
    floor_estimate = _phases.forecast
    grid_independent = (floor_estimate.envelope_intercept_bytes
                        + ENVELOPE_UNMODELLED_BYTES)
    min_dims = dims

    # Candidate suites are PRICED at this exact minimum layout by the same
    # estimator that refuses, so a suite whose envelope is larger (the
    # legacy-RRTMG call-peak workspace measured 2.1x the rte-rrtmgp
    # default on the 3080) can never be advised, and neither can one
    # that is cheaper but still over the budget.  Scheme choices the
    # request carries ride along: following the advice keeps them, with
    # the named suite as the base they change (:func:`physics_mix_request`
    # names the suite the request names).
    def _price(candidate_profile: str) -> int | None:
        # Muted: this loads a file the reader will not get, and its
        # loader warnings are about that file, not the one written.
        with muted_warnings():
            return _price_loud(candidate_profile)

    def _price_loud(candidate_profile: str) -> int | None:
        candidate_mix = (None if not physics_mix
                         else {**physics_mix, "suite": candidate_profile})
        try:
            candidate_text = render_config(
                noah_mosaic_options=noah_mosaic_options,
                name=name, start_time=start_time, hours=hours,
                projection=projection, dims=min_dims, ratios=ratios,
                fetch_hints=_candidate_fetch_hints(source),
                case_data=None, root_dx_m=root_dx_m,
                profile=candidate_profile,
                # Pricing a suite the user would have to NAME to get,
                # so it is priced as a named suite: verbatim.
                cumulus_requested=True,
                acknowledgements=acknowledgements, nz=nz, tiles=tiles,
                history_interval_s=history_interval_s,
                nest_history_interval_s=nest_history_interval_s,
                physics_mix=candidate_mix, clock=clock)
            candidate_exp = experiment_from_text(
                candidate_text, source=f"<candidate {label} "
                                       f"{candidate_profile}>")
            return _sizing_phases(
                candidate_exp, machine=target_machine,
                forcing_intervals=forcing_intervals,
                free_bytes=free_bytes, source=source,
                forcing_interval_seconds=interval,
                vram_gib=vram_gib,
                profile=device_profile).peak_envelope_bytes
        except Exception:
            return None

    def _lighter_that_fit() -> LighterProfiles:
        # A template is scientific authority, not a named suite we may
        # replace; and with no budget at all nothing can fit it.
        if candidate_builder is not None or budget <= 0:
            return LighterProfiles()
        # The suite is the door's default by grid spacing exactly when
        # ``profile_at`` is given, and then the source's own default is
        # the first way out to name when it fits.
        preferred = None
        if profile_at is not None:
            from woof.physics_menu import default_profile_for

            preferred = default_profile_for(source)
        return _lighter_profiles_than(profile, source, _price,
                                      budget_bytes=budget,
                                      domains=len(ratios) + 1,
                                      preferred=preferred)

    def _shallower_that_fits() -> str | None:
        # Dropping nests is a way out only when the shallower ladder's
        # own minimum layout fits the budget, priced by the same
        # estimator with the same suite, and the source still forces it:
        # the two checks that ladder's own fit makes before it can
        # return its minimum layout.  A template's tree is its own, so
        # it has no shallower ladder this loop can price.
        if (candidate_builder is not None or dimensions_builder is not None
                or budget <= 0):
            return None
        # Muted for the reason _price is: each shallower ladder is loaded
        # only to price it.
        with muted_warnings():
            return _shallower_that_fits_loud()

    def _shallower_that_fits_loud() -> str | None:
        for depth in range(len(ratios) - 1, -1, -1):
            shallower = tuple(ratios[:depth])
            shallower_profile = (profile if profile_at is None
                                 else profile_at(shallower))
            try:
                shallower_dims = _dims_for_scale(
                    _min_hosting_scale(shallower,
                                       clearance_rows=clearance_rows,
                                       minimum_axis=minimum_axis),
                    shallower, clearance_rows=clearance_rows)
                shallower_label = "-".join(
                    f"{v:g}" for v in _ladder_dx_km(shallower, root_dx_m))
                shallower_text = render_config(
                    noah_mosaic_options=noah_mosaic_options,
                    name=name, start_time=start_time, hours=hours,
                    projection=projection, dims=shallower_dims,
                    ratios=shallower,
                    fetch_hints=_candidate_fetch_hints(source),
                    case_data=None, root_dx_m=root_dx_m,
                    profile=shallower_profile,
                    cumulus_requested=cumulus_requested,
                    acknowledgements=acknowledgements, nz=nz, tiles=tiles,
                    history_interval_s=history_interval_s,
                    nest_history_interval_s=nest_history_interval_s,
                    physics_mix=physics_mix, clock=clock)
                shallower_exp = experiment_from_text(
                    shallower_text,
                    source=f"<candidate {shallower_label} below {label}>")
                shallower_envelope = _sizing_phases(
                    shallower_exp, machine=target_machine,
                    forcing_intervals=forcing_intervals,
                    free_bytes=free_bytes, source=source,
                    forcing_interval_seconds=interval,
                    vram_gib=vram_gib,
                    profile=device_profile).peak_envelope_bytes
                if shallower_envelope > budget:
                    continue
                if uncovered(shallower_exp, shallower_dims) is not None:
                    continue
            except Exception:
                continue
            return _ladder_request_flags(shallower, root_dx_m)
        return None

    if budget <= 0 or grid_independent >= budget:
        # No smaller layout on this ladder helps here.  Nor on any ladder
        # running this suite -- but a shallower ladder that binds a
        # lighter default suite (``profile_at``) is a way out, and it is
        # named only when its own minimum layout fits.
        shallower = (_shallower_that_fits() if profile_at is not None
                     else None)
        remedy = _minimum_layout_memory_remedy(
            lighter=_lighter_that_fit(), envelope_bytes=envelope,
            free_bytes=free_bytes, budget_bytes=budget, source=source,
            shallower=shallower)
        raise DomainFitError(
            f"this card has no budget for ladder {label} at all: the "
            f"suite's grid-independent envelope (CUDA context + the "
            f"local-memory backing store of its kernel set + the "
            f"measured unmodelled residue) is "
            f"{grid_independent / GIB:.2f} GiB, and with the "
            f"{EXTERNAL_MARGIN_BYTES / GIB:.2f} GiB external margin that "
            f"is already the whole of about "
            f"{free_bytes / GIB:.2f} GiB free -- before the grid asks for "
            f"a single byte, so no smaller layout "
            + ("on any ladder running this suite" if shallower is not None
               else "on any ladder") + f" can help; {remedy}")
    smallest_uncovered = uncovered(exp, dims)
    if smallest_uncovered is not None:
        raise DomainFitError(
            f"ladder {label} cannot be forced by {source} even at the "
            f"minimum layout ({dims[0][0]}x{dims[0][1]} root): "
            f"{smallest_uncovered}")
    if envelope > budget:
        # Say WHY it does not fit.  "your card is too small" is what the
        # bare number reads as, and at the minimum layout the accurate
        # answer is usually that the grid-independent terms dominate --
        # but ONLY when they actually do.  The old wording asserted
        # "so a smaller grid cannot help" beside a printed 0%, which is
        # a sentence contradicting the number in front of it.
        floor = _phases.forecast
        constants = (floor.envelope_intercept_bytes
                     + ENVELOPE_UNMODELLED_BYTES)
        share = (100.0 * constants / envelope if envelope else 0.0)
        if share >= 25.0:
            why = ("is grid-independent (CUDA context, the local-memory "
                   "backing store of the selected kernel set, and the "
                   "measured unmodelled residue), so shrinking the grid "
                   "moves only the rest")
        else:
            why = ("is grid-independent; the grid itself is most of this "
                   "layout, and this IS the smallest layout the ladder "
                   "has")
        detail = (
            f"the model itself wants {floor.alloc_estimate_bytes / GIB:.2f} "
            f"GiB at this layout; the other "
            f"{constants / GIB:.2f} GiB ({share:.0f}% of the envelope) "
            f"{why}.  This is already the minimum layout, so there is no "
            "smaller grid on this ladder to fall back to")
        phases = _sizing_phases(
            exp, machine=target_machine, forcing_intervals=forcing_intervals,
            free_bytes=free_bytes, source=source,
            forcing_interval_seconds=interval,
            vram_gib=vram_gib, profile=device_profile)
        # Name the THIRD lever too.  A large share of the envelope is the
        # selected kernel set's own local-memory backing store, so the
        # suite is often the cheapest thing to change -- and after 1.8
        # gave every source a full-radiation default, it is the lever a
        # small card most often needs.  Omitting it read as "your card is
        # too small" when a lighter shipped profile fits the same grid.
        # Only suites and shallower ladders that fit THIS budget are
        # named.
        remedy = _minimum_layout_memory_remedy(
            lighter=_lighter_that_fit(), envelope_bytes=envelope,
            free_bytes=free_bytes, budget_bytes=budget, source=source,
            shallower=_shallower_that_fits())
        raise DomainFitError(
            f"ladder {label} does not fit a {budget / GIB:.1f} GiB "
            f"budget even at the minimum layout ({dims[0][0]}x{dims[0][1]} "
            f"root): {phases.verdict(budget)}.  {detail}; {remedy}",
            phases=phases, budget_bytes=budget)
    # A requested extent below the smallest root this ladder hosts.  The
    # extent bound is monotone in scale, so every layout the bisection
    # could offer is past it too, and the answer is this minimum layout.
    # It used to fall through: the bisection rejected every candidate on
    # the extent, kept the minimum as `best`, and reported the EXTENT as
    # the bound that capped it, so `--point-extent-km 50` on the 12 km
    # ladder emitted a 720 km root under a line telling the reader to
    # raise the value for a larger domain.  Returned here with its own
    # scope, the plan summary says what happened instead.  Not a
    # refusal: the root is the closest domain to the request this ladder
    # has, and a larger one than asked breaks nothing.
    floored = over_extent(dims)
    if floored is not None and floored[0] == POINT_FIT_EXTENT_SCOPE:
        if stop_out is not None:
            nx, ny = dims[0]
            dx_km = float(root_dx_m) / 1000.0
            stop_out["scope"] = POINT_FIT_FLOOR_SCOPE
            stop_out["reason"] = (
                f"the smallest root ladder {label} hosts is {nx} x {ny} "
                f"at {dx_km:g} km, {max(nx, ny) * dx_km:.0f} km across, "
                f"larger than the {float(point_extent_km):g} km the "
                "request asked for")
        return dims, exp
    lo, hi = min_scale, _MAX_SCALE
    best = (dims, exp)
    # WHY the search stopped where it did, kept as it happens.  A memory
    # bound needs no explanation -- the sizing line prints the envelope
    # against the budget -- but a NON-memory bound is invisible from the
    # outside, and an invisible saturation is the exact defect
    # tests/test_domain_wizard_budget_monotonic.py was written for: a 180
    # GiB card sized like a 64 GiB one and reported a comfortable fit.
    # ``None`` means nothing SOURCE-shaped bound: the card did, or the
    # experiment loader refused the layout on its own terms.
    binding_reason: str | None = None
    # Which bound spoke, for the announcement below: the SOURCE's own
    # coverage, the extent a point request is sized to, or the
    # PROJECTION's polar envelope.
    binding_scope = "SOURCE"
    for _ in range(36):
        mid = 0.5 * (lo + hi)
        try:
            dims, exp, envelope, budget, phases = candidate(mid)
        except DomainFitError:
            # A layout the experiment loader itself refuses -- neither
            # the card nor the source, so the sentence below would name
            # the wrong thing.  Claim nothing.
            hi = mid
            binding_reason = None
            continue
        target = budget - fit_headroom_bytes(budget)
        if (envelope > target
                or _host_preparation_over_fit_target(phases) is not None):
            hi = mid
            binding_reason = None
            continue
        # The REQUEST's own bounds first: a layout past them is not a
        # question about the source at all, and a pole-wrapped
        # footprint would otherwise reach `uncovered` -- which hands
        # that case straight back with no bound (see its closing
        # paragraph) and lets the search keep growing.
        bounded = over_extent(dims)
        scope, reason = bounded if bounded else ("SOURCE",
                                                 uncovered(exp, dims))
        if reason is None:
            best = (dims, exp)
            lo = mid
        else:
            hi = mid
            binding_reason, binding_scope = reason, scope
    # A search that converges on its own upper bracket did not find the grid
    # the budget affords -- it found the largest grid it was willing to look
    # at.  Those are different answers and they used to be indistinguishable
    # from the outside, which is how an 8.0 ceiling sized 64, 96 and 180 GiB
    # cards identically while every one of them reported a comfortable fit.
    # Saying it costs nothing when the bound does not bind.
    if lo >= _MAX_SCALE * (1.0 - 1e-6):
        root = best[0][0]
        warn(f"domain search reached its scale ceiling at {root[0]}x{root[1]}; "
             "this is the largest layout considered, not necessarily the "
             "largest your budget affords",
             why="The wizard brackets its grid-scale bisection between "
                 f"_MIN_SCALE and _MAX_SCALE (currently {_MAX_SCALE}).  A "
                 "result sitting on the upper bracket means memory never "
                 "became the binding constraint, so a larger card will not "
                 "buy a larger domain until the ceiling is raised.")
        # The BRACKET stopped this search, whatever a candidate above it
        # was rejected for, so there is no binding bound to report to a
        # caller reading `stop_out`.
        binding_reason = None
    elif binding_reason is not None and binding_scope == "SOURCE":
        # The same defect one bound over.  A source-shaped ceiling
        # (coverage window, or a forcing box too wide to be fetched as
        # one crop) stops the search below what the card affords, and
        # from the outside that is indistinguishable from a comfortable
        # fit -- the sizing line prints an envelope well under budget
        # and says nothing about why the grid is not larger.  So the
        # wizard says it, and says that a bigger card is not the lever.
        #
        # The REQUEST-shaped bounds do not come through here.  They are
        # the default sizing of a request that carries no extent, so
        # they bind on the ordinary mid-latitude point on any card from
        # about 16 GiB up -- warning about that is warning about the
        # normal case, and it turned the release suite red on the door's
        # own default emission.  They are reported instead as one plain
        # line of plan summary (:func:`point_fit_cap_note`) built from
        # `stop_out` below, which is stdout and is not a warning.
        _warn_source_stop(best[0][0], binding_reason)
    if stop_out is not None and binding_reason is not None:
        stop_out["scope"] = binding_scope
        stop_out["reason"] = binding_reason
    return best


def _maximum_map_factor(grid, south: float, north: float) -> float:
    """Maximum conformal scale over a latitude interval."""

    latitudes = np.linspace(south, north, 257, dtype=np.float64)
    map_factors = np.abs(np.asarray(grid.map_factor(latitudes), dtype=float))
    if not np.all(np.isfinite(map_factors)) or not map_factors.size \
            or float(map_factors.max()) <= 0.0:
        raise ValueError(
            "the polygon cannot be represented finitely in the "
            f"selected {grid.map_proj} projection")
    return float(map_factors.max())


def _polygon_sample_step_deg(projection: dict,
                             footprint: PolygonFootprint,
                             finest_dx_m: float) -> float:
    """Angular segment step whose projection error fits inside cell slack."""

    grid = _root_grid(projection, 2, 2, finest_dx_m)
    scale = _maximum_map_factor(grid, footprint.south, footprint.north)
    # A lon/lat step has spherical path length no greater than
    # sqrt(2)*R*step radians.  Bound its projected length to one quarter of
    # the finest cell.  The fitter adds a whole cell outside every sample,
    # so every point between samples remains inside that proven envelope.
    step = math.degrees(
        finest_dx_m / (4.0 * math.sqrt(2.0) * EARTH_RADIUS_M * scale))
    return min(_POLYGON_SAMPLE_STEP_DEG, step)


def _buffer_cells(grid, footprint: PolygonFootprint,
                  buffer_km: float) -> float:
    """Conservative projected-cell distance for a ground buffer."""

    if buffer_km == 0.0:
        return 0.0
    buffer_m = float(buffer_km) * 1000.0
    latitude_reach = math.degrees(buffer_m / EARTH_RADIUS_M)
    south = footprint.south - latitude_reach
    north = footprint.north + latitude_reach
    if grid.map_proj != "polar" and (south <= -90.0 or north >= 90.0):
        edge = "south" if south <= -90.0 else "north"
        raise ValueError(
            f"--buffer-km {buffer_km:g} on this footprint reaches the "
            f"{edge} pole on Lambert; use --projection polar "
            "for a buffered domain enclosing a pole")
    # These projections are conformal and their scale depends on latitude.
    # Taking the maximum over the whole buffered latitude range makes a
    # ground-distance buffer no smaller in projected grid cells.
    scale = _maximum_map_factor(grid, max(-90.0, south), min(90.0, north))
    return buffer_m * scale / float(grid.dx)


def _round_up_multiple(value: float, multiple: int) -> int:
    return int(multiple * max(1, math.ceil(value / multiple - 1e-12)))


def polygon_minimum_axis(profile: str | None = DEFAULT_PHYSICS_PROFILE) -> int:
    """Smallest axis for the operations the wizard actually authors.

    Specified-boundary tables must fit, boundary/free-interior health needs a
    unique interior, and selected fifth-order geopotential advection needs its
    seven-point stencil. Nest clearance and tile halos are enforced separately.
    """
    shared = shared_physics(profile)
    stencil = (FIFTH_ORDER_STENCIL_AXIS
               if shared["h_sca_adv_order"] == 5 else 1)
    active_width = max(shared["spec_zone"], shared["relax_zone"])
    return max(stencil, boundary_axis(_SPEC_BDY_WIDTH, interior_points=1),
               boundary_axis(active_width, interior_points=1))


def polygon_ladder_dims(*, footprint: PolygonFootprint,
                        projection: dict, ratios: tuple[int, ...],
                        buffers_km: tuple[float, ...],
                        root_dx_m: float = ROOT_DX_M,
                        profile: str | None = DEFAULT_PHYSICS_PROFILE,
                        root_minimum_axis: int | None = None,
                        minimum_axis: int | None = None,
                        clearance_rows: int = _CLEARANCE_ROWS,
                        ) -> list[tuple[int, int]]:
    """Smallest centered legal ladder containing the buffered footprint."""

    count = len(ratios) + 1
    if len(buffers_km) != count:
        raise ValueError(
            f"polygon_ladder_dims needs {count} buffers, got "
            f"{len(buffers_km)}")
    minimum_axis = (polygon_minimum_axis(profile) if minimum_axis is None
                    else minimum_axis)
    finest_dx_m = float(root_dx_m) / math.prod(ratios)
    sample_step = _polygon_sample_step_deg(
        projection, footprint, finest_dx_m)
    sample_lats, sample_lons = _polygon_samples(
        footprint, max_step_deg=sample_step)
    dimensions: list[tuple[int, int]] = []
    dx = float(root_dx_m)
    for level in range(count):
        if level:
            dx /= ratios[level - 1]
        grid = _root_grid(projection, 2, 2, dx)
        i, j = grid.latlon_to_ij(sample_lats, sample_lons)
        i = np.asarray(i, dtype=float)
        j = np.asarray(j, dtype=float)
        if not np.all(np.isfinite(i)) or not np.all(np.isfinite(j)):
            raise ValueError(
                "the polygon cannot be represented finitely in the "
                f"selected {projection['map_proj']} projection")
        margin = _buffer_cells(grid, footprint, buffers_km[level])
        half_x = (float(np.max(np.abs(i - grid.known_x))) + margin
                  + _POLYGON_FIT_SLACK_CELLS)
        half_y = (float(np.max(np.abs(j - grid.known_y))) + margin
                  + _POLYGON_FIT_SLACK_CELLS)
        quantum = 2 if level == 0 else 2 * ratios[level - 1]
        axis = (max(minimum_axis, root_minimum_axis or 0)
                if level == 0 else minimum_axis)
        rounded_minimum = _round_up_multiple(axis, quantum)
        nx = max(rounded_minimum, _round_up_multiple(2.0 * half_x, quantum))
        ny = max(rounded_minimum, _round_up_multiple(2.0 * half_y, quantum))
        dimensions.append((nx, ny))

    # Every child must also clear its parent's external-boundary and blend
    # rows.  Propagate that requirement from the innermost level outward;
    # this can enlarge an outer level beyond its own geometric buffer, but
    # never makes any requested buffer smaller.
    for level in range(count - 1, 0, -1):
        ratio = ratios[level - 1]
        child_nx, child_ny = dimensions[level]
        parent_nx, parent_ny = dimensions[level - 1]
        parent_quantum = 2 if level == 1 else 2 * ratios[level - 2]
        parent_nx = max(parent_nx, _round_up_multiple(
            child_nx // ratio + 2 * clearance_rows, parent_quantum))
        parent_ny = max(parent_ny, _round_up_multiple(
            child_ny // ratio + 2 * clearance_rows, parent_quantum))
        dimensions[level - 1] = parent_nx, parent_ny
    return dimensions


def verify_polygon_containment(exp: ExperimentConfig,
                               footprint: PolygonFootprint,
                               buffers_km: tuple[float, ...]) -> None:
    """Prove each emitted projected grid contains its requested envelope."""

    from woof.static.projection import grids_from_projection_config

    grids = grids_from_projection_config(exp)
    if len(grids) != len(buffers_km):
        raise DomainFitError(
            "internal polygon fit regression: emitted domain count does not "
            "match the per-level buffer count")
    sample_step = _polygon_sample_step_deg(
        {
            "map_proj": grids[0].map_proj,
            "ref_lat": grids[0].ref_lat,
            "ref_lon": grids[0].ref_lon,
            "truelat1": grids[0].truelat1,
            "truelat2": grids[0].truelat2,
            "stand_lon": grids[0].stand_lon,
        }, footprint, min(float(grid.dx) for grid in grids))
    sample_lats, sample_lons = _polygon_samples(
        footprint, max_step_deg=sample_step)
    tolerance = 1e-8
    for level, (grid, buffer_km) in enumerate(zip(grids, buffers_km), 1):
        i, j = grid.latlon_to_ij(sample_lats, sample_lons)
        i = np.asarray(i, dtype=float)
        j = np.asarray(j, dtype=float)
        margin = _buffer_cells(grid, footprint, buffer_km)
        clearances = (
            float(i.min()) - 0.5,
            (float(grid.e_we) - 0.5) - float(i.max()),
            float(j.min()) - 0.5,
            (float(grid.e_sn) - 0.5) - float(j.max()),
        )
        if not np.all(np.isfinite(clearances)) \
                or min(clearances) + tolerance < margin:
            raise DomainFitError(
                f"internal polygon fit regression: domain d{exp.domains[level - 1].grid_id:02d} "
                f"does not contain the footprint plus its {buffer_km:g} km "
                "buffer; refusing to emit a partial target domain")


def _fit_source_projection(projection: dict, source: str,
                           root_dx_m: float) -> dict:
    """Fit about the projection and cell center exact-source emission uses."""
    from woof.source_adapters import source_coverage_window
    from woof.source_coverage import points_outside
    from woof.static.source_defaults import align_source_projection

    # A point the source does not cover has no source cells to align to,
    # and the coverage refusal below names the source, where the target
    # lands in its index space and the way out.  Aligned first (0eae0f675),
    # such a point met the static source's "N cells off the source's mass
    # points" instead, which names none of them.
    if bool(points_outside(source_coverage_window(source),
                           projection["ref_lat"], projection["ref_lon"])):
        return projection
    # 6a69b356f, lane/hrrr-statics, aligns exact-spacing emission to the
    # source projection and lattice. Root dimensions have an even quantum,
    # so a 2x2 planning window establishes their half-cell center before
    # sizing every level. Final actual-dimension crop admission still runs.
    return align_source_projection(projection, (2, 2), root_dx_m, source)


def fit_polygon_ladder(*, footprint: PolygonFootprint,
                       buffers_km: tuple[float, ...],
                       free_bytes: int, hours: int,
                       device_profile=None, target_machine=None,
                       nz: int | None = None, tiles: str | None = None,
                       forcing_interval_seconds: float | None = None,
                       forcing_intervals: int | None = None,
                       history_interval_s: float | None = None,
                       nest_history_interval_s: float | None = None,
                       start_time: datetime, projection: dict, source: str,
                       name: str, ratios: tuple[int, ...],
                       root_dx_m: float = ROOT_DX_M,
                       profile: str | None = DEFAULT_PHYSICS_PROFILE,
                       cumulus_requested: bool = False,
                       vram_gib: float | None = None,
                       acknowledgements: tuple[str, ...] = (),
                       candidate_builder=None,
                       minimum_axis: int | None = None,
                       clearance_rows: int = _CLEARANCE_ROWS,
                       dimensions_builder=None,
                       physics_mix: dict | None = None,
                       clock: str = "fixed",
                       noah_mosaic_options=None,
                       ) -> tuple[list[tuple[int, int]], ExperimentConfig]:
    """Fit one polygon-bound ladder, refusing rather than clipping it.

    Takes FREE VRAM for the same reason :func:`fit_ladder` does: the
    reserve carries the local-memory backing store of the SELECTED kernel
    set, so it is a property of the candidate experiment and cannot be
    computed before that candidate exists.  Sizing this path against a
    flat reserve is what let the point path emit configs that failed the
    product's own ``woof check``, and the polygon path priced its layout
    with the identical arithmetic.

    It does NOT take :func:`fit_headroom_bytes` off the budget, and that
    asymmetry with :func:`fit_ladder` is deliberate.  That headroom is a
    property of a BISECTION: the point fitter grows the grid until the
    envelope touches the budget, so without it every emitted ladder
    landed a rounding error from the wall.  Here the polygon and its
    buffers determine the layout outright -- there is no loop to stop
    short in, and spending the headroom would refuse a user's explicit
    footprint that the enforced ``estimate <= budget`` gate accepts,
    which is a narrowing neither this fix nor the polygon route asked
    for.  The refusal below is still the suite-priced budget, so a
    layout this accepts is a layout ``woof check`` accepts, and it names
    the LAYOUT rather than the card: on this route the footprint is the
    fixed thing the user controls.
    """

    target_interior = get_source_adapter(source).root_target_interior_axis
    root_minimum = (None if target_interior is None else boundary_axis(
        _SPEC_BDY_WIDTH, interior_points=target_interior))
    dimension_options = dict(
        footprint=footprint, projection=projection, ratios=ratios,
        buffers_km=buffers_km, root_dx_m=root_dx_m, profile=profile,
        root_minimum_axis=root_minimum, minimum_axis=minimum_axis,
        clearance_rows=clearance_rows)
    dims = (polygon_ladder_dims(**dimension_options) if dimensions_builder is None
            else dimensions_builder(**dimension_options))
    if candidate_builder is not None:
        exp = candidate_builder(dims)
    else:
        text = render_config(
            noah_mosaic_options=noah_mosaic_options,
            name=name, start_time=start_time, hours=hours,
            projection=projection, dims=dims, ratios=ratios,
            fetch_hints=_candidate_fetch_hints(source), case_data=None,
            root_dx_m=root_dx_m, profile=profile,
            cumulus_requested=cumulus_requested,
            acknowledgements=acknowledgements, nz=nz, tiles=tiles,
            history_interval_s=history_interval_s,
            nest_history_interval_s=nest_history_interval_s,
            physics_mix=physics_mix, clock=clock)
        exp = experiment_from_text(text, source="<polygon candidate>")
    if source == "hrrr":
        try:
            refusal = coverage_refusal(exp)
        except (HrrrRouteInputError, ValueError) as error:
            # A spec-construction failure is its own refusal with its
            # own remedy.  It used to be concatenated into the coverage
            # sentence below, sending the user to move a polygon that
            # was never the problem.
            raise DomainFitError(str(error)) from None
        if refusal is not None:
            raise DomainFitError(
                "polygon plus the requested per-level buffers falls outside "
                f"HRRR coverage: {refusal}.  Move --polygon inside the HRRR "
                "grid or choose a source whose coverage includes it")
    else:
        # Every other regional source is bounded by its declared window.
        # A polygon cannot be shrunk toward coverage the way a fitted
        # ladder can -- the footprint is the user's explicit request --
        # so this is a refusal rather than a sizing bound.
        refusal = source_coverage_refusal(
            projection, dims[0][0], dims[0][1], source=source,
            root_dx_m=root_dx_m, target_option="--polygon")
        if refusal is not None:
            raise DomainFitError(
                "polygon plus the requested per-level buffers falls "
                f"outside {source} coverage: {refusal}")
    interval = (source_forcing_interval_seconds(source)
                if forcing_interval_seconds is None else forcing_interval_seconds)
    phases = _sizing_phases(
        exp, machine=target_machine, forcing_intervals=forcing_intervals,
        free_bytes=free_bytes, source=source,
        forcing_interval_seconds=interval,
        vram_gib=vram_gib, profile=device_profile)
    budget_bytes = sizing_budget_bytes(
        exp, free_bytes=free_bytes, vram_gib=vram_gib,
        forcing_interval_seconds=interval, profile=device_profile)
    if budget_bytes <= 0 or phases.peak_envelope_bytes > budget_bytes:
        layout = ", ".join(
            f"d{domain.grid_id:02d} {nx}x{ny}"
            for domain, (nx, ny) in zip(exp.domains, dims))
        # A non-positive budget is NOT the card-is-too-small case here --
        # that one is layout-independent and already refused in `main` by
        # the CUDA-context-plus-margin floor.  This one is the layout's
        # own doing: the reserve carries a retention fraction of the
        # estimate, so a big enough footprint drives its own reserve past
        # the whole card.  `verdict` would print a negative budget, so
        # say the arithmetic instead -- but keep naming the LAYOUT first
        # either way, because on this route the layout is the fixed thing
        # the user controls and the only thing they can act on.
        if budget_bytes > 0:
            detail = phases.verdict(budget_bytes)
        else:
            detail = (
                "the external margin this card must keep for other "
                f"processes is already "
                f"{(free_bytes - budget_bytes) / GIB:.2f} GiB against "
                f"about {free_bytes / GIB:.2f} GiB free")
        raise DomainFitError(
            "polygon plus the requested per-level buffers requires "
            f"{layout}, but {detail}; reduce the "
            "buffer, choose fewer levels, increase grid spacing, or use a "
            "larger card", phases=phases, budget_bytes=budget_bytes)
    verify_polygon_containment(exp, footprint, buffers_km)
    return dims, exp


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", newline="\n", encoding="utf-8") as stream:
        stream.write(text)
    os.replace(temporary, path)


def render_wps_namelist(projection: dict, dims: list[tuple[int, int]],
                        ratios: tuple[int, ...],
                        root_dx_m: float = ROOT_DX_M,
                        source: str = "era5",
                        forcing_interval_seconds: float | None = None) -> str:
    """Minimal namelist.wps matching the TOML bit-for-bit.

    The config-driven pipeline reads only geog_data_res/max_dom from it,
    but the native-WRF contract checker cross-checks every projection and
    layout key against the [projection]/[[domain]] tables, so the emitted
    pair must agree exactly.

    ``&share/interval_seconds`` uses measured supplied-input cadence when
    available, otherwise the source registry's default. Write an integer:
    the native-WRF contract checker rejects a floating-point value.
    """
    tables = _domain_tables(dims, ratios, root_dx_m=root_dx_m)
    interval_seconds = (source_forcing_interval_seconds(source)
                        if forcing_interval_seconds is None else forcing_interval_seconds)
    if (not math.isfinite(interval_seconds) or interval_seconds <= 0
            or int(interval_seconds) != interval_seconds):
        raise ValueError("WPS forcing interval must be a positive whole number of seconds")
    interval_seconds = int(interval_seconds)

    def csv(values):
        return ", ".join(str(v) for v in values) + ","

    return (
        "&share\n"
        " wrf_core = 'ARW',\n"
        f" max_dom = {len(tables)},\n"
        f" interval_seconds = {interval_seconds},\n"
        " io_form_geogrid = 2,\n"
        "/\n"
        "&geogrid\n"
        f" parent_id         = {csv([1] + [t['parent_id'] for t in tables[1:]])}\n"
        f" parent_grid_ratio = {csv(t['parent_grid_ratio'] for t in tables)}\n"
        f" i_parent_start    = {csv(t['i_parent_start'] for t in tables)}\n"
        f" j_parent_start    = {csv(t['j_parent_start'] for t in tables)}\n"
        f" e_we              = {csv(t['nx'] + 1 for t in tables)}\n"
        f" e_sn              = {csv(t['ny'] + 1 for t in tables)}\n"
        f" geog_data_res     = {csv(chr(39) + 'default' + chr(39) for _ in tables)}\n"
        f" dx = {float(root_dx_m):g},\n"
        f" dy = {float(root_dx_m):g},\n"
        f" map_proj = '{projection['map_proj']}',\n"
        f" ref_lat   = {_namelist_number(projection['ref_lat'])},\n"
        f" ref_lon   = {_namelist_number(projection['ref_lon'])},\n"
        f" truelat1  = {_namelist_number(projection['truelat1'])},\n"
        f" truelat2  = {_namelist_number(projection['truelat2'])},\n"
        f" stand_lon = {_namelist_number(projection['stand_lon'])},\n"
        "/\n")


def _namelist_number(value) -> str:
    """One projection number, as Fortran will read it.

    `{value!r}` was used here, which is right for a builtin and silently
    wrong for anything else: a 0-d ndarray reprs as ``array(-160.)``,
    which the emitted namelist.wps carried verbatim into a file the
    docstring above promises matches the TOML bit-for-bit.  WPS cannot
    parse it.
    """
    scalar = _builtin_scalar(value)
    if scalar is not None:
        value = scalar
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"namelist.wps projection value {value!r} "
            f"({type(value).__name__}) is not a number; the emitted "
            "namelist has to be readable by WPS, and no repr of a "
            "non-number is.")
    return repr(float(value))


def _default_name(lat: float, lon: float) -> str:
    ns = "n" if lat >= 0 else "s"
    ew = "e" if lon >= 0 else "w"
    return (f"area_{abs(lat):.2f}{ns}_{abs(lon):.2f}{ew}"
            .replace(".", "p"))


def sizing_summary(exp: ExperimentConfig, estimate, budget_bytes: int,
                   vram_gib: float, phases=None, *, measured_free_bytes=None) -> str:
    """The whole sizing verdict on one line: what fits, in what.

    The itemized table -- per-domain dx, mass grid, dt, resident bytes,
    the envelope factor and its measurement basis -- is nine lines of
    real accounting that belongs in front of anyone tuning a ladder.
    It is not what the reader of a first run needs, and it was the top
    of the wall the field exhibit opened with.  So the numbers that
    decide whether this config runs at all stay, and the derivation
    moves to ``--explain``.
    """

    envelope = estimate.peak_envelope_bytes
    phase = ""
    if phases is not None:
        envelope = phases.peak_envelope_bytes
        phase = f", binding phase {phases.binding_phase}"
        if not phases.ingest_priced:
            phase += " -- ingest NOT PRICED for this source"
        host_need = getattr(phases, "host_preparation_bytes", 0)
        host_ram = getattr(phases, "host_ram_bytes", None)
        if host_need:
            phase += (f"; preparation on the CPU holds about "
                      f"{host_need / GIB:.2f} GiB of "
                      + ("host RAM" if host_ram is None else
                         f"the host's {host_ram / GIB:.2f} GiB of RAM"))
    basis = (f"an estimate for a declared {vram_gib:g} GiB card, not a "
             "measurement of hardware in this machine; `woof check` on "
             "the real card is what measures it"
             if measured_free_bytes is None else
             f"sized against {measured_free_bytes / GIB:.2f} GiB available "
             "on this GPU; Run checks available memory again before downloading")
    return (f"sizing: {len(exp.domains)} domain(s); alloc "
            f"{estimate.alloc_estimate_bytes / GIB:.2f} GiB, peak "
            f"envelope {envelope / GIB:.2f} GiB of a "
            f"{budget_bytes / GIB:.2f} GiB budget "
            f"({(budget_bytes - envelope) / GIB:.2f} GiB headroom{phase}) "
            f"-- {basis}")


def fit_memory(estimate, phases, budget_bytes: int,
               sizing: SizingBudget) -> dict:
    """The card memory this fit priced, as a record a program reads.

    The figures :func:`sizing_summary` prints, before any words are put
    around them: the binding phase's peak envelope against the budget
    the fit was held to.  Every route prices here, whether or not
    ``woof check`` runs after it, so a front end that reads this record
    gets the same figure for a source whose inputs are already on disk
    and for one whose inputs are fetched later.  It was read off the
    check's printed line instead, and a source that defers the check
    until its inputs are downloaded (ERA5) gave a front end no figure.
    """

    return {
        "peak_envelope_bytes": int(phases.peak_envelope_bytes),
        "budget_bytes": int(budget_bytes),
        "binding_phase": phases.binding_phase,
        "alloc_estimate_bytes": int(estimate.alloc_estimate_bytes),
        "free_bytes": int(sizing.free_bytes),
        "vram_gib": float(sizing.vram_gib),
        "sizing_basis": ("measured-available" if sizing.measured
                         else "declared-capacity"),
    }


def _print_sizing_table(exp: ExperimentConfig, estimate,
                        budget_bytes: int, vram_gib: float,
                        phases=None, *, measured_free_bytes=None) -> None:
    envelope = estimate.peak_envelope_bytes
    print("sizing (itemized preflight estimator, in-process):")
    print("  domain    dx        mass grid      dt         resident")
    for dc, dom in zip(exp.domains, estimate.domains):
        dx_km = exp.dx_exact(dc.grid_id) / 1000
        dt = exp.dt_exact(dc.grid_id)
        dt_text = (f"{int(dt)} s" if dt.denominator == 1
                   else f"{dt.numerator}/{dt.denominator} s")
        print(f"  d{dc.grid_id:02d}     {float(dx_km):6.3f} km  "
              f"{dc.run.nx:4d} x {dc.run.ny:<4d}   {dt_text:>8}   "
              f"{dom.resident_bytes / GIB:6.2f} GiB")
    family = envelope_platform(vram_gib=vram_gib)
    print(f"  peak envelope: {estimate.peak_envelope_terms()}")
    print(f"    envelope basis: {family}; {estimate.envelope_basis}")
    # An unmeasured platform gets the conservative accounting, which is
    # a substitution the user has to be able to see.
    platform_note = unknown_platform_note()
    if platform_note is not None:
        print(f"    {platform_note}")
    if phases is not None and not phases.ingest_priced:
        print(f"  ingest (preprocessing): NOT PRICED for --source "
              f"{phases.source} -- that lane is the native-hybrid-level "
              "ingest, which this estimator does not model; the envelope "
              "above is the forecast phase only")
    elif phases is not None and getattr(phases, "preprocess_backend", "cuda") == "cpu":
        print("  ingest (preprocessing): CPU; no GPU allocation in this phase")
        host = getattr(phases.ingest, "host_preprocess_bytes", None)
        if host is not None:
            print(f"    CPU preprocessing working-set estimate: {host / GIB:.2f} GiB of system RAM")
    elif phases is not None:
        ingest = phases.ingest
        nest_ingest = (
            f" + {len(ingest.nest_state_items)} nest initial state(s) "
            f"{ingest.nest_state_bytes / GIB:.2f} GiB, all resident for "
            f"the single export transaction"
            if ingest.nest_state_items else "")
        print(f"  ingest (preprocessing): root {ingest.n_forcing_times} "
              f"forcing times x {ingest.per_time_bytes / GIB:.2f} GiB each, "
              f"{ingest.resident_times} resident at a time"
              f"{nest_ingest} = "
              f"{ingest.resident_bytes / GIB:.2f} GiB resident; peak "
              f"envelope {ingest.peak_envelope_bytes / GIB:.2f} GiB")
        print(f"    ingest envelope basis: {INGEST_PEAK_ENVELOPE_BASIS}")
    if phases is not None:
        print(f"  BINDING PHASE: {phases.verdict(budget_bytes)}")
        envelope = phases.peak_envelope_bytes
    free_gib = (card_assumed_free_gib(vram_gib) if measured_free_bytes is None
                else measured_free_bytes / GIB)
    reserve_gib = free_gib - budget_bytes / GIB
    free_basis = (f"{vram_gib:g} GiB card presents about {free_gib:g} GiB free"
                  if measured_free_bytes is None else
                  f"{free_gib:.2f} GiB available on this GPU when measured")
    print(f"  budget {budget_bytes / GIB:.2f} GiB "
          f"({free_basis}, minus this "
          f"suite's {reserve_gib:.2f} GiB reserve); headroom "
          f"{(budget_bytes - envelope) / GIB:.2f} GiB")
    if measured_free_bytes is not None:
        print("  ESTIMATED WORKLOAD, MEASURED AVAILABLE MEMORY: Run checks "
              "available memory again before downloading; other GPU work "
              "can change it after this configuration is created.")
        return
    print("  ESTIMATE FOR HARDWARE NOT PRESENT: every figure above is an "
          "estimate for the declared card, not a measurement of hardware "
          "in this machine.  Non-pool terms are priced against the "
          "conservative measured reference device profile (the largest "
          "known-device intercept), so the estimate is never more "
          "optimistic than a present-card measurement; `woof check` on "
          "the real card is what measures it.")


def _missing_case_inputs(out: Path, case_data: dict) -> list[str]:
    from woof.case_data import expand_path_variables
    base = out.parent
    missing = []

    def resolve(raw: str) -> Path:
        expanded = Path(expand_path_variables(raw, "wizard", str(out)))
        return expanded if expanded.is_absolute() else base / expanded

    for raw in case_data["forcing"]:
        if not resolve(raw).is_file():
            missing.append(f"forcing {raw}")
    for key in ("vtable", "wps_namelist"):
        if not resolve(case_data[key]).is_file():
            missing.append(f"{key} {case_data[key]}")
    root = resolve(case_data["geog_root"])
    if not root.is_dir():
        missing.append(f"geog_root {case_data['geog_root']}")
    else:
        for dataset in GEOG_DATASETS:
            if not (root / dataset).is_dir():
                missing.append(f"geog_root dataset {dataset}")
    return missing


def gray_zone_headline(chain_km, shared: dict, les_below_m: float | None = None,
                       les_closure: str | None = None) -> list[str]:
    """The gray-zone advisory's first clause: the finding, no mechanism.

    The full :func:`gray_zone_advisory` sentence is four printed lines
    of correct and essential science, and it is what the emitted
    config carries in its header comment -- permanently, where it is
    read next to the settings it is about.  On stdout it was competing
    with the one line the reader needed, so stdout gets the finding and
    ``--explain`` (and the file itself) keep the reasoning.

    Derived from the same call, never a second transcription of the
    numbers: a headline that could disagree with the advisory would be
    worse than no headline.
    """

    full = gray_zone_advisory(chain_km, shared, les_below_m, les_closure)
    return [line.split(", so ", 1)[0] + "." for line in full]


def cumulus_gray_zone_headline(chain_km, cu_physics_by_domain
                               ) -> list[str]:
    """Each cumulus finding's first clause -- same contract as
    :func:`gray_zone_headline`: derived from the same call, never a
    second transcription of the numbers, so a headline that could
    disagree with the advisory cannot exist.  The emitted config's
    header comment carries the full sentences either way."""

    return [line.split(", so ", 1)[0] + "."
            for line in cumulus_gray_zone_advisory(
                chain_km, cu_physics_by_domain)]


def _print_geog_help() -> None:
    from woof.geog_assets import datasets_required_by, size_phrase
    print("  static geography: woof reads a locally staged WPS_GEOG "
          "tree; `woof fetch-geog` downloads and stages it ("
          + size_phrase(datasets_required_by("wrf"))
          + " for the WRF door, resumable).  The geog_root "
          "directory must contain these dataset directories:")
    print("    " + ", ".join(GEOG_DATASETS))


#: The pairing predicate -- why a source cannot prepare a profile, or
#: ``None`` -- and the emission-route gate table behind it.  Both are
#: DEFINED in :mod:`woof.physics_menu` and imported above; the
#: predicate is re-exported under this name because every reader in the
#: tree spells it ``domain_wizard.profile_route_blocker``.


def profiles_blocked_on_source(source: str) -> tuple[str, ...]:
    """The shipped profiles ``source`` refuses, in listed order.

    Exists so ``--help`` can say which of the eight it advertises are
    not selectable on a given route.  DERIVED from the registry, never
    listed: a hard-coded pair would be a second declaration of the same
    fact and would go stale the moment a route gained the component
    back, leaving the help lying in the other direction.
    """

    return tuple(profile for profile in WIZARD_PHYSICS_PROFILES
                 if profile_route_blocker(profile, source) is not None)


def _profile_help_route_note() -> str:
    """The ``--help`` sentence about profiles a route cannot prepare.

    ``--help`` listed eight profiles with no marker while two of them
    were refused unconditionally on ``--source gfs`` -- which is the
    DEFAULT source -- so a reader choosing from the list had a 1-in-4
    chance of picking something that could never work, and learned it
    only from the refusal.  The refusal itself is accurate and precise;
    the advertisement was not.

    Empty when every listed profile is preparable on every source,
    which is what this note existing at all is waiting for.

    It walks EVERY plannable source rather than the three the door used
    to offer, so widening the door cannot leave a route's refusals
    unadvertised -- the same one-in-four gap, one layer up.
    """

    notes = []
    for source in planable_sources():
        blocked = profiles_blocked_on_source(source)
        if blocked:
            notes.append(f"--source {source} cannot prepare "
                         + " or ".join(blocked))
    if not notes:
        return ""
    return ("  NOT every profile runs on every route: " + "; ".join(notes)
            + " -- the wizard refuses those pairings and names the "
              "missing component rather than emitting a config the "
              "front door would reject.  ")


#: The source ``--source`` binds when none is named.  One constant, read
#: by the flag and by the help sentence below, because the help used to
#: state two sources' defaults as literal text and a route gate moving
#: would have left it lying.
DEFAULT_WIZARD_SOURCE = "era5"
#: What a bare `woof domain` runs on when the default source's keyless
#: store does not hold the requested window yet (ERA5 trails real time by
#: about a week): the public forecast source every box can fetch.
DEFAULT_WIZARD_FALLBACK_SOURCE = "gfs"


def _keyless_default_source(args) -> None:
    """On a computer with no CDS key, the default reads a keyless source.

    The CDS is a keyed job API: a bare `woof domain` wrote a CDS-only
    [fetch] table and the first `woof go` stopped at acquisition on
    any computer without ~/.cdsapirc, while `woof doctor` reported the
    keyless ARCO store available.  With no key:

    * ERA5 is read from Google's keyless ARCO store (the same reanalysis
      from 1940), unless a provider or the CDS-only ensemble members
      were named.
    * A window the store does not hold yet (ERA5 is published about a
      week behind real time, so yesterday's date failed outright at the
      fetch, after the plan had been accepted) runs on the fallback
      source when the source itself was not named, and is refused here,
      naming the store's newest cycle, when `--source era5` was.

    With a key the table is exactly what it always was.  Each choice is
    said in one sentence, so the reader knows which source runs and why.
    """

    from woof.fetch import (cds_credentials_path, cds_credentials_present,
                             resolve_latest_cycle)

    if (args.source != DEFAULT_WIZARD_SOURCE
            or getattr(args, "era5_provider", None) is not None
            or getattr(args, "era5_product", None) == "ensemble_members"
            or cds_credentials_present()):
        return
    keyless = (f"source: ERA5 from Google's keyless ARCO store, because no "
               f"Copernicus CDS key was found at {cds_credentials_path()} "
               "(add one, or pass --era5-provider cds, to use the CDS)")
    args.era5_provider = "arco"
    raw = str(args.cycle).strip()
    cycle = None if raw.lower() == "latest" else parse_cycle(raw, args.source)
    start_hour = getattr(args, "forecast_start_hour", None) or 0
    hours = int(getattr(args, "hours", 0) or 0)
    newest = None
    if cycle is not None:
        try:
            newest = resolve_latest_cycle(
                args.source, start_hour + hours, provider="arco",
                **({"start_hour": start_hour} if start_hour else {}))
        except (RuntimeError, OSError, ValueError):
            # The store's bounds could not be read; the fetch itself
            # checks the same attribute and says so.
            newest = None
    if cycle is None or newest is None or cycle <= newest:
        print(keyless)
        return
    window = (f"{cycle:%Y-%m-%dT%H}Z through "
              f"{cycle + timedelta(hours=start_hour + hours):%Y-%m-%dT%H}Z")
    reason = (f"ERA5 for {window} is not in Google's keyless ARCO store yet "
              f"(its newest cycle covering a {start_hour + hours} h window is "
              f"{newest:%Y-%m-%dT%H}Z; ERA5 is published about a week behind "
              f"real time) and no Copernicus CDS key was found at "
              f"{cds_credentials_path()}")
    if getattr(args, "source_explicit", False):
        raise ValueError(
            f"--source {args.source}: {reason}. The fetch would stop at that "
            f"store's coverage. Choose a cycle up to {newest:%Y-%m-%dT%H}Z, "
            f"pass --source {DEFAULT_WIZARD_FALLBACK_SOURCE}, or fetch again "
            "once the store publishes the window.")
    args.source = DEFAULT_WIZARD_FALLBACK_SOURCE
    args.era5_provider = None
    print(f"source: {args.source.upper()}, because {reason} "
          f"(--source {DEFAULT_WIZARD_SOURCE} keeps ERA5 and waits for it)")


class _DeclaredSourceAction(argparse.Action):
    """Keep the ordinary default and distinguish an explicit forcing choice."""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        namespace.source_explicit = True


def _profile_help_default_note() -> str:
    """The ``--help`` sentence about what a bare run binds.

    DERIVED.  This sentence used to read "(gfs/era5 default: <id> ...;
    hrrr default: <id> ...)" -- two source names and two profile ids
    typed into help text, correct on the day they were written and
    silently wrong the moment a route gate moves or a third gated source
    is registered.  Every source has its own computed default now, so
    help names the DEFAULT source's and points at the door that answers
    for the rest rather than pretending to enumerate them.
    """

    from woof.physics_menu import SPACING_DEFAULTS

    default = resolved_physics_profile(DEFAULT_WIZARD_SOURCE, None)
    # The default by grid spacing, from its own table, so a row added
    # there is a clause here with no edit.
    by_spacing = "".join(
        f"; a run whose finest grid is under "
        f"{float(row['finest_dx_below_m']) / 1000.0:g} km binds "
        f"{row['profile_id']} instead, on every source whose route admits it"
        for row in SPACING_DEFAULTS)
    return (f"(--source {DEFAULT_WIZARD_SOURCE}, the default source, "
            f"binds {default}{by_spacing}; every source has its own "
            "computed default and its own admissible set -- `woof run-plan "
            "--physics-profiles` prints the whole table)")


def _refuse_profile_its_source_cannot_prepare(profile, source, *,
                                              domains: int = 1) -> None:
    """Do not emit a config the named source's front door will refuse.

    The wizard prints, of a profile-bound config, that it "passes the
    prepared single-domain forecast runner's physics guard exactly as
    emitted".  That sentence has to stay true.  When a route withdraws a
    component -- GFS and RUC, whose forecast cannot complete its first
    step -- the wizard must stop offering the pairing rather than write
    the file and let the front door refuse it later, which is the same
    selectable-but-not-usable shape one surface earlier.

    Scoped by the same registry declaration the front door enforces, so
    the two cannot disagree, and silent for every pairing that
    declaration still offers.
    """

    blocker = profile_route_blocker(profile, source, domains=domains)
    if blocker is not None:
        # One sentence at the boundary; the registry pointer and the
        # failure mechanism ride the --explain layer.
        from woof.explain import layered
        head, _, detail = str(blocker).partition(": ")
        raise ValueError(layered(
            f"--physics-profile {profile} cannot be prepared with "
            f"--source {source}: {head}", detail))


def resolve_sizing_card(card: str | None, vram_gib: float | None):
    """The VRAM the wizard sizes against, the CARD, and where they came from.

    Returns ``(vram_gib, device_profile, sentence)``.  ``device_profile``
    is the measured local card when nothing was declared and ``None``
    when the caller declared one -- a declaration says "size for a
    machine that need not be this one", and that machine's shader count
    is unknown, so it keeps the conservative reference profile.

    THE PROFILE IS THE POINT.  The capacity was already measured here;
    what was thrown away was the rest of the probe's answer, so a wizard
    that knew it was looking at a 68-SM RTX 3080 priced that card's
    local-memory backing store against a 170-SM reference and charged it
    1.49 GiB it does not have (task 206).
    """

    return _resolve_vram_budget(card, vram_gib)


def _resolve_vram_budget(card: str | None,
                         vram_gib: float | None):
    """Compatibility view of the sizing card: capacity, profile, note."""
    sizing = resolve_sizing_budget(card, vram_gib)
    return sizing.vram_gib, sizing.device_profile, sizing.note


@dataclass(frozen=True)
class SizingBudget:
    vram_gib: float
    free_bytes: int
    device_profile: object
    note: str | None
    measured: bool = False


#: The capacity options a sizing command can accept, in the order a
#: remedy names them.  Every caller of :func:`resolve_sizing_budget`
#: passes the ones ITS parser defines, because the helper serves doors
#: that accept both (``woof domain``), only ``--vram-gib`` (``woof
#: research hardware``/``create``) and neither (``woof domain-tiles``),
#: and a remedy naming an option the door rejects ends in exit 2.
CAPACITY_OPTIONS = ("--card", "--vram-gib")


def _capacity_words(declare: tuple[str, ...]) -> dict[str, str]:
    """The remedy phrases for the capacity options one door accepts."""

    unknown = [option for option in declare if option not in CAPACITY_OPTIONS]
    if unknown:
        raise ValueError(f"not a capacity option: {', '.join(unknown)}")
    offered = [option for option in CAPACITY_OPTIONS if option in declare]
    spelled = {"--card": f"--card {'/'.join(sorted(CARD_VRAM_GIB))}",
               "--vram-gib": "--vram-gib N"}
    sources = {"--card": "a declared tier", "--vram-gib": "a declared GiB figure"}
    kinds = [sources[option] for option in offered]
    kinds.append("a measurement of the card in this machine")
    if len(kinds) == 1:
        needs = "a measurement of the card in this machine"
    else:
        count = {2: "two", 3: "three"}[len(kinds)]
        needs = (f"one of {count} sources: " + ", ".join(kinds[:-1])
                 + (", or " if len(kinds) > 2 else " or ") + kinds[-1])
    return {
        "slash": "/".join(offered),
        "or": " or ".join(offered),
        "spelled": " or ".join(spelled[option] for option in offered),
        "target": "card" if "--card" in offered else "capacity",
        "needs": needs,
    }


def resolve_sizing_budget(card: str | None, vram_gib: float | None, *,
                          declare: tuple[str, ...] = CAPACITY_OPTIONS) -> SizingBudget:
    """Resolve capacity and available memory from one probe or declaration.

    ``declare`` names the capacity options the calling command accepts
    (:data:`CAPACITY_OPTIONS` or a subset); every remedy below offers
    those and no others.

    Three sources, in the only defensible order:

    1. A DECLARATION (``--card`` tier or ``--vram-gib``) wins outright and
       nothing local is probed -- the caller said "size for a machine that
       need not be this one", and a probe would at best be ignored.
    2. Nothing declared: the local card is MEASURED, through the same
       short-lived subprocess probe the `go` memory gate uses (the
       measured-thresholds rule -- a number this box can produce beats an
       assumed tier). Use its available memory for fitting, so a new file
       fits the same machine state that `go` checks immediately afterward.
    3. Nothing declared and nothing measurable: a refusal that names the
       real choice.  This replaces two prior behaviors, both wrong: a
       silent 24 GiB assumption (a config sized for a card nobody has),
       and -- on CPU-only installs -- a cupy package check for a command
       that integrates nothing on a card (the 2.5.0 persona walks'
       finding).

    Explicit declarations retain their assumed-free policy and never
    probe the local device. A measured budget retains the whole answer,
    including the device profile used to price grid-independent terms.
    """

    if card is not None or vram_gib is not None:
        # --vram-gib beside --card is the capacity; the card is then a
        # label.  A card alone must carry its capacity in one of the three
        # spellings card_capacity_gib reads.
        if vram_gib is not None:
            capacity = float(vram_gib)
        else:
            capacity = declared_card_gib(card)
        if not math.isfinite(capacity) or capacity <= 0:
            raise ValueError(f"--vram-gib {capacity:g} is not a size: "
                             "pass a finite positive card capacity in GiB")
        from woof.core.device_probe import pool_lease_budget_bytes
        free = int(card_assumed_free_gib(capacity) * GIB)
        lease = pool_lease_budget_bytes()
        note = None
        if lease is not None:
            free = min(free, lease)
            note = f"domain: sizing against the pool lease budget {lease / 2**20:g} MiB"
        return SizingBudget(capacity, free, None, note)
    words = _capacity_words(tuple(declare))
    probe = device_memory_probe_subprocess()
    total = probe.get("total_bytes") if isinstance(probe, dict) else None
    if isinstance(total, int) and not isinstance(total, bool) and total > 0:
        free = probe.get("free_bytes")
        if (not isinstance(free, int) or isinstance(free, bool)
                or not 0 <= free <= total):
            raise ValueError(
                "The local GPU probe did not report valid available memory; "
                "make the local card readable"
                + (f" or declare {words['slash']} to size for another machine."
                   if declare else "."))
        measured = total / GIB
        from woof.core.device_probe import sizing_free_memory
        free, lease_note = sizing_free_memory(free)
        # The SAME probe answer, used WHOLE.  Reading the capacity out of
        # it and dropping the shader census beside it is exactly how a
        # measured 68-SM card came to be priced on a 170-SM profile.
        device_profile = profile_from_device_probe(probe)
        name = device_profile.name if device_profile is not None else None
        card_words = f"{name}, " if name else ""
        basis = ("" if device_profile is None else
                 f"; grid-independent terms {non_pool_basis(device_profile)}")
        lead = (f"no {words['slash']} declared, so the budget is"
                if declare else "the budget is")
        tail = (f"; declare {words['or']} to size for another machine"
                if declare else "")
        return SizingBudget(measured, free, device_profile, (
            f"domain: {lead} the "
            f"measured local card ({card_words}{measured:g} GiB total, "
            f"{free / GIB:.2f} GiB available)"
            f"{tail}{basis}" + (f"; {lease_note}" if lease_note else "")), measured=True)
    reason = (device_memory_probe_reason()
              or "the local card could not be measured")
    # The whole reason, not a word in it: a probe that failed on a CuPy
    # it did import quotes that error, and "cupy" in its text told the
    # reader to install the CuPy that was already there.
    cupy_missing = reason == PROBE_REASON_NO_RUNTIME
    install_cupy = ("install cupy (pip install 'recast-woof[gpu-cu12]', or "
                    "'recast-woof[gpu-cu13]' on a CUDA-13 box)")
    from woof.explain import layered
    why = ("It used to assume a 24 GiB card when nothing was "
           "declared, which sized domains for hardware nobody stated exists; "
           "an assumption is not a budget.  The measurement runs in a "
           "short-lived subprocess (no CUDA context survives in this "
           "process) and GPUWM_NO_LOCAL_GPU suppresses it entirely.")
    if not declare:
        # A door that plans against the memory it MEASURES has no
        # declared-capacity route at all, so the only ways back are a
        # readable card here or running on the machine that has one.
        way_back = (f"{install_cupy} so it can measure the local card"
                    if cupy_missing else "make the local card readable")
        raise ValueError(layered(
            f"this command plans against the memory it measures on the "
            f"local card and takes no declared capacity, and {reason}.  "
            f"Run it on the machine whose card it plans for, or "
            f"{way_back}; `woof doctor` shows whether the card is readable.",
            f"The measured memory decides the plan, so the command needs "
            f"{words['needs']}.  {why}"))
    if cupy_missing:
        way_back = f"or {install_cupy} so the wizard can measure the local card"
    else:
        way_back = ("or make the local card readable so the wizard can "
                    "measure it")
    raise ValueError(layered(
        f"this wizard sizes every emitted level against a VRAM budget, "
        f"and there is none: no {words['slash']} was declared, and "
        f"{reason}.  Declare the target {words['target']} -- "
        f"{words['spelled']} -- {way_back}.",
        "The budget decides every grid dimension in the emitted file, so "
        f"the wizard needs {words['needs']}.  {why}"))


def _supplied_forcing_schedule(args, start_time):
    """One native time inventory, resolved before any candidate is priced."""
    if not args.forcing:
        return None, None, None
    if args.source != "era5":
        raise ValueError("--forcing supplies native ERA5 GRIB1 inputs; use --source era5")
    from woof.case_data import _resolve_forcing
    from woof.ingest.grib import inspect_era5_forcing_times
    from woof.ingest.preflight import build_lbc_records

    paths = _resolve_forcing(Path.cwd(), list(args.forcing), "woof domain --forcing")
    # Before the native inventory opens anything: a missing file used to
    # leave as a FileNotFoundError traceback from its stat(), and a folder
    # was read as a zero-length GRIB and refused as "empty".
    for path in paths:
        if not path.is_file():
            what = "is a folder" if path.is_dir() else "does not exist"
            raise ValueError(
                f"--forcing {path} {what}; --forcing takes forcing files, "
                "or a glob pattern that matches files")
    vtable = Path(args.vtable) if args.vtable else _PACKAGED_VTABLE
    times = inspect_era5_forcing_times(paths, vtable)
    if start_time not in times:
        raise ValueError(f"supplied forcing is missing the requested start time {start_time}")
    records = build_lbc_records(times)
    if not records:
        raise ValueError("supplied forcing needs at least two pressure-level valid times")
    deltas = {record.delta_seconds for record in records}
    if len(deltas) != 1:
        shortest = min(deltas)
        gap = next(record for record in records if record.delta_seconds != shortest)
        raise ValueError(
            f"supplied forcing has a gap or nonuniform cadence: {gap.start_time} "
            f"to {gap.end_time} is {gap.delta_seconds:g} s, while the shortest "
            f"interval is {shortest:g} s; supply a continuous uniform time series")
    interval = deltas.pop()
    if interval % 3600 != 0:
        raise ValueError(
            "supplied ERA5 cadence must be a whole number of hours "
            "for the fetch contract")
    end = start_time + timedelta(hours=args.hours)
    if times[-1] < end:
        raise ValueError(
            f"supplied forcing ends at {times[-1]}, before the requested end {end}; "
            "supply the remaining forcing times or shorten --hours")
    # Runtime retains every interval at/after start, even beyond run_seconds.
    count = sum(when >= start_time for when in times) - 1
    return paths, float(interval), count


def _check_emitted_config(out: Path, sizing: SizingBudget, *,
                          target_machine=None, remote_hardware=False) -> int:
    """Check with the same sizing sample, retaining its measured device profile."""
    from woof.cli import EXIT_GEOGRAPHY_MISSING
    from woof.core.mynn_pbl_scratch import (
        mynn_pricing_memory, mynn_pricing_total_bytes)

    with mynn_pricing_memory(total_bytes=mynn_pricing_total_bytes(
            sizing.vram_gib, measured=sizing.measured or remote_hardware),
                             free_bytes=sizing.free_bytes):
        rc, geography_missing_only = _check_emitted_config_for_card(
            out, sizing, target_machine=target_machine,
            remote_hardware=remote_hardware)
    # Creation must succeed before fetch-geog can read this configuration.
    # The geography-only verdict means memory passed. A refused tiled plan
    # also uses 6, so retain the check's origin rather than just its number.
    # Its dataset list and exact remedy have already been printed once.
    if rc == EXIT_GEOGRAPHY_MISSING and geography_missing_only:
        print("woof check: geography staging is the next step; "
              "the configuration was created. Follow the remedy above "
              "before running.")
        return 0
    if rc == 0:
        print("woof check: PASS (rc 0)")
    return rc


def _check_emitted_config_for_card(out: Path, sizing: SizingBudget, *,
                                  target_machine=None,
                                  remote_hardware=False) -> tuple[int, bool]:
    from woof.cli import build_parser

    argv = ["check", str(out), "--free-gib", f"{sizing.free_bytes / GIB:.17g}",
            "--vram-gib", f"{sizing.vram_gib:.17g}"]
    # This is an in-process handoff, not a new CLI option. The composed
    # handler still runs input preflight before the memory check.
    args = build_parser().parse_args(argv)
    if sizing.measured:
        args._shared_sizing_budget = sizing
    if remote_hardware:
        args._shared_target_machine = target_machine
        args._target_hardware_supplied = True
    rc = args.func(args)
    return rc, getattr(args, "_geography_missing_only", False)


def _domain_target_hardware(args, sizing_budget=None):
    """Reuse domain-fit's measured target contract without a local probe."""
    hardware = getattr(args, "hardware_json", None)
    host_path = getattr(args, "target_host_memory_json", None)
    if hardware is not None and (args.card is not None or args.vram_gib is not None
                                 or sizing_budget is not None):
        raise ValueError("Choose the selected hardware snapshot or an explicit card capacity, not both")
    if hardware is not None and host_path is not None:
        raise ValueError("The selected hardware snapshot already carries target host memory; choose only one host measurement")
    if host_path is not None and args.card is None and args.vram_gib is None:
        raise ValueError("--target-host-memory-json requires an explicit --card or --vram-gib budget")
    identity = None
    if hardware is not None:
        from woof.starter_template import hardware_sizing
        sizing, identity = hardware_sizing(hardware)
    else:
        sizing = (sizing_budget if sizing_budget is not None
                  else resolve_sizing_budget(args.card, args.vram_gib))
    if host_path is not None:
        from woof.target_hardware import validate_host_memory
        host_path = Path(host_path).expanduser().resolve(strict=True)
        if not host_path.is_file() or host_path.stat().st_size > 512 * 1024:
            raise ValueError("Selected target host snapshot must be a JSON file no larger than 512 KiB")
        payload = host_path.read_bytes()
        document = json.loads(payload)
        if not isinstance(document, dict):
            raise ValueError("Selected target host snapshot must contain a JSON object")
        identity = {"path": str(host_path), "sha256": hashlib.sha256(payload).hexdigest(),
                    "host_memory": validate_host_memory(document.get("host_memory", document))}
    target_machine = None
    if identity is not None:
        from woof.target_hardware import validate_host_memory
        host = identity.get("host_memory")
        if host is not None or getattr(args, "tiles", None) not in (None, "off"):
            host = validate_host_memory(host)
            from tilestream.autoplan import Machine
            target_machine = Machine(vram_bytes=sizing.free_bytes,
                host_bytes=host["total_bytes"], name="selected forecast target",
                host_source="probe", device_profile=sizing.device_profile)
    return sizing, target_machine, identity is not None


def domain_main(args, *, sizing_budget: SizingBudget | None = None,
                memory: dict | None = None) -> int:
    """``woof domain``: fit, write and check one configuration.

    ``memory``, when given, receives :func:`fit_memory`'s record for the
    written configuration, so an in-process caller reads the priced
    figure rather than the printed sizing line.
    """

    from woof.earth2_licensing import check_namespace
    check_namespace(args)
    les_below_m = getattr(args, "les_below_m", None)
    les_closure = getattr(args, "les_closure", None)
    les_closure_keys(les_closure)
    # FIRST, before any geometry: the source name becomes a registry row,
    # or the run stops with the registry's own words.  Everything below
    # reads the row -- coverage, cadence, forecast horizon -- so a name
    # that never resolved would have been re-guessed at four later points.
    if getattr(args, "recipe", None):
        from woof.region_recipe import domain_recipe_main
        return domain_recipe_main(args)
    if getattr(args, "smoke", False):
        # Breakage it prevents: the plain wizard composes no HRRR-Smoke set,
        # so the flag would be read by nothing.
        raise ValueError("--smoke requires --recipe; it turns HRRR-Smoke on in the emitted recipe")
    if getattr(args, "region", None) or getattr(args, "spacing_km", None):
        raise ValueError("--region and --spacing-km require --recipe; otherwise the requested recipe settings would be lost")
    if getattr(args, "point", None) is None and getattr(args, "polygon", None) is None:
        raise ValueError("one of --point LAT,LON or --polygon GEOJSON is required "
                         "(or --recipe ID, optionally with --region/--polygon and "
                         "--spacing-km); without a target there is no domain to fit")
    from woof.source_adapters import source_for_cycle
    args.source = source_for_cycle(args.source, getattr(args, "cycle", None))
    args.source = resolve_source(args.source,
                                 cadence_h=getattr(args, "cadence", None))
    _keyless_default_source(args)
    nz = getattr(args, "nz", None)
    tiles = getattr(args, "tiles", None)
    if nz is not None and nz < 4:
        raise ValueError("--nz must be at least 4 (the vertical stencil width)")
    polygon = None
    level_buffer_values = None
    polygon_path = getattr(args, "polygon", None)
    if polygon_path is None:
        lat, lon = _parse_point(args.point)
        if getattr(args, "buffer_km", None) is not None:
            raise ValueError("--buffer-km requires --polygon")
    else:
        polygon = load_polygon_footprint(polygon_path)
        lat, lon = polygon.center_lat, polygon.center_lon
        level_buffer_values = parse_level_buffers(
            getattr(args, "buffer_km", None))
    point_extent_km = getattr(args, "point_extent_km", None)
    refuse_point_extent_on_polygon(point_extent_km, polygon)
    if point_extent_km is None:
        point_extent_km = POINT_FIT_MAX_EXTENT_KM
    if args.card is not None and args.vram_gib is not None:
        raise ValueError("--card and --vram-gib are mutually exclusive")
    _refuse_profile_its_source_cannot_prepare(
        getattr(args, "physics_profile", None), args.source)
    sizing, target_machine, remote_hardware = _domain_target_hardware(
        args, sizing_budget)
    vram_gib, device_profile, budget_sentence = (
        sizing.vram_gib, sizing.device_profile, sizing.note)
    if budget_sentence is not None:
        print(budget_sentence)
    # The floor is the reserve NOTHING can be sized below: one CUDA
    # context plus the external margin.  Deliberately not the flat
    # `vram_reserve_gib` any more -- that figure is retired from the
    # sizing path, and using it here refused cards the suite-priced
    # reserve would have sized.  Everything above this floor goes to
    # the fit loop, whose refusal names the layout and the
    # arithmetic.
    reserve_floor_gib = (CUDA_CONTEXT_BYTES + EXTERNAL_MARGIN_BYTES) / GIB
    if sizing.free_bytes / GIB <= reserve_floor_gib:
        raise ValueError(
            f"GPU sizing leaves no budget: {sizing.free_bytes / GIB:.2f} GiB "
            f"is available for fitting, and one CUDA context plus the external margin is "
            f"already {reserve_floor_gib:.2f} GiB of it")
    # FREE, not a budget.  The reserve belongs to the candidate experiment
    # -- it carries that suite's local-memory backing store -- so it is
    # subtracted inside the fit loop, by the same call `woof check`
    # makes. Automatic sizing retains sampled free bytes; a declared
    # card retains the conservative assumed-free allowance.
    free_bytes = sizing.free_bytes
    free_gib = free_bytes / GIB
    if args.hours < 1:
        raise ValueError("--hours must be at least 1")
    # The model's time zero.  It is the cycle when the run is initialized
    # from the analysis, and cycle + K when it is initialized from the
    # f{K} forecast lead -- which is a routine thing to want and used to
    # be impossible to ask for on this door.  Checked BEFORE the cycle is
    # resolved, because `--cycle latest` probes the mirrors for a cycle
    # complete through lead + length and a bad lead should not spend a
    # network round trip to be refused.
    start_hour = getattr(args, "forecast_start_hour", None) or 0
    lead_refusal = forecast_start_hour_refusal(args.source, start_hour, args.hours)
    if lead_refusal:
        raise ValueError(lead_refusal)
    from woof.fetch import era5_combined_name, validate_fetch_hints
    acquisition = {"source": args.source, "cycle": args.cycle,
                   "hours": args.hours, "forecast_start_hour": start_hour}
    for key in ("member", "cadence", "era5_product", "era5_provider",
                *(hint[0] for hint in get_source_adapter(args.source).acquisition_hints)):
        value = getattr(args, key, None)
        if value is not None:
            acquisition[key] = value
    # The spacing this window's own ladder publishes, when it is coarser
    # than the source's usual one (a 240 h IFS window runs past f144, where
    # the files come every 6 h).  Taken as if it had been asked for, so the
    # fetch, the boundary interval and the staging check all read the one
    # spacing; a named --cadence is honoured as named.
    requested_cadence = getattr(args, "cadence", None)
    chosen_by_ladder = False
    if requested_cadence is None:
        # A named cycle is asked about its own ladder, as `woof fetch`
        # asks it; `latest` takes the finest spacing any cycle hour
        # publishes over the window, which is the one its walk admits.
        named_cycle = (None if str(args.cycle).strip().lower() == "latest"
                       else parse_cycle(args.cycle, args.source))
        usual = _fetch_cadence_h(args.source, start_hour)
        by_lead = _fetch_cadence_h(args.source, start_hour, args.hours,
                                   cycle=named_cycle)
        if usual is not None and by_lead is not None and by_lead != usual:
            requested_cadence = by_lead
            chosen_by_ladder = True
            acquisition["cadence"] = by_lead
            print(f"cadence: {args.source} publishes every {by_lead} h over "
                  f"f{start_hour:03d}..f{start_hour + math.ceil(args.hours):03d} "
                  f"and not every {usual} h, so the fetch and the boundary "
                  f"interval take {by_lead} h (--cadence names another)")
    cadence, fetch_hours = fetch_window(args.source, args.hours, start_hour, acquisition.get("cadence"))
    if cadence is not None:
        acquisition["cadence"] = cadence
        acquisition["hours"] = fetch_hours
    if acquisition.get("era5_product") == "ensemble_members":
        acquisition.setdefault("era5_provider", "cds")
    if get_source_adapter(args.source).fetch_requires_retrieve:
        acquisition["retrieve"] = True
    from woof.runplan import drivability_for
    acquisition_reachable = (source_has_fetch_front_door(args.source) or
                             drivability_for(args.source).get("requires_source_root"))
    if acquisition_reachable:
        validate_fetch_hints(acquisition, source="domain arguments")
    elif any(key in acquisition for key in ("member", "era5_product", "era5_provider")):
        raise ValueError("This source has no acquisition selection contract. Supply a prepared bundle "
                         "or declare its product and preparation authority before selecting a member.")
    selection = {key: acquisition[key] for key in ("cadence", "member") if key in acquisition}
    if start_hour:
        selection["start_hour"] = start_hour
    # The provider a keyless or mirrored copy is fetched from can trail the
    # source's own publication delay, so `latest` resolves against it.
    provider = ({"provider": acquisition["era5_provider"]}
                if "era5_provider" in acquisition else {})
    cycle = _resolve_cycle(
        args.cycle, source=args.source, hours=acquisition["hours"],
        start_hour=start_hour, **{key: value for key, value in selection.items() if key != "start_hour"},
        **provider)
    if args.source == "hrrr":
        # The cycle horizon is a property of the cycle hour (48 h at
        # 00/06/12/18Z, 18 h otherwise), so a lead can walk a window off
        # the end of what NOAA published.  Refused HERE, with the horizon
        # named, rather than at the fetch after the file has been written.
        from woof.hrrr_forecast import hrrr_source_window
        hrrr_source_window(cycle=cycle, start_hour=start_hour,
                           run_seconds=args.hours * 3600.0)
    start_time = cycle + timedelta(hours=start_hour)
    supplied_forcing, forcing_interval_seconds, forcing_intervals = (
        _supplied_forcing_schedule(args, start_time))
    if requested_cadence is not None and not (chosen_by_ladder and forcing_interval_seconds is not None):
        # Inputs already on disk state their own spacing; the ladder's
        # choice is for a download, so it does not overrule them.
        requested_interval = requested_cadence * 3600
        if forcing_interval_seconds is not None and forcing_interval_seconds != requested_interval:
            raise ValueError("--cadence differs from the supplied input spacing. Use that spacing or omit --cadence.")
        forcing_interval_seconds = requested_interval
    if forcing_interval_seconds is not None:
        print(f"forcing: native inputs supply {forcing_interval_seconds:g} s cadence "
              f"and {forcing_intervals} retained boundary interval(s)")
    name = args.name or _default_name(lat, lon)
    projection = _projection_entries(
        lat, lon, getattr(args, 'projection', 'auto'))
    out: Path = args.out

    mosaic_option = getattr(args, "sf_surface_mosaic", None)
    mosaic_count = getattr(args, "mosaic_cat", None)
    mosaic_canopy = getattr(args, "mosaic_urban_canopy", None)
    noah_mosaic_options = (None if mosaic_option is None and mosaic_count is None
                           and mosaic_canopy is None
                           else (mosaic_option, mosaic_count, mosaic_canopy))

    profile = resolved_physics_profile(
        args.source, getattr(args, "physics_profile", None))
    # NAMING THE SUITE IS THE EXPLICIT CUMULUS REQUEST.  It is the only
    # cumulus statement this door takes, and it is a strong one: naming
    # --physics-profile asserts the config IS that shipped suite, which
    # both prepared routes then enforce switch for switch
    # (woof.gfs_direct.front_door_physics_selection).  Emitting it with
    # a switch retired would hand the user a file the runner they were
    # steered to refuses.  Without the flag the suite is DERIVED, nobody
    # asserted its cumulus scheme, and the grid decides
    # (see _domain_tables).
    #
    # The interactive session composes --physics-profile with the same
    # DERIVED default and prints the command for re-running, so on that
    # door the flag is not a person's assertion.  It costs nothing
    # today: that door pins --ladder 12, whose root is three times the
    # convection-permitting bound, so neither branch can differ there.
    # A future interactive --root-dx has to decide this deliberately
    # rather than inherit it.
    #
    # --cumulus says it outright instead: ``grid`` emits the named suite
    # with its root cumulus left to the grid, the way the derived suite
    # is emitted, and ``suite`` keeps the suite's cumulus at any spacing.
    # A front end that composes physics scheme by scheme names the suite
    # its picks match and says ``grid`` when no cumulus scheme was picked,
    # so the run gets the cumulus the physics check described.
    cumulus_requested = cumulus_requested_by(args)
    # Schemes picked in place of the suite's own (--physics-choices): the
    # check request every render below writes them with, the fit's
    # candidates included.
    physics_mix = physics_mix_request(
        getattr(args, "physics_choices", None), source=args.source,
        profile=getattr(args, "physics_profile", None))
    if physics_mix and "cumulus" in physics_mix["choices"] and getattr(args, "cumulus", None) is None:
        # A picked cumulus scheme is a request for the root's cumulus, as
        # the physics check reads it, so the grid does not retire it and
        # the file does not say it did.
        cumulus_requested = True
    # How the run steps (--clock): every render below, the fit's
    # candidates included, writes the same clock the file will carry.
    # The interactive session's namespace has no such answer and takes
    # the door's default.
    clock = getattr(args, "clock", None) or DEFAULT_CLOCK

    # THE NOCTURNAL DECLARATION IS THE USER'S TO MAKE, AND THIS IS WHERE
    # THEY MAKE IT (2026-08-09).
    #
    # Until today this door wrote
    # acknowledgements = [ASYMMETRIC_RADIATION_NOCTURNAL_ACK] into the
    # emitted [experiment] by itself whenever an asymmetric suite met a
    # window with local night in it.  Every downstream door reads that
    # line and falls silent: `woof check`, `woof run`, `woof go`,
    # run-plan and both prepared runners.  So the wizard was manufacturing
    # the user's consent, into a file that outlives the session, for the
    # exact failure v1.7.1 shipped a guard for -- and the audit reproduced
    # a real user's journey ending in a config that could never be
    # refused by anything.  Speaking the declaration aloud (the advisory
    # below, from lane/advisory) was necessary and is not sufficient: the
    # FILE still carried a statement nobody made.
    #
    # Refuse instead, here, before any fitting or fetching, and name the
    # two ways forward.  --ack is the project's existing idiom for
    # exactly this (woof/cli.py, gfs_direct, prepared_single_domain_
    # forecast, source_cli), so the user types the token themselves and
    # the emitted file records a decision that was actually taken.
    acknowledgements = tuple(getattr(args, "ack", None) or ())
    declared_night = declared_nocturnal_night(
        profile, start_time=start_time, hours=args.hours,
        projection=projection)
    if (declared_night is not None
            and ASYMMETRIC_RADIATION_NOCTURNAL_ACK not in acknowledgements):
        from woof.explain import layered
        from woof.physics_menu import nocturnal_remedy

        # THE REMEDY IS THIS SOURCE'S, NOT A FIXED ID (2026-08-20).
        #
        # This sentence used to name MORRISON_PROFILE_ID on every
        # source.  On --source hrrr that is a Kain-Fritsch suite, and
        # the pairing refusal above (_refuse_profile_its_source_cannot_
        # prepare) then refuses it for cu_physics=1, because the route's
        # 3 km grid resolves its own convection.  So the user was handed
        # a remedy that leads to the next refusal, which names nothing.
        # The remedy comes off the computed per-source menu now -- the
        # same table `woof run-plan --physics-profiles` serves -- so it
        # cannot name a suite this source's route refuses.
        remedy = nocturnal_remedy(args.source)
        raise ValueError(layered(
            f"profile {profile} runs shortwave with longwave OFF and this "
            f"window includes local night (first at "
            f"{declared_night:%Y-%m-%dT%H:%M}Z at "
            f"{projection['ref_lat']:.4g}, {projection['ref_lon']:.4g}), so "
            f"the config this would emit is one every front door refuses "
            f"at load.  Choose a nocturnally valid profile with both "
            f"radiation streams on that --source {args.source} can "
            f"actually prepare -- {remedy['instruction']} -- or, if you "
            f"mean the daytime-only suite and accept the night, "
            f"declare it yourself with --ack "
            f"{ASYMMETRIC_RADIATION_NOCTURNAL_ACK}.  `woof run-plan "
            f"--physics-profiles` lists every suite this source admits",
            "Shortwave heats the surface by day while no longwave scheme "
            "runs, so after sunset the surface radiates to space with no "
            "downward longwave to balance it: skin temperature craters, "
            "the surface saturation humidity collapses with it, and 2 m "
            "dewpoints read far below the airmass.  This wizard used to "
            "write that acknowledgement into the emitted [experiment] for "
            "you, which silenced the guard at every later command for the "
            "life of the file.  A declaration nobody made is not a "
            "declaration.  See docs/public/PHYSICS.md, 'Nocturnal "
            "validity'."))

    if args.ladder is None:
        # Absence, resolved: bare means the single-domain `go` shape
        # (DEFAULT_LADDER); with --root-dx/--chain it means the custom
        # form, which has always ridden on the permissive "auto" value
        # -- the guard below refuses only a --ladder someone TYPED next
        # to the custom flags.
        args.ladder = ("auto"
                       if (getattr(args, "root_dx", None) is not None
                           or getattr(args, "chain", None) is not None)
                       else DEFAULT_LADDER)
    custom = parse_custom_ladder(
        root_dx_km=getattr(args, "root_dx", None),
        chain=getattr(args, "chain", None),
        ladder=args.ladder)
    level_buffers = None
    # What stopped the point fit, straight from the search that stopped.
    # A polygon fit never writes here (it is sized to the drawing, so no
    # request bound is in force), and a memory-bound point fit leaves it
    # empty.
    fit_stop: dict = {}

    def profile_at(ratios_here, root_dx_here) -> str | None:
        # The default by grid spacing binds once the ladder's finest grid
        # is known, and the fit prices the suite the file will carry.  An
        # explicit --physics-profile is returned as named.  The nocturnal
        # refusal above read the spacing-free default, which is the same
        # answer for it: a spacing row binds only a suite with both
        # radiation streams, so it cannot make a window refusable.
        # Schemes picked with --physics-choices and no suite are written
        # over this same suite: the physics check reads the default at
        # the file's finest grid (woof.physics_catalog.experiment_grid),
        # so the base a mix changes is the suite written here.
        return resolved_physics_profile(
            args.source, getattr(args, "physics_profile", None),
            finest_dx_m=finest_spacing_m(root_dx_here, ratios_here),
            domains=len(ratios_here) + 1)

    def derived_profile_at(root_dx_here):
        # For the fit's ladder lever: the suite a shallower ladder binds,
        # when the suite is the door's default rather than one named.
        if getattr(args, "physics_profile", None) is not None:
            return None
        return lambda ratios_here: profile_at(ratios_here, root_dx_here)

    projection = _fit_source_projection(
        projection, args.source, custom[0] if custom is not None else ROOT_DX_M)
    if custom is not None:
        root_dx_m, ratios = custom
        profile = profile_at(ratios, root_dx_m)
        if polygon is None:
            dims, _ = fit_ladder(
                noah_mosaic_options=noah_mosaic_options,
                ratios=ratios, root_dx_m=root_dx_m, free_bytes=free_bytes,
                hours=args.hours, start_time=start_time,
                projection=projection, source=args.source, name=name,
                profile=profile, cumulus_requested=cumulus_requested,
                vram_gib=vram_gib,
                device_profile=device_profile,
                target_machine=target_machine,
                acknowledgements=acknowledgements, nz=nz, tiles=tiles,
                forcing_interval_seconds=forcing_interval_seconds,
                forcing_intervals=forcing_intervals,
                history_interval_s=args.history_interval,
                nest_history_interval_s=args.nest_history_interval,
                stop_out=fit_stop,
                physics_mix=physics_mix, clock=clock,
                point_extent_km=point_extent_km,
                profile_at=derived_profile_at(root_dx_m))
        else:
            level_buffers = _buffers_for_levels(
                level_buffer_values, len(ratios) + 1)
            dims, _ = fit_polygon_ladder(
                noah_mosaic_options=noah_mosaic_options,
                footprint=polygon, buffers_km=level_buffers,
                ratios=ratios, root_dx_m=root_dx_m, free_bytes=free_bytes,
                hours=args.hours, start_time=start_time,
                projection=projection, source=args.source, name=name,
                profile=profile, cumulus_requested=cumulus_requested,
                vram_gib=vram_gib,
                device_profile=device_profile,
                target_machine=target_machine,
                acknowledgements=acknowledgements, nz=nz, tiles=tiles,
                forcing_interval_seconds=forcing_interval_seconds,
                forcing_intervals=forcing_intervals,
                history_interval_s=args.history_interval,
                nest_history_interval_s=args.nest_history_interval,
                physics_mix=physics_mix, clock=clock)
        ladder = "-".join(f"{v:g}" for v in _ladder_dx_km(ratios, root_dx_m))
    else:
        root_dx_m = ROOT_DX_M
        ladders = ([args.ladder] if args.ladder != "auto"
                   else list(_LADDERS_DEEPEST_FIRST))
        fixed_buffer_depth = False
        if polygon is not None and level_buffer_values is not None \
                and len(level_buffer_values) > 1 and args.ladder == "auto":
            fixed_buffer_depth = True
            ladders = [candidate for candidate in ladders
                       if len(LADDER_RATIOS[candidate]) + 1
                       == len(level_buffer_values)]
            if not ladders:
                raise ValueError(
                    f"--buffer-km supplies {len(level_buffer_values)} "
                    "per-level distances, but no preset ladder has that "
                    "many levels; use --root-dx / --chain for a custom "
                    "ladder")
        chosen = None
        shallowest_refusal = None
        for candidate_ladder in ladders:
            try:
                candidate_ratios = LADDER_RATIOS[candidate_ladder]
                candidate_profile = profile_at(candidate_ratios, root_dx_m)
                if polygon is None:
                    dims, _ = fit_ladder(
                        noah_mosaic_options=noah_mosaic_options,
                        ladder=candidate_ladder, free_bytes=free_bytes,
                        hours=args.hours, start_time=start_time,
                        projection=projection, source=args.source, name=name,
                        profile=candidate_profile,
                        cumulus_requested=cumulus_requested,
                        vram_gib=vram_gib,
                        device_profile=device_profile,
                        target_machine=target_machine,
                        acknowledgements=acknowledgements, nz=nz, tiles=tiles,
                        forcing_interval_seconds=forcing_interval_seconds,
                        forcing_intervals=forcing_intervals,
                        history_interval_s=args.history_interval,
                        nest_history_interval_s=args.nest_history_interval,
                        stop_out=fit_stop,
                        physics_mix=physics_mix, clock=clock,
                        point_extent_km=point_extent_km,
                        profile_at=derived_profile_at(root_dx_m))
                    candidate_buffers = None
                else:
                    candidate_buffers = _buffers_for_levels(
                        level_buffer_values, len(candidate_ratios) + 1)
                    dims, _ = fit_polygon_ladder(
                        noah_mosaic_options=noah_mosaic_options,
                        footprint=polygon, buffers_km=candidate_buffers,
                        ratios=candidate_ratios, root_dx_m=root_dx_m,
                        free_bytes=free_bytes, hours=args.hours,
                        start_time=start_time, projection=projection,
                        source=args.source, name=name,
                        profile=candidate_profile,
                        cumulus_requested=cumulus_requested,
                        vram_gib=vram_gib, device_profile=device_profile,
                        target_machine=target_machine,
                        acknowledgements=acknowledgements, nz=nz, tiles=tiles,
                        forcing_interval_seconds=forcing_interval_seconds,
                        forcing_intervals=forcing_intervals,
                        history_interval_s=args.history_interval,
                        nest_history_interval_s=args.nest_history_interval,
                        physics_mix=physics_mix, clock=clock)
            except DomainFitError as error:
                if args.ladder != "auto" or fixed_buffer_depth:
                    raise
                # One line per candidate; the full envelope arithmetic
                # is the same for every ladder and prints with the
                # final refusal if none fits.
                first_sentence = str(error).split(":", 1)[0]
                print(f"ladder {candidate_ladder}: does not fit "
                      f"({first_sentence}); trying the next shallower one")
                shallowest_refusal = error
                continue
            chosen = (candidate_ladder, dims, candidate_buffers,
                      candidate_profile)
            break
        if chosen is None:
            raise DomainFitError(
                "no ladder fits the requested card; even the shallowest "
                f"ladder's smallest layout exceeds the budget a "
                f"{vram_gib:g} GiB card leaves (about {free_gib:g} GiB "
                "free, minus this suite's reserve)",
                # The figures are the shallowest ladder's, the layout
                # this sentence names.
                phases=getattr(shallowest_refusal, "phases", None),
                budget_bytes=getattr(shallowest_refusal, "budget_bytes",
                                     None)) from shallowest_refusal
        ladder, dims, level_buffers, profile = chosen
        ratios = LADDER_RATIOS[ladder]
    # A named suite asked again now the domain count is known, before a
    # byte is written: a route whose nests are built by a stage a single
    # domain never meets refuses there (the nested HRRR route's certified
    # soil layer count), and that refusal belongs here, not after the
    # download.
    _refuse_profile_its_source_cannot_prepare(
        getattr(args, "physics_profile", None), args.source,
        domains=len(ratios) + 1)
    from woof.static.source_defaults import align_source_projection
    projection = align_source_projection(
        projection, dims[0], root_dx_m, args.source)
    # Which bound stopped the POINT fit, if one did -- read off the
    # search itself rather than reconstructed from the emitted root.  It
    # decides two things below: the plain fact the plan summary states,
    # and which flag the oversized-footprint advisory is allowed to name
    # (a bound that makes --vram-gib inert takes it out of that
    # sentence).  A drawn area was sized to the drawing, so no request
    # bound applies to it.
    request_bound = (None if polygon is not None
                     else fit_stop.get("scope"))
    if request_bound not in POINT_FIT_SCOPES:
        request_bound = None
    # Genuine-limit refusal first (its message names the real problem;
    # a pole-containing footprint would otherwise also trip the
    # 180-degree fetch-span refusal below with a less useful message).
    target_option = "--polygon" if polygon is not None else "--point"
    _pole_clearance_refusal(
        projection, *dims[0], root_dx_m,
        target_option=target_option)

    # Fetch hints from the fitted root footprint.  The default data
    # directory lives beside the emitted TOML so the declared forcing
    # paths stay short and the config directory stays relocatable.  The
    # area hint is bounded by the SOURCE's own coverage envelope -- the
    # same grid-derived data the fetch guard enforces -- so the printed
    # next command cannot name ground the source does not carry.
    area_notes: list[str] = []
    coverage_notes: list[str] = []
    area_hint = fetch_area_hint(
        projection, *dims[0], source=args.source, root_dx_m=root_dx_m,
        target_option=target_option, notes=area_notes,
        coverage_notes=coverage_notes)
    # Is this source prepared from bytes already on disk?  Asked here
    # because the cadence fallback below depends on it, and again by the
    # emission further down -- one verdict, from the one function that
    # canonicalizes the spelling first.
    from woof.runplan import drivability_for
    local_source = bool(drivability_for(args.source).get(
        "requires_source_root"))
    cadence = _fetch_cadence_h(args.source, start_hour)
    if cadence is None and local_source:
        # A local-input source has no download ladder to take a spacing
        # from, and the staging check needs one to know which valid times
        # the files on disk must cover, so the registry row's own forcing
        # interval fills it.  Gated on the local-input verdict: applied to
        # every source whose row declares a whole-hour interval, it wrote
        # `cadence = 1` into tables for sources whose own `woof fetch`
        # refuses a cadence outright, so `woof check` passed a config
        # that then died at stage 1 of the run it had been checked for.
        interval = get_source_adapter(args.source).forcing_interval_seconds
        if interval is not None and interval % 3600 == 0:
            cadence = int(interval / 3600)
    if forcing_interval_seconds is not None:
        cadence = int(forcing_interval_seconds / 3600)
    data_dir = (Path(args.data_dir) if args.data_dir
                else out.parent / "data" / name)
    fetch_hints = {
        # The RESOLVED cycle, never the literal "latest": the emitted
        # config is a record of one start time, not of a query.  And the
        # CYCLE, never the start time: they differ by the forecast lead,
        # and a fetch aimed at start_time would ask for the wrong cycle
        # entirely.
        "source": args.source, "cycle": cycle.strftime("%Y-%m-%dT%H"),
        "hours": (args.hours if cadence is None else
                  max(cadence, math.ceil(args.hours / cadence) * cadence)),
        "area": area_hint,
        "out": _relative_or_absolute(data_dir, Path.cwd()),
    }
    from woof import fetch_routes
    for key in ("era5_product", "era5_provider", "retrieve", "member",
                *(hint[0] for hint in get_source_adapter(args.source).acquisition_hints)):
        if key in acquisition:
            fetch_hints[key] = acquisition[key]
    if args.source in fetch_routes.route_ids():
        route = fetch_routes.route_for(args.source)
        if route.members is not None:
            fetch_hints["member"] = fetch_routes.resolve_member(route, acquisition.get("member"))[0]
    elif args.source == "era5":
        from woof.era5_member import validate_selection
        selected = validate_selection(product_type=acquisition.get("era5_product", "reanalysis"),
            provider=acquisition.get("era5_provider", "cds"), member=acquisition.get("member"),
            cadence=cadence, cycle=cycle)
        if selected is not None:
            fetch_hints["member"] = selected
    if cadence is not None:
        fetch_hints["cadence"] = cadence
    if start_hour:
        fetch_hints["forecast_start_hour"] = start_hour
    # Preserve source/cycle metadata for local preparation as well.
    # The run-plan review requires an explicit input root before execution.
    if local_source:
        fetch_hints["source_root"] = str(data_dir.expanduser().resolve())
    emitted_fetch_hints = (fetch_hints
                           if source_has_fetch_front_door(args.source) or local_source
                           else None)
    # And the crop key comes out for a source whose fetch takes whole
    # published objects.  The area is still COMPUTED (the advisories and
    # the hand-staging note below both say what window this config needs)
    # -- it is not ADVERTISED as a download flag the fetch would refuse.
    if (emitted_fetch_hints is not None
            and not source_fetch_takes_a_crop_box(args.source)):
        emitted_fetch_hints = {k: v for k, v in emitted_fetch_hints.items()
                               if k not in {"area", "point", "radius_km"}}
    # And `out` comes out for a source nothing downloads: it names where a
    # download would write, this source is prepared from bytes already on
    # disk, and the key would be read and then have nothing to write.  The
    # directory that matters for such a source is source_root.
    if emitted_fetch_hints is not None and local_source:
        emitted_fetch_hints = {k: v for k, v in emitted_fetch_hints.items()
                               if k != "out"}
    # The TOML's out is relative to the TOML, as [case_data].forcing is;
    # the printed fetch command below keeps the working-directory form,
    # because it is pasted here.  D-11 (2.8.8 acceptance): writing the
    # working-directory form into the file made a config emitted to
    # sub/experiment.toml run only from the folder it was written in.
    if emitted_fetch_hints is not None and "out" in emitted_fetch_hints:
        emitted_fetch_hints = {**emitted_fetch_hints,
                               "out": _relative_or_absolute(data_dir, out.parent)}
    # Prove every emitted hint against the REAL fetch validators before
    # anything is written.  A config the wizard cannot fetch is a config
    # whose printed step 1 exits 2, and that shipped twice: any lead not
    # a multiple of three was written with `cadence = 3` and refused as
    # "not on the 3 h cadence", and an HRRR emission's --area was
    # refused by the coverage guard the wizard had never consulted.
    # Running the fetch's own parsers and planners here is the only
    # check that cannot drift from what the fetch will do; the whole
    # [fetch] table (area included, through parse_area and the
    # per-source coverage gate) is round-tripped again through
    # `validate_fetch_hints` when `experiment_from_text` re-loads the
    # rendered bytes below, still before the file lands on disk.
    parse_cycle(fetch_hints["cycle"], args.source)
    if emitted_fetch_hints is not None:
        validate_fetch_hints(emitted_fetch_hints, source=str(out))

    # [case_data] only where the config-driven front door can accurately
    # consume the fetched data (the native GRIB1 = ERA5 route).
    case_data = None
    vtable_path = None
    if args.source == "era5":
        if args.vtable is not None:
            vtable_path = Path(args.vtable)
            vtable_text = _posix(vtable_path.resolve())
        else:
            vtable_path = out.parent / _PACKAGED_VTABLE.name
            vtable_text = _PACKAGED_VTABLE.name
        # The name the FETCH publishes, not a literal.  The CDS provider
        # hands back GRIB1 and the keyless ARCO reader hands back NetCDF,
        # so a config that spelled one name was a config the other
        # provider's own printed step 1 could not satisfy: the download
        # ran, wrote its file, and `woof go` refused because
        # [case_data].forcing named a file [fetch].out does not produce.
        # Read from the emitted table so the two cannot drift.
        combined = era5_combined_name(fetch_hints.get("era5_provider"))
        forcing = ([_posix(path) for path in supplied_forcing]
                   if supplied_forcing
                   else [_relative_or_absolute(
                       data_dir / combined, out.parent)])
        case_data = {
            "forcing": forcing,
            "vtable": vtable_text,
            "forcing_interval_s": (source_forcing_interval_seconds("era5")
                                   if forcing_interval_seconds is None
                                   else forcing_interval_seconds),
            "wps_namelist": f"{out.stem}.namelist.wps",
            "geog_root": (_posix(Path(args.geog_root).resolve())
                          if args.geog_root
                          else "${GPUWM_CASE_DATA_ROOT}/WPS_GEOG"),
            "sfcp_to_sfcp": True,
            "output_domain": 1,
            "output_title": f"woof {name}",
        }

    text = render_config(
        noah_mosaic_options=noah_mosaic_options,
        name=name, start_time=start_time, hours=args.hours,
        projection=projection, dims=dims, ratios=ratios,
        fetch_hints=emitted_fetch_hints, case_data=case_data,
        root_dx_m=root_dx_m, profile=profile,
        cumulus_requested=cumulus_requested,
        interactive=getattr(args, "interactive", False),
        level_buffers_km=level_buffers,
        history_interval_s=args.history_interval,
        nest_history_interval_s=args.nest_history_interval,
        acknowledgements=acknowledgements, nz=nz, tiles=tiles,
        physics_mix=physics_mix, clock=clock,
        les_below_m=les_below_m, les_closure=les_closure)
    text = with_surface_flux_option(text, getattr(args, "isftcflx", None))
    text = with_les_closure(text, les_below_m, les_closure,
                            inflow_seeding=getattr(args, "les_inflow_seeding",
                                                   "on") != "off")
    text = with_quicklook(text, getattr(args, "quicklook_interval", None),
                          getattr(args, "quicklook_vars", None))
    from woof.static.source_defaults import with_source_static_defaults_text
    text = with_source_static_defaults_text(text, args.source)
    if args.geog_root and case_data is None:
        # D2: the prepared routes have no [case_data], but must retain
        # the explicit tree for check and the printed go command alike.
        root_line = ("geog_root = "
                     + _toml_value(_posix(Path(args.geog_root).resolve())) + "\n")
        if "\n[static]\n" in text:
            text = text.replace("\n[static]\n", "\n[static]\n" + root_line, 1)
        else:
            text += "\n[static]\n" + root_line
    smoothing_spec = getattr(args, "terrain_smoothing", None)
    smoothing_precision = getattr(args, "terrain_smoothing_precision", None)
    if smoothing_spec or smoothing_precision:
        # --terrain-smoothing / --terrain-smoothing-precision: a static line
        # under each [[domain]] whose setting is not WPS's default smoother
        # in ArWen's float64 arithmetic (woof.static.terrain_smoothing).
        from woof.static.terrain_smoothing import (WPS_DEFAULT,
                                                    emit_smoothing,
                                                    parse_smoothing_spec,
                                                    with_precision)
        settings = (parse_smoothing_spec(smoothing_spec) if smoothing_spec
                    else (WPS_DEFAULT,))
        text = emit_smoothing(text, tuple(
            with_precision(setting, smoothing_precision)
            for setting in settings))
    # Round-trip the exact bytes through the real loader before writing.
    exp = experiment_from_text(text, source=str(out))
    terrain_autosmooth = []
    if (getattr(args, "geog_root", None)
            and (Path(args.geog_root) / "topo_gmted2010_30s" / "index").is_file()):
        from woof.static.terrain_autosmooth import survey_domain_terrain
        terrain_autosmooth = survey_domain_terrain(
            exp, Path(args.geog_root), smoothing_settings=(
                tuple(with_precision(setting, smoothing_precision) for setting in settings)
                if smoothing_spec is not None or smoothing_precision is not None else ()))
    # The clock rides on the header line, as the extent does: a line of
    # its own would push the default emission past its screen.
    clock_words = (", adaptive time step"
                   if exp.root.run.use_adaptive_time_step else "")
    interval = (source_forcing_interval_seconds(args.source)
                if forcing_interval_seconds is None else forcing_interval_seconds)
    # On the tables the run holds, as the phases below and `woof check`
    # price them: the root's boundary carries the analysed hydrometeors
    # the source publishes.  The fit record's allocation estimate and the
    # sizing table used to leave them out beside a phase envelope that
    # carried them.
    phases = _sizing_phases(
        exp, machine=target_machine, forcing_intervals=forcing_intervals,
        free_bytes=free_bytes, source=args.source,
        forcing_interval_seconds=interval,
        vram_gib=vram_gib, profile=device_profile)
    estimate = phases.forecast
    envelope = phases.peak_envelope_bytes
    # The same free VRAM is passed to check below; both gates subtract
    # the external margin from it to price this emitted configuration.
    budget = sizing_budget_bytes(
        exp, free_bytes=free_bytes, vram_gib=vram_gib,
        forcing_interval_seconds=interval, profile=device_profile)
    budget_gib = budget / GIB
    if envelope > budget:
        raise DomainFitError(
            "internal fit regression: emitted config's "
            f"{phases.binding_phase} envelope "
            f"{envelope / GIB:.2f} GiB exceeds the budget "
            f"{budget_gib:.2f} GiB")
    if memory is not None:
        memory.update(fit_memory(estimate, phases, budget, sizing))
    if polygon is not None:
        verify_polygon_containment(exp, polygon, level_buffers)

    _write_atomic(out, text)
    wps_text = render_wps_namelist(
        projection, dims, ratios, root_dx_m=root_dx_m, source=args.source,
        forcing_interval_seconds=forcing_interval_seconds)
    if args.source == "hrrr":
        # The HRRR routes read namelists and a target-domain document,
        # not this TOML.  Emitting only the TOML left every one of them
        # to be hand-authored -- which is why the route's own gate had
        # to borrow a proof harness to run at all.
        written = [out] + write_hrrr_route_inputs(
            out, exp, wps_text=wps_text, writer=_write_atomic)
    else:
        wps_path = out.parent / f"{out.stem}.namelist.wps"
        _write_atomic(wps_path, wps_text)
        written = [out, wps_path]
    if args.source == "era5" and args.vtable is None:
        if vtable_path.exists():
            if vtable_path.read_bytes() != _PACKAGED_VTABLE.read_bytes():
                warn(f"kept your existing {vtable_path.name} (it differs "
                     "from the packaged Vtable.ERA5_CDO); pass --vtable "
                     "to name one explicitly")
        else:
            shutil.copyfile(_PACKAGED_VTABLE, vtable_path)
            written.append(vtable_path)

    explain = explain_enabled(args)
    if polygon is None:
        # The extent the fit used rides on the header line, against the
        # --point-extent-km it was sized under: a line of its own would
        # push the default emission past the screen it is held to.
        print(f"woof domain: {name!r} at ({lat:g}, {lon:g}), ladder {ladder} "
              f"({'-'.join(f'{v:g}' for v in _ladder_dx_km(ratios, root_dx_m))} km), "
              f"card {vram_gib:g} GiB, root extent "
              f"{max(dims[0]) * root_dx_m / 1000.0:.0f} km "
              f"(--point-extent-km {point_extent_km:g}){clock_words}")
        # A point carries no extent, so the fit chose one and bounded
        # its own choice.  Stated as fact, once, beside the sizing line
        # that prints the envelope it is under -- never as a warning:
        # this is the ordinary request on the ordinary card.
        if request_bound is not None:
            print("domain: " + point_fit_cap_note(
                request_bound, dims, root_dx_m, point_extent_km))
    else:
        buffers = ",".join(f"{value:g}" for value in level_buffers)
        print(f"woof domain: {name!r}, polygon center ({lat:g}, {lon:g}), "
              f"ladder {ladder} "
              f"({'-'.join(f'{v:g}' for v in _ladder_dx_km(ratios, root_dx_m))} km), "
              f"buffers {buffers} km, card {vram_gib:g} GiB{clock_words}")
        grid_extents = ", ".join(
            f"d{domain.grid_id:02d} {domain.run.nx}x{domain.run.ny} cells "
            f"({domain.run.nx * domain.run.dx / 1000:g}x"
            f"{domain.run.ny * domain.run.dy / 1000:g} km projected)"
            for domain in exp.domains)
        print(f"domain: fitted grid {grid_extents}; contains the requested "
              "polygon and buffers")
    # The spoken half of snap_cadences_to_clock: what the author moved
    # to keep its own two derivations compatible, one line each, or
    # nothing (UX finding N14 -- the silent alternative was exit 2).
    recommended_clock = root_time_step_s(projection["ref_lat"], root_dx_m)
    root = exp.domains[0]
    selected_clock = Fraction(root.time_step) + Fraction(
        root.time_step_fract_num, root.time_step_fract_den)
    if selected_clock != recommended_clock:
        print("domain: " + _derived_clock_note(
            recommended_clock, selected_clock, _root_physics_periods(vars(root.run), vars(root.run))))
    for note in snap_cadences_to_clock(
            selected_clock,
            {key: profile_switches(profile)[key]
             for key in _PER_DOMAIN_PHYSICS})[1]:
        print(f"domain: {note}")
    # Why the file carries the clock it does, behind --explain; the
    # default screen names an adaptive clock on the header line above.
    if explain:
        _, clock_why = clock_decision(clock, time_step=selected_clock,
                                      root_dx_m=root_dx_m, ratios=ratios)
        stepping = ("adaptive time step" if root.run.use_adaptive_time_step
                    else "fixed time step")
        print(f"domain: {stepping} ({clock_why}); --clock "
              f"{'fixed' if root.run.use_adaptive_time_step else 'adaptive'}"
              " changes it")
    if explain:
        _print_sizing_table(exp, estimate, budget, vram_gib,
                            phases=phases, measured_free_bytes=(
                                free_bytes if sizing.measured else None))
    else:
        print(sizing_summary(exp, estimate, budget, vram_gib,
                             phases=phases, measured_free_bytes=(
                                 free_bytes if sizing.measured else None)))
    print(f"wrote {out}"
          + (f" (+ {', '.join(p.name for p in written[1:])})"
             if len(written) > 1 else ""))
    if terrain_autosmooth:
        terrain_receipt_path = out.with_suffix(".terrain-autosmooth.json")
        _write_atomic(terrain_receipt_path, json.dumps({
            "schema": "gpuwm-domain-terrain-autosmooth-survey-v1",
            "terrain_source": "local WPS_GEOG survey; final overlays rechecked in prepare",
            "domains": terrain_autosmooth,
        }, indent=2, sort_keys=True) + "\n")
    # The printed command must be pasteable as-is from THIS directory.
    # A relative data path that climbs out of the cwd silently targets
    # the wrong place when pasted from anywhere else, so it is printed
    # resolved; the TOML keeps the relative form for relocatability.
    printed_out = fetch_hints["out"]
    if printed_out.startswith("../") or args.data_dir:
        printed_out = _posix(Path(data_dir).resolve())
    # `--area=-58.58,...` -- the "=" form is what a leading minus needs
    # in every argument parser, including shells' own.
    area_flag = (f"--area={fetch_hints['area']}"
                 if fetch_hints["area"].startswith("-")
                 else f"--area {fetch_hints['area']}")
    # --cadence is printed whenever it is not the source's own default:
    # the emitted [fetch] table carries it, and a printed command that
    # omits it downloads a different window from the one this config was
    # written for.  At a lead not on the default grid it does not merely
    # differ -- it is refused.
    cadence_flag = (
        f"--cadence {cadence} "
        if cadence is not None and cadence != _SOURCE_CADENCE_H.get(args.source)
        else "")
    if not source_fetch_takes_a_crop_box(args.source):
        # No crop flag for a route that publishes whole objects: `woof
        # fetch` refuses --area on one, so printing it made step 1 exit 2
        # for every table-driven model.  The window is stated after the
        # command instead, where it belongs -- it is a prep fact.
        area_flag = ""
    # The model top this config's ladder needs, when the source's fetch
    # has to be asked for it: the same registry answer `woof go` reads,
    # so the pasted command downloads what the run then prepares.
    from woof.source_adapters import fetch_model_top_pa
    fetch_top = fetch_model_top_pa(args.source, exp.vertical.p_top)
    top_flag = (f"--p-top-pa {fetch_top:g} " if fetch_top is not None else "")
    if source_has_fetch_front_door(args.source):
        fetch_command = ("woof fetch "
                         f"--source {args.source} "
                         f"--cycle {fetch_hints['cycle']} "
                         f"--hours {fetch_hints['hours']} "
                         + (f"{area_flag} " if area_flag else "")
                         + cadence_flag
                         + (f"--era5-product {fetch_hints['era5_product']} "
                            if "era5_product" in fetch_hints else "")
                         + (f"--era5-provider {fetch_hints['era5_provider']} "
                            if "era5_provider" in fetch_hints else "")
                         + ("--retrieve " if fetch_hints.get("retrieve") else "")
                         + (f"--forecast-start-hour {start_hour} "
                            if start_hour else "")
                         + (f"--member {fetch_hints['member']} "
                             if "member" in fetch_hints else "")
                         + "".join(f"--{key.replace('_', '-')} {shlex.quote(str(fetch_hints[key]))} "
                                   for key, *_ in get_source_adapter(args.source).acquisition_hints
                                   if key in fetch_hints)
                         + top_flag
                         + f"--out {_printed_path(printed_out)}")
    else:
        # NOT a `woof fetch` line.  This source has a runnable profile
        # and no download route, and printing a command that refuses is
        # how a reader concludes the tool is broken -- the 1.3.0 field
        # exhibit.  Say what is true, and say what the config is FOR: the
        # geometry, levels, physics and boundary cadence below are
        # complete, so a hand-staged directory of this cycle's files runs
        # the same chain every other mapped source runs.
        spacing_h = int(interval) // 3600
        fetch_command = "\n".join((
            f"# stage {args.source}'s bytes for cycle "
            f"{fetch_hints['cycle']} yourself into",
            f"#   {_printed_path(printed_out)}",
            f"#   (`woof fetch` has no download route for {args.source} "
            f"yet; the window this config",
            f"#    needs is {fetch_hints['area']}, "
            f"{fetch_hints['hours']} h at {spacing_h} h spacing)",
            f"#   `woof prep --show-source {args.source}` names the exact "
            f"products this route requires.",
            # The request and the supplement's own fetch line, from the
            # source's row, so the folder step 3 reads can be filled by
            # following step 1 alone.
            *local_staging_lines(
                args.source, cycle=fetch_hints["cycle"],
                hours=fetch_hints["hours"], cadence=cadence,
                start_hour=start_hour, area=fetch_hints["area"],
                out=_printed_path(printed_out))))
    # The BARE form is the next step, because it is the one that measures
    # this machine.  v1.4.0 printed only the declared form, and on a real
    # 16 GB card the two returned opposite verdicts on the same file --
    # rc 0 for the declared budget, rc 4 for the measured one -- because
    # the tier had sized against free VRAM the card does not have.  They
    # agree now (the tier is conservative against its class), and where
    # they cannot -- sizing for a card that is not in this machine -- the
    # declared form is printed beside it and labelled as such.
    check_command = f"woof check {_printed_path(out)}"
    check_command_declared = (f"woof check {_printed_path(out)} "
                              f"--free-gib {free_bytes / GIB:.17g} "
                              f"--vram-gib {vram_gib:g}")
    # `--explain` describes the route the emitted config takes; it does
    # not invent an input root.  A local-input source with no root has no
    # runnable next step, and handing one to `final_step_command` printed
    # `woof go` on a config that refuses at plan review for want of the
    # very directory this door did not write into its [fetch] table.
    printed_root = (printed_out if args.data_dir or local_source
                    else printed_out if explain and not local_source
                    else None)
    run_command = final_step_command(
        out, source=args.source, profile=profile,
        domain_count=len(dims), data_dir=printed_root,
        case_data=case_data, exp=exp, cycle=cycle,
        forecast_start_hour=start_hour)

    # The nocturnal declaration, SPOKEN.  `render_config` writes
    # acknowledgements = [ASYMMETRIC_RADIATION_NOCTURNAL_ACK] into the
    # emitted [experiment] whenever an explicitly selected asymmetric
    # suite meets a window with local night in it, and that line disarms
    # the load-time guard at every other front door for this file -- so
    # `woof check`, `woof run`, `woof go` and run-plan all fall
    # silent on a run PHYSICS.md classes as nocturnally invalid.  Until
    # now the only statement of that was a comment inside the emitted
    # TOML, which is not a warning: this door was measured emitting the
    # declaration with nothing on stdout or stderr, not even under
    # --explain.  warn() is the right voice twice over -- it never
    # blocks a deliberately selected validation suite, and it reaches
    # run-plan's structured warning sink, so a front end driving the
    # intent route sees the same sentence as a field.
    if declared_night is not None:
        warn(f"{out.name} is NOT NOCTURNALLY VALID: profile {profile} runs "
             f"shortwave with longwave OFF and this window includes local "
             f"night (first at {declared_night:%Y-%m-%dT%H:%M}Z).  The "
             f"emitted [experiment] declares "
             f'acknowledgements = ["{ASYMMETRIC_RADIATION_NOCTURNAL_ACK}", '
             f'"{CONSTANT_DOWNWARD_LONGWAVE_ACK}"] -- the first because '
             f"you passed --ack, the second because this suite fabricates "
             f"its downward longwave and the file would not load without "
             f"it -- so no later command will stop this run.  Re-emit "
             f"with a full lw+sw "
             f"profile (e.g. {MORRISON_PROFILE_ID}) for a forecast.",
             why="Shortwave heats the surface by day while no longwave "
                 "scheme runs, so after sunset the surface radiates to "
                 "space with no downward longwave to balance it: skin "
                 "temperature craters, the surface saturation humidity "
                 "collapses with it, and 2 m dewpoints read far below "
                 "the airmass.  See docs/public/PHYSICS.md, 'Nocturnal "
                 "validity'.")
    for note in area_notes:
        warn(note,
             why="forcing must cover the complete projected footprint and "
                 "its interpolation donors, including every longitude "
                 "when a polar stereographic grid encloses a pole.")
    for note in coverage_notes:
        # Advisory, never a refusal: the DOMAIN was already bounded by
        # source coverage during fitting, so only the margined bbox --
        # a projection artifact plus the fetch margin -- overran.
        warn(note,
             why="`woof fetch` validates --area against the source "
                 "grid's own lat/lon envelope and refuses a box naming "
                 "ground the grid does not carry; the clamped box still "
                 "contains the whole fitted domain, whose source "
                 "coverage was proven during fitting.")
    # Which flag the advisory may name depends on what stopped the fit
    # (`request_bound`, resolved above): when one of the request's own
    # bounds chose the size, the card is not the lever any more.
    for note in oversized_footprint_advisory(fetch_hints["area"],
                                             request_bound=request_bound):
        print(f"advisory: {note}")
    if args.source == "hrrr":
        for note in coverage_advisory(exp):
            print(f"advisory: {note}")
    # The cadence statement is LAYERED like every other advisory here:
    # the default screen gets a clause on the physics line it already
    # prints (zero lines -- the one-screen cap is a measured gate, a
    # user's next command went unfound under a 20-line wall), and
    # --explain gets the full one-line advisory with the mechanism and
    # the override path.
    chain_km = _ladder_dx_km(ratios, root_dx_m)
    shared = shared_physics(profile)
    cu_by_domain = cumulus_by_domain(
        dims, ratios, profile=profile, root_dx_m=root_dx_m,
        cumulus_requested=cumulus_requested)
    # The line describes the FILE, not the catalogue entry: the root's
    # emitted cumulus switch, which the grid may have retired.
    summary = (mix_physics_summary(text, physics_mix) if physics_mix else
               physics_summary(profile, cu_physics=cu_by_domain[0]))
    if len(dims) > 1:
        summary += (f" -- one radiation cadence: all {len(dims)} domains "
                    "run the root's radt, nests inherit it")
    print(f"physics: {summary}")
    if physics_mix:
        print(f"physics: {physics_mix_words(physics_mix['choices'])} replace the suite's own "
              "schemes in the file; no suite is asserted")
    elif explain and getattr(args, "physics_profile", None) is None:
        from woof.physics_menu import default_basis

        print(f"physics: {profile} is the default for this grid, "
              f"{default_basis(args.source, finest_spacing_m(root_dx_m, ratios), len(ratios) + 1)}")
    # A changed switch is reported on the DEFAULT screen, not behind
    # --explain: the user asked for a suite by not naming one, and the
    # emission moved one of its switches.
    speak = cumulus_retired_note if explain else cumulus_retired_headline
    for note in speak(profile, root_dx_m / 1000.0,
                      cumulus_requested=cumulus_requested):
        print(f"advisory: {note}")
    if explain:
        for line in prepared_route_physics_notice(profile, args.source):
            print(line)
        for note in radiation_cadence_advisory(profile, len(dims)):
            print(f"advisory: {note}")
        for note in gray_zone_advisory(chain_km, shared, les_below_m,
                                       les_closure):
            print(f"advisory: {note}")
        for note in cumulus_gray_zone_advisory(chain_km, cu_by_domain):
            print(f"advisory: {note}")
    else:
        # The headline says what was found; the emitted config carries
        # the whole advisory in its header comment either way, so the
        # reasoning is never only one flag away -- it is also in the
        # file, next to the settings it is about.
        for note in gray_zone_headline(chain_km, shared, les_below_m,
                                       les_closure):
            print(f"advisory: {note}")
        for note in cumulus_gray_zone_headline(chain_km, cu_by_domain):
            print(f"advisory: {note}")

    # ---- woof check, where it can accurately run ----------------------
    #
    # Run it BEFORE the next-steps block rather than after.  The block
    # is the last thing printed, on purpose: the field exhibit's whole
    # failure was a correct next command with output after it, which
    # reads as "and then this other thing happened" rather than "do
    # this".
    deferred: list[str] = []
    if local_source:
        deferred = [str(data_dir)]
        print(f"Local inputs: stage the declared series in {data_dir}, then run {check_command}.")
    elif case_data is None:
        if explain:
            # "fetched" only where a fetch exists; every other source's
            # bytes arrive by hand, and calling that a fetch is the same
            # false claim the omitted [fetch] table exists to avoid.
            arrival = ("fetched" if emitted_fetch_hints is not None
                       else "hand-staged")
            print(
                f"note: {arrival} {args.source.upper()} data feeds the "
                "rw-wps/gpuwm-wrf-init native initialization front door, "
                "not the [case_data] run path (the config-driven route "
                "decodes native GRIB1 = ERA5 today); this TOML's "
                "[experiment]/[[domain]]/[projection] tables are what "
                "that front door consumes.  `woof check` validates the "
                "geometry and memory preflight for this file; the native "
                "front door validates its own inputs.")
        # Pass the same free VRAM used by the fit planner. Reconstructing
        # it from an allocation budget changes the streamed route and verdict.
        rc = _check_emitted_config(out, sizing, target_machine=target_machine,
                                   remote_hardware=remote_hardware)
        if rc != 0:
            print(f"woof check FAILED (rc {rc}) on the emitted config.  "
                  "The files above were still written, so nothing is "
                  "lost: fix the gap that check names, then re-run "
                  "`woof check` on the same file.", flush=True)
            return rc
    else:
        deferred = _missing_case_inputs(out, case_data)
        if deferred:
            # Not a stanza of its own.  "woof check: deferred" followed
            # by an indented inventory reads as a failure report in the
            # middle of a success, which is how the field exhibit's
            # reader met it.  What it actually means is "step 2 comes
            # after step 1", and that is where it now says it.
            if explain:
                print("woof check: deferred -- declared inputs not on "
                      "disk yet:")
                for item in deferred:
                    print(f"  missing {item}")
                _print_geog_help()
        else:
            rc = _check_emitted_config(out, sizing, target_machine=target_machine,
                                       remote_hardware=remote_hardware)
            if rc != 0:
                print(f"woof check FAILED (rc {rc}) on the emitted "
                      "config.  The files above were still written, so "
                      "nothing is lost: fix the gap that check names, "
                      "then re-run `woof check` on the same file.",
                      flush=True)
                return rc

    # Preserve the caller's explicit terms in executable follow-up commands.
    # Otherwise a successful restricted-source plan prints a fetch/go that
    # immediately refuses the same source for missing acceptance.
    licence_flags = "".join(
        f" --accept-licence {shlex.quote(licence)}"
        for licence in (getattr(args, "accept_licence", ()) or ()))
    if licence_flags:
        if fetch_command.startswith("woof fetch "):
            first, *remaining = fetch_command.splitlines()
            fetch_command = "\n".join((first + licence_flags, *remaining))
        if run_command.startswith("woof go "):
            run_command += licence_flags
    _print_next_steps(fetch_command, check_command, run_command,
                      source=args.source, deferred=bool(deferred),
                      explain=explain,
                      check_command_declared=check_command_declared)
    return 0


def _print_next_steps(fetch_command: str, check_command: str,
                      run_command: str, *, source: str, deferred: bool,
                      explain: bool,
                      check_command_declared: str | None = None) -> None:
    """End with one launch command when it owns acquisition and preparation.

    The ordinary automatic route prints a preview and one launch command.
    Manual acquisition routes and ``--explain`` retain the individual
    acquisition, check and run steps. Credential instructions come from
    the source registry and appear before the next command when needed.
    """

    # The CDS key is the CDS transport's; a fetch from the keyless ARCO
    # store needs none, and telling that user to go and get one sends them
    # to register for an account the command never reads.
    keyless = "--era5-provider arco" in fetch_command
    if (not explain and run_command.startswith("woof go ")
            and fetch_command.startswith("woof fetch ")):
        print("\nConfiguration created. Review it before launching; "
              "Run checks current GPU memory before downloading.")
        for note in ([] if keyless else source_credential_notes(source)):
            print(note)
        print("Preview the plan without downloading or running:")
        print(f"  {run_command} --dry-run")
        print("\nnext:")
        print(f"  {run_command}")
        print("  Fetches inputs, prepares them, runs the forecast and renders products.")
        return

    if not explain:
        # Before the block, never after it: the numbered steps are the
        # last thing on the screen, so the eye lands on step 1.
        print("Re-run with --explain for the sizing table, the physics "
              "notes and the full advisories.")
    print("")
    print("next:")
    # Step 1 is a command where `woof fetch` has a route and a short
    # acquisition note where it does not.  Printing a `woof fetch
    # --source <x>` line for a source the fetch door refuses is the exact
    # numbered-list-to-a-refusal shape this block exists to end.
    acquire = fetch_command.splitlines()
    print(f"  1. {acquire[0]}")
    for line in acquire[1:]:
        print(f"     {line}")
    for note in ([] if keyless else source_credential_notes(source)):
        print(f"     {note}")
    print(f"  2. {check_command}"
          + ("   # after the fetch lands" if deferred else ""))
    if check_command_declared:
        # The bare form above measures THIS machine.  Say, once, what the
        # other form is for -- v1.4.0 printed only the declared form and
        # a reader who ran the documented bare one got the opposite exit
        # code with no explanation of why two commands disagreed.
        print(f"     # that measures THIS machine's free VRAM.  Sizing "
              f"for a card that is not in it (or no GPU here at all):")
        print(f"     #   {check_command_declared}")
    # Step 3 is a command for most configs and a short route note for
    # the ones no single command finishes; either way it is indented
    # into the numbered block rather than trailing off the end of it.
    lines = run_command.split("\n")
    print(f"  3. {lines[0]}")
    for line in lines[1:]:
        print(f"     {line}")


def register_cli(subparsers) -> None:
    from woof.starter_template import register_cli as register_template_cli
    register_template_cli(subparsers)
    parser = subparsers.add_parser(
        "domain",
        help="wizard: emit an experiment TOML for a point or polygon + GPU budget, "
             "sized by the in-process VRAM estimator")
    # Acquisition extensions belong to registry rows, including a route's
    # model key and generated-member roster. The emitted fetch table and
    # the source's ordinary CLI consume these same declarations.
    for adapter in source_adapters():
        for name, kind, default, description in adapter.acquisition_hints:
            flag = "--" + name.replace("_", "-")
            if flag not in parser._option_string_actions:
                parser.add_argument(flag, type={"string": str, "integer": int,
                    "number": float}[kind], default=None, help=description)
    # Not required at parse time: --recipe takes --region or --polygon
    # (or neither, for its row's own declared region), never --point, and
    # region_recipe.domain_recipe_main names exactly what is missing.  An
    # argparse-required group refused a bare --recipe with "one of the
    # arguments --region --point --polygon is required" before that check
    # could speak.  domain_main checks the target for the plain wizard.
    target = parser.add_mutually_exclusive_group(required=False)
    target.add_argument("--region", metavar="ID|GEOJSON",
                        help="with --recipe: region row id (e.g. US) or local "
                             "polygon GeoJSON the recipe is laid out on; "
                             "requires --spacing-km")
    parser.add_argument("--recipe", metavar="ID|TOML",
                        help="complete recipe from the packaged table or a local "
                             "TOML, laid out on --region (or --polygon) at "
                             "--spacing-km; a packaged row given neither uses "
                             "its declared default region and spacing "
                             "(hrrr_v4_gsd41: US at 3 km), stated in the "
                             "output and the receipt")
    from woof.smoke_execution import SMOKE_FLAG_HELP
    parser.add_argument("--smoke", action="store_true", default=False,
                        help="with --recipe: " + SMOKE_FLAG_HELP)
    parser.add_argument("--region-table", type=Path, default=None,
                        help="region JSON table override; adding a region requires data only")
    parser.add_argument("--recipe-table", type=Path, default=None,
                        help="recipe registry override; new recipes and aerosol policies are data")
    parser.add_argument("--spacing-km", type=positive_float, default=None,
                        help="with --recipe: grid spacing in kilometres; "
                             "required whenever --region or --polygon is given "
                             "with --recipe")
    target.add_argument("--point", metavar="LAT,LON",
                        help="domain center in decimal degrees. |lat| 90 "
                             "is refused. A point carries no extent, so "
                             "the fit chooses one: the largest layout "
                             "the budget affords, capped at "
                             "--point-extent-km per axis, "
                             "kept within the projection envelope, and "
                             "kept to less than one trip around the "
                             "globe in longitude. These "
                             "caps SHRINK the domain rather than refuse "
                             "it, and the plan summary states which one "
                             "bound; the "
                             "pole refusal is left for a center so close "
                             "to one that even the smallest layout "
                             "contains it. Draw a --polygon to ask for "
                             "more ground than the cap. The projection "
                             "is auto-selected from |lat| (<25 Mercator, "
                             "25-60 Lambert conformal, >60 polar "
                             "stereographic) unless --projection is set. "
                             "Negative (southern/western) values work in "
                             "both forms: --point -33.87,151.21 and "
                             "--point=-33.87,151.21")
    parser.add_argument(
        "--point-extent-km", type=point_extent_argument,
        default=POINT_FIT_MAX_EXTENT_KM, metavar="KM",
        help="largest root extent a --point request is sized to, on the "
             "longer root axis (x); the root keeps the ladder's 5:4 shape, "
             "so y is 0.8 of it "
             f"(default {POINT_FIT_MAX_EXTENT_KM:.0f}).  The projection "
             "pole, one trip around the globe, the source's coverage and "
             "the card still bound the fit; an extent below the smallest "
             "root the ladder hosts gets that root.  The plan summary "
             "states the extent used")
    target.add_argument(
        "--polygon", type=Path, metavar="GEOJSON",
        help="local GeoJSON Polygon, MultiPolygon, Feature, or "
             "FeatureCollection; the minimum antimeridian-aware bounds "
             "supply the center and every emitted level is fitted around "
             "the geometry")
    parser.add_argument(
        "--buffer-km", default=None, metavar="KM[,KM...]",
        help="with --polygon, nonnegative geometry buffer in kilometres; "
             "one value applies to every domain, or supply exactly one "
             "outer-to-inner value per level.  Every value is measured "
             "from the polygon itself, not from the next inner grid: "
             "'800,300,0' puts the outer grid 800 km from the polygon, "
             "about 500 km beyond the middle one. With --ladder auto, a "
             "multi-value list selects the preset of that depth "
             "(default: zero)")
    parser.add_argument("--projection", default="auto",
                        choices=("auto", "lambert", "mercator", "polar"),
                        help="map projection override (default: auto by "
                             "center latitude; all three are oracle-gated "
                             "against WRF v4.6.1 module_llxy)")
    parser.add_argument("--name", default=None,
                        help="experiment name (default derived from the "
                             "center)")
    parser.add_argument("--card", default=None,
                        help="GPU to size for: a tier (12gb/16gb/24gb/32gb), "
                             "a size ('10gb'), or a model with a recorded "
                             "size ('RTX 3080', '5070 Ti'); sets the VRAM "
                             "budget with no local probe. With no --card, --vram-gib or "
                             "--hardware-json the wizard MEASURES the local "
                             "card's capacity (short-lived probe, "
                             "suppressed by GPUWM_NO_LOCAL_GPU) and "
                             "refuses, naming both flags, when there is "
                             "nothing to measure")
    parser.add_argument("--vram-gib", type=float, default=None,
                        metavar="N",
                        help="total VRAM in GiB (alternative to --card)")
    parser.add_argument("--hardware-json", type=Path,
                        help="selected target hardware snapshot with measured GPU capacity, available memory and device profile; no local GPU probe")
    parser.add_argument("--target-host-memory-json", type=Path,
                        help="selected target host-memory snapshot for an explicit --card or --vram-gib budget; no local RAM sizing")
    parser.add_argument("--nz", type=int, default=None, metavar="N",
                        help="vertical mass levels (default: 49); resamples "
                             "the default eta ladder while preserving its stretching")
    parser.add_argument("--tiles", nargs="?", const="auto", default=None,
                        choices=("off", "auto", "on"),
                        help="streaming mode (bare --tiles means auto); sizes "
                             "with the forecast planner using the selected "
                             "target's GPU and RAM when supplied, otherwise "
                             "local hardware or an explicit card budget; "
                             "on forces streaming")
    parser.add_argument("--ladder", default=None,
                        choices=(*LADDER_RATIOS, "auto"),
                        help="preset nest dx chain in km (default: 12 -- "
                             "one 12 km domain, the shape `woof go` runs "
                             "end to end, same as the interactive "
                             "session).  Nest trees are explicit opt-in: "
                             "a deeper preset, `auto` (the deepest preset "
                             "that fits the card), or --root-dx / --chain "
                             "for anything else; their closing block "
                             "names the tree runner they route to")
    parser.add_argument("--physics-profile", default=None, type=canonical_template_id,
                        choices=WIZARD_PHYSICS_PROFILES,
                        help="shipped physics suite to emit; taken verbatim "
                             "from the registry the prepared-forecast "
                             "runner validates against, so the emitted "
                             "config passes its guard as written.  Read "
                             "the resolved radiation selectors: a suite with "
                             "shortwave ON and longwave OFF is a daytime-only "
                             "experiment; "
                             "selecting it for a window that "
                             "includes local night is REFUSED unless you "
                             "declare it yourself with --ack.  "
                             + _profile_help_route_note()
                             + _profile_help_default_note())
    parser.add_argument("--cumulus", default=None, choices=("suite", "grid"),
                        help="who decides the root's cumulus scheme.  "
                             "suite: the suite's own scheme at any spacing "
                             "(what naming --physics-profile means when "
                             "this is left out).  grid: the suite's scheme "
                             "is turned off on a root finer than the "
                             f"{CUMULUS_CONVECTION_PERMITTING_DX_KM:g} km "
                             "convection-permitting bound, where the "
                             "dynamics resolve deep convection (what an "
                             "unnamed suite gets)")
    parser.add_argument(
        "--physics-choices", default=None, metavar="JSON",
        help="schemes to run in place of the suite's own, family by "
             "family, as the physics composer picks them: "
             "'{\"microphysics\": \"thompson-mp8\", \"pbl\": \"myj\", "
             "\"surface_layer\": \"eta-similarity\"}'.  Checked by the "
             "engine the way `woof physics-catalog --check` checks them "
             "and written into the config the way `--into` writes them, "
             "on every size the fit tries, so the card is priced for the "
             "schemes that run.  The suite (--physics-profile, or the "
             "default at the finest grid) is the base the choices change; no suite "
             "is asserted, so a mix no named suite matches runs as "
             "written")
    parser.add_argument(
        "--ack", action="append", default=[], metavar="ID",
        help="declare a governed experiment, written verbatim into the "
             "emitted [experiment].acknowledgements.  Repeatable.  This "
             "door used to write the nocturnal declaration for you, which "
             "silenced the load guard at check/run/go/run-plan and both "
             "prepared runners for the life of the file; it no longer "
             "does, and refuses instead.  The id it accepts is "
             + ASYMMETRIC_RADIATION_NOCTURNAL_ACK
             + ": a longwave-OFF suite over a window that includes local "
             "night, which you are running deliberately as a "
             "daytime-only experiment")
    parser.add_argument("--root-dx", type=float, default=None,
                        metavar="KM",
                        help="custom root grid spacing in km "
                             f"[{MIN_ROOT_DX_KM:g}, {MAX_ROOT_DX_KM:g}]; "
                             "use with --chain instead of --ladder")
    parser.add_argument("--chain", default=None, metavar="R1,R2,...",
                        help="custom nest refinement ratios, integers in "
                             f"[{MIN_CHAIN_RATIO}, {MAX_CHAIN_RATIO}] "
                             "(e.g. --root-dx 3 --chain 4 for 3 km -> "
                             "750 m); omit for a single domain at "
                             "--root-dx.  Sized by the same estimator fit "
                             "loop as the presets")
    parser.add_argument(
        "--history-interval", type=positive_float, default=None, metavar="SECONDS",
        help="how often the ROOT domain writes a wrfout, in seconds "
             f"(default {DEFAULT_ROOT_HISTORY_INTERVAL_S:g}).  Must be a "
             "whole number of seconds and a whole number of that "
             "domain's time steps -- the loader checks both against the "
             "exact rational dt and refuses the emitted file otherwise, "
             "before it is written")
    parser.add_argument(
        "--nest-history-interval", type=positive_float_list, default=None,
        metavar="SECONDS[,SECONDS...]",
        help="the same, for the NESTED domains (default "
             f"{DEFAULT_NEST_HISTORY_INTERVAL_S:g}): one value for every "
             "nest, or one per nest, outer to inner (1800,600 is d02 "
             "every 30 min and d03 every 10 min).  Nests write more "
             "often than the root by default because resolving what the "
             "root cannot, over a shorter window, is the point of "
             "running one.  Ignored for a single-domain ladder")
    parser.add_argument(
        "--quicklook-interval", type=positive_float, default=None,
        metavar="SECONDS",
        help="a quick-look stream on the FINEST domain: a small subset "
             "(--quicklook-vars) every SECONDS between its full frames, "
             "written to quicklook/wrfout_dNN_<time>.  Must divide that "
             "domain's history interval.  Measured 13 B a cell against "
             "172 for a full frame")
    parser.add_argument(
        "--quicklook-vars", default=None, metavar="VAR[,VAR...]",
        type=lambda text: tuple(v.strip() for v in text.split(",") if v.strip()),
        help="the quick-look subset's wrfout variables (default: "
             "REFL_10CM, UP_HELI_MAX, U10, V10, T2, Q2, PSFC, RAINNC, RAINC)")
    parser.add_argument(
        "--les-below-m", type=float, default=DEFAULT_LES_BELOW_M,
        metavar="METRES",
        help="write the resolved-turbulence closure (PBL off, a 3-D "
             "closure, isotropic mixing) on every real-data domain at or "
             f"below this spacing (default {DEFAULT_LES_BELOW_M:g} m; 0 never)")
    parser.add_argument(
        "--les-closure", choices=tuple(sorted(LES_CLOSURES)),
        default=DEFAULT_LES_CLOSURE,
        help="the 3-D closure an LES domain runs: smagorinsky (km_opt 3, "
             "the default) or tke (km_opt 2, 1.5-order prognostic TKE)")
    parser.add_argument(
        "--les-inflow-seeding", choices=("on", "off"), default="on",
        help="cell perturbations on the inflow faces of an LES nest whose "
             "parent runs a 1-D PBL (default on)")
    parser.add_argument("--sf-surface-mosaic", type=int, choices=(0, 1), default=None,
                        help="Noah land-use tiles on every grid (WRF sf_surface_mosaic)")
    parser.add_argument("--mosaic-cat", type=int, default=None,
                        help="Noah mosaic tile count on every grid (WRF mosaic_cat)")
    from woof.config import MOSAIC_URBAN_CANOPY_RULES
    parser.add_argument(
        "--mosaic-urban-canopy", choices=tuple(MOSAIC_URBAN_CANOPY_RULES),
        default=None,
        help="where Noah mosaic runs the urban canopy with sf_urban_physics "
             "= 1: dominant (WRF's rule, the default: only cells that are "
             "mostly urban) or every_tile (also the town tiles of mostly "
             "rural cells)")
    parser.add_argument(
        "--isftcflx", type=int, choices=(0, 1, 2), default=None,
        help="surface flux over water on every grid (WRF isftcflx): "
             "0 standard MM5 roughness, 1 Donelan drag with constant heat "
             "and moisture roughness (the tropical cyclone option), 2 "
             "Donelan drag with Garratt heat and moisture roughness.  "
             "Default: the suite's own (0)")
    parser.add_argument(
        "--clock", choices=CLOCK_CHOICES, default=DEFAULT_CLOCK,
        help="how the run steps.  adaptive: each grid starts at the "
             "emitted time_step and then follows its own Courant number "
             "between 3 and 8 s per km of its spacing (WRF's "
             "use_adaptive_time_step, written into [shared]); the "
             "terrain clock still caps it over steep ground at launch.  "
             "fixed: one step throughout.  auto (default): adaptive when "
             "every grid starts inside those bounds and lies within the "
             "500 m to 12 km spacings the terrain clock measured, fixed "
             "otherwise (the tropical clock's 2.5 s per km is always "
             "fixed).  adaptive on a grid outside the bounds is refused, "
             "naming the grid")
    parser.add_argument("--hours", type=int, default=6, metavar="N",
                        help="forecast length (run_seconds = N*3600)")
    parser.add_argument(
        "--source", default=DEFAULT_WIZARD_SOURCE, metavar="SOURCE",
        action=_DeclaredSourceAction,
        # NO `choices=`, deliberately.  This used to be
        # choices=("gfs", "hrrr", "era5") while the product shipped
        # sixteen runnable sources, so argparse answered `--source rap`
        # with "invalid choice" -- a refusal that says nothing about RAP.
        # The registry answers instead (`resolve_source`), which lets an
        # alias resolve, a registered-but-unrunnable row explain itself,
        # and a new row reach this door with no edit here.
        help="forcing source: any registered source id or alias -- "
             + ", ".join(wizard_planable_source_ids())
             + " today, plus "
             + ", ".join(wizard_cadence_named_source_ids())
             + " with --cadence naming the spacing of the history files "
               "it will be handed (`woof prep --list-sources` lists the "
               "whole registry).  It "
               "sets the boundary cadence written into the companion "
               "namelist.wps, bounds the domain by the source's own grid "
               "where that grid is regional, and (era5) declares "
               "[case_data].  A source `woof fetch` cannot download "
               "still emits the same geometry: one whose registry row "
               "declares a local input contract gets a [fetch] table "
               "(source, cycle, hours and its staging source_root) "
               "with the staging step named beside it, and any other has "
               "the acquisition step named in place of the table")
    parser.add_argument("--cycle", required=True,
                        metavar="YYYY-MM-DDTHH|latest",
                        help="the forcing CYCLE (UTC), which is the run's "
                             "start time unless --forecast-start-hour "
                             "moves it; 'latest' probes the public mirrors "
                             "for the newest complete gfs/hrrr cycle "
                             "covering the whole window and prints what it "
                             "picked; sources without a probe use their declared publication delay)")
    parser.add_argument("--era5-product", choices=("reanalysis", "ensemble_members"), default=None,
                        help="explicit ERA5 product; default reanalysis has no member axis")
    parser.add_argument("--era5-provider", choices=("cds", "arco"), default=None,
                        help="ERA5 provider (default: cds when a Copernicus CDS key is present, otherwise the keyless arco store); ensemble_members requires CDS")
    parser.add_argument("--cadence", type=positive_int, default=None, metavar="HOURS",
                        help="boundary spacing in whole hours, validated against the selected product; "
                             "required for a source that declares no cadence of its own "
                             f"({', '.join(wizard_cadence_named_source_ids())}), where it is "
                             "the spacing of the history files handed to woof prep")
    parser.add_argument("--member", default=None,
                        help="ensemble trajectory member; defaults to the route's control")
    parser.add_argument("--forecast-start-hour", type=int, default=None,
                        metavar="K",
                        help="initialize the run from the cycle's f{K} "
                             "FORECAST lead instead of its analysis, so "
                             "start_time = cycle + K h and the boundaries "
                             "come from f{K+i}.  This is how a window deep "
                             "in a forecast (say f174..f240) is reached "
                             "without integrating from f000.  Every source "
                             "whose registry row publishes forecast leads "
                             "takes it; a row that declares none refuses it "
                             "by name.  The initial condition is then itself "
                             "a K-hour forecast, and every receipt says so")
    parser.add_argument("--out", type=Path, required=True, metavar="TOML",
                        help="emitted experiment TOML path")
    parser.add_argument("--data-dir", default=None, metavar="DIR",
                        help="explicit forcing directory; automatic go launches "
                             "otherwise manage request-specific downloads. Manual "
                             "acquisition and ERA5 paths default to data/<name>")
    parser.add_argument("--forcing", nargs="+", default=None,
                        metavar="GRIB",
                        help="era5: explicit forcing GRIB path(s) already "
                             "on disk (default <data-dir>/"
                             "era5-combined.grib)")
    parser.add_argument("--vtable", type=Path, default=None,
                        help="era5: Vtable override (default: the "
                             "packaged Vtable.ERA5_CDO, copied beside "
                             "the TOML)")
    parser.add_argument("--terrain-smoothing", default=None, metavar="SPEC",
                        help="WPS terrain smoothing per domain, in domain "
                             "order, the last repeating: none, 1-2-1, "
                             "smth-desmth or smth-desmth_special, each "
                             "with an optional :PASSES (e.g. none or "
                             "smth-desmth_special,none); default: WPS's "
                             "one smth-desmth_special pass")
    parser.add_argument("--terrain-smoothing-precision", default=None,
                        choices=("float64", "wps-float32"),
                        help="arithmetic of every domain whose terrain "
                             "smoother is WPS's default smth-desmth_special "
                             "x1: wps-float32 reproduces geogrid.exe's "
                             "HGT_M exactly; default float64, WOOF's own "
                             "smoother. Every other smoother always runs "
                             "WPS's float32")
    parser.add_argument("--geog-root", type=Path, default=None,
                        metavar="DIR",
                        help="staged WPS_GEOG tree, saved for every source (default "
                             "${GPUWM_CASE_DATA_ROOT}/WPS_GEOG)")
    parser.set_defaults(func=domain_main)
    return parser


__all__ = [
    "CARD_VRAM_GIB", "CLOCK_CHOICES", "CUMULUS_CONVECTION_PERMITTING_DX_KM",
    "DEFAULT_CLOCK", "clock_decision",
    "CUMULUS_GRAY_ZONE_TOP_DX_KM",
    "DEFAULT_LADDER", "DEFAULT_WIZARD_SOURCE",
    "DomainFitError", "GEOG_DATASETS", "GRAY_ZONE_DX_KM",
    "LADDER_RATIOS", "MAX_FETCH_ABS_LAT", "POLE_CLEARANCE_DEG",
    "ROOT_DX_M", "ROOT_TIME_STEP_S", "TROPICAL_ROOT_TIME_STEP_S",
    "convection_permitting",
    "cumulus_by_domain", "cumulus_gray_zone_advisory",
    "cumulus_gray_zone_headline", "cumulus_retired_headline", "root_cumulus",
    "cumulus_retired_note", "cumulus_requested_by", "declared_nocturnal_night",
    "domain_main", "experiment_from_text", "fetch_window", "final_step_command",
    "fit_ladder", "fit_memory", "fit_polygon_ladder", "gray_zone_advisory",
    "load_polygon_footprint", "max_fetch_abs_lat",
    "oversized_footprint_advisory", "parse_chain",
    "parse_custom_ladder", "parse_level_buffers", "physics_mix_request",
    "physics_mix_words", "pole_clearance_deg", "with_physics_mix",
    "POINT_FIT_EXTENT_SCOPE", "POINT_FIT_MAX_EXTENT_KM",
    "POINT_FIT_PROJECTION_SCOPE", "POINT_FIT_SCOPES",
    "POINT_FIT_BAND_SCOPE", "POINT_FIT_FLOOR_SCOPE",
    "point_extent_argument", "point_extent_note", "point_fit_cap_note",
    "point_request_bound",
    "refuse_point_extent_on_polygon",
    "polygon_ladder_dims", "radiation_cadence_advisory",
    "radt_ladder_minutes", "register_cli", "render_config",
    "render_wps_namelist", "root_time_step_s", "seconds_per_km",
    "verify_polygon_containment", "vram_reserve_gib",
    "CARD_UNAVAILABLE_VRAM_GIB", "CARD_UNAVAILABLE_VRAM_FRACTION",
    "FIT_HEADROOM_FRACTION",
    "FIT_HEADROOM_MIN_BYTES", "card_assumed_free_gib",
    "fit_headroom_bytes", "sizing_budget_bytes",
]
