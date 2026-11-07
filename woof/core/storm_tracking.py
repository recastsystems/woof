"""Storm tracking for the storm-following moving nest: the WHEN and WHERE.

:mod:`woof.core.nest_relocation` is the mechanism half of a moving nest
and says so in its own docstring: *"it does not decide when or where to
move -- there is no tracker, no scoring, no hysteresis here.  Placement is
an input."*  This module is that missing half.  It watches the RUNNING
MODEL'S OWN FIELDS on the parent domain and turns them into discrete
whole-parent-cell shift proposals a relocation runner can hand to
:func:`woof.core.nest_relocation.relocate_child`.

THE SEAM (the plan-provider contract).  The runner side of the moving-nest
program consumes a tracker through one callable, evaluated once per
relocation cadence (a cycle boundary, never mid-step)::

    desired_shift(parent_state, nest_footprint, t) -> (di, dj) | None

``parent_state`` is the PARENT domain's live state (host or device);
``nest_footprint`` is where the child currently sits, as a
:class:`NestFootprint` or anything carrying the five
:class:`woof.experiment.DomainConfig` placement/extent attributes;
``t`` is model seconds since the experiment start (model time, not
wallclock, so the hysteresis is deterministic and restart-independent).
The return is the desired move in WHOLE PARENT CELLS -- ``di`` along the
x/i axis (the LAST array axis; positive means a larger ``i_parent_start``)
and ``dj`` along y/j (axis -2, positive means larger ``j_parent_start``)
-- or ``None`` for "hold".  The runner applies it as
``i_parent_start + di`` / ``j_parent_start + dj`` and the leg-1
admissibility bounds (:func:`woof.core.nest_relocation.check_admissible`)
remain the runner's to enforce; a tracker proposes, it never moves
anything.  :class:`StormTracker` implements the contract and ``__call__``
aliases it, so the provider is literally "a callable returning the desired
whole-parent-cell shift each cadence".  One optional duck-typed hook
completes the seam: the runner calls
:meth:`StormTracker.notify_move_executed` with the shift it really
applied after each successful relocation, and from then on the cooldown
counts from EXECUTED moves; a runner that never calls it leaves the
proposal-burn fallback in force.

WHAT IT TRACKS.  Primary signal: a column updraft-helicity running-max
plane, folded every step by :mod:`woof.core.uh_diag` from WRF's own
cal_helicity columns.  It is THIS CONSUMER'S window
(``uh_diag.UH_FOLLOW_WINDOW_SLOT``), reset by the relocation runner at
every evaluation, so it is exactly "the strongest rotation since the
tracker last had a chance to move".

It is deliberately NOT WRF's UP_HELI_MAX diagnostic, which is folded from
the same columns in the same pass but zeroed by the HISTORY WRITER.
Reading that one made a nest's placement a function of
``history_interval_s``: an output knob silently steering the model
(closed 2026-08-07).  The window is a max over the interval rather than
an instantaneous sample because UH is spiky -- sampling it at cadence
boundaries would alias transient mesocyclone pulses, and a threshold
cannot un-alias a signal.

A rotating storm is what a tornado nest exists to follow, so rotation is
the primary vote.  Fallback:
the column-max (composite) simulated reflectivity from the persistent
``refl_10cm`` slot, for the window before the storm rotates -- an echo
centroid keeps the nest on the convection until a mesocyclone exists to
take over.  The centroid arithmetic is the WaH survey's proven idiom
(``echo_stats`` / ``motion_from_centroids`` in tools/da_nowcast.py):
threshold, then a weighted centroid of the exceedance -- adapted here to
grid index space rather than radar gate space, and kept in core so the
model never imports from tools.

WHY HYSTERESIS.  A centroid jitters: convective cells pulse, split and
merge, and a nest that re-centers on every wobble spends its life paying
relocation spin-up strips for moves that bought nothing.  Three guards,
all configured, all logged when they fire:

- dead-band: a proposed shift below ``min_shift_cells`` (Chebyshev norm)
  is suppressed -- the storm has not really left the middle of the nest;
- clamp: a shift is limited to ``max_shift_cells`` per axis per event, so
  one bad centroid (or one real jump) cannot trade away most of the
  overlap transplant in a single move;
- cooldown: after a proposal, further proposals are suppressed for
  ``cooldown_seconds`` of model time, so the donor tables are not rebuilt
  every cadence while a storm crosses cells at speed.

Every call appends a receipt carrying the centroid evidence and the
decision, so a run's move/hold history is auditable after the fact.

CONFIG.  ``[relocation.follow]`` under the leg-1 ``[relocation]`` table
(:func:`woof.experiment._build_relocation` wires it): ``field``
(``"uh"`` | ``"reflectivity"`` | ``"pressure"``), ``threshold`` (m2 s-2
for uh, dBZ for reflectivity, and under ``"pressure"`` either metres of
geopotential height above the search box's minimum or an absolute hPa
sea-level ceiling -- which one is decided by ``level_hpa``),
``fallback_threshold`` (dBZ; required with ``field = "uh"``, refused
otherwise -- a m2 s-2 number cannot be reused as a dBZ one),
``search_margin_cells``, ``min_shift_cells``, ``max_shift_cells``,
``cooldown_seconds``.  Optional: ``level_hpa``, ``radius_km``,
``refine_grid_id``.  Unknown keys refuse, and the constructed config
echoes its values (:meth:`FollowConfig.to_json`).

WHICH SURFACE, since it is the one default that decides what the
threshold MEANS.  Under ``field = "pressure"`` an absent ``level_hpa``
is ``DEFAULT_LEVEL_HPA`` (850 hPa), the low-level circulation centre a
nest should be centred on.  ``level_hpa = 0`` (``SEA_LEVEL_HPA``) is the
sea-level-pressure tracker, and is the only form whose threshold is an
absolute hPa ceiling.  The two threshold bands are disjoint by
construction, so a config that means one and is read as the other
refuses at load rather than tracking the wrong thing.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace

import numpy as np

from woof.core import streaming, uh_diag
from woof.core.attribute_tracking import (ATTRIBUTE_KEYS, attribute_metadata,
                                           attribute_plane, validate_attribute_config)

#: Versioned label carried by every receipt this module emits.
FOLLOW_CONTRACT = "gpuwm-storm-follow.v1"

#: The signals the tracker knows how to read off a parent state.
TRACKED_FIELDS = ("uh", "reflectivity", "pressure", "attribute")

#: Fields whose plane is a MICROPHYSICS STASH rather than something the
#: tracker can derive on demand.  Only these are subject to the
#: history-cadence contract enforced at admission
#: (woof.experiment._refuse_unservable_follow_cadence): ``refl_10cm``
#: exists only at history instants.  ``"pressure"`` is deliberately NOT
#: here -- it is reduced from the live prognostic state, which is valid
#: at every cycle boundary, so a pressure tracker's cadence is free of
#: the output knob entirely.
STASH_BACKED_FIELDS = ("uh", "reflectivity")

#: Plausibility band (hPa) for a ``field = "pressure"`` threshold.  The
#: same band woof.verify.obs.contracts uses for an mslp observation,
#: in hPa rather than Pa.  A threshold outside it is a units error --
#: almost always Pa typed into an hPa knob -- and is refused at
#: admission rather than silently matching every cell or none.
PRESSURE_THRESHOLD_MIN_HPA = 800.0
PRESSURE_THRESHOLD_MAX_HPA = 1100.0

#: Nine-point smoother applied to MSLP before the minimum is taken.
#: WRF's moving-nest vortex finder smooths before it searches for the
#: same reason: an unsmoothed grid-scale pressure dimple can outrank the
#: real centre and make the nest chatter.  Reuses the Shuman 1957
#: parameters woof.core.mslp already applies to MSLP for display
#: (three passes, centre weight 4), so both paths smooth identically.
PRESSURE_SMOOTH_PASSES = 3

#: Admissible ``level_hpa`` for the isobaric tracker.  Bounded well inside
#: a troposphere so a Pa value (85000) or a model-level index (12) refuses
#: at admission rather than tracking whatever the clamp happened to reach.
LEVEL_HPA_MIN = 200.0
LEVEL_HPA_MAX = 1000.0

#: The surface a ``field = "pressure"`` tracker uses when the config
#: names none.  850 hPa is the low-level circulation centre -- the one a
#: nest should be centred on -- and it is what every validated config in
#: this campaign asks for by hand.  It is a DEFAULT and not a fixed
#: choice: ``level_hpa`` still names any surface in the band above, and
#: ``level_hpa = 0`` is the sea-level reduction (see SEA_LEVEL_HPA).
#:
#: The measured reason to prefer it over the sea-level default it
#: replaces: the MSLP reduction is an EXTRAPOLATION below ground, so over
#: terrain it carries the terrain's own signature into the field steering
#: the nest, and its threshold is an absolute ceiling that can match every
#: cell or none as the storm deepens.  The 850 hPa height field is a
#: prognostic surface with a threshold measured from the search box's own
#: minimum, so it cannot go blind that way.
DEFAULT_LEVEL_HPA = 850.0

#: ``level_hpa = 0`` -- track on SEA LEVEL, i.e. the MSLP reduction that
#: an absent ``level_hpa`` used to mean.  Spelled as a number in the same
#: key rather than as a separate field, because it is the same question
#: ("which surface?") with the one answer that is not an isobaric
#: surface.  Zero can never collide with a real choice: the admissible
#: band starts at LEVEL_HPA_MIN.  Under it the threshold reverts to an
#: absolute hPa ceiling, which is what makes the two forms impossible to
#: mix up -- the bands are disjoint.
SEA_LEVEL_HPA = 0.0

#: How many isobaric surfaces may be named.  THERE IS NO LIMIT, and the
#: absence is deliberate: the cap here used to be 8, justified as "a
#: deep-layer mean does not get better past a handful of levels", which
#: is true of STEERING and was never true of REPORTING -- a vertical
#: profile of where the vortex sits is exactly what more surfaces buy.
#:
#: What made the cap defensible was an unmeasured fear of the cost.
#: Measured, on this card at 49x378x378: one surface is 2.28 ms and the
#: scaling is FLAT -- 3 surfaces 8.0 ms, 8 surfaces 20.6 ms, 20 surfaces
#: 45 ms, 37 surfaces 101 ms, all within noise of 2.3 ms each.  Against
#: the relocation cadence a moving nest actually runs (360 s), a 3-day
#: forecast consults 720 times: 20 surfaces is 32 seconds, or 0.15% of
#: the run.  A limit that costs a user their vertical profile to save
#: 0.15% is not a limit worth having.
#:
#: Each surface still has to be INSIDE the column (LEVEL_HPA_MIN..MAX,
#: and refused per level when it is underground everywhere), which is the
#: bound that means something.
MAX_TRACKED_LEVELS = None

#: Admissible ``threshold`` under ``level_hpa``, in METRES of geopotential
#: height above the search box's own minimum.  A vortex core on an
#: isobaric surface is tens of metres deep, so 1-500 m brackets every
#: usable choice and rejects an hPa ceiling (1000) pasted in by habit.
LEVEL_DEFICIT_MIN_M = 1.0
LEVEL_DEFICIT_MAX_M = 500.0

#: WRF's g (share/module_model_constants.F), as the MSLP reduction uses.
GRAVITY_M_S2 = 9.80665

#: Scratch slots the signals live in (woof.core.uh_diag /
#: woof.core.refl).  Literal names: the preflight scratch registry and
#: this module must agree on them.
#:
#: The UH signal is a CONSUMER-OWNED window, never WRF's UP_HELI_MAX.
#: The diagnostic's own accumulator is zeroed by the history writer, so
#: reading it made this tracker's decisions a function of the output
#: cadence -- change ``history_interval_s`` and the nest goes somewhere
#: else.  ``uh_follow_window`` is folded from the same columns in the
#: same pass and is reset by THIS consumer at every evaluation, so its
#: window is exactly one relocation cadence (ArWen's ruling, 2026-08-07).
#: A caller that owns a different evaluation rhythm passes its own slot:
#: woof.core.nest_spawn reads ``uh_spawn_window`` on leg boundaries.
UH_SLOT = uh_diag.UH_FOLLOW_WINDOW_SLOT
REFLECTIVITY_SLOT = "refl_10cm"

#: How far from the extremum the centroid may draw, in kilometres.
#: THIS BOUND IS NOT OPTIONAL IN EFFECT, only in the config: without it
#: the qualifying region is whatever the threshold happens to select,
#: and MEASURED on Melissa's d03 that was 44-65% of the whole nest,
#: CLIPPED BY THE DOMAIN EDGE on most frames -- so the centroid was
#: partly an average of where the grid stops, which makes the nest's own
#: position an input to the centre it computes.  The default is a real
#: radius for that reason: absent means 50 km, never "unbounded".
#:
#: 50 km measured 1.05 km from the field minimum against 10.68 km for
#: the unbounded region, on 21 frames of the 2025-10-24 12Z run, and
#: jittered LESS frame to frame than the bare minimum did (5.74 vs
#: 6.02 km).  25 km sits closer still (0.28 km) and 75 km begins to
#: drift (4.84 km); the knob exists because a tropical cyclone's core
#: is 20-100 km and a mesoscale vortex is smaller.
DEFAULT_CENTROID_RADIUS_KM = 50.0

#: Fixed-point iteration limits.  MEASURED on the reference run: the
#: iteration converges in 16-31 steps on every frame, so 60 is a wide
#: margin and a non-converging field is a real finding rather than a
#: budget problem.  The tolerance is in CELLS: a tenth of a cell is 64 m
#: on d03, far below anything the shift quantisation can express.
CENTROID_MAX_ITERATIONS = 60
CENTROID_TOLERANCE_CELLS = 0.02

#: A second minimum outside the located core counts as a COMPETING
#: CENTRE when it is within this fraction of the core's own depth below
#: the search region's ceiling.  Not a refusal and not a vote: the
#: tracker keeps the deeper one and the receipt says the other was
#: there, because a centre reformation looks exactly like this one frame
#: before it happens and an operator should be able to see it coming.
COMPETING_CENTRE_FRACTION = 0.6

#: Admissible ``radius_km``.  The floor is two cells at any sane
#: spacing; the ceiling is wider than any vortex a nest would follow,
#: and a value beyond it is asking for the unbounded behaviour back.
RADIUS_KM_MIN = 1.0
RADIUS_KM_MAX = 500.0

#: Keys of ``[relocation.follow]``.  All required; unknown keys refuse.
FOLLOW_KEYS = ATTRIBUTE_KEYS | frozenset({
    "pressure_center", "fallback_level_hpa", "fallback_deficit_m",
    # OPTIONAL: how far from the extremum the centroid may draw.
    # Omitting it takes DEFAULT_CENTROID_RADIUS_KM, which is a real
    # radius -- there is no "unbounded" setting, because unbounded is
    # the defect this key was added to close.
    "radius_km",
    # OPTIONAL: which surface the vortex is tracked ON.  Omitting it
    # takes DEFAULT_LEVEL_HPA (850), the low-level circulation centre;
    # level_hpa = 0 is the sea-level-pressure tracker, and is the only
    # form whose threshold is an absolute hPa ceiling.
    "level_hpa",
    # OPTIONAL: the second stage.  Omitting it is the single-stage
    # tracker every existing config means.
    "refine_grid_id",
    "field", "threshold", "fallback_threshold", "search_margin_cells",
    "min_shift_cells", "max_shift_cells", "cooldown_seconds",
})

#: A proposal is clipped so the child's footprint keeps this many parent
#: cells clear of the parent's edge.  5 is WRF's default specified-boundary
#: width (Registry.EM_COMMON spec_bdy_width; spec_zone 1 + relax_zone 4):
#: a nest whose edge sits inside the parent's boundary relaxation zone is
#: being forced by boundary blending, not by the parent's interior
#: solution.  Tracker policy, not config: no case should want a nest in
#: the parent's spec zone.
PARENT_EDGE_KEEPOUT_CELLS = 5

_LOG = logging.getLogger("gpuwm.storm_tracking")


class TrackerRefusal(ValueError):
    """A tracking request this module will not serve quietly."""


# ---------------------------------------------------------------------------
# Config: [relocation.follow]
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FollowConfig:
    """The validated ``[relocation.follow]`` block.

    ``threshold`` is in the primary field's own units (m2 s-2 for
    ``"uh"``, dBZ for ``"reflectivity"``); ``fallback_threshold`` is
    always dBZ and exists only under ``field = "uh"``, where it gates the
    reflectivity handoff for the pre-rotation window.
    """

    field: str
    threshold: float
    search_margin_cells: int
    min_shift_cells: int
    max_shift_cells: int
    cooldown_seconds: float
    fallback_threshold: float | None = None
    #: The isobaric surface(s) the vortex is tracked ON.  A single
    #: number, or several -- normalised to a tuple by __post_init__, so
    #: every consumer sees one shape.  With more than one, the centre is
    #: the MEAN of the per-level centres, which is the deep-layer centre
    #: a nest should follow: a sheared storm's 850 hPa and 500 hPa
    #: circulations are genuinely in different places, and steering on
    #: either alone chases a tilt the other does not have.
    #:
    #: ``None`` on the way IN means "not configured" and becomes
    #: ``(DEFAULT_LEVEL_HPA,)`` under ``field = "pressure"``.  ``None``
    #: on the way OUT means SEA LEVEL -- the MSLP tracker, asked for as
    #: ``level_hpa = 0`` and normalised to ``None`` because that is the
    #: shape every consumer already reads.  The two never overlap: after
    #: ``__post_init__`` there is no such thing as "not configured".
    level_hpa: "float | tuple[float, ...] | None" = None
    refine_grid_id: int | None = None
    radius_km: float = DEFAULT_CENTROID_RADIUS_KM
    #: Surfaces the tracker computes a centre on FOR THE FILE ONLY.
    #:
    #: ``level_hpa`` answers "what steers the nest"; this answers "what
    #: else should the track file report".  They are different questions
    #: and conflating them costs one or the other: a nest wants a curated
    #: handful (850/700/500 is the classic steering set, and a mean over
    #: outflow-layer centres would drag it off the eyewall), while a
    #: forecaster reading vortex tilt wants the whole profile.
    #:
    #: A report-only surface gets the same plane, the same centre search
    #: and its own LevelFix -- so it is a real answer, not an
    #: interpolation -- and is simply absent from the mean in
    #: :func:`centre_over_levels`.  Derived from ``[relocation.track]``
    #: ``output_level`` by :mod:`woof.experiment`, which is the only
    #: place that sees both blocks; nothing else may set it, because a
    #: surface nobody asked to print would then be paid for and thrown
    #: away.
    report_level_hpa: tuple = ()
    attribute: str | None = None
    extremum: str | None = None
    reduction: str | None = None
    model_level: int | None = None
    pressure_center: str = "centroid"
    fallback_level_hpa: float | None = None
    fallback_deficit_m: float = 25.0

    def __post_init__(self) -> None:
        validate_attribute_config(self)
        if self.field not in TRACKED_FIELDS:
            raise ValueError(
                f"follow field must be one of {TRACKED_FIELDS}, got "
                f"{self.field!r}")
        if not math.isfinite(float(self.threshold)):
            raise ValueError(
                f"follow threshold must be finite, got {self.threshold!r}")
        if self.field == "uh":
            if self.fallback_threshold is None:
                raise ValueError(
                    "follow field = 'uh' requires fallback_threshold (dBZ): "
                    "the reflectivity handoff for the window before the "
                    "storm rotates cannot reuse a m2 s-2 number as dBZ")
            if not math.isfinite(float(self.fallback_threshold)):
                raise ValueError(
                    "follow fallback_threshold must be finite, got "
                    f"{self.fallback_threshold!r}")
        elif self.fallback_threshold is not None:
            raise ValueError(
                f"follow field = {self.field!r} refuses fallback_threshold: "
                "the echo handoff exists only under field = 'uh', where it "
                "covers the window before a storm rotates; there is nothing "
                "below this field to hand off to")
        # -- which surface ------------------------------------------------
        # Delegated so the follow block, a dormant domain's spawn table
        # and its retire table draw the two disjoint threshold bands in
        # exactly one place.  The field check inside runs on the RAW
        # value, before the sea-level normalisation turns an explicit 0
        # into None: otherwise `field = "uh"` with `level_hpa = 0` would
        # normalise itself out of the refusal and be quietly accepted.
        levels, defaulted = normalise_pressure_surface(
            self.level_hpa, field=self.field, threshold=self.threshold,
            label="follow", selector="field")
        object.__setattr__(self, "_level_defaulted", defaulted)
        object.__setattr__(self, "level_hpa", levels)
        if self.pressure_center not in ("centroid", "minimum"):
            raise ValueError("pressure_center must be centroid or minimum")
        if self.pressure_center == "minimum" and (self.field != "pressure" or levels is not None):
            raise ValueError("pressure_center=minimum requires the sea-level pressure tracker")
        if self.fallback_level_hpa is not None:
            if self.field != "pressure" or levels is not None:
                raise ValueError("fallback_level_hpa requires the sea-level pressure tracker")
            if not LEVEL_HPA_MIN <= self.fallback_level_hpa <= LEVEL_HPA_MAX:
                raise ValueError("fallback_level_hpa is outside the supported pressure column")
            if not LEVEL_DEFICIT_MIN_M <= self.fallback_deficit_m <= LEVEL_DEFICIT_MAX_M:
                raise ValueError("fallback_deficit_m is outside the supported height-deficit band")
        # -- the surfaces the FILE reports and the nest is not steered by
        report = tuple(float(v) for v in (self.report_level_hpa or ()))
        if report:
            if self.field != "pressure":
                raise ValueError(
                    f"follow field = {self.field!r} cannot report isobaric "
                    "surfaces; an isobaric surface is a property of the "
                    "MASS field")
            if self.level_hpa is None:
                raise ValueError(
                    "report_level_hpa alongside the sea-level tracker "
                    f"(level_hpa = {SEA_LEVEL_HPA:g}): the surface block "
                    "and an isobaric profile are different reductions in "
                    "different units, and this run computes only the "
                    "first. Track on an isobaric surface to report others.")
            if len(set(report)) != len(report):
                raise ValueError(
                    f"report_level_hpa = {list(report)} repeats a surface")
            both = sorted(set(report) & set(self.level_hpa or ()))
            if both:
                raise ValueError(
                    f"report_level_hpa repeats {both}, which "
                    "[relocation.follow] level_hpa already tracks; a "
                    "surface is computed once and either steers or does "
                    "not, so naming it twice is ambiguous about which")
            for level in report:
                if not (LEVEL_HPA_MIN <= level <= LEVEL_HPA_MAX):
                    raise ValueError(
                        f"report_level_hpa = {level!r} is outside "
                        f"{LEVEL_HPA_MIN}-{LEVEL_HPA_MAX} hPa")
        object.__setattr__(self, "report_level_hpa", report)
        if not (RADIUS_KM_MIN <= float(self.radius_km) <= RADIUS_KM_MAX):
            raise ValueError(
                f"follow radius_km = {self.radius_km!r} is outside "
                f"{RADIUS_KM_MIN}-{RADIUS_KM_MAX} km. It is how far from "
                "the signal's extremum the centroid may draw -- the size "
                "of the vortex, not of the domain. A tropical cyclone's "
                "core is 20-100 km; the default is "
                f"{DEFAULT_CENTROID_RADIUS_KM:g}.")
        if int(self.search_margin_cells) < 0:
            raise ValueError(
                "search_margin_cells must be >= 0 parent cells, got "
                f"{self.search_margin_cells!r}")
        if int(self.min_shift_cells) < 1:
            raise ValueError(
                "min_shift_cells must be >= 1 parent cell (the null shift "
                f"is never proposed), got {self.min_shift_cells!r}")
        if int(self.max_shift_cells) < int(self.min_shift_cells):
            raise ValueError(
                f"max_shift_cells = {self.max_shift_cells!r} is below "
                f"min_shift_cells = {self.min_shift_cells!r}; the dead-band "
                "would suppress every admissible proposal")
        if not (math.isfinite(float(self.cooldown_seconds))
                and float(self.cooldown_seconds) >= 0.0):
            raise ValueError(
                "cooldown_seconds must be a finite non-negative model-time "
                f"duration, got {self.cooldown_seconds!r}")

    def to_json(self) -> dict[str, object]:
        """Echo every configured value (the receipts' config record)."""
        out: dict[str, object] = {
            "contract": FOLLOW_CONTRACT,
            "field": self.field,
            "threshold": float(self.threshold),
            "search_margin_cells": int(self.search_margin_cells),
            "min_shift_cells": int(self.min_shift_cells),
            "max_shift_cells": int(self.max_shift_cells),
            "cooldown_seconds": float(self.cooldown_seconds),
            "radius_km": float(self.radius_km),
        }
        if self.fallback_threshold is not None:
            out["fallback_threshold"] = float(self.fallback_threshold)
        if self.field == "pressure":
            out["pressure_center"] = self.pressure_center
            if self.fallback_level_hpa is not None:
                out.update(fallback_level_hpa=self.fallback_level_hpa,
                           fallback_deficit_m=self.fallback_deficit_m)
        if self.field == "attribute":
            out.update(attribute_metadata(self))
            if self.refine_grid_id is not None:
                out["refine_grid_id"] = self.refine_grid_id
        if self.level_hpa is not None or self.field == "pressure":
            out.update(pressure_surface_json(
                self.level_hpa,
                defaulted=bool(getattr(self, "_level_defaulted", False))))
        return out


def normalise_pressure_surface(level_hpa, *, field, threshold, label,
                               selector="field", allow_multiple=True):
    """The ``level_hpa``/``threshold`` pair, validated once for every table.

    Three config tables now name a pressure surface -- ``[relocation
    .follow]``, a dormant domain's ``spawn`` and its ``retire`` -- and
    the question each is asking is the same one: WHICH SURFACE, and
    therefore what does ``threshold`` mean.  Two copies of that answer
    would be two vocabularies for one decision, and the disjoint
    threshold bands only protect anybody while every table draws them in
    the same place.

    Returns ``(levels, defaulted)``: ``levels`` is a tuple of isobaric
    surfaces, or ``None`` for SEA LEVEL (``level_hpa = 0``, the MSLP
    form, and the only one whose threshold is an absolute hPa ceiling);
    ``defaulted`` says whether the surface was chosen or inherited, which
    is what a receipt's ``level_hpa_source`` reports.

    ``label`` and ``selector`` name the caller's own table and key, so a
    refusal is spelled in the vocabulary of the file that caused it
    ("follow field", "spawn trigger") rather than in this function's.
    ``allow_multiple`` is False for the tables that pick ONE surface.
    """
    raw = level_hpa
    if raw is not None and field != "pressure":
        raise ValueError(
            f"{label} {selector} = {field!r} refuses level_hpa: an "
            "isobaric surface is a property of the MASS field, and "
            "neither the rotation stash nor the echo stash is on one. "
            f"Use {selector} = 'pressure'.")
    # ABSENT MEANS 850 hPa, not sea level.  A tracker has to be ON
    # something and the low-level circulation centre is the one a
    # nest should be centred on; the sea-level reduction is still
    # there, spelled `level_hpa = 0`, and it is the only form whose
    # threshold is an absolute hPa ceiling.  Recorded on the receipt
    # (`level_hpa_source`) so a reader can tell a defaulted surface
    # from a chosen one.
    defaulted = raw is None and field == "pressure"
    if defaulted:
        raw = DEFAULT_LEVEL_HPA
    levels = None
    if raw is not None:
        seq = raw if isinstance(raw, (list, tuple)) else [raw]
        if not seq:
            raise ValueError(
                f"{label} level_hpa is an empty list; give at least one "
                f"isobaric surface in hPa, delete the key for the "
                f"{DEFAULT_LEVEL_HPA:g} hPa default, or say "
                f"level_hpa = {SEA_LEVEL_HPA:g} for the "
                "sea-level-pressure tracker")
        levels = tuple(float(v) for v in seq)
        if len(set(levels)) != len(levels):
            raise ValueError(
                f"{label} level_hpa = {list(levels)} repeats a surface; "
                "each level is tracked once and averaged, so a "
                "duplicate would silently double that level's vote")
        if not allow_multiple and len(levels) > 1:
            raise ValueError(
                f"{label} level_hpa = {list(levels)} names "
                f"{len(levels)} surfaces, and this table tracks one "
                "surface. A deep-layer mean of per-level centres has no "
                "single extremum cell, and the extremum cell is what "
                "CLAIMS a storm here -- without one, two dormant slots "
                "can be born on the same vortex and a retirement timer "
                "can measure a decay the tracked surface never saw. "
                f"Name one surface, or {SEA_LEVEL_HPA:g} for sea level; "
                "[relocation.follow] is where a deep-layer mean steers.")
        # NO COUNT CAP: see MAX_TRACKED_LEVELS, which is None and says
        # why with the measurement.  The per-level checks below are the
        # bound that means something.
        if SEA_LEVEL_HPA in levels:
            if len(levels) > 1:
                raise ValueError(
                    f"{label} level_hpa = {list(levels)} mixes "
                    f"{SEA_LEVEL_HPA:g} (sea level) with isobaric "
                    "surfaces. Sea level is not one of them -- it is a "
                    "reduction BELOW ground, in different units, with a "
                    "different threshold form -- so it cannot be a term "
                    "in a deep-layer mean. Track on it alone, or drop "
                    "it from the list.")
            # Sea level is carried internally as level_hpa = None,
            # which is what every consumer already reads (planes_for,
            # central_pressure_mb, levels_of).  Normalising here means
            # the sea-level tracker is ONE code path, not two.
            levels = None
    for level in (levels or ()):
        if not (LEVEL_HPA_MIN <= float(level) <= LEVEL_HPA_MAX):
            raise ValueError(
                f"{label} level_hpa = {level!r} is outside "
                f"{LEVEL_HPA_MIN}-{LEVEL_HPA_MAX} hPa. It is the "
                "pressure of a surface the vortex is tracked ON, in "
                "hPa -- 850 for a low-level centre, 700 for a sheared "
                "one, 500 for the deep-layer steering centre. A value "
                "in Pa (85000) or a model-level index is a units "
                "error.")
    # THE ONE CONFIG THIS DEFAULT CHANGES, named on its own so the
    # message is the fix rather than a units lecture: a pre-default
    # config wrote an hPa ceiling and no level_hpa, meaning the
    # sea-level tracker, and the absent key now means 850 hPa where
    # the threshold is metres.  Both ways out are spelled.
    if (defaulted and PRESSURE_THRESHOLD_MIN_HPA <= float(threshold)
            <= PRESSURE_THRESHOLD_MAX_HPA):
        raise ValueError(
            f"{label} threshold = {threshold!r} reads as an "
            "absolute sea-level pressure ceiling in hPa, but this "
            "block names no level_hpa and an absent level_hpa now "
            f"means {DEFAULT_LEVEL_HPA:g} hPa, where the threshold is "
            "METRES of geopotential height above the search box's own "
            "minimum. Two ways out, and they are different trackers: "
            f"add level_hpa = {SEA_LEVEL_HPA:g} to keep the "
            "sea-level-pressure tracker this config has been running, "
            f"or keep the {DEFAULT_LEVEL_HPA:g} hPa surface and set a "
            "threshold in metres (20-60 m is a tropical-cyclone inner "
            "core).")
    if levels is not None and not (
            LEVEL_DEFICIT_MIN_M <= float(threshold) <= LEVEL_DEFICIT_MAX_M):
        raise ValueError(
            f"{label} threshold = {threshold!r} is outside "
            f"{LEVEL_DEFICIT_MIN_M}-{LEVEL_DEFICIT_MAX_M} m. UNDER "
            "level_hpa the threshold is RELATIVE: metres of "
            "geopotential height above the search box's own minimum "
            "that still count as the vortex core (20-60 m is a "
            "tropical-cyclone inner core). It is not an absolute "
            "height and not an hPa ceiling -- that is the MSLP "
            f"tracker, which is level_hpa = {SEA_LEVEL_HPA:g}.")
    if (field == "pressure" and levels is None
            and not (PRESSURE_THRESHOLD_MIN_HPA <= float(threshold)
                     <= PRESSURE_THRESHOLD_MAX_HPA)):
        raise ValueError(
            f"{label} {selector} = 'pressure' threshold = {threshold!r} "
            f"is outside {PRESSURE_THRESHOLD_MIN_HPA}-"
            f"{PRESSURE_THRESHOLD_MAX_HPA} hPa. The threshold is an "
            "absolute sea-level pressure CEILING in hPa -- cells at or "
            "below it are the vortex -- so a value in Pa (101300) or a "
            "deficit (20) is a units error. A tropical cyclone ceiling "
            f"is typically 1000-1008. (level_hpa = {SEA_LEVEL_HPA:g} is "
            "what selects this tracker; any other value is an isobaric "
            "surface, where the threshold is metres instead.)")
    return levels, defaulted


def pressure_surface_json(levels, *, defaulted: bool) -> dict:
    """The ``level_hpa``/``threshold_units`` half of a config receipt.

    Shared for the reason :func:`normalise_pressure_surface` is: the two
    pressure forms are told apart by the SAME key on every receipt, and
    a second copy of this could spell one of them differently.
    """
    if levels is not None:
        out: dict[str, object] = {
            "level_hpa": [float(v) for v in levels],
            "threshold_units": "m above search-box minimum",
            # Which surface RAN is not enough on its own once there is a
            # default: a receipt that says 850 should also say whether
            # the config asked for it.
            "level_hpa_source": "default" if defaulted else "config",
        }
        if len(levels) > 1:
            out["centre_rule"] = (
                "mean of the per-level centres (deep-layer centre)")
        return out
    # Sea level, always by explicit request now.  Echoed as the number
    # the config wrote rather than left absent, so the two pressure
    # forms are told apart by the same key on every receipt instead of
    # by one of them being missing.
    return {"level_hpa": [float(SEA_LEVEL_HPA)],
            "level_hpa_source": "config",
            "threshold_units": "hPa (absolute MSLP ceiling)"}


def is_minimum_signal(field) -> bool:
    """Whether ``field``'s feature is a MINIMUM rather than a maximum.

    Rotation and echo are where the field is LARGE; a cyclone is where
    it is SMALL, on either pressure form (a sea-level minimum, or a
    geopotential-height minimum on an isobaric surface).  Every
    evaluator that has to pick an extremum cell -- the tracker, the
    spawn trigger, the retirement decay test -- asks here, so the
    inversion cannot be implemented three times and drift.
    """
    return str(field) == "pressure"


def _require_number(table: dict, key: str, source: str, *,
                    integer: bool = False):
    value = table[key]
    ok = (isinstance(value, int) and not isinstance(value, bool)) if integer \
        else (isinstance(value, (int, float)) and not isinstance(value, bool))
    if not ok:
        kind = "an integer" if integer else "a number"
        raise ValueError(
            f"{key} in [relocation.follow] of {source} must be {kind}, "
            f"got {value!r}")
    return value


def build_follow_config(table: dict, source: str) -> FollowConfig:
    """Validate a parsed ``[relocation.follow]`` table.

    Honored or refused, never ignored: every key is required by name (a
    follow block is seven short lines, and a default the user never chose
    is how a nest starts moving on a threshold nobody picked), and every
    unknown key is refused by name.
    """
    from woof.experiment import did_you_mean

    unknown = sorted(set(table) - FOLLOW_KEYS)
    if unknown:
        named = ", ".join(
            f"{key!r}{did_you_mean(key, FOLLOW_KEYS)}" for key in unknown)
        raise ValueError(
            f"[relocation.follow] of {source} does not have key(s) {named}; "
            "no key is ignored, because a dropped key tracks a storm with a "
            "value nobody chose.")
    required = sorted(
        FOLLOW_KEYS - ATTRIBUTE_KEYS - {"fallback_threshold", "level_hpa", "refine_grid_id",
                       "radius_km", "pressure_center", "fallback_level_hpa", "fallback_deficit_m"})
    missing = [key for key in required if key not in table]
    if missing:
        raise ValueError(
            f"[relocation.follow] of {source} is missing required key(s) "
            f"{missing}; present: {sorted(table)}. Every follow key is "
            "chosen deliberately -- there are no defaults to inherit.")
    field = table["field"]
    if field != "attribute" and set(table) & ATTRIBUTE_KEYS:
        raise ValueError(f"[relocation.follow] of {source}: attribute keys require field = 'attribute'")
    if not isinstance(field, str):
        raise ValueError(
            f"field in [relocation.follow] of {source} must be a string, "
            f"got {field!r}")
    fallback = None
    if "fallback_threshold" in table:
        fallback = float(_require_number(table, "fallback_threshold", source))
    radius_km = DEFAULT_CENTROID_RADIUS_KM
    if "radius_km" in table:
        radius_km = float(_require_number(table, "radius_km", source))
    level_hpa = None
    if "level_hpa" in table:
        raw = table["level_hpa"]
        if isinstance(raw, (list, tuple)):
            for index, item in enumerate(raw):
                if (not isinstance(item, (int, float))
                        or isinstance(item, bool)):
                    raise ValueError(
                        f"level_hpa[{index}] in [relocation.follow] of "
                        f"{source} must be a number in hPa, got {item!r}")
            level_hpa = tuple(float(v) for v in raw)
        else:
            level_hpa = float(_require_number(table, "level_hpa", source))
    refine_grid_id = None
    if "refine_grid_id" in table:
        refine_grid_id = int(_require_number(
            table, "refine_grid_id", source, integer=True))
        if refine_grid_id < 1:
            raise ValueError(
                f"[relocation.follow] of {source}: refine_grid_id = "
                f"{refine_grid_id} is not a grid id; it names the DESCENDANT "
                "of the mover whose own field locates the vortex")
    try:
        return FollowConfig(
            field=field,
            threshold=float(_require_number(table, "threshold", source)),
            fallback_threshold=fallback,
            search_margin_cells=int(_require_number(
                table, "search_margin_cells", source, integer=True)),
            min_shift_cells=int(_require_number(
                table, "min_shift_cells", source, integer=True)),
            max_shift_cells=int(_require_number(
                table, "max_shift_cells", source, integer=True)),
            cooldown_seconds=float(_require_number(
                table, "cooldown_seconds", source)),
            level_hpa=level_hpa,
            refine_grid_id=refine_grid_id,
            **{key: table[key] for key in ("pressure_center", "fallback_level_hpa", "fallback_deficit_m") if key in table},
            **{key: table[key] for key in ATTRIBUTE_KEYS if key in table},
            radius_km=radius_km)
    except ValueError as err:
        raise ValueError(f"[relocation.follow] of {source}: {err}") from None


# ---------------------------------------------------------------------------
# Geometry: where the child sits, in parent index space
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RefinementSource:
    """A DESCENDANT's live state, and the map back to parent cells.

    Stage two of the tracker.  Stage one finds the storm on the mover's
    PARENT, which is the only field that can see outside the nest -- and
    on a 13.5 km parent that is a coarse answer for a nest whose whole
    purpose is 1.125 km.  MEASURED on Melissa (2025-10-22 01:12): d01's
    850 hPa minimum sat 44.9 km from d03's own, d02's sat 54.0 km from
    it, and the tracker's centroid 57.5 km -- so the nest was centred, to
    the tracker's satisfaction, on a point 50-odd km from the vortex the
    1.125 km grid actually resolved.  In a 338 km frame that is 16% off
    centre, which is exactly what it looks like in a plot.

    The fix is not a finer PARENT -- d02 was measured no better than d01
    -- it is to ask the domain that resolves the vortex where the vortex
    is.  This carries that domain's state plus the affine map from its
    own 0-based cell index back to the mover's parent coordinate, which
    is the space every shift in this module is expressed in.

    The map composes :class:`NestFootprint`'s own convention up the
    chain: a child mass cell ``m`` (0-based) sits at parent coordinate
    ``(i_parent_start - 1) + m / ratio``, so for d03 inside d02 inside
    d01 the composition collapses to ``origin + m/12`` with
    ``origin = (d02.i_parent_start - 1) + (d03.i_parent_start - 1)/3``.
    :func:`refinement_from_node` builds it by walking the tree, so the
    arithmetic lives in one place and no caller re-derives it.
    """

    grid_id: int
    state: object
    origin_i: float
    origin_j: float
    #: Parent cells per refine cell: 1/(product of ratios up the chain).
    scale_i: float
    scale_j: float
    edge_margin_cells: int = 8
    #: The refine grid's own spacing in metres, for the radius bound.
    dx_m: float | None = None

    def to_parent(self, i: float, j: float) -> tuple[float, float]:
        return (self.origin_i + float(i) * self.scale_i,
                self.origin_j + float(j) * self.scale_j)

    def from_parent(self, ci: float, cj: float) -> tuple[float, float]:
        return ((float(ci) - self.origin_i) / self.scale_i,
                (float(cj) - self.origin_j) / self.scale_j)


def refinement_from_node(mover_node, refine_grid_id: int,
                         *, edge_margin_cells: int = 8):
    """Build a :class:`RefinementSource` for one of the mover's descendants.

    Walks the tree DOWN from the mover to the named grid, then composes
    the placement chain back UP, so the returned map takes a cell index
    on that descendant to a coordinate in the mover's parent -- the space
    :meth:`StormTracker.desired_shift` proposes shifts in.

    ``None`` when the grid is not a descendant of the mover, has not
    started, or carries no state yet: every one of those is a reason to
    keep stage one's answer rather than a reason to refuse, because a
    coarse centre is still a centre and a nest that stops moving is worse
    than a nest that moves imprecisely.
    """

    target = int(refine_grid_id)
    chain: list = []

    def walk(node, path):
        if int(node.cfg.grid_id) == target:
            chain.extend(path + [node])
            return True
        return any(walk(child, path + [node]) for child in node.children)

    if not walk(mover_node, []):
        return None
    node = chain[-1]
    if not bool(getattr(node, "_started", True)):
        return None
    if getattr(node, "state", None) is None:
        return None
    # chain is [mover, ..., refine]; each link maps into ITS OWN parent,
    # so composing from the refine end upward and stopping after the
    # mover lands in the mover's parent.
    origin_i = origin_j = 0.0
    scale_i = scale_j = 1.0
    for link in reversed(chain):
        ratio = int(link.cfg.parent_grid_ratio)
        origin_i = (int(link.cfg.i_parent_start) - 1) + origin_i / ratio
        origin_j = (int(link.cfg.j_parent_start) - 1) + origin_j / ratio
        scale_i /= ratio
        scale_j /= ratio
    return RefinementSource(
        grid_id=target, state=node.state,
        origin_i=origin_i, origin_j=origin_j,
        scale_i=scale_i, scale_j=scale_j,
        edge_margin_cells=int(edge_margin_cells),
        dx_m=(lambda run: None if run is None else
              (None if getattr(run, "dx", None) is None
               else float(run.dx)))(getattr(node.cfg, "run", None)))


@dataclass(frozen=True)
class NestFootprint:
    """The child's current placement and extent, in parent-cell terms.

    ``i_parent_start``/``j_parent_start`` keep their 1-based WRF namelist
    semantics (:class:`woof.core.nest_relocation.Placement` refuses 0).
    Continuous parent-array coordinates (0-based) derive from them: child
    mass cell ``n`` (1-based) sits at parent coordinate ``i_parent_start
    + (n-1)/ratio``, WRF's donor arithmetic, so the child spans
    ``(child_nx - 1)/ratio`` parent cells.
    """

    grid_id: int
    i_parent_start: int
    j_parent_start: int
    child_nx: int
    child_ny: int
    parent_grid_ratio: int
    #: The PARENT's grid spacing in metres -- the frame every shift in
    #: this module is expressed in, and the one a radius in kilometres
    #: has to be converted through.  Derived in :meth:`coerce` from the
    #: child's own resolved dx times the ratio, so no caller supplies it
    #: and no second source of truth exists.  ``None`` on a hand-built
    #: footprint, which then takes the unbounded centroid.
    parent_dx_m: float | None = None
    #: The grid these parent-cell indices are counted on.  Indices are
    #: not geography: cell (35, 35) of d01 and cell (35, 35) of d02 are
    #: different places, so a footprint handed to a consumer watching a
    #: different grid would mask the wrong ground.  Taken in
    #: :meth:`coerce` from the child's own ``parent_id``; ``None`` on a
    #: hand-built footprint, which then belongs to whatever single grid
    #: its caller is looking at.
    parent_id: int | None = None

    def __post_init__(self) -> None:
        if int(self.parent_grid_ratio) < 1:
            raise ValueError(
                f"parent_grid_ratio must be >= 1, got "
                f"{self.parent_grid_ratio!r}")
        if int(self.i_parent_start) < 1 or int(self.j_parent_start) < 1:
            raise ValueError(
                "i_parent_start/j_parent_start are 1-based WRF namelist "
                f"semantics and must be >= 1, got "
                f"({self.i_parent_start!r}, {self.j_parent_start!r})")
        if int(self.child_nx) < 2 or int(self.child_ny) < 2:
            raise ValueError(
                f"child extents must be >= 2, got "
                f"({self.child_nx!r}, {self.child_ny!r})")

    @classmethod
    def coerce(cls, value) -> "NestFootprint":
        """Accept a NestFootprint, or duck-type a DomainConfig-like."""
        if isinstance(value, cls):
            return value
        run = getattr(value, "run", None)
        ratio = int(value.parent_grid_ratio)
        child_dx = None if run is None else getattr(run, "dx", None)
        parent_id = getattr(value, "parent_id", None)
        return cls(
            grid_id=int(value.grid_id),
            i_parent_start=int(value.i_parent_start),
            j_parent_start=int(value.j_parent_start),
            child_nx=int(run.nx if run is not None else value.child_nx),
            child_ny=int(run.ny if run is not None else value.child_ny),
            parent_grid_ratio=ratio,
            parent_dx_m=(None if child_dx is None
                         else float(child_dx) * ratio),
            parent_id=None if parent_id is None else int(parent_id))

    @property
    def span_parent_i(self) -> float:
        return (int(self.child_nx) - 1) / int(self.parent_grid_ratio)

    @property
    def span_parent_j(self) -> float:
        return (int(self.child_ny) - 1) / int(self.parent_grid_ratio)

    @property
    def center_parent_ij(self) -> tuple[float, float]:
        """(ci, cj): footprint center in 0-based parent-array coords."""
        return (int(self.i_parent_start) - 1 + self.span_parent_i / 2.0,
                int(self.j_parent_start) - 1 + self.span_parent_j / 2.0)

    def search_box(self, plane_shape: tuple[int, int],
                   margin_cells: int) -> tuple[slice, slice]:
        """(j_slice, i_slice) of the footprint + margin, clipped to the
        parent plane."""
        ny, nx = int(plane_shape[-2]), int(plane_shape[-1])
        margin = int(margin_cells)
        i_lo = int(self.i_parent_start) - 1
        j_lo = int(self.j_parent_start) - 1
        i0 = max(0, i_lo - margin)
        j0 = max(0, j_lo - margin)
        i1 = min(nx, int(math.ceil(i_lo + self.span_parent_i)) + 1 + margin)
        j1 = min(ny, int(math.ceil(j_lo + self.span_parent_j)) + 1 + margin)
        if i0 >= i1 or j0 >= j1:
            raise TrackerRefusal(
                f"the nest footprint (i {i_lo}..{i_lo + self.span_parent_i:g},"
                f" j {j_lo}..{j_lo + self.span_parent_j:g}) lies outside the "
                f"parent plane {(ny, nx)}; the footprint and the parent "
                "state disagree about the tree geometry")
        return slice(j0, j1), slice(i0, i1)


# ---------------------------------------------------------------------------
# Signal extraction and centroid arithmetic
# ---------------------------------------------------------------------------

def _host(value) -> np.ndarray:
    """Read-only host view/copy of a host or device array (the
    nest_relocation idiom)."""
    if isinstance(value, np.ndarray):
        return value
    return value.get()


def _xp(array):
    """``cupy`` or ``numpy``, whichever owns ``array`` (the health_ledger
    idiom, restated locally so this module imports no ledger)."""
    if isinstance(array, np.ndarray):
        return np
    try:
        import cupy
    except Exception:                                    # pragma: no cover
        return np
    getter = getattr(cupy, "get_array_module", None)
    return getter(array) if getter is not None else cupy


def _smooth_nine_point(plane, passes: int, center_weight: float, xp):
    """:func:`woof.core.mslp._nine_point_smooth`, array-module
    agnostic.

    Same kernel, same edge replication, same normalisation -- written
    against ``xp`` so it runs where the plane already lives.  The host
    version stays the reference; this one exists because dragging a
    device plane to the host to blur it and back again is the whole cost
    of the operation.
    """
    out = plane
    for _ in range(int(passes)):
        pad = xp.pad(out, 1, mode="edge")
        out = (center_weight * out
               + pad[:-2, 1:-1] + pad[2:, 1:-1]
               + pad[1:-1, :-2] + pad[1:-1, 2:]
               + pad[:-2, :-2] + pad[:-2, 2:]
               + pad[2:, :-2] + pad[2:, 2:]) / (center_weight + 8.0)
    return out


def level_heights_m_from_state(state, levels) -> list:
    """``[(level_hpa, plane), ...]`` for several surfaces at once.

    THE SHARED HALF IS COMPUTED ONCE.  The state's reading
    (:func:`_isobaric_shared`: its interface stencil, or for a
    height-coordinate state the mass-point height and ``log(p)``) does not
    depend on the level, and asking for it per surface did that work once
    per surface.

    IT IS NOT A SPEEDUP, and the number is here so nobody re-derives it.
    Isolated, the shared half is 0.303 ms against a 2.28 ms surface on
    49x378x378 -- about 15%, and the two larger terms (the 9-point
    smoother at 0.638 ms, the four gathers at 0.469 ms) are genuinely
    per-surface.  Measured end to end through :func:`planes_for`, even
    that 15% does not appear: 1 to 37 surfaces on both production shapes
    lands between -4.9% and +9.8% with no consistent sign, which is
    inside this box's run-to-run noise.

    It is kept for the reason it should be kept -- there is now ONE
    implementation of the isobaric interpolation instead of two copies
    that could drift -- and the arithmetic is unchanged, so every plane
    is bit-for-bit what the per-surface path produced.

    A surface that lies outside the column everywhere refuses on its own
    without touching the others: one bad level in a twenty-level profile
    is a declined column, not a dead run.
    """
    levels = [float(v) for v in levels]
    if not levels:
        return []
    shared = _isobaric_shared(state)
    return [(level, _level_plane(state, level, *shared))
            for level in levels]


class _SharedIsobaric:
    """The level-independent 3-D quantities, computed at most once.

    LAZY ON PURPOSE, for two reasons that point the same way.  It keeps
    :func:`_plane_from_state` the single seam every consumer and every
    test already reaches for -- a caller that monkeypatches it to hand
    back a canned plane never touches a real state, and eager
    computation here would refuse before the patch could intervene.  And
    it means a tracker whose surfaces all come from somewhere else pays
    nothing.
    """

    __slots__ = ("_state", "_value")

    def __init__(self, state):
        self._state = state
        self._value = None

    def get(self):
        if self._value is None:
            self._value = _isobaric_shared(self._state)
        return self._value


class _InterfaceColumns:
    """A state's columns read between layer interfaces, for one surface at
    a time: the CUDA transcription below on a device state, the Rust
    ``rw-isobaric`` reader (:mod:`woof.isobaric_bridge`) on a host one.
    Both are the crate's ``column_isobaric_height`` word for word."""

    __slots__ = ("state", "xp", "znu", "znw")

    def __init__(self, state, xp, znu, znw):
        self.state, self.xp, self.znu, self.znw = state, xp, znu, znw

    def raw_plane(self, level_hpa: float):
        """``(plane, outside)``: height (m) of the surface, NaN and
        ``outside`` where the column has no such surface."""
        state, xp = self.state, self.xp
        level_pa = float(level_hpa) * 100.0
        if xp is np:
            from woof import isobaric_bridge
            try:
                plane = isobaric_bridge.isobaric_heights(
                    state.php, state.p, (level_pa,), eta_interface=self.znw,
                    eta_mass=self.znu, interface_plus=state.phb,
                    per_metre=GRAVITY_M_S2)[0]
            except (FileNotFoundError, OSError,
                    isobaric_bridge.IsobaricBridgeError) as error:
                raise TrackerRefusal(
                    f"the {level_hpa:g} hPa surface cannot be read between "
                    f"the parent's layer interfaces on the host: {error}") from error
        else:
            plane = _device_isobaric_height(state, self.znu, self.znw, level_pa)
        return plane, ~xp.isfinite(plane)


#: The device transcription of ``rw_isobaric::column_isobaric_height`` and
#: ``InterfaceStencil``: one thread per column, a raw kernel because that
#: needs no toolkit headers beyond NVRTC, every floating operation an
#: explicit round-to-nearest intrinsic in the crate's own order (so nothing
#: is contracted into a fused multiply-add), every quotient ``__ddiv_rn`` of
#: two runtime values, and the logarithm ``plm_log`` from
#: ``kernels/portable_libm64.cuh``, the transcription of the ``libm`` 0.2.16
#: ``log`` the crate calls.  The float32 state widens exactly.  Graded word
#: for word against the crate by tests/test_storm_tracking_isobaric_gpu.py.
_ISOBARIC_HEIGHT_SOURCE = r"""
extern "C" __global__ void storm_tracking_isobaric_height(
        const float* p, const float* php, const float* phb, int phb_profile,
        const double* eta_mass, const double* eta_w, int nz, long long cells,
        double level_pa, double g, double* z) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cells) return;
    const long long n = cells;
    // Every interface pressure must be a positive number, or the column
    // has no usable interfaces (InterfaceStencil::interface_ln_pressures).
    bool usable = true;
    for (int k = 0; k <= nz && usable; ++k) {
        int below = k - 1;
        if (below < 0) below = 0;
        if (below > nz - 2) below = nz - 2;
        const double span = __dsub_rn(eta_mass[below + 1], eta_mass[below]);
        const double w = __ddiv_rn(__dsub_rn(eta_w[k], eta_mass[below]), span);
        const double pl = (double)p[below * n + i];
        const double pu = (double)p[(below + 1) * n + i];
        const double pw = __dadd_rn(pl, __dmul_rn(__dsub_rn(pu, pl), w));
        if (!isfinite(pw) || pw <= 0.0) usable = false;
    }
    double out = __longlong_as_double(0x7ff8000000000000LL);
    if (usable) {
        const double target = plm_log(level_pa);
        double lower = 0.0;
        for (int k = 0; k <= nz; ++k) {
            int below = k - 1;
            if (below < 0) below = 0;
            if (below > nz - 2) below = nz - 2;
            const double span = __dsub_rn(eta_mass[below + 1], eta_mass[below]);
            const double w = __ddiv_rn(__dsub_rn(eta_w[k], eta_mass[below]), span);
            const double pl = (double)p[below * n + i];
            const double pu = (double)p[(below + 1) * n + i];
            const double upper = plm_log(
                __dadd_rn(pl, __dmul_rn(__dsub_rn(pu, pl), w)));
            if (k > 0 && lower >= target && upper < target) {
                const double fraction = __ddiv_rn(__dsub_rn(target, lower),
                                                  __dsub_rn(upper, lower));
                const long long a = (long long)(k - 1) * n + i;
                const long long b = (long long)k * n + i;
                const double ba = phb_profile ? (double)phb[k - 1] : (double)phb[a];
                const double bb = phb_profile ? (double)phb[k] : (double)phb[b];
                const double zl = __ddiv_rn(__dadd_rn((double)php[a], ba), g);
                const double zu = __ddiv_rn(__dadd_rn((double)php[b], bb), g);
                if (isfinite(zl) && isfinite(zu)) {
                    out = __dadd_rn(zl, __dmul_rn(__dsub_rn(zu, zl), fraction));
                }
                break;
            }
            lower = upper;
        }
    }
    z[i] = out;
}
"""

_ISOBARIC_HEIGHT_KERNEL = None


def _device_isobaric_height(state, znu, znw, level_pa: float):
    """The surface's height (m, float64) on the device, NaN where a column
    has no such surface."""
    global _ISOBARIC_HEIGHT_KERNEL
    import cupy as cp
    if _ISOBARIC_HEIGHT_KERNEL is None:
        from pathlib import Path
        header = (Path(__file__).resolve().parent / "kernels"
                  / "portable_libm64.cuh").read_text(encoding="utf-8")
        _ISOBARIC_HEIGHT_KERNEL = cp.RawKernel(
            header + _ISOBARIC_HEIGHT_SOURCE, "storm_tracking_isobaric_height",
            options=("-std=c++17",))
    pressure = cp.ascontiguousarray(state.p, dtype=cp.float32)
    nz = int(pressure.shape[0])
    horizontal = tuple(int(n) for n in pressure.shape[1:])
    cells = int(np.prod(horizontal, dtype=np.int64))
    phb = cp.ascontiguousarray(state.phb, dtype=cp.float32)
    plane = cp.empty(horizontal, dtype=cp.float64)
    threads = 128
    _ISOBARIC_HEIGHT_KERNEL(
        ((cells + threads - 1) // threads,), (threads,),
        (pressure, cp.ascontiguousarray(state.php, dtype=cp.float32), phb,
         np.int32(phb.ndim == 1), cp.asarray(znu, dtype=cp.float64),
         cp.asarray(znw, dtype=cp.float64), np.int32(nz), np.int64(cells),
         np.float64(level_pa), np.float64(GRAVITY_M_S2), plane))
    return plane


def _isobaric_shared(state):
    """The level-INDEPENDENT reading of a state's columns.

    Split out so the single-surface and multi-surface paths read them the
    same way and cannot drift; the refusals live here because they are
    about the STATE, not about any one level.

    READ BETWEEN LAYER INTERFACES.  With the state's eta levels (``znu``,
    ``znw``) every surface is the Rust ``rw-isobaric`` reading every chart
    uses (:class:`_InterfaceColumns`): the interface geopotential, read in
    ln p between interfaces whose pressures are linear in eta between the
    mass-level pressures.  The layer-mean height paired with the mass-level
    pressure read every surface about 5 m high at 500 hPa, so the reported
    centre heights were high by that much.  A state without eta levels (a
    height-coordinate column, whose mass level IS its layer's middle in
    height) keeps the layer means, ``(height, log_p, xp)``.
    """
    for name in ("p", "php", "phb"):
        if getattr(state, name, None) is None:
            raise TrackerRefusal(
                f"the parent state carries no {name!r}, so an isobaric "
                "surface cannot be built from it; field = 'pressure' with "
                "level_hpa needs the live prognostic column. Refused "
                "rather than holding forever: a follow block that can "
                "never see its field is a configuration error, not a "
                "quiet storm.")
    pressure = state.p
    xp = _xp(pressure)
    php = state.php
    phb = state.phb
    if php.shape[0] != pressure.shape[0] + 1 or phb.shape[0] != php.shape[0]:
        raise TrackerRefusal(
            f"geopotential has {php.shape[0]} levels against "
            f"{pressure.shape[0]} mass levels; the isobaric tracker needs "
            "the staggered pair it can average to mass points")
    znu, znw = getattr(state, "znu", None), getattr(state, "znw", None)
    if znu is not None and znw is not None:
        znu = np.asarray(_host(znu), dtype=np.float64).ravel()
        znw = np.asarray(_host(znw), dtype=np.float64).ravel()
        nz = int(pressure.shape[0])
        # The stencil's own refusals (InterfaceStencil::new), checked once
        # here because the device path has no error channel.
        problem = None
        if znu.size != nz or znw.size != nz + 1 or nz < 2:
            problem = (f"{znu.size} mass level(s) and {znw.size} interface(s) "
                       f"for {nz} model levels")
        elif not (np.isfinite(znu).all() and np.isfinite(znw).all()):
            problem = "the vertical coordinate holds a value that is not a number"
        elif (np.diff(znu) == 0.0).any():
            problem = "two mass levels share one vertical coordinate"
        if problem is not None:
            raise TrackerRefusal(
                f"the parent state's eta levels cannot place its layer "
                f"interfaces ({problem}), so no isobaric surface can be read "
                "between them")
        return (_InterfaceColumns(state, xp, znu, znw),)
    if getattr(phb, "ndim", 2) == 1:
        phb = phb[:, None, None]
    phi = phb + php
    height = 0.5 * (phi[:-1] + phi[1:]) / GRAVITY_M_S2
    return height, xp.log(pressure), xp


def _level_plane(state, level_hpa: float, *shared) -> np.ndarray:
    """One surface, given :func:`_isobaric_shared`'s reading: between
    interfaces, or the layer means of a height-coordinate state."""
    target_pa = float(level_hpa) * 100.0
    if len(shared) == 1:
        columns = shared[0]
        xp = columns.xp
        plane, outside = columns.raw_plane(level_hpa)
    else:
        height, log_p, xp = shared
        log_target = math.log(target_pa)
        nz = int(log_p.shape[0])
        # The layer that BRACKETS the target: pressure decreases with k, so
        # the count of levels at or below the target's log-pressure, minus
        # one, is the index of the level beneath it.  Clipping to nz-2 keeps
        # the k+1 gather in range; the columns that clipping would have
        # extrapolated are the ones the finiteness mask drops below.
        below = xp.count_nonzero(log_p >= log_target, axis=0) - 1
        k = xp.clip(below, 0, nz - 2)[None, ...]
        log_p0 = xp.take_along_axis(log_p, k, 0)[0]
        log_p1 = xp.take_along_axis(log_p, k + 1, 0)[0]
        z0 = xp.take_along_axis(height, k, 0)[0]
        z1 = xp.take_along_axis(height, k + 1, 0)[0]
        weight = (log_p0 - log_target) / (log_p0 - log_p1)
        plane = z0 + (z1 - z0) * weight
        outside = (log_p[0] < log_target) | (log_p[-1] > log_target)
    if bool(outside.all()):
        raise TrackerRefusal(
            f"the {level_hpa:g} hPa surface lies outside the parent "
            "column at every grid point (below ground everywhere, or "
            "above the model top). Nothing on this level can be tracked; "
            "choose a level inside the domain's pressure range.")
    # Smoothed BEFORE it is searched, exactly as the MSLP path is: a
    # grid-scale dimple can outrank the real centre.  The excluded cells
    # are filled with the plane's own maximum first so they neither drag
    # the blur toward a fictitious low nor spread NaN across it, then
    # restored to NaN so they still cannot vote.
    filled = xp.where(outside, xp.nanmax(xp.where(outside, xp.nan, plane)),
                      plane)
    from woof.core.mslp import MSLP_SMOOTH_CENTER_WEIGHT
    smoothed = _smooth_nine_point(filled, PRESSURE_SMOOTH_PASSES,
                                  MSLP_SMOOTH_CENTER_WEIGHT, xp)
    smoothed = xp.where(outside, xp.nan, smoothed)
    return np.asarray(_host(smoothed), dtype=np.float64)


def level_height_m_from_state(state, level_hpa: float) -> np.ndarray:
    """Smoothed geopotential height (m) on the ``level_hpa`` isobaric
    surface, as a host ``(ny, nx)`` plane.

    THE MID-LEVEL VORTEX, WHICH IS WHAT WRF TRACKS.  WRF's moving nest
    does not follow sea-level pressure: ``&domains track_level`` (Pa,
    default 50000) names an isobaric surface and the vortex is the
    height minimum on it.  This is that surface, chosen in hPa because
    every other pressure in this config surface is.

    WHY IT BEATS MSLP for a nest to ride.  Sea-level pressure is a
    REDUCTION -- below ground it is an extrapolation through a fictitious
    column, so an island in the footprint injects a terrain signal into
    the very field steering the nest, and the reduction's own smoothing
    is what keeps that tolerable.  A geopotential height on an isobaric
    surface above the boundary layer is a measured quantity everywhere
    that surface exists: no reduction, no fictitious column, no terrain
    term.  It is also where the vortex is most coherent -- the low-level
    centre wobbles with convection on a 5-15 km scale, and that wobble is
    exactly the jitter that made the reflectivity tracker walk a nest
    east and then back (see the 2026-08-12 measurement).

    WHERE THE SURFACE DOES NOT EXIST the cells do not vote.  A column
    whose surface pressure is below ``level_hpa`` has that surface
    underground; a column whose top is above it has it out of the domain.
    Both are NaN here and both are dropped by
    :func:`weighted_centroid`'s finiteness mask, rather than being
    extrapolated into a height that would out-vote the real centre.

    OPTIMISED, and deliberately: this runs on the DEVICE, on the two
    fields it needs (``p`` and the geopotential pair), and only the
    finished ``(ny, nx)`` plane crosses to the host.  The MSLP path pulls
    five FULL 3-D fields across and reduces them in host numpy --- for a
    372x284x45 parent that is ~190 MiB per tracker call against ~0.4 MiB
    here.  The reduction itself is one ``count_nonzero`` plus four
    ``take_along_axis`` gathers and a lerp: no per-column search loop,
    no Python over grid points.

    Log-linear in pressure, which is the standard isobaric interpolation
    and is exact for an isothermal layer (hydrostatic + ideal gas make
    ``z`` linear in ``ln p``); over a WRF layer thickness the residual is
    far below the metres this tracker resolves.
    """
    return _level_plane(state, level_hpa, *_isobaric_shared(state))


@dataclass(frozen=True)
class _Crop:
    """A window plus the halo the smoother needs, and the way back."""

    rows: slice
    cols: slice
    #: Offsets of the requested window inside the haloed crop.
    inner_j: int
    inner_i: int
    inner_ny: int
    inner_nx: int

    def replant(self, cropped: np.ndarray,
                full_shape: tuple[int, int]) -> np.ndarray:
        """The window's values, back in FULL-DOMAIN index space.

        Everything outside the window is NaN rather than zero or a fill:
        :func:`weighted_centroid` and :func:`locate_signal` both mask on
        finiteness, so a cell that was never reduced cannot vote, and a
        caller that reads outside the window gets a value that is
        obviously absent instead of one that is quietly wrong.
        """
        out = np.full(full_shape, np.nan, dtype=np.float64)
        out[self.rows.start + self.inner_j:
            self.rows.start + self.inner_j + self.inner_ny,
            self.cols.start + self.inner_i:
            self.cols.start + self.inner_i + self.inner_nx] = cropped[
                self.inner_j:self.inner_j + self.inner_ny,
                self.inner_i:self.inner_i + self.inner_nx]
        return out


def _halo_crop(window, full_shape: tuple[int, int]):
    """``window`` grown by the smoother's reach, clipped to the domain.

    WHY THIS IS EXACT.  :func:`woof.core.mslp._dcomputeseaprs` is
    COLUMN-LOCAL -- every operation in it is elementwise over ``(ny, nx)``
    or a ``take_along_axis`` along k -- so reducing a crop gives, bit for
    bit, what reducing the whole domain gives on that crop.  The
    nine-point smoother is the one part that is not: three passes reach
    exactly three cells, so a three-cell halo reproduces the interior
    bitwise.  Where the halo would run off the domain it is clipped, and
    there the crop's own edge IS the domain's edge, so ``np.pad(...,
    mode="edge")`` replicates the same row the full-domain smooth
    replicates.  Both cases were measured bitwise (max|delta| = 0.0 hPa)
    before this was adopted.

    ``None`` in, ``None`` out: no window means reduce everything, which
    is what every caller before the track writer wanted.
    """
    if window is None:
        return None
    j_slice, i_slice = window
    ny, nx = int(full_shape[0]), int(full_shape[1])
    j0, j1 = int(j_slice.start), int(j_slice.stop)
    i0, i1 = int(i_slice.start), int(i_slice.stop)
    if not (0 <= j0 < j1 <= ny and 0 <= i0 < i1 <= nx):
        raise TrackerRefusal(
            f"the reduction window (j {j0}..{j1}, i {i0}..{i1}) is not "
            f"inside the parent plane {(ny, nx)}; a window that does not "
            "lie on the field it crops would silently reduce a different "
            "piece of the domain than the caller searched")
    halo = int(PRESSURE_SMOOTH_PASSES)
    hj0, hj1 = max(0, j0 - halo), min(ny, j1 + halo)
    hi0, hi1 = max(0, i0 - halo), min(nx, i1 + halo)
    return _Crop(rows=slice(hj0, hj1), cols=slice(hi0, hi1),
                 inner_j=j0 - hj0, inner_i=i0 - hi0,
                 inner_ny=j1 - j0, inner_nx=i1 - i0)


def mslp_hpa_from_state(state, *, window=None, resident=False, smooth=True):
    """Mean-sea-level pressure (hPa), smoothed and host-resident by default.

    ``resident=True`` keeps a full-domain CUDA result on the caller's stream.
    ``smooth=False`` exposes the raw native diagnostic for integral observation
    operators. Existing tracker callers keep their original smoothing and copy.

    ``window`` is an optional ``(j_slice, i_slice)`` -- the shape
    :meth:`NestFootprint.search_box` returns -- and it is the reason this
    function is affordable to call more than once a relocation cadence.
    The column reduction and smoothing run on the state's array backend:
    CUDA states stay on the GPU in float64 until the final 2-D host copy.
    CPU states use the same formula with NumPy. Cropping includes the
    smoother halo so the searched interior agrees with a full reduction.

    Outside the window the returned plane is NaN, so a windowed plane is
    shape-compatible with a full one and no consumer can accidentally
    read a value that was never computed.

    THE VORTEX SIGNAL, AND WHY IT IS NOT A STASH.  ``uh`` and
    ``reflectivity`` are planes somebody else folded and left in a
    scratch slot, which is why their cadence is chained to the history
    writer.  Sea-level pressure is not stored anywhere: it is REDUCED
    here from the live prognostic column, which is valid at every cycle
    boundary.  A pressure tracker therefore has no cadence contract with
    the output knobs at all -- see :data:`STASH_BACKED_FIELDS`.

    THE REDUCTION IS NOT A NEW ONE.  It is
    ``woof.core.mslp._dcomputeseaprs`` -- the wrf-python
    DCOMPUTESEAPRS (Shuell 1995) transcription this tree already
    validated against the wrf-rust oracle to 4.6e-10 hPa -- fed the
    same four fields its wrfout caller feeds it, derived from state
    rather than read from a file:

    ============  ==================  =============================
    wrfout        this state          note
    ============  ==================  =============================
    ``P + PB``    ``state.p``         EOS pressure (Pa)
    ``T + 300``   ``total_theta()``   potential temperature (K)
    ``QVAPOR``    ``state.qv``        zero on a dry state
    ``PH + PHB``  ``php + phb``       geopotential; /9.80665 for z
    ============  ==================  =============================

    The temperature closure and the 9.80665 height divisor are copied
    from that caller verbatim (``woof/verify/metrics.py:195-215``) so
    the tracker and the verification maps cannot drift apart.

    SMOOTHED BEFORE IT IS SEARCHED, which is the substantive half of
    WRF's vortex finder.  A raw grid-scale pressure dimple can outrank
    the real centre; three passes of the Shuman nine-point smoother
    attenuate a checkerboard by 1/27 per interior cell and leave the
    synoptic minimum where it was.  Same parameters the MSLP display
    treatment uses, imported rather than restated.
    """
    from woof.core.mslp import (MSLP_SMOOTH_CENTER_WEIGHT,
                                 _dcomputeseaprs, _nine_point_smooth)

    for name in ("p", "php", "phb"):
        if getattr(state, name, None) is None:
            raise TrackerRefusal(
                f"the parent state carries no {name!r}, so sea-level "
                "pressure cannot be reduced from it; field = 'pressure' "
                "needs the live prognostic column. Refused rather than "
                "holding forever: a follow block that can never see its "
                "field is a configuration error, not a quiet storm.")
    full_shape = (int(state.p.shape[-2]), int(state.p.shape[-1]))
    crop = _halo_crop(window, full_shape)
    if crop is None:
        depth = slice(None)
    else:
        depth = (slice(None), crop.rows, crop.cols)
    xp = _xp(state.p)
    pressure_pa = xp.asarray(state.p[depth], dtype=xp.float64)
    theta = xp.asarray(state.total_theta()[depth], dtype=xp.float64)
    php = xp.asarray(state.php[depth], dtype=xp.float64)
    phb_source = state.phb
    if getattr(phb_source, "ndim", 3) == 1:
        # A 1-D base geopotential is a column, not a field: it broadcasts
        # over every point and there is nothing to crop.
        phb = xp.asarray(phb_source, dtype=xp.float64)[:, None, None]
    else:
        phb = xp.asarray(phb_source[depth], dtype=xp.float64)
    qv = getattr(state, "qv", None)
    qv = (xp.zeros_like(pressure_pa) if qv is None
          else xp.asarray(qv[depth], dtype=xp.float64))
    # WRF rcp = r_d/cp with cp = 7*r_d/2 = 1004.5
    # (share/module_model_constants.F:19-20,31), as the wrfout caller.
    temperature = theta * (pressure_pa / 100000.0) ** (287.0 / 1004.5)
    phi = phb + php
    height_msl = 0.5 * (phi[:-1] + phi[1:]) / 9.80665
    try:
        mslp = _dcomputeseaprs(pressure_pa, temperature, qv, height_msl, array_module=xp)
    except ValueError as exc:
        raise TrackerRefusal(
            f"sea-level pressure could not be reduced from the parent "
            f"column ({exc}); field = 'pressure' cannot track this "
            "state") from exc
    smoothed = (_nine_point_smooth(mslp, PRESSURE_SMOOTH_PASSES,
                                   MSLP_SMOOTH_CENTER_WEIGHT, array_module=xp)
                if smooth else mslp)
    if resident:
        if crop is not None:
            raise TrackerRefusal("resident pressure diagnostics require the full domain, not a host-replanted crop")
        if not hasattr(smoothed, "__cuda_array_interface__"):
            raise TrackerRefusal("resident pressure diagnostics require a CUDA model state")
        return smoothed
    smoothed = _host(smoothed)
    if crop is None:
        return smoothed
    return crop.replant(smoothed, full_shape)


def planes_for(state, config, *, uh_slot: str = UH_SLOT,
               window=None) -> list:
    """``[(level_hpa | None, plane), ...]`` -- what the tracker searches.

    One entry for every configured isobaric surface, or a single entry
    with ``None`` for the tracker shapes that have one plane (``uh``,
    ``reflectivity``, and sea-level pressure).  Keeping the fan-out here
    means :meth:`StormTracker.locate` and :meth:`StormTracker._refine`
    share it and cannot drift.
    """
    if config.field == "attribute":
        try:
            return [(None, attribute_plane(state, config, window=window))]
        except ValueError as error:
            raise TrackerRefusal(str(error)) from error
    levels = all_levels_of(config)
    if not levels:
        return [(None, _plane_from_state(state, config.field,
                                         uh_slot=uh_slot, window=window))]
    # ONE pass for the shared 3-D quantities -- bit-for-bit what the
    # per-surface path produced, and measured as no faster (see
    # level_heights_m_from_state); the win is one implementation, not ms.
    #
    # Still through _plane_from_state, which is the seam consumers and
    # tests patch to substitute a plane -- routing around it to reach the
    # batched path would have made the shared work invisible to every one
    # of them.  The holder is lazy, so a patched seam computes nothing.
    shared = _SharedIsobaric(state)
    return [(level, _plane_from_state(state, config.field, uh_slot=uh_slot,
                                      level_hpa=level, window=window,
                                      shared=shared))
            for level in levels]


def _plane_from_state(state, field: str, *,
                      uh_slot: str = UH_SLOT,
                      level_hpa: float | None = None,
                      window=None, shared=None) -> np.ndarray:
    """The named signal as a host (ny, nx) float plane, fail-loud.

    ``"uh"`` reads the caller's own UH window (``uh_slot``, defaulting to
    the relocation consumer's); ``"reflectivity"`` reads ``refl_10cm``
    and collapses a 3-D volume to its column max (the composite).  A
    missing slot refuses with the configuration that would provide it --
    silently tracking nothing is how a nest quietly stops following its
    storm.

    ``uh_slot`` exists because the UH window belongs to whoever resets
    it.  Passing another consumer's slot would read a window emptied on
    a rhythm this caller does not control, which is the whole defect the
    consumer-owned windows fixed.

    ``"pressure"`` takes neither route: it is reduced from the live
    column by :func:`mslp_hpa_from_state` and returned in hPa, the
    field's own units, POSITIVE.  The caller is what knows a vortex is a
    minimum; see :meth:`StormTracker.desired_shift`.
    """
    if field == "pressure":
        if level_hpa is not None:
            # Already a DEVICE reduction whose only host crossing is the
            # finished (ny, nx) plane -- 2 ms on a 378x378x49 grid --
            # so there is nothing a window would buy here.
            #
            # ``shared`` is the caller's _SharedIsobaric when several
            # surfaces are being built from one state, so phi, the
            # mass-point height and log(p) are computed once for all of
            # them instead of once each.
            if shared is not None:
                return _level_plane(state, level_hpa, *shared.get())
            return level_height_m_from_state(state, level_hpa)
        return mslp_hpa_from_state(state, window=window)
    if not uh_diag.is_tracker_window_slot(uh_slot):
        raise ValueError(
            f"uh_slot={uh_slot!r} is not a consumer-owned tracking "
            "window; use a fixed consumer slot or a slot returned by "
            "uh_diag.follow_window_slot(grid_id). The "
            "WRF UP_HELI_MAX diagnostic is reset by the history writer, "
            "so reading it would make this decision depend on the output "
            "cadence.")
    slot = uh_slot if field == "uh" else REFLECTIVITY_SLOT
    # WHEREVER the domain is keeping it.  Under [tiles] the window is
    # folded in the tile buffers and lands in the store, and the copy on the
    # state stopped moving when streaming.attach took it -- so reading the
    # state would steer the nest on a frozen plane, which is the same defect
    # that swallowed the resets (woof/core/streaming.py:live_scratch).
    # Resident, domain_scratch IS state.existing_scratch(slot).
    buf = streaming.domain_scratch(state, slot)
    if buf is None:
        # Test doubles and reduced states that hang the plane straight off
        # the object rather than in a scratch pool.
        buf = getattr(state, slot, None)
    if buf is None:
        remedy = ("nwp_diagnostics = 1 populates it every step"
                  if field == "uh" else
                  "the microphysics stashes it at history cadence "
                  "(woof.core.refl)")
        raise TrackerRefusal(
            f"the parent state carries no {slot!r} plane, so the tracker's "
            f"{field!r} signal does not exist in this run; {remedy}. "
            "Refused rather than holding forever: a follow block that can "
            "never see its field is a configuration error, not a quiet "
            "storm.")
    plane = np.asarray(_host(buf), dtype=np.float64)
    if plane.ndim == 3:
        plane = plane.max(axis=0)
    if plane.ndim != 2:
        raise TrackerRefusal(
            f"{slot!r} must be a (ny, nx) plane or (nz, ny, nx) volume, "
            f"got shape {plane.shape}")
    return plane


#: Public name for the signal read: the spawn trigger
#: (:mod:`woof.core.nest_spawn`) watches the SAME two parent planes with
#: the SAME missing-slot refusals, so it imports this rather than keeping
#: a second reader that could drift from the tracker's.
signal_plane = _plane_from_state


def weighted_centroid(plane: np.ndarray, threshold: float,
                      box: tuple[slice, slice],
                      radius_cells: float | None = None) -> dict | None:
    """Exceedance-weighted centroid of ``plane >= threshold`` inside
    ``box``, as the FIXED POINT of the bounded-centroid operator.

    Weights are ``value - threshold`` so the centroid sits on the
    signal's core rather than on the area of its threshold-crossing
    plateau (the grid-space adaptation of the WaH echo centroid).
    Returns ``{"ci", "cj", "cells", "max_value", ...}`` in FULL
    parent-array 0-based coordinates, or ``None`` when nothing qualifies.

    WHY A FIXED POINT AND NOT A SINGLE PASS.  Two facts, both measured
    on the reference run's 21 d03 frames:

    * a threshold alone selects whatever it selects -- on a 243 km nest
      whose whole 850 hPa height field spans 41-49 m, a 30 m threshold
      took 44-65% of the grid, CLIPPED BY THE DOMAIN EDGE.  The centroid
      was then partly an average of where the grid stops, which makes
      the nest's own placement an input to the centre steering it;
    * bounding it to a disc fixes that, but a disc has to be anchored,
      and the obvious anchor -- the extremum cell -- is itself unstable:
      the bare minimum jumps 5.94 km per 3-minute frame, max 17.83 km,
      because mesovortices and a broad centre move it around.

    Iterating ``c <- centroid of the deficit within radius of c`` to a
    fixed point removes the anchor from the answer.  MEASURED: started
    from ten points scattered across the whole domain, every frame
    converges (16-31 iterations) to ONE basin, with the ten answers
    within 12-88 METRES of each other.  So the flickering extremum stops
    mattering -- it is a seed, not a determinant.

    Jitter, same 21 frames: 5.94 km for the bare extremum, 5.74 km for a
    disc anchored on it, **4.96 km** for the fixed point.

    WHEN IT DOES NOT CONVERGE TO ONE POINT the field genuinely has more
    than one centre, and that is reported rather than averaged away
    (``competing_centre``).  A centroid of two lows sits between them,
    on neither.

    ``radius_cells`` ``None`` restores the unbounded single pass and
    exists for callers with no grid spacing to convert with; it is not
    reachable from config.
    """
    j_slice, i_slice = box
    window = plane[j_slice, i_slice]
    with np.errstate(invalid="ignore"):
        qualifies = np.isfinite(window) & (window >= threshold)
    cells = int(qualifies.sum())
    if cells == 0:
        return None

    def centroid_of(mask):
        weights = np.where(mask, window - threshold, 0.0)
        total = float(weights.sum())
        if total <= 0.0:
            weights = mask.astype(np.float64)
            total = float(mask.sum())
        jj, ii = np.nonzero(mask)
        w = weights[jj, ii]
        return (float((jj * w).sum() / total),
                float((ii * w).sum() / total))

    if radius_cells is None or cells == 1:
        cj, ci = centroid_of(qualifies)
        return {"ci": ci + i_slice.start, "cj": cj + j_slice.start,
                "cells": cells,
                "max_value": float(window[qualifies].max())}

    ny_w, nx_w = window.shape
    jj, ii = np.ogrid[0:ny_w, 0:nx_w]
    radius2 = float(radius_cells) ** 2
    radius_int = int(math.ceil(float(radius_cells)))
    # Loop-invariant, and MEASURED the single largest cost when it was
    # not: recomputing (window - threshold) inside the iteration was
    # 0.451 ms of a 1.073 ms step.  Hoisting it is bitwise -- the array
    # np.where() sees is the same array either way.
    deficit = window - threshold
    # One full-window scratch buffer, reused.  It exists so `total` is
    # ALWAYS a reduction over shape (ny, nx): numpy's pairwise summation
    # groups by element count, so summing a cropped array rounds
    # differently and a crop there would silently move the answer (it
    # did -- 1 ULP on 5 of 21 real frames, which is why the obvious
    # version of this optimisation was reverted).  Everything else is
    # cropped to the disc's bounding box, which is exact because the
    # disc lies wholly inside it.
    buffer = np.zeros_like(deficit)
    dirty: list[tuple[int, int, int, int]] = []

    def settle(seed):
        """Iterate the bounded centroid from ``seed`` to a fixed point.

        Bitwise identical to the whole-window form, and ~4x cheaper.
        Two facts make the crop exact:

        * ``np.nonzero`` over a contiguous crop that contains every mask
          cell yields those cells in the SAME row-major order as over
          the whole window, so adding the crop origin back (exact
          integer arithmetic) reproduces the same index array, hence the
          same 1-D products and the same 1-D reductions;
        * the only shape-locked reduction, ``buffer.sum()``, is left on
          the full window.
        """
        cj, ci = seed
        found_box = None
        converged = False
        iterations = 0
        for iterations in range(1, CENTROID_MAX_ITERATIONS + 1):
            j0 = max(0, int(math.floor(cj)) - radius_int)
            j1 = min(ny_w, int(math.ceil(cj)) + radius_int + 1)
            i0 = max(0, int(math.floor(ci)) - radius_int)
            i1 = min(nx_w, int(math.ceil(ci)) + radius_int + 1)
            step = qualifies[j0:j1, i0:i1] & (
                ((jj[j0:j1] - cj) ** 2 + (ii[:, i0:i1] - ci) ** 2)
                <= radius2)
            if not step.any():
                break
            for pj0, pj1, pi0, pi1 in dirty:
                buffer[pj0:pj1, pi0:pi1] = 0.0
            dirty.clear()
            np.copyto(buffer[j0:j1, i0:i1], deficit[j0:j1, i0:i1],
                      where=step)
            dirty.append((j0, j1, i0, i1))
            # The EXCEEDANCE total, which is also the basin's weight.
            # Kept separate from the normaliser below: when the field is
            # degenerate the normaliser becomes a cell count, and
            # returning that as the weight would let a bigger-but-flatter
            # basin outrank a deeper one on a tie.
            exceedance = float(buffer.sum())
            wj, wi = np.nonzero(step)
            if exceedance <= 0.0:
                w = np.ones(wj.size, dtype=np.float64)
                norm = float(wj.size)
            else:
                w = buffer[j0:j1, i0:i1][wj, wi]
                norm = exceedance
            new_cj = float(((wj + j0) * w).sum() / norm)
            new_ci = float(((wi + i0) * w).sum() / norm)
            found_box = (step, j0, i0, exceedance)
            moved = math.hypot(new_cj - cj, new_ci - ci)
            cj, ci = new_cj, new_ci
            if moved < CENTROID_TOLERANCE_CELLS:
                converged = True
                break
        if found_box is None:                           # pragma: no cover
            return (cj, ci), None, False, iterations, -math.inf
        step, j0, i0, exceedance = found_box
        # The basin's WEIGHT is what makes it a centre rather than a
        # coincidence: the summed exceedance it is supported by.  It is
        # the same full-window reduction the loop already made.
        return (cj, ci), (step, j0, i0), converged, iterations, exceedance

    # TWO SEEDS, AND THE HEAVIER BASIN WINS.  A fixed point supported by
    # almost nothing is still a fixed point -- seed the iteration on one
    # isolated deep cell and the disc around it can contain only that
    # cell, which is self-consistent and is not a storm.  Seeding also
    # from the region's own unbounded centroid gives a second basin that
    # is multi-cell by construction, and taking the heavier of the two
    # makes the rule explicit rather than lucky.  On a clean field both
    # seeds converge to the same point and the choice is a formality
    # (MEASURED: ten scattered starts, one basin, spread 12-88 m).
    extremum = np.unravel_index(
        int(np.nanargmax(np.where(qualifies, window, -np.inf))),
        window.shape)
    seeds = [(float(extremum[0]), float(extremum[1])), centroid_of(qualifies)]
    best = None
    for seed in seeds:
        candidate = settle(seed)
        if best is None or candidate[4] > best[4]:
            best = candidate
    (cj, ci), mask_box, converged, iterations, _ = best
    if mask_box is None:                                # pragma: no cover
        return None
    step, j0, i0 = mask_box
    # Back to a full-window mask ONCE, outside the iteration, for the
    # reported cell count, the extremum and the competing-centre pass.
    mask = np.zeros_like(qualifies)
    mask[j0:j0 + step.shape[0], i0:i0 + step.shape[1]] = step
    if not mask.any():                                  # pragma: no cover
        return None

    found = {
        "ci": ci + i_slice.start, "cj": cj + j_slice.start,
        "cells": int(mask.sum()),
        "max_value": float(window[mask].max()),
        "iterations": int(iterations),
        "converged": bool(converged),
    }
    # A competing centre: the best qualifying cell OUTSIDE the located
    # core, when it is comparably deep.  One extra pass, and it is the
    # difference between "the storm is here" and "the storm is here, and
    # there is another one that may take over".
    outside = qualifies & ~mask
    if outside.any():
        rival = float(window[outside].max())
        core = float(window[mask].max())
        if core > threshold and (rival - threshold) >= (
                COMPETING_CENTRE_FRACTION * (core - threshold)):
            rj, ri = np.unravel_index(
                int(np.nanargmax(np.where(outside, window, -np.inf))),
                window.shape)
            found["competing_centre"] = {
                "cell_ij": [float(ri) + i_slice.start,
                            float(rj) + j_slice.start],
                "distance_cells": round(math.hypot(float(rj) - cj,
                                                   float(ri) - ci), 3),
                "depth_ratio": round((rival - threshold)
                                     / (core - threshold), 3),
            }
    return found


def locate_signal(plane: np.ndarray, field: str, threshold: float,
                  box: tuple[slice, slice], *,
                  relative_to_minimum: bool = False,
                  radius_cells: float | None = None) -> dict | None:
    """Centroid of ``field``'s EXTREMUM inside ``box``.

    Every signal but pressure is a maximum -- rotation and echo are
    where the field is LARGE -- and :func:`weighted_centroid` weights by
    ``value - threshold``.  Sea-level pressure is the one inverted
    signal: a cyclone is a MINIMUM.  Negating both the plane and the
    threshold turns it into exactly the same problem::

        mask     -mslp >= -ceiling   <=>   mslp <= ceiling
        weight   -mslp -  -ceiling    =    ceiling - mslp

    and that weight is the pressure DEFICIT below the ceiling, so the
    centroid sits on the deep core rather than on the area of the
    closed contour -- which is WRF's vortex weighting, obtained without
    a second centroid implementation to keep in step with the first.

    ``max_value`` comes back on the negated plane; :func:`signal_extremum`
    restores it to real hPa.
    """
    if field != "pressure":
        return weighted_centroid(plane, float(threshold), box, radius_cells)
    ceiling = float(threshold)
    if not relative_to_minimum:
        # Hurricane motion follows the eye. A deficit-weighted area centre
        # moves toward an asymmetric rainband even with the pressure minimum
        # stationary, and can lead the nest away from its eye.
        window = plane[box[0], box[1]]
        valid = np.isfinite(window) & (window <= ceiling)
        if not valid.any():
            return None
        j, i = np.unravel_index(
            int(np.argmin(np.where(valid, window, np.inf))), window.shape)
        return {"ci": float(i + box[1].start),
                "cj": float(j + box[0].start), "cells": 1,
                "max_value": -float(window[j, i]),
                "centre_rule": "MSLP minimum"}
    if relative_to_minimum:
        # THE ISOBARIC THRESHOLD IS RELATIVE, and it has to be.  An MSLP
        # ceiling can be absolute because 1000 hPa means the same thing
        # in every basin and season.  A 850 hPa geopotential height does
        # not: it is ~1500 m in the deep tropics and ~1350 m in a cold
        # airmass, and it drifts under the storm's own warm core -- so an
        # absolute number is a value the user would have to re-tune per
        # case, and would silently track nothing once the environment
        # moved past it.  Anchoring on the search box's own minimum makes
        # the tracker self-calibrating: `threshold` is how deep a slice
        # of the core to weight, and it means the same thing everywhere.
        window = plane[box[0], box[1]]
        finite = window[np.isfinite(window)]
        if finite.size == 0:
            return None
        ceiling = float(finite.min()) + float(threshold)
    return weighted_centroid(-plane, -ceiling, box, radius_cells)


def signal_span(plane: np.ndarray, box: tuple[slice, slice]) -> float | None:
    """Range of finite values inside ``box``; ``None`` if there are none.

    Under a RELATIVE threshold this is the number that says whether the
    threshold can discriminate at all.  ``locate_signal`` builds its
    ceiling as ``box minimum + threshold``, so a box whose whole span is
    below the threshold has EVERY cell qualifying -- and the centroid of
    every cell in a box is the box's own centre, which is the nest's own
    centre, which rounds to a null shift.
    """
    window = plane[box[0], box[1]]
    finite = window[np.isfinite(window)]
    if finite.size == 0:
        return None
    return float(finite.max()) - float(finite.min())


def signal_closure(plane: np.ndarray, box: tuple[slice, slice],
                   threshold: float, *, centre=None,
                   radius_cells: float | None = None) -> dict:
    """Did the SEARCH BOX help decide where the centre is?

    :func:`signal_span` answers whether the threshold can discriminate at
    all.  This answers the question after it, and the two are not the
    same: a search box with a large span has a minimum, but a minimum is
    not a vortex.  Over open ocean under a synoptic gradient the box
    spans hundreds of metres, the threshold discriminates perfectly, and
    the cells within ``threshold`` of the box minimum are a STRIP ALONG
    THE BOX EDGE -- the lowest corner of a slope the box happened to cut.
    The fixed point settles in that strip because the box stopped it
    there, so the shift it implies is the distance from the nest to the
    box's own corner, proposed on no storm.

    :func:`weighted_centroid` names this failure itself: a centroid over
    a region the domain edge clipped "was then partly an average of where
    the grid stops, which makes the nest's own placement an input to the
    centre steering it".  So the criterion is that sentence, tested:

        closed  <=>  none of the cells that PRODUCED the centre lie on a
                     box edge the plane continues past

    The cells that produced the centre are the qualifying cells within
    ``radius_cells`` of it, which is the disc :func:`weighted_centroid`
    draws its answer from; with no radius configured it reduces to the
    single unbounded pass over the whole box, and so does this test.
    There is no tuning constant in either form -- the radius is the
    configured ``radius_km``, and a box side that coincides with the
    plane's own edge is not counted at all, because that is the signal
    leaving the grid rather than a window cut out of a larger field, and
    :func:`woof.core.track_boundary.boundary_reason` and a refine grid's
    ``edge_margin_cells`` already say so in their own words.

    MEASURED at seven hourly instants of each case, with a 5-cell radius
    (60 km at 12 km): on the open-ocean point the settled centre sat
    2.41 to 3.13 cells from an open box edge, every instant; on a
    978.7 hPa cyclone the same tracker sat 25.14 to 39.54 cells from one,
    every instant.  The counts come back beside the verdict, so a hold
    receipt says how close the call was rather than only that it was
    made.
    """
    ny, nx = int(plane.shape[-2]), int(plane.shape[-1])
    j_slice, i_slice = box
    window = plane[j_slice, i_slice]
    finite = np.isfinite(window)
    # ONE SHAPE on every return, including this one: the function is
    # public, and a caller reading a key off the blank verdict that
    # every other path carries would raise KeyError on exactly the
    # empty-or-all-NaN box it exists to report.
    blank = {"closed": False, "qualifying_cells": 0,
             "boundary_qualifying_cells": 0, "boundary_cells_in_box": 0,
             "box_cells": int(window.size),
             "open_sides": 0, "centre_margin_cells": None,
             "radius_cells": (None if radius_cells is None
                              else round(float(radius_cells), 3))}
    if window.size == 0 or not bool(finite.any()):
        return blank
    floor_value = float(window[finite].min())
    with np.errstate(invalid="ignore"):
        qualifies = finite & (window <= floor_value + float(threshold))
    ring = np.zeros(window.shape, dtype=bool)
    margins = []
    sides = 0
    jj, ii = np.mgrid[j_slice.start:j_slice.stop, i_slice.start:i_slice.stop]
    if int(j_slice.start) > 0:
        ring[0, :] = True
        sides += 1
        if centre is not None:
            margins.append(float(centre[1]) - int(j_slice.start))
    if int(j_slice.stop) < ny:
        ring[-1, :] = True
        sides += 1
        if centre is not None:
            margins.append(int(j_slice.stop) - 1 - float(centre[1]))
    if int(i_slice.start) > 0:
        ring[:, 0] = True
        sides += 1
        if centre is not None:
            margins.append(float(centre[0]) - int(i_slice.start))
    if int(i_slice.stop) < nx:
        ring[:, -1] = True
        sides += 1
        if centre is not None:
            margins.append(int(i_slice.stop) - 1 - float(centre[0]))
    contact = qualifies & ring
    if centre is not None and radius_cells is not None:
        # Only the cells the answer was actually drawn from.  A trailing
        # trough that reaches the box edge far from the centre did not
        # move the centroid a millimetre, and holding a nest on it would
        # be refusing a storm that is plainly there.
        reach = ((ii - float(centre[0])) ** 2
                 + (jj - float(centre[1])) ** 2) <= float(radius_cells) ** 2
        contact = contact & reach
    on_edge = int(np.count_nonzero(contact))
    return {"closed": on_edge == 0,
            "qualifying_cells": int(np.count_nonzero(qualifies)),
            "boundary_qualifying_cells": on_edge,
            "boundary_cells_in_box": int(np.count_nonzero(qualifies & ring)),
            "box_cells": int(window.size),
            "open_sides": sides,
            "centre_margin_cells": (None if not margins
                                    else round(min(margins), 3)),
            "radius_cells": (None if radius_cells is None
                             else round(float(radius_cells), 3))}


def centre_over_levels(planes, config, box, radius_cells):
    """Locate the vortex on every plane and MEAN the answers.

    Returns ``(found, level_fixes, declined)``.  ``found`` is the same
    dict shape a single-plane search returns -- so every caller
    downstream is unchanged -- carrying the mean centre, the summed
    qualifying cells and the PRIMARY (first-named) level's extremum.

    WHY THE MEAN.  A sheared tropical cyclone's 850 hPa and 500 hPa
    circulations are genuinely in different places; a nest steered on
    either alone chases a tilt the other does not have, and the nest's
    job is to contain the whole storm.  The deep-layer mean is the
    standard steering centre for exactly that reason.  It is a plain
    unweighted mean because there is no measured basis here for
    weighting one surface over another, and an invented weight would be
    a knob nobody could set accurately.

    A level that cannot produce a centre -- the surface is underground
    everywhere, or nothing on it qualifies -- DECLINES rather than
    refusing, and the mean is over the levels that did.  Losing 700 hPa
    over high terrain is not a reason to stop following a storm the
    other two surfaces can see.
    """
    relative = bool(levels_of(config))
    # WHICH SURFACES VOTE.  Every plane gets the same search and its own
    # LevelFix; only the steering set is averaged into the centre the
    # nest follows.  A report-only surface is there so a forecaster can
    # read the vortex's tilt out of the track file, and letting it steer
    # would be the bug -- a 200 hPa outflow centre has no business
    # pulling a nest off the eyewall.
    steering = frozenset(round(float(v), 6) for v in levels_of(config))
    fixes, declined = [], []
    for level, plane in planes:
        if level is None and config.field == "pressure" and config.pressure_center == "minimum":
            window = np.asarray(plane[box])
            finite = np.isfinite(window) & (window <= float(config.threshold))
            if not finite.any():
                return None, (), [{"reason": "no finite MSLP minimum below the pressure ceiling"}]
            j, i = np.unravel_index(np.argmin(np.where(finite, window, np.inf)), window.shape)
            return {"ci": float(i + box[1].start), "cj": float(j + box[0].start),
                    "cells": 1, "max_value": -float(window[j, i]),
                    "converged": True, "iterations": 1}, (), []
        if relative:
            # THE FLAT BOX.  A relative threshold is self-calibrating,
            # which is why level_hpa never needs re-tuning per basin --
            # and this is what that costs.  When the storm has left the
            # search box entirely, the box is nearly flat, EVERY cell
            # clears `box minimum + threshold`, and the centroid of every
            # cell in a box is the box's own centre.  That is the nest's
            # own centre, so the shift rounds to zero and the tracker
            # reports `suppressed:dead-band` -- which is
            # indistinguishable from "the storm is exactly where it
            # should be".  A nest that has lost its storm must not look
            # like a nest that is perfectly centred.
            #
            # The criterion needs no tuning constant: span < threshold IS
            # "every cell qualifies", exactly.
            span = signal_span(plane, box)
            if span is not None and span < float(config.threshold):
                declined.append({
                    "level_hpa": level,
                    "signal_span": round(span, 4),
                    "reason": (
                        f"the search box spans only {span:.3f} against a "
                        f"threshold of {float(config.threshold):g}, so every "
                        "cell in it qualifies and the centroid would be the "
                        "box's own centre; the storm is not in the box"),
                })
                continue
        try:
            if config.field == "attribute":
                minimum = config.extremum == "min"
                found = weighted_centroid(
                    -plane if minimum else plane,
                    -float(config.threshold) if minimum else float(config.threshold),
                    box, radius_cells)
                if found is not None:
                    found["attribute_extremum"] = config.extremum
            else:
                found = locate_signal(plane, config.field,
                                      float(config.threshold), box,
                                      relative_to_minimum=relative,
                                      radius_cells=radius_cells)
        except TrackerRefusal as error:
            declined.append({"level_hpa": level, "reason": str(error)})
            continue
        if found is None:
            declined.append({"level_hpa": level,
                             "reason": "nothing qualified on this surface"})
            continue
        if relative:
            # THE SLOPED BOX, which the span test above cannot see.  A box
            # under a synoptic gradient spans far more than the threshold,
            # so the threshold discriminates perfectly and still selects a
            # strip along the box's own lowest edge; the fixed point
            # settles in that strip because the box stopped it there, and
            # the shift it implies is the distance to the box's own
            # corner.  Judged on the ANSWER rather than on the input,
            # because the question is whether the box helped decide where
            # the centre is: closure asks whether any cell that produced
            # this centre lies on a box edge the plane continues past.
            closure = signal_closure(plane, box, float(config.threshold),
                                     centre=(found["ci"], found["cj"]),
                                     radius_cells=radius_cells)
            if not closure["closed"]:
                declined.append({
                    "level_hpa": level,
                    "signal_span": (None if span is None else round(span, 4)),
                    **closure,
                    "reason": (
                        "the centre settled "
                        f"{closure['centre_margin_cells']} cells from a "
                        "search-box edge the field continues past, inside "
                        f"the {closure['radius_cells']}-cell radius its own "
                        f"centroid draws from, and "
                        f"{closure['boundary_qualifying_cells']} of the "
                        "cells that produced it lie on that edge; the box "
                        "helped decide where this centre is, so it is a "
                        "position on the box rather than on a storm"),
                })
                continue
        if level is None:
            return found, (), declined
        fixes.append((found, LevelFix(
            level_hpa=float(level), ci=float(found["ci"]),
            cj=float(found["cj"]),
            height_m=_bilinear(plane, found["ci"], found["cj"]),
            cells=int(found["cells"])),
            round(float(level), 6) in steering))
    # The file gets every surface that answered, in configured order.
    level_fixes = tuple(f for _, f, _ in fixes)
    voting = [(d, f) for d, f, votes in fixes if votes]
    if not voting:
        # No STEERING surface answered.  There is no centre to follow --
        # even if report-only surfaces found one, they are not what this
        # nest is steered by, and inventing a centre from them would move
        # the nest on a signal its configuration says not to use.  The
        # level fixes still come back, so the hold is auditable and the
        # receipt can say which surfaces did answer.
        return None, level_fixes, declined
    primary = voting[0][0]
    merged = dict(primary)
    merged["ci"] = sum(f.ci for _, f in voting) / len(voting)
    merged["cj"] = sum(f.cj for _, f in voting) / len(voting)
    merged["cells"] = sum(int(d["cells"]) for d, _ in voting)
    if len(voting) > 1:
        merged["levels_averaged"] = len(voting)
    if len(level_fixes) > len(voting):
        merged["levels_reported"] = len(level_fixes)
    return merged, level_fixes, declined


def extremum_kind(config, field_used: str) -> str:
    """``"minimum"`` or ``"maximum"``: which end of the field is the centre.

    ONE answer, because three sites ask it -- the receipt's
    ``extremum_kind`` column, the units that column is paired with, and
    :func:`woof.core.track_boundary.boundary_reason`, which has to know
    which end of the plane to look for at the grid edge.  Read off the
    CONFIG for a tracked attribute (its own ``extremum``) and off the
    field otherwise, with ``field_used`` rather than ``config.field``
    because the rotation tracker's automatic echo handoff changes the
    field under it and the echo is a maximum either way.
    """

    if config.field == "attribute":
        return "minimum" if config.extremum == "min" else "maximum"
    return "minimum" if field_used == "pressure" else "maximum"


def signal_extremum(found: dict, field: str) -> float:
    """The extremum in the FIELD'S OWN units and sign.

    Undoes :func:`locate_signal`'s negation, so a pressure receipt
    records the minimum MSLP in hPa rather than its negative.
    """
    value = float(found["max_value"])
    return -value if (field == "pressure" or
                      (field == "attribute" and found.get("attribute_extremum") == "min")) else value


def radius_in_cells(radius_km: float, dx_m: float | None) -> float | None:
    """``radius_km`` on a grid of spacing ``dx_m``, in cells.

    ``None`` when the spacing is unknown -- a hand-built footprint, a
    test double -- and the centroid is then unbounded, which is the
    behaviour that predates the bound.  Config cannot reach this: every
    domain the loader builds carries a resolved ``run.dx``.
    """
    if dx_m is None or not math.isfinite(float(dx_m)) or float(dx_m) <= 0.0:
        return None
    return float(radius_km) * 1000.0 / float(dx_m)


def _round_cells(value: float) -> int:
    """Symmetric round-half-away-from-zero, so eastward and westward
    proposals of equal magnitude round identically."""
    if value >= 0.0:
        return int(math.floor(value + 0.5))
    return -int(math.floor(-value + 0.5))


@dataclass(frozen=True)
class LevelFix:
    """One isobaric surface's own answer.

    Carried per level so a track file can report where each surface put
    the centre and how high that surface was there -- which is the
    picture a forecaster reads a tilt off, and the tracker's mean cannot
    show on its own.
    """

    level_hpa: float
    #: Centre in the frame the fix's shifts are expressed in.
    ci: float
    cj: float
    #: Geopotential height AT that centre, metres (dam = /10).
    height_m: float
    cells: int
    #: The centre in the index space of the grid it was FOUND on, when
    #: that is not the frame ``ci``/``cj`` are expressed in.  Under the
    #: two-stage tracker ``ci``/``cj`` are mapped into the mover's-parent
    #: frame for the mean and the spread, but a latitude must be taken
    #: from the grid that found the centre, in that grid's own index
    #: space -- NestFootprint and ProjectedGrid.nest use different
    #: half-cell conventions and mixing them costs a fraction of a cell
    #: (see storm_track_writer.latlon_from_grid).
    source_ij: "tuple[float, float] | None" = None

    @property
    def fix_ij(self) -> tuple[float, float]:
        """Where to sample a projection: the source grid's own cell."""
        return self.source_ij if self.source_ij is not None else (self.ci,
                                                                  self.cj)

    @property
    def height_dam(self) -> float:
        return self.height_m / 10.0


def _bilinear(plane, ci: float, cj: float) -> float:
    """``plane`` sampled at a fractional cell, clamped to the field.

    Bilinear rather than nearest because the centre is fractional by
    construction and a reported height should not carry the grid's
    quantisation -- 643 m of position is ~1 m of 850 hPa height here,
    which is a digit a reader would otherwise see move for no reason.
    """
    ny, nx = plane.shape[-2], plane.shape[-1]
    ci = min(max(float(ci), 0.0), nx - 1.0)
    cj = min(max(float(cj), 0.0), ny - 1.0)
    i0, j0 = int(math.floor(ci)), int(math.floor(cj))
    i1, j1 = min(i0 + 1, nx - 1), min(j0 + 1, ny - 1)
    fi, fj = ci - i0, cj - j0
    top = plane[j0, i0] * (1.0 - fi) + plane[j0, i1] * fi
    bottom = plane[j1, i0] * (1.0 - fi) + plane[j1, i1] * fi
    return float(top * (1.0 - fj) + bottom * fj)


def levels_of(config) -> tuple:
    """The surfaces that STEER: the terms of the deep-layer mean.

    ``()`` when the tracker is on one plane (uh, reflectivity, sea-level
    pressure).  This is the set :func:`centre_over_levels` averages, and
    it is deliberately NOT the set the track file reports -- see
    :func:`all_levels_of`.
    """
    return tuple(getattr(config, "level_hpa", None) or ())


def report_levels_of(config) -> tuple:
    """The surfaces computed for the FILE only, never for the mean."""
    return tuple(getattr(config, "report_level_hpa", None) or ())


def all_levels_of(config) -> tuple:
    """Every surface a consultation computes a centre on.

    Steering surfaces first, in the order ``level_hpa`` named them, then
    the report-only ones in the order ``output_level`` asked for them --
    so the track file's columns stay in a stable, configured order and a
    file written with report levels is the same columns plus more, never
    reshuffled.
    """
    return levels_of(config) + report_levels_of(config)


@dataclass(frozen=True)
class VortexFix:
    """One consultation's answer: where the storm is, and on what.

    Returned by :meth:`StormTracker.locate`.  It is the whole of what a
    consumer other than the mover needs, and deliberately carries no
    decision: ``desired_shift`` turns it into a shift, and
    :mod:`woof.core.storm_track_writer` turns it into a track row, from
    the SAME fix -- so a run's deck and its nest can never disagree
    about where the vortex was.

    ``found`` is ``None`` for a no-signal consultation, and every other
    field is still populated (the search box, the footprint, the
    evidence), because a hold has to be auditable too.

    ``refined_on`` / ``refined_cell_ij`` name the descendant grid whose
    own field produced the centre, and the centre's 0-based cell index
    ON THAT GRID.  They are ``None`` whenever stage two did not apply --
    no ``refine_grid_id``, no source, or a decline -- and a consumer that
    wants the finest available field reads them to know which state to
    ask.  The measured reason they exist is in :class:`RefinementSource`:
    a 13.5 km parent put the same vortex 44.9 km from where the 1.125 km
    grid put it.
    """

    footprint: NestFootprint
    evidence: dict
    found: dict | None
    #: (ny, nx) of the plane stage one searched.
    plane_shape: tuple[int, int]
    search_box: tuple[slice, slice]
    field_used: str
    threshold_used: float
    #: Centre minus footprint centre, in fractional parent cells.
    raw_shift: tuple[float, float]
    refined_on: int | None = None
    refined_cell_ij: tuple[float, float] | None = None
    #: One entry per isobaric surface that produced a centre, in the
    #: order the config named them.  Empty for a single-plane tracker
    #: (uh, reflectivity, sea-level pressure).
    levels: tuple = ()

    @property
    def center_parent_ij(self) -> tuple[float, float] | None:
        """The vortex centre in the mover's-parent 0-based cell frame."""
        if self.found is None:
            return None
        return (float(self.found["ci"]), float(self.found["cj"]))

    @property
    def extremum(self) -> float | None:
        """The signal extremum in the FIELD'S own units and sign."""
        if self.found is None:
            return None
        return signal_extremum(self.found, self.field_used)


# ---------------------------------------------------------------------------
# The tracker (the plan provider)
# ---------------------------------------------------------------------------

#: The two cooldown anchors, as a checkpoint's follower entry names them.
TRACKER_STATE_KEYS = ("last_proposal_t", "last_move_t")
#: The tracker's optional restart keys: the MSLP motion fix, and the two
#: halves of the parking rule (when the storm-identity test last started
#: failing, and the park itself once the track has ended).
TRACKER_OPTIONAL_STATE_KEYS = ("motion_fix", "lost_since", "parked")

# ---------------------------------------------------------------------------
# Storm identity (MSLP-minimum following) and parking
# ---------------------------------------------------------------------------
#
# THE BREAKAGE THIS CLOSES (WOOF 1.1 readiness g4, GFS Isaias, 12 km parent
# with a 3 km following nest): an MSLP-minimum follower took the lowest cell
# of its search box as "the centre" under an absolute 1100 hPa ceiling that
# every cell clears.  After the storm left the parent (the track writer ended
# at 2026-10-12_08:45) the nest made 60+ more moves on no storm -- pinned on
# the parent's east edge, then to Iowa and back to Pennsylvania along
# search-box-edge minima -- so the d02 maps from about f040 showed farmland.
#
# The rules, in order, for pressure_center = "minimum":
#   1. The track writer's end (the tracked minimum reached the PARENT's own
#      boundary) parks the nest for the rest of the run, for every tracked
#      field: the mover never follows a fix the track itself has ended on.
#   2. A minimum is a storm centre only if it is a closed low: not on an
#      interior edge of the search box (the field continues past it, so it is
#      a slope's lowest cell) and at least IDENTITY_MIN_DEFICIT_HPA below the
#      mean MSLP on a ring IDENTITY_RING_KM out.  Measured on GFS 0.25 deg
#      MSLP for this storm (2026-10-11 06Z, f003-f039): the storm's own
#      centre is 2.32-4.42 hPa below its 200 km ring at every lead; the
#      lowest cell of a 480 km box with no storm in it is -0.18-+0.98 hPa,
#      except where the box edge is beside a real low just outside it.
#   3. Continuity: a minimum farther from the last accepted fix than the
#      storm can travel (IDENTITY_MAX_SPEED_M_S, the GFDL vortex tracker's
#      60 kt maxspeed_tc, plus the follow radius for the centre's own
#      jitter) is not taken while a closed low remains within that reach,
#      unless it is deeper than that low by IDENTITY_MIN_DEFICIT_HPA (the
#      deeper centre has taken over, as a coastal redevelopment does).  On
#      the readiness run the argmin flipped between two centres 100 km apart
#      every 15 min (10-11 10:45-12:30), and the nest followed each flip.
#   4. A follower with no centre passing 2-3 for IDENTITY_LOST_AFTER_S parks
#      and its track ends: a storm that has had no closed low for three
#      hours is gone, and the next closed low in the box is another system.

#: Ring-mean MSLP minus the centre that makes a minimum a closed low (hPa).
IDENTITY_MIN_DEFICIT_HPA = 1.5
#: Radius of that ring (km).
IDENTITY_RING_KM = 200.0
#: Fastest storm translation the continuity rule allows (m/s; 60 kt).
IDENTITY_MAX_SPEED_M_S = 30.87
#: Seconds without an identified centre after which the follower parks.
IDENTITY_LOST_AFTER_S = 10800.0


def ring_deficit_hpa(plane, ci: float, cj: float,
                     ring_cells: float) -> float | None:
    """Mean of ``plane`` on a one-cell ring ``ring_cells`` out, minus the centre.

    ``None`` when less than half of the ring lies on finite cells of the
    plane (a centre near the parent's edge), which cannot be judged here --
    the parent-boundary rule owns that case.
    """
    values = np.asarray(plane)
    ny, nx = values.shape[-2], values.shape[-1]
    j_c, i_c = int(round(cj)), int(round(ci))
    if not (0 <= j_c < ny and 0 <= i_c < nx):
        return None
    centre = float(values[j_c, i_c])
    if not math.isfinite(centre):
        return None
    reach = int(math.ceil(ring_cells + 1))
    jj, ii = np.mgrid[-reach:reach + 1, -reach:reach + 1]
    on_ring = np.abs(np.hypot(ii, jj) - float(ring_cells)) <= 0.5
    expected = int(np.count_nonzero(on_ring))
    rj, ri = jj[on_ring] + j_c, ii[on_ring] + i_c
    inside = (rj >= 0) & (rj < ny) & (ri >= 0) & (ri < nx)
    ring = values[rj[inside], ri[inside]]
    ring = ring[np.isfinite(ring)]
    if expected == 0 or ring.size * 2 < expected:
        return None
    return float(ring.mean()) - centre


def _descend(values, box, ci: float, cj: float) -> tuple[int, int] | None:
    """Steepest descent from (ci, cj) to a local minimum inside ``box``."""
    j0, j1 = int(box[0].start), int(box[0].stop)
    i0, i1 = int(box[1].start), int(box[1].stop)
    i, j = int(round(ci)), int(round(cj))
    if not (j0 <= j < j1 and i0 <= i < i1) or not np.isfinite(values[j, i]):
        return None
    for _ in range((j1 - j0) * (i1 - i0)):
        best = (float(values[j, i]), i, j)
        for dj in (-1, 0, 1):
            for di in (-1, 0, 1):
                jn, in_ = j + dj, i + di
                if j0 <= jn < j1 and i0 <= in_ < i1:
                    value = float(values[jn, in_])
                    if math.isfinite(value) and value < best[0]:
                        best = (value, in_, jn)
        if (best[1], best[2]) == (i, j):
            return i, j
        i, j = best[1], best[2]
    return i, j


def on_interior_box_edge(ci: int, cj: int, box: tuple[slice, slice],
                         shape: tuple[int, int]) -> bool:
    """Whether a cell lies on a search-box edge that is not the parent's edge."""
    ny, nx = int(shape[-2]), int(shape[-1])
    j0, j1 = int(box[0].start), int(box[0].stop)
    i0, i1 = int(box[1].start), int(box[1].stop)
    return ((cj in (j0, j1 - 1) and cj not in (0, ny - 1))
            or (ci in (i0, i1 - 1) and ci not in (0, nx - 1)))


def _timer(value, what: str) -> float | None:
    """A cooldown anchor that survives the header's ``allow_nan=False``."""
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise TrackerRefusal(
            f"the tracker's {what} is {value!r}; a non-finite cooldown "
            "anchor compares false against every model time, so the "
            "hysteresis would never suppress and the nest would move at "
            "every cadence boundary")
    return number


class StormTracker:
    """Feature tracking plus hysteresis; implements the plan-provider seam.

    Stateful only in the two ways the contract needs: the model time of
    the last proposal (cooldown) and the receipts ledger.  Position state
    lives with the runner -- the footprint is an argument every call -- so
    a tracker never disagrees with the runner about where the nest is.
    """

    def __init__(self, config: FollowConfig, *, uh_slot: str = UH_SLOT) -> None:
        if not isinstance(config, FollowConfig):
            raise TypeError("config must be a FollowConfig")
        self.config = config
        if not uh_diag.is_tracker_window_slot(uh_slot):
            raise ValueError(f"invalid tracker UH slot {uh_slot!r}")
        self.uh_slot = str(uh_slot)
        self._last_proposal_t: float | None = None
        self._last_move_t: float | None = None
        self._motion_fix = None
        #: Model time the storm-identity test started failing (None while
        #: it passes), and the park once the track has ended (rules 1, 4).
        self._lost_since: float | None = None
        self._parked: dict | None = None
        self.move_interval_seconds = None
        #: The most recent :meth:`locate` answer, so a consumer that runs
        #: at the SAME boundary as the mover (the track writer, at its
        #: default interval) reads the fix the decision was made from
        #: instead of paying for a second reduction to re-derive it.
        #: Never a substitute for calling locate -- it is only ever the
        #: last one, and a caller at a different instant must ask.
        self.last_fix: "VortexFix | None" = None
        self.receipts: list[dict] = []
        self._receipt({"decision": "configured",
                       "config": config.to_json()})

    # -- receipts ----------------------------------------------------------

    def _receipt(self, entry: dict) -> dict:
        entry = {"contract": FOLLOW_CONTRACT, **entry}
        self.receipts.append(entry)
        _LOG.info("storm-follow %s", entry)
        return entry

    def drain_receipts(self) -> list[dict]:
        """Hand the accumulated receipts to the runner's ledger."""
        out, self.receipts = self.receipts, []
        return out

    # -- the restart state -------------------------------------------------

    def state_json(self) -> dict:
        """The tracker's half of a checkpoint's per-follower entry.

        Cooldown anchors and, for MSLP following, the previous motion fix.
        The latter preserves the measured movement budget across restart.
        Pure state -- no I/O, no checkpoint vocabulary.
        """
        return {**({"motion_fix": self._motion_fix}
                   if self.config.field == "pressure" and self.config.level_hpa is None else {}),
                **({"lost_since": _timer(self._lost_since, "lost_since")}
                   if self._lost_since is not None else {}),
                **({"parked": dict(self._parked)}
                   if self._parked is not None else {}),
                "last_proposal_t": _timer(self._last_proposal_t,
                                          "last_proposal_t"),
                "last_move_t": _timer(self._last_move_t, "last_move_t")}

    def restore_state(self, block) -> None:
        """Seed the cooldown anchors from a checkpoint's follower entry.

        Restored rather than recomputed because there is nothing to
        recompute from: a tracker that just moved its nest and forgets it
        proposes again at the resumed run's FIRST cadence boundary, moving
        the nest a full cooldown early and spinning up the strip it just
        paid for.  The two anchors are restored separately because they
        mean different things -- an executed move and a proposal the runner
        declined -- and collapsing them re-anchors the whole hysteresis.
        """
        if not isinstance(block, dict):
            raise TrackerRefusal(
                "a tracker restart block must be a mapping, got "
                f"{type(block).__name__}")
        unknown = sorted(set(block) - set(TRACKER_STATE_KEYS)
                         - set(TRACKER_OPTIONAL_STATE_KEYS))
        missing = sorted(set(TRACKER_STATE_KEYS) - set(block))
        if unknown or missing:
            raise TrackerRefusal(
                f"a tracker restart block has key(s) {unknown} this build "
                f"does not know and is missing {missing}; the cooldown is "
                "honored in full or refused, because an anchor restored by "
                "omission lets the nest move at the first boundary of the "
                "resumed run")
        proposal = _timer(block["last_proposal_t"], "last_proposal_t")
        move = _timer(block["last_move_t"], "last_move_t")
        motion = block.get("motion_fix")
        if motion is not None:
            if (not isinstance(motion, (list, tuple)) or len(motion) != 3
                    or not all(isinstance(v, (int, float)) and math.isfinite(v)
                               for v in motion) or motion[0] < 0):
                raise TrackerRefusal("motion_fix must be [model_seconds, parent_i, parent_j]")
            motion = list(map(float, motion))
        parked = block.get("parked")
        if parked is not None and (not isinstance(parked, dict)
                                   or not isinstance(parked.get("reason"), str)):
            raise TrackerRefusal("parked must be a mapping with its reason")
        self._lost_since = _timer(block.get("lost_since"), "lost_since")
        self._parked = None if parked is None else dict(parked)
        self._motion_fix = motion
        self._last_proposal_t = proposal
        self._last_move_t = move

    # -- the contract ------------------------------------------------------

    def _refine(self, found, refinement, evidence: dict, config=None):
        """Stage two: relocate the centre on the domain that resolves it.

        Stage one's centre is only as good as the mover's PARENT, and on
        a 13.5 km parent that is 45-57 km from where the 1.125 km grid
        puts the same vortex (MEASURED, see :class:`RefinementSource`).
        This re-locates the SAME signal, with the SAME threshold rule, on
        the refine grid's own field, and converts the answer back to
        parent cells.

        The refine grid is searched WHOLE rather than in a box around
        stage one's centre: the grid exists to contain the storm, so its
        extremum IS the vortex, and boxing the search around a centre
        known to be tens of km wrong would just re-import that error.

        Declines -- returning stage one's answer untouched -- when the
        refined centre lands within ``edge_margin_cells`` of the refine
        grid's boundary.  That is the storm leaving the domain that was
        supposed to contain it, and there the coarse field is the more
        trustworthy of the two because it can still see outside.
        """

        cfg = self.config if config is None else config
        try:
            planes = planes_for(refinement.state, cfg)
        except Exception as error:                  # noqa: BLE001
            evidence["refinement"] = {
                "grid_id": int(refinement.grid_id),
                "applied": False,
                "declined": f"signal unavailable on the refine grid: {error}",
            }
            return found
        plane = planes[0][1]
        ny, nx = int(plane.shape[-2]), int(plane.shape[-1])
        box = (slice(0, ny), slice(0, nx))
        # Stage two re-runs the SAME multi-level rule on the grid that
        # resolves the vortex, so a deep-layer centre stays a deep-layer
        # centre after refinement rather than collapsing to one surface.
        fine, fine_levels, fine_declined = centre_over_levels(
            planes, cfg, box, radius_in_cells(cfg.radius_km,
                                              refinement.dx_m))
        if fine is None:
            evidence["refinement"] = {
                "grid_id": int(refinement.grid_id),
                "applied": False,
                "declined": "no signal on the refine grid",
            }
            return found
        margin = int(refinement.edge_margin_cells)
        if not (margin <= fine["ci"] <= nx - 1 - margin
                and margin <= fine["cj"] <= ny - 1 - margin):
            evidence["refinement"] = {
                "grid_id": int(refinement.grid_id),
                "applied": False,
                "declined": (
                    "refined centre is within "
                    f"{margin} cells of the refine grid's edge; the storm "
                    "is leaving the domain that should contain it, and only "
                    "the parent can still see where it goes"),
                "refined_cell_ij": [round(fine["ci"], 3),
                                    round(fine["cj"], 3)],
            }
            return found
        ci, cj = refinement.to_parent(fine["ci"], fine["cj"])
        moved_i = ci - found["ci"]
        moved_j = cj - found["cj"]
        evidence["refinement"] = {
            "grid_id": int(refinement.grid_id),
            "applied": True,
            "coarse_centroid_parent_ij": [round(found["ci"], 3),
                                          round(found["cj"], 3)],
            "refined_centroid_parent_ij": [round(ci, 3), round(cj, 3)],
            "correction_parent_cells": [round(moved_i, 3), round(moved_j, 3)],
            "refine_cells_above_threshold": fine["cells"],
            "refine_extremum": round(signal_extremum(fine, cfg.field), 3),
        }
        if cfg.field == "attribute":
            evidence["refinement"].update(attribute_metadata(cfg))
            evidence["refinement"]["refine_extremum"] = signal_extremum(fine, cfg.field)
        # The refine grid is the one that RESOLVES the vortex, so its
        # convergence and its rival are the ones worth reading -- a
        # competing centre on a 4.5 km parent is two cells apart and
        # means nothing; on a 643 m nest it is a reforming circulation.
        for key in ("iterations", "converged", "competing_centre"):
            if key in fine:
                evidence["refinement"][f"refine_{key}"] = fine[key]
        refined = dict(found)
        refined["ci"] = ci
        refined["cj"] = cj
        # The centre's own cell index ON THE REFINE GRID, kept so a
        # consumer that wants a quantity from the grid that resolves the
        # vortex (central pressure, peak wind) does not have to invert
        # RefinementSource's map to find out where to look.
        refined["refine_cell_ij"] = (float(fine["ci"]), float(fine["cj"]))
        refined["refine_max_value"] = float(fine["max_value"])
        if fine_levels:
            # Each surface's own centre, mapped through the placement
            # chain into the frame every shift here is expressed in, so
            # the track file reports the FINE grid's per-level answer
            # rather than the parent's coarser one.
            mapped = []
            for lf in fine_levels:
                pci, pcj = refinement.to_parent(lf.ci, lf.cj)
                mapped.append(LevelFix(level_hpa=lf.level_hpa, ci=pci,
                                       cj=pcj, height_m=lf.height_m,
                                       cells=lf.cells,
                                       source_ij=(lf.ci, lf.cj)))
            refined["refine_levels"] = tuple(mapped)
            evidence["refinement"]["refine_levels"] = [
                {"level_hpa": f.level_hpa,
                 "centre_parent_ij": [round(f.ci, 3), round(f.cj, 3)],
                 "height_dam": round(f.height_dam, 2)} for f in mapped]
        if fine_declined:
            evidence["refinement"]["refine_levels_declined"] = fine_declined
        return refined

    def _storm_identity(self, plane, box, found, fp, t: float,
                        radius_km: float):
        """Rules 2-3: is this MSLP minimum the storm?  ``(found | None, record)``.

        Reads the last accepted fix (``_motion_fix``) and changes nothing,
        so :meth:`locate` stays stateless.  ``record`` goes on the receipt
        whatever the answer, with each candidate's deficit, so a hold says
        how close the call was.
        """
        values = np.asarray(plane)
        if fp.parent_dx_m is None or not fp.parent_dx_m > 0:
            return found, {"applied": False,
                           "reason": "the footprint carries no parent dx"}
        dx_m = float(fp.parent_dx_m)
        ring_cells = IDENTITY_RING_KM * 1000.0 / dx_m

        def judge(ci: int, cj: int) -> dict:
            deficit = ring_deficit_hpa(values, ci, cj, ring_cells)
            edge = on_interior_box_edge(ci, cj, box, values.shape)
            return {"ci": int(ci), "cj": int(cj),
                    "mslp_hpa": round(float(values[cj, ci]), 3),
                    "deficit_hpa": (None if deficit is None
                                    else round(deficit, 3)),
                    "on_search_box_edge": bool(edge),
                    "closed": bool(not edge and (
                        deficit is None
                        or deficit >= IDENTITY_MIN_DEFICIT_HPA))}

        def accept(candidate: dict) -> dict:
            return {"ci": float(candidate["ci"]), "cj": float(candidate["cj"]),
                    "cells": 1, "max_value": -float(candidate["mslp_hpa"]),
                    "converged": True, "iterations": 1}

        lowest = judge(int(round(found["ci"])), int(round(found["cj"])))
        record = {"min_deficit_hpa": IDENTITY_MIN_DEFICIT_HPA,
                  "ring_km": IDENTITY_RING_KM, "lowest": lowest}
        ny, nx = values.shape[-2], values.shape[-1]
        if lowest["ci"] in (0, nx - 1) or lowest["cj"] in (0, ny - 1):
            # On the parent's own edge: rule 1 (the track writer's
            # boundary test) decides whether the storm has left.
            record["decision"] = "parent-edge"
            return found, record
        anchor = self._motion_fix
        reach = None
        if anchor is not None and t > float(anchor[0]):
            reach = ((IDENTITY_MAX_SPEED_M_S * (t - float(anchor[0]))
                      + float(radius_km) * 1000.0) / dx_m)
            record["reach_parent_cells"] = round(reach, 3)
            record["last_fix_parent_ij"] = [round(float(anchor[1]), 3),
                                            round(float(anchor[2]), 3)]
        far = (reach is not None and math.hypot(
            lowest["ci"] - float(anchor[1]),
            lowest["cj"] - float(anchor[2])) > reach)
        if not far:
            if lowest["closed"]:
                record["decision"] = "lowest"
                return accept(lowest), record
            record["decision"] = "no-closed-low"
            return None, record
        # The lowest cell is beyond the storm's reach: the low the last fix
        # sits in -- reached by steepest descent from it, so a far low's
        # flank inside the reach is never mistaken for it -- is the storm
        # while it is still a closed low within reach.
        near = None
        cell = _descend(values, box, float(anchor[1]), float(anchor[2]))
        if cell is not None and math.hypot(
                cell[0] - float(anchor[1]),
                cell[1] - float(anchor[2])) <= reach:
            near = judge(*cell)
            record["nearest"] = near
        if near is not None and near["closed"] and not (
                lowest["closed"] and lowest["mslp_hpa"]
                <= near["mslp_hpa"] - IDENTITY_MIN_DEFICIT_HPA):
            record["decision"] = "continuity"
            return accept(near), record
        if lowest["closed"]:
            record["decision"] = "redeveloped"
            return accept(lowest), record
        record["decision"] = "no-closed-low"
        return None, record

    def locate(self, parent_state, nest_footprint, t,
               refinement=None) -> "VortexFix":
        """WHERE the storm is, with no opinion about moving anything.

        The first half of :meth:`desired_shift`, extracted so there is
        exactly ONE centre-finding implementation in this module.  The
        second half -- dead-band, cooldown, clamp, parent-edge clip -- is
        hysteresis, and hysteresis is about whether to MOVE; a consumer
        that only wants to know where the vortex is (the track writer,
        :mod:`woof.core.storm_track_writer`) must not have to burn a
        cooldown or leave a decision receipt to ask.

        STATELESS AND SILENT, deliberately.  It mutates none of the
        tracker's hysteresis state and appends no receipt, so a caller
        may consult it at whatever rhythm it likes without changing
        where the nest goes.  That is the property the track writer's
        finer emission interval rests on, and it is why a stash-backed
        field refuses that interval at admission: reading the UH window
        is not free of side effects at the RUNNER (which zeroes it), even
        though it is free of them here.

        ``found`` is ``None`` when nothing crossed the threshold; the
        evidence is still complete, because a hold that records nothing
        cannot be audited.
        """
        fp = NestFootprint.coerce(nest_footprint)
        cfg = self.config
        primary_failure = None
        try:
            planes = planes_for(parent_state, cfg, uh_slot=self.uh_slot)
        except TrackerRefusal as error:
            if cfg.fallback_level_hpa is None:
                raise
            primary_failure = str(error)
            cfg = replace(cfg, level_hpa=cfg.fallback_level_hpa,
                          threshold=cfg.fallback_deficit_m, pressure_center="centroid",
                          fallback_level_hpa=None)
            planes = planes_for(parent_state, cfg, uh_slot=self.uh_slot)
        plane = planes[0][1]
        box = fp.search_box(plane.shape, cfg.search_margin_cells)
        field_used = cfg.field
        threshold_used = float(cfg.threshold)
        relative = cfg.level_hpa is not None
        radius_cells = radius_in_cells(cfg.radius_km, fp.parent_dx_m)
        found, level_fixes, level_declined = centre_over_levels(
            planes, cfg, box, radius_cells)
        identity = None
        if (found is not None and cfg.field == "pressure"
                and cfg.level_hpa is None and cfg.pressure_center == "minimum"):
            # Rules 2-3 of the storm-identity block above.  A minimum that
            # fails them is a slope or another system, not a missing MSLP
            # field, so the 850 hPa fallback is not consulted: it exists for
            # a plane with no qualifying minimum at all.
            found, identity = self._storm_identity(
                plane, box, found, fp, float(t), cfg.radius_km)
        if (found is None and cfg.fallback_level_hpa is not None
                and identity is None):
            primary_failure = "no qualifying MSLP minimum"
            cfg = replace(cfg, level_hpa=cfg.fallback_level_hpa,
                          threshold=cfg.fallback_deficit_m, pressure_center="centroid",
                          fallback_level_hpa=None)
            planes = planes_for(parent_state, cfg, uh_slot=self.uh_slot)
            plane = planes[0][1]
            relative = True
            threshold_used = float(cfg.threshold)
            found, level_fixes, level_declined = centre_over_levels(planes, cfg, box, radius_cells)
        if found is None and cfg.field == "uh":
            # The handoff: no rotation signal yet; follow the echo.
            field_used = "reflectivity"
            threshold_used = float(cfg.fallback_threshold)
            plane = _plane_from_state(parent_state, field_used)
            found = locate_signal(plane, field_used, threshold_used, box,
                                  radius_cells=radius_cells)
        evidence: dict[str, object] = {
            "t": float(t),
            "field_requested": cfg.field,
            "field_used": field_used,
            "threshold_used": threshold_used,
            "signal": ("/".join(f"{v:g}" for v in cfg.level_hpa)
                       + " hPa geopotential height (m)"
                       if relative and field_used == "pressure"
                       else ("sea-level pressure (hPa)"
                             if field_used == "pressure" else field_used)),
            "search_box": [[int(box[0].start), int(box[0].stop)],
                           [int(box[1].start), int(box[1].stop)]],
            "footprint": {
                "grid_id": int(fp.grid_id),
                "i_parent_start": int(fp.i_parent_start),
                "j_parent_start": int(fp.j_parent_start),
                "center_parent_ij": [round(c, 3)
                                     for c in fp.center_parent_ij],
            },
        }
        if primary_failure is not None:
            evidence["primary_failure"] = primary_failure
            evidence["fallback_level_hpa"] = list(cfg.level_hpa)
        if identity is not None:
            evidence["storm_identity"] = identity
        if cfg.field == "attribute":
            evidence.update(attribute_metadata(cfg))
            evidence["signal"] = f"{cfg.attribute} {cfg.reduction} ({evidence['threshold_units']})"
            evidence["extremum_kind"] = extremum_kind(cfg, cfg.field)
            evidence["extremum_units"] = evidence["threshold_units"]
        # The box's own span, recorded whether or not it was a problem:
        # it is the one number that distinguishes "centred on the storm"
        # from "the storm is gone" on a held receipt.
        span = signal_span(plane, box)
        if span is not None:
            evidence["search_box_signal_span"] = round(span, 4)
        # ... and how far the centre it found settled from a box edge the
        # field continues past, which is the difference between a vortex
        # and the lowest corner of a slope.  Recorded on every receipt for
        # the same reason the span is: a held decision has to say how
        # close the call was, not only that one was made.
        if relative and field_used == "pressure" and found is not None:
            evidence["search_box_signal_closure"] = signal_closure(
                plane, box, threshold_used,
                centre=(found["ci"], found["cj"]), radius_cells=radius_cells)
        if level_fixes:
            evidence["levels"] = [
                {"level_hpa": f.level_hpa,
                 "centre_parent_ij": [round(f.ci, 3), round(f.cj, 3)],
                 "height_dam": round(f.height_dam, 2),
                 "cells": f.cells}
                for f in level_fixes]
            if len(level_fixes) > 1:
                spread = max(
                    math.hypot(a.ci - b.ci, a.cj - b.cj)
                    for a in level_fixes for b in level_fixes)
                evidence["level_spread_parent_cells"] = round(spread, 3)
                evidence["centre_rule"] = (
                    f"mean of {len(level_fixes)} per-level centres")
        if level_declined:
            evidence["levels_declined"] = level_declined
        blank = VortexFix(footprint=fp, evidence=evidence, found=None,
                          plane_shape=(int(plane.shape[-2]),
                                       int(plane.shape[-1])),
                          search_box=box, field_used=field_used,
                          threshold_used=threshold_used,
                          raw_shift=(0.0, 0.0), levels=level_fixes)
        if found is None:
            self.last_fix = blank
            return blank
        # EVERY TRACKED FIELD, not just the pressure core.  A rotation,
        # echo or attribute tracker whose signal has left the parent grid
        # is in exactly the position the pressure tracker is in -- the
        # centre it reports is the edge cell the search clipped to, and a
        # row written from it says the storm stopped moving when what
        # happened is that it left.  The end of the field that IS the
        # centre comes from extremum_kind, so the test asks about the same
        # extremum the receipt reports.
        planes_used = ([(None, plane)] if field_used != cfg.field else
                       [(level, p) for level, p in planes
                        if level is None or (level in levels_of(cfg)
                            and any(f.level_hpa == level for f in level_fixes))])
        from woof.core.track_boundary import boundary_reason
        reasons = [boundary_reason(p, box, (found["ci"], found["cj"]),
                                   radius_cells=radius_cells,
                                   extremum=extremum_kind(cfg, field_used),
                                   signal=evidence["signal"])
                   for _level, p in planes_used]
        # All steering surfaces must be boundary limited. A surviving
        # interior circulation at another level keeps the track alive.
        if reasons and all(reason is not None for reason in reasons):
            evidence["track_end_reason"] = reasons[0]
        refined_on = None
        refined_cell_ij = None
        if cfg.refine_grid_id is not None:
            if refinement is None:
                evidence["refinement"] = {
                    "grid_id": int(cfg.refine_grid_id),
                    "applied": False,
                    "declined": ("the runner supplied no refinement source; "
                                 "the grid has not started, is not a "
                                 "descendant of the mover, or carries no "
                                 "state yet"),
                }
            elif int(refinement.grid_id) != int(cfg.refine_grid_id):
                raise ValueError(
                    f"follow refine_grid_id = {cfg.refine_grid_id} but the "
                    f"runner supplied grid {refinement.grid_id}; a tracker "
                    "refines on the grid its config names, never on "
                    "whichever one it is handed")
            else:
                found = self._refine(found, refinement, evidence, cfg)
                if evidence["refinement"].get("applied"):
                    refined_on = int(refinement.grid_id)
                    refined_cell_ij = tuple(found["refine_cell_ij"])
                    # The refine grid's per-level answer supersedes the
                    # parent's: it is the grid that resolves the vortex,
                    # and a track row must not mix the two.
                    if found.get("refine_levels"):
                        level_fixes = found["refine_levels"]
        center_i, center_j = fp.center_parent_ij
        raw_di = found["ci"] - center_i
        raw_dj = found["cj"] - center_j
        evidence.update({
            # "cells_above_threshold" reads literally for every maximum
            # signal and means "cells past the threshold" for pressure,
            # where past is BELOW.  The key is kept rather than split so
            # one receipt shape covers every field; "extremum_kind" says
            # which side of the threshold qualified.
            "cells_above_threshold": found["cells"],
            # The fixed point's own report.  "converged": false is a
            # finding -- the centre is then the last iterate rather than
            # a settled point -- and a competing centre is the frame
            # before a reformation, which is exactly when an operator
            # wants to be looking.
            **{k: found[k] for k in ("iterations", "converged",
                                     "competing_centre") if k in found},
            "extremum_kind": extremum_kind(cfg, field_used),
            "extremum_units": ("m" if (relative and field_used == "pressure")
                               else ("hPa" if field_used == "pressure"
                                     else "field")),
            "max_value": round(signal_extremum(found, field_used), 3),
            "centroid_parent_ij": [round(found["ci"], 3),
                                   round(found["cj"], 3)],
            "raw_shift_parent_cells": [round(raw_di, 3), round(raw_dj, 3)],
        })
        from dataclasses import replace as _replace
        if cfg.field == "attribute":
            evidence["extremum_units"] = evidence["threshold_units"]
            evidence["max_value"] = signal_extremum(found, field_used)
        self.last_fix = _replace(blank, found=found,
                                 raw_shift=(raw_di, raw_dj),
                                 refined_on=refined_on,
                                 refined_cell_ij=refined_cell_ij,
                                 levels=level_fixes)
        return self.last_fix

    def desired_shift(self, parent_state, nest_footprint,
                      t: float, refinement=None) -> tuple[int, int] | None:
        """The plan-provider callable: whole-parent-cell shift or hold.

        Evidence is gathered before any suppression fires, so a
        suppressed receipt still records WHERE the storm was seen -- a
        hold decision without evidence cannot be audited.

        ``refinement`` is the optional second stage
        (:class:`RefinementSource`): stage one locates the storm on the
        parent, which is the only field that sees outside the nest, and
        stage two re-locates it on a descendant that actually resolves
        it.  ``None`` -- and a config without ``refine_grid_id`` -- is
        the single-stage tracker, unchanged.
        """
        if self._parked is not None:
            # Rule 1/4: the track has ended, and the nest stays where it is
            # for the rest of the run.  Nothing is located: the answer
            # cannot move anything.
            self._receipt({"decision": "parked", "t": float(t),
                           **{f"parked_{k}": v
                              for k, v in self._parked.items()}})
            return None
        fix = self.locate(parent_state, nest_footprint, t,
                          refinement=refinement)
        fp, cfg = fix.footprint, self.config
        evidence = fix.evidence
        ended = evidence.get("track_end_reason")
        if ended:
            # Rule 1: the fix the track writer ends on is never followed.
            self._parked = {"t": float(t), "reason": str(ended)}
            self._receipt({"decision": "parked:track-ended", **evidence})
            return None
        identity_mode = (cfg.field == "pressure" and cfg.level_hpa is None
                         and cfg.pressure_center == "minimum")
        if fix.found is None:
            if identity_mode:
                if self._lost_since is None:
                    self._lost_since = float(t)
                lost_for = float(t) - self._lost_since
                evidence["lost_for_seconds"] = lost_for
                if lost_for >= IDENTITY_LOST_AFTER_S:
                    # Rule 4.  The reason rides the fix the track writer
                    # renders next, so the file ends where the nest parks.
                    reason = (
                        f"no MSLP minimum passed the storm-identity test "
                        f"(a closed low {IDENTITY_MIN_DEFICIT_HPA:g} hPa "
                        f"below its {IDENTITY_RING_KM:g} km ring, within "
                        f"reach of the last fix) for {lost_for:.0f} s")
                    self._parked = {"t": float(t), "reason": reason}
                    evidence["track_end_reason"] = reason
                    self._receipt({"decision": "parked:storm-lost",
                                   **evidence})
                    return None
            self._receipt({"decision": "no-signal", **evidence})
            return None
        self._lost_since = None
        motion_limit = int(cfg.max_shift_cells)
        if cfg.field == "pressure" and cfg.level_hpa is None:
            ci, cj = fix.center_parent_ij
            previous = self._motion_fix
            self._motion_fix = [float(t), ci, cj]
            dt = None if previous is None else float(t) - previous[0]
            interval = max(float(self.move_interval_seconds or dt or 0.),
                           float(cfg.cooldown_seconds))
            speed_cells = (None if dt is None or dt <= 0 else
                           math.hypot(ci - previous[1], cj - previous[2]) / dt)
            speed = (None if speed_cells is None or fp.parent_dx_m is None
                     else speed_cells * fp.parent_dx_m)
            motion_cells = (0 if speed_cells is None else
                            math.ceil(speed_cells * interval))
            # Initial placement and a previous bound may leave a backlog.
            # Allow recentering as well as the next interval's translation.
            catchup = math.ceil(max(map(abs, fix.raw_shift)))
            # The runner bounds the actual vector by halo/parent clearance
            # and min_overlap_fraction. A fixed six-cell ceiling here made
            # a cyclone leave the child even when safe overlap was available.
            motion_limit = max(motion_cells, catchup)
            evidence.update({"storm_speed_m_s": speed,
                             "move_interval_seconds": interval,
                             "motion_cells": motion_cells,
                             "catchup_cells": catchup,
                             "motion_cap_parent_cells": motion_limit})
        di = _round_cells(fix.raw_shift[0])
        dj = _round_cells(fix.raw_shift[1])
        if max(abs(di), abs(dj)) < int(cfg.min_shift_cells):
            self._receipt({"decision": "suppressed:dead-band",
                           "shift_parent_cells": [di, dj], **evidence})
            return None
        # Cooldown anchor: the last EXECUTED move once the runner reports
        # one through notify_move_executed; until then, the last proposal
        # (the pre-hook fallback, so an old runner still gets hysteresis).
        anchor = (self._last_move_t if self._last_move_t is not None
                  else self._last_proposal_t)
        if (anchor is not None
                and float(t) - anchor < float(cfg.cooldown_seconds)):
            self._receipt({
                "decision": "suppressed:cooldown",
                "shift_parent_cells": [di, dj],
                "cooldown_anchor": ("executed-move"
                                    if self._last_move_t is not None
                                    else "proposal"),
                "cooldown_remaining_s": round(
                    float(cfg.cooldown_seconds) - (float(t) - anchor), 3),
                **evidence})
            return None
        limit = motion_limit
        clamped = max(abs(di), abs(dj)) > limit
        di = max(-limit, min(limit, di))
        dj = max(-limit, min(limit, dj))
        di, dj, clipped = self._clip_to_parent(fp, fix.plane_shape, di, dj)
        if di == 0 and dj == 0:
            # Clamp/clip can collapse a real proposal to the null move
            # (storm leaving through a parent edge the nest already
            # hugs).  The null move is never proposed.
            self._receipt({"decision": "suppressed:at-parent-edge",
                           "clamped": bool(clamped),
                           "clipped_to_parent": clipped, **evidence})
            return None
        self._last_proposal_t = float(t)
        self._receipt({"decision": "proposed",
                       "shift_parent_cells": [di, dj],
                       "clamped": bool(clamped),
                       "clipped_to_parent": clipped, **evidence})
        return (di, dj)

    __call__ = desired_shift

    def notify_move_executed(self, t: float,
                             executed_shift: tuple[int, int]) -> None:
        """Optional runner hook: a proposal was actually EXECUTED.

        The leg-2 runner probes for this duck-typed hook and calls it with
        the clamped shift it really applied, after the relocation
        succeeds.  From the first call on, the cooldown counts from
        executed moves instead of proposals -- a proposal the runner
        refused (leg-1 admissibility, a busy cycle) no longer burns the
        cooldown, so the tracker may re-propose at the next cadence.  A
        runner that never calls it keeps the proposal-burn fallback.
        """
        di, dj = (int(executed_shift[0]), int(executed_shift[1]))
        self._last_move_t = float(t)
        self._receipt({"decision": "move-executed", "t": float(t),
                       "executed_shift_parent_cells": [di, dj]})

    @staticmethod
    def _clip_to_parent(fp: NestFootprint, plane_shape: tuple[int, int],
                        di: int, dj: int) -> tuple[int, int, bool]:
        """Keep the proposed footprint PARENT_EDGE_KEEPOUT_CELLS clear of
        the parent's edge; the leg-1 admissibility check still has the
        final word."""
        ny, nx = int(plane_shape[-2]), int(plane_shape[-1])
        keep = PARENT_EDGE_KEEPOUT_CELLS
        i_lo = int(fp.i_parent_start) - 1
        j_lo = int(fp.j_parent_start) - 1
        di_min = keep - i_lo
        di_max = int(math.floor(nx - 1 - keep - fp.span_parent_i)) - i_lo
        dj_min = keep - j_lo
        dj_max = int(math.floor(ny - 1 - keep - fp.span_parent_j)) - j_lo
        # A footprint already inside the keepout must not be dragged; only
        # shrink the proposal toward zero, never push past it.
        out_i = _clip_toward_zero(di, di_min, di_max)
        out_j = _clip_toward_zero(dj, dj_min, dj_max)
        return out_i, out_j, (out_i != di or out_j != dj)


def _clip_toward_zero(step: int, lo: int, hi: int) -> int:
    """Clip ``step`` into [lo, hi] without ever crossing zero: a keepout
    violation shrinks a proposal, it never manufactures a move."""
    lo = min(lo, 0)
    hi = max(hi, 0)
    return max(lo, min(hi, step))


def make_plan_provider(experiment) -> StormTracker | None:
    """The runner's one-line hookup: a tracker from the experiment config.

    Returns ``None`` when the experiment carries no ``[relocation.follow]``
    block -- relocation stays available as a manual/API mechanism exactly
    as leg 1 shipped it.
    """
    relocation = getattr(experiment, "relocation", None)
    follow = getattr(relocation, "follow", None)
    if follow is None:
        return None
    if not getattr(relocation, "enabled", False):
        raise TrackerRefusal(
            "a follow block on a disabled [relocation] cannot exist; the "
            "config loader refuses it, so this experiment object was built "
            "by hand inconsistently")
    return StormTracker(follow)


__all__ = [
    "LevelFix", "LEVEL_HPA_MAX", "LEVEL_HPA_MIN", "MAX_TRACKED_LEVELS",
    "VortexFix", "centre_over_levels", "level_height_m_from_state",
    "level_heights_m_from_state", "all_levels_of", "report_levels_of",
    "levels_of", "planes_for", "signal_span", "signal_closure",
    "FOLLOW_CONTRACT", "FOLLOW_KEYS", "FollowConfig", "NestFootprint",
    "CENTROID_MAX_ITERATIONS", "COMPETING_CENTRE_FRACTION",
    "DEFAULT_LEVEL_HPA", "SEA_LEVEL_HPA",
    "PARENT_EDGE_KEEPOUT_CELLS", "REFLECTIVITY_SLOT", "StormTracker",
    "TRACKED_FIELDS", "TRACKER_STATE_KEYS", "TrackerRefusal", "UH_SLOT",
    "DEFAULT_CENTROID_RADIUS_KM", "RADIUS_KM_MAX", "RADIUS_KM_MIN",
    "is_minimum_signal", "locate_signal", "normalise_pressure_surface",
    "extremum_kind", "pressure_surface_json", "radius_in_cells",
    "signal_extremum",
    "build_follow_config",
    "make_plan_provider", "signal_plane", "weighted_centroid",
]
