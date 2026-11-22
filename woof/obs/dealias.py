"""Region-based velocity dealiasing that abstains instead of guessing.

Radial velocity from a pulsed Doppler radar is a phase measurement, so it is
only known modulo the Nyquist interval ``2 * Vn``.  A true ``+40 m/s`` on a
cut whose Nyquist is ``25.51 m/s`` is reported as ``-11.02 m/s``: finite,
plausible, smooth, and wrong by ``51.02 m/s``.  :mod:`woof.obs.superob`
answers that today by masking -- it drops every gate above
``nyquist_reject_fraction`` of Nyquist and counts the loss -- which is safe
and which, on a cut whose Nyquist is 25.51, caps the assimilable wind at
20.4 m/s.  A mesocyclone's rotational couplet is a pair of strong,
opposite-signed low-level velocities; the mask removes it, and removes more
of it the stronger the storm gets.

This module recovers those gates by unfolding them, and it is built around
one rule: **where the unfolding is ambiguous the gate is rejected, not
guessed.**  A confidently wrong dealiased velocity is worse than a missing
one, because a missing one is a mask the filter never sees and a wrong one
enters the filter with a 1-2 m/s observation error attached to a 51 m/s
mistake.

Method
------
Region-based unfolding, the standard approach, with one volume-scale step
in front of it:

0. **Profile.**  Before any sweep is unfolded, a single ``u(z), v(z)`` is
   built from the whole volume: every range band of every velocity cut is
   fitted, converted to a wind, and pooled onto a height grid where the
   median is taken.  A volume reaches most heights several times over -- 4 km
   sits at 60 km range on a 4-degree cut and at 200 km on a 1-degree one --
   and a layer is believed only when cuts at two or more *different*
   elevations reached it and it is continuous with the layers around it.
   This is the only cross-check available without an external model field,
   and it is what stops a sparse far-range band from anchoring thousands of
   gates to a wind no other sweep saw.
1. **Segment.**  Two range- or azimuth-adjacent gates join the same region
   when both are finite and differ by at most
   ``region_join_fraction * Vn``.  A fold between neighbours produces a jump
   of very nearly ``2 * Vn``, so at the 0.5 default a fold can never occur
   *inside* a region unless the true gate-to-gate shear exceeds ``1.5 * Vn``
   (38 m/s across one 250 m gate here), which no atmosphere produces.  That
   bound is the assumption this stage rests on and it is stated so it can be
   argued with.
2. **Vote.**  Every adjacent gate pair that did *not* join contributes an
   integer vote ``f = round((v_lo - v_hi) / (2 * Vn))`` to the edge between
   the two regions it spans, meaning "region *hi* sits ``f`` Nyquist
   intervals above region *lo*".
3. **Anchor.**  An environmental reference -- a fold-aware VAD fitted to the
   sweep itself (Browning & Wexler 1968), or an externally supplied
   background wind profile -- pins whole regions to an absolute fold count.
   Only regions that look *environmental* after unfolding may anchor: the
   median residual against the reference must be small, which a rotational
   couplet's region will fail, so a couplet is never forced onto the
   background wind.
4. **Grow.**  From each anchor, unresolved regions are resolved across their
   strongest confident edge first.  An edge is confident only when it has
   enough boundary pairs, a large enough winning-vote majority, and a mean
   jump close enough to an integer.
5. **Verify, then abstain.**  Every confident edge between two resolved
   regions must agree with the assignment; regions on a violated edge are
   rejected.  Regions never anchored and never reached are rejected.  Gates
   whose unfolded speed leaves the physical bound are rejected, and so are
   gates that depart from the reference by more than a couplet could -- a
   fold error displaces a gate by a whole Nyquist interval, 51 m/s on the
   tilts that matter here, while the physical departures this must not touch
   are smaller.

Every velocity gate therefore ends in exactly one of three states --
:data:`STATE_UNFOLDED`, :data:`STATE_UNCHANGED`, :data:`STATE_REJECTED` --
and all three are counted, per sweep and per volume, with a reason recorded
for every rejection.

Relationship to BowEcho
-----------------------
WOOF-ArWen's Rust radar stack (BowEcho, ``crates/render2d``) already carries four
dealiasing engines, and steps 1-2 here are deliberately the same algorithm
and the same constants as its shared ``region_core`` -- ``REGION_JOIN_FRAC``
0.5, ``REGION_MAX_FOLD`` 5, the same vote convention, the same
strongest-boundary-first resolution order, the same range-band harmonic
reference geometry (``REFERENCE_BAND_GATES`` 16, 48 samples, 5 of 12
azimuth sectors, 12 m/s trim).

Steps 3-5 are where this diverges, and the divergence is the point.
BowEcho's v1 engine anchors *the largest region in each connected group to
fold 0* and always assigns a fold to every region: correct for a display,
where an unresolved region is a visual artifact a human discounts, and
wrong for assimilation, where it is a confident number the filter believes.
This module refuses to assign a fold it cannot justify.

The ingest boundary is why this is Python at all.  ``rw-nexrad`` -- the
mandated ingest crate -- is a binary-only crate with no ``lib.rs`` and no
Python binding, and BowEcho's engines consume ``radar_core::RadarVolume``
while ``rw-nexrad`` emits flat ``<f4`` planes; the two Rust trees share no
crate today.  :func:`dealias_sweep` is therefore written against exactly
what crosses that seam -- a ``(radial, gate)`` plane, an azimuth vector, one
Nyquist scalar -- so that when the plumbing exists it is replaced by a call,
not by a rewrite.

Engines
-------
That call now exists.  :mod:`woof.obs.dealias_region` drives WOOF-ArWen's Rust
``region-global-dealias`` crate -- a port of Py-ART's
``dealias_region_based`` -- through a C ABI, and
:attr:`DealiasParams.engine` selects between them:

``"region-global"`` (the default)
    The Rust crate.  It unfolds the whole region network jointly rather
    than growing outward from anchors, carries no environmental reference,
    and assigns a fold to every region it resolved instead of abstaining.
    Its refinement pass runs by default with it.
``"vad-region"``
    Everything described above: VAD-anchored, and it abstains.  Selectable
    and supported; it is what every run before 2026-08-12 read.

The two engines are *different algorithms* and their gate decisions differ
by design; the shared contract is the three-state result, the one physical
bound of :func:`speed_bounded`, and their accounting -- not the decisions.
``engine`` is a recorded parameter -- it travels into provenance with the
velocities it produced -- because "which solver said this wind was 40 m/s"
is not an implementation detail to anyone reading an assimilated field
back.

The default changed with measurement in hand, not with the newest engine:
on one real 14-cut volume the abstainer rejected both gates of a 98 m/s
couplet as unresolved and the region-global engine recovered them, at
0.3 percent of the dealias stage's wall clock.  Because that decision
moves which velocities reach the filter on every run that asks for
dealiasing, the library the default now needs is a BLOCKING check in
``woof doctor`` rather than an optional one.

References
----------
Browning, K. A. and R. Wexler, 1968: The determination of kinematic
properties of a wind field using Doppler radar. *J. Appl. Meteor.*, **7**,
105-113.

Bergen, W. R. and S. C. Albers, 1988: Two- and three-dimensional
de-aliasing of Doppler radar velocities. *J. Atmos. Oceanic Technol.*, **5**,
305-319.

Jing, Z. and G. Wiener, 1993: Two-dimensional dealiasing of Doppler
velocities. *J. Atmos. Oceanic Technol.*, **10**, 798-808.

James, C. N. and R. A. Houze, 2001: A real-time four-dimensional Doppler
dealiasing scheme. *J. Atmos. Oceanic Technol.*, **18**, 1674-1683.

Helmus, J. J. and S. M. Collis, 2016: The Python ARM Radar Toolkit
(Py-ART). *J. Open Research Software*, **4**(1), e25.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from numbers import Real

import numpy as np

from woof import perf_timing
from woof.obs import coarse_cost
from woof.obs.dealias_region import ENGINE_REGION_GLOBAL
from woof.obs.geometry import beam_geometry

#: The VAD-referenced, abstaining engine implemented in this module.
ENGINE_VAD_REGION = "vad-region"

#: Every engine :func:`dealias_sweep` can dispatch to, in the spelling
#: ``DealiasParams.engine`` takes and provenance records.
ENGINES = (ENGINE_VAD_REGION, ENGINE_REGION_GLOBAL)

#: A velocity gate ended in exactly one of these.  There is no fourth.
STATE_REJECTED = 0
STATE_UNCHANGED = 1
STATE_UNFOLDED = 2

#: Why a rejected gate was rejected.  ``REASON_NONE`` is carried by every
#: gate that was not rejected, so the reason plane is always readable.
REASON_NONE = 0
REASON_NONFINITE = 1
REASON_NO_NYQUIST = 2
REASON_UNRESOLVED = 3
REASON_CONFLICT = 4
REASON_FOLD_RANGE = 5
REASON_SPEED = 6
REASON_DISABLED = 7
REASON_REFERENCE_DEPARTURE = 8

REASON_NAMES = {
    REASON_NONE: "none",
    REASON_NONFINITE: "nonfinite",
    REASON_NO_NYQUIST: "no_nyquist",
    REASON_UNRESOLVED: "unresolved",
    REASON_CONFLICT: "conflict",
    REASON_FOLD_RANGE: "fold_out_of_range",
    REASON_SPEED: "speed_out_of_range",
    REASON_DISABLED: "disabled",
    REASON_REFERENCE_DEPARTURE: "reference_departure",
}


#: What to tell a caller who asked for dealiasing on an install that cannot
#: do it.  Spelled out here, once, so both entry points say the same thing.
#:
#: It no longer offers the dealias extra.  scipy is a runtime dependency
#: now, so that extra is an empty compatibility alias: installing it
#: installs nothing and the next attempt fails identically.  A remedy that
#: does not remedy is worse than none -- reaching this message means the
#: environment lost a declared dependency, which is a different problem
#: with a different fix.
SCIPY_REMEDY = (
    "the vad-region dealiasing engine needs scipy "
    "(scipy.sparse.csgraph.connected_components labels the gate regions). "
    "scipy is a required dependency of woof, so this environment is "
    "incomplete: repair it with `pip install --upgrade --force-reinstall "
    "woof`, or install the one package with `pip install 'scipy>=1.11'`")


def scipy_available() -> bool:
    """Whether the region labeller this module needs can be imported.

    Checked at the front door rather than left to the function-local import
    inside :func:`dealias_sweep`, because the two failures are not the same
    failure.  Reaching that import means a run has already fetched a volume,
    decoded it and started gridding; refusing here means the operator hears
    "install scipy" in the second before the run starts, instead of a
    traceback from inside a loop an hour later.
    """

    from importlib.util import find_spec

    try:
        return (find_spec("scipy.sparse") is not None
                and find_spec("scipy.sparse.csgraph") is not None)
    except (ImportError, ValueError):     # pragma: no cover - broken install
        return False


def engine_unavailable_reason(engine: str) -> str | None:
    """Why ``engine`` cannot run on this install, or None when it can.

    The two engines have different prerequisites -- one needs scipy, the
    other needs a built shared library -- and a front door that checked
    only scipy would clear a run that cannot dealias and let it discover
    that an hour in.  One function, asked once, per the reason
    :func:`scipy_available` exists at all.
    """

    if engine == ENGINE_REGION_GLOBAL:
        from woof.obs.dealias_region import (  # noqa: PLC0415
            region_bridge_remedy, region_engine_available)

        if region_engine_available():
            return None
        return ("the region-global dealiasing engine's shared library is "
                "not on this box\n" + region_bridge_remedy())
    if engine == ENGINE_VAD_REGION:
        return None if scipy_available() else SCIPY_REMEDY
    return (f"unknown dealiasing engine {engine!r}; known engines are "
            f"{list(ENGINES)}")


#: The order a default (unnamed) dealiasing request resolves down.
#:
#: Dealiasing is what a bare radar ingest does now, so the default cannot be
#: a hard prerequisite: an install without the region-global shared library
#: would otherwise have a dead door on every ingest.  The chain is stated
#: rather than implicit -- the shipped engine first, the scipy one next,
#: masking-only last -- and which rung ran is recorded in the file's own
#: provenance, so nothing about the descent is silent.  An engine the
#: OPERATOR named is never resolved away; it is refused by name, because a
#: run that asked for one solver and got another is the A/B break the
#: engine selector exists to prevent.
DEFAULT_ENGINE_CHAIN = (ENGINE_REGION_GLOBAL, ENGINE_VAD_REGION)

#: One warning per resolution per process, not one per constructed object.
_RESOLUTION_WARNED: set = set()


def first_available_engine(unavailable_reason=None) -> str | None:
    """The first engine on :data:`DEFAULT_ENGINE_CHAIN` this install can run.

    ``None`` means neither can, which is masking-only: the behaviour this
    stage has always had, reached by descent rather than by omission.
    """

    if unavailable_reason is None:
        unavailable_reason = engine_unavailable_reason
    for engine in DEFAULT_ENGINE_CHAIN:
        if unavailable_reason(engine) is None:
            return engine
    return None


def resolve_default_dealias(params, unavailable_reason=None):
    """Resolve an UNNAMED dealiasing request against this install.

    ``params`` is a ``DealiasParams`` or None.  None stays None.  A request
    naming an engine other than the chain head is honoured exactly, so a
    named engine that cannot run still fails where it is asked for.  A
    request carrying the chain head is resolved down the chain when that
    engine is not staged here: to the next engine, or to None (masking
    only) when neither can run.  Warns once per process per outcome and
    names the basis.

    ONE function for every door: the tools front doors, the ``woof obs``
    door and the parameter object itself all ask it, so no two of them can
    believe different things about the same install.
    """

    if params is None:
        return None
    engine = str(getattr(params, "engine", DEFAULT_ENGINE_CHAIN[0]))
    if engine != DEFAULT_ENGINE_CHAIN[0]:
        return params
    if unavailable_reason is None:
        unavailable_reason = engine_unavailable_reason
    reason = unavailable_reason(engine)
    if reason is None:
        return params
    from woof.explain import warn                    # noqa: PLC0415

    fallback = first_available_engine(unavailable_reason)
    key = (engine, fallback)
    first_time = key not in _RESOLUTION_WARNED
    _RESOLUTION_WARNED.add(key)
    if fallback is None:
        if first_time:
            warn("velocity dealiasing is on by default and no engine on "
                 f"this install can run it, so this volume is gridded with "
                 f"masking only; the file's own dealiasing statement says "
                 f"so",
                 why=reason)
        return None
    if first_time:
        warn(f"velocity dealiasing is on by default and the {engine} engine "
             f"is not staged here, so the {fallback} engine ran instead; the "
             f"file's own dealiasing statement and superob_params record "
             f"which one",
             why=reason)
    import dataclasses as _dataclasses                # noqa: PLC0415

    # ``refinement=None`` and not the resolved bool: the refinement default
    # belongs to the engine, and the engine just changed.  A caller who
    # NAMED refinement never reaches here, because naming it names
    # region-global and a named engine is refused rather than resolved.
    return _dataclasses.replace(params, engine=fallback, refinement=None)


def dealias_params_from_args(args, params_class=None,
                             unavailable_reason=None):
    """``DealiasParams`` for these flags, or None -- refusing at the door.

    Shared by every front door so the refusal, the message and the
    parameter object are one implementation.  It lives here, beside the
    engines, rather than in a tool, because ``woof`` must not import
    ``tools``; the tools re-export it under their own name.

    ``--dealias-refinement`` without dealiasing is refused rather than
    ignored: a run that asked for a treatment and silently did not get it
    is exactly the failure the A/B discipline exists to prevent.

    ``--dealias-refinement`` unset is None, not False, and it stays None
    all the way into ``DealiasParams``: the refinement default belongs to
    the engine (on for ``region-global``, meaningless for ``vad-region``),
    and a front door that resolved it to a bool here would have to know
    that table too.  Asked for explicitly beside the engine that has no
    such pass, it is refused by name at the door rather than as a
    traceback from the parameter object.

    An engine the operator NAMED and that cannot run is a hard refusal.  An
    unnamed default resolves down :data:`DEFAULT_ENGINE_CHAIN` instead,
    because turning a missing optional library into a dead door for every
    bare ingest would trade one reachability defect for another.
    """

    if params_class is None:
        params_class = DealiasParams
    if unavailable_reason is None:
        unavailable_reason = engine_unavailable_reason

    named = getattr(args, "dealias_engine", None)
    refinement = getattr(args, "dealias_refinement", None)
    if not getattr(args, "dealias", True):
        if refinement is not None:
            raise SystemExit(
                "--dealias-refinement/--no-dealias-refinement refines a "
                "dealiased field; drop --no-dealias, or drop it")
        return None
    engine = DEFAULT_ENGINE_CHAIN[0] if named is None else str(named)
    if refinement and engine != ENGINE_REGION_GLOBAL:
        raise SystemExit(
            f"--dealias-refinement with --dealias-engine {engine}: that "
            f"engine has no refinement pass; only {ENGINE_REGION_GLOBAL} "
            "does. A switch that is accepted and then ignored is how a run "
            "gets reported as having had a treatment it never got")
    reason = unavailable_reason(engine)
    if named is not None or refinement:
        # A NAMED engine is honoured exactly, or refused by name.  It is
        # never resolved away: a run that asked for one solver and silently
        # got another grades itself against a differently-built truth.
        if reason is not None:
            raise SystemExit(f"--dealias-engine {engine}: {reason}")
        return params_class(engine=engine, refinement=refinement)
    return resolve_default_dealias(
        params_class(engine=engine, refinement=refinement),
        unavailable_reason=unavailable_reason)


class DealiasParamsError(ValueError):
    """A dealiasing parameter that cannot mean what the pass needs it to mean.

    Never a warning, for the same reason
    :class:`woof.obs.superob.SuperobParamsError` is never a warning: each of
    these values decides whether a gate is unfolded, and a wrong one produces
    velocities that are finite, plausible and off by a Nyquist interval.
    """


_FRACTION_FIELDS = (
    "region_join_fraction",
    "edge_agreement",
    "edge_max_offset",
    "anchor_max_offset",
    "reference_min_inlier_fraction",
)

_POSITIVE_FIELDS = (
    "profile_max_layer_step_ms",
    "reference_max_profile_departure_ms",
    "profile_layer_m",
    "reference_trim_ms",
    "reference_max_residual_ms",
    "reference_max_speed_ms",
    "reference_max_band_step_ms",
    "anchor_max_residual_ms",
    "max_speed_ms",
    "max_reference_departure_ms",
)

_POSITIVE_INT_FIELDS = (
    "max_fold",
    "edge_min_pairs",
    "reference_band_gates",
    "reference_min_samples",
    "reference_min_sectors",
    "anchor_min_gates",
    "reference_max_iterations",
    "reference_band_window",
    "profile_min_samples",
    "profile_min_elevations",
    "profile_layer_window",
)


@dataclass(frozen=True)
class DealiasParams:
    """Every dealiasing tunable, in one hashable place -- it goes into
    provenance beside the velocities it produced."""

    #: Two adjacent gates join one region when they differ by at most this
    #: fraction of Nyquist.  BowEcho ``REGION_JOIN_FRAC``.  Raising it toward
    #: 1.0 welds genuinely different branches together; lowering it shatters
    #: smooth gradients into regions too small to anchor.
    region_join_fraction: float = 0.5
    #: Hard cap on the integer fold applied to a region.  BowEcho
    #: ``REGION_MAX_FOLD``.  A region resolving past this is not a fold, it
    #: is a failure of the graph, and it is rejected.
    max_fold: int = 5
    #: Boundary pairs an inter-region edge needs before it may resolve a
    #: fold.  BowEcho trusts a single pair because a wrong edge there costs a
    #: pixel; here it costs an observation, so the default is stricter.
    edge_min_pairs: int = 4
    #: Fraction of an edge's votes that must agree on the winning fold.
    edge_agreement: float = 0.75
    #: How far the edge's mean jump may sit from its winning integer, in
    #: units of the Nyquist interval.  A true fold lands within a few
    #: hundredths of an integer; 0.25 is the point past which "fold" and "no
    #: fold" stop being distinguishable and abstention is the accurate answer.
    edge_max_offset: float = 0.25
    #: Gates per range band in the harmonic reference fit.  BowEcho
    #: ``REFERENCE_BAND_GATES``.
    reference_band_gates: int = 16
    #: Samples a band needs before its fit is believed.  BowEcho
    #: ``FIT_MIN_SAMPLES``.
    reference_min_samples: int = 48
    #: Distinct 30-degree azimuth sectors (of 12) a band's samples must span.
    #: BowEcho ``FIT_MIN_SECTORS``.  A harmonic fitted inside one sector is
    #: an extrapolation wearing a fit's clothes.
    reference_min_sectors: int = 5
    #: Outlier trim for the refit pass, m/s.  BowEcho ``FIT_TRIM_MPS``.
    reference_trim_ms: float = 12.0
    #: A band whose inlier residual RMS exceeds this has no usable reference.
    reference_max_residual_ms: float = 8.0
    #: Largest first-harmonic amplitude a band may fit and still be believed,
    #: m/s.  This is an environmental wind speed, and no environmental wind at
    #: the heights a weather radar samples reaches 60 m/s.
    #:
    #: It is here because a wrong harmonic fits its own fold assignment
    #: tightly and therefore passes a residual test: measured on the real
    #: case, bands converged on 57, 67, 71 and 74 m/s winds with residuals
    #: inside the 8 m/s gate, and regions anchored to them came out at 60-86
    #: m/s.  A residual measures self-consistency; only a physical bound
    #: measures whether the answer is a wind.
    reference_max_speed_ms: float = 60.0
    #: Largest step between the fitted winds of nearby range bands, m/s.  The
    #: environmental wind is continuous in height, so a band disagreeing this
    #: much with its neighbours is the odd one out rather than a discovery --
    #: and an isolated wrong fit is exactly what this catches, because a
    #: wrong fit has no reason to agree with the bands either side of it.
    reference_max_band_step_ms: float = 15.0
    #: Bands either side that the continuity check compares a band against.
    reference_band_window: int = 3
    #: How far a band's fitted wind may depart from the volume-wide profile
    #: at the same height before the profile is used instead, m/s.
    #:
    #: A volume samples most heights several times over -- 23 sweeps here,
    #: a given altitude reached at short range on a steep cut and at long
    #: range on a shallow one -- so the volume knows its own wind profile far
    #: better than any single range band does.  Far-range bands are where
    #: single-band fits go wrong: the echo is sparse, azimuthal coverage is
    #: patchy, and a fit can lock onto a branch that no other sweep sees at
    #: that height.  Measured on the real case, this is what produced 5,264
    #: gates unfolded past 45 m/s at 190-200 km on one 1.86-degree cut.
    reference_max_profile_departure_ms: float = 15.0
    #: Height resolution of the volume-wide profile, m.
    profile_layer_m: float = 500.0
    #: Band fits a profile layer needs before that layer is believed.
    profile_min_samples: int = 3
    #: Distinct elevation cuts a profile layer needs.  Three band fits from
    #: one sweep are one measurement read three times, not three
    #: measurements: they share the geometry, the echo and any error in it.
    #: Redundancy only counts when it comes from cuts that reached the layer
    #: by different paths.
    profile_min_elevations: int = 2
    #: Largest step between neighbouring profile layers, m/s.  The
    #: environmental wind is continuous in height; measured on the real case
    #: an unfiltered profile went from 14 m/s at 7.75 km to 48.5 m/s at
    #: 9.25 km with the direction reversed, which is a sampling failure in
    #: the thin upper layers wearing the shape of a jet.
    profile_max_layer_step_ms: float = 20.0
    #: Layers either side that the profile continuity check compares
    #: a layer against.
    profile_layer_window: int = 2
    #: Fraction of a band's samples that must survive the trim.
    reference_min_inlier_fraction: float = 0.5
    #: Iterations of the fold-aware fit.
    reference_max_iterations: int = 12
    #: Gates with a valid reference a region needs before it may anchor.
    #: The estimator is a median, so this is "how many samples before a
    #: median means anything"; below about eight it stops being a statistic.
    #: Measured on a real 23-sweep volume, raising it to 64 costs 58k
    #: additional abstentions and recovers nothing, because the regions it
    #: excludes are 2-7 gate speckle whose fold state genuinely cannot be
    #: established -- the count of *unfolded* gates does not move at all.
    anchor_min_gates: int = 8
    #: How far an anchor's median fold estimate may sit from its integer.
    anchor_max_offset: float = 0.25
    #: An anchor's median residual against the reference, m/s.  This is the
    #: gate that keeps a rotational couplet from being anchored: a couplet's
    #: region departs from the environmental wind by design, fails this, and
    #: must be resolved by continuity from a region that passed it.
    anchor_max_residual_ms: float = 10.0
    #: Physical bound on an unfolded radial velocity, m/s.  No atmosphere
    #: produces a radial wind past this at the scales a superob cell
    #: averages over, so a gate that lands beyond it is a failure of the
    #: unfolding rather than a measurement.  It applies everywhere, including
    #: where no reference exists to check against.
    #:
    #: This is the one tunable above that BOTH engines honour, through
    #: :func:`speed_bounded`.  It is the only check the region-global
    #: engine has: that solver assigns a fold to every region it resolved
    #: and reports no refusal of its own, so without this bound a
    #: misresolved region reaches the filter as a confident 114 m/s wind.
    #: Measured on one real 14-cut volume, it refuses 115 of 1,355,617
    #: gates, 91 of them on two mid-level cuts, and touches nothing in the
    #: violent low-level couplet in that volume.
    max_speed_ms: float = 75.0
    #: How far a resolved gate may depart from the environmental reference,
    #: m/s, where a reference exists.  This is the bound that actually holds
    #: the line, because a fold error displaces a gate by a whole Nyquist
    #: interval -- 51 m/s on the tilts that matter here -- while the physical
    #: departures this must not touch are smaller: 40 m/s allows a couplet
    #: with 20 m/s of rotational velocity either side of the environmental
    #: wind, which is a strong mesocyclone.
    #:
    #: It is deliberately a *departure* rather than a speed, because the
    #: quantity a fold error corrupts is the difference from the surrounding
    #: flow, and a violent storm embedded in a strong jet has a large speed
    #: and a modest departure.  Raising it admits stronger couplets and
    #: weaker fold errors together; there is no setting that separates them,
    #: which is why the default sits where a fold error cannot hide.
    max_reference_departure_ms: float = 40.0
    #: Keep a confidently resolved gate whose speed exceeds
    #: ``nyquist_reject_fraction * Vn``.  This is the recovery: that
    #: threshold exists to drop gates that *might* be folded, and a gate this
    #: stage resolved is one whose fold state is known.  Turning it off keeps
    #: the unfolding and throws away its benefit, which is useful exactly
    #: once -- to show that the recovery and not some other change is what
    #: moved a score.
    keep_beyond_reject_fraction: bool = True
    #: Which solver unfolds a sweep -- one of :data:`ENGINES`.  Every field
    #: above except :attr:`max_speed_ms` tunes ``"vad-region"`` and none of
    #: them reaches ``"region-global"``, whose behaviour is fixed by the
    #: vendored crate; they are kept in one object anyway so that a
    #: provenance reader sees the full parameter set that was in force,
    #: including the tunables the selected engine ignored.
    #:
    #: The default is a decision about the observations, not about the
    #: code, and it was made by measurement and by the owner of the
    #: assimilation (2026-08-12).  ``"region-global"`` keeps 3,894 more
    #: velocity cells per volume than the abstainer and 7,099 more than
    #: the mask, runs the dealias stage in 62 ms against 18.3 s, and
    #: recovers the violent low-level couplet the abstainer throws away --
    #: measured on one real 14-cut volume, with the abstainer rejecting
    #: both couplet gates as "unresolved".  ``"vad-region"`` stays
    #: selectable and stays supported; it is the engine to reach for when
    #: an environmental reference is what should decide, and it is what
    #: every run before this default read.
    engine: str = ENGINE_REGION_GLOBAL
    #: Run the region-global engine's refinement pass (upstream "RIFT"):
    #: after the region-global solve it considers small gate-resolution
    #: proposals where one connected region appears to contain two Nyquist
    #: branches, and applies one only when an independent wrapped-vortex
    #: fit picks the same branch and a bounded cut lowers the local
    #: energy.  Meaningless for ``"vad-region"``, which has no such pass,
    #: and rejected there rather than ignored: a switch that silently does
    #: nothing is how a run gets reported as having had a treatment it
    #: never got.
    #:
    #: ``None`` means "this engine's default", which is the only spelling
    #: that lets one default serve two engines: on ``"region-global"`` it
    #: resolves to on, and on ``"vad-region"`` -- which has no such pass --
    #: to off.  It is resolved to a bool at construction, so provenance
    #: never carries the word "default" where a reader needs a fact.
    #:
    #: On by default because it self-declines: it proposes nothing unless
    #: its own detector fires, and it applies nothing unless an
    #: independent fit agrees.  On the volume this lane measured, it
    #: detected two regions of interest, solved one, made a vortex
    #: proposal and DECLINED it as branch-unstable -- moving zero folds on
    #: any of the 14 cuts -- for about 52 ms on a 62 ms volume.  A pass
    #: that costs milliseconds and abstains where it is unsure belongs on
    #: the default path; leaving it off would mean the case it is for
    #: arrives on a run nobody asked to protect (owner ruling,
    #: 2026-08-12).
    refinement: bool | None = None

    def __post_init__(self) -> None:
        if self.refinement is None:
            object.__setattr__(
                self, "refinement",
                str(self.engine) == ENGINE_REGION_GLOBAL)
        self.validate()
        for field_ in fields(self):
            value = getattr(self, field_.name)
            if field_.name == "engine":
                object.__setattr__(self, field_.name, str(value))
            elif field_.type == "bool" or isinstance(value, bool):
                object.__setattr__(self, field_.name, bool(value))
            elif field_.name in _POSITIVE_INT_FIELDS:
                object.__setattr__(self, field_.name, int(value))
            else:
                object.__setattr__(self, field_.name, float(value))

    def validate(self) -> "DealiasParams":
        """Refuse any value that cannot do the job the field is named for."""

        if not isinstance(self.engine, str):
            raise DealiasParamsError(
                f"engine is {self.engine!r} ({type(self.engine).__name__}); "
                f"it names a solver and must be one of {list(ENGINES)}")
        if self.engine not in ENGINES:
            raise DealiasParamsError(
                f"engine is {self.engine!r}; known engines are "
                f"{list(ENGINES)}. An unknown name cannot be silently "
                "resolved to a default -- the two engines make different "
                "decisions about which velocities a filter may believe")
        if not isinstance(self.refinement, bool):
            raise DealiasParamsError(
                f"refinement is {self.refinement!r} "
                f"({type(self.refinement).__name__}); it is a switch and "
                "only a bool states which way it is thrown")
        if self.refinement and self.engine != ENGINE_REGION_GLOBAL:
            raise DealiasParamsError(
                f"refinement is on with engine={self.engine!r}, which has no "
                f"refinement pass; only {ENGINE_REGION_GLOBAL!r} does. A "
                "switch that is accepted and then ignored is how a run gets "
                "reported as having had a treatment it never got")
        for field_ in fields(self):
            value = getattr(self, field_.name)
            if field_.name in ("keep_beyond_reject_fraction", "refinement"):
                if not isinstance(value, bool):
                    raise DealiasParamsError(
                        f"{field_.name} is {value!r} "
                        f"({type(value).__name__}); it is a switch and only a "
                        "bool states which way it is thrown")
                continue
            if field_.name == "engine":
                continue
            if isinstance(value, bool) or not isinstance(value, Real):
                raise DealiasParamsError(
                    f"{field_.name} is {value!r} ({type(value).__name__}); "
                    "every dealiasing parameter is a real number. A numeric "
                    "string passes float() and then raises at the first "
                    "arithmetic use, and a bool passes every range check as "
                    "0.0 or 1.0")
        for name in _FRACTION_FIELDS:
            value = float(getattr(self, name))
            if not np.isfinite(value) or not (0.0 <= value <= 1.0):
                raise DealiasParamsError(
                    f"{name} is {value!r}; it is a fraction and must lie in "
                    "[0, 1]")
        for name in _POSITIVE_FIELDS:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise DealiasParamsError(
                    f"{name} is {value!r}; it must be finite and strictly "
                    "positive")
        for name in _POSITIVE_INT_FIELDS:
            value = getattr(self, name)
            if float(value) != int(value) or int(value) < 1:
                raise DealiasParamsError(
                    f"{name} is {value!r}; it counts gates, pairs, sectors or "
                    "iterations and must be a whole number of at least 1")
        if int(self.reference_min_sectors) > 12:
            raise DealiasParamsError(
                f"reference_min_sectors is {self.reference_min_sectors!r}; "
                "the fit divides the compass into 12 sectors, so demanding "
                "more than 12 rejects every band unconditionally and leaves "
                "the reference permanently unavailable")
        return self

    def to_payload(self) -> dict:
        payload = {}
        for key, value in asdict(self).items():
            if key == "engine":
                payload[key] = str(value)
            elif isinstance(value, bool):
                payload[key] = bool(value)
            elif key in _POSITIVE_INT_FIELDS:
                payload[key] = int(value)
            else:
                payload[key] = float(value)
        return payload


@dataclass(frozen=True)
class WindProfile:
    """An environmental wind profile to anchor against, ``u``/``v`` on height.

    This is the "model background" route.  Heights are metres above the radar
    antenna, ascending; ``u`` is eastward and ``v`` northward, m/s.  It exists
    beside the self-derived VAD because in a volume the storm dominates, a
    VAD fitted to that volume is pulled by the storm, and a background
    profile is not.
    """

    height_m: np.ndarray
    u_ms: np.ndarray
    v_ms: np.ndarray

    def __post_init__(self) -> None:
        height = np.asarray(self.height_m, dtype=np.float64).ravel()
        u = np.asarray(self.u_ms, dtype=np.float64).ravel()
        v = np.asarray(self.v_ms, dtype=np.float64).ravel()
        if not (height.size == u.size == v.size):
            raise DealiasParamsError(
                f"wind profile has {height.size} heights, {u.size} u and "
                f"{v.size} v; a profile is a triple of equal length")
        if height.size < 2:
            raise DealiasParamsError(
                "a wind profile needs at least two levels to interpolate "
                "between; one level is a constant wearing a profile's name")
        if not np.all(np.isfinite(height)):
            raise DealiasParamsError("wind profile heights are not all finite")
        if not np.all(np.diff(height) > 0.0):
            raise DealiasParamsError(
                "wind profile heights are not strictly ascending; np.interp "
                "would return silently wrong winds rather than raise")
        object.__setattr__(self, "height_m", height)
        object.__setattr__(self, "u_ms", u)
        object.__setattr__(self, "v_ms", v)

    def radial_reference(self, azimuth_deg: np.ndarray,
                         elevation_deg: np.ndarray,
                         height_m: np.ndarray) -> np.ndarray:
        """``Vr`` this profile predicts, on the beam it is asked about.

        ``Vr = (u sin(az) + v cos(az)) cos(elev)``; the vertical term is left
        out because a background profile does not carry hydrometeor fall
        speed and inventing one would be the guess this module exists to
        avoid.  It is bounded by ``w sin(elev)``, under 3 m/s at every
        elevation this stage sees, and the fold decision it feeds has a
        tolerance of a whole Nyquist.
        """

        azimuth = np.radians(np.asarray(azimuth_deg, dtype=np.float64))
        elevation = np.radians(np.asarray(elevation_deg, dtype=np.float64))
        height = np.asarray(height_m, dtype=np.float64)
        # No extrapolation.  ``np.interp`` holds the end value flat beyond
        # the profile, which would quietly assert the 13 km wind at 20 km and
        # anchor gates to it.  Outside the sampled range this profile knows
        # nothing and says so.
        inside = (height >= self.height_m[0]) & (height <= self.height_m[-1])
        u = np.where(inside, np.interp(height, self.height_m, self.u_ms),
                     np.nan)
        v = np.where(inside, np.interp(height, self.height_m, self.v_ms),
                     np.nan)
        return (u * np.sin(azimuth) + v * np.cos(azimuth)) * np.cos(elevation)


@dataclass
class SweepDealiasResult:
    """One sweep, unfolded -- and the full account of what that cost."""

    #: ``(radial, gate)`` dealiased velocity.  NaN wherever the state is
    #: :data:`STATE_REJECTED`, so a consumer that ignores ``state`` still
    #: cannot assimilate a gate this stage refused.
    velocity: np.ndarray
    #: ``(radial, gate)`` of :data:`STATE_REJECTED` / ``UNCHANGED`` /
    #: ``UNFOLDED`` -- ``int8``.
    state: np.ndarray
    #: ``(radial, gate)`` of ``REASON_*`` -- ``int8``.
    reason: np.ndarray
    #: ``(radial, gate)`` integer fold applied, 0 where none.
    fold: np.ndarray
    #: ``(radial, gate)`` reference velocity, NaN where no band qualified.
    reference: np.ndarray
    stats: dict = field(default_factory=dict)

    @property
    def usable(self) -> np.ndarray:
        return self.state != STATE_REJECTED


def speed_bounded(values: np.ndarray, state: np.ndarray, reason: np.ndarray,
                  *, max_speed_ms: float) -> np.ndarray:
    """Refuse every gate whose resolved speed leaves the physical bound.

    The one observation-QC layer both engines apply, in one implementation,
    marking :data:`STATE_REJECTED` with :data:`REASON_SPEED` and returning
    the mask it marked so a caller can NaN the velocities and zero the
    folds it just refused.

    It is a *rejection*, never a clamp and never a pass.  A clamped gate is
    a fabricated observation with a plausible magnitude -- the filter
    cannot tell it from a measurement -- and a gate passed through at 114
    m/s is a misresolution the filter will believe. Both are worse than an
    abstention, which costs one gate and says so by name in the account.

    ``values`` may be the whole plane or the flat vector of finite gates;
    non-finite entries are left alone, so a plane whose no-data gates are
    already ``STATE_REJECTED`` for their own reason keeps that reason.
    """

    beyond = np.isfinite(values) & (np.abs(values) > float(max_speed_ms))
    state[beyond] = STATE_REJECTED
    reason[beyond] = REASON_SPEED
    return beyond


def _sweep_wraps(azimuth_deg: np.ndarray) -> bool:
    """Does this cut close the circle?

    A 360-degree surveillance cut's last radial neighbours its first, and
    leaving that seam open splits every region that crosses north.  A sector
    scan's does not, and welding it shut would invent a boundary.  The test
    is the same one BowEcho's ``sweep_wraps`` makes: the gap from the last
    radial back to the first is no larger than a typical radial step.
    """

    azimuth = np.asarray(azimuth_deg, dtype=np.float64).ravel()
    if azimuth.size < 3:
        return False
    steps = np.abs(np.diff(azimuth))
    steps = np.where(steps > 180.0, 360.0 - steps, steps)
    steps = steps[np.isfinite(steps) & (steps > 0.0)]
    if steps.size == 0:
        return False
    typical = float(np.median(steps))
    closing = abs(float(azimuth[0]) - float(azimuth[-1]))
    if closing > 180.0:
        closing = 360.0 - closing
    return closing <= 2.0 * typical


#: Coarse search grid for the harmonic fit, in wind speed (m/s) and
#: direction (degrees).  The grid is over the *wind vector* rather than over
#: the two harmonic coefficients because the physical range of one is known
#: and of the other is not: no environmental wind exceeds 80 m/s, and 2 m/s
#: by 5 degrees is finer than the fit needs to converge from.
_COARSE_SPEEDS = np.arange(0.0, 82.0, 2.0)
_COARSE_DIRECTIONS = np.radians(np.arange(0.0, 360.0, 5.0))
#: Samples the coarse search looks at.  It only has to find the right basin,
#: and a few hundred samples spread over the compass locate a first harmonic
#: as well as ten thousand do.
_COARSE_SAMPLES = 384
#: The wind grid, as the two harmonic coefficients, in the order the search
#: reports them.  Built once: it is the same grid for every band of every
#: sweep of every volume.
_COARSE_A = (_COARSE_SPEEDS[:, None] * np.cos(_COARSE_DIRECTIONS[None, :])
             ).ravel()
_COARSE_B = (_COARSE_SPEEDS[:, None] * np.sin(_COARSE_DIRECTIONS[None, :])
             ).ravel()
#: How many candidates the native shortlist keeps before the exact cost
#: ranks them.  Larger than it needs to be for a well-separated band, and
#: large enough to swallow the 72 identical zero-speed candidates that a calm
#: band ties on.  It grows when the guard below is not satisfied.
_COARSE_SHORTLIST = 128
#: The absolute error the native cost surface is trusted to within.  The
#: kernel's rotation recurrence drifts by ~1e-11 on a 384-sample band against
#: a directly evaluated cost; this is five orders of magnitude of headroom
#: over that, and still eleven orders below the separation between distinct
#: candidates on this grid.  It is a *bound used in a proof*, not a
#: tolerance on an answer -- see :func:`_coarse_shortlisted`.
_COARSE_COST_TOLERANCE = 1e-6


def _coarse_subsample(values: np.ndarray, sin_az: np.ndarray,
                      cos_az: np.ndarray) -> tuple:
    """The samples the coarse search looks at, and only those."""

    if values.size > _COARSE_SAMPLES:
        pick = np.linspace(0, values.size - 1, _COARSE_SAMPLES).astype(np.intp)
        return values[pick], sin_az[pick], cos_az[pick]
    return values, sin_az, cos_az


def _coarse_cost(values: np.ndarray, sin_az: np.ndarray, cos_az: np.ndarray,
                 nyquist: float, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Wrapped cost of each ``(a, b)``.  The arithmetic of record.

    Every seed this module returns is ranked by this expression, whether the
    candidates reaching it are the whole grid or a natively shortlisted part
    of it.  Nothing else is allowed to order candidates.
    """

    model = a[:, None] * cos_az[None, :] + b[:, None] * sin_az[None, :]
    return np.sum(1.0 - np.cos(np.pi * (values[None, :] - model) / nyquist),
                  axis=1)


def _coarse_shortlisted(approx: np.ndarray, values: np.ndarray,
                        sin_az: np.ndarray, cos_az: np.ndarray,
                        nyquist: float, count: int) -> tuple[list, int]:
    """Rank a shortlist, having first proved the shortlist is enough.

    ``approx`` is the native cost surface: accurate to
    :data:`_COARSE_COST_TOLERANCE` but not NumPy's arithmetic, so it may not
    choose.  It may only *exclude*, and here is the argument that lets it.

    Write ``eps`` for the tolerance, ``T`` for the approximate cost of the
    worst shortlisted candidate and ``C`` for the approximate cost of the
    ``count``-th best.  Every excluded candidate has approximate cost at
    least ``T``, so its true cost is at least ``T - eps``; the ``count``-th
    best shortlisted candidate has true cost at most ``C + eps``.  When
    ``T - C > 2 eps`` every excluded candidate is *strictly* worse than the
    ``count``-th shortlisted one, so the true best ``count`` all lie in the
    shortlist -- and strictly, so no exclusion can have been a tie.  The
    shortlist is then ranked by :func:`_coarse_cost`, in ascending index
    order, which reproduces a stable argsort over the full grid restricted to
    those indices.  The result is the same tuple the exhaustive path returns,
    including which of several tied candidates wins.

    When the guard fails the shortlist grows, and in the limit it is the
    whole grid, which is the exhaustive search itself.  A band whose
    candidates genuinely tie -- a calm band, where all 72 zero-speed
    candidates share a cost -- takes that route rather than a guess.
    Returns the seeds and how many candidates the exact cost had to price.
    """

    order = np.argsort(approx, kind="stable")
    total = int(order.size)
    size = min(_COARSE_SHORTLIST, total)
    while size < total:
        margin = float(approx[order[size - 1]] - approx[order[count - 1]])
        if margin > 2.0 * _COARSE_COST_TOLERANCE:
            break
        size = min(total, size * 4)
    shortlist = (np.arange(total, dtype=np.intp) if size >= total
                 else np.sort(order[:size]))
    cost = _coarse_cost(values, sin_az, cos_az, nyquist,
                        _COARSE_A[shortlist], _COARSE_B[shortlist])
    chosen = shortlist[np.argsort(cost, kind="stable")[:count]]
    return ([(float(_COARSE_A[index]), float(_COARSE_B[index]))
             for index in chosen], int(shortlist.size))


def _coarse_seed_table(bands: list, nyquist: float, count: int = 3,
                       stats: dict | None = None) -> list:
    """Coarse seeds for every band of one sweep, in one native call.

    The band loop that follows is sequential -- each fit carries into the
    next -- but this search is not: a band's cost surface depends on nothing
    but that band.  Handing the whole sweep's worth to the parallel backend
    at once is what makes the kernel worth crossing a language boundary for;
    per band it would be 1214 thread launches on a volume.
    """

    prepared = [None if entry is None else _coarse_subsample(*entry)
                for entry in bands]
    present = [index for index, entry in enumerate(prepared)
               if entry is not None and entry[0].size]
    table: list = [[] for _ in prepared]
    if not present:
        return table
    approx = None
    try:
        approx = coarse_cost.coarse_cost_batch(
            [prepared[index] for index in present],
            _COARSE_SPEEDS, _COARSE_DIRECTIONS, nyquist)
    except ValueError:
        # A refusal from the kernel is a fact about this input, and the
        # NumPy path below answers the same question without it.
        approx = None
    priced = 0
    for slot, band in enumerate(present):
        values, sin_az, cos_az = prepared[band]
        if approx is None:
            table[band] = _coarse_pick(values, sin_az, cos_az, nyquist, count)
            priced += _COARSE_A.size
            continue
        table[band], used = _coarse_shortlisted(
            approx[slot], values, sin_az, cos_az, nyquist, count)
        priced += used
    if stats is not None:
        stats["coarse_native"] = approx is not None
        stats["coarse_bands"] = len(present)
        stats["coarse_candidates_priced"] = priced
    return table


def _coarse_pick(values: np.ndarray, sin_az: np.ndarray, cos_az: np.ndarray,
                 nyquist: float, count: int) -> list:
    """The exhaustive search, priced entirely in NumPy."""

    cost = _coarse_cost(values, sin_az, cos_az, nyquist, _COARSE_A, _COARSE_B)
    order = np.argsort(cost, kind="stable")[:count]
    return [(float(_COARSE_A[index]), float(_COARSE_B[index]))
            for index in order]


def _coarse_seeds(values: np.ndarray, sin_az: np.ndarray, cos_az: np.ndarray,
                  nyquist: float, count: int = 3) -> list:
    """Best ``(a, b)`` starting points by exhaustive search on wrapped cost.

    The alternating fit in :func:`_fit_band` finds the nearest minimum, and
    when most of a sweep is folded there are several: a sinusoid displaced by
    a Nyquist interval over part of the compass explains its own fold
    assignment perfectly and sits in a basin the local step cannot leave.
    Measured on a synthetic 39.7 m/s wind against a 25.51 m/s Nyquist -- more
    than half the sweep folded -- the local fit converged on a wind of the
    wrong sign with a 5.6 m/s residual, comfortably inside the quality gate.
    A wrong reference that passes its own quality gate is the worst failure
    available to this module, because everything downstream anchors to it.

    The wrapped cost is periodic in ``2 * Vn``, so it cannot prefer a
    candidate merely for folding; searching it exhaustively over the physical
    range of winds finds the true basin regardless of how much of the sweep
    is folded.

    This is the definition, kept whole and kept reachable: it is what runs
    when no native library is present, and it is what the parity test holds
    the native path against band for band.
    """

    # v(az) = a cos(az) + b sin(az); (a, b) traced over speed and direction.
    return _coarse_pick(*_coarse_subsample(values, sin_az, cos_az),
                        nyquist, count)


def _fit_band(values: np.ndarray, sin_az: np.ndarray, cos_az: np.ndarray,
              nyquist: float, params: DealiasParams,
              seeds) -> tuple[float, float, float, int] | None:
    """Fold-aware first-harmonic fit for one range band.

    ``v(az) = a cos(az) + b sin(az)`` (Browning & Wexler 1968) fitted to
    velocities that may themselves be folded, by alternating a fold
    assignment against the current model with a linear least squares on the
    unfolded values.  Straight least squares cannot be used here and that is
    the whole difficulty: the samples this reference must be built from are
    the very samples whose folds are unknown, so a plain fit is dragged by
    every folded gate toward a wind that is wrong by a Nyquist interval.

    The alternation has local minima -- a fit sitting one interval off
    explains its own folds perfectly -- so it is run from several seeds and
    the one with the lowest *wrapped* cost wins.  The wrapped cost is
    periodic in ``2 * Vn`` by construction, so it cannot prefer a solution
    merely for being folded.

    Returns ``(a, b, residual_rms, inliers)`` or None when the band does not
    qualify.
    """

    interval = 2.0 * nyquist
    if values.size < params.reference_min_samples:
        return None
    sector = np.floor(np.degrees(np.arctan2(sin_az, cos_az)) % 360.0 / 30.0)
    if np.unique(sector).size < params.reference_min_sectors:
        return None

    design = np.stack([cos_az, sin_az], axis=1)
    best = None
    best_cost = np.inf
    for seed in seeds:
        a, b = float(seed[0]), float(seed[1])
        previous = None
        for _ in range(params.reference_max_iterations):
            model = a * cos_az + b * sin_az
            fold = np.rint((model - values) / interval)
            if previous is not None and np.array_equal(fold, previous):
                break
            previous = fold
            unfolded = values + interval * fold
            try:
                solution, *_ = np.linalg.lstsq(design, unfolded, rcond=None)
            except np.linalg.LinAlgError:      # pragma: no cover - degenerate
                break
            a, b = float(solution[0]), float(solution[1])
        model = a * cos_az + b * sin_az
        cost = float(np.sum(1.0 - np.cos(np.pi * (values - model) / nyquist)))
        if cost < best_cost:
            best_cost = cost
            best = (a, b)
    if best is None:                            # pragma: no cover - no seeds
        return None

    # Trim pass: refit without the samples the first fit could not explain.
    # A storm core inside an otherwise environmental band is exactly such a
    # sample, and letting it set the reference would bend the anchor toward
    # the thing being measured.
    a, b = best
    for _ in range(2):
        model = a * cos_az + b * sin_az
        fold = np.rint((model - values) / interval)
        residual = values + interval * fold - model
        inlier = np.abs(residual) <= params.reference_trim_ms
        if int(inlier.sum()) < params.reference_min_samples:
            return None
        try:
            solution, *_ = np.linalg.lstsq(design[inlier], (values + interval * fold)[inlier],
                                           rcond=None)
        except np.linalg.LinAlgError:           # pragma: no cover - degenerate
            return None
        a, b = float(solution[0]), float(solution[1])

    model = a * cos_az + b * sin_az
    fold = np.rint((model - values) / interval)
    residual = values + interval * fold - model
    inlier = np.abs(residual) <= params.reference_trim_ms
    inliers = int(inlier.sum())
    if inliers < params.reference_min_samples:
        return None
    if inliers < params.reference_min_inlier_fraction * values.size:
        return None
    sector = sector[inlier]
    if np.unique(sector).size < params.reference_min_sectors:
        return None
    rms = float(np.sqrt(np.mean(residual[inlier] ** 2)))
    if not np.isfinite(rms) or rms > params.reference_max_residual_ms:
        return None
    # A residual says the harmonic explains its own fold assignment; only
    # this says the harmonic is a wind.  A fit that has locked onto the wrong
    # branch is internally tidy and physically absurd, and it is the absurdity
    # that gives it away.
    if np.hypot(a, b) > params.reference_max_speed_ms:
        return None
    return a, b, rms, inliers


def vad_reference(velocity: np.ndarray, azimuth_deg: np.ndarray,
                  nyquist: float, params: DealiasParams,
                  *, prior: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """A fold-aware VAD reference field for one sweep.

    Returns ``(reference, stats)`` where ``reference`` is ``(radial, gate)``
    with NaN in every band that did not qualify.  ``prior``, when given, is a
    ``(radial, gate)`` reference from somewhere the volume did not come from
    -- a model background -- and is used only to seed the fit, never to
    override it: a seed changes which minimum the alternation finds, and the
    wrapped cost still decides.
    """

    velocity = np.asarray(velocity, dtype=np.float64)
    rows, gates = velocity.shape
    azimuth = np.radians(np.asarray(azimuth_deg, dtype=np.float64).ravel())
    sin_az = np.sin(azimuth)[:, None]
    cos_az = np.cos(azimuth)[:, None]
    reference = np.full(velocity.shape, np.nan, dtype=np.float64)

    band_gates = int(params.reference_band_gates)
    bands = max(1, -(-gates // band_gates))
    vad_timing = perf_timing.phases("obs.dealias.vad")
    # `band_fit` covers the gather, the batched coarse search and the
    # sequential fits together, which is exactly the span it covered when
    # this was one loop: the phase name and what it accounts for are
    # unchanged, so a receipt from before the search was batched still
    # compares against one from after.
    vad_timing.mark("band_fit", bands=bands)
    # Every band's samples, gathered before any of them is fitted.  The fits
    # are sequential and the searches that seed them are not, so the searches
    # go out together; see :func:`_coarse_seed_table`.
    gathered: list = [None] * bands
    for band in range(bands):
        start = band * band_gates
        stop = min(start + band_gates, gates)
        block = velocity[:, start:stop]
        finite = np.isfinite(block)
        if not finite.any():
            continue
        gathered[band] = (block[finite],
                          np.broadcast_to(sin_az, block.shape)[finite],
                          np.broadcast_to(cos_az, block.shape)[finite],
                          finite)
    coarse_stats: dict = {}
    coarse = _coarse_seed_table(
        [None if entry is None else entry[:3] for entry in gathered],
        nyquist, stats=coarse_stats)

    residuals = []
    carried = (0.0, 0.0)
    fitted: list = [None] * bands
    for band in range(bands):
        if gathered[band] is None:
            continue
        start = band * band_gates
        stop = min(start + band_gates, gates)
        values, sin_block, cos_block, finite = gathered[band]

        seeds = [carried, (0.0, 0.0)]
        seeds.extend(coarse[band])
        # A plain fit on the small-|v| samples: contaminated by folds, but a
        # cheap and usually majority-correct starting point.
        near = np.abs(values) <= 0.5 * nyquist
        if int(near.sum()) >= 8:
            design = np.stack([cos_block[near], sin_block[near]], axis=1)
            try:
                solution, *_ = np.linalg.lstsq(design, values[near], rcond=None)
                seeds.append((float(solution[0]), float(solution[1])))
            except np.linalg.LinAlgError:       # pragma: no cover - degenerate
                pass
        if prior is not None:
            block_prior = np.asarray(prior, dtype=np.float64)[:, start:stop]
            good = finite & np.isfinite(block_prior)
            if int(good.sum()) >= 8:
                design = np.stack([np.broadcast_to(cos_az, finite.shape)[good],
                                   np.broadcast_to(sin_az, finite.shape)[good]],
                                  axis=1)
                try:
                    solution, *_ = np.linalg.lstsq(design, block_prior[good],
                                                   rcond=None)
                    seeds.append((float(solution[0]), float(solution[1])))
                except np.linalg.LinAlgError:   # pragma: no cover - degenerate
                    pass

        fit = _fit_band(values, sin_block, cos_block, nyquist, params, seeds)
        if fit is None:
            continue
        a, b, rms, _inliers = fit
        carried = (a, b)
        fitted[band] = (a, b, rms)

    # Band-to-band continuity.  The environmental wind is continuous in
    # height, so a band that disagrees with the bands either side of it by
    # more than a real shear could is the odd one out.  This is what catches
    # an isolated wrong fit: a wrong branch has no reason to agree with its
    # neighbours, and a right one always does.  Comparisons are against the
    # ORIGINAL fits rather than against a running result, so the outcome does
    # not depend on the order bands are visited in.
    vad_timing.mark("continuity", bands=bands)
    window = int(params.reference_band_window)
    step = float(params.reference_max_band_step_ms)
    kept = list(fitted)
    for band in range(bands):
        if fitted[band] is None:
            continue
        low = max(0, band - window)
        high = min(bands, band + window + 1)
        neighbours = [fitted[other] for other in range(low, high)
                      if other != band and fitted[other] is not None]
        if not neighbours:
            # Nothing to corroborate it.  A lone band is believed only if it
            # is unremarkable on its own terms; the speed bound already
            # applied, so this is not a free pass.
            continue
        median_a = float(np.median([entry[0] for entry in neighbours]))
        median_b = float(np.median([entry[1] for entry in neighbours]))
        if np.hypot(fitted[band][0] - median_a,
                    fitted[band][1] - median_b) > step:
            kept[band] = None

    # Reconcile against the volume-wide profile, when one was supplied.  A
    # single range band sees one height once; the volume sees most heights
    # several times over, from different elevations at different ranges.
    # Where the two disagree past a real shear the volume wins, and the band
    # is replaced rather than dropped -- replacing keeps the anchor, and
    # dropping it would cost the gates this capability exists to recover.
    vad_timing.mark("profile_reconcile", bands=bands)
    replaced = 0
    if prior is not None:
        prior_plane = np.asarray(prior, dtype=np.float64)
        design_full = np.stack([cos_az.ravel(), sin_az.ravel()], axis=1)
        for band in range(bands):
            start = band * band_gates
            stop = min(start + band_gates, gates)
            column = prior_plane[:, start:stop]
            usable = np.isfinite(column).all(axis=1)
            if int(usable.sum()) < 8:
                continue
            try:
                solution, *_ = np.linalg.lstsq(
                    design_full[usable], column[usable].mean(axis=1),
                    rcond=None)
            except np.linalg.LinAlgError:        # pragma: no cover
                continue
            profile_fit = (float(solution[0]), float(solution[1]))
            entry = kept[band]
            if entry is None:
                kept[band] = (profile_fit[0], profile_fit[1], float("nan"))
                continue
            if np.hypot(entry[0] - profile_fit[0],
                        entry[1] - profile_fit[1]) >                     params.reference_max_profile_departure_ms:
                kept[band] = (profile_fit[0], profile_fit[1], float("nan"))
                replaced += 1

    vad_timing.mark("assemble", bands=bands)
    valid_bands = 0
    for band, entry in enumerate(kept):
        if entry is None:
            continue
        a, b, rms = entry
        valid_bands += 1
        if np.isfinite(rms):
            residuals.append(rms)
        start = band * band_gates
        stop = min(start + band_gates, gates)
        reference[:, start:stop] = a * cos_az + b * sin_az

    stats = {
        "bands": int(bands),
        "bands_fitted": int(sum(entry is not None for entry in fitted)),
        "bands_valid": int(valid_bands),
        "bands_from_profile": int(replaced),
        "band_fits": [(band, entry[0], entry[1], entry[2])
                      for band, entry in enumerate(kept) if entry is not None],
        "band_residual_rms_ms": (round(float(np.mean(residuals)), 4)
                                 if residuals else None),
        **coarse_stats,
    }
    vad_timing.close()
    return reference, stats


def volume_wind_profile(cuts, params: DealiasParams) -> "WindProfile | None":
    """One ``u(z), v(z)`` for a whole volume, from its own folded velocities.

    A volume reaches most heights more than once -- 4 km sits at 60 km range
    on a 4-degree cut and at 200 km on a 1-degree one -- and those samples
    are independent measurements of the same wind.  A single range band has
    no such redundancy, which is why an isolated band fit can lock onto a
    branch no other sweep sees and then anchor thousands of gates to it.
    This pools every band fit in the volume onto a height grid and takes the
    median, so one wrong band is outvoted by the sweeps that crossed the same
    layer.

    ``cuts`` is an iterable of ``(elevation_deg, velocity, azimuth_deg,
    nyquist, slant_range_m)``.  Returns None when the volume cannot support a
    profile at all, in which case the caller falls back to per-sweep fits and
    the extra check simply does not happen -- an absent profile must never
    become an invented one.
    """

    layers: dict[int, list] = {}
    layer_m = float(params.profile_layer_m)
    for elevation_deg, velocity, azimuth_deg, nyquist, ranges in cuts:
        if nyquist is None or not np.isfinite(nyquist) or nyquist <= 0.0:
            continue
        velocity = np.asarray(velocity, dtype=np.float64)
        if velocity.ndim != 2 or not np.isfinite(velocity).any():
            continue
        _plane, stats = vad_reference(velocity, azimuth_deg, float(nyquist),
                                      params)
        fits = stats.get("band_fits") or ()
        elevation = np.radians(float(elevation_deg))
        cos_elevation = np.cos(elevation)
        if cos_elevation <= 0.0:                 # pragma: no cover - vertical
            continue
        ranges = np.asarray(ranges, dtype=np.float64)
        for band, a, b, _rms in fits:
            centre = band * int(params.reference_band_gates)                 + int(params.reference_band_gates) // 2
            if centre >= ranges.size:
                continue
            height = _beam_height_m(float(ranges[centre]), elevation)
            # v(az) = a cos(az) + b sin(az) and Vr = (u sin + v cos) cos(el),
            # so a is the northward and b the eastward component, both
            # projected onto the beam.
            layers.setdefault(int(height // layer_m), []).append(
                (a / cos_elevation, b / cos_elevation, float(elevation_deg)))

    heights, us, vs = [], [], []
    for layer in sorted(layers):
        samples = layers[layer]
        if len(samples) < int(params.profile_min_samples):
            continue
        if len({sample[2] for sample in samples})                 < int(params.profile_min_elevations):
            continue
        heights.append((layer + 0.5) * layer_m)
        us.append(float(np.median([sample[1] for sample in samples])))
        vs.append(float(np.median([sample[0] for sample in samples])))

    # Continuity in height, for the same reason the bands get it: a layer
    # that disagrees with the layers above and below it by more than a real
    # shear is the odd one out.  The upper layers are where this bites,
    # because they are reached by the fewest cuts.
    window = int(params.profile_layer_window)
    step = float(params.profile_max_layer_step_ms)
    keep = [True] * len(heights)
    for index in range(len(heights)):
        low = max(0, index - window)
        high = min(len(heights), index + window + 1)
        others = [other for other in range(low, high) if other != index]
        if not others:
            continue
        if np.hypot(us[index] - float(np.median([us[o] for o in others])),
                    vs[index] - float(np.median([vs[o] for o in others])))                 > step:
            keep[index] = False
    heights = [h for h, ok in zip(heights, keep) if ok]
    us = [u for u, ok in zip(us, keep) if ok]
    vs = [v for v, ok in zip(vs, keep) if ok]
    if len(heights) < 2:
        return None
    return WindProfile(height_m=np.asarray(heights, dtype=np.float64),
                       u_ms=np.asarray(us, dtype=np.float64),
                       v_ms=np.asarray(vs, dtype=np.float64))


def _beam_height_m(slant_range_m: float, elevation_rad: float) -> float:
    """Beam centre height above the antenna, four-thirds earth.

    Routed through :func:`woof.obs.geometry.beam_geometry` -- the one
    authority for where a gate is -- rather than repeating the triangle here.
    The repeated copy carried ``1.21 * 6371000.0`` while its own docstring
    said four-thirds: 1.21 is the *height* coefficient from the small-angle
    form ``h = r sin(el) + r^2 / (2 k a)`` with ``k = 4/3`` folded together
    with a unit change, not an effective radius, and using it as one put the
    beam centre 60 m low at 100 km and 375 m low at 250 km.  Against 500 m
    pooling layers that is up to three-quarters of a layer, which moves band
    fits into the wrong height bin and so corrupts the volume profile every
    anchor in the sweep is judged against.

    The shared authority also uses WRF's 6370 km sphere rather than 6371,
    which is what keeps this height in the same georeference as the model
    grid the profile is eventually compared with.
    """

    height_msl, _arc, _local_el = beam_geometry(
        float(slant_range_m), np.degrees(float(elevation_rad)), 0.0)
    return float(height_msl)


def _adjacent_pairs(shape: tuple[int, int], wraps: bool):
    """Flat index pairs of every range- and azimuth-adjacent gate pair."""

    rows, gates = shape
    index = np.arange(rows * gates, dtype=np.int64).reshape(rows, gates)
    pairs = []
    if gates > 1:
        pairs.append((index[:, :-1].ravel(), index[:, 1:].ravel()))
    if rows > 1:
        pairs.append((index[:-1, :].ravel(), index[1:, :].ravel()))
    if wraps and rows > 2:
        pairs.append((index[-1, :].ravel(), index[0, :].ravel()))
    if not pairs:
        return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
    return (np.concatenate([left for left, _ in pairs]),
            np.concatenate([right for _, right in pairs]))


def _edge_table(label_a, label_b, jump_folds, params):
    """Per region pair: winning fold, its support, total pairs, mean jump.

    Votes are aggregated the way BowEcho's ``region_core`` aggregates them --
    ``f = round((v_lo - v_hi) / (2 * Vn))`` with the pair ordered so ``lo``
    is the lower region id -- and ties break the same way, on the smaller
    ``|f|`` then the smaller ``f``, so the resolution order does not depend
    on hash iteration order.
    """

    lo = np.minimum(label_a, label_b)
    hi = np.maximum(label_a, label_b)
    # Orient the jump so it always measures "hi relative to lo".
    signed = np.where(label_a <= label_b, jump_folds, -jump_folds)
    fold = np.rint(signed).astype(np.int64)
    keep = np.abs(fold) <= 2 * int(params.max_fold)
    lo, hi, signed, fold = lo[keep], hi[keep], signed[keep], fold[keep]
    if lo.size == 0:
        return {}

    pair_key = np.stack([lo, hi], axis=1)
    unique_pairs, pair_index = np.unique(pair_key, axis=0, return_inverse=True)
    pair_index = pair_index.ravel()
    n_pairs = unique_pairs.shape[0]
    total = np.bincount(pair_index, minlength=n_pairs)

    span = 4 * int(params.max_fold) + 1
    slot = pair_index * span + (fold + 2 * int(params.max_fold))
    counts = np.bincount(slot, minlength=n_pairs * span).reshape(n_pairs, span)
    jump_sum = np.bincount(slot, weights=signed,
                           minlength=n_pairs * span).reshape(n_pairs, span)

    folds = np.arange(span, dtype=np.int64) - 2 * int(params.max_fold)
    # argmax over (count, -|fold|, -fold): lexsort with the last key dominant.
    order = np.lexsort((-folds[None, :].repeat(n_pairs, axis=0),
                        -np.abs(folds)[None, :].repeat(n_pairs, axis=0),
                        counts), axis=1)[:, -1]
    rows = np.arange(n_pairs)
    winning_fold = folds[order]
    winning_votes = counts[rows, order]
    winning_jump = jump_sum[rows, order]

    table = {}
    for row in range(n_pairs):
        votes = int(winning_votes[row])
        if votes == 0:                          # pragma: no cover - guarded
            continue
        mean_jump = float(winning_jump[row]) / votes
        table[(int(unique_pairs[row, 0]), int(unique_pairs[row, 1]))] = {
            "fold": int(winning_fold[row]),
            "votes": votes,
            "total": int(total[row]),
            "mean_jump": mean_jump,
        }
    return table


def dealias_sweep(velocity: np.ndarray, azimuth_deg: np.ndarray,
                  nyquist: float | None, params: DealiasParams,
                  *, reference: np.ndarray | None = None,
                  wraps: bool | None = None,
                  first_gate_m: float | None = None,
                  gate_spacing_m: float | None = None,
                  nyquist_by_radial: np.ndarray | None = None,
                  nyquist_radials_disagree: bool = False
                  ) -> SweepDealiasResult:
    """Unfold one sweep with the engine ``params`` selects, rejecting every
    gate the evidence cannot resolve.

    ``velocity`` is ``(radial, gate)`` raw radial velocity with NaN for no
    data; ``nyquist`` is the sweep's Nyquist velocity, or None when the sweep
    reports none -- in which case nothing here is knowable and every finite
    gate is rejected with :data:`REASON_NO_NYQUIST`, which is the same
    refusal :mod:`woof.obs.superob` already makes.  Both engines make that
    same refusal, so a caller cannot change it by changing solver.

    ``nyquist_by_radial`` is the per-radial Nyquist velocity, one value per
    row of ``velocity``, with NaN (or a nonpositive value) for a radial whose
    Nyquist is unknown.  **This, not the scalar, is what the fold arithmetic
    uses when it is supplied.**  A VCP that splits a cut between PRFs gives
    different radials different Nyquist intervals -- 32 m/s beside 25.51 m/s
    is ordinary -- and collapsing them to the sweep minimum means a one-fold
    correction on the 32 m/s radial is applied as 51.02 m/s instead of 64.0:
    a 12.98 m/s error that is finite, smooth, well inside ``max_speed_ms``,
    and therefore invisible to every downstream guard.  With the array
    supplied each correction is an exact integer multiple of *that radial's
    own* ``2 * Vn``.

    ``nyquist_radials_disagree`` is the sweep's own statement that its
    radials did not all report the same Nyquist.  Passing it True without
    ``nyquist_by_radial`` is refused: it is a pack that knows it is
    nonuniform and cannot say how, which is exactly the case the scalar
    cannot represent.  The refusal is the point -- a legacy pack must not be
    dealiased on a summary minimum just because the originals were dropped
    before Python saw them.

    ``reference`` is an optional externally supplied ``(radial, gate)``
    environmental field -- e.g. :meth:`WindProfile.radial_reference` on the
    model background.  When given it seeds the fit and then *supplements* the
    VAD: bands the VAD could not fit fall back to it, so a sweep too
    storm-filled to fit a wind profile still has something to anchor
    against.  It is consumed by ``"vad-region"`` only; the region-global
    engine carries no environmental reference at all, and is handed none
    rather than being handed one it would ignore.

    ``first_gate_m`` and ``gate_spacing_m`` are the sweep's physical range
    geometry.  They are needed only by the region-global refinement pass,
    which fits a wrapped vortex in real space and cannot do that on gate
    indices.

    Both engines are handed the per-radial array; they do different things
    with it, and :func:`~woof.obs.dealias_region.dealias_sweep_region`
    documents its own limit rather than being handed a screened copy that
    hides it.
    """

    if getattr(params, "engine", ENGINE_VAD_REGION) == ENGINE_REGION_GLOBAL:
        # Function-local for the same reason scipy is: importing this at
        # module scope would make every install that merely reads
        # DealiasParams pay for ctypes and a library search.
        from woof.obs.dealias_region import dealias_sweep_region  # noqa: PLC0415

        return dealias_sweep_region(
            velocity, azimuth_deg, nyquist, params,
            first_gate_m=first_gate_m, gate_spacing_m=gate_spacing_m,
            nyquist_by_radial=nyquist_by_radial,
            nyquist_radials_disagree=nyquist_radials_disagree)

    velocity = np.asarray(velocity, dtype=np.float64)
    if velocity.ndim != 2:
        raise ValueError(
            f"dealias_sweep needs a (radial, gate) plane, got shape "
            f"{velocity.shape}")
    params = params.validate()
    if nyquist_radials_disagree and nyquist_by_radial is None:
        raise DealiasParamsError(
            "this sweep reports nyquist_radials_disagree=True but carries no "
            "nyquist_velocity_ms_by_radial, so the only Nyquist available is "
            "the sweep minimum -- and unfolding a nonuniform cut in the "
            "minimum's interval mis-corrects every radial above it by a whole "
            "fold difference, finite and plausible and undetectable "
            "downstream.  Re-decode the volume with a decoder that emits the "
            "per-radial Nyquist array, or run this volume without dealiasing")
    state = np.zeros(velocity.shape, dtype=np.int8)
    reason = np.full(velocity.shape, REASON_NONFINITE, dtype=np.int8)
    fold_plane = np.zeros(velocity.shape, dtype=np.int16)
    output = np.full(velocity.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(velocity)
    reference_plane = np.full(velocity.shape, np.nan, dtype=np.float64)

    stats = {
        "gates_finite": int(finite.sum()),
        "gates_unchanged": 0,
        "gates_unfolded": 0,
        "gates_rejected": 0,
        "rejected": {name: 0 for name in REASON_NAMES.values()
                     if name not in ("none", "nonfinite", "disabled")},
        "regions": 0,
        "regions_anchored": 0,
        "regions_linked": 0,
        "regions_unresolved": 0,
        "regions_conflict": 0,
        "edges": 0,
        "edges_confident": 0,
        "edges_violated": 0,
        "fold_histogram": {},
        "reference": {},
        "nyquist_ms": None if nyquist is None else float(nyquist),
    }

    rows, gates = velocity.shape
    # ---- 0. per-radial Nyquist ------------------------------------------
    # One value per row from here on.  The scalar survives only as the
    # summary a receipt reports; nothing below divides by it.
    if nyquist_by_radial is None:
        scalar = (np.nan if nyquist is None else float(nyquist))
        nyquist_row = np.full(rows, scalar, dtype=np.float64)
        stats["nyquist_by_radial"] = False
    else:
        nyquist_row = np.asarray(nyquist_by_radial, dtype=np.float64).ravel()
        if nyquist_row.size != rows:
            raise ValueError(
                f"nyquist_by_radial has {nyquist_row.size} entries, the sweep "
                f"has {rows} radials")
        nyquist_row = nyquist_row.copy()
        stats["nyquist_by_radial"] = True
    valid_row = np.isfinite(nyquist_row) & (nyquist_row > 0.0)
    nyquist_row = np.where(valid_row, nyquist_row, np.nan)

    # A radial with no believable Nyquist is not a radial with the sweep's
    # Nyquist.  Its gates are refused here, individually, instead of being
    # carried on a borrowed interval.
    no_nyquist_rows = ~valid_row
    if no_nyquist_rows.any():
        refused = finite & no_nyquist_rows[:, None]
        reason[refused] = REASON_NO_NYQUIST
        stats["rejected"]["no_nyquist"] = int(refused.sum())
        finite = finite & ~no_nyquist_rows[:, None]

    distinct = np.unique(nyquist_row[valid_row]) if valid_row.any() \
        else np.empty(0, dtype=np.float64)
    stats["nyquist_distinct"] = [float(v) for v in distinct]
    stats["nyquist_min_ms"] = float(distinct.min()) if distinct.size else None
    stats["nyquist_max_ms"] = float(distinct.max()) if distinct.size else None
    stats["nyquist_radials_no_value"] = int(no_nyquist_rows.sum())
    stats["nyquist_transition_pairs"] = 0
    if distinct.size:
        # The scalar summary is the minimum, unchanged in meaning: it never
        # licenses a gate its own radial would have rejected.  It is now
        # reported rather than used.
        stats["nyquist_ms"] = float(distinct.min())

    if not distinct.size:
        reason[np.isfinite(velocity)] = REASON_NO_NYQUIST
        total = int(np.isfinite(velocity).sum())
        stats["gates_rejected"] = total
        stats["rejected"]["no_nyquist"] = total
        return SweepDealiasResult(output, state, reason, fold_plane,
                                  reference_plane, stats)
    if not finite.any():
        stats["gates_rejected"] = int(stats["rejected"]["no_nyquist"])
        return SweepDealiasResult(output, state, reason, fold_plane,
                                  reference_plane, stats)

    uniform = bool(distinct.size == 1 and not no_nyquist_rows.any())
    # Placeholder on refused rows so no NaN reaches a comparison; those
    # gates are already out of ``finite`` and can never be indexed into a
    # kept result.
    nyquist_flat = np.repeat(np.where(valid_row, nyquist_row, 1.0), gates)
    interval_flat = 2.0 * nyquist_flat
    if wraps is None:
        wraps = _sweep_wraps(azimuth_deg)

    timing = perf_timing.phases("obs.dealias.sweep")

    # ---- reference -------------------------------------------------------
    timing.mark("reference", gates_finite=int(finite.sum()),
                gates_total=velocity.size)
    # A VAD fit reasons in one Nyquist interval: its wrapped cost, its coarse
    # seeds and its inlier band are all expressed in Vn.  A nonuniform cut
    # therefore gets one fit per constant-Nyquist sector rather than one fit
    # over the mixture, and a sector too thin in azimuth simply fails to fit
    # and anchors nothing.  Sectors do not exchange gates, and step 1 below
    # refuses to place a graph edge across a transition, so no region -- and
    # so no fold -- ever spans two intervals.
    if uniform:
        vad, vad_stats = vad_reference(velocity, azimuth_deg,
                                       float(distinct[0]), params,
                                       prior=reference)
    else:
        vad = np.full(velocity.shape, np.nan, dtype=np.float64)
        azimuth_all = np.asarray(azimuth_deg, dtype=np.float64).ravel()
        sector_stats = []
        bands = bands_valid = 0
        for value in distinct:
            rows_sel = np.flatnonzero(valid_row & (nyquist_row == value))
            prior_sel = (None if reference is None
                         else np.asarray(reference,
                                         dtype=np.float64)[rows_sel])
            plane, sector = vad_reference(
                velocity[rows_sel], azimuth_all[rows_sel], float(value),
                params, prior=prior_sel)
            vad[rows_sel] = plane
            bands += int(sector.get("bands", 0))
            bands_valid += int(sector.get("bands_valid", 0))
            sector_stats.append({
                "nyquist_ms": float(value),
                "radials": int(rows_sel.size),
                "bands": int(sector.get("bands", 0)),
                "bands_valid": int(sector.get("bands_valid", 0)),
            })
        vad_stats = {"bands": bands, "bands_valid": bands_valid,
                     "band_fits": [], "nyquist_sectors": sector_stats}
    if reference is not None:
        external = np.asarray(reference, dtype=np.float64)
        if external.shape != velocity.shape:
            raise ValueError(
                f"reference has shape {external.shape}, the sweep is "
                f"{velocity.shape}")
        vad = np.where(np.isfinite(vad), vad, external)
        vad_stats["external_supplied"] = True
    else:
        vad_stats["external_supplied"] = False
    reference_plane = vad
    has_reference = np.isfinite(vad)
    stats["reference"] = vad_stats

    # ---- 1. segment ------------------------------------------------------
    timing.mark("segment")
    from scipy.sparse import coo_matrix                    # noqa: PLC0415
    from scipy.sparse.csgraph import connected_components  # noqa: PLC0415

    left, right = _adjacent_pairs(velocity.shape, wraps)
    flat = velocity.ravel()
    flat_finite = finite.ravel()
    comparable = flat_finite[left] & flat_finite[right]
    # A pair straddling a Nyquist transition is not evidence about anything.
    # The two gates are quantised on different lattices, so their difference
    # is neither a shear measurement nor a fold count, and both readings of
    # it are wrong in the same direction: joined, it welds two intervals into
    # one region; voted on, it manufactures a fold edge out of a PRF change.
    # It is dropped from both, and counted so the drop is visible.
    same_nyquist = nyquist_flat[left] == nyquist_flat[right]
    stats["nyquist_transition_pairs"] = int((comparable & ~same_nyquist).sum())
    comparable = comparable & same_nyquist
    delta = np.abs(flat[left] - flat[right])
    joined = comparable & (delta
                           <= params.region_join_fraction * nyquist_flat[left])

    nodes = rows * gates
    adjacency = coo_matrix(
        (np.ones(int(joined.sum()), dtype=np.int8),
         (left[joined], right[joined])), shape=(nodes, nodes))
    _n_components, raw_labels = connected_components(
        adjacency, directed=False, return_labels=True)
    raw_labels = np.where(flat_finite, raw_labels, -1)
    present, labels = np.unique(raw_labels[flat_finite], return_inverse=True)
    label_flat = np.full(nodes, -1, dtype=np.int64)
    label_flat[flat_finite] = labels
    n_regions = int(present.size)
    region_size = np.bincount(labels, minlength=n_regions)
    stats["regions"] = n_regions

    # ---- 2. vote ---------------------------------------------------------
    timing.mark("vote", regions=n_regions)
    boundary = comparable & ~joined
    label_a = label_flat[left[boundary]]
    label_b = label_flat[right[boundary]]
    different = label_a != label_b
    jump = ((flat[left[boundary]] - flat[right[boundary]])
            / interval_flat[left[boundary]])
    edges = _edge_table(label_a[different], label_b[different],
                        jump[different], params)
    stats["edges"] = len(edges)

    confident = {}
    for key, entry in edges.items():
        if entry["total"] < params.edge_min_pairs:
            continue
        if entry["votes"] < params.edge_agreement * entry["total"]:
            continue
        if abs(entry["mean_jump"] - entry["fold"]) > params.edge_max_offset:
            continue
        confident[key] = entry
    stats["edges_confident"] = len(confident)

    neighbours: dict[int, list] = {}
    for (lo, hi), entry in confident.items():
        neighbours.setdefault(lo, []).append((hi, entry))
        neighbours.setdefault(hi, []).append((lo, entry))

    # ---- 3. anchor -------------------------------------------------------
    timing.mark("anchor", regions=n_regions, edges=len(edges))
    # A region may anchor only when it has enough gates with a reference,
    # its fold estimate is unambiguous, and -- the essential one -- it
    # still looks environmental after unfolding.  The last is what stops a
    # rotational couplet, whose whole nature is to depart from the
    # background wind, from being flattened onto it.
    anchor_fold: dict[int, int] = {}
    order_flat = np.argsort(labels, kind="stable")
    sorted_labels = labels[order_flat]
    starts = np.searchsorted(sorted_labels, np.arange(n_regions), "left")
    stops = np.searchsorted(sorted_labels, np.arange(n_regions), "right")
    finite_index = np.flatnonzero(flat_finite)
    vad_flat = vad.ravel()
    ref_ok_flat = has_reference.ravel()

    region_gates = []
    for region in range(n_regions):
        region_gates.append(finite_index[order_flat[starts[region]:stops[region]]])

    for region in range(n_regions):
        members = region_gates[region]
        with_reference = members[ref_ok_flat[members]]
        if with_reference.size < params.anchor_min_gates:
            continue
        member_interval = interval_flat[with_reference]
        ratio = ((vad_flat[with_reference] - flat[with_reference])
                 / member_interval)
        estimate = float(np.median(ratio))
        candidate = int(np.rint(estimate))
        if abs(estimate - candidate) > params.anchor_max_offset:
            continue
        if abs(candidate) > params.max_fold:
            continue
        residual = np.abs(flat[with_reference] + member_interval * candidate
                          - vad_flat[with_reference])
        if float(np.median(residual)) > params.anchor_max_residual_ms:
            continue
        anchor_fold[region] = candidate
    stats["regions_anchored"] = len(anchor_fold)

    # ---- 4. grow ---------------------------------------------------------
    timing.mark("grow", regions_anchored=len(anchor_fold))
    import heapq                                           # noqa: PLC0415

    resolved: dict[int, int] = {}
    linked = 0
    anchor_order = sorted(anchor_fold, key=lambda r: (-int(region_size[r]), r))
    heap: list = []
    counter = 0

    def push(region: int) -> None:
        nonlocal counter
        for other, entry in neighbours.get(region, ()):
            if other in resolved:
                continue
            counter += 1
            heapq.heappush(heap, (-entry["total"], other, counter, region,
                                  entry))

    # Which growth pass -- which connected component of the confident-edge
    # graph -- each resolved region belongs to.  Recorded during the walk
    # because the anchor cross-check below rejects by component, and a
    # component is exactly "the set of regions whose folds were derived from
    # one seed": if two anchors inside it disagree, every fold in it rests on
    # an arbitrary choice between them.
    component: dict[int, int] = {}
    for seed in anchor_order:
        if seed in resolved:
            continue
        resolved[seed] = anchor_fold[seed]
        component[seed] = seed
        push(seed)
        while heap:
            _priority, target, _tie, source, entry = heapq.heappop(heap)
            if target in resolved:
                continue
            lo, hi = min(source, target), max(source, target)
            # entry["fold"] is k[hi] - k[lo].
            if target == hi:
                resolved[target] = resolved[source] + entry["fold"]
            else:
                resolved[target] = resolved[source] - entry["fold"]
            component[target] = seed
            linked += 1
            push(target)
    stats["regions_linked"] = linked
    stats["regions_unresolved"] = n_regions - len(resolved)

    # ---- 5. verify -------------------------------------------------------
    timing.mark("verify", edges_confident=len(confident))
    conflicted: set[int] = set()
    violated = 0
    for (lo, hi), entry in confident.items():
        if lo not in resolved or hi not in resolved:
            continue
        if resolved[hi] - resolved[lo] != entry["fold"]:
            violated += 1
            conflicted.add(lo)
            conflicted.add(hi)
    stats["edges_violated"] = violated

    # Every anchored region must still hold its own anchor after the growth
    # walk.  It need not: the walk seeds from the largest anchored region and
    # then propagates folds across confident edges, so a second anchored
    # region reached over those edges is *overwritten* by whatever the edge
    # chain implies, and until now nothing compared the two.  First anchor
    # wins, silently, and the losing anchor -- an independent measurement
    # against the environmental reference -- is discarded without a count.
    #
    # A disagreement is not a tie to be broken.  It says the edge chain and
    # the reference field give different answers for the same gates, and
    # there is no evidence here for preferring either, so the whole component
    # is rejected rather than resolved on a coin toss.
    anchors_disagreed = 0
    disagreeing_components: set[int] = set()
    for region, expected in anchor_fold.items():
        if region not in resolved:
            continue
        if resolved[region] != expected:
            anchors_disagreed += 1
            disagreeing_components.add(component.get(region, region))
    if disagreeing_components:
        for region, seed in component.items():
            if seed in disagreeing_components:
                conflicted.add(region)
    stats["anchors_disagreed"] = anchors_disagreed
    stats["components_anchor_conflict"] = len(disagreeing_components)
    stats["regions_conflict"] = len(conflicted)

    # ---- assign ----------------------------------------------------------
    timing.mark("assign", regions=n_regions)
    region_fold = np.zeros(n_regions, dtype=np.int64)
    region_state = np.zeros(n_regions, dtype=np.int8)
    region_reason = np.full(n_regions, REASON_UNRESOLVED, dtype=np.int8)
    for region, value in resolved.items():
        if region in conflicted:
            region_reason[region] = REASON_CONFLICT
            continue
        if abs(value) > params.max_fold:
            region_reason[region] = REASON_FOLD_RANGE
            continue
        region_fold[region] = value
        region_state[region] = STATE_UNFOLDED if value else STATE_UNCHANGED
        region_reason[region] = REASON_NONE

    gate_fold = region_fold[labels]
    gate_state = region_state[labels]
    gate_reason = region_reason[labels]
    # The modular identity the whole stage rests on: the correction applied
    # to a gate is an integer multiple of ITS OWN radial's Nyquist interval,
    # never of a sweep-wide summary.
    unfolded = flat[flat_finite] + interval_flat[flat_finite] * gate_fold
    speed_bounded(unfolded, gate_state, gate_reason,
                  max_speed_ms=params.max_speed_ms)
    # The bound that actually holds the line.  A fold error displaces a gate
    # by a whole Nyquist interval; the physical departures this must not
    # touch are smaller.  Only applied where a reference exists to depart
    # from -- absence of evidence is not evidence of a fold.
    reference_here = vad_flat[flat_finite]
    departed = (np.isfinite(reference_here)
                & (np.abs(unfolded - reference_here)
                   > params.max_reference_departure_ms)
                & (gate_state != STATE_REJECTED))
    gate_state[departed] = STATE_REJECTED
    gate_reason[departed] = REASON_REFERENCE_DEPARTURE
    unfolded = np.where(gate_state == STATE_REJECTED, np.nan, unfolded)

    output_flat = output.ravel()
    state_flat = state.ravel()
    reason_flat = reason.ravel()
    fold_flat = fold_plane.ravel()
    output_flat[flat_finite] = unfolded
    state_flat[flat_finite] = gate_state
    reason_flat[flat_finite] = gate_reason
    fold_flat[flat_finite] = np.where(gate_state == STATE_REJECTED, 0,
                                      gate_fold)

    # Gates on radials that had no Nyquist at all never entered ``finite``,
    # so they are not in ``gate_state``; they are still rejections and the
    # offered/unchanged/unfolded/rejected identity has to close over them.
    rows_refused = int(stats["rejected"].get("no_nyquist", 0))
    stats["gates_unchanged"] = int((gate_state == STATE_UNCHANGED).sum())
    stats["gates_unfolded"] = int((gate_state == STATE_UNFOLDED).sum())
    stats["gates_rejected"] = int(
        (gate_state == STATE_REJECTED).sum()) + rows_refused
    for code, name in REASON_NAMES.items():
        if name in stats["rejected"]:
            stats["rejected"][name] = int(
                ((gate_state == STATE_REJECTED) & (gate_reason == code)).sum())
    stats["rejected"]["no_nyquist"] += rows_refused
    applied = gate_fold[gate_state != STATE_REJECTED]
    if applied.size:
        values, counts = np.unique(applied, return_counts=True)
        stats["fold_histogram"] = {int(v): int(c)
                                   for v, c in zip(values, counts)}
    timing.close()
    return SweepDealiasResult(output, state, reason, fold_plane,
                              reference_plane, stats)
