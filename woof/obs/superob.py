"""Gate-to-cell superobbing: many range gates in, one observation per cell out.

Two moments, two very different reductions, for physical reasons rather
than stylistic ones.

**Reflectivity** is averaged in *linear* Z and reported back in dBZ, because
dBZ is a logarithm and the mean of logarithms is not the logarithm of the
mean: averaging dBZ directly biases a cell containing one strong core and
several weak gates low by many dB.  The in-cell maximum is carried beside
the mean because convective assimilation usually wants the core, and a
consumer that has to reconstruct one from the other cannot.

**Radial velocity** is averaged per contributing radar and never across
radars: two radars looking at the same cell measure different projections of
the same wind, and their mean is a number with no observation operator.  The
beam unit vector is averaged with the velocities and renormalized, so every
retained velocity ships the look direction that turns it into an
observation.

**Clear air is an observation, and it is built from measurements only.**
A cell is reported clear when enough gates were *measured* inside it and
all of them came back below the significant-echo floor, and no radar saw
echo there.  The gates that support that claim are finite decoded values;
a missing gate never contributes.

Which gates can support that claim depends on what the pack it came from
was able to tell us, and there are two regimes.

**The measurement-only regime** (:data:`CLEAR_AIR_SOURCE`, the default).
A ``gpuwm-obs.radar-sweeps.v1`` pack carries one plane per moment, in which
every unusable gate is the same NaN.  Three quite different things produce
that NaN: raw gate code 0 (*below threshold* -- the radar looked and
detected nothing, which is precisely the clear-air observation), raw code 1
(*range folded* -- an ambiguous second-trip return that may be a storm),
and a radial that never carried the moment at all.  Downstream they are
indistinguishable, so NaN cannot be evidence of clear air without
fabricating observations wholesale: on a real KDMX volume that is about
9.7 million ambiguous gates against 1.1e4 unambiguous ones.  This regime
therefore uses only the unambiguous remainder -- gates that decoded to a
real number below ``min_reflectivity_dbz`` -- and accepts a thin product.

**The censored regime** (:data:`CLEAR_AIR_SOURCE_CENSOR`), available when
the pack is a ``gpuwm-obs.radar-sweeps.v2`` written by
``rw_nexrad decode --censor-flags``.  There the decoder's own reason for
each NaN rides beside it as a :class:`~woof.obs.sweeps.Censor` code, so
"below threshold" is separable from "range folded" and from "never
collected", and the first of those becomes what it always was in the
signal: a measurement of nothing.  Pass ``clear_air_from_censor=True`` to
:func:`superob_volume` to use it.

**The ODIM regime** (:data:`CLEAR_AIR_SOURCE_ODIM`), the same flag over a
``gpuwm-obs.radar-sweeps.v3`` pack written by ``rw_odim pack``.  ODIM
reserves ``undetect`` for "looked, found nothing" and it arrives as the same
code 1, so the admission test does not change; what changes is the *name*
written into the product, because the claim behind it was made by a European
national processor and not by an RDA, and a consumer reading
``clear_air_source`` is entitled to know which.  ODIM also mints two states
NEXRAD has no word for, and the regime's substance is that neither is
admitted: ``NODATA`` is the radar reporting it did not look, and
``SENTINEL_AMBIGUOUS`` is a file that gave both sentinels the same raw value
so the gate may be either.  Both are counted and refused.

**Range-folded gates are never clear air, in either regime.**  Raw code 1
means the radar cannot say which trip the return came from, and the answer
may be a storm.  It is admitted by no configuration of this module: the
censored regime tests for equality with one code
(:data:`~woof.obs.sweeps.Censor.BELOW_THRESHOLD`) rather than for
"non-echo", and the measurement-only regime never sees a non-finite gate at
all.  ``clear_air_source`` in the written file records which regime
produced the zeroes, so a consumer always knows which coverage and which
error model it is holding.

**Aliasing is masked by default, and corrected on request.**  With
``SuperobParams.dealias`` left at its default ``None`` nothing in this
module has changed: the four fail-closed masks below are the whole of the
alias defense, no velocity is ever modified, and the observation files this
stage produces are byte-for-byte the ones it has always produced.  Setting
that field to a :class:`woof.obs.dealias.DealiasParams` turns on
region-based unfolding -- see :mod:`woof.obs.dealias` for the algorithm and
its abstention rule -- which runs per sweep *before* everything below, so
the masks then see velocities whose fold state is known rather than
suspected.  Two things change when it is on and nothing else does:

* a gate the unfolder could not resolve is dropped and counted, whatever its
  magnitude, because "I cannot tell" is not an observation;
* a gate the unfolder *did* resolve is no longer bounded by
  ``nyquist_reject_fraction`` of Nyquist but by an absolute physical speed,
  because that fraction exists to drop gates that might be folded and this
  one's fold state is known.  This is the entire recovery: at a Nyquist of
  25.51 m/s the 0.8 rule caps the assimilable wind at 20.4 m/s, and a
  mesocyclone's couplet lives above that.

**Aliasing is masked, not corrected** (the default path).  Four structural
defenses run, and it matters exactly what each of them can and cannot see:

1. a gate whose speed exceeds a configurable fraction of the sweep's Nyquist
   velocity is dropped and counted;
2. a sweep reporting no Nyquist velocity, or one outside the plausible band,
   has *every* velocity gate dropped and counted;
3. a cell whose retained gates disagree by more than a configurable fraction
   of Nyquist (a fold caught inside one cell) is dropped whole and counted;
4. a **gate-to-gate radial shear scan** along each radial, on the raw gates
   before any magnitude mask, flags range-adjacent pairs whose difference
   exceeds a fraction of the full Nyquist *interval* ``2 * Vn``.  A fold
   between neighbours produces a jump of very nearly ``2 * Vn``; the two
   gates flanking such a jump are dropped and counted, and per-sweep
   boundary counts go into provenance whether or not anything was dropped.

**What rule 4 can catch:** the *edge* of a folded region, where an unfolded
gate sits beside a folded one.  Both gates in such a pair are necessarily
near opposite Nyquist limits, which is the least trustworthy data in the
sweep, so the loss is narrow and lands where it should.

**What no rule here can catch:** a spatially coherent fold covering a whole
region.  At Nyquist 32 m/s a true +69 m/s folds to +5 m/s; a patch of gates
that all fold together has a present and plausible Nyquist, passes the
0.8 magnitude test, has zero in-cell spread, and has no gate-to-gate jump
anywhere in its interior.  It is a smooth, plausible, wrong wind field, and
it is assimilable.  Nothing short of true dealiasing, unwrapping the region
against a global reference rather than testing neighbours, excludes it.
The counters exist so that a region whose *boundary* was flagged is visible
in provenance even when its interior survived; they are evidence, not a
guarantee, and this module does not claim otherwise.

**A fifth mask is opt-in and dual-pol:** correlation-coefficient QC
(:mod:`woof.obs.cc_qc`), enabled by setting ``SuperobParams.cc_qc``.
It runs at the sweep level and *before* the gate-to-gate shear scan,
because non-meteorological echo corrupts fold-state evidence exactly the
way it corrupts everything else.

One ordering caveat, stated because it is a real divergence rather than a
detail.  QC upstream of *all* alias reasoning is the operational order,
and this mask does not achieve it when the region dealiaser is also on:
the unfolder runs once per sweep, above the block loop this mask lives
in, so it reasons about folds on a velocity plane these gates have not
yet been removed from.  What the mask drops is unaffected -- but not for
the reason first recorded here.  The drop decision *does* read velocity:
the debris-fringe exemption's couplet detector is handed a velocity
plane.  What makes the ordering harmless is WHERE the plans carrying
that detector are built -- once, above the block loop, over the volume's
RAW sweeps -- so the dropped set and every counter below are identical
either way.  It is the unfolder's evidence, not the mask's, that is not
the evidence this docstring would prefer it had.

RULING (2026-08-12, the 2.1 assembly): the ordering as assembled is
ACCEPTED for 2.1.  Masking the raw planes before the unfolder is called
is a change to this stage's shape, so it is to be settled by an A/B --
the couplet instrument plus a control region -- before any restructure,
never adopted on the strength of the operational order alone.

Its rule is per moment.  In
*reflectivity* it is compound (low RhoHV AND low reflectivity), never
RhoHV alone -- hail cores, the melting layer and the tornadic debris
signature legitimately depress RhoHV, and a bare threshold deletes
precisely the echo this pipeline exists to observe.  In *velocity*
there is no reflectivity shield: a low-RhoHV gate loses its velocity
however bright it is, debris core included, because a scatterer that
does not move with the air does not carry a wind observation.  One
band is exempted from that, on by default: the debris-signature
*fringe* -- 30 to 35 dBZ, RhoHV between the debris floor and the
velocity threshold, and beside a clustered velocity couplet in the
same sweep -- keeps its velocity, because that band is where a weak or
distant tornado's rotation lives and low RhoHV alone is not evidence
of debris.  Every exempted gate is counted.  Both the asymmetry and
the exemption are owner rulings and their reasoning is recorded in
:mod:`woof.obs.cc_qc`.  Where the
volume carries no RhoHV -- pre-2013 archives, cuts without a usable
dual-pol source, gates beyond the RHO extent -- the mask does nothing
and counts the absence; absent data never fabricates a pass or a fail.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields
from numbers import Real

import math

import numpy as np

from woof import perf_timing
from woof.obs.cc_qc import (REASON_COMPANION, REASON_NO_REF, REASON_NO_RHO,
                             TDS_TURNED_AWAY_ABOVE_Z_BAND,
                             TDS_TURNED_AWAY_BELOW_Z_BAND,
                             TDS_TURNED_AWAY_NO_ROTATION,
                             TDS_TURNED_AWAY_RHO_BELOW_FLOOR,
                             CcQcParams, CcQcParamsError, build_cc_plans)
from woof.obs.dealias import (ENGINE_VAD_REGION, STATE_REJECTED,
                               DealiasParams, DealiasParamsError,
                               dealias_sweep, resolve_default_dealias,
                               volume_wind_profile)
from woof.obs.geometry import (REFRACTION_FACTOR, gate_locations)
from woof.obs.sweeps import SWEEPS_SCHEMA_ODIM, Censor, RadarVolume
from woof.obs.target_grid import TargetGrid
from woof.static.projection import EARTH_RADIUS_M

#: Moment tokens this stage understands.
REFLECTIVITY = "REF"
VELOCITY = "VEL"

#: How the clear-air observations in a product were established.  Written
#: into the observation file so a consumer can tell what it is trusting; a
#: consumer that does not recognise the value must refuse the file rather
#: than assume either one.
#:
#: The two regimes differ in *coverage*, not merely in count.  The
#: measurement-only zeroes sit where a gate decoded to a real number below
#: the floor, which on a quiet volume is a sparse scatter near the radar.
#: The censored zeroes additionally cover everywhere the radar reported
#: below-threshold, which is most of the clear sky it looked at.  Reading
#: one as the other misstates how much of the domain was observed, which is
#: exactly the error that turns a thin accurate product into a confident
#: wrong one.
CLEAR_AIR_SOURCE = "finite_below_floor"
CLEAR_AIR_SOURCE_CENSOR = "below_threshold_and_finite_below_floor"

#: The censored regime over an ODIM pack (``gpuwm-obs.radar-sweeps.v3``).
#:
#: A separate value from :data:`CLEAR_AIR_SOURCE_CENSOR` even though the same
#: gate code carries it, because the two are different *claims* by different
#: instruments. NEXRAD's code 1 is the RDA reporting a return below its
#: significance threshold; ODIM's is the writer's ``undetect``, the value the
#: national processor reserves for "this gate was looked at and held no
#: echo". They coincide in meaning and not in provenance, and a product's
#: ``clear_air_source`` exists so a consumer can tell what it is holding --
#: which it could not do if a Finnish volume and an Oklahoma one wrote the
#: same string.
#:
#: The prohibition matters more than the addition. ODIM mints two codes NEXRAD
#: has no word for and **neither is ever clear air**:
#: :data:`~woof.obs.sweeps.Censor.NODATA` is the radar reporting it did not
#: look, and :data:`~woof.obs.sweeps.Censor.SENTINEL_AMBIGUOUS` is a file
#: that declared ``nodata`` and ``undetect`` as the same raw value, so the
#: gate may be either. Finnish ``VRADH`` does exactly that -- 721,898 gates in
#: one measured volume, 76 % of a single sweep -- and reading them as clear
#: air would assimilate "no echo" into cells that may hold one. The admission
#: test below is equality with one code, so no configuration of this module
#: admits 4 or 5, and both are counted so the refusal is visible in
#: provenance rather than merely asserted here.
CLEAR_AIR_SOURCE_ODIM = "undetect_and_finite_below_floor"

#: Every value :data:`CLEAR_AIR_SOURCE` may take.
CLEAR_AIR_SOURCES = (CLEAR_AIR_SOURCE, CLEAR_AIR_SOURCE_CENSOR,
                     CLEAR_AIR_SOURCE_ODIM)

#: Radials processed per geometry pass.  Peak memory, not correctness.
_RADIAL_BLOCK = 120


class SuperobParamsError(ValueError):
    """A superob parameter that cannot mean what the pass needs it to mean.

    Separate from :class:`ValueError` at the call site so a caller can tell
    "you configured this stage impossibly" from "this volume is malformed".
    Never a warning: every one of these values multiplies or bounds a
    physical threshold, and a wrong one produces observations that are
    finite, plausible and wrong.
    """


class SuperobError(ValueError):
    """The gridding pass reached a state that would have lost observations.

    Reserved for internal invariants -- a reach window that fails to cover
    its own radar's gates, and nothing a volume or a parameter can cause.
    Loud on purpose: the failure it guards silently drops observations,
    which is exactly the class of bug an analysis cannot detect downstream.
    """


#: Fractions of a Nyquist velocity or Nyquist interval.  Outside ``[0, 1]``
#: they stop being fractions: a reject fraction above 1 admits speeds the
#: sweep cannot unambiguously measure, which is precisely the aliased data
#: the gate exists to drop, and a negative one is not an interval at all.
_FRACTION_FIELDS = (
    "nyquist_reject_fraction",
    "nyquist_spread_fraction",
    "shear_fold_fraction",
)

#: Values that bound or scale a physical quantity and are meaningless at or
#: below zero.  The four error fields are standard deviations: a zero sigma
#: is an infinitely confident observation, which in a filter is not a small
#: error but a hard constraint the analysis cannot argue with.
_POSITIVE_FIELDS = (
    "nyquist_min_ms",
    "nyquist_max_ms",
    "max_range_km",
    "z_error_base_dbz",
    "vr_error_base_ms",
    "z_error_floor_dbz",
    "vr_error_floor_ms",
    "refraction_factor",
    "earth_radius_m",
    "clear_air_min_gates",
    "clear_air_error_dbz",
)

#: Values that may take any sign but must be a number.  A NaN reflectivity
#: floor compares False against every gate, so it silently drops the entire
#: volume; an infinite one drops it loudly.  Neither is a floor.
_FINITE_FIELDS = ("min_reflectivity_dbz",)

#: Fields that are parameter objects (or None), not floats: exempt from the
#: float normalization and the real-number runtime check, validated by
#: delegation instead.  Every such field must appear here, or the float
#: normalization in ``__post_init__`` will try to call ``float()`` on a
#: parameter object and the runtime-type check will refuse it as "not a
#: real number".
_NON_FLOAT_FIELDS = ("dealias", "cc_qc")

#: Floor on ``||sum b_i|| / n`` for a superob velocity cell.  Below it the
#: contributing beams disagree about direction so badly that their mean is
#: not a look direction, and ``Vr = b_eff . x`` stops being a statement about
#: the wind.  0.5 is where the beams span roughly a hemisphere; real cells
#: from one radar sit above 0.99 and multi-elevation cells above 0.9, so the
#: default removes broken geometry without touching ordinary geometry.
MIN_BEAM_COHERENCE = 0.5


def _fold_boundaries(values: np.ndarray, nyquist: float,
                     params) -> tuple[np.ndarray, int, int]:
    """Flag gates flanking a range-adjacent jump that only a fold explains.

    ``values`` is ``(radials, gates)`` of **raw** velocities, range-ordered
    along the second axis, before any magnitude mask.  Raw is the only place
    this test works: once the 0.8 gate has run, a folded gate and its
    unfolded neighbour are both inside the retained band and their
    difference is unremarkable.

    A NaN breaks the chain rather than bridging it: two gates either side
    of a data gap are not neighbours, and comparing them would invent a
    boundary out of a range hole.

    Returns ``(flags, boundaries, pairs_tested)``: the per-gate flags, how
    many adjacent pairs were flagged, and how many were comparable at all.
    The last is the denominator without which the middle number means
    nothing.
    """

    flags = np.zeros(values.shape, dtype=bool)
    if values.shape[1] < 2:
        return flags, 0, 0
    delta = np.abs(np.diff(values, axis=1))
    finite = np.isfinite(delta)
    # ``nyquist`` may be one value or one per radial.  The test is
    # range-adjacent -- both gates of every pair are on the same radial --
    # so a per-radial interval is the right one for its own row, and a cut
    # that mixes PRFs never has one row judged in another row's interval.
    scale = np.asarray(nyquist, dtype=np.float64)
    if scale.ndim == 1:
        scale = scale[:, None]
    boundary = finite & (delta > params.shear_fold_fraction * 2.0 * scale)
    flags[:, :-1] |= boundary
    flags[:, 1:] |= boundary
    return flags, int(boundary.sum()), int(finite.sum())


#: The region-graph bookkeeping only the VAD engine publishes.  Named once
#: so the accumulator and the null-out in ``to_payload`` cannot drift.
_GRAPH_COUNTERS = ("regions", "regions_anchored", "regions_linked",
                   "regions_unresolved", "regions_conflict", "edges",
                   "edges_confident", "edges_violated")


@dataclass
class _DealiasTotals:
    """Volume-wide three-state account, summed over sweeps.

    ``unchanged + unfolded + rejected`` equals the number of finite velocity
    gates the unfolder was offered, exactly, for every volume.  That identity
    is the whole contract: a gate cannot quietly fall out of the accounting,
    and a test asserts it rather than trusting it.
    """

    sweeps_dealiased: int = 0
    gates_offered: int = 0
    gates_unchanged: int = 0
    gates_unfolded: int = 0
    gates_rejected: int = 0
    #: The subset of rejections that reached the gridding stage -- a refused
    #: gate outside the grid or beyond range costs nothing, and reporting it
    #: as a loss would overstate the price of abstention.
    gates_refused_at_grid: int = 0
    regions: int = 0
    regions_anchored: int = 0
    regions_linked: int = 0
    regions_unresolved: int = 0
    regions_conflict: int = 0
    edges: int = 0
    edges_confident: int = 0
    edges_violated: int = 0
    reference_bands: int = 0
    reference_bands_valid: int = 0
    #: Which engine produced these totals, or ``"mixed"`` if -- impossibly
    #: today, since one parameter set governs a volume -- two ever did.
    engine: str = ""
    #: False once any sweep's account arrived without the region-graph
    #: counters.  They are the VAD engine's own bookkeeping; the
    #: region-global engine solves a region network too but exposes no
    #: count of it across its C ABI, and reporting zero regions for a
    #: volume it unfolded 90k gates in would be a measurement claim nobody
    #: made.  :meth:`to_payload` writes them as null in that case.
    graph_counters_reported: bool = True
    rejected: dict = field(default_factory=dict)
    fold_histogram: dict = field(default_factory=dict)

    def add(self, stats: dict) -> None:
        self.sweeps_dealiased += 1
        self.gates_offered += int(stats["gates_finite"])
        self.gates_unchanged += int(stats["gates_unchanged"])
        self.gates_unfolded += int(stats["gates_unfolded"])
        self.gates_rejected += int(stats["gates_rejected"])
        engine = str(stats.get("engine", ENGINE_VAD_REGION))
        self.engine = engine if self.engine in ("", engine) else "mixed"
        for name in _GRAPH_COUNTERS:
            if name not in stats:
                self.graph_counters_reported = False
                continue
            setattr(self, name, getattr(self, name) + int(stats[name]))
        reference = stats.get("reference") or {}
        self.reference_bands += int(reference.get("bands", 0))
        self.reference_bands_valid += int(reference.get("bands_valid", 0))
        for reason, count in stats["rejected"].items():
            self.rejected[reason] = self.rejected.get(reason, 0) + int(count)
        for fold, count in stats["fold_histogram"].items():
            key = str(int(fold))
            self.fold_histogram[key] = self.fold_histogram.get(key, 0) + int(count)

    def to_payload(self) -> dict:
        payload = {key: value for key, value in asdict(self).items()
                   if not isinstance(value, dict)}
        if not self.graph_counters_reported:
            for name in _GRAPH_COUNTERS:
                payload[name] = None
        payload["rejected"] = dict(sorted(self.rejected.items()))
        payload["fold_histogram"] = {
            key: self.fold_histogram[key]
            for key in sorted(self.fold_histogram, key=int)}
        payload["accounting_balances"] = bool(
            self.gates_unchanged + self.gates_unfolded + self.gates_rejected
            == self.gates_offered)
        return payload


def _dealias_velocity_sweep(sweep, nyquist, dealias_params, site,
                            velocity_reference, params) -> dict:
    """Unfold one sweep's velocity and package what the caller needs.

    The returned ``velocity`` plane carries the unfolded value where the
    unfolder resolved the gate and the **raw** value where it did not, so
    downstream finiteness bookkeeping is untouched; ``resolved`` is the mask
    that actually decides what may be assimilated.
    """

    moment = sweep.moments[VELOCITY]
    raw = np.asarray(moment.data, dtype=np.float64)
    reference = None
    if (velocity_reference is not None
            and dealias_params.engine == ENGINE_VAD_REGION):
        ranges = moment.slant_range_m()
        azimuth = np.broadcast_to(sweep.azimuth_deg[:, None], raw.shape)
        elevation = np.broadcast_to(sweep.elevation_deg[:, None], raw.shape)
        _lat, _lon, height, *_ = gate_locations(
            site.lat_deg, site.lon_deg, site.alt_m, azimuth,
            np.broadcast_to(ranges[None, :], raw.shape), elevation,
            earth_radius_m=params.earth_radius_m,
            refraction_factor=params.refraction_factor)
        reference = velocity_reference.radial_reference(
            azimuth, elevation, height - site.alt_m)

    # The clock from development branch around the call as
    # development branch left it: the region engine needs the gate
    # geometry, and the instrument must time the call the front door
    # actually makes, not the one that predates it.
    with perf_timing.stage("obs.dealias.sweep_total", gates=int(raw.size)):
        result = dealias_sweep(
            raw, sweep.azimuth_deg, nyquist, dealias_params,
            reference=reference,
            first_gate_m=moment.first_gate_range_m,
            gate_spacing_m=moment.gate_size_m,
            nyquist_by_radial=_believable_nyquist_by_radial(sweep, params),
            nyquist_radials_disagree=bool(sweep.nyquist_radials_disagree))
    resolved = result.state != STATE_REJECTED
    velocity = np.where(resolved, result.velocity, raw)
    # ``band_fits`` is the raw harmonic coefficients the volume-profile pass
    # consumes -- hundreds of entries per sweep, some carrying a NaN residual
    # where a band was taken from the profile rather than fitted.  It is
    # working state, not provenance: it would bloat the attribute and the
    # JSON writer refuses NaN outright, which is how it announced itself.
    reference = {key: value
                 for key, value in (result.stats.get("reference") or {}).items()
                 if key != "band_fits"}
    record = {
        "sweep_index": int(sweep.sweep_index),
        "elevation_angle_deg": float(sweep.elevation_angle_deg),
        **{key: value for key, value in result.stats.items()
           if key not in ("fold_histogram", "reference")},
        "reference": reference,
        "fold_histogram": {str(int(k)): int(v)
                           for k, v in sorted(result.stats["fold_histogram"].items())},
    }
    return {"result": result, "velocity": velocity, "resolved": resolved,
            "record": record}


def _believable_nyquist_by_radial(sweep, params):
    """The sweep's per-radial Nyquist, screened radial by radial.

    Returns None when the pack carried no such array -- which is not the
    same as "all radials agreed", and the unfolder is told which it is so a
    legacy nonuniform pack fails closed instead of being unfolded in the
    minimum's interval.  A radial whose value falls outside the plausible
    band becomes NaN and its gates are refused individually, exactly as the
    scalar screen refuses a whole sweep.
    """

    values = getattr(sweep, "nyquist_velocity_ms_by_radial", None)
    if values is None:
        return None
    values = np.asarray(values, dtype=np.float64)
    believable = (np.isfinite(values)
                  & (values >= params.nyquist_min_ms)
                  & (values <= params.nyquist_max_ms))
    return np.where(believable, values, np.nan)


def _believable_nyquist(reported, params) -> float | None:
    """The sweep's Nyquist velocity, or None when it cannot be believed."""

    if reported is None:
        return None
    value = float(reported)
    if not np.isfinite(value):
        return None
    if not (params.nyquist_min_ms <= value <= params.nyquist_max_ms):
        return None
    return value


@dataclass(frozen=True)
class SuperobParams:
    """Every tunable, in one hashable place, it goes into provenance."""

    #: Drop a velocity gate whose speed exceeds this fraction of Nyquist.
    nyquist_reject_fraction: float = 0.8
    #: Physically plausible Nyquist velocities for an S-band weather radar,
    #: m/s.  A reported value outside this band is metadata this stage does
    #: not believe -- a mis-parsed field, a non-WSR-88D convention -- and
    #: an unbelieved Nyquist masks every velocity in its sweep rather than
    #: licensing a threshold nothing supports.  The WSR-88D range runs from
    #: about 8 m/s on the slowest surveillance PRF to about 35 m/s on the
    #: fastest Doppler cut; the band is wide enough to admit other radars
    #: and narrow enough to reject a reinterpreted calibration constant.
    nyquist_min_ms: float = 4.0
    nyquist_max_ms: float = 100.0
    #: Drop a cell whose retained velocities span more than this fraction
    #: of Nyquist (a fold inside one cell).
    nyquist_spread_fraction: float = 0.5
    #: Gate-to-gate radial shear: a range-adjacent pair whose velocities
    #: differ by more than this fraction of the **full Nyquist interval**
    #: ``2 * Vn`` is a fold boundary.  A single fold produces a jump of
    #: almost exactly ``2 * Vn``, so 0.75 sits well below that and well
    #: above what a pair can reach once both gates have survived the 0.8
    #: magnitude gate (at most ``1.6 * Vn``, and only for a pair straddling
    #: the Nyquist limits in opposite directions -- which is the fold
    #: signature again).  Raising it toward 1.0 flags only near-perfect
    #: wraps; lowering it starts costing real tornadic gate-to-gate shear,
    #: which at close range is the signal this whole lane exists to carry.
    shear_fold_fraction: float = 0.75
    #: Reflectivity floor, dBZ: gates below are "no echo", not observations.
    min_reflectivity_dbz: float = -15.0
    #: Ignore gates beyond this slant range, km.
    max_range_km: float = 250.0
    #: Ignore sweeps above this antenna elevation, degrees.
    max_elevation_deg: float = 20.0
    #: Base observation-error standard deviations.
    z_error_base_dbz: float = 5.0
    vr_error_base_ms: float = 2.0
    #: Error floors: a cell with many gates must not claim implausible skill.
    z_error_floor_dbz: float = 2.0
    vr_error_floor_ms: float = 1.0
    #: Effective-earth multiplier for beam propagation.
    refraction_factor: float = REFRACTION_FACTOR
    earth_radius_m: float = EARTH_RADIUS_M
    #: How many *finite* below-floor gates must land in a cell before it is
    #: reported as observed clear air.  Stored as a float because every
    #: field of this dataclass is (``__post_init__`` normalizes the lot);
    #: it is used as a ``>=`` threshold against an integer gate count.
    #:
    #: This is not a smoothing knob.  One below-floor gate in a cell that
    #: the beam otherwise clipped is a geometry accident; requiring several
    #: independent gates to agree is what makes "the radar looked here and
    #: measured no significant return" a statement about the cell rather
    #: than about one range bin at its corner.
    clear_air_min_gates: float = 4.0
    #: Observation-error standard deviation for a clear-air zero, dBZ.
    #:
    #: Deliberately NOT ``z_error_base_dbz``.  A zero is a different
    #: measurement with a different error budget: it carries no in-cell
    #: variance to estimate from, its representativeness error is dominated
    #: by partial beam filling (a cell the beam only clipped can be clear
    #: where the beam looked and stormy where it did not), and the
    #: consequence of believing it too hard is erasing real convection.
    #: WoFS-family systems assign clear-air reflectivity a markedly larger
    #: sigma_o than echo for exactly this reason, and this default follows
    #: that practice rather than inheriting the echo error by omission.
    clear_air_error_dbz: float = 7.5
    #: Region-based velocity dealiasing.  ON by default; ``None`` is the
    #: masking-only behaviour this stage used to have, and is now something
    #: a caller states.
    #:
    #: The masks find SIGNATURES of aliasing, not aliasing.  A spatially
    #: coherent fold covering a whole region passes all four of them, so a
    #: default that masked and did not unfold published a smooth, plausible,
    #: wrong wind field and handed it to the filter.  Off was therefore
    #: never the identity it was documented as: it was a decision, made by
    #: omission, for every run that did not know to ask.
    #:
    #: The DEFAULT is resolved against this install as it is built
    #: (:func:`woof.obs.dealias.resolve_default_dealias`): the shipped
    #: region-global engine, else the scipy one, else masking only, warned
    #: once and recorded in the file's own ``superob_params`` and
    #: ``dealiasing`` statement.  A ``DealiasParams`` a caller PASSES is
    #: honoured exactly and is never resolved away: the resolution lives in
    #: the default factory, so it can only ever apply to a request nobody
    #: made by name.
    dealias: DealiasParams | None = field(
        default_factory=lambda: resolve_default_dealias(DealiasParams()))
    #: Correlation-coefficient QC, or ``None`` for the behaviour this
    #: stage has always had.  ``None`` is not merely the default, it is
    #: the *identity*: ``to_payload`` omits the key entirely while this
    #: is None, so the observation file's ``superob_params`` attribute is
    #: unchanged to the byte and a consumer reading a file written
    #: without CC QC cannot tell the field was ever added.  Turning the
    #: mask on is a decision someone makes; off is not a decision at all.
    cc_qc: CcQcParams | None = None

    def __post_init__(self) -> None:
        self.validate()
        # Normalize the runtime type once, at the only point where this
        # object is being built rather than merely used: every field is a
        # Python ``float`` from here on, so ``params.max_range_km * 1000.0``
        # and ``to_payload`` mean the same thing whether the caller passed
        # ``250``, ``np.float32(250)`` or ``250.0``.  ``validate`` above has
        # already refused anything that is not a real number, so this
        # normalizes what is sound and repairs nothing that is not.
        for field_ in fields(self):
            if field_.name in _NON_FLOAT_FIELDS:
                continue
            object.__setattr__(self, field_.name,
                               float(getattr(self, field_.name)))

    def _check_runtime_types(self) -> None:
        """Refuse a field whose runtime type is not a real number.

        ``validate`` used to read every field as ``float(getattr(...))``
        and keep the original object, so
        ``SuperobParams(max_range_km="250", nyquist_reject_fraction=True)``
        constructed successfully: ``float("250")`` is 250.0 and
        ``float(True)`` is 1.0, both of which pass every range check, and
        the string then raised ``TypeError`` at the first arithmetic use
        (``params.max_range_km * 1000.0``).  Failing loudly at first use is
        better than producing observations, but the parameter validator
        that the pipeline calls at every entry point was not validating its
        own runtime schema -- and the Rust side refuses these outright, so
        the Python surface was accepting what its counterpart would not.

        ``bool`` is excluded explicitly because ``True`` is an ``int`` in
        Python: ``nyquist_reject_fraction=True`` is 1.0, an in-range
        fraction, and it means the caller passed a flag where a threshold
        belongs.
        """

        for field_ in fields(self):
            if field_.name in _NON_FLOAT_FIELDS:
                continue
            value = getattr(self, field_.name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise SuperobParamsError(
                    f"{field_.name} is {value!r} ({type(value).__name__}); "
                    "every superob parameter is a real number. A numeric "
                    "string passes float() and then raises TypeError at the "
                    "first arithmetic use, and a bool passes every range "
                    "check as 0.0 or 1.0 -- neither is a value this stage "
                    "will convert on the caller's behalf, because the "
                    "conversion is the caller stating what they meant")
            try:
                float(value)
            except (OverflowError, TypeError, ValueError) as exc:
                raise SuperobParamsError(
                    f"{field_.name} is {value!r} ({type(value).__name__}) "
                    "but cannot be represented as a Python float; every "
                    "superob parameter is stored and consumed as a float, "
                    "so this is not a usable real-number value") from exc

    def validate(self) -> "SuperobParams":
        """Refuse any value that cannot do the job the field is named for.

        Called from ``__post_init__`` *and* from every entry point that
        consumes a parameter set, because construction is not the only way
        one arrives.  ``dataclasses.replace`` re-runs ``__init__`` and is
        covered by the former; ``object.__setattr__`` past the ``frozen=True``
        guard, an instance rebuilt from a JSON payload by a future reader, or
        a subclass that overrides a default are not.  The values are read
        back off ``self`` here rather than trusted from construction time, so
        the check is against what the pass is about to use.

        Returns ``self`` so a caller can write ``params.validate()`` inline.
        """

        # Types before ranges: every range check below reads the field
        # through float(), which is exactly how a numeric string and a bool
        # got past this function.
        self._check_runtime_types()
        for name in _FRACTION_FIELDS:
            value = float(getattr(self, name))
            if not np.isfinite(value) or not (0.0 <= value <= 1.0):
                raise SuperobParamsError(
                    f"{name} is {value!r}; it is a fraction of a Nyquist "
                    "velocity or Nyquist interval and must lie in [0, 1]. "
                    "Above 1 the gate admits speeds the sweep cannot "
                    "unambiguously measure -- the aliased data it exists to "
                    "drop; below 0 it is not an interval")
        for name in _POSITIVE_FIELDS:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise SuperobParamsError(
                    f"{name} is {value!r}; it must be finite and strictly "
                    "positive. Zero or negative makes the quantity it bounds "
                    "or scales meaningless, and for the error fields a zero "
                    "standard deviation is an infinitely confident "
                    "observation rather than a small uncertainty")
        for name in _FINITE_FIELDS:
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise SuperobParamsError(
                    f"{name} is {value!r}; it must be a finite number. A NaN "
                    "threshold compares False against every gate and drops "
                    "the volume in silence")
        if not self.nyquist_min_ms < self.nyquist_max_ms:
            raise SuperobParamsError(
                f"nyquist_min_ms {self.nyquist_min_ms!r} is not below "
                f"nyquist_max_ms {self.nyquist_max_ms!r}; the pair is the "
                "plausible band a reported Nyquist velocity is believed "
                "inside, and a band that is empty or a single point believes "
                "nothing, so every velocity in every sweep would be masked "
                "for an implausible Nyquist")
        min_gates = float(self.clear_air_min_gates)
        if min_gates < 1.0:
            raise SuperobParamsError(
                f"clear_air_min_gates is {min_gates!r}; it counts gates and "
                "must be at least 1. Below 1 every cell the beam never "
                "reached would satisfy the threshold with its zero count, "
                "turning 'no data' into 'observed clear' across the whole "
                "grid -- which is the one failure this product must never "
                "have")
        elevation = float(self.max_elevation_deg)
        if not np.isfinite(elevation) or not (0.0 < elevation <= 90.0):
            raise SuperobParamsError(
                f"max_elevation_deg is {elevation!r}; an antenna elevation "
                "ceiling must lie in (0, 90]. At or below 0 no sweep is ever "
                "used; above 90 the ceiling is not an elevation and cannot "
                "exclude anything")
        if self.dealias is not None:
            if not isinstance(self.dealias, DealiasParams):
                raise SuperobParamsError(
                    f"dealias is {self.dealias!r} "
                    f"({type(self.dealias).__name__}); it is either None -- "
                    "masking only, this stage's original behaviour -- or a "
                    "DealiasParams. A dict or a bool here would be a caller "
                    "asking for dealiasing without saying how it should "
                    "abstain, and the abstention rules are the safety")
            try:
                self.dealias.validate()
            except DealiasParamsError as error:
                raise SuperobParamsError(
                    f"dealias parameters are unusable: {error}") from error
        if self.cc_qc is not None:
            if not isinstance(self.cc_qc, CcQcParams):
                raise SuperobParamsError(
                    f"cc_qc is {self.cc_qc!r} "
                    f"({type(self.cc_qc).__name__}); it is either None -- "
                    "no correlation-coefficient QC -- or a CcQcParams, "
                    "because asking for the mask without saying its "
                    "thresholds is not a configuration")
            try:
                self.cc_qc.validate()
            except CcQcParamsError as error:
                raise SuperobParamsError(
                    f"cc_qc parameters are unusable: {error}") from error
        return self

    def to_payload(self) -> dict:
        payload = {key: float(value)
                   for key, value in asdict(self).items()
                   if key not in _NON_FLOAT_FIELDS}
        # Each optional parameter block is present only when it was
        # configured.  A file written without either carries the same key
        # set it always has, which is what makes the disabled paths
        # byte-identical rather than merely equivalent.
        if self.dealias is not None:
            payload["dealias"] = self.dealias.to_payload()
        if self.cc_qc is not None:
            payload["cc_qc"] = self.cc_qc.to_payload()
        return payload


@dataclass
class CensorCounts:
    """The gate census a ``v2`` pack's censor planes make possible.

    Only reflectivity is broken out, because reflectivity is the only
    moment this stage draws clear air from.  Velocity's censor plane is read
    and checked but never consulted for an observation, so counting it here
    would suggest an influence it does not have.
    """

    #: Reflectivity gates the decoder called measured, below threshold,
    #: range folded, and never collected, before any geometry filter.
    reflectivity_measured: int = 0
    reflectivity_below_threshold: int = 0
    reflectivity_range_folded: int = 0
    reflectivity_not_collected: int = 0
    #: Below-threshold gates that survived every geometry filter and were
    #: counted toward a cell's clear-air support.
    clear_air_gates_admitted: int = 0
    #: Range-folded gates seen anywhere in the pass.  Always equal to
    #: ``reflectivity_range_folded`` plus the velocity plane's, and always
    #: entirely refused: this number exists so the refusal is visible in
    #: provenance rather than merely asserted in a docstring.
    range_folded_gates_refused: int = 0
    #: The two ODIM-only states, counted on reflectivity, and the total of
    #: state 5 across every moment.  ``None`` on a NEXRAD pack, which cannot
    #: mint either code: an all-zero record would say "we looked and there
    #: were none", which is a different statement from "this schema has no
    #: such state", and ``to_payload`` omits them so a v2 pack's provenance
    #: -- and therefore every committed observation digest -- is byte-for-byte
    #: what it was before this field existed.
    #:
    #: Their purpose is arithmetic, not decoration.  Without them an ODIM
    #: census does not add up: measured + below_threshold + range_folded +
    #: not_collected falls short of the gate count by exactly the gates that
    #: were refused, and a census that silently does not close is how a
    #: refusal becomes invisible.
    reflectivity_nodata: int | None = None
    reflectivity_sentinel_ambiguous: int | None = None
    sentinel_ambiguous_gates_refused: int | None = None


@dataclass
class SuperobCounts:
    """What the pass did and, more importantly, what it refused."""

    gates_considered: int = 0
    gates_out_of_grid: int = 0
    gates_out_of_column: int = 0
    gates_below_floor: int = 0
    gates_nonfinite: int = 0
    gates_beyond_range: int = 0
    sweeps_used: int = 0
    sweeps_skipped_elevation: int = 0
    velocity_gates_rejected_nyquist: int = 0
    velocity_gates_rejected_no_nyquist: int = 0
    velocity_cells_rejected_spread: int = 0
    sweeps_without_nyquist: int = 0
    sweeps_with_implausible_nyquist: int = 0
    sweeps_with_nyquist_disagreement: int = 0
    #: Gate-to-gate shear scan.  ``pairs_tested`` is the denominator these
    #: only mean anything against: five boundaries in a million pairs and
    #: five in fifty are very different volumes.
    velocity_gate_pairs_tested: int = 0
    velocity_fold_boundaries: int = 0
    velocity_radials_fold_suspect: int = 0
    velocity_sweeps_fold_suspect: int = 0
    velocity_gates_rejected_shear: int = 0
    #: The censor census, or ``None`` when the pack carried no censor
    #: planes.  ``None`` rather than an all-zero record because the two are
    #: different statements, and because it is what keeps
    #: :meth:`to_payload` -- and therefore every observation file's
    #: ``provenance`` attribute, and therefore every committed obs digest --
    #: byte-identical to what it was before this field existed.
    censor: CensorCounts | None = None
    #: Correlation-coefficient QC (:mod:`woof.obs.cc_qc`).  All zero
    #: unless ``SuperobParams.cc_qc`` is set.  A CC-dropped gate is
    #: NaN-ed before the geometry pass, so it is *also* counted in
    #: ``gates_nonfinite`` downstream, by construction: the mask removes
    #: it the same way the decoder removes a censored gate.  The
    #: ``without`` counters are pass-open provenance, never failures --
    #: a sweep with no usable RhoHV source is 2012 data, not bad data.
    cc_sweeps_masked: int = 0
    cc_sweeps_paired_companion: int = 0
    cc_sweeps_without_rho: int = 0
    cc_sweeps_without_ref: int = 0
    cc_gates_tested: int = 0
    cc_gates_rho_missing: int = 0
    cc_velocity_gates_rejected: int = 0
    cc_reflectivity_gates_rejected: int = 0
    #: Of ``cc_velocity_gates_rejected``, how many sat at or above
    #: ``ref_shield_dbz``: the velocity the reflectivity shield would
    #: have kept, and therefore the measured price of the purity ruling
    #: on this volume.  Debris, hail cores and bright-band velocity land
    #: here.  It is a cost, not a defect, and it is counted so that
    #: revisiting the ruling is an argument about a number.
    cc_velocity_gates_rejected_shielded_z: int = 0
    #: The debris-fringe exemption (owner ruling 2026-08-12), which is
    #: on unless ``CcQcParams.tds_fringe_exempt`` is turned off.
    #: ``cc_velocity_gates_exempt_tds_fringe`` is the number of velocity
    #: gates it kept that the strict rule would have deleted -- 30-35
    #: dBZ, RhoHV between the debris floor and the velocity threshold,
    #: beside a clustered velocity couplet.  It is the whole footprint
    #: of the exemption, and a run with zero here is a run where the
    #: ruling changed nothing.  ``cc_couplet_seed_gates`` is how many
    #: clustered couplet seeds the volume's velocity planes produced:
    #: the exemption cannot fire anywhere in a volume with none, so a
    #: zero there explains a zero above.
    cc_velocity_gates_exempt_tds_fringe: int = 0
    cc_couplet_seed_gates: int = 0
    #: Of the low-RhoHV velocity gates the strict rule would drop, how
    #: many each conjunct turned away.  Disjoint, applied in order, and
    #: together with the exemption count they account for every such
    #: gate -- which is what lets a reader audit the ruling rather than
    #: take it on faith.
    cc_velocity_tds_rho_below_floor: int = 0
    cc_velocity_tds_below_reflectivity: int = 0
    cc_velocity_tds_at_or_above_shield: int = 0
    cc_velocity_tds_no_couplet_nearby: int = 0

    def to_payload(self) -> dict:
        payload = {key: int(value) for key, value in asdict(self).items()
                   if key != "censor"}
        if self.censor is not None:
            payload["censor"] = {key: int(value) for key, value
                                 in asdict(self.censor).items()
                                 if value is not None}
        return payload


@dataclass
class RadarContribution:
    """One radar's gridded contribution, before the multi-radar merge."""

    site_id: str
    lat_deg: float
    lon_deg: float
    alt_m: float
    valid_time: str
    z_linear_sum: np.ndarray
    z_count: np.ndarray
    #: Per cell, the number of gates that were *measured* and came back
    #: below ``min_reflectivity_dbz``.  See :func:`superob_volume` for the
    #: four conditions a gate must already have satisfied to be counted.
    z0_count: np.ndarray
    z_max_dbz: np.ndarray
    z_sumsq_dbz: np.ndarray
    z_sum_dbz: np.ndarray
    vr_sum: np.ndarray
    vr_sumsq: np.ndarray
    vr_count: np.ndarray
    vr_min: np.ndarray
    vr_max: np.ndarray
    beam_east: np.ndarray
    beam_north: np.ndarray
    beam_up: np.ndarray
    nyquist_min: np.ndarray
    vr_rejected: np.ndarray
    counts: SuperobCounts = field(default_factory=SuperobCounts)
    provenance: dict = field(default_factory=dict)
    #: Which of :data:`CLEAR_AIR_SOURCES` produced ``z0_count``.  Carried
    #: per radar because the merge has to refuse a mixture: two radars whose
    #: zeroes mean different things cannot be summed into one count.
    clear_air_source: str = CLEAR_AIR_SOURCE
    #: One record per velocity-carrying sweep, whether or not anything was
    #: flagged: an absence of fold boundaries is evidence too, and only
    #: means something beside the number of pairs that were tested.
    fold_suspicion: list = field(default_factory=list)
    #: The dealiasing account -- three states, every rejection with a reason
    #: -- or empty when dealiasing did not run.  Empty is how a consumer
    #: tells "no folds were found" from "nobody looked", which the counters
    #: alone cannot say.
    dealias: dict = field(default_factory=dict)
    #: The correlation-coefficient QC account -- per-sweep records with
    #: the RhoHV source each sweep used -- or empty when the mask was
    #: not configured.  Empty is how a consumer knows it never ran.
    cc_qc: dict = field(default_factory=dict)
    #: Origin of the arrays above in the analysis grid's horizontal index
    #: space.  The arrays cover ``[j0 : j0 + shape[1]]`` by
    #: ``[i0 : i0 + shape[2]]`` and every level; outside that box this
    #: radar contributed nothing, which is why the merge can leave the
    #: rest of the domain at its identity rather than store it.
    #:
    #: Defaulting to the origin keeps a hand-built full-domain
    #: contribution -- a test, a downstream lane -- valid and unchanged:
    #: a window at (0, 0) whose arrays span the grid IS the dense case.
    j0: int = 0
    i0: int = 0
    #: When this radar's volume was scanned and when it became available.
    #: ``start_time`` and ``end_time`` are the volume's first and last
    #: radial collection instants (``valid_time`` above is its header start
    #: to the second); ``availability_time`` is when the feed published it,
    #: the archive object's LastModified, set by the acquisition stage that
    #: alone knows it.  ``None`` where the pack or the feed did not say.
    start_time: str | None = None
    end_time: str | None = None
    availability_time: str | None = None

    @property
    def window(self) -> tuple[int, int, int, int]:
        """Inclusive ``(j0, j1, i0, i1)`` this contribution covers."""
        return (self.j0, self.j0 + self.z_count.shape[1] - 1,
                self.i0, self.i0 + self.z_count.shape[2] - 1)

    def window_payload(self) -> dict:
        """The window as a receipt records it."""
        j0, j1, i0, i1 = self.window
        cells = self.z_count.size
        return {"j0": j0, "j1": j1, "i0": i0, "i1": i1,
                "nz": int(self.z_count.shape[0]),
                "nj": j1 - j0 + 1, "ni": i1 - i0 + 1,
                "cells": int(cells)}


#: Extra cells kept around the reach disc.  A gate is assigned to the
#: NEAREST mass point, which can sit up to half a cell diagonal beyond the
#: gate itself, and a projected grid's cell size varies across the domain.
#: Two cells is far more than either effect needs and costs nothing against
#: a disc some 160 cells across; the guard in `superob_volume` is what makes
#: an under-estimate impossible rather than merely unlikely.
_WINDOW_MARGIN_CELLS = 2


def horizontal_window(grid: TargetGrid, site, params: SuperobParams):
    """The inclusive ``(j0, j1, i0, i1)`` box of cells this radar can reach.

    Computed by the Rust superob (``rw-superob``) by default; the numpy
    body below is the parity reference, run when ``WOOF_SUPEROB_PYTHON=1``
    says so.  See :func:`_horizontal_window_reference` for the rule.
    """

    from woof.obs import superob_bridge                  # noqa: PLC0415

    lib = superob_bridge.route()
    if lib is None:
        return _horizontal_window_reference(grid, site, params)
    return superob_bridge.horizontal_window(lib, grid, site, params)


def _horizontal_window_reference(grid: TargetGrid, site,
                                 params: SuperobParams):
    """The inclusive ``(j0, j1, i0, i1)`` box of cells this radar can reach.

    Measured against the grid's own mass-point coordinates rather than
    derived from the projection, so it holds for any georeference the
    superob stage accepts -- including one whose cell size varies across
    the domain, where a radius in cells would not.

    A radar entirely off the grid yields a degenerate one-cell window.
    That is not a special case downstream: no gate can land in it, every
    accumulator stays at its identity, and the contribution merges as the
    nothing it is.
    """

    reach_m = params.max_range_km * 1000.0 + _WINDOW_MARGIN_CELLS * max(
        float(grid.dx_m), float(grid.dy_m))
    lat = np.radians(np.asarray(grid.lat, dtype=np.float64))
    lon = np.radians(np.asarray(grid.lon, dtype=np.float64))
    site_lat = math.radians(float(site.lat_deg))
    site_lon = math.radians(float(site.lon_deg))
    # Haversine, matching the discovery stage's metric so "in range here"
    # and "in range there" cannot disagree about the same antenna.
    a = (np.sin((lat - site_lat) / 2.0) ** 2
         + np.cos(site_lat) * np.cos(lat)
         * np.sin((lon - site_lon) / 2.0) ** 2)
    distance = 2.0 * float(params.earth_radius_m) * np.arcsin(
        np.sqrt(np.clip(a, 0.0, 1.0)))
    near = distance <= reach_m
    if not near.any():
        return (0, 0, 0, 0)
    rows = np.flatnonzero(near.any(axis=1))
    cols = np.flatnonzero(near.any(axis=0))
    return (int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1]))


def _superob_prelude(volume, params, clear_air_from_censor,
                     velocity_reference):
    """What both routes check and derive before gridding a volume.

    Returns ``(params, censor_counts, velocity_reference)``: the validated
    parameters, the armed census (or None), and the VAD engine's anchor.
    """

    params = (params or SuperobParams()).validate()
    censor_counts: CensorCounts | None = None
    if clear_air_from_censor:
        missing = [
            f"sweep {sweep.sweep_index} {name}"
            for sweep in volume.sweeps
            for name, moment in sweep.moments.items()
            if name == REFLECTIVITY and moment.censor is None
        ]
        if missing:
            raise ValueError(
                "clear_air_from_censor needs a pack whose reflectivity "
                "carries censor planes, and "
                f"{volume.pack_path.name} (schema {volume.pack_schema}) does "
                f"not: {missing[0]}"
                + (f" and {len(missing) - 1} more" if len(missing) > 1 else "")
                + ". Re-decode the volume with `rw_nexrad decode "
                "--censor-flags`. Falling back to the measurement-only "
                "regime here would publish a thin product under the "
                "censored regime's clear_air_source, which claims a "
                "coverage it does not have")
        censor_counts = CensorCounts()
        if volume.pack_schema == SWEEPS_SCHEMA_ODIM:
            # Arm the ODIM-only counters at zero. Zero and absent are
            # different statements and the pack schema is what decides
            # which one is accurate here: a v3 pack can mint 4 and 5, so
            # "none were seen" is a measurement; a v2 pack cannot, so the
            # keys stay out of its provenance entirely.
            censor_counts.reflectivity_nodata = 0
            censor_counts.reflectivity_sentinel_ambiguous = 0
            censor_counts.sentinel_ambiguous_gates_refused = 0

    dealias_params = params.dealias
    # The volume-wide wind profile is the VAD engine's anchor and nothing
    # else's.  Building it for the region-global engine would fit a
    # harmonic to every range band of every cut -- the most expensive stage
    # in this module -- and then hand the result to a solver that has no
    # reference input to put it in.
    profile_engine = (dealias_params is not None
                      and dealias_params.engine == ENGINE_VAD_REGION)
    if velocity_reference is not None and dealias_params is not None \
            and not profile_engine:
        raise ValueError(
            f"velocity_reference was supplied with dealias engine "
            f"{dealias_params.engine!r}, which carries no environmental "
            f"reference; only {ENGINE_VAD_REGION!r} anchors against one. "
            "Silently ignoring it would produce velocities that never saw "
            "the background wind under provenance that says a background "
            "was supplied")
    if profile_engine and velocity_reference is None:
        # Derive the anchor from the volume itself before unfolding any of
        # it.  A per-sweep fit is one range band's view of one height; the
        # volume crossed most heights several times, from different
        # elevations at different ranges, and pooling those is the only
        # cross-check available without an external model field.  Measured
        # on the real case this is what stops a sparse far-range band from
        # anchoring thousands of gates to a wind no other sweep saw.
        with perf_timing.stage("obs.dealias.volume_profile"):
            velocity_reference = volume_wind_profile(
                ((sweep.elevation_angle_deg,
                  sweep.moments[VELOCITY].data,
                  sweep.azimuth_deg,
                  _believable_nyquist(sweep.nyquist_velocity_ms, params),
                  sweep.moments[VELOCITY].slant_range_m())
                 for sweep in volume.sweeps
                 if VELOCITY in sweep.moments
                 and sweep.elevation_angle_deg <= params.max_elevation_deg),
                dealias_params)
    return params, censor_counts, velocity_reference


def superob_volume(volume: RadarVolume, grid: TargetGrid, *,
                   params: SuperobParams | None = None,
                   clear_air_from_censor: bool = False,
                   velocity_reference=None) -> RadarContribution:
    """Grid one radar volume onto ``grid``.

    Accumulators only: the dBZ/velocity/error reduction happens once, in
    :func:`merge_contributions`, so a multi-radar product and a single-radar
    product go through exactly the same arithmetic.

    ``clear_air_from_censor`` selects the censored regime described in the
    module docstring.  It is off by default and the default path is
    unchanged, arithmetic included.  Asking for it against a pack that has
    no censor planes is a hard error rather than a silent downgrade: the
    caller asked for a coverage this volume cannot supply, and quietly
    returning the thin product under the wrong ``clear_air_source`` is the
    one outcome that would mislead the DA side.

    It is a keyword rather than a :class:`SuperobParams` field on purpose.
    ``SuperobParams.to_payload`` is serialized verbatim into every
    observation file's ``superob_params`` attribute, so a new field there
    would change the bytes of files that are otherwise identical -- and the
    regime is already recorded, exactly once and where a consumer looks for
    it, as ``clear_air_source``.

    ``velocity_reference`` is an optional
    :class:`woof.obs.dealias.WindProfile` -- the model background wind --
    used only when ``params.dealias`` is set, and used only to *supplement*
    the volume's own VAD: it seeds the harmonic fit and fills the range
    bands the fit could not qualify.  It is never allowed to override a
    band the volume itself resolved, because the volume measured the wind
    and the background guessed it.

    It belongs to the ``vad-region`` engine alone.  The region-global
    engine has no environmental reference in it, so supplying one beside
    that engine is refused rather than ignored: a caller who handed a
    background wind to a solver that never read it would get a run whose
    provenance says a treatment was applied and whose velocities were
    produced without it.
    """

    from woof.obs import superob_bridge                  # noqa: PLC0415

    # Refuses here, before any work, when the Rust library is missing and
    # nobody opted into the numpy reference.
    lib = superob_bridge.route()
    params, censor_counts, velocity_reference = _superob_prelude(
        volume, params, clear_air_from_censor, velocity_reference)
    dealias_params = params.dealias
    if lib is not None:
        return _superob_volume_rust(
            lib, volume, grid, params=params, censor_counts=censor_counts,
            clear_air_from_censor=clear_air_from_censor,
            velocity_reference=velocity_reference)
    counts = SuperobCounts()

    # A radar is a local instrument on a domain that need not be.  Its
    # gates occupy a disc ~250 km across; a continental analysis grid is
    # some 5000 km across, so full-domain accumulators spend 99% of their
    # bytes holding zeros this antenna can never write to.  At 120 bytes
    # per cell per site that is 10.9 GB per radar on a CONUS 3 km grid,
    # which is the difference between a regional system and a continental
    # one.  So the accumulators cover only the cells this radar can reach.
    #
    # Vertical is left whole: a beam sweeps every level within its range,
    # and nz is already small.  All the sparsity is horizontal.
    j0, j1, i0, i1 = horizontal_window(grid, volume.site, params)
    shape = (grid.nz, j1 - j0 + 1, i1 - i0 + 1)
    zeros = lambda: np.zeros(shape, dtype=np.float64)     # noqa: E731

    z_linear_sum = zeros()
    z_sum_dbz = zeros()
    z_sumsq_dbz = zeros()
    z_count = np.zeros(shape, dtype=np.int64)
    z0_count = np.zeros(shape, dtype=np.int64)
    z_max_dbz = np.full(shape, -np.inf, dtype=np.float64)
    vr_sum = zeros()
    vr_sumsq = zeros()
    vr_count = np.zeros(shape, dtype=np.int64)
    vr_min = np.full(shape, np.inf, dtype=np.float64)
    vr_max = np.full(shape, -np.inf, dtype=np.float64)
    beam_east = zeros()
    beam_north = zeros()
    beam_up = zeros()
    nyquist_min = np.full(shape, np.inf, dtype=np.float64)
    vr_rejected = np.zeros(shape, dtype=np.int64)

    fold_suspicion: list[dict] = []
    #: Moment names this pass declined to read, kept for the refusal below.
    skipped_products: set[str] = set()
    counts.censor = censor_counts
    dealias_records: list[dict] = []
    dealias_totals = _DealiasTotals()

    # CC QC plans span the whole volume because split-cut pairing needs
    # the neighbours: a sweep skipped below (elevation) can still lend
    # its RHO plane to the cut beside it.
    cc_plans = None
    cc_sweep_records: list[dict] = []
    if params.cc_qc is not None:
        cc_plans = build_cc_plans(volume.sweeps, params.cc_qc)

    site = volume.site
    max_range_m = params.max_range_km * 1000.0
    # Which regime's zeroes these are, decided by the pack that produced
    # them rather than by the caller: the flag says "use the decoder's own
    # gate codes", and which vocabulary those codes belong to is a property
    # of the file.
    clear_air_source = _clear_air_source(volume, clear_air_from_censor)

    grid_timing = perf_timing.phases("obs.superob")
    grid_timing.mark("sweep_loop", sweeps=len(volume.sweeps))
    # `enumerate`, because CC QC's split-cut pairing indexes the plan
    # table by sweep position.
    for sweep_position, sweep in enumerate(volume.sweeps):
        if sweep.elevation_angle_deg > params.max_elevation_deg:
            counts.sweeps_skipped_elevation += 1
            continue
        nyquist = _believable_nyquist(sweep.nyquist_velocity_ms, params)

        # Dealiasing runs once per sweep, on the whole cut, before any
        # blocking or range masking.  Regions span radial blocks and range
        # limits; unfolding a slice at a time would cut every region at the
        # block seam and turn continuity -- the evidence the method rests on
        # -- into an artifact of a memory-management constant.
        dealiased = None
        if dealias_params is not None and VELOCITY in sweep.moments:
            dealiased = _dealias_velocity_sweep(
                sweep, nyquist, dealias_params, site,
                velocity_reference, params)
            dealias_records.append(dealiased["record"])
            dealias_totals.add(dealiased["result"].stats)
        if sweep.nyquist_velocity_ms is None:
            counts.sweeps_without_nyquist += 1
        elif nyquist is None:
            counts.sweeps_with_implausible_nyquist += 1
        if sweep.nyquist_radials_disagree:
            counts.sweeps_with_nyquist_disagreement += 1
        counts.sweeps_used += 1
        sweep_pairs = 0
        sweep_boundaries = 0
        sweep_suspect_radials = 0
        sweep_shear_rejected = 0
        cc_plan = None
        cc_masker = None
        if cc_plans is not None:
            cc_plan = cc_plans[sweep_position]
            cc_masker = cc_plan.masker
            if cc_masker is not None:
                counts.cc_sweeps_masked += 1
                if cc_plan.reason == REASON_COMPANION:
                    counts.cc_sweeps_paired_companion += 1
            elif cc_plan.reason == REASON_NO_RHO:
                counts.cc_sweeps_without_rho += 1
            elif cc_plan.reason == REASON_NO_REF:
                counts.cc_sweeps_without_ref += 1

        for product, moment in sweep.moments.items():
            if product not in (REFLECTIVITY, VELOCITY):
                skipped_products.add(str(product))
                continue
            ranges = moment.slant_range_m()                     # (gates,)
            in_range = ranges <= max_range_m
            counts.gates_beyond_range += int(
                (~in_range).sum()) * sweep.radial_count
            if not np.any(in_range):
                continue
            ranges = ranges[in_range]
            # CC QC evaluates against this moment's own gate ranges: REF
            # and VEL do not owe each other a gate count, and a plane is
            # never masked by another plane's index.
            cc_drop = (None if cc_masker is None
                       else cc_masker.drop_mask(product, ranges))

            # Radial blocks bound peak memory: a super-res sweep is 720 x
            # 1832 gates and the geometry pass holds a dozen float64
            # temporaries per gate, which is gigabytes if done in one go.
            for start in range(0, sweep.radial_count, _RADIAL_BLOCK):
                stop = min(start + _RADIAL_BLOCK, sweep.radial_count)
                values = np.asarray(moment.data[start:stop][:, in_range],
                                    dtype=np.float64)
                # The decoder's reason for each NaN, when the pack carried
                # one.  Read for both moments so the range-folded refusal
                # can be counted accurately, consulted for clear air only on
                # reflectivity.
                codes = (None if moment.censor is None or censor_counts is None
                         else moment.censor[start:stop][:, in_range])
                clear_flag = np.zeros(values.shape, dtype=bool)
                if codes is not None:
                    folded = codes == Censor.RANGE_FOLDED
                    censor_counts.range_folded_gates_refused += int(
                        folded.sum())
                    if censor_counts.sentinel_ambiguous_gates_refused \
                            is not None:
                        # Counted on every moment, like range-folded, and
                        # refused on every one: a gate whose file could not
                        # tell "nothing there" from "did not look" is not an
                        # observation of either.
                        censor_counts.sentinel_ambiguous_gates_refused += int(
                            (codes == Censor.SENTINEL_AMBIGUOUS).sum())
                    if product == REFLECTIVITY:
                        # Equality with ONE code, never "not an echo".  This
                        # is the line that keeps range-folded gates out of
                        # the clear-air path: code 2 is not code 1, and no
                        # setting of any parameter makes it so.
                        clear_flag = codes == Censor.BELOW_THRESHOLD
                        censor_counts.reflectivity_measured += int(
                            (codes == Censor.MEASURED).sum())
                        censor_counts.reflectivity_below_threshold += int(
                            clear_flag.sum())
                        censor_counts.reflectivity_range_folded += int(
                            folded.sum())
                        censor_counts.reflectivity_not_collected += int(
                            (codes == Censor.NOT_COLLECTED).sum())
                        if censor_counts.reflectivity_nodata is not None:
                            censor_counts.reflectivity_nodata += int(
                                (codes == Censor.NODATA).sum())
                            censor_counts.reflectivity_sentinel_ambiguous \
                                += int((codes
                                        == Censor.SENTINEL_AMBIGUOUS).sum())
                # Where dealiasing ran, the velocities from here down are the
                # unfolded ones -- including the shear scan, which now sees a
                # field whose folds have been removed and so measures what
                # the unfolder missed rather than what it was asked to fix.
                # A gate the unfolder rejected keeps its RAW value and is
                # excluded by `resolved` instead, so `finite` and
                # gates_nonfinite still count exactly what they always
                # counted: missing data, not refused data.
                resolved = None
                if dealiased is not None and product == VELOCITY:
                    values = dealiased["velocity"][start:stop][:, in_range]
                    resolved = dealiased["resolved"][start:stop][:, in_range]
                # CC QC before the shear scan sees the block:
                # non-meteorological echo is not fold evidence, and a
                # NaN-ed gate breaks the gate-to-gate chain exactly the
                # way a decoder-censored gate does.  Only finite gates
                # count as dropped -- masking a hole is not a rejection.
                #
                # ORDERING, and it is a merge decision worth stating.  The
                # CC lane was written against a stage that had no
                # dealiaser, and its docstring places this mask "upstream
                # of alias reasoning".  It cannot be, here: the unfolder
                # runs once per sweep, above this block loop, on the raw
                # velocity plane.  So the mask is applied AFTER the
                # unfolded values are substituted in, not before.
                #
                # What that does and does not change, and read this
                # before moving anything: the drop decision DOES read
                # velocity -- the TDS-fringe exemption's couplet detector
                # is handed a velocity plane (`CcSweepMasker(...,
                # velocity=...)` in cc_qc.py).  What makes the ordering
                # harmless is not that velocity goes unread; it is that
                # `build_cc_plans` runs ONCE, above this block loop, over
                # `volume.sweeps`, so every input to the decision --
                # RhoHV, reflectivity and that couplet scan alike -- reads
                # the RAW plane no matter what happens down here.  Move
                # the plan build INTO this loop and the exemption silently
                # starts reasoning about dealiased velocity.  As it
                # stands, WHICH gates are dropped is identical either way
                # and the counters below stay accurate.  Running the mask
                # before the substitution would have been strictly worse:
                # the NaNs would be overwritten by
                # `dealiased["velocity"]` on the very next line while the
                # counters still claimed the drops.  What is genuinely
                # lost is that the unfolder's fold reasoning saw gates
                # this mask would have removed.  Owner ruling 2026-08-12
                # (the 2.1 assembly): this ordering is ACCEPTED for 2.1,
                # and masking the raw planes before the unfolder is to be
                # settled by an A/B -- couplet instrument plus a control
                # region -- before any restructure.
                if cc_drop is not None:
                    finite_before = np.isfinite(values)
                    # The exemption is invisible in the drop counts by
                    # construction -- the gate it saved is the gate that
                    # is no longer there to count -- so account for it
                    # on every block, dropped gates or not.
                    exempt = cc_masker.count_block(
                        product, finite_before, start)
                    if product == VELOCITY:
                        counts.cc_velocity_gates_exempt_tds_fringe += exempt
                    cc_hits = cc_drop[start:stop] & finite_before
                    if np.any(cc_hits):
                        values = np.array(values, dtype=np.float64, copy=True)
                        values[cc_hits] = np.nan
                        dropped = cc_masker.count_dropped(
                            product, cc_hits, start)
                        if product == VELOCITY:
                            counts.cc_velocity_gates_rejected += dropped
                        else:
                            counts.cc_reflectivity_gates_rejected += dropped
                # The shear scan runs here, on raw range-ordered gates,
                # because this is the last point at which a fold and its
                # unfolded neighbour still differ by the Nyquist interval.
                if product == VELOCITY and nyquist is not None:
                    fold_flags, boundaries, pairs = _fold_boundaries(
                        values, nyquist, params)
                    sweep_pairs += pairs
                    sweep_boundaries += boundaries
                    sweep_suspect_radials += int(
                        fold_flags.any(axis=1).sum())
                else:
                    fold_flags = np.zeros(values.shape, dtype=bool)
                azimuth = sweep.azimuth_deg[start:stop, None]
                elevation = sweep.elevation_deg[start:stop, None]
                lat, lon, height, east, north, up = gate_locations(
                    site.lat_deg, site.lon_deg, site.alt_m,
                    np.broadcast_to(azimuth, values.shape),
                    np.broadcast_to(ranges[None, :], values.shape),
                    np.broadcast_to(elevation, values.shape),
                    earth_radius_m=params.earth_radius_m,
                    refraction_factor=params.refraction_factor)

                finite = np.isfinite(values).ravel()
                counts.gates_considered += int(values.size)
                counts.gates_nonfinite += int((~finite).sum())

                # A gate is worth placing if it is a measurement OR if the
                # decoder said the radar looked here and found nothing.  In
                # the measurement-only regime ``clear_flag`` is all false
                # and this is exactly ``finite``, which is why every count
                # below is unchanged there.
                usable = finite | clear_flag.ravel()

                i_frac, j_frac = grid.mass_index(lat.ravel(), lon.ravel())
                i_index = np.rint(i_frac).astype(np.intp)
                j_index = np.rint(j_frac).astype(np.intp)
                on_grid = grid.inside(i_index, j_index) & usable
                counts.gates_out_of_grid += int(usable.sum() - on_grid.sum())
                if not np.any(on_grid):
                    continue

                clear_on_grid = clear_flag.ravel()[on_grid]
                i_index = i_index[on_grid]
                j_index = j_index[on_grid]
                level = grid.level_index(i_index, j_index,
                                         height.ravel()[on_grid])
                in_column = level >= 0
                counts.gates_out_of_column += int((~in_column).sum())
                if not np.any(in_column):
                    continue

                # Into WINDOW coordinates.  The global indices above are
                # what `grid.level_index` needs (column height is a
                # property of the column, not of the window), so the
                # offset is applied here and nowhere else -- this single
                # ravel is the only place the accumulators' shape enters.
                #
                # Every value accumulated below is therefore the same
                # float, applied in the same order, to the same logical
                # cell; only its address changed.  That is why the
                # windowed contribution is bit-identical to the dense one
                # rather than merely close: `np.add.at` sees an unchanged
                # sequence of additions.
                j_local = j_index[in_column] - j0
                i_local = i_index[in_column] - i0
                # Four reductions rather than four boolean temporaries and
                # their ORs: this runs per sweep, per moment, per radial
                # block, and the array form cost ~20% of the whole pass for
                # a check that is never expected to fire.
                if (j_local.min() < 0 or j_local.max() >= shape[1]
                        or i_local.min() < 0 or i_local.max() >= shape[2]):
                    # Unreachable unless the window is wrong, and a wrong
                    # window would silently drop observations.  Fail loudly
                    # instead: this is an accuracy bug, not a bad night.
                    raise SuperobError(
                        f"{volume.site.id}: gates fell outside the computed "
                        f"reach window j[{j0}..{j1}] i[{i0}..{i1}] "
                        f"(saw j[{int(j_local.min()) + j0}.."
                        f"{int(j_local.max()) + j0}] "
                        f"i[{int(i_local.min()) + i0}.."
                        f"{int(i_local.max()) + i0}]) -- the window "
                        "under-covers this radar's gates and observations "
                        "would have been lost.  This is a bug in "
                        "horizontal_window(), not a data problem.")
                flat = np.ravel_multi_index(
                    (level[in_column], j_local, i_local), shape)
                value_sel = values.ravel()[on_grid][in_column]
                clear_sel = clear_on_grid[in_column]

                if product == REFLECTIVITY:
                    # A below-threshold gate is NaN, and NaN >= x is False,
                    # so it lands in ``~echo`` without a special case --
                    # which is the point: it is below the floor by the
                    # radar's own report rather than by our arithmetic.
                    echo = value_sel >= params.min_reflectivity_dbz
                    counts.gates_below_floor += int((~echo & ~clear_sel).sum())
                    if censor_counts is not None:
                        censor_counts.clear_air_gates_admitted += int(
                            clear_sel.sum())
                    # --- clear air: the radar looked here and measured
                    # nothing significant ---
                    #
                    # A gate reaches this line only after four independent
                    # conditions, and every one of them is essential for
                    # the claim "observed clear" rather than "no data":
                    #
                    # 1. ACCOUNTED FOR.  ``on_grid`` above ANDs in
                    #    ``usable``, which is ``finite`` plus -- only in the
                    #    censored regime -- gates the decoder explicitly
                    #    marked below threshold.  A gate that is NaN for any
                    #    OTHER reason never arrives: not a range-folded
                    #    gate (raw 1, which may be a storm), not a gate on a
                    #    radial that never carried the moment.  In the
                    #    measurement-only regime the set is just ``finite``,
                    #    because there NaN is irreducibly ambiguous and is
                    #    never evidence of anything.
                    # 2. WITHIN RANGE.  ``in_range`` trimmed the far end, so
                    #    a cell past ``max_range_km`` accumulates nothing.
                    # 3. ON GRID and IN COLUMN.  The gate was placed in a
                    #    real model cell by the same geometry the echo
                    #    observations use; a cell no beam traverses -- below
                    #    the lowest tilt, behind terrain, outside the scan --
                    #    is never named here at all, and so ends the pass
                    #    with a zero count rather than a clear-air claim.
                    # 4. BELOW THE FLOOR.  The measured value is a real
                    #    number that is smaller than the significant-echo
                    #    threshold.
                    #
                    # Note what is *not* asserted: this counts gates, not
                    # cells.  Whether the cell as a whole is clear is decided
                    # in ``merge_contributions``, where the count meets the
                    # echo count and the minimum-gate threshold, because a
                    # cell containing one clear gate and one echo gate is a
                    # cell with echo in it.
                    flat_clear = flat[~echo]
                    if flat_clear.size:
                        np.add.at(z0_count.reshape(-1), flat_clear, 1)
                    flat_z = flat[echo]
                    dbz = value_sel[echo]
                    if flat_z.size:
                        np.add.at(z_linear_sum.reshape(-1), flat_z,
                                  np.power(10.0, dbz / 10.0))
                        np.add.at(z_sum_dbz.reshape(-1), flat_z, dbz)
                        np.add.at(z_sumsq_dbz.reshape(-1), flat_z, dbz * dbz)
                        np.add.at(z_count.reshape(-1), flat_z, 1)
                        np.maximum.at(z_max_dbz.reshape(-1), flat_z, dbz)
                    continue

                # --- radial velocity: fail closed on aliasing signatures ---
                if nyquist is None:
                    counts.velocity_gates_rejected_no_nyquist += int(flat.size)
                    np.add.at(vr_rejected.reshape(-1), flat, 1)
                    continue
                # A gate whose fold state the unfolder could not establish is
                # dropped here whatever its magnitude.  It is dropped BEFORE
                # the magnitude test rather than after so the two losses can
                # never be confused in the counters: one is "too fast to
                # trust", the other is "I could not tell", and conflating
                # them would hide exactly the number this capability has to
                # be judged on.
                # `vr_rejected` is incremented once, below, off `~keep` --
                # which already excludes these, since `within` is masked by
                # `keep_resolved`.  Adding them here too would count a
                # refused gate twice in the array a consumer reads as "how
                # many gates were dropped over this cell".
                if resolved is not None:
                    keep_resolved = resolved.ravel()[on_grid][in_column]
                    dealias_totals.gates_refused_at_grid += int(
                        (~keep_resolved).sum())
                else:
                    keep_resolved = np.ones(flat.size, dtype=bool)
                if (resolved is not None
                        and dealias_params.keep_beyond_reject_fraction):
                    # The 0.8 rule drops gates that MIGHT be folded.  These
                    # were resolved, so what remains to bound them is
                    # physics, not the Nyquist interval -- and this is where
                    # the couplet that the 0.8 rule removes comes back.
                    within = np.abs(value_sel) <= dealias_params.max_speed_ms
                else:
                    within = (np.abs(value_sel)
                              <= params.nyquist_reject_fraction * nyquist)
                within = within & keep_resolved
                counts.velocity_gates_rejected_nyquist += int(
                    (keep_resolved & ~within).sum())
                # A gate flanking a fold boundary is dropped even when its
                # own magnitude is unremarkable: that is the whole point of
                # the scan, since a folded gate's magnitude is by definition
                # small.  Counted separately so the two losses never blur.
                flanking = fold_flags.ravel()[on_grid][in_column]
                shear_only = within & flanking
                counts.velocity_gates_rejected_shear += int(shear_only.sum())
                sweep_shear_rejected += int(shear_only.sum())
                keep = within & ~flanking
                if np.any(~keep):
                    np.add.at(vr_rejected.reshape(-1), flat[~keep], 1)
                if not np.any(keep):
                    continue
                flat_v = flat[keep]
                speed = value_sel[keep]
                east_sel = east.ravel()[on_grid][in_column][keep]
                north_sel = north.ravel()[on_grid][in_column][keep]
                up_sel = up.ravel()[on_grid][in_column][keep]
                np.add.at(vr_sum.reshape(-1), flat_v, speed)
                np.add.at(vr_sumsq.reshape(-1), flat_v, speed * speed)
                np.add.at(vr_count.reshape(-1), flat_v, 1)
                np.minimum.at(vr_min.reshape(-1), flat_v, speed)
                np.maximum.at(vr_max.reshape(-1), flat_v, speed)
                np.add.at(beam_east.reshape(-1), flat_v, east_sel)
                np.add.at(beam_north.reshape(-1), flat_v, north_sel)
                np.add.at(beam_up.reshape(-1), flat_v, up_sel)
                np.minimum.at(nyquist_min.reshape(-1), flat_v, float(nyquist))

        if VELOCITY in sweep.moments:
            counts.velocity_gate_pairs_tested += sweep_pairs
            counts.velocity_fold_boundaries += sweep_boundaries
            counts.velocity_radials_fold_suspect += sweep_suspect_radials
            if sweep_boundaries:
                counts.velocity_sweeps_fold_suspect += 1
            fold_suspicion.append({
                "sweep_index": int(sweep.sweep_index),
                "elevation_angle_deg": float(sweep.elevation_angle_deg),
                "nyquist_ms": None if nyquist is None else float(nyquist),
                "nyquist_radials_disagree": bool(
                    sweep.nyquist_radials_disagree),
                "radial_count": int(sweep.radial_count),
                "gate_pairs_tested": sweep_pairs,
                "fold_boundaries": sweep_boundaries,
                "radials_fold_suspect": sweep_suspect_radials,
                "gates_rejected_shear": sweep_shear_rejected,
            })

        if cc_plan is not None:
            if cc_masker is not None:
                counts.cc_gates_tested += sum(
                    cc_masker.gates_tested.values())
                counts.cc_gates_rho_missing += sum(
                    cc_masker.gates_rho_missing.values())
                counts.cc_velocity_gates_rejected_shielded_z += (
                    cc_masker.gates_dropped_shielded_z.get(VELOCITY, 0))
                counts.cc_couplet_seed_gates += cc_masker.couplet_seed_gates
                turned = cc_masker.gates_tds_turned_away.get(VELOCITY, {})
                counts.cc_velocity_tds_rho_below_floor += turned.get(
                    TDS_TURNED_AWAY_RHO_BELOW_FLOOR, 0)
                counts.cc_velocity_tds_below_reflectivity += turned.get(
                    TDS_TURNED_AWAY_BELOW_Z_BAND, 0)
                counts.cc_velocity_tds_at_or_above_shield += turned.get(
                    TDS_TURNED_AWAY_ABOVE_Z_BAND, 0)
                counts.cc_velocity_tds_no_couplet_nearby += turned.get(
                    TDS_TURNED_AWAY_NO_ROTATION, 0)
            cc_sweep_records.append(cc_plan.record(sweep))

    # A cell whose retained velocities span too much of the Nyquist interval
    # is a fold caught inside one cell: drop it whole rather than average
    # across the wrap.
    grid_timing.mark("reduce")
    spread = np.where(vr_count > 0, vr_max - vr_min, 0.0)
    folded = ((vr_count > 1)
              & np.isfinite(nyquist_min)
              & (spread > params.nyquist_spread_fraction * nyquist_min))
    counts.velocity_cells_rejected_spread = int(folded.sum())

    # A volume whose sweeps were all used and whose gates were all skipped is
    # not a thin product, it is a mismatch: this pass and the pack that wrote
    # it do not agree on what a moment is called.  It happened, on the first
    # real European volume ever put through here -- `rw_odim` wrote ODIM's own
    # `DBZH`/`VRADH` where this filter tests for `REF`/`VEL`, so every one of
    # 8.5 million gates was skipped, `sweeps_used` still counted fifteen, the
    # observation file wrote successfully, and the record it printed was
    # entirely zeros.  Nothing failed; there was simply nothing in it.
    #
    # So the contradiction is now a refusal.  It names what the pack called
    # its moments, because that is the whole diagnosis and looking for it
    # otherwise means reading a binary pack by hand.
    _require_considered_gates(volume, counts, skipped_products)
    if np.any(folded):
        vr_rejected[folded] += vr_count[folded]
        for array in (vr_sum, vr_sumsq, beam_east, beam_north, beam_up):
            array[folded] = 0.0
        vr_count[folded] = 0

    grid_timing.close()
    return RadarContribution(
        site_id=site.id, lat_deg=site.lat_deg, lon_deg=site.lon_deg,
        alt_m=site.alt_m, valid_time=volume.valid_time,
        z_linear_sum=z_linear_sum, z_count=z_count, z0_count=z0_count,
        z_max_dbz=z_max_dbz,
        z_sumsq_dbz=z_sumsq_dbz, z_sum_dbz=z_sum_dbz,
        vr_sum=vr_sum, vr_sumsq=vr_sumsq, vr_count=vr_count,
        vr_min=vr_min, vr_max=vr_max,
        beam_east=beam_east, beam_north=beam_north, beam_up=beam_up,
        nyquist_min=nyquist_min, vr_rejected=vr_rejected,
        counts=counts, provenance=volume.provenance(),
        clear_air_source=clear_air_source,
        fold_suspicion=fold_suspicion, j0=j0, i0=i0,
        start_time=volume.start_time, end_time=volume.end_time,
        cc_qc=({} if cc_plans is None else {
            "params": params.cc_qc.to_payload(),
            "sweeps": cc_sweep_records,
        }),
        dealias=({} if dealias_params is None else {
            "params": dealias_params.to_payload(),
            "totals": dealias_totals.to_payload(),
            "sweeps": dealias_records,
        }))


def _superob_volume_rust(lib, volume: RadarVolume, grid: TargetGrid, *,
                         params: SuperobParams, censor_counts,
                         clear_air_from_censor: bool,
                         velocity_reference) -> RadarContribution:
    """:func:`superob_volume` through the Rust library (the default).

    The orchestration stays here: the dealiaser is driven sweep by sweep as
    the reference drives it (its region-global engine itself runs inside
    the same library), the reach window comes from
    :func:`horizontal_window`, and the account the library returns is
    assembled into the same provenance records.  Every gate-level
    operation -- geometry, placement, CC QC, census, the alias masks and
    the accumulators -- runs in ``rw-superob``.
    """

    from woof.obs import superob_bridge                  # noqa: PLC0415

    dealias_params = params.dealias
    site = volume.site
    dealiased: dict[int, tuple] = {}
    dealias_records: list[dict] = []
    dealias_totals = _DealiasTotals()
    if dealias_params is not None:
        # Whole sweeps, before any blocking or range masking, in sweep
        # order: exactly where the reference unfolds them.
        for position, sweep in enumerate(volume.sweeps):
            if sweep.elevation_angle_deg > params.max_elevation_deg:
                continue
            if VELOCITY not in sweep.moments:
                continue
            nyquist = _believable_nyquist(sweep.nyquist_velocity_ms, params)
            unfolded = _dealias_velocity_sweep(
                sweep, nyquist, dealias_params, site, velocity_reference,
                params)
            dealias_records.append(unfolded["record"])
            dealias_totals.add(unfolded["result"].stats)
            dealiased[position] = (unfolded["velocity"], unfolded["resolved"])

    window = horizontal_window(grid, site, params)
    odim = (censor_counts is not None
            and censor_counts.reflectivity_nodata is not None)
    with perf_timing.stage("obs.superob.rust", sweeps=len(volume.sweeps)):
        answer, acc = superob_bridge.superob_volume(
            lib, volume, grid, params, window=window,
            clear_air_from_censor=clear_air_from_censor, odim_census=odim,
            dealiased=dealiased)

    counts = SuperobCounts(**answer["counts"])
    if censor_counts is not None:
        counts.censor = CensorCounts(**answer["censor"])
    _require_considered_gates(volume, counts, answer.get("skipped_products") or [])
    dealias_totals.gates_refused_at_grid += int(answer["gates_refused_at_grid"])

    fold_suspicion: list[dict] = []
    cc_sweep_records: list[dict] = []
    for record in answer["sweeps"]:
        sweep = volume.sweeps[int(record["position"])]
        nyquist = _believable_nyquist(sweep.nyquist_velocity_ms, params)
        fold = record.get("fold")
        if fold is not None:
            fold_suspicion.append({
                "sweep_index": int(sweep.sweep_index),
                "elevation_angle_deg": float(sweep.elevation_angle_deg),
                "nyquist_ms": None if nyquist is None else float(nyquist),
                "nyquist_radials_disagree": bool(
                    sweep.nyquist_radials_disagree),
                "radial_count": int(sweep.radial_count),
                "gate_pairs_tested": int(fold["gate_pairs_tested"]),
                "fold_boundaries": int(fold["fold_boundaries"]),
                "radials_fold_suspect": int(fold["radials_fold_suspect"]),
                "gates_rejected_shear": int(fold["gates_rejected_shear"]),
            })
        if "cc" in record:
            cc_sweep_records.append({
                "sweep_index": int(sweep.sweep_index),
                "elevation_angle_deg": float(sweep.elevation_angle_deg),
                **record["cc"]})

    j0, _, i0, _ = window
    return RadarContribution(
        site_id=site.id, lat_deg=site.lat_deg, lon_deg=site.lon_deg,
        alt_m=site.alt_m, valid_time=volume.valid_time,
        **acc,
        counts=counts, provenance=volume.provenance(),
        clear_air_source=_clear_air_source(volume, clear_air_from_censor),
        fold_suspicion=fold_suspicion, j0=int(j0), i0=int(i0),
        start_time=volume.start_time, end_time=volume.end_time,
        cc_qc=({} if params.cc_qc is None else {
            "params": params.cc_qc.to_payload(),
            "sweeps": cc_sweep_records,
        }),
        dealias=({} if dealias_params is None else {
            "params": dealias_params.to_payload(),
            "totals": dealias_totals.to_payload(),
            "sweeps": dealias_records,
        }))


def _require_considered_gates(volume, counts, skipped_products):
    if counts.sweeps_used and not counts.gates_considered:
        detail = (f"; the pack's moments are named "
                  f"{sorted(skipped_products)!r} and this stage reads "
                  f"{[REFLECTIVITY, VELOCITY]!r}"
                  if skipped_products else "")
        raise ValueError(
            f"{volume.pack_path.name}: {counts.sweeps_used} sweeps were read "
            f"and not one gate was considered{detail}. An observation file "
            "built from this would be empty, well-formed and silent about "
            "it, so it is refused here instead")


def _clear_air_source(volume, clear_air_from_censor: bool) -> str:
    """Which regime's zeroes these are, decided by the pack's vocabulary."""
    if not clear_air_from_censor:
        return CLEAR_AIR_SOURCE
    if volume.pack_schema == SWEEPS_SCHEMA_ODIM:
        return CLEAR_AIR_SOURCE_ODIM
    return CLEAR_AIR_SOURCE_CENSOR


@dataclass
class GriddedObservations:
    """The reduced fields, one step short of NetCDF."""

    z_obs: np.ndarray
    z_mask: np.ndarray
    z_err: np.ndarray
    z_max: np.ndarray
    z_mean: np.ndarray
    z_count: np.ndarray
    vr_obs: np.ndarray
    vr_mask: np.ndarray
    vr_err: np.ndarray
    vr_count: np.ndarray
    vr_rejected: np.ndarray
    vr_beam_east: np.ndarray
    vr_beam_north: np.ndarray
    vr_beam_up: np.ndarray
    radars: list[dict]
    counts: list[dict]
    provenance: list[dict]
    #: Per-radar, per-sweep gate-to-gate shear scan records.  Defaulted so
    #: a caller assembling this structure by hand -- a test, a downstream
    #: lane -- is not forced to invent fold statistics it never measured.
    fold_suspicion: list = field(default_factory=list)
    #: ``||sum b_i|| / n`` per cell: how nearly the contributing beams pointed
    #: the same way.  1 for one beam or several parallel ones, and falling
    #: with the spread of look directions.  None on a structure assembled by
    #: hand before the field existed.
    vr_beam_coherence: np.ndarray | None = None
    #: Clear-air ("zero") observations, or None for a product that carries
    #: none.  ``z0_mask`` is 1 where at least one radar measured this cell
    #: and every radar that measured it found no significant echo;
    #: ``z0_count`` is the supporting measured-gate count and ``z0_err``
    #: the standard deviation to assimilate it with.
    #:
    #: ``None`` rather than an all-false mask is the default because the
    #: two are different statements.  An all-false mask asserts "this
    #: volume was examined for clear air and none was established"; None
    #: says "clear air was never assessed here".  A caller assembling this
    #: structure by hand has done the latter, and the writer omits the
    #: variables entirely rather than shipping a mask that claims an
    #: assessment nobody made.
    #:
    #: There is deliberately **no** ``z0_obs``: the dBZ value a zero
    #: differences against is the *forward operator's* clear-air floor
    #: (-35 dBZ for the refl10cm family, 0 dBZ for NSSL mp18, -99 dBZ
    #: for Milbrandt-Yau mp9), which is a
    #: property of the model the DA lane is running, not of the radar.
    #: Writing a value here would bake one scheme's floor into an
    #: observation file that outlives the run that consumed it, and a
    #: floor mismatch manufactures a 35, 64 or 99 dB innovation -- one
    #: per pair of those three floors -- out of two agreeing clear skies.
    #: The DA adapter supplies the value.
    z0_mask: np.ndarray | None = None
    z0_count: np.ndarray | None = None
    z0_err: np.ndarray | None = None
    #: Which of :data:`CLEAR_AIR_SOURCES` established the zeroes above.
    #: Meaningless when ``z0_mask`` is None, and defaulted to the
    #: measurement-only regime so a hand-assembled product reads as the
    #: conservative one.
    clear_air_source: str = CLEAR_AIR_SOURCE
    #: Per-radar dealiasing account -- params, three-state totals, per-sweep
    #: records -- empty when dealiasing did not run.  Empty is meaningful:
    #: it is how a consumer tells "no folds were found" from "nobody
    #: looked", which no counter can say on its own.
    dealias: list = field(default_factory=list)
    #: Per-radar correlation-coefficient QC accounts; each entry is empty
    #: when that radar's mask never ran.  Defaulted for the same reason.
    cc_qc: list = field(default_factory=list)
    #: Per radar, the inclusive ``(j0, j1, i0, i1)`` of the analysis grid
    #: its ``vr_*`` plane covers.  Reflectivity is merged ACROSS radars and
    #: stays whole-domain -- it is one field the whole network contributed
    #: to, and at 37 bytes a cell it is 3.4 GB on a CONUS grid, which is
    #: not the problem.  Radial velocity cannot merge, so it carries a
    #: leading radar axis, and THAT is the array that grew to 715 GB for a
    #: continental network at 49 bytes a cell a radar.
    #:
    #: Empty means whole-domain planes: the v1 layout, still valid, still
    #: what a hand-built object gets.
    radar_windows: list = field(default_factory=list)
    #: Which in-cell reduction ``z_obs`` carries (``"mean"`` or ``"max"``),
    #: or None on a hand-built set that never said.  The writer records it
    #: in the file's ``superob_params`` so ``z_obs`` is never ambiguous.
    z_reduce: str | None = None

    @property
    def windowed(self) -> bool:
        return bool(self.radar_windows)

    def radar_window(self, index: int) -> tuple[int, int, int, int]:
        """Inclusive ``(j0, j1, i0, i1)`` radar ``index`` covers."""
        if self.radar_windows:
            return tuple(self.radar_windows[index])
        return (0, self.vr_obs.shape[2] - 1, 0, self.vr_obs.shape[3] - 1)

    def radar_plane(self, name: str, index: int, *, ny: int, nx: int):
        """One radar's ``vr_*`` plane, expanded to the whole domain.

        The compatibility path, and deliberately one plane at a time: a
        consumer that wants the old dense view can have it without the
        file materialising all of them at once, which is the whole saving.
        Outside the window the value is the fill each field already used
        where a radar saw nothing -- zero for every one of them.
        """
        stored = getattr(self, name)
        plane = stored[index]
        if not self.radar_windows:
            return plane
        j0, j1, i0, i1 = self.radar_window(index)
        out = np.zeros((plane.shape[0], ny, nx), dtype=plane.dtype)
        out[:, j0:j1 + 1, i0:i1 + 1] = plane[:, :j1 - j0 + 1, :i1 - i0 + 1]
        return out


def merge_contributions(contributions, grid: TargetGrid, *,
                        params: SuperobParams | None = None,
                        z_reduce: str = "mean") -> GriddedObservations:
    """Reduce one or more radars' accumulators into the output fields.

    Reflectivity merges across radars (the maximum of the maxima, the
    count-weighted mean of the linear sums); radial velocity does not, and
    keeps a leading ``radar`` axis.

    ``z_reduce`` picks what ``z_obs`` carries; ``z_max`` and ``z_mean`` are
    always written beside it.  The default is the in-cell mean in linear Z,
    because the maximum biases the observations high and wide: measured on
    2026-10-05 it sat 6.6 dB above the mean and put 8.1 times MRMS's 35 dBZ
    area into the observations (woof.da.radar_classes), and the DA reads
    the mean by default.

    The reduction runs in the Rust library by default (``rw-superob``);
    the numpy body is the parity reference (``WOOF_SUPEROB_PYTHON=1``).
    Contributions are reduced in the order given, in both, so a merged
    cell is the same sum of the same addends on every run.
    """

    params = (params or SuperobParams()).validate()
    contributions = list(contributions)
    if not contributions:
        raise ValueError("no radar contributions to merge")
    if z_reduce not in ("max", "mean"):
        raise ValueError(
            f"z_reduce must be 'max' or 'mean', got {z_reduce!r}")
    shape = (grid.nz, grid.ny, grid.nx)
    for contribution in contributions:
        j0, j1, i0, i1 = contribution.window
        if (contribution.z_count.shape[0] != grid.nz
                or j0 < 0 or i0 < 0 or j1 >= grid.ny or i1 >= grid.nx):
            raise ValueError(
                f"contribution from {contribution.site_id} covers "
                f"levels 0..{contribution.z_count.shape[0] - 1}, "
                f"j {j0}..{j1}, i {i0}..{i1}, which does not fit a grid "
                f"of {shape}")

    # Zeroes from the two regimes cover different fractions of the domain,
    # so summing them would produce a count whose coverage nobody can state
    # and a single ``clear_air_source`` that would be a lie about half its
    # inputs.  Refuse rather than pick one.
    sources = {c.clear_air_source for c in contributions}
    if len(sources) > 1:
        raise ValueError(
            "the contributions establish clear air by different means "
            f"({sorted(sources)}), and their z0_counts cannot be summed: the "
            "two regimes cover different fractions of the domain, so the "
            "merged count would describe no coverage in particular. Rebuild "
            "every volume in the set the same way")
    clear_air_source = sources.pop()

    windows = [c.window for c in contributions]
    from woof.obs import superob_bridge                  # noqa: PLC0415

    lib = superob_bridge.route()
    if lib is None:
        out = _merge_arrays_reference(contributions, grid, params, z_reduce,
                                      shape)
    else:
        with perf_timing.stage("obs.superob.merge_rust",
                               radars=len(contributions)):
            out = superob_bridge.merge(lib, contributions, grid, params,
                                       z_reduce=z_reduce, windows=windows)

    radars = [{
        "id": contribution.site_id,
        "lat_deg": float(contribution.lat_deg),
        "lon_deg": float(contribution.lon_deg),
        "alt_m": float(contribution.alt_m),
        "valid_time": contribution.valid_time,
        # ``getattr`` for the reason the dealias account below uses it: a
        # contribution assembled by hand is not forced to invent instants
        # it never read, and None is "not stated", never the header start.
        "start_time": getattr(contribution, "start_time", None),
        "end_time": getattr(contribution, "end_time", None),
        "availability_time": getattr(contribution, "availability_time",
                                     None),
    } for contribution in contributions]

    return GriddedObservations(
        **out,
        clear_air_source=clear_air_source,
        z_reduce=z_reduce,
        radars=radars,
        counts=[c.counts.to_payload() for c in contributions],
        provenance=[c.provenance for c in contributions],
        fold_suspicion=[list(c.fold_suspicion) for c in contributions],
        dealias=[dict(c.dealias) for c in contributions
                 if getattr(c, "dealias", None)],
        # ``getattr`` for the same reason the line above uses it: a caller
        # assembling contributions by hand -- a test, a downstream lane --
        # is not forced to invent an account it never measured.  The list
        # stays per-radar aligned, so an entry is empty exactly when that
        # radar's mask did not run.
        cc_qc=[dict(getattr(c, "cc_qc", None) or {})
               for c in contributions],
        radar_windows=[list(w) for w in windows])


def _merge_arrays_reference(contributions, grid, params, z_reduce, shape):
    """The numpy reduction behind :func:`merge_contributions` (reference)."""

    # Compose the windows rather than adding full-domain arrays.  Outside
    # its own window a radar contributed the identity of every reduction
    # here -- 0 for the sums and counts, -inf for a maximum, +inf for a
    # minimum -- which is exactly what these accumulators are initialised
    # to.  Adding 0.0 and taking max(x, -inf) are both exact, and the
    # contributions are visited in the same order as before, so the merged
    # arrays are bit-identical to the dense path's, not merely close.
    def _slot(c):
        j0, j1, i0, i1 = c.window
        return (slice(None), slice(j0, j1 + 1), slice(i0, i1 + 1))

    z_linear = np.zeros(shape, dtype=np.float64)
    z_count = np.zeros(shape, dtype=np.int64)
    z0_count = np.zeros(shape, dtype=np.int64)
    z_sum_dbz = np.zeros(shape, dtype=np.float64)
    z_sumsq_dbz = np.zeros(shape, dtype=np.float64)
    z_max = np.full(shape, -np.inf, dtype=np.float64)
    for contribution in contributions:
        slot = _slot(contribution)
        z_linear[slot] += contribution.z_linear_sum
        z_count[slot] += contribution.z_count
        z0_count[slot] += contribution.z0_count
        z_sum_dbz[slot] += contribution.z_sum_dbz
        z_sumsq_dbz[slot] += contribution.z_sumsq_dbz
        np.maximum(z_max[slot], contribution.z_max_dbz, out=z_max[slot])

    has_z = z_count > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        z_mean = np.where(has_z,
                          10.0 * np.log10(np.maximum(z_linear, 1e-30)
                                          / np.maximum(z_count, 1)),
                          0.0)
        mean_dbz = np.where(has_z, z_sum_dbz / np.maximum(z_count, 1), 0.0)
        variance = np.where(
            z_count > 1,
            np.maximum(z_sumsq_dbz / np.maximum(z_count, 1)
                       - mean_dbz * mean_dbz, 0.0),
            0.0)
    z_max = np.where(has_z, z_max, 0.0)
    z_obs = np.where(has_z, z_max if z_reduce == "max" else z_mean, 0.0)
    z_err = np.where(
        has_z,
        np.maximum(
            params.z_error_floor_dbz,
            np.sqrt(params.z_error_base_dbz ** 2 / np.maximum(z_count, 1)
                    + variance)),
        0.0)
    z_mask = has_z.astype(np.int8)

    # --- clear-air zeroes -------------------------------------------------
    #
    # Two conditions, and the second is the one that keeps this accurate.
    #
    # ``z0_count >= clear_air_min_gates`` says enough gates independently
    # measured this cell and found nothing.  ``~has_z`` says *no* radar saw
    # echo here.  The second is a veto, not a tiebreak: reflectivity is
    # summed across radars above, so ``has_z`` is true if ANY contributing
    # radar found echo in the cell, and a cell one radar calls clear while
    # another sees a storm in it is not clear.  The nearer radar is usually
    # the one seeing the storm -- the other is looking through it, over it,
    # or at a range where its beam has broadened past the cell -- so
    # deferring to the echo is also the physically right call, not merely
    # the conservative one.
    #
    # What this cannot see, and what therefore belongs in ``z0_err`` rather
    # than in this mask:
    #
    # * ATTENUATION.  A cell behind a heavy core can measure genuinely
    #   below-floor because the signal never got back, not because the sky
    #   is empty.  No attenuation correction exists in this lane
    #   (``woof.da.obsop`` states the same for the forward operator), so
    #   this is real, unmodelled, and one-sided towards false clear air.
    # * PARTIAL BEAM FILLING.  At range the sampling volume is much larger
    #   than a model cell; "clear where the beam looked" and "clear
    #   throughout the cell" diverge with distance.
    # * BEAM BLOCKAGE.  A blocked ray returns clutter (counted as echo, so
    #   harmless here) or nothing (NaN, so never counted at all) -- but a
    #   *partially* blocked ray returns a weakened real echo that can fall
    #   below the floor.
    #
    # None of the three can be detected from the gridded product alone.
    # They are the reason ``clear_air_error_dbz`` is a separate, larger
    # sigma_o rather than an inherited one.
    has_z0 = (z0_count >= params.clear_air_min_gates) & ~has_z
    z0_mask = has_z0.astype(np.int8)
    z0_err = np.where(has_z0, params.clear_air_error_dbz, 0.0)

    n_radar = len(contributions)
    # The velocity planes cover each radar's own window, not the domain.
    # Every radar carries the same range authority, so the windows differ
    # only where the domain edge clips one; padding them all to the widest
    # keeps the array rectangular -- which netCDF wants and a ragged
    # layout would fight -- at a waste of a few edge cells rather than the
    # 99% a whole-domain plane wastes.
    windows = [c.window for c in contributions]
    max_nj = max(j1 - j0 + 1 for j0, j1, _, _ in windows)
    max_ni = max(i1 - i0 + 1 for _, _, i0, i1 in windows)
    vshape = (n_radar, grid.nz, max_nj, max_ni)
    vr_obs = np.zeros(vshape, dtype=np.float64)
    vr_err = np.zeros_like(vr_obs)
    vr_mask = np.zeros(vshape, dtype=np.int8)
    vr_count = np.zeros(vshape, dtype=np.int32)
    vr_rejected = np.zeros(vshape, dtype=np.int32)
    beam = [np.zeros(vshape, dtype=np.float64) for _ in range(3)]
    beam_coherence = np.zeros(vshape, dtype=np.float64)

    for index, contribution in enumerate(contributions):
        count = contribution.vr_count
        vectors = (contribution.beam_east, contribution.beam_north,
                   contribution.beam_up)
        norm = np.sqrt(sum(component ** 2 for component in vectors))
        # A cell whose contributing beams summed to nothing has no look
        # direction, and a radial velocity without one is not an
        # observation: Vr = u*east + v*north + w*up is undefined for a zero
        # vector.  Such a cell is masked and counted here rather than
        # shipped with a zero beam under a true mask, which would silently
        # zero every innovation it touched.
        has_vr = (count > 0) & (norm > 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = np.where(has_vr, contribution.vr_sum
                            / np.maximum(count, 1), 0.0)
            variance = np.where(
                count > 1,
                np.maximum(contribution.vr_sumsq / np.maximum(count, 1)
                           - mean * mean, 0.0),
                0.0)
        # ``y`` is the MEAN of n scalar radial velocities, so the operator
        # that reproduces it is the MEAN of the n beam unit vectors:
        #   mean_i(b_i . x) = (mean_i b_i) . x
        # holds exactly, for any x, and only for the unnormalized mean.
        # Normalising it -- as this did -- divides by ||sum b|| instead of by
        # n, which inflates H(x) by 1/c with c = ||sum b||/n <= 1.  Two beams
        # 60 degrees apart give c = 0.866 and so a 15.5% inflation of every
        # modelled radial velocity in that cell, for a perfectly uniform
        # wind, with no diagnostic anywhere that the observation and its
        # operator had stopped describing the same quantity.
        coherence = np.where(count > 0, norm / np.maximum(count, 1), 0.0)
        beamless = (count > 0) & ~has_vr
        # This radar's plane IS its window, so the contribution lands at
        # the plane's origin and no offset arithmetic is needed here --
        # the window travelled with the contribution.  Only the padding to
        # the widest window is skipped, and it stays zero, which is what
        # every expression below would have produced for it anyway.
        j0, j1, i0, i1 = contribution.window
        slot = (index, slice(None), slice(0, j1 - j0 + 1),
                slice(0, i1 - i0 + 1))
        vr_obs[slot] = np.where(has_vr, mean, 0.0)
        vr_count[slot] = np.where(has_vr, count, 0)
        vr_rejected[slot] = (contribution.vr_rejected
                             + np.where(beamless, count, 0))
        vr_mask[slot] = has_vr.astype(np.int8)
        vr_err[slot] = np.where(
            has_vr,
            np.maximum(
                params.vr_error_floor_ms,
                np.sqrt(params.vr_error_base_ms ** 2 / np.maximum(count, 1)
                        + variance)),
            0.0)
        for axis, component in enumerate(vectors):
            beam[axis][slot] = np.where(has_vr, component
                                         / np.where(count > 0, count, 1), 0.0)
        beam_coherence[slot] = np.where(has_vr, coherence, 0.0)
        # The policy hook.  A cell whose beams point over more than a
        # hemisphere's worth of directions is not measuring one projection of
        # one wind; the mean beam vector shrinks toward zero and the operator
        # it defines is dominated by whichever direction happened to
        # dominate the gate count.  The default floor is deliberately low --
        # it removes geometry that is broken rather than merely spread, so
        # turning it on changes almost nothing on a real volume -- and is
        # stated here rather than buried, because the number decides which
        # observations reach the filter.
        incoherent = has_vr & (coherence < MIN_BEAM_COHERENCE)
        if np.any(incoherent):
            vr_mask[slot] = np.where(incoherent, 0,
                                      vr_mask[slot]).astype(np.int8)
            vr_rejected[slot] = vr_rejected[slot] + np.where(
                incoherent, vr_count[slot], 0)
            vr_count[slot] = np.where(incoherent, 0, vr_count[slot])
            vr_obs[slot] = np.where(incoherent, 0.0, vr_obs[slot])
            vr_err[slot] = np.where(incoherent, 0.0, vr_err[slot])
            for axis in range(3):
                beam[axis][slot] = np.where(incoherent, 0.0,
                                             beam[axis][slot])

    return {
        "z_obs": z_obs, "z_mask": z_mask, "z_err": z_err, "z_max": z_max,
        "z_mean": z_mean, "z_count": z_count.astype(np.int32),
        "z0_mask": z0_mask, "z0_count": z0_count.astype(np.int32),
        "z0_err": z0_err, "vr_obs": vr_obs, "vr_mask": vr_mask,
        "vr_err": vr_err, "vr_count": vr_count, "vr_rejected": vr_rejected,
        "vr_beam_east": beam[0], "vr_beam_north": beam[1],
        "vr_beam_up": beam[2], "vr_beam_coherence": beam_coherence,
    }
