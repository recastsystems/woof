"""EXPERIMENTAL: the hydrometeor positivity policy the filter refuses to own.

:func:`woof.da.letkf.analyze` returns per-member **increments** and does not
clip them.  That is deliberate and it is right: a Gaussian filter applied to a
bounded, heavily zero-inflated variable will routinely propose a negative
mixing ratio, and the three sane responses -- clip, transform, reject -- are
not equivalent, differ in what they do to mass and to the ensemble's own
spread, and are properly the caller's choice.  A filter that clipped would be
making that choice silently for every caller forever.

So the caller must choose, and this module makes the choice explicit,
bounded, and *counted*.

**What ``clip`` actually does, stated accurately.**  Where ``prior + increment``
would be negative, the increment is replaced by ``-prior`` so the analysis is
exactly ``0``.  That is a change of ``-analysis > 0``: **clipping at zero adds
mass.**  It is not conservative and it is biased in one direction, always
wetward for a mixing ratio.  The bias is small when the filter is well tuned
and large when it is not, which makes the count of clipped points and the
mass added the single most useful diagnostic of whether the analysis is
healthy.  Both go into the receipt, per field, per member.  A cycle whose
clipped mass grows leg over leg is a cycle whose covariances are wrong, and
the manifest should be able to show that without a rerun.

**``mean-preserving``, the default every door uses.**  The clip is
one-signed: every member that undershoots zero is raised to it and no
member is lowered, so at every analysis the ensemble gains mass the filter
never proposed -- the positivity twin of the one-signed saturation clip
that dried the members (``woof.ensemble.increments
.mean_preserving_saturation_bound``).  This policy applies the same
mean-preserving rule to the analysed value itself, which must be
non-negative: at a cell where any member's analysis is negative, every
member's non-negative analysis is scaled by one factor so the ensemble
keeps the filter's own mean analysis, and the members that undershot end
exactly at zero.  The ensemble total is therefore the filter's, and mass
is added only where the filter's ensemble MEAN is itself negative (every
member then ends at zero, which is what the clip does).  Cells where no
member undershoots are not touched, bit for bit.  With one member (a
``(nz, ny, nx)`` array, no member axis) there is no ensemble mean to keep
and the rule IS the clip; the receipt says which happened.

**The alternatives, and why neither is the default.**

``anamorphosis``
    Assimilate a transformed variable -- ``log(q + q0)`` or a Gaussian
    anamorphosis against the ensemble's own empirical CDF -- so a negative
    analysis is unrepresentable rather than repaired.  It is the principled
    answer and it is not a policy that can live here: it changes what H(x)
    means, what the observation error means, and what the ensemble spread
    means, so it belongs in the forward operator and the filter's input, not
    in a post-hoc fixup.  The zero-inflation also has to be handled
    explicitly (``log 0`` is not a number and ``q0`` is a tuning knob nobody
    has tuned).
``reject``
    Discard the whole increment at any gridpoint where any constrained field
    would go negative, leaving that column at its background value.  It
    conserves the background exactly and adds no mass, but it introduces
    discontinuities between a rejected gridpoint and its analysed neighbour
    -- the model then has to absorb a gradient the filter invented -- and it
    throws away the correction to *unconstrained* fields at that point, which
    were not the problem.  Implemented here, because it is four lines and the
    comparison is worth being able to run; not the default.

The doors (``tools/da_cycle_prepared.py``, ``tools/da_nowcast.py``,
``woof local-da``, ``woof.ensemble.cycle``) default to
``mean-preserving``; ``clip``, ``reject`` and ``none`` stay selectable.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from woof.core.fast_sbm_contract import BIN_SPECIES as _FSBM_BIN_SPECIES

#: Provenance schema for the receipt this module produces.
POSITIVITY_SCHEMA = "gpuwm-da.positivity.v1"

#: Policies, by name.  ``none`` exists so a caller can state that it
#: considered the question and declined, which is different from a caller
#: that never thought about it -- the receipt records which.
POLICIES = ("mean-preserving", "clip", "reject", "none")

#: The policy every door uses when the caller states none.  The clip it
#: replaced added mass at every analysis, always wetward (module docstring).
DEFAULT_POLICY = "mean-preserving"

#: Policies after which no constrained analysis may be negative; the
#: callers' post-condition (:func:`verify_non_negative`) runs after these.
BOUNDING_POLICIES = ("mean-preserving", "clip", "reject")

#: State attributes that are physically non-negative and must never be
#: analysed below zero.  Taken from the restart prognostic contract
#: (``woof.io.restart.STATE_SERIALIZED_ATTRS``): mixing ratios, number
#: concentrations, and the two-moment volume variables.  Generic names
#: only -- nothing here knows about any case or any source.
NON_NEGATIVE_FIELDS = (
    "qv", "qc", "qr", "qi", "qs", "qg", "qh", "qndrop",
    "nc", "nr", "ni", "ns", "ng",
    "qnr", "qni", "qns", "qng", "qnh", "qnn",
    "qvolg", "qvolh",
    # The number concentrations the prognostic contract grew after this
    # list was first written, and the reason this line exists: a real
    # mp_physics=28 radar cycle analysed the aerosol-aware tracers with
    # the rest of the scheme's state, the filter put nwfa at
    # -1.005e8 kg^-1 in one cell, the policy had no opinion about the
    # field so it passed through untouched, and the next leg's health
    # check refused the state.  A concentration below zero is not a
    # state any of these schemes has: Thompson's own bounds on the pair
    # are MAX(11.1e6, MIN(9999e6, nwfa*rho)) and MAX(5.0e3, MIN(9999e6,
    # nifa*rho)) at module_mp_thompson.F:1805-1806 and :3980-3982, and
    # both floors are far above zero.
    "nwfa", "nifa",     # mp 28 water- and ice-friendly aerosol number
    "nh",               # mp 9 (Milbrandt-Yau) hail number
    "nn",               # mp 16 (WDM6) CCN number
    # mp 50 (P3) rime mass and rime volume, found by the same question
    # asked of the moment tables instead of of this list: they are
    # prognostic, the restart contract carries them, a full-moment
    # analysis moves them, and the sentence above already claimed to
    # cover "the two-moment volume variables" -- it reached NSSL's qvolg
    # and qvolh and stopped there.  The scheme's own authority settles
    # the sign: woof/core/p3.py:1310-1312 zeroes BOTH the moment the
    # rime mass goes negative, so a negative rime moment is not a state
    # P3 has.
    "qir", "qib",
    # TEMPO (WOOF mp 900) graupel volume, prognostic and carried by the
    # restart contract (STATE_SERIALIZED_ATTRS) like NSSL qvolg; the scheme
    # divides graupel mass by it (woof/core/kernels/tempo/tempo_diags.cuh:175),
    # so an analysis that drives it below zero hands the next step a state
    # TEMPO does not have.
    "volg",
    # mp 30 (FAST_SBM) spectral bins: ff1/ff5/ff6 hydrometeor mass per bin
    # (kg/kg) and ff8 aerosol number per bin, 142 prognostic fields the
    # restart contract carries and a full-moment analysis moves, and the
    # policy had no opinion about any of them.  A negative bin is not a
    # state the scheme has: its column kernel stops with a status code on
    # one (woof/core/kernels/fast_sbm_core.cuh, status 8610 for ff1r).
    *_FSBM_BIN_SPECIES,
    # mp 53 (P3 three-moment ice) advected moment (N*Z)**0.5: a square
    # root, so never negative (module_mp_p3.F:916-919 writes 0 or a root).
    "qzi",
    # The second and third free ice categories the restart contract carries
    # (woof/state_serialization_contract.py): mp 52 (P3 two-category) mass,
    # number, rime mass and rime volume, and mp 55 (ISHMAEL) category masses
    # and numbers.  Same reason as their first-category spellings above
    # (qi/ni, qir/qib): a mass, a count or a rime quantity below zero is not
    # a state either scheme has, and a full-moment analysis moves them.
    "qi2", "ni2", "qir2", "qib2", "qi3", "ni3",
    # The unpaired and volume moments the 3.0 ports put in the restart
    # contract, each a moment, mass, count, volume or fraction of a
    # non-negative population: ISHMAEL's per-category volume moments
    # (qvoli*, qaoli*), Ferrier-Aligo advected rime (qrimef), and NTU's
    # ice/snow/graupel/hail shape and volume moments (ai, as_, ag, ah, vi,
    # vs, vg, fi, fs, i3m) and its five-field aerosol budget (qdcn, qtcn,
    # qccn, qrcn, qnin).
    "qvoli", "qvoli2", "qvoli3", "qaoli", "qaoli2", "qaoli3", "qrimef",
    "ai", "as_", "ag", "ah", "vi", "vs", "vg", "fi", "fs", "i3m",
    "qdcn", "qtcn", "qccn", "qrcn", "qnin",
    'qt',
)


class PositivityError(ValueError):
    """A refusal.  Never a silent repair of the repair."""


def constrained_fields(fields: Sequence[str]) -> tuple[str, ...]:
    """Which of ``fields`` this module has a positivity opinion about."""

    return tuple(name for name in fields if name in NON_NEGATIVE_FIELDS)


def _host(array):
    """A host view of a device or host array, without importing cupy."""

    get = getattr(array, "get", None)
    if callable(get) and hasattr(array, "__cuda_array_interface__"):
        return np.asarray(get())
    return np.asarray(array)


def apply_positivity(prior: Mapping[str, object],
                     increments: Mapping[str, object], *,
                     policy: str = DEFAULT_POLICY,
                     fields: Sequence[str] | None = None) -> tuple[dict, dict]:
    """Enforce non-negativity of ``prior + increment``.  Returns a receipt.

    ``prior`` and ``increments`` are ``{field: array}`` of matching shape --
    either ``(nz, ny, nx)`` for one member or ``(R, nz, ny, nx)`` for a whole
    ensemble; this module is shape-agnostic because the constraint is
    pointwise.

    Fields outside :data:`NON_NEGATIVE_FIELDS` pass through **untouched and
    unexamined**: a theta or wind increment has no positivity constraint and
    inventing one would be a bug that looks like caution.

    The returned increments are a new mapping; the inputs are not mutated.
    """

    if policy not in POLICIES:
        raise PositivityError(
            f"unknown positivity policy {policy!r}; known policies are "
            f"{POLICIES}. There is no default that is right for every "
            "caller, which is why the filter does not pick one")
    names = tuple(increments) if fields is None else tuple(fields)
    missing = [name for name in names if name not in increments]
    if missing:
        raise PositivityError(
            f"positivity was asked about {missing}, which the increment "
            "mapping does not carry")

    constrained = constrained_fields(names)
    adjusted = {name: increments[name] for name in increments}
    per_field: list[dict] = []
    total_points = 0
    total_mass = 0.0

    for name in constrained:
        if name not in prior:
            raise PositivityError(
                f"cannot enforce positivity on {name!r} without its "
                "background: the constraint is on prior + increment, and "
                "an increment alone does not say whether the analysis is "
                "negative")
        if np.shape(prior[name]) != np.shape(increments[name]):
            raise PositivityError(
                f"{name}: prior {np.shape(prior[name])} and increment "
                f"{np.shape(increments[name])} disagree")

    def evaluated(name):
        # One field's analysis at a time.  Holding every constrained
        # field's float64 analysis and mask at once was a second
        # whole-ensemble copy of the hydrometeors on the host, for a
        # pointwise constraint that never needs two fields together
        # except through 'reject''s shared mask, which is a bool.
        background = _host(prior[name]).astype(np.float64, copy=False)
        increment = _host(increments[name]).astype(np.float64, copy=False)
        analysis = background + increment
        return background, increment, analysis, analysis < 0.0

    if policy == "mean-preserving":
        for name in constrained:
            background, increment, analysis, negative = evaluated(name)
            entry = _mean_preserving_field(
                name, background, increment, analysis, negative,
                _host(increments[name]).dtype)
            new_increment = entry.pop("_increment")
            if new_increment is not None:
                adjusted[name] = new_increment
            total_points += entry["negative_points"]
            total_mass += entry["mass_added"]
            per_field.append(entry)
    elif policy == "reject":
        # One shared mask: rejecting per field would leave a column whose
        # species disagree about which analysis they came from.
        shared = None
        for name in constrained:
            negative = evaluated(name)[3]
            shared = negative if shared is None else (shared | negative)
            del negative
        for name in constrained:
            background, increment, analysis, negative = evaluated(name)
            count = int(np.count_nonzero(negative))
            mass = float(-analysis[negative].sum()) if count else 0.0
            reverted = np.where(shared, 0.0, increment)
            adjusted[name] = reverted.astype(
                _host(increments[name]).dtype, copy=False)
            total_points += count
            total_mass += mass
            per_field.append({
                "field": name, "policy": "reject",
                "negative_points": count,
                "mass_that_would_have_been_added": mass,
                "points_reverted": int(np.count_nonzero(shared)),
                "total_points": int(negative.size),
            })
    else:
        for name in constrained:
            background, increment, analysis, negative = evaluated(name)
            count = int(np.count_nonzero(negative))
            # The clip raises the analysis from `analysis` to 0, so it ADDS
            # this much.  Recorded with that sign and that name, because
            # "clipped mass" reads as mass removed and it is not.
            mass = float(-analysis[negative].sum()) if count else 0.0
            if policy == "clip" and count:
                clipped = np.where(negative, -background, increment)
                adjusted[name] = clipped.astype(
                    _host(increments[name]).dtype, copy=False)
            total_points += count
            total_mass += mass
            per_field.append({
                "field": name, "policy": policy,
                "negative_points": count,
                "mass_added_by_clip": mass if policy == "clip" else 0.0,
                "mass_left_negative": 0.0 if policy == "clip" else mass,
                "total_points": int(negative.size),
                "worst_negative": (float(analysis[negative].min())
                                   if count else 0.0),
            })

    return adjusted, _receipt(policy, names, constrained, per_field,
                              total_points, total_mass)


def _receipt(policy, names, constrained, per_field, total_points,
             total_mass) -> dict:
    """The positivity receipt, one construction for the host and the card."""
    receipt = {
        "schema": POSITIVITY_SCHEMA,
        "stability": "experimental",
        "policy": policy,
        "constrained_fields": list(constrained),
        "unconstrained_fields": [name for name in names
                                 if name not in constrained],
        "negative_points": total_points,
        "mass_added_by_clip": total_mass if policy == "clip" else 0.0,
        # The mass the policy really added to the analysis (mean-preserving:
        # only where the filter's ensemble mean was itself negative).  For
        # the clip it is the same number as mass_added_by_clip, which is
        # kept under its old name for the receipts that already read it.
        "mass_added": (total_mass if policy in ("clip", "mean-preserving")
                       else 0.0),
        # Only 'none' leaves negative mass standing.  'reject' reverts the
        # increment at every offending point, so the analysis there IS the
        # background -- nonnegative by construction -- and reporting the
        # would-have-been-added mass under this name said the opposite of
        # what happened: a receipt claiming the run shipped negative
        # hydrometeor mass it had in fact refused to ship.  The quantity
        # itself is not lost; it is the per-field
        # ``mass_that_would_have_been_added``, which is what it always was.
        "mass_left_negative": total_mass if policy == "none" else 0.0,
        # What 'reject' actually does, named rather than implied.  It reverts
        # the increment for the CONSTRAINED fields at the shared mask and
        # leaves the same covariance increment standing everywhere else --
        # wind and theta at those points keep their analysed values.  That is
        # a constrained-field reject, not a whole-state one, and the two give
        # different analyses; which one this product should do is a ruling to
        # be taken, not a default to be changed quietly here.
        "positivity_semantics": ("constrained-field-reject"
                                 if policy == "reject" else policy),
        "per_field": per_field,
        "note": (
            "clip-at-zero ADDS mass and is biased wetward; the counts above "
            "are the diagnostic, not a formality. Alternatives are "
            "anamorphosis (belongs in the forward operator) and reject "
            "(conserves the background, invents gradients)."),
    }
    if policy == "mean-preserving":
        receipt["note"] = (
            "mean-preserving: where a member's analysis was negative the "
            "members' non-negative analyses were scaled by one factor per "
            "cell so the ensemble keeps the filter's mean, and the "
            "undershooting members end at zero. mass_added is what the "
            "rule could not avoid adding (cells whose ensemble mean was "
            "negative); mass_redistributed is what the clip would have "
            "added and the rule moved between members instead.")
        receipt["mass_redistributed"] = float(sum(
            entry["mass_redistributed"] for entry in per_field))
    if policy == "none":
        receipt["note"] = (
            "policy 'none': negative analyses were counted and left in "
            "place. This is a stated choice, not an oversight, and the "
            "microphysics will meet them.")
    return receipt


class DevicePositivity:
    """:func:`apply_positivity` and :func:`verify_non_negative`, chunk by
    chunk on the card the increments are computed on.

    A hook for :func:`woof.da.letkf_device.analyze_device`: ``chunk`` runs
    on each chunk's prior and increments, ``(F, R, points)`` float64 on the
    card, and rewrites the increments as the host policy would; ``finish``
    takes the chunks' returns in grid order and gives the receipt the host
    policy would have written for the whole ensemble.

    The rule is pointwise, so a chunk's increments are the host policy's
    bytes there: the same float64 sum of prior and increment, the same
    comparison with zero, ``-prior`` or zero where it acts.  The receipt's
    counts and minimum do not depend on order; its mass is the sum of the
    negative analyses in the host array's own order (member, then
    gridpoint), gathered from the chunks and summed once by numpy, so it is
    the host figure too.

    Breakage it removes: on the 9 km CONUS case (598 x 351 x 49, 32 members,
    fourteen analysed fields) the host policy took 35 to 265 s of one core
    per analysis on whole-ensemble float64 copies while every card sat
    idle, the largest stage of the analysis.
    """

    def __init__(self, policy: str, fields: Sequence[str] | None = None):
        if policy not in POLICIES:
            raise PositivityError(
                f"unknown positivity policy {policy!r}; known policies are "
                f"{POLICIES}. There is no default that is right for every "
                "caller, which is why the filter does not pick one")
        self.policy = policy
        self.fields = None if fields is None else tuple(fields)
        self.names = None
        self.constrained = None
        self.members = None
        self.dtype = np.dtype(np.float64)

    def bind_output_dtype(self, dtype):
        """The dtype the filter returns increments in.  The host policy
        acts on increments already in that dtype and casts its result back
        to it (with the mean-preserving rule's rounding floor), so the
        chunk rounds to it first and last, and the drain's cast is exact."""
        self.dtype = np.dtype(dtype)

    def _bind(self, fields):
        names = tuple(fields) if self.fields is None else self.fields
        missing = [name for name in names if name not in fields]
        if missing:
            raise PositivityError(
                f"positivity was asked about {missing}, which the increment "
                "mapping does not carry")
        if self.names is None:
            self.names = names
            self.constrained = constrained_fields(names)
        return {name: tuple(fields).index(name) for name in self.constrained}

    def chunk(self, fields, pri, inc, start, stop):
        import cupy as cp

        where = self._bind(fields)
        members = int(pri.shape[1])
        self.members = members
        out = {"points": int(stop - start), "fields": {}}
        if self.dtype != np.float64:
            for fi in where.values():
                inc[fi] = inc[fi].astype(self.dtype).astype(np.float64)
        shared = None
        if self.policy == "reject":
            for name, fi in where.items():
                negative = (pri[fi] + inc[fi]) < 0.0
                shared = negative if shared is None else (shared | negative)
                del negative
            out["reverted"] = 0 if shared is None else int(
                cp.count_nonzero(shared))
        if self.policy == "mean-preserving":
            for name, fi in where.items():
                out["fields"][name] = self._mean_preserving(
                    cp, pri, inc, fi, members)
            return out
        for name, fi in where.items():
            analysis = pri[fi] + inc[fi]
            negative = analysis < 0.0
            count = int(cp.count_nonzero(negative))
            entry = {"count": count}
            if count:
                # Member-major within the chunk, as the host array is.
                entry["values"] = cp.asnumpy(analysis[negative])
                entry["per_member"] = cp.asnumpy(
                    cp.count_nonzero(negative, axis=1))
                if self.policy == "clip":
                    inc[fi] = cp.where(negative, -pri[fi], inc[fi])
            if self.policy == "reject" and shared is not None:
                inc[fi] = cp.where(shared, 0.0, inc[fi])
            if self.policy in ("clip", "reject"):
                entry["after_min"] = float((pri[fi] + inc[fi]).min())
            out["fields"][name] = entry
            del analysis, negative
        return out

    @staticmethod
    def _member_mean(cp, values):
        """The host rule's member mean (:func:`_member_mean`): members
        added one at a time in order, then one division by R."""
        total = values[0].copy()
        for m in range(1, int(values.shape[0])):
            total = total + values[m]
        return total / float(values.shape[0])

    def _mean_preserving(self, cp, pri, inc, fi, members):
        """:func:`_mean_preserving_field` on one chunk, ``inc[fi]`` rewritten.

        Elementwise operations in the host rule's order, with the member
        means formed as numpy forms them, so the increments are the host
        policy's bytes; the receipt's order-dependent sums are formed from
        the per-chunk pieces in :meth:`finish`.
        """
        background = pri[fi]
        analysis = background + inc[fi]
        negative = analysis < 0.0
        count = int(cp.count_nonzero(negative))
        entry = {"count": count}
        if not count:
            entry["after_min"] = float(analysis.min())
            return entry
        entry["values"] = cp.asnumpy(analysis[negative])
        entry["per_member"] = cp.asnumpy(cp.count_nonzero(negative, axis=1))
        if members < 2:
            inc[fi] = cp.where(negative, -background, inc[fi])
            entry["cells"] = count
            entry["cells_mean_negative"] = count
            entry["shortfall"] = None
        else:
            hit = negative.any(axis=0)
            cols = cp.nonzero(hit)[0]
            values = analysis[:, cols]
            mean = self._member_mean(cp, values)
            held = cp.maximum(values, 0.0)
            held_mean = self._member_mean(cp, held)
            scale = cp.where(mean > 0.0,
                             mean / cp.where(held_mean > 0.0, held_mean, 1.0),
                             0.0)
            scale = cp.minimum(scale, 1.0)
            bounded = scale[None, :] * held
            back = background[:, cols]
            new = cp.where(bounded > 0.0, bounded - back, -back)
            block = inc[fi]
            block[:, cols] = new
            entry["cells"] = int(cols.size)
            entry["cells_mean_negative"] = int(cp.count_nonzero(mean <= 0.0))
            entry["shortfall"] = cp.asnumpy(cp.maximum(-mean, 0.0))
        # The host rule's cast to the increments' dtype (identity in
        # float64), and its floor at zero of a member the cast -- or the
        # subtraction bounded - background itself -- put a rounding below it.
        cast = inc[fi].astype(self.dtype).astype(np.float64)
        residue = (background + cast) < 0.0
        floored = int(cp.count_nonzero(residue))
        if floored:
            cast = cp.where(residue,
                            (-background).astype(self.dtype)
                            .astype(np.float64), cast)
        inc[fi] = cast
        entry["floored"] = floored
        entry["after_min"] = float((background + inc[fi]).min())
        return entry

    def finish(self, results) -> dict:
        if self.names is None:
            raise PositivityError("the positivity hook saw no chunk")
        if self.policy == "mean-preserving":
            return self._finish_mean_preserving(results)
        members = int(self.members)
        points = sum(int(res["points"]) for res in results)
        per_field = []
        total_points = 0
        total_mass = 0.0
        reverted = sum(int(res.get("reverted", 0)) for res in results)
        for name in self.constrained:
            entries = [res["fields"][name] for res in results]
            count = sum(int(entry["count"]) for entry in entries)
            values = None
            if count:
                # The host order: member 0's negatives over the whole grid,
                # then member 1's, ...; each chunk holds its members' runs
                # back to back.
                runs = []
                for entry in entries:
                    if not entry["count"]:
                        continue
                    edges = np.concatenate(([0], np.cumsum(
                        entry["per_member"])))
                    runs.append([entry["values"][edges[m]:edges[m + 1]]
                                 for m in range(members)])
                values = np.concatenate([run[m] for m in range(members)
                                         for run in runs])
            mass = float(-values.sum()) if count else 0.0
            total_points += count
            total_mass += mass
            if self.policy == "reject":
                per_field.append({
                    "field": name, "policy": "reject",
                    "negative_points": count,
                    "mass_that_would_have_been_added": mass,
                    "points_reverted": int(reverted),
                    "total_points": int(members * points),
                })
            else:
                per_field.append({
                    "field": name, "policy": self.policy,
                    "negative_points": count,
                    "mass_added_by_clip": (mass if self.policy == "clip"
                                           else 0.0),
                    "mass_left_negative": (0.0 if self.policy == "clip"
                                           else mass),
                    "total_points": int(members * points),
                    "worst_negative": (float(values.min()) if count
                                       else 0.0),
                })
        if self.policy in ("clip", "reject"):
            # verify_non_negative's post-condition, field by field in the
            # same order.
            for name in self.constrained:
                worst = min(float(res["fields"][name]["after_min"])
                            for res in results)
                if worst < 0.0:
                    raise PositivityError(
                        f"{name}: the analysis is still negative after the "
                        f"positivity policy ran (minimum {worst:g}); the "
                        "policy was applied to a different mapping than the "
                        "one about to be written")
        return _receipt(self.policy, self.names, self.constrained, per_field,
                        total_points, total_mass)

    def _ordered_negatives(self, entries, members):
        runs = []
        for entry in entries:
            if not entry["count"]:
                continue
            edges = np.concatenate(([0], np.cumsum(entry["per_member"])))
            runs.append([entry["values"][edges[m]:edges[m + 1]]
                         for m in range(members)])
        return np.concatenate([run[m] for m in range(members)
                               for run in runs])

    def _finish_mean_preserving(self, results) -> dict:
        members = int(self.members)
        points = sum(int(res["points"]) for res in results)
        per_field = []
        total_points = 0
        total_mass = 0.0
        for name in self.constrained:
            entries = [res["fields"][name] for res in results]
            count = sum(int(entry["count"]) for entry in entries)
            entry = {"field": name, "policy": "mean-preserving",
                     "member_axis": bool(members >= 2),
                     "negative_points": count,
                     "total_points": int(members * points),
                     "worst_negative": 0.0, "mass_clip_would_add": 0.0,
                     "mass_added": 0.0, "mass_redistributed": 0.0,
                     "cells_touched": 0, "cells_mean_negative": 0,
                     "rounding_floor_points": 0}
            if count:
                values = self._ordered_negatives(entries, members)
                entry["worst_negative"] = float(values.min())
                entry["mass_clip_would_add"] = float(-values.sum())
                touched = [e for e in entries if e["count"]]
                if members >= 2:
                    shortfall = np.concatenate([e["shortfall"]
                                                for e in touched])
                    added = float(members * shortfall.sum())
                    entry.update(
                        mass_added=added,
                        mass_redistributed=(entry["mass_clip_would_add"]
                                            - added),
                        cells_touched=int(sum(e["cells"] for e in touched)),
                        cells_mean_negative=int(sum(
                            e["cells_mean_negative"] for e in touched)),
                        rounding_floor_points=int(sum(
                            e.get("floored", 0) for e in touched)))
                else:
                    entry.update(
                        mass_added=entry["mass_clip_would_add"],
                        cells_touched=count, cells_mean_negative=count,
                        semantics=("one member: no ensemble mean to keep, "
                                   "so the rule is the clip"))
            total_points += entry["negative_points"]
            total_mass += entry["mass_added"]
            per_field.append(entry)
        for name in self.constrained:
            worst = min(float(res["fields"][name]["after_min"])
                        for res in results)
            if worst < 0.0:
                raise PositivityError(
                    f"{name}: the analysis is still negative after the "
                    f"positivity policy ran (minimum {worst:g}); the "
                    "policy was applied to a different mapping than the "
                    "one about to be written")
        return _receipt(self.policy, self.names, self.constrained, per_field,
                        total_points, total_mass)


def _member_mean(values):
    """Mean over the leading member axis, members added in order."""
    total = np.array(values[0], dtype=np.float64, copy=True)
    for member in range(1, int(values.shape[0])):
        total = total + values[member]
    return total / float(values.shape[0])


def _mean_preserving_field(name, background, increment, analysis, negative,
                           dtype) -> dict:
    """One field of the ``mean-preserving`` policy (module docstring).

    A leading member axis is present when the array is 4-D with two or
    more members; anything else is one member, for which the rule is the
    clip.  Returns the per-field receipt plus ``_increment``, the new
    increment in ``dtype`` (``None`` when no point was negative).
    """
    count = int(np.count_nonzero(negative))
    ensemble = analysis.ndim == 4 and analysis.shape[0] >= 2
    entry = {"field": name, "policy": "mean-preserving",
             "member_axis": bool(ensemble),
             "negative_points": count,
             "total_points": int(negative.size),
             "worst_negative": (float(analysis[negative].min())
                                if count else 0.0),
             "mass_clip_would_add": (float(-analysis[negative].sum())
                                     if count else 0.0),
             "mass_added": 0.0, "mass_redistributed": 0.0,
             "cells_touched": 0, "cells_mean_negative": 0,
             "rounding_floor_points": 0, "_increment": None}
    if not count:
        return entry
    if not ensemble:
        new_increment = np.where(negative, -background, increment)
        entry.update(mass_added=entry["mass_clip_would_add"],
                     cells_touched=count, cells_mean_negative=count,
                     semantics=("one member: no ensemble mean to keep, so "
                                "the rule is the clip"))
    else:
        hit = negative.any(axis=0)
        values = analysis[:, hit]
        # The member mean in one stated order: members added one at a
        # time, then one division.  ``values.mean(axis=0)`` summed in the
        # memory order of the fancy-indexed copy (pairwise when it comes
        # back member-contiguous), which the device hook could not match;
        # the two orders differ in the last bit only.
        mean = _member_mean(values)
        held = np.maximum(values, 0.0)
        held_mean = _member_mean(held)
        scale = np.where(mean > 0.0,
                         mean / np.where(held_mean > 0.0, held_mean, 1.0),
                         0.0)
        np.minimum(scale, 1.0, out=scale)
        bounded = scale[None, :] * held
        members = int(values.shape[0])
        added = float(members * np.maximum(-mean, 0.0).sum())
        new_increment = np.array(increment, dtype=np.float64, copy=True)
        # A member that ends at zero takes exactly -background, as the clip
        # does, so its cast increment cannot land a rounding below zero.
        new_increment[:, hit] = np.where(bounded > 0.0,
                                         bounded - background[:, hit],
                                         -background[:, hit])
        entry.update(mass_added=added,
                     mass_redistributed=entry["mass_clip_would_add"] - added,
                     cells_touched=int(np.count_nonzero(hit)),
                     cells_mean_negative=int(np.count_nonzero(mean <= 0.0)))
    cast = np.asarray(new_increment).astype(dtype, copy=False)
    # The increment is kept in the caller's dtype (float32 for a state): a
    # scaled member a few ulp above zero can land a rounding below it after
    # the cast.  Those points are floored at zero and counted; it is
    # rounding, not mass.
    residue = (background + cast.astype(np.float64)) < 0.0
    floored = int(np.count_nonzero(residue))
    if floored:
        cast = np.where(residue, (-background).astype(dtype), cast)
    entry["rounding_floor_points"] = floored
    entry["_increment"] = np.asarray(cast).astype(dtype, copy=False)
    return entry


def verify_non_negative(prior: Mapping[str, object],
                        increments: Mapping[str, object], *,
                        fields: Sequence[str] | None = None) -> None:
    """Post-condition: assert no constrained analysis is below zero.

    Cheap, and it is the check that catches a policy that ran on the wrong
    mapping -- the failure mode where a receipt says 4 117 points were
    clipped and the increments that got written were the unclipped ones.
    """

    names = tuple(increments) if fields is None else tuple(fields)
    for name in constrained_fields(names):
        analysis = (_host(prior[name]).astype(np.float64, copy=False)
                    + _host(increments[name]).astype(np.float64, copy=False))
        worst = float(analysis.min())
        if worst < 0.0:
            raise PositivityError(
                f"{name}: the analysis is still negative after the "
                f"positivity policy ran (minimum {worst:g}); the policy was "
                "applied to a different mapping than the one about to be "
                "written")
