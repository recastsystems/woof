"""EXPERIMENTAL (WOOF-ArWen v1.2): what a multi-moment scheme's analysis must be.

A two-moment microphysics scheme's state is PAIRS.  ``qr`` without ``nr``
is not a rain field with a missing diagnostic; it is not a rain field at
all -- the scheme's own slope closure reads both, and the pair is what
carries the drop size distribution.  An analysis that updates one member
of the pair and not the other produces a state the scheme cannot
evaluate.

This is not hypothetical.  A rung-2 real-radar LETKF cycle ran with

    ANALYSIS_FIELDS = ('thp','qv','u','v','qc','qr','qi','qs','qg')

on a Morrison (``mp_physics=10``) state.  Nine fields, no number
concentrations.  Reflectivity assimilation created hydrometeor mass in
cells the background had left clear, so the analysis held ``q`` up to
6.4 g/kg with ``N`` still at the background's exact zero -- for one
member, **100%** of the in-mask offenders were background-clear cells:
21 332 rain, 37 592 snow, 27 548 graupel.  Morrison's slope math then
does what it accurately must:

    lam = (six_c * N / q)**(1/3)  ->  0
    ilam = 1/lam                  ->  +inf
    n0  = N * ... * lam**e        ->  0
    Ze  = n0 * ... * ilam**e      ->  0 * inf = nan

and the reflectivity operator retains the NaN rather than letting a
corrupt state masquerade as its -35 dBZ clear-air floor
(``woof.verify.npref`` :9231-9239).  The operator was right.  The
ANALYSIS STATE was unphysical, and reflectivity OmA was unrecoverable
for that cycle.

That field list lived in a work script, but a product that lets a work
script do this has the defect.  So:

**Part 1 -- the state vector is derived from the scheme, never typed
out.**  :func:`analysis_fields` builds the update list from the selected
scheme's own prognostic moment set (``woof.core.nssl2_contract`` for
NSSL, the Registry-derived pairs below for the rest), and
:func:`validate_analysis_fields` refuses a list that moves a mass field
while leaving a paired number moment the state actually carries
untouched.  A caller that genuinely wants a single-moment update must say
so -- ``policy="single-moment-with-repair"`` -- and that policy REQUIRES
the guard below, because it is the policy that creates the offenders.

**Part 2 -- the guard, after any update.**  :func:`moment_consistency_report`
counts cells where a mass field is above the scheme's own activity
threshold and its paired number moment is at or below zero.
:func:`repair_moments` fixes them **using the scheme's own authority**,
not an invented intercept: for Morrison that is the PSD limiter at
``module_mp_morr_two_moment.F:1525-1638``, invoked here through its
float64 mirror ``woof.verify.npref._np_morrison_slopes``.  Given
``q > 0`` and ``N = 0`` that limiter computes ``lam = 0``, clamps it to
the species' own lower bound ``lam_min``, and back-computes
``N = q * lam_min**3 / six_c`` -- the largest-particle limit, which is
exactly what the scheme would impose at the first step of the next leg.
The repair is therefore not a new closure; it is the state the scheme
was going to bound the analysis to anyway, applied before the
reflectivity operator is asked about it.

Thompson (8 and 28) has the same kind of authority and it is a better
one, because the scheme states it for exactly this case: the entry block
of ``mp_thompson`` (``module_mp_thompson.F:1827-1899``) walks every
column before any process rate is computed and, where a species has mass
and no number, SETS the number from the mass under the scheme's own
assumed distribution -- a 100 um droplet, a 5 um crystal capped at
999e3 m^-3, a 1 mm median volume drop.  That block runs on the analysis
at the first step of the next leg whatever this module does, so applying
it here changes only WHEN the state becomes one the scheme can evaluate,
not what it becomes.  It reaches this module through its host mirror
:mod:`woof.core.thompson_entry`.

The same block is the authority for the reverse case, which Morrison's
limiter has nothing to say about: a cell the increment left with a
number moment and no mass is zeroed in both moments (:1844-1848,
:1871-1875, :1900-1904).  That state is not a NaN risk and it is NOT
counted against ``consistent`` -- it never was, and a new refusal needs
a breakage to name -- but where the repair runs it is repaired, and the
receipt counts it separately from the depleted pairs.

**Where a scheme's repair authority is not ported, there is no repair.**
NSSL (18) sets number from mass through its own fixed intercepts in
``module_mp_nssl_2mom.F``; that routine is not in this tree, and
inventing an intercept would be tuning science.  NSSL therefore DETECTS
and REFUSES.  A refusal naming 27 548 graupel cells is a usable result;
a number produced by a made-up intercept is not.

Nothing here is wired into a default route, and nothing here changes a
filter: the LETKF still returns increments and still owns no policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from woof.core.thompson_entry import THOMPSON_ENTRY_AUTHORITY

#: Provenance schema for every receipt this module produces.
MOMENT_SCHEMA = "gpuwm-da.moment-policy.v1"

#: What an analysis may do with a multi-moment scheme's state.
#:
#: ``"full-moment"`` (the default) -- every mass field in the update
#:   brings its paired number moment (and, for NSSL, its volume moment)
#:   with it.  Standard EnKF practice for a two-moment scheme, and the
#:   only policy under which the analysis is a state of the scheme that
#:   produced the background.
#:
#: ``"single-moment-with-repair"`` -- the caller updates mass only, on
#:   purpose, and accepts that the numbers are then the background's.
#:   Legitimate (the moment covariances are noisy and some centres do
#:   exactly this) and it REQUIRES the consistency guard, because it is
#:   the policy that manufactures ``q > 0, N = 0``.  Declaring it without
#:   running the repair is refused, not warned about.
MOMENT_POLICIES = ("full-moment", "single-moment-with-repair")
DEFAULT_MOMENT_POLICY = "full-moment"

#: Where each scheme's repair comes from.  ``None`` means the authority
#: is not ported into this tree and the guard refuses instead.
MORRISON_REPAIR_AUTHORITY = (
    "WRF v4.6.1 module_mp_morr_two_moment.F:1525-1638 PSD limiter, via "
    "its float64 mirror woof.verify.npref._np_morrison_slopes")

THOMPSON_REPAIR_AUTHORITY = THOMPSON_ENTRY_AUTHORITY


#: The registry's ``consumers.moments.repair_authority`` is a TOKEN (the
#: registry carries decisions and pointers, not prose with WRF citations);
#: this resolves each token to the sentence the receipts carry.
_REPAIR_AUTHORITIES = {
    "morrison-psd-limiter": MORRISON_REPAIR_AUTHORITY,
    "thompson-entry-block": THOMPSON_REPAIR_AUTHORITY,
}


class MomentPolicyError(ValueError):
    """A refusal about the moment structure of an analysis."""


@dataclass(frozen=True)
class MomentPair:
    """One species' prognostic moments.

    ``mass`` and ``number`` are state attribute names; ``volume`` is
    NSSL's predicted-density moment where the species has one.  Every
    field a scheme advances for this species and nothing else.
    """

    species: str
    mass: str
    number: str
    volume: str | None = None

    @property
    def fields(self) -> tuple[str, ...]:
        names = [self.mass, self.number]
        if self.volume is not None:
            names.append(self.volume)
        return tuple(names)


@dataclass(frozen=True)
class SchemeMoments:
    """A microphysics scheme's prognostic moment structure.

    Derived from the scheme's own contract, never from a list typed at a
    call site: `mp_physics` picks the scheme and the scheme states what
    it advances.  A field list that disagrees with this is a field list
    that has silently truncated somebody's state.
    """

    mp_physics: int
    name: str
    #: Mass fields the scheme advances with NO prognostic number moment.
    mass_only: tuple[str, ...]
    #: Species carrying a prognostic number (and possibly volume) moment.
    pairs: tuple[MomentPair, ...]
    #: Prognostic fields that are neither a mass nor a paired moment --
    #: NSSL's predicted CCN, for one.  Analysed with the rest under the
    #: full-moment policy; they have no consistency partner.
    unpaired: tuple[str, ...] = ()
    #: How this scheme's q>0/N=0 cells are repaired, or ``None``.
    repair_authority: str | None = None
    #: The scheme's own mass activity threshold (kg/kg).  Below it the
    #: scheme treats the species as absent and demands no number.
    q_threshold: float = 1.0e-14
    #: Whether the scheme itself zeroes a number moment that is left
    #: standing at or below that threshold.  Thompson does, in the same
    #: block that supplies its repair; Morrison's limiter does not, and
    #: claiming it did would be this module inventing a rule.
    zero_number_below_threshold: bool = False

    @property
    def two_moment(self) -> bool:
        return bool(self.pairs)

    @property
    def mass_fields(self) -> tuple[str, ...]:
        return self.mass_only + tuple(pair.mass for pair in self.pairs)

    @property
    def number_fields(self) -> tuple[str, ...]:
        return tuple(pair.number for pair in self.pairs)

    @property
    def volume_fields(self) -> tuple[str, ...]:
        return tuple(pair.volume for pair in self.pairs
                     if pair.volume is not None)

    @property
    def hydrometeor_fields(self) -> tuple[str, ...]:
        """Every prognostic hydrometeor field, mass and moment alike."""
        names: list[str] = list(self.mass_only)
        for pair in self.pairs:
            names.extend(pair.fields)
        names.extend(self.unpaired)
        return tuple(names)

    def pair_for_mass(self, mass: str) -> MomentPair | None:
        for pair in self.pairs:
            if pair.mass == mass:
                return pair
        return None


#: Morrison's five prognostic number moments (Registry: nc/nr/ni/ns/ng),
#: the scheme whose limiter supplies the repair.
_MORRISON = SchemeMoments(
    mp_physics=10, name="Morrison two-moment",
    mass_only=("qv",),
    pairs=(
        MomentPair("cloud", "qc", "nc"),
        MomentPair("rain", "qr", "nr"),
        MomentPair("ice", "qi", "ni"),
        MomentPair("snow", "qs", "ns"),
        MomentPair("graupel", "qg", "ng"),
    ),
    repair_authority=MORRISON_REPAIR_AUTHORITY,
    #: MQSMALL, module_mp_morr_two_moment.F / woof morrison.cu:26.  The
    #: scheme's own activity gate, and therefore the exact threshold above
    #: which it will read the number moment.
    q_threshold=1.0e-14,
)

#: Thompson (8) predicts rain number and ice number only; its cloud,
#: snow and graupel are single-moment.
_THOMPSON = SchemeMoments(
    mp_physics=8, name="Thompson",
    mass_only=("qv", "qc", "qs", "qg"),
    pairs=(
        MomentPair("rain", "qr", "nr"),
        MomentPair("ice", "qi", "ni"),
    ),
    repair_authority=THOMPSON_REPAIR_AUTHORITY,
    #: R1, module_mp_thompson.F:183 (woof/core/thompson_aerosol_state.py
    #: R1, woof/core/thompson_entry.py R1): the scheme's OWN activity
    #: gate, the value its entry block compares every mass against, and
    #: two orders of magnitude above the module default that stood here
    #: before the scheme's own threshold was available to read.
    q_threshold=1.0e-12,
    zero_number_below_threshold=True,
)

#: Thompson aerosol-aware (28) adds a prognostic droplet number to the
#: same two pairs, and carries the two aerosol number tracers, which are
#: prognostic but have no mass partner to be consistent with.  The state
#: spellings are woof.core.moist.THOMPSON_AERO_NUMBER_SPECIES.
_THOMPSON_AEROSOL = SchemeMoments(
    mp_physics=28, name="Thompson aerosol-aware",
    mass_only=("qv", "qs", "qg"),
    pairs=(
        MomentPair("cloud", "qc", "nc"),
        MomentPair("rain", "qr", "nr"),
        MomentPair("ice", "qi", "ni"),
    ),
    unpaired=("nwfa", "nifa"),
    repair_authority=THOMPSON_REPAIR_AUTHORITY,
    q_threshold=1.0e-12,
    zero_number_below_threshold=True,
)

_WSM6 = SchemeMoments(
    mp_physics=6, name="WSM6",
    mass_only=("qv", "qc", "qr", "qi", "qs", "qg"),
    pairs=(),
)

_KESSLER = SchemeMoments(
    mp_physics=1, name="Kessler",
    mass_only=("qv", "qc", "qr"),
    pairs=(),
)


def _nssl_scheme(**selectors) -> SchemeMoments:
    """NSSL's moment set, from the scheme's OWN selector resolution.

    ``woof.core.nssl2_contract.resolve_nssl2_mode`` applies WRF's
    option-18 consistency pass, so hail, predicted CCN and the predicted
    volume moments are on or off exactly as the run has them.  Enumerating
    NSSL's fields here instead would be a second contract, and a second
    contract is how a state gets truncated.
    """
    from woof.core import nssl2_contract as contract

    mode = contract.resolve_nssl2_mode(**selectors)
    transported = set(mode.transported_fields)
    pairs = [
        MomentPair("cloud", "qc", contract.TWO_MOMENT_NUMBER_FIELDS[0]),
        MomentPair("rain", "qr", contract.TWO_MOMENT_NUMBER_FIELDS[1]),
        MomentPair("ice", "qi", contract.TWO_MOMENT_NUMBER_FIELDS[2]),
        MomentPair("snow", "qs", contract.TWO_MOMENT_NUMBER_FIELDS[3]),
        MomentPair("graupel", "qg", contract.TWO_MOMENT_NUMBER_FIELDS[4],
                   contract.GRAUPEL_VOLUME_FIELD
                   if contract.GRAUPEL_VOLUME_FIELD in transported else None),
    ]
    if mode.hail:
        pairs.append(MomentPair(
            "hail", contract.HAIL_MASS_FIELD, contract.HAIL_NUMBER_FIELD,
            contract.HAIL_VOLUME_FIELD
            if contract.HAIL_VOLUME_FIELD in transported else None))
    if not mode.two_moment:
        return SchemeMoments(
            mp_physics=contract.MP_PHYSICS, name="NSSL (single-moment mode)",
            mass_only=tuple(name for name in mode.transported_fields
                            if name.startswith("q")
                            and not name.startswith("qn")),
            pairs=())
    kept = tuple(pair for pair in pairs if pair.number in transported)
    unpaired = ((contract.PREDICTED_CCN_FIELD,)
                if mode.predicted_ccn else ())
    return SchemeMoments(
        mp_physics=contract.MP_PHYSICS, name="NSSL two-moment",
        mass_only=("qv",),
        pairs=kept,
        unpaired=unpaired,
        # module_mp_nssl_2mom.F sets number from mass through the scheme's
        # own fixed intercepts (cnor/cnos/cnoh/cnohl).  That routine is not
        # ported into this tree; inventing an intercept here would be
        # tuning science, so NSSL detects and refuses instead of repairing.
        repair_authority=None,
    )


def _scheme_from_registry_row(mp_physics: int, row: Mapping) -> SchemeMoments:
    """One :class:`SchemeMoments` from the registry's ``consumers.moments`` row.

    The row is generated by ``tools/build_registry.py`` from each scheme's
    allocator names; building the object here rather than typing a second
    copy of the pairs is what lets a scheme join the analysis by adding its
    row and nothing else.
    """

    return SchemeMoments(
        mp_physics=int(mp_physics),
        name=str(row["name"]),
        mass_only=tuple(row["mass_only"]),
        pairs=tuple(
            MomentPair(str(pair["species"]), str(pair["mass"]),
                       str(pair["number"]), pair.get("volume"))
            for pair in row["pairs"]),
        unpaired=tuple(row.get("unpaired", ())),
        repair_authority=_repair_authority(mp_physics, row.get("repair_authority")),
        q_threshold=float(row.get("q_threshold", 1.0e-14)),
        zero_number_below_threshold=bool(
            row.get("zero_number_below_threshold", False)),
    )


def _repair_authority(mp_physics: int, token) -> str | None:
    if token is None:
        return None
    try:
        return _REPAIR_AUTHORITIES[token]
    except KeyError:
        raise RuntimeError(
            f"the registry names repair authority {token!r} for "
            f"mp_physics={mp_physics} and woof.da.moments knows no such "
            f"authority; known: {sorted(_REPAIR_AUTHORITIES)}") from None


def _static_schemes_from_the_registry() -> dict[int, SchemeMoments]:
    """Every scheme whose moment structure the registry publishes as data.

    Rows marked ``resolved_by`` (NSSL, whose set depends on its namelist
    switches) and null rows (microphysics off) are not static schemes and
    are left to :func:`scheme_moments`' own arms.  The module-level
    objects above are kept as the named spellings tests and readers use,
    and asserted equal to their registry rows so they cannot drift from
    the copy this table is actually built from.
    """

    from woof.physics_registry import consumer_rows_by_selector

    schemes: dict[int, SchemeMoments] = {}
    for mp, row in consumer_rows_by_selector("microphysics", "moments").items():
        if not isinstance(row, Mapping) or "resolved_by" in row:
            continue
        schemes[int(mp)] = _scheme_from_registry_row(mp, row)
    for spelled in (_KESSLER, _WSM6, _THOMPSON, _THOMPSON_AEROSOL,
                    _MORRISON):
        derived = schemes.get(spelled.mp_physics)
        if derived != spelled:
            raise RuntimeError(
                f"woof.da.moments spells mp_physics={spelled.mp_physics} as "
                f"{spelled!r} but the registry's consumers.moments row "
                f"builds {derived!r}; change the owning row in "
                "tools/build_registry.py and regenerate, or the spelling")
    return schemes


#: Every scheme this module knows the moment structure of.  Keyed on
#: ``mp_physics``, which is the only identity a run has.  DERIVED from the
#: physics registry's per-option ``consumers.moments`` rows rather than
#: typed: until it was, this table knew four schemes while the registry
#: shipped nine, and a Milbrandt-Yau, WDM6, aerosol-Thompson or P3 analysis
#: was either refused at its first leg or -- with ``mp_physics`` left unset
#: -- silently repaired through Morrison's PSD limiter.
_STATIC_SCHEMES = _static_schemes_from_the_registry()


def scheme_moments(mp_physics: int, **selectors) -> SchemeMoments:
    """The prognostic moment structure of ``mp_physics``.

    ``selectors`` are the scheme's own namelist switches where it has
    them (NSSL's ``nssl_hail_on`` and friends); passing them to a scheme
    that has none is a refusal rather than a silently ignored argument.
    """
    key = int(mp_physics)
    if key == 18:
        return _nssl_scheme(**selectors)
    if selectors:
        raise MomentPolicyError(
            f"mp_physics={key} takes no scheme selectors, got "
            f"{sorted(selectors)}; only mp_physics=18 resolves its moment "
            "set from namelist switches")
    if key not in _STATIC_SCHEMES:
        raise MomentPolicyError(
            f"no moment structure is registered for mp_physics={key}; "
            f"known schemes are {sorted(_STATIC_SCHEMES)} and 18. The table "
            "is derived from woof/physics_registry_v2.json "
            "(components.microphysics.options.<option>.consumers.moments), "
            "so a scheme joins it by publishing its row there, not by "
            "guessing at its state vector here")
    return _STATIC_SCHEMES[key]


#: The non-hydrometeor fields an analysis normally updates.  Offered as a
#: default so a caller states the hydrometeor policy and not the whole
#: list; it is a default, not a contract, and any of it can be replaced.
DEFAULT_BASE_FIELDS = ("thp", "qv", "u", "v")


def analysis_fields(mp_physics: int, *,
                    base: Sequence[str] = DEFAULT_BASE_FIELDS,
                    hydrometeors: bool = True,
                    policy: str = DEFAULT_MOMENT_POLICY,
                    **selectors) -> tuple[str, ...]:
    """The LETKF ``analysis_fields`` for a scheme, derived from the scheme.

    Under ``full-moment`` the hydrometeor half is the scheme's whole
    prognostic hydrometeor set -- masses, numbers, and NSSL's volume
    moments together.  Under ``single-moment-with-repair`` it is the mass
    fields only, which is the caller stating the truncation rather than
    performing it by omission.

    ``hydrometeors=False`` analyses none of them, which is a complete and
    consistent choice: the background's pairs stay exactly as the
    background had them and nothing can become inconsistent.
    """
    _check_policy(policy)
    scheme = scheme_moments(mp_physics, **selectors)
    names = list(dict.fromkeys(base))
    if hydrometeors:
        wanted = (scheme.hydrometeor_fields
                  if policy == "full-moment" else scheme.mass_fields)
        for name in wanted:
            if name not in names:
                names.append(name)
    return tuple(names)


def _check_policy(policy: str) -> None:
    if policy not in MOMENT_POLICIES:
        raise MomentPolicyError(
            f"unknown moment policy {policy!r}; choose from "
            f"{', '.join(MOMENT_POLICIES)}. There is no default that is "
            "right for every caller, and the wrong one produces a state "
            "the scheme cannot evaluate")


def pairs_present(available: Sequence[str], *,
                  mp_physics: int | None = None,
                  **selectors) -> tuple[MomentPair, ...]:
    """The moment pairs a state actually carries.

    With ``mp_physics`` this is the named scheme's pairs restricted to
    what the state has.  WITHOUT it, the pairs are detected from the field
    spellings themselves -- Morrison's ``nr`` and NSSL's ``qnr`` are
    different names for different schemes' moments, so a state answers the
    question "which pairs am I" without being told.  Detection exists
    because the code that writes an analysis
    (:mod:`woof.ensemble.increments`) is handed a checkpoint and no
    namelist, and a guard that could be skipped by not passing a config
    is a guard that will be.
    """
    have = set(available)
    if mp_physics is not None:
        scheme = scheme_moments(mp_physics, **selectors)
        candidates = scheme.pairs
    else:
        if selectors:
            raise MomentPolicyError(
                "scheme selectors need an explicit mp_physics; they cannot "
                "be applied to a detected moment structure")
        candidates = tuple(
            pair for scheme in (_MORRISON, _THOMPSON, _nssl_scheme())
            for pair in scheme.pairs)
    found: dict[str, MomentPair] = {}
    for pair in candidates:
        if pair.mass in have and pair.number in have:
            volume = pair.volume if (pair.volume in have) else None
            found.setdefault(pair.number,
                             MomentPair(pair.species, pair.mass,
                                        pair.number, volume))
    return tuple(found.values())


def validate_analysis_fields(fields: Sequence[str], *,
                             available: Sequence[str] | None = None,
                             mp_physics: int | None = None,
                             policy: str = DEFAULT_MOMENT_POLICY,
                             **selectors) -> dict:
    """Refuse an update that truncates a multi-moment state.

    ``fields`` is what the analysis proposes to update; ``available`` is
    what the background actually carries (the state's own capability).  A
    mass field in ``fields`` whose paired number moment is in
    ``available`` but NOT in ``fields`` is the test host H defect exactly, and
    under ``full-moment`` it is a refusal naming every such species.

    Under ``single-moment-with-repair`` the same situation is allowed and
    the returned receipt sets ``repair_required``; a caller that ignores
    it and writes the analysis anyway is refused by the guard in
    :mod:`woof.ensemble.increments`, which is where the state is.
    """
    _check_policy(policy)
    updated = tuple(dict.fromkeys(fields))
    carried = tuple(updated) if available is None else tuple(available)
    pairs = pairs_present(carried, mp_physics=mp_physics, **selectors)
    truncated = [pair for pair in pairs
                 if pair.mass in updated and pair.number not in updated]
    volume_truncated = [
        pair for pair in pairs
        if pair.mass in updated and pair.volume is not None
        and pair.volume not in updated]
    receipt = {
        "schema": MOMENT_SCHEMA,
        "policy": policy,
        "mp_physics": None if mp_physics is None else int(mp_physics),
        "scheme_source": "declared" if mp_physics is not None else "detected",
        "pairs_carried": [pair.mass for pair in pairs],
        "pairs_updated": [pair.mass for pair in pairs
                          if pair.mass in updated
                          and pair.number in updated],
        "mass_only_species": [pair.mass for pair in truncated],
        "volume_omitted_species": [pair.mass for pair in volume_truncated],
        "repair_required": bool(truncated or volume_truncated),
    }
    if not truncated and not volume_truncated:
        return receipt
    if policy == "single-moment-with-repair":
        return receipt
    detail = ", ".join(
        f"{pair.mass} without {pair.number}" for pair in truncated)
    volumes = ", ".join(
        f"{pair.mass} without {pair.volume}" for pair in volume_truncated)
    raise MomentPolicyError(
        "this analysis updates the mass of a multi-moment species and "
        "leaves its prognostic moment at the background's value: "
        + "; ".join(part for part in (detail, volumes) if part)
        + ". The background carries those moments, so the result is not a "
        "state of the scheme that produced it: where the analysis creates "
        "mass in a cell the background left clear, the pair becomes "
        "q > 0 with N = 0 and the scheme's slope closure evaluates to "
        "NaN. Analyse both moments (moment policy 'full-moment', which "
        "woof.da.moments.analysis_fields builds for you), or declare "
        "'single-moment-with-repair' and let the moment-consistency guard "
        "repair the pairs it breaks.")


def _host(array):
    """A host float64 view of a device or host array."""
    get = getattr(array, "get", None)
    if callable(get) and hasattr(array, "__cuda_array_interface__"):
        return np.asarray(get(), dtype=np.float64)
    return np.asarray(array, dtype=np.float64)


def _depleted_offenders(mass: np.ndarray, number: np.ndarray,
                        threshold: float) -> np.ndarray:
    """Cells the scheme calls active whose number moment is spent.

    The ONE place this comparison is spelled, because
    :func:`moment_consistency_report` decides who is repaired and
    :func:`repair_moments` writes them, and two spellings of that pair
    would let the repair miss a cell the guard counted (or write one it
    did not).  ``mass > threshold`` is False for a NaN mass, so this
    predicate deliberately says nothing about non-finite cells --
    :func:`_nonfinite_offenders` is where they are found.
    """
    return np.isfinite(mass) & (mass > threshold) & (number <= 0.0)


def _stranded_numbers(mass: np.ndarray, number: np.ndarray,
                      threshold: float) -> np.ndarray:
    """Cells the scheme calls absent that still carry a number moment.

    The mirror image of :func:`_depleted_offenders`, and only meaningful
    for a scheme that says what to do about it: Thompson's entry block
    zeroes the mass AND the number at every cell whose mass is at or
    below ``R1`` (:1844-1848, :1871-1875, :1900-1904), which is what an
    increment that removes all of a species' mass leaves behind.  Like
    the depleted predicate this is False for a NaN mass, so non-finite
    cells stay the business of :func:`_nonfinite_offenders`.
    """
    return np.isfinite(mass) & (mass <= threshold) & (number > 0.0)


def _nonfinite_offenders(mass: np.ndarray, number: np.ndarray,
                         volume: np.ndarray | None,
                         threshold: float) -> tuple:
    """``(mass, number, volume)`` masks of non-finite active moments.

    IEEE comparison is why this function exists.  ``NaN <= 0.0`` is
    **False** and ``NaN > threshold`` is **False**, so a cell holding
    ``qr = 1e-3`` with ``nr = NaN`` is neither an active cell nor an
    offender by the depleted-number predicate: the guard called it
    consistent, the writer wrote it, and the receipt attested to it.
    That is precisely the state the module's own docstring says the
    scheme evaluates to NaN -- ``lam = (six_c * N / q)**(1/3)`` is NaN
    for a NaN ``N`` just as surely as it is zero for ``N = 0``.

    A non-finite MASS is counted wherever it appears.  ``-inf`` is not
    above the threshold and ``NaN`` compares False against it, so neither
    can be shown to be an inactive cell the scheme would ignore; a
    hydrometeor mass that is not a finite number of kg/kg is not a state
    to decide pair-consistency for at all.  Number and volume moments are
    counted where the mass is active OR is itself non-finite -- below the
    scheme's own activity gate the scheme reads neither.
    """
    finite_mass = np.isfinite(mass)
    # "active, or not provably inactive": the cells the scheme will read
    # the pair's other moments at.
    reads_moments = (~finite_mass) | (mass > threshold)
    bad_mass = ~finite_mass
    bad_number = reads_moments & ~np.isfinite(number)
    bad_volume = (reads_moments & ~np.isfinite(volume)
                  if volume is not None else None)
    return bad_mass, bad_number, bad_volume


def moment_consistency_report(state: Mapping[str, object], *,
                              mp_physics: int | None = None,
                              pairs: Sequence[MomentPair] | None = None,
                              q_threshold: float | None = None,
                              **selectors) -> dict:
    """Count cells whose moment pair the scheme cannot evaluate.

    Two kinds of cell, counted separately because only one of them can be
    repaired:

    ``offending_cells``
        mass above the scheme's activity threshold with a number moment
        at or below zero.  The scheme's own limiter knows what number
        belongs there (:func:`repair_moments`).

    ``nonfinite_cells``
        a mass, number or volume moment that is not a finite number where
        the scheme will read it.  No limiter repairs this: Morrison's
        bound is ``N = q * lam_min**3 / six_c``, which is NaN for a NaN
        ``q`` and cannot be evaluated for a NaN ``N`` in the first place.
        It is a refusal, and it is counted here rather than left to the
        arithmetic because IEEE comparison hides it from every ``<=``
        and ``>`` in the depleted-number check.

    ``consistent`` is true only when BOTH are zero.

    ``state`` is any ``{field: array}`` mapping -- a live state's
    ``__dict__``, a checkpoint's ``state/*`` arrays, or one member's
    analysis.  The threshold defaults to the scheme's own activity gate,
    which is the exact value above which the scheme will read the number
    moment: below it the scheme treats the species as absent and asks
    nothing of the pair.
    """
    if pairs is None:
        pairs = pairs_present(tuple(state), mp_physics=mp_physics,
                              **selectors)
    scheme = (scheme_moments(mp_physics, **selectors)
              if mp_physics is not None else None)
    if q_threshold is None:
        q_threshold = (scheme.q_threshold if scheme is not None
                       else _MORRISON.q_threshold)
    threshold = float(q_threshold)
    # Only a scheme that states the rule has stranded cells: with the
    # structure merely detected, this module does not know whether the
    # scheme zeroes them or carries them, and counting them anyway would
    # be reporting a rule nobody wrote.
    count_stranded = bool(scheme is not None
                          and scheme.zero_number_below_threshold)
    species: list[dict] = []
    total = 0
    nonfinite_total = 0
    stranded_total = 0
    for pair in pairs:
        mass = _host(state[pair.mass])
        number = _host(state[pair.number])
        volume = (_host(state[pair.volume])
                  if pair.volume is not None and pair.volume in state
                  else None)
        offenders = _depleted_offenders(mass, number, threshold)
        count = int(np.count_nonzero(offenders))
        total += count
        stranded = (int(np.count_nonzero(
            _stranded_numbers(mass, number, threshold)))
            if count_stranded else 0)
        stranded_total += stranded
        bad_mass, bad_number, bad_volume = _nonfinite_offenders(
            mass, number, volume, threshold)
        bad_any = bad_mass | bad_number
        if bad_volume is not None:
            bad_any = bad_any | bad_volume
        bad_count = int(np.count_nonzero(bad_any))
        nonfinite_total += bad_count
        entry = {
            "species": pair.species,
            "mass_field": pair.mass,
            "number_field": pair.number,
            "volume_field": pair.volume,
            "offending_cells": count,
            "stranded_number_cells": stranded,
            "max_offending_mass_kg_kg": (
                float(mass[offenders].max()) if count else 0.0),
            "nonfinite_cells": bad_count,
            "nonfinite_mass_cells": int(np.count_nonzero(bad_mass)),
            "nonfinite_number_cells": int(np.count_nonzero(bad_number)),
            "nonfinite_volume_cells": (
                0 if bad_volume is None
                else int(np.count_nonzero(bad_volume))),
        }
        species.append(entry)
    return {
        "schema": MOMENT_SCHEMA,
        "q_threshold_kg_kg": threshold,
        "mp_physics": None if mp_physics is None else int(mp_physics),
        "pairs_checked": [pair.mass for pair in pairs],
        "offending_cells_total": total,
        "nonfinite_cells_total": nonfinite_total,
        # A number moment standing over mass the scheme calls absent is
        # not a state the slope closure evaluates to NaN, so it is
        # reported and repaired but does not make the state
        # inconsistent: a refusal has to name a breakage, and this one
        # has none to name.
        "stranded_number_cells_total": stranded_total,
        "consistent": total == 0 and nonfinite_total == 0,
        "species": species,
    }


def nonfinite_moment_refusal(report: Mapping[str, object], *,
                             where: str) -> str:
    """The message for a state whose moments are not finite.

    One spelling, used by :func:`repair_moments` and by both writers in
    :mod:`woof.ensemble.increments`, so a caller cannot tell which path
    caught it -- and so neither path can be fixed without the other.
    """
    parts = []
    for entry in report["species"]:
        if not entry.get("nonfinite_cells"):
            continue
        detail = [f"{entry['mass_field']}: {entry['nonfinite_mass_cells']}",
                  f"{entry['number_field']}: "
                  f"{entry['nonfinite_number_cells']}"]
        if entry.get("volume_field") and entry.get("nonfinite_volume_cells"):
            detail.append(f"{entry['volume_field']}: "
                          f"{entry['nonfinite_volume_cells']}")
        parts.append(f"{entry['species']} ({', '.join(detail)})")
    return (
        f"the analysis for {where} carries "
        f"{report['nonfinite_cells_total']} cell(s) whose moment pair is "
        "not a finite number where the scheme reads it: "
        + "; ".join(parts)
        + ". IEEE comparison is why this is a separate refusal: NaN is "
          "neither above the activity threshold nor at-or-below zero, so a "
          "NaN number moment beside active mass passes every ordering test "
          "the depleted-number guard makes and would otherwise be attested "
          "as consistent. No scheme limiter repairs it either -- Morrison's "
          "bound N = q * lam_min**3 / six_c is NaN for a NaN q and has "
          "nothing to bound for a NaN N. Fix the background or the "
          "increment; this state cannot be analysed.")


#: Chunk of cells the Morrison limiter is called over at a time.  The
#: mirror allocates roughly twenty float64 arrays of the length it is
#: given, so a whole nest at once is gigabytes for no benefit.
_REPAIR_CHUNK_CELLS = 1 << 20


def _morrison_bounded_numbers(state: Mapping[str, object],
                              pairs: Sequence[MomentPair], *,
                              morr_rimed_ice: int = 1):
    """Morrison's own bounded numbers for the state's five species.

    Calls the scheme mirror -- ``woof.verify.npref._np_morrison_slopes``,
    the float64 transcription of the limiter at
    ``module_mp_morr_two_moment.F:1525-1638`` -- rather than restating its
    constants.  A second copy of ``lam_min`` here is a second copy that
    can drift from the scheme by one edit.

    ``reset_cloud_number=False``: the limiter's ``INUM=1`` branch would
    overwrite every cloud number with 250 cm-3, which is the scheme's
    entry condition and not a repair.

    Density and temperature reach the limiter ONLY through the cloud
    branch's ``pgam``, multiplied by the cloud number -- which at an
    offending cell is zero.  The bounded numbers this returns AT OFFENDING
    CELLS are therefore independent of both, and
    ``tests/test_da_moments.py`` pins that rather than asserting it.
    """
    from woof.verify.npref import _np_morrison_slopes

    by_species = {pair.species: pair for pair in pairs}
    shape = _host(state[pairs[0].mass]).shape
    size = int(np.prod(shape)) if shape else 1
    out = {pair.number: np.empty(size, np.float64) for pair in pairs}
    flat_q, flat_n = {}, {}
    for key, species in (("c", "cloud"), ("r", "rain"), ("i", "ice"),
                         ("s", "snow"), ("g", "graupel")):
        pair = by_species.get(species)
        if pair is None:
            flat_q[key] = np.zeros(size, np.float64)
            flat_n[key] = np.zeros(size, np.float64)
        else:
            flat_q[key] = _host(state[pair.mass]).reshape(-1)
            flat_n[key] = _host(state[pair.number]).reshape(-1)

    for start in range(0, size, _REPAIR_CHUNK_CELLS):
        stop = min(start + _REPAIR_CHUNK_CELLS, size)
        window = slice(start, stop)
        chunk_q = {key: value[window] for key, value in flat_q.items()}
        chunk_n = {key: value[window] for key, value in flat_n.items()}
        # Any positive density and any temperature: see the docstring.
        # The limiter reads them only where it is multiplied by a cloud
        # number that is zero at every cell this repair touches.
        ones = np.ones(stop - start, np.float64)
        _, _, bounded = _np_morrison_slopes(
            chunk_q, chunk_n, ones, 273.15 * ones,
            reset_cloud_number=False, morr_rimed_ice=morr_rimed_ice)
        for key, species in (("c", "cloud"), ("r", "rain"), ("i", "ice"),
                             ("s", "snow"), ("g", "graupel")):
            pair = by_species.get(species)
            if pair is not None:
                out[pair.number][window] = bounded[key]
    return {name: values.reshape(shape) for name, values in out.items()}


#: The state field the density comes from, and the one the scheme is
#: handed by woof.core.microphysics (its module docstring line 25).
THOMPSON_DENSITY_FIELD = "alt"


def _thompson_bounded_numbers(state: Mapping[str, object],
                              pairs: Sequence[MomentPair], *,
                              q_threshold: float):
    """Thompson's own entry-block numbers for the state's three species.

    Calls the host mirror :mod:`woof.core.thompson_entry` rather than
    restating the scheme's distributions, for the reason the Morrison
    arm gives: a second copy of a scheme constant is a copy that drifts
    from the scheme by one edit.

    The entry block works per volume, so this needs the density the
    scheme is handed.  It is taken from the state's own ``alt`` -- the
    inverse dry density every microphysics adapter in this tree passes
    as ``rho`` -- and its ABSENCE is a refusal naming the field, because
    the alternative is a density this module made up, and made-up
    densities are how a repair becomes a tuning decision.
    """
    from woof.core import thompson_entry

    alt = state.get(THOMPSON_DENSITY_FIELD)
    if alt is None:
        raise MomentPolicyError(
            "Thompson's entry block sets a number moment from a mass per "
            "unit VOLUME, so the repair needs the density the scheme is "
            f"handed, and this state carries no {THOMPSON_DENSITY_FIELD!r} "
            "to take it from (rho = 1/alt, the value "
            "woof.core.microphysics passes every scheme). Hand the "
            "repair a state that carries the inverse density rather than "
            "letting it choose one: at a cell being repaired the density "
            "cancels everywhere except the scheme's 999e3 m^-3 ice "
            "ceiling, and that is precisely the cell a radar analysis "
            "creates.")
    inverse_density = _host(alt)
    with np.errstate(divide="ignore", invalid="ignore"):
        density = 1.0 / inverse_density
    if not np.all(np.isfinite(density)):
        raise MomentPolicyError(
            "the state's inverse density is zero or not finite in "
            f"{int(np.count_nonzero(~np.isfinite(density)))} cell(s), so "
            "the density Thompson's entry block works in cannot be formed "
            "there. That is a broken background, not a repairable "
            "analysis.")
    bounded: dict[str, np.ndarray] = {}
    for pair in pairs:
        bounded[pair.number] = thompson_entry.np_thompson_entry_numbers(
            pair.species, _host(state[pair.mass]),
            _host(state[pair.number]), density)
    return bounded


#: Which bounder each authority names.  The authority string is what the
#: receipt carries, so the table is keyed on it: a receipt that claims an
#: authority and a repair that ran a different scheme's limiter cannot
#: both come out of this dictionary.
_BOUNDED_NUMBERS = {
    MORRISON_REPAIR_AUTHORITY: (
        lambda state, pairs, *, morr_rimed_ice, q_threshold:
        _morrison_bounded_numbers(state, pairs,
                                  morr_rimed_ice=morr_rimed_ice)),
    THOMPSON_REPAIR_AUTHORITY: (
        lambda state, pairs, *, morr_rimed_ice, q_threshold:
        _thompson_bounded_numbers(state, pairs, q_threshold=q_threshold)),
}


def repair_moments(state: Mapping[str, object], *,
                   mp_physics: int | None = None,
                   pairs: Sequence[MomentPair] | None = None,
                   q_threshold: float | None = None,
                   morr_rimed_ice: int = 1,
                   **selectors) -> tuple[dict, dict]:
    """``(repaired fields, receipt)`` for the pairs an analysis broke.

    Only OFFENDING cells are written.  The limiter would rebound every
    active cell's number -- that is what the scheme does at its first
    step -- but a data-assimilation repair that quietly rewrote healthy
    moments would be a second analysis nobody asked for, and it would be
    invisible in the receipt.  So the scheme's bounded number is taken,
    and applied where and only where the pair was broken.

    Refuses when the scheme has no ported repair authority: NSSL's
    intercept-based number initialisation is not in this tree, and a
    number invented here would be a tuning decision wearing a repair's
    clothes.

    Refuses BEFORE that, and for every scheme, when any moment is not
    finite where the scheme reads it.  A repair is a limiter evaluated on
    the state; a limiter evaluated on NaN returns NaN, so "repairing" a
    non-finite pair would replace one unevaluable state with another and
    stamp the receipt ``repaired``.
    """
    scheme = (scheme_moments(mp_physics, **selectors)
              if mp_physics is not None else None)
    if pairs is None:
        pairs = pairs_present(tuple(state), mp_physics=mp_physics,
                              **selectors)
    report = moment_consistency_report(
        state, mp_physics=mp_physics, pairs=pairs, q_threshold=q_threshold)
    if report["nonfinite_cells_total"]:
        raise MomentPolicyError(
            nonfinite_moment_refusal(report, where="this state"))
    nothing_to_do = (report["consistent"]
                     and not report.get("stranded_number_cells_total"))
    if nothing_to_do or not pairs:
        return {}, {**report, "repaired": False, "repaired_cells_total": 0,
                    "stranded_cells_repaired": 0,
                    "authority": None if scheme is None
                    else scheme.repair_authority}

    authority = scheme.repair_authority if scheme is not None else None
    if scheme is None:
        # DETECTED STRUCTURE, with no mp_physics to name the scheme.  The
        # subset test alone is not identification: Morrison's number
        # fields are a SUPERSET of Thompson's {nr, ni} and P3's {nr, ni},
        # and are a subset of Milbrandt-Yau's set minus nh, so a state
        # carrying only the pairs an analysis touched matched Morrison and
        # was repaired through Morrison's PSD limiter whatever scheme
        # actually produced it (audit R-016).  A limiter is scheme
        # physics, so the ambiguity is refused by name instead: the caller
        # states mp_physics, which every DA door already has.
        detected = {pair.number for pair in pairs}
        # An EXACT match identifies a scheme where a subset match cannot:
        # a Morrison state carries {nc, nr, ni, ns, ng} and nothing else,
        # while a Milbrandt-Yau state carries those five AND nh, so the
        # exact test separates the two the subset test conflates.  Only
        # when no registered scheme spells exactly this structure -- or
        # when more than one does, which {nr, ni} does for Thompson and
        # for P3 -- is the question genuinely open.
        exact = sorted(mp for mp, candidate in _STATIC_SCHEMES.items()
                       if detected == set(candidate.number_fields))
        candidates = exact if len(exact) == 1 else sorted(
            mp for mp, candidate in _STATIC_SCHEMES.items()
            if detected <= set(candidate.number_fields))
        if len(candidates) == 1:
            identified = _STATIC_SCHEMES[candidates[0]]
            # An exact match names a scheme, but it does not settle WHOSE
            # limiter to run when another registered scheme admits the
            # same structure and repairs it through a different block with
            # a different answer.  {nc, nr, ni} is exactly aerosol-aware
            # Thompson's set and a subset of Morrison's, and the two
            # answer a depleted pair differently -- Morrison's PSD limiter
            # at its lam_min, Thompson's entry block at a 100 um droplet,
            # a 5 um crystal and a 1 mm drop -- so that structure is named
            # as open rather than resolved by whichever test ran first.
            # A scheme with NO ported authority is not a rival: it offers
            # no competing answer, only a competing identity, which is the
            # Milbrandt-Yau case the exact match was written for.
            rivals = sorted(
                mp for mp, candidate in _STATIC_SCHEMES.items()
                if mp != candidates[0]
                and detected <= set(candidate.number_fields)
                and candidate.repair_authority is not None
                and candidate.repair_authority != identified.repair_authority)
            if rivals:
                raise MomentPolicyError(
                    "this state's moment spellings "
                    f"({', '.join(sorted(detected))}) are exactly "
                    f"mp_physics={candidates[0]}'s and are also carried by "
                    "mp_physics "
                    + ", ".join(str(mp) for mp in rivals)
                    + ", which repairs a depleted pair through a different "
                    "scheme's block with a different answer. A number "
                    "repair is the SCHEME's own limiter, so repairing on "
                    "the structure alone would run one scheme's bounds "
                    "over another scheme's state and record the wrong "
                    "authority in the receipt. Pass mp_physics so the "
                    "scheme is declared rather than guessed at.")
            authority = identified.repair_authority
        elif len(candidates) > 1:
            raise MomentPolicyError(
                "this analysis left "
                f"{report['offending_cells_total']} cell(s) holding mass "
                "above the activity threshold with a number moment at or "
                "below zero, and the moment structure that was detected "
                f"({', '.join(sorted(detected))}) belongs to more than one "
                "registered scheme (mp_physics "
                + ", ".join(str(mp) for mp in candidates)
                + "). A number repair is the SCHEME's own limiter, so "
                "repairing on a guess would run one scheme's PSD bounds "
                "over another scheme's state and record the wrong "
                "authority in the receipt. Pass mp_physics to "
                "repair_moments (every DA door carries it) so the "
                "scheme's own row decides.")
    if authority is None:
        raise MomentPolicyError(
            "this analysis left "
            f"{report['offending_cells_total']} cell(s) holding mass above "
            f"{report['q_threshold_kg_kg']:g} kg/kg with a number moment at "
            "or below zero, and no repair authority for this scheme is "
            "ported into this tree ("
            + ", ".join(f"{entry['mass_field']}: {entry['offending_cells']}"
                        for entry in report["species"]
                        if entry["offending_cells"]) + "). "
            "The scheme's own number initialisation is what would have to "
            "supply the missing moments, and inventing an intercept here "
            "would be a science decision, not a repair. Analyse the number "
            "moments with the mass instead.")

    threshold = report["q_threshold_kg_kg"]
    bounded = _BOUNDED_NUMBERS[authority](
        state, pairs, morr_rimed_ice=morr_rimed_ice, q_threshold=threshold)
    repaired: dict[str, np.ndarray] = {}
    total = 0
    stranded_total = 0
    for pair, entry in zip(pairs, report["species"]):
        stranded_count = int(entry.get("stranded_number_cells", 0))
        if not entry["offending_cells"] and not stranded_count:
            continue
        mass = _host(state[pair.mass])
        number = _host(state[pair.number])
        offenders = _depleted_offenders(mass, number, threshold)
        fixed = np.array(number, copy=True)
        if entry["offending_cells"]:
            fixed[offenders] = bounded[pair.number][offenders]
            entry["repaired_cells"] = int(entry["offending_cells"])
            entry["repaired_number_min"] = float(fixed[offenders].min())
            entry["repaired_number_max"] = float(fixed[offenders].max())
            total += int(entry["offending_cells"])
        if stranded_count:
            # The bounder already carries the scheme's own answer here:
            # the entry block writes the number back as zero wherever the
            # mass is at or below its activity gate, so this reads the
            # same array rather than writing a zero of its own.
            stranded = _stranded_numbers(mass, number, threshold)
            fixed[stranded] = bounded[pair.number][stranded]
            entry["stranded_cells_repaired"] = stranded_count
            stranded_total += stranded_count
        repaired[pair.number] = fixed
    return repaired, {
        **report,
        "repaired": True,
        "repaired_cells_total": total,
        "stranded_cells_repaired": stranded_total,
        "authority": authority,
        "note": ("cells with mass above the scheme's activity threshold "
                 "and a non-positive number moment were written, and -- "
                 "where the scheme itself zeroes them -- cells carrying a "
                 "number moment over mass it calls absent; healthy "
                 "moments are untouched"),
    }


__all__ = [
    "DEFAULT_BASE_FIELDS",
    "DEFAULT_MOMENT_POLICY",
    "MOMENT_POLICIES",
    "MOMENT_SCHEMA",
    "MORRISON_REPAIR_AUTHORITY",
    "THOMPSON_REPAIR_AUTHORITY",
    "MomentPair",
    "MomentPolicyError",
    "SchemeMoments",
    "analysis_fields",
    "moment_consistency_report",
    "nonfinite_moment_refusal",
    "pairs_present",
    "repair_moments",
    "scheme_moments",
    "validate_analysis_fields",
]
