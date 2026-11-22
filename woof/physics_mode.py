"""The physics-fidelity axis and the divergence ledger it resolves.

WOOF-ArWen ships one physics identity today: every selector is transcribed from
WRF v4.6.1 and pinned against it.  A *patch* is a place where WOOF-ArWen would
deliberately do something else because the physics is believed to be better,
and the only accurate way to find out is to score both against observations.
This module is the machinery that lets one configuration file express both
sides of that question without a second file that can drift from the first.

THE AXIS
--------
``physics_mode`` is an ``[experiment]`` key with two values:

``"wrf-faithful"``
    Every registered ledger entry resolves to its WRF-faithful value.  This
    is the default and it is what the oracle apparatus governs.
``"arwen-patched"``
    Every *toggleable* entry of a registered patch-set version resolves to
    its patched value.  ``patchset`` names the version (``"v1"`` today);
    ``patches`` optionally restricts the resolution to a subset of that
    version's entries, which is how a single-patch ablation arm is spelled.

``arwen-patched`` NEVER means "whatever is currently believed" -- it always
resolves through a version registered in :data:`PATCHSETS`, so a receipt
that records ``patchset = "v1"`` records a fixed vector, and adding an entry
to the ledger later cannot retroactively change what an old run meant.

GOVERNANCE (why a present key and an absent key are not the same thing)
-----------------------------------------------------------------------
When ``physics_mode`` appears in ``[experiment]``, the axis becomes the
AUTHOR of every key its resolved entries name: the loader writes the
resolved value onto every domain and REFUSES an explicit occurrence of that
key in ``[shared]`` or any ``[[domain]]``.  Two authors for one key is the
defect ``woof.experiment._reject_unknown_keys`` exists to prevent, one
level up: the run would be confident, complete, and computed from a value
whose provenance nobody can state.

When ``physics_mode`` is ABSENT the axis authors nothing and every key stays
where it has always been.  Absent is therefore byte-inert for every
configuration that predates this module -- which is the whole tree -- and
its reported mode is still ``"wrf-faithful"``, because that is true: no
registered patch is applied.  The two states differ only in who writes the
key, and which one is in force is visible in the file.

This is also what lets one battery case file serve every arm: the case TOML
omits the governed keys, and the arm is the two-line ``[experiment]``
overlay that selects the mode.  A case file that DOES pin a governed key is
refused by name rather than silently overridden, and the refusal says which
keys the mode governs so the overlay can be composed against
:func:`governed_keys`.
"""

from __future__ import annotations

from dataclasses import dataclass

#: ``physics_mode`` values.  The faithful value is the default everywhere.
PHYSICS_MODE_WRF_FAITHFUL = "wrf-faithful"
PHYSICS_MODE_ARWEN_PATCHED = "arwen-patched"
PHYSICS_MODES = (PHYSICS_MODE_WRF_FAITHFUL, PHYSICS_MODE_ARWEN_PATCHED)

#: The patch-set version ``arwen-patched`` resolves to when none is named.
DEFAULT_PATCHSET = "v1"


@dataclass(frozen=True)
class LedgerEntry:
    """One divergence-ledger entry: a place WOOF-ArWen may diverge on purpose.

    ``entry_id`` is the ledger's own label (``"L3"``), stable across
    patch-set versions.  ``key`` is the mechanism-named configuration key
    the entry moves and ``faithful``/``patched`` are the two values it moves
    it between -- both stated, so a receipt records the whole edge rather
    than the endpoint it happened to take.

    ``toggleable`` is False for an entry that exists in the ledger but is
    not a bit-flip: a large multi-key overlay, or an entry whose class is
    "already the shipped default, and there is nothing to toggle".  A
    non-toggleable entry can never enter a resolved vector, and naming it in
    ``patches`` is refused with ``gate`` as the reason.
    """

    entry_id: str
    key: str
    faithful: object
    patched: object
    summary: str
    toggleable: bool = True
    gate: str = ""

    def value_for(self, patched: bool) -> object:
        return self.patched if patched else self.faithful


#: Divergence ledger v1.  Class A entries (already the shipped default, no
#: toggle exists or should) carry ``toggleable=False`` and no key; they are
#: registered here so the ledger is one list rather than a table in a
#: document plus a dict in a module.  The prose register with each entry's
#: receipts and battery-arm status is ``PROVENANCE.md``, "Divergence ledger
#: v1"; this is its executable half and the two are checked against each
#: other by tests/test_physics_mode.py.
LEDGER: tuple[LedgerEntry, ...] = (
    LedgerEntry(
        entry_id="L1",
        key="",
        faithful=None,
        patched=None,
        summary=(
            "Thompson rain-presence gate reads a mass concentration, as WRF "
            "does (woof/core/kernels/thompson.cu:450-452, commit cb765336)"),
        toggleable=False,
        gate=(
            "L1 is class A: the correction is the shipped mp8 path and there "
            "is one mp8 path, so there is nothing to toggle and no arm to "
            "run.  It is registered because the roadmap listed it as a "
            "candidate and the register has to say it was overtaken."),
    ),
    LedgerEntry(
        entry_id="L2",
        key="",
        faithful=None,
        patched=None,
        summary=(
            "RUC LSM runs the DEFINED one-layer answer where WRF v4.6.1 "
            "reads an uninitialised ilnb (woof/core/ruc.py:8717-8754)"),
        toggleable=False,
        gate=(
            "L2 is class A and stays that way: a forecast-mode toggle back "
            "onto WRF's uninitialised read would be bit-exactness to a bug, "
            "which is forbidden.  The WRF-bug chain exists only as an "
            "oracle-fixture affordance and is refused on device."),
    ),
    LedgerEntry(
        entry_id="L3",
        key="bl_pbl_physics",
        faithful=1,
        patched=11,
        summary=(
            "gray-zone PBL: YSU (1) -> Shin-Hong 2015 scale-aware PBL (11)"),
    ),
    LedgerEntry(
        entry_id="L4",
        key="moist_mix6_off",
        faithful=False,
        patched=True,
        summary=(
            "6th-order horizontal filter off the moist scalars: WRF's own "
            "&dynamics moist_mix6_off, .false. -> .true."),
    ),
    LedgerEntry(
        entry_id="L5",
        key="bl_pbl_physics",
        faithful=1,
        patched=900,
        summary="SASE unified closure in place of YSU + km_opt = 4",
        toggleable=False,
        gate=(
            "L5 has a registered ENTRY GATE and has not passed it: SASE "
            "earns battery arms only after one full 24 h battery-shape "
            "integration completes with status PASS.  It is also not a "
            "bit-flip -- the closure carries six mandatory companion "
            "selectors -- so it enters as its own overlay, never as a "
            "member of a patch-set vector."),
    ),
    LedgerEntry(
        entry_id="L6",
        key="morr_rimed_ice",
        faithful=1,
        patched=0,
        summary=(
            "Morrison dense-ice identity: hail (1, the WRF default) vs "
            "graupel (0)"),
        toggleable=False,
        gate=(
            "L6 is class C -- dormant.  The knob exists and is validated, "
            "but no mp10 arm is registered in the battery suite, so it is "
            "not a member of any patch-set vector.  It wakes if an mp10 arm "
            "is ever registered; hail-vs-graupel is then its ablation pair."),
    ),
    LedgerEntry(
        entry_id="L7",
        key="",
        faithful=None,
        patched=None,
        summary="gray-zone partition on the SASE side (Phase 0+)",
        toggleable=False,
        gate=(
            "L7 is class C -- dormant, and there is nothing to toggle at "
            "this tip.  It enters the ledger by its own receipt when it "
            "lands."),
    ),
)

LEDGER_BY_ID: dict[str, LedgerEntry] = {e.entry_id: e for e in LEDGER}


@dataclass(frozen=True)
class PatchSet:
    """A registered, versioned patch-set: an ordered list of entry ids.

    Frozen by version, never edited in place.  A later ledger entry gets a
    NEW version beside this one so a receipt reading ``patchset = "v1"``
    keeps meaning exactly the vector it meant on the day it was written.
    """

    version: str
    entries: tuple[str, ...]
    rationale: str = ""

    def ledger_entries(self) -> tuple[LedgerEntry, ...]:
        return tuple(LEDGER_BY_ID[eid] for eid in self.entries)


#: Registered patch-set versions.  v1 is {L3, L4}: small, accurate, and each
#: with a real counterfactual.  L5 is conditional on its entry gate and is
#: therefore deliberately NOT a member.
PATCHSETS: dict[str, PatchSet] = {
    "v1": PatchSet(
        version="v1",
        entries=("L3", "L4"),
        rationale=(
            "The two toggleable ledger entries that are bit-flips with a "
            "real counterfactual at 3 km.  L5 (SASE) is excluded until its "
            "entry gate passes; L1/L2 are already the shipped default and "
            "L6/L7 are dormant."),
    ),
}


@dataclass(frozen=True)
class ResolvedPatch:
    """One entry as this run resolved it."""

    entry_id: str
    key: str
    value: object
    patched: bool
    summary: str

    def receipt(self) -> dict:
        return {
            "entry": self.entry_id,
            "key": self.key,
            "value": self.value,
            "patched": self.patched,
            "summary": self.summary,
        }


@dataclass(frozen=True)
class PhysicsModeResolution:
    """The whole axis as one run resolved it, ready for a receipt.

    ``governed`` is False for the absent-key state: the mode is reported
    (and it is ``wrf-faithful``, truthfully) but the axis authored no key
    and no configuration value was replaced.
    """

    mode: str
    patchset: str | None
    requested_patches: tuple[str, ...]
    resolved: tuple[ResolvedPatch, ...]
    governed: bool = True

    @property
    def settings(self) -> dict:
        """key -> resolved value, the overlay the loader writes per domain."""

        return {patch.key: patch.value for patch in self.resolved}

    def receipt(self) -> dict:
        return {
            "schema": "gpuwm-physics-mode-v1",
            "physics_mode": self.mode,
            "patchset": self.patchset,
            "governed": self.governed,
            "requested_patches": list(self.requested_patches),
            "resolved_patch_vector": [p.receipt() for p in self.resolved],
        }


#: The state of every configuration that does not name the axis: the mode
#: is reported (and ``wrf-faithful`` is true of it -- no registered patch is
#: applied) but no key is authored and no value is replaced.
UNGOVERNED = PhysicsModeResolution(
    mode=PHYSICS_MODE_WRF_FAITHFUL, patchset=None, requested_patches=(),
    resolved=(), governed=False)


def governed_keys(mode: str, patchset: str | None = None,
                  patches=None) -> tuple[str, ...]:
    """The configuration keys the axis authors under this selection.

    Public because a battery arm composer has to strip exactly these keys
    from a case TOML before adding the mode overlay, and a composer that
    guesses the list is a composer that silently stops stripping one the day
    the ledger grows.
    """

    return tuple(
        patch.key
        for patch in resolve(mode, patchset, patches, source="<query>").resolved
    )


def resolve(mode, patchset=None, patches=None, *,
            source: str) -> PhysicsModeResolution:
    """Resolve the axis, refusing every ambiguous or unregistered request.

    ``source`` names the configuration file in every refusal, matching the
    experiment loader's convention.
    """

    if not isinstance(mode, str) or mode not in PHYSICS_MODES:
        raise ValueError(
            f"physics_mode = {mode!r} in [experiment] of {source} is not a "
            f"registered fidelity mode; known modes: {list(PHYSICS_MODES)}.")

    faithful = mode == PHYSICS_MODE_WRF_FAITHFUL
    if patchset is None:
        resolved_patchset = DEFAULT_PATCHSET
    else:
        if not isinstance(patchset, str):
            raise ValueError(
                f"patchset in [experiment] of {source} must be a registered "
                f"patch-set version string, got {patchset!r}.")
        if patchset not in PATCHSETS:
            raise ValueError(
                f"patchset = {patchset!r} in [experiment] of {source} is not "
                "a registered patch-set version; known versions: "
                f"{sorted(PATCHSETS)}.  A patch-set version is frozen when "
                "it is registered, so a run's receipt keeps meaning the "
                "vector it meant -- name an existing version or register a "
                "new one beside it.")
        resolved_patchset = patchset

    registered = PATCHSETS[resolved_patchset]
    known_ids = registered.entries

    if patches is None:
        requested: tuple[str, ...] = known_ids
        explicit_subset = False
    else:
        if not isinstance(patches, (list, tuple)):
            raise ValueError(
                f"patches in [experiment] of {source} must be an array of "
                f"divergence-ledger entry ids, got {patches!r}.")
        seen: list[str] = []
        for index, value in enumerate(patches):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"patches[{index}] in [experiment] of {source} must be a "
                    f"non-empty divergence-ledger entry id, got {value!r}.")
            if value in seen:
                raise ValueError(
                    f"patches in [experiment] of {source} names {value!r} "
                    "twice; each ledger entry appears at most once.")
            entry = LEDGER_BY_ID.get(value)
            if entry is None:
                raise ValueError(
                    f"patches in [experiment] of {source} names {value!r}, "
                    "which is not a divergence-ledger entry; known entries: "
                    f"{sorted(LEDGER_BY_ID)}.")
            if not entry.toggleable:
                raise ValueError(
                    f"patches in [experiment] of {source} names {value!r}, "
                    f"which is not a toggleable patch. {entry.gate}")
            if value not in known_ids:
                raise ValueError(
                    f"patches in [experiment] of {source} names {value!r}, "
                    f"which is not a member of patch-set {resolved_patchset!r} "
                    f"(members: {list(known_ids)}).  Register a new patch-set "
                    "version rather than widening a frozen one.")
            seen.append(value)
        requested = tuple(seen)
        explicit_subset = True

    # The faithful arm resolves the SAME entry list to the faithful side of
    # each edge, so both arms govern an identical key set and one case file
    # composed for either composes for both.  ``patchset`` therefore names
    # the governed set in either mode; only ``patches`` -- a subset of the
    # patches APPLIED -- is meaningless when none is applied.
    if faithful and explicit_subset:
        raise ValueError(
            f"patches in [experiment] of {source} is set with physics_mode = "
            f"{PHYSICS_MODE_WRF_FAITHFUL!r}, which applies no patch; name "
            "the arm's patches under physics_mode = "
            f"{PHYSICS_MODE_ARWEN_PATCHED!r}.")

    entries = tuple(LEDGER_BY_ID[eid] for eid in known_ids)
    resolved = tuple(
        ResolvedPatch(
            entry_id=entry.entry_id, key=entry.key,
            value=entry.value_for(not faithful and entry.entry_id in requested),
            patched=(not faithful) and entry.entry_id in requested,
            summary=entry.summary)
        for entry in entries)
    if len({patch.key for patch in resolved}) != len(resolved):
        # Two entries of one patch-set sharing a key would make the
        # resolved vector order-dependent, which a receipt cannot express.
        raise ValueError(
            f"patch-set {resolved_patchset!r} resolves two entries onto one "
            "configuration key; a patch-set's entries must name distinct "
            "keys.")
    return PhysicsModeResolution(
        mode=mode,
        patchset=resolved_patchset,
        requested_patches=requested if not faithful else (),
        resolved=resolved,
        governed=True)


__all__ = [
    "DEFAULT_PATCHSET", "LEDGER", "LEDGER_BY_ID", "LedgerEntry",
    "PATCHSETS", "PHYSICS_MODES", "PHYSICS_MODE_ARWEN_PATCHED",
    "PHYSICS_MODE_WRF_FAITHFUL", "PatchSet", "PhysicsModeResolution",
    "ResolvedPatch", "UNGOVERNED", "governed_keys", "resolve",
]
