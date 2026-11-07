"""Discrete relocation of a child domain to a new parent-cell placement.

WHAT THIS IS. A moving nest here is a sequence of static nests joined by
a re-grid at synchronized integration-leg boundaries. These are internal
forecast boundaries: a 900-second follower checks within an ensemble
member's forecast, without waiting for a data-assimilation cycle boundary.
The distinction is the whole design.  WRF's mechanism moves the child by
one parent cell inside the integration and therefore invalidates the SINT
donor index/weight tables constantly; this module moves the child at a
leg boundary, rebuilds the tables **once per placement generation**, and
leaves the per-step numerics exactly as they were.  That is why the
namelist keys guarding the per-step mechanism stay rejected
(``woof.experiment._MOVING_NEST_KEYS``) while this path is admissible:
the premise those keys protect -- "the donor tables are precomputed once at
setup" -- is preserved, with "setup" reading "per placement generation".

THE ARITHMETIC THAT MAKES IT EXACT.  A placement is a whole number of
PARENT cells, so a move of ``di`` parent cells is a move of ``di * ratio``
child cells: an exact shift in child index space.  Write the WRF donor
pickup (``interp_fcn.F:975-985``) for child index ``n`` under placement
``p``::

    ci = p + (n-1)/ratio        ip = mod(n-1, ratio)

For the new placement ``p1`` and child index ``n'``, the cell covering the
same ground under the old placement ``p0`` is ``n = n' + (p1-p0)*ratio``,
and substituting gives ``ci`` and ``ip`` **identical** in both.  Two
consequences follow, and both are essential:

1. the overlap transplant is a pure index-space copy -- no interpolation,
   no resampling, no loss; and
2. anything the child derives from its parent by SINT (its base state, its
   terrain) is *bitwise identical* on the overlap before and after the
   move, because it is the same donor cell with the same sub-cell offset.

Property (2) is checked at every relocation (:func:`donor_alignment_check`)
and is an independent instrument: it is computed by the SINT operator, not
by the shift arithmetic this module implements, so a wrong shift cannot
make it pass.

WHAT MOVES AND WHAT DOES NOT.  The parent is read and never written --
:func:`parent_only_init` fills the child out-of-place, so the relocation
inherits the coupler's registered non-mutating property unchanged.  On the
child, the newly covered strip is filled by the ordinary cold-start path
(a full SINT of the live parent) and the overlap is then stamped bitwise
from the outgoing child, so the child keeps every fine-scale structure it
had integrated over ground it still covers.  The seam between strip and
overlap is real and is what the placement policy's "lead the storm, do not
chase it" bias exists to keep out of the way; it is reported as
``spin_up_cells`` rather than hidden.

WHAT THIS MODULE DELIBERATELY DOES NOT DO.  It does not decide *when* or
*where* to move -- there is no tracker, no scoring, no hysteresis here.
Placement is an input.

WHAT A RESTART ACROSS A MOVE PROMISES: THE RUN THAT WROTE IT, BIT FOR
BIT.  It promised NOTHING under the 2026-08-06 ruling -- *"a restart
across a move promises nothing imo its a pure efficiency experiment"* --
and that was a true statement about the machinery as it then stood,
which carried none of the three things a resume needs across a
checkpoint: the placement and the tracker's hysteresis, the acoustic
Omega, and the tracker's consultation window.  All three are carried
now, the resume reproduces the unbroken run bit for bit, and the posture
moved with them.  :data:`RESTART_ACROSS_MOVE_POSTURE` below is the
current wording and is the only text in this module allowed to state it.

Moving nests still exist so that a low-VRAM card can run a smaller nest
that FOLLOWS the weather at a resolution a static nest of the same cost
could never reach; a resume is a second thing they can now do, not the
reason they exist.  WHAT DID NOT CHANGE IS THE IDENTITY.  The
prepared-cache key and the tree restart fingerprint still bind the
placement, a move still invalidates both against a FRESH build, and a
moved checkpoint resumes only by replaying its own chain of move digests
back into that fingerprint -- so the gate keeps its full meaning and
merely stops being unpassable for the run that wrote it.

Determinism WITHIN a placement is unaffected and still holds -- not
because anything here protects it, but because nothing here touches the
machinery that provides it.  :class:`Placement` and
:class:`RelocationSegment` still record what happened for receipts and
diagnostics, but the record digests they carry are what the resume
replays, so a lenient reader of one is a correctness defect and not a
cosmetic one.  See ``docs/nest-relocation-identity-decision.md``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from dataclasses import dataclass, replace

import numpy as np

#: Versioned label for the relocation receipt, hashed into segment ids.
RELOCATION_CONTRACT = "gpuwm-nest-relocation.v1"

#: The posture, spelled once so every receipt that has to state it
#: states the same words.  Stamped into checkpoint headers written after a
#: move and surfaced by any restore that reads one.
#:
#: It used to say a restart across a move promised NOTHING -- "a pure
#: efficiency experiment, never certified, never comparable to an
#: unbroken run" (project ruling, 2026-08-06) -- and the resume was gated behind
#: --allow-restart-across-move on the strength of it.  That was true of
#: the machinery as it stood: three things a resume needs were not
#: carried.  All three are now (the placement and the tracker's
#: hysteresis in the relocation header, the acoustic Omega in
#: CHECKPOINT_ONLY_STATE, the consultation window in
#: restart.CARRIED_SCRATCH_SLOTS), the resume is bit-for-bit against the
#: unbroken run, and the gate is gone.  What remains is a statement of
#: what the checkpoint IS, which a header should carry anyway.
RESTART_ACROSS_MOVE_POSTURE = (
    "this checkpoint was written after a nest relocation; its identity is "
    "chained to the move history, so it resumes only into the run that "
    "wrote it -- and it reproduces that run bit for bit")

#: Placement is Layer 2 of the three-layer identity.  These are the
#: :class:`woof.experiment.DomainConfig` fields it owns, and therefore the
#: fields a *placement-independent* domain identity must not contain.
PLACEMENT_FIELDS = ("i_parent_start", "j_parent_start")


class RelocationRefusal(ValueError):
    """A relocation this module will not perform."""


# ---------------------------------------------------------------------------
# Layer 2 of the three-layer identity: Placement
# ---------------------------------------------------------------------------

#: The keys of :meth:`Placement.to_json`, so the reader is total.
PLACEMENT_JSON_KEYS = ("grid_id", "i_parent_start", "j_parent_start",
                       "generation")


def _exactly(payload, allowed, what: str) -> dict:
    """Every key honored or refused, for anything read back by digest.

    A key quietly dropped or silently accepted changes the SHA-256 the
    chain addresses itself by, and a resumed run re-marks its restart
    fingerprint by folding those digests in order -- so a lenient reader
    produces a fingerprint that matches neither the checkpoint nor a fresh
    build, and the mismatch surfaces as an unexplained refusal much later.
    """
    if not isinstance(payload, dict):
        raise RelocationRefusal(
            f"{what} must be a mapping, got {type(payload).__name__}")
    extra = sorted(set(payload) - set(allowed))
    missing = sorted(set(allowed) - set(payload))
    if extra or missing:
        raise RelocationRefusal(
            f"{what} has key(s) {extra} this build does not know and is "
            f"missing {missing}; it is read in full or refused, because "
            "every key is part of the digest this chain is addressed by")
    return payload


@dataclass(frozen=True)
class Placement:
    """Where a child sits in its parent, and which generation that is.

    ``generation`` counts relocations, not steps: it is 0 for a domain that
    has never moved and increments once per accepted move.  It exists so a
    receipt can say "same domain, third position" without a hash over the
    position having to change meaning.
    """

    grid_id: int
    i_parent_start: int
    j_parent_start: int
    generation: int = 0

    def __post_init__(self) -> None:
        for name in ("i_parent_start", "j_parent_start"):
            if int(getattr(self, name)) < 1:
                raise RelocationRefusal(
                    f"{name} is 1-based WRF namelist semantics and must be "
                    f">= 1, got {getattr(self, name)!r}")
        if int(self.generation) < 0:
            raise RelocationRefusal(
                f"placement generation must be >= 0, got {self.generation!r}")

    @property
    def position(self) -> tuple[int, int]:
        return (int(self.i_parent_start), int(self.j_parent_start))

    def same_position(self, other: "Placement") -> bool:
        """Position equality, deliberately ignoring the generation counter."""
        return self.position == other.position

    def to_json(self) -> dict[str, int]:
        return {
            "grid_id": int(self.grid_id),
            "i_parent_start": int(self.i_parent_start),
            "j_parent_start": int(self.j_parent_start),
            "generation": int(self.generation),
        }

    @classmethod
    def from_json(cls, payload) -> "Placement":
        """Read back :meth:`to_json`, generation included."""
        _exactly(payload, PLACEMENT_JSON_KEYS, "a placement")
        return cls(grid_id=int(payload["grid_id"]),
                   i_parent_start=int(payload["i_parent_start"]),
                   j_parent_start=int(payload["j_parent_start"]),
                   generation=int(payload["generation"]))


def placement_of(domain_config, *, generation: int = 0) -> Placement:
    """Read the placement off a :class:`~woof.experiment.DomainConfig`."""
    return Placement(
        grid_id=int(domain_config.grid_id),
        i_parent_start=int(domain_config.i_parent_start),
        j_parent_start=int(domain_config.j_parent_start),
        generation=int(generation))


def placement_independent_identity(domain_config) -> dict[str, object]:
    """The domain identity with :data:`PLACEMENT_FIELDS` factored out.

    This is Layer 1 ("are these two states the same kind of thing?") of the
    architecture's three-layer identity, computed from the SAME document
    the prepared-cache identity is computed from so the two cannot drift.
    Two domain configs with equal placement-independent identity describe
    the same grid, physics and timestep, and differ at most in where the
    child sits inside its parent.

    NOTHING IN THE TREE CONSUMES THIS, and nothing needs to even now that
    a moved checkpoint resumes.  The resume does not see PAST the
    placement; it replays the move chain back into the fingerprint, which
    reaches the same value by the same route the run took.  So the
    prepared-cache key and the tree restart fingerprint still bind the
    placement and still invalidate against a fresh build, which is the
    intended behaviour rather than a blocker.  This function exists for
    comparability questions and for the segment bookkeeping below -- "are
    these two configs the same domain in different places?" -- not to
    relax a refusal.  See ``docs/nest-relocation-identity-decision.md``.
    """
    from woof.ingest.prepared_cache import prepared_domain_config_identity

    document = dict(prepared_domain_config_identity(domain_config))
    removed = {name: document.pop(name, None) for name in PLACEMENT_FIELDS}
    return {"identity": document, "placement_fields_removed": removed}


# ---------------------------------------------------------------------------
# Layer 3: the append-only relocation record and segment
# ---------------------------------------------------------------------------

def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


#: The keys of :meth:`RelocationRecord.to_json`, derived fields included.
RECORD_JSON_KEYS = (
    "contract", "placement_from", "placement_to", "parent_grid_ratio",
    "shift_child_cells", "overlap_cells", "child_cells", "overlap_fraction",
    "spin_up_cells", "null_move", "parent_state_sha256",
    "child_state_sha256_before", "child_state_sha256_after",
    "donor_alignment", "predecessor_sha256")

#: The keys of :meth:`RelocationSegment.to_json`.
SEGMENT_JSON_KEYS = ("contract", "base_identity_sha256", "segment_id",
                     "generation", "records")


@dataclass(frozen=True)
class RelocationRecord:
    """One accepted move, recorded for the receipt.

    A record names its predecessor's digest so a sequence of moves reads
    as an ordered history rather than a set, which is what makes a
    receipt legible after the fact -- and, since the resume became exact,
    what a restore replays to rebuild the fingerprint the run had.  The
    PRESENCE of a chain is still not permission to resume: what permits
    it is replaying these digests, in this order, onto this build's own
    fingerprint and arriving at the checkpoint's value.  A different
    config still mismatches, because the base of the chain differs.
    """

    placement_from: Placement
    placement_to: Placement
    parent_grid_ratio: int
    shift_i: int
    shift_j: int
    overlap_cells: int
    child_cells: int
    null_move: bool
    parent_state_sha256: str | None = None
    child_state_sha256_before: str | None = None
    child_state_sha256_after: str | None = None
    donor_alignment: dict | None = None
    predecessor_sha256: str | None = None

    @property
    def overlap_fraction(self) -> float:
        return (0.0 if self.child_cells == 0
                else self.overlap_cells / self.child_cells)

    @property
    def spin_up_cells(self) -> int:
        return int(self.child_cells) - int(self.overlap_cells)

    def to_json(self) -> dict[str, object]:
        return {
            "contract": RELOCATION_CONTRACT,
            "placement_from": self.placement_from.to_json(),
            "placement_to": self.placement_to.to_json(),
            "parent_grid_ratio": int(self.parent_grid_ratio),
            "shift_child_cells": [int(self.shift_i), int(self.shift_j)],
            "overlap_cells": int(self.overlap_cells),
            "child_cells": int(self.child_cells),
            "overlap_fraction": float(self.overlap_fraction),
            "spin_up_cells": int(self.spin_up_cells),
            "null_move": bool(self.null_move),
            "parent_state_sha256": self.parent_state_sha256,
            "child_state_sha256_before": self.child_state_sha256_before,
            "child_state_sha256_after": self.child_state_sha256_after,
            "donor_alignment": self.donor_alignment,
            "predecessor_sha256": self.predecessor_sha256,
        }

    @classmethod
    def from_json(cls, payload) -> "RelocationRecord":
        """Read back :meth:`to_json` to the SAME :attr:`sha256`.

        The round-trip is asserted rather than assumed: the reconstructed
        record must re-serialize to the bytes it was read from, which is
        what makes ``segment_id`` and the fingerprint chain reproducible.
        """
        _exactly(payload, RECORD_JSON_KEYS, "a relocation record")
        contract = payload["contract"]
        if contract != RELOCATION_CONTRACT:
            raise RelocationRefusal(
                f"a relocation record names contract {contract!r}, not "
                f"{RELOCATION_CONTRACT!r}; a record's digest is taken over "
                "its own JSON, so reading a foreign one produces a "
                "segment_id and a restart fingerprint that address nothing "
                "on either side of the resume")
        shift = tuple(int(value) for value in payload["shift_child_cells"])
        if len(shift) != 2:
            raise RelocationRefusal(
                "a relocation record's shift_child_cells must be "
                f"[shift_i, shift_j], got {payload['shift_child_cells']!r}")
        record = cls(
            placement_from=Placement.from_json(payload["placement_from"]),
            placement_to=Placement.from_json(payload["placement_to"]),
            parent_grid_ratio=int(payload["parent_grid_ratio"]),
            shift_i=shift[0], shift_j=shift[1],
            overlap_cells=int(payload["overlap_cells"]),
            child_cells=int(payload["child_cells"]),
            null_move=bool(payload["null_move"]),
            parent_state_sha256=payload["parent_state_sha256"],
            child_state_sha256_before=payload["child_state_sha256_before"],
            child_state_sha256_after=payload["child_state_sha256_after"],
            donor_alignment=payload["donor_alignment"],
            predecessor_sha256=payload["predecessor_sha256"])
        if record.to_json() != dict(payload):
            raise RelocationRefusal(
                "a relocation record does not re-serialize to the bytes it "
                "was read from, so its sha256 would move; the derived "
                "fields (overlap_fraction, spin_up_cells) disagree with the "
                "cell counts beside them, which means the payload was "
                "edited rather than written by a run")
        return record

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical(self.to_json())).hexdigest()


@dataclass(frozen=True)
class RelocationSegment:
    """A base preparation plus the append-only chain of moves off it.

    ``base_identity`` is the placement-independent identity of the domain
    as prepared, and every record after it says how the live domain got
    from that preparation to where it is now.  ``segment_id`` changes with
    every move, so a receipt can name the exact placement history a state
    came out of.

    THE SEGMENT ID IS AN ADDRESS, NOT A LICENCE.  It names a history so
    a receipt can say which moves produced the state in front of you.  A
    resume across a placement boundary is permitted by replaying the
    records' digests into the restart fingerprint and matching the
    checkpoint's value, never by two ``segment_id`` strings comparing
    equal -- so nothing may shortcut the replay by reading this field.
    """

    base_identity_sha256: str
    records: tuple[RelocationRecord, ...] = ()

    @property
    def generation(self) -> int:
        return len(self.records)

    @property
    def segment_id(self) -> str:
        digest = hashlib.sha256(RELOCATION_CONTRACT.encode("ascii") + b"\0")
        digest.update(self.base_identity_sha256.encode("ascii") + b"\0")
        for record in self.records:
            digest.update(record.sha256.encode("ascii") + b"\0")
        return digest.hexdigest()

    def append(self, record: RelocationRecord) -> "RelocationSegment":
        """Return the successor segment, chaining the predecessor digest."""
        predecessor = (self.records[-1].sha256 if self.records
                       else self.base_identity_sha256)
        linked = replace(record, predecessor_sha256=predecessor)
        return RelocationSegment(
            base_identity_sha256=self.base_identity_sha256,
            records=(*self.records, linked))

    def to_json(self) -> dict[str, object]:
        return {
            "contract": RELOCATION_CONTRACT,
            "base_identity_sha256": self.base_identity_sha256,
            "segment_id": self.segment_id,
            "generation": self.generation,
            "records": [record.to_json() for record in self.records],
        }

    @classmethod
    def from_json(cls, payload) -> "RelocationSegment":
        """Read back :meth:`to_json`, with the recorded digest as the check.

        Restoring a chain is what lets a resumed follower's NEXT move
        append to its real predecessor instead of to the base preparation,
        so the chain has to come back exactly: the recomputed
        ``segment_id`` is compared against the one the writer recorded, and
        a chain that has lost, gained or edited a record refuses here
        rather than producing a plausible-looking history addressed by a
        digest nothing else agrees with.
        """
        _exactly(payload, SEGMENT_JSON_KEYS, "a relocation segment")
        contract = payload["contract"]
        if contract != RELOCATION_CONTRACT:
            raise RelocationRefusal(
                f"a relocation segment names contract {contract!r}, not "
                f"{RELOCATION_CONTRACT!r}; this build cannot say what a "
                "chain under another contract addresses")
        records = payload["records"]
        if not isinstance(records, (list, tuple)):
            raise RelocationRefusal(
                "a relocation segment's records must be an ordered list, "
                f"got {type(records).__name__}; a chain read out of order "
                "is a set, and a set has no predecessor")
        segment = cls(
            base_identity_sha256=str(payload["base_identity_sha256"]),
            records=tuple(RelocationRecord.from_json(row)
                          for row in records))
        if segment.generation != int(payload["generation"]):
            raise RelocationRefusal(
                f"a relocation segment records generation "
                f"{int(payload['generation'])} but carries "
                f"{segment.generation} move(s); the counter and the chain "
                "disagree about how far the nest has moved from its base "
                "preparation")
        if segment.segment_id != str(payload["segment_id"]):
            raise RelocationRefusal(
                "a relocation chain does not reproduce the segment_id it "
                f"was written under ({payload['segment_id']}; recomputed "
                f"{segment.segment_id}), so a record has been dropped, "
                "added or edited. A resumed run re-marks its restart "
                "fingerprint by folding these record digests in order, and "
                "an unfaithful chain would admit a checkpoint that came out "
                "of a different placement history")
        return segment


def base_segment(domain_config) -> RelocationSegment:
    """The zero-move segment for a domain as prepared."""
    identity = placement_independent_identity(domain_config)
    return RelocationSegment(
        base_identity_sha256=hashlib.sha256(
            _canonical(identity["identity"])).hexdigest())


# ---------------------------------------------------------------------------
# The plan: pure index-space geometry, no device, no state
# ---------------------------------------------------------------------------

def _axis_window(extent: int, shift: int):
    """``dst[d] <- src[d + shift]`` over the valid range, or ``None``.

    Source and destination extents are equal because a relocation changes
    where the child is, never how big it is.
    """
    extent = int(extent)
    shift = int(shift)
    dst_lo = max(0, -shift)
    dst_hi = min(extent, extent - shift)
    if dst_lo >= dst_hi:
        return None
    return slice(dst_lo, dst_hi), slice(dst_lo + shift, dst_hi + shift)


@dataclass(frozen=True)
class RelocationPlan:
    """Everything a relocation needs that does not touch the device."""

    placement_from: Placement
    placement_to: Placement
    parent_grid_ratio: int
    child_nx: int
    child_ny: int
    shift_i: int
    shift_j: int

    @property
    def null_move(self) -> bool:
        return self.shift_i == 0 and self.shift_j == 0

    @property
    def child_cells(self) -> int:
        return int(self.child_nx) * int(self.child_ny)

    @property
    def overlap_cells(self) -> int:
        window = self.window((self.child_ny, self.child_nx))
        if window is None:
            return 0
        (dst_j, _), (dst_i, _) = window
        return ((dst_j.stop - dst_j.start) * (dst_i.stop - dst_i.start))

    @property
    def overlap_fraction(self) -> float:
        return (0.0 if self.child_cells == 0
                else self.overlap_cells / self.child_cells)

    @property
    def disjoint(self) -> bool:
        return self.overlap_cells == 0

    def window(self, shape):
        """Destination/source slice pairs for one array's trailing (y, x).

        Returns ``((dst_j, src_j), (dst_i, src_i))`` or ``None`` when the
        old and new footprints do not overlap at all.  Shape-driven rather
        than stagger-driven on purpose: a staggered extent is ``n+1`` and
        shifts by the same whole number of child cells as the mass extent,
        so reading the extent off the array handles every stagger without
        this module having to know which one it is looking at.
        """
        shape = tuple(int(n) for n in shape)
        if len(shape) < 2:
            raise RelocationRefusal(
                f"a relocatable field needs at least (ny, nx), got {shape}")
        rows = _axis_window(shape[-2], self.shift_j)
        cols = _axis_window(shape[-1], self.shift_i)
        if rows is None or cols is None:
            return None
        return (rows, cols)

    def to_json(self) -> dict[str, object]:
        return {
            "contract": RELOCATION_CONTRACT,
            "placement_from": self.placement_from.to_json(),
            "placement_to": self.placement_to.to_json(),
            "parent_grid_ratio": int(self.parent_grid_ratio),
            "child_nx": int(self.child_nx), "child_ny": int(self.child_ny),
            "shift_child_cells": [int(self.shift_i), int(self.shift_j)],
            "shift_parent_cells": [
                int(self.shift_i) // int(self.parent_grid_ratio),
                int(self.shift_j) // int(self.parent_grid_ratio)],
            "null_move": bool(self.null_move),
            "overlap_cells": int(self.overlap_cells),
            "child_cells": int(self.child_cells),
            "overlap_fraction": float(self.overlap_fraction),
            "spin_up_cells": int(self.child_cells - self.overlap_cells),
        }


def plan_relocation(*, placement_from: Placement, placement_to: Placement,
                    parent_grid_ratio: int, child_nx: int,
                    child_ny: int) -> RelocationPlan:
    """Resolve a move into an index-space shift.

    A null move (``placement_to`` at the same position) is admissible and
    planned exactly like any other; its shift is zero, its overlap is the
    whole child, and the resulting transplant is the identity.  That is
    deliberate -- the null move is the calibration point of the whole
    mechanism, and special-casing it would remove the only case whose
    answer is known in advance.
    """
    if placement_from.grid_id != placement_to.grid_id:
        raise RelocationRefusal(
            f"a relocation moves ONE domain: from grid_id "
            f"{placement_from.grid_id} to {placement_to.grid_id}")
    ratio = int(parent_grid_ratio)
    if ratio < 1:
        raise RelocationRefusal(
            f"parent_grid_ratio must be >= 1, got {parent_grid_ratio!r}")
    return RelocationPlan(
        placement_from=placement_from, placement_to=placement_to,
        parent_grid_ratio=ratio,
        child_nx=int(child_nx), child_ny=int(child_ny),
        shift_i=(placement_to.i_parent_start
                 - placement_from.i_parent_start) * ratio,
        shift_j=(placement_to.j_parent_start
                 - placement_from.j_parent_start) * ratio)


#: Slack on the implied-bound floor division, so a floor whose exact
#: answer is a whole number of cells is not pushed one cell down by the
#: last bit of a square root.  Far below one cell, so it can never admit
#: a move the floor refuses.
_CELL_TOL = 1e-9


def overlap_fraction_for_shift(shift_i: int, shift_j: int, *,
                               parent_grid_ratio: int, child_nx: int,
                               child_ny: int) -> float:
    """The fraction of the child a whole-parent-cell shift would keep.

    Goes through :func:`plan_relocation` rather than repeating the
    arithmetic, so the number a caller checks a move against and the
    number :func:`check_admissible` judges it on are produced by one
    implementation.  The base position is chosen so both placements are
    1-based valid for any sign of the shift; overlap depends on the
    DIFFERENCE, so the base cancels.
    """
    base = 1 + max(abs(int(shift_i)), abs(int(shift_j)))
    grid_id = 2
    plan = plan_relocation(
        placement_from=Placement(grid_id=grid_id, i_parent_start=base,
                                 j_parent_start=base),
        placement_to=Placement(grid_id=grid_id,
                               i_parent_start=base + int(shift_i),
                               j_parent_start=base + int(shift_j),
                               generation=1),
        parent_grid_ratio=int(parent_grid_ratio),
        child_nx=int(child_nx), child_ny=int(child_ny))
    return float(plan.overlap_fraction)


def max_parent_cells_for_overlap(min_overlap_fraction, *,
                                 parent_grid_ratio: int, child_nx: int,
                                 child_ny: int, diagonal: bool = True) -> int | None:
    """The per-axis parent-cell move an overlap floor implies.

    The two bounds in :func:`check_admissible` are not independent, and a
    config that sets them independently can declare a maximum move its
    own floor refuses.  Overlap is separable: a shift of ``(m, n)``
    parent cells keeps ``(1 - m*r/nx) * (1 - n*r/ny)`` of the child, so
    the binding case is the DIAGONAL move, where both factors shrink at
    once.  With the nest ``N = n/r`` parent cells wide on its narrow
    axis, a floor ``f`` admits a per-axis magnitude ``m`` only while
    ``(1 - m/N)**2 >= f``, that is::

        m <= N * (1 - sqrt(f))

    so the per-axis bound is ``1 - sqrt(f)`` of the nest's own width in
    parent cells.  ``None`` in means unbounded and comes back as
    ``None``.  The result is never negative and a floor of 1.0 gives 0:
    a floor that keeps the whole child admits only the null move.

    ``diagonal=False`` returns the single-axis proposal ceiling
    ``N * (1 - f)``. Use it when the runner subsequently clamps the actual
    direction with :func:`clamp_shift_to_overlap`; this preserves greater
    axial reach without reducing the required overlap on diagonal moves.
    """
    if min_overlap_fraction is None:
        return None
    floor = float(min_overlap_fraction)
    ratio = int(parent_grid_ratio)
    narrow = min(int(child_nx), int(child_ny)) / ratio
    # A proposal ceiling need not assume a diagonal displacement: the
    # runner clamps the actual vector to the overlap floor. Applying the
    # diagonal bound to a zonal move needlessly halves its available reach.
    retained = math.sqrt(floor) if diagonal else floor
    return max(0, int(math.floor(narrow * (1.0 - retained) + _CELL_TOL)))


def clamp_shift_to_overlap(shift_i: int, shift_j: int, *,
                           min_overlap_fraction, parent_grid_ratio: int,
                           child_nx: int, child_ny: int
                           ) -> tuple[int, int, bool]:
    """Shrink a move along its own direction until it clears the floor.

    The largest move in the SAME direction that satisfies the floor, and
    the null move if nothing else does -- the null move always satisfies
    it, because it keeps the whole child.  So a follow source whose
    proposal breaches the floor gets the biggest admissible step toward
    the storm and catches up at later cadences, instead of the run ending
    on a bound it could have obeyed.

    Direction is preserved by scaling BOTH axes by the same factor and
    truncating toward zero, so a diagonal proposal stays diagonal and a
    single-axis one stays on its axis.  The walk is over the dominant
    axis's magnitude, which is finite and decreasing, so it terminates.
    """
    di, dj = int(shift_i), int(shift_j)
    if min_overlap_fraction is None:
        return di, dj, False
    floor = float(min_overlap_fraction)

    def keeps(candidate_i: int, candidate_j: int) -> bool:
        return overlap_fraction_for_shift(
            candidate_i, candidate_j, parent_grid_ratio=parent_grid_ratio,
            child_nx=child_nx, child_ny=child_ny) >= floor

    if keeps(di, dj):
        return di, dj, False
    peak = max(abs(di), abs(dj))
    for magnitude in range(peak - 1, -1, -1):
        scale = magnitude / peak
        candidate_i = int(di * scale)
        candidate_j = int(dj * scale)
        if keeps(candidate_i, candidate_j):
            return candidate_i, candidate_j, True
    return 0, 0, True


def check_admissible(plan: RelocationPlan, bounds) -> dict[str, object]:
    """Judge a planned move against a config's relocation bounds.

    ``bounds`` is a :class:`woof.experiment.RelocationConfig`.  A disabled
    one refuses every non-null move: a config that never opted in must not
    acquire a movable nest because a caller passed a placement.  The null
    move is always admissible, because it is not a move.
    """
    from woof.experiment import DISCRETE_RELOCATION_MODE

    if plan.null_move:
        return {"admissible": True, "null_move": True, "checks": {}}
    if not getattr(bounds, "enabled", False):
        raise RelocationRefusal(
            "this experiment does not enable nest relocation; a moving "
            "nest is opt-in, through [relocation] with enabled = true and "
            f"mode = {DISCRETE_RELOCATION_MODE!r}")
    grid_id = getattr(bounds, "grid_id", None)
    checks: dict[str, object] = {}
    if grid_id is not None and int(grid_id) != plan.placement_to.grid_id:
        raise RelocationRefusal(
            f"[relocation] authorises grid_id {grid_id}, but the move "
            f"targets grid_id {plan.placement_to.grid_id}")
    ratio = int(plan.parent_grid_ratio)
    move_cells = max(abs(plan.shift_i), abs(plan.shift_j)) // ratio
    limit = getattr(bounds, "max_move_parent_cells", None)
    checks["move_parent_cells"] = int(move_cells)
    checks["max_move_parent_cells"] = (
        None if limit is None else int(limit))
    if limit is not None and move_cells > int(limit):
        raise RelocationRefusal(
            f"the move is {move_cells} parent cells, over the configured "
            f"maximum of {limit}; a larger jump exposes a spin-up strip "
            "the child cannot recover before it matters")
    floor = getattr(bounds, "min_overlap_fraction", None)
    checks["overlap_fraction"] = float(plan.overlap_fraction)
    checks["min_overlap_fraction"] = (None if floor is None
                                      else float(floor))
    if floor is not None and plan.overlap_fraction < float(floor):
        implied = max_parent_cells_for_overlap(
            floor, parent_grid_ratio=ratio, child_nx=plan.child_nx,
            child_ny=plan.child_ny)
        checks["implied_max_parent_cells"] = implied
        raise RelocationRefusal(
            f"the move keeps only {plan.overlap_fraction:.4f} of the "
            f"child's cells, under the configured floor of "
            f"{float(floor):.4f}; the strip it would expose is larger "
            "than the one the child can spin up before it matters. On "
            "this nest the floor admits at most "
            f"{implied} parent cells per axis on a diagonal move, so ask "
            "for a placement inside that, or lower min_overlap_fraction "
            "deliberately. A scheduled follow source does not reach this "
            "refusal: the runner clamps its proposal to the largest move "
            "in the same direction that clears the floor "
            "(clamp_shift_to_overlap) and records both shifts.")
    return {"admissible": True, "null_move": False, "checks": checks}


# ---------------------------------------------------------------------------
# The transplant
# ---------------------------------------------------------------------------

def relocatable_attrs(state=None) -> tuple[str, ...]:
    """The state inventory a relocation carries across the move.

    The restart layer's serialised-state contract, and nothing invented
    beside it: whatever a checkpoint must carry to resume a domain is
    exactly what a relocation must carry to keep it.  Using one list means
    a field added to the model reaches both paths at once.

    Chemical species and serialized process fields follow the overlap.
    Domain ledger vectors keep the complete accounting history; the next
    chemistry step books the changed footprint's mass under transport.
    Geography-derived chemistry statics belong to the rebuilt footprint.
    """
    from woof.state_serialization_contract import (CHEM_DIAG_PREFIX,
                                                    CHEM_STATE_PREFIX,
                                                    STATE_SERIALIZED_ATTRS)

    # By attribute name, not through ``state.chem``: a host snapshot of a
    # moving child (HostStateSnapshot) carries the fields without the
    # ChemState object and lists them in ``field_names``.
    names = (state.field_names if isinstance(state, HostStateSnapshot)
             else (getattr(state, "__dict__", None) or {}))
    chem = tuple(sorted(name for name in names
                        if name.startswith((CHEM_STATE_PREFIX, CHEM_DIAG_PREFIX))))
    if any(name.startswith(CHEM_DIAG_PREFIX) for name in chem):
        from woof.core.chem_statics import (DUST_STATIC_ALLOCATIONS,
                                              SOLAR_ALLOCATIONS)
        rebuilt = {CHEM_DIAG_PREFIX + alloc.name
                   for alloc in (*DUST_STATIC_ALLOCATIONS, *SOLAR_ALLOCATIONS)}
        chem = tuple(name for name in chem if name not in rebuilt)
    thermal = (("thp0",) if getattr(state, "use_theta_m", 0)
               or isinstance(state, HostStateSnapshot) and "thp0" in names else ())
    return tuple(STATE_SERIALIZED_ATTRS) + thermal + chem


def _bit_mismatches(actual, expected) -> int:
    """Count differing 32-bit patterns, so signed zeros and NaNs count."""
    actual = np.ascontiguousarray(_host(actual))
    expected = np.ascontiguousarray(_host(expected))
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return max(int(actual.size), int(expected.size), 1)
    if actual.dtype == np.float32:
        return int(np.count_nonzero(
            actual.view(np.uint32) != expected.view(np.uint32)))
    return int(np.count_nonzero(actual != expected))


def _host(value) -> np.ndarray:
    get = getattr(value, "get", None)
    if callable(get) and hasattr(value, "__cuda_array_interface__"):
        return np.asarray(get())
    return np.asarray(value)


def transplant_overlap(*, source_state, target_state, plan: RelocationPlan,
                       attrs=None) -> dict[str, object]:
    """Stamp the outgoing child's state onto the incoming one, bitwise.

    ``target_state`` arrives already filled everywhere by the cold-start
    path (a full SINT of the live parent at the NEW placement), so this
    function only has to overwrite the part of it that the outgoing child
    already knows better.  Writing the whole field and then stamping the
    overlap, rather than filling only the strip, is not laziness: it makes
    the null move provably the identity, because at zero shift the stamp
    covers the entire field.
    """
    attrs = (relocatable_attrs(source_state) if attrs is None
             else tuple(attrs))
    stamped: dict[str, object] = {}
    skipped: dict[str, str] = {}
    for name in attrs:
        source = getattr(source_state, name, None)
        target = getattr(target_state, name, None)
        if source is None and target is None:
            continue
        if source is None or target is None:
            skipped[name] = ("absent on the outgoing child" if source is None
                             else "absent on the incoming child")
            continue
        if tuple(source.shape) != tuple(target.shape):
            raise RelocationRefusal(
                f"field {name!r} has shape {tuple(source.shape)} on the "
                f"outgoing child and {tuple(target.shape)} on the incoming "
                "one; a relocation changes position, never extent")
        if name.startswith("chemdiag_ledger_"):
            value = source
            if (hasattr(target, "__cuda_array_interface__")
                    and isinstance(value, np.ndarray)):
                import cupy as cp

                value = cp.asarray(np.ascontiguousarray(value))
            target[...] = value
            stamped[name] = {"shape": list(int(n) for n in source.shape),
                             "stamped_cells": 0, "scope": "domain",
                             "elements": int(source.size)}
            continue
        window = plan.window(source.shape)
        if window is None:
            skipped[name] = "old and new footprints are disjoint"
            continue
        (dst_j, src_j), (dst_i, src_i) = window
        chunk = source[..., src_j, src_i]
        if (hasattr(target, "__cuda_array_interface__")
                and isinstance(chunk, np.ndarray)):
            # Host-staged upload: CuPy's sliced __setitem__ refuses a
            # non-scalar NumPy right-hand side, so the H2D transfer is
            # explicit.  A byte transport either way.
            import cupy as cp

            chunk = cp.asarray(np.ascontiguousarray(chunk))
        target[..., dst_j, dst_i] = chunk
        stamped[name] = {
            "shape": list(int(n) for n in source.shape),
            "stamped_cells": int((dst_j.stop - dst_j.start)
                                 * (dst_i.stop - dst_i.start)),
        }
    return {
        "contract": RELOCATION_CONTRACT,
        "attrs_considered": list(attrs),
        "stamped": stamped,
        "skipped": skipped,
        "stamped_field_count": len(stamped),
    }


#: Base-state fields a child derives from its parent by SINT.  They are
#: NOT transplanted -- the cold start recomputes them at the new placement
#: -- which is precisely why comparing them across the move is a valid
#: independent check of the shift arithmetic (see module docstring).
_DONOR_ALIGNMENT_FIELDS = ("pb", "alb", "thb", "phb", "mub2d")

#: Environment switch for :func:`overlap_prognostic_mismatches`.  Off by
#: default because the check walks every stamped field over the whole
#: overlap, which is real work on a 2.6 M-cell child.
_LOG = logging.getLogger("gpuwm.nest_relocation")

RELOCATION_PROBE_ENV = "WOOF_RELOC_PROBE"


def relocation_probe_enabled() -> bool:
    return os.environ.get(RELOCATION_PROBE_ENV, "").strip() not in ("", "0")


def overlap_prognostic_mismatches(source_state, target_state, plan,
                                  attrs=None) -> dict[str, object]:
    """Per-field bitwise mismatch counts on the shared ground, PROGNOSTIC.

    THE CLAIM NOTHING WAS CHECKING.  ``overlap_statics_mismatches``
    asserts the rebuilt STATICS equal the outgoing child's, and
    ``donor_alignment_check`` asserts the rebuilt BASE state does
    (:data:`_DONOR_ALIGNMENT_FIELDS` is ``pb``/``alb``/``thb``/``phb``/
    ``mub2d`` and nothing else).  Between them they cover everything
    except the fields the forecast is actually made of -- and the
    transplant's whole promise is that those arrive bitwise.

    So this is the missing instrument, and it is deliberately usable at
    TWO points: straight after :func:`transplant_overlap`, where a
    nonzero count means the stamp itself is wrong, and again after the
    initializer's ``post_transplant`` hook, where a count that was zero
    and is now nonzero names a field that hook perturbed.  The pair is
    the diagnosis; either alone is only half of it.

    Returns the same shape as its statics sibling so a caller can log or
    refuse on either identically.
    """
    attrs = (relocatable_attrs(source_state) if attrs is None
             else tuple(attrs))
    fields: dict[str, int] = {}
    absent: list[str] = []
    compared_cells = 0
    for name in attrs:
        source = getattr(source_state, name, None)
        target = getattr(target_state, name, None)
        if source is None or target is None:
            if (source is None) != (target is None):
                absent.append(name)
            continue
        source = np.asarray(_host(source))
        target = np.asarray(_host(target))
        if source.shape != target.shape:
            fields[name] = max(int(source.size), int(target.size), 1)
            continue
        if name.startswith("chemdiag_ledger_"):
            fields[name] = _bit_mismatches(target, source)
            compared_cells += int(source.size)
            continue
        window = plan.window(source.shape)
        if window is None:
            continue
        (dst_j, src_j), (dst_i, src_i) = window
        actual = np.ascontiguousarray(target[..., dst_j, dst_i])
        expected = np.ascontiguousarray(source[..., src_j, src_i])
        fields[name] = _bit_mismatches(actual, expected)
        compared_cells += int(actual.size)
    return {
        "fields": fields,
        "absent_on_one_side": sorted(absent),
        "compared_cells": int(compared_cells),
        "mismatched_fields": {n: c for n, c in fields.items() if c},
        "pass": bool(compared_cells) and not any(fields.values()),
    }


def _interior_window(window, shape, shift_j: int, shift_i: int,
                     frame_width: int):
    """Shrink an overlap window to cells >= frame_width inside BOTH
    footprints, or ``None`` when nothing is left."""
    frame = int(frame_width)
    (dst_j, src_j), (dst_i, src_i) = window
    out = []
    for dst, shift, extent in ((dst_j, shift_j, shape[-2]),
                               (dst_i, shift_i, shape[-1])):
        lo = max(dst.start, frame, frame - shift)
        hi = min(dst.stop, extent - frame, extent - frame - shift)
        if lo >= hi:
            return None
        out.append((slice(lo, hi), slice(lo + shift, hi + shift)))
    return (out[0], out[1])


def donor_alignment_check(*, source_state, target_state,
                          plan: RelocationPlan,
                          frame_width: int = 0) -> dict[str, object]:
    """Prove the shift lands on the same donor cells, without using the shift.

    Every field here is produced by the SINT operator from the parent's
    (non-evolving) base state.  Under a placement change of whole parent
    cells the donor index and sub-cell offset of an overlapped child cell
    are unchanged, so these fields must agree BITWISE on the overlap.  They
    are computed independently of :meth:`RelocationPlan.window`, so a wrong
    shift shows up here as a nonzero mismatch count rather than as a
    plausible-looking field.

    A null move makes this a whole-field identity check, which is the
    stronger statement and is the reason the null move is run first.

    ``frame_width`` scopes the comparison to overlap cells at least that
    many cells inside BOTH footprints.  It exists for the real-data
    initializer, whose base fields are the t=0 machinery's -- analytic
    on the child's own terrain, blended toward the parent inside each
    footprint's ``spec_bdy_width + blend_width`` frame -- so the
    placement invariant genuinely holds only outside both frames.  The
    instrument stays armed there: a wrong shift still breaks the
    interior bitwise equality.  Zero keeps the exact historical
    whole-overlap comparison.
    """
    fields: dict[str, object] = {}
    for name in _DONOR_ALIGNMENT_FIELDS:
        source = getattr(source_state, name, None)
        target = getattr(target_state, name, None)
        if source is None or target is None:
            continue
        source_host = _host(source)
        target_host = _host(target)
        if source_host.ndim < 2:
            # A horizontally uniform base (idealized, terrain_opt off) has
            # nothing to align; recording that is more useful than a skip.
            fields[name] = {
                "horizontally_uniform": True,
                "bit_mismatches": _bit_mismatches(source_host, target_host),
                "compared_cells": int(source_host.size),
            }
            continue
        window = plan.window(source_host.shape)
        if window is None:
            fields[name] = {"disjoint": True, "compared_cells": 0}
            continue
        if frame_width:
            window = _interior_window(
                window, source_host.shape, plan.shift_j, plan.shift_i,
                frame_width)
            if window is None:
                # The frame swallowed the whole overlap: the instrument
                # cannot see, which is a refusal (pass stays False on
                # zero compared cells), never a silent skip.
                fields[name] = {"interior_empty": True, "compared_cells": 0}
                continue
        (dst_j, src_j), (dst_i, src_i) = window
        actual = target_host[..., dst_j, dst_i]
        expected = source_host[..., src_j, src_i]
        fields[name] = {
            "horizontally_uniform": False,
            "bit_mismatches": _bit_mismatches(actual, expected),
            "compared_cells": int(actual.size),
        }
    compared = [item for item in fields.values()
                if item.get("compared_cells", 0) > 0]
    return {
        "fields": fields,
        "compared_field_count": len(compared),
        "pass": bool(compared and all(
            item.get("bit_mismatches") == 0 for item in compared)),
    }


# ---------------------------------------------------------------------------
# Host staging: one child resident on the device, not two
# ---------------------------------------------------------------------------

class HostStateSnapshot:
    """Host copies of a child's state, staged across a relocation.

    Attribute access mirrors the source state (absent field -> absent
    attribute), so this object can stand in as ``source_state`` for
    :func:`transplant_overlap` and :func:`donor_alignment_check` after the
    device allocation it was taken from has been released.  A D2H copy and
    the H2D upload the transplant performs are byte transports, so staging
    through here cannot change a single bit of the overlap -- a property
    the tests assert rather than assume.
    """

    def __init__(self, arrays, *, pinned: bool):
        self._arrays = dict(arrays)
        self._pinned = bool(pinned)

    def __getattr__(self, name):
        arrays = object.__getattribute__(self, "_arrays")
        if name in arrays:
            return arrays[name]
        raise AttributeError(name)

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(sorted(object.__getattribute__(self, "_arrays")))

    @property
    def nbytes(self) -> int:
        arrays = object.__getattribute__(self, "_arrays")
        return int(sum(int(value.nbytes) for value in arrays.values()))

    @property
    def pinned(self) -> bool:
        return object.__getattribute__(self, "_pinned")


def _pinned_host_empty(shape, dtype) -> np.ndarray:
    """A page-locked host array (falls back to pageable without CUDA)."""
    import cupy as cp

    count = int(np.prod(shape)) if len(shape) else 1
    nbytes = int(np.dtype(dtype).itemsize) * max(count, 1)
    memory = cp.cuda.alloc_pinned_memory(nbytes)
    return np.frombuffer(memory, dtype=dtype, count=count).reshape(shape)


def snapshot_state_to_host(state, attrs, *, pin: bool = True
                           ) -> HostStateSnapshot:
    """Copy the named fields to host buffers, pinned where a device exists.

    Fields absent (or ``None``) on the state are absent from the snapshot,
    which is exactly how :func:`transplant_overlap` records them as
    skipped.  Host-resident fields are deep-copied, so releasing the
    source cannot reach back into the snapshot.
    """
    arrays: dict[str, np.ndarray] = {}
    pinned = False
    for name in dict.fromkeys(attrs):
        value = getattr(state, name, None)
        if value is None:
            continue
        if hasattr(value, "__cuda_array_interface__"):
            import cupy as cp

            device = cp.ascontiguousarray(value)
            if pin:
                host = _pinned_host_empty(device.shape, device.dtype)
                pinned = True
            else:
                host = np.empty(device.shape, dtype=device.dtype)
            device.get(out=host)
            arrays[name] = host
        else:
            arrays[name] = np.array(value, copy=True)
    return HostStateSnapshot(arrays, pinned=pinned)


def release_state_arrays(state) -> dict[str, object]:
    """Drop every array reference a state object holds, and count what fell.

    This is the step that keeps peak device residency at ~one child: after
    the snapshot, the outgoing child's arrays are dereferenced BEFORE the
    incoming child allocates, so the allocator can hand the same bytes
    back.  ``owned_device_bytes`` counts allocations the state owns
    outright; views into the shared scratch arena / dycore workspace
    release nothing of their own and are counted apart.  The physics
    driver is dropped too -- its accumulators are per-placement
    continuation state the relocation contract re-initialises anyway.
    """
    owned_device_bytes = 0
    owned_device_arrays = 0
    device_views = 0
    host_arrays = 0
    for name, value in list(vars(state).items()):
        if hasattr(value, "__cuda_array_interface__"):
            if getattr(value, "base", None) is None:
                owned_device_bytes += int(value.nbytes)
                owned_device_arrays += 1
            else:
                device_views += 1
            setattr(state, name, None)
        elif isinstance(value, np.ndarray):
            host_arrays += 1
            setattr(state, name, None)
    if getattr(state, "chem", None) is not None:
        # The species attributes are arena views. Dropping those views
        # alone leaves the arena, prescribed composition and cached
        # chemistry workspaces alive through their owner during a
        # host-staged rebuild. The snapshot already holds the carried
        # fields, and the incoming state constructs its own valid owner.
        chem = state.chem
        from woof.core.chem_context import CHEM_PREP_FIELDS
        arrays = [value for value in vars(chem).values()
                  if hasattr(value, "__cuda_array_interface__")]
        prep = getattr(chem, "prep", None)
        if prep is not None:
            arrays.extend(prep.get(name) for name in CHEM_PREP_FIELDS)
        for value in arrays:
            if (hasattr(value, "__cuda_array_interface__")
                    and getattr(value, "base", None) is None):
                owned_device_bytes += int(value.nbytes)
                owned_device_arrays += 1
        state.chem = None
    cumulus_workspace_bytes = 0
    if getattr(state, "physics", None) is not None:
        # DROPPING state.physics IS NOT ENOUGH ON ITS OWN.  The driver
        # holds cumulus_callable and the adapter holds the driver back
        # (NewTiedtke.bind_driver), so the two form a reference cycle
        # that refcounting cannot collect.  CPython's cyclic collector
        # can, but it triggers on PYTHON object counts and the workspace
        # behind that cycle is 433.2 MiB of VRAM hanging off one small
        # object -- so it stayed resident across relocation after
        # relocation.
        #
        # MEASURED before the fix, matched pair on prepared_nt16_hafs
        # over 30 forecast minutes, identical but for this surface: the
        # relocating arm built FOUR NtWorkspaces against the control's
        # two, and its pool live floor climbed 4425 -> 6923 MiB while
        # the control's held 4425 -> 4451.
        #
        # Asking the adapter to release is deliberate rather than
        # calling gc.collect() here: the collector would also be a
        # whole-heap pause on a path that runs mid-forecast, and it
        # would free this by accident rather than by contract.
        cumulus = getattr(state.physics, "cumulus_callable", None)
        release = getattr(cumulus, "release", None)
        if release is not None:
            cumulus_workspace_bytes = int(release() or 0)
        state.physics = None
    return {
        "owned_device_bytes": int(owned_device_bytes),
        "owned_device_arrays": int(owned_device_arrays),
        "shared_view_arrays": int(device_views),
        "host_arrays": int(host_arrays),
        "cumulus_workspace_bytes": int(cumulus_workspace_bytes),
    }


def _device_used_bytes():
    """Live CuPy default-pool bytes, or ``None`` off the card."""
    try:
        import cupy as cp

        return int(cp.get_default_memory_pool().used_bytes())
    except Exception:
        return None


def _prevalidate_placement(new_dc, parent_node) -> None:
    """Fail an off-parent move BEFORE the outgoing child is released.

    The in-device path gets this refusal for free from the initializer's
    ``register_nest`` call, which fires while the outgoing child is still
    whole.  The host-staged path releases the outgoing child first, so the
    same +-2 SINT stencil rule is evaluated here, on CPU index tables
    alone, while everything is still intact.  ``register_nest`` stays the
    owner of the rule; this merely asks it early.
    """
    from woof.core.nest_interp import register_nest

    child_run = new_dc.run
    parent_run = parent_node.cfg.run
    ratio = int(new_dc.parent_grid_ratio)
    for stagger in ("", "x", "y"):
        register_nest(
            nri=ratio, nrj=ratio,
            i_parent_start=int(new_dc.i_parent_start),
            j_parent_start=int(new_dc.j_parent_start),
            child_nx=child_run.nx, child_ny=child_run.ny,
            parent_nx=parent_run.nx, parent_ny=parent_run.ny,
            stagger=stagger, wrapper="bdy")


# ---------------------------------------------------------------------------
# The primitive
# ---------------------------------------------------------------------------

#: The explicit statics fallback this module ships tonight, stated on
#: every receipt that uses it.  The 2026-08-06 requirement (project ruling, via the
#: WRF moving-nest discussion): a relocated nest's STATIC fields --
#: terrain, landuse, soil categories -- must be REBUILT for the new
#: footprint from the nest's own static source at nest resolution
#: ([static.highres] / the 30s baseline, both footprint-parametric),
#: because over-land storm-following lives on resolved terrain and
#: parent-interpolated terrain on the fresh strip defeats the nest.  The
#: ``initializer`` parameter of :func:`relocate_child` is that seam -- a
#: route passes a per-footprint builder that constructs own-grid statics
#: and runs the SAME terrain adjustment the t=0 nest cold start performs
#: (woof.ingest.nest_init) -- and the default ``parent_only_init`` is
#: the explicit, receipt-recorded fallback, never a silent choice.
#: The real-data initializer landed as
#: :func:`woof.ingest.relocation_init.real_relocation_initializer`:
#: own-source statics for the new footprint plus the t=0 cold-start
#: terrain adjustment on the parent-interpolated state.
PARENT_INTERPOLATED_STATICS_FALLBACK = (
    "parent-interpolated statics (SINT of the live parent): the explicit "
    "fallback for idealized trees; real-data routes rebuild own-source "
    "statics per footprint (woof.ingest.relocation_init, 2026-08-06 "
    "requirement)")


def relocate_child(child_node, *, i_parent_start: int, j_parent_start: int,
                   segment: RelocationSegment | None = None,
                   bounds=None, initializer=None, on_child_built=None,
                   scratch_arena=None, dycore_state_workspace=None,
                   state_digest=None, staging: str = "host",
                   on_before_release=None,
                   static_provenance: str | None = None,
                   reground_descendant=None,
                   earth_fixed_descendants=frozenset(),
                   reconstruction=None, restoring_checkpoint: bool = False
                   ) -> dict[str, object]:
    """Move a live child domain to a new placement, in place on the node.

    The three things this does, in the order they must happen:

    1. build the child afresh at the new placement from the LIVE parent,
       through the ordinary cold-start path, which also rebuilds the
       child's own grid, map factors and base state for where it now is;
    2. stamp the outgoing child's serialised state onto the overlap, so
       everything the child had integrated over ground it still covers
       survives the move bitwise;
    3. rebuild the coupler's SINT donor tables for the new placement, and
       classify the child's rolling boundary tables invalid -- they were
       built for a footprint that no longer exists.

    The parent is not written at any point.  ``parent_state_sha256`` in the
    returned receipt is taken before and after and compared, so that is a
    measured claim rather than an architectural one.

    ``restoring_checkpoint`` rebuilds geometry before the caller restores
    the complete checkpoint tree. Its descendants need no overlap with
    their cold-start footprints: none of that temporary state is integrated.
    Live moves retain the disjoint-descendant refusal.

    ``child_node`` must be at rest: relocation is a cycle-boundary
    operation, and calling it with the parent leading the child would leave
    the rebuilt tables describing an instant the child has not reached.

    MEMORY -- TWO STAGINGS.  ``staging="device"`` allocates the incoming
    child while the outgoing one is still resident, so peak child
    residency is transiently doubled (a cycle-boundary cost, not a
    per-step one).  That was the right trade for proving the mechanism
    against the certified cold-start path, and it is the wrong trade for
    the user this feature is FOR: the point of a moving nest is a low-VRAM
    card running a smaller nest at higher resolution, and a card that
    cannot hold two children cannot move one.  ``staging="host"`` is the
    production answer: snapshot the outgoing child's serialised state
    (plus the SINT base fields the alignment instrument needs) to pinned
    host buffers, RELEASE the outgoing device allocation, build the
    incoming child into the bytes the allocator just got back, and upload
    the overlap from host.  D2H and H2D are byte transports, so the two
    stagings are bitwise-identical on the overlap -- asserted by
    tests/test_nest_relocation_staging.py, not assumed.  The receipt's
    ``staging`` block carries live-pool samples around each phase so the
    peak claim is measured, never architectural.

    Host staging is the DEFAULT.  With ``"device"`` as the default, every
    caller that did not name a staging (the relocation verification case
    among them) held the outgoing and the incoming child on the card at
    once, with nothing pricing the second one, while the production
    runner passed ``"host"`` explicitly; a card that fit the tree could
    run out of memory at a move only such a caller made.  ``"device"``
    stays available by name, for the equivalence tests that compare the
    two stagings.

    ATOMICITY UNDER HOST STAGING, STATED ACCURATELY.  The in-device path
    leaves the tree untouched on any refusal.  The host path releases the
    outgoing child before the rebuild, so a failure AFTER the release
    (an allocation failure, or the donor-alignment self-check tripping on
    a code defect) leaves the child unusable and the run over.  The
    off-parent refusal -- the one plausible input error -- is evaluated
    BEFORE the release (:func:`_prevalidate_placement`), so what remains
    behind the release is the class of failure that would have killed the
    run anyway.  ``on_before_release`` is the executor's seam for dropping
    its own references (health validators) so the release actually frees;
    it fires only on the host path, immediately before the release.

    SCOPE, STATED SO IT CANNOT BE ASSUMED AWAY.  What THIS PRIMITIVE moves
    is the restart layer's serialised state (:func:`relocatable_attrs`).
    Physics CONTINUATION state -- precipitation accumulators, scheme
    timers, surface fields held by the driver -- is not in that contract:
    it is the ``on_child_built`` preparer's to carry.  The front-door
    route preparers (:class:`woof.runtime.RealRelocationChildPreparer`
    and the prepared-tree subclass) DO carry it -- the land-surface
    continuation fields by donor-filled transplant, the registry-
    derived per-column driver state (KF timers/held rates, precipitation
    accumulators, W0AVG) and the surface-radiation carriers with their
    provenance ledger, both through
    :mod:`woof.core.physics_continuation` -- because a preparer that
    re-initialised it cold-restarted cumulus on the whole child at every
    move (the 2026-08-16 moving-nest KF artifact report) and left the
    moved child with no producer for GLW between radiation calls (the
    2026-08-24 test host B campaign refusal).  A bespoke ``on_child_built``
    that carries nothing still gets a cold child, and its receipt says
    so (``accumulators_reinitialized``).
    """
    from woof.ingest.nest_init import (parent_only_init,
                                        require_parent_chem_fields,
                                        seed_rk_time_t_copies)

    if initializer is None:
        initializer = parent_only_init
        if static_provenance is None:
            static_provenance = PARENT_INTERPOLATED_STATICS_FALLBACK
    if state_digest is None:
        from woof.ensemble.state_sha import live_state_sha256
        state_digest = (live_state_sha256 if reconstruction is None
                        else reconstruction.state_digest)

    if staging not in ("device", "host"):
        raise RelocationRefusal(
            f"staging must be 'device' or 'host', got {staging!r}")
    if reconstruction is not None and staging != "host":
        raise RelocationRefusal("bounded streamed reconstruction requires host staging")
    parent_node = child_node.parent
    if parent_node is None:
        raise RelocationRefusal(
            "the root domain has no parent to be placed in; relocation "
            "applies to a child")
    if earth_fixed_descendants:
        _fixed = frozenset(int(g) for g in earth_fixed_descendants)
        _needs_reground = any(
            int(node.cfg.grid_id) not in _fixed
            for node, _di, _dj in descendant_regroundings(
                child_node, 1, 1, earth_fixed=_fixed))
    else:
        _needs_reground = bool(getattr(child_node, "children", None))
    if _needs_reground and reground_descendant is None:
        raise RelocationRefusal(
            f"child d{child_node.cfg.grid_id:02d} has children of its own, "
            "and no reground_descendant handler was supplied; moving it "
            "would silently change the ground under every grandchild while "
            "their statics still describe the old footprint.  A mid-tree "
            "move is implemented (see descendant_regroundings / "
            "plan_descendant_reground) but it needs the ROUTE to rebuild "
            "each descendant's statics for its new ground, which is why "
            "the handler is the seam and not a flag: a route with no "
            "statics source for the descendants cannot honour it.  On the "
            "prepared routes that means a statics corridor for every "
            "descendant of the mover, not just the mover.  (A descendant "
            "named in earth_fixed_descendants is exempt: it compensates "
            "the move instead of riding along, so its ground does not "
            "change.)")
    if child_node.coupler is None:
        raise RelocationRefusal(
            f"child d{child_node.cfg.grid_id:02d} has no coupler; a domain "
            "with no parent forcing edge cannot be relocated coherently")
    lead = int(parent_node.clock.ticks) - int(child_node.clock.ticks)
    if lead != 0:
        raise RelocationRefusal(
            f"relocation is a cycle-boundary operation and requires "
            f"synchronized clocks; parent leads child by {lead} ticks. "
            "Relocate between legs, not mid-step")

    # Validate chemistry donors while the outgoing arena is still intact.
    # The child's rows may be a subset of the parent's, but a missing row
    # would leave no source for its fresh strip or subsequent nest forcing.
    require_parent_chem_fields(child_node.state, parent_node.state)

    old_dc = child_node.cfg
    placement_from = placement_of(
        old_dc, generation=(0 if segment is None else segment.generation))
    placement_to = Placement(
        grid_id=int(old_dc.grid_id),
        i_parent_start=int(i_parent_start),
        j_parent_start=int(j_parent_start),
        generation=placement_from.generation + 1)
    plan = plan_relocation(
        placement_from=placement_from, placement_to=placement_to,
        parent_grid_ratio=int(old_dc.parent_grid_ratio),
        child_nx=int(old_dc.run.nx), child_ny=int(old_dc.run.ny))
    if plan.disjoint and not restoring_checkpoint:
        raise RelocationRefusal(
            f"the new placement {placement_to.position} shares no cell with "
            f"the old one {placement_from.position}; that is a new domain, "
            "not a relocation, and nothing of the child's integrated state "
            "would survive it")
    admissibility = (None if bounds is None
                     else check_admissible(plan, bounds))
    if static_provenance is None:
        raise RelocationRefusal(
            "a custom relocation initializer must state its "
            "static_provenance: whether the rebuilt child's terrain/"
            "landuse/soil categories are footprint-rebuilt from the "
            "nest's own static source or parent-interpolated is a "
            "recorded fact of every move, never a silent property "
            "(2026-08-06 statics-on-move requirement)")

    parent_sha_before = state_digest(parent_node.state)
    child_sha_before = state_digest(child_node.state)

    new_dc = replace(
        old_dc, i_parent_start=int(i_parent_start),
        j_parent_start=int(j_parent_start))
    extra = {}
    if scratch_arena is not None:
        extra["scratch_arena"] = scratch_arena
    if dycore_state_workspace is not None:
        extra["dycore_state_workspace"] = dycore_state_workspace

    from woof.ensemble.runtime_context import (current_member_reconstruction_owner,
                                                bind_reconstructed_member_node)
    member_pattern_owner = current_member_reconstruction_owner(child_node.state)
    used_steady = _device_used_bytes()
    if staging == "host":
        # The off-parent refusal, evaluated while the outgoing child is
        # still whole; everything after the release is committed.
        _prevalidate_placement(new_dc, parent_node)
        source_state = (snapshot_state_to_host(
            child_node.state,
            tuple(relocatable_attrs(child_node.state))
            + _DONOR_ALIGNMENT_FIELDS)
            if reconstruction is None else reconstruction.capture_source(child_node))
        if on_before_release is not None:
            on_before_release()
        released = (release_state_arrays(child_node.state) if reconstruction is None
                    else reconstruction.release_outgoing(child_node))
        used_after_release = _device_used_bytes()
    else:
        source_state = child_node.state
        released = None
        used_after_release = None

    # register_nest refuses a placement whose donors fall outside the
    # parent's +-2 SINT stencil, so on the in-device path an off-grid move
    # fails HERE, before the outgoing child has been touched (the host
    # path asked the same rule above, before the release).
    initialized = (initializer(new_dc, parent_node, **extra) if reconstruction is None
                   else reconstruction.initialize(new_dc, parent_node))
    # The caller's per-domain seam: map policy, and the physics driver the
    # cold-start path does not attach.  It fires BEFORE the transplant, so
    # a driver is initialised from the parent-interpolated fields and the
    # transplant's stamped diagnostics are the last write to survive.
    if on_child_built is not None and reconstruction is None:
        on_child_built(initialized, new_dc, parent_node)

    # An initializer whose base fields are blended toward the parent at
    # each footprint's own edge (the real-data t=0 lineage) declares the
    # frame inside which the placement invariant is undefined; the
    # instrument is scoped to where it IS defined and stays bitwise there.
    alignment_frame = int(
        getattr(initializer, "donor_alignment_frame_width", 0) or 0)
    alignment = ({"pass": None, "reason": "checkpoint reconstruction; no state carried"}
                 if restoring_checkpoint else donor_alignment_check(
                     source_state=source_state, target_state=initialized.state,
                     plan=plan, frame_width=alignment_frame))
    if not restoring_checkpoint and not alignment["pass"]:
        raise RelocationRefusal(
            "the relocated child's SINT-derived base state does not match "
            "the outgoing child's on the overlap, so the two placements do "
            f"not share donor cells: {alignment}")

    transplant = ({"checkpoint_reconstruction": True, "fields": {}}
                  if restoring_checkpoint else transplant_overlap(
                      source_state=source_state, target_state=initialized.state,
                      plan=plan))
    # THE PROBE, in two halves (see overlap_prognostic_mismatches): the
    # stamp on its own, then the same check after the hook below.  A
    # field that is clean here and dirty there was perturbed by the hook,
    # which is the distinction no receipt could previously draw.
    probe = None
    if not restoring_checkpoint and relocation_probe_enabled():
        probe = {"after_transplant": overlap_prognostic_mismatches(
            source_state, initialized.state, plan)}
    # The initializer's post-transplant hook: the real-data route rebases
    # blend-frame perturbations onto the rebuilt base so totals survive
    # exactly where the base bytes differ, then re-derives the EOS
    # diagnostics.  It runs BEFORE the RK time-t reseed below, so the
    # seeds copy the corrected fields.
    post_transplant = getattr(initializer, "post_transplant", None)
    post_receipt = None
    if reconstruction is not None:
        post_receipt = reconstruction.post_transplant(
            source_state=source_state, target_state=initialized.state, plan=plan)
    elif post_transplant is not None and not restoring_checkpoint:
        post_receipt = post_transplant(
            source_state=source_state, target_state=initialized.state,
            plan=plan)
    if probe is not None:
        probe["after_post_transplant"] = overlap_prognostic_mismatches(
            source_state, initialized.state, plan)
        _LOG.warning("relocation probe d%02d: %s",
                     int(new_dc.grid_id), json.dumps(probe, default=str))
    used_after_rebuild = _device_used_bytes()
    staging_receipt: dict[str, object] = {
        "mode": staging,
        "device_pool_used_bytes": {
            "steady_state_before": used_steady,
            "after_release": used_after_release,
            "after_rebuild_and_transplant": used_after_rebuild,
        },
    }
    if staging == "host":
        staging_receipt.update({
            "snapshot_fields": len(source_state.field_names),
            "snapshot_bytes": int(source_state.nbytes),
            "snapshot_pinned": bool(source_state.pinned),
            "released": released,
        })
    else:
        staging_receipt["note"] = (
            "in-device transplant: the incoming child is allocated while "
            "the outgoing one is resident, so peak child residency is "
            "transiently doubled; staging='host' is the production path "
            "and the default")
    # WRF's start_domain seeds the RK time-t copies from the interpolated
    # state; after the stamp the current fields have changed, so the seeds
    # must be taken again or the first substep would read the cold-start
    # values over ground the transplant just corrected.
    seeded = (seed_rk_time_t_copies(initialized.state, preserve_thermal=True) if reconstruction is None
              else reconstruction.rk_seeds)

    # Everything that reads the parent has now run.  Check the parent
    # BEFORE the node is touched, so a violation leaves the live tree
    # exactly as it was rather than half-moved with a traceback.
    parent_sha_after = state_digest(parent_node.state)
    if parent_sha_after != parent_sha_before:
        raise RelocationRefusal(
            "the parent state changed across a relocation; the re-grid "
            "reads the parent and must never write it "
            f"({parent_sha_before} -> {parent_sha_after})")
    child_sha_after = state_digest(initialized.state)

    child_node.cfg = new_dc
    child_node.grid = getattr(initialized, "grid", child_node.grid)
    child_node.state = initialized.state
    child_node.state._nest_restart_classification = "REBUILT"
    coupler_receipt = child_node.coupler.relocate()
    if reconstruction is not None:
        reconstruction.commit(child_node)
    bind_reconstructed_member_node(child_node, previous_owner=member_pattern_owner)

    # ---- the subtree: the ground moved under every descendant ----------
    # Ordered parent-first (descendant_regroundings appends before it
    # recurses), because a descendant is rebuilt by SINT of its PARENT and
    # the parent must already be standing on its new ground.  The mover is
    # committed at this point, so a descendant failure is the same class of
    # unrecoverable-after-release failure the host staging path already
    # documents -- it is not made recoverable by pretending otherwise.
    descendant_receipts: list[dict[str, object]] = []
    earth_fixed_descendants = frozenset(
        int(g) for g in (earth_fixed_descendants or ()))
    if reground_descendant is not None or earth_fixed_descendants:
        move_i = (int(placement_to.i_parent_start)
                  - int(placement_from.i_parent_start))
        move_j = (int(placement_to.j_parent_start)
                  - int(placement_from.j_parent_start))
        for node, delta_i, delta_j in descendant_regroundings(
                child_node, move_i, move_j,
                earth_fixed=earth_fixed_descendants):
            if int(node.cfg.grid_id) in earth_fixed_descendants:
                # COMPENSATED ride-along: this descendant is a mover in
                # its own right, so instead of dragging it across the
                # earth by the ancestor's displacement and transplanting
                # its whole state, its PLACEMENT moves by -delta and its
                # earth position does not move at all.  The integer-cell
                # transplant above resampled the parent's statics at the
                # same earth points, so every input this descendant reads
                # (blend-frame terrain included) is bitwise what it was
                # -- the state is carried untouched, and only the donor
                # geometry (coupler.relocate) and the rolling boundary
                # tables (invalidated by it) change.
                from dataclasses import replace as _replace

                comp = _replace(
                    node.cfg,
                    i_parent_start=int(node.cfg.i_parent_start) - delta_i,
                    j_parent_start=int(node.cfg.j_parent_start) - delta_j)
                if comp.i_parent_start < 1 or comp.j_parent_start < 1:
                    raise RelocationRefusal(
                        f"earth-fixed descendant d{comp.grid_id:02d} "
                        f"compensation ({-delta_i},{-delta_j}) leaves its "
                        "parent's 1-based frame; the mover's shift must "
                        "be clamped by its earth-fixed descendants' "
                        "admissible band before commitment")
                _prevalidate_placement(comp, child_node)
                placement_pair = (
                    [int(node.cfg.i_parent_start),
                     int(node.cfg.j_parent_start)],
                    [int(comp.i_parent_start), int(comp.j_parent_start)])
                node.cfg = comp
                node.state._nest_restart_classification = "REBUILT"
                descendant_receipts.append({
                    "grid_id": int(node.cfg.grid_id),
                    "parent_id": int(node.cfg.parent_id),
                    "delta_parent_cells": [int(delta_i), int(delta_j)],
                    "earth_fixed": True,
                    "placement_from": placement_pair[0],
                    "placement_to": placement_pair[1],
                    "state_carried_bitwise": True,
                    "coupler": node.coupler.relocate(),
                })
                continue
            dplan = plan_descendant_reground(
                node.cfg, delta_i, delta_j,
                generation=placement_from.generation)
            if dplan.disjoint and not restoring_checkpoint:
                raise RelocationRefusal(
                    f"the move carries descendant "
                    f"d{int(node.cfg.grid_id):02d} entirely off its own "
                    f"old ground (shift {dplan.shift_i},{dplan.shift_j} "
                    f"cells of a {dplan.child_cells}-cell domain); nothing "
                    "it integrated would survive, which is a new domain "
                    "rather than a move.  Reduce max_move_parent_cells or "
                    "raise min_overlap_fraction so the mover's step stays "
                    "inside the FINEST descendant's overlap, not just its "
                    "own -- the same move is a larger fraction of a finer "
                    "domain.")
            reground = (getattr(reground_descendant, "restore_checkpoint", reground_descendant)
                        if restoring_checkpoint else reground_descendant)
            receipt = reground(
                node=node, plan=dplan,
                delta_parent_cells=(int(delta_i), int(delta_j)))
            # The descendant's placement never changed, so its donor
            # tables are still valid -- but its parent's state OBJECT was
            # replaced above, so the coupler edge has to be re-seated.
            node.state._nest_restart_classification = "REBUILT"
            descendant_receipts.append({
                "grid_id": int(node.cfg.grid_id),
                "parent_id": int(node.cfg.parent_id),
                "delta_parent_cells": [int(delta_i), int(delta_j)],
                "plan": dplan.to_json(),
                "coupler": node.coupler.relocate(),
                "reground": receipt,
            })

    record = RelocationRecord(
        placement_from=placement_from, placement_to=placement_to,
        parent_grid_ratio=int(old_dc.parent_grid_ratio),
        shift_i=plan.shift_i, shift_j=plan.shift_j,
        overlap_cells=plan.overlap_cells, child_cells=plan.child_cells,
        null_move=plan.null_move,
        parent_state_sha256=parent_sha_before,
        child_state_sha256_before=child_sha_before,
        child_state_sha256_after=child_sha_after,
        donor_alignment=alignment)
    segment = (base_segment(old_dc) if segment is None else segment)
    segment = segment.append(record)
    return {
        "contract": RELOCATION_CONTRACT,
        "plan": plan.to_json(),
        "admissibility": admissibility,
        "staging": staging_receipt,
        "child_rebuild": {
            "initializer": getattr(initializer, "__qualname__",
                                   repr(initializer)),
            "static_fields": static_provenance,
            # The initializer's own rebuild receipt (static source,
            # rebuild timings, adjustment applied) when it produced one.
            "rebuild": getattr(initialized, "preprocess_receipt", None),
            "donor_alignment_frame_width": alignment_frame,
            "post_transplant": post_receipt,
        },
        "coupler": coupler_receipt,
        "prognostic_overlap_probe": probe,
        # Absent on a leaf move, so a leaf receipt is byte-identical to
        # what it was before mid-tree moves existed.
        **({"descendants": descendant_receipts}
           if descendant_receipts else {}),
        "transplant": transplant,
        "rk_seeds_refreshed": list(seeded),
        "donor_alignment": alignment,
        "parent_state_sha256_before": parent_sha_before,
        "parent_state_sha256_after": parent_sha_after,
        "parent_bitwise_unchanged": parent_sha_before == parent_sha_after,
        "child_state_sha256_before": child_sha_before,
        "child_state_sha256_after": child_sha_after,
        "child_state_unchanged": child_sha_before == child_sha_after,
        "segment": segment.to_json(),
        # The live object, so a caller can chain the next move onto this
        # one; "segment" above is its JSON for the receipt.
        "segment_state": segment,
        "record_sha256": segment.records[-1].sha256,
    }


def descendant_regroundings(moving_node, shift_i: int, shift_j: int,
                            *, earth_fixed=frozenset()
                            ) -> list[tuple[object, int, int]]:
    """``(node, delta_i, delta_j)`` for every descendant of a mover.

    WHAT A DESCENDANT EXPERIENCES.  When a mid-tree domain relocates, its
    children do NOT change placement -- a child's ``i_parent_start`` is an
    offset inside its parent, and the parent carried it along -- but the
    GROUND under them moved by exactly the parent's displacement.  So a
    descendant needs the same treatment the mover needs (statics rebuilt
    for the new ground, overlap transplanted, fresh strip filled from its
    parent) and none of the placement arithmetic.

    THE ARITHMETIC, AND WHY IT STAYS EXACT.  ``shift_i`` is the mover's
    placement change in ITS parent's cells.  Descending one level
    multiplies the displacement by that level's refinement ratio, so a
    move of ``di`` cells of the mover's parent is::

        mover's own cells        di * r_mover
        a child of the mover     di * r_mover * r_child
        a grandchild             di * r_mover * r_child * r_grandchild

    Every factor is an integer, so every descendant's displacement is a
    WHOLE number of its own cells -- which is the property the whole
    design rests on (a pure index-space copy, and SINT statics that are
    bitwise identical on the overlap).  A moving mid-tree domain is
    therefore not a weaker operation than a moving leaf; it is the same
    operation applied down a chain of exact integer ratios.

    The returned delta is expressed in each descendant's OWN PARENT's
    cells, which is the frame :func:`plan_relocation` wants, so
    :func:`plan_descendant_reground` can hand it straight over.

    ``earth_fixed`` names descendants that COMPENSATE instead of riding
    along: their placement moves by ``-delta`` so their earth position
    does not move at all.  The walk does not descend past one -- its own
    subtree experiences zero net ground displacement (the ancestor's
    displacement and the compensation cancel exactly, both being whole
    numbers of the same cells), so there is nothing to re-ground below
    it.  A grandchild of the mover whose parent rides along still gets
    its row.
    """
    earth_fixed = frozenset(int(g) for g in (earth_fixed or ()))
    out: list[tuple[object, int, int]] = []

    def walk(node, cum: int) -> None:
        for child in (getattr(node, "children", None) or ()):
            out.append((child, int(shift_i) * cum, int(shift_j) * cum))
            if int(child.cfg.grid_id) in earth_fixed:
                continue
            walk(child, cum * int(child.cfg.parent_grid_ratio))

    walk(moving_node, int(moving_node.cfg.parent_grid_ratio))
    return out


def plan_descendant_reground(child_dc, delta_i: int, delta_j: int,
                             *, generation: int = 0) -> "RelocationPlan":
    """The overlap plan for a descendant whose GROUND moved under it.

    BOTH placements handed to :func:`plan_relocation` are synthetic.  The
    descendant's placement never changed, so there is no real pair to
    give; what the transplant needs is only their DIFFERENCE, which is
    the ground displacement in the descendant's parent's cells.

    They are also BIASED to stay >= 1.  ``Placement`` enforces 1-based
    WRF namelist semantics, and a westward/southward move of a
    descendant already near its parent's low edge drives a naive
    ``i_parent_start - delta`` negative -- which refused a move that is
    perfectly admissible.  Adding the same offset to both sides leaves
    the difference, and therefore the plan, untouched.

    So the placements inside the returned plan are an artefact of reusing
    the planner and are NOT a record of where anything sits; the accurate
    record is ``delta_parent_cells`` on the relocation receipt.

    Reusing the same planner rather than writing a second one is
    deliberate: the disjoint test, the overlap-cell count and the index
    arithmetic the leaf path is asserted against are the ones a
    descendant gets too.
    """
    ref_i, ref_j = int(child_dc.i_parent_start), int(child_dc.j_parent_start)
    di, dj = int(delta_i), int(delta_j)
    bias_i = max(0, 1 - min(ref_i, ref_i + di))
    bias_j = max(0, 1 - min(ref_j, ref_j + dj))
    return plan_relocation(
        placement_from=Placement(
            grid_id=int(child_dc.grid_id),
            i_parent_start=ref_i + bias_i,
            j_parent_start=ref_j + bias_j,
            generation=int(generation)),
        placement_to=Placement(
            grid_id=int(child_dc.grid_id),
            i_parent_start=ref_i + bias_i + di,
            j_parent_start=ref_j + bias_j + dj,
            generation=int(generation) + 1),
        parent_grid_ratio=int(child_dc.parent_grid_ratio),
        child_nx=int(child_dc.run.nx), child_ny=int(child_dc.run.ny))


def mark_fingerprint_across_move(fingerprint: str,
                                 record_sha256: str) -> str:
    """Chain a relocation record into a run's live restart fingerprint.

    The tolerated-experiment convention, extended from config to the live
    tree: relocation BOUNDS are byte-inert on every fingerprint
    (``RESTART_TOLERATED_EXPERIMENT_FIELDS``), while an actual MOVE binds
    -- the placement is part of the restart identity, so a tree that has
    moved must not share a fingerprint with the tree as built.  Chaining
    keeps the marking deterministic (same base, same move history, same
    mark) and one-way: no fresh build ever computes a marked value, so a
    checkpoint written after a move refuses to resume into a fresh run BY
    CONSTRUCTION, and the refusal names relocation instead of shrugging a
    hash (:data:`RESTART_ACROSS_MOVE_POSTURE`).  Nothing here builds a
    resume across a move; per the 2026-08-06 ruling, nothing may.
    """
    digest = hashlib.sha256()
    digest.update(RELOCATION_CONTRACT.encode("ascii") + b"\0moved\0")
    digest.update(str(fingerprint).encode("utf-8") + b"\0")
    digest.update(str(record_sha256).encode("ascii") + b"\0")
    return digest.hexdigest()


def relocating_child_initializer(source_state, *, placement_from: Placement,
                                 initializer=None, on_child_built=None):
    """A child initializer that transplants instead of cold-starting.

    The leg-boundary form of :func:`relocate_child`, shaped to drop into
    the assembly hooks that already take a ``child_initializer`` (the
    idealized tree assembler, and the DA route's per-leg rebuild).  The
    same three-step mechanism runs; only the seam differs, because on that
    route the whole model is rebuilt per leg and there is no live node to
    mutate.
    """
    from woof.ingest.nest_init import parent_only_init, seed_rk_time_t_copies

    if initializer is None:
        initializer = parent_only_init
    receipts: list[dict] = []

    def initialize(child_dc, parent_node, **kwargs):
        initialized = initializer(child_dc, parent_node, **kwargs)
        if on_child_built is not None:
            on_child_built(initialized, child_dc, parent_node)
        placement_to = Placement(
            grid_id=int(child_dc.grid_id),
            i_parent_start=int(child_dc.i_parent_start),
            j_parent_start=int(child_dc.j_parent_start),
            generation=placement_from.generation + 1)
        plan = plan_relocation(
            placement_from=placement_from, placement_to=placement_to,
            parent_grid_ratio=int(child_dc.parent_grid_ratio),
            child_nx=int(child_dc.run.nx), child_ny=int(child_dc.run.ny))
        if plan.disjoint:
            raise RelocationRefusal(
                f"placements {placement_from.position} and "
                f"{placement_to.position} share no cell")
        alignment = donor_alignment_check(
            source_state=source_state, target_state=initialized.state,
            plan=plan)
        if not alignment["pass"]:
            raise RelocationRefusal(
                f"relocated base state disagrees on the overlap: {alignment}")
        transplant = transplant_overlap(
            source_state=source_state, target_state=initialized.state,
            plan=plan)
        seeded = seed_rk_time_t_copies(initialized.state, preserve_thermal=True)
        receipts.append({
            "contract": RELOCATION_CONTRACT,
            "plan": plan.to_json(),
            "transplant": transplant,
            "donor_alignment": alignment,
            "rk_seeds_refreshed": list(seeded),
        })
        return initialized

    initialize.receipts = receipts
    return initialize


__all__ = [
    "HostStateSnapshot", "PARENT_INTERPOLATED_STATICS_FALLBACK",
    "PLACEMENT_FIELDS", "RELOCATION_CONTRACT",
    "RESTART_ACROSS_MOVE_POSTURE", "Placement",
    "RelocationPlan", "RelocationRecord", "RelocationRefusal",
    "RelocationSegment", "base_segment", "check_admissible",
    "clamp_shift_to_overlap", "max_parent_cells_for_overlap",
    "overlap_fraction_for_shift",
    "donor_alignment_check", "mark_fingerprint_across_move",
    "placement_independent_identity", "placement_of", "plan_relocation",
    "release_state_arrays",
    "relocatable_attrs", "relocate_child", "relocating_child_initializer",
    "snapshot_state_to_host", "transplant_overlap",
]
