"""Cadence-driven execution of discrete nest relocations inside a run.

:mod:`woof.core.nest_relocation` is the mechanism -- it moves a live
child once, when told where.  :mod:`woof.core.storm_tracking` is the
policy -- it proposes where.  This module is the runner between them: at
every complete cycle boundary that lies on the configured cadence, ask
the follow source for a desired whole-parent-cell shift and execute it
through :func:`~woof.core.nest_relocation.relocate_child` with host
staging, admissibility bounds, and a receipt appended to the run's
relocation receipts.

WHERE IT FIRES.  ``execute_experiment`` calls
:meth:`RelocationRunner.on_period_begin` at PERIOD_BEGIN, the instant
every domain clock in the tree sits on the same complete d01 boundary.
Every such boundary is a parent-step boundary for every child, which is
the synchronization :func:`relocate_child` requires; the cadence is
therefore validated to be a whole number of periods and can never ask
for a mid-step move.  ``cadence_seconds`` absent means every boundary,
with the tracker's own dead-band/cooldown as the rate limit.

THE FOLLOW SOURCE is the storm tracker's published plan-provider seam
(:mod:`woof.core.storm_tracking`, adopted verbatim)::

    provider(parent_state, nest_footprint, t) -> (di, dj) | None

``parent_state`` is the PARENT domain's live state; ``nest_footprint``
is anything :class:`~woof.core.storm_tracking.NestFootprint` can
coerce (the runner passes the child's live DomainConfig); ``t`` is model
seconds.  The return is the desired shift in WHOLE PARENT CELLS (``di``
increments ``i_parent_start``, ``dj`` increments ``j_parent_start``) or
``None`` for hold.  :class:`woof.core.storm_tracking.StormTracker`
implements it; :class:`ManualMoveProvider` implements the SAME contract
from the config's ``[[relocation.move]]`` rows, so the manual mode is a
standing test of the exact interface a tracker must satisfy.

The provider proposes; the runner enforces.  An executed shift is the
proposal clamped to ``max_move_parent_cells`` and to the parent's
admissible placement band, and both the requested and the executed shift
land in the receipt -- a clamped decision is visible, never silent.
After an executed move the runner calls the provider's OPTIONAL
``notify_move_executed(t, executed_shift)`` hook (accepted-move
semantics for trackers whose cooldown should count acceptances rather
than proposals); a provider without the hook simply is not notified.

After every executed move the runner chains the record into the model's
live restart fingerprint
(:func:`~woof.core.nest_relocation.mark_fingerprint_across_move`), so
checkpoints written later refuse to resume into a FRESH build BY
CONSTRUCTION and the refusal names relocation.  That chaining is also
what makes the resume possible at all: a restore replays the same
digests in the same order onto its own fingerprint, so it reaches the
checkpoint's value exactly when it is the run that wrote it.  See
:data:`~woof.core.nest_relocation.RESTART_ACROSS_MOVE_POSTURE` for what
a moved checkpoint promises, which is no longer nothing.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from fractions import Fraction
from pathlib import Path

from woof.core.nest_relocation import (RelocationRefusal, RelocationSegment,
                                        base_segment,
                                        mark_fingerprint_across_move,
                                        relocate_child)
from woof.progress_log import centre_latlon, model_step_log
from woof.core.storm_tracking import TRACKER_OPTIONAL_STATE_KEYS

#: Versioned label for the runner's own receipt rows.
from woof.core.uh_diag import (UH_FOLLOW_WINDOW_SLOT,
                               reset_tracker_window)

RELOCATION_RUNNER_CONTRACT = "gpuwm-nest-relocation-runner.v1"

#: The keys of a checkpoint's per-follower entry that the RUNNER owns.
RUNNER_STATE_KEYS = ("segment", "moves_executed", "last_proposal_t",
                     "last_move_t")

#: The rest of the same entry, filled by the checkpoint writer because it
#: describes the TREE rather than this runner's history.  Tolerated on the
#: way back in so a resume hands the entry over whole instead of stripping
#: it first, and never read here.
RUNNER_WRITER_KEYS = ("kind", "current_placement", "declared_placement")

#: What fills the strip a move exposes.  Stated on every move receipt so
#: the fill provenance is a recorded fact, not an assumption: the child
#: is rebuilt at the new placement by the ordinary cold-start path (a
#: full SINT of the LIVE parent), and the overlap is then stamped bitwise
#: from the outgoing child, so only the newly exposed strip keeps the
#: parent-interpolated values.
STRIP_FILL_SOURCE = ("parent_only_init: cold start from the live parent "
                     "(SINT), overlap then stamped from the outgoing child")


class ManualMoveProvider:
    """The config-driven follow source: ``[[relocation.move]]`` rows.

    Implements the storm tracker's plan-provider contract
    (``provider(parent_state, nest_footprint, t)``) so the runner cannot
    tell it from a tracker.  Each row fires at its exact ``at_seconds``
    opportunity and exactly once; opportunities with no row are holds.
    The rows were validated at config admission (strictly increasing,
    on-boundary, inside the run), so a row left unconsumed at run end is
    a runner defect, and :meth:`unconsumed` lets the receipts say so.

    A RESUME STARTS PAST ROWS THAT ALREADY FIRED, and the queue is
    strictly ordered: the head is matched EXACTLY, so a row whose
    opportunity is behind the clock can never match again -- and while it
    sits at the head it blocks every later row with it.  MEASURED: a
    resume from 1800 s of a run whose itinerary moved at 1200 s and
    2400 s executed NEITHER move; the nest stayed at its 1200 s
    footprint, the two frames after the missed 2400 s move differed from
    the uninterrupted run, and the restore banner said the resume
    reproduces that run bit for bit.  Rows behind the clock are therefore
    retired at the first consultation and named in the summary receipt
    (:meth:`behind_the_clock`), so the row still AHEAD of the clock
    fires.  Retiring is the right reading, not a convenience: a row
    behind a checkpoint executed before that checkpoint was written, and
    its result is already in the restored state.
    """

    def __init__(self, moves):
        self._pending = list(moves)
        self._behind: list = []

    def __call__(self, parent_state, nest_footprint, t, refinement=None):
        # the itinerary is time-driven, and a scripted move has nothing
        # to refine: the placement IS the input.
        del parent_state, nest_footprint, refinement
        now = float(t)
        while self._pending:
            at = float(self._pending[0].at_seconds)
            if at >= now - 1.0e-6 * max(1.0, abs(at)):
                break
            self._behind.append(self._pending.pop(0))
        if not self._pending:
            return None
        head = self._pending[0]
        if abs(float(head.at_seconds) - now) > (
                1.0e-6 * max(1.0, abs(float(head.at_seconds)))):
            return None
        self._pending.pop(0)
        return (int(head.di_parent_cells), int(head.dj_parent_cells))

    desired_shift = __call__

    def unconsumed(self):
        return tuple(move.to_json() for move in self._pending)

    def behind_the_clock(self):
        """Rows retired because the clock was already past them.

        Empty on a run that started at t = 0, where admission has already
        proved every row is on an opportunity inside the run.  Non-empty
        only on a resume, and then it is the itinerary this leg inherited
        rather than executed -- a fact the receipts must carry, because
        silently dropping a scripted move is the defect one layer over
        from silently never making it.
        """
        return tuple(move.to_json() for move in self._behind)


def _atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True),
                   encoding="utf-8")
    # Windows refuses the atomic rename while ANY reader holds the
    # destination open -- and this file exists precisely to be read
    # while the run is alive (a user tailing the receipts killed a 6 h
    # forecast with WinError 5 here, measured).  Receipts are telemetry:
    # retry briefly, then keep the run alive and leave the .tmp beside
    # the stale receipt rather than aborting integration over it.
    for delay in (0.05, 0.25, 1.0, 2.0):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(delay)
    try:
        os.replace(tmp, path)
    except PermissionError as error:
        warnings.warn(
            f"relocation receipts stayed stale at {path} (a reader holds "
            f"it open: {error}); the current receipts are in {tmp} and "
            "the next move rewrites both", stacklevel=2)


def _cadence_periods(cadence_seconds, schedule) -> int:
    """The cadence as whole periods; ``None`` means every period."""
    if cadence_seconds is None:
        return 1
    tick_den = int(schedule.clock.tick_den)
    period_ticks = int(schedule.period_ticks)
    cadence_ticks = Fraction(float(cadence_seconds)) * tick_den
    if (cadence_ticks.denominator != 1
            or int(cadence_ticks) % period_ticks != 0):
        raise RelocationRefusal(
            f"cadence_seconds = {cadence_seconds} does not resolve to a "
            f"whole number of complete cycle boundaries (period = "
            f"{period_ticks / tick_den} s); the config-time root-step "
            "check and this schedule disagree, which is a defect, not an "
            "input error")
    periods = int(cadence_ticks) // period_ticks
    if periods < 1:
        raise RelocationRefusal(
            f"cadence_seconds = {cadence_seconds} is shorter than one "
            "complete cycle; there is no boundary inside a period to "
            "relocate at")
    return periods


class RelocationRunner:
    """Consult the follow source on cadence and execute admissible moves.

    Construction is the route's job because the route owns the two things
    neither the mechanism nor the policy can invent: the
    ``on_child_built`` preparer that gives a rebuilt child its physics,
    and the model tree the provider watches.  ``staging`` defaults to
    ``"host"`` -- the production trade -- and is a parameter, not a
    config key, because it changes memory behaviour and nothing else.
    """

    def __init__(self, *, config, schedule, on_child_built,
                 provider=None, staging: str = "host",
                 reground_descendant=None,
                 receipts_path=None, initializer=None,
                 static_provenance=None, track_writer=None,
                 reach_clamp=None):
        if not getattr(config, "enabled", False):
            raise RelocationRefusal(
                "RelocationRunner requires [relocation] enabled = true; a "
                "disabled config schedules nothing and must not acquire a "
                "runner")
        if on_child_built is None:
            raise RelocationRefusal(
                "RelocationRunner requires the route's on_child_built "
                "preparer: the cold-start path attaches no physics driver, "
                "and a child rebuilt without one integrates nothing")
        if provider is None:
            if getattr(config, "moves", ()):
                provider = ManualMoveProvider(config.moves)
            elif getattr(config, "follow", None) is not None:
                from woof.core.storm_tracking import StormTracker

                provider = StormTracker(config.follow)
            else:
                raise RelocationRefusal(
                    "[relocation] names no follow source: no "
                    "[relocation.follow] tracker block, no "
                    "[[relocation.move]] itinerary, and no provider was "
                    "handed in.  A runner with nothing to consult is a "
                    "nest that silently never moves, so this refuses")
        follow = getattr(config, "follow", None)
        if (follow is not None and getattr(follow, "field", None) == "pressure"
                and getattr(follow, "level_hpa", ()) is None):
            from dataclasses import replace

            # Hurricane motion has a measured budget, not an arbitrary
            # per-move cap. Keep the overlap, reach and parent-halo bounds.
            config = replace(config, max_move_parent_cells=None)
        self.config = config
        self.tick_den = int(schedule.clock.tick_den)
        self.provider = provider
        self.on_child_built = on_child_built
        self.staging = str(staging)
        # None on a leaf mover, which is every mover before mid-tree
        # moves existed; relocate_child then keeps its refusal for a
        # mover that HAS children, so nothing silently changes.
        self.reground_descendant = reground_descendant
        self.receipts_path = (None if receipts_path is None
                              else Path(receipts_path))
        self.cadence_periods = _cadence_periods(
            getattr(config, "cadence_seconds", None), schedule)
        if hasattr(provider, "move_interval_seconds"):
            provider.move_interval_seconds = (
                self.cadence_periods * int(schedule.period_ticks)
                / int(schedule.clock.tick_den))
        #: The [relocation.containment] leg: the mover's parent slides to
        #: keep the mover contained, with the mover EARTH-FIXED under the
        #: slide.  The route wires the parent's own initializer/preparer
        #: pair through :meth:`wire_containment`; without that wiring a
        #: configured containment refuses at the first opportunity rather
        #: than silently never sliding.
        self.containment = getattr(config, "containment", None)
        self.containment_cadence_periods = (
            None if self.containment is None else _cadence_periods(
                self.containment.cadence_seconds
                if self.containment.cadence_seconds is not None
                else getattr(config, "cadence_seconds", None), schedule))
        self.containment_initializer = None
        self.containment_preparer = None
        self.containment_static_provenance = None
        self._containment_segment = None
        self.containment_moves_executed = 0
        #: The per-footprint child rebuild seam (2026-08-06 statics-on-
        #: move requirement): a route that rebuilds own-source statics at
        #: the new footprint passes its builder here, with the provenance
        #: it must state; None takes relocate_child's explicit
        #: parent-interpolated fallback, which every receipt records.
        self.initializer = initializer
        self.static_provenance = static_provenance
        #: The [[relocation.track]] streams
        #: (:class:`woof.core.storm_track_writer.TrackWriter`), or None.
        #: A DIAGNOSTIC: it renders the fix the tracker already produced
        #: and cannot move anything.  Its cadence joins the mover's and
        #: the containment leg's in :meth:`is_due`, so a stream asking
        #: for a finer interval gets its own opportunities.
        self.track_writer = track_writer
        self.track_cadence_periods = (
            None if track_writer is None
            else self._track_periods(track_writer, config, schedule))
        #: The ``reach_speed_m_s`` bound on the tracked mover
        #: (:class:`woof.core.nest_reach.ReachClamp`), or None for a
        #: scripted itinerary.  The statics corridor is sized to it, so
        #: it is enforced here on every route, like max_move_parent_cells.
        self.reach_clamp = reach_clamp
        self.receipts: list[dict] = []
        self.moves_executed = 0
        self.track_records = 0
        self._segment = None

    @staticmethod
    def _track_periods(track_writer, config, schedule) -> int:
        """The opportunity rhythm the track file needs.

        The writer then decides for itself whether it wants this
        particular boundary; the runner only has to be awake often
        enough.  No ``interval_seconds`` rides the mover's own cadence,
        which is the free case -- the fix already exists at those
        boundaries and nothing extra is reduced.
        """
        mover_periods = _cadence_periods(
            getattr(config, "cadence_seconds", None), schedule)
        interval = track_writer.stream.config.interval_seconds
        if interval is None:
            return mover_periods
        return min(mover_periods, _cadence_periods(interval, schedule))

    @classmethod
    def from_experiment(cls, exp, *, schedule, on_child_built,
                        provider=None, staging: str = "host",
                        receipts_path=None, initializer=None,
                        static_provenance=None, track_writer=None,
                        reground_descendant=None,
                        reach_clamp=None) -> "RelocationRunner":
        """Build from a validated experiment (the routes' entry point).

        ``provider`` overrides the config-derived source only for the
        no-source case (a route handing in a bespoke tracker); a config
        that already names a source refuses a second one.
        """
        relocation = exp.relocation
        if provider is not None and (
                getattr(relocation, "moves", ())
                or getattr(relocation, "follow", None) is not None):
            raise RelocationRefusal(
                "this [relocation] already names its follow source "
                "([relocation.follow] or [[relocation.move]]); a second, "
                "programmatic provider would silently shadow it")
        if reach_clamp is None and relocation.grid_id is not None:
            from woof.core.nest_reach import reach_clamp_for
            reach_clamp = reach_clamp_for(exp, int(relocation.grid_id))
        return cls(config=relocation, schedule=schedule,
                   on_child_built=on_child_built, provider=provider,
                   staging=staging, receipts_path=receipts_path,
                   initializer=initializer, track_writer=track_writer,
                   reground_descendant=reground_descendant,
                   static_provenance=static_provenance,
                   reach_clamp=reach_clamp)

    # -- the restart state -------------------------------------------------
    #
    # PURE STATE, both directions: no I/O, no checkpoint vocabulary, no
    # model.  ``woof.io.restart`` owns where a follower entry lives in the
    # header and fills the three keys that describe the TREE
    # (``kind``, ``current_placement``, ``declared_placement``); the runner
    # owns the four that describe its own history, and tolerates the
    # writer's three so a resume can hand the entry over whole.

    def state_json(self) -> dict:
        """This follower's half of a checkpoint's per-follower entry."""
        provider_state = getattr(self.provider, "state_json", None)
        timers = {} if provider_state is None else dict(provider_state())
        return {
            "segment": (None if self._segment is None
                        else self._segment.to_json()),
            "moves_executed": int(self.moves_executed),
            "last_proposal_t": timers.get("last_proposal_t"),
            "last_move_t": timers.get("last_move_t"),
            **{key: timers[key] for key in TRACKER_OPTIONAL_STATE_KEYS
               if key in timers},
        }

    def restore_state(self, entry) -> None:
        """Seed this follower from a checkpoint's per-follower entry.

        The segment is the part that cannot be recomputed: it counts moves
        off ONE base preparation, and a follower that resumes at generation
        zero chains its next move's record onto the base instead of onto
        its real predecessor -- so the digest the checkpoint's restart
        fingerprint was folded from can never be reproduced, and every
        later checkpoint of the resumed run is addressed by a history that
        did not happen.

        Validated in full before anything is applied, so a refusal leaves
        the runner exactly as it was.
        """
        if not isinstance(entry, dict):
            raise RelocationRefusal(
                "a follower restart entry must be a mapping, got "
                f"{type(entry).__name__}")
        unknown = sorted(set(entry) - set(RUNNER_STATE_KEYS)
                         - set(RUNNER_WRITER_KEYS)
                         - set(TRACKER_OPTIONAL_STATE_KEYS))
        if unknown:
            raise RelocationRefusal(
                f"a follower restart entry carries key(s) {unknown} this "
                "build does not know; a follower seeded from policy state "
                "it can only read in part moves the nest from a history "
                "that is not its own, so this refuses")
        missing = sorted(set(RUNNER_STATE_KEYS) - set(entry))
        if missing:
            raise RelocationRefusal(
                f"a follower restart entry is missing {missing}; the "
                "segment fixes which move the next record chains onto, the "
                "counter is the run's account of what it did, and the two "
                "cooldown anchors are what stop the nest moving again at "
                "the first boundary after the resume")

        raw_segment = entry["segment"]
        segment = (None if raw_segment is None
                   else RelocationSegment.from_json(raw_segment))
        moves = int(entry["moves_executed"])
        if moves < 0:
            raise RelocationRefusal(
                f"a follower reports {moves} executed moves; the run cannot "
                "have un-moved a nest")
        recorded = 0 if segment is None else int(segment.generation)
        if recorded != moves:
            raise RelocationRefusal(
                f"a follower reports {moves} executed move(s) but its "
                f"chain carries {recorded}; the counter and the placement "
                "history disagree, and restoring either one would leave the "
                "next move chaining onto the wrong predecessor")

        timers = {"last_proposal_t": entry["last_proposal_t"],
                  "last_move_t": entry["last_move_t"]}
        for key in TRACKER_OPTIONAL_STATE_KEYS:
            if key in entry:
                timers[key] = entry[key]
        provider_restore = getattr(self.provider, "restore_state", None)
        if provider_restore is None:
            held = sorted(key for key, value in timers.items()
                          if value is not None)
            if held:
                raise RelocationRefusal(
                    f"this follower's source ({type(self.provider).__name__}) "
                    f"holds no cooldown, but the checkpoint carries {held}. "
                    "Dropping a tracker's hysteresis lets the resumed nest "
                    "move at its first cadence boundary, so the mismatched "
                    "source refuses instead of silently discarding it")
        else:
            provider_restore(timers)

        self._segment = segment
        self.moves_executed = moves

    # -- receipts ---------------------------------------------------------

    def _record(self, model, entry: dict, *, unique: bool = False) -> dict:
        """Append one receipt row, or REPLACE the run's single unique row.

        ``unique`` belongs to the run-end summary, which is a statement
        about the whole run rather than about an instant: it is written
        whenever the executor finishes, and the executor finishes once per
        leg on a route that walks legs (``woof.runtime.walk_spawn_legs``)
        rather than once per run.  Appending gave a receipts list with two
        or more byte-identical summary rows in it, so a consumer counting
        them double-counted.

        The superseded row is DROPPED and the new one appended, not
        overwritten in place: an in-place rewrite would leave the summary
        stranded before the moves that followed it, and an idempotence
        guard that kept the first row would freeze ``moves_executed`` at
        the first leg's total.  One row, current, last.
        """

        entry = {"contract": RELOCATION_RUNNER_CONTRACT, **entry}
        fix = getattr(self.provider, "last_fix", None)
        dx = getattr(getattr(fix, "footprint", None), "parent_dx_m", None)
        if (entry.get("event") in ("held", "relocated")
                and fix is not None and fix.found is not None and dx is not None
                and fix.evidence.get("t") == entry.get("elapsed_seconds")):
            import math

            raw_i, raw_j = fix.raw_shift
            moved_i, moved_j = entry.get("executed_shift_parent_cells", (0, 0))
            scale = dx / 1000.
            entry["eye_offset_before_km"] = math.hypot(raw_i, raw_j) * scale
            entry["eye_offset_after_km"] = math.hypot(
                raw_i - moved_i, raw_j - moved_j) * scale
            entry["eye_tracker_field"] = fix.field_used
            if entry["event"] == "relocated" and "move_wall_seconds" in entry:
                print(f"d{self.config.grid_id:02d} move at "
                      f"{entry['elapsed_seconds']:.0f}s: eye-to-centre "
                      f"{entry['eye_offset_before_km']:.3f} -> "
                      f"{entry['eye_offset_after_km']:.3f} km; "
                      f"move wall {entry['move_wall_seconds']:.3f}s", flush=True)
        drain = getattr(self.provider, "drain_receipts", None)
        if callable(drain):
            tracker_rows = drain()
            if tracker_rows:
                entry["tracker_receipts"] = tracker_rows
        if unique:
            self.receipts[:] = [row for row in self.receipts
                                if row.get("event") != entry["event"]]
        self.receipts.append(entry)
        # The checkpoint writer holds the MODEL, not the runner, and it
        # has to ask what the tracker's hysteresis is standing at.  Same
        # attachment the receipts already use.
        owner = getattr(model, "_relocation_runner", None)
        in_collection = (isinstance(owner, RelocationRunnerCollection)
                         and owner.runners.get(int(self.config.grid_id)) is self)
        if in_collection:
            owner.record(int(self.config.grid_id), entry, unique=unique)
            model._relocation_receipts = owner.receipts
        else:
            model._relocation_receipts = self.receipts
            if owner is not self:
                model._relocation_runner = self
        if self.receipts_path is not None:
            # A re-armed slot's runner is a fresh one on the same path: the
            # earlier lives' rows (prior_receipts, handed on by the
            # collection) stay in the file ahead of this life's.
            _atomic_json(self.receipts_path, {
                "contract": RELOCATION_RUNNER_CONTRACT,
                "config": self.config.receipt(),
                "receipts": [*getattr(self, "prior_receipts", ()), *self.receipts],
            })
        return entry

    # -- the shift policy -------------------------------------------------

    def _clamped_shift(self, node, shift, elapsed_seconds: float = 0.0
                       ) -> tuple[int, int, list[str]]:
        """Bound the desired shift; every adjustment is named."""
        from dataclasses import replace as _replace

        from woof.core.nest_relocation import (_prevalidate_placement,
                                                clamp_shift_to_overlap)

        di, dj = (int(shift[0]), int(shift[1]))
        clamps: list[str] = []
        limit = self.config.max_move_parent_cells
        if limit is not None:
            limit = int(limit)
            bounded_i = max(-limit, min(limit, di))
            bounded_j = max(-limit, min(limit, dj))
            if (bounded_i, bounded_j) != (di, dj):
                clamps.append("max_move_parent_cells")
                di, dj = bounded_i, bounded_j

        # THE REACH BOUND, on the same footing: a bound, not a verdict.
        # The statics corridor covers the ground the nest can reach and
        # no more, so a move past reach_speed_m_s would have no statics;
        # the move it is allowed to make is made, and the receipt names
        # the bound.  Measured from the declared start at this model
        # time, so a run extended on restart moves as the longer run
        # would have.
        if self.reach_clamp is not None:
            reached_i, reached_j, reached = self.reach_clamp.bound(
                node, (di, dj), elapsed_seconds)
            if reached:
                clamps.append("reach_speed_m_s")
                di, dj = reached_i, reached_j

        # THE OVERLAP FLOOR IS A BOUND, NOT A VERDICT.  It used to be
        # enforced only by check_admissible, which raises -- so a follow
        # source proposing a move one cell too large ended the run rather
        # than making the move it was allowed to make.  The two bounds
        # are not independent (an overlap floor implies a per-axis bound
        # of 1 - sqrt(floor) of the nest's width in parent cells), so a
        # config whose max_move_parent_cells is the larger of the two
        # could refuse on its own maximum.  Clamped here, on the same
        # footing as the parent edge below, and named in the receipt.
        overlap_i, overlap_j, overlap_clamped = clamp_shift_to_overlap(
            di, dj, min_overlap_fraction=self.config.min_overlap_fraction,
            parent_grid_ratio=int(node.cfg.parent_grid_ratio),
            child_nx=int(node.cfg.run.nx), child_ny=int(node.cfg.run.ny))
        if overlap_clamped:
            clamps.append("min_overlap_fraction")
            di, dj = overlap_i, overlap_j

        # Walk back toward the current placement until register_nest's
        # +-2 SINT stencil admits the target.  The current placement is
        # live, so the walk terminates; each probe is CPU index tables.
        def admissible(candidate_di, candidate_dj) -> bool:
            candidate = _replace(
                node.cfg,
                i_parent_start=int(node.cfg.i_parent_start) + candidate_di,
                j_parent_start=int(node.cfg.j_parent_start) + candidate_dj)
            if (candidate.i_parent_start < 1
                    or candidate.j_parent_start < 1):
                return False
            try:
                _prevalidate_placement(candidate, node.parent)
            except ValueError:
                return False
            return True

        edge_clamped = False
        while (di, dj) != (0, 0) and not admissible(di, dj):
            edge_clamped = True
            if abs(di) >= abs(dj) and di != 0:
                di -= 1 if di > 0 else -1
            elif dj != 0:
                dj -= 1 if dj > 0 else -1
        if edge_clamped:
            clamps.append("parent_edge")
        return di, dj, clamps

    # -- the cadence hook -------------------------------------------------

    def next_deadline_ticks(self, clocks):
        """Next explicit moving/track deadline for the adaptive root.

        An absent cadence means every actual period; it must not pin an
        adaptive run to its nominal step. Scripted moves still name exact
        instants, independent of whether a cadence was supplied.
        """
        now = next(iter(clocks.values())).ticks
        tick_den = self.tick_den
        cadence = getattr(self.config, "cadence_seconds", None)
        intervals = [cadence]
        if self.containment is not None:
            intervals.append(self.containment.cadence_seconds or cadence)
        if self.track_writer is not None:
            intervals.append(self.track_writer.stream.config.interval_seconds or cadence)
        due = []
        for seconds in intervals:
            if seconds is not None:
                interval = int(Fraction(str(seconds)) * tick_den)
                due.append((now // interval + 1) * interval)
        for move in getattr(self.config, "moves", ()):
            instant = int(Fraction(str(move.at_seconds)) * tick_den)
            if instant > now:
                due.append(instant)
        return min(due) if due else None

    def is_due(self, model, clocks) -> bool:
        """Whether THIS boundary is a cadence opportunity.

        The same gate :meth:`on_period_begin` opens with, published so a
        caller can do work that must happen BEFORE the provider is
        consulted and that would be wasteful at every other boundary.  The
        streamed-domain projection is the one that needs it: a streamed
        parent's UH plane lives in its store, so the executor has to copy it
        onto the state before the tracker reduces it -- once per cadence,
        not once per parent step.
        """
        root_clock = clocks[model.root.cfg.grid_id]
        ticks = int(root_clock.ticks)
        period_ticks = int(model.schedule.period_ticks)
        due = ticks != 0 and ticks % (
            period_ticks * self.cadence_periods) == 0
        if self.containment_cadence_periods is not None:
            due = due or (ticks != 0 and ticks % (
                period_ticks * self.containment_cadence_periods) == 0)
        if self.track_cadence_periods is not None:
            due = due or (ticks != 0 and ticks % (
                period_ticks * self.track_cadence_periods) == 0)
        return due

    # -- the containment leg ----------------------------------------------

    def wire_containment(self, *, initializer, on_child_built,
                         static_provenance=None) -> None:
        """Hand the sliding parent its own rebuild pair.

        A separate pair from the mover's on purpose: the initializer is
        resolution-specific (it crops the PARENT's corridor, not the
        mover's) and the preparer holds per-domain capture state that
        must never be shared between two domains (the regrounder learned
        that the hard way).
        """
        self.containment_initializer = initializer
        self.containment_preparer = on_child_built
        self.containment_static_provenance = static_provenance

    def _reground_releasing(self, before_rebuild):
        """Drop executor references before the route rebuilds a descendant."""
        handler = self.reground_descendant
        if handler is None or before_rebuild is None:
            return handler

        def reground(*, node, plan, delta_parent_cells):
            before_rebuild(int(node.cfg.grid_id))
            return handler(node=node, plan=plan,
                           delta_parent_cells=delta_parent_cells)

        return reground

    @staticmethod
    def _check_rebuilt_tree(receipt, grid_id, after_rebuild):
        if after_rebuild is None:
            return
        after_rebuild(grid_id)
        for descendant in receipt.get("descendants", ()):
            # Compensated descendants keep their state and validator.
            if not descendant.get("earth_fixed", False):
                after_rebuild(int(descendant["grid_id"]))

    def _containment_opportunity(self, model, clocks, elapsed,
                                 period, before_rebuild, after_rebuild=None):
        """One containment consultation: slide the mover's parent when
        the mover has drifted past the dead-band, with the mover
        earth-fixed under the slide.  Records rows only when it acts or
        refuses -- an in-band hold is silent, because the mover rows
        already carry the placement every consultation reads."""
        cont = self.containment
        mover = model.node(int(self.config.grid_id))
        parent_node = model.node(int(cont.grid_id))
        if not bool(getattr(mover, "_started", True)) or not bool(
                getattr(parent_node, "_started", True)):
            return None
        ratio_m = int(mover.cfg.parent_grid_ratio)
        span_i = int(mover.cfg.run.nx) // ratio_m
        span_j = int(mover.cfg.run.ny) // ratio_m
        centered_i = (int(parent_node.cfg.run.nx) - span_i) // 2 + 1
        centered_j = (int(parent_node.cfg.run.ny) - span_j) // 2 + 1
        dev_i = int(mover.cfg.i_parent_start) - centered_i
        dev_j = int(mover.cfg.j_parent_start) - centered_j
        deadband = int(cont.deadband_cells)
        if max(abs(dev_i), abs(dev_j)) < deadband:
            return None
        if (self.containment_initializer is None
                or self.containment_preparer is None):
            return self._record(model, {
                "event": "containment_refused",
                "elapsed_seconds": elapsed,
                "grid_id": int(cont.grid_id),
                "reason": "[relocation.containment] is configured but the "
                          "route wired no initializer/preparer pair for "
                          "the sliding parent (wire_containment); the "
                          "mover keeps tracking inside a frame that "
                          "cannot slide"})
        ratio_p = int(parent_node.cfg.parent_grid_ratio)
        # The slide the deviation ASKS for, before any bound.  Recorded
        # as the requested shift for the same reason the mover records
        # the provider's own proposal there: a row whose requested and
        # executed shifts are equal because the request was rewritten in
        # place cannot show which bound did the rewriting.
        want_i = int(round(dev_i / ratio_p))
        want_j = int(round(dev_j / ratio_p))
        # Every bound that cuts this slide is NAMED, on the same footing
        # as the mover's: the row a reader audits has to say which one
        # moved the number, and a bare `clamped = True` says only that
        # something did.
        clamps: list[str] = []
        capped_i, capped_j = want_i, want_j
        limit = cont.max_move_parent_cells
        if limit is not None:
            limit = int(limit)
            capped_i = max(-limit, min(limit, want_i))
            capped_j = max(-limit, min(limit, want_j))
            if (capped_i, capped_j) != (want_i, want_j):
                clamps.append("containment.max_move_parent_cells")

        # Admissible for BOTH: the slid parent inside its own parent, and
        # the earth-fixed mover's compensated placement inside the slid
        # parent.  The walk toward zero mirrors _clamped_shift.
        from dataclasses import replace as _replace

        from woof.core.nest_relocation import _prevalidate_placement

        def refusal(di, dj) -> str | None:
            """Which of the two bands refuses this slide, by name.

            One walk answers to two owners -- the slid parent inside its
            own parent, and the earth-fixed mover's compensated placement
            inside the slid parent -- and a row that named neither left a
            reader guessing which edge the slide met.
            """
            cand = _replace(
                parent_node.cfg,
                i_parent_start=int(parent_node.cfg.i_parent_start) + di,
                j_parent_start=int(parent_node.cfg.j_parent_start) + dj)
            comp = _replace(
                mover.cfg,
                i_parent_start=int(mover.cfg.i_parent_start)
                - di * ratio_p,
                j_parent_start=int(mover.cfg.j_parent_start)
                - dj * ratio_p)
            if cand.i_parent_start < 1 or cand.j_parent_start < 1:
                return "parent_edge"
            try:
                _prevalidate_placement(cand, parent_node.parent)
            except ValueError:
                return "parent_edge"
            if comp.i_parent_start < 1 or comp.j_parent_start < 1:
                return "mover_compensated_placement"
            try:
                _prevalidate_placement(comp, parent_node)
            except ValueError:
                return "mover_compensated_placement"
            return None

        di, dj = capped_i, capped_j
        # The floor applies to the SLIDING ANCESTOR too: relocate_child
        # below is handed this runner's own min_overlap_fraction re-aimed
        # at the ancestor's grid_id, and the ancestor has its own extent
        # and ratio, so its admissible band is its own.  Clamped on the
        # same footing as the mover's, for the same reason: a slide one
        # cell too large must make the slide it is allowed to make, not
        # end the run.
        from woof.core.nest_relocation import clamp_shift_to_overlap

        di, dj, overlap_clamped = clamp_shift_to_overlap(
            di, dj, min_overlap_fraction=self.config.min_overlap_fraction,
            parent_grid_ratio=int(parent_node.cfg.parent_grid_ratio),
            child_nx=int(parent_node.cfg.run.nx),
            child_ny=int(parent_node.cfg.run.ny))
        if overlap_clamped:
            clamps.append("min_overlap_fraction")
        while (di, dj) != (0, 0):
            met = refusal(di, dj)
            if met is None:
                break
            if met not in clamps:
                clamps.append(met)
            if abs(di) >= abs(dj) and di != 0:
                di -= 1 if di > 0 else -1
            elif dj != 0:
                dj -= 1 if dj > 0 else -1
        clamped = bool(clamps)
        if (di, dj) == (0, 0):
            return self._record(model, {
                "event": "containment_held",
                "elapsed_seconds": elapsed,
                "grid_id": int(cont.grid_id),
                "mover_deviation_cells": [dev_i, dev_j],
                "requested_shift_parent_cells": [want_i, want_j],
                "clamped_by": clamps,
                "reason": "the requested slide clamps to the null move at "
                          "the admissible band (parent edge, the mover's "
                          "compensated placement, or the overlap floor)"})
        if self._containment_segment is None:
            self._containment_segment = base_segment(parent_node.cfg)
        # BEFORE the rebuild: relocate_child reassigns the node's grid, so
        # the position it is sliding away from exists only until then.
        slid_from = centre_latlon(getattr(parent_node, "grid", None))
        capture = getattr(self.containment_preparer, "capture_outgoing",
                          None)
        if callable(capture):
            capture(parent_node)
        fingerprint_before = model.experiment_fingerprint
        # The governance block re-aimed at the sliding parent: bounds
        # authorise BY GRID ID, and the config's own id names the tracked
        # mover.  Everything else (mode opt-in, min_overlap_fraction)
        # carries over; the containment table is dropped from the shim so
        # the config's own strict-ancestor validation does not read the
        # re-aimed id as self-containment.
        bounds = _replace(self.config, grid_id=int(cont.grid_id),
                          containment=None)
        reconstruction_args = self._reconstruction_args(
            parent_node, self.containment_initializer, self.containment_preparer)
        receipt = relocate_child(
            parent_node,
            i_parent_start=int(parent_node.cfg.i_parent_start) + di,
            j_parent_start=int(parent_node.cfg.j_parent_start) + dj,
            segment=self._containment_segment, bounds=bounds,
            initializer=self.containment_initializer,
            static_provenance=self.containment_static_provenance,
            on_child_built=self.containment_preparer,
            scratch_arena=getattr(model, "_scratch_arena", None),
            dycore_state_workspace=getattr(
                model, "_dycore_state_workspace", None),
            staging=self.staging,
            reground_descendant=self._reground_releasing(before_rebuild),
            earth_fixed_descendants=frozenset(
                {int(self.config.grid_id)}),
            on_before_release=(
                None if before_rebuild is None
                else (lambda: before_rebuild(int(cont.grid_id)))),
            **reconstruction_args)
        self._containment_segment = receipt["segment_state"]
        self.containment_moves_executed += 1
        from woof.core.state import refresh_model_time

        refresh_model_time(parent_node.state, parent_node.clock)
        after_move = getattr(self.containment_preparer, "after_move", None)
        if callable(after_move):
            after_move(parent_node)
        # A slide rebuilt the ancestor even when the mover holds or is
        # not due. Rebind the executor's released health validator before
        # recording success or letting the tracker read the new parent.
        self._check_rebuilt_tree(receipt, int(cont.grid_id), after_rebuild)
        record_sha = receipt["record_sha256"]
        model.experiment_fingerprint = mark_fingerprint_across_move(
            fingerprint_before, record_sha)
        components = getattr(
            model, "_experiment_fingerprint_components", None)
        if components is not None:
            marked = dict(components)
            history = list(marked.get("relocation", {}).get("records", []))
            history.append(record_sha)
            marked["relocation"] = {"records": history}
            model._experiment_fingerprint_components = marked
        # The live copy of this receipt, for anything watching the run
        # rather than reading it afterwards.
        model_step_log(model).containment_moved(
            domain=int(cont.grid_id), model_seconds=elapsed,
            mover=int(self.config.grid_id),
            placement_from=receipt["plan"]["placement_from"],
            placement_to=receipt["plan"]["placement_to"],
            requested_shift=[want_i, want_j],
            executed_shift=[di, dj], clamped=clamped,
            clamped_by=clamps,
            mover_deviation_cells=[dev_i, dev_j],
            grid=getattr(parent_node, "grid", None),
            lat_from=slid_from[0], lon_from=slid_from[1])
        return self._record(model, {
            "event": "contained",
            "elapsed_seconds": elapsed,
            "period": (None if period is None else int(period)),
            "grid_id": int(cont.grid_id),
            "mover_deviation_cells": [dev_i, dev_j],
            "requested_shift_parent_cells": [want_i, want_j],
            "executed_shift_parent_cells": [di, dj],
            "clamped": clamped,
            # WHICH bound, and what the slide kept.  The floor reaches a
            # slide as a clamp now rather than as a refusal, so the only
            # thing that had kept an under-floor slide off a ledger is
            # gone; the auditor's floor check reads this field, and
            # skips any row that does not carry it.
            "clamped_by": clamps,
            "overlap_fraction": receipt["plan"]["overlap_fraction"],
            "placement_from": receipt["plan"]["placement_from"],
            "placement_to": receipt["plan"]["placement_to"],
            # The earth-fixed compensation rows: the tracked mover's
            # placement change and its bitwise state carry.
            "descendants": receipt.get("descendants", []),
            "donor_alignment_pass": bool(
                receipt["donor_alignment"]["pass"]),
            "parent_bitwise_unchanged": bool(
                receipt["parent_bitwise_unchanged"]),
            "segment_id": receipt["segment"]["segment_id"],
            "generation": receipt["segment"]["generation"],
            "record_sha256": record_sha,
            "experiment_fingerprint": {
                "before": fingerprint_before,
                "after": model.experiment_fingerprint,
            },
        })

    # -- the track leg ----------------------------------------------------

    def _track_opportunity(self, model, elapsed, *, locate: bool, force=False):
        """Render one consultation into the configured track streams.

        ``locate=False`` is the boundary the mover was consulted on: the
        provider's own fix is reused, so a track row at the default
        interval costs a format call and nothing else.  ``locate=True``
        is a track-only boundary, where the writer asks for a fix of its
        own through the stateless :meth:`StormTracker.locate` seam.

        NEVER RAISES, and that is a contract rather than caution.  A
        track file is a diagnostic; a diagnostic that can end a
        twelve-hour forecast is a defect.  Every fault lands on the
        writer's own receipt, exactly as a tracker refusal lands on a
        receipt row instead of stopping the run.
        """
        writer = self.track_writer
        if writer is None:
            return None
        try:
            node = model.node(int(self.config.grid_id))
            if not bool(getattr(node, "_started", True)):
                return None
            fix = getattr(self.provider, "last_fix", None)
            stale = fix is None or float(
                fix.evidence.get("t", -1.0)) != float(elapsed)
            if getattr(writer, "position_only", False):
                # A rotation or echo track row is the MOVER'S OWN centre
                # (storm_track_writer.POSITION_ONLY_FIELDS), which the fix
                # does not enter -- so a track-only boundary skips the
                # locate rather than pay for a whole-plane reduction whose
                # answer never reaches the file.  The consultation's own
                # fix is still reused when it is fresh: it costs nothing
                # and it is what tells the receipt whether the tracker saw
                # anything on this beat.
                if stale:
                    fix = None
            elif locate or stale:
                locator = getattr(self.provider, "locate", None)
                if locator is None:
                    return None
                from woof.core.storm_tracking import refinement_from_node

                refine_grid_id = getattr(
                    getattr(self.config, "follow", None),
                    "refine_grid_id", None)
                refinement = (
                    None if refine_grid_id is None
                    else refinement_from_node(node, int(refine_grid_id)))
                fix = locator(node.parent.state, node.cfg, float(elapsed),
                              refinement=refinement)
            refine_node = (None if fix is None or fix.refined_on is None
                           else model.node(int(fix.refined_on)))
            diagnostic_nodes = None
            swath_grid = None
            if not getattr(writer, "position_only", False) and getattr(writer, "surface", False):
                diagnostic_nodes = sorted(
                    (n for n in model.walk_parent_first()
                     if bool(getattr(n, "_started", True)) and getattr(n, "grid", None) is not None),
                    key=lambda n: float(n.cfg.run.dx))
                if not hasattr(self, "_tc_swath_grid"):
                    root = next(iter(model.walk_parent_first()))
                    g = root.grid
                    points = [g.ij_to_latlon(x, y)
                              for x in (1., g.e_we/2., g.e_we-1.)
                              for y in (1., g.e_sn/2., g.e_sn-1.)]
                    anchor = float(points[4][1])
                    longitudes = [anchor+(float(p[1])-anchor+180.)%360.-180. for p in points]
                    self._tc_swath_grid = dict(
                        path=str(writer.stream.path.with_suffix('.swath.f32')),
                        south=min(float(p[0]) for p in points), north=max(float(p[0]) for p in points),
                        west=min(longitudes), east=max(longitudes), spacing_deg=0.025)
                swath_grid = self._tc_swath_grid
            row = writer.emit(
                fix, t=float(elapsed),
                parent_state=node.parent.state,
                parent_grid=getattr(node.parent, "grid", None),
                # The mover's own grid, rebuilt by relocate_child on every
                # move, so a position-only row is where the nest is NOW.
                mover_grid=getattr(node, "grid", None),
                refine_state=(None if refine_node is None
                              else refine_node.state),
                refine_grid=(None if refine_node is None
                             else getattr(refine_node, "grid", None)),
                diagnostic_nodes=diagnostic_nodes, swath_grid=swath_grid,
                **({"force": True} if force else {}))
        except Exception as error:                   # noqa: BLE001
            return self._record(model, {
                "event": "track_faulted",
                "elapsed_seconds": float(elapsed),
                "reason": f"the track writer could not be served at this "
                          f"boundary and the run continues: {error}"})
        if row is not None and row.get("emitted"):
            self.track_records += 1
            # OUTSIDE the try above, deliberately: an emit fault here is
            # not a track fault, and recording it as one would name the
            # wrong subsystem on the run's receipts.  One row written,
            # one event, at the same instant and with the same position.
            model_step_log(model).track_fix(
                domain=int(self.config.grid_id), model_seconds=float(elapsed),
                lat=row.get("lat"), lon=row.get("lon"),
                found=not bool(row.get("no_signal")),
                refined_on=(None if fix is None else fix.refined_on))
        return row

    def continuity_state(self) -> dict:
        """What a resume has to land on to keep tracking coherent.

        Placement gets the tree's GEOMETRY back; this is the rest of it.
        Without the cooldown anchor a resumed run's hysteresis starts
        cold, so its first consultation is free to move on a beat the
        unbroken run was still suppressing -- a different forecast, from
        a bookkeeping scalar.  ``moves_executed`` is carried for the same
        reason the receipts are: an audit that restarts at zero cannot
        be reconciled with the run it continues.
        """
        return {
            "moves_executed": int(self.moves_executed),
            "last_move_t": (
                None if getattr(self.provider, "_last_move_t", None) is None
                else float(self.provider._last_move_t)),
        }

    def restore_continuity(self, state) -> dict:
        """Put :meth:`continuity_state` back after a restore.

        Duck-typed on the provider, like ``notify_move_executed``: a
        programmatic provider with no cooldown of its own simply has
        nothing to set, and says so in the returned row rather than
        failing.
        """
        applied = {}
        moves = state.get("moves_executed")
        if moves is not None:
            self.moves_executed = int(moves)
            applied["moves_executed"] = int(moves)
        last = state.get("last_move_t")
        if last is not None and hasattr(self.provider, "_last_move_t"):
            self.provider._last_move_t = float(last)
            applied["last_move_t"] = float(last)
        elif last is not None:
            applied["last_move_t_skipped"] = (
                "provider carries no cooldown anchor")
        return applied

    def _reconstruction_args(self, node, initializer, preparer):
        factory = getattr(self, "streamed_reconstruction_factory", None)
        if callable(factory) and getattr(node.state, "_streamed_domain", None) is not None:
            return {"reconstruction": factory(node, initializer=initializer, preparer=preparer)}
        return {}

    def adopt_placement(self, model, node, *, i_parent_start: int,
                        j_parent_start: int, force: bool = False) -> dict:
        """Rebuild ``node`` at an ABSOLUTE placement, for a restore.

        A checkpoint records where each domain actually was.  A tree
        built from the config puts a moved nest back at its original
        placement, and every setup array it owns -- terrain, map factors,
        base state, the parent's SINT donor tables -- then describes the
        wrong ground, which ``setup_fingerprint`` refuses.  This puts the
        node where the checkpoint says it was, BEFORE any state is
        restored into it.

        It is the same mechanism a move uses, with the same corridor
        wiring, because the geometry problem is identical: build the
        child afresh at a placement and rebuild the donor tables for it.
        What it deliberately does NOT do is pretend a move happened --
        the tracker is not told (no cooldown burns), ``moves_executed``
        does not advance, and the receipt is its own event.  The storm
        did not move here; the run is being put back where it was.

        Temporary state is rebuilt without carrying a live-move overlap.
        The complete checkpoint supplies continuation. Descendants can be disjoint from their
        cold-start ground after a long storm track; they still rebuild
        their statics and donor geometry before checkpoint restoration.
        """
        from dataclasses import replace as _replace

        from woof.core.nest_relocation import relocate_child
        from woof.core.state import refresh_model_time

        grid_id = int(node.cfg.grid_id)
        was = (int(node.cfg.i_parent_start), int(node.cfg.j_parent_start))
        want = (int(i_parent_start), int(j_parent_start))
        if was == want and not force:
            return {"event": "placement-already-correct",
                    "grid_id": grid_id, "placement": list(want)}
        # ``force`` exists because the placement NUMBER is not the whole
        # story.  A nest that moved away and came back is at its original
        # i/j with setup arrays that came from a relocation rebuild
        # rather than the prepared cache, and those are not the same
        # bytes -- phb measured 2**-6 apart.  A resume must re-run the
        # rebuild for any grid the checkpointed run relocated, whatever
        # the numbers say.
        preparer = (self.containment_preparer if int(self.config.grid_id) != grid_id
                    and self.containment_preparer is not None else self.on_child_built)
        for_restore = getattr(preparer, "for_checkpoint_restore", None)
        if callable(for_restore):
            preparer = for_restore()
        capture = getattr(preparer, "capture_outgoing", None)
        if callable(capture):
            capture(node)
        if self._segment is None:
            self._segment = base_segment(node.cfg)
        bounds = self.config
        if int(self.config.grid_id) != grid_id:
            # A containment ancestor being put back: authorise BY GRID
            # ID, the same re-aim the containment leg makes.
            bounds = _replace(self.config, grid_id=grid_id,
                              containment=None)
        # A RESTORE IS NOT A STEERING DECISION.  max_move_parent_cells
        # and min_overlap_fraction bound what one CONSULTATION may do --
        # they exist so a single noisy centroid cannot fling the nest
        # across the domain and expose a spin-up strip.  Putting a nest
        # back where the checkpoint says it was is neither noisy nor a
        # decision: the distance is however far the storm travelled, and
        # the state about to be written over it comes from the
        # checkpoint, so there is no integrated state for an overlap
        # floor to protect.  Judged against the per-move cap, a resume
        # after a day of following a storm refuses ("the move is 8 parent
        # cells, over the configured maximum of 2") -- and the `force`
        # flag never reached this, because it only skipped the no-op
        # early return above.
        #
        # The checks that DO still apply are the ones about authority
        # rather than magnitude: relocation enabled, and this grid being
        # the one [relocation] names.
        bounds = _replace(bounds, max_move_parent_cells=None,
                          min_overlap_fraction=None)
        # The containment ancestor has its OWN initializer -- the mover's
        # serves one grid and refuses any other ("this initializer serves
        # grid_id 3, asked to rebuild grid_id 2").  This is the same
        # routing `_containment_opportunity` makes when it slides that
        # grid during a run; a restore has to make it too.
        initializer = self.initializer
        if (int(self.config.grid_id) != grid_id
                and self.containment_initializer is not None):
            initializer = self.containment_initializer
        reconstruction_args = self._reconstruction_args(node, initializer, preparer)
        receipt = relocate_child(
            node,
            i_parent_start=want[0], j_parent_start=want[1],
            segment=self._segment, bounds=bounds,
            initializer=initializer,
            static_provenance=self.static_provenance,
            on_child_built=preparer,
            scratch_arena=getattr(model, "_scratch_arena", None),
            dycore_state_workspace=getattr(
                model, "_dycore_state_workspace", None),
            staging=self.staging,
            reground_descendant=self.reground_descendant,
            restoring_checkpoint=True,
            **reconstruction_args)
        self._segment = receipt["segment_state"]
        refresh_model_time(node.state, node.clock)
        after_move = getattr(preparer, "after_move", None)
        if callable(after_move):
            after_move(node)
        return self._record(model, {
            "event": "placement-restored",
            "grid_id": grid_id,
            "placement_from": {"i_parent_start": was[0],
                               "j_parent_start": was[1]},
            "placement_to": {"i_parent_start": want[0],
                             "j_parent_start": want[1]},
            "donor_alignment_pass": receipt.get("donor_alignment_pass"),
            "parent_bitwise_unchanged": receipt.get(
                "parent_bitwise_unchanged"),
        })

    def on_period_begin(self, model, clocks, *, period=None,
                        before_rebuild=None, after_rebuild=None):
        """One cadence opportunity; returns the receipt row or ``None``.

        ``before_rebuild`` is the executor's seam for dropping references
        it holds to the outgoing child (health validators) so the
        host-staged release actually frees; it receives the grid_id.
        ``after_rebuild`` rebinds and checks executor-owned health state
        after each accepted move or containment slide. Its work belongs
        to the complete move timer, while the remap timer ends before
        the moved child's check.
        """
        root_clock = clocks[model.root.cfg.grid_id]
        ticks = int(root_clock.ticks)
        period_ticks = int(model.schedule.period_ticks)
        mover_due = ticks != 0 and ticks % (
            period_ticks * self.cadence_periods) == 0
        if mover_due:
            # A nest SPAWNED at this boundary was just centred on its storm
            # by its trigger and has not stepped: it waits for its next
            # cadence.  Moving it here re-derived a state that had never
            # been integrated, and the moved spawned nest went non-finite
            # within eight steps (a rented box, 19:23Z, HRRR storm nest d03).
            mover = (getattr(model, "nodes_by_grid_id", None) or {}).get(
                int(self.config.grid_id))
            if mover is not None and getattr(mover, "_born_at_ticks", None) == ticks:
                mover_due = False
        containment_due = (
            self.containment_cadence_periods is not None
            and ticks != 0
            and ticks % (period_ticks
                         * self.containment_cadence_periods) == 0)
        # THE TRACK EMITS AT t = 0, and the mover and the containment leg
        # do not.  A track's first row is the run's own initial
        # position, and a track that starts one cadence in has silently
        # dropped it.  on_period_begin also fires at the START of each
        # period, so t = run_seconds is never an opportunity -- the last
        # row of an N-second run is at the last boundary before N, and
        # without t = 0 the initial position has nowhere to come from.
        #
        # Safe because locate() is stateless and silent: consulting the
        # tracker at t = 0 appends no receipt, burns no cooldown, and
        # cannot move anything.  The mover keeps ticks != 0, because a
        # relocation at t = 0 would move a nest before it has integrated.
        track_due = (
            self.track_cadence_periods is not None
            and (ticks == 0
                 or ticks % (period_ticks * self.track_cadence_periods) == 0))
        if not mover_due and not containment_due and not track_due:
            return None
        opportunity_started = time.perf_counter()
        elapsed = float(root_clock.elapsed_seconds)
        if containment_due:
            # The slide first, so the tracker then evaluates the storm in
            # the already-slid frame: a slide plus a track in one
            # opportunity re-centers AND follows, instead of following
            # first and immediately un-centering what it just did.
            self._containment_opportunity(
                model, clocks, elapsed, period, before_rebuild, after_rebuild)
        if not mover_due:
            if track_due:
                # A track-only boundary: locate WITHOUT proposing, so the
                # writer gets a fix and the tracker's hysteresis is
                # untouched (StormTracker.locate is stateless and
                # silent).  Nothing is reset here either -- the UH window
                # belongs to the mover's consultation and stays there, so
                # however often a track stream looks, the nest goes to
                # exactly the same places.
                self._track_opportunity(model, elapsed, locate=True)
            return None
        grid_id = int(self.config.grid_id)
        node = model.node(grid_id)
        if not bool(getattr(node, "_started", True)):
            if track_due:
                self._track_opportunity(model, elapsed, locate=True)
            return self._record(model, {
                "event": "held", "elapsed_seconds": elapsed,
                "reason": f"d{grid_id:02d} has not started; a domain that "
                          "does not exist yet cannot move"})
        # The tracker seam, verbatim: parent state, footprint, model time
        # -- plus the optional second stage.  The runner is the only side
        # that can walk the tree, so it builds the refinement source and
        # the tracker stays a pure function of the fields it is handed.
        # A provider that wants no refinement never sees the argument.
        from woof.core.storm_tracking import refinement_from_node

        refinement = None
        refine_grid_id = getattr(
            getattr(self.config, "follow", None), "refine_grid_id", None)
        if refine_grid_id is not None:
            refinement = refinement_from_node(node, int(refine_grid_id))
        # Passed ONLY when there is one, so the seam a provider without a
        # second stage sees is byte-for-byte the three-argument call it
        # has always been handed.
        desired = (
            self.provider(node.parent.state, node.cfg, elapsed,
                          refinement=refinement)
            if refinement is not None
            else self.provider(node.parent.state, node.cfg, elapsed))
        # This consumer's window is "max since I last looked", so it is
        # zeroed HERE -- immediately after the look, before any branch on
        # what the look returned, and on HELD evaluations exactly as on
        # accepted ones (ArWen's ruling, 2026-08-07).  Resetting only when
        # a move is accepted would let the window grow across holds and
        # cooldowns, which makes the next move more likely just because
        # more time passed: cadence dependence coming back in through the
        # reset rule.  Tolerant of a provider that reads no UH at all (a
        # scripted move list) and of a state with no window allocated.
        follow_slot = getattr(self.provider, "uh_slot", UH_FOLLOW_WINDOW_SLOT)
        reset_tracker_window(node.parent.state, follow_slot)
        if track_due:
            # The FREE case: the provider just located the storm, so the
            # writer renders the fix that consultation produced rather
            # than paying for a second reduction to re-derive it.  Before
            # the branch on `desired`, so a held cadence still writes a
            # track row -- the storm was there whether or not the nest
            # was allowed to follow it.
            self._track_opportunity(model, elapsed, locate=False)
        if desired is None:
            return self._record(model, {
                "event": "held", "elapsed_seconds": elapsed,
                "reason": "follow source returned None (hold)"})
        di, dj = (int(desired[0]), int(desired[1]))
        if (di, dj) == (0, 0):
            return self._record(model, {
                "event": "held", "elapsed_seconds": elapsed,
                "reason": "follow source requested the null shift"})
        executed_i, executed_j, clamps = self._clamped_shift(
            node, (di, dj), elapsed)
        if (executed_i, executed_j) == (0, 0):
            return self._record(model, {
                "event": "held", "elapsed_seconds": elapsed,
                "requested_shift_parent_cells": [di, dj],
                "clamped_by": clamps,
                "reason": "the requested shift clamps to the null move at "
                          "the parent's admissible band"
                          + (" or the nest's reach (reach_speed_m_s)"
                             if "reach_speed_m_s" in clamps else "")})
        if self._segment is None:
            self._segment = base_segment(node.cfg)
        # The preparer's pre-move seam, duck-typed and optional: a
        # real-data preparer snapshots the outgoing child's driver-held
        # land-surface state (and the statics it will hold the rebuilt
        # footprint's overlap against) while the outgoing child is still
        # whole -- on_child_built itself fires after the host-staged
        # release, when nothing of the outgoing driver remains.
        move_started = time.perf_counter()
        capture = getattr(self.on_child_built, "capture_outgoing", None)
        if callable(capture):
            capture(node)
        # BEFORE the rebuild: relocate_child reassigns the node's grid, so
        # the position the nest is LEAVING exists only until then, and a
        # live map wants it for the origin ghost.
        moved_from = centre_latlon(getattr(node, "grid", None))
        fingerprint_before = model.experiment_fingerprint
        reconstruction_args = self._reconstruction_args(node, self.initializer, self.on_child_built)
        receipt = relocate_child(
            node,
            i_parent_start=int(node.cfg.i_parent_start) + executed_i,
            j_parent_start=int(node.cfg.j_parent_start) + executed_j,
            segment=self._segment, bounds=self.config,
            initializer=self.initializer,
            static_provenance=self.static_provenance,
            on_child_built=self.on_child_built,
            scratch_arena=getattr(model, "_scratch_arena", None),
            dycore_state_workspace=getattr(
                model, "_dycore_state_workspace", None),
            staging=self.staging,
            reground_descendant=self._reground_releasing(before_rebuild),
            on_before_release=(
                None if before_rebuild is None
                else (lambda: before_rebuild(grid_id))),
            **reconstruction_args)
        self._segment = receipt["segment_state"]
        self.moves_executed += 1
        from woof.core.state import refresh_model_time

        refresh_model_time(node.state, node.clock)
        # The preparer's post-move seam: the node now carries the new
        # placement, grid and state, which is what output metadata and
        # the route's prepared-case bookkeeping must be refreshed from.
        after_move = getattr(self.on_child_built, "after_move", None)
        if callable(after_move):
            after_move(node)
        # Accepted-move semantics for providers that want it (a tracker's
        # cooldown should burn on acceptance, not on a clamped-away
        # proposal); the hook is optional and duck-typed.
        notify = getattr(self.provider, "notify_move_executed", None)
        if callable(notify):
            notify(elapsed, (executed_i, executed_j))
        # Chain the record into the live fingerprint so later checkpoints
        # are keyed to the move history: a fresh build is refused by name,
        # and a restore of this run's own checkpoint reaches the same
        # value by replaying the same digests in the same order.
        record_sha = receipt["record_sha256"]
        model.experiment_fingerprint = mark_fingerprint_across_move(
            fingerprint_before, record_sha)
        components = getattr(
            model, "_experiment_fingerprint_components", None)
        if components is not None:
            marked = dict(components)
            history = list(marked.get("relocation", {}).get("records", []))
            history.append(record_sha)
            marked["relocation"] = {"records": history}
            model._experiment_fingerprint_components = marked
        transplant = receipt["transplant"]
        plan = receipt["plan"]
        stamped_cells = sum(
            int(item["stamped_cells"])
            for item in transplant["stamped"].values())
        strip_fill_source = getattr(
            self.initializer, "strip_fill_source", STRIP_FILL_SOURCE)
        land_surface = getattr(self.on_child_built, "last_receipt", None)
        # The live copy of this receipt.  Only an EXECUTED move is an
        # event: a hold redraws nothing, and the receipts above carry
        # every one of them with its reason.
        model_step_log(model).nest_moved(
            domain=grid_id, model_seconds=elapsed,
            placement_from=plan["placement_from"],
            placement_to=plan["placement_to"],
            requested_shift=[di, dj],
            executed_shift=[executed_i, executed_j],
            clamped_by=clamps, grid=getattr(node, "grid", None),
            lat_from=moved_from[0], lon_from=moved_from[1])
        remap_wall_seconds = time.perf_counter() - move_started
        self._check_rebuilt_tree(receipt, grid_id, after_rebuild)
        return self._record(model, {
            "event": "relocated",
            "elapsed_seconds": elapsed,
            "move_wall_seconds": time.perf_counter() - opportunity_started,
            "remap_wall_seconds": remap_wall_seconds,
            "period": (None if period is None else int(period)),
            "grid_id": grid_id,
            "requested_shift_parent_cells": [di, dj],
            "executed_shift_parent_cells": [executed_i, executed_j],
            "clamped_by": clamps,
            "placement_from": plan["placement_from"],
            "placement_to": plan["placement_to"],
            "overlap_fraction": plan["overlap_fraction"],
            "fill": {
                "overlap_cells": plan["overlap_cells"],
                "spin_up_cells": plan["spin_up_cells"],
                "strip_fill_source": strip_fill_source,
                "static_fields": receipt["child_rebuild"]["static_fields"],
                "initializer": receipt["child_rebuild"]["initializer"],
                "rebuild": receipt["child_rebuild"].get("rebuild"),
                "post_transplant": receipt["child_rebuild"].get(
                    "post_transplant"),
                "fields_transplanted": transplant["stamped_field_count"],
                "fields_skipped": sorted(transplant["skipped"]),
                "stamped_cells_total": stamped_cells,
                # The route preparer's land-surface receipt: donor-fill
                # provenance counts, overlap-statics assertion, driver
                # rebuild timings.  None on routes whose preparer keeps
                # no such inventory (the idealized tree).
                "land_surface": land_surface,
            },
            "staging": receipt["staging"],
            # A MID-TREE move re-grounds the whole subtree, and each
            # descendant carries its own bitwise donor-alignment verdict.
            # Those verdicts are the evidence that the ground displacement
            # was carried down correctly, so they belong on the ledger and
            # not only in the exception that would have been raised had one
            # failed.  Absent on a leaf move, which keeps a leaf receipt
            # exactly the shape it has always been.
            **({"descendants": [
                {"grid_id": d["grid_id"],
                 "delta_parent_cells": d["delta_parent_cells"],
                 "shift_child_cells": d["plan"]["shift_child_cells"],
                 "overlap_cells": d["plan"]["overlap_cells"],
                 "donor_alignment_pass": bool(
                     (d.get("reground") or {})
                     .get("donor_alignment", {}).get("pass")),
                 "statics": (d.get("reground") or {}).get("statics")}
                for d in receipt["descendants"]]}
               if receipt.get("descendants") else {}),
            "donor_alignment_pass": bool(receipt["donor_alignment"]["pass"]),
            "parent_bitwise_unchanged": bool(
                receipt["parent_bitwise_unchanged"]),
            "rk_seeds_refreshed": len(receipt["rk_seeds_refreshed"]),
            "segment_id": receipt["segment"]["segment_id"],
            "generation": receipt["segment"]["generation"],
            "record_sha256": record_sha,
            "experiment_fingerprint": {
                "before": fingerprint_before,
                "after": model.experiment_fingerprint,
            },
        })

    def close_receipt(self, model) -> dict:
        """The run-end summary row (unconsumed itinerary is a finding).

        Exactly ONE such row exists in the receipts, however many times the
        executor closes; see :meth:`_record`.
        """
        summary = {
            "event": "summary",
            "moves_executed": int(self.moves_executed),
            "staging": self.staging,
        }
        if self.containment is not None:
            summary["containment_moves_executed"] = int(
                self.containment_moves_executed)
        if self.reach_clamp is not None:
            # The bound every move was held to, as numbers, so a ledger
            # whose clamped_by names reach_speed_m_s says what it was.
            summary["reach"] = self.reach_clamp.receipt()
        if self.track_writer is not None:
            # Consultations happen before integration. Without this final
            # diagnostic, a completed forecast's ATCF/CSV ends one cadence
            # early even though its final history frame exists.
            stream = getattr(self.track_writer, "stream", None)
            handle = getattr(stream, "handle", None)
            root = getattr(model, "root", None)
            if handle is not None and not handle.closed and root is not None:
                elapsed = float(root.clock.elapsed_seconds)
                last = getattr(stream, "_last_emit_t", None)
                synchronized = all(abs(float(n.clock.elapsed_seconds)-elapsed)<1.e-6
                    for n in model.walk_parent_first() if bool(getattr(n,"_started",True)))
                if synchronized and (last is None or elapsed>last+1.e-6):
                    self._track_opportunity(model,elapsed,locate=True,force=True)
            # How many records, how many emissions were skipped, and the
            # first few faults if any.  A track file that quietly wrote
            # nothing is a finding, and a summary that only counted
            # successes could not show it.
            summary["track"] = self.track_writer.close()
            summary["track_records"] = int(self.track_records)
        unconsumed = getattr(self.provider, "unconsumed", None)
        if callable(unconsumed):
            summary["unconsumed_moves"] = list(unconsumed())
        behind = getattr(self.provider, "behind_the_clock", None)
        if callable(behind):
            retired = list(behind())
            if retired:
                summary["moves_behind_the_clock"] = retired
        return self._record(model, summary, unique=True)


class RelocationRunnerCollection:
    """Independent relocation runners behind the executor's scalar seam.

    Each child owns a runner/provider/segment/cooldown state.  The collection
    merely orders them by grid id; it never shares tracker state.
    """

    is_collection = True

    def __init__(self, runners=()):
        self.runners = {int(r.config.grid_id): r for r in runners}
        # One ordered tree ledger for the order-sensitive restart move chain.
        # Per-runner JSON receipts remain independent and unchanged.
        self.receipts = []
        # Receipt rows of runners retired with their nest, per grid id, so a
        # re-armed slot's new runner writes them ahead of its own: the 6 h
        # storm-nest run (box D 2026-10-21) lost d03's whole first life of
        # moves when its second life's runner rewrote relocation_receipts.d03.json.
        self.retired_receipts: dict[int, list] = {}

    def record(self, grid_id, entry, *, unique=False):
        if unique:
            self.receipts[:] = [row for row in self.receipts
                               if not (row.get("follower_grid_id") == grid_id
                                       and row["event"] == entry["event"])]
        self.receipts.append({**entry, "follower_grid_id": grid_id})

    @property
    def target_grid_ids(self):
        return tuple(sorted(self.runners))

    @property
    def config(self):
        # Compatibility for code that only asks whether a named target exists.
        from types import SimpleNamespace
        gid = self.target_grid_ids[0] if self.runners else None
        return SimpleNamespace(grid_id=gid)

    def is_due(self, model, clocks) -> bool:
        return any(r.is_due(model, clocks) for r in self.runners.values()
                   if int(r.config.grid_id) in model.nodes_by_grid_id)

    def next_deadline_ticks(self, clocks):
        due = [r.next_deadline_ticks(clocks) for r in self.runners.values()]
        return min((tick for tick in due if tick is not None), default=None)

    def on_period_begin(self, model, clocks, *, period=None,
                        before_rebuild=None, after_rebuild=None):
        outcomes = []
        for gid in self.target_grid_ids:
            if gid not in model.nodes_by_grid_id:
                continue
            runner = self.runners[gid]
            row = runner.on_period_begin(
                model, clocks, period=period, before_rebuild=before_rebuild,
                after_rebuild=after_rebuild)
            if row is not None:
                outcomes.append(row)
        return None if not outcomes else {"event": "batch", "outcomes": outcomes}

    def close_receipt(self, model):
        return [self.runners[gid].close_receipt(model) for gid in self.target_grid_ids]

    def merge_from(self, other):
        """Add newly-live targets without resetting existing hysteresis."""
        if other is None:
            return self
        incoming = (other.runners if isinstance(other, RelocationRunnerCollection)
                    else {int(other.config.grid_id): other})
        for gid, rows in getattr(other, "retired_receipts", {}).items():
            self.retired_receipts.setdefault(int(gid), []).extend(rows)
        for gid, runner in incoming.items():
            if int(gid) not in self.runners and self.retired_receipts.get(int(gid)):
                runner.prior_receipts = list(self.retired_receipts[int(gid)])
            self.runners.setdefault(int(gid), runner)
        return self

    def drop_absent(self, model):
        # Retired runners are discarded; a later re-arm gets a fresh tracker
        # and therefore a fresh episode-local cooldown/window.
        for gid in list(self.runners):
            if gid not in model.nodes_by_grid_id:
                runner = self.runners.pop(gid, None)
                self.retired_receipts[gid] = [
                    *getattr(runner, "prior_receipts", ()), *runner.receipts]

    def adopt_placement(self, model, node, **placement):
        """Restore through the target's own initializer and physics preparer."""
        gid = int(node.cfg.grid_id)
        runner = self.runners.get(gid)
        if runner is None:
            owners = [r for r in self.runners.values()
                      if getattr(r.config, "containment", None) is not None
                      and int(r.config.containment.grid_id) == gid]
            if len(owners) != 1:
                raise RelocationRefusal(
                    f"d{gid:02d} checkpoint placement has no unique follower "
                    "or containment reconstruction owner")
            runner = owners[0]
        return runner.adopt_placement(model, node, **placement)

    def attach_writers(self, writers):
        for runner in self.runners.values():
            for preparer in (runner.on_child_built,
                             getattr(runner, "reground_descendant", None),
                             getattr(runner, "containment_preparer", None)):
                attach = getattr(preparer, "attach_writers", None)
                if callable(attach):
                    attach(writers)


__all__ = [
    "ManualMoveProvider", "RELOCATION_RUNNER_CONTRACT",
    "RUNNER_STATE_KEYS", "RUNNER_WRITER_KEYS", "RelocationRunner",
    "RelocationRunnerCollection",
    "STRIP_FILL_SOURCE",
]
