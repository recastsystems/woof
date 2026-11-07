"""CPU contracts for storm tracking (the moving nest's WHEN and WHERE).

The instrument rule applies: the tracker is tested against synthetic
features with KNOWN positions and KNOWN motion, in both directions -- a
follow test that only checks "it proposed something" passes just as
happily when the sign is flipped, so eastward and westward motion are
asserted to produce mirrored proposals, and every suppression has a
control that fires without it.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import woof.core.storm_tracking as st
from woof.core.storm_tracking import (FOLLOW_KEYS,
                                       PARENT_EDGE_KEEPOUT_CELLS,
                                       FollowConfig, NestFootprint,
                                       StormTracker, TrackerRefusal,
                                       build_follow_config,
                                       make_plan_provider,
                                       weighted_centroid)

PARENT_NY, PARENT_NX = 100, 120
RATIO = 3
CHILD_N = 45          # spans (45-1)/3 = 14.67 parent cells


class StubState:
    """LOUD stub of the one state surface the tracker reads: named scratch
    planes.  Anything else raises immediately."""

    def __init__(self, **planes):
        self._planes = {k: np.asarray(v) for k, v in planes.items()}

    def existing_scratch(self, slot):
        return self._planes.get(slot)

    def __getattr__(self, name):  # loud: no silent attribute fallbacks
        raise AttributeError(
            f"StubState exposes only existing_scratch(); {name!r} was read")


def _bump(cj, ci, amp, sigma=2.0, shape=(PARENT_NY, PARENT_NX)):
    jj, ii = np.mgrid[0:shape[0], 0:shape[1]]
    return amp * np.exp(-(((ii - ci) ** 2 + (jj - cj) ** 2)
                          / (2.0 * sigma ** 2)))


def _footprint(i=40, j=40):
    return NestFootprint(grid_id=2, i_parent_start=i, j_parent_start=j,
                         child_nx=CHILD_N, child_ny=CHILD_N,
                         parent_grid_ratio=RATIO)


def _follow(**over):
    base = dict(field="uh", threshold=50.0, fallback_threshold=40.0,
                search_margin_cells=10, min_shift_cells=2,
                max_shift_cells=6, cooldown_seconds=0.0)
    base.update(over)
    return FollowConfig(**base)


def _state(uh_at=None, uh_amp=200.0, refl_at=None, refl_amp=55.0):
    """Parent state with optional UH and reflectivity bumps at (cj, ci)."""
    uh = np.zeros((PARENT_NY, PARENT_NX))
    if uh_at is not None:
        uh = _bump(uh_at[0], uh_at[1], uh_amp)
    refl = np.full((PARENT_NY, PARENT_NX), -30.0)
    if refl_at is not None:
        refl = np.maximum(refl, _bump(refl_at[0], refl_at[1], refl_amp))
    # The RELOCATION consumer's own window (woof.core.uh_diag).
    return StubState(uh_follow_window=uh, refl_10cm=refl)


def _center(fp):
    ci, cj = fp.center_parent_ij
    return cj, ci


# ---------------------------------------------------------------------------
# Centroid arithmetic
# ---------------------------------------------------------------------------

def test_centroid_lands_on_a_known_bump():
    plane = _bump(50.0, 60.0, 200.0)
    box = (slice(0, PARENT_NY), slice(0, PARENT_NX))
    found = weighted_centroid(plane, 50.0, box)
    assert found is not None
    assert abs(found["ci"] - 60.0) < 0.05
    assert abs(found["cj"] - 50.0) < 0.05
    assert found["max_value"] == pytest.approx(200.0, rel=1e-6)
    assert found["cells"] > 0


def test_centroid_respects_the_search_box():
    """A second, stronger bump OUTSIDE the box must not drag the centroid:
    the box is the 'around the current nest footprint' contract."""
    plane = _bump(50.0, 60.0, 100.0) + _bump(10.0, 10.0, 500.0)
    box = (slice(40, 61), slice(50, 71))
    found = weighted_centroid(plane, 50.0, box)
    assert abs(found["ci"] - 60.0) < 0.2
    assert abs(found["cj"] - 50.0) < 0.2
    # Control: with the box open the intruder MUST win, or the box check
    # above proves nothing.
    wide = weighted_centroid(plane, 50.0,
                             (slice(0, PARENT_NY), slice(0, PARENT_NX)))
    assert wide["ci"] < 40.0 and wide["cj"] < 40.0


def test_centroid_none_below_threshold():
    plane = _bump(50.0, 60.0, 30.0)
    box = (slice(0, PARENT_NY), slice(0, PARENT_NX))
    assert weighted_centroid(plane, 50.0, box) is None


def test_centroid_ignores_nonfinite_cells():
    plane = _bump(50.0, 60.0, 200.0)
    plane[45:48, 55:58] = np.nan
    box = (slice(0, PARENT_NY), slice(0, PARENT_NX))
    found = weighted_centroid(plane, 50.0, box)
    assert found is not None and np.isfinite(found["ci"])


# ---------------------------------------------------------------------------
# The follow loop: a translating feature is followed within hysteresis
# ---------------------------------------------------------------------------

def _run_follow(vel_ij, cadences, *, cadence_s=600.0, cooldown=0.0,
                min_shift=2, max_shift=6, start_offset=(0.0, 0.0)):
    """Move a UH bump at vel_ij parent cells/cadence; apply every proposed
    shift to the footprint; return (tracker, footprint, storm, proposals).
    """
    tracker = StormTracker(_follow(min_shift_cells=min_shift,
                                   max_shift_cells=max_shift,
                                   cooldown_seconds=cooldown))
    fp = _footprint()
    cj, ci = _center(fp)
    storm = [cj + start_offset[0], ci + start_offset[1]]
    proposals = []
    for k in range(cadences):
        storm[0] += vel_ij[1]
        storm[1] += vel_ij[0]
        state = _state(uh_at=storm)
        shift = tracker.desired_shift(state, fp, t=k * cadence_s)
        if shift is not None:
            proposals.append(shift)
            fp = NestFootprint(
                grid_id=fp.grid_id,
                i_parent_start=fp.i_parent_start + shift[0],
                j_parent_start=fp.j_parent_start + shift[1],
                child_nx=fp.child_nx, child_ny=fp.child_ny,
                parent_grid_ratio=fp.parent_grid_ratio)
    return tracker, fp, storm, proposals


def test_tracker_follows_a_translating_storm_within_the_dead_band():
    tracker, fp, storm, proposals = _run_follow((0.8, 0.5), 20)
    assert len(proposals) >= 3            # it moved, repeatedly
    cj, ci = _center(fp)
    # Within hysteresis: the nest center may lag by up to the dead-band
    # plus rounding, never more.
    assert abs(storm[1] - ci) <= 2 + 0.5
    assert abs(storm[0] - cj) <= 2 + 0.5
    assert all(max(abs(di), abs(dj)) <= 6 for di, dj in proposals)
    # Every proposal chased the storm: strictly positive components only
    # (motion is +i/+j), so a sign flip in the shift arithmetic fails here.
    assert all(di >= 0 and dj >= 0 for di, dj in proposals)
    assert sum(di for di, _ in proposals) > 0


def test_mirrored_motion_gives_mirrored_proposals():
    """The both-directions control for the follow loop itself."""
    _, fp_e, _, prop_e = _run_follow((0.8, 0.0), 12)
    _, fp_w, _, prop_w = _run_follow((-0.8, 0.0), 12)
    assert sum(di for di, _ in prop_e) == -sum(di for di, _ in prop_w)
    assert fp_e.i_parent_start - 40 == -(fp_w.i_parent_start - 40)
    assert fp_e.i_parent_start > 40 > fp_w.i_parent_start
    assert fp_e.j_parent_start == fp_w.j_parent_start == 40


def test_stationary_jitter_is_dead_banded_to_zero_moves():
    """A storm wobbling inside the dead-band must produce NO proposals --
    the anti-chatter contract."""
    tracker = StormTracker(_follow(min_shift_cells=2))
    fp = _footprint()
    cj, ci = _center(fp)
    rng = np.random.default_rng(20110427)
    for k in range(12):
        wobble = rng.uniform(-1.0, 1.0, size=2)
        state = _state(uh_at=(cj + wobble[0], ci + wobble[1]))
        assert tracker.desired_shift(state, fp, t=k * 600.0) is None
    kinds = {r["decision"] for r in tracker.receipts}
    assert "suppressed:dead-band" in kinds
    assert "proposed" not in kinds


def test_dead_band_control_a_real_displacement_does_propose():
    tracker = StormTracker(_follow(min_shift_cells=2))
    fp = _footprint()
    cj, ci = _center(fp)
    state = _state(uh_at=(cj + 4.0, ci + 4.0))
    assert tracker.desired_shift(state, fp, t=0.0) == (4, 4)


def test_min_shift_is_a_chebyshev_dead_band():
    tracker = StormTracker(_follow(min_shift_cells=3))
    fp = _footprint()
    cj, ci = _center(fp)
    # (2, 2) rounds under the band -> suppressed; (3, 0) meets it.
    assert tracker.desired_shift(
        _state(uh_at=(cj + 2.0, ci + 2.0)), fp, t=0.0) is None
    assert tracker.desired_shift(
        _state(uh_at=(cj, ci + 3.0)), fp, t=600.0) == (3, 0)


def test_max_shift_clamps_a_jump_and_records_it():
    tracker = StormTracker(_follow(max_shift_cells=6))
    fp = _footprint()
    cj, ci = _center(fp)
    shift = tracker.desired_shift(
        _state(uh_at=(cj - 3.0, ci + 20.0)), fp, t=0.0)
    assert shift == (6, -3)
    last = tracker.receipts[-1]
    assert last["decision"] == "proposed"
    assert last["clamped"] is True
    # Control: an in-bounds proposal is not marked clamped.
    tracker2 = StormTracker(_follow(max_shift_cells=6))
    tracker2.desired_shift(_state(uh_at=(cj, ci + 4.0)), fp, t=0.0)
    assert tracker2.receipts[-1]["clamped"] is False


def test_cooldown_suppresses_then_releases():
    tracker = StormTracker(_follow(cooldown_seconds=1800.0))
    fp = _footprint()
    cj, ci = _center(fp)
    state = _state(uh_at=(cj, ci + 5.0))
    assert tracker.desired_shift(state, fp, t=0.0) == (5, 0)
    assert tracker.desired_shift(state, fp, t=600.0) is None
    assert tracker.receipts[-1]["decision"] == "suppressed:cooldown"
    assert tracker.receipts[-1]["cooldown_remaining_s"] == 1200.0
    assert tracker.desired_shift(state, fp, t=1800.0) == (5, 0)


def test_cooldown_counts_from_executed_moves_once_the_runner_notifies():
    """The leg-2 hook: notify_move_executed re-anchors the cooldown.

    Discriminating sequence -- a proposal the runner did NOT execute must
    not burn the cooldown: under the old proposal-burn semantics the last
    call here would be suppressed (600 s after a proposal), and under the
    hook semantics it proposes (2400 s after the last EXECUTED move).
    """
    tracker = StormTracker(_follow(cooldown_seconds=1800.0))
    fp = _footprint()
    cj, ci = _center(fp)
    state = _state(uh_at=(cj, ci + 5.0))
    assert tracker.desired_shift(state, fp, t=0.0) == (5, 0)
    tracker.notify_move_executed(0.0, (5, 0))     # runner executed it
    assert tracker.receipts[-1]["decision"] == "move-executed"
    assert tracker.receipts[-1]["executed_shift_parent_cells"] == [5, 0]
    # Inside the cooldown of the executed move: held, anchored on it.
    assert tracker.desired_shift(state, fp, t=600.0) is None
    assert tracker.receipts[-1]["cooldown_anchor"] == "executed-move"
    # Cooldown over: proposes -- but the runner does NOT execute this one.
    assert tracker.desired_shift(state, fp, t=1800.0) == (5, 0)
    # 600 s later: the unexecuted proposal burned nothing, so the tracker
    # may re-propose (2400 - 0 >= 1800 against the executed-move anchor).
    assert tracker.desired_shift(state, fp, t=2400.0) == (5, 0)


def test_without_the_notify_hook_proposals_still_burn_the_cooldown():
    """The fallback control for the hook: an old runner that never
    notifies keeps the original proposal-burn hysteresis."""
    tracker = StormTracker(_follow(cooldown_seconds=1800.0))
    fp = _footprint()
    cj, ci = _center(fp)
    state = _state(uh_at=(cj, ci + 5.0))
    assert tracker.desired_shift(state, fp, t=0.0) == (5, 0)
    assert tracker.desired_shift(state, fp, t=600.0) is None
    assert tracker.receipts[-1]["cooldown_anchor"] == "proposal"


def test_cooldown_suppression_survives_a_state_round_trip():
    """A tracker that just moved must not re-propose at the first cadence
    boundary of a resumed run: position is an argument every call, but the
    two cooldown anchors are the tracker's own and nothing else holds them.
    """
    import json

    fp = _footprint()
    cj, ci = _center(fp)
    state = _state(uh_at=(cj, ci + 5.0))

    tracker = StormTracker(_follow(cooldown_seconds=1800.0))
    assert tracker.desired_shift(state, fp, t=0.0) == (5, 0)
    tracker.notify_move_executed(0.0, (5, 0))
    block = json.loads(json.dumps(tracker.state_json(), allow_nan=False))
    assert block == {"last_proposal_t": 0.0, "last_move_t": 0.0}

    resumed = StormTracker(_follow(cooldown_seconds=1800.0))
    resumed.restore_state(block)
    assert resumed.state_json() == block
    assert resumed.desired_shift(state, fp, t=600.0) is None
    assert resumed.receipts[-1]["decision"] == "suppressed:cooldown"
    assert resumed.receipts[-1]["cooldown_anchor"] == "executed-move"
    assert resumed.receipts[-1]["cooldown_remaining_s"] == 1200.0
    # It releases at the same instant an unbroken tracker would.
    assert resumed.desired_shift(state, fp, t=1800.0) == (5, 0)

    # The control that makes the assertion non-vacuous: a tracker that
    # forgot the anchors moves the nest 1200 s early.
    cold = StormTracker(_follow(cooldown_seconds=1800.0))
    assert cold.desired_shift(state, fp, t=600.0) == (5, 0)


def test_the_two_cooldown_anchors_round_trip_apart():
    """The proposal anchor and the executed-move anchor mean different
    things; collapsing them on restore re-anchors the whole hysteresis."""
    fp = _footprint()
    cj, ci = _center(fp)
    state = _state(uh_at=(cj, ci + 5.0))

    tracker = StormTracker(_follow(cooldown_seconds=1800.0))
    assert tracker.desired_shift(state, fp, t=0.0) == (5, 0)  # not executed
    block = tracker.state_json()
    assert block == {"last_proposal_t": 0.0, "last_move_t": None}

    resumed = StormTracker(_follow(cooldown_seconds=1800.0))
    resumed.restore_state(block)
    assert resumed.desired_shift(state, fp, t=600.0) is None
    assert resumed.receipts[-1]["cooldown_anchor"] == "proposal"


def test_tracker_state_refuses_what_it_cannot_read():
    tracker = StormTracker(_follow())
    with pytest.raises(TrackerRefusal, match="last_move_at"):
        tracker.restore_state({"last_proposal_t": None,
                               "last_move_at": None})
    with pytest.raises(TrackerRefusal, match="last_proposal_t"):
        tracker.restore_state({"last_move_t": None})
    with pytest.raises(TrackerRefusal, match="last_move_t"):
        tracker.restore_state({"last_proposal_t": None,
                               "last_move_t": float("inf")})


def test_suppressed_receipts_still_carry_the_centroid_evidence():
    tracker = StormTracker(_follow(cooldown_seconds=3600.0))
    fp = _footprint()
    cj, ci = _center(fp)
    tracker.desired_shift(_state(uh_at=(cj, ci + 5.0)), fp, t=0.0)
    tracker.desired_shift(_state(uh_at=(cj, ci + 6.0)), fp, t=600.0)
    held = tracker.receipts[-1]
    assert held["decision"] == "suppressed:cooldown"
    assert held["cells_above_threshold"] > 0
    assert abs(held["centroid_parent_ij"][0] - (ci + 6.0)) < 0.2


# ---------------------------------------------------------------------------
# The UH -> reflectivity handoff
# ---------------------------------------------------------------------------

def test_handoff_to_reflectivity_before_the_storm_rotates():
    fp = _footprint()
    cj, ci = _center(fp)
    tracker = StormTracker(_follow())
    # UH quiet (amp 30 < threshold 50), echo displaced +4 i.
    state = _state(uh_at=(cj, ci), uh_amp=30.0, refl_at=(cj, ci + 4.0))
    assert tracker.desired_shift(state, fp, t=0.0) == (4, 0)
    last = tracker.receipts[-1]
    assert last["field_requested"] == "uh"
    assert last["field_used"] == "reflectivity"
    assert last["threshold_used"] == 40.0


def test_rotation_outvotes_the_echo_once_it_exists():
    """Control for the handoff: with UH above threshold the tracker must
    follow the mesocyclone, not the (differently placed) echo centroid."""
    fp = _footprint()
    cj, ci = _center(fp)
    tracker = StormTracker(_follow())
    state = _state(uh_at=(cj, ci + 4.0), uh_amp=200.0,
                   refl_at=(cj + 6.0, ci - 6.0))
    assert tracker.desired_shift(state, fp, t=0.0) == (4, 0)
    assert tracker.receipts[-1]["field_used"] == "uh"


def test_no_signal_anywhere_holds_with_a_receipt():
    fp = _footprint()
    tracker = StormTracker(_follow())
    assert tracker.desired_shift(_state(), fp, t=0.0) is None
    assert tracker.receipts[-1]["decision"] == "no-signal"
    assert tracker.receipts[-1]["field_used"] == "reflectivity"


def test_reflectivity_primary_never_reads_uh():
    fp = _footprint()
    cj, ci = _center(fp)
    tracker = StormTracker(_follow(field="reflectivity", threshold=40.0,
                                   fallback_threshold=None))
    state = StubState(refl_10cm=_bump(cj, ci + 3.0, 55.0))  # no UH plane
    assert tracker.desired_shift(state, fp, t=0.0) == (3, 0)


def test_composite_is_the_column_max_of_a_3d_reflectivity():
    fp = _footprint()
    cj, ci = _center(fp)
    volume = np.stack([np.full((PARENT_NY, PARENT_NX), -30.0),
                       _bump(cj, ci + 3.0, 55.0),
                       np.full((PARENT_NY, PARENT_NX), -30.0)])
    tracker = StormTracker(_follow(field="reflectivity", threshold=40.0,
                                   fallback_threshold=None))
    assert tracker.desired_shift(
        StubState(refl_10cm=volume), fp, t=0.0) == (3, 0)


def test_missing_uh_plane_refuses_loudly():
    tracker = StormTracker(_follow())
    with pytest.raises(TrackerRefusal, match="nwp_diagnostics"):
        tracker.desired_shift(
            StubState(refl_10cm=np.zeros((PARENT_NY, PARENT_NX))),
            _footprint(), t=0.0)


def test_missing_reflectivity_fallback_refuses_loudly():
    tracker = StormTracker(_follow())
    with pytest.raises(TrackerRefusal, match="refl_10cm"):
        tracker.desired_shift(
            StubState(uh_follow_window=np.zeros((PARENT_NY, PARENT_NX))),
            _footprint(), t=0.0)


# ---------------------------------------------------------------------------
# Parent-edge keepout
# ---------------------------------------------------------------------------

def test_proposals_never_push_the_footprint_into_the_boundary_zone():
    fp = _footprint(i=99, j=40)   # east edge: i_lo 98 + span 14.67 -> 112.7
    cj, ci = _center(fp)
    tracker = StormTracker(_follow(max_shift_cells=10))
    shift = tracker.desired_shift(
        _state(uh_at=(cj, ci + 8.0)), fp, t=0.0)
    di, dj = shift
    new_hi = (fp.i_parent_start - 1 + di) + fp.span_parent_i
    assert new_hi <= PARENT_NX - 1 - PARENT_EDGE_KEEPOUT_CELLS
    assert tracker.receipts[-1]["clipped_to_parent"] is True
    # Control: the same storm with room to move is not clipped.
    fp_mid = _footprint()
    tracker2 = StormTracker(_follow(max_shift_cells=10))
    cj2, ci2 = _center(fp_mid)
    tracker2.desired_shift(_state(uh_at=(cj2, ci2 + 8.0)), fp_mid, t=0.0)
    assert tracker2.receipts[-1]["clipped_to_parent"] is False


def test_a_fully_clipped_proposal_becomes_a_hold_not_a_null_move():
    fp = _footprint(i=99, j=40)
    cj, ci = _center(fp)
    tracker = StormTracker(_follow())
    # Footprint already hugging the keepout; storm just over the dead-band
    # eastward; the clip leaves nothing -> hold with its own receipt.
    fp_edge = NestFootprint(grid_id=2, i_parent_start=101, j_parent_start=40,
                            child_nx=CHILD_N, child_ny=CHILD_N,
                            parent_grid_ratio=RATIO)
    cj, ci = _center(fp_edge)
    out = tracker.desired_shift(
        _state(uh_at=(cj, ci + 2.0)), fp_edge, t=0.0)
    assert out is None
    assert tracker.receipts[-1]["decision"] == "suppressed:at-parent-edge"


# ---------------------------------------------------------------------------
# Config governance
# ---------------------------------------------------------------------------

def _follow_table(**over):
    base = dict(field="uh", threshold=50.0, fallback_threshold=40.0,
                search_margin_cells=10, min_shift_cells=2,
                max_shift_cells=6, cooldown_seconds=900.0)
    base.update(over)
    return {k: v for k, v in base.items() if v is not None}


def _relocation_raw(follow=None, enabled=True, cadence_seconds=900.0):
    table = {"enabled": enabled, "grid_id": 2, "max_move_parent_cells": 6}
    if follow is not None:
        table["follow"] = follow
        # A tracker needs a cadence the refl stash can serve (issue
        # #111); the fixture's watched domain writes history every 900 s.
        if cadence_seconds is not None:
            table["cadence_seconds"] = cadence_seconds
    if not enabled:
        table = {"enabled": False, "follow": follow}
    return {"relocation": table}


def _build(raw):
    from woof.experiment import _build_relocation

    # Leg 2 gave _build_relocation the domain list (for the root's exact
    # dt, which gates cadence/at_seconds alignment) and the run length.
    # These stand-ins carry exactly what that validation reads.
    domains = [
        SimpleNamespace(grid_id=1, parent_id=None, time_step=60,
                        time_step_fract_num=0, time_step_fract_den=1,
                        history_interval_s=900.0),
        SimpleNamespace(grid_id=2, parent_id=1, time_step=None,
                        history_interval_s=900.0),
    ]
    return _build_relocation(raw, "test.toml", domains, 43200.0)


def test_follow_block_loads_and_echoes_every_value():
    cfg = _build(_relocation_raw(follow=_follow_table()))
    follow = cfg.follow
    assert follow is not None
    echoed = follow.to_json()
    for key, value in _follow_table().items():
        assert echoed[key] == value


def test_follow_unknown_key_is_refused_by_name_with_a_suggestion():
    with pytest.raises(ValueError, match="'thresold'.*did you mean"):
        _build(_relocation_raw(follow=_follow_table(thresold=1.0)))


def test_follow_missing_key_is_refused_by_name():
    table = _follow_table()
    del table["cooldown_seconds"]
    with pytest.raises(ValueError, match="cooldown_seconds"):
        _build(_relocation_raw(follow=table))


def test_follow_on_disabled_relocation_is_refused():
    with pytest.raises(ValueError, match="enabled"):
        _build(_relocation_raw(follow=_follow_table(), enabled=False))


def test_follow_must_be_a_table():
    with pytest.raises(ValueError, match="TABLE"):
        _build(_relocation_raw(follow="uh"))


def test_follow_field_must_be_a_tracked_field():
    with pytest.raises(ValueError, match="uh.*reflectivity"):
        _build(_relocation_raw(follow=_follow_table(field="vorticity")))


def test_uh_requires_a_fallback_threshold():
    with pytest.raises(ValueError, match="fallback_threshold"):
        _build(_relocation_raw(follow=_follow_table(
            fallback_threshold=None)))


def test_reflectivity_refuses_a_fallback_threshold():
    with pytest.raises(ValueError, match="refuses fallback_threshold"):
        _build(_relocation_raw(follow=_follow_table(
            field="reflectivity", threshold=40.0)))


def test_degenerate_hysteresis_is_refused():
    with pytest.raises(ValueError, match="min_shift_cells"):
        FollowConfig(**dict(_follow_table(min_shift_cells=0)))
    with pytest.raises(ValueError, match="below"):
        FollowConfig(**dict(_follow_table(min_shift_cells=4,
                                          max_shift_cells=3)))
    with pytest.raises(ValueError, match="cooldown_seconds"):
        FollowConfig(**dict(_follow_table(cooldown_seconds=-1.0)))


def test_booleans_are_not_numbers():
    with pytest.raises(ValueError, match="min_shift_cells"):
        _build(_relocation_raw(follow=_follow_table(min_shift_cells=True)))


def test_relocation_without_follow_still_loads_with_follow_none():
    cfg = _build(_relocation_raw())
    assert cfg.enabled and cfg.follow is None


# ---------------------------------------------------------------------------
# The plan-provider hookup (the leg-2 seam)
# ---------------------------------------------------------------------------

def test_make_plan_provider_returns_none_without_a_follow_block():
    from types import SimpleNamespace
    from woof.experiment import RelocationConfig

    exp = SimpleNamespace(relocation=RelocationConfig())
    assert make_plan_provider(exp) is None
    exp = SimpleNamespace(relocation=RelocationConfig(
        enabled=True, grid_id=2))
    assert make_plan_provider(exp) is None


def test_make_plan_provider_builds_a_callable_tracker():
    from types import SimpleNamespace
    from woof.experiment import RelocationConfig

    exp = SimpleNamespace(relocation=RelocationConfig(
        enabled=True, grid_id=2, follow=_follow()))
    provider = make_plan_provider(exp)
    assert isinstance(provider, StormTracker)
    fp = _footprint()
    cj, ci = _center(fp)
    # The seam contract, called AS the callable the runner receives.
    assert provider(_state(uh_at=(cj, ci + 4.0)), fp, t=0.0) == (4, 0)


def test_footprint_coerces_a_domain_config_shape():
    from types import SimpleNamespace

    dc = SimpleNamespace(grid_id=2, i_parent_start=40, j_parent_start=40,
                         parent_grid_ratio=RATIO,
                         run=SimpleNamespace(nx=CHILD_N, ny=CHILD_N))
    assert NestFootprint.coerce(dc) == _footprint()


def test_tracker_receipts_open_with_the_echoed_config():
    tracker = StormTracker(_follow())
    first = tracker.receipts[0]
    assert first["decision"] == "configured"
    assert first["config"]["threshold"] == 50.0
    # Optional pressure and fallback controls are echoed only when used.
    assert set(first["config"]) >= (
        FOLLOW_KEYS - st.ATTRIBUTE_KEYS - {"fallback_threshold", "level_hpa", "refine_grid_id",
                                         "pressure_center", "fallback_level_hpa", "fallback_deficit_m"})
    assert tracker.drain_receipts()          # hands them over ...
    assert tracker.receipts == []            # ... and clears the ledger


# ---------------------------------------------------------------------------
# The isobaric (mid-level) tracker: WRF's `track_level`, discretely
# ---------------------------------------------------------------------------

def _isobaric_state(*, ny=120, nx=140, nz=40, ic=45.0, jc=70.0,
                    depth_pa=4000.0, scale_height=8000.0, p_top=5000.0,
                    eta_levels=True):
    """An isothermal hydrostatic column on an eta grid, with a Gaussian
    surface low at (ic, jc).

    Built the way a WRF eta core lays its column out, so the layer-mean
    pairing is NOT exact by construction: interface pressures are linear
    in eta, ``p_w = p_top + eta (psfc - p_top)``; each mass level sits at
    the ARITHMETIC mean of its two interfaces' pressures; the interface
    geopotential is the isothermal ``g H ln(psfc / p_w)``.  The height of
    the ``P`` surface is then exactly ``H ln(psfc / P)``, and the surface
    low puts a height MINIMUM on every isobaric surface at the cell the
    low is centred on.  Reading the surface off layer-mean heights at the
    mass-level pressures lands about 1 m high at 850 hPa on this grid,
    which the analytic check below catches.

    ``eta_levels=False`` leaves ``znu``/``znw`` off the state.  With them
    the surfaces are read by the Rust rw-isobaric library, so a box that
    has not built it skips with the remedy rather than failing.
    """
    from woof.core.storm_tracking import GRAVITY_M_S2
    if eta_levels:
        from woof import isobaric_bridge
        reason = isobaric_bridge.unavailable_reason()
        if reason is not None:
            pytest.skip(f"the rw-isobaric library is not built here ({reason}); "
                        "cd tools/rustwx && cargo build --release -p rw-isobaric "
                        "--offline")

    znw = 1.0 - (np.arange(nz + 1) / nz) ** 1.3
    znu = 0.5 * (znw[:-1] + znw[1:])
    jj, ii = np.mgrid[0:ny, 0:nx]
    psfc = 101000.0 - depth_pa * np.exp(
        -(((ii - ic) / 8.0) ** 2 + ((jj - jc) / 8.0) ** 2))
    p_w = p_top + znw[:, None, None] * (psfc[None] - p_top)
    pressure = 0.5 * (p_w[:-1] + p_w[1:])
    phi = GRAVITY_M_S2 * scale_height * np.log(psfc[None] / p_w)
    state = SimpleNamespace(p=pressure, php=phi, phb=np.zeros_like(phi),
                            qv=np.zeros_like(pressure))
    if eta_levels:
        state.znu, state.znw = znu, znw
    state.total_theta = lambda: np.full_like(pressure, 300.0)
    return state, psfc


def test_isobaric_height_matches_the_analytic_surface():
    """The reduction is the surface, not an approximation of it."""
    state, psfc = _isobaric_state()
    plane = st.level_height_m_from_state(state, 850.0)
    analytic = 8000.0 * np.log(psfc / 85000.0)
    # The minimum sits on the same cell; the smoother broadens the core,
    # so agreement is checked where the field is not curved.
    assert np.unravel_index(int(plane.argmin()), plane.shape) == \
        np.unravel_index(int(analytic.argmin()), analytic.shape)
    far = np.ones_like(plane, dtype=bool)
    far[40:100, 15:75] = False
    assert np.abs(plane[far] - analytic[far]).max() < 0.05


def test_isobaric_height_is_read_between_interfaces_not_off_layer_means():
    """THE 2.8.6 DEFECT, pinned.  Without the state's eta levels the
    tracker falls back to layer-mean heights at the mass-level pressures,
    and on an eta column that reads high: the fallback is kept only for a
    height-coordinate column, where it is exact."""
    state, psfc = _isobaric_state()
    bare, _ = _isobaric_state(eta_levels=False)
    analytic = 8000.0 * np.log(psfc / 50000.0)
    far = np.ones_like(analytic, dtype=bool)
    far[40:100, 15:75] = False
    read = st.level_height_m_from_state(state, 500.0)
    paired = st.level_height_m_from_state(bare, 500.0)
    assert np.abs(read[far] - analytic[far]).max() < 0.05
    offset = float(np.mean(paired[far] - analytic[far]))
    assert offset > 1.0, f"the layer-mean pairing reads {offset:.2f} m high"


def test_isobaric_threshold_is_relative_so_the_centre_is_threshold_free():
    """THE POINT OF THE RELATIVE THRESHOLD.

    An absolute ceiling has to be re-tuned whenever the environment
    moves; anchoring on the search box's own minimum means the centroid
    is the same cell whether the user asked for a 10 m slice of the core
    or a 60 m one.  That insensitivity is what makes the choice safe.
    """
    state, _ = _isobaric_state(ic=45.0, jc=70.0)
    plane = st.level_height_m_from_state(state, 850.0)
    fp = st.NestFootprint(grid_id=2, i_parent_start=20, j_parent_start=45,
                          child_nx=90, child_ny=90, parent_grid_ratio=3)
    box = fp.search_box(plane.shape, 20)
    centres = []
    for threshold in (10.0, 30.0, 60.0):
        found = st.locate_signal(plane, "pressure", threshold, box,
                                 relative_to_minimum=True)
        assert found is not None
        centres.append((round(found["ci"], 6), round(found["cj"], 6)))
        # and the extremum comes back in metres, un-negated
        assert 900.0 < st.signal_extremum(found, "pressure") < 1300.0
    assert centres == [(45.0, 70.0)] * 3


def test_isobaric_surface_below_ground_does_not_vote():
    """A column with no such surface must not be extrapolated onto one.

    Terrain is the reason this matters: an island in the search box would
    otherwise contribute a fabricated height, and a fabricated height can
    outrank the real centre in the very field steering the nest.
    """
    state, _ = _isobaric_state()
    # A "mountain" block: surface pressure there drops to ~909 hPa, so
    # the 950 hPa surface is underground over the block and nowhere
    # else -- the vortex core bottoms out at 970 hPa, still above it.
    state.p = state.p.copy()
    state.p[:, 10:30, 10:30] *= 0.90
    plane = st.level_height_m_from_state(state, 950.0)
    assert not np.isfinite(plane[10:30, 10:30]).any()
    assert np.isfinite(plane[60:100, 20:120]).all()

    # NaN must not have leaked out of the block through the smoother.
    assert np.isfinite(plane).sum() == plane.size - 400

    # An excluded cell cannot vote, whatever its fabricated value would
    # have been: the centroid is unchanged by the block's presence.
    clean, _ = _isobaric_state()
    fp = st.NestFootprint(grid_id=2, i_parent_start=2, j_parent_start=2,
                          child_nx=300, child_ny=300, parent_grid_ratio=3)
    box = fp.search_box(plane.shape, 0)
    with_block = st.locate_signal(plane, "pressure", 30.0, box,
                                  relative_to_minimum=True)
    without = st.locate_signal(
        st.level_height_m_from_state(clean, 950.0), "pressure", 30.0, box,
        relative_to_minimum=True)
    assert (round(with_block["ci"], 6), round(with_block["cj"], 6)) ==            (round(without["ci"], 6), round(without["cj"], 6))


def test_isobaric_level_outside_the_column_everywhere_refuses():
    """Not a hold: a level no column reaches is a configuration error."""
    state, _ = _isobaric_state()
    state.p = state.p * 0.5           # surface now near 505 hPa
    with pytest.raises(st.TrackerRefusal, match="outside the parent"):
        st.level_height_m_from_state(state, 1000.0)


def test_level_hpa_refuses_every_units_error_at_admission():
    base = {"field": "pressure", "threshold": 30.0,
            "search_margin_cells": 40, "min_shift_cells": 1,
            "max_shift_cells": 6, "cooldown_seconds": 1800.0}
    st.build_follow_config({**base, "level_hpa": 850}, "ok")     # admits
    with pytest.raises(ValueError, match="units error"):
        st.build_follow_config({**base, "level_hpa": 85000}, "x")
    with pytest.raises(ValueError, match="units error"):
        st.build_follow_config({**base, "level_hpa": 12}, "x")
    with pytest.raises(ValueError, match="RELATIVE"):
        st.build_follow_config(
            {**base, "level_hpa": 850, "threshold": 1000.0}, "x")
    with pytest.raises(ValueError, match="refuses level_hpa"):
        st.build_follow_config(
            {**base, "field": "uh", "level_hpa": 850,
             "fallback_threshold": 35.0}, "x")


def test_omitting_level_hpa_is_the_850_hpa_surface():
    """The default surface, and the receipt says it was defaulted."""
    base = {"field": "pressure", "threshold": 30.0,
            "search_margin_cells": 40, "min_shift_cells": 1,
            "max_shift_cells": 6, "cooldown_seconds": 1800.0}
    cfg = st.build_follow_config(base, "ok")
    assert cfg.level_hpa == (st.DEFAULT_LEVEL_HPA,) == (850.0,)
    record = cfg.to_json()
    assert record["level_hpa"] == [850.0]
    assert record["level_hpa_source"] == "default"
    assert record["threshold_units"] == "m above search-box minimum"
    # A surface NAMED is the same surface, and says so.
    named = st.build_follow_config({**base, "level_hpa": 850.0}, "ok")
    assert named.level_hpa == cfg.level_hpa
    assert named.to_json()["level_hpa_source"] == "config"


def test_level_hpa_zero_is_the_mslp_tracker_unchanged():
    """The sea-level tracker is still there, spelled as a choice.

    It is the one form whose threshold is an absolute hPa ceiling, and
    every guard that band had still fires.
    """
    base = {"field": "pressure", "threshold": 1004.0, "level_hpa": 0,
            "search_margin_cells": 40, "min_shift_cells": 1,
            "max_shift_cells": 6, "cooldown_seconds": 1800.0}
    cfg = st.build_follow_config(base, "ok")
    # Carried internally as None -- the ONE shape every consumer reads.
    assert cfg.level_hpa is None
    assert st.levels_of(cfg) == ()
    record = cfg.to_json()
    assert record["threshold_units"] == "hPa (absolute MSLP ceiling)"
    assert record["level_hpa"] == [0.0]
    with pytest.raises(ValueError, match="1100"):
        st.build_follow_config({**base, "threshold": 100400.0}, "x")


def test_the_default_refuses_a_pre_default_mslp_config_by_name():
    """An hPa ceiling with no level_hpa used to mean sea level and now
    means 850 hPa, where the threshold is metres.  It must refuse at
    load -- silently reinterpreting the number would steer the nest off
    a different field -- and the message must carry both ways out."""
    base = {"field": "pressure", "threshold": 1005.0,
            "search_margin_cells": 40, "min_shift_cells": 1,
            "max_shift_cells": 6, "cooldown_seconds": 1800.0}
    with pytest.raises(ValueError) as err:
        st.build_follow_config(base, "old")
    assert "level_hpa = 0" in str(err.value)
    assert "850" in str(err.value)
    # ...and both ways out are ways out.
    assert st.build_follow_config(
        {**base, "level_hpa": 0}, "ok").level_hpa is None
    assert st.build_follow_config(
        {**base, "threshold": 30.0}, "ok").level_hpa == (850.0,)


def test_sea_level_cannot_be_a_term_in_a_deep_layer_mean():
    """0 is not an isobaric surface, so it cannot be averaged with any."""
    base = {"field": "pressure", "threshold": 30.0,
            "search_margin_cells": 40, "min_shift_cells": 1,
            "max_shift_cells": 6, "cooldown_seconds": 1800.0}
    with pytest.raises(ValueError, match="mixes 0"):
        st.build_follow_config({**base, "level_hpa": [0, 850.0]}, "x")


def test_a_non_pressure_field_still_refuses_an_explicit_sea_level():
    """level_hpa = 0 normalises to None, and the refusal must be decided
    BEFORE that -- or `field = "uh", level_hpa = 0` would normalise
    itself out of the guard and be quietly accepted."""
    base = {"field": "uh", "threshold": 25.0, "fallback_threshold": 40.0,
            "search_margin_cells": 40, "min_shift_cells": 1,
            "max_shift_cells": 6, "cooldown_seconds": 1800.0}
    assert st.build_follow_config(base, "ok").level_hpa is None
    with pytest.raises(ValueError, match="refuses level_hpa"):
        st.build_follow_config({**base, "level_hpa": 0}, "x")


# ---------------------------------------------------------------------------
# Report-only surfaces: computed, reported, never steering
# ---------------------------------------------------------------------------

def test_report_only_surfaces_are_excluded_from_the_mean():
    """centre_over_levels means the STEERING set and carries all of them.

    Built so the two answers cannot coincide by accident: the steering
    surface's bowl is at one place and both report surfaces' bowls are
    far away, so a mean that wrongly included them would land visibly
    between.
    """
    from types import SimpleNamespace

    from woof.core.storm_tracking import centre_over_levels

    ny = nx = 80
    box = (slice(0, ny), slice(0, nx))

    def bowl(cj, ci):
        j, i = np.mgrid[0:ny, 0:nx]
        r2 = (j - cj) ** 2 + (i - ci) ** 2
        return (1500.0 - 60.0 * np.exp(-r2 / 18.0)).astype(np.float64)

    planes = [(850.0, bowl(20.0, 20.0)),      # steers
              (700.0, bowl(60.0, 60.0)),      # reports only
              (500.0, bowl(70.0, 10.0))]      # reports only
    cfg = SimpleNamespace(field="pressure", threshold=30.0,
                          level_hpa=(850.0,),
                          report_level_hpa=(700.0, 500.0))
    found, level_fixes, declined = centre_over_levels(planes, cfg, box, None)

    assert found is not None
    # The steering surface's own centre, not a three-way mean.
    assert abs(found["ci"] - 20.0) < 1.0 and abs(found["cj"] - 20.0) < 1.0
    assert "levels_averaged" not in found      # one surface voted
    assert found["levels_reported"] == 3
    # ...and all three still reach the file.
    assert sorted(f.level_hpa for f in level_fixes) == [500.0, 700.0, 850.0]
    assert not declined


def test_a_hold_when_only_report_surfaces_answer():
    """No steering surface answered, so there is no centre to follow --
    even though report-only surfaces found one.  Inventing a centre from
    surfaces the configuration says not to steer on would move the nest
    on a signal it was told to ignore."""
    from types import SimpleNamespace

    from woof.core.storm_tracking import centre_over_levels

    ny = nx = 60
    box = (slice(0, ny), slice(0, nx))
    j, i = np.mgrid[0:ny, 0:nx]
    flat = np.full((ny, nx), 1500.0)
    bowl = (1500.0 - 60.0 * np.exp(
        -((j - 30.0) ** 2 + (i - 30.0) ** 2) / 18.0)).astype(np.float64)
    cfg = SimpleNamespace(field="pressure", threshold=30.0,
                          level_hpa=(850.0,), report_level_hpa=(700.0,))
    found, level_fixes, declined = centre_over_levels(
        [(850.0, flat), (700.0, bowl)], cfg, box, None)

    assert found is None                       # a hold, correctly
    assert [f.level_hpa for f in level_fixes] == [700.0]   # still auditable
    assert declined and declined[0]["level_hpa"] == 850.0
