"""Two-way feedback across a MIXED microphysics edge, on the CPU.

Three refusals stood at coupler construction and two of them are retired
here.  The first said two-way feedback "has no ratified reverse mass/moment
mapping"; the second said it "requires identical active parent/child
prognostic field inventories", which is the same predicate restated, a
mixed pair being exactly a pair whose inventories differ.  Neither was a
missing mapping: ``resolve_microphysics_transition`` resolves ordered edges
over the whole ported matrix, so the reverse edge of any accepted forward
edge resolves too, and the CUDA entry is parameterized by
``(source_mp, target_mp)``, is column-local and already takes
``coupled=False``, which is what a restriction reads.  Only the wiring was
missing.  The THIRD refusal, unequal parent/child nz, is a different item
and still stands: the reverse operator has no vertical mapping.

Why a new file rather than tests/test_feedback.py: that module is marked
``gpu`` in its entirety, because its ``_seed_state`` helper imports cupy at
helper scope and tests/conftest.py's ``_cupy_scope`` marks the whole module
when a non-test helper opens a device (verified: 12 deselected, 0
collected under ``-m "not gpu"``).  A test added there would never run in
the CPU gate.  Nothing in this file imports cupy at any scope.
"""

from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from woof.config import RunConfig
from woof.core import nest as nest_mod
from woof.core import streaming
from woof.core.clock import DomainClock, DomainTicks
from woof.core.microphysics_transition import (
    MP8_TO_MP18_POLICY, transition_handles_field)
from woof.core.model import FeedbackScratch
from woof.core.nest import NestCoupler
from woof.core.nest_fields import nest_field_kinds
from woof.experiment import DomainConfig


# --------------------------------------------------------------------------
# configuration-only fixtures: the coupler resolves both edges from cfg
# --------------------------------------------------------------------------

def _domain(grid_id, parent_id, *, nx, ny, ratio=1, nz=2, **run_kwargs):
    root = parent_id == 0
    run = RunConfig(
        nx=nx, ny=ny, nz=nz, dx=3000.0 if root else 3000.0 / ratio,
        dy=3000.0 if root else 3000.0 / ratio, ztop=12000.0,
        dt=30.0 if root else 30.0 / ratio, run_seconds=30.0,
        output_interval_s=30.0, grid_id=grid_id,
        specified=root, nested=not root,
        spec_bdy_width=5, spec_zone=1, relax_zone=4, **run_kwargs)
    return DomainConfig(
        grid_id=grid_id, parent_id=parent_id,
        i_parent_start=1 if root else 4,
        j_parent_start=1 if root else 4,
        parent_grid_ratio=1 if root else ratio,
        parent_time_step_ratio=1 if root else ratio,
        history_interval_s=30.0, run=run,
        time_step=30 if root else None)


def _mixed_pair():
    """An mp=8 parent under an mp=18 child: the pair the first raise named."""
    parent_cfg = _domain(1, 0, nx=14, ny=14, moist=True, moist_cq=True,
                         mp_physics=8)
    child_cfg = _domain(2, 1, nx=9, ny=9, ratio=3, moist=True, moist_cq=True,
                        mp_physics=18,
                        nest_microphysics_transition=MP8_TO_MP18_POLICY)
    parent = SimpleNamespace(cfg=parent_cfg)
    return parent, SimpleNamespace(cfg=child_cfg, parent=parent)


def test_mixed_scheme_feedback_maps_instead_of_refusing():
    """``NestCoupler(child, feedback=1)`` CONSTRUCTS and carries the edge.

    RED before: ValueError matching
    'cross-scheme-feedback-reverse-mapping-unimplemented-v1'.
    """
    from woof.core.microphysics_transition import REVERSE_EDGE_POLICY

    parent, child = _mixed_pair()

    NestCoupler(child, feedback=0)                      # never refused
    coupler = NestCoupler(child, feedback=1)

    reverse = coupler.microphysics_reverse_transition
    assert reverse is not None
    assert reverse.mixed is True
    assert reverse.policy_id == REVERSE_EDGE_POLICY
    # The FORWARD edge is unchanged and still points the other way, so the
    # two are not one contract read twice.
    assert (coupler.microphysics_transition.source_mp_physics,
            coupler.microphysics_transition.target_mp_physics) == (8, 18)
    assert (reverse.source_mp_physics, reverse.target_mp_physics) == (18, 8)


def test_feedback_zero_resolves_no_reverse_edge():
    """The OFF control: a one-way tree pays for no reverse resolution."""
    _parent, child = _mixed_pair()
    assert NestCoupler(child, feedback=0).microphysics_reverse_transition \
        is None


def test_mismatched_inventories_no_longer_refuse_two_way():
    """Different ``nest_field_kinds`` on the two domains is not a refusal.

    RED before: ValueError 'requires identical active parent/child
    prognostic field inventories'.  A dry child under a moist parent is the
    cleanest witness, because the schemes MATCH there -- so this is the
    inventory predicate alone, with the microphysics predicate held out.
    """
    parent_cfg = _domain(1, 0, nx=14, ny=14, moist=True, mp_physics=6)
    child_cfg = _domain(2, 1, nx=9, ny=9, ratio=3, moist=False, mp_physics=6)
    parent = SimpleNamespace(cfg=parent_cfg)
    child = SimpleNamespace(cfg=child_cfg, parent=parent)

    assert nest_field_kinds(parent_cfg.run) != nest_field_kinds(child_cfg.run)
    coupler = NestCoupler(child, feedback=1)
    assert coupler.microphysics_reverse_transition.mixed is False


def test_unequal_vertical_levels_are_still_refused():
    """The third refusal is a DIFFERENT item and must survive this one.

    The reverse operator averages child cells onto parent cells with no
    vertical mapping, so a mismatched ladder would feed the parent values
    from the wrong levels.  That is a missing capability, not a missing
    measurement.
    """
    parent_cfg = _domain(1, 0, nx=14, ny=14, nz=2, moist=True, mp_physics=6)
    child_cfg = _domain(2, 1, nx=9, ny=9, ratio=3, nz=3, moist=True,
                        mp_physics=6)
    parent = SimpleNamespace(cfg=parent_cfg)
    child = SimpleNamespace(cfg=child_cfg, parent=parent)
    with pytest.raises(ValueError, match="horizontal-only.*vertical level"):
        NestCoupler(child, feedback=1)


# --------------------------------------------------------------------------
# the wiring itself: prepare admits the pair, commit diagnoses before it
# restricts
# --------------------------------------------------------------------------

def _clock(grid_id, parent_id, *, step_ticks, dt):
    spec = DomainTicks(
        grid_id=grid_id, parent_id=parent_id, parent_time_step_ratio=3,
        step_ticks=step_ticks, dt_fp32=np.float32(dt), history_ticks=100,
        restart_ticks=None, radt_ticks=None, stepra=None, cudt_ticks=None,
        stepcu=None, bldt_ticks=None, stepbl=None)
    return DomainClock(spec, tick_den=1, run_ticks=1000)


class _State:
    """Plain numpy, carrying the SCHEME's own species and nothing else."""

    def __init__(self, run, species):
        nz, ny, nx = run.nz, run.ny, run.nx
        self.mub2d = np.full((ny, nx), np.float32(90000.0))
        self.mup = np.full((ny, nx), np.float32(0.25))
        self.u = np.full((nz, ny, nx + 1), np.float32(2.0))
        self.v = np.full((nz, ny + 1, nx), np.float32(-1.5))
        self.w = np.full((nz + 1, ny, nx), np.float32(0.75))
        self.thp = np.full((nz, ny, nx), np.float32(1.25))
        self.php = np.full((nz + 1, ny, nx), np.float32(3.0))
        self.phb = np.zeros((nz + 1,), np.float32)
        self.thb = np.full((nz,), np.float32(300.0))
        self.c1h = np.full((nz,), np.float32(1.0))
        self.c2h = np.zeros((nz,), dtype=np.float32)
        self.p = np.full((nz, ny, nx), np.float32(9.0e4))
        self.alt = np.full((nz, ny, nx), np.float32(0.9))
        for index, name in enumerate(species, start=1):
            setattr(self, name, np.full(
                (nz, ny, nx), np.float32(index / 100.0)))
        self._scratch = {}
        self.lateral_boundaries = None

    def scratch(self, shape, slot, dtype=None):
        dtype = np.dtype(np.float32 if dtype is None else dtype)
        shape = tuple(shape)
        if slot not in self._scratch:
            self._scratch[slot] = np.zeros(shape, dtype=dtype)
        result = self._scratch[slot]
        assert result.shape == shape and result.dtype == dtype
        return result


_MP8_SPECIES = ("qv", "qc", "qr", "qi", "qs", "qg", "nr", "ni")
_MP18_SPECIES = ("qv", "qc", "qr", "qi", "qs", "qg", "qh", "qndrop", "qnr",
                 "qni", "qns", "qng", "qnh", "qnn", "qvolg", "qvolh")


def _live_mixed_pair():
    """The same mixed pair, with states, clocks and a matched tick."""
    parent_cfg, child_cfg = _mixed_pair()[1].parent.cfg, _mixed_pair()[1].cfg
    parent = SimpleNamespace(
        cfg=parent_cfg, state=_State(parent_cfg.run, _MP8_SPECIES),
        clock=_clock(1, 0, step_ticks=3, dt=30.0))
    child = SimpleNamespace(
        cfg=child_cfg, state=_State(child_cfg.run, _MP18_SPECIES),
        parent=parent, clock=_clock(2, 1, step_ticks=1, dt=10.0))
    return parent, child


def _prepared(coupler, child):
    out = FeedbackScratch()
    coupler.feedback_prepare(child, out)
    return out.payload


def test_prepare_admits_a_parent_species_the_child_cannot_name():
    """The inventory test asks what the child can SUPPLY, not what it holds.

    The parent runs mp=8 and the child mp=18, so the parent's ``nr``/``ni``
    have no same-named child array at all.  The reverse contract diagnoses
    them from the child's own moments, so they stay in the transaction;
    before this they were 'child state lacks parent feedback fields'.
    """
    parent, child = _live_mixed_pair()
    coupler = NestCoupler(child, feedback=1)
    payload = _prepared(coupler, child)

    assert "nr" in payload["kinds"] and "ni" in payload["kinds"]
    for kind in ("nr", "ni"):
        assert getattr(child.state, kind, None) is None, (
            "the fixture's child carries the species by name, so this "
            "proves nothing about the diagnosed route")
    assert set(payload["kinds"]) <= set(nest_field_kinds(parent.cfg.run))


def test_a_mixed_pair_drops_nothing_and_warns_about_nothing():
    """Every parent microphysics kind is a target field of its own scheme.

    Held explicitly so the drop branch below is not read as the mixed
    case's normal outcome: a mixed edge diagnoses the whole parent
    inventory, so the transaction is complete and silent.
    """
    parent, child = _live_mixed_pair()
    coupler = NestCoupler(child, feedback=1)

    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("error")
        payload = _prepared(coupler, child)
    assert payload["dropped_kinds"] == ()
    assert set(payload["kinds"]) == set(nest_field_kinds(parent.cfg.run))


def test_a_parent_species_nothing_can_supply_is_dropped_and_warned_once():
    """Run what can run, say what did not, do not refuse the step.

    A DRY child under a moist parent: same scheme, so nothing is
    diagnosed, and the child has no moisture array to restrict.  Before
    this that pair was refused outright at coupler construction; the
    dynamics it CAN feed back are fed back, the moisture is dropped, and
    the omission is stated once per coupler rather than once per step.
    """
    parent_cfg = _domain(1, 0, nx=14, ny=14, moist=True, mp_physics=6)
    child_cfg = _domain(2, 1, nx=9, ny=9, ratio=3, moist=False, mp_physics=6)
    parent = SimpleNamespace(
        cfg=parent_cfg, state=_State(parent_cfg.run, ("qv", "qc", "qr",
                                                      "qi", "qs", "qg")),
        clock=_clock(1, 0, step_ticks=3, dt=30.0))
    child = SimpleNamespace(
        cfg=child_cfg, state=_State(child_cfg.run, ()), parent=parent,
        clock=_clock(2, 1, step_ticks=1, dt=10.0))
    coupler = NestCoupler(child, feedback=1)

    out = FeedbackScratch()
    with pytest.warns(RuntimeWarning, match="restricts"):
        coupler.feedback_prepare(child, out)
    payload = out.payload

    dropped = set(payload["dropped_kinds"])
    assert dropped == {"qv", "qc", "qr", "qi", "qs", "qg"}
    assert dropped.isdisjoint(payload["kinds"])
    assert set(payload["kinds"]) | dropped == set(
        nest_field_kinds(parent_cfg.run))
    assert set(payload["kinds"]) == {"u", "v", "w", "t", "ph", "mu"}

    # Once per coupler, not once per parent step.
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("error")
        coupler.feedback_prepare(child, FeedbackScratch())


def test_commit_diagnoses_the_parents_species_before_it_restricts(monkeypatch):
    """The mirror of the FORCE path's mapped_parent branch.

    A species the reverse edge handles must reach ``copy_fcn`` as the
    DIAGNOSIS of the child's own moments, never as a same-named copy.  The
    launcher is stubbed because the real one is a CUDA kernel; what is
    under test is the wiring around it -- that it is called at all, with
    the child as the source, with ``coupled=False``, and that its output is
    what the restriction consumes.
    """
    from woof.core.microphysics_transition import REVERSE_EDGE_POLICY

    parent, child = _live_mixed_pair()
    coupler = NestCoupler(child, feedback=1)
    monkeypatch.setattr(coupler, "_bind_geometry", lambda: None)
    payload = _prepared(coupler, child)

    diagnosed = []

    def _fake_edge(contract, state, field_name, *, out, coupled):
        diagnosed.append((field_name, state is child.state, coupled,
                          contract.policy_id))
        out[...] = np.float32(7.0 + len(diagnosed))
        return out

    restricted = []
    monkeypatch.setattr(nest_mod, "launch_microphysics_edge_field", _fake_edge)
    monkeypatch.setattr(
        nest_mod, "copy_fcn",
        lambda parent_field, child_field, reg, *, spec_zone: restricted.append(
            (tuple(parent_field.shape), float(np.asarray(child_field).ravel()[0]))))
    monkeypatch.setattr(nest_mod, "smoother", lambda *a, **k: None)
    monkeypatch.setattr(coupler, "_feedback_surface", lambda plan: None)

    coupler.feedback_commit(child)

    handled = [name for name, *_ in diagnosed]
    assert handled, "no parent species was diagnosed; the wiring is not there"
    for name, is_child, coupled, policy in diagnosed:
        assert is_child, f"{name} was diagnosed off the wrong grid"
        assert coupled is False, (
            f"{name} was diagnosed COUPLED; a restriction reads the "
            "uncoupled field")
        assert policy == REVERSE_EDGE_POLICY

    # Every diagnosed value reached the restriction, in order.
    values = [value for _shape, value in restricted]
    for index in range(len(diagnosed)):
        assert pytest.approx(7.0 + index + 1) in values


def _bounded(state):
    """Serve this state's arrays the way a canonically streamed domain does."""
    state._streamed_domain = SimpleNamespace(store={}, template_state=None)
    return state


def test_a_streamed_child_drops_what_it_cannot_diagnose_and_says_so_once():
    """The one shape the reverse launcher cannot serve NARROWS the
    transaction; it does not stop a started run.

    The launcher reads the child's planes off a live state, and a
    canonically streamed child serves its arrays through
    ``NestWindowSource`` instead, so those species cannot be diagnosed from
    this child at all.  They are dropped at prepare, with both ways out
    named once, and everything the edge CAN restrict is restricted.
    Serving that child is deferred work (the restriction would have to run
    chunk by chunk through ``NestWindowSource``, which is GPU-only), and
    deferred work is not a reason to end a forecast mid-flight.

    RED before: ValueError
    'cross-scheme-feedback-reverse-mapping-unimplemented-v1' out of
    ``NestCoupler(child, feedback=1)``, which is where the pair used to
    stop.
    """
    _parent, resident_child = _live_mixed_pair()
    control = _prepared(NestCoupler(resident_child, feedback=1), resident_child)
    assert control["dropped_kinds"] == ()          # the resident control

    parent, child = _live_mixed_pair()
    _bounded(child.state)
    coupler = NestCoupler(child, feedback=1)
    reverse = coupler.microphysics_reverse_transition

    out = FeedbackScratch()
    with pytest.warns(RuntimeWarning) as record:
        coupler.feedback_prepare(child, out)
    payload = out.payload

    dropped = set(payload["dropped_kinds"])
    diagnosed = {kind for kind in control["kinds"]
                 if transition_handles_field(reverse, kind)}
    assert diagnosed, "the fixture diagnoses nothing; the control is empty"
    assert dropped == diagnosed
    assert dropped.isdisjoint(payload["kinds"])
    assert set(payload["kinds"]) | dropped == set(
        nest_field_kinds(parent.cfg.run))
    assert "mu" in payload["kinds"], "the mass is not a microphysics species"

    text = str(record[0].message)
    assert "tile-streamed" in text
    assert "tiles.mode = 'off'" in text and "same mp_physics" in text

    # Once per coupler, not once per parent step.
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("error")
        coupler.feedback_prepare(child, FeedbackScratch())


def test_a_streamed_childs_commit_restricts_no_unconverted_species(monkeypatch):
    """The bounded arm never sees a species that owes a conversion.

    ``_restrict_windowed`` reads the child's array BY NAME out of the
    store, chunk by chunk.  Across mixed schemes that same-named array is
    not the parent's species, so restricting it unconverted would be the
    finite, plausible, wrong parent this whole edge exists to prevent --
    which is what makes this a control and not a restatement of the
    prepare-side test above.  The reverse launcher must also never be
    reached with a bounded child as its source.
    """
    parent, child = _live_mixed_pair()
    _bounded(child.state)
    coupler = NestCoupler(child, feedback=1)
    monkeypatch.setattr(coupler, "_bind_geometry", lambda: None)

    restricted = []
    monkeypatch.setattr(
        coupler, "_restrict_windowed",
        lambda kind, child_source, parent_source: restricted.append(kind))

    def _never(*args, **kwargs):
        raise AssertionError(
            "the reverse launcher was handed a tile-streamed child")

    monkeypatch.setattr(nest_mod, "launch_microphysics_edge_field", _never)
    monkeypatch.setattr(nest_mod, "copy_fcn", _never)
    monkeypatch.setattr(nest_mod, "smoother", _never)

    with pytest.warns(RuntimeWarning):
        payload = _prepared(coupler, child)
    coupler.feedback_commit(child)

    reverse = coupler.microphysics_reverse_transition
    assert restricted, "nothing was restricted at all"
    assert set(restricted) == set(payload["kinds"])
    owed = [kind for kind in restricted
            if transition_handles_field(reverse, kind)]
    assert owed == [], (
        f"{owed} reached the windowed restriction unconverted")
