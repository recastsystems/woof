"""The streaming mode's contract, on the CPU.

The expensive half of this feature -- that a streamed domain reproduces a
resident one bit for bit -- is proven on a GPU by ``tilestream/test_gate.py``
and ``tilestream/test_join.py``.  What is proven here is everything that must
hold BEFORE a card is involved, and above all the OFF contract: a tree that
configures no ``[tiles]`` must execute the code it executed before this
feature existed, through the same call, with the same identity, and must
fingerprint the same.

The OFF contract is stated as an object identity rather than as a behaviour,
because that is the only form of it that cannot rot.  ``make_stepper`` with
streaming off returns ``woof.core.dycore.step`` ITSELF; a wrapper that
merely forwarded would satisfy every behavioural test and would still be a
second code path to keep correct.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime
import json

import pytest

from woof.core import streaming
from woof.core.streaming import (OFF, StreamingOptions, StreamedDomain,
                                  StreamingRefused, decide, is_streaming,
                                  make_stepper)


# --------------------------------------------------------------------------
# the OFF contract
# --------------------------------------------------------------------------

def test_off_returns_the_dycore_step_itself():
    """Not a wrapper.  The same function object, so there is no branch."""
    from woof.core.dycore import step

    assert make_stepper(object(), _cfg(), OFF) is step
    assert make_stepper(object(), _cfg(), None) is step
    assert not is_streaming(step)


def test_auto_that_fits_resident_also_returns_the_dycore_step(monkeypatch):
    """``auto`` on a domain that fits is OFF, not "on with one tile"."""
    from woof.core.dycore import step

    monkeypatch.setattr(streaming, "decide",
                        lambda *_a, **_k: streaming.StreamingDecision(
                            False, "fits"))
    assert make_stepper(object(), _cfg(), StreamingOptions(mode="auto")) \
        is step


def test_absent_streaming_block_is_the_shared_off_object():
    assert StreamingOptions.from_mapping(None) is OFF
    assert OFF.enabled is False
    assert StreamingOptions(mode="on").enabled is True


def test_streaming_contributes_nothing_to_the_restart_identity():
    """A checkpoint written resident must resume streamed, and vice versa.

    Held as a payload comparison rather than as a promise: the identity of an
    experiment that streams and the identity of the same experiment that does
    not must be the SAME STRING, so a fingerprint computed before this
    feature existed still matches.
    """
    from woof.core.model import restart_identity_payload

    exp = _experiment()
    resident = restart_identity_payload(exp)
    streamed = restart_identity_payload(dataclasses.replace(
        exp, tiles=StreamingOptions(mode="on", tile_nx=512, tile_ny=512)))
    assert json.dumps(resident, sort_keys=True) == \
        json.dumps(streamed, sort_keys=True)
    assert "tiles" not in resident
    assert streaming.identity_payload_entry(
        StreamingOptions(mode="on", tile_nx=8, tile_ny=8)) == {}


def test_the_executor_binds_the_dycore_step_when_nothing_is_configured(
        monkeypatch):
    """The seam is the call site: an unconfigured tree still calls ``step``.

    The existing executor tests monkeypatch ``woof.core.dycore.step`` and
    count calls; this asserts that the indirection introduced for streaming
    did not quietly stop that from being the function that runs.
    """
    from test_model import _model                      # noqa: PLC0415
    from woof.core.model import execute_experiment

    _exp, model = _model()
    calls = []
    monkeypatch.setattr(
        "woof.core.dycore.step",
        lambda state, cfg, **kw: calls.append(int(cfg.grid_id)))
    report = execute_experiment(model, validate_state=False,
                                pool_trim_per_period=False)
    assert len(calls) == report.steps == 20


def test_a_bound_stepper_replaces_the_dycore_step_for_that_grid_only(
        monkeypatch):
    """One domain streams; its sibling does not.  The seam is per grid."""
    from test_model import _model                      # noqa: PLC0415
    from woof.core.model import execute_experiment

    _exp, model = _model()
    dycore_calls, streamed_calls = [], []
    monkeypatch.setattr(
        "woof.core.dycore.step",
        lambda state, cfg, **kw: dycore_calls.append(int(cfg.grid_id)))

    def fake_streamed(state, cfg, **kw):
        streamed_calls.append((int(cfg.grid_id), bool(kw["refl_10cm_due"])))

    execute_experiment(model, validate_state=False,
                       pool_trim_per_period=False,
                       history_handler=lambda *_a: None,
                       steppers={1: fake_streamed})
    assert set(dycore_calls) == {2}
    assert {gid for gid, _ in streamed_calls} == {1}
    assert len(dycore_calls) == len(streamed_calls) == 10
    # The REFL_10CM handshake has to reach the streamed domain too: it
    # decides whether the step stages a field the frame about to be written
    # will consume, and a streamed sweep must stage it on every tile.
    assert any(due for _gid, due in streamed_calls)


def test_two_way_feedback_reaches_the_coupler_under_a_streamed_parent(
        monkeypatch):
    """``feedback = 1`` + streamed parent flows TO the coupler, not a refusal.

    The executor used to raise here, claiming "nothing projects a write
    back".  That sentence stopped being true in the same merge that landed
    it: ``NestCoupler.feedback_commit`` starts its transaction from the
    parent's store and ends it there (``_sync_in``/``_sync_out`` ->
    ``streaming.refresh_from_store``/``commit_to_store``), and
    ``tilestream/test_nest.py`` leg 3 measures that path bit-identical to
    the all-resident run with both feedback controls firing.  The executor
    and the coupler came from different lanes of the three-way merge
    2358b3c06 and the merge kept the refusal beside the capability it
    refused.

    The refusal was reachable ONLY through ``execute_experiment`` -- the
    coupler gate drives ``feedback_commit`` directly -- which is why no gate
    ever caught the contradiction.  This test is the executor-level pin:
    the schedule's FEEDBACK edges must dispatch into the coupler for a
    parent whose stepper streams, exactly as they do for a resident one.
    The ownership question (who moves the bytes) is the coupler's, and its
    streamed arm is proven where the store actually exists -- on the GPU
    gate ``tilestream/test_nest_executor.py``.
    """
    from types import SimpleNamespace

    from test_model import _model                      # noqa: PLC0415
    from woof.core.model import execute_experiment

    monkeypatch.setattr("woof.core.dycore.step",
                        lambda state, cfg, **kw: None)
    _exp, model = _model()
    child = model.node(2)
    # The stub coupler records its dispatches; giving it ``feedback = 1``
    # makes it a two-way coupler as far as the executor's dispatch is
    # concerned, which is the object under test here.
    child.coupler.feedback = 1
    run = SimpleNamespace(cfg=model.root.cfg.run, store={},
                          sweep=lambda *a, **k: None)
    stepper = StreamedDomain(
        run, streaming.StreamingDecision(True, "pinned by the test"),
        scalars={})
    assert is_streaming(stepper)
    execute_experiment(model, validate_state=False,
                       pool_trim_per_period=False,
                       steppers={1: stepper})
    calls = child.coupler.calls
    assert "commit" in calls, (
        "the executor never dispatched feedback_commit for a streamed "
        "parent -- the stale refusal is back, or the dispatch moved")
    assert "finalize" in calls
    # Dispatch order within one edge is prepare -> commit -> finalize.
    assert calls.index("prepare") < calls.index("commit") \
        < calls.index("finalize")


def test_the_store_seam_moves_only_the_window_when_given_one():
    """``refresh_from_store``/``commit_to_store`` take the coupler's window.

    The FORCE corridor's capacity claim is that traffic per parent step is
    O(child footprint), not O(parent).  That is only true if the seam the
    coupler pulls through can be told the window -- and the slice rule must
    be the same superset rule ``StreamedDomain._window_slices`` applies
    (widen by one for a staggered face, clamp to the array), or the two
    sides of the same window disagree about a face on the window's edge.
    """
    import numpy as np

    state = type("S", (), {})()
    state.thp = np.zeros((3, 8, 8), dtype=np.float32)
    store = {"state/thp": np.arange(3 * 8 * 8, dtype=np.float32)
             .reshape(3, 8, 8)}
    setattr(state, "_streamed_store", store)

    moved = streaming.refresh_from_store(state, ("thp",), window=(2, 5, 3, 6))
    # The superset rule: rows 2..5 and columns 3..6 INCLUSIVE (the +1 is
    # the staggered-face widening, applied unconditionally because a
    # superset is always safe), everything else untouched.
    expect = np.zeros((3, 8, 8), dtype=np.float32)
    expect[:, 2:6, 3:7] = store["state/thp"][:, 2:6, 3:7]
    assert np.array_equal(state.thp, expect)
    assert moved == expect[:, 2:6, 3:7].nbytes

    # The write half, same rule: only the window lands in the store.
    state.thp = np.full((3, 8, 8), 7.0, dtype=np.float32)
    before = store["state/thp"].copy()
    moved = streaming.commit_to_store(state, ("thp",), window=(2, 5, 3, 6))
    assert np.all(store["state/thp"][:, 2:6, 3:7] == 7.0)
    outside = np.ones((8, 8), dtype=bool)
    outside[2:6, 3:7] = False
    assert np.array_equal(store["state/thp"][:, outside], before[:, outside])
    assert moved == before[:, 2:6, 3:7].nbytes

    # window=None is the whole-field path, byte-identical to before the
    # parameter existed.
    state.thp = np.zeros((3, 8, 8), dtype=np.float32)
    assert streaming.refresh_from_store(state, ("thp",)) \
        == store["state/thp"].nbytes
    assert np.array_equal(state.thp, store["state/thp"])


def test_the_parent_footprint_window_is_the_couplers_own():
    """The window is computed where the reads happen, from the live cfg.

    ``model.py`` used to hold this arithmetic beside a projection the
    executor did on the coupler's behalf; the corridor moves both into the
    coupler so a relocation (which rewrites the child's cfg) moves the
    window with no second copy of the geometry to forget.  The halo bound
    and the derivation stay exactly what the executor carried: SINT +-2,
    spec_bdy_width child cells, u/v face averages, 8 bounds all of it at
    every admitted ratio, superset free / subset stale.
    """
    from types import SimpleNamespace

    from woof.core.nest import (NEST_FORCE_HALO_PARENT_CELLS,
                                 parent_footprint_window)

    dc = SimpleNamespace(parent_grid_ratio=3, i_parent_start=81,
                         j_parent_start=81,
                         run=SimpleNamespace(nx=96, ny=96))
    assert NEST_FORCE_HALO_PARENT_CELLS == 8
    # 0-based origin 80, span ceil(96/3) = 32, padded by 8 each side.
    assert parent_footprint_window(dc) == (72, 120, 72, 120)
    # Near an edge the window goes negative and the SLICE clamps -- the
    # split the executor used: raw window here, clamp in _window_slices.
    dc_edge = SimpleNamespace(parent_grid_ratio=3, i_parent_start=2,
                              j_parent_start=2,
                              run=SimpleNamespace(nx=96, ny=96))
    assert parent_footprint_window(dc_edge) == (-7, 41, -7, 41)


def test_the_single_domain_loop_defaults_to_the_dycore_step():
    """``integrate_prepared_case`` binds ``step`` when handed no stepper."""
    import inspect

    from woof import runtime

    src = inspect.getsource(runtime.integrate_prepared_case)
    assert "stepper = step if stepper is None else stepper" in src
    assert "stepper(state, integration_cfg, **step_kwargs)" in src
    assert "        step(state, integration_cfg" not in src


# --------------------------------------------------------------------------
# the configuration surface
# --------------------------------------------------------------------------

def test_unknown_keys_are_refused_not_ignored():
    with pytest.raises(ValueError, match="unknown key"):
        StreamingOptions.from_mapping({"mode": "on", "tiles": 4})


def test_an_off_surface_must_be_empty():
    """The [relocation] discipline: a mode that is off carries no settings."""
    with pytest.raises(ValueError, match="carries a tiling while mode"):
        StreamingOptions(tile_nx=64, tile_ny=64)
    with pytest.raises(ValueError, match="together"):
        StreamingOptions(mode="on", tile_nx=64)


@pytest.mark.parametrize("table,message", [
    ({"mode": "sometimes"}, "not one of"),
    ({"mode": "on", "store": "disk"}, "must be"),
    ({"mode": "on", "write_mode": "inplace"}, "read-at-time-t"),
    ({"mode": "on", "nbuffers": 0}, "must be positive"),
])
def test_refusals(table, message):
    with pytest.raises(ValueError, match=message):
        StreamingOptions.from_mapping(table)


def test_experiment_toml_carries_the_block(tmp_path):
    from woof.experiment import build_experiment

    raw = _raw_experiment()
    exp = build_experiment(raw, source="<test>")
    assert exp.tiles is OFF

    raw["tiles"] = {"mode": "auto"}
    exp = build_experiment(raw, source="<test>")
    assert exp.tiles.mode == "auto"

    raw["tiles"] = {"mode": "auto", "tile_nixe": 64}
    with pytest.raises(ValueError, match="unknown key"):
        build_experiment(raw, source="<test>")


def test_a_stray_table_is_still_refused():
    """Adding [tiles] to the schema must not open the schema."""
    from woof.experiment import build_experiment

    raw = _raw_experiment()
    raw["streamimg"] = {"mode": "auto"}
    with pytest.raises(ValueError, match="does not have a table"):
        build_experiment(raw, source="<test>")


# --------------------------------------------------------------------------
# the decision
# --------------------------------------------------------------------------

def _hide_the_planner(monkeypatch):
    """Make ``tilestream.autoplan`` unreachable, whatever ran before.

    ``sys.modules[name] = None`` alone stops ``import tilestream.autoplan``
    but not ``from tilestream import autoplan`` once anything earlier in the
    process imported the planner, because that form reads the package
    attribute first.  So the attribute goes too: the tests below then mean
    the same thing alone, under xdist and after the planner was imported,
    instead of passing or failing by import order (A177).
    """
    import sys

    import tilestream

    monkeypatch.setitem(sys.modules, "tilestream.autoplan", None)
    monkeypatch.delattr(tilestream, "autoplan", raising=False)


def test_off_decides_without_importing_the_planner(monkeypatch):
    """``mode = "off"`` must not need a card, a planner or cupy."""
    _hide_the_planner(monkeypatch)
    got = decide(_cfg(), OFF)
    assert got.stream is False
    assert "off" in got.explain()


def test_a_pinned_tiling_consults_no_planner(monkeypatch):
    from tilestream import spec

    _hide_the_planner(monkeypatch)
    got = decide(_cfg(), StreamingOptions(mode="on", tile_nx=512,
                                          tile_ny=256, nbuffers=3))
    assert (got.stream, got.tile_nx, got.tile_ny, got.nbuffers) == \
        (True, 512, 256, 3)
    # The halo is taken from harness.halo_radius and NEVER configured: it is
    # 10 + 3*time_step_sound//2 and a smaller one is silent and faster.
    assert got.halo == 10 + 3 * _cfg().time_step_sound // 2
    # The pinned road still states its tile count and redundancy, from the
    # tile geometry rather than the planner.
    assert got.ntiles == 1
    assert got.redundancy == spec.redundancy(256, 192, 512, 256, got.halo)


def test_the_planner_and_a_pinned_tiling_share_one_redundancy():
    """One function, not a copy that could drift from the planner's."""
    from tilestream import autoplan, spec

    assert autoplan.redundancy is spec.redundancy


def test_a_short_halo_warns_loudly():
    with pytest.warns(RuntimeWarning, match="how this defect hides"):
        decide(_cfg(), StreamingOptions(mode="on", tile_nx=64, tile_ny=64,
                                        halo=4))


# --------------------------------------------------------------------------
# the host-memory source, and the order the budget is read in
#
# Streaming could not start AT ALL on Windows through the 2.2.x line, and it
# took two independent defects to do it.  ``autoplan`` knew only
# ``/proc/meminfo``, which that OS does not have, so ``Machine.detect``
# raised there; and ``decide`` probed the machine BEFORE it read
# ``host_budget_bytes``, so the one key that could have supplied the missing
# number was unreachable behind the refusal that asked for it.  Every one of
# these runs CPU-only and on every platform: the defects are in the ordering
# and in the source selection, and neither needs a Windows box to state.
# --------------------------------------------------------------------------

def _detect_that_needs_a_host_figure(calls):
    """A stand-in for ``Machine.detect`` on a box with no readable RAM.

    It reproduces the REAL contract rather than failing unconditionally --
    ``host_bytes=`` supplied means no host source is consulted, so the call
    succeeds; ``host_bytes=None`` means it must be read, and here it cannot
    be.  A fake that raised either way would pass a ``decide`` that had been
    fixed and one that had not.
    """
    from tilestream.autoplan import CannotPlan, Machine

    def fake_detect(*, host_bytes=None, device=0, use_free_vram=True):
        calls.append(host_bytes)
        if host_bytes is None:
            raise CannotPlan("no host-memory source on this platform", "host")
        return Machine(vram_bytes=24 * 2 ** 30, host_bytes=int(host_bytes),
                       name="test", host_source="explicit")

    return fake_detect


def test_the_configured_host_budget_is_read_before_the_machine_is_probed(
        monkeypatch):
    """``host_budget_bytes`` must reach the probe, not arrive after it.

    THE WINDOWS DEFECT, stated without a Windows box.  ``decide`` called
    ``Machine.detect()`` with no arguments and applied the configured budget
    four lines later, so where detection raised -- which on Windows was
    always -- the configured budget was never read at all.
    """
    from tilestream import autoplan

    calls = []
    monkeypatch.setattr(autoplan.Machine, "detect",
                        _detect_that_needs_a_host_figure(calls))
    got = decide(_cfg(), StreamingOptions(mode="on",
                                          host_budget_bytes=48 * 2 ** 30))
    assert got.stream is True
    assert calls == [48 * 2 ** 30], (
        "detect was probed without the configured host figure; the budget "
        "is still being applied after the probe instead of supplied to it")


def test_a_configured_budget_is_still_the_budget_when_detection_works(
        monkeypatch):
    """THE CONTROL for the reordering: a box that CAN detect is unchanged.

    Supplying ``host_bytes=`` to the probe must not double-apply the pinning
    fraction or otherwise move the number, or the fix would have bought
    Windows a plan by silently repricing every Linux node.
    """
    from tilestream import autoplan

    seen = {}
    real_machine = autoplan.Machine(
        vram_bytes=24 * 2 ** 30, host_bytes=512 * 2 ** 30, name="test",
        host_source="explicit")
    real_plan = autoplan.plan          # bound before the patch, not through it

    def fake_plan(cfg, machine, **kwargs):
        seen["host_budget"] = machine.host_budget_bytes
        return real_plan(cfg, machine, **kwargs)

    monkeypatch.setattr(autoplan, "plan", fake_plan)
    decide(_cfg(), StreamingOptions(mode="on",
                                    host_budget_bytes=48 * 2 ** 30),
           machine=real_machine)
    assert seen["host_budget"] == 48 * 2 ** 30, (
        "the configured host budget was multiplied by pinned_fraction "
        "instead of being taken as the budget it names")


def test_the_planner_reads_host_ram_on_windows_not_only_on_linux(monkeypatch):
    """``_host_memtotal`` must have a source on the product's own platform.

    Windows has no procfs, and treating that absence as "unknowable" is what
    made ``Machine.detect`` raise on every Windows box.  The probe is
    selected by platform, so this states the selection rather than the Win32
    call: on ``win32`` the answer must come from the Windows probe.
    """
    from tilestream import autoplan

    monkeypatch.setattr(autoplan.sys, "platform", "win32")
    monkeypatch.setattr(autoplan, "_windows_memtotal", lambda: 95 * 2 ** 30)
    assert autoplan._host_memtotal() == 95 * 2 ** 30
    assert "GlobalMemoryStatusEx" in autoplan._memtotal_source()


def test_the_linux_host_source_is_unchanged(monkeypatch):
    """THE CONTROL for the platform switch: Linux still reads procfs.

    A fix that routed every platform through the Windows probe would satisfy
    the test above and would break every node the campaigns run on.
    """
    from tilestream import autoplan

    monkeypatch.setattr(autoplan.sys, "platform", "linux")
    monkeypatch.setattr(autoplan, "_windows_memtotal",
                        lambda: pytest.fail("Linux consulted the Win32 probe"))
    autoplan._host_memtotal()          # reads /proc/meminfo, or answers None
    assert autoplan._memtotal_source() == "/proc/meminfo MemTotal"


@pytest.mark.gpu
def test_an_unreadable_host_source_is_not_reported_as_a_container(
        monkeypatch):
    """The refusal must name the true limit, not the container it isn't.

    Where nothing can be read, the message said "containerised with no
    cgroup memory limit" -- on a native Windows desktop, which sent the
    reader hunting for a container that was never there.

    Marked ``gpu`` because it OPENS ONE.  The subject is host memory and
    every host probe below is stubbed, but ``Machine.detect`` reads the
    card first -- ``cp.cuda.Device(device)`` then ``memGetInfo`` -- and
    only then reaches the host-source logic under test.  The conftest's
    AST detector cannot see that: this module never spells ``cupy``, so
    the device is reached transitively and the marker was absent.  On a
    leg with no visible device the test therefore FAILED
    (``cudaErrorNoDevice``) instead of being deselected, and a suite that
    cries wolf on every CPU-only run is how a real failure gets waved
    through.
    """
    from tilestream import autoplan

    monkeypatch.setattr(autoplan, "_cgroup_memory_limit", lambda: None)
    monkeypatch.setattr(autoplan, "_host_memtotal", lambda: None)
    monkeypatch.setattr(autoplan, "_in_container", lambda: False)
    monkeypatch.setattr(autoplan.sys, "platform", "win32")
    with pytest.raises(autoplan.CannotPlan) as caught:
        autoplan.Machine.detect(device=0)
    message = str(caught.value)
    assert "containerised" not in message, message
    assert "no host-memory source" in message and "win32" in message


@pytest.mark.gpu
def test_a_container_with_no_cgroup_limit_is_still_refused_as_one(
        monkeypatch):
    """THE CONTROL: the container refusal it replaced must still fire.

    ``/proc/meminfo`` inside a container reports the HOST's RAM (measured:
    503 GiB against a 241.7 GiB cgroup limit), so a readable MemTotal with
    no limit beside it is exactly the case that must keep refusing.

    Marked ``gpu`` for the same reason as its subject above: the control
    goes through the same ``Machine.detect``, which reads the card before
    it reads the host.
    """
    from tilestream import autoplan

    monkeypatch.setattr(autoplan, "_cgroup_memory_limit", lambda: None)
    monkeypatch.setattr(autoplan, "_host_memtotal", lambda: 503 * 2 ** 30)
    monkeypatch.setattr(autoplan, "_in_container", lambda: True)
    with pytest.raises(autoplan.CannotPlan, match="containerised"):
        autoplan.Machine.detect(device=0)


def test_the_pricing_and_planning_paths_read_one_host_probe():
    """They disagreed, and that disagreement WAS the bug.

    ``woof.core.streaming._host_total_bytes`` carried its own Win32 probe,
    so the memory gate could price host RAM on Windows while the planner
    refused to plan on the same box.  One source, in the lower layer.
    """
    import inspect

    from woof.core import streaming as streaming_module

    source = inspect.getsource(streaming_module._host_total_bytes)
    # The BODY, not the prose: the docstring names the probe it no longer
    # carries, and a substring check that read the docstring would fire on
    # the explanation of the fix.  Split on the docstring's own delimiters
    # rather than subtracting ``getdoc``, which returns it dedented and so
    # matches nothing in the raw source.
    body = source.split('"""')[2]
    assert "ctypes" not in body and "GlobalMemoryStatusEx" not in body, (
        "the pricing path grew a second host probe again; it must delegate "
        "to tilestream.autoplan so the two answers cannot diverge")
    assert "_host_memtotal" in body


# --------------------------------------------------------------------------
# what the run RECORDS about which way it decided
#
# The refusal above is the loud failure.  These cover the QUIET one, which
# is the same defect pointing the other way: `auto` on a domain that fits
# returns an empty stepper dict, and an empty stepper dict is also what
# `off` returns, and what a config with no [tiles] table returns.  So a
# run that was asked to stream and declined looked exactly like a run that
# was never asked -- to an operator reading a log, and to a test.
# --------------------------------------------------------------------------

def _fitting_machine():
    """A machine with room to spare, so ``auto`` decides RESIDENT.

    Explicit rather than ``Machine.detect()``: detect reads free VRAM off a
    real card, so on CI or a busy box this test would measure the neighbour
    instead of the logic -- and would flip verdict depending on who else was
    running.
    """
    from tilestream.autoplan import Machine

    return Machine(vram_bytes=80 * 2 ** 30, host_bytes=512 * 2 ** 30,
                   name="test", host_source="explicit")


def _one_grid_model(cfg=None):
    import types

    cfg = _cfg() if cfg is None else cfg
    node = types.SimpleNamespace(
        cfg=types.SimpleNamespace(grid_id=1, run=cfg), state=object())
    return types.SimpleNamespace(walk_parent_first=lambda: [node])


def test_auto_that_declines_is_recorded_rather_than_merely_absent():
    """The stepper dict says nothing; the decisions mapping says everything."""
    model = _one_grid_model()
    decisions: dict = {}
    steppers = streaming.steppers_for_tree(
        model, StreamingOptions(mode="auto"), machine=_fitting_machine(),
        decisions=decisions)
    assert steppers == {}                      # nothing streamed ...
    assert set(decisions) == {1}               # ... but it was DECIDED
    assert decisions[1].stream is False
    assert "fits resident" in decisions[1].reason


def test_the_receipt_tells_a_declined_auto_apart_from_an_unconfigured_run():
    """THE CONTROL, and it is the whole point of the receipt.

    Both runs execute resident and both hand the executor ``{}``.  If the
    receipt cannot separate them, nothing downstream can, and an ``auto``
    run that never streamed is indistinguishable from a streaming success.
    """
    model, machine = _one_grid_model(), _fitting_machine()

    off_decisions: dict = {}
    streaming.steppers_for_tree(model, OFF, machine=machine,
                                decisions=off_decisions)
    off = streaming.streaming_receipt(OFF, off_decisions)

    auto_decisions: dict = {}
    streaming.steppers_for_tree(model, StreamingOptions(mode="auto"),
                                machine=machine, decisions=auto_decisions)
    auto = streaming.streaming_receipt(StreamingOptions(mode="auto"),
                                       auto_decisions)

    assert off == {}                           # unconfigured contributes nothing
    assert auto != {}                          # configured always says something
    assert auto["streamed_any"] is False
    assert auto["configured_mode"] == "auto"
    assert auto["domains"]["1"]["streamed"] is False
    assert "NO domain streamed" in auto["summary"]
    # The two receipts must not merely differ -- the difference has to be
    # the ANSWER.  A reader must not have to know that an absent key means
    # "never asked" while a present one means "asked and declined".
    assert off.get("streamed_any") is None and auto["streamed_any"] is False


def test_the_control_fires_when_the_recording_is_removed():
    """Break it deliberately: without ``decisions``, the two runs are equal.

    This is the state of the code BEFORE this fix, reproduced exactly by
    declining to pass the out-parameter.  A control that cannot fail is
    worth nothing, so this asserts that the failure is reachable.
    """
    model, machine = _one_grid_model(), _fitting_machine()
    off = streaming.steppers_for_tree(model, OFF, machine=machine)
    auto = streaming.steppers_for_tree(model, StreamingOptions(mode="auto"),
                                       machine=machine)
    assert off == auto == {}
    # ... and with nothing recorded the receipt collapses to the same value
    # for both, which is the ambiguity the fix removes.
    assert streaming.streaming_receipt(OFF, {}) == \
        streaming.streaming_receipt(StreamingOptions(mode="auto"), {}) == {}


def test_a_receipt_for_a_domain_that_did_stream_says_so_and_names_the_tiling():
    model = _one_grid_model()
    options = StreamingOptions(mode="on", tile_nx=64, tile_ny=64)
    decisions: dict = {}
    with pytest.raises(StreamingRefused):
        # No builder here; the decision is still recorded BEFORE the refusal,
        # which is what lets a failed run say what it was trying to do.
        streaming.steppers_for_tree(model, options, decisions=decisions)
    receipt = streaming.streaming_receipt(options, decisions)
    assert receipt["streamed_any"] is True
    # The tile count and the redundancy ride beside the tiling (measured
    # 2026-09-26): the two numbers a streamed step's pace follows.
    cfg = _cfg()
    halo = 10 + 3 * cfg.time_step_sound // 2
    ntiles, redundancy = streaming.tiling_shape(cfg, 64, 64, halo)
    assert receipt["domains"]["1"] == {
        "streamed": True, "reason": "[tiles] pins the tiling",
        "tile_nx": 64, "tile_ny": 64, "nbuffers": 2,
        "halo": halo, "store": "host",
        "ntiles": ntiles, "redundancy": round(redundancy, 4)}


@pytest.mark.parametrize("module", [
    "woof.prepared_single_domain_forecast",
    "woof.prepared_domain_tree_forecast"])
def test_both_production_routes_wire_the_builder_AND_record_the_decision(
        module):
    """The regression test for the defect itself, at the ROUTE.

    ``tests`` above prove the seam behaves; this proves the two functions a
    user can actually launch a forecast through USE it.  Both routes once
    called ``steppers_for_tree(model, exp.tiles)`` bare, so every
    ``mode = "on"`` configuration raised ``StreamingRefused`` and no
    forecast the CLI could launch was capable of streaming.  Asserted on the
    source because the alternative is a full prepared-cache forecast, and a
    regression test that needs a GPU and a data bundle is one nobody runs.
    """
    import importlib
    import inspect

    src = inspect.getsource(importlib.import_module(module))
    assert "builders=streaming.builders_for_tree(" in src, \
        "route calls steppers_for_tree with no builder -- it will REFUSE"
    assert "decisions=streaming_decisions" in src, \
        "route does not record which way the decision went"
    assert "streaming.streaming_receipt(" in src
    assert 'report["tiles"] = streaming_report' in src


def test_streaming_without_a_builder_refuses_rather_than_running_resident():
    """The dangerous direction is silently NOT streaming.

    A route that configured streaming and got a resident run would die at
    the allocation the mode existed to avoid, having reported nothing.
    """
    with pytest.raises(StreamingRefused, match="wired no streamed-domain"):
        make_stepper(object(), _cfg(),
                     StreamingOptions(mode="on", tile_nx=64, tile_ny=64))


def test_a_streamed_domain_refuses_a_state_it_was_not_attached_to():
    state, other = object(), object()
    run = _FakeRun(_cfg())
    streamed = StreamedDomain(run, streaming.StreamingDecision(True, "test"),
                              state=state)
    streamed(state, run.cfg, refl_10cm_due=False)
    assert run.sweeps == [(1, {"refl_10cm_due": False})]
    with pytest.raises(StreamingRefused, match="not the one it was attached"):
        streamed(other, run.cfg)


def test_the_step_kwargs_reach_the_sweep():
    """``refl_10cm_due`` is not optional decoration; it reaches every tile."""
    run = _FakeRun(_cfg())
    run.store[streaming.REFL_STORE_KEY] = object()
    streamed = StreamedDomain(run, streaming.StreamingDecision(True, "test"))
    streamed(None, run.cfg, refl_10cm_due=False, health_debug=True)
    assert run.sweeps == [(1, {"refl_10cm_due": False,
                               "health_debug": True})]
    assert streamed.steps == 1


def test_the_kwargs_reach_the_TILE_STEP_and_not_merely_the_sweep():
    """The gap this test used to stop one hop short of.

    ``TiledRun._sweep`` copied the caller's ``step_kwargs`` into a local
    dict and then called ``step(tile, tile_cfg)`` with none of them, so a
    keyword WOOF's loop threaded into a STEP was accepted by ``sweep``,
    documented as forwarded, and dropped at the last hop.  A fake run
    cannot see that -- it never reaches a tile -- so this reads the
    transport's own source.
    """
    import inspect

    from tilestream import driver

    src = inspect.getsource(driver.TiledRun)
    assert "step(tiles[b], tile_cfg, **step_kwargs)" in src, (
        "the tile step no longer forwards step_kwargs; every keyword the "
        "model threads into a STEP is silently dropped under tiling")


def test_a_due_frame_refuses_a_store_with_nowhere_to_put_the_reflectivity():
    """The tiles would each compute their own window and lose all of them.

    ``refl_10cm`` is a REBUILT scratch slot rather than a carrier, so a
    streamed run that omits it integrates the identical forecast -- which is
    exactly why the omission has to be caught at the frame and not at the
    trajectory.  Publishing anyway would put the LAST tile's window over the
    whole domain, right over one tile of N and plausible everywhere.
    """
    run = _FakeRun(_cfg())
    streamed = StreamedDomain(run, streaming.StreamingDecision(True, "test"))
    with pytest.raises(StreamingRefused, match="scratch/refl_10cm"):
        streamed(None, run.cfg, refl_10cm_due=True)
    assert run.sweeps == []


# --------------------------------------------------------------------------
# per-tile lateral boundary windowing
# --------------------------------------------------------------------------

def test_owned_edges_follows_the_window_not_the_interior():
    """Under periodic=False plan_tiles CLAMPS, so the WINDOW decides.

    Both facts matter and they are different tiles' business: the interior
    decides what is scattered back, the window decides which cells the
    boundary kernel writes.
    """
    from tilestream import spec as tspec

    specs = tspec.plan_tiles(128, 96, 32, 32, 16, False)
    census = {0: 0, 1: 0, 2: 0}
    for s in specs:
        owned = streaming.owned_edges(s)
        census[sum(owned.values())] += 1
        assert owned["west"] == (s.ci0 == 0)
        assert owned["east"] == (s.ci0 + s.cnx == s.nx)
    # A 4x3 tiling has four corner tiles (two true edges), six edge tiles
    # (one) and two in the middle with NO true edge at all -- so the plan
    # contains tiles of every kind, which is what makes the seam question
    # meaningful rather than hypothetical.
    assert census == {0: 2, 1: 6, 2: 4}


def test_windowed_side_tables_carry_the_fields_own_staggering():
    """u's south/north tables are cnx+1 wide; v's west/east are cny+1 tall."""
    from tilestream import spec as tspec

    bnd = _boundaries(nz=3, ny=32, nx=48, width=5)
    specs = tspec.plan_tiles(48, 32, 16, 16, 4, False)
    tables = streaming.tile_boundary_tables(bnd, specs)
    for spec_, tbl in zip(specs, tables):
        iv = tbl.intervals[0]
        assert iv.fields["u"].south.value.shape == (3, 5, spec_.cnx + 1)
        assert iv.fields["u"].west.value.shape == (3, spec_.cny, 5)
        assert iv.fields["v"].west.value.shape == (3, spec_.cny + 1, 5)
        assert iv.fields["v"].south.value.shape == (3, 5, spec_.cnx)
        assert iv.fields["mu"].north.value.shape == (3, 5, spec_.cnx)


def test_true_edges_keep_the_domains_values_and_seams_are_inert():
    """The whole mode rests on this split, so it is asserted element-wise."""
    import numpy as np
    from tilestream import spec as tspec

    bnd = _boundaries(nz=2, ny=32, nx=48, width=5)
    specs = tspec.plan_tiles(48, 32, 16, 16, 4, False)
    tables = streaming.tile_boundary_tables(bnd, specs, seam="zeros")
    saw_edge = saw_seam = False
    for spec_, tbl in zip(specs, tables):
        owned = streaming.owned_edges(spec_)
        side = tbl.intervals[0].fields["mu"].west
        if owned["west"]:
            want = np.asarray(
                bnd.intervals[0].fields["mu"].west.value)[
                    :, spec_.cj0:spec_.cj0 + spec_.cny, :]
            assert np.array_equal(side.value, want)
            assert np.any(side.value)
            saw_edge = True
        else:
            assert not np.any(side.value)
            assert not np.any(side.tendency)
            saw_seam = True
    assert saw_edge and saw_seam


def test_the_three_seam_fillings_are_genuinely_different_numbers():
    """Zeros, self-consistent and garbage must not accidentally coincide.

    The claim proven on the GPU is that three COMPLETELY DIFFERENT interior
    seam fillings give the identical answer.  That claim is worthless if the
    three fillings are the same numbers, so the difference is asserted here
    where it is cheap.
    """
    import numpy as np
    from tilestream import spec as tspec

    bnd = _boundaries(nz=2, ny=96, nx=128, width=5)
    specs = tspec.plan_tiles(128, 96, 32, 32, 4, False)
    snapshot = {name: np.full((2, 96 + extra_y, 128 + extra_x), 3.5)
                for name, (extra_y, extra_x) in
                (("mu", (0, 0)), ("u", (0, 1)), ("v", (1, 0)))}
    interior = next(i for i, s in enumerate(specs)
                    if not any(streaming.owned_edges(s).values()))
    got = {}
    for seam in ("zeros", "self", "poison"):
        tables = streaming.tile_boundary_tables(bnd, specs, seam=seam,
                                                snapshot=snapshot)
        got[seam] = np.asarray(
            tables[interior].intervals[0].fields["mu"].west.value)
    assert not np.any(got["zeros"])
    assert np.allclose(got["self"], 3.5)
    assert np.abs(got["poison"]).max() > 1.0e5
    for a, b in (("zeros", "self"), ("self", "poison"),
                 ("zeros", "poison")):
        assert not np.array_equal(got[a], got[b])


def test_seam_self_without_a_snapshot_refuses():
    from tilestream import spec as tspec

    bnd = _boundaries(nz=2, ny=32, nx=48, width=5)
    specs = tspec.plan_tiles(48, 32, 16, 16, 4, False)
    with pytest.raises(ValueError, match="needs the domain snapshot"):
        streaming.tile_boundary_tables(bnd, specs, seam="self")


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

class _FakeRun:
    """Enough of :class:`tilestream.driver.TiledRun` to test the seam."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.store = {}
        self.sweeps = []

    def sweep(self, nsteps=1, *, step_kwargs=None, report=None,
              progress=None, live_config=None, physics_control=None):
        self.live_config = live_config
        self.physics_control = physics_control
        self.sweeps.append((nsteps, dict(step_kwargs or {})))
        if report is not None:
            report.update(steps=nsteps)


def _cfg():
    from woof.config import RunConfig

    return RunConfig(nx=256, ny=192, nz=49, dx=12000.0, dy=12000.0,
                     ztop=20000.0, dt=60.0, run_seconds=600.0)


def _boundaries(*, nz, ny, nx, width):
    """A domain ``LateralBoundaries`` with distinguishable, nonzero tables."""
    import numpy as np
    from woof.ingest.lateral_bc import (BoundaryInterval, FieldBoundary,
                                         LateralBoundaries, SideBoundary)

    def side(shape, base):
        value = base + np.arange(int(np.prod(shape)),
                                 dtype=np.float64).reshape(shape)
        return SideBoundary(value, 0.001 * value)

    fields = {}
    for name, (ey, ex) in (("mu", (0, 0)), ("u", (0, 1)), ("v", (1, 0))):
        fields[name] = FieldBoundary(
            west=side((nz, ny + ey, width), 100.0),
            east=side((nz, ny + ey, width), 200.0),
            south=side((nz, width, nx + ex), 300.0),
            north=side((nz, width, nx + ex), 400.0))
    return LateralBoundaries((BoundaryInterval(0.0, 21600.0, fields),),
                             width, 1, 4)


def _raw_experiment() -> dict:
    """A minimal but VALID experiment table, modelled on configs/.

    Deliberately built here rather than loaded from ``configs/``: the shipped
    configs name real forcing files, and a schema test that cannot run
    without a 2 GB GRIB is a schema test nobody runs.
    """
    return {
        "experiment": {"name": "streaming-test",
                       "start_time": datetime(2026, 8, 9, 0, 0, 0),
                       "run_seconds": 600.0, "feedback": 0,
                       "smooth_option": 0, "blend_width": 5,
                       "spec_bdy_width": 5, "restart_interval_s": 0.0},
        "projection": {"map_proj": "lambert", "ref_lat": 35.0,
                       "ref_lon": -97.0, "truelat1": 30.0,
                       "truelat2": 60.0, "stand_lon": -97.0},
        "shared": {
            "nz": 4, "ztop": 20000.0, "p_top": 10000.0,
            "eta_levels": [1.0, 0.8, 0.55, 0.28, 0.0],
            "hybrid_opt": 2, "etac": 0.2, "base_temp": 290.0,
            "time_step_sound": 4, "terrain_opt": 1, "map_proj": 1,
        },
        "domain": [{
            "grid_id": 1, "parent_id": 0, "i_parent_start": 1,
            "j_parent_start": 1, "parent_grid_ratio": 1,
            "parent_time_step_ratio": 1, "nx": 64, "ny": 64,
            "time_step": 60, "dx": 12000.0, "specified": True,
            "nested": False, "history_interval_s": 300.0,
        }],
    }


def _experiment():
    from woof.experiment import build_experiment

    return build_experiment(_raw_experiment(), source="<test>")


# --------------------------------------------------------------------------
# the ROUTE builders -- the half of the seam that did not exist
# --------------------------------------------------------------------------
#
# Until these, both production routes called steppers_for_tree with no
# builders and every [tiles] mode = "on" forecast raised
# StreamingRefused.  The expensive half of the proof -- that the shipped
# builder is bit-exact through execute_experiment -- is on a GPU in
# tilestream/test_route.py.  What is proven here is that the routes REACH
# it, which is the part that rotted silently.

def test_both_production_routes_pass_builders():
    """The regression that produced this whole item, held as source.

    A source assertion rather than a behavioural one because the defect was
    an ABSENT argument: nothing about the routes' behaviour differed, they
    simply refused every streaming configuration they were given, and a
    route that stops passing builders would go back to refusing without any
    other test noticing.
    """
    import inspect

    import woof.prepared_domain_tree_forecast as tree
    import woof.prepared_single_domain_forecast as single

    for module in (single, tree):
        src = inspect.getsource(module)
        calls = [line for line in src.splitlines()
                 if "steppers_for_tree(" in line and "def " not in line]
        assert calls, f"{module.__name__} no longer calls steppers_for_tree"
        assert "builders=streaming.builders_for_tree(" in src, (
            f"{module.__name__} calls steppers_for_tree without wiring "
            "builders; every [tiles] mode='on' forecast on that route "
            "will raise StreamingRefused")


def test_builders_for_tree_is_empty_and_inert_when_streaming_is_off():
    """The OFF contract again: absent [tiles] must import nothing."""
    import sys

    from test_model import _model                      # noqa: PLC0415

    _exp, model = _model()
    saved = {name: sys.modules.get(name)
             for name in ("tilestream.driver", "tilestream.autoplan")}
    try:
        for name in saved:
            sys.modules[name] = None
        assert streaming.builders_for_tree(model, OFF) == {}
        assert streaming.builders_for_tree(model, None) == {}
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_builders_for_tree_covers_every_grid():
    from test_model import _model                      # noqa: PLC0415

    _exp, model = _model()
    builders = streaming.builders_for_tree(
        model, StreamingOptions(mode="on", tile_nx=32, tile_ny=32))
    assert set(builders) == {1, 2}
    assert all(callable(b) for b in builders.values())


def test_a_nest_that_fires_builds_its_own_forcing_door():
    """A nest streams through the nest tile hook, not a windowed series.

    The categorical "is a NEST" refusal that stood here is gone: the
    streamed-child corridor windows the rolling nest tables per tile
    (woof.core.nest_stream), so the builder wires that hook instead of
    refusing.  What remains refused is the genuine contradiction -- a nest
    that ALSO carries a tabulated LateralBoundaries series, two forcing
    mechanisms for one domain.  Both directions are held here: the
    contradiction fires by name, and the bare nest gets PAST the nest gate
    (this stand-in tree then fails at the vertical-coordinate check, which
    is a statement about the fixture's state, not about nests).
    """
    from test_model import _model                      # noqa: PLC0415

    from woof.ingest.lateral_bc import (BoundaryInterval,
                                         LateralBoundaries)

    _exp, model = _model()
    builders = streaming.builders_for_tree(
        model, StreamingOptions(mode="on", tile_nx=32, tile_ny=32))
    child = model.node(2)
    decision = streaming.StreamingDecision(True, "test", 32, 32, 2, 16)

    child.state.lateral_boundaries = LateralBoundaries(
        (BoundaryInterval(0.0, 3600.0, {}),))
    with pytest.raises(StreamingRefused, match="also.*carries a tabulated|"
                                               "contradict"):
        builders[2](child.state, child.cfg.run, decision)

    child.state.lateral_boundaries = None
    try:
        builders[2](child.state, child.cfg.run, decision)
    except StreamingRefused as err:
        assert "NEST" not in str(err), (
            "the categorical nest refusal is back; the streamed-child "
            "corridor is unreachable through the production builder")
        assert "znw" in str(err) or "vertical" in str(err)


# --------------------------------------------------------------------------
# Stationary nesting honors tree-wide and per-domain streaming choices.
# Device/host budgets and actual unsupported operations remain validated.

def _raw_nested_experiment() -> dict:
    """:func:`_raw_experiment` plus one child, valid at every guard.

    The placement is not arbitrary: i/j_parent_start = 11 is the first
    origin that clears spec_bdy_width + blend_width = 10 parent rows, and
    nx = ny = 39 is a multiple of parent_grid_ratio = 3 (WPS requires
    e_we = n * ratio + 1).  Both were found by running the loader, not
    guessed -- a fixture that fails an unrelated guard would test that
    guard instead of this one.
    """
    raw = _raw_experiment()
    raw["domain"] = list(raw["domain"]) + [{
        "grid_id": 2, "parent_id": 1, "i_parent_start": 11,
        "j_parent_start": 11, "parent_grid_ratio": 3,
        "parent_time_step_ratio": 3, "nx": 39, "ny": 39,
        "time_step": 20, "dx": 4000.0, "specified": False,
        "nested": True, "history_interval_s": 300.0,
    }]
    return raw


def _build(raw):
    from woof.experiment import build_experiment

    return build_experiment(raw, source="<test>")


def test_a_nested_config_honors_both_streamed_endpoints():
    raw = _raw_nested_experiment()
    raw["tiles"] = {"mode": "on"}
    exp = _build(raw)
    assert [streaming.options_for_domain(dc, exp.tiles).mode for dc in exp.domains] == ["on", "on"]


def test_a_nested_config_preserves_the_requested_pinned_tiling():
    raw = _raw_nested_experiment()
    raw["tiles"] = {"mode": "on", "tile_nx": 32, "tile_ny": 32}
    exp = _build(raw)
    for dc in exp.domains:
        options = streaming.options_for_domain(dc, exp.tiles)
        assert (options.mode, options.tile_nx, options.tile_ny) == ("on", 32, 32)


def test_the_moving_domain_is_named_because_a_domain_that_moves_is_a_nest():
    """A [relocation] reader is thinking about the follow domain.

    Without this clause the refusal quotes a parent_id they never typed,
    and reads as being about some other grid.
    """
    raw = _raw_nested_experiment()
    raw["tiles"] = {"mode": "on", "store": "device"}
    raw["relocation"] = {
        "enabled": True, "grid_id": 2,
        "move": [{"at_seconds": 120.0, "di_parent_cells": 1,
                  "dj_parent_cells": 0}]}
    with pytest.raises(StreamingRefused) as refusal:
        _build(raw)
    text = str(refusal.value)
    assert "[relocation] follow domain" in text
    assert "d02" in text and "host store" in text


#: A valid storm-tracking table, the shape ``[[domain]].follow`` wraps.
_DOMAIN_FOLLOW = dict(field="pressure", threshold=30.0, level_hpa=850.0,
                      search_margin_cells=10, min_shift_cells=1,
                      max_shift_cells=4, cooldown_seconds=0.0)


@pytest.mark.xfail(
    strict=True,
    reason="HANDED BACK: the moving-root set is the ONE union "
           "woof.static.corridor.moving_grid_ids already answers -- the "
           "name woof/runplan.py:1041 imports, and the name "
           "woof/core/streamed_relocation.py mark_reconstruction_nodes "
           "now reads instead of recomputing the union inline.  The "
           "refusal that has to read it lives at "
           "woof/core/streaming.py:4462-4483 refuse_streamed_nests, "
           "which is outside this lane's boundary.  Three edits are owed "
           "there.  Drop the exp.relocation-only early returns at "
           ":4464-4469 and iterate moving_grid_ids(exp) instead of "
           "the single int(relocation.grid_id) at :4470, so a per-domain "
           "[[domain]].follow mover is gated too.  Widen the mode test at "
           ":4477 to include 'auto', but ONLY for a store = 'device' the "
           "[[domain]] row declares itself: auto never selects a device "
           "store, so an explicit one under auto is a contradiction, "
           "while a domain with no tiles table of its own inherits the "
           "tree-wide table whole (options_for_domain REPLACES rather "
           "than merges, streaming.py:4268-4272/:4280-4281), and "
           "inheriting a device store from the tree is not a declaration "
           "about this domain.  The test is available today as "
           "getattr(domain_cfg, 'tiles', None) is not None; no new field "
           "on StreamingOptions is needed.  Replace the message at "
           ":4479-4483 with one naming the breakage and both remedies.  "
           "STRICT: when that lands this fails loudly and the marker must "
           "be deleted.")
def test_device_store_follower_is_refused_at_config_load():
    """A per-domain follower with a device store dies at the first move.

    The move captures the OUTGOING child's arrays and closes its tile
    owner; the reconstruction road publishes a pinned HOST store, so a
    device store frees the donor the overlap transplant still has to
    read, and keeping it alive holds two whole domain stores on the card.
    A device store saves no VRAM, so neither remedy costs capability.
    Today this configuration loads and crashes mid-run instead.
    """
    raw = _raw_nested_experiment()
    raw["domain"][1]["follow"] = dict(_DOMAIN_FOLLOW, cadence_seconds=600.0)
    raw["domain"][1]["tiles"] = {"mode": "on", "store": "device"}
    with pytest.raises(StreamingRefused) as refusal:
        _build(raw)
    text = str(refusal.value)
    assert "d02" in text
    assert "tiles.store = 'host'" in text
    assert "tiles.mode = 'off'" in text


def test_bounds_only_relocation_does_not_refuse_a_stationary_streamed_child():
    raw = _raw_nested_experiment()
    raw["tiles"] = {"mode": "on"}
    raw["relocation"] = {"enabled": True, "grid_id": 2}
    exp = _build(raw)
    assert exp.tiles.mode == "on"
    assert not exp.relocation.moves and exp.relocation.follow is None


def test_a_single_domain_config_still_streams_on_demand():
    """The negative control for requirement 1, and it is not optional.

    A refusal keyed on ``mode == "on"`` alone would pass every assertion
    above while breaking the mode entirely.  One [[domain]] table is the
    shape [tiles] exists for.
    """
    raw = _raw_experiment()
    raw["tiles"] = {"mode": "on", "tile_nx": 64, "tile_ny": 64}
    exp = _build(raw)
    assert exp.tiles.mode == "on"
    assert exp.tiles.tile_nx == 64


def test_a_nested_config_on_auto_loads_rather_than_being_refused():
    """The property requirement 2 DEPENDS on, held explicitly.

    ``auto`` over a tree is the only expressible form of "stream d01, keep
    the nests resident" -- [tiles] is a tree-wide block with no per-domain
    key -- so refusing it for having nests at all would remove the shape a
    storm-following run wants.
    """
    raw = _raw_nested_experiment()
    raw["tiles"] = {"mode": "auto"}
    exp = _build(raw)
    assert exp.tiles.mode == "auto"
    assert len(exp.domains) == 2


def test_a_nested_config_with_no_tiles_table_is_untouched():
    """The emptiness contract: a tree that never mentions [tiles] is OFF."""
    exp = _build(_raw_nested_experiment())
    assert exp.tiles is OFF
    assert all(dc.tiles is None for dc in exp.domains)


# ---- the per-domain surface: saying which end streams --------------------

def test_a_nest_that_opts_out_makes_a_streamed_parent_expressible():
    """'Stream d01, keep d02 resident' -- said, not inferred.

    ``mode = "on"`` over a tree used to be refused unconditionally, and
    the refusal's own words were that the shape "is not expressible as
    mode = 'on'" because ``[tiles]`` was a tree-wide block.  It is
    expressible now: the nest carries its own table and says OFF.  What is
    asserted here is the SURFACE -- per-domain tables resolve to per-domain
    modes -- not a refusal; either or both ends of a coupling edge may
    stream, and nothing in the load path refuses a tree for streaming.
    """
    raw = _raw_nested_experiment()
    raw["tiles"] = {"mode": "on"}
    raw["domain"][1]["tiles"] = {"mode": "off"}
    exp = _build(raw)
    assert exp.tiles.mode == "on"
    assert exp.domains[0].tiles is None          # inherits the tree's 'on'
    assert exp.domains[1].tiles.mode == "off"
    assert streaming.options_for_domain(exp.domains[0], exp.tiles).mode == "on"
    assert streaming.options_for_domain(exp.domains[1], exp.tiles).mode == "off"


def test_the_inverse_shape_is_expressible_too_with_no_tree_wide_table():
    """A streamed CHILD under a resident parent, with [tiles] absent.

    The per-domain table is a complete surface, not a modifier: a config
    with no tree-wide ``[tiles]`` at all still streams the one domain that
    asks to.  The tree entry points must notice, which is what
    ``tree_streams_anywhere`` exists for -- reading only the tree-wide
    table here would run the domain resident in silence.
    """
    import types as _types

    raw = _raw_nested_experiment()
    raw["domain"][1]["tiles"] = {"mode": "on", "tile_nx": 13, "tile_ny": 13}
    exp = _build(raw)
    assert exp.tiles is OFF
    assert streaming.options_for_domain(exp.domains[0], exp.tiles) is OFF
    assert streaming.options_for_domain(exp.domains[1], exp.tiles).mode == "on"

    nodes = [_types.SimpleNamespace(cfg=dc, state=object(), parent=None)
             for dc in exp.domains]
    nodes[1].parent = nodes[0]
    model = _types.SimpleNamespace(walk_parent_first=lambda: nodes)
    assert streaming.tree_streams_anywhere(model, exp.tiles) is True


def _decisions_for(exp):
    """``{grid_id: StreamingDecision}``, the way ``steppers_for_tree`` builds it.

    Only the two steps the RECEIPT reads are mirrored: the per-domain
    options and the decision they produce, plus the ``configured_mode``
    stamp the walk leaves on each decision's ``detail``.  The walk's other
    half -- the budget arithmetic -- prices every domain through
    ``tilestream.autoplan``, which imports ``woof.core.physics`` and so
    needs cupy, and the receipt does not read a single one of those
    numbers.  Mirroring the whole walk here would put this test behind a
    card for no coverage; ``tests/test_streaming_clock_arming.py`` drives
    the real walk on one.

    Pinned tilings throughout (``tile_nx``/``tile_ny`` given), so ``decide``
    consults no planner and probes no card.
    """
    out = {}
    for domain_cfg in exp.domains:
        options = streaming.options_for_domain(domain_cfg, exp.tiles)
        decision = streaming.decide(domain_cfg.run, options)
        decision.detail.update(configured_mode=options.mode)
        out[int(domain_cfg.grid_id)] = decision
    return out


def test_the_receipt_names_a_grid_that_only_a_per_domain_table_streamed():
    """Tree-wide ``off`` plus a per-domain ``on`` STREAMS, so it must say so.

    The receipt was keyed on the TREE-WIDE ``[tiles]`` table: ``options``
    not enabled meant an empty dict, whatever the domains had decided.  So
    a tree whose tree-wide mode is ``off`` and whose child carries ``tiles
    = {mode = "on"}`` streamed IN SILENCE -- the run tiled a grid and the
    operator got no line naming which one.

    That is exactly the class ``tree_streams_anywhere`` exists to close: a
    per-domain road the tree-wide table cannot see.  The remedy is the
    same one, applied to the receipt -- read the DECISIONS, which carry
    what actually happened, and treat the tree-wide table as the default
    it is rather than the answer.
    """
    raw = _raw_nested_experiment()
    raw["domain"][1]["tiles"] = {"mode": "on", "tile_nx": 13, "tile_ny": 13}
    exp = _build(raw)
    assert exp.tiles is OFF                    # nothing tree-wide says stream

    decisions = _decisions_for(exp)
    assert decisions[1].stream is False
    assert decisions[2].stream is True         # ... and yet a grid streams

    receipt = streaming.streaming_receipt(exp.tiles, decisions)
    assert receipt != {}, \
        "a grid streamed and the receipt was empty -- silent streaming"
    assert receipt["streamed_any"] is True
    assert receipt["domains"]["2"]["streamed"] is True
    assert receipt["domains"]["1"]["streamed"] is False
    # The per-domain entry keeps saying which table decided it, because the
    # tree-wide mode in `configured_mode` disagrees with what d02 did.
    assert receipt["domains"]["2"]["configured_mode"] == "on"
    # THE LINE THE OPERATOR READS.  It has to NAME the streamed grid; a
    # summary that said only "mode='off'" would be the silence again.
    assert "[2] streamed" in receipt["summary"], receipt["summary"]


def test_the_per_domain_receipt_says_the_tree_wide_mode_was_overridden():
    """The summary must not read as a plain contradiction.

    ``mode='off'`` next to ``grid(s) [2] streamed`` is accurate and, on its
    own, looks like a bug in the receipt rather than a per-domain table
    doing its job.  So the line says which grids overrode the tree-wide
    default.  This text is NEW -- the shape printed nothing at all before
    -- so pinning it moves no shipped byte.
    """
    raw = _raw_nested_experiment()
    raw["domain"][1]["tiles"] = {"mode": "on", "tile_nx": 13, "tile_ny": 13}
    exp = _build(raw)
    receipt = streaming.streaming_receipt(exp.tiles, _decisions_for(exp))
    assert receipt["summary"] == (
        "[tiles] mode='off' tree-wide, overridden per domain "
        "(d02 mode='on'): grid(s) [2] streamed (d02 tile 13x13 + halo 16, "
        "9 tiles at 11.98x redundancy), [1] ran resident")


def test_the_per_domain_receipt_fix_moves_no_shipped_receipt_byte():
    """A tree the tree-wide table already governed keeps its exact line.

    Leg A of the release gate -- a streamed parent over a resident child,
    named tree-wide -- prints this line today, and a receipt an operator
    has learned to read is a shipped interface.  Pinned literally, not by
    content: the whole risk of re-keying the receipt is that it rewrites
    the receipts that were already right.

    The line moved ONCE, deliberately, and only by addition: every
    streamed grid now carries its tiling, tile count and redundancy, and
    the receipt entry the two numbers.  A line that named the streamed
    grid and nothing else printed a 1,190-tile sweep at 49.95x -- 237-547 s
    per step -- exactly as it printed an ordinary streamed run (measured
    2026-09-26).  Everything the line said before it still says, in
    the same words and order.
    """
    raw = _raw_nested_experiment()
    raw["tiles"] = {"mode": "on", "tile_nx": 13, "tile_ny": 13}
    raw["domain"][1]["tiles"] = {"mode": "off"}
    exp = _build(raw)
    receipt = streaming.streaming_receipt(exp.tiles, _decisions_for(exp))
    assert receipt["summary"] == (
        "[tiles] mode='on': grid(s) [1] streamed (d01 tile 13x13 + halo 16, "
        "25 tiles at 12.36x redundancy), [2] ran resident")
    assert receipt["configured_mode"] == "on"
    assert receipt["streamed_any"] is True
    assert receipt["domains"] == {
        "1": {"streamed": True, "reason": "[tiles] pins the tiling",
              "tile_nx": 13, "tile_ny": 13, "nbuffers": 2, "halo": 16,
              "store": "host", "ntiles": 25, "redundancy": 12.3596},
        "2": {"streamed": False, "reason": "[tiles] mode = 'off'",
              "configured_mode": "off"}}


def test_an_unconfigured_run_still_contributes_no_receipt_at_all():
    """The emptiness that keeps pre-``[tiles]`` receipts byte-identical.

    Re-keying on the decisions must not turn "nobody asked" into a line.
    A tree with no ``[tiles]`` anywhere never fills a decisions mapping --
    both tree entry points return early on ``tree_streams_anywhere`` -- so
    the receipt stays empty, and that is asserted here rather than assumed
    because it is the one property the fix could plausibly break.
    """
    raw = _raw_nested_experiment()
    exp = _build(raw)
    assert exp.tiles is OFF
    assert streaming.streaming_receipt(exp.tiles, {}) == {}
    assert streaming.streaming_receipt(None, {}) == {}


def test_both_ends_explicitly_on_preserves_per_domain_controls():
    raw = _raw_nested_experiment()
    raw["domain"][0]["tiles"] = {"mode": "on", "tile_nx": 32, "tile_ny": 32}
    raw["domain"][1]["tiles"] = {"mode": "on", "tile_nx": 16, "tile_ny": 16}
    exp = _build(raw)
    assert exp.tiles is OFF
    assert [(streaming.options_for_domain(dc, exp.tiles).mode,
             streaming.options_for_domain(dc, exp.tiles).tile_nx) for dc in exp.domains] == [("on", 32), ("on", 16)]


def test_a_per_domain_table_may_not_name_the_card():
    """The budget names a CARD; a per-domain copy is a second answer."""
    raw = _raw_nested_experiment()
    raw["domain"][1]["tiles"] = {"mode": "auto",
                                 "vram_budget_bytes": 1 << 30}
    with pytest.raises(ValueError) as refusal:
        _build(raw)
    text = str(refusal.value)
    assert "vram_budget_bytes" in text and "name the CARD" in text


def test_a_per_domain_table_meets_the_same_key_and_value_guards():
    """One parser, one vocabulary: the tree-wide refusals apply per domain."""
    raw = _raw_nested_experiment()
    raw["domain"][1]["tiles"] = {"mdoe": "auto"}
    with pytest.raises(ValueError) as refusal:
        _build(raw)
    assert "unknown key" in str(refusal.value)

    raw = _raw_nested_experiment()
    raw["domain"][1]["tiles"] = "auto"
    with pytest.raises(ValueError) as refusal:
        _build(raw)
    assert "inline TABLE" in str(refusal.value)


def test_a_per_domain_road_binds_nothing_in_the_restart_identity():
    """[tiles] changes no bytes, so it may not change the identity either.

    The tree-wide table has always been excluded; the per-domain one has
    to be excluded on the same law, and UNCONDITIONALLY -- unlike ``spawn``
    beside it, which binds when declared.  A domain that streamed must
    resume resident and a domain that outgrew its card must resume
    streamed, which is the whole reason to stream at all.
    """
    from woof.core.model import restart_identity_payload

    plain = _build(_raw_nested_experiment())
    raw = _raw_nested_experiment()
    raw["tiles"] = {"mode": "on"}
    raw["domain"][1]["tiles"] = {"mode": "off"}
    tiled = _build(raw)

    assert restart_identity_payload(plain) == restart_identity_payload(tiled)
    assert all("tiles" not in d
               for d in restart_identity_payload(tiled)["domains"])


# ---- and the seam, for the mode that is NOT refused ----------------------

def _tree_nodes():
    """A parent and its child, in the only shape the seam reads.

    Deliberately not ``test_model._model()``: that builds real
    ``DomainState`` objects and imports the dycore, which needs a card.
    The NODES are namespaces, because the walk reads only ``node.parent``,
    ``node.cfg`` and ``node.state`` off them -- but ``node.cfg`` is a real
    :class:`~woof.experiment.DomainConfig` and not a namespace, because
    the walk's surface WIDENED when roads became per-domain: pricing a
    child's coupling corridor reaches ``nest_slot_shapes``, which reads
    ``parent_grid_ratio`` and the placement off the domain config.  A
    namespace stub would answer that question by AttributeError, which is
    the fixture lying about the seam rather than the seam being wrong.
    None of it needs a card.
    """
    import types

    from woof.config import RunConfig
    from woof.experiment import DomainConfig

    parent_cfg = DomainConfig(
        grid_id=1, parent_id=0, i_parent_start=1, j_parent_start=1,
        parent_grid_ratio=1, parent_time_step_ratio=1,
        history_interval_s=3600.0, run=_cfg())
    parent = types.SimpleNamespace(cfg=parent_cfg, state=object(),
                                   parent=None)
    child_run = RunConfig(nx=192, ny=192, nz=49, dx=4000.0, dy=4000.0,
                          ztop=20000.0, dt=20.0, run_seconds=600.0,
                          grid_id=2, nested=True, specified=False)
    child_cfg = DomainConfig(
        grid_id=2, parent_id=1, i_parent_start=30, j_parent_start=30,
        parent_grid_ratio=3, parent_time_step_ratio=3,
        history_interval_s=3600.0, run=child_run)
    child = types.SimpleNamespace(cfg=child_cfg, state=object(),
                                  parent=parent)
    return parent, child


def test_auto_consults_the_planner_about_a_child_and_prices_it():
    """The INVERSION the per-domain roads demanded, held explicitly.

    A nest used to short-circuit to RESIDENT under ``auto`` without the
    planner being asked, because the builder could not stream it.  The
    builder streams it now (``woof.core.nest_stream``), so the accurate
    decision is the planner's own, with its arithmetic on the decision --
    a child priced like any domain.  The machine is fabricated, not
    detected, so the test needs no card and no cupy probe.
    """
    from tilestream.autoplan import Machine

    _parent, child = _tree_nodes()
    machine = Machine(vram_bytes=64 << 30, host_bytes=128 << 30)
    decision = streaming.decide(child.cfg.run,
                                StreamingOptions(mode="auto"),
                                machine=machine)
    # 192x192x49 fits a 64 GiB budget comfortably: resident, with the
    # planner's own numbers on the decision -- the short-circuit reported
    # neither.
    assert decision.stream is False
    assert decision.resident_bytes is not None
    assert "NESTED" not in decision.reason


def test_on_over_a_nest_is_not_quietly_turned_into_a_resident_run():
    """``on`` is refused from the config; it is never silently declined.

    A caller that assembled its own ExperimentConfig and skipped the
    loader must still meet the loud refusals downstream (the walk's edge
    check, the coupler's FORCE law).  Silently declining to stream is the
    failure this whole module was written to remove -- the run dies at
    the allocation the mode existed to avoid, with nothing in the log to
    say the mode never engaged.
    """
    _parent, child = _tree_nodes()
    decision = streaming.decide(
        child.cfg.run, StreamingOptions(mode="on", tile_nx=32, tile_ny=32))
    assert decision.stream is True


def test_both_streamed_endpoints_are_priced_before_both_builders(monkeypatch):
    import types
    parent, child = _tree_nodes()
    parent.state = types.SimpleNamespace()
    child.state = types.SimpleNamespace()
    calls = []
    decisions = {}
    def build(state, cfg, decision):
        assert set(decisions) == {1, 2}
        calls.append(cfg.grid_id)
        return _fake_streamed()
    model = types.SimpleNamespace(walk_parent_first=lambda: [parent, child])
    options = StreamingOptions(mode="on", tile_nx=32, tile_ny=32)
    steppers = streaming.steppers_for_tree(
        model, options, builders={1: build, 2: build}, decisions=decisions)
    assert set(steppers) == {1, 2}
    assert calls == [1, 2]
    assert decisions[1].stream and decisions[2].stream
    assert decisions[2].detail["corridor_claim_bytes"] > 0
    assert all(decision.detail["claim_bytes"] > 0 for decision in decisions.values())


def test_the_child_road_is_admitted_at_the_walk_and_priced():
    """A streamed CHILD under a resident parent is legal -- the walk says so.

    The child alone is walked (its parent resident outside the decision
    set), with a pinned tiling so no planner runs.  The decision must
    stream, carry the child's corridor claim in its detail, and the only
    refusal left standing must be the harness's own missing builder --
    NOT a refusal about the domain being a nest, which is what the
    pre-roads admission raised here.
    """
    import types as _types

    _parent, child = _tree_nodes()
    model = _types.SimpleNamespace(walk_parent_first=lambda: [child])
    options = StreamingOptions(mode="on", tile_nx=32, tile_ny=32)
    decisions: dict = {}
    with pytest.raises(StreamingRefused) as refusal:
        streaming.steppers_for_tree(model, options, decisions=decisions)
    text = str(refusal.value)
    assert "builder" in text and "NEST" not in text
    assert decisions[2].stream is True
    assert decisions[2].detail["road"] == "streamed"
    assert decisions[2].detail["corridor_claim_bytes"] > 0


# ---- the joint decision: a streamed parent may not starve its children ---

def _starving_tree_nodes():
    """A parent too big to sit resident, over a child that comfortably fits.

    1536^2 x 49 dry prices at 8.41 GiB resident, so against a 4 GiB budget
    the parent MUST stream; 192^2 x 49 prices at 0.82 GiB, so the child
    must not.  These are the numbers the starvation was measured on: the
    parent's largest clean tile takes 3.94 of the 4.00 GiB it is shown
    (98.5%), which leaves 0.06 GiB -- an order of magnitude less than the
    child needs.
    """
    import types

    from woof.config import RunConfig
    from woof.experiment import DomainConfig

    parent_run = RunConfig(nx=1536, ny=1536, nz=49, dx=12000.0, dy=12000.0,
                           ztop=20000.0, dt=60.0, run_seconds=600.0)
    parent_cfg = DomainConfig(
        grid_id=1, parent_id=0, i_parent_start=1, j_parent_start=1,
        parent_grid_ratio=1, parent_time_step_ratio=1,
        history_interval_s=3600.0, run=parent_run)
    parent = types.SimpleNamespace(cfg=parent_cfg,
                                   state=types.SimpleNamespace(),
                                   parent=None)
    child_run = RunConfig(nx=192, ny=192, nz=49, dx=4000.0, dy=4000.0,
                          ztop=20000.0, dt=20.0, run_seconds=600.0,
                          grid_id=2, nested=True, specified=False)
    child_cfg = DomainConfig(
        grid_id=2, parent_id=1, i_parent_start=30, j_parent_start=30,
        parent_grid_ratio=3, parent_time_step_ratio=3,
        history_interval_s=3600.0, run=child_run)
    child = types.SimpleNamespace(cfg=child_cfg,
                                  state=types.SimpleNamespace(),
                                  parent=parent)
    return parent, child


def _fake_streamed():
    """A StreamedDomain shell with the one attribute make_stepper reads.

    ``publish_store`` marks the resident state with the streamed store; the
    decision arithmetic under test never touches a device.
    """
    import types

    obj = StreamedDomain.__new__(StreamedDomain)
    obj._run = types.SimpleNamespace(store={})
    return obj


def test_a_streamed_parent_reserves_its_children_before_it_picks_a_tile(
        monkeypatch):
    """STREAMED PARENT + RESIDENT CHILD, the shape that could not be reached.

    The walk already decided the whole tree before building it, but the
    DECISION was parent-greedy: the tile search maximises the compute
    window against whatever budget it is shown, so a streamed root took
    98.5% of the card and the child's decision then met
    ``CannotPlan: no tile fits in 0.06 GiB``.  The engine runs this shape
    bit-identically; only the arithmetic that chose the tile made it
    unreachable.

    The fix is an ordering one: every domain still undecided is priced
    FIRST, and the streamed parent's tile search sees budget-minus-
    children.  A smaller clean tile is a slower parent; a starved child is
    no run at all.
    """
    import sys
    import types as _types

    from tilestream.autoplan import Machine

    gib = 1024 ** 3
    parent, child = _starving_tree_nodes()
    stub = _types.ModuleType("woof.core.dycore")
    stub.step = lambda *a, **kw: None
    monkeypatch.setitem(sys.modules, "woof.core.dycore", stub)

    built: list = []

    def build(state, cfg, decision):
        built.append(int(cfg.nx))
        return _fake_streamed()

    model = _types.SimpleNamespace(
        walk_parent_first=lambda: [parent, child])
    machine = Machine(vram_bytes=4 * gib, host_bytes=256 * gib,
                      vram_headroom=0.0)
    options = StreamingOptions(mode="auto")
    decisions: dict = {}
    streaming.steppers_for_tree(model, options, machine=machine,
                                builders={1: build}, decisions=decisions)

    # The shape itself: parent streamed, child resident.  Before the fix
    # the second of these never happened -- the walk raised instead.
    assert decisions[1].stream is True
    assert decisions[2].stream is False

    # The RESERVATION is on the receipt, not merely in the outcome: the
    # parent's decision has to say how much of the card it was holding
    # back and for whom, or an operator reading a smaller-than-expected
    # tile has no way to tell a reservation from a bad plan.
    reserved = decisions[1].detail["reserved_bytes"]
    assert reserved >= int(decisions[2].detail["claim_bytes"])
    assert decisions[1].detail["reserved_for"] == [2]
    assert (decisions[1].detail["budget_before_reserve_bytes"]
            - reserved == decisions[1].budget_bytes)

    # And the parent's tile is genuinely SMALLER than the greedy one it
    # would have taken: 3.94 GiB was the greedy claim at this shape.
    assert decisions[1].detail["claim_bytes"] < int(3.5 * gib)

    # The whole tree still fits the card it was priced against.
    spent = sum(int(d.detail["claim_bytes"])
                + int(d.detail["corridor_claim_bytes"])
                for d in decisions.values())
    assert spent <= machine.vram_budget_bytes


def test_the_reservation_reaches_the_receipt_an_operator_reads():
    """``streaming_receipt`` carries the reservation, not just the tile.

    The receipt is the artifact a run leaves behind.  A parent that chose
    a smaller tile than the card could hold is indistinguishable from a
    planner bug unless the number it held back, and the grids it held it
    back for, are recorded beside the tile.
    """
    import sys
    import types as _types

    from tilestream.autoplan import Machine

    gib = 1024 ** 3
    parent, child = _starving_tree_nodes()
    stub = _types.ModuleType("woof.core.dycore")
    stub.step = lambda *a, **kw: None
    sys.modules.setdefault("woof.core.dycore", stub)

    model = _types.SimpleNamespace(
        walk_parent_first=lambda: [parent, child])
    machine = Machine(vram_bytes=4 * gib, host_bytes=256 * gib,
                      vram_headroom=0.0)
    decisions: dict = {}
    options = StreamingOptions(mode="auto")
    try:
        streaming.steppers_for_tree(
            model, options, machine=machine,
            builders={1: lambda s, c, d: _fake_streamed()},
            decisions=decisions)
    finally:
        if sys.modules.get("woof.core.dycore") is stub:
            del sys.modules["woof.core.dycore"]

    receipt = streaming.streaming_receipt(options, decisions)
    assert receipt["domains"]["1"]["streamed"] is True
    assert receipt["domains"]["1"]["reserved_bytes"] > 0
    assert receipt["domains"]["1"]["reserved_for"] == [2]
    # The child reserved nothing: it is last in the walk, and a
    # reservation for nobody must be absent rather than zero, so the
    # receipts of trees that reserve nothing are untouched.
    assert "reserved_bytes" not in receipt["domains"]["2"]


def test_a_tree_that_cannot_fit_after_reserving_refuses_with_both_numbers():
    """The accurate refusal: what was left, and what the reservation took.

    Reserving is not a licence to pretend.  When the children's claims
    genuinely leave the parent nothing to tile with, the walk must refuse
    naming BOTH numbers -- the budget it started from and the reservation
    that consumed it -- rather than emit the planner's bare "no tile fits
    in 0.02 GiB", which describes an arithmetic the operator never wrote.
    """
    import types as _types

    from tilestream.autoplan import Machine

    gib = 1024 ** 3
    parent, child = _starving_tree_nodes()
    model = _types.SimpleNamespace(
        walk_parent_first=lambda: [parent, child])
    # One byte below the physical one-buffer floor after the child's
    # mandatory resident/coupling reservation. A slower legal tile is now
    # allowed, so the old0.85GiB efficiency-preference refusal was not a
    # genuine shortage.
    nodes = [parent, child]
    options = StreamingOptions(mode="auto")
    reserve = streaming._tree_reservations(nodes, 256 * gib,
                                           [options, options])[0][0]
    halo = streaming._halo_for(parent.cfg.run, options)
    fp = streaming.radiation_footprint(parent.cfg.run, options)
    minimum = int(fp.marginal_bytes((2 * halo + 1) ** 2 * parent.cfg.run.nz, 1))
    allowance = streaming._tree_process_overhead_bytes(nodes) + reserve + minimum - 1
    machine = Machine(vram_bytes=allowance, host_bytes=256 * gib, vram_headroom=0.0)
    with pytest.raises(StreamingRefused) as refusal:
        streaming.steppers_for_tree(model, StreamingOptions(mode="auto"),
                                    machine=machine, decisions={})
    text = str(refusal.value)
    assert "reserv" in text.lower()
    assert "d01" in text and "d02" in text
    # Both numbers, in the units the operator configured.
    assert f"{allowance / gib:.2f} GiB" in text and f"{reserve / gib:.2f} GiB" in text


# ---- one process, one fixed cost: the tree is not N processes ------------

def _poland_tree_nodes():
    """The three-domain 9/3/1 km ladder MEASURED on node 1, as nodes.

    Real numbers, not a fixture shape: 200x160 at 9 km over 402x300 at 3 km
    over 1050x750 at 1 km, 49 eta levels, RTE+RRTMGP with Thompson, YSU and
    Noah -- the ``full`` rung.  The card was 15.92 GiB (16,303 MiB) with
    15.245 GiB free, which the shipped 8% headroom turns into the
    14.025 GiB budget this tree was planned against.
    """
    import types

    from woof.config import RunConfig
    from woof.experiment import DomainConfig

    physics = dict(ra_sw_physics=4, ra_lw_physics=4, mp_physics=8, moist=True,
                   bl_pbl_physics=1, sf_surface_physics=2,
                   sf_sfclay_physics=91, km_opt=4)
    spec = ((1, 200, 160, 9000.0, 30.0, 1, 1, 1),
            (2, 402, 300, 3000.0, 10.0, 3, 34, 42),
            (3, 1050, 750, 1000.0, 5.0, 3, 27, 28))
    nodes, parent = [], None
    for gid, nx, ny, dx, dt, ratio, istart, jstart in spec:
        run = RunConfig(nx=nx, ny=ny, nz=49, dx=dx, dy=dx, ztop=20000.0,
                        dt=dt, run_seconds=600.0, grid_id=gid,
                        nested=(gid > 1), specified=False, **physics)
        cfg = DomainConfig(grid_id=gid, parent_id=gid - 1,
                           i_parent_start=istart, j_parent_start=jstart,
                           parent_grid_ratio=ratio,
                           parent_time_step_ratio=ratio,
                           history_interval_s=600.0, run=run)
        node = types.SimpleNamespace(cfg=cfg, state=types.SimpleNamespace(),
                                     parent=parent)
        nodes.append(node)
        parent = node
    return nodes


def _decide_poland(monkeypatch, vram_gib=15.245):
    """Walk the measured tree and hand back its decisions."""
    import sys
    import types as _types

    from tilestream.autoplan import Machine

    nodes = _poland_tree_nodes()
    stub = _types.ModuleType("woof.core.dycore")
    stub.step = lambda *a, **kw: None
    monkeypatch.setitem(sys.modules, "woof.core.dycore", stub)
    model = _types.SimpleNamespace(walk_parent_first=lambda: nodes)
    machine = Machine(vram_bytes=int(vram_gib * (1024 ** 3)),
                      host_bytes=int(123.25 * (1024 ** 3)))
    decisions: dict = {}
    # The streamed d03 needs a builder to reach a stepper, and building one
    # needs a card.  The DECISIONS are all taken before any builder runs
    # (that split is its own contract above), so the refusal the harness
    # raises here is the missing builder and the arithmetic under test is
    # already in ``decisions``.
    try:
        streaming.steppers_for_tree(model, StreamingOptions(mode="auto"),
                                    machine=machine, decisions=decisions)
    except StreamingRefused as exc:
        assert "builder" in str(exc), exc
    return machine, decisions


def test_the_measured_three_domain_tree_plans_on_the_card_it_ran_on(
        monkeypatch):
    """THE DEFECT, against the run that hit it.

    ``autoplan.Footprint.vram_bytes`` prices ONE domain in ONE process, so
    it carries the CUDA context and the ``full`` rung's k-distribution
    tables -- 3.760 GiB with the safety factor -- inside every answer.  The
    walk subtracted that whole number once per DOMAIN, so a three-domain
    tree paid for three CUDA contexts and three copies of the tables:
    7.519 GiB of a 15.245 GiB card spent on bytes nothing allocates.  This
    tree was refused outright with ``CannotPlan: no tile fits in 2.45 GiB``,
    and the only way to run it was to hand ``[tiles] vram_budget_bytes`` a
    number 7.519 GiB larger than the card -- a workaround, in the config, in
    a shipped experiment.

    Priced once, the tree fits with room to spare, and the two resident
    domains land on the numbers the run itself reported.
    """
    machine, decisions = _decide_poland(monkeypatch)
    gib = 1024 ** 3

    assert set(decisions) == {1, 2, 3}
    assert decisions[1].stream is False and decisions[2].stream is False
    assert decisions[3].stream is True

    # The run's own report: d01 4.614 GiB, d02 6.932 GiB resident.  These
    # are whole-process prices and are NOT what the tree is charged; they
    # are here because they are the two numbers the measured run published,
    # and a model that reproduces them is the model that measured run used.
    assert abs(decisions[1].resident_bytes / gib - 4.614) < 0.005
    assert abs(decisions[2].resident_bytes / gib - 6.932) < 0.005


def test_the_per_process_fixed_cost_is_charged_once_for_the_whole_tree(
        monkeypatch):
    """The ledger identity, stated so it cannot drift back.

    ``claim_bytes`` is MARGINAL on every domain, and the tree's bill is one
    process overhead plus those claims plus the coupling corridors.  The
    discriminating assertion is the last one: the sum of the WHOLE prices
    exceeds the card, so a walk that charged whole prices could not have
    planned this tree at all -- which is exactly what it did.
    """
    machine, decisions = _decide_poland(monkeypatch)
    gib = 1024 ** 3

    overheads = {int(d.detail["tree_process_overhead_bytes"])
                 for d in decisions.values()}
    assert len(overheads) == 1, "the tree pays ONE overhead, not one per grid"
    overhead = overheads.pop()
    assert abs(overhead / gib - 3.760) < 0.005, overhead / gib

    claims = sum(int(d.detail["claim_bytes"])
                 + int(d.detail["corridor_claim_bytes"])
                 for d in decisions.values())
    assert overhead + claims <= machine.vram_budget_bytes, (
        f"{(overhead + claims) / gib:.3f} GiB priced against a "
        f"{machine.vram_budget_bytes / gib:.3f} GiB budget")

    # every claim is genuinely marginal, i.e. strictly below the whole price
    for gid, d in decisions.items():
        assert int(d.detail["claim_bytes"]) < int(d.resident_bytes), gid

    # THE CONTROL.  Charged per domain -- the old arithmetic -- the same
    # tree prices past the card, so this test cannot pass by accident on a
    # tree that was going to fit either way.
    per_domain = sum(int(d.resident_bytes) for d in decisions.values())
    assert per_domain > machine.vram_bytes, (
        f"{per_domain / gib:.3f} GiB of whole prices against a "
        f"{machine.vram_bytes / gib:.3f} GiB card")
    assert abs((per_domain - (overhead + claims)) / gib) > 7.0


def test_the_freed_budget_does_not_buy_a_tile_the_radiation_call_kills(
        monkeypatch):
    """The other half of the fix, and it is measured on both sides.

    Removing the phantom bytes hands the tile search 7.5 GiB it did not
    have, and left alone it spends them: at the accurate budget it picks a
    350x250 tile whose steady footprint the measured run put at ~14.34 GiB.
    Radiation's per-call transient measured +2.74 GiB on that card, so that
    tiling peaks at 17.08 GiB against a 15.92 GiB card -- dead at the first
    radiation call, roughly six minutes in.

    With the transient reserved (``autoplan.RADIATION_TRANSIENT_BYTES``),
    d03 lands on 175x375 instead: the tiling that actually ran, 12 tiles,
    measured peak 15.46 GiB with 467 MiB of card to spare.
    """
    machine, decisions = _decide_poland(monkeypatch)
    d03 = decisions[3]
    from woof.core.nest_stream import corridor_claim_bytes
    nodes = _poland_tree_nodes()
    overhead = streaming._tree_process_overhead_bytes(nodes)
    radiation = streaming._tree_radiation_transient_bytes(nodes)
    hold = overhead + sum(int(d.detail["claim_bytes"])
                          + int(d.detail["corridor_claim_bytes"])
                          for d in decisions.values())
    assert hold + radiation <= machine.vram_bytes
    assert 1050 % d03.tile_nx == 0 and 750 % d03.tile_ny == 0
    # The old dimension pin ignored the child's actual coupling allocation.
    # Keep the original larger unsafe-tile control and price all its claims.
    unsafe = dataclasses.replace(d03, tile_nx=350, tile_ny=250, nbuffers=2,
                                  resident_bytes=None)
    unsafe_claim = streaming._decision_claim_bytes(nodes[2], unsafe)
    unsafe_corridor = corridor_claim_bytes(nodes[2], decision=unsafe)
    unsafe_total = (hold - int(d03.detail["claim_bytes"])
                    - int(d03.detail["corridor_claim_bytes"])
                    + unsafe_claim + unsafe_corridor + radiation)
    assert unsafe_total > machine.vram_bytes



def test_a_tree_with_no_radiation_reserves_nothing_for_it(monkeypatch):
    """CONTROL: the reservation must not leak into a dry tree.

    Every dry plan in this project predates the constant.  If the radiation
    reservation applied where no radiation runs, the whole measured tile
    ladder would move and nothing in the dry suite would say so.
    """
    import sys
    import types as _types

    from tilestream.autoplan import Machine

    parent, child = _tree_nodes()               # dry rung, no radiation
    stub = _types.ModuleType("woof.core.dycore")
    stub.step = lambda *a, **kw: None
    monkeypatch.setitem(sys.modules, "woof.core.dycore", stub)
    model = _types.SimpleNamespace(walk_parent_first=lambda: [parent, child])
    machine = Machine(vram_bytes=8 << 30, host_bytes=64 << 30)
    decisions: dict = {}
    streaming.steppers_for_tree(model, StreamingOptions(mode="auto"),
                                machine=machine, decisions=decisions)
    assert streaming._tree_radiation_transient_bytes([parent, child]) == 0
    assert (streaming._tree_budget_bytes(machine, 0)
            == machine.vram_budget_bytes)
    # the first domain is still planned against the whole budget
    assert decisions[1].budget_bytes == machine.vram_budget_bytes


def test_the_receipt_records_every_grid_of_the_tree_with_its_road(
        monkeypatch):
    """The receipt half, and it is the half that can rot.

    A grid that declined to stream is ABSENT from the stepper dict, and
    absent is exactly what a grid looks like under a [tiles] that was
    never configured -- so without the receipt an operator cannot tell a
    tree that was asked and priced resident from a run that never asked.
    Both grids are decided by the PLANNER against a fabricated machine
    large enough that the whole tree fits resident: the legal shape, and
    the one the old short-circuit misdescribed for the child.

    ``woof.core.dycore`` is stubbed because ``make_stepper`` returns that
    module's ``step`` for every domain that does not stream, and importing
    it needs cupy.  Nothing under test is in it: the subject is which
    DECISION was taken and what the receipt says about it.
    """
    import sys
    import types as _types

    from tilestream.autoplan import Machine

    parent, child = _tree_nodes()
    stub = _types.ModuleType("woof.core.dycore")
    stub.step = lambda *a, **kw: None
    monkeypatch.setitem(sys.modules, "woof.core.dycore", stub)

    model = _types.SimpleNamespace(
        walk_parent_first=lambda: [parent, child])
    options = StreamingOptions(mode="auto")
    machine = Machine(vram_bytes=64 << 30, host_bytes=128 << 30)
    decisions: dict = {}
    steppers = streaming.steppers_for_tree(model, options, machine=machine,
                                           decisions=decisions)

    assert steppers == {}                       # the tree ran resident ...
    assert set(decisions) == {1, 2}             # ... and BOTH were decided
    assert decisions[2].detail["road"] == "resident"
    assert decisions[2].detail["corridor_claim_bytes"] > 0
    receipt = streaming.streaming_receipt(options, decisions)
    assert receipt["configured_mode"] == "auto"      # it WAS configured
    assert receipt["streamed_any"] is False          # it was NOT taken
    assert receipt["domains"]["2"]["streamed"] is False
    # The other receipt surface, the one the run's memory block carries.
    explained = streaming.receipt_entry(options, decisions)
    assert explained["any_streamed"] is False
    assert "d02" in explained["decisions"]


def test_a_scheme_that_cannot_be_twinned_is_refused_not_shared():
    """Sharing the domain's adapter shares its CARRIERS with every buffer.

    ``cumulus/w0avg`` is a carrier, so one Kain-Fritsch instance across the
    buffers means one w0avg array that the transport gathers and scatters as
    if each buffer owned its own.  This is the defect the geography check
    caught during development, and it is held here where it is cheap.
    """
    import numpy as np

    class _Configured:
        def __init__(self, chunk=1024):
            self.chunk = chunk

    lat = lon = np.zeros((4, 4))
    # A constructor that asks for nothing and configures nothing may be
    # rebuilt; the result must be a DIFFERENT object.
    plain = _Configured()
    twin = streaming._tile_scheme(plain, lat, lon)
    assert twin is not plain and twin.chunk == plain.chunk
    # One that carries policy a default reconstruction would drop may not.
    with pytest.raises(StreamingRefused, match="carries chunk="):
        streaming._tile_scheme(_Configured(chunk=50_000), lat, lon)


def test_a_dataclass_scheme_is_twinned_at_the_tiles_extents():
    import dataclasses

    import numpy as np

    @dataclasses.dataclass
    class _Adapter:
        latitude_deg: object
        longitude_deg: object
        column_chunk: int = 4096

    domain = _Adapter(np.zeros((8, 8)), np.zeros((8, 8)), column_chunk=12_500)
    lat, lon = np.ones((4, 4)), np.ones((4, 4))
    twin = streaming._tile_scheme(domain, lat, lon)
    assert twin is not domain
    assert twin.latitude_deg.shape == (4, 4)
    # The POLICY survives; only the geography moved.
    assert twin.column_chunk == 12_500


# --------------------------------------------------------------------------
# the twin RECIPE -- legacy RRTMG, the default suite's radiation
# --------------------------------------------------------------------------
#
# RRTMGLegacyRadiation is a plain class whose constructor requires
# (start_time, latitude_deg, longitude_deg), so _tile_scheme refused it and
# every [tiles] run of the SHIPPED DEFAULT physics suite died in
# TiledRun.__init__ -- streaming was reachable only by selecting rte-rrtmgp.
# These hold the recipe that rebuilds it and the two audits that keep the
# recipe accurate, without constructing one (its constructor wants CUDA).


def _recipe_key(cls):
    """The key _tile_scheme derives for a class."""
    return f"{cls.__module__}:{cls.__qualname__}"


def test_the_default_suites_legacy_rrtmg_has_a_twin_recipe():
    """The registration itself, against the REAL class.

    A source-shaped assertion for the same reason the route-builder test
    above is one: the defect was an ABSENT recipe, and nothing else in this
    suite would notice its removal.
    """
    from woof.core.rrtmg_legacy import RRTMGLegacyRadiation

    key = _recipe_key(RRTMGLegacyRadiation)
    assert key in streaming._TWIN_RECIPES, (
        "legacy RRTMG has no per-buffer twin recipe; every [tiles] run of "
        "the default physics suite will raise StreamingRefused")


def test_the_recipe_reproduces_every_constructor_parameter():
    """The staleness audit, run against the live signature.

    This is the test that fails when someone adds a policy argument to the
    adapter: a recipe that does not carry it would build buffers whose
    physics differs from the domain's at that argument's DEFAULT, which no
    inventory or geography check downstream can see.
    """
    import inspect

    from woof.core.rrtmg_legacy import RRTMGLegacyRadiation

    recipe = streaming._TWIN_RECIPES[_recipe_key(RRTMGLegacyRadiation)]
    declared = {p.name for p in
                inspect.signature(RRTMGLegacyRadiation).parameters.values()
                if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
    assert declared <= recipe.reproduces, (
        f"the recipe does not reproduce {sorted(declared - recipe.reproduces)}")


@pytest.mark.gpu
def test_the_twin_carries_the_transported_smoke_feedback():
    """simple_dir_fdb = 1 (c363fd55ff): a buffer's twin reads the same
    smoke tracer as the domain's adapter, and a smoke-free adapter's twin
    stays smoke-free.  Breakage this prevents: every tiled legacy-RRTMG
    run refused at forecast start ("does not reproduce ...
    ['transported_smoke']"), and a twin left at the default would run a
    smoky domain's tiles with no smoke in the shortwave."""
    from datetime import datetime

    import numpy as np
    from woof.core.rrtmg_legacy import RRTMGLegacyRadiation

    lat = np.zeros((2, 2), np.float32)
    start = datetime(2026, 10, 2, 21)
    for smoke in ("chem_smoke", None):
        scheme = RRTMGLegacyRadiation(start, lat, lat, longwave=False,
                                      shortwave=True, aer_opt=3,
                                      transported_smoke=smoke)
        twin = streaming._twin_rrtmg_legacy(
            scheme, RRTMGLegacyRadiation, lat[:1], lat[:1])
        assert twin._transported_smoke == smoke


class _RequiresArgs:
    """A stand-in with legacy RRTMG's SHAPE: required args plus policy."""

    def __init__(self, start_time, latitude_deg, longitude_deg, *,
                 column_chunk=4096):
        self.start_time = start_time
        self.latitude_deg = latitude_deg
        self.longitude_deg = longitude_deg
        self.column_chunk = column_chunk
        self.update_count = 0


def _register(monkeypatch, cls, **kw):
    kw.setdefault("build", lambda scheme, cls_, lat, lon: cls_(
        scheme.start_time, lat, lon, column_chunk=scheme.column_chunk))
    kw.setdefault("reproduces", frozenset({
        "start_time", "latitude_deg", "longitude_deg", "column_chunk"}))
    kw.setdefault("volatile", frozenset({"update_count"}))
    table = dict(streaming._TWIN_RECIPES)
    table[_recipe_key(cls)] = streaming._TwinRecipe(**kw)
    monkeypatch.setattr(streaming, "_TWIN_RECIPES", table)


def test_a_recipe_twin_takes_the_tiles_geography_and_the_domains_policy(
        monkeypatch):
    import numpy as np

    _register(monkeypatch, _RequiresArgs)
    domain = _RequiresArgs("T0", np.zeros((8, 8)), np.zeros((8, 8)),
                           column_chunk=12_500)
    lat, lon = np.ones((4, 4)), np.ones((4, 4))
    twin = streaming._tile_scheme(domain, lat, lon)

    assert twin is not domain                       # a FRESH object, not shared
    assert twin.latitude_deg.shape == (4, 4)        # the TILE's geography
    assert twin.start_time == "T0"                  # the DOMAIN's start time
    assert twin.column_chunk == 12_500              # the DOMAIN's policy


def test_a_stale_recipe_is_refused_rather_than_defaulting_the_policy(
        monkeypatch):
    """A recipe that does not name every constructor parameter is stale."""
    import numpy as np

    _register(monkeypatch, _RequiresArgs,
              reproduces=frozenset({"start_time", "latitude_deg",
                                    "longitude_deg"}))
    domain = _RequiresArgs("T0", np.zeros((8, 8)), np.zeros((8, 8)))
    with pytest.raises(StreamingRefused, match="stale against the adapter"):
        streaming._tile_scheme(domain, np.ones((4, 4)), np.ones((4, 4)))


def test_a_recipe_that_drops_policy_is_refused(monkeypatch):
    """The scalar audit, applied to the recipe's own output."""
    import numpy as np

    _register(monkeypatch, _RequiresArgs,
              build=lambda scheme, cls_, lat, lon: cls_(
                  scheme.start_time, lat, lon))       # drops column_chunk
    domain = _RequiresArgs("T0", np.zeros((8, 8)), np.zeros((8, 8)),
                           column_chunk=12_500)
    with pytest.raises(StreamingRefused, match="drops policy"):
        streaming._tile_scheme(domain, np.ones((4, 4)), np.ones((4, 4)))


def test_a_running_counter_is_not_mistaken_for_dropped_policy(monkeypatch):
    """The domain's adapter has stepped; a fresh twin has not."""
    import numpy as np

    _register(monkeypatch, _RequiresArgs)
    domain = _RequiresArgs("T0", np.zeros((8, 8)), np.zeros((8, 8)))
    domain.update_count = 17
    twin = streaming._tile_scheme(domain, np.ones((4, 4)), np.ones((4, 4)))
    assert twin.update_count == 0


def test_an_adapter_with_no_recipe_is_still_refused():
    """The guard is EXTENDED by the recipe table, not weakened by it."""
    import numpy as np

    domain = _RequiresArgs("T0", np.zeros((8, 8)), np.zeros((8, 8)))
    with pytest.raises(StreamingRefused, match="_TWIN_RECIPES"):
        streaming._tile_scheme(domain, np.ones((4, 4)), np.ones((4, 4)))


# --------------------------------------------------------------------------
# the buffer's VERTICAL COORDINATE -- the rte-rrtmgp warm-up crash
# --------------------------------------------------------------------------
#
# A tile buffer was built on make_vertical_coord's DEFAULT stretch while
# _impose_domain_setup gave it the domain's eta table, so its 3-D base state
# (thb/pb/alb/phb, not imposed because they are gathered) described a
# different atmosphere from its pressure.  The warm-up step then handed
# rte-rrtmgp 120.3-407.3 K and the gas-table validator refused it.  The
# validator is right and is untouched; the buffer was wrong.


class _EtaState:
    def __init__(self, znw):
        self.znw = znw


class _EtaCfg:
    hybrid_opt, etac = 2, 0.2

    def __init__(self, nz):
        self.nz = nz


def _real_eta(nz=8):
    """A table that is NOT the default stretch, as a real case's is not."""
    import numpy as np
    x = np.linspace(0.0, 1.0, nz + 1)
    znw = (1.0 - x) ** 1.7            # monotone 1 -> 0, clustered low
    znw[0], znw[-1] = 1.0, 0.0
    return znw


def test_the_buffer_is_built_on_the_domains_eta_table_not_the_default():
    import numpy as np

    eta = _real_eta(8)
    coord = streaming.domain_vertical_coord(_EtaState(eta), _EtaCfg(8))
    assert np.allclose(np.asarray(coord.znw, dtype=np.float64), eta)

    from woof.core.grid import make_vertical_coord
    default = make_vertical_coord(8, hybrid_opt=2, etac=0.2)
    # The whole point: the two genuinely differ, so building on the default
    # is a real error rather than a harmless rebuild.
    assert not np.allclose(np.asarray(default.znw, dtype=np.float64), eta)


def test_a_domain_without_an_eta_table_is_refused_not_defaulted():
    with pytest.raises(StreamingRefused, match="carries no znw"):
        streaming.domain_vertical_coord(_EtaState(None), _EtaCfg(8))


def test_an_unrebuildable_eta_table_is_refused_not_defaulted():
    """Falling back to the default stretch IS the defect, so never fall back."""
    import numpy as np

    bad = np.array([1.0, 0.5, 0.6, 0.0])       # not monotone
    with pytest.raises(StreamingRefused, match="cannot be rebuilt"):
        streaming.domain_vertical_coord(_EtaState(bad), _EtaCfg(3))


def test_the_prepared_factory_builds_buffers_on_the_domains_coordinate():
    """Held as source: constructing a buffer needs a card and a real domain."""
    import inspect

    src = inspect.getsource(streaming.prepared_tile_state_factory)
    assert "coord = domain_vertical_coord(state, cfg)" in src
    assert "coord=coord" in src, (
        "the factory computes the domain's coordinate and then does not "
        "hand it to make_physics_state, which rebuilds the DEFAULT stretch")


def test_the_routes_that_stream_do_not_refuse_the_mode_that_asks_for_it():
    """mode = 'on' was refused at admission BY THE ROUTES THAT SUPPORT IT.

    refuse_unrouted_streaming was written when the prepared routes passed no
    builders.  They pass them now -- the test above is the proof -- but the
    admission refusal outlived the wiring, so the explicit form was rejected
    with a message asserting the route wired no builder while mode = 'auto'
    streamed through that very builder.  `woof go` mirrored the refusal
    before the download, making the front door most users type the last
    place that rejected streaming.
    """
    import inspect

    import woof.go_cli as go
    import woof.prepared_domain_tree_forecast as tree
    import woof.prepared_single_domain_forecast as single

    for module in (single, tree, go):
        src = inspect.getsource(module)
        calls = [line for line in src.splitlines()
                 if "refuse_unrouted_streaming(" in line
                 and not line.strip().startswith("#")]
        assert not calls, (
            f"{module.__name__} refuses [tiles] mode='on' at admission but "
            f"wires a streamed-domain builder: {calls}")


def test_the_run_route_reads_tiles_instead_of_refusing_it():
    """``woof run`` streams now, so its front-door refusal is GONE.

    The refusal it replaced was accurate while it stood: the route wired no
    builder, so a [tiles] block was read, validated, echoed into the
    resolved-config report and dropped, and the run went resident saying
    nothing.  The remedy for that was never a permanent refusal -- it was
    the wiring, and the refusal's own message said which wiring
    (``streaming.builders_for_tree``).

    Asserted as the presence of the seam rather than the absence of the
    refusal, because absence is what a deletion also looks like.
    """
    import inspect

    from woof import runtime

    src = inspect.getsource(runtime._run_experiment)
    scheduled = inspect.getsource(runtime._run_built_experiment)
    assert "builders=_streaming.builders_for_tree(model, exp.tiles)" in scheduled
    assert "_run_built_experiment(" in src
    assert "refuse_unrouted_streaming(exp, \"woof run\"" not in src
    # The single-domain arm is wired too, through the same builder the
    # offline child uses -- a root with no tree.
    assert "standalone_domain_builder(" in src
    assert "stepper=single_stepper" in src


def test_the_run_route_refreshes_a_streamed_domain_before_reading_it():
    """The read-back, without which every frame is the initial condition.

    A streamed domain's forecast lives in its pinned host store and the
    ``DomainState`` the writers and the digest hold is the snapshot that
    filled it.  Both readers on this route have to be given the copy, and
    the two places are exactly the cadence ``refresh_state`` publishes:
    the history frame, and the end of the run.
    """
    import inspect

    from woof import runtime

    src = inspect.getsource(runtime._run_built_experiment)
    assert src.count("_streaming.refresh_streamed_state(") == 1
    assert "stepper.canonical_digest(node.clock" in src
    history = src
    assert history.index("_streaming.refresh_streamed_state(") < \
        history.index("_submit_tree_history_frame(writers, node, ticks)")


# --------------------------------------------------------------------------
# REFL_10CM: a REBUILT slot the store still has to hold
# --------------------------------------------------------------------------


class _ScratchState:
    """A state with just the scratch API prime/inventory use."""

    def __init__(self, **slots):
        self._scratch = dict(slots)
        self.requested = []

    def existing_scratch(self, slot):
        return self._scratch.get(slot)

    def scratch(self, shape, slot, dtype=None):
        self.requested.append((slot, tuple(shape)))
        self._scratch.setdefault(slot, ("array", tuple(shape)))
        return self._scratch[slot]


class _Cfg:
    nz, ny, nx = 49, 550, 550


def test_the_reflectivity_slot_is_primed_at_the_domains_extents():
    state = _ScratchState()
    assert streaming.prime_refl_10cm(state, _Cfg()) is True
    assert state.requested == [("refl_10cm", (49, 550, 550))]


def test_priming_an_already_primed_slot_does_nothing():
    """A slot keeps the shape it was first requested with, so never re-ask."""
    state = _ScratchState(refl_10cm=("array", (49, 550, 550)))
    assert streaming.prime_refl_10cm(state, _Cfg()) is False
    assert state.requested == []


def test_the_inventory_carries_the_primed_slot_for_state_and_buffer():
    """The transport applies one inventory_fn to state, store and buffers."""
    state = _ScratchState(refl_10cm="REFL")
    inventory = streaming.refl_inventory(lambda obj, names=None: {"state/thp": 1})
    assert inventory(state) == {"state/thp": 1,
                                streaming.REFL_STORE_KEY: "REFL"}
    # The store already carries it by key and must not be double-added.
    store = {"state/thp": 1, streaming.REFL_STORE_KEY: "REFL"}
    assert streaming.refl_inventory(
        lambda obj, names=None: dict(obj))(store) == store


def test_an_unprimed_slot_is_not_invented_by_the_inventory():
    """prime_refl_10cm creates the slot; the inventory only reports it.

    If the inventory allocated, the refusal for a domain that genuinely
    cannot publish reflectivity could never fire.
    """
    inventory = streaming.refl_inventory(lambda obj, names=None: {"state/thp": 1})
    assert inventory(_ScratchState()) == {"state/thp": 1}


def test_the_prepared_builder_primes_and_carries_the_reflectivity_slot():
    """Held as source: the alternative needs a card and an hour of forecast.

    The defect was an absent call -- the sweep refused the first DUE frame,
    which on an hourly cadence is an hour into a healthy forecast.

    The inventory rule is NAMED (``streamed_store_inventory``) rather
    than spelled out at each call site, because it is handed to three
    things that are compared against each other -- the domain at attach,
    the tile buffers, and the store the store-direct road builds slab by
    slab.  Spelled out, it was already wrong in the third: the store came
    out one carrier short and TiledRun refused the pair with "in TILE not
    in STORE: ['scratch/refl_10cm']".  So the assertion is that the
    builders CALL the shared rule, plus behavioural checks on the shared
    rule itself -- the source check alone would pass if that function
    stopped adding the slots.

    The OUTER wrapper is asserted too, and for a harsher reason: leaving it
    off is not refused anywhere, it publishes a frame short of OLR and
    reports validity PASS.  That is what shipped until 2.2.0, so the
    composition -- diagnostic_inventory OVER refl_inventory, both of them
    over the streaming manifest -- is the thing under test, not either half.
    """
    import inspect

    src = inspect.getsource(streaming.prepared_domain_builder)
    assert "prime_lazy_carriers(state, cfg)" in src
    assert "inventory_fn=streamed_store_inventory()" in src
    store_src = inspect.getsource(streaming.store_domain_builder)
    assert "inventory_fn=streamed_store_inventory()" in store_src
    shared_src = inspect.getsource(streaming.streamed_store_inventory)
    assert "diagnostic_inventory(refl_inventory(" in shared_src
    factory_src = inspect.getsource(streaming.prepared_tile_state_factory)
    assert "prime_lazy_carriers(tile, tile_cfg)" in factory_src

    class _Streamed(_ScratchState):
        """``arrays`` takes streaming_inventory's mapping branch.

        Which is what lets the rule be exercised without a DomainState:
        the store the loader fills is itself a mapping, and this is the
        same branch it takes.
        """

        arrays = {"state/thp": 1}

    inventory = streaming.streamed_store_inventory()(
        _Streamed(refl_10cm=("array", (2, 3, 4))), None)
    assert streaming.REFL_STORE_KEY in inventory
    assert "state/thp" in inventory


def test_the_diagnostic_wrapper_adds_the_output_only_rows_over_any_base():
    """Composable over the STREAMING manifest, not only the carrier one.

    ``tilestream.output.diagnostic_inventory`` composes over
    ``carrier_manifest``; this route's base is ``streaming_inventory``, the
    wider restart manifest.  Wrapping rather than substituting is what keeps
    one naming table, so this pins that the wrapper adds exactly the
    ``diag/`` members and leaves the base's own keys untouched.
    """
    import numpy as np

    class _Driver:
        def __init__(self):
            self.olr = np.zeros((4, 5), dtype=np.float32)

    class _State:
        def __init__(self):
            self.physics = _Driver()

        def existing_scratch(self, name):
            return None

    state = _State()
    base = lambda obj, names=None: {"state/thp": 1}          # noqa: E731
    live = streaming.diagnostic_inventory(base)(state)
    assert live["state/thp"] == 1
    assert live["diag/olr"] is state.physics.olr

    # A store already holds its members by key, so the wrapper adds nothing.
    store = {"state/thp": 1, "diag/olr": "OLR"}
    assert streaming.diagnostic_inventory(
        lambda obj, names=None: dict(obj))(store) == store


def test_the_eddy_viscosity_producers_are_primed_only_when_asked():
    """8 B/cell is 100x OLR's price; a run that does not ask must not pay it.

    And the slot names are LITERALS here for the same reason
    prime_refl_10cm's is: tests/test_preflight.py's scratch-completeness
    gate reads them with an AST scan and classifies a variable expression
    as unpinned.
    """
    import inspect

    src = inspect.getsource(streaming.prime_hmix_k_diag)
    assert '"smag_km"' in src and '"smag_kh"' in src

    class _Cfg:
        nx, ny, nz = 4, 5, 6
        hmix_k_diag = False

    state = _ScratchState()
    assert streaming.prime_hmix_k_diag(state, _Cfg()) == ()

    _Cfg.hmix_k_diag = True
    assert streaming.prime_hmix_k_diag(state, _Cfg()) == (
        "diag/XKMH", "diag/XKHH")
    # Idempotent: a slot that exists is left alone rather than reallocated.
    assert streaming.prime_hmix_k_diag(state, _Cfg()) == ()


# --------------------------------------------------------------------------
# priming REPLACES the throwaway warm-up step
# --------------------------------------------------------------------------


class _KFStandIn:
    def __init__(self):
        self.calls = 0

    def ensure_trigger_history(self, state):
        self.calls += 1
        # The primer names the cumulus arrays the callable now holds
        # (restart.CUMULUS_CALLABLE_ARRAYS), as KainFritsch does.
        self.w0avg = object()
        return "w0avg"


class _PrimeState(_ScratchState):
    def __init__(self, cumulus=None, **slots):
        super().__init__(**slots)
        self.physics = type("_D", (), {"cumulus_callable": cumulus})()


def test_priming_allocates_both_lazy_carriers():
    kf = _KFStandIn()
    state = _PrimeState(cumulus=kf)
    assert streaming.prime_lazy_carriers(state, _Cfg()) == (
        streaming.REFL_STORE_KEY, "cumulus/w0avg")
    assert kf.calls == 1
    assert state.requested == [("refl_10cm", (49, 550, 550))]


def test_priming_a_domain_without_cumulus_primes_only_reflectivity():
    state = _PrimeState(cumulus=None)
    assert streaming.prime_lazy_carriers(state, _Cfg()) == (
        streaming.REFL_STORE_KEY,)


def test_the_prepared_factory_no_longer_steps_a_fabricated_buffer():
    """warmup defaults to 0, because the step was integrating fabricated data.

    A buffer at construction holds an analytic sounding, neutral geography
    and -- on a specified-boundary domain -- the DOMAIN's real lateral
    forcing.  The throwaway step blended real boundary values into an
    analytic interior; MEASURED, that handed rte-rrtmgp 211.0-456.7 K.
    """
    import inspect

    for fn in (streaming.prepared_tile_state_factory,
               streaming.prepared_domain_builder):
        default = inspect.signature(fn).parameters["warmup"].default
        assert default == 0, (
            f"{fn.__name__} still warms buffers by stepping them; the step "
            "integrates an analytic sounding against real lateral forcing")


# --------------------------------------------------------------------------
# the FINAL reads, which were answering for the initial condition
# --------------------------------------------------------------------------


def test_a_resident_stepper_refreshes_nothing_and_says_zero():
    """The OFF contract: a route calls this unconditionally."""
    assert streaming.refresh_streamed_state(object(), "STATE") == 0
    assert streaming.refresh_streamed_state(None, "STATE") == 0


def test_a_streamed_domain_is_copied_back_before_the_final_reads():
    """StateHealthValidator and canonical_state_digest cannot be folded.

    The validator is one block per whole field with no tile-interior form
    and the digest is a whole-trajectory hash, so under a host store both
    answered for the t=0 analysis: a receipt reporting nan_free on a run
    that had integrated for an hour.

    The stepper is a real StreamedDomain built field-wise, because
    ``is_streaming`` is an isinstance check -- a duck-typed stand-in is
    reported RESIDENT and would make this test pass against a no-op.
    """
    seen = {}

    def _refresh(state):
        seen["state"] = state
        return 137

    streamed = object.__new__(streaming.StreamedDomain)
    streamed.refresh_state = _refresh
    assert streaming.refresh_streamed_state(streamed, "STATE") == 137
    assert seen == {"state": "STATE"}


def test_a_resident_store_refresh_reports_zero_rather_than_copying():
    """``store = "device"`` makes the store the state's OWN arrays.

    refresh_state returns 0 there because a refresh would be a self-copy of
    the whole manifest, and the route must record that rather than
    claim a refresh it did not need.
    """
    streamed = object.__new__(streaming.StreamedDomain)
    streamed.refresh_state = lambda state: 0
    assert streaming.refresh_streamed_state(streamed, "STATE") == 0


def test_the_prepared_route_refreshes_before_the_final_health_and_digest():
    """Held as source, and ORDER is the whole point.

    A refresh after the gate reads is a refresh that changed nothing.
    """
    import pathlib

    # Read rather than imported: the runner needs a CUDA build, and this is
    # a question about the ORDER of three statements in its source.
    src = (pathlib.Path(streaming.__file__).parents[1]
           / "prepared_single_domain_forecast.py").read_text(encoding="utf-8")
    refresh = src.index("streaming.refresh_streamed_state(")
    # The FINAL gate specifically.  The runner validates the state at entry
    # too, and that earlier call is not the one this is about: at entry the
    # state IS the domain, because nothing has streamed off it yet.
    health = src.index('validate(phase="final.d01")')
    digest = src.index("final_digest = canonical_state_digest(")
    assert refresh < health < digest, (
        "the streamed domain must be copied back BEFORE the final health "
        "gate and the canonical digest read node.state")
    assert '"carriers_refreshed_before_final_reads": carriers_refreshed' \
        in src, "the receipt must say whether the refresh happened"


# --------------------------------------------------------------------------
# the ANALYSIS frame's health, before any sweep has run
# --------------------------------------------------------------------------


def _bare_fold(**kw):
    """A StreamedStability with no device behind it.

    ``__init__`` allocates cupy buffers; every property these tests exercise
    is decided before a kernel is reached, so the object is built field-wise.
    """
    fold = object.__new__(streaming.StreamedStability)
    fold.cfg = kw.pop("cfg", "CFG")
    fold.boundary_width = kw.pop("boundary_width", 5)
    fold.ntiles = kw.pop("ntiles", 4)
    fold.sweeps_begun = kw.pop("sweeps_begun", 0)
    fold._seen = set(kw.pop("seen", ()))
    fold._report = None
    for key, value in kw.items():
        setattr(fold, key, value)
    return fold


def test_the_analysis_frame_reads_the_state_that_filled_the_store(monkeypatch):
    """A t=0 history frame is asked for its health before anything is stepped.

    There is no fold yet because there has been no sweep -- not a short one,
    none -- and the resident state is still the initial condition the frame
    contains, because attach copied it into the store and neither has moved.
    Refusing here stopped every [tiles] forecast on the product route at its
    ANALYSIS frame, after the sweep had already been proven to build.
    """
    import sys
    import types

    seen = {}

    def _resident(state, cfg, *, boundary_width=None):
        seen.update(state=state, cfg=cfg, boundary_width=boundary_width)
        return {"cfl_max": 0.25}

    # Stubbed as a module: the real woof.core.dycore needs a CUDA build, and
    # what is under test is which reporter the fold reaches for.
    stub = types.ModuleType("woof.core.dycore")
    stub.stability_report = _resident
    monkeypatch.setitem(sys.modules, "woof.core.dycore", stub)
    fold = _bare_fold(sweeps_begun=0)
    assert fold("STATE", "RUNCFG", boundary_width=5) == {"cfl_max": 0.25}
    assert seen == {"state": "STATE", "cfg": "RUNCFG", "boundary_width": 5}


def test_the_analysis_frame_without_a_state_refuses_rather_than_inventing_one():
    fold = _bare_fold(sweeps_begun=0)
    with pytest.raises(StreamingRefused, match="before the first sweep"):
        fold(None, "RUNCFG", boundary_width=5)


def test_a_short_sweep_still_refuses_once_a_sweep_has_run():
    """The guard is NOT weakened: the exemption is keyed on sweeps_begun.

    Once a sweep has run, the resident state is the corpse the fold exists to
    stop reading, so a sweep that skipped tiles must refuse even though a
    state is right there and would answer.
    """
    fold = _bare_fold(sweeps_begun=1, seen=(0, 1))
    with pytest.raises(StreamingRefused, match="contributed a stability"):
        fold("STATE", "RUNCFG", boundary_width=5)


def test_begin_sweep_is_what_ends_the_analysis_frame_exemption():
    fold = _bare_fold(sweeps_begun=0)
    assert fold.sweeps_begun == 0
    streaming.StreamedStability.begin_sweep(fold)
    assert fold.sweeps_begun == 1
    # and the record is cleared, which is begin_sweep's original job
    assert fold._seen == set()


def test_the_ozone_grid_is_carried_by_a_checkpoint_and_not_by_a_sweep():
    """The fourth classification, and why it is not the third.

    Legacy RRTMG's radiation/o33d_grid is host memory the adapter recomputes
    on every radiation call and reads only from a CHILD domain's ozone
    provider.  A checkpoint must carry it; a sweep must not.  With it in the
    sweep's set, a tile buffer that warmed up had it and the prepared domain
    the store was sized from had not run a step, so the inventories differed
    by exactly this key and TiledRun refused every [tiles] forecast of the
    default physics suite.
    """
    from woof.io import restart

    assert "radiation/o33d_grid" in restart.RESTART_ONLY_DRIVER_SLOTS
    # The two directions are genuinely different sets, not one list twice.
    assert not (restart.RESTART_ONLY_DRIVER_SLOTS
                & restart.CARRIED_SCRATCH_SLOTS)
    # The checkpoint half is UNTOUCHED: restart still names the key.
    import inspect
    assert 'manifest["radiation/o33d_grid"] = o33d_grid' in \
        inspect.getsource(restart._driver_manifest)


def test_the_sweep_manifest_subtracts_the_restart_only_slots(monkeypatch):
    """The subtraction itself, on both sides at once.

    Stubbed rather than driven from a real state: carrier_manifest's job here
    is set arithmetic, and the arithmetic is what regressed.
    """
    from woof.io import restart
    from tilestream import physics_inventory as physinv

    monkeypatch.setattr(restart, "state_manifest", lambda s: {"state/thp": 1})
    monkeypatch.setattr(restart, "_scratch_manifest", lambda s: {})
    monkeypatch.setattr(restart, "carried_scratch_manifest", lambda s: {})
    monkeypatch.setattr(restart, "_driver_manifest", lambda d, **_: {
        "cumulus/w0avg": 2, "radiation/o33d_grid": 3})

    class _State:
        physics = object()
        # The streamed driver walk sizes owners from the pressure field.
        p = type("_P", (), {"shape": (2, 3, 4)})()

    manifest = physinv.carrier_manifest(_State())
    assert "radiation/o33d_grid" not in manifest
    # Only that one: the lazy carrier that IS transported still is.
    assert manifest == {"state/thp": 1, "cumulus/w0avg": 2}


def test_the_legacy_sw_engine_is_shared_rather_than_compiled_per_buffer(
        monkeypatch):
    """One NVRTC compile and one set of device tables per process.

    Streaming builds an adapter per TILE BUFFER.  CudaSW compiles
    rrtmg_sw.cu and uploads the packed tables on construction, so an
    uncached engine would multiply both -- spending the device memory
    streaming exists to save.  Held here because the receipt on the card is
    a VRAM number nothing else asserts on.
    """
    from woof.core import rrtmg_legacy

    built = []

    class _Tables:
        pass

    card = [0]
    monkeypatch.setattr(rrtmg_legacy, "_CUDA_SW_CACHE", None)
    monkeypatch.setattr(rrtmg_legacy, "_cuda_sw_device", lambda: card[0])
    monkeypatch.setattr(rrtmg_legacy._sw, "CudaSW",
                        lambda tab: built.append(tab) or object())
    tables = _Tables()
    first = rrtmg_legacy._cuda_sw(tables)
    assert rrtmg_legacy._cuda_sw(tables) is first
    assert len(built) == 1, "the SW engine was compiled more than once"
    # Different coefficients get their own engine rather than borrowing.
    assert rrtmg_legacy._cuda_sw(_Tables()) is not first
    assert len(built) == 2
    # A second CARD gets its own engine: a module and tables live on one
    # card, and a [devices] slab on another must not launch them.
    card[0] = 1
    assert rrtmg_legacy._cuda_sw(tables) is not first
    assert len(built) == 3
    card[0] = 0
    assert rrtmg_legacy._cuda_sw(tables) is first


def test_the_budget_overrides_land_on_the_budget():
    """``vram_budget_bytes`` used to compute a budget of ZERO.

    ``Machine`` reaches its budgets through multipliers -- ``vram_bytes *
    (1 - vram_headroom)`` and ``host_bytes * pinned_fraction`` -- and
    ``decide`` replaced the raw capacity while leaving the multiplier on
    top.  For VRAM the multiplier it installed was ``1 - 1.0``, so every
    configuration that set the key was refused with "no tile fits in
    0.00 GiB of VRAM" however large a number it asked for.

    Held as an arithmetic identity rather than as a planner outcome,
    because the planner's answer depends on the domain and this does not:
    the number in the TOML must BE the budget.
    """
    from tilestream import autoplan

    class _Stop(Exception):
        """Stop inside the planner with the budgets it was actually given."""

    cfg = _cfg()
    seen = {}

    def spy(cfg_, machine, **kw):
        seen["vram"] = machine.vram_budget_bytes
        seen["host"] = machine.host_budget_bytes
        raise _Stop()

    real = autoplan.plan
    autoplan.plan = spy
    try:
        with pytest.raises(_Stop):
            decide(cfg, StreamingOptions(
                mode="auto",
                vram_budget_bytes=20 * 2 ** 30,
                host_budget_bytes=60 * 2 ** 30),
                machine=autoplan.Machine(vram_bytes=32 * 2 ** 30,
                                         host_bytes=128 * 2 ** 30))
    finally:
        autoplan.plan = real
    assert seen["vram"] == 20 * 2 ** 30, (
        f"[tiles] vram_budget_bytes = 20 GiB reached the planner as "
        f"{seen['vram']} bytes")
    assert seen["host"] == 60 * 2 ** 30


def test_a_configured_budget_refusal_names_the_measured_free_figure():
    """Run 3641453f001b81fe, 2026-08-24: the stale-freeze refusal.

    A front end froze ``[tiles] vram_budget_bytes = 65,011,712`` (62 MiB,
    probed while the card was running another forecast) into the config.  At
    launch ``decide`` measured the card -- 8.38 GiB free -- then threw the
    measurement away and refused from the frozen number's arithmetic, so the
    operator was told the card was out of VRAM while it sat mostly idle.  A
    refusal computed from a CONFIGURED budget must carry the measurement the
    decision took beside it: that pairing is what tells the reader the card
    is fine and the configured number is what went stale.
    """
    import re

    from tilestream import autoplan

    cfg = autoplan._config_for_rung(282, 155, 49, "full", specified=True)
    machine = autoplan.Machine(int(8.38 * 2 ** 30), 64 * 2 ** 30,
                               name="RTX 3080")
    with pytest.raises(autoplan.CannotPlan) as info:
        decide(cfg, StreamingOptions(mode="auto",
                                     vram_budget_bytes=65_011_712,
                                     host_budget_bytes=64 * 2 ** 30),
               machine=machine)
    msg = str(info.value)
    assert "8.38 GiB" in msg, f"the measured free figure is missing: {msg}"
    assert "vram_budget_bytes" in msg, (
        f"the configured key is not named as the number refused on: {msg}")
    assert re.search(r"-\d", msg) is None, (
        f"a negative figure reached a user-facing refusal: {msg}")
    assert info.value.resource == "vram"
    assert info.value.detail["measured_free_bytes"] == int(8.38 * 2 ** 30)
    assert info.value.detail["configured_vram_budget_bytes"] == 65_011_712


def test_write_mode_reaches_the_transport_instead_of_being_ignored():
    """``attach`` used to hardcode ``write_mode="ring"``.

    ``decide`` hands ``autoplan.plan`` the configured write mode and the
    planner sizes the HOST budget with it -- shadow keeps a whole second
    store, ring keeps one plus a small arena -- so a run planned for shadow
    and attached as ring was planned for one footprint and executed with
    another.  The value now rides on the decision, which is the object
    ``attach`` is given.
    """
    got = decide(_cfg(), StreamingOptions(mode="on", tile_nx=64, tile_ny=64,
                                          write_mode="shadow"))
    assert got.write_mode == "shadow"
    assert decide(_cfg(), StreamingOptions(mode="on", tile_nx=64,
                                           tile_ny=64)).write_mode == "ring"


def test_the_case_data_route_supplies_the_stepper_it_used_to_refuse_for():
    """The replacement this file asked for, in the words it asked for it.

    The test that stood here asserted ``woof run`` still refused
    ``[tiles]``, and said in its own body that if a caller ever supplied
    ``integrate_prepared_case``'s ``stepper`` the refusal "may be
    replaceable" and the test "should be replaced by a real streamed run,
    not deleted".  A caller supplies it now: the single-domain arm builds
    one through ``standalone_domain_builder`` and the tree arm builds the
    whole mapping through ``builders_for_tree``.

    The real streamed runs are the GPU legs (the streamed-child corridor
    gate and the two-way leg on the card); what is held HERE, on the CPU,
    is that the loop is given something to stream WITH -- because a route
    that decides to stream and then hands the loop nothing is the silent
    resident run all over again.
    """
    import inspect
    import io
    import tokenize

    from woof import runtime

    source = inspect.getsource(runtime)
    rows = source.splitlines(keepends=True)
    for kind, _text, (r0, c0), (_r1, c1), _l in tokenize.generate_tokens(
            io.StringIO(source).readline):
        if kind == tokenize.COMMENT:
            line = rows[r0 - 1]
            rows[r0 - 1] = line[:c0] + " " * (c1 - c0) + line[c1:]
    whole = "".join(rows)
    # The definition itself does not count as a caller.
    assert "stepper=single_stepper" in whole.replace(
        "feedback=None, stepper=None", "")
    # And the decision that chose it is the one the loop runs on: decided
    # once, handed over, never re-derived inside make_stepper.
    src = inspect.getsource(runtime._run_experiment)
    assert "decision=single_decision" in src


# ---- bundle-07 adversarial review: the three findings in this lane's code --

def test_the_tree_wrapper_hands_the_configured_host_budget_to_the_probe():
    """Finding 2: #215 reopened at the PRODUCTION door to the planner.

    ``decide`` reads ``host_budget_bytes`` BEFORE probing and hands it to
    ``Machine.detect``, because ``detect`` skips the host-memory read
    entirely when it is told the budget and RAISES where it can find no
    host source.  That arm runs only when no machine was supplied, and
    ``steppers_for_tree`` supplied one: it probed bare and overrode
    afterwards, so on a box with no procfs and no cgroups every [tiles]
    tree was refused before the override was ever consulted -- including
    the ones that set the key to the number the refusal asked for.

    Held by making detection FAIL unless it is told, which is exactly the
    box the defect was reachable on.
    """
    from tilestream import autoplan

    seen = []
    real_detect = autoplan.Machine.detect

    def picky_detect(*, host_bytes=None, **kwargs):
        seen.append(host_bytes)
        if host_bytes is None:
            raise RuntimeError("no host memory source on this box")
        return autoplan.Machine(vram_bytes=64 << 30,
                                host_bytes=int(host_bytes))

    model = _one_grid_model()
    options = StreamingOptions(mode="auto", host_budget_bytes=17 << 30)
    autoplan.Machine.detect = staticmethod(picky_detect)
    try:
        streaming.steppers_for_tree(model, options)
    finally:
        autoplan.Machine.detect = real_detect
    assert seen == [17 << 30], seen


def test_two_streamed_siblings_do_not_each_get_the_whole_host_budget():
    """Finding 5: the walk keeps a HOST ledger as well as a VRAM one.

    A streamed domain spends both pools -- tile buffers on the card, a
    whole-domain pinned store plus arena on the box -- and the single
    ``spent`` total only ever subtracted the VRAM.  So a later streamed
    domain was priced against the ENTIRE host budget, its plan was
    accepted, and the run met ``cudaHostAlloc`` where a planner refusal
    belonged.  Host RAM is the binding constraint at every capacity limit
    measured, so this is the ledger that most needs keeping.

    TWO domains that both STREAM, which is what makes the ledger visible:
    a resident domain pins no store, so a fixture where everything fits
    reports zero either way and proves nothing.
    """
    import types as _types

    from woof.config import RunConfig
    from woof.experiment import DomainConfig
    from tilestream.autoplan import Machine

    def big(grid_id):
        run = RunConfig(nx=2048, ny=2048, nz=49, dx=3000.0, dy=3000.0,
                        ztop=20000.0, dt=15.0, run_seconds=600.0,
                        grid_id=grid_id, specified=True)
        cfg = DomainConfig(
            grid_id=grid_id, parent_id=0, i_parent_start=1, j_parent_start=1,
            parent_grid_ratio=1, parent_time_step_ratio=1,
            history_interval_s=3600.0, run=run)
        return _types.SimpleNamespace(
            cfg=cfg, state=_types.SimpleNamespace(), parent=None)

    class _Streamed:
        """The least a builder may return: callable, and it has a store."""

        store = {}

        def __call__(self, *args, **kwargs):
            return None

    first, second = big(1), big(2)
    model = _types.SimpleNamespace(
        walk_parent_first=lambda: [first, second],
        nodes_by_grid_id={1: first, 2: second})
    machine = Machine(vram_bytes=8 << 30, host_bytes=512 << 30)
    decisions = {}
    streaming.steppers_for_tree(
        model, StreamingOptions(mode="auto"), machine=machine,
        decisions=decisions,
        builders={1: lambda *a, **k: _Streamed(),
                  2: lambda *a, **k: _Streamed()})

    assert set(decisions) == {1, 2}
    assert all(d.stream for d in decisions.values()), {
        g: d.reason for g, d in decisions.items()}
    for gid, d in decisions.items():
        assert "host_claim_bytes" in d.detail, (gid, d.detail)
        assert "host_spent_before_bytes" in d.detail, (gid, d.detail)
    first_claim = int(decisions[1].detail["host_claim_bytes"])
    # A streamed domain's pinned store is not free, and the SECOND domain
    # was priced only after it was subtracted.  Before the host ledger both
    # numbers were zero and each sibling saw the whole box.
    assert first_claim > 0, decisions[1].detail
    assert decisions[1].detail["host_spent_before_bytes"] == 0
    assert decisions[2].detail["host_spent_before_bytes"] == first_claim


def test_a_state_less_attachment_must_state_its_clock_policy():
    """Finding 1, CRITICAL: no bare-None clock on the store-direct road.

    ``store_domain_builder`` called ``attach(None, ...)``.  With no state,
    ``make_tile_hook``'s lazy ``domain_clock()`` returned ``None`` on the
    first buffer conversion, ``converted[id(tile_state)]`` latched, and
    every buffer for the rest of the forecast took the retired
    ``elapsed - interval.start`` recurrence instead of the bound
    ``DomainClock``'s ``dtbc`` -- the #219 one-timestep phase error, back
    on the road the LARGEST domains take and the one least likely to have
    a resident control beside it.

    Deriving nothing from nothing is indistinguishable from deciding to
    derive nothing, so the decision is a parameter now, and the seam
    refuses rather than defaults.
    """
    import inspect

    src = inspect.getsource(streaming.store_domain_builder)
    assert "ONE TIMESTEP LATE" in src
    attach_src = inspect.getsource(streaming.attach)
    assert "state is None and external_clock is DERIVE_CLOCK" in attach_src
    hook_src = inspect.getsource(streaming.make_tile_hook)
    assert "expected = domain_clock()" in hook_src
    assert "is not expected" in hook_src
    import pathlib

    route = (pathlib.Path(streaming.__file__).parents[1]
             / "prepared_single_domain_forecast.py").read_text(
                 encoding="utf-8")
    assert "bundle, clock=node.clock)" in route, (
        "the store-direct route must bind the node's own clock")


def test_the_tile_hook_binds_the_clock_it_was_given_with_no_state():
    """The behaviour behind the policy, exercised without a card.

    ``external_clock=`` binds THAT object on a state-less attachment, and
    a buffer that loses the binding is refused on its next launch rather
    than stepped one timestep out of phase with the domain.
    """
    import types as _types

    import woof.ingest.lateral_bc as lbc

    clock = object()
    bound = []
    real_attach = lbc.attach_streaming_lateral_boundaries
    real_bind = lbc.bind_lateral_boundary_clock

    def fake_attach(tile_state, lb):
        tile_state._lateral_boundary_device = _types.SimpleNamespace(
            streaming_external=True, clock=None, active_host_interval_id=7)

    def fake_bind(tile_state, c):
        bound.append(c)
        tile_state._lateral_boundary_device.clock = c

    lbc.attach_streaming_lateral_boundaries = fake_attach
    lbc.bind_lateral_boundary_clock = fake_bind
    try:
        hook = streaming.make_tile_hook(
            {0: "tables0", 1: "tables1"}, domain_state=None,
            external_clock=clock)
        buf = _types.SimpleNamespace()
        hook(buf, None, 0, None)
        assert bound == [clock], bound
        hook(buf, None, 1, None)
        assert buf._lateral_boundary_device.active_host_interval_id is None
        buf._lateral_boundary_device.clock = object()
        with pytest.raises(streaming.StreamingRefused) as refusal:
            hook(buf, None, 0, None)
        assert "out of phase" in str(refusal.value)
    finally:
        lbc.attach_streaming_lateral_boundaries = real_attach
        lbc.bind_lateral_boundary_clock = real_bind


# --- a PINNED tree is priced, and priced without a card ------------------


def _pinned_sibling_tree():
    """Two big streamed siblings under no parent, both tilings PINNED.

    The pinned twin of
    :func:`test_two_streamed_siblings_do_not_each_get_the_whole_host_budget`,
    with the planner taken out of it, on the FULL rung -- the only rungs
    this project has measured a radiation transient on are ``full`` and
    ``full+mynn+noahmp``, and a dry fixture would assert that the walk
    carries a reservation of zero, which is true whether or not the gate
    below it works.
    """
    import types as _types

    from woof.config import RunConfig
    from woof.experiment import DomainConfig

    def big(grid_id):
        run = RunConfig(nx=2048, ny=2048, nz=49, dx=3000.0, dy=3000.0,
                        ztop=20000.0, dt=15.0, run_seconds=600.0,
                        grid_id=grid_id, specified=True,
                        moist=True, mp_physics=10, ra_lw_physics=4,
                        ra_sw_physics=4, bl_pbl_physics=1, cu_physics=1,
                        sf_sfclay_physics=91, sf_surface_physics=2)
        cfg = DomainConfig(
            grid_id=grid_id, parent_id=0, i_parent_start=1, j_parent_start=1,
            parent_grid_ratio=1, parent_time_step_ratio=1,
            history_interval_s=3600.0, run=run)
        return _types.SimpleNamespace(
            cfg=cfg, state=_types.SimpleNamespace(), parent=None)

    return [big(1), big(2)]


def test_a_pinned_streamed_domain_carries_its_pinned_host_store():
    """Finding 5 again, on the road that never consulted the planner.

    ``decide``'s pinned short-circuit returned before either place that
    writes ``host_claim_bytes`` and passed no ``detail`` at all, so a
    pinned streamed domain claimed no host RAM as far as the walk was
    concerned.  Two of them therefore priced against the WHOLE box each --
    both plans accepted, both stores pinned, the second meeting
    ``cudaHostAlloc`` -- which is the identical failure the auto-path test
    above was written to prevent, re-opened on the pinned path.

    The number is not a second model of the store: it is the arithmetic
    ``streamed_envelope`` already runs for exactly this case, and the same
    ``store + arena`` that ``autoplan.Plan.host_bytes`` reports.
    """
    from tilestream.autoplan import Machine

    nodes = _pinned_sibling_tree()
    decisions: dict = {}
    streaming.decide_tree(
        nodes, StreamingOptions(mode="on", tile_nx=512, tile_ny=512,
                                nbuffers=2),
        machine=Machine(vram_bytes=8 << 30, host_bytes=512 << 30),
        decisions=decisions)

    assert set(decisions) == {1, 2}
    assert all(d.stream for d in decisions.values())
    for gid, d in decisions.items():
        assert "host_claim_bytes" in d.detail, (gid, d.detail)
        assert int(d.detail["host_claim_bytes"]) > 0, (gid, d.detail)
    first_claim = int(decisions[1].detail["host_claim_bytes"])
    assert decisions[1].detail["host_spent_before_bytes"] == 0
    assert decisions[2].detail["host_spent_before_bytes"] == first_claim


def test_a_pinned_tree_is_priced_without_probing_a_card(monkeypatch):
    """The whole point of the fix, stated as the two facts together.

    A pinned tiling asks no question, so it consults no planner and no
    card -- the law every bit-exactness gate rests on, held here by making
    ``Machine.detect`` fatal.  That was read as "and therefore it has no
    price", which is false: the once-per-process floor and the radiation
    reservation are ``autoplan.footprint_for`` reads that take a config
    and no machine, and the claims are window arithmetic.  So the walk
    prices a fully pinned tree with nothing detected, and the plan may
    stand in the forecast term.
    """
    from tilestream import autoplan

    def no_card(*args, **kwargs):
        raise AssertionError("a pinned tiling must probe no card")

    monkeypatch.setattr(autoplan.Machine, "detect", staticmethod(no_card))

    nodes = _pinned_sibling_tree()
    outcome = streaming.decide_tree(
        nodes, StreamingOptions(mode="on", tile_nx=512, tile_ny=512,
                                nbuffers=2))

    from tilestream.autoplan import footprint_for

    fp = footprint_for(nodes[0].cfg.run)
    assert outcome.priced is True
    # The two terms the walk used to zero, each against the rung's own
    # measurement rather than merely against zero.
    assert outcome.process_overhead_bytes == int(fp.process_overhead_bytes)
    assert outcome.radiation_transient_bytes == int(
        fp.radiation_transient_bytes) > 0
    assert outcome.vram_spent_bytes > 0
    assert outcome.host_spent_bytes > 0
    # No card was read, so there is no BUDGET -- and the walk says so
    # rather than inventing one.  That is preflight's question, asked
    # against its own budget downstream.
    assert outcome.total_budget_bytes == 0


def test_a_decided_off_domain_is_priced_resident_not_at_the_streamed_floor():
    """``_decision_claim_bytes`` honours what its own docstring promises.

    "A domain whose ``[tiles]`` is OFF ... is priced resident, which is
    what it is" -- but it reached that price through
    ``_minimum_claim_bytes``, whose "does the resident price fit what is
    left of the card" test is meaningless for a domain that cannot stream
    and fails outright against the zero budget a card-free walk carries.
    Every resident sibling then priced at the one-buffer STREAMED floor.

    Asserted against the footprint's own ``marginal_resident_bytes``,
    which is what "priced resident" means.
    """
    from tilestream import autoplan

    node = _pinned_sibling_tree()[0]
    cfg = node.cfg.run
    fp = autoplan.footprint_for(cfg)
    cells = int(cfg.nx) * int(cfg.ny) * int(cfg.nz)
    resident = int(fp.marginal_resident_bytes(cells))

    off = streaming.decide(cfg, StreamingOptions(mode="off"))
    assert off.stream is False and off.resident_bytes is None

    # An OFF domain has exactly one road, so its claim is a function of
    # the config alone -- which is also what lets a card-free walk price
    # it.  The helper no longer takes a budget at all.
    assert streaming._decision_claim_bytes(node, off) == resident


# --- Defect B: a mode that is AUTO must not carry the planner's answers --


@pytest.mark.parametrize("table,named", [
    ({"mode": "auto", "nbuffers": 2}, "nbuffers"),
    ({"mode": "auto", "tile_nx": 256, "tile_ny": 256}, "tile_nx"),
    ({"mode": "auto", "halo": 4}, "halo"),
])
def test_auto_with_a_pinned_key_is_refused(table, named):
    """A surface that is SET and silently overridden is a defect.

    MEASURED on this tip before the refusal: ``mode = "auto"`` with
    ``nbuffers = 1`` planned 2 buffers and warned about nothing -- the key
    reaches no callee, because ``decide`` reads ``options.nbuffers`` only
    inside the pinned short-circuit and hands ``autoplan.plan`` neither it
    nor ``max_nbuffers``.  On the four-domain tree this was found on,
    ``nbuffers = 2`` planned 3.  The receipt then records the request
    beside the outcome and reconciles nothing.

    ``tile_nx`` is the worse half: it does not merely constrain the plan,
    it overrides the VERDICT.  The pinned short-circuit returns before
    the planner is consulted at all, so ``mode = "auto"`` with a tiling
    streams a domain that FITS -- measured, 256x192x49 against a 32 GiB
    card: ``auto`` alone answers "the domain fits resident on this card",
    ``auto`` plus a 32x32 tiling streams it.  ``docs/public/TILES.md``
    says in as many words that "auto does not stream a domain that fits".

    REFUSED rather than honoured, because the doc already decides it:
    auto's documented product IS the planner's tiling, and the optional
    keys are documented under ``mode = "on"``.  The remedy that actually
    binds under auto is a smaller budget, and the message says so.
    """
    with pytest.raises(ValueError, match=named):
        StreamingOptions.from_mapping(table)


def test_the_auto_refusal_names_the_breakage_and_a_way_out():
    """The gate law: a refusal that names no breakage does not exist."""
    with pytest.raises(ValueError) as refusal:
        StreamingOptions.from_mapping({"mode": "auto", "nbuffers": 2})
    text = str(refusal.value)

    # The breakage, named.
    assert "SILENTLY IGNORE" in text and "nbuffers" in text
    # ...and named concretely enough to be checkable: the measurement, and
    # the promise auto breaks.
    assert "planning 3" in text
    assert "stream a domain that FITS" in text
    # The three ways out, so the refusal is actionable.
    assert "mode = 'on'" in text
    assert "vram_budget_bytes" in text
    assert "delete nbuffers" in text


# --------------------------------------------------------------------------
# the buffer's DERIVED setup arrays -- the FP32 geopotential residual
# --------------------------------------------------------------------------
#
# ``STATE_DERIVED_SETUP_ARRAYS`` (dphb_resid, dc3f, dc4f) was introduced
# OUTSIDE ``STATE_SETUP_ARRAYS`` on purpose: appending it there would change
# the restart setup digest's byte stream and reject every checkpoint written
# by an earlier release, for no information gain.  That is right for the
# digest and blind for the other three consumers of that tuple, which are
# the mechanisms that keep a streamed tile equal to its domain:
#
#   tilestream/driver.py :: geography_inventory   -- what a tile GATHERS
#   woof/core/streaming.py :: _impose_domain_setup -- what a tile INHERITS
#   tilestream/harness.py :: setup_arrays          -- the offline GATE
#
# A tile buffer is built by ``make_physics_state`` on ``neutral_geography``,
# so ``load_base`` fills its residual for ITS OWN base; the domain's phb is
# then imposed or gathered on top and the residual is left describing a
# different profile.  Measured before this was closed, nz=64: tile vs
# resident p differed on 1838/2048 elements at 6.13e-06 relative
# (hypsometric_opt=1) and 1826/2048 at 6.03e-06 (opt 2) -- an order of
# magnitude above the fixed EOS's own 4e-7, so the tiled run stopped
# reproducing the resident one.  Every in-checkout tilestream gate is
# analytic/flat, where the tile rebuilds the same base, so this was green
# everywhere it was tested and wrong on the real prepared road.


def _derived_setup_names():
    from woof.state_serialization_contract import STATE_DERIVED_SETUP_ARRAYS
    return STATE_DERIVED_SETUP_ARRAYS


class _SetupPair:
    """Minimal stand-ins for a domain state and a tile buffer."""

    def __init__(self, values):
        for name, value in values.items():
            setattr(self, name, value)


def test_the_buffer_inherits_the_derived_setup_arrays_it_cannot_rebuild():
    """1-D derived arrays must be IMPOSED, like the eta table beside them."""
    import numpy as np

    nz = 8
    names = _derived_setup_names()
    assert "dphb_resid" in names

    domain_values = {name: np.linspace(1.0, 2.0, nz) for name in names}
    tile_values = {name: np.zeros(nz) for name in names}
    # The 1-D setup the imposition already carried, so the walk has work to
    # do either way and this is not an empty-input pass.
    domain_values["znw"] = np.linspace(1.0, 0.0, nz + 1)
    tile_values["znw"] = np.zeros(nz + 1)

    state = _SetupPair(domain_values)
    tile = _SetupPair(tile_values)
    streaming._impose_domain_setup(tile, state)

    assert np.allclose(tile.znw, state.znw), "control: znw was always imposed"
    for name in names:
        assert np.allclose(getattr(tile, name), getattr(state, name)), (
            f"{name} was left describing the buffer's own base state")


def test_the_gather_carries_the_derived_setup_arrays_over_terrain():
    """3-D derived arrays must be GATHERED, like phb beside them."""
    import numpy as np

    pytest.importorskip("cupy")
    from tilestream.driver import geography_inventory

    nz, ny, nx = 4, 6, 5
    names = _derived_setup_names()
    values = {name: np.zeros((nz, ny, nx)) for name in names}
    values["phb"] = np.zeros((nz + 1, ny, nx))       # control: always gathered
    values["znw"] = np.zeros(nz + 1)                 # control: never gathered
    inventory = geography_inventory(_SetupPair(values))

    assert "setup/phb" in inventory, "control: phb is gathered"
    assert "setup/znw" not in inventory, "control: 1-D is not gathered"
    for name in names:
        assert f"setup/{name}" in inventory, (
            f"{name} varies horizontally over terrain and is not gathered, "
            "so the tile keeps its own")


def test_the_offline_setup_gate_can_see_the_derived_setup_arrays():
    """The gate that CHECKS the tile==domain claim must cover them too."""
    cp = pytest.importorskip("cupy")
    from tilestream.harness import setup_arrays

    nz = 4
    names = _derived_setup_names()
    values = {name: cp.zeros(nz) for name in names}
    values["znw"] = cp.zeros(nz + 1)                 # control
    seen = setup_arrays(_SetupPair(values))

    assert "znw" in seen, "control: the gate walks the setup tuple"
    for name in names:
        assert name in seen, (
            f"{name} is invisible to setup_window_mismatches, so nothing "
            "checks that the tile's copy equals the domain's")
