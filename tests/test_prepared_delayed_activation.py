"""Prepared children own their activation analysis, clock and first feedback."""
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from woof import prepared_domain_tree_forecast as runner
from woof.core.model import execute_experiment
from test_delayed_nest_activation import (  # noqa: F401 (autouse fixture)
    analysis_reflectivity_stand_in,
    DELAY_SECONDS, _Writers, _stashing_step, _tree)
from test_model import _Coupler, _HistoryState
from test_prepared_domain_tree_forecast import _sha, _synthetic_prepared_tree


def test_preflight_accepts_owned_dated_child_analysis(tmp_path, monkeypatch):
    prepared, receipt, config = _synthetic_prepared_tree(
        tmp_path, monkeypatch, delayed=True)
    inputs = runner.preflight_prepared_tree(
        prepared_root=prepared, preparation_receipt_sha256=_sha(receipt),
        experiment_config=config, experiment_config_sha256=_sha(config))
    assert inputs.experiment.domain_start_offset_exact(2) == 3600
    assert inputs.domains[1].cache_reader.header["metadata"]["user"]["initial_valid_time"] \
        == "2026-07-23T01:00:00"


@pytest.mark.parametrize("stamp", [None, "2026-07-23T00:00:00"])
def test_preflight_refuses_missing_or_root_dated_child_analysis(
        tmp_path, monkeypatch, stamp):
    prepared, receipt, config = _synthetic_prepared_tree(
        tmp_path, monkeypatch, delayed=True)
    header_path = prepared / "hierarchy-artifacts/domains/d02/prepared-cache/header.json"
    header = json.loads(header_path.read_text())
    header["metadata"]["user"]["initial_valid_time"] = stamp
    header_path.write_text(json.dumps(header))
    with pytest.raises(ValueError, match="initial_valid_time.*delayed start_time"):
        runner.preflight_prepared_tree(
            prepared_root=prepared, preparation_receipt_sha256=_sha(receipt),
            experiment_config=config, experiment_config_sha256=_sha(config))


def test_child_receipt_must_agree_with_owned_analysis():
    exp, model = _tree(delay_s=DELAY_SECONDS)
    child = model.node(2).cfg
    reader = SimpleNamespace(header={"metadata": {"user": {
        "initial_valid_time": exp.domain_start_time(2).isoformat()}}})
    with pytest.raises(ValueError, match="domain receipt valid_time"):
        runner._validate_delayed_prepared_time(
            exp, child, reader, {"valid_time": exp.start_time.isoformat()})


def test_delayed_child_refuses_a_moving_ancestor(monkeypatch):
    exp, _ = _tree(delay_s=DELAY_SECONDS)
    monkeypatch.setattr("woof.static.corridor.moving_grid_ids",
                        lambda exp: frozenset({1}))
    with pytest.raises(ValueError, match="moving ancestor d01"):
        runner._validate_delayed_prepared_geometry(exp)
    # Its own follower can start after the child exists; only the
    # preparation footprint's ancestors must remain fixed before birth.
    monkeypatch.setattr("woof.static.corridor.moving_grid_ids",
                        lambda exp: frozenset({2}))
    runner._validate_delayed_prepared_geometry(exp)


@pytest.mark.parametrize("feedback", [0, 1])
def test_prepared_activation_uses_exact_clock_and_shared_feedback(
        monkeypatch, feedback):
    from woof.runtime import _submit_tree_history_frame

    exp, model = _tree(delay_s=DELAY_SECONDS, smooth_option=2)
    exp = replace(exp, feedback=feedback)
    model._activation_context = {"experiment": exp}
    monkeypatch.setattr("woof.core.nest.NestCoupler", _Coupler)
    monkeypatch.setattr("woof.core.dycore.step", _stashing_step)
    calls = []
    analysis = _HistoryState()
    case = SimpleNamespace(initial_result=SimpleNamespace(state=analysis))

    def initialize(node, clock):
        calls.append((node.cfg.grid_id, clock.ticks,
                      node.parent.clock.ticks, node._started))
        return SimpleNamespace(grid=node.grid, state=analysis), case

    writers = _Writers()
    model._io_manager = writers
    execute_experiment(
        model, validate_state=False, delayed_child_initializer=initialize,
        skip_feedback_path=feedback == 0,
        history_handler=lambda _tree, node, ticks: _submit_tree_history_frame(
            writers, node, ticks))
    assert calls == [(2, DELAY_SECONDS, DELAY_SECONDS, False)]
    assert model.node(2)._started
    assert model.node(2).state is analysis
    assert model._prepared_by_grid_id[2] is case
    assert analysis.domain_start_offset == DELAY_SECONDS
    coupler = model.node(2).coupler
    assert (coupler.feedback, coupler.smooth_option) == (feedback, 2)
    if feedback:
        assert coupler.calls[:3] == ["prepare", "commit", "finalize"]
    else:
        assert coupler.calls[0] == "force"
        assert "commit" not in coupler.calls
    child_frames = [(ticks, refl) for gid, ticks, refl in writers.frames if gid == 2]
    assert [ticks for ticks, _ in child_frames] == [120, 180, 240, 300]
    # The birth frame writes the child's analysis reflectivity (S1, development branch
    # 1bce11263f: a newborn's first picture used to carry no REFL_10CM), and every later
    # frame the microphysics stash.
    assert all(refl is not None for _, refl in child_frames)


class _Radiation:
    """A radiation adapter: every one the schemes build defines ``__call__``."""

    def __init__(self, provider=None):
        self._ozone_provider = provider

    def __call__(self, *args, **kwargs):
        raise AssertionError("a startup build never radiates")


class _SlabTemplate:
    """The slab state a store restore classifies against; weak-referenceable."""

    def __init__(self):
        self.qv = np.zeros((2, 2, 4), dtype=np.float32)
        self.pb = np.ones((2, 2, 4), dtype=np.float32)
        self.physics = SimpleNamespace(mp_physics=10, refl_10cm=None,
                                       radiation_callable=_Radiation())


def _store_restore(cfg):
    """What restore_store_domain returns: a store bundle and its view.

    The view is the real :class:`CanonicalStoreState` over the bundle's
    store and geography, as ``StreamedChildReconstruction._facade`` builds
    it, and its slab template's physics carries a callable radiation, so a
    read of that radiation through the view refuses as it does on the card.
    """
    from woof.core.streamed_state import CanonicalStoreState
    from woof.core.streaming import REFL_STORE_KEY

    template = _SlabTemplate()
    store = {"qv": np.zeros((2, 8, 4), dtype=np.float32),
             REFL_STORE_KEY: np.zeros((8, 4), dtype=np.float32)}
    geography = {"setup/pb": np.ones((2, 8, 4), dtype=np.float32)}
    state = CanonicalStoreState(
        template, cfg, store=store, geography=geography,
        scalars={"elapsed_seconds": 0.0}, inventory={"qv": template.qv},
        geography_inventory={"setup/pb": template.pb})
    bundle = SimpleNamespace(store=store, geography=geography,
                             template=template)
    return bundle, SimpleNamespace(initial_result=SimpleNamespace(state=state))


def _attach_store(bundle, state, cfg, decision=None):
    """A real StreamedDomain over ``bundle``'s store, published as the route does."""
    from woof.core.streaming import (STREAMED_SCRATCH_ATTR, StreamedDomain,
                                      StreamingDecision, publish_store)
    from test_delayed_nest_activation import _StandInTiles

    tiles = _StandInTiles(cfg)
    tiles._home = bundle.store
    stream = StreamedDomain(
        tiles, decision or StreamingDecision(True, "stand-in", 8, 8, 2, 4),
        state=state, scalars={}, geography=bundle.geography,
        template=bundle.template)
    publish_store(state, stream)
    setattr(state, STREAMED_SCRATCH_ATTR,
            {key[8:]: value for key, value in bundle.store.items()
             if key.startswith("scratch/")})
    return stream


def test_a_store_restored_nest_starts_late_on_the_prepared_tree(monkeypatch):
    """The prepared domain-tree forecast: a streamed nest starts late.

    A nest streamed through a host store starts as a
    :class:`CanonicalStoreState` over its startup store, and a read of its
    radiation through that view refuses.  The executor's release read it,
    so the run stopped at the nest's start, hours in, with "resident
    methods cannot execute on a canonical host state".  Red at 8606dd377.
    Its route then restores the nest into a new store, and the startup
    view held the whole outgoing store through that restore (the owner's
    store release deleted only the names published on it), so the start
    held two copies of the nest; the slab template, allocated inside the
    reservation the restore allocates in next, stayed too.  Nothing of the
    startup store may be alive when the restore allocates, and the one
    stepper steps the restored nest to the end of the run.
    """
    import weakref

    from woof.runtime import _submit_tree_history_frame

    exp, model = _tree(delay_s=DELAY_SECONDS)
    model._activation_context = {"experiment": exp}
    monkeypatch.setattr("woof.core.nest.NestCoupler", _Coupler)
    monkeypatch.setattr("woof.core.dycore.step", _stashing_step)
    child = model.node(2)
    cfg = child.cfg.run
    bundle, restored = _store_restore(cfg)
    child.state = restored.initial_result.state
    owner = _attach_store(bundle, child.state, cfg)
    # As run_prepared_tree leaves its startup restore in the case map.
    model._prepared_by_grid_id = {2: SimpleNamespace(
        initial_result=restored.initial_result, streamed_store=bundle)}
    startup = {f"store {key}": weakref.ref(value)
               for key, value in bundle.store.items()}
    startup.update({f"geography {key}": weakref.ref(value)
                    for key, value in bundle.geography.items()})
    startup["slab template"] = weakref.ref(bundle.template)
    del bundle, restored
    alive_at_restore = []
    attached = []

    def restore():
        alive_at_restore.append(sorted(
            name for name, ref in startup.items() if ref() is not None))
        return _store_restore(cfg)

    def build(store_bundle, temporary):
        assert temporary is not child and temporary.cfg is child.cfg
        attached.append(temporary.state)
        return _attach_store(store_bundle, temporary.state, cfg,
                             owner.decision)

    def initialize(node, clock):
        # The route's own streamed branch, with the restore and the tile
        # build its cache and card would supply stood in.
        store_bundle, restored = runner._restore_streamed_child_at_start(
            owner, node, restore, build)
        return (SimpleNamespace(grid=node.grid,
                                state=restored.initial_result.state),
                SimpleNamespace(initial_result=restored.initial_result,
                                streamed_store=store_bundle))

    writers = _Writers()
    model._io_manager = writers
    execute_experiment(
        model, validate_state=False, steppers={2: owner},
        delayed_child_initializer=initialize,
        history_handler=lambda _tree, node, ticks: _submit_tree_history_frame(
            writers, node, ticks))

    assert alive_at_restore == [[]]
    rebuilt = model.node(2).state
    assert attached == [rebuilt]
    assert owner.state is rebuilt and rebuilt._streamed_domain is owner
    assert owner.steps == owner.tiled_run.sweeps > 0
    assert [ticks for gid, ticks, _refl in writers.frames if gid == 2] == [
        120, 180, 240, 300]


def test_a_nest_starts_beside_its_waiting_store_restored_child():
    """A starting nest's ozone walk passes a waiting store-restored child.

    On the prepared domain-tree forecast a nest can start while one of its
    own children, streamed through a host store, still waits for its
    start.  The walk that unlinks waiting children from the starting
    nest's outgoing radiation read that child's radiation through its
    :class:`CanonicalStoreState`, which refused, and the run stopped at
    the nest's start.  Red at 8606dd377.  The store-restored child is
    left as it is (its route gives it no parent ozone link), and a waiting
    resident sibling after it is still unlinked.
    """
    import gc
    import weakref

    from woof.core.model import _release_startup_build
    from woof.core.rrtmg_legacy import ParentOzoneProvider

    parent_radiation = _Radiation()
    outgoing = weakref.ref(parent_radiation)
    bundle, restored = _store_restore(SimpleNamespace(nx=4, ny=8, nz=2))
    streamed = SimpleNamespace(_started=False, children=[],
                               state=restored.initial_result.state)
    provider = ParentOzoneProvider(parent_radiation, registration=None)
    resident = SimpleNamespace(
        _started=False, children=[],
        state=SimpleNamespace(physics=SimpleNamespace(
            radiation_callable=_Radiation(provider))))
    node = SimpleNamespace(
        cfg=SimpleNamespace(grid_id=2), children=[streamed, resident],
        coupler=object(),
        state=SimpleNamespace(
            qv=np.ones((1,), dtype=np.float32),
            physics=SimpleNamespace(radiation_callable=parent_radiation,
                                    cumulus_callable=None)))
    del parent_radiation
    _release_startup_build(SimpleNamespace(_prepared_by_grid_id={}), node, {})
    gc.collect()

    assert outgoing() is None
    assert provider.released
    assert streamed.state.qv is bundle.store["qv"]
