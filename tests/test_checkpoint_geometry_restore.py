"""Checkpoint reconstruction must not transplant a cold-start footprint."""
from types import SimpleNamespace
from dataclasses import replace
import numpy as np
import pytest
from test_nest_relocation_staging import (_cpu_tree, _grandchild, _ramp_state,
                                          _initializer)
from woof.core.nest_relocation import relocate_child, RelocationRefusal


@pytest.mark.parametrize('restore', [False, True])
def test_disjoint_descendant_uses_route_checkpoint_rebuild(monkeypatch, restore):
    from woof import runtime
    from woof.ingest import relocation_init
    plane, parent, child = _cpu_tree()
    grand = _grandchild(child, ratio=3, nx=6, ny=6)
    grand.clock.spec = SimpleNamespace(start_ticks=0)
    grand.clock.tick_den = 1
    grand.state.elapsed_seconds = 0.0
    exp = SimpleNamespace(domains=(parent.cfg, child.cfg, grand.cfg), vertical=None)
    nodes = (parent, child, grand)
    old_static = {'LANDMASK': np.zeros((6, 6), dtype=np.float32)}
    new_static = {'LANDMASK': np.ones((6, 6), dtype=np.float32)}
    model = SimpleNamespace(
        _prepared_by_grid_id={3: SimpleNamespace(static_fields=old_static)},
        walk_parent_first=lambda: iter(nodes))
    corridor = SimpleNamespace(cache_sha256='0'*64, crop_at=lambda *args: new_static)
    calls = []
    def factory(**kwargs):
        def initialize(dc, parent_node, **extra):
            calls.append('geometry')
            static = kwargs['statics_builder'](None, dc)
            return SimpleNamespace(state=_ramp_state(6, 6, offset=12345),
                                   grid='restored-grid', static_fields=static)
        return initialize
    def driver(self, initialized, dc, parent_node, moved, **kwargs):
        assert moved == {}, 'checkpoint state must not come from the cold footprint'
        calls.append('physics')
        return 0.0
    monkeypatch.setattr(relocation_init, 'real_relocation_initializer', factory)
    monkeypatch.setattr(runtime.PreparedTreeRelocationChildPreparer, '_rebuild_driver', driver)
    reground = runtime.build_prepared_tree_descendant_regrounder(
        exp, model=model, corridors={3: corridor}, moving_roots={2})
    args = dict(i_parent_start=child.cfg.i_parent_start+4,
                j_parent_start=child.cfg.j_parent_start,
                initializer=_initializer(plane), static_provenance='synthetic corridor',
                state_digest=lambda state:'digest', staging='host',
                reground_descendant=reground)
    if not restore:
        with pytest.raises(RelocationRefusal, match='off its own old ground'):
            relocate_child(child, **args)
        assert not calls
        return
    receipt = relocate_child(child, restoring_checkpoint=True, **args)
    assert calls == ['geometry', 'physics']
    assert grand.grid == 'restored-grid'
    assert model._prepared_by_grid_id[3].static_fields is not old_static
    np.testing.assert_array_equal(model._prepared_by_grid_id[3].static_fields['LANDMASK'], 1)
    np.testing.assert_array_equal(grand.state.thp, _ramp_state(6, 6, offset=12345).thp)
    detail = receipt['descendants'][0]['reground']
    assert detail['donor_alignment']['pass'] is None
    assert detail['transplant']['checkpoint_reconstruction']


def test_checkpoint_preparer_does_not_change_live_preparer():
    from woof.runtime import RealRelocationChildPreparer
    live = object.__new__(RealRelocationChildPreparer)
    live._captured = {'cold': 'state'}
    live._plan_override = object()
    restore = live.for_checkpoint_restore()
    assert restore._restoring_checkpoint
    assert restore._captured is restore._plan_override is None
    assert not live._restoring_checkpoint
    assert live._captured == {'cold': 'state'}


def test_checkpoint_windows_use_full_footprint_latitude_and_no_cold_donors():
    from woof.runtime import RealRelocationChildPreparer
    live = object.__new__(RealRelocationChildPreparer)
    live._captured = None
    restore = live.for_checkpoint_restore()
    seen = []
    restore._rebuild_driver = lambda initial, dc, parent, land, **kw: seen.append((land, kw)) or 0.0
    grid = SimpleNamespace(cen_lat=25.0, latlon_mass=lambda:(np.ones((6,6))*25, None))
    footprint = SimpleNamespace(grid=grid, static_fields={})
    dc = SimpleNamespace(grid_id=3)
    prepare = restore.prepare_windows(dc, None, footprint)
    initial = SimpleNamespace(grid='slab', static_fields={}, state=SimpleNamespace())
    prepare(initial, dc, (slice(0,2), slice(0,6)))
    assert seen == [({}, {'center_lat':25.0})]
    assert prepare.precipitation_archive == {}
    assert restore._pending_refresh == (3, grid, {})


@pytest.mark.parametrize('restore', [False, True])
def test_disjoint_mover_rebuild_is_only_legal_before_checkpoint_load(restore):
    from test_nest_relocation_staging import _footprint_statics
    plane, parent, child = _cpu_tree()
    child.cfg = replace(child.cfg, run=replace(child.cfg.run, nx=6, ny=6))
    child.state = _ramp_state(6, 6, statics=_footprint_statics(plane, child.cfg))
    args = dict(i_parent_start=child.cfg.i_parent_start+4,
                j_parent_start=child.cfg.j_parent_start,
                initializer=_initializer(plane), static_provenance='synthetic corridor',
                state_digest=lambda state:'digest', staging='host')
    if not restore:
        with pytest.raises(RelocationRefusal, match='shares no cell'):
            relocate_child(child, **args)
        return
    receipt = relocate_child(child, restoring_checkpoint=True, **args)
    assert receipt['donor_alignment']['pass'] is None
    assert receipt['parent_bitwise_unchanged']
    assert receipt['transplant']['checkpoint_reconstruction']
    np.testing.assert_array_equal(child.state.thp, _ramp_state(6, 6, offset=5e5).thp)
