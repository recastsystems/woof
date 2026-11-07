"""Two-way nest transactions through resident ranked slabs."""
from __future__ import annotations

import gc
import json
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.gpu


def _trajectory(monkeypatch, *, mode, control=None, steps=6, feedback=1):
    import cupy as cp
    from woof.core import nest_stream, streaming
    from woof.core.devices import DeviceOptions
    from woof.core.model import execute_experiment
    from woof.core.nest import NestCoupler
    from woof.ingest.lateral_bc import bind_lateral_boundary_clock
    from woof.state_digest import canonical_state_digest
    from tilestream import driver, gather, physics_inventory
    from tilestream import test_moving_nest as moving
    from tilestream import test_nest_executor as executor

    monkeypatch.setattr(executor, "I_PARENT_START", 41)
    monkeypatch.setattr(executor, "J_PARENT_START", 41)
    monkeypatch.setattr(executor, "PARENT_DT", 6.)
    cfg = moving.parent_cfg(nx=96, ny=96, nz=24, dt=6.,
                            run_seconds=steps*6., radt_minutes=.5,
                            cudt_minutes=.5)
    child_cfg = moving.child_cfg(cfg, nx=48, ny=48)
    boundaries = moving.domain_boundaries(cfg, seconds=21600.)
    state, geography = moving.build_parent(cfg, boundaries=boundaries)
    experiment = executor.experiment(cfg, child_cfg, steps=steps)
    model = executor.assemble(experiment, state, geography.grid, feedback=feedback)
    # The product root binds its external Davies recurrence to the domain
    # clock. The idealized assembler deliberately leaves that choice open.
    bind_lateral_boundary_clock(model.root.state, model.root.clock)
    executor.initial_feedback(model)
    steppers = {}
    cards = tuple(range(min(2, cp.cuda.runtime.getDeviceCount())))
    options = DeviceOptions(count=2, grid=(1, 2),
                            ids=(cards[0], cards[-1]))
    for node in model.walk_parent_first():
        # Both storage roads own the same primed frame stash, as the
        # prepared-domain product does before its first output frame.
        streaming.prime_lazy_carriers(node.state, node.cfg.run)
        gid = int(node.cfg.grid_id)
        if mode == "resident" or (mode == "nest" and gid == 1):
            continue
        take = streaming.streamed_store_inventory()
        store = {}
        for name, value in take(node.state).items():
            host = cp.asnumpy(value)
            store[name] = gather.pinned_empty(host.shape, host.dtype)
            store[name][...] = host
        bundle = SimpleNamespace(
            store=store, scalars=physics_inventory.carrier_scalars(node.state),
            geography=driver.geography_store(node.state, host=True),
            boundaries=boundaries if gid == 1 else None, template=node.state)
        decision = streaming.ranked_decision(node.cfg.run, options)
        owner = streaming.ranked_domain_builder(
            bundle, clock=node.clock, options=options, node=node,
            check_geography=False)(None, node.cfg.run, decision)
        owner._state = node.state
        node.state._streamed_domain = owner
        steppers[gid] = owner

    force_reads = []
    original_force = NestCoupler.force
    def measured_force(coupler, node):
        before = {gid: dict(owner.tiled_run.output_report)
                  for gid, owner in steppers.items()}
        original_force(coupler, node)
        for gid, owner in steppers.items():
            report = owner.tiled_run.output_report
            row = dict(grid=gid, full_drains=report['full_drains']-before[gid]['full_drains'],
                       window_reads=report['window_reads']-before[gid]['window_reads'],
                       window_read_bytes=report['window_read_bytes']-before[gid]['window_read_bytes'])
            force_reads.append(row)
            assert row['full_drains'] == 0, row
    monkeypatch.setattr(NestCoupler, 'force', measured_force)

    if control == "stale_force":
        original = nest_stream._copy_owned_sides
        def stale_tables(specs, source):
            if source.rolling_generation > 1:
                return 0
            return original(specs, source)
        monkeypatch.setattr(nest_stream, "_copy_owned_sides", stale_tables)
    elif control == "no_feedback":
        monkeypatch.setattr(NestCoupler, "_restrict_windowed", lambda *a: None)
    elif control is not None:
        raise ValueError(control)
    try:
        result = execute_experiment(model, steppers=steppers, validate_state=True,
                                    pool_trim_per_period=False)
        if mode == 'both' or (steppers and not feedback and steps > 1):
            assert sum(row['window_reads'] for row in force_reads) > 0
        print('RANKS FORCE READS ' + json.dumps(dict(
            mode=mode, feedback=feedback, calls=force_reads,
            output_road={gid: dict(owner.tiled_run.output_report)
                         for gid, owner in steppers.items()})), flush=True)
        domains = {}
        for node in model.walk_parent_first():
            owner = steppers.get(int(node.cfg.grid_id))
            source = node.state if owner is None else owner.store
            digest, fields = moving.carrier_digest(source)
            canonical = (canonical_state_digest(node.state, node.clock)
                         if owner is None else owner.canonical_digest(node.clock))
            domains[int(node.cfg.grid_id)] = dict(
                digest=digest, fields=fields, canonical=canonical)
        output = dict(domains=domains, cadence=moving.cadence_census(model, steppers),
                      steps=result.steps, forces=result.forces,
                      feedback=model.node(2).coupler.feedback_count)
        print("RANKS NEST ARM " + json.dumps(dict(
            mode=mode, control=control, cards=list(options.device_ids()),
            steps=output["steps"], forces=output["forces"],
            feedback=output["feedback"], domains={
                gid: dict(sha256=row["digest"], arrays=len(row["fields"]),
                          canonical_sha256=row["canonical"]["sha256"],
                          canonical_arrays=row["canonical"]["array_count"])
                for gid, row in domains.items()})), flush=True)
        return output
    finally:
        for owner in steppers.values():
            owner.tiled_run.close()
        del model, state, steppers
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()


def test_ranked_tree_matches_resident_and_detects_missing_transactions(monkeypatch):
    with monkeypatch.context() as arm:
        resident = _trajectory(arm, mode="resident")
    for mode in ("both", "nest"):
        with monkeypatch.context() as arm:
            actual = _trajectory(arm, mode=mode)
        for gid in (1, 2):
            fields = actual["domains"][gid]["fields"]
            reference = resident["domains"][gid]["fields"]
            differing = sorted(key for key in fields.keys() | reference.keys()
                               if fields.get(key) != reference.get(key))
            assert not differing, (mode, gid, differing)
            names = actual["domains"][gid]["canonical"]["field_order"]
            reference_names = resident["domains"][gid]["canonical"]["field_order"]
            assert names == reference_names, (mode, gid, sorted(set(names)^set(reference_names)))
        assert actual == resident
    assert resident["forces"] == 6 and resident["feedback"] == 7
    for control, affected in (("stale_force", 2), ("no_feedback", 1)):
        with monkeypatch.context() as arm:
            faulty = _trajectory(arm, mode="both", control=control)
        assert faulty["domains"][affected]["digest"] != resident["domains"][affected]["digest"], control
        assert faulty["forces"] == resident["forces"]
        assert faulty["feedback"] == resident["feedback"]


def test_one_way_ranked_force_matches_resident_without_full_drains(monkeypatch):
    with monkeypatch.context() as arm:
        resident = _trajectory(arm, mode='resident', feedback=0, steps=3)
    for mode in ('both', 'nest'):
        with monkeypatch.context() as arm:
            actual = _trajectory(arm, mode=mode, feedback=0, steps=3)
        assert actual == resident


@pytest.mark.parametrize('periodic', [False, True])
@pytest.mark.parametrize('grid', [(1, 2), (2, 2)])
def test_rank_window_words_match_scatter_with_poisoned_halos(periodic, grid):
    import numpy as np
    import cupy as cp
    from tilestream import gather, multigpu
    from tilestream.ranks import RankedRun

    run = RankedRun.__new__(RankedRun)
    run.nz, run.ny, run.nx = 3, 19, 23
    run.specs = multigpu.plan_split(run.nx, run.ny, 2, gx=grid[1], gy=grid[0],
                                    periodic=periodic)
    run.devices = [int(cp.cuda.Device().id)] * len(run.specs)
    run.compute_streams = [cp.cuda.Stream(non_blocking=True) for _ in run.specs]
    run._closed, run._ahead, run._exposed = False, True, False
    run.global_keys = ()
    run._home = {}
    run.tiles = []
    run.output_report = dict(window_reads=0, window_read_bytes=0, window_read_seconds=0.,
                             full_drains=0, full_gathers=0)
    shapes = dict(mup=(19, 23), u=(3, 19, 24), v=(3, 20, 23),
                  php=(4, 19, 23), thp=(3, 19, 23), layers=(2, 3, 19, 23))
    reference = {}
    for name, shape in shapes.items():
        key = f'state/{name}'
        run._home[key] = np.zeros(shape, np.float32)
        reference[key] = np.empty(shape, np.float32)
    for rank, spec in enumerate(run.specs):
        tile = {}
        for key, meta in run._home.items():
            kind = gather.classify(meta.shape, run.nz, run.ny, run.nx, layers_ok=True)
            shape = gather.tile_shape_for(kind, meta.shape, spec.cny, spec.cnx)
            values = np.full(shape, np.float32(-99999.))
            for src, dst in gather.segments_for(spec, 'scatter', kind):
                sy, sx = src[-2:]
                block_shape = (*shape[:-2], sy.stop-sy.start, sx.stop-sx.start)
                # Include arbitrary float words, signed zero and NaN payloads.
                words = np.arange(np.prod(block_shape), dtype=np.uint32).reshape(block_shape)
                words = words * np.uint32(2654435761) + np.uint32(rank+1)
                words.flat[0] = np.uint32(0x80000000 if rank % 2 else 0x7fc00019)
                values[(..., sy, sx)] = words.view(np.float32)
            for src, dst in gather.segments_for(spec, 'scatter', kind):
                reference[key][dst] = values[src]
            tile[key] = cp.asarray(values)
        run.tiles.append(tile)
    run.inventory_fn = lambda tile, names: {key: tile[key] for key in names if key in tile}
    cp.cuda.get_current_stream().synchronize()
    total = 0
    requests = []
    for key, expected in reference.items():
        ny, nx = expected.shape[-2:]
        for window in ((slice(0, ny), slice(0, 1)),
                       (slice(ny-2, ny), slice(nx-3, nx)),
                       (slice(8, 12), slice(10, 15))):
            requests.append((key, window))
            with cp.cuda.Stream(non_blocking=True):
                actual = run.read_window(key, window)
            assert actual.flags.c_contiguous and actual.dtype == expected.dtype
            assert cp.asnumpy(actual).tobytes() == expected[(...,)+window].tobytes()
            total += actual.nbytes
            assert run._ahead and not run._exposed
            assert not np.any(run._home[key])
    assert run.output_report['window_read_bytes'] == total
    assert run.output_report['window_reads'] == len(shapes)*3
    assert run.output_report['full_drains'] == run.output_report['full_gathers'] == 0
    from tilestream.window_batch import token
    coordinator = cp.cuda.Stream(non_blocking=True)
    for attempt in range(3):
        if attempt == 2:
            coordinator = cp.cuda.Stream(non_blocking=True)
        if attempt:
            # Physics may replace arrays between intervals. Cached geometry
            # must never retain old inventory pointers.
            for tile in run.tiles:
                for key in tile:
                    tile[key] = tile[key].copy()
            cp.cuda.get_current_stream().synchronize()
        with coordinator:
            batch = run.read_windows(requests + requests)
        coordinator.synchronize()
        for key, window in requests:
            assert cp.asnumpy(batch[token(key, window)]).tobytes() == reference[key][(...,)+window].tobytes()
        assert run._ahead and not run._exposed
        assert run.output_report['full_drains'] == run.output_report['full_gathers'] == 0
    assert run.output_report['window_reads'] == len(shapes)*12
    assert run.output_report['window_read_bytes'] == total*4
    window = (slice(1, 3), slice(2, 5))
    before = dict(run.output_report)
    run._exposed = True
    assert run.read_window('state/mup', window) is None
    assert run.read_windows([('state/mup', window)]) == {}
    assert run._exposed and run._ahead
    run._exposed, run._ahead = False, False
    assert run.read_window('state/mup', window) is None
    assert run.read_windows([('state/mup', window)]) == {}
    run._ahead = True
    run._home['state/host_only'] = np.ones((19, 23), np.float32)
    assert run.read_window('state/host_only', window) is None
    assert run.read_windows([('state/host_only', window)]) == {}
    assert run.read_window('setup/mub2d', window) is None
    assert run.output_report == before
    print('RANK WINDOW WORDS ' + json.dumps(dict(periodic=periodic, grid=grid,
                                               output_road=run.output_report)), flush=True)


@pytest.mark.parametrize('base3d', [False, True])
def test_ranked_transition_windows_match_resident_words(base3d):
    import numpy as np
    import cupy as cp
    from tilestream import gather, multigpu
    from tilestream.ranks import RankedRun
    from woof.core.nest_operands import NestWindowSource
    from test_milbrandt_nest_edge_gpu import _parent_state, _contract

    state = _parent_state()
    if base3d:
        state.thb = cp.ascontiguousarray(cp.broadcast_to(state.thb[:, None, None], state.qv.shape))
    run = RankedRun.__new__(RankedRun)
    run.nz, run.ny, run.nx = state.qv.shape
    run.specs = multigpu.plan_split(run.nx, run.ny, 0, gx=2, gy=1, periodic=False)
    run.devices = [int(cp.cuda.Device().id)] * 2
    run.compute_streams = [cp.cuda.Stream(non_blocking=True) for _ in run.specs]
    run._closed, run._ahead, run._exposed = False, True, False
    run.global_keys = ()
    carried = {f'state/{name}': value for name, value in vars(state).items()
               if isinstance(value, cp.ndarray) and value.ndim >= 2
               and name not in ('mub2d', 'thb')}
    run._home = {key: np.full(value.shape, -99., np.float32) for key, value in carried.items()}
    run.tiles = []
    for spec in run.specs:
        tile = {}
        for key, value in carried.items():
            kind = gather.classify(value.shape, run.nz, run.ny, run.nx)
            slab = cp.empty(gather.tile_shape_for(kind, value.shape, spec.cny, spec.cnx),
                            dtype=value.dtype)
            for src, dst in gather.segments_for(spec, 'gather', kind):
                slab[dst] = value[src]
            tile[key] = slab
        run.tiles.append(tile)
    run.inventory_fn = lambda tile, names: {key: tile[key] for key in names if key in tile}
    run.output_report = dict(window_reads=0, window_read_bytes=0, window_read_seconds=0.,
                             full_drains=0, full_gathers=0)
    geography = {'setup/mub2d': cp.asnumpy(state.mub2d)}
    if base3d:
        geography['setup/thb'] = cp.asnumpy(state.thb)
    def no_store():
        raise AssertionError('a ranked transition materialized the whole store')
    owner = SimpleNamespace(tiled_run=run, read_store=no_store, _geography=geography,
                            template_state=SimpleNamespace(thb=state.thb, c1h=state.c1h,
                                                           c2h=state.c2h))
    ranked = NestWindowSource(SimpleNamespace(_streamed_domain=owner), read_only=True)
    resident = NestWindowSource(state, read_only=True)
    window = (slice(1, 4), slice(0, 2))
    cp.cuda.get_current_stream().synchronize()
    from woof.core.microphysics_transition import edge_parent_planes
    kinds = ('qv', 'qc', 'qi', 'qg', 'qh', 'nc', 'nr', 'ni', 'nh')
    expected = {kind: cp.asnumpy(resident.transitioned(_contract(), kind, window))
                for kind in kinds}
    requests = [(name, window) for name in edge_parent_planes() + ('thb', 'c1h', 'c2h')]
    resident.prepare(requests)
    ranked.prepare(requests)
    for kind in kinds:
        actual = cp.asnumpy(ranked.transitioned(_contract(), kind, window))
        assert actual.tobytes() == expected[kind].tobytes(), kind
        prepared = cp.asnumpy(resident.transitioned(_contract(), kind, window))
        assert prepared.tobytes() == expected[kind].tobytes(), kind
    assert run.output_report['window_reads'] > 0
    assert run.output_report['full_drains'] == run.output_report['full_gathers'] == 0
    assert run._ahead and not run._exposed


if __name__ == "__main__":
    with pytest.MonkeyPatch.context() as patch:
        test_ranked_tree_matches_resident_and_detects_missing_transactions(patch)
    print("RANKS NEST PASS")
