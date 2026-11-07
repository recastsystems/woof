"""Live host stores supply bounded nest operands without resident shadows."""
from types import SimpleNamespace

import numpy as np
import pytest

from woof.core.nest_interp import (
    bdy_interp1, copy_fcn, feedback_child_window, feedback_parent_bounds,
    register_nest, window_registration)
from woof.core.nest_operands import NestWindowSource, boundary_windows


def _host_facade(host):
    prognostics = {"mup", "u", "v", "w", "thp", "php", "qv", "qc", "qr",
                   "qi", "qs", "qg", "nr", "ni", "ns", "ng", "p", "al", "alt",
                   "p_perturbation"}
    store, geography, metadata = {}, {}, {}
    for name, value in vars(host).items():
        if name in prognostics:
            store[f"state/{name}"] = value.copy()
        elif isinstance(value, np.ndarray) and value.ndim >= 2:
            geography[f"setup/{name}"] = value.copy()
        else:
            metadata[name] = value
    owner = SimpleNamespace(store=store, _geography=geography,
                            template_state=SimpleNamespace(**metadata),
                            decision=SimpleNamespace(tile_ny=7, tile_nx=8))
    facade = SimpleNamespace(_streamed_domain=owner, **metadata)
    for key, value in store.items():
        setattr(facade, key.split("/", 1)[1], value)
    return facade


def test_canonical_source_rejects_horizontal_template_fallback():
    template = SimpleNamespace(mup=np.zeros((2, 3)), c1h=np.ones(4))
    owner = SimpleNamespace(store={}, _geography={}, template_state=template)
    source = NestWindowSource(SimpleNamespace(_streamed_domain=owner))
    assert source.array("c1h") is template.c1h
    with pytest.raises(RuntimeError, match="slab template"):
        source.array("mup")


def test_canonical_source_keeps_numpy_setup_scalars_on_host():
    top = np.float32(5717.7817)
    source = NestWindowSource(SimpleNamespace(p_top=top, has_msf=np.bool_(True)))
    assert source.device("p_top") is top
    assert source.device("has_msf") == np.bool_(True)


def _numpy_device(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, 'cupy', SimpleNamespace(
        ndarray=np.ndarray, float32=np.float32, asarray=np.asarray,
        ascontiguousarray=np.ascontiguousarray, empty=np.empty, asnumpy=np.asarray))


@pytest.mark.parametrize('fallback', [None, 'exposed', 'host_current', 'missing_slab'])
def test_read_only_rank_source_is_lazy_and_preserves_host_truth(monkeypatch, fallback):
    _numpy_device(monkeypatch)
    calls = []
    truth = np.arange(42, dtype=np.float32).reshape(6, 7)
    host = np.full_like(truth, 113.)
    geography = np.full_like(truth, 3.)
    run = SimpleNamespace(ranked=True, raw_store={'state/mup': host})
    def read_window(key, window):
        calls.append(('window', key))
        if fallback is not None or key not in run.raw_store:
            return None
        return truth[(...,)+window].copy()
    run.read_window = read_window
    def read_store():
        calls.append(('store',))
        return run.raw_store
    owner = SimpleNamespace(tiled_run=run, read_store=read_store,
                            _geography={'setup/mub2d': geography},
                            template_state=SimpleNamespace(c1h=np.ones(2, np.float32)))
    source = NestWindowSource(SimpleNamespace(_streamed_domain=owner), read_only=True)
    assert not calls
    assert source.shape('mup') == truth.shape
    assert not calls
    window = (slice(1, 4), slice(2, 6))
    assert np.array_equal(source.device('mub2d', window), geography[(...,)+window])
    assert ('store',) not in calls
    actual = source.device('mup', window)
    expected = host if fallback else truth
    assert actual.tobytes() == expected[(...,)+window].tobytes()
    assert calls.count(('store',)) == int(fallback is not None)
    assert np.all(host == 113.)
    with pytest.raises(RuntimeError, match='read-only'):
        source.write('mup', window, actual)


@pytest.mark.parametrize('writable', [False, True])
def test_rank_read_store_keeps_prior_writes_and_avoids_read_only_gather(monkeypatch, writable):
    from contextlib import nullcontext
    import sys
    from tilestream.ranks import RankedRun
    from tilestream import gather

    monkeypatch.setitem(sys.modules, 'cupy', SimpleNamespace(
        cuda=SimpleNamespace(Device=lambda dev: nullcontext())))
    monkeypatch.setitem(sys.modules, 'tilestream.sfire', SimpleNamespace(
        configure_tile=lambda *args: None))
    run = RankedRun.__new__(RankedRun)
    run._closed, run._ahead, run._exposed = False, True, False
    run._home = {'state/mup': np.zeros((2, 3), np.float32)}
    run.tiles = [{'state/mup': np.arange(6, dtype=np.float32).reshape(2, 3)}]
    run.devices, run.specs, run.compute_streams = [0], [None], [nullcontext()]
    run.nz, run.inventory_fn = 1, None
    run.output_report = dict(full_drains=0, full_gathers=0, full_drain_seconds=0.)
    for method in ('_await_download', '_guard_store', 'sync_all', 'refresh_arrays', 'sync_compute'):
        setattr(run, method, lambda: None)
    def scatter(tile, home, *args, **kwargs):
        home['state/mup'][...] = tile['state/mup']
    def upload(home, tile, *args, **kwargs):
        tile['state/mup'][...] = home['state/mup']
    monkeypatch.setattr(gather, 'scatter_tile', scatter)
    monkeypatch.setattr(gather, 'gather_tile', upload)
    if writable:
        run.drain()
        run._home['state/mup'][0, 0] = 123.
    snapshot = run.read_store()
    expected = np.arange(6, dtype=np.float32).reshape(2, 3)
    if writable:
        expected[0, 0] = 123.
    assert snapshot['state/mup'].tobytes() == expected.tobytes()
    assert run._exposed == writable
    run._gather_exposed()
    assert run.tiles[0]['state/mup'].tobytes() == expected.tobytes()
    assert run.output_report['full_gathers'] == int(writable)
    assert run.output_report['full_drains'] == 1


def test_canonical_nest_flags_come_from_complete_geography(monkeypatch):
    _numpy_device(monkeypatch)
    geography = {'setup/msft': np.ones((4, 6), np.float32),
                 'setup/f': np.zeros((4, 6), np.float32)}
    geography['setup/msft'][0, 0] = 1.25
    template = SimpleNamespace(has_msf=False, rotational=False)
    owner = SimpleNamespace(store={}, _geography=geography, template_state=template)
    source = NestWindowSource(SimpleNamespace(_streamed_domain=owner))
    assert source.array('has_msf') and source.array('rotational')
    assert source.device('has_msf') and source.device('rotational')
    assert not template.has_msf and not template.rotational


@pytest.mark.parametrize('owners,expected', [
    ((True, False), (7, 8)), ((False, True), (3, 4)), ((True, True), (3, 4)),
])
def test_transaction_chunks_use_the_actual_streamed_endpoint(owners, expected):
    from woof.core.nest import NestCoupler
    from test_nest_coupler import _nodes
    parent, child = _nodes()
    if owners[0]:
        parent.state = _host_facade(parent.state)
    if owners[1]:
        child.state = _host_facade(child.state)
        child.state._streamed_domain.decision = SimpleNamespace(tile_ny=3, tile_nx=4)
    coupler = NestCoupler(child)
    assert coupler._transaction_chunk_shape() == expected
    assert coupler._feedback_child_is_bounded() == owners[1]


@pytest.mark.parametrize('parent_streams', [False, True])
def test_force_dispatch_and_transfer_counts_include_parent_only_owner(monkeypatch, parent_streams):
    from woof.core import nest as nest_mod, nest_operands, cam_ozone
    from test_nest_coupler import _nodes
    parent, child = _nodes()
    if parent_streams:
        parent.state = _host_facade(parent.state)
    coupler = nest_mod.NestCoupler(child)
    calls = []
    class Source:
        def __init__(self, state, *, read_only=False):
            assert read_only
            self.host_to_device_bytes = 11 if hasattr(state, '_streamed_domain') else 0
            self.device_to_host_bytes = 0
    monkeypatch.setattr(nest_operands, 'NestWindowSource', Source)
    monkeypatch.setattr(coupler, '_bind_geometry', lambda: None)
    monkeypatch.setattr(cam_ozone, 'transfer_parent_ozone', lambda *args: 0)
    monkeypatch.setattr(nest_mod, 'attach_nest_boundaries', lambda *args, **kwargs: None)
    monkeypatch.setattr(coupler, '_prepare_force_sources', lambda *args: None)
    monkeypatch.setattr(coupler, '_force_windowed',
                        lambda kind, *args: calls.append(('window', kind)))
    monkeypatch.setattr(coupler, '_coupled_parent_field',
                        lambda kind: calls.append(('resident', kind)))
    monkeypatch.setattr(coupler, '_coupled_child_field', lambda *args, **kwargs: None)
    monkeypatch.setattr(nest_mod, 'bdy_interp1', lambda *args, **kwargs: kwargs['out'])
    coupler.force(child)
    assert calls and {route for route, _ in calls} == {
        'window' if parent_streams else 'resident'}
    assert coupler.force_sync_bytes == (11 if parent_streams else 0)
    assert coupler.force_count == 1


@pytest.mark.parametrize('transition', [False, True])
def test_force_window_uses_exact_donor_chunks_for_parent_only_owner(monkeypatch, transition):
    from woof.core import nest as nest_mod
    from test_nest_coupler import _nodes
    parent, child = _nodes()
    parent.state = _host_facade(parent.state)
    coupler = nest_mod.NestCoupler(child)
    calls = []
    class Source:
        def coupled(self, kind, window):
            calls.append(('coupled', window))
            return np.zeros((2, window[0].stop-window[0].start,
                             window[1].stop-window[1].start), np.float32)
        def transitioned(self, contract, kind, window):
            calls.append(('mapped', window))
            return np.zeros((2, window[0].stop-window[0].start,
                             window[1].stop-window[1].start), np.float32)
    monkeypatch.setattr(nest_mod, 'transition_handles_field', lambda *args: transition)
    def interpolate(parent_field, child_field, cropped, **kwargs):
        side = kwargs['sides'][0]
        shape = (2, cropped.nyc, 5) if side in ('west', 'east') else (2, 5, cropped.nxc)
        return {side: (np.ones(shape, np.float32), np.zeros(shape, np.float32))}
    monkeypatch.setattr(nest_mod, 'bdy_interp1', interpolate)
    output = coupler._rolling_out('t')
    coupler._force_windowed('t', output, Source(), Source())
    expected = []
    for _, window, _ in boundary_windows(coupler.registrations['m'], 5, (7, 8)):
        _, donor = window_registration(coupler.registrations['m'], window)
        expected.extend([('mapped' if transition else 'coupled', donor), ('coupled', window)])
    assert calls == expected
    assert all(np.all(value == 1.) and np.all(tendency == 0.)
               for value, tendency in output.values())


def test_transitioned_source_copies_only_canonical_donor_windows(monkeypatch):
    from woof.core import microphysics_transition as edge
    from test_nest_coupler import _nodes
    _numpy_device(monkeypatch)
    parent, _ = _nodes()
    parent.state.alt = np.full(parent.state.qv.shape, 2., np.float32)
    parent.state.p = np.full(parent.state.qv.shape, 90000., np.float32)
    source = NestWindowSource(_host_facade(parent.state))
    window = (slice(2, 5), slice(3, 7))
    seen = []
    def launch(contract, local, kind, *, out, coupled):
        assert coupled and kind == 'qi'
        for name in edge.edge_parent_planes():
            value = getattr(local, name)
            if value is not None:
                assert value.shape[-2:] == (3, 4)
                assert value.flags.c_contiguous
        seen.append(local)
        out.fill(17.)
        return out
    monkeypatch.setattr(edge, 'launch_microphysics_edge_field', launch)
    result = source.transitioned(object(), 'qi', window)
    assert len(seen) == 1 and result.shape == (2, 3, 4) and np.all(result == 17.)
    expected_bytes = 0
    for name in edge.edge_parent_planes() + ('thb', 'c1h', 'c2h'):
        value = source.array(name)
        if isinstance(value, np.ndarray):
            expected_bytes += (value[(...,)+window] if value.ndim >= 2 else value).nbytes
    assert source.host_to_device_bytes == expected_bytes
    assert source.max_operand_bytes < parent.state.qv.nbytes


def test_restriction_chunks_accept_an_already_mapped_resident_child(monkeypatch):
    from woof.core import nest as nest_mod
    from test_nest_coupler import _nodes
    _numpy_device(monkeypatch)
    parent, child = _nodes()
    parent.state = _host_facade(parent.state)
    coupler = nest_mod.NestCoupler(child)
    source, target = NestWindowSource(child.state), NestWindowSource(parent.state)
    target.array('qi').fill(0.)
    mapped = np.full(child.state.qi.shape, 19., np.float32)
    monkeypatch.setattr(source, 'raw', lambda *args: pytest.fail('mapped species read raw child'))
    seen = []
    def restrict(result, value, reg, **kwargs):
        assert np.all(value == 19.)
        seen.append(kwargs['parent_window'])
        result.fill(19.)
    monkeypatch.setattr(nest_mod, 'copy_fcn', restrict)
    coupler._restrict_windowed('qi', source, target, mapped=mapped)
    assert seen
    assert all(win[0].stop-win[0].start <= 2 and win[1].stop-win[1].start <= 2
               for win in seen)
    ilo, ihi, jlo, jhi = feedback_parent_bounds(coupler.registrations['m'])
    expected = np.zeros_like(target.array('qi'))
    expected[:, jlo:jhi+1, ilo:ihi+1] = 19.
    np.testing.assert_array_equal(target.array('qi'), expected)
    assert source.host_to_device_bytes > 0 and target.device_to_host_bytes > 0


def test_parent_only_feedback_keeps_resident_child_reverse_mapping(monkeypatch):
    from woof.core import nest as nest_mod
    from test_nest_coupler import _nodes
    parent, child = _nodes()
    parent.state = _host_facade(parent.state)
    child.clock.ticks = parent.clock.ticks
    coupler = nest_mod.NestCoupler(child, feedback=1, smooth_option=0)
    coupler._prepared_feedback = {'ticks': child.clock.ticks, 'kinds': ('mu', 'qi'), 'surface': ()}
    mapped = np.full(child.state.qi.shape, 23., np.float32)
    seen = []
    monkeypatch.setattr(coupler, '_bind_geometry', lambda: None)
    monkeypatch.setattr(coupler, '_reverse_diagnoses', lambda kind: kind == 'qi')
    monkeypatch.setattr(coupler, '_sync_feedback_source', lambda: None)
    monkeypatch.setattr(coupler, '_mapped_child_field', lambda kind: mapped)
    def restrict(kind, source, target, mapped=None):
        seen.append((kind, mapped))
        source.host_to_device_bytes += 5
        target.device_to_host_bytes += 7
    monkeypatch.setattr(coupler, '_restrict_windowed', restrict)
    coupler.feedback_commit(child)
    assert not coupler._feedback_child_is_bounded()
    assert seen[0] == ('mu', None) and seen[1][0] == 'qi' and seen[1][1] is mapped
    assert coupler.feedback_sync_bytes == 24 and coupler.feedback_count == 1


@pytest.mark.parametrize('parent_streams', [False, True])
def test_feedback_finalize_dispatches_on_parent_owner_independently(monkeypatch, parent_streams):
    from woof.core import nest as nest_mod, nest_operands, diagnostics
    from test_nest_coupler import _nodes
    parent, child = _nodes()
    if parent_streams:
        parent.state = _host_facade(parent.state)
    coupler = nest_mod.NestCoupler(child, feedback=1, smooth_option=0)
    coupler._prepared_feedback = {}
    seen = []
    monkeypatch.setattr(nest_operands, 'diagnose_canonical_parent',
                        lambda *args, **kwargs: seen.append(('window', kwargs['chunk_shape'])))
    monkeypatch.setattr(diagnostics, 'update_diagnostics',
                        lambda *args, **kwargs: seen.append(('resident', None)))
    # Nest-start diagnoses in both arithmetic modes; strict per-step feedback
    # intentionally leaves diagnosis to WRF's next RK preparation.
    coupler.feedback_finalize(child, start=True)
    assert seen == ([('window', (7, 8))] if parent_streams else [('resident', None)])
    assert coupler._prepared_feedback is None


def _marked_parent_moved_on(state):
    """A marked parent whose store the sweep has moved on from its state."""
    facade = _host_facade(state)
    for key, value in facade._streamed_domain.store.items():
        setattr(facade, key.split('/', 1)[1], value.copy())
        value += np.float32(7.0)
    return facade


def test_the_gate_stale_parent_controls_reach_the_bounded_parent_operands(monkeypatch):
    """N, W and the write-back control must disarm the door a marked parent takes.

    ``tilestream/test_nest.py`` and ``tilestream/test_nest_executor.py``
    stream the parent and keep the child resident.  Once a parent that
    ``StreamedDomain`` marked was coupled through ``NestWindowSource``
    straight out of its store, unpublishing ``_STORE_ATTR``, zeroing
    ``NEST_FORCE_HALO_PARENT_CELLS`` and stubbing ``commit_to_store`` no
    longer reached it, and on a card the gates reported those controls dead
    beside passing identity rows.  ``stale_parent_reads`` and
    ``dropped_parent_writes`` are what the gates install instead: the
    frozen parent everywhere (N), the frozen parent outside the halo-free
    child footprint, which the FORCE donor rectangles' SINT halo reaches
    (W), and no write into the parent's store (write-back).
    """
    from woof.core import nest as nest_mod
    from woof.core.streaming import window_slices
    from test_nest_coupler import _nodes
    from tilestream.test_nest import dropped_parent_writes, stale_parent_reads

    _numpy_device(monkeypatch)
    parent, child = _nodes()
    parent.state = _marked_parent_moved_on(parent.state)
    state, store = parent.state, parent.state._streamed_domain.store
    coupler = nest_mod.NestCoupler(child)
    source, child_source = NestWindowSource(state), NestWindowSource(child.state)

    # The live door reads the store, which has moved on from the state.
    assert source.array('thp') is store['state/thp']
    assert not np.array_equal(store['state/thp'], state.thp)

    # N: every parent carrier is the frozen one; geography, flags and every
    # other source pass through.
    frozen = stale_parent_reads(state)
    for name in ('thp', 'mup', 'u', 'qv'):
        assert np.array_equal(frozen(source, name), getattr(state, name))
    assert frozen(source, 'msft') is source.array('msft')
    assert frozen(source, 'thb') is source.array('thb')
    assert frozen(child_source, 'thp') is child.state.thp

    # W: fresh only inside the halo-free footprint.  Every FORCE donor
    # rectangle reaches past it (the SINT halo), and stays inside the
    # footprint at the default halo, so the starved leg reads stale cells
    # exactly where the shipped window would have read live ones.
    padded = nest_mod.parent_footprint_window(child.cfg)
    monkeypatch.setattr(nest_mod, 'NEST_FORCE_HALO_PARENT_CELLS', 0)
    bare = nest_mod.parent_footprint_window(child.cfg)
    starved = stale_parent_reads(state, bare)(source, 'thp')
    _, y, x = window_slices(starved.shape, bare)
    assert np.array_equal(starved[:, y, x], store['state/thp'][:, y, x])
    _, py, px = window_slices(starved.shape, padded)
    reg = coupler.registrations['m']
    for _, window, _ in boundary_windows(reg, 5, (7, 8)):
        _, (dy, dx) = window_registration(reg, window)
        assert py.start <= dy.start and dy.stop <= py.stop
        assert px.start <= dx.start and dx.stop <= px.stop
        donor = starved[:, dy, dx]
        assert np.any(donor == state.thp[:, dy, dx]), (
            f'donor {dy, dx} reads nothing stale outside the bare footprint')

    # Write-back: the parent's store keeps what the sweep left; every other
    # source still writes, and the live door would have written the parent.
    drop = dropped_parent_writes(state)
    window = (slice(4, 6), slice(4, 6))
    value = np.zeros((state.thp.shape[0], 2, 2), np.float32)
    before = store['state/thp'].copy()
    drop(source, 'thp', window, value)
    np.testing.assert_array_equal(store['state/thp'], before)
    drop(child_source, 'thp', window, value)
    assert np.all(child.state.thp[(...,) + window] == 0.)
    NestWindowSource.write(source, 'thp', window, value)
    assert np.all(store['state/thp'][(...,) + window] == 0.)

def test_canonical_digest_keeps_joined_carriers_and_adds_live_coupler_slots(monkeypatch):
    from woof.core.streaming import StreamedDomain
    from woof import state_digest

    joined = {"state/qv": np.full((2, 3, 4), 0.001, np.float32),
              "scratch/lbc_weights_0": np.array([0.25], np.float32)}
    live_table = np.full((2, 3, 4), 7., np.float32)
    facade = SimpleNamespace(qv=np.full((2, 3, 4), -999., np.float32),
                             _scratch={"lbc_weights_0": np.array([-999.], np.float32),
                                       "nest_qv_btxs": live_table})
    owner = SimpleNamespace(ranked=True, _state=facade, scalars={},
                            _run=SimpleNamespace(canonical_store=lambda: joined))
    # canonical_digest reads its inputs through canonical_digest_inputs
    # (725e222f3c, cards released before the end-of-run hashing).
    owner.canonical_digest_inputs = lambda: StreamedDomain.canonical_digest_inputs(owner)
    monkeypatch.setattr(state_digest, "canonical_store_digest",
                        lambda arrays, *a, **kw: arrays)
    actual = StreamedDomain.canonical_digest(owner, None)
    assert actual["state/qv"] is joined["state/qv"]
    assert actual["scratch/lbc_weights_0"] is joined["scratch/lbc_weights_0"]
    assert actual["scratch/nest_qv_btxs"] is live_table
    assert "scratch/nest_qv_btxs" not in joined


@pytest.mark.parametrize("bad_last_shape", [False, True])
def test_cross_card_reload_keeps_every_staging_copy_until_completion(
        monkeypatch, bad_last_shape):
    import sys
    import weakref
    from woof.core import nest_stream

    pending = []
    synchronized = []
    class Allocation:
        pass
    def queue_copy(dst, src):
        allocation = Allocation()
        pending.append(weakref.ref(allocation))
        return allocation
    def synchronize():
        assert len(pending) == 80
        assert all(reference() is not None for reference in pending)
        synchronized.append(True)
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(
        cuda=SimpleNamespace(get_current_stream=lambda: SimpleNamespace(
            synchronize=synchronize))))
    monkeypatch.setattr(nest_stream, "_on_another_card", lambda *a: True)
    monkeypatch.setattr(nest_stream, "_copy_across_cards", queue_copy)
    side = SimpleNamespace(value=np.zeros((1, 1, 1)),
                           tendency=np.zeros((1, 1, 1)))
    fields = {"qv": SimpleNamespace(west=side)}
    source = SimpleNamespace(intervals=[SimpleNamespace(fields=fields)])
    specs = [("qv", "west", np.s_[:, :, :], np.zeros((1, 1, 1)),
              np.zeros((1, 1, 1))) for _ in range(40)]
    if bad_last_shape:
        specs.append(("qv", "west", np.s_[:, :, :],
                      np.zeros((1, 2, 1)), np.zeros((1, 2, 1))))
        with pytest.raises(RuntimeError, match="packed slot holds"):
            nest_stream._copy_owned_sides(specs, source)
    else:
        assert nest_stream._copy_owned_sides(specs, source) == 40
    assert synchronized == [True]
    assert all(reference() is None for reference in pending)


@pytest.mark.parametrize("ratio", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("stagger", ["", "x", "y"])
def test_boundary_chunks_cover_each_global_table_once(ratio, stagger):
    reg = register_nest(nri=ratio, nrj=ratio, i_parent_start=7,
                        j_parent_start=8, child_nx=30, child_ny=33,
                        parent_nx=64, parent_ny=64, stagger=stagger, wrapper="bdy")
    seen = {side: np.zeros((reg.nyc, 5) if side in ("west", "east")
                           else (5, reg.nxc), np.int8)
            for side in ("west", "east", "south", "north")}
    for side, win, dest in boundary_windows(reg, 5, (7, 8)):
        seen[side][dest[1:]] += 1
        cropped, donor = window_registration(reg, win)
        assert (cropped.nyc, cropped.nxc) == (win[0].stop-win[0].start,
                                              win[1].stop-win[1].start)
    for count in seen.values():
        assert np.all(count == 1)


@pytest.mark.parametrize("ratio", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("stagger", ["", "x", "y"])
def test_feedback_donors_fit_one_ratio_footprint_per_parent_cell(ratio, stagger):
    reg = register_nest(nri=ratio, nrj=ratio, i_parent_start=7,
                        j_parent_start=8, child_nx=60, child_ny=60,
                        parent_nx=80, parent_ny=80, stagger=stagger)
    ilo, ihi, jlo, jhi = feedback_parent_bounds(reg)
    for j, i in ((jlo, ilo), (jhi, ihi)):
        donor = feedback_child_window(reg, (slice(j, j+1), slice(i, i+1)))
        assert donor[0].stop-donor[0].start == (1 if stagger == "y" else ratio)
        assert donor[1].stop-donor[1].start == (1 if stagger == "x" else ratio)


@pytest.mark.gpu
def test_gpu_canonical_coupling_windows_match_full_fields_bitwise():
    import cupy as cp
    from test_nest_coupler import _State, _DeviceState, _run
    from woof.ingest.lateral_bc import couple_nest_field

    host = _State(_run(31, 29, nested=True, grid_id=2))
    rng = np.random.default_rng(212)
    for name, value in vars(host).items():
        if isinstance(value, np.ndarray):
            value += rng.uniform(-.05, .05, value.shape).astype(np.float32)
    host.thb = np.broadcast_to(host.thb[:, None, None], (2, 29, 31)).copy()
    host.thb += rng.normal(size=host.thb.shape).astype(np.float32)
    resident = _DeviceState(host, cp)
    source = NestWindowSource(_host_facade(host))
    for kind in ("mu", "u", "v", "w", "t", "ph", "qv", "qc", "qr", "nr"):
        attr = {"mu": "mup", "t": "thp", "ph": "php"}.get(kind, kind)
        field = resident.mup[None] if kind == "mu" else getattr(resident, attr)
        expected = cp.empty(field.shape, dtype=cp.float32)
        couple_nest_field(resident, kind, out=expected)
        ny, nx = field.shape[-2:]
        for win in ((slice(0, 5), slice(0, 7)),
                    (slice(4, 11), slice(8, 13)),
                    (slice(ny-5, ny), slice(nx-7, nx))):
            got = source.coupled(kind, win)
            np.testing.assert_array_equal(cp.asnumpy(got).view(np.uint32),
                cp.asnumpy(expected[(...,)+win]).view(np.uint32))
    assert source.host_to_device_bytes > 0
    assert source.max_operand_bytes < resident.w.nbytes


@pytest.mark.gpu
@pytest.mark.parametrize("ratio", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("stagger", ["", "x", "y"])
def test_gpu_window_forcing_and_restriction_match_full_bitwise(ratio, stagger):
    import cupy as cp
    reg = register_nest(nri=ratio, nrj=ratio, i_parent_start=7,
                        j_parent_start=8, child_nx=30, child_ny=33,
                        parent_nx=64, parent_ny=64, stagger=stagger, wrapper="bdy")
    rng = np.random.default_rng(901)
    parent = cp.asarray(rng.normal(size=(3, reg.nyp, reg.nxp)).astype(np.float32))
    child = cp.asarray(rng.normal(size=(3, reg.nyc, reg.nxc)).astype(np.float32))
    expected = bdy_interp1(parent, child, reg, parent_dt_fp32=np.float32(17.25))
    actual = {s: tuple(cp.empty_like(v) for v in pair) for s, pair in expected.items()}
    for side, win, dest in boundary_windows(reg, 5, (7, 8)):
        cropped, donor = window_registration(reg, win)
        result = bdy_interp1(cp.ascontiguousarray(parent[(...,)+donor]),
            cp.ascontiguousarray(child[(...,)+win]), cropped,
            parent_dt_fp32=np.float32(17.25), sides=(side,))
        for got, value in zip(actual[side], result[side]):
            got[dest] = value
    for side in actual:
        for got, want in zip(actual[side], expected[side]):
            np.testing.assert_array_equal(cp.asnumpy(got).view(np.uint32),
                                          cp.asnumpy(want).view(np.uint32))
    want = parent.copy()
    copy_fcn(want, child, reg)
    got = parent.copy()
    ilo, ihi, jlo, jhi = feedback_parent_bounds(reg)
    for j in range(jlo, jhi+1, 3):
        for i in range(ilo, ihi+1, 2):
            win = (slice(j, min(j+3, jhi+1)), slice(i, min(i+2, ihi+1)))
            donor = feedback_child_window(reg, win)
            target = cp.empty((3, win[0].stop-j, win[1].stop-i), dtype=cp.float32)
            copy_fcn(target, cp.ascontiguousarray(child[(...,)+donor]), reg,
                     parent_window=win, child_window=donor)
            got[(...,)+win] = target
    np.testing.assert_array_equal(cp.asnumpy(got).view(np.uint32),
                                  cp.asnumpy(want).view(np.uint32))


@pytest.mark.gpu
@pytest.mark.parametrize("host_parent", [False, True])
def test_gpu_force_entrypoint_uses_host_child_without_full_field_scratch(host_parent):
    import cupy as cp
    from test_nest_coupler import _DeviceState, _nodes
    from woof.core.nest import NestCoupler

    parent, child = _nodes()
    parent.state = _DeviceState(parent.state, cp)
    child.state = _DeviceState(child.state, cp)
    control = NestCoupler(child)
    control.force(child)
    expected = {kind: {side: tuple(cp.asnumpy(v) for v in pair)
                       for side, pair in sides.items()}
                for kind, sides in control._last_tables.items()}

    parent, child = _nodes()
    parent.state = _host_facade(parent.state) if host_parent else _DeviceState(parent.state, cp)
    child.state = _host_facade(child.state)
    slots = {}
    def scratch(shape, slot, dtype=None):
        assert slot not in ("nest_parent_field", "nest_child_field")
        if slot not in slots:
            slots[slot] = cp.empty(shape, dtype=dtype or np.float32)
        return slots[slot]
    child.state.scratch = scratch
    bounded = NestCoupler(child)
    bounded.force(child)
    for kind, sides in bounded._last_tables.items():
        for side, pair in sides.items():
            for got, want in zip(pair, expected[kind][side]):
                np.testing.assert_array_equal(cp.asnumpy(got).view(np.uint32),
                                              want.view(np.uint32))
    assert bounded.force_sync_bytes > 0
    assert bounded.force_count == 1 and bounded.valid
    assert child.state._lateral_boundary_device.rolling_generation == 1


@pytest.mark.gpu
@pytest.mark.parametrize("smooth_option", [0, 1, 2])
@pytest.mark.parametrize("host_parent", [False, True])
def test_gpu_feedback_entrypoint_reads_host_child_and_matches_resident(smooth_option, host_parent):
    import cupy as cp
    from test_nest_coupler import _DeviceState, _nodes
    from woof.core.model import FeedbackScratch
    from woof.core.nest import NestCoupler

    results = []
    for bounded in (False, True):
        parent, child = _nodes()
        rng = np.random.default_rng(703)
        for state in (parent.state, child.state):
            for name, value in vars(state).items():
                if isinstance(value, np.ndarray):
                    value += rng.normal(size=value.shape).astype(np.float32)
            ny, nx = state.mup.shape
            state.thb = np.broadcast_to(state.thb[:, None, None], (2, ny, nx)).copy()
            state.thb += rng.normal(size=state.thb.shape).astype(np.float32)
            from woof.wrf_exact import ENABLED as wrf_exact
            if wrf_exact:
                state.thb.fill(300.)
        parent.state = (_host_facade(parent.state) if bounded and host_parent
                        else _DeviceState(parent.state, cp))
        if bounded:
            child.state = _host_facade(child.state)
            slots = {}
            def scratch(shape, slot, dtype=None):
                # Masked geometry feedback reuses the preflight-priced F16
                # transaction arenas, rather than allocating carrier shadows.
                if slot not in slots:
                    slots[slot] = cp.empty(shape, dtype=dtype or np.float32)
                return slots[slot]
            child.state.scratch = scratch
        else:
            child.state = _DeviceState(child.state, cp)
        child.clock.ticks = parent.clock.ticks
        coupler = NestCoupler(child, feedback=1, smooth_option=smooth_option)
        coupler.feedback_prepare(child, FeedbackScratch())
        coupler.feedback_commit(child)
        if bounded and host_parent and smooth_option:
            assert coupler.feedback_host_scratch_bytes > 0
        results.append({name: cp.asnumpy(getattr(parent.state, name))
                        for name in ("mup", "u", "v", "w", "thp", "php")})
    for name in results[0]:
        np.testing.assert_array_equal(results[0][name].view(np.uint32),
                                      results[1][name].view(np.uint32), err_msg=name)


@pytest.mark.gpu
@pytest.mark.parametrize("host_parent", [False, True])
def test_gpu_cam_ozone_transfer_writes_only_canonical_child_chunks(host_parent):
    import cupy as cp
    from woof.core.cam_ozone import CARRIER_KEY, transfer_parent_ozone
    from woof.core.nest_interp import sint
    from test_nest_coupler import _State, _run

    reg = register_nest(nri=3, nrj=3, i_parent_start=7, j_parent_start=8,
                        child_nx=30, child_ny=33, parent_nx=64, parent_ny=64)
    parent = _State(_run(64, 64, nested=False, grid_id=1))
    child = _host_facade(_State(_run(30, 33, nested=True, grid_id=2)))
    ozone = np.random.default_rng(317).random((2, 64, 64), dtype=np.float32)
    expected = cp.asnumpy(sint(cp.asarray(ozone), reg))
    if host_parent:
        parent = _host_facade(parent)
        parent._streamed_domain.store[CARRIER_KEY] = ozone
    parent.physics = SimpleNamespace(o3rad=cp.full((2, 2, 2), -99., np.float32)
                                    if host_parent else cp.asarray(ozone),
                                    call_counts={"cam_ozone": 4})
    canonical = np.full((2, 33, 30), -77., np.float32)
    child._streamed_domain.store[CARRIER_KEY] = canonical
    child._streamed_domain.scalars = {"call_counts": {"cam_ozone": 0}}
    child.physics = SimpleNamespace(
        cam_ozone=SimpleNamespace(mode="parent-interpolated"),
        o3rad=cp.full((2, 2, 2), -88., np.float32), call_counts={"cam_ozone": 0})
    node = SimpleNamespace(state=child, parent=SimpleNamespace(state=parent))
    moved = transfer_parent_ozone(node, reg)
    np.testing.assert_array_equal(canonical.view(np.uint32), expected.view(np.uint32))
    assert child._streamed_domain.scalars["call_counts"]["cam_ozone"] == 1
    assert moved >= canonical.nbytes
    assert bool(cp.all(child.physics.o3rad == -88.))


@pytest.mark.gpu
@pytest.mark.parametrize("base3d", [False, True])
@pytest.mark.parametrize("hypsometric_opt", [1, 2])
@pytest.mark.parametrize("smooth_option", [0, 2])
def test_gpu_feedback_finalize_uses_canonical_host_columns_exactly(
        base3d, hypsometric_opt, smooth_option):
    import cupy as cp
    from dataclasses import replace
    from woof.core.nest import NestCoupler
    from test_nest_coupler import _DeviceState, _nodes

    results = []
    for bounded in (False, True):
        parent, child = _nodes()
        parent.cfg.run = replace(parent.cfg.run, hypsometric_opt=hypsometric_opt)
        state = parent.state
        state.mub2d.fill(85000.)
        state.phb = np.array([0., 5000., 15000.], np.float32)
        state.dphb_resid = np.zeros(2, np.float32)
        state.alb = np.ones(2, np.float32)
        state.rdnw = np.full(2, -2., np.float32)
        state.c3h = np.array([.75, .25], np.float32)
        state.c3f = np.array([1., .5, 0.], np.float32)
        state.c4h = state.dc4f = np.zeros(2, np.float32)
        state.c4f = np.zeros(3, np.float32)
        state.dc3f = np.full(2, .5, np.float32)
        state.p_top = 10000.
        state.pb = np.full(2, 50000., np.float32)
        state.p_perturbation = np.full(state.thp.shape, -77., np.float32)
        for name in ("p", "al", "alt"):
            setattr(state, name, np.full(state.thp.shape, -99., np.float32))
        rng = np.random.default_rng(413)
        if base3d:
            for name in ("thb", "phb", "dphb_resid", "alb", "pb"):
                profile = getattr(state, name)
                field = np.broadcast_to(profile[:, None, None],
                                        (len(profile), *state.mup.shape)).copy()
                field += rng.uniform(-.01, .01, field.shape).astype(np.float32)
                setattr(state, name, field)
        if bounded:
            parent.state = _host_facade(state)
            child.state = _host_facade(child.state)
        else:
            parent.state = _DeviceState(state, cp)
            child.state = _DeviceState(child.state, cp)
        coupler = NestCoupler(child, feedback=1, smooth_option=smooth_option)
        coupler._prepared_feedback = {}
        coupler.feedback_finalize(child, start=True)
        results.append({name: cp.asnumpy(getattr(parent.state, name))
                        for name in ("p", "al", "alt", "p_perturbation")})
        assert coupler._prepared_feedback is None
    for name in results[0]:
        assert np.isfinite(results[0][name]).all()
        np.testing.assert_array_equal(results[0][name].view(np.uint32),
                                      results[1][name].view(np.uint32), err_msg=name)


def test_force_window_cache_follows_registration_replacement():
    from dataclasses import replace
    from woof.core.nest import NestCoupler
    from test_nest_coupler import _nodes
    parent, child = _nodes()
    parent.state = _host_facade(parent.state)
    coupler = NestCoupler(child)
    first = coupler._force_windows('t')
    assert coupler._force_windows('qv') is first
    reg = coupler.registrations['m']
    coupler.registrations = dict(coupler.registrations)
    coupler.registrations['m'] = replace(reg, ci=reg.ci+1,
                                        i_parent_start=reg.i_parent_start+1)
    moved = coupler._force_windows('t')
    assert moved is not first
    for old, new in zip(first, moved):
        assert new[4][1].start == old[4][1].start+1
        assert new[4][1].stop == old[4][1].stop+1
        assert new[4][0] == old[4][0]
