"""Resident synchronization ownership, with optional CUDA word comparison."""
from types import SimpleNamespace as NS
import os

import numpy as np
import pytest

from woof.hex import cuda_physics_prep_v841 as prep
from woof.hex.mpasphys import diagnostics_v842 as diagnostics
from woof.hex.mpasphys.runtime_v842 import ResidentNativeCoupling


def setup_coupling(monkeypatch, *, resident=True, les=True):
    shape = (2, 3)
    allocated, launches, writes, fetched, preparations = [], [], [], [], []
    def forbidden(*args, **kwargs):
        raise AssertionError('unexpected host/device conversion or aggregate preparation')
    def empty(shape, dtype):
        allocated.append((tuple(shape), np.dtype(dtype).name))
        return np.empty(shape, dtype=dtype)
    def zeros(size, dtype):
        allocated.append(((size,), np.dtype(dtype).name))
        return np.zeros(size, dtype=dtype)
    cp = NS(empty=empty, zeros=zeros, float32=np.float32, int32=np.int32,
            asarray=forbidden, asnumpy=forbidden)
    geom = NS(n_cells=3, n_edges=4, max_edges=2, regional_garbage_cell=None,
              validate=lambda: preparations.append('geometry'))
    for name in ('edges_on_cell', 'n_edges_on_cell', 'coeffs_reconstruct', 'lat_cell', 'lon_cell'):
        setattr(geom, '_'+name, object())
        setattr(geom, '_sealed_'+name, object())
    fields = {name: np.full(shape, np.nan, np.float32)
              for name in ('uReconstructZonal', 'uReconstructMeridional')}
    def field(pool, name):
        assert pool == 'diag'
        fetched.append(name)
        return fields[name]
    session = NS(device_pools=NS(resident=resident), field=field,
        put=lambda pool, name, value, **kw: writes.append((pool, name, value, kw)),
        layout=NS(fields=[dict(pool='state', name='scalars', components=('qv', 'qc'))]))
    coupling = ResidentNativeCoupling.__new__(ResidentNativeCoupling)
    coupling.session, coupling.cp, coupling.geometry = session, cp, geom
    coupling.names, coupling.physics_cache = ('qv', 'qc'), object()
    coupling.dycore = NS(les_dissipation=NS() if les else None)
    mass = np.ones(shape, np.float32)
    saved = NS(theta_m=mass, normal_velocity=np.ones((2, 4), np.float32),
        vertical_velocity=mass, density_perturbation=mass, rho_theta_perturbation=mass,
        pressure_perturbation=mass, exner=mass)
    atmosphere = NS(state=NS(rho=mass, scalars=np.zeros((2, *shape), np.float32)), saved=saved)
    words = np.array([0x80000000, 0x7fc12345, 0x3f800000, 1, 0xbf800000, 0], np.uint32).reshape(shape)
    def launch(cache, name, count, args):
        assert cache is coupling.physics_cache and name == 'reconstruct_winds_v841_f32' and count == 3
        assert tuple(args[:4]) == (2, 3, 4, 2)
        assert args[4] is saved.normal_velocity
        for i, key in enumerate(('edges_on_cell', 'n_edges_on_cell', 'coeffs_reconstruct', 'lat_cell', 'lon_cell')):
            assert args[5+i] is getattr(geom, '_'+key)
            assert args[10+i] is getattr(geom, '_sealed_'+key)
        assert len({id(value) for value in args[15:20]}) == 5
        if geom.regional_garbage_cell is None and fields["uReconstructZonal"].shape == shape:
            # a cull or a same-host rank reconstructs into local-extent columns, then copies
            assert args[18] is fields['uReconstructZonal'] and args[19] is fields['uReconstructMeridional']
        assert args[20].shape == (1,) and args[20].dtype == np.int32
        # Substitute checks word movement and storage ownership, not CUDA maths.
        args[18].view(np.uint32)[...] = words
        args[19].view(np.uint32)[...] = words ^ np.uint32(0x80000000)
        launches.append(args)
    monkeypatch.setattr(prep, '_launch', launch)
    monkeypatch.setattr(prep, 'prepare_mpas_to_phys_cuda_v841', forbidden)
    monkeypatch.setattr(diagnostics, 'output_diagnostics', lambda *args: (mass, mass, mass))
    return NS(**locals())


@pytest.mark.parametrize('les', [False, True])
def test_resident_synchronization_reuses_pools_and_scratch(monkeypatch, les):
    h = setup_coupling(monkeypatch, les=les)
    previous = None
    for step in range(30):
        for level in (1, 2):
            h.writes.clear()
            result = h.coupling.synchronize(h.atmosphere, level=level)
            if previous is not None:
                assert result is previous
            previous = result
            assert np.array_equal(result.u_p.view(np.uint32), h.words)
            assert np.array_equal(result.v_p.view(np.uint32), h.words ^ np.uint32(0x80000000))
            assert all(row[3]['level'] == level for row in h.writes[:5])
            assert len(h.writes) == 12
            assert not any(row[1].startswith('uReconstruct') for row in h.writes)
            if les:
                assert h.coupling.dycore.les_dissipation.winds is result
    assert h.allocated == [((2, 3), 'float32')] * 3 + [((1,), 'int32')]
    assert h.fetched == ['uReconstructZonal', 'uReconstructMeridional']
    assert h.preparations == ['geometry']
    assert len(h.launches) == 60
    assert all(all(a is b for a, b in zip(args[15:], h.launches[0][15:])) for args in h.launches)


def test_nonresident_synchronization_keeps_aggregate_validation(monkeypatch):
    h = setup_coupling(monkeypatch, resident=False)
    columns = NS(u_p=np.ones((2, 3), np.float32), v_p=np.zeros((2, 3), np.float32))
    calls = []
    def prepare(*args, **kwargs):
        calls.append((args, kwargs))
        return columns
    monkeypatch.setattr(prep, 'prepare_mpas_to_phys_cuda_v841', prepare)
    assert h.coupling.synchronize(h.atmosphere) is columns
    assert len(calls) == 1 and calls[0][1]['registry_scalar_names'] == ('qv', 'qc')
    assert not h.allocated and not h.launches and not h.fetched
    assert [row[1] for row in h.writes[9:11]] == ['uReconstructZonal', 'uReconstructMeridional']


def test_explicit_publication_columns_keep_surface_pressure(monkeypatch):
    h = setup_coupling(monkeypatch)
    columns = NS(u_p=np.ones((2, 3), np.float32), v_p=np.zeros((2, 3), np.float32),
                 psfc_p=np.full(3, 100000, np.float32))
    monkeypatch.setattr(prep, 'prepare_mpas_to_phys_cuda_v841', lambda *args, **kwargs: columns)
    result = h.coupling.synchronize(h.atmosphere, prepared_columns=True)
    assert result is columns
    assert np.all(result.psfc_p == 100000)
    assert not h.allocated and not h.launches
    assert [row[1] for row in h.writes[9:11]] == ['uReconstructZonal', 'uReconstructMeridional']


@pytest.mark.parametrize('donor', [None, 0])
def test_resident_winds_on_a_cull_fill_entity_pools_and_lend_the_garbage_column(monkeypatch, donor):
    # A limited-area cull (3 km region at 33.5N 101W, 2026-10-24) used to refuse here at the first output: the
    # dycore extent carries the garbage column (last), the pools hold the entity extent.
    h = setup_coupling(monkeypatch)
    h.geom.regional_garbage_cell, h.geom.regional_garbage_donor_cell = 2, donor
    for name in ('uReconstructZonal', 'uReconstructMeridional'):
        h.fields[name] = np.full((2, 2), np.nan, np.float32)
    for _ in range(3):
        result = h.coupling.synchronize(h.atmosphere)
    source = 1 if donor is None else donor
    expected_u = h.words.copy(); expected_u[:, 2] = expected_u[:, source]
    expected_v = h.words ^ np.uint32(0x80000000); expected_v[:, 2] = expected_v[:, source]
    assert np.array_equal(result.u_p.view(np.uint32), expected_u)
    assert np.array_equal(result.v_p.view(np.uint32), expected_v)
    assert np.array_equal(h.fields['uReconstructZonal'].view(np.uint32), expected_u[:, :2])
    assert np.array_equal(h.fields['uReconstructMeridional'].view(np.uint32), expected_v[:, :2])
    assert len(h.launches) == 3 and h.fetched == ['uReconstructZonal', 'uReconstructMeridional']
    assert h.allocated == [((2, 3), 'float32')] * 5 + [((1,), 'int32')]


def test_resident_winds_refuse_a_garbage_column_that_is_not_last(monkeypatch):
    h = setup_coupling(monkeypatch)
    h.geom.regional_garbage_cell, h.geom.regional_garbage_donor_cell = 1, None
    with pytest.raises(ValueError, match='last element'):
        h.coupling._resident_winds(h.atmosphere)
    assert not h.launches


@pytest.mark.parametrize('fault', ['geometry', 'output', 'changed'])
def test_resident_wind_extent_errors_precede_launch(monkeypatch, fault):
    h = setup_coupling(monkeypatch)
    if fault == 'geometry':
        h.geom.n_edges = 5
    elif fault == 'output':
        h.fields['uReconstructZonal'] = np.ones((2, 4), np.float32)
    else:
        h.coupling._resident_winds(h.atmosphere)
        h.atmosphere.state.rho = np.ones((3, 3), np.float32)
        h.saved.normal_velocity = np.ones((3, 4), np.float32)
        h.launches.clear()
    if fault == 'output':
        # A pool that is not the state's extent (a same-host rank's rows) takes the aggregate
        # preparation instead (fix/hex-both-routes-gate F-BR2): no launch, no resident winds.
        assert h.coupling._resident_winds(h.atmosphere) is None
        assert h.coupling._wind_fallback
        assert not h.launches
        return
    with pytest.raises(ValueError, match='extent|pool shape'):
        h.coupling._resident_winds(h.atmosphere)
    assert not h.launches


@pytest.mark.skipif(os.environ.get('MPAS_PHYSICS_GPU_TESTS') != '1',
                    reason='explicit GPU slot allocation required')
@pytest.mark.parametrize('numerics', ['fast', 'strict'])
def test_resident_winds_match_aggregate_kernel_words(monkeypatch, tmp_path, numerics):
    import cupy as cp
    from woof.hex.mpasphys.compile_v842 import MpasPhysicsKernelCache
    monkeypatch.setenv('WOOF_HEX_NUMERICS', numerics)
    rng = np.random.default_rng(7731)
    nl, nc, ne, me = 5, 9, 16, 3
    live = dict(edges_on_cell=cp.asarray(rng.integers(0, ne, (nc, me), dtype=np.int32)),
        n_edges_on_cell=cp.full(nc, me, dtype=cp.int32),
        coeffs_reconstruct=cp.asarray(rng.uniform(-1, 1, (nc, me, 3)).astype(np.float32)),
        lat_cell=cp.asarray(rng.uniform(-1.5, 1.5, nc).astype(np.float32)),
        lon_cell=cp.asarray(rng.uniform(-3.1, 3.1, nc).astype(np.float32)))
    geom = NS(n_cells=nc, n_edges=ne, max_edges=me, regional_garbage_cell=None, validate=lambda: None)
    for name, value in live.items():
        setattr(geom, '_'+name, value)
        setattr(geom, '_sealed_'+name, value.copy())
    fields = {name: cp.empty((nl, nc), cp.float32) for name in ('uReconstructZonal', 'uReconstructMeridional')}
    coupling = ResidentNativeCoupling.__new__(ResidentNativeCoupling)
    coupling.cp, coupling.geometry = cp, geom
    coupling.physics_cache = MpasPhysicsKernelCache(cache_dir=tmp_path/'kernel-cache')
    coupling.session = NS(field=lambda pool, name: fields[name])
    atmosphere = NS(state=NS(rho=cp.ones((nl, nc), cp.float32)),
        saved=NS(normal_velocity=cp.empty((nl, ne), cp.float32)))
    reference = [cp.empty((nl, nc), cp.float32) for _ in range(5)]
    flag = cp.zeros(1, cp.int32)
    for step in range(3):
        atmosphere.saved.normal_velocity[...] = cp.asarray(rng.uniform(-50, 50, (nl, ne)).astype(np.float32))
        result = coupling._resident_winds(atmosphere)
        # Exact reference invocation taken from aggregate preparation, retaining
        # distinct temporary/output arrays and the same compiler/cache binding.
        prep._launch(coupling.physics_cache, 'reconstruct_winds_v841_f32', nc, (
            np.int32(nl), np.int32(nc), np.int32(ne), np.int32(me), atmosphere.saved.normal_velocity,
            *(live[name] for name in live), *(getattr(geom, '_sealed_'+name) for name in live),
            *reference, flag))
        for got, expected in zip((result.u_p, result.v_p), reference[3:]):
            np.testing.assert_array_equal(cp.asnumpy(got.view(cp.uint32)), cp.asnumpy(expected.view(cp.uint32)))
    assert int(cp.asnumpy(flag)[0]) == 0


@pytest.mark.parametrize('garbage', [None, 2])
def test_resident_winds_on_a_samehost_rank_fill_the_owned_prefix(monkeypatch, garbage):
    # NOAA conus3km on 8 cards (rrfs/c3-8card-speed 1960569, 2026-10-29): a same-host rank's pools hold its owned
    # columns, so every rank fell back to the aggregate preparation and read its validation word on the host twice a
    # step (outer physics 178 ms on rank 0). The owned prefix of the local extent goes into the pools on the device.
    h = setup_coupling(monkeypatch)
    h.session.layout.samehost_partition = True
    h.geom.regional_garbage_cell, h.geom.regional_garbage_donor_cell = garbage, None
    for name in ('uReconstructZonal', 'uReconstructMeridional'):
        h.fields[name] = np.full((2, 1), np.nan, np.float32)
    for _ in range(3):
        result = h.coupling.synchronize(h.atmosphere)
    expected_u = h.words.copy()
    expected_v = h.words ^ np.uint32(0x80000000)
    if garbage is not None:
        expected_u[:, 2] = expected_u[:, 1]
        expected_v[:, 2] = expected_v[:, 1]
    assert np.array_equal(result.u_p.view(np.uint32), expected_u)
    assert np.array_equal(result.v_p.view(np.uint32), expected_v)
    assert np.array_equal(h.fields['uReconstructZonal'].view(np.uint32), expected_u[:, :1])
    assert np.array_equal(h.fields['uReconstructMeridional'].view(np.uint32), expected_v[:, :1])
    assert len(h.launches) == 3 and not h.coupling.__dict__.get('_wind_fallback', False)


def test_resident_winds_without_samehost_layout_keep_the_aggregate_fallback(monkeypatch):
    h = setup_coupling(monkeypatch)
    for name in ('uReconstructZonal', 'uReconstructMeridional'):
        h.fields[name] = np.full((2, 1), np.nan, np.float32)
    assert h.coupling._resident_winds(h.atmosphere) is None
    assert h.coupling._wind_fallback and not h.launches
