"""GPU height integration preserves sparse, signed and sub-ground donor arithmetic."""
import os
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from woof.hex.moving_remap import ConservativeMap
from woof.hex.moving_vertical import HeightConservativeMap


@pytest.mark.skipif(os.environ.get('HEX_TEST_CUDA') != '1', reason='requires an owned CUDA card')
def test_device_ground_extension_keeps_host_words_and_reuses_one_carried_upload(monkeypatch):
    import cupy as cp
    from woof.hex.moving_vertical_cuda import _below_ground_values
    rng=np.random.default_rng(2391)
    n,levels=19,55
    overlap=np.eye(n)*.75+np.roll(np.eye(n),1,axis=1)*.25
    horizontal=ConservativeMap(csr_matrix(overlap),np.ones(n),np.ones(n),{})
    source=np.linspace(0.,20000.,levels+1)[:,None]*np.ones((1,n))
    source+=(1-source/20000.)*rng.uniform(100.,500.,n)
    target=np.linspace(0.,20000.,levels+1)[:,None]*np.ones((1,n))
    w=HeightConservativeMap(horizontal,source,target,np.diff(source[:,0]),np.diff(target[:,0]),
                           endpoint_alignment=False,backend='cpu')
    theta=rng.uniform(270.,330.,(1,n));exner=rng.uniform(.9,1.,(1,n));loading=rng.uniform(1.,1.05,(1,n))
    w.extend_below_ground(theta,exner,loading)
    expected=w.below_ground.copy()
    w.backend='cuda'
    w.extend_below_ground(theta,exner,loading)
    assert w.below_ground.tobytes()==expected.tobytes()
    cached=_below_ground_values(w)
    assert _below_ground_values(w) is cached
    assert cp.asnumpy(cached['values']).tobytes()==expected.tobytes()
    # The next carried thermodynamics replace the extension even on one map.
    theta=theta+1.
    w.backend='cpu';w.extend_below_ground(theta,exner,loading);expected=w.below_ground.copy()
    w.backend='cuda';w.extend_below_ground(theta,exner,loading)
    assert _below_ground_values(w) is not cached
    assert w.below_ground.tobytes()==expected.tobytes()


@pytest.mark.skipif(os.environ.get('HEX_TEST_CUDA') != '1', reason='requires an owned CUDA card')
@pytest.mark.parametrize('streams',['0','1'])
def test_concurrent_moving_products_share_only_finished_geometry(monkeypatch,streams):
    pytest.importorskip('cupy')
    from woof.hex.host_parallel import run_tasks
    monkeypatch.setenv('WOOF_HEX_MOVING_CUDA_STREAMS',streams)
    rng=np.random.default_rng(3031)
    n,levels=19,9
    matrix=np.eye(n)*.75+np.roll(np.eye(n),1,axis=1)*.25
    horizontal=ConservativeMap(csr_matrix(matrix),np.ones(n),np.ones(n),{})
    source=np.linspace(0.,10000.,levels+1)[:,None]*np.ones((1,n))
    target=source+(1-source/10000.)*rng.uniform(-100.,100.,n)
    w=HeightConservativeMap(horizontal,source,target,np.diff(source[:,0]),np.diff(source[:,0]),
                            endpoint_alignment=False,backend='cpu')
    work=[('mass_intensive',rng.uniform(.1,2.,(levels,n))) for _ in range(3)]
    work += [('layer_intensive',rng.normal(size=(levels,2,n))),
             ('layer_intensive',rng.normal(size=(levels,n))),
             ('interface_intensive',rng.normal(size=(levels+1,n))),
             ('intensive',rng.normal(size=(4,n)))]
    expected=[getattr(w,name)(value) for name,value in work]
    w.backend='cuda'
    actual=run_tasks([lambda name=name,value=value:getattr(w,name)(value) for name,value in work],workers=7)
    assert all(a.tobytes()==e.tobytes() for a,e in zip(actual,expected))


@pytest.mark.skipif(os.environ.get('HEX_TEST_CUDA') != '1', reason='requires an owned CUDA card')
@pytest.mark.parametrize('aligned', [False, True])
@pytest.mark.parametrize('kind', ['positive', 'signed', 'sparse'])
def test_ordered_height_integrals_match_host(monkeypatch, aligned, kind):
    pytest.importorskip('cupy')
    rng=np.random.default_rng(117)
    n=19; levels=9
    overlap=np.eye(n)*.75+np.roll(np.eye(n),1,axis=1)*.25
    horizontal=ConservativeMap(csr_matrix(overlap),np.ones(n),np.ones(n),{})
    z=np.linspace(0.,10000.,levels+1)[:,None]*np.ones((1,n))
    target=z+(1-z/10000.)*rng.uniform(-10.,10.,n)
    w=HeightConservativeMap(horizontal,z,target,np.diff(z[:,0]),np.diff(z[:,0]),endpoint_alignment=aligned)
    assert w.backend == 'cuda'
    density=rng.uniform(.1,1.1,(levels,n))
    value=rng.uniform(0.,2.,(levels,n))
    if kind=='signed': value-=1.
    if kind=='sparse':value[value<1.8]=0.
    if not aligned:
        w.extend_below_ground(np.full_like(density,300.),np.ones_like(density))
    w.backend='cpu'
    expected=w.mass_intensive(value,density=density)
    receipts=(w.last_domain_reconciliation,w.last_positivity_roundoff)
    w.backend='cuda'
    actual=w.mass_intensive(value,density=density)
    assert actual.tobytes()==expected.tobytes()
    assert (w.last_domain_reconciliation,w.last_positivity_roundoff)==receipts
    # Reusing geometry must not retain the previous product or extension.
    if not aligned: w.extend_below_ground(np.full_like(density,310.),np.full_like(density,.98))
    actual=w.mass_intensive(value*.75,density=density)
    w.backend='cpu'
    expected=w.mass_intensive(value*.75,density=density)
    assert actual.tobytes()==expected.tobytes()


@pytest.mark.skipif(os.environ.get('HEX_TEST_CUDA') != '1', reason='requires an owned CUDA card')
@pytest.mark.parametrize('source_levels,target_levels', [(1, 1), (9, 5), (55, 55)])
@pytest.mark.parametrize('middle', [(), (2, 3)])
def test_layer_rates_match_host_and_refresh_values(monkeypatch, source_levels, target_levels, middle):
    pytest.importorskip('cupy')
    rng = np.random.default_rng(461)
    n = 19
    overlap = np.eye(n)*.75 + np.roll(np.eye(n), 1, axis=1)*.25
    horizontal = ConservativeMap(csr_matrix(overlap), np.ones(n), np.ones(n), {})
    source = np.linspace(0., 10000., source_levels+1)[:, None]*np.ones((1, n))
    target = np.linspace(0., 10000., target_levels+1)[:, None]*np.ones((1, n))
    target += (1-target/10000.)*rng.uniform(-500., 500., n)
    w = HeightConservativeMap(horizontal, source, target, np.diff(source[:, 0]),
                             np.diff(np.linspace(0., 10000., target_levels+1)), endpoint_alignment=False)
    assert w.backend == 'cuda'
    values = rng.normal(size=(source_levels, *middle, n))
    for product in (values, values*.25):
        w.backend='cpu'
        expected = w.layer_intensive(product)
        w.backend='cuda'
        actual = w.layer_intensive(product)
        assert actual.shape == expected.shape
        assert actual.tobytes() == expected.tobytes()


@pytest.mark.skipif(os.environ.get('HEX_TEST_CUDA') != '1', reason='requires an owned CUDA card')
@pytest.mark.parametrize('source_levels,target_levels', [(1, 1), (9, 5), (55, 55)])
@pytest.mark.parametrize('fmad', ['false', 'true'])
def test_cached_rate_search_matches_original_kernel(monkeypatch, source_levels, target_levels, fmad):
    pytest.importorskip('cupy')
    from woof.hex import moving_vertical_cuda as cuda
    monkeypatch.setattr(cuda, 'RATE_OPTIONS', (f'--fmad={fmad}', '--prec-div=false'))
    cuda._layer_kernel.cache_clear()
    cuda._rate_kernels.cache_clear()
    rng = np.random.default_rng(1030)
    n = 37
    overlap = np.eye(n)*.6 + np.roll(np.eye(n), 1, axis=1)*.3 + np.roll(np.eye(n), 7, axis=1)*.1
    horizontal = ConservativeMap(csr_matrix(overlap), np.ones(n), np.ones(n), {})
    z = np.linspace(0., 10000., source_levels+1)[:, None]*np.ones((1, n))
    target = np.linspace(0., 10000., target_levels+1)[:, None]*np.ones((1, n))
    target += (1-target/10000.)*rng.uniform(-1000., 1000., n)
    w = HeightConservativeMap(horizontal, z, target, np.diff(z[:, 0]),
                             np.diff(np.linspace(0., 10000., target_levels+1)), endpoint_alignment=False)
    try:
        cache = None
        for scale in (1., .25):
            values = rng.normal(size=(source_levels, 2, 3, n))*scale
            values[:, :, :, 0] = -0.
            monkeypatch.setenv('WOOF_HEX_MOVING_RATE_GEOMETRY', '0')
            expected = cuda.layer_integral(w, values)
            monkeypatch.setenv('WOOF_HEX_MOVING_RATE_GEOMETRY', '1')
            actual = cuda.layer_integral(w, values)
            assert actual.tobytes() == expected.tobytes()
            current = w._cuda_rate_search
            if cache is not None:
                assert current is cache
            cache = current
        assert w._cuda_rate_geometry['arrays'][3] is w._cuda_cumulative_geometry['arrays'][4]
        monkeypatch.setenv('WOOF_HEX_MOVING_PREPARE_BUDGET_MIB', '1')
        assert not cuda.prepare_geometry(w)['prepared']
        monkeypatch.setenv('WOOF_HEX_MOVING_PREPARE_BUDGET_MIB', '20000')
        assert cuda.prepare_geometry(w)['prepared']
    finally:
        cuda._layer_kernel.cache_clear()
        cuda._rate_kernels.cache_clear()


@pytest.mark.skipif(os.environ.get('HEX_TEST_CUDA') != '1', reason='requires an owned CUDA card')
@pytest.mark.parametrize('fmad', ['false', 'true'])
def test_interface_remap_keeps_cpu_words_with_fast_compilation(monkeypatch, fmad):
    pytest.importorskip('cupy')
    from woof.hex import moving_vertical_cuda as cuda
    monkeypatch.setattr(cuda, 'INTERFACE_OPTIONS', (f'--fmad={fmad}',))
    cuda._interface_kernel.cache_clear()
    rng = np.random.default_rng(1029)
    n, levels = 19, 55
    area = rng.uniform(.5, 2., n)
    overlap = np.eye(n)*.75 + np.roll(np.eye(n), 1, axis=1)*.25
    horizontal = ConservativeMap(csr_matrix(overlap), area, np.ones(n), {})
    source = np.cumsum(np.r_[0., rng.uniform(30., 500., levels)])[:, None] * np.ones((1, n))
    target = source + (1-source/source[-1])*rng.uniform(-300., 300., n)
    weights = HeightConservativeMap(horizontal, source, target, np.diff(source[:, 0]),
                                   np.diff(source[:, 0]), endpoint_alignment=False)
    # Terrain endpoints, changing diagnostic values, cancellation and signed zeros.
    values = rng.normal(size=source.shape)
    values[:, 0] = -0.
    values[::2, 1] *= 1e-290
    try:
        for product in (values, values*.25):
            weights.backend = 'cpu'
            expected = weights.interface_intensive(product)
            weights.backend = 'cuda'
            assert weights.interface_intensive(product).tobytes() == expected.tobytes()
    finally:
        cuda._interface_kernel.cache_clear()


@pytest.mark.skipif(os.environ.get('HEX_TEST_CUDA') != '1', reason='requires an owned CUDA card')
@pytest.mark.parametrize('aligned', [False, True])
@pytest.mark.parametrize('kind', ['positive', 'signed', 'sparse', 'zeros'])
@pytest.mark.parametrize('with_density', [False, True])
@pytest.mark.parametrize('fmad', ['false', 'true'])
def test_device_profiles_match_host_profiles(monkeypatch, aligned, kind, with_density, fmad):
    """moving_vertical_cuda.device_layer_result writes the host profiles' words (WOOF_HEX_MOVING_DEVICE_PROFILES=0).

    THE BREAKAGE IT PREVENTS: a storm-following move whose remapped atmosphere differs from the process-per-segment
    route's by the profile or roundoff arithmetic (an FMA, another limiter or clip order, a signed zero).
    """
    pytest.importorskip('cupy')
    from woof.hex import moving_vertical_cuda
    # A fast-numerics run compiles every unit with --fmad=true (woof.hex.numerics); the words must not move.
    monkeypatch.setattr(moving_vertical_cuda, 'PROFILE_OPTIONS', (f'--fmad={fmad}',))
    moving_vertical_cuda._profile_kernels.cache_clear()
    rng = np.random.default_rng(29)
    n, levels = 700, 55
    overlap = np.eye(n)*.6 + np.roll(np.eye(n), 1, axis=1)*.3 + np.roll(np.eye(n), 7, axis=1)*.1
    horizontal = ConservativeMap(csr_matrix(overlap), np.ones(n), np.ones(n), {})
    z = np.cumsum(np.r_[0., rng.uniform(80., 400., levels)])[:, None]*np.ones((1, n))
    target = z + (1 - z/z[-1])*rng.uniform(-60., 60., n)
    w = HeightConservativeMap(horizontal, z, target, np.diff(z[:, 0]), np.diff(z[:, 0]), endpoint_alignment=aligned)
    assert w.backend == 'cuda'
    density = rng.uniform(.1, 1.1, (levels, n)) if with_density else None
    value = rng.uniform(0., 2., (levels, n))
    if kind == 'signed':
        value -= 1.
    if kind == 'sparse':
        value[value < 1.8] = 0.
    if kind == 'zeros':
        value[:, ::3] = 0.
        value[5:9, 1::3] = -0.
        value[:, 2::3] *= 1e-300
    if with_density and not aligned:
        w.extend_below_ground(np.full((levels, n), 300.), np.ones((levels, n)))
    monkeypatch.setenv('WOOF_HEX_MOVING_DEVICE_PROFILES', '0')
    expected = w.mass_intensive_receipted(value, density=density)
    monkeypatch.setenv('WOOF_HEX_MOVING_DEVICE_PROFILES', '1')
    actual = w.mass_intensive_receipted(value, density=density)
    assert actual[0].tobytes() == expected[0].tobytes()
    assert actual[1:] == expected[1:]
    moving_vertical_cuda._profile_kernels.cache_clear()
