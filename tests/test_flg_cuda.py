from pathlib import Path
import hashlib
import json
import os
import numpy as np
import pytest
from woof.core.flg import column_primitives
FIXTURE=Path(__file__).parent/'fixtures'/'flg'/'primitives.npz'

def test_primitives_match_unmodified_wrf_words():
    cp=pytest.importorskip('cupy')
    if cp.cuda.runtime.getDeviceCount()<1: pytest.skip('CUDA device required')
    with np.load(FIXTURE) as fixture:
        outputs=column_primitives(fixture['pressure'],fixture['temperature'],fixture['coszen'],fixture['surface_temperature'])
        for key,value in outputs.items():
            got=cp.asnumpy(value).view(np.uint32); ref=fixture[key].view(np.uint32)
            misses=np.argwhere(got!=ref)
            record_gpu_field('primitives',key,got,ref)
            assert not len(misses), f'{key}: {len(misses)} bit misses; first {misses[:1].tolist()}'


def test_cloud_optics_match_all_unmodified_wrf_options():
    cp=pytest.importorskip('cupy')
    if cp.cuda.runtime.getDeviceCount()<1: pytest.skip('CUDA device required')
    from woof.core.flg import cloud_optics
    with np.load(FIXTURE.with_name('cloud_optics.npz')) as fixture:
        outputs=cloud_optics(*[fixture[name] for name in ('dz','pre','pde','plwc','piwc','prwc','pgwc')])
        for key,value in outputs.items():
            got=cp.asnumpy(value).view(np.uint32); ref=fixture[key].view(np.uint32)
            misses=np.argwhere(got!=ref)
            record_gpu_field('cloud_optics',key,got,ref)
            assert not len(misses), f'{key}: {len(misses)} bit misses; first {misses[:1].tolist()}'


def record_gpu_field(deck,key,got,ref):
    destination=os.environ.get('WOOF_FLG_RECEIPTS')
    if not destination: return
    import cupy as cp
    from woof.core.kernels import module_source,module_options
    target=Path(destination)/(deck+'.json')
    metadata={
        'arithmetic':'strict' if os.environ.get('GPUWM_WRF_EXACT')=='1' else 'default',
        'cupy':cp.__version__,'cuda_runtime':cp.cuda.runtime.runtimeGetVersion(),
        'device':cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
        'kernel_source_sha256':hashlib.sha256(module_source('flg_streams' if deck=='streams' else 'flg_gases' if deck=='gas_driver' else 'flg').encode()).hexdigest(),
        'options':list(module_options('flg_streams' if deck=='streams' else 'flg_gases' if deck=='gas_driver' else 'flg')),'fields':{}}
    from woof.core.flg import TABLE_SHA256,MANIFEST_SHA256
    metadata['coefficient_sha256']=TABLE_SHA256
    metadata['manifest_sha256']=MANIFEST_SHA256
    previous=json.loads(target.read_text()) if target.exists() else {}
    result=previous if all(previous.get(k)==v for k,v in metadata.items() if k!='fields') else metadata
    def order(words):
        return np.where(words & np.uint32(0x80000000), ~words,words | np.uint32(0x80000000)).astype(np.int64)
    misses=np.argwhere(got!=ref)
    result['fields'][key]={'shape':list(got.shape),'words':int(got.size),'bit_misses':int(len(misses)),
        'max_ulp':int(np.max(np.abs(order(got)-order(ref)))),'reference_sha256':hashlib.sha256(ref.tobytes()).hexdigest(),
        'gpu_sha256':hashlib.sha256(got.tobytes()).hexdigest(),'first_miss':misses[:1].tolist()}
    target.write_text(json.dumps(result,indent=2)+'\n')


def test_shared_loader_preserves_subnormals():
    cp=pytest.importorskip('cupy')
    if cp.cuda.runtime.getDeviceCount()<1: pytest.skip('CUDA device required')
    from woof.core.kernels import load_module
    values=np.array([1,0x80000001],dtype=np.uint32).view(np.float32)
    x=cp.asarray(values); factor=cp.asarray([1.],dtype=cp.float32); output=cp.empty_like(x)
    load_module('flg').get_function('flg_noftz_probe')((1,),(2,),(x,factor,output))
    np.testing.assert_array_equal(cp.asnumpy(output).view(np.uint32),values.view(np.uint32))



@pytest.mark.parametrize('method',['shortwave','longwave','ozone'])
def test_absorption_interpolation_matches_wrf(method):
    cp=pytest.importorskip('cupy')
    if cp.cuda.runtime.getDeviceCount()<1: pytest.skip('CUDA device required')
    from woof.core.flg import absorption_interpolation
    with np.load(FIXTURE.with_name('gas_helpers.npz')) as fixture:
        outputs=absorption_interpolation(fixture[method+'_coefficients'],fixture['pressure'],fixture['temperature'],method)
        for key,value in outputs.items():
            got=cp.asnumpy(value).view(np.uint32); ref=fixture[method+'_'+key].view(np.uint32)
            record_gpu_field('gas_helpers',method+'_'+key,got,ref)
            assert np.array_equal(got,ref), f'{method} {key}: {np.count_nonzero(got!=ref)} bit misses'

def test_continuum_matches_unmodified_wrf():
    cp=pytest.importorskip('cupy')
    if cp.cuda.runtime.getDeviceCount()<1: pytest.skip('CUDA device required')
    from woof.core.flg import continuum_depth
    with np.load(FIXTURE.with_name('gas_helpers.npz')) as fixture:
        value=continuum_depth(fixture['pressure'],fixture['temperature'],fixture['humidity'])
        got=cp.asnumpy(value).view(np.uint32); ref=fixture['continuum'].view(np.uint32)
        record_gpu_field('gas_helpers','continuum',got,ref)
        assert np.array_equal(got,ref), f'continuum: {np.count_nonzero(got!=ref)} bit misses'


def test_complete_gas_drivers_match_wrf_all_points_and_controls():
    cp=pytest.importorskip('cupy')
    if cp.cuda.runtime.getDeviceCount()<1: pytest.skip('CUDA device required')
    from woof.core.flg import gas_optical_depth
    with np.load(FIXTURE.with_name('gas_driver.npz')) as fixture:
        outputs=gas_optical_depth(*[fixture[key] for key in ('pressure','temperature','humidity','ozone','gas_values','switches')])
        for key,value in outputs.items():
            got=cp.asnumpy(value).view(np.uint32); ref=fixture[key].view(np.uint32)
            record_gpu_field('gas_driver',key,got,ref)
            misses=np.argwhere(got!=ref)
            assert not len(misses), f'{key}: {len(misses)} bit misses; first {misses[:1].tolist()}'


def test_clear_sky_zero_cloud_sizes_are_accepted():
    cp=pytest.importorskip('cupy')
    if cp.cuda.runtime.getDeviceCount()<1: pytest.skip('CUDA device required')
    from woof.core.flg import cloud_optics
    zero=np.zeros((1,24),dtype=np.float32)
    outputs=cloud_optics(np.ones_like(zero),zero,zero,zero,zero,zero,zero)
    for value in outputs.values():
        np.testing.assert_array_equal(cp.asnumpy(value).view(np.uint32),np.zeros(value.shape,dtype=np.uint32))

def test_zero_pivot_refuses_undefined_flux_solution():
    cp=pytest.importorskip('cupy')
    if cp.cuda.runtime.getDeviceCount()<1: pytest.skip('CUDA device required')
    from woof.core.kernels import load_module
    from woof.core.flg import _refuse_stream_errors
    error=cp.empty(1,dtype=cp.int32)
    load_module('flg_streams').get_function('flg_zero_pivot_probe')((1,),(1,),(error,))
    with pytest.raises(RuntimeError,match='zero pivot'):
        _refuse_stream_errors(error)


def test_streams_match_unmodified_wrf_four_arms():
    cp=pytest.importorskip('cupy')
    if cp.cuda.runtime.getDeviceCount()<1: pytest.skip('CUDA device required')
    from woof.core.flg import stream_fluxes
    with np.load(FIXTURE.with_name('streams.npz')) as fixture:
        keys=('bands','albedo','coszen','f0','emissivity','bf','bs','moments','scattering','depth')
        outputs=stream_fluxes(*[fixture[key] for key in keys])
        for key,value in outputs.items():
            got=cp.asnumpy(value).view(np.uint32); ref=fixture[key].view(np.uint32)
            record_gpu_field('streams',key,got,ref)
            misses=np.argwhere(got!=ref)
            assert not len(misses), f'{key}: {len(misses)} bit misses; first {misses[:1].tolist()}'
