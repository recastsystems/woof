"""Bounded batching and zero reuse: memory, scheduling and bitwise launch checks."""
from types import SimpleNamespace
import os

import numpy as np
import pytest

from woof.hex import cuda_transport_v841 as t


def test_batch_budget_keeps_large_mesh_census_limit(monkeypatch):
    monkeypatch.delenv('WOOF_HEX_LIMITER_SPECIES_PER_PASS', raising=False)
    monkeypatch.delenv('WOOF_HEX_TRANSPORT_WORKSPACE_BYTES', raising=False)
    assert t.limiter_species_per_pass(11, nlev=55, ncells=167254, nedges=501756) == 3
    assert t.limiter_species_per_pass(11, nlev=55, ncells=655362, nedges=1966080) == 1
    assert t.limiter_species_per_pass(11, nlev=7, ncells=17, nedges=17) == 11
    assert t.limiter_species_per_pass(11, nlev=55, ncells=167254, nedges=501756, available_bytes=500_000_000) == 1
    assert t.limiter_species_per_pass(11) == 1


@pytest.mark.parametrize('parts,expected', [(1,11),(2,6),(3,4),(11,1)])
def test_halo_round_schedule_cannot_be_rebatched(monkeypatch, parts, expected):
    monkeypatch.setenv('WOOF_HEX_LIMITER_SPECIES_PER_PASS','7')
    halo = SimpleNamespace(fct_scale_parts=parts)
    assert t.limiter_species_per_pass(11, halo, nlev=55, ncells=167254, nedges=501756) == expected


def test_explicit_width_and_budget_validation(monkeypatch):
    for value, expected in [('1',1), ('4',4), ('0',11)]:
        monkeypatch.setenv('WOOF_HEX_LIMITER_SPECIES_PER_PASS',value)
        assert t.limiter_species_per_pass(11,nlev=55,ncells=167254,nedges=501756) == expected
    monkeypatch.delenv('WOOF_HEX_LIMITER_SPECIES_PER_PASS')
    monkeypatch.setenv('WOOF_HEX_TRANSPORT_WORKSPACE_BYTES','0')
    with pytest.raises(ValueError,match='positive'):
        t.limiter_species_per_pass(11,nlev=55,ncells=167254,nedges=501756)
    with pytest.raises(ValueError,match='cannot be split'):
        t.limiter_species_per_pass(11,SimpleNamespace(fct_scale_parts=7))


def test_repeated_zero_plan_preserves_signed_zero_nan_and_subnormal():
    words = np.array([1,0,0,0x80000000,0x7fc12345,0,0],np.uint32)
    passes,copies = t.species_work_plan(words,3)
    assert passes == [(0,2),(3,5)]
    assert copies == [(1,2,3),(1,5,7)]
    output = np.arange(28,dtype=np.uint32).reshape(7,2,2).view(np.float32)
    output[1].view(np.uint32)[:] = [[0x80000000,0x7fc12345],[0,1]]
    t._copy_zero_species(output,copies)
    assert output[2].tobytes() == output[1].tobytes() == output[6].tobytes()


def test_device_classifier_rechecks_old_stage_and_source(monkeypatch):
    from transport_host_launch import HostArrays
    cp = HostArrays()
    monkeypatch.setattr(t,'_ZERO_CLASSIFIER',None)
    monkeypatch.setattr(t,'_ZERO_LEVEL_REDUCER',None)
    monkeypatch.setenv('WOOF_HEX_TRANSPORT_ZERO_REUSE','1')
    old=np.zeros((4,2,3),np.float32);stage=old.copy();source=old.copy()
    assert t._species_work(cp,old,stage,source,3,None)[1] == [(0,1,4)]
    source[1,0,1]=1;stage[2,1,0]=1;old[3,0,0]=-0.0
    assert t._species_work(cp,old,stage,source,3,None)[1] == []
    source.fill(0);stage.fill(0);old.fill(0)
    assert t._species_work(cp,old,stage,None,3,object())[1] == []
    cp.cuda.get_current_stream=lambda: SimpleNamespace(is_capturing=lambda:True)
    assert t._species_work(cp,old,stage,None,3,None)[1] == []


def test_classifier_signature_accepted_by_cupy_without_device(monkeypatch):
    cupy = pytest.importorskip('cupy')
    from transport_host_launch import HostArrays
    cp = HostArrays()
    host_reduction = cp.ReductionKernel

    def parse_then_host(*args, **kwargs):
        # CuPy parses argument declarations before any device is selected.
        # "unsigned int" is a C++ spelling but is rejected by this API.
        cupy.ReductionKernel(*args, **kwargs)
        return host_reduction(*args, **kwargs)

    cp.ReductionKernel = parse_then_host
    monkeypatch.setattr(t, '_ZERO_CLASSIFIER', None)
    monkeypatch.setattr(t, '_ZERO_LEVEL_REDUCER', None)
    monkeypatch.setenv('WOOF_HEX_TRANSPORT_ZERO_REUSE', '1')
    old = np.zeros((4, 2, 3), np.float32)
    assert t._species_work(cp, old, old, None, 3, None)[1] == [(0, 1, 4)]


@pytest.fixture(scope='module')
def host_launch(tmp_path_factory):
    from transport_host_launch import HostLaunch
    return HostLaunch(tmp_path_factory.mktemp('transport-host'),t._CUDA_SOURCE)


def transport_case(xp, case, nlev=7, ncells=17):
    """Closed ring with nonzero signed fluxes, sharp tracer extrema and source."""
    rng=np.random.default_rng(62291);ntr=11;ne=ncells
    def array(v):return xp.ascontiguousarray(xp.asarray(v))
    def f(shape,lo,hi):return array(rng.uniform(lo,hi,shape).astype(np.float32))
    ids=np.arange(ncells,dtype=np.int32)
    mesh=SimpleNamespace(n_vertices=ncells,
        n_edges_on_cell=array(np.full(ncells,2,np.int32)),
        edges_on_cell=array(np.stack([ids,(ids-1)%ncells],axis=1)),
        cells_on_cell=array(np.stack([(ids+1)%ncells,(ids-1)%ncells],axis=1)),
        cells_on_edge=array(np.stack([ids,(ids+1)%ncells],axis=1)),
        edge_sign_on_cell=array(np.tile(np.array([1,-1],np.float32),(ncells,1))),
        dv_edge=f((ne,),.9,1.1),area_cell=f((ncells,),.9,1.1))
    context=SimpleNamespace(inv_area_cell=array(1/mesh.area_cell),validate=lambda **kw:None)
    coefficients=SimpleNamespace(adv_coefs=array(np.tile(np.array([.5,.5],np.float32),(ne,1))),
        adv_coefs_3rd=array(np.tile(np.array([.1,-.1],np.float32),(ne,1))),
        n_adv_cells_for_edge=array(np.full(ne,2,np.int32)),adv_cells_for_edge=mesh.cells_on_edge,
        validate=lambda:None)
    old=f((ntr,nlev,ncells),0,.02);old[1:4]=0;old[6:8]=0
    stage=old.copy();source=f(old.shape,-.0001,.0001);source[1:4]=0;source[6:8]=0
    velocity=f((nlev,ne),-.02,.02);vertical=f((nlev+1,ncells),-.002,.002)
    vertical[0]=0;vertical[-1]=0
    if case=='all-zero':old[:]=0;stage[:]=0;source=None
    if case=='source-reactivation':source[2,nlev//2,ncells//2]=.001
    if case=='stage-reactivation':stage[2,nlev//2,ncells//2]=.001
    if case=='signed-zero':old[2,nlev//2,ncells//2]=-0.0
    if case=='subnormal':old[2,nlev//2,ncells//2]=np.array(64,np.uint32).view(np.float32)
    if case=='nonfinite-flow':velocity[2,2]=np.array(0x7fc12345,np.uint32).view(np.float32)
    args=(mesh,context,old,stage,f((nlev,ncells),.8,1.2),f((nlev,ncells),.8,1.2),velocity,vertical,.125)
    kw=dict(coefficients=coefficients,fzm=f((nlev,),.4,.6),fzp=f((nlev,),.4,.6),rdzw=f((nlev,),.5,1),
        config_scalar_adv_order=3,config_scalar_vadv_order=3,config_coef_3rd_order=.25,
        scalar_tendency=source,rho_old_prevalidated=True,indices_prevalidated=True)
    return args,kw


def compare_paths(xp, monkeypatch, case, order, density, regional=False):
    args,kw=transport_case(xp,case)
    if regional:
        kw.update(config_apply_lbcs=True,bdy_mask_cell=xp.asarray(np.arange(17,dtype=np.int32)%8),
                  bdy_mask_edge=xp.asarray(np.arange(17,dtype=np.int32)%7))
    results=[]
    for width,reuse in [('1','0'),('3','1'),('0','1'),(None,'1')]:
        if width is None:monkeypatch.delenv('WOOF_HEX_LIMITER_SPECIES_PER_PASS',raising=False)
        else:monkeypatch.setenv('WOOF_HEX_LIMITER_SPECIES_PER_PASS',width)
        monkeypatch.setenv('WOOF_HEX_TRANSPORT_ZERO_REUSE',reuse)
        state=args[3].copy();rounds=[]
        for rk in (1,2,3):
            arguments=(*args[:3],state,*args[4:])
            try:
                result=t.advance_scalar_transport_cuda_v841(*arguments,**kw,rk_step=rk,
                    config_time_integration_order=order,config_scalar_advection=True,
                    config_monotonic=True,config_positive_definite=False,config_split_dynamics_transport=density)
            except ValueError as error:
                assert case=='nonfinite-flow' and density and rk==3
                assert str(error)=='target_density must remain finite and strictly positive'
                rounds.append(('refused',str(error)))
                break
            state=result.scalars
            rounds.append((xp.asnumpy(state).tobytes(),xp.asnumpy(result.density).tobytes()))
        results.append(rounds)
    assert results[0]==results[1]==results[2]==results[3]


@pytest.mark.parametrize('case',['mixed','all-zero','source-reactivation','stage-reactivation','signed-zero','subnormal','nonfinite-flow'])
@pytest.mark.parametrize('order,density,hoist,regional',[(2,False,'0',False),(3,True,'0',True),(2,True,'1',True),(3,False,'1',False)])
def test_host_kernel_launches_bitwise(host_launch,monkeypatch,case,order,density,hoist,regional):
    from transport_host_launch import HostArrays
    cp=HostArrays()
    monkeypatch.setattr(t,'_cupy',lambda:cp)
    monkeypatch.setattr(t._v823,'_cupy',lambda:cp)
    monkeypatch.setattr(t._v823,'_f32',lambda x,name:np.ascontiguousarray(x,dtype=np.float32))
    monkeypatch.setattr(t._v823,'_i32',lambda x:np.ascontiguousarray(x,dtype=np.int32))
    monkeypatch.setattr(t,'_ZERO_CLASSIFIER',None)
    monkeypatch.setattr(t,'_ZERO_LEVEL_REDUCER',None)
    monkeypatch.setattr(t,'_launch',host_launch)
    monkeypatch.setenv('WOOF_HEX_EDGE_HOIST',hoist)
    monkeypatch.setenv('WOOF_HEX_HOIST_LPT','4')
    compare_paths(cp,monkeypatch,case,order,density,regional)


@pytest.mark.gpu
@pytest.mark.parametrize('case',['mixed','all-zero','source-reactivation','stage-reactivation','signed-zero','subnormal','nonfinite-flow'])
@pytest.mark.parametrize('order,density,hoist,regional',[(2,False,'0',False),(3,True,'0',True),(2,True,'1',True),(3,False,'1',False)])
def test_device_kernel_launches_bitwise(monkeypatch,case,order,density,hoist,regional):
    if os.environ.get('WOOF_HEX_TRANSPORT_BATCH_GPU_TEST')!='1':
        pytest.skip('set WOOF_HEX_TRANSPORT_BATCH_GPU_TEST=1 inside a card grant')
    import cupy as cp
    from woof.hex import numerics
    numerics.set_mode(os.environ.get('HEX_TRANSPORT_TEST_NUMERICS','strict'),
                      source='transport bitwise test')
    monkeypatch.setenv('WOOF_HEX_EDGE_HOIST',hoist)
    monkeypatch.setenv('WOOF_HEX_HOIST_LPT','4')
    compare_paths(cp,monkeypatch,case,order,density,regional)
