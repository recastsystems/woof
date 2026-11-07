"""Bulk smoke source, canonical doors, native units and actual continuation."""
from dataclasses import asdict, replace
from datetime import datetime
from types import SimpleNamespace
import json

import numpy as np
import pytest

from conftest import requires_gpu
from woof.config import RunConfig, validate_chem_config


def cfg(**values):
    options=dict(nx=12,ny=10,nz=6,dx=90.0,dy=120.0,ztop=8000.0,dt=.25,
                 run_seconds=2.0,ifire=2,sr_x=4,sr_y=3)
    options.update(values)
    return RunConfig(**options)


def test_shortcut_and_explicit_profile_have_one_canonical_identity():
    a=cfg(fire_smoke=True)
    b=cfg(chem_sets="sfire_smoke")
    assert asdict(a)==asdict(b)
    validate_chem_config(a)
    assert a.chem_sources==""
    from woof.chem_table import load
    table=load(a)
    assert len(table.rows)==1 and table.rows[0].units=="ug kg-1"
    assert table.processes==("emission.sfire",)
    assert not any(d.output_name=="PM2_5_DRY" and
                   any("fire_smoke" in t['species'] for t in d.terms) for d in table.diagnostics)


def test_named_microphysics_and_smoke_share_the_initializer():
    shortcut = cfg(mp_physics="0", fire_smoke=True)
    explicit = cfg(mp_physics=0, chem_sets="sfire_smoke")
    assert type(shortcut.mp_physics) is int
    assert shortcut.mp_physics == 0
    assert shortcut.chem_sets == "sfire_smoke"
    assert shortcut.fire_smoke is True
    assert asdict(shortcut) == asdict(explicit)


def test_bulk_profile_does_not_launder_invalid_shortcut_or_force_pbl():
    assert cfg(fire_smoke=1).fire_smoke==1
    with pytest.raises(ValueError,match="PBL"):
        validate_chem_config(cfg(chem_sets="sfire_smoke_mixed"))
    validate_chem_config(cfg(chem_sets="sfire_smoke_mixed",bl_pbl_physics=1))
    from woof.experiment import experiment_from_run_config
    with pytest.raises(ValueError,match="actual ifire=2"):
        experiment_from_run_config(cfg(ifire=0,fire_smoke=True), datetime(2026, 1, 1))


def test_zero_cache_augmentation_preserves_every_existing_payload(tmp_path):
    from test_prepared_cache import _fixture
    from woof.ingest.prepared_cache import write_prepared_cache, PreparedCacheReader
    from woof.ingest.sfire_smoke_cache import augment_sfire_smoke_cache
    initial,met,boundaries=_fixture()
    old=cfg(nx=2,ny=2,nz=2,ifire=0,sr_x=0,sr_y=0)
    identity={'domain_config':{'run':asdict(old)},'source_identity':{'name':'test'},
              'source_manifest_sha256':'a'*64,'static_cache_sha256':'b'*64}
    source=tmp_path/'source'
    write_prepared_cache(source,identity=identity,initial_result=initial,met=met,boundaries=boundaries)
    before=(source/'header.json').read_bytes()
    wanted={'run':asdict(replace(old,fire_smoke=True))}
    receipt=augment_sfire_smoke_cache(source,tmp_path/'bulk',wanted)
    header=json.loads((tmp_path/'bulk/header.json').read_text())
    reader=PreparedCacheReader(tmp_path/'bulk',expected_identity=header['identity'])
    prior=PreparedCacheReader(source,expected_identity=identity)
    for key in prior.arrays:
        assert prior.read_array(key).tobytes()==reader.read_array(key).tobytes()
    for name in receipt['added_zero_fields']:
        assert not np.any(reader.read_array('state/'+name))
    assert len(receipt['added_zero_fields'])==16
    assert (source/'header.json').read_bytes()==before
    assert header['identity']['source_identity']==identity['source_identity']
    bad={'run':asdict(replace(old,fire_smoke=True,dx=999.0))}
    with pytest.raises(ValueError,match="meteorology or geometry"):
        augment_sfire_smoke_cache(source,tmp_path/'wrong',bad)
    assert not (tmp_path/'wrong').exists()


def test_native_tracer_option_import_carries_one_bulk_row_on_the_tree(tmp_path):
    from woof.experiment import load_experiment
    from woof.namelist_import import import_namelists
    from test_namelist_import import INPUT_TEXT, _pair
    text = (INPUT_TEXT.replace('&domains', '&domains\n sr_x=0,4, sr_y=0,3,')
            .replace('&dynamics', '&dynamics\n tracer_opt=0,3,')
            .replace('cu_physics = 1, 0,', 'cu_physics = 0, 0,')
            + '\n&fire ifire=0,2, fire_fuel_read=0,0, /\n')
    translated, _ = import_namelists(*_pair(tmp_path, inp=text))
    path = tmp_path/'bulk.toml'; path.write_text(translated)
    experiment = load_experiment(path)
    assert all(d.run.fire_smoke and d.run.chem_sets == 'sfire_smoke' for d in experiment.domains)
    assert [d.run.ifire for d in experiment.domains] == [0,2]
    with pytest.raises(ValueError, match='tracer_opt'):
        # Package 2 is the ported non-Chem passive tracer set (0747f1ef907);
        # a package neither port carries is still refused by name.
        import_namelists(*_pair(tmp_path, inp=text.replace('tracer_opt=0,3,','tracer_opt=0,1,')))


def test_history_quote_contains_the_exact_bulk_metadata_inventory():
    from woof.io.history_layout import produced_history_shapes, history_frame_bytes
    from woof.core.chem_history import history_schema
    off=cfg(ifire=0,sr_x=0,sr_y=0,moist=True)
    on=replace(off,fire_smoke=True)
    extra={name:shape for name,(shape,_d,_u) in history_schema(on).items()}
    assert extra == {'FIRE_SMOKE':(6,10,12),'fire_smoke':(6,10,12),
        'SFIRE_SMOKE_FUEL_SOURCE':(10,12),'SFIRE_SMOKE_INJECTED':(10,12),
        'SFIRE_SMOKE_EMITTED':(10,12),'SFIRE_SMOKE_SFC':(10,12),'SFIRE_SMOKE_COLUMN':(10,12)}
    before=produced_history_shapes(off);after=produced_history_shapes(on)
    assert {name:shape for name,shape in after.items() if name not in before} == extra
    assert history_frame_bytes(on)-history_frame_bytes(off) == sum(4*np.prod(s) for s in extra.values())+256*len(extra)


def _source_context(cp, scheme):
    from woof.chem_table import load
    from woof.core.chem_sfire import ALLOCATES
    from woof.core.chem_context import allocation_shape
    from tools.sfire_wrf471_oracle.fixture import load as fixture
    f=fixture('atm/original_smoke_column_defect')
    rho=cp.asarray(f['rho'][:-1]);dz=cp.asarray(f['dz8w'][:-1])
    nz,ny,nx=rho.shape;rx,ry=int(f['sr_x']),int(f['sr_y'])
    options=cfg(nx=nx,ny=ny,nz=nz,sr_x=rx,sr_y=ry,fire_smoke=True,
                fire_tracer_smoke=.015,fire_smk_scheme=scheme,fire_smk_peak=35.0)
    table=load(options)
    grid=SimpleNamespace(time_seconds=.25,interior=(slice(None),slice(None)),
        data={'burnt_area_dt':cp.asarray(f['burnt']),'fgip':cp.asarray(f['fuel'])})
    fire=SimpleNamespace(grid=grid,sr_x=rx,sr_y=ry)
    state=SimpleNamespace(physics=SimpleNamespace(fire=fire),chem=SimpleNamespace(),ht=cp.zeros((ny,nx),cp.float32))
    diag={a.name:cp.zeros(allocation_shape(a,1,nz,ny,nx),dtype=a.dtype) for a in ALLOCATES}
    airq=cp.linspace(.001,.025,rho.size,dtype=cp.float32).reshape(rho.shape)
    values={'rho':rho,'dryrho':rho/(1+airq),'dz8w':dz,
        'z_at_w':cp.concatenate((cp.zeros((1,ny,nx),cp.float32),cp.cumsum(dz,axis=0))),
        'msftx':cp.ones((ny,nx),cp.float32),'msfty':cp.ones((ny,nx),cp.float32)}
    tracer=cp.zeros_like(rho)
    ctx=SimpleNamespace(state=state,table=table,cfg=options,diag=diag,
        clock=SimpleNamespace(curr_secs=0.0),met=lambda n:values[n],field=lambda row:tracer)
    return ctx,tracer,f


@requires_gpu
@pytest.mark.parametrize('scheme',[0,1])
def test_native_increment_and_dry_unit_bridge_conserve_fuel_source_once(scheme):
    import cupy as cp
    from woof.core import chem_sfire
    ctx,tracer,f=_source_context(cp,scheme)
    chem_sfire.init(ctx);chem_sfire.step(ctx,.25,1)
    inc=cp.asnumpy(ctx.diag['sfire_native_increment'])
    if scheme==0:
        original=f['tracer'][0][:-1]
        assert inc[:,:,1].tobytes()==original[:,:,1].tobytes()
    expected=(inc*np.float32(1.e6)*
              np.float32(cp.asnumpy(ctx.met('rho'))/cp.asnumpy(ctx.met('dryrho'))))
    assert cp.asnumpy(tracer).tobytes()==expected.tobytes()
    fuel=float(ctx.diag['sfire_source_expected'].sum().item())
    injected=float(ctx.diag['sfire_source_injected'].sum().item())
    assert fuel>0 and injected==pytest.approx(fuel,rel=2e-6)
    print(json.dumps({'control':'native-smoke-unit-bridge','scheme':scheme,
        'converted_words':int(tracer.size),'expected_fuel_kg':fuel,
        'injected_geometry_kg':injected,'different_converted_words':0}))
    saved={k:cp.asnumpy(v).tobytes() for k,v in ctx.diag.items()}
    old=cp.asnumpy(tracer).tobytes();chem_sfire.step(ctx,.25,1)
    assert cp.asnumpy(tracer).tobytes()==old
    assert {k:cp.asnumpy(v).tobytes() for k,v in ctx.diag.items()}==saved
    # The same buffer gathers a second tile at the same atmosphere time.
    tracer.fill(0);ctx.diag['sfire_source_lasttime'].fill(0)
    chem_sfire.step(ctx,.25,1)
    assert cp.asnumpy(tracer).tobytes()==old


@requires_gpu
@pytest.mark.parametrize('field', ['burnt_area_dt', 'fgip'])
@pytest.mark.parametrize('invalid', [np.nan, np.inf])
def test_nonfinite_source_refuses_before_tracer_or_diagnostics_mutate(field, invalid):
    import cupy as cp
    from woof.core import chem_sfire
    ctx,tracer,_ = _source_context(cp,0)
    chem_sfire.init(ctx)
    tracer.fill(np.float32(.125))
    for name,array in ctx.diag.items():
        if name != 'sfire_source_lasttime':
            array.fill(1.25)
    before={name:cp.asnumpy(array).tobytes() for name,array in ctx.diag.items()}
    tracer_before=cp.asnumpy(tracer).tobytes()
    clock_before=ctx.state.chem.sfire_source_lasttime
    ctx.state.physics.fire.grid.data[field][2,2]=np.float32(invalid)
    with pytest.raises(ValueError,match='finite consumed fraction and native fuel loading'):
        chem_sfire.step(ctx,.25,1)
    assert cp.asnumpy(tracer).tobytes()==tracer_before
    assert {name:cp.asnumpy(array).tobytes() for name,array in ctx.diag.items()}==before
    assert ctx.state.chem.sfire_source_lasttime==clock_before


@requires_gpu
def test_actual_coupled_bulk_smoke_survives_rk_and_disk(tmp_path):
    import cupy as cp
    from test_restart import _physics_state,_assert_restart_equal
    from woof.core.dycore import run_steps
    from woof.core.chem_driver import chem_ledger_receipt
    from woof.io import restart
    def make():
        return _physics_state(cp,nx=16,ny=12,nz=16,dx=90.0,dy=120.0,dt=.25,
            mp_physics=1,cu_physics=0,ra_physics=0,ifire=2,sr_x=4,sr_y=3,
            fire_smoke=True,fire_fuel_read=0,fire_topo_from_atm=True,
            fire_num_ignitions=1,fire_ignition_start_x1=720.0,fire_ignition_start_y1=720.0,
            fire_ignition_end_x1=720.0,fire_ignition_end_y1=720.0,
            fire_ignition_start_time1=0.0,fire_ignition_end_time1=0.0,
            fire_ignition_radius1=70.0,fire_ignition_ros1=280.0)
    full,options,driver=make();run_steps(full,options,6)
    assert bool(cp.any(full.chem_fire_smoke>0))
    report=chem_ledger_receipt(full)
    row=report['rows']['fire_smoke'];source=report['sources']['emission.sfire']['fire_smoke']
    assert row['emitted_kg']>0 and source['expected_fuel_source_kg']>0
    assert abs(row['closure_kg'])<1e-10
    assert source['injected_geometry_kg']==pytest.approx(source['expected_fuel_source_kg'],rel=2e-6)
    reference=restart.write_restart(tmp_path/'full.npz',full,options)
    split,options,_=make();run_steps(split,options,3)
    midpoint=restart.write_restart(tmp_path/'mid.npz',split,options)
    resumed,options,_=make();restart.restore_restart(midpoint,resumed,options)
    run_steps(resumed,options,3)
    result=restart.write_restart(tmp_path/'resumed.npz',resumed,options)
    _assert_restart_equal(reference,result)
    with np.load(reference) as archive:
        arrays=[archive[name] for name in archive.files if not name.startswith('__')]
        print(json.dumps({'control':'coupled-smoke-disk-continuation','steps':6,
            'arrays':len(arrays),'elements':sum(a.size for a in arrays),
            'words32':sum(a.nbytes//4 for a in arrays),
            'different_words':0,'ledger':row,'source':source}))


@requires_gpu
def test_first_emission_transports_in_the_same_step_and_is_weather_passive():
    import cupy as cp
    from test_restart import _physics_state
    from woof.core.dycore import run_steps
    values=dict(nx=16,ny=12,nz=16,dx=90.0,dy=120.0,dt=.25,
        mp_physics=1,cu_physics=0,ra_physics=0,ifire=2,sr_x=4,sr_y=3,
        fire_fuel_read=0,fire_topo_from_atm=True,
        fire_num_ignitions=1,fire_ignition_start_x1=720.0,fire_ignition_start_y1=720.0,
        fire_ignition_end_x1=720.0,fire_ignition_end_y1=720.0,
        fire_ignition_start_time1=0.0,fire_ignition_end_time1=0.0,
        fire_ignition_radius1=70.0,fire_ignition_ros1=280.0)
    plain,base,_=_physics_state(cp,**values)
    smoke,options,_=_physics_state(cp,fire_smoke=True,**values)
    run_steps(plain,base,1);run_steps(smoke,options,1)
    initial=cp.asnumpy(smoke.chem0_fire_smoke)
    final=cp.asnumpy(smoke.chem_fire_smoke)
    assert np.any(initial>0), 'first source must survive the RK time-t copy'
    assert np.any((initial==0)&(final>np.float32(1.e-12))), 'new smoke must move in the emitting step'
    names=('u','v','w','thp','php','mup','p','al','alt','qv','qc','qr')
    for name in names:
        assert cp.asnumpy(getattr(plain,name)).tobytes()==cp.asnumpy(getattr(smoke,name)).tobytes(), name
    for name in plain.physics.fire.grid.data:
        assert cp.asnumpy(plain.physics.fire.grid.data[name]).tobytes()==cp.asnumpy(smoke.physics.fire.grid.data[name]).tobytes(), name
    print(json.dumps({'control':'same-step-emission-and-passive-weather',
        'transported_new_cells':int(np.count_nonzero((initial==0)&(final>np.float32(1.e-12)))),
        'weather_arrays':len(names),'different_weather_words':0}))


@requires_gpu
def test_native_volume_import_and_history_units_are_explicit():
    import cupy as cp
    from test_restart import _physics_state
    from woof.core import chem_sfire
    from woof.core.chem_history import history_fields
    from tools.sfire_wrf471_oracle.fixture import words
    state,options,_=_physics_state(cp,cu_physics=0,ra_physics=0,ifire=0,fire_smoke=True)
    native=np.geomspace(np.float32(1.e-38),np.float32(2.0),state.alt.size,dtype=np.float32).reshape(state.alt.shape)
    native.ravel()[::7]=0
    chem_sfire.import_native(state,native)
    dry=np.float32(np.float32(1)/cp.asnumpy(state.alt))
    rho=np.float32(dry*np.float32(np.float32(1)+cp.asnumpy(state.qv)))
    ratio=np.float32(rho/dry)
    expected=np.float32(np.float32(native*np.float32(1.e6))*ratio)
    assert cp.asnumpy(state.chem_fire_smoke).tobytes()==expected.tobytes()
    assert cp.asnumpy(state.chem0_fire_smoke).tobytes()==expected.tobytes()
    output=history_fields(state)
    from woof.core.chem_history import history_schema
    from woof.io.history_layout import produced_history_shapes
    assert {n:tuple(a.shape) for n,(a,_,_) in output.items()} == {
        name:shape for name,(shape,_d,_u) in history_schema(options).items()}
    quoted=produced_history_shapes(options)
    assert all(quoted[n]==tuple(a.shape) for n,(a,_,_) in output.items())
    assert output['fire_smoke'][2]=='g_smoke/kg_air'
    assert output['FIRE_SMOKE'][2]=='ug kg-1'
    returned=cp.asnumpy(output['fire_smoke'][0])
    inverse=np.float32(np.float32(expected*np.float32(1.e-6))/ratio)
    assert returned.tobytes()==inverse.tobytes()
    distance=words(returned,native)
    assert distance['max_ulp']<=2
    print(json.dumps({'control':'native-volume-unit-roundtrip','words':native.size,
        'different_converted_words':0,'roundtrip':distance,
        'roundtrip_cause':'two finite-precision unit conversions; subnormal inputs preserved'}))
    from woof.certify.kernel_manifest import kernel_manifest
    print(json.dumps({'control':'compiled-bulk-smoke-source',
        'manifest':kernel_manifest()['woof.core.chem_sfire:chem_sfire']}))
