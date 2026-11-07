"""Production launch uses the same columns and table path as the WRF grade."""
import numpy as np
import pytest
from conftest import requires_gpu
from woof.verify.kfcup_oracle import configs, load_capture, pack_column_inputs, unpack_outputs, expected_outputs, differing_words

@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize('cfg',configs(),ids=lambda c:c.name)
def test_production_kfcup_launch(cfg):
    import cupy as cp
    from woof.core.kf import launch_kfcup_columns
    cap=load_capture(cfg.index)
    b={k:cp.asarray(v) for k,v in pack_column_inputs(cap).items()}
    launch_kfcup_columns(b,dt=cfg.dt,dx=cfg.dx,ntst=cfg.stepcu,numbins=cfg.numbins,
        thbinsize=cfg.thbinsize,rbinsize=cfg.rbinsize,mindeepfreq=cfg.mindeepfreq,
        minshallowfreq=cfg.minshallowfreq,flags=cfg.flags)
    differences=differing_words(unpack_outputs({k:v.get() for k,v in b.items()}),expected_outputs(cap))
    assert not any(differences.values()),differences


def test_kfcup_workspace_matches_kernel_layout():
    from woof.core.kf import kfcup_workspace_floats
    assert kfcup_workspace_floats(40,72,21)==96*(148*41+21*21+2*21)


def test_nested_cup_memory_counts_both_retained_arenas():
    from types import SimpleNamespace as S
    from woof.core.kf import kfcup_workspace_floats
    from woof.core.preflight import kfcup_column_workspace_bytes
    parent=S(cu_physics=10,nz=40,nx=73,ny=60,numbins=1)
    child=S(cu_physics=10,nz=40,nx=30,ny=30,numbins=3)
    off=S(cu_physics=0)
    profile=S(multiprocessor_count=170)
    exp=S(domains=[S(run=x) for x in (parent,child,off)])
    expected=sum(kfcup_workspace_floats(x.nz,x.nx*x.ny,x.numbins)*4+4
                 for x in (parent,child))
    assert kfcup_column_workspace_bytes(exp,profile=profile)==expected
    assert kfcup_column_workspace_bytes(S(domains=[S(run=off)]),profile=profile)==0

@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize('cfg',configs(),ids=lambda c:c.name)
def test_production_callable_column_words(cfg):
    import cupy as cp
    from types import SimpleNamespace
    from woof.core.kf import KFCumulusPotential, KFCUP_PARAMETER_DEFAULTS
    cap=load_capture(cfg.index)
    arrays={k:cp.asarray(v[:cap.nz].reshape(cap.nz,1,cap.ncol)) for k,v in cap.level_in.items()}
    shape=(cap.nz,1,cap.ncol)
    state=SimpleNamespace(w=cp.asarray(cap.level_in['w'].reshape(cap.nz+1,1,cap.ncol)),elapsed_seconds=0.,
        qi=cp.zeros(shape) if cfg.f_qi else None,qs=cp.zeros(shape) if cfg.f_qs else None,
        qr=cp.zeros(shape) if cfg.f_qr else None)
    # warm_rain is a microphysics constant, separate from whether QI is allocated.
    run=SimpleNamespace(dt=float(cfg.dt),dx=float(cfg.dx),cudt_minutes=float(cfg.stepcu*cfg.dt/60.),
        use_adaptive_time_step=False,mp_physics=1 if cfg.warm_rain else 10,
        **{k:getattr(cfg,k) for k in KFCUP_PARAMETER_DEFAULTS if k!='shallowcu_forced_ra'})
    scheme=KFCumulusPotential()
    scheme.ensure_trigger_history(state)
    b=pack_column_inputs(cap)
    scheme.cup_held[...]=cp.asarray(b['held']).reshape(scheme.cup_held.shape);scheme.cup_iheld[...]=cp.asarray(b['iheld']).reshape(scheme.cup_iheld.shape);scheme.cup_levels[...]=cp.asarray(b['lev_out']).reshape(scheme.cup_levels.shape)
    scheme.w0avg[...]=cp.asarray(cap.level_out['w0avg'].reshape(shape));scheme._history_time=0.
    atm={'u':arrays['u'],'v':arrays['v'],'theta':arrays['th'],'temperature':arrays['t'],
         'dz':arrays['dz8w'],'pressure':arrays['p'],'exner':arrays['pi'],'rho':arrays['rho'],'qv':arrays['qv'],
         'z_interface':cp.asarray(cap.level_in['z_at_w'].reshape(cap.nz+1,1,cap.ncol)),
         'z':arrays['z']}
    fields={k:cp.asarray(v.reshape(1,cap.ncol)) for k,v in cap.surface_in.items()}
    fields['kpbl']=cp.asarray(cap.kpbl.reshape(1,cap.ncol))
    result=scheme(atmosphere=atm,fields=fields,state=state,cfg=run)
    got=unpack_outputs({'held':scheme.cup_held.get().reshape(b['held'].shape),'iheld':scheme.cup_iheld.get().reshape(b['iheld'].shape),'lev_out':scheme.cup_levels.get().reshape(b['lev_out'].shape)})
    diffs=differing_words(got,expected_outputs(cap))
    assert not any(diffs.values()),diffs
    np.testing.assert_array_equal(result.pratec.get().view(np.uint32),
        (cap.surface_out['raincv']/cfg.dt).reshape(1,-1).view(np.uint32))

@requires_gpu
@pytest.mark.gpu
def test_production_cloud_clock_matches_wrf_advance_capture():
    import cupy as cp
    from woof.core.kernels import get_kernel
    from woof.verify.kfcup_oracle import _rows,word
    rows=list(_rows('kfcup-advance.csv'))
    for dt_word in sorted({r['dt'] for r in rows}):
        group=[r for r in rows if r['dt']==dt_word]
        nca=cp.asarray(np.array([word(r['nca_out']) for r in group],np.float32))
        cf=cp.full((3,len(group)),np.float32(.125))
        get_kernel('kfcup','kfcup_cloud_clock')((1,),(128,),
            (nca,cf,word(dt_word),np.int32(nca.size),np.int32(3)))
        expected=np.array([0. if r['cldfra_cup_cleared']=='1' else .125 for r in group],np.float32)
        np.testing.assert_array_equal(cf.get().view(np.uint32),
            np.broadcast_to(expected,(3,len(group))).copy().view(np.uint32))

@requires_gpu
@pytest.mark.gpu
def test_callable_same_shape_domain_change_cold_starts_held_fields():
    import cupy as cp
    from types import SimpleNamespace
    from woof.core.kf import KFCumulusPotential
    a=SimpleNamespace(w=cp.zeros((9,1,2),dtype=cp.float32))
    b=SimpleNamespace(w=cp.zeros((9,1,2),dtype=cp.float32))
    scheme=KFCumulusPotential();scheme.ensure_trigger_history(a)
    scheme.cup_held[...]=7;scheme.cup_levels[...]=9;scheme.cup_iheld[...]=1
    scheme.ensure_trigger_history(b)
    assert bool(cp.all(scheme.cup_levels==0))
    assert bool(cp.all(scheme.cup_iheld==0))
    assert bool(cp.all(scheme.cup_held[0]==-100))
    assert bool(cp.all(scheme.cup_held[4]==2))


def test_kfcup_metadata_names_the_wrf_override_without_ice_addition():
    from woof.core.kf import KFCUP_METADATA, KFCUP_STATE_ROWS
    assert KFCUP_METADATA['selector']==10
    assert KFCUP_METADATA['cadence']=='held-model-clock'
    assert KFCUP_METADATA['radiation_override']['ice_action']=='preserve'
    assert KFCUP_METADATA['radiation_override']['liquid_floor']==1.e-3
    names={r['name'] for r in KFCUP_STATE_ROWS}
    assert len(names)==len(KFCUP_STATE_ROWS)
    assert {'cldfra_cup','shall','cubot','cutop','cupflag'}<=names

@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize('cfg',configs(),ids=lambda c:c.name)
def test_native_rain_clock_uses_captured_raincv_without_rate_round_trip(cfg):
    import cupy as cp
    from woof.core.kf import KFCumulusPotential
    cap=load_capture(cfg.index)
    cv=cap.surface_out['raincv'];count=cv.size
    # Capture fixtures expose the double-rounding defect that this prevents.
    assert np.count_nonzero(((cv/cfg.dt)*cfg.dt).view(np.uint32)!=cv.view(np.uint32))>0
    initial=np.linspace(0,11,count,dtype=np.float32)
    rainc=cp.asarray(initial);pending=cp.asarray(initial);surface=cp.asarray(initial)
    nca=cp.asarray(cap.surface_out['nca']);mask=cp.zeros(count,dtype=cp.float32)
    scheme=KFCumulusPotential()
    expected=initial.copy()
    for step in range(4):
        scheme.advance_rain_clock(rainc=rainc,pending_rainbl=pending,surface_raincv=surface,
            nca=nca,raincv=cp.asarray(cv),expiring=mask,dt=cfg.dt)
        expected=expected+cv
    for a in (rainc,pending,surface):
        np.testing.assert_array_equal(a.get().view(np.uint32),expected.view(np.uint32))

@requires_gpu
@pytest.mark.gpu
def test_real_driver_allocates_held_rows_before_the_first_call_and_uses_native_clock():
    import cupy as cp
    from woof.core.kf import KFCumulusPotential,KFCUP_STATE_ROWS
    from test_physics_driver import _full_state
    scheme=KFCumulusPotential()
    state,cfg,driver=_full_state(cu_physics=10,cumulus=scheme,mp_physics=1,
        sf_sfclay_physics=0,sf_surface_physics=0,bl_pbl_physics=0,dt=60.,nz=40)
    for row in KFCUP_STATE_ROWS:
        value=driver.fields[row['name']]
        shape=state.p.shape if row['shape']=='mass' else state.p.shape[1:]
        assert value.shape==shape
        assert value.dtype==np.dtype(row['dtype'])
        assert bool(cp.all(value==row['initial']))
    ids={r['name']:id(driver.fields[r['name']]) for r in KFCUP_STATE_ROWS}
    cv=load_capture(1).surface_out['raincv'][:cfg.nx*cfg.ny].reshape(cfg.ny,cfg.nx)
    driver.cu_raincv[...]=cp.asarray(cv)
    driver.cu_pratec[...]=cp.asarray(cv/np.float32(cfg.dt))
    driver.cu_nca[...]=np.float32(120)
    driver.fields['cldfra_cup'][...]=np.float32(.25)
    initial=driver.rainc.get()
    driver._advance_cumulus_clock(state,cfg)
    np.testing.assert_array_equal(driver.rainc.get().view(np.uint32),
        (initial+cv).view(np.uint32))
    assert bool(cp.all(driver.fields['cldfra_cup']==0))
    assert ids=={r['name']:id(driver.fields[r['name']]) for r in KFCUP_STATE_ROWS}


@requires_gpu
@pytest.mark.gpu
def test_kfcup_restart_continues_every_carrier_bitwise(tmp_path):
    """An off-calendar checkpoint must retain the packed and published state."""
    import cupy as cp
    from woof.core.dycore import run_steps
    from woof.io import restart
    from test_restart import _physics_state, _assert_restart_equal
    from tilestream.physics_inventory import carrier_manifest
    from woof.io.history_layout import live_state_history_fields
    from woof.core.kfcup_contract import KFCUP_STATE_ROWS
    from woof.io.wrf_output_schema import SCHEME_OUTPUT_FIELDS

    state,cfg,driver=_physics_state(cp,cu_physics=10,numbins=1)
    run_steps(state,cfg,2)
    adapter=driver.cumulus_callable
    # Seed nonzero hold words at a legitimate held-cloud checkpoint so
    # a test cannot pass by silently rebuilding zeros after restoration.
    driver.cu_nca[...] = 50.
    adapter.cup_held[0] = 50.
    adapter.cup_held[9:] = np.float32(.125)
    adapter.cup_iheld[0] = 1
    driver.fields['cldfra_cup'][...] = np.float32(.25)
    adapter.cup_levels[6] = driver.fields['cldfra_cup']
    mid=restart.write_restart(tmp_path/'mid.npz',state,cfg)
    manifest=carrier_manifest(state)
    for name in ('w0avg','cup_held','cup_iheld','cup_levels'):
        assert 'cumulus/'+name in manifest
        assert manifest['cumulus/'+name].shape[-2:]==(cfg.ny,cfg.nx)
    history=live_state_history_fields(state)
    for row in KFCUP_STATE_ROWS:
        assert SCHEME_OUTPUT_FIELDS[row['name']].netcdf_name in history
    run_steps(state,cfg,2)
    reference=restart.write_restart(tmp_path/'reference.npz',state,cfg)

    resumed,other_cfg,other_driver=_physics_state(cp,cu_physics=10,numbins=1)
    restart.restore_restart(mid,resumed,other_cfg)
    restored=other_driver.cumulus_callable
    assert restored._history_state is resumed
    assert restored._ws is None
    run_steps(resumed,other_cfg,2)
    actual=restart.write_restart(tmp_path/'actual.npz',resumed,other_cfg)
    _assert_restart_equal(reference,actual)


def test_stock_kfcup_namelist_preserves_controls_and_off_identity():
    import dataclasses
    from woof.config import RunConfig, validate_run_config
    from woof.io.restart import configuration_echo
    from woof.core.kfcup_contract import KFCUP_PARAMETER_DEFAULTS
    # Off defaults disappear from the persisted identity, preserving the
    # candidate's old default-off identity rather than adding six keys.
    from test_physics_driver import _base_config
    off=_base_config()
    assert not (set(KFCUP_PARAMETER_DEFAULTS)&set(configuration_echo(off)))
    on=dataclasses.replace(off,cu_physics=10,moist=True,numbins=3,
        thbinsize=.1,rbinsize=1.e-4,mindeepfreq=.5,minshallowfreq=.1)
    validate_run_config(on)
    assert configuration_echo(on)['numbins']==3


def test_stock_namelist_import_resolves_all_cup_controls(tmp_path):
    from test_namelist_import import INPUT_TEXT, _pair
    from woof.namelist_import import import_namelists
    from woof.experiment import load_experiment
    source = INPUT_TEXT.replace('cu_physics = 1, 0,', 'cu_physics = 10, 0,')
    source = source.replace('&physics\n', '&physics\n'
        ' numbins=3, thbinsize=0.1, rbinsize=0.0001,\n'
        ' mindeepfreq=0.5, minshallowfreq=0.1, shallowcu_forced_ra=.true.,\n')
    text, _ = import_namelists(*_pair(tmp_path, inp=source))
    path = tmp_path / 'imported.toml'
    path.write_text(text)
    experiment = load_experiment(path)
    run = experiment.root.run
    assert run.cu_physics == 10 and run.cudt_minutes == 5
    assert (run.numbins, run.thbinsize, run.rbinsize,
            run.mindeepfreq, run.minshallowfreq, run.shallowcu_forced_ra) == (
                3, 0.1, 0.0001, 0.5, 0.1, True)
    assert experiment.domain(2).run.cu_physics == 0


@pytest.mark.parametrize("name,default,changed", [("grav_settling", 0, 1), ("kfeta_trigger", 1, 2)])
def test_restart_matches_omitted_fog_kf_defaults_and_binds_active_values(name, default, changed):
    from dataclasses import replace
    from pathlib import Path
    from woof.config import RunConfig
    from woof.io.restart import configuration_echo, _require_config_match, RestartMismatchError
    cfg = RunConfig(nx=4, ny=4, nz=8, dx=1000., dy=1000., ztop=8000., dt=1., run_seconds=2.)
    stored = configuration_echo(cfg)
    assert name not in stored
    assert getattr(cfg, name) == default
    _require_config_match(stored, cfg, Path("checkpoint.npz"))
    with pytest.raises(RestartMismatchError, match=name):
        _require_config_match(stored, replace(cfg, **{name: changed}), Path("checkpoint.npz"))
