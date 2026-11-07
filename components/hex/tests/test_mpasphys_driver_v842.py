"""Source-derived control/clock tests; these are not forecast gate receipts."""
from dataclasses import asdict
from fractions import Fraction
import ctypes
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from woof.hex.mpasphys.driver_v842 import (
    Alarm, Backend, PhysicsDriver, PhysicsManager, SlotRegistry, interval_seconds,
    resolve_namelist,
)
from woof.hex.physics_suites import load_table, resolve_physics


@pytest.fixture(scope='module')
def native():
    path = os.environ.get('MPAS_PHYSICS_REFERENCE_DIR')
    if path is None:
        pytest.skip('set MPAS_PHYSICS_REFERENCE_DIR to owned standalone build directory')
    path = Path(path)
    assert (path / 'reference.json').is_file(), 'build standalone authority first'
    return path


def deck(suite='none', **options):
    values = {'config_physics_suite':suite, **options}
    return '&physics\n' + '\n'.join(f'{k} = {v!r}' for k,v in values.items()) + '\n/'


def native_control(native, suite, overrides):
    slots = ('microphysics','convection','pbl','gwdo','radiation_lw','radiation_sw',
             'cloudiness','surface_layer','land_surface')
    spec = load_table()['mpasphys_v842']['slots']
    values = [suite] + [overrides.get(spec[s]['selector'],'suite') for s in slots]
    proc = subprocess.run([str(native/'control')],input='\n'.join(values)+'\n',
                          capture_output=True,text=True)
    return proc, slots


_CONTROL_CASES = [('none',{}),('mesoscale_reference',{}),('convection_permitting',{})]
for slot, row in load_table()['mpasphys_v842']['slots'].items():
    if row['selector'] is None:continue
    for scheme in row['schemes']:
        overrides = {row['selector']:row.get('option_values',{}).get(scheme,scheme)}
        if slot in ('pbl','land_surface') and scheme != 'off':
            overrides['config_sfclayer_scheme'] = 'sf_monin_obukhov_rev'
        _CONTROL_CASES.append(('none', overrides))
_CONTROL_CASES += [
    ('none', {'config_pbl_scheme':'bl_ysu'}),
    ('none', {'config_lsm_scheme':'sf_noah'}),
    ('none', {'config_microp_scheme':'mp_nonexistent'}),
    ('none', {'config_radt_lw_scheme':'rrtmg_lw','config_radt_cld_scheme':'off'}),
    ('mesoscale_reference', {'config_microp_scheme':'mp_thompson_aerosols'}),
]


@pytest.mark.parametrize('suite,overrides',_CONTROL_CASES)
def test_control_matches_pristine_fortran(native,suite,overrides):
    proc, slots = native_control(native,suite,overrides)
    if proc.returncode:
        with pytest.raises(ValueError):
            resolve_namelist(deck(suite,**overrides))
    else:
        result = resolve_namelist(deck(suite,**overrides))
        lines = proc.stdout.splitlines()
        assert [result.selectors[s] for s in slots] == lines[:9]
        assert result.moist_physics == (lines[9]=='T')


@pytest.mark.parametrize('dt,period',[(360,1800),(700,1800),(1000,360),(360,360),(90,135)])
@pytest.mark.parametrize('delayed',[False,True])
@pytest.mark.parametrize('lookahead',[False,True])
def test_alarm_matches_pristine_clock(native,dt,period,delayed,lookahead):
    times = np.arange(0,10000,dt,dtype=np.int64)
    # A gap verifies reset skips missed alarms without moving the phase.
    times[3:] += period*4
    flags = np.zeros(len(times),dtype=np.int32)
    nexts = np.zeros(len(times),dtype=np.int64)
    lib = ctypes.CDLL(str(native/'libclock.so'))
    lib.schedule.argtypes = [ctypes.c_int64,ctypes.c_int64,ctypes.c_int,
                             ctypes.c_void_p,ctypes.c_int64,ctypes.c_int,
                             ctypes.c_void_p,ctypes.c_void_p]
    start = period if delayed else 0
    ahead = dt if lookahead else 0
    lib.schedule(period,start,len(times),times.ctypes.data,ahead,1,
                 flags.ctypes.data,nexts.ctypes.data)
    alarm = Alarm(Fraction(period),Fraction(start))
    got_flags,got_nexts=[],[]
    for now in times:
        got_flags.append(int(alarm.ringing(Fraction(int(now)),Fraction(ahead))))
        alarm.reset(Fraction(int(now)),Fraction(ahead))
        got_nexts.append(int(alarm.next_ring))
    assert got_flags == flags.tolist()
    assert got_nexts == nexts.tolist()


def test_independent_intervals_pbl_and_bucket_lookahead():
    config = resolve_namelist(deck('mesoscale_reference',
        config_radtlw_interval='00:30:00',config_radtsw_interval='00:20:00',
        config_conv_interval='00:15:00',config_pbl_interval='00:10:00',
        config_bucket_update='01:00:00'))
    manager = PhysicsManager(config,700,streams={'output':'01:00:00'})
    records = [manager.advance(t) for t in range(0,4200,700)]
    assert [i for i,r in enumerate(records) if r.radiation_lw] == [0,3]
    assert [i for i,r in enumerate(records) if r.radiation_sw] == [0,2,4]
    assert [i for i,r in enumerate(records) if r.convection] == [0,2,3,4]
    assert [i for i,r in enumerate(records) if r.radiation_bucket] == [5]
    assert manager.scheme_dt['pbl'] == 600
    assert manager.alarms['pbl'].next_ring == 0  # Native never queries PBL alarm.
    assert records[5].diagnostics


def test_restore_preserves_alarm_phase_and_rejects_mismatch():
    config = resolve_namelist(deck('mesoscale_reference'))
    a = PhysicsManager(config,700)
    a.advance(0); a.advance(700); a.advance(1400)
    b = PhysicsManager(config,700)
    b.restore_restart_state(json.loads(json.dumps(a.restart_state())))
    assert a.advance(2100) == b.advance(2100)
    payload = b.restart_state()
    payload['alarms']['radiation_lw']['next_ring']='1'
    before = b.restart_state()
    with pytest.raises(ValueError):
        b.restore_restart_state(payload)
    assert b.restart_state() == before
    with pytest.raises(ValueError):
        b.advance(2100)


class RecordingHooks:
    def __init__(self):
        self.operations=[]
    def invoke(self,operation,call):
        self.operations.append(operation)
    def restart_state(self):
        return {}
    def restore_restart_state(self,payload):
        assert payload == {}


def test_driver_order_switch_and_post_transport_boundary():
    config = resolve_namelist(deck('mesoscale_reference'),backends={'radiation_lw':'hex-cuda'})
    hooks = RecordingHooks()
    registry = SlotRegistry()
    switched=[]
    for slot in config.selectors:
        if not config.active(slot):
            continue
        def callback(call):
            hooks.operations.append(call.slot)
            if call.slot=='radiation_lw':
                switched.append(call.backend)
        registry.register(slot,config.selectors[slot],config.backends[slot],callback)
    driver = PhysicsDriver(config,PhysicsManager(config,360),registry,hooks)
    driver.initialize(None)
    hooks.operations.clear()
    driver.begin_step(itimestep=1,time_seconds=0,pools=None)
    assert hooks.operations == [
        'physics_timetracker_state','physics_update_surface','o3climatology_from_MPAS',
        'allocate_forall_physics','MPAS_to_physics','allocate_cloudiness','cloudiness',
        'allocate_radiation_sw','radiation_sw','allocate_radiation_lw','radiation_lw',
        'deallocate_cloudiness','deallocate_radiation_sw','deallocate_radiation_lw',
        'allocate_sfclayer','surface_layer','deallocate_sfclayer','allocate_lsm',
        'land_surface','deallocate_lsm','allocate_seaice','seaice',
        'deallocate_seaice','atmphys_sfc_diagnostics','allocate_pbl','pbl',
        'deallocate_pbl','allocate_gwdo','gwdo','deallocate_gwdo',
        'update_convection_step1','allocate_convection','convection',
        'deallocate_convection','update_convection_step2','deallocate_forall_physics']
    assert switched == [Backend.HEX_CUDA]
    pbl_call = next(row for row in driver.calls if row['operation'] == 'pbl')
    assert pbl_call['flags']['radiation_lw'] is True
    assert pbl_call['flags']['convection'] is True
    for substep in (1,2,3):
        for rk in (1,2,3):
            driver.physics_get_tend(rk_step=rk,dynamics_substep=substep,pools=None)
    driver.finish_step(pools=None)
    assert hooks.operations[-5:] == ['physics_get_tend'] + [
        'post_transport_prepare','pre_microphysics','microphysics','post_microphysics']
    driver.begin_step(itimestep=2,time_seconds=360,pools=None)
    assert driver.calls[0]['operation'] == 'physics_timetracker_state'
    assert not any(c['slot']=='radiation_lw' for c in driver.calls)
    assert next(row for row in driver.calls if row['operation']=='pbl')['flags']['radiation_lw'] is False
    assert any(c['slot']=='pbl' for c in driver.calls)
    with pytest.raises(RuntimeError, match='before the first RK stage'):
        driver.physics_get_tend(rk_step=2,dynamics_substep=1,pools=None)
    driver.physics_get_tend(rk_step=1,dynamics_substep=1,pools=None)
    assert [c['operation'] for c in driver.calls].count('physics_get_tend') == 1


def test_tendency_handoff_releases_driver_reference_and_refuses_second_consumer():
    import weakref
    class Sources:
        pass
    config = resolve_namelist(deck())
    hooks = RecordingHooks()
    driver = PhysicsDriver(config, PhysicsManager(config,360), SlotRegistry(), hooks)
    driver.initialize(None)
    driver.begin_step(itimestep=1,time_seconds=0,pools=None)
    sources = Sources()
    reference = weakref.ref(sources)
    driver._call = lambda *args, **kwargs: sources
    runtime_sources = driver.physics_get_tend(rk_step=1,dynamics_substep=1,pools=None)
    del sources
    driver.release_held_tendencies()
    assert reference() is runtime_sources
    with pytest.raises(RuntimeError, match='already handed'):
        driver.physics_get_tend(rk_step=2,dynamics_substep=1,pools=None)
    del runtime_sources
    assert reference() is None


def test_suite_none_assembles_zero_sources_and_missing_registration_fails_early():
    config=resolve_namelist(deck())
    hooks=RecordingHooks()
    driver=PhysicsDriver(config,PhysicsManager(config,360),SlotRegistry(),hooks)
    driver.initialize(None)
    driver.begin_step(itimestep=1,time_seconds=0,pools=None)
    driver.physics_get_tend(rk_step=1,dynamics_substep=1,pools=None)
    driver.finish_step(pools=None)
    assert hooks.operations == ['physics_get_tend']
    assert driver.restart_state()['initialized']
    on=resolve_namelist(deck('mesoscale_reference'))
    with pytest.raises(RuntimeError,match='not registered'):
        PhysicsDriver(on,PhysicsManager(on,360),SlotRegistry(),hooks)
    assert hooks.operations == ['physics_get_tend']


def test_backend_off_masks_native_options():
    config=resolve_namelist(deck('mesoscale_reference'),backends={'convection':'off'})
    assert config.execution_options['config_convection_scheme']=='off'
    assert config.selectors['convection']=='cu_ntiedtke'
    assert not PhysicsManager(config,360).advance(0).convection
    with pytest.raises(ValueError):
        resolve_namelist(deck(),backends={'unknown':'hex-cuda'})
    with pytest.raises(ValueError):
        resolve_namelist(deck(config_unknown=True))


def test_backend_namelist_uses_the_same_resolved_slot_switch():
    text=deck('mesoscale_reference')+'''\n&hex_mpasphys
 config_convection_backend='off'
 config_radt_lw_backend='hex-cuda'
 config_pbl_backend='native-cpu'
 /'''
    config=resolve_namelist(text)
    assert config.backends['convection'] == Backend.OFF
    assert config.backends['radiation_lw'] == Backend.HEX_CUDA
    assert config.backends['pbl'] == Backend.NATIVE_CPU
    assert config.execution_options['config_convection_scheme'] == 'off'
    assert config.execution_options['config_radt_lw_scheme'] == 'rrtmg_lw'
    assert resolve_namelist(text,backends={'radiation_lw':'hex-cuda'}) == config
    with pytest.raises(ValueError,match='conflicting'):
        resolve_namelist(text,backends={'radiation_lw':'native-cpu'})
    with pytest.raises(ValueError,match='unknown'):
        resolve_namelist(deck()+"\n&hex_mpasphys config_unknown_backend='off' /")


def test_default_and_original_kernel_digest_unchanged():
    assert resolve_physics().is_default()
    from woof.hex.cuda_physics_v841 import CUDA_PHYSICS_V841_KERNEL_SHA256
    # Extracted from _CUDA_SOURCE at lane BASE a256ff6, with its same maths header.
    # Moved by the Morrison signed-fallout validator (signed-fallout check
    # fd1d2cbb): validate_wsm6_surface_v841_f32 gains a per-bucket
    # signed_fallout flag. WSM6 buckets keep the strict positive-sum checks;
    # the old word is now also finiteness-checked. Validation only, no state
    # arithmetic. Lane-reported source hash fb0aa7da...e4b.
    assert CUDA_PHYSICS_V841_KERNEL_SHA256 == 'fb0aa7da49c1a5e7f5d6bacce6503214d1e550b9a60c4faaaeff237e2d719e4b'


@pytest.mark.parametrize('interval,seconds', [('none',None),('01:30:00',5400),
    ('2_01:00:00',176400),('00:00:00.125',Fraction(1,8))])
def test_fixed_intervals(interval,seconds):
    assert interval_seconds(interval)==seconds


def test_microphysics_only_skips_top_driver_but_keeps_manager_and_transport():
    config=resolve_namelist(deck(config_microp_scheme='mp_wsm6'))
    assert config.moist_physics and not config.driver_physics
    registry=SlotRegistry(); hooks=RecordingHooks()
    registry.register('microphysics','mp_wsm6','native-cpu',
                      lambda c: hooks.operations.append(c.slot))
    driver=PhysicsDriver(config,PhysicsManager(config,360),registry,hooks)
    driver.initialize(None); hooks.operations.clear()
    driver.begin_step(itimestep=1,time_seconds=0,pools=None)
    assert hooks.operations == ['physics_timetracker_state','physics_update_surface']
    driver.finish_step(pools=None)
    assert hooks.operations[-4:] == [
        'post_transport_prepare','pre_microphysics','microphysics','post_microphysics']


def test_oml_logical_selector_backend_mask_and_pbl_interval():
    config=resolve_namelist('&physics\nconfig_physics_suite="mesoscale_reference"\nconfig_oml1d=.true.\nconfig_pbl_interval="0_00:12:00"\n/')
    assert config.selectors['oml']=='oml1d' and config.execution_options['config_oml1d'] is True
    assert PhysicsManager(config,360).scheme_dt['oml']==720
    off=resolve_namelist('&physics\nconfig_physics_suite="mesoscale_reference"\nconfig_oml1d=.true.\n/',backends={'oml':'off'})
    assert off.execution_options['config_oml1d'] is False
    only=resolve_namelist('&physics\nconfig_physics_suite="none"\nconfig_oml1d=.true.\n/')
    assert not only.moist_physics and not only.driver_physics

@pytest.mark.parametrize('scheme',['sf_noah','sf_noahmp'])
@pytest.mark.parametrize('backend',['off','native-cpu','hex-cuda'])
def test_seaice_switch_preserves_land_slot_and_native_order(scheme,backend):
    config=resolve_namelist(deck('none',config_lsm_scheme=scheme,
        config_sfclayer_scheme='sf_monin_obukhov_rev',config_pbl_interval='00:12:00'),
        backends={'seaice':backend})
    assert config.selectors['seaice']==scheme
    assert config.execution_options['config_lsm_scheme']==scheme
    assert config.active('land_surface')
    assert config.active('seaice')==(backend!='off')
    hooks=RecordingHooks();registry=SlotRegistry();seen=[]
    for slot in config.selectors:
        if config.active(slot):
            def callback(call):
                hooks.operations.append(call.slot);seen.append(call)
            registry.register(slot,config.selectors[slot],config.backends[slot],callback)
    driver=PhysicsDriver(config,PhysicsManager(config,360),registry,hooks)
    driver.initialize(None);hooks.operations.clear()
    driver.begin_step(itimestep=1,time_seconds=0,pools=None)
    if backend=='off':
        assert 'allocate_seaice' not in hooks.operations and 'seaice' not in hooks.operations
    else:
        start=hooks.operations.index('seaice')
        assert hooks.operations[start-1:start+3]==['allocate_seaice','seaice','deallocate_seaice','atmphys_sfc_diagnostics']
        assert next(c for c in seen if c.slot=='seaice').dt==720
    assert hooks.operations.index('land_surface')<hooks.operations.index('atmphys_sfc_diagnostics')
    parent_off=resolve_namelist(deck('none',config_lsm_scheme=scheme,
        config_sfclayer_scheme='sf_monin_obukhov_rev'),backends={'land_surface':'off','seaice':'hex-cuda'})
    assert not parent_off.active('seaice')

@pytest.mark.parametrize('parent_backend',['native-cpu','hex-cuda'])
@pytest.mark.parametrize('backend',['off','native-cpu','hex-cuda'])
def test_ngw_dispatch_is_embedded_once_or_ordered_after_parent(parent_backend,backend):
    config=resolve_namelist(deck('none',config_gwdo_scheme='bl_ugwp_gwdo',
        config_ngw_scheme=True,config_sfclayer_scheme='sf_monin_obukhov_rev',
        config_pbl_interval='00:12:00'),backends={'gwdo':parent_backend,'ngw':backend})
    assert config.selectors['ngw']=='ugwp_ngw'
    embedded=parent_backend==backend=='native-cpu'
    assert config.native_embedded('ngw')==embedded
    assert config.execution_options['config_ngw_scheme']==(backend!='off')
    hooks=RecordingHooks();registry=SlotRegistry();seen=[]
    for slot in config.selectors:
        if config.active(slot) and not config.native_embedded(slot):
            def callback(call):
                hooks.operations.append(call.slot);seen.append(call)
            registry.register(slot,config.selectors[slot],config.backends[slot],callback)
    driver=PhysicsDriver(config,PhysicsManager(config,360),registry,hooks)
    driver.initialize(None);hooks.operations.clear()
    driver.begin_step(itimestep=1,time_seconds=0,pools=None)
    parent=next(c for c in seen if c.slot=='gwdo')
    assert parent.dt==720 and parent.options['config_ngw_scheme']==embedded
    separate=backend!='off' and not embedded
    assert ('snapshot_gwdo_tendencies' in hooks.operations)==separate
    assert ('ngw' in hooks.operations)==separate
    if separate:
        ix=hooks.operations.index('gwdo')
        assert hooks.operations[ix-1:ix+3]==['snapshot_gwdo_tendencies','gwdo','ngw','deallocate_gwdo']
        assert next(c for c in seen if c.slot=='ngw').dt==720
    wrong_parent=resolve_namelist(deck('none',config_gwdo_scheme='bl_ysu_gwdo',config_ngw_scheme=True))
    assert not wrong_parent.active('ngw')
    parent_off=resolve_namelist(deck('none',config_gwdo_scheme='bl_ugwp_gwdo',config_ngw_scheme=True),backends={'gwdo':'off'})
    assert not parent_off.active('ngw')
