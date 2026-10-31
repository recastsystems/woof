"""Real-door selection and stream declarations, without claiming forecast identity."""
import site_support
from pathlib import Path
from types import SimpleNamespace
import pytest
from woof.hex.mpasphys.forecast_v842 import native_route, read_streams, Stream, stream_input_bindings, bind_stream_inputs
from woof.hex.mpasphys.pool_v842 import RegistryLayout


def test_default_native_clock_uses_init_and_preserves_explicit_start_and_restart(tmp_path):
    import netCDF4
    import numpy as np
    from woof.hex.mpasphys.forecast_v842 import bind_init_start, namelist_option
    init = tmp_path / 'init.nc'
    stamp = '2023-03-31_12:00:00'
    with netCDF4.Dataset(init, 'w') as dataset:
        dataset.createDimension('Time', 1)
        dataset.createDimension('StrLen', 64)
        variable = dataset.createVariable('xtime', 'S1', ('Time', 'StrLen'))
        variable[:] = np.array(list(stamp.ljust(64)), dtype='S1')[None, :]
    text = bind_init_start('', init)
    assert namelist_option(text, 'nhyd_model', 'config_start_time') == stamp
    explicit = '&nhyd_model\n config_start_time="2024-01-01_00:00:00"\n/\n'
    assert bind_init_start(explicit, init) == explicit
    restart = '&restart\n config_do_restart=.true.\n/\n'
    assert bind_init_start(restart, init) == restart
    with pytest.raises(ValueError, match='no readable xtime'):
        bind_init_start('', tmp_path / 'missing.nc')


def arguments(tmp_path, text='', suite=None, physics=None, default_backend=None):
    path=tmp_path/'namelist.atmosphere'
    path.write_text(text)
    return SimpleNamespace(physics_namelist=path,physics_suite=suite,physics=physics,
                           physics_default_backend=default_backend)


def test_absent_suite_preserves_default_and_explicit_mpasmaps_native(tmp_path):
    assert native_route(arguments(tmp_path))[0] is None
    config,_=native_route(arguments(tmp_path,suite='mesoscale_reference'))
    assert config.suite=='mesoscale_reference'
    # The route's default is the GPU port; the alias route takes it like the namelist route.
    assert str(config.backends['microphysics'])=='hex-cuda'
    text='&physics\nconfig_physics_suite="mesoscale_reference"\n/'
    assert native_route(arguments(tmp_path,text))[0].backends==config.backends
    referee,_=native_route(arguments(tmp_path,suite='mesoscale_reference',default_backend='native-cpu'))
    assert set(map(str,referee.backends.values()))=={'native-cpu'}


def test_command_and_native_namelist_conflict_cannot_silently_select_different_suite(tmp_path):
    with pytest.raises(ValueError,match='suites differ'):
        native_route(arguments(tmp_path,'&physics\nconfig_physics_suite="none"\n/',suite='mesoscale_reference'))


def test_explicit_fork_alias_keeps_native_suite_and_namelist_bytes(tmp_path):
    text='&physics\nconfig_physics_suite="mesoscale_reference"\n/'
    config, overlay=native_route(arguments(tmp_path,text,suite='rrfs_mesoscale_reference'))
    assert config.native_suite=='mesoscale_reference'
    assert config.pin_set=='rrfs'
    assert overlay==text
    default,_=native_route(arguments(tmp_path,text))
    from woof.hex.mpasphys import driver_v842
    assert type(default) is driver_v842.DriverConfig
    assert default.suite=='mesoscale_reference'


@pytest.mark.parametrize('suite', [None, 'convection_permitting'])
def test_public_mpasphys_backend_selects_registry_default_or_explicit_suite(tmp_path, suite):
    a = arguments(tmp_path, '', suite=suite)
    a.physics_backend = 'mpasphys'
    config, text = native_route(a, default_mode='correct')
    assert config.suite == (suite or 'mesoscale_reference')
    assert config.mode == 'correct'
    assert config.backends['microphysics'].value == 'hex-cuda'


def test_explicit_fork_alias_rejects_a_different_native_suite(tmp_path):
    with pytest.raises(ValueError,match='suites differ'):
        native_route(arguments(tmp_path,'&physics\nconfig_physics_suite="convection_permitting"\n/',
                               suite='rrfs_mesoscale_reference'))


def test_native_overrides_keep_other_namelist_groups_and_bool_kinds(tmp_path):
    a=arguments(tmp_path,'&physics\nconfig_physics_suite="none"\n/\n&nhyd_model\nconfig_dt=450\n/',
                physics=['config_microp_scheme="mp_wsm6"','config_microp_re=.false.'])
    config,text=native_route(a)
    assert config.selectors['microphysics']=='mp_wsm6'
    assert config.execution_options['config_microp_re'] is False
    assert 'config_dt=450' in text


@pytest.mark.parametrize('option,slot',[('config_oml_backend','oml'),
    ('config_sfclayer_backend','surface_layer'),('config_microp_backend','microphysics')])
def test_backend_cli_overlay_preserves_native_namelist_bytes(tmp_path,option,slot):
    text='&physics\nconfig_physics_suite="mesoscale_reference"\n/\n&nhyd_model\nconfig_dt=1200\n/'
    a=arguments(tmp_path,text,physics=[option+"='native-cpu'"])
    config,overlay=native_route(a)
    assert str(config.backends[slot])=='native-cpu'
    assert all(str(value)=='hex-cuda' for name,value in config.backends.items() if name!=slot)
    assert a.physics_namelist.read_text()==text
    assert '&hex_mpasphys' in overlay


def test_actual_registry_expands_restart_and_file_members(tmp_path):
    source=site_support.mpas_source()
    if not source.exists():pytest.skip('pinned Registry is required')
    config,text=native_route(arguments(tmp_path,suite='mesoscale_reference'))
    layout=RegistryLayout(source,config,text,dict(nCells=2,nEdges=3,nVertices=2,
                       maxEdges=6,maxEdges2=12,vertexDegree=3,nVertLevels=4))
    (tmp_path/'fields.txt').write_text('u\nqv\n# comment\npressure\n')
    streams=tmp_path/'streams.atmosphere'
    streams.write_text('<streams><immutable_stream name="restart" type="input;output" filename_template="r.nc" output_interval="0_01:00:00"/><stream name="output" type="output" filename_template="o.nc" output_interval="0_01:00:00" filename_interval="output_interval"><file name="fields.txt"/></stream></streams>')
    restart,output=read_streams(streams,layout)
    assert restart.directions==('input','output')
    assert 'scalars' in restart.fields and 'theta_m' in restart.fields
    assert output.fields==('u','qv','pressure')
    assert output.filename_interval=='0_01:00:00'
    assert restart.filename_interval=='none'


def test_native_output_delegates_alarm_steps_and_initial_restart_policy():
    from woof.hex.mpasphys.runtime_v842 import initialize_native_output, write_native_output
    calls=[]
    native=SimpleNamespace(initialize_state_time=lambda:calls.append(('time',)),
        reset_alarm=lambda name:calls.append(('reset',name)),
        alarm_ringing=lambda name:name=='output:out',
        transfer_stream=lambda name,**kw:calls.append(('write',name,kw)))
    streams=[SimpleNamespace(name=n,directions=('output',),output_interval=interval)
             for n,interval in [('restart','00:10:00'),('output','00:10:00'),('diagnostics','none')]]
    initialize_native_output(native,streams)
    assert calls==[('time',),('reset','restart:out')]
    receipt={'streams':[]};write_native_output(native,streams,step=2,receipt=receipt)
    assert calls[-2:]==[('write','output',{'read':False,'level':1}),('reset','output:out')]
    assert receipt['streams']==[{'name':'output','step':2}]
    calls.clear();initialize_native_output(native,streams,restarting=True)
    assert calls==[('time',),('reset','restart:out'),('reset','output:out')]


def test_native_input_binding_validates_active_stream_and_preserves_output_template(tmp_path):
    source=tmp_path/'source.nc';source.write_bytes(b'native input words')
    streams=(Stream('input',('input',),'ignored.nc','initial_only','none',()),
             Stream('surface',('input','output'),'s.$Y-$M-$D_$h.$m.$s.nc','01:00:00','01:00:00',()))
    bindings=stream_input_bindings([f'input={source}',f'surface={source}'])
    bound=bind_stream_inputs(streams,bindings)
    assert bound[0].input_file==str(source.resolve())
    assert bound[1].filename==streams[1].filename
    with pytest.raises(ValueError,match='active input'):bind_stream_inputs(streams,{'missing':source})
    with pytest.raises(ValueError,match='duplicate'):stream_input_bindings([f'input={source}',f'input={source}'])
    with pytest.raises(ValueError,match='NAME=FILE'):stream_input_bindings(['input'])
    with pytest.raises(ValueError,match='does not exist'):stream_input_bindings([f'input={tmp_path/"missing"}'])


def test_dual_direction_input_is_owned_copy_and_native_tokens_are_preserved(tmp_path):
    from woof.hex.mpasphys.runtime_v842 import native_stream_filename, expand_input_filename
    source=tmp_path/'ref.nc';source.write_bytes(bytes(range(256)))
    out=tmp_path/'out';out.mkdir()
    stream=Stream('restart',('input','output'),'restart.$Y-$M-$D_$h.$m.$s.nc',
                  'initial_only','01:00:00',(),input_file=str(source))
    template=native_stream_filename(stream,out,'2024-02-29_12:34:56')
    assert template==str(out/stream.filename)
    staged=out/'restart.2024-02-29_12.34.56.nc'
    assert staged.read_bytes()==source.read_bytes() and not staged.is_symlink()
    staged.write_bytes(b'forecast changed its own file')
    assert source.read_bytes()==bytes(range(256))
    with pytest.raises(ValueError,match='already differs'):native_stream_filename(stream,out,'2024-02-29_12:34:56')
    assert expand_input_filename('$Y-$d-$S-$B','2024-02-29_12:34:56')=='2024-060-45296-all'
    with pytest.raises(ValueError,match='invalid native'):expand_input_filename('$z','2024-02-29_12:34:56')


def test_native_input_only_reads_reference_but_writable_stream_cannot_follow_foreign_symlink(tmp_path):
    from woof.hex.mpasphys.runtime_v842 import native_stream_filename
    source=tmp_path/'reference.nc';source.write_bytes(b'pristine')
    out=tmp_path/'out';out.mkdir()
    stream=Stream('input',('input',),'unused','initial_only','none',(),input_file=str(source))
    assert native_stream_filename(stream,out,'2024-01-01_00:00:00')==str(source)
    (out/'danger.nc').symlink_to(source)
    stream=Stream('output',('output',),'danger.nc','none','01:00:00',())
    with pytest.raises(ValueError,match='escapes'):native_stream_filename(stream,out,'2024-01-01_00:00:00')
    stream=Stream('output',('output',),'../reference.nc','none','01:00:00',())
    with pytest.raises(ValueError,match='escapes'):native_stream_filename(stream,out,'2024-01-01_00:00:00')


@pytest.mark.parametrize('suite', ['none', 'mesoscale_reference', 'convection_permitting'])
def test_native_dycore_adapter_preserves_suite_and_dynamics_validation(suite):
    from woof.hex.config_v841 import V841DryDycoreConfig
    from woof.hex.errors import ConfigurationRefusal
    from woof.hex.mpasphys.config_v842 import NativeMpasDycoreConfigV842
    config = NativeMpasDycoreConfigV842(config_physics_suite=suite,
                                      config_moist_physics=True)
    config.validate()
    assert config.config_physics_suite == suite and config.config_moist_physics
    with pytest.raises(ConfigurationRefusal, match='config_dt'):
        NativeMpasDycoreConfigV842(config_physics_suite=suite, config_dt=-1).validate()
    if suite != 'none':
        with pytest.raises(ConfigurationRefusal, match='no-physics dry authority'):
            V841DryDycoreConfig(config_physics_suite=suite).validate()


def test_native_held_sources_upload_before_contiguous_and_are_cached():
    import numpy as np
    from woof.hex.mpasphys.runtime_v842 import ResidentNativeCoupling
    class DeviceArray(np.ndarray):
        pass
    uploads=[]
    def upload(value):
        uploads.append(value)
        return np.asarray(value).view(DeviceArray)
    def contiguous(value):
        assert isinstance(value, DeviceArray)
        return np.array(value, order='C', copy=True).view(DeviceArray)
    host=np.array([[1.,-0.],[np.nextafter(np.float32(0),np.float32(1)),3.]],
                  dtype=np.float32,order='F')
    sources=SimpleNamespace(rho=host,rho_u=host,rho_theta=host,scalars=host[None])
    calls=[]
    session=SimpleNamespace(put=lambda *a,**kw:calls.append((a,kw)),
        driver=SimpleNamespace(physics_get_tend=lambda **kw:sources),
        native=SimpleNamespace(get_field=lambda *a: host),
        field=lambda *a: host)
    coupling=ResidentNativeCoupling.__new__(ResidentNativeCoupling)
    coupling.session=session
    coupling.cp=SimpleNamespace(asarray=upload,ascontiguousarray=contiguous)
    coupling.sources=None
    coupling.widen_sources=None  # one card: no multi-card source widening
    coupling.exchange_sources=None  # nor halo exchange of the held sources
    kwargs=dict(rk_step=1,dynamics_substep=1,state=SimpleNamespace(rho=host),
                saved=None,diagnostics=SimpleNamespace(h_edge=host))
    result=coupling.stage(**kwargs)
    assert len(uploads)==5 and len(calls)==2
    for name in ('rho','rho_u','rho_theta','scalars'):
        actual=getattr(result,name)
        assert actual.flags.c_contiguous
        np.testing.assert_array_equal(actual.view(np.uint32),getattr(sources,name).view(np.uint32))
    np.testing.assert_array_equal(result.diabatic_theta.view(np.uint32),host.view(np.uint32))
    kwargs['rk_step']=2
    assert coupling.stage(**kwargs) is result
    assert len(uploads)==5 and len(calls)==2


def test_native_accumulated_reflectivity_resets_after_all_streams():
    import numpy as np
    from woof.hex.mpasphys.runtime_v842 import write_native_output
    words=np.array([12.0,3.0,0.0],dtype=np.float32)
    events=[]
    def write(name,**kwargs):
        events.append(('write',name,words.copy()))
    def put(pool,name,value,**kwargs):
        events.append(('clear',name));words[...] = value
    native=SimpleNamespace(_fields={('diag_physics','refl10cm_1km_max'):
        (words.dtype,words.shape,1)},alarm_ringing=lambda name:True,
        transfer_stream=write,reset_alarm=lambda name:None,put_field=put)
    streams=[SimpleNamespace(name=name,directions=('output',),output_interval='00:20:00')
             for name in ('diagnostics','restart','output')]
    write_native_output(native,streams,step=1,receipt={'streams':[]})
    assert [event[:2] for event in events]==[
        ('write','diagnostics'),('write','restart'),('write','output'),('clear','refl10cm_1km_max')]
    for event in events[:-1]:np.testing.assert_array_equal(event[2],[12,3,0])
    np.testing.assert_array_equal(words.view(np.uint32),[0,0,0])
    words[...] = 12;events.clear()
    write_native_output(native,streams,step=0,receipt={'streams':[]})
    assert all(event[0]=='write' for event in events)
    np.testing.assert_array_equal(words,[12,12,12])


def test_lagged_heating_restart_roundtrip_uses_native_stream_storage(tmp_path):
    """Exercise restart heating through actual native I/O; Registry.xml:704,2041."""
    import os
    import numpy as np
    from woof.hex.mpasphys.native_backend import NativePhysics
    from woof.hex.mpasphys.runtime_v842 import _configure_streams
    from test_mpas_native_backend import prepare_stream_block
    library=os.environ.get('MPAS_NATIVE_LIBRARY')
    if not library:
        pytest.skip('production native stream library is required')
    words=np.array([[1,0x80000000,0x3f800000,0xbe800000],
                    [0x80000001,0x1f7fffff,0x3f800001,0x7f7fffff]],dtype=np.uint32)
    layout=SimpleNamespace(configs={'config_start_time':'2000-01-01_00:00:00'},
        fields=[dict(name=name,pool='tend',components=[])
                for name in ('rt_diabatic_tend','tend_sfc_pressure')])
    with NativePhysics(library,test_only=True) as native:
        prepare_stream_block(native,[1,2,3,4],4)
        native.decomposition('nCells',global_ids=np.arange(1,5,dtype=np.int32),owned=4)
        native.put_field('tend','rt_diabatic_tend',words.view(np.float32))
        native.register_field('tend','rt_diabatic_tend',
                              dimensions=('nVertLevels','nCells'),decomposed=True)
        stream=Stream('restart',('output',),'heating.nc','none','01:00:00',
                      ('rt_diabatic_tend','tend_sfc_pressure'))
        missing=_configure_streams(native,[stream],tmp_path,layout)
        assert missing=={'restart':['tend_sfc_pressure']}
        native.transfer_stream('restart',read=False)
        native.put_field('tend','rt_diabatic_tend',np.zeros(words.shape,np.float32))
        restore=Stream('restore',('input',),'heating.nc','initial_only','none',
                       ('rt_diabatic_tend',))
        assert _configure_streams(native,[restore],tmp_path,layout)=={'restore':[]}
        native.transfer_stream('restore',read=True)
        np.testing.assert_array_equal(native.get_field('tend','rt_diabatic_tend').view(np.uint32),words)


@pytest.mark.parametrize('step,steps,moving,lazy', [
    (0, 8, None, True), (8, 8, None, True), (3, 8, object(), True),
    (3, 8, None, False),
])
def test_native_publication_keeps_consuming_boundaries(step, steps, moving, lazy):
    from woof.hex.mpasphys.runtime_v842 import native_publication_required
    def unexpected_query(name):
        raise AssertionError('mandatory boundary needs no alarm query')
    native=SimpleNamespace(alarm_ringing=unexpected_query)
    assert native_publication_required(native, [], step=step, steps=steps, moving=moving, lazy=lazy)


def test_native_publication_defers_only_without_active_output_alarm():
    from woof.hex.mpasphys.runtime_v842 import native_publication_required
    calls=[]; rings=set()
    def alarm(name):
        calls.append(name)
        return name in rings
    native=SimpleNamespace(alarm_ringing=alarm)
    streams=[SimpleNamespace(name=n,directions=d,output_interval=i) for n,d,i in [
        ('input', ('input',), '00:01:00'), ('disabled', ('output',), 'none'),
        ('output', ('output',), '00:01:00'), ('diagnostics', ('output',), '00:00:30')]]
    def required():
        return native_publication_required(native,streams,step=3,steps=8,moving=None,lazy=True)
    assert not required()
    assert calls==['output:out','diagnostics:out']
    calls.clear();rings.add('diagnostics:out')
    assert required()
    assert calls==['output:out','diagnostics:out']
    # Policy inspection must not consume the alarm; the writer owns its reset.
    assert required()



def test_deck_invariant_stream_is_read_by_default(tmp_path):
    """The RRFS deck's invariant file is bound without --stream-input (the 129 sea-ice cell land/soil defect)."""
    from woof.hex.mpasphys.forecast_v842 import deck_stream_inputs
    deck, store = tmp_path / 'deck', tmp_path / 'store'
    deck.mkdir(); store.mkdir()
    streams = deck / 'streams.atmosphere'
    streams.write_text(
        '<streams>\n'
        '<immutable_stream name="invariant" type="input" filename_template="x1.2562.invariant.nc" input_interval="initial_only" />\n'
        '<immutable_stream name="input" type="input" filename_template="x1.2562.init.nc" input_interval="initial_only" />\n'
        '<immutable_stream name="restart" type="input;output" filename_template="restart.$Y-$M-$D_$h.$m.$s.nc" input_interval="initial_only" />\n'
        '<immutable_stream name="iau" type="input" filename_template="AmB.nc" packages="iau" input_interval="initial_only" />\n'
        '<stream name="surface" type="input" filename_template="sfc_update.nc" input_interval="none" />\n'
        '<immutable_stream name="ugwp_oro_data_in" type="input" filename_template="x1.2562.ugwp_oro_data.nc" input_interval="initial_only" />\n'
        '</streams>\n')
    for name in ('x1.2562.invariant.nc', 'AmB.nc', 'sfc_update.nc', 'x1.2562.ugwp_oro_data.nc'):
        (deck / name).write_bytes(b'x')
    (store / 'init.nc').write_bytes(b'x')
    (deck / 'x1.2562.init.nc').symlink_to(store / 'init.nc')
    init = deck / 'x1.2562.init.nc'
    # Only the fixed-name start-time streams the door does not bind itself; IAU (package), surface (not at start),
    # restart (time tokens) and input stay with their own machinery.
    assert deck_stream_inputs(streams, init, explicit={'ugwp_oro_data_in'}) == {
        'invariant': (deck / 'x1.2562.invariant.nc').resolve()}
    assert set(deck_stream_inputs(streams, init)) == {'invariant', 'ugwp_oro_data_in'}
    # An explicit binding is never replaced.
    assert deck_stream_inputs(streams, init, bound={'invariant': 'mine'}, explicit={'ugwp_oro_data_in'}) == {}
    # A deck without the file: the init's own directory (as named, not its link target) is searched next.
    bare = tmp_path / 'bare'; bare.mkdir()
    (bare / 'streams.atmosphere').write_text(streams.read_text())
    assert deck_stream_inputs(bare / 'streams.atmosphere', init, explicit={'ugwp_oro_data_in'}) == {
        'invariant': (deck / 'x1.2562.invariant.nc').resolve()}
    assert deck_stream_inputs(bare / 'streams.atmosphere', store / 'init.nc', explicit={'ugwp_oro_data_in'}) == {}


def test_a_deck_start_time_input_the_run_reads_and_cannot_find_is_refused_by_name(tmp_path):
    """THE BREAKAGE: an init named by its resolved path (the door refused links) left the deck's invariant file
    behind; the run went on with the init's land use and soil category (12,800,256 of 29,644,818 words by f006)."""
    from woof.hex.mpasphys.forecast_v842 import deck_stream_inputs, unread_deck_inputs_refusal
    deck, store = tmp_path / 'deck', tmp_path / 'store'
    deck.mkdir(); store.mkdir()
    streams = deck / 'streams.atmosphere'
    streams.write_text(
        '<streams>\n'
        '<immutable_stream name="invariant" type="input" filename_template="x1.2562.invariant.nc" input_interval="initial_only" />\n'
        '<immutable_stream name="input" type="input" filename_template="x1.2562.init.nc" input_interval="initial_only" />\n'
        '</streams>\n')
    (store / 'init.nc').write_bytes(b'x')
    missing = []
    assert deck_stream_inputs(streams, store / 'init.nc', missing=missing) == {}
    assert missing == [('invariant', 'x1.2562.invariant.nc', (deck, store))]
    refusal = unread_deck_inputs_refusal(streams, missing, {'input', 'invariant'})
    assert 'stream invariant reads x1.2562.invariant.nc' in refusal and str(store) in refusal
    assert '--stream-input NAME=FILE' in refusal
    # A stream the Registry's packages leave inactive is never read, so it is not refused.
    assert unread_deck_inputs_refusal(streams, missing, {'input'}) is None
    # The same file beside the init as named (a link into the store) is found, and nothing is missing.
    run = tmp_path / 'run'; run.mkdir()
    (run / 'x1.2562.init.nc').symlink_to(store / 'init.nc')
    (store / 'invariant.nc').write_bytes(b'x')
    (run / 'x1.2562.invariant.nc').symlink_to(store / 'invariant.nc')
    missing = []
    assert deck_stream_inputs(streams, run / 'x1.2562.init.nc', missing=missing) == {
        'invariant': (store / 'invariant.nc').resolve()}
    assert missing == []


def _rrfs_init(tmp_path, producer):
    import json
    init = tmp_path / producer / 'init.nc'
    init.parent.mkdir()
    init.write_bytes(b'')
    (init.parent / 'init.nc.provenance.json').write_text(json.dumps(
        {'schema': 'gpuwm-hex.init-provenance/rrfs-v1', 'producer': {'key': producer}}))
    link = tmp_path / producer / 'deck.init.nc'
    try:
        link.symlink_to(init)
    except OSError:
        link = init
    return link


def test_hrrrv5_forecast_refuses_an_init_from_the_other_rrfs_pin(tmp_path):
    # hrrrv5da 2026-10-29: 1d6e370e start deck and member inits queued under the 44c0849 forecast.
    namelist = tmp_path / 'namelist.atmosphere'
    namelist.write_text("&physics\n config_physics_suite = 'hrrrv5'\n/\n")
    old = _rrfs_init(tmp_path, 'rrfs-mpas-1d6e370e')
    arguments = SimpleNamespace(physics_namelist=namelist, physics_backend='mpasphys', init=old)
    with pytest.raises(ValueError, match='rrfs-mpas-1d6e370e.*neither pin'):
        native_route(arguments, default_mode='correct')


def test_init_pin_check_passes_its_own_producer_and_unreceipted_inits(tmp_path):
    from woof.hex.mpasphys.forecast_v842 import init_pin_refusal
    from woof.hex.physics_suites import load_table
    key = 'mpasphys_rrfs_44c0849'
    route = load_table()[key]
    init_pin_refusal(_rrfs_init(tmp_path, 'rrfs-mpas-44c0849'), key, route, 'hrrrv5')
    bare = tmp_path / 'analysis.nc'
    bare.write_bytes(b'')
    init_pin_refusal(bare, key, route, 'hrrrv5')
    init_pin_refusal(None, key, route, 'hrrrv5')
    with pytest.raises(ValueError, match='--physics-pin-set rrfs-44c0849'):
        init_pin_refusal(_rrfs_init(tmp_path, 'rrfs-mpas-1d6e370e'), key, route, 'hrrrv5')
