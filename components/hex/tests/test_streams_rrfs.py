"""Fork stream contract tests; these are not whole-forecast acceptance."""
import json,os,re,shutil,subprocess
from datetime import datetime,timedelta
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from woof.hex.streams_rrfs import parse_streams,registry_stream,Stream,StreamAlarms,_all_packages,timelevel_query,_REGISTRY
from woof.hex.restart_rrfs import RestartControl,write_stream,read_stream,read_restart,write_done_marker
from woof.hex.output import HistoryField
from woof.hex.output_rrfs import StreamSession

REF=Path(os.environ.get('RRFS_MPAS_REF') or Path(os.environ.get('MPAS_LANES', '.')) / 'ref/rrfs/MPAS-Model-rrfs')
WF=Path(os.environ.get('RRFS_WORKFLOW_REF') or Path(os.environ.get('MPAS_LANES', '.')) / 'ref/rrfs/rrfs-workflow')
START=datetime(2023,3,31,12)

@pytest.mark.parametrize('seed',[0,1,1234567,4294967295])
@pytest.mark.parametrize('ordinal',[0,1,7,31])
@pytest.mark.usefixtures('rrfs_clock')
def test_original_native_file_id(seed,ordinal):
    from woof.hex.output_rrfs import native_file_id
    oracle=os.environ.get('RRFS_NATIVE_FILE_ID')
    if not oracle:pytest.skip('original fork file ID oracle not supplied')
    expected=subprocess.run([oracle,str(seed),str(ordinal)],text=True,capture_output=True,check=True).stdout.strip()
    assert native_file_id(seed,ordinal)==expected

@pytest.mark.parametrize('mode',['never_modify','append','truncate','replace_files','overwrite'])
@pytest.mark.usefixtures('rrfs_clock','rrfs_bridge')
def test_session_preserves_records_and_attributes_in_open_file(tmp_path,mode):
    streams={'sample':Stream('sample',dict(type='output',output_interval='1:00:00',
        filename_interval='none',filename_template='sample.nc',clobber_mode=mode),
        (('var','xtime',''),))}
    alarms=StreamAlarms(streams,START,START,START+timedelta(hours=2))
    session=StreamSession(streams,alarms,RestartControl(False,False),tmp_path,random_seed=1234567)
    mesh=SimpleNamespace(dimensions={},arrays={},attrs={})
    session.publish_due(START,mesh,lambda stream:{},global_attrs={'build_only':7},scratch=tmp_path)
    session.publish_due(START+timedelta(hours=1),mesh,lambda stream:{},global_attrs={'build_only':8},
        record_indices={'sample':1},scratch=tmp_path)
    from woof.hex.output_rrfs import native_file_id
    with read_stream(tmp_path/'sample.nc',scratch=tmp_path) as state:
        assert state.variables['xtime'].text==('2023-03-31_12:00:00','2023-03-31_13:00:00')
        assert state.global_attrs['file_id']==native_file_id(1234567,0)
        assert state.global_attrs['build_only']=={'dtype':'i4','values':[7]}
    assert session._random_ordinal==1

@pytest.mark.parametrize('interval,minutes',[('00:15:01.5',15),('2_01:30:00',2970)])
@pytest.mark.usefixtures('rrfs_clock')
def test_native_stream_build_interval_attribute(tmp_path,interval,minutes):
    oracle=os.environ.get('RRFS_NATIVE_STREAM_CLOCK')
    if not oracle:pytest.skip('native calendar oracle not supplied')
    expected=int(subprocess.run([oracle,'--interval-minutes',interval],text=True,capture_output=True,check=True).stdout.strip())
    stream=Stream('sample',dict(type='output',filename_interval=interval),())
    session=StreamSession({},None,RestartControl(False,False),tmp_path,random_seed=1)
    attrs=session.build_attributes(stream,{'file_id':'inherited','output_interval':999})
    assert attrs['output_interval']==expected==minutes
    assert attrs['file_id']!='inherited'

@pytest.mark.usefixtures('rrfs_clock','rrfs_bridge')
def test_session_file_rollover_and_disabled_global_updates(tmp_path):
    streams={name:Stream(name,dict(type='output',output_interval='1:00:00',
        filename_interval='output',filename_template=name+'.$h.nc',
        gattr_update=update), (('var','xtime',''),))
        for name,update in [('silent','no'),('sample','yes')]}
    alarms=StreamAlarms(streams,START,START,START+timedelta(hours=2))
    session=StreamSession(streams,alarms,RestartControl(False,False),tmp_path,random_seed=1)
    mesh=SimpleNamespace(dimensions={},arrays={},attrs={'mesh_attribute':9})
    from woof.hex.output_rrfs import native_file_id
    for ordinal in range(2):
        when=START+timedelta(hours=ordinal)
        session.publish_due(when,mesh,lambda stream:{},global_attrs={'authority_attribute':7},scratch=tmp_path)
        with read_stream(tmp_path/f'silent.{when:%H}.nc',scratch=tmp_path) as state:
            assert state.global_attrs=={}
        with read_stream(tmp_path/f'sample.{when:%H}.nc',scratch=tmp_path) as state:
            assert state.global_attrs['file_id']==native_file_id(1,ordinal)
            assert state.global_attrs['authority_attribute']=={'dtype':'i4','values':[7]}
    assert session._random_ordinal==2

@pytest.fixture
def rrfs_clock():
    """The stream clock, XML filter and file IDs are Python (mpasphys.rrfs.stream_xml_rrfs); nothing to supply."""
    yield

@pytest.fixture
def rrfs_bridge():
    artifact=os.environ.get('WOOF_HEX_RW_MPAS_STREAM') or shutil.which('rw_mpas_stream')
    if not artifact:pytest.skip('compiled native Rust stream artifact not supplied')
    assert Path(artifact).is_file()

def deck(tmp_path,levels=None):
    if not (WF/'parm/streams.atmosphere').is_file():pytest.skip('pinned rrfs-workflow reference not supplied (RRFS_WORKFLOW_REF)')
    # Same substitutions the workflow performs, with unchanged field-list files.
    source=(WF/'parm/streams.atmosphere').read_text()
    for key,value in {'restart':'12:00:00','history':'1:00:00','diag':'1:00:00','mpasout':'1:00:00','lbc':'1'}.items():
        source=source.replace('@'+key+'_interval@',value)
    if levels is not None:source=source.replace('output_interval="1:00:00" >\n        <file name="stream_list/stream_list.atmosphere.da_state"',f'output_timelevels="{levels}" >\n        <file name="stream_list/stream_list.atmosphere.da_state"')
    (tmp_path/'streams.atmosphere').write_text(source)
    shutil.copytree(WF/'fix/stream_list/hrrrv5',tmp_path/'stream_list',dirs_exist_ok=True)
    return parse_streams(tmp_path/'streams.atmosphere')

@pytest.mark.parametrize('levels',[None,'0-3-15m 4-24','1 2 3 6 9 12 18 24'])
@pytest.mark.usefixtures('rrfs_clock')
def test_workflow_deck_and_lists(tmp_path,levels):
    streams=deck(tmp_path,levels)
    assert not streams['da_state'].immutable
    tokens=(WF/'fix/stream_list/hrrrv5/stream_list.atmosphere.da_state').read_text().split()
    active=_all_packages()
    expected=set()
    for token in tokens:
        expected.update(_REGISTRY['groups'][token] if token in _REGISTRY['arrays'] else (token,))
    assert set(streams['da_state'].fields(active,streams=streams))==expected
    assert streams['da_state'].fields((),streams=streams)==()
    for name in ('output','diagnostics','surface'):
        assert streams[name].members==tuple(('var',token,'') for token in (WF/f'fix/stream_list/hrrrv5/stream_list.atmosphere.{name}').read_text().split())
    if levels:
        assert streams['da_state'].filename(START+timedelta(hours=6),START)=='mpasout.2023-03-31_18.00.00.nc'

@pytest.mark.usefixtures('rrfs_clock','rrfs_bridge')
def test_workflow_manager_order_and_simultaneous_output_ids(tmp_path):
    streams=deck(tmp_path)
    assert list(streams).index('restart')<list(streams).index('da_state')
    captured={}
    dimensions={}
    for name in ('restart','da_state','output','diagnostics'):
        mesh,values=fixture_fields(streams[name],streams)
        dimensions.update(mesh.dimensions)
        captured[name]=values
    mesh=SimpleNamespace(dimensions=dimensions,arrays={},attrs={})
    alarms=StreamAlarms(streams,START,START,START+timedelta(hours=24),active_packages=_all_packages())
    session=StreamSession(streams,alarms,RestartControl(False,False),tmp_path/'out',random_seed=1)
    ordinal=0
    from woof.hex.output_rrfs import native_file_id
    for when in (START,START+timedelta(hours=12)):
        due=[stream for name,stream in streams.items() if alarms.due(name,when)]
        session.publish_due(when,mesh,lambda stream:captured[stream.name],scratch=tmp_path)
        for stream in due:
            with read_stream(session.output_directory/stream.filename(when,START),scratch=tmp_path) as state:
                assert state.global_attrs['file_id']==native_file_id(1,ordinal)
            ordinal+=1
    assert session._random_ordinal==ordinal==7

@pytest.mark.parametrize('restart,cycling,expected,suppressed',[
    (False,False,'input',('restart',)),(False,True,'da_state',('restart','da_state')),
    (True,False,'restart',('restart','output','diagnostics')),
    (True,True,'restart',('restart','output','diagnostics'))])
@pytest.mark.usefixtures('rrfs_clock')
def test_controls_and_initial_alarms(tmp_path,restart,cycling,expected,suppressed):
    c=RestartControl.from_namelist(f'&nhyd_model\n config_do_restart={str(restart).lower()}, config_do_DAcycling={str(cycling).lower()}\n/')
    assert c.initial_stream==expected
    assert c.suppressed_initial_outputs==suppressed
    streams=deck(tmp_path,'0-24')
    alarms=StreamAlarms(streams,START,START-timedelta(hours=6),START+timedelta(hours=24),do_restart=restart,do_DAcycling=cycling,active_packages=_all_packages())
    for name in ('restart','output','diagnostics','da_state'):
        assert alarms.due(name,START)==(name not in suppressed)
    assert alarms.due('da_state',START+timedelta(hours=1))
    for name in ('invariant','input','restart'):
        assert not alarms.due(name,START,'input')
    assert alarms.due('da_state',START,'input')== (not (cycling and not restart))

@pytest.mark.parametrize('spec', ['0-3-15m 4-24 27-36-3','1 2 3 6 9 12 24',
    '0m 15m 1h3m 2 3','1h30m-2h-15m','0-90s-30s','0-1D-3h',
    '0-1-2','4-2','0-6-0','0  1  2','0-30m-0.5m','0-2-1e-2h'])
@pytest.mark.usefixtures('rrfs_clock')
def test_original_fortran_timelevel_words(spec):
    native=os.environ.get('RRFS_NATIVE_TIMELEVELS')
    if not native:pytest.skip('original fork standalone artifact not supplied')
    for seconds in (0,1,29,30,59,60,899,900,1800,3599,3600,5400,7200,10800,21600,86400,129600):
        run=subprocess.run([native,spec,str(seconds)],text=True,capture_output=True)
        try: ours=timelevel_query(spec,seconds)
        except ValueError:
            assert run.returncode!=0
            continue
        assert run.returncode==0,run.stderr
        rows=[[int(word)&0xffffffff for word in line.split()] for line in run.stdout.splitlines()]
        assert rows[:-1]==ours['words']
        assert rows[-1][1:]==ours['query_words']
        assert (rows[-1][0]==0)==ours['matched']
        receipt=os.environ.get('RRFS_TIMELEVEL_RECEIPT')
        if receipt:
            with Path(receipt).open('a') as output:
                output.write(json.dumps({'schema':'rrfs-stream-timelevel-replay/v1',
                    'specification':spec,'seconds':seconds,
                    'compared_words':sum(len(row) for row in rows[:-1])+5,
                    'different_words':0,'native_process_count':1,
                    'whole_model_proof':False})+'\n')

@pytest.mark.parametrize('spec',['',' ','0--2','-2','2-','0-2-','1H','xm','1h2','0-1-1-1',' '.join(['0']*33)])
@pytest.mark.usefixtures('rrfs_clock')
def test_native_parser_errors(spec):
    with pytest.raises(ValueError):timelevel_query(spec,0)
    native=os.environ.get('RRFS_NATIVE_TIMELEVELS')
    if native:assert subprocess.run([native,spec,'0'],capture_output=True).returncode!=0

@pytest.mark.parametrize('spec,expected',[
    ('0-3-15m 4-8', [*range(0,181,15),240,300,360,420,480]),
    ('0m 15m 1h3m 2 3',[0,15,63,120,180]),
    ('1 2 3 6',[60,120,180,360]),
    ('0-1h-30m 3-4-30m',[0,30,60,180,210,240])])
@pytest.mark.usefixtures('rrfs_clock')
def test_native_schedule_gaps_and_endpoints(spec,expected):
    s=Stream('chosen',{'type':'output','output_timelevels':spec,'filename_template':'x.$h.$m.nc'},(('var','xtime',''),))
    alarms=StreamAlarms({'chosen':s},START,START-timedelta(hours=12),START+timedelta(hours=8))
    emitted=[]
    for minute in range(481):
        now=START+timedelta(minutes=minute)
        if alarms.due('chosen',now):
            emitted.append(minute);alarms.reset('chosen',now)
    assert emitted==expected


def fixture_fields(stream,streams=None):
    dims={'StrLen':64,'Time':1}
    fields={}
    for name in stream.fields(_all_packages(),streams=streams):
        if name in ('xtime','initial_time','Time'):continue
        definition=_REGISTRY['fields'][name]
        axes=tuple(d for d in definition['dimensions'] if d!='Time')
        shape=tuple(dims.setdefault(d,2) for d in axes)
        if definition['type']=='text':fields[name]='fixture'
        else:
            size=int(np.prod(shape));dtype=np.int32 if definition['type'] in ('integer','logical') else np.float32
            value=np.arange(size,dtype=dtype).reshape(shape)
            if dtype==np.float32:
                # Negative zero, payload NaN, and subnormal transport.
                value.view(np.uint32).flat[:min(3,size)]=[0x80000000,0x7fc00123,0x00000001][:min(3,size)]
            fields[name]=HistoryField(value,axes)
    mesh=SimpleNamespace(dimensions=dims,arrays={},variable_dimensions={},attrs={'source':'fixture'})
    return mesh,fields

@pytest.mark.parametrize('name',['da_state','output','diagnostics','restart'])
@pytest.mark.usefixtures('rrfs_clock','rrfs_bridge')
def test_every_workflow_active_field_through_rust(tmp_path,name):
    streams=deck(tmp_path);s=streams[name];mesh,fields=fixture_fields(s,streams)
    target=tmp_path/(name+'.nc')
    write_stream(target,s,mesh,fields,START,initial_time=START,active_packages=_all_packages(),streams=streams)
    with read_stream(target,scratch=tmp_path) as f:
        assert set(f.variables)==set(s.fields(_all_packages(),streams=streams))
        for name,field in fields.items():
            if isinstance(field,str):assert f.variables[name].text[0]==field
            else:
                # Rust owns permutation into reversed Fortran dimensions.
                assert f.variables[name].values.tobytes()==field.values.transpose().tobytes(),name
    if 'yes' in s.options.get('output_done_marker',''):
        assert re.fullmatch(r'\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\n',Path(str(target)+'.done').read_text())

@pytest.mark.parametrize('setting,enabled',[('yes',True),('no',False),('YES',True),('junk',True),('yesterday-no',True),('none',False)])
@pytest.mark.usefixtures('rrfs_bridge')
def test_global_attribute_native_substring_policy(tmp_path,setting,enabled):
    s=Stream('fields',{'type':'output','gattr_update':setting,'filename_template':'x.nc'},(('var','xtime',''),))
    mesh=SimpleNamespace(dimensions={},arrays={},attrs={'mesh_attr':1})
    target=tmp_path/'x.nc';write_stream(target,s,mesh,{},START,initial_time=START,global_attrs={'runtime':'yes'})
    with read_stream(target,scratch=tmp_path) as f:
        assert ('runtime' in f.global_attrs)==enabled
        assert ('mesh_attr' in f.global_attrs)==enabled


def test_staged_rrfs_bridge_preserves_words_without_environment_or_path(tmp_path, monkeypatch, rrfs_bridge):
    from woof.hex import engines, restart_rrfs
    artifact = restart_rrfs._bridge()
    directory = tmp_path/'bridges'; directory.mkdir()
    executable = directory/engines.executable_name('rw_mpas_stream')
    shutil.copy2(artifact, executable)
    monkeypatch.delenv('WOOF_HEX_RW_MPAS_STREAM', raising=False)
    monkeypatch.delenv('WOOF_RW_MPAS_STREAM', raising=False)
    monkeypatch.setenv('PATH', '')
    monkeypatch.setattr(engines, 'USER_BRIDGE_DIR', directory)
    assert Path(restart_rrfs._bridge()) == executable.resolve()
    words = np.array([1, 0x80000000, 0x3f800000], dtype=np.uint32)
    stream = Stream('fields', {'type':'output'}, (('var','u',''),))
    mesh = SimpleNamespace(dimensions={'nEdges':3,'nVertLevels':1}, arrays={}, attrs={})
    target = tmp_path/'staged.nc'
    write_stream(target, stream, mesh,
        {'u':HistoryField(words.view(np.float32).reshape(1,3), ('nVertLevels','nEdges'))},
        START, initial_time=START, scratch=tmp_path)
    with read_stream(target, scratch=tmp_path) as state:
        np.testing.assert_array_equal(state.variables['u'].values.view(np.uint32).ravel(), words)


def test_explicit_missing_rrfs_bridge_does_not_fall_through(tmp_path, monkeypatch):
    from woof.hex import engines, restart_rrfs
    monkeypatch.setenv('WOOF_HEX_RW_MPAS_STREAM', str(tmp_path/'missing'))
    with pytest.raises(engines.EngineRefusal, match='explicit setting'):
        restart_rrfs._bridge()


def test_done_marker_rank_and_native_content(tmp_path):
    p=tmp_path/'forecast.nc'
    assert write_done_marker(p,now=START,rank=1) is None
    marker=write_done_marker(p,now=START,rank=1,block_write=True)
    assert marker.read_text()=='2023-03-31 12:00:00\n'


@pytest.mark.usefixtures('rrfs_bridge','rrfs_clock')
def test_warm_state_transport_and_publication_transaction(tmp_path):
    streams=deck(tmp_path,'0-3');s=streams['da_state'];mesh,fields=fixture_fields(s,streams)
    write_stream(tmp_path/s.filename(START,START),s,mesh,fields,START,initial_time=START,active_packages=_all_packages(),streams=streams)
    session=StreamSession.from_namelist('&nhyd_model\nconfig_do_DAcycling=true\n/',tmp_path/'streams.atmosphere',start_time=START,initial_time=START,final_time=START+timedelta(hours=3),output_directory=tmp_path/'out',active_packages=_all_packages())
    with session.read_initial_state(input_directory=tmp_path,scratch=tmp_path) as restored:
        for name,field in fields.items():
            if not isinstance(field,str):assert restored.variables[name].values.tobytes()==field.values.tobytes(),name
    assert not session.alarms.due('da_state',START)
    when=START+timedelta(hours=1)
    with pytest.raises(ValueError,match='missing active native fields'):
        session.publish_due(when,mesh,lambda stream:{})
    assert session.alarms.due('da_state',when)
    # Limit this publication test to the fully supplied DA stream.
    session.streams={'da_state':s};session.alarms.streams=session.streams
    result=session.publish_due(when,mesh,lambda stream:fields)
    assert len(result)==1 and Path(str(result[0])+'.done').is_file()
    assert not session.alarms.due('da_state',when)

@pytest.mark.parametrize('levels',[None,'0-3-15m 4-24','1 2 3 6 12 24'])
@pytest.mark.usefixtures('rrfs_clock')
def test_original_c_parser_unedited_workflow_lists(tmp_path,levels):
    native=os.environ.get('RRFS_NATIVE_XML')
    if not native:pytest.skip('original fork C parser artifact not supplied')
    streams=deck(tmp_path,levels)
    run=subprocess.run([native,'streams.atmosphere'],cwd=tmp_path,text=True,capture_output=True)
    assert run.returncode==0,run.stderr
    rows=[line.split('\t') for line in run.stdout.splitlines()]
    for name,stream in streams.items():
        create=next(row for row in rows if row[:2]==['create',name])
        assert create[2]==stream.options['filename_template']
        assert int(create[-2])==1
        fields=[row[2] for row in rows if row[:2]==['field',name]]
        if not stream.immutable:assert fields==[m[1] for m in stream.members]
        markers=[row for row in rows if row[:3]==['property',name,'output_done_marker']]
        assert bool(markers)==('yes' in stream.options.get('output_done_marker',''))
    if levels:
        assert ['variable','da_state'] in rows
        assert ['property','da_state','output_timelevels',levels] in rows
        create=next(row for row in rows if row[:2]==['create','da_state'])
        assert create[3]=='output'

@pytest.mark.parametrize('direction,inputs,outputs,levels,expected',[
    ('input;output','1:00:00',None,'0-24','1:00:00'),
    ('input;output','initial_only',None,'0-24','output'),
    ('input','initial_only',None,None,'none'),
    ('output',None,'1:00:00',None,'1:00:00'),
    ('output',None,'initial_only',None,'none')])
@pytest.mark.usefixtures('rrfs_clock')
def test_native_filename_default_precedence(tmp_path,direction,inputs,outputs,levels,expected):
    attrs={'name':'sample','type':direction,'filename_template':'x.$h.nc'}
    if inputs:attrs['input_interval']=inputs
    if outputs:attrs['output_interval']=outputs
    if levels:attrs['output_timelevels']=levels
    from xml.etree.ElementTree import Element,SubElement,ElementTree
    root=Element('streams');stream=SubElement(root,'stream',attrs);SubElement(stream,'var',{'name':'xtime'})
    path=tmp_path/'streams.atmosphere';ElementTree(root).write(path)
    ours=parse_streams(path)['sample']
    assert ours.options['filename_interval']==expected
    native=os.environ.get('RRFS_NATIVE_XML')
    if native:
        result=subprocess.run([native,'streams.atmosphere'],cwd=tmp_path,capture_output=True,text=True,check=True)
        row=next(line.split('\t') for line in result.stdout.splitlines() if line.startswith('create\tsample\t'))
        assert row[3]==expected


@pytest.mark.usefixtures('rrfs_bridge')
def test_gattr_no_append_keeps_existing_native_attributes(tmp_path):
    s=Stream('sample',{'type':'output','clobber_mode':'append','filename_template':'x.nc'},(('var','xtime',''),))
    mesh=SimpleNamespace(dimensions={},arrays={},attrs={})
    target=tmp_path/'x.nc'
    write_stream(target,s,mesh,{},START,initial_time=START,global_attrs={'retained':7})
    from dataclasses import replace
    s=replace(s,options=dict(s.options,gattr_update='no'))
    write_stream(target,s,mesh,{},START+timedelta(hours=1),initial_time=START,record_index=1,global_attrs={'retained':8,'new':4})
    with read_stream(target,scratch=tmp_path) as f:
        assert f.global_attrs['retained']=={'dtype':'i4','values':[7]}
        assert 'new' not in f.global_attrs
        assert f.variables['xtime'].text==('2023-03-31_12:00:00','2023-03-31_13:00:00')


@pytest.mark.parametrize('precision,global_dtype',[('native','f4'),('single','f4'),('double','f8')])
@pytest.mark.usefixtures('rrfs_bridge')
def test_native_real_attribute_precision(tmp_path,precision,global_dtype):
    stream=Stream('sample',{'type':'output','precision':precision,'filename_template':'x.nc'},
                  (('var','xtime',''),('var','latCell','')))
    mesh=SimpleNamespace(dimensions={'nCells':2},arrays={},attrs={})
    values=HistoryField(np.asarray([0.5,1.0],dtype=np.float32),('nCells',),
                        {'scale_factor':1.25,'_FillValue':-999.0})
    target=write_stream(tmp_path/'x.nc',stream,mesh,{'latCell':values},START,
                        initial_time=START,global_attrs={'real_scalar':np.float32(1.25),
                        'real_vector':np.asarray([1.25,2.5],dtype=np.float32),
                        'explicit_double':{'dtype':'f8','values':[1.25]}})
    with read_stream(target,scratch=tmp_path) as native:
        assert native.global_attrs['real_scalar']=={'dtype':global_dtype,'values':[1.25]}
        assert native.global_attrs['real_vector']=={'dtype':global_dtype,'values':[1.25,2.5]}
        assert native.global_attrs['explicit_double']=={'dtype':'f8','values':[1.25]}
        assert native.variables['latCell'].attrs['scale_factor']=={'dtype':'f4','values':[1.25]}
        assert native.variables['latCell'].attrs['_FillValue']=={'dtype':global_dtype,'values':[-999.0]}

@pytest.mark.parametrize('name',['ufscommunity.hrrrv5.gfs.winter','ufscommunity.hrrrv5.rap.winter','ufscommunity.hrrrv5.rap.summer'])
@pytest.mark.usefixtures('rrfs_clock')
def test_published_ci_decks_keep_first_duplicate_attribute(tmp_path,name):
    root=os.environ.get('RRFS_PUBLIC_CI_DECKS') or str(Path(os.environ.get('MPAS_LANES', '.')) / 'ref/rrfs/cases/gsl-ci-public-v1/decks')
    source=Path(root)/name
    if not source.is_dir():pytest.skip('published pinned fork CI deck absent')
    shutil.copytree(source,tmp_path,dirs_exist_ok=True)
    streams=parse_streams(tmp_path/'streams.atmosphere')
    native=os.environ.get('RRFS_NATIVE_XML')
    if not native:pytest.skip('original fork C parser artifact absent')
    run=subprocess.run([native,'streams.atmosphere'],cwd=tmp_path,capture_output=True,text=True,check=True)
    rows=[line.split('\t') for line in run.stdout.splitlines()]
    for name,stream in streams.items():
        row=next(row for row in rows if row[:2]==['create',name])
        assert row[2]==stream.options['filename_template']
        assert row[3]==stream.options['filename_interval']
        if not stream.immutable:
            assert [r[2] for r in rows if r[:2]==['field',name]]==[m[1] for m in stream.members]


@pytest.mark.parametrize('text',[
    '<streams><stream name="sample" type="output" filename_template="x.$h.nc" filename_template="ignored.nc" output_interval="1:00:00" output_interval="2:00:00"><var name="xtime"/></stream></streams>',
    '<?xml version="1.0"?><!-- a="first" a="second" --><streams>\n<stream name="sample" type="output" filename_template="x.$h.nc" output_interval="1:00:00" output_interval="2:00:00" ><var name="xtime" /></stream>\n</streams>'
])
@pytest.mark.usefixtures('rrfs_clock')
def test_native_duplicate_attributes_and_xml_boundaries(tmp_path,text):
    path=tmp_path/'streams.atmosphere';path.write_text(text)
    stream=parse_streams(path)['sample']
    assert stream.options['filename_template']=='x.$h.nc'
    assert stream.options['output_interval']=='1:00:00'
    native=os.environ.get('RRFS_NATIVE_XML')
    if native:
        run=subprocess.run([native,path.name],cwd=tmp_path,capture_output=True,text=True,check=True)
        row=next(line.split('\t') for line in run.stdout.splitlines() if line.startswith('create\tsample\t'))
        assert row[2]=='x.$h.nc' and row[3]=='1:00:00'


@pytest.mark.parametrize('precision,clobber,io_type,expected',[
    ('junk','junk','junk',(0,0,0)),
    ('single-double','append-truncate','netcdf4-pnetcdf',(4,1,0)),
    ('double','truncate','pnetcdf,cdf5',(8,2,1)),
    ('prefix-single-suffix','replace_files','netcdf4',(4,2,3)),
    ('native','overwrite','netcdf',(0,3,2))])
@pytest.mark.usefixtures('rrfs_clock')
def test_native_option_substrings_and_fallbacks(tmp_path,precision,clobber,io_type,expected):
    from xml.etree.ElementTree import Element,SubElement,ElementTree
    root=Element('streams');el=SubElement(root,'stream',dict(name='sample',type='prefix-output-suffix',
        filename_template='x.nc',output_interval='1:00:00',precision=precision,
        clobber_mode=clobber,io_type=io_type,unrecognized_attribute='ignored'))
    SubElement(el,'var',dict(name='xtime'));path=tmp_path/'streams.atmosphere';ElementTree(root).write(path)
    stream=parse_streams(path)['sample'];native=os.environ.get('RRFS_NATIVE_XML')
    if not native:pytest.skip('original fork C parser artifact absent')
    run=subprocess.run([native,path.name],cwd=tmp_path,capture_output=True,text=True,check=True)
    row=next(line.split('\t') for line in run.stdout.splitlines() if line.startswith('create\tsample\t'))
    assert (int(row[5]),int(row[6]),int(row[8]))==expected
    assert {'native':0,'single':4,'double':8}[stream.options['precision']]==expected[0]
    assert {'never_modify':0,'append':1,'truncate':2,'replace_files':2,'overwrite':3}[stream.options['clobber_mode']]==expected[1]
    assert {'pnetcdf':0,'pnetcdf,cdf5':1,'netcdf':2,'netcdf4':3}[stream.options['io_type']]==expected[2]


@pytest.mark.usefixtures('rrfs_clock')
def test_every_published_deck_against_original_c_parser(tmp_path):
    import hashlib
    index=Path(os.environ.get('RRFS_DECK_INDEX') or Path(os.environ.get('MPAS_LANES', '.')) / 'ref/rrfs/decks/INDEX.json')
    native=os.environ.get('RRFS_NATIVE_XML')
    if not index.is_file() or not native:pytest.skip('published matrix or native C parser absent')
    matrix=json.loads(index.read_text());witness=[]
    generated=Path(os.environ['RRFS_NATIVE_REGISTRY_GENERATED'])/'full/setup_immutable_streams.inc'
    immutable_order=re.findall(r"call MPAS_stream_mgr_create_stream\(manager, '([^']+)'",generated.read_text())
    for item in matrix['decks']:
        source=index.parent/item['group']/item['deck'];destination=tmp_path/item['group']/item['deck']
        for filename,sha in item['files'].items():
            original=source/filename;assert hashlib.sha256(original.read_bytes()).hexdigest()==sha
            target=destination/filename;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(original,target)
        streams=parse_streams(destination/'streams.atmosphere')
        run=subprocess.run([native,'streams.atmosphere'],cwd=destination,capture_output=True,text=True,check=True)
        rows=[line.split('\t') for line in run.stdout.splitlines()]
        expected_order=[name for name in immutable_order if name in streams and streams[name].immutable]
        expected_order.extend(row[1] for row in rows if row[0]=='create' and row[4]=='0')
        assert list(streams)==expected_order
        for name,stream in streams.items():
            row=next(row for row in rows if row[:2]==['create',name])
            assert row[2:4]==[stream.options['filename_template'],stream.options['filename_interval']]
            assert int(row[4])==stream.immutable
            assert int(row[5])=={'native':0,'single':4,'double':8}[stream.options.get('precision','native')]
            assert int(row[6])=={'never_modify':0,'append':1,'truncate':2,'replace_files':2,'overwrite':3}[stream.options.get('clobber_mode','never_modify')]
            raw=stream.options.get('gattr_update','yes');assert int(row[7])==(1 if 'yes' in raw or 'no' not in raw else 0)
            assert int(row[8])=={'pnetcdf':0,'pnetcdf,cdf5':1,'netcdf':2,'netcdf4':3}[stream.options.get('io_type','pnetcdf')]
            if not stream.immutable:
                assert [r[2] for r in rows if r[:2]==['field',name]]==[m[1] for m in stream.members]
            for direction in ('input','output'):
                if direction in stream.options['type'].split(';') and not (direction=='output' and 'output_timelevels' in stream.options):
                    assert ['alarm',name,direction,stream.interval(direction)] in rows
            assert bool([r for r in rows if r[:3]==['property',name,'output_done_marker']])==('yes' in stream.options.get('output_done_marker',''))
            stream.fields(_all_packages(),streams=streams)
        witness.append(dict(group=item['group'],deck=item['deck'],streams=len(streams),file_hashes=item['files'],native_callback_diff=0,manager_order_diff=0))
    receipt=os.environ.get('RRFS_DECK_REPLAY_RECEIPT')
    if receipt:Path(receipt).write_text(json.dumps(dict(schema='rrfs-native-deck-replay/v1',index_sha256=hashlib.sha256(index.read_bytes()).hexdigest(),native_process_count=1,whole_model_proof=False,decks=witness),indent=2)+'\n')
