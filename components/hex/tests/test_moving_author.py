"""Actual console case authoring and the historical M2 retry contract."""
import json
from pathlib import Path
import pytest
from woof.hex.cli import main
from woof.hex.moving_author import author_case
from woof.hex.moving_case import MovingCaseError, load_case, resolve_vertical_spec
from woof.hex import moving_run


def test_moving_lts_observes_the_step_mean_mass_flux(monkeypatch):
    """The real run's mass-flux tap must replace the resolver's live entry."""
    from woof.hex import cuda_dynamics_v841 as dynamics
    from woof.hex.cuda_fullstep_lts_scalars import tap_kernel
    launches, observed = [], []

    class Cache:
        def raw_kernel(self, name, source, **kwargs):
            return lambda grid, block, args: launches.append(args)

    cache = Cache()
    monkeypatch.setattr(dynamics, '_KERNELS', {})
    tap_kernel(dynamics, cache, 'split_flux_finish_v841_f32',
               after=lambda args: observed.append(args))
    args = (55, 100, 1.0, 'accumulator', 'result')
    dynamics._kernel('split_flux_finish_v841_f32', cache)((1,), (128,), args)
    assert launches == observed == [args]


def track(tmp_path, rows=None):
    p = tmp_path/'track.json'
    p.write_text(json.dumps(dict(sid='storm-id', rows=rows or [
        dict(time='2022-09-27T18:00:00Z', lat=20, lon=179),
        dict(time='2022-09-28T00:00:00Z', lat=20, lon=-179),
        dict(time='2022-09-28T06:00:00Z', lat=70, lon=30)])))
    return p


def test_console_authors_usable_case_without_future_track(tmp_path):
    p=track(tmp_path); out=tmp_path/'case.json'
    assert main(['moving','case','--best-track',str(p),'--cycle','2022-09-28T00',
                 '--out',str(out),'--hours','6']) == 0
    case=load_case(out)
    assert case['storm']['initial_center_deg']==[20,-179]
    motion=case['storm']['initial_motion']
    assert 80 < motion['heading_deg'] < 100
    assert 30 < motion['speed_kmh'] < 40
    assert case['storm']['ibtracs_sid']=='storm-id'
    assert case['authoring']['future_positions_used'] is False
    assert len(case['authoring']['motion_times_utc'])==2
    assert resolve_vertical_spec(case).is_file()
    # Moving launch uses the authored step and the ordinary default physics route.
    cmd=moving_run.segment_command(case,dict(mesh='m',grid='g',static='s',init='i',dt_seconds=case['dt_seconds']),
                                  tmp_path/'seg',6,30,init_source='gfs')
    assert cmd[cmd.index('--fullstep-lts-max-rate')+1]=='8'


def test_best_track_analysis_row_needs_no_future_or_prior_row(tmp_path):
    p=tmp_path/'row.json';p.write_text(json.dumps(dict(time='2022-09-28T00Z',lat=22,lon=278)))
    case=author_case(tmp_path/'c.json',track=p,cycle='2022-09-28T00Z',best_track=True)
    assert case['storm']['initial_center_deg']==[22,-82]
    assert case['storm']['initial_motion']['speed_kmh']==0


def test_real_tc_tracks_schema_selection(tmp_path):
    p=tmp_path/'tracks.json'
    p.write_text(json.dumps(dict(tracks=[dict(track_id=k,fixes=[
        dict(valid='2022-09-28T00Z',lat=10+i,lon=30)]) for i,k in enumerate(['a','b'])])))
    with pytest.raises(MovingCaseError,match='exactly one track'):
        author_case(tmp_path/'bad.json',track=p,cycle='2022-09-28T00')
    case=author_case(tmp_path/'ok.json',track=p,cycle='2022-09-28T00',storm='b')
    assert case['storm']['initial_center_deg']==[11,30]
    assert 'ibtracs_sid' not in case['storm']


@pytest.mark.parametrize('change,match',[
    ({'dt_seconds':18},'radiation interval'),
    ({'fine_km':1,'background_km':48},'power of two'),
    ({'fine_radius_km':2},'fine radius'),
    ({'hours':float('nan')},'finite number'),
    ({'cycle':'2022-09-28T03'},'exactly at'),
    ({'cycle':'2022-09-28T00:30'},'whole UTC'),
])
def test_bad_authoring_inputs_fail_before_writing(tmp_path,change,match):
    kwargs=dict(track=track(tmp_path),cycle='2022-09-28T00');kwargs.update(change)
    out=tmp_path/'bad.json'
    with pytest.raises(MovingCaseError,match=match):author_case(out,**kwargs)
    assert not out.exists()


def test_template_event_seed_is_replaced_and_case_is_not_overwritten(tmp_path):
    p=track(tmp_path)
    template=Path(__file__).parents[1]/'verification/moving48/cases/tc-wp-2025110812.json'
    case=author_case(tmp_path/'case.json',track=p,cycle='2022-09-28T00',template=template)
    assert case['storm']['initial_center_deg']==[20,-179]
    assert 'gfs_cycle' not in case
    assert 'ibtracs_sid' not in case['storm']
    assert case['mesh']['fine_km']==3
    with pytest.raises(FileExistsError):
        author_case(tmp_path/'case.json',track=p,cycle='2022-09-28T00')


def test_conflicting_track_positions_are_rejected(tmp_path):
    p=track(tmp_path,[dict(time='2022-09-28T00Z',lat=10,lon=20),
                      dict(time='2022-09-28T00Z',lat=11,lon=20)])
    with pytest.raises(MovingCaseError,match='conflicting positions'):
        author_case(tmp_path/'c.json',track=p,cycle='2022-09-28T00')


def broken_executor(tmp_path,end):
    """Model the recorded M2 child: rc0, no carrier, no moving progress."""
    from types import SimpleNamespace
    c=object.__new__(moving_run.Controller)
    c.a=SimpleNamespace(resume=False,max_retries=2,member_fallback=False)
    c.duration=21600; c.interior_km=100
    # 9e662b5b: run() asks the track session for a startup member; a session with no
    # move policy prepares none, which keeps this model to the executor contract.
    c.session=SimpleNamespace(policy=None)
    c.state=dict(t=0,boundary=None,segments=[],frames=[],events=[])
    c.save=lambda:None;c.drop_prelaunch=lambda reason:None
    c.event=lambda kind,**data:c.state['events'].append(dict(kind=kind,**data))
    c.segment_receipt=lambda seg:{}
    c.monitor=lambda *args:0
    c.newest_seal=lambda *args:None
    c.failure_kind=lambda *args:'other'
    calls=[]
    def launch(kind):
        seg=tmp_path/f'segment-{len(calls):03d}';seg.mkdir();control=seg/'control';control.mkdir()
        if end is not None:
            (seg/'boundary.pkl').write_bytes(b'carrier')
            (seg/'boundary.pkl.json').write_text(json.dumps(dict(time_seconds=end)))
        record=dict(index=len(calls),path=str(seg),begin_seconds=0,launched_wall=0,kind='start',center=[0,0],member='start')
        calls.append(record);c.state['segments'].append(record)
        return None,seg,control,record
    c.launch=launch
    return c,calls


def test_m2_success_without_carrier_is_never_retried(tmp_path):
    c,calls=broken_executor(tmp_path,None)
    with pytest.raises(SystemExit,match='returned 0 without moving carrier'):
        c.run()
    assert len(calls)==1
    assert c.state['events'][-1]['kind']=='segment-contract-failed'


@pytest.mark.parametrize('end',[0,-1,21601,float('nan')])
def test_invalid_carrier_cannot_stall_or_complete_the_controller(tmp_path,end):
    c,calls=broken_executor(tmp_path,end)
    with pytest.raises(SystemExit,match='carrier time'):
        c.run()
    assert len(calls)==1
    assert c.state['t']==0


@pytest.mark.parametrize('flags',[['--fullstep-lts-max-rate','1'],['--fullstep-lts-max-rate=1']])
def test_m2_override_refuses_before_preparation(tmp_path,flags):
    from types import SimpleNamespace
    path=tmp_path/'case.json'
    author_case(path,track=track(tmp_path),cycle='2022-09-28T00',physics_args=flags)
    args=SimpleNamespace(case=str(path),casedir=str(tmp_path/'prepared'),out=str(tmp_path/'run'),members=None)
    with pytest.raises(SystemExit,match='without moving progress or carrier hooks'):
        moving_run.Controller(args)
    assert not (tmp_path/'run').exists()


def test_product_metadata_spans_moves_recovery_and_gaps(tmp_path):
    from woof.hex.moving_products import track_document
    case=tmp_path/'case.json'
    author_case(case,track=track(tmp_path),cycle='2022-09-28T00')
    def frame(t,ok=True,member='start',segment=0):
        return dict(model_seconds=t,member=member,segment=segment,offset_km=4,inside_interior=True,
                    fix=dict(ok=ok,lat=20,lon=-179,mslp_min_hpa=950,vmax10_ms=42))
    state=dict(case=str(case),t=7200,fixes_source='run-time tracker on model state',
               frames=[frame(1800),frame(3600,False),frame(5400),frame(5400,True,'move001',1)])
    (tmp_path/'controller.json').write_text(json.dumps(state))
    best=tmp_path/'best.json';best.write_text(json.dumps(dict(rows=[
        dict(time='2022-09-28T00Z',lat=20,lon=-179),
        dict(time='2022-09-28T06Z',lat=25,lon=-179)])))
    doc,rows=track_document(tmp_path,best)
    assert len(doc['tracks'][0]['fixes'])==3
    assert doc['tracks'][0]['fixes'][1] is None
    assert doc['tracks'][0]['fixes'][0]['mslp_pa']==95000
    assert rows[-1]['member']=='move001' and rows[-1]['segment']==1
    assert len(doc['best_track']['fixes'])==1


def test_tracks_accept_null_gaps_and_reject_malformed_records(tmp_path):
    p=tmp_path/'tracks.json'
    p.write_text(json.dumps(dict(tracks=[dict(track_id='a',fixes=[None,
        dict(valid='2022-09-28T00Z',lat=20,lon=-80)])])))
    author_case(tmp_path/'case.json',track=p,cycle='2022-09-28T00')
    p.write_text(json.dumps(dict(tracks=[None])))
    with pytest.raises(MovingCaseError,match='track objects'):
        author_case(tmp_path/'bad.json',track=p,cycle='2022-09-28T00')


def test_unknown_init_source_is_rejected_before_authoring(tmp_path):
    from woof.hex.init_sources import InitSourceRefusal
    with pytest.raises(InitSourceRefusal,match='no init source is named'):
        author_case(tmp_path/'bad.json',track=track(tmp_path),cycle='2022-09-28T00',source='missing-source')


def test_one_km_request_derives_valid_background_spacing(tmp_path):
    case=author_case(tmp_path/'case.json',track=track(tmp_path),cycle='2022-09-28T00',fine_km=1)
    assert case['mesh']['background_km']==64
    assert case['dt_seconds']==6.25
