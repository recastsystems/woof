"""Analysis cadence must remain independent of lateral-boundary cadence."""
from datetime import datetime, timedelta
from types import SimpleNamespace
import json
import numpy as np
import pytest
from woof.spectral_nudging.config import NudgingConfig, NudgingOptions
from woof.spectral_nudging.prepared import AnalysisWriter, PreparedNudging, FIELDS


def experiment(**controls):
    run = SimpleNamespace(nz=2, ny=3, nx=4, dt=60., use_adaptive_time_step=False)
    root = SimpleNamespace(grid_id=1, run=run)
    return SimpleNamespace(root=root, start_time=datetime(2023, 7, 1), run_seconds=21600.,
        nudging=NudgingOptions(((1, NudgingConfig(enabled=True, **controls)),)))


def test_analysis_writer_selects_requested_cadence_and_writes_native_arrays(tmp_path):
    exp = experiment(gfdda_interval_m=180.)
    times = [exp.start_time+timedelta(hours=h) for h in range(7)]
    writer = AnalysisWriter(tmp_path/'prepared', exp, times, 'source-digest')
    assert writer.header['offsets'] == [0.,10800.,21600.]
    assert writer.source_indices == {0:0,3:1,6:2}
    state = SimpleNamespace(**{k:np.full((2,3,4),i,dtype='f') for i,k in enumerate(FIELDS) if k!='pressure'})
    initial = SimpleNamespace(state=state,total_pressure=np.full((2,3,4),80000.,dtype='f'))
    writer.write(1,initial)
    assert not (writer.root/'00001.json').exists()
    writer.write(3,initial)
    record = json.loads((writer.root/'00001.json').read_text())
    assert record['offset_seconds'] == 10800.
    for name,row in record['files'].items():
        value = np.load(writer.root/row['file'])
        expected = initial.total_pressure if name=='pressure' else getattr(state,name)
        np.testing.assert_array_equal(value,expected)


@pytest.mark.parametrize('settings,hours,message', [
    ({},[0,1,2],'COVERAGE'),
    ({'gfdda_interval_m':90.},[0,1,2,3,4,5,6],'CADENCE'),
    ({'gfdda_end_h':2.5},[0,1,2,3,4,5,6],'END_ANALYSIS'),
])
def test_incomplete_or_ambiguous_analysis_clock_refuses(tmp_path,settings,hours,message):
    exp=experiment(**settings)
    with pytest.raises(ValueError,match=message):
        AnalysisWriter(tmp_path/'p',exp,[exp.start_time+timedelta(hours=h) for h in hours],'source')


def test_refresh_uses_analysis_slots_not_lateral_slots():
    nudger=object.__new__(PreparedNudging)
    nudger.controls=NudgingConfig(gfdda_end_h=6.)
    nudger.offsets=[0.,10800.,21600.,32400.]
    for elapsed,wanted in [(0,(0,10800)),(3600,(0,10800)),(10800,(10800,21600)),(21600,(10800,21600)),(30000,(10800,21600))]:
        slot=nudger.analysis_interval(elapsed)
        assert (slot.start_seconds,slot.end_seconds)==wanted


def test_nests_are_selected_explicitly_and_moving_analysis_grids_refuse():
    from woof.spectral_nudging.config import from_mapping
    from woof.spectral_nudging.prepared import validate_fixed_domains
    start=datetime(2023,7,1)
    root=SimpleNamespace(grid_id=1,parent_id=0,start_time=start)
    child=SimpleNamespace(grid_id=2,parent_id=1,start_time=start)
    exp=SimpleNamespace(start_time=start,domains=(root,child),relocation=SimpleNamespace(enabled=True),nudging=from_mapping({'mode':'spectral'},(1,2)))
    # Unselected moving nests do not prevent anchoring the fixed parent.
    validate_fixed_domains(exp)
    exp.nudging=from_mapping({'domain':{'2':{'mode':'grid'}}},(1,2))
    with pytest.raises(ValueError,match='FIXED_ANALYSIS_GRID'):
        validate_fixed_domains(exp)
    exp.relocation.enabled=False
    validate_fixed_domains(exp)
    child.start_time=start+timedelta(hours=1)
    with pytest.raises(ValueError,match='FIXED_ANALYSIS_GRID'):
        validate_fixed_domains(exp)


def test_native_request_binds_source_manifest_in_sealed_and_chained_modes(tmp_path):
    from woof.spectral_nudging.prepared import analysis_writer_from_request
    exp=experiment()
    times=[exp.start_time,exp.start_time+timedelta(hours=6)]
    args=SimpleNamespace(prepare_only=True,as_posted=None,
        publish_experiment_config=tmp_path/'sealed/experiment.toml',
        source_manifest_sha256='a'*64,manifest_sha256='b'*64)
    sealed=analysis_writer_from_request(args,exp,None,times)
    assert sealed.root==tmp_path/'sealed-nudging/d01'
    assert sealed.header['source_sha256']=='a'*64
    chained=analysis_writer_from_request(args,exp,{'output_root':str(tmp_path/'chain')},times)
    assert chained.root==tmp_path/'chain-nudging/d01'
    assert chained.header['source_sha256']=='a'*64
    args.as_posted=tmp_path/'unsealed-plan.json'
    with pytest.raises(ValueError,match='UNSEALED_ANALYSIS_IDENTITY'):
        analysis_writer_from_request(args,exp,None,times)
