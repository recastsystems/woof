import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

V=Path(__file__).parents[1]/'verification/wofs-ensemble'
spec=importlib.util.spec_from_file_location('wofs_init',V/'initialize.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def test_resume_requires_retained_output_bytes_to_match_receipt(tmp_path):
    path=tmp_path/'init.nc'; receipt=tmp_path/'receipt.json'
    assert not module.retained_output(path,receipt)
    path.write_bytes(b'finished')
    receipt.write_text(json.dumps(dict(output=dict(bytes=8,sha256=hashlib.sha256(b'finished').hexdigest()))))
    assert module.retained_output(path,receipt)
    receipt.write_text(json.dumps(dict(output_bytes=8,sha256=hashlib.sha256(b'finished').hexdigest())))
    assert module.retained_output(path,receipt)
    path.write_bytes(b'changed!')
    with pytest.raises(ValueError,match='changed'):
        module.retained_output(path,receipt)

def test_reusing_prepared_roster_applies_current_case_flags():
    roster_spec=importlib.util.spec_from_file_location('roster',V/'roster.py')
    roster=importlib.util.module_from_spec(roster_spec); roster_spec.loader.exec_module(roster)
    case=dict(members=['control','perturbed'],forecast_flags=['--dt','18.75'],
              member_physics={'perturbed':['--physics','config_microp_scheme=mp_thompson']})
    data=dict(members=[dict(id='control'),dict(id='perturbed')],forecast_flags=['--no-products'])
    result=roster.apply_roster(data,case)
    assert result['forecast_flags']==['--dt','18.75']
    assert result['members'][0]['forecast_flags']==[]
    assert result['members'][1]['forecast_flags']==case['member_physics']['perturbed']
    with pytest.raises(ValueError,match='roster differs'):
        roster.apply_roster(dict(members=[dict(id='control')]),case)


def test_new_queue_attempt_clears_stale_exit_before_forecast(tmp_path,monkeypatch):
    run_spec=importlib.util.spec_from_file_location('wofs_run',V/'run.py')
    run=importlib.util.module_from_spec(run_spec); run_spec.loader.exec_module(run)
    monkeypatch.setattr(run,'W',tmp_path)
    (tmp_path/'evidence').mkdir()
    (tmp_path/'evidence/inputs-ready.json').write_text('{}')
    exit_path=tmp_path/'evidence/pilot-exit.json'
    exit_path.write_text('{"returncode":2}')
    monkeypatch.setenv('CARD_QUEUE_CARDS','4')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','GPU-one-card')
    monkeypatch.setenv('E','/owned/engine')
    def forecast(command):
        current=json.loads(exit_path.read_text())
        assert current['status']=='running' and current['returncode'] is None
        assert command[command.index('--members')+1]=='1'
        return type('Result',(),{'returncode':0})()
    monkeypatch.setattr(run.subprocess,'run',forecast)
    assert run.main('pilot')==0
    assert json.loads(exit_path.read_text())['status']=='passed'
