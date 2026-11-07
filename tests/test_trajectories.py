"""Native particle input, optional state and checkpoint integrity contracts."""
from dataclasses import asdict
import hashlib
import numpy as np
import pytest
from woof.core.trajectory_contract import (COORDINATES, active, enabled, drop_inactive,
    parse_points, validate, validate_checkpoint, scratch_shapes)
from woof.io import restart
from test_restart import _cfg, _shim_state, _fill_setup

POINTS="""&traj_default
 traj_def%dyn_name='p','T','u', traj_def%start_time='2019-09-04_12:00:00'
/
&traj_spec
 traj_type(1)%lon=-78., traj_type(1)%lat=30., traj_type(1)%lev=1000.,
 traj_type(1)%is_stationary=.true.,
 traj_type(2)%lon=-79., traj_type(2)%lat=31., traj_type(2)%lev=2000.,
 traj_type(2)%dyn_var='v', '', 'w',
/
"""

def test_off_identity_and_inventory(monkeypatch):
 cfg=_cfg();state=_shim_state(cfg,monkeypatch);identity=asdict(cfg)
 drop_inactive(identity)
 assert not any(k in identity for k in ('traj_opt','dm_has_traj','num_traj','trajectory_input'))
 assert not hasattr(state,'traj_coordinates') and not scratch_shapes(cfg)

@pytest.mark.parametrize('tracking',[False,True])
def test_global_registry_coordinates_and_restart(monkeypatch,tmp_path,tracking):
 from woof.core.device_inventory import state_array_shapes
 from woof.io.history_layout import live_state_history_fields,produced_history_shapes
 cfg=_cfg(traj_opt=1,dm_has_traj=tracking,num_traj=7,trajectory_input=POINTS)
 state=_shim_state(cfg,monkeypatch);_fill_setup(state)
 assert enabled(cfg) and active(cfg)==tracking
 assert state_array_shapes(cfg)['traj_coordinates']==(5,7)
 assert not np.any(state.traj_coordinates)
 assert int(state.traj_status[0])==(-1 if tracking else 0)
 for i,name in enumerate(COORDINATES):
  state.traj_coordinates[i]=np.arange(7,dtype=np.float32)+i*.125
  np.testing.assert_array_equal(live_state_history_fields(state)[name.upper()],state.traj_coordinates[i])
  assert produced_history_shapes(cfg)[name.upper()]==(7,)
 saved=restart.write_restart(tmp_path/'coordinates.npz',state,cfg)
 target=_shim_state(cfg,monkeypatch);_fill_setup(target)
 restart.restore_restart(saved,target,cfg)
 np.testing.assert_array_equal(target.traj_coordinates.view('u4'),state.traj_coordinates.view('u4'))


def test_native_point_defaults_gaps_and_quoted_components():
 points=parse_points(POINTS)
 assert len(points)==2 and points[0]['is_stationary']
 assert points[0]['dyn']==('p','T','u') and points[1]['dyn']==('v','w')
 assert points[1]['z_is_agl'] and points[1]['start_time']=='2019-09-04_12:00:00'
 text=POINTS.replace("traj_type(2)%lon=-79.","traj_type(3)%lon=-79.")
 assert len(parse_points(text))==1
 text=POINTS.replace("traj_def%dyn_name='p','T','u'","traj_def%dyn_name='traj_type(2)%x ! quoted'")
 assert parse_points(text)[0]['dyn']==('traj_type(2)%x ! quoted',)
 assert parse_points('')==()

@pytest.mark.parametrize('changes,match',[(dict(traj_opt=2),'traj_opt'),(dict(num_traj=1001),'num_traj'),(dict(dm_has_traj=1),'bool'),(dict(num_traj=1),'point count')])
def test_native_capacity_and_types(changes,match):
 values=dict(traj_opt=1,dm_has_traj=True,trajectory_input=POINTS);values.update(changes)
 with pytest.raises((ValueError,TypeError),match=match):validate(_cfg(**values))


def _prefix():
 prefix=np.frombuffer(b'CDF\x02'+b'\0'*40,dtype=np.uint8).copy()
 points=list(parse_points(POINTS))
 row=dict(sha256=hashlib.sha256(prefix).hexdigest(),records=0,points=points)
 stored={'trajectory/output_prefix':prefix,'state/traj_status':np.asarray([2,0],np.float32)}
 return row,stored


def test_checkpoint_prefix_clock_roster_and_hash_are_validated():
 cfg=_cfg(traj_opt=1,dm_has_traj=True,trajectory_input=POINTS)
 row,stored=_prefix();validate_checkpoint({'trajectory':row},stored,cfg)
 for defect in ('bytes','clock','roster','orphan'):
  row,stored=_prefix()
  if defect=='bytes':stored['trajectory/output_prefix'][-1]=1
  if defect=='clock':stored['state/traj_status'][1]=1
  if defect=='roster':row['points'][0]['lat']=80.
  if defect=='orphan':row=None
  with pytest.raises(restart.RestartMismatchError):validate_checkpoint({'trajectory':row},stored,cfg)


def test_stock_namelist_embeds_points_and_per_domain_tracking(tmp_path):
 from test_namelist_import import WPS_TEXT,INPUT_TEXT
 from woof.namelist_import import import_namelists
 from woof.experiment import load_experiment
 wps=tmp_path/'namelist.wps';inp=tmp_path/'namelist.input'
 wps.write_text(WPS_TEXT)
 inp.write_text(INPUT_TEXT.replace('&physics','&physics\n traj_opt=1, dm_has_traj=.true.,.false.,').replace('&domains','&domains\n num_traj=7,'))
 (tmp_path/'wrfinput_traj_d01').write_text(POINTS)
 text,_=import_namelists(wps,inp,name='points');path=tmp_path/'points.toml';path.write_text(text)
 experiment=load_experiment(path)
 assert experiment.root.run.traj_opt==1 and experiment.root.run.num_traj==7
 assert experiment.root.run.dm_has_traj
 assert parse_points(experiment.root.run.trajectory_input)==parse_points(POINTS)
 assert experiment.domains[1].run.traj_opt==1 and not experiment.domains[1].run.dm_has_traj
