"""Native auxiliary level streams: clocks, per-domain import and carry state."""
from dataclasses import replace
from pathlib import Path
import importlib.util
import numpy as np
from woof.level_output_config import DEFAULTS, drop_inert, validate
from woof.io import restart
from test_restart import _cfg, _shim_state, _fill_setup


def test_level_outputs_survive_checkpoint_between_calculation_alarms(monkeypatch, tmp_path):
    cfg = _cfg(p_lev_diags=1, z_lev_diags=1, press_levels=(92500.,85000.),
               z_levels=(-10.,1000.,3000.), auxhist22_interval_s=70.,
               auxhist23_interval_s=50.)
    source = _shim_state(cfg, monkeypatch)
    _fill_setup(source)
    expected = {}
    for name in ('level_pl','level_zl'):
        a = source.existing_scratch(name)
        a[...] = np.arange(a.size,dtype='f4').reshape(a.shape)+np.float32(.125)
        expected[name] = a.tobytes()
    path = restart.write_restart(tmp_path/'levels.npz',source,cfg)
    dest = _shim_state(cfg,monkeypatch)
    _fill_setup(dest)
    restart.restore_restart(path,dest,cfg)
    assert all(dest.existing_scratch(k).tobytes()==v for k,v in expected.items())
    with np.load(path,allow_pickle=False) as archive:
        assert all('scratch/'+k in archive.files for k in expected)
        assert 'scratch/level_pressure' not in archive.files
    np.testing.assert_array_equal(dest.existing_scratch('level_press_targets'),[92500.,85000.])


def test_off_config_does_not_change_checkpoint_identity_document():
    original = {'nx':6,'other':'kept'}
    new = {**original,**DEFAULTS}
    drop_inert(new)
    assert new == original
    active = {**DEFAULTS,'p_lev_diags':1}
    drop_inert(active)
    assert 'press_levels' in active


def test_native_nonmonotone_targets_and_non_aligned_alarms_are_admitted():
    cfg=_cfg(p_lev_diags=1, press_levels=(85000.,92500.),
             auxhist23_interval_s=61.,p_lev_interval_s=121.,auxhist23_begin_s=7.)
    validate(cfg)


def test_stock_import_preserves_native_separate_calculation_clock():
    from woof.fortran_namelist import parse_namelist_text
    from woof.namelist_import import import_parsed_namelists
    from woof.experiment import load_experiment
    import tomllib
    path=Path(__file__).parents[1]/'tools/diag_wrf471_oracle/census.py'
    spec=importlib.util.spec_from_file_location('level_census',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    fixture,_=module.fixture()
    wps=parse_namelist_text(fixture['WPS_TEXT'])
    inp=parse_namelist_text(fixture['INPUT_TEXT'])
    inp['diags']={'p_lev_diags':[1],'num_press_levels':[2], 'press_levels':[92500.,85000.]}
    inp['time_control'].update(auxhist23_interval=[1,1],auxhist23_interval_m=[2,2],
                               auxhist23_begin=[1,1],auxhist23_begin_s=[7,7],
                               auxhist23_interval_y=[1,1],io_form_auxhist23=[2])
    text,_=import_parsed_namelists(wps,inp,wps_path='namelist.wps',input_path='namelist.input',
                                  name='levels',source_text=None)
    values=tomllib.loads(text)['shared']
    assert values['auxhist23_interval_s']==120.
    assert values['p_lev_interval_s']==180.
    assert values['auxhist23_begin_s']==67.
    assert values['press_levels']==[92500.,85000.]


def test_level_writer_retains_native_dimension_and_vector_attributes(tmp_path):
    from woof.io.level_output import LevelTape
    from woof import netcdf_bridge
    path=tmp_path/'aux23.nc'
    fields={'P_PL':np.array([92500.,85000.],dtype='f4'),
            'T_PL':np.arange(48,dtype='f4').reshape(2,4,6)}
    writer=LevelTape(path,nx=6,ny=4,nz=5,dx=3000.,dy=3000.,
                     global_attrs={'START_DATE':'2024-05-25_18:00:00','DT':12.},
                     field_schema=fields)
    writer.write_frame('2024-05-25_18:00:00',fields)
    writer.write_frame('2024-05-25_19:00:00',fields)
    writer.close()
    with netcdf_bridge.open_dataset(path) as ds:
        assert ds.variables['T_PL'].dimensions==('Time','num_press_levels_stag','south_north','west_east')
        assert ds.variables['P_PL'].dimensions==('Time','num_press_levels_stag')
        assert ds.variables['T_PL'].stagger=='Z'
        np.testing.assert_array_equal(ds.variables['T_PL'][1],fields['T_PL'])


def test_native_warm_start_rebuilds_level_products_before_output():
    from woof.ingest.wrfrst import scratch_restore_plan
    for name in ('level_pl', 'level_zl'):
        plan = scratch_restore_plan(name, 0)
        assert plan[0:2] == ('cold', 0.0)
        assert 'before any output or step' in plan[2]
