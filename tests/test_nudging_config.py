"""Configuration and namelist semantics without importing a CUDA runtime."""
import pytest
from woof.spectral_nudging.config import NudgingConfig, from_mapping
from woof.spectral_nudging.namelist import translate
from woof.namelist_import import _Section


def test_selection_is_root_only_unless_explicit():
    options = from_mapping({'mode':'spectral'}, (1,2,3))
    assert options.for_domain(1).mode == 'spectral'
    assert options.for_domain(2) is None
    assert from_mapping({'enabled':False}) is None
    with pytest.raises(ValueError,match='does not exist'):
        from_mapping({'domain':{'4':{'mode':'grid'}}}, (1,2))


def test_namelist_keeps_unspecified_nests_off_and_field_controls():
    section = _Section('fdda', {
        'grid_fdda':[2], 'guv':[.0003], 'gt':[.0001], 'gph':[.0002], 'gq':[0.],
        'xwavenum':[3], 'ywavenum':[2], 'gfdda_interval_m':[360], 'gfdda_end_h':[120],
        'if_no_pbl_nudging_uv':[1], 'if_zfac_t':[1], 'k_zfac_t':[8],
        'fgdt':[5.], 'fgdtzero':[1], 'if_ramping':[1], 'dtramp_min':[-60.],
    }, 'test')
    row = translate(section,(1,2),lambda *args:None)
    assert set(row['domain']) == {'1'}
    config = from_mapping(row,(1,2)).for_domain(1)
    assert config.convention == 'wrf'
    assert config.coefficient('geopotential') == .0002
    assert config.k_zfac_t == 8 and config.if_no_pbl_nudging_t == 0
    assert config.cadence_steps(45.) == 7
    assert config.dtramp_min == -60.


def test_grid_nudging_and_independent_domain_selection():
    row = translate(_Section('fdda',{'grid_fdda':[0,1], 'guv':[0.,.0003],
        'gfdda_interval_m':[360,60], 'gfdda_end_h':[24,24]}, 'test'),(1,2),lambda *args:None)
    options = from_mapping(row,(1,2))
    assert options.for_domain(1) is None
    assert options.for_domain(2).mode == 'grid'
    assert options.for_domain(2).coefficient('geopotential') == 0.


@pytest.mark.parametrize('row', [{'guv':-1.}, {'gq':float('nan')}, {'fgdt':-1.},
    {'fgdtzero':2}, {'gfdda_interval_m':0.}, {'dk_zfac_t':0}])
def test_invalid_science_controls_refuse(row):
    with pytest.raises(ValueError):
        NudgingConfig(**row)


def test_off_does_not_read_fields_or_import_cuda():
    from woof.spectral_nudging import build_nudger
    assert build_nudger(NudgingConfig(), None) is None


@pytest.mark.parametrize('mode',[1,2])
def test_complete_namelist_import_retains_controls(tmp_path,mode):
    import tomllib
    from tests.test_namelist_import import _import_with
    from woof.experiment import build_experiment
    text,report = _import_with(tmp_path,extra_input=f'''&fdda
 grid_fdda = {mode}, 0,
 gfdda_interval_m = 360,
 gfdda_end_h = 24,
 guv = 0.0003,
 gt = 0.0003,
 gph = 0.0003,
 xwavenum = 1,
 ywavenum = 1,
/
''')
    exp = build_experiment(tomllib.loads(text),'nudging test')
    assert exp.nudging.for_domain(1).mode == ('grid' if mode == 1 else 'spectral')
    assert exp.nudging.for_domain(2) is None
    assert ('fdda','guv') in {(r.section,r.key) for r in report.translated}


def test_default_source_preparation_is_inert():
    from woof.spectral_nudging.cli import source_preparation
    from types import SimpleNamespace
    assert source_preparation(SimpleNamespace(),None,{}) is None


def test_source_preparation_keeps_suite_opt_in():
    from woof.source_cli import _parser, _era5_command
    base = _parser().parse_args(['--source','era5'])
    assert '--experiment-config-suite' not in _era5_command(base)
    selected = _parser().parse_args(['--source','era5','--experiment-config-suite'])
    assert '--experiment-config-suite' in _era5_command(selected)


def test_changed_request_gets_an_owned_download_directory(tmp_path):
    from types import SimpleNamespace
    from woof.spectral_nudging.cli import preparation_data_directory
    args=SimpleNamespace(data_dir=None)
    raw={'fetch':{'source':'era5','cycle':'2023-07-01T00','hours':24,
        'area':'30,-105,40,-95','out':'data','era5_provider':'arco','cadence':6}}
    first=preparation_data_directory(args,raw,tmp_path)
    first.mkdir(parents=True);(first/'prior-data').write_bytes(b'preserve me')
    raw['fetch']['hours']=48
    second=preparation_data_directory(args,raw,tmp_path)
    assert first!=second and first.parent==second.parent
    assert (first/'prior-data').read_bytes()==b'preserve me'
    args.data_dir=tmp_path/'explicit'
    assert preparation_data_directory(args,raw,tmp_path)==args.data_dir


def test_prepared_root_selection_preserves_explicit_nest_override(monkeypatch,tmp_path):
    from types import SimpleNamespace
    from woof import experiment,go_cli
    from woof.spectral_nudging.cli import at_flag
    options=from_mapping({'mode':'spectral','domain':{'2':{'mode':'grid'}}},(1,2))
    monkeypatch.setattr(experiment,'load_experiment',lambda _:SimpleNamespace(nudging=options,root=SimpleNamespace(grid_id=1)))
    monkeypatch.setattr(go_cli,'_prepared_launch',lambda _:True)
    config=tmp_path/'experiment.toml';payload={}
    assert at_flag(SimpleNamespace(nudging='spectral'),config,payload)==(config,payload)


def test_native_root_authority_keeps_only_its_domain_nudging(tmp_path):
    import copy
    from tests.test_hrrr_configured_physics import _native_tables
    from woof.experiment import build_experiment
    from woof.experiment_document import render_experiment_document
    from woof.hrrr_configuration import resolve_root_experiment
    from woof.hrrr_route_inputs import render_namelist_input
    from woof.hrrr_prepared_bundle import render_wps_namelist
    from tools.prepare_hrrr_wrf import _chains
    from types import SimpleNamespace
    raw,target=_native_tables();child=copy.deepcopy(raw['domain'][0])
    child.update(grid_id=2,parent_id=1,i_parent_start=12,j_parent_start=12,
        parent_grid_ratio=3,parent_time_step_ratio=3,nx=30,ny=30,
        dx=target.dx_m/3,dy=target.dx_m/3,specified=False,nested=True)
    child.pop('time_step',None)
    child.pop('time_step_fract_num',None)
    child.pop('time_step_fract_den',None)
    child.pop('clock_dt',None)
    child.pop('dt',None)
    raw['domain'].append(child)
    raw['nudging']={'mode':'spectral','domain':{'2':{'mode':'grid','convention':'wrf'}}}
    full=build_experiment(raw,'native nudging authority')
    config=tmp_path/'experiment.toml';config.write_text(render_experiment_document(raw))
    namelist=tmp_path/'namelist.input';namelist.write_text(render_namelist_input(full))
    wps=tmp_path/'namelist.wps';wps.write_text(render_wps_namelist(full))
    root,table=resolve_root_experiment(target=target,vertical=full.vertical,
        namelist_input=namelist,start_time=full.start_time,run_seconds=full.run_seconds,
        experiment_config=config,wps_namelist=wps)
    assert len(root.domains)==1
    assert root.nudging.for_domain(1).mode=='spectral'
    assert root.nudging.for_domain(2) is None
    assert not _chains(SimpleNamespace(experiment_config=config))
