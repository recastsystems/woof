"""Stock global controls select their actual spherical operators and IO."""
from dataclasses import replace
from pathlib import Path

import pytest

from woof.core.global_state import validate_config
from woof.namelist_import import import_global_ideal_namelist


STOCK = Path(__file__).parent / 'data' / 'global_heldsuarez.input'


def test_unedited_stock_global_namelist():
    cfg,receipt=import_global_ideal_namelist(STOCK)
    validate_config(cfg)
    assert (cfg.nx,cfg.ny,cfg.nz)==(64,36,40)
    assert cfg.polar and cfg.ra_lw_physics==31 and cfg.ra_sw_physics==0
    assert cfg.run_seconds==1000*86400
    assert receipt['frames_per_outfile']==400
    assert receipt['document']['domains']['parent_grid_ratio']==[1,5,5]


@pytest.mark.parametrize('key,value,reason',[
    ('khdif','10','khdif'),('non_hydrostatic','.true.,.true.,.true.','non_hydrostatic'),
    ('frames_per_outfile','0','frames_per_outfile'),('max_dom','2','max_dom'),
    ('ra_sw_physics','1','ra_sw_physics'),('time_step_sound','3','acoustic')])
def test_global_import_never_drops_an_active_operator(tmp_path,key,value,reason):
    import re
    text,count=re.subn(r'(^\s*'+key+r'\s*=)[^\n]*',r'\g<1> '+value+',',STOCK.read_text(),flags=re.M)
    assert count==1
    source=tmp_path/'namelist.input';source.write_text(text)
    with pytest.raises(ValueError,match=reason):import_global_ideal_namelist(source)


@pytest.mark.parametrize('setting',[
    dict(kvdif=1.),dict(moist=True),dict(nested=True),dict(terrain_opt=1),
    dict(diff_6th_opt=1),dict(fft_filter_lat=91.)])
def test_direct_global_state_cannot_bypass_admission(setting):
    cfg,_=import_global_ideal_namelist(STOCK)
    with pytest.raises(ValueError,match='global'):validate_config(replace(cfg,**setting))


def test_stock_global_public_dry_run_and_shared_ensemble_admission(tmp_path,capsys):
    from test_go_wrfinput import arguments
    from woof.go_wrfinput import launch_if_wrf_directory
    (tmp_path/'namelist.input').write_bytes(STOCK.read_bytes())
    assert launch_if_wrf_directory(arguments(tmp_path,'--dry-run'))==0
    assert 'validated native global namelist' in capsys.readouterr().out
    # A single trajectory cannot be duplicated into a purported ensemble,
    # including through an inherited prepared-run request.
    with pytest.raises(ValueError,match='(one|same|trajectory|member)'):
        launch_if_wrf_directory(arguments(tmp_path,'--dry-run','--members','2'))


@pytest.mark.parametrize('stamp', ['on', 'off'])
def test_global_public_go_consumes_run_stamp(tmp_path, monkeypatch, stamp):
    from test_go_wrfinput import arguments
    from woof import global_forecast, run_stamp
    from woof.go_wrfinput import launch_if_wrf_directory
    (tmp_path/'namelist.input').write_bytes(STOCK.read_bytes())
    output = tmp_path/'output'
    seen = []
    monkeypatch.setattr(global_forecast, 'run',
                        lambda source, destination, **kw: seen.append(destination) or 17)
    args = arguments(tmp_path, '--outdir', str(output), '--run-stamp', stamp,
                     '--no-verify-visuals', '--products', 'none')
    args._supplied_go_options = {'run_stamp'}
    assert launch_if_wrf_directory(args) == 17
    if stamp == 'off':
        assert seen == [output]
    else:
        assert seen[0].parent == output
        parsed = run_stamp.parse_stamp(seen[0].name)
        assert parsed is not None and parsed['init'].year == 1
