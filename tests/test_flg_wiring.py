"""FLG selector admission and the stock-input go door."""
from dataclasses import replace
from types import SimpleNamespace
from datetime import datetime
import argparse
import numpy as np
import pytest
from woof.config import RunConfig, validate_run_config, ConfigurationRefused


def config(lw=7,sw=7,nz=32):
    return RunConfig(nx=12,ny=12,nz=nz,dx=3000.,dy=3000.,ztop=20000.,
                     dt=10.,run_seconds=60.,ra_lw_physics=lw,ra_sw_physics=sw)


@pytest.mark.parametrize('pair',[(7,7),(7,1),(7,4),(7,5),(1,7),(4,7),(5,7)])
def test_native_flg_selections_are_admitted(pair):
    validate_run_config(config(*pair))


@pytest.mark.parametrize('pair',[(7,0),(0,7)])
def test_native_fatal_half_pair_is_refused(pair):
    with pytest.raises(ConfigurationRefused,match='WRF'):
        validate_run_config(config(*pair))


@pytest.mark.parametrize('nz',[3,101])
def test_native_cloud_group_and_vertical_scratch_bounds(nz):
    with pytest.raises(ConfigurationRefused,match='FLG|nz must'):
        validate_run_config(config(nz=nz))


def test_passive_sw_case_does_not_invoke_flg_vertical_limit():
    from woof.physics_compat import validate_resolved_physics_vertical_levels
    cfg=config(1,7,101)
    receipt=validate_resolved_physics_vertical_levels(cfg,p_top=5000.)
    assert not any('Fu-Liou' in row['component'] for row in receipt['checks'])


def test_go_stock_directory_delegates_without_toml_loading(tmp_path,monkeypatch):
    from woof.go_cli import go_main
    from woof import wrfinput_go
    (tmp_path/'namelist.input').write_text('&physics\nra_lw_physics=7,ra_sw_physics=7\n/\n')
    (tmp_path/'wrfinput_d01').touch()
    marker=object();seen=[]
    monkeypatch.setattr(wrfinput_go,'launch',lambda args, **kw:seen.append(args.config) or marker)
    assert go_main(SimpleNamespace(config=tmp_path)) is marker
    assert seen==[tmp_path]


def test_wrf_directory_cannot_silently_drop_a_go_control(tmp_path):
    from woof.go_cli import _add_go_arguments,GoRefusal
    from woof.go_wrfinput import launch
    parser=argparse.ArgumentParser();_add_go_arguments(parser)
    args=parser.parse_args([str(tmp_path),'--prepare-only'])
    with pytest.raises(GoRefusal,match='prepare-only'):launch(args)


def test_classic_default_kernel_sources_and_options_are_unchanged():
    # Compare the candidate loader itself, with exactly the same source assets.
    # Protect legacy modules, including composed strict headers. Other admitted
    # ports have their own source/options checks and intentionally new loaders.
    import subprocess
    from woof.core import kernels
    source=subprocess.check_output(['git','show','4123f1916e3:woof/core/kernels/__init__.py'],text=True)
    old={'__name__':'woof.core.kernels','__package__':'woof.core.kernels','__file__':kernels.__file__}
    exec(compile(source,'candidate-kernel-loader','exec'),old)
    checked=0
    port_modules={'km5','km5_geometry','kfcup','temf','temf_sfclay',
                  'gsfcgce','gsfcgce_refl','gbmpbl','eeps','mrf',
                  'sbu_ylin','sbu_ylin_edge','wdm5','wdm7','wdm7_refl','sstskin',
                  # The lightning lane changed its fast FMA policy.
                  'lightning'}
    candidate_paths = subprocess.check_output(
        ['git', 'ls-tree', '--name-only', '4123f1916e3:woof/core/kernels'], text=True).splitlines()
    for name in candidate_paths:
        if not name.endswith('.cu'): continue
        path = kernels._KDIR / name
        if path.stem.startswith('flg') or path.stem in port_modules:continue
        assert kernels.module_source(path.stem)==old['module_source'](path.stem),path.name
        assert kernels.module_options(path.stem)==old['module_options'](path.stem),path.name
        checked+=1
    assert checked>200


@pytest.mark.parametrize('pair,contains,excludes',[
    ((7,7),{'flg_column','flg_driver'},set()),
    ((7,5),{'flg_column','goddard'},set()),
    ((5,7),{'goddard'},{'flg_column'}),
    ((1,7),set(),{'flg_column','flg_driver'}),
])
def test_native_mixed_pair_prices_only_its_actual_column_modules(pair,contains,excludes):
    from woof.core.preflight import domain_kernel_modules
    selected=SimpleNamespace(run=config(*pair),grid_id=1)
    modules=domain_kernel_modules(selected,prices_refl=False)
    assert contains<=modules
    assert not modules&excludes
