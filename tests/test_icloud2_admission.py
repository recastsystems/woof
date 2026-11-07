"""Binary cloud fraction must be reachable through the actual config door."""
import pytest
from woof.config import load_config
from test_config import _write_toml

@pytest.mark.parametrize('variant', ['rrtmg_legacy', 'rte-rrtmgp'])
def test_binary_cloud_fraction_config(tmp_path, variant):
    p = _write_toml(tmp_path, dynamics=f'ra_lw_physics=4\nra_sw_physics=4\nra_rrtmg_variant="{variant}"\nicloud=2')
    assert load_config(p).icloud == 2

@pytest.mark.parametrize('pair', [(1,1), (4,4), (1,4), (4,1)])
@pytest.mark.parametrize('variant', ['rrtmg_legacy', 'rte-rrtmgp'])
def test_binary_cloud_fraction_namelist(tmp_path, pair, variant):
    from test_namelist_import import _pair, INPUT_TEXT
    from woof.namelist_import import import_namelists
    inp = (INPUT_TEXT.replace(' ra_lw_physics = 4, 4,', f' ra_lw_physics = {pair[0]}, {pair[0]},')
        .replace(' ra_sw_physics = 4, 4,', f' ra_sw_physics = {pair[1]}, {pair[1]},')
        .replace('mp_physics = 55, 55', 'mp_physics = 10, 10')
        .replace('&physics\n', '&physics\n icloud = 2,\n'))
    text, _ = import_namelists(*_pair(tmp_path, inp=inp), name='binary_cloud', rrtmg_variant=variant)
    assert 'icloud = 2' in text

def test_icloud3_without_cloud_species_names_breakage(tmp_path):
    # cal_cldfra3 is ported (3.0); WRF's own check refuses it without QC/QI.
    # Replaces test_unported_cloud_fraction_names_breakage, the pre-port
    # refusal of every icloud=3 run.
    p = _write_toml(tmp_path, dynamics='icloud=3')
    with pytest.raises(ValueError, match='icloud = 3 option, missing QC or QI'):
        load_config(p)

def test_binary_fraction_does_not_enter_the_mynn_cloud_merge_guard(tmp_path):
    p = _write_toml(tmp_path, dynamics='ra_lw_physics=4\nra_sw_physics=4\nra_rrtmg_variant="rte-rrtmgp"\nicloud=2\nbl_pbl_physics=5\nsf_sfclay_physics=5\nbl_mynn_version="gsd_41"\nmoist=true\nmp_physics=1')
    assert load_config(p).icloud == 2
