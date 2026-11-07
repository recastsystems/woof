"""Selected land physics survives stock namelist import and runtime binding."""
import pytest
from test_noahmp_no_consumer_knobs import _noahmp_cfg
from test_namelist_import import _import_with
from woof.config import validate_run_config


@pytest.mark.parametrize("options", [dict(opt_btr=2,opt_rsf=2),dict(opt_alb=1,opt_btr=3,opt_rsf=3), dict(opt_rsf=4), *(dict(opt_snf=v) for v in (2,3,4,5)), *(dict(dveg=v) for v in (1,3)), dict(opt_inf=2), dict(opt_run=4), dict(opt_inf=2,opt_run=4), dict(opt_run=6), dict(opt_inf=2,opt_run=6), dict(opt_run=7), dict(opt_inf=2,opt_run=7), dict(opt_rad=2), dict(opt_rad=2,opt_alb=1), dict(opt_rad=1), dict(opt_rad=1,opt_alb=1), dict(opt_frz=2), dict(opt_frz=2,opt_inf=2), dict(opt_stc=2), dict(opt_tbot=1), dict(opt_stc=3), dict(opt_stc=2,opt_tbot=1), dict(opt_crs=2), dict(opt_crs=2,opt_stc=2), dict(opt_sfc=2), dict(opt_sfc=2,opt_crs=2,opt_stc=2), *(dict(dveg=v) for v in (7,8,9)), *(dict(opt_run=v) for v in (1,2)), *(dict(opt_run=v,opt_inf=2) for v in (1,2)), dict(opt_soil=2), dict(opt_soil=3), dict(opt_soil=4), *(dict(dveg=v) for v in (2,5,6)), *(dict(opt_run=8,opt_infdv=v) for v in (1,2,3)), dict(opt_tdrn=1), dict(opt_tdrn=2)])
def test_real_replayed_options_are_admitted_together(options):
    validate_run_config(_noahmp_cfg(**options))


@pytest.mark.parametrize("values", [(2,2,2),(1,3,3)])
def test_namelist_preserves_selected_options(tmp_path,values):
    alb,btr,rsf=values
    text, report = _import_with(tmp_path, extra_input=f'&noah_mp opt_alb={alb}, opt_btr={btr}, opt_rsf={rsf} /\n')
    for name,value in zip(('opt_alb','opt_btr','opt_rsf'),values):
        if name == 'opt_alb' and value == 2:
            assert 'opt_alb = 1' not in text  # Default values may be omitted.
        else:
            assert f'{name} = {value}' in text

@pytest.mark.parametrize("value", [2,3,4,5])
def test_namelist_preserves_snow_partition(tmp_path,value):
    text, report = _import_with(tmp_path, extra_input=f'&noah_mp opt_snf={value} /\n')
    assert f'opt_snf = {value}' in text

@pytest.mark.parametrize("value", [1,2,3,5,6,7,8,9])
def test_namelist_preserves_vegetation_fraction(tmp_path,value):
    text, report = _import_with(tmp_path, extra_input=f"&noah_mp dveg={value} /\n")
    assert f"dveg = {value}" in text

@pytest.mark.parametrize("value", [1,3,4,7,8,9])
def test_prescribed_vegetation_cold_input_disposition(value):
    from woof.ingest.wrfinput_noahmp import input_dispositions
    rows = input_dispositions(_noahmp_cfg(dveg=value))
    assert rows["LFMASS"] == "inactive under selected surface options"

@pytest.mark.parametrize("value", [2,5,6])
def test_dynamic_vegetation_initializes_native_carbon_pools(value):
    from woof.ingest.wrfinput_noahmp import input_dispositions
    assert input_dispositions(_noahmp_cfg(dveg=value))["LFMASS"] == "initialized from common cold-start fields"


@pytest.mark.parametrize("options", [dict(opt_inf=2), dict(opt_run=4),
                                      dict(opt_inf=2,opt_run=4), dict(opt_run=6), dict(opt_inf=2,opt_run=6), dict(opt_run=7), dict(opt_inf=2,opt_run=7), dict(opt_rad=2), dict(opt_rad=2,opt_alb=1), dict(opt_rad=1), dict(opt_rad=1,opt_alb=1), dict(opt_frz=2), dict(opt_frz=2,opt_inf=2), dict(opt_stc=2), dict(opt_tbot=1), dict(opt_stc=3), dict(opt_stc=2,opt_tbot=1), dict(opt_crs=2), dict(opt_crs=2,opt_stc=2), dict(opt_sfc=2), dict(opt_sfc=2,opt_crs=2,opt_stc=2), *(dict(dveg=v) for v in (7,8,9)), *(dict(opt_run=v) for v in (1,2)), *(dict(opt_run=v,opt_inf=2) for v in (1,2)), dict(opt_soil=2), dict(opt_soil=3), dict(opt_soil=4), *(dict(dveg=v) for v in (2,5,6)), *(dict(opt_run=8,opt_infdv=v) for v in (1,2,3)), dict(opt_tdrn=1), dict(opt_tdrn=2)])
def test_water_selectors_survive_stock_import_and_cold_initialization(tmp_path, options):
    from woof.ingest.wrfinput_noahmp import input_dispositions
    assignments = ", ".join(f"{key}={value}" for key,value in options.items())
    text, _ = _import_with(tmp_path, extra_input=f"&noah_mp {assignments} /\n")
    for key,value in options.items():
        assert f"{key} = {value}" in text
    assert input_dispositions(_noahmp_cfg(**options))["LFMASS"] == ("initialized from common cold-start fields" if options.get("dveg") in (2,5,6) else "inactive under selected surface options")


@pytest.mark.parametrize("options", [dict(opt_rad=2), dict(opt_rad=2,opt_alb=1), dict(opt_rad=1), dict(opt_rad=1,opt_alb=1), dict(opt_frz=2), dict(opt_frz=2,opt_inf=2), dict(opt_stc=2), dict(opt_tbot=1), dict(opt_stc=3), dict(opt_stc=2,opt_tbot=1), dict(opt_crs=2), dict(opt_crs=2,opt_stc=2), dict(opt_sfc=2), dict(opt_sfc=2,opt_crs=2,opt_stc=2), *(dict(dveg=v) for v in (7,8,9)), *(dict(opt_run=v) for v in (1,2)), *(dict(opt_run=v,opt_inf=2) for v in (1,2)), dict(opt_soil=2), dict(opt_soil=3), dict(opt_soil=4), *(dict(dveg=v) for v in (2,5,6)), *(dict(opt_run=8,opt_infdv=v) for v in (1,2,3)), dict(opt_tdrn=1), dict(opt_tdrn=2)])
def test_canopy_radiation_survives_stock_import_and_cold_initialization(tmp_path, options):
    from woof.ingest.wrfinput_noahmp import input_dispositions
    assignments = ", ".join(f"{key}={value}" for key,value in options.items())
    text, _ = _import_with(tmp_path, extra_input=f"&noah_mp {assignments} /\n")
    for key,value in options.items():
        assert f"{key} = {value}" in text
    assert input_dispositions(_noahmp_cfg(**options))["LFMASS"] == ("initialized from common cold-start fields" if options.get("dveg") in (2,5,6) else "inactive under selected surface options")
