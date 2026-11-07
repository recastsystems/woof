"""Exercise the real configuration, registry, calendar and import doors."""
from dataclasses import replace
from datetime import datetime
from types import SimpleNamespace
import pytest
from woof.config import RunConfig, validate_run_config, CU_SCHEMES, cumulus_adaptive_cudt_zero_site
from woof.experiment import DomainConfig, ExperimentConfig, VerticalConfig
from woof.core.clock import resolve_clock
from woof.core.sas import SAS_VARIANTS, workspace_bytes, constants_sha256, WRF_MODEL_CONSTANTS
from woof.core.preflight import sas_column_workspace_bytes
from woof.physics_registry import component_options, stock_callable_class
def cfg(**kw):
    return RunConfig(nx=30,ny=30,nz=45,dx=12000.,dy=12000.,ztop=18000.,dt=60.,run_seconds=600.,moist=True,cu_physics=14,**kw)
def test_bare_default_config_selects_14():
    assert 14 in CU_SCHEMES
    run=cfg()
    assert validate_run_config(run) is run
    assert stock_callable_class('cumulus',{'cu_physics':14})=='woof.core.sas.SasCumulus'
    row=component_options('cumulus')['kiaps-sas']
    assert row['implemented'] and row['selectors']=={'cu_physics':14}
@pytest.mark.parametrize('minutes',[0.,1.,5.])
def test_real_clock_keeps_fixed_step_cumulus_cadence(minutes):
    run=cfg(cudt_minutes=minutes)
    dc=DomainConfig(grid_id=1,parent_id=0,i_parent_start=1,j_parent_start=1,parent_grid_ratio=1,parent_time_step_ratio=1,history_interval_s=60.,run=run,time_step=60)
    exp=ExperimentConfig(name='sas-clock',start_time=datetime(2000,1,1),run_seconds=600.,vertical=VerticalConfig(eta_levels=(),p_top=0.,hybrid_opt=0,etac=.2),projection=None,restart_interval_s=0.,domains=(dc,))
    spec=resolve_clock(exp).domain_clock(1).spec
    assert spec.stepcu==max(1,int(minutes))
def test_adaptive_rule_names_interval_breakage():
    assert cumulus_adaptive_cudt_zero_site(14)=='share/module_check_a_mundo.F:2141-2160'
    assert cumulus_adaptive_cudt_zero_site(1) is None
    with pytest.raises(ValueError,match='closure is formed over'):
        validate_run_config(cfg(use_adaptive_time_step=True,cudt_minutes=5.))
    validate_run_config(cfg(use_adaptive_time_step=True,cudt_minutes=0.))
def test_workspace_budget_scans_all_domains_and_uses_allocator_formula():
    r=cfg()
    exp=SimpleNamespace(domains=[SimpleNamespace(run=replace(r,cu_physics=0)),SimpleNamespace(run=r)])
    assert sas_column_workspace_bytes(exp)==workspace_bytes(SAS_VARIANTS[14],45,900)+8*900
    assert sas_column_workspace_bytes(SimpleNamespace(domains=[exp.domains[0]]))==0
    assert workspace_bytes(SAS_VARIANTS[14],45,40000)==49*46*32768*4
def test_constants_identity_changes_with_a_constant_bit():
    import numpy as np
    c=dict(WRF_MODEL_CONSTANTS)
    c['cp']=np.nextafter(c['cp'],np.float32(2000.))
    assert constants_sha256(c)!=constants_sha256()
@pytest.mark.parametrize('adaptive',[False,True])
def test_namelist_import_keeps_14_and_applies_adaptive_metadata(tmp_path,adaptive):
    from test_namelist_import import _pair, INPUT_TEXT
    from woof.namelist_import import import_namelists
    import re
    text=re.sub(r'cu_physics\s*=.*', 'cu_physics = 14, 14,',INPUT_TEXT)
    text=re.sub(r'cudt\s*=.*','cudt = 5., 5.,',text)
    if adaptive:text=text.replace('&domains','&domains\n use_adaptive_time_step = .true.,')
    out,report=import_namelists(*_pair(tmp_path,inp=text),name='sas-import')
    assert 'cu_physics = 14' in out
    assert ('cudt_minutes = 0.0' if adaptive else 'cudt_minutes = 5.0') in out


def test_family_constant_rows_have_exact_kernel_anchors():
    from woof import physics_params as pp
    from pathlib import Path
    import re
    source = Path(__file__).parents[1] / "woof/core/kernels/sas.cu"
    keys = re.findall(r"^#define KSAS_(\w+)\s+[-+0-9.e]+f\s*$",
                      source.read_text(), re.M)
    assert len(keys) >= 20
    rows = {n: r for n, r in pp.registry().items() if n.startswith("sas.")}
    assert set(rows) == {"sas." + k.lower() for k in keys}
    for row in rows.values():
        for site in row.sites:
            assert site.module == "sas"
            assert source.read_text().count(site.anchor) == site.count


def test_custom_callable_has_no_stock_scratch_allowance():
    from woof.io.restart import _stock_cumulus_rebuild_arrays
    from woof.core.sas import SasCumulus
    class Custom(SasCumulus):
        pass
    assert not _stock_cumulus_rebuild_arrays(Custom())
    assert _stock_cumulus_rebuild_arrays(SasCumulus()) == frozenset({
        "_hold", "_workspace", "raincv", "hbot", "htop"})
