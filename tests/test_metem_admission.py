"""Actual metgrid inventories, preserved controls, and memory admission."""
import gc
from pathlib import Path
from types import SimpleNamespace
import weakref

import numpy as np
import pytest
from conftest import requires_wrf_eta_bridge


def experiment(tmp_path):
    import tomllib
    from test_namelist_import import _pair
    from woof.namelist_import import import_namelists
    from woof.experiment import build_experiment
    text,_=import_namelists(*_pair(tmp_path))
    return build_experiment(tomllib.loads(text), source="metgrid test")


def test_actual_shapes_price_unregistered_source_and_preserve_default(tmp_path):
    from woof.core.preflight import (estimate_ingest, estimate_phases, ingest_analysis_shapes,
                                      source_analysis_fields_per_time)
    exp=experiment(tmp_path)
    # The GFS level count follows the model top the fetch is asked for.
    top=exp.vertical.p_top
    shapes={d.grid_id:ingest_analysis_shapes(d.run,source='gfs',p_top_pa=top) for d in exp.domains}
    default=estimate_ingest(exp,source='gfs')
    actual=estimate_ingest(exp,source='unregistered-format',analysis_shapes_by_domain=shapes,
                           source_fields_per_time=source_analysis_fields_per_time('gfs',p_top_pa=top))
    assert actual.items == default.items
    assert actual.peak_envelope_bytes == default.peak_envelope_bytes
    assert actual.host_fields_per_time == default.host_fields_per_time
    phased=estimate_phases(exp,source='unregistered-format',analysis_shapes_by_domain=shapes,
                           forcing_intervals=7,sequential_domains=True)
    assert phased.ingest.n_forcing_times == 8
    assert phased.ingest.nest_state_bytes == 0
    assert phased.ingest.resident_bytes >= phased.ingest.widest_domain_time_bytes
    assert phased.ingest.resident_bytes < actual.resident_bytes
    assert phased.peak_envelope_bytes == max(phased.forecast_envelope_bytes,phased.ingest_envelope_bytes)
    with pytest.raises(ValueError,match='every experiment domain'):
        estimate_ingest(exp,source='unregistered-format',analysis_shapes_by_domain={})


def test_analyzed_active_number_fields_are_not_silently_zeroed(tmp_path):
    from dataclasses import replace
    from woof.metem_door import check_analyzed_scalar_capability
    cfg=experiment(tmp_path).root.run
    check_analyzed_scalar_capability({'FLAG_QNI':1},replace(cfg,mp_physics=6))
    check_analyzed_scalar_capability({'FLAG_QNI':1},replace(cfg,mp_physics=8))
    with pytest.raises(ValueError,match='FLAG_QNI must be 0 or 1'):
        check_analyzed_scalar_capability({'FLAG_QNI':2},replace(cfg,mp_physics=8))
    check_analyzed_scalar_capability({'FLAG_QNI':0},replace(cfg,mp_physics=8))
    # The water/ice-friendly pair is a TABLE ROW now, not a refusal: the
    # generic metgrid number route carries it onto nwfa/nifa for a package
    # that transports them, and discards it for one that does not, so this
    # door has nothing left to say about either name.
    check_analyzed_scalar_capability({'FLAG_QNWFA':1,'FLAG_QNIFA':1},
                                     replace(cfg,mp_physics=28))
    check_analyzed_scalar_capability({'FLAG_QNWFA':1,'FLAG_QNIFA':1},
                                     replace(cfg,mp_physics=8))
    for name in ('QNWFA','QNIFA','QNBCA'):
        with pytest.raises(ValueError,match=f'FLAG_{name} must be 0 or 1'):
            check_analyzed_scalar_capability({'FLAG_'+name:2},replace(cfg,mp_physics=28))


def test_analyzed_black_carbon_number_is_refused_by_name_at_plan_review(tmp_path):
    """The one real gap, and it used to be the one that never fired.

    The old arm intersected the alias ``nbca`` with ``nest_field_kinds``,
    and no package in this build declares an ``nbca`` species, so the
    intersection was always empty and FLAG_QNBCA=1 walked straight past a
    check written to stop it.  It is tested directly now.
    """
    from dataclasses import replace
    from woof.metem_door import check_analyzed_scalar_capability
    cfg=experiment(tmp_path).root.run
    with pytest.raises(ValueError,match='qnbca') as refused:
        check_analyzed_scalar_capability({'FLAG_QNBCA':1},replace(cfg,mp_physics=28))
    message=str(refused.value)
    assert 'silently dropped' in message and 'Regenerate met_em without QNBCA' in message
    # A package with no aerosol at all loses nothing, so it is not refused.
    check_analyzed_scalar_capability({'FLAG_QNBCA':1},replace(cfg,mp_physics=8))


def test_metgrid_number_units_accept_the_registry_spelling():
    """A real met_em must not be turned away on a units string.

    ``Registry/registry.new3d_wif:88`` spells QNWFA's units ``"# kg(-1)"``
    and a built WRF resolves the ``#`` away to ``"  kg(-1)"``.  Neither
    was accepted, so the WIF stream this ingest now reads would have been
    refused at the reader.
    """
    from types import SimpleNamespace
    from woof.ingest.metem import _unit_transform, MetgridRefusal
    for spelling in ('# kg(-1)', '  kg(-1)', '# kg-1', '#/kg', '1/kg'):
        variable=SimpleNamespace(attributes={'units':spelling})
        assert _unit_transform(variable,'QNWFA')==(1.0,0.0)
    with pytest.raises(MetgridRefusal,match='number per kilogram'):
        _unit_transform(SimpleNamespace(attributes={'units':'kg kg-1'}),'QNWFA')


def test_analyzed_aerosol_is_an_input_in_the_run_report():
    """``metgrid-analyzed`` is an aerosol INPUT, exactly as the file is."""
    from woof.aerosol_source_receipt import (AEROSOL_SOURCE_KEY,
        AEROSOL_SOURCES_FROM_INPUT, aerosol_source_report_entry)
    assert 'metgrid-analyzed' in AEROSOL_SOURCES_FROM_INPUT
    entry=aerosol_source_report_entry({'aerosol_source':'metgrid-analyzed',
        'mp28_aerosol_source':'auto'},mp_physics=28,
        when_unrecorded='not reached')[AEROSOL_SOURCE_KEY]
    assert entry['aerosol_source']=='metgrid-analyzed'
    assert entry['dataset_used'] is True
    assert entry['synthetic_fallback_in_use'] is False


@requires_wrf_eta_bridge
def test_external_wrf_default_preserves_radiation_and_omitted_levels(tmp_path, monkeypatch, capsys):
    from woof.metem_door import resolve_metem_run
    from woof.wrfinput_door import resolve_wrfinput_run
    from woof.wrfinput_forecast import build_parser as wrf_parser
    from woof.metem_forecast import prepare_metem_run
    import inspect
    assert inspect.signature(resolve_metem_run).parameters['rrtmg_variant'].default is None
    assert inspect.signature(resolve_wrfinput_run).parameters['rrtmg_variant'].default is None
    assert wrf_parser().parse_args(['--wrfinput','wrf','--outdir','out']).rrtmg_variant is None
    from test_namelist_import import INPUT_TEXT, _pair
    from woof.namelist_import import import_namelists, parse_namelist_text
    from dataclasses import replace
    import re
    import json
    import tomllib
    from woof.experiment import build_experiment
    fixture = json.loads((Path(__file__).parent/'fixtures/wrf_eta_v461.json').read_text())
    column = next(case for case in fixture['cases'] if case['name'] == 'option2-n80-top5000')
    expected_eta = np.array(column['eta_bits'], np.uint32).view(np.float32)
    namelist = INPUT_TEXT.replace('e_vert = 9, 9,', 'e_vert = 80, 80,')
    namelist = re.sub(r' eta_levels = .*?0\.0,\n',
                      ' eta_levels = ' + ', '.join(map(repr, expected_eta.tolist())) + ',\n',
                      namelist, flags=re.S)
    text,_=import_namelists(*_pair(tmp_path, inp=namelist), metgrid_initialization=True)
    exp=build_experiment(tomllib.loads(text), source='metgrid admission fixture')
    seen=[]
    class AdmissionReached(Exception): pass
    def admission(run, resolved):
        seen.append(resolved)
        raise AdmissionReached
    monkeypatch.setattr('woof.metem_door.metgrid_memory_admission',admission)
    run=SimpleNamespace(toml_text=text,experiment=exp,namelist_input=tmp_path/'namelist.input',
                        controls=parse_namelist_text(namelist))
    with pytest.raises(AdmissionReached): prepare_metem_run(run,tmp_path/'explicit')
    assert seen[-1].vertical.eta_levels == exp.vertical.eta_levels
    run.toml_text=re.sub(r'eta_levels = \[.*?\]\n','',text,flags=re.S)
    run.experiment=replace(exp,vertical=replace(exp.vertical,eta_levels=()))
    run.controls['domains'].pop('eta_levels')
    with pytest.raises(AdmissionReached): prepare_metem_run(run,tmp_path/'default')
    assert len(seen[-1].vertical.eta_levels) == exp.root.run.nz+1
    np.testing.assert_array_equal(np.array(seen[-1].vertical.eta_levels, np.float32), expected_eta)
    assert 'using WRF automatic eta option 2' in capsys.readouterr().out
    assert not (tmp_path/'default').exists()  # admission precedes preparation


def test_uncached_rust_read_releases_promoted_variable(tmp_path):
    from test_metem_ingest import case
    from woof.netcdf_bridge import open_dataset
    with open_dataset(case(tmp_path)) as ds:
        variable=ds.variables['TT']
        values=variable.read_transformed(cache=False)
        witness=values.copy()
        held=weakref.ref(values)
        del values
        gc.collect()
        assert held() is None
        cached=variable.read_transformed()
        np.testing.assert_array_equal(cached,witness)
        held=weakref.ref(variable._values)
        del cached
        gc.collect()
        assert held() is not None  # existing default read still caches


def test_wps_layer_depth_identity_uses_named_bounds_and_shared_midpoints():
    from woof.metem_forecast import metgrid_soil_columns
    from woof.ingest.soil import _remap_declared_soil,NOAH_LAYER_MIDPOINTS_M
    bounds=((0,7),(7,28),(28,100),(100,289))
    mid=np.array([(a+b)//2/100 for a,b in bounds])
    temperature=(280+mid*3)[:,None,None]
    moisture=(.2+mid*.01)[:,None,None]
    soil={'ST':temperature[::-1].copy(),'SM':moisture[::-1].copy(),
          'SOIL_LAYERS':np.array([b for _,b in bounds][::-1])[:,None,None]}
    attrs={}
    for i,(a,b) in enumerate(bounds):
        for prefix,values in (('ST',temperature),('SM',moisture)):
            name=f'{prefix}{a:03d}{b:03d}'
            soil[name]=values[i].copy();attrs['FLAG_'+name]=1
    case=SimpleNamespace(path=Path('met_em'),soil=soil,attributes=attrs,shape=(1,1),
        variable_units={'ST':'K','SM':'fraction','SOIL_LAYERS':''})
    contract,t,m=metgrid_soil_columns(case)
    remapped,_=_remap_declared_soil(t,m,contract,tsk=np.array([[280.]]),deep=np.array([[289.]]))
    np.testing.assert_allclose(remapped[:,0,0],280+3*NOAH_LAYER_MIDPOINTS_M,rtol=0,atol=1e-12)
    soil['ST000007']=soil['ST000007']+1
    with pytest.raises(ValueError,match='differs from the soil stack'):
        metgrid_soil_columns(case)


@pytest.mark.parametrize('lw,sw',[(0,0),(1,1),(4,4)])
def test_preserved_external_radiation_selection(tmp_path,lw,sw):
    from test_namelist_import import INPUT_TEXT,_pair
    from woof.namelist_import import import_namelists
    from woof.experiment import build_experiment
    from woof.config import radiation_scheme_ids
    import tomllib
    text=INPUT_TEXT.replace('ra_lw_physics = 4, 4',f'ra_lw_physics = {lw}, {lw}').replace('ra_sw_physics = 4, 4',f'ra_sw_physics = {sw}, {sw}')
    if lw == sw == 0:
        # Keep this radiation selector test independent of the shared surface
        # energy-forcing policy, which otherwise requires declared GLW.
        text=text.replace('sf_surface_physics = 2, 2','sf_surface_physics = 0, 0').replace('sf_sfclay_physics = 91, 91','sf_sfclay_physics = 0, 0').replace('bl_pbl_physics = 11, 11','bl_pbl_physics = 0, 0').replace('cu_physics = 1, 0','cu_physics = 0, 0')
    paths=_pair(tmp_path,inp=text)
    translated,_=import_namelists(*paths,rrtmg_variant=None)
    cfg=build_experiment(tomllib.loads(translated),source='preserved WRF radiation').root.run
    assert radiation_scheme_ids(cfg) == (lw,sw)
    if lw == 4: assert cfg.ra_rrtmg_variant == 'rrtmg_legacy'
    # Omitted argument on the standalone translator still has its old result.
    legacy_default,_=import_namelists(*paths)
    explicit_modern,_=import_namelists(*paths,rrtmg_variant='rte-rrtmgp')
    assert legacy_default == explicit_modern


@pytest.mark.parametrize("variant", [None, "rte-rrtmgp"])
def test_mixed_external_radiation_preserves_each_selected_arm(tmp_path, variant):
    from test_namelist_import import INPUT_TEXT, _pair
    from woof.namelist_import import import_namelists
    from woof.experiment import build_experiment
    from woof.config import radiation_scheme_ids
    import tomllib
    paths = _pair(tmp_path, inp=INPUT_TEXT.replace(
        'ra_sw_physics = 4, 4', 'ra_sw_physics = 1, 1'))
    text, _ = import_namelists(*paths, rrtmg_variant=variant)
    exp = build_experiment(tomllib.loads(text), source="independent external spectra")
    for domain in exp.domains:
        assert radiation_scheme_ids(domain.run) == (4, 1)
        assert domain.run.ra_rrtmg_variant == (variant or "rrtmg_legacy")


def test_explicit_package_is_not_replaced_by_announced_substitution(tmp_path):
    from test_namelist_import import INPUT_TEXT,_pair
    from woof.namelist_import import import_namelists
    from woof.wrfinput_door import require_preserved_wrf_selectors
    # mp_physics = 55 used to import as an announced Morrison substitution
    # that this gate then refused; the importer now refuses it by name
    # itself (woof.config._MP_UNPORTED_SCHEMES), before any door.
    # ISHMAEL (55) is ported now; the old Eta Ferrier scheme (95) is the
    # WRF package the importer still refuses by name.
    unported=INPUT_TEXT.replace('mp_physics = 10, 10','mp_physics = 95, 95')
    with pytest.raises(ValueError,match='mp_physics=95.*is not ported'):
        import_namelists(*_pair(tmp_path,inp=unported),rrtmg_variant=None)
    # The gate still refuses a package replacement however it arises.
    from dataclasses import replace
    from woof.namelist_import import Substitution
    _,report=import_namelists(*_pair(tmp_path),rrtmg_variant=None)
    replaced=replace(report,substitutions=report.substitutions+(Substitution(
        key='mp_physics',wrf_value=55,wrf_name='ISHMAEL',gpuwm_key='mp_physics',
        gpuwm_value=10,gpuwm_name='Morrison 2-moment'),))
    with pytest.raises(ValueError,match='ISHMAEL.*no native implementation'):
        require_preserved_wrf_selectors(replaced)
    # A native selector remains admitted, including preserved legacy radiation.
    require_preserved_wrf_selectors(report)


@pytest.mark.parametrize("changes", [
    {"key": "mp_physics", "gpuwm_key": "mp_physics", "wrf_value": 55,
     "gpuwm_value": 8},
    {"key": "mp_physics"},
    {"gpuwm_key": "mp_physics"},
    {"wrf_value": 3},
    {"gpuwm_value": 3},
    {"reason": None},
    {"reason": ""},
])
def test_a_reason_only_admits_a_declared_divergence(changes):
    # use_theta_m 1 -> 0 left ADMITTED_DECLARED_DIVERGENCES when moist
    # theta was implemented (bfc7f4d09db); the gate is exercised on a row
    # that is still admitted, fine_input_stream 2 -> 0.
    from dataclasses import replace
    from woof.namelist_import import Substitution
    from woof.wrfinput_door import require_preserved_wrf_selectors

    declared = Substitution(
        key="fine_input_stream", wrf_value=2, wrf_name="delayed nest start",
        gpuwm_key="fine_input_stream", gpuwm_value=0,
        gpuwm_name="own input", reason="The child starts from its own input.")
    require_preserved_wrf_selectors(SimpleNamespace(substitutions=(declared,)))
    with pytest.raises(ValueError, match="no native implementation"):
        require_preserved_wrf_selectors(SimpleNamespace(
            substitutions=(replace(declared, **changes),)))


def test_moist_theta_is_no_longer_a_declared_divergence():
    """use_theta_m=1 runs natively now, so replacing it is refused."""
    from woof.namelist_import import Substitution
    from woof.wrfinput_door import require_preserved_wrf_selectors

    with pytest.raises(ValueError, match="no native implementation"):
        require_preserved_wrf_selectors(SimpleNamespace(substitutions=(
            Substitution(key="use_theta_m", wrf_value=1,
                         wrf_name="moist theta", gpuwm_key="use_theta_m",
                         gpuwm_value=0, gpuwm_name="dry theta",
                         reason="The engine integrates dry theta."),)))
