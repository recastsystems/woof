"""PWP public controls, scientific axes, and memory admission contracts."""
import dataclasses
import pytest
from test_oml_contract import run_config
from woof.surface_ocean import ocean_fields, ocean_input_fields, drop_inert_ocean, validate_pwp_profile


def test_pwp_inventory_has_independent_ocean_axis():
    from woof.ocean.mixed_layer import PROFILE_FIELDS, STATE_FIELDS
    from woof.core.preflight import physics_array_shapes
    from woof.io.history_layout import produced_history_shapes
    from woof.ingest.wrfinput import OPTIONAL_WRFINPUT, WRFINPUT_DIMENSIONS
    cfg=run_config(sf_ocean_physics=2, ocean_levels=19)
    shapes=physics_array_shapes(cfg)
    assert ocean_fields(cfg)==STATE_FIELDS
    for name in PROFILE_FIELDS:
        assert shapes['fields/'+name]==(19,cfg.ny,cfg.nx)
    assert shapes['fields/_pwp_base']==(76,cfg.ny,cfg.nx)
    assert shapes['fields/_pwp_status']==(1,)
    assert shapes['fields/om_active']==(cfg.ny,cfg.nx)
    outputs=produced_history_shapes(cfg)
    for name in ('OCEAN_SST', 'OCEAN_SST_CHANGE'):
        assert outputs[name]==(cfg.ny,cfg.nx)
    for name in PROFILE_FIELDS:
        assert outputs[name.upper()]==(19,cfg.ny,cfg.nx)
    for name in ('OM_TMP','OM_S','OM_U','OM_V','OM_DEPTH','OM_TINI','OM_SINI'):
        assert name in OPTIONAL_WRFINPUT
        assert WRFINPUT_DIMENSIONS[name]==('ocean_levels','south_north','west_east')
    marker=object()
    assert ocean_input_fields(cfg,{'OM_TMP':marker})=={'ocean_initial':{'om_tmp':marker}}


def test_off_identity_omits_every_new_ocean_control():
    values=dataclasses.asdict(run_config())
    drop_inert_ocean(values)
    assert not any(name in values for name in ('ocean_levels','ocean_analysis','ocean_analysis_source','ocean_mld_climatology','omdt','ocean_z','ocean_t','ocean_s'))


@pytest.mark.parametrize('ocean_levels', [4, 7])
def test_pwp_history_writes_native_stagger_on_independent_ocean_axis(tmp_path, ocean_levels):
    """Initial storm history must write every profile, even at nz levels."""
    import netCDF4
    import numpy as np
    from woof.io.wrfout import WrfoutWriter
    names = ('OM_TMP', 'OM_S', 'OM_U', 'OM_V', 'OM_DEPTH',
             'OM_DENSITY', 'OM_TINI', 'OM_SINI')
    shape = (ocean_levels, 3, 5)
    fields = {name: np.full(shape, i + 1, np.float32)
              for i, name in enumerate(names)}
    fields['T'] = np.zeros((4, 3, 5), np.float32)
    fields['W'] = np.zeros((5, 3, 5), np.float32)
    fields['OM_ML'] = np.full((3, 5), 15., np.float32)
    path = tmp_path / 'wrfout.nc'
    with WrfoutWriter(path, nx=5, ny=3, nz=4, dx=1000., dy=1000.) as writer:
        writer.write_frame('2024-10-08_00:00:00', fields)
    with netCDF4.Dataset(path) as ds:
        assert len(ds.dimensions['ocean_levels']) == ocean_levels
        for name in names:
            var = ds.variables[name]
            assert var.dimensions == ('Time', 'ocean_levels', 'south_north', 'west_east')
            assert var.stagger == 'Z'
            np.testing.assert_array_equal(var[0], fields[name])
        assert ds.variables['T'].stagger == ''
        assert ds.variables['W'].stagger == 'Z'
        assert ds.variables['OM_ML'].stagger == ''


@pytest.mark.parametrize('changes,reason',[
    ({'ocean_z':[5.,10.]},'ocean_t'),
    ({'ocean_z':[5.,5.],'ocean_t':[300.,299.],'ocean_s':[35.,35.]},'strictly increasing'),
    ({'ocean_z':[5.,10.],'ocean_t':[26.,25.],'ocean_s':[35.,35.]},'Kelvin'),
    ({'ocean_analysis_source':'unknown'},'unknown ocean_analysis_source'),
    ({'omdt':0},'positive minutes'),
    ({'ocean_mld_climatology':'unknown'},'ocean_mld_climatology'),
])
def test_profiles_refuse_named_density_corruption(changes,reason):
    cfg=run_config(sf_ocean_physics=2,ocean_levels=2,**changes)
    with pytest.raises(ValueError,match=reason):validate_pwp_profile(cfg)


def test_real_namelist_pwp_profiles_are_consumed(tmp_path):
    from test_namelist_import import WPS_TEXT, INPUT_TEXT
    from woof.namelist_import import import_namelists
    import tomllib
    inp=INPUT_TEXT.replace('&physics','&physics\n sf_ocean_physics=2, omdt=1.25, omlcall=0,',1)
    inp=inp.replace('&domains','&domains\n ocean_levels=3, ocean_z=5.,10.,20., ocean_t=300.,299.,297., ocean_s=35.,35.,35.,',1)
    wps=tmp_path/'namelist.wps';path=tmp_path/'namelist.input'
    wps.write_text(WPS_TEXT);path.write_text(inp)
    text,report=import_namelists(wps,path)
    assert 'sf_ocean_physics = 2' in text
    assert 'ocean_levels = 3' in text
    assert 'omdt = 1.25' in text
    assert 'ocean_z = [5.0, 10.0, 20.0]' in text
    tomllib.loads(text)


@pytest.mark.parametrize('month',[0,13,True,9.5])
def test_woa18_refuses_invalid_calendar_month_before_fetch(month,tmp_path):
    from woof.ocean.mixed_layer.climatology import acquire
    with pytest.raises(ValueError,match='calendar month'):
        acquire(month,cache=tmp_path)


def test_pwp_solver_status_is_checked_as_integer_and_requires_zero():
    """The full forecast failed admission before step one without this policy."""
    import numpy as np
    from woof.core.health import field_from_array, gpu_integer_policy, validate_fields_cpu
    name = "surface._pwp_status"
    assert gpu_integer_policy(name, np.int32) is not None
    for status in (0, 1, 2, 3, -1):
        field = field_from_array(name, np.array([status], dtype=np.int32))
        report = validate_fields_cpu((field,))
        assert report.ok == (status == 0)
        if status:
            assert report.first_bad_field == name
            assert report.failing_classes == ("surface",)


def test_history_masks_ocean_diagnostics_without_changing_live_state(monkeypatch):
    import sys
    import numpy as np
    from woof.core.surface_ocean import ocean_history_diagnostic
    from woof.io.history_layout import _HistoryShape
    monkeypatch.setitem(sys.modules, 'cupy', np)
    value = np.array([[300., 0., 0.]], dtype=np.float32)
    fields = {'xland': np.array([[2., 1., 2.]], dtype=np.float32),
              'lakemask': np.array([[0., 0., 1.]], dtype=np.float32)}
    output = ocean_history_diagnostic(value, fields)
    np.testing.assert_array_equal(value, [[300., 0., 0.]])
    assert output[0, 0] == 300.
    assert np.isnan(output[0, 1:]).all()
    shape = _HistoryShape((3, 4))
    assert ocean_history_diagnostic(shape, {}) is shape


@pytest.mark.parametrize('duration,selection,expected', [
    (172800, None, 2), (259200, None, 2), (172740, None, 0),
    (259200, 0, 0), (259200, 1, 1), (259200, 2, 2),
])
def test_experiment_multiday_ocean_default_and_explicit_optout(tmp_path, duration, selection, expected):
    from test_experiment import _write
    from woof.experiment import load_experiment
    shared = 'sf_sfclay_physics = 91\nsf_surface_physics = 2'
    if selection is not None:
        shared += f'\nsf_ocean_physics = {selection}'
    path = _write(tmp_path, shared=shared)
    path.write_text(path.read_text().replace('run_seconds = 3600.0', f'run_seconds = {duration}.0'))
    exp = load_experiment(path)
    assert all(dc.run.sf_ocean_physics == expected for dc in exp.domains)


def test_namelist_preserves_explicit_off_for_multiday_default(tmp_path):
    from test_namelist_import import WPS_TEXT, INPUT_TEXT
    from woof.namelist_import import import_namelists
    path = tmp_path/'namelist.input'; wps = tmp_path/'namelist.wps'
    path.write_text(INPUT_TEXT.replace('&physics', '&physics\n sf_ocean_physics=0,', 1))
    wps.write_text(WPS_TEXT)
    text, _ = import_namelists(wps, path)
    assert 'sf_ocean_physics = 0' in text
