"""Supplied RUC/MYNN fields reach their selected existing physics carriers."""
from dataclasses import replace
import sys
from types import SimpleNamespace

import netCDF4
import numpy as np
import pytest

from woof.config import RunConfig, soil_layer_count
from woof.ingest import wrfinput as wi
from wrf_input_fixtures import _small_wrfinput


def _cfg(**changes):
    return RunConfig(**(dict(nx=6, ny=4, nz=3, dx=3000., dy=3000., ztop=10000.,
                             dt=1., run_seconds=0., moist=True, mp_physics=6,
                             sf_surface_physics=3, num_soil_layers=9,
                             sf_sfclay_physics=5, bl_pbl_physics=5) | changes))


def _dimensions(cfg):
    return dict(west_east=cfg.nx, west_east_stag=cfg.nx+1,
                south_north=cfg.ny, south_north_stag=cfg.ny+1,
                bottom_top=cfg.nz, bottom_top_stag=cfg.nz+1,
                soil_layers_stag=soil_layer_count(cfg))


def _values(cfg, qke_name='qke'):
    shape = cfg.ny, cfg.nx
    column = np.arange(cfg.ny*cfg.nx, dtype=np.float32).reshape(shape)
    return {'ACRUNOFF': column+.25, 'RHOSNF': column+100.,
            'SNOWFALLAC': column+.125, 'SOILT1': column+270.,
            qke_name: np.broadcast_to(column*.125+.25, (cfg.nz, *shape)).copy(),
            'qke_adv': np.full((cfg.nz, *shape), 37., dtype=np.float32)}


def _input(path, cfg, values, *, dimensions_override=None):
    _small_wrfinput(path, nz=cfg.nz, ny=cfg.ny, nx=cfg.nx,
                    moisture_names=tuple(wi.active_moisture_map(cfg)))
    with netCDF4.Dataset(path, 'a') as ds:
        ds.createDimension('soil_layers_stag', soil_layer_count(cfg))
        ds.setncatts(dict(GRID_ID=1, MP_PHYSICS=cfg.mp_physics,
                         SF_SURFACE_PHYSICS=cfg.sf_surface_physics,
                         BL_PBL_PHYSICS=cfg.bl_pbl_physics,
                         MMINLU='MODIFIED_IGBP_MODIS_NOAH'))
        for name, value in values.items():
            dims = (dimensions_override or {}).get(name, wi.WRFINPUT_DIMENSIONS[name])
            dtype = 'i4' if np.asarray(value).dtype.kind in 'iu' else 'f4'
            ds.createVariable(name, dtype, ('Time', *dims))[:] = value
    return path


def _read(path, cfg):
    return wi.read_wrfinput(path, cfg=cfg, expected_dimensions=_dimensions(cfg),
                            require_complete=False)


def test_native_drag_statistics_are_preserved_and_reach_the_active_initializer(tmp_path):
    from woof.static.orographic import required_static_fields
    cfg=_cfg(gwd_opt=3)
    names=required_static_fields(0,3)
    values={name:np.arange(cfg.ny*cfg.nx,dtype=np.float32).reshape(cfg.ny,cfg.nx)+i*.125
            for i,name in enumerate(names)}
    values['HGT']=np.full((cfg.ny,cfg.nx),250.,np.float32)
    restored=_read(_input(tmp_path/'drag-input',cfg,values),cfg)
    carried=wi.wrfinput_terrain_drag_inputs(restored,cfg)['terrain_drag_static']
    for name in names:
        np.testing.assert_array_equal(np.asarray(carried[name],dtype=np.float32).view('u4'),values[name].view('u4'))
    np.testing.assert_array_equal(carried['HGT_M'],values['HGT'])


def test_native_drag_rejects_missing_or_wrong_column_geometry(tmp_path):
    cfg=_cfg(gwd_opt=3)
    with pytest.raises(ValueError,match='CONLS'):
        wi.wrfinput_terrain_drag_inputs(SimpleNamespace(raw={'HGT':np.zeros((4,6))}),cfg)
    path=_input(tmp_path/'wrong-drag',cfg,{'CONLS':np.ones((cfg.nx,cfg.ny),np.float32)},
                dimensions_override={'CONLS':('west_east','south_north')})
    with pytest.raises(ValueError,match='CONLS.*shape mismatch'):
        _read(path,cfg)


def test_off_drag_does_not_change_existing_initialization_arguments():
    assert wi.wrfinput_terrain_drag_inputs(SimpleNamespace(raw={}),_cfg())=={}


def test_microphysics_off_admits_only_passive_vapor_when_moist():
    cfg = _cfg(mp_physics=0)
    vapor = frozenset(("QVAPOR",))
    assert wi.active_moisture_inventory(cfg) == (vapor, vapor)
    assert dict(wi.active_moisture_map(cfg)) == {"QVAPOR": "qv"}
    assert wi.active_moisture_inventory(replace(cfg, moist=False)) == (
        frozenset(), frozenset())


def test_passive_vapor_reader_restores_exact_words_and_rk_initial_copy(tmp_path):
    cfg = _cfg(mp_physics=0, sf_surface_physics=0, num_soil_layers=4)
    values = np.asarray(np.arange(cfg.nz * cfg.ny * cfg.nx).reshape(
        cfg.nz, cfg.ny, cfg.nx) * np.float32(1.e-5) + np.float32(.001),
        dtype=np.float32)
    path = _input(tmp_path / "input", cfg, {})
    with netCDF4.Dataset(path, "a") as dataset:
        dataset["QVAPOR"][:] = values
    restored = _read(path, cfg)
    state = SimpleNamespace(qv=np.full(values.shape, -1., np.float32),
                            qv0=np.full(values.shape, -2., np.float32),
                            qc=np.full(values.shape, 7., np.float32))
    wi._restore_active_moisture(state, restored.raw, cfg, np)
    np.testing.assert_array_equal(state.qv.view("u4"), values.view("u4"))
    np.testing.assert_array_equal(state.qv0.view("u4"), values.view("u4"))
    assert np.all(state.qc == np.float32(7.))
    assert wi.read_wrfinput_metadata(path).has_qv is True


def test_passive_vapor_keeps_missing_and_inactive_field_refusals(tmp_path):
    cfg = _cfg(mp_physics=0, sf_surface_physics=0, num_soil_layers=4)
    dry = replace(cfg, moist=False)
    dry_path = _input(tmp_path / "dry", dry, {})
    restored = _read(dry_path, dry)
    assert "QVAPOR" not in restored.raw
    assert wi.read_wrfinput_metadata(dry_path).has_qv is False
    with pytest.raises(ValueError, match="missing mapped.*QVAPOR"):
        wi.read_wrfinput(dry_path, cfg=cfg,
                        expected_dimensions=_dimensions(cfg))
    vapor_path = _input(tmp_path / "vapor", cfg, {})
    with pytest.raises(ValueError, match="inactive WRF moisture.*QVAPOR"):
        _read(vapor_path, dry)
    with netCDF4.Dataset(vapor_path, "a") as dataset:
        dataset.createVariable("QCLOUD", "f4", (
            "Time", *wi.WRFINPUT_DIMENSIONS["QCLOUD"]))[:] = 0.
    with pytest.raises(ValueError, match="inactive WRF moisture.*QCLOUD"):
        _read(vapor_path, cfg)


@pytest.mark.parametrize('qke_name', ['qke', 'QKE'])
def test_reader_retains_exact_ruc_and_mynn_words_and_inactive_advected_record(tmp_path, qke_name):
    cfg = _cfg(); values = _values(cfg, qke_name)
    result = _read(_input(tmp_path/'input', cfg, values), cfg)
    for name, value in values.items():
        # The Rust bridge exposes decoded reals as float64. Every source
        # float32 value is represented exactly and restores to the same word.
        np.testing.assert_array_equal(result.raw[name], value)
        np.testing.assert_array_equal(result.raw[name].astype('f4').view('u4'), value.view('u4'))
        assert name in result.mapped_variables


@pytest.mark.parametrize('name', ['ACRUNOFF', 'RHOSNF', 'SNOWFALLAC', 'SOILT1', 'qke', 'QKE', 'qke_adv'])
@pytest.mark.parametrize('poison', ['shape', 'nan', 'masked'])
def test_new_physics_fields_keep_closed_geometry_and_value_checks(tmp_path, name, poison):
    cfg = _cfg()
    dimensions = ({name: ('bottom_top', 'south_north', 'west_east_stag')}
                  if poison == 'shape' else None)
    value = 5. if dimensions else np.full(
        tuple(_dimensions(cfg)[d] for d in wi.WRFINPUT_DIMENSIONS[name]), 5., np.float32)
    path = _input(tmp_path/'input', cfg, {name: value}, dimensions_override=dimensions)
    if poison != 'shape':
        with netCDF4.Dataset(path, 'a') as ds:
            if poison == 'nan': ds[name][:] = np.nan
            else:
                ds[name].missing_value = np.float32(9.e20)
                ds[name][:] = np.float32(9.e20)
    with pytest.raises(ValueError, match='shape mismatch|non-finite|masked'):
        _read(path, cfg)


def test_new_fields_require_their_selected_consumer_and_qke_aliases_must_agree(tmp_path):
    cfg = _cfg(); values = _values(cfg)
    path = _input(tmp_path/'input', cfg, values)
    with pytest.raises(ValueError, match='RUC.*sf_surface_physics=3'):
        _read(path, replace(cfg, sf_surface_physics=2, num_soil_layers=4))
    with pytest.raises(ValueError, match='MYNN.*bl_pbl_physics=5'):
        _read(path, replace(cfg, bl_pbl_physics=1))
    with netCDF4.Dataset(path, 'a') as ds:
        ds.createVariable('QKE', 'f4', ('Time', *wi.WRFINPUT_DIMENSIONS['QKE']))[:] = values['qke']
    _read(path, cfg)
    with netCDF4.Dataset(path, 'a') as ds: ds['QKE'][0,0,0,0] += 1.
    with pytest.raises(ValueError, match='conflicting aliases'):
        _read(path, cfg)


def test_qke_adv_is_retained_for_the_selected_transport(tmp_path):
    cfg = _cfg(); path = _input(tmp_path/'input', cfg, _values(cfg))
    restored = _read(path, replace(cfg, bl_mynn_tkeadvect=True))
    np.testing.assert_array_equal(restored.raw['qke_adv'], _values(cfg)['qke_adv'])
    wi.read_wrfinput(path, cfg=None, expected_dimensions=_dimensions(cfg),
                     require_complete=False)


def _surface_values(cfg):
    shape = cfg.ny, cfg.nx
    defaults = dict(LANDMASK=1., XICE=0., ALBBCK=.2, LAI=2., TSK=290.,
                    LU_INDEX=10, ISLTYP=6, VEGFRA=60., TMN=288., SNOW=0.,
                    SNOWH=0., GLW=300.)
    values = {name: np.full(shape, value, np.int32 if name in ('LU_INDEX','ISLTYP')
                           else np.float32) for name, value in defaults.items()}
    soil = soil_layer_count(cfg), *shape
    values.update(TSLB=np.full(soil, 288., np.float32),
                  SMOIS=np.full(soil, .3, np.float32), SH2O=np.full(soil, .3, np.float32))
    return values


@pytest.mark.parametrize('supplied', [True, False])
def test_actual_initializer_forwards_supplied_fields_and_preserves_absent_cold_state(
        tmp_path, monkeypatch, supplied):
    cfg = _cfg(); values = _values(cfg)
    raw = _surface_values(cfg) | (values if supplied else {})
    restored = _read(_input(tmp_path/'input', cfg, raw), cfg)
    fields = {name.lower(): np.full(value.shape, -77., np.float32)
              for name, value in values.items() if name != 'qke_adv'}
    fields.update(albbck=np.zeros((cfg.ny,cfg.nx), np.float32),
                  lai=np.zeros((cfg.ny,cfg.nx), np.float32))
    driver = SimpleNamespace(fields=fields, rainc=None, ruc_params=None)
    monkeypatch.setitem(sys.modules, 'cupy', np)
    monkeypatch.setitem(sys.modules, 'woof.core.physics', SimpleNamespace(
        physics_driver_required=lambda cfg: True,
        initialize_physics=lambda *args, **kwargs: driver))
    monkeypatch.setitem(sys.modules, "woof.core.ruc_runtime", SimpleNamespace(
        ruc_cold_start=lambda *args,**kwargs: None))
    assert wi.initialize_wrfinput_physics(object(), restored, cfg) is driver
    for name, value in values.items():
        if name == 'qke_adv':
            assert name not in driver.fields
        else:
            expected = value if supplied else np.full(value.shape, -77., np.float32)
            np.testing.assert_array_equal(fields[name.lower()].view('u4'), expected.view('u4'))


@pytest.mark.parametrize('surface_layer,pbl', [(2, 2), (4, 4)])
@pytest.mark.parametrize('theta_m', [0, 1])
@pytest.mark.parametrize('warm', [False, True])
def test_native_special_surface_winds_seed_every_cold_start(
        tmp_path, monkeypatch, surface_layer, pbl, theta_m, warm):
    cfg = _cfg(sf_surface_physics=1, num_soil_layers=5,
               sf_sfclay_physics=surface_layer, bl_pbl_physics=pbl,
               use_theta_m=theta_m)
    restored = _read(_input(tmp_path/'input', cfg, _surface_values(cfg)), cfg)
    if warm:
        restored = SimpleNamespace(raw=restored.raw,
            global_attributes=restored.global_attributes, warm_start=object())
    shape = (cfg.ny, cfg.nx)
    fields = dict(albbck=np.zeros(shape, np.float32), lai=np.zeros(shape, np.float32),
                  u10=np.arange(cfg.nx*cfg.ny, dtype=np.float32).reshape(shape)+.25,
                  v10=np.full(shape, -3.5, np.float32),
                  u10e=np.full(shape, -77., np.float32),
                  v10e=np.full(shape, -88., np.float32))
    driver = SimpleNamespace(fields=fields, rainc=None, noah_params=None)
    monkeypatch.setitem(sys.modules, 'cupy', np)
    monkeypatch.setitem(sys.modules, 'woof.core.physics', SimpleNamespace(
        physics_driver_required=lambda cfg: True,
        initialize_physics=lambda *args, **kwargs: driver))
    wi.initialize_wrfinput_physics(object(), restored, cfg)
    for target, source, preserved in [('u10e', 'u10', -77.), ('v10e', 'v10', -88.)]:
        expected = np.full(shape, preserved, np.float32) if warm else fields[source]
        np.testing.assert_array_equal(fields[target].view('u4'), expected.view('u4'))


def _from_driver(driver, cp):
    raw = {aliases[0]: cp.asnumpy(driver.fields[field])
           for field, aliases in wi.PHYSICS_FIELD_ALIASES.items()
           if field in driver.fields and aliases[0] in wi.WRFINPUT_DIMENSIONS}
    raw['LU_INDEX'] = raw['IVGTYP']
    return raw


def test_cold_fixture_omits_restart_only_driver_fields():
    vegetation = np.ones((4, 6), dtype=np.int32)
    driver = SimpleNamespace(fields={
        'ivgtyp': vegetation, 'sstsk': np.full((4, 6), 280., np.float32),
        'dtw': np.full((4, 6), 0.25, np.float32)})
    raw = _from_driver(driver, SimpleNamespace(asnumpy=np.asarray))
    assert set(raw) == {'IVGTYP', 'LU_INDEX'}
    np.testing.assert_array_equal(raw['IVGTYP'], vegetation)


class _ConsumerReached(Exception):
    pass


@pytest.mark.gpu
def test_file_values_reach_actual_ruc_driver_arguments(tmp_path, monkeypatch):
    cp = pytest.importorskip('cupy')
    from test_ruc_runtime import _build
    from woof.core import ruc_runtime
    from woof.core.dycore import step
    state, cfg, template = _build(nx=6, ny=4, nz=20)
    values = {name: value for name, value in _values(cfg).items() if name in wi.RUC_INPUT_FIELDS}
    raw = _from_driver(template, cp) | values
    restored = _read(_input(tmp_path/'input', cfg, raw), cfg)
    driver = wi.initialize_wrfinput_physics(state, restored, cfg)
    driver.set_forcing(gsw=0.)
    for name, value in values.items():
        np.testing.assert_array_equal(cp.asnumpy(driver.fields[name.lower()]).view('u4'), value.view('u4'))

    def consume(arguments, **kwargs):
        for name, value in values.items():
            np.testing.assert_array_equal(cp.asnumpy(arguments[name.lower()]).view('u4'), value.view('u4'))
        raise _ConsumerReached

    monkeypatch.setattr(ruc_runtime, 'ruc_land_surface_step', consume)
    with pytest.raises(_ConsumerReached): step(state, cfg)


@pytest.mark.gpu
@pytest.mark.parametrize('qke_name', ['qke', 'QKE'])
def test_file_qke_reaches_actual_mynn_column_arguments(tmp_path, monkeypatch, qke_name):
    cp = pytest.importorskip('cupy')
    from test_mynn_pbl_runtime import _build
    from woof.core import mynn_pbl_runtime
    from woof.core.dycore import step
    state, cfg, template = _build(nx=6, ny=4)
    values = _values(cfg, qke_name)
    raw = _from_driver(template, cp)
    raw.pop('qke', None)
    raw.update({qke_name: values[qke_name], 'qke_adv': values['qke_adv']})
    restored = _read(_input(tmp_path/'input', cfg, raw), cfg)
    driver = wi.initialize_wrfinput_physics(state, restored, cfg)
    np.testing.assert_array_equal(cp.asnumpy(driver.fields['qke']).view('u4'), values[qke_name].view('u4'))

    def consume(arguments, **kwargs):
        expected = values[qke_name].reshape(cfg.nz, -1).T
        np.testing.assert_array_equal(cp.asnumpy(arguments['qke']).view('u4'), expected.view('u4'))
        assert kwargs['initflag'] == 1  # Solver retains stock cold-start authority.
        raise _ConsumerReached

    monkeypatch.setattr(mynn_pbl_runtime, 'mynn_bl_driver_cuda', consume)
    with pytest.raises(_ConsumerReached): step(state, cfg)


def _cloud(cfg):
    shape = cfg.nz, cfg.ny, cfg.nx
    layer = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    return {'QC_BL': layer * np.float32(1.e-6) + np.float32(1.e-5),
            'CLDFRA_BL': (layer % 7) * np.float32(.125)}


def test_a_cycled_start_reads_the_mynn_subgrid_cloud_and_a_cold_one_passes_it(tmp_path):
    """RunConfig.cycling (audit P18/A10): the input QC_BL and CLDFRA_BL reach
    the physics carriers the first radiation call and the gsd_41 MYNN first
    call read; a cold start passes them through unread, as before."""
    cycled = _cfg(bl_mynn_version='gsd_41', cycling=True)
    cloud = _cloud(cycled)
    path = _input(tmp_path/'input', cycled, _surface_values(cycled) | cloud)
    restored = _read(path, cycled)
    for name, value in cloud.items():
        np.testing.assert_array_equal(
            restored.raw[name].astype('f4').view('u4'), value.view('u4'))
    cold = _read(path, replace(cycled, cycling=False))
    assert not (set(cloud) & set(cold.raw))


def test_a_cycled_start_forwards_the_subgrid_cloud_to_the_physics_fields(
        tmp_path, monkeypatch):
    cfg = _cfg(bl_mynn_version='gsd_41', cycling=True)
    cloud = _cloud(cfg)
    restored = _read(_input(tmp_path/'input', cfg,
                            _surface_values(cfg) | cloud), cfg)
    fields = {name.lower(): np.full(value.shape, -77., np.float32)
              for name, value in cloud.items()}
    fields.update(albbck=np.zeros((cfg.ny, cfg.nx), np.float32),
                  lai=np.zeros((cfg.ny, cfg.nx), np.float32))
    driver = SimpleNamespace(fields=fields, rainc=None, ruc_params=None)
    monkeypatch.setitem(sys.modules, 'cupy', np)
    monkeypatch.setitem(sys.modules, 'woof.core.physics', SimpleNamespace(
        physics_driver_required=lambda cfg: True,
        initialize_physics=lambda *args, **kwargs: driver))
    monkeypatch.setitem(sys.modules, "woof.core.ruc_runtime", SimpleNamespace(
        ruc_cold_start=lambda *args,**kwargs: None))
    assert wi.initialize_wrfinput_physics(object(), restored, cfg) is driver
    for name, value in cloud.items():
        np.testing.assert_array_equal(fields[name.lower()].view('u4'),
                                      value.view('u4'))
