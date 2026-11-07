"""File-defined WRF identity is checked before restoring any model state."""
from datetime import datetime
from types import SimpleNamespace
import ast
from pathlib import Path

import numpy as np
import pytest

from woof import netcdf_bridge
from woof.ingest.wrfinput_identity import read_wrfinput_identity, check_wrfinput_identity
from conftest import requires_netcdf_bridge

netCDF4 = pytest.importorskip('netCDF4')


def _file(tmp_path, **attrs):
    try:
        netcdf_bridge.resolve_netcdf_bin()
    except netcdf_bridge.NetcdfBridgeMissing:
        pytest.skip('build rw_netcdf to exercise the actual decoder')
    path = tmp_path/'wrfinput_d01'
    with netCDF4.Dataset(path, 'w') as ds:
        for name, size in {'Time':1, 'DateStrLen':19, 'west_east':3, 'west_east_stag':4,
                           'south_north':2, 'south_north_stag':3, 'bottom_top':2,
                           'bottom_top_stag':3, 'soil_layers_stag':4}.items():
            ds.createDimension(name, size)
        metadata={'GRID_ID':1,'MP_PHYSICS':6,'SF_SURFACE_PHYSICS':2,
                  'USE_THETA_M':1,'HYBRID_OPT':2,'ETAC':np.float32(.2),
                  'DX':3000.,'DY':3000.,'START_DATE':'2021-12-30_17:00:00'}
        ds.setncatts(metadata | attrs)
        ds.createVariable('ZNW', 'f4', ('Time','bottom_top_stag'))[:] = [[1,.6,0]]
        ds.createVariable('ZNU', 'f4', ('Time','bottom_top'))[:] = [[.8,.3]]
        ds.createVariable('P_TOP', 'f4', ('Time',))[:] = [5000.]
        ds.createVariable('Times', 'S1', ('Time','DateStrLen'))[:] = np.frombuffer(b'2021-12-30_17:00:00', dtype='S1').reshape(1,19)
    return path


def _request():
    return dict(domain=SimpleNamespace(grid_id=1,run=SimpleNamespace(nx=3,ny=2,nz=2,
                    dx=3000.,dy=3000.,mp_physics=6,sf_surface_physics=2)),
                vertical=SimpleNamespace(p_top=5000.,hybrid_opt=2,etac=.2,eta_levels=(1.,.6,0.)),
                start_time=datetime(2021,12,30,17),soil_layers=4)


@requires_netcdf_bridge
@pytest.mark.parametrize('representation', [0,1])
def test_file_identity_accepts_both_standard_theta_flags(tmp_path, representation):
    identity=read_wrfinput_identity(_file(tmp_path,USE_THETA_M=representation))
    check_wrfinput_identity(identity, **_request())


@pytest.mark.parametrize('field,value,label', [
    ('p_top',7000.,'P_TOP'),('hybrid_opt',1,'HYBRID_OPT'),('etac',.3,'ETAC'),
    ('eta_levels',(1.,.5,0.),'ZNW')])
@requires_netcdf_bridge
def test_vertical_value_change_is_refused_even_when_level_count_matches(tmp_path,field,value,label):
    identity=read_wrfinput_identity(_file(tmp_path))
    request=_request();setattr(request['vertical'],field,value)
    with pytest.raises(ValueError, match=label):
        check_wrfinput_identity(identity, **request)


@requires_netcdf_bridge
def test_land_surface_identity_is_not_inferred_from_soil_layer_count(tmp_path):
    identity=read_wrfinput_identity(_file(tmp_path,SF_SURFACE_PHYSICS=4))
    with pytest.raises(ValueError, match='SF_SURFACE_PHYSICS'):
        check_wrfinput_identity(identity, **_request())


# PER CASE, not per function: only the Times mutation is read through the
# bridge.  The other three are refused off the vertical arrays the reader
# already holds and pass with no staged estate, and they are refusal
# tests, so a function-wide gate lost three of them to a stale binary.
@pytest.mark.parametrize('mutation,label', [
    pytest.param('time','Times',marks=requires_netcdf_bridge),
    ('midpoint','ZNU'),('missing_eta','ZNW'),
    ('fractional_flag','finite integer')])
def test_malformed_file_identity_is_refused(tmp_path,mutation,label):
    path=_file(tmp_path)
    with netCDF4.Dataset(path,'a') as ds:
        if mutation=='time': ds['Times'][:] = np.frombuffer(b'2021-12-30_18:00:00',dtype='S1').reshape(1,19)
        elif mutation=='midpoint': ds['ZNU'][:] = [[.7,.2]]
        elif mutation=='missing_eta': ds.renameVariable('ZNW','unrelated')
        else: ds.HYBRID_OPT=1.5
    with pytest.raises(ValueError,match=label): read_wrfinput_identity(path)


def test_identity_module_has_no_forecast_or_python_decoder_dependency():
    source=Path(__file__).resolve().parents[1]/'woof/ingest/wrfinput_identity.py'
    imported=[]
    for node in ast.walk(ast.parse(source.read_text())):
        if isinstance(node,ast.Import): imported.extend(alias.name for alias in node.names)
        if isinstance(node,ast.ImportFrom): imported.append(node.module or '')
    assert not any(name.startswith(('woof.core','woof.runtime','netCDF4','h5py','xarray','cupy')) for name in imported)


@requires_netcdf_bridge
def test_native_written_child_retains_root_origin_but_requires_its_record_time(tmp_path):
    path = _file(tmp_path, GRID_ID=2)
    with netCDF4.Dataset(path, 'a') as ds:
        ds['Times'][:] = np.frombuffer(b'2021-12-30_17:00:12', dtype='S1').reshape(1, 19)
    with pytest.raises(ValueError, match='Times'):
        read_wrfinput_identity(path)
    identity = read_wrfinput_identity(path, child=True)
    request = _request()
    request['domain'].grid_id = 2
    with pytest.raises(ValueError, match='start_time'):
        check_wrfinput_identity(identity, **request)
    request['start_time'] = datetime(2021, 12, 30, 17, 0, 12)
    check_wrfinput_identity(identity, **request)
