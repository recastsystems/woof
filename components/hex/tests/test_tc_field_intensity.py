"""Intensity is a native-cell diagnostic, independent of motion smoothing."""
import os
import numpy as np
import pytest
from woof.hex import storm_tracker as st
from test_storm_tracker import cloud_about, vortex_state


@pytest.mark.skipif(not os.environ.get('WOOF_TC_FRAMES_LIB'), reason='TC native library required')
def test_native_eye_does_not_change_motion_consensus():
    centre = (20.,-80.)
    lat,lon = cloud_about(*centre)
    fields = vortex_state(lat,lon,centre)
    without = st.TCFixer().fix(0.,fields,centre,150.)
    order = np.argsort(st.distance_km(*centre,lat,lon))
    pressure = fields.mslp_pa.copy()
    land = np.zeros(lat.size)
    pressure[order[0]] = 85000.
    land[order[0]] = 1.
    pressure[order[1]] = 90200.
    wind = np.zeros(lat.size)
    wind[order[2]] = 60.
    wind[order[-1]] = 100.
    fields.diagnostic_arrays = dict(lat=lat,lon=lon,mslp_pa=pressure,wind10=wind,landmask=land)
    fixed = st.TCFixer().fix(0.,fields,centre,150.)
    assert fixed.ok and without.ok
    assert (fixed.lat,fixed.lon) == (without.lat,without.lon)
    assert fixed.mslp_min_hpa == 902.
    assert fixed.vmax10_ms == 60.
    assert fixed.field_fix['cell_index'] == int(order[1])
    fields.diagnostic_arrays['landmask'] = np.ones(lat.size)
    inland = st.TCFixer().fix(0.,fields,centre,150.)
    assert inland.ok and inland.mslp_min_hpa is None
    assert 'no finite water-cell pressure' in inland.field_fix_error


def test_history_native_and_arwen_field_names_agree(tmp_path):
    nc = pytest.importorskip('netCDF4')
    from woof.hex.moving_run import history_fields
    results = []
    for native in (False, True):
        path = tmp_path / f'history-{native}.nc'
        with nc.Dataset(path, 'w') as dataset:
            for name,size in [('Time',1),('nCells',4),('nVertLevels',3),('nVertLevelsP1',4)]:
                dataset.createDimension(name,size)
            def put(name,dims,values):
                dataset.createVariable(name,'f8',dims)[:] = values
            put('latCell',('nCells',),np.radians([20.,20.1,20.2,20.3]))
            put('lonCell',('nCells',),np.radians([-80.,-80.1,-80.2,-80.3]))
            if native:
                put('zgrid',('nCells','nVertLevelsP1'),np.array([0.,100.,5000.,15000.])[None,:]+np.arange(4)[:,None])
            else:
                put('ter',('nCells',),np.arange(4))
            dims=('Time','nCells','nVertLevels')
            put('pressure',dims,np.broadcast_to([95000.,70000.,20000.],(1,4,3)))
            put('theta',dims,300.)
            put('uReconstructZonal' if native else 'u_zonal',dims,12.)
            put('uReconstructMeridional' if native else 'v_meridional',dims,8.)
            for name,value in [('surface_pressure',97000.),('u10',15.),('v10',20.)]:
                put(name,('Time','nCells'),value)
        results.append(history_fields(st,path,(20.,-80.),150.,native_landmask=np.zeros(4)))
    left,right=results
    for name in ('lat','lon','mslp_pa','wind10','land'):
        np.testing.assert_array_equal(getattr(left,name),getattr(right,name))
    for key in left.u_p:
        np.testing.assert_array_equal(left.u_p[key],right.u_p[key])
        np.testing.assert_array_equal(left.v_p[key],right.v_p[key])
    for key in left.diagnostic_arrays:
        np.testing.assert_array_equal(left.diagnostic_arrays[key],right.diagnostic_arrays[key])
