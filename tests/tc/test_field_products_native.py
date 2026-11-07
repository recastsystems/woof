"""Exercise the shipped ABI on a land low, distant wind and swath resume."""
from datetime import datetime, timedelta
import os
from types import SimpleNamespace as N
import numpy as np
import pytest
from woof.tc.field_products import diagnose, diagnose_arrays, atcf_rows


@pytest.mark.skipif(not os.environ.get('WOOF_TC_FRAMES_LIB'), reason='native TC library not configured')
def test_water_minimum_local_wind_and_swath_resume(tmp_path):
    lat, lon = np.meshgrid(np.linspace(-3,3,61), np.linspace(-3,3,61), indexing='ij')
    fields = {k:np.full((61,61),v) for k,v in
              dict(psfc=101000.,t2=300.,q2=.015,u10=0.,v10=0.,xland=2.).items()}
    fields['psfc'][30,30] = 90200.
    fields['psfc'][30,31] = 85000.
    fields['xland'][30,31] = 1.
    fields['u10'][31,30] = 60.
    fields['u10'][60,60] = 100.
    state = N(physics=N(fields=fields), ht=np.zeros((61,61)))
    grid = N(latlon_mass=lambda:(lat,lon))
    swath = dict(path=str(tmp_path/'wind.f32'),south=-3.,north=3.,west=-3.,east=3.,spacing_deg=.1)
    first = diagnose(state,grid,(0.,0.),swath=swath)
    assert (first['mslp_hpa'],first['vmax10_m_s']) == (902.,60.)
    assert (first['latitude_deg'],first['longitude_deg']) == (0.,0.)
    assert first['radii_nm']['64'][0] > 0.
    cloud = diagnose_arrays(dict(lat=lat.ravel(),lon=lon.ravel(),
        mslp_pa=fields['psfc'].ravel(),wind10=fields['u10'].ravel(),
        landmask=(fields['xland'].ravel()<1.5).astype(float)), (0.,0.), unstructured=True)
    assert (cloud['mslp_hpa'],cloud['vmax10_m_s']) == (902.,60.)
    fields['u10'][31,30] = 40.
    metadata = tmp_path/'wind.json'
    metadata.chmod(0o444)
    try:
        second = diagnose(state,grid,(0.,0.),swath=swath)
    finally:
        metadata.chmod(0o644)
    assert second['vmax10_m_s'] == 40.
    values = np.fromfile(tmp_path/'wind.f32',dtype='<f4').reshape(61,61)
    assert values[31,30] == 60. and values[60,60] == 100.
    issue = datetime(2024,10,8)
    rows = atcf_rows(first,storm_id='AL142024',initial_time=issue,valid_time=issue+timedelta(hours=6))
    assert len(rows) == 3
    assert rows[0].split(', ')[8:10] == ['117','902']
    with pytest.raises(ValueError, match='changed on resume'):
        diagnose(state,grid,(0.,0.),swath=dict(swath,spacing_deg=.2))
