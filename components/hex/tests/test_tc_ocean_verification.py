"""Verification conventions which could otherwise turn an apparent gain into an artifact."""
import gzip,importlib.util
from datetime import datetime,timedelta
from pathlib import Path
import numpy as np
from netCDF4 import Dataset
import pytest
spec=importlib.util.spec_from_file_location('tc_verification',Path(__file__).parents[1]/'tools/evaluate_tc_pwp.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_metric_excludes_missing_pairs_and_keeps_bias():
    r=m.metric([1,3,np.nan,7],[2,1,4,np.nan])
    assert r['n']==2 and r['bias']==.5
    assert r['rmse']==pytest.approx(np.sqrt(2.5))

@pytest.mark.parametrize('compressed',[False,True])
def test_buoy_response_compression_is_detected_from_bytes(tmp_path,compressed):
    raw=b'#YY MM DD hh mm WTMP\n#yr mo dy hr mn degC\n2022 09 26 00 00 29.2\n2022 09 26 01 00 99.0\n'
    p=tmp_path/'station.txt.gz';p.write_bytes(gzip.compress(raw) if compressed else raw)
    assert m.buoy_text(p)==[(datetime(2022,9,26),29.2)]

def test_daily_sst_uses_endpoint_weights_and_common_analysis_support(tmp_path):
    start=datetime(2022,9,26);on={};off={};out=tmp_path/'out';out.mkdir()
    for h in range(0,49,6):
        for arm,files in [('on',on),('off',off)]:
            valid=start+timedelta(hours=h);p=tmp_path/f'{arm}-{h}.npz';temp=300+h/24 if arm=='on' else 300
            np.savez(p,valid=valid.isoformat(),latCell=[20.,20.],lonCell=[-80.,-79.],coordinate_units='degrees',sst=[temp,350.],tsk=[temp,350.],ocean_pwp_active=[1,0]);files[valid]=p
    for day in (0,1):
        key=(start+timedelta(days=day)).strftime('%Y%m%d');d=tmp_path/'oisst'/key;d.mkdir(parents=True)
        with Dataset(d/f'oisst-avhrr-v02r01.{key}.nc','w') as nc:
            nc.createDimension('lat',1);nc.createDimension('lon',2)
            nc.createVariable('lat','f8',('lat',))[:]=[20.];nc.createVariable('lon','f8',('lon',))[:]=[280.,281.]
            nc.createVariable('sst','f8',('lat','lon'))[:]=[[300+.5+day-273.15,0]]
    observed={start:dict(LAT='20',LON='-80')}
    result=m.daily_sst(on,off,observed,tmp_path,out,25.)
    assert [r['pixels'] for r in result['days']]==[1,1]
    assert result['all_on']['rmse']==pytest.approx(0,abs=1e-12)
    assert result['all_off']['rmse']==pytest.approx(np.sqrt((.5**2+1.5**2)/2))
    assert result['days'][1]['change_on']['rmse']==pytest.approx(0,abs=1e-12)


def test_hex_ocean_reduction_uses_named_level_axis(tmp_path):
    spec=importlib.util.spec_from_file_location('tc_reduce',Path(__file__).parents[1]/'tools/reduce_coupled_ocean_history.py');reduce=importlib.util.module_from_spec(spec);spec.loader.exec_module(reduce)
    source=tmp_path/'frame.nc';out=tmp_path/'out';out.mkdir()
    values=np.arange(6,dtype=np.float32).reshape(3,2)
    with Dataset(source,'w') as d:
        for name,size in [('Time',1),('nCells',3),('nOceanLevels',2)]:d.createDimension(name,size)
        d.createVariable('ocean_temperature','f4',('Time','nCells','nOceanLevels'))[:]=values[None]
    record=reduce.reduce((str(source),str(out)))
    with np.load(out/'frame.npz') as z:np.testing.assert_array_equal(z['ocean_temperature'],values.T)
    assert record['arrays']['ocean_temperature']['shape']==(1,3,2)


def test_initializer_missing_wind_is_not_a_hurricane_maximum():
    d=dict(u10=np.array([-1.e30,0.,30.],dtype=np.float32),v10=np.array([-1.e30,0.,40.],dtype=np.float32))
    out=m.wind10(d)
    assert np.isnan(out[0])
    np.testing.assert_array_equal(out[1:],[0.,50.])
    assert m.metric(out,[10.,0.,50.])['n']==2
