"""SSiB's real-data initialization uses init_soil_2_real at three depths."""
import numpy as np
from woof.ingest.ruc_soil import preprocess_land_surface_soil

def source():
    a=lambda x: np.array([[x,x]],dtype=np.float32)
    fields={'LANDSEA':a(1), 'SKINTEMP':a(280), 'TMN':a(270)}
    for name,temp,water in zip(('000010','010040','040100','100200'),(290,300,310,320),(.15,.2,.3,.4)):
        fields['GFS_ST'+name]=a(temp)
        fields['GFS_SM'+name]=a(water)
    return fields

def test_selected_soil_geometry_reaches_the_real_initializer():
    soil=preprocess_land_surface_soil(source(),sf_surface_physics=8,
        num_soil_layers=3,soil_type=np.array([[2,2]]),fractional_seaice=True,
        landmask=np.array([[1,0]]))
    assert soil.soil_temperature.shape == (3,1,2)
    assert soil.soil_moisture.shape == (3,1,2)
    # WRF init_soil_2_real samples TSK at zero depth and the 0-10 cm
    # layer at 5 cm. Its deep target lies between 0.7 and 1.5 m.
    np.testing.assert_allclose(soil.soil_temperature[:,0,0], [280,290,314.375],atol=4e-5,rtol=0)
    np.testing.assert_allclose(soil.soil_moisture[:,0,0], [.15,.15,.34375],atol=4e-8,rtol=0)
    np.testing.assert_array_equal(soil.soil_temperature[:,0,1],[280]*3)
    np.testing.assert_array_equal(soil.soil_moisture[:,0,1],[1]*3)


def test_native_sampler_matches_the_compiled_wrf_soil_columns():
    from pathlib import Path
    from woof.ingest.soil_points import sample_points
    path=Path(__file__).parent/'data/oracles/ssib/soil-columns.bin'
    raw=path.read_bytes(); nc,ns,nt=np.frombuffer(raw[:12],'<i4')
    words=np.frombuffer(raw[12:],'<f4'); z=words[:ns]; t=words[ns:ns+nt]; at=ns+nt
    for _ in range(2):
        at+=ns*nc
    output=at
    for k in range(2):
        start=ns+nt+k*ns*nc
        values=words[start:start+ns*nc].reshape(ns,1,nc)
        expected=words[output+k*nt*nc:output+(k+1)*nt*nc].reshape(nt,1,nc)
        actual=sample_points(values,z,t)
        np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))


def test_direct_writer_uses_the_registered_soil_geometry():
    from woof.wrf_direct import _direct_export_soil_geometry
    z,dz=_direct_export_soil_geometry(8,3)
    np.testing.assert_array_equal(z,[0,.05,1.05])
    np.testing.assert_array_equal(dz,[.05,1,0])
