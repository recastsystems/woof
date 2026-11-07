from datetime import datetime
import pytest
from woof.config import RunConfig
from woof.experiment import experiment_from_run_config

@pytest.mark.gpu
@pytest.mark.parametrize('nz',[3,30,44,50,70,100,192,256])
def test_price_covers_the_production_driver_frame(nz):
    from woof.core.eeps_pbl import local_frame_bytes
    from woof.core.preflight import kernel_local_frame_bytes
    cfg=RunConfig(nx=16,ny=16,nz=nz,dx=3000,dy=3000,ztop=15000,dt=15,run_seconds=60,moist=True,mp_physics=1,bl_pbl_physics=16,sf_sfclay_physics=1,sf_surface_physics=2)
    exp=experiment_from_run_config(cfg,datetime(2020,1,1))
    observed=local_frame_bytes(nz)
    assert kernel_local_frame_bytes(exp)['eeps']>=observed
