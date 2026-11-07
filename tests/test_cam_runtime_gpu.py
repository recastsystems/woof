"""Real CAM construction through the registered factory with WRF host geography."""
import os
from datetime import datetime
import numpy as np
import pytest

@pytest.mark.gpu
def test_cam_factory_accepts_host_geography(monkeypatch):
    cp=pytest.importorskip('cupy')
    root=os.environ.get('CAM_WRF_TABLE_ROOT')
    if not root:pytest.skip('real WRF CAM assets required')
    monkeypatch.setenv('WOOF_CAM_TABLE_ROOT',root)
    from woof.config import RunConfig
    from woof.core.radiation_composition import make_radiation
    cfg=RunConfig(nx=12,ny=12,nz=48,dx=3000.,dy=3000.,ztop=20000.,dt=15.,run_seconds=120.,ra_lw_physics=3,ra_sw_physics=3)
    latitude=np.array([[35.,45.]],np.float32);longitude=np.zeros_like(latitude)
    radiation=make_radiation(cfg,datetime(2024,5,21,21),latitude,longitude)
    assert isinstance(radiation.latitude_deg,cp.ndarray)
    np.testing.assert_array_equal(cp.asnumpy(radiation.latitude_deg),latitude)
    assert radiation._ozone.shape==(2,59*13)
    assert bool(cp.isfinite(radiation._ozone).all())
