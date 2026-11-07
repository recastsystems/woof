"""Native initialization, integrity and read-once ownership on real WRF assets."""
from pathlib import Path
import os
import numpy as np
import pytest
from woof.core.cam3_tables import Cam3Tables

@pytest.fixture
def real_root():
    value=os.environ.get('CAM_WRF_TABLE_ROOT')
    if not value:pytest.skip('real WRF table root required')
    return Path(value)

def test_native_table_context_is_read_once(real_root,tmp_path):
    names=('CAM_ABS_DATA','CAM_AEROPT_DATA','CAMtr_volume_mixing_ratio','ozone_plev.formatted','ozone_lat.formatted','ozone.formatted')
    for name in names:(tmp_path/name).symlink_to(real_root/name)
    first=Cam3Tables(tmp_path);sample=first.state[:256].copy()
    for name in names:(tmp_path/name).unlink()
    second=Cam3Tables(tmp_path)
    assert second.handle==first.handle
    assert np.array_equal(second.state[:256],sample)
    assert np.isfinite(second.gases(2026,np.float32(150.25))).all()
    second.close();first.close()

def test_changed_gas_record_is_rejected(real_root,tmp_path):
    for name in ('CAM_ABS_DATA','CAM_AEROPT_DATA','ozone_plev.formatted','ozone_lat.formatted','ozone.formatted'):(tmp_path/name).symlink_to(real_root/name)
    data=bytearray((real_root/'CAMtr_volume_mixing_ratio').read_bytes());data[-5]^=1
    (tmp_path/'CAMtr_volume_mixing_ratio').write_bytes(data)
    with pytest.raises(RuntimeError,match='SHA-256'):
        Cam3Tables(tmp_path)

def test_native_ozone_records_preserve_wrf_element_order(real_root):
    # The current shared ozone family pins a different text serialization.
    # Decode the official CAM files independently rather than weakening it.
    native=Cam3Tables(real_root)
    pressure=np.loadtxt(real_root/'ozone_plev.formatted',dtype=np.float32)*np.float32(100.)
    latitude=np.loadtxt(real_root/'ozone_lat.formatted',dtype=np.float32)
    ozone=np.loadtxt(real_root/'ozone.formatted',dtype=np.float32).reshape(12,64,59).transpose(2,1,0).copy()
    for left,right in ((native.pin,pressure),(native.latitude,latitude),(native.ozone,ozone)):
        assert np.array_equal(left.view(np.uint32),right.view(np.uint32))
    native.close()


def test_shared_staging_creates_root_and_is_idempotent(real_root,tmp_path):
    from woof.table_assets import stage_cam3_tables,CAM3_TABLE_ASSETS
    root=tmp_path/'new-cache'
    assert stage_cam3_tables(real_root,root)==0
    stamps={a.filename:(root/a.filename).stat().st_mtime_ns for a in CAM3_TABLE_ASSETS}
    assert stage_cam3_tables(real_root,root)==0
    assert stamps=={a.filename:(root/a.filename).stat().st_mtime_ns for a in CAM3_TABLE_ASSETS}
    native=Cam3Tables(root)
    assert np.isfinite(native.gases(2024,np.float32(141.875))).all()
    native.close()
