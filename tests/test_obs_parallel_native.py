"""Native field and file identity under simultaneous superob and merge calls.

Synthetic radar gates exercise the real Rust ABI and writer. This does not
replace the public Iowa/CONUS full-run identity gate or its timings.
"""
import hashlib
import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from woof.obs import superob_bridge
from woof.obs.cc_qc import CcQcParams
from woof.obs.dealias import DealiasParams
from woof.obs.radar_grid import write_radar_grid
from woof.obs.superob import SuperobParams, merge_contributions, superob_volume
from test_obs_superob_rust_parity import _grid, _volume, ACCUMULATORS, FIELDS

pytestmark = pytest.mark.skipif(superob_bridge.unavailable_reason() is not None,
                               reason='native rw-superob must be staged')


def digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def test_concurrent_native_fields_and_radar_files_are_identical(tmp_path, monkeypatch):
    monkeypatch.delenv('WOOF_SUPEROB_PYTHON', raising=False)
    grid = _grid(nx=31, ny=31, nz=8)
    volumes = [_volume(grid, j=15, i=15, site_id=f'KP{i:02d}', seed=i)
               for i in range(2)]
    params = SuperobParams(dealias=DealiasParams(engine='region-global'), cc_qc=CcQcParams())
    def run(volume):
        return superob_volume(volume, grid, params=params, clear_air_from_censor=True)
    serial = [run(volume) for volume in volumes]
    with ThreadPoolExecutor(max_workers=2) as pool:
        concurrent = list(pool.map(run, volumes))
    for left, right in zip(serial, concurrent):
        for field in ACCUMULATORS:
            assert digest(getattr(left, field)) == digest(getattr(right, field)), field
        assert left.counts.to_payload() == right.counts.to_payload()
        assert left.dealias == right.dealias and left.cc_qc == right.cc_qc
    serial_merge = merge_contributions(serial, grid, params=params)
    with ThreadPoolExecutor(max_workers=2) as pool:
        merges = list(pool.map(lambda _: merge_contributions(concurrent, grid, params=params), range(2)))
    for merged in merges:
        for field in FIELDS:
            assert digest(getattr(serial_merge, field)) == digest(getattr(merged, field)), field
    path = tmp_path / 'radar.nc'
    write_radar_grid(path, serial_merge, grid, valid_time='2026-10-07T12:00:00Z', params=params)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    write_radar_grid(path, merges[0], grid, valid_time='2026-10-07T12:00:00Z', params=params, overwrite=True)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    receipt = {'scope': 'synthetic native concurrency check, not a public replay',
        'status': 'PASS', 'workers': 2, 'radar_file_sha256': before,
        'accumulator_sha256': {f'{i}/{field}': digest(getattr(c, field))
            for i, c in enumerate(serial) for field in ACCUMULATORS},
        'merged_field_sha256': {field: digest(getattr(serial_merge, field)) for field in FIELDS}}
    target = Path(os.environ.get('WOOF_OBS_NATIVE_RECEIPT', str(tmp_path / 'native-identity.json')))
    target.write_text(json.dumps(receipt, indent=2) + '\n')


def test_replay_target_preserves_columns_and_keeps_production_georef_refusal(tmp_path):
    from woof.io.classic_product import ClassicProduct
    from woof.obs.target_grid import TargetGrid, GridMismatchError
    from woof.obs.replay_target_grid import build
    grid = _grid(nx=7, ny=7, nz=3)
    source, destination = tmp_path / 'recorded.nc', tmp_path / 'replay.nc'
    with ClassicProduct(source) as product:
        for name, count in (('Time', 1), ('south_north', 7), ('west_east', 7), ('bottom_top_stag', 4)):
            product.createDimension(name, count)
        product.setncatts(dict(MAP_PROJ=1, TRUELAT1=grid.truelat1, TRUELAT2=grid.truelat2,
            STAND_LON=grid.stand_lon, CEN_LAT=grid.ref_lat, CEN_LON=grid.ref_lon, DX=grid.dx_m, DY=grid.dy_m))
        for name, value in dict(PH=grid.z_w * 9.81, PHB=np.zeros_like(grid.z_w), HGT=grid.terrain_m,
                                XLAT=grid.lat + 0.01, XLONG=grid.lon).items():
            dims = ('Time', 'bottom_top_stag', 'south_north', 'west_east') if value.ndim == 3 else ('Time', 'south_north', 'west_east')
            product.createVariable(name, value.dtype, dims)[:] = value[None]
    with pytest.raises(GridMismatchError, match='disagrees with its own XLAT/XLONG'):
        TargetGrid.from_wrfout(source)
    receipt = build(source, destination)
    assert receipt['shape'] == [3, 7, 7]
    assert set(receipt['preserved_column_words']) == {'PH', 'PHB', 'HGT'}
    assert TargetGrid.from_wrfout(destination).lat.tobytes() == grid.lat.tobytes()
    with pytest.raises(ValueError, match='already exists'):
        build(source, destination)
