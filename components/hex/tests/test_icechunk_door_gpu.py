"""The public forecast door must commit complete native histories beside a real forecast."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not os.environ.get('HEX_ICECHUNK_FORECAST_ARGS'),
    reason='set HEX_ICECHUNK_FORECAST_ARGS to a JSON argv for a real prepared forecast')]


def test_forecast_native_snapshots(tmp_path):
    root = Path(os.environ.get('WOOF_ICECHUNK_TEST_OUT', str(tmp_path/'forecast')))
    args = json.loads(os.environ['HEX_ICECHUNK_FORECAST_ARGS'])
    result = subprocess.run([sys.executable, '-m', 'hexcore', 'forecast', *args,
        '--out', str(root), '--output-format', 'icechunk'], capture_output=True, text=True, timeout=1800)
    assert result.returncode == 0, result.stdout + result.stderr
    _compare(root)


def _compare(root):
    import icechunk
    import numpy as np
    import xarray as xr
    from netCDF4 import Dataset
    receipts = list(root.rglob('output-formats/receipt.json'))
    assert receipts
    for path in receipts:
        receipt = json.loads(path.read_text())
        assert receipt['status'] == 'complete'
        commits = [e for e in receipt['events'] if e['event'] == 'committed']
        assert len(commits) >= 2
        for event in commits:
            repo = icechunk.Repository.open(icechunk.local_filesystem_storage(event['store']))
            session = repo.readonly_session(snapshot_id=event['snapshot'])
            with xr.open_zarr(session.store, consolidated=False, mask_and_scale=False, decode_times=False) as ds, Dataset(event['source']) as source:
                source.set_auto_maskandscale(False)
                assert ds.time.values[-1] == event['time']
                for name, variable in source.variables.items():
                    value = variable[:]
                    if 'Time' in variable.dimensions:
                        value = np.take(value, 0, axis=variable.dimensions.index('Time'))
                    if value.dtype.kind == 'S':
                        value = value.view('u1')
                    np.testing.assert_array_equal(ds[name].isel(time=-1).values, value)
