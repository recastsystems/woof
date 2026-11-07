"""Replay real WRF call-boundary streams through production CUDA launchers.

Usage: GPUWM_WRF_EXACT=1 python replay_native.py CAPTURE_DIR RECEIPT.json
The instrumenter next to this script records 32 columns from each of the
first two real calls per MPI tile. Both integer and real outputs compare
as uint32 words, preserving signed zeros and NaN payloads. A nonzero word
count or an empty capture directory returns failure. Fast arithmetic is
also measurable by leaving GPUWM_WRF_EXACT unset in a fresh process.
"""
from pathlib import Path
import numpy as np
import cupy as cp
import json, hashlib, sys
from woof.core.gfs_pbl import gfs_pbl_step, GFS_PBL_COLUMN_OUTPUTS
from woof.core.sf_gfs import launch_gfs_sfclay, GFS_SFCLAY_INPUTS, GFS_SFCLAY_OUTPUTS, GFS_SFCLAY_INOUT
root = Path(sys.argv[1])
receipt = Path(sys.argv[2])
rows = []
receipt.parent.mkdir(parents=True, exist_ok=True)

def compare(name, want, got):
    a = np.asarray(want).view(np.uint32)
    b = np.asarray(got).view(np.uint32)
    bad = a != b
    return {'field': name, 'words': a.size, 'differing_words': int(bad.sum()), 'nonfinite_native': int((~np.isfinite(want)).sum()) if want.dtype.kind == 'f' else 0, 'max_abs': float(np.max(np.abs(want.astype(float) - got.astype(float)))) if want.dtype.kind == 'f' else int(np.max(np.abs(want - got)))}
for p in sorted(root.glob('capture-*.bin')):
    with p.open('rb') as f:

        def read(n, dtype='<f4'):
            result = np.fromfile(f, dtype=dtype, count=n)
            if result.size != n:
                raise ValueError(f'{p}: truncated stream at {f.tell()}')
            return result

        def dev(a):
            return cp.asarray(np.ascontiguousarray(a))
        fields = []
        if p.name.startswith('capture-bl'):
            nx, nz, ny, flqi = map(int, read(4, '<i4'))
            dt = float(read(1)[0])
            n = nx * ny
            vol = {name: read(n * nz).reshape(ny, nz, nx).transpose(1, 0, 2).copy() for name in ('u', 'v', 'theta', 't', 'qv', 'qc', 'qi', 'p', 'exner', 'dz', 'z')}
            sfc = {name: read(n).reshape(ny, nx) for name in ('psfc', 'ust', 'pblh', 'psim', 'psih', 'hfx', 'qfx', 'tsk', 'gz1oz0', 'wspd', 'br')}
            want = {name: read(n * nz).reshape(ny, nz, nx).transpose(1, 0, 2).copy() for name in GFS_PBL_COLUMN_OUTPUTS}
            for name in ('ust', 'wspd', 'pblh', 'kpbl'):
                want[name] = read(n, '<i4' if name == 'kpbl' else '<f4').reshape(ny, nx)
            dvol = {k: dev(v) for k, v in vol.items()}
            dsfc = {k: dev(v) for k, v in sfc.items()}
            got = gfs_pbl_step(dvol, dsfc, dt=dt, flqi=bool(flqi))
            got['wspd'] = dsfc['wspd']
            for name, a in want.items():
                fields.append(compare(name, a, cp.asnumpy(got[name])))
        else:
            nx, ny, isfflx, itimestep = map(int, read(4, '<i4'))
            n = nx * ny
            constants = read(7)
            inputs = {name: read(n).reshape(ny, nx) for name in GFS_SFCLAY_INPUTS}
            state = {name: read(n).reshape(ny, nx) for name in GFS_SFCLAY_INOUT}
            want = {name: read(n).reshape(ny, nx) for name in (*GFS_SFCLAY_OUTPUTS, *GFS_SFCLAY_INOUT)}
            outs = {name: cp.empty((ny, nx), cp.float32) for name in GFS_SFCLAY_OUTPUTS}
            ds = {k: dev(v) for k, v in state.items()}
            launch_gfs_sfclay({k: dev(v) for k, v in inputs.items()}, ds, outs, isfflx=isfflx)
            got = dict(outs, **ds)
            for name, a in want.items():
                fields.append(compare(name, a, cp.asnumpy(got[name])))
        assert f.read() == b'', p
    rows.append({'capture': p.name, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size, 'fields': fields, 'words': sum((v['words'] for v in fields)), 'differing_words': sum((v['differing_words'] for v in fields))})
summary = {'schema': 'gfs-native-module-replay-v1', 'strict': __import__('woof.numerics', fromlist=['is_strict']).is_strict(), 'numerics': __import__('woof.numerics', fromlist=['receipt']).receipt(), 'compiler_options': {name: __import__('woof.core.kernels', fromlist=['module_options']).module_options(name) for name in ('gfs_pbl', 'sf_gfs')}, 'gpu': cp.cuda.runtime.getDeviceProperties(0)['name'].decode(), 'rows': rows, 'words': sum((r['words'] for r in rows)), 'differing_words': sum((r['differing_words'] for r in rows))}
receipt.write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}))
sys.exit(bool(summary['differing_words']) or not rows)
