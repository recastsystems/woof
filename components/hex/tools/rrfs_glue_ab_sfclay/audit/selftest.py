"""Audit coverage on actual CUDA calls, including implicit scalar conversions."""
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit import Audit
import cupy as cp
import numpy as np

cp.zeros(1)  # context setup is not the coverage test
collector = Audit(os.environ['RRFS_AUDIT_CUPTI']).start()
a = cp.array([7], dtype=cp.int32)
assert a.item() == 7
assert int(a[0]) == 7
assert a.get()[0] == 7
assert cp.asnumpy(a)[0] == 7
pinned = cp.cuda.alloc_pinned_memory(a.nbytes)
out = np.frombuffer(pinned, dtype=np.int32, count=a.size)
a.get(out=out, blocking=False)
cp.cuda.get_current_stream().synchronize()
cp.cuda.Device().synchronize()
collector.stop(os.environ['RRFS_AUDIT_OUT'])
import json
data = json.loads(Path(os.environ['RRFS_AUDIT_OUT']).read_text())
rows = [r for r in data['cuda'] if r['layer'] == 'driver']
assert sum(r['count'] for r in rows if r['category'] == 'D2H') == 5, rows
assert any('StreamSynchronize' in r['api'] for r in rows)
assert any('CtxSynchronize' in r['api'] for r in rows)
assert not data['errors']
assert sum(r['count'] for r in data['python'] if r['api'] == 'implicit.int') == 1
print('PASS: five D2H copies, item/int/get/asnumpy/pinned get, stream/device sync')
