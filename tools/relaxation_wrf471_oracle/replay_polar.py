"""Replay real WRF polar-filter module calls with resident binary64 FFTs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from tools.relaxation_wrf471_oracle.replay import grade


def read_capture(path):
    raw = Path(path).read_bytes()
    magic, call, nx, ny, nz, jstart, global_ny, _ = map(int, np.frombuffer(raw, '<i4', 8))
    if magic != 3103:
        raise ValueError("not a native polar-filter capture")
    filt, dvlat = np.frombuffer(raw, '<f4', 2, 32)
    lat = np.frombuffer(raw, '<f4', nx*ny, 40).reshape((nx, ny), order='F').T
    count = nx*ny*nz
    before = np.frombuffer(raw, '<f4', count, 40+nx*ny*4).reshape((nx, nz, ny), order='F').transpose(1, 2, 0)
    after = np.frombuffer(raw, '<f4', count, 40+nx*ny*4+count*4).reshape((nx, nz, ny), order='F').transpose(1, 2, 0)
    if len(raw) != 40+4*nx*ny+8*count:
        raise ValueError("invalid polar capture size")
    last = min(ny-1, global_ny-1-jstart)
    if dvlat != 0 and last == global_ny-1-jstart:
        last = min(ny-1, global_ny-jstart)
    return (nx, ny, nz), lat, before, after, dict(filter_lat=float(filt), dvlat=float(dvlat), last_row=last), dict(
        path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw), call=call)


def run(directory):
    import cupy as cp
    from woof.core.global_polar import PolarFFT
    from woof.certify.kernel_manifest import kernel_manifest
    result = []
    for path in sorted(Path(directory).glob('polar-*.bin')):
        dims, lat, before, after, kw, capture = read_capture(path)
        field = cp.asarray(np.ascontiguousarray(before))
        operator = PolarFFT(*dims, latitude=cp.asarray(np.ascontiguousarray(lat)), **kw)
        operator.apply(field)
        row = dict(capture=capture, shape=list(field.shape), configuration=kw,
                   output=grade(cp.asnumpy(field), after),
                   native_changed_words=int(np.count_nonzero(np.ascontiguousarray(before).view(np.uint32)
                                                            != np.ascontiguousarray(after).view(np.uint32))))
        result.append(row)
    return dict(captures=result, words=sum(x['output']['words'] for x in result),
                different_words=sum(x['output']['different_words'] for x in result),
                passed=bool(result) and all(x['output']['different_words'] == 0 for x in result),
                kernel_manifest=kernel_manifest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('receipt', type=Path)
    args = parser.parse_args()
    result = run(args.directory)
    args.receipt.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['captures'], indent=2))
    raise SystemExit(0 if result['passed'] else 1)
