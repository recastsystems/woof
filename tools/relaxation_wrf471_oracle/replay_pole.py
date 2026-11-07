"""Replay native evolved meridional pole-face prescription inputs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from tools.relaxation_wrf471_oracle.replay import grade


def run(path):
    import cupy as cp
    from woof.core.global_polar import PoleFaces
    from woof.certify.kernel_manifest import kernel_manifest
    raw = Path(path).read_bytes()
    magic, nx, ny, nz = map(int, np.frombuffer(raw, '<i4', 4))
    if magic != 3106 or len(raw) != 16+8*nx*ny*nz:
        raise ValueError('invalid native pole-face capture')
    before, after = (
        np.frombuffer(raw, '<f4', nx*ny*nz, 16+k*4*nx*ny*nz)
        .reshape((nx, nz, ny), order='F').transpose(1, 2, 0)
        for k in (0, 1))
    field = cp.asarray(np.ascontiguousarray(before))
    PoleFaces(nx, ny-1, nz).apply(field, copy_neighbor=True)
    result = grade(cp.asnumpy(field), after)
    return dict(
        capture=dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw)),
        output=result,
        native_changed_words=int(np.count_nonzero(
            np.ascontiguousarray(before).view(np.uint32)
            != np.ascontiguousarray(after).view(np.uint32))),
        passed=result['different_words'] == 0,
        kernel_manifest=kernel_manifest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('receipt', type=Path)
    args = parser.parse_args()
    result = run(args.capture)
    args.receipt.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['output'], indent=2))
    raise SystemExit(0 if result['passed'] else 1)
