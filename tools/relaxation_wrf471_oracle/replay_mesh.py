"""Replay native evolved C-grid faces through the mesh-edge operator.

The topology pairs are the exact adjacent mass cells read by WRF. This
checks the edge equation without inventing pressure or momentum profiles.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from tools.relaxation_wrf471_oracle.replay import read_capture, grade


def run(path):
    import cupy as cp
    from woof import numerics
    from woof.core.held_suarez_mesh import HeldSuarezMeshMomentum
    from woof.certify.kernel_manifest import kernel_manifest
    b, a, capture = read_capture(path)
    ix, iy, iz = b['its']-b['ims'], b['jts']-b['jms'], b['kts']-b['kms']
    nx = min(b['ite'], b['ide']-1)-b['its']+1
    ny = min(b['jte'], b['jde']-1)-b['jts']+1
    nz = min(b['kte'], b['kde']-1)-b['kts']+1
    if b['its'] != b['ids'] or b['jts'] != b['jds'] or b['kts'] != 1:
        raise ValueError('requires a whole-domain native capture')
    nj, ni = a['p'].shape[1:]
    cells = np.arange(nj*ni, dtype=np.int32).reshape(nj, ni)
    topology, winds, before, after = [], [], [], []
    for axis, name in ((0, 'u'), (1, 'v')):
        # Native v excludes the south and north polar faces; a closed mesh
        # has no such boundary edges. Native u includes the periodic seam.
        y = slice(iy+(axis == 1), iy+ny)
        x = slice(ix, ix+nx+(axis == 0))
        previous = (cells[y, ix-1:ix+nx] if axis == 0
                    else cells[iy:iy+ny-1, x])
        topology.append(np.stack((previous.ravel(), cells[y,x].ravel()), axis=1))
        for target, field in ((winds, name), (before, name+'_before'),
                              (after, name+'_after')):
            target.append(a[field][iz:iz+nz, y, x].reshape(nz, -1))
    device = lambda value: cp.asarray(np.ascontiguousarray(value))
    pairs = np.concatenate(topology)
    pressure = lambda name: device(a[name][iz:iz+nz].reshape(nz, -1))
    binding = HeldSuarezMeshMomentum()
    binding.bind(device(pairs), pressure('pb'))
    wind = device(np.concatenate(winds, axis=1))
    tendency = device(np.concatenate(before, axis=1))
    binding.apply(tendency, wind, pressure('p'))
    result = grade(cp.asnumpy(tendency), np.concatenate(after, axis=1))
    return dict(capture=capture, numerics=numerics.receipt(),
                edges=int(pairs.shape[0]), output=result,
                kernel_manifest=kernel_manifest(),
                passed=result['different_words'] == 0)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture',type=Path)
    parser.add_argument('receipt',type=Path)
    args=parser.parse_args()
    receipt=run(args.capture)
    args.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt['output'],indent=2))
    raise SystemExit(0 if receipt['passed'] else 1)
