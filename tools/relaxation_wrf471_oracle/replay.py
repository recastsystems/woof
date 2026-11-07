"""Replay active words captured from the strict native global ideal run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read_capture(path):
    raw = Path(path).read_bytes()
    header = np.frombuffer(raw, dtype='<i4', count=20)
    kind, call, *bounds = map(int, header)
    names = 'ids ide jds jde kds kde ims ime jms jme kms kme its ite jts jte kts kte'.split()
    b = dict(zip(names, bounds))
    ni, nj, nk = (b[last]-b[first]+1 for first, last in
                  (('ims','ime'), ('jms','jme'), ('kms','kme')))
    offset = 80

    def take(count, shape):
        nonlocal offset
        a = np.frombuffer(raw, dtype='<f4', count=count, offset=offset).reshape(shape, order='F')
        offset += 4*count
        return a

    if kind == 1:
        constants = take(3, (3,))
        lat = take(ni*nj, (ni,nj)).T
        arrays = [take(ni*nk*nj, (ni,nk,nj)).transpose(1,2,0) for _ in range(6)]
        values = dict(zip(('p8w','p','pi','t','before','after'), arrays))
        values.update(constants=constants, lat=lat)
    elif kind == 2:
        arrays = [take(ni*nk*nj, (ni,nk,nj)).transpose(1,2,0) for _ in range(8)]
        values = dict(zip(('p','pb','u','v','u_before','v_before','u_after','v_after'), arrays))
    else:
        raise ValueError(f'unknown capture kind {kind}')
    if offset != len(raw):
        raise ValueError(f'{path}: consumed {offset} bytes of {len(raw)}')
    return b, values, dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                           bytes=len(raw), call=call, bounds=b)


def grade(actual, expected):
    a = np.ascontiguousarray(actual).view(np.uint32)
    e = np.ascontiguousarray(expected).view(np.uint32)
    order = lambda x: np.where(x & np.uint32(0x80000000), ~x, x | np.uint32(0x80000000)).astype(np.int64)
    return dict(words=a.size, different_words=int(np.count_nonzero(a != e)),
                max_ulp=int(np.abs(order(a)-order(e)).max(initial=0)),
                actual_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
                native_sha256=hashlib.sha256(e.tobytes()).hexdigest())


def run(directory):
    import cupy as cp
    from woof.core.kernels import load_module_no_ftz
    from woof import numerics, wrf_exact
    from woof.certify.kernel_manifest import kernel_manifest

    module = load_module_no_ftz('idealized_forcing')
    device = lambda a: cp.asarray(np.ascontiguousarray(a))
    result = dict(numerics=numerics.receipt(), wrf_exact=wrf_exact.ENABLED,
                  device=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
                  captures=[], outputs={})
    for kind in (1,2):
        b, a, receipt = read_capture(Path(directory)/f'capture-{kind}.bin')
        result['captures'].append(receipt)
        ix, iy, iz = b['its']-b['ims'], b['jts']-b['jms'], b['kts']-b['kms']
        nx = min(b['ite'],b['ide']-1)-b['its']+1
        ny = min(b['jte'],b['jde']-1)-b['jts']+1
        nz = min(b['kte'],b['kde']-1)-b['kts']+1
        mass = np.s_[iz:iz+nz,iy:iy+ny,ix:ix+nx]
        if kind == 1:
            heat = device(a['before'][mass])
            degrad, rd, specific_heat = a['constants']
            kernel = module.get_function('relax_temperature')
            kernel(((nx*ny*nz+255)//256,), (256,), (
                np.int32(nx),np.int32(ny),np.int32(nz),
                device(a['lat'][iy:iy+ny,ix:ix+nx]),device(a['p'][mass]),
                device(a['p8w'][1-b['kms'],iy:iy+ny,ix:ix+nx]),
                device(a['pi'][mass]),device(a['t'][mass]),
                np.float32(degrad),np.float32(rd),np.float32(specific_heat),heat))
            result['outputs']['RTHRATEN'] = grade(cp.asnumpy(heat), a['after'][mass])
        else:
            if b['its'] != b['ids'] or b['jts'] != b['jds'] or b['kts'] != 1:
                raise ValueError('this whole-domain replay requires one native MPI rank')
            halo = np.s_[iz:iz+nz,iy-1:iy+ny+1,ix-1:ix+nx+1]
            p, pb = device(a['p'][halo]), device(a['pb'][halo])
            for axis, name in ((0,'u'),(1,'v')):
                face = np.s_[iz:iz+nz,iy:iy+ny+(axis==1),ix:ix+nx+(axis==0)]
                tendency = device(a[name+'_before'][face])
                wind = device(a[name][face])
                stop = nx+1 if axis == 0 else ny
                if axis == 1:
                    # Production uses this gathered physical-boundary mask
                    # for both resident domains and reused compute windows.
                    active = cp.ones((ny+1,nx),cp.float32)
                    active[0] = 0
                    active[-1] = 0
                    module.get_function('relax_momentum_masked')(
                        ((wind.size+255)//256,), (256,), (
                            np.int32(nx),np.int32(ny),np.int32(nz),
                            active,p,pb,wind,tendency))
                else:
                    module.get_function('relax_momentum')(
                        ((wind.size+255)//256,), (256,), (
                            np.int32(nx),np.int32(ny),np.int32(nz),np.int32(axis),np.int32(stop),
                            p,pb,wind,tendency))
                result['outputs'][name.upper()+'_TEND'] = grade(cp.asnumpy(tendency),a[name+'_after'][face])
    result['kernel_manifest'] = kernel_manifest()
    result['passed'] = all(x['different_words'] == 0 for x in result['outputs'].values())
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture_directory', type=Path)
    parser.add_argument('receipt', type=Path)
    args = parser.parse_args()
    result = run(args.capture_directory)
    args.receipt.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['outputs'],indent=2))
    raise SystemExit(0 if result['passed'] else 1)
