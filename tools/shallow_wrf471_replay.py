"""Replay stream captures from the strict real WRF shallow driver.

Capture recipe: tools/shallow_wrf471_capture.py. No synthetic columns are
created here. The stream contains the driver arguments before each call and
the native returned words after it.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


MASS_IN = ('t', 'qv', 'qc', 'qi', 'p', 'pi', 'rho', 'u', 'v', 'dz8w')
SURFACE_IN = ('raincv', 'xland', 'pblh', 'hfx', 'qfx')
MASS_OUT = ('rthshten', 'rqvshten', 'rqcshten', 'rqishten', 'rushten',
            'rvshten', 'rqrshten', 'rqsshten', 'rqgshten')


def read_capture(path):
    payload = Path(path).read_bytes()
    offset = 0
    def take(count, dtype='<f4'):
        nonlocal offset
        a = np.frombuffer(payload, dtype=dtype, count=count, offset=offset).copy()
        offset += a.nbytes
        return a
    row, step, its, ite, jts, jte, kts, kte = map(int, take(8, '<i4'))
    nx, ny, nz = ite-its+1, jte-jts+1, kte-kts+1
    constants = take(16)
    mp, pbl = map(int, take(2, '<i4'))
    znu = take(nz)
    def mass(n=nz):
        return np.ascontiguousarray(take(nx*ny*n).reshape(ny,n,nx).transpose(1,0,2))
    def surf(dtype='<f4'):
        return take(nx*ny,dtype).reshape(ny,nx)
    fields = {name: mass() for name in MASS_IN}
    fields.update({name: mass(nz+1) for name in ('p8w','z_at_w','w')})
    fields.update({name: surf() for name in SURFACE_IN})
    if row == 3:
        fields.update(wstar=surf(), delta=surf(), kpbl=surf('<i4'))
    expected = {name: mass() for name in MASS_OUT}
    expected.update({name: surf() for name in ('pratesh','hbot','htop')})
    if offset != len(payload):
        raise ValueError(f'{path}: decoded {offset} of {len(payload)} bytes')
    return dict(row=row,step=step,shape=(nz,ny,nx),dt=float(constants[0]),
                mp=mp,pbl=pbl,znu=znu,constants=constants,fields=fields,
                expected=expected,sha256=hashlib.sha256(payload).hexdigest())


def witness_masks(path, capture, directory):
    """Only output words changed by the two named native diagnostic seeds."""
    if directory is None:
        return {}
    nz,ny,nx=capture['shape'];copies=[]
    for seed in (-12345,-23456):
        payload=(Path(directory)/(Path(path).stem+f'.{seed}.bin')).read_bytes()
        values=np.frombuffer(payload,dtype='<f4');offset=0;out={}
        for n in MASS_OUT[:6]:
            out[n]=np.ascontiguousarray(values[offset:offset+nz*ny*nx].reshape(ny,nz,nx).transpose(1,0,2));offset+=nz*ny*nx
        for n in ('pratesh','hbot','htop'):
            out[n]=values[offset:offset+ny*nx].reshape(ny,nx);offset+=ny*nx
        assert offset == len(values)
        copies.append(out)
    masks={}
    for name in copies[0]:
        a,b=copies[0][name],copies[1][name]
        changed=a.view('u4')!=b.view('u4')
        if name not in ('hbot','htop'):
            assert not changed.any() and np.array_equal(a.view('u4'),capture['expected'][name].view('u4')), name
        else:
            # Every defined diagnostic must also agree with the original
            # whole-forecast capture, not merely agree between seeds.
            assert np.array_equal(a.view('u4')[~changed],capture['expected'][name].view('u4')[~changed]),name
            masks[name]=changed
    return masks


def replay(path, witness=None):
    import cupy as cp
    c = read_capture(path)
    excluded = witness_masks(path,c,witness) if c['row']==4 else {}
    f = {n: cp.asarray(a) for n,a in c['fields'].items()}
    row = c['row']
    if row == 3:
        from woof.core.grims import GrimsShallowCumulus
        out = GrimsShallowCumulus(c['znu'], c['shape'][0])(
            qv=f['qv'],temperature=f['t'],pressure=f['p'],exner=f['pi'],
            p_interface=f['p8w'],z_interface=f['z_at_w'],wstar=f['wstar'],
            pblh=f['pblh'],delta=f['delta'],raincv=f['raincv'],kpbl=f['kpbl'],dt=c['dt'])
        actual = dict(rthshten=out.rthshten,rqvshten=out.rqvshten)
    elif row == 4:
        from woof.core.shcu_nscv import launch_shcu_nscv
        hbot=cp.zeros(c['shape'][1:],dtype=cp.float32)
        htop=cp.zeros_like(hbot)
        out = launch_shcu_nscv(**{n:f[n] for n in ('t','qv','qc','qi','p','p8w','pi','rho','u','v','dz8w','w','raincv','xland','hfx','qfx')},
            hpbl=f['pblh'],hbot=hbot,htop=htop,dt=c['dt'],mp_physics=c['mp'],
            pgcon=.55,f_qc=True,f_qi=True)
        actual = {n:out[n] for n in MASS_OUT[:6]}
        actual.update(pratesh=out['pratesh'],hbot=hbot,htop=htop)
    elif row == 1:
        actual = {n:cp.zeros_like(cp.asarray(a)) for n,a in c['expected'].items()}
    else:
        raise ValueError(f'unsupported capture row {row}')
    for n in MASS_OUT:
        if n not in actual:
            actual[n]=cp.zeros(c['shape'],dtype=cp.float32)
    if row==3:
        for n in ('pratesh','hbot','htop'):
            actual[n]=cp.zeros(c['shape'][1:],dtype=cp.float32)
    results={}
    for n,a in actual.items():
        a=cp.asnumpy(a);b=c['expected'][n]
        mask=a.view('u4')!=b.view('u4')
        omit=excluded.get(n,np.zeros(a.shape,dtype=bool))
        mask &= ~omit
        results[n]=dict(words=int(a.size-omit.sum()),excluded_unwritten=int(omit.sum()),differing=int(mask.sum()),
                        active_native=int(np.count_nonzero(b)),
                        max_abs=float(np.max(np.abs(a-b))),
                        examples=[dict(index=[int(v) for v in idx],gpu=float(a[tuple(idx)]),native=float(b[tuple(idx)]),gpu_word=int(a.view('u4')[tuple(idx)]),native_word=int(b.view('u4')[tuple(idx)])) for idx in np.argwhere(mask)[:5]])
    return dict(capture=str(path),sha256=c['sha256'],row=row,step=c['step'],
                shape=c['shape'],outputs=results,
                words=sum(v['words'] for v in results.values()),
                differing=sum(v['differing'] for v in results.values()))


def main():
    p=argparse.ArgumentParser();p.add_argument('captures',nargs='+');p.add_argument('--output',required=True)
    p.add_argument('--witness',type=Path)
    a=p.parse_args()
    r=[replay(Path(path),a.witness) for path in a.captures]
    Path(a.output).write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(dict(captures=len(r),words=sum(v['words'] for v in r),differing=sum(v['differing'] for v in r))))
    return int(any(v['differing'] for v in r))


if __name__=='__main__':
    raise SystemExit(main())
