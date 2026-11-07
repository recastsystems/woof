#!/usr/bin/env python3
"""Replay pinned native cal_helicity on a real WRF restart snapshot.

The wrapper changes no arithmetic. It only pads the WRF memory bounds,
supplies the public model constants/configuration type, and exposes C ABI.
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('restart', type=Path)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from netCDF4 import Dataset
    import cupy as cp
    from woof.wrf_exact import ENABLED
    from woof.core.uh_diag import device_uh_step
    if not ENABLED:
        raise ValueError('requires GPUWM_WRF_EXACT=1')
    manifest = json.loads((HERE/'sources.json').read_text())
    pin = next(r for r in manifest['sources'] if r['path'] == 'dyn_em/module_diffusion_em.F')
    source = (HERE/'upstream/module_diffusion_em.F').read_bytes()
    assert hashlib.sha256(source).hexdigest() == pin['sha256']
    regions = [re.search(rb'^    SUBROUTINE '+name+rb'[ (].*?^    END SUBROUTINE '+name+rb'[^\n]*\n', source, re.M|re.S).group()
               for name in (b'compute_diff_metrics', b'cal_helicity')]
    module = '''module replay_mod
use iso_c_binding
implicit none
real, parameter :: g=9.81
type grid_config_rec_type
logical :: open_xs=.false., open_xe=.false., open_ys=.false., open_ye=.false.
logical :: specified=.true., nested=.false., periodic_x=.false., periodic_y=.false.
end type
contains
'''.encode()+b'\n'.join(regions)+b'''
subroutine replay(nx,ny,nz,u,v,w,ph,phb,mfu,mfv,ht,dn,dnw,fnm,fnp,cf1,cf2,cf3,rdx,rdy,up) bind(C)
integer(c_int),value :: nx,ny,nz
real(c_float),intent(in) :: u(0:nx+1,1:nz+1,0:ny+1),v(0:nx+1,1:nz+1,0:ny+1)
real(c_float),intent(in) :: w(0:nx+1,1:nz+1,0:ny+1),ph(0:nx+1,1:nz+1,0:ny+1),phb(0:nx+1,1:nz+1,0:ny+1)
real(c_float),intent(in) :: mfu(0:nx+1,0:ny+1),mfv(0:nx+1,0:ny+1),ht(0:nx+1,0:ny+1)
real(c_float),intent(in) :: dn(nz+1),dnw(nz+1),fnm(nz+1),fnp(nz+1)
real(c_float),value :: cf1,cf2,cf3,rdx,rdy
real(c_float),intent(inout) :: up(0:nx+1,0:ny+1)
real :: z(0:nx+1,1:nz+1,0:ny+1),rdz(0:nx+1,1:nz+1,0:ny+1),rdzw(0:nx+1,1:nz+1,0:ny+1)
real :: zx(0:nx+1,1:nz+1,0:ny+1),zy(0:nx+1,1:nz+1,0:ny+1),uh(0:nx+1,0:ny+1)
type(grid_config_rec_type) :: flags
z=0.;rdz=0.;rdzw=0.;zx=0.;zy=0.;uh=0.
call compute_diff_metrics(flags,ph,phb,z,rdz,rdzw,zx,zy,rdx,rdy, &
 1,nx+1,1,ny+1,1,nz+1,0,nx+1,0,ny+1,1,nz+1,1,nx,1,ny,1,nz+1)
call cal_helicity(flags,u,v,w,uh,up,ph,phb,mfu,mfu,mfv,mfv,ht, &
 rdx,rdy,dn,dnw,rdz,rdzw,fnm,fnp,cf1,cf2,cf3,zx,zy, &
 1,nx+1,1,ny+1,1,nz+1,0,nx+1,0,ny+1,1,nz+1,1,nx,1,ny,1,nz+1)
end subroutine
end module
'''
    args.build.mkdir(parents=True, exist_ok=True)
    generated = args.build/'uh471.f90'
    generated.write_bytes(module)
    library = args.build/'libuh471.so'
    command = ['gfortran', '-O0', '-fno-fast-math', '-ffp-contract=off', '-fPIC', '-shared',
               '-ffree-line-length-none', '-frecursive', '-o', str(library), str(generated)]
    subprocess.run(command, check=True, cwd=args.build)
    with Dataset(args.restart) as ds:
        def read(name):
            return np.ascontiguousarray(np.asarray(ds.variables[name][0], dtype='f4'))
        data = {k: read(n) for k,n in dict(u='U_2',v='V_2',w='W_2',ph='PH_2',phb='PHB',
               msfu='MAPFAC_UX',msfv='MAPFAC_VY',ht='HGT',dn='DN',dnw='DNW',fnm='FNM',fnp='FNP').items()}
        scalars = [np.float32(read(n).item()) for n in ('CF1','CF2','CF3')]
        scalars += [np.float32(1)/np.float32(ds.DX),np.float32(1)/np.float32(ds.DY)]
    nzf,ny,nx = data['w'].shape
    nz = nzf-1
    def pad(value):
        if value.ndim == 1:
            out = np.zeros(nz+1, dtype='f4'); out[:value.size] = value
        elif value.ndim == 2:
            y,x = value.shape
            out = np.zeros((nx+2,ny+2), dtype='f4', order='F'); out[1:x+1,1:y+1] = value.T
        else:
            z,y,x = value.shape
            out = np.zeros((nx+2,nz+1,ny+2), dtype='f4', order='F'); out[1:x+1,:z,1:y+1] = value.transpose(2,0,1)
        return out
    order = ('u','v','w','ph','phb','msfu','msfv','ht','dn','dnw','fnm','fnp')
    padded = [pad(data[k]) for k in order]
    native = np.zeros((nx+2,ny+2), dtype='f4', order='F')
    fun = ctypes.CDLL(str(library)).replay
    fun.argtypes = [ctypes.c_int]*3+[ctypes.c_void_p]*12+[ctypes.c_float]*5+[ctypes.c_void_p]
    fun(nx,ny,nz,*(v.ctypes.data for v in padded),*scalars,native.ctypes.data)
    want = np.ascontiguousarray(native[1:nx+1,1:ny+1].T)
    device = {k: cp.asarray(v) for k,v in data.items()}
    outputs = [cp.zeros((ny,nx), dtype='f4') for _ in range(3)]
    device_uh_step(*(device[k] for k in order),*scalars,*outputs)
    got = cp.asnumpy(outputs[-1])
    different = got.view('u4') != want.view('u4')
    result = dict(kind='strict real-native-restart cal_helicity module replay',
                  input=str(args.restart), input_sha256=hashlib.sha256(args.restart.read_bytes()).hexdigest(),
                  input_fields={k:dict(shape=v.shape,sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k,v in data.items()},
                  region_sha256=[hashlib.sha256(r).hexdigest() for r in regions], compiler_command=command,
                  words=int(want.size),differing_words=int(different.sum()),nonzero_words=int(np.count_nonzero(want)),
                  native_min=float(want.min()),native_max=float(want.max()))
    if different.any():
        ix=tuple(map(int,np.argwhere(different)[0]))
        result['first_difference']=dict(index=ix,native=float(want[ix]),cuda=float(got[ix]),
                                       native_word=int(want.view('u4')[ix]),cuda_word=int(got.view('u4')[ix]))
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('words','differing_words','nonzero_words','native_max')}))
    if different.any():
        raise SystemExit(1)


if __name__ == '__main__':
    main()
