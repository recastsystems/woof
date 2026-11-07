"""Replay unformatted FER_HIRES_ADVECT calls captured inside pinned real WRF."""
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np


def read_capture(path):
    raw=path.read_bytes()
    ints=np.frombuffer(raw,dtype='<i4',count=9)
    magic,its,ite,jts,jte,kts,kte,step,gid=map(int,ints)
    if magic != 150471: raise ValueError(f'{path}: not an MP15 real capture')
    dt,dx,dy=map(float,np.frombuffer(raw,dtype='<f4',offset=36,count=3))
    nx,ny,nz=ite-its+1,jte-jts+1,kte-kts+1
    pos=48
    def take(surface=False):
        nonlocal pos
        count=nx*ny*(1 if surface else nz)
        a=np.frombuffer(raw,dtype='<f4',offset=pos,count=count);pos+=count*4
        return (a.reshape(ny,nx) if surface else a.reshape(ny,nz,nx).transpose(1,0,2)).copy(order='C')
    before={n:take() for n in ('th','qv','qc','qr','qi','qrimef','rho','p','pii','dz')}
    before.update({n:take(True) for n in ('rainnc','rainncv')})
    after={n:take() for n in ('th','qv','qc','qr','qi','qrimef')}
    after.update({n:take(True) for n in ('rainnc','rainncv','sr')})
    if pos != len(raw):raise ValueError(f'{path}: consumed {pos} of {len(raw)} bytes')
    return before,after,dict(file=path.name,sha256=hashlib.sha256(raw).hexdigest(),
        step=step,gid=gid,dt=dt,dx=dx,dy=dy,shape=[nz,ny,nx],bounds=list(map(int,ints[1:7])))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('capture',type=Path,nargs='+');ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    import cupy as cp
    from woof.core.fer_hires import device_tables, launch_fer_hires
    from woof.wrf_exact import ENABLED,compile_receipt
    if not ENABLED:raise ValueError('real replay requires GPUWM_WRF_EXACT=1 before import')
    records=[];total=0;different=0
    for path in sorted(args.capture):
        before,after,meta=read_capture(path)
        arrays={n:cp.asarray(a) for n,a in before.items()};shape=meta['shape']
        unused=cp.empty(tuple(shape),dtype=cp.float32);sr=cp.empty(tuple(shape[1:]),dtype=cp.float32)
        launch_fer_hires(arrays['th'],arrays['qv'],arrays['qc'],arrays['qr'],arrays['qi'],
            unused,unused,unused,unused,arrays['p'],arrays['pii'],arrays['rho'],arrays['dz'],
            arrays['rainnc'],arrays['rainncv'],sr,device_tables(meta['dt']),meta['dt'],meta['gid'],qrimef=arrays['qrimef'])
        arrays['sr']=sr
        comparisons={}
        for name,expected in after.items():
            actual=arrays[name].get();mask=actual.view('u4')!=expected.view('u4');count=int(mask.sum())
            total+=expected.size;different+=count
            row={'words':int(expected.size),'different':count,'native_nonfinite':int((~np.isfinite(expected)).sum()),'gpu_nonfinite':int((~np.isfinite(actual)).sum())}
            if count:
                idx=tuple(map(int,np.argwhere(mask)[0]));row['first']={'index':idx,'native_hex':hex(int(expected.view('u4')[idx])),'gpu_hex':hex(int(actual.view('u4')[idx])),'native':float(expected[idx]),'gpu':float(actual[idx])}
            comparisons[name]=row
        meta['fields']=comparisons
        meta['input_ice_cells']=int((before['qi']>1e-12).sum());meta['input_rain_cells']=int((before['qr']>1e-12).sum())
        records.append(meta);print(meta['file'],sum(x['different'] for x in comparisons.values()),flush=True)
    result={'schema':'gpuwm-ferrier-real-replay-v1','words':int(total),'different_words':different,'strict':ENABLED,'native_corrections':['omit inactive optional MP15 fractions at init','remove unused legacy arrays from MP15 driver guard; require QR_CURR','pass QRIMEF_CURR and F_QRIMEF from solve_em to the driver'],'captures':records,'compile':compile_receipt()}
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('words',total,'different',different)
    return int(different!=0)
if __name__=='__main__':sys.exit(main())
