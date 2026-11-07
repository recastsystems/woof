#!/usr/bin/env python3
"""Replay native BouLac capture files and write a word-count receipt."""
from __future__ import annotations
import argparse,hashlib,json,platform
from pathlib import Path
import numpy as np

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('captures',type=Path)
    parser.add_argument('--receipt',type=Path,required=True)
    args=parser.parse_args()
    import cupy as cp
    from woof import wrf_exact
    from woof.core.boulac import launch_boulac
    from woof.verify.boulac_oracle import read_inputs,read_record,BEP_LEVEL_INPUTS
    if not wrf_exact.ENABLED: parser.error('GPUWM_WRF_EXACT=1 is required')
    rows=[]
    for ipath in sorted(args.captures.glob('*.in.bin')):
        rpath=ipath.with_name(ipath.name.replace('.in.bin','.rec.bin'))
        inp,rec=read_inputs(ipath),read_record(rpath)
        nz,nc=inp['nz'],inp['ncol']
        def d(a):return cp.asarray(np.ascontiguousarray(a.reshape(nz,1,nc)))
        def s(a):return cp.asarray(np.ascontiguousarray(a.reshape(1,nc)))
        tke=d(inp['tke'])
        bep=({k:d(inp[k]) for k in BEP_LEVEL_INPUTS} if inp['flag_bep'] else None)
        if bep is not None: bep['sf_bep']=cp.asarray(np.ascontiguousarray(inp['sf_bep'].reshape(nz+1,1,nc)))
        row={'input':ipath.name,'input_sha256':hashlib.sha256(ipath.read_bytes()).hexdigest(),
             'record_sha256':hashlib.sha256(rpath.read_bytes()).hexdigest(),'nz':nz,'columns':nc,
             'compared_words':0,'differing_words':0,'native_undefined_words':0,'fields':{}}
        for step in rec['steps']:
            si=step['in']
            columns={k:d(si[k]) for k in ('u','v','th','qv','qc')}
            columns.update(dz=d(inp['dz8w']),rho=d(inp['rho']))
            surface={k:s(inp[k]) for k in ('hfx','qfx','ust')}
            out=launch_boulac(columns,surface,tke,dt=float(inp['dt']),cp_air=float(inp['cp']),g=float(inp['g']),
                outputs={'pblh':s(si['pblh'])},bep=bep,frc_urb2d=s(inp['frc_urb2d']) if bep is not None else None,idiff=int(inp['idiff']))
            got={k:cp.asnumpy(v).reshape(step['out'][k].shape) for k,v in out.items() if k in step['out']}
            got['tke']=cp.asnumpy(tke).reshape(nz,nc)
            if bep is not None:got['dlg_bep']=cp.asnumpy(bep['dlg_bep']).reshape(nz,nc)
            calm=(inp['ust']==0)&(si['u'][0]==0)&(si['v'][0]==0)&(not inp['flag_bep'])
            for key,a in got.items():
                b=step['out'][key]
                undefined=np.isnan(b)&(calm[None,:] if a.ndim==2 and key in ('rublten','rvblten','wu','wv') else False)
                bad=(a.view(np.uint32)!=b.view(np.uint32))&~undefined
                result={'words':int(a.size-undefined.sum()),'different':int(bad.sum()),'native_undefined':int(undefined.sum())}
                if bad.any():
                    where=tuple(int(v) for v in np.argwhere(bad)[0])
                    result['first']={'index':where,'gpu_word':int(a.view(np.uint32)[where]),'native_word':int(b.view(np.uint32)[where]),'gpu_value':float(a[where]),'native_value':float(b[where])}
                row['fields'][key]=result
                row['compared_words']+=result['words'];row['differing_words']+=result['different'];row['native_undefined_words']+=result['native_undefined']
        rows.append(row)
    if not rows:parser.error('no captured inputs found')
    result={'schema':'boulac-native-wrf471-replay-v1','python':platform.python_version(),'cupy':cp.__version__,
            'gpu':cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)['name'].decode(),
            'compared_words':sum(r['compared_words'] for r in rows),'differing_words':sum(r['differing_words'] for r in rows),
            'native_undefined_words':sum(r['native_undefined_words'] for r in rows),'captures':rows,'compile':wrf_exact.compile_receipt()}
    args.receipt.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('captures','compile')},indent=2))
    return int(result['differing_words']!=0)
if __name__=='__main__':raise SystemExit(main())
