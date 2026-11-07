"""Replay native module_em:trajectory captures, outside the forecast loop."""
from pathlib import Path
import sys, json, os, re
import numpy as np
import cupy as cp
from woof.core.kernels import get_kernel
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tracer_wrf461_oracle'))
from replay import Stream,compare,digest


def replay(path):
    stream=Stream(path)
    h=stream.read((21,),'i4')
    version,domain,count,*bounds=map(int,h)
    ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte=bounds
    dt,rdx,rdy=stream.read((3,))
    # Native capture carries Registry's fixed capacity, independent of count.
    capacity=1000
    coords=stream.read((5,capacity))
    shape=(jme-jms+1,kme-kms+1,ime-ims+1)
    three=[stream.read(shape) for _ in range(3)]
    two=[stream.read((shape[0],shape[2])) for _ in range(6)]
    vertical=[stream.read((shape[1],)) for _ in range(5)]
    code=int(stream.read((1,),'i4')[0]);proj=stream.read((14,));stream.finish()
    nx,ny,nz=ide-ids,jde-jds,kde-kds
    def dev(a,stagger=''):
        sy,sx,sz=int(stagger=='v'),int(stagger=='u'),int(stagger=='w')
        if a.ndim==3:
            a=a[jds-jms:jds-jms+ny+sy,kds-kms:kds-kms+nz+sz,ids-ims:ids-ims+nx+sx].transpose(1,0,2)
        else:a=a[jds-jms:jds-jms+ny+sy,ids-ims:ids-ims+nx+sx]
        return cp.asarray(np.ascontiguousarray(a))
    fields=[dev(a,s) for a,s in zip(three,('u','v','w'))]
    fields += [dev(a,s) for a,s in zip(two,('','u','v','','u','v'))]
    vert=[cp.asarray(a[kds-kms:kds-kms+nz+1]) for a in vertical]
    flags=np.ones(count,np.int32)
    spec=(path.parent/f'wrfinput_traj_d{domain:02d}').read_text()
    for m in re.finditer(r'traj_type\((\d+)\)%is_stationary\s*=\s*\.(true|false)\.',spec,re.I):
        flags[int(m[1])-1]=int(m[2].lower()=='false')
    devicecoords=cp.asarray(coords)
    get_kernel('trajectory','trajectory_advance')(((count+127)//128,),(128,),
        (devicecoords,np.int32(capacity),np.int32(count),cp.asarray(flags),np.int32(code),cp.asarray(proj),
         dt,rdx,rdy,*fields,*vert,np.int32(nx),np.int32(ny),np.int32(nz)))
    output=path.with_name(path.name.replace('.in.bin','.out.bin'))
    ref=np.fromfile(output,dtype=stream.endian+'f4').astype(np.float32).reshape(coords.shape)
    got=cp.asnumpy(devicecoords)
    row=compare(got,ref)
    row['exact_words']=[dict(index=i.tolist(),native=hex(int(ref.view('u4')[tuple(i)])),cuda=hex(int(got.view('u4')[tuple(i)]))) for i in np.argwhere(got.view('u4')!=ref.view('u4'))]
    row.update(domain=domain,count=count,capture=path.name,coordinates=got[:,:count].tolist(),
               reference=ref[:,:count].tolist(),input=coords[:,:count].tolist())
    return row

if __name__=='__main__':
    if os.environ.get('GPUWM_WRF_EXACT')!='1':raise ValueError('strict proof requires GPUWM_WRF_EXACT=1')
    root=Path(sys.argv[1]);rows=[replay(p) for p in sorted(root.glob('trajectory-[0-9][0-9].in.bin'))]
    receipt=dict(rows=rows,words=sum(r['words'] for r in rows),differing_words=sum(r['differing_words'] for r in rows),
                 captures={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(root.glob('trajectory-*.bin'))})
    Path(sys.argv[2]).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
    if receipt['differing_words']:raise SystemExit(1)
