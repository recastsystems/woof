"""Replay kept stream captures from a strict, whole WRF 4.7.1 real forecast."""
from pathlib import Path
import argparse,json
import numpy as np
import cupy as cp

class Reader:
    def __init__(self,p):
        self.p=p;self.f=open(p,'rb');self.endian='>'
        v=np.frombuffer(self.f.read(4),dtype='>i4')[0];self.f.seek(0)
        if abs(int(v))>100000:self.endian='<'
    def get(self,n,dtype='f4',shape=None):
        x=np.frombuffer(self.f.read(n*4),dtype=self.endian+dtype).astype(dtype)
        assert x.size==n,(self.p,n,x.size)
        return x.reshape(shape) if shape else x
    def three(self,nx,nz,ny):return np.ascontiguousarray(self.get(nx*nz*ny).reshape(ny,nz,nx).transpose(1,0,2))
    def end(self):assert self.f.read()==b'',self.p

def compare(name,a,b,mask=None):
    a=np.ascontiguousarray(a);b=np.ascontiguousarray(cp.asnumpy(b))
    if mask is not None:a=a[...,mask];b=b[...,mask]
    bad=a.view('u4')!=b.view('u4')
    r={'name':name,'words':int(a.size),'different':int(bad.sum()),'max_abs':float(np.max(np.abs(a-b),initial=0))}
    if bad.any():
        ids=np.flatnonzero(bad);r['examples']=[{'index':int(i),'native':float(a.flat[i]),'gpu':float(b.flat[i]),'native_hex':hex(int(a.view('u4').flat[i])),'gpu_hex':hex(int(b.view('u4').flat[i]))} for i in ids[:8]]
    return r

def fog(directory, landuse):
    from woof.core.fog_settling import surface_deposition,settling_tendency,categories_for_landuse
    rows=[]
    assert list(directory.glob("sf.*.bin")) and list(directory.glob("bl.*.bin")), "missing native fog captures"
    for p in sorted(directory.glob('sf.*.bin')):
        r=Reader(p);nx,nz,ny,gs=r.get(4,'i4');dt=r.get(1)[0];nlcat=r.get(1,'i4')[0];n=nx*ny
        qc,rho,dz=[r.get(n,shape=(ny,nx)) for _ in range(3)]
        ivg=r.get(n,'i4',shape=(ny,nx));lai,wspd,fg=[r.get(n,shape=(ny,nx)) for _ in range(3)]
        ref=[r.get(n,shape=(ny,nx)) for _ in range(3)];r.end()
        outs=[cp.zeros((ny,nx),dtype='f4'),cp.asarray(fg),cp.zeros((ny,nx),dtype='f4')]
        surface_deposition(grav_settling=gs,qc1=cp.asarray(qc),rho1=cp.asarray(rho),dz1=cp.asarray(dz),ivgtyp=cp.asarray(ivg),lai=cp.asarray(lai),wspd=cp.asarray(wspd),vdfg=outs[0],fgdp=outs[1],dfgdp=outs[2],dtbl=dt,categories=categories_for_landuse(landuse))
        rows += [compare(p.name+':'+name,a,b) for name,a,b in zip(('vdfg','fgdp','dfgdp'),ref,outs)]
    for p in sorted(directory.glob('bl.*.bin')):
        r=Reader(p);nx,nz,ny,gs=r.get(4,'i4');qc,dz=[r.three(nx,nz,ny) for _ in range(2)];vd=r.get(nx*ny,shape=(ny,nx));dqc=r.three(nx,nz,ny);ref=r.three(nx,nz,ny);r.end()
        out=cp.asarray(dqc);settling_tendency(grav_settling=gs,qc=cp.asarray(qc),dz=cp.asarray(dz),vdfg=cp.asarray(vd),dqc=out)
        rows.append(compare(p.name+':dqc',ref,out))
    return rows

def kf(directory):
    from woof.core.kf import launch_kf,trigger2_perturbations,KFPhaseMode
    tiles=[]
    keys=('temperature','qv','pressure','rho','exner','dz','u','v','w0avg','rqvften')
    rates=('rthcuten','rqvcuten','rqccuten','rqicuten','rqrcuten','rqscuten')
    for p in sorted(directory.glob('kf.*.bin')):
        r=Reader(p);h=r.get(17,'i4');ims,ime,kms,kme,jms,jme,its,ite,kts,kte,jts,jte,ids,ide,jds,jde,tr=h
        dt,dx,cudt=r.get(3);stepcu=r.get(1,'i4')[0];mx,mz,my=ime-ims+1,kme-kms+1,jme-jms+1
        read3=lambda:r.three(mx,mz,my)
        read2=lambda:r.get(mx*my,shape=(my,mx))
        data=dict(zip(keys,[read3() for _ in keys]));nca_in=read2();before=[read3() for _ in rates];after=dict(zip(rates,[read3() for _ in rates]));after.update(zip(('raincv','pratec','nca_seconds','cutop','cubot','shall'),[read2() for _ in range(6)]))
        if tr==2:
            # These native local arrays have tile bounds, unlike the state
            # and tendency dummies above, which carry memory/halo bounds.
            for name in ('tpart_h','tpart_v'):
                a=np.zeros((mz,my,mx),'f4')
                a[kts-kms:kte-kms+1,jts-jms:jte-jms+1,its-ims:ite-ims+1]=r.three(ite-its+1,kte-kts+1,jte-jts+1)
                after[name]=a
        r.end();tiles.append((h,dt,dx,cudt,data,nca_in,after))
    assert tiles,'no native KF captures'
    h,dt,dx,cudt,*_=tiles[0];ims,ime,kms,kme,jms,jme,its,ite,kts,kte,jts,jte,ids,ide,jds,jde,tr=h
    shape=(kte-kts+1,jde-jds,ide-ids);inputs={k:np.empty(shape,'f4') for k in keys};reference={k:np.empty(shape if k in rates or k.startswith('tpart_') else shape[1:],'f4') for k in tiles[0][-1]};nca=np.empty(shape[1:],'f4');covered=np.zeros(shape[1:],bool)
    for h,dt,dx,cudt,data,nca_in,after in tiles:
        ims,ime,kms,kme,jms,jme,its,ite,kts,kte,jts,jte,ids,ide,jds,jde,tr=h
        xs=slice(its-ims,ite-ims+1);ys=slice(jts-jms,jte-jms+1);zs=slice(kts-kms,kte-kms+1);gx=slice(its-ids,ite-ids+1);gy=slice(jts-jds,jte-jds+1)
        for k,a in data.items():inputs[k][:,gy,gx]=a[zs,ys,xs]
        for k,a in after.items():reference[k][(...,gy,gx)]=a[(zs,ys,xs) if a.ndim==3 else (ys,xs)]
        nca[gy,gx]=nca_in[ys,xs];covered[gy,gx]=True
    assert covered.all()
    gpu={k:cp.asarray(v) for k,v in inputs.items()};rows=[]
    if tr==2:
        th,tv=trigger2_perturbations(gpu['temperature'],gpu['rqvften'],gpu['dz'],separate=True)
        rows += [compare('tpart_h',reference['tpart_h'],th),compare('tpart_v',reference['tpart_v'],tv)]
    out=launch_kf(*(gpu[k] for k in ('u','v','temperature','qv','pressure','rho','exner','dz','w0avg')),dx=float(dx),dt=float(dt),cudt_minutes=float(cudt),phase_mode=KFPhaseMode.SEPARATE_ICE_SNOW,trigger=int(tr),rqvften=gpu['rqvften'])
    eligible=nca < np.float32(.5)*dt
    for k in (*rates,'raincv','pratec','cutop','cubot','shall'):
        rows.append(compare(k,reference[k],out[k],eligible))
    # Native preserves the previous NCA when no new convection starts.
    native_nca=cp.where(out['triggered']!=0,out['nca_seconds'],cp.asarray(nca))
    rows.append(compare('nca_seconds',reference['nca_seconds'],native_nca,eligible))
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('captures',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--landuse',required=True,help='captured wrfinput MMINLU');a=p.parse_args()
    rows=fog(a.captures,a.landuse)+kf(a.captures)
    result={'captures':str(a.captures),'rows':rows,'different':sum(x['different'] for x in rows),'words':sum(x['words'] for x in rows)}
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}));raise SystemExit(bool(result['different']))
if __name__=='__main__':main()
