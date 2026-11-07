from pathlib import Path
import json,hashlib,struct,argparse,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import cupy as cp
from woof.core.noah import _F2D,_F3D,launch_noah,load_tables,pack_params
parser=argparse.ArgumentParser(description="Replay every physical output of a real native Noah FASDAS capture.")
parser.add_argument('capture',type=Path);parser.add_argument('--report',type=Path,required=True)
args=parser.parse_args();p=args.capture
f=p.open('rb');assert f.read(8)==b'NOAHFAS1'
nx,ny,step,fasdas,isurban,isice,opt=struct.unpack('<7i',f.read(28));dt,*rest=struct.unpack('<6f',f.read(24));dzs=rest[:4];threshold=rest[4]
frpcpn,monthly,lai=struct.unpack('<3i',f.read(12))
def read(shape,dtype='<f4'):
 size=int(np.prod(shape));v=np.frombuffer(f.read(size*4),dtype=dtype).copy();assert v.size==size;return v.reshape(shape)
def fields():
 a={name:read((ny,nx)) for name in _F2D}
 a.update({name:read((ny,4,nx)).transpose(1,0,2).copy() for name in _F3D});return a
a=fields();a['ivgtyp']=read((ny,nx),'<i4');a['isltyp']=read((ny,nx),'<i4')
rho,cpm,h,q,norm=[read((ny,nx)) for _ in range(5)];b=fields();assert not f.read();f.close()
dev={k:cp.asarray(v) for k,v in a.items()};dev['ebal']=cp.zeros((ny,nx),cp.int32)
heat=((h*rho)*cpm)*a['dz8w1'];moisture=(q*rho)*a['dz8w1']
params=pack_params(load_tables())
launch_noah(dev,params,dt,dzs,isurban=isurban,isice=isice,xice_threshold=threshold,frpcpn=bool(frpcpn),usemonalb=bool(monthly),rdlai2d=bool(lai),opt_thcnd=opt,itimestep=step,fasdas={k:cp.asarray(v) for k,v in [('heat',heat),('moisture',moisture),('qnorm',norm)]})
cp.cuda.get_current_stream().synchronize()
rows={};words=0;diffs=0
for name in (*_F2D,*_F3D):
 if name=='reslin':continue
 got=dev[name].get();mask=got.view(np.uint32)!=b[name].view(np.uint32);n=int(mask.sum());words+=got.size;diffs+=n
 rows[name]={'words':got.size,'differing_words':n,'max_abs':float(np.nanmax(np.abs(got-b[name]))),'first':np.argwhere(mask)[:5].tolist()}
 if n:
  ids=np.flatnonzero(mask)[:5];rows[name]['native']=b[name].ravel()[ids].tolist();rows[name]['cuda']=got.ravel()[ids].tolist()
  rows[name]['differing_words_exact']=[{'index':idx.tolist(),'native':hex(int(b[name].view('u4')[tuple(idx)])),'cuda':hex(int(got.view('u4')[tuple(idx)]))} for idx in np.argwhere(mask)]
report=dict(words=words,differing_words=diffs,fields=rows,shape=[ny,nx],step=step,fasdas=fasdas,nonzero_sda_hfx=int(np.count_nonzero(h)),nonzero_qnorm=int(np.count_nonzero(norm)),capture_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
args.report.write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
assert diffs==0
