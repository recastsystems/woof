"""Replay kept WRF 4.7.1 real-input driver captures, comparing raw REAL32 words."""
from pathlib import Path
import argparse,json
import numpy as np
p=argparse.ArgumentParser();p.add_argument('case',type=Path);p.add_argument('--layout',type=Path,required=True);p.add_argument('--tables',type=Path,required=True);p.add_argument('--dataset',default='MODIFIED_IGBP_MODIS_NOAH');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
import cupy as cp
from woof.core.noah import launch_noah,launch_noah_seaice,load_tables,pack_params
params=pack_params(load_tables(mminlu=a.dataset,tbl_dir=a.tables))
rows=[]
def read(path,kind):
 layout=json.loads((a.layout/(kind+'-layout.json')).read_text());raw=path.read_bytes();offset=0
 def take(dtype,count):
  nonlocal offset
  result=np.frombuffer(raw,dtype=dtype,count=count,offset=offset).copy();offset+=result.nbytes;return result
 ints=take('<i4',11 if kind=='noah' else 7);nx,ny,ns=map(int,ints[:3]);floats=take('<f4',ns+2 if kind=='noah' else 6);dev={}
 if kind=='noah':
  for name in ('ivgtyp','isltyp'):dev[name]=take('<i4',nx*ny).reshape(ny,nx)
 for name in layout['f2']:dev[name]=take('<f4',nx*ny).reshape(ny,nx)
 for name in layout['f3']:dev[name]=np.ascontiguousarray(take('<f4',nx*ny*ns).reshape(ny,ns,nx).transpose(1,0,2))
 for name in layout.get('extras',[]):dev[name]=take('<f4',nx*ny).reshape(ny,nx)
 assert offset==len(raw),(path,offset,len(raw))
 return ints,floats,dev
for path in sorted((a.case/'captures').glob('*-in.bin')):
 kind=path.name.split('-')[0]
 if kind not in ('noah','seaice'):continue
 h,s,host=read(path,kind);_,_,ref=read(path.with_name(path.name.replace('-in.bin','-out.bin')),kind)
 dev={n:cp.asarray(v) for n,v in host.items()};dev['ebal']=cp.zeros_like(dev['tsk'],dtype=cp.int32)
 if kind=='noah':
  launch_noah(dev,params,float(s[0]),s[2:],isurban=int(h[4]),isice=int(h[5]),
    xice_threshold=float(s[1]),itimestep=int(h[3]),frpcpn=bool(h[6]),
    usemonalb=bool(h[7]),rdlai2d=bool(h[8]),opt_thcnd=int(h[9]),ua_phys=bool(h[10]),
    irrigation_channel=dev['irrigation_channel'] if np.any(host['irrigation_channel']) else None)
 else:
  launch_noah_seaice(dev,float(s[0]),frpcpn=bool(h[3]),xice_threshold=float(s[1]),
    albedo_default=float(s[2]),thickness_default=float(s[3]),snowdepth_min=float(s[4]),snowdepth_max=float(s[5]),
    albedo_opt=int(h[4]),thickness_opt=int(h[5]),snowdepth_opt=int(h[6]),check_snow_density=True)
 cp.cuda.get_current_stream().synchronize()
 result={};words=diff=0
 for n,expected in ref.items():
  # RESLIN is an engine-only diagnostic; capture fills a named zero plane.
  if n=='reslin' or kind=='noah' and not h[10] and n in ('flx4','fvb','fbur','fgsn'):continue
  actual=cp.asnumpy(dev[n]);left=actual.view(np.uint32);right=expected.view(np.uint32)
  count=int((left!=right).sum());words+=left.size;diff+=count
  if count:
   positions=np.argwhere(left!=right);pos=tuple(positions[0]);result[n]={'differing':count,'first_index':list(map(int,pos)),'gpu':float(actual[pos]),'native':float(expected[pos]),'gpu_word':hex(int(left[pos])),'native_word':hex(int(right[pos]))}
 row={'capture':path.name,'words':words,'differing':diff,'fields':result};rows.append(row);print(json.dumps(row),flush=True)
a.out.write_text(json.dumps({'words':sum(x['words'] for x in rows),'differing':sum(x['differing'] for x in rows),'captures':rows},indent=2)+'\n')
