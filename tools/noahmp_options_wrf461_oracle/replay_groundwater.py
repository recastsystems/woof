"""Bitwise spatial WTABLE replay, including captured native neighbor halos."""
from pathlib import Path
import argparse,csv,json,hashlib,os
import numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--captures',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args()
assert os.environ.get('GPUWM_WRF_EXACT')=='1'
import cupy as cp
from woof.core.noahmp_groundwater_slab import evaluate_rows,_parameters
from woof.core.noahmp_runtime import NoahmpRuntimeParameters
paths=sorted(a.captures.glob('groundwater-real-*.csv'));assert paths
reports=[];total=0;diffs=[]
for path in paths:
 rows=list(csv.DictReader(path.open()));scalars={r['field']:int(r['hex'],16)for r in rows if r['i']=='0' and r['j']=='0' and r['k']=='0'}
 def scalar(k,integer=True):return np.array([scalars[k]],np.uint32).view(np.int32 if integer else np.float32)[0].item()
 ims,ime,jms,jme=[scalar(k)for k in ('ims','ime','jms','jme')];nx=ime-ims+1;ny=jme-jms+1;shape=(ny,nx)
 floats={};outputs={};integers={'ivgtyp','isltyp'}
 for r in rows:
  if int(r['i'])==0 and int(r['j'])==0:continue
  dest=floats if r['stage']=='input'else outputs;name=r['field'];k=int(r['k']);key=(name,k)
  if key not in dest:dest[key]=np.zeros(shape,np.uint32)
  dest[key][int(r['j'])-jms,int(r['i'])-ims]=int(r['hex'],16)
 def field(name,k=0):return floats[(name,k)].view(np.int32 if name in integers else np.float32)
 def words(name):return floats[(name,0)].view(np.float32)
 Y=np.zeros((ny,nx,10),np.float32)
 for slot,name in enumerate(('wtd','smcwtd','deeprech','qslat','qrfs','qsprings','rech')):Y[:,:,slot]=words(name)
 # QSPRING is zeroed by native GROUNDWATER_INIT and only assigned on land by UPDATEWTD.
 # Captured formal outputs on all nonland cells independently verify those initial words.
 assert np.all(outputs[('qspring',0)][field('xland')>=1.5]==0)
 S=np.stack([words(n)for n in ('fdepth','area','topo','eqwtd','pexp','riverbed','rivercond')]+[np.zeros(shape,np.float32),words('xland'),words('xice')],axis=-1)
 Z=np.stack([field(n,k)for n in ('smois','sh2oxy','smoiseq')for k in range(1,5)],axis=-1)
 f={n:cp.asarray(field(n))for n in ('ivgtyp','isltyp')};f['zwtxy']=cp.asarray(words('wtd'))
 params=NoahmpRuntimeParameters(opt_run=5)
 # Native WTABLE uses the single ISURBAN table category, while cold initialization also recognizes LCZ.
 P=_parameters(f,params);soil=cp.ascontiguousarray(f['isltyp'].reshape(-1));land=((words('xland')-np.float32(1.5)<0)&(words('xice')<np.float32(scalar('xice_threshold',False)))&(field('ivgtyp')!=scalar('isice')))
 ii,jj=np.meshgrid(np.arange(ims,ime+1),np.arange(jms,jme+1));flow=land&(ii>scalar('ids'))&(ii<scalar('ide')-1)&(jj>scalar('jds'))&(jj<scalar('jde')-1)
 dzs=[np.array([int(r['hex'],16)],np.uint32).view(np.float32)[0]for r in rows if r['field']=='dzs']
 s,y,z=evaluate_rows(P,cp.asarray(S.reshape(-1,10)),cp.asarray(Y.reshape(-1,10)),cp.asarray(Z.reshape(-1,12)),soil,cp.asarray(land.astype(np.int32).reshape(-1)),dzs,shape,np.float32(scalar('wtddt',False))*np.float32(60.),flow_land=cp.asarray(flow.astype(np.int32).reshape(-1)))
 yy=cp.asnumpy(y).reshape(ny,nx,10);zz=cp.asnumpy(z).reshape(ny,nx,12)
 owned=(ii>=scalar('its'))&(ii<=scalar('ite'))&(jj>=scalar('jts'))&(jj<=scalar('jte'))
 count=0;bad=0;by_field={}
 for name,k in outputs:
  if name in ('smois','sh2oxy'):got=zz[:,:,k-1+(4 if name=='sh2oxy'else 0)]
  else:got=yy[:,:,(('wtd','smcwtd','deeprech','qslat','qrfs','qsprings','rech','qlat','qrf','qspring').index(name))]
  got=np.ascontiguousarray(got).view(np.uint32);wanted=outputs[(name,k)];mask=owned;errors=(got!=wanted)&mask;num=int(errors.sum());by_field[f'{name}:{k}']=dict(compared=int(mask.sum()),differing=num);count+=int(mask.sum());bad+=num
  for j,i in np.argwhere(errors)[:12]:diffs.append(dict(capture=path.name,field=name,k=k,i=int(i+ims),j=int(j+jms),got=f'{got[j,i]:08x}',native=f'{wanted[j,i]:08x}'))
 reports.append(dict(capture=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bounds={k:scalar(k)for k in ('ids','ide','jds','jde','ims','ime','jms','jme','its','ite','jts','jte')},compared_words=count,differing_words=bad,fields=by_field));total+=count
result=dict(origin='real native WRF v4.6.1 WTABLE captures, including exact native neighbor halos',strict=True,compared_words=total,differing_words=sum(r['differing_words']for r in reports),formal_output_fields=12,native_undefined=[],captures=reports,differences=diffs)
a.receipt.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k not in ('captures','differences')},indent=2));print(json.dumps(diffs[:20],indent=2));raise SystemExit(bool(result['differing_words']))
