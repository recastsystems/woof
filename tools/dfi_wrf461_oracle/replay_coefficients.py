"""Replay original DFCOEF with time-window arguments from a native capture.

Weather fields do not enter DFCOEF. All eight defined filters use the actual
captured solve count, signed dt and cutoff. The selected native filter's
captured coefficients additionally bind the extraction to the live run.
"""
from pathlib import Path
import argparse,ctypes as C,hashlib,json,subprocess
from collections import defaultdict
import numpy as np
from replay_capture import read_capture
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--capture',type=Path,required=True);p.add_argument('--namelist',type=Path,required=True)
 p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 from woof.namelist_import import parse_namelist
 cfg=parse_namelist(a.namelist);dt=float(cfg['domains']['time_step'][0]);cutoff=float(cfg['dfi_control']['dfi_cutoff_seconds'][0]);selected=int(cfg['dfi_control']['dfi_nfilter'][0])
 pin=json.loads((HERE/'SOURCE.json').read_text())['files']['share/dfi.F']
 source=a.source.read_bytes();assert hashlib.sha256(source).hexdigest()==pin
 text=source.decode();extracted=a.output/'dfcoef.F90';extracted.write_text(text[text.index('   SUBROUTINE dfcoef ('):text.index('    SUBROUTINE optfil (')])
 captures=sorted(a.capture.glob('lane-dfi-*.bin'));groups=defaultdict(dict)
 for path in captures:
  sample=read_capture(path,np);h=sample['header']
  if int(h[5])==0:groups[tuple(int(h[i]) for i in (1,2,3))][int(h[4])]=sample['coefficient']
 assert groups
 receipt=dict(scope='all defined DFCOEF filters replayed with real native captured window arguments',real_forecast_capture=True,source_sha256=pin,namelist_sha256=hashlib.sha256(a.namelist.read_bytes()).hexdigest(),capture_files=[dict(name=x.name,sha256=hashlib.sha256(x.read_bytes()).hexdigest()) for x in captures],cases=[],words=0,differing_words=0)
 def run(argv):
  r=subprocess.run(list(map(str,argv)),capture_output=True,text=True,cwd=a.output)
  with (a.output/'commands.log').open('a') as f:f.write('$ '+' '.join(map(str,argv))+'\n'+r.stdout+r.stderr)
  assert r.returncode==0,r.stderr
 flags=['-cpp','-DWRFPLUS=0','-ffree-form','-ffree-line-length-none','-fno-fast-math','-ffp-contract=off','-fno-tree-vectorize']
 for arm in ('o0','o2'):run(['gfortran',*flags,'-'+arm.upper(),extracted.resolve(),HERE/'driver.F90','-o',(a.output/arm).resolve()])
 libpath=(a.output/'libarwen_dfi.so').resolve();run(['rustc','--edition=2021','--crate-type=cdylib','-C','opt-level=3',ROOT/'tools/dfi-runtime/src/lib.rs','-o',libpath])
 f=C.CDLL(str(libpath)).aw_dfi_weights;f.argtypes=[C.c_int,C.c_float,C.c_float,C.c_int,C.c_void_p,C.c_size_t,C.POINTER(C.c_size_t)];f.restype=C.c_int
 for key,sequence in groups.items():
  assert key[0]==1,'this replay requires the captured single-domain namelist'
  count=max(sequence)+1;assert set(sequence)==set(range(count))
  signed=-dt if key[2]==1 else dt
  for nf in range(8):
   native=[]
   for arm in ('o0','o2'):
    out=(a.output/f'{key[1]}-{key[2]}-{nf}-{arm}.bin').resolve()
    run([(a.output/arm).resolve(),(count+1)//2-1,nf,signed,cutoff,out])
    h=np.fromfile(out,np.float32);native.append(np.concatenate((h,h[:-1][::-1] if count%2 else h[::-1])))
   assert native[0].tobytes()==native[1].tobytes()
   if nf==selected:assert native[0].tobytes()==np.asarray([sequence[i] for i in range(count)],np.float32).tobytes()
   rust=np.empty(count,np.float32);written=C.c_size_t();status=f(int((count-1)*dt),signed,cutoff,nf,rust.ctypes.data,count,C.byref(written));assert status==0 and written.value==count
   differing=int(np.count_nonzero(native[0].view(np.uint32)!=rust.view(np.uint32)))
   receipt['cases'].append(dict(domain=key[0],option=key[1],stage=key[2],filter=nf,dt=signed,cutoff=cutoff,words=count,differing_words=differing,native_sha256=hashlib.sha256(native[0].tobytes()).hexdigest()))
   receipt['words']+=count;receipt['differing_words']+=differing
 (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k not in ('capture_files','cases')},indent=2))
 return int(receipt['differing_words']!=0)
if __name__=='__main__':raise SystemExit(main())
