"""Control device optimization when localizing a production-only divergence."""
import argparse,json,sys,re
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from woof.core import ntu
from run_gpu import main as run
from compare import read,OUT,INIT,S2
import numpy as np

def main():
 ap=argparse.ArgumentParser();ap.add_argument('input',type=Path);ap.add_argument('reference',type=Path);ap.add_argument('output',type=Path);ap.add_argument('--option',action='append',default=[]);ap.add_argument('--barrier-unit');ap.add_argument('--word-unit');ap.add_argument('--no-unroll-unit');a=ap.parse_args()
 if a.barrier_unit:
  original_source=ntu.module_source
  def altered(*args,**kw):
   source=original_source(*args,**kw);found=re.search(r'NTU_DEV_BIG void ntu_'+re.escape(a.barrier_unit)+r'\([^;]+\) \{',source);assert found is not None
   start=found.end();end=source.index('\n}\n',start);body=source[start:end];body=re.sub(r'(^[ \t]+[^\n;]+=[^\n;]+;)$',r'\1\n    asm volatile ("" ::: "memory");',body,flags=re.M);return source[:start]+body+source[end:]
  ntu.module_source=altered
 if a.word_unit:
  original_source=ntu.module_source
  def altered_word(*args,**kw):
   source=original_source(*args,**kw);found=re.search(r'NTU_DEV_BIG void ntu_'+re.escape(a.word_unit)+r'\([^;]+\) \{',source);assert found is not None
   start=found.end();end=source.index('\n}\n',start);body=source[start:end]
   def word(m):
    left=m.group(1).strip()
    if left.startswith(('float ','double ','int ','bool ','const ')):return m.group(0)
    return m.group(0)+'\n    '+left+' = ntu_codegen_word('+left+');'
   body=re.sub(r'^[ \t]+([A-Z][A-Z0-9]*(?:\[[^\n;]+\])?) = [^\n;]+;$',word,body,flags=re.M)
   helper='__device__ float ntu_codegen_word(float x){float y;asm volatile ("mov.b32 %0,%1;":"=f"(y):"f"(x));return y;}\n__device__ double ntu_codegen_word(double x){double y;asm volatile ("mov.b64 %0,%1;":"=d"(y):"d"(x));return y;}\n__device__ int ntu_codegen_word(int x){return x;}\n'
   return source[:found.start()]+helper+source[found.start():start]+body+source[end:]
  ntu.module_source=altered_word
 if a.no_unroll_unit:
  original_source=ntu.module_source
  def no_unroll(*args,**kw):
   source=original_source(*args,**kw);found=re.search(r'NTU_DEV_BIG void ntu_'+re.escape(a.no_unroll_unit)+r'\([^;]+\) \{',source);assert found is not None
   start=found.end();end=source.index('\n}\n',start);body=re.sub(r'(^[ \t]*)for \(',r'\1#pragma unroll 1\n\1for (',source[start:end],flags=re.M)
   return source[:start]+body+source[end:]
  ntu.module_source=no_unroll
 original=ntu.compile_options
 ntu.compile_options=lambda:original()+tuple(a.option)
 run(a.input,a.output)
 hdr=np.fromfile(a.input,'<i4',count=8);meta=dict(nz=int(hdr[2]),columns=list(range(int(hdr[1]))),steps=int(hdr[3]))
 x=read(a.reference,meta);y=read(a.output,meta);res={'options':ntu.compile_options(),'fields':{}}
 def record(prefix,left,right,names):
  for i,name in enumerate(names):res['fields'][prefix+'.'+name]={'bit_mismatches':int(np.count_nonzero(left[i].view('u4')!=right[i].view('u4'))),'max_abs_error':float(np.max(np.abs(left[i].astype('f8')-right[i].astype('f8'))))}
 record('init',x[0],y[0],INIT)
 for st in range(meta['steps']):
  record(f'step{st+1}',x[1][st][0],y[1][st][0],OUT);record(f'surface{st+1}',x[1][st][1],y[1][st][1],S2)
 a.output.with_suffix('.json').write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
if __name__=='__main__':main()
