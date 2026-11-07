"""Generate a bounded assignment trace for host/device localization."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.ntu_wrf461_oracle import f2cu

HEADER=r'''
#ifdef NTU_HOST
static FILE* ntu_debug_file=nullptr;
static unsigned int ntu_debug_used=0;
static void ntu_debug_word(unsigned int line,unsigned int type,unsigned long long bits) {
 if(ntu_debug_used++>=1000000u)return;
 unsigned long long record[2]={(unsigned long long)line|((unsigned long long)type<<32),bits};fwrite(record,8,2,ntu_debug_file);
}
#else
__device__ unsigned long long* ntu_debug_buffer;
__device__ unsigned int ntu_debug_used=0;
__device__ void ntu_debug_word(unsigned int line,unsigned int type,unsigned long long bits) {
 unsigned int at=ntu_debug_used++;if(at>=1000000u)return;
 ntu_debug_buffer[2*at]=(unsigned long long)line|((unsigned long long)type<<32);ntu_debug_buffer[2*at+1]=bits;
}
extern "C" __global__ void ntu_debug_bind(unsigned long long* buffer){ntu_debug_buffer=buffer;ntu_debug_used=0;}
extern "C" __global__ void ntu_debug_count(unsigned int* count){count[0]=ntu_debug_used;}
#endif
NTU_DEV void ntu_debug(unsigned int line,float value){ntu_debug_word(line,1,__float_as_uint(value));}
NTU_DEV void ntu_debug(unsigned int line,double value){ntu_debug_word(line,2,(unsigned long long)__double_as_longlong(value));}
'''

def generate(source,parameters,out,limit=1000000,units=None,first_line=0,last_line=1000000,barrier=False):
 units=set(units or ("NTU_MICRO","ICENU","LARGE_DT","SMALL_DT","SEDI_FALL"))
 if limit <= 0: raise ValueError("trace limit must be positive")
 original=f2cu.Emitter.stmt
 def stmt(self,node,ind):
  lines=original(self,node,ind)
  if isinstance(node,f2cu.Simple) and self.unit.name in units and first_line <= node.ln <= last_line:
   try: assignment=f2cu.split_assign(node.text)
   except SyntaxError: assignment=None
   if assignment:
    left,right=assignment; lhs=f2cu.parse_lhs(left); sym=self.sym(lhs.name)
    if sym and sym.type in ('f','d') and (not sym.dims or isinstance(lhs,f2cu.Ref)):
     code,_=self.emit(lhs)
     addition='NTU_COMPILER_BARRIER();' if barrier else f'ntu_debug({node.ln}u, {code});'
     lines+='\n'+'    '*ind+addition
  return lines
 f2cu.Emitter.stmt=stmt
 old=sys.argv
 try:
  sys.argv=['f2cu.py',str(source),'--params-in',str(parameters),'--out',str(out)];f2cu.main()
 finally:sys.argv=old
 header=('#ifdef NTU_HOST\n#define NTU_COMPILER_BARRIER() do {} while(0)\n#else\n#define NTU_COMPILER_BARRIER() asm volatile (\"\" ::: \"memory\")\n#endif\n' if barrier else HEADER.replace('1000000u',f'{limit}u'))
 out.write_text(header+'\n'+out.read_text())

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('source',type=Path);ap.add_argument('parameters',type=Path);ap.add_argument('out',type=Path);ap.add_argument('--limit',type=int,default=1000000);ap.add_argument('--unit',action='append');ap.add_argument('--first-line',type=int,default=0);ap.add_argument('--last-line',type=int,default=1000000);ap.add_argument('--barrier',action='store_true');a=ap.parse_args();generate(a.source,a.parameters,a.out,a.limit,a.unit,a.first_line,a.last_line,a.barrier)
