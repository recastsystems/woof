#!/usr/bin/env python3
"""Check the actual portable header's upper/lower 32-lane grouping off card.

The host intrinsic model checks bit and index arithmetic. ELF metadata proves
which physical wave size the real AMD compiler used. Only --run executes a
physical GPU, and that is required for hardware qualification.
"""
import argparse,json,os,re,subprocess,tempfile
from pathlib import Path
from woof.hip_backend import compile_aot,toolchain_identity

SOURCE='''extern "C" __global__ void warp_audit(unsigned* p) {
 unsigned i=threadIdx.x, l=i&31;
 p[i*7+0]=__ballot_sync(0xffffffffu,(i%5)==0);
 p[i*7+1]=__shfl_sync(0xffffffffu,i,3);
 p[i*7+2]=__shfl_up_sync(0xffffffffu,i,1);
 p[i*7+3]=__shfl_down_sync(0xffffffffu,i,1);
 p[i*7+4]=__shfl_xor_sync(0xffffffffu,i,1);
 // CUDA requires only named lanes to execute the masked collective. Bits
 // outside that mask are unspecified, so compare the defined bits only.
 p[i*7+5]=(i&1) ? 0u : (__ballot_sync(0x55555555u,(i%3)==0)&0x55555555u);
 p[i*7+6]=__reduce_max_sync(0xffffffffu,i);
}'''

def host_model():
 header=Path(__file__).resolve().parents[1]/'woof/hip_portable.h'
 pre='''#include <cmath>
#include <cstdint>
#include <cassert>
#define __HIPCC_RTC__ 1
#define __HIP_PLATFORM_AMD__ 1
#define __device__
#define __constant__
#define __forceinline__ inline
#define __launch_bounds__(...)
unsigned lane=0; unsigned long long ballot_bits;
unsigned lo(unsigned,unsigned){return lane&31;}
unsigned hi(unsigned,unsigned n){return (lane&32)+n;}
#define __builtin_amdgcn_mbcnt_lo lo
#define __builtin_amdgcn_mbcnt_hi hi
#define __builtin_amdgcn_s_memrealtime() 0ull
#define wall_clock64() 0ull
#define __builtin_amdgcn_fence(...)
#define __builtin_amdgcn_wave_barrier()
unsigned long long __ballot(int){return ballot_bits;}
unsigned __shfl(unsigned,int s,int w){return (lane/w)*w+s;}
unsigned __shfl_up(unsigned v,unsigned d,int w){return lane%w>=d?v-d:v;}
unsigned __shfl_down(unsigned v,unsigned d,int w){return lane%w+d<(unsigned)w?v+d:v;}
unsigned __shfl_xor(unsigned,int d,int w){return (lane/w)*w+((lane%w)^d);}
'''
 main='''int main(){
 for(lane=0;lane<64;lane++){
  unsigned base=lane&32;
  ballot_bits=0x8421084210842109ull;
  assert(gpuwm_lane()==lane);
  assert(gpuwm_activemask()==unsigned(ballot_bits>>base));
  assert(gpuwm_ballot_sync(0x55555555u,1)==(unsigned(ballot_bits>>base)&0x55555555u));
  assert(gpuwm_shfl_sync(~0u,lane,3)==base+3);
  assert(gpuwm_shfl_up_sync(~0u,lane,1)==(lane%32?lane-1:lane));
  assert(gpuwm_shfl_down_sync(~0u,lane,1)==(lane%32<31?lane+1:lane));
  assert(gpuwm_shfl_xor_sync(~0u,lane,1)==(lane^1));
 }
}'''
 with tempfile.TemporaryDirectory(prefix='gpuwm-wave-model-') as d:
  source=Path(d)/'model.cpp';source.write_text(pre+'\n#include "'+str(header)+'"\n'+main)
  subprocess.run(['c++','-O2','-std=c++17',str(source),'-o',d+'/model'],check=True,capture_output=True)
  subprocess.run([d+'/model'],check=True)
 return {'lanes':64,'checks':['upper ballot narrowing','masked ballot','shuffle width','up/down edges','xor grouping'],'physical_execution':False}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--run',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 result={'host_model':host_model(),'images':[]}
 hipcc,_=toolchain_identity();readelf=str(Path(hipcc).parent.parent/'llvm/bin/llvm-readelf')
 for target,extra,wave in [('gfx942',(),64),('gfx1100',(),32)]:
  image=a.output/(target+'-wave'+str(wave)+'.hsaco');blob,receipt=compile_aot(SOURCE,('-std=c++17',*extra),target,output=image)
  notes=subprocess.run([readelf,'--notes',str(image)],check=True,capture_output=True,text=True).stdout
  image.with_suffix('.notes.txt').write_text(notes)
  sizes=list(map(int,re.findall(r'\.wavefront_size:\s*(\d+)',notes)))
  assert sizes and set(sizes)=={wave},sizes
  result['images'].append(dict(arch=target,wavefront_size=wave,image_sha256=receipt['image_sha256']))
 if a.run:
  import cupy as cp,numpy as np
  from woof.hip_backend import install,device_info
  install();out=cp.empty((64,7),cp.uint32)
  cp.RawKernel(SOURCE,'warp_audit')((1,),(64,),(out,))
  expected=np.empty((64,7),np.uint32)
  for i in range(64):
   base=i&32;l=i&31
   expected[i]=[sum(1<<j for j in range(32) if (base+j)%5==0),base+3,i-1 if l else i,i+1 if l<31 else i,i^1,0 if i&1 else sum(1<<j for j in range(32) if (base+j)%3==0)&0x55555555,base+31]
  np.testing.assert_array_equal(out.get(),expected);result['physical_execution']=device_info()
 print(json.dumps(result,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
