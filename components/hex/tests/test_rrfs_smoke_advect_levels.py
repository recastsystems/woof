"""Execute the actual native and distributed advection arithmetic on the host."""
from pathlib import Path
import re
import shutil
import subprocess
import pytest


def test_native_advection_matches_distributed_loops(tmp_path):
    from woof.hex.mpasphys.rrfs import smoke_9652bbf as m, smoke_columns
    from woof.hex.mpasphys.rrfs.smoke_advect_levels import functions, clones
    cxx = shutil.which('c++')
    if not cxx:
        pytest.skip('C++ compiler unavailable')
    source = smoke_columns.source(m.CUDA_SOURCE, m.ROUTINES, m._NATIVE_BOUNDARIES)
    native = list(functions(source).values())
    # Four host workers exercise cross-worker stencil dependencies without
    # reserving 32 CPU cores. GPU replay checks the actual 32-lane execution.
    workers = [s.replace('+=32', '+=4') for s in clones(source)]
    fields = sorted(set(re.findall(r'coms->(\w+)\(', '\n'.join(native))) |
                    set(re.findall(r'coms->(\w+)[,)]', '\n'.join(native))))
    fields.remove('dt') if 'dt' in fields else None
    header = (Path(__file__).parent/'rrfs_smoke_oracle/host_prefix.cuh').read_text()
    helpers = m.CUDA_SOURCE[m.CUDA_SOURCE.index('// Native Fortran unary'):m.CUDA_SOURCE.index('template<class T> struct ScalarExpr')]
    types = '''
#include <cassert>
#include <cstdio>
#include <thread>
#include <barrier>
std::barrier sync_workers(4);
#define __syncwarp(mask) sync_workers.arrive_and_wait()
struct State{};struct Arena{};struct ArenaScope{ArenaScope(Arena&) {}};
struct FStr{FStr(const char*){}};
template<class T> struct FVector {
 T *p;void bind(int,int){};T &operator()(int k){return p[k-1];}
};
struct Array {
 float p[200];float &operator()(int k){return p[k-1];}
 operator FVector<float>(){return {p};}
};
FVector<float> sd_sequence(Array &a,int k){return {a.p+k-1};}
void sd_assign(Array &a,const Array &b,Arena &){std::memcpy(a.p,b.p,sizeof(a.p));}
struct Plume {float dt;
''' + ''.join('Array '+n+';\n' for n in fields) + '};\n'
    bodies=native+workers
    declarations='\n'.join(s.split(' {\n',1)[0]+';' for s in bodies)
    program=header+helpers+types+declarations+'\n'+'\n'.join(bodies)+'''
int main(){
 unsigned long words=0;
 for(int c=0;c<240;c++){
  Plume a{};float *p=reinterpret_cast<float*>(&a);
  for(unsigned j=0;j<sizeof(a)/4;j++)p[j]=float((j*13+c*7)%71+1)*0.01f;
  a.dt=0.1f*(c%7+1);
  for(int k=1;k<=200;k++){
   if(c%5==0 && k%11==0)a.qv(k)=__uint_as_float(0x7fc12345u);
   if(c%7==0 && k%5==0)a.qi(k)=__uint_as_float(0xff800001u);
   if(c%11==0 && k%3==0)a.qc(k)=-0.f;
  }
  Plume b=a;State state;Arena arena;int n=2+c%198;
  scl_advectc_plumerise(state,arena,&a,FStr("SC"),n);
  std::thread workers[4];
  for(int lane=0;lane<4;lane++)workers[lane]=std::thread([&,lane]{
   State local_state;Arena local_arena;
   sd_worker_scl_advectc_plumerise(local_state,local_arena,lane,&b,FStr("SC"),n);
  });
  for(auto &worker:workers)worker.join();
  assert(std::memcmp(&a,&b,sizeof(a))==0);words+=sizeof(a)/4;
 }
 std::printf("240 cases, %lu words, zero differences\\n",words);
}
'''
    cpp=tmp_path/'advect.cpp';cpp.write_text(program);exe=tmp_path/'advect'
    subprocess.run([cxx,'-std=c++20','-pthread','-O2','-ffp-contract=off','-fno-fast-math',str(cpp),'-o',str(exe)],check=True,timeout=60)
    subprocess.run([str(exe)],check=True,timeout=60)
