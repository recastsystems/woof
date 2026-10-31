"""Execute the native and deferred microphysics orders on the host."""
from pathlib import Path
import re
import shutil
import subprocess
import pytest


def test_native_mixed_stencil_survives_deferred_reverse_level_order(tmp_path):
    from woof.hex.mpasphys.rrfs import smoke_9652bbf as m, smoke_columns
    from woof.hex.mpasphys.rrfs.smoke_micro_levels import functions, NAMES, VECTORS
    cxx = shutil.which('c++')
    if not cxx:
        pytest.skip('C++ compiler unavailable')
    source = smoke_columns.source(m.CUDA_SOURCE, m.ROUTINES, m._NATIVE_BOUNDARIES)
    native = functions(m.CUDA_SOURCE)
    def function(name):
        match = re.search(r'^SDDEV (?:void|float) '+name+r'\([^\n]+\) \{', source, re.M)
        end = source.index('\n}\n', match.start())+3
        return source[match.start():end]
    clones = [function('sd_micro_'+n) for n in NAMES]
    header = (Path(__file__).parent/'rrfs_smoke_oracle/host_prefix.cuh').read_text()
    helpers = m.CUDA_SOURCE[m.CUDA_SOURCE.index('// Native Fortran unary'):m.CUDA_SOURCE.index('template<class T> struct ScalarExpr')]
    fields = sorted(set(VECTORS) | {'w','pe','zt'})
    types = '''
#include <cassert>
#include <cstdio>
struct State{};struct Arena{};struct ArenaScope{ArenaScope(Arena&) {}};
struct Array {float p[200];float &operator()(int k){return p[k-1];}};
struct Plume {int l,nm1;float dt,wbar,dqsdz;
''' + ''.join('Array '+n+';\n' for n in fields) + '};\n'
    types += source[source.index('struct SdMicroArray'):source.index('struct SdMicroWork')]
    routines = list(native.values()) + clones + [function('esat_pr')]
    declarations = '\n'.join(s.split(' {\n',1)[0]+';' for s in routines)
    prep = m.CUDA_SOURCE[m.CUDA_SOURCE.index('coms->wbar = sd_mul(sd_add(coms->w(coms->l)'):m.CUDA_SOURCE.index('waterbal(state,arena,coms); // L591')]
    program = header+helpers+types+declarations+'\n'+'\n'.join(routines)+'''
void prepare(State &state,Arena &arena,Plume *coms){
 float eps=__uint_as_float(0x3F1F3B64U),es;
'''+prep+'''
}
void native_order(State &state,Arena &arena,Plume *coms){
 coms->l=2;
 while(coms->l<=coms->nm1-1){prepare(state,arena,coms);waterbal(state,arena,coms);coms->l++;}
}
void deferred_order(State &state,Arena &arena,Plume *coms){
 float wbar[200],dqsdz[200];coms->l=2;
 while(coms->l<=coms->nm1-1){
  prepare(state,arena,coms);wbar[coms->l-1]=coms->wbar;dqsdz[coms->l-1]=coms->dqsdz;coms->l++;
 }
 SdMicroPlume view;view.dt=coms->dt;
'''+''.join('view.'+n+'={coms->'+n+'.p};\n' for n in VECTORS)+'''
 for(int k=coms->nm1-1;k>=2;k--){
  view.l=k;view.wbar=wbar[k-1];view.dqsdz=dqsdz[k-1];sd_micro_waterbal(state,arena,&view);
 }
}
int main(){
 State state;Arena arena;unsigned long words=0;
 for(int c=0;c<240;c++){
  Plume a{};a.nm1=2+c%199;a.dt=(c%7+1)*0.2f;a.wbar=0.7f;a.dqsdz=0.02f;
  for(int k=1;k<=200;k++){
   a.t(k)=220.f+(c+7*k)%75;a.pe(k)=100.f-k*0.3f;a.zt(k)=k*100.f;
   a.w(k)=float((k+c)%13-6);a.qv(k)=0.001f*((k+c)%12);
   a.qsat(k)=0.001f*((k+c)%9);a.qc(k)=0.0001f*((k+c)%6);
   a.qh(k)=0.0001f*((k+c)%5);a.qi(k)=0.0001f*((k+c)%4);
   a.rho(k)=0.8f;a.est(k)=0.1f;a.cvi(k)=0.5f;
   if(c%3==0 && k%11==0)a.qv(k)=__uint_as_float(0x7fc12345u);
   if(c%7==0 && k%5==0)a.qi(k)=__uint_as_float(0xff800001u);
   if(c%11==0 && k%3==0)a.qc(k)=-0.f;
  }
  Plume b=a;native_order(state,arena,&a);deferred_order(state,arena,&b);
  assert(std::memcmp(&a,&b,sizeof(a))==0);words+=sizeof(a)/4;
 }
 std::printf("240 cases, %lu words, zero differences\\n",words);
}
'''
    cpp=tmp_path/'micro.cpp';cpp.write_text(program);exe=tmp_path/'micro'
    subprocess.run([cxx,'-std=c++17','-O2','-ffp-contract=off','-fno-fast-math',str(cpp),'-o',str(exe)],check=True,timeout=60)
    subprocess.run([str(exe)],check=True,timeout=30)
