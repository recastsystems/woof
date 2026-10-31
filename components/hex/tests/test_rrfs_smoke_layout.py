"""Host execution of the production array descriptors used by CUDA smoke.

A changed physical layout must retain sequence and slice associations before
it can be tested against complete native calls on the GPU.
"""
from pathlib import Path
import shutil
import subprocess
import pytest


def _run(tmp_path, body, assignment=False, vectors=False, fixed=False):
    from woof.hex.mpasphys.rrfs import smoke_9652bbf as smoke, smoke_columns
    compiler = shutil.which('c++')
    if compiler is None:
        pytest.skip('C++ compiler unavailable')
    source = smoke_columns.source(smoke.CUDA_SOURCE, smoke.ROUTINES, smoke._NATIVE_BOUNDARIES)
    # This host fixture extracts only array descriptors. Device synchronization
    # is exercised by the real GPU replay and sanitizers.
    source = source.replace('#include <cuda/barrier>\n', '')
    helpers = ''
    if assignment:
        helpers = source[source.index('template<class T> struct ScalarExpr'):source.index('struct OpAdd')]
        helpers += source[source.index('template<class E> struct SdScalar'):source.index('template<class T> SDDEV T sd_sum')]
    if vectors:
        from woof.hex.mpasphys.rrfs.smoke_vectors import VECTOR_SOURCE
        helpers += VECTOR_SOURCE
    if fixed:
        from woof.hex.mpasphys.rrfs.smoke_vectors import FIXED_VECTOR_SOURCE
        helpers += FIXED_VECTOR_SOURCE
    source = source[:source.index('struct FStr {')] + helpers
    path = tmp_path / 'layout.cpp'
    path.write_text('#include <cassert>\n#define SDDEV\n' + source + '\nint main(){\n' + body + '\n}\n')
    binary = tmp_path / 'layout'
    subprocess.run([compiler, '-std=c++17', '-O2', str(path), '-o', str(binary)], check=True, timeout=30)
    subprocess.run([str(binary)], check=True, timeout=5)


def test_linear_index_preserves_strided_and_compact_views(tmp_path):
    _run(tmp_path, '''
    float words[32*40]; for(int i=0;i<32*40;i++) words[i]=float(i);
    FArray<float> vector; vector.p=words+7; vector.stride[0]=32; vector.bind(-3,12);
    for(int i=0;i<vector.size();i++) assert(vector.linear(i)==7+32*i);
    FArray<float> compact; compact.p=words+7; compact.compact=true; compact.bind(10,14,-2,5);
    assert(compact.size()==8);
    for(int i=0;i<compact.size();i++) assert(compact.linear(i)==7+i);
    FArray<float> full; full.p=words+7; full.stride[0]=32; full.bind(1,4,1,5);
    auto cut=full.section(Slice(2,3),Slice(2,4));
    assert(cut.size()==6);
    for(int i=0;i<cut.size();i++) assert(cut.linear(i)==167+(i%2)*32+(i/2)*128);
    ''')


def test_assignment_scalar_fill_and_overlap_keep_native_association(tmp_path):
    _run(tmp_path, '''
    float words[64];for(int i=0;i<64;i++)words[i]=float(i);
    float temporary[64];Arena arena{temporary,0,64,true,3,1,4};
    FArray<float> full;full.p=words;full.bind(1,4,1,5);
    sd_assign(full,7.f,arena);
    for(int i=0;i<20;i++)assert(words[i]==(i%4==2?7.f:float(i)));
    assert(arena.used==0);
    arena.column_mode=false;
    FArray<float> vector;vector.p=words+20;vector.bind(1,10);
    sd_assign(vector.section(Slice(2,10)),vector.section(Slice(1,9)),arena);
    for(int i=1;i<10;i++)assert(words[20+i]==float(19+i));
    FArray<float> compact;compact.p=words+40;compact.compact=true;compact.bind(1,4,1,5);
    sd_assign(compact,9,arena);
    for(int i=0;i<5;i++)assert(words[40+i]==9.f);
    assert(words[45]==45.f && arena.used==0);
    ''', assignment=True)


def test_plume_vectors_preserve_bounds_stride_and_overlapping_assignment(tmp_path):
    _run(tmp_path, '''
    float words[64];for(int i=0;i<64;i++)words[i]=float(i);
    float temporary[64];Arena arena{temporary,0,64};
    FArray<float> descriptor;descriptor.p=words;descriptor.stride[0]=2;descriptor.bind(-3,6);
    FVector<float> vector;vector=descriptor;
    assert(vector.size()==10 && vector(-2)==2 && vector.linear(3)==6);
    auto cut=vector.section(Slice(-2,3));
    assert(cut.size()==6 && cut.linear(3)==8);
    FArray<float> back=vector;
    assert(back.lo[0]==-3 && back.hi[0]==6 && back.stride[0]==2);
    FVector<float> left,right;
    descriptor.p=words+30;descriptor.stride[0]=1;descriptor.rank=0;descriptor.bind(1,10);
    left=descriptor;right=vector;
    sd_assign(left,right,arena);
    for(int i=0;i<10;i++)assert(words[30+i]==float(i*2));
    left=descriptor.section(Slice(2,10));right=descriptor.section(Slice(1,9));
    sd_assign(left,right,arena);
    for(int i=1;i<10;i++)assert(words[30+i]==float((i-1)*2));
    assert(arena.used==0);
    auto sequence=sd_sequence(vector,-1);sequence.bind(1,4);
    // Sequence association starts a contiguous sequence at the selected word.
    assert(sequence(1)==words[4] && sequence(2)==words[5]);
    arena.column_mode=true;arena.column=2;arena.ims=1;arena.ime=10;
    sd_assign(vector,3.f,arena);
    for(int i=0;i<10;i++)assert(words[i*2]==3.f);
    ''', assignment=True, vectors=True)


def test_fixed_native_plume_fields_preserve_views_and_complete_assignment(tmp_path):
    _run(tmp_path, '''
    float words[400],temporary[200];for(int i=0;i<400;i++)words[i]=float(i);
    Arena arena{temporary,0,200};
    FArray<float> descriptor;descriptor.p=words;descriptor.bind(1,200);
    FPlumeVector<float> left,right;left=descriptor;
    descriptor.p=words+200;right=descriptor;
    static_assert(sizeof(left)==200*sizeof(float));
    assert(left.size()==200 && left.lo[0]==1 && left.hi[0]==200);
    FVector<float> dummy=left;dummy.bind(-3,196);
    assert(dummy(-3)==0 && dummy(196)==199);
    auto cut=left.section(Slice(5,9));
    assert(cut.size()==5 && cut(1)==4 && cut(5)==8);
    auto sequence=sd_sequence(left,5);sequence.bind(1,10);
    assert(sequence(1)==4 && sequence(10)==13);
    sd_assign(left,right,arena);
    for(int i=0;i<200;i++)assert(left(i+1)==float(i+200));
    sd_assign(right,7.f,arena);
    for(int i=0;i<200;i++)assert(right(i+1)==7.f);
    sd_assign(left,left,arena);
    for(int i=0;i<200;i++)assert(left(i+1)==float(i+200));
    ''', assignment=True, vectors=True, fixed=True)


@pytest.mark.parametrize("fast", [False, True])
def test_arithmetic_mode_preserves_ieee_values_and_strict_nan_priority(tmp_path, fast):
    from woof.hex.mpasphys.rrfs import smoke_9652bbf as smoke
    compiler = shutil.which('c++')
    if compiler is None:
        pytest.skip('C++ compiler unavailable')
    start = smoke.CUDA_SOURCE.index('SDDEV bool sd_float_nan_word')
    end = smoke.CUDA_SOURCE.index('SDDEV float sd_add', start)
    source = r'''
#include <cassert>
#include <cstdint>
#include <cstring>
#define SDDEV
float __uint_as_float(uint32_t u){float x;std::memcpy(&x,&u,4);return x;}
uint32_t __float_as_uint(float x){uint32_t u;std::memcpy(&u,&x,4);return u;}
''' + smoke.CUDA_SOURCE[start:end] + r'''
float before(float a,float b,float result){
 unsigned int x=__float_as_uint(a),y=__float_as_uint(b);
 if(sd_float_nan_word(x))return __uint_as_float(x|0x00400000u);
 if(sd_float_nan_word(y))return __uint_as_float(y|0x00400000u);
 return sd_float_nan_word(__float_as_uint(result))?__uint_as_float(0xffc00000u):result;
}
void check(uint32_t x,uint32_t y){
 volatile float a=__uint_as_float(x),b=__uint_as_float(y);
 float result[]={a+b,a-b,a*b,a/b};
 for(float r:result){
#ifdef HEX_NUMERICS_FAST
  float expected=r;
#else
  float expected=before(a,b,r);
#endif
  assert(__float_as_uint(expected)==__float_as_uint(sd_sse_binary(a,b,r)));
 }
}
int main(){
 uint32_t words[]={0,0x80000000u,1,0x80000001u,0x007fffffu,0x00800000u,
  0x3f800000u,0xbf800000u,0x7f7fffffu,0xff7fffffu,0x7f800000u,0xff800000u,
  0x7f800001u,0xff800001u,0x7fc00001u,0xffc12345u,0x7fffffffu,0xffffffffu};
 for(auto a:words)for(auto b:words)check(a,b);
 uint32_t state=193;
 for(int i=0;i<100000;i++){
  state=1664525u*state+1013904223u;auto a=state;
  state=1664525u*state+1013904223u;check(a,state);
 }
}
'''
    path = tmp_path / 'arithmetic.cpp'
    path.write_text(('#define HEX_NUMERICS_FAST 1\n' if fast else '') + source)
    binary = tmp_path / 'arithmetic'
    subprocess.run([compiler, '-std=c++17', '-O2', '-fno-fast-math', str(path), '-o', str(binary)], check=True, timeout=30)
    subprocess.run([str(binary)], check=True, timeout=10)
