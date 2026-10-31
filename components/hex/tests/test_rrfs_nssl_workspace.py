"""Column workspace stays bounded when the real mesh outgrows one launch."""
import pytest
from woof.hex.mpasphys.rrfs.mp_nssl2m import workspace_plan


def test_workspace_tiles_cover_mesh_and_respect_memory_budget():
    for nc, nz, free in ((2562,59,16*1024**3), (40962,59,16*1024**3),
                         (1000000,64,4*1024**3), (17,59,128*1024**2)):
        batch, stride = workspace_plan(nc,nz,free)
        assert batch*stride <= min(2*1024**3,free//4)
        if batch < nc:assert batch%32 == 0
        intervals = [(first,min(first+batch,nc)) for first in range(0,nc,batch)]
        assert intervals[0][0] == 0 and intervals[-1][1] == nc
        assert all(a[1] == b[0] for a,b in zip(intervals,intervals[1:]))
        assert sum(b-a for a,b in intervals) == nc
    assert workspace_plan(2562,59,16*1024**3) == (2592,669696)
    assert workspace_plan(40962,59,16*1024**3) == (3200,669696)
    boundary=32*669696*4
    assert workspace_plan(2562,59,boundary) == (32,669696)
    with pytest.raises(MemoryError,match="cannot hold 32 padded columns"):
        workspace_plan(2562,59,boundary-1)


@pytest.mark.parametrize("tile_width", (8,32))
def test_interleaved_descriptors_preserve_divergent_column_storage(tmp_path,tile_width):
    import shutil
    import re
    import subprocess
    from woof.hex.mpasphys.rrfs import mp_nssl2m as m
    compiler=shutil.which("g++")
    if compiler is None:pytest.skip("C++ compiler needed for descriptor storage check")
    source=tmp_path/"check.cpp";binary=tmp_path/"check"
    sequence_helper=re.search(r"template<class T,int R> __device__ NsslArray<T,R> nssl_sequence\(.*?\n}", m.OPERATOR_SUPPORT, re.S).group()
    source_text=("#define __device__\n"+m.ARRAY_SUPPORT+sequence_helper+r'''

#include <cassert>
#include <vector>
int main() {
 std::vector<unsigned char> storage(32*65536,0);
 NsslArray<float,2> a[32]; NsslArray<double,1> b[32]; NsslArray<bool,1> c[32];
 for(int lane=0;lane<32;++lane) {
  NsslArena arena{storage.data(),0,32,lane,0,49152};
  // Unlike allocation order and extent across lanes must never alias.
  if(lane%2) c[lane]=nssl_array<bool,1>(arena,{1},{7+lane});
  a[lane]=nssl_array<float,2>(arena,{-2,1},{4,13});
  b[lane]=nssl_array<double,1>(arena,{1},{7});
  if(!(lane%2)) c[lane]=nssl_array<bool,1>(arena,{1},{7+lane});
  for(int i=0;i<a[lane].size();++i)a[lane].linear(i)=lane*1000+i;
  for(int i=1;i<=7;++i)b[lane](i)=lane*1000+i+0.125;
  for(int i=1;i<=7+lane;++i)c[lane](i)=i%2;
 }
 assert(a[2].data-a[0].data==2); // Adjacent four-byte lanes, unlike old eight-byte cells.
 assert(b[2].data-b[0].data==2); // Doubles occupy their separate eight-byte bank.
 for(int lane=0;lane<32;++lane) {
  assert(a[lane].stride[0]==32 && b[lane].stride[0]==32 && c[lane].stride[0]==128);
  for(int i=0;i<a[lane].size();++i)assert(a[lane].linear(i)==lane*1000+i);
  auto section=a[lane].slice(NsslRange{-1,2},5);
  auto pointer_sequence=nssl_sequence<float,1>(&a[lane](-1,5),a[lane].stride[0],{1},{4});
  auto descriptor_sequence=a[lane].sequence<1>({1},{91});
  for(int i=0;i<4;++i) {
   assert(section.linear(i)==lane*1000+4*7+1+i);
   assert(pointer_sequence.linear(i)==section.linear(i));
  }
  for(int i=0;i<91;++i)assert(descriptor_sequence.linear(i)==a[lane].linear(i));
  for(int i=1;i<=7;++i)assert(b[lane](i)==lane*1000+i+0.125);
  for(int i=1;i<=7+lane;++i)assert(c[lane](i)==bool(i%2));
 }
 NsslArena init{storage.data(),0};auto x=nssl_array<float,1>(init,{1},{9});
 assert(x.stride[0]==1 && init.offset==36);
}
''').replace('32',str(tile_width)).replace('==128',f'=={tile_width*4}')
    source.write_text(source_text)
    subprocess.run([compiler,"-std=c++17","-O2",str(source),"-o",str(binary)],check=True)
    subprocess.run([str(binary)],check=True)


def test_native_phase_split_preserves_two_and_three_moment_driver(tmp_path):
    import os
    from pathlib import Path
    import shutil
    import subprocess
    import sys
    if shutil.which("g++") is None:
        pytest.skip("C++ compiler needed for native phase identity witness")
    script = Path(__file__).parent/"rrfs_nssl2m_native"/"phase_identity.py"
    subprocess.run([sys.executable,str(script),str(tmp_path)],check=True,
                   env={**os.environ,"CUDA_VISIBLE_DEVICES":""},timeout=240)


def test_standalone_replay_loads_same_cuda_source():
    import importlib.util
    from woof.hex.mpasphys.rrfs import mp_nssl2m as m
    spec = importlib.util.spec_from_file_location("nssl_replay_target", m.__file__)
    standalone = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(standalone)
    assert standalone.cuda_source(shared_math_source="") == m.cuda_source(shared_math_source="")


def test_streamed_fallout_preserves_native_flux_words(tmp_path):
    import os
    from pathlib import Path
    import shutil
    import subprocess
    import sys
    if shutil.which("g++") is None:
        pytest.skip("C++ compiler needed for fallout identity witness")
    script = Path(__file__).parent/"rrfs_nssl2m_native"/"fallout_identity.py"
    subprocess.run([sys.executable,str(script),str(tmp_path)],check=True,
                   env={**os.environ,"CUDA_VISIBLE_DEVICES":""},timeout=90)


def test_banked_arena_bounds_every_native_call_stack(tmp_path):
    from pathlib import Path
    import subprocess
    import sys
    script=Path(__file__).parent/'rrfs_nssl2m_native'/'arena_bank_bounds.py'
    subprocess.run([sys.executable,str(script),str(tmp_path/'banks.json')],check=True,timeout=30)
