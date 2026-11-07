"""Grade WOOF's math helpers on the operands the KSAS columns use."""
from pathlib import Path
import gzip
import hashlib
import json
import numpy as np
import pytest
cp=pytest.importorskip('cupy')
def test_column_math_operands():
    from woof.core.kernels import module_source,module_options
    root=Path(__file__).parent/'fixtures'/'ksas_wrf461'
    raw=gzip.decompress((root/'math.bin.gz').read_bytes())
    meta=json.loads((root/'math.json').read_text())
    assert hashlib.sha256(raw).hexdigest()==meta['raw_sha256']
    rows=np.frombuffer(raw,dtype='<u4').reshape(-1,4)
    probe=r'''
extern "C" __global__ void sas_math_probe(const unsigned* a,unsigned* out,int n) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 float x=__uint_as_float(a[4*i+1]),y=__uint_as_float(a[4*i+2]),v;
 switch(a[4*i]){case 0:v=gfk_exp(x);break;case 1:v=gfk_pow(x,y);break;
 case 2:v=gfk_log(x);break;case 3:v=glibc_tanf(x);break;default:v=wrf_atan_f32(x);}
 out[i]=__float_as_uint(v);
}
'''
    from woof.core.kernels import _load_module_no_ftz
    m=_load_module_no_ftz('sas-math-probe',module_source('sas')+probe)
    out=cp.empty(len(rows),dtype=cp.uint32)
    m.get_function('sas_math_probe')(((len(rows)+127)//128,),(128,),(cp.asarray(rows),out,np.int32(len(rows))))
    got=cp.asnumpy(out)
    bad=np.flatnonzero(got!=rows[:,3])
    print('KSAS math operand mismatches:',len(bad),'/',len(rows))
    for i in bad[:30]:
        print(int(rows[i,0]),[float(v) for v in rows[i,1:3].view(np.float32)],hex(int(rows[i,3])),hex(int(got[i])))
    assert not len(bad)


def test_strict_loader_preserves_subnormal_arithmetic():
    from woof.core.kernels import _load_module_no_ftz
    source = r"""
extern "C" __global__ void sas_ftz_probe(const float* x,float* out) {
 out[0]=x[0]*x[1]; out[1]=x[2]/x[3]; out[2]=x[4]<0.f ? 1.f : 0.f;
}
"""
    m = _load_module_no_ftz("sas-ftz-probe", source)
    x = np.array([1.e-30,1.e-10,1.e-39,2.,-1.e-39],np.float32)
    out = cp.empty(3,np.float32)
    m.get_function("sas_ftz_probe")((1,),(1,),(cp.asarray(x),out))
    expected = np.array([x[0]*x[1],x[2]/x[3],1.],np.float32)
    assert np.array_equal(cp.asnumpy(out).view(np.uint32),expected.view(np.uint32))
