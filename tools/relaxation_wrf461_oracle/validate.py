"""Compare actual CUDA entry points with a pinned WRF column oracle."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def fixture(path):
    rows = [line.split() for line in Path(path).read_text().splitlines() if not line.startswith('#')]
    a = np.zeros((15, 9, 10, 10), np.float32)
    for row in rows:
        i, j, k = map(int, row[:3])
        a[:,k-1,j,i] = np.array([int(x,16) for x in row[3:]], np.uint32).view(np.float32)
    return a


def grade(actual, expected):
    a = np.ascontiguousarray(actual).view(np.uint32)
    e = np.ascontiguousarray(expected).view(np.uint32)
    def ordered(x):
        return np.where(x & np.uint32(0x80000000), ~x, x | np.uint32(0x80000000)).astype(np.int64)
    ulp = np.abs(ordered(a)-ordered(e))
    return {'count':a.size,'different_words':int(np.count_nonzero(a!=e)), 'max_ulp':int(ulp.max()),
            'max_abs':float(np.max(np.abs(actual.astype(np.float64)-expected))),
            'actual_sha256':hashlib.sha256(a.tobytes()).hexdigest(),
            'oracle_sha256':hashlib.sha256(e.tobytes()).hexdigest()}


def run(path, mutate=False):
    import cupy as cp
    from woof.core.kernels import load_module_no_ftz, module_source
    from woof.wrf_exact import ENABLED, compile_receipt
    a = fixture(path)
    interior = (slice(0,8),slice(1,9),slice(1,9))
    device = lambda x: cp.asarray(np.ascontiguousarray(x))
    lat,p,pb,pw,pi,t,r0,rth,ru,rv,u0,v0,ut,vt,degrad = a
    heating = device(r0[interior]); pressure = device((p+pb)[interior])
    kernel = load_module_no_ftz('idealized_forcing').get_function('relax_temperature')
    if mutate:
        src=module_source('idealized_forcing')
        original='float kkt = __fadd_rn(kka, seasonal);'
        assert original in src
        src=src.replace(original,original+' kkt = __uint_as_float(__float_as_uint(kkt) ^ 1u);')
        from woof.wrf_exact import effective_options
        opts=effective_options(('-std=c++17',)) if ENABLED else ('-std=c++17', '--ftz=false')
        from woof import nvrtc_ptx_cache as compiler
        binary,_=compiler.compile_using_nvrtc(src,opts,None,'mutation.cu')
        mod=cp.cuda.function.Module(); mod.load(binary.encode() if isinstance(binary,str) else binary)
        kernel=mod.get_function('relax_temperature')
    kernel((2,),(256,),(np.int32(8),np.int32(8),np.int32(8), device(lat[0,1:9,1:9]),
           pressure,device(pw[0,1:9,1:9]),device(pi[interior]),device(t[interior]),
           np.float32(degrad[0,0,0]),np.float32(287.),np.float32(1004.5),heating))
    u=device(u0[:8,1:9,1:10]); v=device(v0[:8,1:10,1:9])
    for axis,wind,tend,stop in ((0,ru[:8,1:9,1:10],u,8),(1,rv[:8,1:10,1:9],v,8)):
        k=load_module_no_ftz('idealized_forcing').get_function('relax_momentum')
        k((3,),(256,),(np.int32(8),np.int32(8),np.int32(8),np.int32(axis),np.int32(stop),
            device(p[:8]),device(pb[:8]),device(wind),tend))
    result={'strict':ENABLED,'mutation':mutate,'columns':64,'levels':8,
            'rthraten':grade(cp.asnumpy(heating),rth[interior]),
            'ru_tend':grade(cp.asnumpy(u),ut[:8,1:9,1:10]),
            'rv_tend':grade(cp.asnumpy(v),vt[:8,1:10,1:9]),
            'compile':compile_receipt()}
    from woof.certify.kernel_manifest import kernel_manifest
    result['kernel_manifest']=kernel_manifest()
    result['passed'] = (result['rthraten']['different_words'] > 0 if mutate else
                        all(result[n]['max_ulp']==0 for n in ('rthraten','ru_tend','rv_tend')))
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('oracle'); parser.add_argument('out'); parser.add_argument('--mutate',action='store_true')
    args=parser.parse_args()
    result=run(args.oracle,args.mutate)
    Path(args.out).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='compile'},indent=2))
    assert result['passed'], 'forcing comparison failed; see the written receipt'
