import sys
"""Profile the real forecast's daylight columns and launch occupancy."""
from pathlib import Path
import hashlib,json
import numpy as np
from netCDF4 import Dataset
import cupy as cp
from woof.core.kernels import get_kernel,module_source

root=Path(sys.argv[1]).resolve()
src=root/'native-frames/wrfout_d01_2000-01-24_14:00:00'
with Dataset(src) as ds:
    read=lambda key:np.asarray(ds[key][0],dtype='f4')
    pressure=read('P')+read('PB')
    temperature=(read('T')+np.float32(300))*(pressure/np.float32(100000))**np.float32(287/1004.5)
    qv=read('QVAPOR')
    rho=pressure/(np.float32(287)*temperature*(1+np.float32(.608)*qv))
    dz=np.diff((read('PH')+read('PHB'))/np.float32(9.81),axis=0)
    nz,ny,nx=qv.shape;n=ny*nx
    vol=lambda a:cp.asarray(np.ascontiguousarray(a.reshape(nz,n)))
    surf=lambda a:cp.asarray(np.ascontiguousarray(a.reshape(n)))
    p,rho,dz,qv,qc,qi,qs,cf=map(vol,(pressure,rho,dz,qv,read('QCLOUD'),read('QICE'),read('QSNOW'),read('CLDFRA')))
    alb,cz=map(surf,(read('ALBEDO'),read('COSZEN')))
zero=cp.zeros(n,cp.float32);one=cp.ones(n,cp.float32)
vzero=cp.zeros_like(qv);out=cp.empty((6,n),cp.float32)
k=get_kernel('farms','farms_column')
args=(np.int32(n),np.int32(nz-1),np.int32(0),np.int32(0),np.int32(0),np.float32(24),p,rho,dz,alb,zero,zero,zero,zero,cz,qv,qi,qs,qc,vzero,vzero,vzero,cf,out)
interp=get_kernel('swint','swint_interp');io=[cp.empty_like(zero) for _ in range(5)]
def timing(fn):
    for _ in range(3):fn()
    stream=cp.cuda.Stream(non_blocking=True)
    with stream:
        stream.begin_capture()
        for _ in range(100):fn()
        graph=stream.end_capture()
        a,b=cp.cuda.Event(),cp.cuda.Event()
        a.record();graph.launch(stream);b.record();b.synchronize()
    return cp.cuda.get_elapsed_time(a,b)/100
result={'input':str(src),'input_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),
        'columns':n,'levels':nz,'scope':'CUDA graph of 100 unchanged production kernels, real 14 UTC forecast profiles; driver packing excluded',
        'device':cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
        'milliseconds':{}}
# Diagnostic-only attribution: retain every path integral and publish its
# intermediates instead of invoking the radiative solve. Never a forecast arm.
source=module_source('farms')
needle='farms_farms(p[i],alb[i],ssa[i],g[i],cz[i],beta,ang[i],pmw,tc,ti,ts,cwp>0.f?FDIV(ncf,cwp):0.f,c,ice,snow,(int)julian,out[i],out[2*n+i],out[3*n+i],out[n+i],out[4*n+i],out[5*n+i]);'
assert source.count(needle)==1
replacement='out[i]=pmw; out[n+i]=tc; out[2*n+i]=ti; out[3*n+i]=ts; out[4*n+i]=cwp>0.f?FDIV(ncf,cwp):0.f; out[5*n+i]=c+ice+snow+beta;'
from woof.core.kernels import module_options
part=cp.RawModule(code=source.replace(needle,replacement),options=module_options('farms')).get_function('farms_column')
result['milliseconds']['full']=timing(lambda:k(((n+63)//64,),(64,),args))
result['milliseconds']['paths_only_diagnostic']=timing(lambda:part(((n+63)//64,),(64,),args))
result['milliseconds']['swint1']=timing(lambda:interp(((n+127)//128,),(128,),
    (np.int64(n),cz,cz,out[1],zero,one,out[0],zero,one,alb,*io)))
bird=source[source.index('__device__ void farms_bird('):source.index('__device__ void farms_watermodel(')]
result['native_work']={'integrated_levels':nz-1,'clear_sky_Bird_calls_per_daylight_column':11,
 'Bird_pow_calls_per_call':bird.count('gfk_pow('),'Bird_exp_calls_per_call':bird.count('gfk_exp('),
 'diagnostic_variant':'same integrated paths, radiative solver replaced by six intermediate writes, not shipped',
 'additional_cloud_solvers':'water, ice, snow optical paths and conditional transmissivity'}
(root/'evidence/resume/farms-attribution.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
