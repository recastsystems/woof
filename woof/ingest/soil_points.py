"""Native point sampling for a registered soil initialization geometry."""
import ctypes
import importlib.util
import os
import numpy as np


def sample_points(values, depths, targets):
    a=np.ascontiguousarray(values,dtype=np.float32)
    z=np.ascontiguousarray(depths,dtype=np.float32)
    t=np.ascontiguousarray(targets,dtype=np.float32)
    if a.ndim < 2 or z.ndim != 1 or t.ndim != 1 or len(z) != len(a) or len(z) < 2 or not len(t):
        raise ValueError('soil point samples require a profile and two ordered source depths')
    if not np.isfinite(z).all() or not np.isfinite(t).all() or np.any(z[1:]<=z[:-1]) or np.any((t<z[0])|(t>z[-1])):
        raise ValueError('soil source depths must be ordered and bracket every target depth')
    from woof.ingest.cpu_backend import _selected_cpu_bridge, shared_cpu_backend
    cpu=(_selected_cpu_bridge.get() is not None or os.environ.get('GPUWM_NO_LOCAL_GPU')=='1'
         or os.environ.get('CUDA_VISIBLE_DEVICES')=='' or importlib.util.find_spec('cupy') is None)
    out=np.empty((len(t),)+a.shape[1:],dtype=np.float32)
    columns=int(np.prod(a.shape[1:]))
    if cpu:
        native=shared_cpu_backend()
        try:
            call=native._library.gpuwm_soil_points_f32
        except AttributeError as exc:
            raise RuntimeError('CPU preprocessing bridge lacks registered soil point sampling; rebuild tools/grib1_bridge') from exc
        call.argtypes=[ctypes.c_void_p]*4+[ctypes.c_size_t]*3
        call.restype=ctypes.c_int32
        code=call(a.ctypes.data,z.ctypes.data,t.ctypes.data,out.ctypes.data,len(z),len(t),columns)
        native._raise(code,'soil point sampling')
        return out
    import cupy as cp
    from woof.core.kernels import get_kernel
    inputs,z_cuda,t_cuda=(cp.asarray(v) for v in (a,z,t))
    result=cp.empty(out.shape,dtype=cp.float32)
    get_kernel('soil_points','soil_points')(((columns+255)//256,), (256,),
        (inputs,z_cuda,t_cuda,result,np.int32(len(z)),np.int32(len(t)),np.int32(columns)))
    return result.get()
