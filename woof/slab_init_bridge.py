"""Buffers for Rust slab soil preparation, not a model-specific source adapter."""
import ctypes as c
from functools import lru_cache
import numpy as np

@lru_cache(maxsize=4)
def _library(path):
    lib=c.CDLL(str(path))
    specs={
      'gpuwm_slab_threshold_v1': [c.c_size_t]+[c.c_void_p]*4+[c.c_double]+[c.c_void_p]*2,
      'gpuwm_slab_initialize_v1': [c.c_size_t]*2+[c.c_void_p]*2+[c.c_double]+[c.c_void_p]*6+[c.c_uint32]*2+[c.c_void_p]*2,
      'gpuwm_slab_profile_v1': [c.c_size_t]+[c.c_void_p]*5,
    }
    for name,args in specs.items():
        try: fn=getattr(lib,name)
        except AttributeError as exc:
            raise RuntimeError(f'The preparation bridge lacks {name}; rebuild tools/grib1_bridge. Slab soil must be initialized on its five-layer geometry, with its own lower boundary.') from exc
        fn.argtypes=args; fn.restype=c.c_int32
    return lib

def _call(name,*args):
    from woof.ingest.cpu_backend import resolve_cpu_bridge
    args=tuple(c.c_void_p(v.ctypes.data) if isinstance(v,np.ndarray) else v for v in args)
    code=getattr(_library(resolve_cpu_bridge()),name)(*args)
    if code: raise ValueError(f'Rust slab soil preparation rejected invalid geometry or buffers (code {code}); prevents building an undefined soil column')

def threshold_ice(decision,lake,skin,ice,threshold):
    arrays=[np.ascontiguousarray(v,np.float64) for v in (decision,lake,skin,ice)]
    if any(v.shape!=arrays[0].shape for v in arrays): raise ValueError('slab threshold fields must have the same shape')
    terr=np.empty(arrays[0].shape,np.uint8); converted=np.empty_like(arrays[0])
    _call('gpuwm_slab_threshold_v1',terr.size,*arrays,float(threshold),terr,converted)
    return terr.astype(bool),converted

def initialize(temperatures,levels,scale,terrestrial,land,skin,sst,delta,delta_zero,has_terrain,exact):
    temp=np.ascontiguousarray(temperatures,np.float64); levels=np.ascontiguousarray(levels,np.float64)
    terr=np.ascontiguousarray(terrestrial,np.uint8)
    arrays=[np.ascontiguousarray(v,np.float64) for v in (land,skin,sst,delta,delta_zero)]
    if temp.shape!=(levels.size,)+terr.shape or any(v.shape!=terr.shape for v in arrays): raise ValueError('slab source profiles and surface fields must share their column shape')
    bottom=np.empty(terr.shape,np.float64); out=np.empty((5,)+terr.shape,np.float32)
    _call('gpuwm_slab_initialize_v1',terr.size,levels.size,levels,temp,float(scale),terr,*arrays,int(has_terrain),int(exact),bottom,out)
    return bottom,out

def profile(skin,bottom,land,water):
    arrays=[np.ascontiguousarray(v,np.float32) for v in (skin,bottom)]
    terr=np.ascontiguousarray(land,np.uint8); water=np.ascontiguousarray(water,np.float32)
    if any(v.shape!=terr.shape for v in (*arrays,water)): raise ValueError('slab profile fields must have the same shape')
    out=np.empty((5,)+terr.shape,np.float32)
    _call('gpuwm_slab_profile_v1',terr.size,*arrays,terr,water,out)
    return out
