"""Native CAM data ownership and transport through the registered table root."""
from __future__ import annotations
import ctypes,os
from functools import lru_cache
from pathlib import Path
import numpy as np

CAM_TABLES_ENV = "WOOF_CAM_TABLES_LIB"
ABI_VERSION = 1


def library_name():
    import sys
    return ('rw_cam_tables.dll' if os.name == 'nt' else
            'librw_cam_tables.dylib' if sys.platform == 'darwin' else
            'librw_cam_tables.so')


def resolve():
    from woof.bridges import default_bridge_dir, packaged_bridge_dir
    root = Path(__file__).resolve().parents[2]
    name = library_name()
    override = os.environ.get(CAM_TABLES_ENV)
    if override:
        path = Path(override)
        if not path.is_file():
            raise FileNotFoundError(f"{CAM_TABLES_ENV} names a missing file: {path}")
        return path
    candidates = (root / 'tools/cam_tables/target/release' / name,
                  root / 'tools/cam_tables/target/debug' / name,
                  root / 'libexec/bridges' / name,
                  packaged_bridge_dir() / name, default_bridge_dir() / name)
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        'CAM requires the native reader to preserve Fortran records and initialized table words. '
        'Build tools/cam_tables with cargo build --release --locked or set WOOF_CAM_TABLES_LIB.')


@lru_cache(maxsize=1)
def _library():
    path = resolve()
    lib=ctypes.CDLL(str(path));handle=ctypes.c_void_p;size=ctypes.POINTER(ctypes.c_size_t)
    lib.gpuwm_cam_tables_abi_version.restype=ctypes.c_uint32
    if lib.gpuwm_cam_tables_abi_version()!=ABI_VERSION:raise RuntimeError('CAM table library ABI differs; rebuild tools/cam_tables')
    lib.cam_tables_open.argtypes=[ctypes.c_char_p];lib.cam_tables_open.restype=handle
    lib.cam_tables_close.argtypes=[handle];lib.cam_tables_close.restype=None
    lib.cam_tables_state.argtypes=[handle,size];lib.cam_tables_state.restype=ctypes.POINTER(ctypes.c_ubyte)
    lib.cam_tables_ozone.argtypes=[handle,ctypes.c_int32,size];lib.cam_tables_ozone.restype=ctypes.POINTER(ctypes.c_float)
    lib.cam_tables_gases.argtypes=[handle,ctypes.c_int32,ctypes.c_float,handle];lib.cam_tables_gases.restype=ctypes.c_int32
    lib.cam_tables_copy_state.argtypes=[handle,handle,ctypes.c_int32,handle,ctypes.c_size_t];lib.cam_tables_copy_state.restype=ctypes.c_int32
    lib.cam_tables_error.argtypes=[handle,ctypes.c_size_t];lib.cam_tables_error.restype=ctypes.c_size_t
    return lib

class Cam3Tables:
    def __init__(self,root):
        self.root=Path(root).resolve();self.library=_library();self.handle=self.library.cam_tables_open(str(self.root).encode())
        if not self.handle:
            message=ctypes.create_string_buffer(4096);self.library.cam_tables_error(message,len(message));raise RuntimeError(message.value.decode())
        n=ctypes.c_size_t();p=self.library.cam_tables_state(self.handle,ctypes.byref(n));self.state=np.ctypeslib.as_array(p,shape=(n.value,))
        arrays=[]
        for i in range(4):
            p=self.library.cam_tables_ozone(self.handle,i,ctypes.byref(n));arrays.append(np.ctypeslib.as_array(p,shape=(n.value,)))
        self.pin,self.latitude,self.ozone,self.hybi=arrays;self.ozone=self.ozone.reshape(59,64,12)
        for value in (self.state,self.pin,self.latitude,self.ozone,self.hybi):value.setflags(write=False)
    def initialized_state(self,physical,mxaerl,*,fast=False):
        physical=np.ascontiguousarray(physical,np.float64)
        if physical.shape!=(12,):raise ValueError("CAM initialization requires twelve physical inputs")
        copy=self.library.cam_tables_copy_state
        if fast:
            try:copy=self.library.cam_tables_copy_state_fast
            except AttributeError as error:
                raise RuntimeError('Fast CAM requires the REAL32 state reader; rebuild tools/cam_tables with cargo build --release --locked') from error
            copy.argtypes=self.library.cam_tables_copy_state.argtypes
            copy.restype=ctypes.c_int32
        state=np.empty_like(self.state);status=copy(self.handle,ctypes.c_void_p(physical.ctypes.data),int(mxaerl),ctypes.c_void_p(state.ctypes.data),state.size)
        if status:raise ValueError("CAM initialization requires positive finite physical inputs and a positive aerosol layer count")
        return state
    def gases(self,year,julian):
        value=np.empty(5,np.float64);status=self.library.cam_tables_gases(self.handle,int(year),np.float32(julian),ctypes.c_void_p(value.ctypes.data))
        if status:raise ValueError('CAM gas interpolation requires a finite WRF Julian day')
        return value
    def ozone_columns(self,latitude):
        from woof.ingest.wrf_ozone import _native_ozone_entries,_native_ozone_status
        from woof.core import portable_math
        entries=_native_ozone_entries()
        if entries is None:
            raise RuntimeError('CAM ozone requires the existing Rust latitude interpolation primitive; the Python fallback cannot satisfy the native data-path contract')
        values=np.ascontiguousarray(latitude,dtype=np.float32).reshape(-1);output=np.empty((len(values),59,12),np.float32)
        status=entries[1](values.ctypes.data,values.size,self.latitude.ctypes.data,self.latitude.size,self.ozone.ctypes.data,59,12,portable_math._workers(None),output.ctypes.data)
        if not _native_ozone_status(status):raise RuntimeError('Rust CAM ozone primitive does not support the supplied grid')
        return output
    def close(self):
        if self.handle:self.library.cam_tables_close(self.handle);self.handle=None


@lru_cache(maxsize=8)
def cached_tables(root):
    """Own one immutable native table image for each resolved root."""
    return Cam3Tables(Path(root).resolve())
