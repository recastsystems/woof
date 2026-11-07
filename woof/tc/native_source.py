"""Exact-lattice TC initial-state handoff to the native GFS preparation.

All scientific arithmetic happened in the CUDA canonical transform. Rust
copies changed source cells onto the original decoder's storage, preserving
its original bits outside the transform and preserving every later lead.
"""
import ctypes as C
import hashlib
import json
from pathlib import Path
import time
from collections.abc import Sequence

from . import frames
from .frame_transform import _axis

ALIASES={'GHT':'geopotential_height','T':'air_temperature','RH':'relative_humidity',
         'U':'eastward_wind','V':'northward_wind','PSFC':'surface_pressure',
         'T2':'air_temperature_2m','RH2':'relative_humidity_2m',
         'U10':'eastward_wind_10m','V10':'northward_wind_10m'}


def _hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _indices(source,target,longitude=False):
    def key(v):return round(float(v)%360. if longitude else float(v),9)
    lookup={key(v):i for i,v in enumerate(source)}
    if len(lookup)!=len(source):raise ValueError('duplicate TC source coordinates')
    try:return [lookup[key(v)] for v in target]
    except KeyError as exc:raise ValueError('TC source lattice differs from native input') from exc


class InitialSeries(Sequence):
    def __init__(self,original,initial):self.original,self.initial=original,initial
    def __len__(self):return len(self.original)
    def __getitem__(self,index):
        if isinstance(index,slice):return _Slice(self,range(*index.indices(len(self))))
        if index<0:index+=len(self)
        if not 0<=index<len(self):raise IndexError(index)
        return self.initial if index==0 else self.original[index]
    def __getattr__(self,name):return getattr(self.original,name)
    def surface_field(self,index,name):
        if index==0:return self.initial.fields.get(name)
        method=getattr(self.original,'surface_field',None)
        return method(index,name) if method else self.original[index].fields.get(name)


class _Slice(Sequence):
    def __init__(self,series,positions):self.series,self.positions=series,positions
    def __len__(self):return len(self.positions)
    def __getitem__(self,index):
        if isinstance(index,slice):return _Slice(self.series,self.positions[index])
        return self.series[self.positions[index]]


def apply(snapshots,request_path,*,valid_time,source_sha256):
    """Return a separate initial snapshot plus the binding for native receipts."""
    import numpy as np
    from woof.ingest.grib import Era5Snapshot
    start=time.perf_counter();request_path=Path(request_path).resolve(strict=True)
    request_bytes=request_path.read_bytes();request=json.loads(request_bytes)
    if set(request)!={'schema','before','after','library','before_sha256','after_sha256','library_sha256'} or request['schema']!='gpuwm-tc-source-init/1':
        raise ValueError('invalid native TC source declaration')
    def path(name):
        p=Path(request[name]);return (request_path.parent/p).resolve(strict=True) if not p.is_absolute() else p.resolve(strict=True)
    before,after,library=(path(n) for n in ('before','after','library'))
    for name,p in (('before',before/'frames.json'),('after',after/'frames.json'),('library',library)):
        if _hash(p)!=request[name+'_sha256']:raise ValueError(f'TC {name} identity changed')
    transform=json.loads((after/'tc-transform.json').read_text());declaration=transform['declaration']
    patch=json.loads((after/'tc-patch-receipt.json').read_text())
    transform_digest=hashlib.sha256(json.dumps(declaration,sort_keys=True,allow_nan=False).encode()).hexdigest()
    if (declaration['mode']!='relocate' or declaration['valid_time']!=valid_time
        or declaration['background_manifest_sha256']!=request['before_sha256']
        or transform['output_manifest_sha256']!=request['after_sha256']
        or patch['transform_sha256']!=transform_digest
        or patch['background_manifest_sha256']!=request['before_sha256']
        or declaration['relative_humidity_convention']!='gfs_mixed'
        or declaration['carried_fields']):
        raise ValueError('native GFS TC handoff has incompatible clock, inventory, humidity or provenance')
    prior=snapshots[0]
    if prior.valid_time.isoformat()!=valid_time or prior.projection is not None:
        raise ValueError('native TC snapshot clock or projection differs')
    lib=C.CDLL(str(library));lib.tc_frames_abi_version.restype=C.c_uint32
    if lib.tc_frames_abi_version() not in (3,4):raise RuntimeError('native TC overlay needs frame I/O ABI 3 or 4')
    lib.tc_frames_overlay.argtypes=[C.c_void_p]*3+[C.c_size_t]*3+[C.c_void_p]*3+[C.c_size_t]*3
    lib.tc_frames_overlay.restype=C.c_int;lib.tc_frames_last_error.restype=C.c_char_p
    output=dict(prior.fields);timing={}
    for native,canonical in ALIASES.items():
        tick=time.perf_counter()
        old,oldmeta,oldsha=frames.read(library=library,input_directory=before,valid_time=valid_time,fields=(canonical,),device=False)
        new,newmeta,newsha=frames.read(library=library,input_directory=after,valid_time=valid_time,fields=(canonical,),device=False)
        if oldsha!=request['before_sha256'] or newsha!=request['after_sha256']:
            raise ValueError('TC manifest changed during native insertion')
        if source_sha256 not in oldmeta['input_sha256'].values():
            raise ValueError('TC transform does not originate from this native GRIB lead')
        for name in ('latitude','longitude','vertical_values','vertical_kind','vertical_units'):
            if oldmeta[name]!=newmeta[name]:raise ValueError('TC transform changed source geometry')
        lat=_axis(oldmeta,'latitude');lon=_axis(oldmeta,'longitude');levels=_axis(oldmeta,'vertical_values')
        y=np.asarray(_indices(lat,prior.latitude),dtype=np.uintp)
        x=np.asarray(_indices(lon,prior.longitude,True),dtype=np.uintp)
        z=np.asarray(_indices(levels,[float(v)*100. for v in prior.levels_hpa]) if prior.fields[native].ndim==3 else [0],dtype=np.uintp)
        a,b=old[canonical],new[canonical];out=np.array(prior.fields[native],dtype=np.float64,order='C',copy=True)
        if a.shape!=b.shape or a.ndim!=out.ndim or a.ndim not in (2,3):raise ValueError('TC native field rank differs')
        if lib.tc_frames_overlay(a.ctypes.data,b.ctypes.data,out.ctypes.data,a.shape[-1],a.shape[-2],a.shape[0] if a.ndim==3 else 1,
            x.ctypes.data,y.ctypes.data,z.ctypes.data,len(x),len(y),len(z)):
            raise RuntimeError(lib.tc_frames_last_error().decode())
        output[native]=out;timing[native]=time.perf_counter()-tick
    initialized=Era5Snapshot(valid_time=prior.valid_time,levels_hpa=prior.levels_hpa,
        latitude=prior.latitude,longitude=prior.longitude,fields=output)
    binding={'schema':'gpuwm-native-tc-source-init/1','mode':'relocate','valid_time':valid_time,
        'request_sha256':hashlib.sha256(request_bytes).hexdigest(),'before_sha256':request['before_sha256'],
        'after_sha256':request['after_sha256'],'library_sha256':request['library_sha256'],
        'transform_sha256':transform_digest,'declaration':declaration,
        'field_io_copy_seconds':timing,'total_seconds':time.perf_counter()-start,
        'native_cells':'original bits wherever canonical before/after bits agree',
        'later_leads':'original sequence','qualification':'requires forecast verification'}
    return InitialSeries(snapshots,initialized),binding
