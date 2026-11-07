"""Batched member initialization on the GPU, in the CPU reference's arithmetic.

An ensemble's members are initialized from their WPS intermediate files in
one batch on one card: horizontal interpolation here (``horizontal_batch``),
then vertical interpolation, hydrostatic adjustment, winds and soil/surface
filling in :mod:`woof.hex.init_gpu.physics` (``physics_batch``).  Every
kernel follows ``rw_mpas_init``'s operation order (rw-mpas
``src/init/hinterp.rs``, ``vinterp.rs``, ``dynamics.rs``, ``soil.rs``) and is
compiled without contraction or flush-to-zero.

THE BREAKAGE THIS PREVENTS (measured 2026-10-14, 36-member ensembles on one
96 GB card): initializing members one ``rw_mpas_init`` process at a time took
66.8 s for a 167,266-cell storm-scale mesh and 275.2 s for a CONUS 5 km mesh;
the batch takes 20.5 s and 68.4 s.  All 139 init fields of all 72 members
were bitwise against ``rw_mpas_init`` (31bb2034).

The claim is per case: that all-field identity gate (every field of every
member against ``rw_mpas_init`` run on the same inputs) is the caller's, and
must be re-run when the reference engine, the math headers or a numerical
setting changes.  The caller also reserves the card; on a shared host that is
its card queue's grant (``Kernels`` does not look for one).
"""
from __future__ import annotations

from woof.backend import compilation_target as _compilation_target
from woof.hex import numerics as _hex_numerics
from dataclasses import dataclass
from pathlib import Path
import struct
import time
import numpy as np

SURFACE = np.float32(200100.)

@dataclass
class Slab:
    field: str
    level: np.float32
    nx: int
    ny: int
    projection: tuple
    values: np.ndarray


def read_met(path):
    """Read WPS v5 in original record order, without re-decoding GRIB."""
    result = []
    with open(path, 'rb') as stream:
        def record(eof=False):
            size=stream.read(4)
            if not size and eof:return None
            if len(size)!=4:raise ValueError('truncated WPS record marker')
            n=struct.unpack('>I',size)[0]
            if n>1<<30:raise ValueError('WPS record exceeds 1 GiB')
            data=stream.read(n); tail=stream.read(4)
            if len(data)!=n or tail!=size:raise ValueError('invalid WPS record framing')
            return data
        while (version:=record(True)) is not None:
            if version!=struct.pack('>i',5):raise ValueError('only WPS v5 is supported')
            h=record()
            if len(h)!=156:raise ValueError('invalid WPS v5 header length')
            field=h[60:69].decode().strip();field='GHT' if field=='HGT' else field
            level,nx,ny,proj=struct.unpack('>fiii',h[140:])
            if proj!=0 or nx<3 or ny<1:raise ValueError(f'unsupported source geometry {field} {proj} {nx} {ny}')
            p=record()
            if len(p)!=28:raise ValueError('invalid latlon projection block')
            geo=struct.unpack('>5f',p[8:])[:4]
            record()  # is_wind_grid_rel, as in the CPU reader
            data=record()
            if len(data)!=nx*ny*4:raise ValueError('source slab length mismatch')
            result.append(Slab(field,np.float32(level),nx,ny,geo,np.frombuffer(data,dtype='>f4').astype('f4')))
    if not result:raise ValueError('empty intermediate')
    return result


PROFILE = {'TT':'t','RH':'rh','SPECHUMD':'sh','GHT':'z','PRES':'p','PRESSURE':'p','UU':'u','VV':'v'}
SURFACE_FIELDS = {'PSFC':'psfc','PMSL':'pmsl','SOILHGT':'soilz','SKINTEMP':'skintemp','SNOW':'snow','SEAICE':'xice'}
SOIL = {'000010':(0,10.),'010200':(1,190.),'010040':(1,30.),'040100':(2,60.),'100200':(3,100.),
        '000007':(0,7.),'007028':(1,21.),'028100':(2,72.),'100255':(3,155.),'100289':(3,189.)}
SOIL_LONG = {f'{label:03d}':(i,dz) for i,(label,dz) in enumerate(zip([1,2,6,18,54,162,486,999],[1.,2.,6.,18.,54.,162.,486.,1458.]))}


def rule(field):
    # Methods are nibbles, first operation in the low nibble.
    if field in PROFILE:return PROFILE[field],0x41,-1.,-1,0.,None
    if field in SURFACE_FIELDS:
        method=0x32 if field=='SNOW' else 0x432 if field=='SEAICE' else 0x41
        return SURFACE_FIELDS[field],method,1. if field=='SEAICE' else -1.,1 if field=='SEAICE' else -1,0.,None
    for prefix,mapping,target,method,fill in [('ST',SOIL,'st',0x4321,285.),('SM',SOIL,'sm',0x432,1.),
                                              ('SOILT',SOIL_LONG,'st',0x4321,285.),('SOILM',SOIL_LONG,'sm',0x432,1.)]:
        if field.startswith(prefix) and field[len(prefix):] in mapping:
            layer,depth=mapping[field[len(prefix):]]
            return target,method,0.,0,fill,(layer,depth)
    return None


class Kernels:
    """NVRTC-compiled init kernels on the current CUDA device (strict IEEE: no FMA, no flush-to-zero)."""

    def __init__(self):
        import cupy as cp
        self.cp=cp
        self.source=Path(__file__).with_name('horizontal.cu').read_text()
        self.options=_hex_numerics.options(('--std=c++17','--fmad=false','--ftz=false','--prec-div=true','--prec-sqrt=true'))
        self.modules={}
    def compile(self,source):
        # RawModule's cache compiler appends -ftz=true after caller options in
        # CuPy 14. Call NVRTC directly so --ftz=false is actually respected.
        from cupy.cuda.compiler import compile_using_nvrtc
        from cupy.cuda.function import Module
        binary,_=compile_using_nvrtc(source,options=self.options,
            arch=_compilation_target(xp=self.cp),cache_in_memory=True)
        module=Module();module.load(binary)
        return module
    def module(self,cap=256):
        if cap not in self.modules:
            self.modules[cap]=self.compile(f'#define SEARCH_CAP {cap}\n'+self.source)
        return self.modules[cap]
    def sample(self,values,offsets,meta,proj,lat,lon,land):
        cp=self.cp;npnt=lat.size;ns=len(meta)
        offsets=cp.asarray(offsets,dtype='i8');meta=cp.asarray(meta,dtype='i4');proj=cp.asarray(proj,dtype='f4')
        out=cp.empty((ns,npnt),dtype='f4');search=cp.zeros(out.shape,dtype='u1');n=out.size
        self.module().get_function('horizontal')(((n+255)//256,),(256,),
            (values,offsets,meta,proj,lat,lon,land,np.int32(npnt),np.int32(ns),out,search))
        indices=cp.flatnonzero(search).astype('i8');count=indices.size;retries=[]
        cap=256
        while indices.size:
            status=cp.zeros(indices.size,dtype='u1')
            self.module(cap).get_function('search_horizontal')(((indices.size+63)//64,),(64,),
                (values,offsets,meta,proj,lat,lon,np.int32(npnt),indices,np.int32(indices.size),out,status))
            indices=indices[status!=0]
            retries.append(dict(capacity=cap,overflow=int(indices.size)))
            if indices.size:
                cap*=4
                if cap>262144:
                    raise RuntimeError(f'horizontal BFS needs more than 262144 visited sites at {indices.size} points')
        return out,dict(search_points=count,search_batches=retries)


def horizontal_batch(kernels,met_members,mesh,nfg,nsoil,underflow='preserve'):
    """One batched launch per cell/edge grid, followed by exact GPU search retries."""
    cp=kernels.cp;started=time.monotonic();nb=len(met_members)
    if underflow not in ('preserve','reproduce-ifx-ftz'):raise ValueError(underflow)
    nc,ne=len(mesh['latCell']),len(mesh['latEdge'])
    result={k:cp.zeros((nb,n,levels),dtype='f4') for k,n,levels in
            [(k,nc,nfg) for k in ('t','rh','sh','z','p')]+[(k,ne,nfg) for k in ('u','v')]+[(k,nc,nsoil) for k in ('st','sm')]}
    result.update({k:cp.zeros((nb,nc),dtype='f4') for k in ('psfc','pmsl','soilz','skintemp','snow','xice','u10','v10')})
    arrays=[];offset=0;groups={False:[],True:[]};level_tables=[];depths=[];ignored=[]
    for b,slabs in enumerate(met_members):
        mask=next((s for s in slabs if s.field=='LANDSEA'),None)
        if mask is None:raise ValueError('LANDSEA missing')
        maskoff=offset;arrays.append(mask.values);offset+=mask.values.size
        levels=[];dz=np.zeros(nsoil,dtype='f4')
        for s in slabs:
            r=rule(s.field)
            if r is None:
                if s.field!='LANDSEA':ignored.append(s.field)
                continue
            target,methods,mv,flag,fill,soil=r
            if mv>=0 and (s.nx,s.ny,s.projection)!=(mask.nx,mask.ny,mask.projection):
                raise ValueError('mask and field geometry differ')
            lev=None
            if s.field in PROFILE:
                if s.level not in levels:levels.append(s.level)
                lev=levels.index(s.level)
                if len(levels)>nfg:raise ValueError('more source levels than declared')
            edge=target in ('u','v') and s.level!=SURFACE
            if target in ('u','v') and not edge:target+='10';lev=None
            if soil is not None:
                lev,depth=soil
                if lev>=nsoil:raise ValueError('more source soil levels than declared')
                dz[lev]=depth
            groups[edge].append((b,target,lev,(offset,maskoff),(s.nx,s.ny,methods,flag,int(underflow!='preserve')),
                                 (*s.projection,mv,fill)))
            arrays.append(s.values);offset+=s.values.size
        if SURFACE not in levels:raise ValueError('surface level missing')
        if np.any(dz<=0):raise ValueError('incomplete soil column')
        level_tables.append(levels);depths.append(dz)
    if any(x!=level_tables[0] for x in level_tables[1:]):raise ValueError('member level order differs')
    values=cp.asarray(np.concatenate(arrays));receipts=[]
    for edge,rows in groups.items():
        if not rows:continue
        lat=cp.asarray(mesh['latEdge' if edge else 'latCell'],dtype='f4')
        lon=cp.asarray(mesh['lonEdge' if edge else 'lonCell'],dtype='f4')
        land=cp.asarray(mesh['landmask'],dtype='i4')
        vals,receipt=kernels.sample(values,[r[3] for r in rows],[r[4] for r in rows],[r[5] for r in rows],lat,lon,land)
        for i,(b,target,lev,*_) in enumerate(rows):
            if lev is None:result[target][b]=vals[i]
            else:result[target][b,:,lev]=vals[i]
        receipts.append(dict(edge=edge,slabs=len(rows),**receipt))
        del vals
    result['vert_level']=np.asarray(level_tables[0],dtype='f4');result['dzs_cm']=np.asarray(depths,dtype='f4')
    cp.cuda.get_current_stream().synchronize()
    return result,dict(seconds=time.monotonic()-started,members=nb,groups=receipts,ignored=sorted(set(ignored)))
