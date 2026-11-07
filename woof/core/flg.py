"""Pinned WRF Fu-Liou-Gu coefficients, components and full CUDA column."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

TABLE_ROOT=Path(__file__).resolve().parents[1]/'data'/'wrf_radiation'/'flg'
SOURCE_SHA256='c4977cdf2ddf3f2279c9be7e1fc659ef23bbb37376e5cf11f9a98a73d33b0266'
MANIFEST_SHA256='06c467ffbe95732b72dc32075a984233fda8ebe3207fe5d04164937fbc7492b5'
TABLE_SHA256='d93bdc9b5ddd33cb6ca61704fb5cddf768f0d922244e07169d28905f6a3b3c23'

def load_tables(root=TABLE_ROOT):
    root=Path(root)
    metadata=(root/'manifest.json').read_bytes()
    if hashlib.sha256(metadata).hexdigest()!=MANIFEST_SHA256:
        raise ValueError('FLG table manifest differs from the pinned extraction')
    manifest=json.loads(metadata)
    data=(root/'tables.bin').read_bytes()
    if (manifest['format']!='flg-fp32-tables-v1' or manifest['dtype']!='<f4' or manifest['order']!='F'
            or manifest['source_sha256']!=SOURCE_SHA256 or manifest['sha256']!=TABLE_SHA256
            or len(data)!=manifest['bytes'] or hashlib.sha256(data).hexdigest()!=TABLE_SHA256):
        raise ValueError('FLG coefficient bytes differ from the pinned WRF extraction; changed k-distribution or optics would change fluxes')
    tables={}; offset=0
    for name,row in manifest['tables'].items():
        count=row['count']; shape=tuple(row['shape'])
        if row['offset']!=offset or count!=(int(np.prod(shape)) if shape else 1):
            raise ValueError('FLG table manifest has an invalid shape or offset')
        chunk=data[offset:offset+4*count]
        if hashlib.sha256(chunk).hexdigest()!=row['sha256']:
            raise ValueError('FLG coefficient table hash differs: '+name)
        tables[name]=np.frombuffer(chunk,dtype='<f4').reshape(shape,order='F')
        offset+=4*count
    if offset!=len(data):
        raise ValueError('FLG table manifest does not cover the coefficient file')
    return tables

def column_primitives(pressure,temperature,coszen,surface_temperature,*,table_root=TABLE_ROOT):
    """Evaluate thicks, rayle2, rayle, planck and all five ozone profiles on CUDA.

    Inputs have shape (columns,levels), pressure in mb and top to surface order.
    This does not produce a radiation tendency and is not a runtime adapter.
    """
    import cupy as cp
    from woof.core.kernels import load_module
    pp=cp.asarray(pressure,dtype=cp.float32,order='C'); pt=cp.asarray(temperature,dtype=cp.float32,order='C')
    if pp.ndim!=2 or pp.shape!=pt.shape or not 2<=pp.shape[1]<=101:
        raise ValueError('FLG primitive profiles need matching column by level arrays with 2 to 101 levels')
    n,nv1=pp.shape; nv=nv1-1
    if not n:
        raise ValueError('FLG primitive deck must contain a column')
    u0=cp.ascontiguousarray(cp.asarray(coszen,dtype=cp.float32)); pts=cp.ascontiguousarray(cp.asarray(surface_temperature,dtype=cp.float32))
    if u0.shape!=(n,) or pts.shape!=(n,):
        raise ValueError('FLG primitive deck needs one solar cosine and surface temperature per column')
    if bool(cp.any(~cp.isfinite(pp))) or bool(cp.any(pp<=0)) or bool(cp.any(cp.diff(pp,axis=1)<=0)):
        raise ValueError('FLG positive monotone pressure prevents invalid logarithms and ozone interpolation bounds')
    if bool(cp.any(~cp.isfinite(pt))) or bool(cp.any(pt<=0)) or bool(cp.any(~cp.isfinite(pts))) or bool(cp.any(pts<=0)) or bool(cp.any(~cp.isfinite(u0))) or bool(cp.any(cp.abs(u0)>1)):
        raise ValueError('FLG temperatures must be finite and positive and solar cosine finite within [-1,1]')
    tables=load_tables(table_root)
    args=[cp.asarray(tables[key].ravel(order='F'),dtype=cp.float32,order='C') for key in
          ('rayle1.ri','planck.ve','planck.nd','module_ozone.pres','module_ozone.ozone')]
    shapes={'dz':(n,nv),'trp':(n,nv),'tr':(n,18,nv),'wr':(n,18,nv),'wwr':(n,18,nv,4),
            'bf':(n,12,nv1),'bs':(n,12),'ozone':(n,5,nv1)}
    output={key:cp.empty(shape,dtype=cp.float32) for key,shape in shapes.items()}
    load_module('flg').get_function('flg_primitives')((int((n+31)//32),),(32,),
        (np.int32(n),np.int32(nv1),pp,pt,u0,pts,*args,*output.values()))
    return output


def cloud_optics(dz,liquid_radius,ice_size,liquid,ice,rain,graupel,*,table_root=TABLE_ROOT):
    """Evaluate all eight FLG ice options plus water, rain and graupel optics.

    Shapes are (columns,layers). Thickness is km, sizes microns and contents
    g/m3, as in the WRF column core. This is not a radiation driver.
    """
    import cupy as cp
    from woof.core.kernels import load_module
    inputs=[cp.asarray(value,dtype=cp.float32,order='C') for value in
            (dz,liquid_radius,ice_size,liquid,ice,rain,graupel)]
    shape=inputs[0].shape
    if len(shape)!=2 or not shape[0] or not 1<=shape[1]<=100 or any(a.shape!=shape for a in inputs):
        raise ValueError('FLG optics need matching nonempty column by layer arrays, at most 100 layers')
    if any(bool(cp.any(~cp.isfinite(a))) for a in inputs):
        raise ValueError('FLG optics need finite input operands')
    if bool(cp.any(inputs[0]<0)) or bool(cp.any(inputs[1]<0)) or bool(cp.any((inputs[4]>=np.float32(1.0e-5)) & (inputs[2]<=0))) or any(bool(cp.any(a<0)) for a in inputs[3:]):
        raise ValueError('FLG optics require nonnegative thickness and contents and positive sizes to prevent invalid extinction divisions')
    tables=load_tables(table_root)
    offsets={}; packed=[]; pos=0
    for name,value in tables.items():
        offsets[name]=pos; flat=value.ravel(order='F'); packed.append(flat); pos+=flat.size
    matrix=[]
    for mode in range(8):
        names=('ap','bp','cps','dps','cpir','bps','bpir','cp')
        required={'ap','bps','bpir','cp','dps'} if mode==5 else {'ap','bp','cps','cpir'} | ({'dps'} if mode in (0,6) else set())
        for name in required:
            if 'ice'+str(mode)+'.'+name not in offsets:
                raise ValueError('FLG ice optics are missing a required coefficient: '+str(mode)+'.'+name)
        matrix.append([offsets.get('ice'+str(mode)+'.'+name,-1) for name in names])
    for module,names in [('water1',('re','fl','bz','wz','gz')),('rain1',('rwc','brn','wrnf','grn')),('graup1',('gwc','bg','wgf','gg'))]:
        row=[offsets[module+'.'+name] for name in names]; matrix.append(row+[-1]*(8-len(row)))
    coefficients=cp.asarray(np.concatenate(packed),dtype=cp.float32)
    addresses=cp.asarray(matrix,dtype=cp.int32)
    n,nv=shape; size=n*11*18*nv
    output={'tau':cp.empty((n,11,18,nv),dtype=cp.float32),'ssa':cp.empty((n,11,18,nv),dtype=cp.float32),
            'moments':cp.empty((n,11,18,nv,4),dtype=cp.float32)}
    load_module('flg').get_function('flg_cloud_optics')((int((size+127)//128),),(128,),
        (np.int32(n),np.int32(nv),*inputs,coefficients,addresses,*output.values()))
    return output


def absorption_interpolation(coefficients,pressure,temperature,method,*,table_root=TABLE_ROOT):
    """WRF qks, qki or qkio3 for one coefficient slice per column.

    Coefficients have C shape (columns,pressure_nodes,3). The returned
    temperature contains qki's original in-place clamp, for the caller to
    carry into subsequent operations. Input arrays are preserved.
    """
    import cupy as cp
    from woof.core.kernels import load_module
    methods={'shortwave':(0,'qks.stanp',11),'longwave':(1,'qki.stanp',19),'ozone':(2,'qkio3.stanp',19)}
    if method not in methods: raise ValueError('FLG coefficient interpolation method is unknown')
    arm,key,nodes=methods[method]
    p=cp.asarray(pressure,dtype=cp.float32,order='C'); t=cp.asarray(temperature,dtype=cp.float32,order='C')
    if p.ndim!=2 or p.shape!=t.shape or not p.shape[0] or not 2<=p.shape[1]<=101:
        raise ValueError('FLG interpolation needs matching nonempty column by level profiles, 2 to 101 levels')
    c=cp.asarray(coefficients,dtype=cp.float32,order='C')
    if c.shape!=(p.shape[0],nodes,3): raise ValueError('FLG coefficient slice shape does not match the interpolation nodes')
    if bool(cp.any(~cp.isfinite(p))) or bool(cp.any(p<=0)) or bool(cp.any(cp.diff(p,axis=1)<=0)) or bool(cp.any(~cp.isfinite(t))) or bool(cp.any(t<=0)) or bool(cp.any(~cp.isfinite(c))):
        raise ValueError('FLG interpolation needs finite coefficients, positive temperature and positive increasing pressure to keep node search in bounds')
    stanp=cp.asarray(load_tables(table_root)[key],dtype=cp.float32)
    output={'fkg':cp.empty_like(p),'temperature_out':cp.empty_like(t)}
    n,nv1=p.shape
    load_module('flg').get_function('flg_absorption_interpolation')((int((n+31)//32),),(32,),
        (np.int32(n),np.int32(nv1),np.int32(arm),np.int32(nodes),c,stanp,p,t,*output.values()))
    return output

def continuum_depth(pressure,temperature,humidity,*,table_root=TABLE_ROOT):
    """WRF gascon water-vapor continuum for all 18 bands."""
    import cupy as cp
    from woof.core.kernels import load_module
    p,t,h=[cp.asarray(a,dtype=cp.float32,order='C') for a in (pressure,temperature,humidity)]
    if p.ndim!=2 or p.shape!=t.shape or p.shape!=h.shape or not p.shape[0] or not 2<=p.shape[1]<=101:
        raise ValueError('FLG continuum needs matching nonempty column by level profiles, 2 to 101 levels')
    if any(bool(cp.any(~cp.isfinite(a))) for a in (p,t,h)) or bool(cp.any(p<=0)) or bool(cp.any(cp.diff(p,axis=1)<=0)) or bool(cp.any(t<=0)) or bool(cp.any(h<0)):
        raise ValueError('FLG continuum needs positive increasing pressure and temperature and nonnegative humidity')
    vv=cp.asarray(load_tables(table_root)['gascon.vv'],dtype=cp.float32)
    n,nv1=p.shape; output=cp.empty((n,18,nv1-1),dtype=cp.float32)
    load_module('flg').get_function('flg_continuum')((int((n*18+31)//32),),(32,),
        (np.int32(n),np.int32(nv1),vv,p,t,h,output))
    return output


def gas_points(tables=None):
    """Both WRF gas arms' g-quadrature metadata, read from the coefficient asset."""
    tables=load_tables() if tables is None else tables
    rows=[]
    for arm in range(2):
        one=tables['rad_all.kg'] if arm==0 else tables['rad_all.kg1']
        two=np.ones(18,dtype=np.int32) if arm==0 else tables['rad_all.kg2'].astype(np.int32)
        for band in range(18):
            for ig1 in range(1,int(one[band])+1):
                for ig2 in range(1,int(two[band])+1): rows.append((arm,band+1,ig1,ig2))
    return np.asarray(rows,dtype=np.int32)

def gas_optical_depth(pressure,temperature,humidity,ozone,gas_values,switches,*,table_root=TABLE_ROOT):
    """Both complete WRF gases and gases_new drivers on all g points.

    Arrays use columns by levels. gas_values has 11 columns in the order
    co2,ch4,n2o,o2,no,so2,no2,ch3cl,co,cfc11,cfc12. switches has 20 columns,
    in the WRF control order documented by generate_gas_kernel.py.
    Temperature changes are returned separately for the radiation driver
    to carry forward in its original band and g-point sequence.
    """
    import cupy as cp
    from woof.core.kernels import load_module
    p,t,h,o=[cp.asarray(a,dtype=cp.float32,order='C') for a in (pressure,temperature,humidity,ozone)]
    if p.ndim!=2 or not p.shape[0] or not 2<=p.shape[1]<=101 or any(a.shape!=p.shape for a in (t,h,o)):
        raise ValueError('FLG gases need matching nonempty column by level profiles, 2 to 101 levels')
    values=cp.asarray(gas_values,dtype=cp.float32,order='C'); flags=cp.asarray(switches,dtype=cp.int32,order='C')
    n,nv1=p.shape
    if values.shape!=(n,11) or flags.shape!=(n,20): raise ValueError('FLG gas controls have incorrect column shape')
    if any(bool(cp.any(~cp.isfinite(a))) for a in (p,t,h,o,values)) or bool(cp.any(p<=0)) or bool(cp.any(cp.diff(p,axis=1)<=0)) or bool(cp.any(t<=0)) or any(bool(cp.any(a<0)) for a in (h,o,values)) or bool(cp.any((flags!=0)&(flags!=1))):
        raise ValueError('FLG gases require finite positive monotone pressure and temperature, nonnegative composition and binary gas switches')
    tables=load_tables(table_root); meta=gas_points(tables); packed=[]; offsets=[]; pos=0
    for key,value in tables.items():
        offsets.append(pos); flat=value.ravel(order='F'); packed.append(flat); pos+=flat.size
    points=cp.asarray(meta); coefficients=cp.asarray(np.concatenate(packed),dtype=cp.float32); addresses=cp.asarray(offsets,dtype=cp.int32)
    count=len(meta); size=n*count
    output={'optical_depth':cp.empty((n,count,nv1-1),dtype=cp.float32),'weights':cp.empty((n,count),dtype=cp.float32),
            'temperature_out':cp.empty((n,count,nv1),dtype=cp.float32)}
    load_module('flg_gases').get_function('flg_gas_driver')((int((size+63)//64),),(64,),
        (np.int32(n),np.int32(nv1),np.int32(count),points,p,t,h,o,coefficients,addresses,values,flags,*output.values()))
    return output


def _refuse_stream_errors(errors):
    import cupy as cp
    if bool(cp.any(errors)):
        raise RuntimeError('FLG tridiagonal stream system has a zero pivot; returning its fluxes would use an undefined linear-system solution')

def stream_fluxes(bands,albedo,coszen,f0,emissivity,bf,bs,moments,scattering,depth,*,table_root=TABLE_ROOT):
    """Four-stream SW, two-stream SW, four-stream IR and mixed two/four IR.

    This column-core check returns direct/diffuse only for qfts, which
    defines them. The unused qftits routine has uninitialized source inputs
    and is not exposed. Radiation drivers skip SW calls for night columns.
    """
    import cupy as cp
    from woof.core.kernels import load_module
    b=cp.asarray(bands,dtype=cp.int32,order='C')
    a,u,s,e,bb=[cp.asarray(v,dtype=cp.float32,order='C') for v in (albedo,coszen,f0,emissivity,bs)]
    black=cp.asarray(bf,dtype=cp.float32,order='C'); phase=cp.asarray(moments,dtype=cp.float32,order='C')
    w=cp.asarray(scattering,dtype=cp.float32,order='C'); t=cp.asarray(depth,dtype=cp.float32,order='C')
    if black.ndim!=2 or not black.shape[0] or not 2<=black.shape[1]<=101:
        raise ValueError('FLG stream profiles need nonempty column by level arrays, 2 to 101 levels')
    n,nv1=black.shape; nv=nv1-1
    if b.shape!=(n,2) or any(v.shape!=(n,) for v in (a,u,s,e,bb)) or phase.shape!=(n,4,nv) or w.shape!=(n,nv) or t.shape!=(n,nv):
        raise ValueError('FLG stream input shapes differ from the column interface')
    if any(bool(cp.any(~cp.isfinite(v))) for v in (a,u,s,e,bb,black,phase,w,t)):
        raise ValueError('FLG streams require finite radiation operands')
    if bool(cp.any((b[:,0]<1)|(b[:,0]>6))) or bool(cp.any((b[:,1]<7)|(b[:,1]>18))) or bool(cp.any(u<=np.float32(1.e-4))) or bool(cp.any(u>1)):
        raise ValueError('FLG SW streams divide by solar cosine and require a day column and an SW band; IR requires a thermal band')
    if bool(cp.any((a<0)|(a>1))) or bool(cp.any((e<0)|(e>1))) or bool(cp.any((w<0)|(w>1))) or bool(cp.any(t<0)) or bool(cp.any(cp.diff(t,axis=1)<0)) or bool(cp.any(black<=0)) or bool(cp.any(bb<=0)):
        raise ValueError('FLG streams require bounded albedo, emissivity and scattering, cumulative nonnegative depth, and positive Planck sources')
    tables=load_tables(table_root); packed=[]; offsets=[]; pos=0
    for value in tables.values():
        offsets.append(pos); flat=value.ravel(order='F'); packed.append(flat); pos+=flat.size
    coefficients=cp.asarray(np.concatenate(packed),dtype=cp.float32); addresses=cp.asarray(offsets,dtype=cp.int32)
    output={'up':cp.empty((n,4,nv1),dtype=cp.float32),'down':cp.empty((n,4,nv1),dtype=cp.float32),
            'direct':cp.empty((n,nv1),dtype=cp.float32),'diffuse':cp.empty((n,nv1),dtype=cp.float32)}
    errors=cp.empty((n,4),dtype=cp.int32)
    load_module('flg_streams').get_function('flg_stream_driver')((int((n*4+31)//32),),(32,),
        (np.int32(n),np.int32(nv1),b,a,u,s,e,black,bb,phase,w,t,coefficients,addresses,*output.values(),errors))
    _refuse_stream_errors(errors)
    return output


class FLGColumn:
    """Persistent device buffers for the WRF ``rad_all`` column.

    This is a module API, not the RAD_FLG forecast adapter. All arguments to
    ``__call__`` are contiguous CUDA float32 arrays in native FLG units.
    Tables, controls and outputs are allocated once. The launch does no
    scalar device reads or host copies. ``check_errors`` is an explicit
    proof/finalization operation outside a forecast step.
    """
    profile_names = ('pressure','temperature','humidity','ozone')
    layer_names = ('ice','ice_size','liquid','liquid_radius','rain','graupel',
                   'cloud_fraction','inhomogeneity')
    scalar_names = ('coszen','solar_constant','skin_temperature')
    output_names = ('fds','fus','dts','fdir','fuir','dtir','fd','fu','dt','direct','diffuse')

    def __init__(self, columns, layers, *, nice=2, ngas=0, table_root=TABLE_ROOT):
        import cupy as cp
        from woof.core.kernels import load_module_unflushed
        if not isinstance(columns,int) or columns<1 or not isinstance(layers,int) or not 4<=layers<=100:
            raise ValueError('FLG needs positive columns and 4 to 100 layers for its three cloud groups')
        if nice not in range(8) or ngas not in (0,1):
            raise ValueError('FLG ice and gas controls are outside the native choices')
        tables=load_tables(table_root); packed=[]; offsets=[]; pos=0
        for value in tables.values():
            offsets.append(pos); flat=value.ravel(order='F'); packed.append(flat); pos+=flat.size
        self.tables=cp.asarray(np.concatenate(packed),dtype=cp.float32)
        self.offsets=cp.asarray(offsets,dtype=cp.int32)
        # RAD_FLG zeros the additional trace gases in the old-gas arm.
        gas=np.array([345.,1.6,.28,209480.,.0005,.001,.001,.0005,.16,.00022,.000375],dtype=np.float32)
        if ngas==0: gas[3:]=0.
        self.gas_values=cp.asarray(np.tile(gas,(columns,1)))
        flags=np.ones((columns,29),dtype=np.int32)
        flags[:,20:27]=[1,0,0,1,0,0,0]
        flags[:,27]=nice; flags[:,28]=ngas
        self.controls=cp.asarray(flags)
        self.columns=columns; self.layers=layers
        self.outputs={name:cp.empty((columns,layers if name in ('dts','dtir','dt') else layers+1),dtype=cp.float32)
                      for name in self.output_names}
        self.errors=cp.empty(columns,dtype=cp.int32)
        from woof.core.kernels import module_source
        import re
        source=module_source('flg_column')
        scale=int(re.search(r'#define FLG_WORKSPACE_SCALE (\d+)',source)[1])
        bias=int(re.search(r'#define FLG_WORKSPACE_BIAS (-?\d+)',source)[1])
        words=scale*(layers+1)+bias
        self.workspace=cp.empty((((columns+31)//32)*32,words),dtype=cp.float32)
        self.kernel=load_module_unflushed('flg_column',(('FLG_NV1',layers+1),)).get_function('flg_column_driver')

    def __call__(self, **fields):
        import cupy as cp
        expected=set(self.profile_names+self.layer_names+self.scalar_names+('albedo','emissivity'))
        if set(fields)!=expected: raise ValueError('FLG column field inventory differs: '+str(sorted(set(fields)^expected)))
        for name,value in fields.items():
            width=(self.layers+1 if name in self.profile_names else self.layers if name in self.layer_names
                   else 6 if name=='albedo' else 12 if name=='emissivity' else None)
            shape=(self.columns,width) if width is not None else (self.columns,)
            if not isinstance(value,cp.ndarray) or value.dtype!=cp.float32 or value.shape!=shape or not value.flags.c_contiguous:
                raise ValueError('FLG needs contiguous CUDA float32 '+name+' with shape '+str(shape))
        order=self.profile_names+self.layer_names+('albedo','emissivity')+self.scalar_names
        self.kernel(((self.columns+31)//32,),(32,),
                    (np.int32(self.columns),np.int32(self.layers+1),*[fields[k] for k in order],
                     self.tables,self.offsets,self.gas_values,self.controls,*self.outputs.values(),self.errors,self.workspace))
        return self.outputs

    def check_errors(self):
        import cupy as cp
        errors=cp.asnumpy(self.errors)
        if np.any(errors):
            raise RuntimeError('FLG native column diagnostic: '+str(errors.tolist())+
                               ' (1 zero stream pivot, 2 unsupported offline aerosol call, 3 negative optical depth)')
