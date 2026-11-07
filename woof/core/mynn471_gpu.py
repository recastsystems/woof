"""Source-pinned WRF 4.7.1 MYNN column solver.

The default WRF 4.6.1 route does not import this module. Source and schema
are generated offline from WRF's public-domain module; native Fortran is
used only by the retained replay recipe, never by a forecast.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

DATA = Path(__file__).parents[1]/'data'/'mynn471'
BUDGET = ('qshear', 'qbuoy', 'qdiss', 'qwt', 'dqke')
PLUMES = ('edmf_a', 'edmf_w', 'edmf_thl', 'edmf_qt', 'edmf_ent', 'edmf_qc',
          'sub_thl3d', 'sub_sqv3d', 'det_thl3d', 'det_sqv3d')
PHASES = (0, 1, 7, 8, 9, 3, 4, 5, 6)


@lru_cache(maxsize=32)
def schema(nz):
    """Specialize affine array offsets; no CUDA import during planning."""
    first=json.loads((DATA/'schema49.json').read_text())
    second=json.loads((DATA/'schema50.json').read_text())
    def affine(x,y): return x+(y-x)*(int(nz)-49)
    first['nz']=int(nz)
    for name,field in first['arguments'].items():
        if field['type']=='character': continue
        for key in ('count','offset'):
            field[key]=affine(field[key],second['arguments'][name][key])
    for key in first['slab_fields']:
        first['slab_fields'][key]=affine(first['slab_fields'][key],second['slab_fields'][key])
    first['scratch_bytes']=affine(first['scratch_bytes'],second['scratch_bytes'])
    return first


def scratch_shapes(ncol,nz):
    meta=schema(nz)
    # DomainState's shared arena prices FP32 words, including reinterpreted
    # integer and byte storage. Every slot is bounded by the column chunk.
    return {f'mynn471_{slab}': (count*int(ncol),)
            for slab,count in meta['slab_fields'].items()} | {
        'mynn471_work': (meta['scratch_bytes']//4*int(ncol),)}


def diagnostic_shapes(cfg):
    if int(cfg.bl_pbl_physics)!=5: return {}
    result={}
    if cfg.tke_budget:
        for name in BUDGET:
            result[name]=(cfg.nz+(name!='dqke'),cfg.ny,cfg.nx)
    if cfg.bl_mynn_output:
        result.update({name:(cfg.nz,cfg.ny,cfg.nx) for name in PLUMES})
    return result


def source(nz):
    from woof.core.kernels import module_source
    old=module_source('mynn_pbl')
    prefix=old[:old.index('extern "C" __global__')]
    start=old.index('__device__ __forceinline__ real mynn_scale_exponent(')
    end=old.index('// Fortran real**real,',start)
    return (prefix+old[start:end]+FAST_MATH+'\n#define MYNN471_NZ '+str(int(nz))+'\n'
            +(DATA/'model.cu').read_text()+POST_SOURCE)


# Strict mode retains the referee's libm operation order. The regular
# forecast uses CUDA's single-precision elementary functions, as requested
# by fast numerics, rather than paying for the strict helpers' FP64 tables.
FAST_MATH = r'''
#if !GPUWM_WRF_EXACT
#define mynn_glibc_powf powf
#define mynn_glibc_expf expf
#define mynn_glibc_logf logf
#define mynn_glibc_log10f log10f
#define mynn_glibc_atanf atanf
#define mynn_tanhf tanhf
#endif
'''


@lru_cache(maxsize=32)
def image(nz,device,exact):
    import cupy as cp
    text=source(nz)
    from woof.hip_backend import active
    if active():
        # AMD backend: nvcc cannot build a gfx code object. The same source and
        # arithmetic controls go through the shared HIP route (prebuilt lookup,
        # then hiprtc). CUDA keeps the nvcc image route below unchanged.
        from woof.nvrtc_ptx_cache import compile_using_nvrtc
        options=('-std=c++17','--ftz=false','--fmad='+('false' if exact else 'true'),
                 '--prec-div=true','--prec-sqrt=true')+(('-DGPUWM_WRF_EXACT=1',) if exact else ())
        with cp.cuda.Device(device):
            raw,_=compile_using_nvrtc(text,options,None,'mynn471.cu')
            module=cp.cuda.function.Module()
            module.load(raw if isinstance(raw,bytes) else raw.encode())
        names=['mynn471_step'] if exact else ['mynn471_phase'+str(i) for i in PHASES]
        return module,tuple(module.get_function(name) for name in names)
    arch='sm_'+''.join(map(str,cp.cuda.Device(device).compute_capability))
    mode='strict' if exact else 'fast'
    digest=hashlib.sha256(text.encode()).hexdigest()
    filename=f'{arch}-nz{nz}-{mode}-{digest[:16]}.cubin'
    prebuilt=DATA/'images'/filename
    cache=Path(os.environ.get('XDG_CACHE_HOME',str(Path.home()/'.cache')))/'woof'/'mynn471'
    binary=prebuilt if prebuilt.exists() else cache/filename
    if not binary.exists():
        nvcc=shutil.which('nvcc') or '/usr/local/cuda/bin/nvcc'
        if not Path(nvcc).exists():
            raise RuntimeError(f'MYNN 4.7.1 needs its source-bound {arch} nz={nz} {mode} image; '
                               'run the parallel MYNN image prewarm with CUDA nvcc installed')
        cache.mkdir(parents=True,exist_ok=True)
        unit=cache/(filename+'.cu')
        unit.write_text(text)
        temporary=cache/(filename+f'.{os.getpid()}.tmp')
        command=[nvcc,'--cubin','-std=c++17','-arch='+arch,'--ftz=false',
                 '--fmad='+('false' if exact else 'true'),'--prec-div=true',
                 '--prec-sqrt=true','--split-compile=4']
        if exact:command+=['-DGPUWM_WRF_EXACT=1']
        subprocess.run([*command,str(unit),'-o',str(temporary)],check=True)
        temporary.replace(binary)
    # The filename binds the complete numerical source, math helpers,
    # vertical size, architecture and arithmetic mode.
    raw=binary.read_bytes()
    receipt=binary.with_suffix('.json')
    if receipt.exists():
        manifest=json.loads(receipt.read_text())
        if manifest['source_sha256']!=digest or manifest['image_sha256']!=hashlib.sha256(raw).hexdigest():
            raise RuntimeError('MYNN source-bound image receipt does not match '+str(binary))
    elif binary==prebuilt:
        raise RuntimeError('MYNN prebuilt image is missing its source receipt: '+str(binary))
    module=cp.cuda.function.Module()
    module.load(raw)
    names=['mynn471_step'] if exact else ['mynn471_phase'+str(i) for i in PHASES]
    return module,tuple(module.get_function(name) for name in names)


PACK_SOURCE = r'''
template<typename T> __device__ T mynn471_read(const char*p,int type){
 switch(type){
  case 0:return (T)*(const float*)p;
  case 1:return (T)*(const int*)p;
  case 2:return (T)*(const unsigned char*)p;
  case 3:return (T)*(const double*)p;
  case 4:return (T)*(const long long*)p;
  case 5:return (T)*(const unsigned int*)p;
  case 6:return (T)*(const unsigned long long*)p;
 }return (T)0;
}
extern "C" __global__ void mynn471_pack(int n,const unsigned long long *table,
 float *fp,int *ip){
 const unsigned long long *m=table+8*blockIdx.y;
 int k=blockIdx.x*blockDim.x+threadIdx.x;
 if(k>=n*m[2])return;
 int c=k%n,z=k/n;
 unsigned int bits=(unsigned int)m[0];
 unsigned long long dst=m[1]*n+k;
 if(m[5]==0){
   if(m[3]==0)fp[dst]=__uint_as_float(bits);else ip[dst]=(int)bits;
 }else{
   const char *p=(const char*)m[0]+c*m[6]+z*m[7];
   if(m[3]==0)fp[dst]=mynn471_read<float>(p,m[4]);
   else ip[dst]=mynn471_read<int>(p,m[4]);
 }
}
'''


@lru_cache(maxsize=8)
def pack_kernel(device):
    import cupy as cp
    return cp.RawKernel(PACK_SOURCE,'mynn471_pack',options=('-std=c++17',))


def pack_inputs(meta,slabs,arena,values,aliases,scalars,n):
    """Pack all fields in one launch, preserving each input's actual strides."""
    import cupy as cp
    fields=[(name,f) for name,f in meta['arguments'].items() if f['type']!='character']
    table=np.zeros((len(fields),8),dtype=np.uint64)
    codes={np.dtype(t):i for i,t in enumerate(('f4','i4','bool','f8','i8','u4','u8'))}
    for j,(name,f) in enumerate(fields):
        value=values.get(name,values.get(aliases.get(name,name),scalars.get(name,0)))
        table[j,1:4]=(f['offset'],f['count'],f['slab']=='i32')
        if isinstance(value,cp.ndarray):
            if value.ndim>2:raise ValueError('MYNN input rank exceeds two: '+name)
            table[j,0]=value.data.ptr
            table[j,4]=codes[value.dtype]
            table[j,5]=1
            if value.ndim:table[j,6]=value.strides[0] if value.shape[0]>1 else 0
            if value.ndim==2:table[j,7]=value.strides[1] if value.shape[1]>1 else 0
        else:
            dtype=np.float32 if f['slab']=='f32' else np.int32
            table[j,0]=np.asarray(value,dtype=dtype).view(np.uint32).item()
    # The numerical kernel has not used its work arena yet. Its first bytes
    # carry this transient table, consumed on the same stream before reuse.
    metadata=arena[:table.nbytes].view(cp.uint64).reshape(table.shape)
    metadata.set(table)
    width=n*max(f['count'] for _,f in fields)
    pack_kernel(cp.cuda.Device().id)(((width+127)//128,len(fields)),(128,),
                                    (np.int32(n),metadata,slabs['f32'],slabs['i32']))


def driver(values,*,initflag,delt,scratch=None,closure=2.6,**options):
    import cupy as cp
    n,nz=values['dz'].shape
    meta=schema(nz)
    def buffer(slot,count):
        if scratch is None:return cp.empty(count,dtype=cp.float32)
        return scratch.one(slot,(count,))
    slabs={key:buffer('mynn471_'+key,count*n).reshape(count,n)
           for key,count in meta['slab_fields'].items()}
    slabs['i32']=slabs['i32'].view(cp.int32)
    aliases={'dz1':'dz','u1':'u','v1':'v','w1':'w','th1':'th','p1':'p',
             'ex1':'exner','rho1':'rho','tk1':'tk','kh1':'exch_h','km1':'exch_m','pattern_spp_pbl1':'rstoch'}
    aliases.update({name+'1':name for name in
                    ('sqv','sqc','sqi','sqs','qnc','qni','qnwfa','qnifa','qnbca',
                     'ozone','qke','qke_adv','el','sh','sm','tsq','qsq','cov',
                     'qc_bl','qi_bl','cldfra_bl')})
    scalars=dict(i=1,j=1,initflag=initflag,delt=delt,kts=1,kte=nz,nchem=1,
                 kdvel=1,ndvel=1,closure=closure,bl_mynn_cloudpdf=2,
                 bl_mynn_mixlength=1,bl_mynn_edmf=1,bl_mynn_edmf_mom=1,
                 bl_mynn_edmf_tke=0,bl_mynn_mixscalars=0,bl_mynn_output=0,
                 bl_mynn_cloudmix=1,bl_mynn_mixqt=0,icloud_bl=1,tke_budget=0,
                 flag_qc=True,flag_qi=True)
    scalars.update(options)
    arena=buffer('mynn471_work',n*meta['scratch_bytes']//4).view(cp.uint8)
    pack_inputs(meta,slabs,arena,values,aliases,scalars,n)
    exact=os.environ.get('GPUWM_WRF_EXACT')=='1'
    module,kernels=image(nz,cp.cuda.Device().id,exact)
    if os.environ.get('MYNN471_PHASE_PROOF')=='1':
        kernels=tuple(module.get_function('mynn471_phase'+str(i)) for i in PHASES)
    phases=PHASES if len(kernels)>1 else (-1,)
    threads=32 if exact else 128
    launch_args=(np.int32(n),slabs['f32'],slabs['i32'],arena)
    if len(kernels)>1 and scratch is not None:
        owner=getattr(scratch,'_mynn471_graph_owner',scratch)
        graphs=getattr(owner,'_mynn471_graphs',None)
        if graphs is None:
            graphs={};owner._mynn471_graphs=graphs
        key=(n,nz,exact,cp.cuda.Device().id,arena.data.ptr,
             slabs['f32'].data.ptr,slabs['i32'].data.ptr)
        graph=graphs.get(key)
        if graph is None:
            stream=cp.cuda.Stream(non_blocking=True)
            with stream:
                stream.begin_capture()
                for phase,kernel in zip(phases,kernels):
                    grid=((n+threads-1)//threads,8) if phase==8 else ((n+threads-1)//threads,)
                    kernel(grid,(threads,),launch_args)
                graph=stream.end_capture()
            if len(graphs)>=4:graphs.clear()
            graphs[key]=graph
        graph.launch(stream=cp.cuda.get_current_stream())
    else:
        for phase,kernel in zip(phases,kernels):
            grid=((n+threads-1)//threads,8) if phase==8 else ((n+threads-1)//threads,)
            kernel(grid,(threads,),launch_args)
    def get(name):
        f=meta['arguments'][name]
        result=slabs[f['slab']][f['offset']:f['offset']+f['count']]
        return result.T if f['dims'] else result[0]
    out={name:get(name+'1') for name in
         ('qke','qke_adv','el','sh','sm','tsq','qsq','cov','qc_bl','qi_bl','cldfra_bl')}
    out.update(exch_h=get('kh1'),exch_m=get('km1'),sqv=get('sqv1'))
    for native,target in [('du','rublten'),('dv','rvblten'),('dth','rthblten'),
                          ('dqv','rqvblten'),('dqc','rqcblten'),('dqi','rqiblten'),
                          ('dqs','rqsblten'),('dqnc','rqncblten'),('dqni','rqniblten'),
                          ('dqnwfa','rqnwfablten'),('dqnifa','rqnifablten'),
                          ('dqnbca','rqnbcablten'),('dozone','dozone')]:
        out[target]=get(native+'1')
    out.update({name:get(name) for name in ('pblh','kpbl','maxwidth','maxmf','ztop_plume')})
    # WRF 4.7.1 does not export these two old wrapper diagnostics.
    out['rmol']=values['rmol']
    out['ktop_plume']=cp.zeros(n,dtype=cp.int32)
    if options.get('tke_budget',0):
        out.update({name:get(name+'1') for name in BUDGET})
    if options.get('bl_mynn_output',0):
        out.update({name:get(name.replace('3d','')+'1') for name in PLUMES})
    return out


# WRF 4.7.1 module_bl_mynnedmf_driver.F:878-953 and :666-673.
POST_SOURCE = r'''
extern "C" __global__ void mynn471_post(int count,int nz,int scol,int slev,
 int ocol,int olev,float dt,int cloud,
 const float*qv,const float*qc,const float*qi,const float*sqv,
 float*dqv,float*dqc,float*dqi,float*qcbl,float*qibl){
 int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=count)return;
 int c=k/nz,z=k%nz,s=c*scol+z*slev,o=c*ocol+z*olev;
 float den=1.f+qv[s];
 float sv=qv[s]/den;
 float next=sv+dqv[o]*dt;
 float rq=next/(1.f-next);
 dqv[o]=(rq-qv[s])/dt;
 float sc=qc[s]/den;
 sc=sc+dqc[o]*dt;
 rq=sc*(1.f+next);
 dqc[o]=(rq-qc[s])/dt;
 float si=qi[s]/den;
 si=si+dqi[o]*dt;
 rq=si*(1.f+next);
 dqi[o]=(rq-qi[s])/dt;
 if(cloud){qcbl[o]=qcbl[o]/(1.f-sqv[o]);qibl[o]=qibl[o]/(1.f-sqv[o]);}
}
'''


def post_run(stage,out,delt,*,icloud):
    import cupy as cp
    n,nz=stage['qv'].shape
    module,_=image(nz,cp.cuda.Device().id,os.environ.get('GPUWM_WRF_EXACT')=='1')
    kernel=module.get_function('mynn471_post')
    count=n*nz
    kernel(((count+127)//128,),(128,),
           (np.int32(count),np.int32(nz),
            *(np.int32(s//4) for s in stage['qv'].strides),
            *(np.int32(s//4) for s in out['sqv'].strides),
            np.float32(delt),np.int32(icloud),
            stage['qv'],stage['qc'],stage['qi'],out['sqv'],
            out['rqvblten'],out['rqcblten'],out['rqiblten'],out['qc_bl'],out['qi_bl']))
