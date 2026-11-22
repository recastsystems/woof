"""Source-pinned WRF 4.8.0 MYNN-EDMF column solver (bl_mynn_version="wrf_480").

WRF v4.8.0 (06d4240ae989) builds MYNN-EDMF as the submodule NCAR/MYNN-EDMF
8f8e8b59, kept as WRF ships it (WRF public-domain statement; the commit has
no licence file of its own): ``module_bl_mynnedmf.F90`` sha256
4aee355aefaaa06c7d0838f97f69b64eda9d0402885c1c35b24454070df25a83 and the WRF
``module_bl_mynnedmf_common.F90`` sha256
20c0e0c7c3ee0c08e4de67a6479049ef948022c11c70cff37e1a22b0bfd398ba.

The column source is generated offline by ``tools/mynn_wrf480_port`` (the
4.7.1 recipe: lower to F77, f2c, CUDA, generalise in nz, interleave private
arrays across columns). No Fortran runtime, f2c runtime or native library is
used by a forecast. The default ``wrf_461`` and the ``wrf_471`` routes never
import this module, so their kernels and cubins are unchanged.

WRF 4.8.0's own driver (``WRF/module_bl_mynnedmf_driver.F90``) differs from
4.7.1's only in plumbing: ``icloud_bl`` is gone (the SGS cloud is always
converted to mixing ratio), seven new per-column diagnostics are written
when EDMF is on, and the pre/post specific-content conversions keep their
arithmetic. :func:`post_run` is that post-conversion.

Known upstream defect carried as-is (it is what 4.8.0 ships): the theta
tendency is still formed from the pre-mixing condensate (fixed upstream in
MYNN v6.1.0, e447691). The separate ``v61fix`` option owns that fix.
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

from woof.core.mynn471_gpu import pack_inputs, BUDGET, PLUMES

DATA = Path(__file__).parents[1]/'data'/'mynn480'
SOURCE_SHA256 = {
    'module_bl_mynnedmf.F90': '4aee355aefaaa06c7d0838f97f69b64eda9d0402885c1c35b24454070df25a83',
    'WRF/module_bl_mynnedmf_common.F90': '20c0e0c7c3ee0c08e4de67a6479049ef948022c11c70cff37e1a22b0bfd398ba',
    'WRF/module_bl_mynnedmf_driver.F90': 'b3e0646a53ad917521cf858818897c4eed019101b77686e32c8488326cee2cd4',
}
#: WRF 4.8.0 Registry.EM_COMMON defaults for the controls 4.8.0 adds or
#: changes. bl_mynn_mixscalars is 0 in 4.8.0 (it was 1 in 4.7.1).
WRF480_DEFAULTS = dict(bl_mynn_ess=1, bl_mynn_edmf_dd=0, bl_mynn_mixaerosols=1,
                       bl_mynn_mixnumcon=0, bl_mynn_mixscalars=0)
#: New 4.8.0 per-column EDMF diagnostics, written only when EDMF is on.
EDMF_2D_NEW = ('excess_h', 'excess_q', 'maxmf_dd', 'maxwidth_dd', 'maxtkeprod',
               'cldtop_cooling', 'ent_eff')
EDMF_2D = ('maxwidth', 'maxmf', 'ztop_plume', *EDMF_2D_NEW)


@lru_cache(maxsize=32)
def schema(nz):
    """Specialise affine array offsets; no CUDA import during planning."""
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
    return {f'mynn480_{slab}': (count*int(ncol),)
            for slab,count in meta['slab_fields'].items()} | {
        'mynn480_work': (meta['scratch_bytes']//4*int(ncol),)}


def diagnostic_shapes(cfg):
    if int(cfg.bl_pbl_physics)!=5: return {}
    # WRF 4.8.0 Registry.EM_COMMON package mynn_dmp_edmf1: seven new 2-D
    # history fields beside ztop_plume/maxmf/maxwidth.
    result={name:(cfg.ny,cfg.nx) for name in EDMF_2D_NEW}
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
        from woof.nvrtc_ptx_cache import compile_using_nvrtc
        options=('-std=c++17','--ftz=false','--fmad='+('false' if exact else 'true'),
                 '--prec-div=true','--prec-sqrt=true')+(('-DGPUWM_WRF_EXACT=1',) if exact else ())
        with cp.cuda.Device(device):
            raw,_=compile_using_nvrtc(text,options,None,'mynn480.cu')
            module=cp.cuda.function.Module()
            module.load(raw if isinstance(raw,bytes) else raw.encode())
        return module,module.get_function('mynn471_step')
    arch='sm_'+''.join(map(str,cp.cuda.Device(device).compute_capability))
    mode='strict' if exact else 'fast'
    digest=hashlib.sha256(text.encode()).hexdigest()
    filename=f'{arch}-nz{nz}-{mode}-{digest[:16]}.cubin'
    # A checkout reads woof/data/mynn480/images; an install reads the gpuwm-kernels
    # companion, which carries the set (woof.data_assets.VERIFY_ORACLE_TREES).
    from woof.data_assets import verify_oracle_path
    prebuilt=verify_oracle_path('mynn480/images')/filename
    cache=Path(os.environ.get('XDG_CACHE_HOME',str(Path.home()/'.cache')))/'woof'/'mynn480'
    binary=prebuilt if prebuilt.exists() else cache/filename
    if not binary.exists():
        nvcc=shutil.which('nvcc') or '/usr/local/cuda/bin/nvcc'
        if not Path(nvcc).exists():
            raise RuntimeError(f'MYNN 4.8.0 needs its source-bound {arch} nz={nz} {mode} image; '
                               'run tools/mynn_wrf480_port/prewarm.py with CUDA nvcc installed')
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
    raw=binary.read_bytes()
    receipt=binary.with_suffix('.json')
    if receipt.exists():
        manifest=json.loads(receipt.read_text())
        if manifest['source_sha256']!=digest or manifest['image_sha256']!=hashlib.sha256(raw).hexdigest():
            raise RuntimeError('MYNN 4.8.0 source-bound image receipt does not match '+str(binary))
    elif binary==prebuilt:
        raise RuntimeError('MYNN 4.8.0 prebuilt image is missing its source receipt: '+str(binary))
    module=cp.cuda.function.Module()
    module.load(raw)
    return module,module.get_function('mynn471_step')


#: Fast-route launch order: the phase kernels cut the native driver at its
#: call boundaries (tools/mynn_wrf480_port/phases.py), launched in native
#: order on one stream.
PHASES = (0, 1, 2, 3, 4, 5, 6)


_PLANS = {}


def launch_plan(module, exact):
    """(name, function, grid y) in native order; strict runs the one kernel.

    ``MYNN480_PHASE_PROOF=1`` runs the phase sequence under strict numerics:
    the replay's proof that the cut changes no word. Plans are kept per
    loaded module (no per-step function lookups).
    """
    proof = bool(exact) and os.environ.get('MYNN480_PHASE_PROOF') == '1'
    key = (id(module), bool(exact), proof)
    plan = _PLANS.get(key)
    if plan is not None and plan[0] is module:
        return plan[1]
    try:
        module.get_function('mynn471_phase0')
        phased = not exact or proof
    except Exception:
        phased = False
    plan = ([('mynn471_phase'+str(i), module.get_function('mynn471_phase'+str(i)), 1) for i in PHASES]
            if phased else [('mynn471_step', module.get_function('mynn471_step'), 1)])
    _PLANS[key] = (module, plan)
    return plan


def driver(values,*,initflag,delt,scratch=None,closure=2.6,restart=False,cycling=False,**options):
    import cupy as cp
    n,nz=values['dz'].shape
    meta=schema(nz)
    def buffer(slot,count):
        if scratch is None:return cp.empty(count,dtype=cp.float32)
        return scratch.one(slot,(count,))
    slabs={key:buffer('mynn480_'+key,count*n).reshape(count,n)
           for key,count in meta['slab_fields'].items()}
    slabs['i32']=slabs['i32'].view(cp.int32)
    aliases={'dz1':'dz','u1':'u','v1':'v','w1':'w','th1':'th','pres1':'p',
             'ex1':'exner','rho1':'rho','tk1':'tk','kh1':'exch_h','km1':'exch_m',
             'pattern_spp_pbl1':'rstoch'}
    aliases.update({name+'1':name for name in
                    ('sqv','sqc','sqi','sqs','qnc','qni','qnwfa','qnifa','qnbca',
                     'ozone','qke','qke_adv','el','sh','sm','tsq','qsq','cov',
                     'qc_bl','qi_bl','cldfra_bl')})
    # Old wrapper option names map onto 8f8e8b59's argument names.
    options=dict(options)
    options.pop('icloud_bl',None)
    scalars=dict(i=1,j=1,initflag=initflag,delt=delt,kts=1,kte=nz,nchem=1,
                 ndvel=1,nscalars=1,restart=bool(restart),cycling=bool(cycling),
                 bl_mynn_closure=closure,bl_mynn_cloudpdf=2,
                 bl_mynn_mixlength=1,bl_mynn_edmf=1,bl_mynn_edmf_mom=1,
                 bl_mynn_edmf_tke=0,bl_mynn_output=0,
                 bl_mynn_cloudmix=1,bl_mynn_mixqt=0,tke_budget=0,
                 bl_mynn_tkeadvect=False,spp_pbl=0,mix_chem=False,enh_mix=False,
                 flag_qc=True,flag_qi=True,flag_qs=False,flag_qnc=False,flag_qni=False,
                 flag_qnwfa=False,flag_qnifa=False,flag_qnbca=False,flag_ozone=False,
                 **WRF480_DEFAULTS)
    scalars.update(options)
    arena=buffer('mynn480_work',n*meta['scratch_bytes']//4).view(cp.uint8)
    pack_inputs(meta,slabs,arena,values,aliases,scalars,n)
    exact=os.environ.get('GPUWM_WRF_EXACT')=='1'
    module,_=image(nz,cp.cuda.Device().id,exact)
    threads=32 if exact else 128
    for _,kernel,plumes in launch_plan(module,exact):
        kernel(((n+threads-1)//threads,plumes),(threads,),(np.int32(n),slabs['f32'],slabs['i32'],arena))
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
    out.update({name:get(name) for name in ('pblh','kpbl',*EDMF_2D)})
    # WRF 4.8.0 does not export these two old wrapper diagnostics.
    out['rmol']=values['rmol']
    out['ktop_plume']=cp.zeros(n,dtype=cp.int32)
    if scalars.get('tke_budget',0):
        out.update({name:get(name+'1') for name in BUDGET})
    if scalars.get('bl_mynn_output',0):
        out.update({name:get(name.replace('3d','')+'1') for name in PLUMES})
    return out


# WRF 4.8.0 WRF/module_bl_mynnedmf_driver.F90 mynnedmf_post_run :919-997
# (flag_qc and flag_qi true; WRF passes no RQSBLTEN) and :695-699, where the
# SGS cloud is converted unconditionally (icloud_bl was removed).
POST_SOURCE = r'''
extern "C" __global__ void mynn480_post(int count,int nz,int scol,int slev,
 int ocol,int olev,float dt,
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
 qcbl[o]=qcbl[o]/(1.f-sqv[o]);qibl[o]=qibl[o]/(1.f-sqv[o]);
}
'''


def post_run(stage,out,delt):
    import cupy as cp
    n,nz=stage['qv'].shape
    module,_=image(nz,cp.cuda.Device().id,os.environ.get('GPUWM_WRF_EXACT')=='1')
    kernel=module.get_function('mynn480_post')
    count=n*nz
    kernel(((count+127)//128,),(128,),
           (np.int32(count),np.int32(nz),
            *(np.int32(s//4) for s in stage['qv'].strides),
            *(np.int32(s//4) for s in out['sqv'].strides),
            np.float32(delt),
            stage['qv'],stage['qc'],stage['qi'],out['sqv'],
            out['rqvblten'],out['rqcblten'],out['rqiblten'],out['qc_bl'],out['qi_bl']))
