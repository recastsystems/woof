"""RRFS sf_mynn sibling: identical MMM numerics, fork driver semantics.

RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/module_sf_mynn.F:104-171 @1d6e370e adds comments only.
MMM sf_mynn.F90 and sf_mynn_pre.F are byte-identical to MPAS-A v8.4.2.
Source identity is pinned by tests/mpas_rrfs_mynnsfc/audit.py. The shared
v8.4.2 module stays independently selectable. RRFS table registration selects
this sibling; there is no suite-name or fork-pin branch in the launcher.
"""

from woof.backend import compilation_target as _compilation_target
from woof.hex import numerics as _hex_numerics
from importlib.resources import files
import numpy as np
from . import sf_mynnsfc_v600 as sfc

SCHEME='sf_mynn'
BACKEND='hex-cuda'
_MODULES={}


def shared_scheme():
    """RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_mmm/sf_mynn.F90:1-2237 @1d6e370e: byte-identical reuse."""
    from woof.hex.mpasphys import sf_mynn_v842
    return sf_mynn_v842


def cuda_source():
    """RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_sfclayer.F:434-677,772-972 @1d6e370e."""
    base=shared_scheme()
    macros='\n'.join('#define IDX_'+name+' '+str(i) for i,name in enumerate(base.FIELDS))
    macros+='\n#define FIELD_COUNT '+str(len(base.FIELDS))
    macros+='\n#define F(name) f[IDX_##name*n+i]\n'
    macros+='#define BLEND_FIELD(j) ('+' || '.join('(j)==IDX_'+k for k in sfc._BLEND)+')\n'
    source=sfc.DRIVER_SOURCE.replace('mynnsfc_from_mpas','mynn_rrfs_from_mpas').replace('mynnsfc_to_mpas','mynn_rrfs_to_mpas')
    source+='''
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_sfclayer.F:917 @1d6e370e
extern "C" __global__ void mynn_rrfs_regime(float *f,const float *ice,
    const float *hold,float threshold,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n && ice[i]>=threshold && ice[i]<1.0f && ice[i]>=0.5f)
        F(regime)=hold[i];
}
'''
    kernels=files('woof').joinpath('core/kernels')
    return ('typedef float real;\n#define SL_SQRT __fsqrt_rn\n#define SL_POW mynn_glibc_powf\n'
            +kernels.joinpath('mynn_libm.cuh').read_text()+macros+source)


def module():
    """RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_sfclayer.F:434-972 @1d6e370e."""
    import cupy as cp
    from cupy.cuda import compiler,function
    dev=cp.cuda.runtime.getDevice()
    if dev not in _MODULES:
        binary,_=compiler.compile_using_nvrtc(cuda_source(),options=_hex_numerics.options(('--std=c++17','--fmad=false',
            '--ftz=false','--prec-div=true','--prec-sqrt=true')),
            arch=_compilation_target(dev, xp=cp),cache_in_memory=True)
        obj=function.Module();obj.load(binary);_MODULES[dev]=obj
    return _MODULES[dev]


def driver(mesh,diagnostics,surface,profiles,constants,*,itimestep,len_disp,xice_threshold,
           config_frac_seaice=False,config_do_restart=False,config_do_DAcycling=False,
           sea_state=None):
    """RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_sfclayer.F:434-677,772-972,1325-1388 @1d6e370e."""
    import cupy as cp
    base=shared_scheme();n=surface['xland'].size
    packed=cp.zeros((len(base.FIELDS),n),cp.float32)
    for name in base._DIAG_INPUT:packed[base.FIELDS.index(name)]=diagnostics[name]
    packed[base.FIELDS.index('pblh')]=diagnostics['hpbl']
    for name,key in sfc._SURFACE.items():
        if name in base.FIELDS: packed[base.FIELDS.index(name)]=surface[key]
    packed[base.FIELDS.index('psfcpa')]=profiles['psfcpa']
    sea=cp.empty_like(packed) if config_frac_seaice else packed
    hold=diagnostics['regime'].copy() if config_frac_seaice else packed[0]
    sst=surface['sst'] if config_frac_seaice else packed[0]
    if n:
        module().get_function('mynn_rrfs_from_mpas')(((n+127)//128,),(128,),
            (packed,sea,mesh['meshDensity'],sst,surface['xice'],np.float32(len_disp),
             np.float32(xice_threshold),np.int32(config_frac_seaice),np.int32(n)))
    initflag=int(not(config_do_restart or config_do_DAcycling or itimestep>1))
    scalars=cp.asarray([constants[k] for k in base.CONSTANTS],cp.float32)
    base.preprocess(packed,profiles,itimestep=initflag)
    base.run_packed(packed,scalars,itimestep=initflag,isfflx=1,isftcflx=0,iz0tlnd=0,spp_pbl=False)
    if config_frac_seaice:
        base.preprocess(sea,profiles,itimestep=initflag)
        base.run_packed(sea,scalars,itimestep=initflag,isfflx=1,isftcflx=0,iz0tlnd=0,spp_pbl=False)
        # RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_sfclayer.F:335-361,1364-1374 @1d6e370e
        # RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_seaice.F:574-597 @1d6e370e
        if sea_state is not None:
            for name in sfc._SEA_HANDOFF:
                if name in base.FIELDS:
                    sea_state[name+'_sea'][...]=sea[base.FIELDS.index(name)]
        if n:
            module().get_function('mynn_rrfs_to_mpas')(((n+127)//128,),(128,),
                (packed,sea,surface['xice'],np.float32(xice_threshold),np.int32(n)))
            module().get_function('mynn_rrfs_regime')(((n+127)//128,),(128,),
                (packed,surface['xice'],hold,np.float32(xice_threshold),np.int32(n)))
    for name in base.OUTPUTS:
        if name not in ('wstar','qstar'):
            diagnostics[base._DIAG_NAMES.get(name,name)][...]=packed[base.FIELDS.index(name)]
    return diagnostics


def launch_slot(call):
    """RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_sfclayer.F:1325-1388 @1d6e370e."""
    if (call.slot,call.scheme,str(call.backend))!=('surface_layer',SCHEME,BACKEND):
        raise ValueError('RRFS MYNN launcher requires surface_layer/sf_mynn/hex-cuda')
    p=call.pools;configs=p['configs']
    return driver(p['mesh'],p['diag_physics'],p['sfc_input'],p['atm_input'],p['constants'],
        itimestep=call.itimestep,len_disp=configs['config_len_disp'],
        xice_threshold=p['scalar_state']['xice_threshold'],
        config_do_restart=configs['config_do_restart'],config_do_DAcycling=configs['config_do_DAcycling'],
        config_frac_seaice=call.options['config_frac_seaice'],sea_state=p['prepared'])


def register_backend(registry):
    """RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_sfclayer.F:1325-1388 @1d6e370e."""
    registry.register('surface_layer',SCHEME,BACKEND,launch_slot)
