"""Resident NSSL transaction, fork driver_microphysics.F:548-825 at 1d6e370e.

The numerical module and its initializer live in mp_nssl2m. This adapter owns
precipitation, MPAS carriers, radar diagnostics and fork writeback on the GPU.
"""
from functools import lru_cache
import numpy as np
from . import driver_rrfs as glue
from .mp_nssl2m import Nssl2Moment, PREPARED_FIELDS
from .nssl_math import source as math_source


def initialize(provider):
    if not hasattr(provider, 'nssl_runtime'):
        # A supplied image uses the same source/options/architecture validation
        # as module replay. Without one, the ordinary AOT/warm-cache path applies.
        from ..resources_build import load_packet
        from .mp_nssl2m import load_resources
        packet = getattr(provider, 'resources_packet', None)
        resources = load_packet(packet) if packet is not None else {}
        provider.nssl_runtime = (load_resources(resources['mp_nssl2m'], provider.cp)
            if 'mp_nssl2m' in resources else Nssl2Moment(shared_math_source=math_source()))
        provider.nssl_runtime.moments = provider.session.config.options['config_nssl_moments']
        provider.nssl_runtime.initialize()
    return provider.nssl_runtime


@lru_cache(maxsize=1)
def driver_source():
    from ..kessler_v842 import CUDA_RELHUM_SOURCE
    from ..thompson_v842 import _PRECIP
    # Reuse only the source-identical precipitation routine, not its scheme.
    precip = _PRECIP[:_PRECIP.index('// MPAS mpas_atmphys_driver_microphysics.F:798')]
    for old,new in (('tc_add','A'),('tc_sub','S'),('tc_mul','M'),('tc_div','D')):
        precip = precip.replace(old,new)
    return glue.microphysics_handoff_source()+CUDA_RELHUM_SOURCE+precip+RADAR


RADAR = r'''
// RRFS 44c0849 mpas_atmphys_driver_microphysics.F:1038,1049-1061 (calc_prate_max): the same as tempo_v303.tempo_prate.
extern "C" __global__ void nssl_prate(int nc,int longest,float dt,
 const float* step,const float* total,float* peak,float* rolling,
 float* ave1,float* ave5,float* ave10,int n1,int n5,int n10,
 float time1,float time5,float time10) {
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
 float v=D(step[c],dt);peak[c]=v>peak[c]?v:peak[c];
 if(!rolling)return;
 for(int k=longest-1;k>0;--k)rolling[k*nc+c]=rolling[(k-1)*nc+c];
 rolling[c]=total[c];
 v=D(S(rolling[c],rolling[(n1-1)*nc+c]),time1);ave1[c]=v>ave1[c]?v:ave1[c];
 v=D(S(rolling[c],rolling[(n5-1)*nc+c]),time5);ave5[c]=v>ave5[c]?v:ave5[c];
 v=D(S(rolling[c],rolling[(n10-1)*nc+c]),time10);ave10[c]=v>ave10[c]?v:ave10[c];
}
// RRFS mpas_atmphys_driver_microphysics.F:1098-1131,1300-1345.
extern "C" __global__ void nssl_radar(int nc,int nz,int diagnose,int hourly,
 const float* z,const float* dz,const float* refl,const float* module1km,
 float* maxrefl,float* refl1km,float* hourly1km) {
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
 if(diagnose) {
  int kp=0;float mx=-10.f;
  for(int k=0;k<nz;++k){int p=k*nc+c;
   mx=MAX(mx,MAX(-10.f,refl[p]));
   float h=A(S(z[p],z[c]),M(0.5f,dz[p]));if(h<1000.f)kp=k;
  }
  float h1=A(S(z[kp*nc+c],z[c]),M(0.5f,dz[kp*nc+c]));
  float h2=A(S(z[(kp+1)*nc+c],z[c]),M(0.5f,dz[(kp+1)*nc+c]));
  float w1=D(S(h2,1000.f),S(h2,h1)),w2=S(1.f,w1);
  maxrefl[c]=mx;refl1km[c]=A(M(w1,MAX(-10.f,refl[kp*nc+c])),M(w2,MAX(-10.f,refl[(kp+1)*nc+c])));
 }
 if(hourly)hourly1km[c]=MAX(hourly1km[c],module1km[c]);
}
'''


def _rolling_prate(runtime, cp, dt_microp, nc):
    """Native SAVE state of the 1/5/10-minute rolling precipitation (44c0849 driver_microphysics.F:994-1020)."""
    rate=getattr(runtime,'rolling_prate',None)
    if rate is None:
        dt=np.float32(dt_microp)
        counts=np.asarray((60,300,600),dtype=np.float32)/dt
        # Native FRACTION is the binary significand, not the fractional part.
        fractional=np.abs(np.frexp(counts)[0])>np.float32(1.e-6)
        limits=np.where(fractional,np.floor(counts+np.float32(.5)),np.trunc(counts)).astype(np.int32)+1
        times=np.where(fractional,dt*(limits-1).astype(np.float32),np.asarray((60,300,600),dtype=np.float32))
        longest=int(limits.max())
        rolling=None if np.any(limits==1) else cp.zeros((longest,nc),dtype=cp.float32)
        rate=runtime.rolling_prate=dict(rolling=rolling,longest=longest,limits=limits,times=times)
    return rate


def bind_native_pools(provider, call):
    if provider.level != 2:
        raise ValueError('NSSL microphysics requires MPAS state time level 2')
    return {'provider':provider, 'runtime':initialize(provider)}


def run_slot(call):
    from ..thompson_v842 import _pointer_table, PRECIP_ARGUMENTS
    p=call.pools['provider'];cp=p.cp;runtime=call.pools['runtime']
    if not hasattr(runtime,'driver_cache'):runtime.driver_cache=glue.StrictPhysicsKernelCache()
    cache=runtime.driver_cache
    scalars=p.field('state','scalars');nz,nc=scalars.shape[1:]
    names=glue._microphysics_carriers(call.options)
    species={n:cp.ascontiguousarray(scalars[p.session.layout.indices['state','index_'+n]-1]) for n in names}
    fields={n:p.field('state',n) for n in ('theta_m','rho_zz','w')}
    fields.update({n:p.field('mesh',n) for n in ('zz','zgrid')})
    fields.update({n:p.field('diag',n) for n in ('exner','exner_base','rtheta_base','rtheta_p','pressure_base','pressure_p','surface_pressure')})
    diagnostics={n:p.field('diag_physics',n) for n in glue._microphysics_diagnostics(call.options,entering=True)}
    prepared,previous=glue.microphysics_from_mpas(fields=fields,species=species,diagnostics=diagnostics,
                                                options=call.options,kernel_cache=cache)
    surface=cp.zeros((7,nc),dtype=cp.float32)
    for i,name in enumerate(PREPARED_FIELDS['surface']):prepared[name]=surface[i]
    prepared['refl10cm_1km_p']=p.prepared('refl10cm_1km_p')
    groups=dict(PREPARED_FIELDS)
    if runtime.moments=='nssl3m':groups['state']+=('zrw_p','zgw_p','zhw_p')
    arrays={n:cp.stack([prepared[k] for k in members]) if len(members)>1 else prepared[members[0]]
            for n,members in groups.items()}
    precip={n:p.field('diag_physics',n) for n in (*PRECIP_ARGUMENTS,'i_rainnc')}
    diag_ptrs=_pointer_table(tuple(precip[n] for n in PRECIP_ARGUMENTS))
    surface_ptrs=_pointer_table(tuple(arrays['surface'][i] for i in range(7)))
    kernel=cache.raw_kernel('mpas_thompson_precip',driver_source(),module_key=__name__)
    def precipitation(phase):
        kernel(((nc+127)//128,),(128,), (arrays['state'][1],prepared['rho_p'],prepared['dz_p'],
            surface_ptrs,diag_ptrs,precip['i_rainnc'],np.int32(nz),np.int32(nc),np.int32(phase),
            np.int32(p['scalar_state']['l_acrain']),np.float32(call.options['config_bucket_rainnc'])))
    precipitation(0)
    # do_diag_dbz is a host clock/stream flag, not a device reduction.
    hourly=int(p.native._host.get_field('diag_physics','do_diag_dbz').item()==1) if getattr(p,'resident',False) else int(p.native.get_field('diag_physics','do_diag_dbz').item()==1)
    s=p['scalar_state']
    runtime.last_errors=runtime.run(arrays,nt=call.itimestep,nmicro=int(s['n_microp']),
        diagnose=int(s['l_diags']),dbz_due=hourly,reqc=int(s['has_reqc']),reqi=int(s['has_reqi']),
        reqs=int(s['has_reqs']),dt=float(s['dt_microp']))
    from ..deferred_checks_v842 import defer
    defer(runtime.last_errors, 'NSSL CUDA reached a native STOP or fatal error')
    for group,members in groups.items():
        for i,name in enumerate(members):prepared[name]=arrays[group][i] if len(members)>1 else arrays[group]
    kernel=cache.raw_kernel('nssl_radar',driver_source(),module_key=__name__)
    kernel(((nc+127)//128,),(128,), (np.int32(nc),np.int32(nz),np.int32(s['l_diags']),np.int32(hourly),
        prepared['z_p'],prepared['dz_p'],prepared['refl10cm_p'],prepared['refl10cm_1km_p'],
        p.field('diag_physics','refl10cm_max'),p.field('diag_physics','refl10cm_1km'),p.field('diag_physics','refl10cm_1km_max')))
    if s['l_diags']:
        kernel=cache.raw_kernel('kessler_relhum',driver_source(),module_key=__name__)
        kernel(((nz*nc+127)//128,),(128,), (prepared['th_p'],prepared['pi_p'],prepared['qv_p'],
            fields['pressure_base'],fields['pressure_p'],p.field('diag','relhum'),np.int32(nz*nc)))
    kernel=cache.raw_kernel('mpas_thompson_precip',driver_source(),module_key=__name__)
    precipitation(1)
    if call.options.get('calc_prate_max',False):
        # RRFS 44c0849 driver_microphysics.F:994-1020,1038,1049-1061 (calc_prate_max), as the TEMPO route
        # (tempo_v303 rolling_prate/tempo_prate). THE BREAKAGE THIS ENDS: with calc_prate_max on, the NSSL route left
        # prate_max and ave_prate_1/5/10min untouched while native fills them for every microphysics scheme.
        rate=_rolling_prate(runtime,cp,float(s['dt_microp']),nc)
        kernel=cache.raw_kernel('nssl_prate',driver_source(),module_key=__name__)
        kernel(((nc+127)//128,),(128,), (np.int32(nc),np.int32(rate['longest']),np.float32(s['dt_microp']),
            prepared['rainnc_p'],precip['rainnc'],p.field('diag_physics','prate_max'),
            rate['rolling'] if rate['rolling'] is not None else np.uint64(0),
            *(p.field('diag_physics',n) for n in ('ave_prate_1min','ave_prate_5min','ave_prate_10min')),
            *(np.int32(v) for v in rate['limits']),*(np.float32(v) for v in rate['times'])))
    updated,new_species,diags,tend=glue.microphysics_to_mpas(fields=fields,species=species,prepared=prepared,
        previous_tendencies=previous,options=call.options,dt_dyn=call.dt,kernel_cache=cache)
    for n,v in updated.items():
        if n in ('rho_zz','zz','exner_base','rtheta_base','pressure_base'):continue
        pool='state' if n=='theta_m' else 'tend' if n in ('rt_diabatic_tend','tend_sfc_pressure') else 'diag'
        p.field(pool,n)[...]=v
    for n,v in new_species.items():scalars[p.session.layout.indices['state','index_'+n]-1]=v
    for n,v in diags.items():p.field('diag_physics',n)[...]=v
    for n,v in {**previous,**tend}.items():p.field('tend_physics',n)[...]=v
    return runtime.last_errors
