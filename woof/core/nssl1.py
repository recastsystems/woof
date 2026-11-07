"""Pinned native WRF one-moment NSSL driver (legacy options 19 and 21).

Sedimentation executes in parallel over columns; source processes, complete
native condensation cleanup and RADARDD02 execute in parallel over cells.
No host numerical work, device readback or synchronization occurs in a step.
Native CALCNFROMCUTEN only updates number moments; all its guards are false
for ipconc=0 (module_mp_nssl_2mom.F:5624-5713).
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from pathlib import Path
import numpy as np
from woof.core.device_cache import cuda_cache

STATE_FIELDS = ('theta','qv','qc','qr','qi','qs','qg','qh','qndrop','qnr',
                'qni','qns','qng','qnh','qnn','qvolg','qvolh','qzr','qzg','qzh')
ENV_FIELDS = ('pressure','exner','rho','dz','w')
OUTPUT_FIELDS = (*STATE_FIELDS,'refl_10cm','effc_m','effi_m','effs_m')
SURFACE_FIELDS = ('rainnc','rainncv','snownc','snowncv','graupelnc',
                  'graupelncv','hailnc','hailncv','sr')


@cuda_cache(maxsize=None)
def kernels(nz: int, hail: bool, cccn=0.5e9, shapes=(0.0,1.0)):
    from woof.core.kernels import module_source, _load_module_preserving_subnormals
    from woof.wrf_exact import ENABLED, effective_options
    k = Path(__file__).with_name('kernels')
    # This unit supplies the already pinned WRF math and native lookup tables.
    # One-moment source paths do not consume the two-moment table buffers.
    from woof.core.nssl2_shape import shape_defines
    from woof.core.nssl1_coefficients import specialize
    prefix = ''.join(f'#define {n} {v}ULL\n' for n,v in shape_defines(shapes))
    source = prefix + module_source('nssl2_shape_gs')
    source += ('\n#define GPUWM_WRF_EXACT 1\n' if ENABLED else '\n')
    source += f'\n#define NSSL1_NZ {int(nz)}\n'
    source += (k/'nssl1_support.cuh').read_text()
    body, initialization = specialize((k/f'nssl1_mp{19 if hail else 21}.cuh').read_text(), cccn, shapes, hail)
    source += body + initialization
    source += (k/'nssl1_driver.cuh').read_text()
    options = ('-std=c++17','--ftz=false', '--fmad=false' if ENABLED else '--fmad=true')
    if ENABLED:
        options=effective_options(options)
    module = _load_module_preserving_subnormals(f'nssl1_{int(hail)}_{nz}',source,options)
    if initialization:
        module.get_function('nssl1_init_coefficients')((1,),(64,),())
    return (module.get_function('nssl1_sed'),module.get_function('nssl1_process'),module.get_function('nssl1_radar'),module.get_function('nssl1_velocity'),module.get_function('nssl1_precip'))


def launch(device, environment, diagnostics, surface, cells, dt, *, hail, due=True,
           compiled=None):
    """Shared numerical entry for production and captured native module replay."""
    nz, *horizontal = device['theta'].shape
    ncol = int(np.prod(horizontal))
    sediment, process = (kernels(nz,hail) if compiled is None else compiled)[:2]
    pointers = tuple(device[n] for n in STATE_FIELDS)
    pointers += tuple(environment[n] for n in ENV_FIELDS)
    pointers += tuple(device[n] for n in STATE_FIELDS)
    pointers += tuple(diagnostics[n] for n in OUTPUT_FIELDS[len(STATE_FIELDS):])
    pointers += tuple(surface[n] for n in SURFACE_FIELDS)
    args = (*pointers,cells,np.float32(dt),np.int32(ncol))
    velocity = (kernels(nz,hail) if compiled is None else compiled)[3]
    velocity(((ncol*nz+63)//64,),(64,),args)
    if nz <= 256:
        levels=1<<(nz-1).bit_length()
        sediment(((ncol+3)//4,6 if hail else 5),(4*levels,),args)
    else:
        sediment(((ncol*(6 if hail else 5)+63)//64,),(64,),args)
    precipitation=(kernels(nz,hail) if compiled is None else compiled)[4]
    precipitation(((ncol+127)//128,),(128,),args)
    process(((ncol*nz+63)//64,),(64,),(*args,np.int32(due)))


@dataclass(frozen=True)
class NSSL1ProductionBinding:
    state: object
    dt_s: np.float32
    mode: object
    cells: object
    compiled: tuple
    nssl_cccn: float = 0.5e9
    shapes: tuple = (0.0,1.0)

    def with_step(self,dt):
        from woof.core.nssl2_driver_support import _step32
        step = _step32(dt)
        return self if step == self.dt_s else replace(self,dt_s=step)


def make_binding(state,dt,mode,*,nssl_cccn=0.5e9,shapes=(0.0,1.0)):
    from woof.core.nssl2_driver_support import _step32
    from woof.core.nssl2_shape import resolved_shapes
    shapes=resolved_shapes(*shapes)
    nz,ny,nx=state.p.shape
    return NSSL1ProductionBinding(state,_step32(dt),mode,
        state.scratch((27 if mode.hail else 22,nz,ny,nx),'nssl1_driver_state'),
        kernels(nz,mode.hail,nssl_cccn,shapes),nssl_cccn,shapes)


def apply(state,cfg,dt,*,refl_10cm_due=False):
    import cupy as cp
    from woof.core.nssl2_runtime import (_validate_state_structure,
        _validate_driver_bindings,_prepare_fields,pin_absent_nssl2_fields)
    from woof.core.nssl2_contract import resolve_nssl2_mode_for_config
    from woof.core.microphysics import save_pre_mp_theta,moist_physics_finish,MicrophysicsDiagnostics
    from woof.core.refl import stash_refl_10cm
    binding = state.physics.nssl2_binding
    mode = resolve_nssl2_mode_for_config(cfg)
    if not isinstance(binding,NSSL1ProductionBinding) or binding.state is not state or binding.mode!=mode:
        raise ValueError('one-moment NSSL binding differs from its DomainState/configuration')
    from woof.core.nssl2_shape import resolved_shapes
    shapes=resolved_shapes(cfg.nssl_alphah,cfg.nssl_alphahl,cfg.nssl_icdx,cfg.nssl_icdxhl)
    if binding.nssl_cccn!=cfg.nssl_cccn or binding.shapes!=shapes:
        raise ValueError('one-moment NSSL coefficients differ from the prepared binding')
    binding=binding.with_step(dt)
    state.physics.nssl2_binding=binding
    shape=_validate_state_structure(state)
    pin_absent_nssl2_fields(state,mode)
    _,surface,_=_validate_driver_bindings(state,cfg,shape)
    fields=_prepare_fields(state,shape)
    device={n:fields.theta if n=='theta' else getattr(state,n,np.uint64(0)) for n in STATE_FIELDS}
    environment=dict(pressure=fields.pressure,exner=fields.pii,rho=fields.rho,dz=fields.dz,w=fields.w)
    diagnostics=dict(refl_10cm=state.scratch(shape,'refl_10cm'),
        effc_m=state.effc,effi_m=state.effi,effs_m=state.effs)
    save_pre_mp_theta(state)
    launch(device,environment,diagnostics,surface,binding.cells,dt,hail=mode.hail,
           due=refl_10cm_due,compiled=binding.compiled)
    if refl_10cm_due:
        temperature=state.scratch(shape,'refl_t')
        cp.multiply(fields.theta,fields.pii,out=temperature)
        stash_refl_10cm(state,diagnostics['refl_10cm'])
    moist_physics_finish(state,cfg,fields.theta,dt)
    return MicrophysicsDiagnostics(**surface)


def reflectivity(state,cfg,temperature=None,pressure=None):
    """Native RADARDD02, without advancing the one-moment state."""
    from woof.core.nssl2_runtime import _prepare_fields,_validate_driver_bindings
    from woof.core.nssl2_contract import resolve_nssl2_mode_for_config
    mode=resolve_nssl2_mode_for_config(cfg)
    shape=tuple(state.p.shape);nz,ny,nx=shape;ncol=ny*nx
    f=_prepare_fields(state,shape)
    device={n:f.theta if n=='theta' else getattr(state,n,np.uint64(0)) for n in STATE_FIELDS}
    environment=dict(pressure=f.pressure,exner=f.pii,rho=f.rho,dz=f.dz,w=f.w)
    _,surface,_=_validate_driver_bindings(state,cfg,shape)
    diagnostic=dict(refl_10cm=state.scratch(shape,'refl_10cm'),effc_m=state.effc,effi_m=state.effi,effs_m=state.effs)
    args=tuple(device[n] for n in STATE_FIELDS)+tuple(environment[n] for n in ENV_FIELDS)
    args+=tuple(device[n] for n in STATE_FIELDS)+tuple(diagnostic[n] for n in OUTPUT_FIELDS[20:])
    args+=tuple(surface[n] for n in SURFACE_FIELDS)+(state.physics.nssl2_binding.cells,np.float32(0),np.int32(ncol))
    state.physics.nssl2_binding.compiled[2](((ncol*nz+63)//64,),(64,),args)
    return diagnostic['refl_10cm']
