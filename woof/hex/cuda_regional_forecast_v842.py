"""Complete native regional forecast binding.

MPAS-A v8.4.2 mpas_atm_core.F:539-596,735-781,1036-1040.
The native physics driver stays unchanged; this module owns regional
geometry, allocation extents and the resident boundary clock.
"""
from __future__ import annotations
from dataclasses import dataclass, fields
from datetime import datetime
import hashlib
import inspect
import json
from pathlib import Path
from .mpasphys.native_backend import NativePhysics
from .mpasphys.pool_v842 import NativePoolHydration
from .mpasphys.session_v842 import NativePhysicsSession, native_working_directory
from .mpasphys.config_v842 import NativeMpasDycoreConfigV842
from .mpasphys.runtime_v842 import (_configure_streams, initialize_native_output,
                                   write_native_output)

@dataclass(frozen=True, slots=True)
class RegionalNativeDycoreConfigV842(NativeMpasDycoreConfigV842):
    """Regional native option; MPAS-A v8.4.2 Registry.xml:284."""
    config_relax_zone_divdamp_coef: float = 6.0

class RegionalNativePhysicsSession(NativePhysicsSession):
    """Preserve allocation columns; MPAS-A v8.4.2 mpas_block_creator.F:1490-1496."""
    def device_field(self, pool, name, *, level=1):
        """Return the full pool; MPAS-A v8.4.2 mpas_pool_routines.F:383-395."""
        return self.native.get_field(pool,name,level=level)

    def after_call(self, entry, call):
        """Hold native sources; MPAS-A v8.4.2 mpas_atmphys_todynamics.F:98-498."""
        from .cuda_physics_v841 import CudaMpasStageSourcesV842
        if entry == 'physics_get_tend':
            return CudaMpasStageSourcesV842(
                self.sources['tend_rho_physics'].copy(),
                self.sources['tend_ru_physics'].copy(),
                self.sources['tend_rtheta_physics'].copy(),
                self.device_field('tend','scalars_tend').copy())
        return None

class RegionalResidentNativeCoupling:
    """Hold outer sources and hand transport to native; mpas_atm_time_integration.F:2140-2816."""

    # MPAS outer-step coupling ownership: mpas_atm_time_integration.F:2140-2816.
    def __init__(self, session, dycore, geometry, names):
        self.session, self.dycore = session, dycore
        self.geometry, self.names = geometry, tuple(names)
        self.cp = dycore.cp
        from .mpasphys.compile_v842 import MpasPhysicsKernelCache
        self.physics_cache = MpasPhysicsKernelCache(
            capability=dycore.cache.capability,
            cache_dir=Path(dycore.cache.cache_directory)/'mpasphys')
        self.sources = None
        self.transport = None

    def synchronize(self, atmosphere, *, level=1):
        """Copy native carriers/diagnostics; mpas_atmphys_interface.F:284-305."""
        from .cuda_physics_prep_v841 import prepare_mpas_to_phys_cuda_v841
        s, d, native = atmosphere.state, atmosphere.saved, self.session
        for name, value in (('rho_zz',s.rho),('theta_m',d.theta_m),
                            ('u',d.normal_velocity),('w',d.vertical_velocity),
                            ('scalars',s.scalars)):
            native.put('state',name,value,level=level)
        for name, value in (('rho_p',d.density_perturbation),
                            ('rtheta_p',d.rho_theta_perturbation),
                            ('pressure_p',d.pressure_perturbation),('exner',d.exner)):
            native.put('diag',name,value)
        columns=prepare_mpas_to_phys_cuda_v841(atmosphere,scalar_names=self.names,
                                              geometry=self.geometry,kernel_cache=self.physics_cache,
                                              registry_scalar_names=next(row['components'] for row in native.layout.fields
                                                if row['pool']=='state' and row['name']=='scalars'))
        from .mpasphys.diagnostics_v842 import output_diagnostics
        theta,rho,pressure=output_diagnostics(atmosphere,self.names,self.physics_cache,self.cp)
        for name,value in (('uReconstructZonal',columns.u_p),('uReconstructMeridional',columns.v_p),
                           ('rho',rho),('theta',theta),('pressure',pressure)):
            # Native does not enter the garbage diagnostic column.
            native.put('diag',name,value[...,:-1])
        return columns

    def stage(self, *, rk_step, dynamics_substep, state, saved, diagnostics):
        """Assemble outer sources; mpas_atm_time_integration.F:2140-2176; mpas_atm_time_integration.F:3088-3185."""
        if self.sources is None:
            from .cuda_physics_v841 import CudaMpasStageSourcesV842
            # Native RK setup copies level1 mass to level2 before physics_get_tend
            # reads it (mpas_atmphys_todynamics.F:148-153). Initial Registry
            # hydration has not derived rho_zz; reusing it would zero first-step
            # physics sources and then retain stale mass on subsequent steps.
            self.session.put('state','rho_zz',state.rho,level=2)
            # Native's outer rho_edge is diagnostic h_edge (= edge density).
            self.session.put('diag','rho_edge',diagnostics.h_edge)
            sources=self.session.driver.physics_get_tend(rk_step=rk_step,
                              dynamics_substep=dynamics_substep,pools=self.session)
            # microphysics_to_MPAS leaves this lagged heating in the native
            # tendency pool for the next RK step (interface.F:912-926). Native
            # adds it after rthdynten, then removes it at the final recovery
            # (time_integration.F:6931-6948,4436-4437). Keep it separate from
            # physics_get_tend so density weighting/order remain unchanged.
            diabatic=self.cp.ascontiguousarray(self.cp.asarray(
                self.session.device_field('tend','rt_diabatic_tend')))
            self.sources=CudaMpasStageSourcesV842(*(self.cp.ascontiguousarray(self.cp.asarray(value))
                              for value in (sources.rho,sources.rho_u,sources.rho_theta,sources.scalars)),
                              diabatic_theta=diabatic)
        self.dycore.rt_diabatic_tend=self.sources.diabatic_theta
        return self.sources

    def post_transport(self, *, state, saved, outer_state, modified_theta_dynamics_rate):
        """Enter before any scalar clamp; mpas_atm_time_integration.F:2770-2816."""
        self.transport=(state,saved,outer_state,modified_theta_dynamics_rate)
        # This diagnostic is formed for every DO_PHYSICS build, including
        # suite none (mpas_atm_time_integration.F:6931-6941; Registry.xml:3535).
        if modified_theta_dynamics_rate is not None:
            self.session.put('tend_physics','rthdynten',modified_theta_dynamics_rate)
        self.session.put('tend','rt_diabatic_tend',self.dycore.rt_diabatic_tend)
        self.session.driver.finish_step(pools=self.session)
        native=self.session.native
        if not self.session.config.active('microphysics'):
            # MPAS v8.4.2 mpas_atm_time_integration.F:2798-2800 applies
            # this WHERE assignment in every DO_PHYSICS build, including
            # a dry suite. Boundary nudging can make upper-level qv negative.
            self.dycore.regional_v841.clamp_negative_scalars(state.scalars)
            self.transport=None
            return
        # MPAS microphysics_to_MPAS supplies its own Exner and perturbation
        # words. Do not recompute those from the product on the host.
        for target,pool,name in ((state.scalars,'state','scalars'),
                                 (saved.theta_m,'state','theta_m'),
                                 (saved.exner,'diag','exner'),
                                 (saved.rho_theta_perturbation,'diag','rtheta_p'),
                                 (saved.pressure_perturbation,'diag','pressure_p')):
            target[...] = self.cp.asarray(self.session.device_field(pool,name,level=2 if pool=='state' else 1))
        # Native keeps rtheta_p after the rounded product/subtraction, then
        # recovers full rtheta by adding the base (mpas_atmphys_interface.F:923-926).
        state.rho_theta[...] = saved.rho_theta_perturbation
        self.dycore._add(state.rho_theta,self.dycore.atmosphere.reference.rho_theta_base)
        self.transport=None

    def prepare_transport(self, call):
        """Prepare GF/NTiedtke heating before microphysics; mpas_atm_time_integration.F:2770-2802."""
        from .cuda_backend import DeviceAtmosphere
        from .cuda_physics_v841 import (prepare_mpas_post_transport_v842,
                                        prepare_mpas_transport_source_baseline_v842)
        if self.transport is None:
            raise RuntimeError('native post-transport callback was not bound to the current CUDA step')
        state,saved,outer,rate=self.transport
        rth=self.cp.zeros_like(state.rho) if rate is None else rate
        rqv=self.cp.empty_like(state.rho)
        convection=self.session.config.selectors['convection']
        if convection in ('cu_grell_freitas','cu_ntiedtke') and rate is None:
            raise RuntimeError('native convection needs the final RK modified-theta rate')
        before=outer.scalars
        if (self.dycore.config.config_scalar_advection
                and (self.dycore.config.config_monotonic
                     or self.dycore.config.config_positive_definite)):
            # Native F:5208 modifies scalars_1 before F:2783 computes rqvdynten.
            # Subtracting the untouched outer state folds physical vapor sources
            # into the resolved-advection rate and then convection sees them twice.
            before=prepare_mpas_transport_source_baseline_v842(before=before,
                source=self.sources.scalars,rho_old=outer.rho,
                dt=self.dycore.config.config_dt,n_cells_solve=self.session.layout.dimensions['nCellsSolve'],
                kernel_cache=self.physics_cache)
            self.session.put('state','scalars',before,level=1)
        prepare_mpas_post_transport_v842(before=before,after=state.scalars,
            theta_m=saved.theta_m,scalar_names=self.names,rqvdynten=rqv,rthdynten=rth,
            convection_scheme=convection,monotonic=self.dycore.config.config_monotonic,
            dt=self.dycore.config.config_dt,n_cells_solve=self.session.layout.dimensions['nCellsSolve'],
            kernel_cache=self.physics_cache)
        atmosphere=self.dycore.atmosphere
        candidate=DeviceAtmosphere(atmosphere.mesh,state,atmosphere.vertical,
                    atmosphere.reference,saved,atmosphere.terrain,atmosphere.h2d)
        self.synchronize(candidate,level=2)
        # Registry.xml:3631-3633 allocates this field only for GF/Tiedtke.
        if ('tend_physics','rqvdynten') in self.session.native._fields:
            self.session.put('tend_physics','rqvdynten',rqv)
        self.session.put('tend_physics','rthdynten',rth)

def load_native_regional_inputs(arguments, layout, payload):
    """Complete input ownership before CUDA; mpas_atm_core.F:116-200,589-596."""
    import numpy as np
    from .mesh import Mesh
    from .vector import initialize_reconstruction_coefficients
    from .driver import load_mpas_vertical_grid,load_mpas_initial_state
    from .dynamics_v841 import load_v841_reference_wind_profiles
    init=Path(arguments.init)
    hydration=NativePoolHydration(None,layout,payload)
    # The native input stream overlays the grid with initialized mesh fields,
    # including fEdge/fVertex; Registry.xml:498-545; mpas_atm_core.F:116-151.
    mesh=Mesh.from_netcdf(getattr(arguments,'grid',None) or init,init)
    mesh.arrays['nominalMinDc']=hydration.horizontal_length()
    # MPAS initializes these vectors before RBF coefficients (core.F:539-544;
    # mpas_rbf_interpolation.F:137). Stored init vectors are not runtime vectors.
    from .rust_geometry import _run
    values=_run(mesh,mode=0,dtype=np.dtype(np.float32),include_halos=True)
    for name,value in zip(('edgeNormalVectors','localVerticalUnitVectors',
                           'cellTangentPlane','coeffs_reconstruct'),values):
        mesh.arrays[name]=value
    grid=load_mpas_vertical_grid(init,mesh,config_coef_3rd_order=layout.configs['config_coef_3rd_order'],allow_regional_sentinels=True)
    profiles=load_v841_reference_wind_profiles(init,n_vert_levels=grid.vertical_grid.n_vert_levels)
    names=next(r['components'] for r in layout.fields if r['pool']=='state' and r['name']=='scalars')
    state,reference,saved=load_mpas_initial_state(init,mesh,grid.vertical_grid,
            scalar_names=(),reference_path=init,terrain_metrics=grid.terrain_metrics,
            return_saved_diagnostics=True,cold_start=True,allow_regional_sentinels=True)
    # Native allocates every selected carrier, including absent input species;
    # Registry.xml:1701-1754; mpas_atmphys_init_microphysics.F:98-226. Rust already
    # applied their Registry defaults while decoding the complete scalar buffer.
    state.scalars=np.ascontiguousarray(hydration.field_buffer('state','scalars')[...,:-1])
    for name in ('defc_a','defc_b'):
        mesh.arrays[name]=np.zeros((mesh.n_cells,mesh.arrays['edgesOnCell'].shape[1]),np.float32)
    return mesh,grid,profiles,state,reference,saved,names


def run_native_regional_dynamics(arguments, config, layout, payload, streams, library,
                        start, stop, dt, steps, out):
    """Real-door native physics plus resident CUDA dynamics; mpas_atm_core.F:589-701,1036-1040."""
    from .cuda_driver import CudaDryDycoreDriver
    from .cuda_backend import require_cuda
    from .mpasphys.compile_v842 import MpasPhysicsKernelCache
    from .cuda_physics_prep_v841 import CudaMpasToPhysGeometryV841

    signature=inspect.signature(CudaDryDycoreDriver._step_device_v841).parameters
    missing={'physics_post_transport_callback','physics_mass_loading_count'}-set(signature)
    if config.moist_physics and missing:
        raise RuntimeError('native physics dycore reconciliation is absent: '+', '.join(sorted(missing)))
    for method in ('initialize_state_time','update_state_time'):
        if not callable(getattr(NativePhysics,method,None)):
            raise RuntimeError(f'native output time API {method} is absent: stale xtime/CF Time would invalidate every output')
    mesh,grid,profiles,state,reference,saved,names=load_native_regional_inputs(arguments,layout,payload)
    options={f.name:layout.configs[f.name] for f in fields(RegionalNativeDycoreConfigV842)
             if f.init and f.name in layout.configs}
    options.update(config_dt=dt,config_moist_physics=config.moist_physics)
    from .cuda_regional_forecast_v841 import open_regional_forecast_v841
    from .lbc_stream import initial_paths
    from .mpasphys.runtime_v842 import native_lbc_scalar_order
    dycore=open_regional_forecast_v841(mesh,state,grid.vertical_grid,reference,
        saved,grid.terrain_metrics,RegionalNativeDycoreConfigV842(**options),
        reference_wind_profiles=profiles,lbc_paths=initial_paths(arguments.lbc_dir),
        start_time=datetime.strptime(start,'%Y-%m-%d_%H:%M:%S'),
        scalar_names=native_lbc_scalar_order(layout,config,names)[0],
        kernel_cache=MpasPhysicsKernelCache(capability=require_cuda(min_compute=(12,0)),
                                cache_dir=str(payload.parent/'cuda-cache')))
    geometry=CudaMpasToPhysGeometryV841.from_host(dycore.regional_v841.padded)
    receipt=dict(schema='mpas-native-forecast-v1',identity_checked=False,completed=False,
                 physics=config.receipt(),steps_requested=steps,steps_completed=0,
                 native_library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),streams=[])
    table_roots=[layout.source/'src/core_atmosphere/physics/physics_wrf/files',
                 *(getattr(arguments,'native_physics_tables',None) or [])]
    receipt['table_sha256']={}
    for root in table_roots:
        for table in sorted(root.iterdir()):
            if table.is_file() and (table.suffix in ('.TBL','.DBL') or table.name.startswith(('RRTMG_','CCN_'))):
                target=out/table.name
                if target.exists() and target.resolve()!=table.resolve():
                    raise ValueError(f'native table destination already differs: {target}')
                if not target.exists():target.symlink_to(table.resolve())
                receipt['table_sha256'][table.name]=hashlib.sha256(table.read_bytes()).hexdigest()
    with NativePhysics(library,test_only=True) as native:
        coupling=None
        session=RegionalNativePhysicsSession(native,layout,payload,config,dt=dt,start=start,stop=stop,
            stream_intervals={s.name:(s.input_interval if s.name=='surface' else s.output_interval) for s in streams},
            post_transport=lambda call:coupling.prepare_transport(call),cuda_device=dycore.cp,
            cuda_resources=getattr(arguments,'physics_resources',None))
        receipt['missing_fields']=_configure_streams(native,streams,out,layout)
        coupling=RegionalResidentNativeCoupling(session,dycore,geometry,names)
        if not layout.configs['config_do_restart']:
            from .mpasphys.diagnostics_v842 import initialize_thermodynamics
            initialize_thermodynamics(dycore.atmosphere,coupling.physics_cache)
        resident_reference=dycore.atmosphere.reference
        for name,value in (('rho_base',resident_reference.rho_base),('rtheta_base',resident_reference.rho_theta_base),
                           ('pressure_base',resident_reference.pressure_base),('exner_base',resident_reference.exner_base)):
            session.put('diag',name,value)
        session.put('mesh','zz',dycore.atmosphere.vertical.zz)
        session.put('mesh','dss',dycore.atmosphere.vertical.dss)
        import numpy as np
        for name,axes in (('edgeNormalVectors',(1,0)),
                          ('localVerticalUnitVectors',(1,0)),
                          ('cellTangentPlane',(2,1,0)),
                          ('coeffs_reconstruct',(2,1,0))):
            session.put('mesh',name,np.ascontiguousarray(mesh.arrays[name].transpose(axes)))
        coupling.synchronize(dycore.atmosphere)
        with native_working_directory(out):
            session.initialize()
            initialize_native_output(native,streams,restarting=layout.configs['config_do_restart'])
            dycore.atmosphere.state.scalars[...] = dycore.cp.asarray(session.device_field('state','scalars'))
            for step in range(steps+1):
                if step:
                    coupling.sources=None
                    coupling.synchronize(dycore.atmosphere)
                    session.begin_step(step,(step-1)*dt)
                    # Registry's qv group exists in the dry suite too. Native
                    # always computes the moist coefficients from that active
                    # group (mpas_atm_time_integration.F:2125-2152).
                    result=dycore._step_device_v841(physics_stage_callback=coupling.stage,
                        physics_post_transport_callback=coupling.post_transport,
                        physics_mass_loading_count=layout.indices['state','moist_end'])
                    dycore.atmosphere=result.atmosphere
                    # Resident state is published at level1; compute the scalar
                    # clock word using the same native RKIND operation on that level.
                    native.update_state_time(dt,source_level=1,target_level=1)
                    native.advance_clock()
                    receipt['steps_completed']=step
                    with (out/'physics-calls.jsonl').open('a') as log:
                        log.write(json.dumps(dict(step=step,calls=session.driver.calls))+'\n')
                coupling.synchronize(dycore.atmosphere)
                write_native_output(native,streams,step=step,receipt=receipt)
                (out/'native-physics-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
            session.driver.finalize(session)
    receipt['completed']=True
    receipt['physics_compile_manifest']=coupling.physics_cache.compile_manifest()
    receipt['dynamics_compile_manifest']=dycore.cache.compile_manifest()
    from .device_memory_release import process_release
    receipt['device_memory_release']=process_release().receipt()
    (out/'native-physics-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    from .correct_mode import terminal_record
    print(json.dumps(terminal_record(receipt)))
    return 0
