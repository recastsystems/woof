"""Whole split-explicit RRFS step driver, beside the frozen default driver.

Source: RRFSx/MPAS-Model MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:1928-7759
at 1d6e370e. Container preparation, recovery and shared stencils are inherited;
the complete altered subcycle and outer split step are composed here.
"""
from __future__ import annotations
from dataclasses import replace
from typing import Any
import numpy as np
from .cuda_driver import CudaDryDycoreDriver
from .config_rrfs import RRFSDryDycoreConfig
from .cuda_horizontal_rrfs import CudaHorizontalRRFS
from .cuda_acoustic_rrfs import compute_vertical_implicit_coefficients_cuda_rrfs, advance_acoustic_step_cuda_rrfs
from .cuda_dynamics_rrfs import vector_momentum_tendency_cuda_rrfs
from .cuda_regional_rrfs import apply_w_boundary_cuda_rrfs
from .cuda_transport_rrfs import advance_scalar_transport_cuda_rrfs
from .damping_rrfs import build_rrfs_vertical_velocity_damping
from .cuda_compile_rrfs import RRFSKernelCache
import time
from .cuda_driver import (_validate_v841_host_execution, _validate_v841_host_coefficients, DryDycoreConfig, PrognosticState, DrySavedDiagnostics, KernelCache, CudaAuthorityRulerBinder, V841ReferenceWindProfiles, CudaAdvectionCoefficients, CudaV841Context, build_advection_coefficients, build_v841_acoustic_offcentering, _frozen_vertical_damping, DryDycoreDriver, cuda_authority_initial_fingerprint, cuda_configuration_payload, CUDA_T0_REBUILT_DIAGNOSTICS, CUDA_T0_EXACT_SIDECAR)
from .cuda_driver import (CUDA_LAYOUT_CONTRACT, CUDA_V841_AUTHORITY_NONCLAIMS, CUDA_V841_IMPLEMENTED_UNLINKED_EVIDENCE, ConfigurationRefusal, CudaAcousticForcing, CudaV841GfDynamicsTendencies, CudaV841PhysicsStepReceipt, CudaV841StepReceipt, DeviceAtmosphere, SPLIT_FLUX_REDUCTION, TransferStats, V841_CUDA_SOURCE, V841_MOIST_DYNAMICS_NEGATIVE_QV_POLICY, V841_MOIST_DYNAMICS_SOURCE, V841_MOIST_DYNAMICS_SOURCE_SHA256, _sum_transfer_stats, _v841_physics_cadences, _v841_physics_receipt_lane, accumulate_finite_array_cuda_v841, accumulate_split_flux_cuda_v841, canonical_sha256, convert_w_tendency_to_omega_cuda, enforce_rw_endpoints_cuda_v841, finish_split_flux_cuda_v841, recover_state, species_row_for_selector, validate_recovered_state_cuda_v841)
from .cuda_driver import (Any, CudaAcousticState, CudaDeviceStepResult, CudaV841PhysicsStepCandidate, DevicePrognosticState, DeviceSavedDiagnostics, RKSchedule, _CudaDynamicsSubcycleResult, _CudaV841MoistDynamicsCoefficients)


RRFS_CUDA_SOURCE = (
    'RRFSx/MPAS-Model commit=1d6e370e7b43a332074809f170532a216c2212fb; '
    'MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:1928-7759; '
    'CUDA implementation; whole-forecast authority unproven'
)
RRFS_IMPLEMENTED_EVIDENCE = 'implemented-rrfs-cuda-dycore-no-forecast-authority-claim'
RRFS_AUTHORITY_NONCLAIMS = ('strict fork whole forecast', 'default G1/G2 trial merge')


class CudaRRFSDycoreDriver(CudaDryDycoreDriver):
    """Fork mpas_atm_time_integration.F:1928-7759 at 1d6e370e."""
    # MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:1928-7759 at 1d6e370e.
    def __init__(self,*args,**kwargs):
        cache=kwargs.get('kernel_cache')
        if not isinstance(cache,RRFSKernelCache):
            kwargs['kernel_cache']=RRFSKernelCache(**({} if cache is None else {'capability':cache.capability,'cache_dir':cache.cache_directory}))
        super().__init__(*args,**kwargs)
        if not isinstance(self.config,RRFSDryDycoreConfig):
            raise TypeError('RRFS numerics require the table-resolved RRFS configuration')
        self.horizontal=CudaHorizontalRRFS(self.atmosphere.mesh,self.nlev,
            self.v841_context,kernel_cache=self.cache)
        from dataclasses import fields
        from .mixing_rrfs import RRFSMixingConfig
        self.mixing_config_v841=RRFSMixingConfig(**{f.name:getattr(self.config,f.name) for f in fields(RRFSMixingConfig)})
        self._rrfs_kdiff=None
        self._rrfs_schedule_cache = None

    def _rrfs_schedules(self, dt, dynamics_splits):
        """Retain the two immutable schedules while their configuration is unchanged."""
        key = (dt, self.config.config_time_integration_order,
               self.config.config_number_of_sub_steps, dynamics_splits)
        cached = self._rrfs_schedule_cache
        if cached is None or cached[0] != key:
            dynamics = RKSchedule.from_mpas(dt, order=key[1], acoustic_substeps=key[2],
                                            dynamics_splits=dynamics_splits)
            scalars = RKSchedule.from_mpas(dt, order=key[1], acoustic_substeps=key[2],
                                          dynamics_splits=1)
            cached = self._rrfs_schedule_cache = (key, dynamics, scalars)
        return cached[1], cached[2]

    # MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:5904-5915,6147-6155 at 1d6e370e.
    def _launch(self,name,count,args):
        # Native pool adv_coefs_3rd already includes the configured third-order
        # coefficient. The shared host-prepared carrier stores it separately.
        # Vertical transport still receives the native coefficient unchanged.
        if getattr(self,'_rrfs_native_advection_coefficients',False):
            slot=5 if name in ('theta_edge_flux_f32','theta_edge_flux_hoist_f32') else 4 if name in ('w_edge_flux_f32','w_edge_flux_hoist_f32') else None
            if slot is not None:
                args=(*args[:slot],np.float32(1.0),*args[slot+1:])
        super()._launch(name,count,args)

    @staticmethod
    # MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:1928-7759 at 1d6e370e.
    def _validate_config(config):
        """Fork time integration 5393–6409 at 1d6e370e: fixed/vertical paths active."""
        neutral=replace(config,config_h_mom_eddy_visc2=0.0,config_h_theta_eddy_visc2=0.0,
            config_h_mom_eddy_visc4=0.0,config_h_theta_eddy_visc4=0.0,
            config_v_mom_eddy_visc2=0.0,config_v_theta_eddy_visc2=0.0,
            config_dynamics_split_steps=3,config_split_dynamics_transport=True)
        CudaDryDycoreDriver._validate_config(neutral)

    @classmethod
    # MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:1928-7759 at 1d6e370e.
    def from_host(
        cls,
        mesh: Any,
        state: PrognosticState,
        vertical: Any,
        reference: Any,
        config: DryDycoreConfig,
        *,
        saved_diagnostics: DrySavedDiagnostics | None = None,
        terrain_metrics: Any,
        advection_coefficients: Any | None = None,
        kernel_cache: KernelCache | None = None,
        authority_ruler: CudaAuthorityRulerBinder | None = None,
        reference_wind_profiles: V841ReferenceWindProfiles | None = None,
        theta_initial: Any | None = None,
    ) -> "CudaRRFSDycoreDriver":
        """Fork mpas_atm_core.F:1266-1309, advection.F:743-939 at 1d6e370e."""

        from .deformation_rrfs import select_deformation_weights_rrfs,initialize_edge_trigonometry_rrfs
        kernel_cache = RRFSKernelCache() if kernel_cache is None else kernel_cache
        if not isinstance(kernel_cache, RRFSKernelCache):
            kernel_cache = RRFSKernelCache(capability=kernel_cache.capability,
                cache_dir=kernel_cache.cache_directory)
        config.validate()
        cls._validate_config(config)
        source_release = getattr(config, "source_release", "v8.2.3")
        is_v841 = source_release == "v8.4.1"
        if is_v841:
            from .config_v841 import V841DryDycoreConfig

            if not isinstance(config, V841DryDycoreConfig):
                raise ConfigurationRefusal(
                    "source_release",
                    source_release,
                    "receipt relabeling cannot select v8.4.1 CUDA numerics",
                    "V841DryDycoreConfig(source_release='v8.4.1')",
                )
            if authority_ruler is not None:
                raise ConfigurationRefusal(
                    "authority_ruler",
                    "provided",
                    "the v8.4.1 CUDA path has no released native nonzero-tracer ruler",
                    "authority_ruler=None",
                )
            nlev_host, ncells_host, nedges_host = (
                _validate_v841_host_execution(
                    mesh,
                    state,
                    vertical,
                    reference,
                    saved_diagnostics,
                    terrain_metrics,
                )
            )
            if np.dtype(state.rho.dtype) != np.dtype(np.float32):
                raise TypeError("the v8.4.1 CUDA mirror requires float32 state")
            if reference_wind_profiles is None:
                # Fork 5524-5553 has no reference-wind Coriolis subtraction.
                # Only perturbation vertical mixing reads the initialized
                # profiles (MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:5755-5828 at 1d6e370e).
                if not config.config_mix_full and config.config_v_mom_eddy_visc2 > 0:
                    raise ValueError('native perturbation momentum mixing reads u_init/v_init; supply their initialized pool words')
                reference_wind_profiles = V841ReferenceWindProfiles(
                    u_init=np.zeros(nlev_host, dtype=np.float32),
                    v_init=np.zeros(nlev_host, dtype=np.float32))
            reference_wind_profiles.validate(nlev_host, np.dtype(np.float32))
            offcentering = build_v841_acoustic_offcentering(
                np.asarray(vertical.rdzw),
                minimum=config.config_epssm_minimum,
                maximum=config.config_epssm_maximum,
                transition_bottom_z=config.config_epssm_transition_bottom_z,
                transition_top_z=config.config_epssm_transition_top_z,
            )
        else:
            if reference_wind_profiles is not None:
                raise ConfigurationRefusal(
                    "u_init",
                    "provided",
                    "v8.2.3 CUDA does not consume v8.4.1 reference profiles",
                    "omit reference_wind_profiles",
                )
            offcentering = None
        if authority_ruler is not None and not isinstance(
            authority_ruler, CudaAuthorityRulerBinder
        ):
            raise TypeError(
                "authority_ruler must be a validated CudaAuthorityRulerBinder"
            )
        coefficients = advection_coefficients
        if coefficients is None:
            coefficients = build_advection_coefficients(
                mesh,
                config_scalar_adv_order=config.config_scalar_adv_order,
                n_vert_levels=state.rho.shape[0],
                source_order_v841=is_v841,
            )
        if is_v841:
            _validate_v841_host_coefficients(
                coefficients,
                n_cells=ncells_host,
                n_edges=nedges_host,
            )
        # The initialized file's dss is only a placeholder.  Core always
        # rebuilds it from config_xnutr/config_zd; doing this even for xnutr=0
        # prevents an input file from silently activating upper damping.
        selected_dss = build_rrfs_vertical_velocity_damping(
            vertical.zgrid, mesh.mesh_density,
            xnutr=config.config_xnutr, damping_start_z=config.config_zd,
        )
        selected_vertical = replace(
            vertical,
            dss=selected_dss,
            **(
                {
                    name: float(np.float32(getattr(vertical, name)))
                    for name in ("cf1", "cf2", "cf3")
                }
                if is_v841
                else {}
            ),
        )
        authority_initial_fingerprint = None
        if authority_ruler is not None:
            authority_saved = saved_diagnostics
            if authority_saved is None:
                authority_saved = DryDycoreDriver(
                    mesh,
                    selected_vertical,
                    reference,
                    config,
                    advection_coefficients=coefficients,
                    terrain_metrics=terrain_metrics,
                    reference_wind_profiles=reference_wind_profiles,
                )._rebuild_saved_diagnostics(state)
            authority_initial_fingerprint = cuda_authority_initial_fingerprint(
                state,
                authority_saved,
                mesh=mesh,
                vertical=vertical,
                reference=reference,
                terrain_metrics=terrain_metrics,
                config=config,
                advection_coefficients=coefficients,
            )
            authority_ruler.validate(
                canonical_sha256(cuda_configuration_payload(config)),
                authority_initial_fingerprint,
            )
        v841_context = None
        if is_v841:
            assert offcentering is not None
            assert reference_wind_profiles is not None
            v841_context = CudaV841Context.from_host(
                mesh,
                offcentering,
                reference_wind_profiles,
                n_vert_levels=state.rho.shape[0],
                dtype=np.float32,
            )
        atmosphere = DeviceAtmosphere.from_host(
            mesh,
            state,
            selected_vertical,
            reference,
            saved_diagnostics,
            terrain_metrics,
            dtype=np.float32,
            index_dtype=np.int32,
        )
        device_coefficients = CudaAdvectionCoefficients.from_host(coefficients)
        diagnostics_source = (
            CUDA_T0_REBUILT_DIAGNOSTICS
            if saved_diagnostics is None
            else CUDA_T0_EXACT_SIDECAR
        )
        driver = cls(
            atmosphere,
            config,
            device_coefficients,
            kernel_cache=kernel_cache,
            t0_diagnostics_source=diagnostics_source,
            authority_ruler=authority_ruler,
            v841_context=v841_context,
            _authority_initial_fingerprint=authority_initial_fingerprint,
        )
        weights = select_deformation_weights_rrfs(
            mesh, on_a_sphere=str(mesh.attrs.get('on_a_sphere','NO')).strip().upper()=='YES',
            sphere_radius=float(mesh.attrs.get('sphere_radius',1.0)),
        )
        driver.atmosphere.mesh = replace(driver.atmosphere.mesh,
            defc_a=driver.cp.asarray(weights['defc_a']), defc_b=driver.cp.asarray(weights['defc_b']))
        driver.horizontal.mesh = driver.atmosphere.mesh
        driver._rrfs_edge_cos,driver._rrfs_edge_sin=(driver.cp.asarray(x) for x in initialize_edge_trigonometry_rrfs(mesh.angleEdge))
        theta_initial=getattr(mesh,'t_init',None) if theta_initial is None else theta_initial
        if theta_initial is None:
            if not config.config_mix_full and config.config_v_theta_eddy_visc2>0:
                raise ValueError('native perturbation theta mixing reads t_init; supply its initialized pool words')
            driver._rrfs_theta_init=driver.cp.zeros_like(driver.atmosphere.state.rho)
        else:
            driver._rrfs_theta_init=driver.cp.asarray(theta_initial,dtype=driver.cp.float32)
        if saved_diagnostics is None:
            driver.atmosphere.saved = driver._rebuild_saved_diagnostics()
        if authority_ruler is not None:
            driver._arm_authority_snapshot()
        return driver

    # MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:1928-7759 at 1d6e370e.
    def _vertical_coefficients(self,state,dts,theta_m,rtheta_p,*,exner=None,
        validation_flag=None,cqw=None,qtot=None):
        """Fork mpas_atm_time_integration.F:2225-2366 at 1d6e370e."""
        return compute_vertical_implicit_coefficients_cuda_rrfs(dts=dts,
            epssm=self.config.config_epssm,zz=self.atmosphere.vertical.zz,
            cqw=self.cp.ones_like(state.rho) if cqw is None else cqw,
            pressure=self.atmosphere.saved.exner if exner is None else exner,
            theta=theta_m,rho_base=self.atmosphere.reference.rho_base,
            rho_theta_base=self.atmosphere.reference.rho_theta_base,
            pressure_base=self.atmosphere.reference.exner_base,rho_theta_perturbation=rtheta_p,
            qtot=self.cp.zeros_like(state.rho) if qtot is None else qtot,
            rdzw=self.atmosphere.vertical.rdzw,fzm=self.atmosphere.vertical.fzm,
            fzp=self.atmosphere.vertical.fzp,rdzu=self.atmosphere.vertical.rdzu,
            cell_solve_count=None if self.regional_v841 is None else self.regional_v841.n_cells_solve,
            kernel_cache=self.cache)

    # MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:1928-7759 at 1d6e370e.
    def _vector_momentum(self,u,rho_edge,pv_edge,kinetic,mass_divergence,initial_tendency=None):
        """Fork mpas_atm_time_integration.F:5524-5553 at 1d6e370e."""
        return vector_momentum_tendency_cuda_rrfs(self.atmosphere.mesh,self.v841_context,
            normal_velocity=u,rho_edge=rho_edge,pv_edge=pv_edge,kinetic_energy=kinetic,
            horizontal_divergence=mass_divergence,initial_tendency=initial_tendency,kernel_cache=self.cache)

    # MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:1928-7759 at 1d6e370e.
    def _nearest_relaxation_cell_rrfs(self):
        """Fork mpas_atm_boundaries.F:702-751 at 1d6e370e; shared indices."""
        if not hasattr(self,'_rrfs_nearest'):
            host=self.regional_v841.masks_host.nearest_relaxation_cell
            self._rrfs_nearest=self.cp.full(self.ncells,self.ncells-1,dtype=self.cp.int32)
            self._rrfs_nearest[:len(host)]=self.cp.asarray(host,dtype=self.cp.int32)
        return self._rrfs_nearest

    # MPAS-Model-rrfs/src/core_atmosphere/mpas_atm_core.F:486,1460-1479 at 1d6e370e.
    def _rrfs_coupled_transport_coefficients(self,coefficients):
        """The fork's scalar transport reads adv_coefs_3rd already scaled by config_coef_3rd_order.

        atm_couple_coef_3rd_order multiplies the stored third-order weights once at start-up
        (mpas_atm_core.F:486), and atm_advance_scalars_work / atm_advance_scalars_mono_work then use
        adv_coefs + sign(uhAvg)*adv_coefs_3rd with no further factor (time_integration.F:3850,3892,4539).
        The shared host carrier keeps the third-order weights uncoupled (the dynamics kernels apply the
        coefficient at the flux site), so the RRFS transport kernels receive the native coupled words: one
        single-precision product per weight, as native forms them. Passing the uncoupled carrier made the
        third-order upwind term four times too strong at config_coef_3rd_order=0.25.
        """
        cached=getattr(self,'_rrfs_coupled_coefficients',None)
        if cached is not None and cached[0] is coefficients:
            return cached[1]
        coefficient=self.cp.float32(self.config.config_coef_3rd_order)
        coupled=replace(coefficients,adv_coefs_3rd=self.cp.ascontiguousarray(
            self.cp.multiply(coefficient,coefficients.adv_coefs_3rd,dtype=self.cp.float32)))
        self._rrfs_coupled_coefficients=(coefficients,coupled)
        return coupled

    # MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:1928-7759 at 1d6e370e.
    def _rrfs_scalar_transport(self,*args,**kwargs):
        """Fork mpas_atm_time_integration.F:3615-4023 at 1d6e370e."""
        if not getattr(self,'_rrfs_native_advection_coefficients',False):
            kwargs['coefficients']=self._rrfs_coupled_transport_coefficients(kwargs['coefficients'])
        enabled=self.config.config_scalar_eddy_mix
        length=np.float32(self.config.config_len_disp or self.atmosphere.mesh.nominal_min_dc)
        h4=np.float32(np.float32(self.config.config_h_scalar_filter4)*((length*length)*length))
        return advance_scalar_transport_cuda_rrfs(*args,**kwargs,
            config_scalar_eddy_mix=enabled,config_h_scalar_filter4=h4,
            kdiff=self._rrfs_kdiff,rho_edge=self._rrfs_scalar_rho_edge,
            mesh_scaling_del2=getattr(self.horizontal,"rrfs_scaling_del2",None),mesh_scaling_del4=getattr(self.horizontal,"rrfs_scaling_del4",None),
            mixing_cell_solve_count=None if self.regional_v841 is None else self.regional_v841.n_cells_solve)
    # MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:1928-7759 at 1d6e370e.
    def _recover_candidate(
        self,
        saved_state: DevicePrognosticState,
        saved_diag: DeviceSavedDiagnostics,
        acoustic: CudaAcousticState,
        stage: int,
        acoustic_steps: int,
        scalars: Any,
        *, diabatic_theta: Any | None = None, large_timestep: float = 0.0,
    ) -> tuple[DevicePrognosticState, DeviceSavedDiagnostics, Any, Any]:
        cp = self.cp
        rho = cp.empty_like(saved_state.rho)
        rtheta = cp.empty_like(saved_state.rho_theta)
        rho_p = cp.empty_like(saved_state.rho)
        rtheta_p = cp.empty_like(saved_state.rho_theta)
        theta = cp.empty_like(saved_state.rho)
        exner = cp.empty_like(saved_diag.exner)
        pressure_p = cp.empty_like(saved_diag.pressure_perturbation)
        self._launch(
            "recover_cells_f32",
            self.ncells * self.nlev,
            (
                np.int32(self.nlev),
                np.int32(self.ncells),
                np.int32(stage),
                np.int32(self.source_release == "v8.4.1"),
                np.float32(287.0),
                np.float32(1004.5),
                np.float32(100_000.0),
                saved_diag.density_perturbation,
                saved_diag.rho_theta_perturbation,
                acoustic.rho_pp,
                acoustic.rtheta_pp,
                np.int32(diabatic_theta is not None),
                np.float32(large_timestep),
                saved_diag.density_perturbation if diabatic_theta is None else diabatic_theta,
                self.atmosphere.reference.rho_base,
                self.atmosphere.reference.rho_theta_base,
                self.atmosphere.reference.exner_base,
                self.atmosphere.vertical.zz,
                saved_diag.exner,
                saved_diag.pressure_perturbation,
                rho,
                rtheta,
                rho_p,
                rtheta_p,
                theta,
                exner,
                pressure_p,
            ),
        )
        ru = cp.empty_like(saved_state.rho_u)
        flux_u = cp.empty_like(saved_state.rho_u)
        self._launch(
            "recover_edges_f32",
            self.nedges * self.nlev,
            (
                np.int32(self.nlev),
                np.int32(self.nedges),
                np.int32(acoustic_steps),
                saved_state.rho_u,
                acoustic.ru_p,
                acoustic.ru_avg,
                ru,
                flux_u,
            ),
        )
        rw = cp.empty_like(saved_state.rho_w)
        flux_w = cp.empty_like(saved_state.rho_w)
        self._launch(
            "recover_interfaces_f32",
            self.ncells * (self.nlev + 1),
            (
                np.int32(self.nlev),
                np.int32(self.ncells),
                np.int32(acoustic_steps),
                saved_state.rho_w,
                acoustic.rw_p,
                acoustic.ww_avg,
                rw,
                flux_w,
            ),
        )
        if self.v841_context is not None:
            enforce_rw_endpoints_cuda_v841(rw, kernel_cache=self.cache)
        regional_u_overwrite = None
        if self.regional_v841 is not None:
            # driver.py:2228-2265 and 2390-2394: every regional recover runs
            # in native's padded memory model with rho_zz's garbage cell at
            # 1.0 (F:4385-4392), so the shared recover_edges kernel divides a
            # one-cell ring-7 edge by rho(present)+1.0 exactly as
            # regional_normal_velocity does on the host.
            self.regional_v841.bind_state_rho(rho)
            # driver.py:2935-2937 -- the ru half of atm_srk3:2442-2485,
            # before the candidate is validated and before u is recovered.
            regional_u_overwrite = self.regional_v841.overwrite_speczone_rho_u(
                ru, stage
            )
        # The candidate carries the caller's scalar block by reference.  Nothing
        # between here and the transport stage writes through it: the acoustic
        # solver, the tendency helpers and the recovery kernels all read the
        # scalars and allocate their own outputs, and both callers REBIND
        # ``.scalars`` to a transport result rather than filling it in place.
        # Copying it here allocated a whole six-species block (nScalars x nlev x
        # nCells) per RK stage, three times per dynamics subcycle, and every one
        # of those copies was dropped unmodified.
        state = self._device_state(rho, rtheta, ru, rw, scalars)
        temporary = DeviceAtmosphere(
            self.atmosphere.mesh,
            state,
            self.atmosphere.vertical,
            self.atmosphere.reference,
            saved_diag,
            self.atmosphere.terrain,
            self.atmosphere.h2d,
        )
        # Only normal_velocity and vertical_velocity are read below; theta,
        # exner, pressure, rho_p, rtheta_p and pressure_p come from
        # recover_cells_f32 above, so asking recover_state for its own copies
        # allocated and filled six cell fields per stage for the bin.
        recovered = recover_state(
            temporary,
            cache=self.cache,
            warmup=0,
            timing_repeats=0,
            include_pressure=False,
        )
        if self.regional_v841 is not None:
            # driver.py:3005-3009 -- the u half of the same native overwrite,
            # with the SAME interpolated driving values the ru half used --
            # and driver.py:2405, atm_zero_gradient_w_bdy (F:7868-7902).
            self.regional_v841.overwrite_speczone_u(
                recovered.normal_velocity, regional_u_overwrite
            )
            # Fork atm_recover_large_step_variables (mpas_atm_time_integration.F:3304-3405 at 1d6e370e) sets
            # w(1) = 0 in every cell and adds the terrain-following lower-boundary flux only where bdyMaskCell <=
            # nRelaxZone ("no spec zone update").  The fork's atm_zero_gradient_w_bdy then sets levels 2..nVertLevels
            # alone (apply_w_boundary_cuda_rrfs), unlike v8.4.1's whole-column zero, so the specified zone's surface
            # w must be the recovery's 0.  THE BREAKAGE THIS FIXES: hex kept the terrain flux there (lam7k step 1:
            # 760 surface w words in rings 6-7, up to 0.03 m/s where native has 0).
            masks = self.regional_v841.masks
            if int(masks.n_spec_cells):
                recovered.vertical_velocity[0, masks.spec_cells] = 0
        diagnostics = self._device_saved(
            theta,
            exner,
            rho_p,
            rtheta_p,
            pressure_p,
            recovered.normal_velocity,
            recovered.vertical_velocity,
        )
        return state, diagnostics, flux_u, flux_w

    # MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:1928-7759 at 1d6e370e.
    def _advance_dynamics_subcycle_v841(
        self,
        state: DevicePrognosticState,
        *,
        time_level_one: DeviceSavedDiagnostics,
        initial_diagnostics: Any,
        schedule: RKSchedule,
        outer_dt: float,
        validation_flag: Any,
        physics_tendencies: Any | None = None,
        moist_coefficients: _CudaV841MoistDynamicsCoefficients | None = None,
        physics_stage_callback: Any | None = None,
        dynamics_substep: int = 1,
        acoustic_output_callback: Any | None = None,
    ) -> _CudaDynamicsSubcycleResult:
        """Advance one native chained v8.4.1 dynamics subcycle on device."""

        if self.v841_context is None:
            raise RuntimeError("v8.4.1 device context is missing")
        if (physics_tendencies is not None or physics_stage_callback is not None) and moist_coefficients is None:
            # Full physics without the moist coefficients would run the dry
            # pressure-gradient/buoyancy forms under a moist state.  The other
            # half of the old pairing law (coefficients only WITH physics) is
            # retired: native atm_srk3 (mpas_atm_time_integration.F:2125-2139)
            # forms them on every step whatever the physics suite, so the
            # physics-off global path and the regional lane both carry them.
            raise ValueError(
                "full-physics tendencies require native moist coefficients"
            )
        if moist_coefficients is not None:
            moist_coefficients.validate(
                cp=self.cp,
                n_vert_levels=self.nlev,
                n_cells=self.ncells,
                n_edges=self.nedges,
            )
        if schedule.full_timestep != outer_dt:
            raise ValueError("RRFS subcycle schedule and outer timestep differ")
        # The scalar source is the fork's scalars_tend: physics_get_tend fills it once at the top of atm_srk3
        # (mpas_atm_time_integration.F:1077-1084 at 1d6e370e). Under the native stage callback those sources
        # exist only once the first RK stage has called it, so the buffer is formed there (below), never here.
        coupled_source = (None if physics_tendencies is None and physics_stage_callback is not None
                          else self.cp.zeros_like(state.scalars) if physics_tendencies is None
                          else self.cp.array(physics_tendencies.scalars,copy=True))
        saved_state = self._copy_state(state, share_scalars=True)
        saved_diag = self._copy_saved(time_level_one)
        # RK stage 1 begins at the substep-start state, so current_* and saved_*
        # are the same field values.  Stage 1 READS both handles and writes
        # through neither -- every tendency helper below allocates its own
        # output, the acoustic solver works on CudaAcousticState, and the loop
        # body REBINDS current_state/current_diag to the recovered candidate
        # before stage 2 reads them.  A private image here was a second copy of
        # a read-only field set: five prognostic arrays plus seven diagnostic
        # arrays, allocated once per dynamics subcycle and never written.
        current_state = saved_state
        current_diag = saved_diag
        diagnostics = initial_diagnostics
        initial_diagnostics = None
        cached_v = diagnostics.tangential_velocity
        cp = self.cp
        final_modified_theta_dynamics_rate = None

        coefficients = self._vertical_coefficients(
            saved_state,
            schedule.stages[0].acoustic_timestep,
            saved_diag.theta_m,
            saved_diag.rho_theta_perturbation,
            exner=saved_diag.exner,
            validation_flag=validation_flag,
            cqw=None if moist_coefficients is None else moist_coefficients.cqw,
            qtot=None if moist_coefficients is None else moist_coefficients.qtot,
        )
        if moist_coefficients is None:
            cqu = cp.ones_like(saved_state.rho_u)
            dpdz = self._scale(saved_diag.density_perturbation, -9.80616)
        else:
            cqu = moist_coefficients.cqu
            dpdz = self._moist_dpdz_v841(
                saved_diag.density_perturbation,
                moist_coefficients.qtot,
                validation_flag,
            )
        saved_euler_ru = self.horizontal.pressure_gradient_euler_tendency(
            saved_diag.pressure_perturbation,
            dpdz,
            cqu,
            self.atmosphere.vertical.zz,
            self.atmosphere.vertical.zxu,
        )
        tend_rho_dynamics = self.horizontal.density_tendency(
            saved_state.rho_u,
            saved_state.rho_w,
            self.atmosphere.vertical.rdzw,
        )
        tend_rho_saved = (
            tend_rho_dynamics
            if physics_tendencies is None
            else self.horizontal.density_tendency(
                saved_state.rho_u,
                saved_state.rho_w,
                self.atmosphere.vertical.rdzw,
                physics_tendency=physics_tendencies.rho,
            )
        )
        if self.regional_v841 is not None:
            # atm_compute_dyn_tend_work F:6455-6466 writes tend_rho only at
            # rk_step==1, so the pool carries stage 1's values AND every
            # regional adjustment applied to them into stages 2 and 3, where
            # the regional stages adjust the already-adjusted array again.
            # The CUDA subcycle already forms tend_rho once per subcycle, so
            # the native quirk is inherited; the private image only keeps the
            # in-place regional mutation out of tend_rho_dynamics.
            tend_rho_saved = cp.array(tend_rho_saved, copy=True)
        if moist_coefficients is None:
            saved_euler_rw = cp.empty_like(saved_state.rho_w)
            self._launch(
                "euler_w_f32",
                self.ncells * (self.nlev + 1),
                (
                    np.int32(self.nlev),
                    np.int32(self.ncells),
                    saved_diag.pressure_perturbation,
                    dpdz,
                    self.atmosphere.vertical.rdzu,
                    self.atmosphere.vertical.fzm,
                    self.atmosphere.vertical.fzp,
                    saved_euler_rw,
                ),
            )
        else:
            saved_euler_rw = self._moist_euler_w_v841(
                saved_diag.pressure_perturbation,
                dpdz,
                moist_coefficients.cqw,
                validation_flag,
            )
        saved_euler_theta = None
        if self.mixing_config_v841 is not None:
            # Native computes the horizontal-mixing Euler tendencies once per
            # dynamics substep at rk_step==1 from the substep-start state
            # (mpas_atm_time_integration.F:6329,6585,6725,6885) and reuses
            # them for RK2/RK3.  ``diagnostics`` here is the substep-start
            # solve (tangential velocity, divergence, vorticity, rho_edge).
            saved_mixing = self.horizontal.compute_dry_mixing_tendencies_v841(
                saved_diag.normal_velocity,
                diagnostics.tangential_velocity,
                saved_diag.vertical_velocity,
                saved_diag.theta_m,
                diagnostics.h_edge,
                diagnostics.divergence,
                diagnostics.vorticity,
                dt=outer_dt,
                config=self.mixing_config_v841,
                cam_damping_levels=getattr(self.config,"config_number_cam_damping_levels",4),
                n_cells_solve=None if self.regional_v841 is None else self.regional_v841.n_cells_solve,
                initial_u_tendency=saved_euler_ru,
            )
            # The u filter accumulates into the pressure-gradient seed in
            # native order: (pressure + del2) - del4.  Its return is the
            # complete saved pool, so do not add the seed a second time.
            saved_euler_ru = saved_mixing.tend_u_euler
            # Fork6070-6076 subtracts pressure from accumulated horizontal
            # mixing. Reversing that subtraction into an added precomputed
            # pressure tendency changes rounding and signed-zero storage.
            from .cuda_dynamics_rrfs import accumulate_euler_w_cuda_rrfs
            saved_euler_rw = accumulate_euler_w_cuda_rrfs(
                saved_mixing.tend_w_euler, saved_diag.pressure_perturbation,
                dpdz, cp.ones_like(saved_state.rho) if moist_coefficients is None else moist_coefficients.cqw,
                self.atmosphere.vertical.rdzu,self.atmosphere.vertical.fzm,self.atmosphere.vertical.fzp,
                cell_solve_count=None if self.regional_v841 is None else self.regional_v841.n_cells_solve,
                kernel_cache=self.cache)
            saved_euler_theta = cp.array(
                saved_mixing.tend_theta_euler, copy=True
            )
            saved_mixing_kdiff = saved_mixing.kdiff
            del saved_mixing
            self.mixing_calls_v841 += 1
            if self.config.config_v_mom_eddy_visc2>0 or self.config.config_v_theta_eddy_visc2>0:
                from .cuda_vertical_mixing_rrfs import apply_vertical_mixing_cuda_rrfs
                apply_vertical_mixing_cuda_rrfs(self.atmosphere.mesh,self.v841_context,self.config,
                    zgrid=self.atmosphere.vertical.zgrid,normal=saved_diag.normal_velocity,
                    vertical=saved_diag.vertical_velocity,theta=saved_diag.theta_m,
                    rho=saved_state.rho,rho_edge=diagnostics.h_edge,
                    rdzw=self.atmosphere.vertical.rdzw,rdzu=self.atmosphere.vertical.rdzu,
                    theta_init=self._rrfs_theta_init,edge_cos=self._rrfs_edge_cos,edge_sin=self._rrfs_edge_sin,
                    tend_u=saved_euler_ru,tend_w=saved_euler_rw,tend_theta=saved_euler_theta,
                    cell_solve_count=None if self.regional_v841 is None else self.regional_v841.n_cells_solve,
                    edge_solve_count=None if self.regional_v841 is None else self.regional_v841.n_edges_solve,
                    kernel_cache=self.cache)
        if self.mixing_config_v841 is not None:
            self._rrfs_kdiff = saved_mixing_kdiff
        final_flux_u = None
        final_flux_w = None

        for stage in schedule.stages:
            # mpas_atm_time_integration.F:2146-2165: reweight held rates by
            # current RK mass. The callback retains outer time-level-one
            # theta/scalars, performs native halo/projection work, and returns
            # separate sources. MPAS's density physics source is zero.
            if physics_stage_callback is not None:
                physics_tendencies = physics_stage_callback(
                    rk_step=stage.stage, dynamics_substep=dynamics_substep,
                    state=current_state, saved=current_diag,
                    diagnostics=diagnostics,
                )
                if physics_tendencies is None:
                    raise RuntimeError("active MPAS RK callback returned no sources")
                if coupled_source is None:
                    coupled_source = self.cp.array(physics_tendencies.scalars,copy=True)
            if stage.stage == 2 and self.config.config_time_integration_order == 3:
                coefficients = self._vertical_coefficients(
                    current_state,
                    stage.acoustic_timestep,
                    current_diag.theta_m,
                    current_diag.rho_theta_perturbation,
                    exner=saved_diag.exner,
                    validation_flag=validation_flag,
                    cqw=None if moist_coefficients is None else moist_coefficients.cqw,
                    qtot=None if moist_coefficients is None else moist_coefficients.qtot,
                )
            mass_divergence = self.horizontal.mass_flux_divergence(
                current_state.rho_u
            )
            tend_ru = self._vertical_u(
                current_diag.normal_velocity, current_state.rho_w
            )
            tend_ru = self._vector_momentum(
                current_diag.normal_velocity,
                diagnostics.h_edge,
                diagnostics.pv_edge,
                diagnostics.kinetic_energy,
                mass_divergence,
                initial_tendency=tend_ru,
            )
            del mass_divergence
            if self.config.config_rayleigh_damp_u:
                from .cuda_upper_damping_v841 import apply_rayleigh_u_cuda
                apply_rayleigh_u_cuda(tend_ru,diagnostics.h_edge,current_diag.normal_velocity,
                    levels=getattr(self.config,"config_number_rayleigh_damp_u_levels",6),
                    days=getattr(self.config,"config_rayleigh_damp_u_timescale_days",5.0),
                    solved=None if self.regional_v841 is None else self.regional_v841.n_edges_solve,
                    cache=self.cache)
            self._add(tend_ru, saved_euler_ru)
            if physics_tendencies is not None:
                self._add(tend_ru, physics_tendencies.rho_u)
            tend_rt = self._theta_tendency(
                current_state,
                current_diag.theta_m,
                saved_state,
                saved_diag.theta_m,
                stage.stage,
            )
            if physics_tendencies is not None and stage.stage == 3:
                final_modified_theta_dynamics_rate = cp.empty_like(tend_rt)
                # Fork mpas_atm_time_integration.F:6291 forms rthdynten from the tend_rho POOL.  On a limited-area run
                # that pool is stage 1's tend_rho plus every regional adjustment of stages 1 and 2 (it is written only
                # at rk_step 1; atm_srk3:1217-1240 adjusts it in place each stage), which is tend_rho_saved here.
                # THE BREAKAGE THIS FIXES: rthdynten in boundary rings 2-7 took the unadjusted tendency (lam7k step 1:
                # -0.20 vs native -1.36 K/s at a ring-7 cell), and microphysics reads rthdynten.  A closed mesh keeps
                # the term it was proven with.
                rho_rate = tend_rho_saved if self.regional_v841 is not None else tend_rho_dynamics
                self._launch_physics_driver(
                    "modified_theta_dynamics_rate_v841_f32",
                    self.ncells,
                    (
                        np.int32(self.nlev),
                        np.int32(self.ncells),
                        tend_rt,
                        rho_rate,
                        current_state.rho,
                        current_diag.theta_m,
                        final_modified_theta_dynamics_rate,
                        validation_flag,
                    ),
                )
            # MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:6291-6293 at 1d6e370e.
            diabatic_theta = getattr(physics_tendencies, 'diabatic_theta', None)
            if diabatic_theta is not None:
                self._launch_physics_driver(
                    'add_lagged_diabatic_theta_v842_f32', self.ncells,
                    (np.int32(self.nlev), np.int32(self.ncells),
                     current_state.rho, diabatic_theta, tend_rt),
                )
            if saved_euler_theta is not None:
                # Fork6291-6293/6403: form rthdynten, add lagged heating,
                # then saved mixing and physics in native order.
                self._add(tend_rt, saved_euler_theta)
            if physics_tendencies is not None:
                self._add(tend_rt, physics_tendencies.rho_theta)
            tend_rw = self._w_tendency(
                current_state, current_diag.vertical_velocity
            )
            self._add(tend_rw, saved_euler_rw)
            tend_omega = convert_w_tendency_to_omega_cuda(
                self.atmosphere.mesh,
                tend_rw,
                tend_ru,
                fzm=self.atmosphere.vertical.fzm,
                fzp=self.atmosphere.vertical.fzp,
                zz=self.atmosphere.vertical.zz,
                zb_cell=self.atmosphere.terrain.zb_cell,
                zb3_cell=self.atmosphere.terrain.zb3_cell,
                kernel_cache=self.cache,
            )
            del tend_rw
            if self.regional_v841 is not None:
                # atm_srk3:2300-2351 -- the specified-zone tendency
                # assignment and the relaxation-zone Rayleigh/Laplacian
                # stages, against the SHARED tend_rho pool and the omega
                # tendency (not tend_rw), exactly as driver.py:2734-2782.
                self.regional_v841.adjust_dynamics_tendencies(
                    tend_ru=tend_ru,
                    tend_rho=tend_rho_saved,
                    tend_rt=tend_rt,
                    tend_omega=tend_omega,
                    rho_u=current_state.rho_u,
                    theta_m=current_diag.theta_m,
                    rho_zz=current_state.rho,
                    rk_step=stage.stage,
                )
                if diabatic_theta is not None:
                    # Fork atm_bdy_adjust_dynamics_speczone_tend (mpas_atm_time_integration.F:7471-7498 at 1d6e370e)
                    # zeroes rt_diabatic_tend in the specified zone, so later stages and the final recovery
                    # (F:3330) carry no lagged heating there; cuda_driver's MPAS-A route does the same (SG2).
                    # THE BREAKAGE THIS FIXES: hrrrv5 on a limited-area mesh matched native through step 1 (no lagged
                    # heating yet) and left it at step 2 in every relaxation-zone column (lam7k: u, theta_m, rho_zz
                    # in rings 1-5), through the relaxation filter's reads of specified-zone neighbours.
                    self._zero_speczone_lagged_heating(diabatic_theta)
            acoustic = CudaAcousticState.zeros(self.nlev, self.ncells, self.nedges)
            forcing = CudaAcousticForcing(
                rho_zz=current_state.rho,
                theta_m=saved_diag.theta_m,
                zz=self.atmosphere.vertical.zz,
                exner=saved_diag.exner,
                cqu=cqu,
                zxu=self.atmosphere.vertical.zxu,
                dss=self.atmosphere.vertical.dss,
                tend_ru=tend_ru,
                tend_rho=tend_rho_saved,
                tend_rt=tend_rt,
                tend_rw=tend_omega,
                w=current_diag.vertical_velocity,
                rw=current_state.rho_w,
                rw_save=saved_state.rho_w,
            )
            # Only cached_v survives this stage's diagnostic solve inputs.
            self._rrfs_scalar_rho_edge=diagnostics.h_edge
            diagnostics = None
            for small_step in range(1, stage.acoustic_steps + 1):
                # native mpas_atm_time_integration.F:3959 / :3968 -- the damping
                # reference is zeroed on the first sub-step and otherwise holds
                # rtheta_pp as it stood BEFORE this sub-step's advance.
                rtheta_pp_old = self.horizontal.capture_rtheta_pp_old(
                    acoustic.rtheta_pp,
                    small_step=small_step,
                )
                acoustic = advance_acoustic_step_cuda_rrfs(
                    self.atmosphere.mesh,acoustic,forcing,coefficients,
                    context=self.v841_context,dts=stage.acoustic_timestep,
                    small_step=small_step,epssm=self.config.config_epssm,
                    fzm=self.atmosphere.vertical.fzm,fzp=self.atmosphere.vertical.fzp,
                    rdzw=self.atmosphere.vertical.rdzw,
                    cell_solve_count=None if self.regional_v841 is None else self.regional_v841.n_cells_solve,
                    spec_zone_mask_cell=None if self.regional_v841 is None else self.regional_v841.spec_zone_mask_cell,
                    spec_zone_mask_edge=None if self.regional_v841 is None else self.regional_v841.spec_zone_mask_edge,
                    in_place=True,kernel_cache=self.cache)
                interior_mask = getattr(self.halo_exchanger_v841, "interior_edge_mask", None)
                if self.halo_exchanger_v841 is not None:
                    # Round C: between the advance and the damping, because the
                    # damping at an owned cut edge reads THIS sub-step's ring-1
                    # rtheta_pp, which only its owner computes exactly (the
                    # ring-1 column's divergence reads ring-2 edges outside the
                    # K=2 cone).  ru_p moves in the same frame so the peer
                    # damps owner-truth momentum.
                    interior_work = None
                    if interior_mask is not None:
                        # MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:1928-7759 at 1d6e370e.
                        def interior_work():
                            self.horizontal.divergence_damping_3d(
                                acoustic.ru_p, saved_diag.theta_m,
                                acoustic.rtheta_pp, rtheta_pp_old,
                                native_stored_inverse=True,
                                dts=stage.acoustic_timestep,
                                config_smdiv=self.config.config_smdiv,
                                config_len_disp=self.config.config_len_disp,
                                in_place=True, _partition_edges=(interior_mask, 1),
                                **({} if self.regional_v841 is None else
                                   {"n_cells_solve": self.regional_v841.n_cells_solve}),
                            )
                    if interior_work is None:
                        self.halo_exchanger_v841.round_acoustic(acoustic)
                    else:
                        self.halo_exchanger_v841.round_acoustic(acoustic, interior_work=interior_work)
                # native :2395-2406 -- "complete update of horizontal momentum by
                # including 3d divergence damping at the end of the acoustic
                # step".  The authority's call carries no branch and no config
                # flag, so this one carries none either: config_smdiv alone sets
                # the strength (Registry.xml:264 default 0.1) and a zero
                # coefficient is the only way to switch the term off, which is
                # exactly how native behaves.
                acoustic.ru_p = self.horizontal.divergence_damping_3d(
                    acoustic.ru_p,
                    saved_diag.theta_m,
                    acoustic.rtheta_pp,
                    rtheta_pp_old,
                    native_stored_inverse=True,
                    dts=stage.acoustic_timestep,
                    config_smdiv=self.config.config_smdiv,
                    config_len_disp=self.config.config_len_disp,
                    in_place=True,
                    **({} if interior_mask is None else {"_partition_edges": (interior_mask, 0)}),
                    # atm_divergence_damping_3d F:4183 scales the update by
                    # (1 - specZoneMaskEdge), which the shared kernel already
                    # carries; the regional lane only has to name the SOLVE
                    # cell count, because the kernel's two-cell early-out
                    # compares against it and the padded garbage cell sits at
                    # exactly that index (driver.py:2875-2898).
                    **(
                        {}
                        if self.regional_v841 is None
                        else {
                            "n_cells_solve": self.regional_v841.n_cells_solve
                        }
                    ),
                )
            # All acoustic tendency reads are queued on this rank stream.
            # Retire their owners before allocating the recovered endpoint.
            forcing = None
            interior_work = None
            del tend_ru, tend_rt, tend_omega, rtheta_pp_old
            current_state, current_diag, flux_u, flux_w = self._recover_candidate(
                saved_state,
                saved_diag,
                acoustic,
                stage.stage,
                stage.acoustic_steps,
                current_state.scalars,
                diabatic_theta=diabatic_theta, large_timestep=stage.large_timestep,
            )
            if stage.stage == 3 and acoustic_output_callback is not None:
                # Registry.xml (fork 1d6e370e) puts the final acoustic ru_p/rw_p in the restart stream, as v8.4.2
                # does. Without this a restart (or any stream naming them) refused the step it was due on.
                acoustic_output_callback(ru_p=acoustic.ru_p, rw_p=acoustic.rw_p)
            acoustic = None
            if self.regional_v841 is None:
                validate_recovered_state_cuda_v841(
                    current_state,
                    current_diag,
                    invalid_flag=validation_flag,
                    kernel_cache=self.cache,
                )
            else:
                # The same test, term for term, over the elements native
                # solves: the shared kernel is launched over the padded
                # extent and would refuse the garbage element for holding
                # the native allocation zero where it demands a positive
                # density.  See CudaRegionalRuntimeV841.validate_recovered.
                self.regional_v841.validate_recovered(
                    current_state, current_diag, validation_flag
                )
            if self.halo_exchanger_v841 is not None:
                # Round B (design 44-round schedule): the recovered state and
                # saved diagnostics become owner truth on the halo BEFORE
                # solve_diagnostics reads ring-1 rho at owned cut edges.
                self.halo_exchanger_v841.round_stage_entry(
                    current_state, current_diag
                )
            if self.config.config_scalar_advection and not self.config.config_split_dynamics_transport:
                # Fork atm_srk3:1403–1433: each dynamics RK transports scalars
                # with its own averaged acoustic flux, before solve diagnostics.
                if self.halo_exchanger_v841 is not None:
                    self.halo_exchanger_v841.round_transport(current_state.scalars)
                options={}
                if self.regional_v841 is not None:
                    options=dict(config_apply_lbcs=True,
                        bdy_mask_cell=self.regional_v841.bdy_mask_cell,
                        bdy_mask_edge=self.regional_v841.bdy_mask_edge,
                        garbage_cell=int(self.regional_v841.padded.garbage_cell),
                        bind_density_pool=self.regional_v841.bind_state_rho)
                transported=self._rrfs_scalar_transport(
                    self.atmosphere.mesh,self.v841_context,saved_state.scalars,
                    current_state.scalars,saved_state.rho,current_state.rho,
                    flux_u,flux_w,stage.large_timestep,
                    coefficients=self.coefficients,fzm=self.atmosphere.vertical.fzm,
                    fzp=self.atmosphere.vertical.fzp,rdzw=self.atmosphere.vertical.rdzw,
                    rk_step=stage.stage,config_time_integration_order=self.config.config_time_integration_order,
                    config_coef_3rd_order=self.config.config_coef_3rd_order,
                    config_scalar_advection=True,config_monotonic=self.config.config_monotonic,
                    config_positive_definite=self.config.config_positive_definite,
                    config_split_dynamics_transport=False,
                    config_scalar_adv_order=3,config_scalar_vadv_order=3,
                    scalar_tendency=coupled_source,validation_flag=validation_flag,
                    kernel_cache=self.cache,halo_exchange=self.halo_exchanger_v841,**options)
                current_state.scalars=transported.scalars
                if self.regional_v841 is not None:
                    self.regional_v841.bdy_adjust_scalars(current_state.scalars,stage.stage)
                accumulate_finite_array_cuda_v841(current_state.scalars,
                    invalid_flag=validation_flag,kernel_cache=self.cache)
            if stage.stage == 3:
                final_flux_u = flux_u
                final_flux_w = flux_w
            diagnostics = self.horizontal.solve_diagnostics(
                current_state.rho,
                current_state.rho_u,
                dt=outer_dt,
                apvm_upwinding=self.config.config_apvm_upwinding,
                normal_velocity=current_diag.normal_velocity,
                cached_tangential_velocity=cached_v,
                rk_step=stage.stage,
            )
            if self.regional_v841 is not None:
                # Fork atm_srk3:1477–1489 follows solve diagnostics and halo work.
                apply_w_boundary_cuda_rrfs(current_diag.vertical_velocity,
                    self.regional_v841.bdy_mask_cell,self._nearest_relaxation_cell_rrfs(),
                    config_lbc_w=self.config.config_lbc_w,
                    cell_solve_end=self.regional_v841.n_cells_solve,kernel_cache=self.cache)
            cached_v = diagnostics.tangential_velocity
            if stage.stage != 3:
                del flux_u, flux_w

        if final_flux_u is None or final_flux_w is None:
            raise RuntimeError("RRFS dynamics subcycle did not reach its third RK stage")
        return _CudaDynamicsSubcycleResult(
            state=current_state,
            saved=current_diag,
            diagnostics=diagnostics,
            mass_flux_u=final_flux_u,
            mass_flux_w=final_flux_w,
            modified_theta_dynamics_rate=final_modified_theta_dynamics_rate,
            stage_physics_tendencies=physics_tendencies,
        )

    # MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:1928-7759 at 1d6e370e.
    def _step_device_v841(
        self, physics_tendencies: Any | None = None, *,
        physics_stage_callback: Any | None = None,
        physics_post_transport_callback: Any | None = None,
        physics_mass_loading_count: int | None = None,
        acoustic_output_callback: Any | None = None,
    ) -> CudaDeviceStepResult | CudaV841PhysicsStepCandidate:
        """Run the complete fork outer step on the shared resident field layout."""

        if self.v841_context is None:
            raise RuntimeError("v8.4.1 device context is missing")
        from .config_v841 import V841MpasColumnPhysicsConfig

        full_physics = isinstance(self.config, V841MpasColumnPhysicsConfig)
        physics_lane = None
        phase_one_provenance = None
        if full_physics and physics_tendencies is None:
            raise ConfigurationRefusal(
                "held_physics_tendencies",
                None,
                "the full-physics lane must compute and validate phase one before RK",
                "step_device_with_physics(CudaHeldMpasPhysicsTendenciesV841)",
            )
        if not full_physics and physics_tendencies is not None:
            raise ConfigurationRefusal(
                "held_physics_tendencies",
                "provided",
                "a dry v8.4.1 config cannot be relabelled by supplying physics arrays",
                "V841MpasColumnPhysicsConfig()",
            )
        cp = self.cp
        validation_flag = cp.zeros((1,), dtype=cp.int32)
        if self.regional_v841 is not None:
            # driver.py:3085-3089 -- mpas_atm_core.F:735-781 reads lbc_in and
            # re-forms the boundary tendencies at the start of any step whose
            # clock reached an interval end, before any dynamics.
            self.regional_v841.begin_step(self.atmosphere.state)
        outer_saved = self._copy_state(self.atmosphere.state)
        time_level_one = self._copy_saved(self.atmosphere.saved)
        if self.regional_v841 is not None:
            self.regional_v841.bind_state_rho(outer_saved.rho)
        physics_contract_sha256 = None
        scalar_names: tuple[str, ...] = ()
        moist_coefficients = None
        if full_physics:
            from .cuda_physics_v841 import (
                CUDA_PHYSICS_V841_CONTRACT_SHA256,
                CudaHeldMpasPhysicsTendenciesV841,
                CudaPhaseOneExecutionProvenanceV841,
            )

            if not isinstance(
                physics_tendencies, CudaHeldMpasPhysicsTendenciesV841
            ):
                raise TypeError(
                    "full physics requires CudaHeldMpasPhysicsTendenciesV841"
                )
            if physics_tendencies.contract_sha256 != (
                CUDA_PHYSICS_V841_CONTRACT_SHA256
            ):
                raise ValueError("held physics contract digest changed")
            # The row the RUN selected, not the frozen lane's: the scalar
            # block's own width decides which declared row this is, and the
            # dynamics, the preparation and the seam all read the same one.
            run_row = species_row_for_selector(
                self.config.config_microp_scheme)
            if int(outer_saved.scalars.shape[0]) != run_row.n_species():
                raise ValueError(
                    f"{run_row.scheme_name!r} declares "
                    f"{run_row.n_species()} scalars and the state carries "
                    f"{int(outer_saved.scalars.shape[0])}"
                )
            physics_tendencies.validate(
                n_vert_levels=self.nlev,
                n_cells=self.ncells,
                n_edges=self.nedges,
                n_scalars=run_row.n_species(),
            )
            if float(physics_tendencies.time_seconds) != float(
                outer_saved.time_seconds
            ):
                raise ValueError(
                    "held physics time must equal the candidate start exactly: "
                    f"{physics_tendencies.time_seconds} != {outer_saved.time_seconds}"
                )
            phase_one_provenance = physics_tendencies.execution_provenance
            if not isinstance(
                phase_one_provenance, CudaPhaseOneExecutionProvenanceV841
            ):
                raise TypeError("held physics lost typed phase-one provenance")
            physics_lane = _v841_physics_receipt_lane(
                self.config, phase_one_provenance
            )
            physics_contract_sha256 = CUDA_PHYSICS_V841_CONTRACT_SHA256
            scalar_names = run_row.names()
            moist_coefficients = self._compute_moist_dynamics_coefficients_v841(
                outer_saved.scalars,
                scalar_names=scalar_names,
                validation_flag=validation_flag,
            )
        outer_dt = float(self.config.config_dt)
        dynamics_splits = (int(self.config.config_dynamics_split_steps)
            if self.config.config_split_dynamics_transport else 1)
        dynamics_schedule, scalar_schedule = self._rrfs_schedules(outer_dt, dynamics_splits)
        diagnostics = self.horizontal.solve_diagnostics(
            outer_saved.rho,
            outer_saved.rho_u,
            dt=outer_dt,
            apvm_upwinding=self.config.config_apvm_upwinding,
            normal_velocity=time_level_one.normal_velocity,
            rk_step=3,
        )
        # A limited-area suite run (the post-transport callback with its Registry mass-loading count) loads the air
        # with every moist species exactly as the global run does: the fork's atm_compute_moist_coefficients
        # (mpas_atm_time_integration.F at 1d6e370e, called every step from atm_srk3) has no limited-area branch.
        # The regional qv-only path below stays for the dycore's own one-species regional runs, as cuda_driver's
        # MPAS-A route does (regional_suite_loading there).  THE BREAKAGE THIS FIXES: hrrrv5 on a limited-area mesh
        # weighed qv alone in qtot/cqw/cqu, so every column's pressure gradient and buoyancy left native's at step 1
        # (lam7k 2026-10-28: rho_zz 3.8e-5 relative low, u 1.6e-4, in every interior column).
        regional_suite_loading = (
            self.regional_v841 is not None
            and physics_mass_loading_count is not None
        )
        if (
            self.regional_v841 is not None
            and not regional_suite_loading
            and moist_coefficients is None
            and bool(getattr(self.config, "config_moist_physics", False))
        ):
            # driver.py:3090-3105 -- formed once per outer step from the
            # step-start scalars and reused by every dynamics subcycle.
            moist_coefficients = self.regional_v841.moist_coefficients(
                outer_saved.scalars
            )
        if (self.regional_v841 is None or regional_suite_loading) and moist_coefficients is None:
            # Native atm_srk3 calls atm_compute_moist_coefficients on EVERY
            # step (mpas_atm_time_integration.F:2125-2139), physics suite
            # 'none' included: qtot/cqw/cqu enter dpdz, the Euler w and u
            # pressure terms, the implicit coefficients and the acoustic
            # pressure gradient.  Without them a physics-off run carrying
            # water vapour integrated the dry equations against native's
            # moist ones.  With physics off native's Registry activates only
            # moist-group species (qc..qg need a scheme package; number
            # concentrations need Thompson, which is the physics config), so
            # every carried species loads the air.  On a dry state qtot=0,
            # cqu=cqw=1 and every product/divide is exact: dry bits are kept.
            moist_coefficients = self._compute_moist_dynamics_coefficients_v841(
                outer_saved.scalars,
                scalar_names=None,
                n_mass_loading=(int(outer_saved.scalars.shape[0])
                    if physics_mass_loading_count is None
                    else int(physics_mass_loading_count)),
                validation_flag=validation_flag,
            )
        current = outer_saved
        carried = time_level_one
        split_flux_u_sum = None
        split_flux_w_sum = None
        final_modified_theta_dynamics_rate = None
        for _subcycle in range(dynamics_splits):
            if self.regional_v841 is not None:
                # The in-step offset of every driving interpolation is
                # dt_dynamics*(substep-1) + rk_timestep (F:491-551 through
                # regional_v841.dynamics_time_offset); the substep index is
                # one-based, as in driver.py's loop.
                self.regional_v841.begin_dynamics_substep(_subcycle + 1)
            final_modified_theta_dynamics_rate = None
            result = self._advance_dynamics_subcycle_v841(
                current,
                time_level_one=carried,
                initial_diagnostics=diagnostics,
                schedule=dynamics_schedule,
                outer_dt=outer_dt,
                validation_flag=validation_flag,
                physics_tendencies=physics_tendencies,
                moist_coefficients=moist_coefficients,
                physics_stage_callback=physics_stage_callback,
                dynamics_substep=_subcycle + 1,
                **({"acoustic_output_callback": acoustic_output_callback}
                   if acoustic_output_callback is not None and
                   _subcycle == dynamics_splits - 1 else {}),
            )
            if physics_stage_callback is not None:
                physics_tendencies = result.stage_physics_tendencies
            current = result.state
            carried = result.saved
            diagnostics = result.diagnostics
            final_modified_theta_dynamics_rate = (
                result.modified_theta_dynamics_rate
            )
            split_flux_u_sum = accumulate_split_flux_cuda_v841(
                result.mass_flux_u,
                split_flux_u_sum,
                kernel_cache=self.cache,
            )
            split_flux_w_sum = accumulate_split_flux_cuda_v841(
                result.mass_flux_w,
                split_flux_w_sum,
                kernel_cache=self.cache,
            )
            del result
        if split_flux_u_sum is None or split_flux_w_sum is None:
            raise RuntimeError("RRFS dynamics subcycle loop did not execute")
        split_flux_u = finish_split_flux_cuda_v841(
            split_flux_u_sum, dynamics_splits, kernel_cache=self.cache
        )
        split_flux_w = finish_split_flux_cuda_v841(
            split_flux_w_sum, dynamics_splits, kernel_cache=self.cache
        )

        del split_flux_u_sum, split_flux_w_sum
        # The split transport reads the fork's scalars_tend, which physics_get_tend fills once at the top of
        # atm_srk3 (mpas_atm_time_integration.F:1077-1084 at 1d6e370e) and atm_advance_scalars carries across
        # the three transport stages as scalar_tend_save (F:3527,3770-3810). Under the native stage callback
        # the sources exist only after the dynamics subcycles have run their first RK stage, so the buffer is
        # formed here. Formed before the subcycles it was all zeros: the PBL vapour, cloud and aerosol sources
        # never reached the transported scalars while theta_m kept their moist share, so cells with strong
        # surface evaporation (warm water, coasts) heated instead of moistening.
        rrfs_scalar_source = (cp.zeros_like(outer_saved.scalars) if physics_tendencies is None
                              else cp.array(physics_tendencies.scalars,copy=True))
        scalar_stage_timesteps = None
        if (
            self.config.config_scalar_advection
            and self.config.config_split_dynamics_transport
            and current.scalars.shape[0] > 0
        ):
            scalar_stage = cp.array(outer_saved.scalars, copy=True)
            for stage in scalar_schedule.stages:
                if self.halo_exchanger_v841 is not None:
                    # Round D: the six-species stage block becomes owner truth
                    # before the stage's edge fluxes read it at ring<=2.
                    self.halo_exchanger_v841.round_transport(scalar_stage)
                if self.regional_v841 is not None:
                    # driver.py:3186-3257: the regional split transport, with
                    # the mask-4/5 first-order edge downgrade and the
                    # specified-zone cell skip of F:4764-4861, then the
                    # per-stage boundary adjust of atm_srk3:2688-2717.
                    from .cuda_regional_forecast_v841 import (
                        advance_scalars_regional_cuda_v841,
                    )

                    # Native takes the limited routine on the last stage when
                    # EITHER switch is on (mpas_atm_time_integration.F:2953,
                    # "rk_step < 3 .or. (.not. config_monotonic .and. .not.
                    # config_positive_definite)" selects the unlimited one),
                    # as the global path does (cuda_transport_v841).  Testing
                    # config_monotonic alone left a regional run with
                    # positive_definite=True and monotonic off unlimited
                    # while its receipt recorded the switch as on.
                    if (
                        self.config.config_monotonic
                        or self.config.config_positive_definite
                    ) and stage.stage == 3:
                        transported = self._rrfs_scalar_transport(
                            self.atmosphere.mesh, self.v841_context,
                            outer_saved.scalars, scalar_stage, outer_saved.rho,
                            current.rho, split_flux_u, split_flux_w, stage.large_timestep,
                            rk_step=3, config_scalar_advection=True,
                            config_monotonic=bool(self.config.config_monotonic),
                            config_positive_definite=bool(self.config.config_positive_definite),
                            config_split_dynamics_transport=True,
                            config_time_integration_order=self.config.config_time_integration_order,
                            config_scalar_adv_order=3, config_scalar_vadv_order=3,
                            config_coef_3rd_order=self.config.config_coef_3rd_order,
                            coefficients=self.coefficients, fzm=self.atmosphere.vertical.fzm,
                            fzp=self.atmosphere.vertical.fzp, rdzw=self.atmosphere.vertical.rdzw,
                            validation_flag=validation_flag, indices_prevalidated=True,
                            rho_old_prevalidated=True, kernel_cache=self.cache,
                            config_apply_lbcs=True,
                            bdy_mask_cell=self.regional_v841.bdy_mask_cell,
                            bdy_mask_edge=self.regional_v841.bdy_mask_edge,
                            garbage_cell=int(self.regional_v841.padded.garbage_cell),
                            bind_density_pool=self.regional_v841.bind_state_rho,
                            # Round E on N ranks, as on the global path: the
                            # limiter reads both endpoint cells' scale factors
                            # at owned seam edges.  One card has no exchanger.
                            halo_exchange=self.halo_exchanger_v841,
                            **({} if physics_tendencies is None else
                               {"scalar_tendency": physics_tendencies.scalars}),
                        )
                        scalar_stage = transported.scalars
                        self.regional_v841.bdy_adjust_scalars(scalar_stage, stage.stage)
                        accumulate_finite_array_cuda_v841(
                            scalar_stage, invalid_flag=validation_flag, kernel_cache=self.cache,
                        )
                        continue
                    scalar_result = self._rrfs_scalar_transport(
                        self.atmosphere.mesh,
                        self.v841_context,
                        outer_saved.scalars,
                        scalar_stage,
                        outer_saved.rho,
                        current.rho,
                        split_flux_u,
                        split_flux_w,
                        stage.large_timestep,
                        coefficients=self.coefficients,
                        fzm=self.atmosphere.vertical.fzm,
                        fzp=self.atmosphere.vertical.fzp,
                        rdzw=self.atmosphere.vertical.rdzw,
                        rk_step=stage.stage,
                        config_time_integration_order=self.config.config_time_integration_order,
                        config_coef_3rd_order=(
                            self.config.config_coef_3rd_order
                        ),
                        validation_flag=validation_flag,config_scalar_advection=True,
                        config_monotonic=False,config_positive_definite=False,
                        config_split_dynamics_transport=True,config_apply_lbcs=True,
                        bdy_mask_cell=self.regional_v841.bdy_mask_cell,
                        bdy_mask_edge=self.regional_v841.bdy_mask_edge,
                        scalar_tendency=rrfs_scalar_source,
                    )
                    scalar_stage=scalar_result.scalars
                    self.regional_v841.bdy_adjust_scalars(
                        scalar_stage, stage.stage
                    )
                    accumulate_finite_array_cuda_v841(
                        scalar_stage,
                        invalid_flag=validation_flag,
                        kernel_cache=self.cache,
                    )
                    continue
                transported = self._rrfs_scalar_transport(
                    self.atmosphere.mesh,
                    self.v841_context,
                    outer_saved.scalars,
                    scalar_stage,
                    outer_saved.rho,
                    current.rho,
                    split_flux_u,
                    split_flux_w,
                    stage.large_timestep,
                    rk_step=stage.stage,
                    config_scalar_advection=True,
                    config_monotonic=self.config.config_monotonic,
                    config_positive_definite=self.config.config_positive_definite,
                    config_split_dynamics_transport=True,
                    config_time_integration_order=self.config.config_time_integration_order,
                    coefficients=self.coefficients,
                    fzm=self.atmosphere.vertical.fzm,
                    fzp=self.atmosphere.vertical.fzp,
                    rdzw=self.atmosphere.vertical.rdzw,
                    config_scalar_adv_order=3,
                    config_scalar_vadv_order=3,
                    config_coef_3rd_order=self.config.config_coef_3rd_order,
                    validation_flag=validation_flag,
                    indices_prevalidated=True,
                    rho_old_prevalidated=True,
                    kernel_cache=self.cache,
                    halo_exchange=self.halo_exchanger_v841,
                    scalar_tendency=rrfs_scalar_source,
                )
                scalar_stage = transported.scalars
                accumulate_finite_array_cuda_v841(
                    scalar_stage,
                    invalid_flag=validation_flag,
                    kernel_cache=self.cache,
                )
            if self.halo_exchanger_v841 is not None:
                # Final D round: the last stage's transported scalars become
                # owner truth on the halo BEFORE phase-2 (WSM6) consumes the
                # halo columns.  Without it, phase-2 writes backend-held
                # per-column products (h_diabatic, effc/effi) from garbage
                # halo scalars; the next radiation call freezes that garbage
                # into held tendencies whose ring-1 values owned cut edges
                # consume -- the measured step-12 divergence of the first
                # cross-node run.
                self.halo_exchanger_v841.round_transport(scalar_stage)
            current.scalars = scalar_stage
            scalar_stage_timesteps = tuple(
                stage.large_timestep for stage in scalar_schedule.stages
            )
        if physics_post_transport_callback is not None:
            # Fork atm_srk3:1597–1687: transported scalars and dynamics heating
            # enter the provider's rate conversion, clamp, and microphysics, then
            # return here for boundary specification.
            physics_post_transport_callback(
                state=current, saved=carried, outer_state=outer_saved,
                modified_theta_dynamics_rate=final_modified_theta_dynamics_rate,
            )
        if (self.regional_v841 is None and not full_physics
                and physics_post_transport_callback is None):
            # atm_srk3 F:2799-2800 (DO_PHYSICS builds, every suite): negative
            # mixing ratios of every species are set to zero after transport.
            # A where-assignment, not max(): -0.0 and NaN keep native's bits.
            cp.putmask(current.scalars, current.scalars < 0, 0)
        if self.regional_v841 is not None:
            # driver.py:3259-3301, in this order: the unconditional clamp
            # (atm_srk3:2798-2800), then the specified-zone resets and the
            # F:8238 perturbation write and bdy_set_scalars, all reading the
            # PRE-advance clock, then the clock advance last.
            if self.config.config_moist_physics and physics_post_transport_callback is None:
                self.regional_v841.clamp_negative_scalars(current.scalars)
            self.regional_v841.reset_speczone_values(
                theta_m=carried.theta_m,
                rho_theta=current.rho_theta,
                rho_theta_perturbation=carried.rho_theta_perturbation,
                scalars=current.scalars,
            )
            self.regional_v841.advance_clock(outer_dt)

        gf_rate_arrays = None
        if full_physics:
            if final_modified_theta_dynamics_rate is None:
                raise RuntimeError(
                    "full-physics split loop did not capture final RK dynamics heating"
                )
            rthdynten = cp.empty_like(current.rho)
            rqvdynten = cp.empty_like(current.rho)
            self._launch_physics_driver(
                "gf_dry_theta_dynamics_rate_v841_f32",
                self.ncells,
                (
                    np.int32(self.nlev),
                    np.int32(self.ncells),
                    np.float32(np.float32(461.6) / np.float32(287.0)),
                    final_modified_theta_dynamics_rate,
                    carried.theta_m,
                    current.scalars[0],
                    rthdynten,
                    rqvdynten,
                    validation_flag,
                ),
            )
            gf_rate_arrays = (rthdynten, rqvdynten)

        # THE COST THIS ENDS (SPEED-2 PROFILE item 1): reading the flag here made the host wait for the whole
        # step to drain, every step (18 ms on x1.2562, RTX 5090), so the next step's outer-physics launches could
        # not queue under this step's dycore and TEMPO tail. Inside a forecast the word is now read off the step
        # path (deferred_checks_v842): the forecast still stops with this FloatingPointError, at most one step
        # later, before any output is written, and ahead of any later error the refused state causes. Outside a
        # forecast (replays, referee helpers) it is read at once, as before.
        from .mpasphys.deferred_checks_v842 import defer
        validation_started = time.perf_counter()
        defer(validation_flag, "RRFS CUDA validation flag refused the outer step before publish",
              error=FloatingPointError)
        validation_seconds = time.perf_counter() - validation_started
        validation_transfer = TransferStats(
            int(validation_flag.nbytes), validation_seconds
        )
        current.time_seconds = outer_saved.time_seconds + outer_dt
        dynamics_tendencies = None
        if full_physics:
            assert gf_rate_arrays is not None
            dynamics_tendencies = CudaV841GfDynamicsTendencies(
                rthdynten=gf_rate_arrays[0],
                rqvdynten=gf_rate_arrays[1],
                time_seconds=current.time_seconds,
            )
            dynamics_tendencies.validate(
                cp=cp, n_vert_levels=self.nlev, n_cells=self.ncells
            )
        result_atmosphere = DeviceAtmosphere(
            self.atmosphere.mesh,
            current,
            self.atmosphere.vertical,
            self.atmosphere.reference,
            carried,
            self.atmosphere.terrain,
            self.atmosphere.h2d,
        )
        compile_manifest = self.cache.compile_manifest()
        dynamics_stage_timesteps = tuple(
            stage.large_timestep for stage in dynamics_schedule.stages
        )
        if full_physics:
            assert physics_tendencies is not None
            assert physics_contract_sha256 is not None
            assert dynamics_tendencies is not None
            held_validation = physics_tendencies.validation_d2h
            assert phase_one_provenance is not None
            gwdo_validation = phase_one_provenance.gwdo_validation_d2h
            candidate_d2h = _sum_transfer_stats(
                held_validation,
                validation_transfer,
                gwdo_validation,
            )
            assert physics_lane is not None
            receipt = CudaV841PhysicsStepReceipt(
                evidence=physics_lane["candidate_evidence"],
                configuration=dict(self.configuration),
                configuration_sha256=self.configuration_sha256,
                authority_ruler=None,
                authority_ruler_sha256=None,
                frozen_source=RRFS_CUDA_SOURCE,
                t0_diagnostics_source=self.t0_diagnostics_source,
                stage_acoustic_steps=tuple(
                    stage.acoustic_steps for stage in dynamics_schedule.stages
                ),
                start_time_seconds=outer_saved.time_seconds,
                end_time_seconds=current.time_seconds,
                h2d=self.h2d,
                d2h=candidate_d2h,
                compile_manifest=compile_manifest,
                compile_manifest_sha256=canonical_sha256(compile_manifest),
                layout_contract=dict(CUDA_LAYOUT_CONTRACT),
                source_release="v8.3.1-noaa",
                dynamics_split_steps=dynamics_splits,
                dynamics_timestep_seconds=outer_dt / dynamics_splits,
                dynamics_stage_timesteps=dynamics_stage_timesteps,
                scalar_transport_stage_timesteps=scalar_stage_timesteps,
                split_flux_reduction=SPLIT_FLUX_REDUCTION,
                authority_nonclaims=physics_lane["authority_nonclaims"],
                bulk_physics_contract_sha256=physics_contract_sha256,
                bulk_physics_tendencies_applied=True,
                physics_components=physics_lane["physics_components"],
                physics_cadences_seconds=_v841_physics_cadences(self.config),
                gwd_scheme=physics_lane["gwd_scheme"],
                gwd_evidence=(
                    None
                    if physics_lane["gwd_evidence"] is None
                    else dict(physics_lane["gwd_evidence"])
                ),
                phase_one_execution_provenance=phase_one_provenance,
                gwdo_validation_d2h=gwdo_validation,
                scalar_order=scalar_names,
                held_tendency_time_seconds=float(
                    physics_tendencies.time_seconds
                ),
                held_tendency_validation_d2h=held_validation,
                dycore_validation_d2h=validation_transfer,
                phase2_validation_d2h=None,
                final_commit_validation_d2h=None,
                moist_dynamics_coefficients_applied=True,
                moist_dynamics_scalar_order=species_row_for_selector(
                    self.config.config_microp_scheme).names(),
                moist_dynamics_negative_qv_policy=(
                    V841_MOIST_DYNAMICS_NEGATIVE_QV_POLICY
                ),
                moist_dynamics_source=V841_MOIST_DYNAMICS_SOURCE,
                moist_dynamics_source_sha256=(
                    V841_MOIST_DYNAMICS_SOURCE_SHA256
                ),
                post_wsm6_h_diabatic_reapplied=False,
                composite_committed=False,
                final_authority_claim=False,
            )
            candidate = CudaV841PhysicsStepCandidate(
                atmosphere=result_atmosphere,
                dynamics_tendencies=dynamics_tendencies,
                receipt=receipt,
            )
            self._pending_v841_physics_candidate = candidate
            return candidate

        receipt = CudaV841StepReceipt(
            evidence=RRFS_IMPLEMENTED_EVIDENCE,
            configuration=dict(self.configuration),
            configuration_sha256=self.configuration_sha256,
            authority_ruler=None,
            authority_ruler_sha256=None,
            frozen_source=RRFS_CUDA_SOURCE,
            t0_diagnostics_source=self.t0_diagnostics_source,
            stage_acoustic_steps=tuple(
                stage.acoustic_steps for stage in dynamics_schedule.stages
            ),
            start_time_seconds=outer_saved.time_seconds,
            end_time_seconds=current.time_seconds,
            h2d=self.h2d,
            d2h=validation_transfer,
            compile_manifest=compile_manifest,
            compile_manifest_sha256=canonical_sha256(compile_manifest),
            layout_contract=dict(CUDA_LAYOUT_CONTRACT),
            source_release="v8.3.1-noaa",
            dynamics_split_steps=dynamics_splits,
            dynamics_timestep_seconds=outer_dt / dynamics_splits,
            dynamics_stage_timesteps=dynamics_stage_timesteps,
            scalar_transport_stage_timesteps=scalar_stage_timesteps,
            split_flux_reduction=SPLIT_FLUX_REDUCTION,
            authority_nonclaims=RRFS_AUTHORITY_NONCLAIMS,
        )
        return CudaDeviceStepResult(result_atmosphere, receipt)
