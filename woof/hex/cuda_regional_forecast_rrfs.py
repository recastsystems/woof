"""Complete regional RRFS constructor using shared unchanged LBC/runtime arithmetic.

RRFSx/MPAS-Model MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:1928-7759
and MPAS-Model-rrfs/src/core_atmosphere/mpas_atm_core.F:1266-1309 at 1d6e370e.
"""
from __future__ import annotations
from dataclasses import replace
from datetime import datetime
from typing import Any, Sequence
import numpy as np
from .cuda_regional_forecast_v841 import (derive_regional_masks, PaddedRegionalHostMesh,
    build_regional_device_atmosphere, regional_padded_context_v841,
    regional_padded_advection_coefficients, CudaRegionalRuntimeV841, _pad_axis)

# MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:1928-7759 at 1d6e370e.
def assemble_regional_driver_rrfs(
    mesh: object,
    state: object,
    vertical: object,
    reference: object,
    saved: object,
    terrain: object,
    config: object,
    *,
    reference_wind_profiles: object,
    lbc_paths: Sequence[str],
    start_time: datetime,
    kernel_cache: Any = None,
    measure_garbage: bool = False,
    scalar_names: Sequence[str] = ("lbc_qv",),
    theta_initial: Any | None = None,
    config_relax_zone_divdamp_coef: float = 6.0,
    driver_type: Any = None,
) -> Any:
    """Fork time integration 1928–7759/core 1266–1309 at 1d6e370e.

    ``driver_type`` is the fork's driver class or a subclass of it (the MPAS suite route's IAU stage-density
    wrapper, runtime_v842.with_stage_density); None constructs the fork's own class.
    """

    from .cuda_backend import KernelCache, require_cuda
    from .cuda_driver_rrfs import CudaRRFSDycoreDriver
    from .cuda_transport import CudaAdvectionCoefficients
    from .damping_rrfs import build_rrfs_vertical_velocity_damping
    from .deformation_rrfs import select_deformation_weights_rrfs,initialize_edge_trigonometry_rrfs
    from .cuda_compile_rrfs import RRFSKernelCache
    from .offcentering_v841 import build_v841_acoustic_offcentering
    from .transport import build_advection_coefficients

    dtype = np.dtype(np.float32)
    masks = derive_regional_masks(mesh, dtype)
    require_cuda(min_compute=(12, 0))
    cache = RRFSKernelCache() if kernel_cache is None else kernel_cache
    if not isinstance(cache, RRFSKernelCache):
        cache = RRFSKernelCache(capability=cache.capability,
                                cache_dir=cache.cache_directory)
    padded = PaddedRegionalHostMesh(mesh, masks)

    n_vert_levels = int(np.asarray(getattr(state, "rho")).shape[0])
    offcentering = build_v841_acoustic_offcentering(
        np.asarray(vertical.rdzw),
        minimum=config.config_epssm_minimum,
        maximum=config.config_epssm_maximum,
        transition_bottom_z=config.config_epssm_transition_bottom_z,
        transition_top_z=config.config_epssm_transition_top_z,
    )
    # Core always rebuilds dss from config_xnutr/config_zd; it is built on the
    # UNPADDED zgrid and padded, because the garbage column of zgrid is the
    # pool zero and a damping profile of it means nothing.
    selected_dss = build_rrfs_vertical_velocity_damping(
        np.asarray(vertical.zgrid), mesh.mesh_density,
        xnutr=config.config_xnutr,
        damping_start_z=config.config_zd,
    )

    class _VerticalWithDss:
        # MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:1928-7759 at 1d6e370e.
        def __getattr__(self, name: str) -> Any:
            if name == "dss":
                return selected_dss
            return getattr(vertical, name)

    device_atmosphere = build_regional_device_atmosphere(
        padded, state, _VerticalWithDss(), reference, saved, terrain
    )
    context = regional_padded_context_v841(
        mesh, offcentering, reference_wind_profiles, n_vert_levels=n_vert_levels
    )
    host_coefficients = build_advection_coefficients(
        mesh,
        config_scalar_adv_order=config.config_scalar_adv_order,
        n_vert_levels=n_vert_levels,
        source_order_v841=True,
        allow_regional_sentinels=True,
    )
    device_coefficients = CudaAdvectionCoefficients.from_host(
        regional_padded_advection_coefficients(
            host_coefficients, padded.garbage_cell
        )
    )
    driver_class = CudaRRFSDycoreDriver if driver_type is None else driver_type
    if not (isinstance(driver_class, type) and issubclass(driver_class, CudaRRFSDycoreDriver)):
        # THE BREAKAGE THIS PREVENTS: another dycore class under the fork's regional runtime would run MPAS-A's
        # integration with the fork's boundary words and say nothing.
        raise TypeError(f"the RRFS regional driver must be the fork's driver class or a subclass, not {driver_class}")
    driver = driver_class(
        device_atmosphere,
        config,
        device_coefficients,
        kernel_cache=cache,
        v841_context=context,
    )
    weights = select_deformation_weights_rrfs(mesh,
        on_a_sphere=str(mesh.attrs.get('on_a_sphere','NO')).strip().upper()=='YES',
        sphere_radius=float(mesh.attrs.get('sphere_radius',1.0)))
    driver.atmosphere.mesh=replace(driver.atmosphere.mesh,
        defc_a=driver.cp.asarray(_pad_axis(weights['defc_a'],0,0.0)),
        defc_b=driver.cp.asarray(_pad_axis(weights['defc_b'],0,0.0)))
    driver.horizontal.mesh = driver.atmosphere.mesh
    driver._rrfs_edge_cos,driver._rrfs_edge_sin=(driver.cp.asarray(_pad_axis(x,0,0.0)) for x in initialize_edge_trigonometry_rrfs(mesh.angleEdge))
    theta_initial=getattr(mesh,'t_init',None) if theta_initial is None else theta_initial
    if theta_initial is None:
        if not config.config_mix_full and config.config_v_theta_eddy_visc2>0:
            raise ValueError('native perturbation theta mixing reads t_init; supply its initialized pool words')
        driver._rrfs_theta_init=driver.cp.zeros_like(driver.atmosphere.state.rho)
    else:
        driver._rrfs_theta_init=driver.cp.asarray(_pad_axis(theta_initial,1,0.0))
    padded.inactive_deformation_placeholders=[]
    runtime = CudaRegionalRuntimeV841(
        mesh,
        padded,
        lbc_paths=lbc_paths,
        start_time=start_time,
        config_h_scale_with_mesh=bool(config.config_h_ScaleWithMesh),
        config_apply_lbcs=bool(config.config_apply_lbcs),
        outer_dt=float(config.config_dt),
        dynamics_split=int(config.config_dynamics_split_steps),
        time_integration_order=int(config.config_time_integration_order),
        n_vert_levels=n_vert_levels,
        kernel_cache=cache,
        device_atmosphere=device_atmosphere,
        v841_context=context,
        measure_garbage=measure_garbage,
        scalar_names=tuple(scalar_names),
        # MPAS-Model-rrfs Registry.xml:268 (default 6.0); mpas_atm_time_integration.F:7598.
        config_relax_zone_divdamp_coef=float(config_relax_zone_divdamp_coef),
    )
    driver.regional_v841 = runtime
    return driver
