"""Next-region class construction beside the current forecast, under its GPU lease.

Only geometry, cold input buffers and pool hydration are prepared here. Carried
state is copied into the prepared driver before its first use; physics clock,
coupling, carried pools and state-dependent physics initialization still run in
their original order. The running forecast keeps a transient-memory reserve.
"""
from dataclasses import fields, is_dataclass
from importlib import import_module
import inspect
import os
import sys
import threading
import time

import numpy as np

_GPU_BUILD = threading.Lock()


def private_pool_mapping():
    if os.name != 'posix' or os.environ.get('WOOF_HEX_MOVING_POOL_MMAP','1') == '0':
        return False
    import resource
    soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    return soft == resource.RLIM_INFINITY or soft >= 8192


def driver_spec(config, layout, dt, *, configs=None):
    from .config_v842 import NativeMpasDycoreConfigV842
    from ..cuda_driver import CudaDryDycoreDriver
    from ..cuda_backend import KernelCache
    from ..physics_suites import load_table
    values = layout.configs if configs is None else configs
    options = {f.name: values[f.name] for f in fields(NativeMpasDycoreConfigV842)
               if f.init and f.name in values}
    options.update(config_dt=dt, config_moist_physics=config.moist_physics)
    selected = NativeMpasDycoreConfigV842(**options)
    row = load_table()[getattr(config, 'table_key', 'mpasphys_v842')]
    driver, cache = CudaDryDycoreDriver, KernelCache
    if row.get('dycore') is not None:
        from ..dycore_variants import resolve_dycore_variant, dycore_driver_type, dycore_kernel_cache_type
        selected = resolve_dycore_variant(selected, row)
        extra = {f.name: values[f.name] for f in fields(type(selected))
                 if f.init and f.name in values}
        selected = resolve_dycore_variant(NativeMpasDycoreConfigV842(**options), row, overrides=extra)
        driver, cache = dycore_driver_type(selected), dycore_kernel_cache_type(selected)
    return driver, cache, selected


def _array_bytes(value, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, dict):
        return sum(_array_bytes(v, seen) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_array_bytes(v, seen) for v in value)
    if is_dataclass(value):
        return sum(_array_bytes(getattr(value, f.name), seen) for f in fields(value))
    if hasattr(value, '__dict__'):
        return _array_bytes(vars(value), seen)
    return 0


def _driver_input_bytes(inputs):
    """Only local fields a driver uploads, excluding the parent mesh backlink."""
    mesh, grid, profiles, state, reference, saved, _ = inputs
    return _array_bytes((getattr(mesh, 'arrays', None),
                         getattr(grid, 'vertical_grid', None),
                         getattr(grid, 'terrain_metrics', None),
                         profiles, state, reference, saved))


def prepare(item, inputs, config, arguments, dt):
    """Prepare this exact class. Never initialize physics on a predicted state."""
    from .route_table import framework_entry
    from .pool_v842 import NativePoolHydration
    layout = item['layout']
    # Reading the init used the non-restart package selection. Pool hydration,
    # like the actual moved runtime, uses the restart flag only after that read.
    layout.configs = dict(layout.configs, config_do_restart=True)
    # The actual class runtime snapshots its output configs before hydration.
    # Its dycore is built from those namelist words; physics resolves mutable
    # core metadata afterwards. Do not choose a different dycore config here.
    driver_configs = dict(layout.configs)
    native_type = framework_entry(config, 'backend')
    resources = getattr(arguments, 'physics_resources', None)
    options = {}
    if getattr(arguments, 'native_physics_tables', None) and 'tables' in inspect.signature(native_type).parameters:
        from pathlib import Path
        options['tables'] = [Path(p).resolve() for p in arguments.native_physics_tables]
    begin = time.perf_counter()
    prepare_layout = getattr(config, 'native_layout_prepare', None)
    if prepare_layout:
        module, name = prepare_layout.split(':', 1)
        getattr(import_module(module), name)(layout)
    native = native_type(layout=layout, resources=resources, **options)
    hydration = NativePoolHydration(native, layout, item['payload'])
    hydration.copy_on_write = private_pool_mapping()
    hydration.populate()
    item['physics_backend'] = native
    item['physics_hydration'] = hydration
    item['gpu_preparation'] = dict(pool_hydration_seconds=time.perf_counter()-begin)
    prepare_driver(item, inputs, config, arguments, dt, driver_configs=driver_configs)


def prepare_driver(item, inputs, config, arguments, dt, *, driver_configs=None):
    """Build actual admitted class geometry while its host pools are projected.

    No physics pool/clock is touched here. The actual carried state is used,
    and restore_driver refreshes its words again before the first class step.
    """
    layout = item['layout']
    item.setdefault('gpu_preparation', {})
    if driver_configs is None:
        driver_configs = dict(layout.configs)
    cp = sys.modules.get('cupy')
    if cp is None or os.environ.get('CUDA_VISIBLE_DEVICES') == '':
        return
    mesh, grid, profiles, state, reference, saved, _ = inputs
    driver_type, cache_type, selected = driver_spec(config, layout, dt, configs=driver_configs)
    if (selected.config_les_model != 'none' or selected.config_mix_scalars
            or layout.configs.get('config_apply_lbcs', False)
            or str(layout.configs.get('config_IAU_option', 'off')).strip() != 'off'):
        item['gpu_preparation']['dycore_deferred'] = 'LES, IAU or limited-area wrapper'
        return
    from ..cuda_backend import require_cuda
    from ..cuda_physics_prep_v841 import CudaMpasToPhysGeometryV841
    with _GPU_BUILD:
        # This is a bound on simultaneous allocations, not a reduction of the
        # case or its physics. A class that cannot fit is built at the move.
        limit = int(os.environ.get('WOOF_HEX_MOVING_PREPARE_BUDGET_MIB', '20000')) * 2**20
        reserve = int(os.environ.get('WOOF_HEX_MOVING_TRANSIENT_MIB', '4096')) * 2**20
        estimate = int(1.5 * _driver_input_bytes(inputs)) + (64 << 20)
        live = cp.get_default_memory_pool().used_bytes()
        from ..class_work_arena import _ARENAS
        live += sum(a.large.used_bytes() for a in _ARENAS)
        if live + estimate + reserve > limit:
            item['gpu_preparation'].update(dycore_deferred='running forecast transient reserve',
                                            live_bytes=live, estimated_bytes=estimate)
            return
        begin = time.perf_counter()
        cache = cache_type(capability=require_cuda(min_compute=(7, 5)))
        coefficients = moving_coefficients(mesh, state, selected)
        upload = {}
        if ('upload_cache' in inspect.signature(driver_type.from_host).parameters
                and os.environ.get('WOOF_HEX_MOVING_INCREMENTAL_UPLOAD', '1') != '0'):
            from ..moving_upload_cache import UploadCache
            item.setdefault('upload_cache', UploadCache())
            upload['upload_cache'] = item['upload_cache']
        driver = driver_type.from_host(mesh, state, grid.vertical_grid, reference, selected,
            saved_diagnostics=saved, terrain_metrics=grid.terrain_metrics,
            reference_wind_profiles=profiles, kernel_cache=cache,
            **({'advection_coefficients': coefficients} if coefficients is not None else {}), **upload)
        item['gpu_driver'] = driver
        item['gpu_geometry'] = CudaMpasToPhysGeometryV841.from_host(mesh)
        item['gpu_preparation']['dycore_geometry_seconds'] = time.perf_counter()-begin
        if upload:
            item['gpu_preparation']['incremental_upload'] = item['upload_cache'].receipt()
        item['gpu_preparation'].update(estimated_bytes=estimate, live_bytes_before=int(live),
            live_bytes_after=int(cp.get_default_memory_pool().used_bytes()
                                 +sum(a.large.used_bytes() for a in _ARENAS)))


def moving_coefficients(mesh, state, config):
    if (getattr(config, 'source_release', None) != 'v8.4.1'
            or os.environ.get('WOOF_HEX_MOVING_ADVECTION_GPU','1') == '0'):
        return None
    from ..moving_advection_cuda import build
    return build(mesh,config_scalar_adv_order=config.config_scalar_adv_order,
                 n_vert_levels=state.rho.shape[0])


def restore_driver(item, factory, mesh, state, vertical, reference, config, *, gpu_coefficients=False, **kwargs):
    """The original constructor's driver with fresh carried dynamic words."""
    driver = None if item is None else item.pop('gpu_driver', None)
    if item is not None:
        # The one-use prediction cache must not retain snapshots/arrays once
        # the admitted driver becomes live and can mutate its state.
        item.pop('upload_cache', None)
    if driver is None:
        if gpu_coefficients and kwargs.get('advection_coefficients') is None:
            coefficients = moving_coefficients(mesh, state, config)
            if coefficients is not None:
                kwargs['advection_coefficients'] = coefficients
        return factory(mesh, state, vertical, reference, config, **kwargs)
    if type(driver) is not getattr(factory, '__self__', None) or driver.config != config:
        raise ValueError('prepared moving class driver/config differs from its admitted runtime')
    saved = kwargs['saved_diagnostics']
    for name in ('rho', 'rho_theta', 'rho_u', 'rho_w', 'scalars'):
        getattr(driver.atmosphere.state, name).set(np.ascontiguousarray(getattr(state, name)))
    driver.atmosphere.state.time_seconds = float(state.time_seconds)
    for name in ('theta_m', 'exner', 'density_perturbation', 'rho_theta_perturbation',
                 'pressure_perturbation', 'normal_velocity', 'vertical_velocity'):
        getattr(driver.atmosphere.saved, name).set(np.ascontiguousarray(getattr(saved, name)))
    return driver
