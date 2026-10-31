"""Real native pool/session binding; mpas_atm_core.F:589-596,1036-1040.

All arithmetic remains in pristine native routines or the common CUDA glue.
The forecast owner supplies device diagnostics and the post-transport callback.
"""
from __future__ import annotations

from contextlib import contextmanager
import ctypes as ct
import os
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .driver_v842 import (NativeCalendarState, NativeDriverHooks, NativePoolArguments,
                          PhysicsDriver, PhysicsManager, SlotRegistry)
from .native_backend import NativePhysics
from .pool_v842 import NativePoolHydration, RegistryLayout
from ..cuda_physics_v841 import CudaMpasStageSourcesV842


@contextmanager
def native_working_directory(path: Path):
    """Native table paths are relative; mpas_atmphys_control.F:393-440."""
    before = Path.cwd()
    if before == Path(path).resolve():
        # Same-host rank threads share one process working directory; the
        # launcher enters it once (samehost_v842). A per-thread chdir/restore
        # would race.
        yield
        return
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(before)


class NativePhysicsSession:
    """Bind complete driver calls to hydrated pools; mpas_atmphys_init.F:83-481."""

    # MPAS native initialization: mpas_atmphys_init.F:83-481.
    def __init__(self, native: NativePhysics, layout: RegistryLayout, payload: Path,
                 config: Any, *, dt: float, start: str, stop: str,
                 stream_intervals: Mapping[str, str],
                 post_transport: Callable[[Any], Any],
                 registry: SlotRegistry | None = None,
                 cuda_device: Any | None = None,
                 cuda_resources: Path | None = None,
                 core_operations: Mapping[str, Callable] | None = None,
                 surface_record_reader: Callable | None = None, prehydrated=None,
                 pool_copy_on_write=False):
        import time
        clock = time.perf_counter()
        self.preparation_timings = {}
        def mark(name):
            nonlocal clock
            now = time.perf_counter()
            self.preparation_timings[name] = now-clock
            clock = now
        self.native, self.layout = native, layout
        self.config = config
        prepare = getattr(config, 'native_layout_prepare', None)
        if prepare is not None:
            from importlib import import_module
            module, operation = prepare.split(':', 1)
            getattr(import_module(module), operation)(layout)
        from importlib import import_module
        provider = import_module(getattr(config,'driver_module','woof.hex.mpasphys.driver_v842'))
        NativePoolArguments = provider.NativePoolArguments
        NativeCalendarState = provider.NativeCalendarState
        NativeDriverHooks = provider.NativeDriverHooks
        PhysicsDriver, PhysicsManager = provider.PhysicsDriver, provider.PhysicsManager
        SlotRegistry = provider.SlotRegistry
        self.gwdo_before = {}
        mark('driver-metadata')
        if prehydrated is None:
            self.hydration = NativePoolHydration(native, layout, payload)
            self.hydration.copy_on_write = bool(pool_copy_on_write)
            self.hydration.populate()
        else:
            if (prehydrated.native is not native or prehydrated.layout is not layout
                    or prehydrated.payload.resolve() != Path(payload).resolve()):
                raise ValueError('prepared physics pools belong to another moving class')
            self.hydration = prehydrated
            for row in layout.fields:
                record = prehydrated.records[row['pool'], row['name']]
                if any(record[key] != row[key] for key in ('kind','shape','dimensions')):
                    raise ValueError('prepared physics field extent/type differs from admitted class')
            # The carrier determines the actual restart clock after preparation.
            # Field words and dimensions are already this member's; refresh the
            # runtime config values in the original Registry order.
            for name, value in layout.configs.items():
                native.config(name, value)
            # Class input construction restores the original output configs,
            # including an unresolved length. Replay populate's final metadata
            # assignment after that reset, before physics reads runtime configs.
            if ('mesh', 'nominalMinDc') in self.hydration.records:
                nominal = self.hydration.horizontal_length()
                native.config('config_len_disp', layout.configs['config_len_disp'])
                native.put_field('mesh', 'nominalMinDc', nominal)
        mark('pool-hydration')
        native.clock(start=start, stop=stop, dt=dt,
                     calendar=layout.configs['config_calendar_type'])
        native.streams()
        self.hydration.register()
        mark('clock-and-registration')
        dims = layout.dimensions
        native.mpas_atm_set_dims(nvertlevels_val=dims['nVertLevels'],
                                maxedges_val=dims['maxEdges'],
                                maxedges2_val=dims['maxEdges2'],
                                num_scalars_val=dims['num_scalars'])
        self.sources = {name: np.zeros((dims['nVertLevels'], dims[entity]+1),
                                      dtype=np.float32, order='F')
                        for name, entity in (('tend_ru_physics','nEdges'),
                                             ('tend_rtheta_physics','nCells'),
                                             ('tend_rho_physics','nCells'))}
        self.arguments = NativePoolArguments(native,
            pools={**self.sources, **{p:p for p in layout.pools | {'configs'}}},
            its=1, ite=dims['nCellsSolve'], synchronize=lambda entry,call:None)
        calendar = NativeCalendarState(native, native.lib, self.arguments)
        external = {'physics_timetracker_state':calendar,
                    'post_transport_prepare':post_transport,
                    'snapshot_gwdo_tendencies':self.snapshot_gwdo}
        # RP6 (correct mode with config_pbl_interval): the land-surface and sea-ice slots receive the
        # precipitation added up since their last call (precip_window_v842). Exact mode never builds it.
        self.precipitation_window = None
        if getattr(config, 'correct', None) is not None and config.correct('RP6') and config.active('land_surface'):
            if cuda_device is None:
                raise RuntimeError('correct mode with config_pbl_interval (RP6) adds up the surface precipitation '
                                   'on the CUDA device; this session has none')
            from .precip_window_v842 import SurfacePrecipitationWindow
            self.precipitation_window = SurfacePrecipitationWindow(native, dims['nCellsSolve'],
                                                                   config.selectors, cuda_device)
            external.update(self.precipitation_window.operations())
        external.update(core_operations or {})
        self.hooks = NativeDriverHooks(native, self.arguments,
            after_call=self.after_call, external=external,
            save_fields=self.snapshot, restore_fields=self.restore)
        slots = registry or SlotRegistry()
        self.hooks.register_slots(slots, config)
        mark('sources-and-hooks')
        if cuda_device is not None:
            from .device_pools_v842 import NativeDevicePoolViews, register_cuda_slots
            self.device_pools=NativeDevicePoolViews(self,cuda_device)
            register_cuda_slots(slots,config,self.device_pools,cuda_resources)
        mark('cuda-slot-bindings')
        self.surface_service = None
        # The owned framework has no Fortran surface entry points. Bind the
        # existing CUDA service before resident-suite admission, including
        # monthly vegetation/albedo updates when SST input is disabled.
        # mpas_atmphys_manager.F:231-253; update_surface.F:54-568.
        if (config.service_backends.get("surface_update") == "hex-cuda"
                or (getattr(native, 'framework_only', False) and cuda_device is not None)):
            if cuda_device is None:
                raise RuntimeError("surface CUDA service requires its resident device binding")
            # The suite row's service (the RRFS fork adds dust and lai12m -> lai), route_table.
            from .route_table import service_type
            NativeSurfaceService = service_type(config, 'surface_update',
                                                'woof.hex.mpasphys.surface_update_v842:NativeSurfaceService')
            self.surface_service = NativeSurfaceService(self, cuda_device,
                read_surface_record=surface_record_reader)
            self.hooks.external.update(self.surface_service.callbacks())
        mark('surface-service')
        self.driver = PhysicsDriver(config, PhysicsManager(config, dt,
                                     streams=stream_intervals), slots, self.hooks)
        if getattr(native, 'framework_only', False):
            # The suite row's ported scheme initialization (v8.4.2 or the RRFS fork), route_table.
            from .route_table import framework_entry
            self.framework_init = framework_entry(config, 'initialization')(self)
            self.hooks.external.update(self.framework_init.operations())
        mark('driver-initialization')
        self._initialized = False

    def snapshot_gwdo(self, call) -> None:
        """Copy pre-drag winds; mpas_atmphys_driver_gwdo.F:540-547,676-683."""
        self.gwdo_before = {name:self.field('tend_physics',name).copy()
                            for name in ('rublten','rvblten')}

    def put(self, pool: str, name: str, value: Any, *, level: int = 1) -> None:
        """Copy a device/host view into native dummy storage; mpas_atmphys_interface.F:304-305."""
        device_views = getattr(self, 'device_pools', None)
        if device_views is not None and device_views.resident:
            device_views.write_field(pool, name, value, level=level)
            return
        if (device_views is not None and isinstance(value, device_views.cp.ndarray)
                and value.dtype in (np.dtype('float32'), np.dtype('int32'))):
            # Pack on the GPU before the existing synchronous host transfer.
            # Integer words preserve NaN payloads: a float layout conversion
            # can canonicalize them. Native storage and halo ownership stay
            # unchanged; no asynchronous pointer outlives this call.
            host = value.view(device_views.cp.uint32).get(order='F').view(value.dtype)
        else:
            host = value.get() if hasattr(value, 'get') else np.asarray(value)
        dtype, shape, levels = self.native._fields[pool, name]
        if host.dtype != dtype or host.ndim != len(shape):
            raise ValueError(f'native {pool}.{name} shape/dtype differs')
        if host.shape == shape:
            buffer = host
        elif shape[:-1] == host.shape[:-1] and shape[-1] == host.shape[-1]+1:
            buffer = self.native.get_field(pool, name, level=level)
            buffer[..., :-1] = host
        else:
            raise ValueError(f'native {pool}.{name} extent differs')
        self.native.put_field(pool, name, buffer, level=level, nlevels=levels)

    def field(self, pool: str, name: str, *, level: int = 1):
        """Return real entity words from the current authoritative storage."""
        views = getattr(self, 'device_pools', None)
        if views is not None and views.resident:
            old, views.level = views.level, level
            try:
                return views.field(pool, name)
            finally:
                views.level = old
        return self.native.get_field(pool, name, level=level)[..., :-1]

    def publish(self):
        """Publish resident field words only at an output/restart boundary."""
        views = getattr(self, 'device_pools', None)
        if views is not None and views.resident:
            views.publish()

    def after_call(self, entry: str, call: Any) -> Any:
        """Return held native tendency arrays; mpas_atmphys_todynamics.F:98-498."""
        if entry == 'physics_get_tend':
            return CudaMpasStageSourcesV842(
                self.sources['tend_rho_physics'][:,:-1].copy(),
                self.sources['tend_ru_physics'][:,:-1].copy(),
                self.sources['tend_rtheta_physics'][:,:-1].copy(),
                self.native.get_field('tend','scalars_tend')[...,:-1].copy())
        return None

    def initialize(self) -> None:
        """Run native tables, manager and scheme initialization; mpas_atm_core.F:589-596."""
        # MPAS's pinned non-MPAS_DEBUG framework retains SILENT pool lookups.
        # Optional landuse/tendency arrays are null when their package is off;
        # a stand-in's FATAL setting aborts calls which native permits.
        # MPAS source: src/framework/mpas_pool_routines.F:108,123-136;
        # src/framework/mpas_pool_types.inc:3-5; mpas_framework.F:107-109.
        # This is the pristine routine in the authenticated strict GNU library,
        # not a replacement for any field/driver operation. The ABI is GNU's
        # default INTEGER by-reference ABI, pinned by the reference recipe.
        if not getattr(self.native, 'framework_only', False):
            policy = getattr(self.native.lib, '__mpas_pool_routines_MOD_mpas_pool_set_error_level')
            policy.argtypes = [ct.POINTER(ct.c_int)]
            policy.restype = None
            policy(ct.byref(ct.c_int(1001)))
        from woof.hex.cycle.land_state import initialize_preserving_land
        initialize_preserving_land(self)
        self._initialized = True

    def begin_step(self, itimestep: int, time_seconds: float) -> Any:
        """Physics before dynamics; mpas_atm_core.F:1036-1040."""
        return self.driver.begin_step(itimestep=itimestep, time_seconds=time_seconds, pools=self)

    def snapshot(self) -> Mapping[str, Any]:
        """Registry pool snapshot; Registry.xml:650-925; mpas_atmphys_vars.F:24-953.

        This is pool storage only. Private scheme SAVE/derived state is still
        owned by f3's checkpoint provider; this method claims no complete restart.
        """
        self.publish()
        result = {f'{pool}.{name}.{level}':self.native.get_field(pool,name,level=level)
                  for (pool,name),(_,_,levels) in self.native._fields.items()
                  for level in range(1,levels+1)}
        # RP6 (correct mode with config_pbl_interval): native's restart precipitation window.
        if self.precipitation_window is not None:
            result.update(self.precipitation_window.snapshot())
        if self.surface_service is not None:
            result["surface_service"] = self.surface_service.restart_state()
        return result

    def restore(self, fields: Mapping[str, Any]) -> None:
        """Restore explicit native time levels; mpas_atmphys_init.F:174-396."""
        fields = dict(fields)
        state = fields.pop('surface_service', None)
        if (state is None) != (self.surface_service is None):
            raise ValueError('surface service restart state/backend differs')
        if state is not None and state.get('february') not in (28, 29):
            raise ValueError('surface service restart February differs')
        window = self.precipitation_window
        window_keys = set() if window is None else window.keys()
        expected = {f'{pool}.{name}.{level}'
                    for (pool,name),(_,_,levels) in self.native._fields.items()
                    for level in range(1,levels+1)} | window_keys
        if set(fields) != expected:
            raise ValueError('native pool snapshot field set differs')
        saved_window = {key: fields[key] for key in window_keys}
        fields = {key: value for key, value in fields.items() if key not in window_keys}
        for key,value in fields.items():
            pool,name,level=key.rsplit('.',2)
            _,shape,_=self.native._fields[pool,name]
            if value.shape != shape or value.dtype != self.native._fields[pool,name][0]:
                raise ValueError(f'native pool snapshot layout differs at {key}')
        if window is not None:
            window.restore(saved_window)    # validates its own layout before writing
        for key,value in fields.items():
            pool,name,level=key.rsplit('.',2)
            self.put(pool,name,value,level=int(level))
        if state is not None:
            self.surface_service.restore_restart_state(state)
