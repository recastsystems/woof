"""Native forecast orchestration; mpas_atm_core.F:589-701,1036-1040.

CUDA owns dynamics and common conversion kernels. Rust owns stream output;
Python schedules immutable snapshots and the ported clock's alarms.
"""
from __future__ import annotations

from dataclasses import fields
from contextlib import ExitStack
from datetime import datetime
from fractions import Fraction
import hashlib
import inspect
import json
import os
import shutil
from pathlib import Path
from typing import Any

from ..stage_timing import stage as timed_stage
from ..stochastic_timing import measure_stochastic_step
from .driver_v842 import interval_seconds
from .native_backend import NativePhysics
from .pool_v842 import NativePoolHydration
from .session_v842 import NativePhysicsSession, native_working_directory


def commit_outer_scalar_groups(session, atmosphere, cp):
    """Carry direct outer-physics updates into the next dynamics step.

    RRFS mpas_atm_core.F:1045-1049 calls physics_driver before atm_timestep
    on the same state. Smoke and MYNN update the chemistry slice directly
    (driver_smoke.F:879-907, driver_pbl.F:1007-1016), not through tendencies.
    The port's separate pool and dycore allocations must preserve that alias.
    Groups are route metadata; empty groups and routes without them do no work.
    """
    from .route_table import route_row
    groups = route_row(session.config).get('outer_scalar_groups', ())
    ranges = [(int(session.layout.indices.get(('state', group + '_start'), 0)),
               int(session.layout.indices.get(('state', group + '_end'), -1)))
              for group in groups]
    ranges = [(first - 1, last) for first, last in ranges if 0 < first <= last]
    if not ranges:
        return
    source = session.field('state', 'scalars', level=1)
    target = atmosphere.state.scalars
    for first, last in ranges:
        # Owned columns only: leave appended species and regional/parallel
        # halo columns under their existing owners.
        target[first:last, :, :source.shape[-1]] = cp.asarray(source[first:last])


class ResidentNativeCoupling:
    """Hold outer sources and hand transport to native; mpas_atm_time_integration.F:2140-2816."""

    # MPAS outer-step coupling ownership: mpas_atm_time_integration.F:2140-2816.
    def __init__(self, session, dycore, geometry, names):
        self.session, self.dycore = session, dycore
        self.geometry, self.names = geometry, tuple(names)
        self.cp = dycore.cp
        from .compile_v842 import MpasPhysicsKernelCache
        cache_type=MpasPhysicsKernelCache
        module=getattr(session.config,'driver_module',None)
        if module is not None:
            from importlib import import_module
            cache_type=import_module(module).StrictPhysicsKernelCache
        self.physics_cache = cache_type(
            capability=dycore.cache.capability,
            cache_dir=Path(dycore.cache.cache_directory)/'mpasphys')
        self.sources = None
        self.transport = None
        # A limited-area run's dycore carries native's garbage column (nCells+1, nEdges+1); the physics pools hold the
        # entity extent.  None on a closed mesh, where every array below keeps its global path.
        self.regional = getattr(dycore, 'regional_v841', None)
        # IAU (SuiteIau): None without config_IAU_option; itimestep is the run's step (MPAS itimestep).
        self.iau = None
        self.itimestep = 0
        # A same-host rank (samehost_v842) widens the owned-column sources to
        # its local cells and edges before stochastic forcing and IAU (which
        # size themselves from the local state), applies IAU to owned columns
        # (solve_extent: nCellsSolve, nEdgesSolve) and then fills the halos
        # from the owners, so halo words carry the owners' complete sources.
        self.widen_sources = None
        self.exchange_sources = None
        self.solve_extent = None

    def synchronize(self, atmosphere, *, level=1, prepared_columns=False):
        """Copy native carriers/diagnostics; mpas_atmphys_interface.F:284-305."""
        from ..cuda_physics_prep_v841 import prepare_mpas_to_phys_cuda_v841
        chemistry = getattr(self, 'chemistry', None)
        if chemistry is not None:
            # The MPAS physics reads the Registry's scalar rows; the appended species rows stay in the dycore.
            atmosphere = chemistry.registry_view(atmosphere)
        s, d, native = atmosphere.state, atmosphere.saved, self.session
        for name, value in (('rho_zz',s.rho),('theta_m',d.theta_m),
                            ('u',d.normal_velocity),('w',d.vertical_velocity),
                            ('scalars',s.scalars)):
            native.put('state',name,value,level=level)
        for name, value in (('rho_p',d.density_perturbation),
                            ('rtheta_p',d.rho_theta_perturbation),
                            ('pressure_p',d.pressure_perturbation),('exner',d.exner)):
            native.put('diag',name,value)
        views = getattr(native, 'device_pools', None)
        resident_winds = views is not None and views.resident and not prepared_columns
        columns = self._resident_winds(atmosphere) if resident_winds else None
        if columns is None:
            resident_winds = False
            columns=prepare_mpas_to_phys_cuda_v841(atmosphere,scalar_names=self.names,
                                                  geometry=self.geometry,kernel_cache=self.physics_cache,
                                                  registry_scalar_names=next(row['components'] for row in native.layout.fields
                                                    if row['pool']=='state' and row['name']=='scalars'))
        from .diagnostics_v842 import output_diagnostics
        theta,rho,pressure=output_diagnostics(atmosphere,self.names,self.physics_cache,self.cp)
        if not resident_winds:
            for name,value in (('uReconstructZonal',columns.u_p),('uReconstructMeridional',columns.v_p)):
                native.put('diag',name,value)
        for name,value in (('rho',rho),('theta',theta),('pressure',pressure)):
            native.put('diag',name,value)
        if getattr(self.dycore,'les_dissipation',None) is not None:
            # Native reconstructs at the outer boundary (time_integration.F:2735),
            # then holds these cell winds throughout the next dynamics splits.
            self.dycore.les_dissipation.winds=columns
        return columns

    def _resident_winds(self, atmosphere):
        """Reconstruct directly into resident pools with the shared CUDA kernel.

        ResidentPhysicsGlue owns the other MPAS_to_physics carriers. Running
        aggregate preparation here duplicated them, uploaded a species pointer
        table and downloaded a validation flag on every synchronization. Keep
        the flag device-side as in resident preparation, and retain scratch
        through the forecast. The validated standalone preparation is unchanged.
        """
        from types import SimpleNamespace
        import numpy as np
        from ..cuda_physics_prep_v841 import _launch

        shape = tuple(atmosphere.state.rho.shape)
        geometry = self.geometry
        if (shape[1] != geometry.n_cells
                or atmosphere.saved.normal_velocity.shape != (shape[0], geometry.n_edges)):
            raise ValueError('resident wind geometry/state extent differs')
        if getattr(self, '_wind_fallback', False):
            return None
        garbage = geometry.regional_garbage_cell
        if not hasattr(self, '_wind_storage'):
            geometry.validate()
            pools = SimpleNamespace(
                u_p=self.session.field('diag', 'uReconstructZonal'),
                v_p=self.session.field('diag', 'uReconstructMeridional'))
            # A limited-area run's dycore carries native's garbage column as its last element (nCells+1); the pools
            # hold the entity extent.  The kernel then writes retained dycore-extent columns and the entity columns
            # are copied into the pools on the device, as the aggregate path's write_field did.
            # THE BREAKAGE THIS PREVENTS: this path refused every cull ("resident wind reconstruction requires a
            # closed global mesh", then "resident wind pool shape/dtype/layout differs") at the first output, so no
            # MPAS-suite region ran once fix/hex-mpasphys-host met fix/hex-mpas-regional (3 km GFS 2026-10-23 18Z
            # f348 region at 33.5N 101W, box L, 2026-10-24).
            if garbage is not None and int(garbage) != shape[1] - 1:
                raise ValueError(f'the regional garbage column is the last element of the padded extent; '
                                 f'got {garbage} with {shape[1]} columns')
            pool_shape = shape if garbage is None else (shape[0], shape[1] - 1)
            # A same-host rank's pools hold its owned columns, which come first in its local order (the prefix
            # NativeDevicePoolViews.write_field takes from a local-extent source).  The kernel writes the rank's local
            # extent into retained columns and the owned prefix is copied into the pools on the device: the words the
            # aggregate path's write_field put there.
            # THE BREAKAGE THIS PREVENTS (rrfs/c3-8card-speed 1960569, NOAA conus3km on 8 x RTX PRO 6000, 2026-10-29):
            # every same-host rank took the aggregate preparation at both synchronizations of every step (outer
            # physics and post-transport) and read its validation word on the host (int(cp.asnumpy(invalid)[0])), so
            # each rank's host waited for its card to drain twice a step (outer physics 178 ms per step on rank 0,
            # 10 ms on one card), and the aggregate kernels ran over every local column twice a step for nothing.
            samehost = bool(getattr(getattr(self.session, 'layout', None), 'samehost_partition', False))
            prefix = False
            for value in (pools.u_p, pools.v_p):
                owned = (samehost and value.ndim == 2 and value.shape[0] == shape[0]
                         and 0 < value.shape[1] < pool_shape[1])
                if ((value.shape != pool_shape and not owned) or value.dtype != np.dtype('float32')
                        or not value.flags.c_contiguous):
                    # Pool rows that are neither the local extent nor an owned prefix: the winds take the
                    # aggregate preparation and are put into the pool as before resident winds existed.
                    # THE BREAKAGE THIS PREVENTS: every MPAS-suite run on 2+ cards stopped at its first
                    # synchronization with "resident wind pool shape/dtype/layout differs".
                    self._wind_fallback = True
                    return None
                prefix = prefix or owned
            if garbage is None and not prefix:
                outputs, entity_pools = pools, None
            else:
                outputs = SimpleNamespace(u_p=self.cp.empty(shape, dtype=self.cp.float32),
                                          v_p=self.cp.empty(shape, dtype=self.cp.float32))
                entity_pools = pools
            self._wind_storage = (shape, outputs,
                tuple(self.cp.empty(shape, dtype=self.cp.float32) for _ in range(3)),
                self.cp.zeros(1, dtype=self.cp.int32), entity_pools)
        held_shape, outputs, scratch, invalid, entity_pools = self._wind_storage
        if held_shape != shape:
            raise ValueError('resident wind extent changed during forecast')
        _launch(self.physics_cache, 'reconstruct_winds_v841_f32', shape[1], (
            np.int32(shape[0]), np.int32(shape[1]), np.int32(geometry.n_edges),
            np.int32(geometry.max_edges), atmosphere.saved.normal_velocity,
            geometry._edges_on_cell, geometry._n_edges_on_cell,
            geometry._coeffs_reconstruct, geometry._lat_cell, geometry._lon_cell,
            geometry._sealed_edges_on_cell, geometry._sealed_n_edges_on_cell,
            geometry._sealed_coeffs_reconstruct, geometry._sealed_lat_cell,
            geometry._sealed_lon_cell, *scratch, outputs.u_p, outputs.v_p, invalid))
        if entity_pools is not None:
            # The garbage column has nEdgesOnCell = 0, so the kernel leaves it at u = v = 0.  It is lent the donor
            # column's wind, exactly as prepare_mpas_to_phys_cuda_v841 does (YSU refused a zero-shear sounding
            # there, r4.75.11020), so the returned columns and the pools hold the aggregate path's words.
            garbage = None if garbage is None else int(garbage)
            donor = (None if garbage is None else garbage - 1 if geometry.regional_garbage_donor_cell is None
                     else int(geometry.regional_garbage_donor_cell))
            for value, pool in ((outputs.u_p, entity_pools.u_p), (outputs.v_p, entity_pools.v_p)):
                if garbage is not None:
                    value[..., garbage] = value[..., donor]
                pool[...] = value[..., :pool.shape[-1]]
        # The aggregate preparation this replaces refused a sealed-geometry change by reading this
        # word every synchronization. The same refusal now reads it off the step path.
        from .deferred_checks_v842 import defer
        defer(invalid, 'MPAS_to_physics CUDA wind reconstruction failed geometry validation: '
                       'the mesh geometry differs from its sealed copy (invalid word {value})')
        return outputs

    def stage(self, *, rk_step, dynamics_substep, state, saved, diagnostics):
        """Assemble outer sources; mpas_atm_time_integration.F:2140-2176; mpas_atm_time_integration.F:3088-3185."""
        if self.sources is None:
            from ..cuda_physics_v841 import CudaMpasStageSourcesV842
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
                self.session.field('tend','rt_diabatic_tend')))
            scalar_sources=self.cp.ascontiguousarray(self.cp.asarray(sources.scalars))
            if getattr(self, 'chemistry', None) is not None:
                scalar_sources=self.chemistry.pad_sources(self.cp,scalar_sources)
            self.sources=CudaMpasStageSourcesV842(*(self.cp.ascontiguousarray(self.cp.asarray(value))
                              for value in (sources.rho,sources.rho_u,sources.rho_theta)),scalar_sources,
                              diabatic_theta=diabatic)
            if self.widen_sources is not None:
                # Same-host rank: physics_get_tend filled owned columns only;
                # stochastic forcing and IAU read the local-extent state.
                self.sources=self.widen_sources(self.sources,state)
            if getattr(self, 'regional', None) is not None:
                self.sources=self.pad_regional_sources(self.sources, state=state, saved=saved)
            if getattr(self, 'stochastic', None) is not None:
                self.stochastic.apply(self.sources, state=state,
                    rho_edge=diagnostics.h_edge, normal_velocity=saved.normal_velocity,
                    time_seconds=float(self.dycore.atmosphere.state.time_seconds),
                    dt=self.stochastic_dt, names=self.names)
            if getattr(self, 'iau', None) is not None:
                # mpas_atm_time_integration.F:2220-2230: once per step, after physics_get_tend and before
                # DYNAMICS_SUBSTEPS, onto the held physics sources.
                self.sources=self.iau.apply(itimestep=self.itimestep,sources=self.sources,state=state,
                                            saved=saved,diagnostics=diagnostics,cache=self.dycore.cache,
                                            solve=self.solve_extent)
            if self.exchange_sources is not None:
                self.sources=self.exchange_sources(self.sources)
            # The driver's held copy of physics_get_tend's result is dead from here (the RRFS driver's
            # release_held_tendencies says why); drivers without the hook keep their behaviour.
            release=getattr(self.session.driver,'release_held_tendencies',None)
            if release is not None:
                release()
            # Physics is done for this step's sources: its prepared work arrays go back to the pool for the
            # dycore (NativeDevicePoolViews.release_prepared says why).
            release_prepared=getattr(getattr(self.session,'device_pools',None),'release_prepared',None)
            if release_prepared is not None:
                release_prepared()
        return self.sources

    def _registry_rows(self, scalars):
        chemistry = getattr(self, 'chemistry', None)
        return scalars if chemistry is None else scalars[:chemistry.n_registry]

    def pad_regional_sources(self, sources, *, state, saved):
        """Entity-extent physics sources onto the regional dycore's padded stride.

        Native's tendency at the garbage element is zero (mpas_atmphys_todynamics.F allocates nCells+1 and writes
        1..nCellsSolve; cuda_physics_v841 zeroes it beside the WOOF route's projection for the same reason).  The
        boundary zones then replace the specified-zone tendencies with the lateral-boundary ones.
        """
        from ..cuda_physics_v841 import CudaMpasStageSourcesV842
        cp = self.cp

        def pad(value, like):
            if value is None or value.shape == like.shape:
                return value
            if value.shape[:-1] != like.shape[:-1] or value.shape[-1] + 1 != like.shape[-1]:
                raise ValueError(f'limited-area physics source extent {value.shape} does not fit the regional '
                                 f'dycore extent {like.shape}')
            padded = cp.zeros(like.shape, dtype=value.dtype)
            padded[..., :value.shape[-1]] = value
            return padded

        return CudaMpasStageSourcesV842(pad(sources.rho, state.rho), pad(sources.rho_u, saved.normal_velocity),
                                        pad(sources.rho_theta, state.rho), pad(sources.scalars, state.scalars),
                                        diabatic_theta=pad(sources.diabatic_theta, state.rho))

    def release_dynamics_sources(self):
        """Keep only the scalar source once the last RK stage has run.

        mem/state-levels.  Past the dynamics subcycles the dycore reads the
        scalar source alone (transport's scalar_tendency) and so does
        prepare_transport; the mass, momentum, theta and lagged-heating
        sources are dead until the next step rebuilds them.  Holding them
        through transport kept 1.3 KB per column on the ordinary-step peak.
        """
        if self.sources is not None:
            from ..cuda_physics_v841 import CudaMpasStageSourcesV842
            self.sources = CudaMpasStageSourcesV842(
                rho=None, rho_u=None, rho_theta=None,
                scalars=self.sources.scalars, diabatic_theta=None)
        return self.sources

    def post_transport(self, *, state, saved, outer_state, modified_theta_dynamics_rate):
        """Enter before any scalar clamp; mpas_atm_time_integration.F:2770-2816."""
        self.transport=(state,saved,outer_state,modified_theta_dynamics_rate)
        # This diagnostic is formed for every DO_PHYSICS build, including
        # suite none (mpas_atm_time_integration.F:6931-6941; Registry.xml:3535).
        if modified_theta_dynamics_rate is not None:
            self.session.put('tend_physics','rthdynten',modified_theta_dynamics_rate)
        self.session.driver.finish_step(pools=self.session)
        native=self.session.native
        if not self.session.config.active('microphysics'):
            self.transport=None
            return
        # MPAS microphysics_to_MPAS supplies its own Exner and perturbation
        # words. Do not recompute those from the product on the host.
        registry_rows = (state.scalars if getattr(self, 'chemistry', None) is None
                         else state.scalars[:self.chemistry.n_registry])
        for target,pool,name in ((registry_rows,'state','scalars'),
                                 (saved.theta_m,'state','theta_m'),
                                 (saved.exner,'diag','exner'),
                                 (saved.rho_theta_perturbation,'diag','rtheta_p'),
                                 (saved.pressure_perturbation,'diag','pressure_p')):
            value = self.cp.asarray(self.session.field(pool,name,level=2 if pool=='state' else 1))
            # Owned columns; a same-host rank's halo columns take their owners'
            # words at the step-boundary exchange (round A).
            target[..., :value.shape[-1]] = value
        # Native keeps rtheta_p after the rounded product/subtraction, then
        # recovers full rtheta by adding the base (mpas_atmphys_interface.F:923-926).
        state.rho_theta[...] = saved.rho_theta_perturbation
        self.dycore._add(state.rho_theta,self.dycore.atmosphere.reference.rho_theta_base)
        self.transport=None

    def boundary_forcing_columns(self):
        """Boundary-zone columns whose resolved-advection forcing GF and New Tiedtke must not read (LB1), or None.

        On a limited-area run the dycore relaxes rings 1-5 toward the lateral-boundary series and resets rings 6-7
        to it every step (atm_srk3:2688-2717, atm_bdy_set_scalars), inside the transport step that rthdynten and
        rqvdynten difference.  The convective closure then takes the boundary's nudging, and the specified zone's
        per-step reset to an undisturbed environment, as moisture convergence: on 2024-05-06 18Z the scale-aware
        GF rained up to 19 mm in the outermost ring of a 937.5 m region in one hour (0.8 mm in the interior) and
        about 5 mm a ring across a 30 km region's whole boundary zone (0.1 mm inside): a ring of rain on every
        QPF map (fix-group-1 #3).  Correct mode (the door default) zeroes that forcing in the boundary zone;
        exact mode keeps native MPAS-A v8.4.2's behaviour.  ``WOOF_HEX_MPAS_LB1=0`` keeps native behaviour in
        correct mode too.
        """
        regional = getattr(self, 'regional', None)
        if regional is None or os.environ.get('WOOF_HEX_MPAS_LB1', '1') == '0':
            return None
        if getattr(self.session.config, 'mode', 'exact') != 'correct':
            return None
        cached = getattr(self, '_boundary_columns', None)
        if cached is None:
            cached = self._boundary_columns = self.cp.asarray(regional.bdy_mask_cell) > 0
        return cached

    def solve_cells(self):
        """Columns physics owns: the dycore's on a closed mesh, the entity extent (no garbage column) on a region."""
        if getattr(self, 'regional', None) is not None:
            return int(self.session.layout.dimensions['nCellsSolve'])
        return self.dycore.ncells

    def prepare_transport(self, call):
        """Prepare GF/NTiedtke heating before microphysics; mpas_atm_time_integration.F:2770-2802."""
        from ..cuda_backend import DeviceAtmosphere
        from ..cuda_physics_v841 import (prepare_mpas_post_transport_v842,
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
                dt=self.dycore.config.config_dt,n_cells_solve=self.solve_cells(),
                kernel_cache=self.physics_cache)
            self.session.put('state','scalars',self._registry_rows(before),level=1)
        prepare_mpas_post_transport_v842(before=self._registry_rows(before),after=self._registry_rows(state.scalars),
            theta_m=saved.theta_m,scalar_names=self.names,rqvdynten=rqv,rthdynten=rth,
            convection_scheme=convection,monotonic=self.dycore.config.config_monotonic,
            dt=self.dycore.config.config_dt,n_cells_solve=self.solve_cells(),
            kernel_cache=self.physics_cache)
        boundary = self.boundary_forcing_columns()
        if boundary is not None:
            # LB1 (see boundary_forcing_columns): the convective closure reads no boundary-zone relaxation as
            # resolved advection.  ``rate`` is the dycore's own array, so the theta forcing is copied first.
            if rate is not None:
                rth = rth.copy()
            rth[:, boundary] = 0
            rqv[:, boundary] = 0
        atmosphere=self.dycore.atmosphere
        candidate=DeviceAtmosphere(atmosphere.mesh,state,atmosphere.vertical,
                    atmosphere.reference,saved,atmosphere.terrain,atmosphere.h2d)
        resident_transport = getattr(self, '_rrfs_resident_prepare_transport', None)
        if resident_transport is None:
            self.synchronize(candidate,level=2)
        else:
            resident_transport(self, candidate)
        # mpas_atm_time_integration.F:2768-2802 updates this field only for
        # GF/NTiedtke. Original Tiedtke allocates it but leaves it untouched.
        if (convection in ('cu_grell_freitas','cu_ntiedtke')
                and ('tend_physics','rqvdynten') in self.session.native._fields):
            self.session.put('tend_physics','rqvdynten',rqv)
        self.session.put('tend_physics','rthdynten',rth)


class SuiteIau:
    """MPAS-A v8.4.2 incremental analysis update on the explicit suite route.

    The iau stream's increments (tend_iau: u_amb, rho_amb, theta_amb, scalars_amb), read at initialization
    (mpas_atm_core.F:222-233) by the pinned pool reader, are held on the device, and atm_add_tend_anal_incr
    (woof.hex.iau_v842: mpas_atm_iau.F:152-292 as CUDA kernels) adds them to the physics sources once per step,
    after physics_get_tend and before the dynamics substeps (mpas_atm_time_integration.F:2220-2230).  The
    density source then enters tend_rho at rk_step 1 (with_stage_density).  Exact mode is MPAS word for word
    (NINT step count, IN1 and IN2 as native); correct mode applies IN1-IN3 (overlap weights, a window position).
    """

    def __init__(self, *, layout, payload, config, mesh, dycore, dt):
        import numpy as np
        from ..iau_v842 import IAUConfig, iau_window_position

        configs = layout.configs
        mode = getattr(config, 'mode', 'exact')
        self.config = IAUConfig(str(configs.get('config_IAU_option', 'off')).strip(),
                                float(configs.get('config_IAU_window_length_s', 21600.0)), mode)
        self.dt = float(dt)
        cp = dycore.cp
        # The four tend_iau words the pinned pool reader decoded (the same payload the hydration checked).
        manifest = json.loads((Path(payload) / 'manifest.json').read_text())
        records = {(r['pool'], r['name']): r for r in manifest['fields']}
        names = {'u': 'u_amb', 'rho': 'rho_amb', 'theta': 'theta_amb', 'scalars': 'scalars_amb'}
        read = [name for name in names.values()
                if (record := records.get(('tend_iau', name))) is not None and record.get('origins')]
        if 'u_amb' not in read or 'rho_amb' not in read or 'theta_amb' not in read:
            # THE BREAKAGE THIS PREVENTS: with no increment file read, the pool holds the Registry's zero
            # defaults, and the run would apply a zero increment while its receipt says IAU was on.
            raise ValueError('config_IAU_option is on but the iau stream read no increments (u_amb, rho_amb, '
                             'theta_amb); bind the increment file with --stream-input iau=FILE or --iau-increment')

        regional = getattr(dycore, 'regional_v841', None) is not None
        self.solve_extent = (int(layout.dimensions['nCellsSolve']), int(layout.dimensions['nEdgesSolve'])) if regional else None

        def held(name):
            record = records[('tend_iau', name)]
            raw = (Path(payload) / record['payload']).read_bytes()
            if hashlib.sha256(raw).hexdigest() != record['sha256']:
                raise ValueError(f'native payload digest differs at tend_iau.{name}')
            words = np.frombuffer(raw, dtype='<f4').reshape(record['shape'], order='F')[..., :-1]
            if regional:
                # The pool's last allocation word is not an analysis column.
                padded = np.zeros(words.shape[:-1] + (words.shape[-1] + 1,), np.float32)
                padded[..., :-1] = words
                words = padded
            return cp.ascontiguousarray(cp.asarray(np.ascontiguousarray(words, dtype=np.float32)))

        self.increments = {key: held(name) for key, name in names.items()}
        self.sources_read = sorted(read)
        n_cells = int(layout.dimensions['nCells'])
        cells_on_edge = np.asarray(mesh.arrays['cellsOnEdge'], dtype=np.int64)
        if cells_on_edge.shape[0] == 2 and cells_on_edge.shape[1] != 2:
            cells_on_edge = cells_on_edge.T
        cells_on_edge = np.where((cells_on_edge >= 0) & (cells_on_edge < n_cells), cells_on_edge, -1)
        if regional:
            cells_on_edge = np.concatenate((cells_on_edge, np.full((1, 2), -1, dtype=np.int64)))
        self.cells_on_edge = cp.asarray(np.ascontiguousarray(cells_on_edge, dtype=np.int32))
        self.zz = dycore.atmosphere.vertical.zz
        self.moist = {key: int(layout.indices['state', key]) for key in ('moist_start', 'moist_end', 'index_qv')}
        self.position = np.array(0.0, dtype=np.float32)
        self.rule = 'exact mode: MPAS-A v8.4.2 begins the window at the start of every run'
        if self.config.correct:
            start = iau_window_position(self.config, restart=bool(configs.get('config_do_restart', False)),
                                        stored=None, da_cycling=bool(configs.get('config_do_DAcycling', False)))
            self.position, self.rule = start.position, start.rule
        self.calls = self.active_calls = 0
        self.weight_seconds = 0.0
        from .iau_witness import IauWitness
        self.witness = IauWitness(cp)

    def apply(self, *, itimestep, sources, state, saved, diagnostics, cache, solve=None):
        import numpy as np
        from ..iau_v842 import atm_add_tend_anal_incr, atm_iau_coef, atm_iau_coef_window

        weight = (atm_iau_coef_window(self.config, float(self.position), self.dt) if self.config.correct
                  else atm_iau_coef(self.config, int(itimestep), self.dt))
        nl, nc = (int(n) for n in state.rho.shape)
        ne = int(diagnostics.h_edge.shape[1])
        # One card: every column is solved. A same-host rank solves its owned
        # columns (mpas_atm_iau.F loops 1..nCellsSolve/nEdgesSolve); the halo
        # words come from their owners at the source exchange that follows.
        solve = self.solve_extent if solve is None else solve
        ncs, nes = (nc, ne) if solve is None else (int(solve[0]), int(solve[1]))
        structs = {
            'mesh': {'nVertLevels': nl, 'nCells': nc, 'nEdges': ne, 'nCellsSolve': ncs, 'nEdgesSolve': nes,
                     'zz': self.zz, 'cellsOnEdge': self.cells_on_edge},
            'state': {'rho_zz': state.rho, 'theta_m': saved.theta_m, 'scalars': state.scalars,
                      'u': saved.normal_velocity, **self.moist},
            'diag': {'rho_edge': diagnostics.h_edge, 'iau_window_elapsed': self.position},
            'tend': {'scalars_tend': sources.scalars},
            'tend_iau': self.increments,
        }
        witness = (self.witness.before(sources, state, saved, diagnostics, self.increments, self.zz,
            self.cells_on_edge, self.moist, (ncs, nes), weight, self.dt, self.config.correct)
            if weight > np.float32(1.0e-12) else None)
        atm_add_tend_anal_incr(self.config, structs, int(itimestep), self.dt, sources.rho_u, sources.rho_theta,
                               sources.rho, cache=cache)
        if witness is not None:
            self.witness.after(witness, sources)
        self.calls += 1
        if weight > np.float32(1.0e-12):
            self.active_calls += 1
            self.weight_seconds += float(weight) * self.dt
        return sources

    def receipt(self):
        return {'admission_witness': self.witness.receipt(),
                'route': 'mpas-suite (atm_add_tend_anal_incr, woof.hex.iau_v842 on CUDA)', 'mode': self.config.config_physics_mode,
                'corrections': ['IN1', 'IN2', 'IN3'] if self.config.correct else [],
                'option': self.config.config_IAU_option, 'window_seconds': float(self.config.config_IAU_window_length_s),
                'window_start': self.rule, 'increments_read': self.sources_read, 'calls': self.calls,
                'active_calls': self.active_calls, 'fraction_admitted': self.weight_seconds,
                **({'window_position_seconds': float(self.position)} if self.config.correct else {})}


def with_stage_density(base):
    """The suite route's dycore class with MPAS's density physics source in tend_rho (an IAU run only).

    MPAS forms tend_rho at rk_step 1 with tend_rho_physics (mpas_atm_time_integration.F:6602), which only
    atm_add_tend_anal_incr writes.  The dycore asks the suite route's stage callback for its sources inside
    the RK loop, after tend_rho is formed, so the first dynamics subcycle of each step would drop the density
    increment (later subcycles already receive the held sources).  This subclass asks the callback first and
    hands its held sources to the subcycle; the callback holds them, so its calls in the loop return the same
    object.  Selected only when config_IAU_option is on, so every other run constructs the unchanged class.
    Named difference (exact mode): native's rthdynten reads that tend_rho too (F:7077); the dycore's rate
    keeps the dynamics-only density tendency, so with GF or New Tiedtke the convective forcing differs from
    native by theta_m*tend_rho_physics/rho_zz during the window.
    """

    class StageDensityDycore(base):
        def _advance_dynamics_subcycle_v841(self, state, **kwargs):
            callback = kwargs.get('physics_stage_callback')
            if kwargs.get('physics_tendencies') is None and callback is not None:
                if kwargs.get('initial_diagnostics') is None:
                    # The lean diagnostics lifetime (cuda_driver._step_device_v841) hands each subcycle
                    # initial_diagnostics=None and lets it form the substep-start solve itself.  The callback
                    # needs that set now (rho_edge = h_edge), so form it here with the subcycle's own call
                    # and pass it on: the subcycle then uses the same words instead of solving again.
                    # THE BREAKAGE THIS FIXES: every IAU run on the suite route stopped in its first step
                    # ("'NoneType' object has no attribute 'h_edge'", HEX-PHYSICS-PARITY run group 6 #23).
                    saved = kwargs['time_level_one']
                    kwargs['initial_diagnostics'] = self.horizontal.solve_diagnostics(
                        state.rho, state.rho_u, dt=kwargs['outer_dt'],
                        apvm_upwinding=self.config.config_apvm_upwinding,
                        normal_velocity=saved.normal_velocity, rk_step=3)
                kwargs['physics_tendencies'] = callback(
                    rk_step=kwargs['schedule'].stages[0].stage, dynamics_substep=kwargs.get('dynamics_substep', 1),
                    state=state, saved=kwargs['time_level_one'], diagnostics=kwargs['initial_diagnostics'])
            return super()._advance_dynamics_subcycle_v841(state, **kwargs)

    StageDensityDycore.__name__ = StageDensityDycore.__qualname__ = base.__name__ + 'StageDensity'
    return StageDensityDycore


def expand_input_filename(template, start):
    """Stage a start-time file with native tokens; mpas_timekeeping.F:2225-2324."""
    time = datetime.strptime(start,'%Y-%m-%d_%H:%M:%S')
    second = time.second
    values = dict(Y=f'{time.year:04d}',M=f'{time.month:02d}',D=f'{time.day:02d}',
                  d=f'{time.timetuple().tm_yday:03d}',h=f'{time.hour:02d}',m=f'{time.minute:02d}',B='all')
    result = ''
    expanding = False
    for character in template:
        if character == '$':
            expanding = True
        elif expanding:
            if character == 'S':
                second += 60*time.minute+3600*time.hour
                result += f'{second:05d}'
            elif character == 's':
                result += f'{second:02d}'
            elif character in values:
                result += values[character]
            else:
                raise ValueError(f'invalid native filename token ${character}')
            expanding = False
        else:
            result += character
    return result


def native_stream_filename(stream, out, start):
    """Read source files, stage writable copies; mpas_stream_manager.F:3187-3190,3695-3697."""
    source = getattr(stream,'input_file',None)
    if source and 'output' not in stream.directions:
        return str(Path(source).resolve())
    root = Path(out).resolve()
    template = root/stream.filename
    if not template.resolve().is_relative_to(root):
        raise ValueError(f'native writable stream {stream.name} escapes its output directory')
    if source:
        target = Path(expand_input_filename(str(template),start))
        if not target.resolve().is_relative_to(root):
            raise ValueError(f'native writable stream {stream.name} resolves outside its output directory')
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            with target.open('rb') as actual, Path(source).open('rb') as expected:
                if hashlib.file_digest(actual,'sha256').digest()!=hashlib.file_digest(expected,'sha256').digest():
                    raise ValueError(f'native staged stream input already differs: {target}')
        else:
            shutil.copyfile(source,target)
    return str(template)


def _configure_streams(native, streams, out, layout):
    """Bind requested native streams and Registry names; mpas_atm_core.F:675-701."""
    active={r['name']:r for r in layout.fields}
    components={c:r for r in layout.fields for c in r['components']}
    supported_diag={'rho','theta','pressure','pressure_base','pressure_p','exner','exner_base',
                    'rho_base','rho_p','rtheta_base','rtheta_p',
                    'uReconstructZonal','uReconstructMeridional',
                    # The native diag pool holds MPAS's own surface_pressure at every write: read from the init,
                    # re-diagnosed by the native MPAS_to_physics glue (mpas_atmphys_interface.F:426-438) and by
                    # the microphysics slot on the GPU, which commits it back (thompson_v842 4778-4781,
                    # interface.F:948-962).  THE BREAKAGE ITS ABSENCE CAUSED: the suite route's output carried
                    # no surface_pressure, so soundings and meteograms had no PSFC (hexmx proof, 2026-10-16).
                    'surface_pressure'}
    # Lagged heating already belongs to the native tendency pool and is
    # restart state (Registry.xml:704,2041; time_integration.F:4436-4437).
    supported_tend={'rt_diabatic_tend'}
    missing={}
    for stream in streams:
        kwargs=dict(filename=native_stream_filename(stream,out,layout.configs['config_start_time']),direction=';'.join(stream.directions),
                    input_interval=stream.input_interval,output_interval=stream.output_interval)
        if 'input_interval' not in inspect.signature(native.stream).parameters:
            raise RuntimeError('native stream ABI lacks independent input/output intervals: restart and surface cannot be reproduced')
        filename_interval=getattr(stream,'filename_interval','none')
        if 'filename_interval' in inspect.signature(native.stream).parameters:
            kwargs['filename_interval']=filename_interval
        elif filename_interval!='none':
            raise RuntimeError(f'native stream {stream.name} lacks filename_interval ABI: expected {filename_interval}, default none would combine native output files')
        native.stream(stream.name,**kwargs)
        missing[stream.name]=[]
        for name in stream.fields:
            row=active.get(name,components.get(name))
            if row is None or (row['pool']=='diag' and name not in supported_diag) or (row['pool']=='tend' and name not in supported_tend):
                missing[stream.name].append(name)
            else:
                native.stream_field(stream.name,name)
    return missing



def initialize_native_output(native, streams, *, restarting=False, reset_stream_alarms=True):
    """Initial time and stream alarms; mpas_atm_core.F:264-277,667-672."""
    native.initialize_state_time()
    if not reset_stream_alarms:
        return
    for stream in streams:
        if ('output' in stream.directions and stream.output_interval != 'none'
                and (stream.name == 'restart' or restarting
                     and stream.name in ('output','diagnostics'))):
            native.reset_alarm(stream.name+':out')


def restore_state_time(native, *, pool='state'):
    """A restarted segment's CF ``Time`` is its valid time since ``initial_time``, as a straight run's is.

    ``Time`` (seconds since ``initial_time``, the history's CF time axis) advances by dt every step from the
    init's value (``update_state_time``).  THE BREAKAGE THIS FIXES: a restart began it again from the init's 0,
    so every history frame after a checkpoint restart named a valid time one restart offset too early (a 13:00
    frame of a 12:00 run restarted at 12:30 said Time = 1800 "seconds since 2023-03-31 12:00:00"; HEX-PHYSICS-
    PARITY run group 6 #21).  The straight run's float32 sum of whole-second steps is exact, so this word is too.
    """
    import numpy as np
    if getattr(native, 'clock_state', None) is None:
        return None     # the native referee library keeps MPAS's own restart stream and clock
    initial = datetime.strptime(str(native.get_text_field(pool, 'initial_time', level=1)).strip(),
                                '%Y-%m-%d_%H:%M:%S')
    start = datetime.strptime(native.clock_state.start_time.stamp(), '%Y-%m-%d_%H:%M:%S')
    seconds = np.float32((start - initial).total_seconds())
    for level in (1, 2):
        native.put_field(pool, 'Time', np.array(seconds, dtype=np.float32), level=level, nlevels=2)
    return float(seconds)


class StepRecordWriter:
    """Receipt rewrites and the physics call log, written off the step path (G6).

    THE BREAKAGE THIS PREVENTS: the step loop serialised the whole growing
    receipt (``json.dumps(indent=2)``, three stage rows a step, so quadratic in
    the run) and opened and appended ``physics-calls.jsonl`` on the one host
    thread that issues the card's work, every step.  The WOOF route hands
    its receipts to a background writer.  Here the call log lines are
    serialised on the step thread (the driver reuses its call list) and
    appended in order by one writer thread; the receipt is serialised at most
    every ``WOOF_HEX_RECEIPT_EVERY_SECONDS`` (30 s; 0 = every step) and at
    step 1, stops and the end, and written by the same thread with an atomic
    rename.  The call log's bytes are unchanged; ``close`` drains both before
    the run returns or fails, so the final files are what the old loop wrote.
    """

    def __init__(self, *, every_seconds=None, clock=None):
        import queue
        import threading
        import time
        every = os.environ.get('WOOF_HEX_RECEIPT_EVERY_SECONDS') if every_seconds is None else every_seconds
        self.every_seconds = 30.0 if every in (None, '') else float(every)
        self._clock = clock or time.monotonic
        self._last = None
        self._queue = queue.Queue()
        self._error = None
        self._handles = {}
        self.receipt_writes = 0
        self.receipt_skips = 0
        self.lines = 0
        self._thread = threading.Thread(target=self._run, name='hex-step-records', daemon=True)
        self._thread.start()

    def _run(self):
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                kind, path, text = item
                if self._error is not None:
                    continue
                if kind == 'append':
                    handle = self._handles.get(path)
                    if handle is None:
                        handle = self._handles[path] = Path(path).open('a')
                    handle.write(text)
                elif kind == 'flush':
                    for handle in self._handles.values():
                        handle.flush()
                else:
                    temporary = Path(str(path) + '.partial')
                    temporary.write_text(text)
                    os.replace(temporary, path)
            except BaseException as error:  # reported on the step thread at the next call
                self._error = error
            finally:
                self._queue.task_done()

    def _raise(self):
        if self._error is not None:
            error, self._error = self._error, None
            raise RuntimeError(f'step record writer failed: {type(error).__name__}: {error}') from error

    def append(self, path, line):
        self._raise()
        self.lines += 1
        self._queue.put(('append', str(path), line))

    def receipt(self, path, receipt, *, force=False):
        """Serialise ``receipt`` now if forced or due; the file write happens on the writer thread."""
        self._raise()
        now = self._clock()
        if not force and self._last is not None and now - self._last < self.every_seconds:
            self.receipt_skips += 1
            return False
        self._last = now
        self.receipt_writes += 1
        self._queue.put(('receipt', str(path), json.dumps(receipt, indent=2, default=str) + '\n'))
        return True

    def drain(self):
        self._queue.put(('flush', None, None))
        self._queue.join()
        self._raise()

    def close(self):
        if self._thread.is_alive():
            self._queue.put(('flush', None, None))
            self._queue.put(None)
            self._thread.join()
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()
        self._raise()

    def summary(self):
        return dict(writer='background thread (hex-step-records)', every_seconds=self.every_seconds,
                    receipt_writes=self.receipt_writes, receipt_writes_skipped=self.receipt_skips,
                    call_log_lines=self.lines)


def moving_publication_due(moving, *, step, steps, dt):
    """Whether a moving route consumes the host pool views after ``step``.

    ``MovingRoute.after_step`` reads host pools only when it writes a tracker frame and may seal a
    carrier (a frame step with a controller), and ``finish`` seals at the final or stopped step, which
    is always a frame or the last step.  The shock series reads only device columns.
    THE BREAKAGE THIS PREVENTS (G6): publishing every step downloaded every resident pool buffer
    (3,460 MB and 307 syncs a step on the storm-following 3 km region, 2.5 s of a 4.2 s step) for a
    controller that reads them once a frame.
    """
    if moving is None:
        return False
    if step == 0 or step >= steps:
        return True
    checkpoint_due = getattr(moving, 'checkpoint_due', None)
    if checkpoint_due is not None and checkpoint_due(step, dt):
        # A checkpoint seals the host pool views at this step ("native state already published").
        # THE BREAKAGE THIS PREVENTS: with no download at a mid-run checkpoint step the carrier sealed stale
        # pools, and a restart from it differed from the straight run in 35 M of 94 M history values.
        return True
    control = getattr(moving, 'control', None)
    frame = float(getattr(moving, 'frame_seconds', 0.0) or 0.0)
    if control is None or frame <= 0.0:
        return False
    model_seconds = float(getattr(moving, 'offset_seconds', 0.0) or 0.0) + step * dt
    return not model_seconds % frame


def moving_columns_due(moving, *, step, steps, dt):
    """Whether the moving hooks read the prepared columns (psfc_p) after ``step``."""
    if moving is None:
        return False
    return bool(getattr(moving, 'shock', False)) or moving_publication_due(moving, step=step, steps=steps, dt=dt)


def native_publication_required(native, streams, *, step, steps, moving, lazy, output=None, when=None):
    """Keep state resident between writes; outer physics refreshes its own pools.

    Initial/final boundaries always consume current pools. ``moving`` is passed
    only at a step where the moving route consumes them (``moving_publication_due``
    or ``moving_columns_due``); pass None otherwise.
    Alarm inspection does not reset or transfer a stream; the writer still owns
    those actions and diagnostic accumulator resets.
    """
    if not lazy or step == 0 or step == steps or moving is not None:
        return True
    if output is not None:
        return output.required(when)
    return any('output' in stream.directions and stream.output_interval != 'none'
               and native.alarm_ringing(stream.name+':out') for stream in streams)


def physics_pool_readers_due(output, *, step, steps, dt, moving, execution):
    """Whether anything reads the physics pools at the end of ``step``, decided BEFORE the step runs.

    The readers of the publication after the step (native_publication_required and the execution hook): a due
    output stream, the run's last step, a moving publication, or an execution route that reads the pools
    itself (cycling backgrounds; LTS classes, taken as reading every step). A rank execution without cycling
    adds no reader. Alarm inspection only: nothing is reset. A scheme may skip a diagnostic whose only
    readers are these and its own scheduled consumers (tempo_v303.radii_due: the effective radii).
    """
    from datetime import timedelta
    from .mpi_v842 import RankExecution
    if step == steps or moving_columns_due(moving, step=step, steps=steps, dt=dt):
        return True
    if execution is not None and not (isinstance(execution, RankExecution) and execution.cycling is None):
        return True
    # The committed clock still holds the start of the step; the end is one dt on.
    return bool(output.required(output.current_time(step=step-1, dt=dt) + timedelta(seconds=float(dt))))


def write_native_output(native, streams, *, step, receipt, session=None, time_seconds=None,
                        output=None, dt=None):
    """Native stream alarm decisions; mpas_atm_core.F:675-705; mpas_timekeeping.F:925-985."""
    if output is not None:
        diagnostics_written = output.write(step=step, dt=dt, receipt=receipt)
    else:
        # Explicit native-cpu referee helpers retain native stream I/O. The
        # forecast door always supplies its Rust output owner below.
        diagnostics_written=False
        for stream in streams:
            if ('output' in stream.directions and stream.output_interval != 'none'
                    and native.alarm_ringing(stream.name+':out')):
                native.transfer_stream(stream.name,read=False,level=1)
                native.reset_alarm(stream.name+':out')
                receipt['streams'].append(dict(name=stream.name,step=step))
                diagnostics_written |= stream.name == 'diagnostics'
    if step and diagnostics_written:
        reset=getattr(getattr(session,'driver',None),'after_output',None)
        if reset is None:
            reset_native_output_diagnostics(native, session=session)
        else:
            first=len(session.driver.calls)
            reset(time_seconds=time_seconds,diagnostics_alarm=True,itimestep=step,pools=session)
            # The step call log is captured before output; retain successful
            # output-boundary resets separately for native alarm comparison.
            receipt.setdefault('physics_output_calls',[]).extend(session.driver.calls[first:])


def reset_native_output_diagnostics(native, *, session=None):
    """Clear the accumulated reflectivity after all writes; mpas_atm_core.F:877-890,968-990."""
    import numpy as np
    # Native's optional associated lookup leaves inactive microphysics alone.
    key=('diag_physics','refl10cm_1km_max')
    metadata=native._fields.get(key)
    if metadata is not None:
        views = getattr(session, 'device_pools', None)
        if views is not None and getattr(views, 'resident', False):
            views.field(*key).fill(0)
            return
        dtype,shape,levels=metadata
        native.put_field(*key,np.zeros(shape,dtype=dtype,order='F'),nlevels=levels)


def native_mesh_state_exclusions(layout):
    """State is loaded separately; retain every Registry mesh spelling.

    MPAS Registry.xml separates the mesh and state pools. Scalar arrays can
    appear under their component names in the init stream. Do not read those
    large payloads into Mesh only to read them again as atmospheric state.
    """
    names = {'state': set(), 'mesh': set()}
    for row in layout.fields:
        if row['pool'] in names:
            selected = names[row['pool']]
            selected.add(row['name'])
            selected.add(row['pool_name'])
            selected.update(row.get('components') or ())
    return frozenset(names['state'] - names['mesh'])


def keeps_initialized_deformation(row):
    """Whether the dycore keeps the init file's defc_a/defc_b words (the suite row's dycore entry decides).

    Every input loader call (one card, same-host ranks, launch, LTS, features) takes this from here. THE BREAKAGE
    THIS PREVENTS: the same-host, launch, LTS and feature paths passed ``row.get('dycore') is not None``, so the
    rrfs-44c0849 row (dycore keep_initialized_deformation false: 44c0849's Registry has no mesh defc_a/defc_b)
    asked the pool for words it does not hold; NOAA conus3km hrrrv5 on 8 cards stopped in the input loader with
    KeyError ('mesh', 'defc_a') (V8X6000, rrfs/one-tree, 2026-10-29). The one-card path read the flag already.
    """
    entry = row.get('dycore')
    return entry is not None and bool(entry.get('keep_initialized_deformation', True))


def load_native_dynamics_inputs(arguments, layout, payload, *, keep_initialized_deformation=False):
    """Complete input ownership before CUDA; mpas_atm_core.F:116-200,589-596."""
    if layout.configs.get('config_apply_lbcs', False):
        # A fork dycore (RRFS) keeps the init's deformation weights on the cut's own extent (nCells, maxEdges);
        # its regional constructor (cuda_regional_forecast_rrfs) pads them with the garbage column itself.
        return load_native_regional_inputs(arguments, layout, payload,
                                           keep_initialized_deformation=keep_initialized_deformation)
    import numpy as np
    from ..mesh import Mesh
    from ..vector import initialize_reconstruction_coefficients
    from ..driver import load_mpas_vertical_grid,load_mpas_initial_state
    from ..dynamics_v841 import load_v841_reference_wind_profiles
    init=Path(arguments.init)
    hydration=NativePoolHydration(None,layout,payload)
    # The native input stream overlays the grid with initialized mesh fields,
    # including fEdge/fVertex; Registry.xml:498-545; mpas_atm_core.F:116-151.
    # MPAS bootstrapping appends ancestor file IDs with newlines. Input
    # lineage is independent of whether this dycore retains deformation.
    mesh_options={'native_parent_chain':True}
    mesh_options['exclude_variables']=native_mesh_state_exclusions(layout)
    mesh=Mesh.from_netcdf(getattr(arguments,'grid',None) or init,init,**mesh_options)
    # Native MPAS (single precision) reads a double-precision coordinate into RKIND on input
    # (mpas_io_streams.F:925-1023). An init_atmosphere init stores them single already, so this is a
    # no-op there (x1.2562: no double array remains after the init overlay). THE BREAKAGE IT PREVENTS:
    # a hex-door init (every moving member) keeps the generator's double coordinates; the edge normals
    # derived from them were double and the native mesh pool refused the run before its first step
    # ("native mesh.edgeNormalVectors shape/dtype differs").
    for name,value in list(mesh.arrays.items()):
        if getattr(value,'dtype',None) is not None and value.dtype==np.float64:
            mesh.arrays[name]=np.ascontiguousarray(value,dtype=np.float32)
    mesh.arrays['nominalMinDc']=hydration.horizontal_length()
    mesh.arrays['coeffs_reconstruct']=initialize_reconstruction_coefficients(mesh,include_halos=True)
    # Native initializes these runtime vectors before tend_toEdges reads them;
    # mpas_vector_operations.F:652-771; mpas_atmphys_todynamics.F:510-550.
    from ..vector import initialize_vector_geometry
    vectors=initialize_vector_geometry(mesh)
    mesh.arrays['edgeNormalVectors']=vectors.edge_normal_vectors
    mesh.arrays['localVerticalUnitVectors']=vectors.local_vertical_unit_vectors
    mesh.arrays['cellTangentPlane']=vectors.cell_tangent_plane
    grid=load_mpas_vertical_grid(init,mesh,config_coef_3rd_order=layout.configs['config_coef_3rd_order'])
    profiles=load_v841_reference_wind_profiles(init,n_vert_levels=grid.vertical_grid.n_vert_levels)
    names=next(r['components'] for r in layout.fields if r['pool']=='state' and r['name']=='scalars')
    state,reference,saved=load_mpas_initial_state(init,mesh,grid.vertical_grid,
            scalar_names=(),reference_path=init,terrain_metrics=grid.terrain_metrics,
            return_saved_diagnostics=True,cold_start=True)
    # Native allocates every selected carrier, including absent input species;
    # Registry.xml:1701-1754; mpas_atmphys_init_microphysics.F:98-226. Rust already
    # applied their Registry defaults while decoding the complete scalar buffer.
    # The same bytes as np.ascontiguousarray, copied in blocks on host threads (woof.hex.host_parallel): the Fortran-
    # ordered scalars of a 30 km init are ~1.4 GB, a strided copy of ~6 s on the main thread before (2026-10-27).
    from ..host_parallel import contiguous_copy
    scalars=hydration.field_buffer('state','scalars')[...,:-1]
    state.scalars=contiguous_copy(scalars,dtype=scalars.dtype)
    for name in ('defc_a','defc_b'):
        if keep_initialized_deformation:
            # MPAS-Model-rrfs/src/core_atmosphere/mpas_atm_core.F:373-614:
            # retain native initialized deformation weights, including cut borders.
            mesh.arrays[name]=np.ascontiguousarray(hydration.field_buffer('mesh',name)[...,:mesh.n_cells].T)
        else:
            mesh.arrays[name]=np.zeros((mesh.n_cells,mesh.arrays['edgesOnCell'].shape[1]),np.float32)
    return mesh,grid,profiles,state,reference,saved,names


#: Payload-free regional inputs read beside the pool transport (start_regional_inputs), by id(layout):
#: (layout, input key, result or None, error or None). Taken once by load_native_regional_inputs.
_EARLY_REGIONAL: dict[int, tuple] = {}


def _regional_input_key(arguments, layout):
    init = Path(arguments.init)
    return (str(Path(getattr(arguments, 'grid', None) or init)), str(init),
            native_mesh_state_exclusions(layout), layout.configs['config_coef_3rd_order'])


def _regional_source_inputs(arguments, layout):
    """The regional loader's steps that read only the grid and init files, in the loader's own order.

    Mesh, Rust geometry with halos, vertical grid, reference wind profiles and initial state: none of them reads the
    pool payload, and none reads ``nominalMinDc`` (only the dycore and mixing do, after loading).  Returns them with
    the mesh keys present where the loader sets ``nominalMinDc`` from the pool, so the caller puts it back at the
    same place in ``mesh.arrays``.
    """
    import numpy as np
    from ..mesh import Mesh
    from ..driver import load_mpas_vertical_grid, load_mpas_initial_state
    from ..dynamics_v841 import load_v841_reference_wind_profiles
    from ..rust_geometry import _run
    init = Path(arguments.init)
    mesh = Mesh.from_netcdf(getattr(arguments, 'grid', None) or init, init,
                            native_parent_chain=True,
                            exclude_variables=native_mesh_state_exclusions(layout))
    for name, value in list(mesh.arrays.items()):
        if getattr(value, 'dtype', None) is not None and value.dtype == np.float64:
            mesh.arrays[name] = np.ascontiguousarray(value, dtype=np.float32)
    keys_at_nominal = tuple(mesh.arrays)
    values = _run(mesh, mode=0, dtype=np.dtype(np.float32), include_halos=True)
    for name, value in zip(('edgeNormalVectors', 'localVerticalUnitVectors', 'cellTangentPlane',
                            'coeffs_reconstruct'), values):
        mesh.arrays[name] = value
    grid = load_mpas_vertical_grid(init, mesh, config_coef_3rd_order=layout.configs['config_coef_3rd_order'],
                                   allow_regional_sentinels=True)
    profiles = load_v841_reference_wind_profiles(init, n_vert_levels=grid.vertical_grid.n_vert_levels)
    state, reference, saved = load_mpas_initial_state(init, mesh, grid.vertical_grid,
            scalar_names=(), reference_path=init, terrain_metrics=grid.terrain_metrics,
            return_saved_diagnostics=True, cold_start=True, allow_regional_sentinels=True)
    return mesh, grid, profiles, state, reference, saved, keys_at_nominal


def _set_at(arrays, name, value, keys_before):
    """``arrays[name] = value`` as if it had been set when ``arrays`` held exactly ``keys_before``."""
    if name in arrays or tuple(arrays) == tuple(keys_before):
        arrays[name] = value
        return
    known = set(keys_before)
    later = [(key, arrays.pop(key)) for key in [key for key in arrays if key not in known]]
    arrays[name] = value
    for key, item in later:
        arrays[key] = item


def start_regional_inputs(arguments, layout):
    """Read a limited-area run's payload-free inputs on a side thread while the pool transport runs.

    THE COST THIS REMOVES (NOAA RRFS v2 conus3km, 8 x RTX PRO 6000, V8X6000, 2026-10-29): the same-host start ran
    the pool transport (34-36 s, bound by its two 7.2 GB scalar fields: 8, 16 or 32 reader processes all took 34 s)
    and then, one after the other, the mesh read (8.3 s), geometry (3.1 s), vertical grid (10.4 s) and initial state
    (16.9 s) on the main thread, 39 s more, while all 8 cards sat at 0 %.  Those four read only the grid and init
    files, so they now run while the reader processes run, and the loader takes their result.  The main thread only
    waits on the reader subprocesses meanwhile, so no other netCDF call runs beside them in this process.

    Returns the callable decode_registry_pools calls when the transport ends: it waits for the side thread and keeps
    the result (or the error, raised where the loader would have raised it) for load_native_regional_inputs.
    """
    import threading
    key = _regional_input_key(arguments, layout)
    outcome = {}

    def work():
        try:
            outcome['result'] = _regional_source_inputs(arguments, layout)
        except BaseException as error:   # raised by the loader, where the serial order raised it
            outcome['error'] = error

    thread = threading.Thread(target=work, name='hex-regional-source-inputs', daemon=True)
    thread.start()

    def finish():
        thread.join()
        _EARLY_REGIONAL[id(layout)] = (layout, key, outcome.get('result'), outcome.get('error'))
    return finish


def discard_regional_inputs(layout):
    """Drop inputs read beside the transport that no loader took (a route that returned before loading)."""
    _EARLY_REGIONAL.pop(id(layout), None)


def _take_regional_inputs(arguments, layout):
    entry = _EARLY_REGIONAL.pop(id(layout), None)
    if entry is None or entry[0] is not layout or entry[1] != _regional_input_key(arguments, layout):
        return None
    if entry[3] is not None:
        raise entry[3]
    return entry[2]


def load_native_regional_inputs(arguments, layout, payload, *, keep_initialized_deformation=False):
    """A limited-area cull's inputs, as the regional dycore takes them; mpas_atm_core.F:116-200,589-596.

    The cull stores 0 (read as a negative sentinel) in one cellsOnEdge slot of every ring-7 boundary edge; native MPAS
    maps it to the garbage cell nCells+1 at read time.  The vertical grid and the initial state are loaded with those
    sentinels admitted, and the runtime vectors come from the Rust geometry with halos (the regional native binding's
    own order, cuda_regional_forecast_v842.load_native_regional_inputs), because the closed-sphere reconstruction
    reads every cellsOnEdge slot as a cell.  The regional dycore (open_regional_forecast_v841) pads every array with
    the garbage column itself.
    """
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    from ..mesh import Mesh
    from ..driver import load_mpas_vertical_grid, load_mpas_initial_state
    from ..dynamics_v841 import load_v841_reference_wind_profiles
    from ..host_parallel import contiguous_copy
    from ..rust_geometry import _run
    init = Path(arguments.init)

    # THE COST THIS REMOVES (mv3km 3 km limited area, 206,726 cells, V6000, 2026-10-28): this loader ran
    # 10-13 s on one host thread while the card waited (about 80-100 s at e3km/conus3km).  The Rust pool
    # payload work does not touch the mesh or the netCDF reads, so it runs on a side thread beside them:
    # the payload digests (hashlib releases the GIL) beside the mesh read, then the scalar words copied in
    # blocks on host threads (woof.hex.host_parallel, as the closed-mesh loader above does; a strided copy of
    # the 59 scalars on the main thread before) and the deformation weights beside the geometry, vertical
    # grid and state.  Same words and the same checks: a payload refusal is still raised before anything
    # is built from the payload.
    def pool_words(hydration):
        scalars = hydration.field_buffer('state', 'scalars')[..., :-1]
        scalars = contiguous_copy(scalars, dtype=scalars.dtype)
        deformation = ({name: hydration.field_buffer('mesh', name) for name in ('defc_a', 'defc_b')}
                       if keep_initialized_deformation else None)
        return scalars, deformation

    early = _take_regional_inputs(arguments, layout)
    if early is not None:
        # The payload-free part was read beside the pool transport (start_regional_inputs); only the pool words remain.
        mesh, grid, profiles, state, reference, saved, keys_at_nominal = early
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix='hex-pool-words') as side:
            hydration = NativePoolHydration(None, layout, payload)
            words_future = side.submit(pool_words, hydration)
            _set_at(mesh.arrays, 'nominalMinDc', hydration.horizontal_length(), keys_at_nominal)
            names = next(r['components'] for r in layout.fields if r['pool'] == 'state' and r['name'] == 'scalars')
            scalars, deformation = words_future.result()
        return _finish_regional_inputs(mesh, grid, profiles, state, reference, saved, names, scalars, deformation,
                                       keep_initialized_deformation)

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='hex-pool-words') as side:
        hydration_future = side.submit(NativePoolHydration, None, layout, payload)
        mesh = Mesh.from_netcdf(getattr(arguments, 'grid', None) or init, init,
                                native_parent_chain=True,
                                exclude_variables=native_mesh_state_exclusions(layout))
        for name, value in list(mesh.arrays.items()):
            if getattr(value, 'dtype', None) is not None and value.dtype == np.float64:
                mesh.arrays[name] = np.ascontiguousarray(value, dtype=np.float32)
        hydration = hydration_future.result()
        # The scalar copy and the deformation weights run beside the geometry, vertical grid and state.
        words_future = side.submit(pool_words, hydration)
        mesh.arrays['nominalMinDc'] = hydration.horizontal_length()
        values = _run(mesh, mode=0, dtype=np.dtype(np.float32), include_halos=True)
        for name, value in zip(('edgeNormalVectors', 'localVerticalUnitVectors', 'cellTangentPlane',
                                'coeffs_reconstruct'), values):
            mesh.arrays[name] = value
        grid = load_mpas_vertical_grid(init, mesh, config_coef_3rd_order=layout.configs['config_coef_3rd_order'],
                                       allow_regional_sentinels=True)
        profiles = load_v841_reference_wind_profiles(init, n_vert_levels=grid.vertical_grid.n_vert_levels)
        names = next(r['components'] for r in layout.fields if r['pool'] == 'state' and r['name'] == 'scalars')
        state, reference, saved = load_mpas_initial_state(init, mesh, grid.vertical_grid,
                scalar_names=(), reference_path=init, terrain_metrics=grid.terrain_metrics,
                return_saved_diagnostics=True, cold_start=True, allow_regional_sentinels=True)
        scalars, deformation = words_future.result()
    return _finish_regional_inputs(mesh, grid, profiles, state, reference, saved, names, scalars, deformation,
                                   keep_initialized_deformation)


def _finish_regional_inputs(mesh, grid, profiles, state, reference, saved, names, scalars, deformation,
                            keep_initialized_deformation):
    import numpy as np
    state.scalars = scalars
    for name in ('defc_a', 'defc_b'):
        if keep_initialized_deformation:
            # MPAS-Model-rrfs/src/core_atmosphere/mpas_atm_core.F:373-614: the fork atmosphere never rebuilds these;
            # it reads the init's words, cut-border cells included (deformation_rrfs.select_deformation_weights_rrfs).
            # The pool carries native's garbage column (nCells+1); the cut's own cells come first.
            mesh.arrays[name] = np.ascontiguousarray(deformation[name][..., :mesh.n_cells].T)
        else:
            mesh.arrays[name] = np.zeros((mesh.n_cells, mesh.arrays['edgesOnCell'].shape[1]), np.float32)
    return mesh, grid, profiles, state, reference, saved, names


#: The lake class of the 21-class MODIS land use (modis_landuse_20class_30s_with_lakes, WPS index islake=21),
#: which ``rw_mpas_static`` writes; MPAS's VEGPARM.TBL row MODIFIED_IGBP_MODIS_NOAH has 20 classes.
MODIS_LAKE_CLASS = 21


def fold_lake_class(native, layout):
    """Lake cells become water when no lake model runs, as WRF real does (module_initialize_real.F, islake).

    THE BREAKAGE THIS PREVENTS: a hex-door static (every moving member, any ``rw_mpas_static`` mesh) takes its
    land use from the 21-class MODIS set with lakes. MPAS-A v8.4.2 has no lake model and its Noah table holds
    20 MODIS classes, so one lake cell stopped every MPAS-suite run in the land-surface init
    ("MPAS Noah native fatal condition at source line 2441": too many input landuse types). An
    init_atmosphere static has no class 21, so this changes nothing there. Returns the folded cell count.
    """
    import numpy as np
    fields = getattr(native, '_fields', {})
    # Registry names -> native pool names (iswater_lu is held as iswater).
    names = {row['name']: row['pool_name'] for row in layout.fields if row['pool'] == 'sfc_input'}
    water_name = names.get('iswater_lu')
    if ('sfc_input', names.get('ivgtyp')) not in fields or ('sfc_input', water_name) not in fields:
        return 0
    try:
        mminlu = str(native.get_text_field('sfc_input', names.get('mminlu', 'mminlu')))
    except Exception:
        return 0    # the class table is unknown: class 21 is a land class in USGS, so nothing is folded
    if 'MODIS' not in mminlu.upper():
        return 0
    water = int(np.asarray(native.get_field('sfc_input', water_name)).reshape(-1)[0])
    folded = 0
    for name in (names.get('ivgtyp'), names.get('lu_index')):
        if name is None or ('sfc_input', name) not in fields:
            continue
        values = native.get_field('sfc_input', name)
        count = layout.dimensions['nCells']
        lake = values[..., :count] == MODIS_LAKE_CLASS
        if lake.any():
            values[..., :count][lake] = water
            native.put_field('sfc_input', name, values, nlevels=fields['sfc_input', name][2])
            folded = max(folded, int(lake.sum()))
    return folded


def native_table_roots(source,config,extra=None):
    """Folders whose tables a native run stages in its run directory, in order.

    THE BREAKAGE THIS PREVENTS: Noah-MP reads NoahmpTable.TBL from the run directory, and MPAS keeps it in
    physics_noahmp/parameters, not in physics_wrf/files.  A door run that selected sf_noahmp (by the licence
    opt-in, woof.hex.licence_gate) on native-cpu stopped in Noah-MP's table reader with a Fortran end-of-file
    error unless the user also named that folder.  The folder is added only when the run selects Noah-MP, so
    every other run stages the same tables, and records the same table digests, as before.
    """
    roots=[Path(source)/'src/core_atmosphere/physics/physics_wrf/files',*(extra or [])]
    noahmp=Path(source)/'src/core_atmosphere/physics/physics_noahmp/parameters'
    if config.selectors.get('land_surface')=='sf_noahmp' and noahmp.is_dir() and noahmp not in roots:
        roots.append(noahmp)
    return roots


def balance_series_sampler(arguments, layout, names, vertical, dycore, out):
    """``--balance-series`` on the MPAS suite route: the WOOF route's sampler on this dycore's own state.

    woof.hex.iau_forcing.BalanceSeries reads rho_zz and the mass species of the owned columns from
    ``driver.atmosphere.state``, which the suite route's CUDA dycore holds exactly as the WOOF route's does.
    THE BREAKAGE THIS RETIRES (hex parity run group 7, #25): the door refused the flag on this route because
    no sampler was wired here. Returns None when the flag is off.
    """
    if not getattr(arguments, 'balance_series', False):
        return None
    from types import SimpleNamespace
    import numpy as np
    from ..iau_forcing import BalanceSeries
    cells = int(layout.dimensions['nCells'])
    solve = int(layout.dimensions.get('nCellsSolve', cells))
    if solve != cells:
        # THE BREAKAGE THIS PREVENTS: each rank would write its own partial series to one path.
        raise ValueError('the balance series is sampled on one card; this rank owns '
                         f'{solve} of {cells} local columns, so drop --balance-series or run on one card')
    balance = BalanceSeries.build(Path(out) / 'balance-series.jsonl', tuple(names),
                                  np.asarray(vertical.dzw, dtype=np.float64), dycore.cp)
    balance.view = SimpleNamespace(classes=[SimpleNamespace(driver=dycore, n_owned_cells=solve,
                                                            tables=SimpleNamespace(rate=1))])
    return balance


#: Mesh connectivity whose stored-0 "no neighbour" entries native MPAS maps to the garbage element at read time
#: (the element count of that kind plus one), by the kind of element each array names.
LIMITED_AREA_CONNECTIVITY = {
    'cellsOnEdge': 'nCells', 'cellsOnCell': 'nCells', 'cellsOnVertex': 'nCells',
    'edgesOnCell': 'nEdges', 'edgesOnEdge': 'nEdges', 'edgesOnVertex': 'nEdges',
    'verticesOnCell': 'nVertices', 'verticesOnEdge': 'nVertices',
}


def remap_limited_area_sentinels(native, layout):
    """A cull's stored-0 neighbours become the garbage element, as native MPAS's block creator does.

    THE BREAKAGE THIS PREVENTS: the pinned pool reader keeps a limited-area cull's 0 ("no neighbour") in
    cellsOnEdge, cellsOnVertex, edgesOnVertex and the rest, where native MPAS reads it as nCells+1 (nEdges+1,
    nVertices+1).  Every one-based consumer on the suite route (the isobaric and convective output diagnostics,
    which gather through cellsOnVertex) then indexed element -1, one word before its buffer: an illegal memory
    access at the first output on the 937.5 m region of 2024-05-06 whenever the allocation before the buffer was
    unmapped (4 of 6 runs; fix-group-1 #3).  Returns the count remapped per array.
    """
    import numpy as np
    dimensions = layout.dimensions
    rows = {row['name']: row for row in layout.fields if row['pool'] == 'mesh'}
    remapped = {}
    for name, kind in LIMITED_AREA_CONNECTIVITY.items():
        row = rows.get(name)
        if row is None or ('mesh', row['pool_name']) not in native._fields:
            continue
        values = native.get_field('mesh', row['pool_name'])
        stored = values == 0
        count = int(stored.sum())
        if count:
            values[stored] = np.int32(dimensions[kind] + 1)
            native.put_field('mesh', row['pool_name'], values, nlevels=native._fields['mesh', row['pool_name']][2])
        remapped[name] = count
    return remapped


def framework_only(config):
    """Choose the ported framework for the GPU route without an opt-in.

    Explicit native-cpu physics remains a referee. Each suite row names its
    ported framework (route_table): the RRFS fork's is a separate source
    contract with its own ABI metadata, so a row whose framework modules are
    not in this distribution keeps the native library. MPAS
    mpas_atmphys_control.F:310-319 selects active schemes; disabled slots do
    not require native runtime storage.
    """
    from .route_table import framework_absence
    if framework_absence(config) is not None:
        return False
    if any(str(backend) == 'native-cpu' and config.active(slot)
           for slot, backend in config.backends.items()):
        return False
    return not (config.options.get('config_sst_update', False) and
                str(config.service_backends.get('surface_update', 'native-cpu')) == 'native-cpu')


def open_limited_area_dycore(arguments, layout, mesh, grid, profiles, state, reference, saved, names, *,
                             dt, start, config, les, kernel_cache, dycore_config=None, driver_type=None,
                             iau_wrapped=False):
    """The regional dycore for a limited-area suite run; mpas_atm_core.F:539-596, mpas_atm_boundaries.F.

    This is the WOOF route's regional assembler (cuda_regional_forecast_v841.open_regional_forecast_v841): the same
    boundary driver, specified and relaxation zones and lateral-boundary series, so a limited-area run differs between
    the two routes only in its physics.  The boundary drives every scalar the suite carries (lbc_<name>); a species
    the series does not hold keeps native's recycled allocation, as on the WOOF route.

    A suite row with a fork dycore (``dycore_config`` resolved from its table row, e.g. RRFS's
    mpas-v8.3.1-noaa) takes that variant's own regional constructor (dycore_variants.dycore_regional_constructor:
    cuda_regional_forecast_rrfs.assemble_regional_driver_rrfs), which puts the same v8.4.1 regional runtime (boundary
    driver, masks, LBC pool) under the fork's dycore driver.  THE BREAKAGE THIS FIXES: hrrrv5 on a limited-area mesh
    refused at load ("runs through its own regional route"), and no caller of that constructor existed, so NOAA's own
    RRFS v2 domain (conus3km) could not run on the RRFS route on any card count (2026-10-28).
    """
    from ..cuda_regional_forecast_v842 import RegionalNativeDycoreConfigV842
    from ..cuda_regional_forecast_v841 import open_regional_forecast_v841
    from ..lbc_stream import initial_paths
    lbc_dir = getattr(arguments, 'lbc_dir', None)
    if lbc_dir is None:
        raise ValueError('config_apply_lbcs is set but the run names no --lbc-dir')
    paths = initial_paths(lbc_dir)
    if not paths:
        raise ValueError(f'--lbc-dir {lbc_dir} holds no lbc.*.nc: the limited-area run has no boundary series')
    start_time = datetime.strptime(start, '%Y-%m-%d_%H:%M:%S')
    scalar_names = tuple('lbc_' + name for name in names)
    from ..dycore_variants import dycore_regional_constructor
    from ..cuda_regional_forecast_v841 import open_regional_forecast_v841 as _shared_regional
    # A fork dycore with its own regional driver builds it here. A variant on the shared regional driver (the
    # NOAA v8.4.1 row) resolves through the regional configuration below, which carries the regional fields.
    if (dycore_config is not None and getattr(dycore_config, 'dycore_variant', None) is not None
            and dycore_regional_constructor(dycore_config) is not _shared_regional):
        from dataclasses import replace
        if les is not None:
            # THE BREAKAGE THIS PREVENTS: the fork's regional constructor builds no LES dissipation frame, so an LES
            # namelist would run without it and say nothing.
            raise ValueError('a limited-area fork-dycore run has no LES dissipation frame; run it without LES')
        construct = dycore_regional_constructor(dycore_config)
        # An IAU run hands the fork's regional constructor the stage-density driver (with_stage_density over the
        # fork's own class); every other run constructs the unchanged class (fix/hex-lam-da-v842).
        wrapped = {'driver_type': driver_type} if iau_wrapped else {}
        return construct(mesh, state, grid.vertical_grid, reference, saved, grid.terrain_metrics,
            replace(dycore_config, config_apply_lbcs=True), reference_wind_profiles=profiles, lbc_paths=paths,
            start_time=start_time, kernel_cache=kernel_cache, scalar_names=scalar_names, **wrapped,
            # Fork Registry.xml:268 (default 6.0), read by atm_bdy_adjust_dynamics_relaxzone_tend.
            config_relax_zone_divdamp_coef=float(layout.configs.get('config_relax_zone_divdamp_coef', 6.0)))
    options = {f.name: layout.configs[f.name] for f in fields(RegionalNativeDycoreConfigV842)
               if f.init and f.name in layout.configs}
    options.update(config_dt=dt, config_moist_physics=config.moist_physics, config_apply_lbcs=True)
    from .route_table import route_row
    from ..dycore_variants import resolve_dycore_variant
    regional_config = RegionalNativeDycoreConfigV842(**options)
    row = route_row(config)
    if row.get('dycore') is not None:
        regional_config = resolve_dycore_variant(regional_config, row)
        variant_options = {f.name: layout.configs[f.name] for f in fields(type(regional_config))
                           if f.init and f.name in layout.configs}
        regional_config = resolve_dycore_variant(RegionalNativeDycoreConfigV842(**options), row,
                                                overrides=variant_options)
    return open_regional_forecast_v841(mesh, state, grid.vertical_grid, reference, saved, grid.terrain_metrics,
        regional_config, reference_wind_profiles=profiles, lbc_paths=paths,
        start_time=start_time, scalar_names=scalar_names, kernel_cache=kernel_cache,
        les_dissipation=les, driver_type=driver_type)


def limited_area_receipt(arguments, dycore, coupling=None):
    """What drove the boundary, for the run's receipt."""
    runtime = dycore.regional_v841
    block = dict(lbc_dir=str(Path(arguments.lbc_dir).resolve()), dycore='regional (open_regional_forecast_v841)',
                 anchor=getattr(runtime, 'anchor', None))
    if coupling is not None:
        applied = coupling.boundary_forcing_columns() is not None
        block['LB1'] = dict(applied=applied,
            what='GF / New Tiedtke resolved-advection forcing (rthdynten, rqvdynten) zeroed in boundary rings 1-7',
            why='the boundary relaxation and the specified-zone reset are not convergence; native MPAS-A v8.4.2 '
                'rains a ring along the boundary from them',
            off=('exact mode' if not applied and getattr(coupling.session.config, 'mode', 'exact') != 'correct'
                 else 'WOOF_HEX_MPAS_LB1=0' if not applied else None))
    try:
        block['runtime'] = runtime.receipt()
    except Exception as error:      # a receipt detail never fails a finished run
        block['runtime_receipt_error'] = repr(error)
    return block


def run_native_dynamics(arguments, config, layout, payload, streams, library,
                        start, stop, dt, steps, out, *, moving=None, execution=None):
    """Retain startup timings on successful and failed sessions."""
    timings = []
    runtime_rows = []
    from .. import time_ledger
    try:
        with timed_stage(runtime_rows, 'native-runtime',
                         parallelism={'device':'GPU plus native CPU' if library is not None
                                      else 'GPU, issued by one host thread (ported framework, no native library)',
                                      'cards':1},
                         timeline='post.forecast_process.native.runtime'):
            return _run_native_dynamics(arguments, config, layout, payload, streams, library,
                start, stop, dt, steps, out, moving=moving, execution=execution, timings=timings)
    finally:
        time_ledger.end_if_open('prep.door')
        time_ledger.end_if_open('forecast.construct')
        time_ledger.end_if_open('forecast.integrate')
        # An LTS rate class writes its own file, which run_lts_classes folds into the run's timeline.  THE BREAKAGE THIS
        # PREVENTS (HEX-SHAPES-1CARD 2026-10-29): every class thread wrote this one path, so an LTS run's timeline was
        # whichever class finished last, and the other classes' steps (most of each macro) were in no labelled stage.
        timings_path = getattr(execution, 'timings_path', None)
        path = Path(timings_path(out)) if callable(timings_path) else out/'native-runtime-timings.json'
        path.write_text(json.dumps({'stages':runtime_rows + timings},indent=2)+'\n')


def _run_native_dynamics(arguments, config, layout, payload, streams, library,
                        start, stop, dt, steps, out, *, moving=None, execution=None, timings=None):
    """Real-door native physics plus resident CUDA dynamics; mpas_atm_core.F:589-701,1036-1040.

    ``moving`` (a ``moving_v842.MovingRoute``, only when a WOOF_HEX_MOVING_* switch is set) adds the moving
    region's hooks: remapped resume, physics carry, shock series, frames, stop and carrier export.
    ``execution`` supplies inputs, an optional driver factory and process-local hooks. It does not
    relax the native singleton or the resident glue's global-domain admission.
    """
    from .config_v842 import NativeMpasDycoreConfigV842
    from ..cuda_driver import CudaDryDycoreDriver
    from ..cuda_backend import KernelCache,require_cuda
    from ..cuda_physics_prep_v841 import CudaMpasToPhysGeometryV841

    from .. import time_ledger
    card_attrs = {"card": time_ledger.card_index(), "cards": 1, "backend": "hex-cuda"}
    # Initialization loads scheme resources inside native_working_directory.
    # Resolve the caller's packet before that directory change, for both the
    # framework and the CUDA provider, without changing the caller's arguments.
    cuda_resources = getattr(arguments, 'physics_resources', None)
    if cuda_resources is not None:
        cuda_resources = Path(cuda_resources).resolve()
    # MPAS mpas_subdriver.F:364 copies configs into stream attributes before
    # core init. Hydration resolves config_len_disp later (mpas_atm_core.F:159-191);
    # that resolved runtime value must not replace the earlier output attribute.
    output_config_snapshot = dict(layout.configs)

    receipt_path = (out/'native-physics-receipt.json' if execution is None
                    else execution.receipt_path(out))
    resumed = moving is not None and moving.carrier is not None
    # RP6 (correct mode with config_pbl_interval). Breakage this refusal prevents: the surface precipitation
    # window (precip_window_v842) lives on the device and the pristine Registry restart stream cannot hold it, so
    # a native-file restart would silently drop the precipitation that fell since the last land-surface call.
    # Native correct mode keeps the window in its restart stream (package bl_precip_window_in).  A moving
    # resume is not a native-file restart: its carrier holds the window (moving_v842), so it is not refused.
    if getattr(config,'correct',None) is not None and config.correct('RP6') and config.active('land_surface') \
            and layout.configs['config_do_restart'] and not resumed:
        raise ValueError('a native-file restart of correct mode with config_pbl_interval (RP6) would drop the '
                         'precipitation added up since the last land-surface call: the pristine restart stream '
                         'cannot hold the window. Restart in exact mode, without config_pbl_interval, or with RP6 '
                         'in config_physics_correct_skip')
    # IN3 (correct mode with IAU on). Breakage this refusal prevents: the pristine Registry restart stream holds no
    # IAU window position (iau_window_elapsed), so a native-file restart inside a window would begin the window
    # again and admit the increment a second time. A DA-cycling restart (config_do_DAcycling) begins a new
    # window by rule and runs; exact mode runs MPAS-A v8.4.2's own behaviour (a new window every run).
    if (str(layout.configs.get('config_IAU_option','off')).strip()=='on' and layout.configs['config_do_restart']
            and getattr(config,'mode','exact')=='correct' and not layout.configs.get('config_do_DAcycling',False)):
        raise ValueError('a native-file restart with IAU on in correct mode would admit the window again: the '
                         'pristine restart stream holds no IAU window position. Restart with config_do_DAcycling '
                         '= true (a new DA cycle, a new window), in exact mode, or start the window on a cold start')
    signature=inspect.signature(CudaDryDycoreDriver._step_device_v841).parameters
    missing={'physics_post_transport_callback','physics_mass_loading_count'}-set(signature)
    if config.moist_physics and missing:
        raise RuntimeError('native physics dycore reconciliation is absent: '+', '.join(sorted(missing)))
    # The forecast door resolves admission and passes no library for the
    # ported route. Explicit proof tools can supply a native referee even
    # with GPU physics; repeating config selection here would ignore it.
    ported_framework = library is None
    if ported_framework:
        # The suite row's own ported framework (v8.4.2 or the RRFS fork), route_table.
        from .route_table import framework_entry
        backend_type = framework_entry(config, 'backend')
    else:
        backend_type = NativePhysics
    # A table can bind constructor arguments without changing the backend API.
    from functools import partial
    backend_api = backend_type.func if isinstance(backend_type, partial) else backend_type
    for method in ('initialize_state_time','update_state_time'):
        if not callable(getattr(backend_api,method,None)):
            raise RuntimeError(f'native output time API {method} is absent: stale xtime/CF Time would invalidate every output')
    from ..physics_suites import load_table
    row=load_table()[getattr(config,'table_key','mpasphys_v842')]
    keep_initialized_deformation = keeps_initialized_deformation(row)
    from ..host_parallel import threads as host_threads
    if ported_framework:
        # Scheme lookup files that depend on nothing this run computes are read on a background thread beside the
        # dycore inputs and build; physics_init takes the same words (FrameworkInitializationRRFS.prefetch_tables).
        try:
            prefetch_tables = getattr(framework_entry(config, 'initialization'), 'prefetch_tables', None)
        except RuntimeError:
            prefetch_tables = None
        if prefetch_tables is not None:
            # The directories the backend below receives as ``tables`` (the same condition), so the paths match.
            named_tables = getattr(arguments, 'native_physics_tables', None)
            named_tables = ([Path(t).resolve() for t in named_tables]
                            if named_tables and 'tables' in inspect.signature(backend_type).parameters else None)
            # The fork source the backend reads its tables from: the row's bound resource and pin set (the 44c0849
            # row binds its own), so the read-ahead takes the files read_tempo_lookup_files will ask for.
            bound = backend_type.keywords if isinstance(backend_type, partial) else {}
            prefetch_tables(layout.configs, named_tables,
                            **{name: bound[name] for name in ('resource', 'pin_set') if name in bound})
    with timed_stage(timings, 'dycore-inputs', parallelism={'device':'CPU','host_threads':host_threads(),
                     'host_schedule':'main-thread chain (mesh, vertical grid, state) with threaded reads and copies; payload on a side thread'},
                     timeline='prep.native.dycore_inputs'):
        input_loader = getattr(execution, 'load_inputs', None)
        if input_loader is not None:
            inputs = input_loader(load_native_dynamics_inputs, arguments, layout, payload,
                                  keep_initialized_deformation=keep_initialized_deformation)
        elif execution is not None:
            inputs = execution.inputs
        else:
            inputs = load_native_dynamics_inputs(arguments,layout,payload,
                keep_initialized_deformation=keep_initialized_deformation)
    mesh,grid,profiles,state,reference,saved,names=inputs
    physics_map=None
    if moving is not None:
        moving.begin(layout.configs['config_start_time'] if not resumed else None)
    if resumed:
        # The carrier's atmosphere on this member (the remap the ArWen route uses), before the dycore exists.
        state,saved,physics_map=moving.resume_dycore(mesh=mesh,grid=grid,reference=reference,state=state,
                                                     saved=saved,names=names,
                                                     target_grid=getattr(arguments,'grid',None) or arguments.init)
    if execution is not None:
        state,saved=execution.couple_input(state,saved,mesh,grid,reference)
    chemistry=None
    if getattr(arguments,'chemistry_config',None) is not None:
        # The ArWen route's aerosol transport on this route (woof.hex.mpasphys.chemistry_v842).
        if execution is not None or physics_map is not None or (moving is not None and moving.control is not None):
            # THE BREAKAGE THIS PREVENTS: a multi-rank or moving run would remap or partition the Registry
            # scalars but not the appended species rows, so the plume would be cut at the first move or halo.
            raise ValueError('chemistry on the MPAS suite route runs on one card without a moving region: the '
                             'moving remap and the rank partition carry the Registry scalars only')
        from .chemistry_v842 import SuiteChemistry
        chemistry=SuiteChemistry(arguments.chemistry_config,registry_names=names,mesh=mesh,
                                 nlev=grid.vertical_grid.n_vert_levels,
                                 grid_path=getattr(arguments,'grid',None) or arguments.init,
                                 start_text=layout.configs['config_start_time'])
        if resumed:
            chemistry.restart=moving.carrier.get('chemistry')
            if chemistry.restart is None:
                raise ValueError('this checkpoint carries no chemistry; a chemistry run restarts from its own checkpoint')
        state=chemistry.extend_state(state)
    elif resumed and moving.carrier.get('chemistry') is not None:
        raise ValueError('this checkpoint carries chemistry: restart it with the same --chemistry-config')
    options={f.name:layout.configs[f.name] for f in fields(NativeMpasDycoreConfigV842)
             if f.init and f.name in layout.configs}
    options.update(config_dt=dt,config_moist_physics=config.moist_physics)
    from ..physics_suites import load_table
    row=load_table()[getattr(config,'table_key','mpasphys_v842')]
    dycore_config=NativeMpasDycoreConfigV842(**options)
    driver_type=CudaDryDycoreDriver
    cache_type=KernelCache
    if row.get('dycore') is not None:
        from ..dycore_variants import resolve_dycore_variant,dycore_driver_type,dycore_kernel_cache_type
        from dataclasses import fields as config_fields
        # The Registry supplies variant-only namelist settings too.
        dycore_config=resolve_dycore_variant(dycore_config,row)
        fork_options={f.name:layout.configs[f.name] for f in config_fields(type(dycore_config))
                      if f.init and f.name in layout.configs}
        dycore_config=resolve_dycore_variant(NativeMpasDycoreConfigV842(**options),row,overrides=fork_options)
        driver_type=dycore_driver_type(dycore_config)
        cache_type=dycore_kernel_cache_type(dycore_config)
        selected_signature=inspect.signature(driver_type._step_device_v841).parameters
        absent={'physics_post_transport_callback','physics_mass_loading_count'}-set(selected_signature)
        if config.moist_physics and absent:
            raise RuntimeError('selected fork dycore lacks the post-transport physics boundary: '+', '.join(sorted(absent)))
    base_driver_type = driver_type
    if str(layout.configs.get('config_IAU_option','off')).strip()!='off':
        driver_type=with_stage_density(driver_type)
    les=None
    if dycore_config.config_les_model != 'none' or dycore_config.config_mix_scalars:
        # The v8.4.2 LES dissipation module (mpas_atm_dissipation_models.F)
        # at the native RK boundary, mpas_atm_time_integration.F:6331-6390.
        # Default (LES off, no scalar mixing) runs never construct it.
        from ..cuda_les_v842 import ResidentLesV842
        hydration=NativePoolHydration(None,layout,payload)
        les=ResidentLesV842(mesh,grid.vertical_grid,profiles,dycore_config,names,
                            hydration.field_buffer('mesh','t_init')[...,:-1])
    # Select the factory only after variant and IAU wrappers are resolved.
    build_dycore = getattr(execution, 'build_dycore', None)
    factory = driver_type.from_host
    if build_dycore is not None:
        from functools import partial
        factory = partial(build_dycore, factory)
    limited_area = bool(layout.configs.get('config_apply_lbcs', False))
    # The plain dycore class of this row: the fork's own driver for a fork-dycore row (its regional constructor builds
    # that class under the regional runtime), else MPAS-A's.
    plain_type = (dycore_driver_type(dycore_config) if row.get('dycore') is not None else CudaDryDycoreDriver)
    iau_wrapped = driver_type is not base_driver_type
    # A checkpoint, restart or DA analysis on the run's own mesh (checkpoint_v842 fixed_mesh) and the one-card JEDI
    # execution (features_v842.CyclingExecution) build no dycore of their own: the regional dycore opens below on the
    # carried or analysed state at the resume time (fix/hex-lam-da-v842).
    moves = moving is not None and not getattr(moving, 'fixed_mesh', False)
    rank_execution = execution is not None and not getattr(execution, 'single_card', False)
    if limited_area and (moves or base_driver_type is not plain_type or (iau_wrapped and rank_execution)):
        # THE BREAKAGE THIS PREVENTS: the moving wrapper remaps onto another member's mesh and a rank execution builds
        # its own dycore class; applied to the regional dycore they would drop its boundary driver.  IAU's
        # stage-density wrapper is carried onto the MPAS-A and the RRFS fork's regional dycores on one card
        # (guide/l11-regional-iau, fix/hex-lam-da-v842); a same-host rank's regional dycore does not take it yet.
        # bind_limited_area refuses these options at the door; this names any other path that reaches here.
        raise ValueError('a limited-area MPAS suite run takes the plain regional dycore: no moving wrapper, '
                         'and IAU only on one card, is ported onto it yet')
    with timed_stage(timings, 'dycore-build', parallelism={'device':'GPU','cards':1,'host_threads':1}, timeline='prep.native.dycore_build'):
        if limited_area:
            dycore=open_limited_area_dycore(arguments, layout, mesh, grid, profiles, state, reference, saved,
                names, dt=dt, start=start, config=config, les=les, driver_type=driver_type, iau_wrapped=iau_wrapped,
                kernel_cache=cache_type(capability=require_cuda(min_compute=(7,5))),
                dycore_config=dycore_config if row.get('dycore') is not None else None)
        else:
            dycore=factory(mesh,state,grid.vertical_grid,reference,
                dycore_config,saved_diagnostics=saved,
                terrain_metrics=grid.terrain_metrics,reference_wind_profiles=profiles,
                **({'les_dissipation':les} if les is not None else {}),
                # Use the architecture/compiler-keyed persistent cache. A cache
                # under pool scratch forced every forecast to rebuild unchanged
                # dycore and physics kernels after disposable output was removed.
                kernel_cache=cache_type(capability=require_cuda(min_compute=(7,5))))
    if build_dycore is not None:
        # Validate the actual returned driver before entering the native context.
        # Binding accepts forwarding **kwargs wrappers as well as named keywords.
        try:
            inspect.signature(getattr(dycore, '_step_device_v841', None)).bind(
                physics_stage_callback=None, physics_post_transport_callback=None,
                physics_mass_loading_count=0)
        except (TypeError, ValueError) as error:
            raise RuntimeError('execution-built dycore lacks the required physics callback interface') from error
    if execution is not None:
        execution.bind_dycore(dycore)
    # --fullstep-local-timestep in shadow mode: the ArWen route's observer on this dycore (woof.hex.mpasphys.lts_v842).
    fullstep_shadow=None
    if execution is None and moving is None and getattr(arguments,'fullstep_local_timestep',False):
        from .lts_v842 import attach_shadow, shadow_requested
        if shadow_requested(arguments):
            fullstep_shadow=attach_shadow(arguments,dycore,out)
    # The regional dycore's arrays carry native's garbage column (nCells+1, nEdges+1); the preparation geometry must
    # be the same padded view (cuda_regional_forecast_v842 does the same).
    geometry_factory = getattr(execution, 'build_physics_geometry', CudaMpasToPhysGeometryV841.from_host)
    geometry_mesh = dycore.regional_v841.padded if limited_area else mesh
    geometry = (geometry_factory(CudaMpasToPhysGeometryV841.from_host, geometry_mesh)
                if execution is not None and hasattr(execution, 'build_physics_geometry')
                else geometry_factory(geometry_mesh))
    receipt=dict(schema='mpas-native-forecast-v1',identity_checked=False,completed=False,
                 physics=config.receipt(),steps_requested=steps,steps_completed=0,
                 native_library_sha256=None if ported_framework else hashlib.sha256(library.read_bytes()).hexdigest(),
                 framework='ported' if ported_framework else 'native referee',streams=[],stage_timings=timings)
    table_roots=([] if ported_framework else
        native_table_roots(layout.source,config,getattr(arguments,'native_physics_tables',None)))
    receipt['table_sha256']={}
    selector=row.get('native_table_selector')
    if ported_framework:
        # Scheme resources are authenticated and loaded by the CUDA provider.
        # The installed Registry contains no native source/table directory.
        tables=()
    elif selector is None:
        tables=(table for root in table_roots for table in sorted(root.iterdir())
                if table.is_file() and (table.suffix in ('.TBL','.DBL')
                    or table.name.startswith(('RRTMG_','CCN_'))))
    else:
        from importlib import import_module
        module,operation=selector.split(':',1)
        callback=getattr(import_module(module),operation)
        keyword=row.get('native_table_override_keyword')
        tables=(callback(table_roots[:1], **{keyword: table_roots[1:]})
                if keyword else callback(table_roots))
    for table in tables:
        target=out/table.name
        if target.exists() and target.resolve()!=table.resolve():
            raise ValueError(f'native table destination already differs: {target}')
        if not target.exists():target.symlink_to(table.resolve())
        receipt['table_sha256'][table.name]=hashlib.sha256(table.read_bytes()).hexdigest()
    selector=row.get('native_initialization_namelist')
    if selector is not None:
        from importlib import import_module
        module, operation=selector.split(':',1)
        # The run's namelist file, or the one it resolved from --physics-suite (forecast_v842.run_forecast).
        staged=(getattr(arguments,'physics_namelist',None)
                or getattr(arguments,'resolved_physics_namelist',None))
        if staged is None:
            raise ValueError('the route stages the run namelist into the native working directory, and this run '
                             'has neither --physics-namelist nor a resolved suite namelist')
        receipt['initialization_namelist_sha256']=getattr(import_module(module),operation)(staged,out)
    native_options={'pin_set':config.pin_set} if getattr(config,'pin_set',None) is not None else {}
    surface_record_reader = None
    if config.service_backends.get('surface_update') == 'hex-cuda' and config.options['config_sst_update']:
        from .surface_update_v842 import RustSurfaceRecordReader
        resource_path = cuda_resources
        resource = None
        if resource_path is not None:
            resource_path = Path(resource_path).resolve()
            from .resources_build import load_packet
            resource = load_packet(resource_path).get('surface_update', {}).get('reader_library')
        if resource is not None:
            reader_library = Path(resource['path'])
            if not reader_library.is_absolute():
                reader_library = resource_path.parent / reader_library
            reader_library = reader_library.resolve()
            if hashlib.sha256(reader_library.read_bytes()).hexdigest() != resource['sha256']:
                raise ValueError('surface update Rust reader_library SHA256 differs')
        else:
            # Default-on surface updating (s2s-surface-update): the packaged
            # source-built Rust reader serves runs without a resource packet.
            from ..surface_series_reader import reader_library as packaged_reader
            reader_library = packaged_reader()
        reader_sha256 = hashlib.sha256(reader_library.read_bytes()).hexdigest()
        surface = next(s for s in streams if s.name == 'surface' and 'input' in s.directions)
        surface_file = Path(surface.input_file or expand_input_filename(surface.filename, start)).resolve()
        surface_record_reader = (RustSurfaceRecordReader(reader_library, surface_file,
                                                        n_cells=layout.dimensions['nCells'])
            if execution is None else execution.surface_reader(
                RustSurfaceRecordReader, reader_library, surface_file, layout))
        receipt['surface_input'] = dict(path=str(surface_file), sha256=hashlib.sha256(surface_file.read_bytes()).hexdigest(),
                                       reader_library=str(reader_library), reader_sha256=reader_sha256, reads=[])
    ported_options = {}
    # The RRFS fork framework reads its NOAA lookup files from a directory, so a run's --native-physics-tables
    # reaches it (framework_init_rrfs.tempo_lookup_path). The v8.4.2 framework's tables come from the resources
    # packet; it keeps ignoring the option, as the gate decks that name it for the library route expect.
    if (ported_framework and getattr(arguments, 'native_physics_tables', None)
            and 'tables' in inspect.signature(backend_type).parameters):
        ported_options['tables'] = [Path(t).resolve() for t in arguments.native_physics_tables]
    take_backend = getattr(execution, 'take_physics_backend', None)
    ahead = take_backend() if ported_framework and take_backend is not None else None
    prehydrated = None
    if ahead is not None:
        backend, prehydrated = ahead
    else:
        backend = (backend_type(layout=layout, resources=cuda_resources, **ported_options)
                   if ported_framework else backend_type(library,test_only=True,**native_options))
    with backend as native, ExitStack() as output_lifetime:
        coupling=None
        core_operations={}
        core_initialize=None
        module=row.get('native_core_module')
        if module is not None:
            from importlib import import_module
            core_provider=import_module(module)
            core_initialize=core_provider.initialize_native_core
            core_operations['atm_reset_diagnostics']=lambda call:core_provider.reset_native_diagnostics(native,session=call.pools)
        with timed_stage(timings, 'physics-session', parallelism={'device':'CPU','host_threads':1}, timeline='prep.native.physics_session'):
            pool_options = {}
            if getattr(execution, 'moving_gpu_build', False):
                from .moving_class_gpu_v842 import private_pool_mapping
                pool_options['pool_copy_on_write'] = private_pool_mapping()
            session=NativePhysicsSession(native,layout,payload,config,dt=dt,start=start,stop=stop,
                core_operations=core_operations,
                stream_intervals={s.name:(s.input_interval if s.name=='surface' else s.output_interval) for s in streams},
                post_transport=lambda call:coupling.prepare_transport(call),cuda_device=dycore.cp,
                cuda_resources=cuda_resources,
                surface_record_reader=surface_record_reader,
                **({'prehydrated': prehydrated} if prehydrated is not None else {}), **pool_options)
        session.output_config_snapshot=output_config_snapshot
        if os.environ.get('WOOF_HEX_MOVING_RESIDENT') == '1':
            receipt['physics_session_preparation'] = session.preparation_timings
        # A rank's scheme initialisation takes global extrema across ranks, as native's mpas_dmpar_min/max_real
        # (RRFS TEMPO aerosols, framework_init_rrfs); None on one card.
        session.rank_reduce=getattr(execution,'reduce',None)
        if not ported_framework and getattr(config,'pin_set',None)=='rrfs':
            # The native library's physics_run_init reads the output and diagnostics stream intervals from the
            # native stream manager (MPAS-Model-rrfs mpas_atmphys_manager.F:646-665), which MPAS's subdriver fills
            # before core init.  THE BREAKAGE THIS PREVENTS: with no stream registered there the fork logged
            # "Stream output does not exist in call to MPAS_stream_mgr_get_property()" and stopped in
            # mpas_timekeeping ("Fortran runtime error: Bad integer for item 1 in list input"), so the RRFS route
            # on its native library never reached step 1 (hrrrv5 x1.2562, 2026-10-28).  The regional runner
            # registers them at the same point (cuda_regional_forecast_v842).  The ported framework reads the
            # intervals from the Python stream list instead and needs no native stream manager.
            receipt['native_stream_missing_fields']=_configure_streams(native,streams,out,layout)
        receipt['lake_cells_folded_to_water']=fold_lake_class(native,layout)
        if limited_area:
            receipt['limited_area_sentinels']=remap_limited_area_sentinels(native,layout)
        from .streams_output_v842 import PhysicsStreams, output_inventory
        receipt['missing_fields']=output_inventory(streams,layout)
        coupling=ResidentNativeCoupling(session,dycore,geometry,names)
        if chemistry is not None:
            coupling.chemistry=chemistry
            dycore.chemistry=chemistry
        from ..stochastic import resolve_options, StochasticPhysics
        stochastic_path = getattr(arguments, 'physics_namelist', None)
        stochastic_config = resolve_options(getattr(arguments, 'stochastic', None),
            None if stochastic_path is None else Path(stochastic_path).read_text())
        stochastic_restart = None if not resumed else moving.carrier.get('stochastic')
        if stochastic_config.enabled:
            partition = getattr(execution, 'stochastic_partition', None)
            # An LTS rate class forces through its member's one class-aware process (lts_v842, U-H5), never a
            # pattern of its own on its truncated local mesh.
            class_stochastic = getattr(execution, 'stochastic_physics', None)
            coupling.stochastic = (class_stochastic(stochastic_config) if class_stochastic is not None else
                StochasticPhysics(mesh, grid.vertical_grid, stochastic_config)
                if partition is None else
                StochasticPhysics(mesh, grid.vertical_grid, stochastic_config, partition=partition()))
            dycore.stochastic = coupling.stochastic
            coupling.stochastic_dt = dt
            if stochastic_restart is not None:
                coupling.stochastic.process.restore(stochastic_restart)
            elif resumed or stochastic_config.restart_required:
                raise ValueError('stochastic restart is absent from this native carrier')
            receipt['stochastic'] = coupling.stochastic.receipt()
        elif stochastic_restart is not None:
            raise ValueError('checkpoint stochastic physics must remain enabled on resume')

        if str(layout.configs.get('config_IAU_option','off')).strip()!='off':
            coupling.iau=SuiteIau(layout=layout,payload=payload,config=config,mesh=mesh,dycore=dycore,dt=dt)
            receipt['iau']=coupling.iau.receipt()
            print(json.dumps({'iau':receipt['iau']}),flush=True)
        from .diagnostics_v842 import initialize_thermodynamics
        # A restart forms the cold start's base state too (the restart stream's base words), never the
        # perturbation state it carries.
        initialize_thermodynamics(dycore.atmosphere,coupling.physics_cache,
                                  base_only=bool(layout.configs['config_do_restart']))
        resident_reference=dycore.atmosphere.reference
        for name,value in (('rho_base',resident_reference.rho_base),('rtheta_base',resident_reference.rho_theta_base),
                           ('pressure_base',resident_reference.pressure_base),('exner_base',resident_reference.exner_base)):
            if limited_area:
                # Native forms the base state for cells 1..nCells only (atm_init_coupled_diagnostics, fork
                # mpas_atm_time_integration.F:7120-7160) and reads rho_base for the same cells, so the garbage
                # element of each pool keeps its allocation 0.  The regional dycore's own garbage column holds 1.0
                # in three of them (_PaddedReference: a non-trapping denominator) and the base state formed from it.
                # THE BREAKAGE THIS FIXES: the isobaric diagnostics average cell pressure onto vertices through the
                # garbage cell (pressureCp1, mpas_isobaric_diagnostics.F:584-640), so every outermost-ring vertex's
                # isobaric vorticity differed from native (lam7k f000: 772 vertices at 50-250 hPa).
                value=dycore.cp.array(value,copy=True)
                value[...,-1]=0
            session.put('diag',name,value)
        session.put('mesh','zz',dycore.atmosphere.vertical.zz)
        session.put('mesh','dss',dycore.atmosphere.vertical.dss)
        session.put('mesh','edgeNormalVectors',mesh.arrays['edgeNormalVectors'].T)
        # Native initializes the complete vector geometry before physics init;
        # publishing only edge normals leaves the other mesh pool fields zero.
        session.put('mesh','localVerticalUnitVectors',mesh.arrays['localVerticalUnitVectors'].T)
        session.put('mesh','cellTangentPlane',mesh.arrays['cellTangentPlane'].transpose(2,1,0))
        session.put('mesh','coeffs_reconstruct',mesh.arrays['coeffs_reconstruct'].transpose(2,1,0))
        coupling.synchronize(dycore.atmosphere)
        if resumed:
            # Carried physics state into the pools before the physics init, which runs in MPAS's restart mode
            # (config_do_restart), as a native restart reads its restart stream before physics_init.
            receipt['moving_physics']=moving.restore_physics(session,layout,physics_map,mesh)
        stopped_at=steps
        with native_working_directory(out):
            # The ported framework loads no native library: its stages are GPU work issued by one host thread.
            host_label = ('GPU, issued by one host thread (ported framework, no native library)'
                          if library is None else 'GPU plus native CPU')
            with timed_stage(receipt['stage_timings'], 'physics-initialize', timeline='prep.native.physics_initialize', parallelism={'device':host_label,'cards':1,'host_threads':1}):
                session.initialize()
            # A correct-mode checkpoint resume continues the straight run's physics clock (MPAS itimestep and xtime_s);
            # native's restart starts both again (CheckpointRoute.physics_clock_offset).
            physics_step_offset,physics_seconds_offset=0,0.0
            if resumed:
                receipt['physics_alarm_phase']=moving.restore_alarm_phase(session,config)
                clock_offset=getattr(moving,'physics_clock_offset',None)
                if callable(clock_offset):
                    physics_step_offset,physics_seconds_offset=clock_offset(dt)
            if core_initialize is not None:
                core_initialize(session,mesh)
                bind_output_diagnostics=getattr(core_provider,'bind_native_output_diagnostics',None)
                if bind_output_diagnostics is not None:
                    bind_output_diagnostics(session,streams)
            initialize_native_output(native,streams,restarting=layout.configs['config_do_restart'],
                                     reset_stream_alarms=False)
            if layout.configs['config_do_restart']:
                restore_state_time(native)
            from .resident_v842 import resident_requested
            from .route_table import resident_glue_type
            # A same-host rank (samehost_v842) owns a partition of the mesh. Its
            # halo owner supplies the exchanges the closed single-rank mesh never
            # needed: the PBL momentum group and the isobaric pressure halo.
            halo = getattr(execution, 'physics_halo', None)
            resident = (resident_requested(config, layout.dimensions) if halo is None else
                        resident_requested(config, layout.dimensions, partitioned=True))
            diagnostics = None
            # Output diagnostics belong to the dycore, including dry runs
            # without resident physics and runs that load a native physics
            # library (it carries no MPAS diagnostics). The provider's halo
            # operation is a no-op only on the closed single-rank meshes it
            # supports. The row names the provider (route_table). THE BREAKAGE
            # THE LIBRARY CASE FIXES: the RRFS route on its native library
            # refused its diagnostics stream (110 isobaric and convective
            # fields "lack computed authority"; hrrrv5 x1.2562, 2026-10-28).
            closed_mesh = all(layout.dimensions.get(name) == layout.dimensions.get(name+'Solve')
                              for name in ('nCells', 'nEdges', 'nVertices'))
            if resident or closed_mesh or halo is not None:
                from .route_table import output_diagnostics_type
                OutputDiagnostics = output_diagnostics_type(config)
                diagnostics = (OutputDiagnostics(session, dycore) if halo is None else
                               OutputDiagnostics(session, dycore, halo=halo))
            # The writer binds the stream contract of the session's route row (route_table.stream_contract).
            streams_type = getattr(execution, 'streams_type', None) or PhysicsStreams
            output=output_lifetime.enter_context(streams_type(session,streams,out,final_time=stop,
                                                              diagnostics=diagnostics))
            if (getattr(config,'mode',None)=='correct' and moving is None
                    and (halo is None or getattr(output,'gathers_first_call',False))
                    and hasattr(output,'first_frame_from_first_call')):
                # Correct mode completes the initial frame's surface-layer diagnostics from
                # the first physics call (woof.hex.f000_first_call); exact mode is native's.
                # Same-host ranks gather every rank's owned first-call words to rank 0's global
                # frame (samehost_v842 GatheredPhysicsStreams._read_first_call).
                output.first_frame_from_first_call=True
            receipt['missing_fields'] = output.missing
            session.output_streams=output
            restore_continuation=getattr(moving,'restore_continuation',None) if resumed else None
            if callable(restore_continuation):
                # After the physics init and the writer: the scheme SAVE flags and the running accumulators.
                receipt['checkpoint_continuation']=restore_continuation(session=session,diagnostics=diagnostics,dt=dt)
            if chemistry is not None:
                chemistry.bind(dycore=dycore,cache=coupling.physics_cache,coupling=coupling)
                output.extra_history=chemistry.history_fields
                output.extra_history_attrs={'CHEMISTRY':chemistry.history_attribute()}
            receipt['output_metadata']=getattr(output,'metadata_receipt',None)
            initialized_scalars = dycore.cp.asarray(native.get_field('state','scalars')[...,:-1])
            registry_scalars = coupling._registry_rows(dycore.atmosphere.state.scalars)
            if limited_area:
                # The garbage column (the last) keeps the regional dycore's own word.
                registry_scalars[..., :initialized_scalars.shape[-1]] = initialized_scalars
            else:
                registry_scalars[...] = initialized_scalars
            # The initial-scalars upload is spent once copied.  THE BREAKAGE THIS PREVENTS: as a local of this
            # function it stayed allocated for the whole forecast, a scalars-sized block (3 KB per cell on hrrrv5,
            # 0.9 GB on the largest conus3km rank of 8) under every phase (device-memory census, 2026-10-28).
            del initialized_scalars, registry_scalars
            if resident:
                ResidentPhysicsGlue = resident_glue_type(config)
                if halo is None:
                    ResidentPhysicsGlue(session, coupling, mesh)
                else:
                    ResidentPhysicsGlue(session, coupling, mesh, halo=halo)
            if getattr(arguments, 'wind_farms', None) not in (None, '', 'off'):
                from ..wind_farm import attach_native
                wind = attach_native(arguments, session, mesh, output, dt=dt, resident=resident, coupling=coupling, halo=halo)
                receipt['wind_farms'] = json.loads(wind.metadata())
            receipt['resident_physics'] = dict(enabled=resident,
                scope=(f"{config.options.get('config_physics_suite') or 'selected'} suite on a limited-area cull "
                       '(regional dycore)' if limited_area
                       else f"global {config.options.get('config_physics_suite') or 'selected'} suite"),
                schemes={slot: config.selectors[slot] for slot in config.selectors if config.active(slot)},
                field_authority='GPU' if resident else 'native pools')
            lazy_publication = resident or os.environ.get('WOOF_HEX_NATIVE_LAZY_PUBLICATION') == '1'
            publication = receipt['native_publication'] = dict(
                lazy=lazy_publication, published=0, deferred=0)
            # Slot refusals (GWDO, Noah, sea ice, outer winds) are read off the step path; see
            # deferred_checks_v842 (U-MPASPHYS-1). Every check still stops the forecast.
            from .deferred_checks_v842 import DeferredChecks, collecting
            deferred_checks = DeferredChecks(dycore.cp)
            # Single-process file output validates this frame's deferred checks on the writer before encoding.
            # THE BREAKAGE THE MULTI-RANK EXCLUSION PREVENTS: a gathered (samehost) or class (LTS) frame is written
            # by rank/class 0 from every rank's words, but each validator sees only its own rank's checks, so a
            # refused state on another rank could reach that file; those executions keep the forecast-thread wait.
            output.output_validator = deferred_checks.output_validator if execution is None else None
            host_check_wait = execution is not None
            output_lifetime.enter_context(collecting(deferred_checks))
            output_lifetime.push(deferred_checks.prefer_earlier)
            if execution is not None:
                # A rank-process failure aborts these checks (process_ranks_v842.run_process_ranks.stop_all).
                execution.deferred_checks = deferred_checks
                # Bind after initialization and optional resident glue so wrappers
                # see the final coupling callbacks and initialized field authority.
                execution.bind_coupling(coupling)
                execution.initialized(dycore)
            # [hex_nests] trigger looks (forecast --nest-triggers), armed by the same
            # woof.hex.nest_triggers.arm_loop as the ArWen route; None without a table.
            from .nest_look_v842 import arm as arm_nest_look
            nest_look = arm_nest_look(arguments, config, mesh=mesh,
                                      owned_cells=(layout.dimensions['nCellsSolve'] if 'nCellsSolve' in layout.dimensions
                                                   else mesh.n_cells),
                                      start=start, steps=steps, dt=dt, out=out, cp=dycore.cp,
                                      ported_framework=ported_framework, multi_rank=execution is not None,
                                      # The dycore's scalars slot of qv (zero-based), for a cape test.
                                      qv_index=(int(layout.indices['state', 'index_qv']) - 1
                                                if ('state', 'index_qv') in layout.indices else None))
            session.nest_look = nest_look
            balance=balance_series_sampler(arguments,layout,names,grid.vertical_grid,dycore,out)
            records = StepRecordWriter()
            output_lifetime.callback(records.close)
            receipt['step_records'] = records.summary()
            calls_path = (out/'physics-calls.jsonl' if execution is None
                          else execution.calls_path(out))
            # The step whose end-of-step publication synchronised the coupling pools from the
            # same atmosphere the next outer-physics stage would synchronise again (G6).
            synchronized_after = None
            columns = None
            for step in range(steps+1):
                if step and nest_look is not None:
                    nest_look.begin(step)
                if step:
                    if step == 1:
                        time_ledger.end_if_open("forecast.construct")
                        time_ledger.emit("forecast.integrate", "begin", attrs=card_attrs)
                        time_ledger.emit("forecast.first_step", "begin", attrs=card_attrs)
                    with measure_stochastic_step(coupling, 'mpas-outer-step'):
                        if execution is not None:
                            execution.before_step(step)
                        coupling.sources=None
                        with timed_stage(receipt['stage_timings'], 'outer-physics', timeline='post.forecast_process.native.outer_physics', parallelism={'device':host_label,'cards':1,'host_threads':1}) as outer_row:
                            # THE BREAKAGE THIS PREVENTS (G6): synchronize ran three times a step; the end-of-step
                            # publication already copied this exact atmosphere into the pools on one card, and
                            # nothing between the two touches it, so the copy and its kernels ran twice.
                            if execution is None and synchronized_after == step-1:
                                outer_row['synchronize'] = 'reused end-of-step publication'
                            else:
                                coupling.synchronize(dycore.atmosphere)
                            synchronized_after = None
                            if diagnostics is not None:
                                # The previous step's running maxima (G1), on the state the pools
                                # now hold: this synchronize's, or the reused publication's.
                                diagnostics.flush()
                            # Device memory stays with the per-slot brackets (U-MPASPHYS-1's keep rule).
                            # A bracket per physics half was measured slower on a 24 GB card (G6, an
                            # RTX 4090, zoom: outer physics 0.82 -> 1.91 s/step): with no slot boundary
                            # inside the half, the pool grew to the half's peak and CuPy freed and retried.
                            # Keyed by the physics call's itimestep (tempo_v303.radii_due), the continued one on a resume.
                            session.pool_readers_after_step = (step+physics_step_offset, physics_pool_readers_due(
                                output, step=step, steps=steps, dt=dt, moving=moving, execution=execution))
                            session.begin_step(step+physics_step_offset,physics_seconds_offset+(step-1)*dt)
                            commit_outer_scalar_groups(session, dycore.atmosphere, dycore.cp)
                        coupling.itimestep=step
                        acoustic_output = {}
                        if diagnostics is not None:
                            callback = diagnostics.prepare_step(step=step, dt=dt)
                            if callback is not None:
                                acoustic_output['acoustic_output_callback'] = callback
                        if les is not None:
                            # les_surface='varying' reads the physics fluxes
                            # (mpas_atm_dissipation_models.F:919,1533-1534).
                            for name in ('ustm','hfx','qfx'):
                                if ('diag_physics',name) in native._fields:
                                    les.set(name,session.field('diag_physics',name))
                        # Registry's qv group exists in the dry suite too. Native
                        # always computes the moist coefficients from that active
                        # group (mpas_atm_time_integration.F:2125-2152).
                        with timed_stage(receipt['stage_timings'], 'dycore-and-post-transport', timeline='post.forecast_process.native.dycore', parallelism={'device':host_label if library is None else 'GPU plus native CPU callbacks','cards':1,'host_threads':1}):
                            result=dycore._step_device_v841(physics_stage_callback=coupling.stage,
                                physics_post_transport_callback=coupling.post_transport,
                                physics_mass_loading_count=layout.indices['state','moist_end'],
                                **acoustic_output)
                        dycore.atmosphere=result.atmosphere
                        if chemistry is not None:
                            with timed_stage(receipt['stage_timings'], 'chemistry', timeline='post.forecast_process.native.chemistry',
                                             parallelism={'device':'GPU','cards':1,'host_threads':1}):
                                chemistry.step(session=session,dt=dt)
                        if execution is not None:
                            # Complete boundary exchange before committing the native clock.
                            execution.after_step(dycore)
                        # Resident state is published at level1; compute the scalar
                        # clock word using the same native RKIND operation on that level.
                        native.update_state_time(dt,source_level=1,target_level=1)
                        native.advance_clock()
                        if nest_look is not None:
                            nest_look.after_step(step, dycore)
                        deferred_checks.flush(step)
                    if step == 1:
                        time_ledger.emit("forecast.first_step", "end", attrs=card_attrs)
                        with timed_stage(receipt['stage_timings'], 'f000-first-call', timeline='forecast.output.f000_first_call',
                                parallelism={'device':'GPU copy','cards':1,'host_threads':1}):
                            complete_first_frame=getattr(output,'complete_first_frame',None)
                            if complete_first_frame is not None:
                                deferred_checks.before_output()
                                complete_first_frame()
                    receipt['steps_completed']=step
                    if getattr(coupling, 'stochastic', None) is not None:
                        receipt['stochastic'] = coupling.stochastic.receipt()
                    if coupling.iau is not None:
                        receipt['iau']=coupling.iau.receipt()
                    if les is not None:
                        receipt['les_execution']=les.receipt(step)
                    records.append(calls_path, json.dumps(dict(step=step,calls=session.driver.calls))+'\n')
                if balance is not None:
                    # Step 0 is the initial state; then every step, as the ArWen route samples its macro steps.
                    balance.sample(balance.view, float(step)*float(dt), dycore.cp)
                    receipt['balance_series']=dict(path=str(balance.path),rows=len(balance.rows))
                moving_pools = moving_publication_due(moving, step=step, steps=steps, dt=dt)
                moving_columns = moving_pools or moving_columns_due(moving, step=step, steps=steps, dt=dt)
                publish = native_publication_required(native, streams, step=step, steps=steps,
                    moving=moving if moving_columns else None, lazy=lazy_publication,output=output,
                    when=output.current_time(step=step,dt=dt))
                # Always evaluate this hook: it may contain a rank collective,
                # including when the native stream schedule already needs output.
                consumer_publication = (execution is not None and
                    execution.publication_required(step, steps, dt))
                if consumer_publication:
                    publish = True
                with timed_stage(receipt['stage_timings'], 'publish-and-streams', timeline='forecast.output.history.capture',
                        parallelism={'device':'GPU capture plus background Rust I/O' if publish
                                     else 'none (no capture due; maxima deferred to the next synchronize)',
                                     'cards':int(publish),'host_threads':1}) as timing:
                    timing['state_published'] = publish
                    columns = None
                    if publish:
                        # File output validates the captured checks on its writer
                        # before encoding. Host consumers still need completed checks.
                        if host_check_wait or consumer_publication or moving_pools or moving_columns:
                            deferred_checks.before_output()
                        # Moving diagnostics explicitly consume psfc_p from a
                        # complete preparation at this publication boundary.
                        if moving_columns:
                            columns=coupling.synchronize(dycore.atmosphere, prepared_columns=True)
                        else:
                            columns=coupling.synchronize(dycore.atmosphere)
                            synchronized_after = step
                        # Synchronization can add a wind-reconstruction refusal.
                        # The writer snapshots these too; host consumers wait here.
                        if host_check_wait or consumer_publication or moving_pools or moving_columns:
                            deferred_checks.before_output()
                        # JEDI and moving carriers still consume host pool
                        # views, a moving carrier only at a frame or its last
                        # step (moving_publication_due). Ordinary stream capture
                        # reads device buffers directly and never needs this copy.
                        if consumer_publication or moving_pools:
                            session.publish()
                            timing['pools_downloaded'] = True
                        write_native_output(native,streams,step=step,receipt=receipt,session=session,
                                            time_seconds=step*dt,output=output,dt=dt)
                    elif diagnostics is not None:
                        diagnostics.defer(step=step, when=output.current_time(step=step,dt=dt))
                    publication['published' if publish else 'deferred'] += 1
                if execution is not None:
                    execution.output(step=step,steps=steps,dt=dt,session=session,layout=layout,names=names,out=out)
                if session.surface_service is not None:
                    receipt['surface_service'] = session.surface_service.receipt()
                    if surface_record_reader is not None and surface_record_reader.last_record is not None:
                        record = surface_record_reader.last_record
                        reads = receipt['surface_input']['reads']
                        if not reads or reads[-1]['record'] != record:
                            reads.append(dict(record=record, timestamp=surface_record_reader.times[record].strftime('%Y-%m-%d_%H:%M:%S'),
                                              model_clock=session.arguments.calendar['current_date'], step=step,
                                              callback_execution=receipt['surface_service']['operations']['physics_update_sst']))
                if resident:
                    receipt['resident_physics']['transfers'] = dict(session.device_pools.transfers)
                stop=False
                if moving is not None:
                    stop=moving.after_step(step,steps=steps,dt=dt,columns=columns,session=session,dycore=dycore,
                                           mesh=mesh,vertical=grid.vertical_grid,names=names,layout=layout,out=out)
                    receipt.update(moving.receipt())
                receipt['step_records'] = records.summary()
                records.receipt(receipt_path, receipt, force=stop or step <= 1)
                if stop:
                    # The controller asked to stop at this frame: the carrier below seals this instant.
                    stopped_at=step
                    break
            deferred_checks.finish()
            receipt['deferred_checks'] = deferred_checks.receipt()
            time_ledger.end_if_open("forecast.integrate")
            time_ledger.emit("post.forecast", "begin", attrs=card_attrs)
            if nest_look is not None:
                receipt['nest_triggers']=nest_look.finish(out)
            if moving is not None:
                moving.finish(step=stopped_at,session=session,dycore=dycore,mesh=mesh,vertical=grid.vertical_grid,
                              names=names,layout=layout,dt=dt)
                receipt.update(moving.receipt())
            output.close()
            if chemistry is not None:
                ledger=chemistry.write_ledger(out)
                receipt['chemistry']={k:ledger[k] for k in ('schema','config_sha256','committed_steps',
                                      'mass_closure_kg','final_mass_kg','emitted_kg','route')}
            session.driver.finalize(session)
    if coupling.iau is not None:
        receipt['iau']=coupling.iau.receipt()
        if not receipt['iau']['admission_witness']['passed']:
            raise RuntimeError(f"IAU source admission differs from the analysis increments: {receipt['iau']['admission_witness']}")
        # Correct mode admits exactly one increment over a window the run covers (IN3); exact mode keeps
        # native's NINT step count, which IN3 names.
        if (coupling.iau.config.correct and coupling.iau.config.config_IAU_option=='on'
                and steps*dt>=coupling.iau.config.config_IAU_window_length_s
                and abs(receipt['iau']['fraction_admitted']-1.0)>1e-5):
            raise RuntimeError(f"IAU window not fully applied: {receipt['iau']}")
    if execution is not None:
        receipt['rank_execution']=execution.receipt()
    if limited_area:
        receipt['limited_area']=limited_area_receipt(arguments, dycore, coupling)
    if fullstep_shadow is not None:
        receipt['fullstep_lts_treatment']=fullstep_shadow.receipt()
    # A ported framework records the lookup files its scheme initialization read (RRFS TEMPO).
    receipt['table_sha256'].update(getattr(backend,'lookup_sha256',None) or {})
    receipt['completed']=True
    receipt['physics_compile_manifest']=coupling.physics_cache.compile_manifest()
    from ..device_memory_release import process_release
    receipt['device_memory_release']=process_release().receipt()
    receipt_path.write_text(json.dumps(receipt,indent=2,default=str)+'\n')
    print(json.dumps(receipt,default=str))
    return 0
