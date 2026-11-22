"""Fixed-clock real-input member forecast with aggregate-only Rust output.

The input authority and terrain clock are the ordinary prepared WRF door's.
One common scalar restore supplies the initialization words; no independent
member forecast driver is retained or advanced by this executor.
"""
from __future__ import annotations

from dataclasses import asdict
import math
import numpy as np


class PreparedRealEnsemble:
    def __init__(self, inputs, *, members, member_seeds, member_initializer=None):
        import cupy as cp
        from woof.core.clock import resolve_clock
        from woof.core.device_inventory import state_array_shapes
        from woof.ingest.wrfinput import restore_domain_state
        from woof.ensemble.batch_state import (BatchedDomainState, PreparedHostMember,
                                                 SHARED_STATE_CANDIDATES)
        from woof.ensemble.batch_moist_dycore import workspace_specs, required_scratch_slots, prepare_moist_step
        from woof.ensemble.batch_boundaries import MemberBoundaryTables
        from woof.ensemble.batch_physics_init import initialize_wrfinput_member_physics
        from woof.core.cam_ozone import cam_ozone_setup
        from woof.case_data import trace_gas_overrides_from_config
        from woof.runtime import declared_constant_glw
        if len(inputs.domains) != 1 or members < 2:
            raise ValueError('this retained real executor owns one root domain and at least two members')
        exp = inputs.experiment
        domain, bundle, grid = exp.root, inputs.domains[0], inputs.grids[0]
        cfg = domain.run
        authority = resolve_clock(exp,lbc_interval_s=float(inputs.boundary_interval_seconds))
        clock = authority.domain_clock(domain.grid_id)
        scalar = restore_domain_state(bundle.restored,cfg)
        names = state_array_shapes(cfg)
        arrays = {name: getattr(scalar,name).get() for name in names}
        extras = workspace_specs(cfg)
        arrays.update({spec.name:np.zeros(spec.shape,spec.dtype) for spec in extras})
        controls = {'physics','lateral_boundaries','_scratch','_scratch_arena','_host_setup_state','_phb_host'}
        scalars = {name:value for name,value in vars(scalar).items() if name not in names and name not in controls}
        snapshot = dict(ticks=clock.ticks,step_ticks=clock.step_ticks,tick_den=clock.tick_den,
                        run_ticks=clock.run_ticks,step_count=clock.step_count,
                        dt_fp32=clock.dt_fp32,dtbc_fp32=clock.dtbc_fp32)
        prepared = PreparedHostMember(cfg,arrays,scalars,snapshot,
            scratch={name:value.get() for name,value in scalar._scratch.items()},phb_host=scalar._phb_host)
        del scalar
        cp.cuda.get_current_stream().synchronize()
        free = lambda: int(cp.cuda.runtime.memGetInfo()[0]) - (512 << 20)
        batch = BatchedDomainState.from_prepared((prepared,)*members,array_module=cp,
            available_bytes=free(),shared_fields=tuple(sorted(SHARED_STATE_CANDIDATES & names.keys())),
            extra_specs=extras,scratch_slots=required_scratch_slots(cfg))
        del arrays,prepared
        tables = MemberBoundaryTables.from_prepared((inputs.boundaries,)*members,cfg,
            array_module=cp,available_bytes=free())
        cam = cam_ozone_setup(exp=exp,dc=domain,grid=grid)
        trace = trace_gas_overrides_from_config(inputs.experiment_config,
                            expected_sha256=inputs.authority_sha256['experiment_config'])
        physics = initialize_wrfinput_member_physics(batch,bundle.restored,
            start_time=exp.start_time,landuse=bundle.landuse,cam_ozone=cam,
            column_chunk=exp.column_chunk,trace_gas_overrides=trace,
            constant_glw_wm2=declared_constant_glw(exp),fractional_seaice=bundle.fractional_seaice,
            available_bytes=free(),hydro_parent_geometry=grid)
        initializer_receipt = None
        if member_initializer is not None:
            initializer_receipt = member_initializer(state=batch,cfg=cfg,
                member_indices=tuple(range(members)),seeds=member_seeds,phase='after_physics_before_step')
        self.inputs,self.cfg,self.batch,self.physics,self.tables = inputs,cfg,batch,physics,tables
        self.clock,self.authority = clock,authority
        self.advance = prepare_moist_step(batch,physics_adapter=physics,tables=tables,boundary_clock=clock)
        self.initializer_receipt = initializer_receipt
        from woof.ensemble.batch_health import PreparedBatchStability,PreparedBatchStateHealth
        self.stability = PreparedBatchStability(batch,cfg,boundary_width=cfg.spec_bdy_width,available_bytes=free())
        self.validator = PreparedBatchStateHealth(batch,physics_driver=physics.driver,tables=tables,available_bytes=free())
        self.initial_health = self.validator.require_healthy(phase='initialized')
        self.final_health = None
        self.final_stability = None

    def step(self):
        exchange = getattr(self, '_device_exchange', None)
        if exchange is not None:
            exchange.require_forecast_ready()
        clock = self.clock
        if clock.at_stop_time:
            raise ValueError('ensemble clock reached its declared stop time')
        if clock.lbc_reset_due():
            clock.mark_force()
        clock.prepare_step()
        self.batch.elapsed_seconds = float(clock.elapsed_seconds_fp32)
        self.advance(refl_10cm_due=clock.history_rings_within_step())
        for member,report in enumerate(self.stability()):
            if report['nan']:
                raise RuntimeError(f'member {member} integration produced a non-finite state at step {clock.step_count+1}')
            if report['cfl'] is not None and not math.isfinite(float(report['cfl'])):
                raise RuntimeError(f'member {member} integration produced a non-finite vertical Courant number at step {clock.step_count+1}')
        clock.advance()
        self.batch.elapsed_seconds = clock.elapsed_seconds
        self.batch.clock['dtbc_fp32'] = clock.dtbc_fp32

    def device_exchange(self, *, writable_fields):
        """Borrow this retained batch between steps, keeping its prepared graph."""
        from woof.ensemble.device_exchange import cuda_exchange
        exchange = getattr(self, '_device_exchange', None)
        if exchange is None:
            exchange = cuda_exchange(self.batch, writable_fields=writable_fields)
            self._device_exchange = exchange
        elif exchange.writable != frozenset(writable_fields):
            raise ValueError('cannot change writable field ownership on a live exchange')
        return exchange

    def receipt(self):
        import cupy as cp
        return {'members':self.batch.members,'dt_seconds':self.cfg.dt,
                'time_step_sound':self.cfg.time_step_sound,'steps':self.clock.step_count,
                'state_plan_bytes':self.batch.plan.required_bytes(self.batch.members),
                'boundary_plan_bytes':self.tables.plan.required_bytes(self.batch.members),
                'physics':self.physics.receipt,'member_initializer':self.initializer_receipt,
                'health':{'stability':self.stability.receipt,'full_state':self.validator.receipt,
                          'initial':[asdict(report) for report in self.initial_health],
                          'final':None if self.final_health is None else [asdict(report) for report in self.final_health],
                          'final_stability':self.final_stability},
                'pool_live_bytes':cp.get_default_memory_pool().used_bytes(),
                'pool_reserved_bytes':cp.get_default_memory_pool().total_bytes(),
                'scalar_forecast_members_advanced':0,
                # As one run's record states them (lateral_bc.
                # ensemble_hydrometeor_ring_receipt); None for vapour only.
                'lateral_boundary_hydrometeors':self.tables.hydrometeor_receipt(
                    tuple(range(self.batch.members)),through_seconds=float(self.clock.elapsed_seconds))}

    def consume_output_diagnostics(self):
        """Complete the original one-frame reflectivity handoff on output."""
        if self.clock.ticks == self.clock.spec.start_ticks:
            return None
        from woof.core.refl import consume_refl_10cm
        return consume_refl_10cm(self.physics.state)
