"""WRF digital-filter startup, with signed weather integration on CUDA.

The phase boundaries follow share/dfi.F:75-466. Python constructs each
finite graph at setup; the Rust driver submits it once. No weather fields
cross to the host between solves.
"""
from __future__ import annotations

from dataclasses import replace
from fractions import Fraction


def _interval(start, stop, dt):
    from woof.core.dfi_graph import PassInterval
    step = Fraction(str(dt))
    return PassInterval(start*step.denominator, stop*step.denominator,
                        step.numerator, step.denominator)


def _prepare_model_graph(state, cfg):
    """Upload immutable scheme inputs without advancing the model.

    Prepared-cache starts need not run Noah's input-file cold initializer,
    so its parameter cache can still be empty at the first DFI solve.
    That upload must precede capture, like the microphysics coefficients.
    """
    from woof.core.wsm6 import prepare_model_graph
    prepare_model_graph(cfg)
    if cfg.mp_physics == 8:
        from woof.thompson_generations import CLASSIC_GENERATIONS
        generation = CLASSIC_GENERATIONS[getattr(cfg, 'thompson_version', 'wrf_461')]
        if generation.backend == 'column':
            import cupy as cp
            from woof.core.thompson_column import table_identity, table_root, tables
            table_identity(cfg)
            tables(str(table_root(cfg)), cp.cuda.Device().id)
        else:
            from woof.physics_compat import thompson_table_root
            from woof.core.thompson_runtime import load_classic_device_tables
            load_classic_device_tables(thompson_table_root())
    if cfg.sf_surface_physics == 2:
        from woof.core.noah import _device_tables
        from woof.ingest.soil import NOAH_LAYER_THICKNESS_M
        _device_tables(state.physics.noah_params, NOAH_LAYER_THICKNESS_M)


def initialize_tree(model, *, start_time, output_directory):
    """Apply DFI before the forecast executor opens its initial history."""
    root = model.root
    cfg = root.cfg.run
    if root.children:
        return initialize_nested_tree(model,start_time=start_time,
                                      output_directory=output_directory)
    from woof.core.dfi_output import DfiOutput
    output = DfiOutput(model,start_time,output_directory)
    epoch,receipts = initialize_domain(root.state,cfg,selector=cfg.dfi_opt,
        backward_stop=cfg.dfi_bckstop_seconds,forward_stop=cfg.dfi_fwdstop_seconds,
        cutoff=cfg.dfi_cutoff_seconds,filter_selector=cfg.dfi_nfilter,
        save_hydrometeors=bool(cfg.dfi_savehydmeteors),
        time_step=cfg.time_step_dfi,moist_adv_opt=cfg.moist_adv_dfi_opt,
        history=lambda elapsed,phase:output.capture(root,elapsed,phase),
        pass_complete=output.flush)
    output.initialized(epoch)
    clock = root.clock
    clock.ticks = int(epoch*clock.tick_den)
    clock.dfi_start_ticks = clock.ticks
    clock.step_count = 0
    clock.mark_force()
    if epoch:
        import numpy as np
        clock.dtbc_fp32 = np.float32(epoch)
    from woof.prepared_domain_tree_forecast import _atomic_json
    from pathlib import Path
    _atomic_json(Path(output_directory)/'dfi-initialization.json',dict(
        selector=cfg.dfi_opt,filter=cfg.dfi_nfilter,forecast_start_seconds=epoch,
        passes=list(receipts)))
    return receipts


def initialize_domain(state, cfg, *, selector, backward_stop, forward_stop,
                      cutoff=3600., filter_selector=7, save_hydrometeors=False,
                      time_step=-1., moist_adv_opt=0, stream=None,
                      history=None, pass_complete=None):
    """Initialize an already restored domain; return its forecast epoch.

    ``backward_stop`` and ``forward_stop`` are signed seconds from the
    original analysis. The caller retains the original UTC reference and
    stops the forecast at the original requested end time.
    """
    import cupy as cp
    from woof.core.dfi_model import ModelDfiState, backward_configuration
    from woof.core.dfi_graph import DfiBoundaryClock, capture_model_pass
    from woof.core.dfi_runtime import lifecycle, weights
    from woof.core.radiation_composition import radiation_adapters
    from woof.ingest.lateral_bc import prepare_lateral_boundary_graph

    plan, forecast_start = lifecycle(selector, backward_stop, forward_stop)
    if not plan:
        return 0, ()
    factory = getattr(state, '_dfi_physics_initialization', None)
    if factory is None:
        raise ValueError('DFI requires its retained native physics initialization inputs')
    if cfg.nested:
        raise ValueError('a nested DFI domain must be initialized with its coupled tree')
    if stream is None:
        stream = cp.cuda.get_current_stream()
    if stream.ptr == 0:
        stream.synchronize()
        stream = cp.cuda.Stream(non_blocking=True)
    binding = ModelDfiState(state,cfg,save_hydrometeors=save_hydrometeors)
    resident = getattr(state, '_lateral_boundary_device', None)
    original_clock = None if resident is None else resident.clock
    positive_dt = float(time_step if time_step > 0 else cfg.dt)
    resources, receipts = [], []
    with stream:
        for phase in plan:
            dt = -positive_dt if phase.backward else positive_dt
            interval = _interval(phase.start,phase.stop,dt)
            phase_cfg = (backward_configuration(cfg,dt=dt,moist_adv_opt=moist_adv_opt)
                         if phase.backward else replace(cfg,dt=dt,clock_dt=dt,
                             use_adaptive_time_step=False,sf_ocean_physics=0))
            state.elapsed_seconds = float(phase.start)
            state.domain_start_offset = float(phase.start)
            state._dfi_boundary_start = min(0,backward_stop)
            # start_domain precedes the backward RH snapshot. Forward
            # clear precedes phy_init, which restores the live radii.
            if phase.backward:
                factory.restore_base(phase_cfg)
                binding.export()
                binding.native.save(backward=True)
                state.physics = None
            elif selector == 1:
                binding.export()
                binding.native.save(backward=False)
            binding.native.clear()
            if not phase.backward:
                factory.reinitialize(phase_cfg,epoch=phase.start)
                binding.rebind_physics()
                binding.export()
                binding.native.restore_hydrometeors()
                binding.import_filtered()
                for adapter in radiation_adapters(state.physics.radiation_callable):
                    prepare = getattr(adapter,'prepare_graph_times',None)
                    if prepare is not None:
                        prepare(phase_cfg,interval.times(),model_levels=cfg.nz,
                                p_top=state.p_top)
            clock = DfiBoundaryClock(interval,dt)
            if resident is not None:
                resident.clock = clock
            _prepare_model_graph(state,phase_cfg)
            prepare_lateral_boundary_graph(state,phase_cfg)
            coefficients = cp.asarray(weights(abs(phase.stop-phase.start),dt,
                                               cutoff,filter_selector))
            from woof.core.dfi_timing import PassTiming
            timing = PassTiming(stream,interval.steps)

            def before(index):
                if history is not None:
                    history(state.elapsed_seconds,phase)
                timing.record(timing.events[index][0])
                binding.accumulate(coefficients)
                timing.record(timing.events[index][1])
                from woof.core.dfi_graph import _prepare_external_clock
                clock.ticks = interval.start_ticks+index*interval.step_ticks
                _prepare_external_clock(state,clock)
                clock.prepare(index)

            def after(index):
                timing.record(timing.events[index][2])
                binding.native.advance()

            def finish():
                if history is not None:
                    history(phase.stop,phase)
                timing.record(timing.tail[0])
                binding.accumulate(coefficients)
                if phase.reset:
                    binding.native.reset(backward=bool(phase.backward))
                    binding.import_filtered()
                timing.record(timing.tail[1])

            compiled = capture_model_pass(state,phase_cfg,interval,stream=stream,
                before_step=before,after_step=after,at_end=finish)
            compiled.submit()
            compiled.finish()
            if pass_complete is not None:
                pass_complete()
            error = int(binding.native.error.get()[0])
            if error:
                raise FloatingPointError(f'DFI device filter failed ({error})')
            receipts.append(dict(backward=bool(phase.backward),start=phase.start,
                stop=phase.stop,dt=dt,steps=interval.steps,reset=bool(phase.reset),
                coefficient_count=coefficients.size,pool_bytes=compiled.pool.total_bytes(),
                timing=timing.result()))
            # Model-owned held arrays may have been allocated while
            # capturing. Keep their allocator alive after graph disposal.
            resources.append((compiled.pool,coefficients))
            pool = compiled.pool
            del compiled
            # The submitted graph is complete and destroyed. Return its
            # unused transient blocks before allocating the next physics
            # driver; retained model arrays keep their live blocks owned.
            pool.free_all_blocks()
        # Release the DFI state; the pass closures above share this cell.
        binding = None
        state.elapsed_seconds = float(forecast_start)
        factory.reinitialize(cfg,epoch=forecast_start)
        if resident is not None:
            resident.clock = original_clock
        del state._dfi_boundary_start
        if hasattr(state,'_dfi_constant_tables'):
            del state._dfi_constant_tables
        del state._dfi_diagnostics
        state._dfi_graph_resources = resources
        state._dfi_receipts = receipts
        state._dfi_filtered = True
    stream.synchronize()  # Completed forecast phy_init, before another stream consumes it.
    return forecast_start, tuple(receipts)


def initialize_nested_tree(model, *, start_time, output_directory):
    """Run the native recursive DFI walk with ordinary resident nest forcing."""
    import cupy as cp
    import numpy as np
    from math import lcm
    from pathlib import Path
    from woof.core.dfi_graph import (DfiBoundaryClock, PassInterval,
        capture_tree_pass)
    from woof.core.dfi_model import ModelDfiState, backward_configuration
    from woof.core.dfi_runtime import lifecycle, weights
    from woof.core.radiation_composition import radiation_adapters
    from woof.core.nest import nest_field_kinds, _APPLICATION_NAME
    from woof.ingest.lateral_bc import (attach_nest_boundaries,
        prepare_lateral_boundary_graph)
    from woof.prepared_domain_tree_forecast import _atomic_json
    from woof.core.dfi_output import DfiOutput

    nodes = tuple(model.walk_parent_first())
    cfg = model.root.cfg.run
    output = DfiOutput(model,start_time,output_directory)
    plan, forecast_start = lifecycle(cfg.dfi_opt,cfg.dfi_bckstop_seconds,
                                     cfg.dfi_fwdstop_seconds)
    originals = {n.cfg.grid_id:n.clock for n in nodes}
    steps = {}
    for node in nodes:
        gid = node.cfg.grid_id
        if node.parent is None:
            steps[gid] = (Fraction(cfg.time_step_dfi) if cfg.time_step_dfi > 0 else
                Fraction(node.clock.step_ticks,node.clock.tick_den))
        else:
            steps[gid] = steps[node.parent.cfg.grid_id]/node.cfg.parent_time_step_ratio
    den = lcm(*(dt.denominator for dt in steps.values()))
    bindings = {n.cfg.grid_id:ModelDfiState(n.state,n.cfg.run,
        save_hydrometeors=bool(cfg.dfi_savehydmeteors)) for n in nodes}
    factories = {n.cfg.grid_id:n.state._dfi_physics_initialization for n in nodes}
    stream = cp.cuda.get_current_stream()
    if stream.ptr == 0:
        stream.synchronize()
        stream = cp.cuda.Stream(non_blocking=True)
    resources, receipts = [], []
    root_resident = getattr(model.root.state,'_lateral_boundary_device',None)
    old_root_clock = None if root_resident is None else root_resident.clock
    with stream:
        for phase in plan:
            configurations, coefficients = {}, {}
            for node in nodes:
                gid, state = node.cfg.grid_id,node.state
                dt = steps[gid]*(-1 if phase.backward else 1)
                interval = PassInterval(phase.start*den,phase.stop*den,int(dt*den),den)
                # Kernel dt follows WRF's chained REAL division, whereas
                # the recursive stop time follows the exact tick lattice.
                fpdt = (np.float32(float(dt)) if node.parent is None else
                    np.float32(node.parent.clock.dt_fp32/
                               np.float32(node.cfg.parent_time_step_ratio)))
                clock = DfiBoundaryClock(interval,fpdt)
                clock._dfi_backward = bool(phase.backward)
                node.clock = clock
                native_cfg = node.cfg.run
                active = (backward_configuration(native_cfg,dt=float(fpdt),
                    moist_adv_opt=native_cfg.moist_adv_dfi_opt) if phase.backward else
                    replace(native_cfg,dt=float(fpdt),clock_dt=float(fpdt),
                            use_adaptive_time_step=False,sf_ocean_physics=0))
                configurations[gid] = active
                state.elapsed_seconds = float(phase.start)
                state.domain_start_offset = float(phase.start)
                state._dfi_boundary_start = min(0,cfg.dfi_bckstop_seconds)
                binding, factory = bindings[gid],factories[gid]
                if phase.backward:
                    factory.restore_base(active)
                    binding.export()
                    binding.native.save(backward=True)
                    state.physics = None
                elif cfg.dfi_opt == 1:
                    binding.export()
                    binding.native.save(backward=False)
                binding.native.clear()
                if not phase.backward:
                    factory.reinitialize(active,epoch=phase.start)
                    binding.rebind_physics()
                    binding.export()
                    binding.native.restore_hydrometeors()
                    binding.import_filtered()
                    for adapter in radiation_adapters(state.physics.radiation_callable):
                        prepare = getattr(adapter,'prepare_graph_times',None)
                        if prepare is not None:
                            prepare(active,interval.times(),model_levels=native_cfg.nz,
                                    p_top=state.p_top)
                if node.parent is None:
                    if root_resident is not None:
                        root_resident.clock = clock
                else:
                    coupler = node.coupler
                    coupler._bind_geometry()
                    # Allocate/attach metadata only. The captured FORCE
                    # overwrites these tables before any child solves.
                    fields = {_APPLICATION_NAME.get(k,k):coupler._rolling_out(k)
                              for k in nest_field_kinds(native_cfg)}
                    attach_nest_boundaries(state,fields,clock=clock,
                        spec_bdy_width=native_cfg.spec_bdy_width,
                        spec_zone=native_cfg.spec_zone,relax_zone=native_cfg.relax_zone)
                _prepare_model_graph(state,active)
                prepare_lateral_boundary_graph(state,active)
                coefficients[gid] = cp.asarray(weights(abs(phase.stop-phase.start),
                    float(fpdt),cfg.dfi_cutoff_seconds,cfg.dfi_nfilter))
            interval = model.root.clock.interval
            from woof.core.dfi_timing import TreePassTiming
            timing = TreePassTiming(stream,nodes)
            compiled = capture_tree_pass(model,configurations,bindings,coefficients,
                interval,stream=stream,reset=bool(phase.reset),backward=bool(phase.backward),
                history=lambda node:output.capture(node,node.clock.elapsed_seconds,phase),
                timing=timing)
            compiled.submit()
            compiled.finish()
            output.flush()
            for gid,binding in bindings.items():
                error = int(binding.native.error.get()[0])
                if error:
                    raise FloatingPointError(f'DFI d{gid:02d} failed ({error})')
            receipts.append(dict(backward=bool(phase.backward),start=phase.start,
                stop=phase.stop,reset=bool(phase.reset),pool_bytes=compiled.pool.total_bytes(),
                domains={str(n.cfg.grid_id):dict(steps=n.clock.step_count,
                    dt=float(n.clock.dt_fp32),coefficients=coefficients[n.cfg.grid_id].size)
                    for n in nodes},timing=timing.result()))
            resources.append((compiled.pool,coefficients))
            pool = compiled.pool
            del compiled
            pool.free_all_blocks()
        bindings.clear()
        for node in nodes:
            gid,state = node.cfg.grid_id,node.state
            state.elapsed_seconds = float(forecast_start)
            factories[gid].reinitialize(node.cfg.run,epoch=forecast_start)
            node.clock = originals[gid]
            node.clock.ticks = forecast_start*node.clock.tick_den
            node.clock.dfi_start_ticks = node.clock.ticks
            node.clock.step_count = 0
            node.clock.mark_force()
            if node.parent is None and root_resident is not None:
                root_resident.clock = old_root_clock
                node.clock.dtbc_fp32 = np.float32(forecast_start)
            if node.parent is not None:
                resident = getattr(state,'_lateral_boundary_device',None)
                if resident is not None:
                    resident.clock = node.clock
            if node.coupler is not None:
                node.coupler.invalidate()
                node.coupler.reset_restart_observation()
            del state._dfi_boundary_start
            if hasattr(state,'_dfi_constant_tables'):
                del state._dfi_constant_tables
            del state._dfi_diagnostics
            state._dfi_filtered = True
        model.root.state._dfi_graph_resources = resources
    stream.synchronize()
    output.initialized(forecast_start)
    _atomic_json(Path(output_directory)/'dfi-initialization.json',dict(
        selector=cfg.dfi_opt,filter=cfg.dfi_nfilter,forecast_start_seconds=forecast_start,
        passes=receipts,feedback_during_dfi=0,
        forecast_feedback={str(n.cfg.grid_id):n.coupler.feedback for n in nodes if n.coupler}))
    return tuple(receipts)
