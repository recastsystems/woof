"""Prepared moist RK graph with fixed mixing and admitted physics/forcing.

Member-local arithmetic follows the original fixed-step stage schedule.
Initialized column physics and common-clock boundary tables are explicit
owners supplied by the real-input executor.
"""
from __future__ import annotations

import numpy as np

from woof.ensemble.batch_state import BatchedDomainState, BatchStateUnsupported


def workspace_specs(cfg):
    from woof.ensemble import batch_glue, batch_moist
    from woof.ensemble.batch_storage import BatchArraySpec
    from woof.core.physics import physics_enabled
    result = batch_glue.workspace_specs(cfg) + batch_moist.workspace_specs(cfg)
    if cfg.km_opt == 4 or cfg.diff_6th_opt > 0:
        from woof.ensemble.batch_moist_mixing import workspace_specs as mixing_specs
        result += mixing_specs(cfg, has_msf=True)
    if physics_enabled(cfg):
        result += tuple(BatchArraySpec('batch_phys_'+name, shape, 'member') for name, shape in (
            ('ru', (cfg.nz,cfg.ny,cfg.nx+1)), ('rv',(cfg.nz,cfg.ny+1,cfg.nx)),
            ('rw',(cfg.nz+1,cfg.ny,cfg.nx)),
            *((name,(cfg.nz,cfg.ny,cfg.nx)) for name in ('rtheta','rqv','rqc','rqr','rqi','rqs'))))
    return result


def required_scratch_slots(cfg):
    from woof.ensemble import batch_moist
    from woof.core.preflight import scratch_slot_registry
    names = {'rk_ru', 'rk_rv', 'rk_ww', 'acoustic_mu_pp_old', 'acoustic_th_pp_old',
             'acoustic_c2a', 'acoustic_a', 'acoustic_alpha', 'acoustic_gamma'}
    if cfg.moist and cfg.moist_cq:
        names.update(('acoustic_cqu', 'acoustic_cqv', 'acoustic_cqw'))
    if cfg.emdiv > 0:
        names.add('acoustic_mudf')
    names.update(batch_moist.required_scratch_slots(cfg))
    if cfg.km_opt == 4 or cfg.diff_6th_opt > 0:
        from woof.ensemble.batch_moist_mixing import required_scratch_slots as mixing_slots
        names.update(mixing_slots(cfg))
    registry = scratch_slot_registry(cfg)
    if cfg.specified or cfg.nested:
        names.update(name for name in registry if name.startswith('lbc_'))
    if names - registry.keys():
        raise BatchStateUnsupported('moist RK scratch registry changed; audit the stage lifetimes')
    return {name: np.dtype('float32') for name in sorted(names)}


class _MemberHeldPhysics:
    """Persistent outermost carriers for the packed column driver's rates."""
    def __init__(self, state, sample, *, components=()):
        self.state = state
        self.active = {name for name in ('ru','rv','rw','rtheta','rqv','rqc','rqr','rqi','rqs')
                       if any(getattr(source,name,None) is not None
                              for source in (sample, *components))}
    def update(self, packed):
        import cupy as cp
        for name in self.active:
            value = getattr(packed,name)
            output = self.state.storage.arrays['batch_phys_'+name]
            levels,ny,nx = output.shape[1:]
            cp.copyto(output, value.reshape(levels,self.state.members,ny,nx).transpose(1,0,2,3))
    def scalar_for(self, name):
        field = {'qv':'rqv','qc':'rqc','qr':'rqr','qi':'rqi','qs':'rqs'}.get(name)
        return self.state.storage.arrays['batch_phys_'+field] if field in self.active else None
    def add_to_slow(self):
        import cupy as cp
        for name,target in (('ru','ru_t'),('rv','rv_t'),('rtheta','rth_t'),('rw','rw_t')):
            if name in self.active:
                cp.add(getattr(self.state,target), self.state.storage.arrays['batch_phys_'+name],
                       out=getattr(self.state,target))


def prepare_moist_step(state, *, physics_adapter=None, tables=None, boundary_clock=None):
    """One launch per original operation advances every resident member."""
    from woof.config import validate_km_opt
    from woof.core import dycore
    from woof.core.physics import physics_enabled
    from woof.wrf_exact import ENABLED
    if not isinstance(state, BatchedDomainState):
        raise TypeError('moist RK needs an admitted member state')
    cfg = state.cfg
    validate_km_opt(cfg)
    dycore._validate_geopotential_config(cfg, cfg.nx, cfg.ny)
    if not cfg.moist or (cfg.mp_physics or physics_enabled(cfg) or state.physics is not None) and physics_adapter is None:
        raise BatchStateUnsupported('active moist physics requires its initialized retained-tendency and microphysics adapter')
    if cfg.km_opt not in (1,4) or cfg.khdif > 0 or cfg.kvdif > 0:
        raise BatchStateUnsupported('this graph binds metric km4/diff6; constant second-order or other closures need their scalar graph')
    if cfg.diff_6th_opt == 1:
        raise BatchStateUnsupported('original moist dynamics refuses unimplemented non-monotonic sixth-order diffusion')
    if getattr(cfg, 'mp_zero_out', 0):
        raise BatchStateUnsupported('mp_zero_out runs after the ordinary microphysics call; the batched physics adapter has no zero-out pass')
    if getattr(state,'rthften',None) is not None or getattr(state,'rqvften',None) is not None:
        raise BatchStateUnsupported('advective cumulus forcing needs its held member export before this full graph advances')
    if cfg.open_x or cfg.open_y or cfg.nested:
        raise BatchStateUnsupported('this forecast graph needs its open/nested stage producer before advancing those boundaries')
    if cfg.specified and (tables is None or boundary_clock is None):
        raise BatchStateUnsupported('specified forcing requires admitted member tables and a common solve-entry clock')
    if (cfg.zadvect_implicit or cfg.nwp_diagnostics or cfg.tke_budget
            or cfg.use_adaptive_time_step or ENABLED):
        raise BatchStateUnsupported('optional or strict moist RK operations require their complete stage identity proof')
    if cfg.time_step_sound < 2 or cfg.time_step_sound % 2:
        raise ValueError('moist acoustic RK needs an even positive sound-step count')
    backings = tuple((name, id(value)) for name, value in state.storage.arrays.items())
    def clock():
        state.elapsed_seconds = state.elapsed_seconds + cfg.dt
        state.clock['ticks'] += state.clock['step_ticks']
        state.clock['step_count'] += 1
    def check():
        if backings != tuple((name, id(value)) for name, value in state.storage.arrays.items()):
            raise BatchStateUnsupported('moist RK backings changed; bind the graph again before advancing')
    if state.members == 1:
        from woof.ensemble.batch_dycore import scalar_domain_view
        scalar = scalar_domain_view(state)
        if physics_adapter is not None:
            scalar._scratch.update(physics_adapter.driver.state._scratch)
            scalar.physics = physics_adapter.driver
            physics_adapter.driver.state = scalar
        def original_step(*,refl_10cm_due=False):
            check()
            if tables is not None:
                from woof.ensemble.batch_moist import _scalar_table_view
                metadata = _scalar_table_view(state,cfg,tables,boundary_clock)
                scalar.lateral_boundaries = metadata.lateral_boundaries
                scalar._lateral_boundary_device = metadata._lateral_boundary_device
            scalar.elapsed_seconds = state.elapsed_seconds
            dycore.step(scalar, cfg, acoustic=True,refl_10cm_due=refl_10cm_due)
            clock()
        return original_step

    from woof.ensemble import batch_acoustic, batch_bigstep, batch_glue, batch_moist
    from woof.ensemble.batch_bookkeeping import prepare_bookkeeping
    from woof.ensemble.batch_diagnostics import prepare_update_diagnostics
    from woof.ensemble.batch_fluxes import prepare_couple_momentum, prepare_omega_columns
    from woof.ensemble.batch_operators import prepare_flux_div
    from woof.core.moist import moist_species
    shared = ('dnw', 'rdnw', 'rdn', 'fnm', 'fnp', 'c1h', 'c2h', 'msft', 'msfu', 'msfv')
    if any(state.storage.specs[name].ownership != 'shared' for name in shared):
        raise BatchStateUnsupported('moist fluxes need byte-verified common vertical/map coefficients')
    save = prepare_bookkeeping(tuple((getattr(state, name), getattr(state, name+'0'))
        for name in (*dycore._PROGNOSTICS, *moist_species(state))), members=state.members)
    zero = prepare_bookkeeping(tuple((getattr(state, name), getattr(state, name))
        for name in dycore._TENDENCIES), members=state.members, zero=True)
    diagnostics = prepare_update_diagnostics(state, cfg.hypsometric_opt)
    # CU99 adds rain/snow forcing when the first call composes its target.
    # The cold target alone therefore cannot describe the scalar inventory
    # of the prepared RK graph. Its separately retained coupling is already
    # bound and species-gated by the ordinary driver before preparation.
    components = (() if physics_adapter is None or cfg.cu_physics != 99 else
                  (physics_adapter.driver.cumulus_tendencies,))
    held = None if physics_adapter is None else _MemberHeldPhysics(
        state, physics_adapter.driver.tendencies, components=components)
    fixed_mixing = add_mixing = None
    fixed_scalars = None
    if cfg.km_opt == 4 or cfg.diff_6th_opt > 0:
        from woof.ensemble import batch_mixing
        fixed_mixing = batch_mixing.prepare_fixed_tendencies(state)
        add_mixing = batch_mixing.prepare_add_fixed_dry_tendencies(state)
        fixed_scalars = fixed_mixing.scalar_tendencies
    mass, faces, theta = (batch_glue.prepare_total_mass(state), batch_glue.prepare_face_masses(state),
                          batch_glue.prepare_total_theta(state))
    physics_theta = (batch_glue.prepare_total_theta(state, perturbation="thp0")
                     if cfg.use_theta_m else None)
    ru, rv, ww = (state.existing_scratch(name) for name in ('rk_ru', 'rk_rv', 'rk_ww'))
    couple_u = prepare_couple_momentum(state.u, state.batch_mux, ru, state.c1h, state.c2h,
                                      has_msf=state.has_msf, msf=state.msfu)
    couple_v = prepare_couple_momentum(state.v, state.batch_muy, rv, state.c1h, state.c2h,
                                      has_msf=state.has_msf, msf=state.msfv)
    omega = prepare_omega_columns(ru, rv, ww, state.dnw, state.c1h, dx=cfg.dx, dy=cfg.dy,
                                 has_msf=state.has_msf, msft=state.msft)
    damping = batch_bigstep.prepare_w_damping(state)
    from woof.core.advection import vertical_orders
    vsca, vmom = vertical_orders(cfg)
    row_orders = {'': vsca, 'x': vmom, 'y': vmom, 'z': vsca}
    from woof.ensemble.batch_operators import flux_row_horizontal_orders
    row_horders = flux_row_horizontal_orders(cfg)
    theta_offset = not dycore.WRF_EXACT and cfg.h_sca_adv_order == 0
    advection = tuple(prepare_flux_div(field, ru, rv, ww, tendency, spacing, state.fnm, state.fnp,
        msf, dx=cfg.dx, dy=cfg.dy, stagger=stagger, has_msf=state.has_msf,
        open_x=dycore._boundary_x(cfg),open_y=dycore._boundary_y(cfg),spec=dycore._boundary_forced(cfg),
        vorder=row_orders[stagger], horder=row_horders[stagger],
        theta_offset=theta_offset and stagger == '')
        for field, tendency, spacing, msf, stagger in (
            (state.batch_theta, state.rth_t, state.rdnw, state.msft, ''),
            (state.u, state.ru_t, state.rdnw, state.msfu, 'x'),
            (state.v, state.rv_t, state.rdnw, state.msfv, 'y'),
            (state.w, state.rw_t, state.rdn, state.msft, 'z')))
    cq = batch_acoustic.prepare_moist_cq(state, cfg)
    slow = (batch_bigstep.prepare_slow_pgf(state, cq=cq), batch_bigstep.prepare_slow_buoyancy(state),
            batch_bigstep.prepare_slow_geopotential(state))
    rotation = batch_bigstep.prepare_coriolis_curvature(state, mut='batch_mass') if state.rotational else None
    initialize = batch_bigstep.prepare_small_step_init(state)
    finish = batch_bigstep.prepare_small_step_finish(state)
    surface = batch_bigstep.prepare_w_surface(state)
    alias = batch_glue.prepare_periodic_alias(state)
    ns = cfg.time_step_sound
    stages = ((1, cfg.dt/3.0), (max(ns//2, 1), cfg.dt/ns), (ns, cfg.dt/ns))
    final_finish = batch_bigstep.prepare_small_step_finish(state,hdiab_dt=stages[-1][0]*stages[-1][1]) if cfg.mp_physics else finish
    heating = None
    if cfg.mp_physics:
        from woof.ensemble.batch_kernel import KernelSpec,PointerSpec,prepare_batch_kernel_launch
        rows = (('rth','rth_t'),('hd','h_diabatic'),('mub','mub2d'),('mup','mup'),
                ('c1','c1h'),('c2','c2h'),('msft','msft'))
        spec = KernelSpec('held_heating','add_held_heating',tuple(PointerSpec(p,state.storage.specs[n].ownership) for p,n in rows))
        count=cfg.nz*cfg.ny*cfg.nx
        heating = prepare_batch_kernel_launch(spec,state.members,((count+127)//128,),(128,),
            tuple(state.storage.arrays[n] for _,n in rows)+(np.uint64(count),np.int32(cfg.ny*cfg.nx),np.int32(state.has_msf)),
            pointer_strides={p:state.storage.pointer_stride_bytes(n) for p,n in rows})
    mudf = state.existing_scratch('acoustic_mudf') if cfg.emdiv > 0 else None
    emdiv = batch_acoustic.prepare_emdiv_filter_launch(state, cfg, mudf) if mudf is not None else None
    means = tuple(state.existing_scratch(name) for name in ('rk_ru_m', 'rk_rv_m', 'rk_ww_m'))
    zero_flux = batch_moist.prepare_sumflux_launch(state, 'zero_sumflux', means)
    accumulate = batch_moist.prepare_sumflux_launch(state, 'accumulate_sumflux', means,
                                           (state.u_pp, state.v_pp, state.ww_pp))
    finish_flux = tuple(batch_moist.prepare_sumflux_launch(state, 'finish_sumflux', means,
                        (ru, rv, ww), nsub=nsub) for nsub, _dtau in stages)
    scalar_stages = tuple(batch_moist.prepare_scalars_stage(state, cfg, *means, nsub*dtau,
        final=index == 2, apply_relax=index == 0, physics_tendencies=held,
        fixed_tendencies=fixed_scalars) for index, (nsub, dtau) in enumerate(stages)) if tables is None else None
    def step(*,refl_10cm_due=False):
        check()
        if physics_adapter is not None and physics_theta is not None:
            # Native first_rk_step_part2 converts with grid%t_1 before
            # small_step_prep replaces it with this step's entry state.
            # Preserve the old thp0 sum in the existing batch workspace.
            physics_theta()
        save()
        if physics_adapter is not None:
            diagnostics()
            held.update(physics_adapter.compute())
            if cfg.use_theta_m:
                # Convert the summed dry-physics rates once, before mixing.
                # batch_theta is already admitted with member-major layout.
                from woof.core.theta_m_physics import _kernel
                from woof.core import constants as c
                _kernel("physics_tendency")(
                    state.batch_theta, state.qv0, state.storage.arrays["batch_phys_rqv"],
                    np.float32(c.RVOVRD), state.storage.arrays["batch_phys_rtheta"])
        if fixed_mixing is not None:
            if cfg.km_opt == 4:
                diagnostics()
            fixed_mixing()
        for index, (nsub, dtau) in enumerate(stages):
            zero()
            diagnostics()
            mass(); faces(); couple_u(); couple_v(); omega(); theta()
            batch_acoustic.prepare_moist_cq(state, cfg)
            for launch in advection:
                launch()
            for launch in slow:
                launch()
            if rotation is not None:
                rotation()
            if held is not None:
                held.add_to_slow()
            if heating is not None:
                heating()
            if add_mixing is not None:
                add_mixing()
            damping()
            if tables is not None:
                from woof.ensemble.batch_boundaries import prepare_dry_lateral_tendencies
                prepare_dry_lateral_tendencies(state,cfg,tables,boundary_clock,rk_stage=index)()
            initialize()
            coefficients = batch_acoustic.prepare_acoustic_coefficients(state, cfg, dtau, cq=cq)
            if mudf is not None and index == 0:
                mudf.fill(0)
            acoustic = batch_acoustic.prepare_acoustic_substep_launch(state, cfg, dtau, coefficients, mudf=mudf)
            zero_flux()
            for substep in range(nsub):
                if emdiv is not None:
                    emdiv()
                acoustic(first=substep == 0)
                accumulate()
            (final_finish if index == 2 else finish)()
            finish_flux[index]()
            if scalar_stages is None:
                batch_moist.prepare_scalars_stage(state,cfg,*means,nsub*dtau,final=index==2,
                    apply_relax=index==0,physics_tendencies=held,fixed_tendencies=fixed_scalars,
                    tables=tables,clock=boundary_clock)()
            else:
                scalar_stages[index]()
        if tables is not None:
            from woof.ensemble.batch_boundaries import prepare_dry_boundary_values
            prepare_dry_boundary_values(state,cfg,tables,boundary_clock)()
        surface()
        diagnostics()
        if physics_adapter is not None and cfg.mp_physics:
            physics_adapter.apply_microphysics(refl_10cm_due=refl_10cm_due)
            diagnostics()
        alias()
        clock()
    return step
