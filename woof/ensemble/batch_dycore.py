"""Prepared dry acoustic RK3 execution over independent member state slabs.

This first executor binds the periodic dry, fixed-step configuration. It is an
internal qualification path. Physics and source preparation are separate
integrations and are refused until their bindings exist.
"""
from __future__ import annotations

from woof.ensemble.batch_state import BatchedDomainState, BatchStateUnsupported


def member_domain_view(state, member):
    """A scalar-shaped readout view for the established history writer.

    The owning batch and stream ordering remain the caller's responsibility.
    This view is for I/O; the member batch advances through prepare_dry_step.
    """
    from woof.core.state import DomainState
    if not isinstance(state, BatchedDomainState):
        raise TypeError("member readout needs an inventoried batch")
    result = DomainState.__new__(DomainState)
    result.__dict__.update(state.scalars)
    result.__dict__.update({name: state.member_view(name, member)
                            for name in state.storage.specs
                            if not name.startswith("scratch:")})
    result.elapsed_seconds = state.elapsed_seconds
    result._scratch = {slot: state.scratch_member_view(slot, member)
                       for slot in state._scratch}
    result._scratch_arena = None
    result._host_setup_state = False
    result._phb_host = state.phb_host_members[member]
    result.physics = None
    result.lateral_boundaries = None
    return result


def scalar_domain_view(state):
    """Delegate the original driver only when the batch has one member."""
    if not isinstance(state, BatchedDomainState) or state.members != 1:
        raise TypeError("scalar delegation needs a one-member batch")
    return member_domain_view(state, 0)


def _validate_configuration(state):
    from woof.core.dycore import _boundary_forced, _validate_geopotential_config
    from woof.core.physics import physics_enabled
    from woof.config import validate_km_opt
    from woof.wrf_exact import ENABLED
    if not isinstance(state, BatchedDomainState):
        raise TypeError("prepared batched stepping needs an admitted BatchedDomainState")
    cfg = state.cfg
    validate_km_opt(cfg)
    _validate_geopotential_config(cfg, cfg.nx, cfg.ny)
    if cfg.time_step_sound < 2 or cfg.time_step_sound % 2:
        raise ValueError("acoustic RK3 requires an even positive time_step_sound")
    if cfg.moist or physics_enabled(cfg) or state.physics is not None:
        raise BatchStateUnsupported("this executor needs dry physics-off member inputs")
    if (cfg.km_opt not in (1, 4) or cfg.khdif > 0 or cfg.kvdif > 0
            or (cfg.km_opt == 4 and cfg.diff_opt != 2)):
        raise BatchStateUnsupported("this executor binds metric km_opt=4 and diff6; other mixing needs its own member graph")
    if cfg.open_x or cfg.open_y or _boundary_forced(cfg) or state.lateral_boundaries is not None:
        raise BatchStateUnsupported("lateral forcing/open-boundary bindings are not attached to this executor")
    if cfg.w_damping or cfg.zadvect_implicit or cfg.nwp_diagnostics or cfg.tke_budget:
        raise BatchStateUnsupported("vertical damping/implicit/optional diagnostics require their member bindings")
    if getattr(cfg, "use_adaptive_time_step", False):
        raise BatchStateUnsupported("adaptive schedules need a member trajectory identity proof")
    if state.members > 1 and ENABLED:
        raise BatchStateUnsupported("strict helper and face-mass bindings await member qualification")
    return cfg


def prepare_dry_step(state, *, layout_trial="none", advection_family=False,
                     acoustic_fusion=False, acoustic_shared=False):
    """Prepare one full acoustic RK3 step with no loop over advancing members.

    N=1 delegates the original DomainState driver. N>1 executes each numerical
    operation once over the batch. The closure retains backings and must be
    rebuilt if the allocation inventory changes. Its clock is fixed and every
    scalar reference run must use exactly the same step and boundary calendar.
    Explicit layout trials replace only Omega and advection/its RK-zero rows.
    Inner trial conversions remain inside the timed step; N=1 stays original.
    """
    from woof.ensemble.batch_layout_trials import LAYOUT_TRIALS
    if layout_trial not in LAYOUT_TRIALS:
        raise ValueError("layout_trial must be none, outermost or innermost")
    if advection_family and layout_trial == "none":
        raise ValueError("advection_family needs an explicit outermost or innermost trial")
    if type(acoustic_fusion) is not bool:
        raise TypeError("acoustic_fusion must be an explicit boolean trial selection")
    if type(acoustic_shared) is not bool or (acoustic_shared and not acoustic_fusion):
        raise ValueError("acoustic_shared requires the joined acoustic trial")
    cfg = _validate_configuration(state)
    from woof.core import dycore
    bound = tuple((name, id(array)) for name, array in state.storage.arrays.items())

    def check_binding():
        if bound != tuple((name, id(array)) for name, array in state.storage.arrays.items()):
            raise BatchStateUnsupported("state backings changed; rebind the complete step before advancing")

    def advance_clock():
        state.elapsed_seconds = state.elapsed_seconds + cfg.dt
        state.clock["ticks"] += state.clock["step_ticks"]
        state.clock["step_count"] += 1

    if state.members == 1:
        scalar = scalar_domain_view(state)

        def single_step():
            check_binding()
            dycore.step(scalar, cfg, acoustic=True)
            advance_clock()

        single_step.trial_receipt = {"selection": layout_trial, "effective": "original_n1",
                                    "additional_workspace_bytes": 0,
                                    "fused_zero_fields": (), "transitions_in_step": False,
                                    "advection_family_requested": bool(advection_family)}
        return single_step

    from woof.ensemble import batch_acoustic, batch_bigstep, batch_glue
    acoustic_substep_factory = batch_acoustic.prepare_acoustic_substep_launch
    if acoustic_fusion:
        from woof.ensemble.batch_acoustic_fusion import prepare_acoustic_substep_launch
        acoustic_substep_factory = prepare_acoustic_substep_launch
    from woof.ensemble.batch_bookkeeping import prepare_bookkeeping
    from woof.ensemble.batch_diagnostics import prepare_update_diagnostics
    from woof.ensemble.batch_fluxes import prepare_couple_momentum, prepare_omega_columns
    from woof.ensemble.batch_operators import prepare_flux_div

    # The current flux adapters consume shared vertical/map coefficients.
    # Sharing must have been established from every prepared member's bytes.
    required_shared = ("dnw", "rdnw", "rdn", "fnm", "fnp", "c1h", "c2h",
                       "msft", "msfu", "msfv")
    if any(state.storage.specs[name].ownership != "shared" for name in required_shared):
        raise BatchStateUnsupported("this flux graph needs byte-verified shared vertical/map coefficients")

    save = prepare_bookkeeping(tuple((getattr(state, name), getattr(state, name + "0"))
                                    for name in dycore._PROGNOSTICS), members=state.members)
    fused_zero_fields = frozenset(("rth_t", "ru_t", "rv_t", "rw_t")) if layout_trial != "none" else frozenset()
    zero = prepare_bookkeeping(tuple((getattr(state, name), getattr(state, name))
                                    for name in dycore._TENDENCIES if name not in fused_zero_fields),
                               members=state.members, zero=True)
    diagnostics = prepare_update_diagnostics(state, cfg.hypsometric_opt)
    fixed_mixing = add_mixing = None
    if cfg.km_opt == 4 or cfg.diff_6th_opt > 0:
        from woof.ensemble.batch_mixing import (
            prepare_fixed_tendencies, prepare_add_fixed_dry_tendencies)
        fixed_mixing = prepare_fixed_tendencies(state)
        add_mixing = prepare_add_fixed_dry_tendencies(state)
    mass = batch_glue.prepare_total_mass(state)
    faces = batch_glue.prepare_face_masses(state)
    theta = batch_glue.prepare_total_theta(state)
    ru, rv, ww = (state.existing_scratch(slot) for slot in ("rk_ru", "rk_rv", "rk_ww"))
    couple_u = prepare_couple_momentum(state.u, state.batch_mux, ru, state.c1h, state.c2h,
                                      has_msf=state.has_msf, msf=state.msfu)
    couple_v = prepare_couple_momentum(state.v, state.batch_muy, rv, state.c1h, state.c2h,
                                      has_msf=state.has_msf,
                                      msf=state.msfv_inv if dycore.WRF_EXACT else state.msfv,
                                      reciprocal=dycore.WRF_EXACT)
    from woof.core.advection import vertical_orders
    vsca, vmom = vertical_orders(cfg)
    advection_rows = ((state.batch_theta, state.rth_t, state.rdnw, state.msft, ""),
                      (state.u, state.ru_t, state.rdnw, state.msfu, "x"),
                      (state.v, state.rv_t, state.rdnw, state.msfv, "y"),
                      (state.w, state.rw_t, state.rdn, state.msft, "z"))
    row_orders = {"": vsca, "x": vmom, "y": vmom, "z": vsca}
    if layout_trial == "none":
        omega = prepare_omega_columns(ru, rv, ww, state.dnw, state.c1h,
                                     dx=cfg.dx, dy=cfg.dy, has_msf=state.has_msf, msft=state.msft)
        advection = tuple(prepare_flux_div(field, ru, rv, ww, tendency, spacing,
                                          state.fnm, state.fnp, msf, dx=cfg.dx, dy=cfg.dy,
                                          stagger=stagger, has_msf=state.has_msf,
                                          vorder=row_orders[stagger])
                          for field, tendency, spacing, msf, stagger in advection_rows)
        trial_receipt = {"selection": "none", "additional_workspace_bytes": 0,
                         "fused_zero_fields": (), "transitions_in_step": False}
    else:
        from woof.ensemble.batch_layout_trials import prepare_stage_trials
        omega, advection, trial_receipt = prepare_stage_trials(
            state, ru, rv, ww, advection_rows, layout=layout_trial, family=advection_family)
    slow = (batch_bigstep.prepare_slow_pgf(state),
            batch_bigstep.prepare_slow_buoyancy(state),
            batch_bigstep.prepare_slow_geopotential(state))
    rotation = (batch_bigstep.prepare_coriolis_curvature(state, mut="batch_mass")
                if state.rotational else None)
    initialize = batch_bigstep.prepare_small_step_init(state)
    finish = batch_bigstep.prepare_small_step_finish(state)
    surface = batch_bigstep.prepare_w_surface(state)
    alias = batch_glue.prepare_periodic_alias(state)
    ns = cfg.time_step_sound
    stages = ((1, cfg.dt / 3.0), (max(ns // 2, 1), cfg.dt / ns), (ns, cfg.dt / ns))
    mudf = state.existing_scratch("acoustic_mudf") if cfg.emdiv > 0 else None
    if cfg.emdiv > 0 and mudf is None:
        raise BatchStateUnsupported("emdiv needs its planned acoustic_mudf backing")
    emdiv = batch_acoustic.prepare_emdiv_filter_launch(state, cfg, mudf) if mudf is not None else None
    prepared_substeps = None
    if acoustic_fusion:
        # Coefficient values change by stage, but their admitted allocations
        # remain fixed. Build source, audit the ABI and bind handles once.
        coefficient_backings = tuple(state.existing_scratch(name) for name in (
            "acoustic_c2a", "acoustic_a", "acoustic_alpha", "acoustic_gamma"))
        if any(value is None for value in coefficient_backings):
            raise BatchStateUnsupported("joined acoustic submission needs all four admitted coefficient backings")
        prepared_substeps = tuple(acoustic_substep_factory(
            state, cfg, dtau, coefficient_backings, mudf=mudf,
            shared_intermediates=acoustic_shared) for _nsub, dtau in stages)

    def step():
        check_binding()
        save()
        if fixed_mixing is not None:
            if cfg.km_opt == 4:
                diagnostics()
            fixed_mixing()
        for stage_index, (nsub, dtau) in enumerate(stages):
            zero()
            diagnostics()
            mass()
            faces()
            couple_u()
            couple_v()
            omega()
            theta()
            for launch in advection:
                launch()
            for launch in slow:
                launch()
            if rotation is not None:
                rotation()
            if add_mixing is not None:
                add_mixing()
            initialize()
            coefficients = batch_acoustic.prepare_acoustic_coefficients(state, cfg, dtau)
            if mudf is not None and stage_index == 0:
                mudf.fill(0)
            substep = (prepared_substeps[stage_index] if prepared_substeps is not None
                       else acoustic_substep_factory(state, cfg, dtau, coefficients, mudf=mudf))
            for substep_index in range(nsub):
                if emdiv is not None:
                    emdiv()
                substep(first=(substep_index == 0))
            finish()
        surface()
        diagnostics()
        alias()
        advance_clock()

    step.trial_receipt = {**trial_receipt, "acoustic_fusion_requested": acoustic_fusion,
                         "acoustic_shared_requested": acoustic_shared,
                         "acoustic_launches_prepared_once": prepared_substeps is not None}
    return step
