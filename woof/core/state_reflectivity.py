"""Current-state reflectivity shared by analysis history and radar observation operators.

Uses the production microphysics diagnostic and its particle-size distribution.
This entry runs between steps and never advances the prognostic state.
"""
from woof.core import constants as c

def _nssl_reflectivity(state, temperature, pressure, cfg=None):
    """``mp_physics=18`` through the product's own NSSL diagnostic.

    Not a fourth Z formula and not a mapping onto Morrison: the same
    ``woof.core.nssl2_diagnostics.diagnose_radardd02_if_due`` the
    production coordinator calls, reading the state's own five ice
    categories, five number moments and two volume moments.

    Two things differ from the 1/6/8/10 routes and a DA caller has to know
    both.  NSSL's ``radardd02`` floors at **0 dBZ**, not -35, so clear air
    in an NSSL H(x) column reads 0 and an innovation against a -35 dBZ
    observation floor is meaningless -- give reflectivity observations a
    matching floor or mask the clear-air cells.  And the moments here are
    per unit mass of dry air (``concentration_space=False``), which is what
    a ``DomainState`` carries; the coordinator passes ``True`` because it
    hands over the scheme's internal per-volume slab.
    """
    if cfg is not None:
        from woof.core.nssl2_contract import resolve_nssl2_mode_for_config
        if not resolve_nssl2_mode_for_config(cfg).two_moment:
            from woof.core.nssl1 import reflectivity
            return reflectivity(state,cfg,temperature,pressure)
    import cupy as cp

    from woof.core.nssl2_diagnostics import (       # noqa: PLC0415
        diagnose_radardd02_if_due,
    )
    from woof.core.state import DTYPE               # noqa: PLC0415

    shape = tuple(state.p.shape)
    # Dry-air density, formed the way the NSSL runtime forms it
    # (nssl2_runtime._prepare_fields: rho = 1/alt).
    rho = state.scratch(shape, "da_nssl_rho")
    rho[...] = DTYPE(1.0) / state.alt
    if temperature is None:
        thb = state.thb if state.thb.ndim == 3 else state.thb[:, None, None]
        t = state.scratch(shape, "da_nssl_t")
        t[...] = (state.total_theta() if getattr(state, "use_theta_m", 0)
                  else thb + state.thp) * cp.power(
            state.p / DTYPE(c.P0), DTYPE(c.RCP))
    else:
        t = temperature
    refl = state.scratch(shape, "refl_10cm")
    diagnose_radardd02_if_due(state, rho, t, refl, output_due=True,
                              concentration_space=False,
                              shapes=(getattr(cfg, "nssl_alphah", 0.0),
                                      getattr(cfg, "nssl_alphahl", 1.0)))
    return refl


def _thompson_graupel_number(state, temperature, pressure, *,
                             version: str = "wrf_461", aerosol: bool = False):
    """Classic Thompson's transient graupel number, diagnosed here.

    ``version`` is the generation (:func:`thompson_generation`): the
    v4.6.1 wrapper diagnosis below, or, for the operational fork, the
    fork's own reflectivity intercept.

    ``mp_physics=8`` carries no graupel number in the Registry.  WRF's
    ``mp_gt_driver`` DIAGNOSES one from qg at the beginning of every
    call, evolves that private moment through the source and
    sedimentation operators, applies one final lower bound and MVD
    reconstruction, and hands the result to ``calc_refl10cm``
    (module_mp_thompson.F 1265-1281, 1915-1939 and 4059-4077).  woof
    transcribes the two ends of that as
    ``launch_classic_graupel_number_init`` and
    ``launch_classic_graupel_number_finalize``, and the microphysics
    adapter passes the evolved moment to the reflectivity operator on
    every output-due step.  The operator requires it rather than
    inventing one, which is right for an in-call caller: a shadow that
    skipped the call's source and fallout tendencies would not be the
    moment WRF's own Z read.

    AN OBSERVATION OPERATOR IS NOT AN IN-CALL CALLER.  H_Z(x) is
    evaluated BETWEEN steps, on a state no microphysics call is in the
    middle of, and for that state the wrapper's diagnosis IS the moment:
    no tendency has acted since, so init followed by finalize is exactly
    the sequence WRF would carry into ``calc_refl10cm`` on a call whose
    graupel tendency was zero.  Both ends read only qg, T, p and qv off
    the state, so the derivation asks for nothing the analysis does not
    already hold.

    Written into the microphysics adapter's own shadow slot rather than
    a second buffer: it is the same quantity, the slot is scratch the
    scheme rewrites unconditionally at the entry of its next call, and a
    DA operator runs between calls, so there is no value here for this
    to destroy.

    The graupel MASS is a different matter, and the finalize runs on a
    private copy of it.  ``launch_classic_graupel_number_finalize`` also
    carries WRF's terminal graupel zero (module_mp_thompson.F:4058-4063):
    every ``qg <= R1`` is written as zero IN PLACE.  Inside the call that
    is a state update the scheme owns; here it rewrote the background the
    operator observes.  The DA cycle evaluates H_Z(x) on a restored
    member right after its pending increment is applied (the start
    composite of a leg 0, which a ``--resume-ensemble`` run has and the
    same leg of the uninterrupted run does not), so the zero moved the
    resumed trajectory off the continuous one: every member of a resumed
    step diverged while the unanalysed control, which carries no
    sub-R1 residue, stayed bit-identical.  The copy changes neither
    output: the shadow is formed from identical values, and
    ``refl10cm_thompson_column`` reads graupel only where ``qg > 1e-6``,
    far above R1.

    Returns ``(temperature, pressure, shadow)`` because the temperature
    the shadow was diagnosed at is the temperature the reflectivity has
    to be diagnosed at; handing the pair on is what keeps the two from
    being formed twice and differing.
    """
    import cupy as cp                                  # noqa: PLC0415

    from woof.core.state import DTYPE                 # noqa: PLC0415
    from woof.core.thompson import (                  # noqa: PLC0415
        launch_classic_graupel_number_finalize,
        launch_classic_graupel_number_init,
    )

    shape = tuple(state.p.shape)
    if temperature is None:
        thb = state.thb if state.thb.ndim == 3 else state.thb[:, None, None]
        pressure = state.p
        temperature = state.scratch(shape, "refl_t")
        temperature[...] = (state.total_theta() if getattr(state, "use_theta_m", 0)
                  else thb + state.thp) * cp.power(
            state.p / DTYPE(c.P0), DTYPE(c.RCP))
    shadow = state.scratch(shape, "mp_thompson_graupel_number_shadow")
    # A transient copy, not a scratch slot: it is dead when this returns,
    # and a slot would enter the pinned mp=8 arena layout for a DA-only
    # temporary.
    if version == "wrf_39_noaa":
        # The fork's calc_refl10cm does not read an evolved number at all:
        # it diagnoses its own graupel intercept from the final state
        # (fork :5486-5510, from graupel content and supercooled rain,
        # non-increasing downward), and the mp=28 adapter forms exactly
        # that moment on every output-due step
        # (woof/core/microphysics_aerosol.py, WRF39_INTERCEPT_REFLECTIVITY)
        # before the shared column kernel reads it.  H_Z(x) under the fork
        # has to form the same moment: the v4.6.1 wrapper diagnosis below
        # (gonv 1e2..1e6, graupel content alone) is a different PSD, and a
        # DA that inverts dBZ through it against history written through
        # the fork's intercept is inverting a different model than it
        # integrates.  Measured on the Iowa 2024-05-21 18Z f01 crop: the
        # two intercepts on one state differ by 1..5 dBZ in the 30..50 dBZ
        # band (a rented box, WOOF-FIX-PROGRAM-2026-10-06/dbz/REPORT.md).
        from woof.core.thompson_aerosol_launch import (   # noqa: PLC0415
            thompson_version_scope)
        from woof.core.thompson_aerosol_state import (    # noqa: PLC0415
            WRF39_INTERCEPT_REFLECTIVITY, launch_wrf39_graupel_intercept)

        with thompson_version_scope(version):
            launch_wrf39_graupel_intercept(
                state.qg, state.qr, state.nr, temperature, pressure,
                state.qv, shadow, mode=WRF39_INTERCEPT_REFLECTIVITY)
        return temperature, pressure, shadow
    graupel_mass = state.qg.copy()
    if aerosol:
        from woof.core.thompson_aerosol_state import (
            launch_aa_graupel_number_init, launch_aa_graupel_number_finalize,
            launch_tau1_density)
        launch_aa_graupel_number_init(
            graupel_mass, temperature, pressure, state.qv, shadow)
        # Between steps there is no evaporation tendency. Form the scheme's
        # terminal density on the observed state, without writing that state.
        density = cp.empty_like(graupel_mass)
        launch_tau1_density(temperature, pressure, state.qv, density)
        launch_aa_graupel_number_finalize(graupel_mass, shadow, density)
    else:
        launch_classic_graupel_number_init(
            graupel_mass, temperature, pressure, state.qv, shadow)
        launch_classic_graupel_number_finalize(
            graupel_mass, temperature, pressure, state.qv, shadow)
    del graupel_mass
    return temperature, pressure, shadow


def thompson_generation(cfg) -> str:
    """The Thompson generation whose graupel PSD H_Z(x) must read.

    ``RunConfig.thompson_version`` (``wrf_461`` by default, ``wrf_39_noaa``
    for the operational fork) is what the mp=28 adapter runs the scheme
    and writes REFL_10CM through; the operator reads the same selector so
    the two never diverge.  Classic mp=8 has one generation.  A config
    object without the field (the ``SimpleNamespace`` the tests build) is
    the default generation, as it is for the adapter.
    """
    from woof.core.thompson_aerosol_launch import (       # noqa: PLC0415
        THOMPSON_VERSIONS)

    if int(cfg.mp_physics) != 28:
        return "wrf_461"
    version = getattr(cfg, "thompson_version", "wrf_461")
    if version not in THOMPSON_VERSIONS:
        raise ValueError(
            f"thompson_version={version!r} is not one of {THOMPSON_VERSIONS}")
    return version



def diagnose(state, cfg):
    """Run the scheme's own diagnostic over a resident GPU state."""
    import cupy as cp
    if int(cfg.mp_physics) == 97:
        from woof.core.mp_column_schemes import reflectivity
        return reflectivity(state, cfg)
    if int(cfg.mp_physics) in (5, 15):
        from woof.core.fer_hires import reflectivity
        return reflectivity(state, cfg)
    if int(cfg.mp_physics) == 900:
        from woof.core.tempo import _runtime
        return _runtime(cfg, state.p.shape[0]).reflectivity(state, cfg)
    if int(cfg.mp_physics) == 55:
        from woof.core.ishmael import reflectivity
        return reflectivity(state)
    if int(cfg.mp_physics) in (50, 51, 52, 53):
        from woof.core.p3_device import reflectivity
        return reflectivity(state)
    if int(cfg.mp_physics) == 56:
        from woof.core.ntu import reflectivity
        return reflectivity(state, cfg=cfg)
    if int(cfg.mp_physics) == 9:
        from woof.core.milbrandt2 import reflectivity
        return reflectivity(state)
    if int(cfg.mp_physics) == 18:
        return _nssl_reflectivity(state, None, None, cfg)
    from woof.core.wsm_family import family_selectors, row, launch_refl10cm
    if int(cfg.mp_physics) in family_selectors():
        mp = int(cfg.mp_physics)
        thb = state.thb if state.thb.ndim == 3 else state.thb[:, None, None]
        temperature = state.scratch(tuple(state.p.shape), "refl_t")
        temperature[...] = (state.total_theta() if getattr(state, "use_theta_m", 0)
                  else thb + state.thp) * cp.power(state.p / c.P0, c.RCP)
        refl = state.scratch(tuple(state.p.shape), "refl_10cm")
        launch_refl10cm(mp, qv=state.qv,
                       species={n: getattr(state, n) for n in row(mp).refl_species},
                       t=temperature, pressure=state.p, refl=refl)
        return refl
    temperature = pressure = shadow = None
    if int(cfg.mp_physics) == 28 and thompson_generation(cfg) == "wrf_461":
        from woof.core.thompson_aerosol_state import launch_aerosol_exner
        pressure = state.p
        exner = cp.empty_like(pressure)
        launch_aerosol_exner(pressure, exner)
        thb = state.thb if state.thb.ndim == 3 else state.thb[:, None, None]
        temperature = state.scratch(tuple(pressure.shape), "refl_t")
        temperature[...] = (state.total_theta() if getattr(state, "use_theta_m", 0)
                  else thb + state.thp) * exner
    if int(cfg.mp_physics) in (8, 28):
        temperature, pressure, shadow = _thompson_graupel_number(
            state, temperature, pressure, version=thompson_generation(cfg),
            aerosol=int(cfg.mp_physics) == 28)
    from woof.core.refl import compute_refl_10cm
    return compute_refl_10cm(state, cfg, temperature=temperature, pressure=pressure,
                             thompson_graupel_number=shadow)
