"""Forecast adapter for WRF v4.6.1 Thompson aerosol-aware (``mp_physics=28``).

:func:`_apply_thompson_aerosol` is a **sibling** of
``woof.core.microphysics._apply_thompson``, not a branch inside it.  That is
deliberate: the port's original isolation can be checked against its
historical receipts in ``tests/test_mp8_frozen.py``. Those receipts describe
the isolation of this aerosol port, not a current mp=8 byte-freeze claim.
The classic kernels changed after their July 2026 matched WRF run.
Current component comparisons and their limits are documented in PHYSICS.md.

Numerical authority is ``wrf461-pristine/phys/module_mp_thompson.F``
(WRF v4.6.1, commit ``d66e442``).  Bare ``:NNNN`` line numbers refer to it.

THE FIVE THINGS THIS FILE IS RESPONSIBLE FOR
--------------------------------------------
Every kernel below was unit-gated by its own package against the Fortran
oracle before this adapter existed.  Unit gates cannot see composition
defects, so the five properties this file uniquely owns are stated here and
each is pinned by a named test in ``tests/test_thompson_aerosol_adapter.py``:

1.  **The accumulator contract.**  ``state.nc`` / ``state.nwfa`` /
    ``state.nifa`` are READ-ONLY entry state for the whole call.  The three
    scratch accumulators are ZEROED at entry (they are persistent scratch --
    ``state.scratch`` survives across steps by design, ``state.py:710-735`` --
    so carrying last step's aerosol tendency forward would be a slow,
    plausible-looking drift no single-step column test could catch), written
    by every aerosol kernel, and applied EXACTLY ONCE by
    ``launch_aerosol_state_finalize`` (:3972-4021), which carries WRF's only
    clamps.
    Pinned by ``test_entry_state_is_read_only_until_the_single_terminal_apply``
    and ``test_accumulators_are_zeroed_at_entry_not_carried_between_calls``.

1b. **The cloud-water accumulator.**  ``state.qc`` obeys the same contract:
    it is WRF's ``qc1d``, read-only from the entry rewrite to the terminal
    apply, and every stage that moves cloud water -- the cold and warm
    source networks (:2987), the condensation (:3480), the cloud fallout
    (:3832) and the phase cleanup (:3949, :3962) -- adds its tendency to
    ``qcten`` and reads the working cloud as ``qc1d + qcten*DT``, formed
    the way WRF forms it.  The sum is applied ONCE, at :3975.  Applying each
    stage to ``state.qc`` in turn rounded the cloud once per stage where WRF
    rounds it once per call: one float32 ulp of the entry value on
    aero-cloud-freeze-nc (4.926e-06 against the 2e-06 gate), and on
    aero-cold-overlap the same ulp left the cloud at exactly zero where WRF
    keeps 2**-36, flipping the :4007 branch and zeroing the droplets.
    Measured stage by stage against WRF's own running qcten with
    tools/thompson_wrf461_oracle/compare_port_stages_aero.py: bitwise after
    the condensation, the fallout and the cleanup on every G3 fixture.

2.  **Warm/cold entry-mask disjointness.**  The cold network returns for
    ``T >= 273.15`` and the warm network returns unless its held mask is set,
    and both kernel headers state that the two gates are exact complements.
    That is only true if the mask is captured on the **entry** temperature,
    BEFORE the cold network writes its latent heating in place.  Capturing it
    afterwards lets a cold cell that the cold network heated across 0 C run
    the warm path as well, double-counting ``pnc_wau``/``pnc_rcw``/
    ``pna_rca``/``pnd_rcd`` at that level -- finite, stable and wrong.
    Pinned by ``test_warm_entry_mask_is_captured_before_the_cold_network``.

3.  **Launch order.**  Surface emission is the LAST aerosol write of the call
    and follows the terminal apply (mp_gt_driver:1310-1327 runs after
    ``mp_thompson`` returns, so it is deliberately unclamped).  The ``ncten``
    balance limiter (:2996-3019) runs ONCE, between the warm network and the
    saturation adjustment -- WRF applies it once per column after every
    ``ncten`` source and before ``pnc_wcd``; calling it from inside both
    networks double-applies it.
    Pinned by ``test_launcher_call_order_is_exactly_the_wrf_driver_order``.

4.  **No Cooper.**  In mp=28 ``iceDeMott`` REPLACES Cooper nucleation
    (:2537-2551 selects one or the other on ``is_aerosol_aware``).  The
    classic ``launch_frozen_vapor_network*`` / ``launch_ice_nucleation``
    launchers carry Cooper and must never appear on this path; using them
    would ADD a second ice source rather than substitute for it.
    Pinned by ``test_no_cooper_bearing_classic_launcher_is_ever_called``.

1c. **The rain and ice accumulators** (the v4.6.1 generation).  ``qr``,
    ``nr``, ``qi`` and ``ni`` follow the same contract: the source networks
    add to ``qrten``/``nrten``/``qiten``/``niten`` (:3022-3067) and run the
    :3033-3055 / :3070-3091 balances in WRF's tendency form, the rain
    evaporation adds :3562/:3564, mp=28's own rain and ice fallouts add
    :3800-3810 / :3851-3862 to them, the cleanup adds :3949-3964, and
    ``launch_terminal_rain_ice`` applies them once with WRF's :4023-4053
    size bounds -- after the freeze, where WRF has them.  In-place, the ice
    size bound sat in the ice fallout, before the freeze, and the freeze's
    negative droplet number then landed on an already bounded ice number
    (wp08-freeze level 4); stage by stage the in-place rain also rounded
    once per stage.  The fork keeps the in-place form.

5.  **Which density feeds which kernel.**  WRF keeps TWO air densities in
    play across the condensation seam and uses both.  :3193 diagnoses the
    TAU+1 density; :3242-3243 builds the working rain mass and number from
    it and :3384-3388 freezes ``ilamr``/``N0_r`` from those; :3490 then
    OVERWRITES ``rho(k)`` inside the condensation loop and :3505-3520's
    ``orho``/``rhof``/``vsc2``/``rvs`` read the newer one.  ``prv_rev`` and
    ``pnr_rev`` therefore scale with the OLDER density while everything
    around them scales with the newer.  Both kernels expose exactly what
    they need -- ``launch_aerosol_saturation_adjust`` writes the :3193
    density into ``reference_density`` unconditionally, and
    ``launch_aerosol_rain_evaporation`` takes it as ``entry_density`` -- so
    the only thing that can get this wrong is the wiring in this file, and
    getting it wrong scales rain evaporation by ``rho_post/rho_pre``: 1.9e-3
    on aero-reduces-to-classic, which was the port's ONLY carved-out
    end-to-end tolerance until WP-12a.
    Pinned by
    ``test_the_adapter_hands_rain_evaporation_wrfs_pre_condensation_density``
    and ``test_suppressing_the_pre_condensation_density_reproduces_the_old_
    defect``.

5b. **And which density SEDIMENTATION reads, which is a third decision.**
    ``launch_rain_sedimentation`` forms its sedimenting rain mass and number
    as ``qr*rho`` and ``nr*rho`` from the ``reference_density`` it is handed;
    those are WRF's ``rr(k)``/``nr(k)`` at :3794-3795, and WRF builds them in
    exactly two places -- :3237-3238 from the :3193 TAU+1 density at every
    level with ``L_qr``, and :3568-3570 from the :3490 post-condensation
    density ONLY where the :3501-3502 rain-evaporation gate passed.  So the
    correct buffer is a LEVEL-WISE MIXTURE, not either density.
    ``launch_aerosol_rain_evaporation`` produces exactly that mixture in its
    ``reference_density`` output: it writes ``entry_density`` by default and
    overwrites with its own post-condensation rho after all three of WRF's
    gates pass.  Before WP-13a it wrote the post-condensation rho
    unconditionally, which gave every level the :3568 answer including the
    ones WRF never rewrote -- worth 5.165e-04 on aero-drop-evap's RAINNC
    (whose column passes the gate at NO level, so the :3237 pair should have
    stood everywhere) and 1.279e-04 on aero-ice-demott-idxin's.  Both
    fixtures now clear the flat G3 gate.
    Pinned by ``test_rain_sedimentation_gets_wrfs_level_wise_working_rain_
    density`` and, at the kernel, by tests/test_thompson_aerosol_sat_gpu.py::
    ``test_rain_evaporation_exports_the_sedimentation_density_wrf_actually_
    used``.

    WHAT WAS WRONG THERE, AND HOW IT CLOSED: the shared rain fallout
    re-applied WRF's :3240-3250 ``mvd_r`` clamp at sedimentation time, which
    :3568-3570 does not carry, and gated rain presence on a MIXING RATIO
    where :3616 tests a MASS CONCENTRATION built only where L_qr holds.  The
    clamp is gone from thompson_rain_sediment_impl, and both schemes' rain
    evaporations now hand the fallout WRF's L_qr and the :3568 rewrite in
    this density (zero where :3236 failed, negative where :3568 floored the
    pair), read by the ``_with_presence`` entry points this adapter launches
    and, since the classic rain fallout repair (7727fda3c), microphysics.py
    launches for mp=8 as well.  On the v4.6.1 accumulator path (1c) mp=28's
    own tendency-form rain fallout forms the gate and the pair where WRF
    does; ``test_the_frozen_kernel_mechanisms_no_longer_reach_mp28`` keeps
    the record of both.

WHAT IS REUSED FROM THE FROZEN mp=8 MODULE, AND WHY THAT IS SOUND
-----------------------------------------------------------------
Two launchers come from ``woof.core.thompson`` unchanged on the v4.6.1
path: ``launch_hydrometeor_column_mask`` (twice) and
``launch_graupel_fallout_column_mask``, pure column reductions over mass
with no arithmetic to round.  Everything else is mp=28's own: the rain and
ice fallout in tendency form (property 1c), and the snow and graupel
fallout, the surface totals in mp_gt_driver's order of addition, the
classic graupel number's entry diagnosis and terminal bound, and the 10 cm
reflectivity, which used to be thompson.cu's and refl.cu's classic kernels.
Those units are byte-frozen for mp=8 and their arithmetic is not WRF's
(CUDA's powf/pow/log10f, the graupel slope taken from the intercept with a
size clamp WRF does not apply, the substep factor folded as DT*onstep,
crg(4) = 720, binary64 radar constants); the 0 ULP column oracle
(tools/thompson_aerosol_column_oracle) measured them.  ``nwfa`` and
``nifa`` have NO sedimentation term anywhere in module_mp_thompson.F; any
implementation that adds one is wrong.  The fork generation still reuses
the classic reflectivity on its own intercept.

The two reused masks are called with the identical ordered argument tuples
mp=8 uses, which
``test_reused_classic_launchers_receive_the_mp8_argument_shape`` pins.

ENTRY-STATE ALIASING (why there are no ``nc_entry`` scratch copies)
-------------------------------------------------------------------
``launch_aerosol_entry_cloud_number`` reproduces :1826-1848 including its
in-place side effect: ``qc1d`` and ``nc1d`` are ZEROED wherever
``qc1d <= R1`` (:1844-1845).  After that call ``state.nc`` *is* WRF's
``nc1d`` and ``state.qc`` *is* WRF's ``qc1d``, and nothing writes either
again until the terminal apply at step 13 -- so both are passed as the
entry arguments everywhere, with no copy.  The same holds for
``state.nwfa``/``state.nifa``, which no kernel between the entry snapshot
and the finalize writes at all.  ``woof/core/preflight.py``'s mp=28 slot
registry reflects exactly this: it budgets the ``qcten`` accumulator and an
``ni_entry`` copy, and no ``qc_entry`` or ``nc_entry`` copy.
"""

from __future__ import annotations

import cupy as cp
import numpy as np

from woof.config import RunConfig
from woof.core import constants as c
from woof.core.microphysics import (
    MicrophysicsDiagnostics,
    moist_physics_finish,
    save_pre_mp_theta,
)
from woof.core.state import DTYPE, DomainState
from woof.physics_compat import thompson_table_root as _thompson_table_root

#: module_mp_thompson.F:183.  The entry threshold that decides whether a
#: level carries a species at all.
R1 = 1.0e-12

#: The state fields an mp=28 call requires.  ``nwfa2d``/``nifa2d`` are checked
#: separately because they are ``(ny, nx)`` and are INTENT(IN) to WRF.
REQUIRED_STATE_FIELDS = (
    "qc", "qi", "ni", "qs", "qg", "qr", "nr", "nc", "nwfa", "nifa",
)

#: The two 2-D surface emission constants (Registry ``QNWFA2D``/``QNIFA2D``,
#: # kg-1 s-1).  microphysics never writes them.
REQUIRED_SURFACE_FIELDS = ("nwfa2d", "nifa2d")

#: Every ``mp_thompson_aero_*`` scratch slot this adapter draws, in the order
#: it draws them.  Must equal ``woof/core/preflight.py``'s mp=28 aerosol
#: working set exactly -- a slot this adapter creates that preflight does not
#: budget is an unpriced allocation behind the arena gate, and a slot
#: preflight budgets that this adapter never draws is dead reserved memory.
#: ``test_scratch_slots_match_the_preflight_registry_exactly`` pins both
#: directions.
AEROSOL_SCRATCH_SLOTS = (
    "mp_thompson_aero_ncten",
    "mp_thompson_aero_nwfaten",
    "mp_thompson_aero_nifaten",
    "mp_thompson_aero_entry_density",
    "mp_thompson_aero_tau1_density",
    "mp_thompson_aero_nwfa_work_m3",
    "mp_thompson_aero_qcten",
    "mp_thompson_aero_qrten",
    "mp_thompson_aero_nrten",
    "mp_thompson_aero_qiten",
    "mp_thompson_aero_niten",
    "mp_thompson_aero_qvten",
    "mp_thompson_aero_tten",
    "mp_thompson_aero_qsten",
    "mp_thompson_aero_qgten",
    "mp_thompson_aero_ngten",
    "mp_thompson_aero_condensation_rate",
)

#: The fused source networks diagnose their integer entry flags inline.
#: Keep the exported inventory empty for consumers that classify dtypes.
AEROSOL_INT_SCRATCH_SLOTS = ()


def _apply_thompson_aerosol(
        state: DomainState, cfg: RunConfig, dt: float, *,
        refl_10cm_due: bool = False) -> MicrophysicsDiagnostics:
    """One ``mp_physics=28`` call as the configured Thompson generation.

    ``cfg.thompson_version`` (default ``"wrf_461"``) picks the generation:
    every aerosol kernel launched inside compiles for it
    (woof.core.thompson_aerosol_launch.thompson_version_scope).
    """
    from woof.core.thompson_aerosol_launch import thompson_version_scope
    version = getattr(cfg, "thompson_version", "wrf_461")
    with thompson_version_scope(version):
        return _apply_thompson_aerosol_call(
            state, cfg, dt, refl_10cm_due=refl_10cm_due, version=version)


def _wrf39_table_root() -> str:
    """The fork's table root, or a refusal naming how to build it.

    The fork's lookup tables differ in shape and content from v4.6.1's
    (28 snow and graupel entries, no graupel density axis, D0s = 200
    microns), so running the fork's kernels on the v4.6.1 set would index
    past its records; there is no substitute to fall back to.
    """
    from woof.thompson_fork_assets import ensure_thompson_fork_tables

    return str(ensure_thompson_fork_tables())


def _apply_thompson_fork_call(
        state: DomainState, cfg: RunConfig, dt: float, table_root: str, *,
        refl_10cm_due: bool) -> MicrophysicsDiagnostics:
    """One ``mp_physics=28`` call as the operational HRRR fork.

    The whole of the fork's mp_gt_driver runs in one column kernel
    (woof/core/kernels/thompson_fork.cu, woof.core.thompson_fork), in the
    fork's own tendency form and arithmetic, so a column agrees with the
    fork's Fortran word for word (tools/thompson_fork_oracle/
    gpu_column_oracle.py).  The bracket around it is WOOF's: the full theta
    and the Exner function formed as every scheme forms them, the pre-call
    theta saved for ``h_diabatic``, the reflectivity stashed for output and
    moist_physics_finish's clamped theta increment.
    """
    from woof.core.mynn_radiation import wrf_itimestep
    from woof.core.refl import refl_melting_for, stash_refl_10cm
    from woof.core.thompson_aerosol_runtime import load_aerosol_device_tables
    from woof.core.thompson_fork import launch_thompson_fork_column
    from woof.core.thompson_runtime import load_classic_device_tables

    nz, ny, nx = state.p.shape
    thb = state.thb if state.thb.ndim == 3 else state.thb[:, None, None]
    phb = state.phb if state.phb.ndim == 3 else state.phb[:, None, None]
    th = state.scratch((nz, ny, nx), "mp_th")
    pii = state.scratch((nz, ny, nx), "mp_pii")
    dz = state.scratch((nz, ny, nx), "mp_dz8w")
    z8w = state.scratch((nz + 1, ny, nx), "mp_z8w")
    th[...] = thb + state.thp
    from woof.core.microphysics import prepare_microphysics_exner
    prepare_microphysics_exner(state.p, pii)
    z8w[...] = (phb + state.php) / DTYPE(c.G)
    dz[...] = z8w[1:] - z8w[:-1]

    surface_shape = (ny, nx)
    rainnc = state.scratch(surface_shape, "mp_rainnc")
    rainncv = state.scratch(surface_shape, "mp_rainncv")
    snownc = state.scratch(surface_shape, "mp_snownc")
    snowncv = state.scratch(surface_shape, "mp_snowncv")
    graupelnc = state.scratch(surface_shape, "mp_graupelnc")
    graupelncv = state.scratch(surface_shape, "mp_graupelncv")
    sr = state.scratch(surface_shape, "mp_sr")

    table_owner = load_classic_device_tables(_wrf39_table_root(),
                                             version="wrf_39_noaa")
    ccn_table = load_aerosol_device_tables(table_root).ccn_activation_table

    save_pre_mp_theta(state)
    refl = state.scratch((nz, ny, nx), "refl_10cm") if refl_10cm_due \
        else None
    # calc_refl10cm's melting-snow term: off on the fork's first step
    # (itimestep 1, :5374-5379) and wherever WOOF's state policy turns the
    # melting term off (woof.core.refl.refl_melting_for).
    elapsed = getattr(state, "elapsed_seconds", None)
    first_step = (elapsed is not None
                  and wrf_itimestep(float(elapsed), float(dt)) == 1)
    launch_thompson_fork_column(
        th, pii, state.p, state.w, dz, state.qv, state.qc, state.qr,
        state.qi, state.qs, state.qg, state.ni, state.nr, state.nc,
        state.nwfa, state.nifa, state.nwfa2d, th, rainnc, rainncv, snownc,
        snowncv, graupelnc, graupelncv, sr, refl, state.effc, state.effi,
        state.effs, table_owner, ccn_table, dt,
        first_time_step=first_step or not refl_melting_for(state),
        do_refl=bool(refl_10cm_due),
        singular_fall=getattr(cfg, "thompson_fork_snow_fall",
                              "blend") == "wrf_39_noaa")
    if refl_10cm_due:
        stash_refl_10cm(state, refl)
    moist_physics_finish(state, cfg, th, dt)
    return MicrophysicsDiagnostics(
        rainnc=rainnc, rainncv=rainncv, sr=sr,
        snownc=snownc, snowncv=snowncv,
        graupelnc=graupelnc, graupelncv=graupelncv)


def _apply_thompson_aerosol_call(
        state: DomainState, cfg: RunConfig, dt: float, *,
        refl_10cm_due: bool, version: str) -> MicrophysicsDiagnostics:
    """One complete ``mp_physics=28`` microphysics call, in WRF driver order.

    The skeleton is ``woof.core.microphysics._apply_thompson`` with the
    aerosol network substituted; every deviation is annotated with the WRF
    line that forces it.  See the module docstring for the four composition
    properties this function is uniquely responsible for.
    """
    table_root = _thompson_table_root()
    missing = [name for name in REQUIRED_STATE_FIELDS
               if getattr(state, name, None) is None]
    missing += [name for name in REQUIRED_SURFACE_FIELDS
                if getattr(state, name, None) is None]
    if missing:
        raise ValueError(
            "Thompson mp=28 state lacks " + ", ".join(missing))

    from woof.wrf_exact import ENABLED
    if ENABLED and version == "wrf_39_noaa":
        return _apply_thompson_fork_call(state, cfg, dt, table_root,
                                        refl_10cm_due=refl_10cm_due)

    # The frozen mp=8 column masks, reused byte-for-byte.  Imported here
    # (not at module scope) for the same reason _apply_thompson does: it
    # keeps the import graph acyclic and lets a call-recording test
    # monkeypatch the owning module's attribute.
    from woof.core.thompson import (
        launch_graupel_fallout_column_mask,
        launch_hydrometeor_column_mask,
    )
    from woof.core.thompson_aerosol_cold import (
        launch_aa_cold_network_from_owner,
    )
    from woof.core.thompson_aerosol_runtime import (
        device_drop_evaporation_number_table,
        load_aerosol_device_tables,
    )
    from woof.core.thompson_aerosol_sat import (
        launch_aerosol_rain_evaporation,
        launch_aerosol_saturation_adjust,
    )
    from woof.core.thompson_aerosol_sed import (
        launch_aa_cloud_sedimentation,
        launch_aa_final_phase_cleanup,
        launch_aa_graupel_sedimentation,
        launch_aa_ice_sedimentation_accumulate,
        launch_aa_rain_sedimentation_accumulate,
        launch_aa_snow_sedimentation,
        launch_aa_surface_precipitation,
        launch_wrf39_graupel_sedimentation,
        launch_wrf39_ice_sedimentation,
        launch_wrf39_rain_sedimentation,
        launch_wrf39_snow_sedimentation,
        launch_wrf39_warm_snow_boost,
    )
    from woof.core.thompson_aerosol_state import (
        launch_aa_entry_warm_mask,
        launch_aa_graupel_number_finalize,
        launch_aa_graupel_number_init,
        launch_aerosol_effective_radius,
        launch_aerosol_exner,
        launch_aerosol_entry_cloud_number,
        launch_aerosol_entry_snapshot,
        launch_aerosol_micro_columns,
        launch_aerosol_state_finalize,
        launch_aerosol_surface_emission,
        launch_aerosol_vapour_fill,
        launch_aerosol_working_number,
        launch_tau1_density,
        launch_terminal_rain_ice,
        launch_wrf39_graupel_finalize,
        launch_wrf39_graupel_intercept,
        WRF39_INTERCEPT_ENTRY,
        WRF39_INTERCEPT_POST_SOURCE,
        WRF39_INTERCEPT_REFLECTIVITY,
        zero_aerosol_accumulators,
    )
    from woof.core.thompson_aerosol_warm import (
        launch_aerosol_warm_source_network_from_owner,
        launch_ncten_balance,
    )
    from woof.core.thompson_runtime import load_classic_device_tables

    nz, ny, nx = state.p.shape
    thb = state.thb if state.thb.ndim == 3 else state.thb[:, None, None]
    phb = state.phb if state.phb.ndim == 3 else state.phb[:, None, None]
    th = state.scratch((nz, ny, nx), "mp_th")
    pii = state.scratch((nz, ny, nx), "mp_pii")
    temperature = state.scratch((nz, ny, nx), "mp_thompson_temperature")
    dz = state.scratch((nz, ny, nx), "mp_dz8w")
    frozen_reference_density = state.scratch(
        (nz, ny, nx), "mp_thompson_frozen_reference_density")
    frozen_reference_temperature = state.scratch(
        (nz, ny, nx), "mp_thompson_frozen_reference_temperature")
    rain_reference_density = state.scratch(
        (nz, ny, nx), "mp_thompson_rain_reference_density")
    snow_melt_marker = state.scratch(
        (nz, ny, nx), "mp_thompson_snow_melt_marker")
    graupel_melt_marker = state.scratch(
        (nz, ny, nx), "mp_thompson_graupel_melt_marker")
    snow_velocity_boost = state.scratch(
        (nz, ny, nx), "mp_thompson_snow_velocity_boost")
    z8w = state.scratch((nz + 1, ny, nx), "mp_z8w")
    th[...] = thb + state.thp
    # WRF's pi_phy = (p/p1000mb)**rcp is a REAL(4) powf; CuPy's power is
    # CUDA's powf, a different function (the 0 ULP column oracle measured
    # 1 ULP at 15 % of cells).  WOOF's own powf word forms it.
    launch_aerosol_exner(state.p, pii)
    temperature[...] = th * pii
    z8w[...] = (phb + state.php) / DTYPE(c.G)
    dz[...] = z8w[1:] - z8w[:-1]

    surface_shape = (ny, nx)
    rainnc = state.scratch(surface_shape, "mp_rainnc")
    rainncv = state.scratch(surface_shape, "mp_rainncv")
    snownc = state.scratch(surface_shape, "mp_snownc")
    snowncv = state.scratch(surface_shape, "mp_snowncv")
    graupelnc = state.scratch(surface_shape, "mp_graupelnc")
    graupelncv = state.scratch(surface_shape, "mp_graupelncv")
    sr = state.scratch(surface_shape, "mp_sr")

    # Both table owners resolve from the SAME root.  A second root would let
    # tnccn_act and the four classic caches come from different WRF builds.
    # The fork generation reads its own process tables (thompson_contract.
    # FORK_TABLE_ASSETS); its CCN activation table is the v4.6.1 file byte
    # for byte, so the aerosol owner stays on the classic root.
    wrf39 = version == "wrf_39_noaa"
    if wrf39:
        table_owner = load_classic_device_tables(
            _wrf39_table_root(), version=version)
    else:
        table_owner = load_classic_device_tables(table_root)
    aerosol_table_owner = load_aerosol_device_tables(table_root)
    drop_evaporation_number = device_drop_evaporation_number_table(
        table_owner)

    graupel_number_shadow = state.scratch(
        (nz, ny, nx), "mp_thompson_graupel_number_shadow")

    # --- the mp=28 aerosol working set (preflight budgets each of these) ---
    ncten = state.scratch((nz, ny, nx), "mp_thompson_aero_ncten")
    nwfaten = state.scratch((nz, ny, nx), "mp_thompson_aero_nwfaten")
    nifaten = state.scratch((nz, ny, nx), "mp_thompson_aero_nifaten")
    entry_density = state.scratch(
        (nz, ny, nx), "mp_thompson_aero_entry_density")
    tau1_density = state.scratch(
        (nz, ny, nx), "mp_thompson_aero_tau1_density")
    nwfa_work_m3 = state.scratch(
        (nz, ny, nx), "mp_thompson_aero_nwfa_work_m3")
    qcten = state.scratch((nz, ny, nx), "mp_thompson_aero_qcten")
    # THE RAIN AND ICE ACCUMULATORS (property 1c), v4.6.1 generation only:
    # the fork keeps its in-place rain and ice and holds the entry ice
    # number for the cleanup's melt credit instead.
    accumulate_rain_ice = not wrf39
    if accumulate_rain_ice:
        qrten = state.scratch((nz, ny, nx), "mp_thompson_aero_qrten")
        nrten = state.scratch((nz, ny, nx), "mp_thompson_aero_nrten")
        qiten = state.scratch((nz, ny, nx), "mp_thompson_aero_qiten")
        niten = state.scratch((nz, ny, nx), "mp_thompson_aero_niten")
        rain_ice_tendencies = dict(qrten=qrten, nrten=nrten, qiten=qiten,
                                   niten=niten)
        ni_entry = state.ni
        # THE VAPOUR AND TEMPERATURE ACCUMULATORS, v4.6.1 generation only.
        # WRF's qvten/tten (:1668-1669 zero) collect the sources (:2982,
        # :3164-3179), the condensation (:3479, :3483) and the rain
        # evaporation (:3563, :3566) as REAL tendencies, and every block
        # re-forms its working state from them: MAX(1.E-10, qv1d +
        # DT*qvten) and t1d + DT*tten.  So state.qv stays qv1d from the
        # networks through the rain evaporation, and the running vapour is
        # formed once after it; temperature is re-formed by every writer as
        # t1d + DT*tten, t1d = th*pii.  Adding each block's increment to the
        # running state instead rounded once per block and moved the
        # post-condensation ssatw whose sign decides the rain evaporation
        # (:3501) at levels the adjustment brought to saturation.
        qvten = state.scratch((nz, ny, nx), "mp_thompson_aero_qvten")
        tten = state.scratch((nz, ny, nx), "mp_thompson_aero_tten")
        vapor_tendencies = dict(qvten=qvten, tten=tten)
        vapor_block = dict(qvten=qvten, tten=tten, theta=th, exner=pii)
        # THE SNOW AND GRAUPEL ACCUMULATORS, v4.6.1 generation only.  WRF's
        # qsten/qgten/ngten (:1674-1677 zero) collect the sources
        # (:3094-3161) and the snow and graupel fallout (:3871-3937) as REAL
        # tendencies; the fallout forms its working content as qs1d +
        # qsten*DT (:3257) and qg1d + qgten*DT (:3283), and :4054-4059
        # applies each sum once.  So state.qs, state.qg and the private
        # graupel number stay the entry state until the fallout.  Applying
        # the sources in place and the fallout on top of that rounded twice
        # where WRF rounds once (632 qs and 417 qg cells of the column
        # oracle at strict dt 20 s).
        qsten = state.scratch((nz, ny, nx), "mp_thompson_aero_qsten")
        qgten = state.scratch((nz, ny, nx), "mp_thompson_aero_qgten")
        ngten = state.scratch((nz, ny, nx), "mp_thompson_aero_ngten")
        frozen_tendencies = dict(qsten=qsten, qgten=qgten, ngten=ngten)
    else:
        qrten = nrten = qiten = niten = None
        rain_ice_tendencies = {}
        qvten = tten = None
        vapor_tendencies = {}
        vapor_block = {}
        qsten = qgten = ngten = None
        frozen_tendencies = {}
        ni_entry = state.scratch((nz, ny, nx), "mp_thompson_aero_ni_entry")
    condensation_rate = state.scratch(
        (nz, ny, nx), "mp_thompson_aero_condensation_rate")

    # (1) ACCUMULATOR ZEROING, :1679-1681.  Explicit and unconditional: these
    # are persistent scratch slots and every one of them is read before it is
    # written somewhere in the call (the balance limiter OVERWRITES ncten, the
    # working-number refresh READS nwfaten, the terminal apply reads all
    # three), so a carried-over value from the previous step is a live
    # physics input, not a harmless stale buffer.
    zero_aerosol_accumulators(ncten, nwfaten, nifaten)
    # :1670, and the same reasoning: the cloud-water accumulator (property
    # 1b) is persistent scratch read by every stage after the networks.
    qcten.fill(DTYPE(0.0))
    for accumulator in (qrten, nrten, qiten, niten, qvten, tten, qsten,
                        qgten, ngten):
        if accumulator is not None:
            accumulator.fill(DTYPE(0.0))
    dt32 = DTYPE(dt)

    def working(entry, tendency, out):
        """WRF's working value ``X1d + Xten*DT`` into ``out``, product
        rounded before the add (two CuPy kernels, no fused multiply-add)."""
        cp.multiply(tendency, dt32, out=out)
        cp.add(entry, out, out=out)
        return out

    def working_cloud(out):
        """WRF's working cloud ``qc1d + qcten*DT`` into ``out``.

        Two separate CuPy kernels, so the product is rounded to float32
        before the add, as the gfortran oracle (no FMA) rounds it.
        """
        cp.multiply(qcten, dt32, out=out)
        cp.add(state.qc, out, out=out)
        return out

    # Lifetime-alias the zero/one entry-graupel marker with the held-
    # temperature buffer, exactly as mp=8 does: the column mask consumes it
    # before the saturation adjustment overwrites every element with the
    # reference temperature snow fallout needs.
    cp.greater(
        state.qg, DTYPE(1.0e-12), out=frozen_reference_temperature)

    # (2) THE WARM ENTRY MASK, captured on the ENTRY temperature and BEFORE
    # the cold network writes its latent heating in place.  See the module
    # docstring; this single line is the disjointness guarantee both kernel
    # headers depend on.  The warm network consumes it and overwrites the
    # same buffer with WRF's held ``prr_gml > 0`` decision.
    # The v4.6.1 mask also carries WRF's melting level (:1971-2013): where
    # a level at exactly 273.15 K has no warmer level at or above it, WRF
    # keeps twet = temp (thompson_aa_entry_warm_mask).
    if wrf39:
        cp.greater_equal(
            temperature, DTYPE(273.15), out=graupel_melt_marker)
    else:
        launch_aa_entry_warm_mask(temperature, graupel_melt_marker)

    # GRAUPELNCV is a current-call diagnostic with no earlier species kernel
    # to reset it.
    graupelncv.fill(DTYPE(0.0))
    save_pre_mp_theta(state)

    # ---- 1. entry snapshot (:1795-1812) and entry droplet diagnosis -------
    # (:1826-1848).  The second call ZEROES state.qc and state.nc wherever
    # qc <= R1, which is what makes state.qc and state.nc a legitimate
    # "qc1d" and "nc1d" for every later kernel.
    # The fused source networks diagnose aerosol concentrations and the
    # droplet distribution inline.  Only entry density and the qc/nc
    # rewrite have consumers here; six diagnostic volumes were dead stores.
    launch_aerosol_entry_snapshot(
        temperature, state.p, state.qv, state.nwfa, state.nifa,
        entry_density, None, None)
    launch_aerosol_entry_cloud_number(
        state.qc, state.nc, entry_density,
        None, None, None, None)
    # The rest of WRF's entry rewrite (:1871-1872, :1900-1901, :1911 and
    # :1941-1942): ice, rain, snow and graupel whose entry mixing ratio is at
    # or below R1 are ZEROED, mass and number, before any process runs, and
    # in every column, because mp_gt_driver copies the rewritten 1-D arrays
    # back whether or not the column had microphysics.  Every reader that
    # gates on q > R1 first cannot tell; the ones that do not are the ice
    # nucleation (:2627 takes ni(k), which is R2 on such a level), the
    # terminal numbers and the reflectivity.  An orphan number (q <= R1,
    # n > 0), which advection leaves at cloud edges and an analysis
    # increment can leave anywhere, otherwise suppressed nucleation and was
    # carried through the call into the state and the echo (up to 45 dB on
    # saved real-data analysis columns against WRF v4.6.1's own Fortran,
    # tools/thompson_real_column_parity).  The mask is taken before either
    # array is written; ``cp.where`` writes a +0.0, as WRF does.
    for mass, number in ((state.qi, state.ni), (state.qr, state.nr),
                         (state.qs, None), (state.qg, None)):
        present = mass > DTYPE(R1)
        if number is not None:
            number[...] = cp.where(present, number, DTYPE(0.0))
        mass[...] = cp.where(present, mass, DTYPE(0.0))
    # The final phase cleanup's melt credit ``ncten += ni1d*odt`` (:3949)
    # reads the zeroed entry ice number.  On the accumulator path state.ni
    # IS that number for the whole call; the fork holds a copy because its
    # sources write state.ni in place.
    if not accumulate_rain_ice:
        ni_entry[...] = state.ni
    # WRF's column exit (:1646, :1827-1990, :2020): a column whose entry
    # condensate is all at or below R1 and which is nowhere supersaturated
    # over ice leaves mp_thompson before the source loop and never reaches
    # the terminal apply, so its vapour is not floored at 1.E-10 (:3974) and
    # its aerosol is not clamped (:3979-3982).  The flag is taken on the
    # entry state here and read by the terminal apply (step 13).
    micro_columns = state.scratch((ny, nx), "mp_thompson_micro_columns")
    launch_aerosol_micro_columns(
        state.qc, state.qi, state.qr, state.qs, state.qg,
        temperature, state.p, state.qv, micro_columns)

    # ---- 2. classic graupel number (is_hail_aware false for 8 and 28) -----
    # The fork carries no graupel number: its intercept is a column pass
    # over graupel content and supercooled rain (fork :2031-2054), and the
    # slot holds that N0_exp per level for the two source networks.
    if wrf39:
        launch_wrf39_graupel_intercept(
            state.qg, state.qr, state.nr, temperature, state.p, state.qv,
            graupel_number_shadow, mode=WRF39_INTERCEPT_ENTRY)
    else:
        # mp_gt_driver :1266-1281, mp=28's own transcription.
        launch_aa_graupel_number_init(
            state.qg, temperature, state.p, state.qv,
            graupel_number_shadow)

    # ---- 3. the cold source network --------------------------------------
    launch_aa_cold_network_from_owner(
        state.qi, state.ni, state.qs, state.qg, state.qr, state.nr,
        state.qc, temperature, state.p, state.qv,
        state.nc, state.nwfa, state.nifa,
        ncten, nwfaten, nifaten,
        graupel_number_shadow, snow_velocity_boost, table_owner, dt,
        qcten=qcten, **rain_ice_tendencies, **vapor_tendencies,
        **frozen_tendencies)

    # The fork's singular snow fall (thompson_fork_snow_fall = "wrf_39_noaa")
    # starts every level the sources find at or above 0 C at vts_boost 1.5
    # (fork :2151); the graupel melt marker still holds the entry warm mask
    # here, and the warm network overwrites it next.
    singular_snow_fall = wrf39 and getattr(
        cfg, "thompson_fork_snow_fall", "blend") == "wrf_39_noaa"
    if singular_snow_fall:
        launch_wrf39_warm_snow_boost(graupel_melt_marker, snow_velocity_boost)

    # ---- 4. the warm source network --------------------------------------
    launch_aerosol_warm_source_network_from_owner(
        state.qc, state.qr, state.nr, state.qs, state.qg,
        graupel_number_shadow, graupel_melt_marker, snow_melt_marker,
        temperature, state.p, state.qv,
        state.nc, state.nwfa, state.nifa,
        ncten, nwfaten, nifaten, table_owner, dt, qcten=qcten,
        **({} if not accumulate_rain_ice else dict(qrten=qrten, nrten=nrten)),
        **vapor_tendencies, **frozen_tendencies)

    # ---- 5. the ncten balance limiter, ONCE (:2996-3019) ------------------
    # After every ncten source, before the saturation adjustment's pnc_wcd.
    # It OVERWRITES ncten where a clamp fires; running it twice is finite,
    # plausible and wrong.  ``entry_density`` is WRF's :1802 density, not a
    # density rediagnosed from the mutated temperature and vapour.  It reads
    # the entry cloud and the post-source working cloud; the condensation
    # rate's slot holds the latter until the adjustment writes that rate.
    launch_ncten_balance(
        state.qc, working_cloud(condensation_rate), state.nc,
        entry_density, ncten, dt)

    # The fork's fallout intercept (fork :3110-3133) is diagnosed from the
    # post-source state, before the condensation and rain evaporation move
    # the temperature and the rain: the networks' last reader of the entry
    # intercept has run, so the slot takes the post-source one.
    if wrf39:
        launch_wrf39_graupel_intercept(
            state.qg, state.qr, state.nr, temperature, state.p, state.qv,
            graupel_number_shadow, mode=WRF39_INTERCEPT_POST_SOURCE)

    # ---- 6. the three column masks, unchanged from mp=8 -------------------
    # The cloud column's own mask is taken after the saturation adjustment
    # instead (step 10): WRF's cloud fallout gate reads L_qc as :3485 leaves
    # it, not as the sources left it.
    # WRF's L_qr here is the post-source (qr1d + qrten*DT) > R1; on the
    # accumulator path that working rain sits in the condensation rate's
    # slot until the adjustment writes the rate.
    launch_hydrometeor_column_mask(
        working(state.qr, qrten, condensation_rate)
        if accumulate_rain_ice else state.qr, rainncv)
    # ANY(L_qg) at :3903: entry graupel (:1917) whose post-source content
    # qg1d + qgten*DT is above R1 (:3283-3302 clears L_qg and never sets
    # it); with the accumulator that content is formed in the condensation
    # rate's slot, which the rain mask above has finished reading.
    launch_graupel_fallout_column_mask(
        frozen_reference_temperature,
        working(state.qg, qgten, condensation_rate)
        if qgten is not None else state.qg, sr)

    # ---- 7. the working aerosol number (:3189-3193, :3211) ----------------
    # The TAU+1 density, recomputed from the post-source temperature and
    # vapour -- a genuinely different density from ``entry_density``.
    # With the vapour accumulator state.qv is still qv1d; the working vapour
    # qv1d + DT*qvten is formed in the working-aerosol slot, which the next
    # launch overwrites (the density kernel floors it, :3192).
    tau1_vapor = (working(state.qv, qvten, nwfa_work_m3)
                  if qvten is not None else state.qv)
    launch_tau1_density(temperature, state.p, tau1_vapor, tau1_density)
    launch_aerosol_working_number(
        state.nwfa, nwfaten, tau1_density, dt, nwfa_work_m3)

    # ---- 8/9. condensation + CCN activation, then rain evaporation --------
    # ``cloud_presence`` carries WRF's L_qc(k) out of the adjustment for the
    # cloud fallout's column gate.  It is the TAU+1 density's slot: the
    # working aerosol number above was that density's last reader.
    cloud_presence = tau1_density
    # WRF passes w1d(k) = w(i,k,j) once at mp_gt_driver:1224 with no
    # averaging, so the lower full-level slice is the exact analogue.
    # THE PHASE CLEANUP'S LATENT-HEAT FACTORS (v4.6.1 generation).  :3953
    # and :3964 multiply by the ocp(k) and lvap(k) WRF last formed: the
    # TAU+1 refresh's (:3203, :3207), re-formed by the rain evaporation
    # where its gate passes (:3517, :3519).  The adjustment writes :3207's
    # ocp into the entry density's slot (the ncten balance was that
    # density's last reader); the rain evaporation overwrites it where its
    # gate passes and leaves lvap(k) in the condensation rate's slot, whose
    # prw_vcd it is the last reader of.  Forming the heat from the final
    # vapour and temperature instead moved theta wherever ice melted or
    # cloud froze.
    heat_ocp = entry_density if accumulate_rain_ice else None
    heat_lvap = condensation_rate if accumulate_rain_ice else None
    launch_aerosol_saturation_adjust(
        temperature, state.p, state.qv, state.qc, state.nc, ncten, nwfaten,
        nwfa_work_m3, state.w[:-1],
        aerosol_table_owner.ccn_activation_table, drop_evaporation_number,
        dt,
        reference_density=frozen_reference_density,
        reference_temperature=frozen_reference_temperature,
        condensation_rate=condensation_rate,
        cloud_presence=cloud_presence, qcten=qcten, heat_ocp=heat_ocp,
        **vapor_block)
    # TWO DENSITIES, AND WRF USES BOTH.  :3242-3243 forms the working rain
    # mass and number from the TAU+1 density diagnosed at :3193 -- BEFORE the
    # condensation block -- and :3384-3388 freezes ilamr/N0_r from them.
    # :3490 then OVERWRITES rho(k) inside the condensation loop, and
    # :3505-3520's orho/rhof/vsc2/rvs all read that post-condensation value.
    # Passing only the post-condensation density (the mp=8 kernel's single
    # density, and this adapter's behaviour before WP-12a) scales prv_rev and
    # pnr_rev by rho_post/rho_pre.  ``launch_aerosol_saturation_adjust``
    # writes exactly WRF's :3193 density into ``reference_density``,
    # unconditionally and at every level, before its own gate -- so the
    # buffer above already holds it and no extra scratch slot is needed.
    launch_aerosol_rain_evaporation(
        state.qr, state.nr, temperature, state.p, state.qv, nwfaten, dt,
        reference_density=rain_reference_density,
        graupel_melt_marker=graupel_melt_marker,
        condensation_rate=condensation_rate,
        entry_density=frozen_reference_density,
        **({} if not accumulate_rain_ice else dict(
            qrten=qrten, nrten=nrten, heat_ocp=heat_ocp,
            tau1_temperature=frozen_reference_temperature)),
        **vapor_block)
    if qvten is not None:
        # WRF writes no qvten after :3563, so the running vapour qv1d +
        # DT*qvten (unfloored; :3974 floors it) is formed here, once, for
        # every later reader.  The spent accumulator is its own scratch
        # (the condensation rate's slot now carries lvap(k)) and is zeroed
        # so nothing can add it to the vapour a second time.
        cp.multiply(qvten, dt32, out=qvten)
        cp.add(state.qv, qvten, out=state.qv)
        qvten.fill(DTYPE(0.0))

    # WRF's rho(k) as the terminal apply finds it: :3193, rewritten at :3490
    # and :3572 wherever those blocks ran, and untouched after :3574.
    # temperature and qv hold exactly WRF's temp/qv of that moment HERE, before
    # the fallout and the phase cleanup add their latent heat.  Handing the
    # terminal apply the ENTRY density instead moved its 2/rho droplet floor,
    # its nu_c selector and its Nt_c_max/rho ceiling: up to 4e-5 on nc at 400
    # to 1,000 levels of every saved 19,600-column real-data frame against
    # WRF v4.6.1.  It is carried in the working-aerosol slot, which the
    # saturation adjustment was its last reader of; the TAU+1 slot keeps
    # WRF's :3193 density.
    terminal_density = nwfa_work_m3
    launch_tau1_density(temperature, state.p, state.qv, terminal_density)

    # ---- 10. fallout: cloud from mp=28, the other four REUSED from mp=8 ---
    # :3645 ``if (ANY(L_qc .eqv. .true.))`` reads L_qc as the sources set it
    # (:3215-3223) and the adjustment cleared it (:3485).  The adjustment
    # never SETS it, so a column whose only cloud condensed this step keeps
    # that cloud where it formed; the post-source mask sedimented it.
    if wrf39:
        # The fork has no ANY(L_qc) guard and never clears L_qc after the
        # adjustment (both are v4.6.1 additions, :3485 and :3646): its cloud
        # fallout runs in every column on the cloud the adjustment left,
        # newly condensed cloud included.  The condensation rate's slot
        # (the rain evaporation was its last reader) takes the working cloud.
        launch_hydrometeor_column_mask(
            working_cloud(condensation_rate), snowncv)
    else:
        launch_hydrometeor_column_mask(cloud_presence, snowncv)
    # Correct mode, finding WB1 (RunConfig.thompson_cloud_surface_booking,
    # woof/core/thompson_cloud_booking.py): the cloud water this fallout
    # takes out of the lowest level is booked in CLOUDNC instead of
    # vanishing.  The fallout adds its divergence to qcten, so the column sum
    # of qcten on the kernel's own density, before minus after, times DT is
    # that water.  Exact mode (the key off) issues exactly the launcher
    # sequence above and below, nothing more.
    book_cloud = getattr(cfg, "thompson_cloud_surface_booking", False) is True
    if book_cloud:
        from woof.core.thompson_cloud_booking import (
            book_settled_cloud_tendency, cloud_column_water)
        cloud_before = cloud_column_water(
            qcten, temperature, state.p, state.qv, dz)
    launch_aa_cloud_sedimentation(
        state.qc, state.nc, ncten, temperature, state.p, state.qv,
        state.w[:-1], dz, dt,
        reference_density=frozen_reference_density,
        rain_active_columns=rainncv, cloud_active_columns=snowncv,
        qcten=qcten)
    if book_cloud:
        book_settled_cloud_tendency(state, cfg, cloud_before, qcten, dt,
                                    temperature, state.p, state.qv, dz)
    if wrf39:
        launch_wrf39_ice_sedimentation(
            state.qi, state.ni, temperature, state.p, state.qv, dz,
            rainnc, rainncv, snownc, snowncv, dt,
            reference_density=frozen_reference_density)
    else:
        # :3664-3698 and :3838-3870 in WRF's tendency form; the size bound
        # waits for the terminal apply, after the cleanup's freeze.
        # The surface totals are added once, in mp_gt_driver's order
        # (:1294-1308), by launch_aa_surface_precipitation after the rain
        # fallout; until then each pass leaves its own column total in one
        # of the seven surface slots: pptice in snowncv, pptsnow in rainncv
        # (the rain column mask's last reader is this ice pass), pptgraul in
        # graupelncv and pptrain in sr (the graupel column mask's last
        # reader is the graupel pass).
        launch_aa_ice_sedimentation_accumulate(
            state.qi, state.ni, qiten, niten, temperature, state.p,
            state.qv, dz, rainnc, rainncv, snownc, snowncv, dt,
            reference_density=frozen_reference_density,
            rain_active_columns=rainncv, export_surface=True)
    # Melting snow falls at its speed blended with the rain fall speed
    # vtrk(k) by SR = rs/(rs+rr) (:3722-3724), and vtrk(k) is the rain
    # pass's own (:3612-3634): a level whose rr(k) is at or below R1 takes
    # the speed from above, and a column with no L_qr has none.  So the
    # blend reads the rain fallout's own reference density, with WRF's L_qr
    # and :3568 rewrite in it, below.
    if wrf39:
        launch_wrf39_snow_sedimentation(
            state.qs, temperature, state.p, state.qv, dz,
            rainnc, rainncv, snownc, snowncv, dt,
            reference_density=frozen_reference_density,
            reference_temperature=frozen_reference_temperature,
            snow_melt_marker=snow_melt_marker,
            melt_rain_qr=state.qr, melt_rain_nr=state.nr,
            melt_rain_density=rain_reference_density,
            velocity_boost=snow_velocity_boost,
            singular_fall=singular_snow_fall)
    else:
        # :3257-3262, :3313-3353, :3699-3733, :3871-3902, mp=28's own pass.
        # The melting blend forms the rain pass's vtrk(k) and rr(k) from the
        # entry rain, its tendency so far and the evaporation's density
        # export, as the rain fallout below does.
        launch_aa_snow_sedimentation(
            state.qs, temperature, state.p, state.qv, dz, rainncv, dt,
            reference_density=frozen_reference_density,
            reference_temperature=frozen_reference_temperature,
            snow_melt_marker=snow_melt_marker,
            velocity_boost=snow_velocity_boost,
            qr1d=state.qr, nr1d=state.nr, qrten=qrten, nrten=nrten,
            rain_density=rain_reference_density, qsten=qsten)
    if wrf39:
        # The fork's graupel falls at least as fast as the rain above 0 C
        # (fork :3501-3502), so it reads the rain pass's inputs exactly as
        # the melting-snow blend above does.  Its L_qg is the post-source
        # content alone (fork :3040-3046) and it has no ANY(L_qg) guard, so
        # graupel the sources made in a column that entered without any
        # falls this call: the column mask is the post-source graupel's.
        launch_hydrometeor_column_mask(state.qg, sr)
        launch_wrf39_graupel_sedimentation(
            state.qg, graupel_number_shadow, temperature, state.p,
            state.qv, dz, rainnc, rainncv, graupelnc, graupelncv, dt,
            reference_density=frozen_reference_density,
            melt_rain_qr=state.qr, melt_rain_nr=state.nr,
            melt_rain_density=rain_reference_density,
            active_columns=sr)
    else:
        # :3283-3303, :3370-3376, :3740-3773, :3903-3937, mp=28's own pass:
        # the slope from the number WRF diagnoses from rg(k), the private
        # ng1d evolved by the number fallout alongside.
        launch_aa_graupel_sedimentation(
            state.qg, graupel_number_shadow, temperature, state.p, state.qv,
            dz, graupelncv, dt,
            reference_density=frozen_reference_density,
            active_columns=sr, rain_density=rain_reference_density,
            qgten=qgten, ngten=ngten)
    # THE THIRD DENSITY DECISION, and it is not the same as the one above.
    # This kernel builds WRF's rr(k)/nr(k) (:3794-3795) as qr*rho / nr*rho
    # from the buffer below, and WRF builds those at :3237-3238 from the :3193
    # TAU+1 density EVERYWHERE and rebuilds them at :3568-3570 from the :3490
    # post-condensation density ONLY inside the :3501-3502 gate.
    # ``rain_reference_density`` is exactly that level-wise mixture: the rain
    # evaporation kernel writes ``entry_density`` into it by default and
    # overwrites with its own rho once all three of WRF's gates pass.  See
    # note 5b in the module docstring for what handing it a single density
    # costs (5.165e-04 on aero-drop-evap's RAINNC).
    if wrf39:
        # The fork keeps its in-place rain (MP28_ACCUMULATOR_REWORK) and
        # counts surface rain above R1*10 (fork :3556, audit T21), where
        # v4.6.1 counts it above R1*1000 (:3817); every other statement is
        # thompson.cu's in-place rain-presence pass.
        launch_wrf39_rain_sedimentation(
            state.qr, state.nr, temperature, state.p, state.qv, dz,
            rainnc, rainncv, dt, reference_density=rain_reference_density)
    else:
        # :3611-3640 and :3790-3812 in WRF's tendency form; the size bounds
        # wait for the terminal apply, like the ice above.
        launch_aa_rain_sedimentation_accumulate(
            state.qr, state.nr, qrten, nrten, temperature, state.p,
            state.qv, dz, rainnc, sr, dt,
            reference_density=rain_reference_density,
            export_surface=True)
        launch_aa_surface_precipitation(
            rainnc, rainncv, snownc, snowncv, graupelnc, graupelncv, sr)

    # ---- 11. number-conserving phase cleanup (:3943-3966) -----------------
    launch_aa_final_phase_cleanup(
        state.qc, state.qi, state.ni, temperature,
        state.nc, ni_entry, ncten, state.p, state.qv, dt, qcten=qcten,
        **({} if not accumulate_rain_ice else dict(
            qiten=qiten, niten=niten, tten=tten, heat_ocp=heat_ocp,
            heat_lvap=heat_lvap, theta=th, exner=pii)))

    # ---- 12. classic graupel-number finalize ------------------------------
    if wrf39:
        launch_wrf39_graupel_finalize(state.qg)
    else:
        # :4058-4077 on rho(k) as the terminal apply finds it.
        launch_aa_graupel_number_finalize(
            state.qg, graupel_number_shadow, terminal_density)

    # ---- 13. THE single terminal apply and clamp (:3972-4021) -------------
    # :3975, `qc1d(k) = qc1d(k) + qcten(k)*DT`: the cloud water's one
    # rounding (property 1b).  The finalize below zeroes it at or below R1.
    cp.multiply(qcten, dt32, out=condensation_rate)
    cp.add(state.qc, condensation_rate, out=state.qc)
    if accumulate_rain_ice:
        # :4023-4053, the rain and ice accumulators' one application and
        # WRF's size bounds, on the terminal density.
        launch_terminal_rain_ice(
            state.qr, state.nr, state.qi, state.ni, qrten, nrten, qiten,
            niten, terminal_density, micro_columns, dt)
    # The only place in the whole call that writes nc/nwfa/nifa from the
    # accumulators.  Each output aliases its own input, which the kernel
    # supports (every thread reads its element before writing it).  It
    # skips the columns WRF left at :2020 and floors the vapour of every
    # other level at 1.E-10 (:3974).
    launch_aerosol_state_finalize(
        state.qc, state.nc, state.nwfa, state.nifa,
        ncten, nwfaten, nifaten, terminal_density, dt,
        state.nc, state.nwfa, state.nifa,
        qv=state.qv, micro_columns=micro_columns)

    # ---- 14. reflectivity, if due ----------------------------------------
    if refl_10cm_due:
        from woof.core.refl import compute_and_stash_refl_10cm
        # calc_refl10cm (:5710) takes no nc argument and never re-reads rc
        # after :5764, so cloud water and droplet number contribute exactly
        # zero: this is the mp=8 path with the mp=8 arguments.
        if wrf39:
            # The fork's calc_refl10cm diagnoses its own graupel intercept
            # from the final state (fork :5486-5510); handed over as the
            # graupel number of that distribution, the shared column forms
            # the fork's graupel echo (thompson_aerosol_state.cu, mode 2).
            launch_wrf39_graupel_intercept(
                state.qg, state.qr, state.nr, temperature, state.p,
                state.qv, graupel_number_shadow,
                mode=WRF39_INTERCEPT_REFLECTIVITY)
        compute_and_stash_refl_10cm(
            state, cfg, temperature, state.p,
            thompson_graupel_number=graupel_number_shadow)

    # ---- 15. effective radii, with the PROGNOSTIC droplet number ----------
    # The kernel applies WRF's mp_gt_driver:1476-1478 clamps and then the
    # metre->micron multiply, so state.effc/effi/effs receive woof's
    # radiation-facing MICRON contract directly.
    launch_aerosol_effective_radius(
        temperature, state.p, state.qv, state.qc, state.nc,
        state.qi, state.ni, state.qs,
        state.effc, state.effi, state.effs)

    # ---- 15b. mp_gt_driver's negative-vapour fill (:1436-1445) ----------
    # After the reflectivity and effective radii (WRF hands them the
    # unfilled qv1d) and before moist_physics_finish.  Both generations:
    # the WRFV3.9 fork's driver carries the same loop word for word
    # (NOAA-EMC/HRRR v4.1.21 module_mp_thompson.F:1440-1453, sha256
    # 4d600111...; same no-microphysics exit :1938 and 1.E-10 floor :3690),
    # which the strict fork column (kernels/thompson_fork.cu) already runs;
    # this is the default-mode fork call reaching it too.
    launch_aerosol_vapour_fill(state.qv)

    # ---- 16. surface emission: LAST aerosol write, deliberately unclamped -
    # mp_gt_driver:1310-1327 runs it AFTER mp_thompson has applied its
    # terminal ceiling, so between here and the next call's entry pack
    # (:1805-1806) nwfa/nifa may legitimately exceed 9999.E6.  Adding a
    # ceiling here would silently change the boundary-layer aerosol budget on
    # every step.
    launch_aerosol_surface_emission(
        state.nwfa, state.nifa, state.nwfa2d, state.nifa2d, dt)

    # ---- 17. the unchanged prep/finish bracket and the SR diagnostic ------
    th[...] = temperature / pii
    moist_physics_finish(state, cfg, th, dt)
    # mp_gt_driver:1308, transcribed exactly:
    #     SR(i,j) = (pptsnow + pptgraul + pptice)/(RAINNCV(i,j)+1.e-12)
    # ``snowncv`` is :1301's ``pptsnow + pptice`` and ``graupelncv`` is
    # :1305's ``pptgraul``, so the numerator is WRF's.  The denominator
    # carries WRF's OWN epsilon rather than a guarded division: WRF has no
    # threshold and no MIN here, and the epsilon is not a divide-by-zero
    # guard that a ``where`` could stand in for -- it is part of the value.
    # RAINNCV is pptrain plus the same three frozen terms, so an all-frozen
    # column has SR = f/(f+1e-12) < 1 in WRF, and a guarded f/f returns
    # exactly 1.  On the committed fixtures that is worth 3.701e-04
    # (aero-ice-demott-dep, RAINNCV 2.70e-09 mm) and 3.113e-04
    # (aero-ice-koop, RAINNCV 3.21e-09 mm) -- two orders above the port's
    # 2e-06 gate, on a field the WRF history stream publishes.  The MIN is
    # dropped with it: the numerator is a subset of the denominator's terms,
    # all non-negative, so the ratio cannot exceed 1 and WRF does not clamp
    # it.  Pinned by tests/test_thompson_aerosol_adapter.py::
    # test_the_sr_diagnostic_is_wrfs_epsilon_quotient_not_a_guarded_ratio.
    # The v4.6.1 path's SR is launch_aa_surface_precipitation's, in the
    # driver's own order of addition, (pptsnow + pptgraul + pptice).
    if wrf39:
        frozen = snowncv + graupelncv
        sr[...] = frozen / (rainncv + DTYPE(1.0e-12))
    return MicrophysicsDiagnostics(
        rainnc=rainnc, rainncv=rainncv, sr=sr,
        snownc=snownc, snowncv=snowncv,
        graupelnc=graupelnc, graupelncv=graupelncv)


def wrf39_repair_negative_aerosol(field) -> int:
    """The WRF 3.9 fork's start-time repair of negative aerosol numbers.

    NOAA-EMC/HRRR v4.1.21 module_mp_thompson.F:499-512 (QNWFA) and :555-566
    (QNIFA), audit T25, run in thompson_init when the analysed profile is
    present: walking each column upward, a negative number takes the larger
    of its (already repaired) neighbour below and its neighbour above, then
    is floored at zero; the end levels use themselves as the missing
    neighbour.  WRF v4.6.1 has no such repair.  ``field`` is ``(nz, ny, nx)``
    and is edited in place; ``max`` is exact, so the result is the fork's
    word for word.  Returns how many cells were negative.
    """
    import cupy as cp

    negative = int(cp.count_nonzero(field < 0))
    if negative == 0:
        return 0
    nz = field.shape[0]
    zero = field.dtype.type(0)
    for k in range(nz):
        below = field[max(k - 1, 0)]
        above = field[min(k + 1, nz - 1)]
        repaired = cp.maximum(cp.maximum(below, above), zero)
        field[k] = cp.where(field[k] < 0, repaired, field[k])
    return negative


def thompson_aerosol_init_fill(state: DomainState, cfg: RunConfig) -> dict:
    """``thompson_init``'s synthetic CCN/IN profile, ONCE per domain.

    Invoke from the physics INIT path, never per step.  WRF calls
    ``thompson_init`` from ``phys/module_physics_init.F:4517-4544`` at domain
    construction; nothing in ``mp_gt_driver`` ever refills the profile, and a
    per-step call would overwrite an advected, scavenged aerosol field with
    the synthetic one every step while leaving every bound intact.

    Returns ``{'ccn': bool, 'in': bool}`` -- which of the two independent
    fills ran.  A domain can legitimately get one and not the other:
    ``thompson_init`` makes the CCN decision at :493 and the IN decision at
    :530 from two separate ``MAXVAL`` reductions.

    Under ``wrf_39_noaa``, the fork's :587-603 also replaces surface CCN
    emission from the lowest-level analysed number on every domain start,
    including starts with both profiles present. The receipt's booleans
    still describe only the synthetic profile fills.

    THE HEIGHT FIELD.  ``hgt`` is ``z8w[:nz]``, the FULL (w) level
    geopotential height above sea level.  WRF's argument is named ``z_at_q``
    but ``dyn_em/start_em.F:870-876`` fills it from the Z-staggered
    ``ph_2 + phb``, so ``hgt(i,1,j)`` is the terrain elevation the ABSOLUTE
    1000 m / 2500 m ``h_01`` thresholds test.  See
    ``woof.core.thompson_aerosol_state``'s module docstring.

    THE TILE-BOUND DIFFERENCE, stated rather than silently absorbed.  WRF's
    presence test reduces over ``nwfa(its:ite-1,:,jts:jte-1)`` (:489/:526)
    while its fill loop covers ``i = its..min(ide-1,ite)`` (:499-500) -- the
    test excludes the last mass column and row that the fill then writes.
    That asymmetry is in WRF, not a transcription slip.  woof's ``(ny, nx)``
    arrays are the mass grid with no staggered extra, so
    :func:`~woof.core.thompson_aerosol_state.aerosol_profile_needs_fill`
    reduces over the WHOLE mass grid and therefore INCLUDES the one column
    and one row WRF's test omits.  The two can only disagree if a domain has
    aerosol in exactly that outermost column/row and nowhere else, in which
    case woof declines the fill and WRF performs it.  woof's behaviour is
    the safer of the two (it never overwrites real aerosol data), and the
    difference is unreachable for the only supported v1 ingest, which is
    all-zero or all-filled.
    """
    if cfg.mp_physics != 28:
        raise ValueError(
            "thompson_aerosol_init_fill is the mp_physics=28 profile fill, "
            f"got mp_physics={cfg.mp_physics}")
    missing = [name for name in ("nwfa", "nifa", "nwfa2d")
               if getattr(state, name, None) is None]
    if missing:
        raise ValueError(
            "mp=28 aerosol profile fill needs " + ", ".join(missing))

    from woof.core.thompson_aerosol_state import (
        aerosol_profile_needs_fill,
        launch_aerosol_init_profile,
    )

    nz, ny, nx = state.p.shape
    phb = state.phb if state.phb.ndim == 3 else state.phb[:, None, None]
    z8w = state.scratch((nz + 1, ny, nx), "mp_z8w")
    z8w[...] = (phb + state.php) / DTYPE(c.G)
    # z8w[:nz] is a leading-axis slice of a C-contiguous array, so it is
    # itself C-contiguous and needs no copy.
    hgt = z8w[:nz]

    # The sealed forcing inventory records presence independently of values:
    # an analyzed zero remains data, including a source mass-to-number zero.
    supplied = set(getattr(state, "_external_scalar_boundary_fields", ()))
    fill_ccn = "nwfa" not in supplied and aerosol_profile_needs_fill(state.nwfa)
    fill_in = "nifa" not in supplied and aerosol_profile_needs_fill(state.nifa)
    if fill_ccn or fill_in:
        launch_aerosol_init_profile(
            hgt, state.nwfa, state.nifa, state.nwfa2d,
            fill_ccn=fill_ccn, fill_in=fill_in)
    version = getattr(cfg, "thompson_version", "wrf_461")
    if version == "wrf_39_noaa":
        # fork thompson_init :499-512 (CCN) and :555-566 (IN): a profile
        # that arrived (no synthetic fill) has its negative numbers
        # repaired before the lowest level is read for emission below.
        if not fill_ccn:
            wrf39_repair_negative_aerosol(state.nwfa)
        if not fill_in:
            wrf39_repair_negative_aerosol(state.nifa)
        # fork thompson_init :587-603 runs even with analysed aerosols.
        # This belongs to domain initialization, never the per-step call.
        from woof.core.thompson_aerosol_launch import thompson_version_scope
        from woof.core.thompson_aerosol_state import launch_wrf39_start_emission
        with thompson_version_scope(version):
            launch_wrf39_start_emission(state.nwfa, state.nwfa2d,
                                      dx=cfg.dx, dy=cfg.dy)
    return {"ccn": bool(fill_ccn), "in": bool(fill_in)}


def aerosol_profile_fill(state: DomainState) -> dict:
    """The thompson_init synthetic CCN/IN profile, for any aerosol-aware
    scheme whose state carries nwfa/nifa/nwfa2d (mp=28's arm above plus the
    named schemes).  Fills only an all-zero field, exactly as above.  It
    sits after the per-step adapter's code on purpose: it is domain
    construction, never a per-step call."""
    from woof.core.thompson_aerosol_state import (
        aerosol_profile_needs_fill,
        launch_aerosol_init_profile,
    )

    nz, ny, nx = state.p.shape
    phb = state.phb if state.phb.ndim == 3 else state.phb[:, None, None]
    z8w = state.scratch((nz + 1, ny, nx), "mp_z8w")
    z8w[...] = (phb + state.php) / DTYPE(c.G)
    hgt = z8w[:nz]
    fill_ccn = aerosol_profile_needs_fill(state.nwfa)
    fill_in = aerosol_profile_needs_fill(state.nifa)
    if fill_ccn or fill_in:
        launch_aerosol_init_profile(
            hgt, state.nwfa, state.nifa, state.nwfa2d,
            fill_ccn=fill_ccn, fill_in=fill_in)
    return {"ccn": bool(fill_ccn), "in": bool(fill_in)}


__all__ = [
    "AEROSOL_INT_SCRATCH_SLOTS",
    "AEROSOL_SCRATCH_SLOTS",
    "REQUIRED_STATE_FIELDS",
    "REQUIRED_SURFACE_FIELDS",
    "_apply_thompson_aerosol",
    "thompson_aerosol_init_fill",
]
