"""Full-physics forecast with every rate class on its own whole step (one card or several).

The forecast door's executor for ``--fullstep-local-timestep --fullstep-lts-mode
active``.  It reuses, unchanged, the host preparation, the per-partition slicer
and stack builder of the same-host executor, the two-phase physics seam, the
history capture, the physical gate and the NetCDF writer.  What it adds is the
clock:

* **Classes.**  ``woof.hex.fullstep_lts_admission`` assigns each cell a dyadic
  rate from the mesh and the initial state.  A mesh that classes entirely at
  rate one never reaches this module's stepping: the request is handed to the
  unchanged native executor, so on a uniform mesh "on" is "off".
* **Physics per class step.**  Each class owns one resident column backend
  built for the class's local columns and SEALED at the class's own step
  ``rate * dt``.  Surface, boundary layer, drag and cumulus run once per class
  step with that duration and their tendencies are held over it; the
  microphysics runs once at the class's endpoint with the same duration.
* **Radiation.**  Each backend counts its own steps; ``require_class_clock``
  has already shown that the radiation interval is a whole number of every
  class's step, so every class calls radiation at the same model times.
* **History.**  A frame is written only at a macro boundary, where every class
  stands at the same instant and the interface registers are empty.
* **Restart identity.**  At every frame the SHA-256 of the owned-union
  prognostic state and each class's physics restart identity are recorded in
  ``fullstep-lts-restart-identity.jsonl``: the bytes a restart must reproduce.
  A run resumed from a class checkpoint reproduces these lines and the
  history bytes (``fullstep_lts_restart_v841``; measured in
  ``docs/fullstep-local-timestep.md``).

A class's non-owned columns (its scratch zone of coarser cells and the halo
rings) run the class's physics too, as the halo columns of a partition do.
Their atmosphere is reset from the owner at the owner's step boundaries; their
land-surface state is the class backend's own and is never published.

The drag scheme and the proof module compare a step length against module
constants bound once per run (``cuda_gwdo_v841.X4_DT_SECONDS_F32``).  Classes
step one after another on one stream, so the constant is rebound to the
stepping class's length for exactly the duration of that class step
(``_class_step_length``) and restored after it.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
from typing import Any, Mapping

import numpy as np

RESTART_IDENTITY_NAME = "fullstep-lts-restart-identity.jsonl"
RECEIPT_NAME = "fullstep-lts-receipt.json"
_STATE = ("rho", "rho_theta", "rho_u", "rho_w")


def require_stochastic_support(value) -> None:
    """Shared admission for the driver and moving-ensemble orchestration.

    Keep this check usable before CUDA preparation or subprocess launch. The
    ensemble must not count a resolved metadata roster as an admitted forecast.
    """
    from woof.hex.stochastic import StochasticConfig
    StochasticConfig.parse(value)


def _refuse_unsupported(*, host, cycling, park_physics_tier, local_timestep_classing_path) -> None:
    """Each refusal names what would break."""
    require_stochastic_support(host.get("stochastic"))

    # Retired 2026-10-12: --checkpoint-hours, --restart-from and --da-analysis
    # were refused because a single-stack checkpoint holds one state and one
    # backend.  The class checkpoint (woof.hex.drivers.fullstep_lts_restart_v841)
    # holds one of each per class, the scratch columns and the interface
    # mailboxes; a plain resume is bit-identical to the uninterrupted run.
    # --jedi-background: the native writer runs on global views assembled from
    # every class's OWNED columns (fullstep_lts_restart_v841.write_class_background).
    # Retired 2026-10-12: a limited-area mesh was refused because boundary
    # relaxation and the specified zone ran on one outer clock.  Each class
    # driver now carries its own boundary runtime on its local columns (the
    # LBC pool projected through the class layout), advanced by its own class
    # step; the admission holds the whole boundary zone in one class at the
    # base rate (``fullstep_lts_admission.BOUNDARY_RATE``), and every boundary
    # file time is a class-clock event (``_boundary_events``).
    if park_physics_tier:
        raise ValueError(
            "a parked physics tier moves one backend off the device across the dynamics; with "
            "one backend per class the park would empty arrays another class is about to read")
    if local_timestep_classing_path is not None or getattr(host["config"], "config_local_timestep", False):
        raise ValueError(
            "the acoustic local-time-step attachment owns a different flux and clock schedule; "
            "the two schedules cannot describe the same accepted interval")


def _garbage_donor(host: Mapping[str, Any], mesh: Any, n_cells: int) -> tuple[int, ...]:
    """The global cell every class of a limited-area run must hold locally.

    The regional driver lends one real column's physics inputs to native's
    garbage column and refuses a sub-mesh without it
    (``cuda_regional_forecast_v841.PaddedRegionalHostMesh``); the same-host
    executor replicates the same cell into every rank.
    """

    if host.get("regional") is None:
        return ()
    # regional_garbage_donor's rule, without importing the CUDA module here
    donor = (getattr(mesh, "attrs", {}) or {}).get("regional_garbage_donor_cell", int(n_cells) - 1)
    if isinstance(donor, (bool, np.bool_)) or not isinstance(donor, (int, np.integer)) or not 0 <= donor < n_cells:
        raise ValueError("regional garbage donor must identify a real cell; prevents lending an absent physics column")
    return (int(donor),)


def _boundary_rate(max_rate: int) -> int:
    """The limited-area boundary zone's rate: the admission's default, or the instrument knob.

    ``WOOF_HEX_LTS_BOUNDARY_RATE`` is an EXPERIMENT KNOB (A/B of the zone at a
    coarser class, whose relaxation is then weaker per unit time); the
    default is ``fullstep_lts_admission.BOUNDARY_RATE``.
    """

    import os

    from woof.hex.fullstep_lts_admission import BOUNDARY_RATE

    text = os.environ.get("WOOF_HEX_LTS_BOUNDARY_RATE")
    if text is None:
        return BOUNDARY_RATE
    value = int(text)
    if value not in (1, 2, 4, 8, 16) or value > int(max_rate):
        raise ValueError("WOOF_HEX_LTS_BOUNDARY_RATE must be a dyadic rate no larger than the top rate; "
                         "another value names a class the run does not have")
    return value


def _boundary_events(host: Mapping[str, Any], run_seconds: float) -> dict[str, float]:
    """Every lateral-boundary file time inside the run, in seconds from its start.

    THE BREAKAGE THIS PREVENTS: a class driver reads the next boundary file
    when its own clock reaches the current interval's end, at a step
    boundary.  A class step that straddled a file time would integrate part
    of a step on the expired interval's tendency, and a coarser class would
    straddle where a finer one does not, so two classes would apply different
    boundary data over the same instant.
    """

    regional = host.get("regional")
    if regional is None:
        return {}
    from datetime import datetime

    start = regional["start_time"]
    found: dict[str, float] = {}
    for path in regional["lbc_paths"]:
        name = Path(path).name
        stamp = name[len("lbc."):-len(".nc")] if name.startswith("lbc.") and name.endswith(".nc") else None
        if stamp is None:
            continue
        when = datetime.strptime(stamp, "%Y-%m-%d_%H.%M.%S")
        offset = (when - start).total_seconds()
        if 0.0 < offset < float(run_seconds):
            found[f"lateral boundary file {name}"] = offset
    return found


@contextmanager
def _class_step_length(seconds: float):
    """Bind the run-wide step-length constants to the stepping class, then restore them."""

    from woof.hex import cuda_gwdo_v841 as gwdo

    before = gwdo.X4_DT_SECONDS_F32
    gwdo.X4_DT_SECONDS_F32 = np.float32(seconds)
    try:
        yield
    finally:
        gwdo.X4_DT_SECONDS_F32 = before


def class_physics_host(host: Mapping[str, Any], layout: Any, *, base_dt: float, rate: int) -> dict[str, Any]:
    """The prepared host of one class: its local columns, its own step, its own cadences.

    A cadence that was welded to the base step (surface/PBL and cumulus, in
    the shipped configuration) is welded to the class step; one that was an
    independent interval (radiation; an explicit PBL cadence) is kept, and
    must then be a whole number of class steps -- the sealed constructor and
    the configuration both refuse otherwise.
    """

    from woof.hex.drivers.samehost_forecast_v841 import _prepared_attribute_view
    from woof.hex.partition_local_mesh_v841 import slice_prepared_host

    class_dt = float(base_dt) * int(rate)
    local = dict(host)
    # The member owns one global spectral process, never one per truncated
    # class mesh. The synchronized executor supplies the held-tendency hook.
    local["stochastic"] = None
    for name in ("prepared", "constructor_values", "gwdo_host", "f000_surface_diagnostics"):
        item = _prepared_attribute_view(host[name]) if name == "prepared" else host[name]
        local[name] = slice_prepared_host(item, layout)
    values = dict(local["constructor_values"])
    # a limited-area class driver carries native's garbage column (as the
    # same-host executor's regional ranks do)
    values["n_columns"] = int(layout.n_local_cells) + int(host.get("regional") is not None)
    for key in ("surface_pbl_seconds", "cumulus_seconds"):
        if values.get(key) is not None and float(values[key]) == float(values["dt"]):
            values[key] = class_dt
    values["dt"] = class_dt
    local["constructor_values"] = values
    config = host["config"]
    changes: dict[str, Any] = {"config_dt": class_dt}
    for key in ("config_bldt_seconds", "config_cudt_seconds"):
        value = getattr(config, key, None)
        if value is not None and float(value) == float(config.config_dt):
            changes[key] = class_dt
    local["config"] = replace(config, **changes)
    local["config"].validate()
    return local


def class_composite_step(*, driver, backend, scalar_names, physics_geometry, kernel_cache, dt,
                         previous_surface_updates, dynamics_tendencies, refl_10cm_due, before_phase_two,
                         after_phase_one=None, held_forcing=None):
    """One two-phase physics transaction of one class at its own step length.

    The transaction is the proof module's ``execute_composite_step`` with the
    step length as an argument instead of that module's run-wide constant,
    and with the halo round ahead of the scalar clamp (see below).  This run
    arms no rollback, so a refused step ends the run.
    """

    from woof.hex.cuda_physics_v841 import (
        clamp_wsm6_scalars_in_place_v841,
        couple_raw_column_physics_v841,
        recover_post_rk_wsm6_state_v841,
        with_lagged_diabatic_theta_v841,
    )

    start_state = driver.atmosphere.state
    start_time = float(start_state.time_seconds)
    candidate = None
    try:
        raw = backend.begin_step(atmosphere=driver.atmosphere, scalar_names=scalar_names, dt=dt,
                                 dynamics_tendencies=dynamics_tendencies)
        if after_phase_one is not None:
            after_phase_one(raw)
        edge_fields = driver.horizontal.recover_edge_fields(start_state.rho, start_state.rho_u)
        held = couple_raw_column_physics_v841(
            raw, state=start_state, scalar_names=scalar_names, geometry=physics_geometry,
            rho_edge=edge_fields.rho_edge, kernel_cache=kernel_cache)
        # SG2: this class's previous microphysics heating rides every RK stage
        # of this class step (MPAS rt_diabatic_tend; per class, at its own dt).
        held = with_lagged_diabatic_theta_v841(held, backend)
        if held_forcing is not None:
            # IAU (woof.hex.iau_forcing, or woof.hex.iau_native on MPAS's own kernels): the
            # increment's share for this step joins the held physics tendencies, the slot
            # MPAS-A's atm_add_tend_anal_incr writes, before dynamics.  The native route reads
            # this step's state, recovered edge fields and edge connectivity.
            held_forcing(held, start_time, dt, driver=driver, rho_edge=edge_fields.rho_edge,
                         normal_velocity=edge_fields.normal_velocity, geometry=physics_geometry,
                         state=start_state)
        candidate = driver.step_device_with_physics(held)
        next_dynamics = getattr(candidate, "dynamics_tendencies", None)
        # The halo round comes BEFORE the clamp here (the proof module clamps
        # first, and its ranks then exchange values every owner has clamped).
        # A coarser owner's frozen state can be one no step has clamped yet:
        # the GFS initial file holds negative water vapour, and the column
        # preparation refuses a negative species in any local column
        # (measured 2026-10-11: the first class step of the first macro).
        before_phase_two(candidate.atmosphere)
        clamp_wsm6_scalars_in_place_v841(candidate.atmosphere.state.scalars, scalar_names=scalar_names,
                                         kernel_cache=kernel_cache)
        update = backend.finish_step(atmosphere=candidate.atmosphere, scalar_names=scalar_names, dt=dt,
                                     refl_10cm_due=refl_10cm_due)
        recovery = recover_post_rk_wsm6_state_v841(
            candidate.atmosphere.state, update, scalar_names=scalar_names, kernel_cache=kernel_cache,
            phase2_dt_seconds=dt, previous_surface_updates=previous_surface_updates)
        committed = driver.commit_post_wsm6_candidate(candidate, recovery)
    except Exception:
        if candidate is not None and getattr(driver, "_pending_v841_physics_candidate", None) is candidate:
            driver.abort_post_wsm6_candidate(candidate)
        raise
    backend.commit_step()
    if float(committed.atmosphere.state.time_seconds) != start_time + dt:
        raise RuntimeError("the class transaction committed the wrong endpoint time")
    return committed, next_dynamics


def _release_pool_policy() -> bool:
    """``WOOF_HEX_RELEASE_POOL=macro``: hand the device pool's unused blocks back.

    Set by woof.hex.forecast_batch for members it places several to a card.
    CuPy's pool keeps every block a member ever freed, so a member holds the
    high-water mark of its transients (the radiation call, the first frame's
    gathers) for the rest of its run, not its live arrays: measured
    2026-10-13 on the derecho patch, 6.8 GB live against a 17.9 GB pool.  With
    the policy on, the unused blocks go back to the card after frame 0 and at
    every macro boundary (the loop has already synchronized there): the
    radiation macro's peak stays (16.5 GB), the rest of the run holds 9.6-12.3
    GB, and the macros take the same time (3.34 s).  The arithmetic does not
    change, only which addresses later arrays get (histories bitwise equal
    with and without).  Unset (a member alone on its card) keeps the pool.
    """

    import os

    text = os.environ.get("WOOF_HEX_RELEASE_POOL", "").strip().lower()
    if text not in ("", "macro"):
        raise ValueError("WOOF_HEX_RELEASE_POOL must be unset or 'macro'; prevents a "
                         "misspelled policy silently keeping every transient block")
    return text == "macro"


def _release_pool(cp, stream) -> None:
    stream.synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def _sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _shock_zone_masks(n_cells: int):
    """The latest move's changed-zone target cells (``WOOF_HEX_MOVING_SHOCK_ZONE``); diagnostic only."""

    if not os.environ.get("WOOF_HEX_MOVING_SHOCK_ZONE"):
        return None, None
    zone_column_mask = None
    with np.load(os.environ["WOOF_HEX_MOVING_SHOCK_ZONE"], allow_pickle=False) as zone_file:
        zone_shock_mask = np.zeros(int(zone_file["sizes"][1]), dtype=bool)
        zone_shock_mask[zone_file["zone_target"]] = True
        if "column_target" in zone_file:
            zone_shock_mask[zone_file["column_target"]] = True
            zone_column_mask = np.zeros_like(zone_shock_mask)
            zone_column_mask[zone_file["column_target"]] = True
    if len(zone_shock_mask) != int(n_cells):
        return None, zone_column_mask
    return zone_shock_mask, zone_column_mask


def _pressure_shock_row(current, previous, layouts, mesh_arrays, *, macro_seconds, model_seconds,
                        zone_shock_mask=None, zone_column_mask=None) -> dict[str, Any]:
    """One row of the post-move shock series: the surface-pressure tendency over the last macro step.

    ``current`` and ``previous`` hold each partition's owned surface pressure (classes on one
    card, blocks on several); ``layouts`` are the same partition's layouts.
    """

    tendency = None
    details: dict[str, Any] = {}
    if previous is not None:
        differences = [(a - b).reshape(-1) / macro_seconds for a, b in zip(current, previous)]
        peaks = [float(np.max(abs(delta))) for delta in differences]
        tendency = max(peaks)
        owner = int(np.argmax(peaks))
        local = int(np.argmax(abs(differences[owner])))
        cell = int(layouts[owner].cell_l2g[local])
        areas = np.asarray(mesh_arrays['areaCell'], dtype=np.float64)
        weighted_square = sum(float(np.dot(delta.astype(np.float64) ** 2,
                                           areas[layout.cell_l2g[:layout.n_owned_cells]]))
                              for delta, layout in zip(differences, layouts))
        x, y, z = (float(mesh_arrays[key][cell]) for key in ('xCell', 'yCell', 'zCell'))
        if zone_shock_mask is not None:
            # Split the tendency between the changed zone and the carried cells.
            for label, wanted in (("zone", True), ("carried", False)):
                peak, square, area_sum, peak_cell = 0.0, 0.0, 0.0, -1
                for delta, layout in zip(differences, layouts):
                    ids = layout.cell_l2g[:layout.n_owned_cells]
                    pick = zone_shock_mask[ids] == wanted
                    if pick.any():
                        values = delta[pick].astype(np.float64)
                        if float(np.max(abs(values))) > peak:
                            peak_cell = int(ids[pick][int(np.argmax(abs(values)))])
                        peak = max(peak, float(np.max(abs(values))))
                        square += float(np.dot(values ** 2, areas[ids][pick]))
                        area_sum += float(areas[ids][pick].sum())
                details[f"{label}_surface_pressure_tendency_abs_max_pa_s"] = peak
                details[f"{label}_surface_pressure_tendency_area_rms_pa_s"] = (
                    float(np.sqrt(square / area_sum)) if area_sum else 0.0)
                if peak_cell >= 0:
                    px, py, pz = (float(mesh_arrays[key][peak_cell]) for key in ('xCell', 'yCell', 'zCell'))
                    details[f"{label}_peak_cell"] = peak_cell
                    details[f"{label}_peak_latitude_deg"] = float(np.degrees(np.arctan2(pz, np.hypot(px, py))))
                    details[f"{label}_peak_longitude_deg"] = float(np.degrees(np.arctan2(py, px)))
                    if zone_column_mask is not None:
                        details[f"{label}_peak_is_column_cell"] = bool(zone_column_mask[peak_cell])
        details.update(surface_pressure_tendency_area_rms_pa_s=float(np.sqrt(weighted_square / areas.sum())),
                       surface_pressure_tendency_peak_cell=cell,
                       surface_pressure_tendency_peak_signed_pa_s=float(differences[owner][local]),
                       surface_pressure_tendency_peak_latitude_deg=float(np.degrees(np.arctan2(z, np.hypot(x, y)))),
                       surface_pressure_tendency_peak_longitude_deg=float(np.degrees(np.arctan2(y, x))))
    return dict(model_seconds=model_seconds, surface_pressure_tendency_abs_max_pa_s=tendency, **details)


@contextmanager
def installed(forecast, *, policy: Mapping[str, Any]):
    """Route this invocation's ``execute_forecast`` through the class executor."""

    previous = forecast.execute_forecast

    def execute(**kwargs):
        return execute_fullstep_lts(forecast, native_execute=previous, policy=dict(policy), **kwargs)

    forecast.execute_forecast = execute
    try:
        yield
    finally:
        forecast.execute_forecast = previous


def execute_fullstep_lts(forecast, *, native_execute, policy, host, schedule, cache_root, output_root,
                         arwen_checkout, source_receipt, authority_receipt, fingerprint_every,
                         stop_on_refusal=False, grid_path=None, park_physics_tier=False,
                         required_free_bytes=None, local_timestep_classing_path=None,
                         fullstep_lts_options=None, cycling=None, devices=None,
                         memory_floor_by_device=None, iau=None, balance_series=False):
    from woof.hex.fullstep_lts_admission import (admit_mesh_rates, metric_lengths, moving_region_requested,
                                                require_class_clock)

    proof = forecast.proof
    # Moving variable-resolution region (woof.hex.moving_control).  Every hook below
    # is behind a WOOF_HEX_MOVING_* switch; with none set this function is the
    # fixed-mesh LTS forecast unchanged: no carrier is read or written, the clock
    # offset is 0, histories are written synchronously in the original order and
    # the receipt carries no moving keys.
    moving_requested = moving_region_requested()
    moving_resume = None
    moving_weights = None
    moving_receipt = None
    if os.environ.get("WOOF_HEX_MOVING_CHECKPOINT_IN"):
        if cycling is not None:
            raise ValueError(
                "a moving-region segment resumes every class from its sealed carrier at the carrier's "
                "model time; --restart-from, --da-analysis, --checkpoint-hours and --jedi-background "
                "set or record class state on the run's own clock, so the two would overwrite or "
                "mislabel each other. Run the moving segment without LTS cycling")
        from woof.hex.moving_checkpoint import load_boundary, resume_same_mesh_host
        if os.environ.get("WOOF_HEX_MOVING_CHECKPOINT_WAIT"):
            # A pre-launched move segment: process start and host preparation ran while
            # the previous member was still integrating; the carrier is sealed when it stops.
            from woof.hex.moving_control import wait_for_carrier
            wait_for_carrier(os.environ["WOOF_HEX_MOVING_CHECKPOINT_IN"],
                             float(os.environ["WOOF_HEX_MOVING_CHECKPOINT_WAIT"]))
        moving_resume = load_boundary(os.environ["WOOF_HEX_MOVING_CHECKPOINT_IN"])
        if os.environ.get("WOOF_HEX_MOVING_ZONE"):
            # Frozen-far-field move: exact carry outside the changed zone (woof.hex.moving_zone).
            from woof.hex.moving_zone_state import resume_zone_host
            host, moving_weights, moving_receipt = resume_zone_host(
                host, moving_resume, zone_path=os.environ["WOOF_HEX_MOVING_ZONE"],
                receipt_path=Path(output_root) / "moving-remap.json")
        elif os.environ.get("WOOF_HEX_MOVING_OPERATOR"):
            from woof.hex.moving_checkpoint import resume_remapped_host
            host, moving_weights, moving_receipt = resume_remapped_host(
                host, moving_resume, source_grid=os.environ["WOOF_HEX_MOVING_SOURCE_GRID"],
                target_grid=grid_path, operator_path=os.environ["WOOF_HEX_MOVING_OPERATOR"],
                receipt_path=Path(output_root) / "moving-remap.json")
        else:
            host = resume_same_mesh_host(host, moving_resume)
        from woof.hex.moving_checkpoint import carry_physics_top_pressure
        host = carry_physics_top_pressure(host, moving_resume)
    config = host["config"]
    base_dt = float(config.config_dt)
    # A resumed moving segment starts at its carrier's model time; every other run at 0.
    clock_offset = float(host["prepared"].state.time_seconds) if moving_resume is not None else 0.0
    offset_steps = int(round(clock_offset / base_dt))
    if offset_steps * base_dt != clock_offset:
        raise ValueError("moving restart boundary does not lie on the base clock")
    if clock_offset:
        from datetime import datetime, timedelta
        schedule = dict(schedule)
        for key, format_text in (("labels", "%Y-%m-%d_%H.%M.%S"),
                                 ("valid_times", "%Y-%m-%d_%H:%M:%S")):
            schedule[key] = {step: (datetime.strptime(label, format_text) +
                                   timedelta(seconds=clock_offset)).strftime(format_text)
                             for step, label in schedule[key].items()}
    prepared = host["prepared"]
    mesh = prepared.mesh
    transports = bool(config.config_scalar_advection)
    # GUARD (2026-10-12): the full-physics top rate defaults to 8.  Breakage it prevents: the
    # 48-3 km patch mesh forecast (GFS 2026-10-03 12Z, base step 18.75 s) at top rate 16 is
    # stopped by the validation flag (non-finite) inside a class step with physics: at step
    # 1984 (10.3 h) with the matched interface flux, at step 2112 (11.0 h) with the earlier
    # coupling (c06273b), at step 1168 (6.1 h) with the matched momentum flux alone.  Top rate
    # 8 runs the same 24 h.  The dry dynamics and transport run 24 h at top rates 16 and 32,
    # so this is not the retired rate-16/8 interface instability of the dry dynamics; the
    # cause is not found (first suspect: the column physics at a 300 s class step).  Rates 16
    # and 32 stay selectable.
    max_rate = int(policy.get("fullstep_lts_max_rate", 8))
    rates, rate_info = admit_mesh_rates(
        mesh, prepared.state, prepared.saved_diagnostics, prepared.vertical, dt=base_dt, max_rate=max_rate,
        target_advection=float(policy.get("fullstep_lts_target_advection", 0.80)),
        target_acoustic=float(policy.get("fullstep_lts_target_acoustic", 0.70)),
        acoustic_substeps=int(config.config_number_of_sub_steps),
        target_transport=float(policy.get("fullstep_lts_target_transport", 0.90)) if transports else None,
        boundary_rate=_boundary_rate(max_rate))
    if moving_resume is not None and moving_weights is None:
        rates = np.asarray(moving_resume["rates"], dtype=np.int32)
        rate_info["source"] = "carried same-mesh class assignment for identity control"
    native_kwargs = dict(
        host=host, schedule=schedule, cache_root=cache_root, output_root=output_root,
        arwen_checkout=arwen_checkout, source_receipt=source_receipt, authority_receipt=authority_receipt,
        fingerprint_every=fingerprint_every, stop_on_refusal=stop_on_refusal, grid_path=grid_path,
        park_physics_tier=park_physics_tier, required_free_bytes=required_free_bytes,
        local_timestep_classing_path=local_timestep_classing_path, fullstep_lts_options=None, cycling=cycling)
    if int(rates.max()) == 1:
        from woof.hex.fullstep_lts_admission import moving_route_refusal
        # THE BREAKAGE: a mesh that classes entirely at rate one runs the native step (the same-host
        # ranks on several cards, the native single-class step on one), which reads none of the move
        # switches, so a moving segment there would run one mesh to its end and never move.  On one
        # card only the controller's intent refuses (completeness row #42): a same-mesh resume with no
        # controller is the identity control and runs (tests/test_moving_inactive_noop.py).
        refusal = moving_route_refusal("same-host" if devices is not None and len(devices) > 1 else "single-class")
        if refusal is not None:
            raise ValueError(refusal)
        if balance_series and devices is not None and len(devices) > 1:
            # The same-host ranks have no balance-series sampler: the flag would be accepted and nothing
            # written.  (IAU runs there since the ranks gained the held-tendency hook.)
            raise ValueError(
                "the balance series runs on one card; this mesh classes entirely at rate one and several "
                "cards were given, so it would run the same-host ranks, which have no balance-series sampler")
        # One class: the unchanged native executor (one card, or the same-host
        # ranks when several cards were given), before any class object exists.
        # Its one-card step carries the IAU hook since lane W1 (woof.hex.iau_native),
        # which retired the refusal that stood here for want of one.
        if iau is not None or balance_series:
            native_kwargs.update(iau=iau, balance_series=balance_series)
        result = native_execute(**native_kwargs)
        result["fullstep_lts"] = {"dispatch": "native single-class step", "rate_assignment": rate_info}
        return result
    _refuse_unsupported(host=host, cycling=cycling, park_physics_tier=park_physics_tier,
                        local_timestep_classing_path=local_timestep_classing_path)
    from woof.hex.physics_backend_admission import resolve_backend

    row = resolve_backend(str(host.get("physics_backend", "wsm6_column")))
    resident = ("woof.core.mpas_column_batch", "woof.core.mpas_column_suite")
    if row.batch_module not in resident:
        raise ValueError(
            f"active full-step local time stepping builds each class's physics from a resident "
            f"column seam {resident}; backend row {row.name} binds {row.batch_module}, which no "
            "class builder constructs")
    steps = int(schedule["steps"])
    captures = set(int(step) for step in schedule["capture_steps"])
    from woof.hex.forecast_start import RadarSnapshots
    top = int(rates.max())
    radar = RadarSnapshots(output_root, schedule, base_dt, stride=top)
    events = {"run length": steps * base_dt,
              "radiation interval": float(host["constructor_values"]["radiation_seconds"])}
    for step in sorted(captures - {0}):
        events[f"history frame at base step {step}"] = step * base_dt
    events.update(_boundary_events(host, steps * base_dt))
    clock = require_class_clock(base_dt, [top], events)
    if devices is not None and len(devices) > 1:
        from woof.hex.stochastic import StochasticConfig
        if StochasticConfig.parse(host.get("stochastic")).enabled:
            raise ValueError("multi-card LTS stochastic forcing needs a cross-device owned-cell "
                             "budget reduction; the one-card member runtime cannot publish "
                             "that budget to other cards. Run each ensemble member on one card")
        if iau is not None or balance_series:
            raise ValueError(
                "IAU forcing is wired into the one-card class executor; the multi-card block "
                "executor has no held-tendency hook and would drop the increment")
        if cycling is not None:
            raise ValueError("multi-card full-step LTS cannot checkpoint or apply DA: its class-to-rank state transfer does not persist backend accumulators or pending tendencies. Use one card for LTS cycling")
        return _execute_multi(forecast, policy=policy, host=host, schedule=schedule, cache_root=cache_root,
                              output_root=output_root, arwen_checkout=arwen_checkout,
                              stop_on_refusal=stop_on_refusal, required_free_bytes=required_free_bytes,
                              rates=rates, rate_info=rate_info, clock=clock, devices=tuple(devices),
                              memory_floor_by_device=memory_floor_by_device,
                              moving=dict(requested=moving_requested, resume=moving_resume,
                                          weights=moving_weights, receipt=moving_receipt,
                                          clock_offset=clock_offset, offset_steps=offset_steps))

    from woof.hex.cuda_arwen_physics_v841 import pin_arwen_physics_v841
    arwen_pin = dict(forecast._pin_for_run(host, arwen_checkout, pin_arwen_physics_v841))
    from woof.hex.cuda_backend import KernelCache, require_cuda
    import cupy as cp

    from woof.hex.cuda_fullstep_lts_active import FullStepLtsActive
    from woof.hex.fullstep_lts_classes import build_class_plan
    from woof.hex.drivers.samehost_forecast_v841 import _capture_rank
    from woof.hex.drivers.samehost_identity_v841 import assemble_owned_arrays
    from woof.hex.drivers.samehost_physics_scope_v841 import scoped_physics_resources

    capability = require_cuda(min_compute=(12, 0), cache_dir=cache_root)
    memory = (proof.gpu_memory_admission(cp) if required_free_bytes is None
              else proof.gpu_memory_admission(cp, minimum=int(required_free_bytes)))
    declarations = dict(getattr(mesh, "variable_dimensions", {}) or {})
    for name in ("defc_a", "defc_b"):
        if name in mesh.arrays:
            declarations.setdefault(name, ("nCells", "maxEdges"))
    mesh.variable_dimensions = declarations
    area = np.asarray(mesh.area_cell, dtype=np.float64)
    dzw = np.asarray(prepared.vertical.dzw, dtype=np.float64)
    plan = build_class_plan(
        rates=rates, cells_on_cell=np.asarray(mesh.arrays["cellsOnCell"]),
        n_edges_on_cell=np.asarray(mesh.arrays["nEdgesOnCell"]),
        cells_on_edge=np.asarray(mesh.arrays["cellsOnEdge"]),
        cells_on_vertex=np.asarray(mesh.arrays["cellsOnVertex"]), area_cell=area,
        dc_edge_m=metric_lengths(mesh)[0], base_dt=base_dt,
        edges_on_edge=mesh.arrays.get("edgesOnEdge"), n_edges_on_edge=mesh.arrays.get("nEdgesOnEdge"),
        replicated_cells=_garbage_donor(host, mesh, int(rates.size)))
    layouts = [tables.layout for tables in plan.classes]
    if moving_resume is not None and moving_weights is None:
        from woof.hex.moving_checkpoint import require_same_layouts
        require_same_layouts(layouts, moving_resume["layouts"])
    print(json.dumps({"fullstep_lts": {"rate_assignment": rate_info, "plan": plan.receipt(),
                                       "clock": clock}}), flush=True)

    from woof.hex.drivers import fullstep_lts_restart_v841 as lts_restart

    n_global_cells = int(layouts[0]._global_cells)
    n_global_edges = int(layouts[0]._global_edges)
    start_step = 0
    checkpoint = None
    restart_receipt = None
    class_restart: list[dict[str, Any]] = [{} for _ in plan.classes]
    if cycling is not None and cycling.get("restart_from") is not None:
        import pickle

        with Path(cycling["restart_from"]).open("rb") as stream:
            checkpoint = pickle.load(stream)
        start_step = lts_restart.validate_checkpoint(
            checkpoint, base_dt=base_dt, top=top, rates=rates, plan=plan, n_cells=n_global_cells,
            schedule=schedule)
        from woof.hex.iau_native import IAU_WINDOW_KEY, resolve_restart_window

        iau = resolve_restart_window(iau, carried=checkpoint.get(IAU_WINDOW_KEY),
                                     restart_seconds=float(start_step) * base_dt)
        restart_receipt = {
            "checkpoint": str(cycling["restart_from"]),
            "checkpoint_sha256": proof.sha256_file(Path(cycling["restart_from"])),
            "step": start_step, "valid_time": checkpoint["valid_time"], "da_cycling": None,
            "interface": "restored from the checkpoint"}
        analysis = None
        if cycling.get("da_analysis") is not None:
            from woof.hex import jedi_contract

            first = checkpoint["global_state"]
            state, saved, da_receipt = jedi_contract.recouple_da_cycling(
                Path(cycling["da_analysis"]), init_path=Path(cycling["init_path"]), mesh=mesh,
                vertical=prepared.vertical, terrain_metrics=prepared.terrain_metrics,
                expected_reference=prepared.reference, scalar_names=host["scalar_names"],
                n_cells=n_global_cells, n_edges=n_global_edges, n_levels=int(first["rho"].shape[0]),
                allow_regional_sentinels=host.get("regional") is not None,
                time_seconds=float(start_step) * base_dt, work_dir=Path(output_root))
            restart_receipt["da_cycling"] = da_receipt
            restart_receipt["interface"] = (
                "rebuilt from the analysis on every local column of every class, as a cold start "
                "builds it; physics backends resumed from the checkpoint")
            analysis = (state, saved)
        for index, (tables, entry) in enumerate(zip(plan.classes, checkpoint["classes"])):
            if int(entry["index"]) != int(tables.index) or int(entry["rate"]) != int(tables.rate):
                raise ValueError("the LTS checkpoint's classes are in a different order or rate")
            if analysis is None:
                local_state, local_saved = lts_restart.class_restart_state(entry["state"])
            else:
                local_state, local_saved = lts_restart.slice_global_state(*analysis, tables.layout)
            class_restart[index] = {"state": local_state, "saved_diagnostics": local_saved,
                                    "backend_restart": entry["backend_state"]}
        if host.get("regional") is not None:
            host = dict(host)
            host["regional"] = dict(host["regional"])
            from datetime import datetime

            host["regional"]["start_time"] = datetime.strptime(checkpoint["valid_time"], "%Y-%m-%d_%H:%M:%S")

    result: dict[str, Any] = {}
    # The classes step one after another on this card, so their large work buffers share one
    # arena (woof.hex.class_work_arena: the breakage it prevents is named there).
    from woof.hex.class_work_arena import shared_large_work_buffers
    with scoped_physics_resources() as physics_scope, shared_large_work_buffers(cp) as work_arena:
        construct_started = time.perf_counter()
        stacks, areas, physics_remaps = [], [], []
        mapped_physics_cache = {"target_landmask": host["constructor_values"]["landmask"]}
        for tables in plan.classes:
            class_cache = Path(cache_root) / f"class{tables.index}-of-{len(plan.classes)}"
            class_cache.mkdir(parents=True, exist_ok=True)
            cache = KernelCache(capability=capability, cache_dir=str(class_cache))
            local_host = class_physics_host(host, tables.layout, base_dt=base_dt, rate=tables.rate)
            with physics_scope.rank(tables.index, int(cp.cuda.Device().id)):
                moving_kwargs = {}
                if moving_resume is not None:
                    def initialize_moving_backend(backend):
                        from woof.hex.moving_physics import restore_remapped_backend
                        physics_remaps.append(restore_remapped_backend(
                            backend, moving_resume, moving_weights, tables.layout,
                            base_dt=base_dt, rate=tables.rate,mapped_cache=mapped_physics_cache,target_mesh=mesh))
                    moving_kwargs = dict(
                        backend_restart=(None if moving_weights is not None
                                         else moving_resume["backends"][tables.index]),
                        backend_initializer=(initialize_moving_backend if moving_weights is not None else None))
                stack = proof._construct_device_stack(
                    host=local_host, cache=cache, arwen_checkout=arwen_checkout,
                    backend_builder=forecast._backend_builder(host, arwen_checkout),
                    physics_rollback_snapshot=False, **class_restart[tables.index], **moving_kwargs)
                from woof.hex.arwen_surface_update import attach_host
                attach_host(stack["backend"], host, schedule, output_root,
                            dt=base_dt, layout=tables.layout)
                if moving_resume is not None and moving_weights is None and "local_atmospheres" in moving_resume:
                    from woof.hex.moving_checkpoint import restore_local_atmosphere
                    restore_local_atmosphere(stack["driver"],moving_resume["local_atmospheres"][tables.index])
                cp.cuda.get_current_stream().synchronize()
            edge_extent = int(stack["driver"].horizontal.nedges)
            stack["owned_health_extents"] = {
                int(stack["driver"].horizontal.ncells): int(tables.layout.n_owned_cells),
                edge_extent: int(tables.layout.n_owned_edges)}
            stack["class_dt"] = base_dt * tables.rate
            stack["scalar_names"] = host["scalar_names"]
            stacks.append(stack)
            areas.append(area[tables.layout.cell_l2g.astype(np.int64)])
        mapped_physics_cache.clear()
        shared_cache = KernelCache(capability=capability, cache_dir=str(Path(cache_root) / "fullstep-lts"))
        active = FullStepLtsActive(plan, [stack["driver"] for stack in stacks], base_dt=base_dt,
                                   kernel_cache=shared_cache, area_cell=areas, dzw=dzw)
        if moving_resume is not None and moving_weights is None:
            from woof.hex.moving_checkpoint import restore_mailboxes
            restore_mailboxes(active,moving_resume)
        from woof.hex.stochastic import LtsStochasticPhysics, StochasticConfig
        stochastic = None
        if StochasticConfig.parse(host.get("stochastic")).enabled:
            stochastic = LtsStochasticPhysics(mesh, prepared.vertical, host["stochastic"], layouts)
            saved_stochastic = (moving_resume if moving_resume is not None else checkpoint)
            if saved_stochastic is not None:
                if saved_stochastic.get("stochastic") is None:
                    raise ValueError("stochastic LTS restart lacks its member coefficients and "
                                     "dissipation budget; redrawing would change the trajectory")
                stochastic.restore(saved_stochastic["stochastic"])
        elif ((moving_resume or checkpoint or {}).get("stochastic") is not None):
            raise ValueError("stochastic LTS restart cannot disable the carried member forcing")
        active.stochastic = stochastic
        construct_seconds = time.perf_counter() - construct_started
        interior_budget = None
        if host.get("regional") is not None:
            interior_budget = active.attach_interior_budget(
                interior=np.asarray(mesh.arrays["bdyMaskCell"]).reshape(-1)[:n_global_cells] == 0,
                dzw=dzw, cells_on_edge=np.asarray(mesh.arrays["cellsOnEdge"]))
        previous = []
        for stack in stacks:
            previous.append(proof._previous_surface_updates(stack, host["surface_accumulators"]))
        cp.cuda.get_current_stream().synchronize()
        forcing: list[Any] = [None] * len(stacks)
        iau_forcing = None
        iau_receipt = None
        if iau is not None:
            from woof.hex.iau_native import build_iau_forcing
            window = float(iau["window_seconds"])
            if abs(window / (base_dt * top) - round(window / (base_dt * top))) > 1e-9:
                # Not a whole number of macro steps: classes straddle the window end with partial
                # steps; the arithmetic stays exact (overlap weights), the receipt says so.
                print(json.dumps({"iau": {"note": "window is not a whole number of macro steps",
                                          "window_seconds": window, "macro_seconds": base_dt * top}}),
                      flush=True)
            iau_forcing, iau_receipt = build_iau_forcing(
                iau, prepared=prepared, scalar_names=host["scalar_names"], layouts=layouts, cp=cp,
                work_dir=Path(output_root), n_cells=None, n_edges=n_global_edges,
                n_levels=int(np.asarray(prepared.vertical.zz).shape[0]))
            print(json.dumps({"iau": iau_receipt}), flush=True)
        balance = None
        if balance_series:
            from woof.hex.iau_forcing import BalanceSeries
            balance = BalanceSeries.build(Path(output_root) / "balance-series.jsonl",
                                          host["scalar_names"], dzw, cp)
        class_calls = [{"steps": 0, "radiation": 0, "surface_pbl": 0, "cumulus": 0,
                        "radiation_times_seconds": []} for _ in stacks]
        checkpoint_files: dict[str, Any] = {}
        jedi_files: dict[str, Any] = {}
        init_fallback_cache: dict[str, Any] = {}
        if checkpoint is not None:
            lts_restart.restore_interface(active, checkpoint,
                                          interface=restart_receipt["da_cycling"] is None)
            for index, entry in enumerate(checkpoint["classes"]):
                forcing[index] = lts_restart.restore_forcing(
                    cp, entry["gf_dynamics_tendencies"], time_seconds=start_step * base_dt)
                class_calls[index] = {key: (list(value) if isinstance(value, list) else value)
                                      for key, value in entry["class_calls"].items()}
            del checkpoint, class_restart
            checkpoint = None
        if moving_resume is not None:
            from woof.hex.moving_checkpoint import restore_step_carriers
            carriers = moving_resume
            if moving_weights is not None:
                from woof.hex.moving_physics import remap_step_carriers
                carriers = remap_step_carriers(moving_resume, moving_weights, layouts)
            previous, forcing = restore_step_carriers(carriers, stacks)
        if moving_receipt is not None:
            moving_receipt["physics"] = physics_remaps
            moving_receipt["new_rates"] = rate_info
            Path(output_root, "moving-remap.json").write_text(json.dumps(moving_receipt,indent=2)+"\n")

        # F000's surface-layer diagnostics come from each class's first physics
        # call (woof.hex.f000_first_call): every class's first step starts from
        # the start state, so its seam then holds the scheme's diagnosis of its
        # own start columns.  The frame is assembled before the first macro and
        # published after it.  A resumed or moving-resumed start captures no f000.
        from woof.hex import f000_first_call
        first_call = (f000_first_call.FirstCallHandoff()
                      if (0 in captures and start_step == 0 and moving_resume is None and steps > 0)
                      else None)
        first_reads: dict[int, Any] = {}
        held_f000: list[Any] = []

        def stepper(klass) -> None:
            index = klass.index
            stack = stacks[index]
            seconds = klass.step_length * base_dt
            if seconds != stack["class_dt"]:
                raise RuntimeError(
                    f"class {index} was asked for a {seconds} s step but its physics is sealed at "
                    f"{stack['class_dt']} s; an event tail reached a sealed backend")
            ticks = active.planes_per_step       # clock ticks per base step: 3 N
            end_step = (klass.step_start_tick + ticks * klass.step_length) // ticks
            try:
                def member_forcing(held, start, length, **context):
                    if stochastic is not None:
                        stochastic.apply(index, held, state=context["state"],
                                         rho_edge=context["rho_edge"],
                                         normal_velocity=context["normal_velocity"],
                                         time_seconds=start, dt=length, names=host["scalar_names"])
                    if iau_forcing is not None:
                        iau_forcing.add(index, held, start, length, **context)
                with physics_scope.rank(index, int(cp.cuda.Device().id)), _class_step_length(seconds):
                    committed, next_dynamics = class_composite_step(
                        driver=stack["driver"], backend=stack["backend"], scalar_names=host["scalar_names"],
                        physics_geometry=stack["physics_geometry"], kernel_cache=stack["driver"].cache,
                        dt=seconds, previous_surface_updates=previous[index],
                        dynamics_tendencies=forcing[index], refl_10cm_due=(end_step in captures or radar.due(end_step)),
                        before_phase_two=klass.exchanger.round_physics_candidate,
                        held_forcing=(member_forcing if stochastic is not None or iau_forcing is not None
                                      else None))
            except Exception as error:
                # Which rate class refused (a failure at a class's own dt is a different
                # diagnosis from one at the base step); carried into the first-failure receipt.
                error.add_note(f"rate class {index} (rate {getattr(klass, 'rate', '?')}, step {seconds} s, "
                               f"class start tick {klass.step_start_tick})")
                raise
            previous[index] = committed.surface_updates
            forcing[index] = next_dynamics
            if first_call is not None and not first_call.done and index not in first_reads:
                with physics_scope.rank(index, int(cp.cuda.Device().id)):
                    first_reads[index] = f000_first_call.read_arwen_first_call(stack["backend"], cp)
            cadence = dict(stack["backend"].step_receipt()).get("cadence") or {}
            calls = class_calls[index]
            calls["steps"] += 1
            if cadence.get("radiation_ran"):
                calls["radiation"] += 1
                if len(calls["radiation_times_seconds"]) < 64:
                    calls["radiation_times_seconds"].append(
                        clock_offset + klass.step_start_tick * base_dt / ticks)
            calls["surface_pbl"] += int(bool(cadence.get("surface_pbl_ran")))
            calls["cumulus"] += int(bool(cadence.get("cumulus_ran")))

        active.stepper = stepper

        static = proof._static_output_fields(host)
        frame_records: dict[str, Any] = {}
        frame_hashes: dict[str, Any] = {}
        physical_gates: dict[str, Any] = {}
        health: list[Any] = []
        series: list[dict[str, Any]] = []
        macro_rows: list[dict[str, Any]] = []
        refusal = None
        identity_path = Path(output_root) / RESTART_IDENTITY_NAME
        capture_seconds = write_seconds = health_seconds = 0.0
        shock_rows = []
        pressure_previous = None
        zone_shock_mask, zone_column_mask = _shock_zone_masks(int(host["prepared"].mesh.arrays["areaCell"].shape[0]))

        def measure_pressure_shock(step):
            nonlocal pressure_previous
            from woof.hex.cuda_physics_prep_v841 import prepare_mpas_to_phys_cuda_v841
            current=[]
            for index, stack in enumerate(stacks):
                with physics_scope.rank(index, int(cp.cuda.Device().id)):
                    fields=prepare_mpas_to_phys_cuda_v841(
                        stack["driver"].atmosphere,scalar_names=host["scalar_names"],
                        geometry=stack["prep_geometry"],kernel_cache=stack["driver"].cache,post_rk_wsm6=False)
                    current.append(cp.asnumpy(fields.psfc_p)[..., :layouts[index].n_owned_cells])
            shock_rows.append(_pressure_shock_row(
                current, pressure_previous, layouts, host['prepared'].mesh.arrays, macro_seconds=top * base_dt,
                model_seconds=clock_offset + step * base_dt, zone_shock_mask=zone_shock_mask,
                zone_column_mask=zone_column_mask))
            pressure_previous=current

        def restart_identity(step: int) -> dict[str, Any]:
            state = active.gather(_STATE)
            record = {"step": int(step), "model_seconds": clock_offset + step * base_dt,
                      "registers_empty": bool(active.registers_empty()),
                      "class_ticks": [int(v) for v in active.class_tick],
                      "ticks_per_base_step": int(active.planes_per_step),
                      "state_sha256": {name: _sha(value) for name, value in state.items()}}
            if active.scalars is not None:
                record["state_sha256"]["scalars"] = _sha(active.scalars.gather())
            del state
            physics = []
            for stack in stacks:
                identity = getattr(stack["backend"], "_restart_identity", None)
                receipt = dict(stack["backend"].step_receipt())
                physics.append({
                    # the digest of the backend's own restart identity (its
                    # contract, engine sources and sealed constructor)
                    "restart_identity_sha256": None if identity is None else hashlib.sha256(
                        json.dumps(dict(identity()), sort_keys=True, default=str).encode()).hexdigest(),
                    "phase": receipt.get("phase"), "arwen_step_index": receipt.get("arwen_step_index"),
                    "end_time_seconds": receipt.get("end_time_seconds")})
            record["class_physics"] = physics
            with identity_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            return record

        def totals(step: int) -> None:
            row = {"step": int(step), "seconds": clock_offset + step * base_dt, **active.totals()}
            row.pop("tracer_min", None)
            row.pop("tracer_max", None)
            series.append(row)

        def capture_radar(step):
            def source():
                if not active.registers_empty():
                    raise RuntimeError('radar columns need a committed LTS macro boundary')
                arrays = []
                for index, stack in enumerate(stacks):
                    with physics_scope.rank(index, int(cp.cuda.Device().id)):
                        arrays.append(radar.columns(stack, host))
                return assemble_owned_arrays(layouts, arrays,
                    kinds={name: "cell" for name in arrays[0]}, native_pads=())
            radar.capture_from(step, source, proof, static)

        def capture(step: int) -> None:
            nonlocal capture_seconds, write_seconds
            if not active.registers_empty():
                raise RuntimeError(f"an interface register is not empty at history step {step}; "
                                   "the frame would be written between two halves of a transfer")
            began = time.perf_counter()
            arrays = []
            for index, stack in enumerate(stacks):
                with physics_scope.rank(index, int(cp.cuda.Device().id)):
                    arrays.append(_capture_rank(proof, host, stack, offset_steps + step, schedule["labels"][step]))
            kinds = {name: "edge" if name == "normal_u" else "cell" for name in arrays[0]}
            # a limited-area class frame strips the garbage column from every
            # cell field; normal_u keeps native's garbage edge (as the
            # same-host executor's regional ranks hand it over)
            native_pads = ("normal_u",) if host.get("regional") is not None else ()
            snapshot = {"arrays": assemble_owned_arrays(layouts, arrays, kinds=kinds, native_pads=native_pads),
                        "receipt": {"step": step, "arwen_v2_surface_execution": {"last_noahmp_census": None}}}
            del arrays
            if step == 0 and moving_resume is None:
                proof.overlay_f000_initialized_surface_diagnostics(
                    snapshot["arrays"], host["f000_surface_diagnostics"]["arrays"])
                snapshot["receipt"]["f000_completed_surface_diagnostics"] = (
                    forecast.complete_f000_surface_diagnostics(snapshot, host["f000_start_humidity"]))
            capture_seconds += time.perf_counter() - began
            if step == 0 and first_call is not None and not first_call.done:
                # The boundary's live-state work runs now; the frame is published
                # after the first macro, with the first call's diagnostics.
                boundary_work(step, snapshot, defer_background=True)
                held_f000.append(snapshot)
                return
            publish(step, snapshot)

        held_background: list[Any] = []

        def boundary_work(step: int, snapshot, defer_background: bool = False) -> None:
            restart_identity(step)
            if cycling is not None and step in cycling.get("checkpoint_steps", ()):
                write_checkpoint(step)
            if cycling is not None and cycling.get("jedi_background"):
                if "fields" not in init_fallback_cache:
                    init_fallback_cache["fields"] = forecast._init_fallback_fields(Path(cycling["init_path"]))
                writer = lts_restart.class_background_writer(
                    forecast, active=active, stacks=stacks, host=host, snapshot=snapshot,
                    label=schedule["labels"][step], valid_time=schedule["valid_times"][step],
                    soil_levels=int(cycling["soil_levels"]), init_fallback=init_fallback_cache["fields"],
                    output_root=Path(output_root), init_path=Path(cycling["init_path"]))
                if defer_background:
                    held_background.append((step, writer))  # f000: written from the completed frame
                else:
                    jedi_files[str(step)] = writer()

        def hand_f000_first_call() -> None:
            names = (sorted(set.intersection(*(set(first_reads[i]) for i in range(len(stacks)))))
                     if len(first_reads) == len(stacks) else [])
            first_call.put(assemble_owned_arrays(
                layouts, [{n: first_reads[i][n] for n in names} for i in range(len(stacks))],
                kinds={n: "cell" for n in names}, native_pads=()) if names else None)
            first_reads.clear()
            if held_f000:
                publish(0, held_f000.pop(), boundary_done=True)

        def publish(step: int, snapshot, boundary_done: bool = False) -> None:
            nonlocal write_seconds
            radar.note_history(step)
            if step == 0 and first_call is not None:
                snapshot["receipt"]["f000_first_call_diagnostics"] = (
                    forecast.complete_f000_from_first_call(snapshot, first_call.take()))
            physical_gates[str(step)] = proof.physical_snapshot_gate(
                snapshot, allow_initial_negative_qv=(step == 0), scalar_names=host["scalar_names"],
                surface_accumulators=host["surface_accumulators"])
            frame_hashes[str(step)] = {name: proof.array_sha256(array)
                                       for name, array in snapshot["arrays"].items()}
            began = time.perf_counter()
            path = Path(output_root) / f"cuda-history.{schedule['labels'][step]}.nc"
            if history_writer is None and process_history is None:
                # Synchronous (WOOF_HEX_ASYNC_HISTORY=0): the original order.
                frame_records[str(step)] = proof.write_snapshot_netcdf(path, snapshot, static)
            write_seconds += time.perf_counter() - began
            if not boundary_done:
                boundary_work(step, snapshot)
            while held_background:
                held_step, writer = held_background.pop()
                jedi_files[str(held_step)] = writer()
            if history_writer is None and process_history is None:
                with (Path(output_root) / "history-ready.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"step": int(step), "path": str(path)}) + "\n")
                return
            if process_history is not None:
                process_history.submit(step, path, snapshot, static)
                return
            model_seconds = clock_offset + step * base_dt

            def write_frame():
                began = time.perf_counter()
                record = proof.write_snapshot_netcdf(path, snapshot, static)
                seconds = time.perf_counter() - began
                with history_lock:
                    with (Path(output_root) / "history-ready.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps({"step": int(step), "path": str(path)}) + "\n")
                    if moving_control is not None:
                        moving_control.frame_written(model_seconds=model_seconds, history_path=str(path),
                                                     final=step >= steps)
                return record, seconds

            # The NetCDF write (about 20 s for a 354k-cell frame) overlaps the next
            # macro steps; the snapshot is host memory that nothing else touches.
            pending_writes[step] = history_writer.submit(write_frame)

        def drain_writes(until_step=None) -> None:
            nonlocal write_seconds
            if process_history is not None:
                process_history.drain(until_step=until_step)
                write_seconds = sum(row["seconds"] for row in process_history.receipt()["frames"])
            for step in sorted(pending_writes):
                if until_step is not None and step > until_step:
                    continue
                record, seconds = pending_writes.pop(step).result()
                frame_records[str(step)] = record
                write_seconds += seconds

        moving_control = None
        if os.environ.get("WOOF_HEX_MOVING_CONTROL"):
            from woof.hex.moving_control import MovingControl
            moving_control = MovingControl.from_environ()
        pending_writes: dict[int, Any] = {}
        history_lock = history_writer = process_history = None

        def process_frame_written(step, path, record):
            frame_records[str(step)] = record
            with (Path(output_root) / "history-ready.jsonl").open("a", encoding="utf-8") as ready:
                ready.write(json.dumps({"step": int(step), "path": str(path)}) + "\n")
            if moving_control is not None:
                moving_control.frame_written(model_seconds=clock_offset + step * base_dt,
                                             history_path=str(path), final=step >= steps)

        # The history writer, ``WOOF_HEX_ASYNC_HISTORY``: ``process`` (the
        # default without the moving controller) writes each frame in the native
        # writer's child process, bounded to two frames, so the NetCDF write
        # overlaps the next macros and never shares the NetCDF/HDF5 library with
        # a boundary read on this thread; ``1`` (the default under the moving
        # controller, unchanged) writes on a thread; ``0`` writes inline, the
        # original order.  The written bytes are identical in all three; only
        # their timing moves.  Breakage the default prevents (HEX-LTS-REGRESSION
        # 2026-10-27): every one-card LTS run without the moving controller wrote
        # its frames on the step thread, 6.8 s per 1.4 GB frame on the 149k-cell
        # 96 -> 3 km zoom globe (27.2 s of a 3 h run), about 10 % of the LTS
        # forecast hour, while the fixed step's same-host executor writes off the
        # step path by default.
        async_history = os.environ.get("WOOF_HEX_ASYNC_HISTORY",
                                       "1" if moving_control is not None else "process")
        if async_history not in ("0", "1", "process"):
            raise ValueError("WOOF_HEX_ASYNC_HISTORY must be 0, 1 or process; prevents an ambiguous "
                             "history-writer receipt")
        if async_history == "process":
            from woof.hex.history_frames import ProcessHistoryFrames
            process_history = ProcessHistoryFrames(proof.write_snapshot_netcdf, process_frame_written)
        # Thread writer: the moving controller's frames are hourly 0.7 GB writes,
        # ~22 min of a 48 h run when synchronous.
        elif async_history == "1":
            import concurrent.futures
            import threading
            history_lock = threading.Lock()
            history_writer = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="history")

        def recovery_seal(path):
            from woof.hex.moving_checkpoint import export_boundary
            return export_boundary(path, active=active, stacks=stacks, previous=previous, forcing=forcing,
                                   host=host, physics_scope=physics_scope)

        def write_checkpoint(step: int) -> None:
            import pickle

            payload = lts_restart.download_checkpoint(
                active=active, stacks=stacks, forcing=forcing, class_calls=class_calls, step=step,
                label=schedule["labels"][step], valid_time=schedule["valid_times"][step],
                base_dt=base_dt, top=top, rates=rates, n_cells=n_global_cells, n_edges=n_global_edges,
                global_state={**active.gather(_STATE),
                              **({} if active.scalars is None else {"scalars": active.scalars.gather()})})
            if stochastic is not None:
                payload["stochastic"] = stochastic.snapshot(device=False)
            if iau is not None:
                from woof.hex.iau_native import IAU_WINDOW_KEY, window_record

                payload[IAU_WINDOW_KEY] = window_record(iau)
            target = Path(output_root) / f"checkpoint.{schedule['labels'][step]}.pkl"
            with target.open("xb") as stream:
                pickle.dump(payload, stream, protocol=5)
            del payload
            checkpoint_files[str(step)] = {"path": str(target), "bytes": target.stat().st_size,
                                           "sha256": proof.sha256_file(target),
                                           "schema": lts_restart.CHECKPOINT_SCHEMA}

        totals(start_step)
        if os.environ.get("WOOF_HEX_MOVING_SHOCK") == "1":
            measure_pressure_shock(start_step)
        # A moving segment resumed from a carrier does not re-capture its first frame:
        # the history-only execution census and reflectivity are rebuilt by the first
        # committed physics call and are not persistent restart state.
        if 0 in captures and start_step == 0 and moving_resume is None:
            capture(0)
        if balance is not None:
            balance.sample(active, 0.0, cp)
        from woof.hex import time_ledger
        time_ledger.end_if_open("forecast.construct")
        stream = cp.cuda.get_current_stream()
        release_pool = _release_pool_policy()
        if release_pool:
            _release_pool(cp, stream)
        started = time.perf_counter()
        done = start_step
        totals_every = max(top, int(round(3600.0 / base_dt / top)) * top)
        try:
            while done < steps:
                if process_history is not None:
                    process_history.drain(ready_only=True)
                if release_pool and done:
                    _release_pool(cp, stream)
                stream.synchronize()
                mark = time.perf_counter()
                try:
                    if stochastic is not None:
                        stochastic.begin_macro(clock_offset + done * base_dt, top * base_dt)
                    active.advance_macro(top)
                    if stochastic is not None:
                        stochastic.end_macro()
                    stream.synchronize()
                except Exception as error:
                    failure = {"step": done + top, "last_committed_shared_step": done,
                               "exception": type(error).__name__, "message": str(error),
                               "notes": list(getattr(error, "__notes__", ())),
                               "traceback": traceback.format_exc()}
                    try:
                        # Where it went: the largest |w| in the classes' current (unpublished-step
                        # excluded) state, so a refusal is located, not only counted.
                        state = active.gather(("rho", "rho_w"))
                        rho_i = np.empty_like(state["rho_w"])
                        rho_i[1:-1] = 0.5 * (state["rho"][1:] + state["rho"][:-1])
                        rho_i[0], rho_i[-1] = state["rho"][0], state["rho"][-1]
                        w = np.abs(state["rho_w"] / rho_i)
                        k, cell = np.unravel_index(int(np.nanargmax(np.where(np.isfinite(w), w, -1.0))), w.shape)
                        arrays_ = host["prepared"].mesh.arrays
                        failure["w_peak_in_current_state"] = {
                            "w_abs_m_s": float(w[k, cell]), "interface_level": int(k), "cell": int(cell),
                            "latitude_deg": float(np.degrees(arrays_["latCell"][cell])),
                            "longitude_deg": float(np.degrees(arrays_["lonCell"][cell])),
                            "rate": int(rates[cell]) if len(rates) > cell else None,
                            "nonfinite_w_values": int(np.count_nonzero(~np.isfinite(w)))}
                    except Exception as locate_error:  # localisation is evidence only
                        failure["w_peak_in_current_state"] = {"unavailable": repr(locate_error)}
                    Path(output_root, "fullstep-lts-first-failure.json").write_text(
                        json.dumps(failure, indent=2) + "\n")
                    if not stop_on_refusal:
                        raise
                    refusal = {**failure, "final_state_retained": False,
                               "reason": "a class step was refused; the classes no longer share a "
                                         "clock, so only frames from completed macro boundaries remain"}
                    break
                elapsed = time.perf_counter() - mark
                done += top
                if first_call is not None and not first_call.done:
                    with time_ledger.stage("forecast.output.f000_first_call"):
                        hand_f000_first_call()
                macro_rows.append({"step": done, "base_steps": top, "seconds": elapsed})
                if balance is not None:
                    balance.sample(active, done * base_dt, cp)
                began = time.perf_counter()
                rows = [forecast.step_health_gate(stack, done, cp) for stack in stacks]
                health.append(rows)
                health_seconds += time.perf_counter() - began
                if os.environ.get("WOOF_HEX_MOVING_SHOCK") == "1":
                    measure_pressure_shock(done)
                    shock_rows[-1]["vertical_velocity_abs_max"]=max(r["vertical_velocity_abs_max"] for r in rows)
                print(json.dumps({"step": done, "seconds": elapsed, "time_seconds": clock_offset + done * base_dt,
                                  "fullstep_lts_classes": len(stacks),
                                  "vertical_velocity_abs_max": max(r["vertical_velocity_abs_max"] for r in rows)}),
                      flush=True)
                if done % totals_every == 0 or done == steps:
                    totals(done)
                if radar.due(done) and done not in captures:
                    capture_radar(done)
                if done in captures:
                    capture(done)
                    if moving_control is not None and moving_control.at_frame(
                            model_seconds=clock_offset + done * base_dt, final=done >= steps, seal=recovery_seal,
                            wait_written=lambda step=done: drain_writes(until_step=step)):
                        break
            if first_call is not None and held_f000:
                first_call.put(None)  # the first macro was refused: publish f000 as captured
                publish(0, held_f000.pop(), boundary_done=True)
        finally:
            if first_call is not None:
                first_call.put(None)
            radar.close()
            loop_seconds = time.perf_counter() - started
            try:
                drain_writes()
            finally:
                if process_history is not None:
                    process_history.close()
                if history_writer is not None:
                    history_writer.shutdown(wait=True)
        if refusal is None and not active.registers_empty():
            raise RuntimeError("an interface register is not empty at the end of the run")
        moving_export = None
        if refusal is None and os.environ.get("WOOF_HEX_MOVING_CHECKPOINT_OUT"):
            from woof.hex.moving_checkpoint import export_boundary
            moving_export = export_boundary(
                os.environ["WOOF_HEX_MOVING_CHECKPOINT_OUT"], active=active,
                stacks=stacks, previous=previous, forcing=forcing, host=host,
                physics_scope=physics_scope)
        integration = sum(row["seconds"] for row in macro_rows)
        first = macro_rows[0]["seconds"] if macro_rows else None
        warm = macro_rows[1:]
        first_mass = series[0]["dry_mass"]
        moving_fields = {} if not moving_requested else {
            "moving_boundary_export": moving_export,
            "moving_remap": moving_receipt,
            "moving_control": None if moving_control is None else moving_control.receipt(),
            "clock_offset_seconds": clock_offset,
            "remap_shock_series": shock_rows,
        }
        lts_receipt = {
            "schema": "gpuwm-hex.fullstep-lts-forecast/v1",
            "dispatch": "rate classes",
            **moving_fields,
            "base_dt": base_dt,
            "rate_assignment": rate_info,
            "rates_sha256": _sha(np.ascontiguousarray(rates)),
            "clock": clock,
            "active": active.receipt(),
            "class_physics": [
                {"class": index, "rate": int(tables.rate), "step_seconds": stacks[index]["class_dt"],
                 "owned_columns": int(tables.layout.n_owned_cells),
                 "local_columns": int(tables.layout.n_local_cells),
                 "constructor": {key: stacks[index]["constructor"].receipt().get(key)
                                 for key in ("dt", "radiation_seconds", "surface_pbl_seconds", "cumulus_seconds")
                                 if key in stacks[index]["constructor"].receipt()},
                 **class_calls[index]}
                for index, tables in enumerate(plan.classes)],
            "physics_cadence_note": (
                "surface, boundary layer, drag and cumulus once per class step with the class's "
                "duration; microphysics once at the class endpoint; radiation on each class's own "
                "step count, at the same model times in every class"),
            "scratch_column_physics": (
                "a class's non-owned columns run the class's physics on the class backend's own "
                "land-surface state; their atmosphere is reset from the owner, their land state "
                "is never published"),
            "dry_mass_series": series,
            "dry_mass_relative_drift": abs(series[-1]["dry_mass"] / first_mass - 1.0),
            "interior_budget": None if interior_budget is None else {
                **interior_budget,
                "law": ("interior dry mass (bdyMaskCell == 0) plus the integrated accepted outflow through "
                        "the interior/zone edges is constant; the zone itself is forced by the boundary data"),
                "relative_closure_max": max(
                    abs(row["interior_mass_plus_outflow"] / series[0]["interior_mass_plus_outflow"] - 1.0)
                    for row in series),
                "interior_relative_change": series[-1]["interior_dry_mass"] / series[0]["interior_dry_mass"] - 1.0,
                "outflow_relative": series[-1]["interior_boundary_outflow"] / series[0]["interior_dry_mass"]},
            "dry_mass_max_relative_excursion": max(abs(row["dry_mass"] / first_mass - 1.0) for row in series),
            "registers_empty_at_end": bool(active.registers_empty()),
            "restart_identity": str(identity_path),
            "restart": restart_receipt,
            "checkpoints": checkpoint_files,
            "construct_seconds": construct_seconds,
            "work_buffer_arena": work_arena.receipt(),
            "macros": macro_rows,
            "warm_seconds_per_forecast_hour": (
                sum(row["seconds"] for row in warm) / (sum(row["base_steps"] for row in warm) * base_dt) * 3600.0
                if warm else None),
            "stochastic": None if stochastic is None else stochastic.receipt(),
            "iau": (None if iau_forcing is None else {**iau_receipt, **iau_forcing.receipt()}),
            "balance_series": (None if balance is None else str(balance.path)),
        }
        if (iau_forcing is not None and done * base_dt >= iau_forcing.window_start + iau_forcing.window
                and not iau_forcing.receipt()["complete"]):
            raise RuntimeError(f"IAU window not fully applied: {iau_forcing.receipt()}")
        Path(output_root, RECEIPT_NAME).write_text(json.dumps(lts_receipt, indent=2, default=str) + "\n")
        result = {
            "schema": "gpuwm-hex.fullstep-lts-forecast/v1",
            "fullstep_lts": {key: value for key, value in lts_receipt.items() if key != "macros"},
            "steps_executed": done, "steps_requested": steps,
            "arwen_pin": arwen_pin, "memory": memory,
            "step_health": health, "health": health,
            "physical_gates": physical_gates, "refusal": refusal,
            "schedule": {key: value for key, value in schedule.items() if key != "labels"},
            "history_labels": schedule["labels"],
            "walls": {
                "steps": done, "integration_seconds": integration,
                "integration_note": "sum of macro walls, each closed by a stream synchronize; excludes "
                                    "construction, health, conservation totals, capture and history writing",
                "first_step_seconds": first,
                "seconds_per_step_after_first": (
                    sum(row["seconds"] for row in warm) / sum(row["base_steps"] for row in warm)
                    if warm else None),
                "loop_seconds_including_io": loop_seconds,
                "snapshot_capture_seconds": capture_seconds,
                "history_write_seconds": write_seconds,
                "history_writer": None if process_history is None else process_history.receipt(),
                "history_writer_mode": {"process": "child process", "1": "thread",
                                        "0": "inline"}[async_history],
                "history_write_note": ("worker work overlaps integration; not caller blocking time"
                                       if process_history is not None else "synchronous or legacy thread writer"),
                "health_gate_seconds": health_seconds,
                "forecast_seconds": done * base_dt,
            },
            "snapshot_files": frame_records, "all_published_field_hashes": frame_hashes,
            "physics_resource_scope": None,
            "jedi_contract": None if cycling is None else {
                "background_files": jedi_files, "soil_levels": int(cycling.get("soil_levels", 9)),
                "checkpoint_files": checkpoint_files, "restart": restart_receipt,
                "start_step": start_step},
        }
    result["physics_resource_scope"] = physics_scope.receipt()
    return result


def _execute_multi(forecast, *, policy, host, schedule, cache_root, output_root, arwen_checkout,
                   stop_on_refusal, required_free_bytes, rates, rate_info, clock, devices,
                   memory_floor_by_device=None, moving=None):
    """The class executor across several cards of one host: every class cut into blocks.

    Same law as the one-card executor above, block by block: each block owns
    one resident column backend sealed at its class's step; the blocks of a
    class step concurrently on their own cards (``FullStepLtsMulti``); frames
    are written at macro boundaries from the owned union of every block.

    A moving region (``woof.hex.moving_control``) has the one-card executor's hooks here,
    per block; ``moving`` carries what :func:`execute_fullstep_lts` read before dispatch
    (the carrier, the remap weights and receipt, the clock offset).  With no
    ``WOOF_HEX_MOVING_*`` switch set nothing below reads or writes a carrier, the clock
    offset is 0 and the receipt carries no moving keys.

    * **Stop request and recovery seals.**  At every history frame the controller may stop
      the run (every block stands at one instant, registers empty) or seal a recovery carrier.
    * **Carrier export** (``moving_checkpoint.export_boundary_blocks``): one backend, local
      atmosphere and step carrier per block, with the blocks' layouts.
    * **Remapped resume per block.**  The atmosphere arrives remapped on the whole target mesh
      (``execute_fullstep_lts``); each block's column physics is rebuilt from the carrier's
      owner columns sliced into the block's layout, and its step carriers likewise
      (``moving_physics``), whatever partition the carrier was sealed on (one card or several).
      A same-mesh resume (no operator) restores each block exactly and needs the same cards.
    * **Shared clock offset.**  Every block starts at the carrier's model time; frames, labels,
      totals, the restart identity and the shock series are on that clock.
    """

    import os
    import threading
    from woof.hex.cuda_arwen_physics_v841 import pin_arwen_physics_v841
    from woof.hex.cuda_backend import KernelCache, require_cuda
    import cupy as cp

    from woof.hex.cuda_fullstep_lts_multi import FullStepLtsMulti
    from woof.hex.fullstep_lts_admission import metric_lengths
    from woof.hex.fullstep_lts_blocks import DEFAULT_MIN_BLOCK_CELLS, blocks_per_class, build_block_plan
    from woof.hex.fullstep_lts_classes import build_class_plan
    from woof.hex.drivers.samehost_forecast_v841 import _capture_rank
    from woof.hex.drivers.samehost_identity_v841 import assemble_owned_arrays
    from woof.hex.drivers.samehost_physics_scope_v841 import scoped_physics_resources
    from woof.hex.partition_samehost_v841 import SameHostRanks

    moving = dict(moving or {})
    moving_requested = bool(moving.get("requested"))
    moving_resume = moving.get("resume")
    moving_weights = moving.get("weights")
    moving_receipt = moving.get("receipt")
    clock_offset = float(moving.get("clock_offset", 0.0))
    offset_steps = int(moving.get("offset_steps", 0))
    # A same-mesh resume (a recovery seal, or a segment restarted from its own carrier):
    # every block is restored exactly, so the partition must be the one the carrier was sealed on.
    same_partition = moving_resume is not None and moving_weights is None
    if same_partition and (moving_resume.get("partition") or {}).get("kind") != "blocks":
        raise ValueError(
            "a same-mesh moving resume restores every partition exactly (local halos, mailboxes); this "
            "carrier was sealed on one card, so its classes are not this run's blocks. Resume it on one "
            "card, or move it through an operator, which remaps onto any partition")
    proof = forecast.proof
    config = host["config"]
    base_dt = float(config.config_dt)
    prepared = host["prepared"]
    mesh = prepared.mesh
    steps = int(schedule["steps"])
    captures = set(int(step) for step in schedule["capture_steps"])
    from woof.hex.forecast_start import RadarSnapshots
    top = int(rates.max())
    radar = RadarSnapshots(output_root, schedule, base_dt, stride=top)
    arwen_pin = dict(forecast._pin_for_run(host, arwen_checkout, pin_arwen_physics_v841))
    declarations = dict(getattr(mesh, "variable_dimensions", {}) or {})
    for name in ("defc_a", "defc_b"):
        if name in mesh.arrays:
            declarations.setdefault(name, ("nCells", "maxEdges"))
    mesh.variable_dimensions = declarations
    area = np.asarray(mesh.area_cell, dtype=np.float64)
    dzw = np.asarray(prepared.vertical.dzw, dtype=np.float64)
    plan = build_class_plan(
        rates=rates, cells_on_cell=np.asarray(mesh.arrays["cellsOnCell"]),
        n_edges_on_cell=np.asarray(mesh.arrays["nEdgesOnCell"]),
        cells_on_edge=np.asarray(mesh.arrays["cellsOnEdge"]),
        cells_on_vertex=np.asarray(mesh.arrays["cellsOnVertex"]), area_cell=area,
        dc_edge_m=metric_lengths(mesh)[0], base_dt=base_dt,
        edges_on_edge=mesh.arrays.get("edgesOnEdge"), n_edges_on_edge=mesh.arrays.get("nEdgesOnEdge"))
    block_cells = int(policy.get("fullstep_lts_block_cells")
                      or os.environ.get("WOOF_HEX_FULLSTEP_LTS_BLOCK_CELLS", str(DEFAULT_MIN_BLOCK_CELLS)))
    counts = blocks_per_class(plan, len(devices), min_block_cells=block_cells)
    explicit = os.environ.get("WOOF_HEX_FULLSTEP_LTS_BLOCKS")
    if explicit:
        # an experiment knob: blocks per class, coarsest first
        counts = tuple(int(value) for value in explicit.split(","))
        if len(counts) != len(plan.classes):
            raise ValueError(f"WOOF_HEX_FULLSTEP_LTS_BLOCKS names {len(counts)} classes; the plan has "
                             f"{len(plan.classes)}, so blocks would be assigned to the wrong classes")
    lat = np.asarray(mesh.arrays["latCell"], np.float64)
    lon = np.asarray(mesh.arrays["lonCell"], np.float64)
    xyz = np.stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)), axis=1)
    bplan = build_block_plan(
        plan, devices=devices, n_blocks=counts, cells_on_cell=np.asarray(mesh.arrays["cellsOnCell"]),
        n_edges_on_cell=np.asarray(mesh.arrays["nEdgesOnCell"]), cells_on_edge=np.asarray(mesh.arrays["cellsOnEdge"]),
        cells_on_vertex=np.asarray(mesh.arrays["cellsOnVertex"]), cell_xyz=xyz, area_cell=area,
        edges_on_edge=mesh.arrays.get("edgesOnEdge"), n_edges_on_edge=mesh.arrays.get("nEdgesOnEdge"))
    if [int(tables.gid) for tables in bplan.blocks] != list(range(len(bplan.blocks))):
        raise RuntimeError("block plan is not in block order; stacks and drivers would be paired wrongly")
    layouts = [tables.layout for tables in bplan.blocks]
    block_devices = [int(tables.device) for tables in bplan.blocks]
    if same_partition:
        from woof.hex.moving_checkpoint import require_same_layouts
        require_same_layouts(layouts, moving_resume["layouts"])
        if list(moving_resume["partition"].get("devices", [])) != block_devices:
            raise ValueError(f"same-mesh moving resume on other cards: carrier blocks on "
                             f"{moving_resume['partition'].get('devices')}, this run's on {block_devices}")
    print(json.dumps({"fullstep_lts": {"rate_assignment": rate_info, "plan": plan.receipt(),
                                       "blocks": bplan.receipt(), "clock": clock}}), flush=True)
    memory = []
    for device in sorted(set(devices)):
        # The per-card floor the same-host admission derived for these cards
        # (the same rule the same-host executor applies).  The whole-mesh
        # single-card floor refused every multi-card run: measured 2026-10-11,
        # 60-3 km on four cards, 111.6 GB demanded of a 101.5 GB free card.
        floor = (memory_floor_by_device or {}).get(device, required_free_bytes or proof.MIN_FREE_DEVICE_BYTES)
        with cp.cuda.Device(device):
            memory.append({"device": device, **proof.gpu_memory_admission(cp, minimum=int(floor))})
    result: dict[str, Any] = {}
    with scoped_physics_resources() as physics_scope:
        construct_started = time.perf_counter()
        stacks: list[Any] = [None] * len(bplan.blocks)
        areas: list[Any] = [None] * len(bplan.blocks)
        lane_caches: dict[int, Any] = {}
        for device in sorted(set(devices)):
            lane_dir = Path(cache_root) / f"fullstep-lts-card{device}"
            lane_dir.mkdir(parents=True, exist_ok=True)
            with cp.cuda.Device(device):
                capability = require_cuda(device_id=device, min_compute=(12, 0), cache_dir=lane_dir)
                lane_caches[device] = KernelCache(capability=capability, cache_dir=str(lane_dir))
        # The remapped physics of every block is read from one host-side map of the carrier's
        # owner columns (moving_physics.mapped_owner_field), built once and shared: the blocks
        # of different cards construct on their own threads, so its use is serialised.
        physics_lock = threading.Lock()
        mapped_physics_cache: dict[str, Any] = {"target_landmask": host["constructor_values"]["landmask"]}
        physics_remaps: list[Any] = [None] * len(bplan.blocks)

        def construct(rank: int) -> None:
            for tables in bplan.blocks:
                if tables.rank != rank:
                    continue
                block_dir = Path(cache_root) / f"block{tables.gid}-class{tables.index}-card{tables.device}"
                block_dir.mkdir(parents=True, exist_ok=True)
                capability = require_cuda(device_id=tables.device, min_compute=(12, 0), cache_dir=block_dir)
                cache = KernelCache(capability=capability, cache_dir=str(block_dir))
                local_host = class_physics_host(host, tables.layout, base_dt=base_dt, rate=tables.rate)
                with physics_scope.rank(tables.gid, tables.device):
                    moving_kwargs: dict[str, Any] = {}
                    if same_partition:
                        moving_kwargs = dict(backend_restart=moving_resume["backends"][tables.gid])
                    elif moving_resume is not None:
                        def initialize_moving_backend(backend, tables=tables):
                            from woof.hex.moving_physics import restore_remapped_backend
                            with physics_lock:
                                physics_remaps[tables.gid] = restore_remapped_backend(
                                    backend, moving_resume, moving_weights, tables.layout, base_dt=base_dt,
                                    rate=tables.rate, mapped_cache=mapped_physics_cache, target_mesh=mesh)
                        moving_kwargs = dict(backend_initializer=initialize_moving_backend)
                    stack = proof._construct_device_stack(
                        host=local_host, cache=cache, arwen_checkout=arwen_checkout,
                        backend_builder=forecast._backend_builder(host, arwen_checkout),
                        physics_rollback_snapshot=False, **moving_kwargs)
                    from woof.hex.arwen_surface_update import attach_host
                    attach_host(stack["backend"], host, schedule, output_root,
                                dt=base_dt, layout=tables.layout)
                    if same_partition and "local_atmospheres" in moving_resume:
                        from woof.hex.moving_checkpoint import restore_local_atmosphere
                        restore_local_atmosphere(stack["driver"], moving_resume["local_atmospheres"][tables.gid])
                    cp.cuda.get_current_stream().synchronize()
                edge_extent = int(stack["driver"].horizontal.nedges)
                stack["owned_health_extents"] = {
                    int(stack["driver"].horizontal.ncells): int(tables.layout.n_owned_cells),
                    edge_extent: int(tables.layout.n_owned_edges)}
                stack["class_dt"] = base_dt * tables.rate
                stacks[tables.gid] = stack
                areas[tables.gid] = area[tables.layout.cell_l2g.astype(np.int64)]

        if len(set(devices)) == len(devices):
            with SameHostRanks(devices, None, cp=cp) as setup:
                setup.call(construct)
        else:
            # Ranks sharing a card are built one after another, as the same-host
            # executor builds them (measured 2026-10-11: two ranks constructing
            # concurrently on one card ended in an illegal address at the first step).
            for rank, device in enumerate(devices):
                with cp.cuda.Device(device):
                    construct(rank)
        mapped_physics_cache.clear()
        active = FullStepLtsMulti(bplan, [stack["driver"] for stack in stacks], base_dt=base_dt,
                                  kernel_caches=lane_caches, area_cell=areas, dzw=dzw)
        if same_partition:
            from woof.hex.moving_checkpoint import restore_lane_mailboxes
            restore_lane_mailboxes(active, moving_resume)
        construct_seconds = time.perf_counter() - construct_started
        previous: list[Any] = [None] * len(stacks)
        for block in active.blocks:
            with cp.cuda.Device(block.device), physics_scope.rank(block.gid, block.device):
                previous[block.gid] = proof._previous_surface_updates(stacks[block.gid], host["surface_accumulators"])
                cp.cuda.get_current_stream().synchronize()
        forcing: list[Any] = [None] * len(stacks)
        if moving_resume is not None:
            from woof.hex.moving_checkpoint import restore_step_carriers
            carriers = moving_resume
            if moving_weights is not None:
                from woof.hex.moving_physics import remap_step_carriers
                carriers = remap_step_carriers(moving_resume, moving_weights, layouts)
            restored_previous, restored_forcing = restore_step_carriers(carriers, stacks, devices=block_devices)
            previous[:] = restored_previous
            forcing[:] = restored_forcing
        if moving_receipt is not None:
            moving_receipt["physics"] = physics_remaps
            moving_receipt["new_rates"] = rate_info
            moving_receipt["partition"] = dict(kind="blocks", devices=block_devices)
            Path(output_root, "moving-remap.json").write_text(json.dumps(moving_receipt, indent=2) + "\n")
        class_calls = [{"steps": 0, "radiation": 0, "surface_pbl": 0, "cumulus": 0,
                        "radiation_times_seconds": []} for _ in stacks]
        raw_exchange = os.environ.get("WOOF_HEX_FULLSTEP_LTS_RAW_EXCHANGE", "0") == "1"

        # F000's surface-layer diagnostics come from each block's first physics
        # call (woof.hex.f000_first_call), read in the block's own rank thread;
        # the frame is published after the first macro.
        from woof.hex import f000_first_call
        first_call = (f000_first_call.FirstCallHandoff()
                      if (0 in captures and moving_resume is None and steps > 0) else None)
        first_reads: dict[int, Any] = {}
        held_f000: list[Any] = []

        def stepper(block) -> None:
            gid = block.gid
            stack = stacks[gid]
            seconds = block.step_length * base_dt
            if seconds != stack["class_dt"]:
                raise RuntimeError(
                    f"class {block.index} was asked for a {seconds} s step but its physics is sealed at "
                    f"{stack['class_dt']} s; an event tail reached a sealed backend")
            ticks = active.planes_per_step       # clock ticks per base step: 3 N
            end_step = (block.step_start_tick + ticks * block.step_length) // ticks
            with physics_scope.rank(gid, block.device):
                committed, next_dynamics = class_composite_step(
                    driver=stack["driver"], backend=stack["backend"], scalar_names=host["scalar_names"],
                    physics_geometry=stack["physics_geometry"], kernel_cache=stack["driver"].cache,
                    dt=seconds, previous_surface_updates=previous[gid],
                    dynamics_tendencies=forcing[gid], refl_10cm_due=(end_step in captures or radar.due(end_step)),
                    before_phase_two=block.exchanger.round_physics_candidate,
                    after_phase_one=block.exchanger.round_phase_one if raw_exchange else None)
            previous[gid] = committed.surface_updates
            forcing[gid] = next_dynamics
            if first_call is not None and not first_call.done and gid not in first_reads:
                with physics_scope.rank(gid, block.device):
                    first_reads[gid] = f000_first_call.read_arwen_first_call(stack["backend"], cp)
            cadence = dict(stack["backend"].step_receipt()).get("cadence") or {}
            calls = class_calls[gid]
            calls["steps"] += 1
            if cadence.get("radiation_ran"):
                calls["radiation"] += 1
                if len(calls["radiation_times_seconds"]) < 64:
                    calls["radiation_times_seconds"].append(clock_offset + block.step_start_tick * base_dt / ticks)
            calls["surface_pbl"] += int(bool(cadence.get("surface_pbl_ran")))
            calls["cumulus"] += int(bool(cadence.get("cumulus_ran")))

        active.stepper = stepper
        # the block's held cumulus forcing crosses to its same-class peers at round A
        active.forcing_of = lambda block: forcing[block.gid]
        # the drag scheme's run-wide step constant: every block of a stepping class shares its length
        active.around_class_step = lambda index, seconds: _class_step_length(seconds)

        static = proof._static_output_fields(host)
        frame_records: dict[str, Any] = {}
        frame_hashes: dict[str, Any] = {}
        physical_gates: dict[str, Any] = {}
        health: list[Any] = []
        series: list[dict[str, Any]] = []
        macro_rows: list[dict[str, Any]] = []
        refusal = None
        identity_path = Path(output_root) / RESTART_IDENTITY_NAME
        capture_seconds = write_seconds = health_seconds = 0.0
        peak_pool = {device: 0 for device in sorted(set(devices))}
        moving_control = None
        if os.environ.get("WOOF_HEX_MOVING_CONTROL"):
            from woof.hex.moving_control import MovingControl
            moving_control = MovingControl.from_environ()
        shock_on = os.environ.get("WOOF_HEX_MOVING_SHOCK") == "1"
        shock_rows: list[dict[str, Any]] = []
        pressure_previous = None
        zone_shock_mask, zone_column_mask = _shock_zone_masks(int(mesh.arrays["areaCell"].shape[0]))

        def sample_pool() -> None:
            for device in peak_pool:
                with cp.cuda.Device(device):
                    peak_pool[device] = max(peak_pool[device], int(cp.get_default_memory_pool().total_bytes()))

        def measure_pressure_shock(step: int) -> None:
            nonlocal pressure_previous
            from woof.hex.cuda_physics_prep_v841 import prepare_mpas_to_phys_cuda_v841
            current: list[Any] = [None] * len(stacks)

            def surface(rank: int) -> None:
                for block in active.blocks:
                    if block.rank == rank:
                        stack = stacks[block.gid]
                        with physics_scope.rank(block.gid, block.device):
                            fields = prepare_mpas_to_phys_cuda_v841(
                                stack["driver"].atmosphere, scalar_names=host["scalar_names"],
                                geometry=stack["prep_geometry"], kernel_cache=stack["driver"].cache,
                                post_rk_wsm6=False)
                            current[block.gid] = cp.asnumpy(fields.psfc_p)[..., :layouts[block.gid].n_owned_cells]

            active.ranks.call(surface)
            shock_rows.append(_pressure_shock_row(
                current, pressure_previous, layouts, mesh.arrays, macro_seconds=top * base_dt,
                model_seconds=clock_offset + step * base_dt, zone_shock_mask=zone_shock_mask,
                zone_column_mask=zone_column_mask))
            pressure_previous = current

        def restart_identity(step: int) -> None:
            state = active.gather(_STATE)
            record = {"step": int(step), "model_seconds": clock_offset + step * base_dt,
                      "registers_empty": bool(active.registers_empty()),
                      "class_ticks": [int(v) for v in active.class_tick],
                      "ticks_per_base_step": int(active.planes_per_step),
                      "state_sha256": {name: _sha(value) for name, value in state.items()}}
            if active.ntr:
                record["state_sha256"]["scalars"] = _sha(active.gather_scalars())
            del state
            physics = []
            for block in active.blocks:
                stack = stacks[block.gid]
                identity = getattr(stack["backend"], "_restart_identity", None)
                receipt = dict(stack["backend"].step_receipt())
                physics.append({
                    "block": block.gid, "class": block.index, "card": block.device,
                    "restart_identity_sha256": None if identity is None else hashlib.sha256(
                        json.dumps(dict(identity()), sort_keys=True, default=str).encode()).hexdigest(),
                    "phase": receipt.get("phase"), "arwen_step_index": receipt.get("arwen_step_index"),
                    "end_time_seconds": receipt.get("end_time_seconds")})
            record["block_physics"] = physics
            with identity_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")

        def totals(step: int) -> None:
            row = {"step": int(step), "seconds": clock_offset + step * base_dt, **active.totals()}
            row.pop("tracer_min", None)
            row.pop("tracer_max", None)
            series.append(row)

        def capture_radar(step):
            def source():
                if not active.registers_empty():
                    raise RuntimeError('radar columns need a committed LTS macro boundary')
                arrays = [None] * len(stacks)
                def gather(rank):
                    for block in active.blocks:
                        if block.rank == rank:
                            with physics_scope.rank(block.gid, block.device):
                                arrays[block.gid] = radar.columns(stacks[block.gid], host)
                active.ranks.call(gather)
                return assemble_owned_arrays(layouts, arrays,
                    kinds={name: "cell" for name in arrays[0]}, native_pads=())
            radar.capture_from(step, source, proof, static)

        def capture(step: int) -> None:
            nonlocal capture_seconds, write_seconds
            if not active.registers_empty():
                raise RuntimeError(f"an interface register is not empty at history step {step}; "
                                   "the frame would be written between two halves of a transfer")
            began = time.perf_counter()
            arrays: list[Any] = [None] * len(stacks)

            def gather(rank: int) -> None:
                for block in active.blocks:
                    if block.rank == rank:
                        with physics_scope.rank(block.gid, block.device):
                            arrays[block.gid] = {name: np.array(value, copy=True) for name, value in _capture_rank(
                                proof, host, stacks[block.gid], offset_steps + step,
                                schedule["labels"][step]).items()}

            active.ranks.call(gather)
            kinds = {name: "edge" if name == "normal_u" else "cell" for name in arrays[0]}
            snapshot = {"arrays": assemble_owned_arrays(layouts, arrays, kinds=kinds, native_pads=()),
                        "receipt": {"step": step, "arwen_v2_surface_execution": {"last_noahmp_census": None}}}
            del arrays
            if step == 0 and moving_resume is None:
                proof.overlay_f000_initialized_surface_diagnostics(
                    snapshot["arrays"], host["f000_surface_diagnostics"]["arrays"])
                snapshot["receipt"]["f000_completed_surface_diagnostics"] = (
                    forecast.complete_f000_surface_diagnostics(snapshot, host["f000_start_humidity"]))
            capture_seconds += time.perf_counter() - began
            if step == 0 and first_call is not None and not first_call.done:
                restart_identity(step)  # the boundary's live state, read now
                held_f000.append(snapshot)
                return
            publish(step, snapshot)

        def hand_f000_first_call() -> None:
            names = (sorted(set.intersection(*(set(first_reads[i]) for i in range(len(stacks)))))
                     if len(first_reads) == len(stacks) else [])
            first_call.put(assemble_owned_arrays(
                layouts, [{n: first_reads[i][n] for n in names} for i in range(len(stacks))],
                kinds={n: "cell" for n in names}, native_pads=()) if names else None)
            first_reads.clear()
            if held_f000:
                publish(0, held_f000.pop(), boundary_done=True)

        def publish(step: int, snapshot, boundary_done: bool = False) -> None:
            nonlocal write_seconds
            radar.note_history(step)
            if step == 0 and first_call is not None:
                snapshot["receipt"]["f000_first_call_diagnostics"] = (
                    forecast.complete_f000_from_first_call(snapshot, first_call.take()))
            physical_gates[str(step)] = proof.physical_snapshot_gate(
                snapshot, allow_initial_negative_qv=(step == 0), scalar_names=host["scalar_names"],
                surface_accumulators=host["surface_accumulators"])
            frame_hashes[str(step)] = {name: proof.array_sha256(array) for name, array in snapshot["arrays"].items()}
            began = time.perf_counter()
            path = Path(output_root) / f"cuda-history.{schedule['labels'][step]}.nc"
            frame_records[str(step)] = proof.write_snapshot_netcdf(path, snapshot, static)
            write_seconds += time.perf_counter() - began
            if not boundary_done:
                restart_identity(step)
            with (Path(output_root) / "history-ready.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"step": int(step), "path": str(path)}) + "\n")
            if moving_control is not None:
                # Written synchronously: the frame is closed on disk before the controller is told.
                moving_control.frame_written(model_seconds=clock_offset + step * base_dt, history_path=str(path),
                                             final=step >= steps)

        def health_rows(done: int) -> list[Any]:
            rows: list[Any] = [None] * len(stacks)

            def check(rank: int) -> None:
                for block in active.blocks:
                    if block.rank == rank:
                        rows[block.gid] = forecast.step_health_gate(stacks[block.gid], done, cp)

            active.ranks.call(check)
            return rows

        def recovery_seal(path):
            from woof.hex.moving_checkpoint import export_boundary_blocks
            return export_boundary_blocks(path, active=active, stacks=stacks, previous=previous, forcing=forcing,
                                          host=host, physics_scope=physics_scope)

        try:
            totals(0)
            if shock_on:
                measure_pressure_shock(0)
            # A segment resumed from a carrier does not re-capture its first frame (as on one card).
            if 0 in captures and moving_resume is None:
                capture(0)
            from woof.hex import time_ledger
            time_ledger.end_if_open("forecast.construct")
            started = time.perf_counter()
            done = 0
            totals_every = max(top, int(round(3600.0 / base_dt / top)) * top)
            try:
                while done < steps:
                    mark = time.perf_counter()
                    try:
                        active.advance_macro(top)
                    except Exception as error:
                        failure = {"step": done + top, "last_committed_shared_step": done,
                                   "exception": type(error).__name__, "message": str(error),
                                   "notes": list(getattr(error, "__notes__", ())),
                                   "traceback": traceback.format_exc()}
                        Path(output_root, "fullstep-lts-first-failure.json").write_text(
                            json.dumps(failure, indent=2) + "\n")
                        if not stop_on_refusal:
                            raise
                        refusal = {**failure, "final_state_retained": False,
                                   "reason": "a class step was refused; the classes no longer share a "
                                             "clock, so only frames from completed macro boundaries remain"}
                        break
                    elapsed = time.perf_counter() - mark
                    done += top
                    if first_call is not None and not first_call.done:
                        with time_ledger.stage("forecast.output.f000_first_call"):
                            hand_f000_first_call()
                    macro_rows.append({"step": done, "base_steps": top, "seconds": elapsed})
                    began = time.perf_counter()
                    rows = health_rows(done)
                    health.append(rows)
                    health_seconds += time.perf_counter() - began
                    if shock_on:
                        measure_pressure_shock(done)
                        shock_rows[-1]["vertical_velocity_abs_max"] = max(r["vertical_velocity_abs_max"] for r in rows)
                    print(json.dumps({"step": done, "seconds": elapsed, "time_seconds": clock_offset + done * base_dt,
                                      "fullstep_lts_blocks": len(stacks), "cards": len(set(devices)),
                                      "vertical_velocity_abs_max": max(r["vertical_velocity_abs_max"] for r in rows)}),
                          flush=True)
                    if done % totals_every == 0 or done == steps:
                        totals(done)
                        sample_pool()
                    if radar.due(done) and done not in captures:
                        capture_radar(done)
                    if done in captures:
                        capture(done)
                        if moving_control is not None and moving_control.at_frame(
                                model_seconds=clock_offset + done * base_dt, final=done >= steps,
                                seal=recovery_seal):
                            if not (done % totals_every == 0 or done == steps):
                                totals(done)
                            break
                if first_call is not None and held_f000:
                    first_call.put(None)  # the first macro was refused: publish f000 as captured
                    publish(0, held_f000.pop(), boundary_done=True)
            finally:
                if first_call is not None:
                    first_call.put(None)
                radar.close()
                loop_seconds = time.perf_counter() - started
            if refusal is None and not active.registers_empty():
                raise RuntimeError("an interface register is not empty at the end of the run")
            moving_export = None
            if refusal is None and os.environ.get("WOOF_HEX_MOVING_CHECKPOINT_OUT"):
                from woof.hex.moving_checkpoint import export_boundary_blocks
                moving_export = export_boundary_blocks(
                    os.environ["WOOF_HEX_MOVING_CHECKPOINT_OUT"], active=active, stacks=stacks,
                    previous=previous, forcing=forcing, host=host, physics_scope=physics_scope)
            sample_pool()
            integration = sum(row["seconds"] for row in macro_rows)
            warm = macro_rows[1:]
            first_mass = series[0]["dry_mass"]
            moving_fields = {} if not moving_requested else {
                "moving_boundary_export": moving_export,
                "moving_remap": moving_receipt,
                "moving_control": None if moving_control is None else moving_control.receipt(),
                "clock_offset_seconds": clock_offset,
                "remap_shock_series": shock_rows,
            }
            lts_receipt = {
                "schema": "gpuwm-hex.fullstep-lts-forecast/v2",
                "dispatch": "rate classes cut into blocks on several cards",
                **moving_fields,
                "devices": list(devices),
                "base_dt": base_dt,
                "rate_assignment": rate_info,
                "rates_sha256": _sha(np.ascontiguousarray(rates)),
                "clock": clock,
                "active": active.receipt(),
                "block_physics": [
                    {"block": t.gid, "class": t.index, "rate": int(t.rate), "card": t.device,
                     "step_seconds": stacks[t.gid]["class_dt"],
                     "owned_columns": int(t.layout.n_owned_cells), "local_columns": int(t.layout.n_local_cells),
                     **class_calls[t.gid]} for t in bplan.blocks],
                "dry_mass_series": series,
                "dry_mass_relative_drift": abs(series[-1]["dry_mass"] / first_mass - 1.0),
                "dry_mass_max_relative_excursion": max(abs(row["dry_mass"] / first_mass - 1.0) for row in series),
                "registers_empty_at_end": bool(active.registers_empty()),
                "restart_identity": str(identity_path),
                "construct_seconds": construct_seconds,
                "memory_pool_peak_bytes_per_card": {str(k): v for k, v in peak_pool.items()},
                "macros": macro_rows,
                "warm_seconds_per_forecast_hour": (
                    sum(row["seconds"] for row in warm) / (sum(row["base_steps"] for row in warm) * base_dt) * 3600.0
                    if warm else None),
            }
            Path(output_root, RECEIPT_NAME).write_text(json.dumps(lts_receipt, indent=2, default=str) + "\n")
            result = {
                "schema": "gpuwm-hex.fullstep-lts-forecast/v2",
                "fullstep_lts": {key: value for key, value in lts_receipt.items() if key != "macros"},
                "steps_executed": done, "steps_requested": steps,
                "arwen_pin": arwen_pin, "memory": memory,
                "step_health": health, "health": health,
                "physical_gates": physical_gates, "refusal": refusal,
                "schedule": {key: value for key, value in schedule.items() if key != "labels"},
                "history_labels": schedule["labels"],
                "walls": {
                    "steps": done, "integration_seconds": integration,
                    "integration_note": "sum of macro walls (every event ends with all cards synchronized); excludes "
                                        "construction, health, conservation totals, capture and history writing",
                    "first_step_seconds": macro_rows[0]["seconds"] if macro_rows else None,
                    "seconds_per_step_after_first": (
                        sum(row["seconds"] for row in warm) / sum(row["base_steps"] for row in warm)
                        if warm else None),
                    "loop_seconds_including_io": loop_seconds,
                    "snapshot_capture_seconds": capture_seconds,
                    "history_write_seconds": write_seconds,
                    "health_gate_seconds": health_seconds,
                    "forecast_seconds": done * base_dt,
                },
                "snapshot_files": frame_records, "all_published_field_hashes": frame_hashes,
                "physics_resource_scope": None,
            }
        finally:
            active.close()
    result["physics_resource_scope"] = physics_scope.receipt()
    return result
