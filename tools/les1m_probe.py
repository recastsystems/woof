"""Is a 1 m LES actually viable in WOOF-ArWen?  A measurement, not an estimate.

Four questions, one driver:

``footprint``  bytes/cell of the resident carrier set at an LES rung, and the
               largest box a given VRAM budget holds.  Feeds the extrapolation.
``cost``       ns/cell/step at the LES rung, medians of >=3 reps with the
               warmup discarded and a state digest on every timed
               configuration, so a transport that skipped work is visible.
``maxdt``      the largest dt that actually integrates, found by running, not
               by quoting a CFL formula.
``endurance``  thousands-to-100k steps at the working dt, with health,
               CFL and closure diagnostics sampled throughout.
``closure``    subgrid vs resolved TKE and the w spectrum, at a ladder of dx
               on ONE fixed physical box, so the closure's share is measured
               against resolution rather than asserted.

The case is ``woof.verify.cases.convective_boundary_layer`` -- the engine's
own em_les-shaped dry CBL (doubly periodic, flat, prescribed surface heat
flux, PBL off, km_opt 2 or 3).  That case is the LES oracle's counterpart, so
using anything else here would measure a setup nobody has validated.

The one thing this driver changes about the case is the SIZE of the
boundary layer: the shipped profile mixes to 1000 m, which at dx = 1 m would
need a 4 km box and ~4000 s of spin-up to develop.  ``_scaled_theta`` builds
the same shape at a chosen z_i so a 1 m run reaches statistically developed
turbulence inside a measurable number of steps.  Everything else -- the
forcing, the closure constants, the perturbation seeding, the topology -- is
the case's own.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from woof.config import RunConfig, validate_run_config
from woof.verify.cases import convective_boundary_layer as cbl


THETA0 = 300.0
#: Capping lapse above z_i, K/m.  3 K/km is the shipped profile's; a 1 m box
#: is 20x shallower, so the cap is strengthened to hold the same TOTAL
#: temperature jump across the inversion and keep the layer from eroding
#: through the lid inside the run.  Recorded because it is a case change.
CAP_LAPSE = 0.03


def _scaled_theta(zi0: float, lapse: float = CAP_LAPSE):
    """The case's profile shape at a chosen mixed-layer depth."""
    def profile(z):
        z = np.asarray(z, float)
        return THETA0 + np.where(z > zi0, lapse * (z - zi0), 0.0)
    return profile


#: Set from --moist.  The moist LES rung is qv transport plus a
#: microphysics scheme on top of the dry closure -- no cumulus, no PBL, no
#: radiation, no land surface, which is what an LES rung actually is.
_MOIST = {"on": False, "mp": 6, "seed": False}


def _seed_hydrometeors(state, cfg) -> None:
    """Put real condensate in the box so the microphysics does real work.

    WITHOUT this the moist rung is a measurement of nothing: the case's
    sounding is ~23% RH, so every WSM6 branch that costs time (condensation,
    accretion, melting, sedimentation) is skipped and the scheme reduces to
    its guard clauses.  A cost quoted from that would understate a moist
    tornado LES by whatever the scheme actually costs, which is the entire
    quantity of interest.  Cloud and rain are seeded at ordinary convective
    magnitudes and the vapour is raised to near saturation; ice species are
    filled where the state carries them.  This is a COST fixture, not a
    physical initial condition, and nothing physical is claimed from a run
    that uses it.
    """
    import cupy as cp
    rng = cp.random.RandomState(12345)
    shape = state.qc.shape

    def fill(name, amp):
        arr = getattr(state, name, None)
        if arr is not None and arr.shape == shape:
            arr[...] = (amp * (0.5 + rng.random_sample(shape))).astype(
                arr.dtype)

    fill("qc", 1.0e-3)
    fill("qr", 2.0e-4)
    fill("qi", 1.0e-4)
    fill("qs", 2.0e-4)
    fill("qg", 1.0e-4)
    if getattr(state, "qv", None) is not None:
        state.qv *= cp.asarray(3.5, dtype=state.qv.dtype)


def cfg_for(nx, ny, nz, dx, ztop, dt, km_opt, *, ts_sound=4, minutes=1.0,
            **over) -> RunConfig:
    cfg = cbl.make_config(nx=nx, ny=ny, nz=nz, dx=dx, ztop=ztop, dt=dt,
                          minutes=minutes, km_opt=km_opt)
    if ts_sound != cfg.time_step_sound:
        over.setdefault("time_step_sound", int(ts_sound))
    if _MOIST["on"]:
        over.setdefault("moist", True)
        over.setdefault("mp_physics", _MOIST["mp"])
    if over:
        cfg = replace(cfg, **over)
    return validate_run_config(cfg)


def build(cfg, zi0, seed=0, mean_wind=0.0):
    """The case's own builder, on the scaled profile.

    ``mean_wind`` adds a uniform u to the whole domain.  On a doubly
    periodic box a uniform translation is a Galilean shift and changes no
    physics -- but it DOES change the advective Courant number the
    5th-order scheme runs at, and that is the whole point.  The quiescent
    CBL runs at a horizontal advective CFL of order 1e-4 at dx = 1 m, so a
    dt limit measured on it is the ACOUSTIC limit and says nothing about a
    tornado.  100 m/s of mean wind puts the advection operator at the
    Courant number a 1 m tornado simulation would actually see.
    """
    saved = cbl._theta_profile
    cbl._theta_profile = _scaled_theta(zi0)
    try:
        state = cbl.build(cfg, seed=seed)
    finally:
        cbl._theta_profile = saved
    if _MOIST["seed"] and getattr(state, "qc", None) is not None:
        _seed_hydrometeors(state, cfg)
    if mean_wind:
        from woof.core.diagnostics import update_diagnostics
        state.u += float(mean_wind)
        # p/alpha depend on thp/php/mup only, so a uniform u leaves them
        # untouched; refreshed anyway so the state is unambiguously
        # consistent before the first step.
        update_diagnostics(state, cfg.hypsometric_opt)
    return state


# --------------------------------------------------------------------------
# health / diagnostics
# --------------------------------------------------------------------------

def cfl_numbers(state, cfg) -> dict:
    """Every CFL this configuration can be limited by, measured.

    ``stability_report`` returns the VERTICAL advective one.  At dx = 1 m the
    binding constraints are horizontal, and one of them is acoustic: the
    split-explicit scheme treats horizontal sound explicitly on the small
    step ``dt / time_step_sound``, so a 1 m grid can be sound-limited while
    the advective number still looks comfortable.  Reporting only the
    advective number would hide that.
    """
    import cupy as cp
    from woof.core.dycore import stability_report

    rep = stability_report(state, cfg)
    umax = float(cp.abs(state.u).max())
    vmax = float(cp.abs(state.v).max())
    wmax = float(cp.abs(state.w).max())
    thb = state.thb
    th = (thb if thb.ndim == 3 else thb[:, None, None]) + state.thp
    # Sound speed from the live pressure/temperature field.
    p = state.p.astype(cp.float64)
    alt = state.alt.astype(cp.float64) if getattr(state, "alt", None) \
        is not None else None
    if alt is not None:
        csnd = float(cp.sqrt(1.4 * p / cp.maximum(1.0 / cp.maximum(alt, 1e-12),
                                                  1e-12)).max())
    else:
        csnd = float(cp.sqrt(1.4 * 287.0 * th.max() * 0.9))
    ns = int(cfg.time_step_sound)
    dts = cfg.dt / ns
    return {
        "u_max": umax, "v_max": vmax, "w_max": wmax,
        "thp_absmax": float(cp.abs(state.thp).max()),
        "cfl_vert_adv": None if rep["cfl"] is None else float(rep["cfl"]),
        "cfl_horiz_adv": max(umax, vmax) * cfg.dt / cfg.dx,
        "cfl_sound_horiz": csnd * dts / cfg.dx,
        "sound_speed": csnd,
        "nan": bool(rep["nan"]),
    }


def closure_diag(state, cfg) -> dict:
    """Subgrid share of the turbulence, both closures.

    km_opt=2 carries a prognostic SGS TKE, so the ratio e_sgs / (e_sgs +
    E_res) is direct.  km_opt=3 has no TKE carrier at all; its subgrid
    energy is inferred from the live eddy viscosity through the same
    Smagorinsky relation the closure itself uses, e_sgs = (Km / (c_s l))^2,
    which is a DERIVED quantity and is labelled as such.
    """
    import cupy as cp
    nz = cfg.nz
    wm = 0.5 * (state.w[:-1] + state.w[1:]).astype(cp.float64)
    uc = state.u[:, :, :cfg.nx].astype(cp.float64)
    vc = state.v[:, :cfg.ny, :].astype(cp.float64)

    def slab(a):
        return a.mean(axis=(1, 2))

    up = uc - slab(uc)[:, None, None]
    vp = vc - slab(vc)[:, None, None]
    wp = wm - slab(wm)[:, None, None]
    e_res = 0.5 * (slab(up * up) + slab(vp * vp) + slab(wp * wp))
    out = {"e_res": cp.asnumpy(e_res)}

    if getattr(state, "tke", None) is not None:
        e_sgs = slab(state.tke.astype(cp.float64))
        out["e_sgs"] = cp.asnumpy(e_sgs)
        out["e_sgs_basis"] = "prognostic (km_opt=2)"
    else:
        kmv = state.existing_scratch("smag_kmv")
        if kmv is None:
            out["e_sgs"] = np.zeros(nz)
            out["e_sgs_basis"] = "unavailable"
        else:
            phb = state.phb
            phi = ((phb if phb.ndim == 3 else phb[:, None, None])
                   + state.php).astype(cp.float64)
            dz = cp.asnumpy((phi[1:] - phi[:-1]).mean(axis=(1, 2))) / 9.81
            ell = np.cbrt(cfg.dx * cfg.dy * np.maximum(dz, 1e-6))
            km = cp.asnumpy(slab(kmv.astype(cp.float64)))
            # km_opt=3 carries no SGS TKE, so it must be inferred from the
            # live eddy viscosity -- and the constant in that inversion is
            # a CHOICE, worth a factor of (c_s/c_k)^2 = 3.24 here.  The
            # 1.5-order model's own relation is Km = c_k l sqrt(e), so
            # e = (Km/(c_k l))^2 is the one that makes a km_opt=3 number
            # commensurable with km_opt=2's prognostic e; the Smagorinsky
            # constant c_s appears in Km = (c_s l)^2 |S| and is NOT the
            # right constant to invert with.  Both are reported so no
            # reader has to take the choice on trust.
            out["e_sgs"] = (km / np.maximum(cfg.c_k * ell, 1e-12)) ** 2
            out["e_sgs_cs"] = (km / np.maximum(cfg.c_s * ell, 1e-12)) ** 2
            out["e_sgs_basis"] = ("derived from live Km via "
                                  "e = (Km/(c_k l))^2, c_k="
                                  f"{cfg.c_k} (km_opt=3); the c_s-based "
                                  "inversion is in e_sgs_cs")
            out["km_mean"] = km
            out["mixing_length"] = ell
    kmv = state.existing_scratch("smag_kmv")
    if kmv is not None:
        out["km_max"] = float(kmv.max())
    # The case module's OWN closure-independent measure (its docstring
    # names it): the resolved and subgrid heat fluxes, available under
    # both closures because the SGS one comes from the live khv field
    # rather than from a TKE carrier only km_opt=2 has.  Wrapped because a
    # diagnostic must never be able to kill a 100 000-step integration.
    try:
        prof = cbl._slab_profiles(state, cfg)
        wth_res = np.asarray(prof["wth_res"], float)
        wth_sgs = np.asarray(prof["wth_sgs"], float)
        wth_sgs_mass = 0.5 * (wth_sgs[:-1] + wth_sgs[1:])
        qs = max(float(cfg.tke_heat_flux), 1e-30)
        out["wth_res_max_over_qs"] = float(wth_res.max() / qs)
        out["wth_sgs_max_over_qs"] = float(wth_sgs_mass.max() / qs)
        denom = np.abs(wth_res) + np.abs(wth_sgs_mass)
        top = max(2, min(len(denom) - 1, int(0.7 * cfg.nz)))
        out["sgs_flux_fraction"] = float(
            np.abs(wth_sgs_mass[1:top]).sum()
            / max(denom[1:top].sum(), 1e-30))
    except Exception as exc:                          # noqa: BLE001
        out["flux_split_error"] = f"{type(exc).__name__}: {exc}"
    return out


def w_spectrum(state, cfg, k_level: int):
    """Radially-averaged PSD of the w plane at one level, engine pins."""
    import cupy as cp
    from woof.verify import spectral
    plane = cp.asnumpy(
        0.5 * (state.w[k_level] + state.w[k_level + 1])).astype(np.float64)
    psd = spectral.radial_psd(plane, cfg.dx)
    return {"k": np.asarray(psd["wavenumber_cycles_per_m"], float),
            "power": np.asarray(psd["power"], float),
            "mode_count": np.asarray(psd["mode_count"], float)}


def digest(state) -> str:
    from tilestream.harness import hash_state
    return hash_state(state)[:16]


def vram() -> dict:
    import cupy as cp
    pool = cp.get_default_memory_pool()
    free, total = cp.cuda.runtime.memGetInfo()
    return {"pool_used_gib": pool.used_bytes() / 2 ** 30,
            "pool_total_gib": pool.total_bytes() / 2 ** 30,
            "device_used_gib": (total - free) / 2 ** 30,
            "device_total_gib": total / 2 ** 30}


# --------------------------------------------------------------------------
# modes
# --------------------------------------------------------------------------

def mode_footprint(args) -> dict:
    import cupy as cp
    from tilestream.physics_inventory import (carrier_bytes_per_cell,
                                              carrier_manifest)
    out = []
    for km_opt in args.km_opt:
        cfg = cfg_for(args.nx, args.ny, args.nz, args.dx, args.ztop,
                      args.dt, km_opt)
        cp.get_default_memory_pool().free_all_blocks()
        before = vram()
        state = build(cfg, args.zi, mean_wind=args.mean_wind)
        from woof.core.dycore import run_steps
        run_steps(state, cfg, 3)          # allocate every scratch buffer
        cp.cuda.Device().synchronize()
        after = vram()
        cells = cfg.nx * cfg.ny * cfg.nz
        man = carrier_manifest(state)
        row = {
            "km_opt": km_opt,
            "cells": cells,
            "carrier_bytes_per_cell": float(carrier_bytes_per_cell(state,
                                                                   cfg)),
            "n_carriers": len(man),
            "resident_bytes_per_cell":
                (after["pool_total_gib"] - before["pool_total_gib"])
                * 2 ** 30 / cells,
            "pool_total_gib": after["pool_total_gib"],
            "device_used_gib": after["device_used_gib"],
        }
        out.append(row)
        print(json.dumps(row), flush=True)
        del state
        cp.get_default_memory_pool().free_all_blocks()
    return {"footprint": out}


def _time_steps(state, cfg, n) -> float:
    import cupy as cp
    from woof.core.dycore import run_steps
    cp.cuda.Device().synchronize()
    t0 = time.perf_counter()
    run_steps(state, cfg, n)
    cp.cuda.Device().synchronize()
    return time.perf_counter() - t0


def mode_cost(args) -> dict:
    import cupy as cp
    rows = []
    for km_opt in args.km_opt:
        for (nx, ny, nz) in args.shapes:
            cfg = cfg_for(nx, ny, nz, args.dx, args.ztop, args.dt, km_opt)
            cells = nx * ny * nz
            cp.get_default_memory_pool().free_all_blocks()
            state = build(cfg, args.zi, mean_wind=args.mean_wind)
            _time_steps(state, cfg, args.warmup)      # discarded
            reps = [_time_steps(state, cfg, args.steps)
                    for _ in range(args.reps)]
            per_step = [r / args.steps for r in reps]
            med = statistics.median(per_step)
            fastest = min(per_step)
            spread = max(per_step) / min(per_step)
            row = {
                "km_opt": km_opt, "nx": nx, "ny": ny, "nz": nz,
                "cells": cells, "dx": cfg.dx, "dt": cfg.dt,
                "ms_per_step": med * 1e3,
                "ns_per_cell_step": med * 1e9 / cells,
                # Under a contended card the MEDIAN measures the other
                # tenants as much as this kernel.  The MINIMUM over reps is
                # the least-contended sample and is the accurate estimator of
                # the uncontended cost -- reported alongside, never instead.
                "ms_per_step_min": fastest * 1e3,
                "ns_per_cell_step_min": fastest * 1e9 / cells,
                "per_step_seconds": per_step,
                "reps": args.reps, "steps_per_rep": args.steps,
                "spread": spread,
                "spread_flag": "OVER 10%" if spread > 1.10 else "ok",
                "digest": digest(state),
                "steps_integrated": args.warmup + args.reps * args.steps,
                **{k: v for k, v in vram().items()
                   if k in ("pool_total_gib", "device_used_gib")},
                **{k: v for k, v in cfl_numbers(state, cfg).items()
                   if k in ("w_max", "cfl_horiz_adv", "cfl_sound_horiz",
                            "nan")},
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
            del state
            cp.get_default_memory_pool().free_all_blocks()
    return {"cost": rows}


def mode_maxdt(args) -> dict:
    """Largest dt that survives ``--probe-steps``, by running it.

    Every trial branches from ONE spun-up state, not from the seeded
    initial condition.  That matters more than the bisection does: a dt
    that integrates still air for 400 steps says nothing about a dt that
    has to integrate developed turbulence, and quoting the former as a
    stability limit is exactly the kind of result this project has had to
    retract before.  The spin-up runs at ``--dt`` (the reference, assumed
    safe and checked), its carriers are snapshotted, and each trial
    restores that snapshot into a fresh state before changing dt.

    A trial fails on a NaN, on a raise, or on a w that leaves the physical
    range.  The last criterion is deliberately crude and absolute: LES w in
    a 40 m CBL is O(1) m/s, so a trial that reaches ``--w-fail`` has left
    the physics whatever the arithmetic says.
    """
    import cupy as cp
    from woof.core.dycore import run_steps
    from woof.core.diagnostics import update_diagnostics
    from tilestream.physics_inventory import (load_carriers,
                                              snapshot_carriers)

    results = []
    for km_opt in args.km_opt:
        for ts_sound in args.ts_sound:
            # --- one spin-up, at the reference dt, shared by every trial
            base = cfg_for(args.nx, args.ny, args.nz, args.dx, args.ztop,
                           args.dt, km_opt, ts_sound=ts_sound)
            cp.get_default_memory_pool().free_all_blocks()
            spin = build(base, args.zi, mean_wind=args.mean_wind)
            run_steps(spin, base, 1)            # allocate lazy carriers
            if args.spinup_steps > 1:
                run_steps(spin, base, args.spinup_steps - 1)
            snap = snapshot_carriers(spin)
            spin_health = cfl_numbers(spin, base)
            print(json.dumps({"spinup": args.spinup_steps,
                              "km_opt": km_opt, "ts_sound": ts_sound,
                              "sim_seconds": args.spinup_steps * base.dt,
                              **{k: spin_health[k]
                                 for k in ("w_max", "u_max", "thp_absmax",
                                           "cfl_horiz_adv",
                                           "cfl_sound_horiz")}}),
                  flush=True)
            del spin
            cp.get_default_memory_pool().free_all_blocks()
            trials = {}

            def trial(dt):
                if dt in trials:
                    return trials[dt]
                cfg = cfg_for(args.nx, args.ny, args.nz, args.dx, args.ztop,
                              dt, km_opt, ts_sound=ts_sound)
                cp.get_default_memory_pool().free_all_blocks()
                state = build(cfg, args.zi, mean_wind=args.mean_wind)
                run_steps(state, cfg, 1)        # match the lazy manifest
                load_carriers(state, snap)
                update_diagnostics(state, cfg.hypsometric_opt)
                ok, note, peak = True, "survived", 0.0
                done = 0
                chunk = max(args.probe_steps // 20, 1)
                h = cfl_numbers(state, cfg)
                while done < args.probe_steps:
                    n = min(chunk, args.probe_steps - done)
                    try:
                        run_steps(state, cfg, n)
                    except Exception as exc:            # noqa: BLE001
                        ok, note = False, f"raised {type(exc).__name__}: {exc}"
                        break
                    done += n
                    h = cfl_numbers(state, cfg)
                    peak = max(peak, h["w_max"])
                    if h["nan"] or not math.isfinite(h["w_max"]):
                        ok, note = False, f"non-finite at step {done}"
                        break
                    if h["w_max"] > args.w_fail:
                        ok, note = False, (f"w_max {h['w_max']:.3g} m/s at "
                                           f"step {done}")
                        break
                rec = {"dt": dt, "ok": ok, "note": note, "w_peak": peak,
                       "steps": done,
                       "cfl_sound_horiz": h["cfl_sound_horiz"],
                       "cfl_horiz_adv": h["cfl_horiz_adv"]}
                trials[dt] = rec
                print(json.dumps({"km_opt": km_opt, "ts_sound": ts_sound,
                                  **rec}), flush=True)
                del state
                cp.get_default_memory_pool().free_all_blocks()
                return rec

            # geometric ladder up until failure, then bisect
            lo, hi = None, None
            dt = args.dt
            for _ in range(12):
                if trial(dt)["ok"]:
                    lo = dt
                    dt *= 1.5
                else:
                    hi = dt
                    break
            if lo is None:                    # even the start failed
                dt = args.dt
                for _ in range(12):
                    dt /= 1.5
                    if trial(dt)["ok"]:
                        lo, hi = dt, dt * 1.5
                        break
            if lo is not None and hi is not None:
                for _ in range(args.bisect):
                    mid = math.sqrt(lo * hi)
                    if trial(mid)["ok"]:
                        lo = mid
                    else:
                        hi = mid
            results.append({"km_opt": km_opt, "ts_sound": ts_sound,
                            "spinup_steps": args.spinup_steps,
                            "spinup_health": spin_health,
                            "largest_stable_dt": lo,
                            "smallest_failing_dt": hi,
                            "probe_steps": args.probe_steps,
                            "trials": list(trials.values())})
            print(json.dumps({"km_opt": km_opt, "ts_sound": ts_sound,
                              "LARGEST_STABLE_DT": lo,
                              "smallest_failing_dt": hi}), flush=True)
    return {"maxdt": results}


def mode_endurance(args) -> dict:
    import cupy as cp
    from woof.core.dycore import run_steps

    cfg = cfg_for(args.nx, args.ny, args.nz, args.dx, args.ztop, args.dt,
                  args.km_opt[0])
    state = build(cfg, args.zi, mean_wind=args.mean_wind)
    cells = cfg.nx * cfg.ny * cfg.nz
    mass0 = cbl._column_mass(state)
    z = state.height_half()
    z_mass = np.asarray(cp.asnumpy(z) if hasattr(z, "get") else z, float)
    if z_mass.ndim == 3:
        z_mass = z_mass.mean(axis=(1, 2))
    k_probe = int(np.argmin(np.abs(z_mass - 0.5 * args.zi)))

    samples = []
    done = 0
    step_seconds = 0.0
    t0 = time.perf_counter()
    while done < args.steps:
        n = min(args.sample_every, args.steps - done)
        # Timed SEPARATELY from the diagnostics below, so the endurance
        # run yields a second, independent ns/cell/step on a DEVELOPED
        # turbulent field -- the control on the cost mode, which times a
        # nearly laminar one.
        cp.cuda.Device().synchronize()
        tstep = time.perf_counter()
        run_steps(state, cfg, n)
        cp.cuda.Device().synchronize()
        step_seconds += time.perf_counter() - tstep
        done += n
        h = cfl_numbers(state, cfg)
        c = closure_diag(state, cfg)
        top = max(2, int(0.9 * args.zi / max(z_mass[1] - z_mass[0], 1e-6)))
        top = min(top, cfg.nz - 1)
        e_res_ml = float(np.mean(c["e_res"][1:top]))
        e_sgs_ml = float(np.mean(c["e_sgs"][1:top]))
        rec = {
            "step": done, "t_sim_s": done * cfg.dt,
            "wall_s": time.perf_counter() - t0,
            "step_ns_per_cell": step_seconds * 1e9 / (done * cells),
            **{k: h[k] for k in ("u_max", "w_max", "thp_absmax", "nan",
                                 "cfl_vert_adv", "cfl_horiz_adv",
                                 "cfl_sound_horiz")},
            "mass_drift_rel": abs(cbl._column_mass(state) - mass0)
                              / abs(mass0),
            "e_res_ml": e_res_ml, "e_sgs_ml": e_sgs_ml,
            "sgs_fraction": e_sgs_ml / max(e_res_ml + e_sgs_ml, 1e-30),
            "e_sgs_basis": c["e_sgs_basis"],
            "km_max": c.get("km_max"),
            "wth_res_max_over_qs": c.get("wth_res_max_over_qs"),
            "wth_sgs_max_over_qs": c.get("wth_sgs_max_over_qs"),
            "sgs_flux_fraction": c.get("sgs_flux_fraction"),
        }
        samples.append(rec)
        print(json.dumps(rec), flush=True)
        if rec["nan"] or not math.isfinite(rec["w_max"]):
            print("ABORT: non-finite state", flush=True)
            break
    wall = time.perf_counter() - t0
    spec = w_spectrum(state, cfg, k_probe)
    out = {
        "endurance": {
            "config": {"nx": cfg.nx, "ny": cfg.ny, "nz": cfg.nz,
                       "dx": cfg.dx, "ztop": cfg.ztop, "dt": cfg.dt,
                       "km_opt": cfg.km_opt, "c_s": cfg.c_s, "c_k": cfg.c_k,
                       "time_step_sound": cfg.time_step_sound,
                       "zi0": args.zi, "cells": cells},
            "steps_requested": args.steps, "steps_done": done,
            "sim_seconds": done * cfg.dt,
            "wall_seconds": wall,
            "step_seconds": step_seconds,
            "ns_per_cell_step": wall * 1e9 / (done * cells),
            "step_only_ns_per_cell_step": step_seconds * 1e9 / (done * cells),
            "digest": digest(state),
            "samples": samples,
            "spectrum_k": spec["k"].tolist(),
            "spectrum_power": spec["power"].tolist(),
            "spectrum_height_m": float(z_mass[k_probe]),
            "z_mass": z_mass.tolist(),
            "e_res_profile": closure_diag(state, cfg)["e_res"].tolist(),
            "e_sgs_profile": closure_diag(state, cfg)["e_sgs"].tolist(),
            **vram(),
        }
    }
    return out


def mode_closure(args) -> dict:
    """One physical box, a ladder of dx: does the closure's share fall?

    A closure that is doing its job hands more of the turbulence to the
    resolved field as dx shrinks.  One resolution cannot show that; a
    ladder on a FIXED physical box can, and the fixed box is what makes the
    comparison a resolution comparison rather than a different-case one.
    """
    import cupy as cp
    from woof.core.dycore import run_steps

    rows = []
    for km_opt in args.km_opt:
        for dx in args.dx_ladder:
            nx = int(round(args.box_m / dx))
            nz = int(round(args.ztop / (dx * args.aspect)))
            cfg = cfg_for(nx, nx, nz, dx, args.ztop,
                          args.dt * (dx / args.dx), km_opt)
            n_steps = int(round(args.sim_seconds / cfg.dt))
            cp.get_default_memory_pool().free_all_blocks()
            state = build(cfg, args.zi, mean_wind=args.mean_wind)
            z = state.height_half()
            z_mass = np.asarray(cp.asnumpy(z) if hasattr(z, "get") else z,
                                float)
            if z_mass.ndim == 3:
                z_mass = z_mass.mean(axis=(1, 2))
            t0 = time.perf_counter()
            done = 0
            chunk = max(n_steps // 10, 1)
            series = []
            while done < n_steps:
                n = min(chunk, n_steps - done)
                run_steps(state, cfg, n)
                done += n
                h = cfl_numbers(state, cfg)
                c = closure_diag(state, cfg)
                top = min(max(2, int(np.searchsorted(z_mass,
                                                     0.9 * args.zi))),
                          cfg.nz - 1)
                er = float(np.mean(c["e_res"][1:top]))
                es = float(np.mean(c["e_sgs"][1:top]))
                series.append({"step": done, "t_sim_s": done * cfg.dt,
                               "w_max": h["w_max"], "e_res_ml": er,
                               "e_sgs_ml": es,
                               "sgs_fraction": es / max(er + es, 1e-30)})
                if h["nan"]:
                    break
            wall = time.perf_counter() - t0
            c = closure_diag(state, cfg)
            k_probe = int(np.argmin(np.abs(z_mass - 0.5 * args.zi)))
            spec = w_spectrum(state, cfg, k_probe)
            top = min(max(2, int(np.searchsorted(z_mass, 0.9 * args.zi))),
                      cfg.nz - 1)
            er = float(np.mean(c["e_res"][1:top]))
            es = float(np.mean(c["e_sgs"][1:top]))
            row = {
                "km_opt": km_opt, "dx": dx, "nx": nx, "nz": nz,
                "dt": cfg.dt, "steps": done, "sim_seconds": done * cfg.dt,
                "cells": nx * nx * nz,
                "wall_seconds": wall,
                "ns_per_cell_step": wall * 1e9 / (done * nx * nx * nz),
                "e_res_ml": er, "e_sgs_ml": es,
                "sgs_fraction": es / max(er + es, 1e-30),
                "e_sgs_basis": c["e_sgs_basis"],
                "km_max": c.get("km_max"),
                "wth_res_max_over_qs": c.get("wth_res_max_over_qs"),
                "wth_sgs_max_over_qs": c.get("wth_sgs_max_over_qs"),
                "sgs_flux_fraction": c.get("sgs_flux_fraction"),
                "w_max": float(cp.abs(state.w).max()),
                "digest": digest(state),
                "spectrum_k": spec["k"].tolist(),
                "spectrum_power": spec["power"].tolist(),
                "spectrum_height_m": float(z_mass[k_probe]),
                "z_mass": z_mass.tolist(),
                "e_res_profile": c["e_res"].tolist(),
                "e_sgs_profile": np.asarray(c["e_sgs"]).tolist(),
                "series": series,
            }
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items()
                              if not isinstance(v, list)}), flush=True)
            del state
            cp.get_default_memory_pool().free_all_blocks()
    return {"closure": rows}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("footprint", "cost", "maxdt",
                                    "endurance", "closure"))
    p.add_argument("--nx", type=int, default=192)
    p.add_argument("--ny", type=int, default=192)
    p.add_argument("--nz", type=int, default=120)
    p.add_argument("--dx", type=float, default=1.0)
    p.add_argument("--ztop", type=float, default=160.0)
    p.add_argument("--dt", type=float, default=0.006)
    p.add_argument("--zi", type=float, default=40.0,
                   help="initial mixed-layer depth, m")
    p.add_argument("--km-opt", type=int, nargs="+", default=[3],
                   dest="km_opt")
    p.add_argument("--ts-sound", type=int, nargs="+", default=[4],
                   dest="ts_sound")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--sample-every", type=int, default=500)
    p.add_argument("--probe-steps", type=int, default=400)
    p.add_argument("--spinup-steps", type=int, default=4000,
                   help="maxdt mode: steps at --dt before branching, so "
                        "every trial faces DEVELOPED turbulence")
    p.add_argument("--bisect", type=int, default=5)
    p.add_argument("--w-fail", type=float, default=50.0)
    p.add_argument("--shapes", type=str, default=None,
                   help="cost mode: 'nx,ny,nz;nx,ny,nz;...'")
    p.add_argument("--dx-ladder", type=float, nargs="+",
                   default=[1.0, 2.0, 4.0], dest="dx_ladder")
    p.add_argument("--box-m", type=float, default=192.0)
    p.add_argument("--aspect", type=float, default=1.33,
                   help="closure mode: dz / dx")
    p.add_argument("--sim-seconds", type=float, default=120.0)
    p.add_argument("--mean-wind", type=float, default=0.0,
                   help="uniform u added to the box, m/s: a Galilean "
                        "shift that changes no physics but puts the "
                        "advection operator at a tornadic Courant number")
    p.add_argument("--moist", action="store_true",
                   help="add the moist rung: qv transport + microphysics")
    p.add_argument("--mp-physics", type=int, default=6)
    p.add_argument("--seed-hydrometeors", action="store_true",
                   help="fill qc/qr/qi/qs/qg and raise qv so the "
                        "microphysics actually runs its expensive branches "
                        "-- a COST fixture, never a physical state")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    if args.shapes:
        args.shapes = [tuple(int(v) for v in s.split(","))
                       for s in args.shapes.split(";")]
    else:
        args.shapes = [(args.nx, args.ny, args.nz)]

    if args.moist:
        _MOIST["on"] = True
        _MOIST["mp"] = int(args.mp_physics)
        _MOIST["seed"] = bool(args.seed_hydrometeors)

    fn = {"footprint": mode_footprint, "cost": mode_cost,
          "maxdt": mode_maxdt, "endurance": mode_endurance,
          "closure": mode_closure}[args.mode]
    result = fn(args)
    result["argv"] = vars(args) | {"out": str(args.out),
                                   "shapes": [list(s) for s in args.shapes]}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1, default=float)
                            + "\n", encoding="utf-8")
        print(f"WROTE {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
