"""Dry convective boundary layer: one closure against another.

A flat, horizontally homogeneous, dry column set under a surface warmer
than the air above it.  The surface layer converts that contrast into a
positive heat flux, the boundary-layer closure mixes it upward, and a
well-mixed layer grows into the overlying stable stratification.  This
is the oldest and least ambiguous boundary-layer test there is: the
answer has a known SHAPE (a near-adiabatic mixed layer capped by an
inversion that rises through the morning) even where no reference run
exists to compare against number by number.

Why it is in the tree
---------------------
It is the smallest configuration in which two boundary-layer schemes can
be asked the same question and their answers laid side by side.  The
gates below therefore check VALIDITY -- did the closure produce a
physically admissible boundary layer at all -- and deliberately do NOT
encode a preferred profile.  A scheme that mixes more deeply than
another is not thereby wrong, and this case is not the instrument that
would decide it.

Both the WRF-transcribed schemes and the experimental ArWen-only closure
can drive it, selected exactly as a user selects them: through
``bl_pbl_physics``.  ``run`` executes one closure and returns its gate
metrics; ``compare`` runs two and returns both metric sets plus the
profile pair, which is what makes the side-by-side readable.

Setup
-----
Doubly periodic 16 x 16 columns at dx = 500 m, 40 levels to 2 km
(dz ~ 50 m), initial theta 300 K with a 3 K/km stable lapse rate, at
rest apart from a light 2 m/s geostrophic-scale wind so the surface
layer has a shear scale to work with, land surface at a fixed 305 K --
5 K warmer than the air it touches.  Two hours of model time.  No
moisture, no radiation, no cumulus: the only thing moving heat upward is
the boundary-layer closure under test, which is the point.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from woof.config import RunConfig, validate_run_config
from woof.core.grid import (BaseState, VerticalCoord, make_base_state,
                             make_vertical_coord)
from woof.verify import gray_zone

#: Surface temperature (K) and the air temperature it sits under.  The
#: 5 K contrast is what drives the whole case.
TSK = 305.0
THETA_SURFACE = 300.0
#: Stable lapse rate the mixed layer grows into (K/m).
LAPSE = 0.003
#: Initial wind (m/s).  Small but nonzero: a surface layer with no shear
#: scale produces no friction velocity, and the closures under test read
#: one.
U_INITIAL = 2.0
#: Model seconds.
RUN_SECONDS = 7200.0

#: Pass gates, metric -> (lo, hi), None = unbounded on that side.
#:
#: VALIDITY, not skill.  Each bound below is a statement about whether a
#: convective boundary layer formed and stayed physical, and every one of
#: them is wide enough to admit closures that disagree with each other:
#:
#: ``nan`` is deliberately NOT a gate row: the driver checks it first
#: and by itself, and the returned metrics carry it beside the gate keys
#: (the case-registry contract).  Gates here are OPEN intervals with
#: strict bounds, ``lo < v < hi``, which is why a zero-valued metric can
#: never be gated with a zero bound.
#:
#: * ``theta_surface_warming`` -- the surface heat flux reached the air.
#:   A closure that mixes nothing leaves this at zero.
#: * ``mixed_layer_depth`` -- an adiabatic layer formed and is bounded by
#:   the domain.  The lower bound rejects "no boundary layer at all"; the
#:   upper rejects a layer that has run away through the model top.
#: * ``theta_gradient_aloft`` -- the stable stratification above the
#:   mixed layer SURVIVED.  This is the gate that catches a closure
#:   mixing the whole column indiscriminately, which is the classic
#:   failure and is invisible in a depth check alone.
#: * ``wind_speed_max`` -- momentum stayed bounded; surface drag did not
#:   run away or reverse.
#: * ``surface_heat_consistency`` -- the column's integrated heat GAIN,
#:   divided by an independent bulk-aerodynamic estimate of what the
#:   surface could have delivered over the run (C_H |V| dTheta t, with
#:   C_H |V| = 0.02 m/s at this case's 2 m/s wind).  Order unity is the
#:   correct answer and the band says so: near zero means the closure
#:   moved no heat off the surface, and far above one means it created
#:   heat the surface never supplied.  The band is deliberately a factor
#:   of a few wide -- the denominator is an estimate, not a budget, and
#:   this gate is here to catch a broken energy path, not to score one
#:   closure against another.
GATES = {
    "theta_surface_warming": (0.05, 20.0),
    "mixed_layer_depth": (100.0, 1800.0),
    "theta_gradient_aloft": (1.0e-3, None),
    "wind_speed_max": (None, 30.0),
    "surface_heat_consistency": (0.2, 3.0),
}


def default_config(bl_pbl_physics: int = 1) -> RunConfig:
    """The case configuration for one boundary-layer closure.

    ``km_opt`` follows the closure: the experimental closure supplies the
    mixing the km_opt operator would otherwise apply, and stacking the
    two would double-count it, so it is refused at config load.  Reading
    the requirement off the registry rather than hard-coding a number
    keeps this case accurate if another such scheme is ever registered.
    """
    from woof.config import SASE_PBL_SCHEME

    supplies_own_mixing = bl_pbl_physics == SASE_PBL_SCHEME
    return validate_run_config(RunConfig(
        nx=16, ny=16, nz=40, dx=500.0, dy=500.0, ztop=2000.0,
        dt=3.0, run_seconds=RUN_SECONDS, time_step_sound=4,
        # The state carries moisture ARRAYS (every boundary-layer
        # scheme in this engine mixes water vapour and is refused a dry
        # state) but the sounding is dry and no microphysics runs, so
        # the case stays thermodynamically a dry CBL.
        moist=True, mp_physics=0,
        bl_pbl_physics=bl_pbl_physics,
        sf_sfclay_physics=91,          # classic MM5 surface layer
        sf_surface_physics=0,          # prescribed surface temperature
        km_opt=0 if supplies_own_mixing else 4,
        c_s=0.25 if not supplies_own_mixing else 0.25,
        bldt=0.0, radt=0.0, cu_physics=0,
        case="cbl_dry"))


def sounding(z) -> np.ndarray:
    """Stably stratified dry column the mixed layer must grow into."""
    return THETA_SURFACE + LAPSE * np.asarray(z, dtype=np.float64)


def build(cfg: RunConfig, coord: VerticalCoord, base: BaseState, *,
          seed: int = 20260801):
    """At rest apart from a light uniform wind; no theta perturbation.

    The boundary layer is driven entirely by the surface contrast, so
    there is nothing to seed: any structure that appears was made by the
    closure under test.  A small deterministic wind perturbation breaks
    exact horizontal homogeneity so a three-dimensional closure has
    something to sense, without seeding the thermodynamics.

    ``seed`` selects the wind-perturbation draw and nothing else; the
    default is the case's registered seed, so ``run`` is bitwise the run
    it always was.  Independent draws through this one path are how the
    partition sweep below builds its seed ensembles (campaign spec
    2026-08-02, I1 "through the case's existing perturbation path").
    """
    import cupy as cp

    from woof.core.state import init_at_rest

    state = init_at_rest(cfg, coord, base)
    rng = np.random.default_rng(int(seed))
    field = np.full(tuple(state.u.shape), U_INITIAL, dtype=np.float64)
    field += 0.01 * rng.standard_normal(field.shape)
    field[:, :, -1] = field[:, :, 0]                    # periodic seam
    state.u[...] = cp.asarray(field, dtype=state.u.dtype)
    return state


def _profiles(state, cfg) -> dict[str, np.ndarray]:
    """Horizontally averaged column profiles, host-side FP64."""
    import cupy as cp

    def mean(field):
        return np.asarray(cp.asnumpy(field), dtype=np.float64).mean(
            axis=(1, 2))

    theta = mean(state.thp) + np.asarray(
        cp.asnumpy(state.thb), dtype=np.float64).reshape(-1)[:cfg.nz]
    phb = np.asarray(cp.asnumpy(state.phb), dtype=np.float64)
    php = np.asarray(cp.asnumpy(state.php), dtype=np.float64)
    phi = phb.reshape(-1, 1, 1) + php if phb.ndim == 1 else phb + php
    z_face = phi.mean(axis=(1, 2)) / 9.81
    z = 0.5 * (z_face[:-1] + z_face[1:])
    u = mean(state.u[:, :, :cfg.nx])
    v = mean(state.v[:, :cfg.ny, :])
    out = {"z": z, "theta": theta, "u": u, "v": v,
           "speed": np.hypot(u, v)}
    e_sgs = getattr(state, "e_sgs", None)
    if e_sgs is not None:
        out["tke"] = mean(e_sgs)
    return out


def _metrics(initial, final, cfg) -> dict[str, float]:
    """Gate metrics from the initial/final horizontally averaged state."""
    z, theta = final["z"], final["theta"]
    warming = float(theta[0] - initial["theta"][0])
    # Mixed-layer depth: the highest level whose potential temperature is
    # still within 0.5 K of the surface value.  A blunt definition on
    # purpose -- it makes no assumption about which closure produced it.
    within = np.nonzero(np.abs(theta - theta[0]) <= 0.5)[0]
    depth = float(z[within[-1]]) if within.size else 0.0
    # Stratification retained above the mixed layer, over the top
    # quarter of the column.
    top = slice(int(0.75 * cfg.nz), cfg.nz)
    gradient = float(np.gradient(theta[top], z[top]).mean())
    speed_max = float(np.max(final["speed"]))
    # Column heat gain against an INDEPENDENT bulk-aerodynamic estimate
    # of the surface supply.  The estimate is deliberately not taken
    # from the model's own flux: a check that reads back the number it
    # is checking proves nothing.
    d_theta = float(np.trapezoid(theta - initial["theta"], z))
    supply = abs(TSK - THETA_SURFACE) * 0.02 * RUN_SECONDS
    consistency = d_theta / supply if supply > 0.0 else 0.0
    nan = float(not np.all(np.isfinite(np.concatenate(
        [theta, final["u"], final["v"]]))))
    return {"nan": nan,
            "theta_surface_warming": warming,
            "mixed_layer_depth": depth,
            "theta_gradient_aloft": gradient,
            "wind_speed_max": speed_max,
            "surface_heat_consistency": consistency}


def run(outdir=None, *, bl_pbl_physics: int = 1) -> dict[str, float]:
    """Integrate one closure and return its gate metrics.

    With ``outdir`` the horizontally averaged initial and final profiles
    are written beside the metrics as ``cbl_dry_<selector>.npz`` so a
    caller can plot the pair without re-running.
    """
    from woof.core.dycore import run_steps
    from woof.core.physics import initialize_physics

    cfg = default_config(bl_pbl_physics)
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, sounding, p_surf=cfg.p_surf,
                           ztop=cfg.ztop)
    state = build(cfg, coord, base)
    initialize_physics(state, cfg, landmask=1.0, tsk=TSK)
    initial = _profiles(state, cfg)
    run_steps(state, cfg, int(round(RUN_SECONDS / cfg.dt)))
    final = _profiles(state, cfg)
    metrics = _metrics(initial, final, cfg)
    if outdir is not None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        np.savez(outdir / f"cbl_dry_{int(bl_pbl_physics)}.npz",
                 **{f"initial_{k}": v for k, v in initial.items()},
                 **{f"final_{k}": v for k, v in final.items()})
    return metrics


def compare(outdir=None, *, schemes=(1,)) -> dict[str, object]:
    """Run several closures on the identical case and return both sets.

    Differences are RETURNED, never reduced to a verdict: this case
    states whether each closure produced a physically admissible
    convective boundary layer, and leaves which one is better to an
    instrument that can actually answer it.
    """
    results: dict[str, object] = {"gates": dict(GATES), "schemes": {}}
    for selector in schemes:
        metrics = run(outdir, bl_pbl_physics=int(selector))
        passed = all(
            (lo is None or metrics[name] >= lo)
            and (hi is None or metrics[name] <= hi)
            for name, (lo, hi) in GATES.items())
        results["schemes"][str(int(selector))] = {
            "metrics": metrics, "pass": bool(passed)}
    return results


# ---------------------------------------------------------------------------
# Gray-zone partition sweep (campaign deliverable D1).
#
# The instrument registered by docs/internal/specs/
# 2026-08-02-sase-grayzone-campaign.md section 2 (I1): the same dry CBL,
# widened to 64 x 64 columns and run across a declared dx ladder, scored
# by the Honnert et al. (2011) partition construction that
# ``woof.verify.gray_zone`` transcribes.  The reference module stays a
# reference -- this runner CONSUMES it for scoring and nothing model-side
# reads it (the gray_zone.py firewall).
# ---------------------------------------------------------------------------

#: Columns per side of the sweep domain.  64 x 64 keeps many cells per
#: boundary-layer depth at every rung (the requirement Honnert's eq. (7)
#: plane-mean construction carries), and is the width the measured
#: baseline in handoffs/SASE-STATE-OF-THE-ART-20260802.md section 6 used.
SWEEP_COLUMNS = 64

#: The declared dx ladder [m], coarse to fine (campaign spec I1).  The
#: 3200 m rung is ADVISORY: outside the gray zone (Delta/h > 2) with 64
#: points across a 205 km domain -- recorded, never gated.
SWEEP_DX = (3200.0, 1600.0, 800.0, 400.0, 200.0, 100.0)

#: Model seconds per sweep run, and the scored tail.  4 h with the FINAL
#: HOUR scored answers the literature's slow gray-zone spin-up caveat
#: (campaign spec I1, declared change (i) from the 2 h handoff sweep).
SWEEP_SECONDS = 14400.0
SWEEP_SCORE_SECONDS = 3600.0
#: Scoring cadence inside the final hour: 12 snapshots at 300 s.
SWEEP_SAMPLE_SECONDS = 300.0

#: Time step per rung [s].  The case's registered operating point is
#: 3 s at dx = 500 m (6e-3 s/m, the 6*dx acoustic-split rule of thumb
#: this engine inherits with time_step_sound = 4).  Each rung scales
#: that ratio with dx and then snaps DOWN where needed so every dt
#: divides the 300 s scoring cadence, the 3 h spin-up and the 4 h run
#: into whole steps: never above the registered ratio, so every rung is
#: at least as stability-conservative as the case itself.
SWEEP_DT = {3200.0: 15.0, 1600.0: 7.5, 800.0: 4.0,
            400.0: 2.4, 200.0: 1.2, 100.0: 0.6}

#: Seed ensemble for the perturbation path in :func:`build`; n = 6 per
#: rung per leg (campaign spec I1).  The first member IS the case's
#: registered seed, so ensemble member 0 at the case's own geometry is
#: the case run itself.
SWEEP_SEEDS = (20260801, 20260802, 20260803, 20260804, 20260805, 20260806)

#: Scored window, z/h: Honnert et al. (2011) eq. (9)'s published
#: validity range for the MIXED-LAYER fit (gray_zone.TKE_MIXED_LAYER).
MIXED_LAYER_WINDOW = (0.05, 0.85)
#: Advisory window, z/h: eq. (10)'s entrainment-zone validity range.
ENTRAINMENT_WINDOW = (0.85, 1.10)


def sweep_config(dx_m: float, bl_pbl_physics: int) -> RunConfig:
    """One rung's configuration: the case at sweep width and spacing.

    Everything except nx/ny, dx/dy, dt and the run length is
    :func:`default_config` verbatim -- same sounding, same surface
    contrast, same km_opt rule -- so a sweep run at 500 m differs from
    the registered case only by domain width and duration.
    """
    from woof.config import SASE_PBL_SCHEME

    dx_m = float(dx_m)
    if dx_m not in SWEEP_DT:
        raise ValueError(f"dx={dx_m} is not a declared sweep rung "
                         f"{sorted(SWEEP_DT)}")
    # Every non-SASE closure falls through to km_opt=4, and for
    # Shin-Hong (11) that fall-through is the DECLARED reading, not an
    # accident: vertical transport from the PBL scheme with horizontal
    # Smagorinsky is the configuration class Shin & Hong (2015) ran
    # (registered in docs/internal/specs/
    # 2026-08-03-shinhong-grayzone-expectation.md), so the predicate
    # stays exactly SASE-only.
    supplies_own_mixing = bl_pbl_physics == SASE_PBL_SCHEME
    return validate_run_config(RunConfig(
        nx=SWEEP_COLUMNS, ny=SWEEP_COLUMNS, nz=40,
        dx=dx_m, dy=dx_m, ztop=2000.0,
        dt=SWEEP_DT[dx_m], run_seconds=SWEEP_SECONDS, time_step_sound=4,
        moist=True, mp_physics=0,
        bl_pbl_physics=bl_pbl_physics,
        sf_sfclay_physics=91,
        sf_surface_physics=0,
        km_opt=0 if supplies_own_mixing else 4,
        c_s=0.25,
        bldt=0.0, radt=0.0, cu_physics=0,
        case="cbl_dry"))


def window_mean(values, z, h: float, window) -> float:
    """Mean of a per-level quantity over a z/h window, level-weighted.

    The levels whose centers satisfy ``window[0] <= z/h <= window[1]``
    enter with equal weight -- the plain construction, no interpolation
    at the window edges, declared as such.  NaN when the window catches
    no level (an h below the first layer center would do it), so a
    degenerate run scores as unusable rather than as something.
    """
    v = np.asarray(values, dtype=np.float64)
    zh = np.asarray(z, dtype=np.float64) / float(h)
    lo, hi = float(window[0]), float(window[1])
    mask = (zh >= lo) & (zh <= hi) & np.isfinite(v)
    return float(v[mask].mean()) if mask.any() else float("nan")


def sweep_band(envelope, sigma_seed: float, n: int = 6):
    """The campaign's declared band rule (spec I1).

    ``[env_lo - 2*sigma_seed/sqrt(n), env_hi + 2*sigma_seed/sqrt(n)]``:
    the published envelope is the observations' own scatter, and the
    additive term is the standard error of the n-seed mean at two
    sigma -- the band is accurate about our own noise without ever being
    narrower than the published spread.
    """
    lo, hi = (float(v) for v in envelope)
    half = 2.0 * float(sigma_seed) / float(np.sqrt(n))
    return lo - half, hi + half


def _destagger(state, cfg):
    """Mass-point FP64 u, v, w, theta, e from the staggered device state."""
    import cupy as cp

    def host(a):
        return np.asarray(cp.asnumpy(a), dtype=np.float64)

    u = host(state.u)
    v = host(state.v)
    w = host(state.w)
    u_m = 0.5 * (u[:, :, :cfg.nx] + u[:, :, 1:cfg.nx + 1])
    v_m = 0.5 * (v[:, :cfg.ny, :] + v[:, 1:cfg.ny + 1, :])
    w_m = 0.5 * (w[:cfg.nz] + w[1:cfg.nz + 1])
    thb = host(state.thb).reshape(-1)[:cfg.nz]
    theta = thb[:, None, None] + host(state.thp)
    e_sgs = getattr(state, "e_sgs", None)
    if e_sgs is None:
        raise ValueError(
            "the partition instrument needs a closure that carries a "
            "prognostic subgrid energy field (state.e_sgs)")
    return u_m, v_m, w_m, theta, host(e_sgs)


def _sample_partition(state, cfg) -> dict[str, object]:
    """One scoring snapshot: Honnert eq. (7) partition + the case's own h.

    h is the domain-mean S3-6f bulk-Richardson boundary-layer depth
    (:func:`woof.verify.sase_ref.bulk_richardson_zi`, the authority
    the closure's own partition cap reads), computed host-side in FP64
    from the same de-staggered fields the partition uses -- the spec's
    "h from the case's own bulk-Richardson diagnostic".
    """
    import cupy as cp

    from woof.verify.sase_ref import bulk_richardson_zi

    u_m, v_m, w_m, theta, e_sgs = _destagger(state, cfg)
    phb = np.asarray(cp.asnumpy(state.phb), dtype=np.float64)
    php = np.asarray(cp.asnumpy(state.php), dtype=np.float64)
    phi = phb.reshape(-1, 1, 1) + php if phb.ndim == 1 else phb + php
    z_face = phi.mean(axis=(1, 2)) / 9.81
    z = 0.5 * (z_face[:-1] + z_face[1:])
    part = gray_zone.partition_from_profiles(e_sgs, u_m, v_m, w_m)
    h = float(np.mean(bulk_richardson_zi(
        u_m, v_m, theta, z[:, None, None])))
    frac = part["subgrid_fraction"]
    return {"z": z, "h": h,
            "subgrid_fraction": frac,
            "e_resolved": part["e_resolved"],
            "e_subgrid": part["e_subgrid"],
            "mixed_layer_fraction": window_mean(
                frac, z, h, MIXED_LAYER_WINDOW),
            "entrainment_fraction": window_mean(
                frac, z, h, ENTRAINMENT_WINDOW),
            "column_fraction": part["subgrid_fraction_column"]}


def sweep_digest(samples) -> str:
    """SHA-256 over the raw sample arrays, for the determinism pair.

    Two runs of the same seed on the same card must produce IDENTICAL
    bytes here -- the 5090 dual-run screen (no ECC; bitwise comparison
    is the corruption detector).
    """
    import hashlib

    digest = hashlib.sha256()
    for s in samples:
        for key in ("z", "subgrid_fraction", "e_resolved", "e_subgrid"):
            digest.update(np.ascontiguousarray(
                np.asarray(s[key], dtype=np.float64)).tobytes())
        digest.update(np.float64(s["h"]).tobytes())
    return digest.hexdigest()


def partition_run(dx_m: float, seed: int, *, bl_pbl_physics: int,
                  outdir=None, tag: str = "") -> dict[str, object]:
    """One sweep member: integrate 4 h, score the final hour.

    Returns the run receipt: the scored scalar (final-hour mean of the
    per-snapshot mixed-layer subgrid TKE fraction), the run's own
    abscissa ``x = dx/h`` at its measured h, the published target and
    envelope evaluated AT that x, the advisory arms (entrainment-zone
    fraction, column-integrated fraction -- the handoff table's
    quantity), the per-snapshot f ledger retained by the driver
    (f_solved / f_cap / f_w / f_used / zi -- what makes "binding"
    checkable), and the determinism digest.
    """
    from woof.core.dycore import run_steps
    from woof.core.physics import initialize_physics

    cfg = sweep_config(dx_m, bl_pbl_physics)
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, sounding, p_surf=cfg.p_surf,
                           ztop=cfg.ztop)
    state = build(cfg, coord, base, seed=seed)
    driver = initialize_physics(state, cfg, landmask=1.0, tsk=TSK)
    spin_steps = int(round((SWEEP_SECONDS - SWEEP_SCORE_SECONDS) / cfg.dt))
    chunk = int(round(SWEEP_SAMPLE_SECONDS / cfg.dt))
    n_samples = int(round(SWEEP_SCORE_SECONDS / SWEEP_SAMPLE_SECONDS))
    run_steps(state, cfg, spin_steps)
    samples, ledgers = [], []
    for k in range(n_samples):
        run_steps(state, cfg, chunk)
        sample = _sample_partition(state, cfg)
        sample["t"] = (spin_steps + (k + 1) * chunk) * cfg.dt
        samples.append(sample)
        ledger = getattr(driver, "last_sase_ledger", None)
        if ledger is not None:
            ledgers.append({key: float(ledger[key]) for key in
                            ("f", "f_solved", "f_cap", "f_w", "zi")
                            if key in ledger})
    scored = float(np.mean([s["mixed_layer_fraction"] for s in samples]))
    h_run = float(np.mean([s["h"] for s in samples]))
    x_run = float(dx_m) / h_run
    env_lo, env_hi = (float(v) for v in
                      gray_zone.subgrid_tke_envelope(x_run))
    receipt = {
        "dx": float(dx_m), "seed": int(seed),
        "bl_pbl_physics": int(bl_pbl_physics), "tag": str(tag),
        "dt": cfg.dt, "columns": SWEEP_COLUMNS, "nz": cfg.nz,
        "run_seconds": SWEEP_SECONDS,
        "scored_mixed_layer_fraction": scored,
        "h": h_run, "x": x_run,
        "target": float(gray_zone.subgrid_tke_fraction(x_run)),
        "envelope": (env_lo, env_hi),
        "entrainment_fraction": float(np.mean(
            [s["entrainment_fraction"] for s in samples])),
        "entrainment_target": float(gray_zone.subgrid_tke_fraction(
            x_run, entrainment_zone=True)),
        "column_fraction": float(np.mean(
            [s["column_fraction"] for s in samples])),
        "samples": [{"t": s["t"], "h": s["h"],
                     "mixed_layer_fraction": s["mixed_layer_fraction"],
                     "entrainment_fraction": s["entrainment_fraction"],
                     "column_fraction": s["column_fraction"]}
                    for s in samples],
        "ledger": ledgers,
        "digest": sweep_digest(samples),
    }
    if outdir is not None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        stem = (f"cbl_dry_partition_{int(dx_m)}m_seed{int(seed)}"
                + (f"_{tag}" if tag else ""))
        np.savez(outdir / f"{stem}.npz",
                 z=samples[-1]["z"],
                 subgrid_fraction=np.stack(
                     [s["subgrid_fraction"] for s in samples]),
                 e_resolved=np.stack([s["e_resolved"] for s in samples]),
                 e_subgrid=np.stack([s["e_subgrid"] for s in samples]),
                 h=np.array([s["h"] for s in samples]),
                 t=np.array([s["t"] for s in samples]))
    return receipt


def partition_sweep(outdir=None, *, bl_pbl_physics: int,
                    dx_ladder=SWEEP_DX,
                    seeds=SWEEP_SEEDS) -> dict[str, object]:
    """The full ladder: every rung x every seed, seed-mean scored.

    Returns per-rung receipts plus the seed-ensemble mean and stdev of
    the scored scalar -- sigma_seed is exactly the stdev reported here
    (ddof=1), the number the campaign's band rule consumes.
    """
    rungs = {}
    for dx_m in dx_ladder:
        runs = [partition_run(dx_m, seed, bl_pbl_physics=bl_pbl_physics,
                              outdir=outdir) for seed in seeds]
        scores = np.array([r["scored_mixed_layer_fraction"] for r in runs],
                          dtype=np.float64)
        rungs[str(int(dx_m))] = {
            "runs": runs,
            "mean": float(scores.mean()),
            "sigma_seed": float(scores.std(ddof=1)),
            "x_mean": float(np.mean([r["x"] for r in runs])),
        }
    return {"ladder": list(dx_ladder), "seeds": list(seeds),
            "bl_pbl_physics": int(bl_pbl_physics), "rungs": rungs}


def main(argv=None) -> int:
    """``python -m woof.verify.cases.cbl_dry [--out DIR] [SELECTOR...]``"""
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("selectors", nargs="*", type=int, default=[1],
                        help="bl_pbl_physics values to compare")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    report = compare(args.out, schemes=tuple(args.selectors))
    print(json.dumps(report, indent=2, sort_keys=True, default=float))
    return 0 if all(entry["pass"]
                    for entry in report["schemes"].values()) else 1


if __name__ == "__main__":                       # pragma: no cover
    raise SystemExit(main())
