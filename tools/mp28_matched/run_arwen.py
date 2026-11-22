"""Run the matched idealized periodic-boundary case in WOOF-ArWen.

The companion of the WRF side of the ``mp_physics=28`` matched-trajectory
comparison.  WOOF-ArWen is initialised FROM WRF's own ``wrfinput_d01`` -- the file
``ideal.exe`` writes -- so the two models start from the same discrete state
to float32, and the initialisation is removed as a source of divergence.
What is NOT removed, and is measured instead, is everything downstream: the
dycore, the transport of the aerosol species, and the microphysics itself.

The lateral boundaries are periodic in both models, so no boundary condition
enters the comparison at all.  That is the whole reason the case exists: with
specified boundaries there is no aerosol lateral BC in WOOF-ArWen and the two
models would be integrating different problems within the hour.

Usage:
    python run_arwen.py --wrfinput PATH --out DIR --mp {8,28}
                        [--steps N] [--frame-steps N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

#: Prognostic 3-D fields carried in every frame, ArWen name -> WRF name.
#: The mass species are WRF's ``moist`` package; the number species are its
#: ``scalar`` package.  ``nwfa``/``nifa`` are the aerosol the port exists for.
FRAME_FIELDS_MP8 = {
    "qv": "QVAPOR", "qc": "QCLOUD", "qr": "QRAIN",
    "qi": "QICE", "qs": "QSNOW", "qg": "QGRAUP",
    "nr": "QNRAIN", "ni": "QNICE",
}
FRAME_FIELDS_MP28 = dict(FRAME_FIELDS_MP8,
                         nc="QNCLOUD", nwfa="QNWFA", nifa="QNIFA")


def frame_fields(mp: int) -> dict[str, str]:
    return FRAME_FIELDS_MP28 if mp == 28 else FRAME_FIELDS_MP8


def read_wrfinput(path: Path, frame: int = 0) -> dict:
    """Everything the WOOF-ArWen state needs, as float64 numpy, (nz, ny, nx).

    ``frame`` selects a time index, so the same reader loads a ``wrfout``
    history frame and WOOF-ArWen can be restarted from a MATURE WRF state -- the
    only regime in which a "matched trajectory" can mean anything, because
    over a short window the two models have not had time to decorrelate.
    A flat idealized base state is time-invariant, so ``PHB``/``MUB``/
    ``T_INIT``/``PB``/``ALB`` are read from whichever file carries them.
    """
    import netCDF4

    out = {}
    with netCDF4.Dataset(path) as ds:
        def g(name):
            i = frame if ds.variables[name].shape[0] > frame else 0
            return np.asarray(ds.variables[name][i], dtype=np.float64)
        for name in ("U", "V", "W", "PH", "PHB", "T", "MU", "MUB",
                     "T_INIT", "PB", "ALB", "ZNU", "ZNW",
                     "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW",
                     "QGRAUP", "QNRAIN", "QNICE"):
            # A wrfout history frame carries the prognostics but not every
            # time-invariant base-state field (no T_INIT, no ALB); those come
            # from the wrfinput the caller also passes.
            if name in ds.variables:
                out[name] = g(name)
        for name in ("QNCLOUD", "QNWFA", "QNIFA", "QNWFA2D", "QNIFA2D"):
            if name in ds.variables:
                out[name] = g(name)
        out["P_TOP"] = float(np.asarray(ds.variables["P_TOP"][0]))
        out["_have"] = set(ds.variables)
        out["DX"] = float(ds.DX)
        out["DY"] = float(ds.DY)
        out["attrs"] = {k: str(getattr(ds, k)) for k in
                        ("TITLE", "MP_PHYSICS", "DIFF_OPT", "KM_OPT",
                         "DAMP_OPT", "DIFF_6TH_OPT", "PERIODIC_X",
                         "PERIODIC_Y", "DT")
                        if hasattr(ds, k)}
    return out


def build_config(w: dict, mp: int, dt: float, steps: int, ztop: float):
    """The RunConfig matched to the WRF namelist this wrfinput came from."""
    from woof.config import RunConfig, validate_run_config

    nz, ny, nx = w["T"].shape
    return validate_run_config(RunConfig(
        nx=nx, ny=ny, nz=nz, dx=w["DX"], dy=w["DY"], ztop=ztop,
        dt=dt, run_seconds=steps * dt,
        p_surf=1.0e5,
        time_step_sound=6, epssm=0.1, smdiv=0.1, emdiv=0.01,
        khdif=0.0, kvdif=0.0,
        km_opt=4, c_s=0.25,
        diff_6th_opt=2, diff_6th_factor=0.12,
        damp_opt=3, zdamp=5000.0, dampcoef=0.2, w_damping=0,
        h_sca_adv_order=5,
        moist=True, mp_physics=mp, moist_adv_opt=1,
        # WRF's use_theta_m = 1 (Registry default) -- rk_step_prep calls
        # calc_cq on every moist run, so the cq momentum coupling is ON.
        # Every wrf-matched-run config in configs/ sets this.
        moist_cq=True,
        open_x=False, open_y=False,        # periodic, both axes
        hybrid_opt=0, hypsometric_opt=1,
        output_interval_s=steps * dt,
        case="mp28_matched_periodic",
    ))


def build_state(cfg, w: dict):
    """A DomainState whose every prognostic is WRF's ``wrfinput_d01`` value.

    The base state is WRF's own (``MUB``/``PHB``/``PB``/``ALB``/``T_INIT``),
    the eta coordinate is WRF's own ``ZNW``, and the perturbations are read
    straight across.  ``T_INIT`` and ``T`` are WRF's ``theta - t0`` storage
    convention; WOOF-ArWen's ``thb``/``thp`` are absolute base theta and its
    perturbation, so ``t0 = 300`` is added back to the base and the
    perturbation is the difference of the two WRF fields.
    """
    import cupy as cp

    from woof.core.grid import BaseState, make_vertical_coord
    from woof.core.state import DTYPE, init_at_rest

    t0 = 300.0
    znw = w["ZNW"]
    coord = make_vertical_coord(cfg.nz, eta_levels=znw)

    # Flat idealized terrain: every column of the base state is identical, so
    # the 1-D BaseState form applies.  Asserted, not assumed.
    for name in ("MUB", "PHB", "T_INIT", "PB", "ALB"):
        arr = w[name]
        spread = float(np.ptp(arr.reshape(arr.shape[0], -1), axis=-1).max()
                       if arr.ndim == 3 else np.ptp(arr))
        if spread > 1.0e-3 * max(1.0, float(np.abs(arr).max())):
            raise ValueError(f"{name} is not horizontally uniform "
                             f"(spread {spread:g}); this case is flat")

    base = BaseState(
        mub=float(w["MUB"].mean()),
        p_top=w["P_TOP"],
        pb=w["PB"].mean(axis=(1, 2)),
        alb=w["ALB"].mean(axis=(1, 2)),
        thb=w["T_INIT"].mean(axis=(1, 2)) + t0,
        phb=w["PHB"].mean(axis=(1, 2)),
    )
    s = init_at_rest(cfg, coord, base)

    def put(dst, src):
        dst[...] = cp.asarray(np.ascontiguousarray(src), dtype=DTYPE)

    put(s.u, w["U"])
    put(s.v, w["V"])
    put(s.w, w["W"])
    put(s.php, w["PH"])
    put(s.mup, w["MU"])
    put(s.thp, w["T"] + t0 - base.thb[:, None, None])
    for arwen, wrf in (("qv", "QVAPOR"), ("qc", "QCLOUD"), ("qr", "QRAIN"),
                       ("qi", "QICE"), ("qs", "QSNOW"), ("qg", "QGRAUP"),
                       ("nr", "QNRAIN"), ("ni", "QNICE")):
        target = getattr(s, arwen, None)
        if target is not None:
            put(target, w[wrf])
    if cfg.mp_physics == 28:
        for arwen, wrf in (("nc", "QNCLOUD"), ("nwfa", "QNWFA"),
                           ("nifa", "QNIFA")):
            target = getattr(s, arwen, None)
            if target is not None and wrf in w:
                put(target, w[wrf])
    if getattr(s, "qv0", None) is not None:
        s.qv0[...] = s.qv
    return s, coord, base


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wrfinput", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--mp", type=int, required=True, choices=(8, 28))
    ap.add_argument("--dt", type=float, default=12.0)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--frame-steps", type=int, default=50)
    ap.add_argument("--ztop", type=float, default=20000.0)
    ap.add_argument("--restart-from", type=Path, default=None,
                    help="a wrfout to take the PROGNOSTICS from (the base "
                         "state still comes from --wrfinput)")
    ap.add_argument("--restart-frame", type=int, default=0)
    args = ap.parse_args()

    import cupy as cp

    from woof.core import dycore
    from woof.core.diagnostics import update_diagnostics
    from woof.core.physics import initialize_physics

    args.out.mkdir(parents=True, exist_ok=True)
    w = read_wrfinput(args.wrfinput)
    if args.restart_from is not None:
        hist = read_wrfinput(args.restart_from, args.restart_frame)
        base_only = {"MUB", "PHB", "T_INIT", "PB", "ALB", "ZNU", "ZNW",
                     "P_TOP", "DX", "DY", "attrs", "_have"}
        for k, v in hist.items():
            if k not in base_only:
                w[k] = v
    cfg = build_config(w, args.mp, args.dt, args.steps, args.ztop)
    state, coord, base = build_state(cfg, w)
    update_diagnostics(state, cfg.hypsometric_opt)

    driver = initialize_physics(state, cfg)
    receipt = getattr(driver, "microphysics_init_receipt", {}) or {}
    rainnc = state.scratch((cfg.ny, cfg.nx), "mp_rainnc")
    cp.cuda.Stream.null.synchronize()

    names = frame_fields(args.mp)
    series = []
    frames = {}

    def snapshot(step: int) -> None:
        t = step * args.dt
        frames[f"t{int(round(t))}"] = {
            wrf: cp.asnumpy(getattr(state, a)).astype(np.float32)
            for a, wrf in names.items()
        } | {
            # U/V/PH/MU were absent from the first pass's frames;
            # mp28-matched-trajectory.md §11 names that gap as the thing a
            # repeat fixes first.  They are stored on WRF's own staggering,
            # exactly as build_state installed them, so the comparison is
            # elementwise against the wrfout variables of the same names.
            "U": cp.asnumpy(state.u).astype(np.float32),
            "V": cp.asnumpy(state.v).astype(np.float32),
            "W": cp.asnumpy(state.w).astype(np.float32),
            "PH": cp.asnumpy(state.php).astype(np.float32),
            "MU": cp.asnumpy(state.mup).astype(np.float32),
            "T": (cp.asnumpy(state.total_theta()).astype(np.float32)
                  - np.float32(300.0)),
            "RAINNC": cp.asnumpy(rainnc).astype(np.float32),
        }

    def sample(step: int) -> None:
        row = {"step": step, "time_s": step * args.dt,
               "w_max": float(state.w.max()), "w_min": float(state.w.min()),
               "rainnc_sum": float(cp.sum(rainnc.astype(cp.float64))),
               "rainnc_max": float(rainnc.max())}
        for a in ("qc", "qr", "qi", "qs", "qg"):
            f = getattr(state, a, None)
            if f is not None:
                row[f"{a}_max"] = float(f.max())
                row[f"{a}_mean"] = float(cp.mean(f.astype(cp.float64)))
        for a in ("nc", "nr", "ni", "nwfa", "nifa"):
            f = getattr(state, a, None)
            if f is not None:
                row[f"{a}_max"] = float(f.max())
                row[f"{a}_mean"] = float(cp.mean(f.astype(cp.float64)))
        series.append(row)

    snapshot(0)
    sample(0)
    t_start = time.time()
    for n in range(1, args.steps + 1):
        dycore.step(state, cfg)
        if n % 10 == 0 or n == args.steps:
            sample(n)
        if n % args.frame_steps == 0:
            snapshot(n)
    wall = time.time() - t_start

    for key, fields in frames.items():
        np.savez_compressed(args.out / f"frame_{key}.npz", **fields)
    (args.out / "series.json").write_text(json.dumps(series, indent=1))
    (args.out / "run.json").write_text(json.dumps({
        "model": "woof", "mp_physics": args.mp,
        "nx": cfg.nx, "ny": cfg.ny, "nz": cfg.nz,
        "dx": cfg.dx, "dt": args.dt, "steps": args.steps,
        "frame_steps": args.frame_steps,
        "wrfinput": str(args.wrfinput),
        "wrfinput_attrs": w["attrs"],
        "p_top": base.p_top, "mub": float(base.mub),
        "microphysics_init_receipt": {k: (v if isinstance(
            v, (int, float, str, bool, type(None))) else str(v))
            for k, v in receipt.items()},
        "wall_seconds": wall,
        "cupy": cp.__version__,
        "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
    }, indent=1))
    print(f"woof mp={args.mp}: {args.steps} steps in {wall:.1f}s -> "
          f"{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
