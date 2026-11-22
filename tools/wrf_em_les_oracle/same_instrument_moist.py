#!/usr/bin/env python3
"""Reduce BOTH sides' moist profiles with the same code (P1).

The dry lane's `same_instrument.py` exists because a headline-number
comparison carries a confound: each side's numbers were produced by its own
code with its own conventions.  The moist metrics are *new* instrument
definitions, so the confound would be worse, not better, if each side reduced
its own arrays.  One routine, applied to both.

Usage: same_instrument_moist.py <arwen_moist.npz> <wrf_moist.npz> <label>
                                [--window-min 30] [--outdir .]

------------------------------------------------------------------------
THE npz CONTRACT (this is what the ArWen-side P1 case module must write)
------------------------------------------------------------------------
Required, float64, per-frame stacks unless noted:

  z_mass         (nz,)      mass-level heights, metres
  t_seconds      (nt,)      frame times, seconds since run start
  wthv_res       (nt, nz)   resolved   <w' theta_v'>, K m/s, mass levels
  wthv_sgs       (nt, nz+1) subgrid    <w' theta_v'>, K m/s, w-levels
  wqv_res        (nt, nz)   resolved   <w' qv'>, (kg/kg) m/s, mass levels
  wqv_sgs        (nt, nz+1) subgrid    <w' qv'>, (kg/kg) m/s, w-levels
  qv, qc, qr     (nt, nz)   slab-mean mixing ratios, kg/kg
  theta, thetav  (nt, nz)   slab-mean potential / virtual pot. temperature, K
  cloud_frac     (nt, nz)   fraction of points with qc >= 1e-5
  sat_frac       (nt, nz)   fraction of points with qv >= qvs
  n2_moist_frac  (nt, nz)   fraction on WRF's saturated-BN2 arm:
                            qv >= qvs .OR. qc >= 1e-5
  lwp            (nt,)      slab-mean liquid-water path, kg/m2

Optional:
  e_sgs          (nt, nz)   prognostic SGS TKE (km_opt=2 only; an all-zero
                            array is read as an ABSENT carrier, not a zero
                            one -- same guard as the dry reducer)
  rwp, rainnc    (nt,)      rain-water path, accumulated surface rain (mm)

theta_v convention: theta*(1 + 0.60829*qv - qc - qr - qi), condensate loading
included.  Both sides must use it; the WRF scorer also emits the vapour-only
variant under `wthv_res_novload` / `wthv_sgs_novload` so a convention
disagreement can be measured rather than argued.

------------------------------------------------------------------------
READ THIS BEFORE COMPARING z_i: ONE NAME, TWO HEIGHTS
------------------------------------------------------------------------
`zi_thetav_m` here -- and `zi_thetav_load_m` in the WRF receipts -- is the
height of the MINIMUM of the total buoyancy flux.  In a **clear** CBL that is
the inversion.  In a **cloud-topped** CBL it is **cloud base**, because the
buoyancy flux reverses there.  This is not a modelling choice and not a bug;
it is what the definition does, and it was measured on two runs of the same
capped family that differ only in their vapour column:

    capped-DRY   anchor : zi = 1526.3 m   (the inversion, base at 1500 m)
    capped-MOIST r1     : zi = 1274.4514 m
                          cloud_base_m = 1274.4514 m   <- the same number,
                                                          to every digit

So a moist-vs-dry z_i difference is NOT a change in boundary-layer depth, and
comparing a cloudy WOOF-ArWen z_i against a clear WRF z_i (or the reverse) compares
two different heights while appearing to compare one quantity.  Both sides
must reduce it identically, and any receipt quoting z_i for a cloudy case must
say which of the two it means.  The inversion height in a cloudy case is not
recoverable from this metric; use `cloud_top_m` or the theta profile.

Nothing here cuts a band.  It reports both sides and their relative
difference; the bands are ratified by the owner from committed receipts.
"""
import sys
import os
import json

import numpy as np

QC_CR = 1.0e-5


def reduce_moist(z_mass, t_seconds, wthv_res_t, wthv_sgs_t, wqv_res_t,
                 wqv_sgs_t, qv_t, qc_t, qr_t, cloud_t, sat_t, n2_t, lwp_t,
                 window_s):
    """One reduction, applied identically to either model's arrays."""
    t = np.asarray(t_seconds, dtype=np.float64)
    sel = t >= (t.max() - window_s - 1e-9)
    n = int(sel.sum())

    def w(a):
        return np.asarray(a, dtype=np.float64)[sel].mean(axis=0)

    wthv_res = w(wthv_res_t)
    wthv_sgs = w(wthv_sgs_t)
    wqv_res = w(wqv_res_t)
    wqv_sgs = w(wqv_sgs_t)
    qv = w(qv_t)
    qc = w(qc_t)
    qr = w(qr_t)
    cloud = w(cloud_t)
    sat = w(sat_t)
    n2 = w(n2_t)
    lwp = float(np.asarray(lwp_t, dtype=np.float64)[sel].mean())

    total_thv = wthv_res + 0.5 * (wthv_sgs[:-1] + wthv_sgs[1:])
    total_qv = wqv_res + 0.5 * (wqv_sgs[:-1] + wqv_sgs[1:])
    zi = float(z_mass[int(np.argmin(total_thv))])

    def edges(profile, thresh):
        hit = np.where(profile >= thresh)[0]
        if not len(hit):
            return None, None
        return float(z_mass[hit[0]]), float(z_mass[hit[-1]])

    cb, ct = edges(cloud, 0.01)

    # Resolved share of the buoyancy flux over the mixed layer, the moist
    # counterpart of the dry lane's resolved-fraction measure.  Index band
    # 0.1-0.7 z_i in height, not index, because the two sides may not share
    # a level count.
    band = (z_mass >= 0.1 * zi) & (z_mass <= 0.7 * zi)
    sgs_m = 0.5 * (wthv_sgs[:-1] + wthv_sgs[1:])
    num = float(np.abs(wthv_res[band]).mean())
    den = num + float(np.abs(sgs_m[band]).mean())
    res_frac_thv = num / den if den > 0 else float("nan")

    sgs_q = 0.5 * (wqv_sgs[:-1] + wqv_sgs[1:])
    numq = float(np.abs(wqv_res[band]).mean())
    denq = numq + float(np.abs(sgs_q[band]).mean())
    res_frac_qv = numq / denq if denq > 0 else float("nan")

    return dict(
        n_samples=n,
        zi_thetav_m=zi,
        wthv_res_max=float(wthv_res.max()),
        wthv_total_min=float(total_thv.min()),
        wqv_res_max=float(wqv_res.max()),
        wqv_total_max=float(np.abs(total_qv).max()),
        resolved_fraction_wthv=res_frac_thv,
        resolved_fraction_wqv=res_frac_qv,
        qv_surface=float(qv[0]),
        qc_profile_max=float(qc.max()),
        qr_profile_max=float(qr.max()),
        cloud_fraction_max=float(cloud.max()),
        cloud_base_m=cb, cloud_top_m=ct,
        sat_fraction_max=float(sat.max()),
        n2_moist_fraction_max=float(n2.max()),
        n2_engaged_somewhere=bool(float(n2.max()) > 0.0),
        n2_engaged_everywhere=bool(float(n2.min()) >= 1.0),
        lwp_kg_m2=lwp,
        _profiles=dict(z=z_mass, thv_res=wthv_res, thv_sgs=wthv_sgs,
                       thv_total=total_thv, qv_res=wqv_res, qv_sgs=wqv_sgs,
                       qc=qc, cloud=cloud, n2=n2),
    )


REQUIRED = ("z_mass", "t_seconds", "wthv_res", "wthv_sgs", "wqv_res",
            "wqv_sgs", "qv", "qc", "qr", "cloud_frac", "sat_frac",
            "n2_moist_frac", "lwp")


def load(path, window_s):
    d = np.load(path)
    missing = [k for k in REQUIRED if k not in d.files]
    if missing:
        raise SystemExit(
            "%s is missing required moist arrays %s.\n"
            "The contract is documented at the top of this file; a side that "
            "cannot supply them has not implemented the moist instrument, and "
            "filling them with zeros would be worse than failing here."
            % (path, missing))
    return reduce_moist(d["z_mass"], d["t_seconds"], d["wthv_res"],
                        d["wthv_sgs"], d["wqv_res"], d["wqv_sgs"], d["qv"],
                        d["qc"], d["qr"], d["cloud_frac"], d["sat_frac"],
                        d["n2_moist_frac"], d["lwp"], window_s), d


def main():
    ap, wp, label = sys.argv[1], sys.argv[2], sys.argv[3]
    win = 30.0
    if "--window-min" in sys.argv:
        win = float(sys.argv[sys.argv.index("--window-min") + 1])
    outdir = "."
    if "--outdir" in sys.argv:
        outdir = sys.argv[sys.argv.index("--outdir") + 1]
    ws = win * 60.0

    A, _ = load(ap, ws)
    W, _ = load(wp, ws)

    keys = ["zi_thetav_m", "wthv_res_max", "wthv_total_min", "wqv_res_max",
            "wqv_total_max", "resolved_fraction_wthv",
            "resolved_fraction_wqv", "qv_surface", "qc_profile_max",
            "qr_profile_max", "cloud_fraction_max", "cloud_base_m",
            "cloud_top_m", "sat_fraction_max", "n2_moist_fraction_max",
            "lwp_kg_m2"]
    print("=== %s : both sides reduced by the SAME routine, %.0f-min window ==="
          % (label, win))
    print("WOOF-ArWen frames in window: %d   WRF frames in window: %d"
          % (A["n_samples"], W["n_samples"]))
    print("%-28s %14s %14s %12s" % ("metric", "WOOF-ArWen", "WRF em_les", "rel diff"))
    rows = {}
    for k in keys:
        a, w = A[k], W[k]
        if a is None or w is None:
            print("%-28s %14s %14s %12s"
                  % (k, "n/a" if a is None else "%.5g" % a,
                     "n/a" if w is None else "%.5g" % w, "-"))
            rows[k] = dict(arwen=a, wrf=w, rel=None)
            continue
        rel = abs(w - a) / max(abs(a), 1e-300)
        print("%-28s %14.6g %14.6g %11.2f%%" % (k, a, w, 100 * rel))
        rows[k] = dict(arwen=float(a), wrf=float(w), rel=float(rel))

    for k in ("n2_engaged_somewhere", "n2_engaged_everywhere"):
        print("%-28s %14s %14s" % (k, A[k], W[k]))
        rows[k] = dict(arwen=bool(A[k]), wrf=bool(W[k]), rel=None)

    out = dict(label=label, window_min=win,
               arwen_frames=A["n_samples"], wrf_frames=W["n_samples"],
               arwen_npz=os.path.abspath(ap), wrf_npz=os.path.abspath(wp),
               metrics=rows)
    with open(os.path.join(outdir, "same_instrument_moist_%s.json" % label),
              "w", newline="\n") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)

    np.savez_compressed(
        os.path.join(outdir, "same_instrument_moist_%s_profiles.npz" % label),
        **{("a_" + k): v for k, v in A["_profiles"].items()},
        **{("w_" + k): v for k, v in W["_profiles"].items()})


if __name__ == "__main__":
    main()
