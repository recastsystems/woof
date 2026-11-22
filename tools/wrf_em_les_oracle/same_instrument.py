#!/usr/bin/env python3
"""Score BOTH sides with the same code, from raw profile arrays.

The headline-number comparison (my WRF numbers vs WOOF-ArWen's published numbers)
carries a confound: each side's numbers were produced by its own code with its
own conventions. WOOF-ArWen commits per-minute slab profiles alongside its receipts,
and this lane's scorer writes the same arrays, so both can be reduced by one
identical routine. Any remaining difference is then the models, not the
bookkeeping.

Usage: same_instrument.py <arwen.npz> <wrf.npz> <label> [--window-min 30]
"""
import sys
import json

import numpy as np

G = 9.81
QS = 0.24


def reduce_profiles(z_mass, wth_res_t, wth_sgs_t, tke_res_t, theta_t,
                    e_sgs_t, t_seconds, window_s):
    """One reduction, applied identically to either model's arrays."""
    t = np.asarray(t_seconds, dtype=np.float64)
    sel = t >= (t.max() - window_s - 1e-9)
    n = int(sel.sum())

    wth_res = np.asarray(wth_res_t)[sel].mean(axis=0)
    wth_sgs = np.asarray(wth_sgs_t)[sel].mean(axis=0)
    tke_res = np.asarray(tke_res_t)[sel].mean(axis=0)
    theta = np.asarray(theta_t)[sel].mean(axis=0)
    e_sgs = (np.asarray(e_sgs_t)[sel].mean(axis=0)
             if e_sgs_t is not None else None)

    total = wth_res + 0.5 * (wth_sgs[:-1] + wth_sgs[1:])
    kzi = int(np.argmin(total))
    zi = float(z_mass[kzi])

    band = (z_mass >= 0.2 * zi) & (z_mass <= 0.7 * zi)
    theta_ml = float(theta[band].mean())
    wstar = float((G / theta_ml * QS * zi) ** (1.0 / 3.0))

    fb = (z_mass >= 0.1 * zi) & (z_mass <= 0.7 * zi)
    x = z_mass[fb] / zi
    y = total[fb] / QS
    b, a = np.polyfit(x, y, 1)
    rms = float(np.sqrt(np.mean((y - (a + b * x)) ** 2)))

    nz = len(z_mass)
    hi = max(2, int(0.7 * nz))
    if e_sgs is not None:
        num = float(tke_res[1:hi].mean())
        den = num + float(e_sgs[1:hi].mean())
        rf = num / den if den > 0 else float("nan")
        esgs_ml = float(e_sgs[1:hi].mean())
    else:
        rf = None
        esgs_ml = None

    return dict(
        n_samples=n, zi_m=zi, theta_ml=theta_ml, wstar=wstar,
        tstar_s=zi / wstar,
        flux_fit_intercept=float(a), flux_fit_slope=float(b),
        flux_fit_rms=rms,
        entrainment_min_over_qs=float(total.min() / QS),
        wth_res_max_over_qs=float(wth_res.max() / QS),
        resolved_fraction_ml=rf,
        e_sgs_ml_mean=esgs_ml,
        tke_res_max=float(tke_res.max()),
        surface_resolved_frac=float(wth_res[0] / QS),
        surface_sgs_frac=float(wth_sgs[1] / QS),
        _profiles=dict(z=z_mass, total=total, res=wth_res, sgs=wth_sgs,
                       theta=theta, tke=tke_res),
    )


def _sgs_carrier_or_none(a):
    """An all-zero e_sgs array is an ABSENT carrier, not a zero one.

    km_opt=3 has no prognostic SGS TKE. Some receipts still ship an e_sgs key
    filled with zeros; taking it at face value makes the resolved-fraction
    ratio evaluate to exactly 1.0 -- "100% resolved" produced by dividing by
    its own numerator. This is the same artifact the WRF-side scorer guards
    against, and it has to be guarded on both sides or the guard is decorative.
    """
    if a is None:
        return None
    arr = np.asarray(a)
    return None if float(np.abs(arr).max()) == 0.0 else arr


def load_arwen(path, window_s):
    d = np.load(path)
    e = _sgs_carrier_or_none(d["e_sgs"] if "e_sgs" in d.files else None)
    return reduce_profiles(d["z_mass"], d["wth_res"], d["wth_sgs"],
                           d["tke_res"], d["theta"], e, d["t_seconds"],
                           window_s), d


def load_wrf(path, window_s):
    """This lane's npz already stores window-averaged profiles, so the
    reduction is applied to a single 'sample' -- the averaging was done with
    the identical window upstream."""
    d = np.load(path)
    one = lambda a: np.asarray(a)[None, :]
    e = _sgs_carrier_or_none(d["e_sgs"])
    if e is not None:
        e = one(e)
    return reduce_profiles(d["z_mass"], one(d["wth_res"]), one(d["wth_sgs"]),
                           one(d["tke_res_trunc"]), one(d["theta"]), e,
                           np.array([0.0]), window_s), d


def main():
    ap, wp, label = sys.argv[1], sys.argv[2], sys.argv[3]
    win = 30.0
    if "--window-min" in sys.argv:
        win = float(sys.argv[sys.argv.index("--window-min") + 1])
    ws = win * 60.0

    A, ad = load_arwen(ap, ws)
    W, wd = load_wrf(wp, ws)

    keys = ["zi_m", "theta_ml", "wstar", "tstar_s", "flux_fit_intercept",
            "flux_fit_slope", "flux_fit_rms", "entrainment_min_over_qs",
            "wth_res_max_over_qs", "resolved_fraction_ml", "e_sgs_ml_mean",
            "tke_res_max", "surface_resolved_frac", "surface_sgs_frac"]
    print("=== %s : both sides reduced by the SAME routine, %.0f-min window ==="
          % (label, win))
    print("WOOF-ArWen samples in window: %d" % A["n_samples"])
    print("%-26s %14s %14s %12s" % ("metric", "WOOF-ArWen", "WRF em_les", "rel diff"))
    rows = {}
    for k in keys:
        a, w = A[k], W[k]
        if a is None or w is None:
            print("%-26s %14s %14s %12s"
                  % (k, "n/a" if a is None else "%.5g" % a,
                     "n/a" if w is None else "%.5g" % w, "-"))
            rows[k] = dict(arwen=a, wrf=w, rel=None)
            continue
        rel = abs(w - a) / max(abs(a), 1e-300)
        print("%-26s %14.6g %14.6g %11.2f%%" % (k, a, w, 100 * rel))
        rows[k] = dict(arwen=float(a), wrf=float(w), rel=float(rel))

    out = dict(label=label, window_min=win,
               arwen_samples=A["n_samples"], metrics=rows)
    json.dump(out, open("same_instrument_%s.json" % label, "w"),
              indent=2, sort_keys=True)

    np.savez_compressed(
        "same_instrument_%s_profiles.npz" % label,
        az=A["_profiles"]["z"], atotal=A["_profiles"]["total"],
        ares=A["_profiles"]["res"], asgs=A["_profiles"]["sgs"],
        atheta=A["_profiles"]["theta"], atke=A["_profiles"]["tke"],
        wz=W["_profiles"]["z"], wtotal=W["_profiles"]["total"],
        wres=W["_profiles"]["res"], wsgs=W["_profiles"]["sgs"],
        wtheta=W["_profiles"]["theta"], wtke=W["_profiles"]["tke"],
        azi=A["zi_m"], wzi=W["zi_m"])


if __name__ == "__main__":
    main()
