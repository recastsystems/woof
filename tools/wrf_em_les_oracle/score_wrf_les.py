#!/usr/bin/env python3
"""Independent scorer for WRF em_les output (WP-L7 oracle side).

No woof import, by construction: this is the oracle-side instrument and it
must be able to disagree with the engine it is scoring.

Metric definitions are transcribed from the WOOF-ArWen CBL case so the comparison is
like-for-like; where WOOF-ArWen makes a convention choice that is not forced by the
physics (unstaggering by truncation vs averaging) BOTH are computed and
reported, so a definitional difference can never be mistaken for a physics one.

Usage:  score_wrf_les.py <run_dir> <out_prefix> [--window-min N]
"""
import sys
import os
import json
import glob
import math
import hashlib

import numpy as np
from netCDF4 import Dataset

G = 9.81
T0 = 300.0          # WRF base potential temperature (t0)
RD = 287.0
CP = 1004.5


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def slab_mean(a):
    """Horizontal slab mean of a (nz, ny, nx) array, in float64."""
    return a.astype(np.float64).mean(axis=(1, 2))


def unstagger_z(w):
    """(nz+1, ny, nx) -> (nz, ny, nx): average w to mass levels."""
    return 0.5 * (w[:-1] + w[1:])


def unstagger_x_avg(u):
    """(nz, ny, nx+1) -> (nz, ny, nx) by averaging (physically correct)."""
    return 0.5 * (u[:, :, :-1] + u[:, :, 1:])


def unstagger_x_trunc(u, nx):
    """(nz, ny, nx+1) -> (nz, ny, nx) by truncation (WOOF-ArWen convention)."""
    return u[:, :, :nx]


def unstagger_y_avg(v):
    return 0.5 * (v[:, :-1, :] + v[:, 1:, :])


def unstagger_y_trunc(v, ny):
    return v[:, :ny, :]


def frame_profiles(nc, it, nz, ny, nx, has_tke):
    """All slab profiles for one history frame."""
    T = nc.variables["T"][it].astype(np.float64) + T0           # (nz,ny,nx) theta
    W = nc.variables["W"][it].astype(np.float64)                # (nz+1,ny,nx)
    U = nc.variables["U"][it].astype(np.float64)                # (nz,ny,nx+1)
    V = nc.variables["V"][it].astype(np.float64)                # (nz,ny+1,nx)
    PH = nc.variables["PH"][it].astype(np.float64)
    PHB = nc.variables["PHB"][it].astype(np.float64)

    zw = slab_mean((PH + PHB) / G)                              # (nz+1,) w-level heights
    zm = 0.5 * (zw[:-1] + zw[1:])                               # (nz,) mass-level heights

    wm = unstagger_z(W)
    ua = unstagger_x_avg(U)
    va = unstagger_y_avg(V)
    ut = unstagger_x_trunc(U, nx)
    vt = unstagger_y_trunc(V, ny)

    th_bar = slab_mean(T)
    thp = T - th_bar[:, None, None]
    wp = wm - slab_mean(wm)[:, None, None]

    out = {}
    out["z_mass"] = zm
    out["z_w"] = zw
    out["theta"] = th_bar
    out["wth_res"] = slab_mean(wp * thp)
    # resolved TKE, both unstagger conventions
    for tag, uu, vv in (("avg", ua, va), ("trunc", ut, vt)):
        up = uu - slab_mean(uu)[:, None, None]
        vp = vv - slab_mean(vv)[:, None, None]
        out["tke_res_" + tag] = 0.5 * (slab_mean(up * up) + slab_mean(vp * vp)
                                       + slab_mean(wp * wp))
    out["u_mean"] = slab_mean(ua)
    out["v_mean"] = slab_mean(va)
    out["w_var"] = slab_mean(wp * wp)
    out["w_absmax"] = float(np.abs(W).max())

    # SGS heat flux on interior w-levels, ArWen's fnm/fnp construction
    wth_sgs = np.zeros(nz + 1)
    if "XKHV" in nc.variables:
        khv = nc.variables["XKHV"][it].astype(np.float64)
        fnm = nc.variables["FNM"][it].astype(np.float64)
        fnp = nc.variables["FNP"][it].astype(np.float64)
        for kw in range(1, nz):
            rdz = 1.0 / (zm[kw] - zm[kw - 1])
            kw_c = fnm[kw] * khv[kw] + fnp[kw] * khv[kw - 1]
            wth_sgs[kw] = np.mean(-kw_c * (T[kw] - T[kw - 1]) * rdz)
    out["wth_sgs"] = wth_sgs

    if has_tke and "TKE" in nc.variables:
        out["e_sgs"] = slab_mean(nc.variables["TKE"][it].astype(np.float64))
    else:
        out["e_sgs"] = np.zeros(nz)

    # dry mass for the conservation check
    MU = nc.variables["MU"][it].astype(np.float64)
    MUB = nc.variables["MUB"][it].astype(np.float64)
    out["mass"] = float((MU + MUB).sum())
    return out


def radial_spectrum(f2d, dx):
    """Radial E(k) of a 2-D field with an explicit Parseval check.

    Returns (k_cyc_per_m, E, parseval_residual_rel).  No windowing: the domain
    is doubly periodic, so the field is already periodic and a window would only
    bias the variance.  Normalisation is 1/N^2 on the transform, which makes
    sum(E) equal the field variance exactly (Parseval).
    """
    f = f2d.astype(np.float64)
    f = f - f.mean()
    n = f.shape[0]
    assert f.shape[0] == f.shape[1]
    coef = np.fft.fft2(f) / (n * n)
    p = np.abs(coef) ** 2
    kx = np.fft.fftfreq(n, d=1.0)
    KX, KY = np.meshgrid(kx, kx, indexing="xy")
    kmag = np.sqrt(KX ** 2 + KY ** 2) * n            # in index units
    kbin = np.rint(kmag).astype(int)
    nb = n // 2
    E = np.zeros(nb + 1)
    for b in range(nb + 1):
        E[b] = p[kbin == b].sum()
    var = float(f.var())
    tot = float(p.sum()) - float(p[0, 0])
    parseval_rel = abs(tot - var) / max(var, 1e-300)
    # keep only bins that were actually summed; bins beyond nb hold the corners
    k_cyc = np.arange(nb + 1) / (n * dx)
    return k_cyc, E, parseval_rel, var, float(E[1:].sum())


def log_log_slope(k, E, lo, hi):
    """Least-squares slope of log E vs log k over bin window [lo, hi)."""
    kk = k[lo:hi]
    EE = E[lo:hi]
    m = (kk > 0) & (EE > 0)
    if m.sum() < 3:
        return float("nan")
    A = np.polyfit(np.log(kk[m]), np.log(EE[m]), 1)
    return float(A[0])


def integral_time_scale(series, dt_s):
    """Integral time scale: integrate the autocorrelation to its first zero."""
    x = np.asarray(series, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    if n < 4 or x.std() == 0:
        return float("nan")
    ac = np.correlate(x, x, mode="full")[n - 1:]
    ac = ac / ac[0]
    tint = 0.0
    for i in range(1, len(ac)):
        if ac[i] <= 0.0:
            break
        tint += 0.5 * (ac[i - 1] + ac[i]) * dt_s
    return float(tint)


def main():
    run_dir = sys.argv[1]
    out_prefix = sys.argv[2]
    window_min = 30.0
    if "--window-min" in sys.argv:
        window_min = float(sys.argv[sys.argv.index("--window-min") + 1])

    files = sorted(glob.glob(os.path.join(run_dir, "wrfout_d01_*")))
    if not files:
        print("no wrfout in %s" % run_dir)
        return 2

    # namelist knobs we need
    nml = open(os.path.join(run_dir, "namelist.input")).read()

    def nml_get(key, cast=float, default=None):
        for line in nml.splitlines():
            s = line.strip()
            if s.startswith(key):
                after = s.split("=", 1)
                if len(after) == 2:
                    return cast(after[1].strip().rstrip(",").strip())
        return default

    Qs = nml_get("tke_heat_flux", float, 0.24)
    km_opt = nml_get("km_opt", int, 3)
    dx = nml_get("dx", float, 100.0)

    nc = Dataset(files[0])
    times = []
    for f in files:
        d = Dataset(f)
        times.append(d.variables["XTIME"][:].astype(np.float64))
        d.close()
    nc.close()

    # accumulate over all frames, keep the last window
    prof_t = []
    tmin = []
    mass = []
    wmax_run = 0.0
    nz = ny = nx = None
    for f in files:
        d = Dataset(f)
        xt = d.variables["XTIME"][:].astype(np.float64)     # minutes
        if nz is None:
            nz = d.dimensions["bottom_top"].size
            ny = d.dimensions["south_north"].size
            nx = d.dimensions["west_east"].size
        for it in range(len(xt)):
            p = frame_profiles(d, it, nz, ny, nx, km_opt == 2)
            wmax_run = max(wmax_run, p["w_absmax"])
            prof_t.append(p)
            tmin.append(float(xt[it]))
            mass.append(p["mass"])
        d.close()

    tmin = np.array(tmin)
    t_end = tmin.max()
    in_win = tmin >= (t_end - window_min - 1e-9)
    idx = np.where(in_win)[0]
    nwin = len(idx)

    def avg(key):
        return np.mean([prof_t[i][key] for i in idx], axis=0)

    zm = avg("z_mass")
    zw = avg("z_w")
    theta = avg("theta")
    wth_res = avg("wth_res")
    wth_sgs = avg("wth_sgs")
    e_sgs = avg("e_sgs")
    w_var = avg("w_var")
    tke_avg = avg("tke_res_avg")
    tke_trunc = avg("tke_res_trunc")

    total_flux = wth_res + 0.5 * (wth_sgs[:-1] + wth_sgs[1:])
    kzi = int(np.argmin(total_flux))
    zi = float(zm[min(kzi, len(zm) - 1)])

    # mixed-layer theta over 0.2-0.7 zi
    band = (zm >= 0.2 * zi) & (zm <= 0.7 * zi)
    theta_ml = float(theta[band].mean()) if band.any() else float(theta[1])
    wstar = float((G / theta_ml * Qs * zi) ** (1.0 / 3.0))
    tstar = zi / wstar

    # linear fit of total flux over 0.1-0.7 zi
    fitband = (zm >= 0.1 * zi) & (zm <= 0.7 * zi)
    x = zm[fitband] / zi
    y = total_flux[fitband] / Qs
    if len(x) >= 2:
        b, a = np.polyfit(x, y, 1)
        rms = float(np.sqrt(np.mean((y - (a + b * x)) ** 2)))
    else:
        a = b = rms = float("nan")

    # resolved TKE fraction, ArWen's index band 1:int(0.7*nz)
    hi = max(2, int(0.7 * nz))

    def res_frac(tke):
        # km_opt=3 is diagnostic Smagorinsky: there is no prognostic SGS TKE,
        # so the denominator of this ratio does not exist.  Returning 1.0 here
        # would claim "100% resolved" from an absent denominator, which is a
        # measurement artifact and not a result.  Report it as absent instead,
        # exactly as the ArWen case module does.
        if km_opt != 2:
            return None
        num = float(tke[1:hi].mean())
        den = num + float(np.asarray(e_sgs)[1:hi].mean())
        return num / den if den > 0 else float("nan")
    rf_avg = res_frac(tke_avg)
    rf_trunc = res_frac(tke_trunc)

    # updraft fraction near 0.3 zi, from the final frame
    klow = int(np.argmin(np.abs(zm - 0.3 * max(zi, zm[1]))))
    dlast = Dataset(files[-1])
    itl = dlast.variables["XTIME"].shape[0] - 1
    Wl = dlast.variables["W"][itl].astype(np.float64)
    wml = 0.5 * (Wl[klow] + Wl[klow + 1])
    updraft = float((wml > 0.0).mean())

    # spectra at 0.5 zi from the final frame, with Parseval
    kmid = int(np.argmin(np.abs(zm - 0.5 * zi)))
    Wm = 0.5 * (Wl[kmid] + Wl[kmid + 1])
    kcyc, Ek, pars_rel, fvar, ebinsum = radial_spectrum(Wm, dx)
    nbin = len(Ek)
    slope = log_log_slope(kcyc, Ek, max(1, nbin // 8), max(3, nbin // 2))
    dlast.close()

    # integral time scale from the mid-CBL w-variance series
    wvar_series = [prof_t[i]["w_var"][kmid] for i in range(len(prof_t))]
    dt_out = float(np.median(np.diff(tmin)) * 60.0) if len(tmin) > 1 else 60.0
    tint = integral_time_scale(wvar_series[-max(nwin, 3):], dt_out)
    window_s = window_min * 60.0
    n_t = window_s / (2.0 * tint) if tint and tint == tint and tint > 0 else float("nan")

    Lx = nx * dx
    n_h = (Lx * Lx) / (zi * zi)

    m0 = mass[0]
    mass_drift = float(max(abs(m - m0) for m in mass) / abs(m0))

    # Final-sample counterparts. ArWen's committed wth_res_max / tke_res_max
    # are taken from the LAST sample, not from a window average, so the
    # window-averaged numbers above are not the like-for-like partner for
    # those particular receipt entries. Both are reported.
    lastp = prof_t[-1]
    last_total = lastp["wth_res"] + 0.5 * (lastp["wth_sgs"][:-1]
                                           + lastp["wth_sgs"][1:])
    zi_final = float(lastp["z_mass"][int(np.argmin(last_total))])
    final_sample = dict(
        zi_m=zi_final,
        wth_res_max_over_qs=float(lastp["wth_res"].max() / Qs),
        wth_total_min_over_qs=float(last_total.min() / Qs),
        tke_res_max_trunc=float(lastp["tke_res_trunc"].max()),
        tke_res_max_avg=float(lastp["tke_res_avg"].max()),
        e_sgs_volume=(float(np.asarray(lastp["e_sgs"]).mean())
                      if km_opt == 2 else None),
    )
    # ArWen's committed resolved_fraction_ml is computed from the FINAL
    # sample over index band 1:int(0.7*nz), not from a window average
    # (convective_boundary_layer.py, "resolved_fraction_ml"). This is its
    # exact partner.
    if km_opt == 2:
        for tag in ("trunc", "avg"):
            num = float(lastp["tke_res_" + tag][1:hi].mean())
            den = num + float(np.asarray(lastp["e_sgs"])[1:hi].mean())
            final_sample["resolved_fraction_ml_" + tag] = (
                num / den if den > 0 else float("nan"))
    else:
        final_sample["resolved_fraction_ml_trunc"] = None
        final_sample["resolved_fraction_ml_avg"] = None

    res = dict(
        run_dir=os.path.abspath(run_dir),
        wrfout_files=[os.path.basename(f) for f in files],
        wrfout_sha256={os.path.basename(f): sha256_file(f) for f in files},
        wrfout_bytes={os.path.basename(f): os.path.getsize(f) for f in files},
        km_opt=km_opt, dx_m=dx, nx=nx, ny=ny, nz=nz,
        surface_heat_flux=Qs,
        n_frames=len(prof_t), n_frames_in_window=nwin,
        window_minutes=window_min, t_end_minutes=float(t_end),
        history_dt_s=dt_out,
        # headline statistics, ArWen's definitions
        zi_m=zi, theta_ml=theta_ml, wstar=wstar, tstar_s=tstar,
        w_absmax_frames=wmax_run, wmax_over_wstar=wmax_run / wstar,
        flux_fit_intercept=float(a), flux_fit_slope=float(b), flux_fit_rms=rms,
        entrainment_min_over_qs=float(total_flux.min() / Qs),
        wth_res_max_over_qs=float(wth_res.max() / Qs),
        resolved_fraction_ml_avg=rf_avg,
        resolved_fraction_ml_trunc=rf_trunc,
        tke_res_max_avg=float(tke_avg.max()),
        tke_res_max_trunc=float(tke_trunc.max()),
        e_sgs_ml_mean=(float(np.asarray(e_sgs)[1:hi].mean()) if km_opt == 2 else None),
        updraft_fraction_low=updraft,
        mass_drift_rel=mass_drift,
        final_sample=final_sample,
        # section 2.4 power receipt
        power=dict(
            tstar_s=tstar,
            window_over_tstar=window_s / tstar,
            spinup_over_tstar=(t_end * 60.0 - window_s) / tstar,
            T_int_s=tint,
            N_t=n_t,
            L_x_m=Lx, L_x_over_zi=Lx / zi,
            l_c_assumed_m=zi, N_h=n_h,
        ),
        spectrum=dict(
            level_index=kmid, level_z_m=float(zm[kmid]),
            parseval_residual_rel=pars_rel,
            field_variance=fvar, binned_sum=ebinsum,
            slope_Ek=slope,
            fit_bins=[max(1, nbin // 8), max(3, nbin // 2)],
            n_bins=nbin,
        ),
    )

    with open(out_prefix + ".json", "w") as fh:
        json.dump(res, fh, indent=2, sort_keys=True)
    np.savez_compressed(
        out_prefix + "_profiles.npz",
        z_mass=zm, z_w=zw, theta=theta, wth_res=wth_res, wth_sgs=wth_sgs,
        total_flux=total_flux, e_sgs=e_sgs, w_var=w_var,
        tke_res_avg=tke_avg, tke_res_trunc=tke_trunc,
        t_minutes=tmin, mass=np.array(mass),
        spec_k=kcyc, spec_E=Ek,
        wvar_series=np.array(wvar_series),
    )
    for k in ("zi_m", "wstar", "wmax_over_wstar", "flux_fit_intercept",
              "flux_fit_slope", "flux_fit_rms", "entrainment_min_over_qs",
              "wth_res_max_over_qs", "resolved_fraction_ml_avg",
              "resolved_fraction_ml_trunc", "mass_drift_rel"):
        print("%-28s %s" % (k, res[k]))
    print("%-28s %s" % ("parseval_residual_rel", pars_rel))
    return 0


if __name__ == "__main__":
    sys.exit(main())
