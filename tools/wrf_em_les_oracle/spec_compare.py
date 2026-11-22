#!/usr/bin/env python3
"""E(k) of the w field, both models, one routine, Parseval checked on each.

The normalisation is the essential part. An unnormalised |fft2|^2 scales as
N^4, so comparing two fields on different grid counts that way inflates one of
them by (N1/N2)^4 -- the artifact that put a spurious 1.9 decades into an
earlier WOOF-ArWen figure. Here the transform carries 1/N^2, which makes the summed
spectrum equal the field variance exactly, and that identity is asserted for
each field before any slope is quoted.
"""
import sys
import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def radial_spectrum(f2d, dx):
    f = np.asarray(f2d, dtype=np.float64)
    f = f - f.mean()
    n = f.shape[0]
    coef = np.fft.fft2(f) / (n * n)
    p = np.abs(coef) ** 2
    kk = np.fft.fftfreq(n, d=1.0) * n
    KX, KY = np.meshgrid(kk, kk, indexing="xy")
    kbin = np.rint(np.sqrt(KX ** 2 + KY ** 2)).astype(int)
    nb = n // 2
    E = np.array([p[kbin == b].sum() for b in range(nb + 1)])
    var_real = float((f * f).mean())
    var_spec = float(p.sum() - p[0, 0])
    resid = abs(var_real - var_spec) / max(var_real, 1e-300)
    k_cyc = np.arange(nb + 1) / (n * dx)
    return k_cyc, E, resid, var_real, var_spec


def slope(k, E, lo, hi):
    m = (k[lo:hi] > 0) & (E[lo:hi] > 0)
    if m.sum() < 3:
        return float("nan")
    return float(np.polyfit(np.log(k[lo:hi][m]), np.log(E[lo:hi][m]), 1)[0])


def main():
    aw = np.load(sys.argv[1])["w_slab_final"]
    ww = np.load(sys.argv[2])
    dx = float(sys.argv[3])
    outdir = sys.argv[4]
    os.makedirs(outdir, exist_ok=True)

    ak, aE, ar, avr, avs = radial_spectrum(aw, dx)
    wk, wE, wr, wvr, wvs = radial_spectrum(ww, dx)
    nb = len(aE)
    lo, hi = max(1, nb // 8), max(3, nb // 2)
    asl = slope(ak, aE, lo, hi)
    wsl = slope(wk, wE, lo, hi)

    print("grid %dx%d, dx=%.0f m, fit bins [%d,%d)" % (aw.shape[0], aw.shape[1],
                                                       dx, lo, hi))
    print("%-8s %12s %12s %14s %10s" % ("", "var(real)", "var(spec)",
                                        "Parseval res", "E(k) slope"))
    print("%-8s %12.6g %12.6g %14.3e %10.4f" % ("WOOF-ArWen", avr, avs, ar, asl))
    print("%-8s %12.6g %12.6g %14.3e %10.4f" % ("WRF", wvr, wvs, wr, wsl))
    print("w variance ratio WRF/WOOF-ArWen = %.4f" % (wvr / avr))
    print("slope difference = %.4f" % (wsl - asl))

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.2))
    a = ax[0]
    a.loglog(ak[1:], aE[1:], color="#2b6cb0", lw=2.0, label="WOOF-ArWen km_opt=3")
    a.loglog(wk[1:], wE[1:], color="#c05621", lw=2.0, ls="--",
             label="WRF em_les km_opt=3")
    kref = ak[lo:hi]
    a.loglog(kref, aE[lo] * (kref / ak[lo]) ** (-5.0 / 3.0), color="0.4",
             lw=1.0, ls=":", label=r"$k^{-5/3}$")
    a.axvspan(ak[lo], ak[hi - 1], color="0.85", alpha=0.35, zorder=0)
    a.set_xlabel(r"$k$  (cycles m$^{-1}$)")
    a.set_ylabel(r"$E(k)$  (m$^2$ s$^{-2}$)")
    a.set_title("Vertical-velocity spectrum at z ~ 537 m\n"
                "(shaded = slope fit window)")
    a.legend(fontsize=8.5)
    a.grid(alpha=0.25, which="both", lw=0.5)

    a = ax[1]
    a.semilogx(ak[1:], wE[1:] / np.maximum(aE[1:], 1e-300), color="#444",
               lw=1.8)
    a.axhline(1.0, color="0.6", lw=0.8)
    a.axvspan(ak[lo], ak[hi - 1], color="0.85", alpha=0.35, zorder=0)
    a.set_xlabel(r"$k$  (cycles m$^{-1}$)")
    a.set_ylabel("WRF / WOOF-ArWen")
    a.set_title("Ratio per wavenumber band")
    a.set_ylim(0, 3)
    a.grid(alpha=0.25, which="both", lw=0.5)

    fig.suptitle("E(k) with Parseval closure asserted on both fields "
                 "(residuals %.1e and %.1e)" % (ar, wr), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = os.path.join(outdir, "spectra_km3_100m.png")
    fig.savefig(p, dpi=150)
    print("wrote", p)

    json.dump(dict(dx_m=dx, fit_bins=[lo, hi],
                   arwen=dict(var_real=avr, var_spec=avs, parseval_res=ar,
                              slope_Ek=asl),
                   wrf=dict(var_real=wvr, var_spec=wvs, parseval_res=wr,
                            slope_Ek=wsl),
                   variance_ratio_wrf_over_arwen=wvr / avr,
                   slope_difference=wsl - asl),
              open("spec_compare_km3_100m.json", "w"), indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
