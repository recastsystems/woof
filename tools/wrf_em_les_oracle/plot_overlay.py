#!/usr/bin/env python3
"""Overlay WOOF-ArWen and WRF em_les profiles, both reduced by the same routine.

Usage: plot_overlay.py <same_instrument_LABEL_profiles.npz> <label> <outdir>
"""
import sys
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

QS = 0.24

A_COL = "#2b6cb0"     # ArWen
W_COL = "#c05621"     # WRF


def main():
    npz, label, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(outdir, exist_ok=True)
    d = np.load(npz)
    azi = float(d["azi"])
    wzi = float(d["wzi"])

    fig, ax = plt.subplots(1, 4, figsize=(16.5, 5.6))

    # 1: total, resolved, SGS heat flux normalised by Qs, vs z/zi
    a = ax[0]
    a.plot(d["atotal"] / QS, d["az"] / azi, color=A_COL, lw=2.2,
           label="WOOF-ArWen total")
    a.plot(d["wtotal"] / QS, d["wz"] / wzi, color=W_COL, lw=2.2, ls="--",
           label="WRF total")
    a.plot(d["ares"] / QS, d["az"] / azi, color=A_COL, lw=1.2, alpha=0.65,
           label="WOOF-ArWen resolved")
    a.plot(d["wres"] / QS, d["wz"] / wzi, color=W_COL, lw=1.2, ls="--",
           alpha=0.65, label="WRF resolved")
    asg = 0.5 * (d["asgs"][:-1] + d["asgs"][1:])
    wsg = 0.5 * (d["wsgs"][:-1] + d["wsgs"][1:])
    a.plot(asg / QS, d["az"] / azi, color=A_COL, lw=1.0, ls=":",
           label="WOOF-ArWen SGS")
    a.plot(wsg / QS, d["wz"] / wzi, color=W_COL, lw=1.0, ls="-.",
           label="WRF SGS")
    a.axvline(0, color="0.6", lw=0.7)
    a.axhline(1.0, color="0.6", lw=0.7)
    a.set_xlabel(r"heat flux / $Q_s$")
    a.set_ylabel(r"$z/z_i$")
    a.set_title("Heat flux partition")
    a.set_ylim(0, 1.35)
    a.legend(fontsize=7.5, loc="upper right")

    # 2: theta
    a = ax[1]
    a.plot(d["atheta"], d["az"], color=A_COL, lw=2.2, label="WOOF-ArWen")
    a.plot(d["wtheta"], d["wz"], color=W_COL, lw=2.2, ls="--", label="WRF")
    a.axhline(azi, color=A_COL, lw=0.8, ls=":")
    a.axhline(wzi, color=W_COL, lw=0.8, ls=":")
    a.set_xlabel(r"$\overline{\theta}$  (K)")
    a.set_ylabel("z (m)")
    a.set_title(r"Potential temperature ($z_i$ dotted)")
    a.set_xlim(299.8, 303.5)
    a.legend(fontsize=8)

    # 3: resolved TKE
    a = ax[2]
    a.plot(d["atke"], d["az"] / azi, color=A_COL, lw=2.2, label="WOOF-ArWen")
    a.plot(d["wtke"], d["wz"] / wzi, color=W_COL, lw=2.2, ls="--", label="WRF")
    a.set_xlabel(r"resolved TKE  (m$^2$ s$^{-2}$)")
    a.set_ylabel(r"$z/z_i$")
    a.set_title("Resolved TKE")
    a.set_ylim(0, 1.35)
    a.legend(fontsize=8)

    # 4: difference in total flux
    a = ax[3]
    zc = np.linspace(0.02, 1.25, 200)
    ai = np.interp(zc, d["az"] / azi, d["atotal"] / QS)
    wi = np.interp(zc, d["wz"] / wzi, d["wtotal"] / QS)
    a.plot(wi - ai, zc, color="#444444", lw=2.0)
    a.axvline(0, color="0.6", lw=0.7)
    a.set_xlabel(r"(WRF $-$ WOOF-ArWen) total flux / $Q_s$")
    a.set_ylabel(r"$z/z_i$")
    a.set_title("Difference")
    a.set_ylim(0, 1.35)
    mx = float(np.max(np.abs(wi - ai)))
    a.set_xlim(-max(mx * 1.3, 0.01), max(mx * 1.3, 0.01))
    a.text(0.03, 0.03, "max |diff| = %.4f $Q_s$" % mx, transform=a.transAxes,
           fontsize=8.5)

    for a in ax:
        a.grid(alpha=0.25, lw=0.6)
    fig.suptitle("WOOF-ArWen CBL vs WRF v4.6.1 em_les  --  %s  --  "
                 "both reduced by the same routine, final 30 min" % label,
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(outdir, "profiles_%s.png" % label)
    fig.savefig(out, dpi=150)
    print("wrote", out, "max|diff| = %.5f Qs" % mx)


if __name__ == "__main__":
    main()
