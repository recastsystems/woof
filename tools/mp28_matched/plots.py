"""Complete PNG set for the mp=28 matched-trajectory comparison.

Organised ``<out>/d01/<product>/<run>_t<seconds>.png`` -- every product, every
frame, every run.  Nothing is curated.

Colour follows the *entity* (which model/build produced the number), never its
rank; the microphysics option is carried by line style, so series identity is
never colour-alone.  The three entity hues are Okabe-Ito blue / vermillion /
bluish-green, validated: worst adjacent CVD dE 11.0 (deutan), normal-vision
dE 25.8, all three inside the lightness band and above 3:1 on the surface.

Field panels use one-hue sequential ramps for magnitudes and a two-hue
diverging ramp with a neutral midpoint for the signed fields (w, theta').

Usage:  python plots.py --runs DIR --wrfinput PATH --out DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

#: Entity colours (Okabe-Ito).  Fixed order, never cycled.
ENTITY = {
    "arwen": "#0072B2",
    "wrf-vec": "#D55E00",
    "wrf-novec": "#009E73",
}
#: Microphysics option -> line style.  The secondary encoding.
MP_STYLE = {8: "-", 28: "--"}

RUNS = (
    ("arwen-mp08-a", "arwen", 8, "WOOF-ArWen mp=8"),
    ("arwen-mp28-a", "arwen", 28, "WOOF-ArWen mp=28"),
    ("wrf-vec-mp08-x", "wrf-vec", 8, "WRF -ftree-vectorize mp=8"),
    ("wrf-vec-mp28-x", "wrf-vec", 28, "WRF -ftree-vectorize mp=28"),
    ("wrf-novec-mp08-x", "wrf-novec", 8, "WRF -fno-tree-vectorize mp=8"),
    ("wrf-novec-mp28-x", "wrf-novec", 28, "WRF -fno-tree-vectorize mp=28"),
)

#: 3-D field -> (product name, colour ramp, unit scale, unit label, diverging)
XZ_PRODUCTS = {
    "QCLOUD": ("qc-xz", "Blues", 1.0e3, "qc (g/kg)", False),
    "QRAIN": ("qr-xz", "Greens", 1.0e3, "qr (g/kg)", False),
    "QICE": ("qi-xz", "Purples", 1.0e3, "qi (g/kg)", False),
    "QSNOW": ("qs-xz", "PuBu", 1.0e3, "qs (g/kg)", False),
    "QGRAUP": ("qg-xz", "YlOrBr", 1.0e3, "qg (g/kg)", False),
    "W": ("w-xz", "RdBu_r", 1.0, "w (m/s)", True),
    "QNCLOUD": ("nc-xz", "Blues", 1.0e-6, "nc (10^6 /kg)", False),
    "QNWFA": ("nwfa-xz", "Oranges", 1.0e-6, "nwfa (10^6 /kg)", False),
    "QNIFA": ("nifa-xz", "Reds", 1.0e-3, "nifa (10^3 /kg)", False),
}

GRID = dict(color="#d9d9d6", linewidth=0.6)
INK = "#2b2b28"


def style_axes(ax):
    ax.tick_params(colors=INK, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#b8b8b2")
    ax.grid(True, **GRID)
    ax.set_axisbelow(True)


def half_level_heights(wrfinput: Path) -> np.ndarray:
    import netCDF4
    with netCDF4.Dataset(wrfinput) as ds:
        phb = np.asarray(ds.variables["PHB"][0], dtype=np.float64)
    zf = phb.mean(axis=(1, 2)) / 9.81
    return 0.5 * (zf[:-1] + zf[1:]), zf


def xz_panel(field, x_km, z_km, path, title, cmap, label, diverging):
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    if diverging:
        lim = float(np.abs(field).max()) or 1.0
        cs = ax.contourf(x_km, z_km, field, levels=np.linspace(-lim, lim, 21),
                         cmap=cmap)
    else:
        hi = float(field.max())
        levels = np.linspace(0.0, hi, 15) if hi > 0 else np.linspace(0, 1, 3)
        cs = ax.contourf(x_km, z_km, field, levels=levels, cmap=cmap)
    ax.set_xlabel("x (km)", color=INK, fontsize=9)
    ax.set_ylabel("z (km)", color=INK, fontsize=9)
    ax.set_ylim(0.0, 18.0)
    style_axes(ax)
    cb = fig.colorbar(cs, ax=ax, pad=0.02)
    cb.set_label(label, color=INK, fontsize=8)
    cb.ax.tick_params(labelsize=7, colors=INK)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)


def plan_panel(field, x_km, y_km, path, title, cmap, label, diverging=False):
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    if diverging:
        lim = float(np.abs(field).max()) or 1.0
        cs = ax.contourf(x_km, y_km, field,
                         levels=np.linspace(-lim, lim, 21), cmap=cmap)
    else:
        hi = float(field.max())
        levels = np.linspace(0.0, hi, 15) if hi > 0 else np.linspace(0, 1, 3)
        cs = ax.contourf(x_km, y_km, field, levels=levels, cmap=cmap)
    ax.set_xlabel("x (km)", color=INK, fontsize=9)
    ax.set_ylabel("y (km)", color=INK, fontsize=9)
    ax.set_aspect("equal")
    style_axes(ax)
    cb = fig.colorbar(cs, ax=ax, pad=0.02)
    cb.set_label(label, color=INK, fontsize=8)
    cb.ax.tick_params(labelsize=7, colors=INK)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, type=Path)
    ap.add_argument("--wrfinput", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--comparison", type=Path, default=None)
    args = ap.parse_args()
    root = args.out / "d01"
    root.mkdir(parents=True, exist_ok=True)

    z_half, z_full = half_level_heights(args.wrfinput)
    zh_km, zf_km = z_half / 1000.0, z_full / 1000.0

    available = [(d, e, mp, lab) for d, e, mp, lab in RUNS
                 if (args.runs / d / "series.json").exists()]

    # ---- per-frame field products -------------------------------------
    for d, entity, mp, label in available:
        rd = args.runs / d
        for p in sorted(rd.glob("frame_t*.npz")):
            t = int(float(p.stem.split("frame_t")[1]))
            z = np.load(p)
            nz, ny, nx = z["QCLOUD"].shape
            x_km = (np.arange(nx) + 0.5) * 2.0
            y_km = (np.arange(ny) + 0.5) * 2.0
            jc = ny // 2
            for name, (prod, cmap, scale, unit, div) in XZ_PRODUCTS.items():
                if name not in z.files:
                    continue
                out = root / prod
                out.mkdir(parents=True, exist_ok=True)
                f = z[name].astype(np.float64) * scale
                zz = zf_km if name == "W" else zh_km
                xz_panel(f[:, jc, :], x_km, zz, out / f"{d}_t{t:05d}.png",
                         f"{label}   t = {t} s   y = {y_km[jc]:.0f} km",
                         cmap, unit, div)
            # plan views
            out = root / "wmax-plan"
            out.mkdir(parents=True, exist_ok=True)
            plan_panel(z["W"].astype(np.float64).max(axis=0), x_km, y_km,
                       out / f"{d}_t{t:05d}.png",
                       f"{label}   t = {t} s   column-max w",
                       "Reds", "max w (m/s)")
            out = root / "condensate-path-plan"
            out.mkdir(parents=True, exist_ok=True)
            tot = sum(z[k].astype(np.float64) for k in
                      ("QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP")
                      if k in z.files)
            plan_panel(tot.sum(axis=0) * 1.0e3, x_km, y_km,
                       out / f"{d}_t{t:05d}.png",
                       f"{label}   t = {t} s   summed condensate",
                       "Blues", "sum_k q (g/kg)")
            out = root / "rainnc-plan"
            out.mkdir(parents=True, exist_ok=True)
            plan_panel(z["RAINNC"].astype(np.float64), x_km, y_km,
                       out / f"{d}_t{t:05d}.png",
                       f"{label}   t = {t} s   accumulated surface rain",
                       "GnBu", "RAINNC (mm)")
            # mean profiles
            out = root / "mean-profiles"
            out.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(4.6, 5.2))
            for k, c in (("QCLOUD", "#0072B2"), ("QRAIN", "#009E73"),
                         ("QICE", "#CC79A7"), ("QSNOW", "#56B4E9"),
                         ("QGRAUP", "#D55E00")):
                if k in z.files:
                    ax.plot(z[k].astype(np.float64).mean(axis=(1, 2)) * 1e3,
                            zh_km, color=c, linewidth=2.0, label=k)
            ax.set_xlabel("domain-mean mixing ratio (g/kg)", color=INK,
                          fontsize=9)
            ax.set_ylabel("z (km)", color=INK, fontsize=9)
            ax.set_ylim(0.0, 18.0)
            style_axes(ax)
            ax.legend(frameon=False, fontsize=8, labelcolor=INK)
            ax.set_title(f"{label}   t = {t} s", color=INK, fontsize=10,
                         loc="left")
            fig.savefig(out / f"{d}_t{t:05d}.png", dpi=130,
                        bbox_inches="tight", facecolor="#fcfcfb")
            plt.close(fig)

    # ---- time series, all runs overlaid --------------------------------
    series = {d: json.loads((args.runs / d / "series.json").read_text())
              for d, _, _, _ in available}
    keys = sorted({k for s in series.values() for r in s for k in r
                   if k not in ("step", "time_s")})
    out = root / "timeseries"
    out.mkdir(parents=True, exist_ok=True)
    for key in keys:
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        drew = 0
        for d, entity, mp, label in available:
            rows = [r for r in series[d] if r.get(key) is not None]
            if not rows:
                continue
            ax.plot([r["time_s"] for r in rows], [r[key] for r in rows],
                    color=ENTITY[entity], linestyle=MP_STYLE[mp],
                    linewidth=2.0, label=label)
            drew += 1
        if not drew:
            plt.close(fig)
            continue
        ax.set_xlabel("time (s)", color=INK, fontsize=9)
        ax.set_ylabel(key, color=INK, fontsize=9)
        ax.axvline(5400.0, color="#b8b8b2", linewidth=1.0, linestyle=":")
        style_axes(ax)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK, ncol=2)
        ax.set_title(f"{key} -- matched periodic case (dotted line: end of "
                     "the primary analysis window)",
                     color=INK, fontsize=10, loc="left")
        fig.savefig(out / f"{key}.png", dpi=130, bbox_inches="tight",
                    facecolor="#fcfcfb")
        plt.close(fig)

    # ---- signature and divergence, from comparison.json -----------------
    if args.comparison and args.comparison.exists():
        rep = json.loads(args.comparison.read_text())
        times = rep["times"]
        out = root / "signature"
        out.mkdir(parents=True, exist_ok=True)
        for m, s in rep["signature"].items():
            if all(v is None for v in s["wrf"]):
                continue
            fig, ax = plt.subplots(figsize=(7.2, 4.0))
            fl = rep["floor_mp08_model_pair"][m]
            fl_num = [0.0 if v is None else v for v in fl]
            ax.fill_between(times, [-v for v in fl_num], fl_num,
                            color="#e8e8e4", label="mp=8 model-pair floor")
            ax.plot(times, s["wrf"], color=ENTITY["wrf-vec"], linewidth=2.0,
                    label="WRF  mp28 - mp8")
            ax.plot(times, s["arwen"], color=ENTITY["arwen"], linewidth=2.0,
                    linestyle="--", label="WOOF-ArWen  mp28 - mp8")
            ax.axhline(0.0, color="#b8b8b2", linewidth=1.0)
            ax.axvline(5400.0, color="#b8b8b2", linewidth=1.0, linestyle=":")
            ax.set_xlabel("time (s)", color=INK, fontsize=9)
            ax.set_ylabel(f"delta {m}", color=INK, fontsize=9)
            style_axes(ax)
            ax.legend(frameon=False, fontsize=8, labelcolor=INK)
            ax.set_title(f"aerosol signature: {m}", color=INK, fontsize=10,
                         loc="left")
            fig.savefig(out / f"{m}.png", dpi=130, bbox_inches="tight",
                        facecolor="#fcfcfb")
            plt.close(fig)

        out = root / "divergence"
        out.mkdir(parents=True, exist_ok=True)
        div = rep["m8_divergence"]
        fields = sorted({f for k in div for r in div[k].values() for f in r})
        for f in fields:
            fig, ax = plt.subplots(figsize=(7.2, 4.0))
            for tag, mp in (("mp08", 8), ("mp28", 28)):
                ts = sorted(float(t) for t in div[tag])
                vals = [div[tag][str(t)].get(f) for t in ts]
                pts = [(t, v) for t, v in zip(ts, vals) if v is not None]
                if not pts:
                    continue
                ax.plot([p[0] for p in pts], [p[1] for p in pts],
                        color=ENTITY["arwen"], linestyle=MP_STYLE[mp],
                        linewidth=2.0, label=f"WOOF-ArWen vs WRF, mp={mp}")
            ax.set_xlabel("time (s)", color=INK, fontsize=9)
            ax.set_ylabel(f"||WOOF-ArWen - WRF||2 / ||WRF||2   [{f}]", color=INK,
                          fontsize=9)
            ax.axvline(5400.0, color="#b8b8b2", linewidth=1.0, linestyle=":")
            style_axes(ax)
            ax.legend(frameon=False, fontsize=8, labelcolor=INK)
            ax.set_title(f"trajectory divergence: {f}", color=INK,
                         fontsize=10, loc="left")
            fig.savefig(out / f"{f}.png", dpi=130, bbox_inches="tight",
                        facecolor="#fcfcfb")
            plt.close(fig)

    n = sum(1 for _ in root.rglob("*.png"))
    print(f"wrote {n} PNGs under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
