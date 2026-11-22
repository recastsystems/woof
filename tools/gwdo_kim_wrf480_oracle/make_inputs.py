#!/usr/bin/env python3
"""Build the WRF v4.8.0 gwd_opt = 1 replay inputs.

    python make_inputs.py synthetic OUT_DIR
    python make_inputs.py real OUT_DIR --wrfout F [--wrfout F ...]
        --static-grid GRID.nc --hrrr-geo HRRR_GEO_EM.nc --dem DEM.npy

Each input set S becomes ``gwdo-kim480-inputs-S.bin`` (the 4.8.0 driver's
stream, layout in run_gwdo_kim480.F90), ``gwdo-kim480-inputs-S.npz`` (the
same float32 words plus per-column labels) and ``gwdo-kim471-inputs-S.bin``
(the same columns in the WRF v4.6.1/v4.7.1 driver's layout, for the
two-sided replay of the frozen kernel).

synthetic
    ``L50`` and ``L90`` are the committed regime columns of the WRF v4.6.1
    oracle (tests/data/oracles/gwdo_kim, 149 columns in 12 regimes) with
    ELVMAX added; ``L60top3`` builds the same regimes on 60 levels topped
    at 3 hPa (tools/gwdo_kim_wrf461_oracle/make_columns.py with P_TOP =
    300 Pa) so the mesospheric limiter (p <= 7.5 hPa) is reached.  ELVMAX
    is the column's terrain height (z(1) - dz(1)/2) plus c*VAR, c uniform in
    [0.5, 4.5] (the maximum of the sub-grid terrain sits a few standard
    deviations above its mean), 0 m extra over water (VAR = 0), with three
    deterministic overrides that reach the new branches: ELVMAX = 30 m (no
    drag, ELVMAX <= 50 m), ELVMAX = terrain + 8 km (komax and kbomax high,
    deep blocking search) and ELVMAX = terrain (komax = kref).  Each set
    runs its six (dx, dt) cases twice: WRF v4.8.0's defaults (gwd_dx_factor
    2, gwd_if_nonhyd true) and (1, false).

real
    Every column of each WOOF-ArWen history frame given (the HRRR-recipe
    3 km domain of datasets/era5-hrrr-arwen-pairs), as the PBL driver hands
    gwdo the words (module_pbl_driver.F, phy_prep): mass-point winds (the
    staggered U and V averaged), T = (T + 300) * Exner, p = P + PB, Exner =
    (p/1e5)**(R/cp), z = the mean of the two bounding w-level heights
    (PH + PHB)/g, interface pressure linear in eta between mass levels with
    PSFC at the surface and P_TOP at the top.  The orographic statistics
    VAR, CON, OA1-4 and OL1-4 are the HRRR static's (NCEP's
    hrrr_geo_em.d01.nc) at the nearest HRRR cell; ELVMAX is the maximum of
    the 30 arc-second terrain over the cell, which is how WPS v4.8 writes
    it (util/OrographyStats.py: the box maximum).  Cases: 3 km / 20 s at
    the defaults and at (1, false), and 9 km / 54 s at the defaults.

RUBLTEN/RVBLTEN, which the scheme adds to, are a few 1e-4 m/s2 of seeded
noise in every set, as in the v4.6.1 oracle.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from woof.verify import gwdo_kim480_oracle as go480  # noqa: E402

SEED = 20261109
G, RD = 9.81, 287.0
CP = 7.0 * RD / 2.0
OLD_FIXTURE = REPO / "tests" / "data" / "oracles" / "gwdo_kim"
#: (dx_factor, if_nonhyd): WRF v4.8.0's defaults, then the 4.7.1-like
#: grid length without the nonhydrostatic factor.
SWITCHES = ((2.0, 1), (1.0, 0))
REAL_CASES = ((3000.0, 20.0, 2.0, 1), (3000.0, 20.0, 1.0, 0),
              (9000.0, 54.0, 2.0, 1))


def _make_columns():
    spec = importlib.util.spec_from_file_location(
        "make_columns461", REPO / "tools" / "gwdo_kim_wrf461_oracle"
        / "make_columns.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _expand_cases(f, dx, dt):
    rows = [(a, b, s, n) for s, n in SWITCHES for a, b in zip(dx, dt)]
    f["dx"] = np.asarray([r[0] for r in rows], np.float32)
    f["dt"] = np.asarray([r[1] for r in rows], np.float32)
    f["dx_factor"] = np.asarray([r[2] for r in rows], np.float32)
    f["if_nonhyd"] = np.asarray([r[3] for r in rows], np.int32)


def add_elvmax(f, rng):
    """ELVMAX for regime columns (module docstring)."""
    z = f["z"].astype(np.float64)
    dz = f["dz"].astype(np.float64)
    ht = z[0] - 0.5 * dz[0]
    var = f["var"].astype(np.float64)
    c = rng.uniform(0.5, 4.5, var.size)
    omax = np.where(var > 0.0, ht + c * var, ht)
    idx = np.arange(var.size)
    omax = np.where(idx % 13 == 5, 30.0, omax)
    omax = np.where(idx % 11 == 3, ht + 8000.0, omax)
    omax = np.where(idx % 19 == 7, ht, omax)
    # keep ELVMAX below the second-highest mass level everywhere: WRF reads
    # one level past the column above it (terrain_drag_gwdo480.cu).
    omax = np.minimum(omax, z[-2] - 1.0)
    f["elvmax"] = omax.astype(np.float32)


def synthetic(out: Path) -> None:
    rng = np.random.default_rng(SEED)
    sets = {}
    for grid in ("L50", "L90"):
        with np.load(OLD_FIXTURE / f"gwdo-kim-inputs-{grid}.npz") as d:
            f = {k: d[k] for k in d.files}
        dx, dt = f["dx"], f["dt"]
        _expand_cases(f, dx, dt)
        add_elvmax(f, rng)
        sets[grid] = f
    mc = _make_columns()
    mc.P_TOP = 300.0
    mc.EDGE_NOTES.clear()
    f = mc.build_grid(60, np.random.default_rng(SEED + 7),
                      np.random.default_rng(SEED + 8))
    f["edge_notes"] = np.asarray(mc.EDGE_NOTES)
    _expand_cases(f, f["dx"], f["dt"])
    add_elvmax(f, rng)
    sets["L60top3"] = f
    for name, f in sets.items():
        _write(out, name, f)


def _lambert_xy(lat, lon, *, truelat, stand_lon):
    """HRRR's Lambert conformal plane (WRF sphere, R = 6370 km)."""
    from pyproj import Proj
    p = Proj(proj="lcc", lat_1=truelat, lat_2=truelat, lat_0=truelat,
             lon_0=stand_lon, a=6370000.0, b=6370000.0)
    return p(lon, lat)


def _hrrr_index(lat, lon, geo):
    """Nearest HRRR mass point to each (lat, lon)."""
    glat = np.asarray(geo["XLAT_M"][0], np.float64)
    glon = np.asarray(geo["XLONG_M"][0], np.float64)
    t = float(geo.getncattr("TRUELAT1"))
    s = float(geo.getncattr("STAND_LON"))
    dx = float(geo.getncattr("DX"))
    x0, y0 = _lambert_xy(glat[0, 0], glon[0, 0], truelat=t, stand_lon=s)
    x, y = _lambert_xy(lat, lon, truelat=t, stand_lon=s)
    i = np.rint((np.asarray(x) - x0) / dx).astype(int)
    j = np.rint((np.asarray(y) - y0) / dx).astype(int)
    ny, nx = glat.shape
    if i.min() < 0 or j.min() < 0 or i.max() >= nx or j.max() >= ny:
        raise SystemExit("domain leaves the HRRR grid")
    err = np.hypot(glat[j, i] - lat, (glon[j, i] - lon)
                   * np.cos(np.radians(lat)))
    if err.max() > 0.03:
        raise SystemExit(f"HRRR nearest-cell match off by {err.max():.3f} deg")
    return j, i


def _elvmax(lat, lon, dx, dem):
    """Maximum of the 30" terrain over each cell (+-dx/2 in metres).

    The DEM is (21600, 43200) int16, north row first, west column first at
    180 W; pixel (r, c) is centred at 90 - (r + 0.5)/120, -180 + (c + 0.5)/120.
    """
    half_lat = 0.5 * dx / 111195.0
    out = np.empty(lat.size, np.float64)
    flat_lat, flat_lon = lat.ravel(), lon.ravel()
    for n, (a, b) in enumerate(zip(flat_lat, flat_lon)):
        half_lon = half_lat / np.cos(np.radians(a))
        r0 = int(np.floor((90.0 - (a + half_lat)) * 120.0))
        r1 = int(np.ceil((90.0 - (a - half_lat)) * 120.0))
        c0 = int(np.floor((b - half_lon + 180.0) * 120.0))
        c1 = int(np.ceil((b + half_lon + 180.0) * 120.0))
        out[n] = float(dem[r0:r1, c0:c1].max())
    return np.maximum(out, 0.0).reshape(lat.shape)


def real(out: Path, wrfouts, static_grid, hrrr_geo, dem_path) -> None:
    import netCDF4 as nc

    dem = np.load(dem_path, mmap_mode="r")
    with nc.Dataset(static_grid) as g:
        lat = np.asarray(g["XLAT"][0], np.float64)
        lon = np.asarray(g["XLONG"][0], np.float64)
        sina = np.asarray(g["SINALPHA"][0], np.float32)
        cosa = np.asarray(g["COSALPHA"][0], np.float32)
    ny, nx = lat.shape
    with nc.Dataset(hrrr_geo) as geo:
        j, i = _hrrr_index(lat, lon, geo)
        stats = {name: np.asarray(geo[name][0], np.float32)[j, i]
                 for name in ("VAR", "CON", "OA1", "OA2", "OA3", "OA4",
                              "OL1", "OL2", "OL3", "OL4")}
    elvmax = _elvmax(lat, lon, 3000.0, dem).astype(np.float32)
    rng = np.random.default_rng(SEED + 100)
    for path in wrfouts:
        path = Path(path)
        with nc.Dataset(path) as d:
            u = np.asarray(d["U"][0], np.float64)
            v = np.asarray(d["V"][0], np.float64)
            th = np.asarray(d["T"][0], np.float64) + 300.0
            p = np.asarray(d["P"][0], np.float64) + np.asarray(
                d["PB"][0], np.float64)
            zw = (np.asarray(d["PH"][0], np.float64)
                  + np.asarray(d["PHB"][0], np.float64)) / G
            qv = np.asarray(d["QVAPOR"][0], np.float64)
            psfc = np.asarray(d["PSFC"][0], np.float64)
            ptop = float(np.asarray(d["P_TOP"][0]))
            znu = np.asarray(d["ZNU"][0], np.float64)
            znw = np.asarray(d["ZNW"][0], np.float64)
        nz = th.shape[0]
        um = 0.5 * (u[:, :, :-1] + u[:, :, 1:])
        vm = 0.5 * (v[:, :-1, :] + v[:, 1:, :])
        exner = (p / 1.0e5) ** (RD / CP)
        t = th * exner
        z = 0.5 * (zw[:-1] + zw[1:])
        pw = np.empty((nz + 1, ny, nx))
        pw[0] = psfc
        pw[nz] = ptop
        for k in range(1, nz):
            w = (znw[k] - znu[k - 1]) / (znu[k] - znu[k - 1])
            pw[k] = p[k - 1] + w * (p[k] - p[k - 1])
        ncol = ny * nx
        f = {"u": um, "v": vm, "t": t, "qv": np.maximum(qv, 0.0), "p": p,
             "pi": exner, "z": z, "dz": np.diff(zw, axis=0),
             "p_interface": pw}
        f = {k: np.ascontiguousarray(a.reshape(a.shape[0], ncol),
                                     dtype=np.float32) for k, a in f.items()}
        f["rublten0"] = rng.normal(0.0, 2.0e-4, (nz, ncol)).astype(np.float32)
        f["rvblten0"] = rng.normal(0.0, 2.0e-4, (nz, ncol)).astype(np.float32)
        f["var"] = stats["VAR"].reshape(ncol)
        f["oc1"] = stats["CON"].reshape(ncol)
        f["oa"] = np.stack([stats[f"OA{m}"].reshape(ncol) for m in range(1, 5)])
        f["ol"] = np.stack([stats[f"OL{m}"].reshape(ncol) for m in range(1, 5)])
        f["sina"] = sina.reshape(ncol)
        f["cosa"] = cosa.reshape(ncol)
        f["elvmax"] = elvmax.reshape(ncol)
        f["dx"] = np.asarray([c[0] for c in REAL_CASES], np.float32)
        f["dt"] = np.asarray([c[1] for c in REAL_CASES], np.float32)
        f["dx_factor"] = np.asarray([c[2] for c in REAL_CASES], np.float32)
        f["if_nonhyd"] = np.asarray([c[3] for c in REAL_CASES], np.int32)
        stamp = path.name.replace("wrfout_d01_", "").replace(":", "")
        f["regime"] = np.asarray([f"real-{stamp}"] * ncol)
        f["source"] = np.asarray(str(path))
        f["column_ij"] = np.stack(np.unravel_index(np.arange(ncol),
                                                   (ny, nx))).astype(np.int32)
        _write(out, f"real-{stamp}", f)


def _write(out: Path, name: str, f: dict) -> None:
    for k in go480.LEVEL_INPUTS + ("p_interface",) + go480.COLUMN_INPUTS:
        if not np.all(np.isfinite(f[k])):
            raise SystemExit(f"{name}: non-finite {k}")
    out.mkdir(parents=True, exist_ok=True)
    go480.write_inputs_bin(out / f"gwdo-kim480-inputs-{name}.bin", f)
    go480.write_inputs_bin471(out / f"gwdo-kim471-inputs-{name}.bin", f)
    np.savez(out / f"gwdo-kim480-inputs-{name}.npz", **f)
    print(f"{name}: {f['u'].shape[1]} columns x {f['u'].shape[0]} levels, "
          f"{len(f['dx'])} cases")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("which", choices=("synthetic", "real"))
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--wrfout", action="append", default=[])
    ap.add_argument("--static-grid", type=Path)
    ap.add_argument("--hrrr-geo", type=Path)
    ap.add_argument("--dem", type=Path)
    a = ap.parse_args()
    if a.which == "synthetic":
        synthetic(a.out_dir)
    else:
        real(a.out_dir, a.wrfout, a.static_grid, a.hrrr_geo, a.dem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
