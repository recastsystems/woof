"""Build New Tiedtke driver-boundary columns from a wrfout file.

THE CROSS-FEED. run_nt_live.F90 already drives the byte-unmodified WRF
Fortran over columns dumped live from WOOF-ArWen. This produces the same column
format from a wrfout instead, so the SAME implementation can be driven
from WRF's own state and from WOOF-ArWen's, and the two compared. Identical
code, two states -- the mirror of the first experiment, which held the
state fixed and varied the code.

WHY IT IS POSSIBLE AT ALL. wrfout does not carry rthften/rqvften, so a
cross-feed would have been blocked on an input we cannot reconstruct. The
forcing ablation measured that lane at 0.017 K per 100% perturbation in
New Tiedtke, so it can be zeroed in BOTH column sets without materially
changing either answer. The one input we cannot get is the one we proved
does not matter.

EVERY FIELD HERE IS A RECONSTRUCTION, and reconstruction is where this
goes wrong. Two traps already found and closed:

  * WRF's T is DRY theta - 300 and THM is the moist one, under
    USE_THETA_M = 1. Verified numerically rather than read: the identity
    THM = (T+300)(1 + Rv/Rd qv) - 300 holds to 6.9e-05 K over the whole
    d02 field. Taking THM would have been an 11.2 K error at low levels.
  * Only fields BOTH WOOF-ArWen and WRF write are used. WOOF-ArWen's wrfout carries
    77 variables against WRF's 203 and has no DNW/DN/C1F/C2F/P_HYD/THM --
    so dnw and dn are derived from ZNW, which both carry. That is what
    makes the round-trip guard possible: the same code path can be run on
    WOOF-ArWen's own wrfout and checked against the live dump, which is the
    only way to know the reconstruction is right.

Formulas are WRF's own, from the 4.8.0 tree:
  dz8w, p8w      dyn_em/module_big_step_utilities_em.F:4874-4930
                 (note dz8w(kte) = 0 -- WRF's choice, replicated)
  fnm, fnp       dyn_em/module_initialize_real.F:3744-3749
  alt (EOS)      p = p0 (Rd theta_m / (p0 alt))^gamma

    python wrfout_columns.py <wrfout> <out-prefix> [--cols N] [--like FILE]
"""
from __future__ import annotations

import argparse
import pathlib
import struct

import numpy as np
from netCDF4 import Dataset

G = 9.81
RD = 287.0
CP = 7.0 * RD / 2.0
RVOVRD = 461.6 / 287.0
P1000 = 100000.0
GAMMA = CP / (CP - RD)

LEV_IN = ("t3d", "qv3d", "qc3d", "qi3d", "u3d", "v3d", "pcps", "dz8w",
          "rho3d", "exner", "qvften", "thften")
IFACE_IN = ("p8w", "w")
SFC_IN = ("xland", "hfx", "qfx")


def _hex(x) -> str:
    return "%08x" % struct.unpack("<I", struct.pack("<f", float(x)))[0]


def build(path: str, coeffs: str | None = None):
    """Every driver-boundary field, on the mass grid, bottom-up.

    ``coeffs`` names a file to take C1H/C2H from. WOOF-ArWen's wrfout writes 77
    variables against WRF's 203 and carries no C1H/C2H, but the hybrid
    coefficients are functions of ZNW and etac alone -- and ZNW is BIT-
    IDENTICAL between the two models here, with the same P_TOP, so WRF's
    are WOOF-ArWen's. Checked, not assumed.
    """
    d = Dataset(path)
    _co = Dataset(coeffs) if coeffs else d
    v = lambda n: np.asarray(d.variables[n][0], dtype=np.float64)

    p = v("P") + v("PB")                       # (nz, ny, nx)
    exner = (p / P1000) ** (RD / CP)
    theta = v("T") + 300.0                     # DRY, verified
    t3d = theta * exner
    qv, qc, qi = v("QVAPOR"), v("QCLOUD"), v("QICE")

    U, V = v("U"), v("V")                      # staggered
    u3d = 0.5 * (U[:, :, :-1] + U[:, :, 1:])
    v3d = 0.5 * (V[:, :-1, :] + V[:, 1:, :])

    zw = (v("PH") + v("PHB")) / G              # (nz+1, ny, nx)
    nz = p.shape[0]
    dz8w = np.zeros_like(p)
    dz8w[:nz - 1] = zw[1:nz] - zw[:nz - 1]
    dz8w[nz - 1] = 0.0                         # WRF sets dz8w(kte) = 0

    # phy_prep:4856 is rho = 1/alt * (1 + qv) -- the MOIST density, not
    # 1/alt. I wrote 1/alt first, the round-trip against ArWen's live dump
    # failed at 2.3e-02, and I was one step from reporting "ArWen hands
    # New Tiedtke a moist density where WRF hands it a dry one" as a
    # finding. ArWen's own comment cites this exact line and is right.
    # The guard existed precisely for this.
    theta_m = theta * (1.0 + RVOVRD * qv)
    alt = (RD * theta_m / P1000) * (p / P1000) ** (-1.0 / GAMMA)
    rho3d = (1.0 + qv) / alt

    # THE PRESSURE THE CUMULUS DRIVER RECEIVES IS HYDROSTATIC, NOT P+PB.
    # module_first_rk_step_part1.F:1565 passes P=grid%p_hyd and
    # P8W=grid%p_hyd_w, and phy_prep:4946-4957 builds p_hyd_w by downward
    # integration from p_top:
    #   p_hyd_w(kte) = p_top
    #   p_hyd_w(k)   = p_hyd_w(k+1) - (1+qtot)*(c1(k)*MUT + c2(k))*dnw(k)
    # which is line-for-line ArWen's _prepare_atmosphere. I first wrote
    # P+PB with an fnm/fnp interpolation, the round-trip failed at 8e-04,
    # and the non-hydrostatic term is 3.0e-03 of p on this frame -- so
    # that was a real error, not rounding. Second of three reconstructions
    # the guard caught.
    znw = np.asarray(d.variables["ZNW"][0], dtype=np.float64)
    dnw = znw[1:] - znw[:-1]
    mut = v("MU") + v("MUB")
    c1h = np.asarray(_co.variables["C1H"][0], dtype=np.float64)
    c2h = np.asarray(_co.variables["C2H"][0], dtype=np.float64)
    ptop = float(np.asarray(d.variables["P_TOP"][0]))
    qtot = qv + qc + qi
    for extra in ("QRAIN", "QSNOW", "QGRAUP"):
        if extra in d.variables:
            qtot = qtot + v(extra)
    p8w = np.zeros_like(zw)
    p8w[nz] = ptop
    for k in range(nz - 1, -1, -1):
        p8w[k] = p8w[k + 1] - ((1.0 + qtot[k])
                               * (c1h[k] * mut + c2h[k]) * dnw[k])
    p = 0.5 * (p8w[:nz] + p8w[1:nz + 1])       # p_hyd on mass levels
    exner = (p / P1000) ** (RD / CP)
    t3d = theta * exner

    w = v("W")                                 # already at interfaces
    landmask = v("LANDMASK")
    xland = 2.0 - landmask                     # WRF: 1 land, 2 water
    out = {
        "t3d": t3d, "qv3d": qv, "qc3d": qc, "qi3d": qi, "u3d": u3d,
        "v3d": v3d, "pcps": p, "dz8w": dz8w, "rho3d": rho3d,
        "exner": exner,
        # THE FORCING PAIR IS ZEROED, not reconstructed. wrfout does not
        # carry it, and the ablation measured its authority at 0.017 K per
        # 100% in NT -- so zeroing it in BOTH sets costs less than the
        # difference being looked for. Stated, not hidden.
        "qvften": np.zeros_like(p), "thften": np.zeros_like(p),
    }
    iface = {"p8w": p8w, "w": w}
    sfc = {"xland": xland, "hfx": v("HFX"), "qfx": v("QFX")}
    # HGT is carried for SELECTION only -- it is not a driver
    # input. The land mask alone is not enough: this domain has
    # points flagged sea (xland 2) sitting at 423 m, whose low
    # surface pressure is elevation and not a storm. Selecting
    # on the mask alone put the "centre" on one of them at a
    # fixed 963.6 hPa across five forecast hours while the real
    # storm went 988 -> 975. Third instance of this trap today.
    sfc["_hgt"] = v("HGT")
    d.close()
    return out, iface, sfc, float(getattr(Dataset(path), "DX", 4500.0))


def write(prefix, sel, lev, iface, sfc, dx, nz, meta_extra=()):
    """sel is a list of (j, i) mass-point indices."""
    nl = chr(10)
    root = pathlib.Path(prefix)
    root.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{prefix}-lev.csv", "w", newline=nl) as f:
        f.write("col,k," + ",".join(LEV_IN) + nl)
        for c, (j, i) in enumerate(sel, start=1):
            for k in range(nz):
                f.write(f"{c},{k + 1}," + ",".join(
                    _hex(lev[n][k, j, i]) for n in LEV_IN) + nl)
    with open(f"{prefix}-iface.csv", "w", newline=nl) as f:
        f.write("col,k," + ",".join(IFACE_IN) + nl)
        for c, (j, i) in enumerate(sel, start=1):
            for k in range(nz + 1):
                f.write(f"{c},{k + 1}," + ",".join(
                    _hex(iface[n][k, j, i]) for n in IFACE_IN) + nl)
    with open(f"{prefix}-sfc.csv", "w", newline=nl) as f:
        f.write("col,gridcol," + ",".join(SFC_IN) + ",dx" + nl)
        for c, (j, i) in enumerate(sel, start=1):
            f.write(f"{c},{j * 100000 + i}," + ",".join(
                _hex(sfc[n][j, i]) for n in SFC_IN) + "," + _hex(dx) + nl)
    with open(f"{prefix}-meta.txt", "w", newline=nl) as f:
        f.write(f"ncol_selected {len(sel)}{nl}nz {nz}{nl}")
        f.write(f"scheme_dt 20.0{nl}stepcu 1{nl}itimestep 2{nl}")
        f.write(f"surface_first 1{nl}dx {dx!r}{nl}")
        for line in meta_extra:
            f.write(line + nl)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("wrfout")
    ap.add_argument("prefix")
    ap.add_argument("--cols", type=int, default=96)
    ap.add_argument("--coeffs", default=None,
                    help="file to take C1H/C2H from when the "
                         "input lacks them")
    ap.add_argument("--gridcols", default=None,
                    help="comma-separated j*100000+i keys, to reuse a "
                         "selection across files")
    a = ap.parse_args()

    lev, iface, sfc, dx = build(a.wrfout, a.coeffs)
    nz, ny, nx = lev["t3d"].shape

    if a.gridcols:
        keys = [int(x) for x in a.gridcols.split(",")]
        sel = [(k // 100000, k % 100000) for k in keys]
    else:
        # lowest surface pressure among SEA points -- the same selection
        # rule the live dump uses, and for the same reason: taking the
        # minimum outright returns Jamaica's mountains.
        psfc = iface["p8w"][0]
        sea = sfc["xland"] > 1.5
        cand = np.where(sea, psfc, np.inf)
        flat = np.argsort(cand, axis=None)[:a.cols]
        sel = [(int(k // nx), int(k % nx)) for k in flat]
        sel.sort()

    write(a.prefix, sel, lev, iface, sfc, dx, nz, meta_extra=(
        f"source {pathlib.Path(a.wrfout).name}",
        "forcing_pair zeroed_deliberately",
    ))
    ps = iface["p8w"][0]
    print(f"wrote {len(sel)} columns x {nz} levels -> {a.prefix}")
    print(f"  psfc over selection: {min(ps[j, i] for j, i in sel) / 100:.2f}"
          f" to {max(ps[j, i] for j, i in sel) / 100:.2f} hPa")
    print("  gridcols " + ",".join(str(j * 100000 + i) for j, i in sel[:6])
          + " ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
