#!/usr/bin/env python3
"""Interface instrument for a gray-zone parent chain (P4): measurement only.

Reads paired parent/child wrfout frames and emits one JSON receipt with
every number the registered interface screens consume:

* **Partition handoff** -- the parent's own mixed-layer subgrid TKE
  fraction over the child's footprint, scored the way the acceptance
  ladder scored it: Honnert eq. (7) partition
  (:func:`woof.verify.gray_zone.partition_from_profiles`), h from the
  closure-independent S3-6f bulk-Richardson depth
  (:func:`woof.verify.sase_ref.bulk_richardson_zi`), mixed-layer window
  from :data:`woof.verify.cases.cbl_dry.MIXED_LAYER_WINDOW`, band =
  Honnert envelope at the run's own x widened by the registered sigma
  term (:func:`woof.verify.cases.cbl_dry.sweep_band`).  Requires the
  parent to publish a subgrid TKE variable (``--tke-var``); without one
  the block is recorded as absent, never faked.
* **Rim/interior w-variance ratio** on the child (ringing detector).
* **Spectral overlay statistic** -- the Parseval-normalised radial
  spectrum transplanted from the shipped nested-LES instrument
  (docs/internal/receipts/les/score_nest.py, the two-correction
  version), reduced to the mean |log10(child/parent)| over the shared
  large-scale bins (wavelength >= ``--converge-above-km``).
* **Far-field child statistics** -- mid-CBL var(w), var ratio against
  the parent footprint, max |w| and updraft area fraction, all over the
  fetch-excluded far-field region.

Every geometric and band constant is a CLI argument echoed into the
receipt: this file carries no case, no band and no verdict of its own
beyond the partition band arithmetic it is handed.  Bands and directions
live in the registration document that invokes it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from woof.verify import gray_zone  # noqa: E402
from woof.verify.cases.cbl_dry import (  # noqa: E402
    MIXED_LAYER_WINDOW, sweep_band, window_mean)
from woof.verify.sase_ref import bulk_richardson_zi  # noqa: E402


def _radial_spectrum(field, dx_km):
    """1-D isotropic spectrum whose SUM over bins IS the field variance.

    Transplanted from docs/internal/receipts/les/score_nest.py, the
    instrument proven on the shipped nested case AFTER its two recorded
    normalisation corrections (the N**2 scale and the exact-closure
    rescale that replaced the Hann mean-square approximation; measured
    residual there: worst 3.7e-14%).  The shape is the windowed
    periodogram's; the total is the field's own unwindowed variance, so
    closure is exact by construction and two curves compare as variance
    per bin.
    """
    f = np.asarray(field, dtype=np.float64)
    n = min(f.shape)
    f = f[:n, :n]
    f = f - f.mean()
    target = float((f ** 2).mean())
    win = np.hanning(n)
    fw = f * (win[:, None] * win[None, :])
    fw = fw - fw.mean()          # the window reintroduces a mean
    coef = np.fft.fftshift(np.fft.fft2(fw)) / float(n * n)
    p = np.abs(coef) ** 2
    ky, kx = np.meshgrid(np.fft.fftshift(np.fft.fftfreq(n)) * n,
                         np.fft.fftshift(np.fft.fftfreq(n)) * n,
                         indexing="ij")
    kbin = np.rint(np.sqrt(kx ** 2 + ky ** 2)).astype(int)
    nb = n // 2
    power = np.array([p[kbin == k].sum() for k in range(nb)])
    total = float(power.sum())
    if total > 0.0 and target > 0.0:
        power = power * (target / total)
    k = np.arange(nb)
    lam = n * dx_km / np.maximum(k, 1)
    return lam[1:], power[1:]


def _unstag_w(w):
    return 0.5 * (w[:-1] + w[1:])


def _unstag_u(u):
    return 0.5 * (u[:, :, :-1] + u[:, :, 1:])


def _unstag_v(v):
    return 0.5 * (v[:, :-1, :] + v[:, 1:, :])


def _heights_agl(ds):
    ph = np.asarray(ds.variables["PH"][0], dtype=np.float64)
    phb = np.asarray(ds.variables["PHB"][0], dtype=np.float64)
    z = (ph + phb) / 9.81
    zm = 0.5 * (z[:-1] + z[1:])
    hgt = np.asarray(ds.variables["HGT"][0], dtype=np.float64)
    return zm - hgt[None]


def spectral_overlay_stat(child_field, child_dx_km, parent_field,
                          parent_dx_km, converge_above_km):
    """Mean |log10(child/parent)| over the shared bins above the cutoff.

    Both domains cover the same ground, so their bin wavelengths
    ``extent/k`` coincide bin for bin; the reduction pairs them by k and
    keeps every bin whose wavelength is >= the cutoff.  NaN when no bin
    qualifies or a qualifying bin is empty on either side -- an unusable
    frame must score as unusable.
    """
    lam_c, pc = _radial_spectrum(child_field, child_dx_km)
    lam_p, pp = _radial_spectrum(parent_field, parent_dx_km)
    n = min(len(lam_c), len(lam_p))
    lam_c, pc, lam_p, pp = lam_c[:n], pc[:n], lam_p[:n], pp[:n]
    if not np.allclose(lam_c, lam_p, rtol=1e-9):
        raise ValueError(
            "parent and child spectral bins do not share wavelengths; "
            "the overlay statistic is only defined over the same ground")
    keep = lam_c >= float(converge_above_km)
    if not keep.any():
        return float("nan"), 0
    pc_k, pp_k = pc[keep], pp[keep]
    if (pc_k <= 0.0).any() or (pp_k <= 0.0).any():
        return float("nan"), int(keep.sum())
    return (float(np.mean(np.abs(np.log10(pc_k / pp_k)))),
            int(keep.sum()))


def region_masks(ny, nx, exclude_west, exclude_south, rim_offset,
                 rim_width):
    """(far_field, rim) boolean masks on a (ny, nx) grid.

    * ``rim`` is the ring whose distance from the nearest lateral edge
      lies in ``[rim_offset, rim_offset + rim_width)`` -- the band just
      INSIDE the directly-forced specified/relaxation zone, which is
      where boundary ringing would live (the forced cells themselves are
      blended toward the parent and belong to neither band).
    * ``far_field`` drops the registered fetch strips -- the westernmost
      ``exclude_west`` columns and southernmost ``exclude_south`` rows
      (west = low i, south = low j, the wrfout array convention) -- and
      the whole ``rim_offset + rim_width`` edge band on every side.
    """
    jj, ii = np.mgrid[0:ny, 0:nx]
    dist = np.minimum(np.minimum(ii, nx - 1 - ii),
                      np.minimum(jj, ny - 1 - jj))
    band = int(rim_offset) + int(rim_width)
    rim = (dist >= int(rim_offset)) & (dist < band)
    far = ((ii >= int(exclude_west)) & (jj >= int(exclude_south))
           & (dist >= band))
    return far, rim


def score_frame(ds_parent, ds_child, args) -> dict:
    import netCDF4  # noqa: F401  (imported by caller; kept for clarity)

    ds_parent.set_auto_mask(False)
    ds_child.set_auto_mask(False)
    ratio = int(args.ratio)
    j0, i0 = int(args.j_parent_start) - 1, int(args.i_parent_start) - 1

    w_child = _unstag_w(np.asarray(ds_child.variables["W"][0],
                                   dtype=np.float64))
    nz, cny, cnx = w_child.shape
    foot_ny, foot_nx = cny // ratio, cnx // ratio

    w_par = _unstag_w(np.asarray(ds_parent.variables["W"][0],
                                 dtype=np.float64))
    w_foot = w_par[:, j0:j0 + foot_ny, i0:i0 + foot_nx]

    # Mid-CBL level: the shipped rule -- PBLH from the PARENT footprint
    # (a PBL-off child has none), level nearest half of it on the child's
    # own column heights.
    pblh = float(np.mean(np.asarray(
        ds_parent.variables["PBLH"][0],
        dtype=np.float64)[j0:j0 + foot_ny, i0:i0 + foot_nx]))
    z_child = _heights_agl(ds_child)
    zcol = z_child.mean(axis=(1, 2))
    k = int(np.argmin(np.abs(zcol - 0.5 * max(pblh, 200.0))))
    k = max(1, min(k, nz - 1))

    far, rim = region_masks(
        cny, cnx, args.exclude_west_cells, args.exclude_south_cells,
        args.rim_offset_cells, args.rim_width_cells)
    band = (int(args.rim_offset_cells) + int(args.rim_width_cells))
    far_p, _ = region_masks(
        foot_ny, foot_nx,
        int(args.exclude_west_cells) // ratio,
        int(args.exclude_south_cells) // ratio,
        0, max(1, band // ratio))

    wk = w_child[k]
    wk_foot = w_foot[k]
    var_rim = float(wk[rim].var())
    var_far = float(wk[far].var())
    var_far_parent = float(wk_foot[far_p].var())

    overlay, overlay_bins = spectral_overlay_stat(
        wk, args.child_dx_km, wk_foot, args.parent_dx_km,
        args.converge_above_km)

    row = {
        "pblh_mean_m_parent_footprint": pblh,
        "level_index": k,
        "level_height_m": float(zcol[k]),
        "rim_w_var": var_rim,
        "rim_over_far_field": (var_rim / var_far
                               if var_far > 0.0 else float("nan")),
        "far_field_w_var_child": var_far,
        "far_field_w_var_parent_footprint": var_far_parent,
        "far_field_w_var_ratio": (var_far / var_far_parent
                                  if var_far_parent > 0.0
                                  else float("nan")),
        "far_field_w_max_child": float(np.abs(w_child[:, far].max())),
        "far_field_updraft_area_fraction": float((wk[far] > 0.5).mean()),
        "whole_domain_w_var_child": float(wk.var()),
        "whole_domain_w_var_parent_footprint": float(wk_foot.var()),
        "spectral_overlay_mean_abs_log10": overlay,
        "spectral_overlay_bins": overlay_bins,
    }

    # ---- partition handoff on the parent, over the same footprint ----
    if args.tke_var and args.tke_var in ds_parent.variables:
        sl_j, sl_i = (slice(j0, j0 + foot_ny), slice(i0, i0 + foot_nx))
        u = _unstag_u(np.asarray(ds_parent.variables["U"][0],
                                 dtype=np.float64))[:, sl_j, sl_i]
        v = _unstag_v(np.asarray(ds_parent.variables["V"][0],
                                 dtype=np.float64))[:, sl_j, sl_i]
        wp = w_foot
        theta = np.asarray(ds_parent.variables["T"][0],
                           dtype=np.float64)[:, sl_j, sl_i] + 300.0
        e_sgs = np.asarray(ds_parent.variables[args.tke_var][0],
                           dtype=np.float64)[:, sl_j, sl_i]
        z_par = _heights_agl(ds_parent)[:, sl_j, sl_i]
        part = gray_zone.partition_from_profiles(e_sgs, u, v, wp)
        h = float(np.mean(bulk_richardson_zi(u, v, theta, z_par)))
        z_mean = z_par.mean(axis=(1, 2))
        frac_ml = window_mean(part["subgrid_fraction"], z_mean, h,
                              MIXED_LAYER_WINDOW)
        x = float(args.parent_dx_km) * 1000.0 / h
        env = tuple(float(np.asarray(b))
                    for b in gray_zone.subgrid_tke_envelope(x))
        band = sweep_band(env, args.band_sigma, args.band_n)
        row["partition"] = {
            "tke_var": args.tke_var,
            "h_bulk_richardson_m": h,
            "x_own": x,
            "mixed_layer_subgrid_fraction": frac_ml,
            "eq9": float(np.asarray(gray_zone.subgrid_tke_fraction(x))),
            "envelope": list(env),
            "band_sigma": float(args.band_sigma),
            "band": list(band),
            "in_band": bool(band[0] <= frac_ml <= band[1]),
        }
    else:
        row["partition"] = {
            "tke_var": args.tke_var or None,
            "absent": True,
            "reason": ("parent publishes no subgrid TKE variable under "
                       "this closure; the partition screen applies to "
                       "the scale-aware parent only"),
        }
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--wrfout", required=True, type=Path)
    ap.add_argument("--parent-domain", default="d02")
    ap.add_argument("--child-domain", default="d03")
    ap.add_argument("--i-parent-start", required=True, type=int)
    ap.add_argument("--j-parent-start", required=True, type=int)
    ap.add_argument("--ratio", required=True, type=int)
    ap.add_argument("--parent-dx-km", required=True, type=float)
    ap.add_argument("--child-dx-km", required=True, type=float)
    ap.add_argument("--exclude-west-cells", required=True, type=int)
    ap.add_argument("--exclude-south-cells", required=True, type=int)
    ap.add_argument("--rim-offset-cells", required=True, type=int,
                    help="width of the directly-forced edge zone to "
                         "skip before the rim band starts")
    ap.add_argument("--rim-width-cells", required=True, type=int)
    ap.add_argument("--converge-above-km", required=True, type=float)
    ap.add_argument("--tke-var", default="")
    ap.add_argument("--band-sigma", type=float, default=0.0,
                    help="sigma for the partition band widening "
                         "(sweep_band); from the registration")
    ap.add_argument("--band-n", type=int, default=6,
                    help="n in sweep_band's 2*sigma/sqrt(n); the "
                         "ladder's registered rule uses 6")
    ap.add_argument("--frames", default="",
                    help="comma-separated stamps (wrfout suffix form); "
                         "default: every paired frame")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    import netCDF4 as nc

    stamps = sorted(
        re.sub(rf"^wrfout_{args.child_domain}_", "", p.name)
        for p in args.wrfout.glob(f"wrfout_{args.child_domain}_*"))
    if args.frames:
        wanted = {s.strip() for s in args.frames.split(",") if s.strip()}
        missing = wanted - set(stamps)
        if missing:
            raise SystemExit(f"frames not found: {sorted(missing)}")
        stamps = [s for s in stamps if s in wanted]

    rows = {}
    for stamp in stamps:
        pp = args.wrfout / f"wrfout_{args.parent_domain}_{stamp}"
        cp_ = args.wrfout / f"wrfout_{args.child_domain}_{stamp}"
        if not pp.exists():
            continue
        with nc.Dataset(pp) as a, nc.Dataset(cp_) as b:
            rows[stamp] = score_frame(a, b, args)
        r = rows[stamp]
        part = r["partition"]
        frac = part.get("mixed_layer_subgrid_fraction")
        print(f"{stamp}  z={r['level_height_m']:6.0f}m "
              f"far_var={r['far_field_w_var_child']:7.4f} "
              f"rim/far={r['rim_over_far_field']:6.3f} "
              f"overlay={r['spectral_overlay_mean_abs_log10']:.4f} "
              + (f"partition={frac:.4f} in_band={part['in_band']}"
                 if frac is not None else "partition=absent"))

    receipt = {
        "instrument": "grayzone_p4_score",
        "arguments": {k: (str(v) if isinstance(v, Path) else v)
                      for k, v in vars(args).items()},
        "frames": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=1) + "\n",
                        encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
