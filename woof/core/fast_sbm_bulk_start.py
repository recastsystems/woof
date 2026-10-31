"""Bulk-to-bin start for FAST_SBM (mp_physics=30) from a bulk analysis.

A real-data start (HRRR, RAP, GFS ...) carries bulk condensate mixing ratios
(qc, qr, qi, qs, qg) and no spectra. FAST_SBM integrates 99 hydrometeor bins
(ff1 liquid, ff5 snow, ff6 dense ice), so the first call used to refuse such a
start: dropping the bulk water would silently lose the analysis' cloud.

This module distributes each category's mass over the scheme's own mass grid
(``xl`` from the packaged FAST_SBM tables; the snow and dense-ice grids are the
same grid) with a fixed, documented size distribution, so every gram of the
analysis' condensate is carried into the bins exactly (per cell, to float32
rounding) and nothing else in the state changes:

- cloud water ``qc`` -> ff1, lognormal in diameter, N = 100 cm-3 over land and
  50 cm-3 over water, geometric sigma 1.35;
- rain ``qr`` -> ff1, Marshall-Palmer exponential, N0 = 8e6 m-4;
- cloud ice ``qi`` -> ff5, lognormal, N = 100 per litre, sigma 1.5 (FAST_SBM
  folds crystals into snow every step);
- snow ``qs`` -> ff5, exponential in melted-equivalent diameter, N0 = 3e6 m-4;
- graupel ``qg`` -> ff6, exponential in melted-equivalent diameter, N0 = 4e6 m-4.

Sizes are melted-equivalent diameters of the bin masses (``xl`` in grams), and
bin edges sit at the geometric midpoints of the doubling mass grid. A
distribution whose mass would fall outside the grid is renormalized onto it;
the stored value of a bin is its mass mixing ratio (kg/kg), which is the state
convention (woof/core/kernels/fast_sbm_driver.cuh fsbm_water). Cells that
already carry any hydrometeor bin are left alone, so a restart or a child
interpolated from a spectral parent is never touched.

The spin-up this start implies is the scheme's own: the shapes above are a
reasonable first guess, and the spectra relax to FAST_SBM's own within the
first tens of minutes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

_TABLES = Path(__file__).resolve().parents[1] / "data" / "fast_sbm" / "tables-float64.npz"
NBIN = 33
RHO_W = 1000.0  # kg m-3

#: (bulk name, bin plane offset, kind, parameter) in the order applied.
CATEGORIES = (
    ("qc", 0, "lognormal-cloud", None),
    ("qr", 0, "exponential", 8.0e6),
    ("qi", 33, "lognormal", (1.0e5, 1.5)),
    ("qs", 33, "exponential", 3.0e6),
    ("qg", 66, "exponential", 4.0e6),
)
CLOUD_N_LAND = 1.0e8   # m-3 (100 cm-3)
CLOUD_N_WATER = 5.0e7  # m-3 (50 cm-3)
CLOUD_SIGMA = 1.35
THRESHOLD = 1.0e-13    # the refusal's own "carries condensate" threshold


def mass_grid_kg() -> np.ndarray:
    """FAST_SBM bin masses (kg), the packaged ``xl`` table (grams) / 1000."""
    from woof.tables import table_path
    with np.load(table_path("fast-sbm", _TABLES)) as t:
        return np.asarray(t["xl"], dtype=np.float64) * 1.0e-3


def diameter_edges_m(x_kg: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Melted-equivalent diameters (m): bin centres and lower/upper edges."""
    ratio = x_kg[1:] / x_kg[:-1]
    r = float(np.exp(np.mean(np.log(ratio))))
    lo, hi = x_kg / np.sqrt(r), x_kg * np.sqrt(r)
    d = lambda m: np.cbrt(6.0 * m / (np.pi * RHO_W))
    return d(x_kg), d(lo), d(hi)


def _exponential_weights(xp, q, rho, n0, x, dlo, dhi):
    """Mass fraction per bin for N(D) = n0 exp(-lam D), lam from the mass q."""
    lam = xp.power(np.pi * RHO_W * n0 / xp.maximum(rho * q, 1e-30), 0.25)  # (cells,)
    lam = lam[None, :]
    number = (n0 / lam) * (xp.exp(-lam * xp.asarray(dlo)[:, None]) - xp.exp(-lam * xp.asarray(dhi)[:, None]))
    return number * xp.asarray(x)[:, None]


def _lognormal_weights(xp, erf, q, rho, n, sigma, x, dlo, dhi):
    """Mass fraction per bin for a lognormal number distribution of total n."""
    ls = float(np.log(sigma))
    dg = xp.cbrt(6.0 * rho * q / (np.pi * RHO_W * n * np.exp(4.5 * ls * ls)))
    lg = xp.log(xp.maximum(dg, 1e-12))[None, :]
    a = (xp.log(xp.asarray(dlo))[:, None] - lg) / (np.sqrt(2.0) * ls)
    b = (xp.log(xp.asarray(dhi))[:, None] - lg) / (np.sqrt(2.0) * ls)
    number = 0.5 * n * (erf(b) - erf(a))
    return number * xp.asarray(x)[:, None]


def bins_from_bulk(xp, bulk: dict, rho, xland, *, erf=None):
    """Return the (99, ...) mass mixing ratio planes for the given bulk fields.

    ``bulk`` maps qc/qr/qi/qs/qg to arrays of one shape; ``rho`` is air
    density (kg m-3) of that shape; ``xland`` is 1 over land and 2 over water,
    broadcastable to it. Every category's planes sum to its bulk value.
    """
    if erf is None:
        if xp is np:
            from scipy.special import erf  # noqa: PLC0415
        else:
            from cupyx.scipy.special import erf  # noqa: PLC0415
    x = mass_grid_kg()
    _, dlo, dhi = diameter_edges_m(x)
    shape = rho.shape
    out = xp.zeros((99,) + tuple(shape), dtype=xp.float64)
    rho64 = rho.astype(xp.float64).ravel()
    land = (xp.broadcast_to(xland, shape).astype(xp.float64).ravel() < 1.5)
    for name, offset, kind, param in CATEGORIES:
        q = bulk[name].astype(xp.float64).ravel()
        live = q > THRESHOLD
        if not bool(xp.any(live)):
            continue
        idx = xp.nonzero(live)[0]
        qv, rv = q[idx], rho64[idx]
        if kind == "exponential":
            w = _exponential_weights(xp, qv, rv, param, x, dlo, dhi)
        elif kind == "lognormal":
            w = _lognormal_weights(xp, erf, qv, rv, param[0], param[1], x, dlo, dhi)
        else:
            n = xp.where(land[idx], CLOUD_N_LAND, CLOUD_N_WATER)
            w = _lognormal_weights(xp, erf, qv, rv, n, CLOUD_SIGMA, x, dlo, dhi)
        total = w.sum(axis=0)
        # A distribution centred off the grid (vanishing total) goes whole
        # into the nearest end bin, so no mass is ever dropped.
        bad = ~(total > 0) | ~xp.isfinite(total)
        if bool(xp.any(bad)):
            w[:, bad] = 0.0
            dmean = xp.cbrt(6.0 * rv[bad] * qv[bad] / (np.pi * RHO_W * 1.0e6))
            end = xp.where(dmean > float(np.sqrt(dlo[0] * dhi[-1])), NBIN - 1, 0)
            w[end, xp.nonzero(bad)[0]] = 1.0
            total = w.sum(axis=0)
        frac = w / total[None, :]
        plane = out[offset:offset + NBIN].reshape(NBIN, -1)
        plane[:, idx] += frac * qv[None, :]
    return out


def start_bins_from_bulk(state, xland) -> dict:
    """Fill empty hydrometeor bins of ``state`` from its bulk condensate.

    Only cells whose 99 hydrometeor bins are all zero and whose bulk
    condensate exceeds the refusal threshold are written. Returns a small
    receipt: cells filled and the per-category domain mass carried
    (kg/kg summed over cells) before and after, for the run log.
    """
    import cupy as cp
    bins = state.fast_sbm_bins
    names = ("qc", "qr", "qi", "qs", "qg")
    bulk = {n: getattr(state, n) for n in names}
    total = sum(bulk[n] for n in names)
    empty = cp.sum(bins[:99], axis=0) == 0
    target = empty & (total > np.float32(THRESHOLD))
    nfill = int(cp.count_nonzero(target))
    receipt = {"cells_filled": nfill}
    if nfill == 0:
        return receipt
    rho = (1.0 / state.alt).astype(cp.float64)
    sel = {n: cp.where(target, bulk[n], 0).astype(cp.float64) for n in names}
    planes = bins_from_bulk(cp, sel, rho, cp.asarray(xland)[None, :, :])
    for n, off in (("liquid", 0), ("snow", 33), ("dense", 66)):
        receipt[f"bin_mass_{n}"] = float(planes[off:off + NBIN].sum())
    receipt["bulk_mass"] = {n: float(sel[n].sum()) for n in names}
    bins[:99] = cp.where(target[None], planes.astype(bins.dtype), bins[:99])
    return receipt
