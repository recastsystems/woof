"""WRF v4.8.0 dynamic lightning (``dyn_lightning_option=1``) on the GPU.

Native: phys/module_calc_lpi_new.F (LPI2D/LPI3D), phys/module_ltng_pe.F
(charging energy LIGHTNING_PE/NE/NEU) and phys/module_ltng_strokes.F (flash
counts LPOS/LNEG/LNEU), called in that order at the end of
module_microphysics_driver.F after the scheme. Kernels:
woof/core/kernels/dyn_lightning.cu.

calc_dyn_pe carries two values (t_pos, t_neg) from column to column in WRF's
DO j / DO i order, reset at every tile. The launcher reproduces that carry with
a hit pass, an inclusive scan and a rank scatter, so a column reads the latest
preceding hit with no host sync. The carry is per tile, so like native WRF the
result depends on the tile/patch decomposition; a claim states it.
"""
from __future__ import annotations

import numpy as np

DYN_LIGHTNING_DEFAULTS = {
    # Registry/registry.dyn_light (WRF v4.8.0)
    "dyn_lightning_option": 0,
    "coul_pos": 0.25e-4, "coul_neg": 0.25e-4, "coul_neu": 0.25e-4,
    "j_pos": 5.e9, "j_neg": 1.e9, "j_neu": 1.e9,
}
#: The three advected WRF ``scalar``-array species (Registry ikjftb scalar).
DYN_LIGHTNING_SCALARS = ("light_ne", "light_pe", "light_neu")
#: History/restart misc fields (Registry ij / ikj misc).
DYN_LIGHTNING_FIELDS_2D = ("lpos", "lneg", "lneu", "lpi2d")
DYN_LIGHTNING_FIELDS_3D = ("lpi3d",)


def _tiles(nx, ny, tiles):
    """Disjoint in-domain tiles. They need not cover the domain: the driver
    tile leaves out the specified ring, which WRF never charges."""
    tiles = tuple(tuple(int(v) for v in t) for t in (tiles or ((0, nx, 0, ny),)))
    for index, (i0, i1, j0, j1) in enumerate(tiles):
        if not (0 <= i0 < i1 <= nx and 0 <= j0 < j1 <= ny):
            raise ValueError("dynamic lightning tile lies outside the domain")
        for a0, a1, b0, b1 in tiles[:index]:
            if max(i0, a0) < min(i1, a1) and max(j0, b0) < min(j1, b1):
                raise ValueError("dynamic lightning tiles overlap; an overlapped column "
                                 "would be charged and discharged twice in one step")
    return tiles


def launch_dyn_lightning(fields, out, *, dx, dy, dt, coul_pos, coul_neg, coul_neu,
                         j_pos, j_neg, j_neu, tiles=None, block=128):
    """One microphysics-driver call of the three native routines.

    ``fields``: contiguous float32 [nz, ny, nx] th (post-scheme full theta),
    pii, rho, dz8w, qc, qr, qi, qs, qg, qh (zeros where the scheme has none),
    and w [nz+1, ny, nx]. ``out``: light_pe, light_ne, light_neu [nz,ny,nx]
    (updated in place), lpos, lneg, lneu [ny,nx] (accumulated), lpi2d [ny,nx]
    and lpi3d [nz,ny,nx] (overwritten).
    """
    import cupy as cp
    from woof.core.kernels import get_kernel
    nz, ny, nx = fields["th"].shape
    i32, f32 = np.int32, np.float32
    search = get_kernel("dyn_lightning", "dyn_lightning_search")
    rank = get_kernel("dyn_lightning", "dyn_lightning_rank")
    columns = get_kernel("dyn_lightning", "dyn_lightning_columns")
    for i0, i1, j0, j1 in _tiles(nx, ny, tiles):
        ncol = (i1 - i0) * (j1 - j0)
        grid = ((ncol + block - 1) // block,)
        geometry = (i32(nz), i32(ny), i32(nx), i32(i0), i32(i1), i32(j0), i32(j1))
        hit_pos = cp.empty(ncol, cp.int32); val_pos = cp.empty(ncol, cp.float32)
        hit_neg = cp.empty(ncol, cp.int32); val_neg = cp.empty(ncol, cp.float32)
        search(grid, (block,), (fields["th"], fields["pii"], fields["dz8w"],
                                fields["qc"], fields["qr"], fields["qi"], fields["qs"],
                                fields["qg"], *geometry, hit_pos, val_pos, hit_neg, val_neg))
        rank_pos = cp.cumsum(hit_pos, dtype=cp.int32)
        rank_neg = cp.cumsum(hit_neg, dtype=cp.int32)
        by_pos = cp.empty(ncol, cp.float32); by_neg = cp.empty(ncol, cp.float32)
        rank(grid, (block,), (hit_pos, rank_pos, val_pos, hit_neg, rank_neg, val_neg,
                              i32(ncol), by_pos, by_neg))
        columns(grid, (block,), (
            fields["th"], fields["pii"], fields["rho"], fields["dz8w"], fields["w"],
            fields["qc"], fields["qr"], fields["qi"], fields["qs"], fields["qg"], fields["qh"],
            rank_pos, by_pos, rank_neg, by_neg, *geometry,
            f32(dx), f32(dy), f32(dt), f32(coul_pos), f32(coul_neg), f32(coul_neu),
            f32(j_pos), f32(j_neg), f32(j_neu),
            out["light_pe"], out["light_ne"], out["light_neu"],
            out["lpos"], out["lneg"], out["lneu"], out["lpi2d"], out["lpi3d"]))


# ---------------------------------------------------------------- contract --

def active(cfg) -> bool:
    return bool(getattr(cfg, "dyn_lightning_option", 0))


def validate_dyn_lightning(cfg) -> None:
    import math
    option = getattr(cfg, "dyn_lightning_option", 0)
    if isinstance(option, bool) or option not in (0, 1):
        raise ValueError("dyn_lightning_option must be 0 or 1; WRF v4.8.0 registry.dyn_light "
                         "defines only 1 (the Lynn dynamic lightning scheme)")
    if not option:
        return
    if getattr(cfg, "wrf_reference", "4.8.0") == "4.7.1":
        raise ValueError("dyn_lightning_option is a WRF v4.8.0 key; wrf_reference=4.7.1 has no "
                         "dynamic lightning package, so the pinned run would not be WRF 4.7.1")
    if not cfg.moist or cfg.mp_physics == 0:
        raise ValueError("dyn_lightning_option=1 needs moist microphysics: WRF's microphysics "
                         "driver returns before its lightning block when mp_physics=0, so no "
                         "charge, flash or LPI field would ever be produced")
    for key in ("coul_pos", "coul_neg", "coul_neu", "j_pos", "j_neg", "j_neu"):
        value = getattr(cfg, key)
        if isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f"{key} must be finite; a non-finite charging constant or flash "
                             "threshold has no defined lightning output")
    for key in ("j_pos", "j_neg", "j_neu"):
        if getattr(cfg, key) <= 0:
            raise ValueError(f"{key} must be positive: the flash count divides the column "
                             "energy by it (module_ltng_strokes.F flash)")


def drop_default_dyn_lightning(values: dict) -> None:
    """Off defaults keep every earlier checkpoint and preparation identity."""
    if values.get("dyn_lightning_option", 0) == 0:
        for key in DYN_LIGHTNING_DEFAULTS:
            values.pop(key, None)


def allocate(state, cfg, zeros) -> None:
    """Registry state for the package: three advected scalars (with their RK
    time-level copies made by the caller), the flash counts and the LPI."""
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    for name in DYN_LIGHTNING_FIELDS_2D:
        setattr(state, name, zeros(ny, nx))
    for name in DYN_LIGHTNING_FIELDS_3D:
        setattr(state, name, zeros(nz, ny, nx))
    # Post-scheme th_phy, captured where each scheme hands it to the finish.
    state.dyn_lightning_theta = zeros(nz, ny, nx)
    state._dyn_lightning_theta_at = None


# ----------------------------------------------------------------- runtime --

def capture_theta(state, cfg, theta) -> None:
    """The microphysics driver's lightning block reads TH after the scheme,
    before moist_physics_finish (module_microphysics_driver.F)."""
    if not active(cfg):
        return
    state.dyn_lightning_theta[...] = theta
    state._dyn_lightning_theta_at = float(state.elapsed_seconds)


def _tile(state, cfg):
    """The driver's tile: spec_zone ring excluded iff specified .or. nested
    (solve_em.F specified_bdy; module_microphysics_driver.F sz)."""
    _, ny, nx = state.p.shape
    sz = int(cfg.spec_zone) if (getattr(cfg, "specified", False) or getattr(cfg, "nested", False)) else 0
    if 2 * sz >= nx or 2 * sz >= ny:
        raise ValueError("dynamic lightning: the spec_zone ring leaves no interior columns")
    return ((sz, nx - sz, sz, ny - sz),)


def update_dyn_lightning(state, cfg, dt) -> None:
    """Run after the scheme (and its ring restore), before mp_zero_out, with
    moist_physics_prep_em's pre-scheme rho, pii, p and dz8w."""
    if not active(cfg):
        return
    import cupy as cp
    from woof.core import constants as c
    from woof.core.microphysics import prepare_microphysics_exner
    if state._dyn_lightning_theta_at != float(state.elapsed_seconds):
        raise RuntimeError(f"dynamic lightning: mp_physics={cfg.mp_physics} did not hand its "
                           "post-scheme theta to the lightning block this step; charging would "
                           "read a stale temperature")
    nz, ny, nx = state.p.shape
    phb = state.phb if state.phb.ndim == 3 else state.phb[:, None, None]
    z8w = (phb + state.php) / np.float32(c.G)
    pii = cp.empty((nz, ny, nx), cp.float32)
    prepare_microphysics_exner(state.p, pii)
    fields = {"th": state.dyn_lightning_theta, "pii": pii,
              "rho": cp.ascontiguousarray(np.float32(1.0) / state.alt),
              "dz8w": cp.ascontiguousarray(z8w[1:] - z8w[:-1]),
              "w": cp.ascontiguousarray(state.w)}
    zero = None
    for name in ("qc", "qr", "qi", "qs", "qg", "qh"):
        value = getattr(state, name, None)
        if value is None:
            # WRF points an absent species at the zero dummy moist slot.
            if zero is None:
                zero = cp.zeros((nz, ny, nx), cp.float32)
            value = zero
        fields[name] = value
    out = {name: getattr(state, name) for name in DYN_LIGHTNING_SCALARS
           + DYN_LIGHTNING_FIELDS_2D + DYN_LIGHTNING_FIELDS_3D}
    launch_dyn_lightning(fields, out, dx=cfg.dx, dy=cfg.dy, dt=dt,
                         coul_pos=cfg.coul_pos, coul_neg=cfg.coul_neg, coul_neu=cfg.coul_neu,
                         j_pos=cfg.j_pos, j_neg=cfg.j_neg, j_neu=cfg.j_neu,
                         tiles=_tile(state, cfg))
