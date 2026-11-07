"""Low-level updraft helicity: 0-1, 0-2 and 0-3 km AGL running maxima.

The HRRR/RRFS severe-weather fields ``UP_HELI_MAX01``, ``UP_HELI_MAX02`` and
``UP_HELI_MAX03`` (m2 s-2), beside WRF's own 2-5 km ``UP_HELI_MAX``
(:mod:`woof.core.uh_diag`), switched on by ``uh_layers = 1`` (default 0).

Definition: the operational HRRR fork's ``cal_helicity`` (NOAA-EMC/HRRR
40ee6058, module_diffusion_em.F:6068-6816), the same kernels the
``gsd_diagnostics`` package runs (``gsd_helicity.cu``): w times vertical
vorticity on the corner column, integrated over the model layers lying
ENTIRELY inside the band (bottom AGL >= band bottom, top AGL <= band top),
a column whose 8-point-average w is non-positive anywhere in the band
contributing nothing that step; then WRF's 9-point smoother and a running
maximum.  0-2 and 0-3 km are exactly HRRR's bands; 0-1 km is RRFS's and is
the same operator with a 1000 m top.

Window: like UP_HELI_MAX, each value is the maximum since the previous
history write (the history writer resets them in
:func:`woof.core.uh_diag.reset_up_heli_max`), so an hourly history holds
HRRR's hourly-maximum UH.  The sample is the completed step's state, the
same instant ``update_up_heli_max`` folds.

Cost: one vorticity pass over the lowest levels only (those that can reach
3 km AGL, found once from the first state with a 1500 m margin), three
column passes and three smoother folds per step, all on the device, no
host synchronisation after the first step.

Under ``gsd_diagnostics = 1`` the fork already keeps UP_HELI_MAX02/03 (on its
own reset cadence), so this package adds only the 0-1 km band and never
touches the fork's slots.

DIAGNOSTIC ONLY: reads model state, writes only its own scratch slots; with
``uh_layers = 0`` nothing is allocated, launched or written.
"""
from __future__ import annotations

import numpy as np

#: (suffix, bottom m AGL, top m AGL); the slot is ``up_heli_max<suffix>``.
BANDS = (("01", 0.0, 1000.0), ("02", 0.0, 2000.0), ("03", 0.0, 3000.0))

HISTORY = {
    f"up_heli_max{suffix}": (
        f"UP_HELI_MAX{suffix}", "m2 s-2",
        f"Max updraft helicity {bottom:g}-{top:g} m AGL in history interval")
    for suffix, bottom, top in BANDS}
PERSISTENT = tuple(HISTORY)

#: Highest band top, and the margin the vorticity levels keep above it so a
#: column whose levels rise during the run still has its band's top layer.
_TOP_M = max(top for _, _, top in BANDS)
_LEVEL_MARGIN_M = 1500.0

# Work planes: the "uh_diag_" prefix is restart.REBUILT_SCRATCH_PREFIXES's
# per-call work class (the stock diagnostic's own uh_diag_col/uh_diag_use).
_RV_SLOT = "uh_diag_rv_low"
_COL_SLOT = "uh_diag_col_bands"
_USE_SLOT = "uh_diag_use_bands"

_THREADS = 128


def active(cfg) -> bool:
    return int(getattr(cfg, "uh_layers", 0) or 0) == 1


def _owned_bands(gsd: bool):
    # The GSD fork owns 02/03 when it runs; only 0-1 km is added then.
    return tuple(band for band in BANDS if not (gsd and band[0] in ("02", "03")))


def scratch_shapes(cfg):
    """The persistent maxima, allocated with the state so every history
    frame (the first included) carries the same schema."""
    if not active(cfg):
        return {}
    plane = (cfg.ny, cfg.nx)
    return {f"up_heli_max{suffix}": plane
            for suffix, _, _ in _owned_bands(bool(cfg.gsd_diagnostics))}


def owned_slots(state):
    """The maxima this package keeps on ``state`` (empty when off)."""
    existing = getattr(state, "existing_scratch", None)
    if existing is None:
        return ()
    gsd = existing("gsd_step_count") is not None
    return tuple(f"up_heli_max{suffix}" for suffix, _, _ in _owned_bands(gsd)
                 if existing(f"up_heli_max{suffix}") is not None)


def _levels(state, cfg) -> int:
    """Mass levels whose vorticity a band can read, fixed at the first step.

    The lowest w level whose AGL height is above ``_TOP_M + margin`` in EVERY
    column bounds the band layers; one more level is kept for the pair's upper
    value.  Measured once (one small device->host copy); afterwards the
    vorticity work plane's own depth carries the count (not a DomainState
    attribute: every attribute must be restart-classified), so nothing here
    synchronises again.
    """
    held = state.existing_scratch(_RV_SLOT)
    if held is not None:
        return int(held.shape[0]) - _spare(int(cfg.ny) * int(cfg.nx))
    from woof.core.constants import G
    nz = int(cfg.nz)
    g = np.float32(G)
    phb = state.phb
    if phb.ndim == 1:
        phb = phb[:, None, None]
    agl = (state.php + phb) / g - state.ht[None]
    zmin = agl.reshape(nz + 1, -1).min(axis=1)
    zmin = zmin.get() if hasattr(zmin, "get") else np.asarray(zmin)
    above = np.nonzero(zmin > _TOP_M + _LEVEL_MARGIN_M)[0]
    kw_top = int(above[0]) if above.size else nz
    return max(2, min(nz, kw_top + 1))


def _spare(n: int) -> int:
    # gsd_vorticity's guard is the full nz, so its last block may write up to
    # _THREADS - 1 values past levels*n: one spare level (more for tiny
    # planes) keeps those writes inside the buffer.
    return 1 + (_THREADS // n if n < _THREADS else 0)


def update(state, cfg) -> None:
    """Fold this step's 0-1/0-2/0-3 km updraft helicity into the maxima."""
    if not active(cfg):
        return
    slots = owned_slots(state)
    if not slots:
        return
    nz, ny, nx = int(cfg.nz), int(cfg.ny), int(cfg.nx)
    rdx = np.float32(1.0) / np.float32(cfg.dx)
    rdy = np.float32(1.0) / np.float32(cfg.dy)
    bands = [band for band in BANDS if f"up_heli_max{band[0]}" in slots]

    if isinstance(state.u, np.ndarray):
        from woof.core.uh_diag import (
            _to_host, apply_uh_smoother_max_np, cal_helicity_uh_columns_np)
        for suffix, bottom, top in bands:
            uh, use = cal_helicity_uh_columns_np(
                state.u, state.v, state.w, state.php, state.phb,
                state.msfu, state.msfv, state.ht,
                _to_host(state.dn), _to_host(state.dnw),
                _to_host(state.fnm), _to_host(state.fnp),
                state.cf1, state.cf2, state.cf3, rdx, rdy,
                bottom_m=bottom, top_m=top)
            apply_uh_smoother_max_np(uh, use, state.existing_scratch(f"up_heli_max{suffix}"))
        return

    from woof.core.kernels import get_kernel

    levels = _levels(state, cfg)
    n = ny * nx
    rv = state.scratch((levels + _spare(n), ny, nx), _RV_SLOT)
    cols = state.scratch((len(BANDS), ny, nx), _COL_SLOT)
    uses = state.scratch((len(BANDS), ny, nx), _USE_SLOT)
    f, i = np.float32, np.int32
    block = (_THREADS,)
    phb3d = i(state.phb.ndim == 3)
    get_kernel("gsd_helicity", "gsd_vorticity")(
        ((levels * n + _THREADS - 1) // _THREADS,), block,
        (state.u, state.v, state.php, state.phb, phb3d,
         state.msfu, state.msfv, state.dn, state.dnw, state.fnm, state.fnp,
         f(state.cf1), f(state.cf2), f(state.cf3), rdx, rdy,
         rv, i(nz), i(ny), i(nx)))
    grid = ((n + _THREADS - 1) // _THREADS,)
    smooth = get_kernel("uh_diag", "uh_smooth_max")
    for index, (suffix, bottom, top) in enumerate(bands):
        column, use = cols[index], uses[index]
        # nz = levels: the loop stops below the cached top; a band layer at
        # levels-1 cannot occur (its bottom is above the band top everywhere).
        get_kernel("gsd_helicity", "gsd_uh_columns")(
            grid, block,
            (state.w, state.php, state.phb, phb3d, state.ht, rv,
             column, use, f(bottom), f(top), i(levels), i(ny), i(nx)))
        target = state.existing_scratch(f"up_heli_max{suffix}")
        smooth(grid, block, (column, use, target, i(ny), i(nx)))
        # WRF's boundary copies (:7541-7557), as for UP_HELI_MAX.
        target[:, 0] = target[:, 1]
        target[0, :] = target[1, :]


def reset(state) -> None:
    """Zero this package's maxima after their history frame is written."""
    from woof.core.uh_diag import _zero_domain_slot
    for slot in owned_slots(state):
        _zero_domain_slot(state, slot)
