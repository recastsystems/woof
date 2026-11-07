"""Table-admitted column microphysics: one launcher for every scheme whose
registry option publishes a ``column_kernel`` consumer row.

A WRF bulk scheme whose driver call is "update theta and the moist species
in place on one column, from the moist_physics_prep_em inputs, and add the
column's surface precipitation" needs nothing scheme-specific on the Python
side beyond the facts its registry row already states: which CUDA unit and
entry point, which species and scheme-declared state fields it updates, in
what order, which surface slots it writes, and which level tiers it is
compiled for.  This module turns that row into the call, so admitting such a
scheme is a registry row plus its kernel, never an adapter module.

The inputs are WRF ``moist_physics_prep_em``
(dyn_em/module_big_step_utilities_em.F:5494-5537) with the selected thermodynamic carrier:
dry theta (converted from moist theta when selected), dry density ``1/alt``, Exner ``(p/p0)**rcp``,
mass-level height ``0.5*(z_at_w(k) + z_at_w(k+1))``, layer depth
``z_at_w(k+1) - z_at_w(k)`` and the terrain height ``ht``.  The kernel's
argument order is fixed by convention so the row need not spell it:

    (theta, <species...>, <state_fields...>,
     rho, pii, p, z, dz8w, ht, <surface_slots...>, dt, nz, ny, nx)

The bracket around the call is the one every scheme uses:
:func:`woof.core.microphysics.save_pre_mp_theta` before and
:func:`woof.core.microphysics.moist_physics_finish` after, so latent heating
reaches the next step's RK tendencies exactly as for the adapter schemes.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

# CuPy, the kernel loader and the state module are imported where a launch
# needs them, so the registry half of this module (the row parsing the
# allocator, the inventories and the dispatcher read) stays importable on a
# card-free host.
DTYPE = np.float32

_COLUMN_TPB = 64
#: The prep inputs a column kernel may read, in kernel argument order.  A
#: row naming anything else is refused: the launcher has no way to build it.
PREP_INPUTS = ("rho", "pii", "p", "z", "dz8w", "ht")
#: Scratch slots of the prep bracket.  Spelled as literals at each call so
#: the preflight scratch census resolves them statically.
_SURFACE_SLOT_NAMES = ("mp_rainnc", "mp_rainncv", "mp_snownc", "mp_snowncv",
                       "mp_graupelnc", "mp_graupelncv", "mp_sr")


@dataclass(frozen=True)
class ColumnKernel:
    """One registry ``column_kernel`` row, validated."""

    selector: int
    module: str
    entry: str
    level_define: str
    level_tiers: tuple[int, ...]
    species: tuple[str, ...]
    state_fields: tuple[str, ...]
    surface_slots: tuple[str, ...]
    inputs: tuple[str, ...]

    def capacity(self, nz: int) -> int:
        """The smallest compiled level tier holding ``nz`` levels.

        Refused above the largest tier: the kernel keeps per-level working
        arrays in a fixed local frame, and a deeper column would index past
        it (the kernel itself also returns untouched rather than write out
        of bounds, which would leave the column silently unadvanced).
        """
        for tier in self.level_tiers:
            if nz <= tier:
                return tier
        raise ValueError(
            f"mp_physics={self.selector} ({self.module}) is compiled for at "
            f"most {self.level_tiers[-1]} mass levels; this domain has {nz}")


def _parse_row(selector: int, row: dict) -> ColumnKernel:
    tiers = tuple(int(t) for t in row["level_tiers"])
    if not tiers or list(tiers) != sorted(set(tiers)) or tiers[0] < 1:
        raise ValueError(f"column_kernel row {selector}: level tiers must be "
                         f"strictly increasing positive integers, got {tiers}")
    inputs = tuple(row.get("inputs", PREP_INPUTS))
    if inputs != PREP_INPUTS:
        raise ValueError(f"column_kernel row {selector}: inputs {inputs} are "
                         f"not the moist_physics_prep_em set {PREP_INPUTS}")
    surface = tuple(row["surface_slots"])
    unknown = [s for s in surface if s not in _SURFACE_SLOT_NAMES]
    if unknown:
        raise ValueError(f"column_kernel row {selector}: unknown surface "
                         f"slots {unknown}")
    species = tuple(row["species"])
    if species[:1] != ("qv",):
        raise ValueError(f"column_kernel row {selector}: species must start "
                         f"with qv, got {species}")
    return ColumnKernel(
        selector=int(selector), module=str(row["module"]),
        entry=str(row["entry"]), level_define=str(row["level_define"]),
        level_tiers=tiers, species=species,
        state_fields=tuple(row.get("state_fields", ())),
        surface_slots=surface, inputs=inputs)


@lru_cache(maxsize=None)
def column_kernel_row(mp_physics: int) -> ColumnKernel | None:
    """The validated ``column_kernel`` row for ``mp_physics``, or None."""
    from woof.physics_registry import consumer_row_for_selector

    row = consumer_row_for_selector("microphysics", "column_kernel",
                                    int(mp_physics))
    if not isinstance(row, dict):
        return None
    return _parse_row(int(mp_physics), row)


def column_kernel_selectors() -> tuple[int, ...]:
    """Every microphysics selector the registry admits through this door."""
    from woof.physics_registry import consumer_rows_by_selector

    rows = consumer_rows_by_selector("microphysics", "column_kernel")
    return tuple(sorted(k for k, v in rows.items() if isinstance(v, dict)))


@lru_cache(maxsize=1)
def transported_state_fields() -> tuple[str, ...]:
    """Scheme state transported by the scalar door, as declared in metadata."""
    from woof.physics_registry import consumer_rows_by_selector
    rows = consumer_rows_by_selector("microphysics", "column_kernel")
    return tuple(dict.fromkeys(
        name for row in rows.values() if isinstance(row, dict)
        for name in row.get("state_fields", ())
        if row.get("state_field_io", {}).get(name, {}).get("advected")))


def _check_volume(name: str, value, shape) -> None:
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if value.dtype != DTYPE:
        raise TypeError(f"{name} must be float32, got {value.dtype}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")


def launch_column_kernel(row: ColumnKernel, theta, species: dict,
                         state_fields: dict, prep: dict, surface: dict,
                         dt: float) -> None:
    """Launch ``row``'s kernel over contiguous FP32 ``(nz, ny, nx)`` fields.

    ``species``/``state_fields``/``surface`` are keyed by the row's names;
    ``prep`` by :data:`PREP_INPUTS` (``ht`` is ``(ny, nx)``).
    """
    shape = theta.shape
    if len(shape) != 3:
        raise ValueError(f"column microphysics fields must be 3-D, got {shape}")
    nz, ny, nx = shape
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    capacity = row.capacity(nz)
    volume = [("theta", theta)]
    volume += [(n, species[n]) for n in row.species]
    volume += [(n, state_fields[n]) for n in row.state_fields]
    volume += [(n, prep[n]) for n in row.inputs if n != "ht"]
    for name, value in volume:
        _check_volume(name, value, shape)
    plane = [("ht", prep["ht"])] + [(n, surface[n]) for n in row.surface_slots]
    for name, value in plane:
        _check_volume(name, value, (ny, nx))
    args = tuple(v for _, v in volume) + tuple(v for _, v in plane)
    from woof.core.kernels import get_kernel, get_kernel_int_defines
    ncol = ny * nx
    blocks = (ncol + _COLUMN_TPB - 1) // _COLUMN_TPB
    kernel = (get_kernel(row.module, row.entry)
              if capacity == row.level_tiers[0] else
              get_kernel_int_defines(row.module, row.entry,
                                     ((row.level_define, capacity),)))
    kernel((blocks,), (_COLUMN_TPB,),
           args + (DTYPE(dt), np.int32(nz), np.int32(ny), np.int32(nx)))


def prepare_inputs(state):
    """WRF moist_physics_prep_em's theta and column inputs for ``state``."""
    import cupy as cp
    from woof.core import constants as c
    nz, ny, nx = state.p.shape
    thb = state.thb if state.thb.ndim == 3 else state.thb[:, None, None]
    phb = state.phb if state.phb.ndim == 3 else state.phb[:, None, None]
    theta = state.scratch((nz, ny, nx), "column_mp_theta")
    rho = state.scratch((nz, ny, nx), "column_mp_rho")
    pii = state.scratch((nz, ny, nx), "column_mp_pii")
    zmass = state.scratch((nz, ny, nx), "column_mp_z")
    dz8w = state.scratch((nz, ny, nx), "column_mp_dz8w")
    z8w = state.scratch((nz + 1, ny, nx), "column_mp_z8w")
    theta[...] = (state.total_theta() if getattr(state, "use_theta_m", 0)
                   else thb + state.thp)
    rho[...] = 1.0 / state.alt
    pii[...] = cp.power(state.p / DTYPE(c.P0), DTYPE(c.RCP))
    z8w[...] = (phb + state.php) / DTYPE(c.G)
    zmass[...] = DTYPE(0.5) * (z8w[:-1] + z8w[1:])
    dz8w[...] = z8w[1:] - z8w[:-1]
    return theta, {"rho": rho, "pii": pii, "p": state.p, "z": zmass,
                   "dz8w": dz8w, "ht": state.ht}


def _surface_slot(state, name: str, shape):
    # Literal slot names, one per branch, for the static scratch census.
    if name == "mp_rainnc":
        return state.scratch(shape, "mp_rainnc")
    if name == "mp_rainncv":
        return state.scratch(shape, "mp_rainncv")
    if name == "mp_snownc":
        return state.scratch(shape, "mp_snownc")
    if name == "mp_snowncv":
        return state.scratch(shape, "mp_snowncv")
    if name == "mp_graupelnc":
        return state.scratch(shape, "mp_graupelnc")
    if name == "mp_graupelncv":
        return state.scratch(shape, "mp_graupelncv")
    if name == "mp_sr":
        return state.scratch(shape, "mp_sr")
    raise ValueError(f"unknown microphysics surface slot {name!r}")


def apply(state, cfg, dt: float, *, refl_10cm_due: bool = False):
    """Prepare WRF fields, run the row's column kernel, finish the heating."""
    row = column_kernel_row(int(cfg.mp_physics))
    if row is None:
        raise ValueError(f"mp_physics={cfg.mp_physics} publishes no "
                         "column_kernel registry row")
    state_names = row.species[1:] + row.state_fields
    missing = [n for n in state_names if getattr(state, n, None) is None]
    if missing:
        raise ValueError(f"mp_physics={cfg.mp_physics} state lacks "
                         + ", ".join(missing))
    nz, ny, nx = state.p.shape
    theta, prep = prepare_inputs(state)
    surface = {name: _surface_slot(state, name, (ny, nx))
               for name in row.surface_slots}
    # SR: WRF zeroes grid%sr whole-field before every microphysics call
    # (solve_em.F:3691); a scheme whose driver call passes no SR leaves it 0.
    sr = _surface_slot(state, "mp_sr", (ny, nx))
    if "mp_sr" not in row.surface_slots:
        sr[...] = 0.0

    from woof.core.microphysics import (MicrophysicsDiagnostics,
                                         moist_physics_finish,
                                         save_pre_mp_theta)
    save_pre_mp_theta(state)
    launch_column_kernel(
        row, theta,
        {name: getattr(state, name) for name in row.species},
        {name: getattr(state, name) for name in row.state_fields},
        prep, surface, dt)
    if refl_10cm_due:
        from woof.core.refl import compute_and_stash_refl_10cm
        refl_t = state.scratch((nz, ny, nx), "refl_t")
        refl_t[...] = theta * prep["pii"]
        compute_and_stash_refl_10cm(state, cfg, refl_t, state.p)
    moist_physics_finish(state, cfg, theta, dt)
    return MicrophysicsDiagnostics(
        rainnc=surface["mp_rainnc"], rainncv=surface["mp_rainncv"], sr=sr,
        snownc=surface.get("mp_snownc"), snowncv=surface.get("mp_snowncv"),
        graupelnc=surface.get("mp_graupelnc"),
        graupelncv=surface.get("mp_graupelncv"))


__all__ = ["ColumnKernel", "PREP_INPUTS", "apply", "column_kernel_row",
           "column_kernel_selectors", "launch_column_kernel",
           "prepare_inputs"]
