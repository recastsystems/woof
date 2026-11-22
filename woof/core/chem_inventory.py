"""Anthropogenic inventory emissions for the GOCART rows (process
``emission.inventory``, DESIGN 6.2).

WRF-Chem's only gridded-inventory path into the GOCART species is the
``emiss_opt == 6`` block of ``emissions_driver`` (chem/emissions_driver.F:
1597-1619): SO2 and sulfate from ``emis_ant`` in mol km-2 hr-1 through
``conv = 4.828e-4/rho_phy*dtstep/(dz8w*60.)``, and BC, OC, PM2.5 and PM10 in
ug m-2 s-1 through ``alt*dtstep/dz8w``, on levels ``kts .. kemit``.  The
kernel (``woof/core/kernels/chem_inventory.cu``) is that block with the
species taken from the table: every emission entry of a row that names this
process and whose ``vertical`` is ``surface`` or ``kemit`` is one term (the
``plumerise`` entries belong to the fire process).  WHICH arithmetic path a
term takes is decided by the row's units -- ``ppmv`` rows are WRF's gases and
sulfate and take ``conv``; ``ug kg-1`` rows take ``alt*dtstep/dz8w`` -- never
by its name.

WRF puts every anthropogenic BC and OC into the hydrophobic tracers (the
block adds ``e_bc`` to ``bc1`` and ``e_oc`` to ``oc1`` only); aging moves them
to the hydrophilic ones.  An entry's ``weight`` is therefore 1 in WRF's
split, and the rows say so.

Frames come from :class:`gpuwm.chem_sources.SourceFrames` already on the
model grid; the entry's ``weight`` turns the source's units into WRF's
``emis_ant`` units before WRF's own expression runs (for a flux in
kg m-2 s-1: 1e9 to ug m-2 s-1, or ``1e6 * 3600 / (M/1000)`` to
mol km-2 hr-1 for a gas of molar mass ``M`` g mol-1).  That conversion is
WOOF-ArWen's and is written in the row, next to the source it converts.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np

KEY = "emission.inventory"
LEDGER = "emitted"
#: The kernel translation units step() launches (priced by the preflight).
KERNEL_MODULES = ("chem_inventory",)
REQUIRES: tuple[str, ...] = ("rho", "alt", "dz8w")
ALLOCATES: tuple = ()

#: The WRF Registry default of ``kemit`` (registry.chem:3780).
WRF_KEMIT_DEFAULT = 9

#: Emission verticals this process owns; ``plumerise`` is the fire process's.
VERTICALS = ("surface", "kemit")


def rows(table):
    return table.rows_for(KEY)


@dataclass(frozen=True)
class InventoryTerm:
    row: str
    source: str
    field: str
    vertical: str
    weight: float     # float32 value, applied in float32 like the rest
    mode: int         # 0: ug m-2 s-1 path; 1: mol km-2 hr-1 path


def arithmetic_mode(units: str) -> int:
    """The WRF emiss_opt=6 path a row of these units takes."""
    if units == "ppmv":
        return 1
    if units == "ug kg-1":
        return 0
    raise ValueError(f"units {units!r} have no emiss_opt=6 arithmetic")


def pack_terms(row_objs) -> list[InventoryTerm]:
    terms = []
    for row in row_objs:
        get = (row.get if isinstance(row, dict)
               else lambda key, default=None, _r=row: getattr(_r, key, default))
        for entry in get("emissions") or ():
            if entry["vertical"] not in VERTICALS:
                continue
            terms.append(InventoryTerm(
                row=get("name"), source=entry["source"], field=entry["field"],
                vertical=entry["vertical"],
                weight=float(np.float32(entry["weight"])),
                mode=arithmetic_mode(get("units"))))
    return terms


def launch_inventory_add(rho_phy, alt, dz8w, fields, emis, terms, *,
                         kemit: int, dtstep: float) -> None:
    """Add every term's emission into its row's field, in place.

    ``fields[i]`` is the float32 ``(nz, ny, nx)`` field term ``i`` adds to and
    ``emis[i]`` its float32 ``(nlev, ny, nx)`` or ``(ny, nx)`` frame (a 2-D
    frame is one level).  Terms add in list order, so two entries into one
    row add in the row's own emission order.
    """
    import cupy as cp

    from woof.core.kernels import get_kernel

    if not terms:
        return
    nz, ny, nx = rho_phy.shape
    for arr in (rho_phy, alt, dz8w, *fields):
        if arr.dtype != cp.float32 or arr.shape != (nz, ny, nx) \
                or not arr.flags.c_contiguous:
            raise ValueError("chem_inventory: fields must be C-contiguous "
                             f"float32 {(nz, ny, nx)}")
    frames = []
    nlev = []
    for e in emis:
        if e.ndim == 2:
            e = e.reshape(1, ny, nx)
        if e.dtype != cp.float32 or e.shape[1:] != (ny, nx) \
                or not e.flags.c_contiguous:
            raise ValueError("chem_inventory: emission frames must be "
                             f"C-contiguous float32 (nlev, {ny}, {nx})")
        frames.append(e)
        nlev.append(int(e.shape[0]))
    ncol = ny * nx
    kern = get_kernel("chem_inventory", "chem_inventory_add")
    threads = 128
    kern(((ncol + threads - 1) // threads,), (threads,),
         (rho_phy, alt, dz8w,
          cp.asarray(np.array([f.data.ptr for f in fields], np.uint64)),
          cp.asarray(np.array([f.data.ptr for f in frames], np.uint64)),
          cp.asarray(np.array(nlev, np.int32)),
          cp.asarray(np.array([t.mode for t in terms], np.int32)),
          cp.asarray(np.array([t.weight for t in terms], np.float32)),
          np.int32(len(terms)), np.int32(kemit), np.float32(dtstep),
          np.int32(nz), np.int64(ncol)))


def _valid_time(ctx):
    start = getattr(ctx.cfg, "start_time", None)
    if start is None:
        raise ValueError("emission.inventory needs cfg.start_time to pick "
                         "an emission frame")
    return start + timedelta(seconds=float(ctx.clock.curr_secs))


def _terms(ctx):
    """The terms of the sources this run enables (``chem_sources``).  An
    inventory row whose source is not enabled emits nothing from it, as WRF
    with ``emiss_opt = 0``: enabling a source is the opt-in."""
    enabled = set(getattr(ctx.table, "enabled_sources", ()))
    return [t for t in pack_terms(rows(ctx.table)) if t.source in enabled]


def init(ctx):
    for term in _terms(ctx):
        if ctx.frames is None:
            raise ValueError(
                f"row {term.row!r} takes inventory emissions from the "
                f"enabled source {term.source!r}, but this run has no source "
                "frames to read them from")


def step(ctx, dt, ktau):
    terms = _terms(ctx)
    if not terms:
        return
    when = _valid_time(ctx)
    fields = [ctx.field(ctx.table.row(t.row)) for t in terms]
    emis = [ctx.frames.at(t.source, t.field, when) for t in terms]
    kemit = int(getattr(ctx.cfg, "kemit", WRF_KEMIT_DEFAULT))
    launch_inventory_add(ctx.met("rho"), ctx.met("alt"), ctx.met("dz8w"),
                         fields, emis, terms, kemit=kemit,
                         dtstep=float(dt))
