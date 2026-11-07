"""REAL initial conditions into a pinned host store, one row slab at a time.

Every gate in this prototype until now has run on a seeded synthetic state --
``harness.make_state`` draws from ``numpy.random.default_rng`` and calls it
weather.  That proves the TRANSPORT and it proves the halo, but a forecast you
cannot initialise is not a forecast, and nothing in the streaming lane had ever
seen a GRIB file.  This module is the route from real data to the pinned host
store, and the constraint it exists to satisfy is the one that makes the whole
project non-trivial:

    THE DOMAIN IS LARGER THAN VRAM, SO THE INGEST MAY NEVER MATERIALISE A
    FULL-DOMAIN ``DomainState`` ON THE DEVICE.

``woof``'s production ingest does exactly that, on purpose, and is right to:
every existing caller runs a domain that fits.  Read the path before changing
it.


WHAT ARWEN DOES TODAY (measured, not inferred)
----------------------------------------------
Two front doors reach the same three stages.

``woof.runtime.prepare_real_case``
    the production one.  Builds the geogrid statics
    (``runtime._cached_static_build`` -> ``woof.static.build.build_static``),
    decodes each forcing time, interpolates it, calls ``initialize_real``,
    accumulates the perimeter frames into ``StateBoundaryFrames``, builds the
    soil/landuse state and constructs a ``PhysicsDriver``.

``woof.verify.cases.real74_d01.build_forcing``
    the frozen dynamics-only one.  Same three stages with the statics replaced
    by the bundle's own ``geo_em`` HGT_M/MAPFAC/F/E/SIN/COSALPHA, no soil and
    no physics driver.  This module reproduces THAT one, because it is the
    smallest path that is still real data end to end.

The three stages, and what each one costs on a domain of ``ny x nx``:

1.  DECODE.  ``woof.ingest.grib.cached_era5_snapshots(grib, vtable)`` shells
    out to the packaged Rust bridge and returns one ``Era5Snapshot`` per valid
    time: host float64, shape ``(nlevel, nlat, nlon)`` on the SOURCE grid.
    MEASURED on the April-1974 bundle: 37 levels x 121 x 201, 22 fields, three
    valid times, from a 29 MB GRIB.  **The source is bounded by the SOURCE
    grid and does not grow with the target domain at all** -- this stage is
    already out-of-core-safe and needs nothing.

2.  HORIZONTAL.  ``woof.ingest.horiz.interpolate_era5_to_lambert(snapshot,
    grid)`` evaluates ``grid.latlon_mass()/latlon_u()/latlon_v()``, builds one
    interpolation plan per staggering and applies it to every field.  Output
    is CuPy FP32 at TARGET shape: ``(nlevel, ny, nx)`` and the two staggered
    wind variants.  **This is O(target) and it is on the device.**

3.  VERTICAL + STATE.  ``woof.ingest.real.initialize_real(met, cfg, coord,
    terrain)`` does the WRF-real column work in float64 on the host, allocates
    ``DomainState(cfg)`` at FULL domain shape and uploads.  **This is O(target)
    twice over -- once in the host float64 intermediates, once in the device
    state.**

Stage 3 is the wall.  At ``real74_d01``'s own 250x200x49 the device high-water
of stages 2+3 is 0.68 GiB; it is linear in ``ny*nx``, so the same code at
4000x4000 asks for 217.5 GiB of VRAM before a single step runs.


WHAT THIS MODULE DOES INSTEAD
-----------------------------
Stages 2 and 3 are run on a ROW SLAB of the target domain and the slab's state
is written straight into the pinned host store's window.  Nothing full-domain
is ever allocated anywhere except the pinned store itself, which is the point
of the store.  Peak device residency becomes a function of the slab height,
which the caller chooses, and NOT of the domain.

Three facts make that exact rather than approximate, and each one is a way
this could have been faked:

*   THE WINDOW GRID IS THE PARENT'S OWN ARITHMETIC.  :func:`window_grid` is
    ``ProjectedGrid.translated(i0, j0)`` -- which delegates ``ij_to_latlon(x,
    y)`` to the reference grid at ``(x + i0, y + j0)`` -- with the extents
    reduced to the window.  A slab therefore evaluates the projection at the
    PARENT's indices through the PARENT's floats, so ``latlon_mass``,
    ``mapfac_u``, ``mapfac_v``, ``coriolis_m`` and ``rotation_m`` come out
    bit-identical to the parent's slice.  MEASURED: all ten arrays
    ``np.array_equal`` against the parent slice, in both axes.  Rebuilding a
    ``LambertGrid`` from the slab's own extents does NOT do this --
    ``projection.py:122-123`` defaults the reference point to the domain
    centre, so a rebuilt slab believes it sits where the whole domain sits.
    That is the same bug ``driver.geography_inventory`` exists to prevent, one
    stage earlier.

*   THE SLAB CARRIES A ONE-ROW HALO IN Y, AND ONLY BECAUSE OF ``VV``.
    ``initialize_real`` is column-local everywhere except two lines:
    ``real.py:1930`` ``_pressure_at_u`` averages the mass-point dry pressure
    onto x faces and ``:1939`` ``_pressure_at_v`` averages it onto y faces
    (one-sided copies at the domain edge).  A row slab spans the full x
    extent, so ``_pressure_at_u`` never crosses a slab seam; ``_pressure_at_v``
    does, and the v-face target pressure at the slab's first row needs the
    mass row BELOW it.  :data:`SLAB_Y_HALO` is that one row.  It is not a
    guess and it is not a safety margin: ``y_halo=0`` is a negative control in
    :mod:`tilestream.test_realdata` and it fails on ``state/v`` alone, at
    exactly one row per interior seam.

*   THE WRITE-BACK IS STAGGERING-AWARE.  A mass field's slab owns global rows
    ``[j0, j1)``; a v-staggered field's slab owns ``[j0, j1)`` too, except the
    LAST slab, which additionally owns the closing face ``ny``.  Getting that
    wrong writes real numbers into the wrong rows and leaves no trace -- the
    field is still finite, still smooth, still plausible.
    :func:`window_rows` is the single place that decides it and
    :mod:`tilestream.test_realdata` drives it wrong three ways on purpose.


WHAT A SLAB IS NOT ALLOWED TO SEE
---------------------------------
``initialize_real`` raises on a non-positive dry column mass and on a
non-monotonic hybrid pressure column, both per column, so a slab reaches the
same verdict as the domain for the columns it holds -- a slab ingest cannot
accept a domain the monolithic one refuses, or the other way round.

One WART, recorded rather than hidden: those refusals name a ROW, and
``real.py``'s ``hybrid_column_ordering_refusal(..., row_offset=)`` seam is
reachable only from ``_make_real_base``'s own threading, not from
``initialize_real``'s argument list.  A slab therefore reports the row index
WITHIN THE SLAB.  Nothing computes wrongly because of it and no state is
affected; a reader chasing a refusal has to add ``slab.g0`` by hand.  Fixing
it means threading a ``row_offset`` through ``initialize_real``, which is a
change to production ingest and not to this lane.


THE PERIODIC CLOSURE, stated plainly because it is the one thing here that
edits real data
--------------------------------------------------------------------------
The bit-exact tiled lane is periodic (``spec.TileSpec._axis_gather`` reduces
every window mod ``nx``), and a periodic domain is only self-consistent when
the closing faces alias the opening ones: ``u[..., nx] == u[..., 0]`` and
``v[..., ny, :] == v[..., 0, :]``.  ``woof.verify.npref`` enforces exactly
that on its seeded states; real ERA5 winds on a CONUS Lambert grid do not
satisfy it, because the domain is not periodic.  :func:`periodic_closure`
imposes it, and the streaming comparison applies it to BOTH sides -- it cannot
manufacture agreement, it only removes an alias ambiguity that a periodic
gate would otherwise report as a transport bug.  It is the only edit this
module makes to ingested values, and a real forecast with specified lateral
boundaries would not make it.


WHAT A LARGER-THAN-VRAM DOMAIN NEEDS THAT A SMALLER ONE DOES NOT
----------------------------------------------------------------
The audit, measured on real74 d01 (250x200x49 = 50 000 mass cells) so every
figure is per mass cell and multiplies straight out to any size.

=============================================  ===========  =================
stage                                           B/mass cell  scales with
=============================================  ===========  =================
ERA5 decode, one valid time (host)                  786.06   the SOURCE grid
geogrid statics ``build_static`` (host f64)         776.00   the domain
ingest DEVICE HIGH-WATER, monolithic             14 593.89   the domain
ingest DEVICE HIGH-WATER, slabs of 64 rows        4 848.40   the SLAB
ingest DEVICE HIGH-WATER, slabs of 8 rows         1 968.20   the SLAB
ingest DEVICE HIGH-WATER, slabs of 1 row          1 626.80   the SLAB
``RealInitResult`` retained host float64          3 200.00   the domain
carrier state on the device (13 fields)           2 365.76   the domain
geography on the device (13 fields)                 824.04   the domain
lateral boundary frames, all three times            533.99   the PERIMETER
=============================================  ===========  =================

Read that table as five separate statements.

1.  DECODE NEEDS NOTHING.  786 B/cell is 37.5 MiB divided by THIS domain's
    cells; the snapshot is 37 levels x 121 x 201 on the ERA5 grid and does
    not grow with the target at all.  A 4000x4000 target decodes the same
    37.5 MiB.  Nothing to do.

2.  THE INGEST IS THE WALL, AND SLABS MOVE IT.  14.6 kB/cell of VRAM at
    monolithic is 217.5 GiB at 4000x4000 -- before a single step.  Cut into
    single rows it is 1.63 kB/cell, and even that is not per cell: it is the
    ~76 MiB fixed cost of the CUDA context, the interpolation plans and the
    source grid, which does not grow with the domain either.  The DOMAIN-
    dependent part of the slab path is ``O(rows_per_slab * nx)`` and the
    caller sets ``rows_per_slab``.  That is the whole result.

3.  THE HOST SIDE HAS ITS OWN WALL AND IT IS NOT SMALL.  ``RealInitResult``
    RETAINS 3.2 kB/cell of float64 after ``initialize_real`` returns --
    ``dry_pressure``, ``total_pressure``, ``total_geopotential``,
    ``total_specific_volume`` at 392-400 B/cell each, plus the whole
    ``BaseState`` again at 1 592 B/cell.  A caller that keeps the result (and
    ``prepare_real_case`` keeps ``initial_result`` for the entire run) is
    holding 47.7 GiB at 4000x4000 in ORDINARY host memory, on top of the
    pinned store.  The slab loop drops each slab's copy before building the
    next, which is why it never appears in the high-water column.

4.  THE BOUNDARIES ARE ALREADY FINE.  ``ingest.lateral_bc.
    StateBoundaryFrames`` keeps only the four ``spec_bdy_width`` perimeter
    strips per forcing time.  MEASURED, all three real74 forcing times:
    26 699 520 B, which reads as 534 B/cell here but is really
    **121.09 B per BOUNDARY cell** over ``2 * (nx + ny) * spec_bdy_width *
    nz`` of them -- so it costs ``O(perimeter)`` and its share of the domain
    SHRINKS as the domain grows: 534 B/cell at 250x200 becomes 29.7 B/cell
    at 4000x4000, where the whole three-time boundary set is 0.44 GiB.  What
    it still WANTS is a whole state to slice from; a slab world hands it each
    slab's contribution to the west/east strips and the first/last slab's
    south/north strips, which is arithmetic, not a redesign.

5.  WHAT STILL ASSUMES A RESIDENT STATE, named so nobody has to re-find it:

    ``runtime._cached_static_build`` -> ``static.build.build_static``
        the geogrid statics: 14 arrays, 776 B/cell, ALL float64 and all on
        the host.  Three of them are twelve months deep (``GREENFRAC``,
        ``LAI12M``, ``ALBEDO12M``, 96 B/cell each) and three are category
        fractions (``LANDUSEF`` 21 deep at 168 B/cell, ``SOILCTOP`` and
        ``SOILCBOT`` 16 deep at 128 B/cell each) -- so 92% of the bill is six
        arrays, and at 4000x4000 the set is 11.6 GiB of host RAM.  It is
        driven entirely by the grid, so :func:`window_grid` slabs it the same
        way this module slabs the analysis -- but ``_cached_static_build``'s
        ``lru_cache`` is keyed on the grid, so a slabbed caller would hold one
        cache entry per SLAB, which is a cache that grows with the domain.
    ``core.physics.initialize_physics``
        builds the whole ``PhysicsDriver`` on the device: 88 to 162
        surface/soil/snow arrays plus the scheme tables, 229 B/cell of
        carriers at the real74 rung.  A physics-on out-of-core ingest has to
        build it per slab and scatter ``fields/*`` into the store exactly as
        this module scatters ``state/*``; the tile buffers already own their
        own driver, so nothing about the STEP path needs it resident.
    ``core.landuse.initialize_landuse(..., cen_lat=...)``
        takes a DOMAIN scalar to pick the LANDUSE.TBL season and flips it
        south of the equator (landuse.py:327, :389).  A slab must be handed
        the domain's ``cen_lat``, never its own, or a domain straddling the
        equator gets summer tables in one slab and winter in the next.
        ``driver.py`` records the same trap one stage later, for tiles.
    ``ingest.real.hydrostatic_residual`` and ``runtime.write_case_output``
        full-domain readers.  Diagnostics and output, not ingest.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np


__all__ = [
    "REAL74_START",
    "SLAB_Y_HALO",
    "Real74Bundle",
    "RowSlab",
    "SlabIngestError",
    "allocate_domain_arrays",
    "device_high_water",
    "find_real74_bundle",
    "ingest_slab_state",
    "make_real_tile_state",
    "monolithic_ingest",
    "periodic_closure",
    "plan_row_slabs",
    "real74_geography",
    "rebuilt_grid",
    "stream_real_into_store",
    "window_grid",
    "window_rows",
    "write_window",
]


class SlabIngestError(RuntimeError):
    """A slab ingest could not be set up consistently."""


#: Mass rows a slab must carry BELOW its own first row, and the reason.
#:
#: ``woof.ingest.real._pressure_at_v`` (real.py:1939) forms the v-face target
#: dry pressure as ``0.5 * (pd[j-1] + pd[j])`` for every interior face and
#: copies the neighbouring mass value at the two domain edges.  A slab starting
#: at global row ``j0 > 0`` therefore needs mass row ``j0 - 1`` to reproduce the
#: domain's own value at v face ``j0``; without it the slab takes its OWN edge
#: branch and the face gets ``pd[j0]`` instead of the average.
#:
#: One row, not two: no other operator in ``initialize_real`` reads across a
#: row.  ``_pressure_at_u`` averages in X only and a row slab is full-width, so
#: it never crosses a seam.  MEASURED -- ``y_halo=0`` differs on ``state/v``
#: and on nothing else, in exactly ``nslabs - 1`` rows.
SLAB_Y_HALO = 1


# --------------------------------------------------------------------------
# the bundle
# --------------------------------------------------------------------------

#: Where the April-1974 reference bundle lives, in the order tried.  The
#: first environment variable is ``woof.verify.cases.real74_d01``'s own
#: name, so a machine already configured for the acceptance case needs
#: nothing new.  ``..._LOCAL`` is tried FIRST because the canonical bundle
#: is often a mounted Windows drive, and re-reading a 29 MB GRIB across 9p
#: per process is minutes rather than seconds: point it at a copy on the
#: box's own filesystem and nothing else changes.
_BUNDLE_CANDIDATES = (
    lambda: os.environ.get("WOOF_REAL74_REFERENCE_BUNDLE_LOCAL"),
    lambda: os.environ.get("WOOF_REAL74_REFERENCE_BUNDLE"),
    lambda: str(Path.home() / "real74-bundle-local"),
    lambda: str(Path.home() / "Downloads" / "WRF_1974_MP55_reference_bundle"),
)

#: The one forcing time this module initialises from.  The bundle carries
#: 12/18/00Z; the later two are lateral-boundary material, which the periodic
#: streaming lane does not consume (see the module docstring).
REAL74_START = datetime(1974, 4, 3, 12)


@dataclass(frozen=True)
class Real74Bundle:
    """The four real inputs of the April-1974 d01 case, resolved on disk.

    ``namelist_wps`` is the authority for the Lambert grid (the projection
    parameters are NOT on ``RunConfig`` -- searched, and it has none of
    ref_lat/ref_lon/truelat/stand_lon/known_x/known_y), ``geo_em`` for the
    model terrain and the map/Coriolis fields, ``met_em`` for ``SOILHGT``
    (the retained ERA5 GRIB omits invariant geopotential -- see
    ``woof/ingest/README.md``), and the GRIB plus its Vtable for the forcing
    itself.
    """

    root: Path
    grib: Path
    vtable: Path
    geo_em: Path
    met_em: Path
    namelist_wps: Path

    def check(self) -> "Real74Bundle":
        missing = [str(p) for p in (self.grib, self.vtable, self.geo_em,
                                    self.met_em, self.namelist_wps)
                   if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                "real74 bundle is incomplete; missing " + ", ".join(missing))
        return self


def find_real74_bundle(root=None) -> Real74Bundle:
    """Resolve the reference bundle, or raise naming everywhere it looked."""
    tried: list[str] = []
    candidates = ([str(root)] if root is not None
                  else [c() for c in _BUNDLE_CANDIDATES])
    for candidate in candidates:
        if not candidate:
            continue
        tried.append(candidate)
        base = Path(candidate)
        bundle = Real74Bundle(
            root=base,
            grib=base / "era5_grib" / "era5_19740403.grb",
            vtable=base / "era5_grib" / "Vtable.ERA5_CDO",
            geo_em=base / "geo_em" / "geo_em.d01.nc",
            met_em=(base / "met_em"
                    / "met_em.d01.1974-04-03_12_00_00.nc"),
            namelist_wps=base / "namelists" / "namelist.wps",
        )
        if bundle.grib.is_file() and bundle.namelist_wps.is_file():
            return bundle.check()
    raise FileNotFoundError(
        "no real74 reference bundle found; set WOOF_REAL74_REFERENCE_BUNDLE. "
        "Tried: " + ", ".join(tried))


# --------------------------------------------------------------------------
# window grids
# --------------------------------------------------------------------------

def window_grid(grid, i0: int, j0: int, nx: int, ny: int):
    """The sub-rectangle ``[j0, j0+ny) x [i0, i0+nx)`` of ``grid``, exactly.

    Built from ``ProjectedGrid.translated`` so the transforms DELEGATE to
    ``grid`` at ``(x + i0, y + j0)``: every cell is evaluated at the parent's
    own index through the parent's own floats, which is what makes the window
    bit-identical rather than merely close.  ``translated`` keeps the parent's
    extents (it exists for moving nests, which do not shrink), so the extents
    are reduced here and ``cen_lat``/``cen_lon`` -- metadata, read by
    ``woof/core/landuse.py:327`` for the LANDUSE.TBL season and by the wrfout
    header -- are re-derived through the delegated transform.

    MEASURED against ``real74_d01``'s 250x200 Lambert grid, window
    ``[37, 62) x [11, 51)``: ``latlon_mass``, ``latlon_u``, ``latlon_v``,
    ``mapfac_u``, ``mapfac_v``, ``coriolis_m`` and ``rotation_m`` all
    ``np.array_equal`` the parent slice.

    Do NOT substitute ``type(grid)(...)`` with a recomputed reference point:
    re-solving the pole from a shifted known point rounds in a different
    binade, which is the reason ``translated`` delegates in the first place
    (projection.py:282-288).
    """
    i0, j0, nx, ny = int(i0), int(j0), int(nx), int(ny)
    if nx < 1 or ny < 1:
        raise SlabIngestError(f"window must be non-empty, got {nx}x{ny}")
    if i0 < 0 or j0 < 0 or i0 + nx > grid.e_we - 1 or j0 + ny > grid.e_sn - 1:
        raise SlabIngestError(
            f"window [{j0}, {j0 + ny}) x [{i0}, {i0 + nx}) does not fit a "
            f"{grid.e_sn - 1}x{grid.e_we - 1} grid")
    out = grid.translated(i0, j0)
    out.e_we = nx + 1
    out.e_sn = ny + 1
    cen_lat, cen_lon = out.ij_to_latlon(out.e_we / 2.0, out.e_sn / 2.0)
    out.cen_lat = float(cen_lat)
    out.cen_lon = float(cen_lon)
    return out


# --------------------------------------------------------------------------
# the slab plan
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RowSlab:
    """One horizontal row band of the target domain, with its halo.

    ``j0``/``j1`` are the GLOBAL mass rows this slab OWNS -- the rows written
    back.  ``g0``/``g1`` are the global mass rows it is BUILT over, which
    extend below ``j0`` by :data:`SLAB_Y_HALO` wherever the domain has rows to
    give.  ``last`` marks the slab that additionally owns the closing v face at
    global row ``ny``.
    """

    index: int
    j0: int
    j1: int
    g0: int
    g1: int
    last: bool

    @property
    def owned_rows(self) -> int:
        return self.j1 - self.j0

    @property
    def built_rows(self) -> int:
        return self.g1 - self.g0

    @property
    def offset(self) -> int:
        """Where the owned rows start inside the built slab."""
        return self.j0 - self.g0


def plan_row_slabs(ny: int, rows_per_slab: int, *,
                   y_halo: int = SLAB_Y_HALO) -> tuple[RowSlab, ...]:
    """Partition ``ny`` mass rows into slabs of at most ``rows_per_slab``.

    The owned rows partition ``[0, ny)`` exactly -- that is what makes the
    write-back a partition rather than an overlay, and
    :mod:`tilestream.test_realdata` asserts it before it asserts anything
    about values.  ``y_halo`` is a parameter ONLY so the gate can set it to
    zero and watch the answer break; production has one value and it is
    :data:`SLAB_Y_HALO`.
    """
    ny = int(ny)
    rows_per_slab = int(rows_per_slab)
    if ny < 1:
        raise SlabIngestError(f"ny must be positive, got {ny}")
    if rows_per_slab < 1:
        raise SlabIngestError(
            f"rows_per_slab must be positive, got {rows_per_slab}")
    if y_halo < 0:
        raise SlabIngestError(f"y_halo must be non-negative, got {y_halo}")
    slabs: list[RowSlab] = []
    j0 = 0
    while j0 < ny:
        j1 = min(j0 + rows_per_slab, ny)
        g0 = max(0, j0 - y_halo)
        slabs.append(RowSlab(index=len(slabs), j0=j0, j1=j1, g0=g0, g1=j1,
                             last=(j1 == ny)))
        j0 = j1
    return tuple(slabs)


# --------------------------------------------------------------------------
# staggering-aware write-back
# --------------------------------------------------------------------------

def window_rows(kind: str, slab: RowSlab) -> tuple[int, int, int, int]:
    """``(dst_lo, dst_hi, src_lo, src_hi)`` rows for a field of ``kind``.

    ``kind`` is ``gather.classify``'s answer: ``"mass"``/``"w"``/``"u"`` are
    unstaggered in y, ``"v"`` carries the closing face.

    This is the ONLY place the staggering decides a row range, and it is the
    one line that can be wrong while everything stays finite and smooth.  The
    v rule is the interesting one: every slab owns v faces ``[j0, j1)``, and
    the LAST slab owns ``[j0, ny]`` -- one more than its mass rows -- because
    the domain's closing face belongs to somebody.  Give the closing face to
    every slab and the next slab overwrites it (harmless, and therefore
    invisible until the last slab, which has no successor).  Give it to
    nobody and ``v[ny]`` is never written at all, which on a poisoned store
    is a NaN and on a zeroed one is a silent zero row.
    """
    if kind == "v":
        extra = 1 if slab.last else 0
    elif kind in ("mass", "w", "u"):
        extra = 0
    else:
        raise SlabIngestError(f"unknown stagger kind {kind!r}")
    src_lo = slab.offset
    count = slab.owned_rows + extra
    return slab.j0, slab.j0 + count, src_lo, src_lo + count


def write_window(store_arrays: Mapping[str, np.ndarray],
                 slab_arrays: Mapping[str, Any],
                 slab: RowSlab, *, nz: int, ny: int, nx: int,
                 row_shift: int = 0, force_kind: str | None = None,
                 allow_pageable: bool = False) -> int:
    """Copy one slab's fields into the store's rows.  Returns bytes written.

    ``store_arrays`` are the pinned full-domain buffers; ``slab_arrays`` are
    the slab state's, device or host.  Both are keyed identically -- the
    caller passes the same inventory function to both ends -- and a key
    present on one side only is an error, not a skip.

    The copy is ``cudaMemcpy3D``, through ``gather``'s own resolved-block
    machinery, and it has to be.  The destination is a ROW WINDOW of a
    full-domain array, which for any 3-D field is strided in the leading axis;
    ``cupy.ndarray.get(out=...)`` refuses a non-contiguous ``out`` outright
    (measured: ``RuntimeError: 'out' cannot be specified when copying to
    non-contiguous ndarray``), and the obvious workaround -- ``cupy.asnumpy``
    into a temporary, then a NumPy slice assignment -- allocates a second copy
    of the slab in PAGEABLE host memory and moves every byte twice.  On a
    domain sized for this project that temporary is the thing that would
    reintroduce the full-domain host allocation the slab loop exists to
    avoid.  The 3-D block copy writes straight into the pinned store.

    ``row_shift`` and ``force_kind`` exist for the gate's negative controls
    and have no production caller: ``row_shift`` writes the slab at the wrong
    global rows and ``force_kind`` classifies every field as one stagger, so
    a v field loses its closing face.  Both produce a store that is finite,
    smooth and wrong, which is precisely the point.
    """
    import cupy as cp

    from tilestream import gather as _gather

    missing = set(store_arrays) ^ set(slab_arrays)
    if missing:
        raise SlabIngestError(
            f"slab and store inventories differ on {sorted(missing)}")
    copies = []
    for name, dst in store_arrays.items():
        src = slab_arrays[name]
        if tuple(src.shape[:-2]) != tuple(dst.shape[:-2]):
            raise SlabIngestError(
                f"{name}: slab {src.shape} and store {dst.shape} differ "
                "outside the horizontal axes")
        # The block copy is measured in BYTES off the destination's itemsize
        # (``_FieldCopy`` takes one itemsize, not two), so a dtype mismatch
        # would copy the right number of bytes from the wrong number of
        # elements and leave a field that is finite and shifted.  Refuse.
        if np.dtype(src.dtype) != np.dtype(dst.dtype):
            raise SlabIngestError(
                f"{name}: slab dtype {src.dtype} != store dtype {dst.dtype}")
        kind = force_kind or _gather.classify(
            src.shape, nz, slab.built_rows, nx, layers_ok=True)
        dst_lo, dst_hi, src_lo, src_hi = window_rows(kind, slab)
        dst_lo += row_shift
        dst_hi += row_shift
        if dst_lo < 0 or dst_hi > int(dst.shape[-2]):
            raise SlabIngestError(
                f"{name}: rows [{dst_lo}, {dst_hi}) fall outside a "
                f"{dst.shape} store field")
        if src_hi > int(src.shape[-2]):
            raise SlabIngestError(
                f"{name}: rows [{src_lo}, {src_hi}) fall outside a "
                f"{src.shape} slab field")
        _gather._check_host(dst, name, "destination", allow_pageable)
        copies.append(_gather._FieldCopy(
            name, kind, int(dst.dtype.itemsize),
            _gather._memcpy_kind(_gather.is_device_array(src),
                                 _gather.is_device_array(dst)),
            src.shape, dst.shape,
            src_lo, 0, dst_lo, 0,
            dst_hi - dst_lo, int(dst.shape[-1]),
            int(np.prod(dst.shape[:-2])) if dst.ndim > 2 else 1))
    plan = _gather.TilePlan("ingest", slab, copies, ("ingest", slab.index),
                            tuple(store_arrays))
    stream = cp.cuda.get_current_stream()
    plan.execute(slab_arrays, store_arrays, stream)
    stream.synchronize()
    return int(plan.nbytes)


# --------------------------------------------------------------------------
# the ingest itself
# --------------------------------------------------------------------------

def _decode(bundle: Real74Bundle, valid_time: datetime):
    from woof.ingest.grib import cached_era5_snapshots

    snapshots = {s.valid_time: s for s in cached_era5_snapshots(
        bundle.grib, bundle.vtable)}
    try:
        return snapshots[valid_time]
    except KeyError as exc:
        raise SlabIngestError(
            f"the bundle GRIB has no snapshot at {valid_time}; it carries "
            f"{sorted(snapshots)}") from exc


def real74_geography(bundle: Real74Bundle) -> dict[str, Any]:
    """The bundle's own grid, terrain, source orography and map fields.

    ``HGT_M`` and the map/Coriolis/rotation fields come from ``geo_em`` rather
    than being recomputed, because that is what ``build_forcing`` does and
    the acceptance case is graded against it.  ``SOILHGT`` comes from
    ``met_em`` -- see ``woof/ingest/README.md``: the retained April-1974 ERA5
    GRIB omits invariant geopotential, and this case reads the source
    orography from the reference ``met_em`` as a declared input-data
    exception.
    """
    import netCDF4

    from woof.static.lambert import grids_from_wps_namelist

    grid = grids_from_wps_namelist(bundle.namelist_wps)[0]
    with netCDF4.Dataset(bundle.geo_em) as ds:
        terrain = np.asarray(ds.variables["HGT_M"][0], dtype=np.float64)
        map_fields = {
            name: np.asarray(ds.variables[name][0], dtype=np.float64)
            for name in ("MAPFAC_M", "MAPFAC_U", "MAPFAC_V", "F", "E",
                         "SINALPHA", "COSALPHA")
        }
    with netCDF4.Dataset(bundle.met_em) as ds:
        source_orography = np.asarray(ds.variables["SOILHGT"][0],
                                      dtype=np.float64)
    return dict(grid=grid, terrain=terrain, source_orography=source_orography,
                map_fields=map_fields)


def rebuilt_grid(grid, i0: int, j0: int, nx: int, ny: int):
    """The window's projection REBUILT from its own extents.  WRONG ON PURPOSE.

    This is what a slab gets if nobody thinks about it: a fresh grid of the
    same class with the same projection parameters and the slab's ``e_we`` /
    ``e_sn``.  ``projection.py:122-123`` then defaults ``known_x``/``known_y``
    to ``(e_we/2, e_sn/2)``, so the slab believes ITS centre is the reference
    point -- every slab lands on top of every other, all of them centred where
    the whole domain is centred.  The result is smooth, finite, correctly
    shaped, and 100% wrong away from the middle slab, which is exactly the
    kind of defect this project keeps catching only with a control.

    Kept in the module rather than in the gate because the gate must call the
    SAME entry point as production with one argument changed, not a
    re-implementation of it.
    """
    del i0, j0
    return type(grid)(
        ref_lat=grid.ref_lat, ref_lon=grid.ref_lon, truelat1=grid.truelat1,
        truelat2=grid.truelat2, stand_lon=grid.stand_lon,
        dx=grid.dx, dy=grid.dy, e_we=int(nx) + 1, e_sn=int(ny) + 1)


def ingest_slab_state(source, geo: Mapping[str, Any], cfg, coord, slab: RowSlab,
                      *, p_top: float = 10000.0, backend: str = "cuda",
                      grid_fn=None, diagnose: bool = True):
    """Horizontally interpolate and real-initialise ONE row slab.

    Returns the slab's ``DomainState`` (shape ``cfg.nz x slab.built_rows x
    cfg.nx``).  Nothing full-domain is allocated: the target grid is
    :func:`window_grid`'s exact sub-rectangle, the interpolation writes the
    slab's rows only, and ``initialize_real`` allocates a slab-shaped
    ``DomainState``.

    ``grid_fn`` is the seam the negative control uses: it is
    :func:`window_grid` in production and :func:`rebuilt_grid` when the gate
    wants to show what happens if a slab derives its own projection.
    """
    from woof.ingest.horiz import interpolate_era5_to_lambert
    from woof.ingest.real import initialize_real

    grid = geo["grid"]
    nx = int(cfg.nx)
    gw = (grid_fn or window_grid)(grid, 0, slab.g0, nx, slab.built_rows)
    met = interpolate_era5_to_lambert(source, gw, backend=backend)
    slab_cfg = replace(cfg, ny=slab.built_rows)
    result = initialize_real(
        met, slab_cfg, coord,
        geo["terrain"][slab.g0:slab.g1],
        source_orography=geo["source_orography"][slab.g0:slab.g1],
        # The two routes did not reach it.
        #  ``gw`` is the SLAB's window grid, not
        # the full domain's -- which is the correct carrier here:
        # this initialize_real builds exactly those rows, so the
        # mass-point lat/lon the WIF interpolation needs are the
        # slab's own.  Passing the full grid would interpolate the
        # global climatology onto the wrong rows.
        p_top=p_top, sfcp_to_sfcp=True, grid=gw)
    mf = geo["map_fields"]
    result.state.set_map_coriolis(
        mf["MAPFAC_M"][slab.g0:slab.g1],
        mf["MAPFAC_U"][slab.g0:slab.g1],
        mf["MAPFAC_V"][slab.g0:slab.g1 + 1],
        mf["F"][slab.g0:slab.g1], mf["E"][slab.g0:slab.g1],
        sina=mf["SINALPHA"][slab.g0:slab.g1],
        cosa=mf["COSALPHA"][slab.g0:slab.g1])
    if diagnose:
        # ``runtime.py:826``: "initialize_real loads prognostics but does not
        # launch the EOS kernel."  ``prepare_real_case`` therefore diagnoses
        # the time-zero atmosphere before anything reads it, and a store
        # filled without this holds an UNDIAGNOSED ``p``/``al``/``alt`` --
        # whatever ``DomainState`` allocated -- which the first step then
        # consumes.  The EOS is per column, so the slab's answer is the
        # domain's; the bit-exact ingest comparison is what says so.
        from woof.core.diagnostics import update_diagnostics

        update_diagnostics(result.state, cfg.hypsometric_opt)
    return result


def monolithic_ingest(bundle: Real74Bundle, cfg, coord, *,
                      valid_time: datetime = REAL74_START,
                      p_top: float = 10000.0, backend: str = "cuda",
                      geo: Mapping[str, Any] | None = None,
                      diagnose: bool = True):
    """The reference: ``build_forcing``'s own single full-domain ingest.

    This is what the streaming ingest is compared against and it is exactly
    the thing a larger-than-VRAM domain cannot do.  Kept here so both sides
    of the comparison read the same GRIB through the same decode cache and
    differ ONLY in whether the domain was cut into slabs.
    """
    geo = real74_geography(bundle) if geo is None else geo
    source = _decode(bundle, valid_time)
    whole = RowSlab(index=0, j0=0, j1=int(cfg.ny), g0=0, g1=int(cfg.ny),
                    last=True)
    return ingest_slab_state(source, geo, cfg, coord, whole,
                             p_top=p_top, backend=backend, diagnose=diagnose)


def allocate_domain_arrays(probe: Mapping[str, Any], *, nz: int, probe_ny: int,
                           ny: int, nx: int) -> dict[str, np.ndarray]:
    """Pinned FULL-DOMAIN buffers shaped like ``probe``'s slab arrays.

    ``probe`` is one slab's inventory (any inventory function).  Each array's
    staggering is read off its shape against the SLAB's extents and the y
    extent is re-expanded to the domain's -- the same probe-then-read
    discipline ``hoststore.build_manifest`` uses for the carrier set, and for
    the same reason: transcribing which of ``woof``'s setup arrays are 2-D
    or 3-D under which ``terrain_opt`` would drift the day a branch changes.

    This is how the GEOGRAPHY store is built out of core.  The carrier store
    has :class:`tilestream.hoststore.HostDomainStore`; geography has no such
    class because it is not the serialization contract, and building it from
    a full-domain ``DomainState`` -- the obvious thing -- is exactly the
    allocation this lane exists to avoid.
    """
    from tilestream import gather as _gather

    out: dict[str, np.ndarray] = {}
    for name, array in probe.items():
        kind = _gather.classify(array.shape, nz, probe_ny, nx, layers_ok=True)
        shape = tuple(array.shape[:-2]) + (ny + (kind == "v"),
                                           int(array.shape[-1]))
        buffer = _gather.pinned_empty(shape, array.dtype)
        # POISON, for the same reason ``driver._empty_like_store`` poisons:
        # a row the slab loop never writes must be an obvious NaN, not a
        # plausible zero.  ``force_kind='mass'`` is a negative control that
        # leaves exactly one v row unwritten, and on a zeroed buffer it would
        # be a smooth, finite, wrong wind.
        if buffer.dtype.kind == "f":
            buffer.fill(np.nan)
        out[name] = buffer
    return out


def stream_real_into_store(store, bundle: Real74Bundle, cfg, coord, *,
                           rows_per_slab: int,
                           valid_time: datetime = REAL74_START,
                           p_top: float = 10000.0, backend: str = "cuda",
                           y_halo: int = SLAB_Y_HALO,
                           geo: Mapping[str, Any] | None = None,
                           inventory_fn=None,
                           geography=None,
                           geography_inventory_fn=None,
                           row_shift: int = 0,
                           force_kind: str | None = None,
                           grid_fn=None,
                           report: dict | None = None) -> None:
    """Fill ``store`` from real data, slab by slab.  THE PRODUCTION PATH.

    ``store`` is a :class:`tilestream.hoststore.HostDomainStore` (or anything
    exposing ``.arrays``); it is filled IN PLACE and nothing is returned.
    Peak device residency is set by ``rows_per_slab``, not by ``cfg.ny``:
    every device allocation in the loop is ``O(rows_per_slab * nx)`` and the
    slab state is released before the next one is built.

    ``geography`` is the second, equally necessary store: the horizontally
    varying INPUT (``ht``, ``mub2d``, the map factors, Coriolis, rotation and
    the terrain-following ``thb/pb/alb/phb``) that
    ``tilestream.driver.run_tiled`` gathers per tile and never rebuilds.  It
    is filled from the SAME slabs by the same window writer, because it is
    produced by the same ``initialize_real`` call -- a separate full-domain
    pass to build it would put the allocation straight back.  Pass a dict
    from :func:`allocate_domain_arrays`, or ``None`` to skip it (a dry
    periodic-uniform lane does not need one).

    ``y_halo``, ``row_shift``, ``force_kind`` and ``grid_fn`` are the gate's
    four negative controls and have no production caller.  Pass ``report={}`` for
    the per-slab diagnostics, including the device high-water mark, which is
    the number this whole module exists to hold down.
    """
    import cupy as cp

    from tilestream import driver as _driver
    from tilestream import gather as _gather

    geo = real74_geography(bundle) if geo is None else geo
    if (geo["grid"].e_we - 1, geo["grid"].e_sn - 1) != (int(cfg.nx),
                                                        int(cfg.ny)):
        raise SlabIngestError(
            f"cfg is {cfg.ny}x{cfg.nx} but the bundle grid is "
            f"{geo['grid'].e_sn - 1}x{geo['grid'].e_we - 1}")
    source = _decode(bundle, valid_time)
    slabs = plan_row_slabs(int(cfg.ny), rows_per_slab, y_halo=y_halo)
    take = inventory_fn or _gather.inventory
    take_geo = geography_inventory_fn or _driver.geography_inventory
    dst = store.arrays if hasattr(store, "arrays") else dict(store)

    pool = cp.get_default_memory_pool()
    high_water = 0
    written = 0
    for slab in slabs:
        pool.free_all_blocks()
        result = ingest_slab_state(source, geo, cfg, coord, slab,
                                   p_top=p_top, backend=backend,
                                   grid_fn=grid_fn)
        written += write_window(dst, take(result.state), slab,
                                nz=int(cfg.nz), ny=int(cfg.ny), nx=int(cfg.nx),
                                row_shift=row_shift, force_kind=force_kind)
        if geography is not None:
            written += write_window(geography, take_geo(result.state), slab,
                                    nz=int(cfg.nz), ny=int(cfg.ny),
                                    nx=int(cfg.nx), row_shift=row_shift,
                                    force_kind=force_kind)
        high_water = max(high_water, int(pool.total_bytes()))
        del result
    pool.free_all_blocks()
    if report is not None:
        report.update(slabs=len(slabs), rows_per_slab=int(rows_per_slab),
                      y_halo=int(y_halo), bytes_written=written,
                      device_high_water_bytes=high_water,
                      built_rows=tuple(s.built_rows for s in slabs))


# --------------------------------------------------------------------------
# the tile buffer
# --------------------------------------------------------------------------

def make_real_tile_state(tile_cfg, coord_fn, *, p_top: float = 10000.0,
                         terrain_height: float = 0.0):
    """A tile buffer for a REAL-ETA, real-terrain domain.

    ``driver.make_tile_state`` and ``harness.geography_builder`` both call
    ``make_vertical_coord(nz, hybrid_opt, etac)`` with no ``eta_levels``, i.e.
    they build the DEFAULT stretch.  ``real74_d01`` runs an explicit 50-level
    eta table (``verify/cases/real74_d01.ETA_LEVELS``), and the purely
    vertical setup arrays -- ``c1h..c4f``, ``znu``, ``znw``, ``dnw``,
    ``rdnw``, ``dn``, ``rdn``, ``fnp``, ``fnm``, ``cf1..cfn1`` -- are
    deliberately NOT gathered (``driver.geography_inventory``'s docstring: a
    tile rebuilds them exactly).  A buffer built on the default stretch
    therefore rebuilds them WRONG, and nothing in the pipeline would notice:
    the arrays are the right shape and the right dtype and the run produces
    plausible weather.  ``coord_fn()`` is the domain's own coordinate
    factory, so the buffer rebuilds the same table.

    ``run_tiled`` does NOT check this -- ``driver.setup_window_mismatches``
    and ``driver.setup_scalar_mismatches`` exist but nothing calls them from
    the run path -- so :mod:`tilestream.test_realdata` calls both directly on
    a real buffer against a real parent, and runs a buffer built on the
    default stretch as a negative control.  MEASURED: the default stretch
    puts ``znu`` up to 0.1763 away from the case's table, disagrees on all
    sixteen vertical setup arrays and on the five ``cf*`` scalars, and the
    streamed eight-step answer then differs in ten of the thirteen carriers
    by up to 4.5e5.

    The base state is the WRF analytic one ``initialize_real`` itself uses
    (``woof.ingest.real._make_real_base``), on a CONSTANT terrain: identical
    machinery so every vertical scalar matches by construction, obviously
    wrong horizontally so a geography array the gather misses shows up as
    flat ground rather than as somebody else's mountains.
    """
    from woof.core.state import init_at_rest
    from woof.ingest.real import _make_real_base

    from tilestream import harness as _h

    coord = coord_fn()
    terrain = np.full((int(tile_cfg.ny), int(tile_cfg.nx)),
                      float(terrain_height), dtype=np.float64)
    base = _make_real_base(coord, terrain, float(p_top),
                           float(tile_cfg.base_temp),
                           hypsometric_opt=int(tile_cfg.hypsometric_opt))
    state = init_at_rest(tile_cfg, coord, base)
    _h.install_geography(state, _h.neutral_geography(
        tile_cfg, terrain_height=terrain_height))
    return state


# --------------------------------------------------------------------------
# the periodic closure
# --------------------------------------------------------------------------

def periodic_closure(arrays: Mapping[str, np.ndarray], *,
                     nz: int, ny: int, nx: int) -> tuple[str, ...]:
    """Alias every closing face onto its opening one.  Returns what it touched.

    A periodic domain's ``u[..., nx]`` IS ``u[..., 0]`` and its
    ``v[..., ny, :]`` IS ``v[..., 0, :]``; ``woof.verify.npref`` enforces the
    same duplicate on its seeded states (npref.py:2284-2314) and the tiled
    gate depends on it, because ``spec.TileSpec._axis_gather`` reduces every
    window mod ``nx`` and so never reads the alias slot -- a tile takes column
    0 where a monolithic run reads column ``nx``.

    Real ERA5 winds on a CONUS Lambert grid do not satisfy it: the domain is
    not periodic.  This function makes them, in place, and the comparison
    applies it to BOTH sides.  Applying it to one side only would be a way to
    fake agreement; applying it to both cannot be, because it is not a
    function of how the state was built.
    """
    touched: list[str] = []
    for name, array in arrays.items():
        if array.ndim < 2:
            continue
        h, w = int(array.shape[-2]), int(array.shape[-1])
        if w == nx + 1 and h == ny:
            array[..., :, -1] = array[..., :, 0]
            touched.append(name)
        elif h == ny + 1 and w == nx:
            array[..., -1, :] = array[..., 0, :]
            touched.append(name)
    return tuple(touched)


# --------------------------------------------------------------------------
# measurement helpers
# --------------------------------------------------------------------------

def device_high_water(fn, *args, **kwargs) -> tuple[Any, int]:
    """``(result, peak device bytes)`` for one call.

    ``cupy``'s default pool reports ``total_bytes()`` = pool capacity, which
    only ever grows, so it is a true high-water mark ONCE the pool has been
    released.  ``free_all_blocks`` before and after is what makes the number
    mean "this call", and it is why this helper exists instead of two inline
    reads.
    """
    import cupy as cp

    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()
    before = int(pool.total_bytes())
    out = fn(*args, **kwargs)
    peak = int(pool.total_bytes())
    return out, peak - before
