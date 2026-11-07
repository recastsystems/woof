"""A REAL forecast, from a REAL analysis, streamed tile by tile.

Everything the streaming lane had run until now was periodic: ``test_gate``
says so in its own first paragraph -- *"Periodic is deliberate.  With
open_x=open_y=specified=nested=False every lateral-boundary special case
disappears, which removes the entire BC surface from the comparison."*  That
is the right choice for a bit-exact transport gate and it is the wrong world
for a forecast.  A real case is SPECIFIED: its edges are driven by an
analysis, on the analysis's own cadence, and the boundary treatment is not a
detail of the edge -- it is what puts the storms where they are.

This module is the missing piece, and the whole of it is one idea.


THE PROBLEM, STATED EXACTLY
---------------------------
``woof``'s lateral boundary is applied INSIDE the step
(``dycore.step`` -> ``lateral_bc.apply_state_lateral_boundaries`` at every RK
stage), on the outermost ``max(spec_zone, relax_zone)`` cells of whatever
array it is handed.  A tile buffer IS such an array.  So a tile is not
merely *allowed* to apply a boundary -- it is UNABLE NOT TO, once
``cfg.specified`` is set, and ``cfg`` is the domain's.

Four of a 4x4 tiling's sixteen tiles have no domain edge at all, and the
other twelve have one or two.  Every tile nonetheless has four compute-window
edges, and the relaxation kernel will treat all four of them.  Eight of the
sixteen tiles' twenty-four real edges are real; the remaining forty are
FICTION -- interior seams the kernel cannot tell apart from a domain edge.


THE ANSWER: THE FICTION IS QUARANTINED, AND THE HALO IS WHAT QUARANTINES IT
--------------------------------------------------------------------------
The streaming lane's entire correctness argument is a domain-of-dependence
argument: a tile's interior after one step depends only on cells within
``harness.halo_radius(cfg)`` of it at the start of the step, so a halo that
wide makes the interior exact.  The same argument absorbs the boundary,
because a spurious relaxation is not a special kind of wrongness -- it is
just a wrong value, at a known place, and wrong values propagate at the same
finite speed as right ones.

    A spurious boundary application touches at most the outermost
    ``S`` cells of the compute window, where ``S`` is what the boundary
    kernels still write on a seam side (:func:`seam_fiction_width`).
    Over one step that contamination reaches at most ``S + R`` cells in,
    where ``R = harness.halo_radius(cfg)``.  A halo of ``R + S`` therefore
    leaves every tile interior untouched by it.

``S`` used to be the whole frame, ``max(spec_zone, relax_zone)``: the
relaxation kernel treated all four window edges as domain edges.  Since
``woof.ingest.lateral_bc`` masks the relaxation band per side
(``LateralBoundaries.seam_sides``, set by ``streaming.window_boundaries``
and ``TileBoundaryTables`` for every tile and slab), a seam relaxes
nothing, and the only perimeter write left on a seam is the specified
zone itself (``state_specified_relaxation``'s ``boundary_index`` branch
and the end-of-step install): rows ``0 .. spec_zone - 1``, which is
``spec_zone`` cells.  So ``S = spec_zone``, and at WRF's standard
``spec_zone=1`` the forced halo is the dependency radius plus one cell.
Not ``spec_zone - 1``: fiction installed at row ``spec_zone - 1`` reaches
row ``spec_zone - 1 + R`` within the step and the first owned row is row
``halo``, so ``halo = R + spec_zone - 1`` lets it land on an owned cell
by the proof's own counting and holds only on the slack inside
``halo_radius`` (its 16 covers a measured 14 at ``ns = 4``).  ``R + S``
keeps that slack where it was measured.  MEASURED, not assumed: a 3 km
specified crop (499 x 567 x 50, ``spec_zone=1, relax_zone=9``,
time_step_sound at the adaptive ceiling, ``R = 19``) split 2x1 over two
RTX 5090s ran 12 h with halo 28 (``R + 9``, the old rule), 22, 20
(``R + 1``, this rule) and 19 (``R``), and every hourly history file is
byte-identical across the four and to the one-card run.  The dycore's
own edge narrowing (advection order, diff6 strip) corrupts a seam only
DURING the step, so what it leaves to propagate is strictly inside ``R``
(see ``harness.tile_config``).

:func:`halo_for` is that arithmetic and nothing else.  At WOOF's
``time_step_sound=4`` and WRF's standard ``spec_zone=1`` it is
``16 + 1 = 17``.  Nothing here is tuned: both terms are read off the
config.

Two consequences, and both are essential:

*   THE PLAN MUST BE NON-PERIODIC.  ``spec.plan_tiles(periodic=False)``
    CLAMPS an edge tile's compute window into the domain instead of wrapping
    it, which is what makes the tile's own edge coincide with the domain's.
    Every tile keeps the same compute shape, so one ``tile_cfg`` still
    serves them all.  Under ``periodic=True`` the gathered window would wrap
    round the far side of the domain and the west boundary strip would be
    fed east-boundary air -- the same class of defect the ``bcfix`` commit
    removed from the small-step ``uv`` kernels one layer down.

*   THE TILE'S BOUNDARY TABLES ARE THE DOMAIN'S, WINDOWED.
    :func:`tile_boundaries` slices the domain's ``LateralBoundaries`` to each
    tile's compute window: west/east strips are cut in y at
    ``[cj0, cj0 + cny)``, south/north strips in x at ``[ci0, ci0 + cnx)``,
    each with the field's own staggering added.  A tile that owns the
    domain's west edge therefore relaxes toward THE ANALYSIS, at the right
    latitudes, on the right cadence.  A tile that does not gets real
    analysed air applied at the wrong place, which is fiction -- and lands
    entirely inside the quarantine.

That the fiction is bounded rather than merely small is why this is stated as
a proof and not as a hope, and :func:`gate` is the measurement:  a monolithic
specified run and a streamed one, from the same real initial state, over the
same real boundary series, compared carrier by carrier.


WHAT IS NOT SOLVED HERE
-----------------------
The INGEST is still monolithic.  ``runtime.prepare_real_case`` builds a
full-domain ``DomainState`` per forcing time, and MEASURED on this case that
peaks at far more device memory per cell than the forecast itself does --
so on any given card the largest domain that can be PREPARED is smaller than
the largest that can be STREAMED.  :mod:`tilestream.realdata` closes exactly
this gap for the dynamics carriers by row slabs and is bit-exact doing it;
what it does not yet do is the soil/landuse/physics-driver half of
``prepare_real_case``.  Until that lands, a streamed real forecast is capped
by its own preparation, and the accurate way to report a domain size is to say
which of the two limits it was up against.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Any, Mapping, Sequence

import numpy as np


__all__ = [
    "CaseStores",
    "RealCaseError",
    "boundary_stagger",
    "build_stores",
    "composite_reflectivity",
    "halo_for",
    "plan",
    "prepare",
    "seam_fiction_width",
    "surface_snapshot",
    "tile_boundaries",
    "tile_boundary_binder",
]


class RealCaseError(RuntimeError):
    """A real streamed case could not be set up consistently."""


# --------------------------------------------------------------------------
# the halo
# --------------------------------------------------------------------------

def boundary_width(cfg) -> int:
    """Cells a specified-boundary application touches on a TRUE domain edge,
    from the edge in: ``max(spec_zone, relax_zone)``, the width
    ``lateral_bc._launch_state_relaxation`` enumerates as ``active_width``.

    Not the seam figure.  On a seam side the relaxation band is masked
    (``LateralBoundaries.seam_sides``) and only the specified zone is
    written, so the halo pays :func:`seam_fiction_width`, not this.  This
    is the whole frame a domain edge carries; nothing on the halo road
    reads it any more (the stranded-zone refusal in
    ``streaming.tile_seam_sides`` computes the same band itself).  Kept as
    the public name of that width.
    """
    return max(int(getattr(cfg, "spec_zone", 1)),
               int(getattr(cfg, "relax_zone", 4)))


def seam_fiction_width(cfg) -> int:
    """Cells the boundary kernels still write on a SEAM side, from the edge
    in, before a step starts to propagate: ``spec_zone``.

    The relaxation band is masked per side (``lateral_bc._relax_side_mask``
    reads ``LateralBoundaries.seam_sides``), so a seam relaxes nothing.
    What remains unmasked is the specified zone: ``state_specified_relaxation``
    writes rows ``0 .. spec_zone - 1`` of every side's tendency at each RK
    stage (``boundary_index`` tests every side, seam or not), and
    ``install_mu_boundary`` / ``finalize_state_field`` install the same
    rows at the end of the step.  That is ``spec_zone`` cells of fiction,
    and the halo pays exactly that beyond the radius; the module docstring
    says why not one less.  At WRF's standard ``spec_zone=1`` that is one
    cell, which the 2x1 split of a 3 km specified crop confirmed byte for
    byte over 12 h at halo ``R + 1`` (and at ``R``, on the radius's own
    slack).  :func:`boundary_width` is the whole frame a true domain edge
    carries; it no longer sizes the halo.
    """
    return int(getattr(cfg, "spec_zone", 1))


def halo_for(cfg) -> int:
    """The halo a SPECIFIED domain needs: dependency radius + seam fiction.

    See the module docstring.  ``harness.halo_radius(cfg)`` alone is correct
    only when every compute-window edge is either a true domain edge or a
    faithful periodic neighbour; on a specified domain an interior tile's
    window edge is neither, and the specified-zone write puts
    :func:`seam_fiction_width` cells of fiction into it before the step
    even starts to propagate.
    """
    from tilestream import harness

    return int(harness.halo_radius(cfg)) + seam_fiction_width(cfg)


def plan(cfg, tile_nx: int, tile_ny: int, halo: int | None = None):
    """The NON-PERIODIC tile plan for ``cfg``, validated."""
    from tilestream import spec as _spec

    halo = halo_for(cfg) if halo is None else int(halo)
    specs = _spec.plan_tiles(int(cfg.nx), int(cfg.ny), int(tile_nx),
                             int(tile_ny), halo, False)
    _spec.validate_plan(specs, int(cfg.ny), int(cfg.nx))
    return specs, halo


# --------------------------------------------------------------------------
# windowing the boundary tables
# --------------------------------------------------------------------------

def boundary_stagger(boundary, ny: int, nx: int) -> tuple[int, int]:
    """``(ey, ex)`` staggering of one field's boundary tables.

    A boundary carries the shape of the field it was cut from:
    ``west.value`` is ``(nz, NY, width)`` and ``south.value`` is
    ``(nz, width, NX)``, with ``NY = ny + ey`` and ``NX = nx + ex``.  Read it
    off rather than keeping a table of field names, because the inventory is
    ``lateral_bc._coupled_device_fields``'s and it changes with
    ``mp_physics``.
    """
    from woof.ingest.lateral_bc import _boundary_field_shape

    _nz, ny_b, nx_b, _w = _boundary_field_shape(boundary)
    ey, ex = int(ny_b) - int(ny), int(nx_b) - int(nx)
    if ey not in (0, 1) or ex not in (0, 1):
        raise RealCaseError(
            f"boundary tables are {ny_b}x{nx_b} against a {ny}x{nx} domain; "
            "a field must be unstaggered or staggered by exactly one face")
    return ey, ex


def _window_side(side, y: slice | None, x: slice | None):
    from woof.ingest.lateral_bc import SideBoundary

    def cut(a):
        if y is not None:
            a = a[:, y, :]
        if x is not None:
            a = a[:, :, x]
        return np.ascontiguousarray(a)

    return SideBoundary(cut(side.value), cut(side.tendency))


def tile_boundaries(boundaries, spec, cfg):
    """The domain's ``LateralBoundaries``, cut to one tile's compute window.

    West/east are cut in y over ``[cj0, cj0 + cny + ey)`` and south/north in x
    over ``[ci0, ci0 + cnx + ex)``.  The width axis is never touched: east and
    north are stored outermost-first (``lateral_bc._field_boundary``) and the
    reversal is along width, so a y/x cut commutes with it.

    Every interval is cut, not just the active one, so the returned object is
    a complete forcing series for that tile and
    ``LateralBoundaries.interval_at`` selects inside it exactly as it does for
    the domain.
    """
    from woof.ingest.lateral_bc import (BoundaryInterval, FieldBoundary,
                                         LateralBoundaries)

    ny, nx = int(cfg.ny), int(cfg.nx)
    ys = spec.cj0
    xs = spec.ci0
    intervals = []
    for interval in boundaries.intervals:
        fields = {}
        for name, boundary in interval.fields.items():
            ey, ex = boundary_stagger(boundary, ny, nx)
            ysl = slice(ys, ys + spec.cny + ey)
            xsl = slice(xs, xs + spec.cnx + ex)
            fields[name] = FieldBoundary(
                west=_window_side(boundary.west, ysl, None),
                east=_window_side(boundary.east, ysl, None),
                south=_window_side(boundary.south, None, xsl),
                north=_window_side(boundary.north, None, xsl))
        intervals.append(BoundaryInterval(
            interval.start_seconds, interval.end_seconds, fields))
    return LateralBoundaries(tuple(intervals), boundaries.spec_bdy_width,
                             boundaries.spec_zone, boundaries.relax_zone)


def tile_boundary_binder(boundaries, specs, cfg, *, verbose=None):
    """Build every tile's windowed forcing once; return a bind callback.

    The callback is :func:`tilestream.driver.run_tiled`'s ``tile_hook``: it
    fires when a buffer STARTS SERVING A DIFFERENT TILE, in the same place
    and for the same reason the geography gather does.

    The first bind per buffer calls ``attach_streaming_lateral_boundaries``,
    which allocates ONE packed device slot sized for a single interval;
    every later bind swaps the host tables and invalidates the resident
    interval id, so the next kernel launch reloads that same slot.  The
    eager :func:`woof.ingest.lateral_bc.attach_lateral_boundaries` would
    upload every interval on every tile change instead -- correct, and
    ``interval_count`` times the traffic, per tile, per sweep.

    ``attach_*`` resets ``state.elapsed_seconds`` to zero; the driver
    restores the domain clock onto the buffer before every tile step
    (``run_tiled``'s ``set_carrier_scalars``), so the reset is overwritten
    before it can be read.  Binding on a run WITHOUT ``scalars`` would leave
    the buffer at t=0 and silently freeze the boundary in its first interval,
    so that combination is refused.
    """
    from woof.ingest.lateral_bc import attach_streaming_lateral_boundaries

    per_tile = [tile_boundaries(boundaries, spec, cfg) for spec in specs]
    nbytes = sum(int(a.nbytes)
                 for lb in per_tile
                 for interval in lb.intervals
                 for fb in interval.fields.values()
                 for sb in (fb.west, fb.east, fb.south, fb.north)
                 for a in (sb.value, sb.tendency))
    if verbose is not None:
        verbose(f"    windowed lateral boundaries: {len(per_tile)} tiles x "
                f"{len(per_tile[0].intervals)} intervals x "
                f"{len(per_tile[0].intervals[0].fields)} fields = "
                f"{nbytes / 2**30:.2f} GiB host")

    bound: dict[int, bool] = {}

    def bind(tile_state, _spec, itile, _stream=None) -> None:
        # driver.run_tiled calls tile_hook(tile_state, tspec, itile,
        # stream); see driver.py's docstring at the call site.
        lb = per_tile[itile]
        key = id(tile_state)
        if not bound.get(key):
            attach_streaming_lateral_boundaries(tile_state, lb)
            bound[key] = True
            return
        tile_state.lateral_boundaries = lb
        resident = getattr(tile_state, "_lateral_boundary_device", None)
        if resident is None or not resident.streaming_external:
            raise RealCaseError(
                "tile buffer lost its streaming lateral attachment")
        # Force the next _resident_interval() to refill the packed slot from
        # THIS tile's tables.  Without it a buffer keeps serving the previous
        # tile's boundary whenever the two tiles happen to be in the same
        # forcing interval -- which is almost always.
        resident.active_host_interval_id = None

    bind.per_tile = per_tile
    bind.nbytes = nbytes
    return bind


# --------------------------------------------------------------------------
# the tile buffer
# --------------------------------------------------------------------------

def tile_state_builder(exp, boundaries=None):
    """A ``(cfg, seed) -> (state, driver)`` buffer builder for a REAL case.

    ``harness.make_physics_state`` cannot serve here, for two reasons that
    are both about the buffer's PRE-GATHER content rather than its shape:

    * it builds on ``make_vertical_coord``'s default stretch.  A real case
      runs its own eta table, and ``driver.geography_inventory`` deliberately
      does NOT gather the purely vertical setup arrays because a tile is
      supposed to rebuild them exactly.  ``coord`` here is the case's own.

    * it SEEDS every prognostic with Gaussian noise whose amplitudes were
      chosen against the harness's own 8 km-top config.  On this case's eta
      table -- whose lowest layer is ten times thinner (``znu[0]`` 0.9989
      against 0.9898) -- the seeded ``thp``/``mup`` drive the diagnosed
      temperature to 438 K and RRTMGP REFUSES the buffer's warmup step
      outright (valid over [160, 355] K).  MEASURED, and it is why this
      exists.  Nothing is lost by not seeding: every persisted array is
      overwritten by the first gather, and ``driver.make_tile_state``
      already argues that an at-rest buffer is the better failure mode --
      a field a gather failed to write shows up at rest rather than as
      plausible weather.

    ``qv`` is the one exception and it is not seeding: it is set to the
    analytic WK82 profile because RRTMGP and the microphysics both want a
    moist column for the single warmup step, whose only purpose is to force
    the lazily-allocated carriers into existence.

    ``boundaries`` must be a tile-shaped :class:`LateralBoundaries` whenever
    ``cfg.specified``: ``dycore.step`` calls
    ``apply_state_lateral_boundaries`` at every RK stage and that function
    REFUSES a specified config with nothing attached, so the buffer cannot
    take its warmup step without one.  Any tile's tables will do -- the
    warmup's values are overwritten by the first gather and the ``tile_hook``
    rebinds the right ones before the buffer serves a tile -- but the LAYOUT
    has to be the tile's, which is why this takes an argument rather than
    reaching for the domain's.
    """
    def builder(cfg, seed: int = 0, **_kwargs):
        import cupy as cp

        from woof.core.diagnostics import update_diagnostics
        from woof.core.physics import (initialize_physics,
                                        physics_driver_required)
        from woof.core.state import init_at_rest
        from woof.ingest.real import _make_real_base
        from woof.runtime import vertical_coord_for
        from woof.verify.cases.wk82 import wk82_sounding
        from tilestream import harness

        coord = vertical_coord_for(exp.vertical, cfg.nz)
        geo = harness.neutral_geography(cfg)
        # THE BASE STATE IS BUILT THE INGEST'S WAY, NOT THE HARNESS'S, AND
        # THE REASON IS ONE SCALAR.  ``grid.make_base_state`` DERIVES p_top
        # by integrating the sounding to ``ztop``; ``initialize_real`` is
        # HANDED p_top from the namelist.  On this case those disagree --
        # MEASURED 5717.78 Pa against the declared 10000.0 -- and
        # ``state.p_top`` is a setup SCALAR the dycore reads on every column,
        # while ``coord``'s pressure-weighted hybrid coefficients c2h/c2f/
        # c4h/c4f carry a factor (p0 - p_top) and are among the arrays
        # ``driver.geography_inventory`` deliberately does NOT gather.  A
        # buffer built the harness's way therefore runs a DIFFERENT vertical
        # coordinate from the domain, with every array the right shape, and
        # the streamed answer differs everywhere -- which is exactly what
        # ``driver.setup_scalar_mismatches`` reported before this line
        # existed.
        base = _make_real_base(coord, np.asarray(geo.terrain, dtype=np.float64),
                               float(exp.vertical.p_top),
                               float(cfg.base_temp),
                               int(cfg.hypsometric_opt))
        state = init_at_rest(cfg, coord, base)
        harness.install_geography(state, geo)
        update_diagnostics(state, cfg.hypsometric_opt)
        if getattr(state, "qv", None) is not None:
            z = np.asarray(state.height_half(), dtype=np.float64)
            z_col = z.mean(axis=(1, 2)) if z.ndim == 3 else z
            qv_col = np.maximum(wk82_sounding(z_col)[1], 1e-9)
            state.qv[...] = cp.asarray(
                np.broadcast_to(qv_col[:, None, None], state.qv.shape),
                dtype=state.qv.dtype)
            update_diagnostics(state, cfg.hypsometric_opt)
        driver = None
        if physics_driver_required(cfg):
            driver = initialize_physics(
                state, cfg,
                radiation_start_time=exp.start_time,
                radiation_latitude=geo.lat, radiation_longitude=geo.lon,
                noahmp_start_time=exp.start_time,
                noahmp_latitude=geo.lat, noahmp_longitude=geo.lon)
        if boundaries is not None:
            from woof.ingest.lateral_bc import (
                attach_streaming_lateral_boundaries)
            attach_streaming_lateral_boundaries(state, boundaries)
        return state, driver

    return builder


# --------------------------------------------------------------------------
# preparation and the stores
# --------------------------------------------------------------------------

@dataclass
class CaseStores:
    """Everything the streamed loop needs, and nothing device-resident."""

    cfg: Any
    grid: Any
    store: dict
    geo_store: dict
    scalars: dict
    boundaries: Any
    start_time: datetime
    forcing_times: tuple
    lat: np.ndarray
    lon: np.ndarray
    terrain: np.ndarray
    coord: Any
    store_bytes: int
    geo_bytes: int
    #: Whether ``store[REFL_KEY]`` holds a field a microphysics call wrote.
    #:
    #: ``"absent"``    the store carries no reflectivity slot at all.
    #: ``"primed"``   the slot exists, sized and zeroed, and NOTHING has
    #:                written it yet -- the state it is in on a domain past
    #:                the resident ceiling, which cannot take the settling
    #:                step that would compute the analysis-time field.
    #: ``"computed"`` a microphysics call has written it: the settling step
    #:                did, or a sweep that carried ``refl_10cm_due`` has.
    #:
    #: :func:`composite_reflectivity` reads the slot only in the last state.
    #: Zeros are the trap this field exists to close: a primed slot rendered
    #: as dBZ is a domain-wide 0 dBZ field, which is a plausible weak-echo
    #: product and not a failure anybody would notice.
    refl_stash: str = "absent"

    def reflectivity_swept(self) -> None:
        """Record that a sweep carrying ``refl_10cm_due`` has just run.

        One line at the call site instead of a spelling to remember, and the
        only transition that exists: a primed slot becomes a computed one
        the moment a sweep writes every tile's window into it, and it never
        goes back -- :func:`reflectivity_run_kwargs` puts the keyword on
        EVERY step of every sweep it configures, so there is no later sweep
        that leaves the slot behind.  A case whose store has no slot stays
        ``"absent"``: nothing to mark, and marking it would make
        :func:`composite_reflectivity` claim a field that is not there.
        """
        if self.refl_stash != "absent":
            self.refl_stash = "computed"


def prepare_low_water(config_path: str, *, verbose=print):
    """``prepare_real_case``, reordered so ONE forcing time is ever resident.

    THE MEASUREMENT THAT MOTIVATES THIS.  Instrumented on this case at
    480x384x49, ``runtime.prepare_real_case``'s device pool goes:

        interp[0]            0.000 -> 1.732 GiB
        initialize_real[0]   1.732 -> 3.151 GiB   (2.076 GiB RETAINED)
        interp[k>0]          2.421 -> 4.075 GiB
        initialize_real[k>0] 4.075 -> 5.528 GiB   <- the peak

    One ingest costs 3.151 GiB (17.1 kB per mass cell).  The peak is 5.528
    (30.0 kB/cell) and the whole difference is that ``initial_result`` and
    ``initial_met`` -- the START time's full-domain state and its
    interpolated analysis, 11.3 kB/cell of device memory -- are HELD for the
    entire loop while every later time is ingested underneath them.  That is
    correct and cheap on a domain that fits.  On a domain sized for the
    streaming lane it is the binding constraint, and it binds BEFORE the
    forecast does: at 35.9 kB/cell the largest preparable square on a 32 GiB
    card is about 910, while the measured vanilla FORECAST ceiling for the
    same physics is 928.  The ingest, not the integration, is what refuses
    first.

    The fix needs no new numerics, only an order.  ``StateBoundaryFrames``
    keeps four ``spec_bdy_width`` perimeter strips per time -- host memory,
    O(perimeter) -- and nothing else about a forcing time survives it except
    at the start time.  So the later times are ingested FIRST, each one
    released before the next is built, their perimeter snapshots stacked on
    the host; then the start time is ingested LAST and is the only state ever
    retained.  The snapshots are fed to the accumulator in TIME order, so the
    intervals and their tendencies are exactly the ones the in-order loop
    would have built -- the accumulator is a pure function of the snapshot
    sequence, which is why reordering the WORK cannot reorder the ANSWER.

    Everything after the loop -- soil, landuse, the physics driver, the
    time-zero surface diagnostics -- is ``prepare_real_case``'s own sequence,
    unchanged and in the same order, because it is O(domain) once rather than
    O(domain) per forcing time.
    """
    import time

    import cupy as cp

    from woof.case_data import load_experiment_case
    from woof.core.diagnostics import update_diagnostics
    from woof.core.landuse import (initialize_landuse,
                                    reconciled_soil_category,
                                    ruc_fractional_seaice,
                                    usemonalb_landuse_inputs)
    from woof.core.noah import noah_initial_snow_albedo
    from woof.core.physics import initialize_physics
    from woof.ingest.horiz import interpolate_era5_to_lambert
    from woof.ingest.lateral_bc import (LateralBoundaries,
                                         attach_lateral_boundaries,
                                         build_lateral_interval_from_sides,
                                         domain_boundary_snapshot,
                                         extract_lateral_side)
    from woof.ingest.preflight import build_input_catalog
    from woof.ingest.cpu_backend import host_step_workers
    from woof.ingest.preprocess_backend import (CudaPreprocessBackend,
                                                 release_backend_memory)
    from woof.ingest.real import initialize_real
    from woof.ingest.ruc_soil import preprocess_land_surface_soil
    from woof.ingest.soil_downscale import soil_mesh_plan_from_case
    from woof.ingest.soil_contract import MAPPED_SOIL_TEMPERATURE
    from woof.runtime import (GeogSelection, PreparedRealCase,
                               _cached_static_build, experiment_grid,
                               forcing_schedule, forcing_snapshots,
                               monthly_interp_to_date, radiation_scheme_ids,
                               vertical_coord_for)

    exp, data = load_experiment_case(config_path)
    dc = exp.root
    cfg = dc.run
    grid = experiment_grid(exp, data)
    geog_selection = GeogSelection.from_case_data(data, domain_id=dc.grid_id)
    from woof.static.lake import with_lake_statics
    geog_selection = with_lake_statics(geog_selection, cfg)
    catalog = build_input_catalog(data)
    snapshots = forcing_snapshots(data, catalog)
    times = forcing_schedule(exp, data, snapshots)
    start_time = exp.start_time
    if times[0] != start_time:
        raise RealCaseError("forcing_times must begin at start_time")

    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()
    t0 = time.perf_counter()
    static = _cached_static_build(grid, data.geog_root,
                                  selection=geog_selection)
    landuse_attrs = geog_selection.landuse_global_attrs()
    coord = vertical_coord_for(exp.vertical, cfg.nz)
    landmask = np.asarray(static["LANDMASK"]) >= 0.5
    if verbose:
        verbose(f"    statics built in {time.perf_counter() - t0:.1f} s "
                f"({sum(np.asarray(v).nbytes for v in static.values())/2**30:.2f}"
                f" GiB host)")

    backend = CudaPreprocessBackend()
    frames: dict = {}
    init_kwargs = dict(source_orography=None, p_top=exp.vertical.p_top,
                       sfcp_to_sfcp=data.sfcp_to_sfcp)
    order = tuple(times[1:]) + (times[0],)
    high_water = 0
    for valid_time in order:
        met = interpolate_era5_to_lambert(
            snapshots[valid_time], grid, source_orography_catalog=catalog,
            target_landmask=landmask)
        result = initialize_real(met, cfg, coord, static["HGT_M"],
                                 grid=grid, **init_kwargs)
        f, e = grid.coriolis_m()
        sina, cosa = grid.rotation_m()
        result.state.set_map_coriolis(grid.mapfac_m(), grid.mapfac_u(),
                                      grid.mapfac_v(), f, e,
                                      sina=sina, cosa=cosa)
        # Only the four perimeter strips survive this line.  Keeping the
        # whole coupled snapshot instead (which is what
        # ``StateBoundaryFrames.add_state`` would be handed) is six
        # full-domain float64 host arrays per forcing time -- 2.1 GiB each at
        # 1024^2 -- for the sake of 121 B per boundary cell.
        snapshot = domain_boundary_snapshot(result.state)
        frames[valid_time] = {
            side: extract_lateral_side(snapshot, side, cfg.spec_bdy_width)
            for side in ("west", "east", "south", "north")}
        del snapshot
        high_water = max(high_water, int(pool.total_bytes()))
        if valid_time == start_time:
            initial_met, initial_result = met, result
            break
        del met, result
        release_backend_memory(backend)
        pool.free_all_blocks()
        if verbose:
            verbose(f"    forcing {valid_time}: perimeter kept, state "
                    f"released; device pool {pool.total_bytes()/2**30:.2f} "
                    f"GiB")

    seconds = [(t - times[0]).total_seconds() for t in times]
    boundaries = LateralBoundaries(
        tuple(build_lateral_interval_from_sides(
            frames[times[k]], frames[times[k + 1]],
            start_seconds=seconds[k], end_seconds=seconds[k + 1])
            for k in range(len(times) - 1)),
        cfg.spec_bdy_width, cfg.spec_zone, cfg.relax_zone)
    del frames
    state = initial_result.state
    attach_lateral_boundaries(state, boundaries)

    soil_fields = dict(initial_met.fields)
    reconciled_soil_type = reconciled_soil_category(
        static["LU_INDEX"], soil_type=static["SCT_DOM"],
        xice=soil_fields.get("XICE", 0.0),
        iswater=int(landuse_attrs["ISWATER"]),
        islake=int(landuse_attrs["ISLAKE"]),
        isice=int(landuse_attrs["ISICE"]),
        soil_temperature=soil_fields.get(MAPPED_SOIL_TEMPERATURE,
                                         soil_fields.get("ST000007")),
        sst=soil_fields.get("SST"))
    lat, lon = grid.latlon_mass()
    # Task #168's water-temperature assembly: the same call the mainline
    # ERA5 mapping makes, fed this preparer's own full-domain fields.
    # Without it the soil router refuses the raw SST/SKINTEMP pair by name.
    water, water_policy = assembled_water_temperature(
        soil_fields, static=static, landuse_attrs=landuse_attrs, data=data,
        source_snapshot=snapshots[start_time], target_lat=lat,
        target_lon=lon, route=WATER_ROUTE_LOW_WATER,
        workers=host_step_workers(backend))
    from woof.core.landuse import (
        ruc_fractional_seaice as _ruc_fractional_seaice,
        seaice_threshold_of as _seaice_threshold_of)
    soil = preprocess_land_surface_soil(
        soil_fields, sf_surface_physics=int(cfg.sf_surface_physics),
        fractional_seaice=_ruc_fractional_seaice(cfg),
        seaice_threshold=_seaice_threshold_of(cfg),
        soil_type=reconciled_soil_type,
        deep_soil_temperature=static["TMN"],
        landmask=static["LANDMASK"], terrain=None, source_orography=None,
        water_temperature=water.values,
        water_temperature_policy=water_policy,
        # Same source, same interpolation, same defect: the tiles
        # preparers get the sub-source-cell reconstitution too, or
        # they stop being bit-comparable with the mainline route.
        soil_mesh=soil_mesh_plan_from_case(
            snapshots[start_time], grid, data),
        route=WATER_ROUTE_LOW_WATER)
    vegfra = 100.0 * monthly_interp_to_date(static["GREENFRAC"], start_time)
    from woof.core.landuse import surface_leaf_area
    lai = surface_leaf_area(cfg, static["LAI12M"], start_time)
    update_diagnostics(state, cfg.hypsometric_opt)
    radiation = None
    if radiation_scheme_ids(cfg) == (4, 4):
        from woof.core.rrtmgp import RRTMGPRadiation
        radiation = RRTMGPRadiation(start_time, lat, lon,
                                    trace_gas_overrides=None)
    landuse = initialize_landuse(
        static["LU_INDEX"], soil_type=reconciled_soil_type,
        landmask=static["LANDMASK"], snow=soil.snow_water, xice=soil.xice,
        valid_time=start_time,
        cen_lat=float(getattr(grid, "cen_lat", np.mean(lat))),
        mminlu=str(landuse_attrs["MMINLU"]),
        iswater=int(landuse_attrs["ISWATER"]),
        islake=int(landuse_attrs["ISLAKE"]),
        isice=int(landuse_attrs["ISICE"]),
        soil_temperature=soil.soil_temperature,
        # As woof/runtime.py: the sea-ice branch and usemonalb's monthly
        # ALBBCK follow the configuration; off, the call is unchanged.
        fractional_seaice=ruc_fractional_seaice(cfg),
        sea_ice_cells=getattr(soil, "sea_ice_cells", None),
        **usemonalb_landuse_inputs(cfg, static, start_time))
    from woof.ingest.ruc_mosaic import ruc_mosaic_physics_inputs
    from woof.ingest.lake_physics import lake_physics_inputs
    driver = initialize_physics(
        state, cfg, landuse=landuse, tsk=soil.tsk,
        soil_temperature=soil.soil_temperature,
        soil_moisture=soil.soil_moisture,
        liquid_moisture=soil.liquid_moisture,
        ivgtyp=static["LU_INDEX"], isltyp=static["SCT_DOM"],
        vegfra=vegfra, tmn=soil.deep_soil_temperature,
        xice=soil.xice, snow=soil.snow_water, snow_depth=soil.snow_depth,
        sst=soil_fields.get("SST", soil.tsk),
        radiation=radiation, radiation_start_time=start_time,
        radiation_latitude=lat, radiation_longitude=lon,
        **ruc_mosaic_physics_inputs(
            cfg, static, landuse_attrs=landuse_attrs, xice=soil.xice,
            fractional_seaice=ruc_fractional_seaice(cfg)),
        **lake_physics_inputs(cfg, static))
    from woof.core.landuse import surface_snow_albedo
    driver.fields["snoalb"][...] = cp.asarray(
        surface_snow_albedo(cfg, static, driver.noah_params),
        dtype=cp.float32)
    driver.fields["lai"][...] = cp.asarray(lai, dtype=cp.float32)
    driver.fields["shdmin"][...] = cp.asarray(
        100.0 * static["GREENFRAC"].min(axis=0), dtype=cp.float32)
    driver.fields["shdmax"][...] = cp.asarray(
        100.0 * static["GREENFRAC"].max(axis=0), dtype=cp.float32)
    met0 = initial_met.fields
    driver.fields["psfc"][...] = cp.asarray(
        initial_result.surface_pressure, dtype=cp.float32)
    driver.fields["t2"][...] = met0["T2"]
    driver.fields["q2"][...] = cp.asarray(
        initial_result.surface_qv, dtype=cp.float32)
    driver.fields["th2"][...] = (driver.fields["t2"]
                                 * (cp.float32(100000.0)
                                    / driver.fields["psfc"])
                                 ** cp.float32(287.0 / 1004.0))
    driver.fields["u10"][...] = 0.5 * (met0["U10"][:, :-1] + met0["U10"][:, 1:])
    driver.fields["v10"][...] = 0.5 * (met0["V10"][:-1] + met0["V10"][1:])
    cp.cuda.runtime.deviceSynchronize()
    high_water = max(high_water, int(pool.total_bytes()))
    cells = int(cfg.nx) * int(cfg.ny)
    if verbose:
        verbose(f"    prepared {cfg.nx}x{cfg.ny}x{cfg.nz} from "
                f"{len(times)} forcing times in "
                f"{time.perf_counter() - t0:.1f} s")
        verbose(f"    ingest device high-water {high_water / 2**30:.2f} GiB "
                f"({high_water / cells:.0f} B per mass cell)")
    prep = PreparedRealCase(
        cfg=cfg, grid=grid, static_fields=static,
        initial_result=initial_result, final_analysis=initial_met,
        initial_snow_water_kgm2=np.array(soil.snow_water, dtype=np.float64,
                                         copy=True),
        forcing_times=times, geog_selection=geog_selection)
    return prep, exp, high_water


def _copy_rows(dst_state, src_state, slab, *, nz, ny, nx, inventory_fns):
    """Copy one slab's owned rows into a full-domain state, on the device.

    Staggering-aware through :func:`tilestream.realdata.window_rows`, which
    is the ONE place that decides which rows a v-staggered slab owns and the
    one line that can be wrong while every field stays finite and smooth.
    """
    from tilestream import gather as _gather
    from tilestream import realdata as _realdata

    moved = 0
    for take in inventory_fns:
        src = take(src_state)
        dst = take(dst_state)
        if set(src) != set(dst):
            raise RealCaseError(
                f"slab inventory {sorted(set(src) ^ set(dst))} does not match "
                "the domain's")
        for name, s in src.items():
            kind = _gather.classify(s.shape, nz, slab.built_rows, nx,
                                    layers_ok=True)
            dlo, dhi, slo, shi = _realdata.window_rows(kind, slab)
            dst[name][..., dlo:dhi, :] = s[..., slo:shi, :]
            moved += int(s[..., slo:shi, :].nbytes)
    return moved


def _assemble_2d(store, arrays, slab, *, nz, ny, nx):
    """Accumulate a slab's 2-D host/device fields into full-domain buffers."""
    import cupy as cp

    from tilestream import gather as _gather
    from tilestream import realdata as _realdata

    for name, value in arrays.items():
        a = np.asarray(cp.asnumpy(value)) if isinstance(value, cp.ndarray) \
            else np.asarray(value)
        if a.ndim != 2:
            continue
        kind = _gather.classify(a.shape, nz, slab.built_rows, nx,
                                layers_ok=True)
        dlo, dhi, slo, shi = _realdata.window_rows(kind, slab)
        if name not in store:
            store[name] = np.empty(
                (ny + (1 if kind == "v" else 0), a.shape[-1]), dtype=a.dtype)
        store[name][dlo:dhi, :] = a[slo:shi, :]


#: Route names the water-temperature assembly and the soil router carry,
#: so a refusal in either seam can be attributed to the exact preparer.
WATER_ROUTE_SLABBED = "the tiles slabbed ERA5 route"
WATER_ROUTE_LOW_WATER = "the tiles low-water ERA5 route"


def assembled_water_temperature(fields, *, static, landuse_attrs, data,
                                source_snapshot, target_lat, target_lon,
                                route, workers=None):
    """The finished water temperature this route hands the soil router.

    The #168 lake fix made a raw SST-beside-SKINTEMP pair a refusal at
    soil preprocessing (``woof.ingest.water_temperature``:
    ``require_assembled_water_temperature``): choosing between two
    differently-mapped fields cell by cell is what painted lakes as a
    quilt of both.  The mainline ERA5 route adopted the assembly inside
    ``interpolate_era5_to_lambert``; the tiles preparers never did, so
    every streamed real-case run on an ERA5 source carrying SST was
    refused at prepare -- the 2.0 battery's twin_a row.

    This is the SAME call the mapping makes (woof/ingest/horiz.py), fed
    the same inputs at full-domain scope: the mapped SST/SKINTEMP pair,
    the raw source SST with its own lat/lon for the per-body donor
    search, and route statics built from the full-domain LANDMASK and
    LU_INDEX.  Full-domain on purpose -- the slabbed preparer maps by row
    slab, and statics sliced to a slab would label a body crossing the
    slab seam as two bodies and could hand its halves different
    providers.  Because the slab mapping is bit-exact against the
    monolithic mapping (tilestream.realdata's proof), assembling the
    stitched full-domain fields here yields byte-for-byte the mainline
    route's decisions.

    Returns ``(assembly, policy)``; the caller forwards
    ``assembly.values``, the policy and the route into
    ``preprocess_land_surface_soil``, which is the receipt-implies-
    consumption shape the water-temperature gate demands.
    """
    from woof.ingest.horiz import _as_host_float64
    from woof.ingest.water_temperature import (
        WaterTemperatureStatics, announce_water_temperature,
        assemble_for_route, resolve_water_temperature_policy)

    policy = resolve_water_temperature_policy(data)
    statics = WaterTemperatureStatics.for_route(
        route=route, policy=policy, landmask=static["LANDMASK"],
        lu_index=static["LU_INDEX"], landuse_attrs=landuse_attrs)
    if "SKINTEMP" not in fields:
        raise RealCaseError(
            f"{route}: the water-temperature assembly needs a mapped "
            "SKINTEMP and this source carries none")
    source_sst = source_snapshot.fields.get("SST")
    assembly = assemble_for_route(
        statics,
        mapped_sst=(None if "SST" not in fields
                    else _as_host_float64(fields["SST"])),
        mapped_skin=_as_host_float64(fields["SKINTEMP"]),
        source_sst=(None if source_sst is None
                    else _as_host_float64(source_sst)),
        source_lat=np.asarray(source_snapshot.latitude, dtype=np.float64),
        source_lon=np.asarray(source_snapshot.longitude, dtype=np.float64),
        target_lat=np.asarray(target_lat, dtype=np.float64),
        target_lon=np.asarray(target_lon, dtype=np.float64),
        workers=workers)
    announce_water_temperature(assembly.receipt,
                               scope=np.asarray(target_lat).shape)
    return assembly, policy


def prepare_slabbed(config_path: str, *, rows_per_slab: int = 64,
                    verbose=print):
    """The ingest, by ROW SLABS, so the domain stops capping itself.

    :func:`prepare_low_water` fixed the ORDER of the forcing loop and got the
    peak from 35.9 to 21.0 kB per mass cell.  What it did not fix is that one
    forcing time's ingest is still built at FULL DOMAIN WIDTH AND HEIGHT: the
    ERA5 interpolation writes ``(nlevel, ny, nx)`` and ``initialize_real``
    allocates a full ``DomainState``, so 21.0 kB/cell is a floor no ordering
    can move.  On a 32 GiB card that floor is about 1200^2 -- and on a SHARED
    card it is whatever is left over, which is why this exists: MEASURED on
    the rented boxes, a 960^2 preparation that needs 18 GiB is refused
    outright whenever another lane holds half the card, while the streamed
    FORECAST it feeds needs about 6 GiB and coexists happily.

    ``tilestream.realdata`` already proved the piece that makes this legal --
    a row slab of ``interpolate_era5_to_lambert`` + ``initialize_real``, built
    on ``window_grid``'s exact sub-rectangle of the parent's projection, is
    BIT-EXACT against the monolithic ingest at every slab height down to a
    single row, over the carriers AND the geography.  This is that loop with
    the two things a physics-on real case needs added:

      * the slabs are written into ONE persistent full-domain ``DomainState``
        on the device, not into a host store, because everything after the
        ingest -- ``domain_boundary_snapshot``, ``initialize_physics``,
        ``update_diagnostics`` -- wants a resident state and costs O(domain)
        ONCE rather than per forcing time.  That state is 2365.8 B/mass cell
        of carriers plus 824.0 of geography: 2.8 GiB at 960^2, against the
        18.0 GiB the full-width ingest peaked at.
      * the 2-D outputs the soil/landuse/driver stages read -- the met
        surface fields, ``surface_pressure`` and ``surface_qv`` -- are
        assembled row by row from the same slabs, with the same
        staggering rule.

    The slab carries ``realdata.SLAB_Y_HALO`` = one mass row below its own
    first, and the reason is exactly one line of ``initialize_real``:
    ``_pressure_at_v`` averages the mass-point dry pressure onto y faces, so
    the v-face target at a slab's first row needs the mass row beneath it.
    Nothing else in the ingest reads across a row.
    """
    import time

    import cupy as cp

    from woof.case_data import load_experiment_case
    from woof.core.diagnostics import update_diagnostics
    from woof.core.landuse import (initialize_landuse,
                                    reconciled_soil_category,
                                    ruc_fractional_seaice,
                                    usemonalb_landuse_inputs)
    from woof.core.noah import noah_initial_snow_albedo
    from woof.core.physics import initialize_physics
    from woof.ingest.horiz import interpolate_era5_to_lambert
    from woof.ingest.lateral_bc import (LateralBoundaries,
                                         attach_lateral_boundaries,
                                         build_lateral_interval_from_sides,
                                         domain_boundary_snapshot,
                                         extract_lateral_side)
    from woof.ingest.preflight import build_input_catalog
    from woof.ingest.cpu_backend import host_step_workers
    from woof.ingest.preprocess_backend import (CudaPreprocessBackend,
                                                 release_backend_memory)
    from woof.ingest.real import initialize_real
    from woof.ingest.ruc_soil import preprocess_land_surface_soil
    from woof.ingest.soil_downscale import soil_mesh_plan_from_case
    from woof.ingest.soil_contract import MAPPED_SOIL_TEMPERATURE
    from woof.runtime import (GeogSelection, PreparedRealCase,
                               _cached_static_build, experiment_grid,
                               forcing_schedule, forcing_snapshots,
                               monthly_interp_to_date, radiation_scheme_ids,
                               vertical_coord_for)
    from tilestream import driver as _driver
    from tilestream import gather as _gather
    from tilestream import realdata as _realdata

    exp, data = load_experiment_case(config_path)
    dc = exp.root
    cfg = dc.run
    grid = experiment_grid(exp, data)
    geog_selection = GeogSelection.from_case_data(data, domain_id=dc.grid_id)
    from woof.static.lake import with_lake_statics
    geog_selection = with_lake_statics(geog_selection, cfg)
    catalog = build_input_catalog(data)
    snapshots = forcing_snapshots(data, catalog)
    times = forcing_schedule(exp, data, snapshots)
    start_time = exp.start_time
    nz, ny, nx = int(cfg.nz), int(cfg.ny), int(cfg.nx)

    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()
    t0 = time.perf_counter()
    static = _cached_static_build(grid, data.geog_root,
                                  selection=geog_selection)
    landuse_attrs = geog_selection.landuse_global_attrs()
    coord = vertical_coord_for(exp.vertical, cfg.nz)
    landmask = np.asarray(static["LANDMASK"]) >= 0.5
    terrain = np.asarray(static["HGT_M"])
    if verbose:
        verbose(f"    statics built in {time.perf_counter() - t0:.1f} s")

    slabs = _realdata.plan_row_slabs(ny, int(rows_per_slab))
    if verbose:
        verbose(f"    {len(slabs)} row slabs of <= {rows_per_slab} rows "
                f"(+{_realdata.SLAB_Y_HALO} halo row)")

    # The one persistent full-domain allocation.  Built with the case's own
    # coordinate and base state, exactly as initialize_real builds each
    # slab's, so every setup array the slabs do not carry is already right.
    from woof.ingest.real import _make_real_base
    base = _make_real_base(coord, terrain.astype(np.float64),
                           float(exp.vertical.p_top), float(cfg.base_temp),
                           int(cfg.hypsometric_opt))
    from woof.core.state import init_at_rest as _init_at_rest
    state = _init_at_rest(cfg, coord, base)
    inventory_fns = (_gather.inventory, _driver.geography_inventory)

    backend = CudaPreprocessBackend()
    frames: dict = {}
    order = tuple(times[1:]) + (times[0],)
    high_water = 0
    met2d: dict = {}
    psfc = np.empty((ny, nx), dtype=np.float64)
    qv2 = np.empty((ny, nx), dtype=np.float64)

    for valid_time in order:
        is_start = valid_time == start_time
        for slab in slabs:
            gw = _realdata.window_grid(grid, 0, slab.g0, nx, slab.built_rows)
            met = interpolate_era5_to_lambert(
                snapshots[valid_time], gw,
                source_orography_catalog=catalog,
                target_landmask=landmask[slab.g0:slab.g1])
            slab_cfg = replace(cfg, ny=slab.built_rows)
            result = initialize_real(
                met, slab_cfg, coord, terrain[slab.g0:slab.g1],
                source_orography=None, p_top=exp.vertical.p_top,
                # The two routes did not reach it.
                #  ``gw`` is the SLAB's window grid, not
                # the full domain's -- which is the correct carrier here:
                # this initialize_real builds exactly those rows, so the
                # mass-point lat/lon the WIF interpolation needs are the
                # slab's own.  Passing the full grid would interpolate the
                # global climatology onto the wrong rows.
                sfcp_to_sfcp=data.sfcp_to_sfcp, grid=gw)
            _copy_rows(state, result.state, slab, nz=nz, ny=ny, nx=nx,
                       inventory_fns=inventory_fns)
            if is_start:
                _assemble_2d(met2d, dict(met.fields), slab,
                             nz=nz, ny=ny, nx=nx)
                dlo, dhi, slo, shi = _realdata.window_rows("mass", slab)
                psfc[dlo:dhi] = np.asarray(result.surface_pressure)[slo:shi]
                qv2[dlo:dhi] = np.asarray(result.surface_qv)[slo:shi]
            high_water = max(high_water, int(pool.total_bytes()))
            del met, result
            release_backend_memory(backend)
        f, e = grid.coriolis_m()
        sina, cosa = grid.rotation_m()
        state.set_map_coriolis(grid.mapfac_m(), grid.mapfac_u(),
                               grid.mapfac_v(), f, e, sina=sina, cosa=cosa)
        snapshot = domain_boundary_snapshot(state)
        frames[valid_time] = {
            side: extract_lateral_side(snapshot, side, cfg.spec_bdy_width)
            for side in ("west", "east", "south", "north")}
        del snapshot
        high_water = max(high_water, int(pool.total_bytes()))
        pool.free_all_blocks()
        if verbose:
            verbose(f"    forcing {valid_time}: {len(slabs)} slabs, "
                    f"perimeter kept; pool {pool.total_bytes()/2**30:.2f} GiB")

    seconds = [(t - times[0]).total_seconds() for t in times]
    boundaries = LateralBoundaries(
        tuple(build_lateral_interval_from_sides(
            frames[times[k]], frames[times[k + 1]],
            start_seconds=seconds[k], end_seconds=seconds[k + 1])
            for k in range(len(times) - 1)),
        cfg.spec_bdy_width, cfg.spec_zone, cfg.relax_zone)
    del frames
    attach_lateral_boundaries(state, boundaries)

    soil_fields = dict(met2d)
    reconciled_soil_type = reconciled_soil_category(
        static["LU_INDEX"], soil_type=static["SCT_DOM"],
        xice=soil_fields.get("XICE", 0.0),
        iswater=int(landuse_attrs["ISWATER"]),
        islake=int(landuse_attrs["ISLAKE"]),
        isice=int(landuse_attrs["ISICE"]),
        soil_temperature=soil_fields.get(MAPPED_SOIL_TEMPERATURE,
                                         soil_fields.get("ST000007")),
        sst=soil_fields.get("SST"))
    lat, lon = grid.latlon_mass()
    # Task #168's water-temperature assembly, on the stitched full-domain
    # fields: the slab mapping is bit-exact against the monolithic one, so
    # this reproduces the mainline ERA5 route's per-body decisions.
    # Without it the soil router refuses the raw SST/SKINTEMP pair by name.
    water, water_policy = assembled_water_temperature(
        soil_fields, static=static, landuse_attrs=landuse_attrs, data=data,
        source_snapshot=snapshots[start_time], target_lat=lat,
        target_lon=lon, route=WATER_ROUTE_SLABBED,
        workers=host_step_workers(backend))
    from woof.core.landuse import (
        ruc_fractional_seaice as _ruc_fractional_seaice,
        seaice_threshold_of as _seaice_threshold_of)
    soil = preprocess_land_surface_soil(
        soil_fields, sf_surface_physics=int(cfg.sf_surface_physics),
        fractional_seaice=_ruc_fractional_seaice(cfg),
        seaice_threshold=_seaice_threshold_of(cfg),
        soil_type=reconciled_soil_type,
        deep_soil_temperature=static["TMN"],
        landmask=static["LANDMASK"], terrain=None, source_orography=None,
        water_temperature=water.values,
        water_temperature_policy=water_policy,
        # Same source, same interpolation, same defect: the tiles
        # preparers get the sub-source-cell reconstitution too, or
        # they stop being bit-comparable with the mainline route.
        soil_mesh=soil_mesh_plan_from_case(
            snapshots[start_time], grid, data),
        route=WATER_ROUTE_SLABBED)
    vegfra = 100.0 * monthly_interp_to_date(static["GREENFRAC"], start_time)
    from woof.core.landuse import surface_leaf_area
    lai = surface_leaf_area(cfg, static["LAI12M"], start_time)
    update_diagnostics(state, cfg.hypsometric_opt)
    radiation = None
    if radiation_scheme_ids(cfg) == (4, 4):
        from woof.core.rrtmgp import RRTMGPRadiation
        radiation = RRTMGPRadiation(start_time, lat, lon,
                                    trace_gas_overrides=None)
    landuse = initialize_landuse(
        static["LU_INDEX"], soil_type=reconciled_soil_type,
        landmask=static["LANDMASK"], snow=soil.snow_water, xice=soil.xice,
        valid_time=start_time,
        cen_lat=float(getattr(grid, "cen_lat", np.mean(lat))),
        mminlu=str(landuse_attrs["MMINLU"]),
        iswater=int(landuse_attrs["ISWATER"]),
        islake=int(landuse_attrs["ISLAKE"]),
        isice=int(landuse_attrs["ISICE"]),
        soil_temperature=soil.soil_temperature,
        # As woof/runtime.py: the sea-ice branch and usemonalb's monthly
        # ALBBCK follow the configuration; off, the call is unchanged.
        fractional_seaice=ruc_fractional_seaice(cfg),
        sea_ice_cells=getattr(soil, "sea_ice_cells", None),
        **usemonalb_landuse_inputs(cfg, static, start_time))
    from woof.ingest.ruc_mosaic import ruc_mosaic_physics_inputs
    from woof.ingest.lake_physics import lake_physics_inputs
    driver = initialize_physics(
        state, cfg, landuse=landuse, tsk=soil.tsk,
        soil_temperature=soil.soil_temperature,
        soil_moisture=soil.soil_moisture,
        liquid_moisture=soil.liquid_moisture,
        ivgtyp=static["LU_INDEX"], isltyp=static["SCT_DOM"],
        vegfra=vegfra, tmn=soil.deep_soil_temperature,
        xice=soil.xice, snow=soil.snow_water, snow_depth=soil.snow_depth,
        sst=soil_fields.get("SST", soil.tsk),
        radiation=radiation, radiation_start_time=start_time,
        radiation_latitude=lat, radiation_longitude=lon,
        **ruc_mosaic_physics_inputs(
            cfg, static, landuse_attrs=landuse_attrs, xice=soil.xice,
            fractional_seaice=ruc_fractional_seaice(cfg)),
        **lake_physics_inputs(cfg, static))
    from woof.core.landuse import surface_snow_albedo
    driver.fields["snoalb"][...] = cp.asarray(
        surface_snow_albedo(cfg, static, driver.noah_params),
        dtype=cp.float32)
    driver.fields["lai"][...] = cp.asarray(lai, dtype=cp.float32)
    driver.fields["shdmin"][...] = cp.asarray(
        100.0 * static["GREENFRAC"].min(axis=0), dtype=cp.float32)
    driver.fields["shdmax"][...] = cp.asarray(
        100.0 * static["GREENFRAC"].max(axis=0), dtype=cp.float32)
    driver.fields["psfc"][...] = cp.asarray(psfc, dtype=cp.float32)
    driver.fields["t2"][...] = cp.asarray(met2d["T2"], dtype=cp.float32)
    driver.fields["q2"][...] = cp.asarray(qv2, dtype=cp.float32)
    driver.fields["th2"][...] = (driver.fields["t2"]
                                 * (cp.float32(100000.0)
                                    / driver.fields["psfc"])
                                 ** cp.float32(287.0 / 1004.0))
    driver.fields["u10"][...] = cp.asarray(
        0.5 * (met2d["U10"][:, :-1] + met2d["U10"][:, 1:]), dtype=cp.float32)
    driver.fields["v10"][...] = cp.asarray(
        0.5 * (met2d["V10"][:-1] + met2d["V10"][1:]), dtype=cp.float32)
    cp.cuda.runtime.deviceSynchronize()
    high_water = max(high_water, int(pool.total_bytes()))
    cells = nx * ny
    if verbose:
        verbose(f"    prepared {nx}x{ny}x{nz} from {len(times)} forcing "
                f"times in {time.perf_counter() - t0:.1f} s")
        verbose(f"    ingest device high-water {high_water / 2**30:.2f} GiB "
                f"({high_water / cells:.0f} B per mass cell)")

    class _Result:
        pass

    res = _Result()
    res.state = state
    res.surface_pressure = psfc
    res.surface_qv = qv2
    prep = PreparedRealCase(
        cfg=cfg, grid=grid, static_fields=static, initial_result=res,
        final_analysis=None,
        initial_snow_water_kgm2=np.array(soil.snow_water, dtype=np.float64,
                                         copy=True),
        forcing_times=times, geog_selection=geog_selection)
    return prep, exp, high_water


def prepare(config_path: str, *, verbose=print):
    """``runtime.prepare_root_experiment_case``, with the numbers reported."""
    import time

    import cupy as cp

    from woof.case_data import load_experiment_case
    from woof.runtime import prepare_root_experiment_case

    exp, data = load_experiment_case(config_path)
    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()
    t0 = time.perf_counter()
    prep = prepare_root_experiment_case(exp, data)
    cp.cuda.runtime.deviceSynchronize()
    peak = int(pool.total_bytes())
    cells = int(prep.cfg.nx) * int(prep.cfg.ny)
    if verbose:
        verbose(f"    prepared {prep.cfg.nx}x{prep.cfg.ny}x{prep.cfg.nz} from "
                f"{len(prep.forcing_times)} forcing times in "
                f"{time.perf_counter() - t0:.1f} s")
        verbose(f"    ingest device high-water {peak / 2**30:.2f} GiB "
                f"({peak / cells:.0f} B per mass cell)")
    return prep, exp, peak


def release_prepared(prep, *, verbose=print) -> None:
    """Give the card back the PREPARED case, once the stores exist.

    ``build_stores`` cannot do this.  Its ``prep = None`` rebinds a LOCAL
    name; the caller still holds the same ``PreparedRealCase``, and through
    it ``initial_result.state`` -- a full resident ``DomainState``, its
    ``PhysicsDriver``'s domain-shaped Noah-MP / MYNN / RRTMGP arrays -- and
    ``final_analysis``, the ERA5 analysis interpolated to the model grid.
    All of it stayed on the card for the whole forecast, which is the exact
    opposite of the design's claim that only tiles are resident.

    MEASURED, 960x960x49 on a 23.52 GiB RTX 4090: with the prepared case
    retained, the FIRST tile step refused at tile 240 (window 280^2, pool
    23.33 GB) and again at tile 192 (window 232^2, pool 22.93 GB).  A 31%
    smaller window moving the pool by 1.7% is the signature of a large term
    that does not scale with the window.

    The dataclass is frozen, so the fields are cleared through
    ``object.__setattr__``; the object is unusable afterwards and the caller
    must drop its own reference too, which is why this returns None and the
    caller is expected to write ``prep = None`` on the next line.
    """
    import gc

    import cupy as cp

    pool = cp.get_default_memory_pool()

    def _state(tag):
        free, total = cp.cuda.runtime.memGetInfo()
        return (f"{tag}: pool live {pool.used_bytes() / 2**30:.2f} GiB, "
                f"pool held {pool.total_bytes() / 2**30:.2f} GiB, "
                f"device free {free / 2**30:.2f}/{total / 2**30:.2f} GiB")

    before = _state("before release")
    for field in ("initial_result", "final_analysis",
                  "initial_snow_water_kgm2"):
        if getattr(prep, field, None) is not None:
            object.__setattr__(prep, field, None)
    gc.collect()
    pool.free_all_blocks()
    if verbose:
        # THREE numbers, not one.  "pool live" is what cupy objects still
        # reference, "pool held" adds the blocks it is caching, and "device
        # free" adds everything allocated outside the pool.  A release that
        # moves the first two and not the third has not released anything a
        # tile can use, and one number could not tell them apart.
        verbose("    " + before)
        verbose("    " + _state("after release "))


def build_stores(prep, exp, *, verbose=print, on_state=None,
                 settle: bool = True,
                 reflectivity: bool = True) -> CaseStores:
    """Move the prepared domain off the device into pinned host stores.

    The carriers go into a pinned :mod:`tilestream.hoststore` block and the
    geography into a second one; between them they are everything
    ``run_tiled`` reads.  The device state is released before returning, so
    the forecast starts with the card empty apart from the tile buffers.

    ``reflectivity=True`` (the default, audit R-052) puts the ``refl_10cm``
    scratch slot in the store, which is the ONLY route by which a streamed
    run can publish the dBZ a scheme computes for itself -- mp=9, 18 and 50
    compute Z inside the scheme call and publish it through no operator, and
    mp=8 and 28 need a graupel-number shadow the restart classifies as
    rebuilt and never serialises.  A store that carries the slot must be
    swept by a run that WRITES it, which is what
    :func:`reflectivity_run_kwargs` arranges; a caller that sweeps such a
    store without those keywords is refused by ``TiledRun``'s inventory
    comparison, naming the key.  ``reflectivity=False`` is for a lane whose
    product is the transport rather than the weather.
    """
    import time

    import cupy as cp

    from tilestream import gather as _gather
    from tilestream import driver as _driver
    from tilestream import physics_inventory as physinv

    cfg = prep.cfg
    state = prep.initial_result.state

    # ONE STEP FIRST.  Two carriers are allocated lazily (Kain-Fritsch's
    # cumulus/w0avg above all), so a manifest taken at t=0 is short of the
    # set a later sweep will want to scatter.  physics_inventory says so;
    # this is the caller obeying it.
    #
    # AND IT IS THE ONE THING A DOMAIN PAST THE RESIDENT CEILING CANNOT DO.
    # MEASURED on an RTX 4090 (23.52 GiB) at 960x960x49, dx = 3 km,
    # full+MYNN+Noah-MP+RRTMGP: the slabbed ingest completed at a 15.55 GiB
    # high-water and this step then refused -- cupy OutOfMemoryError after
    # 24,429,406,208 bytes, inside RRTMGP's ``hydrometeor_paths``.  That
    # refusal is the vanilla ceiling, in the forecast's own log.
    #
    # ``settle=False`` skips it.  Nothing is taken on trust: ``run_tiled``
    # compares the tile buffer's carrier inventory against the store's and
    # raises ``TiledRunError`` naming the difference before it steps
    # anything, and the tile buffers DO warm up, so a rung that really
    # allocates a carrier lazily is refused loudly at the first sweep rather
    # than quietly dropping it.  At mp10 + MYNN + Noah-MP + RRTMGP with no
    # cumulus scheme there is no such carrier -- MEASURED at 192x192x49, the
    # inventory and the wrfout frame plan are identical either side of this
    # step -- so on this configuration skipping it changes nothing but the
    # peak.
    t0 = time.perf_counter()
    refl_stash = "absent"
    if settle:
        from woof.core.dycore import step as _step
        from woof.core.refl import consume_refl_10cm, refl_10cm_is_stashed
        # AND THE ANALYSIS-TIME REFLECTIVITY, in the same step.  This is
        # the one microphysics call this route makes on a DOMAIN-shaped
        # state, so it is the only chance the store has to hold a computed
        # ``refl_10cm`` before the first sweep -- and the f000 product is
        # rendered from exactly that store.  Every scheme writes the slot
        # here, including the three that compute Z inside the scheme call
        # and hand the finished array to ``stash_refl_10cm``.  The stash is
        # a one-frame handoff reference and is consumed immediately: the
        # numbers stay in the state-owned slot, which is what the gather
        # below copies, and leaving it parked would make the next
        # output-due call on this state raise "not consumed before reuse".
        _step(state, cfg, refl_10cm_due=bool(reflectivity))
        if reflectivity and refl_10cm_is_stashed(state):
            consume_refl_10cm(state)
            refl_stash = "computed"
        cp.cuda.runtime.deviceSynchronize()
        if verbose:
            verbose(f"    one monolithic step to settle the lazy carriers "
                    f"({time.perf_counter() - t0:.1f} s)"
                    + ("; REFL_10CM computed for f000" if reflectivity
                       else ""))
    else:
        if reflectivity:
            # PAST THE RESIDENT CEILING: no domain-shaped step is possible,
            # so the slot is ALLOCATED and left empty for the sweep to fill.
            # It has to exist before the store is sized -- a slot that
            # appears later has nowhere to land and ``TiledRun`` refuses the
            # sweep -- and it has to be MARKED, because an empty slot
            # rendered as dBZ is a domain-wide 0 dBZ field that reads like
            # weak echo rather than like a missing field.
            from woof.core.streaming import prime_refl_10cm
            prime_refl_10cm(state, cfg)
            refl_stash = "primed"
        if verbose:
            verbose("    settling step SKIPPED (settle=False): this domain "
                    "is past the resident ceiling, and run_tiled's "
                    "inventory match is the guard that a lazy carrier "
                    "cannot slip through"
                    + ("; the REFL_10CM slot is primed EMPTY and the first "
                       "sweep carrying refl_10cm_due fills it"
                       if reflectivity else ""))

    # The one place a DOMAIN-shaped resident state exists in a streamed run.
    # A wrfout frame plan and the DomainSetup a frame's statics come from can
    # only be read HERE -- after the settling step, so the plan sees the lazy
    # carriers, and before the state is released.  The caller supplies the
    # callback rather than this function knowing about output, because a
    # forecast that writes no history should not pay for either.
    if on_state is not None:
        on_state(state, cfg)

    # WITH the reflectivity stash (audit R-052).  ``carrier_inventory`` is
    # the RESTART manifest and deliberately omits ``refl_10cm``, a pure
    # diagnostic; this store's whole product is that diagnostic, and for
    # mp=9/18/50 -- which compute their own dBZ inside the scheme call and
    # publish it through no operator -- the stash is the ONLY route to it.
    # The slot is present here exactly when ``reflectivity`` asked for it:
    # computed by the settling step above, or primed empty for the sweep.
    inv = (physinv.carrier_inventory_with_refl(state) if reflectivity
           else physinv.carrier_inventory(state))
    geo_inv = _driver.geography_inventory(state)
    scalars = physinv.carrier_scalars(state)

    store: dict[str, np.ndarray] = {}
    for name, arr in inv.items():
        host = _gather.pinned_empty(arr.shape, arr.dtype)
        host[...] = cp.asnumpy(arr)
        store[name] = host
    geo_store: dict[str, np.ndarray] = {}
    for name, arr in geo_inv.items():
        host = _gather.pinned_empty(arr.shape, arr.dtype)
        host[...] = cp.asnumpy(arr) if isinstance(arr, cp.ndarray) \
            else np.asarray(arr)
        geo_store[name] = host

    store_bytes = sum(int(a.nbytes) for a in store.values())
    geo_bytes = sum(int(a.nbytes) for a in geo_store.values())
    cells = int(cfg.nx) * int(cfg.ny) * int(cfg.nz)
    if verbose:
        verbose(f"    store {len(store)} carriers, "
                f"{store_bytes / 2**30:.2f} GiB pinned "
                f"({store_bytes / cells:.1f} B/cell); geography "
                f"{len(geo_store)} arrays, {geo_bytes / 2**30:.2f} GiB")

    grid = prep.grid
    forcing_times = tuple(prep.forcing_times)
    lat, lon = grid.latlon_mass()
    terrain = np.asarray(prep.static_fields["HGT_M"], dtype=np.float32)
    boundaries = state.lateral_boundaries
    from woof.runtime import vertical_coord_for
    coord = vertical_coord_for(exp.vertical, cfg.nz)

    # RELEASE THE PREPARED CASE, from the object rather than from this
    # function's own name.  ``prep = None`` rebinds a LOCAL: the caller still
    # holds the same PreparedRealCase, so ``initial_result.state`` -- a full
    # resident DomainState, its PhysicsDriver's domain-shaped Noah-MP /
    # MYNN / RRTMGP arrays, and ``final_analysis``, the ERA5 interpolated to
    # the model grid -- stayed on the card for the whole forecast.
    #
    # MEASURED, 960x960x49 on a 23.52 GiB RTX 4090: with it retained, the
    # first tile step refused at both tile 240 (window 280^2, pool 23.33 GB)
    # AND tile 192 (window 232^2, pool 22.93 GB).  A 31% smaller window
    # moving the pool by 1.7% is the signature of a large term that does not
    # scale with the window -- which is the definition of a leak in a design
    # whose whole claim is that only tiles are resident.
    # PreparedRealCase is a frozen dataclass, so this function cannot empty
    # it and must not try; the CALLER owns the release and
    # :func:`release_prepared` is what it should call.
    del state, inv, geo_inv
    prep = None
    cp.get_default_memory_pool().free_all_blocks()

    return CaseStores(
        cfg=cfg, grid=grid, store=store, geo_store=geo_store,
        scalars=scalars, boundaries=boundaries, start_time=exp.start_time,
        forcing_times=forcing_times,
        lat=np.asarray(lat, dtype=np.float32),
        lon=np.asarray(lon, dtype=np.float32),
        terrain=terrain, coord=coord,
        store_bytes=store_bytes, geo_bytes=geo_bytes,
        refl_stash=refl_stash)


def reflectivity_run_kwargs(kwargs: dict, case: CaseStores) -> dict:
    """:func:`tilestream.driver.reflectivity_run_kwargs` for a case's store.

    The rule is the transport's and lives there, with the four things that
    have to be true at once for a streamed sweep to WRITE the REFL_10CM
    slot; this is the case-shaped call site.  Mark the case with
    :meth:`CaseStores.reflectivity_swept` after each sweep so
    :func:`composite_reflectivity` knows the slot holds a field a
    microphysics call wrote rather than the zeros it was primed with.
    """
    from tilestream import driver as _driver

    return _driver.reflectivity_run_kwargs(kwargs, case.store)


# --------------------------------------------------------------------------
# products, computed out of core
# --------------------------------------------------------------------------

SURFACE_CARRIERS: dict = {
    "T2": ("fields/t2", "K"),
    "U10": ("fields/u10", "m s-1"),
    "V10": ("fields/v10", "m s-1"),
    "PSFC": ("fields/psfc", "Pa"),
    "RAINNC": ("scratch/mp_rainnc", "mm"),
    "RAINC": ("scratch/cu_rainc", "mm"),
    "SNOWNC": ("scratch/mp_snownc", "mm"),
    "PBLH": ("fields/pblh", "m"),
    "HFX": ("fields/hfx", "W m-2"),
    "SWDOWN": ("fields/swdown", "W m-2"),
    "TSK": ("fields/tsk", "K"),
}

#: The two fields a reflectivity slab needs BESIDE the active scheme's own
#: input species: the standalone-current-state path forms ``T`` from
#: ``(thb + thp) * (p/p0)^(Rd/cp)``, so pressure and theta' are read for
#: every scheme and are not in any scheme's row.
_REFL_THERMODYNAMIC_CARRIERS = ("p", "thp")


def _refl_slab_carriers(mp_physics: int) -> tuple[str, ...]:
    """State carriers a reflectivity slab has to be given, for ONE scheme.

    Derived from ``woof.core.refl.REFL_10CM_INPUT_SPECIES`` -- the single
    published table the operator itself dispatches on -- rather than
    restated here.  Audit R-052: the tuple this replaces was Morrison's
    six-moment set verbatim, so a WSM6, Thompson, WDM6 or aerosol-aware
    Thompson store was refused for lacking ``nr``/``ns`` that its scheme
    never allocates and its reflectivity formulation never reads, and the
    refusal fired inside the product step rather than at plan review.
    """
    from woof.core.refl import refl_10cm_input_species

    return tuple(refl_10cm_input_species(int(mp_physics))
                 + _REFL_THERMODYNAMIC_CARRIERS)


def composite_reflectivity(case: CaseStores, *, slab_rows: int = 64,
                           column_max: bool = True):
    """Column-max ``REFL_10CM`` (dBZ) from the host store, by row slabs.

    ``woof.core.refl.compute_refl_10cm`` is WOOF's own diagnostic and this
    calls it unmodified.  Reflectivity is COLUMN-LOCAL -- Morrison's
    ``refl10cm_hm`` walks one column of ``qr/nr/qs/ns/qg/ng`` against
    ``t``/``p`` and reads no neighbour -- so a bare row slab with no halo
    gives exactly the domain's answer, which is what lets a domain that does
    not fit be rendered at all.

    The base state (``thb``) is taken from the GEOGRAPHY store rather than
    rebuilt: on a ``terrain_opt=1`` domain it is 3-D and terrain-following,
    and a slab that rebuilt it from its own rows would rebuild it for a
    different domain.

    THE SCHEME'S OWN FIELD WINS, and ``case.refl_stash`` is what says it is
    there to win with.  A store built by :func:`build_stores` with
    ``reflectivity=True`` carries ``scratch/refl_10cm``; that slot holds a
    field a microphysics call WROTE only once the settling step has computed
    it or a sweep carrying ``refl_10cm_due`` has filled it -- see
    :func:`reflectivity_run_kwargs`, and mark the case with
    :meth:`CaseStores.reflectivity_swept` after such a sweep.  In that state
    the array IS the model's reflectivity and is returned with no recompute
    and no slab loop: the only route for mp=9, 18 and 50, whose dBZ is
    produced inside the scheme call and published through no operator, and
    the only correct route for mp=8 and 28, whose ``compute_refl_10cm``
    branch needs the Thompson graupel-number shadow that the restart
    classifies as rebuilt and never serialises.

    A slot that is present but PRIMED -- a domain past the resident ceiling,
    before its first sweep -- is not read: it is zeros, and zeros rendered
    as dBZ are a domain-wide 0 dBZ field that reads like weak echo.  The
    slab recompute below is what that store gets, for the schemes the
    operator dispatches, and a named refusal for the schemes it does not.
    """
    from woof.core.refl import NativeReflectivityScheme
    from tilestream.physics_inventory import REFL_KEY

    cfg = case.cfg
    nz, ny, nx = int(cfg.nz), int(cfg.ny), int(cfg.nx)
    rows = int(slab_rows)
    if ny % rows:
        raise RealCaseError(f"slab_rows={rows} must divide ny={ny}")
    if case.refl_stash == "computed":
        stashed = case.store.get(REFL_KEY)
        if stashed is None:
            raise RealCaseError(
                f"this case says its reflectivity is computed but carries "
                f"no {REFL_KEY!r}; build_stores(reflectivity=True) is what "
                "puts the slot in the store")
        # COPIED, both ways.  ``store`` holds PINNED host arrays the sweep
        # scatters into, so handing the caller the array itself hands it a
        # view a later sweep overwrites underneath a product it has already
        # been given.
        field = np.array(stashed, dtype=np.float32)
        return field.max(axis=0) if column_max else field

    try:
        carriers = _refl_slab_carriers(cfg.mp_physics)
    except NativeReflectivityScheme as exc:
        # The scheme HAS reflectivity and this store does not have a written
        # copy of it, so say where it comes from rather than reporting a
        # missing operator input for an operator this scheme does not use.
        raise RealCaseError(
            f"{exc}  This case's {REFL_KEY!r} is {case.refl_stash!r}: "
            "build the store with build_stores(reflectivity=True) -- the "
            "default, whose settling step computes the analysis-time field "
            "-- and sweep it with the keywords of "
            "tilestream.realcase.reflectivity_run_kwargs, which are what "
            "make every tile write the slot; a domain past the resident "
            "ceiling has no computed field until its first such sweep.  "
            "Or render a scheme whose reflectivity the operator "
            "computes.") from exc

    # The device is needed only by the recompute, so both the stash route
    # and the refusal above stay host-side and cost no context.
    import cupy as cp

    from woof.core.refl import compute_refl_10cm
    from woof.core.state import DomainState

    out = (np.empty((ny, nx), dtype=np.float32) if column_max
           else np.empty((nz, ny, nx), dtype=np.float32))
    cfg_slab = replace(cfg, ny=rows)
    pool = cp.get_default_memory_pool()

    for j0 in range(0, ny, rows):
        sl = slice(j0, j0 + rows)
        state = DomainState(cfg_slab)
        state.thb[...] = cp.asarray(case.geo_store["setup/thb"][:, sl]
                                    if case.geo_store["setup/thb"].ndim == 3
                                    else case.geo_store["setup/thb"],
                                    dtype=state.thb.dtype)
        for name in carriers:
            arr = getattr(state, name, None)
            if arr is None:
                raise RealCaseError(
                    f"reflectivity needs state.{name}, absent at "
                    f"mp_physics={cfg.mp_physics}")
            arr[...] = cp.asarray(case.store[f"state/{name}"][:, sl],
                                  dtype=arr.dtype)
        refl = compute_refl_10cm(state, cfg)
        if column_max:
            out[sl] = cp.asnumpy(cp.max(refl, axis=0))
        else:
            out[:, sl] = cp.asnumpy(refl)
        del state, refl
        pool.free_all_blocks()
    return out


def reflectivity_3d(case: CaseStores, *, slab_rows: int = 64):
    """The full ``REFL_10CM`` field (nz, ny, nx), same slab loop.

    A wrfout carries the 3-D field, not the column maximum: ``woof render``
    takes the column max itself (``woof.render``: "Composite reflectivity is
    the column maximum of the model-native ``REFL_10CM``"), and the isobaric
    and cross-section products want the levels.
    """
    return composite_reflectivity(case, slab_rows=slab_rows, column_max=False)


def updraft_helicity(case: CaseStores, *, zbot: float = 2000.0,
                     ztop: float = 5000.0) -> np.ndarray:
    """2-5 km updraft helicity (m2 s-2) from the host store.

    UH = integral over the layer of ``w * zeta`` where ``zeta`` is the
    vertical vorticity of the horizontal wind.  This is the standard
    supercell/tornado proxy and it is computed here on the HOST from the
    stored ``u``/``v``/``w``/``php``/``phb`` rather than through a device
    state, because it is a diagnostic of the store and needs no scheme.

    Only positive contributions are accumulated, which is the SPC/HWT
    convention for the plotted field (a mirror-image left-mover would
    otherwise cancel a right-mover in the same box).
    """
    cfg = case.cfg
    g = 9.81
    php = np.asarray(case.store["state/php"], dtype=np.float64)
    phb = case.geo_store.get("setup/phb")
    if phb is None:
        raise RealCaseError("the geography store carries no setup/phb")
    phb = np.asarray(phb, dtype=np.float64)
    if phb.ndim == 1:
        phb = phb[:, None, None]
    z_w = (php + phb) / g                              # (nz+1, ny, nx)
    ht = np.asarray(case.geo_store["setup/ht"], dtype=np.float64)[None]
    agl_w = z_w - ht

    u = np.asarray(case.store["state/u"], dtype=np.float64)
    v = np.asarray(case.store["state/v"], dtype=np.float64)
    w = np.asarray(case.store["state/w"], dtype=np.float64)
    dx = float(cfg.dx)
    dy = float(cfg.dy)
    # zeta at mass points from the C grid: dv/dx - du/dy, centred.
    vm = 0.5 * (v[:, :-1, :] + v[:, 1:, :])            # (nz, ny, nx)
    um = 0.5 * (u[..., :-1] + u[..., 1:])              # (nz, ny, nx)
    zeta = np.zeros_like(vm)
    zeta[:, :, 1:-1] += (vm[:, :, 2:] - vm[:, :, :-2]) / (2.0 * dx)
    zeta[:, 1:-1, :] -= (um[:, 2:, :] - um[:, :-2, :]) / (2.0 * dy)

    uh = np.zeros(vm.shape[-2:], dtype=np.float64)
    for k in range(vm.shape[0]):
        zb, zt = agl_w[k], agl_w[k + 1]
        thick = zt - zb
        overlap = (np.minimum(zt, ztop) - np.maximum(zb, zbot))
        frac = np.clip(overlap, 0.0, None) / np.maximum(thick, 1e-6)
        wl = 0.5 * (w[k] + w[k + 1])
        contrib = wl * zeta[k] * thick * frac
        uh += np.where(contrib > 0.0, contrib, 0.0)
    return uh.astype(np.float32)


def surface_snapshot(case: CaseStores, *, elapsed_s: float,
                     refl: bool = True, uh: bool = True,
                     slab_rows: int = 64) -> dict:
    """Everything one figure needs, as plain host arrays."""
    cfg = case.cfg
    out: dict = {
        "elapsed_s": float(elapsed_s),
        "nx": int(cfg.nx), "ny": int(cfg.ny), "nz": int(cfg.nz),
        "dx": float(cfg.dx), "dt": float(cfg.dt),
        "valid_time": (case.start_time
                       + timedelta(seconds=float(elapsed_s))).isoformat(),
        "init_time": case.start_time.isoformat(),
        "LAT": case.lat, "LON": case.lon, "HT": case.terrain,
    }
    for label, (key, units) in SURFACE_CARRIERS.items():
        if key in case.store:
            out[label] = np.asarray(case.store[key], dtype=np.float32).copy()
            out[label + "_units"] = units
    w = np.asarray(case.store["state/w"], dtype=np.float32)
    out["WMAX"] = w.max(axis=0)
    out["WMIN"] = w.min(axis=0)
    if refl:
        out["REFL_COMPOSITE"] = composite_reflectivity(
            case, slab_rows=slab_rows)
    if uh:
        out["UH25"] = updraft_helicity(case)
    return out
