"""The tiled stepping loop: gather -> step -> scatter, one tile at a time.

This is the module that turns the geometry (:mod:`tilestream.spec`), the
transport (:mod:`tilestream.gather`) and the pinned full-domain store
(:mod:`tilestream.hoststore`) into an actual integration.  It is also where the
one bug that would silently invalidate the whole idea lives, so that bug gets
its own section.


THE READ-AT-TIME-t RULE
-----------------------
A tile's halo exists so the tile can see its neighbours *as they were at the
start of the step*.  If tile A's freshly-stepped interior is written back into
the same array that tile B then gathers its halo from, B sees A at time
``t + dt`` while seeing itself at time ``t``.  The result is not "slightly
wrong": it is a different, inconsistent time-stepping scheme that happens to
run and happens to look plausible.  Nothing about a hash tells you which of the
two you got, which is why this module makes the distinction explicit and
testable:

``write_mode="ring"`` (default, CORRECT, and the one that scales)
    One store.  Before a tile is stepped, the part of its interior that a
    LATER tile will read -- its ring -- is copied out to a small arena, and a
    tile's gather is patched from the arenas of the tiles already written, so
    every window holds time-t values everywhere.  Costs a fraction of a
    second store instead of a whole one: MEASURED 4.97% of the store's bytes
    at 1950^2 tile 650 and 2.39% at 5412^2 tile 1353, against 100% for the
    shadow.  The 8x4090 box measured the same quantity on ITS hardware and
    got 5.2% at 1950^2 tile 650 and 5.8% at 3276^2 tile 546 -- carried here
    because the two boxes DISAGREE at the one plan they share (4.97% against
    5.2% at 1950^2 tile 650), and a ring-cost figure quoted without its box
    is therefore not a number.  See ``tilestream/RECONCILIATION.md``.
    :mod:`tilestream.rings` owns the geometry and states the four ways the
    scheme goes silently wrong; the ordering it needs from THIS module is in
    the loop's block comment.

``write_mode="shadow"`` (also CORRECT, kept as the reference)
    Every tile reads buffer *R* and writes buffer *W*.  The interiors partition
    the domain exactly (``spec.validate_plan`` proves it), so at the end of a
    sweep *W* is a complete new domain and the two buffers are swapped.  Costs
    a second full-domain store.  Retained because it is the independent
    implementation the ring path is checked against -- the gate runs both and
    demands identical digests -- and because it has no ordering requirements
    at all, so it is the thing to fall back to if a ring plan cannot be built.

``write_mode="inplace"`` (WRONG ON PURPOSE)
    Tiles read and write the same buffer, with no ring saved and no patch.
    Provided so the gate can show that it *detects* the error rather than
    merely avoiding it -- a gate that has never seen a failure is not a gate.
    It is also precisely "the ring scheme minus the ring", which is what makes
    it the right negative control for this module.  ``run_tiled`` warns every
    time it is selected.


WHAT A TILE IS STEPPED AS
-------------------------
A tile is just a smaller domain.  ``tile_cfg`` is the parent ``RunConfig`` with
``nx``/``ny`` replaced by the tile's GATHERED extents (interior + 2*halo), not
its interior extents -- the ~15 sites that read ``cfg.nx``/``cfg.ny`` are
computing launch geometry for the array in hand, and that array is the gathered
one.  Everything else (dx, dt, sound steps, damping, physics selectors) is
carried through untouched.

The tile ``DomainState`` is built ONCE per buffer and reused for every tile.
That is safe for two independent reasons, and both are checked rather than
assumed: the gather covers the tile's persisted arrays completely
(``require_full_gather=True`` in :mod:`tilestream.gather` asserts the
rectangles tile the destination exactly), and nothing outside the persisted set
carries information across a step -- which
:func:`assert_streaming_inventory_complete` demonstrates by dirtying a state,
refilling only the streamed fields, and showing the trajectory is unchanged.


PHYSICS
-------
A physics-on domain carries far more than ``STATE_SERIALIZED_ATTRS``, and
three of this module's assumptions had to be replaced rather than widened.
:func:`physics_run_kwargs` supplies all four pieces; the reasons are:

``inventory_fn=physics_inventory.carrier_inventory``
    ``gather.inventory`` reaches the contract attributes by ``getattr`` and
    therefore cannot see ``state._scratch`` (the mp/cu precipitation
    accumulators) or ``state.physics`` (the held tendencies and the 88-to-162
    surface/soil/snow ``fields``) AT ALL.  Streaming the contract set at a
    physics rung leaves 1.79x more carried bytes behind than it moves, and
    MEASURED it changes 127 of 229 carriers after eight steps.

``nz=cfg.nz``
    Noah's soil columns are ``(4, ny, nx)``, Noah-MP's snow layers
    ``(3, ny, nx)`` and its snow-soil coordinate ``(7, ny, nx)``.  Their
    leading axis is not vertical, so ``gather.domain_extents`` cannot infer
    the vertical extent from the inventory -- it used to answer 3.

``tile_state_factory=make_physics_tile_state``
    The buffer needs a ``PhysicsDriver`` (``dycore.step`` raises without one)
    and needs to have STEPPED ONCE (Kain-Fritsch's ``w0avg`` and friends are
    allocated lazily, so a never-stepped buffer has a shorter inventory than
    the store).

``scalars=carrier_scalars(parent)``
    The domain clock.  See ``run_tiled``'s docstring: without it a buffer
    serving k tiles runs k*dt ahead of the domain inside one sweep and tiles
    disagree about which schemes are due.

GEOGRAPHY
---------
A tile's ``DomainState`` and ``PhysicsDriver`` are constructed on
``tile_cfg``, so everything they DERIVE is derived for the tile's extents.
For a real map projection that is catastrophic and silent:
``woof/static/projection.py:122-123`` defaults the reference point to
``(e_we/2, e_sn/2)`` and ``_grid_xy`` (:176-179) counts DOMAIN indices, so a
rebuilt tile believes it is centred where the whole domain is centred.  The
displacement is a rigid translation by ``ci0 + (cnx+1)/2 - (nx+1)/2`` cells
per axis -- and the exactly-centred tile is BIT-EXACT, which is precisely
what lets a one-tile test certify the bug.

The rule this module implements is: GEOGRAPHY IS INPUT.  It is computed once
for the whole domain and GATHERED per tile like any other field; it is never
rebuilt from ``tile_cfg`` and never scattered back.  ``run_tiled(geography=
...)`` takes the domain's :func:`geography_inventory` and installs each
tile's window into the buffer.  Three details make the difference between
that working and looking like it works:

* it is gathered ONCE PER BUFFER OCCUPANCY, not once per step.  MEASURED
  (:func:`tilestream.harness.make_physics_state`, 8 steps, 229 carriers,
  real Lambert + real terrain): all 33 geography arrays are bit-identical
  after the run, so re-sending them every step is pure traffic.  With
  ``nbuffers >= len(specs)`` each buffer serves one tile and the gather
  happens exactly once for the whole run.
* ``has_msf``/``rotational`` are re-imposed from the DOMAIN.
  ``set_map_coriolis`` (state.py:799-803) recomputes both from ``.any()``
  over whatever window it was handed, and they gate whole code paths
  (``dycore.py:539``, ``moist.py:470-559``, ``physics.py:1268``).  A tile
  whose window happens to be uniform would flip them.
* the four scheme latitude/longitude grids are gathered too.  Every
  solar-zenith path in the tree reads PER-COLUMN lat/lon arrays and only the
  clock terms are scalars, but those arrays live on the SCHEME object, not
  on the state and not in the carrier manifest -- and two of the four
  (legacy RRTMG, Noah-MP) are HOST numpy rather than device.

:func:`assert_geography_gathered` refuses anything the gather cannot reach.
Its one known customer is legacy RRTMG's ozone cache, which is interpolated
to the domain's latitudes at CONSTRUCTION and stored ``(ny*nx, 59, 12)`` --
horizontal axes FLATTENED and LEADING, so the transport cannot window it and
gathering ``latitude_deg`` would not fix it.  That configuration is refused,
not approximated.

THE SCALARS, stated explicitly because a scalar cannot be gathered
-----------------------------------------------------------------
Searched rather than assumed.  ``RunConfig`` has 147 fields and the only two
that touch geography are ``map_proj`` and ``terrain_opt`` -- there is no
``ref_lat``/``ref_lon``/``truelat``/``stand_lon``/``known_x``/``known_y``
anywhere in it, so a tile config cannot carry a reference point even in
principle, and there is no f-plane reference latitude to get wrong.  Three
scalar families remain and each is handled here:

``has_msf`` / ``rotational``
    RECOMPUTED per tile by ``set_map_coriolis`` from ``.any()`` over the
    window.  Imposed from the domain -- see :func:`geography_scalars`.  The
    gate's ``impose_geography_flags=False`` control measures what happens
    otherwise: 32 of 229 carriers differ.

``mub`` / ``p_top`` / ``cf1,cf2,cf3,cfn,cfn1``
    Pure functions of the vertical coordinate and the base sounding, so a
    tile rebuilds them exactly -- EXCEPT ``mub``, which ``load_base``
    retires to ``None`` the moment terrain makes ``mub2d`` the authority
    (state.py:744-751).  A buffer built flat against a parent with terrain
    disagrees on a scalar rather than on an array;
    :func:`setup_scalar_mismatches` is what catches it, and
    :func:`geography_run_kwargs` builds the buffer with terrain so it
    cannot happen.

the solar clock, and ``cen_lat``
    Every zenith path in the tree takes per-column lat/lon ARRAYS; the only
    scalars in it (declination, equation of time, solcon, gmt/xtime) are
    functions of the CLOCK, which ``scalars=`` already carries.  MEASURED at
    96x80 with a real Lambert grid: COSZEN spans 0.84649 to 0.93134 across
    the domain, i.e. it genuinely varies per column.  The one true
    geography SCALAR in the tree is ``cen_lat``, taken by
    ``woof/core/landuse.py:327`` to pick the LANDUSE.TBL season (reversed
    south of the equator, :389).  It is INITIALIZATION-only: the 15 fields
    ``LanduseInitialization`` produces (landmask, xland, lakemask, ivgtyp,
    isltyp, snowc, pblh, ust, mavail, z0, znt, albbck, albedo, embck,
    emiss) are all ``fields/*`` carriers, so a tile buffer's own landuse is
    overwritten by the first gather and its ``cen_lat`` never reaches a
    step.  A caller that does construct landuse per tile must pass the
    DOMAIN's ``cen_lat``, not the tile's -- on a domain straddling the
    equator the season would otherwise flip between tiles.
"""

from __future__ import annotations

import time as _time
import types as _types
import warnings

import numpy as np

from tilestream import gather as _gather
from tilestream import harness as _harness
from tilestream import spec as _spec


__all__ = [
    "GeographyNotGatherable",
    "TiledRunError",
    "assert_geography_gathered",
    "assert_streaming_inventory_complete",
    "geography_inventory",
    "geography_run_kwargs",
    "geography_scalars",
    "geography_store",
    "geography_window_mismatches",
    "make_physics_tile_state",
    "make_tile_state",
    "physics_run_kwargs",
    "plan_for",
    "ring_bytes_vs_shadow",
    "run_tiled",
    "setup_scalar_mismatches",
    "setup_window_mismatches",
]


class TiledRunError(RuntimeError):
    """A tiled run could not be set up consistently."""


class GeographyNotGatherable(TiledRunError):
    """A tile derives geography the transport cannot window."""


# --------------------------------------------------------------------------
# the tile state
# --------------------------------------------------------------------------

def make_tile_state(tile_cfg, *, base_theta: float = 300.0, stretch=None):
    """A zeroed ``DomainState`` on ``tile_cfg`` with the harness's base state.

    Built the same way :func:`tilestream.harness.make_state` builds its
    parent -- same vertical coordinate, same constant-theta base state, same
    ``init_at_rest`` -- minus the seeding, because every persisted array is
    about to be overwritten by a gather.  Starting from ``init_at_rest``
    rather than a seeded state means the tile buffer contains no random junk
    anywhere, so a field that a gather failed to write shows up as an
    at-rest value rather than as plausible weather.

    Under the milestone-one settings (``terrain_opt=0``, ``map_proj=0``,
    ``hybrid_opt=0``) every setup array this builds is either 1-D in z or
    horizontally uniform, so it equals the parent's window for any tile.
    :func:`setup_window_mismatches` checks that instead of trusting it.
    """
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.state import init_at_rest

    coord = make_vertical_coord(tile_cfg.nz, stretch=stretch,
                                hybrid_opt=tile_cfg.hybrid_opt,
                                etac=tile_cfg.etac)
    base = make_base_state(coord,
                           lambda z: np.full_like(z, float(base_theta)),
                           p_surf=tile_cfg.p_surf, ztop=tile_cfg.ztop,
                           terrain_z=None)
    return init_at_rest(tile_cfg, coord, base)


def make_physics_tile_state(tile_cfg, *, builder=None, seed: int = 4242,
                            warmup: int = 1, shared=None):
    """A tile buffer with a ``PhysicsDriver`` attached, ready to be gathered into.

    Three things this does that :func:`make_tile_state` cannot, each of them
    essential:

    * it attaches a driver.  ``dycore.step`` raises ``physics is enabled but
      the state has no PhysicsDriver`` the moment ``physics_enabled(cfg)``,
      and ``harness.make_state`` attaches none -- which is why every physics
      rung of the milestone-one gate died before it could compare anything.
    * it builds on the WK82 sounding rather than a constant 300 K theta.  At
      a 20 km model top a constant-theta base puts the top TEMPERATURE at
      102 K and RRTMGP refuses it outright (valid over [160, 355] K).
    * it RUNS ``warmup`` steps.  Two carriers are allocated lazily on first
      use -- Kain-Fritsch's ``cumulus/w0avg`` above all (kf.py:335) -- so a
      buffer that has never stepped is missing fields the store holds, and
      the gather would fail the inventory match.  One step is enough and the
      values it leaves behind are overwritten by the first gather.

    The seed is fixed and different from any gate seed on purpose: whatever
    this leaves in the buffer must be overwritten by the gather, and if it is
    not, an answer that depends on it is easier to spot when it is nobody's
    data.
    """
    from tilestream import physics_inventory as _physics

    build = builder or _physics.default_builder
    state, _driver = (build(tile_cfg, seed) if shared is None
                      else build(tile_cfg, seed, shared=shared))
    if warmup:
        _harness.run_steps(state, tile_cfg, int(warmup))
    return state


def physics_run_kwargs(cfg, parent_state, *, builder=None, seed: int = 4242,
                       warmup: int = 1, shared=None) -> dict:
    """The ``run_tiled`` keyword set for a physics-on domain.

    One call site instead of five things to remember::

        run_tiled(store, cfg, T, T, halo, nsteps,
                  **physics_run_kwargs(cfg, parent))

    Supplies the carrier inventory (so ``state._scratch`` and
    ``state.physics`` are streamed rather than silently left behind), the
    explicit ``nz`` the layered soil/snow fields make necessary, the tile
    factory that attaches a driver, and the DOMAIN's scalar carriers --
    without which every tile after the first in a sweep evaluates its physics
    cadence one step further ahead than the domain actually is.

    ``parent_state=None`` yields ``scalars=None``, for a caller that already
    holds the domain clock (a gate that snapshots the start state and frees
    the parent before running).  It is the only piece here the caller can
    reasonably own; the other three are structural.
    """
    from tilestream import physics_inventory as _physics

    return dict(
        inventory_fn=_physics.carrier_inventory,
        nz=int(cfg.nz),
        tile_state_factory=lambda tile_cfg: make_physics_tile_state(
            tile_cfg, builder=builder, seed=seed, warmup=warmup,
            shared=shared),
        scalars=(None if parent_state is None
                 else _physics.carrier_scalars(parent_state)),
        shared=shared,
    )


def reflectivity_run_kwargs(kwargs: dict, store) -> dict:
    """``run_tiled`` keywords that make a sweep WRITE the REFL_10CM slot.

    A streamed run publishes the model's own reflectivity only when four
    things are true at once, and every one of them is a published piece of
    :mod:`woof.core.streaming` rather than a rule invented here:

    * the STORE carries ``scratch/refl_10cm``, because its builder put it
      there: ``tilestream.realcase.build_stores(reflectivity=True)``, the
      default, and ``tilestream.bigdomain.carrier_manifest_for``;
    * every TILE BUFFER carries it too, or ``TiledRun`` refuses the sweep
      for an inventory differing from the store's by exactly that key.
      ``prime_refl_10cm`` allocates the slot and steps nothing, which is
      what a buffer needs before its inventory is taken;
    * the inventory rule the transport gathers and scatters by NAMES it --
      ``physics_inventory.carrier_inventory_with_refl``, which is
      ``streaming.refl_inventory`` over this package's manifest -- because
      the plain manifest is the RESTART manifest and a pure diagnostic is
      correctly absent from it; and
    * every tile step is told a frame is due.  ``refl_10cm`` is written from
      the ``refl_10cm_due`` branch of the microphysics drivers and from
      nowhere else, so without the keyword each buffer scatters an unwritten
      window and the domain field is whatever the buffers were primed with.
      Audit R-052: the store-side inventory alone left exactly this half
      undone, and the composite silently fell back to recomputing one
      scheme's formulation from another scheme's species.

    ``streaming.refl_handoff_hook`` is the fifth thing, and it is why this
    is a function rather than a paragraph in a docstring: ``stash_refl_10cm``
    parks one frame's array on a driver and REFUSES to overwrite an
    unconsumed handoff.  That refusal is right for a resident domain, where
    two microphysics calls between two frames is a cadence bug, and wrong
    for a sweep, where the handoff is per TILE and the frame is per DOMAIN
    -- without the hook the second tile a buffer serves raises "REFL_10CM
    stash was not consumed before reuse" and stops the forecast.

    Returns ``kwargs`` unchanged for a store that carries no slot, so a lane
    that asked for ``reflectivity=False`` is unaffected.  Each lane marks
    its own store as swept afterwards -- ``CaseStores.reflectivity_swept``,
    ``bigdomain.snapshot(refl_stash=...)`` -- because that is what tells the
    composite the slot holds a field a microphysics call WROTE rather than
    the zeros it was primed with, and zeros rendered as dBZ are a
    domain-wide 0 dBZ product that reads like weak echo.
    """
    from woof.core.streaming import prime_refl_10cm, refl_handoff_hook
    from tilestream import physics_inventory as physinv

    if physinv.REFL_KEY not in store:
        return dict(kwargs)
    out = dict(kwargs)
    out["inventory_fn"] = physinv.carrier_inventory_with_refl
    factory = out.get("tile_state_factory")
    if factory is None:
        raise TiledRunError(
            "a reflectivity-carrying sweep needs the physics tile factory "
            "(driver.physics_run_kwargs / geography_run_kwargs); there is "
            "no buffer to prime the refl_10cm slot on")

    def _factory_with_refl(tile_cfg):
        tile = factory(tile_cfg)
        prime_refl_10cm(tile, tile_cfg)
        return tile

    out["tile_state_factory"] = _factory_with_refl
    step_kwargs = dict(out.get("step_kwargs") or {})
    step_kwargs["refl_10cm_due"] = True
    out["step_kwargs"] = step_kwargs
    existing = out.get("post_step_hook")
    if existing is not None:
        raise TiledRunError(
            "this sweep already carries a post_step_hook and the "
            "reflectivity route needs its own (streaming.refl_handoff_hook, "
            "which clears each tile's one-frame REFL stash); compose them "
            "at the call site rather than losing one silently")
    out["post_step_hook"] = refl_handoff_hook()
    return out


def geography_run_kwargs(cfg, parent_state, *, geography_fn=None,
                         geography=None, seed: int = 4242, warmup: int = 1,
                         host: bool | None = None, coord_fn=None) -> dict:
    """:func:`physics_run_kwargs` plus everything a REAL projection needs.

    Three additions, and the middle one is the one that surprises::

        run_tiled(store, cfg, T, T, halo, nsteps,
                  **geography_run_kwargs(cfg, parent))

    * ``geography`` -- the domain's arrays, copied once into a store of their
      own (``host=True`` for pinned host RAM, ``False`` for VRAM, ``None`` to
      keep each array where the parent has it).
    * ``tile_state_factory`` -- a buffer built by
      :func:`tilestream.harness.geography_builder`, i.e. with TERRAIN.  It
      has to be: ``cfg.terrain_opt`` decides whether ``thb/pb/alb/phb`` are
      1-D or 3-D (state.py:605-607) and whether ``load_base`` retires the
      scalar ``mub`` in favour of ``mub2d`` (state.py:744-751).  A buffer
      built flat against a parent with terrain has the wrong SHAPES, and the
      inventory check would reject it -- late, and with a confusing message.
      The geography that buffer computes is wrong (it is the per-tile
      rebuild) and is overwritten by the first gather.
    * the physics carrier set, ``nz`` and the domain clock, unchanged.
    """
    from tilestream import harness as _h

    build = geography_fn or _h.make_geography
    kwargs = physics_run_kwargs(
        cfg, parent_state,
        builder=_h.geography_builder(build, coord_fn=coord_fn),
        seed=seed, warmup=warmup)
    if geography is None and parent_state is None:
        raise TiledRunError(
            "geography_run_kwargs needs either a parent_state to copy the "
            "domain geography from or an explicit geography= store")
    kwargs["geography"] = (geography_store(parent_state, host=host)
                           if geography is None else geography)
    return kwargs


def setup_window_mismatches(tile_state, parent_state, spec) -> dict[str, float]:
    """``{name: max_abs_diff}`` for setup arrays that differ from the window.

    Empty means every entry of ``STATE_SETUP_ARRAYS`` on the tile equals the
    corresponding window of the parent -- the assumption that lets a tile be
    stepped without gathering its setup.  1-D (vertical) arrays are compared
    whole; horizontal ones are compared through the spec's own gather
    rectangles, so the staggering and the periodic wrap are handled by the
    same code the data path uses.

    A non-empty result means the tiled run CANNOT reproduce the monolithic one
    and the setup arrays have to be streamed too (terrain and real map
    projections both do this).
    """
    import cupy as cp

    bad: dict[str, float] = {}
    parent = _harness.setup_arrays(parent_state)
    tile = _harness.setup_arrays(tile_state)
    for name, ref in parent.items():
        got = tile.get(name)
        if got is None:
            bad[name] = float("inf")
            continue
        if ref.ndim < 2:
            if ref.shape != got.shape or not bool(cp.all(ref == got)):
                bad[name] = float(cp.abs(ref.astype(cp.float64)
                                         - got.astype(cp.float64)).max())
            continue
        kind = _gather.classify(ref.shape, parent_state.p.shape[0],
                                spec.ny, spec.nx)
        window = cp.empty_like(got)
        spec.apply_gather(ref, window, kind)
        if not bool(cp.all(window == got)):
            bad[name] = float(cp.abs(window.astype(cp.float64)
                                     - got.astype(cp.float64)).max())
    return bad


# --------------------------------------------------------------------------
# geography: gathered, never rebuilt
# --------------------------------------------------------------------------

#: ``(key prefix, PhysicsDriver attribute)`` for every scheme that holds its
#: own per-column latitude/longitude grid.  These are NOT on the state and
#: NOT in the carrier manifest, so nothing else in this pipeline would move
#: them -- and every solar-zenith path in the tree reads them per column
#: (dudhia.py:55-103, rrtmgp.py:1842-1882, rrtmg_legacy.py:155-181,
#: noahmp_runtime.py:617-624, analytic_radiation.py:68-77).  MEASURED
#: residency, which decides the copy direction: RRTMGP and Dudhia keep them
#: as cupy DEVICE arrays, legacy RRTMG and Noah-MP as HOST numpy.
_SCHEME_GEOGRAPHY: tuple[tuple[str, str], ...] = (
    ("radiation", "radiation_callable"),
    # OSAS's counter draw follows domain column identity through tile reuse.
    ("cumulus", "cumulus_callable"),
    ("noahmp", "noahmp_geometry"),
    ("cam_ozone", "cam_ozone"),
    # WRF's swint_opt = 1 carrier (woof.core.swint): its per-step zenith
    # cosine reads its own device latitude/longitude grids.
    ("swint", "swint"),
    ("farms", "farms"),
)
_SCHEME_GEOGRAPHY_ATTRS: tuple[str, ...] = (
    "latitude_deg", "longitude_deg", "column_indices")


def _scheme_geography_owners(driver):
    """Each live scheme setup, including independently selected spectra."""
    from woof.core.radiation_composition import radiation_adapters
    for prefix, attr in _SCHEME_GEOGRAPHY:
        scheme = getattr(driver, attr, None)
        if scheme is None:
            continue
        yield prefix, scheme
        if attr == "radiation_callable":
            for index, component in enumerate(radiation_adapters(scheme)):
                if component is not scheme:
                    yield f"{prefix}/component_{index}", component

#: The two STATE_SETUP_SCALARS a tile must INHERIT rather than recompute.
#: ``set_map_coriolis`` derives both from ``.any()`` over whatever window it
#: was handed (state.py:799-803).  The other five (``mub``, ``p_top``,
#: ``cf1..cfn1``) are functions of the vertical coordinate and the base
#: sounding, so a tile rebuilds them exactly; ``setup_scalar_mismatches``
#: checks that rather than assuming it.
_SETUP_FLAGS: tuple[str, ...] = ("has_msf", "rotational")


def geography_inventory(obj, names=None) -> dict:
    """``{key: array}`` of every horizontally-varying INPUT a tile must gather.

    Accepts a ``DomainState`` (the driver is found at ``state.physics``), a
    plain mapping or a :class:`tilestream.gather.HostStore`, so the same
    function serves as ``run_tiled``'s geography ``inventory_fn`` on both
    sides of the copy.

    Keys are ``setup/<name>`` for :data:`STATE_SETUP_ARRAYS` entries with a
    horizontal extent, ``<scheme>/<attr>`` for the scheme lat/lon grids,
    and ``terrain_drag/*`` for drag statistics and topo coefficients.
    The coefficients carry the domain's terrain laplacian; rebuilding them
    from a tile's terrain would clamp the stencil at its internal edges.
    The purely vertical setup arrays (``c1h..c4f``, ``dnw``, ``rdnw``,
    ``dn``, ``rdn``, ``fnp``, ``fnm``, ``znu``, ``znw``, and ``thb/pb/alb/
    phb`` when ``terrain_opt == 0``) are deliberately ABSENT: they are pure
    functions of ``nz``/``hybrid_opt``/``etac``/``p_top``, so a tile rebuilds
    them exactly.  :func:`setup_window_mismatches` is what checks that claim
    rather than assuming it, and it covers the 1-D arrays this skips.
    """
    import cupy as cp

    from woof.state_serialization_contract import (
        STATE_DERIVED_SETUP_ARRAYS, STATE_SETUP_ARRAYS)

    inner = getattr(obj, "arrays", None)
    if isinstance(inner, dict):
        obj = inner
    out: dict = {}
    if isinstance(obj, dict) or isinstance(obj, _gather.HostStore):
        source = obj if isinstance(obj, dict) else obj.as_dict()
        for key in sorted(source):
            value = source[key]
            if value is not None and _gather._is_array(value):
                out[key] = value
    else:
        # The DERIVED entries gather with the base state they came from:
        # over terrain dphb_resid is 3-D like phb, and a tile that gathered
        # phb without it would hold a correction describing a different
        # column.  They sit outside STATE_SETUP_ARRAYS only to keep the
        # restart setup digest's byte stream stable.
        for name in STATE_SETUP_ARRAYS + STATE_DERIVED_SETUP_ARRAYS:
            value = getattr(obj, name, None)
            if (isinstance(value, (cp.ndarray, np.ndarray))
                    and value.ndim >= 2):
                out[f"setup/{name}"] = value
        phys = getattr(obj, "physics", None)
        drag = getattr(phys, "terrain_drag", None)
        if drag is not None:
            for name in ("ctopo", "ctopo2"):
                value = getattr(drag, name, None)
                if isinstance(value, (cp.ndarray, np.ndarray)):
                    out[f"terrain_drag/{name}"] = value
            for name, value in (getattr(drag, "gwd", None) or {}).items():
                out[f"terrain_drag/gwd/{name}"] = value
        for prefix, scheme in _scheme_geography_owners(phys):
            for field in _SCHEME_GEOGRAPHY_ATTRS:
                value = getattr(scheme, field, None)
                if isinstance(value, (cp.ndarray, np.ndarray)):
                    out[f"{prefix}/{field}"] = value
        terrain_drag = getattr(phys, "terrain_drag", None)
        # Both lanes' gathers name the same keys; an object without the
        # TerrainDrag accessor is covered by the attribute walk above.
        if callable(getattr(terrain_drag, "geography", None)):
            out.update(terrain_drag.geography())
        out = {k: out[k] for k in sorted(out)}
    if names is not None:
        keep = set(names)
        out = {k: v for k, v in out.items() if k in keep}
    return out


def geography_store(source, *, host: bool | None = None) -> dict:
    """A domain-side copy of ``source``'s geography, ready to gather from.

    ``host=None`` keeps each array's own memory class (device stays device,
    host stays host) and only PINS the host ones, which is what a run whose
    carriers live in VRAM wants.  ``host=True`` puts everything in pinned
    host RAM -- the out-of-core case -- and ``host=False`` puts everything in
    VRAM.  Both directions are legal for every field: the transport is
    ``cudaMemcpy3DAsync`` and the tile side is a mixture whatever the store
    is (RRTMGP's grid is device, Noah-MP's is host), so the copies are a mix
    of H2D, D2H and H2H by construction rather than by accident.
    """
    import cupy as cp

    out: dict = {}
    for key, array in geography_inventory(source).items():
        if host is True or (host is None
                            and not _gather.is_device_array(array)):
            out[key] = _gather.pinned_copy(_as_host(array))
        else:
            out[key] = cp.asarray(array)
    return out


def _as_host(array) -> np.ndarray:
    import cupy as cp

    if isinstance(array, cp.ndarray):
        with array.device:
            return cp.asnumpy(array)
    return np.ascontiguousarray(array)


def geography_scalars(arrays) -> dict[str, bool]:
    """``has_msf``/``rotational`` for the WHOLE domain, state.py's own rule.

    ``DomainState.set_map_coriolis`` ends by deriving both flags from
    ``.any()`` reductions over the arrays it was handed (state.py:799-803).
    Handed a TILE's window that is what it measures, and a tile whose window
    happens to be uniform silently takes a different branch in
    ``dycore.py:539`` (the Coriolis+curvature kernel), ``moist.py:470-559``
    and ``physics.py:1268``.  So the flags are computed ONCE here, from the
    domain arrays, and re-imposed on every buffer.
    """
    import cupy as cp

    def any_ne(key: str, value: float) -> bool:
        array = arrays.get(f"setup/{key}")
        if array is None:
            return False
        if isinstance(array, cp.ndarray):
            return bool((array != value).any())
        return bool(np.any(np.asarray(array) != value))

    has_msf = (any_ne("msft", 1.0) or any_ne("msfu", 1.0)
               or any_ne("msfv", 1.0))
    rotational = has_msf or any_ne("f", 0.0) or any_ne("e", 0.0)
    return {"has_msf": has_msf, "rotational": rotational}


def _pin_scheme_geography(state) -> None:
    """Replace a tile buffer's HOST scheme lat/lon grids with pinned ones.

    Noah-MP's ``NoahmpSolarGeometry`` stores ``np.ascontiguousarray(...)``
    (noahmp_runtime.py:604-607) and legacy RRTMG the same, i.e. PAGEABLE host
    memory -- which ``gather._check_host`` refuses outright, and rightly:
    a pageable destination makes the copy synchronous and quarters the
    bandwidth.  The buffer is ours to re-point, and the schemes only ever
    read the attribute, so swapping in a pinned array of the same dtype and
    shape is invisible to them.
    """
    driver = getattr(state, "physics", None)
    for _prefix, scheme in _scheme_geography_owners(driver):
        for field in _SCHEME_GEOGRAPHY_ATTRS:
            value = getattr(scheme, field, None)
            if (isinstance(value, np.ndarray)
                    and not _gather.is_device_array(value)
                    and not _gather.is_pinned(value)):
                setattr(scheme, field, _gather.pinned_copy(value))


def assert_geography_gathered(state, driver=None, *, keys=None,
                              allow=()) -> None:
    """Raise unless every scheme geography array is one the gather reaches.

    The rule this checks is deliberately narrow, and the reason is a
    measurement.  A tile buffer holds ~113 to 210 horizontally-shaped driver
    arrays at the physics rungs (MEASURED, 40x32x49, real projection), and
    with a real projection MOST of them vary horizontally -- so "varies
    horizontally and is not gathered" is not a usable rule: it flags
    ``driver.tendencies``, ``driver.last_ysu`` and ``driver.olr``, which are
    physics scratch and output.  Those are already proven harmless by
    :func:`assert_streaming_inventory_complete`, which dirties everything
    outside the carrier set and shows the trajectory is unchanged.

    Geography is exactly what that proof CANNOT see, because its two probe
    states are built with the same lat/lon and so it never dirties them.  So
    this checks the things that are unambiguous:

    ``latitude_deg`` / ``longitude_deg`` on any scheme
        found by walking the driver, so a scheme added tomorrow is caught the
        day it is added rather than the day someone notices the forecast is
        lit at the wrong hour.  Must be in the gathered set, and UNIFORMITY
        IS NOT AN EXCUSE here -- this runs on the tile BUFFER, whose grids are
        the neutral constant :func:`tilestream.harness.neutral_geography`
        installs, so a uniformity escape would make the check vacuous exactly
        when it matters.  The domain's grid varies; that is why it is being
        gathered.

    Terrain drag coefficients and orographic statistics
        must be gathered even when neutral initialization makes every value
        equal. A missing gather would silently disable drag on the tile.

    ``(ny*nx, ...)`` -- horizontal axes FLATTENED into a leading column index
        the transport windows trailing axes and cannot touch this layout.
        An owner may declare a cache bound to gathered inputs through
        ``geography_cache_dependencies`` and refresh it before every read.
        Those caches need no transport; their live inputs must be gathered
        and their retained input snapshots must be present. Undeclared
        varying flattened arrays are still refused. Uniform constants are
        exempt, but uniformity never exempts a declared live dependency.
    """
    import cupy as cp

    from tilestream import physics_inventory as _physics

    if driver is None:
        driver = getattr(state, "physics", None)
    if driver is None:
        return
    ny, nx = int(state.p.shape[-2]), int(state.p.shape[-1])
    ncol = ny * nx
    gathered = {id(v) for v in geography_inventory(state, keys).values()}
    allow = set(allow)
    bound_caches: set[int] = set()
    seen: set[int] = {id(state)}
    bad: list[tuple] = []

    def note(array, path, family) -> None:
        if id(array) in gathered or id(array) in bound_caches or path in allow:
            return
        if family == "flattened-column":
            flat = np.asarray(_as_host(array)).reshape(ncol, -1)
            if not flat.size or bool(np.all(flat == flat[:1])):
                return
        bad.append((path, tuple(int(s) for s in array.shape),
                    str(array.dtype), family))

    terrain_drag = getattr(driver, "terrain_drag", None)
    if callable(getattr(terrain_drag, "geography", None)):
        for name, array in terrain_drag.geography().items():
            note(array, name, "terrain-drag-static")

    def walk(obj, path, depth) -> None:
        if depth > 4 or id(obj) in seen:
            return
        seen.add(id(obj))
        # A module bound on a scheme (legacy RRTMG keeps woof.ingest.wrf_ozone
        # as self._ozone) is code, not per-domain state.  Walking into it
        # reached ctypes.cdll, whose attribute lookup dlopen()s the name: on
        # Python 3.11 that raised OSError "geography_cache_dependencies:
        # cannot open shared object file" and failed every ranked build with
        # legacy RRTMG and o3input on (3.12 turns it into AttributeError).
        if isinstance(obj, _types.ModuleType):
            return
        if isinstance(obj, (cp.ndarray, np.ndarray)):
            shape = tuple(int(s) for s in obj.shape)
            if ncol > 1 and shape and shape[0] == ncol:
                note(obj, path, "flattened-column")
            return
        if isinstance(obj, dict):
            for key, value in obj.items():
                walk(value, f"{path}[{key!r}]", depth + 1)
            return
        if isinstance(obj, (list, tuple)):
            for i, value in enumerate(obj):
                walk(value, f"{path}[{i}]", depth + 1)
            return
        members = getattr(obj, "__dict__", None)
        if members is None:
            return
        # This declaration is a read-time invalidation contract, not a
        # claim that the cached layout itself is transported by the gather.
        for cache_name, dependencies in getattr(
                obj, "geography_cache_dependencies", {}).items():
            cache = getattr(obj, cache_name, None)
            if cache is None:
                continue
            valid = bool(dependencies)
            for live_name, snapshot_name in dependencies:
                live = getattr(obj, live_name, None)
                snapshot = getattr(obj, snapshot_name, None)
                if (id(live) not in gathered
                        or not isinstance(snapshot, np.ndarray)):
                    valid = False
                    bad.append((f"{path}.{cache_name} <- {live_name}",
                                tuple(getattr(live, "shape", ())),
                                str(getattr(live, "dtype", "missing")),
                                "cache input not gathered or snapshot missing"))
            if valid:
                bound_caches.add(id(cache))
                bound_caches.update(id(getattr(obj, snapshot))
                                    for _live, snapshot in dependencies)
            elif not dependencies:
                bad.append((f"{path}.{cache_name}", (), "missing",
                            "cache has no declared inputs"))
        for field in _SCHEME_GEOGRAPHY_ATTRS:
            value = members.get(field)
            if isinstance(value, (cp.ndarray, np.ndarray)):
                note(value, f"{path}.{field}", "scheme lat/lon")
        for key, value in list(members.items()):
            if depth == 0 and key in _physics.OUTPUT_ONLY_DRIVER_ATTRS:
                continue
            walk(value, f"{path}.{key}", depth + 1)

    # Drag statistics and the already-derived topo coefficients are geography,
    # even though their trailing horizontal axes are directly gatherable.
    # Missing one leaves a buffer's neutral fill in real forecast columns.
    for key, array in geography_inventory(state).items():
        if key.startswith("terrain_drag/"):
            note(array, key, "terrain drag")
    walk(driver, "driver", 0)
    if not bad:
        return
    lines = "\n".join(f"    {p} {s} {d}  [{fam}]" for p, s, d, fam in bad)
    raise GeographyNotGatherable(
        "this configuration DERIVES horizontally-varying geography that the "
        "per-tile gather does not reach, so every tile except the one centred "
        "on the domain would integrate a different problem:\n" + lines
        + "\n  Gather each live geography input. Derived caches must bind "
          "their retained input snapshots and refresh before use through "
          "geography_cache; a flattened-column cache cannot be repaired "
          "by widening a halo.")


def geography_window_mismatches(tile_state, parent_state, spec,
                                keys=None) -> dict[str, float]:
    """``{key: max_abs_diff}`` for geography that differs from the window.

    Empty means every entry of :func:`geography_inventory` on the tile equals
    the corresponding window of the parent -- bitwise, since the comparison
    is ``==`` and the reported number is only produced once something has
    already failed.  The windows come from the spec's own gather rectangles,
    so the staggering and the periodic wrap are handled by the same code the
    data path uses rather than by a second implementation of it.
    """
    bad: dict[str, float] = {}
    parent = geography_inventory(parent_state, keys)
    tile = geography_inventory(tile_state, keys)
    nz = int(parent_state.p.shape[0])
    for key, ref in parent.items():
        got = tile.get(key)
        if got is None:
            bad[key] = float("inf")
            continue
        kind = _gather.classify(ref.shape, nz, spec.ny, spec.nx,
                                layers_ok=True)
        host_ref = _as_host(ref)
        window = np.empty(_gather.tile_shape_for(kind, host_ref.shape,
                                                 spec.cny, spec.cnx),
                          dtype=host_ref.dtype)
        spec.apply_gather(host_ref, window, kind)
        host_got = _as_host(got)
        if window.shape != host_got.shape:
            bad[key] = float("inf")
            continue
        if not bool((window == host_got).all()):
            bad[key] = float(np.abs(window.astype(np.float64)
                                    - host_got.astype(np.float64)).max())
    return bad


def setup_scalar_mismatches(tile_state, parent_state) -> dict[str, tuple]:
    """``{name: (parent, tile)}`` for STATE_SETUP_SCALARS that disagree.

    ``mub``/``p_top``/``cf1..cfn1`` are pure functions of the vertical
    coordinate and the base sounding, so a tile rebuilds them exactly -- but
    ``mub`` flips to ``None`` the moment terrain makes ``mub2d`` the
    authority (state.py:744-751), so a tile built WITHOUT terrain against a
    parent WITH it disagrees on a scalar rather than on an array, and no
    amount of gathering would show it.  ``has_msf``/``rotational`` are
    excluded here: ``run_tiled`` imposes the domain's, and comparing them on
    a freshly built buffer would flag the very thing it fixes.
    """
    from woof.state_serialization_contract import STATE_SETUP_SCALARS

    bad: dict[str, tuple] = {}
    for name in STATE_SETUP_SCALARS:
        if name in _SETUP_FLAGS:
            continue
        want = getattr(parent_state, name, None)
        got = getattr(tile_state, name, None)
        if want is None or got is None:
            if want is not got:
                bad[name] = (want, got)
            continue
        if not np.array_equal(np.asarray(want), np.asarray(got)):
            bad[name] = (want, got)
    return bad


# --------------------------------------------------------------------------
# the assumption the whole design rests on
# --------------------------------------------------------------------------

def assert_streaming_inventory_complete(cfg, *, seed_a: int = 11, seed_b: int = 22,
                                        nsteps: int = 8, names=None,
                                        builder=None, warmup: int = 1,
                                        stream_scalars: bool = True) -> None:
    """Raise unless the streamed inventory is the ENTIRE cross-step state.

    A tiled run streams one inventory.  If any OTHER thing on a
    ``DomainState`` survives a step and is read by the next one, then reusing
    one tile buffer for many tiles leaks tile A's leftovers into tile B, and
    the tiled run cannot match a monolithic one no matter how wide the halo
    is.  This check is the structural guarantee behind the whole design: the
    hash gate can pass by luck on a short run, this cannot.

    The check is operational, not a code read.  A state is stepped to dirty
    every scratch/tendency/accumulator it owns, then ONLY the streamed set is
    overwritten with a different state's data, then both are stepped
    ``nsteps`` more.  If the streamed set really is the whole carried state,
    the results must be bit-identical.

    WHAT CHANGED, AND WHY IT HAD TO
    -------------------------------
    Through milestone one this compared ``harness.hash_field_map``, which
    walks ``STATE_SERIALIZED_ATTRS``.  That made it BLIND in exactly the
    region it was supposed to police: anything in ``state._scratch`` or on
    ``state.physics`` was never refilled (so it stayed dirty) and never
    compared (so the dirt never showed).  MEASURED: at ``moist=True,
    mp_physics=10`` the old check returned PASS while
    ``scratch/mp_rainnc`` and ``driver/pending_rainbl`` were demonstrably
    carried and unstreamed.  At any rung with ``physics_enabled(cfg)`` it did
    not run at all -- its probe states came from ``harness.make_state``,
    which attaches no ``PhysicsDriver``, so ``dycore.step`` raised first.

    It now compares -- and refills -- the whole restart manifest
    (:func:`tilestream.physics_inventory.carrier_manifest`), and builds its
    probes with a driver attached.  For a dry config that manifest IS the
    nine contract arrays, so the milestone-one meaning is unchanged; for
    every physics rung it is the 138-to-229 carriers that actually exist.
    ``nsteps`` defaults to 8 rather than 2 on this project's own precedent:
    the minimum halo was certified wrong twice by tests at N <= 3, and at a
    long radiation/cumulus cadence a 2-step run passes with the scalar
    carriers dropped purely because no cadence boundary falls inside it.

    ``names`` selects a SUBSET to stream while still comparing everything --
    which is how a candidate smaller inventory gets refuted rather than
    assumed.  Passing the milestone-one contract names at a physics rung is
    the negative control, and it fails as it should.
    """
    from tilestream import physics_inventory as _physics

    streamed = None
    if names is not None:
        streamed = list(names)
    try:
        _physics.assert_carrier_inventory_complete(
            cfg, streamed=streamed, nsteps=int(nsteps), warmup=int(warmup),
            seed_a=seed_a, seed_b=seed_b, builder=builder,
            stream_scalars=stream_scalars)
    except _physics.CarrierIncompleteError as exc:
        raise TiledRunError(str(exc)) from exc


# --------------------------------------------------------------------------
# planning helpers
# --------------------------------------------------------------------------

def _arrays_of(obj, names=None, inventory_fn=None) -> dict:
    """``{name: array}`` from a store, state or mapping.

    ``hoststore.HostDomainStore`` keeps its pinned buffers in an ``.arrays``
    dict rather than as attributes, so it is unwrapped first; everything else
    goes through ``inventory_fn`` (default
    :func:`tilestream.gather.inventory`), which handles a ``DomainState``, a
    ``gather.HostStore`` and a plain mapping alike.  A physics run passes
    :func:`tilestream.physics_inventory.carrier_inventory` instead, because
    ``getattr(state, name)`` cannot reach ``state._scratch`` or
    ``state.physics``.
    """
    take = inventory_fn or _gather.inventory
    inner = getattr(obj, "arrays", None)
    if isinstance(inner, dict):
        obj = inner
    return take(obj, names)


def plan_for(store, cfg, tile_nx, tile_ny, halo=None, *, periodic: bool = True,
             periodic_x: bool | None = None, periodic_y: bool | None = None,
             names=None, inventory_fn=None, nz=None):
    """``(specs, tile_cfg, (nz, ny, nx))`` for a run, without running it.

    Exposed so a caller can size buffers, count tiles or check plan efficiency
    before committing to an integration.  ``halo=None`` takes
    :func:`tilestream.harness.halo_radius`, which is ``16`` only when
    ``time_step_sound == 4``.
    """
    arrays = _arrays_of(store, names, inventory_fn)
    if not arrays:
        raise TiledRunError("store holds none of the persisted attributes")
    nz, ny, nx = _gather.domain_extents(arrays, nz=nz)
    if halo is None:
        halo = _harness.halo_radius(cfg)
    specs = _spec.plan_tiles(nx, ny, int(tile_nx), int(tile_ny), int(halo),
                             periodic, periodic_x=periodic_x,
                             periodic_y=periodic_y)
    _spec.validate_plan(specs, ny, nx)
    tile_cfg = _harness.tile_config(cfg, specs[0].cnx, specs[0].cny)
    return specs, tile_cfg, (nz, ny, nx)


def ring_bytes_vs_shadow(specs, nz: int, bytes_per_cell: float) -> dict[str, float]:
    """Cost of the shadow store versus the ring scheme that replaces it.

    ``shadow`` is a whole second domain.  ``ring`` is what
    :mod:`tilestream.rings` actually saves for this plan, so the ratio is
    measured from the plan rather than estimated from the halo width.

    THE ESTIMATE THIS REPLACES WAS WRONG IN BOTH DIRECTIONS, which is worth
    keeping written down.  It assumed a ``halo``-wide band on all four sides
    of every tile.  Too big, because a band read only by an EARLIER tile
    never has to be saved (the loop orders that read ahead of the write
    instead): for a 3x3 plan of 650-cell tiles the real figure is 5.2%, not
    the 9.6% the four-sided formula gives.  And too small, because the
    formula erodes by exactly ``halo``, while the staggered variants are read
    one face deeper, a ragged trailing tile is read right through, and a
    non-periodic edge tile's window slides ``2*halo`` into its neighbour --
    each of which the formula silently omits.  ``ring_estimate_bytes`` keeps
    the old number so the difference stays visible.
    """
    from tilestream import rings as _rings

    if not specs:
        raise ValueError("empty plan")
    s0 = specs[0]
    domain = float(s0.nx * s0.ny)
    h = float(s0.halo)
    estimate = 0.0
    for s in specs:
        iy, ix = float(s.interior_ny), float(s.interior_nx)
        estimate += iy * ix - max(iy - 2.0 * h, 0.0) * max(ix - 2.0 * h, 0.0)
    plan = _rings.build_ring_plan(specs)
    report = _rings.ring_report(plan)
    ring = report["ring_cells"] / max(1, len(plan.kinds))
    return {
        "shadow_bytes": domain * bytes_per_cell,
        "ring_bytes": ring * bytes_per_cell,
        "ring_fraction": ring / domain,
        "ring_estimate_bytes": estimate * bytes_per_cell,
        "ring_estimate_fraction": estimate / domain,
        "bands": report["bands"],
        "patch_blocks": report["patches"],
    }


# --------------------------------------------------------------------------
# buffers
# --------------------------------------------------------------------------

def _empty_like_store(arrays, *, poison: bool):
    """A second set of buffers with the same shapes and memory class.

    ``poison`` fills them with NaN so a scatter that fails to cover part of
    the domain shows up as NaN rather than as plausible weather.  It only
    guards the FIRST sweep -- after the swap the write target holds the
    previous sweep's real values -- which is enough, because coverage is a
    property of the plan and ``spec.validate_plan`` proves it statically for
    every sweep at once.  The NaN fill is the runtime cross-check on that
    proof, not a substitute for it.
    """
    import cupy as cp

    out = {}
    for name, arr in arrays.items():
        poison_value = None
        if poison:
            # Integer carriers (ivgtyp, isltyp, kpbl, ebal) cannot hold NaN,
            # and leaving them un-poisoned would leave the coverage
            # cross-check blind on exactly the fields whose plausible values
            # are the hardest to eyeball.  INT_MIN is not a land-use class.
            if arr.dtype.kind == "f":
                poison_value = np.nan
            elif arr.dtype.kind in "iu":
                poison_value = np.iinfo(arr.dtype).min
        if _gather.is_device_array(arr):
            buf = cp.empty_like(arr)
            if poison_value is not None:
                buf.fill(poison_value)
        else:
            buf = _gather.pinned_empty_like(arr)
            if poison_value is not None:
                buf[...] = poison_value
        out[name] = buf
    return out


def _copy_into(dst_arrays, src_arrays) -> None:
    import cupy as cp

    for name, dst in dst_arrays.items():
        src = src_arrays[name]
        if _gather.is_device_array(dst) and _gather.is_device_array(src):
            cp.copyto(dst, src)
        elif _gather.is_device_array(dst):
            dst.set(np.ascontiguousarray(src))
        elif _gather.is_device_array(src):
            src.get(out=dst)
        else:
            dst[...] = src
    cp.cuda.runtime.deviceSynchronize()


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------

class TiledRun:
    """A domain streamed tile-by-tile through the GPU, one SWEEP per call.

    A ``TiledRun`` owns everything that must survive between model steps --
    the tile buffers, their streams, the ring arena, each buffer's geography
    and boundary occupancy, and the domain clock -- and :meth:`sweep`
    advances the store by one model step (or ``nsteps`` of them).  The split
    exists because WOOF's own run loop has to BE the loop: woof.core.model
    and woof.runtime decide when output, restart, diagnostics and nest
    coupling happen, and a streaming mode that owned the time loop could only
    ever be a parallel universe beside the model, never a mode of it.

    It is not a style question.  MEASURED: constructing the tile buffers
    costs 3.3 s for a 544-cell dry buffer, so driving a model step with
    ``run_tiled(..., nsteps=1)`` reads 8.7x slower over four steps than one
    call of four.  Worse, it silently breaks the property the geography
    argument rests on -- a buffer's geography is gathered when that buffer
    CHANGES TILE, and buffers rebuilt every step change tile every step, so
    "gather once per buffer" becomes "gather once per tile per step" and the
    read-only arrays turn into the dominant transfer.

    :func:`run_tiled` is this class constructed and swept once, and remains
    the surface the gate drives.

    ``store`` holds the full domain -- a
    :class:`tilestream.hoststore.HostDomainStore`, a
    :class:`tilestream.gather.HostStore`, a plain ``{name: array}`` mapping or
    even a ``DomainState``.  Host buffers must be pinned (pass
    ``allow_pageable=True`` only to debug; pageable transfers run at a quarter
    of the speed and are not truly asynchronous).  The store is updated IN
    PLACE and nothing is returned; pass ``report={}`` for the diagnostics.

    ``tile_nx``/``tile_ny`` are INTERIOR extents.  Each tile is stepped over
    ``tile_nx + 2*halo`` by ``tile_ny + 2*halo`` mass cells and only its
    interior is written back.  ``halo`` must be at least the per-step
    dependency radius, ``tilestream.harness.halo_radius(cfg)``; the default 16
    is correct for ``time_step_sound == 4`` and too small above it.

    ``nbuffers`` tile states and streams are cycled so tile *i+1*'s gather
    overlaps tile *i*'s compute.  Within a sweep the tiles are completely
    independent (they read one buffer and write another, over disjoint
    interiors), so the only ordering required is a sync at the end of each
    sweep.  ``nbuffers`` changes no arithmetic and must not change the answer;
    the gate checks that it does not.

    ``pipeline`` chooses the order the copies are SUBMITTED in, which on a
    single-copy-engine card decides whether any of them overlap at all -- see
    the block comment on the loop below, and the 1.30x it is worth.
    ``"prefetch"`` (default) issues tile *i+depth*'s gather before tile *i*'s
    scatter; ``"naive"`` is the obvious per-tile order and is kept so the
    difference stays measurable.  Neither changes the answer.

    ``periodic`` sets both axes; ``periodic_x`` / ``periodic_y`` override one
    and are what a MIXED domain needs.  ``cfg.open_x=True, open_y=False`` --
    radiative-open across the flow, periodic along it -- is non-periodic in x
    and wrapping in y, and a plan that clamps y hands the south-most tile an
    owned row 0 with no halo beneath it while the kernels stepping it wrap to
    the far side of the WINDOW.  Take them from
    ``autoplan.is_periodic_x``/``is_periodic_y``, which are the dycore's own
    ``_boundary_x``/``_boundary_y`` negated; never guess.

    ``write_mode`` picks how time-t neighbour values are kept alive:
    ``"ring"`` (default) saves each tile's ring into a small arena and
    patches the gathers from it, keeping ONE store; ``"shadow"`` keeps a
    whole second store; ``"inplace"`` keeps neither and is wrong on purpose.
    See the module docstring.  ``ring_margin`` selects a deliberately broken
    ring for the gate's negative controls (``"halo"``, ``"x_only"``; see
    :func:`tilestream.rings.build_ring_plan`) and must be ``"exact"`` for a
    forecast.

    ``ring_ordering`` decides how the two cross-stream hazards are enforced:
    ``"events"`` (default) records one event per tile and waits on the
    neighbours the plan names, which is correct on any hardware;
    ``"submission"`` relies on this card's single DMA queue draining in
    submission order.  MEASURED, and the answer is why the default is what it
    is: ``"submission"`` is bit-exact on an IDLE card and WRONG BY 3.7e+02 on
    the same 3x3 plan while another process shares the GPU.  It is kept only
    as that measurement.  Never use it for a forecast.

    ``overlap`` moves the ring sweep's copies onto DEDICATED STREAMS so they
    can run beside the compute: per buffer, a copy-in stream (geography,
    boundary bind, carrier gather, ring patch), the compute stream, and a
    copy-out stream (ring save, scatter), ordered by the event chain
    :mod:`tilestream.overlap` derives from the ring plan and re-checks
    against the plan's own rectangles at construction.  When nothing armed
    on the run needs a hard sweep barrier (no graphs, no health fold, no
    timeline, no ``on_sweep``), the end-of-sweep synchronization is deferred
    too: the next step's gathers chain on the previous step's scatter events
    and the pipeline stays primed across ``sweep()`` calls, which is the
    single biggest exposed term OVERLAP-ATTRIBUTION.md measured.  Anything
    that reads the store between sweeps goes through :attr:`store` or
    :meth:`drain`.  ``"on"`` (default) changes no arithmetic and must not
    change the answer -- the gate checks that it does not; ``"off"`` is the
    single-stream loop, kept as the reference; ``"unchained"`` (WRONG ON
    PURPOSE) keeps the dedicated streams and drops every cross-stream wait,
    the negative control that shows the chain is essential.  A non-ring
    ``write_mode`` degrades ``"on"`` to ``"off"``: the chain is derived from
    a RingPlan and the shadow already pays for its ordering with a second
    store.  ``ring_ordering="submission"`` also degrades to ``"off"`` -- it
    is kept as a measurement of the single-stream loop's single DMA queue,
    and on dedicated copy streams that queue does not exist, so the
    observation only means anything on the loop it was made on.  The
    broken-on-purpose ring margins skip the construction-time hazard check
    (their plans are unsound by construction and exist to be caught by the
    digest); a forecast is always ``ring_margin="exact"`` and always gets
    the check.

    ``inventory_fn`` / ``nz`` / ``scalars`` are the three things a PHYSICS run
    needs and a dry one does not; :func:`physics_run_kwargs` supplies all
    three plus the tile factory.

    ``inventory_fn``
        How a store and a tile state are turned into ``{name: array}``.  The
        default reaches ``STATE_SERIALIZED_ATTRS`` by ``getattr`` and cannot
        see ``state._scratch`` or ``state.physics`` at all; pass
        :func:`tilestream.physics_inventory.carrier_inventory` to stream the
        whole restart manifest.

    ``nz``
        The vertical extent, when the inventory holds layered fields whose
        leading axis is not vertical (Noah's 4 soil levels, Noah-MP's 3 snow
        levels and 7-deep snow-soil coordinate).  Without it
        ``gather.domain_extents`` would have to infer nz, and for that
        inventory the accurate answer is that it cannot.

    ``scalars``
        The DOMAIN's :func:`tilestream.physics_inventory.carrier_scalars`
        -- ``elapsed_seconds`` and the driver's call counters.  Restored onto
        the buffer before EVERY tile step and advanced once per sweep, not
        once per tile.  Without this a buffer serving k tiles runs k*dt ahead
        of the domain inside a single sweep, so tiles disagree about whether
        radiation/cumulus/PBL is due and integrate different physics -- with
        no NaN, no warning and a perfectly plausible answer.  Updated in
        place, so the caller's dict reflects the domain clock on return.

    ``geography``
        The DOMAIN's :func:`geography_inventory` -- map factors, Coriolis,
        rotation, terrain, the terrain-following base state and the scheme
        latitude/longitude grids.  READ-ONLY: gathered into a buffer when
        that buffer starts serving a different tile, never scattered, never
        re-sent per step.  ``has_msf``/``rotational`` are computed from the
        DOMAIN (see :func:`geography_scalars`) and imposed on every buffer,
        because ``set_map_coriolis`` would otherwise derive them from the
        tile's own window.  Omitting it leaves every tile with the geography
        it REBUILT from ``tile_cfg`` -- centred on the tile rather than on
        the domain, correct only for the exactly-centred tile, and that is
        the gate's negative control.

    ``tile_hook``
        ``tile_hook(tile_state, tspec, itile, stream)``, called when a buffer
        starts serving a different tile -- the same condition the geography
        gather fires on, and inside that buffer's stream, before the carrier
        gather.

        It exists for LATERAL BOUNDARIES, which are the one per-tile input
        the array transport cannot express.  Geography is an inventory of
        arrays on the domain grid and windows like any carrier; the four
        boundary side tables have a DIFFERENT SHAPE PER SIDE, are attached
        through ``woof.ingest.lateral_bc.attach_lateral_boundaries`` (which
        uploads into state-owned scratch and installs a device descriptor
        rather than writing an inventory array), and which side is a TRUE
        DOMAIN EDGE versus an interior seam is a property of the
        :class:`~tilestream.spec.TileSpec`, not of the ``tile_cfg`` the tile
        factory is handed.

        A tile's true edges take the domain's own tables, sliced along the
        tangential axis; its interior seams take tables that are inert.
        MEASURED, at 256x192x49, tile 32x32, real Lambert + real terrain +
        specified BCs, halo 16 from ``harness.halo_radius``: seam tables of
        ZEROS, seam tables holding the domain's own coupled values with zero
        tendency, and seam tables of DELIBERATE GARBAGE (1e6 in coupled
        units, 1e4/s of tendency) all give the BIT-IDENTICAL answer, out to
        N=24 steps.  The seam relaxation provably cannot reach the interior.
        The controls that must fire do: tables not windowed at all, and true
        edges scaled by 1.000001, both differ in all nine dry carriers.

    ``observer``
        ``observer(tile_state, tspec, itile, stream)``, called inside that
        tile's stream immediately AFTER its step and before its interior is
        scattered back -- the one instant in a sweep at which a buffer holds
        real, current domain data on the card.

        It exists for the run loop's SAFETY OBSERVERS.  ``woof.runtime
        .integrate_prepared_case`` guards a forecast with a whole-domain
        ``stability_report`` every dynamics substep and with
        ``StateHealthValidator.require_healthy`` on a cadence, and it hands
        both the prepared ``DomainState``.  Under a HOST store that state is
        a copy taken at t=0 (``streaming.attach`` fills the store with
        ``gather.pinned_copy``) which no sweep ever writes again, so the NaN
        guard, the w_max monitor and the CFL monitor all observe a snapshot
        that is healthy by construction: ``nan_free`` stays true, ``w_max``
        stays at its initial value, and a domain that went non-finite in the
        store completes and checkpoints as a clean forecast.

        The observers are pure -- they carry nothing into the answer -- which
        is exactly why nothing caught it.  They are also max/OR folds, so
        they are associative, and tile interiors PARTITION the domain: one
        record per tile, folded, IS the whole-domain record.  Keeping the
        DomainState current instead is not an option and not a shortcut that
        was skipped -- the premise of the mode is that the domain does not
        fit on the card, so there is no resident copy to refresh.  Every
        whole-domain observer either becomes a fold or stops being armed.

        The observer must read the INTERIOR, never the whole buffer.  The
        halo is at least the per-step dependency radius so the interior is
        bit-exact, but the halo itself was stepped with insufficient
        neighbours; folding it in would let a discarded halo raise a NaN the
        domain never had, and a false alarm on the run's only safety gate is
        no better than a missing one.

    ``post_step_hook``
        ``post_step_hook(tile_state, tspec, itile, stream)``, called on EVERY
        tile immediately after its ``dycore.step`` and before the scatter,
        inside that buffer's stream.  ``tile_hook`` fires only when a buffer
        changes tile, which is the right cadence for an INPUT; this one fires
        every step, which is the right cadence for something a step PRODUCES.

        It exists for ONE-FRAME PRODUCER/CONSUMER HANDOFFS, of which the
        model has exactly one: ``woof.core.refl.stash_refl_10cm`` parks the
        step's REFL_10CM on the physics driver and refuses to overwrite an
        unconsumed handoff, because in a resident run a second unconsumed
        field IS a cadence bug.  In a sweep the handoff is per TILE and the
        frame is per DOMAIN, so the SECOND tile a buffer serves raises
        "REFL_10CM stash was not consumed before reuse" -- an error whose
        text describes a defect that is not present.  The values are not in
        the handoff: ``stash_refl_10cm`` stores a reference to the
        state-owned ``refl_10cm`` scratch slot, and it is that slot the
        scatter carries.  So the hook clears the reference and the transport
        joins the four tile windows exactly as it joins everything else.
        :func:`woof.core.streaming.attach` installs it; see
        :func:`woof.core.streaming.refl_handoff_hook`.

        THE SAME INSTANT AS ``observer``, and deliberately a second
        parameter rather than one shared slot.  Two branches invented this
        hook independently for two different owners -- a READER that folds
        the safety record out of the buffer, and a WRITER that clears the
        reflectivity handoff -- and a streamed forecast wants both at once.
        Collapsing them would make one consumer wrap the other and decide
        the ordering on the other's behalf.  ``observer`` runs first: it
        only reads, so it cannot be disturbed by the hook, while the hook
        mutates the buffer's driver.

    ``check_geography``
        Run :func:`assert_geography_gathered` on the first buffer, once, so a
        scheme that derives geography the transport cannot window (legacy
        RRTMG's ozone cache) is refused rather than silently rebuilt.

    ``tile_hook(tile_state, tspec, itile, stream)``
        Called when a buffer STARTS SERVING A DIFFERENT TILE, immediately
        after that buffer's geography gather is issued and before its
        carriers are gathered -- the one place per (buffer, tile) pair that
        exists.  It is the seam a SPECIFIED domain needs:
        :func:`tilestream.realcase.tile_boundary_binder` uses it to swap the
        buffer's lateral-boundary tables to the ones cut to THIS tile's
        window.  Nothing in the periodic lane passes it and the default
        ``None`` is exactly the old behaviour.

    ``impose_geography_flags=False`` (WRONG ON PURPOSE)
        Leaves ``has_msf``/``rotational`` at whatever the BUFFER derived from
        its own build, so a buffer built on
        :func:`tilestream.harness.neutral_geography` runs with the Coriolis
        kernel and the msf-weighted paths switched off while holding the
        domain's real map factors.  It exists so the gate can show that the
        imposition is essential rather than decorative.

    ``use_graph`` captures each buffer's step as a CUDA graph and REPLAYS it
    for every other tile that buffer serves.  This is the one optimisation
    that is worth more to a tiled sweep than to a resident run: the step's
    ~1,200 kernel launches are re-issued once per TILE, so the launch
    overhead is multiplied by the tile count, and it is worst on the small
    compute windows a small card is forced into.  ``True``/``"auto"`` falls
    back to ordinary stream launching for a step that cannot be captured and
    reports why; ``"require"`` raises instead, which is what a benchmark
    wants.  See :mod:`tilestream.graphcap` for the four ways a graph goes
    silently wrong and the mechanism that answers each.

    ``graph_reuse``
        ``"sweep"`` (default) keys the graph cache on the SWEEP as well as
        the cadence, so a graph is captured once per step and replayed once
        per tile -- which is exactly the re-use a tiled run needs, since the
        launch cost is what the tile count multiplies.  ``"run"`` re-uses a
        graph across steps of the same cadence: much cheaper (MEASURED at
        96x80, 2x2, N=8: 2 captures totalling 130 ms against 16 totalling
        423 ms) and sound only while no radiation-due step falls inside the
        run, because a graph bakes in every kernel's scalar arguments and
        RRTMGP takes the solar hour angle as one.  Bit-exact at mp10, +YSU
        and the ship config over N=8 at radt=12 min; WRONG at radt=0.05 min,
        which the gate runs as a negative control.

    ``graph_reuse="run"``, ``graph_key="none"`` and ``graph_scalars=False``
    are the gate's three negative controls: re-use a graph across steps,
    ignore the cadence as well, and skip the scalar-carrier increment a
    replay owes.  All three must produce a mismatch, and the gate asserts it
    -- and refuses to count an out-of-memory as a detection, because three
    of them once "passed" that way on a starved card.

    ``shared`` / ``chain_compute``
        The tile buffers' step-local scratch, rebuilt state symbols and
        RRTMGP chunk workspace can be ONE allocation instead of ``nbuffers``
        of them -- see :mod:`tilestream.shared_workspace`, and note that
        98.2% of a buffer's device footprint was per-buffer before it
        existed.  Pass the same
        :class:`~tilestream.shared_workspace.SharedTileWorkspaces` here that
        the tile factory was built with (``physics_run_kwargs(...,
        shared=...)`` does both).

        Passing it here does exactly one thing to the loop: it CHAINS THE
        COMPUTE.  Each buffer owns a non-blocking stream, and tile *i*'s
        ``step`` on stream ``i % nbuffers`` has nothing ordering it against
        tile *i+1*'s ``step`` on the next stream -- they are independent by
        construction, which is why the pipeline exists and why two of them
        may be executing at once.  Two concurrent steps sharing one arena
        would be writing the same bytes.  So an event is recorded after each
        ``step`` and the next tile's stream waits on it.  Copies are NOT
        chained, so the prefetch order still hides the transfer behind the
        compute; only compute/compute overlap is given up.

        ``chain_compute`` defaults to ``shared is not None`` and exists to be
        set to ``False`` WITH ``shared`` set: that is the negative control,
        it is the reclaimed buffer being reused while it is still live.

    ``timeline``
        Record a CUDA event either side of every tile's ``step`` and report
        ``overlapping_steps`` -- how many tiles began computing before their
        predecessor had finished.  WHETHER TWO TILES OVERLAP IS A PROPERTY OF
        THE CARD, NOT OF THIS CODE, and measuring it is the only portable way
        to state the safety property.  MEASURED at 96x96x49 tiles, dry, 3x3,
        nbuffers=2: on an RTX 5090 the unchained loop overlaps and a shared
        arena makes all nine carriers differ, four runs of four; on an RTX
        4090 the same run does NOT overlap and comes out bit-exact, because
        one tile already occupies the whole card.  A control that only
        watched the answer would have called the 4090 safe.  Off by default:
        it costs two events per tile and a synchronization at the end.

    Raises ``TiledRunError`` for a setup that cannot be right.  Warns, loudly,
    if ``write_mode="inplace"`` is selected -- see the module docstring.
    """

    def __init__(self, store, cfg, tile_nx, tile_ny, halo: int = 16,
                 nbuffers: int = 2, *, periodic: bool = True,
                 periodic_x: bool | None = None,
                 periodic_y: bool | None = None,
                 write_mode: str = "ring", shadow=None,
                 tile_state_factory=None, tile_states=None, names=None,
                 allow_pageable: bool = False, poison: bool = True,
                 pipeline: str = "prefetch", inventory_fn=None,
                 nz=None, scalars=None, geography=None,
                 geography_names=None, check_geography: bool = True,
                 tile_hook=None, observer=None, post_step_hook=None,
                 impose_geography_flags: bool = True,
                 ring_margin: str = "exact",
                 ring_ordering: str = "events",
                 overlap: str = "on",
                 health_width: int | None = None,
                 shared=None, chain_compute=None,
                 timeline: bool = False,
                 use_graph=False, graph_reuse: str = "sweep",
                 graph_key: str = "cadence", graph_scalars: bool = True,
                 graph_verify_host: bool = True,
                 graph_verify_topology: bool = False,
                 on_sweep=None, global_keys=()) -> None:
        import cupy as cp

        # Every tile buffer and stream belongs to the construction card.
        # Later store reads may arrive from a thread still on card 0.
        self._device_id = int(cp.cuda.Device().id)

        from woof.core.dycore import (step, begin_wrf_cfl_domain_step,
            set_wrf_cfl_tile_window, finish_wrf_cfl_tile,
            finish_wrf_cfl_domain_step)

        # SETUP IS NOT A PER-STEP COST.  Tile buffers, the ring arena and the
        # transfer plans are built ONCE -- here -- and then serve every sweep,
        # but a benchmark that constructs a TiledRun per timed rep divides that
        # one-off over its handful of steps and reports it as streaming
        # overhead.  At the physics rungs the setup is a whole extra physics
        # state per buffer -- ``initialize_physics`` plus a warmup step that
        # fires radiation -- so it is tens of seconds while a step is tenths,
        # and it swamps the answer.  ``report['sweep_seconds']`` therefore
        # carries the wall time of each sweep on its own, measured between full
        # device synchronizations, and ``setup_seconds``/``factory_seconds``
        # carry the one-off separately so neither can be lost.  From the 8x4090
        # box, where ``bench_window`` was written precisely because the
        # benchmark before it had timed the setup.
        _t_call = _time.perf_counter()

        if write_mode not in ("ring", "shadow", "inplace"):
            raise ValueError(
                f"write_mode must be 'ring', 'shadow' or 'inplace', "
                f"got {write_mode!r}")
        if shadow is not None and write_mode != "shadow":
            raise TiledRunError(
                f"a shadow buffer was supplied but write_mode={write_mode!r}; "
                "the ring path keeps a single store and would silently "
                "ignore it")
        if pipeline not in ("prefetch", "naive"):
            raise ValueError(f"pipeline must be 'prefetch' or 'naive', "
                             f"got {pipeline!r}")
        if ring_ordering not in ("events", "submission"):
            raise ValueError(
                f"ring_ordering must be 'events' or 'submission', "
                f"got {ring_ordering!r}")
        if overlap not in ("on", "off", "unchained"):
            raise ValueError(
                f"overlap must be 'on', 'off' or 'unchained', "
                f"got {overlap!r}")
        # The overlap chain is the RING's: it is derived from a RingPlan's
        # rectangles, the shadow already pays for its ordering with a second
        # store and inplace is wrong on purpose.  Degrading (rather than
        # refusing) keeps 'on' a safe default for every write_mode.
        if write_mode != "ring":
            overlap = "off"
        # ``ring_ordering="submission"`` is kept as a MEASUREMENT of the
        # single-stream loop's single DMA queue; on dedicated copy streams
        # that queue does not exist, so the observation is only reproducible
        # on the loop it was made on.  Degraded rather than refused, for the
        # same reason a non-ring write_mode is: the mode names the legacy
        # loop, and the gate's observation row must still be able to run it.
        if ring_ordering == "submission":
            overlap = "off"
        if overlap == "unchained":
            warnings.warn(
                "overlap='unchained' issues the sweep's copies on dedicated "
                "streams WITHOUT the event chain that orders them against "
                "the compute and each other.  It exists as the negative "
                "control for the chain.  Results are not a forecast.",
                RuntimeWarning, stacklevel=2)
        nbuffers = max(1, int(nbuffers))
        chain = ((shared is not None) if chain_compute is None
                 else bool(chain_compute))

        home = _arrays_of(store, names, inventory_fn)
        global_keys = tuple(global_keys)
        if not global_keys:
            # The 0-d domain-owned carriers (tilestream.global_inventory:
            # the SCM forcing scalars) when the caller names none.
            from tilestream.global_inventory import global_keys as _owned
            global_keys = tuple(key for key in _owned(home)
                                if getattr(home[key], "ndim", 1) == 0)
        if global_keys:
            from tilestream.global_inventory import global_keys as domain_global_keys
            if set(global_keys) != set(domain_global_keys(home)):
                raise TiledRunError("global transfer keys differ from the domain owners' complete inventory")
        if not home:
            raise TiledRunError("store holds none of the persisted attributes")
        nz, ny, nx = _gather.domain_extents(home, nz=nz)
        for axis, got, want in (("nz", nz, cfg.nz), ("ny", ny, cfg.ny),
                                ("nx", nx, cfg.nx)):
            if int(got) != int(want):
                raise TiledRunError(
                    f"store {axis}={got} but cfg.{axis}={want}; the "
                    "config that steps the tiles must describe the same "
                    "domain the store holds")

        specs = _spec.plan_tiles(nx, ny, int(tile_nx), int(tile_ny),
                                 int(halo), periodic,
                                 periodic_x=periodic_x, periodic_y=periodic_y)
        _spec.validate_plan(specs, ny, nx)
        # A buffer beyond the tile count is never selected: ``b = itile %
        # nbuffers`` cannot reach it and the sweep's clock epilogue already
        # slices to ``min(nbuffers, len(specs))``.  Building it costs a whole
        # window of VRAM and, at a physics rung, a whole state build plus a
        # cadence-firing warmup step.  Clamping is arithmetic-neutral.
        nbuffers = min(nbuffers, len(specs))
        cnx, cny = specs[0].cnx, specs[0].cny
        tile_cfg = _harness.tile_config(cfg, cnx, cny)

        need = _harness.halo_radius(cfg)
        if int(halo) < need:
            warnings.warn(
                f"halo={halo} is below the per-step dependency radius "
                f"{need} for time_step_sound={cfg.time_step_sound}; tile "
                "interiors will be silently wrong (and faster, which is "
                "how this bug hides)",
                RuntimeWarning, stacklevel=2)

        factory = tile_state_factory or make_tile_state
        take = inventory_fn or _gather.inventory
        # ``tile_states=`` HANDS THE RUN ITS BUFFERS instead of building them.
        # From the 8x4090 box: a benchmark that sweeps the same plan many times
        # pays a whole physics state per buffer on every construction, which at
        # the physics rungs is tens of seconds against a step of tenths.  A
        # caller that already holds correctly-shaped buffers can lend them.
        # The shape check is not optional -- a buffer built for a different
        # plan's compute window gathers and scatters the wrong rectangle, which
        # is a silently wrong forecast rather than a crash.
        if tile_states is None:
            tiles = [factory(tile_cfg) for _ in range(nbuffers)]
        else:
            if tile_state_factory is not None:
                raise TiledRunError(
                    "tile_states= and tile_state_factory= were both supplied; "
                    "the factory would be ignored, so say which one is meant")
            # Materialised ONCE: tile_states is allowed to be any iterable,
            # and consuming a generator twice would report "0 buffers" in the
            # error message for a caller that supplied plenty.
            supplied = list(tile_states)
            tiles = supplied[:nbuffers]
            if len(tiles) != nbuffers:
                raise TiledRunError(
                    f"tile_states supplied {len(supplied)} buffers but "
                    f"nbuffers={nbuffers} are needed")
            for t in tiles:
                got = tuple(int(v) for v in t.thp.shape[1:])
                if got != (cny, cnx):
                    raise TiledRunError(
                        f"a supplied tile buffer is {got[0]}x{got[1]} but the "
                        f"plan's compute window is {cny}x{cnx}")
        cp.cuda.runtime.deviceSynchronize()
        _t_factory = _time.perf_counter()
        tile_inv = [take(t, names) for t in tiles]
        for inv in tile_inv:
            if set(inv) != set(home):
                raise TiledRunError(
                    f"tile state inventory {sorted(inv)} != store inventory "
                    f"{sorted(home)}; missing on tile "
                    f"{sorted(set(home) - set(inv))}, extra on tile "
                    f"{sorted(set(inv) - set(home))}.  Two carriers are "
                    "allocated LAZILY on first use (Kain-Fritsch's "
                    "cumulus/w0avg above all), so both the store and the "
                    "tile buffers must be built from a state that has "
                    "already run one step.")

        # GEOGRAPHY. Bound before any buffer is stepped, because a step reads
        # msft/f/ht on its very first kernel. Never scattered: it is INPUT, and
        # MEASURED read-only across 8 steps at the 229-carrier rung with a real
        # Lambert projection and real terrain (all 33 arrays bit-identical,
        # with both negative controls firing).
        geo_home = None
        geo_tile: list[int | None] = [None] * nbuffers
        geo_fields = 0
        if geography is not None:
            geo_home = geography_inventory(geography, geography_names)
            if not geo_home:
                raise TiledRunError(
                    "geography= holds no gatherable arrays; pass the DOMAIN's "
                    "driver.geography_inventory(parent_state), not the config")
            gz, gy, gx = _gather.domain_extents(geo_home, nz=nz)
            if (gy, gx) != (ny, nx):
                raise TiledRunError(
                    f"geography describes a {gy}x{gx} domain but the "
                    f"store holds {ny}x{nx}; the two must be the same "
                    "domain")
            flags = geography_scalars(geo_home)
            for tile in tiles:
                _pin_scheme_geography(tile)
                dst = geography_inventory(tile, geography_names)
                if set(dst) != set(geo_home):
                    raise TiledRunError(
                        f"tile geography inventory {sorted(dst)} != domain "
                        f"geography inventory {sorted(geo_home)}; missing "
                        f"on tile {sorted(set(geo_home) - set(dst))}, "
                        f"extra on tile {sorted(set(dst) - set(geo_home))}."
                        "  A tile buffer must be built with the same "
                        "physics selectors as the domain, so it owns the "
                        "same scheme lat/lon grids.")
                # state.py:799-803 derives these from .any() over whatever
                # window set_map_coriolis was handed. The domain's are the only
                # correct answer and nothing downstream recomputes them.
                if impose_geography_flags:
                    for name, value in flags.items():
                        setattr(tile, name, value)
            geo_fields = len(geo_home)
            if check_geography:
                assert_geography_gathered(tiles[0], keys=geography_names)

        # Noah-MP's sparse land layout belongs to a footprint, not a reusable
        # GPU buffer. Classify all footprints once before integration. Only
        # pinned 2-D metadata survives here; parameter expansion stays on GPU.
        noahmp_footprints = {}
        for itile, tspec in enumerate(specs):
            b = itile % nbuffers
            physics = getattr(tiles[b], "physics", None)
            params = getattr(physics, "noahmp_params", None)
            if params is None or not getattr(params, "static_surface_identity", False):
                continue
            identity_names = ("xland", "xice", "ivgtyp", "isltyp")
            source = {name: home["fields/" + name] for name in identity_names}
            target = {name: physics.fields[name] for name in identity_names}
            plan = _gather.make_plan(source, target, tspec, "gather",
                                     allow_pageable=allow_pageable, nz=nz)
            plan.execute(source, target)
            if params.prepare_surface_footprint(
                    itile, physics.fields, nsoil=physics.fields["tslb"].shape[0],
                    sf_urban_physics=int(getattr(cfg, "sf_urban_physics", 0))):
                noahmp_footprints[b] = params

        ring = None
        fire_shadow = None
        if write_mode == "inplace":
            warnings.warn(
                "write_mode='inplace' feeds each tile's NEW interior into the "
                "next tile's halo. This is the read-at-time-t bug and is only "
                "correct when there is exactly one tile. Results are not a "
                "forecast.", RuntimeWarning, stacklevel=2)
            other = None
        elif write_mode == "ring":
            from tilestream import rings as _rings
            from tilestream.fire_inventory import fire_field_layout

            other = None
            fire_home = {name: arr for name, arr in home.items()
                         if fire_field_layout(name, arr.shape, cfg.ny, cfg.nx) is not None}
            ring_home = {name: arr for name, arr in home.items()
                         if name not in fire_home and name not in global_keys}
            if fire_home:
                # Preserve a complete old-time fire plane. Atmospheric rings
                # carry periodic staggered geometry; SFIRE instead continues
                # its physical boundary halo. Both read the same old step.
                fire_shadow = _empty_like_store(fire_home, poison=False)
            kinds = sorted({k for _n, k, _d, _i
                            in _rings.field_geometry(ring_home, nz=nz)})
            ring_plan = _rings.build_ring_plan(specs, kinds,
                                               margin_mode=ring_margin)
            ring = _rings.RingArena(ring_plan, ring_home, tiles, nz=nz, names=names,
                                    inventory_fn=inventory_fn,
                                    allow_pageable=allow_pageable)
        else:
            other = shadow if shadow is not None else _empty_like_store(
                {name: arr for name, arr in home.items() if name not in global_keys}, poison=poison)
            if not isinstance(other, dict):
                other = _arrays_of(other, names, inventory_fn)
            other.update({name: home[name] for name in global_keys})
            for name, arr in home.items():
                if (name not in other
                        or tuple(other[name].shape) != tuple(arr.shape)):
                    raise TiledRunError(
                        f"shadow buffer is missing or mis-shaped for {name!r}")

        # THE GRAPH STEPPERS, one per buffer.  One per buffer and not one per
        # run, because two buffers step CONCURRENTLY on two streams and a graph
        # bakes in the addresses of the temporaries its capture allocated: a
        # shared capture pool would hand both buffers the same scratch and the
        # two replays would overwrite each other, with no error and a plausible
        # answer.  Each stepper also owns its own health ledger for the same
        # reason -- a shared status word is a cross-stream read-modify-write that
        # can only lose a fault flag.
        graph_steppers = None
        if use_graph and not cfg.diag_print:
            from tilestream import graphcap as _graphcap
            from tilestream import physics_inventory as _pi

            graph_steppers = [
                _graphcap.GraphStepper(
                    tile_cfg,
                    mode=("require" if use_graph == "require" else "auto"),
                    reuse=graph_reuse,
                    key_fn=(_graphcap.cadence_key if graph_key == "cadence"
                            else (lambda _s, _c: ("fixed",))),
                    replay_scalars=bool(graph_scalars),
                    verify_host=bool(graph_verify_host),
                    verify_topology=bool(graph_verify_topology),
                    scalars_fn=_pi.carrier_scalars,
                    set_scalars_fn=_pi.set_carrier_scalars)
                for _ in range(nbuffers)]
            if graph_key not in ("cadence", "none"):
                raise ValueError(
                    f"graph_key must be 'cadence' or 'none', got {graph_key!r}")

        # THE SHARING AND THE CHAIN MUST AGREE, and the dangerous direction
        # is the one where they silently do not: a factory that hands out
        # buffers backed by one arena while this run was not told, so the
        # compute is never chained and two tiles write the same scratch.
        # That configuration is bit-exact at full physics today only because
        # the physics guards host-synchronise inside the step (see
        # tilestream/test_share.py), which is not a property anything
        # promises.  So it is refused here rather than left to be discovered.
        arenas = {id(getattr(t, "_scratch_arena", None)) for t in tiles}
        want = id(None if shared is None else shared.arena)
        if arenas != {want}:
            raise TiledRunError(
                "the tile buffers' scratch arena does not match the "
                f"``shared`` argument: buffers carry {len(arenas)} distinct "
                f"arena identities and "
                f"shared={'None' if shared is None else 'set'}"
                ". Build the buffers and this call from the SAME "
                "shared_workspace.SharedTileWorkspaces (physics_run_kwargs("
                "..., shared=...) does both), or pass neither.")

        # THE COMPUTE CHAIN.  ``None`` until the first tile has been stepped;
        # thereafter it is the event that says "the previous tile's step has
        # finished on the device", which is the only thing that makes one
        # shared scratch arena legal across nbuffers buffers.  It is carried
        # ACROSS sweeps deliberately: the last tile of sweep k and the first
        # tile of sweep k+1 use the same buffer index only when
        # len(specs) % nbuffers == 0, so a per-sweep reset would leave
        # exactly the ragged case unordered.
        compute_done = None

        # THE TIMELINE.  ``(start, end)`` CUDA events around every tile's
        # step, all measured against one reference event so intervals from
        # different streams are comparable.  Recorded only when asked for;
        # the events themselves are free to record and the elapsed-time reads
        # happen after the final synchronization, outside anything timed.
        marks: list[tuple] = []
        origin = None
        if timeline:
            cp.cuda.runtime.deviceSynchronize()
            origin = cp.cuda.Event()
            origin.record()
            origin.synchronize()

        streams = [cp.cuda.Stream(non_blocking=True) for _ in range(nbuffers)]
        # One event per tile, recorded after that tile's gather AND its ring
        # save. Both cross-tile hazards are covered by that single point: a
        # later tile's scatter needs this tile's READ of the store to have
        # happened (WAR), and a later tile's patch needs this tile's SAVE to
        # have landed (RAW).  LEGACY: only the single-stream loop uses this
        # single point; the overlap loop splits it, because a save that has
        # moved to the copy-out stream is no longer covered by an event on
        # the gather's stream.
        ring_events = ([cp.cuda.Event() for _ in specs]
                       if ring is not None and ring_ordering == "events"
                       and overlap == "off"
                       else None)

        # THE OVERLAP MACHINERY: two dedicated copy streams per buffer and
        # one event per (tile, operation) class.  The compute stream keeps
        # the step, the hooks and the observers; copy-in takes everything
        # H2D-shaped (geography, boundary bind, carrier gather, ring patch)
        # and copy-out everything D2H-shaped (ring save, scatter), so on a
        # card with a copy engine per direction all three queues can run at
        # once.  What orders them is tilestream.overlap's event chain --
        # see that module for the six structural edges and the four
        # plan-shaped hazard lists, and OVERLAP-ATTRIBUTION.md for the
        # measured exposure this exists to hide.  ``overlap='unchained'``
        # keeps the streams and drops every cross-stream wait: the negative
        # control that shows the chain is essential.
        sched = None
        copy_in = copy_out = None
        ev_gather = ev_save = ev_ready = ev_stepped = ev_scatter = None
        chained = False
        if overlap != "off" and ring is not None:
            from tilestream import overlap as _overlap

            sched = _overlap.OverlapSchedule.from_plan(ring.plan)
            chained = overlap == "on"
            if chained and ring_margin == "exact":
                # The independent re-check, at construction: a schedule with
                # a missing edge is a refusal at setup, never a plausible
                # forecast.  Skipped -- deliberately -- by "unchained", and
                # by the broken-on-purpose ring margins: their plans are
                # UNSOUND BY CONSTRUCTION, the checker would refuse them
                # before they could run, and their whole reason to exist is
                # to run and be caught by the digest.  A forecast is always
                # ring_margin="exact", so a forecast always gets the check.
                _overlap.assert_schedule_covers_hazards(sched, ring.plan)
            copy_in = [cp.cuda.Stream(non_blocking=True)
                       for _ in range(nbuffers)]
            copy_out = [cp.cuda.Stream(non_blocking=True)
                        for _ in range(nbuffers)]
            _ev = lambda: cp.cuda.Event(disable_timing=True)  # noqa: E731
            ev_gather = [_ev() for _ in specs]
            ev_save = [_ev() for _ in specs]
            ev_ready = [_ev() for _ in specs]
            ev_stepped = [_ev() for _ in specs]
            ev_scatter = [_ev() for _ in specs]

        #: Which tile each buffer last served, across sweeps: the buffer
        #: reuse WAR (a gather overwrites the window the previous
        #: occupant's scatter is reading) was ordered by the single stream
        #: and needs an explicit wait on the split streams.
        occupant: list[int | None] = [None] * nbuffers

        # MEASURED HAZARD, do not remove. A non-blocking stream does NOT
        # synchronise with the legacy default stream, and everything queued
        # before this point -- the caller's store, and the H2D uploads
        # DomainState's constructor and load_base issue while building the tile
        # buffers -- was queued on the default stream. Without this barrier the
        # FIRST tile's kernels read setup arrays that have not landed yet and
        # the tile comes out entirely NaN, while every later tile is fine (by
        # then the default stream has drained). Reproduced exactly: 128x128x49
        # split in x, tile 0 all-NaN in all nine fields, tile 1 clean; a single
        # deviceSynchronize here, or non_blocking=False streams, makes the run
        # bit-exact. Blocking streams would also fix it but they serialise
        # against the null stream and destroy the copy/compute overlap this
        # pipeline exists for, so the barrier is the right cure. It runs once
        # per call, not per tile.
        cp.cuda.runtime.deviceSynchronize()

        src, dst = home, (home if other is None else other)
        gathered = scattered = saved_ring = patched_ring = 0
        geo_gathered = geo_gathers = 0
        # Which tile each buffer's LATERAL FORCING currently describes,
        # tracked separately from geo_tile because the two are refreshed by
        # different mechanisms and a future caller may pass one without the
        # other.
        hook_tile: list[int | None] = [None] * nbuffers
        hook_calls = 0

        # THE RUN LOOP'S SAFETY GATE, folded per tile out of the memory the
        # forecast is actually in.  ``integrate_prepared_case`` calls
        # ``stability_report(state, ...)`` every substep; under a host store
        # that state is never written by the sweep, so the gate observes a
        # corpse and a run that went non-finite completes "successfully".  The
        # report's quantities are max and OR folds, so they are taken per tile
        # here instead -- see ``tilestream.health_fold``.
        health = None
        if health_width is not None:
            from tilestream.health_fold import TileHealthFold

            health = TileHealthFold(cfg, len(specs),
                                    boundary_width=int(health_width))

        # THE SWEEP SEAM IS DEFERRED when nothing armed on this run needs a
        # hard barrier there.  The attribution run measured the seam --
        # every stream synchronized, then deviceSynchronize -- as 45% of the
        # exposed transfer time at its largest arm: the last tiles' scatters
        # drain against an idle GPU and the next step's first gathers fill
        # against one.  With the seam deferred, the barrier is replaced by
        # the two seam wait lists (gather-after-scatter on the store,
        # save-after-patch on the arena) and the pipeline stays primed
        # ACROSS sweep() calls; anything that reads the store must go
        # through :meth:`drain`, which the :attr:`store` property does for
        # every outside reader.  The four consumers that genuinely need the
        # barrier keep it: a CUDA-graph run defers its health readback TO
        # the sweep sync, the health fold reads back at the seam, the
        # timeline reads its events there, and ``on_sweep`` is BY CONTRACT
        # called with nothing in flight.  The clock epilogue is NOT on this
        # list: ``_advance_clock`` reads host-side counters the step's own
        # host code maintains at issue time, measured at 0.011 ms/step --
        # see OVERLAP-ATTRIBUTION.md, which exonerated it.
        defer_seam = (chained and fire_shadow is None
                      and graph_steppers is None
                      and health is None
                      and not timeline
                      and on_sweep is None)

        # ISSUE ORDER. MEASURED, and worth 1.30x on a host-resident store.
        #
        # This card reports asyncEngineCount == 1: ONE DMA queue serves every
        # stream and it is drained in submission order. Issuing ``gather(i);
        # step(i); scatter(i)`` per tile puts scatter(i) -- which cannot start
        # until step(i) finishes -- into that queue AHEAD of gather(i+1), which
        # could start at once. The copy engine then idles at the head of its
        # queue for the whole of step(i) and NOTHING overlaps. Measured at
        # 1950^2, tile 650, pinned host store, 9 tiles: naive 986.13 ms/step
        # (sum(parts)/wall = 0.92, 0% hidden) prefetch 760.85 ms/step (81% of
        # the transfer hidden) against 725.70 ms/step for the identical tiling
        # with the store in VRAM and 691.70 ms/step monolithic. Both orders
        # give the same digest.
        #
        # ``depth`` is nbuffers-1: a tile may only be prefetched into a buffer
        # whose previous occupant has already been scattered, and stream
        # ordering on that buffer's own stream enforces exactly that with no
        # extra events. With one buffer there is nowhere to prefetch and the
        # order degenerates to naive, which is why depth is clamped rather than
        # assumed positive.
        #
        # ``inplace`` is forced to naive. It exists to reproduce ONE specific
        # defect -- a tile reading the interior its predecessor already wrote
        # -- and prefetching moves a gather to before that write, so the
        # prefetch order would produce a DIFFERENT wrong answer. The gate
        # detects either, but a negative control is only worth having if it
        # reproduces the exact thing it is named after.
        depth = nbuffers - 1 if (pipeline == "prefetch"
                                 and write_mode != "inplace") else 0

        # THE RING ORDERING DISCIPLINE, which is the whole correctness argument
        # for write_mode="ring". :mod:`tilestream.rings` decides WHAT is saved;
        # these four lines decide WHEN, and getting them wrong is silent.
        #
        # gather(j) fills the window from the single store. Cells belonging to
        # a tile already scattered may be at t+dt. save(j) copies tile j's OWN
        # ring out. It is still at time t -- nothing has written tile j's
        # interior yet -- and it MUST happen before step(j), which overwrites
        # the buffer. It does not have to wait for the patch: a patch only ever
        # lands on cells owned by OTHER tiles, and interiors are disjoint.
        # patch(j) overwrites exactly the cells of already-written tiles with
        # their saved time-t values. After it the window is at time t
        # everywhere, which is the invariant the whole scheme rests on.
        # scatter(j) writes the whole interior back into the one store.
        #
        # Two hazards cross streams, and on this card both are hidden by
        # asyncEngineCount == 1 (one DMA queue, drained in submission order,
        # and the loop submits in tile order). Hiding is not the same as not
        # existing, so each gets an explicit event:
        #
        # WAR on the store. For j < k, gather(j) READS cells that scatter(k)
        # WRITES, and there is no patch in that direction -- tile k's ring is
        # not saved until tile k is gathered. So scatter(k) waits on the events
        # of the earlier tiles whose windows reach into its interior
        # (rings.RingPlan.war_deps).
        #
        # RAW on the arena. For j > k, patch(j) READS the band save(k) WROTE,
        # so patch(j) waits on those tiles' events (rings.RingPlan.patch_deps).
        #
        # Both dependency lists are computed from the plan's own rectangles and
        # hold only genuine neighbours, so this is a handful of waits per tile,
        # not a barrier. Waits are skipped when the two tiles share a buffer
        # (and therefore a stream), which already orders them.
        #
        # What is NOT a hazard, and is worth stating because it looks like one:
        # gather(j) racing scatter(k) for k < j. The gather may read either
        # generation, but the bytes it can disagree about are exactly the ones
        # patch(j) then overwrites, so the race cannot reach the answer.

        def _gather_into(itile, tspec):
            nonlocal gathered, saved_ring, patched_ring
            nonlocal geo_gathered, geo_gathers, hook_calls
            b = itile % nbuffers
            if occupant[b] != itile:
                lake = getattr(getattr(tiles[b], "physics", None), "lake", None)
                if lake is not None:
                    lake.invalidate_columns()
            # The gather's stream: the buffer's own single stream under the
            # legacy loop, the buffer's copy-in stream under overlap.  The
            # geography gather, the boundary bind and the carrier gather all
            # ride the same stream either way, so nothing below branches.
            stream = streams[b] if sched is None else copy_in[b]
            with stream:
                if sched is not None and chained:
                    # BUFFER REUSE, the WAR the single stream ordered for
                    # free: this gather overwrites the window the previous
                    # occupant's scatter is reading from the copy-out
                    # stream.
                    prev = occupant[b]
                    if prev is not None and prev != itile:
                        stream.wait_event(ev_scatter[prev])
                    if defer_seam:
                        # THE SEAM, as events instead of a device barrier:
                        # this gather must see the PREVIOUS sweep's scatters
                        # from every tile whose writes reach its window.
                        # See tilestream.overlap for why a wait that lands
                        # on this sweep's re-recording is conservative,
                        # never early.
                        for k in sched.gather_seam_waits[itile]:
                            stream.wait_event(ev_scatter[k])
                if b in noahmp_footprints:
                    noahmp_footprints[b].bind_surface_footprint(itile)
                # GEOGRAPHY FIRST, and only when this buffer changes tile. With
                # nbuffers >= len(specs) each buffer serves one tile for the
                # whole run and this fires exactly once per buffer -- the
                # "gather once at tile-buffer setup" case. Below that a buffer
                # cycles through ntiles/nbuffers tiles per sweep and the
                # geography has to follow it; there is no way round that short
                # of a resident per-tile arena, whose size scales with the
                # DOMAIN and so defeats the point. It is still strictly cheaper
                # than treating geography as a carrier, which would also
                # scatter it back every tile.
                if geo_home is not None and geo_tile[b] != itile:
                    geo_gathered += _gather.gather_tile(
                        geo_home, tiles[b], tspec, stream,
                        allow_pageable=allow_pageable, names=geography_names,
                        inventory_fn=geography_inventory, nz=nz).nbytes
                    geo_tile[b] = itile
                    geo_gathers += 1
                if tile_hook is not None and hook_tile[b] != itile:
                    # attach_lateral_boundaries resets state.elapsed_seconds
                    # to 0.0. The domain clock is re-imposed by
                    # set_carrier_scalars before the step, so the run is
                    # correct either way -- but preserving it here keeps a
                    # run with scalars=None from being silently REPAIRED by
                    # the hook, which would disarm the clock control.
                    keep = getattr(tiles[b], "elapsed_seconds", None)
                    tile_hook(tiles[b], tspec, itile, stream)
                    if keep is not None:
                        tiles[b].elapsed_seconds = keep
                    hook_tile[b] = itile
                    hook_calls += 1
                gathered += _gather.gather_tile(
                    src, tiles[b], tspec, stream,
                    allow_pageable=allow_pageable, names=names,
                    inventory_fn=inventory_fn, nz=nz, global_keys=global_keys).nbytes
                from tilestream.sfire import configure_tile
                configure_tile(tiles[b], tspec)
                if ring is not None and sched is None:
                    # Taken FRESH, after whatever step this buffer last served.
                    # PhysicsDriver replaces whole tendency bundles when their
                    # scheme runs, so an inventory cached at setup names arrays
                    # the model has stopped using -- see rings._BlockList.
                    tinv = take(tiles[b], names)
                    saved_ring += ring.save(itile, tinv, stream)
                    if ring_events is not None:
                        ring_events[itile].record(stream)
                        for k in ring.plan.patch_deps[itile]:
                            if k % nbuffers != b:
                                stream.wait_event(ring_events[k])
                    patched_ring += ring.patch(itile, tinv, stream)
            if sched is not None:
                # THE SPLIT of the single ring point.  The save moves to the
                # copy-out stream (it is D2H against a host store, and it is
                # the D2H engine's queue), ordered after the gather that
                # fills the band it reads; the patch stays on copy-in
                # (H2D), ordered after every save it reads from.  The tile
                # inventory is taken fresh at issue time, exactly as the
                # single-stream loop takes it -- see rings._BlockList.
                if chained:
                    ev_gather[itile].record(stream)
                if ring is not None:
                    tinv = take(tiles[b], names)
                    cout = copy_out[b]
                    with cout:
                        if chained:
                            cout.wait_event(ev_gather[itile])
                            if defer_seam:
                                # Arena WAR across the seam: this save
                                # refills bands the previous sweep's patches
                                # read.
                                for i in sched.save_seam_waits[itile]:
                                    cout.wait_event(ev_ready[i])
                        saved_ring += ring.save(itile, tinv, cout)
                        if chained:
                            ev_save[itile].record(cout)
                    with stream:
                        if chained:
                            # RAW on the arena, no same-buffer skip: save
                            # and patch sit on different streams even when
                            # the tiles share a buffer.
                            for k in sched.patch_waits[itile]:
                                stream.wait_event(ev_save[k])
                        patched_ring += ring.patch(itile, tinv, stream)
                with stream:
                    if chained:
                        ev_ready[itile].record(stream)
                occupant[b] = itile

        # THE DOMAIN CLOCK. ``dycore.step`` advances state.elapsed_seconds by
        # dt once per CALL (dycore.py:2474) and PhysicsDriver.compute turns it
        # into itimestep = floor(elapsed/dt + 0.5) + 1, which is the argument
        # to every cadence test (_radiation_step_due / _surface_pbl_step_due /
        # _cumulus_step_due, physics.py:3399-3410). A buffer serving k tiles
        # per sweep would therefore advance k*dt while the DOMAIN advances dt,
        # and tiles within one sweep would disagree about which schemes are
        # due. So the buffer's clock is reset to the domain's before every tile
        # step and the domain's advances exactly once per sweep.
        clock = None if scalars is None else dict(scalars)
        _physics = None
        if clock is not None:
            from tilestream import physics_inventory as _physics

        sweep_sequence = 0

        def _sweep(nsteps, step_kwargs, report, progress, physics_control, step_function=None):
            nonlocal sweep_sequence
            # The byte counters are nonlocal AND reset here: they live in
            # __init__ because _gather_into writes them, and they are
            # per-call because that is what a caller timing one model step
            # needs. Rebinding them locally instead would silently report
            # zero bytes gathered while the gathers really happened.
            nonlocal src, dst, clock
            nonlocal gathered, scattered, saved_ring, patched_ring
            nonlocal compute_done
            gathered = scattered = saved_ring = patched_ring = 0
            health_report = None
            # RE-READ the caller's scalars, do not merely write them back.
            # ``clock`` is this run's private copy and the sweep's epilogue
            # publishes it into ``scalars``; without this line the traffic is
            # one-way and a caller that ADJUSTS the clock between sweeps is
            # silently ignored.  That caller exists:
            # ``woof.core.model.execute_experiment`` refreshes every domain's
            # clock from integer ticks before each STEP -- the clock module is
            # the calendar authority, not the stepper -- and a streamed domain
            # that ignored it ran a second free-running clock.  MEASURED with a
            # one-step warmup before attach: the streamed parent's tiles saw
            # elapsed 30..900 while the model was at 0..870, so radiation fired
            # 8 times against the reference's 9 and cumulus 16 against 17, and
            # the two runs integrated different physics with no NaN and no
            # refusal.  A caller that never touches the dict sees no change at
            # all: it publishes what it last read.
            if clock is not None and scalars is not None:
                clock.clear()
                clock.update(scalars)
            nsteps = int(nsteps)
            step_kwargs = ({} if step_kwargs is None
                           else dict(step_kwargs))
            # PER SWEEP, not cumulative over the run: each entry is one
            # sweep's wall time measured between the full device
            # synchronizations that bracket it, so a caller can see the
            # first sweep's warmup separately from the steady state instead
            # of having it averaged away.
            sweep_seconds: list[float] = []
            sweep_clocks: list[dict] = []
            stochastic = getattr(self, "_ensemble_stochastic_lease", None)
            # The domain's IAU (woof.da.iau.IauSweepLease): each window's
            # step folds the forcing cut to it, and a door's post-step acts
            # reach the window before its scatter.  A captured graph has no
            # place for a per-window forcing, so it steps ordinarily.
            iau = getattr(self, "_iau_sweep_lease", None)
            execution_graphs = (None if stochastic is not None or iau is not None
                                else graph_steppers)
            if stochastic is not None and graph_steppers is not None:
                self._ensemble_stochastic_graph_fallback = "ordinary window step preserves the live full-domain stochastic pattern"
            for istep in range(nsteps):
                fire_records = []
                if fire_shadow is not None:
                    _copy_into(fire_shadow, home)
                    src = {**home, **fire_shadow}
                sweep_sequence += 1
                _t_sweep = _time.perf_counter()
                if stochastic is not None:
                    stochastic.begin(cfg, windows=len(specs))
                if iau is not None:
                    iau.begin(windows=len(specs))
                begin_wrf_cfl_domain_step(cfg)
                if execution_graphs is not None:
                    # Under graph_reuse="sweep" the sweep index IS the cache
                    # key's clock component: every tile of one sweep steps
                    # from the same domain clock and so issues the same
                    # sequence, and no two sweeps share a graph.  MEASURED
                    # what dropping this line costs, because the port of this
                    # feature onto TiledRun did drop it: at the "full fast
                    # cadence" rung (radiation EVERY step) the cache collapsed
                    # from 6 captures to 4 and the run stopped being bit-exact
                    # -- maxabs 1.998e+04 on driver/radiation_tendencies/rv --
                    # while `graph_ok` stayed True and every other graph case
                    # still passed.  A replayed radiation step re-uses the
                    # earlier step's sun; that is the whole reason the default
                    # keys on the sweep.
                    #
                    # Monotonic across calls: repeated sweep(1) must not
                    # reuse a graph containing the preceding absolute time.
                    for gstepper in execution_graphs:
                        gstepper.set_sweep(sweep_sequence)
                if health is not None and health.enabled:
                    health.begin()
                # Prime the pipeline.  Never across a sweep boundary: the
                # buffers
                # swap there, so a tile gathered early would read the wrong
                # generation of the domain.
                for i in range(min(depth, len(specs))):
                    _gather_into(i, specs[i])
                last_b = 0
                for itile, tspec in enumerate(specs):
                    if depth:
                        nxt = itile + depth
                        if nxt < len(specs):
                            _gather_into(nxt, specs[nxt])
                    else:
                        _gather_into(itile, tspec)
                    b = itile % nbuffers
                    last_b = b
                    stream = streams[b]
                    if clock is not None:
                        _physics.set_carrier_scalars(tiles[b], clock)
                    if physics_control is not None:
                        physics_control.apply(tiles[b])
                    with stream:
                        if sched is not None and chained:
                            # The step reads the window the copy-in stream
                            # filled and patched, and overwrites the band
                            # the copy-out stream is saving.
                            stream.wait_event(ev_ready[itile])
                            stream.wait_event(ev_save[itile])
                        if chain and compute_done is not None:
                            stream.wait_event(compute_done)
                        if stochastic is not None:
                            stochastic.bind_window(tiles[b], tile_cfg, tspec, itile, stream=stream)
                        if iau is not None:
                            iau.bind_window(tiles[b], tspec, itile)
                        elif "_iau_forcing" in vars(tiles[b]):
                            # a released lease's last window forcing
                            del tiles[b]._iau_forcing
                        if cfg.diag_print:
                            from woof.core.misc_diag_print import bind_window
                            bind_window(tiles[b], tspec, cfg.nx)
                        set_wrf_cfl_tile_window(cfg.grid_id, tspec)
                        if self.pre_step_hook is not None:
                            self.pre_step_hook(tiles[b], tspec, itile, stream)
                        if timeline:
                            began = cp.cuda.Event()
                            began.record(stream)
                        # ``**step_kwargs``, and it was missing.  ``_sweep``
                        # copied the caller's mapping into a local dict at
                        # the top and then never used it, so every keyword
                        # ArWen's own loop threads into a STEP was accepted
                        # by ``sweep``, described by its docstring as
                        # "forwarded verbatim to every tile's dycore.step",
                        # and dropped one hop short of the call.
                        # ``tests/test_streaming.py`` asserted the kwargs
                        # reach ``sweep`` against a fake run, which is
                        # exactly one hop before the gap.
                        #
                        # Essential rather than tidy:
                        # :class:`woof.core.streaming.StreamedDomain` hands
                        # ArWen's own per-step keywords straight into
                        # ``sweep``, so dropping them here silently unsets
                        # every one of them for a streamed domain.  Two
                        # measured consequences, both silent:
                        #
                        # * ``refl_10cm_due`` never reaches a tile, so no
                        #   tile ever runs WRF's calc_refl10cm and the
                        #   ``refl_10cm`` scratch slot is never allocated on
                        #   any buffer.  At 96x72x49, tile 24x24, through
                        #   execute_experiment with a history handler
                        #   attached, the resident run staged REFL_10CM for
                        #   1 of its 2 frames and the streamed run for 0 of
                        #   2 -- no error, no NaN, a missing output field.
                        #   The only louder symptom was the CONSUMER on the
                        #   domain state raising "REFL_10CM output is due
                        #   but no microphysics-time field is stashed" one
                        #   layer up, at a call site that looked like the
                        #   culprit and was not.
                        #   ``tilestream.test_history
                        #   .negative_dropped_step_kwargs`` restores the
                        #   drop and must fail.
                        # * ``mass_flux_observer`` /
                        #   ``mass_flux_accumulator`` -- the conservation
                        #   receipt's lateral-flux term -- are never called
                        #   at all.  At 256x192 open-boundary, 4 steps, 16
                        #   tiles, the monolithic run reported 16 increments
                        #   summing to -1.1625e+04 and the tiled run
                        #   reported ZERO increments and an integral of 0.0,
                        #   which reads exactly like a perfectly closed
                        #   budget.
                        #
                        # ``{}`` for a caller that passes none, which is
                        # still every gate.  NOT empty for the streamed
                        # product lanes since audit R-052: ``run_tiled``
                        # forwards a keyword set now, and
                        # ``reflectivity_run_kwargs`` puts ``refl_10cm_due``
                        # in it so every tile writes the ``refl_10cm`` slot
                        # its store carries.
                        if step_function is not None:
                            step_function(tiles[b], tile_cfg, **step_kwargs)
                        elif execution_graphs is None:
                            step(tiles[b], tile_cfg, **step_kwargs)
                        else:
                            # REFUSED rather than dropped.  ``GraphStepper
                            # .run`` captures ``dycore.step(state, cfg)`` and
                            # has nowhere to put a per-step keyword, so a
                            # streamed forecast that asked for
                            # ``refl_10cm_due`` under graph capture would get
                            # a frame with the field silently absent -- which
                            # is the exact defect feat-route-wire and
                            # feat-wrfout-stream each measured and fixed on
                            # the ordinary path.  The graph gate itself
                            # passes no kwargs; a lane that asked for
                            # reflectivity under graph capture is refused
                            # here by name rather than losing the field.
                            if step_kwargs:
                                raise TiledRunError(
                                    "use_graph is on and this sweep carries "
                                    f"step_kwargs {sorted(step_kwargs)}; a "
                                    "captured graph has no way to receive "
                                    "them, so they would be silently "
                                    "dropped for every tile.  Run this sweep "
                                    "without graph capture, or capture a "
                                    "step that takes them.")
                            execution_graphs[b].run(tiles[b], stream)
                        finish_wrf_cfl_tile(cfg.grid_id)
                        fire = getattr(getattr(tiles[b], "physics", None), "fire", None)
                        if fire is not None:
                            from tilestream.sfire import GRID_DIAGNOSTICS
                            fire_records.append({key: getattr(fire.grid, key) for key in GRID_DIAGNOSTICS})
                        if timeline:
                            ended = cp.cuda.Event()
                            ended.record(stream)
                            marks.append((began, ended))
                        if chain:
                            compute_done = cp.cuda.Event(disable_timing=True)
                            compute_done.record(stream)
                        # Inside the stream, after the step, before the
                        # scatter: the only instant a buffer holds current
                        # domain data on the card.  Issued, never
                        # synchronised -- an observer that read back here
                        # would serialise the pipeline the prefetch order
                        # exists to build (measured 1.30x).
                        #
                        # Read off SELF rather than closed over, because the
                        # observer that needs this hook -- the folded
                        # stability record -- has to be constructed from the
                        # tile plan this constructor is still building, so it
                        # can only be attached once the TiledRun exists.
                        if self.observer is not None:
                            self.observer(tiles[b], tspec, itile, stream)
                        if health is not None and health.enabled:
                            # The SECOND fold, from defect2-observer-fold, and
                            # off unless a caller passed ``health_width``.  It
                            # answers the same question as ``observer`` above
                            # by a different route; both are kept because both
                            # carry their own gate, and only one is armed at a
                            # time so a run never pays for two reductions.
                            # Already inside ``with stream``, so this lands on
                            # the buffer's own stream; the tile's INTERIOR is
                            # exactly what the scatter below will write.
                            health.tile(tiles[b], tspec, itile)
                        if iau is not None:
                            # After the readers above, as the resident door
                            # acts after the step's health checks: the
                            # window's share of a post-step increment, ahead
                            # of the scatter (ev_stepped below orders it).
                            iau.after_window(tiles[b], tspec, itile, stream=stream)
                        if self.post_step_hook is not None:
                            # The WRITER at the same instant: it clears the
                            # tile's one-frame REFL_10CM handoff, after the
                            # readers above have folded the buffer.
                            self.post_step_hook(tiles[b], tspec, itile, stream)
                        if sched is not None and chained:
                            # Recorded BEHIND the observers and hooks, so
                            # anything they issued into this stream is
                            # ordered ahead of the scatter too.
                            ev_stepped[itile].record(stream)
                        if ring_events is not None:
                            for j in ring.plan.war_deps[itile]:
                                if j % nbuffers != b:
                                    stream.wait_event(ring_events[j])
                        if sched is None:
                            splan = _gather.scatter_tile(
                                tiles[b], dst, tspec, stream,
                                allow_pageable=allow_pageable, names=names,
                                inventory_fn=inventory_fn, nz=nz, global_keys=global_keys)
                    if sched is not None:
                        # The scatter rides the copy-out stream: after the
                        # step that wrote the interior, and after the READS
                        # of every earlier tile whose window reaches into
                        # it (the WAR the legacy loop ordered with
                        # ring_events).
                        cout = copy_out[b]
                        with cout:
                            if chained:
                                cout.wait_event(ev_stepped[itile])
                                for j in sched.scatter_waits[itile]:
                                    cout.wait_event(ev_gather[j])
                            splan = _gather.scatter_tile(
                                tiles[b], dst, tspec, cout,
                                allow_pageable=allow_pageable, names=names,
                                inventory_fn=inventory_fn, nz=nz, global_keys=global_keys)
                            if chained:
                                ev_scatter[itile].record(cout)
                    scattered += splan.nbytes
                    if progress is not None:
                        progress(istep, itile, tspec)
                finish_wrf_cfl_domain_step(cfg.grid_id)
                if stochastic is not None:
                    for stream in streams:
                        stream.synchronize()
                    stochastic.finish()
                if iau is not None:
                    # host bookkeeping only: the step is booked once, after
                    # every window issued it
                    iau.finish()
                if defer_seam:
                    # THE DEFERRED SEAM: nothing here waits.  The last
                    # tiles' scatters drain UNDER the next step's gathers
                    # and first computes, which chain on the seam events
                    # instead of on a device barrier -- the single biggest
                    # exposed term the attribution measured.  Everything
                    # host-side below (the clock epilogue, the report) reads
                    # state the step's own host code maintained at issue
                    # time; everything that reads the STORE goes through
                    # drain().
                    self._pending = True
                else:
                    for stream in streams:
                        stream.synchronize()
                    if copy_in is not None:
                        for stream in copy_in:
                            stream.synchronize()
                        for stream in copy_out:
                            stream.synchronize()
                    cp.cuda.runtime.deviceSynchronize()
                if execution_graphs is not None:
                    # The sweep's synchronisation is the point the deferred
                    # health readback was deferred TO, and it is already paid
                    # for.  Drain every sweep, not at the end of the run: a
                    # ledger that is recorded into and drained late turns a
                    # non-finite field into a much longer wrong forecast, and
                    # one that is never drained turns it into no error at all.
                    for gstepper in execution_graphs:
                        gstepper.drain()
                if health is not None and health.enabled:
                    health_report = health.finish()
                if clock is not None:
                    # Only buffers that actually served a tile this sweep
                    # hold a current clock; with nbuffers > ntiles the rest
                    # are still on the previous sweep's and would trip the
                    # agreement check.
                    used = min(nbuffers, len(specs))
                    clock = _advance_clock(clock, tiles[:used], last_b,
                                           _physics, fire_records=fire_records)
                if on_sweep is not None:
                    # THE SWEEP SEAM.  Called after every tile of sweep
                    # ``istep`` has been scattered and synchronized and after
                    # the domain clock has advanced, so ``dst`` is the
                    # complete new generation of the whole domain and nothing
                    # is in flight.  It is where a caller acts on the domain
                    # AS A DOMAIN -- a lateral boundary condition, a nudge, an
                    # output write -- from inside a multi-step sweep.  (A
                    # caller that drives one model step per call already has
                    # that seam for free: it is the return of ``sweep``.  This
                    # is for callers driving multi-step sweeps.)
                    # Before the shadow swap, so
                    # ``dst`` is the generation just written in every
                    # write_mode.
                    on_sweep(istep, dst, clock)
                # After the sweep's device synchronization and the clock
                # epilogue, so this is the whole sweep and nothing is still in
                # flight; before the shadow swap, which is bookkeeping.
                sweep_seconds.append(_time.perf_counter() - _t_sweep)
                sweep_clocks.append({} if clock is None else dict(clock))
                if other is not None:
                    src, dst = dst, src

            if other is not None and src is not home:
                # The newest data ended up in the shadow; put it back where the
                # caller's store lives so run_tiled's in-place contract holds.
                _copy_into(home, src)

            if clock is not None and scalars is not None:
                scalars.clear()
                scalars.update(clock)

            overlaps = None
            if timeline and marks:
                cp.cuda.runtime.deviceSynchronize()
                spans = [(cp.cuda.get_elapsed_time(origin, began),
                          cp.cuda.get_elapsed_time(origin, ended))
                         for began, ended in marks]
                # Two tiles overlap when the later one STARTED before the
                # earlier one ENDED.  Compared against the running maximum
                # end time rather than only the immediate predecessor, so a
                # three-deep overlap counts as two rather than one.  The
                # tolerance is the event resolution, about half a
                # microsecond; anything at or below it is not evidence.
                overlaps = 0
                highest_end = float("-inf")
                for began, ended in spans:
                    if began < highest_end - 5e-4:
                        overlaps += 1
                    highest_end = max(highest_end, ended)

            if report is not None:
                if overlaps is not None:
                    report["overlapping_steps"] = overlaps
                report["chain_compute"] = bool(chain)
                report["shared_workspaces"] = shared is not None
                if health_report is not None:
                    # The LAST step's report.  ArWen's run loop sweeps one
                    # model step per call, so last == only; a caller that
                    # sweeps several at once is explicitly not gating each of
                    # them and must not be told that it is.
                    report["health"] = health_report
                report.update(
                    tiles=len(specs), steps=nsteps, nbuffers=nbuffers,
                    halo=int(halo), write_mode=write_mode, pipeline=pipeline,
                    # The EFFECTIVE overlap mode ('on' degrades to 'off' for
                    # a non-ring write mode) and whether the sweep seam was
                    # deferred; with the seam deferred, ``sweep_seconds`` is
                    # ISSUE time and the accurate wall clock of a window is
                    # measured across drain().
                    overlap=overlap, overlap_deferred_seam=defer_seam,
                    tile_cfg=tile_cfg,
                    domain=(nz, ny, nx), compute=(tile_cfg.nz, cny, cnx),
                    gathered_bytes=gathered, scattered_bytes=scattered,
                    efficiency=_spec.plan_efficiency(specs),
                    fields=len(home), scalars=clock,
                    specs=specs,
                    tile_hook_calls=hook_calls,
                    geography_fields=geo_fields,
                    geography_gathers=geo_gathers,
                    geography_bytes=geo_gathered,
                    geography_over_carrier=(geo_gathered / gathered
                                            if gathered else 0.0),
                    # The one-off construction cost, carried on every report
                    # so a caller cannot accidentally fold it into a per-step
                    # number.  ``setup_seconds`` is the whole of __init__ and
                    # ``factory_seconds`` the tile-buffer build inside it;
                    # both are properties of THIS TiledRun, not of the sweep,
                    # and a run that sweeps many times amortises them.
                    setup_seconds=self.setup_seconds,
                    factory_seconds=self.factory_seconds,
                    loop_seconds=sum(sweep_seconds),
                    sweep_seconds=list(sweep_seconds),
                    sweep_clocks=list(sweep_clocks),
                )
                if stochastic is not None:
                    report["ensemble_stochastic"] = stochastic.receipt()
                    report["stochastic_graph_fallback"] = getattr(self, "_ensemble_stochastic_graph_fallback", None)
                if ring is not None:
                    store_bytes = sum(int(a.nbytes) for a in home.values())
                    report.update(
                        ring_bytes=ring.nbytes,
                        ring_over_store=ring.nbytes / max(1, store_bytes),
                        ring_saved_bytes=saved_ring,
                        ring_patched_bytes=patched_ring,
                        ring_bands=len(ring.plan.bands),
                        ring_patch_blocks=sum(
                            len(p) for p in ring.plan.patches),
                        ring_report=_rings.ring_report(ring.plan),
                    )
                if execution_graphs is not None:
                    per = [g.report() for g in execution_graphs]
                    report.update(
                        graph=dict(
                            reuse=graph_reuse, key=graph_key,
                            captures=sum(q["captures"] for q in per),
                            replays=sum(q["replays"] for q in per),
                            fallbacks=sum(q["fallbacks"] for q in per),
                            capture_seconds=sum(q["capture_seconds"]
                                                for q in per),
                            nodes=sorted({n for q in per for n in q["nodes"]}),
                            health_records=sum(q["health_records"]
                                               for q in per),
                            reason=next((q["reason"] for q in per
                                         if q["reason"]), None)))
                if ring is not None or other is None:
                    report.update(second_store_bytes=0)
                else:
                    report.update(second_store_bytes=sum(
                        int(a.nbytes) for a in other.values()))

        def _reseed_clock(new):
            # The sweep's ``clock`` is READ ONCE, here in __init__, and only
            # written back into ``scalars`` at the end of a sweep.  That is
            # right for stepping -- re-reading a caller-owned dict every
            # sweep would let a stray mutation reset the cadence -- and it
            # is wrong for exactly one operation: a RESTART RESTORE, which
            # replaces the domain's carriers AND its clock between sweeps.
            # Without this, a resumed streamed run would carry on with the
            # PRE-RESTORE elapsed_seconds and call counts: the store would
            # hold the checkpoint's atmosphere and the driver would think it
            # was at a different itimestep, so radiation, cumulus and the
            # PBL would fire on the wrong steps and dtbc would interpolate
            # the wrong point of the forcing interval.  No NaN, no warning,
            # a different forecast.  So the reseed is explicit and named,
            # rather than the sweep quietly re-reading the dict.
            nonlocal clock
            if clock is None:
                raise TiledRunError(
                    "this TiledRun carries no scalars, so it has no domain "
                    "clock to reseed; attach with scalars= before restoring "
                    "a restart into its store")
            clock = dict(new)
            if scalars is not None:
                scalars.clear()
                scalars.update(clock)

        def _set_live_config(incoming):
            nonlocal cfg, tile_cfg
            from dataclasses import fields, replace
            from woof.core.adaptive_clock import ADAPTIVE_DERIVED_RUN_FIELDS

            changed = {field.name for field in fields(cfg)
                       if getattr(cfg, field.name) != getattr(incoming, field.name)}
            structural = changed - ADAPTIVE_DERIVED_RUN_FIELDS
            if structural:
                raise TiledRunError(
                    "a prepared streamed domain cannot change "
                    f"{sorted(structural)} between steps; rebuild its state and tiles")
            required = _harness.halo_radius(incoming)
            if required > int(halo):
                raise TiledRunError(
                    f"live time_step_sound={incoming.time_step_sound} needs "
                    f"halo {required}, but these tile buffers allocated halo "
                    f"{halo}; construct the plan with the adaptive acoustic "
                    "ceiling before allocating its boundary windows")
            cfg = incoming
            tile_cfg = replace(tile_cfg, **{
                name: getattr(cfg, name) for name in changed})
            self.cfg, self.tile_cfg = cfg, tile_cfg
            for graph in graph_steppers or ():
                graph.set_config(tile_cfg)

        self._set_live_config = _set_live_config
        self._sweep = _sweep
        self._reseed_clock = _reseed_clock
        self.cfg = cfg
        self.tile_cfg = tile_cfg
        self.specs = specs
        self.tiles = tiles
        self.nbuffers = int(nbuffers)
        self.halo = int(halo)
        self._home = home
        #: The effective overlap mode and whether the sweep seam is
        #: deferred; the schedule itself, for the contract tests.
        self.overlap = overlap
        self.deferred_seam = bool(defer_seam)
        self.schedule = sched
        self._streams = streams
        self._copy_in = copy_in
        self._copy_out = copy_out
        #: True while a deferred-seam sweep may still be in flight on the
        #: device.  Reading the store without draining first reads a
        #: GENERATION IN PROGRESS; :attr:`store` and :meth:`drain` are the
        #: two doors, and both leave this False.
        self._pending = False
        self.geography_fields = geo_fields
        self._closed = False
        self.scalars = scalars
        self.observer = observer
        self.post_step_hook = post_step_hook
        self.pre_step_hook = None
        self.nz = int(nz)
        # Recorded PER AXIS, because since feat-open-lateral-bc the two can
        # differ and a single ``periodic`` attribute would answer the wrong
        # question for an ``open_x``-only domain.  ``periodic`` is kept as
        # the conjunction, which is what its only readers (autoplan's
        # receipt lines) mean by it.
        self.periodic_x = bool(periodic if periodic_x is None else periodic_x)
        self.periodic_y = bool(periodic if periodic_y is None else periodic_y)
        self.periodic = bool(self.periodic_x and self.periodic_y)
        #: The per-tile safety fold, or None when this run is not gated.
        #: Exposed so the negative controls can break it deliberately.
        self.health = health
        #: The per-buffer CUDA-graph steppers, or None when the step is
        #: launched into the stream the ordinary way.
        self.graph_steppers = graph_steppers
        #: Wall seconds spent building the tile buffers alone (a physics
        #: state per buffer at the physics rungs, which is the expensive
        #: part), and the whole of construction.  Measured, not estimated:
        #: the buffer build is followed by a full device synchronization so
        #: an asynchronous allocation cannot be billed to the first sweep.
        self.factory_seconds = _t_factory - _t_call
        self.setup_seconds = _time.perf_counter() - _t_call

    @property
    def store(self):
        """``{name: array}`` of the whole domain, DRAINED before it is read.

        Under the deferred seam a sweep returns with its scatters still in
        flight, so the store an outside reader would see is a generation in
        progress.  Every read through this property drains first; the sweep
        loop itself holds the raw mapping and never pays for it.  Costs one
        flag test when nothing is pending.
        """
        self._require_open()
        self.drain()
        return self._home

    def _require_open(self) -> None:
        if getattr(self, "_closed", False):
            raise TiledRunError("this tiled run is closed; rebuild its owner before reuse")

    @property
    def closed(self) -> bool:
        return bool(getattr(self, "_closed", False))

    def close(self) -> None:
        """Drain and relinquish this run's allocation owners, exactly once.

        This is a lifecycle operation, not a relocation implementation. A
        caller must preserve its canonical carriers/scalars before closing.
        No borrowed array is mutated and no process-wide memory pool is freed.
        In particular, the sweep/config/clock closures retain the tile factory,
        ring arena, packed boundary windows and geography even when ``tiles``
        alone is dropped; all three closures must be released together.

        A drain failure leaves the owner open and intact. Once drained, this
        object cannot read or advance a stale store. External borrowers retain
        their own references; dropping this owner does not invalidate them.
        """
        if self.closed:
            return
        # A failed sweep can leave transfers in flight before reaching the
        # deferred-seam flag write. Closing is a terminal boundary: wait on
        # every owned stream even when that ordinary fast-path flag is clear.
        self._pending = True
        self.drain()
        for name in ("_sweep", "_set_live_config", "_reseed_clock", "tiles",
                     "_home", "_streams", "_copy_in", "_copy_out",
                     "graph_steppers", "health", "observer"):
            setattr(self, name, None)
        self._closed = True

    def drain(self) -> None:
        """Wait until nothing this run issued is still in flight.

        The deferred seam's counterpart: it replaces the per-sweep device
        barrier for the callers that actually need one -- a store read, a
        history frame, a restart capture, a timing window's edge.  Idempotent
        and cheap when nothing is pending.
        """
        if self.closed or not self._pending:
            return
        import cupy as cp

        with cp.cuda.Device(self._device_id):
            for group in (self._streams, self._copy_in, self._copy_out):
                if group is not None:
                    for stream in group:
                        stream.synchronize()
            cp.cuda.runtime.deviceSynchronize()
        self._pending = False

    def sync_compute(self) -> None:
        """Wait for the COMPUTE streams only, leaving copies in flight.

        For the per-step consumers that need every tile's kernels finished
        but not its transfers landed -- the folded stability record above
        all: its per-tile partials are written by kernels on the compute
        streams, and waiting for the scatter tail as well would re-expose
        exactly the drain the deferred seam hides.
        """
        self._require_open()
        if not self._pending:
            return
        import cupy as cp

        with cp.cuda.Device(self._device_id):
            for stream in self._streams:
                stream.synchronize()

    def reseed_clock(self, scalars) -> None:
        """Replace the sweep's cached domain clock (a restart restore did).

        ``sweep`` caches the clock in a closure because ``dycore.step``
        advances ``elapsed_seconds`` once per CALL and the domain must
        advance it once per SWEEP; a restore is the one event that legally
        moves the domain clock from outside the sweep, and it has to say so.
        """
        self._require_open()
        self._reseed_clock(scalars)

    def sweep(self, nsteps: int = 1, *, step_kwargs=None,
              report: dict | None = None, progress=None,
              live_config=None, physics_control=None, step_function=None) -> None:
        """Advance the store by ``nsteps`` model steps, in place.

        ``step_kwargs`` is forwarded verbatim to every tile's
        ``dycore.step``.  WOOF's own loop passes ``refl_10cm_due``
        through it, and it has to reach EVERY tile of a sweep: it
        decides whether the step stages REFL_10CM for the frame
        that is about to be written, and a tile that staged it
        while its neighbour did not would write a frame that is
        half a forecast.

        ``report`` mixes two kinds of number, deliberately.  The byte
        counters are PER CALL, because that is what a caller timing one
        model step wants.  ``geography_gathers`` and
        ``tile_hook_calls`` are CUMULATIVE over the run's life, because
        what they measure is how often a buffer had to change tile --
        the quantity the whole "geography is gathered once per buffer"
        argument is about, and one that means nothing sampled inside a
        single sweep.

        Two keywords are REFUSED rather than forwarded, and refusing them is
        the whole point of naming them: ``mass_flux_observer`` and
        ``mass_flux_accumulator`` sample the conservation receipt's
        telescoped lateral flux from the state's OUTERMOST FACES, which for a
        tile are its compute window's faces, not the domain's.
        ``dycore._boundary_x`` is a CONFIG property, so every tile believes
        all four of its sides are domain edges; and under ``periodic=False``
        neighbouring tiles' clamped windows overlap along a shared domain
        edge by ``2*halo``, so even the sides that ARE domain edges are
        counted more than once.  The result is a plausible number that is
        not the domain's flux -- the worst possible outcome for a receipt --
        so it is refused here and the reason is in
        :mod:`tilestream.receipts`.
        """
        self._require_open()
        for name in ("mass_flux_observer", "mass_flux_accumulator"):
            if (step_kwargs or {}).get(name) is not None:
                raise TiledRunError(
                    f"{name} is a DOMAIN-scope observer and a sweep can "
                    "only offer it TILE windows: every tile reads its own "
                    "compute-window faces as if they were domain edges, and "
                    "clamped windows overlap along a shared domain edge by "
                    "2*halo.  See tilestream.receipts for what a streamed "
                    "domain can and cannot report.")
        import cupy as cp

        with cp.cuda.Device(self._device_id):
            if live_config is not None and live_config != self.cfg:
                # A deferred sweep can still read old kernel arguments and
                # graph workspaces. Finish it before rebinding.
                self.drain()
                self._set_live_config(live_config)
            try:
                if step_function is None:
                    self._sweep(nsteps, step_kwargs, report, progress, physics_control)
                else:
                    self._sweep(nsteps, step_kwargs, report, progress, physics_control,
                                step_function=step_function)
            finally:
                from woof.core.dycore import finish_wrf_cfl_domain_step
                finish_wrf_cfl_domain_step(self.cfg.grid_id, commit=False)


def run_tiled(store, cfg, tile_nx, tile_ny, halo: int = 16,
              nsteps: int = 1,
              nbuffers: int = 2, *, periodic: bool = True,
              periodic_x: bool | None = None,
              periodic_y: bool | None = None,
              write_mode: str = "ring", shadow=None,
              tile_state_factory=None, tile_states=None, names=None,
              allow_pageable: bool = False, poison: bool = True,
              pipeline: str = "prefetch", inventory_fn=None, nz=None,
              scalars=None, geography=None, geography_names=None,
              check_geography: bool = True, tile_hook=None, observer=None,
              post_step_hook=None,
              impose_geography_flags: bool = True,
              ring_margin: str = "exact",
              ring_ordering: str = "events",
              overlap: str = "on",
              health_width: int | None = None,
              shared=None, chain_compute=None, timeline: bool = False,
              use_graph=False, graph_reuse: str = "sweep",
              graph_key: str = "cadence", graph_scalars: bool = True,
              graph_verify_host: bool = True,
              graph_verify_topology: bool = False,
              report: dict | None = None, progress=None,
              on_sweep=None, step_kwargs=None) -> None:
    """Build a :class:`TiledRun` and sweep it ``nsteps`` times.

    The whole-run entry point, and the one the gate drives.  Every
    argument is :class:`TiledRun`'s; see its docstring.  A caller
    that runs more than a handful of steps, or that needs its own
    loop between steps, holds a ``TiledRun`` and calls ``sweep``:
    this function rebuilds the tile buffers on every call.

    ``tile_states=`` is the escape hatch for the callers that cannot hold a
    ``TiledRun`` -- a benchmark timing one construction per rep -- and lends
    pre-built buffers instead.  Holding a ``TiledRun`` remains the better
    answer where it is possible, because it reuses the transfer plans and the
    ring arena too, not only the buffers.

    ``step_kwargs=`` is :meth:`TiledRun.sweep`'s, forwarded verbatim to every
    tile's ``dycore.step``.  It was reachable ONLY by holding a ``TiledRun``
    until audit R-052, and the one keyword the product lanes need is
    ``refl_10cm_due``: without it no tile runs calc_refl10cm, the
    ``refl_10cm`` scratch slot is never written on any buffer, and a store
    that lists the slot has every tile scatter an unwritten window over it.
    That is the half of R-052 the store-side inventory alone did not reach.
    ``None`` is the identical call every existing caller makes.
    """
    run = TiledRun(
        store, cfg, tile_nx, tile_ny, halo, nbuffers,
        periodic=periodic, periodic_x=periodic_x, periodic_y=periodic_y,
        write_mode=write_mode, shadow=shadow,
        tile_state_factory=tile_state_factory, tile_states=tile_states,
        names=names,
        allow_pageable=allow_pageable, poison=poison,
        pipeline=pipeline, inventory_fn=inventory_fn, nz=nz,
        scalars=scalars, geography=geography,
        geography_names=geography_names,
        check_geography=check_geography, tile_hook=tile_hook,
        observer=observer, post_step_hook=post_step_hook,
        impose_geography_flags=impose_geography_flags,
        ring_margin=ring_margin, ring_ordering=ring_ordering,
        overlap=overlap,
        health_width=health_width,
        shared=shared, chain_compute=chain_compute, timeline=timeline,
        use_graph=use_graph, graph_reuse=graph_reuse, graph_key=graph_key,
        graph_scalars=graph_scalars,
        graph_verify_host=graph_verify_host,
        graph_verify_topology=graph_verify_topology,
        on_sweep=on_sweep,
    )
    try:
        run.sweep(nsteps, step_kwargs=step_kwargs, report=report,
                  progress=progress)
    except BaseException as error:
        try:
            run.close()
        except BaseException as cleanup_error:
            error.add_note(
                f"tiled-run cleanup also failed: {type(cleanup_error).__name__}: {cleanup_error}")
        raise
    else:
        # This whole-run entry point owns its temporary buffers, closures
        # and transfers, not the caller's store or borrowed tile_states.
        # Drain on both success and failure, then release only our references.
        run.close()

def _advance_clock(clock, tiles, last_b, _physics, *, fire_records=()) -> dict:
    """The domain's scalar carriers after ONE sweep, cross-checked per buffer.

    Every tile in a sweep starts from the same ``clock`` and takes the same
    cadence branches, so every buffer must end the sweep with the same
    counters.  If they do not, the tiles integrated different physics and the
    run is not a forecast -- so that is an error here, not a shrug.

    The one accurate exception is ``ysu_nan_guard_fires``: it counts a
    data-dependent guard, so a tile whose column blew up increments it and
    its neighbour does not.  It is a diagnostic counter that nothing in the
    step path reads, and the domain figure is the SUM over tiles -- which is
    also what a monolithic run would have counted, since the guard fires per
    column.

    ``carriers`` (the surface-radiation contract's produced-at records,
    riding in ``carrier_scalars`` since the streamed freshness fix) is
    published -- ``dict(ref)`` carries it -- but NOT cross-checked here.
    Eagerly every buffer ends a sweep with identical records (same restore,
    same cadence, same restamp), so a check would hold; under a CUDA-graph
    stepper it would not, structurally: a buffer whose last tile REPLAYED
    holds the capture's re-applied absolute records while a buffer that fell
    back to an eager step restamped at its own call, and under
    ``reuse="run"`` the two can differ by a whole cadence without either
    having integrated different physics.  The freshness law itself is the
    enforcement that matters, and it runs at every eager consumption.
    """
    per_buffer = [_physics.carrier_scalars(t) for t in tiles]
    ref = per_buffer[last_b]
    fires = 0
    for got in per_buffer:
        fires += int(got.get("ysu_nan_guard_fires", 0)) \
            - int(clock.get("ysu_nan_guard_fires", 0))
        for key in ("elapsed_seconds", "call_counts", "microphysics_updates", "fire_clocks",
                    "wrf_mass_carriers_cold", "radiation_owned", "lightning_clock"):
            if key in ref and got.get(key) != ref[key]:
                raise TiledRunError(
                    f"tile buffers disagree on the scalar carrier {key!r} "
                    f"after a sweep ({got.get(key)!r} vs {ref[key]!r}).  "
                    "Every tile in a sweep starts from the same domain clock "
                    "and must take the same cadence branches; a disagreement "
                    "means they integrated different physics.")
    out = dict(ref)
    if "fire_header" in out:
        from tilestream.sfire import fold_diagnostics
        fold_diagnostics(fire_records, out)
    if "ysu_nan_guard_fires" in out:
        out["ysu_nan_guard_fires"] = \
            int(clock.get("ysu_nan_guard_fires", 0)) + fires
    return out
