"""In-process build/step/hash harness for the out-of-core tiling spike.

Everything downstream of the de-risking spike builds against this module.  It
answers one question: can we construct a small ``DomainState``, call
``dycore.step`` on it repeatedly, and get a bit-reproducible answer, entirely
in process and with no data download?

Design notes that matter for tiling
-----------------------------------
* ``make_config`` defaults to a FULLY PERIODIC domain (``open_x=open_y=
  specified=nested=False``), flat terrain (``terrain_opt=0``) and identity map
  factors (``map_proj=0``).  Under those settings every setup array
  (``thb, pb, alb, phb, mub2d, ht, msft, msfu, msfv, f, e``) is horizontally
  UNIFORM and every vertical-coordinate array depends only on ``nz``.  That is
  what makes a tile's setup identical to the corresponding window of the full
  domain, which a tiled bit-exact gate requires.  Change those and the gate
  needs a setup gather too.
* :func:`make_geography` changes exactly that, on purpose.  It builds a REAL
  Lambert conformal grid (``configs/real74_d01.toml``'s ``map_proj=1`` at
  dx=dy=12 km) plus a latitude/longitude-anchored terrain, and the resulting
  ``msft/msfu/msfv/f/e/sina/cosa/ht`` and the terrain-following
  ``thb/pb/alb/phb/mub2d`` all vary horizontally.  A tile CANNOT rebuild
  them -- ``woof/static/projection.py:122-123`` defaults the reference
  point to the domain centre and ``_grid_xy`` (:176-179) counts DOMAIN
  indices, so a rebuilt tile believes it sits where the whole domain sits.
  They are gathered instead; :func:`tilestream.driver.geography_inventory`
  names them and ``run_tiled(geography=...)`` moves them.
* ``make_state`` deliberately re-implements the body of
  ``woof.verify.npref.random_acoustic_state`` instead of calling it, because
  that helper builds its OWN ``RunConfig`` from ``nx/ny/nz`` and we need to
  seed a state onto a config the caller already owns (a tile config differs
  from the parent only in ``nx``/``ny``).  The seeding amplitudes, field list,
  fill order and periodic-duplicate enforcement are copied verbatim from
  npref.py:2275-2314 so a state built here is byte-identical to one built by
  ``random_acoustic_state`` at the same seed and shape.  Verified by
  ``selftest_matches_npref`` below.
* ``hash_state`` hashes the STATE_SERIALIZED_ATTRS set -- the enforced persist
  list, which is exactly the streaming inventory.  ``hash_outputs`` is the
  benchmark's own ``_hash_outputs``, imported (not copied) from
  ``tools/benchmark_seeded_step.py``, kept as a cross-check.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from typing import Any, Iterator

import numpy as np


# The seeded-fill inventory, copied from woof/verify/npref.py:2284-2314.
# (name, amplitude, enforce-x-duplicate, enforce-y-duplicate)
_SEED_FIELDS: tuple[tuple[str, float, bool, bool], ...] = (
    ("u", 0.05, True, False),
    ("v", 0.05, False, True),
    ("w", 0.05, False, False),
    ("thp", 0.5, False, False),
    ("php", 20.0, False, False),
    ("mup", 20.0, False, False),
)
_SEED_TENDENCIES: tuple[tuple[str, float, bool, bool], ...] = (
    ("ru_t", 0.2, True, False),
    ("rv_t", 0.2, False, True),
    ("rw_t", 0.2, False, False),
    ("rth_t", 0.2, False, False),
    ("rph_t", 0.2, False, False),
    ("rmu_t", 1e-3, False, False),
)
_SEED_ACOUSTIC: tuple[tuple[str, float, bool, bool], ...] = (
    ("u_pp", 0.2, True, False),
    ("v_pp", 0.2, False, True),
    ("w_pp", 0.2, False, False),
    ("th_pp", 0.2, False, False),
    ("mu_pp", 0.5, False, False),
    ("ph_pp", 1.5e-3, False, False),
    ("p_pp", 1e-3, False, False),
    ("p_pp_old", 1e-3, False, False),
    ("al_pp", 3e-7, False, False),
)

DEFAULT_SEED = 20_260_731
DEFAULT_NZ = 49

#: Per-step horizontal dependency radius in mass cells FOR THE DEFAULT CONFIG
#: (``time_step_sound=4``).  Prefer :func:`halo_radius`, which is correct for
#: any sound-step count -- this constant is only the default's value.
HALO = 16


def halo_radius(cfg) -> int:
    """Per-step horizontal dependency radius, in mass cells, for ``cfg``.

    ``3 * 3`` from the three RK stages' 5th-order advection plus one cell per
    acoustic substep.  The RK3 stages run ``1``, ``ns // 2`` and ``ns``
    substeps of ``dt / ns`` (dycore.py:2215-2219, :2329), so a step takes
    ``1 + 3 * ns // 2`` substeps in total::

        radius = 9 + 1 + 3 * ns // 2 = 10 + 3 * ns // 2

    ``ns = 4`` gives the established 16.  MEASURED by NaN-cone growth on a
    160x160x49 periodic domain: the per-step increment is 16 / 19 / 22 for
    ``ns`` = 4 / 6 / 8, matching this formula exactly.  ``h_sca_adv_order``
    (2 vs 5) does NOT change the radius -- measured identical.

    Note the radius is field-dependent by one cell of staggering: mass points
    grow 14/30/46 after 1/2/3 steps while ``w`` and the boundary-normal
    momentum grow 15/31/47.  This function returns the CONSERVATIVE 16-per-step
    figure that covers every field.

    PHYSICS DOES NOT WIDEN IT -- measured, not assumed.  On a 256x192x49
    domain split 4x4 into 64x48 tiles (a gathered tile is ~40% of the domain,
    so the halo genuinely decides the answer), with the whole restart
    manifest streamed, the smallest halo that is bit-exact against a
    monolithic run is:

    ===================  ====  ====  ====  ====  ====
    rung                  N=1   N=2   N=3   N=5   N=8
    ===================  ====  ====  ====  ====  ====
    dry                    14    14    14    14    14
    full + MYNN + Noah-MP  13    14    14    14    14
    full, fast cadence     14     -    14     -    14
    ===================  ====  ====  ====  ====  ====

    So 16 bounds every measured rung with two cells to spare, and the
    physics rungs need no more than the dry one.  That is what the influence-
    cone measurement predicted (dry and full physics agree within +-1 cell at
    every amplitude): every scheme in this build is column-local, so physics
    adds no horizontal reach.

    READ FACT 1 BEFORE TOUCHING THIS.  halo 13 at ``full+MYNN+Noah-MP`` is
    BIT-EXACT at N=1 and then differs in 13 / 34 / 107 / 111 carriers at
    N = 2 / 3 / 5 / 8.  A one-step test certifies a halo that is wrong, and
    the wrong one is faster.

    RADIATIVE-OPEN BOUNDARIES DO NOT WIDEN IT EITHER, and that was the open
    prediction.  ``apply_open_zero_gradient`` assigns STATE at the window's
    edge column rather than perturbing a tendency, so the cone was expected
    to be the full per-step radius and the margin to vanish.  MEASURED at
    256x192x49, dx = 12 km, tile 32x32, dry, N=8, on a 4090
    (``tilestream.probe_open_halo``): the smallest bit-exact halo is 14 with
    ``open_x`` alone and 13 with ``open_x`` and ``open_y``, against the same
    14 the dry periodic row reports.  The prescribed 16 clears them by 2 and
    3.  The prediction is refuted for the same reason the specified seams
    were safe: at the START of a step every window cell holds the domain's
    own value, and a perturbation introduced partway through the step has
    strictly less than a full step left to travel.

    A DIAGNOSTIC DOES WIDEN IT, WHICH PHYSICS DOES NOT (2.2.0)
    ----------------------------------------------------------
    Everything above is about the CARRIER set: what a step reads in order
    to integrate the next one.  ``nwp_diagnostics = 1`` adds a consumer of
    a different shape -- ``woof.core.uh_diag`` evaluates ``UP_HELI_MAX``
    in the dycore EPILOGUE from a horizontal stencil two cells wide
    (vorticity radius 1, then the 9-point smoother's radius 1 on top of
    it; :data:`woof.core.uh_diag.UH_DIAGNOSTIC_HALO_CELLS`).  Physics adds
    no horizontal reach because every scheme in this build is column-local.
    This is not physics, and it reads its neighbours at the one instant in
    the step when a tile's halo has already been eaten from the outside in
    by that step's own dependency cone.

    The prescribed 16 clears the measured minimum of 14 by exactly 2, so
    the diagnostic consumes the whole margin and leaves none.  MEASURED at
    the 2.2.0 cut on a 438x350x49 run at tile 200x200: 74 of 75 fields
    bit-identical to the resident twin, and ``UP_HELI_MAX`` differing at
    ONE cell -- ``(j=331, i=200)``, 1.16e-10, one float32 ULP, at exactly
    the first interior column of the second tile, which is the signature
    of a stencil reaching one cell past what the halo still guarantees.
    The same configuration at halo 18 is bit-identical in all 75 fields at
    every frame.

    So the radius is widened by the diagnostic's own reach whenever the
    diagnostic is on -- by default, with no flag to set: a run that emits
    ``UP_HELI_MAX`` gets a correct one.  It changes no forecast, since the
    trajectory is bit-exact for any halo at or above 14, and it costs 4
    cells on each window axis (MEASURED on that run: 232x232 windows to
    236x236, 6.61 GiB of VRAM to 6.71 GiB).
    """
    ns = int(cfg.time_step_sound)
    if ns % 2 != 0:
        raise ValueError(f"time_step_sound must be even, got {ns}")
    radius = 10 + 3 * ns // 2
    # KF trigger 2 adds two nine-point averages to its forcing dependency.
    if int(cfg.cu_physics) == 1 and int(getattr(cfg, "kfeta_trigger", 1)) == 2:
        radius += 2
    if (int(getattr(cfg, "nwp_diagnostics", 0) or 0)
            or int(getattr(cfg, "gsd_diagnostics", 0) or 0)):
        from woof.core.uh_diag import UH_DIAGNOSTIC_HALO_CELLS

        radius += UH_DIAGNOSTIC_HALO_CELLS
    from tilestream.sfire import dependency_halo
    return max(radius, dependency_halo(cfg))


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

def make_config(nx: int, ny: int, nz: int = DEFAULT_NZ, *,
                periodic: bool = True, **overrides):
    """Return a ``RunConfig`` for an ``nx x ny x nz`` idealized domain.

    The non-overridden values reproduce
    ``woof.verify.npref.random_acoustic_state``'s config exactly
    (dx=dy=500 m, ztop=8000 m, dt=3 s, dry, flat, hybrid_opt=0), so a state
    built by :func:`make_state` on this config matches that helper bit for
    bit.  ``periodic=True`` (the default and the milestone-one setting)
    forces every lateral-boundary flag off.  ``periodic=False`` merely
    declines to force them; pass the flags you want through ``overrides``.

    ``run_seconds`` defaults to 0.0 and is NOT consumed by ``dycore.step``;
    it only matters to the case-runner clock.  Override it if you drive a
    higher-level loop.
    """
    from woof.config import RunConfig

    kwargs: dict[str, Any] = dict(
        nx=int(nx), ny=int(ny), nz=int(nz),
        dx=500.0, dy=500.0, ztop=8000.0,
        dt=3.0, run_seconds=0.0,
        hybrid_opt=0, terrain_opt=0, hill_height=0.0, hill_halfwidth=1500.0,
        moist=False, mp_physics=0,
    )
    if periodic:
        kwargs.update(open_x=False, open_y=False, specified=False,
                      nested=False, map_proj=0)
    kwargs.update(overrides)
    return RunConfig(**kwargs)


def tile_config(cfg, tile_nx: int, tile_ny: int):
    """Return ``cfg`` with only the horizontal extents replaced.

    Every physical parameter (dx, dt, sound steps, damping, physics
    selectors) is carried through untouched; only ``nx``/``ny`` change.  This
    is the config a gathered halo tile is stepped under, so ``tile_nx`` and
    ``tile_ny`` are the FULL gathered extents (T + 2*HALO), not the interior.

    THE LATERAL-BOUNDARY FLAGS ARE CARRIED THROUGH TOO, and that is correct
    -- checked rather than assumed, because it is the obvious place to
    suspect.  ``open_x``/``open_y``/``specified`` reach every buffer, so a
    tile applies the open-boundary treatment at all four of its window edges,
    including the ones that are interior seams.  There is no per-tile
    spelling of "open on the west only" either: ``RunConfig`` carries one
    flag per AXIS where WRF carries one per SIDE (``open_xs``/``open_xe``).

    It does not matter, for the same reason the inert seam tables in
    ``woof.core.streaming.window_interval`` do not.  At the start of a step
    every window cell, edge included, holds the domain's own value; the open
    treatment corrupts the edge only DURING the step (the advection bounds
    from stage 1, the zero-gradient overwrite at the end of each stage), so
    what is left to propagate is strictly less than the full-step dependency
    radius the halo is sized for.

    MEASURED, 256x192x49 at dx=12 km on a real Lambert grid, tile 32x32,
    halo 16 from :func:`halo_radius`, ``open_x=True, open_y=True`` -- the
    case where every one of the 48 tiles has two window edges that are
    interior seams: BIT-EXACT against the monolithic run at N = 1, 3 and 8,
    dry and at every physics rung.  The control fires: halo 8 differs in all
    nine dry carriers.  And the margin is not zero -- the smallest bit-exact
    halo at N=8 is 13, three below the prescribed one, so the seam treatment
    does not even reach the edge of the halo.  See :func:`halo_radius`.
    """
    return replace(cfg, nx=int(tile_nx), ny=int(tile_ny))


# --------------------------------------------------------------------------
# geography
# --------------------------------------------------------------------------

#: Lambert conformal parameters for the geography gate.  ``map_proj=1`` and
#: ``dx=dy=12000`` are ``configs/real74_d01.toml``'s own; the reference point
#: and true latitudes are the CONUS d01 values (the case's own ref point is
#: owned by its registered case module, not by the frozen RunConfig surface
#: that TOML carries -- MEASURED: ``RunConfig`` has 149 fields and none of
#: ref_lat/ref_lon/truelat1/truelat2/stand_lon/known_x/known_y).  What a
#: tiling gate needs from a projection is that latitude VARIES, and at
#: 12 km it varies by 0.108 deg per row.
REAL74_PROJECTION: dict[str, float] = dict(
    ref_lat=39.5, ref_lon=-98.5, truelat1=29.5, truelat2=49.5,
    stand_lon=-98.5)

#: The config overrides that turn a harness domain into a real-geography one.
#: ``terrain_opt=1`` is what makes ``thb/pb/alb/phb`` 3-D (state.py:605-607),
#: so it is the switch that moves 16.7 of the 17.1 B/mass-cell of geography.
GEOGRAPHY_OVERRIDES: dict[str, Any] = dict(
    map_proj=1, terrain_opt=1, dx=12000.0, dy=12000.0)


@dataclass(frozen=True)
class Geography:
    """The horizontally-varying INPUT of one domain: gathered, never rebuilt.

    Every field is host float64 at the extents of the config it was built
    for.  ``msfu``/``msfv`` carry the closing face (``(ny, nx+1)`` and
    ``(ny+1, nx)``); everything else is at mass points.

    THE PERIODIC-FACE RULE, which is not cosmetic.  On a PERIODIC axis
    ``spec.TileSpec._axis_gather`` reduces every window mod ``nx`` and never
    reads the alias slot, so a tile's u-face at the wrap takes the domain's
    column 0 -- while a monolithic run reads whatever sits in the domain's
    column ``nx``.  A periodic axis is therefore only self-consistent when
    ``msfu[:, nx] == msfu[:, 0]`` (x) or ``msfv[ny, :] == msfv[0, :]`` (y),
    exactly the duplicate ``woof/verify/npref.py`` already enforces on
    seeded ``u``/``v``.  :func:`make_geography` enforces it by default;
    ``periodic_faces=False`` leaves the raw Lambert values, which is the
    negative control that shows the rule is essential.

    IT IS A PER-AXIS RULE, because periodicity is (``dycore._boundary_x`` and
    ``_boundary_y`` are independent, and so are ``plan_tiles``'s
    ``periodic_x``/``periodic_y``).  ``open_x=True, open_y=False`` needs the
    y face duplicated and the x face left alone; getting only the plan right
    and leaving ``periodic_faces=False`` on both axes leaves the streamed run
    differing from the monolithic one in all nine dry carriers after ONE
    step, in rows 0-8 and 183-191 of a 192-row domain and nowhere else --
    measured, and it is the ONLY thing that was still wrong once the plan was
    per-axis.  Pass ``periodic_faces_x`` / ``periodic_faces_y``.
    """

    grid: Any
    msft: np.ndarray
    msfu: np.ndarray
    msfv: np.ndarray
    f: np.ndarray
    e: np.ndarray
    sina: np.ndarray
    cosa: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    terrain: np.ndarray | None

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(int(s) for s in self.msft.shape)

    @property
    def has_msf(self) -> bool:
        """``state.py:799-801``'s rule, evaluated on the WHOLE domain."""
        return bool((self.msft != 1.0).any() or (self.msfu != 1.0).any()
                    or (self.msfv != 1.0).any())

    @property
    def rotational(self) -> bool:
        """``state.py:802-803``'s rule, evaluated on the WHOLE domain."""
        return bool(self.has_msf or (self.f != 0.0).any()
                    or (self.e != 0.0).any())


def terrain_from_latlon(lat, lon, *, height: float = 800.0,
                        lat0: float = 39.0, lon0: float = -105.0,
                        lat_width: float = 4.0, lon_width: float = 6.0):
    """A smooth ridge anchored to GEOGRAPHY, not to grid indices.

    Anchoring to lat/lon rather than to ``i, j`` is the whole point: a tile
    that rebuilds this from ``tile_cfg`` gets it wrong for exactly the reason
    ``msft`` is wrong, which is what a real geogrid ``HGT_M`` window does.  A
    terrain defined on indices would rebuild correctly on a centred tile and
    hide the bug.
    """
    return height * np.exp(-((np.asarray(lat, dtype=np.float64) - lat0)
                             / lat_width) ** 2
                           - ((np.asarray(lon, dtype=np.float64) - lon0)
                              / lon_width) ** 2)


def make_geography(cfg, *, terrain: bool = True, periodic_faces: bool = True,
                   periodic_faces_x: bool | None = None,
                   periodic_faces_y: bool | None = None,
                   projection: dict | None = None, **terrain_kwargs
                   ) -> Geography:
    """The real Lambert geography of ``cfg``'s OWN extents.

    Built from ``cfg.nx``/``cfg.ny``/``cfg.dx``, so calling it on a
    ``tile_config`` reproduces precisely the per-tile REBUILD this lane
    exists to eliminate -- which is why the gate uses it as its negative
    control as well as its buffer initialiser.  On a real run the parent's
    arrays are gathered and whatever the buffer was built with is
    overwritten.

    MEASURED, 192x192 split 3x3 with halo 16 at dx=12 km: a rebuilt tile is
    displaced by ``ci0 + (cnx+1)/2 - (nx+1)/2`` cells in each axis, up to
    5.83 deg of latitude and 827 km of great circle, 17.9% in Coriolis and
    1.39% in the map factor -- and the exactly-centred tile is BIT-EXACT,
    which is what lets a one-tile test certify the bug.
    """
    from woof.static.lambert import LambertGrid

    params = dict(REAL74_PROJECTION if projection is None else projection)
    grid = LambertGrid(e_we=int(cfg.nx) + 1, e_sn=int(cfg.ny) + 1,
                       dx=float(cfg.dx), dy=float(cfg.dy), **params)
    msft = np.asarray(grid.mapfac_m(), dtype=np.float64)
    msfu = np.asarray(grid.mapfac_u(), dtype=np.float64)
    msfv = np.asarray(grid.mapfac_v(), dtype=np.float64)
    fx = periodic_faces if periodic_faces_x is None else periodic_faces_x
    fy = periodic_faces if periodic_faces_y is None else periodic_faces_y
    if fx:
        msfu[:, -1] = msfu[:, 0]
    if fy:
        msfv[-1, :] = msfv[0, :]
    f, e = grid.coriolis_m()
    sina, cosa = grid.rotation_m()
    lat, lon = grid.latlon_mass()
    terrain_z = (terrain_from_latlon(lat, lon, **terrain_kwargs)
                 if terrain else None)
    return Geography(grid=grid,
                     msft=msft, msfu=msfu, msfv=msfv,
                     f=np.asarray(f, dtype=np.float64),
                     e=np.asarray(e, dtype=np.float64),
                     sina=np.asarray(sina, dtype=np.float64),
                     cosa=np.asarray(cosa, dtype=np.float64),
                     lat=np.asarray(lat, dtype=np.float64),
                     lon=np.asarray(lon, dtype=np.float64),
                     terrain=terrain_z)


def neutral_geography(cfg, *, latitude_deg: float = 35.0,
                      longitude_deg: float = -97.5,
                      terrain_height: float = 0.0) -> Geography:
    """Identity map factors, zero Coriolis, flat terrain -- a POISON buffer.

    This is what a tile buffer is built with, and it is deliberately not
    weather: ``msf == 1``, ``f == e == sina == 0``, ``cosa == 1``, a constant
    terrain and a constant latitude that is nowhere near the domain's.  Two
    things follow, and both are the point.

    * A geography array the gather fails to write stays at an obviously
      wrong value instead of at a plausible one -- the same argument
      ``driver._empty_like_store`` makes for its NaN poison.
    * ``has_msf`` and ``rotational`` come out FALSE (state.py:799-803), so
      the buffer's Coriolis+curvature kernel (dycore.py:539) and every
      msf-weighted path are OFF until ``run_tiled`` imposes the DOMAIN's
      flags.  That makes the imposition essential and therefore testable:
      ``run_tiled(impose_geography_flags=False)`` is a negative control that
      MUST fail, and it does.

    Terrain-enabled buffers carry a full ``(ny, nx)`` placeholder. With
    ``terrain_opt=0``, terrain is ``None`` so the base profiles remain 1-D,
    matching the state allocation. Neutral geometry must honor the same
    shape contract as the prepared domain before any gather can occur.
    """
    ny, nx = int(cfg.ny), int(cfg.nx)
    ones = np.ones((ny, nx), dtype=np.float64)
    zeros = np.zeros((ny, nx), dtype=np.float64)
    return Geography(
        grid=None,
        msft=ones.copy(),
        msfu=np.ones((ny, nx + 1), dtype=np.float64),
        msfv=np.ones((ny + 1, nx), dtype=np.float64),
        f=zeros.copy(), e=zeros.copy(), sina=zeros.copy(), cosa=ones.copy(),
        lat=np.full((ny, nx), float(latitude_deg)),
        lon=np.full((ny, nx), float(longitude_deg)),
        terrain=(None if cfg.terrain_opt == 0 else
                 np.full((ny, nx), float(terrain_height))))


def install_geography(state, geo: Geography) -> None:
    """Push ``geo``'s map factors, Coriolis and rotation onto ``state``.

    ``DomainState.set_map_coriolis`` (state.py:773-803) is the only
    sanctioned setter and this calls it -- but note what it does at the end:
    it RECOMPUTES ``has_msf``/``rotational`` from the arrays it was just
    handed.  On a tile that is the tile's window, not the domain's, and both
    flags gate whole code paths (``moist.py:470-559``, ``physics.py:1268``,
    ``dycore.py:539``).  A tiled run therefore re-imposes the parent's flags
    after the gather; see :func:`tilestream.driver.install_geography_window`.
    Terrain is NOT set here: it enters through the base state, which has to
    be built with it (``grid.make_base_state(terrain_z=...)``).
    """
    state.set_map_coriolis(msft=geo.msft, msfu=geo.msfu, msfv=geo.msfv,
                           f=geo.f, e=geo.e, sina=geo.sina, cosa=geo.cosa)


# --------------------------------------------------------------------------
# state
# --------------------------------------------------------------------------

def make_state(cfg, seed: int = DEFAULT_SEED, *, stretch=None,
               msf_amp: float = 0.0, f_amp: float = 0.0,
               geography: Geography | None = None):
    """Build a seeded, discretely balanced ``DomainState`` on ``cfg``.

    Returns a state whose ``p/al/alt`` diagnostics are consistent with its
    seeded ``thp/php/mup`` (``update_diagnostics`` is run before returning),
    ready to hand straight to :func:`run_steps`.

    ``seed`` drives a ``numpy.random.default_rng``; the draw order and
    amplitudes are npref's.  Because ``rng.standard_normal(arr.shape)``
    consumes a shape-dependent number of variates, a tile built here does
    NOT contain the same numbers as the matching window of a larger domain
    built here -- for a tiling gate you gather from the parent state, you do
    not re-seed the tile.

    ``geography`` (a :class:`Geography` at ``cfg``'s extents) installs a real
    projection and terrain-following base state.  It must have been built
    for THIS cfg; the tiled lane builds one per tile buffer and then
    overwrites it with the parent's window.
    """
    import cupy as cp

    from woof.core.diagnostics import update_diagnostics
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.state import init_at_rest

    coord = make_vertical_coord(cfg.nz, stretch=stretch,
                                hybrid_opt=cfg.hybrid_opt, etac=cfg.etac)
    terrain_z = None if geography is None else geography.terrain
    base = make_base_state(coord, lambda z: np.full_like(z, 300.0),
                           p_surf=cfg.p_surf, ztop=cfg.ztop,
                           terrain_z=terrain_z)
    state = init_at_rest(cfg, coord, base)
    if geography is not None:
        install_geography(state, geography)
    rng = np.random.default_rng(seed)

    def fill(name: str, amp: float, xdup: bool, ydup: bool) -> None:
        arr = getattr(state, name)
        vals = amp * rng.standard_normal(arr.shape)
        if xdup:
            vals[..., -1] = vals[..., 0]
        if ydup:
            vals[:, -1, :] = vals[:, 0, :]
        arr[...] = cp.asarray(vals, dtype=arr.dtype)

    for name, amp, xdup, ydup in _SEED_FIELDS:
        fill(name, amp, xdup, ydup)
    state.w[0] = 0.0
    state.w[-1] = 0.0
    update_diagnostics(state)                 # consistent p, al, alt

    for name, amp, xdup, ydup in _SEED_TENDENCIES:
        fill(name, amp, xdup, ydup)
    for name, amp, xdup, ydup in _SEED_ACOUSTIC:
        fill(name, amp, xdup, ydup)
        if name == "ph_pp":
            state.ph_pp[0] = 0.0              # fixed surface: phi''(sfc) = 0

    if msf_amp > 0.0 or f_amp > 0.0:
        ny, nx = cfg.ny, cfg.nx
        msft = 1.0 + msf_amp * rng.random((ny, nx))
        msfu = 1.0 + msf_amp * rng.random((ny, nx + 1))
        msfv = 1.0 + msf_amp * rng.random((ny + 1, nx))
        msfu[:, -1] = msfu[:, 0]
        msfv[-1, :] = msfv[0, :]
        state.set_map_coriolis(msft=msft, msfu=msfu, msfv=msfv,
                               f=f_amp * rng.random((ny, nx)),
                               e=f_amp * rng.random((ny, nx)))
    return state


def declared_glw_kwargs(cfg) -> dict:
    """``{"glw": ...}`` for a rung whose downward longwave has no scheme.

    The gate's idealised rungs pair a land-surface scheme with radiation
    off, and ``initialize_physics`` refuses that combination by name: no
    longwave scheme computes GLW and Noah reads it every surface step, so
    whatever is in the buffer becomes surface physics.  Through 1.8.7 the
    buffer was silently filled with 300.0 W m-2 and the rungs ran on it.

    Declaring the same constant is therefore a statement of what these
    rungs were always doing, not a change to what they do: every digest
    the gate compares is unchanged, and the run now says out loud which of
    the three accurate origins its longwave has.

    THE CLASSIFICATION IS NOT RESTATED HERE.  It comes from
    ``physics_compat.downward_longwave_disposition``, the same function the
    config-load guard, the initialize-time guard and the receipt line all
    read, so this harness cannot drift wider or narrower than the door.

    Under [tiles] this matters twice over.  A streamed domain builds one
    ``PhysicsDriver`` per tile buffer, and a per-buffer GLW decision would
    make ``glw_provenance`` a coin flip over which buffer answered the
    receipt.  A declared constant is identical in every buffer by
    construction.
    """
    from woof.core.physics import DECLARED_CONSTANT_GLW_WM2
    from woof.physics_compat import downward_longwave_disposition

    kind, _consumer = downward_longwave_disposition(
        ra_lw_physics=int(cfg.ra_lw_physics),
        ra_sw_physics=int(cfg.ra_sw_physics),
        sf_surface_physics=int(cfg.sf_surface_physics))
    if kind in ("consumed", "published"):
        return {"glw": DECLARED_CONSTANT_GLW_WM2}
    return {}


#: The constant the SWDOWN/GSW siblings of :func:`declared_glw_kwargs`
#: declare.  Zero, because zero is what ``initialize_physics`` has always
#: allocated for both shortwave carriers -- declaring it changes no digest
#: the gate compares, it only gives the number a named origin.  Unlike GLW
#: there is no plausible-looking fill to preserve: a radiation-off rung's
#: shortwave was always the allocation zeros.
DECLARED_CONSTANT_SHORTWAVE_WM2 = 0.0


def declared_swdown_kwargs(cfg) -> dict:
    """``{"swdown": ...}`` for a rung whose shortwave has no scheme.

    The SWDOWN sibling of :func:`declared_glw_kwargs`, for the same seam:
    the carrier-contract lane made SWDOWN a carrier with a producer or a
    declaration, and the gate's ``+Noah``-class rungs pair a land-surface
    scheme with radiation off, so nothing produces it and Noah/Noah-MP
    read it every surface step.  Through 1.9.x the buffer was silently the
    allocation zeros and the rungs ran on it; declaring the same zero is a
    statement of what these rungs were always doing, so every digest the
    gate compares is unchanged and the refusal cannot fire.

    THE CLASSIFICATION IS NOT RESTATED HERE.  Which carriers a scheme
    consumes comes from ``woof.core.radiation_carriers.consumer_carriers``
    -- the consumption check's own matrix -- and whether a producer exists
    comes from ``woof.config.radiation_scheme_ids``, the same resolver
    ``initialize_physics`` dispatches radiation on.  With any shortwave
    scheme active the declaration is withheld: WRF ordering runs radiation
    before the surface layer, so the scheme's own flux is in the field
    before anything reads it, and a declaration here would misname a
    scheme-produced carrier.

    COSZEN needs no sibling: under radiation-off Noah-MP,
    ``initialize_physics`` itself attaches the analytic solar-geometry
    provider (a real producer on the radiation cadence), and under any
    shortwave scheme the scheme writes it.  The law already supplies both.
    """
    from woof.config import radiation_scheme_ids
    from woof.core.radiation_carriers import consumer_carriers

    _lw, sw = radiation_scheme_ids(cfg)
    if int(sw) > 0:
        return {}
    if "swdown" in consumer_carriers(int(cfg.sf_surface_physics)):
        return {"swdown": DECLARED_CONSTANT_SHORTWAVE_WM2}
    return {}


def declared_carrier_kwargs(cfg) -> dict:
    """Every ``initialize_physics`` declaration a schemeless sky needs.

    The union of :func:`declared_glw_kwargs` and
    :func:`declared_swdown_kwargs` -- one call for the two carriers that
    are declared through ``initialize_physics`` keywords.  GSW travels
    through the driver's own forcing door instead
    (:func:`declare_offline_gsw`), because ``initialize_physics`` has no
    ``gsw`` keyword: RUC's net shortwave is an offline forcing by design.
    """
    result = {**declared_glw_kwargs(cfg), **declared_swdown_kwargs(cfg)}
    if int(cfg.sf_surface_physics) == 1:
        # The seeded builder uses initialize_physics' default MODIS category
        # 10 and its April Northern Hemisphere date (summer table). Supply
        # the same THERIN/100 that landuse_init would give that category.
        from woof.core.landuse import load_landuse_table
        table = load_landuse_table("MODIFIED_IGBP_MODIS_NOAH")
        result["thermal_inertia"] = np.float32(table.values[0, 9, 4]) / np.float32(100.0)
    return result


def declare_offline_gsw(driver, cfg) -> None:
    """Declare RUC's net shortwave at the allocation zeros, or do nothing.

    The GSW sibling of :func:`declared_glw_kwargs`, through the one door
    that exists for it: ``PhysicsDriver.set_forcing`` is how an offline-
    forced RUC run supplies the net shortwave its LSM consumes, and it
    labels the carrier ``external_array`` with the receipt naming it.  The
    written value is the same zero the buffer already holds, so no digest
    moves; what changes is that the run says out loud where its net
    shortwave came from instead of integrating an unlabelled allocation.

    A ``driver`` of ``None`` (no physics on this rung) and a rung whose
    shortwave scheme produces GSW are both left untouched, for the reason
    :func:`declared_swdown_kwargs` gives.
    """
    from woof.config import radiation_scheme_ids
    from woof.core.radiation_carriers import consumer_carriers

    if driver is None:
        return
    _lw, sw = radiation_scheme_ids(cfg)
    if int(sw) > 0:
        return
    if "gsw" not in consumer_carriers(int(cfg.sf_surface_physics)):
        return
    driver.set_forcing(gsw=DECLARED_CONSTANT_SHORTWAVE_WM2)


def make_physics_state(cfg, seed: int = DEFAULT_SEED, *,
                       geography: Geography | None = None, start_time=None,
                       coord=None, poison: str = "noise", **initialize_kwargs):
    """``(state, driver)`` on a real projection, real terrain and real lat/lon.

    ``geography=None`` delegates verbatim to
    :func:`tilestream.physics_inventory.default_builder`, so every existing
    rung is bit-unchanged.  With a :class:`Geography` it differs in exactly
    four places, and each one is a thing a tile would otherwise get wrong:

    * the base state is built on ``geo.terrain``, so ``ht``, ``mub2d`` and
      the 3-D ``thb/pb/alb/phb`` all vary horizontally (16.8 of the
      17.1 B/mass-cell of geography);
    * the map factors, Coriolis and rotation come from the Lambert grid, so
      ``has_msf`` and ``rotational`` are BOTH TRUE and the msf-weighted and
      Coriolis kernel paths are live rather than skipped;
    * ``initialize_physics`` receives per-column latitude/longitude ARRAYS,
      which is what every solar-zenith path in the tree consumes -- Dudhia
      (dudhia.py:55-103), RRTMGP (rrtmgp.py:1842-1882), legacy RRTMG
      (rrtmg_legacy.py:155-181), Noah-MP (noahmp_runtime.py:617-624) and the
      analytic scheme all take ``(lat, lon)`` arrays and only the clock terms
      (declination, equation of time, solcon) are scalars;
    * ``qv``/``qc`` are seeded off the COLUMN-MEAN height, because with
      terrain ``state.height_half()`` is ``(nz, ny, nx)`` rather than
      ``(nz,)`` and ``default_builder``'s ``qv_col[:, None, None]`` would
      raise.

    ``coord`` overrides the vertical coordinate the buffer is built on, and
    a REAL case must supply it.  The default rebuilds
    ``make_vertical_coord(nz, hybrid_opt, etac)``, i.e. the DEFAULT stretch;
    a real case runs its own explicit eta table and the purely vertical setup
    arrays (``c1h..c4f``, ``znu``, ``znw``, ``dnw``, ``fnm``, ``cf1..``) are
    deliberately NOT gathered by ``driver.geography_inventory`` because a
    tile is supposed to rebuild them exactly.  Built from the wrong table
    they are the right shape, the right dtype, and wrong, and nothing
    downstream notices.  ``tilestream.realdata.make_real_tile_state`` records
    the same trap for the dynamics-only lane.

    ``poison`` names the fill of the seeded fields.  ``"noise"`` is the
    seeded Gaussian every gate integrates.  ``"constant"`` writes each
    field's amplitude as one constant and the sounding columns without
    noise: as obviously not weather as the noise, and for a buffer every
    one of whose carriers is gathered from a prepared store before it serves
    a tile (:func:`woof.core.streaming.prepared_tile_state_factory`) the
    noise bought nothing but its cost -- 16.5 s of host random numbers per
    rank at HRRR's 1799x1059x50, measured on box S 2026-10-06, before step 1
    could start.

    ``initialize_kwargs`` are forwarded to ``initialize_physics`` untouched.
    """
    from datetime import datetime, timezone

    import cupy as cp

    from tilestream import physics_inventory as _physics

    if geography is None:
        return _physics.default_builder(cfg, seed, start_time=start_time,
                                        **initialize_kwargs)

    from woof.core.diagnostics import update_diagnostics
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.physics import initialize_physics, physics_driver_required
    from woof.core.state import init_at_rest
    from woof.verify.cases.wk82 import wk82_sounding

    if coord is None:
        coord = make_vertical_coord(cfg.nz, hybrid_opt=cfg.hybrid_opt,
                                    etac=cfg.etac)
    base = make_base_state(coord, lambda z: wk82_sounding(z)[0],
                           p_surf=cfg.p_surf, ztop=cfg.ztop,
                           terrain_z=geography.terrain)
    state = init_at_rest(cfg, coord, base)
    install_geography(state, geography)
    if poison not in ("noise", "constant"):
        raise ValueError(f"poison must be 'noise' or 'constant', got {poison!r}")
    constant = poison == "constant"
    rng = np.random.default_rng(seed)

    def noise(shape):
        return np.zeros(shape) if constant else rng.standard_normal(shape)

    def fill(name, amp, xdup, ydup):
        arr = getattr(state, name)
        if constant:
            arr[...] = amp
            return
        vals = amp * rng.standard_normal(arr.shape)
        if xdup:
            vals[..., -1] = vals[..., 0]
        if ydup:
            vals[:, -1, :] = vals[:, 0, :]
        arr[...] = cp.asarray(vals, dtype=arr.dtype)

    for name, amp, xdup, ydup in _SEED_FIELDS:
        fill(name, amp, xdup, ydup)
    state.w[0] = 0.0
    state.w[-1] = 0.0
    update_diagnostics(state)
    for name, amp, xdup, ydup in _SEED_TENDENCIES:
        fill(name, amp, xdup, ydup)
    for name, amp, xdup, ydup in _SEED_ACOUSTIC:
        fill(name, amp, xdup, ydup)
        if name == "ph_pp":
            state.ph_pp[0] = 0.0

    if getattr(state, "qv", None) is not None:
        z = np.asarray(state.height_half(), dtype=np.float64)
        z_col = z.mean(axis=(1, 2)) if z.ndim == 3 else z
        qv_col = wk82_sounding(z_col)[1]
        state.qv[...] = cp.asarray(
            np.maximum(qv_col[:, None, None]
                       * (1.0 + 0.20 * noise(state.qv.shape)),
                       1e-9), dtype=state.qv.dtype)
        if getattr(state, "qc", None) is not None:
            blob = 4.0e-4 * np.exp(-((z_col - 4000.0) / 2500.0) ** 2)
            state.qc[...] = cp.asarray(
                np.maximum(blob[:, None, None]
                           * (1.0 + 0.3 * noise(state.qc.shape)), 0.0),
                dtype=state.qc.dtype)
        update_diagnostics(state)

    from woof.core.state import adopt_wrf_theta_carrier
    adopt_wrf_theta_carrier(state)
    if not physics_driver_required(cfg):
        return state, None
    if start_time is None:
        start_time = datetime(2011, 4, 27, 18, 0, 0, tzinfo=timezone.utc)
    driver = initialize_physics(
        state, cfg,
        radiation_start_time=start_time,
        radiation_latitude=geography.lat, radiation_longitude=geography.lon,
        noahmp_start_time=start_time,
        noahmp_latitude=geography.lat, noahmp_longitude=geography.lon,
        **{**declared_carrier_kwargs(cfg), **initialize_kwargs})
    declare_offline_gsw(driver, cfg)
    return state, driver


def geography_builder(geography_fn=None, coord_fn=None, **geo_kwargs):
    """A ``(cfg, seed) -> (state, driver)`` builder for a geography tile buffer.

    ``geography_fn(cfg)`` is evaluated on whatever config it is handed, so
    the two useful choices are:

    :func:`neutral_geography` (the default)
        a poison buffer, overwritten by ``run_tiled``'s geography gather.
        This is what a real tiled run uses.

    :func:`make_geography`
        the per-tile REBUILD, i.e. exactly the defect this lane removes: the
        buffer gets the geography of a domain centred on the TILE.  Run with
        ``geography=None`` it reproduces today's behaviour and is the gate's
        negative control.
    """
    build = geography_fn or (lambda cfg: neutral_geography(cfg, **geo_kwargs))

    def builder(cfg, seed: int = DEFAULT_SEED, **kwargs):
        if coord_fn is not None:
            kwargs.setdefault("coord", coord_fn(cfg))
        return make_physics_state(cfg, seed, geography=build(cfg), **kwargs)

    return builder


# --------------------------------------------------------------------------
# stepping
# --------------------------------------------------------------------------

def run_steps(state, cfg, n: int, *, sync: bool = True, **step_kwargs) -> None:
    """Call ``dycore.step(state, cfg)`` ``n`` times.

    ``sync=True`` does one ``deviceSynchronize`` at the end (not per step), so
    an exception raised by a kernel launched during the loop surfaces before
    the caller reads anything.  Pass ``sync=False`` only when the caller owns
    the synchronization (e.g. a stream-overlapped tile pipeline).
    """
    import cupy as cp
    from woof.core.dycore import step

    for _ in range(int(n)):
        step(state, cfg, **step_kwargs)
    if sync:
        cp.cuda.runtime.deviceSynchronize()


# --------------------------------------------------------------------------
# identity
# --------------------------------------------------------------------------

def state_arrays(state) -> dict[str, Any]:
    """The persisted inventory: ``{name: cupy array}`` over the contract set.

    Keys are drawn from
    ``woof.state_serialization_contract.STATE_SERIALIZED_ATTRS`` in that
    tuple's order.  Names the configuration did not allocate are ABSENT (the
    attribute is missing or ``None``) -- that is the contract's own
    convention, and the dry periodic milestone-one state populates only a
    subset.  This dict is the streaming list: exactly these arrays must live
    in pinned host RAM and be gathered/scattered per tile.
    """
    import cupy as cp

    from woof.state_serialization_contract import STATE_SERIALIZED_ATTRS

    out: dict[str, Any] = {}
    for name in STATE_SERIALIZED_ATTRS:
        value = getattr(state, name, None)
        if isinstance(value, cp.ndarray):
            out[name] = value
    return out


def setup_arrays(state) -> dict[str, Any]:
    """The horizontally-uniform setup inventory (``STATE_SETUP_ARRAYS``).

    Not streamed per tile under milestone-one settings: with terrain_opt=0
    and map_proj=0 every entry is either 1-D in z or constant in (y, x), so a
    tile can hold a full copy.  Exposed so a later terrain/projection lane can
    check that assumption instead of inheriting it.
    """
    import cupy as cp

    from woof.state_serialization_contract import (
        STATE_DERIVED_SETUP_ARRAYS, STATE_SETUP_ARRAYS)

    out: dict[str, Any] = {}
    # Including the DERIVED entries: this feeds setup_window_mismatches,
    # the gate that CHECKS the tile-equals-domain claim rather than
    # assuming it, and a gate that cannot see an array cannot check it.
    for name in STATE_SETUP_ARRAYS + STATE_DERIVED_SETUP_ARRAYS:
        value = getattr(state, name, None)
        if isinstance(value, cp.ndarray):
            out[name] = value
    return out


def _digest_payload(name: str, array) -> bytes:
    import cupy as cp

    host = np.ascontiguousarray(cp.asnumpy(array))
    return (name.encode("utf-8")
            + host.dtype.str.encode("ascii")
            + np.asarray(host.shape, dtype=np.int64).tobytes()
            + host.tobytes(order="C"))


def hash_state(state) -> str:
    """SHA-256 over the persisted set, in ``STATE_SERIALIZED_ATTRS`` order.

    Each contributing array folds in its name, dtype string, shape and raw
    C-order bytes, so a shape or dtype change cannot collide with a value
    change.  This is the bit-exact gate's comparator.
    """
    digest = hashlib.sha256()
    arrays = state_arrays(state)
    if not arrays:
        raise RuntimeError("no persisted arrays found on the state")
    for name, array in arrays.items():
        digest.update(_digest_payload(f"state.{name}", array))
    return digest.hexdigest()


def hash_field_map(state) -> dict[str, str]:
    """Per-field SHA-256 digests, for localizing a gate failure."""
    return {name: hashlib.sha256(_digest_payload(f"state.{name}",
                                                 array)).hexdigest()
            for name, array in state_arrays(state).items()}


def hash_window(state, j0: int, j1: int, i0: int, i1: int) -> str:
    """SHA-256 over a horizontal MASS window of the persisted set.

    Applies the WRF-ARW staggering: mass-point and w/phi fields are sliced
    ``[..., j0:j1, i0:i1]``, ``u`` is sliced ``[..., j0:j1, i0:i1+1]`` and
    ``v`` is sliced ``[..., j0:j1+1, i0:i1]``.  This is the comparator a tiled
    run's interior is scored against -- slicing every array identically is
    wrong and this function exists so nobody does it.
    """
    import cupy as cp

    digest = hashlib.sha256()
    for name, array in state_arrays(state).items():
        if array.ndim < 2:
            window = array
        else:
            xi1 = i1 + 1 if name == "u" else i1
            yj1 = j1 + 1 if name == "v" else j1
            window = array[..., j0:yj1, i0:xi1]
        digest.update(_digest_payload(f"state.{name}",
                                      cp.ascontiguousarray(window)))
    return digest.hexdigest()


def hash_outputs(state) -> tuple[str, dict[str, str]]:
    """The seeded-step benchmark's own output hash, imported not copied.

    ``tools/benchmark_seeded_step.py._hash_outputs`` walks a wider set than
    the persist contract (it includes physics fields and mp/cu scratch), so it
    is a stricter cross-check but a less meaningful streaming inventory.  Kept
    so this harness's numbers can be tied back to the existing benchmark.
    """
    from tools.benchmark_seeded_step import _hash_outputs

    return _hash_outputs(state)


# --------------------------------------------------------------------------
# convenience
# --------------------------------------------------------------------------

def build_and_run(nx: int, ny: int, nz: int = DEFAULT_NZ, steps: int = 3, *,
                  seed: int = DEFAULT_SEED, **cfg_overrides):
    """``(state, cfg, hash)`` for an ``nx x ny x nz`` domain after ``steps``."""
    cfg = make_config(nx, ny, nz, **cfg_overrides)
    state = make_state(cfg, seed=seed)
    run_steps(state, cfg, steps)
    return state, cfg, hash_state(state)


def selftest_matches_npref(nx: int = 16, ny: int = 16, nz: int = 8,
                           seed: int = DEFAULT_SEED) -> bool:
    """True iff :func:`make_state` reproduces ``random_acoustic_state``.

    Guards the copied seeding block against drift in npref.py.
    """
    from woof.verify.npref import random_acoustic_state

    reference, _ref_cfg = random_acoustic_state(seed=seed, nz=nz, ny=ny, nx=nx)
    mine = make_state(make_config(nx, ny, nz), seed=seed)
    names = ("u", "v", "w", "thp", "php", "mup", "p", "al", "alt",
             "ru_t", "rv_t", "rw_t", "rth_t", "rph_t", "rmu_t",
             "u_pp", "v_pp", "w_pp", "th_pp", "mu_pp", "ph_pp",
             "p_pp", "p_pp_old", "al_pp")
    import cupy as cp

    for name in names:
        a = getattr(reference, name, None)
        b = getattr(mine, name, None)
        if a is None and b is None:
            continue
        if a is None or b is None:
            return False
        if not bool(cp.all(a == b)):
            return False
    return True


__all__ = [
    "DEFAULT_NZ", "DEFAULT_SEED", "GEOGRAPHY_OVERRIDES", "HALO",
    "REAL74_PROJECTION", "Geography",
    "build_and_run", "geography_builder", "halo_radius", "hash_field_map",
    "hash_outputs", "hash_state", "hash_window", "install_geography",
    "make_config", "make_geography", "make_physics_state", "make_state",
    "neutral_geography", "run_steps", "selftest_matches_npref",
    "setup_arrays", "state_arrays", "terrain_from_latlon", "tile_config",
]
