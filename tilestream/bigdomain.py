"""A forecast at a domain the resident path cannot run, start to finish.

Everything else in this package proves that a tile stream reproduces a
monolithic run bit for bit.  That is a claim about ARITHMETIC.  This module
exists to produce the other kind of evidence -- a picture of the atmosphere at
a domain size the monolithic path cannot reach at all -- and it turns out that
needs one piece nothing here had: an INITIAL CONDITION that never exists on
the device.

    ``harness.make_physics_state`` allocates the full domain.  At
    ``full+MYNN+Noah-MP`` that is 275.5 B/cell of carriers plus the dycore
    and physics working set, so the state builder hits the same VRAM wall
    the stepping loop was built to get past.  A streamed forecast whose
    initialisation is monolithic is capped by its initialisation.

So the state here is built ONE ROW SLAB AT A TIME and scattered into the
pinned host store, and the slab build is legitimate only because every step
of the construction is COLUMN-LOCAL.  That is a claim, so
:func:`selftest_slab_build_matches_monolithic` measures it against a
monolithic ``make_physics_state``-shaped build at a size where both fit,
rather than leaving it as an argument.  Two places it would NOT have been
true, and what was done about each:

* the random perturbations.  ``numpy.random.default_rng(seed).
  standard_normal(shape)`` consumes a shape-dependent number of variates, so
  a slab that draws its own noise is not the domain's window of anything.
  The noise is therefore drawn ONCE at full domain shape (:class:`SeededNoise`)
  and windowed, which is also what makes it continuous across slab seams --
  a per-slab draw would put a visible discontinuity every ``slab_rows`` rows
  and it would look exactly like weather.
* the vapour profile.  ``harness.make_physics_state`` seeds ``qv`` off
  ``state.height_half().mean(axis=(1, 2))`` -- a DOMAIN mean, which a slab
  cannot compute.  Here the profile is evaluated on the flat-terrain column
  heights instead, which is a function of ``nz``/``ztop`` alone and therefore
  identical for every slab.  It is a different initial condition from the
  harness's by the amount terrain tilts the mean, and it is the one that
  tiles.

WHAT THE INITIAL CONDITION ACTUALLY IS -- SAY IT ON THE PLOT
------------------------------------------------------------
There is no GRIB in this path and no WPS.  The state is IDEALISED and the
figures must say so.  Specifically:

* the Weisman-Klemp 1982 analytic sounding (``woof.verify.cases.wk82``),
  horizontally uniform -- a severe-storm environment, roughly 2200 J/kg CAPE;
* the WK82 quarter-circle hodograph (WRF ``em_quarter_ss``'s own wind
  columns, shipped in that module), also horizontally uniform, so the domain
  has real vertical shear and storms have something to organise against;
* a 0.1 K boundary-layer theta perturbation to break the symmetry, plus
  3 K warm bubbles -- WK82's own trigger -- so deep convection initiates on
  a schedule a two-hour forecast can show.  NOT ``harness``'s seeding
  amplitudes; see :data:`NOISE_AMPLITUDES` for the measurement that made
  that a deliberate departure;
* a REAL Lambert conformal grid -- map factors, Coriolis and per-column
  latitude/longitude that every solar-zenith path in the tree reads -- and a
  latitude/longitude-anchored terrain ridge;
* full physics: Morrison two-moment, MYNN surface layer + PBL, Noah-MP,
  RTE+RRTMGP and Smagorinsky ``km_opt=4`` -- :data:`RUNG`, the top rung of
  the bit-exact matrix.  Its Kain-Fritsch cumulus scheme is the one piece a
  convection-allowing run should NOT have, and ``--no-cumulus`` turns it
  off: MEASURED at 192^2, dx = 3 km, WK82, t+90 min, KF produces
  accumulated precipitation over 93.7% of the domain (mean 2.75 mm) against
  21.5% for the resolved storms (mean 0.80 mm).  A scheme whose job is
  convection the grid cannot resolve, firing in nearly every column of a
  grid that resolves it, is a scale error and it dominates the picture.
  Turning it off REMOVES 16 carriers (229 -> 213) and 48.3 B/cell, so the
  resident-ceiling claim must be re-measured at whichever of the two is
  actually run -- ``run_bigdomain ceiling`` takes the same flag for exactly
  that reason.

What that is NOT is a weather map.  A horizontally uniform sounding has no
fronts, no jet, no dryline and no synoptic pattern, and nothing here should
be described as a forecast of any real day.  What it does have is real
convective overturning on a real projection, and that is what the figures
claim.

THE LATERAL BOUNDARY, MEASURED RATHER THAN CHOSEN
-------------------------------------------------
The brief for this work asked for specified boundaries.  Measured on this
tree, they are not available to the streaming path yet, and the reason is
structural rather than a missing switch:

``specified``
    ``dycore.step`` refuses outright -- ``cfg.specified=True requires
    attach_lateral_boundaries(state, ...)``.  Those tables are attached to
    the STATE and appear in neither
    :func:`tilestream.physics_inventory.carrier_manifest` nor
    :func:`tilestream.driver.geography_inventory`, so a tile holding a true
    domain edge has no way to receive its window of the boundary frame.
    Wiring them is a real piece of work, not a keyword.
``open`` (radiative)
    needs no table, but ``h_sca_adv_order=5`` is explicitly unwired for it
    (``dycore.py:405``), so it costs the 5th-order geopotential advection,
    and no gate row has ever tiled a non-periodic domain.
``periodic``
    what all 14 physics rungs of ``test_gate`` certify bit-exact, and what
    this module runs.  Its cost is accurate and worth stating: on a real
    Lambert grid the index wrap is a GEOGRAPHIC discontinuity -- latitude,
    Coriolis and the solar zenith all jump across it -- so the outermost few
    rows and columns are not physical and the figures crop them.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np


#: ``full + MYNN + Noah-MP``: the top rung of the bit-exact matrix, copied
#: from ``test_gate.PHYSICS_RUNGS`` so this module runs where the gate does
#: not.  ``ztop=20000`` is essential -- at an 8 km top RRTMGP pads to 140
#: layers against its own limit of 128 and raises.
RUNG: dict = dict(
    moist=True, mp_physics=10, ztop=20000.0,
    km_opt=4, sf_sfclay_physics=5, bl_pbl_physics=5, bldt=0.0,
    sf_surface_physics=4, ra_sw_physics=4, ra_lw_physics=4,
    radt_minutes=12.0, cu_physics=1, cudt_minutes=5.0)

#: THE RUNG HAS NO DAMPING, AND A FORECAST NEEDS SOME.  Found the hard way.
#:
#: ``RunConfig`` defaults ``w_damping = 0`` (config.py:84) and
#: ``damp_opt = 0`` (:48), and :data:`RUNG` -- copied from the bit-exact
#: matrix -- overrides neither, because the gate runs 1 to 8 steps and a
#: state that has been stepped eight times has no convection in it.  A
#: FORECAST does.  MEASURED, 1440^2 at dx = 3 km, dt = 15 s: the run reached
#: t+45 min cleanly with 346 cells above 30 m/s and a peak column-max w of
#: 44.8 m/s, and then died between t+45 and t+60 with
#:
#:     ValueError: MYNN mass-flux inputs must be finite
#:      (mynn_pbl_gpu.py:1148, via dycore.step -> physics.compute)
#:
#: i.e. a non-finite value reached the PBL scheme.  With 49 levels to a
#: 20 km top the mid-tropospheric layer is roughly 400 m, so 44.8 m/s at
#: dt = 15 s is a VERTICAL Courant number near 1.7 -- exactly the regime
#: WRF's ``w_damping = 1`` exists to hold, and it was off.
#:
#: The failure is in the CONFIGURATION, not in the streaming path: the state
#: at t+45 is finite everywhere, the tiled loop is bit-exact against a
#: monolithic run of this same configuration, and nothing about the geometry
#: of the tiles enters the vertical Courant number.  Before this rung is used
#: for a multi-hour forecast it wants ``w_damping=1`` and a
#: ``damp_opt=3``/``zdamp``/``dampcoef`` upper damping layer -- and no
#: bit-exactness gate will ever ask for them, which is the point worth
#: recording.
DAMPING_NOTE = ("w_damping=0 and damp_opt=0 are RunConfig defaults and this "
                "rung does not override them; a run that develops 45 m/s "
                "updraughts at dt=15 s needs both")

#: Convection-allowing spacing and the WRF rule-of-thumb time step for it
#: (``dt <= 6 * dx_km``).  ``harness.GEOGRAPHY_OVERRIDES`` carries 12 km,
#: which at these domain sizes would be 16 900 km across -- wider than the
#: Earth, and the Lambert grid would run off the pole.
DX = 3000.0
DT = 15.0

#: Nominal forecast start.  Late April, 18 UTC: mid-afternoon over the
#: central United States, so the solar zenith the radiation and Noah-MP
#: paths compute per column is a daytime one and the surface fluxes are
#: doing something.  Same instant ``physics_inventory.default_builder`` uses.
START_TIME = (2011, 4, 27, 18, 0, 0)

#: WK82's own thermal amplitude and depth.  The HORIZONTAL radius is widened
#: from the case's 10 km because 10 km spans three cells at ``DX`` and an
#: unresolved bubble is a grid-scale spike, not a thermal.
BUBBLE_DELT = 3.0
BUBBLE_RADIUS = 30000.0
BUBBLE_ZC = 1500.0

#: Perturbation amplitudes for a FORECAST, which are not the harness's.
#:
#: ``harness._SEED_FIELDS`` seeds 0.5 K of white noise in theta, 20% white
#: noise in vapour and 20 Pa in the column mass, at EVERY cell.  Those
#: amplitudes exist to make a bit-exactness gate sensitive -- a hash test
#: wants every field excited -- and they are actively wrong for a forecast on
#: an unstable sounding.  MEASURED at 192^2, dx = 3 km, WK82: with the gate
#: amplitudes the composite reflectivity field at t+15 min is salt-and-pepper
#: at the GRID SCALE, isolated cells at 55-65 dBZ with no coherent
#: structure, because 20% white noise in qv saturates individual columns and
#: 0.5 K white noise gives each of them its own LFC.  It is noise being
#: amplified by real convective instability, and it looks like a broken
#: model.
#:
#: So the noise here does one job -- break the symmetry of a horizontally
#: uniform sounding so the bubbles are not the only thing in the domain --
#: and the trigger is the WK82 thermal, as in the case this sounding comes
#: from.  Theta only, 0.1 K, and tapered out above :data:`NOISE_DEPTH_M` so
#: it perturbs the boundary layer rather than the free troposphere.
NOISE_AMPLITUDES: dict = dict(u=0.0, v=0.0, w=0.0, thp=0.1, php=0.0,
                              mup=0.0, qv_rel=0.0)

#: Depth over which the theta perturbation is applied, with a cosine taper to
#: zero at the top.  Boundary-layer noise is what an idealised WRF run
#: perturbs; noise at 10 km has nothing to do and shows up as gravity waves.
NOISE_DEPTH_M = 2000.0

#: The gate's amplitudes, kept so the difference stays measurable rather than
#: asserted.  ``--perturbation gate`` reproduces the salt-and-pepper result
#: above, which is the control for the paragraph on :data:`NOISE_AMPLITUDES`.
GATE_AMPLITUDES: dict = dict(u=0.05, v=0.05, w=0.05, thp=0.5, php=20.0,
                             mup=20.0, qv_rel=0.20)


def big_config(nx: int, ny: int | None = None, nz: int = 49, *,
               dx: float = DX, dt: float = DT, **overrides):
    """The forecast ``RunConfig``: full physics, real Lambert, periodic.

    ``map_proj=1`` and ``terrain_opt=1`` come from
    ``harness.GEOGRAPHY_OVERRIDES``; the 12 km spacing that ships with them
    does not (see :data:`DX`).  ``terrain_opt=1`` is the switch that makes
    ``thb/pb/alb/phb`` three-dimensional, i.e. the one that moves 16.7 of the
    17.1 B/mass-cell of geography a tile has to gather.
    """
    from tilestream import harness

    over = dict(RUNG)
    over.update(harness.GEOGRAPHY_OVERRIDES)
    over["dx"] = over["dy"] = float(dx)
    over["dt"] = float(dt)
    over.update(overrides)
    return harness.make_config(int(nx), int(ny if ny is not None else nx),
                               int(nz), **over)


def start_datetime():
    from datetime import datetime, timezone
    return datetime(*START_TIME, tzinfo=timezone.utc)


# --------------------------------------------------------------------------
# the initial condition
# --------------------------------------------------------------------------

@dataclass
class SeededNoise:
    """Full-domain random perturbations, drawn once so slabs can window them.

    Held as float32 because that is the dtype they land in; the draw is
    float64 first, exactly as ``harness.make_state``'s ``fill`` does, so the
    rounding is the same one.

    The two duplicate rules are not cosmetic.  Under a periodic domain
    ``spec.TileSpec._axis_gather`` reduces every window mod ``nx`` and never
    reads a staggered array's alias slot, so a tile's u-face at the wrap
    takes column 0 while a monolithic run reads column ``nx``.  The domain is
    self-consistent only when the alias duplicates the opposite face, which
    is why ``u[..., -1] = u[..., 0]`` and ``v[:, -1, :] = v[:, 0, :]`` are
    applied to the noise before anything reads it -- the same duplicate
    ``woof/verify/npref.py`` enforces.
    """

    u: np.ndarray
    v: np.ndarray
    w: np.ndarray
    thp: np.ndarray
    php: np.ndarray
    mup: np.ndarray
    qv: np.ndarray

    @staticmethod
    def draw(nz: int, ny: int, nx: int, seed: int = 20_260_808) -> "SeededNoise":
        rng = np.random.default_rng(int(seed))

        def draw(shape, xdup=False, ydup=False):
            vals = rng.standard_normal(shape)
            if xdup:
                vals[..., -1] = vals[..., 0]
            if ydup:
                vals[..., -1, :] = vals[..., 0, :]
            return np.ascontiguousarray(vals, dtype=np.float32)

        return SeededNoise(
            u=draw((nz, ny, nx + 1), xdup=True),
            v=draw((nz, ny + 1, nx), ydup=True),
            w=draw((nz + 1, ny, nx)),
            thp=draw((nz, ny, nx)),
            php=draw((nz + 1, ny, nx)),
            mup=draw((ny, nx)),
            qv=draw((nz, ny, nx)))

    @property
    def nbytes(self) -> int:
        return sum(getattr(self, f).nbytes for f in
                   ("u", "v", "w", "thp", "php", "mup", "qv"))


def bubble_centres(nx: int, ny: int, dx: float, *, count: int,
                   seed: int = 8_080_808, edge_cells: int = 64):
    """``(x, y)`` metres of ``count`` warm bubbles, random but reproducible.

    Kept ``edge_cells`` away from every boundary.  On a periodic domain the
    wrap is a geographic discontinuity (module docstring), so a storm
    initiated on top of it would be a storm initiated on an artefact, and the
    figures crop that margin anyway.
    """
    rng = np.random.default_rng(int(seed))
    lo, hi_x = edge_cells * dx, (nx - edge_cells) * dx
    hi_y = (ny - edge_cells) * dx
    return np.stack([rng.uniform(lo, hi_x, int(count)),
                     rng.uniform(lo, hi_y, int(count))], axis=1)


def window_geography(geo, j0: int, rows: int):
    """The ``rows`` rows of ``geo`` starting at ``j0``, staggering respected.

    ``msfv`` carries the closing face, so it takes ``rows + 1`` rows: slab
    k's last v-row IS slab k+1's first, and both are the same function of
    position, so the overlap is a duplicate rather than a conflict.
    """
    from tilestream.harness import Geography

    s = slice(j0, j0 + rows)
    return Geography(
        grid=None,
        msft=geo.msft[s].copy(), msfu=geo.msfu[s].copy(),
        msfv=geo.msfv[j0:j0 + rows + 1].copy(),
        f=geo.f[s].copy(), e=geo.e[s].copy(),
        sina=geo.sina[s].copy(), cosa=geo.cosa[s].copy(),
        lat=geo.lat[s].copy(), lon=geo.lon[s].copy(),
        terrain=None if geo.terrain is None else geo.terrain[s].copy())


def build_slab_state(cfg_slab, geo_slab, noise: SeededNoise, j0: int, *,
                     bubbles=None, dx: float = DX, amplitudes=None):
    """One row slab of the initial condition, as a live ``(state, driver)``.

    The construction order is ``harness.make_physics_state``'s, with the two
    slab-safety changes the module docstring names, and with WK82's
    hodograph and thermals added:

    1. WK82 theta as the base-state profile, on the SLAB's terrain, so
       ``ht``, ``mub2d`` and the 3-D ``thb/pb/alb/phb`` all follow the
       terrain;
    2. ``init_at_rest`` + the domain's map factors, Coriolis and rotation;
    3. the hodograph columns broadcast onto the staggered u/v faces, plus
       the windowed noise;
    4. ``update_diagnostics`` -- a purely vertical integration, which is why
       it is column-local and a slab may run it;
    5. vapour from the WK82 profile on the flat-terrain column heights, times
       ``1 + 0.2 * noise``, floored at 1e-9, then ``update_diagnostics``
       again;
    6. the warm bubbles, added to ``thp`` WITHOUT recomputing pressure --
       which is what WRF's own quarter_ss initializer does
       (``module_initialize_ideal.F``: the theta perturbation does not enter
       the pressure construction);
    7. ``initialize_physics`` on the slab's per-column latitude/longitude.

    Step 5 leaves the moist column DISCRETELY UNBALANCED -- vapour is added
    to a state whose pressure was built dry.  ``moist.init_moist_balanced``
    is the routine that fixes that, and it refuses a terrain-following base
    state outright ("flat base states only"), so with real terrain there is
    no balanced moist initialiser in this tree to call.  The consequence is
    real and shows up in the first frames as a domain-wide hydrostatic
    adjustment; it is the same imbalance every physics rung of the gate runs
    on, and it is stated here rather than discovered on a plot.
    """
    import cupy as cp

    from woof.core.diagnostics import update_diagnostics
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.physics import initialize_physics
    from woof.core.state import _height_half_from_phb, init_at_rest
    from woof.verify.cases.wk82 import hodograph_uv, wk82_sounding

    from tilestream import harness

    rows = int(cfg_slab.ny)
    coord = make_vertical_coord(cfg_slab.nz, hybrid_opt=cfg_slab.hybrid_opt,
                                etac=cfg_slab.etac)
    base = make_base_state(coord, lambda z: wk82_sounding(z)[0],
                           p_surf=cfg_slab.p_surf, ztop=cfg_slab.ztop,
                           terrain_z=geo_slab.terrain)
    state = init_at_rest(cfg_slab, coord, base)
    harness.install_geography(state, geo_slab)

    u_hod, v_hod = hodograph_uv(coord, cfg_slab.ztop)
    a = dict(NOISE_AMPLITUDES if amplitudes is None else amplitudes)
    sl = slice(j0, j0 + rows)
    slv = slice(j0, j0 + rows + 1)
    state.u[...] = cp.asarray(
        u_hod[:, None, None] + a["u"] * noise.u[:, sl], dtype=state.u.dtype)
    state.v[...] = cp.asarray(
        v_hod[:, None, None] + a["v"] * noise.v[:, slv], dtype=state.v.dtype)
    w = a["w"] * noise.w[:, sl].copy()
    w[0] = 0.0
    w[-1] = 0.0
    state.w[...] = cp.asarray(w, dtype=state.w.dtype)
    # The theta perturbation is tapered out above NOISE_DEPTH_M.  The taper
    # is evaluated on the FLAT column heights for the same reason the vapour
    # profile is (below): a terrain-following height is a different function
    # of k in every column and a slab cannot reproduce the domain's.
    flat = make_base_state(coord, lambda z: wk82_sounding(z)[0],
                           p_surf=cfg_slab.p_surf, ztop=cfg_slab.ztop,
                           terrain_z=None)
    z_col = np.asarray(_height_half_from_phb(np.asarray(flat.phb)),
                       dtype=np.float64)
    taper = np.where(z_col < NOISE_DEPTH_M,
                     np.cos(0.5 * np.pi * z_col / NOISE_DEPTH_M) ** 2, 0.0)
    state.thp[...] = cp.asarray(
        a["thp"] * taper[:, None, None] * noise.thp[:, sl],
        dtype=state.thp.dtype)
    state.php[...] = cp.asarray(a["php"] * noise.php[:, sl],
                                dtype=state.php.dtype)
    state.php[0] = 0.0                     # fixed surface: phi'(sfc) = 0
    state.mup[...] = cp.asarray(a["mup"] * noise.mup[sl],
                                dtype=state.mup.dtype)
    update_diagnostics(state)

    # Vapour on the FLAT-terrain column heights (``z_col`` above): a function
    # of nz/ztop alone, so every slab evaluates the same profile.  See the
    # module docstring for why the harness's domain-mean height cannot be
    # used here.  ``DomainState.height_half`` is the sanctioned reader for
    # that profile but it cannot serve: ``load_base`` refuses a 1-D (flat)
    # base state on a ``terrain_opt=1`` config, and building a second flat
    # state just to read a (nz,) profile would allocate a whole domain, so
    # this calls the same module-level function ``height_half`` itself calls.
    qv_col = wk82_sounding(z_col)[1]
    state.qv[...] = cp.asarray(
        np.maximum(qv_col[:, None, None]
                   * (1.0 + a["qv_rel"] * noise.qv[:, sl]), 1e-9),
        dtype=state.qv.dtype)
    update_diagnostics(state)

    if bubbles is not None and len(bubbles):
        z = np.asarray(state.height_half(), dtype=np.float64)
        if z.ndim == 1:
            z = np.broadcast_to(z[:, None, None],
                                (cfg_slab.nz, rows, cfg_slab.nx))
        xc = (np.arange(cfg_slab.nx) + 0.5) * dx
        yc = (np.arange(j0, j0 + rows) + 0.5) * dx
        zrad = (z - BUBBLE_ZC) / BUBBLE_ZC
        bump = np.zeros((cfg_slab.nz, rows, cfg_slab.nx), dtype=np.float64)
        for bx, by in bubbles:
            # Only the bubbles that reach this slab: the cos^2 form is
            # exactly zero outside the unit ellipsoid, so this is a skip,
            # not an approximation.
            if abs(by - yc[0]) > BUBBLE_RADIUS + rows * dx and \
                    abs(by - yc[-1]) > BUBBLE_RADIUS:
                continue
            xr = (xc[None, None, :] - bx) / BUBBLE_RADIUS
            yr = (yc[None, :, None] - by) / BUBBLE_RADIUS
            rad = np.sqrt(xr ** 2 + yr ** 2 + zrad ** 2)
            bump += np.where(rad <= 1.0,
                             BUBBLE_DELT * np.cos(0.5 * np.pi * rad) ** 2, 0.0)
        state.thp[...] = state.thp + cp.asarray(bump, dtype=state.thp.dtype)
        update_diagnostics(state)

    driver = initialize_physics(
        state, cfg_slab,
        radiation_start_time=start_datetime(),
        radiation_latitude=geo_slab.lat, radiation_longitude=geo_slab.lon,
        noahmp_start_time=start_datetime(),
        noahmp_latitude=geo_slab.lat, noahmp_longitude=geo_slab.lon)
    return state, driver


def _hold_domain_carrier(dst: np.ndarray, src, name: str, k: int) -> None:
    """Keep a domain-owned (fixed-shape) carrier whole; slabs must agree.

    ``global_keys`` carriers such as the 3D PWP status word
    ``fields/_pwp_status`` (shape (1,)) have no rows: scattering them raised
    "IndexError: tuple index out of range" at the start of a streamed
    forecast with sf_ocean_physics = 2 (the 48 h default;
    2026-10-21).  Slab 0's words are kept and a slab that built different
    words is refused.
    """
    host = np.asarray(src)
    if host.shape != dst.shape:
        raise ValueError(
            f"slab {k} built domain carrier {name!r} at shape {host.shape}, "
            f"not its fixed shape {dst.shape}")
    if k == 0:
        dst[...] = host
    elif not np.array_equal(dst.view(np.uint8),
                            np.ascontiguousarray(host).view(np.uint8)):
        raise ValueError(
            f"slab {k} built domain carrier {name!r} with words that differ "
            "from slab 0's; a domain-owned carrier must be slab-invariant")


def _scatter_rows(dst: np.ndarray, src, j0: int) -> None:
    """Write ``src``'s rows into ``dst`` at ``j0``, inferring the staggering.

    The horizontal axes are the LAST two, always -- that is the rule
    ``hoststore.manifest_from_arrays`` derives the whole store from -- so the
    destination row span is just the source's own second-to-last extent.  A
    v-staggered field therefore writes ``rows + 1`` rows and the overlap with
    the next slab is a duplicate of the same value.
    """
    rows = src.shape[-2]
    dst[..., j0:j0 + rows, :] = src


def carrier_manifest_for(cfg, *, probe_nx: int = 48, probe_ny: int = 40,
                         reflectivity: bool = True):
    """The shape rules for ALL the carriers, from a probe that has stepped.

    ``manifest_from_arrays`` read off a freshly built state is one name
    short.  Two carriers in this rung are allocated on first use --
    Kain-Fritsch's ``cumulus/w0avg`` (``kf.py:335``) is the one that bites --
    so a never-stepped state reports 228 members while
    ``make_physics_tile_state``, which steps once for exactly this reason,
    reports 229.  ``run_tiled`` compares the two key sets and refuses the
    run, which is the right behaviour and a confusing message: the store is
    not missing weather, it is missing an accumulator that has not been
    created yet.

    So the manifest is taken from a SMALL state that HAS stepped, and the
    slabs fill what they can.  ``probe_ny != probe_nx`` because
    ``manifest_from_arrays`` refuses a square probe -- on a square domain a
    y/x transposition in the shape rules would pass unnoticed.

    ``reflectivity=True`` (the default, audit R-052) puts ``refl_10cm`` in
    the manifest, and therefore in the store, and therefore in the gather
    and the scatter.  It is a REBUILT diagnostic that no restart manifest
    names, so without this the field each tile computes has nowhere to land
    and the domain never holds a copy -- and for mp=9, 18 and 50, whose dBZ
    is produced inside the scheme call and published through no operator, it
    is the only route to reflectivity there is.  The slot is PRIMED here
    rather than stepped into existence: :func:`woof.core.streaming
    .prime_refl_10cm` allocates it and integrates nothing, which is the
    published way to make a manifest name it.  The store it sizes starts
    empty and is filled by the first sweep that carries ``refl_10cm_due``
    (:func:`tilestream.driver.reflectivity_run_kwargs`), which is why
    :func:`snapshot` has to be told whether that sweep has happened.
    """
    import cupy as cp

    from tilestream import harness, hoststore
    from tilestream import physics_inventory as physinv

    probe = _replace_nxny(cfg, int(probe_nx), int(probe_ny))
    state, drv = build_slab_state(probe, harness.make_geography(probe),
                                  SeededNoise.draw(int(cfg.nz), int(probe_ny),
                                                   int(probe_nx)),
                                  0, bubbles=None, dx=float(cfg.dx))
    harness.run_steps(state, probe, 1)
    if reflectivity:
        from woof.core.streaming import prime_refl_10cm
        prime_refl_10cm(state, probe)
    manifest = hoststore.manifest_from_arrays(
        physinv.carrier_inventory_with_refl(state) if reflectivity
        else physinv.carrier_inventory(state), int(cfg.nz), int(probe_ny),
        int(probe_nx))
    del state, drv
    cp.get_default_memory_pool().free_all_blocks()
    return manifest


def _replace_nxny(cfg, nx: int, ny: int):
    from dataclasses import replace
    return replace(cfg, nx=int(nx), ny=int(ny))


def build_store_by_slabs(cfg, geo, *, slab_rows: int, noise: SeededNoise,
                         bubbles=None, store=None, log=print,
                         manifest=None, amplitudes=None,
                         budget_bytes: int | None = None,
                         inventory_fn=None, prepare=None):
    """The full-domain initial store, built one slab at a time.

    Returns ``(store, geo_store, scalars, missing)``.  ``store`` and
    ``geo_store`` are dicts of pinned full-domain host arrays keyed exactly
    as :func:`tilestream.physics_inventory.carrier_inventory` and
    :func:`tilestream.driver.geography_inventory` key them, which is what
    ``run_tiled`` requires of both.  ``missing`` names the carriers no slab
    produced.

    Ranked callers may pass their full streamed inventory and its lazy
    carrier priming function.  The default retains the original tile-stream
    inventory; priming allocates carriers without advancing the atmosphere.

    ``missing`` is not an error case to be silently tolerated: two carriers
    in this rung are allocated LAZILY on first use (Kain-Fritsch's
    ``cumulus/w0avg`` above all, ``kf.py:335``), so a state that has never
    stepped has a shorter inventory than a tile buffer, which
    ``make_physics_tile_state`` steps once precisely to avoid.  Those are
    filled with ZEROS here, which is not a convenience: ``kf.py:335``
    allocates ``cp.zeros_like`` and the very next line forms
    ``(w0avg * (tst - 1) + instantaneous) / tst``, so zero is exactly the
    value a fresh monolithic run carries into its first cumulus call.  Any
    name that shows up in ``missing`` for another reason is a real hole and
    the caller is told about it rather than left to find it in the weather.
    """
    import cupy as cp

    from tilestream import hoststore
    from tilestream import physics_inventory as physinv
    from tilestream import driver as tdriver

    nz, ny, nx = int(cfg.nz), int(cfg.ny), int(cfg.nx)
    if ny % int(slab_rows):
        raise ValueError(f"slab_rows={slab_rows} must divide ny={ny}")
    nslabs = ny // int(slab_rows)

    store = {} if store is None else store
    geo_store: dict = {}
    scalars: dict = {}
    allocated = False
    seen: set = set()

    for k in range(nslabs):
        j0 = k * int(slab_rows)
        t0 = time.perf_counter()
        cfg_slab = _replace_ny(cfg, int(slab_rows))
        geo_slab = window_geography(geo, j0, int(slab_rows))
        state, drv = build_slab_state(cfg_slab, geo_slab, noise, j0,
                                      bubbles=bubbles, dx=float(cfg.dx),
                                      amplitudes=amplitudes)
        if prepare is not None:
            prepare(state, cfg_slab)
        inv = (inventory_fn or physinv.carrier_inventory)(state)
        ginv = tdriver.geography_inventory(state)
        if not allocated:
            allocated = True
            if manifest is None:
                manifest = hoststore.manifest_from_arrays(
                    inv, nz, int(slab_rows), nx)
            geo_shapes = {
                name: (tuple(arr.shape[:-2])
                       + (ny + (int(arr.shape[-2]) - int(slab_rows)),
                          int(arr.shape[-1])))
                for name, arr in ginv.items()}
            # THE WHOLE REQUEST, BEFORE THE FIRST BYTE OF IT IS TAKEN.
            # hoststore.check_allocatable is the fail-closed budget refusal
            # the class-based store path has always called (hoststore.py:840);
            # this builder is the newer road and it never did, so it took
            # pinned memory a slab at a time with nothing between it and the
            # machine.  MEASURED: a real store reached 87.8 GiB of page-locked
            # RAM in silence, past the documented ceiling, and starved every
            # other lane on the box -- pinned pages cannot be swapped, so this
            # is the failure mode that takes a machine down rather than a run.
            # Priced from the manifest rather than observed while allocating,
            # because a guard that fires on the way up has already taken most
            # of what it is refusing.
            planned = sum(spec.nbytes(nz, ny, nx) for spec in manifest)
            planned += sum(int(np.prod(shape))
                           * np.dtype(ginv[name].dtype).itemsize
                           for name, shape in geo_shapes.items())
            hoststore.check_allocatable(planned, budget_bytes=budget_bytes)
            log(f"store: {planned / hoststore.GIB:.2f} GiB pinned "
                f"({len(manifest)} carriers + {len(geo_shapes)} geography), "
                "budget checked")
            for spec in manifest:
                store[spec.name] = hoststore.alloc_pinned_array(
                    spec.shape(nz, ny, nx), spec.dtype)
            domain_carriers = {spec.name for spec in manifest
                               if spec.fixed_shape is not None}
            for name, shape in geo_shapes.items():
                geo_store[name] = hoststore.alloc_pinned_array(
                    shape, np.dtype(ginv[name].dtype))
            scalars = physinv.carrier_scalars(state)
        for name, arr in inv.items():
            if name not in store:
                raise KeyError(
                    f"slab {k} produced carrier {name!r} that slab 0 did "
                    "not; the manifest is not slab-invariant")
            if name in domain_carriers:
                _hold_domain_carrier(store[name], cp.asnumpy(arr), name, k)
            else:
                _scatter_rows(store[name], cp.asnumpy(arr), j0)
            seen.add(name)
        for name, arr in ginv.items():
            _scatter_rows(geo_store[name], _as_host(arr), j0)
        flags = tdriver.geography_scalars(ginv) if k == 0 else None
        del state, drv, inv, ginv
        cp.get_default_memory_pool().free_all_blocks()
        log(f"    slab {k + 1}/{nslabs} rows {j0}..{j0 + slab_rows} "
            f"in {time.perf_counter() - t0:.1f}s")
        _ = flags

    missing = sorted(set(store) - seen)
    return store, geo_store, scalars, missing


def _as_host(arr) -> np.ndarray:
    import cupy as cp
    return cp.asnumpy(arr) if isinstance(arr, cp.ndarray) else np.asarray(arr)


def _replace_ny(cfg, ny: int):
    from dataclasses import replace
    return replace(cfg, ny=int(ny))


# --------------------------------------------------------------------------
# products
# --------------------------------------------------------------------------

#: What comes straight out of the carrier store as a 2-D field, and the units
#: to put on a colour bar.  Every one of these is a MODEL-CARRIED array, not
#: a re-derivation: ``fields/*`` are the surface/PBL scheme outputs and
#: ``scratch/mp_rainnc``/``scratch/cu_rainc`` are WRF's own accumulation
#: buckets (the grid-scale and convective halves of total precipitation).
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
    "COSZEN": ("fields/coszen", "1"),
    "TSK": ("fields/tsk", "K"),
}


def composite_reflectivity(store, geo_store, cfg, *, slab_rows: int = 64,
                           refl_stash: str = "absent"):
    """Column-maximum REFL_10CM (dBZ), computed out of core, WOOF's own way.

    ``woof.core.refl.compute_refl_10cm`` is the model's diagnostic and this
    calls it -- it is not re-derived here.  What is arranged here is only
    that it can be called at all on a domain that does not fit: reflectivity
    is COLUMN-LOCAL (Morrison's ``refl10cm_hm`` walks one column of
    ``qr/nr/qs/ns/qg/ng`` against ``t``/``p`` and nothing else), so a slab
    with no halo whatsoever gives exactly the answer the whole domain would.
    That is why this can use bare row slabs while the stepping loop needs a
    16-cell halo.

    It is the STANDALONE-current-state path, not the microphysics-time one:
    ``compute_refl_10cm`` is handed no explicit ``temperature``/``pressure``
    and so forms ``T = (thb + thp) * (p / p0)^(Rd/cp)`` from the state in
    hand.  Production history output passes the scheme's post-call T and the
    prepared p instead (PROVENANCE.md D2).  The difference is one
    microphysics call's worth of temperature and it is named here so nobody
    reports this as the history-step field.

    The column max is the standard composite: ``woof/render.py`` takes
    exactly ``REFL_10CM.max(axis=vertical)`` for its composite product.

    THE SCHEME'S OWN FIELD WINS, exactly as it does for the streamed real
    case -- and ``refl_stash`` is what says there is one to win with.  This
    lane's store carries ``scratch/refl_10cm`` from the manifest onwards
    (:func:`carrier_manifest_for`), but it starts EMPTY: no slab computes
    reflectivity while the store is being built, and the slot holds a field
    only once a sweep carrying ``refl_10cm_due`` has written every tile's
    window into it.  So the caller says which it has:

    ``"computed"``
        a sweep configured by :func:`tilestream.driver
        .reflectivity_run_kwargs` has run, and that array IS the model's
        reflectivity -- the only route for mp=9, 18 and 50, whose dBZ is
        produced inside the scheme call and published through no operator,
        and the only correct route for mp=8 and 28, whose operator branch
        needs a graupel-number shadow that is rebuilt and never serialised.
    ``"absent"`` (the default) or ``"primed"``
        the slab recompute below, because zeros rendered as dBZ are a
        domain-wide 0 dBZ field that reads like weak echo rather than like
        a field nothing has written.

    The species the recompute reads come from
    ``woof.core.refl.REFL_10CM_INPUT_SPECIES`` -- audit R-052: the tuple
    that used to be typed here was Morrison's six-moment set, duplicated
    from the real-case lane, so every other scheme was refused for lacking
    a moment its reflectivity formulation never reads.
    """
    import cupy as cp

    from woof.core.refl import (NativeReflectivityScheme,
                                 compute_refl_10cm, refl_10cm_input_species)
    from woof.core.state import init_at_rest
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.verify.cases.wk82 import wk82_sounding
    from tilestream.physics_inventory import REFL_KEY

    nz, ny, nx = int(cfg.nz), int(cfg.ny), int(cfg.nx)
    if ny % int(slab_rows):
        raise ValueError(f"slab_rows={slab_rows} must divide ny={ny}")
    if refl_stash == "computed":
        stashed = store.get(REFL_KEY)
        if stashed is None:
            raise KeyError(
                f"this store is said to carry a computed reflectivity and "
                f"has no {REFL_KEY!r}; carrier_manifest_for(reflectivity="
                "True) is what puts the slot in the manifest")
        return np.asarray(stashed, dtype=np.float32).max(axis=0)
    out = np.empty((ny, nx), dtype=np.float32)

    coord = make_vertical_coord(nz, hybrid_opt=cfg.hybrid_opt, etac=cfg.etac)
    cfg_slab = _replace_ny(cfg, int(slab_rows))
    try:
        species = tuple(refl_10cm_input_species(int(cfg.mp_physics))
                        + ("p", "thp"))
    except NativeReflectivityScheme as exc:
        raise KeyError(
            f"{exc}  This store's {REFL_KEY!r} is {refl_stash!r}: sweep it "
            "with the keywords of tilestream.driver.reflectivity_run_kwargs "
            "-- which make every tile write the slot -- and pass "
            "refl_stash='computed' here, or render a scheme whose "
            "reflectivity the operator computes.") from exc

    for j0 in range(0, ny, int(slab_rows)):
        sl = slice(j0, j0 + int(slab_rows))
        terrain = geo_store["setup/ht"][sl]
        base = make_base_state(coord, lambda z: wk82_sounding(z)[0],
                               p_surf=cfg_slab.p_surf, ztop=cfg_slab.ztop,
                               terrain_z=np.asarray(terrain, dtype=np.float64))
        state = init_at_rest(cfg_slab, coord, base)
        # thb is GEOGRAPHY on a terrain-following grid (3-D), so it is taken
        # from the domain's gathered store rather than from the slab's own
        # rebuild -- the same rule the stepping loop follows.
        state.thb[...] = cp.asarray(geo_store["setup/thb"][:, sl],
                                    dtype=state.thb.dtype)
        for name in species:
            arr = getattr(state, name, None)
            if arr is None:
                raise KeyError(f"reflectivity needs state.{name}, absent "
                               f"at mp_physics={cfg.mp_physics}")
            arr[...] = cp.asarray(store[f"state/{name}"][:, sl],
                                  dtype=arr.dtype)
        refl = compute_refl_10cm(state, cfg)
        out[sl] = cp.asnumpy(cp.max(refl, axis=0))
        del state, refl
        cp.get_default_memory_pool().free_all_blocks()
    return out


def snapshot(store, geo_store, cfg, *, elapsed_s: float, refl: bool = True,
             slab_rows: int = 64, refl_stash: str = "absent") -> dict:
    """Everything one figure needs, as plain host arrays.

    Column-max ``w`` is in here because it is the single most accurate test of
    whether a spin-up is doing anything: it is a prognostic the model
    integrates, it is exactly zero in a state that is not moving, and no
    colour scale can make it look like convection when it is 0.2 m/s.
    """
    out: dict = {"elapsed_s": float(elapsed_s),
                 "nx": int(cfg.nx), "ny": int(cfg.ny), "nz": int(cfg.nz),
                 "dx": float(cfg.dx), "dt": float(cfg.dt)}
    for label, (key, units) in SURFACE_CARRIERS.items():
        if key in store:
            out[label] = np.asarray(store[key], dtype=np.float32).copy()
            out[label + "_units"] = units
    w = np.asarray(store["state/w"], dtype=np.float32)
    out["WMAX"] = w.max(axis=0)
    out["WMIN"] = w.min(axis=0)
    out["WMAX_units"] = out["WMIN_units"] = "m s-1"
    out["HT"] = np.asarray(geo_store["setup/ht"], dtype=np.float32).copy()
    out["HT_units"] = "m"
    for name in ("radiation/latitude_deg", "radiation/longitude_deg"):
        if name in geo_store:
            out[name.split("/")[1][:3].upper()] = np.asarray(
                geo_store[name], dtype=np.float32).copy()
    if refl:
        out["REFL_COMPOSITE"] = composite_reflectivity(
            store, geo_store, cfg, slab_rows=slab_rows,
            refl_stash=refl_stash)
        out["REFL_COMPOSITE_units"] = "dBZ"
    return out


# --------------------------------------------------------------------------
# selftest
# --------------------------------------------------------------------------

def selftest_slab_build_matches_monolithic(nx: int = 128, ny: int = 96,
                                           nz: int = 49, slab_rows: int = 32,
                                           ) -> dict:
    """Does the slab build produce the domain a one-shot build would?

    The whole out-of-core initialisation rests on every construction step
    being column-local.  This runs the SAME :func:`build_slab_state` at the
    full domain (one slab covering everything) and in ``ny // slab_rows``
    pieces, and compares per-carrier digests.  A mismatch names the fields,
    which is what tells you WHICH step is not column-local rather than only
    that one is not.
    """
    import cupy as cp

    from tilestream import physics_inventory as physinv

    cfg = big_config(nx, ny, nz)
    from tilestream import harness
    geo = harness.make_geography(cfg)
    noise = SeededNoise.draw(nz, ny, nx)
    bubbles = bubble_centres(nx, ny, float(cfg.dx), count=3, edge_cells=8)

    state, drv = build_slab_state(cfg, geo, noise, 0, bubbles=bubbles,
                                 dx=float(cfg.dx))
    whole = physinv.field_digests(physinv.carrier_inventory(state))
    del state, drv
    cp.get_default_memory_pool().free_all_blocks()

    store, geo_store, _scal, missing = build_store_by_slabs(
        cfg, geo, slab_rows=slab_rows, noise=noise, bubbles=bubbles,
        log=lambda *_: None, manifest=None)
    piecewise = physinv.field_digests(store)
    differ = sorted(k for k in whole if whole[k] != piecewise.get(k))
    return {"carriers": len(whole), "differ": differ, "missing": missing,
            "agree": not differ}
