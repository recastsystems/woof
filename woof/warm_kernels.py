"""`woof warm-kernels`: pay the first run's kernel compile up front.

A fresh runtime's first forecast compiles its GPU kernels for the local
card before model step 1 can finish -- measured 2026-09-24 at about two
minutes and 157 kernel cache entries on an RTX 5070 Ti -- and the
screen says little while it happens.  CuPy keeps what it compiles in its
on-disk kernel cache (:func:`woof.kernel_compile_notice.
cupy_kernel_cache_dir`), so the cost is paid once per card.  This door
pays it on purpose, at a time the user chooses, and reports what it did:
how many kernels were compiled into the cache and how long it took.

HOW THE SET IS CHOSEN.  The kernels a forecast compiles depend on its
physics, so the door does not keep a list of them.  It runs the forecast
code itself: for each physics profile it builds a small synthetic domain
(a flat, moist, resting column over land), attaches that profile's
physics through the same :func:`woof.core.physics.initialize_physics`
a forecast calls, and advances it two model steps with the production
:func:`woof.core.dycore.step`.  Whatever that compiles is what a
forecast with that physics compiles.  By default the profiles are the
ones the sources default to (:func:`woof.physics_menu.
default_profile_for`), so a plain first forecast finds its kernels
cached; ``--profile`` names others and ``--all-profiles`` takes every
shipped suite.

EVERY CORE (2026-10-16).  Each profile's two steps compile their kernels one
after another on one thread, so the pass used to hold one core for minutes
(``--all-profiles --levels 49`` warming an image on an RTX PRO 6000 box sat
at 100 % of one core for 186 s and more).  The profiles now run side by
side, one spawned worker process per profile, as many at once as there are
usable cores and card memory for (:func:`profile_workers`;
``WOOF_WARM_KERNELS_WORKERS`` caps it).  Each worker compiles exactly what
the serial pass compiled for its profile, through the same code, into the
same caches under the same keys.  Each worker also binds a
:mod:`woof.kernel_plan` for its profile, so a later warm of a changed tree
compiles that profile's recorded kernel set on a pool as well.  One profile,
or no card answer, keeps the single in-process pass.

What it does not reach: kernels only a real case uses (terrain, lateral
boundaries, output diagnostics) still compile on the first real run, and
a run at a vertical level count above a kernel's compiled ceiling adds
its own tier.  Those are seconds, not minutes.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import fields
from datetime import datetime

import numpy as np

WARM_KERNELS_SCHEMA = "gpuwm.warm-kernels.v1"

#: Vertical levels of the synthetic domain.  Below every kernel's
#: compiled level ceiling, so the modules compiled are the ones a
#: forecast at an ordinary level count loads.
DEFAULT_LEVELS = 40

#: Horizontal size and spacing of the synthetic domain.  Small enough to
#: cost nothing but compile time.
_NX = _NY = 16
_DX_M = 3000.0
_ZTOP_M = 16000.0
_DT_S = 12.0

#: Two steps: the first runs radiation and the surface schemes on their
#: first call, the second the path every later step takes.
_STEPS = 2

#: The synthetic column's valid time and location only have to put the
#: sun above the horizon so the shortwave kernels run.
_VALID_TIME = datetime(2020, 6, 21, 12)
_LATITUDE = 30.0
_LONGITUDE = 0.0


def default_profiles() -> tuple[str, ...]:
    """Every profile some source defaults to, in first-seen order.

    At every spacing: a source's own default, then each grid-spacing row
    it admits (:data:`woof.physics_menu.SPACING_DEFAULTS`), so a plain
    sub-km first forecast finds its kernels cached as a coarser one does.
    """

    from woof import physics_menu

    seen: list[str] = []
    for source in physics_menu.registered_sources():
        profiles = [physics_menu.default_profile_for(source)]
        profiles += [row["profile_id"] for row in
                     physics_menu.spacing_default_rows(source) if row["admitted"]]
        for profile in profiles:
            if profile is not None and profile not in seen:
                seen.append(profile)
    return tuple(seen)


#: The synthetic column's temperature profile: (base height m, lapse rate
#: K/m) from a 300 K surface, a 6.5 K/km troposphere to a 210 K tropopause,
#: then the standard atmosphere's isothermal and warming stratosphere and
#: an isothermal layer above 47 km.  Every layer of every model top a run
#: can ask for stays inside the radiation tables' temperature range
#: (RRTMG and RRTMGP refuse a layer below 160 K).
_SURFACE_T_K = 300.0
_TROPOPAUSE_T_K = 210.0
_LAPSE_LAYERS = ((0.0, 0.0065),
                 ((_SURFACE_T_K - _TROPOPAUSE_T_K) / 0.0065, 0.0),
                 (20000.0, -0.0010),
                 (32000.0, -0.0028),
                 (47000.0, 0.0))


def _temperature(z):
    """Temperature (K) of the synthetic column at height ``z`` (m)."""

    z = np.asarray(z, dtype=np.float64)
    temperature = np.full(z.shape, _SURFACE_T_K)
    base_t = _SURFACE_T_K
    for index, (base, lapse) in enumerate(_LAPSE_LAYERS):
        top = (_LAPSE_LAYERS[index + 1][0] if index + 1 < len(_LAPSE_LAYERS)
               else np.inf)
        inside = z >= base
        temperature = np.where(inside, base_t - lapse * (np.minimum(z, top) - base),
                               temperature)
        if np.isfinite(top):
            base_t -= lapse * (top - base)
    return temperature


def _theta(z):
    """Potential temperature of the synthetic column (K, referenced to P0).

    Built from :func:`_temperature`: theta = T exp(int_0^z g / (cp T) dz'),
    the hydrostatic Exner relation for a 1000 hPa surface, so the column's
    layer temperatures are the profile's whatever the run's model top.
    Breakage (2.9.4 landing, V58 2026-10-20): the earlier constant
    N = 0.01/s theta profile cooled without limit aloft, so the column of
    an Earth-2 FCN3 nest (ztop 20 km, p_top 50 hPa) had a 159.2 K top
    layer, below RRTMG's 160 K floor, and the pre-warm failed.
    """

    from woof.core import constants as c

    z = np.asarray(z, dtype=np.float64)
    top = float(np.max(z, initial=0.0))
    fine = np.append(np.arange(0.0, top, 10.0), top)
    integrand = c.G / (c.CP * _temperature(fine))
    exponent = np.concatenate(([0.0], np.cumsum(
        0.5 * (integrand[1:] + integrand[:-1]) * np.diff(fine))))
    return _temperature(z) * np.exp(np.interp(z, fine, exponent))


def _vapour(z):
    """A moist boundary layer thinning with height, kg/kg."""

    return 0.012 * np.exp(-np.asarray(z, dtype=np.float64) / 2500.0)


def synthetic_downward_longwave(cfg) -> tuple[float | None, str]:
    """The downward longwave the synthetic column is handed, and why.

    A profile with a longwave scheme writes GLW itself and is handed
    nothing.  A profile with none, whose land surface reads GLW or whose
    shortwave publishes it, is the configuration
    :func:`woof.core.physics.initialize_physics` refuses to invent a
    number for.  A real run of such a profile declares the constant with
    :data:`~woof.physics_compat.CONSTANT_DOWNWARD_LONGWAVE_ACK`, and the
    runtime then types :data:`~woof.physics_compat.DECLARED_CONSTANT_GLW_WM2`
    into that call (:func:`woof.runtime.declared_constant_glw`).  The
    column declares that same constant, so it compiles the kernels such a
    run compiles.  The refusal a real run meets without the
    acknowledgement is untouched: it lives at config load and in
    ``initialize_physics``, and this door hands that call a declaration.

    Returns ``(glw, sentence)``: ``glw`` is ``None`` where nothing is
    declared, and the sentence is what the report says about it.
    """

    from woof.config import radiation_scheme_ids
    from woof.physics_compat import (CONSTANT_DOWNWARD_LONGWAVE_ACK,
                                      DECLARED_CONSTANT_GLW_WM2,
                                      downward_longwave_disposition)

    longwave, shortwave = radiation_scheme_ids(cfg)
    kind, _consumer = downward_longwave_disposition(
        ra_lw_physics=longwave, ra_sw_physics=shortwave,
        sf_surface_physics=int(cfg.sf_surface_physics))
    if kind in ("consumed", "published"):
        return (float(DECLARED_CONSTANT_GLW_WM2),
                f"declared constant {DECLARED_CONSTANT_GLW_WM2:g} W m-2, as a "
                f"run of this suite declares it ({CONSTANT_DOWNWARD_LONGWAVE_ACK})")
    if kind == "scheme":
        return None, f"computed by ra_lw_physics={longwave}"
    return None, "not read by this suite"


def _profile_config(profile: str, levels: int):
    """The synthetic column's run configuration under ``profile``'s physics."""

    from woof.config import RunConfig, validate_run_config
    from woof.physics_compat import single_domain_runtime_switches

    names = {field.name for field in fields(RunConfig)}
    settings = {key: value for key, value
                in single_domain_runtime_switches(profile).items()
                if key in names}
    # The synthetic column is flat: the moist balanced builder takes flat
    # base states only.
    settings["terrain_opt"] = 0
    return validate_run_config(RunConfig(
        nx=_NX, ny=_NY, nz=int(levels), dx=_DX_M, dy=_DX_M,
        ztop=_ZTOP_M, dt=_DT_S, run_seconds=_DT_S * _STEPS,
        clock_dt=_DT_S, output_interval_s=_DT_S * _STEPS,
        open_x=False, open_y=False, **settings))


def _bind_profile_plan(profile: str, levels: int) -> bool:
    """Bind ``profile``'s kernel plan and start its compiles on every core.

    The recorded plan (an earlier warm of this profile on this compiler)
    and the static plan (:func:`woof.kernel_prewarm.submit_static`: the
    modules the profile's selectors launch, captured through their own
    loaders) both go to one pool as wide as the usable cores; the column's
    steps then find their modules built or being built.  The breakage this
    prevents: on a fresh box (no recorded plan) each profile compiled its
    kernels one after another inside its worker at 100 % of one core for
    60 to 120 s while the other cores sat idle (`woof warm-kernels` 173 s
    for 161 kernels, 3 workers, 48 cores, RTX PRO 6000 box, 2026-10-17).

    A worker warms several profiles one after another: each gets its own
    plan under its own tag (rebind), not the first profile's.  False when
    no plan could be bound; the profile still warms, serially.
    """

    from woof import kernel_plan, kernel_prewarm

    tag = f"warm-kernels {profile} levels={int(levels)}"
    try:
        planned = (kernel_plan.rebind(tag) if kernel_plan._state["installed"]
                   else kernel_plan.install(tag_text=tag))
        if planned:
            kernel_plan.start()
            try:
                kernel_prewarm.submit_static(
                    kernel_prewarm.single_domain_experiment(_profile_config(profile, levels)))
            except Exception:  # noqa: BLE001 - a refused profile is reported by _run_profile
                pass
            kernel_plan.close()
        return bool(planned)
    except Exception:  # noqa: BLE001 - the profile still warms without a plan
        return False


def forecast_column_config(run):
    """The synthetic column under a real run's own configuration.

    Every option of ``run`` is kept -- the physics selections and their
    generation/variant selectors (``thompson_version``, ``bl_mynn_version``,
    ``diff_6th_form``, ``upper_wind_limiter_form`` ...), the vertical levels
    and the numerics -- because those are what choose a kernel's source.
    Only the extents, the clock and what the flat moist builder needs
    change: a 16 x 16 flat column, two steps, no lateral forcing to read.
    The column is open on both axes, not periodic: a real run's specified
    or nested boundaries need lateral forcing the column does not have, and
    the periodic path is refused by the updraft-helicity diagnostic every
    HRRR-shape run carries (nwp_diagnostics = 1; core/uh_diag.py transcribes
    only WRF's non-periodic cal_helicity bounds).  Breakage (2.9.4 landing,
    V58 2026-10-20): the periodic column failed after compiling everything
    else, and the forecast's step 1 compiled 51 s cold with the card idle.

    A SPECIFIED (or nested) run warms on a specified column, forced by its
    own initial state (:func:`run_column`), wide enough for the run's
    boundary width.  That is the boundary geometry its step launches, and
    the one zadvect_implicit = 1 accepts: the open column was refused
    ("zadvect_implicit = 1 with radiative open boundaries is not wired")
    on the packaged hrrr_v4_gsd41 recipe (2026-10-21), so its
    implicit-advection, MYNN and Thompson-fork variants compiled in step 1.
    """

    from dataclasses import replace

    from woof.config import validate_run_config

    dt = float(run.dt)
    common = dict(terrain_opt=0, run_seconds=dt * _STEPS, clock_dt=dt,
                  output_interval_s=dt * _STEPS, open_x=False, open_y=False,
                  nested=False)
    # A geographic farm belongs to its real domain. The synthetic column
    # warms the background physics; its unrelated grid cannot locate that farm.
    if getattr(run, "wind_farm_scheme", "off") != "off":
        common.update(wind_farm_scheme="off", wind_farm_table="", wind_farms="")
    if bool(getattr(run, "specified", False) or getattr(run, "nested", False)):
        extent = max(_NX, 2 * int(run.spec_bdy_width) + 4)
        return validate_run_config(replace(
            run, nx=extent, ny=extent, specified=True, **common))
    return validate_run_config(replace(
        run, nx=_NX, ny=_NY, **{**common, "open_x": True, "open_y": True},
        specified=False))


def _run_profile(profile: str, levels: int) -> str:
    """Build the synthetic domain with ``profile``'s physics; step it.

    Returns the sentence :func:`synthetic_downward_longwave` gave for the
    downward longwave the column ran on.
    """

    return run_column(_profile_config(profile, levels))


def column_p_top(cfg) -> float:
    """The model-top pressure (Pa) :func:`run_column` builds for ``cfg``, on the host.

    The synthetic column takes its top from its own sounding at ``cfg.ztop``,
    not from the run's declared ``p_top``, so its radiation compiles the RTE
    for its own above-model layer count (``woof.core.rrtmgp.
    rrtmgp_above_model_layer_counts``).  The static plan reads this to compile
    that variant beside the others (woof.kernel_prewarm.static_jobs).
    The state carries it as float32, as here.
    """

    from woof.core.grid import make_base_state, make_vertical_coord

    coord = make_vertical_coord(cfg.nz, hybrid_opt=cfg.hybrid_opt, etac=cfg.etac)
    base = make_base_state(coord, _theta, p_surf=cfg.p_surf, ztop=cfg.ztop)
    return float(np.float32(base.p_top))


def run_column(cfg, *, reflectivity: bool = False) -> str:
    """Build the synthetic domain under ``cfg``; step it twice.

    ``reflectivity`` also asks the second step for the 10 cm reflectivity
    a history write takes, as a forecast's output steps do.
    """

    import cupy as cp

    from woof.core.dycore import step
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.moist import init_moist_balanced

    coord = make_vertical_coord(cfg.nz, hybrid_opt=cfg.hybrid_opt,
                                etac=cfg.etac)
    base = make_base_state(coord, _theta, p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_moist_balanced(cfg, coord, base, _vapour)
    grid = (cfg.ny, cfg.nx)
    glw, longwave = synthetic_downward_longwave(cfg)
    driver = _initialize_column_physics(
        state, cfg, radiation_start_time=_VALID_TIME,
        radiation_latitude=np.full(grid, _LATITUDE),
        radiation_longitude=np.full(grid, _LONGITUDE),
        noahmp_start_time=_VALID_TIME,
        noahmp_latitude=np.full(grid, _LATITUDE),
        noahmp_longitude=np.full(grid, _LONGITUDE),
        **_synthetic_mosaic_fractions(cfg, grid),
        **_synthetic_orographic_static(cfg, grid),
        **({} if glw is None else {"glw": glw}))
    _seed_monthly_leaf_area(driver, cfg, grid)
    if cfg.specified:
        _attach_constant_boundaries(state, cfg)
    for index in range(_STEPS):
        step(state, cfg, refl_10cm_due=reflectivity and index == _STEPS - 1)
    cp.cuda.runtime.deviceSynchronize()
    return longwave


def _initialize_column_physics(*args, **kwargs):
    """Compile-probe physics owns no geographical hydro network.

    A forecast's inherited hydro settings describe its actual projected grid,
    not this synthetic column. Constructing that network here failed the
    default-on CONUS pre-warm with "needs the actual forecast ProjectedGrid"
    (also 2.9.4 R11, ERA5 door with hydro on, 2026-10-22: no column kernel was
    warmed before the first step).
    The real forecast constructs and warms its coupled hydro on its own grid;
    this local scope neither changes its settings nor constructs a second copy.
    """
    from woof.core.physics import initialize_physics
    from woof.hydro.runtime import settings_scope

    with settings_scope({"enabled": False, "activation_reason": "synthetic_kernel_probe"}):
        return initialize_physics(*args, **kwargs)


#: The synthetic column's dominant categories: initialize_physics's own
#: ivgtyp / isltyp defaults.
_SYNTHETIC_IVGTYP = 10
_SYNTHETIC_ISLTYP = 6


def _synthetic_mosaic_fractions(cfg, grid) -> dict:
    """LANDUSEF / SOILCTOP for a RUC mosaic suite: all of each cell in its
    dominant category, one row per RUC table category.

    A real road reads them from the static fields; the column has none, and
    initialize_physics refuses a mosaic run without them ("RUC mosaic needs
    LANDUSEF category fractions"), which left the packaged hrrr_v4_gsd41
    recipe's (mosaic_lu = mosaic_soil = 1) forecast column unwarmed
    (2026-10-21).  The category count is a kernel argument, not a compile
    option, so the values do not change what compiles.
    """

    if int(cfg.sf_surface_physics) != 3 or not (cfg.mosaic_lu or cfg.mosaic_soil):
        return {}
    from woof.core.ruc import load_ruc_parameters

    bundle = load_ruc_parameters()
    out = {}
    for switch, name, rows, dominant in (
            (cfg.mosaic_lu, "landusef",
             len(bundle.vegetation_for("MODIFIED_IGBP_MODIS_NOAH").rows), _SYNTHETIC_IVGTYP),
            (cfg.mosaic_soil, "soilctop", len(bundle.soil.rows), _SYNTHETIC_ISLTYP)):
        if switch:
            fractions = np.zeros((rows, *grid), dtype=np.float32)
            fractions[dominant - 1] = 1.0
            out[name] = fractions
    return out


#: The synthetic column's sub-grid orographic statistics (m for VAR*, a
#: fraction or ratio for the rest): finite, nonzero, flat-column scale.
_SYNTHETIC_OROGRAPHY_VALUE = {"VAR": 50.0, "VARLS": 50.0, "VARSS": 20.0, "VAR_SSO": 50.0}


def _synthetic_orographic_static(cfg, grid) -> dict:
    """The static field set topo_wind / gwd_opt read, for the flat column.

    A real road hands initialize_physics its prepared static fields; the
    column has none, and the terrain drag refuses without them ("this
    initialization route hands the physics no static field set"), which left
    the packaged hrrr_v4_gsd41 recipe's (gwd_opt = 3) forecast column
    unwarmed (2026-10-21).  The values only need to be finite: the
    kernels the drag compiles do not depend on them.
    """

    topo_wind = int(getattr(cfg, "topo_wind", 0) or 0)
    gwd_opt = int(getattr(cfg, "gwd_opt", 0) or 0)
    if not (topo_wind or gwd_opt):
        return {}
    from woof.static.orographic import required_static_fields

    static = {"HGT_M": np.zeros(grid, dtype=np.float32)}
    for name in required_static_fields(topo_wind, gwd_opt):
        static[name] = np.full(grid, _SYNTHETIC_OROGRAPHY_VALUE.get(name, 0.1), dtype=np.float32)
    return {"terrain_drag_static": static}


def _attach_constant_boundaries(state, cfg) -> None:
    """Specified forcing that holds the column's own initial state.

    Two equal frames, so the boundary tendencies are zero: the
    relaxation and spec_bdy_final kernels run as a specified run's do,
    on values the column already holds.
    """

    from woof.ingest.lateral_bc import StateBoundaryFrames, attach_lateral_boundaries

    frames = StateBoundaryFrames(spec_bdy_width=cfg.spec_bdy_width,
                                 spec_zone=cfg.spec_zone, relax_zone=cfg.relax_zone)
    frames.add_state(state, index=0)
    frames.add_state(state, index=1)
    attach_lateral_boundaries(state, frames.build((0.0, float(cfg.run_seconds))))


#: The synthetic column's monthly leaf area (m2 m-2), the same in every
#: month.  Only its being finite matters: the kernels a profile compiles do
#: not depend on the value.
_SYNTHETIC_LAI12M = 2.0


def _seed_monthly_leaf_area(driver, cfg, grid) -> None:
    """Seed LAI from a synthetic LAI12M the way every real road does.

    Under RUC with ``rdlai2d=true`` :func:`initialize_physics` leaves LAI
    not-a-number and each real road seeds it from the static LAI12M
    (:mod:`woof.runtime` via :func:`woof.core.landuse.surface_leaf_area`).
    The synthetic column has no static catalogue, so without this seed the
    RUC land-surface step refuses (:mod:`woof.core.ruc_runtime`: "this road
    seeded no LAI field") and the two RUC monthly-LAI profiles warmed
    nothing (warm-kernels on the unified box image, 2026-10-16: 30 of 32 profiles).
    """

    if not (int(getattr(cfg, "sf_surface_physics", 0)) == 3
            and bool(getattr(cfg, "rdlai2d", False))):
        return
    driver_fields = getattr(driver, "fields", None)
    if driver_fields is None or "lai" not in driver_fields:
        return
    import cupy as cp

    from woof.core.landuse import surface_leaf_area

    lai12m = np.full((12, *grid), _SYNTHETIC_LAI12M, dtype=np.float32)
    driver_fields["lai"][...] = cp.asarray(
        surface_leaf_area(cfg, lai12m, _VALID_TIME), dtype=cp.float32)


#: What a profile that reads an unstaged asset is told.  The physics tables
#: published outside the package are staged by ``woof fetch-tables``; any
#: other absent asset is named by ``woof doctor`` with its own command.
MISSING_ASSET_REMEDY = ("woof fetch-tables stages the physics tables this "
                        "suite reads; woof doctor names any other missing "
                        "asset and the command that installs it")


#: Card memory each profile worker is budgeted: a CUDA context, CuPy's pool,
#: the 16x16 synthetic domain and the physics tables the profile stages.
#: Measured 2026-10-17 on an RTX 4090 (--all-profiles --levels 49/50): 8
#: workers held 16.9-19.8 GB, about 2.1-2.5 GB each; the old 1 GiB budget let
#: 23 workers start on the 24 GB card and one died with
#: CUDA_ERROR_OUT_OF_MEMORY.  3 GiB fits 7 workers on a 4090, 10 on a 5090.
_WORKER_CARD_BYTES = 3 << 30
WORKERS_ENVIRONMENT = "WOOF_WARM_KERNELS_WORKERS"


def profile_workers(count: int) -> int:
    """How many profiles warm at once: usable cores, card memory, the cap.

    1 when the card does not answer a memory query (the pass then runs in
    this process, as it always did).
    """

    if count <= 1:
        return 1
    try:
        import cupy as cp

        free, _total = cp.cuda.runtime.memGetInfo()
    except Exception:
        return 1
    from woof.ingest.preparation_workers import cpu_budget

    # Affinity capped by the cgroup CPU quota (U-E5: nproc 256, quota 61.44).
    cores = cpu_budget()["available_cpus"]
    workers = min(count, cores, max(1, int(free) // _WORKER_CARD_BYTES))
    cap = os.environ.get(WORKERS_ENVIRONMENT, "").strip()
    if cap.isdigit() and int(cap) > 0:
        workers = min(workers, int(cap))
    return max(1, workers)


def _out_of_memory(error: BaseException) -> bool:
    """A card-memory refusal (a property of how many workers share the card, not of the profile)."""

    try:
        from cupy.cuda.memory import OutOfMemoryError

        if isinstance(error, OutOfMemoryError):
            return True
    except Exception:  # noqa: BLE001 - no CuPy: judge by the text
        pass
    return "OUT_OF_MEMORY" in str(error) or "out of memory" in str(error).lower()


def _profile_worker(profile: str, levels: int) -> dict:
    """One profile in a spawned worker process; returns a picklable outcome."""

    from woof import kernel_plan
    from woof import nvrtc_ptx_cache as direct_cache

    planned = _bind_profile_plan(profile, levels)
    direct_cache.cache_events(clear=True)  # this profile's events only
    began = time.perf_counter()
    outcome: dict = {"profile": profile}
    try:
        outcome["longwave"] = _run_profile(profile, levels)
    except ValueError as refusal:
        outcome["refused"] = str(refusal)
    except FileNotFoundError as missing:
        outcome["missing"] = str(missing)
    except Exception as error:  # noqa: BLE001 - a device fault stops the pass
        key = "oom" if _out_of_memory(error) else "fatal"
        outcome[key] = f"{profile}: {type(error).__name__}: {error}"
    outcome["seconds"] = round(time.perf_counter() - began, 1)
    outcome["direct_events"] = direct_cache.cache_events()
    if planned:
        receipt = kernel_plan.receipt()
        calls, replay = receipt.get("calls") or {}, receipt.get("replay") or {}
        outcome["nvrtc_compiles"] = int(calls.get("inline_compiles", 0)) + int(replay.get("pool_compiled", 0))
        kernel_plan.save()  # a pool worker exits without running atexit
    return outcome


def _warm_side_by_side(profiles, levels, workers, say, rows, failed):
    """Warm ``profiles`` in ``workers`` spawned processes; fill rows/failed."""

    import concurrent.futures
    import multiprocessing

    outcomes = {}
    fatal = None
    context = multiprocessing.get_context("spawn")

    def take(profile, outcome, retried=False):
        nonlocal fatal
        if retried:
            outcome["retried_after_out_of_memory"] = True
        outcomes[profile] = outcome
        if "oom" in outcome:
            if retried:  # alone on the card and still out of memory: a real fault
                fatal = fatal or outcome["oom"]
            return
        if "fatal" in outcome:
            fatal = fatal or outcome["fatal"]
            return
        if say is None:
            return
        if "refused" in outcome:
            say(f"warm-kernels: {profile}: FAILED: {outcome['refused']}")
        elif "missing" in outcome:
            say(f"warm-kernels: {profile}: FAILED: {outcome['missing']}. Next: "
                f"{MISSING_ASSET_REMEDY}")
        else:
            compiled = outcome.get("nvrtc_compiles")
            say(f"warm-kernels: {profile}: "
                f"{'?' if compiled is None else compiled} kernel(s) compiled in "
                f"{outcome['seconds']:.1f} s; downward longwave {outcome['longwave']}"
                + (" (retried alone after running out of card memory)" if retried else ""))

    with concurrent.futures.ProcessPoolExecutor(workers, mp_context=context) as pool:
        futures = {pool.submit(_profile_worker, profile, int(levels)): profile
                   for profile in profiles}
        for future in concurrent.futures.as_completed(futures):
            profile = futures[future]
            try:
                outcome = future.result()
            except Exception as error:  # noqa: BLE001 - a worker that died
                key = "oom" if _out_of_memory(error) else "fatal"
                outcome = {"profile": profile, key: f"{profile}: {type(error).__name__}: {error}",
                           "seconds": None, "direct_events": []}
            take(profile, outcome)
    # A profile that ran out of card memory beside the others is not a fault of
    # the profile: it warms again after the pool, alone, in a fresh process.
    retry = [profile for profile in profiles if "oom" in outcomes.get(profile, {})]
    if retry and fatal is None:
        if say is not None:
            say(f"warm-kernels: {len(retry)} profile(s) ran out of card memory beside the others; "
                "warming them one at a time")
        with concurrent.futures.ProcessPoolExecutor(1, mp_context=context,
                                                    max_tasks_per_child=1) as alone:
            for profile in retry:
                try:
                    outcome = alone.submit(_profile_worker, profile, int(levels)).result()
                except Exception as error:  # noqa: BLE001 - a worker that died
                    outcome = {"profile": profile, "fatal": f"{profile}: {type(error).__name__}: {error}",
                               "seconds": None, "direct_events": []}
                take(profile, outcome, retried=True)
    if fatal is not None:
        raise RuntimeError(f"warm-kernels stopped: {fatal}")
    events = []
    for profile in profiles:  # the report keeps the requested order
        outcome = outcomes[profile]
        events.extend(outcome["direct_events"])
        if "refused" in outcome:
            failed.append({"profile": profile, "error": outcome["refused"]})
            continue
        if "missing" in outcome:
            failed.append({"profile": profile, "error": outcome["missing"],
                           "remedy": MISSING_ASSET_REMEDY})
            continue
        row = {"profile": profile,
               # NVRTC compiles this profile's worker made: side by side,
               # a cache-entry difference cannot be split between profiles.
               "kernels_compiled": int(outcome.get("nvrtc_compiles") or 0),
               "seconds": outcome["seconds"],
               "downward_longwave": outcome["longwave"]}
        row.update(_direct_cache_counts(outcome["direct_events"]))
        rows.append(row)
    return events


def warm_kernels(profiles=None, *, levels: int = DEFAULT_LEVELS,
                 say=None) -> dict:
    """Compile the forecast kernels of ``profiles`` into the kernel cache.

    Returns the report the door prints: per profile, the cache entries it
    wrote, its seconds and where its downward longwave came from, and the
    totals.  ``say`` receives one line per profile as it finishes.

    A profile whose configuration is refused (a ``ValueError`` from the
    forecast code it runs) is recorded under ``failed`` with the refusal,
    and the others still run: one refused suite used to stop the whole
    ``--all-profiles`` pass at the first such profile, leaving every
    later one cold.  A profile that reads a table asset this install has
    not staged (a ``FileNotFoundError``, such as a Thompson table before
    ``woof fetch-tables``) is recorded the same way, with
    :data:`MISSING_ASSET_REMEDY`, because the suites that do not read it
    can still warm.  Anything else, such as no card or a CUDA error,
    still stops the pass, because a device fault is not a property of
    one profile and the ones after it would meet it too.
    """

    from woof import kernel_compile_notice as notice
    from woof import nvrtc_ptx_cache as direct_cache

    profiles = tuple(profiles) if profiles else default_profiles()
    capability = notice.current_compute_capability()
    if capability is None:
        raise RuntimeError(
            "no CUDA device answered: warm-kernels compiles for the local "
            "card, so it needs one (woof doctor says why the card is "
            "unreachable)")
    import cupy as cp

    device = cp.cuda.runtime.getDeviceProperties(
        cp.cuda.Device().id)["name"]
    if isinstance(device, bytes):
        device = device.decode(errors="replace")
    cache_dir = notice.cupy_kernel_cache_dir()
    entries_before = notice.scan_kernel_cache(cache_dir)[0]
    direct_before = _direct_cache_inventory(cache_dir)
    direct_event_start = len(direct_cache.cache_events())
    rows = []
    failed = []
    started = time.perf_counter()
    workers = profile_workers(len(profiles))
    side_events = None
    if workers > 1:
        side_events = _warm_side_by_side(profiles, levels, workers, say, rows, failed)
    for profile in (() if workers > 1 else profiles):
        before = notice.scan_kernel_cache(cache_dir)[0]
        direct_profile_start = len(direct_cache.cache_events())
        began = time.perf_counter()
        # The in-process pass compiles on every core too (one profile, or a
        # card that does not answer): the same plan a worker binds.
        _bind_profile_plan(profile, levels)
        try:
            longwave = _run_profile(profile, levels)
        except ValueError as refusal:
            failed.append({"profile": profile, "error": str(refusal)})
            if say is not None:
                say(f"warm-kernels: {profile}: FAILED: {refusal}")
            continue
        except FileNotFoundError as missing:
            failed.append({"profile": profile, "error": str(missing),
                           "remedy": MISSING_ASSET_REMEDY})
            if say is not None:
                say(f"warm-kernels: {profile}: FAILED: {missing}. Next: "
                    f"{MISSING_ASSET_REMEDY}")
            continue
        row = {"profile": profile,
               "kernels_compiled": max(
                   0, notice.scan_kernel_cache(cache_dir)[0] - before),
               "seconds": round(time.perf_counter() - began, 1),
               "downward_longwave": longwave}
        direct_events = direct_cache.cache_events()[direct_profile_start:]
        row.update(_direct_cache_counts(direct_events))
        rows.append(row)
        if say is not None:
            say(f"warm-kernels: {profile}: {row['kernels_compiled']} "
                f"kernel(s) compiled in {row['seconds']:.1f} s; "
                f"downward longwave {longwave}")
    finished = notice.kernel_cache_state(
        cache_dir, compute_capability=capability)
    direct_after = _direct_cache_inventory(cache_dir)
    direct_events = (side_events if side_events is not None
                     else direct_cache.cache_events()[direct_event_start:])
    entries_after = notice.scan_kernel_cache(cache_dir)[0]
    return {
        "schema": WARM_KERNELS_SCHEMA,
        "device": str(device),
        "compute_capability": str(capability),
        "cache_dir": str(cache_dir),
        "levels": int(levels),
        "profiles": rows,
        "failed": failed,
        "kernels_compiled": (sum(row["kernels_compiled"] for row in rows)
                             if side_events is None
                             else max(0, entries_after - entries_before)),
        "seconds": round(time.perf_counter() - started, 1),
        "workers": workers,
        "cache_entries_before": int(entries_before),
        "cache_entries_for_this_card": int(finished.entries_for_capability),
        "direct_nvrtc_cache": {
            "directory": str(cache_dir / "gpuwm-direct-nvrtc"),
            "before": direct_before,
            "after": direct_after,
            "new_image_files": max(0, direct_after["image_files"]
                                    - direct_before["image_files"]),
            **_direct_cache_counts(direct_events),
            "events": direct_events,
        },
    }


def _direct_cache_inventory(cache_dir) -> dict:
    """Count the direct compiler's image files separately from CuPy entries."""
    count = size = 0
    for root, _directories, names in os.walk(cache_dir / "gpuwm-direct-nvrtc"):
        for name in names:
            if not name.endswith(".image"):
                continue
            try:
                size += os.stat(os.path.join(root, name)).st_size
                count += 1
            except OSError:
                pass
    return {"image_files": count, "bytes": size}


def _direct_cache_counts(events) -> dict:
    return {
        "direct_nvrtc_images_compiled": sum(
            row["status"] != "hit" for row in events),
        "direct_nvrtc_cache_hits": sum(row["status"] == "hit" for row in events),
        "direct_nvrtc_compile_seconds": sum(
            row["compile_seconds"] for row in events),
    }


def warm_kernels_main(args) -> int:
    from woof import physics_menu

    if args.all_profiles:
        profiles = physics_menu.shipped_profiles()
    else:
        profiles = tuple(args.profile or ()) or None
    say = None if args.json else (lambda text: print(text, flush=True))
    try:
        report = warm_kernels(profiles, levels=args.levels, say=say)
    except (RuntimeError, ValueError) as error:
        if args.json:
            print(json.dumps({"schema": WARM_KERNELS_SCHEMA,
                              "error": str(error)}))
        else:
            print(f"warm-kernels: {error}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report, allow_nan=False))
    else:
        print(f"warm-kernels: {report['kernels_compiled']} kernel(s) "
              f"compiled in {report['seconds']:.1f} s for "
              f"{report['device']} (sm_{report['compute_capability']}); "
              f"the cache at {report['cache_dir']} holds "
              f"{report['cache_entries_for_this_card']} for this card",
              flush=True)
        direct = report["direct_nvrtc_cache"]
        if direct["events"]:
            print("warm-kernels: direct NVRTC cache: "
                  f"{direct['direct_nvrtc_images_compiled']} image(s) compiled, "
                  f"{direct['direct_nvrtc_cache_hits']} cache hit(s); "
                  f"{direct['after']['image_files']} image file(s) in "
                  f"{direct['directory']}", flush=True)
        if report["failed"]:
            print(f"warm-kernels: {len(report['failed'])} profile(s) "
                  "failed: " + ", ".join(row["profile"]
                                         for row in report["failed"]),
                  file=sys.stderr)
    return 1 if report["failed"] else 0


def register_cli(subparsers) -> None:
    from woof.cli_numbers import int_at_least

    parser = subparsers.add_parser(
        "warm-kernels",
        help="compile the forecast GPU kernels for this card now, so the "
             "first forecast does not pay the compile")
    parser.add_argument(
        "--profile", action="append", default=None, metavar="PROFILE",
        help="a physics profile to compile for (repeatable); default: "
             "every profile a source defaults to")
    parser.add_argument(
        "--all-profiles", action="store_true",
        help="compile for every shipped physics profile")
    # Four is the vertical stencil width RunConfig refuses below; the
    # parser says so before a CUDA context is made for a domain that
    # cannot be built.
    parser.add_argument(
        "--levels", type=int_at_least(4), default=DEFAULT_LEVELS,
        help=f"vertical levels of the compile domain (default "
             f"{DEFAULT_LEVELS})")
    parser.add_argument(
        "--json", action="store_true",
        help="print the report as one JSON document")
    parser.set_defaults(func=warm_kernels_main)


__all__ = ["DEFAULT_LEVELS", "MISSING_ASSET_REMEDY", "WARM_KERNELS_SCHEMA",
           "WORKERS_ENVIRONMENT", "default_profiles", "profile_workers",
           "register_cli", "synthetic_downward_longwave", "warm_kernels",
           "warm_kernels_main"]
