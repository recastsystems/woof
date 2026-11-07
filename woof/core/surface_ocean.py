"""CUDA launchers for the metadata-selected surface-ocean column model."""
import numpy as np
from woof.surface_ocean import ocean_option, validate_ocean_inputs
from woof.core.kernels import get_kernel
from woof.core import constants

STATE_FIELDS = tuple(ocean_option()["state_fields"])
FORCING_FIELDS = ("hfx", "lh", "gsw", "glw", "uair", "vair", "ust", "f", "emiss", "xland")
_DIAGNOSTICS = None


def update_ocean_diagnostics(fields):
    """Update finite live SST diagnostics; history masks non-ocean cells.

    These stored 2D fields are directly reachable through rw_wrfbatch's
    var:wrf_ocean_sst and var:wrf_ocean_sst_change products.
    """
    import cupy as cp
    global _DIAGNOSTICS
    if _DIAGNOSTICS is None:
        _DIAGNOSTICS = cp.ElementwiseKernel(
            'float32 t, float32 t0, float32 land, float32 lake',
            'float32 sst, float32 change',
            'bool wet = land > 1.5f && lake < 0.5f; '
            'sst = wet ? t : 0.0f; change = wet ? t-t0 : 0.0f;',
            'coupled_ocean_sst_diagnostics')
    for name in ('ocean_sst', 'ocean_sst_change'):
        if name not in fields:
            fields[name] = cp.empty_like(fields['tsk'])
    t0 = fields['om_tini'][0] if 'om_tini' in fields else fields['t0ml']
    _DIAGNOSTICS(fields['tsk'], t0, fields.get('_pwp_land', fields.get('xland', np.float32(1))), fields.get('lakemask', np.float32(0)),
                 fields['ocean_sst'], fields['ocean_sst_change'])

def ocean_history_diagnostic(value, fields):
    """Mask output on CUDA, preserving finite health-checked live carriers."""
    if not hasattr(value, 'dtype'):
        return value  # The disk planner walks shape-only descriptors.
    import cupy as cp
    wet = (fields.get('_pwp_land', fields['xland']) > 1.5) & (fields.get('lakemask', 0) < 0.5)
    return cp.where(wet, value, cp.float32('nan'))


def _check(fields, names):
    shape = fields["tsk"].shape
    for name in names:
        a = fields[name]
        if a.dtype != np.dtype("float32") or a.shape != shape or not a.flags.c_contiguous:
            raise ValueError(f"surface ocean {name} requires contiguous float32 columns with shape {shape}")
    return fields["tsk"].size

def launch_ocean_init(fields, hml0, *, start=True):
    n = _check(fields, (*STATE_FIELDS, "tsk"))
    get_kernel("oml", "oml_init")(((n+127)//128,), (128,),
        tuple(fields[name] for name in STATE_FIELDS) + (fields["tsk"], np.float32(hml0), np.int32(start), np.int32(n)))

_ICE_OWNED = None


def _ice_owned(fields):
    """Columns whose OML surface heat is the sea-ice step's ICE_OCEAN_HEAT.

    ``ice_ocean_mask`` is every non-lake water column, so passing it as the
    ice mask swapped the ice model's own open-water flux into all open ocean
    as soon as ice sat anywhere on the grid. The ice step's flux applies only
    where it set the surface fluxes (``ice_ocean_coupled``, as on the 3D PWP
    route) and on masked water the atmosphere sees as land (XLAND 1 sea ice
    after it melts), where no atmospheric water flux exists. Every masked
    column stays a mask column for oml_step's ownership test (XLAND > 1.5 or
    this mask), so which columns the ocean steps is unchanged.
    """
    import cupy as cp
    global _ICE_OWNED
    if _ICE_OWNED is None:
        _ICE_OWNED = cp.ElementwiseKernel(
            'float32 mask, float32 coupled, float32 land', 'float32 owned',
            'owned = (mask > 0.0f && (coupled > 0.0f || land <= 1.5f)) ? 1.0f : 0.0f;',
            'oml_ice_owned_columns')
    return _ICE_OWNED(fields["ice_ocean_mask"], fields["ice_ocean_coupled"], fields["xland"])


def launch_ocean_step(fields, *, dt, gamma, relaxation_time, constants_override=None, dx=None, dy=None, ocean_timestep_minutes=1.0):
    if "om_tmp" in fields:
        from woof.ocean.mixed_layer import step
        import math
        # OMDT is multiplied by 60 in WRF DPWP/PWP despite the Registry's
        # misleading "s" units. Subcycle the surface interval so coupling
        # never advances beyond the atmospheric clock or loses a remainder.
        substeps=max(1,math.ceil(dt/(ocean_timestep_minutes*60)))
        for _ in range(substeps):
            step(fields, dt=dt/substeps, dx=dx, dy=dy)
        update_ocean_diagnostics(fields)
        return
    n = _check(fields, (*STATE_FIELDS, "tsk", *FORCING_FIELDS))
    c = dict(ocean_option()["constants"])
    if constants_override is not None:
        c.update(constants_override)
    coupled = "ice_ocean_heat" in fields
    get_kernel("oml", "oml_step")(((n+127)//128,), (128,),
        tuple(fields[name] for name in (*STATE_FIELDS, "tsk", *FORCING_FIELDS)) +
        (fields.get("ice_ocean_heat", fields["tsk"]),
         _ice_owned(fields) if coupled else fields["xland"], np.int32(coupled)) +
        tuple(np.float32(v) for v in (dt, gamma, relaxation_time, c["rhowater"], c["cwater"], c["stbolt"], constants.G)) + (np.int32(n),))
    update_ocean_diagnostics(fields)

def initialize_ocean(fields, cfg, initial=None, *, start=True, continuing=False, latitude=None, longitude=None, rotation=None, expected_time=None):
    if cfg.sf_ocean_physics == 0:
        # An explicitly supplied analysis may also initialize a fixed-SST
        # control. It allocates no carried ocean state and advances no ocean.
        # Moving domains inherit the live surface instead of reloading t=0.
        if start and not continuing and cfg.ocean_analysis:
            from woof.ocean.mixed_layer.analysis import load
            from woof.ocean.mixed_layer import ocean_land
            import cupy as cp
            mapped = load(cfg.ocean_analysis, latitude=latitude, longitude=longitude,
                          land=ocean_land(fields), levels=cfg.ocean_levels,
                          source=cfg.ocean_analysis_source, rotation=rotation,
                          expected_time=expected_time, uncovered="prescribed")
            wet = mapped['om_active'] > .5
            cp.copyto(fields['tsk'], mapped['om_tmp'][0], where=wet)
            if 'sst' in fields:
                cp.copyto(fields['sst'], mapped['om_tmp'][0], where=wet)
        return
    if cfg.sf_ocean_physics == 2:
        if continuing:
            from dataclasses import replace
            from woof.ocean.mixed_layer import STATE_FIELDS as pwp_fields
            if not initial or not set(pwp_fields) <= set(initial):
                raise ValueError("PWP continuation requires every ocean state field")
            cfg = replace(cfg, ocean_analysis="", ocean_z=(), ocean_t=(), ocean_s=())
        if start and not cfg.ocean_analysis and not cfg.ocean_z and not initial and cfg.ocean_mld_climatology == 'woa18':
            from woof.ocean.mixed_layer.climatology import load
            from woof.ocean.mixed_layer import ocean_land
            initial = load(latitude=latitude, longitude=longitude, land=ocean_land(fields),
                           sst=fields['tsk'], levels=cfg.ocean_levels,
                           gamma=cfg.oml_gamma, expected_time=expected_time)
        if start and cfg.ocean_analysis:
            if initial:
                raise ValueError("ocean_analysis and explicit ocean_initial cannot both supply profiles")
            if latitude is None or longitude is None:
                raise ValueError("ocean_analysis needs the forecast latitude and longitude for GPU remapping")
            from woof.ocean.mixed_layer.analysis import load
            from woof.ocean.mixed_layer import ocean_land
            initial = load(cfg.ocean_analysis, latitude=latitude, longitude=longitude,
                           land=ocean_land(fields), levels=cfg.ocean_levels,
                           source=cfg.ocean_analysis_source, rotation=rotation, expected_time=expected_time, uncovered="prescribed")
        from woof.ocean.mixed_layer import initialize
        initialize(fields, cfg, initial, start=start)
        if start:
            from woof.ocean.seaice import seed_pwp_profile
            seed_pwp_profile(fields)
        update_ocean_diagnostics(fields)
        return
    import cupy as cp
    if start:
        validate_ocean_inputs(cfg, initial)
    initial = initial or {}
    shape = fields["tsk"].shape
    for name in STATE_FIELDS:
        value = initial.get(name, 0.0)
        if start and name == "h0ml" and cfg.oml_hml0 <= 0 and getattr(value, "ndim", None) != 2:
            raise ValueError("H0ML must be a 2D input: a scalar does not initialize mixed-layer depth per column")
        a = cp.asarray(value, dtype=cp.float32)
        if a.ndim and a.shape != shape:
            raise ValueError(f"surface ocean {name} requires a 2D input with shape {shape}")
        fields[name] = cp.full(shape, value, dtype=cp.float32) if a.ndim == 0 else cp.ascontiguousarray(a)
    if start and cfg.oml_hml0 <= 0 and bool(cp.any(~cp.isfinite(fields["h0ml"]) | (fields["h0ml"] < 0))):
        raise ValueError("H0ML must be finite and nonnegative: invalid input would leave mixed-layer depth uninitialised")
    launch_ocean_init(fields, cfg.oml_hml0, start=start)
    update_ocean_diagnostics(fields)
