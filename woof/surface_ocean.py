"""Surface-ocean metadata shared by config, input, runtime and restart."""
from functools import lru_cache
import json
from pathlib import Path

@lru_cache(maxsize=1)
def ocean_option(option=1):
    table = json.loads((Path(__file__).parent / "data/surface_ocean.json").read_text(encoding="utf-8"))
    return table["options"][str(option)]

def apply_ocean_defaults(values):
    """Resolve omitted ocean selection for multi-day surface-coupled forecasts.

    An explicit 0/1/2 remains authoritative. Land/lakes are excluded by the
    CUDA ocean mask; short and flux-free ideal forecasts keep their selection.
    Resolution precedes memory planning and config hashing in both loaders.
    """
    from woof.config import RunConfig
    table = json.loads((Path(__file__).parent / "data/surface_ocean.json").read_text(encoding="utf-8"))
    policy = table['multiday_default']
    if ('sf_ocean_physics' not in values
            and values.get('run_seconds', 0) >= policy['minimum_run_seconds']
            and (not policy['requires_surface_fluxes'] or
                 values.get('sf_sfclay_physics', RunConfig.sf_sfclay_physics) != 0)):
        values['sf_ocean_physics'] = policy['option']
        values.setdefault('ocean_mld_climatology', policy['initialization_fallback'])
    return values

def ocean_fields(cfg):
    option = int(getattr(cfg, "sf_ocean_physics", 0) or 0)
    return tuple(ocean_option(option)["state_fields"]) if option else ()

def ocean_namelist_defaults():
    return dict(ocean_option()["namelist_defaults"])

def drop_inert_ocean(values):
    if not values.get("sf_ocean_physics", 0):
        active = ({name: values[name] for name in
                   ("ocean_analysis", "ocean_analysis_source", "ocean_levels")
                   if name in values} if values.get("ocean_analysis") else {})
        for name in ("sf_ocean_physics", "ocean_levels", "ocean_analysis", "ocean_analysis_source", "ocean_mld_climatology", "omdt", "ocean_z", "ocean_t", "ocean_s", *ocean_namelist_defaults()):
            values.pop(name, None)
        values.update(active)

def validate_pwp_profile(cfg):
    """Check configuration metadata before allocating the ocean device state."""
    if cfg.sf_ocean_physics != 2:
        return
    import math
    if isinstance(cfg.omdt,bool) or not isinstance(cfg.omdt,(float,int)) or not math.isfinite(cfg.omdt) or cfg.omdt <= 0:
        raise ValueError('omdt must be positive minutes: a zero ocean timestep cannot advance heat and momentum')
    from woof.ocean.mixed_layer.analysis import SOURCES
    if cfg.ocean_analysis_source not in SOURCES:
        raise ValueError(f"unknown ocean_analysis_source {cfg.ocean_analysis_source!r}; choose {tuple(SOURCES)}")
    if cfg.ocean_mld_climatology not in ('woa18', 'configured_uniform_mld'):
        raise ValueError('ocean_mld_climatology must be woa18 or configured_uniform_mld')
    values = (cfg.ocean_z, cfg.ocean_t, cfg.ocean_s)
    if not any(values):
        return
    if cfg.ocean_analysis:
        raise ValueError("ocean_analysis and ocean_z/ocean_t/ocean_s both specify initialization; choose one source")
    for name, column in zip(('ocean_z', 'ocean_t', 'ocean_s'), values):
        if len(column) != cfg.ocean_levels or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in column):
            raise ValueError(f"{name} must contain ocean_levels finite numbers; a partial reference profile corrupts PWP density")
    if cfg.ocean_z[0] <= 0 or any(b <= a for a,b in zip(cfg.ocean_z, cfg.ocean_z[1:])):
        raise ValueError("ocean_z layer bottoms must be positive and strictly increasing in metres")
    if any(t < 260 or t > 330 for t in cfg.ocean_t) or any(s < 0 or s > 50 for s in cfg.ocean_s):
        raise ValueError("ocean_t must be Kelvin in [260,330] and ocean_s PSU in [0,50]")

def ocean_input_fields(cfg, fields):
    if not getattr(cfg, "sf_ocean_physics", 0):
        return {}
    return {"ocean_initial": {name.lower(): fields[name] for name in ocean_option(cfg.sf_ocean_physics)["input_fields"] if name in fields}}

def validate_ocean_inputs(cfg, initial):
    if not cfg.sf_ocean_physics:
        return
    if cfg.sf_ocean_physics == 2:
        return  # Full profile checks run on the device during initialization.
    initial = initial or {}
    if cfg.oml_hml0 <= 0 and initial.get("h0ml") is None:
        raise ValueError(f"sf_ocean_physics=1 with oml_hml0={cfg.oml_hml0} requires a 2D H0ML input: otherwise mixed-layer depth is uninitialised")
    if cfg.oml_hml0 < 0:
        missing = [name.upper() for name in ("huml", "hvml", "tmoml") if initial.get(name) is None]
        if missing:
            raise ValueError("oml_hml0<0 requires supplied " + ", ".join(missing) + ": WRF's 2D branch preserves currents and the ocean mean temperature; these states would be uninitialised")
