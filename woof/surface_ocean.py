"""Surface-ocean metadata shared by config, input, runtime and restart."""
from functools import lru_cache
import json
from pathlib import Path

@lru_cache(maxsize=1)
def ocean_option(option=1):
    table = json.loads((Path(__file__).parent / "data/surface_ocean.json").read_text(encoding="utf-8"))
    return table["options"][str(option)]

@lru_cache(maxsize=1)
def _table():
    return json.loads((Path(__file__).parent / "data/surface_ocean.json").read_text(encoding="utf-8"))

def _reader(cfg):
    """Read a setting from a RunConfig or from a loader's values dict."""
    if isinstance(cfg, dict):
        return cfg.get
    return lambda name, default: getattr(cfg, name, default)

def ocean_statement(cfg):
    """One line naming the ocean this configuration runs, as the plan, the dry
    run, the run log and the run record all state it.

    The ocean is chosen by ``sf_ocean_physics`` alone (RunConfig default 0,
    WRF's), never by run length: the omitted selection used to become the 3-D
    PWP ocean at 48 h, so a 47 h and a 48 h run of one case differed from the
    first hour (88 of 165 history variables) with nothing said before the run.
    """
    words = _table()["statement"]
    get = _reader(cfg)
    option = int(get("sf_ocean_physics", 0) or 0)
    analysis = str(get("ocean_analysis", "") or "")
    source = str(get("ocean_analysis_source", "rtofs") or "rtofs")
    named = {"source": source, "analysis": Path(analysis).name if analysis else ""}
    if option == 0:
        sst = (words["sst_analysis"].format(**named) if analysis else words["sst_input"])
        return words["0"].format(sst=sst)
    if option == 1:
        depth = float(get("oml_hml0", ocean_namelist_defaults()["oml_hml0"]))
        mixed = (words["mixed_layer_depth"].format(oml_hml0=depth) if depth > 0
                 else words["mixed_layer_input"])
        return words["1"].format(mixed_layer=mixed)
    if analysis:
        start = words["start_analysis"].format(**named)
    elif get("ocean_z", ()):
        start = words["start_profile"]
    else:
        start = words["start_" + str(get("ocean_mld_climatology", "woa18"))]
    return words[str(option)].format(start=start)

def ocean_offer(cfg, run_seconds=None):
    """The one-line offer of the long-run ocean, or None.

    Only an offer: option 2 is never selected by run length.  Stated for
    surface-coupled runs at least ``long_run_offer.minimum_run_seconds`` long
    that run no ocean model, so a user who wants the mixed layer hears of it.
    ``run_seconds`` is a door's run-length override (``--run-seconds``).
    """
    from woof.config import RunConfig
    offer = _table()["long_run_offer"]
    get = _reader(cfg)
    seconds = float(run_seconds if run_seconds is not None else
                    get("run_seconds", 0) or 0)
    if (int(get("sf_ocean_physics", 0) or 0) != 0
            or seconds < offer["minimum_run_seconds"]
            or (offer["requires_surface_fluxes"] and
                int(get("sf_sfclay_physics", RunConfig.sf_sfclay_physics) or 0) == 0)):
        return None
    return offer["words"].format(hours=seconds / 3600.0)

def ocean_lines(cfg, prefix="ocean: ", run_seconds=None):
    """The statement, then the offer when there is one, ready to print."""
    lines = [prefix + ocean_statement(cfg)]
    offer = ocean_offer(cfg, run_seconds)
    if offer is not None:
        lines.append(prefix + "offer: " + offer)
    return lines

def domains_ocean_lines(configs, prefix="ocean: ", run_seconds=None):
    """:func:`ocean_lines` for a domain tree: one set when every domain runs
    the same ocean, else one set per domain (``dNN``), root first."""
    configs = list(configs)
    sets = [ocean_lines(cfg, prefix, run_seconds) for cfg in configs]
    if not sets or all(lines == sets[0] for lines in sets):
        return sets[0] if sets else []
    out = []
    for cfg, lines in zip(configs, sets):
        tag = f"d{int(getattr(cfg, 'grid_id', 1) or 1):02d} "
        out.extend(prefix + tag + line[len(prefix):] for line in lines)
    return out

def config_run_configs(path):
    """The per-domain RunConfigs of a config file, root first ([] if none)."""
    from woof.experiment import is_experiment_toml, load_experiment
    if path is None:
        return []
    if is_experiment_toml(path):
        return [dc.run for dc in load_experiment(path).domains]
    from woof.config import load_config
    return [load_config(path)]

def config_ocean_lines(path, prefix="ocean: "):
    """:func:`domains_ocean_lines` for a config file, for a plan or dry run.

    A config that cannot be read prints nothing here: the loader that
    diagnoses it runs a moment later in the same command.
    """
    try:
        configs = config_run_configs(path)
    except Exception:  # noqa: BLE001 - the loader reports it, not the banner
        return []
    return domains_ocean_lines(configs, prefix)

def ocean_receipt(cfg):
    """What the run record carries about one domain's ocean."""
    return {"sf_ocean_physics": int(getattr(cfg, "sf_ocean_physics", 0) or 0),
            "statement": ocean_statement(cfg), "offer": ocean_offer(cfg)}

def domains_ocean_receipt(configs):
    """The run record's ``ocean`` block: one :func:`ocean_receipt` per domain."""
    return {f"d{int(getattr(cfg, 'grid_id', 1) or 1):02d}": ocean_receipt(cfg)
            for cfg in configs}

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
