"""Public sounding-driven WRF ideal-fire preparation and integration."""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields, replace
from datetime import datetime, timedelta
from fractions import Fraction
import hashlib
import json
from pathlib import Path


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _first(document, section, name, default=None):
    return document.get(section, {}).get(name, [default])[0]


def _native_date(document, prefix, *, required=False):
    defaults = dict(year=1, month=1, day=1, hour=0, minute=0, second=0)
    control = document.get("time_control", {})
    if required and not all(f"{prefix}_{name}" in control for name in ("year", "month", "day")):
        raise ValueError(f"native zero-duration namelist needs the {prefix} date")
    return datetime(**{name: int(_first(document, "time_control", f"{prefix}_{name}", default))
                       for name, default in defaults.items()})


def _native_dt(document):
    values = tuple(_first(document, "domains", name, default) for name, default in (
        ("time_step", 0), ("time_step_fract_num", 0), ("time_step_fract_den", 1)))
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError("native timestep whole, numerator and denominator must be integers")
    if values[2] <= 0:
        raise ValueError("native timestep denominator must be positive")
    return Fraction(values[0]) + Fraction(values[1], values[2])


def load_native_configuration(path, *, input_directory=None, run_seconds=None):
    """Read the single-domain native ideal namelist without WPS scaffolding.

    Fire fields retain their registry defaults from FireRunFields. Native
    dimensions count the terminal staggered point; RunConfig counts cells.
    """
    from woof.config import RunConfig, validate_run_config
    from woof.fortran_namelist import parse_namelist
    from woof.sfire_config import FIRE_CONFIG_FIELDS
    import struct

    path = Path(path).resolve()
    document = parse_namelist(path)
    directory = path.parent if input_directory is None else Path(input_directory).resolve()
    resolved_path = directory / "namelist.output"
    resolved = parse_namelist(resolved_path) if resolved_path.is_file() else {}
    max_dom = _first(document, "domains", "max_dom", 1)
    if max_dom != 1:
        raise ValueError(
            "fire-ideal initializes one sounding-driven domain; nested native "
            "wrfinput domains use woof run --wrfinput so their parent boundary "
            "and interpolation state are retained")
    unknown_fire = sorted(set(document.get("fire", {})) - set(FIRE_CONFIG_FIELDS))
    if unknown_fire:
        raise ValueError(f"unconsumed &fire keys: {unknown_fire}")
    defaults = {field.name: field.default for field in fields(RunConfig)}
    known = set(defaults)
    values = {}
    for source in (resolved, document):
        for section in ("domains", "physics", "dynamics", "fire"):
            for name, entries in source.get(section, {}).items():
                if name not in known or not entries:
                    continue
                if source is resolved and section == "physics" and name not in document.get("physics", {}) and name != "num_soil_layers":
                    # Inactive package defaults do not activate or configure
                    # an unselected atmospheric physics package.
                    continue
                value = entries[0]
                default = defaults[name]
                if isinstance(value, float) and isinstance(default, float):
                    if struct.pack("<f", value) == struct.pack("<f", default):
                        value = default
                values[name] = value
    domains = document.get("domains", {})
    for name in ("e_we", "e_sn", "e_vert", "dx", "dy", "ztop", "time_step"):
        if name not in domains:
            raise ValueError(f"native ideal namelist needs &domains {name}")
    dt = _native_dt(document)
    duration = sum(float(_first(document, "time_control", name, 0)) * factor
                   for name, factor in (("run_days", 86400), ("run_hours", 3600),
                                        ("run_minutes", 60), ("run_seconds", 1)))
    if run_seconds is not None:
        duration = float(run_seconds)
    elif duration == 0:
        duration = (_native_date(document, "end", required=True)
                    - _native_date(document, "start")).total_seconds()
    history = _first(document, "time_control", "history_interval_s")
    if history is None:
        history = float(_first(document, "time_control", "history_interval", 5)) * 60
    values.update(
        nx=int(domains["e_we"][0]) - int(_first(document, "domains", "s_we", 1)),
        ny=int(domains["e_sn"][0]) - int(_first(document, "domains", "s_sn", 1)),
        nz=int(domains["e_vert"][0]) - int(_first(document, "domains", "s_vert", 1)),
        dx=float(domains["dx"][0]), dy=float(domains["dy"][0]),
        ztop=float(domains["ztop"][0]), dt=float(dt), run_seconds=duration,
        output_interval_s=float(history), moist=True, terrain_opt=1,
        # Native ideal input forces this irrespective of real-case settings.
        hypsometric_opt=1,
        restart_interval_s=float(_first(document, "time_control", "restart_interval", 0)) * 60)
    if "eta_levels" in domains and domains["eta_levels"][0] != -1:
        values["eta_levels"] = tuple(domains["eta_levels"])
    else:
        values.pop("eta_levels", None)
    for axis in ("x", "y"):
        sides = [bool(_first(document, "bdy_control", f"open_{axis}{side}", False))
                 for side in ("s", "e")]
        if sides[0] != sides[1]:
            raise ValueError(f"the engine open_{axis} carrier requires matching native boundary sides")
        values[f"open_{axis}"] = sides[0]
    if bool(_first(document, "bdy_control", "specified", False)):
        raise ValueError("specified ideal boundaries need an external wrfbdy producer; use woof run --wrfinput")
    tracer_opt = _first(document, "dynamics", "tracer_opt", 0)
    if isinstance(tracer_opt, bool) or tracer_opt not in (0, 3):
        raise ValueError("native SFIRE tracer_opt must be 0 or 3 to select the implemented bulk-smoke row")
    values["fire_smoke"] = tracer_opt == 3
    # SFIRE tracer_opt = 3 is the bulk-smoke row above, not the non-Chem EM
    # passive-tracer package RunConfig.tracer_opt selects (0747f1ef907),
    # which refuses 3; the native key is consumed here.
    values["tracer_opt"] = 0
    fuel_file = directory / "namelist.fire"
    if fuel_file.is_file() and not values.get("fire_fuel_namelist"):
        values["fire_fuel_namelist"] = str(fuel_file)
    from woof.config import _anchor_config_file_paths
    _anchor_config_file_paths(values, path)
    cfg = validate_run_config(RunConfig(**values), native_fire_ideal=True)
    return cfg


def _make_clock(cfg, start, *, allow_zero_run=False, exact_dt=None):
    from woof.core.clock import DomainTicks, TickClock
    import numpy as np

    dt = Fraction(str(cfg.dt)) if exact_dt is None else exact_dt
    intervals = {"run_seconds": cfg.run_seconds,
                 "output_interval_s": cfg.output_interval_s}
    if cfg.restart_interval_s:
        intervals["restart_interval_s"] = cfg.restart_interval_s
    ticks = {}
    for name, value in intervals.items():
        interval = Fraction(str(value))
        zero_preparation = name == "run_seconds" and interval == 0 and allow_zero_run
        if (interval <= 0 and not zero_preparation) or interval % dt:
            raise ValueError(
                f"{name} must be positive and lie on the dt lattice: "
                "the ideal fixed clock must reach every requested alarm exactly")
        ticks[name] = int(interval * dt.denominator)
    spec = DomainTicks(
        grid_id=cfg.grid_id, parent_id=0, parent_time_step_ratio=1,
        step_ticks=dt.numerator, dt_fp32=np.float32(cfg.dt),
        history_ticks=ticks["output_interval_s"],
        restart_ticks=ticks.get("restart_interval_s"),
        radt_ticks=None, stepra=None, cudt_ticks=None, stepcu=None,
        bldt_ticks=None, stepbl=None)
    return TickClock(dt.denominator, ticks["run_seconds"], start, (spec,)).domain_clock(cfg.grid_id)


def _timestamp(start, seconds):
    valid = start + timedelta(seconds=seconds)
    return (f"{valid.year:04d}-{valid.month:02d}-{valid.day:02d}_"
            f"{valid.hour:02d}:{valid.minute:02d}:{valid.second:02d}")


def run_ideal(config, *, sounding, input_directory, outdir, namelist=False,
              run_seconds=None, restart=None, prepare_only=False):
    """Initialize and advance the ordinary coupled dycore from native inputs."""
    from woof.config import load_config, soil_layer_count, validate_run_config
    from woof.static.sfire import read_fire_ideal_inputs, fire_ideal_input_requests
    from woof.core.sfire_ideal import build_state
    from woof.core.physics import initialize_physics
    from woof.core.state import refresh_model_time
    from woof.core import dycore
    from woof.io.restart import restore_restart, write_restart
    from woof.io.wrfout import WrfoutWriter, state_frame, wrf_physics_selector_attrs
    import cupy as cp
    import numpy as np
    import time

    config = Path(config).resolve()
    sounding = Path(sounding).resolve()
    directory = Path(input_directory).resolve()
    outdir = Path(outdir).resolve()
    if any(outdir.glob("wrfout*")) or (outdir / "run-receipt.json").exists():
        raise ValueError("choose a fresh output folder to preserve existing ideal histories and receipts")
    if namelist:
        cfg = load_native_configuration(config, input_directory=directory, run_seconds=run_seconds)
    else:
        cfg = load_config(config, native_fire_ideal=True)
        changes = {"terrain_opt": 1, "hypsometric_opt": 1}
        if run_seconds is not None:
            changes["run_seconds"] = float(run_seconds)
        cfg = validate_run_config(replace(cfg, **changes), native_fire_ideal=True)
    if cfg.use_adaptive_time_step:
        raise ValueError(
            "fire-ideal uses the native fixed ideal clock; adaptive nested "
            "forecasts use woof run with their declared experiment clock")
    if cfg.fire_static:
        raise ValueError(
            "fire-ideal uses Cartesian native input grids; a geographic "
            "fire_static bundle needs the real experiment door to preserve "
            "its projection and geographic ignitions")
    if namelist:
        from woof.fortran_namelist import parse_namelist
        document = parse_namelist(config)
        start = _native_date(document, "start")
        exact_dt = _native_dt(document)
    else:
        start = datetime(1, 1, 1)
        exact_dt = None
    clock = _make_clock(cfg, start, allow_zero_run=prepare_only, exact_dt=exact_dt)
    requests = fire_ideal_input_requests(cfg)
    parsed = read_fire_ideal_inputs(directory, requests, sounding=sounding)
    parsed["fields"]["_IDEAL_JULDAY"] = start.timetuple().tm_yday
    state, fire_static, surface, initial_fields, preparation = build_state(
        cfg, parsed["sounding"], parsed["fields"])
    driver = initialize_physics(state, cfg, fire_static_data=fire_static, **surface)
    if driver.fire is None:
        raise RuntimeError("ifire=2 did not attach the coupled fire driver")
    inputs = {"sounding": _sha256(sounding), "native_fields": {
        request["name"]: _sha256(directory / request["filename"])
        for request in requests}}
    if cfg.fire_fuel_namelist:
        inputs["fuel_namelist"] = _sha256(cfg.fire_fuel_namelist)
    if preparation.get("landuse_table"):
        inputs["landuse_table"] = preparation["landuse_table"]
    receipt = {"schema": "gpuwm-fire-ideal-run-v1", "configuration": asdict(cfg),
               "initialization": preparation, "inputs": inputs,
               "configuration_sha256": _sha256(config),
               "native_input_metadata": parsed["_metadata"],
               "prepared_only": prepare_only, "completed": False, "restarted": restart is not None}
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "prepared-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if prepare_only:
        return receipt
    if restart is not None:
        info = restore_restart(restart, state, cfg)
        prior_inputs = (info.run_trackers or {}).get("ideal_inputs")
        if prior_inputs != inputs:
            raise ValueError("ideal restart input bytes differ from the sounding/static authority saved in its checkpoint")
        restored = Fraction(str(info.elapsed_seconds)) * clock.tick_den
        if restored.denominator != 1 or int(restored) % clock.step_ticks:
            raise ValueError("ideal restart time does not lie on the fixed model clock")
        clock.ticks = int(restored)
        clock.step_count = clock.ticks // clock.step_ticks
        if clock.ticks > clock.run_ticks:
            raise ValueError("ideal restart is later than the requested final model time")
    refresh_model_time(state, clock)
    attrs = {
        "MAP_PROJ": np.int32(0), "MAP_PROJ_CHAR": "Cartesian",
        "START_DATE": _timestamp(start, 0), "SIMULATION_START_DATE": _timestamp(start, 0),
        "DT": np.float32(cfg.dt), "HYPSOMETRIC_OPT": np.int32(1),
        "HYBRID_OPT": np.int32(cfg.hybrid_opt), "ETAC": np.float32(cfg.etac),
        "GRID_ID": np.int32(cfg.grid_id), "PARENT_ID": np.int32(0),
        "CEN_LAT": np.float32(cfg.fire_lat_init), "CEN_LON": np.float32(cfg.fire_lon_init),
        **wrf_physics_selector_attrs(cfg), **driver.fire.output_attributes()}
    attrs.update({name: np.float32(value) if isinstance(value, float)
                  else np.int32(value) if isinstance(value, int) else value
                  for name, value in preparation.get("global_attrs", {}).items()})
    histories = {}
    checkpoints = {}

    def snapshot():
        frame = state_frame(state, include_diagnostic_pressure=True)
        frame.update({name: cp.asnumpy(cp.asarray(value)) for name, value in initial_fields.items()})
        for name, value in frame.items():
            if not bool(cp.isfinite(cp.asarray(value)).all().item()):
                raise FloatingPointError(f"nonfinite ideal output {name} at {clock.elapsed_seconds}s")
        valid = _timestamp(start, clock.elapsed_seconds)
        output = outdir / f"wrfout_d{cfg.grid_id:02d}_{valid}"
        with WrfoutWriter(output, nx=cfg.nx, ny=cfg.ny, nz=cfg.nz,
                          dx=cfg.dx, dy=cfg.dy, global_attrs=attrs,
                          title="Coupled SFIRE ideal", field_schema=frame,
                          soil_layers=soil_layer_count(cfg), engine="rust") as writer:
            writer.write_frame(valid, frame)
        histories[output.name] = _sha256(output)

    cp.cuda.Stream.null.synchronize()
    began = time.perf_counter()
    snapshot()
    while not clock.at_stop_time:
        refresh_model_time(state, clock, kernel_launch=True)
        dycore.step(state, cfg)
        refresh_model_time(state, clock, after_step=True)
        clock.advance()
        if clock.history_due() or clock.at_stop_time:
            snapshot()
        if clock.restart_due():
            destination = outdir / f"gpuwmrst_d{cfg.grid_id:02d}_{_timestamp(start, clock.elapsed_seconds)}.npz"
            write_restart(destination, state, cfg, run_trackers={"ideal_inputs": inputs})
            checkpoints[destination.name] = _sha256(destination)
    cp.cuda.Stream.null.synchronize()
    health = dycore.stability_report(state, cfg)
    receipt.update(completed=True, final_time_seconds=clock.elapsed_seconds,
                   step_count=clock.step_count, wall_seconds=time.perf_counter() - began,
                   health=health, fire=driver.fire.metadata(),
                   histories_sha256=histories, checkpoints_sha256=checkpoints,
                   device=cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"].decode())
    (outdir / "run-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (outdir / "run.done").write_text("0\n")
    return receipt


def _arguments(parser):
    parser.add_argument("config", type=Path, help="RunConfig TOML or native namelist.input with --namelist")
    parser.add_argument("--namelist", action="store_true", help="read native WRF ideal namelist.input")
    parser.add_argument("--sounding", type=Path, default=None, help="native input_sounding (default: input directory)")
    parser.add_argument("--input-directory", type=Path, default=None, help="native input files (default: configuration directory)")
    parser.add_argument("--outdir", type=Path, default=Path("out/fire-ideal"))
    parser.add_argument("--run-seconds", type=float, default=None)
    parser.add_argument("--restart", type=Path, default=None, help="restore the byte-exact fire and atmosphere checkpoint")
    parser.add_argument("--prepare-only", action="store_true", help="initialize the complete ideal state and write its receipt")


def _run(args):
    directory = args.config.resolve().parent if args.input_directory is None else args.input_directory
    sounding = directory / "input_sounding" if args.sounding is None else args.sounding
    receipt = run_ideal(args.config, sounding=sounding, input_directory=directory,
                        outdir=args.outdir, namelist=args.namelist,
                        run_seconds=args.run_seconds, restart=args.restart,
                        prepare_only=args.prepare_only)
    print(json.dumps({"outdir": str(args.outdir), "completed": receipt["completed"],
                      "prepared_only": receipt["prepared_only"],
                      "completed_seconds": receipt.get("final_time_seconds", 0),
                      "history_count": len(receipt.get("histories_sha256", {}))}))
    return 0


def register_cli(subparsers):
    parser = subparsers.add_parser("fire-ideal", help="initialize and run coupled SFIRE from a native sounding")
    _arguments(parser)
    parser.set_defaults(func=_run)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    _arguments(parser)
    args = parser.parse_args(argv)
    from woof import capabilities
    try:
        capabilities.require_for_command("fire-ideal")
        return _run(args)
    except capabilities.CapabilityMissing as error:
        import sys
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
