"""Run the complete native em_fire physical setup through WOOF-ArWen's dycore.

The default uses ordinary engine arithmetic. Strict WRF arithmetic is an
explicit diagnostic control, recorded separately in the run receipt.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields, replace
from datetime import datetime, timedelta
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import struct
import time


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def first(document, section, key, default=None):
    values = document.get(section, {}).get(key, ())
    return values[0] if values else default


def build_configuration(reference: Path, run_seconds: float | None):
    from woof.config import RunConfig
    from woof.fortran_namelist import parse_namelist

    original = parse_namelist(reference / "namelist.input")
    resolved_path = reference / "namelist.output"
    resolved = parse_namelist(resolved_path) if resolved_path.is_file() else original
    field_defaults = {field.name: field.default for field in fields(RunConfig)}
    known = set(field_defaults)
    values = {}
    inactive_physics_defaults = {}
    # The complete native namelist output provides registry defaults. The
    # original explicit values remain authoritative where they are present.
    for document in (resolved, original):
        for section in ("physics", "dynamics", "fire"):
            values.update({key: entries[0] for key, entries in document.get(section, {}).items()
                           if key in known and entries and
                           (document is original or section != "physics" or
                            key in original.get("physics", {}) or key == "num_soil_layers")})
    # Every atmosphere-physics package is disabled in this official case.
    # Native registry defaults for inactive MYNN, RUC, microphysics and
    # radiation arms do not select those arms. Preserve their native values
    # as evidence without declaring an inactive scheme's options as active.
    inactive_physics_defaults = {key: entries[0] for key, entries in resolved.get("physics", {}).items()
                                 if key in known and entries and key not in original.get("physics", {})
                                 and key != "num_soil_layers"}
    normalizations = {}
    for key, value in tuple(values.items()):
        default = field_defaults[key]
        # Native namelist.output spells REAL defaults as their promoted
        # binary32 values. Equal stored REAL words are the same default;
        # preserve that identity for host-side scheme-scoping checks.
        if isinstance(value, float) and isinstance(default, float):
            if struct.pack("<f", value) == struct.pack("<f", default) and value != default:
                values[key] = default
                normalizations[key] = {"native_text_value": value, "host_default": default,
                                       "stored_real_word": struct.pack("<f", value).hex()}
    domains = original["domains"]
    exact_dt = (Fraction(int(domains["time_step"][0]))
                + Fraction(int(first(original, "domains", "time_step_fract_num", 0)),
                           int(first(original, "domains", "time_step_fract_den", 1))))
    if int(domains["max_dom"][0]) != 1:
        raise ValueError("this official ideal runner requires the complete single-domain em_fire case")
    control = original["time_control"]
    duration = sum(float(control.get(name, [0])[0]) * multiplier for name, multiplier in
                   (("run_days", 86400), ("run_hours", 3600), ("run_minutes", 60), ("run_seconds", 1)))
    window = duration if run_seconds is None else float(run_seconds)
    if not 0 < window <= duration or Fraction(str(window)) % exact_dt:
        raise ValueError("the precursor window must be positive, within the official duration and on the native timestep lattice")
    bdy = original["bdy_control"]
    native_tracer_opt = first(original, "dynamics", "tracer_opt",
                              first(resolved, "dynamics", "tracer_opt", 0))
    # Match the public typed namelist door: option 3 selects the shared
    # bulk row; this single-domain harness has only its emitting fire domain.
    if isinstance(native_tracer_opt, bool) or not isinstance(native_tracer_opt, int) or native_tracer_opt not in (0, 3):
        raise ValueError("native SFIRE tracer option must be the implemented integer 0 or 3")
    values["fire_smoke"] = native_tracer_opt == 3
    for axis in ("x", "y"):
        if not bool(bdy[f"open_{axis}s"][0]) or not bool(bdy[f"open_{axis}e"][0]):
            raise ValueError("the official coupled ideal case requires both open boundaries on each axis")
    values.update(nx=int(domains["e_we"][0]) - int(domains["s_we"][0]),
                  ny=int(domains["e_sn"][0]) - int(domains["s_sn"][0]),
                  nz=int(domains["e_vert"][0]) - int(domains["s_vert"][0]),
                  dx=float(domains["dx"][0]), dy=float(domains["dy"][0]),
                  ztop=float(domains["ztop"][0]), dt=float(exact_dt), run_seconds=window,
                  output_interval_s=float(control["history_interval_s"][0]),
                  sr_x=int(domains["sr_x"][0]), sr_y=int(domains["sr_y"][0]),
                  open_x=True, open_y=True, moist=True, moist_cq=True,
                  # share/input_wrf.F:1033-1042 overrides the pre-import
                  # HYPSOMETRIC_OPT metadata for every native ideal run.
                  hypsometric_opt=1,
                  top_lid=bool(first(resolved, "dynamics", "top_lid", False)),
                  fire_fuel_namelist=str((reference / "namelist.fire").resolve()))
    cfg = RunConfig(**values)
    if cfg.ifire != 2 or cfg.h_sca_adv_order != 5:
        raise ValueError("the native SFIRE and fifth-order horizontal scalar choices must remain active")
    return cfg, exact_dt, {"official_duration_seconds": duration, "requested_window_seconds": window,
                           "identical_registry_default_words": normalizations,
                           "inactive_native_physics_defaults": inactive_physics_defaults,
                           "hypsometric_override": "WRF share/input_wrf.F:1033-1042 forces ideal hypsometric_opt=1",
                           "native_use_theta_m": int(first(resolved, "dynamics", "use_theta_m", 1)),
                           "native_tracer_opt": native_tracer_opt,
                           "native_bulk_smoke": "tracer_opt=3 selects the canonical ug/kg-dry bulk row; native input and output use g/kg-air",
                           "native_khdif": float(first(original, "dynamics", "khdif", 0.05)),
                           "native_kvdif": float(first(original, "dynamics", "kvdif", 0.05)),
                           "constant_k_controls": "inactive under native km_opt=2",
                           "roughness_source": "WRF module_physics_init.F:1076,1225-1227 sets land Z0/ZNT=0.1 m",
                           "arwen_thermodynamic_variable": "dry potential temperature; native WRF moist-theta numerics are a separate comparison cause"}


def timestamp(start: datetime, seconds: float) -> str:
    valid = start + timedelta(seconds=seconds)
    return f"{valid.year:04d}-{valid.month:02d}-{valid.day:02d}_{valid.hour:02d}:{valid.minute:02d}:{valid.second:02d}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-seconds", type=float)
    parser.add_argument("--arithmetic", choices=("default", "wrf-exact"), default="default")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.arithmetic == "wrf-exact":
        for key in ("GPUWM_WRF_EXACT", "WOOF_WRF_EXACT_DIAGNOSTICS", "WOOF_WRF_EXACT_ADVECTION",
                    "GPUWM_WRF_EXACT_DIFFUSION", "GPUWM_WRF_EXACT_BIGSTEP"):
            os.environ[key] = "1"
    elif any(os.environ.get(key) == "1" for key in ("GPUWM_WRF_EXACT", "WOOF_WRF_EXACT_DIAGNOSTICS",
                                                   "WOOF_WRF_EXACT_ADVECTION", "GPUWM_WRF_EXACT_DIFFUSION",
                                                   "GPUWM_WRF_EXACT_BIGSTEP")):
        raise ValueError("default arithmetic run inherited strict WRF switches; clear them or select --arithmetic wrf-exact")
    args.reference = args.reference.resolve()
    args.out = args.out.resolve()
    if (args.out / "run-receipt.json").exists() or any(args.out.glob("wrfout*")):
        raise ValueError("use a new output directory; a completed or partial run must not be overwritten")
    args.out.mkdir(parents=True, exist_ok=True)
    from tools.sfire_coupled_ideal.initial_state import read_ideal_initial
    from woof.config import validate_run_config, soil_layer_count
    cfg, exact_dt, semantics = build_configuration(args.reference, args.run_seconds)
    restored = read_ideal_initial(args.reference / "wrfinput_d01", cfg)
    # A native ideal.exe base is stored per column. Allocate that exact
    # geometry before load_base rather than regenerate a hill or column.
    cfg = replace(cfg, terrain_opt=1,
                  hybrid_opt=int(restored.global_attributes["HYBRID_OPT"]),
                  etac=float(restored.global_attributes["ETAC"]))
    semantics["terrain_geometry"] = "native per-column PHB/PB/ALB/T_INIT, allocated as 3-D profiles"
    code_root = Path(__file__).resolve().parents[2]
    sources = {str(path.relative_to(code_root)): sha256(path)
               for path in sorted((code_root / "woof").rglob("*"))
               if path.is_file() and path.suffix in (".py", ".cu", ".cuh")}
    source_provenance_path = code_root / "SFIRE_SOURCE_PROVENANCE.json"
    source_provenance = None
    if source_provenance_path.is_file():
        source_provenance = json.loads(source_provenance_path.read_text())
        if source_provenance.get("engine_sources_sha256") != sources:
            raise ValueError("staged engine source manifest differs from the executed source snapshot")
    harness_sources = {str(path.relative_to(code_root)): sha256(path)
                       for path in sorted(Path(__file__).parent.glob("*.py"))}
    receipt = {"schema": "sfire-coupled-arwen-run-v1", "arithmetic": args.arithmetic,
               "configuration": asdict(cfg), "case_semantics": semantics,
               "sources_sha256": sources,
               "source_provenance": source_provenance,
               "harness_sources_sha256": harness_sources,
               "inputs_sha256": {name: sha256(args.reference / name)
                                  for name in ("namelist.input", "namelist.fire", "wrfinput_d01", "input_sounding")},
               "effective_mapfactor_sources": restored.global_attributes.get("IDEAL_EFFECTIVE_MAPFACTORS", {}),
               "prepared_only": args.prepare_only, "completed": False}
    (args.out / "prepared-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    validate_run_config(cfg)
    if args.prepare_only:
        print(json.dumps({"prepared": True, "nx": cfg.nx, "ny": cfg.ny, "nz": cfg.nz,
                          "dt": cfg.dt, "run_seconds": cfg.run_seconds, "arithmetic": args.arithmetic}))
        return

    import cupy as cp
    from woof.core import dycore
    from woof.core.clock import DomainTicks, TickClock
    from woof.core.state import refresh_model_time
    from woof.core.physics import initialize_physics
    from woof.core.diagnostics import update_diagnostics
    from woof.ingest.wrfinput import restore_domain_state
    from woof.io.wrfout import WrfoutWriter, state_frame
    from woof.wrf_exact import compile_receipt, install
    install()
    raw = restored.raw
    state = restore_domain_state(restored, cfg)
    update_diagnostics(state, cfg.hypsometric_opt)
    static = {name: raw[name] for name in ("NFUEL_CAT", "ZSF", "DZDXF", "DZDYF", "FMC_G", "LFN_HIST") if name in raw}
    # This is the native initializer's explicit roughness assignment, not
    # a configurable replacement for a missing real-case roughness field.
    static["ZNT"] = cp.full((cfg.ny, cfg.nx), cp.float32(0.1), cp.float32)
    native_surface = {target: raw[name] for name, target in
                      (("TSK", "tsk"), ("TMN", "tmn"), ("XLAND", "xland"),
                       ("LANDMASK", "landmask"), ("LU_INDEX", "ivgtyp"), ("ISLTYP", "isltyp"),
                       ("VEGFRA", "vegfra"), ("TSLB", "soil_temperature"), ("SMOIS", "soil_moisture"),
                       ("SH2O", "liquid_moisture"), ("GLW", "glw"), ("SWDOWN", "swdown")) if name in raw}
    driver = initialize_physics(state, cfg, fire_static_data=static, **native_surface)
    if driver.fire is None:
        raise RuntimeError("the native ifire=2 initializer did not attach its FireCoupler")
    start = datetime(1, 1, 1)
    tick_den = exact_dt.denominator
    step_ticks = exact_dt.numerator
    run_ticks = int(Fraction(str(cfg.run_seconds)) * tick_den)
    history_ticks = int(Fraction(str(cfg.output_interval_s)) * tick_den)
    spec = DomainTicks(grid_id=1, parent_id=0, parent_time_step_ratio=1, step_ticks=step_ticks,
                       dt_fp32=cp.asnumpy(cp.float32(cfg.dt)).item(), history_ticks=history_ticks,
                       restart_ticks=None, radt_ticks=None, stepra=None, cudt_ticks=None, stepcu=None,
                       bldt_ticks=None, stepbl=None)
    clock = TickClock(tick_den, run_ticks, start, (spec,)).domain_clock(1)
    refresh_model_time(state, clock)
    attributes = {name: value for name, value in restored.global_attributes.items() if not isinstance(value, dict)}
    attributes.update(START_DATE="0001-01-01_00:00:00", SIMULATION_START_DATE="0001-01-01_00:00:00",
                      DT=cp.asnumpy(cp.float32(cfg.dt)).item(), HYPSOMETRIC_OPT=cfg.hypsometric_opt,
                      HYBRID_OPT=cfg.hybrid_opt, IFIRE=cfg.ifire, SR_X=cfg.sr_x, SR_Y=cfg.sr_y)
    attributes.update(driver.fire.output_attributes())
    rows = []
    output_hashes = {}
    initial_coordinate_fields = ("DNW", "DN", "RDNW", "RDN", "FNM", "FNP", "C1H", "C2H", "C3H", "C4H",
                                 "C1F", "C2F", "C3F", "C4F", "MAPFAC_M", "MAPFAC_U", "MAPFAC_V",
                                 "MAPFAC_MX", "MAPFAC_MY", "MAPFAC_UX", "MAPFAC_UY", "MAPFAC_VX", "MAPFAC_VY")

    def snapshot():
        frame = state_frame(state, include_diagnostic_pressure=True)
        frame.update({name: cp.asnumpy(cp.asarray(raw[name], cp.float32))
                      for name in initial_coordinate_fields if name in raw})
        for name in ("XLAT", "XLONG"):
            if name in raw:
                frame[name] = cp.asnumpy(cp.asarray(raw[name], cp.float32))
        for name, values in frame.items():
            if not bool(cp.isfinite(cp.asarray(values)).all().item()):
                raise RuntimeError(f"nonfinite {name} at {clock.elapsed_seconds} seconds")
        valid = timestamp(start, clock.elapsed_seconds)
        output = args.out / f"wrfout_d01_{valid}"
        with WrfoutWriter(output, nx=cfg.nx, ny=cfg.ny, nz=cfg.nz, dx=cfg.dx, dy=cfg.dy,
                          title="WOOF-ArWen coupled SFIRE ideal verification", global_attrs=attributes,
                          field_schema=frame, soil_layers=soil_layer_count(cfg), engine="rust") as writer:
            writer.write_frame(valid, frame)
        output_hashes[output.name] = sha256(output)
        fire = driver.fire.grid
        interior = fire.interior
        row = {"time_seconds": clock.elapsed_seconds, "step_count": clock.step_count,
               "fire_clock_seconds": fire.time_seconds,
               "fire_area_m2": float(cp.sum(fire.data["fire_area"][interior].astype(cp.float64)).item() * fire.dx * fire.dy),
               "sensible_power_w": float(cp.sum(fire.data["fgrnhfx"][interior].astype(cp.float64)).item() * fire.dx * fire.dy),
               "latent_power_w": float(cp.sum(fire.data["fgrnqfx"][interior].astype(cp.float64)).item() * fire.dx * fire.dy),
               "health": dycore.stability_report(state, cfg)}
        rows.append(row)
        (args.out / "progress.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(json.dumps(row), flush=True)

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
    cp.cuda.Stream.null.synchronize()
    receipt.update(completed=True, final_time_seconds=clock.elapsed_seconds, final_ticks=clock.ticks,
                   step_count=clock.step_count, wall_seconds=time.perf_counter() - began,
                   fire_metadata=driver.fire.metadata(), fire_setup=driver.fire.setup_identity(),
                   output_sha256=output_hashes, series=rows, compile=compile_receipt(),
                   device=cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"].decode(),
                   cupy_version=cp.__version__)
    (args.out / "run-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (args.out / "run.done").write_text("0\n")


if __name__ == "__main__":
    main()
