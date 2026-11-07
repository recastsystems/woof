"""Spectral wave door. Python orchestrates Rust boundaries and CUDA kernels."""
from __future__ import annotations
import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from .model import SpectralGrid, WaveModel

PACKAGE = Path(__file__).resolve().parent


def register_ocean_cli(commands):
    """The ordinary ocean door and python -m door share the same runner."""
    parser=commands.add_parser("waves",help="spectral waves from prepared inputs")
    sub=parser.add_subparsers(dest="waves_command",required=True)
    run_parser=sub.add_parser("run",help="run a prepared spectral wave configuration")
    run_parser.add_argument("config",type=Path)
    run_parser.set_defaults(func=lambda args:run(args.config))
    return sub

def native_tools():
    root = Path(os.environ.get("WOOF_WAVES_IO_DIR", PACKAGE / "native" / "bin"))
    binary, library = root / "rw_waves_io", root / "libwaves_io.so"
    if not binary.is_file() or not library.is_file():
        raise FileNotFoundError(f"wave Rust I/O is not built in {root}; run bash {PACKAGE / 'native/build.sh'} {root}, or set WOOF_WAVES_IO_DIR to the built directory")
    return binary, library

def validate_config(config):
    for key in ("forcing", "bathymetry", "output"):
        if not isinstance(config.get(key), str) or not config[key]:
            raise ValueError(f"configuration needs {key}")
    for key, default in (("dt_seconds", 60), ("output_seconds", 3600)):
        value = config.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if "CC-BY-4.0" in config.get("input_licenses", []) and "CC-BY-4.0" not in config.get("research_data_licenses", []):
        raise ValueError("ERA5 is CC-BY-4.0. This research run must explicitly name it in research_data_licenses under the workspace data-licence rule")
    SpectralGrid(**config.get("spectral", {}))
    cells = config.get("capture_cells", [])
    if not isinstance(cells, list) or any(
        not isinstance(c, dict) or set(c) != {"id", "i", "j"}
        or not isinstance(c["id"], str) or not c["id"]
        or any(isinstance(c[k], bool) or not isinstance(c[k], int) or c[k] < 0
               for k in ("i", "j")) for c in cells
    ) or len({c["id"] for c in cells}) != len(cells):
        raise ValueError("capture_cells needs unique ids and nonnegative integer i/j")

def run(config_path):
    candidate = Path(config_path)
    document={} if candidate.suffix.lower()==".toml" else json.loads(candidate.read_text())
    if candidate.suffix.lower() == ".toml" or "prepared" in document.get("waves",document):
        from .prepared_run import run as run_prepared
        return run_prepared(config_path)
    import cupy as cp
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    validate_config(config)
    resolve = lambda value: (config_path.parent / value).resolve()
    output = resolve(config["output"])
    # Repeated ordinary runs retain earlier evidence and receive a fresh folder.
    requested_output = output
    suffix = 0
    while True:
        try:
            output.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            suffix += 1
            output = requested_output.with_name(f"{requested_output.name}-{suffix:03d}")
    print(json.dumps({"output_directory": str(output)}), flush=True)
    receipt_path = output / "receipt.json"
    receipt = {"schema": "arwen.waves.run.v1", "status": "RUNNING", "config": config,
               "command": ["python", "-m", "woof.ocean.waves", "run", str(config_path)],
               "stages": {}, "frames": [], "single_core": [],
               "science": {"source_terms": "WAM cycle 4 wind/whitecapping, linear seeding, JONSWAP bottom friction, depth breaking, mirrored DIA", "initial_spectrum": "calm", "boundary": "zero incident action", "currents": "not supplied", "two_way_atmospheric_feedback": "separate coupling lane", "partition": "component phase speed < 1.2 projected U10 gives wind sea; remainder swell"},
               "source_sha256": {str(p.relative_to(PACKAGE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (PACKAGE / "kernels.cu", PACKAGE / "model.py", PACKAGE / "native/io.rs", PACKAGE / "cli.py")}}
    start = time.monotonic()
    try:
        binary, library_path = native_tools()
        packed = output / "prepared"
        stage = time.monotonic()
        subprocess.run([str(binary), "pack", str(resolve(config["forcing"])),
                        str(resolve(config["bathymetry"])), str(packed)], check=True, timeout=600)
        receipt["stages"]["prepare"] = {"wall_seconds": time.monotonic() - stage, "parallelism": "Rust scoped workers; NetCDF-C boundary serialized", "workers": int(os.environ.get("WAVES_IO_THREADS", 8))}
        manifest = json.loads((packed / "manifest.json").read_text())
        ny, nx, nt = manifest["ny"], manifest["nx"], manifest["nt"]
        # Byte loading and transfer only. Reorder, units, masks, interpolation and
        # NetCDF serialization are native. No Python field arithmetic.
        host = lambda name, dtype: np.frombuffer((packed / name).read_bytes(), dtype=dtype)
        latitude = host("latitude.f32", "<f4")
        longitude = host("longitude.f32", "<f4")
        epochs = host("time.f64", "<f8").tolist()
        winds_u = cp.asarray(host("U10.f32", "<f4")).reshape(nt, ny, nx)
        winds_v = cp.asarray(host("V10.f32", "<f4")).reshape(nt, ny, nx)
        depth = cp.asarray(host("depth.f32", "<f4")).reshape(ny, nx)
        stage = time.monotonic()
        model = WaveModel(depth, cp.asarray(latitude), cp.asarray(longitude),
                          spectral=SpectralGrid(**config.get("spectral", {})))
        cp.cuda.get_current_stream().synchronize()
        receipt["stages"]["initialize"] = {"wall_seconds": time.monotonic() - stage, "parallelism": "CUDA geometry and NVRTC compilation"}
        receipt["grid"] = {**manifest, "nf": model.nf, "nd": model.nd, "spectral_bins": model.spectrum.size, "first_hz": model.spectral.first_hz, "last_hz": float(model.frequency[-1]), "propagation_dt_limit_seconds": model.max_dt}
        lib = ctypes.CDLL(str(library_path))
        lib.waves_write.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p,
                                   ctypes.c_size_t, ctypes.c_size_t, ctypes.c_double,
                                   ctypes.c_void_p, ctypes.c_size_t]
        lib.waves_write.restype = ctypes.c_int
        capture_cells = config.get("capture_cells", [])
        if any(c["i"] >= nx or c["j"] >= ny for c in capture_cells):
            raise ValueError("capture_cells index lies outside the prepared forcing grid")
        if capture_cells:
            lib.waves_capture.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t]
            lib.waves_capture.restype = ctypes.c_int
            (output / "spectra").mkdir()
            receipt["spectral_captures"] = []
        last_dt = 0.0
        nc_path = output / "waves.nc"
        when, end, interval = epochs[0], epochs[-1], float(config.get("output_seconds", 3600))
        next_output, frame, forcing_index, steps = when, 0, 0, 0
        min_dt, min_limiter, max_growth_dt, io_wall = float("inf"), 1.0, 0.0, 0.0
        capture_wall = 0.0
        simulate_start = time.monotonic()
        cpu_start = time.process_time()
        while True:
            while forcing_index + 1 < nt - 1 and when >= epochs[forcing_index + 1] - 1e-7:
                forcing_index += 1
            i = forcing_index
            alpha = (when - epochs[i]) / (epochs[i + 1] - epochs[i])
            if when >= next_output - 1e-6 or when >= end - 1e-6:
                u = winds_u[i] * model.f(1 - alpha) + winds_u[i + 1] * model.f(alpha)
                v = winds_v[i] * model.f(1 - alpha) + winds_v[i + 1] * model.f(alpha)
                values = model.output(u, v)
                model.collect_timing()
                if not bool(cp.all(cp.isfinite(model.spectrum))) or float(cp.min(model.spectrum)) < -1e-10:
                    raise RuntimeError("wave spectrum became nonfinite or negative")
                stage = time.monotonic()
                host_values = cp.asnumpy(values)
                if lib.waves_write(os.fsencode(nc_path), latitude.ctypes.data,
                    longitude.ctypes.data, ny, nx, when, host_values.ctypes.data, frame):
                    raise RuntimeError("native wave NetCDF output failed; see native stderr")
                io_wall += time.monotonic() - stage
                entry = {"epoch": when, "valid": datetime.fromtimestamp(when, timezone.utc).isoformat(),
                         "max_hs_m": float(cp.nanmax(values[0])), "max_wind_ms": float(cp.max(cp.hypot(u, v))),
                         "min_cell_dia_limiter_last_step": float(cp.min(model.limiter)),
                         "steps_completed": steps, "simulation_wall_seconds": time.monotonic() - simulate_start,
                         "simulation_cpu_seconds": time.process_time() - cpu_start,
                         "gpu_seconds_by_stage": dict(model.timing),
                         "spectrum_variance_sum": float(cp.sum(model.spectrum * model.df[None,None,:,None]) * model.f(2*math.pi/model.nd)),
                         "allocated_pool_mib": cp.get_default_memory_pool().total_bytes() / 2**20}
                if capture_cells:
                    capture_start = time.monotonic()
                    capture = capture_spectra(model, capture_cells)
                    capture_path = output / "spectra" / f"frame-{frame:04d}.f32"
                    host_capture = cp.asnumpy(capture)
                    if lib.waves_capture(os.fsencode(capture_path),
                                         host_capture.ctypes.data, host_capture.size):
                        raise RuntimeError("native spectral capture failed")
                    receipt["spectral_captures"].append({
                        "frame": frame, "epoch": when, "path": str(capture_path),
                        "source_step_seconds": last_dt, "dtype": "little-endian float32",
                        "cells": capture_cells, "nf": model.nf, "nd": model.nd,
                        "layout": capture_layout(model.nf, model.nd),
                        "note": "Spectrum/moments at output; budgets and DIA rate/limiter from last split step. Budgets are increments; applied DIA rate is before the cell limiter. Unlimited DIA is evaluated at the captured state only. Frame zero has no source step."})
                    capture_wall += time.monotonic() - capture_start
                receipt["frames"].append(entry)
                print(json.dumps({"frame": frame, **entry}), flush=True)
                receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
                frame += 1
                next_output += interval
            if when >= end - 1e-6:
                break
            limit = min(float(config.get("dt_seconds", 60)), end - when,
                        next_output - when, epochs[i+1] - when)
            dt = model.step(winds_u[i], winds_v[i], winds_u[i+1], winds_v[i+1], alpha=alpha, dt_limit=limit)
            last_dt = dt
            min_dt = min(min_dt, dt)
            min_limiter = min(min_limiter, float(cp.min(model.limiter)))
            max_growth_dt = max(max_growth_dt, dt * float(cp.max(model.bounds)))
            steps += 1
            when += dt
            if steps % 100 == 0:
                model.collect_timing()
        model.collect_timing()
        receipt["stages"]["simulate"] = {"wall_seconds": time.monotonic() - simulate_start,
            "cpu_seconds": time.process_time() - cpu_start, "parallelism": "CUDA: spatial x frequency x direction, 128-thread blocks", "gpu_seconds_by_stage": model.timing,
            "steps": steps, "min_dt_seconds": min_dt, "min_dia_limiter": min_limiter, "max_growth_dt": max_growth_dt}
        receipt["stages"]["write"] = {"wall_seconds": io_wall, "parallelism": "Rust NetCDF container, one frame per output time"}
        receipt["stages"]["spectral_capture"] = {"wall_seconds": capture_wall, "parallelism": "CUDA gather; native Rust serialization of compact bytes"}
        receipt["gpu"] = {"device": cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"].decode(),
                          "peak_pool_mib": cp.get_default_memory_pool().total_bytes() / 2**20,
                          "process_memory_note": "See gpu-memory.csv for process VRAM including context"}
        receipt["status"] = "DONE"
        receipt["output"] = str(nc_path)
    except BaseException as error:
        receipt["status"] = "FAILED"
        receipt["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        receipt["wall_seconds"] = time.monotonic() - start
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return 0

def capture_layout(nf, nd):
    """Native little-endian payload layout, offsets measured in float32 words."""
    fields = [("frequency_hz", nf), ("df_hz", nf)]
    per_cell = [("spectrum_m2_per_hz_rad", nf * nd),
                ("local_source_increments_input_white_bottom_breaking", 4 * nf * nd),
                ("dia_rate_before_cell_limiter", nf * nd),
                ("dia_unlimited_at_captured_state", nf * nd), ("moments", 10),
                ("wavenumber_per_m", nf), ("depth_m", 1),
                ("wind_east_north_ms", 2), ("dia_cell_limiter", 1)]
    return {"shared": fields, "per_cell": per_cell,
            "cell_words": sum(count for _, count in per_cell)}


def capture_spectra(model, cells):
    """Gather requested device cells. No spectral transform executes on the CPU."""
    arrays = [model.frequency, model.df]
    raw_rate = model.cp.zeros_like(model.spectrum)
    model._launch("dia_unlimited", model.nc * model.ns, (model.spectrum, raw_rate,
        model.frequency, model.df, model.depth, model.moments, model.i(model.nc),
        model.i(model.nf), model.i(model.nd), model.f(math.log(model.spectral.ratio))))
    for point in cells:
        i, j = point["i"], point["j"]
        c = j * model.nx + i
        arrays.extend((model.spectrum[j, i].reshape(-1),
                       model.budgets[j, i].reshape(-1),
                       model.rate[j, i].reshape(-1), raw_rate[j, i].reshape(-1), model.moments[c],
                       model.cp.where(model.depth[j, i] > 0, model.k[c], 0),
                       model.depth[j:j+1, i:i+1].reshape(-1), model.wind[c],
                       model.limiter[c:c+1]))
    return model.cp.concatenate(arrays)


def render(source, output, valid):
    from woof import rustwx
    binary = rustwx.find_renderer()
    if binary is None:
        raise FileNotFoundError("rw_wrfbatch is required; set WOOF_RW_WRFBATCH to the built renderer")
    return rustwx.run_cf_renderer(binary, Path(source), variable="HS", out_dir=Path(output),
                                valid=valid, table=PACKAGE / "cf-products.json")

def main(argv=None):
    parser = argparse.ArgumentParser(description="CUDA spectral ocean waves with native Rust I/O")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("run"); p.add_argument("config", type=Path)
    p = commands.add_parser("render"); p.add_argument("source"); p.add_argument("output"); p.add_argument("--valid")
    for name in ("verify", "verify-wind"):
        p = commands.add_parser(name); p.add_argument("model"); p.add_argument("observations"); p.add_argument("station"); p.add_argument("latitude"); p.add_argument("longitude"); p.add_argument("output")
    args = parser.parse_args(argv)
    if args.command == "run":
        return run(args.config)
    if args.command == "render":
        return render(args.source, args.output, args.valid)
    binary, _ = native_tools()
    return subprocess.run([str(binary), args.command, args.model, args.observations,
        args.station, args.latitude, args.longitude, args.output], check=False, timeout=120).returncode
