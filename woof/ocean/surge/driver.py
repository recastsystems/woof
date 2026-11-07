"""Orchestration only. Rust owns field I/O; CUDA owns all time stepping."""
from __future__ import annotations

import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


class Parameters(C.Structure):
    _fields_ = [(n, C.c_int) for n in ("nx", "ny", "fx", "fy", "nt", "ng", "boundary_mask")] + [
        (n, C.c_double) for n in ("west", "south", "dlon", "dlat", "fwest", "fsouth", "fdlon", "fdlat",
                                 "duration", "max_dt", "output_interval")] + [
        (n, C.c_float) for n in ("dry", "manning", "cfl", "reference_pressure", "initial_level")]


def _native(name: str, variable: str) -> Path:
    """The bundled ``rw_bathy`` through the artifact ladder every door uses.

    The override, the packaged and user bridge directories, then the
    tools/rw-bathy build; staged from this release's bundle on first use
    when absent (it is on demand).  Before 2.9.4 this searched only PATH,
    the checkout build and the packaged directory, so an installed surge
    door never found the binary ``woof fetch-bridges`` stages.
    """
    from woof import bridges
    filename = bridges.executable_name(name)

    def find():
        return bridges.find_artifact(variable, filename, crate_relative="tools/rw-bathy")

    found = find()
    if found is None and bridges.fetch_on_demand(name):
        found = find()
    if found is not None:
        return Path(found).resolve()
    raise FileNotFoundError(
        f"Missing {name}; run woof fetch-bridges --partial --artifact {name}, build tools/rw-bathy "
        f"with cargo build --release --locked --offline, or set {variable}")


def _call(command, *, log: Path, timeout=1800):
    with log.open("w") as stream:
        subprocess.run([str(x) for x in command], check=True, stdout=stream, stderr=subprocess.STDOUT,
                       timeout=timeout)


def prepare(config: Path, out: Path) -> Path:
    """Prepare native binary transfer buffers from local public-source NetCDF."""
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    bridge = _native("rw_bathy", "WOOF_RW_BATHY")
    _call([bridge, "prepare", Path(config).resolve(), out], log=out / "prepare.log")
    return out / "prepared.json"


def _library(build: Path) -> Path:
    explicit = os.environ.get("WOOF_SURGE_LIBRARY")
    if explicit:
        p = Path(explicit).resolve()
        if not p.is_file():
            raise FileNotFoundError(p)
        return p
    source = Path(__file__).parent / "kernels" / "surge.cu"
    from woof.hip_backend import active,compile_shared
    if active(): return compile_shared(source.read_text())
    nvcc = shutil.which("nvcc")
    if not nvcc:
        raise FileNotFoundError("nvcc is required to build the surge CUDA driver")
    identity = hashlib.sha256(source.read_bytes() + subprocess.check_output([nvcc, "--version"])).hexdigest()[:20]
    build.mkdir(parents=True, exist_ok=True)
    library = build / f"surge-{identity}.so"
    if not library.exists():
        # Compile for the installed GPU; cache identity includes nvcc and source.
        _call([nvcc, "-O3", "-std=c++17", "--shared", "-Xcompiler=-fPIC", "-arch=native",
               source, "-o", library], log=build / "cuda-build.log")
    return library


def run(config: Path, out: Path, *, render: bool = True) -> dict:
    """Run the complete native door and publish its stage receipt last."""
    started = time.monotonic()
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Prepared buffers are request owned. A new request never silently reuses
    # bathymetry or forcing from an earlier grid or cycle.
    prepared = prepare(config, out / "prepared")
    prep_end = time.monotonic()
    meta = json.loads(prepared.read_text())
    identity = hashlib.sha256(prepared.read_bytes()).hexdigest()[:16]
    library = _library(out / "build")
    build_end = time.monotonic()
    import numpy as np  # binary views only; all field processing is native
    p = Parameters(**meta["parameters"])
    arrays = [np.memmap(prepared.parent / meta[k], dtype=dtype, mode="r")
              for k, dtype in [("bed", "<f4"), ("forcing", "<f4"), ("times", "<f8"),
                               ("tides", "<f8"), ("gauge_cells", "<i4")]]
    native = C.CDLL(str(library))
    native.surge_abi.restype = C.c_int
    if native.surge_abi() != 1:
        raise RuntimeError("surge CUDA driver ABI mismatch")
    native.surge_run.argtypes = [C.POINTER(Parameters)] + [C.c_void_p] * 5 + [C.c_char_p, C.c_char_p, C.c_int]
    native.surge_run.restype = C.c_int
    error = C.create_string_buffer(2048)
    raw = out / "raw"
    raw.mkdir(exist_ok=True)
    solver = dict(cuda_source_sha256=hashlib.sha256(
        (Path(__file__).parent / "kernels" / "surge.cu").read_bytes()).hexdigest(),
        native_library_sha256=hashlib.sha256(library.read_bytes()).hexdigest())
    (raw / "solver.json").write_text(json.dumps(solver, indent=2) + "\n")
    rc = native.surge_run(C.byref(p), *[C.c_void_p(a.ctypes.data) for a in arrays],
                          os.fsencode(raw), error, len(error))
    if rc:
        raise RuntimeError(f"surge CUDA driver: {error.value.decode(errors='replace')}")
    solve_end = time.monotonic()
    bridge = _native("rw_bathy", "WOOF_RW_BATHY")
    _call([bridge, "finish", prepared, raw, out], log=out / "finish.log")
    finish_end = time.monotonic()
    images = _render(out, raw, identity, render)
    render_end = time.monotonic()
    receipt = json.loads((out / "native-receipt.json").read_text())
    receipt.update(**solver, schema="gpuwm.surge-run.v1", config=str(Path(config).resolve()),
                   prepared_contract=meta, images=images,
                   stages=[dict(name=n, wall_seconds=b-a, parallelism=parallelism)
                           for n, a, b, parallelism in [
                               ("prepare", started, prep_end, "Rust Rayon"),
                               ("cuda_build", prep_end, build_end, "nvcc host compiler"),
                               ("solve", build_end, solve_end, receipt["parallelism"]),
                               ("netcdf_and_gauges", solve_end, finish_end, "Rust Rayon"),
                               ("render", finish_end, render_end, "rw_wrfbatch Rayon")]],
                   wall_seconds=render_end-started)
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt

def _render(out: Path, raw: Path, identity: str, enabled: bool) -> list[str]:
    images = []
    if enabled:
        from woof import rustwx, render_layout
        renderer = rustwx.find_renderer()
        if renderer is None:
            raise FileNotFoundError("rw_wrfbatch is required for surge maps")
        for field in ("max_surge", "max_inundation_depth", "water_level"):
            paths, failures, skipped = rustwx.run_renderer(
                renderer, out / "surge.nc", store_root=out / "render-store" / identity / field,
                out_dir=out / "maps",
                products=f"var:wrf_{field}", frames="all" if field == "water_level" else str(len(json.loads((raw / "frames.json").read_text())) - 1),
                source_label="GPU nonlinear shallow water", width=1600, height=1250)
            # The generic importer reports unavailable atmospheric volumes
            # even when all requested surface planes rendered successfully.
            # They are not requested surge products and need no fake levels.
            requested_skips = [row for row in skipped if row[0] != "pressure-level products"]
            if failures or requested_skips or not paths:
                raise RuntimeError(f"surge map {field}: failures={failures}; skipped={skipped}")
            for path in paths:
                parts = render_layout.parse_engine_output(Path(path).name)
                if parts is None:
                    raise RuntimeError(f"renderer returned an unrecognized output name: {path}")
                domain, _, day = parts
                placed, note = render_layout.deliver(out / "maps", path,
                                                     domain=domain, product=field, day=day)
                if Path(placed).parent == out / "maps":
                    raise RuntimeError(f"surge map could not be filed: {note}")
                images.append(str(placed))
    return images


def finish(config: Path, out: Path, *, render: bool = True) -> dict:
    """Finish a completed native run, including recovery after a renderer failure.

    Recheck source hashes and controls before using archived GPU outputs.
    A new atmosphere, bathymetry, grid or tide must never borrow an old solve.
    """
    started = time.monotonic()
    out = Path(out).resolve()
    prepared = out / "prepared" / "prepared.json"
    meta = json.loads(prepared.read_text())
    checked = json.loads(prepare(config, out / "checked-inputs").read_text())
    keys = ("parameters", "grid", "epoch_start", "bathymetry", "atmosphere", "tides_enabled", "gauges")
    if any(meta[key] != checked[key] for key in keys):
        raise ValueError("completed native output belongs to different inputs or controls; run this request afresh")
    # Compare every transferred buffer too: variable selections can change
    # fields even when the source NetCDF file's hash stays the same.
    for key in ("bed", "forcing", "times", "tides", "gauge_cells"):
        if (prepared.parent / meta[key]).read_bytes() != (out / "checked-inputs" / checked[key]).read_bytes():
            raise ValueError(f"completed native output belongs to different {key}; run this request afresh")
    raw = out / "raw"
    native = json.loads((raw / "native-receipt.json").read_text())
    if (raw / "solver.json").is_file():
        native.update(json.loads((raw / "solver.json").read_text()))
    if native["duration_seconds"] != meta["parameters"]["duration"]:
        raise ValueError("native run is incomplete; postprocessing cannot replace the missing integration")
    bridge = _native("rw_bathy", "WOOF_RW_BATHY")
    _call([bridge, "finish", prepared, raw, out], log=out / "finish.log")
    finished = time.monotonic()
    identity = hashlib.sha256(prepared.read_bytes()).hexdigest()[:16]
    images = _render(out, raw, identity, render)
    ended = time.monotonic()
    native.update(schema="gpuwm.surge-run.v1", config=str(Path(config).resolve()),
                  prepared_contract=meta, images=images, recovered_postprocessing=True,
                  stages=[dict(name="prepare", wall_seconds=meta["wall_seconds"], parallelism="Rust Rayon"),
                          dict(name="solve", wall_seconds=native["native_wall_seconds"], parallelism=native["parallelism"]),
                          dict(name="checked_netcdf_and_gauges", wall_seconds=finished-started, parallelism="Rust Rayon"),
                          dict(name="render", wall_seconds=ended-finished, parallelism="rw_wrfbatch Rayon")],
                  postprocessing_wall_seconds=ended-started,
                  wall_seconds=meta["wall_seconds"]+native["native_wall_seconds"]+ended-started)
    (out / "receipt.json").write_text(json.dumps(native, indent=2) + "\n")
    return native
