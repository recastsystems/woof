"""Rust NetCDF decode and GPU consumption of one paired verification day.

The request names local forecasts and either aligned or raw GFS analyses.
No Python NetCDF decoder or CPU field-arithmetic fallback is provided.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import math
from pathlib import Path
import subprocess
import tempfile
import time

from .stochastic_verification import forecast_fields_gpu, score_pair_gpu, utc_time


@contextmanager
def decoded_transport(path, names, *, reader, work, audit):
    """Validate the existing Rust ABI and own its temporary flat payloads.

    One Rust invocation decodes the requested variables concurrently.
    Yield paths and metadata without loading or calculating weather fields.
    Payloads remain valid only inside the context, and are deleted on exit.
    """
    names = list(dict.fromkeys(names))
    started = time.perf_counter()
    row = dict(source=str(path), parallelism="Rust concurrent variables/chunks",
               observed_threads=None, decoded_files_deleted=[])
    audit.append(row)
    with tempfile.TemporaryDirectory(prefix="stochastic-day-", dir=work) as directory:
        folder = Path(directory)
        command = [str(reader), "dump", str(path), str(folder), *names]
        row["command"] = command
        try:
            before = time.perf_counter()
            result = subprocess.run(command, capture_output=True, text=True, timeout=900)
            row.update(decode_wall_seconds=time.perf_counter()-before,
                       exit_code=result.returncode, stderr=result.stderr[-8192:])
            if result.returncode:
                raise RuntimeError(f"Rust NetCDF decode failed: {result.stderr.strip()}")
            meta = json.loads((folder/"metadata.json").read_text())
            records = meta.get("variables", [])
            if (meta.get("schema") != "gpuwm-rw-netcdf-dump-v1"
                    or len(records) != len(names) or {r["name"] for r in records} != set(names)):
                raise ValueError("Rust decoded-field schema or variable roster differs")
            payloads = []
            for record in records:
                filename = record["filename"]
                if not isinstance(filename, str) or Path(filename).name != filename:
                    raise ValueError("decoded payload must be inside its temporary directory")
                payload = folder/filename
                if payload.is_symlink() or not payload.is_file():
                    raise ValueError("decoded payload must be a regular file")
                shape, dims = record["shape"], record["dimensions"]
                if (len(shape) != len(dims) or len(set(dims)) != len(dims)
                        or any(type(n) is not int or n <= 0 for n in shape)):
                    raise ValueError("decoded field has invalid dimensions")
                dtype = record["dtype"]
                if dtype not in ("<f8", "|S1"):
                    raise ValueError("unsupported decoded field dtype")
                if payload.stat().st_size != math.prod(shape)*(8 if dtype == "<f8" else 1):
                    raise ValueError("decoded payload byte count differs from shape")
                if dtype == "<f8":
                    if not record.get("cf", {}).get("applied"):
                        raise ValueError("Rust reader did not apply CF unpacking/missing values")
                payloads.append((payload, record))
            row["reader_provenance"] = meta.get("metadata")
            yield payloads
        finally:
            row["decoded_files_deleted"] = [dict(path=str(p), bytes=p.lstat().st_size)
                for p in folder.iterdir() if p.is_file() or p.is_symlink()]
            row["transport_lifetime_wall_seconds"] = time.perf_counter()-started


def decode_gpu(path, names, *, reader, work, audit):
    """Decode through Rust and upload numeric fields directly with CuPy."""
    import cupy as cp
    started = time.perf_counter()
    try:
        with decoded_transport(path, names, reader=reader, work=work, audit=audit) as payloads:
            audit[-1]["parallelism"] += "; GPU transfer"
            fields = {}
            for payload, record in payloads:
                values = (cp.fromfile(payload, dtype="<f8").reshape(record["shape"])
                          if record["dtype"] == "<f8" else payload.read_bytes())
                fields[record["name"]] = (values, record)
            return fields
    finally:
        if audit:
            audit[-1]["decode_and_transfer_wall_seconds"] = time.perf_counter()-started


def field(fields, name, dimensions, units):
    """Reorder by named axes on the GPU; accept only one Time record."""
    values, record = fields[name]
    if record.get("units") not in units:
        raise ValueError(f"{name}: expected units {units}, got {record.get('units')!r}")
    dims = list(record["dimensions"])
    if "Time" in dims:
        axis = dims.index("Time")
        if values.shape[axis] != 1:
            raise ValueError(f"{name}: one history record is required")
        values = values.squeeze(axis=axis)
        dims.pop(axis)
    if set(dims) != set(dimensions) or len(dims) != len(dimensions):
        raise ValueError(f"{name}: expected dimensions {dimensions}, got {dims}")
    return values.transpose(tuple(dims.index(d) for d in dimensions))


def check_clock(fields, name, valid):
    values, record = fields[name]
    if record["dtype"] == "|S1":
        if record["shape"] != [1, record["shape"][-1]]:
            raise ValueError("clock requires one timestamp record")
        text = values.decode("ascii").strip("\x00 ")
        # MPAS xtime is a UTC calendar without an offset, by its file contract.
        instant = utc_time(text.replace("_", "T")+"Z")
    else:
        times = record.get("times", [])
        if len(times) != 1:
            raise ValueError("clock requires one Rust-decoded CF timestamp")
        instant = utc_time(times[0])
    if instant != utc_time(valid):
        raise ValueError("embedded input time differs from verification valid time")


def score_day(request, *, reader, work, audit=None, grib_bridge=None):
    """Consume native-cell forecasts and a shared GFS analysis.

    GFS may be aligned NetCDF or regular global GRIB remapped on the GPU.
    The request supplies a mass-level geopotential-height variable in its
    geometry file, or a declared MPAS constant-gravity interface height.
    Cell IDs and embedded times are checked in every input frame.
    """
    import cupy as cp
    started = time.perf_counter()
    if request.get("schema") != "hex-stochastic-day-v1":
        raise ValueError("unsupported stochastic day request")
    control, stochastic = request["control"], request["stochastic"]
    if (len(control) < 2 or set(control) != set(stochastic)
            or not all(isinstance(m, str) and m for m in control)):
        raise ValueError("paired day requires identical member identities")
    if request["analysis"].get("source") != "GFS":
        raise ValueError("paired day requires a GFS analysis")
    if type(request["lead_hours"]) is not int or request["lead_hours"] < 0:
        raise ValueError("lead_hours must be a nonnegative integer")
    if audit is None:
        audit = []
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    def read(path, names):
        return decode_gpu(path, names, reader=reader, work=work, audit=audit)
    def ids(data):
        result = field(data, "indexToCellID", ("nCells",), (None, "unitless", "1", "-"))
        if not bool(cp.all(cp.isfinite(result) & (result > 0) & (result == cp.floor(result)))):
            raise ValueError("cell identities must be finite positive integers")
        return result
    # The RRFS fork's history and diagnostics streams carry no indexToCellID: a request may
    # declare that its forecasts are written in the geometry file's own cell order.
    mesh_order = request.get("cell_order") == "mesh"

    def aligned(data):
        if mesh_order and "indexToCellID" not in data:
            check_clock(data, "xtime", request["valid_time"])
            return
        actual = ids(data)
        if actual.shape != cell_ids.shape or not bool(cp.all(actual == cell_ids)):
            raise ValueError("input cell identities/order differ from verification mesh")
        check_clock(data, "xtime", request["valid_time"])
    geometry = request["geometry"]
    height_name = geometry["height_variable"]
    grib_analysis = request["analysis"].get("format") == "gfs_grib2"
    if request["analysis"].get("format", "native_netcdf") not in ("native_netcdf", "gfs_grib2"):
        raise ValueError("unsupported GFS analysis format")
    data = read(geometry["path"], [height_name, "areaCell", "indexToCellID"]
                + (["latCell", "lonCell"] if grib_analysis else []))
    cell_ids = ids(data)
    if int(cp.unique(cell_ids).size) != cell_ids.size:
        raise ValueError("geometry cell identities must be unique")
    area = field(data, "areaCell", ("nCells",), ("m2", "m^2", "m**2"))
    if grib_analysis:
        latitude, longitude = [field(data, name, ("nCells",), ("radian", "radians", "rad"))
                               for name in ("latCell", "lonCell")]
        if latitude.shape != area.shape or longitude.shape != area.shape:
            raise ValueError("geometry coordinates must match cell areas")
    height_kind = geometry["height_kind"]
    if height_kind == "geopotential_mass":
        height = field(data, height_name, ("nCells", "nVertLevels"), ("m", "gpm"))
    elif height_kind == "mpas_constant_gravity_interfaces":
        # Native-free init writes geometric interface height as "m MSL"
        # (vertical_spec.py); both labels denote metres before conversion.
        height = field(data, height_name, ("nCells", "nVertLevelsP1"), ("m", "m MSL"))
        gravity = geometry["model_gravity_m_s2"]
        if not isinstance(gravity, (int, float)) or not math.isfinite(gravity) or gravity <= 0:
            raise ValueError("MPAS height conversion requires positive model gravity")
        height = .5*(height[:, :-1]+height[:, 1:])*(gravity/9.80665)
    else:
        raise ValueError("height_kind must declare geopotential mass or MPAS constant-gravity interfaces")
    del data
    arms = []
    # 2 m temperature is "t2" on the ArWen route's history and "t2m" in the MPAS
    # suite route's (the hex default) history and diagnostics streams. THE BREAKAGE
    # THIS PREVENTS: every MPAS-route ensemble failed scoring with "variable not
    # found ... t2" (feat/stochastic-physics, 2026-10-29).
    t2_name = request.get("t2_variable", "t2")
    if t2_name not in ("t2", "t2m"):
        raise ValueError("t2_variable must be t2 (WOOF route) or t2m (MPAS suite route)")
    for roster in (control, stochastic):
        members = {}
        for member, spec in roster.items():
            # A separate diagnostics file may hold t2, as on native MPAS.
            id_names = [] if mesh_order else ["indexToCellID"]
            data = read(spec["path"], ["pressure", *id_names, "xtime"]
                        + ([] if spec.get("diagnostics") else [t2_name]))
            aligned(data)
            pressure = field(data, "pressure", ("nCells", "nVertLevels"), ("Pa",))
            surface = data
            if spec.get("diagnostics"):
                surface = read(spec["diagnostics"], [t2_name, *id_names, "xtime"])
                aligned(surface)
            t2 = field(surface, t2_name, ("nCells",), ("K",))
            members[member] = forecast_fields_gpu(pressure, height, t2)
            del data, surface, pressure, t2
        arms.append(members)
    spec = request["analysis"]
    if grib_analysis:
        from .stochastic_gfs import analysis_gpu
        if spec.get("terrain_mask_variable"):
            raise ValueError("GRIB analysis terrain support comes from its surface pressure")
        analysis, analysis_id, provenance = analysis_gpu(spec, request["valid_time"], latitude, longitude,
            work=work, audit=audit, bridge=grib_bridge)
        shared_mask = "GFS source-corner surface pressure >=50000 Pa"
    else:
        data = read(spec["path"], ["z500", "t2m", "indexToCellID", "xtime"])
        aligned(data)
        analysis = {name: field(data, name, ("nCells",), (unit,))
                    for name, unit in (("z500", "m"), ("t2m", "K"))}
        analysis_id, provenance = spec["analysis_id"], spec["remapping_provenance"]
        shared_mask = spec.get("terrain_mask_variable")
    if not grib_analysis and shared_mask:
        # Only an explicitly supplied, shared terrain mask can remove
        # below-ground cells from the observation support before scoring.
        mask_data = read(spec["path"], [spec["terrain_mask_variable"]])
        mask = field(mask_data, spec["terrain_mask_variable"], ("nCells",), ("1", "unitless"))
        if mask.shape != area.shape or not bool(cp.all((mask == 0) | (mask == 1))):
            raise ValueError("shared z500 terrain mask must contain only 0 and 1 per cell")
        analysis["z500"] = cp.where(mask == 1, analysis["z500"], cp.nan)
    result = score_pair_gpu(*arms, analysis, area, lead_hours=request["lead_hours"],
                            valid_time=request["valid_time"], analysis_id=analysis_id)
    result.update(input_reads=audit, consumer_wall_seconds=time.perf_counter()-started,
                  height_convention=geometry, shared_terrain_mask=shared_mask,
                  gfs_remapping_provenance=provenance,
                  qualification_complete=False)
    return result
