"""Hourly global ensembles through the existing hex forecast door.

Each process sees one granted card. Multiple independent members can reside
on it; no forecast module globals or mutable physics buffers are shared.
NetCDF work is serial within each isolated export process. No numerical
kernel is changed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import hashlib
import importlib.util
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

from .errors import MpasPortError

SCHEMA = "gpuwm-hex.hourly-ensemble/v1"
FIELDS = ("theta", "rho", "qv", "qc", "qr", "qi", "qs", "qg", "pressure",
          "surface_pressure", "uReconstructZonal", "uReconstructMeridional", "u", "w")
ALIASES = {"u": "normal_u", "uReconstructZonal": "u_zonal",
           "uReconstructMeridional": "v_meridional"}


def history_name(dataset, target):
    """The history variable holding ``target`` on either physics route.

    The WOOF route's history names the winds ``normal_u``/``u_zonal``/``v_meridional`` (``ALIASES``); the MPAS
    suite route writes MPAS's own stream, which uses the Registry names themselves (``u``,
    ``uReconstructZonal``, ``uReconstructMeridional``).  THE BREAKAGE THIS PREVENTS: an MPAS-route ensemble's
    export refused with "forecast history lacks u_zonal".  An
    WOOF history carries the alias, so it reads exactly as before.
    """
    alias = ALIASES.get(target, target)
    return alias if alias in dataset.variables or target not in dataset.variables else target


class EnsembleRefusal(MpasPortError):
    pass


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def lts_run_facts(history):
    """Facts from this member's executor, never a packing calibration."""
    path = Path(history) / "fullstep-lts-receipt.json"
    record = json.loads(path.read_text())
    schema = record.get("schema")
    if schema == "gpuwm-hex.fullstep-lts-forecast/v1":
        # v1 is emitted only by _execute_single. Its class table describes
        # rate classes on one device, not device partitions.
        if record.get("dispatch") != "rate classes":
            raise EnsembleRefusal("unrecognized one-card LTS dispatch receipt")
        devices = [0]
    else:
        devices = record["devices"]
    if len(devices) != 1:
        raise EnsembleRefusal(f"LTS member used {len(devices)} resident cards; expected one")
    return {"integration_seconds": sum(row["seconds"] for row in record["macros"]),
            "integration_note": "sum of synchronized macro walls, including first macro; excludes construction, health and history IO",
            "lts_receipt": str(path), "resident_devices": devices,
            "lts_receipt_schema": schema,
            "memory_pool_peak_bytes_per_card": record.get("memory_pool_peak_bytes_per_card")}


def read_manifest(path, count=20, resolution=60, *, forecast_only=False):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA:
        raise EnsembleRefusal(f"{path}: expected {SCHEMA}; an untyped roster could mix cycles or meshes")
    if count < 1 or (count < 2 and not forecast_only):
        raise EnsembleRefusal("an ensemble needs at least two members to estimate sample variance")
    if data.get("resolution_km") != resolution:
        raise EnsembleRefusal(f"requested {resolution} km but manifest declares {data.get('resolution_km')} km")
    records = data.get("members", [])
    if len(records) < count:
        raise EnsembleRefusal(f"requested {count} distinct members, only {len(records)} supplied; duplicating members understates covariance rank")
    # The ensemble owns each member's product stage (frame products, member
    # exports), so the door's own end-of-forecast product flags are its too.
    controlled = {"--out", "--scratch", "--init", "--init-source", "--hours", "--history-every-minutes",
                  "--members", "--grid", "--static", "--mesh", "--start-time", "--preflight",
                  "--no-products", "--products-detached", "--no-products-detached"}
    def check_flags(flags, owner):
        if not isinstance(flags, list) or not all(isinstance(flag, str) for flag in flags):
            raise EnsembleRefusal(f"{owner} forecast_flags must be an argv list; a shell string could run anything")
        for flag in flags:
            key = flag.split("=",1)[0]
            if key in controlled or key.startswith("--samehost"):
                raise EnsembleRefusal(f"{owner} forecast flag {flag} would replace a member, its clock, output or single-card assignment")
        from .cli import build_parser
        import contextlib, io
        detail = io.StringIO()
        try:
            with contextlib.redirect_stderr(detail):
                build_parser().parse_args(["forecast", *flags])
        except SystemExit as error:
            raise EnsembleRefusal(f"{owner} forecast flags are not executable: {detail.getvalue().strip()}") from error
    check_flags(data.get("forecast_flags", []), "manifest")
    def resolve(value):
        p = Path(value)
        return str((path.parent / p).resolve()) if not p.is_absolute() else str(p.resolve())
    for key in ("grid", "static", "mesh_rows", "native_template"):
        data[key] = resolve(data[key])
        if not Path(data[key]).is_file():
            raise EnsembleRefusal(f"manifest {key} file missing: {data[key]}")
    datetime.strptime(data["start_time"], "%Y-%m-%d_%H:%M:%S")
    selected = []
    ids, paths = set(), set()
    for row in records[:count]:
        row = dict(row)
        name = row["id"]
        if (not isinstance(name, str) or not name or name in (".", "..") or
                any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for c in name)):
            raise EnsembleRefusal(f"unsafe member id {name!r}; it would escape or alias its output folder")
        row["init"] = resolve(row["init"])
        if name in ids or row["init"] in paths or not row.get("source"):
            raise EnsembleRefusal(f"member {name}: duplicate identity/input or absent meteorological provenance")
        prepare = row.get("prepare")
        if prepare is not None and (not isinstance(prepare, list) or not prepare
                                    or not all(isinstance(part, str) for part in prepare)):
            raise EnsembleRefusal(f"member {name}: prepare must be an argv list; a shell string could run anything")
        # A streamed member (prepare argv) is built just before its forecast,
        # so a 30 km ensemble never needs every 6 GB init on disk at once.
        if prepare is None and not Path(row["init"]).is_file():
            raise EnsembleRefusal(f"member {name} init missing: {row['init']}")
        # Perturbed-physics members: a member may add its own forecast-door
        # flags (--convection, --physics KEY=VALUE, --physics-namelist) after
        # the manifest's, so a multi-physics or perturbed-parameter ensemble
        # is manifest rows, not a second launcher.
        row["forecast_flags"] = row.get("forecast_flags", [])
        check_flags(row["forecast_flags"], f"member {name}")
        from .stochastic import member_options, member_number
        source_count = data.get("source_members", count)
        explicit = {}
        # Finish the shared scope before applying member overrides. A shared
        # argv option must not silently re-enable a member's control arm.
        for scope in (data, row):
            explicit.update(scope.get("stochastic", {}))
            flags = scope.get("forecast_flags", [])
            for i, flag in enumerate(flags):
                if flag == '--stochastic':
                    explicit.update(json.loads(flags[i+1]))
                elif flag.startswith('--stochastic='):
                    explicit.update(json.loads(flag.split('=',1)[1]))
        seed = explicit.pop("seed", data.get("stochastic_seed", 53))
        identity = explicit.pop("member", member_number(name))
        row["stochastic"] = member_options(identity, seed=seed,
            source_members=source_count, ensemble_size=count, **explicit)
        # Assets belong to this member's segment, not to an ensemble-wide
        # cell order. A later segment runner can replace this single segment.
        for key in ("mesh", "grid", "static", "mesh_rows", "native_template", "reference_init"):
            if key in row:
                row[key] = resolve(row[key]) if key != "mesh" else row[key]
                if key != "mesh" and not Path(row[key]).is_file():
                    raise EnsembleRefusal(f"member {name} {key} file missing: {row[key]}")
        ids.add(name); paths.add(row["init"])
        selected.append(row)
    data["members"] = selected
    return data


def checked_array(variable, key=Ellipsis):
    import numpy as np
    value = variable[key]
    if np.ma.isMaskedArray(value) and np.any(np.ma.getmaskarray(value)):
        raise EnsembleRefusal(f"{variable.name}: missing data would contaminate ensemble covariance")
    value = np.asarray(value)
    if value.dtype.kind in "fc" and not np.isfinite(value).all():
        raise EnsembleRefusal(f"{variable.name}: nonfinite member state")
    return value


def validate_inputs(data, members=None, state_hashes=None, reference_path=None):
    """Compare actual order, vertical grid and clocks, not just array shapes.

    ``members``/``state_hashes``/``reference_path`` let a streamed ensemble
    validate each member as it is prepared, against the same reference and
    the same running set of physical-state hashes.
    """
    import numpy as np
    from netCDF4 import Dataset
    from contextlib import ExitStack
    members = data["members"] if members is None else members
    if any(any(key in row for key in ("grid", "static")) for row in members):
        dimensions = []
        for row in members:
            local = dict(data, **{key: row.get(key, data[key]) for key in ("grid", "static")})
            plain = {key: value for key, value in row.items() if key not in ("grid", "static")}
            dims = validate_inputs(local, [plain], state_hashes, row.get("reference_init", row["init"]))
            row.update(plain)
            row["dimensions"] = dims
            dimensions.append(dims)
        return max(dimensions, key=lambda d: d["cells"] * d["levels"])
    with ExitStack() as stack:
        grid = stack.enter_context(Dataset(data["grid"]))
        static = stack.enter_context(Dataset(data["static"]))
        reference = stack.enter_context(Dataset(reference_path or members[0]["init"]))
        for name in ("indexToCellID", "indexToEdgeID"):
            expected = checked_array(grid[name])
            if len(np.unique(expected)) != expected.size:
                raise EnsembleRefusal(f"grid {name} is not a unique identity map")
            if not np.array_equal(expected, checked_array(static[name])):
                raise EnsembleRefusal(f"static {name} differs from grid order")
        nominal = float(checked_array(static["nominalMinDc"]).item()) / 1000
        if abs(nominal - data["resolution_km"]) > 0.1 * data["resolution_km"]:
            raise EnsembleRefusal(f"mesh nominalMinDc={nominal:g} km disagrees with ensemble resolution")
        state_hashes = set() if state_hashes is None else state_hashes
        for row in members:
            with Dataset(row["init"]) as ds:
                for name in ("indexToCellID", "indexToEdgeID"):
                    if not np.array_equal(checked_array(ds[name]), checked_array(grid[name])):
                        raise EnsembleRefusal(f"{row['id']}: {name} differs from forecast grid; permute through hex ordering first")
                for name in ("zgrid", "rho_base", "theta_base"):
                    if not np.array_equal(checked_array(ds[name]), checked_array(reference[name])):
                        raise EnsembleRefusal(f"{row['id']}: {name} differs; these members do not share a vertical/reference state")
                clock = b"".join(ds["xtime"][0]).decode().strip()
                if clock != data["start_time"]:
                    raise EnsembleRefusal(f"{row['id']}: xtime {clock} differs from cycle {data['start_time']}")
                fingerprint = hashlib.sha256()
                for name in ("theta", "rho", "qv", "qc", "qr", "u", "w", "surface_pressure"):
                    if ds[name].dimensions != reference[name].dimensions or ds[name].shape != reference[name].shape:
                        raise EnsembleRefusal(f"{row['id']}: {name} dimensions differ from reference member")
                    value = np.ascontiguousarray(checked_array(ds[name]))
                    fingerprint.update(name.encode()); fingerprint.update(memoryview(value).cast("B"))
                state_hash = fingerprint.hexdigest()
                if state_hash in state_hashes:
                    raise EnsembleRefusal(f"{row['id']}: duplicate physical state; copying files does not create ensemble covariance")
                state_hashes.add(state_hash)
                row["physical_state_sha256"] = state_hash
        return {"cells": len(grid.dimensions["nCells"]),
                "levels": len(reference.dimensions["nVertLevels"]), "nominal_km": nominal}


def recenter(paths, analysis, output):
    """Analysis plus member-minus-mean, using physical MPAS prognostic fields.

    Reject inadmissible humidity/density instead of clipping perturbations.
    The caller runs the regular cold-start door which rebuilds coupled fields.
    """
    import numpy as np
    from netCDF4 import Dataset
    fields = ("theta", "rho", "qv", "qc", "qr", "u", "w", "surface_pressure")
    optional = ("qi", "qs", "qg", "skintemp", "tslb", "smois", "snow", "snowh")
    output = Path(output); output.mkdir()
    means = {}
    with Dataset(analysis) as base:
        with Dataset(paths[0]) as first:
            inherited = tuple(name for name in optional if name in first.variables and name not in base.variables)
            fields += tuple(name for name in optional if name in first.variables and name in base.variables)
        for path in paths:
            with Dataset(path) as ds:
                for name in ("indexToCellID", "indexToEdgeID", "zgrid", "rho_base", "theta_base", "xtime"):
                    if not np.array_equal(checked_array(ds[name]), checked_array(base[name])):
                        raise EnsembleRefusal(f"recenter {name} differs between member and analysis")
        for name in fields:
            mean = np.zeros(base[name].shape, dtype=np.float64)
            for path in paths:
                with Dataset(path) as ds:
                    mean += checked_array(ds[name]).astype(np.float64) / len(paths)
            means[name] = mean
        outputs = []
        for index, path in enumerate(paths):
            target = output / f"member-{index+1:03d}.nc"
            shutil.copyfile(analysis, target)
            with Dataset(path) as ds, Dataset(target, "r+") as result:
                for name in fields:
                    value = checked_array(base[name]).astype(np.float64) + checked_array(ds[name]) - means[name]
                    if ((name in ("qv", "qc", "qr", "qi", "qs", "qg", "snow", "snowh") and np.any(value < 0)) or
                            (name in ("theta", "rho", "surface_pressure", "skintemp", "tslb") and np.any(value <= 0)) or
                            (name == "smois" and np.any((value < 0) | (value > 1)))):
                        raise EnsembleRefusal(f"recenter {name} produces nonphysical values; clipping would alter covariance")
                    if not np.isfinite(value).all():
                        raise EnsembleRefusal(f"recenter {name} produces nonfinite values")
                    result[name][:] = value
                for name in inherited:
                    var = ds[name]
                    target_var = result.createVariable(name, var.datatype, var.dimensions)
                    target_var.setncatts({key:var.getncattr(key) for key in var.ncattrs() if key != "_FillValue"})
                    target_var[:] = checked_array(var)
                result.setncattr("ensemble_recenter", "analysis + member - ensemble mean; coupled cold start rebuilt by hex")
                result.setncattr("ensemble_recenter_fields", ",".join(fields))
                result.setncattr("ensemble_inherited_fields", ",".join(inherited))
            outputs.append(target)
    return outputs


def export_native(history, template, output, valid_time, member_init=None):
    """Publish physical forecast state in the template's MPAS cell/edge order."""
    import numpy as np
    from netCDF4 import Dataset
    output = Path(output)
    temporary = output.with_suffix(".partial.nc")
    if output.exists() or temporary.exists():
        raise EnsembleRefusal(f"output exists: {output}; overwriting would mix two cycles")
    shutil.copyfile(template, temporary)
    copied = []
    try:
        with Dataset(history) as src, Dataset(temporary, "r+") as dst:
            maps = {}
            for dimension, key in (("nCells", "indexToCellID"), ("nEdges", "indexToEdgeID")):
                original, desired = checked_array(src[key]), checked_array(dst[key])
                order = np.argsort(original)
                if (len(np.unique(original)) != original.size or
                        not np.array_equal(original[order], np.sort(desired))):
                    raise EnsembleRefusal(f"history {key} does not match native template mesh")
                maps[dimension] = order[np.searchsorted(original[order], desired)]
                for coordinate in (("latCell", "lonCell") if dimension == "nCells" else ("latEdge", "lonEdge")):
                    difference = checked_array(src[coordinate])[maps[dimension]].astype(np.float64) - checked_array(dst[coordinate])
                    if coordinate.startswith("lon"):
                        difference = (difference + np.pi) % (2*np.pi) - np.pi
                    if np.any(np.abs(difference) > 5e-7):
                        raise EnsembleRefusal(f"{coordinate}: matching IDs describe different meshes; native export would misplace weather fields")
            if member_init is not None:
                # The native template supplies geometry. Member-dependent
                # initial surface/static carriers must not become member 1's.
                with Dataset(member_init) as initial:
                    initial_maps = {}
                    for dimension,key in (("nCells","indexToCellID"),("nEdges","indexToEdgeID")):
                        original, desired = checked_array(initial[key]), checked_array(dst[key])
                        order = np.argsort(original)
                        if not np.array_equal(original[order], np.sort(desired)):
                            raise EnsembleRefusal(f"member init {key} differs from template")
                        initial_maps[dimension] = order[np.searchsorted(original[order],desired)]
                    for name,var in initial.variables.items():
                        if "Time" not in var.dimensions or name not in dst.variables or var.dtype.kind not in "fiu":
                            continue
                        value = checked_array(var)
                        for dim,order in initial_maps.items():
                            if dim in var.dimensions:
                                value = np.take(value,order,axis=var.dimensions.index(dim))
                        dst[name][:] = value
            mappings = {target:history_name(src, target) for target in FIELDS}
            mappings.update({name:name for name,var in src.variables.items()
                             if name in dst.variables and "Time" in var.dimensions})
            # These are the existing hex constructor's physical surface names.
            if "tsk" in src.variables and "skintemp" in dst.variables:
                mappings["skintemp"] = "tsk"
            if "t2" in src.variables and "t2m" in dst.variables:
                mappings["t2m"] = "t2"
            for target,name in mappings.items():
                if name not in src.variables:
                    raise EnsembleRefusal(f"forecast history lacks {name}; cannot export complete member state")
                var = src[name]
                value = checked_array(var)
                for dim, order in maps.items():
                    if dim in var.dimensions:
                        value = np.take(value, order, axis=var.dimensions.index(dim))
                if target not in dst.variables:
                    dst.createVariable(target, "f4", var.dimensions, zlib=True, complevel=1)
                if dst[target].dimensions != var.dimensions:
                    raise EnsembleRefusal(f"{target}: template dimensions differ from history")
                dst[target][:] = value
                if "units" in var.ncattrs():
                    dst[target].setncattr("units", var.units)
                copied.append(target)
            # JEDI extra states do not inherit geometry's diagnostic pressure.
            # Carry a complete split whose sum is the advanced total pressure.
            # MPAS-JEDI uses that sum; the canonical dry base EOS follows
            # MPAS atm_init_coupled_diagnostics (Rd=287, cp=1004.5, p0=1e5).
            if "pressure_base" not in dst.variables:
                base_pressure = 1e5 * (287.0 * checked_array(dst["rho_base"]).astype(np.float64) *
                                       checked_array(dst["theta_base"]).astype(np.float64) / 1e5) ** (1004.5 / (1004.5-287.0))
                dst.createVariable("pressure_base", "f4", dst["pressure"].dimensions)[:] = base_pressure
                dst["pressure_base"].setncattr("units", "Pa")
            if "pressure_p" not in dst.variables:
                dst.createVariable("pressure_p", "f4", dst["pressure"].dimensions)
                dst["pressure_p"].setncattr("units", "Pa")
            # Derived increments must follow the advanced physical state.
            for physical, base, pert in (("rho", "rho_base", "rho_p"),
                                         ("theta", "theta_base", "theta_m"),
                                         ("pressure", "pressure_base", "pressure_p")):
                if base in dst.variables and pert in dst.variables:
                    value = checked_array(dst[physical])
                    if pert == "theta_m":
                        # MPAS theta_m is moist theta, not a base-state difference.
                        value = value * (1 + (461.6 / 287.0) * checked_array(dst["qv"]))
                    else:
                        value = value - checked_array(dst[base])
                    dst[pert][:] = value
            text = valid_time.encode("ascii")
            length = len(dst.dimensions["StrLen"])
            dst["xtime"][0] = np.frombuffer(text.ljust(length, b" "), dtype="S1")
            dst.setncattr("config_start_time", valid_time)
            dst.setncattr("ensemble_schema", SCHEMA)
            dst.setncattr("ensemble_format", "native MPAS order; JEDI ingestion requires independent check")
            dst.setncattr("ensemble_history_sha256", digest(history))
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {"path": str(output), "bytes": output.stat().st_size,
            "sha256": digest(output), "advanced_variables": copied, "valid_time": valid_time}


# MPAS-JEDI reads ensemble members through its "ensemble" stream, whose list
# (mpas-bundle 4.0.1 stream_list.atmosphere.ensemble) is exactly these fields
# plus the scalars var_array. Geometry comes from the separate geometry file,
# so a member file needs nothing else; the full init template is 8 GB at 30 km.
JEDI_ENSEMBLE_FIELDS = ("theta", "surface_pressure", "uReconstructZonal",
                        "uReconstructMeridional", "pressure_p", "pressure_base")
JEDI_SCALARS = ("qv", "qc", "qr", "qi", "qs", "qg")
# Carried only so spread and audits can run on the member file alone; MPAS
# streams ignore variables that are not in the stream list.
JEDI_AUDIT_FIELDS = ("indexToCellID", "latCell", "lonCell", "areaCell")


def _native_cell_map(src, dst):
    """Index array taking history cells to the template's native cell order."""
    import numpy as np
    original, desired = checked_array(src["indexToCellID"]), checked_array(dst["indexToCellID"])
    order = np.argsort(original)
    if (len(np.unique(original)) != original.size or
            not np.array_equal(original[order], np.sort(desired))):
        raise EnsembleRefusal("history indexToCellID does not match native template mesh")
    mapping = order[np.searchsorted(original[order], desired)]
    for coordinate in ("latCell", "lonCell"):
        difference = checked_array(src[coordinate])[mapping].astype(np.float64) - checked_array(dst[coordinate])
        if coordinate.startswith("lon"):
            difference = (difference + np.pi) % (2*np.pi) - np.pi
        if np.any(np.abs(difference) > 5e-7):
            raise EnsembleRefusal(f"{coordinate}: matching IDs describe different meshes; native export would misplace weather fields")
    return mapping


def export_jedi_member(history, template, output, valid_time):
    """Write one MPAS-JEDI ensemble member: the ensemble-stream fields only.

    Native MPAS cell order (template's indexToCellID), CDF-5, float32, MPAS
    names. pressure_base follows MPAS atm_init_coupled_diagnostics on the
    template's dry base state; pressure_p is the advanced total pressure minus
    it, so pressure_p + pressure_base is the member's own pressure (what
    MPAS-JEDI uses). Nothing is clipped or filled.
    """
    import numpy as np
    from netCDF4 import Dataset
    output = Path(output)
    temporary = output.with_suffix(".partial.nc")
    if output.exists() or temporary.exists():
        raise EnsembleRefusal(f"output exists: {output}; overwriting would mix two cycles")
    try:
        with Dataset(history) as src, Dataset(template) as tpl:
            mapping = _native_cell_map(src, tpl)
            levels = len(src.dimensions["nVertLevels"])
            cells = len(tpl.dimensions["nCells"])
            with Dataset(temporary, "w", format="NETCDF3_64BIT_DATA") as dst:
                dst.createDimension("Time", None)
                dst.createDimension("nCells", cells)
                dst.createDimension("nVertLevels", levels)
                dst.createDimension("StrLen", 64)
                def put(name, value, dims, units):
                    var = dst.createVariable(name, "f4" if value.dtype.kind == "f" else value.dtype, dims)
                    var[:] = value
                    if units:
                        var.setncattr("units", units)
                for name in JEDI_AUDIT_FIELDS:
                    put(name, checked_array(tpl[name]), ("nCells",), getattr(tpl[name], "units", ""))
                def cells_of(name):
                    value = checked_array(src[name])[0]
                    return np.take(value, mapping, axis=0)
                theta = cells_of("theta")
                pressure = cells_of("pressure").astype(np.float64)
                rho_base = checked_array(tpl["rho_base"])[0].astype(np.float64)
                theta_base = checked_array(tpl["theta_base"])[0].astype(np.float64)
                base = 1e5 * (287.0 * rho_base * theta_base / 1e5) ** (1004.5 / (1004.5 - 287.0))
                cube = ("Time", "nCells", "nVertLevels")
                put("theta", theta[None], cube, "K")
                put("pressure_base", base.astype(np.float32)[None], cube, "Pa")
                put("pressure_p", (pressure - base).astype(np.float32)[None], cube, "Pa")
                put("uReconstructZonal", cells_of(history_name(src, "uReconstructZonal"))[None], cube, "m s^-1")
                put("uReconstructMeridional", cells_of(history_name(src, "uReconstructMeridional"))[None], cube, "m s^-1")
                put("surface_pressure", cells_of("surface_pressure")[None], ("Time", "nCells"), "Pa")
                for name in JEDI_SCALARS:
                    if name not in src.variables:
                        raise EnsembleRefusal(f"forecast history lacks {name}; a member without it would read as zero condensate")
                    put(name, cells_of(name)[None], cube, "kg kg^-1")
                xtime = dst.createVariable("xtime", "S1", ("Time", "StrLen"))
                xtime[0] = np.frombuffer(valid_time.encode("ascii").ljust(64, b" "), dtype="S1")
                dst.setncattr("config_start_time", valid_time)
                dst.setncattr("ensemble_schema", SCHEMA)
                dst.setncattr("ensemble_format", "MPAS-JEDI ensemble stream member, native MPAS cell order")
                dst.setncattr("ensemble_history_sha256", digest(history))
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {"path": str(output), "bytes": output.stat().st_size, "sha256": digest(output),
            "valid_time": valid_time, "format": "jedi-ensemble-stream"}


PRESSURE_LEVELS_PA = (85000.0, 70000.0, 50000.0, 25000.0)


def _interp_lnp(p, field, target):
    """Linear in ln p between bracketing model levels; NaN outside the column."""
    import numpy as np
    above = p < target
    k1 = np.argmax(above, axis=1)
    ok = above.any(axis=1) & (k1 > 0)
    k1 = np.where(ok, k1, 1)
    k0 = k1 - 1
    rows = np.arange(p.shape[0])
    p0, p1 = p[rows, k0], p[rows, k1]
    w = (np.log(target) - np.log(p0)) / (np.log(p1) - np.log(p0))
    return np.where(ok, field[rows, k0] + w * (field[rows, k1] - field[rows, k0]), np.nan)


def latlon_regrid_weights(lat_cell, lon_cell, step_deg, neighbours=4):
    """4-nearest-cell inverse-distance-squared weights onto a global grid
    (90..-90 by step, 0..360-step), the hex-to-grid method of the scored
    30 km run. Returns (index, weight, shape)."""
    import numpy as np
    from scipy.spatial import cKDTree
    lats = np.linspace(90.0, -90.0, int(round(180 / step_deg)) + 1)
    lons = np.arange(int(round(360 / step_deg))) * step_deg
    la, lo = np.asarray(lat_cell, np.float64), np.asarray(lon_cell, np.float64)
    xyz = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], 1)
    LA, LO = np.meshgrid(np.radians(lats), np.radians(lons), indexing="ij")
    target = np.stack([np.cos(LA) * np.cos(LO), np.cos(LA) * np.sin(LO), np.sin(LA)], -1).reshape(-1, 3)
    distance, index = cKDTree(xyz).query(target, k=neighbours, workers=8)
    weight = 1.0 / np.maximum(distance, 1e-9) ** 2
    return index.astype(np.int64), weight / weight.sum(1, keepdims=True), (len(lats), len(lons))


def extract_pressure_levels(history, zmid, output, regrid=None):
    """Per-cell height, temperature, winds and humidity on standard pressure
    levels plus sea-level pressure, for verification against analyses and
    radiosondes. Same diagnostics as the scored 30 km run's extractor (ln p
    interpolation from layer midpoints; 6.5 K/km lowest-level reduction)."""
    import numpy as np
    from netCDF4 import Dataset
    RD, CP, G0, LAPSE = 287.04, 1004.5, 9.80665, 0.0065
    with Dataset(history) as ds:
        ds.set_auto_mask(False)
        p = np.asarray(ds["pressure"][0], np.float64)
        theta = np.asarray(ds["theta"][0], np.float64)
        u = np.asarray(ds[history_name(ds, "uReconstructZonal")][0], np.float64)
        v = np.asarray(ds[history_name(ds, "uReconstructMeridional")][0], np.float64)
        qv = np.asarray(ds["qv"][0], np.float64)
    t = theta * (p / 1.0e5) ** (RD / CP)
    out = {"mslp": (p[:, 0] * (1.0 + LAPSE * zmid[:, 0] / t[:, 0]) ** (G0 / (RD * LAPSE)) / 100.0).astype(np.float32)}
    for level in PRESSURE_LEVELS_PA:
        tag = int(level / 100)
        out[f"z{tag}"] = _interp_lnp(p, zmid, level).astype(np.float32)
        out[f"t{tag}"] = _interp_lnp(p, t, level).astype(np.float32)
        out[f"u{tag}"] = _interp_lnp(p, u, level).astype(np.float32)
        out[f"v{tag}"] = _interp_lnp(p, v, level).astype(np.float32)
        out[f"q{tag}"] = _interp_lnp(p, qv, level).astype(np.float32)
    if regrid is not None:
        # NaN (below ground) at any donor cell stays NaN on the grid.
        index, weight, shape = regrid
        out = {key: (value[index] * weight).sum(1).reshape(shape).astype(np.float32)
               for key, value in out.items()}
    output = Path(output)
    temporary = output.with_name(output.name + ".partial.npz")
    np.savez_compressed(temporary, **out)
    temporary.replace(output)
    return {"path": str(output), "bytes": output.stat().st_size}


_WEIGHTS = {}


def frame_products(frame, stamp, hour, valid_text, member, export, template, export_dir,
                   extract_dir, weights_path, discard, zmid=None, regrid=None):
    """Every product of one durable frame. Top level so it can run in a
    spawned process: netCDF is not thread-safe, and the first 30 km run lost
    its in-thread consumers at their first frames (2026-10-12)."""
    frame = Path(frame)
    row = {"member": member, "forecast_hour": hour, "frame": str(frame),
           "frame_bytes": frame.stat().st_size, "frame_sha256": digest(frame)}
    if export:
        target = Path(export_dir) / f"mpas.{member}.{stamp}.nc"
        row["jedi"] = export_jedi_member(frame, template, target, valid_text)
    if extract_dir is not None:
        if weights_path is not None:
            if weights_path not in _WEIGHTS:
                import numpy as np
                loaded = np.load(weights_path)
                _WEIGHTS[weights_path] = (loaded["zmid"], None if "index" not in loaded.files else
                                          (loaded["index"], loaded["weight"], tuple(int(n) for n in loaded["shape"])))
            zmid, regrid = _WEIGHTS[weights_path]
        row["extract"] = extract_pressure_levels(frame, zmid, Path(extract_dir) / f"{stamp}.npz", regrid)
    if discard:
        frame.unlink()
        row["frame_deleted"] = True
    return row


def consume_frames(history_dir, start_time, done, *, export_hours, template, export_dir,
                   member, extract_dir=None, zmid=None, discard=False, poll=5.0,
                   regrid=None, on_first_frame=None, pool=None, weights_path=None):
    """Turn each durable history frame into its products as the member runs.

    A frame counts once the history writer's durable marker exists. Products:
    an MPAS-JEDI member file at each export hour, a pressure-level extract
    for verification, then (``discard``) the frame itself is deleted so a
    72 h member never holds more than one or two 1.2 GB frames on disk.
    Returns the product records; raises on the first failed product.
    """
    from .history_writer import durable_marker
    history_dir = Path(history_dir)
    start = datetime.strptime(start_time, "%Y-%m-%d_%H:%M:%S")
    records, seen = [], set()
    from .forecast_door import history_files
    def sweep():
        # history_files reads either route's frames (the MPAS suite route writes history.*.nc).
        for frame in history_files(history_dir):
            if frame in seen or not durable_marker(frame).is_file():
                continue
            stamp = frame.name.split(".", 1)[1][:-len(".nc")]
            valid = datetime.strptime(stamp, "%Y-%m-%d_%H.%M.%S")
            hour = (valid - start).total_seconds() / 3600
            valid_text = valid.strftime("%Y-%m-%d_%H:%M:%S")
            export = any(abs(hour - h) < 1e-6 for h in export_hours)
            arguments = (str(frame), stamp, hour, valid_text, member, export, str(template), str(export_dir),
                         None if extract_dir is None else str(extract_dir), weights_path, False)
            if pool is None:
                row = frame_products(*arguments, zmid=zmid, regrid=regrid)
            else:
                # .result() re-raises the worker's error here, so a failed
                # product fails this member instead of passing silently.
                row = pool.submit(frame_products, *arguments).result()
            seen.add(frame)
            records.append(row)
            if on_first_frame is not None and len(records) == 1:
                on_first_frame()
    while not done.is_set():
        sweep()
        done.wait(poll)
    sweep()
    # The forecast door counts histories and re-verifies its init at closure.
    # Deleting durable frames earlier makes an otherwise passed run fail.
    if discard:
        for row in records:
            Path(row["frame"]).unlink()
            row["frame_deleted"] = True
    return records


def spread(paths, variables=FIELDS, chunk=8192):
    """Area-weighted RMS of pointwise sample SD, for every model level.

    Welford's algorithm and spatial chunks keep memory independent of N.
    Edge-normal wind uses dcEdge*dvEdge dual areas. Units follow the files.
    """
    import numpy as np
    from netCDF4 import Dataset
    from contextlib import ExitStack
    if len(paths) < 2:
        raise EnsembleRefusal("sample spread needs at least two members")
    rows = []
    with ExitStack() as stack:
        datasets = [stack.enter_context(Dataset(path)) for path in paths]
        ref = datasets[0]
        for name in variables:
            var = ref[name]
            dims = var.dimensions
            spatial = "nEdges" if "nEdges" in dims else "nCells"
            axis = dims.index(spatial)
            levdim = next((d for d in dims if d.startswith("nVertLevels")), None)
            levels = len(ref.dimensions[levdim]) if levdim else 1
            total = np.zeros(levels); weight_sum = 0.0
            for start in range(0, len(ref.dimensions[spatial]), chunk):
                stop = min(start + chunk, len(ref.dimensions[spatial]))
                key = [slice(None)] * len(dims); key[axis] = slice(start, stop)
                mean = m2 = None
                for i, ds in enumerate(datasets, 1):
                    if ds[name].dimensions != dims:
                        raise EnsembleRefusal(f"spread {name}: member dimensions differ")
                    value = checked_array(ds[name], tuple(key)).astype(np.float64)
                    value = np.moveaxis(value, axis, 0).reshape(stop-start, levels)
                    if mean is None:
                        mean = np.zeros_like(value); m2 = np.zeros_like(value)
                    delta = value - mean; mean += delta / i; m2 += delta * (value - mean)
                if spatial == "nCells":
                    weight = checked_array(ref["areaCell"], slice(start, stop)).reshape(-1)
                else:
                    weight = (checked_array(ref["dcEdge"], slice(start, stop)) *
                              checked_array(ref["dvEdge"], slice(start, stop))).reshape(-1)
                if np.any(weight <= 0):
                    raise EnsembleRefusal(f"spread {spatial}: nonpositive area weight")
                total += (m2 / (len(paths)-1) * weight[:, None]).sum(axis=0)
                weight_sum += float(weight.sum())
            rms = np.sqrt(total / weight_sum)
            rows.append({"variable": name, "units": getattr(var, "units", ""),
                         "level_dimension": levdim, "spread_per_level": rms.tolist(),
                         "mean_level_spread": float(rms.mean()), "members": len(paths)})
    return rows


def packing(capacities, required, count, requested=None):
    if required <= 0 or count < 1:
        raise EnsembleRefusal("invalid memory footprint or ensemble count")
    limits = []
    for card, free in enumerate(capacities):
        maximum = min(count, int(free) // int(required))
        if requested is not None:
            if requested < 1 or requested > maximum:
                raise EnsembleRefusal(f"card {card}: {requested} members need {requested*required} bytes, only {free} free; concurrent forecast would exhaust device memory")
            maximum = requested
        limits.append(maximum)
    # Use every granted card before placing a second resident on any card.
    slots = [card for depth in range(max(limits, default=0))
             for card, limit in enumerate(limits) if depth < limit]
    if not slots:
        raise EnsembleRefusal("no granted card fits a full member; use the coarser ensemble mesh")
    return slots[:count]


def execute_jobs(records, slots, launch, admit=None):
    """A persistent worker per slot; a slot never runs two processes at once.

    ``admit(row)`` (optional) is asked before each launch; False leaves the
    row unlaunched (a deadline that the next member could not meet).
    """
    from queue import Queue, Empty
    pending = Queue()
    for record in records:
        pending.put(record)
    stop = threading.Event()
    def worker(card):
        results = []
        while not stop.is_set():
            try:
                row = pending.get_nowait()
            except Empty:
                break
            if admit is not None and not admit(row):
                pending.put(row)
                break
            result = launch(row, card)
            results.append(result)
            if result["returncode"]:
                stop.set()
        return results
    results = []
    with ThreadPoolExecutor(max_workers=len(slots)) as pool:
        for future in as_completed([pool.submit(worker, card) for card in slots]):
            results.extend(future.result())
    return results


def export_ensemble(data, output, receipt, hours, io_workers):
    """Export completed hourly histories; shared by the run and recovery doors."""
    from .forecast_door import history_files
    files = {}
    for row in data["members"]:
        histories = history_files(output / row["id"] / "history")
        if len(histories) != int(hours)+1:
            raise EnsembleRefusal(f"{row['id']}: expected initial and hourly frames, found {len(histories)}")
        files[row["id"]] = histories
    spreads = []
    receipt["io_workers"] = min(io_workers, len(data["members"]))
    export_started = time.monotonic()
    known = {row["path"]:row for row in receipt["outputs"]}
    moments = {(row["member"],row["forecast_hour"]):row for row in receipt["outputs"]}
    with ProcessPoolExecutor(max_workers=receipt["io_workers"],
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        for hour in range(int(hours)+1):
            valid = (datetime.strptime(data["start_time"], "%Y-%m-%d_%H:%M:%S") + timedelta(hours=hour)).strftime("%Y-%m-%d_%H:%M:%S")
            states, futures = [], {}
            for row in data["members"]:
                prior_moment = moments.get((row["id"],hour))
                target = (Path(prior_moment["path"]) if prior_moment else
                          output / f"mpas.{row['id']}.{valid.replace(':', '.')}.nc")
                if not target.resolve().is_relative_to(output.resolve()):
                    raise EnsembleRefusal("recorded export escapes its run; refusing recovery outside the owned output directory")
                if target.exists():
                    prior = known.get(str(target))
                    if prior is None or digest(target) != prior["sha256"]:
                        raise EnsembleRefusal(f"existing export {target} has no matching receipt; refusing to replace a consumer's state")
                else:
                    future = pool.submit(export_native, files[row["id"]][hour], data["native_template"], target, valid, row["init"])
                    futures[future] = row["id"]
                states.append(target)
            for future in as_completed(futures):
                description = future.result()
                description.update(member=futures[future], forecast_hour=hour)
                receipt["outputs"].append(description)
                write_json(output / "ensemble-receipt.json", receipt)
            measured = spread(states)
            spreads.append(measured)
            write_json(output / f"spread.f{hour:03d}.json", {"schema": SCHEMA, "forecast_hour": hour,
                       "method": "area-weighted RMS pointwise sample SD, ddof=1; native model levels", "variables": measured})
    receipt["export_and_spread_wall_seconds"] = time.monotonic()-export_started
    growth = []
    for initial, final in zip(spreads[0], spreads[1]):
        a, b = initial["spread_per_level"], final["spread_per_level"]
        growth.append({"variable": initial["variable"], "units": initial["units"],
                       "initial": a, "one_hour": b,
                       "ratio_per_level": [y/x if x else None for x,y in zip(a,b)],
                       "levels_with_growth": sum(y>x for x,y in zip(a,b)), "levels": len(a)})
    write_json(output / "spread-growth.json", {"schema": SCHEMA, "variables": growth,
               "note": "Growth is measured, not assumed. Zero initial spread has no defined ratio."})


def run_ensemble(args):
    from woof.output_formats import argument_values, options
    output_rows, output_store = options(args)
    started = time.monotonic()
    data = read_manifest(args.manifest, args.members, args.resolution_km,
                         forecast_only=args.forecast_only)
    if args.forecast_only and (args.discard_history or args.export_hours is not None or args.analysis):
        raise EnsembleRefusal("--forecast-only retains histories for scoring after command closure; "
                              "it cannot discard them, export JEDI states or recenter")
    if not math.isfinite(args.cycle_hours) or args.cycle_hours < 1:
        raise EnsembleRefusal("cycle length must be finite and at least one hour; covariance requires the one-hour boundary")
    if abs(args.cycle_hours - round(args.cycle_hours)) > 1e-9:
        raise EnsembleRefusal("cycle length must be whole hours so the one-hour covariance frame is written")
    if args.io_workers < 1:
        raise EnsembleRefusal("io-workers must be positive; otherwise no member states would be published")
    output = args.out.resolve()
    if output.exists():
        raise EnsembleRefusal(f"ensemble output exists: {output}; mixed-cycle state is unsafe")
    streamed = any(row.get("prepare") for row in data["members"])
    frames_mode = args.export_hours is not None
    if streamed and not frames_mode and not args.forecast_only:
        raise EnsembleRefusal("streamed (prepare-argv) members need --export-hours: hourly end-of-run export reads every member's frames after the run")
    reference_path = data.get("reference_init")
    if reference_path is not None:
        reference_path = str((args.manifest.resolve().parent / reference_path).resolve())
    if streamed and not (reference_path and Path(reference_path).is_file()):
        raise EnsembleRefusal("a streamed ensemble needs manifest reference_init (an init in forecast order) to validate members as they arrive")
    state_hashes, validation_lock = set(), threading.Lock()
    if streamed:
        dimensions = validate_inputs(data, members=[], reference_path=reference_path)
    else:
        dimensions = validate_inputs(data, state_hashes=state_hashes, reference_path=reference_path)
    visible = [s.strip() for s in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if s.strip()]
    if not visible or not os.environ.get("GPU_MUTEX_CARDS"):
        raise EnsembleRefusal("no card grant; run the ensemble through the host card queue with --cards N --min-cards 1")
    import cupy as cp
    from .device_admission import card_profile_from_attributes, model_for_card
    capacity, footprints, margins, cards = [], [], [], []
    for ordinal, token in enumerate(visible):
        with cp.cuda.Device(ordinal) as device:
            free, total = cp.cuda.runtime.memGetInfo()
            properties = cp.cuda.runtime.getDeviceProperties(ordinal)
            profile = card_profile_from_attributes(str(properties["name"]), device.attributes)
            model = model_for_card(profile, levels=dimensions["levels"])
            need = model.required_bytes(dimensions["cells"])
            capacity.append(int(free)); footprints.append(need)
            margins.append(model.margin_bytes())
            cards.append({"visible_token": token, "free_bytes": int(free), "total_bytes": int(total),
                          "device_name": str(properties["name"]), "multiprocessors": profile.multiprocessors,
                          "member_required_bytes": need, "model_measured": model.measured,
                          "model_provenance": model.provenance})
    # Heterogeneous grants use the most expensive footprint for safe packing.
    requested = None if args.members_per_card == "auto" else int(args.members_per_card)
    flags = data.get("forecast_flags", []) + [flag for row in data["members"] for flag in row["forecast_flags"]]
    if any(flag.split("=", 1)[0] == "--fullstep-local-timestep" for flag in flags) and requested != 1:
        raise EnsembleRefusal("LTS members require --members-per-card 1: the uniform-mesh packing "
                              "calibration does not price per-class resident states")
    if requested != 1 and any(row.get("forecast_flags") for row in data["members"]):
        raise EnsembleRefusal("members carry their own physics flags: the packing pilot prices one configuration, "
                              "so a card packed on that figure can run out of memory when a member with a "
                              "different suite allocates; pass --members-per-card 1")
    # The existing global fit does not cover every physics/card combination.
    # An unmeasured multi-resident launch can exceed memory before any worker
    # publishes a receipt. Start one real ensemble member, measure it, then
    # pack the remaining members. Reuse only the same configuration's receipt.
    packing(capacity, max(footprints), args.members, 1)
    slots = []
    output.mkdir(parents=True)
    receipt = {"schema": SCHEMA, "status": "running", "manifest": str(args.manifest.resolve()),
               "manifest_sha256": digest(args.manifest), "members": args.members,
               "resolution_km": args.resolution_km, "cycle_hours": args.cycle_hours,
               "cards": cards, "slots": slots, "dimensions": dimensions,
               "jedi_open_checked": False, "outputs": [], "jobs": []}
    receipt["patch"] = data.get("patch")
    write_json(output / "ensemble-receipt.json", receipt)
    try:
        if args.analysis:
            new = recenter([r["init"] for r in data["members"]], args.analysis, output / "recentered")
            for row, path in zip(data["members"], new):
                row["init"] = str(path)
                row["source"] += f"; recentered on {args.analysis} sha256={digest(args.analysis)}"
        samples = []
        def card_used(token):
            result = subprocess.run(["nvidia-smi", "-i", token, "--query-gpu=memory.used",
                                     "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
            return int(result.stdout.strip())
        baselines = {token: card_used(token) for token in visible}
        stopped = threading.Event()
        def monitor():
            while not stopped.is_set():
                result = subprocess.run(["nvidia-smi", "--query-gpu=uuid,memory.used,utilization.gpu",
                                         "--format=csv,noheader,nounits"], capture_output=True, text=True)
                if result.returncode == 0:
                    samples.append({"wall_seconds": time.monotonic()-started,
                                    "gpu_rows": result.stdout.strip().splitlines()})
                stopped.wait(args.sample_seconds)
        watcher = threading.Thread(target=monitor, daemon=True); watcher.start()
        zmid = None
        regrid = None
        extract_root = output / "extracts"
        member_root = (args.member_root or output).resolve()
        if frames_mode and args.extract_levels:
            import numpy as np
            from netCDF4 import Dataset
            extract_root.mkdir()
            with Dataset(reference_path or data["members"][0]["init"]) as ds:
                zg = np.asarray(ds["zgrid"][0] if ds["zgrid"].ndim == 3 else ds["zgrid"][:], np.float64)
            zmid = 0.5 * (zg[:, :-1] + zg[:, 1:])
            with Dataset(data["grid"]) as g:
                lat_cell, lon_cell = np.asarray(g["latCell"][:]), np.asarray(g["lonCell"][:])
            np.savez(extract_root / "grid.npz", latCell=lat_cell, lonCell=lon_cell)
            weights = {"zmid": zmid}
            if args.extract_grid_deg:
                regrid = latlon_regrid_weights(lat_cell, lon_cell, args.extract_grid_deg)
                weights.update(index=regrid[0], weight=regrid[1], shape=np.array(regrid[2]))
                write_json(extract_root / "grid.json", {"step_deg": args.extract_grid_deg, "shape": list(regrid[2]),
                           "lat_first": 90.0, "lon_first": 0.0,
                           "method": "4 nearest cells, inverse distance squared; NaN if any donor is below ground"})
        weights_path = None
        product_pool = None
        if frames_mode:
            if zmid is not None:
                weights_path = str(extract_root / "_weights.npz")
                np.savez(weights_path, **weights)
            product_pool = ProcessPoolExecutor(max_workers=max(1, min(8, len(visible) * 2)),
                                               mp_context=multiprocessing.get_context("spawn"))
        prepared, prepare_lock = {}, threading.Lock()
        prepare_pool = ThreadPoolExecutor(max_workers=max(1, args.prepare_ahead))
        def prepare(row):
            """Build a streamed member's init, then validate it like any other."""
            record = {"id": row["id"], "prepared": False}
            if row.get("prepare") and not Path(row["init"]).is_file():
                before = time.monotonic()
                log = output / f"prepare-{row['id']}.log"
                with log.open("w") as stream:
                    result = subprocess.run(row["prepare"], stdout=stream, stderr=subprocess.STDOUT,
                                            env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
                record.update(prepared=True, returncode=result.returncode, log=str(log),
                              wall_seconds=time.monotonic()-before)
                if result.returncode or not Path(row["init"]).is_file():
                    raise EnsembleRefusal(f"member {row['id']} preparation failed: {log}")
            if streamed:
                with validation_lock:
                    validate_inputs(data, members=[row], state_hashes=state_hashes, reference_path=reference_path)
            return record
        def ensure_prepared(row):
            with prepare_lock:
                if row["id"] not in prepared:
                    prepared[row["id"]] = prepare_pool.submit(prepare, row)
                return prepared[row["id"]]
        order = [row["id"] for row in data["members"]]
        by_id = {row["id"]: row for row in data["members"]}
        def launch(row, card):
            root = output / row["id"]; root.mkdir()
            work = member_root / row["id"]; work.mkdir(parents=True, exist_ok=work == root)
            try:
                preparation = ensure_prepared(row).result()
            except Exception as error:
                return {"id": row["id"], "card_ordinal": card, "returncode": 1,
                        "product_error": f"preparation: {type(error).__name__}: {error}",
                        "products": [], "source": row["source"]}
            if args.prepare_ahead:
                # Overlap the next members' CPU preparation with this forecast.
                position = order.index(row["id"])
                for later in order[position+1:position+1+args.prepare_ahead]:
                    ensure_prepared(by_id[later])
            assets = {key: row.get(key, data[key]) for key in ("mesh", "grid", "static", "mesh_rows")}
            command = [sys.executable, "-m", "woof.hex.cli", "forecast", "--mesh", assets["mesh"],
                       "--grid", assets["grid"], "--static", assets["static"], "--init", row["init"],
                       "--init-source", row["source"], "--start-time", data["start_time"],
                       "--hours", str(args.cycle_hours), "--history-every-minutes", str(args.history_every_minutes),
                       "--out", str(work / "history"), "--scratch", str(work / "scratch"),
                       # The forecast door draws the standard product set by
                       # default since the products door (0.4.0).  A member's
                       # products are the ensemble's frame products; the door's
                       # own stage would draw every view per member and, in
                       # frame mode, read history frames the consumer deletes.
                       "--no-products",
                       *data.get("forecast_flags", []), *row.get("forecast_flags", []),
                       '--stochastic', json.dumps(row['stochastic'],sort_keys=True)]
            command += argument_values(args, relative=row["id"])
            if work != root and output_store is None and any(r["writer"] for r in output_rows):
                command += ["--icechunk-store", str(root / "icechunk")]
            if not frames_mode and not args.forecast_only:
                # Hourly mode exports after the run, so a refused step must
                # still leave a receipt for the committed frames. Frame mode
                # has already published every durable frame's products, and
                # arming the rollback costs 31.5 GB at 30 km (two committed
                # boundaries), which pushes a member past a 96 GB card: the
                # forecast door refused 114.6 GB needed against 96.1 GB free.
                command.insert(command.index("--out"), "--stop-on-refusal")
            if args.gpuwm_checkout:
                command += ["--gpuwm-checkout", str(args.gpuwm_checkout.resolve())]
            residents = slots.count(card) if slots else 1
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=visible[card],
                       WOOF_HEX_MEMBER_MEMORY="shared-card" if residents > 1 else "single",
                       GPU_MUTEX_CARDS=os.environ["GPU_MUTEX_CARDS"].replace(",", " ").split()[card],
                       WOOF_HEX_MESH_ROWS=assets["mesh_rows"])
            # Respect a caller's warm caches; a fresh root would recompile and
            # re-prepare the 30 km host state once per member.
            env.setdefault("WOOF_HEX_PREP_CACHE", str(output.parent / "hostprep-cache"))
            env.setdefault("WOOF_HEX_CACHE_ROOT", str(output.parent / "cache"))
            if residents > 1:
                # Several residents share one card's memory: keep history
                # writes in-process rather than adding a writer per member.
                env["WOOF_HEX_HISTORY_BACKGROUND"] = "off"
            done = threading.Event()
            consumer = None
            if frames_mode:
                consumer_pool = ThreadPoolExecutor(max_workers=1)
                extract_dir = None
                if zmid is not None:
                    extract_dir = extract_root / row["id"]; extract_dir.mkdir()
                init_digest = digest(row["init"])
                # The init is NOT dropped at frame 0: the driver re-verifies
                # it after the last step (verify_forecast_authorities), and a
                # member whose init was deleted mid-run ends with no receipt
                # (measured: member gep02, 2026-10-12). --discard-inputs acts
                # after the member passes, below.
                drop_input = None
                consumer = consumer_pool.submit(
                    consume_frames, work / "history", data["start_time"], done,
                    export_hours=args.export_hours, template=data["native_template"],
                    export_dir=args.export_dir or output, member=row["id"], extract_dir=extract_dir, zmid=zmid,
                    discard=args.discard_history, on_first_frame=drop_input,
                    pool=product_pool, weights_path=weights_path)
            before = time.monotonic()
            peak = [card_used(visible[card])]
            memory_done = threading.Event()
            def sample_member():
                while not memory_done.wait(args.sample_seconds):
                    peak.append(card_used(visible[card]))
            memory_thread = threading.Thread(target=sample_member, daemon=True)
            memory_thread.start()
            with (root / "forecast.log").open("w") as log:
                try:
                    result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                finally:
                    memory_done.set(); memory_thread.join()
            wall = time.monotonic()-before
            products, product_error = [], None
            if consumer is not None:
                done.set()
                try:
                    products = consumer.result()
                except Exception as error:  # reported as a failed member below
                    product_error = f"{type(error).__name__}: {error}"
                consumer_pool.shutdown()
            init_sha = init_digest if frames_mode else digest(row["init"])
            run_facts = {}
            member_flags = data.get("forecast_flags", []) + row.get("forecast_flags", [])
            if result.returncode == 0 and any(flag.split("=", 1)[0] == "--fullstep-local-timestep"
                                             for flag in member_flags):
                try:
                    run_facts = lts_run_facts(work / "history")
                except (OSError, ValueError, KeyError, EnsembleRefusal) as error:
                    product_error = f"LTS execution receipt: {error}"
            if args.discard_inputs and row.get("prepare") and result.returncode == 0 and product_error is None:
                Path(row["init"]).unlink(missing_ok=True)
            if work != root and result.returncode == 0 and product_error is None:
                for name in ("history", "scratch"):
                    for leftover in sorted((work / name).glob("*")):
                        if leftover.is_file() and not leftover.name.startswith("cuda-history."):
                            shutil.copy2(leftover, root / f"{name}-{leftover.name}")
                # Preserve native stores and their receipts before temporary member cleanup.
                for name in ("icechunk", "output-formats"):
                    source = work / "history" / name
                    if source.is_dir():
                        shutil.copytree(source, root / name, dirs_exist_ok=True)
                shutil.rmtree(work)
            segment = dict(segment=0, **assets, grid_sha256=digest(assets["grid"]),
                           static_sha256=digest(assets["static"]), init_sha256=init_sha,
                           peak_whole_card_mib=max(peak), baseline_whole_card_mib=peak[0],
                           wall_seconds=wall,
                           visible_device=visible[card], **run_facts,
                           forecast_receipt=str(work / "history" / "forecast-receipt.json"))
            return {"id": row["id"], "card_ordinal": card, "command": command,
                    "returncode": result.returncode or (1 if product_error else 0),
                    "product_error": product_error, "wall_seconds": wall,
                    "preparation": preparation, "products": products,
                    "init_sha256": init_sha, "source": row["source"],
                    "forecast_flags": row.get("forecast_flags", []),
                    "stochastic": row['stochastic'],
                    "segments": [segment], "peak_whole_card_mib": max(peak),
                    **run_facts,
                    "log": str(root / "forecast.log")}
        try:
            engine = args.gpuwm_checkout
            if engine is None:
                spec = importlib.util.find_spec("woof")
                engine = Path(spec.origin).parent.parent if spec and spec.origin else None
            engine_hash = digest(engine / "woof" / "config.py") if engine else None
            signature = {"grid_sha256": digest(data["grid"]), "dimensions": dimensions,
                         "cycle_hours":args.cycle_hours,
                         "forecast_flags": data.get("forecast_flags", []), "engine_config_sha256":engine_hash,
                         "cupy_version":cp.__version__, "hardware":cards[0]["device_name"],
                         "multiprocessors":cards[0]["multiprocessors"]}
            key = hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
            cache = output.parent / ("memory-calibration-" + key + ".json")
            calibration = None
            if cache.exists():
                candidate = json.loads(cache.read_text())
                source = Path(candidate["receipt"])
                if source.is_file():
                    prior = json.loads(source.read_text())
                    if prior.get("calibration", {}).get("key") == key:
                        calibration = prior["calibration"]
            jobs = []
            remaining = data["members"]
            if calibration is None and requested == 1:
                # One resident per card is exactly what the footprint model
                # already admitted above (packing(..., 1)); the pilot only
                # prices extra residents. Running it alone would idle every
                # other granted card for a whole member (50 min at 30 km/72 h).
                calibration = {"key": key, "signature": signature, "pilot": None,
                               "member_peak_bytes": 0, "receipt": None,
                               "note": "members-per-card 1: no pilot; per-card peaks are in memory_samples"}
            if calibration is None:
                pilot = launch(remaining[0], 0)
                jobs.append(pilot)
                receipt["jobs"] = jobs
                if pilot["returncode"]:
                    raise EnsembleRefusal(f"memory calibration member failed: {pilot['log']}")
                observed = baselines[visible[0]]
                for sample in samples:
                    for line in sample["gpu_rows"]:
                        token, used, _ = (part.strip() for part in line.split(","))
                        if token == visible[0]:
                            observed = max(observed, int(used))
                peak_bytes = (observed-baselines[visible[0]]) * (1 << 20)
                if peak_bytes <= 0:
                    raise EnsembleRefusal("calibration had no measured device allocation; cannot price concurrent members")
                calibration = {"key":key, "signature":signature, "pilot":pilot,
                               "whole_card_peak_mib":observed, "baseline_mib":baselines[visible[0]],
                               "member_peak_bytes":peak_bytes, "receipt":str(output / "ensemble-receipt.json")}
                remaining = remaining[1:]
            same_hardware = all((c["device_name"],c["multiprocessors"]) ==
                                (cards[0]["device_name"],cards[0]["multiprocessors"]) for c in cards)
            calibrated_need = max(max(footprints), calibration["member_peak_bytes"] + max(margins))
            if not same_hardware:
                # A different card's memory behavior has not been measured.
                slots = packing(capacity, max(footprints), args.members, 1)
                if requested not in (None,1):
                    raise EnsembleRefusal("heterogeneous grant has no per-hardware calibration; multiple residents could exhaust an unmeasured card")
            else:
                slots = packing(capacity, calibrated_need, args.members, requested)
            receipt.update(calibration=calibration, slots=slots, calibrated_member_required_bytes=calibrated_need)
            write_json(output / "ensemble-receipt.json", receipt)
            if calibration.get("pilot") is not None:
                write_json(cache, {"key":key,"receipt":calibration["receipt"]})
            admit = None
            if args.deadline_seconds:
                # Launch a member only if it can finish before the deadline,
                # priced by the slowest member finished so far (or the
                # caller's estimate before any has finished). A backfill
                # grant is killed at its estimate; a member cut there is lost.
                walls = [args.member_seconds_estimate]
                def admit(row):
                    finished = [j["wall_seconds"] for j in jobs_live if j.get("wall_seconds")]
                    expected = max(finished) if finished else max(walls)
                    return time.monotonic() - started + expected <= args.deadline_seconds
            jobs_live = jobs
            def tracked_launch(row, card):
                result = launch(row, card)
                jobs_live.append(result)
                return result
            before_count = len(jobs)
            execute_jobs(remaining, slots, tracked_launch, admit)
            launched = {j["id"] for j in jobs}
            receipt["unlaunched"] = [r["id"] for r in data["members"] if r["id"] not in launched]
        finally:
            stopped.set(); watcher.join()
            # Queued preparations must not keep a failed run (and its cards)
            # alive: measured on G1, a refused first member left the door
            # holding two cards for 4.5 min while later members prepared.
            prepare_pool.shutdown(wait=False, cancel_futures=True)
            if product_pool is not None:
                product_pool.shutdown()
            if weights_path is not None:
                Path(weights_path).unlink(missing_ok=True)
        receipt.update(jobs=jobs, forecast_wall_seconds=time.monotonic()-started,
                       memory_samples=samples)
        if any(row["returncode"] for row in jobs) or (len(jobs) != args.members and not receipt.get("unlaunched")):
            raise EnsembleRefusal("ensemble incomplete; failed member logs are named in ensemble-receipt.json")
        if args.forecast_only:
            receipt["status"] = ("partial_deadline_forecast_passed" if receipt.get("unlaunched")
                                 else "forecast_passed")
        elif frames_mode:
            receipt["outputs"] = [dict(p["jedi"], member=p["member"], forecast_hour=p["forecast_hour"])
                                  for job in jobs for p in job["products"] if "jedi" in p]
            for hour in args.export_hours:
                states = sorted(Path(row["path"]) for row in receipt["outputs"]
                                if abs(row["forecast_hour"]-hour) < 1e-6)
                if len(states) != len(jobs):
                    raise EnsembleRefusal(f"hour {hour:g}: {len(states)} member files for {len(jobs)} finished members")
                if len(states) < 2:
                    continue
                measured = spread(states, variables=("theta", "qv", "surface_pressure", "uReconstructZonal",
                                                     "uReconstructMeridional", "pressure_p"))
                write_json(output / f"spread.f{int(round(hour)):03d}.json", {"schema": SCHEMA, "forecast_hour": hour,
                           "method": "area-weighted RMS pointwise sample SD, ddof=1; native model levels; MPAS-JEDI member files",
                           "variables": measured})
            receipt["status"] = ("partial_deadline_jedi_export_passed" if receipt.get("unlaunched")
                                 else "forecast_and_jedi_export_passed")
        else:
            export_ensemble(data, output, receipt, args.cycle_hours, args.io_workers)
            receipt["status"] = "forecast_and_native_export_passed"
        receipt["total_wall_seconds"] = time.monotonic()-started
        return 0
    except BaseException as error:
        receipt["status"] = "failed"; receipt["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        receipt["elapsed_wall_seconds"] = time.monotonic()-started
        write_json(output / "ensemble-receipt.json", receipt)


def add_ensemble_parser(commands):
    moving = commands.add_parser("moving-ensemble",
        help="moving-region members with physics diversity, track spread and intensity plumes")
    from .moving_members import add_moving_ensemble_arguments
    add_moving_ensemble_arguments(moving)
    parser = commands.add_parser("ensemble", help="hourly global MPAS ensemble, 20 members at 60 km by default")
    from woof.output_formats import add_arguments
    add_arguments(parser)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--members", type=int, default=20, help="distinct members; no upper count cap")
    parser.add_argument("--resolution-km", type=float, default=60)
    parser.add_argument("--forecast-only", action="store_true",
                        help="retain closed histories for observational scoring; supports a one-member pilot, "
                        "without covariance export or rollback snapshots")
    parser.add_argument("--cycle-hours", type=float, default=1)
    parser.add_argument("--members-per-card", default="auto", help="auto from existing footprint model, or a positive integer")
    parser.add_argument("--analysis", type=Path, help="hex analysis in the forecast mesh order; recenter perturbations")
    parser.add_argument("--gpuwm-checkout", type=Path, help="pinned physics engine checkout; normal seam checks apply")
    parser.add_argument("--io-workers", type=int, default=8, help="isolated CPU export processes (default 8)")
    parser.add_argument("--history-every-minutes", type=int, default=60,
                        help="member history cadence (default 60, the hourly covariance frames)")
    parser.add_argument("--export-hours", type=lambda s: [float(x) for x in s.split(",") if x.strip()],
                        default=None, help="comma list of forecast hours to publish as MPAS-JEDI ensemble members "
                        "while members run (frame mode); default: every hour, exported after all members finish")
    parser.add_argument("--extract-levels", action="store_true",
                        help="frame mode: write per-frame pressure-level/MSLP extracts for verification")
    parser.add_argument("--discard-history", action="store_true",
                        help="frame mode: delete each history frame once its products are written")
    parser.add_argument("--discard-inputs", action="store_true",
                        help="delete a streamed member's prepared init after its forecast and products pass "
                        "(not earlier: the driver re-verifies the init after the last step)")
    parser.add_argument("--prepare-ahead", type=int, default=1,
                        help="streamed members prepared ahead of the running ones (CPU overlap)")
    parser.add_argument("--sample-seconds", type=float, default=5.0, help="GPU memory sample interval")
    parser.add_argument("--deadline-seconds", type=float, default=None,
                        help="launch no member that would end after this many seconds (fits a backfill grant); "
                        "unlaunched members are listed in the receipt for the next run")
    parser.add_argument("--member-seconds-estimate", type=float, default=900.0,
                        help="member wall used by --deadline-seconds until one member has finished")
    parser.add_argument("--member-root", type=Path, default=None,
                        help="where member history/scratch live while running (e.g. a RAM disk); receipts stay in --out")
    parser.add_argument("--export-dir", type=Path, default=None, help="frame mode: where MPAS-JEDI member files go (default --out)")
    parser.add_argument("--extract-grid-deg", type=float, default=None,
                        help="frame mode: regrid extracts to a global lat-lon grid of this step (compressed)")
    parser.set_defaults(handler=run_ensemble)
