"""GPU verification against dated analyses projected onto the native mesh."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from .s2s_door import S2SError, cores, write_json
from .s2s_products import TIME_FORMAT, history_index, interp_log_pressure, native_diagnostics, read_native


def supplied_references(entries):
    result = {}
    for entry in entries:
        try:
            key, path = entry.split("=", 1)
            source, cycle = key.split(":", 1)
            valid = datetime.strptime(cycle, "%Y-%m-%dT%H")
            if source not in ("gfs", "era5"):
                raise ValueError("source must be gfs or era5")
        except ValueError as error:
            raise S2SError("--analysis-reference needs gfs|era5:YYYY-MM-DDTHH=INIT.nc: " + str(error)) from error
        result[source, valid] = Path(path).expanduser().absolute()
    return result


def validate_reference(path, source, valid):
    try:
        receipt = json.loads((path.parent / "init-source-receipt.json").read_text())
        actual = "".join(ch for ch in receipt["cycle"] if ch.isdigit())[:10]
    except (OSError, ValueError, KeyError) as error:
        raise S2SError(f"analysis requires its source receipt: {path}: {error}") from error
    if (not path.is_file() or receipt.get("source") != source or actual != valid.strftime("%Y%m%d%H")
            or receipt.get("lead_hours", 0) != 0):
        raise S2SError(f"analysis source, cycle or zero lead does not match {source} {valid}: {path}")


def owned_init_complete(path):
    """True when PATH and the receipt beside it record a finished init-source build (the receipt is written
    last, atomically, and names the init it finished)."""
    try:
        receipt = json.loads((path.parent / "init-source-receipt.json").read_text())
    except (OSError, ValueError):
        return False
    return (path.is_file() and not receipt.get("failed")
            and (receipt.get("init") or {}).get("path") == str(path))


def weighted_metrics(xp, forecast, reference, weights):
    mask = xp.isfinite(forecast) & xp.isfinite(reference) & (weights > 0)
    count = int(mask.sum())
    if not count:
        return {"finite": 0, "bias": None, "rmse": None}
    difference = forecast[mask] - reference[mask]
    area = weights[mask]
    total = area.sum()
    return {"finite": count, "bias": float((difference * area).sum() / total),
            "rmse": float(xp.sqrt((difference * difference * area).sum() / total))}


def read_analysis(path):
    from woof.netcdf_bridge import open_dataset
    with open_dataset(path) as ds:
        arrays = {}
        for name in ("t2m", "rho", "theta", "qv", "zgrid", "latCell", "lonCell"):
            variable = ds.variables[name]
            array = variable[:]
            if variable.dimensions and variable.dimensions[0] == "Time":
                array = array[0]
            arrays[name] = array
    return arrays


def same_mesh_coordinates(xp, reference, native):
    """Compare the geometry at the float32 precision used by the GPU route.

    Prepared init files can store float32-rounded coordinates in float64
    variables. A fixed 1e-7 radian tolerance rejects valid longitudes near
    2*pi, where half a float32 ulp is about 2.4e-7 radians.
    """
    reference = xp.asarray(reference, xp.float32)
    native = xp.asarray(native, xp.float32)
    return reference.shape == native.shape and bool(xp.array_equal(reference, native))


def analysis_diagnostics(cp, arrays):
    from .mpasdiag.isobaric import compute_slp
    from .regional_v841 import RVORD_F32
    theta = cp.asarray(arrays["theta"], cp.float32)
    rho = cp.asarray(arrays["rho"], cp.float32)
    qv = cp.asarray(arrays["qv"], cp.float32)
    # Native init stores physical dry density, dry theta and qv. Apply
    # MPAS recovery.py's equation of state after undoing the density metric;
    # zz cancels. t_init is an unpopulated optional diagnostic in these files.
    argument = (287.0 / 100000) * rho * theta * (1 + RVORD_F32 * qv)
    pressure = cp.ascontiguousarray(100000 * argument ** (1004.5 / (1004.5 - 287.0)))
    temperature = cp.ascontiguousarray(theta * (pressure / 100000) ** (287.0 / 1004.5))
    heights = cp.ascontiguousarray(cp.asarray(arrays["zgrid"], cp.float32))
    if heights.ndim == 3:
        heights = heights[0]
    return {"temperature2": cp.asarray(arrays["t2m"], cp.float32),
            "z500": interp_log_pressure(cp, pressure, (heights[:, 1:] + heights[:, :-1]) * 0.5),
            "mslp": compute_slp(cp.ascontiguousarray(pressure / 100), temperature, heights,
                                 cp.ascontiguousarray(qv))}


def verify_analyses(value, state, args, env):
    from .global_ensemble import _cli
    work = Path(value["paths"]["work"])
    output = work / "analysis-verification"
    output.mkdir(parents=True, exist_ok=True)
    start = datetime.strptime(value["start_time"], TIME_FORMAT)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    supplied = supplied_references(getattr(args, "analysis_reference", []))
    jobs = [(source, start + timedelta(days=day)) for day in range(min(10, value["days"]) + 1) for source in ("gfs", "era5")]
    available = sum(valid <= now for source, valid in jobs)
    workers = min(cores(), 4, max(1, available))
    threads = max(1, cores() // workers)

    def prepare(job):
        source, valid = job
        row = {"source": source, "valid_time": valid.strftime(TIME_FORMAT)}
        if valid > now:
            return {**row, "status": "missing", "reason": "analysis valid time is in the future"}
        path = supplied.get(job)
        if path is None and source == "gfs" and value["source"] == "gfs":
            path = next((Path(member["init"]) for member in value["members"]
                         if member["start_time"] == row["valid_time"]
                         and member.get("lead_hours", 0) == 0), None)
        if path is not None:
            validate_reference(path, source, valid)
            return {**row, "path": str(path), "owned": False, "status": "ready"}
        # ERA5 is keyless (ARCO) by default, so it is prepared like GFS.
        folder = work / "inits" / (f"analysis-{source}-" + valid.strftime("%Y%m%d%H"))
        path = folder / "init.nc"
        if owned_init_complete(path):
            validate_reference(path, source, valid)
            return {**row, "path": str(path), "owned": True, "status": "ready"}
        # THE BREAKAGE THIS PREVENTS (fit16-rows s2s2, verification host, 2026-11-07 05:50Z): the data disk filled while
        # two ERA5 analysis inits were being written, leaving init.nc present beside a 0-byte receipt. The resume
        # took the bare init.nc as finished and then refused the whole verify stage ("analysis requires its
        # source receipt ... Expecting value"). An owned folder whose receipt does not record its finished init
        # is an interrupted build, so it is rebuilt here like a missing one.
        argv = list(value["commands"]["init"])
        for flag, text in (("--source", source), ("--cycle", valid.strftime("%Y-%m-%dT%H")),
                           ("--out-dir", str(folder))):
            argv[argv.index(flag) + 1] = text
        log = output / (f"{source}-" + valid.strftime("%Y%m%d%H") + ".log")
        rc = _cli(argv, log, {**env, "RAYON_NUM_THREADS": str(threads)})
        if rc:
            return {**row, "status": "missing", "reason": f"{source.upper()} analysis preparation exited {rc}",
                    "argv": argv, "log": str(log)}
        validate_reference(path, source, valid)
        return {**row, "path": str(path), "owned": True, "status": "ready", "argv": argv, "log": str(log)}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        references = list(pool.map(prepare, jobs))
    import cupy as cp
    from woof.netcdf_bridge import open_dataset
    with open_dataset(value["paths"]["grid"]) as ds:
        weights = cp.asarray(ds.variables["areaCell"][:], cp.float64)
        latitude = cp.asarray(ds.variables["latCell"][:])
        longitude = cp.asarray(ds.variables["lonCell"][:])
    indices = {member["run_id"]: history_index(member, state["attempts"]) for member in value["members"]}
    with open_dataset(value["members"][0]["init"]) as ds:
        height_array = ds.variables["zgrid"][:]
    if height_array.ndim == 3:
        height_array = height_array[0]
    heights = cp.ascontiguousarray(cp.asarray(height_array, cp.float32))
    result = {"schema": "gpuwm-hex.s2s-analysis-verification/v1", "references": references,
              "metrics": [], "deleted_files": [], "parallelism": {"preparation_processes": workers,
              "rust_threads_per_process": threads, "history_reader_threads": min(cores(), 4), "metrics": "GPU"},
              "scope": "area-weighted native-mesh comparison to projected analysis; lead 0 is an initialization check",
              "precipitation": "not measured: instantaneous analyses do not define accumulated precipitation"}
    receipt_path = output / "receipt.json"
    write_json(receipt_path, result)
    for reference in references:
        if reference["status"] != "ready":
            continue
        valid = datetime.strptime(reference["valid_time"], TIME_FORMAT)
        path = Path(reference["path"])
        arrays = read_analysis(path)
        if (not same_mesh_coordinates(cp, arrays["latCell"], latitude)
                or not same_mesh_coordinates(cp, arrays["lonCell"], longitude)):
            raise S2SError(f"analysis native mesh differs from forecast mesh: {path}")
        observed = analysis_diagnostics(cp, arrays)
        selected = [(member, indices[member["run_id"]].get(valid)) for member in value["members"]]
        if any(history is None for _, history in selected):
            raise S2SError(f"analysis verification lacks member history at {valid}")
        with ThreadPoolExecutor(max_workers=min(cores(), 4)) as pool:
            payloads = pool.map(read_native, [history for _, history in selected])
            ensemble = {field: [] for field in observed}
            for (member, history), payload in zip(selected, payloads):
                forecast = native_diagnostics(cp, payload, heights)
                metrics = {field: weighted_metrics(cp, forecast[field], plane, weights)
                           for field, plane in observed.items()}
                result["metrics"].append({"member": member["run_id"], "source": reference["source"],
                    "valid_time": reference["valid_time"], "lead_hours": (valid - start).total_seconds() / 3600,
                    "fields": metrics, "history": str(history)})
                for field in observed:
                    ensemble[field].append(forecast[field])
        result["metrics"].append({"member": "ensemble-mean", "source": reference["source"],
            "valid_time": reference["valid_time"], "lead_hours": (valid - start).total_seconds() / 3600,
            "fields": {field: weighted_metrics(cp, cp.stack(ensemble[field]).mean(axis=0), plane, weights)
                       for field, plane in observed.items()}})
        reference["status"] = "measured"
        write_json(receipt_path, result)
        if reference["owned"]:
            # Only an init created for this verification is disposable here.
            deletion = {"path": str(path), "bytes": path.stat().st_size}
            result["deleted_files"].append(deletion)
            write_json(receipt_path, result)
            path.unlink()
    write_json(receipt_path, result)
    return {"receipt": str(receipt_path), "measured_references": sum(r["status"] == "measured" for r in references),
            "missing_references": sum(r["status"] == "missing" for r in references),
            "metric_rows": len(result["metrics"]), "parallelism": result["parallelism"]}
