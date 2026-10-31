"""Orchestrate the unchanged Rust writer as each MET endpoint is published."""
from datetime import datetime, timedelta
import json
from pathlib import Path
import time
import subprocess

from .lbc_stream import atomic_json, checked_marker, file_name, FORMAT, READY, SERIES, sha256

#: ``--physics-mode`` of the boundary door (physics.mode, the shared key of woof.hex.correct_mode).
#: ``correct`` is this door's default: boundary edge winds are extrapolated below and above the
#: first-guess column the way the initial conditions are (IN4), and every boundary file must carry its
#: own valid time (IN5).  ``exact`` reproduces native init_atmosphere v8.4.2 case 9.
LBC_PHYSICS_MODES = ("correct", "exact")


def lbc_physics_mode(engine, mode):
    """``(argv, record)``: the rw_mpas_lbc arguments for one physics mode and the correct-mode record
    for the door receipt (``None`` in exact mode).

    ``exact`` adds nothing, so an exact run passes the argument vector this door always passed and
    the engine writes the same bytes.  ``correct`` asks the engine first: an rw_mpas_lbc built before
    ``--physics-mode`` existed reads unknown flags without complaint, so it would silently write the
    native case-9 boundary winds.  Such an engine is not handed the flag; the run writes the native
    case-9 series and its record names IN4 and IN5 as gaps, printed loudly (woof.hex.correct_mode: a
    correction the run cannot apply is a named gap, never a refusal).  The record never says a
    correction was applied when the engine could not apply it.
    """
    from .hrrr_intermediate import IntermediateRefusal
    from . import correct_mode
    if mode == "exact":
        return [], None
    if mode != "correct":
        raise IntermediateRefusal(f"--physics-mode {mode!r} is not one of {LBC_PHYSICS_MODES}")
    probe = subprocess.run([str(engine), "--abi"], capture_output=True, text=True, timeout=60)
    if probe.returncode != 0 or "--physics-mode" not in probe.stdout:
        gaps = [correct_mode.CorrectionGap(
            finding, "engine rw_mpas_lbc",
            f"{engine} does not speak --physics-mode (its --abi does not name it); the boundary series "
            "is native init_atmosphere v8.4.2 case 9") for finding in correct_mode.LBC_FINDINGS]
        return [], correct_mode.receipt_block(
            "correct", gaps=correct_mode.report_gaps(gaps, context="boundary door"))
    return ["--physics-mode", "correct"], correct_mode.receipt_block(
        "correct", applied=correct_mode.LBC_FINDINGS)


def lbc_physics_mode_args(engine, mode):
    """The rw_mpas_lbc arguments for one physics mode (see :func:`lbc_physics_mode`)."""
    return lbc_physics_mode(engine, mode)[0]


def run_stream(arguments, engine):
    from .hrrr_intermediate import IntermediateRefusal
    from . import time_ledger
    start, stop = (datetime.strptime(value, FORMAT) for value in (arguments.start_time, arguments.stop_time))
    span = (stop - start).total_seconds()
    if span < 3600 or span % 3600:
        raise IntermediateRefusal("streamed LBC window must cover complete hours; refuses missing interpolation endpoints")
    output = Path(arguments.out_dir).absolute()
    output.mkdir(parents=True, exist_ok=True)
    times = [start + timedelta(hours=hour) for hour in range(int(span / 3600) + 1)]
    physics, correct_record = lbc_physics_mode(engine, getattr(arguments, "physics_mode", "exact"))
    atomic_json(output / "lbc-series.json", {"schema": SERIES, "valid_times": [t.strftime(FORMAT) for t in times]})
    files = []
    try:
        for hour, moment in enumerate(times):
            met = Path(arguments.met_dir).absolute() / f"MET:{moment:%Y-%m-%d_%H}"
            marker = met.with_name(met.name + ".ready.json")
            started = time.monotonic()
            while not marker.exists():
                if time.monotonic() - started > 1800:
                    raise IntermediateRefusal(f"missing_intermediate_hour {hour}: {marker}; refuses incomplete boundary input")
                time.sleep(0.05)
            admitted = checked_marker(marker, schema="gpuwm-hex.intermediate.hour/v1")
            if admitted != met:
                raise IntermediateRefusal(f"{marker}: unexpected MET path; refuses the wrong boundary hour")
            staging = output / f".hour-{hour:04d}"
            staging.mkdir()
            argv = [str(engine), "--grid", str(Path(arguments.grid).absolute()), "--out-dir", str(staging),
                    "--start-time", arguments.start_time, "--stop-time", arguments.stop_time,
                    "--interval", f"{moment:{FORMAT}}={met}", "--nfglevels", str(arguments.nfglevels),
                    "--fg-interval-seconds", "3600", "--extrap-airtemp", arguments.extrap_airtemp,
                    "--use-spechumd", arguments.use_spechumd, "--theta-adv-order", str(arguments.theta_adv_order),
                    "--coef-3rd-order", repr(float(arguments.coef_3rd_order)), "--oned-underflow", arguments.oned_underflow,
                    *physics, "--receipt", str(staging / "receipt.json")]
            attrs = {"hour": hour, "valid_time": moment.strftime(FORMAT)}
            time_ledger.emit("prep.lbc.hour", "begin", attrs=attrs)
            completed = subprocess.run(argv, capture_output=True, text=True)
            (output / f"lbc-hour-{hour:04d}.log").write_text(
                "$ " + " ".join(argv) + f"\nexit {completed.returncode}\n" + completed.stdout + completed.stderr)
            if completed.returncode:
                raise IntermediateRefusal(f"rw_mpas_lbc exited {completed.returncode}; log {output / f'lbc-hour-{hour:04d}.log'}")
            produced = staging / file_name(moment.strftime(FORMAT))
            destination = output / produced.name
            produced.replace(destination)
            document = {"schema": READY, **attrs, "path": str(destination),
                        "bytes": destination.stat().st_size, "sha256": sha256(destination)}
            atomic_json(destination.with_name(destination.name + ".ready.json"), document)
            files.append(document)
            time_ledger.emit("prep.lbc.hour", "end", attrs={**attrs, "status": "complete"})
            time_ledger.emit("prep.lbc.hour", attrs=document)
            print(f"READY {destination}", flush=True)
        atomic_json(output / "lbc-door-receipt.json", {"schema": "gpuwm-hex.lbc-door/v1", "streamed": True, "files": files,
                                                        **({"correct_mode": correct_record} if correct_record else {})})
    except BaseException as error:
        time_ledger.emit("prep.lbc.hour", "end", attrs={"status": "failed", "error": str(error)})
        atomic_json(output / "lbc-failed.json", {"error": str(error)})
        raise
    return 0


def source_endpoint_commands(*, source, cycle, hours, out_dir, staged=(), member=None, table=None, lead_hours=0,
                             boundary_interval_hours=None):
    """Plan source-table endpoints, including the first endpoint beyond the run.

    ``lead_hours`` starts the run at that forecast hour of ``cycle`` (a forecast row only): the run's start is
    cycle + lead_hours, and endpoint ``index`` reads the same cycle at lead_hours + index * cadence. Before it, a
    run could only start at a cycle's first hour, so a run over a source's day-14 pattern had no front door.

    The cadence of a forecast row is the finest step its ladder posts over the run (init_sources.boundary_step:
    GFS 1 h to 120 h, IFS and GEFS 3 h), or ``boundary_interval_hours`` when given. It was the row's
    ``interval_hours`` (6 h for all of them), so a region from GFS was driven by 6-hourly boundaries.
    """
    import math
    from .init_sources import load_table, resolve_source, Cycle, plan_products, boundary_step
    from .errors import MpasPortError
    table = load_table(table)
    row = resolve_source(table, source)
    if not math.isfinite(hours) or hours <= 0:
        raise MpasPortError("limited-area hours must be finite and positive")
    if boundary_interval_hours is not None and int(boundary_interval_hours) < 1:
        raise MpasPortError(f"--boundary-interval-hours {boundary_interval_hours} is not a positive number of hours")
    issued = Cycle.parse(cycle)
    forecast = row.get("forecast")
    if forecast:
        # No posted step: fall back to the asked or declared step, so plan_products below refuses the lead by name.
        cadence = (boundary_step(forecast, issued.when.hour, lead_hours, hours, requested=boundary_interval_hours)
                   or int(boundary_interval_hours or forecast["interval_hours"]))
    else:
        cadence = int(boundary_interval_hours or 1)
    count = math.ceil(hours / cadence)
    if lead_hours and not row.get("forecast"):
        raise MpasPortError(
            f"--lead-hours {lead_hours}: {row['id']} is an analysis source with no forecast hours; "
            "start from one of its analysis times instead")
    start = Cycle(issued.when + timedelta(hours=lead_hours))
    commands = []
    for index in range(count + 1):
        lead = index * cadence
        source_cycle = issued if row.get("forecast") else Cycle(start.when + timedelta(hours=lead))
        source_lead = lead_hours + lead if row.get("forecast") else 0
        # Validate cadence, member, and product templates before any transfer.
        plan_products(row, source_cycle, member, source_lead)
        argv = ["init-source", "--source", row["id"], "--cycle", source_cycle.when.strftime("%Y-%m-%dT%H"),
                "--lead-hours", str(source_lead), "--stop-after", "decode",
                "--out-dir", str(Path(out_dir) / f"endpoint-{lead:04d}")]
        # Only the first endpoint is the run's initial state (country_door points it at the init's own decode),
        # so only it takes the cold land donor. Later endpoints are boundary meteorology: GFS's donor row is
        # analysis-only and refused every one of them, so each GFS limited-area region stopped at its 1 h
        # endpoint (the WOOF 0.4.10 bake's region warm, 2026-10-30; regional_door already did this).
        if lead:
            argv += ["--soil-remap", "off"]
        if table.get("_path"):
            argv += ["--table", table["_path"]]
        if member is not None:
            argv += ["--member", str(member)]
        for path in staged:
            argv += ["--staged", str(path)]
        commands.append({"valid_time": (start.when + timedelta(hours=lead)).strftime(FORMAT), "argv": argv})
    return commands


def produce_source_endpoints(commands, met_dir, *, workers=2, runner=None):
    """Acquire/decode endpoints concurrently; only Rust touches field data."""
    from concurrent.futures import ThreadPoolExecutor
    from .errors import MpasPortError
    from .init_sources import load_table
    if workers < 1:
        raise MpasPortError("boundary endpoint workers must be positive")
    if runner is None:
        from .cli import main
        runner = main
    met_dir = Path(met_dir)
    met_dir.mkdir(parents=True, exist_ok=True)
    def produce(item):
        argv = item["argv"]
        root = Path(argv[argv.index("--out-dir") + 1])
        receipt = root / "init-source-receipt.json"
        data = None
        if receipt.is_file():
            candidate = json.loads(receipt.read_text())
            expected_cycle = argv[argv.index("--cycle") + 1].replace("T", "_") + ":00:00"
            expected_member = int(argv[argv.index("--member") + 1]) if "--member" in argv else None
            expected_lead = int(argv[argv.index("--lead-hours") + 1])
            expected_table = load_table(Path(argv[argv.index("--table") + 1]) if "--table" in argv else None)
            met = Path(candidate.get("decode", {}).get("met", ""))
            if (candidate.get("table", {}).get("sha256") == expected_table["_sha256"]
                    and candidate.get("source") == argv[argv.index("--source") + 1]
                    and candidate.get("source_cycle") == expected_cycle
                    and candidate.get("lead_hours") == expected_lead
                    and candidate.get("member") == expected_member
                    and met.is_file() and sha256(met) == candidate["decode"].get("met_sha256")):
                data = candidate
        if data is None:
            rc = runner(argv)
            if rc:
                raise MpasPortError(f"boundary source endpoint {item['valid_time']} exited {rc}")
            data = json.loads(receipt.read_text())
        if data["valid_time"] != item["valid_time"]:
            raise MpasPortError(f"boundary endpoint clock differs: {receipt}")
        source = Path(data["decode"]["met"])
        if sha256(source) != data["decode"]["met_sha256"]:
            raise MpasPortError(f"boundary intermediate digest differs: {source}")
        dest = met_dir / ("MET:" + item["valid_time"][:13])
        if dest.is_symlink():
            dest.unlink()
        elif dest.exists():
            raise MpasPortError(f"boundary destination already exists: {dest}")
        dest.symlink_to(source)
        return {"valid_time": item["valid_time"], "receipt": str(receipt), "met": str(dest)}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        result = list(pool.map(produce, commands))
    atomic_json(met_dir / "source-endpoints.json", {"schema": "gpuwm-hex.source-endpoints/v1",
                "workers": workers, "endpoints": result})
    return result
