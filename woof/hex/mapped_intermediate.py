"""Regional intermediate series from the shared canonical composition table.

Each lead is independently composed because a source grid may move. Source
names, products, cadence, selectors and donors are all table declarations.
"""
from __future__ import annotations

from datetime import timedelta
import json
from pathlib import Path
import subprocess
import sys
import time
import tempfile
import shutil


def run(args, row):
    from .hrrr_intermediate import (parse_cycle, parse_hours, target_from_plan,
                                   target_for_cap, IntermediateRefusal, sha256_file)
    from woof.initial_source import initial_inputs_from_fetch
    from woof.storm_merge import command_arguments
    spec = row["intermediate"]
    engine_source = spec.get("engine_source_id", row["engine_source_id"])
    parameters = getattr(args, "source_parameters", None)
    cycle = parse_cycle(args.cycle)
    requested = parse_hours(args.hours)
    cadence = spec["cadence_hours"]
    if requested[0] % cadence or requested[-1] % cadence:
        raise IntermediateRefusal("requested endpoints must lie on the source table's lead cadence")
    hours = range(requested[0], requested[-1] + 1, cadence)
    if args.from_plan:
        target, basis = target_from_plan(args.from_plan, margin_km=args.margin_km,
                                         spacing_deg=args.spacing_deg)
    elif args.point and args.radius_km:
        from .mesh_point import parse_point
        target = target_for_cap(parse_point(args.point), args.radius_km,
                                margin_km=args.margin_km, spacing_deg=args.spacing_deg)
        basis = {"center": args.point, "radius_km": args.radius_km}
    else:
        raise IntermediateRefusal("a mapped regional source needs --from-plan or --point and --radius-km")
    area = f"{target.south},{target.west},{target.south+(target.ny-1)*target.dlat},{target.west+(target.nx-1)*target.dlon}"
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    files = []
    started = time.perf_counter()
    for hour in hours:
        moment = cycle + timedelta(hours=hour)
        dest = Path(args.grib_dir).resolve() / f"lead-{hour:03d}"
        dest.mkdir(parents=True, exist_ok=True)
        fetch = [sys.executable, "-m", "woof", "fetch", "--source", engine_source,
                 "--cycle", args.cycle, "--hours", "0", "--forecast-start-hour", str(hour),
                 "--whole-cycle", "--out", str(dest)]
        if parameters:
            fetch += ["--source-parameters", parameters]
        subprocess.run(fetch, check=True)
        inventory = initial_inputs_from_fetch(engine_source, dest, dest / "initial-inputs.json")
        declaration = json.loads(inventory.read_text())
        met = out / f"{args.prefix}:{moment:%Y-%m-%d_%H}"
        # The receipt and declarations survive; full canonical weather arrays
        # belong only to this lead and are removed before fetching the next.
        workspace = tempfile.TemporaryDirectory(prefix=f"composition-{hour:03d}-", dir=out)
        work = Path(workspace.name)
        try:
            command = [sys.executable, "-m", "woof", "source-intermediate", "--source", engine_source,
                       "--profile", spec["profile"], "--valid-time", moment.isoformat(),
                       "--out", str(met), "--work-dir", str(work), "--workers", str(args.workers),
                       "--coverage", "regional", "--target-area", area]
            for path in declaration["input_files"]:
                command += ["--input", path]
            for role, paths in declaration["supplements"].items():
                for path in paths:
                    command += ["--supplement", f"{role}={path}"]
            # A merge changes the initial state only; later lateral forcing remains
            # the background source. Every lead's ordinary output is still retained.
            if hour == requested[0]:
                command += command_arguments(args)
            subprocess.run(command, check=True)
            export = json.loads(met.with_name(met.name + ".receipt.json").read_text())
            evidence = out / f"evidence-{hour:03d}"
            for path in work.rglob("*.json"):
                target_path = evidence / path.relative_to(work)
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target_path)
            expected = row["first_guess"]
            if (export["nfglevels"], export["nfgsoillevels"]) != (expected["nfglevels"], expected["nfgsoillevels"]):
                raise IntermediateRefusal("exported vertical dimensions differ from the source table")
            files.append(dict(path=str(met), sha256=sha256_file(met), valid_time=moment.isoformat(),
                              export_receipt=str(met.with_name(met.name + ".receipt.json"))))
        finally:
            workspace.cleanup()
    receipt = dict(schema="gpuwm-hex.mapped-intermediate-series/v1", source=row["id"],
                   profile=spec["profile"], source_parameters=parameters,
                   cadence_hours=cadence, target_area=area, target_basis=basis, files=files,
                   levels=dict(count=row["first_guess"]["nfglevels"]),
                   elapsed_seconds=time.perf_counter()-started)
    (out / "intermediate-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))
    return 0
