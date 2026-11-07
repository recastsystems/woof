"""Prepare a table-defined regional cycle using public bytes and native doors.

This module transfers and describes files. Native engines construct the
meteorological, vertical and boundary fields. A manifest is published only
after every door has succeeded. No supplied native initialization is used.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
from importlib import import_module, resources
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import urlopen

from .source_profiles import source_profile
from .errors import MpasPortError


def entry(name):
    module, function = name.split(":", 1)
    return getattr(import_module(module), function)


def describe(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def acquire(url, row, path):
    """Require the table's object identity; publish complete transfers only."""
    receipt = path.with_name(path.name + ".source.json")
    if path.exists():
        held = json.loads(receipt.read_text()) if receipt.exists() else None
        actual = describe(path)
        if (held and held["url"] == url and actual == held["file"]
                and (not row.get("s3_etag") or held['s3_etag'] == row['s3_etag'])
                and (not row.get("bytes") or actual['bytes'] == row['bytes'])):
            return held
        raise ValueError(f"{path} exists without matching source bytes; refusing cache reuse")
    partial = path.with_name(path.name + ".partial")
    if partial.exists():
        raise ValueError(f"{partial} is an incomplete transfer; move it aside before retrying")
    digest = hashlib.sha256()
    count = 0
    with urlopen(url, timeout=120) as response:
        etag = response.headers.get("ETag", "").strip('"')
        if row.get("s3_etag") and etag != row["s3_etag"]:
            raise ValueError(f"public asset {url} ETag changed: {etag}")
        with partial.open("xb") as stream:
            for block in iter(lambda: response.read(8 << 20), b""):
                count += len(block)
                digest.update(block)
                stream.write(block)
    if row.get("bytes") and count != row["bytes"]:
        raise ValueError(f"public asset {url} is truncated: {count} bytes")
    partial.rename(path)
    result = {"url": url, "s3_etag": etag,
              "file": {"path": str(path), "bytes": count, "sha256": digest.hexdigest()}}
    receipt.write_text(json.dumps(result, indent=2) + "\n")
    return result


def checked_command(argv, log):
    # Keep the failure that explains a retry; never overwrite its evidence.
    attempt = 1
    original = log
    while log.exists():
        log = original.with_name(f"{original.stem}.attempt-{attempt}{original.suffix}")
        attempt += 1
    start = time.perf_counter()
    with log.open("x") as stream:
        stream.write(json.dumps(argv) + "\n")
        stream.flush()
        result = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT)
    record = {"argv": argv, "log": str(log), "seconds": time.perf_counter()-start,
              "returncode": result.returncode}
    log.with_suffix(".timing.json").write_text(json.dumps(record, indent=2) + "\n")
    if result.returncode:
        raise ValueError(f"preparation stopped with exit {result.returncode}; see {log}")
    return record


def publish_config(path, content):
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"{path} differs from the selected source row; refusing mixed preparation")
        return
    with path.open("xb") as stream:
        stream.write(content)


def request(args):
    cycle = datetime.strptime(args.cycle, "%Y%m%d%H").replace(tzinfo=timezone.utc)
    if cycle.strftime("%Y%m%d%H") != args.cycle or cycle.hour % 6:
        raise ValueError("cycle must be zero-padded UTC YYYYMMDDHH at a six-hour cycle")
    profile = source_profile(args.source_profile)
    mesh_set = profile["mesh_set"]
    mesh = mesh_set["meshes"][args.mesh]
    source = source_profile(mesh_set["preparation_source"])
    interval = source["interval_hours"]
    if args.hours < 1 or args.hours % interval:
        raise ValueError(f"hours must be a positive multiple of {interval}")
    return cycle, profile, mesh, source


def prepare(args):
    cycle, profile, mesh, source = request(args)
    out = args.out.expanduser().absolute()
    # Re-running into a published cycle must not mix old and new inputs.
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "prepared.json"
    if manifest.exists():
        raise ValueError(f"{manifest} already exists; select a fresh cycle directory")
    from .init_door import resolve_engine
    engine = resolve_engine(None)
    help_result = subprocess.run([str(engine), "--help"], capture_output=True, text=True)
    from .physics_suites import load_table, suite_selectors
    from .species_row import species_row_for_scheme
    selectors = suite_selectors(args.physics_suite, args.mesh)
    soil = load_table()["options"]["config_lsm_scheme"][selectors["config_lsm_scheme"]]
    target_levels = soil.get("carried", soil).get("n_soil_levels", 4)
    from .cli import build_parser
    init_parser = build_parser()._subparsers._group_actions[0].choices["init"]
    suite_available = any(action.dest == "physics_suite" for action in init_parser._actions)
    species = species_row_for_scheme(selectors["config_microp_scheme"]).names()
    if target_levels != 4 and (not suite_available or "--land-surface" not in help_result.stdout + help_result.stderr):
        raise ValueError(f"native init lacks suite soil geometry: its four layers cannot initialize "
                         f"the selected {target_levels}-level suite. Install the suite-aware init door and bridge")
    if "nwfa" in species and not suite_available:
        raise ValueError("init lacks suite aerosol initialization: the requested species would receive "
                         "synthetic concentrations. Install the suite-aware init door and bridge")
    from .engines import EngineSpec, resolve
    met_engine = resolve(EngineSpec(source["intermediate"]["binary"], "--met-engine",
        ("WOOF_HEX_MET_INTERMEDIATE",), "WOOF_MET_INTERMEDIATE",
        "the regular-grid intermediate producer", "no driving MET or boundary fields can be built",
        "met_intermediate"), None)
    logs = out / "logs"
    logs.mkdir(exist_ok=True)
    inputs = {}
    for name, key in profile["mesh_set"]["preparation_assets"].items():
        # A name may carry {mesh}: the forecast door finds the suite's
        # orographic statistics beside the grid as <mesh>.ugwp_oro_data.nc.
        name = name.format(mesh=args.mesh)
        path = out / (name + ".nc")
        row = mesh[key]
        inputs[name] = acquire(profile["mesh_set"]["fix_root"] + row["key"], row, path)
    spec_member = resources.files("woof.hex").joinpath(profile["vertical"]["preparation_vertical_spec"])
    spec = out / "vertical.json"
    publish_config(spec, spec_member.read_bytes())
    # Archive selection, inventories and transfers stay on the engine's fetch road.
    raw = out / "raw"
    entry(source["fetch_entry"])(cycle=cycle,
        hours=tuple(range(0, args.hours+1, source["interval_hours"])), area=None,
        out=raw, source=source["fetch_source"], all_levels=True)
    met = out / "met"
    met.mkdir(exist_ok=True)
    table_row = source["intermediate"]["vtable"]
    table_member = resources.files("woof.hex").joinpath(table_row["resource"])
    table_bytes = table_member.read_bytes()
    if hashlib.sha256(table_bytes).hexdigest() != table_row["sha256"]:
        raise ValueError("shipped intermediate table differs from its source-profile identity")
    vtable = out / "Vtable"
    publish_config(vtable, table_bytes)
    commands = []
    for hour in range(0, args.hours+1, source["interval_hours"]):
        candidates = list(raw.glob(f"*.pgrb2.*.f{hour:03d}"))
        if len(candidates) != 1:
            raise ValueError(f"native fetch produced {len(candidates)} GRIB files for hour {hour}")
        stamp = (cycle + timedelta(hours=hour)).strftime("%Y-%m-%d_%H:%M:%S")
        commands.append(checked_command([str(met_engine), "--vtable", str(vtable),
            "--date", stamp, "--map-source", source["intermediate"]["map_source"],
            "--out", str(met / ("MET:" + stamp)), str(candidates[0])], logs / f"met-{hour:03d}.log"))
    stamp = cycle.strftime("%Y-%m-%d_%H:%M:%S")
    switches = {**source["init_switches"],
                "landuse-table": profile["mesh_set"]["preparation_landuse_table"]}
    argv = [sys.executable, "-m", "woof.hex.cli", "init", "--grid", str(out / "grid.nc"),
        "--static", str(out / "static.nc"),
        "--met", str(met / ("MET:" + stamp)), "--out", str(out / "init.nc"),
        "--start-time", stamp, "--nfglevels", str(source["intermediate"]["nfglevels"]),
        "--nfgsoillevels", str(source["intermediate"]["nfgsoillevels"])]
    if suite_available:
        argv.extend(["--physics-suite", args.physics_suite])
    for flag, value in switches.items():
        argv.extend(["--" + flag, str(value)])
    vertical_row = profile['mesh_set']['preparation_vertical']
    if vertical_row['mode'] == 'published':
        from .published_vertical import materialize
        from .init_door import run_declarations
        vertical_source = out / (vertical_row['asset'] + '.nc')
        capsule = out / 'vertical-artifact.nc'
        argv.extend(['--capsule', str(capsule), '--reference', str(vertical_source)])
        init_arguments = build_parser().parse_args(argv[3:])
        materialize(vertical_source, out / 'grid.nc', capsule,
                    run_declarations(init_arguments), static=out / 'static.nc')
    else:
        argv.extend(['--vertical-spec', str(spec)])
    commands.append(checked_command(argv, logs / "init.log"))
    argv = [sys.executable, "-m", "woof.hex.cli", "lbc", "--grid", str(out / "init.nc"),
        "--met-dir", str(met), "--out-dir", str(out / "lbc"), "--start-time", stamp,
        "--stop-time", (cycle+timedelta(hours=args.hours)).strftime("%Y-%m-%d_%H:%M:%S"),
        "--nfglevels", str(source["intermediate"]["nfglevels"])]
    for flag in ("extrap-airtemp", "use-spechumd", "theta-adv-order", "coef-3rd-order", "oned-underflow"):
        argv.extend(["--" + flag, str(switches[flag])])
    commands.append(checked_command(argv, logs / "lbc.log"))
    result = {"schema": "gpuwm-hex.prepared-cycle/v1", "source_profile": args.source_profile,
        "physics_suite": args.physics_suite,
        "driving_profile": profile["mesh_set"]["preparation_source"], "mesh": args.mesh,
        "cycle": args.cycle, "hours": args.hours, "inputs": inputs, "commands": commands,
        "init": describe(out / "init.nc"),
        "lbc": [describe(path) for path in sorted((out / "lbc").glob("*.nc"))]}
    if len(result["lbc"]) != args.hours // source["interval_hours"] + 1:
        raise ValueError("boundary door did not publish every requested time")
    with manifest.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-profile", required=True)
    parser.add_argument("--physics-suite", required=True)
    parser.add_argument("--cycle", required=True)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--hours", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(prepare(args), indent=2))
        return 0
    except (MpasPortError, OSError, ValueError, KeyError) as error:
        print(f"public preparation: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
