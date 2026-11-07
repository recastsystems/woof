"""Compose a packaged source and export its canonical state to WPS in Rust.

The output is consumed by the existing hex init door. No weather arrays enter
Python and no model-specific hex adapter is selected here.

Two shapes, one door:

* **global** (the original): one valid time, the source's own lattice, which
  must cover the globe;
* **a regional box** (``--target-latlon``): any packaged source -- regular,
  rotated (HRDPS, RDPS) or Lambert -- composed once over the box's
  atmospheric window and resampled in Rust onto a regular lat-lon box, one
  WPS file per valid time, for a limited-area run's init and boundaries.

``--fetch-dir`` reads the inputs and supplements a ``woof fetch`` of a
table route recorded in its manifest (the same ``prep`` block ``woof prep``
reads), so a caller names a fetch, not a list of files.
"""
from __future__ import annotations
from datetime import datetime
import json
from pathlib import Path
import subprocess


def register_cli(sub):
    parser = sub.add_parser("source-intermediate",
                            help="compose a packaged source into WPS intermediate for init (global, or a regional box)")
    parser.add_argument("--source", help="source registry name, including earth2-<table key>")
    parser.add_argument("--soil-set", choices=("nodes", "layers"), default="layers")
    parser.add_argument("--profile", help="packaged composition profile (overrides the source default)")
    parser.add_argument("--input", action="append", type=Path, default=[])
    parser.add_argument("--fetch-dir", type=Path,
                        help="a `woof fetch` output folder: its manifest's prep block names the inputs, "
                             "the supplements and the source")
    parser.add_argument("--supplement", action="append", default=[], metavar="ROLE=FILE")
    parser.add_argument("--valid-time", action="append", default=[],
                        help="frame time YYYY-MM-DDTHH:MM:SS (repeat for several; with --out-dir and none "
                             "given, every composed valid time)")
    parser.add_argument("--member", help="the frame's ensemble member (the GRIB member number the compose decodes) "
                        "that the WPS export selects")
    parser.add_argument("--member-identity", help="also seal --member into the input manifest under this identity "
                        "policy (mapped_authoring); without it the member only selects the composed frame")
    parser.add_argument("--target-latlon", metavar="SOUTH,WEST,DLAT,DLON,NX,NY",
                        help="a regional run's regular lat-lon box: compose over its window and resample onto it")
    parser.add_argument("--target-support", type=Path,
                        help="mesh support contract for this lattice (cell and edge interpolation)")
    parser.add_argument("--target-support-sha256", help="caller-authenticated support contract digest")
    parser.add_argument("--out", type=Path, help="the WPS file (one valid time)")
    parser.add_argument("--out-dir", type=Path, help="write PREFIX:YYYY-MM-DD_HH per valid time here")
    parser.add_argument("--prefix", default="MET")
    parser.add_argument("--coverage", choices=("global", "regional"), default="global")
    parser.add_argument("--target-area", help="target coverage SOUTH,WEST,NORTH,EAST with source-grid halo")
    from woof.storm_merge import add_arguments
    add_arguments(parser)
    from woof.tc.mapped_vortex import add_arguments as add_vortex_arguments
    add_vortex_arguments(parser)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.set_defaults(func=main)


class _LatLonBox:
    """The target box as the atmospheric-window planner reads a grid: mass,
    u and v points all at the box's nodes (its resampling reads no other)."""

    def __init__(self, south, west, dlat, dlon, nx, ny):
        import numpy as np
        latitudes = south + dlat * np.arange(ny, dtype=np.float64)
        longitudes = west + dlon * np.arange(nx, dtype=np.float64)
        self._mesh = (np.repeat(latitudes[:, None], nx, axis=1),
                      np.repeat(longitudes[None, :], ny, axis=0))

    def latlon_mass(self):
        return self._mesh

    latlon_u = latlon_mass
    latlon_v = latlon_mass


def _parse_box(text: str) -> tuple[float, float, float, float, int, int]:
    parts = [part.strip() for part in str(text).split(",")]
    if len(parts) != 6:
        raise ValueError("--target-latlon takes SOUTH,WEST,DLAT,DLON,NX,NY")
    south, west, dlat, dlon = (float(value) for value in parts[:4])
    nx, ny = int(parts[4]), int(parts[5])
    if dlat <= 0 or dlon <= 0 or nx < 2 or ny < 2:
        raise ValueError("--target-latlon needs positive increments and at least 2 x 2 points")
    return south, west, dlat, dlon, nx, ny


def _fetch_inputs(fetch_dir: Path) -> tuple[str, list[Path], dict[str, list[Path]]]:
    """``(source, primary files, supplements)`` a fetch recorded for its prep."""

    manifest_path = Path(fetch_dir) / "fetch-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"--fetch-dir {fetch_dir}: no readable fetch-manifest.json ({error})") from error
    if not manifest.get("complete", True):
        raise ValueError(f"--fetch-dir {fetch_dir}: the fetch is not complete; rerun woof fetch first")
    prep = manifest.get("prep") or {}
    primary = [Path(fetch_dir, name).resolve() for name in prep.get("primary_files") or ()]
    if not primary:
        raise ValueError(f"--fetch-dir {fetch_dir}: its manifest names no primary files to compose")
    supplements: dict[str, list[Path]] = {}
    role = prep.get("supplement_role")
    if role:
        supplements[str(role)] = [Path(fetch_dir, name).resolve() for name in prep.get("supplement_files") or ()]
    return str(prep.get("source") or manifest.get("request", {}).get("source") or ""), primary, supplements


def _met_name(prefix: str, valid_time: str) -> str:
    moment = datetime.strptime(valid_time, "%Y-%m-%dT%H:%M:%S")
    stamp = moment.strftime("%Y-%m-%d_%H") if moment.minute == 0 and moment.second == 0 \
        else moment.strftime("%Y-%m-%d_%H:%M:%S")
    return f"{prefix}:{stamp}"


def _file_sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _deep_soil(composition_path, out: Path, valid_time: str, receipt: dict, work_dir: Path):
    """The composition's declared deep-soil donor, appended to one WPS file."""

    from woof.ingest.deep_soil_donor import append_deep_soil_to_wps, has_deep_donor

    composition = json.loads(Path(composition_path).read_text(encoding="utf-8"))
    contract = composition.get("soil_layers")
    if not has_deep_donor(contract):
        return None
    cycle = receipt.get("source_cycle")
    deep = append_deep_soil_to_wps(
        out, contract, valid_time=datetime.strptime(valid_time, "%Y-%m-%dT%H:%M:%S"),
        primary_cycle=datetime.fromisoformat(str(cycle).replace("Z", "")) if cycle else None,
        cache_dir=Path(work_dir) / "deep-soil")
    deep["layers_cm"] = [[round(float(layer["top"]) * 100), round(float(layer["bottom"]) * 100)]
                         for layer in contract["deep_donor"]["layers"]]
    return deep


def _upper_air(composition_path, out: Path, valid_time: str, receipt: dict, work_dir: Path):
    """The composition's declared upper-air donor, applied to one WPS file."""

    from woof.ingest.upper_air_donor import (UPPER_AIR_DONOR_KEY, append_upper_air_to_wps,
                                              has_upper_air_donor, validate_upper_air_donor)

    composition = json.loads(Path(composition_path).read_text(encoding="utf-8"))
    if not has_upper_air_donor(composition):
        return None
    cycle = receipt.get("source_cycle")
    return append_upper_air_to_wps(
        out, validate_upper_air_donor(composition[UPPER_AIR_DONOR_KEY]),
        valid_time=datetime.strptime(valid_time, "%Y-%m-%dT%H:%M:%S"),
        primary_cycle=datetime.fromisoformat(str(cycle).replace("Z", "")) if cycle else None,
        cache_dir=Path(work_dir) / "upper-air")


def _donors(composition_path, out: Path, valid_time: str, receipt: dict, work_dir: Path) -> dict:
    """Every declared donor on one written WPS file: the deep soil, then the air above the top."""

    deep_soil = _deep_soil(composition_path, out, valid_time, receipt, work_dir)
    if deep_soil is not None:
        # The donor's layers sit below the source's own in the file, so
        # the init reads the stacked first guess (woof/ingest/deep_soil_donor.py).
        receipt["deep_soil"] = deep_soil
        receipt["soil_layers_cm"] = list(receipt["soil_layers_cm"]) + deep_soil["layers_cm"]
        receipt["nfgsoillevels"] = len(receipt["soil_layers_cm"])
    upper = _upper_air(composition_path, out, valid_time, receipt, work_dir)
    if upper is not None:
        # The donor's levels sit above the source's top in the file, so the
        # init and the boundaries read the deeper column (woof/ingest/upper_air_donor.py).
        receipt["upper_air"] = upper
        receipt["nfglevels"] = int(receipt["nfglevels"]) + int(upper["appended_levels"])
    if deep_soil is not None or upper is not None:
        receipt["output"]["bytes"] = out.stat().st_size
        receipt["output"]["sha256"] = _file_sha256(out)
    return receipt


def _tc_transform(args, valid_time):
    from woof.storm_merge import declaration, compose_donor, run_merge
    merge = declaration(args)
    frame_dir = args.work_dir / "composed"
    merge_receipt = None
    if merge:
        donor, donor_execution, donor_evidence = compose_donor(merge, args.work_dir / "donor", args.workers)
        merge_receipt, merge_request = run_merge(frame_dir, donor, args.work_dir / "merged",
            valid_time=valid_time, declaration=merge, workers=args.workers)
        merge_receipt["donor_compose"] = donor_execution["receipt"]
        frame_dir = args.work_dir / "merged"
        evidence_dir = args.work_dir / "merge-evidence"
        evidence_dir.mkdir()
        for name, payload in donor_evidence.items():
            (evidence_dir / name).write_bytes(payload)
        (evidence_dir / "merge-request.json").write_bytes(merge_request)
    # TC-VI1 (woof/tc/mapped_vortex.py) on the canonical frame the WPS export reads, after any storm merge (HWRF/HAFS
    # order: the merged core is then relocated and corrected as one vortex). Breakage it prevents (the storm box M,
    # 2026-10-27, hex port): the hex init door had no vortex step for any source but a hand-sealed GFS request, so
    # GEFS members reached the hex mesh with their own weak, broad vortices.
    from woof.tc.mapped_vortex import declaration as vortex_declaration
    vortex = vortex_declaration(args)
    vortex_receipt = None
    if vortex:
        # one member's own vortex: the compose holds exactly that member's frame (the WPS export below selects it with
        # --member); relocate() refuses unless exactly one frame is valid at the initial time
        import hashlib
        from woof.tc.mapped_vortex import library_path, relocate
        library = library_path()
        manifest, transform, member_vortex, record = relocate(
            library=library, input_directory=frame_dir, output_directory=args.work_dir / "relocated",
            valid_time=valid_time, vitals_path=vortex["vitals"],
            size_change_limit=vortex.get("size_change_limit", .15))
        applied = transform.get("declaration", {})
        vortex_receipt = {
            "schema": "gpuwm-source-intermediate-vortex-init-v1", "mode": "relocate", "declaration": vortex,
            "member_diagnosis": member_vortex, "vitals_record": record,
            "applied_target_rmw_m": applied.get("target_rmw_m"),
            "background_manifest_sha256": hashlib.sha256((frame_dir / "frames.json").read_bytes()).hexdigest(),
            "relocated_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
            "transform_wall_seconds": transform.get("total_wall_seconds")}
        print(f"vortex-init: member centre {[round(x, 2) for x in member_vortex['centre_deg']]} "
              f"{member_vortex['minimum_sea_level_pressure_pa'] / 100:.1f} hPa RMW {member_vortex['rmw_m'] / 1000:.0f} km "
              f"-> vitals {record['latitude_deg']},{record['longitude_deg']} {record['minimum_pressure_pa'] / 100:.0f} hPa "
              f"{record['maximum_wind_m_s']:.0f} m/s RMW {record['radius_maximum_wind_m'] / 1000:.0f} km", flush=True)
        frame_dir = args.work_dir / "relocated"
    return frame_dir, merge_receipt, vortex_receipt


def main(args):
    from woof.mapped_authoring import author_input_manifest
    from woof.mapped_engine_bridge import require_engine, run_engine
    from woof.source_adapters import get_source_adapter
    from woof.source_authorities import (packaged_authorities,
        packaged_contributing_mappings, packaged_provenance_files)

    inputs = [Path(path) for path in args.input]
    supplements: dict[str, list[Path]] = {}
    source = args.source
    if args.fetch_dir is not None:
        fetched_source, fetched, fetched_supplements = _fetch_inputs(args.fetch_dir)
        source = source or fetched_source
        inputs += fetched
        for role, paths in fetched_supplements.items():
            supplements.setdefault(role, []).extend(paths)
    if not inputs:
        raise ValueError("source-intermediate needs --input FILE... or --fetch-dir DIR")
    profile = args.profile
    if not profile and source:
        adapter = get_source_adapter(source)
        profile = adapter.packaged_profile
        # The family's registry aliases share an adapter, while the model table
        # owns each model's profile. Resolve the alias through that same table.
        if adapter.source_id == "earth2":
            from woof.earth2_run import resolve_model
            _, row = resolve_model(source.removeprefix("earth2-"))
            profile = row["profile"]
    if not profile:
        raise ValueError("source-intermediate requires --profile or a source with a packaged profile")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if (args.out is None) == (args.out_dir is None):
        raise ValueError("source-intermediate writes either --out FILE (one valid time) or --out-dir DIR")
    if args.out is not None and len(args.valid_time) != 1:
        raise ValueError("--out names one WPS file, so it takes exactly one --valid-time; use --out-dir for several")
    box = _parse_box(args.target_latlon) if args.target_latlon else None
    support = getattr(args, "target_support", None)
    support_sha256 = getattr(args, "target_support_sha256", None)
    if bool(support) != bool(support_sha256) or (support and box is None):
        raise ValueError("--target-support requires --target-support-sha256 and --target-latlon")
    if support and _file_sha256(support) != support_sha256:
        raise ValueError("target support contract digest differs")
    authorities = packaged_authorities(profile)
    contributors = packaged_contributing_mappings(profile)
    provenance = packaged_provenance_files(profile)
    for binding in args.supplement:
        role, sep, path = binding.partition("=")
        if not sep or not role or not path:
            raise ValueError("--supplement takes ROLE=FILE")
        supplements.setdefault(role, []).append(Path(path).resolve())
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
    else:
        args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = author_input_manifest(
        args.work_dir / "inputs.json", mapping_path=authorities["mapping"],
        composition_path=authorities["composition"], primary_files=inputs,
        supplement_files=supplements, provenance_files=provenance,
        contributing_mappings=contributors,
        # Breakage it prevents: --member alone always refused here ("member and member_identity are an atomic pair"),
        # so no single GEFS member could reach hex through this door (hex TC init, 2026-10-27); the composed frame
        # carries its own GRIB member number either way
        member=args.member if getattr(args, "member_identity", None) else None,
        member_identity=getattr(args, "member_identity", None))
    engine = require_engine()
    capabilities = subprocess.run([str(engine), "capabilities"], check=True,
                                  capture_output=True, text=True)
    features = json.loads(capabilities.stdout.splitlines()[-1])
    if support and not features.get("wps_mesh_support_v1"):
        raise ValueError("source-intermediate needs the mapped engine with mesh support export")
    if not features.get("wps_intermediate_v5"):
        raise ValueError("source-intermediate needs this release's mapped engine with WPS intermediate v5 export")
    composed = run_engine(
        "compose", mapping=authorities["mapping"], files=inputs,
        composition=authorities["composition"], supplements=supplements,
        provenance=provenance, contributing_mappings=contributors,
        input_manifest=manifest["manifest"]["path"],
        input_manifest_sha256=manifest["manifest"]["sha256"],
        output=args.work_dir / "composed", engine=engine, threads=args.workers,
        # The box's window: atmospheric fields are decoded over the box's
        # support alone (a 2.5 km continental source is 3.3 M points a plane).
        atmospheric_grids=(_LatLonBox(*box),) if box else (),
        lead_batch=True, on_progress=lambda event: print(json.dumps(event), flush=True))
    valid_times = list(args.valid_time)
    if not valid_times:
        frames = json.loads((args.work_dir / "composed" / "frames.json").read_text(encoding="utf-8"))["frames"]
        valid_times = sorted({frame["valid_time"] for frame in frames
                              if frame.get("member") == (args.member or None)})
    contract = Path(__file__).with_name("authorities") / "wps-intermediate-fields.v1.json"
    from woof.ingest.preparation_workers import worker_environment
    exported = []
    for valid_time in valid_times:
        out = args.out if args.out is not None else args.out_dir / _met_name(args.prefix, valid_time)
        frame_dir, merge_receipt, vortex_receipt = _tc_transform(args, valid_time)
        if getattr(args, "target_area", None):
            check_target_coverage(frame_dir, valid_time, args.target_area, args.member)
        command = [str(engine), "wps", "--frameset", str(frame_dir),
                   "--composition", str(authorities["composition"]), "--contract", str(contract),
                   "--valid-time", valid_time, "--out", str(out),
                   "--soil-set", getattr(args, "soil_set", "layers")]
        if getattr(args, "coverage", "global") != "global":
            command += ["--coverage", args.coverage]
        if getattr(args, "target_area", None):
            command += ["--target-area", args.target_area]
        if args.member:
            command += ["--member", args.member]
        if box:
            command += ["--target-latlon", ",".join(repr(value) for value in box)]
        if support:
            command += ["--target-support", str(support), "--target-support-sha256", support_sha256]
        result = subprocess.run(command, check=False, text=True, capture_output=True,
                                env=worker_environment(args.workers))
        if result.returncode:
            try:
                refusal = json.loads(result.stderr.splitlines()[-1])
            except (ValueError, IndexError):
                raise RuntimeError(result.stderr.strip() or "Rust WPS export failed without a diagnostic") from None
            message = str(refusal.get("message", "Rust WPS export refused"))
            if refusal.get("class") == "frame_invalid" and "source coverage" in message:
                name = source or profile
                raise ValueError(f"The region extends past {name}'s coverage. Choose a smaller region, "
                                 f"move it inside {name}, or choose a source with a larger domain.")
            raise ValueError(message)
        receipt = json.loads(result.stdout.splitlines()[-1])
        if merge_receipt is not None:
            receipt["storm_merge"] = merge_receipt
        if vortex_receipt is not None:
            receipt["vortex_initialization"] = vortex_receipt
        exported.append((valid_time, out, command, receipt))
    # The declared donors (deep soil, upper air) fetch and sample per valid
    # time, independently of each other: every hour at once, not one by one.
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max(1, min(len(exported), int(args.workers)))) as pool:
        donated = list(pool.map(
            lambda item: _donors(authorities["composition"], item[1], item[0], item[3], args.work_dir),
            exported))
    written = []
    for (valid_time, out, command, _), receipt in zip(exported, donated):
        receipt.update(source=source, profile=profile, compose=composed["receipt"],
                       input_manifest=manifest["manifest"], command=command)
        receipt_path = out.with_name(out.name + ".receipt.json")
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        written.append({"receipt": str(receipt_path), "output": receipt["output"], "valid_time": valid_time,
                        "nfglevels": receipt["nfglevels"], "nfgsoillevels": receipt["nfgsoillevels"]})
    if len(written) == 1 and args.out is not None:
        print(json.dumps({key: written[0][key] for key in ("receipt", "output", "nfglevels", "nfgsoillevels")}))
    else:
        print(json.dumps({"written": written}))
    return 0


def check_target_coverage(frame_dir, valid_time, area, member=None):
    """Reject regional extrapolation before the periodic WPS consumer sees it."""
    import math
    south, west, north, east = map(float, area.split(","))
    if not all(map(math.isfinite, (south, west, north, east))) or south >= north or not 0 < east-west < 360:
        raise ValueError("target area requires finite SOUTH,WEST,NORTH,EAST")
    frames = json.loads((Path(frame_dir) / "frames.json").read_text())["frames"]
    selected = [f for f in frames if f["valid_time"] == valid_time and f.get("member") == member]
    if len(selected) != 1:
        raise ValueError("target coverage needs exactly one selected frame")
    lat = selected[0]["latitude"]["values"]
    lon = selected[0]["longitude"]["values"]
    dy, dx = abs(lat[1] - lat[0]), abs(lon[1] - lon[0])
    if south < min(lat) + 2 * dy or north > max(lat) - 2 * dy:
        raise ValueError("target latitude plus interpolation halo exceeds source footprint")
    if abs(len(lon) * dx - 360) < 1e-5:
        return
    midpoint = (min(lon) + max(lon)) / 2
    width = (east - west) % 360
    west = midpoint + (west - midpoint + 180) % 360 - 180
    east = west + width
    if not 0 < width < 360 or west < min(lon) + 2 * dx or east > max(lon) - 2 * dx:
        raise ValueError("target longitude plus interpolation halo exceeds source footprint")
