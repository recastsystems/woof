"""Source-independent native storm merge of canonical pressure-coordinate states.

Acquisition and composition use the existing source tables. This module carries
only declarations, paths and receipts; weather arrays remain in Rust.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess


def add_arguments(parser):
    parser.add_argument("--storm-merge", metavar="SOURCE", help="inner-core donor source; opt-in balanced initialization")
    parser.add_argument("--storm-merge-input", action="append", type=Path, default=[], metavar="FILE")
    parser.add_argument("--storm-merge-supplement", action="append", default=[], metavar="ROLE=FILE")
    parser.add_argument("--storm-center", metavar="LAT,LON", help="vitals/tracker storm center in degrees")
    parser.add_argument("--storm-merge-radius-km", type=float, default=150.)
    parser.add_argument("--storm-merge-width-km", type=float, default=150.)
    parser.add_argument("--storm-merge-grid", metavar="SOUTH,WEST,DEGREES,NX,NY",
                        help="initialization lattice; background is resampled before insertion")
    parser.add_argument("--storm-merge-into-member", action="store_true",
                        help="the background is one ensemble member and the donor a deterministic core: each "
                             "member takes the same inner core and keeps its own environment")
    parser.add_argument("--storm-merge-perturb-member", metavar="K/N",
                        help="TC-VP1: member K of an N-member ensemble takes its own balanced perturbation of the "
                             "merged core (intensity, RMW, position) drawn from the packaged table")
    parser.add_argument("--storm-merge-perturb-vitals", type=Path, metavar="VITALS.json",
                        help="tc-vitals/1 document (rw_tc_vitals) whose record at the initial time centres the draws")
    parser.add_argument("--storm-merge-perturb-row", default="default", metavar="ROW",
                        help="row of woof/data/tc/vortex_perturbation.json (default: default)")
    parser.add_argument("--storm-merge-perturb-table", type=Path, metavar="TABLE.json",
                        help="a vortex perturbation table other than the packaged one")


def declaration(args):
    source = getattr(args, "storm_merge", None)
    if not source:
        if getattr(args, "storm_merge_input", None) or getattr(args, "storm_center", None):
            raise ValueError("storm donor inputs and center require --storm-merge SOURCE")
        return None
    from woof.source_adapters import get_source_adapter
    adapter = get_source_adapter(source)
    if not adapter.packaged_profile:
        raise ValueError("storm donor requires a packaged composition profile")
    try:
        center = [float(v) for v in args.storm_center.split(",")]
    except (ValueError, AttributeError):
        raise ValueError("--storm-center requires LAT,LON from vitals or a tracker") from None
    if len(center) != 2 or not all(math.isfinite(v) for v in center):
        raise ValueError("storm center needs two finite coordinates")
    if not args.storm_merge_input:
        raise ValueError("--storm-merge requires --storm-merge-input FILE (the ordinary fetch source row supplies files)")
    supplements = {}
    for entry in args.storm_merge_supplement:
        role, sep, value = entry.partition("=")
        if not sep or not role or not value:
            raise ValueError("storm supplement requires ROLE=FILE")
        supplements.setdefault(role, []).append(str(Path(value).resolve()))
    result = dict(source=adapter.source_id,
                input_files=[str(p.resolve()) for p in args.storm_merge_input],
                supplements=supplements, center_deg=center,
                radius_km=args.storm_merge_radius_km, blend_width_km=args.storm_merge_width_km)
    if getattr(args, "storm_merge_into_member", False):
        result["deterministic_donor_into_member"] = True
    perturbation = perturbation_declaration(args)
    if perturbation is not None:
        result["core_perturbation"] = perturbation
    if getattr(args, "storm_merge_grid", None):
        try:
            south, west, spacing, nx, ny = args.storm_merge_grid.split(",")
            result["target_grid"] = dict(south=float(south), west=float(west),
                spacing_deg=float(spacing), nx=int(nx), ny=int(ny))
        except ValueError:
            raise ValueError("storm merge grid requires SOUTH,WEST,DEGREES,NX,NY") from None
    return result


def command_arguments(args):
    doc = declaration(args)
    if doc is None:
        return []
    result = ["--storm-merge", doc["source"], "--storm-center", ",".join(map(str, doc["center_deg"])),
              "--storm-merge-radius-km", str(doc["radius_km"]), "--storm-merge-width-km", str(doc["blend_width_km"])]
    for path in doc["input_files"]:
        result += ["--storm-merge-input", path]
    if "target_grid" in doc:
        grid = doc["target_grid"]
        result += ["--storm-merge-grid", ",".join(str(grid[k]) for k in ("south", "west", "spacing_deg", "nx", "ny"))]
    for role, paths in doc["supplements"].items():
        for path in paths:
            result += ["--storm-merge-supplement", f"{role}={path}"]
    if doc.get("deterministic_donor_into_member"):
        result += ["--storm-merge-into-member"]
    if doc.get("core_perturbation"):
        p = doc["core_perturbation"]
        result += ["--storm-merge-perturb-member", f"{p['member']}/{p['members']}",
                   "--storm-merge-perturb-vitals", p["vitals"], "--storm-merge-perturb-row", p["row"],
                   "--storm-merge-perturb-table", p["table"]]
    return result


# ------------------------------------------------------------------------------------------- TC-VP1 core perturbation
PERTURBATION_TABLE = Path(__file__).resolve().parent / "data/tc/vortex_perturbation.json"
PERTURBATION_SCHEMA = "gpuwm-tc-vortex-perturbation-table-v1"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def perturbation_row(table, row):
    doc = json.loads(Path(table).read_text())
    if doc.get("schema") != PERTURBATION_SCHEMA:
        raise ValueError(f"{table}: not a {PERTURBATION_SCHEMA} table")
    rows = doc.get("rows") or {}
    if row not in rows:
        raise ValueError(f"{table}: no vortex perturbation row {row!r} (rows: {sorted(rows)})")
    value = rows[row]
    need = {"central_pressure_offset_hpa", "rmw_km", "shift_max_km", "outer_radius_km", "boundary_layer_top_hpa",
            "intensity_taper_hpa", "intensity_scale_bounds", "seed"}
    if set(value) - need - {"note"} or need - set(value):
        raise ValueError(f"{table}: row {row!r} must hold exactly {sorted(need)} (and an optional note)")
    lo, hi = value["central_pressure_offset_hpa"]
    rlo, rhi = value["rmw_km"]
    if not (lo < hi and 0 < rlo < rhi and value["shift_max_km"] >= 0 and value["outer_radius_km"] > 0):
        raise ValueError(f"{table}: row {row!r} has an empty or inverted range")
    return value


def perturbation_declaration(args):
    member = getattr(args, "storm_merge_perturb_member", None)
    vitals = getattr(args, "storm_merge_perturb_vitals", None)
    if member is None:
        if vitals is not None or getattr(args, "storm_merge_perturb_table", None) is not None:
            raise ValueError("--storm-merge-perturb-vitals/-table need --storm-merge-perturb-member K/N")
        return None
    try:
        k, count = (int(x) for x in member.split("/"))
    except ValueError:
        raise ValueError("--storm-merge-perturb-member takes K/N, e.g. 3/36") from None
    if not 1 <= k <= count or count < 2:
        raise ValueError("--storm-merge-perturb-member needs 1 <= K <= N and N >= 2")
    if vitals is None:
        raise ValueError("a core perturbation is drawn around the TC vitals: give --storm-merge-perturb-vitals")
    table = Path(getattr(args, "storm_merge_perturb_table", None) or PERTURBATION_TABLE).resolve(strict=True)
    row = getattr(args, "storm_merge_perturb_row", None) or "default"
    perturbation_row(table, row)
    vitals = Path(vitals).resolve(strict=True)
    return dict(member=k, members=count, row=row, table=str(table), table_sha256=_sha(table),
                vitals=str(vitals), vitals_sha256=_sha(vitals))


def _rank(seed, axis, count, k):
    """Member k's stratum on one axis: a fixed permutation of 0..count-1 (sha256 order, platform independent)."""
    order = sorted(range(count), key=lambda i: hashlib.sha256(f"{seed}:{axis}:{count}:{i}".encode()).hexdigest())
    return order.index(k - 1)


def perturbation_targets(doc, valid_time):
    """Latin-hypercube draws for member K of N: central pressure uniform on vitals + offsets, RMW log-uniform,
    centre shift uniform in area on a disk (golden-angle azimuths). Deterministic in (table row, K, N, vitals)."""
    from woof.tc.mapped_vortex import vitals_record
    if _sha(doc["table"]) != doc["table_sha256"] or _sha(doc["vitals"]) != doc["vitals_sha256"]:
        raise ValueError("vortex perturbation table or vitals changed after the run was declared")
    row = perturbation_row(doc["table"], doc["row"])
    record = vitals_record(doc["vitals"], valid_time)
    k, n, seed = doc["member"], doc["members"], row["seed"]
    u = {axis: (_rank(seed, axis, n, k) + 0.5) / n for axis in ("pressure", "rmw", "shift")}
    lo, hi = row["central_pressure_offset_hpa"]
    rlo, rhi = row["rmw_km"]
    pressure = float(record["minimum_pressure_pa"]) + 100.0 * (lo + u["pressure"] * (hi - lo))
    rmw = 1000.0 * math.exp(math.log(rlo) + u["rmw"] * (math.log(rhi) - math.log(rlo)))
    radius = 1000.0 * row["shift_max_km"] * math.sqrt(u["shift"])
    angle = k * math.pi * (3.0 - math.sqrt(5.0))
    return dict(target_central_pressure_pa=pressure, target_rmw_m=rmw,
                shift_east_m=radius * math.cos(angle), shift_north_m=radius * math.sin(angle),
                outer_radius_m=1000.0 * row["outer_radius_km"],
                boundary_layer_top_pa=100.0 * row["boundary_layer_top_hpa"],
                intensity_taper_pa=[100.0 * x for x in row["intensity_taper_hpa"]],
                intensity_scale_bounds=list(row["intensity_scale_bounds"]),
                label=f"member {k}/{n} row {doc['row']} vitals {record['valid_time']} "
                      f"{float(record['minimum_pressure_pa']) / 100:.0f} hPa")


def compose_donor(doc, work_dir, workers=None):
    from woof.mapped_authoring import author_input_manifest
    from woof.mapped_engine_bridge import run_engine
    from woof.source_adapters import get_source_adapter
    from woof.source_authorities import (packaged_authorities, packaged_contributing_mappings,
                                         packaged_provenance_files)
    profile = get_source_adapter(doc["source"]).packaged_profile
    authorities = packaged_authorities(profile)
    contributing = packaged_contributing_mappings(profile)
    provenance = packaged_provenance_files(profile)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=False)
    primary = [Path(v) for v in doc["input_files"]]
    supplements = {k: [Path(v) for v in values] for k, values in doc["supplements"].items()}
    manifest = author_input_manifest(work_dir / "inputs.json", mapping_path=authorities["mapping"],
        composition_path=authorities["composition"], primary_files=primary,
        supplement_files=supplements, provenance_files=provenance, contributing_mappings=contributing)
    execution = run_engine("compose", mapping=authorities["mapping"], files=primary,
        composition=authorities["composition"], supplements=supplements, provenance=provenance,
        contributing_mappings=contributing, input_manifest=manifest["manifest"]["path"],
        input_manifest_sha256=manifest["manifest"]["sha256"], output=work_dir / "composed",
        threads=workers, lead_batch=True)
    evidence = {"donor-input-manifest.json": (work_dir / "inputs.json").read_bytes()}
    for role, path in {**authorities, **contributing, **provenance}.items():
        evidence[f"donor-{role}.json"] = Path(path).read_bytes()
    return work_dir / "composed", execution, evidence


def run_merge(background, donor, output, *, valid_time, declaration, workers=None):
    from woof import bridges
    binary = bridges.find_artifact("GPUWM_STORM_MERGE_BIN", bridges.executable_name("rw_storm_merge"),
                                   crate_relative="tools/storm_merge")
    if binary is None:
        raise FileNotFoundError("build tools/storm_merge and set GPUWM_STORM_MERGE_BIN to rw_storm_merge")
    output = Path(output)
    request = dict(schema="gpuwm-storm-merge-v1", background=str(Path(background).resolve()),
        donor=str(Path(donor).resolve()), output=str(output.resolve()), valid_time=valid_time,
        center_latitude_deg=declaration["center_deg"][0], center_longitude_deg=declaration["center_deg"][1],
        radius_km=declaration["radius_km"], blend_width_km=declaration["blend_width_km"])
    if "target_grid" in declaration:
        request["target_grid"] = declaration["target_grid"]
    if declaration.get("deterministic_donor_into_member"):
        request["deterministic_donor_into_member"] = True
    if declaration.get("core_perturbation"):
        request["core_perturbation"] = perturbation_targets(declaration["core_perturbation"], valid_time)
    request_path = output.with_name(output.name + "-request.json")
    request_path.write_text(json.dumps(request, indent=2) + "\n")
    from woof.ingest.preparation_workers import worker_environment
    completed = subprocess.run([str(binary), "--request", str(request_path)],
        env=worker_environment(workers), capture_output=True, text=True)
    if completed.returncode:
        raise ValueError(completed.stderr.strip() or "native storm merge failed")
    receipt = json.loads(completed.stdout)
    receipt["engine_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
    return receipt, request_path.read_bytes()


class _MergedBundle:
    """A declared initialization transform, separate from composition evidence.

    Later forcing frames retain their original grid and terrain authority. The
    first frame may use a finer initialization lattice; this is not a new
    composition claiming that its terrain arrays are identical across times.
    """
    def __init__(self, original, frames, work):
        self.original, self.frames, self.work = original, frames, work

    def __getattr__(self, name):
        return getattr(self.original, name)

    def regular_snapshots(self):
        from woof.mapped_composition import _RegularSnapshots
        return _RegularSnapshots(self)

    def close(self):
        self.frames.close()
        self.original.close()
        self.work.cleanup()


def initialization_lattice(background, donor, *, valid_time, native_grids, centre, radius_km=0.):
    """Resolution/extent metadata for a donor core on the receiving domains.

    Inserting a kilometre-scale donor on a global background's coarse lattice
    destroys its eye before native interpolation. Use the coarser of the donor
    and receiving resolutions, covering the receiving footprint plus its halo.
    The native Rust merge owns resampling; later boundary frames stay unchanged.
    """
    def descriptor(root):
        frames = json.loads((Path(root)/'frames.json').read_bytes())['frames']
        return next(f for f in frames if f['valid_time'] == valid_time)
    def steps(frame):
        return [abs(frame[name]['values'][1]-frame[name]['values'][0])
                for name in ('latitude','longitude')]
    background_step = min(steps(descriptor(background)))
    donor_step = max(steps(descriptor(donor)))
    model_step = min(min(g.dx,g.dy) for g in native_grids)/111194.92664455874
    spacing = max(donor_step, model_step)
    if spacing >= background_step*(1.-1.e-8):
        return None
    points = [g.ij_to_latlon(x,y) for g in native_grids
              for x in (-2.,g.e_we/2.,g.e_we+2.)
              for y in (-2.,g.e_sn/2.,g.e_sn+2.)]
    latitude = [float(p[0]) for p in points]
    longitude = [centre[1]+math.remainder(float(p[1])-centre[1],360.) for p in points]
    if radius_km:
        delta = radius_km/6371.
        lon_delta = math.degrees(math.asin(min(1.,math.sin(delta)/math.cos(math.radians(centre[0])))))
        latitude += [centre[0]-math.degrees(delta)-2*spacing,centre[0]+math.degrees(delta)+2*spacing]
        longitude += [centre[1]-lon_delta-2*spacing,centre[1]+lon_delta+2*spacing]
    south = centre[0]+math.floor((min(latitude)-centre[0])/spacing)*spacing
    west = centre[1]+math.floor((min(longitude)-centre[1])/spacing)*spacing
    return dict(south=south,west=west,spacing_deg=spacing,
                nx=math.ceil((max(longitude)-west)/spacing)+1,
                ny=math.ceil((max(latitude)-south)/spacing)+1)


def apply_to_bundle(bundle, doc, *, valid_time, scratch, workers=None, native_grids=None):
    """Insert once, before every root/child initialization interpolates."""
    import tempfile
    from woof.mapped_engine_bridge import FrameSet
    if not isinstance(bundle.frames, FrameSet):
        raise ValueError("storm merge requires the native canonical frame writer")
    work = tempfile.TemporaryDirectory(prefix="gpuwm-storm-merge-", dir=scratch)
    try:
        root = Path(work.name)
        donor, execution, evidence = compose_donor(doc, root / "donor", workers)
        applied = dict(doc)
        if 'target_grid' not in applied and native_grids:
            radius = doc['radius_km']+doc['blend_width_km']
            if doc.get('core_perturbation'):
                row = perturbation_row(doc['core_perturbation']['table'],doc['core_perturbation']['row'])
                radius = max(radius,row['outer_radius_km']+row['shift_max_km'])
            lattice = initialization_lattice(bundle.frames.directory,donor,valid_time=valid_time,
                native_grids=native_grids,centre=doc['center_deg'],radius_km=radius)
            if lattice is not None:
                applied['target_grid'] = lattice
                print(f"storm-merge: preserve donor core on {lattice['nx']} x {lattice['ny']} "
                      f"initialization points at {lattice['spacing_deg']:.6g} degrees; "
                      "resolution from donor and receiving domains",flush=True)
        receipt, request = run_merge(bundle.frames.directory, donor, root / "merged",
            valid_time=valid_time, declaration=applied, workers=workers)
        frames = FrameSet(root / "merged")
        if frames.valid_times != bundle.frames.valid_times:
            raise ValueError("storm merge changed the forcing clock")
        evidence.update({"request.json": request,
            "receipt.json": (json.dumps(receipt, sort_keys=True) + "\n").encode(),
            "declaration.json": (json.dumps(doc, sort_keys=True) + "\n").encode(),
            "background-frames.json": (bundle.frames.directory / "frames.json").read_bytes(),
            "donor-frames.json": (donor / "frames.json").read_bytes(),
            "merged-frames.json": (root / "merged/frames.json").read_bytes()})
        if doc.get("core_perturbation"):
            evidence["perturbation-table.json"] = Path(doc["core_perturbation"]["table"]).read_bytes()
            evidence["perturbation-vitals.json"] = Path(doc["core_perturbation"]["vitals"]).read_bytes()
            cp = receipt.get("core_perturbation") or {}
            print(f"storm-merge: TC-VP1 {cp.get('label')}: merged core {cp.get('merged_minimum_reduced_pressure_pa', 0) / 100:.1f} hPa "
                  f"RMW {cp.get('merged_rmw_m', 0) / 1000:.0f} km -> target {cp.get('target_central_pressure_pa', 0) / 100:.1f} hPa "
                  f"RMW {cp.get('target_rmw_m', 0) / 1000:.0f} km shift {[round(x / 1000, 1) for x in cp.get('shift_m', [0, 0])]} km: "
                  f"intensity scale {cp.get('intensity_scale', float('nan')):.3f}"
                  f"{' (CLIPPED)' if cp.get('intensity_scale_clipped') else ''}, predicted "
                  f"{cp.get('predicted_central_pressure_pa', 0) / 100:.1f} hPa", flush=True)
        binding = {"schema": "gpuwm-prepared-storm-merge-v1", "valid_time": valid_time,
            "source": doc["source"], "engine_sha256": receipt["engine_sha256"],
            "evidence_sha256": {name: hashlib.sha256(data).hexdigest()
                                for name, data in evidence.items()}}
        return _MergedBundle(bundle, frames, work), binding, evidence
    except BaseException:
        work.cleanup()
        raise


def write_evidence(evidence_root, evidence):
    root = Path(evidence_root) / "storm-merge"
    root.mkdir()
    for name, data in evidence.items():
        (root / name).write_bytes(data)


def validate_evidence(prepared_root, binding, *, valid_time):
    if (not isinstance(binding, dict)
            or set(binding) != {"schema", "valid_time", "source", "engine_sha256", "evidence_sha256"}
            or binding["schema"] != "gpuwm-prepared-storm-merge-v1"
            or binding["valid_time"] != valid_time):
        raise ValueError("invalid prepared storm merge binding or clock")
    root = Path(prepared_root) / "source-evidence/storm-merge"
    payloads = {}
    for name, digest in binding["evidence_sha256"].items():
        if not name or Path(name).name != name or any(c in name for c in ("\\", ":", "\x00")):
            raise ValueError("invalid storm merge evidence path")
        data = (root / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"storm merge evidence changed: {name}")
        payloads[name] = data
    request = json.loads(payloads["request.json"])
    receipt = json.loads(payloads["receipt.json"])
    if (request["valid_time"] != valid_time
            or receipt["engine_sha256"] != binding["engine_sha256"]
            or receipt["request_sha256"] != binding["evidence_sha256"]["request.json"]
            or receipt["background_frames_sha256"] != binding["evidence_sha256"]["background-frames.json"]
            or receipt["donor_frames_sha256"] != binding["evidence_sha256"]["donor-frames.json"]
            or json.loads(payloads["declaration.json"])["source"] != binding["source"]):
        raise ValueError("storm merge evidence chain differs")


def normalize_arguments(tokens, base):
    import argparse
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False, exit_on_error=False)
    add_arguments(parser)
    try:
        args, extra = parser.parse_known_args(tokens)
    except argparse.ArgumentError as exc:
        raise ValueError(str(exc)) from exc
    if extra or not args.storm_merge:
        raise ValueError("run option accepts only a complete storm merge declaration")
    args.storm_merge_input = [p if p.is_absolute() else Path(base) / p
                             for p in args.storm_merge_input]
    bindings = []
    for entry in args.storm_merge_supplement:
        role, sep, value = entry.partition("=")
        if not sep:
            raise ValueError("storm supplement requires ROLE=FILE")
        path = Path(value)
        bindings.append(f"{role}={path if path.is_absolute() else Path(base) / path}")
    args.storm_merge_supplement = bindings
    return command_arguments(args)


def preparation_source(source):
    """Select the registry's complete-state representation for a transform.

    Legacy source preparation remains unchanged unless a transform is requested.
    The selected row is an ordinary mapped source with its own sealed inventory.
    """
    from woof.source_adapters import get_source_adapter
    adapter = get_source_adapter(source)
    selected = adapter.canonical_preparation_source or adapter.source_id
    result = get_source_adapter(selected)
    if not result.packaged_profile or result.runner != "mapped_composition_v1":
        raise ValueError(f"{source} has no declared canonical preparation representation")
    return result.source_id
