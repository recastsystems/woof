"""Stage a moving-region case and prepare its mesh-family members (CPU only).

The stages, each skipped when its receipt closed (so two controllers asking
for the same member wait for one preparation and then reuse it):

* :func:`stage_case` -- the case's initial-condition source through the
  table-driven init-source path (:mod:`woof.hex.init_sources`: acquire, then
  decode into one WPS intermediate), and optionally the storm's IBTrACS best
  track for verification;
* :func:`prepare_case` -- the start member; for a corridor case first the
  FRAME (fixed rings alone) and the TEMPLATE (the generator's mesh of the
  full request about the initial centre);
* :func:`prepare_member` -- one member: generator spec, grid (``rw_mpas_mesh``,
  or a stitched / frozen-far-field member through ``--from-centres``),
  static (``rw_mpas_static``), init (the init door, with the source row's
  first-guess declarations) and its mesh row;
* :func:`build_operator` / :func:`build_zone` -- the conservative overlap (or
  zone carry) from one member to the next.

Every Rust tool resolves through :mod:`woof.hex.engines` (explicit variable,
the installed woof's bundled bridge, then PATH), so a pip install runs it.
Moved here from the source tree's ``tools/moving48`` shell scripts, which
were bound to one rented host's queue and work folders.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable

from .moving_case import (
    MovingCaseError,
    best_track_from_ibtracs,
    cycle_of,
    frame_spec,
    load_case,
    locked,
    register_member,
    resolve_vertical_spec,
    spec as member_spec,
    stitch,
)

STAGE_FILE = "stage.json"
IBTRACS_URL = ("https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/"
               "v04r01/access/csv/ibtracs.{subset}.list.v04r01.csv")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _stamp(folder: Path, name: str, text: str) -> None:
    """One prep stage's line, in logs/prep-stages.txt and on stdout.

    On stdout too, so the prepare log a WOOF box uploads carries the stage profile (mesh, static, init, each
    with its wall time and parallelism). THE BREAKAGE THIS PREVENTS: a new box spent 940 s in "Preparing the
    start member" (production run_312c8803de02ae46, 2026-10-25) and the uploaded log said nothing about which
    stage took it; prep-stages.txt stayed on the box."""
    line = f"{_now()} {name} {text}"
    (folder / "logs").mkdir(parents=True, exist_ok=True)
    with (folder / "logs" / "prep-stages.txt").open("a") as stream:
        stream.write(line + "\n")
    print(f"prepare-stage: {line}", flush=True)


def _cores(env: dict[str, str] | None) -> str:
    """The parallelism a Rust prep tool was given, for its stage line."""
    env = os.environ if env is None else env
    return f"CPU, {env.get('RAYON_NUM_THREADS', 'all')} threads"


def _run_logged(argv: list[str], log: Path, *, env: dict[str, str] | None = None) -> float:
    """Run one stage with its output in ``log``; refuse by name on a non-zero exit."""

    log.parent.mkdir(parents=True, exist_ok=True)
    from .stage_timing import stage

    stages = []
    execution_env = os.environ if env is None else env
    try:
        with stage(stages, log.stem, parallelism={
            "device": "CPU",
            "requested_rayon_threads": execution_env.get("RAYON_NUM_THREADS", "runtime default"),
            "requested_omp_threads": execution_env.get("OMP_NUM_THREADS", "runtime default"),
            "requested_tile_workers": execution_env.get("RW_MPAS_TILE_WORKERS", "runtime default"),
            "visible_devices": execution_env.get("CUDA_VISIBLE_DEVICES", "inherited"),
        }) as timing:
            with log.open("w") as stream:
                rc = subprocess.call([str(a) for a in argv], stdout=stream, stderr=subprocess.STDOUT, env=env)
            timing["returncode"] = rc
    finally:
        if stages and stages[0].get("returncode", 0) != 0:
            stages[0]["status"] = "failed"
        log.with_suffix(".timing.json").write_text(json.dumps(dict(
            schema="gpuwm-hex-stage-timing-v1", argv=[str(a) for a in argv], stages=stages), indent=2) + "\n")
    if rc != 0:
        tail = log.read_text(errors="replace").strip().splitlines()[-5:]
        raise MovingCaseError(f"{Path(str(argv[0])).name} exited {rc} (log {log}): " + " | ".join(tail))
    return stages[0]["wall_seconds"]


def _tool(spec_name: str) -> Path:
    from . import engines

    try:
        return engines.resolve(getattr(engines, spec_name))
    except engines.EngineRefusal as error:
        raise MovingCaseError(str(error)) from None


# ---------------------------------------------------------------------------
# stage
# ---------------------------------------------------------------------------
def read_stage(casedir: Path) -> dict[str, Any]:
    path = Path(casedir) / STAGE_FILE
    if not path.is_file():
        raise MovingCaseError(
            f"{path} does not exist: stage the case first (woof hex moving stage --case ... --casedir {casedir})")
    return json.loads(path.read_text())


def stage_case(case_path: Path, casedir: Path, *, source: str | None = None, staged: list[Path] = (),
               best_track: bool = False, ibtracs_subset: str = "last3years", ibtracs_file: Path | None = None,
               jobs: int = 3, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Initial-condition source -> one WPS intermediate; optionally the best track."""

    from . import init_sources

    case = load_case(case_path)
    casedir = Path(casedir).expanduser().absolute()
    casedir.mkdir(parents=True, exist_ok=True)
    table = init_sources.load_table()
    source_id = source or case.get("init_source") or "gfs"
    row = init_sources.resolve_source(table, source_id)
    cycle = init_sources.Cycle.parse(cycle_of(case))
    # A case started from the source's forecast at a lead (moving case --lead-hours / --from-last-hour): the
    # products are the source cycle's at that lead, the init clock is the case start (cycle + lead).
    lead = int(case.get("init_lead_hours") or 0)
    # An analysis-only cold-soil donor has no soil for a forecast hour: a case started from a later hour stages
    # without the donor and initializes from the source's own interpolated soil (--soil-remap off at init, from
    # init_sources.cold_soil_init_flags), as every case did before the remap, instead of refusing. THE BREAKAGE THIS
    # PREVENTS (WOOF woof hex 0.4.10 candidate canary, 2026-10-31, setup-following run_5f2a4e9019516a11): every GFS
    # moving case from a later hour stopped here.
    soil = row.get("cold_soil")
    if lead and (soil or {}).get("analysis_only"):
        log(f"stage: cold soil remap off at f{lead:03d}: the {row['id']} donor is analysis-only")
        soil = None
    from datetime import timedelta
    source_cycle = init_sources.Cycle(cycle.when - timedelta(hours=lead))
    source_member = case.get("init_member")
    if source_member is not None and (type(source_member) is not int or source_member < 0):
        raise MovingCaseError("init_member must be a nonnegative integer source-member number")
    # Validate against the source table before either fetching or reusing a stage.
    plans = init_sources.plan_products(row, source_cycle, source_member, lead)
    if row.get("cold_soil") and soil is None:
        plans = [p for p in plans if p.role != row["cold_soil"]["product_role"]]
    out_dir = casedir / "source"
    previous = casedir / STAGE_FILE
    if previous.is_file():
        stage = json.loads(previous.read_text())
        same_source = (stage.get("source") == row["id"]
                       and stage.get("cycle") == cycle.start_time
                       and int(stage.get("lead_hours") or 0) == lead
                       and stage.get("source_member") == source_member)
        if not same_source and (casedir / "family.json").is_file():
            raise MovingCaseError(
                f"{casedir}: prepared family belongs to another source, cycle or member; "
                "use a separate casedir so its initial conditions cannot be mislabeled")
        if same_source and Path(stage.get("met", "")).is_file():
            log(f"stage: {row['id']} {cycle.start_time} already staged ({stage['met']})")
            if best_track and not (casedir / "obs" / "best-track.json").is_file():
                stage["best_track"] = _best_track(case, casedir, ibtracs_subset, ibtracs_file, log)
                previous.write_text(json.dumps(stage, indent=2) + "\n")
            return stage
    writer = init_sources.resolve_writer(None)
    init_sources.resolve_vtable(row)
    init_sources.resolve_surface_input_vtable(row)
    began = time.monotonic()
    log(f"stage: acquiring {row['id']} for {cycle.start_time}")
    products, missing = init_sources.acquire_with_cold_soil(
        plans, soil, out_dir / "products", cycle, staged=[Path(p).expanduser().absolute() for p in staged], jobs=jobs)
    if missing:
        log(f"stage: cold soil remap off: {missing}")
    donors = init_sources.cold_soil_products(products, soil)
    companions = init_sources.surface_input_products(products, row)
    inputs = [product["file"] for product in products if product not in donors and product not in companions]
    remap = None
    if row.get("remap"):
        remap = init_sources.remap(row, products, cycle, out_dir, init_sources.resolve_remapper(None))
        inputs = remap.pop("outputs")
    log("stage: decoding to a WPS intermediate")
    decoded = init_sources.decode(row, inputs, cycle, out_dir, writer)
    surface_input = (init_sources.decode_surface_input(row, [product["file"] for product in companions],
                                                       source_cycle, out_dir, writer) if companions else None)
    stage = dict(schema="gpuwm-hex.moving-stage/v1", created_utc=_now(), case=case["_path"],
                 source=row["id"], source_member=source_member, cycle=cycle.start_time,
                 source_cycle=source_cycle.start_time, lead_hours=lead,
                 met=decoded["met"], products=products,
                 decode=decoded, surface_input=surface_input, remap=remap,
                 seconds=round(time.monotonic() - began, 1))
    if best_track:
        stage["best_track"] = _best_track(case, casedir, ibtracs_subset, ibtracs_file, log)
    tmp = previous.with_suffix(".json.partial")
    tmp.write_text(json.dumps(stage, indent=2) + "\n")
    tmp.replace(previous)
    return stage


def _best_track(case, casedir: Path, subset: str, local: Path | None, log) -> dict[str, Any]:
    sid = case["storm"].get("ibtracs_sid")
    if not sid:
        raise MovingCaseError("the case names no storm.ibtracs_sid, so there is no best track to fetch")
    obs = casedir / "obs"
    obs.mkdir(parents=True, exist_ok=True)
    if local is None:
        url = IBTRACS_URL.format(subset=subset)
        local = obs / Path(url).name
        if not local.is_file() or local.stat().st_size == 0:
            log(f"stage: fetching {url}")
            partial = local.with_name(local.name + ".part")
            request = urllib.request.Request(url, headers={"User-Agent": "woof hex"})
            with urllib.request.urlopen(request, timeout=300) as response, partial.open("wb") as out:
                while True:
                    block = response.read(1 << 22)
                    if not block:
                        break
                    out.write(block)
            partial.replace(local)
    track = best_track_from_ibtracs(Path(local), sid)
    out = obs / "best-track.json"
    out.write_text(json.dumps(track, indent=1) + "\n")
    log(f"stage: best track {sid} {track['name']}: {len(track['rows'])} rows")
    return dict(path=str(out), ibtracs=str(local), ibtracs_sha256=_sha256(Path(local)), sid=sid)


# ---------------------------------------------------------------------------
# members
# ---------------------------------------------------------------------------
def _grid_centres(path: Path):
    import numpy as np
    from netCDF4 import Dataset

    with Dataset(path) as grid:
        return np.stack([np.asarray(grid.variables[k][:], dtype=np.float64) for k in ("xCell", "yCell", "zCell")], -1)


def _min_dc_edge(path: Path) -> float:
    import numpy as np
    from netCDF4 import Dataset

    with Dataset(path) as grid:
        return float(np.min(np.asarray(grid.variables["dcEdge"][:])))


def _from_centres(request: Path, grid: Path, receipt: Path) -> None:
    tool = _tool("MESH")
    subprocess.run([str(tool), "--from-centres", str(request), "--out", str(grid), "--receipt", str(receipt),
                    "--clobber"], check=True)


def build_stitched_mesh(case: dict[str, Any], name: str, lat: float, lon: float, folder: Path) -> dict[str, Any]:
    """A member stitched from FOLDER/frame.grid.nc and FOLDER/template.grid.nc (corridor case)."""

    from .moving_zone import build_insert_member, spec_anchor, unit_from_deg, write_centres

    template_grid = folder / "template.grid.nc"
    template_spec = json.loads((folder / "template.spec.json").read_text())
    new_spec = member_spec(case, name, float(lat), float(lon))
    (folder / f"{name}.spec.json").write_text(json.dumps(new_spec, indent=2) + "\n")
    radii = stitch(case)
    started = time.time()
    frame, template = _grid_centres(folder / "frame.grid.nc"), _grid_centres(template_grid)
    rows = int(case["mesh"]["corridor"].get("seam_rows", 6))
    attempts = []
    for extra in (0, 1, 2, 3):
        # A stubborn seam defect belongs to where the seam lies: a wider annulus moves it.
        try:
            result = build_insert_member(
                frame, template, spec_anchor(template_spec), unit_from_deg(float(lat), float(lon)), new_spec,
                inner_radius_km=radii["inner_radius_km"],
                outer_radius_km=radii["outer_radius_km"] + extra * radii["fixed_spacing_km"],
                fixed_spacing_km=radii["fixed_spacing_km"],
                progress=lambda row: print(json.dumps(row), flush=True))
            break
        except ValueError as error:
            if "outside the fixed corridor" in str(error):
                raise
            attempts.append(dict(seam_rows=rows + extra, refusal=str(error)))
            print(json.dumps(dict(retry=attempts[-1])), flush=True)
    else:
        raise MovingCaseError(f"no seam width produced an admissible member: {attempts}")
    result.receipt["refused_attempts"] = attempts
    request = folder / f"{name}.centres.nc"
    write_centres(request, result, template_grid, spec=new_spec)
    grid = folder / f"{name}.grid.nc"
    generator_receipt = folder / f"{name}-from-centres.json"
    began = time.time()
    _from_centres(request, grid, generator_receipt)
    receipt = dict(emitted=dict(path=str(grid), sha256=_sha256(grid), n_cells=len(result.centres)),
                   receipt=json.loads(generator_receipt.read_text()).get("receipt", {}),
                   moving_zone=dict(result.receipt, frame_grid=str(folder / "frame.grid.nc"),
                                    template_grid=str(template_grid), min_dc_edge=_min_dc_edge(grid),
                                    template_min_dc_edge=_min_dc_edge(template_grid),
                                    from_centres_seconds=time.time() - began,
                                    total_seconds=time.time() - started))
    (folder / f"{name}-mesh.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def build_zone_mesh(case: dict[str, Any], prev: str, name: str, lat: float, lon: float,
                    folder: Path) -> dict[str, Any]:
    """The next frozen-far-field member's grid, from the previous member's generators."""

    import numpy as np
    from netCDF4 import Dataset

    from .moving_zone import build_member_centres, write_centres

    prev_grid = folder / f"{prev}.grid.nc"
    prev_spec = json.loads((folder / f"{prev}.spec.json").read_text())
    new_spec = member_spec(case, name, float(lat), float(lon))
    (folder / f"{name}.spec.json").write_text(json.dumps(new_spec, indent=2) + "\n")
    zone = case.get("moving_zone", {})
    with Dataset(prev_grid) as grid:
        centres = np.stack([np.asarray(grid.variables[k][:], dtype=np.float64) for k in ("xCell", "yCell", "zCell")], -1)
        density = np.asarray(grid.variables["meshDensity"][:], dtype=np.float64)
        cells_on_cell = np.asarray(grid.variables["cellsOnCell"][:], dtype=np.int64)
        counts = np.asarray(grid.variables["nEdgesOnCell"][:], dtype=np.int64)
    started = time.time()
    halo = int(zone.get("halo_rings", 2))
    attempts = []
    for rings in (halo, halo + 1, halo + 2):
        # A stubborn seam defect is a property of where the band ends: a wider band moves the seam.
        try:
            result = build_member_centres(
                centres, density, cells_on_cell, counts, prev_spec, new_spec,
                spacing_tolerance=float(zone.get("spacing_tolerance", 5e-3)),
                halo_rings=rings, fixed_rings=int(zone.get("fixed_rings", 3)),
                progress=lambda row: print(json.dumps(row), flush=True))
            break
        except ValueError as error:
            attempts.append(dict(halo_rings=rings, refusal=str(error)))
            print(json.dumps(dict(retry=attempts[-1])), flush=True)
    else:
        raise MovingCaseError(f"no band width produced an admissible member: {attempts}")
    result.receipt["refused_attempts"] = attempts
    request = folder / f"{name}.centres.nc"
    write_centres(request, result, prev_grid, spec=new_spec)
    grid = folder / f"{name}.grid.nc"
    generator_receipt = folder / f"{name}-from-centres.json"
    began = time.time()
    _from_centres(request, grid, generator_receipt)
    receipt = dict(emitted=dict(path=str(grid), sha256=_sha256(grid), n_cells=len(result.centres)),
                   receipt=json.loads(generator_receipt.read_text()).get("receipt", {}),
                   moving_zone=dict(result.receipt, previous_member=prev, previous_grid=str(prev_grid),
                                    from_centres_seconds=time.time() - began,
                                    total_seconds=time.time() - started))
    (folder / f"{name}-mesh.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def _tool_env(threads: int | None, tile_workers: int | None) -> dict[str, str]:
    env = dict(os.environ)
    if threads:
        env["RAYON_NUM_THREADS"] = str(threads)
        env["OMP_NUM_THREADS"] = str(threads)
    if tile_workers:
        env["RW_MPAS_TILE_WORKERS"] = str(tile_workers)
    return env


def generate_mesh(spec_path: Path, grid: Path, receipt: Path, log: Path, env: dict[str, str]) -> float:
    return _run_logged([_tool("MESH"), "--spec", spec_path, "--out", grid, "--receipt", receipt], log, env=env)


def prepare_member(case_path: Path, folder: Path, name: str, lat: float, lon: float, *,
                   casedir: Path | None = None, prev: str | None = None, stitched: bool = False,
                   rotate_from: str | None = None,
                   geog: Path | None = None, threads: int | None = None, tile_workers: int | None = None,
                   log: Callable[[str], None] = print) -> dict[str, Any]:
    """spec -> grid -> static -> init -> mesh row, reusing closed stages with their artifacts.

    ``folder`` holds the members; ``casedir`` (default: ``folder``) holds the
    staged source (:func:`stage_case`).
    """

    from . import init_sources

    case = load_case(case_path)
    if rotate_from is not None and (prev or stitched or case["mesh"].get("corridor")):
        raise MovingCaseError("template rotation requires a whole global cap family, without corridor or frozen far field")
    if rotate_from == name:
        raise MovingCaseError("template rotation needs a different source member")
    folder = Path(folder).expanduser().absolute()
    folder.mkdir(parents=True, exist_ok=True)
    stage = read_stage(Path(casedir or folder).expanduser().absolute())
    vertical = resolve_vertical_spec(case)
    env = _tool_env(threads, tile_workers)
    logs = folder / "logs"
    with locked(folder / f"{name}.prep.lock"):
        spec_path = folder / f"{name}.spec.json"
        if not spec_path.is_file():
            spec_path.write_text(json.dumps(member_spec(case, name, float(lat), float(lon)), indent=2) + "\n")
        mesh_receipt = folder / f"{name}-mesh.json"
        grid = folder / f"{name}.grid.nc"
        def mesh_ready():
            return mesh_receipt.is_file() and grid.is_file()
        mesh_rebuilt = not mesh_ready()
        if not mesh_ready() and rotate_from is not None:
            from .moving_mesh import rotate_grid
            from .stage_timing import stage as timed_stage

            source_spec = json.loads((folder / f"{rotate_from}.spec.json").read_text())
            source_center = source_spec["regions"][-1]["shape"]["center_deg"]
            if source_spec != member_spec(case, rotate_from, *source_center):
                raise MovingCaseError("rotation template does not match this case's resolution and concentric cap family")
            source_grid = folder / f"{rotate_from}.grid.nc"
            source_receipt = json.loads((folder / f"{rotate_from}-mesh.json").read_text())
            source_sha256 = _sha256(source_grid)
            if source_sha256 != source_receipt["emitted"]["sha256"]:
                raise MovingCaseError("rotation template bytes differ from its mesh receipt")
            grid = folder / f"{name}.grid.nc"
            timings = []
            with timed_stage(timings, "rotate-mesh", parallelism={"device": "CPU", "threads": 1}):
                rotation = rotate_grid(source_grid, grid,
                                       start=source_center, end=(float(lat), float(lon)))
            rotation["source_sha256"] = source_sha256
            receipt = dict(emitted=dict(path=str(grid), sha256=rotation["sha256"],
                                       n_cells=source_receipt["emitted"]["n_cells"]),
                           rotation=rotation, stages=timings)
            mesh_receipt.write_text(json.dumps(receipt, indent=2) + "\n")
            _stamp(folder, name, f"rotate-mesh from {rotate_from} {rotation['wall_seconds']:.1f}s "
                                 "(CPU, 1 thread, vectorised array rotation)")
        if not mesh_ready() and stitched:
            # Stitched member: the corridor frame's far field + the template's fine region moved
            # rigidly.  A refused stitch never falls through to a whole-globe generator mesh.
            began = time.time()
            build_stitched_mesh(case, name, lat, lon, folder)
            _stamp(folder, name, f"stitch-mesh {time.time() - began:.0f}s")
        if not mesh_ready() and prev:
            began = time.time()
            build_zone_mesh(case, prev, name, lat, lon, folder)
            _stamp(folder, name, f"zone-mesh {time.time() - began:.0f}s")
        if not mesh_ready():
            seconds = generate_mesh(spec_path, folder / f"{name}.grid.nc", mesh_receipt, logs / f"{name}-mesh.log", env)
            _stamp(folder, name, f"mesh {seconds:.0f}s ({_cores(env)})")
        static_receipt = folder / f"{name}-static.json"
        static_rebuilt = mesh_rebuilt or not static_receipt.is_file() or not (folder / f"{name}.static.nc").is_file()
        if static_rebuilt:
            argv = [_tool("STATIC"), "--grid", folder / f"{name}.grid.nc", "--out", folder / f"{name}.static.nc",
                    "--receipt", static_receipt]
            from .engines import static_geog_argv

            argv += static_geog_argv(geog)
            # Sub-kilometre members: a cell narrower than one 30-arc-second pixel owns no sample and its terrain is
            # 0/0 (init then stops at "terrain contains non-finite values"; derecho 1 km member, 859 cells). The same
            # factor the point door passes (mesh_point.thirty_second_supersample, hex 51d84998); N = 1 above 1.31 km.
            from .mesh_point import thirty_second_supersample
            supersample = thirty_second_supersample(folder / f"{name}.grid.nc")
            if supersample > 1:
                argv += ["--supersample-30s", str(supersample)]
            seconds = _run_logged(argv, logs / f"{name}-static.log", env=env)
            _stamp(folder, name, f"static {seconds:.0f}s ({_cores(env)})")
        init_receipt = folder / f"{name}-init.json"
        init = folder / f"{name}.init.nc"
        from .init_physics import profile_init_matches
        from .moving_case import resolve_physics
        init_physics_flags = resolve_physics(case, source=stage["source"])
        if (static_rebuilt or not init_receipt.is_file() or not init.is_file()
                or not profile_init_matches(init, init_physics_flags)):
            init.unlink(missing_ok=True)
            table = init_sources.load_table()
            row = init_sources.resolve_source(table, stage["source"])
            cycle = init_sources.Cycle.parse(stage["cycle"])
            argv = init_sources.init_argv(table, row, met=Path(stage["met"]), grid=folder / f"{name}.grid.nc",
                                          static=folder / f"{name}.static.nc", vertical_spec=vertical,
                                          out=init, cycle=cycle)
            from .physics_default import init_flags
            argv += init_flags(init_physics_flags)
            argv += init_sources.cold_soil_init_flags(stage.get("products", []), row.get("cold_soil"))
            argv += init_sources.surface_input_init_flags(stage.get("surface_input"))
            seconds = _run_logged([sys.executable, "-m", "woof.hex.cli", *argv], logs / f"{name}-init.log", env=env)
            init_receipt.write_text(json.dumps(dict(argv=argv, seconds=round(seconds, 1), source=stage["source"],
                                                    met=stage["met"], init_sha256=_sha256(init)), indent=2) + "\n")
            _stamp(folder, name, f"init {seconds:.0f}s ({_cores(env)})")
        member = register_member(case, name, float(lat), float(lon), folder)
        _stamp(folder, name, "rows-done")
    log(json.dumps(member))
    return member


def prepare_case(case_path: Path, casedir: Path, *, geog: Path | None = None, threads: int | None = None,
                 tile_workers: int | None = None, log: Callable[[str], None] = print) -> dict[str, Any]:
    """The start member (and, for a corridor case, its frame and template) in CASEDIR."""

    case = load_case(case_path)
    casedir = Path(casedir).expanduser().absolute()
    read_stage(casedir)
    lat, lon = (float(v) for v in case["storm"]["initial_center_deg"])
    corridor = bool(case["mesh"].get("corridor"))
    if corridor:
        env = _tool_env(threads, tile_workers)
        for name, body in (("frame", frame_spec(case)), ("template", member_spec(case, "template", lat, lon))):
            spec_path = casedir / f"{name}.spec.json"
            if not spec_path.is_file():
                spec_path.write_text(json.dumps(body, indent=2) + "\n")
            if not (casedir / f"{name}-mesh.json").is_file() or not (casedir / f"{name}.grid.nc").is_file():
                log(f"prepare: {name} mesh")
                seconds = generate_mesh(spec_path, casedir / f"{name}.grid.nc", casedir / f"{name}-mesh.json",
                                        casedir / "logs" / f"{name}-mesh.log", env)
                _stamp(casedir, name, f"mesh {seconds:.0f}s")
    rotate_from = None
    if not corridor and not _near_antipode((lat, lon), TEMPLATE_CENTER):
        log("prepare: the premade rotation template")
        rotation_template(case, casedir, env=_tool_env(threads, tile_workers), log=log)
        rotate_from = TEMPLATE_NAME
    log("prepare: start member")
    return prepare_member(case_path, casedir, "start", lat, lon, casedir=casedir, stitched=corridor,
                          rotate_from=rotate_from, geog=geog, threads=threads, tile_workers=tile_workers, log=log)


#: The premade family mesh the start member is rotated from: the case's own concentric cap family centred here.
#: THE BREAKAGE THIS PREVENTS (owner ruling 2026-10-23: premade meshes, no lengthy mesh creation on the start path;
#: and the moving start-member stage of 940 s on new boxes, 2026-10-25): every storm- and setup-following run
#: generated its start mesh on the box before the first step. The template is keyed in hex's prep store by its
#: generator spec alone (no storm, no date), so a WOOF image carries it (the bake runs `woof hex moving template`)
#: and a box rotates it to the storm in about a second, as the anticipated members already rotate the start mesh.
TEMPLATE_NAME = "rotation-template"
TEMPLATE_CENTER = (0.0, 0.0)


def _near_antipode(point, centre, degrees: float = 2.0) -> bool:
    """Whether ``point`` lies within ``degrees`` of ``centre``'s antipode, where the minimal rotation is ill posed
    (:func:`woof.hex.moving_mesh.rotation_between` refuses the antipode itself); such a start generates its mesh."""
    import math

    from .moving_case import EARTH_RADIUS_KM, great_circle_km

    return math.pi * EARTH_RADIUS_KM - great_circle_km(point, centre) < math.radians(degrees) * EARTH_RADIUS_KM


def template_spec(case: dict[str, Any]) -> dict[str, Any]:
    """The rotation template's generator spec: the case's member spec about TEMPLATE_CENTER."""
    return member_spec(case, TEMPLATE_NAME, *TEMPLATE_CENTER)


def rotation_template(case: dict[str, Any], folder: Path, *, env: dict[str, str] | None = None,
                      log: Callable[[str], None] = print) -> Path:
    """The family template's grid in FOLDER: fetched from hex's prep store when an image carries it, else
    generated once here (and staged into the store's outbox when one is set, as the bake does).

    Its spec file in FOLDER is what :func:`prepare_member` checks before it rotates (``rotate_from``)."""
    from . import prep_store

    folder = Path(folder).expanduser().absolute()
    folder.mkdir(parents=True, exist_ok=True)
    spec_path = folder / f"{TEMPLATE_NAME}.spec.json"
    grid = folder / f"{TEMPLATE_NAME}.grid.nc"
    receipt = folder / f"{TEMPLATE_NAME}-mesh.json"
    with locked(folder / f"{TEMPLATE_NAME}.prep.lock"):
        body = template_spec(case)
        if not spec_path.is_file() or json.loads(spec_path.read_text()) != body:
            grid.unlink(missing_ok=True)
            receipt.unlink(missing_ok=True)
            spec_path.write_text(json.dumps(body, indent=2) + "\n")
        if grid.is_file() and receipt.is_file():
            return grid
        began = time.time()
        logs = folder / "logs"
        got = prep_store.run_or_fetch(
            [_tool("MESH"), "--spec", spec_path, "--out", grid, "--receipt", receipt],
            kind="mesh", runner=lambda argv: _run_logged(argv, logs / f"{TEMPLATE_NAME}-mesh.log", env=env))
        how = ("fetched from the prep store" if got.get("source") == "store"
               else f"generated here ({_cores(env)}); no prep store carries it")
        _stamp(folder, TEMPLATE_NAME, f"template {time.time() - began:.1f}s, {how}")
        log(f"rotation template: {how}")
    return grid


# ---------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------
def build_operator(source: Path, target: Path, out: Path) -> dict[str, Any]:
    from .moving_remap import ConservativeMap, SphericalCells

    from .stage_timing import stage

    began = time.monotonic()
    stages = []
    with stage(stages, "read-operator-geometry", parallelism={"device": "CPU", "threads": 1}):
        source_cells, target_cells = SphericalCells.read(source), SphericalCells.read(target)
    with stage(stages, "build-conservative-operator", parallelism={"device": "CPU"}) as timing:
        weights = ConservativeMap.build(source_cells, target_cells,
            progress=lambda row: print(json.dumps(row), file=sys.stderr, flush=True))
        execution = weights.receipt["overlap_execution"]
        timing["parallelism"] = {"device": "CPU", "executor": execution["executor"],
                                 "workers": execution["workers"],
                                 "scope": "overlap parallel; validation/reconciliation coordinator"}
    tmp = Path(f"{out}.{os.getpid()}.partial.npz")
    with stage(stages, "save-conservative-operator", parallelism={"device": "CPU", "threads": 1}):
        try:
            weights.save(tmp)
            tmp.replace(out)
        finally:
            tmp.unlink(missing_ok=True)
    # Keep execution evidence in the controller's log after disposable NPZs
    # are retired. This is the same receipt saved inside the operator archive.
    return dict(source=str(source), target=str(target), out=str(out), seconds=time.monotonic() - began,
                stages=stages, overlap_execution=execution, operator_receipt=weights.receipt)


def build_zone(source_grid, target_grid, source_static, target_static, source_init, target_init,
               out: Path) -> dict[str, Any]:
    from .moving_zone import ZoneMap

    began = time.monotonic()
    zone = ZoneMap.build(source_grid, target_grid, source_static=source_static, target_static=target_static,
                         source_init=source_init, target_init=target_init)
    tmp = Path(f"{out}.{os.getpid()}.partial.npz")
    zone.save(tmp)
    tmp.replace(out)
    return dict(out=str(out), seconds=time.monotonic() - began)


# Mesh-family members above are geometry. Each ensemble member below owns
# an entire moving controller, including its private physics/restart state.
ENSEMBLE_SCHEMA = "gpuwm-hex.moving-ensemble/v1"
ROSTER_SCHEMA = "gpuwm-hex.moving-ensemble-roster/v1"


def case_physics_suite(case) -> str | None:
    """The explicit MPAS suite a moving case's ``physics_args`` select, or None for the WOOF route.

    Read through the forecast door's own parser and :func:`woof.hex.mpasphys.forecast_v842.native_route`,
    so a ``--physics-suite``, a ``--physics-namelist`` or ``--physics-backend mpasphys`` all count, exactly
    as the segment's forecast door will decide.
    """
    import argparse
    args = [str(a) for a in case.get("physics_args", []) or []]
    if not any(a.startswith(("--physics-suite", "--physics-namelist", "--physics-backend")) for a in args):
        return None
    from .forecast_door import add_forecast_arguments
    from .mpasphys.forecast_v842 import native_route
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    add_forecast_arguments(parser)
    namespace, _ = parser.parse_known_args(args)
    config, _ = native_route(namespace, default_mode="correct")
    if config is None:
        return None
    options = getattr(config, "options", None) or {}
    return str(options.get("config_physics_suite") or getattr(namespace, "physics_suite", None) or "mpas")


def roster_physics_rows(default, suite: str | None):
    """The packaged roster's physics rows for the route a case runs on.

    The WOOF route alternates its two microphysics schemes (``physics``).  An MPAS suite route runs the
    suite's own slot schemes only (the ported glue admits convection_permitting's slots and no other), so
    its physics diversity comes from that suite's own Registry options (``mpas_suite_physics``); a suite
    with no listed rows takes its default physics, and its members differ by SPPT/SKEB and member identity.
    THE BREAKAGE THIS PREVENTS (HEX-PHYSICS-PARITY run group 2 #6): the WOOF rows on the MPAS route
    asked for mp_wsm6, which the route refuses, so half the default ensemble could never run there.
    """
    if suite is None:
        return default["physics"]
    rows = (default.get("mpas_suite_physics") or {}).get(suite)
    return rows or [dict(id=f"{suite}-default", forecast_flags=[])]


def moving_roster(case_path, casedir, *, manifest=None, count=8, seed=53):
    """Resolve weighted metadata rows to independent moving forecasts."""
    import math
    import re
    from .stochastic import member_number, member_options
    from .init_sources import Cycle
    if type(count) is not int or count < 2:
        raise MovingCaseError("moving ensemble needs at least two members to estimate spread")
    case = load_case(case_path)
    default = json.loads((Path(__file__).with_name("data") / "moving-ensemble.v1.json").read_text())
    if manifest is None:
        physics_rows = roster_physics_rows(default, case_physics_suite(case))
        body = dict(schema=ROSTER_SCHEMA, source_members=1, members=[
            dict(id=f"member-{i+1:03d}", weight=1.0,
                 physics=physics_rows[i % len(physics_rows)]["id"],
                 forecast_flags=physics_rows[i % len(physics_rows)]["forecast_flags"])
            for i in range(count)])
        base = Path(case["_path"]).parent
    else:
        manifest = Path(manifest).expanduser().resolve()
        body = json.loads(manifest.read_text())
        base = manifest.parent
    if body.get("schema") != ROSTER_SCHEMA:
        raise MovingCaseError(f"roster schema must be {ROSTER_SCHEMA}; untyped rows can mix clocks")
    records = body.get("members", [])
    if len(records) != count:
        raise MovingCaseError(f"roster contains {len(records)} rows but --count is {count}; selection must be explicit")
    blocked = {"--out", "--scratch", "--hours", "--init", "--grid", "--static",
               "--mesh", "--members", "--start-time", "--init-source", "--stochastic",
               "--history-every-minutes", "--history-from-minutes", "--dt", "--receipt",
               "--restart-from", "--preflight", "--cards", "--gpuwm-checkout",
               "--fullstep-local-timestep", "--fullstep-lts-mode", "--fullstep-lts-max-rate"}
    import argparse
    from .forecast_door import add_forecast_arguments
    # Manifest argv must spell options in full. argparse abbreviations could
    # otherwise bypass the controller-owned option check (for example --hour).
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    add_forecast_arguments(parser)
    for action in parser._actions:
        if blocked.intersection(action.option_strings):
            blocked.update(action.option_strings)
    ids, identities, hashes, recipes = set(), set(), {}, set()
    resolved = []
    for raw in records:
        row = dict(raw)
        name = row.get("id")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or name in ids:
            raise MovingCaseError(f"unsafe or duplicate ensemble member id {name!r}")
        weight = row.get("weight", 1.0)
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
            raise MovingCaseError(f"member {name}: weight must be finite and positive")
        if not isinstance(body.get("forecast_flags", []), list) or not isinstance(row.get("forecast_flags", []), list):
            raise MovingCaseError(f"member {name}: forecast_flags must be argv lists")
        flags = [*body.get("forecast_flags", []), *row.get("forecast_flags", [])]
        if any(not isinstance(a, str) or a.split("=", 1)[0] in blocked or a.startswith("--samehost")
               for a in flags):
            raise MovingCaseError(f"member {name}: flags would replace controller state, clock or device assignment")
        try:
            parser.parse_args(flags)
        except SystemExit as error:
            raise MovingCaseError(f"member {name}: invalid forecast flags {flags}") from error
        identity = member_number(name)
        options = dict(body.get("stochastic", {}), **row.get("stochastic", {}))
        if "member" in options or "seed" in options:
            raise MovingCaseError(f"member {name}: identity and seed are owned by the roster")
        stochastic = member_options(identity, seed=seed,
            source_members=body.get("source_members", 1), ensemble_size=count, **options)
        if identity in identities:
            raise MovingCaseError(f"member {name}: stochastic identity collision")
        source_dir = Path(row.get("casedir", casedir)).expanduser()
        if "casedir" in row and not source_dir.is_absolute():
            source_dir = base / source_dir
        source_dir = source_dir.resolve()
        source = read_stage(source_dir)
        if Cycle.parse(source["cycle"]).when != Cycle.parse(cycle_of(case)).when:
            raise MovingCaseError(f"member {name}: prepared cycle {source['cycle']} differs from {cycle_of(case)}")
        start = row.get("start_member", "start")
        family = json.loads((source_dir / "family.json").read_text())
        member = next((m for m in family if m["name"] == start), None)
        if member is None:
            raise MovingCaseError(f"member {name}: {source_dir}/family.json has no {start}")
        for key in ("grid", "static", "init", "mesh_rows"):
            if not Path(member[key]).is_file():
                raise MovingCaseError(f"member {name}: prepared {key} missing: {member[key]}")
        if float(member["dt_seconds"]) != float(case["dt_seconds"]):
            raise MovingCaseError(f"member {name}: prepared time step differs from the moving case")
        init = str(Path(member["init"]).resolve())
        if init not in hashes:
            hashes[init] = _sha256(Path(init))
        recipe = (hashes[init], json.dumps(flags), json.dumps(stochastic, sort_keys=True)
                  if stochastic["sppt"] or stochastic["skeb"] else "no stochastic forcing")
        if recipe in recipes:
            raise MovingCaseError(f"member {name}: identical input and forcing would duplicate a forecast")
        recipes.add(recipe)
        ids.add(name); identities.add(identity)
        resolved.append(dict(row, id=name, weight=float(weight), casedir=str(source_dir),
            start_member=start, forecast_flags=flags, stochastic=stochastic,
            provenance=dict(source=source["source"], cycle=source["cycle"],
                source_member=source.get("source_member"),
                stage_sha256=_sha256(source_dir / STAGE_FILE), init_sha256=hashes[init],
                prepared_member=member,
                perturbation=dict(sppt=stochastic["sppt"], skeb=stochastic["skeb"]),
                physics=row.get("physics", "explicit forecast flags"))))
    scale = max(row["weight"] for row in resolved)
    total = math.fsum(row["weight"] / scale for row in resolved)
    for row in resolved:
        row["weight"] = (row["weight"] / scale) / total
        if row["weight"] == 0:
            raise MovingCaseError(f"member {row['id']}: weight ratio is too small to represent")
    return resolved


def moving_ensemble_products(rows, output):
    """Aggregate tracker receipt scalars at coincident valid times.

    No fields on different moving grids are ever averaged. Missing fixes and
    intensities remain missing, with included counts and weights reported.
    The track mean is spherical so dateline crossings do not create jumps.
    """
    import math
    from .ensemble import write_json
    from .moving_case import great_circle_km
    leads, tracks = {}, []
    for row in rows:
        state_path = Path(row.get("out", ".")) / "controller.json"
        if not state_path.is_file():
            continue
        controller = json.loads(state_path.read_text())
        for frame in controller["frames"]:
            fix = frame["fix"]
            record = dict(id=row["id"], weight=row["weight"], seconds=frame["model_seconds"],
                **{k: fix.get(k) for k in ("ok", "lat", "lon", "mslp_min_hpa", "vmax10_ms")})
            tracks.append(record)
            leads.setdefault(record["seconds"], []).append(record)
    summaries = []
    def quantile(records, key, fraction):
        ordered = sorted((r[key], r["weight"]) for r in records)
        total, cumulative = sum(w for _, w in ordered), 0.0
        for value, weight in ordered:
            cumulative += weight
            if cumulative >= fraction * total:
                return value
        return None
    for seconds, records in sorted(leads.items()):
        good = [r for r in records if r["ok"] and r["lat"] is not None and r["lon"] is not None]
        summary = dict(seconds=seconds, fixes=len(good), members_expected=len(rows),
                       included_weight=sum(r["weight"] for r in good))
        if good:
            vectors = [(math.cos(math.radians(r["lat"]))*math.cos(math.radians(r["lon"])),
                        math.cos(math.radians(r["lat"]))*math.sin(math.radians(r["lon"])),
                        math.sin(math.radians(r["lat"]))) for r in good]
            centre = [sum(v[i]*r["weight"] for v, r in zip(vectors, good)) for i in range(3)]
            norm = math.sqrt(sum(v*v for v in centre))
            if norm > 1e-12:
                lat = math.degrees(math.asin(max(-1, min(1, centre[2]/norm))))
                lon = math.degrees(math.atan2(centre[1], centre[0]))
                distances = [great_circle_km((lat, lon), (r["lat"], r["lon"])) for r in good]
                summary.update(lat=lat, lon=lon, track_rms_km=math.sqrt(
                    sum(r["weight"]*d*d for r, d in zip(good, distances))/summary["included_weight"]))
        for key in ("mslp_min_hpa", "vmax10_ms"):
            available = [r for r in good if r[key] is not None and math.isfinite(r[key])]
            summary[key] = dict(count=len(available), included_weight=sum(r["weight"] for r in available),
                p10=quantile(available, key, .1), p50=quantile(available, key, .5),
                p90=quantile(available, key, .9))
        summaries.append(summary)
    product = dict(schema=ENSEMBLE_SCHEMA, diagnostic="model-state tracker; research spread, uncalibrated",
                   members=rows, tracks=tracks, leads=summaries)
    write_json(Path(output) / "ensemble-products.json", product)
    return product


def render_moving_tracks(product, output, analysis_time):
    """Drive the native renderer over tracker receipt scalars, retaining gaps."""
    from datetime import timedelta
    from woof import rustwx
    from .init_sources import Cycle
    from .render_door import resolve_renderer_exe, resolve_basemap_dir, renderer_environment
    from .ensemble import write_json

    root = Path(output)
    origin = Cycle.parse(analysis_time).when
    seconds = sorted({row["seconds"] for row in product["tracks"]})
    records = {(row["id"], row["seconds"]): row for row in product["tracks"]}
    members, tracks = [], []
    for member in product["members"]:
        metadata = dict(member_id=member["id"], weight=member["weight"])
        members.append(metadata)
        fixes = []
        for lead in seconds:
            row = records.get((member["id"], lead))
            if row is None or not row["ok"]:
                fixes.append(None)
                continue
            pressure = row["mslp_min_hpa"]
            fixes.append(dict(valid=(origin + timedelta(seconds=lead)).isoformat(),
                lat=row["lat"], lon=row["lon"],
                mslp_pa=None if pressure is None else pressure * 100,
                max_wind10_ms=row["vmax10_ms"]))
        tracks.append(dict(metadata, track_id=member["id"], fixes=fixes))
    manifest, picture = root / "native-tracks.json", root / "track-plume.png"
    write_json(manifest, dict(schema="gpuwm.tc-tracks.v1", members=members, tracks=tracks))
    renderer = resolve_renderer_exe()
    env = rustwx.renderer_env()
    env.update(renderer_environment(resolve_basemap_dir(renderer)))
    command = [str(renderer), "--tc-track-plume", str(manifest), str(picture)]
    started = time.monotonic()
    result = subprocess.run(command, text=True, capture_output=True, env=env, timeout=180)
    write_json(root / "track-render-receipt.json", dict(command=command,
        renderer_sha256=_sha256(renderer), wrapper="woof.rustwx", parallelism="native Rust",
        wall_seconds=time.monotonic()-started, returncode=result.returncode,
        stdout=result.stdout, stderr=result.stderr))
    if result.returncode or not picture.is_file():
        raise MovingCaseError(f"native track plume failed: {result.stderr.strip()}; "
                              f"receipt {root / 'track-render-receipt.json'}")
    return str(picture)


def plot_moving_ensemble(product, output):
    """Analysis charts of receipt scalars; weather-field maps use the Rust door."""
    import math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    colours = plt.get_cmap("tab20")
    positioned = [r for r in product["tracks"] if r["ok"] and r["lat"] is not None and r["lon"] is not None]
    origin = min(positioned, key=lambda r: (r["seconds"], r["id"])) if positioned else None
    for i, member in enumerate(product["members"]):
        track = sorted((r for r in product["tracks"] if r["id"] == member["id"] and r["ok"]),
                       key=lambda r: r["seconds"])
        for key, ax in (("mslp_min_hpa", axes[1, 0]), ("vmax10_ms", axes[1, 1])):
            values = [r for r in track if r[key] is not None]
            ax.plot([r["seconds"]/3600 for r in values], [r[key] for r in values],
                    label=member["id"], color=colours(i % 20), lw=1)
        points = [r for r in track if r["lat"] is not None and r["lon"] is not None]
        if points:
            x = [6371.229*math.cos(math.radians(origin["lat"]))*math.radians(
                (r["lon"]-origin["lon"]+180)%360-180) for r in points]
            y = [6371.229*math.radians(r["lat"]-origin["lat"]) for r in points]
            axes[0, 0].plot(x, y, marker=".", label=member["id"], color=colours(i % 20))
    leads = product["leads"]
    axes[0, 1].plot([r["seconds"]/3600 for r in leads if "track_rms_km" in r],
                    [r["track_rms_km"] for r in leads if "track_rms_km" in r])
    for key, ax in (("mslp_min_hpa", axes[1, 0]), ("vmax10_ms", axes[1, 1])):
        valid = [r for r in leads if r[key]["count"]]
        ax.fill_between([r["seconds"]/3600 for r in valid], [r[key]["p10"] for r in valid],
                        [r[key]["p90"] for r in valid], color="grey", alpha=.2, label="weighted P10-P90")
        ax.set_xlabel("Forecast hour")
        ax.legend(fontsize=6)
    axes[0, 0].set(xlabel="East displacement from common first fix (km)",
                   ylabel="North displacement from common first fix (km)", title="Member tracks")
    axes[0, 0].legend(fontsize=7)
    axes[0, 1].set(xlabel="Forecast hour", ylabel="Weighted RMS spread (km)", title="Track spread")
    axes[1, 0].set(ylabel="Central MSLP (hPa)", title="Intensity plume")
    axes[1, 1].set(ylabel="Maximum 10 m wind (m/s)", title="Intensity plume")
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.suptitle("Moving-region ensemble | model tracker | research spread, uncalibrated")
    path = Path(output) / "track-intensity.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


RAW_SUFFIXES = (".nc", ".pkl", ".npz", ".npy", ".grib", ".grb", ".bin")


def member_products_settled(run_dir, alive=None) -> bool:
    """True once no product stage of this member can still read its raw files.

    Each closed segment's forecast starts its product door detached, and those
    stages keep drawing after the member's controller has exited: they read the
    segment histories and the member grids.  A segment is settled when its
    products.log has the door's ``PRODUCTS done`` line, or when it has no
    products.log, or when no live process is its product door (a stage that
    died will never read again; ``moving run --resume`` redraws those).
    """
    from .moving_run import product_stage_alive
    alive = alive or product_stage_alive
    state_path = Path(run_dir) / "controller.json"
    if not state_path.is_file():
        return False
    for record in json.loads(state_path.read_text()).get("segments", []):
        fc = Path(record.get("path", "")) / "fc"
        log = fc / "products.log"
        if not log.is_file():
            continue
        if any(line.startswith("PRODUCTS done ") for line in log.read_text(errors="replace").splitlines()):
            continue
        if alive(fc):
            return False
    return True


def retire_member_raw(run_dir, *, min_bytes=10_000_000, alive=None):
    """Delete a passed member's raw arrays once its product stages are settled.

    THE BREAKAGE THIS PREVENTS: the ensemble launchers deleted each member's raw
    netCDF (histories, member grids) as soon as its receipt row said passed.
    The member's last detached product stage was still drawing, so on a rented box
    (2026-10-21, ensemble-08) members 002, 003 and 005 lost their last
    segment's refined-context maps: "--mesh names a missing file
    .../members/<member>.grid.nc".  Returns None while a stage still runs,
    else the deleted files.  The ensemble products need only controller.json.
    """
    if not member_products_settled(run_dir, alive=alive):
        return None
    deleted = []
    for folder, _dirs, files in os.walk(run_dir, followlinks=False):
        for name in files:
            path = Path(folder) / name
            if path.is_symlink() or path.suffix not in RAW_SUFFIXES:
                continue
            size = path.stat().st_size
            if size < min_bytes:
                continue
            path.unlink()
            deleted.append(dict(path=str(path), bytes=size))
    return deleted

def run_moving_ensemble(args):
    """One independent controller process per granted card, queued by card."""
    from concurrent.futures import ThreadPoolExecutor
    import math
    import threading
    from .ensemble import write_json

    if args.hours <= 0 or not math.isfinite(args.hours):
        raise MovingCaseError("--hours must be positive and finite")
    cards = (args.devices or os.environ.get("CUDA_VISIBLE_DEVICES", "0")).split(",")
    if not cards or any(not c.strip() for c in cards) or len(set(cards)) != len(cards):
        raise MovingCaseError("--devices must name distinct granted cards")
    if args.prep_threads < 1:
        raise MovingCaseError("--prep-threads must be positive")
    output = Path(args.out).expanduser().resolve()
    rows = moving_roster(args.case, args.casedir, manifest=args.manifest, count=args.count, seed=args.seed)
    from .drivers.fullstep_lts_forecast_v841 import require_stochastic_support
    for row in rows:
        try:
            require_stochastic_support(row["stochastic"])
        except ValueError as error:
            raise MovingCaseError(f"member {row['id']}: {error}") from error
    case = load_case(args.case)
    case["vertical_spec"] = str(resolve_vertical_spec(case))
    case.pop("_path", None)
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(schema=ENSEMBLE_SCHEMA, status="running", started_utc=_now(),
                   command=sys.argv, cards=cards, members=rows,
                   cuda_graphs=False,
                   parallelism=dict(controllers=len(cards), members_per_card=1,
                                    preparation_threads=args.prep_threads),
                   peak_vram_mib=None, stages=[],
                   case_sha256=_sha256(Path(args.case)),
                   manifest_sha256=_sha256(Path(args.manifest)) if args.manifest else None)
    write_json(output / "ensemble-receipt.json", receipt)
    began, stop = time.monotonic(), threading.Event()
    def descendants(pid):
        found = {pid}
        try:
            children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
        except OSError:
            children = []
        for child in children:
            found.update(descendants(int(child)))
        return found
    def retire_settled():
        for row in rows:
            if row.get("status") != "passed" or row.get("raw_retired") is not None:
                continue
            deleted = retire_member_raw(row["out"])
            if deleted is not None:
                with receipt_lock:
                    row["raw_retired"] = dict(files=len(deleted), bytes=sum(d["bytes"] for d in deleted),
                        rule="after every segment product stage is done or gone")
                checkpoint()
    def sample():
        peak = 0
        while not stop.is_set():
            try:
                card = subprocess.run(["nvidia-smi", "-i", ",".join(cards),
                    "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    text=True, capture_output=True, timeout=5)
                applications = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits"], text=True, capture_output=True, timeout=5)
                allocations = {}
                for line in applications.stdout.splitlines():
                    pid, memory = (v.strip() for v in line.split(","))
                    allocations[int(pid)] = allocations.get(int(pid), 0) + int(memory)
                with receipt_lock:
                    for row in rows:
                        if row.get("pid") and row.get("returncode") is None:
                            memory = sum(allocations.get(pid, 0) for pid in descendants(row["pid"]))
                            row["peak_sampled_vram_mib"] = max(row.get("peak_sampled_vram_mib", 0), memory)
                peak = max(peak, sum(int(v) for v in card.stdout.splitlines()))
                receipt["peak_vram_mib"] = peak
                receipt["vram_scope"] = "whole granted cards, includes other pool owners; not allocator peak"
            except (ValueError, OSError, subprocess.TimeoutExpired):
                pass
            if getattr(args, "retire_member_raw", False):
                try:
                    retire_settled()
                except (OSError, ValueError, KeyError):
                    pass  # the next sample retries; a raw file left is disk, not a result
            stop.wait(5)
    # SIGINT lets each controller execute its child-tree cleanup in finally.
    # Killing only the ensemble parent would leave forecast grandchildren alive.
    import signal
    previous_handlers = {}
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, lambda *_: stop.set())
    receipt_lock = threading.Lock()
    def checkpoint():
        with receipt_lock:
            write_json(output / "ensemble-receipt.json", receipt)
    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    def worker(device, assigned):
        for row in assigned:
            if stop.is_set():
                row["status"] = "cancelled"
                continue
            shared = row
            row = dict(row)
            root = output / row["id"]
            root.mkdir()
            local = dict(case)
            local["init_source"] = row["provenance"]["source"]
            local.pop("init_member", None)
            if row["provenance"]["source_member"] is not None:
                local["init_member"] = row["provenance"]["source_member"]
            local.pop("_requested_physics_args", None)
            local["physics_args"] = [*case.get("_requested_physics_args", case.get("physics_args", [])), *row["forecast_flags"],
                "--stochastic", json.dumps(row["stochastic"], sort_keys=True)]
            case_file = root / "case.json"
            write_json(case_file, local)
            row["out"] = str(root / "run")
            command = [sys.executable, "-m", "woof.hex.cli", "moving", "run",
                "--case", str(case_file), "--casedir", row["casedir"], "--start-member", row["start_member"],
                "--out", row["out"], "--hours", str(args.hours), "--device", device,
                "--prep-threads", str(args.prep_threads), "--prep-tile-workers", str(args.prep_threads)]
            from woof.output_formats import argument_values
            command += argument_values(args, relative=row["id"])
            if args.engine:
                command += ["--engine", str(args.engine)]
            if args.geog:
                command += ["--geog", str(args.geog)]
            row.update(command=command, device=device, log=str(root / "controller-launch.log"))
            start = time.monotonic()
            with Path(row["log"]).open("w") as stream:
                proc = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                        env=dict(os.environ, WOOF_HEX_CUDA_GRAPHS="0",
                                                 WOOF_HEX_CUDA_GRAPHS_REQUIRE_REPLAY="0"),
                                        start_new_session=True)
                row["pid"] = proc.pid
                with receipt_lock:
                    shared.update(row)
                checkpoint()
                while True:
                    try:
                        row["returncode"] = proc.wait(timeout=1)
                        break
                    except subprocess.TimeoutExpired:
                        if stop.is_set():
                            proc.send_signal(signal.SIGINT)
                            try:
                                row["returncode"] = proc.wait(timeout=120)
                            except subprocess.TimeoutExpired:
                                proc.kill()
                                row["returncode"] = proc.wait()
                            break
            row["wall_seconds"] = time.monotonic()-start
            state_path = Path(row["out"]) / "controller.json"
            if state_path.is_file():
                state = json.loads(state_path.read_text())
                row["summary"] = state.get("summary")
                row["stage_parallelism"] = dict(atmosphere="GPU", controller="CPU coordinator",
                    member_preparation=dict(rust_threads=args.prep_threads, tile_workers=args.prep_threads),
                    tracker="existing CPU tracker; per-frame timings in controller.json")
                row["vram_scope"] = "sampled controller descendants; 5 s cadence, not allocator peak"
                row["stages"] = [dict(segment=s.get("path"), wall_seconds=s.get("wall_seconds"),
                    integration_seconds=s.get("integration_seconds")) for s in state.get("segments", [])]
            row["status"] = "passed" if row["returncode"] == 0 and (row.get("summary") or {}).get("status") == "passed" else "failed"
            with receipt_lock:
                shared.update(row)
            checkpoint()
    try:
        with ThreadPoolExecutor(max_workers=len(cards)) as pool:
            futures = [pool.submit(worker, card, rows[i::len(cards)]) for i, card in enumerate(cards)]
            for future in futures:
                future.result()
        receipt["status"] = "passed" if all(r.get("status") == "passed" for r in rows) else "failed"
        failures = {r["id"]: member_failure(r) for r in rows if r.get("status") != "passed"}
        if failures:
            # Named first: a member that never ran leaves no fixes, and the track products below would
            # otherwise be the only error anyone sees (run group 2 #6: "native track plume failed").
            receipt["member_failures"] = failures
            for name, why in failures.items():
                print(f"woof hex: moving-ensemble member {name} failed: {why}", file=sys.stderr, flush=True)
        completed = [r for r in rows if (Path(r.get("out", ".")) / "controller.json").is_file()
                     and json.loads((Path(r["out"]) / "controller.json").read_text()).get("frames")]
        if completed:
            started = time.monotonic()
            try:
                product = moving_ensemble_products(completed, output)
                receipt["pictures"] = [render_moving_tracks(product, output, cycle_of(case)),
                                       plot_moving_ensemble(product, output)]
            except MovingCaseError as error:
                if not failures:
                    raise
                receipt["products_error"] = str(error)
            receipt["stages"].append(dict(name="receipt-analysis-products", wall_seconds=time.monotonic()-started,
                                           parallelism="native Rust track map; CPU scalar receipt analysis chart"))
        if failures and receipt.get("error") is None:
            receipt["error"] = "; ".join(f"{name}: {why}" for name, why in failures.items())
        return 0 if receipt["status"] == "passed" else 1
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        stop.set()
        monitor.join(timeout=12)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        receipt["wall_seconds"] = time.monotonic()-began
        write_json(output / "ensemble-receipt.json", receipt)


def member_failure(row) -> str:
    """Why one ensemble member failed, from its controller's own last words."""
    if row.get("status") == "cancelled":
        return "cancelled before it started"
    log = Path(row.get("log", ""))
    lines = log.read_text(errors="replace").splitlines()[-80:] if log.is_file() else []
    for line in reversed(lines):
        text = line.strip()
        if text.startswith("segment ") and ": " in text:
            return text  # the controller's SystemExit text (woof.hex.moving_run)
        if text.startswith(("SystemExit:", "woof hex:", "RuntimeError:", "MovingCaseError:", "ValueError:")):
            return text.split(":", 1)[1].strip() or text
    return f"controller exited {row.get('returncode')}"


def add_moving_ensemble_arguments(parser):
    from woof.output_formats import add_arguments
    add_arguments(parser)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--casedir", type=Path, required=True, help="shared staged/prepared moving case")
    parser.add_argument("--manifest", type=Path, help=f"weighted roster ({ROSTER_SCHEMA})")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--hours", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--devices", help="granted device IDs; default inherited visible cards")
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--geog", type=Path)
    parser.add_argument("--prep-threads", type=int, default=max(1, (os.cpu_count() or 1)//4))
    parser.add_argument("--retire-member-raw", action="store_true",
                        help="delete each passed member's raw netCDF once its product stages finish")
    parser.set_defaults(handler=run_moving_ensemble)
