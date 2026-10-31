"""Compose existing mesh, init, forecast, products and score doors from data."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
import time
from datetime import datetime, timedelta
import copy
import math

from .engines import static_geog_argv
from .errors import MpasPortError

DATA = Path(__file__).with_name("data")
TABLE = DATA / "countries.v1.json"
SCHEMA = "gpuwm-hex.countries/v1"
STAGES = ("prepare", "forecast", "products", "score")
REGIONS = DATA / "regions.v1.json"


def load_row(name, table=TABLE):
    table = Path(table)
    data = json.loads(table.read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA:
        raise MpasPortError("country table schema differs; the row's stage bindings cannot be interpreted")
    rows = [r for r in data["countries"] if r["id"] == name]
    if len(rows) != 1:
        raise MpasPortError(f"country row {name!r} must occur exactly once")
    row = rows[0]
    for key in ("mesh", "init_source", "vertical_spec", "scorecard", "product_set"):
        if key not in row:
            raise MpasPortError(f"country row lacks {key}; its operational stages would be disconnected")
    from .init_sources import load_table, resolve_source
    resolve_source(load_table(), row["init_source"])
    if row["product_set"] not in data["product_sets"]:
        raise MpasPortError("country row names an absent product set")
    return row, data["product_sets"][row["product_set"]], table.parent



def region_request(args):
    """Geometry, views and resolution are table data; no event branching."""
    region = getattr(args, "region", None)
    point = getattr(args, "point", None)
    polygon = getattr(args, "polygon", None)
    drawn = _drawn_region(args)
    if drawn is not None and drawn["kind"] == "cap":
        # A drawn circle is the --point cap: the same geometry, the same identity and prepared mesh.
        point = f"{drawn['center_deg'][0]},{drawn['center_deg'][1]}"
        args.radius_km = float(drawn["radius_km"])
        drawn = None
    if not (region or point or polygon or drawn):
        return load_row(args.row, args.table)
    table = REGIONS if args.table == TABLE else args.table
    if region:
        args.row = region
        return load_row(region, table)
    data = json.loads(Path(table).read_text())
    if data.get("schema") != SCHEMA:
        raise MpasPortError("region table schema differs; geometry defaults cannot be interpreted")
    row = copy.deepcopy(data["defaults"])
    if point:
        try:
            lat, lon = [float(v) for v in point.split(",")]
        except (ValueError, TypeError) as error:
            raise MpasPortError("point must be LAT,LON") from error
        radius = getattr(args, "radius_km", None)
        if not (-90 <= lat <= 90 and -180 <= lon <= 180 and radius and math.isfinite(radius) and 0 < radius < 10000):
            raise MpasPortError("point latitude/longitude or radius is outside the finite geographic range")
        geometry = {"kind": "cap", "center_deg": [lat, lon], "radius_km": radius}
        half = math.degrees(radius / 6371.229)
        south, north = max(-90, lat-half), min(90, lat+half)
        longitude_half = (180.0 if south == -90 or north == 90 else
                          math.degrees(math.asin(min(1.0, math.sin(math.radians(half)) / math.cos(math.radians(lat))))))
        bounds = {"west": lon-longitude_half, "east": lon+longitude_half,
                  "south": south, "north": north}
        view = [lat, lon, radius*2, radius*2]
    else:
        geometry = drawn if drawn is not None else json.loads(polygon.read_text())
        points = geometry.get("vertices_deg", [])
        if geometry.get("kind") != "polygon" or len(points) < 3 or any(len(p) != 2 or not (-90 <= p[0] <= 90 and -180 <= p[1] <= 180) for p in points):
            raise MpasPortError("polygon needs at least three finite LAT,LON vertices")
        lats, lons = zip(*points)
        # Remove the largest empty longitude arc. A polygon around the date
        # line must not select a view centred on the opposite hemisphere.
        circular = sorted(set(value % 360 for value in lons))
        gaps = [(circular[(i+1) % len(circular)] + (360 if i == len(circular)-1 else 0) - value, i)
                for i, value in enumerate(circular)]
        gap, index = max(gaps)
        west = circular[(index+1) % len(circular)]
        east = west + 360-gap
        lat, lon = (min(lats)+max(lats))/2, ((west+east)/2+180) % 360-180
        west += lon-(west+east)/2
        east = west+360-gap
        bounds = dict(west=west, east=east, south=min(lats), north=max(lats))
        view = [lat, lon, max(100, (east-west)*111*math.cos(math.radians(lat))), max(100, (max(lats)-min(lats))*111)]
    import hashlib
    # A finer region (--fine-dx-m, with --background-km): the generator's
    # halving ladder from that background down to the asked spacing, and a
    # step of 6 s per km of it (the zoom globes' MPAS rule) where that is
    # shorter than the table's.  Part of the region's identity, so a finer
    # region never reuses a coarser one's prepared mesh.
    resolution = {}
    fine = getattr(args, "fine_dx_m", None)
    background = getattr(args, "background_km", None)
    if fine is not None or background is not None:
        from .mesh_point import ladder_rungs
        fine = float(fine if fine is not None else row["mesh"]["fine_dx_m"])
        background = float(background if background is not None else row["mesh"]["background_km"])
        ladder_rungs(background, fine)
        step = min(float(row["mesh"]["dt_seconds"]), 6.0 * fine / 1000.0)
        resolution = {"fine_dx_m": fine, "background_km": background, "dt_seconds": step}
    keyed = {**geometry, **({"resolution": resolution} if resolution else {})}
    identity = "region-" + hashlib.sha256(json.dumps(keyed, sort_keys=True).encode()).hexdigest()[:12]
    row.update(id=identity, geometry=geometry)
    row["mesh"].update(id=identity, point_deg=[lat, lon], **resolution)
    row["scorecard"]["region"] = {**bounds, "step_degrees": 0.25}
    args.row = identity
    products = ["--no-global", "--no-mesh-view", "--region", "region=region:" + ",".join(map(str, view)), "--jobs", "2", "--frame-jobs", "1"]
    return row, products, Path(table).parent


def _drawn_region(args):
    """The region ``--ellipse`` or ``--shape`` draws (a polygon row; a drawn circle stays a cap), or None.

    THE BREAKAGE THIS PREVENTS (hrrrv5 everywhere, 2026-10-31): the site offers circle, ellipse and drawn-polygon
    regions, and the country door took only a cap (--point) or a polygon file, while ``regional`` took --ellipse and
    --shape (GeoJSON, lat_lon_box). Both doors now read the same shapes through woof.hex.mesh_shape, so an ellipse or a
    GeoJSON outline is a region on either door, on every physics suite."""
    ellipse, shape = getattr(args, "ellipse", None), getattr(args, "shape", None)
    if ellipse is None and shape is None:
        return None
    from . import mesh_shape
    try:
        return mesh_shape.parse_ellipse(ellipse) if ellipse is not None else mesh_shape.load_shape(Path(shape))
    except mesh_shape.ShapeRefusal as error:
        raise MpasPortError(str(error)) from error


def producer_suite(named):
    """The forecast's suite when it initializes through its own producer row (hrrrv5, the rrfs_* suites), else None."""
    suite = named.get("suite")
    if not suite:
        return None
    from .physics_default import init_producer
    return suite if init_producer(suite) else None


def _producer_init_argv(args, value):
    """The producer's case-7 init on the cut, with the geography and the first guess's soil and surface inputs.

    The start state is the producer's own (its L60 vertical grid, RUC's nine soil levels, its static built from the
    geography with the fields the suite reads), from the start time's decoded intermediate (boundary endpoint 0), as
    ``regional`` does (regional_door._stage_init_state)."""
    argv = list(value["commands"]["producer_init"])
    from .mesh_point import resolve_geog
    argv += ["--geog", str(resolve_geog(args.geog).expanduser().absolute())]
    from .init_sources import cold_soil_init_flags, load_table, resolve_source, surface_input_init_flags
    source = resolve_source(load_table(getattr(args, "source_table", None)), value["row"]["init_source"])
    if source.get("cold_soil") or source.get("surface_input"):
        first = value["boundary_endpoints"][0]["argv"]
        decoded = json.loads((Path(first[first.index("--out-dir") + 1]) / "init-source-receipt.json").read_text())
        if source.get("cold_soil"):
            argv += cold_soil_init_flags(decoded["products"], source["cold_soil"])
        argv += surface_input_init_flags(decoded.get("surface_input"))
    return argv


def region_specification(row):
    """Use the generator's admitted halving ladder for table resolution."""
    from .mesh_point import ladder_spec, ladder_rungs
    mesh = row["mesh"]
    if row["geometry"]["kind"] == "cap":
        return ladder_spec(tuple(row["geometry"]["center_deg"]), fine_dx_m=mesh["fine_dx_m"],
            radius_km=row["geometry"]["radius_km"], background_km=mesh["background_km"], name=row["id"])
    # A drawn region (--polygon) takes the regional door's grown halving ladder, rung for rung as a cap does.
    # THE BREAKAGE THIS PREVENTS (hex-shapes-1card, V8X6000 2026-10-29 06:05-06:16Z, re-run on an RTX 4090 host 07:09Z with
    # the hex 0.4.10 generator): `country --polygon CONUS --fine-dx-m 3000 --background-km 96` (and 120 -> 3.75 km)
    # wrote ONE region with the table default's 540 km ramp (18 x its 30 km), whatever --fine-dx-m asked. The ramp is
    # a tanh centred on the outline, so a polygon a couple of ramps deep never reads its spacing to 0.1 % anywhere:
    # rw_mpas_mesh found no level-5 plateau (PLATEAU 5 0 0) and refused ("level 5 (h_l = 3.0 km) missed its
    # delivery contract: ... median = 0.9589"), and a polygon small enough to pass got cells well above the asked
    # spacing near its edge. shape_ladder_spec grows the outline by one fine ramp (18 x fine) and nests each
    # coarser rung outside it, so the whole drawn area is at the asked spacing.
    from .mesh_point import DEFAULT_RING_FACTOR, DEFAULT_TRANSITION_FACTOR
    from .mesh_shape import shape_ladder_spec
    return shape_ladder_spec(row["geometry"], rungs_km=ladder_rungs(mesh["background_km"], mesh["fine_dx_m"]),
        background_km=mesh["background_km"], transition_factor=DEFAULT_TRANSITION_FACTOR,
        ring_factor=DEFAULT_RING_FACTOR, name=row["id"])


def _suite_parent_init(value, parent_init, mesh):
    """The parent init completed for the planned forecast's physics suite, or None when it needs nothing.

    THE BREAKAGE THIS PREVENTS (3 km region from GFS 2026-10-23 18Z f348 with
    rrfs_v2_dev, box L, 2026-10-24): RUC integrates 9 soil levels, the init
    writes Noah's 4, and the forecast door replays the soil from the init's
    recorded source analysis -- which a CUT init does not carry, so every
    RRFS-shape region refused ("ruc needs 9 soil levels; mesh.init.nc has 4 and
    no source-analysis provenance").  The parent init still has its provenance:
    complete it for the suite here (soil replayed from the source analysis,
    the suite's species initialized) and cut that, so the region's init is the
    suite's from the start.
    """
    from .global_door import forecast_arg_physics
    forecast = [str(a) for a in value["commands"]["forecast"]]
    named = forecast_arg_physics([a.split("=", 1)[1] if a.startswith("--forecast-arg=") else a for a in forecast])
    if "suite" not in named:
        return None
    from .physics_suites import load_table
    if named["suite"] in set(load_table().get("mpasphys_v842", {}).get("suite_routes", {})):
        # An MPAS-A v8.4.2 suite runs from the plain init (case-7 fields, Noah's four soil levels; the forecast
        # allocates the suite's species), so there is nothing to complete. THE BREAKAGE THIS PREVENTS: the ArWen
        # column seam below refused convection_permitting ("config_radt_cld_scheme='cld_fraction' is refused"),
        # so every `country --forecast-arg=--physics-suite=convection_permitting` region stopped after its init.
        return None
    overrides = []
    for i, item in enumerate(forecast):
        text = item.split("=", 1)[1] if item.startswith("--forecast-arg=") else item
        if text == "--physics" and i + 1 < len(forecast):
            overrides.append(tuple(forecast[i + 1].split("=", 1)))
        elif text.startswith("--physics="):
            overrides.append(tuple(text.split("=", 1)[1].split("=", 1)))
    from .physics_suites import resolve_physics
    from .init_physics import prepare_forecast_init
    physics = resolve_physics(named["suite"], overrides or None, mesh=mesh["id"],
                              nominal_dx_m=mesh.get("fine_dx_m")).receipt()
    out = Path(parent_init).with_name("init.suite.nc")
    for stale in (out, out.with_name(out.name + ".provenance.json")):
        stale.unlink(missing_ok=True)
    prepared, receipt = prepare_forecast_init(Path(parent_init), out, physics, None)
    return Path(prepared) if receipt is not None else None


def _prepare_region(args, value, status):
    from .cli import main
    from . import mesh_rows
    from .mesh_point import admit_pair
    from .engines import resolve, ENGINES
    from .prep_store import run_or_fetch
    from .lbc_producer import produce_source_endpoints
    if args.endpoint_workers < 1:
        raise MpasPortError("endpoint workers must be positive")
    root = args.work_dir.resolve()
    parent = root / "parent"
    parent.mkdir(exist_ok=True)
    from .lbc_producer import atomic_json
    timings = {"schema": "gpuwm-hex.region-preparation/v1", "stages": {},
        "parallelism": {"endpoint_workers": args.endpoint_workers,
                        "rayon_threads": os.environ.get("RAYON_NUM_THREADS", "engine default")}}
    def measured(name, action):
        started = time.monotonic()
        status.note(name)
        stage = timings["stages"][name] = {}
        try:
            result = action()
            if isinstance(result, int):
                stage["returncode"] = result
            return result
        except Exception as error:
            stage["error"] = str(error)
            raise
        finally:
            stage["seconds"] = time.monotonic() - started
            atomic_json(root / "region-preparation-receipt.json", timings)
    row, mesh = value["row"], value["row"]["mesh"]
    spec = Path(value["paths"]["spec"])
    specification = region_specification(row)
    spec.write_text(json.dumps(specification))
    (root / "region.json").write_text(json.dumps(row["geometry"]))
    def engine(name):
        return str(resolve(next(e for e in ENGINES if e.name == name)))
    grid, static = parent / "mesh.grid.nc", parent / "mesh.static.nc"
    # A file at the destination is not evidence for this request. The store
    # keys the current geometry/engine and grid/geography before reusing bytes.
    measured("mesh", lambda: run_or_fetch([engine("rw_mpas_mesh"), "--spec", str(spec), "--out", str(grid),
                      "--receipt", str(parent / "mesh-receipt.json"), "--clobber"]))
    # The parent's static from the cut's halo and the parent's terrain (woof.hex.region_static: a new box spent
    # 15.5 min here building the whole parent globe's static for a 600 km region); the whole parent's static
    # only when the halo does not hold.
    from .region_static import build_parent_static
    center = mesh.get("point_deg") or row["geometry"].get("center_deg")

    def halo_static():
        if not center:
            return {"built": False, "why": "the region row names no centre to build a halo around"}
        try:
            return build_parent_static(
                parent_grid=grid, region=root / "region.json", center_deg=tuple(center), geog=args.geog,
                mesh_exe=Path(engine("rw_mpas_mesh")), static_exe=Path(engine("rw_mpas_static")), out=static,
                receipt=parent / "static-receipt.json", work=parent, run=run_or_fetch)
        except Exception as error:  # the whole parent's static below still builds the region, as before
            return {"built": False, "why": f"{type(error).__name__}: {error}"}
    halo = measured("halo-static", halo_static)
    timings["halo_static"] = {k: v for k, v in halo.items() if k not in ("halo_checks", "assembly")}
    if not halo["built"]:
        print(f"woof hex country: the region's halo static was not used ({halo.get('why')}); building the "
              f"whole parent's static", file=sys.stderr)
        measured("static", lambda: run_or_fetch([engine("rw_mpas_static"), "--grid", str(grid),
                          *static_geog_argv(args.geog), "--out", str(static),
                          "--receipt", str(parent / "static-receipt.json"), "--clobber"]))
    producer = "producer_init" in value["commands"]
    if not producer:
        # A producer suite (hrrrv5) starts from its own case 7 on the cut below, so the parent's v8.4.2 init is
        # neither built nor cut: the start time's intermediate comes from boundary endpoint 0's decode.
        status.note("initializing the global parent from the selected source")
        rc = measured("init", lambda: main(value["commands"]["init"]))
        if rc:
            return rc
        suite_init = measured("suite-init", lambda: _suite_parent_init(value, parent / "init" / "init.nc", mesh))
        if suite_init is not None:
            cull = list(value["commands"]["cull"])
            cull[cull.index("--parent-init") + 1] = str(suite_init)
            value["commands"]["cull"] = cull
    rc = measured("cull", lambda: main(value["commands"]["cull"]))
    if rc:
        return rc
    admission = measured("parent-admission", lambda: admit_pair(grid, static, fine_dx_m=mesh["fine_dx_m"]))
    registered = mesh_rows.describe_generated(name=mesh["id"] + "-parent", grid=grid, static=static,
        spec_path=spec, generator_receipt=parent / "mesh-receipt.json", static_receipt=parent / "static-receipt.json",
        point_deg=tuple(mesh["point_deg"]), fine_dx_m=mesh["fine_dx_m"], core_radius_km=mesh["core_radius_km"],
        background_km=mesh["background_km"], dt_seconds=mesh["dt_seconds"], n_levels=mesh["n_levels"], admission=admission)
    cut_admission = measured("regional-admission", lambda: admit_pair(root / "mesh.grid.nc", root / "mesh.static.nc", fine_dx_m=mesh["fine_dx_m"]))
    # The RRFS-shape suite's GSL drag reads the cut mesh's own orographic statistics (woof.hex.ugwp_oro).
    from .ugwp_oro import prepare_for_forecast
    measured("ugwp-oro-data", lambda: prepare_for_forecast(value["commands"]["forecast"], root / "mesh.grid.nc", args.geog))
    cut = mesh_rows.describe_cull(name=mesh["id"], parent=registered, grid=root / "mesh.grid.nc", static=root / "mesh.static.nc",
        cull_receipt=root / "mesh.cull.json", cull_region=row["geometry"], cull_pad_scale=1.0,
        lbc_source=f"{row['init_source']} source table endpoints in {root / 'lbc'}", admission=cut_admission,
        dt_seconds=mesh["dt_seconds"])
    mesh_rows.write_rows(root / "mesh-rows.json", [registered, cut])
    measured("boundary-endpoints", lambda: produce_source_endpoints(
        value["boundary_endpoints"], root / "boundary-met", workers=args.endpoint_workers))
    if producer:
        # THE BREAKAGE THIS PREVENTS (hrrrv5 physics on regular regions, 2026-10-31): the door built hex's own
        # 55-level v8.4.2 start state, completed RUC's nine soil levels with the ArWen seam, and wrote v8.4.2
        # boundaries, so the RRFS route found none of the fields NOAA's registry declares and every
        # `country --forecast-arg=--physics-suite=hrrrv5` region died as its forecast started (KeyError 'nscat' in
        # pool_v842, from GFS and from IFS). The producer's case 7 on the cut and its case 9 boundaries, as
        # `regional` does, are the state the suite's route reads.
        Path(value["paths"]["init"]).parent.mkdir(parents=True, exist_ok=True)
        rc = measured("producer-init", lambda: main(_producer_init_argv(args, value)))
        if rc:
            return rc
    return measured("lbc", lambda: main(value["commands"]["lbc"]))

def plan(args):
    try:
        issued = datetime.strptime(args.cycle, "%Y-%m-%dT%H")
    except ValueError as error:
        raise MpasPortError("country cycle must be YYYY-MM-DDTHH; it determines the init's valid time") from error
    lead = int(getattr(args, "lead_hours", 0) or 0)
    if lead < 0:
        raise MpasPortError(f"--lead-hours {lead} is negative; it is the forecast hour of --cycle the run starts from")
    row, products, data_dir = region_request(args)
    from .init_sources import load_table, resolve_source
    if "geometry" in row:
        region_specification(row)  # Reject invalid table resolution before acquisition.
        if args.grid or args.static or args.init:
            raise MpasPortError("region geometry prepares its own parent and cull; grid/static/init overrides cannot identify that chain")
    source_table = load_table(getattr(args, "source_table", None))
    source = resolve_source(source_table, getattr(args, "source", None) or row["init_source"])
    row = copy.deepcopy(row)
    row["init_source"] = source["id"]
    root = args.work_dir.resolve()
    mesh = row["mesh"]
    grid = args.grid or root / "mesh.grid.nc"
    static = args.static or root / "mesh.static.nc"
    init = args.init or root / "init" / "init.nc"
    spec = (root / "region-mesh.spec.json" if "geometry" in row else (data_dir / mesh["spec"]).resolve())
    vertical = (data_dir / row["vertical_spec"]).resolve()
    # The run starts at the cycle's forecast hour --lead-hours (0: the cycle itself): the init is that hour's
    # forecast and every boundary endpoint reads the same cycle further on (lbc_producer.source_endpoint_commands).
    start = (issued + timedelta(hours=lead)).strftime("%Y-%m-%d_%H:00:00")
    issued_label = f"{args.cycle} f{lead:03d}" if lead else args.cycle
    common = ["--grid", str(grid), "--static", str(static)]
    initialization = ["init-source", "--source", row["init_source"], "--cycle", args.cycle,
                      "--out-dir", str(root / "init"), *common, "--vertical-spec", str(vertical)]
    if lead:
        initialization += ["--lead-hours", str(lead)]
    for staged in args.staged:
        initialization += ["--staged", str(staged)]
    if getattr(args, "source_table", None):
        initialization += ["--table", str(args.source_table)]
    if getattr(args, "member", None) is not None:
        initialization += ["--member", str(args.member)]
    # A named suite (--forecast-arg=--physics-suite=NAME) takes the step the forecast door gives that
    # suite, as the global door does (global_door.forecast_physics_arm): the mesh's own --dt is the
    # ArWen column's and is not passed beside it.
    from .global_door import forecast_arg_physics
    from .physics_default import ensure
    # No physics named: the hex default MPAS suite by mesh (woof.hex.physics_default), written here so the
    # step rule below and the region's init completion see the suite the forecast runs.
    extra = ensure([str(item) for item in getattr(args, "forecast_arg", None) or ()], mesh.get("fine_dx_m"))
    named = forecast_arg_physics(extra)
    from .physics_default import door_step
    step = door_step(named, mesh["dt_seconds"], mesh.get("fine_dx_m"))
    forecast = ["forecast", "--mesh", mesh["id"], *common, "--init", str(init),
                "--start-time", start, "--init-source", f"{row['init_source']} {issued_label}; country row {args.row}",
                "--hours", str(args.hours), *step,
                "--history-every-minutes", str(args.history_every_minutes), "--out", str(root / "forecast"),
                "--scratch", str(root / "scratch"), "--no-products"]
    if args.gpuwm_checkout is not None:
        # Without a checkout the forecast door runs the installed engine and
        # records it by version and RECORD digest, so a pip install reaches
        # this door too.
        forecast += ["--gpuwm-checkout", str(args.gpuwm_checkout)]
    from .regional_door import samehost_argv
    forecast += samehost_argv(args)
    # The caller's forecast flags come last, so a --physics-suite (or a --dt) given here wins.
    from woof.output_formats import argument_values
    forecast += [*extra, *argument_values(args)]
    from .products_door import label_argv

    product_argv = ["products", str(root / "forecast"), "--out", str(root / "products"), *products,
                    *label_argv(args)]
    # Explicit regional views do not discover a refined-core centre in the
    # products door. Bind point products to this row instead of global cities.
    if "--points" not in products and "--no-points" not in products:
        points = row.get("product_points", [{"name": row["id"], "point_deg": mesh["point_deg"]}])
        product_argv += ["--points", ";".join(
            f"{point['name']}@{point['point_deg'][0]},{point['point_deg'][1]}" for point in points)]
    config = dict(row["scorecard"])
    if args.observation_config:
        config.update(json.loads(args.observation_config.read_text(encoding="utf-8")))
    # Region belongs to the country row; observation file metadata does not move it.
    config["region"] = row["scorecard"]["region"]
    score = ["score", str(root / "forecast"), "--config", str(root / "observation-scorecard.config.json")]
    extra = {}
    if "geometry" in row:
        from .lbc_producer import source_endpoint_commands
        endpoints = source_endpoint_commands(source=source["id"], cycle=args.cycle, hours=args.hours,
            out_dir=root / "boundary-source", staged=args.staged, member=getattr(args, "member", None),
            table=getattr(args, "source_table", None), lead_hours=lead,
            boundary_interval_hours=getattr(args, "boundary_interval_hours", None))
        parent = root / "parent"
        endpoints[0]["argv"][endpoints[0]["argv"].index("--out-dir") + 1] = str(parent / "init")
        initialization[initialization.index("--grid") + 1] = str(parent / "mesh.grid.nc")
        initialization[initialization.index("--static") + 1] = str(parent / "mesh.static.nc")
        initialization[initialization.index("--out-dir") + 1] = str(parent / "init")
        forecast += ["--lbc-dir", str(root / "lbc")]
        cut = ["cull", "--parent-grid", str(parent / "mesh.grid.nc"), "--parent-static", str(parent / "mesh.static.nc"),
               "--parent-init", str(parent / "init/init.nc"), "--region", str(root / "region.json"),
               "--out-dir", str(root), "--name", "mesh", "--clobber"]
        fg, switches = source["first_guess"], source_table["run_switches"]
        boundary = ["lbc", "--grid", str(root / "mesh.init.nc"), "--met-dir", str(root / "boundary-met"),
                    "--out-dir", str(root / "lbc"), "--start-time", start, "--stop-time", endpoints[-1]["valid_time"],
                    "--nfglevels", str(fg["nfglevels"]), "--use-spechumd", fg["use_spechumd"],
                    "--extrap-airtemp", switches["extrap_airtemp"], "--theta-adv-order", str(switches["theta_adv_order"]),
                    "--coef-3rd-order", str(switches["coef_3rd_order"]), "--oned-underflow", switches["oned_underflow"]]
        forecast[forecast.index("--init") + 1] = str(root / "mesh.init.nc")
        init = root / "mesh.init.nc"
        extra = {"cull": cut, "lbc": boundary}
        producer = producer_suite(named)
        if producer:
            # A producer suite (hrrrv5): the cut carries grid and static only, the start state is the producer's
            # case 7 on it (its own static from the geography, the decoded start intermediate) and the boundaries
            # are its case 9 on that init (_prepare_region).
            init = root / "rrfs-init" / "init.nc"
            cut[cut.index("--parent-init"):cut.index("--parent-init") + 2] = []
            boundary[boundary.index("--grid") + 1] = str(init)
            boundary += ["--physics-suite", producer]
            forecast[forecast.index("--init") + 1] = str(init)
            producer_init = ["init", "--met", str(root / "boundary-met" / ("MET:" + start[:13])),
                             "--static", str(root / "mesh.static.nc"), "--grid", str(root / "mesh.grid.nc"),
                             "--out", str(init), "--start-time", start,
                             "--nfglevels", str(fg["nfglevels"]), "--nfgsoillevels", str(fg["nfgsoillevels"]),
                             "--physics-suite", producer]
            if getattr(args, "deep_soil_temperature", None):
                producer_init += ["--deep-soil-temperature", args.deep_soil_temperature]
            extra["producer_init"] = producer_init
    # The case named, not read from the forecast's receipt afterwards: the first look draws into it while the
    # forecast runs, before that receipt exists (the same name the receipt gives, the mesh row and the cycle).
    from .products_door import case_token
    product_argv += ["--case", case_token(None, Path(forecast[forecast.index("--grid") + 1]), start, mesh["id"])]
    return {"schema": "gpuwm-hex.country-plan/v1", "row": row, "cycle": args.cycle, "lead_hours": lead,
            "start_time": start,
            "paths": {"grid": str(grid), "static": str(static), "init": str(init), "spec": str(spec)},
            "commands": {"init": initialization, "forecast": forecast, "products": product_argv, "score": score, **extra},
            **({"boundary_endpoints": endpoints} if "geometry" in row else {}),
            "scorecard": config}


def _prepare(args, value, status):
    from .cli import main
    if "geometry" in value["row"]:
        return _prepare_region(args, value, status)
    root = args.work_dir.resolve()
    rows = root / "mesh-rows.json"
    from . import mesh_rows
    from .engines import resolve, ENGINES
    def resolve_engine(name):
        return resolve(next(spec for spec in ENGINES if spec.name == name))
    from .prep_store import run_or_fetch
    paths = {k: Path(v) for k, v in value["paths"].items()}
    # Mesh and static depend on the mesh request alone, never the cycle:
    # fetched from the per-mesh store when it holds them (woof.hex.prep_store).
    if not paths["grid"].exists():
        run_or_fetch([str(resolve_engine("rw_mpas_mesh")), "--spec", str(paths["spec"]),
                      "--out", str(paths["grid"]), "--receipt", str(root / "mesh-receipt.json")])
    if not paths["static"].exists():
        def build_static(argv):
            from .engines import refusal_text
            log = root / "static.log"
            result = subprocess.run(argv, capture_output=True, text=True)
            log.write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
            if result.returncode != 0:
                raise MpasPortError(f"rw_mpas_static exited {result.returncode} (log {log}):\n"
                                    f"{refusal_text(result.stderr)}")
            return result
        # The root woof fetch-geog stages (static_geog_argv), as the global door
        # does; with none found the static helper owns missing-archive diagnostics.
        run_or_fetch([str(resolve_engine("rw_mpas_static")), "--grid", str(paths["grid"]),
                      *static_geog_argv(args.geog),
                      "--out", str(paths["static"]),
                      "--receipt", str(root / "static-receipt.json")], runner=build_static)
    mesh = value["row"]["mesh"]
    # The RRFS-shape suite's GSL drag reads the mesh's orographic statistics (woof.hex.ugwp_oro).
    from .ugwp_oro import prepare_for_forecast
    prepare_for_forecast(value["commands"]["forecast"], paths["grid"], args.geog)
    registered = mesh_rows.describe_generated(name=mesh["id"], grid=paths["grid"], static=paths["static"],
        spec_path=paths["spec"], generator_receipt=root / "mesh-receipt.json", static_receipt=root / "static-receipt.json",
        point_deg=tuple(mesh["point_deg"]), fine_dx_m=mesh["fine_dx_m"], core_radius_km=mesh["core_radius_km"],
        background_km=mesh["background_km"], dt_seconds=mesh["dt_seconds"], n_levels=mesh["n_levels"],
        admission={"scope": "country row; geometry admissions remain at forecast binding"})
    mesh_rows.write_rows(rows, [registered])
    if paths["init"].is_file():
        status.note(f"init already present: {paths['init']}")
        return 0
    status.note("woof hex " + " ".join(value["commands"]["init"]))
    return main(value["commands"]["init"])


def first_look(value, root):
    """The region's maps hour by hour while its forecast runs (first_look.py), into the full pass's tree and case.

    THE BREAKAGE THIS PREVENTS (WOOF hex region runs on woof hex 0.4.4, 2026-10-24): the country door had no first
    look, so a region run showed no map until its forecast and its full products pass had both ended, while global
    runs streamed every hour. The draws take the products pass's own view and label flags, without point soundings
    (the full pass draws those at the row's points)."""
    from .first_look import FirstLook
    forecast, products = value["commands"]["forecast"], value["commands"]["products"]
    flags, skip = [], {"--out", "--case", "--points"}
    rest = products[2:]
    i = 0
    while i < len(rest):
        if rest[i] in skip:
            i += 2
            continue
        flags.append(str(rest[i]))
        i += 1
    return FirstLook(root / "forecast", grid=forecast[forecast.index("--grid") + 1],
                     init=forecast[forecast.index("--init") + 1], start_time=forecast[forecast.index("--start-time") + 1],
                     out=products[products.index("--out") + 1], case=products[products.index("--case") + 1],
                     extra=[*flags, "--no-points"])


def _run_stage(name, args, value, status, stages=(), state=None):
    state = {} if state is None else state
    if name == "prepare":
        return {"returncode": _prepare(args, value, status)}
    root = args.work_dir.resolve()
    if not (root / "mesh-rows.json").is_file():
        raise MpasPortError("country stages need 'country --stage prepare' first; the mesh has no registered byte identity")
    config = root / "observation-scorecard.config.json"
    if name == "score":
        from .observation_scorecard import completed_scorecard, join_automatic_scorer
        joined = (completed_scorecard(root / "forecast", config)
                  or join_automatic_scorer(root / "forecast", config))
        if joined is not None:
            status.note(joined["note"])
            print(json.dumps(joined), flush=True)
            return {**joined, "returncode": 0}
    command = list(value["commands"][name])
    if name == "forecast" and "producer_init" in value["commands"] and "--orographic-statistics" not in command:
        # The producer's GSL drag statistics win when it made them (the RRFS default, geography with topo_ugwp);
        # otherwise the forecast reads hex's own beside the cut's grid, built at the prepare stage.
        made = Path(value["paths"]["init"]).with_name("ugwp_oro_data.nc")
        if made.is_file():
            command += ["--orographic-statistics", str(made)]
    if name == "products" and state.get("live_receipt") is not None:
        command += ["--live-frames", str(state["live_receipt"])]  # the hours drawn live keep their maps
    status.note("woof hex " + " ".join(command))
    look = first_look(value, root) if name == "forecast" and "products" in stages else None
    if look is not None:
        look.start()
    # The forecast runs in its own process, so its device memory ends with it (cli.DEVICE_COMMANDS).
    from . import cli
    try:
        rc = cli.run_stage(command)
    finally:
        if look is not None:
            state["first_look"] = look.finish()
            state["live_receipt"] = look.receipt()
            status.note(f"first look: {state['first_look'].get('status')}, "
                        f"{state['first_look']['live']['drawn']} frames drawn while the forecast ran")
    if name == "score" and rc == 3:
        completed = completed_scorecard(root / "forecast", config)
        if completed is not None:
            status.note(completed["note"])
            return {**completed, "scorer_returncode": rc}
    return {"returncode": rc, **({"first_look": state["first_look"]} if look is not None else {})}


def run_country(args):
    from .run_status import StatusWriter, utc_now
    value = plan(args)
    if args.stage == "plan":
        print(json.dumps(value, indent=2))
        return 0
    stages = [name for name in STAGES if (args.stage == "all" or args.stage == name)
              and name not in args.skip]
    root = args.work_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "country-plan.json").write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    config = root / "observation-scorecard.config.json"
    config.write_text(json.dumps(value["scorecard"], indent=2) + "\n", encoding="utf-8")
    status = StatusWriter(root, "country", stages=tuple(stages), row=args.row)
    if status.log_path is not None:
        print(f"log: {status.log_path}", flush=True)
    receipt = {"schema": "gpuwm-hex.country-run/v1", "plan": str(root / "country-plan.json"),
               "row": args.row, "cycle": args.cycle, "stages": {}}
    rows_env = os.pathsep.join(filter(None, [str(root / "mesh-rows.json"), os.environ.get("WOOF_HEX_MESH_ROWS")]))
    env = {"WOOF_HEX_MESH_ROWS": rows_env, "WOOF_HEX_OBSERVATION_CONFIG": str(config)}
    if "score" in set(args.skip or []):
        # --skip score: the forecast door queues no automatic scorer either (observation_scorecard.SCORING_ENV).
        env["WOOF_HEX_OBSERVATION_SCORING"] = "off"
    saved = {key: os.environ.get(key) for key in env}
    os.environ.update(env)
    state: dict = {}
    try:
        for name in stages:
            started = time.monotonic()
            status.stage(name)
            timing = receipt["stages"][name] = {"started_utc": utc_now()}
            try:
                result = _run_stage(name, args, value, status, stages, state)
            finally:
                timing.update(seconds=time.monotonic() - started, finished_utc=utc_now())
            timing.update(result)
            if result["returncode"] != 0:
                status.failed(f"the {name} stage exited {result['returncode']}")
                return result["returncode"]
            if name == "forecast":
                # The forecast is the last GPU stage (products and score are CPU work): a card queue gets its
                # card back now instead of after them (woof.hex.card_lease).
                from .card_lease import gpu_work_done
                timing["card_given_back"] = gpu_work_done()
        status.done(f"{args.row} {args.cycle}: {', '.join(stages)}")
        return 0
    except MpasPortError as error:
        status.refused(str(error))
        receipt["refused"] = str(error)
        raise
    except Exception as error:
        status.failed(f"{type(error).__name__}: {error}")
        receipt["failed"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        for key, previous in saved.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous
        (root / "country-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


def add_country_parser(commands):
    parser = commands.add_parser("country", help="compose operational doors from a country row")
    from woof.output_formats import add_arguments
    add_arguments(parser)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--row")
    selection.add_argument("--region", help="registered limited-area row")
    selection.add_argument("--point", help="LAT,LON for an arbitrary limited-area cap")
    selection.add_argument("--polygon", type=Path, help="Shape polygon JSON, vertices in latitude,longitude order")
    selection.add_argument("--ellipse", metavar="LAT,LON,SEMI_MAJOR_KM,SEMI_MINOR_KM[,AZIMUTH_DEG]",
                           help="an elliptical region (azimuth of the major axis from north)")
    selection.add_argument("--shape", type=Path, metavar="FILE",
                           help="a drawn region: a Shape row (polygon with [lat, lon] vertices, lat_lon_box or cap) "
                                "or GeoJSON, read as `regional --shape` reads it")
    parser.add_argument("--radius-km", type=float)
    parser.add_argument("--fine-dx-m", type=float, default=None,
                        help="with --point or --polygon: the region's spacing, a rung of the halving ladder "
                             "under --background-km (default the table's 30 km)")
    parser.add_argument("--background-km", type=float, default=None,
                        help="with --fine-dx-m: the ladder's coarsest spacing (default the table's 120 km)")
    parser.add_argument("--source", help="init-source row override, used for init and every boundary endpoint")
    parser.add_argument("--source-table", type=Path)
    parser.add_argument("--member", type=int)
    parser.add_argument("--endpoint-workers", type=int, default=2)
    parser.add_argument("--table", type=Path, default=TABLE)
    parser.add_argument("--cycle", required=True, help="YYYY-MM-DDTHH, the init cycle")
    parser.add_argument("--lead-hours", type=int, default=0, metavar="H",
                        help="start from this forecast hour of --cycle (a forecast source's own cadence, e.g. GFS "
                             "2026-10-23T18 --lead-hours 348): the init is that hour and the boundaries follow it")
    parser.add_argument("--boundary-interval-hours", type=int, default=None, metavar="H",
                        help="boundary step of a table source on a geometry row. Default: the finest step the "
                             "source posts over the run (GFS 1 h to 120 h, IFS and GEFS 3 h)")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("plan", "all", *STAGES), default="all",
                        help="run all stages (default), one named stage, or print the plan")
    parser.add_argument("--skip", action="append", default=[], choices=STAGES, metavar="STAGE",
                        help="leave a stage out; repeat to leave out several")
    parser.add_argument("--grid", type=Path)
    parser.add_argument("--static", type=Path)
    parser.add_argument("--init", type=Path)
    parser.add_argument("--geog", type=Path, default=None,
                        help="WPS_GEOG root; defaults to rw_mpas_static's lookup "
                             "(the root `woof fetch-geog` stages)")
    parser.add_argument("--staged", type=Path, action="append", default=[])
    parser.add_argument("--gpuwm-checkout", type=Path,
                        help="woof tree whose seam files the forecast runs (default: the installed engine)")
    parser.add_argument("--observation-config", type=Path)
    parser.add_argument("--hours", type=float, default=6.0)
    parser.add_argument("--history-every-minutes", type=int, default=60)
    from .products_door import add_label_arguments

    add_label_arguments(parser)
    from .regional_door import add_samehost_arguments
    add_samehost_arguments(parser)
    parser.add_argument("--deep-soil-temperature", choices=("era5", "islscp-i"), default=None,
                        help="a producer suite (hrrrv5): the deep-soil temperature its static is built with "
                             "(default era5; islscp-i is NOAA's soiltemp_1deg, licence unclear)")
    parser.add_argument("--forecast-arg", action="append", default=[], metavar="ARG",
                        help="a flag for the forecast door, e.g. --forecast-arg=--physics-suite=rrfs_v2_dev "
                             "(`woof hex physics list`); repeat for several")
    parser.set_defaults(handler=run_country)
